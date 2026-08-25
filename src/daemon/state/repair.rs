//! Repairing a shard whose bytes turned out to be wrong.

use super::SharedState;
use crate::types::ShardId;

impl SharedState {
    /// This shard's bytes are wrong — arrange for a fresh, verified copy.
    ///
    /// **The single way to ask for a corrupt shard to be replaced.** Three
    /// places can catch a bad shard, and all three must do the same thing: the
    /// P2P accept path, the background verification sweep, and the auto-manage
    /// rescan. Before this existed each of them removed the file and stopped
    /// there — "removed" was implemented three times and "and get a good one"
    /// nowhere. Repair happened only as a side effect of auto-manage noticing
    /// the shard had gone missing, so a node with auto-manage switched off kept
    /// a permanently incomplete model, and every rescan re-hashed the same bad
    /// file to reach the same conclusion.
    ///
    /// Quarantining the file is the CALLER's job (`verify_shard` already does
    /// it, and only the caller knows whether the bytes are wrong or merely
    /// unverifiable). This schedules the replacement.
    ///
    /// Deliberately does NOT set `shard_p2p_failed`: that forces the
    /// HuggingFace path, and a repair should be free to fetch from a peer.
    /// Having DETECTED the corruption means we hold the real hash, so a peer
    /// copy is checked against it — and if that one is bad too, the accept path
    /// quarantines it and docks the sender, which is what we want to happen.
    pub fn mark_shard_for_repair(&self, shard_id: &ShardId) {
        // A shard the user deleted is an instruction, not a gap — do not
        // resurrect it under the guise of a repair.
        if self.shard_removed_by_user(shard_id) {
            return;
        }
        self.models.shards_needing_repair.insert(shard_id.clone());
        // Clear the stale per-shard progress entry, or `is_shard_in_progress`
        // reports a download that is not running and the refetch is skipped
        // forever. (The accept path returns on verify failure before marking
        // the shard Complete, so its entry is left mid-Downloading.)
        //
        // **Only call this when no download is actually running for the shard.**
        // Nothing here can tell a stale entry from a live one — both read as
        // `Downloading` — and clearing a live one lets a second task append to
        // the same `.tmp` concurrently, producing the right size and wrong
        // bytes, which is the failure this whole path exists to prevent. True
        // of all three current callers: the transfer has finished (accept
        // path), or `is_shard_in_progress` was already checked (rescan), or no
        // download exists (background sweep).
        if let Some(mut entry) = self.models.acquisition_progress.get_mut(&shard_id.model_id) {
            entry.shard_progress.remove(&shard_id.index);
        }
        tracing::info!(
            model = %shard_id.model_id,
            shard = shard_id.index,
            "Shard failed verification — queued for replacement"
        );
        self.models.auto_manage_notify.notify_one();
    }

    /// Can a fresh copy of this model's shards actually be fetched from the
    /// model's ORIGIN right now?
    ///
    /// **"Will it actually happen", not "does an origin exist".** Every caller
    /// that is about to discard local bytes in favour of an origin copy must ask
    /// this first — **never throw away data you cannot replace.** Keeping it in
    /// one place is what stops the two conditions drifting apart; they were
    /// found one at a time, each after reasoning that the other was the only one.
    ///
    /// Offline mode counts because `trigger_download` skips the HuggingFace
    /// branch entirely when it is set, by design. Auto-manage being switched off
    /// deliberately does NOT count: `complete_pending_shard_fetches` runs
    /// outside that gate, because it means "do not decide what to fetch for me",
    /// not "abandon a shard I already asked for".
    pub fn can_fetch_shard_from_origin(&self, model_id: &crate::types::ModelId) -> bool {
        self.models.hf_sources.contains_key(model_id)
            && !self
                .credits
                .offline_mode
                .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Record a shard hash taken from the model's ORIGIN, in memory and on disk.
    ///
    /// **Only the origin settles a hash.** A peer that self-certified a corrupt
    /// shard gossips the wrong hash, and manifest registration is
    /// last-writer-wins — so without this the wrong hash displaces the right one
    /// and the re-check quarantines our GOOD copy, refetches, and judges the
    /// replacement against the same wrong reference, forever (gotcha #384).
    ///
    /// Persisted because the fact outlives the process: relearning it would mean
    /// re-downloading from the origin, and until then gossip wins again.
    pub fn record_origin_verified_hash(
        &self,
        shard_id: crate::types::ShardId,
        hash: crate::types::Blake3Hash,
    ) {
        if hash == [0u8; 32] {
            return;
        }
        let key = match serde_json::to_string(&shard_id) {
            Ok(k) => k,
            Err(_) => return,
        };
        if let Err(e) =
            self.db
                .insert_raw(crate::model::registry::ORIGIN_VERIFIED_TREE, &key, &hash)
        {
            tracing::warn!(
                model = %shard_id.model_id,
                shard = shard_id.index,
                error = %e,
                "Could not persist an origin-verified shard hash — a peer's \
                 claim could displace it after a restart"
            );
        }
        self.model_registry
            .record_origin_verified_hash(shard_id, hash);
    }

    /// Drop the repair request once the shard is back. Called when a download
    /// completes so the set stays bounded and nothing re-fetches a good shard.
    pub fn clear_shard_repair(&self, shard_id: &ShardId) {
        self.models.shards_needing_repair.remove(shard_id);
    }
}

#[cfg(test)]
mod tests {
    use crate::types::{ModelId, ShardId};

    fn test_state() -> std::sync::Arc<crate::daemon::SharedState> {
        use crate::identity::Identity;
        use crate::inference::executor::ModelExecutor;
        use crate::storage::db::Database;
        use tokio::sync::Mutex;

        let temp = tempfile::tempdir().unwrap();
        let db = Database::open(temp.path()).unwrap();
        let executor = std::sync::Arc::new(Mutex::new(ModelExecutor::new()));
        let (state, _, _) = crate::daemon::SharedState::new(
            crate::config::Config::default(),
            Identity::generate(),
            db,
            executor,
            None,
        );
        state
    }

    fn sid() -> ShardId {
        ShardId {
            model_id: ModelId("m".into()),
            index: 3,
        }
    }

    /// Removing the bad bytes is only half of it. Before this, all three
    /// detection sites quarantined and stopped, so a node that was not
    /// auto-managing kept a permanently incomplete model and every rescan
    /// re-hashed the same bad file to reach the same conclusion.
    #[test]
    fn a_corrupt_shard_is_queued_for_replacement() {
        let state = test_state();
        let s = sid();
        state.mark_shard_for_repair(&s);
        assert!(
            state.models.shards_needing_repair.contains(&s),
            "a shard found corrupt must be queued for a fresh copy"
        );

        // Deliberately NOT steered to the origin: detecting the corruption
        // means we hold the real hash, so a peer copy gets checked against it.
        assert!(
            !state.models.shard_p2p_failed.contains(&s),
            "a repair must stay free to fetch from a peer"
        );

        state.clear_shard_repair(&s);
        assert!(!state.models.shards_needing_repair.contains(&s));
    }

    /// A shard the user deleted is an instruction, not a gap.
    #[test]
    fn a_user_deleted_shard_is_not_resurrected_as_a_repair() {
        let state = test_state();
        let s = sid();
        state.mark_shard_removed_by_user(&s);
        state.mark_shard_for_repair(&s);
        assert!(
            !state.models.shards_needing_repair.contains(&s),
            "repair must not undo a deliberate deletion"
        );
    }

    /// "Will the fetch actually happen", not "does an origin exist" — the
    /// question every caller must ask before discarding local bytes.
    #[test]
    fn an_offline_node_reports_no_origin_to_fetch_from() {
        use std::sync::atomic::Ordering::Relaxed;
        let state = test_state();
        let mid = ModelId("m".into());
        assert!(
            !state.can_fetch_shard_from_origin(&mid),
            "no recorded origin means no origin fetch"
        );

        state.models.hf_sources.insert(
            mid.clone(),
            crate::daemon::state::hf::HfSource {
                repo_id: "r/x".into(),
                filename: "x.gguf".into(),
                mmproj_filename: None,
            },
        );
        assert!(state.can_fetch_shard_from_origin(&mid));

        // Offline mode makes `trigger_download` skip the HuggingFace branch, so
        // discarding local bytes in favour of it would replace them with
        // nothing.
        state.credits.offline_mode.store(true, Relaxed);
        assert!(
            !state.can_fetch_shard_from_origin(&mid),
            "offline mode must not be read as a reachable origin"
        );
    }
}
