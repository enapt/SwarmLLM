//! Repairing a shard whose bytes turned out to be wrong.

use super::SharedState;
use crate::types::ShardId;

/// What an origin download's bytes are measured against, and the verdict
/// ([`SharedState::accept_origin_part`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum OriginPartVerdict {
    /// No connected holder has checked its copy against the upload.
    NothingToCompare,
    /// A holder that checked its own copy holds these bytes.
    Corroborated,
    /// Checked holders hold other bytes, but the previous download of this
    /// part brought exactly these.
    Repeated,
    /// Checked holders hold other bytes, and no earlier download agrees.
    Uncorroborated,
}

/// Judge a part just downloaded from the origin: `hash` its bytes, `checked`
/// the tags connected holders that checked their copies announce, `previous`
/// the last uncorroborated download of this part.
pub(crate) fn judge_origin_part(
    hash: &crate::types::Blake3Hash,
    checked: &[u64],
    previous: Option<&crate::types::Blake3Hash>,
) -> OriginPartVerdict {
    if checked.is_empty() {
        OriginPartVerdict::NothingToCompare
    } else if checked.contains(&swarmllm_types::build_tag_from_hash(hash)) {
        OriginPartVerdict::Corroborated
    } else if previous == Some(hash) {
        OriginPartVerdict::Repeated
    } else {
        OriginPartVerdict::Uncorroborated
    }
}

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
        // These bytes were just written from the origin: whatever this node
        // disagreed with the swarm about, it no longer holds.
        self.clear_shard_dispute(&shard_id);
        self.model_registry
            .record_origin_verified_hash(shard_id, hash);
    }

    /// A part has just been written from the model's ORIGIN: keep it — record
    /// its hash as origin-derived and put it in the manifest — or, when
    /// nothing corroborates its bytes, delete it so it is fetched again.
    /// Returns whether it was kept; a part that was not kept is a failed
    /// download to the caller.
    ///
    /// **The one place a download from the origin becomes a part this node
    /// vouches for**, called by BOTH origin-download paths — the auto-manage
    /// downloader and the admin "download this part" handler (the second once
    /// recorded no provenance at all, #384).
    ///
    /// Before this, a download from HuggingFace was believed outright: hashed
    /// after the fact and recorded as the origin's own bytes. A node's parts
    /// came out of two downloads in a row as two DIFFERENT wrong builds, and it
    /// vouched for each; its heal then exempted the part for the rest of the
    /// run as "settled by the upload" (FUTURE_WORK #217, 2026-10-04 09:01-09:05
    /// UTC). Large HuggingFace downloads have been reported to arrive the right
    /// size with different content on each attempt
    /// (huggingface/huggingface_hub#3643: four attempts, four hashes); a disk
    /// losing writes looks the same. So an origin download is kept when
    /// something corroborates it ([`judge_origin_part`]): a connected holder
    /// that checked its own copy against the upload holds these bytes, or the
    /// previous download of this part brought the same ones — two corrupted
    /// transfers do not agree on 500 MB. When no holder has checked a copy
    /// there is nothing to compare with, and it is kept, as before.
    pub fn accept_origin_part(
        &self,
        shard_id: &ShardId,
        part: crate::model::huggingface::DownloadedPart,
    ) -> bool {
        let checked = self
            .model_registry
            .checked_holder_tags(shard_id, |n| self.peer_registry.contains_key(n));
        let previous = self
            .models
            .uncorroborated_origin_parts
            .get(shard_id)
            .map(|h| *h);
        let verdict = judge_origin_part(&part.hash, &checked, previous.as_ref());
        match verdict {
            OriginPartVerdict::Uncorroborated => {
                self.models
                    .uncorroborated_origin_parts
                    .insert(shard_id.clone(), part.hash);
                let _ = std::fs::remove_file(&part.path);
                let tag = |h: &crate::types::Blake3Hash| {
                    format!("{:016x}", swarmllm_types::build_tag_from_hash(h))
                };
                tracing::warn!(
                    model = %shard_id.model_id,
                    shard = shard_id.index,
                    downloaded_build = %tag(&part.hash),
                    previous_download = ?previous.as_ref().map(tag),
                    checked_holders = ?checked.iter().map(|t| format!("{t:016x}")).collect::<Vec<_>>(),
                    "DIAG: a part downloaded from HuggingFace is not the bytes the computers that checked theirs hold, and no earlier download agrees — discarded, fetching it again"
                );
                if previous.is_some() {
                    // Two downloads of one part, two different results: the
                    // transfer or this computer's disk is changing data.
                    self.emit_activity(
                        crate::daemon::state::ActivityEvent::new(
                            "download",
                            "origin_part_unstable",
                            format!(
                                "{}: part {} came out different on two downloads from HuggingFace —                                  discarded and fetched again. If this keeps happening, this computer's                                  disk or connection may be damaging data.",
                                self.model_registry.display_name(&shard_id.model_id),
                                crate::types::ShardId::display_index_short(shard_id.index),
                            ),
                        )
                        .with_model(shard_id.model_id.0.clone())
                        .with_detail_num(shard_id.index as i64)
                        .with_toast("warning", 8000),
                    );
                }
                false
            }
            OriginPartVerdict::Repeated => {
                // Kept with the entry: a part held under this hash is the twice-
                // downloaded copy, which the heal does not fetch again
                // ([`Self::origin_part_downloaded_twice`]).
                tracing::info!(
                    model = %shard_id.model_id,
                    shard = shard_id.index,
                    checked_holders = ?checked.iter().map(|t| format!("{t:016x}")).collect::<Vec<_>>(),
                    "DIAG: two downloads of this part from HuggingFace agree, though the computers that checked theirs hold other bytes — keeping it"
                );
                self.record_origin_part_hash(shard_id, part.hash);
                true
            }
            OriginPartVerdict::Corroborated | OriginPartVerdict::NothingToCompare => {
                tracing::info!(
                    model = %shard_id.model_id,
                    shard = shard_id.index,
                    ?verdict,
                    checked_holders = checked.len(),
                    "DIAG: a part downloaded from HuggingFace is kept"
                );
                self.models.uncorroborated_origin_parts.remove(shard_id);
                self.record_origin_part_hash(shard_id, part.hash);
                true
            }
        }
    }

    /// Is the part held here the copy two downloads from the origin agreed on
    /// ([`OriginPartVerdict::Repeated`])? The upload's own bytes have settled
    /// it, so the heal does not fetch it again when checked holders disagree.
    pub fn origin_part_downloaded_twice(&self, shard_id: &ShardId) -> bool {
        let Some(twice) = self
            .models
            .uncorroborated_origin_parts
            .get(shard_id)
            .map(|h| *h)
        else {
            return false;
        };
        self.model_registry.origin_verified_hash(shard_id) == Some(twice)
    }

    /// Record a kept origin part's hash as origin-derived, and put it in the
    /// manifest — in memory, on disk and in the database, so startup
    /// verification passes after a restart and a peer's claim cannot displace
    /// it (#384).
    fn record_origin_part_hash(&self, shard_id: &ShardId, hash: crate::types::Blake3Hash) {
        self.record_origin_verified_hash(shard_id.clone(), hash);
        let Some(mut manifest) = self.model_registry.get_manifest(&shard_id.model_id) else {
            return;
        };
        match manifest
            .shards
            .iter_mut()
            .find(|s| s.index == shard_id.index)
        {
            Some(si) if si.hash == hash => return,
            Some(si) => si.hash = hash,
            None => return,
        }
        manifest.manifest_hash = crate::model::manifest::ModelManifestExt::compute_hash(&manifest);
        let dir = crate::model::shard::model_dir(&self.config.node.data_dir, &shard_id.model_id.0);
        if let Err(e) = crate::model::manifest::ModelManifestExt::save_to_dir(&manifest, &dir) {
            tracing::warn!(
                model = %shard_id.model_id,
                error = %e,
                "Could not save the manifest after a part arrived from the origin — its hash is in memory only"
            );
        }
        self.model_registry.register_manifest(manifest.clone());
        if let Err(e) = self.model_registry.persist_manifest(&self.db, &manifest) {
            tracing::warn!(
                model = %shard_id.model_id,
                error = %e,
                "Could not store the manifest after a part arrived from the origin"
            );
        }
    }

    /// Drop the repair request once the shard is back. Called when a download
    /// completes so the set stays bounded and nothing re-fetches a good shard.
    pub fn clear_shard_repair(&self, shard_id: &ShardId) {
        self.models.shards_needing_repair.remove(shard_id);
    }

    /// Our bytes disagree with the hash the swarm reports, and we are keeping
    /// them — record it where something other than the log can see.
    ///
    /// **The single way a dispute is written down.** This is deliberately NOT
    /// `mark_shard_for_repair`: that set is drained by
    /// `complete_pending_shard_fetches`, whose first action is to treat a shard
    /// whose file is on disk as already repaired and clear the mark — and a
    /// disputed shard is on disk by definition, so marking it fetches nothing
    /// and only churns the set. The first cut of the 2026-09-13 fix called it
    /// anyway and claimed a settlement that could not happen.
    ///
    /// Nothing here resolves the disagreement; that is open work, and its
    /// stated precondition is knowing how often this fires in the field
    /// (`docs/FUTURE_WORK.md` § "A disputed shard is kept but the disagreement
    /// is never settled"). Until then the job is to make it countable and
    /// visible rather than to guess at a policy.
    ///
    /// `verdict` is the check's own error: when it hashed the bytes
    /// (`ShardIntegrity`), that hash is kept, and it is what this node
    /// ANNOUNCES for the part from now on (`ModelRegistry::announced_build_tag`)
    /// — the manifest carries the hash the swarm reported, which our bytes are
    /// not, and announcing it told peers we held their bytes.
    pub fn note_shard_disputed(&self, shard_id: &ShardId, verdict: &crate::error::SwarmError) {
        let bytes = crate::model::shard::bytes_hash_in_verdict(verdict);
        if self
            .model_registry
            .note_bytes_disputed(shard_id.clone(), bytes)
        {
            tracing::info!(
                model = %shard_id.model_id,
                shard = shard_id.index,
                bytes_build = %bytes.map_or_else(
                    || "unhashed".to_string(),
                    |h| format!("{:016x}", swarmllm_types::build_tag_from_hash(&h))
                ),
                "This node is keeping bytes the swarm disagrees with — the \
                 claim has no origin backing, so it is not evidence enough to \
                 destroy a copy that may be the last one. It announces the \
                 part as the bytes it is, not as the build it was told of"
            );
        }
    }

    /// The disagreement is over: this shard's bytes verified against the hash
    /// we now hold for it, or the file is gone.
    ///
    /// Called on **every** successful verification and on every quarantine, not
    /// only where a dispute is known to exist — `DashSet::remove` on an absent
    /// key is free, and a clear that has to be predicted is a clear that gets
    /// forgotten. A quarantined shard is not disputed but repaired: the bytes
    /// were destroyed against origin-backed evidence and a replacement is
    /// queued.
    pub fn clear_shard_dispute(&self, shard_id: &ShardId) {
        self.model_registry.clear_bytes_dispute(shard_id);
    }

    /// Is this shard one we hold, and disagree with the swarm about?
    pub fn shard_is_disputed(&self, shard_id: &ShardId) -> bool {
        self.model_registry.bytes_dispute_recorded(shard_id)
    }

    /// The shards currently in dispute, dropping any whose file has since gone.
    ///
    /// **Self-evicting on read, deliberately, rather than cleared by every path
    /// that can delete a shard.** A dispute ends in one of three ways: a later
    /// check passes, the bytes are quarantined, or the file simply stops being
    /// here — the user deletes the part, `delete_model` takes the lot, or
    /// auto-manage prunes it. The first two clear the entry where they happen;
    /// the third is three call sites today and every future one would have to
    /// remember. A stale entry is not harmless either: this set exists to
    /// MEASURE how often disputes occur, and one naming a shard that no longer
    /// exists corrupts the figure the decision in `docs/FUTURE_WORK.md` turns
    /// on, while the diagnostics report prints it as a real one.
    ///
    /// Same shape as `shard_in_backoff`, which self-evicts on read for the same
    /// reason: the map stays bounded with no dedicated sweep and no obligation
    /// on code that has not been written yet.
    pub fn disputed_shards_now(&self) -> Vec<ShardId> {
        let store = self.shard_store();
        let mut gone: Vec<ShardId> = Vec::new();
        let mut live: Vec<ShardId> = Vec::new();
        for sid in self.model_registry.bytes_disputed_shards() {
            if store.shard_path(&sid.model_id, sid.index).exists() {
                live.push(sid);
            } else {
                gone.push(sid);
            }
        }
        for sid in gone {
            self.model_registry.clear_bytes_dispute(&sid);
        }
        live
    }

    /// How many shards this node holds and disagrees with the swarm about.
    ///
    /// The figure the diagnostics report prints, and the one that decides
    /// whether the settlement designs in `docs/FUTURE_WORK.md` are worth
    /// building. Zero is a real answer and worth reporting as one.
    pub fn disputed_shard_count(&self) -> usize {
        self.disputed_shards_now().len()
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

    use super::{judge_origin_part, OriginPartVerdict};

    /// What an origin download is measured against: a holder that checked
    /// its copy, then the previous download — and with neither, nothing.
    #[test]
    fn an_origin_download_is_kept_only_when_something_corroborates_it() {
        let (good, bad, worse) = ([1u8; 32], [2u8; 32], [3u8; 32]);
        let good_tag = swarmllm_types::build_tag_from_hash(&good);
        assert_eq!(
            judge_origin_part(&bad, &[], None),
            OriginPartVerdict::NothingToCompare,
            "no checked holder: nothing to compare with, kept as before"
        );
        assert_eq!(
            judge_origin_part(&good, &[good_tag], None),
            OriginPartVerdict::Corroborated
        );
        assert_eq!(
            judge_origin_part(&good, &[0xdead, good_tag], Some(&bad)),
            OriginPartVerdict::Corroborated,
            "one checked holder agreeing is enough, whatever came before"
        );
        assert_eq!(
            judge_origin_part(&bad, &[good_tag], None),
            OriginPartVerdict::Uncorroborated,
            "the first download checked holders disagree with is not believed"
        );
        assert_eq!(
            judge_origin_part(&worse, &[good_tag], Some(&bad)),
            OriginPartVerdict::Uncorroborated,
            "two downloads, two results: still nothing agrees"
        );
        assert_eq!(
            judge_origin_part(&bad, &[good_tag], Some(&bad)),
            OriginPartVerdict::Repeated,
            "two downloads agreeing are the upload's bytes"
        );
    }

    /// A state whose data directory is a temp dir — `accept_origin_part`
    /// writes the manifest beside the parts.
    fn state_in(dir: &std::path::Path) -> std::sync::Arc<crate::daemon::SharedState> {
        let mut config = crate::config::Config::default();
        config.node.data_dir = dir.to_path_buf();
        let db = crate::storage::db::Database::open(dir).unwrap();
        let executor = std::sync::Arc::new(tokio::sync::Mutex::new(
            crate::inference::executor::ModelExecutor::new(),
        ));
        let (state, _, _) = crate::daemon::SharedState::new(
            config,
            crate::identity::Identity::generate(),
            db,
            executor,
            None,
        );
        state
    }

    /// **#217's field shape: two downloads from HuggingFace in a row, two
    /// different wrong builds, each vouched for.** A part checked holders
    /// disagree with is deleted until a second download brings the same bytes;
    /// one they agree with is kept at once and ends the doubt.
    #[test]
    fn a_part_downloaded_from_the_origin_is_vouched_for_only_when_corroborated() {
        let temp = tempfile::tempdir().unwrap();
        let state = state_in(temp.path());
        register_with_hash(&state, [0u8; 32]);
        let (good, bad, worse) = (
            b"the upload's bytes",
            b"other bytes, size",
            b"third version, sz",
        );
        let hash = |b: &[u8]| -> [u8; 32] { *blake3::hash(b).as_bytes() };
        let checker = crate::types::NodeId([7u8; 32]);
        state.model_registry.record_shard_holder_with_build(
            sid(),
            checker.clone(),
            swarmllm_types::build_tag_from_hash(&hash(good)),
            Some(true),
        );
        // Connected now: only connected holders count.
        state.peer_registry.insert(
            checker.clone(),
            crate::types::PeerInfo {
                node_id: checker.clone(),
                addresses: vec![],
                capability: None,
                last_seen: chrono::Utc::now(),
                latency_ms: Some(50),
                trust_score: 0.5,
                peer_id_bytes: None,
                ack_srtt_ms: None,
                active_request_count: 0,
                first_seen: 0,
                verified_transaction_count: 0,
                is_lan_peer: false,
                goodput_bytes_per_sec: None,
                goodput_samples: 0,
            },
        );
        let path = temp.path().join("part.bin");
        let arrive = |bytes: &[u8]| {
            std::fs::write(&path, bytes).unwrap();
            state.accept_origin_part(
                &sid(),
                crate::model::huggingface::DownloadedPart {
                    path: path.clone(),
                    hash: hash(bytes),
                },
            )
        };

        assert!(!arrive(bad), "the first wrong download is not kept");
        assert!(!path.exists(), "and its file is gone, to be fetched again");
        assert_eq!(state.model_registry.origin_verified_hash(&sid()), None);
        assert!(
            !arrive(worse),
            "a second, different result is not kept either"
        );
        assert!(arrive(worse), "the same bytes twice are the upload's");
        assert_eq!(
            state.model_registry.origin_verified_hash(&sid()),
            Some(hash(worse))
        );
        assert!(
            state.origin_part_downloaded_twice(&sid()),
            "the heal knows not to fetch it again"
        );
        assert_eq!(
            state
                .model_registry
                .get_manifest(&ModelId("m".into()))
                .unwrap()
                .shards[3]
                .hash,
            hash(worse),
            "and the manifest names its bytes"
        );

        assert!(
            arrive(good),
            "bytes a checked holder holds are kept at once"
        );
        assert_eq!(
            state.model_registry.origin_verified_hash(&sid()),
            Some(hash(good))
        );
        assert!(
            !state.origin_part_downloaded_twice(&sid()),
            "nothing left in doubt"
        );
    }

    /// With no holder that checked its copy, an origin download is kept as it
    /// always was — the first copy in the swarm has nothing to agree with.
    #[test]
    fn the_first_copy_in_the_swarm_is_kept() {
        let temp = tempfile::tempdir().unwrap();
        let state = state_in(temp.path());
        register_with_hash(&state, [0u8; 32]);
        let path = temp.path().join("part.bin");
        std::fs::write(&path, b"bytes").unwrap();
        let hash = *blake3::hash(b"bytes").as_bytes();
        assert!(state.accept_origin_part(
            &sid(),
            crate::model::huggingface::DownloadedPart {
                path: path.clone(),
                hash,
            },
        ));
        assert!(path.exists());
        assert_eq!(
            state.model_registry.origin_verified_hash(&sid()),
            Some(hash)
        );
    }

    /// What `verify_shard` returns when it hashed bytes that are not the
    /// expected ones.
    fn integrity_failure(bytes: [u8; 32]) -> crate::error::SwarmError {
        crate::error::SwarmError::ShardIntegrity {
            expected: hex::encode([9u8; 32]),
            actual: hex::encode(bytes),
        }
    }

    /// A one-part manifest for model "m" whose part 3 carries `hash`.
    fn register_with_hash(state: &crate::daemon::SharedState, hash: [u8; 32]) {
        state
            .model_registry
            .register_manifest(crate::model::manifest::build_manifest_from_gguf(
                crate::model::manifest::ManifestFromGguf {
                    id: ModelId("m".into()),
                    name: "m".into(),
                    architecture: crate::types::ModelArchitecture::Llama,
                    num_layers: 4,
                    total_size_bytes: 1_000,
                    shard_count: 4,
                    shards: (0..4)
                        .map(|index| crate::types::ShardInfo {
                            index,
                            layer_range: (index, index + 1),
                            size_bytes: 250,
                            hash: if index == 3 { hash } else { [0; 32] },
                            tensors: Vec::new(),
                        })
                        .collect(),
                    publisher: crate::types::NodeId([0; 32]),
                    context_length: None,
                },
            ));
    }

    fn announced_tag(state: &crate::daemon::SharedState, s: &ShardId) -> u64 {
        crate::model::manifest::shard_announce(
            &state.model_registry,
            state.identity.node_id().clone(),
            vec![s.clone()],
            Vec::new(),
        )
        .shard_builds[0]
    }

    /// **A part in dispute is announced as the bytes it is.**
    ///
    /// The field shape (2026-10-03): a node holding another upload's part took
    /// the swarm's hash into its manifest (#382 adopts a contradicting hash for
    /// a held part, to queue the re-check), the re-check kept its bytes, and
    /// its announcements went on carrying the SWARM's build — so every
    /// coordinator routed that part's layers to bytes that were not those.
    /// Announcing from the manifest makes this test fail on the first assert.
    #[test]
    fn a_part_in_dispute_is_announced_as_the_bytes_it_is() {
        let state = test_state();
        let s = sid();
        let swarms = [5u8; 32];
        let ours = [6u8; 32];
        register_with_hash(&state, swarms);
        let swarm_tag = swarmllm_types::build_tag_from_hash(&swarms);
        assert_eq!(
            announced_tag(&state, &s),
            swarm_tag,
            "no dispute: the manifest"
        );

        state.note_shard_disputed(&s, &integrity_failure(ours));
        assert_eq!(
            announced_tag(&state, &s),
            swarmllm_types::build_tag_from_hash(&ours),
            "a disputed part is announced under its own bytes' build"
        );
        assert!(swarmllm_types::build_tags_conflict(
            state.model_registry.expected_build_tag(&s),
            announced_tag(&state, &s)
        ));

        // Bytes found wrong without being hashed (a size mismatch): announced
        // under a tag every expectation conflicts with — never "unknown",
        // which peers read as "do not judge".
        state.note_shard_disputed(
            &s,
            &crate::error::SwarmError::ShardIncomplete {
                expected_bytes: 250,
                actual_bytes: 10,
            },
        );
        let unhashed = announced_tag(&state, &s);
        assert_ne!(unhashed, swarmllm_types::BUILD_TAG_UNKNOWN);
        assert!(swarmllm_types::build_tags_conflict(swarm_tag, unhashed));

        // Bytes written from the origin end the dispute.
        state.record_origin_verified_hash(s.clone(), swarms);
        assert_eq!(announced_tag(&state, &s), swarm_tag);
    }

    /// **A withheld model is not offered**: its parts leave every announcement
    /// while the announcement still says it is complete for the model — which
    /// is what makes the peers that receive it retract the parts they had.
    #[test]
    fn a_withheld_model_is_retracted_not_announced() {
        let state = test_state();
        let s = sid();
        register_with_hash(&state, [5; 32]);
        let me = state.identity.node_id().clone();
        state
            .model_registry
            .record_shard_holder(s.clone(), me.clone());
        let m = ModelId("m".into());
        assert!(state.model_registry.set_model_withheld(&m, true));
        let announce = crate::model::manifest::shard_announce(
            &state.model_registry,
            me.clone(),
            vec![s.clone()],
            vec![m.clone()],
        );
        assert!(announce.shards.is_empty(), "no part of a withheld model");
        assert!(announce.shard_builds.is_empty());
        assert_eq!(announce.complete_for_models, vec![m.clone()]);
        assert!(
            state.model_registry.manifests_to_gossip(&me).is_empty(),
            "nor its manifest, whose hashes are not the swarm's upload"
        );

        assert!(state.model_registry.set_model_withheld(&m, false));
        let announce = crate::model::manifest::shard_announce(
            &state.model_registry,
            me.clone(),
            vec![s.clone()],
            vec![m],
        );
        assert_eq!(announce.shards, vec![s]);
        assert_eq!(state.model_registry.manifests_to_gossip(&me).len(), 1);
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

    /// **A dispute is written down, and deliberately NOT queued for repair.**
    ///
    /// `mark_shard_for_repair` is the trap here. It looks like the answer and
    /// is a no-op for a disputed shard: `complete_pending_shard_fetches` begins
    /// by treating any shard whose file is on disk as already repaired and
    /// clearing the mark — and a disputed shard is on disk by definition, since
    /// keeping it is the whole point. The quarantine path only ever worked
    /// because it deleted the file first. The first cut of the 2026-09-13 fix
    /// called it anyway and its commit message claimed a settlement that could
    /// not happen.
    #[test]
    fn a_disputed_shard_is_recorded_and_not_queued_for_a_repair_that_cannot_run() {
        let state = test_state();
        let s = sid();

        state.note_shard_disputed(&s, &integrity_failure([7; 32]));
        // Membership, not the reported count: `disputed_shards_now` drops a
        // shard whose file is gone, and this test writes no file. The two are
        // different questions — see
        // `a_dispute_about_a_shard_that_is_gone_evicts_itself`.
        assert!(state.shard_is_disputed(&s));
        assert!(
            !state.models.shards_needing_repair.contains(&s),
            "the repair set is drained by a loop that clears any mark whose file \
             is on disk, so queueing a shard we are KEEPING fetches nothing and \
             only churns the set"
        );

        // Recording it twice is one dispute, not two — the sweep re-runs and
        // the count is what decides whether the settlement designs in
        // FUTURE_WORK are worth building.
        state.note_shard_disputed(&s, &integrity_failure([7; 32]));
        assert_eq!(state.model_registry.bytes_disputed_shards().len(), 1);

        // A later check that passes is what ends it: the hash was corrected,
        // or the origin's copy arrived.
        state.clear_shard_dispute(&s);
        assert!(!state.shard_is_disputed(&s));
        assert!(state.model_registry.bytes_disputed_shards().is_empty());
    }

    /// Zero is a measurement. The whole reason this state exists is that the
    /// count lived in a local variable inside one startup task, so a field
    /// report could not say either "it happened" or "it did not".
    #[test]
    fn a_node_with_no_disputes_can_say_so() {
        assert_eq!(test_state().disputed_shard_count(), 0);
    }

    /// **A dispute about a shard that is no longer here is not a dispute.**
    ///
    /// Three paths delete a shard this node holds — the user deleting a part,
    /// `delete_model` taking the lot, auto-manage pruning — and none of them
    /// knows about this set. Rather than oblige each of them (and every future
    /// one) to remember, the read drops what is gone. It matters because the
    /// whole point of the set is to MEASURE how often this happens: a phantom
    /// inflates the figure the settlement decision turns on, and the
    /// diagnostics report prints it as real.
    #[test]
    fn a_dispute_about_a_shard_that_is_gone_evicts_itself() {
        let state = test_state();
        let s = sid();
        state.note_shard_disputed(&s, &integrity_failure([7; 32]));
        assert!(state.shard_is_disputed(&s));

        // No file was ever written for it, which is the same thing the reader
        // sees after a delete or a prune.
        assert_eq!(
            state.disputed_shard_count(),
            0,
            "a shard with no file on disk is not in dispute"
        );
        assert!(
            !state.shard_is_disputed(&s),
            "and the entry is dropped, so the set stays bounded without a sweep"
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
