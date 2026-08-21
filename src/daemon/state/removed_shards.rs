//! Shards the operator removed by hand, remembered so auto-manage does not
//! quietly bring them back.
//!
//! Why this exists (external report, 2026-08-21): a user split a model across
//! two machines by deleting shards from one of them, and within the hour the
//! deleted shards were downloading again — the scorer's gap-filling rule
//! ("already hosting part of this model, so complete it") saw a partial model
//! and did what it was built to do. The operator's topology was erased with
//! no error and no message, and the only thing that would have held it was
//! `inference.shard_range`, which nobody reading the dashboard can know about.
//!
//! The rule that follows: **a removal the user performed is an instruction,
//! not a gap.** It is recorded here, persisted across restarts (the fact is
//! about what the user wants, and a restart does not change that), honoured
//! by `gather_candidates`, and cleared only by an equally explicit request
//! for the shard again — an HF shard download of the model, a single-shard
//! download, or a pool pin that names this device. A configured
//! `shard_range` and a pin to this node both outrank it in the scorer, since
//! each is itself an explicit instruction about what this node should hold.

use super::SharedState;
use crate::types::{ModelId, ShardId};

/// DB tree: key = `serde_json(ShardId)`, value = `b"1"`. Same shape as
/// `locked_shards`, so the startup load in `SharedState::new` mirrors it.
pub(super) const REMOVED_SHARDS_TREE: &str = "removed_shards";

fn key_for(shard_id: &ShardId) -> String {
    // ShardId's serde form is a flat struct; infallible for our own type.
    serde_json::to_string(shard_id).unwrap_or_default()
}

impl SharedState {
    /// Record that the operator removed this shard from this node. Persisted.
    pub fn mark_shard_removed_by_user(&self, shard_id: &ShardId) {
        self.models.removed_by_user.insert(shard_id.clone(), true);
        if let Err(e) = self
            .db
            .insert_raw(REMOVED_SHARDS_TREE, &key_for(shard_id), b"1")
        {
            tracing::warn!(
                error = %e,
                shard = ?shard_id,
                "Could not persist the shard removal — auto-manage may re-download it after a restart"
            );
        }
    }

    /// Was this shard removed by the operator (and not explicitly asked for
    /// again since)?
    pub fn shard_removed_by_user(&self, shard_id: &ShardId) -> bool {
        self.models.removed_by_user.contains_key(shard_id)
    }

    /// The operator asked for this shard again: forget the removal. Returns
    /// whether there was one to forget.
    pub fn clear_shard_removed_by_user(&self, shard_id: &ShardId) -> bool {
        let had = self.models.removed_by_user.remove(shard_id).is_some();
        if had {
            if let Err(e) = self.db.remove(REMOVED_SHARDS_TREE, &key_for(shard_id)) {
                tracing::warn!(error = %e, shard = ?shard_id, "Could not clear a persisted shard removal");
            }
        }
        had
    }

    /// The operator asked for this model again (e.g. an HF shard download of
    /// it): forget every removal recorded for it. Returns how many there were.
    pub fn clear_removed_by_user_for_model(&self, model_id: &ModelId) -> usize {
        let shards: Vec<ShardId> = self
            .models
            .removed_by_user
            .iter()
            .filter(|e| &e.key().model_id == model_id)
            .map(|e| e.key().clone())
            .collect();
        for s in &shards {
            self.clear_shard_removed_by_user(s);
        }
        shards.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use crate::daemon::state::SharedState;
    use crate::identity::Identity;
    use crate::inference::executor::ModelExecutor;
    use crate::storage::db::Database;
    use std::sync::Arc;
    use tokio::sync::Mutex;

    fn state() -> Arc<SharedState> {
        let temp = tempfile::tempdir().expect("tempdir");
        let db = Database::open(temp.path()).expect("db");
        let executor = Arc::new(Mutex::new(ModelExecutor::new()));
        let (state, _, _) =
            SharedState::new(Config::default(), Identity::generate(), db, executor, None);
        state
    }

    fn sid(model: &str, index: u32) -> ShardId {
        ShardId {
            model_id: ModelId(model.to_string()),
            index,
        }
    }

    #[test]
    fn a_removal_is_remembered_until_the_shard_is_asked_for_again() {
        let st = state();
        let s = sid("m", 2);
        assert!(!st.shard_removed_by_user(&s));
        st.mark_shard_removed_by_user(&s);
        assert!(st.shard_removed_by_user(&s));
        // Persisted, not just in memory: the fact outlives the process.
        let on_disk = st.db.iter_raw(REMOVED_SHARDS_TREE).expect("iter");
        assert_eq!(on_disk.len(), 1);
        assert!(st.clear_shard_removed_by_user(&s));
        assert!(!st.shard_removed_by_user(&s));
        assert!(st
            .db
            .iter_raw(REMOVED_SHARDS_TREE)
            .expect("iter")
            .is_empty());
        assert!(
            !st.clear_shard_removed_by_user(&s),
            "nothing left to forget"
        );
    }

    #[test]
    fn asking_for_the_model_again_forgets_every_removal_for_it_and_no_other() {
        let st = state();
        st.mark_shard_removed_by_user(&sid("a", 0));
        st.mark_shard_removed_by_user(&sid("a", 3));
        st.mark_shard_removed_by_user(&sid("b", 1));
        assert_eq!(st.clear_removed_by_user_for_model(&ModelId("a".into())), 2);
        assert!(!st.shard_removed_by_user(&sid("a", 0)));
        assert!(!st.shard_removed_by_user(&sid("a", 3)));
        assert!(
            st.shard_removed_by_user(&sid("b", 1)),
            "another model's removal stands"
        );
        assert_eq!(st.db.iter_raw(REMOVED_SHARDS_TREE).expect("iter").len(), 1);
    }
}
