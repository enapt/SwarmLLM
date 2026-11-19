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
//! for the shard again — the dashboard's Download of the model, an HF shard
//! download of it, a single-shard download, a pool pin that names this device,
//! or "Stop keeping it for me"
//! (`SharedState::let_use_decide`), which hands the model to the use rule. A configured
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

    /// The owner stops keeping `model_id` — the dashboard's "Stop keeping it
    /// for me": their download no longer pins it, "Keep on this computer" goes
    /// off, and parts they removed may be fetched again. Whether this computer
    /// holds the model is then decided as for any model — by who runs it here
    /// or through here, the swarm's demand and what people run outside the
    /// swarm (`AutoShardManager::unused_verdict`, the acquisition's trust
    /// gate). Nothing is deleted now: prune gives the parts back once nobody
    /// runs the model and other computers hold them. The default model and
    /// “Start and finish on this computer” turned on by hand stay the owner's
    /// ([`SharedState::owner_keeps`]): they say what the owner RUNS.
    ///
    /// "Remove" stays the owner's "do not download its parts again"
    /// ([`Self::mark_shard_removed_by_user`]), and this is the way back from it.
    /// Prior art: Windows Storage Sense's "Free up space" beside Delete (an
    /// "Always keep on this device" file goes online-only and comes back when
    /// opened); iOS's "Offload App", which keeps the app's place and fetches it
    /// again on a tap.
    pub fn let_use_decide(
        &self,
        model_id: &ModelId,
    ) -> Result<OwnersChoiceForgotten, crate::error::SwarmError> {
        let mut forgotten = OwnersChoiceForgotten::default();
        // Changed under the guard, stored after it: no map guard is held
        // across a database write.
        let unpinned = self
            .models
            .model_trust
            .get_mut(model_id)
            .and_then(|mut trust| {
                if !trust.pinned_by_user {
                    return None;
                }
                trust.pinned_by_user = false;
                // Pinned only by the download: back to where demand and
                // popularity promote a model from. A level earned by use stays.
                if trust.trust_level == crate::types::ModelTrustLevel::Pinned {
                    trust.trust_level = crate::types::ModelTrustLevel::Discovered;
                }
                Some(trust.value().clone())
            });
        if let Some(trust) = unpinned {
            forgotten.download = true;
            self.db.put_json("model_trust", &model_id.0, &trust)?;
        }
        if self.kept_on_this_computer(model_id) {
            self.keep_on_this_computer(model_id, false)?;
            forgotten.kept_here = true;
        }
        forgotten.removed_parts = self.clear_removed_by_user_for_model(model_id);
        Ok(forgotten)
    }

    /// The owner asked for `model_id` — the dashboard's Download or a
    /// HuggingFace download of its parts: their download pins it
    /// ([`SharedState::owner_fetched`]), stored so a restart keeps it. The
    /// inverse of [`Self::let_use_decide`]; the caller forgets whichever
    /// removals its request names.
    pub fn owner_fetches(&self, model_id: &ModelId) -> Result<(), crate::error::SwarmError> {
        // Changed under the guard, stored after it.
        let trust = {
            let mut trust = self
                .models
                .model_trust
                .entry(model_id.clone())
                .or_insert_with(crate::types::ModelTrustInfo::new_pinned);
            trust.pinned_by_user = true;
            if trust.trust_level < crate::types::ModelTrustLevel::Pinned {
                trust.trust_level = crate::types::ModelTrustLevel::Pinned;
            }
            trust.value().clone()
        };
        self.db.put_json("model_trust", &model_id.0, &trust)
    }
}

/// What [`SharedState::let_use_decide`] forgot about a model.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub struct OwnersChoiceForgotten {
    /// The owner's download no longer pins it.
    pub download: bool,
    /// "Keep on this computer" was on and is now off.
    pub kept_here: bool,
    /// Parts the owner had removed that may now be fetched again.
    pub removed_parts: usize,
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

    /// "Stop keeping it for me" forgets the owner's download, their "Keep on
    /// this computer" and their removals of THAT model, stored — and leaves a
    /// trust level earned by use alone.
    #[test]
    fn letting_use_decide_forgets_only_what_the_owner_chose_for_that_model() {
        use crate::types::{ModelTrustInfo, ModelTrustLevel};
        let st = state();
        let m = ModelId("m".into());
        st.models
            .model_trust
            .insert(m.clone(), ModelTrustInfo::new_pinned());
        st.keep_on_this_computer(&m, true).expect("keep");
        st.mark_shard_removed_by_user(&sid("m", 1));
        st.mark_shard_removed_by_user(&sid("other", 0));
        assert!(st.owner_chose(&m));

        let forgotten = st.let_use_decide(&m).expect("forgotten");
        assert_eq!(
            forgotten,
            OwnersChoiceForgotten {
                download: true,
                kept_here: true,
                removed_parts: 1
            }
        );
        assert!(!st.owner_chose(&m), "nothing the owner chose holds it now");
        assert!(!st.shard_removed_by_user(&sid("m", 1)));
        assert!(
            st.shard_removed_by_user(&sid("other", 0)),
            "another model's removal stands"
        );
        let level = st
            .models
            .model_trust
            .get(&m)
            .expect("trust")
            .trust_level
            .clone();
        assert_eq!(
            level,
            ModelTrustLevel::Discovered,
            "back where demand and popularity promote from"
        );
        let stored: ModelTrustInfo = st
            .db
            .get_json("model_trust", "m")
            .expect("read")
            .expect("stored");
        assert!(!stored.pinned_by_user, "a restart does not pin it again");

        let earned = ModelId("earned".into());
        let mut used = ModelTrustInfo::new_pinned();
        used.trust_level = ModelTrustLevel::DemandVerified;
        st.models.model_trust.insert(earned.clone(), used);
        st.let_use_decide(&earned).expect("earned");
        let level = st
            .models
            .model_trust
            .get(&earned)
            .expect("trust")
            .trust_level
            .clone();
        assert_eq!(
            level,
            ModelTrustLevel::DemandVerified,
            "a level earned by use stays"
        );

        assert_eq!(
            st.let_use_decide(&m).expect("again"),
            OwnersChoiceForgotten::default(),
            "nothing left to forget is not an error"
        );
    }

    /// The owner's download pins a model, stored — and "Stop keeping it for
    /// me" is its way back.
    #[test]
    fn the_owners_download_is_stored_and_letting_use_decide_undoes_it() {
        use crate::types::{ModelTrustInfo, ModelTrustLevel};
        let st = state();
        let m = ModelId("m".into());
        st.models
            .model_trust
            .insert(m.clone(), ModelTrustInfo::new_discovered());
        st.owner_fetches(&m).expect("fetch");
        assert!(st.owner_fetched(&m));
        let stored: ModelTrustInfo = st
            .db
            .get_json("model_trust", "m")
            .expect("read")
            .expect("stored");
        assert!(
            stored.pinned_by_user,
            "a restart keeps the owner's download"
        );
        assert_eq!(stored.trust_level, ModelTrustLevel::Pinned);

        st.let_use_decide(&m).expect("forget");
        assert!(!st.owner_fetched(&m));
    }
}
