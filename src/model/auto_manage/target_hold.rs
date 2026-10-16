//! The replica target prune sheds against: the highest the target has read
//! over the last day (field report 2026-10-16, FUTURE_WORK #286).
//!
//! `geo_target_replicas` moves while nothing about a part changes. One request
//! raises a model's demand factor to 1.5 — a target of 2 becomes 3 on a swarm
//! of up to eight — and the average it feeds falls back under the threshold
//! about half an hour later; a ninth connected peer moves the floor from 2 to
//! 3. The download pass fetches below the target as it reads when it looks, and
//! prune sheds above it as it reads when IT looks, so the one predicate both
//! ask (`would_shed_copy`, gotcha #795) still gave two answers about the same
//! part, minutes apart. A tester's node holding a whole Qwen3-30B fetched part
//! 25 and deleted it 28 s later, three times in 13 hours; from our own log, one
//! peer dropped and fetched back parts of a 14B about every six minutes for a
//! day and a half.
//!
//! Kubernetes' Horizontal Pod Autoscaler meets the same flapping the same way
//! (KEP-853, `behavior.scaleDown.stabilizationWindowSeconds`): it scales UP at
//! once and scales DOWN only to the HIGHEST recommendation within a window. Here
//! the download pass reads the target as it is, and prune reads the highest it
//! has been within [`HOLD_SECS`] — a day rather than HPA's five minutes, because the
//! demand behind a model follows the day: a model somebody uses each evening
//! keeps the copies that use fetched, where an hour's window would shed them
//! every night and fetch them back the next evening. Disk pressure still lowers
//! the target the held figure feeds (`effective_prune_target`), so a filling
//! disk frees space as before.
//!
//! Kept across restarts, as the load rate is (`daemon::state::load_rate`): a
//! node restarts into every release's update and its demand figures are in
//! memory only, so without the record every update read as "nobody needs the
//! extra copies" — a peer began dropping parts of a model three minutes after
//! its v0.3.240 restart.

use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicBool, Ordering};

use dashmap::DashMap;

use crate::storage::db::Database;
use crate::types::ModelId;

/// How long prune keeps shedding against a target the model has read.
pub(super) const HOLD_SECS: i64 = 24 * 3600;

/// How far a target's last sighting moves before it is recorded again. The
/// hold is therefore between `HOLD_SECS - STEP_SECS` and `HOLD_SECS`, and the
/// record is written at most once a step per model however often the target is
/// read (every part, every cycle).
const STEP_SECS: i64 = 600;

/// The redb tree and key the record lives under — beside the load rate.
const TREE: &str = "measured_here";
const KEY: &str = "replica_targets_seen";

/// Each model's targets, and when each was last read (Unix seconds).
#[derive(Default)]
pub(crate) struct TargetsSeen {
    seen: DashMap<ModelId, BTreeMap<u32, i64>>,
    unwritten: AtomicBool,
}

impl TargetsSeen {
    /// What an earlier run recorded. A record that cannot be read starts
    /// empty — prune then sheds against the target as it reads, as it did
    /// before this existed.
    pub(super) fn restore(db: &Database) -> Self {
        let restored = Self::default();
        match db.get_json::<HashMap<String, BTreeMap<u32, i64>>>(TREE, KEY) {
            Ok(Some(record)) => {
                for (model, targets) in record {
                    restored.seen.insert(ModelId(model), targets);
                }
            }
            Ok(None) => {}
            Err(e) => {
                tracing::debug!(error = %e, "Could not read the replica targets an earlier run saw — starting afresh")
            }
        }
        restored
    }

    /// Note that `model`'s target read `target` at `now`.
    pub(super) fn observe(&self, model: &ModelId, target: u32, now: i64) {
        let mut targets = self.seen.entry(model.clone()).or_default();
        let last = targets.get(&target).copied();
        if last.is_none_or(|at| now - at >= STEP_SECS) {
            targets.insert(target, now);
            self.unwritten.store(true, Ordering::Relaxed);
        }
    }

    /// The highest target `model` has read within [`HOLD_SECS`] of `now`.
    pub(super) fn held(&self, model: &ModelId, now: i64) -> Option<u32> {
        self.seen.get(model).and_then(|targets| {
            targets
                .iter()
                .rev()
                .find(|(_, &at)| now - at <= HOLD_SECS)
                .map(|(&target, _)| target)
        })
    }

    /// Write the record if it changed, dropping what the hold no longer reads.
    pub(super) fn persist(&self, db: &Database, now: i64) {
        if !self.unwritten.swap(false, Ordering::Relaxed) {
            return;
        }
        self.seen.retain(|_, targets| {
            targets.retain(|_, at| now - *at <= HOLD_SECS);
            !targets.is_empty()
        });
        let record: HashMap<String, BTreeMap<u32, i64>> = self
            .seen
            .iter()
            .map(|e| (e.key().0.clone(), e.value().clone()))
            .collect();
        if let Err(e) = db.put_json(TREE, KEY, &record) {
            self.unwritten.store(true, Ordering::Relaxed);
            tracing::debug!(error = %e, "Could not store the replica targets seen — next cycle retries");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model() -> ModelId {
        ModelId("qwen3-30b".into())
    }

    /// **What it is for**: a target that read 3 for one request is still what
    /// prune sheds against once the demand behind it has decayed to 2 — the
    /// part that request fetched is not surplus half an hour later.
    #[test]
    fn prune_sheds_against_the_highest_target_of_the_last_day() {
        let seen = TargetsSeen::default();
        seen.observe(&model(), 3, 0);
        seen.observe(&model(), 2, 1_800);
        assert_eq!(seen.held(&model(), 1_800), Some(3));
        assert_eq!(
            seen.held(&model(), HOLD_SECS),
            Some(3),
            "a day on, still held"
        );
        seen.observe(&model(), 2, HOLD_SECS + 1);
        assert_eq!(
            seen.held(&model(), HOLD_SECS + 1),
            Some(2),
            "past the day, the target as it reads"
        );
        assert_eq!(seen.held(&ModelId("other".into()), 0), None);
    }

    /// A target read again within a step is not rewritten, and one read again
    /// past it is — so the record stays within a step of the truth and is
    /// written at most once a step.
    #[test]
    fn a_target_seen_again_is_recorded_once_a_step() {
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(dir.path()).unwrap();
        let seen = TargetsSeen::default();
        seen.observe(&model(), 3, 0);
        seen.persist(&db, 0);
        seen.observe(&model(), 3, STEP_SECS - 1);
        assert!(!seen.unwritten.load(Ordering::Relaxed), "within a step");
        seen.observe(&model(), 3, STEP_SECS);
        assert!(seen.unwritten.load(Ordering::Relaxed), "a step on");
    }

    /// A restart keeps what the last run saw, so the update every release
    /// brings does not read as "nobody needs the extra copies"; what the hold
    /// no longer reads is not carried forward.
    #[test]
    fn a_restart_keeps_the_targets_the_last_run_saw() {
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(dir.path()).unwrap();
        let first = TargetsSeen::default();
        first.observe(&model(), 3, 1_000);
        first.observe(&ModelId("old".into()), 4, 0);
        first.persist(&db, HOLD_SECS + 500);

        let second = TargetsSeen::restore(&db);
        assert_eq!(second.held(&model(), HOLD_SECS + 500), Some(3));
        assert_eq!(
            second.held(&ModelId("old".into()), HOLD_SECS + 500),
            None,
            "past the day when written: dropped"
        );
        assert!(!second.seen.contains_key(&ModelId("old".into())));
    }
}
