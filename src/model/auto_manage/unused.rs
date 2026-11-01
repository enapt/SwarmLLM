//! A model nobody runs is given back (#389 (d)).
//!
//! Prune acted on disk pressure and the replica target, and trust decays after
//! seven idle days; neither ever gave a model back for not being used, so a
//! swarm held whatever its testers had once tried (34 models on 2026-11-01,
//! nearly all of 2024, the 0.5B in four copies). The rule here is TinyLFU's
//! admission (Einziger et al., ACM TOS 2017): a model is held only on evidence
//! that it is used — people outside the swarm running it
//! (`huggingface::popular`), the swarm asking for it now, this computer's owner
//! choosing it, or this computer having run it within the window. A copy with
//! none of those is given back however few hold it, its last holders included —
//! the evidence says nobody wants it — and the download pass, which asks the
//! same question (`AutoShardManager::would_shed_once_fetched`), does not fetch
//! it again. Nothing here judges content (#389 (c)).
//!
//! Only a copy WATCHED for the whole window can be judged unused: a model is
//! watched from when this node first held a part of it under a build that
//! records use, so a new copy has the window as its trial and an upgrade gives
//! back nothing for `USE_WINDOW_SECS`. Without a fresh reading of what people
//! run, nothing is given back for want of use: an offline node cannot tell.

use super::manager::AutoShardManager;
use crate::types::ModelId;

/// How long a held model goes without a request before it may be given back —
/// and so the trial a new copy has. Twice trust's seven idle days: a model
/// someone runs at weekends survives a missed one.
pub(crate) const USE_WINDOW_SECS: i64 = 14 * 86_400;

/// Where the records are saved.
pub(crate) const USE_TREE: &str = "model_use";

/// A record moves in memory and on disk only when its last use moves this
/// far: against a 14-day window, an hour late is nothing, and a request does
/// not cost a database write.
const MOVES_BY_SECS: i64 = 3_600;

/// What this node has seen of one model's use (#389 (d)).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ModelUse {
    /// When this node first held a part of the model under a build that
    /// records use (Unix seconds).
    pub watched_since: i64,
    /// The last request for the model that ran here or that this node routed
    /// (Unix seconds; 0 = none seen).
    pub last_used: i64,
}

/// Has a model with this record gone the whole window without a request? No
/// record — a model this node holds no part of, which the download pass asks
/// about — says nothing either way, and the other evidence decides.
pub(crate) fn quiet_for_the_window(record: Option<ModelUse>, now: i64) -> bool {
    record.is_none_or(|r| now.saturating_sub(r.watched_since.max(r.last_used)) >= USE_WINDOW_SECS)
}

/// The record after a pass: a new one for a model held now and not before, the
/// last use moved where a later one was seen by `MOVES_BY_SECS` or more, and
/// `None` when nothing changed.
pub(crate) fn next_record(
    before: Option<ModelUse>,
    used_at: Option<i64>,
    now: i64,
) -> Option<ModelUse> {
    let used_at = used_at.filter(|&t| t > 0).map(|t| t.min(now));
    match before {
        None => Some(ModelUse {
            watched_since: now,
            last_used: used_at.unwrap_or(0),
        }),
        Some(r) => used_at
            .filter(|&t| t >= r.last_used.saturating_add(MOVES_BY_SECS))
            .map(|t| ModelUse { last_used: t, ..r }),
    }
}

impl AutoShardManager {
    /// Fold what this node saw of each model's use into its records, saved
    /// (`USE_TREE`): a request a worker ran (`ModelProcessPool::model_used_secs_ago`
    /// — every execution path, the owner's and peers' alike) or one the router
    /// took (`ModelTrustInfo::last_request_at`). A record begins when this node
    /// first holds a part of the model and ends when it holds none. Called at the
    /// start of every prune pass, so no copy is judged without one.
    pub(super) fn watch_model_use(&self) {
        let state = &self.shared_state;
        let now = chrono::Utc::now().timestamp();
        let me = state.identity.node_id().clone();
        let mut used_at: std::collections::HashMap<ModelId, i64> = state
            .model_process_pool
            .model_used_secs_ago()
            .into_iter()
            .map(|(m, ago)| {
                (
                    m,
                    now.saturating_sub(i64::try_from(ago).unwrap_or(i64::MAX)),
                )
            })
            .collect();
        for entry in state.models.model_trust.iter() {
            if let Some(at) = entry.value().last_request_at {
                let slot = used_at.entry(entry.key().clone()).or_insert(0);
                *slot = (*slot).max(at.timestamp());
            }
        }
        let mut held = std::collections::HashSet::new();
        for manifest in state.model_registry.models() {
            if state
                .model_registry
                .local_shard_indices_in(&manifest, &me)
                .is_empty()
            {
                continue;
            }
            let before = state.models.model_use.get(&manifest.id).map(|r| *r);
            if let Some(record) = next_record(before, used_at.get(&manifest.id).copied(), now) {
                state.models.model_use.insert(manifest.id.clone(), record);
                if let Err(e) = state.db.put_json(USE_TREE, &manifest.id.0, &record) {
                    tracing::warn!(model = %manifest.id, error = %e, "A model's use was not saved — watched again from a restart");
                }
            }
            held.insert(manifest.id);
        }
        let gone: Vec<ModelId> = state
            .models
            .model_use
            .iter()
            .filter(|e| !held.contains(e.key()))
            .map(|e| e.key().clone())
            .collect();
        for model in gone {
            state.models.model_use.remove(&model);
            let _ = state.db.remove(USE_TREE, &model.0);
        }
    }

    /// May this node give back every part of `model_id` (#389 (d))? Yes only
    /// when a fresh reading of what people run does not name it
    /// ([`crate::daemon::SharedState::popular_outside_the_swarm`]), and it is
    /// none of: run here within the window after being watched for it
    /// ([`quiet_for_the_window`]), chosen by this computer's owner
    /// ([`crate::daemon::SharedState::owner_chose`]), a reference model testers
    /// compare results on (`model::reference`), asked for on the swarm now
    /// (`asked_for`), or the drafter of a split this computer runs
    /// (`drafts_for_a_split` — its use counts toward the model it drafts for).
    pub(super) fn nobody_runs(&self, model_id: &ModelId) -> bool {
        let state = &self.shared_state;
        if state.popular_outside_the_swarm(model_id) != Some(false) {
            return false;
        }
        let record = state.models.model_use.get(model_id).map(|r| *r);
        quiet_for_the_window(record, chrono::Utc::now().timestamp())
            && !state.owner_chose(model_id)
            && !crate::model::reference::is_reference_model(&model_id.0)
            && !self.asked_for(model_id)
            && !self.drafts_for_a_split(model_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DAY: i64 = 86_400;

    #[test]
    fn a_copy_is_quiet_only_once_watched_for_the_whole_window() {
        let now = 100 * DAY;
        let watched = |since, last| {
            Some(ModelUse {
                watched_since: since,
                last_used: last,
            })
        };
        assert!(
            quiet_for_the_window(None, now),
            "no record: the record says nothing"
        );
        assert!(
            !quiet_for_the_window(watched(now - 13 * DAY, 0), now),
            "watched 13 days, never used: still on trial"
        );
        assert!(quiet_for_the_window(watched(now - 14 * DAY, 0), now));
        assert!(
            !quiet_for_the_window(watched(now - 60 * DAY, now - 2 * DAY), now),
            "used two days ago"
        );
        assert!(quiet_for_the_window(
            watched(now - 60 * DAY, now - 15 * DAY),
            now
        ));
        assert!(
            !quiet_for_the_window(watched(now - 2 * DAY, now - 40 * DAY), now),
            "a use from before the record began does not end the trial"
        );
    }

    #[test]
    fn a_record_begins_when_held_and_moves_only_by_the_hour() {
        let now = 50 * DAY;
        assert_eq!(
            next_record(None, None, now),
            Some(ModelUse {
                watched_since: now,
                last_used: 0
            })
        );
        assert_eq!(
            next_record(None, Some(now - 10), now),
            Some(ModelUse {
                watched_since: now,
                last_used: now - 10
            })
        );
        let r = ModelUse {
            watched_since: now - DAY,
            last_used: now - 7_200,
        };
        assert_eq!(next_record(Some(r), None, now), None, "nothing seen");
        assert_eq!(
            next_record(Some(r), Some(now - 7_100), now),
            None,
            "under an hour later: not worth a write"
        );
        assert_eq!(
            next_record(Some(r), Some(now), now),
            Some(ModelUse {
                last_used: now,
                ..r
            })
        );
        assert_eq!(
            next_record(Some(r), Some(now + 9 * DAY), now),
            Some(ModelUse {
                last_used: now,
                ..r
            }),
            "a clock ahead of now is held to now"
        );
        assert_eq!(next_record(Some(r), Some(0), now), None, "0 is no use");
    }
}
