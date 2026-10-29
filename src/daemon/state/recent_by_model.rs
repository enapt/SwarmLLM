//! The recent figures this node keeps per model to price its next route —
//! how long a reply runs (`reply_lengths`, FUTURE_WORK #3 (b)) and how many
//! tokens a round of a split yields (`split_rounds`, #3 (d)) — kept across
//! restarts.
//!
//! Both lived in memory only, and neither is trusted before
//! [`BEFORE_TRUSTED`] entries, so every restart — and so every node at every
//! release's auto-update — priced its next requests at the priors: a reply of
//! `ASSUMED_FORWARD_PASSES` and a split at one round a token. The 70B's first
//! three splits after an update (2026-10-29 10:46-10:53Z, this node + an RTX
//! 3060 in Australia) were priced at 1.7-2.4x the time they took, each round
//! charged as one token where 2.2 were written; a model whose split competes
//! with running whole here pays that as a lost route.
//!
//! Kept as `state::load_rate` keeps the load rate — and as Linux's
//! `tcp_metrics` keeps a measured round trip for the next connection — and as
//! Petals keeps its measured throughput (`server/throughput.py`: a JSON cache
//! keyed by model, device and dtype, under a versioned name,
//! `throughput_v5.json`, so a change in what is measured starts afresh). Here
//! the version is the record's `counted_as`: a record counted another way is
//! dropped at start, never priced.
//!
//! **What is stored is counts** — a reply's length in tokens, a split's tokens
//! and rounds — per model id, never anything a reply said.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, Ordering};

use dashmap::DashMap;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

use crate::storage::db::Database;

/// Figures remembered per model: enough that one outlier moves a mean by a
/// thirty-second of itself, few enough that a change in how the model is used
/// shows within a session.
pub const RECENT: usize = 32;

/// Figures a model must have before they are trusted over the caller's
/// prior. Below it one reply would decide the next route.
pub const BEFORE_TRUSTED: usize = 4;

/// At most this many models are tracked — a model id reaches here only from a
/// request that completed, but the map is bounded anyway.
const MAX_MODELS: usize = 256;

/// The redb tree, shared with the other figures this node measured
/// (`load_rate`, `auto_manage::target_hold`).
const TREE: &str = "measured_here";

/// One stored record: what a figure meant when it was written, and the
/// figures, oldest first.
#[derive(Serialize, Deserialize)]
struct Record<T> {
    counted_as: u32,
    by_model: HashMap<String, Vec<T>>,
}

/// The last [`RECENT`] figures of each model, kept under `key`.
pub struct RecentByModel<T> {
    by_model: DashMap<String, VecDeque<T>>,
    /// Something was recorded since the last write.
    unwritten: AtomicBool,
    key: &'static str,
    /// Bumped where what a figure counts changes, so a record from before is
    /// dropped rather than priced.
    counted_as: u32,
}

impl<T: Copy + Serialize + DeserializeOwned> RecentByModel<T> {
    pub fn new(key: &'static str, counted_as: u32) -> Self {
        Self {
            by_model: DashMap::new(),
            unwritten: AtomicBool::new(false),
            key,
            counted_as,
        }
    }

    /// One more figure of `model`; the oldest goes past [`RECENT`].
    pub fn push(&self, model: &str, figure: T) {
        if !self.by_model.contains_key(model) && self.by_model.len() >= MAX_MODELS {
            return;
        }
        {
            let mut recent = self.by_model.entry(model.to_string()).or_default();
            if recent.len() >= RECENT {
                recent.pop_front();
            }
            recent.push_back(figure);
        }
        self.unwritten.store(true, Ordering::Relaxed);
    }

    /// `f` over `model`'s recent figures, oldest first, once there are
    /// [`BEFORE_TRUSTED`] of them; `None` until then.
    pub fn with_trusted<R>(&self, model: &str, f: impl FnOnce(&VecDeque<T>) -> R) -> Option<R> {
        let recent = self.by_model.get(model)?;
        (recent.len() >= BEFORE_TRUSTED).then(|| f(&recent))
    }

    /// Seed from what an earlier run stored: only figures `usable` accepts,
    /// the last [`RECENT`] a model, never over one recorded since start. A
    /// record that cannot be read, or was counted another way, leaves the
    /// figures empty, as they were before this existed.
    pub fn restore(&self, db: &Database, usable: impl Fn(&T) -> bool) {
        let record = match db.get_json::<Record<T>>(TREE, self.key) {
            Ok(Some(record)) => record,
            Ok(None) => return,
            Err(e) => {
                tracing::debug!(key = self.key, error = %e, "Could not read figures an earlier run stored — measuring them afresh");
                return;
            }
        };
        if record.counted_as != self.counted_as {
            tracing::info!(
                key = self.key,
                stored = record.counted_as,
                now = self.counted_as,
                "Figures an earlier run stored were counted another way — measuring them afresh"
            );
            return;
        }
        let mut models = 0usize;
        for (model, figures) in record.by_model {
            if self.by_model.len() >= MAX_MODELS {
                break;
            }
            let kept: Vec<T> = figures.into_iter().filter(|f| usable(f)).collect();
            let recent: VecDeque<T> = kept[kept.len().saturating_sub(RECENT)..]
                .iter()
                .copied()
                .collect();
            if recent.is_empty() {
                continue;
            }
            self.by_model.entry(model).or_insert_with(|| {
                models += 1;
                recent
            });
        }
        if models > 0 {
            tracing::info!(
                key = self.key,
                models,
                "Pricing starts from the figures an earlier run measured"
            );
        }
    }

    /// Write the figures if something was recorded since the last write. A
    /// failed write is retried at the next call.
    pub fn persist(&self, db: &Database) {
        if !self.unwritten.swap(false, Ordering::Relaxed) {
            return;
        }
        let record = Record {
            counted_as: self.counted_as,
            by_model: self
                .by_model
                .iter()
                .map(|e| (e.key().clone(), e.value().iter().copied().collect()))
                .collect(),
        };
        if let Err(e) = db.put_json(TREE, self.key, &record) {
            self.unwritten.store(true, Ordering::Relaxed);
            tracing::debug!(key = self.key, error = %e, "Could not store the figures this node measured — next tick retries");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rounds(db: &Database) -> RecentByModel<(u32, u32)> {
        let figures = RecentByModel::new("split_rounds_test", 1);
        figures.restore(db, |&(t, r)| t > 0 && r > 0);
        figures
    }

    fn all(figures: &RecentByModel<(u32, u32)>, model: &str) -> Option<Vec<(u32, u32)>> {
        figures.with_trusted(model, |recent| recent.iter().copied().collect())
    }

    /// **What it is for**: a restart keeps what the last run measured, so the
    /// next route is priced at it rather than at the prior.
    #[test]
    fn a_restart_keeps_the_figures_the_last_run_measured() {
        let db = Database::open_temp().unwrap();
        let first = rounds(&db);
        assert_eq!(all(&first, "m"), None, "nothing stored yet");
        for i in 1..=BEFORE_TRUSTED as u32 {
            first.push("m", (100 * i, 40));
        }
        first.persist(&db);

        let second = rounds(&db);
        assert_eq!(
            all(&second, "m"),
            Some(vec![(100, 40), (200, 40), (300, 40), (400, 40)])
        );
        // …and the next figure joins them as it always did.
        second.push("m", (50, 50));
        assert_eq!(all(&second, "m").unwrap().last(), Some(&(50, 50)));
    }

    /// A record counted another way is dropped, never priced — the next
    /// write replaces it.
    #[test]
    fn figures_counted_another_way_are_dropped() {
        let db = Database::open_temp().unwrap();
        let old = RecentByModel::<(u32, u32)>::new("split_rounds_test", 0);
        for _ in 0..BEFORE_TRUSTED {
            old.push("m", (300, 600));
        }
        old.persist(&db);
        let now = rounds(&db);
        assert_eq!(all(&now, "m"), None);
    }

    /// Only usable figures are restored, the last [`RECENT`] of a model, and a
    /// figure recorded before the restore is never overwritten.
    #[test]
    fn a_restore_keeps_only_what_it_could_have_recorded() {
        let db = Database::open_temp().unwrap();
        let mut stored: HashMap<String, Vec<(u32, u32)>> = HashMap::new();
        let mut long: Vec<(u32, u32)> = (1..=RECENT as u32 + 8).map(|i| (i, 1)).collect();
        long.insert(3, (0, 5));
        stored.insert("long".into(), long);
        stored.insert("here".into(), vec![(9, 9); BEFORE_TRUSTED]);
        db.put_json(
            TREE,
            "split_rounds_test",
            &Record {
                counted_as: 1,
                by_model: stored,
            },
        )
        .unwrap();

        let figures = RecentByModel::<(u32, u32)>::new("split_rounds_test", 1);
        for _ in 0..BEFORE_TRUSTED {
            figures.push("here", (2, 2));
        }
        figures.restore(&db, |&(t, r)| t > 0 && r > 0);
        let long = all(&figures, "long").unwrap();
        assert_eq!(long.len(), RECENT);
        assert_eq!(long.first(), Some(&(9, 1)), "the oldest went");
        assert!(long.iter().all(|&(t, r)| t > 0 && r > 0));
        assert_eq!(all(&figures, "here"), Some(vec![(2, 2); BEFORE_TRUSTED]));
    }

    /// Nothing recorded, nothing written; a failed write is retried.
    #[test]
    fn only_a_change_is_written_and_a_failed_write_is_retried() {
        let db = Database::open_temp().unwrap();
        let figures = rounds(&db);
        figures.persist(&db);
        assert!(db
            .get_json::<Record<(u32, u32)>>(TREE, "split_rounds_test")
            .unwrap()
            .is_none());
        for _ in 0..BEFORE_TRUSTED {
            figures.push("m", (5, 2));
        }
        db.set_write_failure(Some(TREE));
        figures.persist(&db);
        db.set_write_failure(None);
        figures.persist(&db);
        assert_eq!(all(&rounds(&db), "m"), Some(vec![(5, 2); BEFORE_TRUSTED]));
    }
}
