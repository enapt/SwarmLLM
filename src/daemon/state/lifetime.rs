//! What this node has served for other people since it was first started —
//! kept across restarts (FUTURE_WORK #226).
//!
//! Every serving counter in `MetricsProviders` lives in memory, so a restart
//! zeroes it, and auto-update restarts a node about once a day. A tester's node
//! read `99.3 ms per layer served` before an update and `(no segments served
//! yet)` after it (2026-10-06): the one question an operator deciding whether to
//! keep a node running asks — "has my computer ever helped anyone?" — had no
//! answer on the node itself, and a node that served badly lost the evidence at
//! its next update.
//!
//! The record is the totals as they stood when this run started, plus this
//! run's counters, written as one absolute figure — so writing it twice, or
//! after a crash that lost the last write, never counts anything twice.
//! Long-running P2P clients keep all-time totals the same way: qBittorrent
//! persists `alltime_ul`/`alltime_dl` beside its per-session figures.
//!
//! ⚠ This is a description of work done, not a balance. Credits are DORMANT
//! (`docs/CREDITS_DESIGN.md`); nothing may read these totals as an entitlement.

use std::sync::atomic::Ordering::Relaxed;

use serde::{Deserialize, Serialize};

/// The redb tree and key the record lives under.
const TREE: &str = "lifetime";
const KEY: &str = "served";

/// Work served for other people. Every field defaults, so a record written by
/// an older build that knew fewer of them still reads.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ServedTotals {
    /// Whole requests answered for a peer (the hand-off).
    #[serde(default)]
    pub requests: u64,
    /// Forwards computed for a peer — a whole request or one segment of one.
    #[serde(default)]
    pub forwards: u64,
    #[serde(default)]
    pub segments: u64,
    #[serde(default)]
    pub layers: u64,
    #[serde(default)]
    pub tokens: u64,
    #[serde(default)]
    pub serve_micros: u64,
    #[serde(default)]
    pub activation_bytes_out: u64,
}

impl ServedTotals {
    fn plus(self, other: Self) -> Self {
        Self {
            requests: self.requests.saturating_add(other.requests),
            forwards: self.forwards.saturating_add(other.forwards),
            segments: self.segments.saturating_add(other.segments),
            layers: self.layers.saturating_add(other.layers),
            tokens: self.tokens.saturating_add(other.tokens),
            serve_micros: self.serve_micros.saturating_add(other.serve_micros),
            activation_bytes_out: self
                .activation_bytes_out
                .saturating_add(other.activation_bytes_out),
        }
    }
}

/// The lifetime record: since when, and how much.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LifetimeServed {
    /// When this node first started keeping the record.
    pub since: chrono::DateTime<chrono::Utc>,
    pub totals: ServedTotals,
}

/// The record as this run found it, and the last figure written back.
pub struct LifetimeLedger {
    at_start: LifetimeServed,
    last_written: parking_lot::Mutex<Option<ServedTotals>>,
}

impl LifetimeLedger {
    /// Read the record a previous run left, or start one now. A record that
    /// cannot be read is a fresh start, said once — never a reason not to run.
    pub fn load(db: &crate::storage::db::Database) -> Self {
        let at_start = match db.get_json::<LifetimeServed>(TREE, KEY) {
            Ok(Some(record)) => record,
            Ok(None) => LifetimeServed {
                since: chrono::Utc::now(),
                totals: ServedTotals::default(),
            },
            Err(e) => {
                tracing::warn!(error = %e, "Could not read the lifetime serving record — starting it afresh");
                LifetimeServed {
                    since: chrono::Utc::now(),
                    totals: ServedTotals::default(),
                }
            }
        };
        Self {
            at_start,
            last_written: parking_lot::Mutex::new(None),
        }
    }
}

impl super::SharedState {
    /// What this run has served, read from the live counters
    /// (`SharedState::record_peer_serve` is their one writer).
    pub fn served_this_run(&self) -> ServedTotals {
        let m = &self.metrics;
        ServedTotals {
            requests: m.requests_served_atomic.load(Relaxed),
            forwards: m.forwards_served_atomic.load(Relaxed),
            segments: m.segments_served.load(Relaxed),
            layers: m.layers_served.load(Relaxed),
            tokens: m.tokens_served.load(Relaxed),
            serve_micros: m.segment_serve_micros.load(Relaxed),
            activation_bytes_out: m.segment_bytes_out.load(Relaxed),
        }
    }

    /// Everything served since the record began: what earlier runs left plus
    /// this run.
    pub fn served_lifetime(&self) -> LifetimeServed {
        let ledger = &self.metrics.lifetime_served;
        LifetimeServed {
            since: ledger.at_start.since,
            totals: ledger.at_start.totals.plus(self.served_this_run()),
        }
    }

    /// Write the lifetime record if it changed since the last write. Called on
    /// the health monitor's tick and once at shutdown; an idle node writes
    /// nothing.
    pub fn persist_served_lifetime(&self) {
        let record = self.served_lifetime();
        let mut last = self.metrics.lifetime_served.last_written.lock();
        if *last == Some(record.totals) {
            return;
        }
        match self.db.put_json(TREE, KEY, &record) {
            Ok(()) => *last = Some(record.totals),
            Err(e) => {
                tracing::debug!(error = %e, "Could not write the lifetime serving record — next tick retries")
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **What it is for**: work served before a restart is still counted after
    /// it — the tester's node lost its whole record at every update.
    #[test]
    fn a_restart_keeps_what_was_served() {
        let dir = tempfile::tempdir().unwrap();
        let db = crate::storage::db::Database::open(dir.path()).unwrap();

        let first = LifetimeLedger::load(&db);
        assert_eq!(first.at_start.totals, ServedTotals::default());
        let served = ServedTotals {
            requests: 3,
            tokens: 1200,
            segments: 40,
            ..Default::default()
        };
        db.put_json(
            TREE,
            KEY,
            &LifetimeServed {
                since: first.at_start.since,
                totals: first.at_start.totals.plus(served),
            },
        )
        .unwrap();

        let second = LifetimeLedger::load(&db);
        assert_eq!(
            second.at_start.totals, served,
            "the next run starts from what was served"
        );
        assert_eq!(
            second.at_start.since, first.at_start.since,
            "and keeps when the record began"
        );
    }

    /// A record from a build that knew fewer fields still reads.
    #[test]
    fn an_older_record_still_reads() {
        let old = serde_json::json!({
            "since": "2026-09-14T00:00:00Z",
            "totals": { "requests": 5, "tokens": 900 }
        });
        let record: LifetimeServed = serde_json::from_value(old).unwrap();
        assert_eq!(record.totals.requests, 5);
        assert_eq!(record.totals.segments, 0);
    }
}
