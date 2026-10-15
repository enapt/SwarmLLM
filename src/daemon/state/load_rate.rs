//! How fast this node reads a model into memory (`process_pool::LoadRate`),
//! kept across restarts.
//!
//! The planner charges a cold candidate its load (`NodeCandidate::cold_load_ms`,
//! FUTURE_WORK #129 (1)) at the rate that node measured and advertises
//! (`NodeCapability::model_load_ms_per_gib`), and at `UNMEASURED_LOAD_MS_PER_GIB`
//! — 10 s/GiB, the pessimistic side of a 1.3-26 s/GiB spread — where it has
//! none. The rate lived in memory only, so every restart, and so every node at
//! every release's auto-update, went back to the prior: this node's log for
//! 2026-10-14 priced the first request after each restart with a ~58 s cold
//! load that took ~9-20 s (FUTURE_WORK #3, 34 requests at 0.52× their price).
//!
//! Kept as TCP keeps a measured round trip for the next connection (Linux's
//! `tcp_metrics`) rather than starting each from RFC 6298's conservative 1 s:
//! the stored figure seeds the average, and the next timed load moves it.

use super::SharedState;
use crate::inference::process_pool::LoadRate;
use crate::storage::db::Database;

/// The redb tree and key the rate lives under.
const TREE: &str = "measured_here";
const KEY: &str = "model_load_ms_per_gib";

/// Seed `rate` from what an earlier run stored. A figure that cannot be read
/// leaves the rate unmeasured, as it was before this existed.
pub(super) fn restore(db: &Database, rate: &LoadRate) {
    match db.get_json::<u32>(TREE, KEY) {
        Ok(Some(ms_per_gib)) => {
            rate.seed(ms_per_gib);
            tracing::info!(
                ms_per_gib,
                "This node loads a model at the speed an earlier run measured until its next load is timed"
            );
        }
        Ok(None) => {}
        Err(e) => {
            tracing::debug!(error = %e, "Could not read this node's stored load rate — measuring it afresh")
        }
    }
}

/// Write `rate` if it changed since the last write.
fn persist(db: &Database, rate: &LoadRate) {
    let Some(ms_per_gib) = rate.unwritten() else {
        return;
    };
    match db.put_json(TREE, KEY, &ms_per_gib) {
        Ok(()) => rate.note_written(ms_per_gib),
        Err(e) => {
            tracing::debug!(error = %e, "Could not store this node's load rate — next tick retries")
        }
    }
}

impl SharedState {
    /// Store this node's load rate if it changed. Called on the health
    /// monitor's tick and once at shutdown, beside the lifetime record; a node
    /// that loaded nothing writes nothing.
    pub fn persist_load_rate(&self) {
        persist(&self.db, self.model_process_pool.load_rate());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **What it is for**: a restart keeps the rate the node measured, so its
    /// first cold load is priced at it rather than at the 10 s/GiB prior.
    #[test]
    fn a_restart_keeps_the_measured_load_rate() {
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(dir.path()).unwrap();

        let first = LoadRate::default();
        restore(&db, &first);
        assert_eq!(first.ms_per_gib(), None, "nothing stored yet");
        first.observe(4 << 30, 5_200);
        persist(&db, &first);

        let second = LoadRate::default();
        restore(&db, &second);
        assert_eq!(
            second.ms_per_gib(),
            Some(1_300),
            "the rate the last run measured"
        );
        // …and the next timed load moves it as the average always did.
        second.observe(1 << 30, 4_000);
        assert_eq!(second.ms_per_gib(), Some(2_110));
    }

    /// An unchanged rate is not written again, and a load timed before the
    /// stored figure is read wins over it.
    #[test]
    fn an_unchanged_rate_is_not_rewritten_and_a_newer_one_wins() {
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(dir.path()).unwrap();
        db.put_json(TREE, KEY, &1_300u32).unwrap();

        let rate = LoadRate::default();
        restore(&db, &rate);
        assert_eq!(rate.unwritten(), None, "storage already holds it");

        let measured = LoadRate::default();
        measured.observe(1 << 30, 900);
        restore(&db, &measured);
        assert_eq!(
            measured.ms_per_gib(),
            Some(900),
            "this run's own timing is newer"
        );
        assert_eq!(measured.unwritten(), Some(900));
        persist(&db, &measured);
        assert_eq!(db.get_json::<u32>(TREE, KEY).unwrap(), Some(900));
        assert_eq!(measured.unwritten(), None);
    }

    /// A figure past any load the fleet has measured — a machine suspended
    /// mid-load — is not a speed: stored, it is taken in bounded and the
    /// bounded figure written back; timed in this run, it moves the average
    /// only as far as the bound.
    #[test]
    fn an_absurd_rate_is_bounded_whether_stored_or_timed() {
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(dir.path()).unwrap();
        db.put_json(TREE, KEY, &900_000u32).unwrap();
        let rate = LoadRate::default();
        restore(&db, &rate);
        assert_eq!(rate.ms_per_gib(), Some(60_000));
        persist(&db, &rate);
        assert_eq!(db.get_json::<u32>(TREE, KEY).unwrap(), Some(60_000));

        let timed = LoadRate::default();
        timed.observe(1 << 30, 3_600_000);
        assert_eq!(timed.ms_per_gib(), Some(60_000));
    }

    /// A store that refuses the write leaves the figure unwritten, so the next
    /// tick tries again.
    #[test]
    fn a_failed_write_is_retried() {
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(dir.path()).unwrap();
        let rate = LoadRate::default();
        rate.observe(1 << 30, 3_000);
        db.set_write_failure(Some(TREE));
        persist(&db, &rate);
        assert_eq!(rate.unwritten(), Some(3_000));
        db.set_write_failure(None);
        persist(&db, &rate);
        assert_eq!(db.get_json::<u32>(TREE, KEY).unwrap(), Some(3_000));
    }
}
