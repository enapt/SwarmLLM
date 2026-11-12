//! A whole-model hand-off's prompt read, reported while it runs (#430,
//! `SwarmMessage::PromptProgress`).
//!
//! A requester waited a fixed budget for a hand-off's first token: 600 s, or
//! longer where the peer's read was measured (`MeasuredRead`). A peer reading
//! a long prompt slower than that failed every such prompt, knowably — the
//! budget was a minimum prompt speed nobody chose (`architecture.md` §
//! Timeouts, rule 1: prefer an inactivity timeout to a total one). So the peer
//! now says how far it has read, and the requester waits as long as that keeps
//! moving, up to a ceiling.
//!
//! Two halves, one per end:
//! - [`ProgressTargets`] (the serving peer): which hand-offs asked to be told,
//!   and to whom; each report at most every [`EVERY`], and only one that moved.
//! - [`ProgressSeen`] (the requester): the last report that ADVANCED, from the
//!   peer serving the attempt — what the first-token wait reads.
//!
//! Prior art: llama.cpp's server streams `prompt_progress {total, cache,
//! processed, time_ms}` while it reads a prompt when asked (`return_progress`)
//! and pings a silent stream every 30 s; vLLM sends SSE keep-alive comments
//! while a request is queued or in prefill.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::types::PromptProgress;

/// The serving peer reports a hand-off's read at most this often.
pub(crate) const EVERY: Duration = Duration::from_secs(2);

/// A report counts the pass's own read in this many parts (`total` on the
/// wire): a share, never a position — see [`ProgressTargets::frame_due`].
pub(crate) const OUT_OF: u32 = 1_000;

/// Hand-offs reporting at once; past it a new one is simply not reported (its
/// requester waits as it did before the reports existed).
const MOST_TARGETS: usize = 256;

struct Target {
    requester_peer: Vec<u8>,
    last_sent: Option<Instant>,
    last_processed: u32,
}

/// Lives on `state.events`. Started ONLY by the remote-generate handler for a
/// whole-model hand-off that asked (`ProgressTargets::start`, ended by the
/// guard's drop); read ONLY by the daemon's progress forwarder
/// (`daemon::background::spawn_progress_forwarder`).
#[derive(Default)]
pub struct ProgressTargets {
    map: parking_lot::Mutex<HashMap<uuid::Uuid, Target>>,
}

/// Ends a hand-off's reports when the reply ends, however it ends.
pub(crate) struct ProgressTargetGuard {
    targets: Arc<crate::daemon::SharedState>,
    request_id: uuid::Uuid,
}

impl Drop for ProgressTargetGuard {
    fn drop(&mut self) {
        self.targets
            .events
            .prompt_progress_to
            .map
            .lock()
            .remove(&self.request_id);
    }
}

impl ProgressTargets {
    /// Report `request_id`'s prompt read to `requester_peer` (the
    /// authenticated sender of the hand-off) until the guard drops. `None`
    /// where too many are reporting already, or the id is taken.
    pub(crate) fn start(
        state: &Arc<crate::daemon::SharedState>,
        request_id: uuid::Uuid,
        requester_peer: Vec<u8>,
    ) -> Option<ProgressTargetGuard> {
        let mut map = state.events.prompt_progress_to.map.lock();
        if map.len() >= MOST_TARGETS || map.contains_key(&request_id) {
            return None;
        }
        map.insert(
            request_id,
            Target {
                requester_peer,
                last_sent: None,
                last_processed: 0,
            },
        );
        Some(ProgressTargetGuard {
            targets: Arc::clone(state),
            request_id,
        })
    }

    /// The worker has read up to `done` of the prompt's `total` positions, in
    /// a pass that began at `from`: the report to send, and to whom — where
    /// this request reports, the read moved since the last report and
    /// [`EVERY`] has passed since it.
    ///
    /// The report is the PASS'S OWN share, in [`OUT_OF`]ths — never a
    /// position. The positions before `from` came from this node's prompt
    /// cache, which other people's requests fill: counted in, the first
    /// report would tell the requester how much of its prompt's opening
    /// someone else had sent here, exactly (security review, 2026-11-12).
    /// A share of the pass tells it no more than its first token's time does.
    pub(crate) fn frame_due(
        &self,
        request_id: uuid::Uuid,
        done: u32,
        total: u32,
        from: u32,
        now: Instant,
    ) -> Option<(Vec<u8>, PromptProgress)> {
        if done > total {
            return None;
        }
        let to_read = total.checked_sub(from).filter(|n| *n > 0)?;
        let read = done.checked_sub(from)?;
        let processed =
            u32::try_from(u64::from(read) * u64::from(OUT_OF) / u64::from(to_read)).ok()?;
        let mut map = self.map.lock();
        let target = map.get_mut(&request_id)?;
        if processed <= target.last_processed {
            return None;
        }
        if target
            .last_sent
            .is_some_and(|at| now.saturating_duration_since(at) < EVERY)
        {
            return None;
        }
        target.last_sent = Some(now);
        target.last_processed = processed;
        Some((
            target.requester_peer.clone(),
            PromptProgress {
                request_id,
                processed,
                total: OUT_OF,
            },
        ))
    }

    #[cfg(test)]
    fn reporting(&self) -> usize {
        self.map.lock().len()
    }
}

#[derive(Debug, Clone, Copy)]
struct Seen {
    processed: u32,
    total: u32,
    at: Instant,
    longest_gap: Duration,
}

/// The read's last advance as the first-token wait reads it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct LastAdvance {
    pub(crate) at: Instant,
    pub(crate) processed: u32,
    pub(crate) total: u32,
    /// The longest gap between two reports that moved — a layer of the
    /// peer's read, which sizes how long a silence means it stalled. Zero
    /// until a second report.
    pub(crate) longest_gap: Duration,
}

/// The requester's record of one attempt's reports (#430): held by the
/// attempt's `StreamingTokenSink`, written by dispatch's `PromptProgress` arm
/// for the peer serving the attempt only, read by the first-token wait.
#[derive(Default, Debug)]
pub struct ProgressSeen {
    last: parking_lot::Mutex<Option<Seen>>,
}

impl ProgressSeen {
    /// A report arrived: kept where it ADVANCES — the same `total` as the
    /// first, `processed` past the last and no further than `total`. Answers
    /// whether it did.
    pub(crate) fn note(&self, processed: u32, total: u32, now: Instant) -> bool {
        if total == 0 || processed > total {
            return false;
        }
        let mut last = self.last.lock();
        let advances = match *last {
            None => processed > 0,
            Some(seen) => seen.total == total && processed > seen.processed,
        };
        if advances {
            let longest_gap = last.map_or(Duration::ZERO, |seen| {
                seen.longest_gap.max(now.saturating_duration_since(seen.at))
            });
            *last = Some(Seen {
                processed,
                total,
                at: now,
                longest_gap,
            });
        }
        advances
    }

    /// When the read last moved, how far it had got, and the longest gap
    /// between two reports that moved.
    pub(crate) fn last_advance(&self) -> Option<LastAdvance> {
        self.last.lock().map(|s| LastAdvance {
            at: s.at,
            processed: s.processed,
            total: s.total,
            longest_gap: s.longest_gap,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state() -> Arc<crate::daemon::SharedState> {
        let dir = tempfile::tempdir().unwrap().keep();
        let mut config = crate::config::Config::default();
        config.node.data_dir = dir.clone();
        let db = crate::storage::db::Database::open(&dir).unwrap();
        let executor = Arc::new(tokio::sync::Mutex::new(
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

    /// A hand-off that asked is reported to its requester, no more often than
    /// `EVERY` and only when the read moved; once its guard drops, nothing.
    #[test]
    fn a_hand_off_is_reported_to_its_requester_while_it_reads() {
        let state = state();
        let targets = &state.events.prompt_progress_to;
        let rid = uuid::Uuid::new_v4();
        let t0 = Instant::now();
        assert!(
            targets.frame_due(rid, 10, 100, 0, t0).is_none(),
            "not asked"
        );
        let guard = ProgressTargets::start(&state, rid, vec![7]).expect("started");
        let (to, frame) = targets
            .frame_due(rid, 10, 100, 0, t0)
            .expect("first report");
        assert_eq!(to, vec![7]);
        assert_eq!((frame.processed, frame.total), (100, OUT_OF));
        assert!(
            targets.frame_due(rid, 20, 100, 0, t0 + EVERY / 2).is_none(),
            "too soon"
        );
        assert!(
            targets.frame_due(rid, 10, 100, 0, t0 + EVERY).is_none(),
            "did not move"
        );
        assert!(
            targets.frame_due(rid, 101, 100, 0, t0 + EVERY).is_none(),
            "past the prompt"
        );
        assert!(targets.frame_due(rid, 30, 100, 0, t0 + EVERY).is_some());
        drop(guard);
        assert_eq!(targets.reporting(), 0);
        assert!(targets.frame_due(rid, 40, 100, 0, t0 + EVERY * 2).is_none());
    }

    /// What this node's prompt cache held is never told: two passes over the
    /// same prompt, one that read it all and one that found half of it cached,
    /// report the same share once each has read the same share of its own.
    #[test]
    fn what_the_cache_held_is_never_reported() {
        let state = state();
        let targets = &state.events.prompt_progress_to;
        let (cold, warm) = (uuid::Uuid::new_v4(), uuid::Uuid::new_v4());
        let _c = ProgressTargets::start(&state, cold, vec![1]).expect("started");
        let _w = ProgressTargets::start(&state, warm, vec![1]).expect("started");
        let t0 = Instant::now();
        let (_, cold_frame) = targets.frame_due(cold, 200, 1_000, 0, t0).expect("cold");
        let (_, warm_frame) = targets.frame_due(warm, 600, 1_000, 500, t0).expect("warm");
        assert_eq!((cold_frame.processed, cold_frame.total), (200, OUT_OF));
        assert_eq!(
            (warm_frame.processed, warm_frame.total),
            (200, OUT_OF),
            "the 500 positions the cache held are not in the report"
        );
        assert!(
            targets
                .frame_due(warm, 400, 1_000, 500, t0 + EVERY)
                .is_none(),
            "a position inside what the cache held is no read"
        );
        assert!(
            targets
                .frame_due(warm, 1_000, 1_000, 1_000, t0 + EVERY)
                .is_none(),
            "nothing to read, nothing to tell"
        );
    }

    /// Only so many hand-offs report at once, and one id once.
    #[test]
    fn the_reports_are_bounded() {
        let state = state();
        let rid = uuid::Uuid::new_v4();
        let _first = ProgressTargets::start(&state, rid, vec![1]).expect("started");
        assert!(
            ProgressTargets::start(&state, rid, vec![2]).is_none(),
            "taken"
        );
        let _rest: Vec<_> = (1..MOST_TARGETS)
            .map(|_| ProgressTargets::start(&state, uuid::Uuid::new_v4(), vec![1]).expect("room"))
            .collect();
        assert!(ProgressTargets::start(&state, uuid::Uuid::new_v4(), vec![1]).is_none());
    }

    /// The requester keeps a report only where the read advances under the
    /// total it first named.
    #[test]
    fn only_an_advancing_report_is_kept() {
        let seen = ProgressSeen::default();
        let t0 = Instant::now();
        assert!(!seen.note(0, 100, t0), "nothing read yet");
        assert!(!seen.note(5, 0, t0), "no total");
        assert!(!seen.note(150, 100, t0), "past the total");
        assert!(seen.note(10, 100, t0));
        assert!(!seen.note(10, 100, t0 + EVERY), "did not move");
        assert!(!seen.note(5, 100, t0 + EVERY), "went back");
        assert!(!seen.note(20, 200, t0 + EVERY), "another total");
        let advance = |at, processed, longest_gap| LastAdvance {
            at,
            processed,
            total: 100,
            longest_gap,
        };
        assert_eq!(seen.last_advance(), Some(advance(t0, 10, Duration::ZERO)));
        assert!(seen.note(20, 100, t0 + EVERY));
        assert_eq!(seen.last_advance(), Some(advance(t0 + EVERY, 20, EVERY)));
        assert!(seen.note(30, 100, t0 + EVERY * 6));
        assert!(seen.note(40, 100, t0 + EVERY * 7));
        assert_eq!(
            seen.last_advance(),
            Some(advance(t0 + EVERY * 7, 40, EVERY * 5)),
            "the longest gap is kept"
        );
    }
}
