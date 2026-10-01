//! A streamed verify's forwards run on this node in their stream's order.
//!
//! A coordinator streaming speculative checks (`docs/plans/split_speculation.md`
//! § 4b) keeps several verify forwards of ONE request in flight to the segment
//! that samples, numbered by `LayerForward::stream_seq`. They must run here in
//! that order: the worker writes each at its cache's CURRENT length while RoPE
//! rotates by the forward's `index_pos`, so two chunks swapped are read into
//! each other's positions with nothing failing. And nothing else orders them —
//! an encrypted forward is opened in its own task and the dispatcher spawns a
//! handler per forward, so two that arrive a few milliseconds apart race to the
//! worker.
//!
//! PipeInfer (SC'24) meets the same need with MPI's non-overtaking rule for one
//! sender, receiver and tag; this is that rule for one request's stream: a
//! forward waits here until the one numbered before it has ended, whatever order
//! the network delivered them in. It also keeps one forward per request at the
//! worker, whose reply routing is keyed by request id (gotcha #180).
//!
//! Keyed by (request, layer range, attempt) — the range is the segment, and the
//! attempt is the high bits of the number (`types::inference::stream_seq`), so a
//! router retry under the same request id is a stream of its own and a dead
//! attempt's chunks never block or answer it (gotcha #749). Turns count from 0
//! within each. Idle streams are swept on the health tick.
//!
//! **A restart skips what it supersedes.** A streamed forward that cuts the
//! cache back (`truncate_kv_to`) starts its stream over: the coordinator sends
//! one only after a check refused a guess, so every earlier turn of the stream
//! that has not yet run here was built on that guess and its answer will be
//! thrown away. Those are skipped — each still takes its turn, in order, and
//! gives it straight back, so nothing ever runs beside the forward before it
//! (the worker routes replies by request id, gotcha #180). PipeInfer's early
//! inference cancellation, done without a message of its own: the restart is
//! the signal. It matters where this node is busy — the work skipped is work
//! the restart would otherwise wait behind.
//!
//! **A turn refused on arrival is stepped over, never waited for.** A forward
//! the dispatcher turns away at an admission cap never reaches [`ForwardStreams::turn`],
//! so the stream's next number would never advance past it and every later
//! forward — a restart included, which only skips turns that ARRIVE — waited
//! [`STREAM_TURN_WAIT`] for it (the .215 release gate, 2026-10-01: a 60 s stall,
//! then a re-plan away from a healthy peer). [`ForwardStreams::refused_on_arrival`]
//! records the number, and the stream steps over it when it comes due — the
//! same as a forward that took its turn and failed inside the handler gives it
//! straight back. The coordinator was answered either way; it ends the stream
//! at that answer, or a restart it had already sent cuts the cache back past
//! the hole. Nothing the coordinator keeps is ever built on the missing rows.
//! TCP's receiver closes a hole by the sender's retransmission; nobody resends
//! a refused check, so the node that refused it closes the hole itself.

use std::collections::BTreeSet;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use dashmap::DashMap;

use crate::types::inference::stream_seq;

/// How long a streamed forward waits for the one before it. Far past any
/// chunk's own run — a stream keeps a few in flight, each a few positions —
/// so reaching it means the one before was lost or refused on arrival, and the
/// coordinator has already failed the stream.
pub(crate) const STREAM_TURN_WAIT: Duration = Duration::from_secs(60);

/// A stream nobody has touched for this long is forgotten. A conversation's
/// cache expires after 10 minutes idle, so a stream cannot outlive it usefully.
pub(crate) const FORWARD_STREAM_IDLE: Duration = Duration::from_secs(600);

/// The most chunks a coordinator keeps out at once (`SWARMLLM_SPEC_STREAM_WINDOW`
/// is clamped to it). A serving node admits a stream's chunks up to twice this —
/// a restart leaves the chunks it superseded queued here until each takes its
/// turn and is skipped — and refuses past it, so a sender cannot queue without
/// bound behind the one slot and permit its stream holds (`dispatch::StreamWorkSlot`).
pub(crate) const MAX_STREAM_WINDOW: u32 = 8;

/// Chunks of ONE stream this node holds at once (see [`MAX_STREAM_WINDOW`]).
pub(crate) const MAX_STREAM_CHUNKS_HERE: usize = 2 * MAX_STREAM_WINDOW as usize;

/// Turns a stream remembers as refused on arrival. Far past what a coordinator
/// can have out (it numbers at most [`MAX_STREAM_CHUNKS_HERE`] beyond what has
/// run); a sender past it is not one this has to serve well, and a hole it
/// leaves unrecorded costs only the old [`STREAM_TURN_WAIT`].
const MAX_REFUSED_TURNS: usize = 64;

type StreamKey = (uuid::Uuid, (u32, u32), u32);

/// Every streamed verify this node is serving, by (request, layer range,
/// attempt).
#[derive(Default)]
pub(crate) struct ForwardStreams {
    streams: DashMap<StreamKey, Arc<Stream>>,
}

struct Stream {
    /// The turn that may run next — advanced as each forward's turn ends.
    next: tokio::sync::watch::Sender<u32>,
    /// Turns below this were superseded by a restart and are skipped.
    skip_below: AtomicU32,
    /// Turns this node refused on arrival, not yet due: stepped over when
    /// `next` reaches them. Every advance of `next` happens under this lock, so
    /// a refusal recorded just as the turn before it ends is never missed.
    refused: parking_lot::Mutex<BTreeSet<u32>>,
    /// Epoch milliseconds of the last turn taken or ended, for the sweep.
    touched_ms: AtomicU64,
}

impl Stream {
    fn new() -> Self {
        let (next, _) = tokio::sync::watch::channel(0);
        let stream = Self {
            next,
            skip_below: AtomicU32::new(0),
            refused: parking_lot::Mutex::new(BTreeSet::new()),
            touched_ms: AtomicU64::new(0),
        };
        stream.touch();
        stream
    }

    fn touch(&self) {
        self.touched_ms.store(now_ms(), Ordering::Relaxed);
    }

    /// Let the stream reach at least `after`, stepping over every turn refused
    /// on arrival that then comes due.
    fn advance_to(&self, after: u32) {
        let mut refused = self.refused.lock();
        self.next.send_modify(|n| {
            let mut next = (*n).max(after);
            while refused.remove(&next) {
                next = next.saturating_add(1);
            }
            *n = next;
        });
    }

    fn idle_for(&self) -> Duration {
        Duration::from_millis(now_ms().saturating_sub(self.touched_ms.load(Ordering::Relaxed)))
    }
}

fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_millis() as u64)
}

/// A streamed forward's turn. While it is held no later forward of the stream
/// runs; dropping it — however the forward ended, an abort included — lets the
/// next one go.
pub(crate) struct Turn {
    stream: Arc<Stream>,
    turn: u32,
}

impl Drop for Turn {
    fn drop(&mut self) {
        self.stream.advance_to(self.turn.saturating_add(1));
        self.stream.touch();
    }
}

/// Why a streamed forward did not get its turn.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum TurnRefused {
    /// Its turn already ran — a duplicate copy.
    AlreadyRan { next: u32 },
    /// The forward before it never ended here within [`STREAM_TURN_WAIT`].
    Waited,
    /// A restart later in the stream superseded it before it ran.
    Skipped,
}

impl TurnRefused {
    /// What the coordinator is told.
    pub(crate) fn reason(&self, seq: u32) -> String {
        let turn = stream_seq::turn(seq);
        match self {
            TurnRefused::AlreadyRan { next } => format!(
                "streamed check {turn} arrived after check {next} had started — it already ran"
            ),
            TurnRefused::Waited => format!(
                "streamed check {turn} waited {}s for the check before it, which never came",
                STREAM_TURN_WAIT.as_secs()
            ),
            TurnRefused::Skipped => {
                format!("streamed check {turn} was skipped — a later check restarted the stream")
            }
        }
    }
}

impl ForwardStreams {
    /// Wait until forward `seq` (`LayerForward::stream_seq`) may run — its
    /// attempt's stream has reached its turn — at most `wait`. `restarts`: the
    /// forward cuts the cache back, so the turns before it that have not run
    /// are skipped (module doc).
    pub(crate) async fn turn(
        &self,
        request_id: uuid::Uuid,
        layer_range: (u32, u32),
        seq: u32,
        restarts: bool,
        wait: Duration,
    ) -> Result<Turn, TurnRefused> {
        let turn = stream_seq::turn(seq);
        // The entry's guard is dropped at the end of this statement, before
        // any await (`clippy.toml`: a DashMap guard never lives across one).
        let stream = self
            .streams
            .entry((request_id, layer_range, stream_seq::attempt(seq)))
            .or_insert_with(|| Arc::new(Stream::new()))
            .clone();
        stream.touch();
        if restarts {
            stream.skip_below.fetch_max(turn, Ordering::AcqRel);
        }
        let mut rx = stream.next.subscribe();
        let reached = tokio::time::timeout(wait, async {
            rx.wait_for(|&n| n >= turn).await.map(|n| *n)
        })
        .await;
        match reached {
            Ok(Ok(n)) if n == turn => {
                let superseded = turn < stream.skip_below.load(Ordering::Acquire);
                let held = Turn { stream, turn };
                if superseded {
                    // Given straight back: the next turn goes at once.
                    drop(held);
                    return Err(TurnRefused::Skipped);
                }
                Ok(held)
            }
            Ok(Ok(n)) => Err(TurnRefused::AlreadyRan { next: n }),
            // The sender lives in `stream`, which this holds, so the channel
            // cannot close; a timeout is the only way here.
            Ok(Err(_)) | Err(_) => Err(TurnRefused::Waited),
        }
    }

    /// Forward `seq` was refused on arrival — an admission cap answered it
    /// before it could wait for its turn — so its turn is stepped over when it
    /// comes due (module doc). `restarts`: it cut the cache back, so the turns
    /// before it that have not run are superseded as if it had arrived.
    pub(crate) fn refused_on_arrival(
        &self,
        request_id: uuid::Uuid,
        layer_range: (u32, u32),
        seq: u32,
        restarts: bool,
    ) {
        let turn = stream_seq::turn(seq);
        let stream = self
            .streams
            .entry((request_id, layer_range, stream_seq::attempt(seq)))
            .or_insert_with(|| Arc::new(Stream::new()))
            .clone();
        stream.touch();
        if restarts {
            stream.skip_below.fetch_max(turn, Ordering::AcqRel);
        }
        {
            let mut refused = stream.refused.lock();
            // Already past it (a second copy), or a sender far ahead of what
            // any coordinator keeps out: nothing to record.
            if turn < *stream.next.borrow() || refused.len() >= MAX_REFUSED_TURNS {
                return;
            }
            refused.insert(turn);
        }
        // Due now if the turn before it has already ended.
        stream.advance_to(0);
    }

    /// Forget streams idle for `idle` that no forward is waiting in or
    /// running; how many were forgotten.
    pub(crate) fn sweep(&self, idle: Duration) -> usize {
        let before = self.streams.len();
        self.streams
            .retain(|_, s| Arc::strong_count(s) > 1 || s.idle_for() < idle);
        before - self.streams.len()
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.streams.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const RANGE: (u32, u32) = (14, 28);

    /// Chunks that arrive in the wrong order still run in the stream's order:
    /// 2 and 1 arrive first and wait; 0 runs, then 1, then 2.
    #[tokio::test]
    async fn forwards_run_in_their_streams_order_whatever_order_they_arrive_in() {
        let streams = Arc::new(ForwardStreams::default());
        let id = uuid::Uuid::new_v4();
        let ran = Arc::new(std::sync::Mutex::new(Vec::new()));
        let mut tasks = Vec::new();
        for seq in [2u32, 1, 0] {
            let (streams, ran) = (streams.clone(), ran.clone());
            tasks.push(tokio::spawn(async move {
                let turn = streams
                    .turn(id, RANGE, seq, false, STREAM_TURN_WAIT)
                    .await
                    .unwrap();
                ran.lock().unwrap().push(seq);
                tokio::task::yield_now().await;
                drop(turn);
            }));
            tokio::task::yield_now().await;
        }
        for t in tasks {
            t.await.unwrap();
        }
        assert_eq!(*ran.lock().unwrap(), vec![0, 1, 2]);
    }

    /// A turn that already ran is refused, never run a second time.
    #[tokio::test]
    async fn a_turn_that_already_ran_is_refused() {
        let streams = ForwardStreams::default();
        let id = uuid::Uuid::new_v4();
        drop(
            streams
                .turn(id, RANGE, 0, false, STREAM_TURN_WAIT)
                .await
                .unwrap(),
        );
        drop(
            streams
                .turn(id, RANGE, 1, false, STREAM_TURN_WAIT)
                .await
                .unwrap(),
        );
        assert_eq!(
            streams
                .turn(id, RANGE, 0, false, STREAM_TURN_WAIT)
                .await
                .err(),
            Some(TurnRefused::AlreadyRan { next: 2 })
        );
    }

    /// A gap that never fills ends the wait instead of holding the forward.
    #[tokio::test(start_paused = true)]
    async fn a_forward_whose_predecessor_never_comes_stops_waiting() {
        let streams = ForwardStreams::default();
        let id = uuid::Uuid::new_v4();
        assert_eq!(
            streams
                .turn(id, RANGE, 1, false, STREAM_TURN_WAIT)
                .await
                .err(),
            Some(TurnRefused::Waited)
        );
    }

    /// A retry under the same request id is its own stream: it starts at
    /// turn 0 while the dead attempt's chunk still holds its turn (#749).
    #[tokio::test]
    async fn each_attempt_is_its_own_stream() {
        let streams = ForwardStreams::default();
        let id = uuid::Uuid::new_v4();
        let seq = |attempt, turn| stream_seq::compose(attempt, turn).unwrap();
        drop(
            streams
                .turn(id, RANGE, seq(1, 0), false, STREAM_TURN_WAIT)
                .await
                .unwrap(),
        );
        let _dead = streams
            .turn(id, RANGE, seq(1, 1), false, STREAM_TURN_WAIT)
            .await
            .unwrap();
        assert!(
            streams
                .turn(id, RANGE, seq(2, 0), false, STREAM_TURN_WAIT)
                .await
                .is_ok(),
            "the retry's first chunk is not held behind the dead attempt's"
        );
    }

    /// A restart skips the turns before it that have not run — in order, and
    /// never beside the one that is running.
    #[tokio::test]
    async fn a_restart_skips_the_stale_turns_waiting_before_it() {
        let streams = Arc::new(ForwardStreams::default());
        let id = uuid::Uuid::new_v4();
        let running = streams
            .turn(id, RANGE, 0, false, STREAM_TURN_WAIT)
            .await
            .unwrap();
        // Turns 1 and 2 were built on a guess turn 0's check refused; 3 is the
        // restart, and arrives while 0 still runs and 1 waits.
        let stale = {
            let streams = streams.clone();
            tokio::spawn(async move { streams.turn(id, RANGE, 1, false, STREAM_TURN_WAIT).await })
        };
        tokio::task::yield_now().await;
        let restart = {
            let streams = streams.clone();
            tokio::spawn(async move { streams.turn(id, RANGE, 3, true, STREAM_TURN_WAIT).await })
        };
        tokio::task::yield_now().await;
        assert!(!restart.is_finished(), "never beside the running turn");
        drop(running);
        assert_eq!(stale.await.unwrap().err(), Some(TurnRefused::Skipped));
        // Turn 2 arrives late: skipped too, which lets the restart go.
        assert_eq!(
            streams
                .turn(id, RANGE, 2, false, STREAM_TURN_WAIT)
                .await
                .err(),
            Some(TurnRefused::Skipped)
        );
        assert!(restart.await.unwrap().is_ok(), "the restart runs");
        assert!(
            streams
                .turn(id, RANGE, 4, false, STREAM_TURN_WAIT)
                .await
                .is_ok(),
            "and what follows it"
        );
    }

    /// The .215 gate's stall: check 20 refused on arrival, then the restart
    /// at 21 and what follows it. Before the fix the restart waited the full
    /// [`STREAM_TURN_WAIT`] for a turn that could never come.
    #[tokio::test(start_paused = true)]
    async fn a_restart_does_not_wait_for_a_turn_refused_on_arrival() {
        let streams = Arc::new(ForwardStreams::default());
        let id = uuid::Uuid::new_v4();
        for seq in 0..20 {
            drop(
                streams
                    .turn(id, RANGE, seq, false, STREAM_TURN_WAIT)
                    .await
                    .unwrap(),
            );
        }
        streams.refused_on_arrival(id, RANGE, 20, false);
        let started = tokio::time::Instant::now();
        let restart = streams.turn(id, RANGE, 21, true, STREAM_TURN_WAIT).await;
        assert!(restart.is_ok(), "the restart runs: {:?}", restart.err());
        assert!(
            started.elapsed() < Duration::from_secs(1),
            "without waiting"
        );
        drop(restart);
        assert!(streams
            .turn(id, RANGE, 22, false, STREAM_TURN_WAIT)
            .await
            .is_ok());
    }

    /// Refused while the turn before it is still running: the stream steps
    /// over it the moment that one ends, and the forward waiting behind the
    /// hole goes at once.
    #[tokio::test(start_paused = true)]
    async fn a_turn_refused_on_arrival_is_stepped_over_when_it_comes_due() {
        let streams = Arc::new(ForwardStreams::default());
        let id = uuid::Uuid::new_v4();
        let running = streams
            .turn(id, RANGE, 0, false, STREAM_TURN_WAIT)
            .await
            .unwrap();
        let behind = {
            let streams = streams.clone();
            tokio::spawn(async move { streams.turn(id, RANGE, 2, false, STREAM_TURN_WAIT).await })
        };
        tokio::task::yield_now().await;
        streams.refused_on_arrival(id, RANGE, 1, false);
        tokio::task::yield_now().await;
        assert!(!behind.is_finished(), "never beside the running turn");
        let started = tokio::time::Instant::now();
        drop(running);
        assert!(behind.await.unwrap().is_ok());
        assert!(started.elapsed() < Duration::from_secs(1));
    }

    /// A refused turn that is already due, with nothing running before it,
    /// is stepped over at once; several in a row are stepped over together.
    #[tokio::test(start_paused = true)]
    async fn consecutive_refused_turns_are_all_stepped_over() {
        let streams = ForwardStreams::default();
        let id = uuid::Uuid::new_v4();
        streams.refused_on_arrival(id, RANGE, 0, false);
        streams.refused_on_arrival(id, RANGE, 2, false);
        streams.refused_on_arrival(id, RANGE, 1, false);
        let started = tokio::time::Instant::now();
        assert!(streams
            .turn(id, RANGE, 3, false, STREAM_TURN_WAIT)
            .await
            .is_ok());
        assert!(started.elapsed() < Duration::from_secs(1));
    }

    /// A refused RESTART still supersedes the turns before it that have not
    /// run — the coordinator dropped them when it sent it.
    #[tokio::test]
    async fn a_refused_restart_still_supersedes_the_turns_before_it() {
        let streams = Arc::new(ForwardStreams::default());
        let id = uuid::Uuid::new_v4();
        let running = streams
            .turn(id, RANGE, 0, false, STREAM_TURN_WAIT)
            .await
            .unwrap();
        let stale = {
            let streams = streams.clone();
            tokio::spawn(async move { streams.turn(id, RANGE, 1, false, STREAM_TURN_WAIT).await })
        };
        tokio::task::yield_now().await;
        streams.refused_on_arrival(id, RANGE, 2, true);
        drop(running);
        assert_eq!(stale.await.unwrap().err(), Some(TurnRefused::Skipped));
        assert!(streams
            .turn(id, RANGE, 3, false, STREAM_TURN_WAIT)
            .await
            .is_ok());
    }

    /// A copy of a turn that already ran, refused on arrival, changes nothing.
    #[tokio::test]
    async fn a_refused_copy_of_a_turn_that_ran_changes_nothing() {
        let streams = ForwardStreams::default();
        let id = uuid::Uuid::new_v4();
        drop(
            streams
                .turn(id, RANGE, 0, false, STREAM_TURN_WAIT)
                .await
                .unwrap(),
        );
        streams.refused_on_arrival(id, RANGE, 0, false);
        assert!(streams
            .turn(id, RANGE, 1, false, STREAM_TURN_WAIT)
            .await
            .is_ok());
    }

    /// Two segments of one request are two streams.
    #[tokio::test]
    async fn each_layer_range_is_its_own_stream() {
        let streams = ForwardStreams::default();
        let id = uuid::Uuid::new_v4();
        let _head = streams
            .turn(id, (0, 14), 0, false, STREAM_TURN_WAIT)
            .await
            .unwrap();
        assert!(streams
            .turn(id, RANGE, 0, false, STREAM_TURN_WAIT)
            .await
            .is_ok());
    }

    /// The sweep forgets an idle stream but never one a forward is in.
    #[tokio::test]
    async fn the_sweep_keeps_a_stream_a_forward_is_in() {
        let streams = ForwardStreams::default();
        let (busy, idle) = (uuid::Uuid::new_v4(), uuid::Uuid::new_v4());
        let _turn = streams
            .turn(busy, RANGE, 0, false, STREAM_TURN_WAIT)
            .await
            .unwrap();
        drop(
            streams
                .turn(idle, RANGE, 0, false, STREAM_TURN_WAIT)
                .await
                .unwrap(),
        );
        assert_eq!(streams.sweep(Duration::ZERO), 1);
        assert_eq!(streams.len(), 1);
    }
}
