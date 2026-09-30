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
    /// Epoch milliseconds of the last turn taken or ended, for the sweep.
    touched_ms: AtomicU64,
}

impl Stream {
    fn new() -> Self {
        let (next, _) = tokio::sync::watch::channel(0);
        let stream = Self {
            next,
            skip_below: AtomicU32::new(0),
            touched_ms: AtomicU64::new(0),
        };
        stream.touch();
        stream
    }

    fn touch(&self) {
        self.touched_ms.store(now_ms(), Ordering::Relaxed);
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
        let after = self.turn.saturating_add(1);
        self.stream.next.send_modify(|n| *n = (*n).max(after));
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
