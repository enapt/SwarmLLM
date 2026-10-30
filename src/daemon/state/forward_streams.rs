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
//! thrown away. Those are skipped, and the restart goes the moment nothing
//! runs — never beside the turn running before it (the worker routes replies
//! by request id, gotcha #180), and never waiting for a superseded turn to
//! ARRIVE. In a chain the hop before this one may have skipped it already and
//! will never hand it on (#318); a skipped turn that turns up later is
//! answered unrun. Safe because every turn the restart builds on ran here: the
//! coordinator sends a restart only after the check it follows was answered,
//! and that check ran on every hop on its way to the tail. PipeInfer's early
//! inference cancellation, done without a message of its own: the restart is
//! the signal. It matters where this node is busy — the work skipped is work
//! the restart would otherwise wait behind.
//!
//! **A refusal walked here skips what stood on it (2026-10-26).** Where this
//! node walks the stream's checks (`spec_walk_at_tail`), it knows a check
//! refused a guess the moment its forward ends — a round trip before the
//! coordinator's restart can say so. Every later turn that does not cut the
//! cache back was built on that guess (after a refused check the coordinator's
//! next chunk always restarts: `dsd_stream`'s `take_answer`), so each is
//! skipped as it comes due ([`Turn::refused_a_guess`]); the restart clears the
//! mark when it runs, and is never skipped by it. The 70B's far end (request
//! 344f21c8, 2026-10-26) took 1.14 s a check and the restart reached it 1.0 s
//! after the refusal, so each miss ran one stale check: 10 of its 29 decode
//! forwards. The saving is the far end's check time less the restart's latency,
//! and the far end's time for other requests. A wrong LOOK-AHEAD (every guess
//! kept, the far end's own next token different) is the same refusal where
//! the coordinator names the look-ahead (`LayerForward::stream_lookahead`,
//! `features::STREAM_LOOKAHEAD`, #382: to a lone far segment that walks); an
//! older coordinator names none, and that quarter of the restarts then waits
//! for it. A/B `SWARMLLM_STREAM_TAIL_SKIP=0` (here),
//! `SWARMLLM_STREAM_LOOKAHEAD=0` (the coordinator).
//!
//! **A spare restart runs the moment its refusal happens (#381).** Skipping
//! stops this node computing stale checks; it cannot start the right one,
//! which must first pass the coordinator's layers. So the coordinator sends,
//! beside a check, the restart it would send if that check refused one
//! particular guess for one particular token (`LayerForward::stream_branch`,
//! `StreamBranch::Spare`). It waits here in its turn like any check and, when
//! it comes due, runs only where the walk it names refused exactly that guess
//! for exactly that token ([`TurnKind::Spare`]) — cutting the cache back as a
//! restart would — and is skipped unrun everywhere else. It supersedes
//! nothing on arrival (the check it waits on may still be queued), and
//! running it leaves the refusal standing, so the stale turns built on the
//! refused guess are still skipped. The coordinator reads the same walk and
//! knows the spare ran; the checks it then builds on the spare name it
//! ([`TurnKind::AfterSpare`]) — they run only where it did, are refused out
//! loud where it did not (a spare refused on arrival, say), and are skipped
//! where the spare itself refused a guess. The 70B's far end idled ~0.7 s a
//! refused guess, waiting for the restart (`docs/FUTURE_WORK.md` #381).
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

/// How long a PIECE of a prompt pass (`LayerForward::prompt_span`, #171) waits
/// for the piece before it. [`STREAM_TURN_WAIT`] is sized for a check of a few
/// positions; a piece is hundreds, and on a slow processor the one before it
/// can run for minutes — refused at 60 s, every such pass failed and was read
/// again whole (review of #171, 2026-10-10). The coordinator's own cap on one
/// warm forward (`pipeline::SEGMENT_TIMEOUT_MAX_SECS`) bounds it: a piece is
/// only queued here behind one the coordinator is still waiting for, and the
/// first piece — the one that may load the model — is never queued behind.
pub(crate) const PIECE_TURN_WAIT: Duration =
    Duration::from_secs(crate::inference::pipeline::SEGMENT_TIMEOUT_MAX_SECS);

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
    /// What decides how far `next` may move, and which turn may run when it
    /// comes due. Every change to `next` happens under this lock, so a
    /// refusal recorded, or a restart arriving, just as the turn before it
    /// ends is never missed.
    order: parking_lot::Mutex<Order>,
    /// Epoch milliseconds of the last turn taken or ended, for the sweep.
    touched_ms: AtomicU64,
}

#[derive(Default)]
struct Order {
    /// Turns this node refused on arrival, not yet due: stepped over when
    /// `next` reaches them.
    refused: BTreeSet<u32>,
    /// A turn is held — its forward is at the worker. While it is, `next`
    /// names it and nothing may jump past it.
    running: bool,
    /// The last check walked HERE that refused a guess, since a restart last
    /// ran. Every later turn that does not cut the cache back was built on
    /// that guess and is skipped (module doc) — and a spare restart for
    /// exactly this refusal runs.
    refusal: Option<Refusal>,
    /// The spare restart that ran here for `refusal`, if one did: the turn a
    /// check built on it names.
    taken: Option<TakenSpare>,
}

/// A check walked here that refused a guess: its turn, the guesses it kept
/// and the token it drew in the refused one's place.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Refusal {
    turn: u32,
    kept: u32,
    token: u32,
}

/// A spare restart that ran here, and the check whose refusal it restarted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct TakenSpare {
    spare: u32,
    of: u32,
}

/// What a streamed forward is to the turns before it — read off the forward
/// by [`TurnKind::of`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TurnKind {
    /// It builds on the turn before it: skipped where a check walked here
    /// since the last restart refused a guess.
    Follows,
    /// It cuts the cache back (`truncate_kv_to`): it restarts the stream, and
    /// the turns before it that have not run are superseded.
    Restarts,
    /// A spare restart (#381): run only where turn `of`'s walk here kept
    /// `kept` guesses and drew `token` — the restart its coordinator would
    /// otherwise send a round trip later — and skipped unrun everywhere else.
    /// It cuts the cache back when it runs, but supersedes nothing on
    /// arrival: the check it waits on may not have run yet.
    Spare { of: u32, kept: u32, token: u32 },
    /// It builds on spare `spare`'s rows: refused out loud unless that spare
    /// ran here, skipped where the line through it has since been refused.
    AfterSpare { spare: u32 },
}

impl TurnKind {
    /// What forward `seq` (`LayerForward::stream_seq`) is, from its cut and
    /// its branch (`LayerForward::stream_branch`). Refused where the branch
    /// cannot mean what it says: either cuts the cache back to its own first
    /// row (`truncate_kv_to` = `index_pos` — a spare as the restart it is, a
    /// check built on one to drop anything run after the spare), and either
    /// names an EARLIER turn of its own attempt.
    pub(crate) fn of(
        seq: u32,
        index_pos: u32,
        truncate_kv_to: Option<u32>,
        branch: Option<crate::types::StreamBranch>,
    ) -> Result<Self, &'static str> {
        use crate::types::StreamBranch;
        let earlier = |named: u32| {
            (stream_seq::attempt(named) == stream_seq::attempt(seq)
                && stream_seq::turn(named) < stream_seq::turn(seq))
            .then_some(stream_seq::turn(named))
        };
        match branch {
            None if truncate_kv_to.is_some() => Ok(TurnKind::Restarts),
            None => Ok(TurnKind::Follows),
            Some(StreamBranch::Spare { of, kept, token }) => {
                if truncate_kv_to != Some(index_pos) {
                    return Err("a spare restart must cut the cache back to its own first row");
                }
                let of = earlier(of)
                    .ok_or("a spare restart must wait on an earlier check of its own stream")?;
                Ok(TurnKind::Spare { of, kept, token })
            }
            Some(StreamBranch::AfterSpare { spare }) => {
                if truncate_kv_to != Some(index_pos) {
                    return Err(
                        "a check built on a spare must cut the cache back to its own first row",
                    );
                }
                let spare = earlier(spare)
                    .ok_or("a check must be built on an earlier spare of its own stream")?;
                Ok(TurnKind::AfterSpare { spare })
            }
        }
    }

    /// It supersedes, on arrival, every turn before it that has not run.
    fn supersedes(self) -> bool {
        self == TurnKind::Restarts
    }
}

impl Order {
    /// The refusal a check that follows the line stood on, if turn `turn` must
    /// be skipped for it: with `tail_skip` (the A/B switch,
    /// `skip_on_a_refusal_here`), any refusal before it; without, only one a
    /// spare has run for. That skip is not an A/B choice: the spare cut the
    /// cache back and wrote its own rows, which a stale turn run now would
    /// overwrite under the checks built on them.
    fn stood_on(&self, turn: u32, tail_skip: bool) -> Option<Refusal> {
        self.refusal
            .filter(|r| turn > r.turn && (tail_skip || self.taken.is_some_and(|t| t.of == r.turn)))
    }

    /// Whether turn `turn`, of `kind`, runs now that it has come due with
    /// nothing running — recording what running it changes — or why not.
    fn admit(&mut self, turn: u32, kind: TurnKind, tail_skip: bool) -> Result<(), TurnRefused> {
        match kind {
            TurnKind::Follows => {
                if let Some(r) = self.stood_on(turn, tail_skip) {
                    return Err(TurnRefused::StoodOnARefusal { refused_at: r.turn });
                }
            }
            TurnKind::Restarts => {
                // What a refusal here left stale is behind it now.
                self.refusal = None;
                self.taken = None;
            }
            TurnKind::Spare { of, kept, token } => {
                let wanted = self.refusal
                    == Some(Refusal {
                        turn: of,
                        kept,
                        token,
                    });
                // One restart per refusal: a second spare for the same one is
                // a coordinator's mistake, and the first already cut the cache.
                let one_ran = self.taken.is_some_and(|t| t.of == of);
                if !wanted || one_ran {
                    return Err(TurnRefused::SpareNotNeeded { of });
                }
                // The refusal stands: every turn after it that is not built
                // on this spare stood on the refused guess.
                self.taken = Some(TakenSpare { spare: turn, of });
            }
            TurnKind::AfterSpare { spare } => {
                // A refusal at or after the spare ended the line through it:
                // the coordinator restarts after that answer.
                if let Some(r) = self.refusal.filter(|r| r.turn >= spare) {
                    return Err(TurnRefused::StoodOnARefusal { refused_at: r.turn });
                }
                if self.taken.map(|t| t.spare) != Some(spare) {
                    return Err(TurnRefused::SpareMissing { spare });
                }
                // Built on what the spare kept: the refusal it restarted is
                // behind it now.
                self.refusal = None;
            }
        }
        Ok(())
    }
}

impl Stream {
    fn new() -> Self {
        let (next, _) = tokio::sync::watch::channel(0);
        let stream = Self {
            next,
            skip_below: AtomicU32::new(0),
            order: parking_lot::Mutex::new(Order::default()),
            touched_ms: AtomicU64::new(0),
        };
        stream.touch();
        stream
    }

    fn touch(&self) {
        self.touched_ms.store(now_ms(), Ordering::Relaxed);
    }

    /// Let the stream reach at least `after`, stepping over every turn refused
    /// on arrival that then comes due and — while nothing runs — every turn a
    /// restart superseded, whether or not it has arrived.
    fn advance_to(&self, after: u32) {
        let mut order = self.order.lock();
        self.advance_locked(&mut order, after);
    }

    fn advance_locked(&self, order: &mut Order, after: u32) {
        let skip_below = self.skip_below.load(Ordering::Acquire);
        self.next.send_modify(|n| {
            let mut next = (*n).max(after);
            loop {
                if order.refused.remove(&next) {
                    next = next.saturating_add(1);
                } else if !order.running && next < skip_below {
                    next = skip_below;
                } else {
                    break;
                }
            }
            *n = next;
        });
        // A jump to `skip_below` passes refused turns it never stepped onto.
        // Kept, they filled `MAX_REFUSED_TURNS`, and a later refusal went
        // unrecorded: the forward behind it waited out the gap (pre-release
        // review, 2026-10-20). A record below `next` is never read again.
        let next = *self.next.borrow();
        order.refused = order.refused.split_off(&next);
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

impl Turn {
    /// This turn's check, walked here, kept `kept` guesses and drew `token`
    /// in place of the next: every later turn that does not restart the
    /// stream stood on that guess and is skipped when it comes due, without
    /// waiting for the restart to say so, and a spare restart for exactly
    /// this refusal runs (module doc). Called before the turn ends, so the
    /// next turn sees it.
    pub(crate) fn refused_a_guess(&self, kept: u32, token: u32) {
        // Recorded whatever the A/B switch says: a spare restart's coordinator
        // reads the same walk and builds on the spare as having run.
        self.stream.order.lock().refusal = Some(Refusal {
            turn: self.turn,
            kept,
            token,
        });
    }
}

/// `SWARMLLM_STREAM_TAIL_SKIP=0` keeps the turns after a refusal running until
/// the coordinator's restart arrives — the behaviour before 2026-10-26, for
/// A/B inside one binary.
fn skip_on_a_refusal_here() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        !matches!(
            std::env::var("SWARMLLM_STREAM_TAIL_SKIP").ok().as_deref(),
            Some("0")
        )
    })
}

impl Drop for Turn {
    fn drop(&mut self) {
        {
            let mut order = self.stream.order.lock();
            order.running = false;
            self.stream
                .advance_locked(&mut order, self.turn.saturating_add(1));
        }
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
    /// A check before it, walked here, refused a guess it was built on.
    StoodOnARefusal { refused_at: u32 },
    /// A spare restart for a refusal that did not happen here (#381) — the
    /// check it waited on kept its guesses, or refused another — or for one a
    /// spare already restarted.
    SpareNotNeeded { of: u32 },
    /// Built on a spare restart that did not run here. Not a skip: its
    /// coordinator believed the spare ran, and is told it did not.
    SpareMissing { spare: u32 },
}

impl TurnRefused {
    /// The turn was skipped by the stream's early cancellation working — a
    /// restart or a refusal — not lost: answered like a failure, so the
    /// coordinator's wait ends, but not logged as one.
    pub(crate) fn is_a_skip(&self) -> bool {
        matches!(
            self,
            TurnRefused::Skipped
                | TurnRefused::StoodOnARefusal { .. }
                | TurnRefused::SpareNotNeeded { .. }
        )
    }
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
            TurnRefused::StoodOnARefusal { refused_at } => format!(
                "streamed check {turn} was skipped — check {refused_at} refused a guess it was built on"
            ),
            TurnRefused::SpareNotNeeded { of } => format!(
                "streamed check {turn} was skipped — check {of} did not refuse the guess it is the restart for"
            ),
            TurnRefused::SpareMissing { spare } => format!(
                "streamed check {turn} is built on spare restart {spare}, which did not run here"
            ),
        }
    }
}

impl ForwardStreams {
    /// Wait until forward `seq` (`LayerForward::stream_seq`) may run — its
    /// attempt's stream has reached its turn — at most `wait`. `kind`: what it
    /// is to the turns before it ([`TurnKind::of`]); a restart supersedes the
    /// turns before it that have not run, and a spare runs only for the
    /// refusal it names (module doc).
    pub(crate) async fn turn(
        &self,
        request_id: uuid::Uuid,
        layer_range: (u32, u32),
        seq: u32,
        kind: TurnKind,
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
        if kind.supersedes() {
            stream.skip_below.fetch_max(turn, Ordering::AcqRel);
            // Past the superseded turns at once if nothing runs; otherwise
            // the running turn's end does it.
            stream.advance_to(0);
        }
        let mut rx = stream.next.subscribe();
        let reached = tokio::time::timeout(wait, async {
            rx.wait_for(|&n| n >= turn).await.map(|n| *n)
        })
        .await;
        match reached {
            // Claimed under the lock: the stream may have moved past this turn
            // since the wait saw it (a restart arriving), and a second copy of
            // the turn must not run beside the first.
            Ok(Ok(_)) => {
                let mut order = stream.order.lock();
                let next = *stream.next.borrow();
                let superseded = turn < stream.skip_below.load(Ordering::Acquire);
                if next == turn && !order.running {
                    let admitted = if superseded {
                        Err(TurnRefused::Skipped)
                    } else {
                        order.admit(turn, kind, skip_on_a_refusal_here())
                    };
                    if let Err(refused) = admitted {
                        // Due, with nothing running: stepped over now, which
                        // lets the restart go.
                        stream.advance_locked(&mut order, turn.saturating_add(1));
                        return Err(refused);
                    }
                    order.running = true;
                    drop(order);
                    return Ok(Turn { stream, turn });
                }
                // Passed: superseded, stood on a refusal, or a second copy.
                if superseded {
                    Err(TurnRefused::Skipped)
                } else if let Some(r) = (kind == TurnKind::Follows)
                    .then(|| order.stood_on(turn, skip_on_a_refusal_here()))
                    .flatten()
                {
                    Err(TurnRefused::StoodOnARefusal { refused_at: r.turn })
                } else {
                    Err(TurnRefused::AlreadyRan { next })
                }
            }
            // The sender lives in `stream`, which this holds, so the channel
            // cannot close; a timeout is the only way here.
            Ok(Err(_)) | Err(_) => Err(TurnRefused::Waited),
        }
    }

    /// Forward `seq` was refused on arrival — an admission cap answered it
    /// before it could wait for its turn — so its turn is stepped over when it
    /// comes due (module doc). A restart (`kind`) still supersedes the turns
    /// before it that have not run, as if it had arrived; a spare stepped over
    /// never ran, so a check built on it is refused.
    pub(crate) fn refused_on_arrival(
        &self,
        request_id: uuid::Uuid,
        layer_range: (u32, u32),
        seq: u32,
        kind: TurnKind,
    ) {
        let turn = stream_seq::turn(seq);
        let stream = self
            .streams
            .entry((request_id, layer_range, stream_seq::attempt(seq)))
            .or_insert_with(|| Arc::new(Stream::new()))
            .clone();
        stream.touch();
        if kind.supersedes() {
            stream.skip_below.fetch_max(turn, Ordering::AcqRel);
        }
        let mut order = stream.order.lock();
        let next = *stream.next.borrow();
        // Already past it, a copy of the turn running now (recording it would
        // step `next` past a turn still at the worker and refuse the honest
        // one after it), or a sender far ahead of what any coordinator keeps
        // out: nothing to record — but a restart refused here still lets the
        // stream past what it superseded.
        let copy_of_the_running_turn = turn == next && order.running;
        if turn >= next && !copy_of_the_running_turn && order.refused.len() < MAX_REFUSED_TURNS {
            order.refused.insert(turn);
        }
        // Due now if the turn before it has already ended.
        stream.advance_locked(&mut order, 0);
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
                    .turn(id, RANGE, seq, TurnKind::Follows, STREAM_TURN_WAIT)
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
                .turn(id, RANGE, 0, TurnKind::Follows, STREAM_TURN_WAIT)
                .await
                .unwrap(),
        );
        drop(
            streams
                .turn(id, RANGE, 1, TurnKind::Follows, STREAM_TURN_WAIT)
                .await
                .unwrap(),
        );
        assert_eq!(
            streams
                .turn(id, RANGE, 0, TurnKind::Follows, STREAM_TURN_WAIT)
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
                .turn(id, RANGE, 1, TurnKind::Follows, STREAM_TURN_WAIT)
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
                .turn(id, RANGE, seq(1, 0), TurnKind::Follows, STREAM_TURN_WAIT)
                .await
                .unwrap(),
        );
        let _dead = streams
            .turn(id, RANGE, seq(1, 1), TurnKind::Follows, STREAM_TURN_WAIT)
            .await
            .unwrap();
        assert!(
            streams
                .turn(id, RANGE, seq(2, 0), TurnKind::Follows, STREAM_TURN_WAIT)
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
            .turn(id, RANGE, 0, TurnKind::Follows, STREAM_TURN_WAIT)
            .await
            .unwrap();
        // Turns 1 and 2 were built on a guess turn 0's check refused; 3 is the
        // restart, and arrives while 0 still runs and 1 waits.
        let stale = {
            let streams = streams.clone();
            tokio::spawn(async move {
                streams
                    .turn(id, RANGE, 1, TurnKind::Follows, STREAM_TURN_WAIT)
                    .await
            })
        };
        tokio::task::yield_now().await;
        let restart = {
            let streams = streams.clone();
            tokio::spawn(async move {
                streams
                    .turn(id, RANGE, 3, TurnKind::Restarts, STREAM_TURN_WAIT)
                    .await
            })
        };
        tokio::task::yield_now().await;
        assert!(!restart.is_finished(), "never beside the running turn");
        drop(running);
        assert_eq!(stale.await.unwrap().err(), Some(TurnRefused::Skipped));
        let restart = restart.await.unwrap();
        assert!(restart.is_ok(), "the restart runs, turn 2 or no turn 2");
        // Turn 2 arrives late: answered unrun.
        assert_eq!(
            streams
                .turn(id, RANGE, 2, TurnKind::Follows, STREAM_TURN_WAIT)
                .await
                .err(),
            Some(TurnRefused::Skipped)
        );
        drop(restart);
        assert!(
            streams
                .turn(id, RANGE, 4, TurnKind::Follows, STREAM_TURN_WAIT)
                .await
                .is_ok(),
            "and what follows it"
        );
    }

    /// A check walked here that refused a guess: the turns built on it are
    /// skipped before the coordinator's restart arrives — the one waiting
    /// while the check ran, and one arriving after — and the restart, and
    /// what follows it, run. A walk that kept every guess skips nothing.
    #[tokio::test]
    async fn a_refusal_walked_here_skips_the_turns_built_on_it() {
        let streams = Arc::new(ForwardStreams::default());
        let id = uuid::Uuid::new_v4();
        let kept = streams
            .turn(id, RANGE, 0, TurnKind::Follows, STREAM_TURN_WAIT)
            .await
            .unwrap();
        drop(kept);
        let refusing = streams
            .turn(id, RANGE, 1, TurnKind::Follows, STREAM_TURN_WAIT)
            .await
            .expect("a walk that kept every guess skips nothing");
        let waiting = {
            let streams = streams.clone();
            tokio::spawn(async move {
                streams
                    .turn(id, RANGE, 2, TurnKind::Follows, STREAM_TURN_WAIT)
                    .await
            })
        };
        tokio::task::yield_now().await;
        refusing.refused_a_guess(0, 7);
        drop(refusing);
        assert_eq!(
            waiting.await.unwrap().err(),
            Some(TurnRefused::StoodOnARefusal { refused_at: 1 }),
            "waiting while the check ran"
        );
        assert_eq!(
            streams
                .turn(id, RANGE, 3, TurnKind::Follows, STREAM_TURN_WAIT)
                .await
                .err(),
            Some(TurnRefused::StoodOnARefusal { refused_at: 1 }),
            "arriving after it"
        );
        let restart = streams
            .turn(id, RANGE, 4, TurnKind::Restarts, STREAM_TURN_WAIT)
            .await
            .expect("the restart runs");
        drop(restart);
        assert!(
            streams
                .turn(id, RANGE, 5, TurnKind::Follows, STREAM_TURN_WAIT)
                .await
                .is_ok(),
            "and what follows it"
        );
    }

    /// #318: in a chain, the hop before this one skipped turns 1 and 2 when
    /// the restart reached it, so they never come here. The restart goes as
    /// soon as turn 0 ends — before the fix it waited [`STREAM_TURN_WAIT`] for
    /// them and was refused.
    #[tokio::test(start_paused = true)]
    async fn a_restart_does_not_wait_for_superseded_turns_that_never_come() {
        let streams = Arc::new(ForwardStreams::default());
        let id = uuid::Uuid::new_v4();
        let running = streams
            .turn(id, RANGE, 0, TurnKind::Follows, STREAM_TURN_WAIT)
            .await
            .unwrap();
        let restart = {
            let streams = streams.clone();
            tokio::spawn(async move {
                streams
                    .turn(id, RANGE, 3, TurnKind::Restarts, STREAM_TURN_WAIT)
                    .await
            })
        };
        tokio::task::yield_now().await;
        assert!(!restart.is_finished(), "never beside the running turn");
        let started = tokio::time::Instant::now();
        drop(running);
        let restart = restart.await.unwrap();
        assert!(restart.is_ok(), "the restart runs: {:?}", restart.err());
        assert!(
            started.elapsed() < Duration::from_secs(1),
            "without waiting for turns 1 and 2"
        );
        drop(restart);
        // And with nothing running when it arrives, it goes at once.
        let again = streams
            .turn(id, RANGE, 6, TurnKind::Restarts, STREAM_TURN_WAIT)
            .await;
        assert!(again.is_ok(), "{:?}", again.err());
        drop(again);
        assert!(streams
            .turn(id, RANGE, 7, TurnKind::Follows, STREAM_TURN_WAIT)
            .await
            .is_ok());
    }

    /// A second copy of a turn that is running never runs beside it.
    #[tokio::test(start_paused = true)]
    async fn a_second_copy_of_a_running_turn_does_not_run_beside_it() {
        let streams = ForwardStreams::default();
        let id = uuid::Uuid::new_v4();
        let _running = streams
            .turn(id, RANGE, 0, TurnKind::Follows, STREAM_TURN_WAIT)
            .await
            .unwrap();
        assert_eq!(
            streams
                .turn(id, RANGE, 0, TurnKind::Follows, STREAM_TURN_WAIT)
                .await
                .err(),
            Some(TurnRefused::AlreadyRan { next: 0 })
        );
    }

    /// A refused copy of the turn running now is not a gap to step over: the
    /// turn after it must still run once the running one ends (the 10-20
    /// security review's finding 1).
    #[tokio::test(start_paused = true)]
    async fn a_refused_copy_of_the_running_turn_does_not_refuse_the_next() {
        let streams = Arc::new(ForwardStreams::default());
        let id = uuid::Uuid::new_v4();
        let running = streams
            .turn(id, RANGE, 0, TurnKind::Follows, STREAM_TURN_WAIT)
            .await
            .unwrap();
        streams.refused_on_arrival(id, RANGE, 0, TurnKind::Follows);
        let waiting = {
            let streams = streams.clone();
            tokio::spawn(async move {
                streams
                    .turn(id, RANGE, 1, TurnKind::Follows, STREAM_TURN_WAIT)
                    .await
            })
        };
        tokio::task::yield_now().await;
        drop(running);
        let next = waiting.await.unwrap();
        assert!(next.is_ok(), "turn 1 runs: {:?}", next.err());
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
                    .turn(id, RANGE, seq, TurnKind::Follows, STREAM_TURN_WAIT)
                    .await
                    .unwrap(),
            );
        }
        streams.refused_on_arrival(id, RANGE, 20, TurnKind::Follows);
        let started = tokio::time::Instant::now();
        let restart = streams
            .turn(id, RANGE, 21, TurnKind::Restarts, STREAM_TURN_WAIT)
            .await;
        assert!(restart.is_ok(), "the restart runs: {:?}", restart.err());
        assert!(
            started.elapsed() < Duration::from_secs(1),
            "without waiting"
        );
        drop(restart);
        assert!(streams
            .turn(id, RANGE, 22, TurnKind::Follows, STREAM_TURN_WAIT)
            .await
            .is_ok());
    }

    /// A restart that jumps the stream past turns refused on arrival forgets
    /// them: kept, they filled [`MAX_REFUSED_TURNS`] and a later refusal went
    /// unrecorded, so the forward behind it waited out the gap (pre-release
    /// review, 2026-10-20).
    #[tokio::test(start_paused = true)]
    async fn a_restart_forgets_the_refusals_it_jumped_past() {
        let streams = Arc::new(ForwardStreams::default());
        let id = uuid::Uuid::new_v4();
        drop(
            streams
                .turn(id, RANGE, 0, TurnKind::Follows, STREAM_TURN_WAIT)
                .await
                .unwrap(),
        );
        for seq in 2..2 + MAX_REFUSED_TURNS as u32 {
            streams.refused_on_arrival(id, RANGE, seq, TurnKind::Follows);
        }
        drop(
            streams
                .turn(id, RANGE, 100, TurnKind::Restarts, STREAM_TURN_WAIT)
                .await
                .unwrap(),
        );
        streams.refused_on_arrival(id, RANGE, 101, TurnKind::Follows);
        let started = tokio::time::Instant::now();
        assert!(streams
            .turn(id, RANGE, 102, TurnKind::Follows, STREAM_TURN_WAIT)
            .await
            .is_ok());
        assert!(
            started.elapsed() < Duration::from_secs(1),
            "the refused turn is stepped over, not waited for"
        );
    }

    /// Refused while the turn before it is still running: the stream steps
    /// over it the moment that one ends, and the forward waiting behind the
    /// hole goes at once.
    #[tokio::test(start_paused = true)]
    async fn a_turn_refused_on_arrival_is_stepped_over_when_it_comes_due() {
        let streams = Arc::new(ForwardStreams::default());
        let id = uuid::Uuid::new_v4();
        let running = streams
            .turn(id, RANGE, 0, TurnKind::Follows, STREAM_TURN_WAIT)
            .await
            .unwrap();
        let behind = {
            let streams = streams.clone();
            tokio::spawn(async move {
                streams
                    .turn(id, RANGE, 2, TurnKind::Follows, STREAM_TURN_WAIT)
                    .await
            })
        };
        tokio::task::yield_now().await;
        streams.refused_on_arrival(id, RANGE, 1, TurnKind::Follows);
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
        streams.refused_on_arrival(id, RANGE, 0, TurnKind::Follows);
        streams.refused_on_arrival(id, RANGE, 2, TurnKind::Follows);
        streams.refused_on_arrival(id, RANGE, 1, TurnKind::Follows);
        let started = tokio::time::Instant::now();
        assert!(streams
            .turn(id, RANGE, 3, TurnKind::Follows, STREAM_TURN_WAIT)
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
            .turn(id, RANGE, 0, TurnKind::Follows, STREAM_TURN_WAIT)
            .await
            .unwrap();
        let stale = {
            let streams = streams.clone();
            tokio::spawn(async move {
                streams
                    .turn(id, RANGE, 1, TurnKind::Follows, STREAM_TURN_WAIT)
                    .await
            })
        };
        tokio::task::yield_now().await;
        streams.refused_on_arrival(id, RANGE, 2, TurnKind::Restarts);
        drop(running);
        assert_eq!(stale.await.unwrap().err(), Some(TurnRefused::Skipped));
        assert!(streams
            .turn(id, RANGE, 3, TurnKind::Follows, STREAM_TURN_WAIT)
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
                .turn(id, RANGE, 0, TurnKind::Follows, STREAM_TURN_WAIT)
                .await
                .unwrap(),
        );
        streams.refused_on_arrival(id, RANGE, 0, TurnKind::Follows);
        assert!(streams
            .turn(id, RANGE, 1, TurnKind::Follows, STREAM_TURN_WAIT)
            .await
            .is_ok());
    }

    /// Two segments of one request are two streams.
    #[tokio::test]
    async fn each_layer_range_is_its_own_stream() {
        let streams = ForwardStreams::default();
        let id = uuid::Uuid::new_v4();
        let _head = streams
            .turn(id, (0, 14), 0, TurnKind::Follows, STREAM_TURN_WAIT)
            .await
            .unwrap();
        assert!(streams
            .turn(id, RANGE, 0, TurnKind::Follows, STREAM_TURN_WAIT)
            .await
            .is_ok());
    }

    /// The sweep forgets an idle stream but never one a forward is in.
    #[tokio::test]
    async fn the_sweep_keeps_a_stream_a_forward_is_in() {
        let streams = ForwardStreams::default();
        let (busy, idle) = (uuid::Uuid::new_v4(), uuid::Uuid::new_v4());
        let _turn = streams
            .turn(busy, RANGE, 0, TurnKind::Follows, STREAM_TURN_WAIT)
            .await
            .unwrap();
        drop(
            streams
                .turn(idle, RANGE, 0, TurnKind::Follows, STREAM_TURN_WAIT)
                .await
                .unwrap(),
        );
        assert_eq!(streams.sweep(Duration::ZERO), 1);
        assert_eq!(streams.len(), 1);
    }
    fn spare(of: u32, kept: u32, token: u32) -> TurnKind {
        TurnKind::Spare { of, kept, token }
    }

    /// Take turn `turn` of a stream and end it at once.
    async fn run(streams: &ForwardStreams, id: uuid::Uuid, turn: u32, kind: TurnKind) {
        drop(
            streams
                .turn(id, RANGE, turn, kind, STREAM_TURN_WAIT)
                .await
                .unwrap_or_else(|e| panic!("turn {turn} ({kind:?}) refused: {e:?}")),
        );
    }

    /// #381: a spare restart runs where the check it waits on refused exactly
    /// its guess for exactly its token — the stale turn after it is still
    /// skipped, the check built on the spare runs, and so does what follows.
    #[tokio::test]
    async fn a_spare_restart_runs_for_the_refusal_it_names() {
        let streams = ForwardStreams::default();
        let id = uuid::Uuid::new_v4();
        let check = streams
            .turn(id, RANGE, 0, TurnKind::Follows, STREAM_TURN_WAIT)
            .await
            .unwrap();
        check.refused_a_guess(1, 42);
        drop(check);
        run(&streams, id, 1, spare(0, 1, 42)).await;
        assert_eq!(
            streams
                .turn(id, RANGE, 2, TurnKind::Follows, STREAM_TURN_WAIT)
                .await
                .err(),
            Some(TurnRefused::StoodOnARefusal { refused_at: 0 }),
            "the stale turn built on the refused guess, though the spare ran"
        );
        run(&streams, id, 3, TurnKind::AfterSpare { spare: 1 }).await;
        run(&streams, id, 4, TurnKind::Follows).await;
    }

    /// The control: a spare for any other outcome of the check it waits on is
    /// skipped unrun — another token, another guess refused, every guess kept
    /// — and the turn after a check that kept everything runs.
    #[tokio::test]
    async fn a_spare_restart_for_another_outcome_is_skipped() {
        for refusal in [Some((1, 43)), Some((2, 42)), Some((0, 42)), None] {
            let streams = ForwardStreams::default();
            let id = uuid::Uuid::new_v4();
            let check = streams
                .turn(id, RANGE, 0, TurnKind::Follows, STREAM_TURN_WAIT)
                .await
                .unwrap();
            if let Some((kept, token)) = refusal {
                check.refused_a_guess(kept, token);
            }
            drop(check);
            assert_eq!(
                streams
                    .turn(id, RANGE, 1, spare(0, 1, 42), STREAM_TURN_WAIT)
                    .await
                    .err(),
                Some(TurnRefused::SpareNotNeeded { of: 0 }),
                "check 0 refused {refusal:?}"
            );
            if refusal.is_none() {
                run(&streams, id, 2, TurnKind::Follows).await;
            }
        }
    }

    /// A check built on a spare that did not run here is refused out loud —
    /// never run on rows it was not built on: the spare was not needed, or it
    /// was refused on arrival.
    #[tokio::test]
    async fn a_check_built_on_a_spare_that_did_not_run_is_refused() {
        let streams = ForwardStreams::default();
        let id = uuid::Uuid::new_v4();
        run(&streams, id, 0, TurnKind::Follows).await;
        assert!(streams
            .turn(id, RANGE, 1, spare(0, 1, 42), STREAM_TURN_WAIT)
            .await
            .is_err());
        assert_eq!(
            streams
                .turn(
                    id,
                    RANGE,
                    2,
                    TurnKind::AfterSpare { spare: 1 },
                    STREAM_TURN_WAIT
                )
                .await
                .err(),
            Some(TurnRefused::SpareMissing { spare: 1 }),
            "the spare was not needed"
        );

        let id = uuid::Uuid::new_v4();
        let check = streams
            .turn(id, RANGE, 0, TurnKind::Follows, STREAM_TURN_WAIT)
            .await
            .unwrap();
        check.refused_a_guess(1, 42);
        drop(check);
        streams.refused_on_arrival(id, RANGE, 1, spare(0, 1, 42));
        assert_eq!(
            streams
                .turn(
                    id,
                    RANGE,
                    2,
                    TurnKind::AfterSpare { spare: 1 },
                    STREAM_TURN_WAIT
                )
                .await
                .err(),
            Some(TurnRefused::SpareMissing { spare: 1 }),
            "the spare was refused on arrival"
        );
        assert!(!TurnRefused::SpareMissing { spare: 1 }.is_a_skip());
        assert!(TurnRefused::SpareNotNeeded { of: 0 }.is_a_skip());
    }

    /// A spare that refused a guess itself ends the line through it: the
    /// check built on it is skipped, and the restart after the spare runs.
    #[tokio::test]
    async fn a_check_built_on_a_spare_that_refused_a_guess_is_skipped() {
        let streams = ForwardStreams::default();
        let id = uuid::Uuid::new_v4();
        let check = streams
            .turn(id, RANGE, 0, TurnKind::Follows, STREAM_TURN_WAIT)
            .await
            .unwrap();
        check.refused_a_guess(1, 42);
        drop(check);
        let spare_turn = streams
            .turn(id, RANGE, 1, spare(0, 1, 42), STREAM_TURN_WAIT)
            .await
            .unwrap();
        spare_turn.refused_a_guess(0, 9);
        drop(spare_turn);
        assert_eq!(
            streams
                .turn(
                    id,
                    RANGE,
                    2,
                    TurnKind::AfterSpare { spare: 1 },
                    STREAM_TURN_WAIT
                )
                .await
                .err(),
            Some(TurnRefused::StoodOnARefusal { refused_at: 1 })
        );
        run(&streams, id, 3, TurnKind::Restarts).await;
        run(&streams, id, 4, TurnKind::Follows).await;
    }

    /// With the tail skip switched off (the A/B arm) a stale turn still runs
    /// after a refusal — but never once a spare has run for it: the spare's
    /// rows are in the cache, and the stale turn would write over them.
    #[test]
    fn a_taken_spare_skips_the_stale_line_whatever_the_switch() {
        let mut order = Order::default();
        order.admit(0, TurnKind::Follows, false).unwrap();
        order.refusal = Some(Refusal {
            turn: 0,
            kept: 1,
            token: 42,
        });
        assert_eq!(
            order.admit(1, TurnKind::Follows, false),
            Ok(()),
            "the A/B arm runs a stale turn while no spare has"
        );
        order.admit(2, spare(0, 1, 42), false).unwrap();
        assert_eq!(
            order.admit(3, TurnKind::Follows, false),
            Err(TurnRefused::StoodOnARefusal { refused_at: 0 })
        );
        order
            .admit(4, TurnKind::AfterSpare { spare: 2 }, false)
            .unwrap();
        assert_eq!(order.admit(5, TurnKind::Follows, false), Ok(()));
    }

    /// A spare supersedes nothing on arrival — the check it waits on may not
    /// have run yet, and must — where a restart arriving the same way
    /// supersedes it. One spare per refusal: a second for the same one is
    /// skipped, the cache already cut.
    #[tokio::test]
    async fn a_spare_supersedes_nothing_and_only_one_runs_per_refusal() {
        let streams = Arc::new(ForwardStreams::default());
        let id = uuid::Uuid::new_v4();
        let early = {
            let streams = streams.clone();
            tokio::spawn(async move {
                streams
                    .turn(id, RANGE, 1, spare(0, 1, 42), STREAM_TURN_WAIT)
                    .await
                    .map(drop)
            })
        };
        tokio::task::yield_now().await;
        let check = streams
            .turn(id, RANGE, 0, TurnKind::Follows, STREAM_TURN_WAIT)
            .await
            .expect("the check a spare waits on is not superseded by it");
        check.refused_a_guess(1, 42);
        drop(check);
        assert!(early.await.unwrap().is_ok(), "the spare then runs");
        assert_eq!(
            streams
                .turn(id, RANGE, 2, spare(0, 1, 42), STREAM_TURN_WAIT)
                .await
                .err(),
            Some(TurnRefused::SpareNotNeeded { of: 0 })
        );

        // A restart in the spare's place supersedes the check queued before it.
        let id = uuid::Uuid::new_v4();
        let restart = {
            let streams = streams.clone();
            tokio::spawn(async move {
                streams
                    .turn(id, RANGE, 1, TurnKind::Restarts, STREAM_TURN_WAIT)
                    .await
                    .map(drop)
            })
        };
        tokio::task::yield_now().await;
        assert_eq!(
            streams
                .turn(id, RANGE, 0, TurnKind::Follows, STREAM_TURN_WAIT)
                .await
                .err(),
            Some(TurnRefused::Skipped)
        );
        assert!(restart.await.unwrap().is_ok());
    }

    /// What a forward is, from its cut and branch: a branch that cannot mean
    /// what it says — a spare that does not cut to its own first row, a
    /// check on a spare that does not either, a later or another attempt's
    /// turn named — is refused.
    #[test]
    fn a_forwards_kind_is_read_off_its_cut_and_branch() {
        use crate::types::StreamBranch;
        let seq = |attempt, turn| stream_seq::compose(attempt, turn).unwrap();
        let at = seq(3, 7);
        let spare_of = |of| StreamBranch::Spare {
            of,
            kept: 1,
            token: 42,
        };
        assert_eq!(TurnKind::of(at, 50, None, None), Ok(TurnKind::Follows));
        assert_eq!(TurnKind::of(at, 50, Some(48), None), Ok(TurnKind::Restarts));
        assert_eq!(
            TurnKind::of(at, 50, Some(50), Some(spare_of(seq(3, 6)))),
            Ok(spare(6, 1, 42))
        );
        assert_eq!(
            TurnKind::of(
                at,
                50,
                Some(50),
                Some(StreamBranch::AfterSpare { spare: seq(3, 5) })
            ),
            Ok(TurnKind::AfterSpare { spare: 5 })
        );
        for (cut, branch) in [
            (None, spare_of(seq(3, 6))),
            (Some(48), spare_of(seq(3, 6))),
            (Some(50), spare_of(seq(3, 7))),
            (Some(50), spare_of(seq(3, 8))),
            (Some(50), spare_of(seq(2, 6))),
            (None, StreamBranch::AfterSpare { spare: seq(3, 5) }),
            (Some(50), StreamBranch::AfterSpare { spare: seq(4, 5) }),
            (Some(50), StreamBranch::AfterSpare { spare: seq(3, 9) }),
        ] {
            assert!(
                TurnKind::of(at, 50, cut, Some(branch)).is_err(),
                "{cut:?} {branch:?}"
            );
        }
    }
}
