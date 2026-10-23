//! A tree of guesses that FLOWS (#134 (b), `docs/plans/large_model_splits.md`
//! § "A tree that flows"): the drafter grows one tree a few rows a call, each
//! call's rows go through this node's layers and on to the far segment at
//! once, and the far segment answers every row with the token the model draws
//! there. This node walks those draws (`draft_tree::walk_draws`) and emits
//! each token the moment the walk reaches it — so the reply advances as deep
//! as the tree reaches along the model's own path, and waits a round trip only
//! where no row holds the draw.
//!
//! **Why not rounds.** A tree round waits a whole round trip per check, and
//! on TH↔AU (~411 ms) a third of each round was this node waiting for the
//! drafter. Here nothing waits for an answer but the walk: forwards keep
//! leaving while earlier ones are read.
//!
//! **How the machines keep one tree.** Every segment's cache keeps each row
//! sent since the tree's start (`LayerForward::draft_tree_from`), in the order
//! sent, and every forward carries the whole tree's shape — so a row reads the
//! context and its own ancestors, and a lost branch stays in the caches, read
//! by nothing. The rows change only by [`FlowOp`]s, which this node and the
//! drafter's worker apply alike (`SplitModel::grow_flow`),
//! and by the drafter's proposals, rows of its tree once made and pushed here
//! in its order. A call grows several LEVELS, each read by the next: at one a
//! call, a branch grew one row deeper a call and the walk with it.
//! A tree that nears its row limit (`MAX_TREE_ROWS`) is REBASED: the walk's
//! row becomes the new start, one forward cuts every cache back to the old
//! start and reads the walk's path there, then the rows still live — the
//! stream's restart (`daemon::state::forward_streams`), which skips the turns
//! before it not yet run; their rows are read again by it. A rebase is sent
//! only once the one before it was ANSWERED: a restart is safe only where every
//! turn it builds on ran (`forward_streams`), and one sent early could skip the
//! rebase before it there and leave that tree's rows where its path belongs.
//!
//! **A failed flow is finished in tree rounds** where the peers still answer
//! (`rounds_may_finish`): every cache is cut back to the last start every
//! segment ANSWERED at and the reply past it read again (`rounds_take_over`) —
//! past the start the cache holds rows in the order SENT, never the reply's.
//!
//! **Exact at any temperature.** A row's draw is the model's sample after
//! that row's path, its shared noise keyed by position (`tree_start + 1 +
//! depth`), whichever forward carried it — the walk's tokens are the ones a
//! plain decode would emit with that noise. A request whose sampler reads its
//! own history (penalties) cannot be drawn so and keeps the rounds.
//!
//! FlowSpec (2507.02620) streams score-ordered tree segments on a LAN with
//! its verdict sent back to the drafter and a miss starting a new round;
//! nothing published measures this shape over hundreds of ms.

use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::Ordering;
use std::sync::{Arc, OnceLock};

use futures::stream::{FuturesUnordered, StreamExt};
use futures::FutureExt;

use crate::daemon::state::{ExpectedStep, PendingLayerResult, SharedState, WaiterKey};
use crate::error::SwarmError;
use crate::inference::draft_tree::{walk_draws, FlowOp, FlowTree, WalkEnd, MAX_TREE_ROWS};
use crate::types::inference::stream_seq;
use crate::types::{LayerResult, NetworkCommand, NetworkFinishReason};

use super::dsd_stream::{StreamIo, StreamReply, StreamShape, StreamWaiter, NEXT_ATTEMPT};
use super::engine_drafter::EngineDrafter;
use super::local::{ActivationUnits, ResendOnRefusal, SegmentBudget};
use super::{PipelineExecutor, TailWalk};

/// Rows the drafter proposes a level (`levels_a_call`). 8, EAGLE-2's top-k a
/// depth. `SWARMLLM_FLOW_ROWS` (1-64).
fn rows_a_level() -> u32 {
    static N: OnceLock<u32> = OnceLock::new();
    *N.get_or_init(|| knob("SWARMLLM_FLOW_ROWS", 8, 1, 64))
}

/// Levels a drafter call grows, each read by the next in the same call
/// (`split::FlowStep::levels`). At one, the walk advanced 0.63-0.74 tokens a
/// call whatever the rows a call (8, 16 or 32; `flow318.sh` on the card,
/// 2026-10-22): a branch grew one row deeper a call, and the walk can go no
/// deeper than the drafter has. `SWARMLLM_FLOW_LEVELS` (1-8).
fn levels_a_call() -> u32 {
    static N: OnceLock<u32> = OnceLock::new();
    *N.get_or_init(|| knob("SWARMLLM_FLOW_LEVELS", 3, 1, 8))
}

/// Live rows (the walk's row and those following it) past which the drafter
/// is not asked for more: past ~128 a row's chance of lying on the reply's
/// path is too small to pay for its place (the model: 128 lost nothing
/// against 1024, 64 lost a fifth). `SWARMLLM_FLOW_LIVE` (16-448).
fn live_most() -> u32 {
    static N: OnceLock<u32> = OnceLock::new();
    *N.get_or_init(|| knob("SWARMLLM_FLOW_LIVE", 128, 16, 448))
}

/// Whether the flow is held down while its forwards queue ([`FlowGuard`]): ON
/// unless `SWARMLLM_FLOW_GUARD=0` (or `off` / `false`), which keeps
/// `rows_a_level` and `live_most` — the A/B inside one binary.
fn guard_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        !matches!(
            std::env::var("SWARMLLM_FLOW_GUARD").ok().as_deref(),
            Some("0") | Some("off") | Some("false")
        )
    })
}

/// `SWARMLLM_FLOW_COPY=0|off|false`: no copied runs (`FlowRun::copy_ahead`),
/// for an A/B inside one binary. Read once.
fn copy_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        !matches!(
            std::env::var("SWARMLLM_FLOW_COPY").ok().as_deref(),
            Some("0") | Some("off") | Some("false")
        )
    })
}

/// The most rows of a copied run kept ahead of the walk. A run the walk keeps
/// following is extended as it goes, so a round trip carries at most this many
/// copied tokens: 32 a 385 ms trip is ~83 a second, past what this node's own
/// layers keep up with.
const COPY_LEAD_MOST: usize = 32;

/// Where a copied run leaves the tree: it follows the rows after `at` that
/// already carry its tokens — the drafter's own included — and the rest, at
/// most `room` of them, are new rows in a line after the row it reached.
fn copy_under<'r>(flow: &FlowTree, at: usize, run: &'r [u32], room: usize) -> (usize, &'r [u32]) {
    let mut row = at;
    let mut held = 0;
    for &token in run {
        match flow.child_carrying(row, token) {
            Some(child) => {
                row = child;
                held += 1;
            }
            None => break,
        }
    }
    let rest = &run[held..];
    (row, &rest[..rest.len().min(room)])
}

/// LEDBAT++'s queueing-delay target (draft-irtf-iccrg-ledbat-plus-plus-06
/// § 4.5, 60 ms against RFC 6817's 100): the flow is narrowed while the
/// forwards' answers come back later than this past the least delay seen.
const QUEUE_TARGET_MS: f64 = 60.0;

/// The rows a level the guard narrows to before it touches the live budget:
/// at 2 a flow sent ~10 rows a token and never waited on a 5 Mbit/s link
/// (6.3-6.7 tok/s, its own layers' forwards the bound), at 8 ~26 (4.5-4.8).
const WIDTH_LEAST: u32 = 2;

/// The live budget the guard cuts no lower than. At 5 Mbit/s and 8 rows a
/// level, 32 live rows ran 6.3-6.5 tok/s, 64 ran 5.7-5.8 and 128 4.5-4.8 — but
/// at 32 the walk waited on the round trip for 45 of 60 s: too little tree
/// ahead of it, never too many bytes.
const LIVE_LEAST: u32 = 32;

/// Answers whose least delay is read as the queue's: LEDBAT++ takes the least
/// of the 4 latest samples, against the far side's own time varying with the
/// rows a forward carries. Each sample is the answer's ARRIVAL, stamped on a
/// task of its own (`Arrival`), never when this node reads it — it reads one
/// only between its drafter calls and its own layers' forwards, and with the
/// reading time the guard cut an unthrottled link from 8x128 to 2x32 (24 cuts,
/// queue up to 141 ms, 7.8-8.0 tok/s against 8.8-8.9 unguarded): RFC 6817
/// § 3.4.1 timestamps a packet on arrival for the same reason.
const DELAY_SAMPLES: usize = 4;

/// How long a run's guard is kept for its next reply: LEDBAT's base-delay
/// window (RFC 6817 § 3.3, 10 minutes).
const GUARD_KEPT_FOR: std::time::Duration = std::time::Duration::from_secs(600);

/// The rows a flow grows a level and keeps ahead of the walk, held down by the
/// forwards' own queueing delay — a congestion window in rows (LEDBAT, RFC
/// 6817, as LEDBAT++ tunes it). What a flow costs its uplink is the rows that
/// die: each token the walk advances kills the rows not under it, and the
/// drafter grows the tree back. A WIDE call grows rows the next advance kills
/// before they are read — 8 rows a level sent ~26 rows a token, 2 sent ~10
/// (5 Mbit/s, 128 live rows, `thinlink.sh` 2026-10-22) — while a SMALL live
/// budget leaves the walk too little tree to go on (32 rows: waiting on the
/// round trip 45 s of 60). So a queue narrows the levels first, to
/// [`WIDTH_LEAST`], and only then the budget, to [`LIVE_LEAST`]; room widens
/// the budget back first, then the levels.
///
/// A cut is `min(1/2, queue / target − 1)` of the one it narrows (LEDBAT++'s
/// multiplicative decrease, capped at half), and only on an answer to a
/// forward sent AFTER the last cut — one already on its way then shows the
/// queue the cut has not acted on yet (TCP's one reduction a window). Below
/// the target it grows back a round trip at a time by up to a level's rows
/// (the budget) or one row (a level), LEDBAT's `off_target` share of it. The
/// least delay is this reply's own: its first forward leaves into an empty
/// queue.
///
/// Measured (`guardrig2.sh`, 2026-10-22, 7B split at 385 ms, warm replies):
/// at 5 Mbit/s 7.44-7.50 tok/s against 4.73-4.80 unguarded — past every fixed
/// setting and the tree rounds (6.8-7.1) —, at 10 Mbit/s 8.52-8.82 against
/// 8.23, and with no rate limit no cut but in a burst of the far half's own
/// queue (it shares this rig's one card).
#[derive(Debug, Clone)]
struct FlowGuard {
    /// Rows a level, fractional so a small rise accumulates.
    width: f64,
    width_most: u32,
    /// Live rows, fractional likewise.
    live: f64,
    live_most: u32,
    guarded: bool,
    least_delay_ms: Option<f64>,
    recent: std::collections::VecDeque<f64>,
    /// The last cut: a later one needs an answer to a forward sent after it.
    cut_at: Option<std::time::Instant>,
    /// The last change, cut or rise: a rise waits a round trip after it.
    changed_at: Option<std::time::Instant>,
    cuts: u32,
    least_width: u32,
    least_live: u32,
    queue_ms_most: f64,
}

/// Where a guard stands — what a reply leaves for the next on its run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct GuardAt {
    width: u32,
    live: u32,
}

impl FlowGuard {
    fn new(width_most: u32, live_most: u32, guarded: bool, kept: Option<GuardAt>) -> Self {
        let mut guard = Self {
            width: f64::from(width_most),
            width_most,
            live: f64::from(live_most),
            live_most,
            guarded,
            least_delay_ms: None,
            recent: std::collections::VecDeque::with_capacity(DELAY_SAMPLES),
            cut_at: None,
            changed_at: None,
            cuts: 0,
            least_width: width_most,
            least_live: live_most,
            queue_ms_most: 0.0,
        };
        if let Some(kept) = kept.filter(|_| guarded) {
            guard.width = f64::from(kept.width);
            guard.live = f64::from(kept.live);
            let at = guard.at();
            guard.width = f64::from(at.width);
            guard.live = f64::from(at.live);
            guard.least_width = at.width;
            guard.least_live = at.live;
        }
        guard
    }

    /// Rows a level and the live budget, now.
    fn at(&self) -> GuardAt {
        GuardAt {
            width: (self.width.round() as u32)
                .clamp(WIDTH_LEAST.min(self.width_most), self.width_most),
            live: (self.live.round() as u32).clamp(LIVE_LEAST.min(self.live_most), self.live_most),
        }
    }

    /// A forward of the tree's own rows (never a rebase's), sent at `sent`,
    /// answered `delay_ms` after it left, read at `now`.
    fn answered(&mut self, sent: std::time::Instant, delay_ms: f64, now: std::time::Instant) {
        if !self.guarded || !delay_ms.is_finite() {
            return;
        }
        let base = self.least_delay_ms.map_or(delay_ms, |b| b.min(delay_ms));
        self.least_delay_ms = Some(base);
        if self.recent.len() == DELAY_SAMPLES {
            self.recent.pop_front();
        }
        self.recent.push_back(delay_ms);
        if self.recent.len() < DELAY_SAMPLES {
            return;
        }
        let queue = self.recent.iter().copied().fold(f64::INFINITY, f64::min) - base;
        self.queue_ms_most = self.queue_ms_most.max(queue);
        let width_least = f64::from(WIDTH_LEAST.min(self.width_most));
        let live_least = f64::from(LIVE_LEAST.min(self.live_most));
        if queue > QUEUE_TARGET_MS {
            if self.cut_at.is_some_and(|c| sent <= c) {
                return;
            }
            let keep = 1.0 - (queue / QUEUE_TARGET_MS - 1.0).min(0.5);
            let before = self.at();
            if self.width > width_least {
                self.width = (self.width * keep).max(width_least);
            } else {
                self.live = (self.live * keep).max(live_least);
            }
            let after = self.at();
            if after != before {
                self.cuts += 1;
                self.least_width = self.least_width.min(after.width);
                self.least_live = self.least_live.min(after.live);
                self.cut_at = Some(now);
            }
        } else {
            if self
                .changed_at
                .is_some_and(|c| now.duration_since(c).as_secs_f64() * 1000.0 < base)
            {
                return;
            }
            let room = 1.0 - queue.max(0.0) / QUEUE_TARGET_MS;
            if self.live < f64::from(self.live_most) {
                self.live =
                    (self.live + f64::from(self.width_most) * room).min(f64::from(self.live_most));
            } else {
                self.width = (self.width + room).min(f64::from(self.width_most));
            }
        }
        self.changed_at = Some(now);
    }
}

/// The most distinct runs whose guards are kept; past it the memory starts
/// again (`dsd_controller`'s rule).
const GUARDS_KEPT_MOST: usize = 64;

type GuardStore =
    std::sync::Mutex<std::collections::HashMap<String, (GuardAt, std::time::Instant)>>;

fn guard_store() -> &'static GuardStore {
    static STORE: OnceLock<GuardStore> = OnceLock::new();
    STORE.get_or_init(Default::default)
}

/// Where the last reply over `key` (the model and the run's machines, in
/// order) left its guard, within [`GUARD_KEPT_FOR`].
fn kept_guard(key: &str, now: std::time::Instant) -> Option<GuardAt> {
    guard_store()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .get(key)
        .filter(|(_, at)| now.duration_since(*at) < GUARD_KEPT_FOR)
        .map(|(guard, _)| *guard)
}

fn keep_guard(key: String, guard: GuardAt, now: std::time::Instant) {
    let mut store = guard_store().lock().unwrap_or_else(|e| e.into_inner());
    if store.len() >= GUARDS_KEPT_MOST && !store.contains_key(&key) {
        store.clear();
    }
    store.insert(key, (guard, now));
}

/// Forwards out at once. 12: in the schedule model at 425 ms (each forward
/// its own read at the far end, a full window growing on into the next) 8 gave
/// 9.30 tok/s, 12 gave 9.81, no limit 9.86. A serving node holds up to
/// `forward_streams::MAX_STREAM_CHUNKS_HERE` (16) of one stream, and a flow
/// keeps one or two there: the rest are on the wire, out or back.
/// `SWARMLLM_FLOW_WINDOW` (1-16). Through a chain, at most
/// `forward_streams::MAX_STREAM_WINDOW` (8): a hop remembers 16 of a stream's
/// forwards on the rule that a coordinator keeps half that out
/// (`hop_forwards::MAX_STREAM_RECORDS`).
fn forwards_in_flight(chained: bool) -> usize {
    static N: OnceLock<u32> = OnceLock::new();
    let most = *N.get_or_init(|| {
        knob(
            "SWARMLLM_FLOW_WINDOW",
            12,
            1,
            crate::daemon::state::forward_streams::MAX_STREAM_CHUNKS_HERE as u32,
        )
    });
    most.min(if chained {
        crate::daemon::state::forward_streams::MAX_STREAM_WINDOW
    } else {
        u32::MAX
    }) as usize
}

/// A walk this deep from the tree's start rebases it whatever its size, so a
/// rebase's forward — the path, then the live rows, every one answered with a
/// draw — stays a few hundred rows at most.
const REBASE_PATH_ROWS: usize = 128;

/// `SWARMLLM_FAULT_FLOW_FAIL=<n>`: this node's flow fails as it takes its
/// `n`th answer of every reply (counting from 1), as a flow a peer refuses
/// would. The test that the tree rounds then finish the reply from the last
/// token the flow emitted (`rounds_may_finish`) — the only way to fail a flow
/// on demand with a peer that reads one. Unset in production; read once.
fn fault_flow_fail_at() -> Option<u32> {
    static N: OnceLock<Option<u32>> = OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("SWARMLLM_FAULT_FLOW_FAIL")
            .ok()
            .and_then(|v| v.trim().parse::<u32>().ok())
    })
}

/// `SWARMLLM_FAULT_DRAFT_DELAY_MS=<ms>`: every drafter call of a flow takes
/// this much longer, as a costlier drafter's would — the same rows, the same
/// guesses, only the time. The instrument for "does this node's drafting
/// bind the flow" (`docs/plans/faster_than_local.md` § 6.1): the slope of
/// tok/s against it says what taking the drafter off this node could buy.
/// Unset in production; read once.
fn fault_draft_delay() -> Option<std::time::Duration> {
    static D: OnceLock<Option<std::time::Duration>> = OnceLock::new();
    *D.get_or_init(|| {
        std::env::var("SWARMLLM_FAULT_DRAFT_DELAY_MS")
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok())
            .filter(|&ms| ms > 0)
            .map(std::time::Duration::from_millis)
    })
}

/// Whether the tree rounds may finish a reply whose flow failed: the peers
/// ANSWERED — refused a forward (a stale reading of a peer's features, a
/// security review's case) or answered one this node could not use — so a
/// round still reaches them. A peer gone silent or out of reach is the
/// router's to re-plan, never waited on twice.
pub(super) fn rounds_may_finish(e: &SwarmError) -> bool {
    matches!(
        e,
        SwarmError::Validation(_) | SwarmError::Inference(_) | SwarmError::Internal(_)
    )
}

/// A flow that failed, and where the reply stands in every segment's cache.
pub(super) struct FlowFailed {
    pub(super) error: SwarmError,
    /// The tree's start: the caches hold the reply up to it, and past it the
    /// tree's rows in the order they were SENT, not the reply's.
    pub(super) base: usize,
}

/// How the tree rounds take over a reply whose flow failed (`FlowFailed`).
#[derive(Debug, PartialEq, Eq)]
pub(super) struct RoundsTakeOver {
    /// The reply's tokens from the tree's start to before its last, read again
    /// in front of the next check (`TreeRounds::reread`).
    pub(super) reread: Vec<u32>,
    /// The caches are cut here: the tree's start.
    pub(super) kv_len: u32,
}

/// Where the rounds pick a failed flow's reply up: every cache is cut back to
/// the tree's start and the reply's tokens past it are read again — a kept
/// path's own rule (`docs/invariants/scheduling.md` § "A split checks a tree of
/// guesses": cut and read again, never gather). Cutting to the reply's length
/// instead (the chain stream's take-over) kept the first rows SENT as if they
/// were the reply, and the next token ranked 8th against llama.cpp
/// (2026-10-22, `flow318.sh` arm `fault`). `generated[i]` sits at
/// `prompt_len + i`; the last is the next check's own token.
pub(super) fn rounds_take_over(
    prompt_len: usize,
    generated: &[u32],
    base: usize,
) -> RoundsTakeOver {
    let last_at = prompt_len + generated.len().saturating_sub(1);
    let base = base.clamp(prompt_len, last_at);
    RoundsTakeOver {
        reread: generated[base - prompt_len..generated.len().saturating_sub(1)].to_vec(),
        kv_len: base as u32,
    }
}

fn knob(name: &str, default: u32, min: u32, max: u32) -> u32 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.trim().parse::<u32>().ok())
        .map_or(default, |v| v.clamp(min, max))
}

/// Whether a reply that checks trees lets them flow: ON unless
/// `SWARMLLM_TREE_FLOW=0` (or `off` / `false`), which keeps the tree rounds —
/// the A/B inside one binary.
pub(super) fn flow_requested() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        !matches!(
            std::env::var("SWARMLLM_TREE_FLOW").ok().as_deref(),
            Some("0") | Some("off") | Some("false")
        )
    })
}

/// Can a reply's tree flow on this stream shape: the run's tail walks (it
/// holds the model's last layers), every peer of the run reads a tree that
/// flows (`features::DRAFT_TREE_FLOW`, beside `DRAFT_TREES`), and the sampler
/// reads no history — a draw per row reads no path's tokens.
pub(super) fn flow_possible(
    state: &SharedState,
    shape: &StreamShape,
    sampling: &crate::types::SamplingParams,
) -> bool {
    use swarmllm_types::node::features::{DRAFT_TREES, DRAFT_TREE_FLOW};
    shape.far_walks()
        && !crate::inference::sampling::sampler_reads_history(sampling)
        && std::iter::once(&shape.far.node_id)
            .chain(shape.chain.iter().map(|h| &h.node_id))
            .all(|n| state.peer_advertises_feature(n, DRAFT_TREE_FLOW | DRAFT_TREES))
}

/// A forward out being read.
struct Sent {
    seq: u32,
    /// The tree's rebases before it: an answer from before the last one is
    /// mapped onto the rows kept, one from before that is dropped.
    epoch: u32,
    /// Rows at the forward's front that are not the tree's — a rebase's path.
    lead: usize,
    /// The tree's row its first own row is.
    first_row: usize,
    /// Rows the forward carried, the lead included.
    rows: usize,
    /// Nothing else was out when it left: its time is a clean sample of what
    /// a check costs.
    alone: bool,
    started: std::time::Instant,
    /// A rebase's forward: the tree's start it sets, settled when answered.
    rebases_to: Option<usize>,
}

/// A forward's answer, and when it ARRIVED — not when this node got to it.
type Answer = (Sent, Result<LayerResult, SwarmError>, std::time::Instant);
type Waiting = Pin<Box<dyn Future<Output = Answer> + Send>>;

/// A forward's wait runs on a task of its own, so its answer is timed when it
/// lands (`DELAY_SAMPLES`); the task goes when the run lets go of the
/// forward, as the wait did when it was polled in place.
struct Arrival(tokio::task::AbortHandle);

impl Drop for Arrival {
    fn drop(&mut self) {
        self.0.abort();
    }
}

enum Taken {
    Continue,
    Done,
}

/// What a forward carries.
struct Rows {
    index_pos: usize,
    tokens: Vec<u32>,
    wire: Vec<u16>,
    from: usize,
    truncate: Option<u32>,
    lead: usize,
    first_row: usize,
    rebases_to: Option<usize>,
}

struct FlowRun<'a, 'r> {
    exec: &'a PipelineExecutor,
    state: Arc<SharedState>,
    shape: StreamShape,
    attempt: u32,
    turn: u32,
    flow: FlowTree,
    /// Each row's draw at the far end, once its answer is in.
    draws: Vec<Option<u32>>,
    /// Rows every segment has been sent.
    sent: usize,
    /// Changes the drafter has not been told of, in order.
    ops: Vec<FlowOp>,
    /// The drafter's tree has started.
    growing: bool,
    epoch: u32,
    /// The last rebase's old rows → rows kept, for answers from before it.
    remap: Option<(u32, Vec<Option<usize>>)>,
    out: FuturesUnordered<Waiting>,
    drafter: &'a mut EngineDrafter,
    drafting_off: bool,
    drafter_warm: bool,
    coupling: Option<u64>,
    io: StreamIo<'a>,
    reply: StreamReply<'r>,
    forwards: u32,
    rows_sent: u32,
    rebases: u32,
    misses: u32,
    stale_answers: u32,
    /// Where this node's time went (ms): the drafter's calls, its own layers'
    /// forwards, and waiting with nothing to grow or send.
    drafting_ms: f64,
    near_ms: f64,
    waiting_ms: f64,
    answers: u32,
    /// The tree's start every segment has been ANSWERED at: the caches hold
    /// the reply up to it whatever is still on its way. A rebase builds only
    /// on one settled — `forward_streams`' premise for a restart is that every
    /// turn it builds on ran, and a rebase sent before the last one's answer
    /// could skip it there and leave that rebase's rows unread.
    settled_base: usize,
    /// A rebase is out and not yet answered: the next waits for it.
    unsettled: bool,
    /// A miss the full tree had no room for while a rebase was unsettled: its
    /// token is emitted, its row pushed once the rebase is answered.
    bonus_waiting: Option<u32>,
    /// The rows a level and the live rows the drafter may keep ahead of the
    /// walk, held down while the forwards queue.
    guard: FlowGuard,
    /// Each row was copied (`copy_ahead`), beside `draws`.
    copied: Vec<bool>,
    /// The lookup's shape; `None` where copied runs are off.
    copy: Option<crate::inference::ngram_lookup::NgramLookupConfig>,
    /// The walk's row, by epoch, a run was last looked up from.
    copied_from: Option<(u32, usize)>,
    copied_runs: u32,
    copied_rows: u32,
    /// Copied rows the walk went through: drawn there as copied.
    copied_walked: u32,
}

impl PipelineExecutor {
    /// Let the reply's tree flow (module doc) until the reply ends, from the
    /// prompt pass's token at `start_pos`, the last of `reply.generated`.
    pub(super) async fn flow_checks(
        &self,
        shape: StreamShape,
        drafter: &mut EngineDrafter,
        start_pos: u32,
        io: StreamIo<'_>,
        reply: StreamReply<'_>,
    ) -> Result<(), FlowFailed> {
        let state = self.shared_state.clone();
        let coupling = io.noise.map(|n| n.seed()).filter(|_| {
            state.peer_advertises_feature(
                shape.tail(),
                swarmllm_types::node::features::COUPLED_SAMPLING,
            )
        });
        let root = *reply
            .generated
            .last()
            .expect("a reply holds its first token before any check");
        let chained = shape.chained();
        // The guard is the run's link's: the model (a row's bytes) and the
        // machines in order.
        let guard_key = format!(
            "{}|{}",
            self.request.model_id.0,
            self.assignment
                .segments
                .iter()
                .map(|s| s.node_id.to_string())
                .collect::<Vec<_>>()
                .join(",")
        );
        let guard = FlowGuard::new(
            rows_a_level(),
            live_most(),
            guard_on(),
            kept_guard(&guard_key, std::time::Instant::now()),
        );
        let guard_at_start = guard.at();
        let mut run = FlowRun {
            exec: self,
            state,
            shape,
            attempt: NEXT_ATTEMPT.fetch_add(1, Ordering::Relaxed) & stream_seq::MAX_ATTEMPT,
            turn: 0,
            flow: FlowTree::new(start_pos as usize, root).map_err(|e| FlowFailed {
                error: SwarmError::Internal(e),
                base: start_pos as usize,
            })?,
            draws: vec![None],
            sent: 0,
            ops: Vec::new(),
            growing: false,
            epoch: 0,
            remap: None,
            out: FuturesUnordered::new(),
            drafter,
            drafting_off: false,
            drafter_warm: false,
            coupling,
            io,
            reply,
            forwards: 0,
            rows_sent: 0,
            rebases: 0,
            misses: 0,
            stale_answers: 0,
            drafting_ms: 0.0,
            near_ms: 0.0,
            waiting_ms: 0.0,
            answers: 0,
            settled_base: start_pos as usize,
            unsettled: false,
            bonus_waiting: None,
            guard,
            copied: vec![false],
            copy: super::speculative::ngram_lookup_config(&self.shared_state.cfg().inference)
                .filter(|_| copy_on()),
            copied_from: None,
            copied_runs: 0,
            copied_rows: 0,
            copied_walked: 0,
        };
        let started = std::time::Instant::now();
        let emitted_before = run.reply.generated.len();
        let outcome = run.run_with().await;
        let flowed_ms = started.elapsed().as_secs_f64() * 1000.0;
        // A failed flow leaves no forward on its way where anything follows
        // it — a chain (`dsd_stream`'s rule) or the rounds finishing the reply
        // (`rounds_may_finish`): each still out is answered first.
        if outcome
            .as_ref()
            .is_err_and(|e| chained || rounds_may_finish(e))
        {
            while run.out.next().await.is_some() {}
        }
        run.drafter.flow_ended(run.flow.base());
        if run.guard.guarded {
            keep_guard(guard_key, run.guard.at(), std::time::Instant::now());
        }
        tracing::info!(
            request_id = %self.request.id,
            peers = run.shape.chain.len() + 1,
            forwards = run.forwards,
            rows_a_forward = format_args!(
                "{:.1}",
                f64::from(run.rows_sent) / f64::from(run.forwards.max(1))
            ),
            rebases = run.rebases,
            misses = run.misses,
            stale_answers = run.stale_answers,
            emitted = run.reply.generated.len(),
            flowed_ms = format_args!("{flowed_ms:.0}"),
            tokens_per_sec = format_args!(
                "{:.2}",
                (run.reply.generated.len() - emitted_before) as f64 * 1000.0 / flowed_ms.max(1.0)
            ),
            drafting_ms = format_args!("{:.0}", run.drafting_ms),
            near_ms = format_args!("{:.0}", run.near_ms),
            waiting_ms = format_args!("{:.0}", run.waiting_ms),
            rows_a_level = rows_a_level(),
            levels = levels_a_call(),
            live_most = live_most(),
            guard = run.guard.guarded,
            guard_at_start = format_args!("{}x{}", guard_at_start.width, guard_at_start.live),
            guard_least = format_args!("{}x{}", run.guard.least_width, run.guard.least_live),
            guard_at_end = format_args!("{}x{}", run.guard.at().width, run.guard.at().live),
            guard_cuts = run.guard.cuts,
            least_answer_ms = format_args!("{:.0}", run.guard.least_delay_ms.unwrap_or(0.0)),
            queue_ms_most = format_args!("{:.0}", run.guard.queue_ms_most),
            copy = run.copy.is_some(),
            copied_runs = run.copied_runs,
            copied_rows = run.copied_rows,
            copied_walked = run.copied_walked,
            window = forwards_in_flight(chained),
            "DSD: a tree that flowed is complete"
        );
        outcome.map_err(|error| FlowFailed {
            error,
            base: run.settled_base,
        })
    }
}

impl FlowRun<'_, '_> {
    async fn run_with(&mut self) -> Result<(), SwarmError> {
        let window = forwards_in_flight(self.shape.chained());
        if self.reply.generated.len() as u32 >= self.io.max_tokens {
            return Ok(());
        }
        loop {
            if self.exec.request.is_cancelled() {
                tracing::info!(
                    request_id = %self.exec.request.id,
                    "DIAG: DSD inference cancelled externally"
                );
                *self.reply.finish_reason = "stop".to_string();
                return Ok(());
            }
            // Answers already in go first: the walk they move decides what
            // is worth growing.
            while let Some(Some(answer)) = self.out.next().now_or_never() {
                if let Taken::Done = self.take_answer(answer).await? {
                    return Ok(());
                }
            }
            if self.bonus_waiting.is_some() && !self.unsettled {
                self.push_waiting_bonus().await?;
            }
            let grew = self.grow().await?;
            if self.sent < self.flow.len() && self.out.len() < window {
                let rows = self.rows_unsent();
                let forward = self.send(rows).await?;
                self.out.push(forward);
                continue;
            }
            if grew {
                continue;
            }
            let waited = std::time::Instant::now();
            let next = self.out.next().await;
            self.waiting_ms += waited.elapsed().as_secs_f64() * 1000.0;
            match next {
                Some(answer) => {
                    if let Taken::Done = self.take_answer(answer).await? {
                        return Ok(());
                    }
                }
                None => {
                    return Err(SwarmError::Internal(
                        "a tree that flows stalled: nothing out, nothing to send".into(),
                    ))
                }
            }
        }
    }

    /// One drafter call: rows proposed after the live rows, `levels_a_call`
    /// deep, made rows here as the drafter made them in its tree. `false`
    /// when nothing grew.
    async fn grow(&mut self) -> Result<bool, SwarmError> {
        let live = self.flow.live_rows() as u32;
        let guard = self.guard.at();
        if self.drafting_off
            || live >= guard.live
            || self.reply.generated.len() as u32 >= self.io.max_tokens
        {
            return Ok(false);
        }
        let (rows, levels) = (guard.width, levels_a_call());
        if self.flow.len() + (rows * levels) as usize + 1 > MAX_TREE_ROWS
            || self.flow.tree().depth(self.flow.at()) >= REBASE_PATH_ROWS
        {
            if self.unsettled || self.bonus_waiting.is_some() {
                return Ok(false);
            }
            self.rebase().await?;
        }
        // A copied run goes out at once, the drafter's call after it; it
        // leaves that call room in the tree.
        let room = ((guard.live - live) as usize)
            .min(MAX_TREE_ROWS.saturating_sub(self.flow.len() + (rows * levels) as usize + 1));
        if self.copy_ahead(room) {
            return Ok(true);
        }
        let exec = self.exec;
        let started = std::time::Instant::now();
        let asked = self
            .drafter
            .flow(
                &self.state,
                !self.growing,
                std::mem::take(&mut self.ops),
                self.flow.base(),
                rows,
                levels,
                guard.live,
                &exec.request.sampling_params,
                self.io.noise.map(|n| n.seed()),
                exec.request.cancel.clone(),
            )
            .await;
        if let Some(delay) = fault_draft_delay() {
            tokio::time::sleep(delay).await;
        }
        self.drafting_ms += started.elapsed().as_secs_f64() * 1000.0;
        let proposed = match asked {
            Ok(proposed) => proposed,
            Err(e) => {
                tracing::warn!(
                    request_id = %exec.request.id,
                    error = %e,
                    "DSD: the drafter failed — the tree goes on with the model's own draws only"
                );
                self.drafting_off = true;
                return Ok(false);
            }
        };
        // The first call reads the prompt's end and the root, and is not a
        // guess's cost (`dsd`'s rule).
        if self.drafter_warm && !proposed.is_empty() {
            self.reply
                .draft_cost
                .record(started.elapsed().as_secs_f64() * 1000.0 / proposed.len() as f64);
        }
        self.drafter_warm = true;
        self.growing = true;
        // Each proposal is a row of the drafter's tree already, made on the
        // tree as the ops left it — this one — in this order: every one is
        // pushed here too, and no op says so. A refusal would leave the two
        // trees apart, so it ends the flow.
        let pushed = proposed.len() as u32;
        for (parent, token) in proposed {
            self.flow
                .push(usize::from(parent), token)
                .map_err(|e| SwarmError::Internal(format!("a drafter's proposal: {e}")))?;
            self.draws.push(None);
            self.copied.push(false);
        }
        *self.reply.proposed += pushed;
        Ok(pushed > 0)
    }

    /// Cut the tree to the walk's row and its followers, and send the forward
    /// that cuts every cache to the old start and reads the path and the kept
    /// rows there. Answers still out from before are mapped onto the rows
    /// kept as they come in.
    async fn rebase(&mut self) -> Result<(), SwarmError> {
        let old_len = self.flow.len();
        let rebased = self.flow.rebase().map_err(SwarmError::Internal)?;
        self.ops.push(FlowOp::Rebase);
        let mut new_index = vec![None; old_len];
        for (new, &old) in rebased.kept.iter().enumerate() {
            new_index[old] = Some(new);
        }
        self.draws = rebased.kept.iter().map(|&old| self.draws[old]).collect();
        self.copied = rebased.kept.iter().map(|&old| self.copied[old]).collect();
        self.remap = Some((self.epoch, new_index));
        self.epoch += 1;
        self.rebases += 1;
        let (tokens, wire) = self.flow.rebase_forward(&rebased);
        let rows = Rows {
            index_pos: rebased.old_base,
            tokens,
            wire,
            from: rebased.old_base,
            truncate: Some(rebased.old_base as u32),
            lead: rebased.path.len(),
            first_row: 0,
            rebases_to: Some(self.flow.base()),
        };
        let forward = self.send(rows).await?;
        self.out.push(forward);
        self.unsettled = true;
        Ok(())
    }

    /// The miss held while a rebase was unsettled (`bonus_waiting`): its row,
    /// after a rebase if the tree is still full, and the walk on it.
    async fn push_waiting_bonus(&mut self) -> Result<(), SwarmError> {
        let Some(token) = self.bonus_waiting.take() else {
            return Ok(());
        };
        if self.flow.len() + 1 > MAX_TREE_ROWS {
            self.rebase().await?;
        }
        self.push_bonus(token)
    }

    /// Grow a copied run under the walk's row: what followed the reply's last
    /// tokens where they occurred before, in the prompt or the reply
    /// (`ngram_lookup::copied_run`, the n-gram lookup's own cascade), through
    /// the rows the tree already holds and new rows for the rest — sent with
    /// the next forward, before the drafter's next call. The run is as long as
    /// its match goes back, at least the lookup's own width and at most
    /// `COPY_LEAD_MOST` (SuffixDecoding's α·p, α = 1): a copy the walk keeps
    /// following runs further ahead. Looked up once a walk position; `true`
    /// when rows were added. The drafter rates each copied row by its own
    /// guess (`FlowOp::Copy`), so a wrong copy's followers do not crowd out
    /// its own.
    fn copy_ahead(&mut self, room: usize) -> bool {
        let Some(cfg) = self.copy else {
            return false;
        };
        let at = self.flow.at();
        if room == 0 || self.bonus_waiting.is_some() || self.copied_from == Some((self.epoch, at)) {
            return false;
        }
        self.copied_from = Some((self.epoch, at));
        let seq = self.drafter.tokens();
        // The walk's row carries the reply's last token: the run follows it.
        if seq.last() != Some(&self.flow.tokens()[at]) {
            return false;
        }
        let prompt_len = seq.len().saturating_sub(self.reply.generated.len());
        let Some(run) = crate::inference::ngram_lookup::copied_run(
            seq,
            prompt_len,
            super::speculative::NGRAM_RECENT_REPLY,
            cfg,
            COPY_LEAD_MOST,
        ) else {
            return false;
        };
        let lead = run.matched.max(cfg.num_pred_tokens).min(COPY_LEAD_MOST);
        // A row the drafter has no embedding for would end its tree.
        let reads = self.drafter.reads();
        let usable = run
            .tokens
            .iter()
            .take(lead)
            .take_while(|&&t| t < reads)
            .count();
        let (mut parent, new) = copy_under(&self.flow, at, &run.tokens[..usable], room);
        let mut added = 0;
        for &token in new {
            let Ok(row) = self.flow.push(parent, token) else {
                break;
            };
            self.draws.push(None);
            self.copied.push(true);
            self.ops.push(FlowOp::Copy {
                parent: parent as u16,
                token,
            });
            parent = row;
            added += 1;
        }
        if added > 0 {
            self.copied_runs += 1;
            self.copied_rows += added;
        }
        added > 0
    }

    /// The copied rows on the walk's path after `from`, down to `to`.
    fn copied_between(&self, from: usize, to: usize) -> u32 {
        let mut copied = 0;
        let mut row = to;
        while row != from {
            copied += u32::from(self.copied.get(row).copied().unwrap_or(false));
            match self.flow.tree().parent(row) {
                Some(parent) => row = parent,
                None => break,
            }
        }
        copied
    }

    /// The draw no row carried, a row of its own under the walk's row, and the
    /// walk on it; sent with the drafter's next rows after it.
    fn push_bonus(&mut self, token: u32) -> Result<(), SwarmError> {
        let at = self.flow.at();
        let row = self.flow.push(at, token).map_err(SwarmError::Internal)?;
        self.draws.push(None);
        self.copied.push(false);
        self.ops.push(FlowOp::Push {
            parent: at as u16,
            token,
        });
        self.advance(row)
    }

    /// The rows not yet sent, after those every segment holds.
    fn rows_unsent(&self) -> Rows {
        let base = self.flow.base();
        Rows {
            index_pos: base + self.sent,
            tokens: self.flow.tokens()[self.sent..].to_vec(),
            wire: self.flow.tree().to_wire(),
            from: base,
            truncate: None,
            lead: 0,
            first_row: self.sent,
            rebases_to: None,
        }
    }

    /// Run this node's layers over `rows`, send them on to the far segment,
    /// and hand back the wait on its draws.
    async fn send(&mut self, rows: Rows) -> Result<Waiting, SwarmError> {
        let exec = self.exec;
        let request_id = exec.request.id;
        let sampling = &exec.request.sampling_params;
        let alone = self.out.is_empty();
        let started = std::time::Instant::now();
        let seq = stream_seq::compose(self.attempt, self.turn).ok_or_else(|| {
            SwarmError::Internal("a tree that flows ran past its forwards' numbering".into())
        })?;
        self.turn += 1;
        self.sent = self.flow.len();
        self.forwards += 1;
        self.rows_sent += rows.tokens.len() as u32;
        // Each forward is counted a round (FUTURE_WORK #3 (d)), so a flow's
        // yield reads lower than it is, never higher.
        self.state.note_decode_round(request_id);
        let index_pos = rows.index_pos as u32;
        let count = rows.tokens.len();
        let tree = |forward: &mut crate::types::LayerForward| {
            forward.draft_parents = rows.wire.clone();
            forward.draft_tree_from = Some(rows.from as u32);
        };

        let mut activations = super::pack_verify_tokens_to_le_bytes(&rows.tokens);
        let near_started = std::time::Instant::now();
        for segment in &self.shape.near {
            let mut forward = super::build_spec_verify_forward(
                request_id,
                index_pos,
                activations,
                segment,
                rows.truncate,
                None,
                Some(seq),
            );
            tree(&mut forward);
            let result = self
                .state
                .model_process_pool
                .forward_for_request(forward, None, super::worker_requester(&exec.request))
                .await?;
            if let Some(NetworkFinishReason::Error(msg)) = &result.finish_reason {
                return Err(
                    crate::error::reclassify_flattened_error(msg).unwrap_or_else(|| {
                        SwarmError::Inference(format!("a tree that flows: {msg}"))
                    }),
                );
            }
            activations = result.activations;
        }
        self.near_ms += near_started.elapsed().as_secs_f64() * 1000.0;

        let walk = TailWalk {
            drafts: &[],
            sampling,
            generated: &[],
            coupling: self.coupling,
            parents: &[],
        };
        let far = &self.shape.far;
        let mut forward = super::build_spec_verify_forward(
            request_id,
            index_pos,
            activations,
            far,
            rows.truncate,
            Some(&walk),
            Some(seq),
        );
        tree(&mut forward);
        if !self.shape.chain.is_empty() {
            forward.chain = self.shape.chain.clone();
            forward.requester_node_id = Some(self.state.identity.node_id().0);
        }
        let Some(peer) = self.state.resolve_peer_id_bytes(&far.node_id) else {
            return Err(SwarmError::Inference(format!(
                "a tree that flows: no route to segment holder {} — it left mid-request",
                far.node_id
            )));
        };
        if self.state.pending_layer_results.len() >= super::MAX_PENDING_LAYER_RESULTS {
            return Err(SwarmError::ServiceUnavailable(
                "Pipeline overloaded — too many pending layer results".into(),
            ));
        }
        let key = WaiterKey::streamed(request_id, seq);
        let (tx, rx) = tokio::sync::oneshot::channel();
        let chain_members: Vec<crate::types::NodeId> = if self.shape.chain.is_empty() {
            Vec::new()
        } else {
            std::iter::once(far.node_id.clone())
                .chain(self.shape.chain.iter().map(|h| h.node_id.clone()))
                .collect()
        };
        self.state.pending_layer_results.insert(
            key,
            PendingLayerResult {
                tx,
                awaiting: Some(self.shape.tail().clone()),
                chain_members,
                expects_step: Some(ExpectedStep {
                    index_pos,
                    layer_ranges: self.shape.far_ranges(),
                }),
            },
        );
        let waiter = StreamWaiter {
            state: self.state.clone(),
            key,
        };
        let activation_bytes = forward.activations.len();
        if exec
            .network_tx
            .send(NetworkCommand::SendTensor {
                target_peer_bytes: peer.clone(),
                forward: forward.clone(),
            })
            .await
            .is_err()
        {
            return Err(SwarmError::Network(
                "a tree that flows: send dropped".into(),
            ));
        }
        let num_layers = self
            .shape
            .far_ranges()
            .iter()
            .map(|(a, b)| b - a)
            .sum::<u32>();
        let budget = SegmentBudget::for_forward(
            &self.state,
            &far.node_id,
            &far.shard_id.model_id,
            crate::daemon::state::WorkKind::Decode,
            num_layers,
            activation_bytes,
            ActivationUnits::HiddenStates,
        );
        let sent = Sent {
            seq,
            epoch: self.epoch,
            lead: rows.lead,
            first_row: rows.first_row,
            rows: count,
            alone,
            started,
            rebases_to: rows.rebases_to,
        };
        let state = self.state.clone();
        let network_tx = exec.network_tx.clone();
        let node = far.node_id.clone();
        let cancel = exec.request.cancel.clone();
        let segment_idx = self.shape.near.len();
        let chained = !self.shape.chain.is_empty();
        let waiting = tokio::spawn(async move {
            let _waiter = waiter;
            let rebuild = move || forward.clone();
            let result = PipelineExecutor::wait_for_result(
                &state,
                rx,
                request_id,
                segment_idx,
                &node,
                num_layers,
                activation_bytes,
                budget,
                cancel.as_ref(),
                if chained {
                    ResendOnRefusal::Never(
                        "a chained check's refusal may be any hop's, and the run is not re-sent",
                    )
                } else {
                    ResendOnRefusal::SameForward {
                        network_tx: &network_tx,
                        target_peer_bytes: &peer,
                        rebuild: &rebuild,
                    }
                },
            )
            .await;
            (result, std::time::Instant::now())
        });
        let arrival = Arrival(waiting.abort_handle());
        Ok(Box::pin(async move {
            let _arrival = arrival;
            match waiting.await {
                Ok((result, arrived)) => (sent, result, arrived),
                Err(e) => (
                    sent,
                    Err(SwarmError::Internal(format!(
                        "a tree that flows: a forward's wait ended without an answer: {e}"
                    ))),
                    std::time::Instant::now(),
                ),
            }
        }))
    }

    /// Record a forward's draws, walk them, emit what the walk reached, and
    /// grow from where it stands.
    async fn take_answer(&mut self, (sent, result, arrived): Answer) -> Result<Taken, SwarmError> {
        self.answers += 1;
        if fault_flow_fail_at() == Some(self.answers) {
            tracing::warn!(
                request_id = %self.exec.request.id,
                answers = self.answers,
                "FAULT: failing this flow (SWARMLLM_FAULT_FLOW_FAIL)"
            );
            return Err(SwarmError::Inference(
                "a tree that flows: failed on demand".into(),
            ));
        }
        let current = sent.epoch == self.epoch;
        let refused = match &result {
            Err(e) => Some(e.to_string()),
            Ok(r) => match &r.finish_reason {
                Some(NetworkFinishReason::Error(msg)) => Some(msg.clone()),
                _ => None,
            },
        };
        if let Some(why) = refused {
            if !current {
                // A turn the rebase skipped, or one whose rows it read again.
                self.stale_answers += 1;
                tracing::debug!(
                    request_id = %self.exec.request.id,
                    turn = stream_seq::turn(sent.seq),
                    why,
                    "DSD: a tree that flows dropped an answer from before its rebase"
                );
                return Ok(Taken::Continue);
            }
            return Err(match result {
                Err(e) => e,
                Ok(_) => crate::error::reclassify_flattened_error(&why)
                    .unwrap_or_else(|| SwarmError::Inference(format!("a tree that flows: {why}"))),
            });
        }
        let result = result.expect("an error answer returned above");
        if result.token_ids.len() != sent.rows {
            if !current {
                self.stale_answers += 1;
                return Ok(Taken::Continue);
            }
            return Err(SwarmError::Inference(format!(
                "a tree that flows: {} rows answered with {} draws",
                sent.rows,
                result.token_ids.len()
            )));
        }
        if let Some(base) = sent.rebases_to {
            // Answered by the tail, so read by every segment before it.
            self.settled_base = base;
            self.unsettled = false;
        }
        let remap = match (&self.remap, current) {
            (_, true) => None,
            (Some((epoch, map)), false) if *epoch + 1 == self.epoch && *epoch == sent.epoch => {
                Some(map)
            }
            _ => {
                self.stale_answers += 1;
                return Ok(Taken::Continue);
            }
        };
        for (i, &draw) in result.token_ids.iter().enumerate().skip(sent.lead) {
            let row = sent.first_row + i - sent.lead;
            let row = match remap {
                None => Some(row),
                Some(map) => map.get(row).copied().flatten(),
            };
            if let Some(slot) = row.and_then(|r| self.draws.get_mut(r)) {
                // A row's draw does not depend on the forward that carried it:
                // the first one in stands.
                slot.get_or_insert(draw);
            }
        }
        // When it ARRIVED: the time it then sat while this node drafted or ran
        // its layers is neither the check's cost nor the link's queue.
        let delay_ms = arrived
            .saturating_duration_since(sent.started)
            .as_secs_f64()
            * 1000.0;
        if sent.alone && current && sent.lead == 0 {
            self.reply.check.record(sent.rows as u32, delay_ms);
        }
        if sent.lead == 0 {
            // A rebase's forward carries the path too: its time is its size's.
            let before = self.guard.at();
            self.guard
                .answered(sent.started, delay_ms, std::time::Instant::now());
            let after = self.guard.at();
            if after.width < before.width || after.live < before.live {
                tracing::debug!(
                    request_id = %self.exec.request.id,
                    from = format_args!("{}x{}", before.width, before.live),
                    to = format_args!("{}x{}", after.width, after.live),
                    delay_ms = format_args!("{delay_ms:.0}"),
                    least_ms = format_args!("{:.0}", self.guard.least_delay_ms.unwrap_or(0.0)),
                    "DSD: a tree that flows grows less ahead — its forwards queue on the way"
                );
            }
        }
        if self.bonus_waiting.is_some() {
            // The walk already emitted the miss's token; it goes on from the
            // bonus's row once that is pushed (`push_waiting_bonus`).
            return Ok(Taken::Continue);
        }
        self.walk().await
    }

    /// Walk the draws from the walk's row, emit, and move the tree.
    async fn walk(&mut self) -> Result<Taken, SwarmError> {
        let (emitted, end) = walk_draws(
            self.flow.tree(),
            self.flow.tokens(),
            &self.draws,
            self.flow.at(),
        );
        if emitted.is_empty() {
            return Ok(Taken::Continue);
        }
        let (WalkEnd::Waiting { row: reached } | WalkEnd::Missed { row: reached, .. }) = end;
        self.copied_walked += self.copied_between(self.flow.at(), reached);
        let carried = emitted.len() - usize::from(matches!(end, WalkEnd::Missed { .. }));
        *self.reply.accepted += carried as u32;
        self.reply
            .acceptance
            .record(carried as u32, emitted.len() as u32);
        let room = (self.io.max_tokens as usize).saturating_sub(self.reply.generated.len());
        let mut reply_tokens = emitted;
        if let Some(at) = reply_tokens.iter().position(|t| self.io.eos.contains(t)) {
            reply_tokens.truncate(at + 1);
        }
        reply_tokens.truncate(room.max(1));
        super::emit_streaming_batch(
            &self.exec.partial_reply,
            self.io.token_tx,
            self.io.decoder,
            &reply_tokens,
            self.io.eos,
            self.reply.finish_reason,
        )
        .await;
        if !self.reply.finish_reason.is_empty() {
            return Ok(Taken::Done);
        }
        self.reply.generated.extend_from_slice(&reply_tokens);
        self.drafter.push(&reply_tokens);
        let last = *reply_tokens
            .last()
            .expect("a walk emits at least one token");
        if self.io.eos.contains(&last) {
            *self.reply.finish_reason = "stop".to_string();
            return Ok(Taken::Done);
        }
        if self.reply.generated.len() as u32 >= self.io.max_tokens {
            return Ok(Taken::Done);
        }
        match end {
            WalkEnd::Waiting { row } => self.advance(row)?,
            WalkEnd::Missed { row, token } => {
                self.advance(row)?;
                self.misses += 1;
                if self.flow.len() + 1 > MAX_TREE_ROWS {
                    if self.unsettled {
                        self.bonus_waiting = Some(token);
                        return Ok(Taken::Continue);
                    }
                    self.rebase().await?;
                }
                self.push_bonus(token)?;
            }
        }
        Ok(Taken::Continue)
    }

    fn advance(&mut self, row: usize) -> Result<(), SwarmError> {
        if row != self.flow.at() {
            self.flow.advance(row).map_err(SwarmError::Internal)?;
            self.ops.push(FlowOp::Advance { row: row as u16 });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};

    fn at(t0: Instant, ms: u64) -> Instant {
        t0 + Duration::from_millis(ms)
    }

    fn feed(g: &mut FlowGuard, t0: Instant, sent_ms: u64, delay: f64) {
        g.answered(at(t0, sent_ms), delay, at(t0, sent_ms + delay as u64));
    }

    /// A queue past the target narrows the levels first, to the least width,
    /// then the live budget, to its floor — at most by half, once for each
    /// forward sent after the last cut (one already on its way shows the
    /// queue the cut has not acted on); one late answer among four is not a
    /// queue.
    #[test]
    fn a_queue_narrows_the_levels_first_then_the_live_budget() {
        let t0 = Instant::now();
        let mut g = FlowGuard::new(8, 128, true, None);
        for i in 0..4 {
            feed(&mut g, t0, i * 10, 400.0);
        }
        assert_eq!(
            g.at(),
            GuardAt {
                width: 8,
                live: 128
            },
            "no queue"
        );
        // One answer 300 ms late among on-time ones: the least of four is on
        // time.
        feed(&mut g, t0, 50, 700.0);
        assert_eq!(
            g.at(),
            GuardAt {
                width: 8,
                live: 128
            },
            "one late answer"
        );
        // Every answer 200 ms late: the width halves (200 / 60 − 1 > 1/2).
        for i in 0..4 {
            feed(&mut g, t0, 1000 + i * 10, 600.0);
        }
        assert_eq!(
            g.at(),
            GuardAt {
                width: 4,
                live: 128
            }
        );
        assert_eq!(g.cuts, 1);
        // Sent before that cut: the queue it shows is the one already cut for.
        feed(&mut g, t0, 1500, 600.0);
        assert_eq!(
            g.at().width,
            4,
            "an answer sent before the cut cuts nothing"
        );
        // Sent after it and still late: narrower, then the budget.
        feed(&mut g, t0, 1700, 600.0);
        assert_eq!(
            g.at(),
            GuardAt {
                width: 2,
                live: 128
            }
        );
        feed(&mut g, t0, 2400, 600.0);
        assert_eq!(g.at(), GuardAt { width: 2, live: 64 });
        feed(&mut g, t0, 3100, 600.0);
        feed(&mut g, t0, 3800, 600.0);
        assert_eq!(
            g.at(),
            GuardAt {
                width: 2,
                live: LIVE_LEAST
            },
            "never below the floors"
        );
        assert_eq!((g.least_width, g.least_live, g.cuts), (2, LIVE_LEAST, 4));
        // A queue just past the target cuts a little: 75 ms is a quarter past
        // 60, so 3/4 of 8 rows a level.
        let mut g = FlowGuard::new(8, 128, true, None);
        for i in 0..4 {
            feed(&mut g, t0, i * 10, 400.0);
        }
        for i in 0..4 {
            feed(&mut g, t0, 1000 + i * 10, 475.0);
        }
        assert_eq!(
            g.at(),
            GuardAt {
                width: 6,
                live: 128
            }
        );
    }

    /// Below the target the budget grows back first — by up to a level's rows
    /// a round trip — then the levels, by up to a row; no further than the
    /// knobs; unguarded nothing moves.
    #[test]
    fn under_the_target_the_budget_grows_back_first_then_the_levels() {
        let t0 = Instant::now();
        let kept = GuardAt { width: 2, live: 32 };
        let mut g = FlowGuard::new(8, 128, true, Some(kept));
        assert_eq!(g.at(), kept, "a kept guard is where a reply starts");
        for i in 0..4 {
            feed(&mut g, t0, i * 10, 400.0);
        }
        assert_eq!(
            g.at(),
            GuardAt { width: 2, live: 40 },
            "a level's rows of budget"
        );
        // Within a round trip (the least delay, 400 ms) of that rise: none.
        feed(&mut g, t0, 100, 400.0);
        assert_eq!(g.at().live, 40);
        // Half the target's queue rises by half: once the least of the four
        // latest shows it and a round trip has passed.
        for ms in [600u64, 700, 800] {
            g.answered(at(t0, ms - 430), 430.0, at(t0, ms));
        }
        assert_eq!(g.at().live, 40, "within a round trip of the last rise");
        g.answered(at(t0, 900 - 430), 430.0, at(t0, 900));
        assert_eq!(g.at().live, 44);
        let mut ms = 900;
        while g.at().live < 128 {
            ms += 450;
            g.answered(at(t0, ms - 400), 400.0, at(t0, ms));
        }
        assert_eq!(g.at().width, 2, "the levels wait for the whole budget");
        for _ in 0..3 {
            ms += 450;
            g.answered(at(t0, ms - 400), 400.0, at(t0, ms));
        }
        assert_eq!(
            g.at(),
            GuardAt {
                width: 5,
                live: 128
            },
            "then a row a round trip"
        );
        for _ in 0..20 {
            ms += 450;
            g.answered(at(t0, ms - 400), 400.0, at(t0, ms));
        }
        assert_eq!(
            g.at(),
            GuardAt {
                width: 8,
                live: 128
            },
            "no further than the knobs"
        );
        // A kept guard past the knobs, or under the floors, is clamped; a
        // knob below a floor is the floor.
        let wide = GuardAt {
            width: 20,
            live: 300,
        };
        assert_eq!(
            FlowGuard::new(8, 64, true, Some(wide)).at(),
            GuardAt { width: 8, live: 64 }
        );
        let thin = GuardAt { width: 1, live: 3 };
        assert_eq!(
            FlowGuard::new(8, 64, true, Some(thin)).at(),
            GuardAt { width: 2, live: 32 }
        );
        assert_eq!(
            FlowGuard::new(1, 16, true, None).at(),
            GuardAt { width: 1, live: 16 }
        );
        // Unguarded (`SWARMLLM_FLOW_GUARD=0`): the knobs, whatever the queue.
        let mut g = FlowGuard::new(8, 128, false, Some(kept));
        assert_eq!(
            g.at(),
            GuardAt {
                width: 8,
                live: 128
            }
        );
        for i in 0..8 {
            feed(&mut g, t0, i * 10, 400.0 + 300.0 * i as f64);
        }
        assert_eq!(
            g.at(),
            GuardAt {
                width: 8,
                live: 128
            }
        );
        assert_eq!(g.cuts, 0);
    }

    #[test]
    fn a_copied_run_goes_through_the_rows_the_tree_holds_and_adds_the_rest() {
        // The walk's row 0 carries 5; the drafter grew 6 then 7 after it, and 9.
        let mut flow = FlowTree::new(100, 5).unwrap();
        let six = flow.push(0, 6).unwrap();
        let seven = flow.push(six, 7).unwrap();
        flow.push(0, 9).unwrap();
        // The run 6 7 8 4 goes through rows 6 and 7, and adds 8 and 4 after 7.
        assert_eq!(
            copy_under(&flow, 0, &[6, 7, 8, 4], 10),
            (seven, &[8u32, 4][..])
        );
        // At most `room` new rows.
        assert_eq!(copy_under(&flow, 0, &[6, 7, 8, 4], 1), (seven, &[8u32][..]));
        // A run the tree holds whole adds nothing.
        assert_eq!(copy_under(&flow, 0, &[6, 7], 10), (seven, &[][..]));
        // One the tree does not start is every row of it, after the walk's.
        assert_eq!(copy_under(&flow, 0, &[3, 6], 10), (0, &[3u32, 6][..]));
        // From a row further down, only what follows that row counts.
        assert_eq!(copy_under(&flow, six, &[9, 1], 10), (six, &[9u32, 1][..]));
    }

    #[test]
    fn the_rounds_take_a_failed_flow_over_from_the_trees_start() {
        // A 10-token prompt; the reply is at positions 10..=15, the tree
        // started at 12 (after a rebase): the caches are cut to 12 and the
        // tokens at 12..=14 are read again before the check's own, at 15.
        let reply = [100, 101, 102, 103, 104, 105];
        assert_eq!(
            rounds_take_over(10, &reply, 12),
            RoundsTakeOver {
                reread: vec![102, 103, 104],
                kv_len: 12,
            }
        );
        // Failed at once: the tree starts at the reply's last token, nothing
        // is read again.
        assert_eq!(
            rounds_take_over(10, &reply, 15),
            RoundsTakeOver {
                reread: vec![],
                kv_len: 15,
            }
        );
        // The first tree, from the prompt pass's token.
        assert_eq!(
            rounds_take_over(10, &reply, 10),
            RoundsTakeOver {
                reread: vec![100, 101, 102, 103, 104],
                kv_len: 10,
            }
        );
    }
}
