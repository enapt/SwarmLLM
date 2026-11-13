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
//! **A run that finishes here flows too** (#369): where this node holds the
//! model's last layers — "start and finish on this computer", automatic
//! wherever it holds both ends — the peers are sent hidden states and the
//! tree's shape alone and answer hidden states; this node's layers after
//! theirs read each answer in the order its forward was SENT (`TurnOrder`,
//! the order every peer ran them) and the last of them draws every row
//! (`FlowRun::finish_here`). The peers never see a draw. That shape ran tree
//! rounds before, one check a round trip: 1.6-2.2x slower than the flow at
//! the same split on TH↔AU (2026-10-29). Answers that are due together —
//! they land in bursts — are read here in ONE forward
//! (`FlowRun::due_together`, `tensor_util::join_positions`): this shape is
//! bound by this node's cost a forward, not by the link. And
//! (`dead_wait_on`, +17% on prose) an answer none of whose rows the walk can
//! still reach waits to be read with the next one it can, or is never read
//! where a rebase cuts this node's cache below it first.
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

/// Whether the guard takes the far computer's own time off an answer's delay
/// (#388, [`believed_served_ms`]): ON unless `SWARMLLM_FLOW_SERVED_TIME=0`
/// (or `off` / `false`), which reads every delay whole, as before. Measured
/// where #388 came from, a card far on the real link (2026-11-06, TH↔AU,
/// ABBA × 3): prose 14.8 against 11.7 tok/s, code 32.7 against 30.1 — read
/// whole, the far's own busy spells cut the tree to 2 rows a level 18-24
/// times a prose reply. It lost only on a processor-only rig, where every
/// row costs this node's drafter and layers too (`docs/invariants/
/// scheduling.md` § "A far computer's own time is not the link's"). A far
/// older than `features::SERVED_TIME` reports nothing, and its delays are
/// read whole.
fn served_time_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        !matches!(
            std::env::var("SWARMLLM_FLOW_SERVED_TIME").ok().as_deref(),
            Some("0") | Some("off") | Some("false")
        )
    })
}

/// The far computer's own time to take off an answer's delay, in ms — where
/// it may be believed. It is a number the far chose, so it is believed only
/// where what it leaves of the delay stays at or above the least round trip
/// measured to that computer (`floor_ms`): RFC 9002 § 5.3's rule for a
/// peer's `ack_delay`, which "MUST NOT" be subtracted where the result would
/// fall under `min_rtt`. A far claiming more is read as claiming nothing —
/// the whole delay, as from a far that says nothing. With no floor measured
/// (a fresh node), nothing is believed. A far can still claim up to the
/// floor's worth of a queue as its own and keep a link's queue from cutting
/// its flow; that slows only the reply it is already serving, which it could
/// do by being slow.
fn believed_served_ms(delay_ms: f64, served_us: Option<u32>, floor_ms: Option<f64>) -> Option<f64> {
    let served_ms = f64::from(served_us?) / 1000.0;
    (delay_ms - served_ms >= floor_ms?).then_some(served_ms)
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

/// Who a flow's rows reach (`owner_replies::Exposure`) — a copied run it would
/// send and the reply it keeps alike: every peer of the run, and as token ids
/// only the first, where this node runs none of the first layers (`send` packs
/// the rows as ids for it). The tail is sent no ids (`TailWalk::drafts` is
/// empty): it draws on hidden states — and where this node's layers after
/// the run draw (`Here`), no peer draws at all.
fn flow_exposure(shape: &StreamShape) -> crate::inference::owner_replies::Exposure {
    crate::inference::owner_replies::Exposure::new(
        std::iter::once(shape.far.node_id.clone())
            .chain(shape.chain.iter().map(|h| h.node_id.clone())),
        shape.near.is_empty().then(|| shape.far.node_id.clone()),
    )
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
/// **Two windows, the smaller of them** (#388) — Swift's split (Kumar et al.,
/// SIGCOMM 2020 § 3.3: a fabric window and an endpoint window, the effective
/// one their minimum). Where the far computer reports its own time
/// ([`believed_served_ms`]), the LINK's window reads an answer's delay less
/// it and the FAR's window reads it alone; from a far that says nothing the
/// link's reads the whole delay, as before. On one delay the guard took the
/// far's own slow spells for a queue on the link: a far slow for 12 s and for
/// 37 s (TH↔AU, 2026-11-01) cut to 2 rows a level, and those replies were
/// their runs' slowest, though fewer rows could not shorten the far's time
/// (forwards cut 130 → 33 KB left its spikes as long, gotcha #193). Swift
/// cuts its endpoint window on any endpoint delay — in a datacenter that
/// queue is the flows' own. A far here is shared with other people's work, so
/// its window cuts only where a cut SHORTENS the far's queue: each far cut is
/// a [`FarTrial`] — it keeps [`FAR_CUT_KEEPS`], and stands only if the far's
/// queue on forwards sent after it falls to [`FAR_CUT_SHORTENED`] of what it
/// was; else it is undone and the far's window holds until the far's queue is
/// back under the target. The far's window cuts the live BUDGET first
/// ([`Window::cut_budget_first`]): a queue our flow builds at the far is our
/// forwards waiting there, and a smaller budget is fewer drafter calls out at
/// once — a forward each — which shortens it whatever a forward costs a row.
/// Fewer rows a level did not: on a processor far (2026-11-01 rig, half a 7B
/// a forward ~350 ms) the far's time reached 5-9 s of our own forwards, and
/// 7 of 9 width cuts were undone as "not ours" while the flow kept them all
/// out.
///
/// Measured (`guardrig2.sh`, 2026-10-22, 7B split at 385 ms, warm replies):
/// at 5 Mbit/s 7.44-7.50 tok/s against 4.73-4.80 unguarded — past every fixed
/// setting and the tree rounds (6.8-7.1) —, at 10 Mbit/s 8.52-8.82 against
/// 8.23, and with no rate limit no cut but in a burst of the far half's own
/// queue (it shares this rig's one card).
#[derive(Debug, Clone)]
struct FlowGuard {
    width_most: u32,
    live_most: u32,
    guarded: bool,
    /// Held down by a queue on the way, both ways: each answer's delay less
    /// the far's own time, or the whole delay where the far says nothing.
    link: Window,
    link_delays: Delays,
    /// The last link cut: a later one needs an answer to a forward sent after it.
    cut_at: Option<std::time::Instant>,
    /// The link window's last change, cut or rise: a rise waits a round trip.
    changed_at: Option<std::time::Instant>,
    /// Held down only by a queue at the far that this flow's own rows build.
    far: Window,
    far_delays: Delays,
    far_changed_at: Option<std::time::Instant>,
    far_trial: Option<FarTrial>,
    /// A far cut did not shorten the far's queue: no other until that queue is
    /// back under the target.
    far_not_ours: bool,
    /// Cuts of the link's window.
    cuts: u32,
    least_width: u32,
    least_live: u32,
    queue_ms_most: f64,
    /// Answers whose far time was believed and read.
    served_read: u32,
    far_queue_ms_most: f64,
    far_cuts: u32,
    far_undone: u32,
}

/// Rows a level and live rows, fractional so a small rise accumulates.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Window {
    width: f64,
    live: f64,
}

impl Window {
    fn at(width: u32, live: u32) -> Self {
        Self {
            width: f64::from(width),
            live: f64::from(live),
        }
    }

    /// Keep `keep` of the rows a level, or — once they are at their least —
    /// of the live budget.
    fn cut(&mut self, keep: f64, width_least: f64, live_least: f64) {
        if self.width > width_least {
            self.width = (self.width * keep).max(width_least);
        } else {
            self.live = (self.live * keep).max(live_least);
        }
    }

    /// The far's cut (#388): `keep` of the live budget, or — once it is at
    /// its least — of the rows a level. The budget bounds the forwards out
    /// (each drafter call's rows leave as one), and fewer forwards out is what
    /// shortens a queue they build at the far, whatever a row costs there.
    fn cut_budget_first(&mut self, keep: f64, width_least: f64, live_least: f64) {
        if self.live > live_least {
            self.live = (self.live * keep).max(live_least);
        } else {
            self.width = (self.width * keep).max(width_least);
        }
    }

    /// `room` of a level's rows back on the live budget, or — once it is
    /// whole — of a row on the levels.
    fn rise(&mut self, room: f64, width_most: u32, live_most: u32) {
        if self.live < f64::from(live_most) {
            self.live = (self.live + f64::from(width_most) * room).min(f64::from(live_most));
        } else {
            self.width = (self.width + room).min(f64::from(width_most));
        }
    }
}

/// Delays as LEDBAT reads them: the least seen (the base) and the latest
/// [`DELAY_SAMPLES`].
#[derive(Debug, Clone, Default)]
struct Delays {
    least: Option<f64>,
    recent: std::collections::VecDeque<f64>,
}

impl Delays {
    /// Take `ms`; the queue — the least of the latest samples over the least
    /// seen — once there are [`DELAY_SAMPLES`] of them.
    fn queue_after(&mut self, ms: f64) -> Option<f64> {
        let base = self.least.map_or(ms, |b| b.min(ms));
        self.least = Some(base);
        if self.recent.len() == DELAY_SAMPLES {
            self.recent.pop_front();
        }
        self.recent.push_back(ms);
        (self.recent.len() == DELAY_SAMPLES)
            .then(|| self.recent.iter().copied().fold(f64::INFINITY, f64::min) - base)
    }
}

/// A far cut on trial (#388): it stands only if the far's queue shortens.
#[derive(Debug, Clone, Copy)]
struct FarTrial {
    /// Answers to forwards sent from here on judge it: by then the far has had
    /// the queue's worth of time to work off the forwards sent before the cut.
    judged_from: std::time::Instant,
    /// The far window before the cut — back, if the cut did not shorten it.
    before: Window,
    queue_before_ms: f64,
    /// Answers seen that judge it; the verdict waits for [`DELAY_SAMPLES`],
    /// so the queue it reads is theirs alone.
    answers: usize,
}

/// What a far cut keeps: half — enough that a queue our forwards build visibly
/// shortens, so the trial tells it from a far slow on its own.
const FAR_CUT_KEEPS: f64 = 0.5;

/// The far's queue on forwards a far cut sent lighter, against before it: at
/// most this, and the cut shortened it. A controller that moves X to hit a
/// target checks that moving X moves the target (gotcha #193: a chunk
/// quantum sized on a processor pinned a card at its floor, six times worse
/// than none). Half the budget is about half the forwards waiting behind a
/// queue they build, on a card (a forward's cost mostly per call) and on a
/// processor alike; a far slow on its own stays as slow — there the trial
/// undoes the cut.
const FAR_CUT_SHORTENED: f64 = 0.75;

/// Where a guard stands — what a reply leaves for the next on its run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct GuardAt {
    width: u32,
    live: u32,
}

fn ms_since(then: std::time::Instant, now: std::time::Instant) -> f64 {
    now.saturating_duration_since(then).as_secs_f64() * 1000.0
}

impl FlowGuard {
    /// `kept`, where the last reply on this run left its guard, starts the
    /// LINK's window: the far's starts whole, so a far that says nothing —
    /// whose window never moves — leaves the guard as it always was.
    fn new(width_most: u32, live_most: u32, guarded: bool, kept: Option<GuardAt>) -> Self {
        let full = Window::at(width_most, live_most);
        let mut guard = Self {
            width_most,
            live_most,
            guarded,
            link: full,
            link_delays: Delays::default(),
            cut_at: None,
            changed_at: None,
            far: full,
            far_delays: Delays::default(),
            far_changed_at: None,
            far_trial: None,
            far_not_ours: false,
            cuts: 0,
            least_width: width_most,
            least_live: live_most,
            queue_ms_most: 0.0,
            served_read: 0,
            far_queue_ms_most: 0.0,
            far_cuts: 0,
            far_undone: 0,
        };
        if let Some(kept) = kept.filter(|_| guarded) {
            let at = guard.snap(Window::at(kept.width, kept.live));
            guard.link = Window::at(at.width, at.live);
            guard.least_width = at.width;
            guard.least_live = at.live;
        }
        guard
    }

    /// A window in whole rows, within its bounds.
    fn snap(&self, window: Window) -> GuardAt {
        GuardAt {
            width: (window.width.round() as u32)
                .clamp(WIDTH_LEAST.min(self.width_most), self.width_most),
            live: (window.live.round() as u32)
                .clamp(LIVE_LEAST.min(self.live_most), self.live_most),
        }
    }

    /// Rows a level and the live budget, now: the smaller of the two windows.
    fn at(&self) -> GuardAt {
        self.snap(Window {
            width: self.link.width.min(self.far.width),
            live: self.link.live.min(self.far.live),
        })
    }

    /// What the next reply through the same machines starts from: the LINK's
    /// window alone. The far's window is what the far was doing this reply —
    /// learned again each reply — and a far cut still on trial must never
    /// outlive its trial as a link cut, which nothing would undo (security
    /// review of #388: a far's claims could ratchet the next replies down).
    fn to_keep(&self) -> GuardAt {
        self.snap(self.link)
    }

    fn width_least(&self) -> f64 {
        f64::from(WIDTH_LEAST.min(self.width_most))
    }

    fn live_least(&self) -> f64 {
        f64::from(LIVE_LEAST.min(self.live_most))
    }

    /// The least round trip: the link's least delay and the far's least time.
    fn least_round_trip_ms(&self) -> f64 {
        self.link_delays.least.unwrap_or(0.0) + self.far_delays.least.unwrap_or(0.0)
    }

    /// A forward of the tree's own rows (never a rebase's), sent at `sent`,
    /// answered `delay_ms` after it left — `served_ms` of it the far's own
    /// time, where that is believed ([`believed_served_ms`]) — read at `now`.
    fn answered(
        &mut self,
        sent: std::time::Instant,
        delay_ms: f64,
        served_ms: Option<f64>,
        now: std::time::Instant,
    ) {
        if !self.guarded || !delay_ms.is_finite() {
            return;
        }
        let served_ms = served_ms.filter(|s| s.is_finite() && (0.0..=delay_ms).contains(s));
        let link_cut = self.link_answered(sent, delay_ms - served_ms.unwrap_or(0.0), now);
        if let Some(far_ms) = served_ms {
            self.served_read += 1;
            self.far_answered(sent, far_ms, link_cut, now);
        }
        let at = self.at();
        self.least_width = self.least_width.min(at.width);
        self.least_live = self.least_live.min(at.live);
    }

    /// The link's window on one answer's delay less the far's time. Whether
    /// it cut.
    fn link_answered(
        &mut self,
        sent: std::time::Instant,
        link_ms: f64,
        now: std::time::Instant,
    ) -> bool {
        let Some(queue) = self.link_delays.queue_after(link_ms) else {
            return false;
        };
        self.queue_ms_most = self.queue_ms_most.max(queue);
        if queue > QUEUE_TARGET_MS {
            if self.cut_at.is_some_and(|c| sent <= c) {
                return false;
            }
            let keep = 1.0 - (queue / QUEUE_TARGET_MS - 1.0).min(0.5);
            let before = self.snap(self.link);
            let (width_least, live_least) = (self.width_least(), self.live_least());
            self.link.cut(keep, width_least, live_least);
            let cut = self.snap(self.link) != before;
            if cut {
                self.cuts += 1;
                self.cut_at = Some(now);
            }
            self.changed_at = Some(now);
            cut
        } else {
            if self
                .changed_at
                .is_some_and(|c| ms_since(c, now) < self.least_round_trip_ms())
            {
                return false;
            }
            let room = 1.0 - queue.max(0.0) / QUEUE_TARGET_MS;
            self.link.rise(room, self.width_most, self.live_most);
            self.changed_at = Some(now);
            false
        }
    }

    /// The far's window on one answer's far time (`far_ms`). An answer the
    /// link's window cut on cuts nothing more.
    fn far_answered(
        &mut self,
        sent: std::time::Instant,
        far_ms: f64,
        link_cut: bool,
        now: std::time::Instant,
    ) {
        let Some(queue) = self.far_delays.queue_after(far_ms) else {
            return;
        };
        self.far_queue_ms_most = self.far_queue_ms_most.max(queue);
        if queue <= QUEUE_TARGET_MS {
            // A cut on trial stands; the far may be tried again.
            self.far_trial = None;
            self.far_not_ours = false;
            if self
                .far_changed_at
                .is_some_and(|c| ms_since(c, now) < self.least_round_trip_ms())
            {
                return;
            }
            let room = 1.0 - queue.max(0.0) / QUEUE_TARGET_MS;
            self.far.rise(room, self.width_most, self.live_most);
            self.far_changed_at = Some(now);
            return;
        }
        if self.far_not_ours {
            return;
        }
        if let Some(trial) = self.far_trial.as_mut() {
            if sent < trial.judged_from {
                return;
            }
            trial.answers += 1;
            if trial.answers < DELAY_SAMPLES {
                return;
            }
            if queue > trial.queue_before_ms * FAR_CUT_SHORTENED {
                // Fewer rows did not shorten it: the far is slow on its own.
                self.far = trial.before;
                self.far_trial = None;
                self.far_not_ours = true;
                self.far_undone += 1;
                self.far_changed_at = Some(now);
                return;
            }
        }
        if link_cut {
            return;
        }
        // Ours, or not yet tried: a cut on trial — of the window the flow runs
        // at, so it binds. Cut from its own, a far window above the link's
        // changed nothing and was then judged on a queue it never touched:
        // "not ours", and no other far cut while the far stayed slow (rig,
        // 2026-11-01: `2x32 → 2x32`, then undone).
        let before = self.far;
        let running = self.at();
        let (width_least, live_least) = (self.width_least(), self.live_least());
        self.far = Window {
            width: self.far.width.min(self.link.width),
            live: self.far.live.min(self.link.live),
        };
        self.far
            .cut_budget_first(FAR_CUT_KEEPS, width_least, live_least);
        if self.at() == running {
            // At its least already: nothing to try.
            self.far = before;
            self.far_trial = None;
            return;
        }
        self.far_cuts += 1;
        self.far_changed_at = Some(now);
        self.far_trial = Some(FarTrial {
            judged_from: now + std::time::Duration::from_secs_f64(queue / 1000.0),
            before,
            queue_before_ms: queue,
            answers: 0,
        });
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

/// Whether a run that finishes on this node's own last layers — "start and
/// finish on this computer", automatic wherever this node holds both ends —
/// flows too, its rows drawn here (`FlowRun::finish_here`, #369): ON unless
/// `SWARMLLM_FLOW_FINISH_HERE=0` (or `off` / `false`), which keeps that
/// shape's tree rounds — the A/B inside one binary.
fn finish_here_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        !matches!(
            std::env::var("SWARMLLM_FLOW_FINISH_HERE").ok().as_deref(),
            Some("0") | Some("off") | Some("false")
        )
    })
}

/// Whether answers due here together are read in ONE forward of this node's
/// layers after the peers' (`FlowRun::due_together`, #369): ON unless
/// `SWARMLLM_FLOW_TAIL_JOIN=0` (or `off` / `false`), which reads each answer
/// in a forward of its own — the A/B inside one binary.
///
/// Why: a split that starts and finishes here is bound by this node's cost a
/// forward, not by the link (2026-10-29: 285 tail forwards a prose reply at
/// ~21 ms each, ~26 ms at 56 rows). The answers arrive in bursts — a fifth to
/// three fifths of the tail forwards had another answer waiting behind them,
/// up to 18 — so one forward per burst reads the same rows for one fixed cost.
///
/// Measured (2026-10-30, rig `tailjoin1030`, TH↔AU, 0.5B Q8_0 drafter, arms
/// interleaved on a link that degraded through the run): this node's tail
/// time −25-30% (4.0-4.5 s a prose reply against 5.4-6.6; 117-151 answers in
/// 31-36 forwards), and every joined arm beat the arm beside it read alone —
/// prose 7.80 / 7.27 / 5.40 tok/s against 6.88 / 5.89 / 3.90, code 19.39 /
/// 18.34 / 13.44 against 17.46 / 12.68 / 9.77.
fn tail_join_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        !matches!(
            std::env::var("SWARMLLM_FLOW_TAIL_JOIN").ok().as_deref(),
            Some("0") | Some("off") | Some("false")
        )
    })
}

/// The most rows one joined forward of this node's layers after the peers'
/// reads: the widest forward a flow already sends — a rebase's, its path
/// (`REBASE_PATH_ROWS`) and the default live rows (`live_most`). The last
/// layer scores every row against the whole vocabulary on the card (~0.6 MB a
/// row at 152k), so a join is held to a size the card has already met
/// rather than a tree's whole (`MAX_TREE_ROWS`, ~310 MB of scores).
const TAIL_JOIN_ROWS_MOST: usize = REBASE_PATH_ROWS + 128;

/// Does the forward `next` read on exactly where `last` left this node's
/// cache, so the two can be read as ONE forward — the rows of both, at
/// `last`'s position, with `next`'s tree shape? The worker reads a flow's
/// rows only where the tree's length is the rows before the forward plus the
/// forward's own (`model_worker::draft_tree_to_apply`), so:
/// - `next` cuts nothing back (a rebase's `truncate` starts a new tree);
/// - the same tree start, and `next`'s position is `last`'s plus its rows;
/// - `next`'s shape is `last`'s with exactly `next`'s rows appended — the
///   tree only appends between rebases (`FlowTree::push`).
fn reads_on_from(last: &Here, last_rows: usize, next: &Here, next_rows: usize) -> bool {
    next.truncate.is_none()
        && next.from == last.from
        && u32::try_from(last_rows)
            .ok()
            .and_then(|rows| last.index_pos.checked_add(rows))
            == Some(next.index_pos)
        && next.wire.len() == last.wire.len() + next_rows
        && next.wire.starts_with(&last.wire)
}

/// The peers ran the forward through and answered it: no error, no refusal.
fn answered_whole(result: &Result<LayerResult, SwarmError>) -> bool {
    result
        .as_ref()
        .is_ok_and(|r| !matches!(r.finish_reason, Some(NetworkFinishReason::Error(_))))
}

/// Answers none of whose rows the walk can reach WAIT to be read here with the
/// next answer it can, or are never read where a rebase cuts this node's cache
/// below them first (`FlowRun::due_together`, #369): ON with the batched tail,
/// off with `SWARMLLM_FLOW_DEAD_WAIT=0` (or `off` / `false`).
///
/// Why it exists: a row's draw is read only along the walk, yet this node
/// reads every answer the peers ran — to keep its cache in step with theirs.
/// On the real link (2026-10-30, rig `prose1030` arm 1) 121 of a prose reply's
/// 257 tail forwards carried only rows already off the walk's way. Read beside
/// a live answer such rows cost the joined forward a few ms; cut first,
/// nothing.
///
/// Measured (2026-11-01, `real_link.sh` TAG=deadwait1101, one binary, TH↔AU,
/// Qwen2.5-14B "start and finish here", the 0.5B Q8_0 drafter, arms ABBA ×3):
/// prose 8.16 tok/s mean (7.46-8.67, 6 replies) against 6.98 (6.07-7.64, 5),
/// faster than the arm beside it in every pair; code level (17.81 against
/// 18.01); this node's tail time 3.6 s a prose reply against 5.2.
///
/// It was first measured no faster (rig `tailjoin1030`, 2026-10-30) and kept
/// off on a guess — that the sends it frees leave in a burst the transport
/// paces out. That run's own logs refute it: no forward left within 5 ms of
/// the one before (0 of ~2,400), the delay sat after the far machine's
/// receipt (65-80% of it), and the far's 2-4 s spikes hit the old arm as hard
/// (147 against 138 ms a forward) — its two dead-wait arms ran back to back
/// inside one.
fn dead_wait_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        tail_join_on()
            && !matches!(
                std::env::var("SWARMLLM_FLOW_DEAD_WAIT").ok().as_deref(),
                Some("0") | Some("off") | Some("false")
            )
    })
}

/// Can an answer's draws move the walk? A row's draw is read only where the
/// walk can still reach the row: sent in this `current` epoch, the walk's row
/// or one under it (`live`); sent in the epoch the last rebase ended, a row
/// that rebase kept (`remap`, old row → kept row) and live; sent before that,
/// dropped as stale once read (`FlowRun::take_answer`).
fn rows_reach_the_walk(
    rows: std::ops::Range<usize>,
    epoch: u32,
    current: u32,
    remap: Option<&(u32, Vec<Option<usize>>)>,
    live: impl Fn(usize) -> bool,
) -> bool {
    if epoch == current {
        return rows.into_iter().any(live);
    }
    match remap {
        Some((before, map)) if *before + 1 == current && *before == epoch => rows
            .into_iter()
            .any(|row| map.get(row).copied().flatten().is_some_and(&live)),
        _ => false,
    }
}

/// Does `next` cut this node's cache to at or below where the first answer
/// waiting (`waiting`) would be read — a rebase's `truncate` — so that no
/// answer waiting need be read at all? Every one waiting reads on from the
/// one before (`reads_on_from`), so the first is the lowest.
fn cuts_below(next: &Here, waiting: &Here) -> bool {
    next.truncate.is_some_and(|to| to <= waiting.index_pos)
}

/// Can a reply's tree flow on this stream shape: the run's tail draws (it
/// holds the model's last layers) or this node's own layers after the run do
/// (`finish_here_on`), every peer of the run reads a tree that flows
/// (`features::DRAFT_TREE_FLOW`, beside `DRAFT_TREES` — a peer that does not
/// draw reads the same tree, as a chain's middle hop does), and the sampler
/// reads no history — a draw per row reads no path's tokens.
pub(super) fn flow_possible(
    state: &SharedState,
    shape: &StreamShape,
    sampling: &crate::types::SamplingParams,
) -> bool {
    use swarmllm_types::node::features::{DRAFT_TREES, DRAFT_TREE_FLOW};
    (shape.far_walks() || finish_here_on())
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
    /// Where this node's own layers after the run draw its rows: what they
    /// are to read with the peers' answer.
    here: Option<Here>,
}

/// What a forward carried, kept where this node's layers after the peers'
/// draw its rows (`FlowRun::finish_here`): they read the peers' answer with
/// exactly what the peers were sent — the tree's shape as SENT, never as it
/// has grown since.
struct Here {
    /// What the first peer was sent: its answer is checked against it before
    /// anything here runs on it.
    far_input: Vec<u8>,
    index_pos: u32,
    wire: Vec<u16>,
    from: u32,
    truncate: Option<u32>,
}

/// Answers released in the order their forwards were sent. This node's
/// layers after the peers' read a flow's rows into their cache in that order,
/// as every peer did (`forward_streams` runs a stream by number), and the
/// worker refuses rows that do not continue its cache
/// (`model_worker::draft_tree_to_apply`); answers come out of the forwards
/// out in any order.
struct TurnOrder<T> {
    next: u32,
    held: std::collections::BTreeMap<u32, T>,
}

impl<T> TurnOrder<T> {
    fn new() -> Self {
        Self {
            next: 0,
            held: std::collections::BTreeMap::new(),
        }
    }

    /// Hold `turn`'s answer; `true` when an earlier turn is still due.
    fn arrived(&mut self, turn: u32, answer: T) -> bool {
        self.held.insert(turn, answer);
        turn != self.next
    }

    /// The next answer in turn, once it is in.
    fn next_due(&mut self) -> Option<T> {
        let answer = self.held.remove(&self.next)?;
        self.next += 1;
        Some(answer)
    }

    /// The answer `next_due` would give, left in place.
    fn peek_due(&self) -> Option<&T> {
        self.held.get(&self.next)
    }
}

/// This node's layers after the peers' already read an answer, inside a
/// forward joined with others (`FlowRun::take_together`): its draws, and its
/// share of that forward's time by rows.
struct Tailed {
    result: LayerResult,
    ms: f64,
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

/// Where a row of the flow's tree came from.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Copied {
    /// The drafter's, or a draw no row carried.
    No,
    /// A copied run from this conversation: its prompt or its reply.
    Here,
    /// A copied run from the owner's earlier replies (#337).
    Earlier,
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
    /// Misses at a row nothing had grown under yet: the walk outran the
    /// drafter, where the rest are guesses the model did not draw.
    misses_at_a_leaf: u32,
    stale_answers: u32,
    /// Where this node's time went (ms): the drafter's calls, its own layers'
    /// forwards, and waiting with nothing to grow or send.
    drafting_ms: f64,
    near_ms: f64,
    waiting_ms: f64,
    answers: u32,
    /// The least time a forward took to be answered, whether the guard runs
    /// or not — the unit the flow's rounds are counted in when it ends
    /// (`split_rounds::round_trips_waited`).
    least_answer_ms: Option<f64>,
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
    /// Where each row came from (`copy_ahead`), beside `draws`.
    copied: Vec<Copied>,
    /// The lookup's shape; `None` where copied runs are off.
    copy: Option<crate::inference::ngram_lookup::NgramLookupConfig>,
    /// The owner's earlier replies, which this flow may copy from and is kept
    /// in (#337), and who its rows reach; `None` for anyone else's reply, or
    /// where copied runs are off.
    earlier: Option<(
        crate::inference::owner_replies::EarlierReplies,
        crate::inference::owner_replies::Exposure,
    )>,
    /// The walk's row, by epoch, a run was last looked up from.
    copied_from: Option<(u32, usize)>,
    copied_runs: u32,
    copied_rows: u32,
    /// Copied rows the walk went through: drawn there as copied.
    copied_walked: u32,
    /// The part of the three above copied from the owner's earlier replies.
    earlier_runs: u32,
    earlier_rows: u32,
    earlier_walked: u32,
    /// Where this node's layers after the peers' draw the rows: the answers
    /// in the order sent, this node's time in those layers (ms), and the
    /// answers held for an earlier one — several in at once come out of
    /// `out` in any order, so not all of them arrived early.
    in_turn: TurnOrder<Answer>,
    here_ms: f64,
    out_of_turn: u32,
    /// Forwards of this node's layers after the peers' that read several
    /// answers at once (`tail_join_on`), and the answers they read.
    joined_forwards: u32,
    answers_joined: u32,
    /// Answers read here whose rows were ALL off the walk's way by then
    /// (`FlowTree::is_live`) — read only to keep this node's cache in step
    /// with the peers', and the rows they carried: what a forward that only
    /// held their places could save (#369).
    dead_here: u32,
    dead_rows_here: u32,
    /// Answers held back because the walk can reach none of their rows
    /// (`dead_wait_on`), in turn: read with the next answer it can reach, or
    /// never, where a rebase cuts this node's cache below them first.
    waiting_dead: Vec<Answer>,
    /// Answers held back that way, and those of them never read.
    dead_waited: u32,
    dead_cut: u32,
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
        // Whoever draws reads the shared noise: this node's own layers after
        // the run always do, a peer tail only where it says so.
        let coupling = io.noise.map(|n| n.seed()).filter(|_| {
            !shape.far_walks()
                || state.peer_advertises_feature(
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
        let copy = super::speculative::ngram_lookup_config(&self.shared_state.cfg().inference)
            .filter(|_| copy_on());
        // Every peer of the run is sent each row, a copied one too: the
        // owner's earlier replies are looked in only where all of them saw it,
        // in the form they saw it (`flow_exposure`).
        let earlier = copy
            .and_then(|_| {
                self.shared_state
                    .models
                    .owner_replies
                    .for_the_owner(&self.request)
            })
            .map(|store| (store, flow_exposure(&shape)));
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
            misses_at_a_leaf: 0,
            stale_answers: 0,
            drafting_ms: 0.0,
            near_ms: 0.0,
            waiting_ms: 0.0,
            answers: 0,
            least_answer_ms: None,
            settled_base: start_pos as usize,
            unsettled: false,
            bonus_waiting: None,
            guard,
            copied: vec![Copied::No],
            copy,
            earlier,
            copied_from: None,
            copied_runs: 0,
            copied_rows: 0,
            copied_walked: 0,
            earlier_runs: 0,
            earlier_rows: 0,
            earlier_walked: 0,
            in_turn: TurnOrder::new(),
            here_ms: 0.0,
            out_of_turn: 0,
            joined_forwards: 0,
            answers_joined: 0,
            dead_here: 0,
            dead_rows_here: 0,
            waiting_dead: Vec::new(),
            dead_waited: 0,
            dead_cut: 0,
        };
        let started = std::time::Instant::now();
        let emitted_before = run.reply.generated.len();
        let outcome = run.run_with().await;
        let flowed_ms = started.elapsed().as_secs_f64() * 1000.0;
        // The rounds the price will read for this reply (FUTURE_WORK #3 (d)):
        // the round trips the flow waited, its forwards overlapping.
        let rounds_waited = crate::daemon::state::split_rounds::round_trips_waited(
            flowed_ms,
            run.least_answer_ms,
            run.forwards,
        );
        run.state.note_decode_rounds(self.request.id, rounds_waited);
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
        // The reply so far is kept as seen by this run's peers (#337): every
        // row of it crossed them. A reply the rounds finish keeps this part.
        if let Some((earlier, seen)) = &run.earlier {
            earlier.keep(run.reply.generated, seen);
        }
        if run.guard.guarded {
            keep_guard(guard_key, run.guard.to_keep(), std::time::Instant::now());
        }
        tracing::info!(
            request_id = %self.request.id,
            shape = run.shape.name(),
            peers = run.shape.chain.len() + 1,
            drawn_here = !run.shape.far_walks(),
            here_ms = format_args!("{:.0}", run.here_ms),
            out_of_turn = run.out_of_turn,
            tail_join = tail_join_on(),
            joined_forwards = run.joined_forwards,
            answers_joined = run.answers_joined,
            dead_here = run.dead_here,
            dead_rows_here = run.dead_rows_here,
            dead_wait = dead_wait_on(),
            dead_waited = run.dead_waited,
            // Those still waiting when the reply ended were never read either.
            dead_cut = run.dead_cut + run.waiting_dead.len() as u32,
            forwards = run.forwards,
            rows_a_forward = format_args!(
                "{:.1}",
                f64::from(run.rows_sent) / f64::from(run.forwards.max(1))
            ),
            rebases = run.rebases,
            misses = run.misses,
            misses_at_a_leaf = run.misses_at_a_leaf,
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
            least_answer_ms = format_args!("{:.0}", run.least_answer_ms.unwrap_or(0.0)),
            rounds_waited,
            queue_ms_most = format_args!("{:.0}", run.guard.queue_ms_most),
            served_time = served_time_on(),
            served_read = run.guard.served_read,
            far_least_ms = format_args!("{:.0}", run.guard.far_delays.least.unwrap_or(0.0)),
            far_queue_ms_most = format_args!("{:.0}", run.guard.far_queue_ms_most),
            far_cuts = run.guard.far_cuts,
            far_undone = run.guard.far_undone,
            copy = run.copy.is_some(),
            copied_runs = run.copied_runs,
            copied_rows = run.copied_rows,
            copied_walked = run.copied_walked,
            earlier = run.earlier.is_some(),
            earlier_runs = run.earlier_runs,
            earlier_rows = run.earlier_rows,
            earlier_walked = run.earlier_walked,
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
                if let Taken::Done = self.take(answer).await? {
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
                    if let Taken::Done = self.take(answer).await? {
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
            self.copied.push(Copied::No);
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
    /// (`ngram_lookup::copied_run`, the n-gram lookup's own cascade), else in
    /// the owner's earlier replies this run's peers have seen (`earlier`,
    /// #337), through
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
        let here = crate::inference::ngram_lookup::copied_run(
            seq,
            prompt_len,
            super::speculative::NGRAM_RECENT_REPLY,
            cfg,
            COPY_LEAD_MOST,
        );
        let (run, from) = match here {
            Some(run) => (run, Copied::Here),
            None => match self
                .earlier
                .as_ref()
                .and_then(|(e, shown)| e.copied_run(seq, shown, cfg, COPY_LEAD_MOST))
            {
                Some(run) => (run, Copied::Earlier),
                None => return false,
            },
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
            self.copied.push(from);
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
            if from == Copied::Earlier {
                self.earlier_runs += 1;
                self.earlier_rows += added;
            }
        }
        added > 0
    }

    /// The copied rows on the walk's path after `from`, down to `to`: all of
    /// them, and those from the owner's earlier replies.
    fn copied_between(&self, from: usize, to: usize) -> (u32, u32) {
        let mut copied = (0, 0);
        let mut row = to;
        while row != from {
            let source = self.copied.get(row).copied().unwrap_or(Copied::No);
            copied.0 += u32::from(source != Copied::No);
            copied.1 += u32::from(source == Copied::Earlier);
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
        self.copied.push(Copied::No);
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
        // No round is counted here: forwards overlap, so the flow counts the
        // round trips it waited when it ends (`split_rounds::
        // round_trips_waited`, FUTURE_WORK #3 (d)) — one a forward read a
        // 14B's 7-9 tokens a second as 2.
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
        // Where this node's layers after the run draw, the peers are sent no
        // sampler and nothing to draw — hidden states and the tree's shape
        // alone, as on every other step of a reply that starts and finishes
        // here.
        let here = (!self.shape.far_walks()).then(|| Here {
            far_input: activations.clone(),
            index_pos,
            wire: rows.wire.clone(),
            from: rows.from as u32,
            truncate: rows.truncate,
        });
        let far = &self.shape.far;
        let mut forward = super::build_spec_verify_forward(
            request_id,
            index_pos,
            activations,
            far,
            rows.truncate,
            self.shape.far_walks().then_some(&walk),
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
            // A peer holding the first layers is sent the packed rows.
            if self.shape.near.is_empty() {
                ActivationUnits::PromptBytes
            } else {
                ActivationUnits::HiddenStates
            },
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
            here,
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

    /// Take an answer as it comes — or, where this node's layers after the
    /// peers' draw the rows, every answer in the order its forward was sent
    /// (`TurnOrder`): a later one waits for those before it, and answers due
    /// together are read here together (`due_together`).
    async fn take(&mut self, answer: Answer) -> Result<Taken, SwarmError> {
        if self.shape.far_walks() {
            return self.take_answer(answer, None).await;
        }
        self.hold(answer);
        loop {
            if tail_join_on() {
                // What else has landed joins the forward: an answer behind
                // this one waits for it anyway, and pays a forward's fixed
                // cost again if it is read on its own.
                while let Some(Some(more)) = self.out.next().now_or_never() {
                    self.hold(more);
                }
            }
            let due = self.due_together();
            if due.is_empty() {
                return Ok(Taken::Continue);
            }
            if let Taken::Done = self.take_together(due).await? {
                return Ok(Taken::Done);
            }
        }
    }

    /// Count an answer about to be read here whose rows the walk can no
    /// longer reach (`dead_here`). Only this epoch's own rows: an older
    /// answer's rows are remapped, and a rebase's lead is the walk's path.
    fn note_dead_here(&mut self, sent: &Sent) {
        if sent.epoch != self.epoch || sent.lead != 0 || sent.here.is_none() {
            return;
        }
        if (sent.first_row..sent.first_row + sent.rows).all(|row| !self.flow.is_live(row)) {
            self.dead_here += 1;
            self.dead_rows_here += sent.rows as u32;
        }
    }

    fn hold(&mut self, answer: Answer) {
        if self.in_turn.arrived(stream_seq::turn(answer.0.seq), answer) {
            self.out_of_turn += 1;
        }
    }

    /// Can anything in this answer move the walk (`rows_reach_the_walk`)? A
    /// rebase's forward (it settles the tree's start and carries the path), a
    /// refusal and anything this cannot read always can.
    fn reaches_the_walk(&self, (sent, result, _): &Answer) -> bool {
        if sent.here.is_none() || sent.lead != 0 || sent.rebases_to.is_some() {
            return true;
        }
        if !answered_whole(result) {
            return true;
        }
        rows_reach_the_walk(
            sent.first_row..sent.first_row + sent.rows,
            sent.epoch,
            self.epoch,
            self.remap.as_ref(),
            |row| self.flow.is_live(row),
        )
    }

    /// Does `next` read on from the last answer waiting, within one joined
    /// forward's rows — so it can wait with them, or be read with them?
    fn reads_on_from_the_dead(&self, next: &Answer) -> bool {
        let Some(last) = self.waiting_dead.last() else {
            return true;
        };
        let rows: usize = self.waiting_dead.iter().map(|a| a.0.rows).sum();
        match (&last.0.here, &next.0.here) {
            (Some(l), Some(n)) => {
                answered_whole(&next.1)
                    && reads_on_from(l, last.0.rows, n, next.0.rows)
                    && rows + next.0.rows <= TAIL_JOIN_ROWS_MOST
            }
            _ => false,
        }
    }

    /// The answers due now that this node's layers after the peers' can read
    /// in ONE forward: the next in turn, and each after it that the peers
    /// answered whole and that reads on from the one before
    /// (`reads_on_from`), up to `TAIL_JOIN_ROWS_MOST` rows. Only the next in
    /// turn where joining is off, or where it cannot join — a refusal, a
    /// rebase, another tree's start — so each of those is taken alone, as
    /// before.
    ///
    /// An answer the walk can reach none of (`reaches_the_walk`) waits first
    /// (`dead_wait_on`): for the next it can reach, read in front of it in one
    /// forward; or for a rebase cutting this node's cache below it
    /// (`cuts_below`), when it is never read. Its draws are never read either
    /// way, so waiting costs the reply nothing; this node's cache only has to
    /// hold its rows before any forward reading on past them, which the order
    /// here keeps.
    fn due_together(&mut self) -> Vec<Answer> {
        if dead_wait_on() {
            while let Some(next) = self.in_turn.peek_due() {
                if self.reaches_the_walk(next) || !self.reads_on_from_the_dead(next) {
                    break;
                }
                let answer = self.in_turn.next_due().expect("peeked");
                self.dead_waited += 1;
                self.waiting_dead.push(answer);
            }
            if let (Some(first), Some(next)) = (self.waiting_dead.first(), self.in_turn.peek_due())
            {
                let cut = match (&first.0.here, &next.0.here) {
                    (Some(waiting), Some(n)) => cuts_below(n, waiting),
                    _ => false,
                };
                if cut {
                    self.dead_cut += self.waiting_dead.len() as u32;
                    self.waiting_dead.clear();
                } else if !self.reads_on_from_the_dead(next) {
                    // Whatever is due next cannot be read with them: they are
                    // read first, on their own, so the cache holds their rows.
                    return std::mem::take(&mut self.waiting_dead);
                }
            }
        }
        // Nothing due: those waiting go on waiting.
        if self.in_turn.peek_due().is_none() {
            return Vec::new();
        }
        let mut due = std::mem::take(&mut self.waiting_dead);
        due.push(self.in_turn.next_due().expect("peeked"));
        let mut rows: usize = due.iter().map(|a| a.0.rows).sum();
        if !tail_join_on() {
            return due;
        }
        while let Some(next) = self.in_turn.peek_due() {
            let last = due.last().expect("one is due");
            let joins = match (&last.0.here, &next.0.here) {
                (Some(l), Some(n)) => {
                    l.truncate.is_none()
                        && answered_whole(&last.1)
                        && answered_whole(&next.1)
                        && reads_on_from(l, last.0.rows, n, next.0.rows)
                }
                _ => false,
            };
            if !joins || rows + next.0.rows > TAIL_JOIN_ROWS_MOST {
                break;
            }
            rows += next.0.rows;
            due.push(self.in_turn.next_due().expect("peeked"));
        }
        due
    }

    /// Read answers due together through this node's layers after the
    /// peers' in ONE forward, then take each in turn with its own draws —
    /// what reading each alone would have done, for one forward's fixed cost.
    /// Anything that cannot be joined exactly (a payload, a check of one
    /// answer) is taken one at a time instead, which reports it as before.
    async fn take_together(&mut self, due: Vec<Answer>) -> Result<Taken, SwarmError> {
        if due.len() == 1 {
            let answer = due.into_iter().next().expect("one is due");
            return self.take_answer(answer, None).await;
        }
        let joined = {
            let near = self.shape.near.len();
            let parts: Option<Vec<&[u8]>> = due
                .iter()
                .map(|(sent, result, _)| {
                    let (here, far) = (sent.here.as_ref()?, result.as_ref().ok()?);
                    super::check_intermediate_activations(near, &here.far_input, &far.activations)
                        .ok()?;
                    Some(far.activations.as_slice())
                })
                .collect();
            parts.and_then(|parts| crate::inference::tensor_util::join_positions(&parts))
        };
        let (Some(activations), Some(first), Some(last)) = (
            joined,
            due.first().and_then(|a| a.0.here.as_ref()),
            due.last().and_then(|a| a.0.here.as_ref()),
        ) else {
            for answer in due {
                if let Taken::Done = self.take_answer(answer, None).await? {
                    return Ok(Taken::Done);
                }
            }
            return Ok(Taken::Continue);
        };
        let read = Here {
            far_input: Vec::new(),
            index_pos: first.index_pos,
            wire: last.wire.clone(),
            from: first.from,
            truncate: None,
        };
        for answer in &due {
            self.note_dead_here(&answer.0);
        }
        let seq = due[0].0.seq;
        let started = std::time::Instant::now();
        let mut tail = self.read_here(&read, seq, activations).await?;
        let ms = started.elapsed().as_secs_f64() * 1000.0;
        self.joined_forwards += 1;
        self.answers_joined += due.len() as u32;
        let rows: usize = due.iter().map(|a| a.0.rows).sum();
        let draws = std::mem::take(&mut tail.token_ids);
        // Each answer's result is the forward's, with its own rows' draws; a
        // count that does not match leaves each with none, and `take_answer`'s
        // own check then says so for each, as it would have alone.
        let whole = draws.len() == rows;
        tail.activations = Vec::new();
        tail.spec_logits = Vec::new();
        tail.token_logprobs = Vec::new();
        let mut at = 0;
        for answer in due {
            let n = answer.0.rows;
            let mut result = tail.clone();
            if whole {
                result.token_ids = draws[at..at + n].to_vec();
            }
            at += n;
            let tailed = Tailed {
                result,
                ms: ms * n as f64 / rows.max(1) as f64,
            };
            if let Taken::Done = self.take_answer(answer, Some(tailed)).await? {
                return Ok(Taken::Done);
            }
        }
        Ok(Taken::Continue)
    }

    /// This node's layers after the peers', over the first peer's answer to a
    /// forward, the last of them drawing every row (#369) — what the run's
    /// tail does where it holds the model's last layers. Run as each answer
    /// is TAKEN, in turn, never as it arrives: two forwards of one request at
    /// a worker at once would cross their replies (gotcha #180), and the
    /// cache reads the rows in the order the peers did.
    ///
    /// `&mut self` only so the future is `Send` (`dsd_stream`'s
    /// `finish_here`: the drafter this run borrows is not `Sync`).
    async fn finish_here(
        &mut self,
        here: &Here,
        seq: u32,
        far: LayerResult,
    ) -> Result<LayerResult, SwarmError> {
        super::check_intermediate_activations(
            self.shape.near.len(),
            &here.far_input,
            &far.activations,
        )
        .map_err(|e| SwarmError::Inference(format!("a tree that flows: {e}")))?;
        self.read_here(here, seq, far.activations).await
    }

    /// This node's layers after the peers' over `activations` — one answer's,
    /// or several joined (`take_together`) — read at `read`'s position with
    /// its tree shape, the last of them drawing every row.
    async fn read_here(
        &mut self,
        read: &Here,
        seq: u32,
        activations: Vec<u8>,
    ) -> Result<LayerResult, SwarmError> {
        let exec = self.exec;
        let walk = TailWalk {
            drafts: &[],
            sampling: &exec.request.sampling_params,
            generated: &[],
            coupling: self.coupling,
            parents: &[],
        };
        let mut activations = activations;
        let segments = self.shape.after.len();
        for (i, segment) in self.shape.after.iter().enumerate() {
            let draws = i + 1 == segments;
            let mut forward = super::build_spec_verify_forward(
                exec.request.id,
                read.index_pos,
                activations,
                segment,
                read.truncate,
                draws.then_some(&walk),
                Some(seq),
            );
            forward.draft_parents = read.wire.clone();
            forward.draft_tree_from = Some(read.from);
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
            if draws {
                return Ok(result);
            }
            activations = result.activations;
        }
        Err(SwarmError::Internal(
            "a tree that flows: no layers of this node's after the peers' to draw its rows".into(),
        ))
    }

    /// Record a forward's draws, walk them, emit what the walk reached, and
    /// grow from where it stands.
    /// `tailed`: this node's layers after the peers' already read it, in a
    /// forward joined with others (`take_together`).
    async fn take_answer(
        &mut self,
        (sent, result, arrived): Answer,
        tailed: Option<Tailed>,
    ) -> Result<Taken, SwarmError> {
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
        let mut result = result.expect("an error answer returned above");
        // The peer's own time with the answer (#388), before this node's
        // layers after the peers' replace the result.
        let served_us = result.served_us;
        let mut here_ms = 0.0;
        // A joined forward's time is shared out by rows: no clean sample of
        // what one check costs.
        let joined = tailed.is_some();
        if let Some(tailed) = tailed {
            result = tailed.result;
            here_ms = tailed.ms;
            self.here_ms += here_ms;
        } else if let Some(here) = &sent.here {
            // The peers ran it, so this node's layers after theirs run it
            // too, whatever its epoch: their cache then holds the rows the
            // peers' does, in the same order, and a rebase cuts both alike.
            self.note_dead_here(&sent);
            let started = std::time::Instant::now();
            result = self.finish_here(here, sent.seq, result).await?;
            here_ms = started.elapsed().as_secs_f64() * 1000.0;
            self.here_ms += here_ms;
        }
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
        self.least_answer_ms = Some(self.least_answer_ms.map_or(delay_ms, |l| l.min(delay_ms)));
        if sent.alone && current && sent.lead == 0 && !joined {
            // A check's cost runs to its draws: this node's layers after the
            // peers' too, where they draw.
            self.reply
                .check
                .record(sent.rows as u32, delay_ms + here_ms);
        }
        if sent.lead == 0 {
            // A rebase's forward carries the path too: its time is its size's.
            // The far's own time (#388) is read from a run of ONE peer only:
            // through a chain the hops before the tail are neither its time
            // nor the link's least round trip.
            let floor_ms = (served_time_on() && !self.shape.chained())
                .then(|| {
                    super::farthest_segment_round_trip_ms(
                        &self.state,
                        &self.exec.assignment.segments,
                    )
                })
                .flatten()
                .map(f64::from);
            let served_ms = believed_served_ms(delay_ms, served_us, floor_ms);
            let before = self.guard.at();
            let (cuts, far_cuts, far_undone) =
                (self.guard.cuts, self.guard.far_cuts, self.guard.far_undone);
            self.guard
                .answered(sent.started, delay_ms, served_ms, std::time::Instant::now());
            let after = self.guard.at();
            let why = if self.guard.far_undone > far_undone {
                Some("DSD: a tree that flows takes back its far cut — fewer forwards out did not shorten the far computer's time")
            } else if self.guard.far_cuts > far_cuts {
                Some("DSD: a tree that flows tries fewer forwards out — the far computer's queue may be its own forwards")
            } else if self.guard.cuts > cuts {
                Some("DSD: a tree that flows grows less ahead — its forwards queue on the way")
            } else {
                None
            };
            if let Some(why) = why {
                tracing::debug!(
                    request_id = %self.exec.request.id,
                    from = format_args!("{}x{}", before.width, before.live),
                    to = format_args!("{}x{}", after.width, after.live),
                    delay_ms = format_args!("{delay_ms:.0}"),
                    served_ms = format_args!("{:.0}", served_ms.unwrap_or(-1.0)),
                    served_us,
                    floor_ms = format_args!("{:.0}", floor_ms.unwrap_or(-1.0)),
                    link_least_ms = format_args!("{:.0}", self.guard.link_delays.least.unwrap_or(0.0)),
                    far_least_ms = format_args!("{:.0}", self.guard.far_delays.least.unwrap_or(0.0)),
                    "{why}"
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
        let (walked, earlier) = self.copied_between(self.flow.at(), reached);
        self.copied_walked += walked;
        self.earlier_walked += earlier;
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
                // The walk ran past the tree (tree rounds count no miss
                // there, `TreeRounds::grow_size`), or the drafter guessed wrong.
                if self.flow.tree().children(Some(row)).next().is_none() {
                    self.misses_at_a_leaf += 1;
                }
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
        g.answered(at(t0, sent_ms), delay, None, at(t0, sent_ms + delay as u64));
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

    /// An answer `delay` ms after its forward left, `served` of it the far's
    /// own time (#388).
    fn feed_served(g: &mut FlowGuard, t0: Instant, sent_ms: u64, delay: f64, served: f64) {
        g.answered(
            at(t0, sent_ms),
            delay,
            Some(served),
            at(t0, sent_ms + delay as u64),
        );
    }

    fn calm(g: &mut FlowGuard, t0: Instant, from_ms: u64) {
        for i in 0..4 {
            feed_served(g, t0, from_ms + i * 10, 450.0, 40.0);
        }
    }

    /// #388: a far slow on its own — every answer 300 ms later, all of it the
    /// far's time — cuts nothing on the link. The far's window tries half; the
    /// far is as slow on the forwards the cut sent lighter, so the cut is
    /// undone and no other is tried while the far stays slow. Read whole (a
    /// far that says nothing), the same answers cut the link's window to its
    /// floor — the 12 s and 37 s spells of 2026-11-01.
    #[test]
    fn a_far_slow_on_its_own_narrows_nothing_past_one_trial() {
        let t0 = Instant::now();
        let full = GuardAt {
            width: 8,
            live: 128,
        };
        let mut g = FlowGuard::new(8, 128, true, None);
        calm(&mut g, t0, 0);
        for i in 0..4 {
            feed_served(&mut g, t0, 1000 + i * 10, 750.0, 340.0);
        }
        assert_eq!((g.cuts, g.far_cuts), (0, 1), "the far's window tries a cut");
        assert_eq!(
            g.at(),
            GuardAt { width: 8, live: 64 },
            "on the budget: fewer forwards out"
        );
        assert_eq!(
            g.to_keep(),
            full,
            "a far cut on trial is never kept for the next reply"
        );
        // Answers to forwards sent before the far had the queue's worth of
        // time to work off judge nothing (the cut came at 1,780 ms with a
        // 300 ms queue).
        for i in 0..4 {
            feed_served(&mut g, t0, 1100 + i * 100, 750.0, 340.0);
        }
        assert_eq!(g.at(), GuardAt { width: 8, live: 64 });
        assert_eq!(g.far_undone, 0);
        // Four sent after it, the far as slow: not ours — back, and hold.
        for i in 0..4 {
            feed_served(&mut g, t0, 2100 + i * 10, 750.0, 340.0);
        }
        assert_eq!(g.far_undone, 1);
        assert_eq!(g.at(), full, "the cut is undone");
        for i in 0..12 {
            feed_served(&mut g, t0, 3000 + i * 100, 750.0, 340.0);
        }
        assert_eq!((g.cuts, g.far_cuts, g.at()), (0, 1, full), "and none more");
        assert_eq!(
            (g.least_width, g.least_live),
            (8, 64),
            "the trial is on record"
        );
        assert_eq!(g.served_read, 28);
        // Back under the target, the far may be tried again later.
        calm(&mut g, t0, 5000);
        assert!(!g.far_not_ours);

        // The fix toggled off: the same answers read whole.
        let mut whole = FlowGuard::new(8, 128, true, None);
        for i in 0..4 {
            feed(&mut whole, t0, i * 10, 450.0);
        }
        let mut sent = 1000;
        for _ in 0..24 {
            feed(&mut whole, t0, sent, 750.0);
            sent += 250;
        }
        assert!(whole.cuts >= 3, "{} cuts", whole.cuts);
        assert_eq!(
            whole.at(),
            GuardAt {
                width: 2,
                live: LIVE_LEAST
            }
        );
    }

    /// #388: a queue at the far that our own forwards build — it shortens
    /// when fewer are out — is narrowed: the trial's cut stands and the next
    /// is tried, the budget to its least and only then the rows a level;
    /// under the target they stay.
    #[test]
    fn a_far_queue_our_forwards_build_is_still_narrowed() {
        let t0 = Instant::now();
        let mut g = FlowGuard::new(8, 128, true, None);
        calm(&mut g, t0, 0);
        for i in 0..4 {
            feed_served(&mut g, t0, 1000 + i * 10, 750.0, 340.0);
        }
        assert_eq!(g.at().live, 64);
        // Sent after the cut acted, the far's queue is a third of before:
        // ours. Still past the target, so another trial.
        for i in 0..4 {
            feed_served(&mut g, t0, 2100 + i * 10, 550.0, 140.0);
        }
        assert_eq!((g.far_cuts, g.far_undone), (2, 0));
        assert_eq!(
            g.at(),
            GuardAt {
                width: 8,
                live: LIVE_LEAST
            }
        );
        // Sent after that cut acted (its trial judges from 2,780 ms), shorter
        // again: the budget is at its least, so the rows a level.
        for i in 0..4 {
            feed_served(&mut g, t0, 2800 + i * 10, 400.0, 105.0);
        }
        assert_eq!((g.far_cuts, g.far_undone), (3, 0));
        assert_eq!(
            g.at(),
            GuardAt {
                width: 4,
                live: LIVE_LEAST
            }
        );
        // Under the target: the cuts stand.
        for i in 0..4 {
            feed_served(&mut g, t0, 3600 + i * 10, 400.0, 80.0);
        }
        assert_eq!(g.far_undone, 0);
        assert_eq!(g.at().width, 4);
        assert_eq!(g.cuts, 0, "the link's window never cut");
    }

    /// #388: a far cut is taken from the window the flow runs at. With the
    /// link's window already under the far's, a cut of the far's own would
    /// change nothing — and be judged, and undone, on a queue it never
    /// touched.
    #[test]
    fn a_far_cut_binds_under_a_link_window_already_cut() {
        let t0 = Instant::now();
        let mut g = FlowGuard::new(8, 128, true, Some(GuardAt { width: 8, live: 64 }));
        // The link's window held where the last reply left it: its last
        // change is "later", so it never rises here.
        g.changed_at = Some(at(t0, 1_000_000));
        calm(&mut g, t0, 0);
        assert_eq!(
            g.at().live,
            64,
            "the link's window, kept from the last reply"
        );
        for i in 0..4 {
            feed_served(&mut g, t0, 1000 + i * 10, 750.0, 340.0);
        }
        assert_eq!(g.far_cuts, 1);
        assert_eq!(
            g.at(),
            GuardAt { width: 8, live: 32 },
            "half of what the flow ran at, not of the far's own 128"
        );
        // Not shortened: undone back to the far's own window; the link's
        // still holds the flow where it was.
        for i in 0..4 {
            feed_served(&mut g, t0, 2100 + i * 10, 750.0, 340.0);
        }
        assert_eq!(g.far_undone, 1);
        assert_eq!(g.far.live, 128.0);
        assert_eq!(g.at().live, 64);
        // At the least already: nothing to try, nothing counted.
        let mut floor = FlowGuard::new(8, 128, true, Some(GuardAt { width: 2, live: 32 }));
        floor.changed_at = Some(at(t0, 1_000_000));
        calm(&mut floor, t0, 0);
        for i in 0..4 {
            feed_served(&mut floor, t0, 1000 + i * 10, 750.0, 340.0);
        }
        assert_eq!((floor.far_cuts, floor.far_trial.is_none()), (0, true));
        assert_eq!(floor.far.live, 128.0, "the far's own window untouched");
    }

    /// #388: a queue on the link — the far's time steady — still cuts the
    /// link's window, by its own LEDBAT rule, with the far's time taken off.
    #[test]
    fn a_queue_on_the_link_cuts_the_links_window_with_the_far_time_off() {
        let t0 = Instant::now();
        let mut g = FlowGuard::new(8, 128, true, None);
        calm(&mut g, t0, 0);
        for i in 0..4 {
            feed_served(&mut g, t0, 1000 + i * 10, 650.0, 40.0);
        }
        assert_eq!((g.cuts, g.far_cuts), (1, 0));
        assert_eq!(
            g.at(),
            GuardAt {
                width: 4,
                live: 128
            }
        );
        assert_eq!(g.link_delays.least, Some(410.0));
        assert_eq!(g.far_delays.least, Some(40.0));
        // A far time past the delay itself is no reading.
        let mut h = FlowGuard::new(8, 128, true, None);
        h.answered(t0, 400.0, Some(500.0), at(t0, 400));
        assert_eq!((h.served_read, h.far_delays.least), (0, None));
        assert_eq!(h.link_delays.least, Some(400.0));
    }

    /// RFC 9002 § 5.3: a far's own time is taken off only where what is left
    /// stays at or above the least round trip measured to it; a far claiming
    /// more is read as claiming nothing, and with no round trip measured
    /// nothing is believed.
    #[test]
    fn a_far_time_is_believed_only_above_the_least_round_trip() {
        assert_eq!(
            believed_served_ms(450.0, Some(40_000), Some(400.0)),
            Some(40.0)
        );
        assert_eq!(
            believed_served_ms(450.0, Some(50_000), Some(400.0)),
            Some(50.0),
            "exactly the floor"
        );
        assert_eq!(believed_served_ms(450.0, Some(60_000), Some(400.0)), None);
        assert_eq!(believed_served_ms(450.0, Some(u32::MAX), Some(400.0)), None);
        assert_eq!(believed_served_ms(450.0, Some(40_000), None), None);
        assert_eq!(believed_served_ms(450.0, None, Some(400.0)), None);
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
            g.answered(at(t0, ms - 430), 430.0, None, at(t0, ms));
        }
        assert_eq!(g.at().live, 40, "within a round trip of the last rise");
        g.answered(at(t0, 900 - 430), 430.0, None, at(t0, 900));
        assert_eq!(g.at().live, 44);
        let mut ms = 900;
        while g.at().live < 128 {
            ms += 450;
            g.answered(at(t0, ms - 400), 400.0, None, at(t0, ms));
        }
        assert_eq!(g.at().width, 2, "the levels wait for the whole budget");
        for _ in 0..3 {
            ms += 450;
            g.answered(at(t0, ms - 400), 400.0, None, at(t0, ms));
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
            g.answered(at(t0, ms - 400), 400.0, None, at(t0, ms));
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

    #[test]
    fn answers_drawn_here_are_taken_in_the_order_their_forwards_were_sent() {
        let mut order = TurnOrder::new();
        // Turn 1 lands first: it waits for turn 0, and says it is early.
        assert!(order.arrived(1, "one"));
        assert_eq!(order.next_due(), None);
        assert!(!order.arrived(0, "zero"));
        assert_eq!(order.next_due(), Some("zero"));
        assert_eq!(order.next_due(), Some("one"));
        assert_eq!(order.next_due(), None);
        // Three out of turn, then the one they waited on: all four in turn.
        for (turn, answer) in [(5, "five"), (3, "three"), (4, "four")] {
            assert!(order.arrived(turn, answer));
        }
        assert_eq!(order.next_due(), None, "turn 2 is still out");
        assert!(!order.arrived(2, "two"));
        let taken: Vec<_> = std::iter::from_fn(|| order.next_due()).collect();
        assert_eq!(taken, ["two", "three", "four", "five"]);
    }

    #[test]
    fn peeking_at_the_next_answer_due_leaves_it_due() {
        let mut order = TurnOrder::new();
        assert!(order.arrived(1, "one"));
        assert_eq!(order.peek_due(), None, "turn 0 is still out");
        assert!(!order.arrived(0, "zero"));
        assert_eq!(order.peek_due(), Some(&"zero"));
        assert_eq!(order.next_due(), Some("zero"));
        assert_eq!(order.peek_due(), Some(&"one"));
        assert_eq!(order.next_due(), Some("one"));
        assert_eq!(order.peek_due(), None);
    }

    fn here(index_pos: u32, wire_len: usize, from: u32, truncate: Option<u32>) -> Here {
        Here {
            far_input: Vec::new(),
            index_pos,
            wire: (0..wire_len).map(|i| i as u16).collect(),
            from,
            truncate,
        }
    }

    /// An answer waits only where the walk can reach none of its rows: in this
    /// epoch by liveness, from the epoch the last rebase ended only through the
    /// rows it kept, from any earlier one never.
    #[test]
    fn an_answer_waits_only_where_the_walk_can_reach_none_of_its_rows() {
        let live = |row: usize| (10..20).contains(&row);
        assert!(rows_reach_the_walk(8..12, 3, 3, None, live), "one row live");
        assert!(
            !rows_reach_the_walk(0..10, 3, 3, None, live),
            "every row off the way"
        );
        // The rebase that ended epoch 2 kept old rows 4 and 5 as rows 10 and 30.
        let mut map = vec![None; 8];
        map[4] = Some(10);
        map[5] = Some(30);
        let remap = (2, map);
        assert!(
            rows_reach_the_walk(3..6, 2, 3, Some(&remap), live),
            "a kept row the walk can reach"
        );
        assert!(
            !rows_reach_the_walk(5..8, 2, 3, Some(&remap), live),
            "kept, but off the way now — or not kept"
        );
        assert!(
            !rows_reach_the_walk(3..6, 1, 3, Some(&remap), live),
            "from before the rebase before: stale"
        );
        assert!(
            !rows_reach_the_walk(3..6, 2, 3, None, live),
            "no map: nothing of it is kept"
        );
    }

    /// The answers waiting are never read only where the next forward cuts
    /// this node's cache to at or below the first of them.
    #[test]
    fn only_a_rebase_cutting_below_the_first_waiting_drops_them() {
        let waiting = here(124, 40, 100, None);
        assert!(cuts_below(&here(100, 6, 100, Some(100)), &waiting));
        assert!(cuts_below(&here(124, 6, 124, Some(124)), &waiting), "at it");
        assert!(
            !cuts_below(&here(130, 6, 130, Some(130)), &waiting),
            "above it: their rows must be read first"
        );
        assert!(!cuts_below(&here(164, 46, 100, None), &waiting), "no cut");
    }

    /// A forward joins the one before only where the worker would read the
    /// pair as one: the rows of both after the first's position, the second's
    /// tree being the first's with its own rows appended.
    #[test]
    fn a_forward_reads_on_from_the_last_only_where_its_rows_continue_the_same_tree() {
        // Tree from 100: 24 rows at 100 (tree 24 long), 16 more at 124 (40).
        let last = here(100, 24, 100, None);
        assert!(reads_on_from(&last, 24, &here(124, 40, 100, None), 16));
        assert!(
            !reads_on_from(&last, 24, &here(124, 40, 100, Some(100)), 16),
            "a rebase cuts the caches back first"
        );
        assert!(
            !reads_on_from(&last, 24, &here(124, 40, 124, None), 16),
            "another tree's start"
        );
        assert!(
            !reads_on_from(&last, 24, &here(125, 41, 100, None), 16),
            "a gap: a forward in between is still out"
        );
        assert!(
            !reads_on_from(&last, 24, &here(124, 41, 100, None), 16),
            "a shape longer than its rows"
        );
        let mut grown_elsewhere = here(124, 40, 100, None);
        grown_elsewhere.wire[3] = 99;
        assert!(
            !reads_on_from(&last, 24, &grown_elsewhere, 16),
            "a shape that is not the last one's with rows appended"
        );
    }

    #[test]
    fn only_an_answer_the_peers_ran_through_is_joined() {
        let ok = LayerResult {
            request_id: uuid::Uuid::nil(),
            token_ids: Vec::new(),
            finish_reason: None,
            activations: Vec::new(),
            sealed_token_ids: None,
            spec_logits: Vec::new(),
            matched_stop_sequence: None,
            token_logprobs: Vec::new(),
            locally_constructed: false,
            refusal: None,
            answers_step: None,
            prompt_blocks_stored: None,
            block_rows: None,
            served_us: None,
            how_served: None,
        };
        assert!(answered_whole(&Ok(ok.clone())));
        let refused = LayerResult {
            finish_reason: Some(NetworkFinishReason::Error("no".into())),
            ..ok
        };
        assert!(!answered_whole(&Ok(refused)));
        assert!(!answered_whole(&Err(SwarmError::Inference("gone".into()))));
    }
}
