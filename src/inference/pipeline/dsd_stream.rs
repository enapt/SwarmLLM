//! Split speculation's continuous stream (`docs/plans/split_speculation.md`
//! § 4b): the next chunk of guesses is drafted, run through this node's own
//! layers and sent while the chunks before it are still being checked.
//!
//! A round of split speculation is strictly serial — the drafter guesses, this
//! node runs its layers over the guesses, the far segment checks them, and only
//! then does the next round start — so the drafting and the near half sit inside
//! every round trip. Measured on the emulated 24 ms link (2026-09-30, 7B split
//! 14/14 on one card): ~17 ms of drafting and 11 ms of near half in a ~75 ms
//! round that kept 2.35 tokens.
//!
//! **How a chunk is built on chunks not yet checked.** Each carries one guess
//! past the rows it sends — the LOOK-AHEAD — and the next chunk starts from it.
//! The far segment walks a chunk's rows exactly as it walks a round (keep the
//! guesses while its own sample agrees, answer with them and the token it
//! sampled where they parted, or after the last), so no new rule runs there:
//! the answer's last token IS its choice for the look-ahead's position. A chunk
//! whose guesses were all kept and whose look-ahead that token matches leaves
//! the chunks after it valid. Any other answer ends them: they are dropped
//! unanswered, the drafter is rewound to what was kept, and the next chunk
//! truncates both halves' caches back to it — the far segment runs the dropped
//! chunks first, since it runs a stream in order (`daemon::state::forward_streams`),
//! and they cost it work, never a wrong cache.
//!
//! **Where the guesses are walked.** A plan streams when its peers' segments are
//! ONE run between this node's ([`stream_shape`]): one peer, or a chain of them
//! each handing the chunk to the next (#318 — a model too large for any two
//! machines is split across three or more, and before this every such reply
//! was checked round by round). Where the run holds the model's last layers its
//! tail walks each chunk, as above. Where this node does — the boomerang a node
//! holding both ends of a model runs by default ("Start and finish on this
//! computer"), or peers holding the first layers — the run answers each chunk
//! with hidden states and this node's last segment walks it, as each answer is
//! taken in order (FUTURE_WORK #152); the peers are sent no guess, history or
//! sampler, as on any other step of such a reply. A chunk a restart drops is
//! then never run on this side at all. PipeInfer's head node samples the same
//! way: the logits come back to it.
//!
//! **Through a chain** every hop runs the stream in its order: the chunk's
//! number travels with the activations (`features::CHAINED_STREAM`), so each
//! hop's `forward_streams` orders its own layers' turns, and a restart passes
//! the turns a hop before it skipped and so never handed on. The tail answers
//! here, as a chained check's does (`forward_verify_through_segments`).
//!
//! PipeInfer (arXiv 2407.11798, SC'24) runs this scheme across MPI nodes; it
//! found small chunks (1-4 tokens) better than large ones once several are in
//! flight, which [`guesses_per_chunk`]'s default follows, and needs runs executed
//! in the order sent, which the far segment's gate gives. Its early cancellation
//! of runs already known to be wrong has a counterpart since 2026-10-26: a far
//! segment that walks a chunk and refuses a guess skips the chunks built on it
//! until the restart arrives (`forward_streams`, `StoodOnARefusal`); a chunk
//! already running is not stopped.

use std::collections::HashSet;
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, OnceLock};

use futures::stream::{FuturesOrdered, StreamExt};
use futures::FutureExt;

use crate::daemon::state::{ExpectedStep, PendingLayerResult, SharedState, WaiterKey};
use crate::error::SwarmError;
use crate::inference::coupled_noise::CoupledNoise;
use crate::inference::dsd_controller::{AcceptanceEstimate, CheckCost, RecentMedian};
use crate::inference::router::StreamingTokenTx;
use crate::types::inference::stream_seq;
use crate::types::{ChainHop, LayerResult, NetworkCommand, NetworkFinishReason, PipelineSegment};

use super::engine_drafter::EngineDrafter;
use super::local::{ActivationUnits, ResendOnRefusal, SegmentBudget};
use super::{prompt, PipelineExecutor, TailWalk, VerifyReply};

/// Guesses per chunk, the look-ahead included: 3 checks two rows past the
/// bootstrap and guesses one more. `SWARMLLM_SPEC_STREAM_GUESSES` (1-8).
fn guesses_per_chunk() -> u32 {
    static N: OnceLock<u32> = OnceLock::new();
    *N.get_or_init(|| knob("SWARMLLM_SPEC_STREAM_GUESSES", 3, 8))
}

/// Chunks out at once. Enough to cover a round trip with chunks being built
/// (~20 ms each on the rig against ~45-70 ms to an answer); each one past that
/// is only more work thrown away at a refusal. `SWARMLLM_SPEC_STREAM_WINDOW`
/// (1-8 — `forward_streams::MAX_STREAM_WINDOW`, which the serving side sizes
/// its per-stream admission from).
fn chunks_in_flight() -> usize {
    static N: OnceLock<u32> = OnceLock::new();
    *N.get_or_init(|| {
        knob(
            "SWARMLLM_SPEC_STREAM_WINDOW",
            3,
            crate::daemon::state::forward_streams::MAX_STREAM_WINDOW,
        )
    }) as usize
}

fn knob(name: &str, default: u32, max: u32) -> u32 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.trim().parse::<u32>().ok())
        .map_or(default, |v| v.clamp(1, max))
}

/// The guesses the next chunk carries: the stream's width, even where the
/// reply has room for fewer. A chunk narrower than the rest is a shape the
/// card has not run yet — on this node's near half AND the far end's — and a
/// card's first graph of a shape late in a worker's life waited 10-59 s for
/// fresh memory under WSL2 at 84 h uptime (2026-10-26, `faster_than_local.md`
/// § 8.6; gotcha #883 is the same driver growing a pool): with every chunk
/// full, the reply's last chunk was the only narrow one, and the second reply
/// to end on one stalled there. Guesses past the reply's end are checked and
/// thrown away (`round_tokens_for_reply` emits none). Only the positions the
/// chain serves bound it (`positions_left`: from the chunk's first row to the
/// smallest context any of its machines serves, `None` unknown — then the
/// reply's room, as before), since a row past them would be refused; never
/// fewer than the room allows. `SWARMLLM_STREAM_FULL_CHUNKS=0` sends the
/// narrow chunk, for an A/B.
fn chunk_guesses(width: u32, room: u32, positions_left: Option<u32>) -> u32 {
    chunk_guesses_with(width, room, positions_left.filter(|_| full_chunks()))
}

/// [`chunk_guesses`] with its switch read: `positions_left` is `None` where
/// the chunk is held to the reply's room.
fn chunk_guesses_with(width: u32, room: u32, positions_left: Option<u32>) -> u32 {
    let fitted = width.min(room);
    if room == 0 {
        return 0;
    }
    positions_left.map_or(fitted, |left| fitted.max(width.min(left)))
}

/// The wide width [`StreamWidth`] may choose: two guesses past the narrow one
/// ([`guesses_per_chunk`]), at most 8.
fn wide_guesses_per_chunk() -> u32 {
    (guesses_per_chunk() + 2).min(8)
}

/// `SWARMLLM_STREAM_WIDTH=fixed` sends every chunk at the narrow width, as
/// the stream did before 2026-10-30 — for an A/B inside one binary.
fn width_is_fixed() -> bool {
    static FIXED: OnceLock<bool> = OnceLock::new();
    *FIXED.get_or_init(|| std::env::var("SWARMLLM_STREAM_WIDTH").as_deref() == Ok("fixed"))
}

/// How many guesses each chunk carries: of two widths, the one that makes
/// more tokens a millisecond at what this reply's chunks are being kept at and
/// what the chain's checks cost ([`chunk_rate`]).
///
/// One width does not suit every chain (2026-10-30). On the 70B split — this
/// node and an RTX 3060 in Australia running most of its 64 layers on its
/// processor, ~420 ms apart — 5 guesses a chunk against 3, arms alternating,
/// one binary: 25% and 13% faster on two replies the drafter guessed well,
/// 11% SLOWER on one it guessed at 0.70 (33 of 39 chunks restarted, and each
/// wider check cost the far end ~75 ms more). A simulation over those replies'
/// real greedy agreement (`examples/stream_width_sim.py`, within 5% of all six
/// measurements) put a fixed 5 at −16% on a LAN with a card at the far end and
/// −8% where the far end's rows are dear, and this choice at −0.5% to +8%
/// everywhere. → `docs/invariants/scheduling.md` § "A chunk is as wide as it pays".
///
/// What it knows, all measured on this node: the check's cost at each width
/// (`CheckCost`, fitted from chunks sent alone, which nothing queued ahead of
/// — the round trip, the far end and this node's own layers), this node's own
/// layers' share of it (timed per width here), the drafter's cost a guess, the
/// round trip ([`CheckCost::floor_ms`]) and the reply's RECENT acceptance:
/// kept guesses come in runs, so a reply's easy stretch chooses differently
/// from its hard one. The fit learns a row's price only from checks of two
/// widths, so until it has (`CheckCost::slope_is_measured`), chunks sent alone
/// alternate between them and the rest stay narrow; it is remembered for the
/// next request on the same machines, and decays, so a width left unsent is
/// tried again after ~10 lone chunks.
///
/// Research: PipeInfer (arXiv 2407.11798) keeps its runs small and fixed;
/// SmartSpec chooses γ by goodput — tokens over a fitted step cost — which
/// this is for a pipelined stream; PEARL (arXiv 2408.11850) lengthens drafts
/// after an acceptance. Ending a chunk early by the drafter's confidence
/// (SpecDec++, AdaEDL) was measured here and cost 27%: a short chunk holds the
/// stream ([`super::engine_drafter::stream_confidence_floor`]). Two widths
/// only — each is a shape a card captures once ([`chunk_guesses`]).
/// `SWARMLLM_STREAM_WIDTH=fixed` keeps the narrow one.
struct StreamWidth {
    narrow: u32,
    wide: u32,
    fixed: bool,
    acceptance: AcceptanceEstimate,
    /// This node's own layers' time for a chunk: narrow, wide.
    near_ms: [RecentMedian; 2],
    /// The width of the last chunk sent alone while the fit had one width.
    last_alone_wide: Option<bool>,
    /// Chunks sent wide, for the reply's line.
    sent_wide: u32,
}

impl StreamWidth {
    fn new() -> Self {
        Self::with(
            guesses_per_chunk(),
            wide_guesses_per_chunk(),
            width_is_fixed(),
        )
    }

    fn with(narrow: u32, wide: u32, fixed: bool) -> Self {
        Self {
            narrow,
            wide,
            fixed,
            acceptance: AcceptanceEstimate::recent(),
            near_ms: [RecentMedian::default(), RecentMedian::default()],
            last_alone_wide: None,
            sent_wide: 0,
        }
    }

    /// The guesses the next chunk carries; `alone`: nothing else is out, so
    /// its check is a clean sample of the cost.
    fn next(
        &mut self,
        alone: bool,
        check: &CheckCost,
        draft_ms_each: Option<f64>,
        window: usize,
    ) -> u32 {
        if self.goes_wide(alone, check, draft_ms_each, window) {
            self.sent_wide += 1;
            self.wide
        } else {
            self.narrow
        }
    }

    fn goes_wide(
        &mut self,
        alone: bool,
        check: &CheckCost,
        draft_ms_each: Option<f64>,
        window: usize,
    ) -> bool {
        if self.fixed || self.wide <= self.narrow {
            return false;
        }
        if !check.slope_is_measured() {
            if !alone {
                return false;
            }
            let wide = self.last_alone_wide != Some(true);
            self.last_alone_wide = Some(wide);
            return wide;
        }
        let (Some(draft_ms_each), Some(narrow_check), Some(wide_check)) = (
            draft_ms_each,
            check.predict(self.narrow),
            check.predict(self.wide),
        ) else {
            return false;
        };
        let round_trip = check.floor_ms().unwrap_or(0.0);
        // A width not yet timed here reads as the other one: this node's own
        // layers cost nearly the same for two more rows.
        let near = |i: usize| {
            self.near_ms[i]
                .median()
                .or_else(|| self.near_ms[1 - i].median())
                .unwrap_or(0.0)
        };
        let alpha = self.acceptance.alpha();
        let rate = |guesses, check_ms, near_ms| {
            chunk_rate(
                guesses,
                alpha,
                check_ms,
                round_trip,
                near_ms,
                draft_ms_each,
                window,
            )
        };
        rate(self.wide, wide_check, near(1)) > rate(self.narrow, narrow_check, near(0))
    }

    /// This node's own layers took `ms` over a chunk of `guesses`.
    fn note_near(&mut self, guesses: u32, ms: f64) {
        let i = usize::from(guesses > self.narrow);
        self.near_ms[i].record(ms);
    }

    /// A chunk was answered: `kept` of its `guessed` guesses were kept.
    fn note_answer(&mut self, kept: u32, guessed: u32) {
        self.acceptance.record(kept, guessed);
    }
}

/// Tokens a millisecond a stream of `guesses`-wide chunks makes
/// ([`StreamWidth`]), each guess kept at `alpha` given the ones before it.
///
/// - A chunk is BUILT here: drafting and this node's own layers.
/// - It is CHECKED in `check_ms`: this node's layers, the round trip and the
///   far end.
/// - While chunks are kept, one comes back every `steady` ms: the slowest of
///   building here, the far end's part, and a round trip shared by the window.
/// - A refusal costs the whole trip again (drafting, then the check), plus
///   half a build: the loop finishes the chunk it is building before it reads
///   an answer ([`StreamRun::run_with`]).
/// - It gains `Σ α^i, i < guesses`: the kept guesses and the far end's token
///   where it refused, every guess where it did not.
fn chunk_rate(
    guesses: u32,
    alpha: f64,
    check_ms: f64,
    round_trip_ms: f64,
    near_ms: f64,
    draft_ms_each: f64,
    window: usize,
) -> f64 {
    let k = f64::from(guesses);
    let alpha = alpha.clamp(0.0, 0.99);
    let build = draft_ms_each * k + near_ms;
    let far = (check_ms - near_ms - round_trip_ms).max(0.0);
    let trip = draft_ms_each * k + check_ms;
    let steady = build.max(far).max(trip / window.max(1) as f64);
    let whole = alpha.powf(k);
    let refused = 1.0 - whole;
    let gain = refused / (1.0 - alpha);
    let cost = steady + refused * (trip - steady + build / 2.0);
    gain / cost.max(1.0)
}

fn full_chunks() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("SWARMLLM_STREAM_FULL_CHUNKS").as_deref() != Ok("0"))
}

/// Whether a coordinator streams its checks: ON unless `SWARMLLM_SPEC_STREAM=0`
/// (or `off` / `false`), which keeps the rounds. Measured over a real link on
/// 2026-10-01 — this node's card holding the near half, a processor in Italy
/// the far half, ~270 ms apart, the release binary with the switch as the only
/// difference: +20% / +51% / +57% decode over rounds on prose, an explanation
/// and code (+25-133% at temperature 0.7), replies scored against llama.cpp
/// like the rounds' and plain decoding's. A rig with both halves on ONE card
/// reads it ~15% slower (gotcha #759) — two processes contending for one
/// device, not a deployment. Where guessing does not pay at all, a streamed
/// request remembers the rounds' verdict of zero and the next one steps aside
/// (`dsd_controller::best_gamma_overall`). A far node is streamed to only from
/// v0.3.216 (`features::STREAM_AS_ONE_WORK`); an older one gets rounds — see
/// [`stream_shape`].
pub(super) fn stream_requested() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| stream_switch(std::env::var("SWARMLLM_SPEC_STREAM").ok().as_deref()))
}

/// The switch's reading, apart from its `OnceLock` so a test can ask it twice.
fn stream_switch(v: Option<&str>) -> bool {
    !matches!(v, Some("0") | Some("off") | Some("false"))
}

/// A plan a stream can run on: ONE run of peers' segments, every other segment
/// this node's own. Their forwards then leave here one at a time, in order, and
/// every peer of the run runs the stream in that order
/// (`daemon::state::forward_streams`).
pub(super) struct StreamShape {
    /// This node's segments before the peers' — none when they hold the
    /// model's first layers.
    pub(super) near: Vec<PipelineSegment>,
    /// The first peer's segment: each chunk is sent to it.
    pub(super) far: PipelineSegment,
    /// The peers after it, each handed the chunk by the one before (#318) —
    /// empty where one peer holds every layer between this node's.
    pub(super) chain: Vec<ChainHop>,
    /// This node's segments after the peers' — the boomerang's tail. When there
    /// are any, the guesses are walked HERE, by the last of them, and the peers
    /// answer each chunk with hidden states (FUTURE_WORK #152).
    pub(super) after: Vec<PipelineSegment>,
}

impl StreamShape {
    /// Whether the run's tail samples, and so walks each chunk's guesses: only
    /// when it holds the model's last layers.
    pub(super) fn far_walks(&self) -> bool {
        self.after.is_empty()
    }

    /// The peer that answers each chunk: the run's last.
    pub(super) fn tail(&self) -> &crate::types::NodeId {
        self.chain.last().map_or(&self.far.node_id, |h| &h.node_id)
    }

    /// The layer ranges of the run, in order — every hop's, since any may
    /// refuse with its own.
    pub(super) fn far_ranges(&self) -> Vec<(u32, u32)> {
        std::iter::once(self.far.layer_range)
            .chain(self.chain.iter().map(|h| h.layer_range))
            .collect()
    }

    /// The peers of the run.
    fn peers(&self) -> usize {
        1 + self.chain.len()
    }

    /// Does the run travel as a chain? A chain that fails is finished in
    /// rounds, unchained (`dsd`); one peer's stream that fails ends the reply.
    pub(super) fn chained(&self) -> bool {
        !self.chain.is_empty()
    }

    /// For the log line that says which shape a request streamed or flowed on.
    pub(super) fn name(&self) -> &'static str {
        match (self.near.is_empty(), self.after.is_empty()) {
            (false, true) => "peer-last",
            (false, false) => "boomerang",
            (true, false) => "peer-first",
            (true, true) => "peer-only",
        }
    }
}

/// The shape a request's checks stream on, or `None` to keep the rounds.
///
/// Every peer must serve a stream as one piece of work: `STREAMED_VERIFY` alone
/// is not enough — v0.3.213-v0.3.215 serve a stream but count each chunk
/// against their per-peer cap of 4, and a chunk refused there stalled the rest
/// of the stream for 60 s (`features::STREAM_AS_ONE_WORK`). And where the run
/// holds the last layers its tail must walk a check; where this node does, the
/// walk is ours and the peers only compute their layers, which every streaming
/// peer does — the serving side has never cared which segment a stream is
/// (`forward_streams` is keyed by request, layer range and attempt).
///
/// A run of several peers streams only as a chain a check may travel
/// (`may_chain`: `PipelineExecutor::verify_may_chain`), every hop of it — the
/// first too, which hands the number on — advertising `CHAINED_STREAM` beside
/// what a chained check needs (`plan_chain`'s rule, as in
/// `forward_verify_through_segments`). Anything less keeps the rounds.
pub(super) fn stream_shape(
    state: &SharedState,
    segments: &[PipelineSegment],
    may_chain: bool,
) -> Option<StreamShape> {
    use swarmllm_types::node::features::{
        CHAINED_STREAM, CHAINED_VERIFY, STREAMED_VERIFY, STREAM_AS_ONE_WORK,
    };
    let me = state.identity.node_id();
    shape_of(
        segments,
        me,
        |peer, last| {
            state.peer_advertises_feature(peer, STREAMED_VERIFY | STREAM_AS_ONE_WORK)
                && (!last || super::peer_walks_at_tail(state, peer))
        },
        |at| {
            if !may_chain {
                return Vec::new();
            }
            super::plan_chain(
                segments,
                at,
                me,
                |n| {
                    state.peer_supports_pipeline_chain(n)
                        && state.peer_advertises_feature(n, CHAINED_VERIFY | CHAINED_STREAM)
                },
                state.cfg().inference.max_chain_hops as usize,
            )
        },
    )
}

/// [`stream_shape`]'s decision, apart from the state it asks: `serves(peer,
/// last)` — does that peer serve a stream, and walk one when it holds the last
/// layers; `chain_from(at)` — the hops a check sent to segment `at` may be
/// handed down (`plan_chain`).
fn shape_of(
    segments: &[PipelineSegment],
    me: &crate::types::NodeId,
    serves: impl Fn(&crate::types::NodeId, bool) -> bool,
    chain_from: impl FnOnce(usize) -> Vec<ChainHop>,
) -> Option<StreamShape> {
    let at = segments.iter().position(|s| s.node_id != *me)?;
    let run = segments[at..]
        .iter()
        .take_while(|s| s.node_id != *me)
        .count();
    // Peers' segments on both sides of one of ours would be two streams to
    // keep in step, and a chunk handed from one run to the other would leave
    // this node's order.
    if segments[at + run..].iter().any(|s| s.node_id != *me) {
        return None;
    }
    let peers = &segments[at..at + run];
    let chain = if run > 1 {
        // Each machine once: two segments of one machine would put two chunks
        // of one request at its worker at once, crossing their replies
        // (gotcha #180).
        let mut seen = HashSet::new();
        if !peers.iter().all(|s| seen.insert(&s.node_id)) {
            return None;
        }
        // The whole run, or the rounds.
        let chain = chain_from(at);
        if chain.len() != run - 1 {
            return None;
        }
        chain
    } else {
        Vec::new()
    };
    let shape = StreamShape {
        near: segments[..at].to_vec(),
        far: peers[0].clone(),
        chain,
        after: segments[at + run..].to_vec(),
    };
    // A plan that is the peers' alone is a hand-off, never driven from here.
    let ours = !shape.near.is_empty() || !shape.after.is_empty();
    let walks = shape.far_walks();
    let served = peers
        .iter()
        .enumerate()
        .all(|(i, s)| serves(&s.node_id, walks && i == run - 1));
    (ours && served).then_some(shape)
}

/// Each stream's attempt tag: the high bits of its numbers
/// (`types::inference::stream_seq`). From a counter, not the request id, so a
/// router retry under the same id is a different stream (gotcha #749).
pub(super) static NEXT_ATTEMPT: AtomicU32 = AtomicU32::new(0);

/// What a request's reply is made of, and what the next request on the same
/// machines learns from this one (`dsd_controller::remember`).
pub(super) struct StreamReply<'a> {
    pub generated: &'a mut Vec<u32>,
    pub finish_reason: &'a mut String,
    pub acceptance: &'a mut AcceptanceEstimate,
    pub check: &'a mut CheckCost,
    pub draft_cost: &'a mut RecentMedian,
    pub proposed: &'a mut u32,
    pub accepted: &'a mut u32,
}

/// Where emitted tokens go and when the reply ends.
pub(super) struct StreamIo<'a> {
    pub token_tx: &'a Option<StreamingTokenTx>,
    pub decoder: &'a prompt::CachedDecoder,
    pub eos: &'a HashSet<u32>,
    pub noise: Option<CoupledNoise>,
    pub max_tokens: u32,
}

/// A chunk out being checked.
struct Sent {
    seq: u32,
    index_pos: u32,
    rows: usize,
    /// The guesses the far segment walks: the rows after the bootstrap.
    drafts: Vec<u32>,
    /// The guess past the rows, which the far segment's own next token
    /// confirms or refutes; `None` once the drafter has stopped.
    lookahead: Option<u32>,
    /// The reply as the walk read it, through the bootstrap.
    history: Vec<u32>,
    /// The cut it carried — applied by the peer and this node's near segments
    /// as it was sent, and by this node's segments after the peer's as its
    /// answer is taken.
    truncate: Option<u32>,
    /// What the peer was sent, where its answer is fed on to this node's
    /// segments after it: the answer is checked against it before anything
    /// here runs on it.
    far_input: Option<Vec<u8>>,
    /// Nothing else was out when it was built, so its time is a clean sample
    /// of what a check costs (`CheckCost`).
    alone: bool,
    started: std::time::Instant,
}

type Answer = (Sent, Result<LayerResult, SwarmError>);
type Waiting = Pin<Box<dyn Future<Output = Answer> + Send>>;

/// A streamed chunk's wait on its answer, withdrawn when dropped — answered,
/// abandoned by a restart, or the reply ending — so a restart drops every
/// chunk after the refused one with nothing left behind in the map.
pub(super) struct StreamWaiter {
    pub(super) state: Arc<SharedState>,
    pub(super) key: WaiterKey,
}

impl Drop for StreamWaiter {
    fn drop(&mut self) {
        self.state.pending_layer_results.remove(&self.key);
    }
}

/// Where the next chunk starts.
struct Frontier {
    /// The position of its bootstrap — the reply's last token so far, kept or
    /// guessed.
    index_pos: u32,
    /// The reply through that bootstrap, guesses taken as kept included.
    history: Vec<u32>,
    /// A restart's cut, for both halves' caches, on the next chunk.
    truncate: Option<u32>,
    /// Nothing is built past this chunk until it is answered: it has no
    /// look-ahead (the drafter failed or had no room), or fewer guesses than
    /// were asked for. A guess the drafter is unsure of no longer holds the
    /// stream — it drafts with no confidence floor
    /// (`engine_drafter::stream_confidence_floor`). PipeInfer's "reactive
    /// speculation" held there, and so did this stream until 2026-10-26: where
    /// the round trip outlasts the far computer's check, each hold left that
    /// computer idle for a round trip (`faster_than_local.md` § 8.6).
    held_by: Option<u32>,
}

enum Taken {
    /// The chunks out are still good.
    Continue,
    /// Every chunk out was built on a refused guess.
    Restart,
    /// The reply is over.
    Done,
}

struct StreamRun<'a, 'r> {
    exec: &'a PipelineExecutor,
    state: Arc<SharedState>,
    shape: StreamShape,
    attempt: u32,
    turn: u32,
    frontier: Frontier,
    drafter: &'a mut EngineDrafter,
    drafting_off: bool,
    drafter_warm: bool,
    /// The shared noise for the walk — this node's own, or the peer's where it
    /// walks and samples with it.
    coupling: Option<u64>,
    io: StreamIo<'a>,
    reply: StreamReply<'r>,
    chunks: u32,
    kept_whole: u32,
    restarts: u32,
    /// Chunks that held the stream ([`Frontier::held_by`]).
    held: u32,
    /// The least time a chunk took to come back, read as it is taken — the
    /// unit the stream's rounds are counted in when it ends
    /// (`split_rounds::round_trips_waited`). Taken in order, a chunk may sit
    /// behind the one before it; the least is one that did not.
    least_answer_ms: Option<f64>,
    /// The longest conversation every machine of the chain serves for this
    /// model, the drafter here included — `None` where one has not said
    /// ([`StreamRun::positions_served`]).
    positions_served: Option<u32>,
    /// How wide each chunk is.
    width: StreamWidth,
}

impl PipelineExecutor {
    /// Check the reply's guesses as a stream (module doc) until it ends, from
    /// the prompt pass's token at `start_pos`, the last of `reply.generated`.
    pub(super) async fn stream_checks(
        &self,
        shape: StreamShape,
        drafter: &mut EngineDrafter,
        start_pos: u32,
        io: StreamIo<'_>,
        reply: StreamReply<'_>,
    ) -> Result<(), SwarmError> {
        let state = self.shared_state.clone();
        // Our own worker always samples with the shared noise; a peer only
        // when it says so.
        let coupling = io.noise.map(|n| n.seed()).filter(|_| {
            !shape.far_walks()
                || state.peer_advertises_feature(
                    shape.tail(),
                    swarmllm_types::node::features::COUPLED_SAMPLING,
                )
        });
        let shape_name = shape.name();
        let peers = shape.peers();
        let history = reply.generated.clone();
        let mut run = StreamRun {
            exec: self,
            state,
            shape,
            attempt: NEXT_ATTEMPT.fetch_add(1, Ordering::Relaxed) & stream_seq::MAX_ATTEMPT,
            turn: 0,
            frontier: Frontier {
                index_pos: start_pos,
                history,
                truncate: None,
                held_by: None,
            },
            drafter,
            drafting_off: false,
            drafter_warm: false,
            coupling,
            io,
            reply,
            chunks: 0,
            kept_whole: 0,
            restarts: 0,
            held: 0,
            least_answer_ms: None,
            positions_served: None,
            width: StreamWidth::new(),
        };
        run.positions_served = run.positions_served();
        let started = std::time::Instant::now();
        let outcome = run.run().await;
        // The rounds the price will read for this reply (FUTURE_WORK #3 (d)):
        // the round trips the stream waited, its chunks overlapping.
        let rounds_waited = crate::daemon::state::split_rounds::round_trips_waited(
            started.elapsed().as_secs_f64() * 1000.0,
            run.least_answer_ms,
            run.turn,
        );
        run.state.note_decode_rounds(self.request.id, rounds_waited);
        tracing::info!(
            request_id = %self.request.id,
            shape = shape_name,
            peers,
            chunks = run.chunks,
            rounds_waited,
            least_answer_ms = format_args!("{:.0}", run.least_answer_ms.unwrap_or(0.0)),
            kept_whole = run.kept_whole,
            restarts = run.restarts,
            held = run.held,
            guesses_per_chunk = run.width.narrow,
            wide_guesses = run.width.wide,
            chunks_wide = run.width.sent_wide,
            check_slope_measured = run.reply.check.slope_is_measured(),
            window = chunks_in_flight(),
            "DSD: streamed checks complete"
        );
        outcome
    }
}

impl StreamRun<'_, '_> {
    /// The longest conversation every machine a chunk reaches serves for this
    /// model ([`chunk_guesses`]'s bound): each peer of the run by what it
    /// advertises (`SharedState::peer_served_context`, the context its worker
    /// refuses above), this node's segments and its drafter by this node's own
    /// setting, the rule its workers apply (`split::effective_context_length_with`).
    /// `None` where any of them is unknown.
    fn positions_served(&self) -> Option<u32> {
        let model = &self.shape.far.shard_id.model_id;
        let cap = self
            .state
            .cfg()
            .inference
            .max_seq_len_override
            .map(|v| v as usize);
        let here = |m: &crate::types::ModelId| {
            self.state
                .model_declared_context(m)
                .map(|d| crate::inference::split::effective_context_length_with(d, cap))
        };
        let mut served = here(self.drafter.model_id())?;
        if !self.shape.near.is_empty() || !self.shape.after.is_empty() {
            served = served.min(here(model)?);
        }
        for node in std::iter::once(&self.shape.far.node_id)
            .chain(self.shape.chain.iter().map(|h| &h.node_id))
        {
            served = served.min(self.state.peer_served_context(node, model)?);
        }
        u32::try_from(served).ok()
    }

    async fn run(&mut self) -> Result<(), SwarmError> {
        let mut out: FuturesOrdered<Waiting> = FuturesOrdered::new();
        let outcome = self.run_with(&mut out).await;
        // A chain that failed is finished in rounds, unchained (`dsd`), and an
        // unchained forward is not held to the stream's order on the peers:
        // a chunk still on its way could run after the rounds' first check
        // and write into the rows it just cut back. Every chunk still out is
        // answered first — refused at a hop that cannot hand it on, or run
        // to the tail — so none is left on the way.
        if outcome.is_err() && self.shape.chained() {
            while out.next().await.is_some() {}
        }
        outcome
    }

    async fn run_with(&mut self, out: &mut FuturesOrdered<Waiting>) -> Result<(), SwarmError> {
        let window = chunks_in_flight();
        loop {
            if self.exec.request.is_cancelled() {
                tracing::info!(
                    request_id = %self.exec.request.id,
                    "DIAG: DSD inference cancelled externally"
                );
                *self.reply.finish_reason = "stop".to_string();
                return Ok(());
            }
            // An answer already in goes first: it may end the chunks the next
            // one would be built on.
            let ready = match out.next().now_or_never() {
                Some(Some(answer)) => Some(answer),
                _ => None,
            };
            let answer = match ready {
                Some(answer) => answer,
                None => {
                    let may_build = self.frontier.held_by.is_none()
                        && out.len() < window
                        && (self.frontier.history.len() as u32) < self.io.max_tokens;
                    if may_build {
                        let chunk = self.send_chunk(out.is_empty()).await?;
                        out.push_back(chunk);
                        continue;
                    }
                    match out.next().await {
                        Some(answer) => answer,
                        // Nothing out and nothing to build: the reply reached
                        // its length.
                        None => return Ok(()),
                    }
                }
            };
            match self.take_answer(answer).await? {
                Taken::Continue => {}
                // Dropping them withdraws their waits; their late answers
                // find none.
                Taken::Restart => *out = FuturesOrdered::new(),
                Taken::Done => return Ok(()),
            }
        }
    }

    /// Guess, run this node's layers over the chunk, send it to the far
    /// segment, and hand back the wait on its answer.
    async fn send_chunk(&mut self, alone: bool) -> Result<Waiting, SwarmError> {
        let exec = self.exec;
        let request_id = exec.request.id;
        let sampling = &exec.request.sampling_params;
        let started = std::time::Instant::now();
        let room = self
            .io
            .max_tokens
            .saturating_sub(self.frontier.history.len() as u32);
        let width = self.width.next(
            alone,
            self.reply.check,
            self.reply.draft_cost.median(),
            chunks_in_flight(),
        );
        let want = chunk_guesses(
            width,
            room,
            self.positions_served
                .map(|served| served.saturating_sub(self.frontier.index_pos)),
        );
        let guesses: Vec<u32> = if self.drafting_off || want == 0 {
            Vec::new()
        } else {
            match self
                .drafter
                .draft(
                    &self.state,
                    want,
                    sampling,
                    &self.frontier.history,
                    self.io.noise.map(|n| n.seed()),
                    super::engine_drafter::stream_confidence_floor(),
                    exec.request.cancel.clone(),
                )
                .await
            {
                Ok(guesses) => {
                    // Taken as kept: the next chunk is drafted on top of them,
                    // and `rewind` undoes it at a refusal.
                    self.drafter.settle(guesses.len());
                    self.drafter.push(&guesses);
                    // The first call reads the prompt too, when the read-ahead
                    // did not, and is not a guess's cost (`dsd`'s rule).
                    if self.drafter_warm && !guesses.is_empty() {
                        self.reply.draft_cost.record(
                            started.elapsed().as_secs_f64() * 1000.0 / guesses.len() as f64,
                        );
                    }
                    self.drafter_warm = true;
                    guesses
                }
                Err(e) => {
                    tracing::warn!(
                        %request_id,
                        error = %e,
                        "DSD: the drafter failed — finishing this reply without guessing ahead"
                    );
                    self.drafting_off = true;
                    Vec::new()
                }
            }
        };
        let (drafts, lookahead) = match guesses.split_last() {
            Some((&last, rest)) => (rest.to_vec(), Some(last)),
            None => (Vec::new(), None),
        };
        let bootstrap = *self
            .frontier
            .history
            .last()
            .expect("a reply holds its first token before any check");
        let rows: Vec<u32> = std::iter::once(bootstrap)
            .chain(drafts.iter().copied())
            .collect();
        let index_pos = self.frontier.index_pos;
        let truncate = self.frontier.truncate.take();
        // A check's cost is timed from here, drafting excluded — the rounds'
        // `CheckCost` measures the same span.
        let checking = std::time::Instant::now();
        let seq = stream_seq::compose(self.attempt, self.turn).ok_or_else(|| {
            SwarmError::Internal("a streamed reply ran past its checks' numbering".into())
        })?;
        self.turn += 1;
        // No round is counted here: chunks overlap, so the stream counts the
        // round trips it waited when it ends (`split_rounds::
        // round_trips_waited`, FUTURE_WORK #3 (d)).

        // This node's own layers before the peer's, one segment after another.
        let mut activations = super::pack_verify_tokens_to_le_bytes(&rows);
        let near_started = std::time::Instant::now();
        for segment in &self.shape.near {
            let forward = super::build_spec_verify_forward(
                request_id,
                index_pos,
                activations,
                segment,
                truncate,
                None,
                Some(seq),
            );
            let result = self
                .state
                .model_process_pool
                .forward_for_request(forward, None, super::worker_requester(&exec.request))
                .await?;
            if let Some(NetworkFinishReason::Error(msg)) = &result.finish_reason {
                return Err(crate::error::reclassify_flattened_error(msg)
                    .unwrap_or_else(|| SwarmError::Inference(format!("streamed check: {msg}"))));
            }
            activations = result.activations;
        }
        if !self.shape.near.is_empty() {
            self.width.note_near(
                rows.len() as u32,
                near_started.elapsed().as_secs_f64() * 1000.0,
            );
        }

        // The walk goes to the peer only where it samples. In the boomerang it
        // is sent no guess, no history and no sampler — hidden states alone,
        // as on every other step of a reply that starts and finishes here.
        let walk = TailWalk {
            drafts: &drafts,
            sampling,
            generated: &self.frontier.history,
            coupling: self.coupling,
            parents: &[],
        };
        let far = &self.shape.far;
        let far_input = (!self.shape.far_walks()).then(|| activations.clone());
        let mut forward = super::build_spec_verify_forward(
            request_id,
            index_pos,
            activations,
            far,
            truncate,
            self.shape.far_walks().then_some(&walk),
            Some(seq),
        );
        if !self.shape.chain.is_empty() {
            // Handed down the run, and answered by its tail: the reply-to it
            // answers (the `0x07` trailer), as on a chained check.
            forward.chain = self.shape.chain.clone();
            forward.requester_node_id = Some(self.state.identity.node_id().0);
        }
        let Some(peer) = self.state.resolve_peer_id_bytes(&far.node_id) else {
            return Err(SwarmError::Inference(format!(
                "streamed check: no route to segment holder {} — it left mid-request",
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
        // The tail answers for the run's layers; any hop may refuse with its
        // own range — the rounds' chained wait (`forward_verify_through_segments`).
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
            return Err(SwarmError::Network("streamed check: send dropped".into()));
        }
        // The whole run's layers: its tail answers for all of them.
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
            // A peer holding the first layers is sent the packed guesses.
            if self.shape.near.is_empty() {
                ActivationUnits::PromptBytes
            } else {
                ActivationUnits::HiddenStates
            },
        );

        let history = self.frontier.history.clone();
        if lookahead.is_some() {
            self.frontier.index_pos += rows.len() as u32;
            self.frontier.history.extend_from_slice(&guesses);
        }
        if lookahead.is_none() || (guesses.len() as u32) < want {
            self.frontier.held_by = Some(seq);
            self.held += 1;
        }
        let sent = Sent {
            seq,
            index_pos,
            rows: rows.len(),
            drafts,
            lookahead,
            history,
            truncate,
            far_input,
            alone,
            started: checking,
        };
        let state = self.state.clone();
        let network_tx = exec.network_tx.clone();
        let node = far.node_id.clone();
        let cancel = exec.request.cancel.clone();
        let segment_idx = self.shape.near.len();
        let chained = !self.shape.chain.is_empty();
        Ok(Box::pin(async move {
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
                    // `arch-scheduling.md` § "A forward the peer could not open".
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
            (sent, result)
        }))
    }

    /// The boomerang's last leg: this node's segments after the peer's, over
    /// the peer's answer to `sent`, the last of them walking its guesses.
    ///
    /// Run here, as each answer is taken in order — never as answers arrive.
    /// Two forwards of one request at a worker at once would cross their
    /// replies (gotcha #180), and a chunk whose answer is dropped by a restart
    /// is then never run on this side at all, so nothing but the cut the next
    /// chunk carries has to be undone here.
    ///
    /// `&mut self` only so the future is `Send`: the drafter this run borrows
    /// is not `Sync`, so a shared borrow of the run cannot cross an await.
    async fn finish_here(
        &mut self,
        sent: &Sent,
        far_input: &[u8],
        far: LayerResult,
    ) -> Result<LayerResult, SwarmError> {
        let exec = self.exec;
        super::check_intermediate_activations(self.shape.near.len(), far_input, &far.activations)
            .map_err(|e| SwarmError::Inference(format!("streamed check: {e}")))?;
        let walk = TailWalk {
            drafts: &sent.drafts,
            sampling: &exec.request.sampling_params,
            generated: &sent.history,
            coupling: self.coupling,
            parents: &[],
        };
        let mut activations = far.activations;
        // Not empty: only a shape with segments after the peer's feeds on.
        let last = self.shape.after.len() - 1;
        for (i, segment) in self.shape.after.iter().enumerate() {
            let is_last = i == last;
            let forward = super::build_spec_verify_forward(
                exec.request.id,
                sent.index_pos,
                activations,
                segment,
                sent.truncate,
                is_last.then_some(&walk),
                Some(sent.seq),
            );
            let result = self
                .state
                .model_process_pool
                .forward_for_request(forward, None, super::worker_requester(&exec.request))
                .await?;
            if let Some(NetworkFinishReason::Error(msg)) = &result.finish_reason {
                return Err(crate::error::reclassify_flattened_error(msg)
                    .unwrap_or_else(|| SwarmError::Inference(format!("streamed check: {msg}"))));
            }
            if is_last {
                return Ok(result);
            }
            activations = result.activations;
        }
        unreachable!("the loop returns at the last segment after the peer's")
    }

    /// Emit what a chunk's check kept and decide what the chunks after it
    /// are worth.
    async fn take_answer(&mut self, (sent, result): Answer) -> Result<Taken, SwarmError> {
        let result = result?;
        if let Some(NetworkFinishReason::Error(msg)) = &result.finish_reason {
            // The class does not survive the wire (gotcha #304).
            return Err(crate::error::reclassify_flattened_error(msg)
                .unwrap_or_else(|| SwarmError::Inference(format!("streamed check: {msg}"))));
        }
        let result = match &sent.far_input {
            Some(far_input) => self.finish_here(&sent, far_input, result).await?,
            None => result,
        };
        let answer = if !result.spec_logits.is_empty() {
            VerifyReply::Logits(result.spec_logits)
        } else if !result.token_ids.is_empty() {
            VerifyReply::Walked(result.token_ids)
        } else {
            return Err(SwarmError::Inference(
                "streamed check: the last segment returned neither logits nor a walk".into(),
            ));
        };
        let (kept, next, all) = answer.accept(
            &sent.drafts,
            &self.exec.request.sampling_params,
            &sent.history,
            self.io
                .noise
                .as_ref()
                .map(|n| (n, u64::from(sent.index_pos) + 1)),
        )?;
        // The far segment's own token at the look-ahead's position is `next`
        // when it kept every row's guess.
        let whole = all && sent.lookahead == Some(next);

        let guessed = sent.drafts.len() + usize::from(sent.lookahead.is_some());
        if guessed > 0 {
            let kept_guesses = kept.len() + usize::from(whole);
            *self.reply.proposed += guessed as u32;
            *self.reply.accepted += kept_guesses as u32;
            self.reply
                .acceptance
                .record(kept_guesses as u32, guessed as u32);
            self.width.note_answer(kept_guesses as u32, guessed as u32);
        }
        let answered_ms = sent.started.elapsed().as_secs_f64() * 1000.0;
        self.least_answer_ms = Some(
            self.least_answer_ms
                .map_or(answered_ms, |l| l.min(answered_ms)),
        );
        if sent.alone {
            self.reply.check.record(sent.rows as u32, answered_ms);
        }
        self.chunks += 1;
        if whole {
            self.kept_whole += 1;
        }

        // Nothing past the first end-of-reply token reaches anyone (R105), and
        // nothing past the budget.
        let emitted = super::round_tokens_for_reply(
            &kept,
            next,
            self.io.eos,
            (self.io.max_tokens as usize).saturating_sub(self.reply.generated.len()),
        );
        super::emit_streaming_batch(
            &self.exec.partial_reply,
            self.io.token_tx,
            self.io.decoder,
            &emitted,
            self.io.eos,
            self.reply.finish_reason,
        )
        .await;
        if !self.reply.finish_reason.is_empty() {
            return Ok(Taken::Done);
        }
        self.reply.generated.extend_from_slice(&emitted);
        let last = *emitted.last().expect("a check answers at least one token");
        if self.io.eos.contains(&last) {
            *self.reply.finish_reason = "stop".to_string();
            return Ok(Taken::Done);
        }
        if self.reply.generated.len() as u32 >= self.io.max_tokens {
            return Ok(Taken::Done);
        }
        if whole {
            // Its look-ahead was right: the frontier already stands on it.
            if self.frontier.held_by == Some(sent.seq) {
                self.frontier.held_by = None;
            }
            return Ok(Taken::Continue);
        }
        // Everything built after this chunk stood on a guess that was not
        // kept. `last` sits at `next_pos`, not yet in any cache: the next
        // chunk starts from it and cuts both halves back to it.
        let next_pos = sent.index_pos + emitted.len() as u32;
        self.drafter.rewind(next_pos as usize);
        self.drafter.push(&[last]);
        self.frontier = Frontier {
            index_pos: next_pos,
            history: self.reply.generated.clone(),
            truncate: Some(next_pos),
            held_by: None,
        };
        Ok(match sent.lookahead {
            Some(_) => {
                self.restarts += 1;
                tracing::debug!(
                    request_id = %self.exec.request.id,
                    turn = stream_seq::turn(sent.seq),
                    kept = kept.len(),
                    of = guessed,
                    "DSD: a streamed check refused a guess — restarting after it"
                );
                Taken::Restart
            }
            // A chunk without a look-ahead had nothing built after it.
            None => Taken::Continue,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::types::{ModelId, NodeId, ShardId};

    const ME: NodeId = NodeId([1; 32]);
    const PEER: NodeId = NodeId([2; 32]);
    const OTHER: NodeId = NodeId([3; 32]);

    fn plan(nodes: &[NodeId]) -> Vec<PipelineSegment> {
        nodes
            .iter()
            .enumerate()
            .map(|(i, n)| PipelineSegment {
                node_id: n.clone(),
                shard_id: ShardId {
                    model_id: ModelId("m".into()),
                    index: i as u32,
                },
                layer_range: (i as u32 * 4, i as u32 * 4 + 4),
            })
            .collect()
    }

    /// A peer that serves a stream but cannot walk a check.
    fn serves_without_walking(_: &NodeId, last: bool) -> bool {
        !last
    }

    /// Every run of peers chains, hop for hop (`plan_chain` with every hop
    /// able to carry a streamed check).
    fn every_hop_chains(segments: &[PipelineSegment]) -> impl FnOnce(usize) -> Vec<ChainHop> + '_ {
        move |at| super::super::plan_chain(segments, at, &ME, |_| true, 8)
    }

    fn shape(nodes: &[NodeId], serves: impl Fn(&NodeId, bool) -> bool) -> Option<&'static str> {
        let segments = plan(nodes);
        shape_of(&segments, &ME, serves, every_hop_chains(&segments)).map(|s| s.name())
    }

    /// FUTURE_WORK #152: the shapes the swarm makes stream — the boomerang
    /// above all (58 of 78 holdings held both ends, 2026-10-01) — and each
    /// keeps its layers where the plan put them.
    #[test]
    fn every_plan_with_one_peer_segment_streams() {
        let every = |_: &NodeId, _: bool| true;
        assert_eq!(shape(&[ME, PEER], every), Some("peer-last"));
        assert_eq!(shape(&[ME, PEER, ME], every), Some("boomerang"));
        assert_eq!(shape(&[PEER, ME], every), Some("peer-first"));
        assert_eq!(shape(&[ME, ME, PEER, ME], every), Some("boomerang"));
        let segments = plan(&[ME, ME, PEER, ME]);
        let s = shape_of(&segments, &ME, every, every_hop_chains(&segments)).unwrap();
        assert_eq!(s.near.len(), 2);
        assert_eq!(s.far.layer_range, (8, 12));
        assert_eq!(s.after.len(), 1);
        assert!(
            !s.far_walks(),
            "this node holds the last layers: the walk is ours"
        );
    }

    /// The walk is asked of the peer only where it holds the last layers: a
    /// peer that cannot walk still serves the boomerang's middle, and a peer
    /// that serves no stream serves none.
    #[test]
    fn a_peer_is_asked_to_walk_only_where_it_holds_the_last_layers() {
        assert_eq!(shape(&[ME, PEER], serves_without_walking), None);
        assert_eq!(
            shape(&[ME, PEER, ME], serves_without_walking),
            Some("boomerang")
        );
        assert_eq!(
            shape(&[PEER, ME], serves_without_walking),
            Some("peer-first")
        );
        assert_eq!(shape(&[ME, PEER, ME], |_, _| false), None);
    }

    /// #318: a model across three or more machines streams too — the peers'
    /// run as a chain the chunk is handed down, answered by its tail, which
    /// walks where it holds the last layers.
    #[test]
    fn a_run_of_peers_streams_as_a_chain() {
        let every = |_: &NodeId, _: bool| true;
        let segments = plan(&[ME, PEER, OTHER]);
        let s = shape_of(&segments, &ME, every, every_hop_chains(&segments)).unwrap();
        assert_eq!(s.name(), "peer-last");
        assert_eq!(s.far.node_id, PEER);
        assert_eq!(s.chain.len(), 1);
        assert_eq!(s.tail(), &OTHER, "the chain's tail answers");
        assert_eq!(s.far_ranges(), vec![(4, 8), (8, 12)]);
        assert!(s.far_walks());
        // The tail is the one asked to walk; the head only computes.
        let walker = |n: &NodeId, last: bool| !last || *n == OTHER;
        assert_eq!(shape(&[ME, PEER, OTHER], walker), Some("peer-last"));
        let wrong_walker = |n: &NodeId, last: bool| !last || *n == PEER;
        assert_eq!(shape(&[ME, PEER, OTHER], wrong_walker), None);
        // A chain in the middle of a boomerang: the walk is ours.
        assert_eq!(
            shape(&[ME, PEER, OTHER, ME], serves_without_walking),
            Some("boomerang")
        );
    }

    /// Every chunk is the stream's width — the last one too, past the reply's
    /// end — unless a row would fall past what a machine of the chain serves;
    /// never narrower than the reply's room allows, and nothing once it is full.
    #[test]
    fn every_chunk_is_the_streams_width_while_the_context_allows() {
        assert_eq!(chunk_guesses_with(3, 40, Some(4000)), 3);
        assert_eq!(
            chunk_guesses_with(3, 2, Some(4000)),
            3,
            "the reply's last chunk"
        );
        assert_eq!(chunk_guesses_with(3, 1, Some(4000)), 3);
        assert_eq!(
            chunk_guesses_with(3, 2, Some(2)),
            2,
            "the context's last positions"
        );
        assert_eq!(chunk_guesses_with(3, 1, Some(2)), 2);
        assert_eq!(
            chunk_guesses_with(3, 2, Some(1)),
            2,
            "never fewer than the room"
        );
        assert_eq!(
            chunk_guesses_with(3, 2, None),
            2,
            "unknown, or switched off: the room"
        );
        assert_eq!(
            chunk_guesses_with(3, 0, Some(4000)),
            0,
            "a full reply sends nothing"
        );
    }

    /// A run that cannot be chained whole — a hop without the bits, or past
    /// `max_chain_hops` — keeps the rounds, as does one machine twice in it.
    #[test]
    fn a_run_that_cannot_travel_as_one_chain_keeps_the_rounds() {
        let every = |_: &NodeId, _: bool| true;
        let segments = plan(&[ME, PEER, OTHER]);
        let unchained = |_| Vec::new();
        assert!(shape_of(&segments, &ME, every, unchained).is_none());
        let partly = |at| super::super::plan_chain(&segments, at, &ME, |n| *n != OTHER, 8);
        assert!(shape_of(&segments, &ME, every, partly).is_none());
        let three = plan(&[ME, PEER, OTHER, NodeId([4; 32])]);
        let short = |at| super::super::plan_chain(&three, at, &ME, |_| true, 1);
        assert!(shape_of(&three, &ME, every, short).is_none());
        assert_eq!(
            shape(&[ME, PEER, OTHER, PEER], every),
            None,
            "one machine twice"
        );
    }

    /// Peers on both sides of one of ours, a plan that is the peers' alone, or
    /// no peer at all: the rounds (or the hand-off, or local generation) run it.
    #[test]
    fn a_plan_that_is_not_one_run_of_peers_beside_ours_does_not_stream() {
        let every = |_: &NodeId, _: bool| true;
        assert_eq!(shape(&[PEER, ME, PEER], every), None);
        assert_eq!(shape(&[PEER, ME, OTHER], every), None);
        assert_eq!(shape(&[PEER], every), None);
        assert_eq!(shape(&[PEER, OTHER], every), None);
        assert_eq!(shape(&[ME, ME], every), None);
    }

    #[test]
    fn the_stream_is_on_unless_switched_off() {
        assert!(stream_switch(None));
        assert!(stream_switch(Some("1")));
        assert!(!stream_switch(Some("0")));
        assert!(!stream_switch(Some("off")));
        assert!(!stream_switch(Some("false")));
    }

    /// Tokens a millisecond at 3 and at 5 guesses a chunk, for checks costing
    /// `check3` / `check5` (this node's layers `near` of it, the round trip
    /// `rtt`), the drafter `draft` a guess, kept at `alpha`, window 3.
    fn rates(alpha: f64, check3: f64, check5: f64, rtt: f64, near: f64, draft: f64) -> (f64, f64) {
        (
            chunk_rate(3, alpha, check3, rtt, near, draft, 3),
            chunk_rate(5, alpha, check5, rtt, near, draft, 3),
        )
    }

    /// Wide where a row is cheap beside the trip, narrow where it is dear or
    /// where building a chunk here is what takes the time.
    #[test]
    fn a_chunk_is_as_wide_as_it_pays() {
        // A far end on a card ~420 ms away: two more rows cost it ~10 ms.
        let (narrow, wide) = rates(0.8, 1000.0, 1010.0, 420.0, 20.0, 10.0);
        assert!(wide > narrow, "cheap rows, long trip: {narrow} vs {wide}");
        // A far processor whose rows cost 200 ms each.
        let (narrow, wide) = rates(0.8, 700.0, 1100.0, 270.0, 20.0, 20.0);
        assert!(narrow > wide, "dear rows: {narrow} vs {wide}");
        // A LAN: the round trip is nothing, and building a chunk here binds.
        let (narrow, wide) = rates(0.8, 60.0, 66.0, 2.0, 20.0, 10.0);
        assert!(narrow > wide, "a LAN: {narrow} vs {wide}");
        // The 70B's measured costs (2026-10-30): the reply guessed at
        // 0.70-0.73, where 5 lost 11%, stays narrow — a refusal costs the
        // whole trip, which `steady` alone does not see — and one guessed
        // well goes wide.
        let (narrow, wide) = rates(0.72, 1250.0, 1400.0, 420.0, 250.0, 61.0);
        assert!(narrow > wide, "guessed poorly: {narrow} vs {wide}");
        let (narrow, wide) = rates(0.92, 1250.0, 1400.0, 420.0, 250.0, 61.0);
        assert!(wide > narrow, "guessed well: {narrow} vs {wide}");
    }

    /// Until the check's fit has seen two widths it cannot price a row, so
    /// the chunks sent alone — clean samples — alternate between them and the
    /// rest stay narrow; once it has, every chunk takes the width that pays.
    /// `SWARMLLM_STREAM_WIDTH=fixed` turns this red: every chunk narrow.
    #[test]
    fn the_stream_learns_a_rows_price_from_chunks_sent_alone_then_chooses() {
        let mut width = StreamWidth::new();
        let (narrow, wide) = (guesses_per_chunk(), wide_guesses_per_chunk());
        let mut check = CheckCost::for_a_plan(Some(420));
        check.record(narrow, 1000.0);
        let alone: Vec<u32> = (0..4)
            .map(|_| width.next(true, &check, Some(10.0), 3))
            .collect();
        assert_eq!(
            alone,
            [wide, narrow, wide, narrow],
            "alone, one width fitted"
        );
        assert_eq!(
            width.next(false, &check, Some(10.0), 3),
            narrow,
            "queued, unpriced"
        );

        // A far end on a card: two more rows cost it ~10 ms.
        check.record(wide, 1010.0);
        width.note_near(narrow, 20.0);
        width.note_near(wide, 21.0);
        for _ in 0..6 {
            width.note_answer(narrow, narrow);
        }
        assert_eq!(
            width.next(false, &check, Some(10.0), 3),
            wide,
            "cheap rows, kept"
        );
        // Dear rows on the same trip: narrow.
        let mut dear = CheckCost::for_a_plan(Some(420));
        dear.record(narrow, 1000.0);
        dear.record(wide, 1800.0);
        assert_eq!(width.next(false, &dear, Some(10.0), 3), narrow, "dear rows");
        assert_eq!(width.sent_wide, 3);
    }
}
