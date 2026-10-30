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
use crate::inference::dsd_controller::{AcceptanceEstimate, CheckCost, FarPace, RecentMedian};
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
/// What it knows, all measured on this node: the far end's own time for a
/// chunk at each width ([`FarPace`], below), this node's own layers' time
/// (timed per width here), the round trip ([`CheckCost::floor_ms`]), the
/// drafter's cost a guess, and the reply's RECENT acceptance: kept guesses come
/// in runs, so a reply's easy stretch chooses differently from its hard one.
///
/// The far end's time is the gap between two answers where the later chunk
/// was already waiting there when the earlier one finished — the round trip
/// and this node's work cancel out of it (packet-pair dispersion: Keshav,
/// SIGCOMM 1991; on a loaded path the true time can be a lesser mode of the
/// gaps, Dovrolis et al., INFOCOM 2001, so a median of recent ones). On the
/// 70B split (2026-10-30, #375) those gaps read 752-767 ms for half of 69
/// chunks of 3, where whole checks sent alone spread 1318-1511 ms and fitted a
/// row's price at 208, 3.7 and 55 ms after three replies — two rows, ~100 ms,
/// buried in noise twenty times larger. A width with no gaps yet is tried on
/// chunks that queue (every other one) until it has [`FAR_PACE_SAMPLES`];
/// both are remembered for the next request on the same machines, and every
/// [`PROBE_EVERY`]th chunk that queues goes the other way to keep the width
/// not chosen current. Where chunks never queue at the far end (it keeps up —
/// a LAN, a card), the check's fit from chunks sent alone (`CheckCost`)
/// prices both widths, alternating lone chunks until it has two.
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
    /// The width of the last chunk that queued while one width had no gaps.
    last_explored_wide: Option<bool>,
    /// Chunks that queued and were sized by [`FarPace`], for [`PROBE_EVERY`].
    paced: u32,
    /// The turn of the last answer taken and when it arrived.
    last_answer: Option<(u32, std::time::Instant)>,
    /// Chunks sent wide, for the reply's line.
    sent_wide: u32,
}

/// Gaps a width needs before [`FarPace`] prices it.
const FAR_PACE_SAMPLES: usize = 3;

/// Every this many chunks priced by [`FarPace`] that queue, one goes the
/// width not chosen, so a remembered figure does not go stale.
const PROBE_EVERY: u32 = 16;

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
            last_explored_wide: None,
            paced: 0,
            last_answer: None,
            sent_wide: 0,
        }
    }

    /// The guesses the next chunk carries; `alone`: nothing else is out, so
    /// its check is a clean sample of the cost.
    fn next(
        &mut self,
        alone: bool,
        check: &CheckCost,
        far: &FarPace,
        draft_ms_each: Option<f64>,
        window: usize,
    ) -> u32 {
        if self.goes_wide(alone, check, far, draft_ms_each, window) {
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
        far: &FarPace,
        draft_ms_each: Option<f64>,
        window: usize,
    ) -> bool {
        if self.fixed || self.wide <= self.narrow {
            return false;
        }
        let round_trip = check.floor_ms().unwrap_or(0.0);
        // A width not yet timed here reads as the other one: this node's own
        // layers cost nearly the same for two more rows.
        let near_ms = [self.near_ms[0].median(), self.near_ms[1].median()];
        let near = |i: usize| near_ms[i].or(near_ms[1 - i]).unwrap_or(0.0);
        let paced = [
            far.narrow.len() >= FAR_PACE_SAMPLES,
            far.wide.len() >= FAR_PACE_SAMPLES,
        ];
        let mut probe = false;
        let checks = match paced {
            // The check at each width: this node's layers, the trip, and the
            // far end's own time.
            [true, true] => {
                if !alone {
                    self.paced += 1;
                    probe = self.paced.is_multiple_of(PROBE_EVERY);
                }
                far.narrow
                    .median()
                    .zip(far.wide.median())
                    .map(|(n, w)| (near(0) + round_trip + n, near(1) + round_trip + w))
            }
            // Chunks wait at the far end, timed at one width only: the other
            // is tried on chunks that queue, every other one, until it has
            // its gaps. A chunk sent alone yields none.
            [true, false] | [false, true] => {
                if alone {
                    return paced[1];
                }
                let wide = self.last_explored_wide.map_or(!paced[1], |w| !w);
                self.last_explored_wide = Some(wide);
                return wide;
            }
            // No chunk has waited there: the check's fit from chunks sent
            // alone, which learns a row's price from two widths.
            [false, false] => {
                if !check.slope_is_measured() {
                    if !alone {
                        return false;
                    }
                    let wide = self.last_alone_wide != Some(true);
                    self.last_alone_wide = Some(wide);
                    return wide;
                }
                check.predict(self.narrow).zip(check.predict(self.wide))
            }
        };
        let (Some(draft_ms_each), Some((narrow_check, wide_check))) = (draft_ms_each, checks)
        else {
            return false;
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
        let wide = rate(self.wide, wide_check, near(1)) > rate(self.narrow, narrow_check, near(0));
        wide != probe
    }

    /// A chunk's answer arrived. Where the chunk before it in the stream was
    /// the last answered and this one left here a round trip before that
    /// answer came back — so it was already waiting at the far end when that
    /// one finished there — the gap between the two answers is the far end's
    /// own time for this one ([`FarPace`]). `turn` is its place on the main
    /// line ([`Sent::order`]): a spare not taken between two chunks is
    /// skipped unrun at the far end, and does not part them.
    fn note_arrival(
        &mut self,
        turn: u32,
        rows: usize,
        left: std::time::Instant,
        arrived: std::time::Instant,
        round_trip_ms: Option<f64>,
        far: &mut FarPace,
    ) {
        let before = self.last_answer.replace((turn, arrived));
        let (Some((before_turn, before_arrived)), Some(round_trip_ms)) = (before, round_trip_ms)
        else {
            return;
        };
        let wide = match u32::try_from(rows) {
            Ok(r) if r == self.narrow => false,
            Ok(r) if r == self.wide => true,
            _ => return,
        };
        let waited_there = turn == before_turn.wrapping_add(1)
            && left + std::time::Duration::from_secs_f64(round_trip_ms.max(0.0) / 1000.0)
                < before_arrived;
        if waited_there {
            far.record(
                wide,
                arrived
                    .saturating_duration_since(before_arrived)
                    .as_secs_f64()
                    * 1000.0,
            );
        }
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

/// The look-ahead a chunk names to its far end (#382): the guess one past its
/// last row, which the next chunk stands on. Told it, a far end that walks the
/// check sees a whole walk that drew another token there as the refusal it is,
/// and skips the turns built on it (`forward_streams`) — where it was not told,
/// the restart waited behind one or two stale chunks (0.65-1.26 s a wrong
/// look-ahead on the 70B, 2026-10-30). Only to a LONE far segment that walks:
/// a chained run's tail is a hop away and nothing hands it on. It shows the
/// far end nothing of a new kind: a far end that walks is sent the chunk's
/// other guesses as ids (the drafter's alone — the stream copies no text) and
/// draws the reply's tokens itself.
fn lookahead_to_send(
    shape: &StreamShape,
    lookahead: Option<u32>,
    far_reads_it: bool,
) -> Option<u32> {
    lookahead.filter(|_| shape.far_walks() && shape.chain.is_empty() && far_reads_it)
}

/// ON unless `SWARMLLM_STREAM_LOOKAHEAD=0`, the A/B arm that names no
/// look-ahead (the far end then skips only after a refused draft).
fn names_the_lookahead() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("SWARMLLM_STREAM_LOOKAHEAD").as_deref() != Ok("0"))
}

/// Spare restarts (#381): OFF unless `SWARMLLM_STREAM_SPARES=1`. Measured
/// on the two-node rig (7B coder, 0.5B drafter, both halves on one
/// processor, 300 ms, 2026-10-30) the stream ran 14-18% slower with them —
/// ~150 spares a 160-token reply, 11 run, each costing this node ~120 ms the
/// answers waited behind — so they stay off until a split where they pay is
/// measured (the 70B, whose far end must first run a release that reads
/// them). Each spare is priced by [`spare_pays`] since 2026-10-31 (the rig's
/// figures above were under a fixed floor). A far end runs a spare only where
/// it advertises `features::STREAM_SPARES`, so an older one is never sent one.
fn spares_switched_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("SWARMLLM_STREAM_SPARES").as_deref() == Ok("1"))
}

/// The guesses a spare restart carries, its first row's token aside: the
/// stream's narrow width by default, as the restart it stands in for would
/// carry. `SWARMLLM_STREAM_SPARE_GUESSES` (1-8).
fn spare_guesses() -> u32 {
    static N: OnceLock<u32> = OnceLock::new();
    *N.get_or_init(|| knob("SWARMLLM_STREAM_SPARE_GUESSES", guesses_per_chunk(), 8))
}

/// A fixed least chance, by the drafter's own reckoning, that a spare is the
/// restart its chunk's answer calls for — the A/B of [`spare_pays`], which
/// decides otherwise. `SWARMLLM_STREAM_SPARE_MIN` (0-1). The first build's
/// rule was a fixed 0.4, tuned on a pair that guessed badly (the rig's 0.5B
/// drafter for the 7B coder: a spare scored 0.08-0.40 ran 3-11% of the time
/// whatever its score); for the 70B's 3B drafter the score IS the chance
/// (0.2-0.3 ran 27%, 0.3-0.4 44%, 2026-10-31), and 0.4 sent three spares a
/// reply.
fn fixed_spare_floor() -> Option<f32> {
    static F: OnceLock<Option<f32>> = OnceLock::new();
    *F.get_or_init(|| {
        std::env::var("SWARMLLM_STREAM_SPARE_MIN")
            .ok()
            .and_then(|v| v.trim().parse::<f32>().ok())
            .filter(|f| f.is_finite())
            .map(|f| f.clamp(0.0, 1.0))
    })
}

/// What [`spare_pays`] prices a spare restart from (#381). Every time is in
/// milliseconds from the moment it is decided — the chunk just sent, the
/// spare not yet drafted — and may be negative for an answer already in.
#[derive(Debug, Clone)]
struct SpareTerms<'a> {
    /// [`spare_at`]'s chance of exactly the refusal the spare bets on.
    score: f64,
    /// The chunks out before this one on the reply's line, oldest first:
    /// when each one's answer is due here, and the chance it refuses a guess.
    earlier: &'a [(f64, f64)],
    /// The chance this chunk refuses a guess.
    this_refused: f64,
    /// When the far end finishes this chunk's check.
    this_done: f64,
    /// Chunks the window still lets go before an answer frees a place.
    room: usize,
    /// This node's time to build a chunk, and to build the spare.
    build: f64,
    spare: f64,
    round_trip: f64,
}

/// Whether a spare restart pays (#381): the chance it runs — its own score,
/// and its chunk not standing on an earlier chunk's refused guess — times
/// the restart cycle it saves (the refusal's trip back, this node's build,
/// the trip out, less however late the spare itself reaches the far end),
/// against what building it delays. This node builds the spare before
/// anything else: every chunk the window lets go next is that much later,
/// and so is the taking of every answer that lands meanwhile. An answer
/// that is a refusal restarts that much later (a chunk's answer is taken
/// between builds, never inside one — [`StreamRun::run_with`]); where every
/// chunk out is kept, the far end may wait for the next chunk.
///
/// The fixed floor it replaced (0.4, [`fixed_spare_floor`]) both sent spares
/// that cost more than they saved and withheld ones that paid. Replayed over
/// six of the 70B's own replies with its 3B drafter and this split's measured
/// times (scratchpad `spare_sim.py`, 2026-10-31), this rule was the best of
/// the rules tried in all ten cases simulated (3 or 5 guesses a chunk; as
/// measured, a cheaper spare, a far end twice as slow, a shorter trip):
/// 1.07-1.10x where the floor gave 0.98-1.06x, never slower than no spares.
fn spare_pays(t: &SpareTerms) -> bool {
    let clamp = |p: f64| {
        if p.is_finite() {
            p.clamp(0.0, 1.0)
        } else {
            0.0
        }
    };
    // This node's next stretch of work, back to back from now: the chunks
    // the window lets go — after the spare, with one.
    let builds = |from: f64| -> Vec<(f64, f64)> {
        (0..t.room)
            .map(|i| {
                let s = from + i as f64 * t.build;
                (s, s + t.build)
            })
            .collect()
    };
    let without = builds(0.0);
    let mut with = vec![(0.0, t.spare)];
    with.extend(builds(t.spare));
    // An answer is taken at the end of the work it lands in; one already in
    // waits for nothing but a spare, which is built before the loop looks.
    let taken = |due: f64, busy: &[(f64, f64)], spare_first: bool| -> f64 {
        if due <= 0.0 {
            return if spare_first { t.spare } else { 0.0 };
        }
        busy.iter()
            .find(|(s, e)| *s <= due && due < *e)
            .map_or(due, |&(_, e)| e)
    };
    let mut delay = 0.0;
    let mut none_before = 1.0;
    for &(due, refused) in t.earlier {
        let refused = clamp(refused);
        delay += none_before * refused * (taken(due, &with, true) - taken(due, &without, false));
        none_before *= 1.0 - refused;
    }
    // Every chunk kept: the next one reaches the far end later. With no room
    // in the window it waits for the oldest answer as well.
    let next_from = |spare: f64| -> f64 {
        if t.room > 0 {
            spare
        } else {
            let oldest = t
                .earlier
                .first()
                .map_or(t.this_done + t.round_trip / 2.0, |e| e.0);
            spare.max(oldest)
        }
    };
    let reaches = |from: f64| from + t.build + t.round_trip / 2.0;
    let idle = |at: f64| (at - t.this_done).max(0.0);
    let kept = none_before * (1.0 - clamp(t.this_refused));
    delay += kept * (idle(reaches(next_from(t.spare))) - idle(reaches(next_from(0.0))));
    let late = (t.spare + t.round_trip / 2.0 - t.this_done).max(0.0);
    let saved = (t.round_trip + t.build - late).max(0.0);
    let runs = clamp(t.score) * none_before;
    runs > 0.0 && runs * saved > delay
}

/// Whether a chunk about to go, `main_out` main-line chunks already out and
/// `spares_out` spares, may take a spare beside it: only while every chunk of
/// the stream out at once — main and spare, the two about to go included —
/// stays within `forward_streams::MAX_STREAM_WINDOW`. The far end holds a
/// stream's chunks up to twice that, the other half for the chunks a restart
/// leaves queued there; spares counted on top of a full window would pass it,
/// the spare would be refused on arrival, and a reply whose check then took
/// that spare would fail (security review of #381, LOW-1). A spare is
/// optional; a main chunk never is.
fn room_for_a_spare(main_out: usize, spares_out: usize) -> bool {
    let window = crate::daemon::state::forward_streams::MAX_STREAM_WINDOW as usize;
    // The chunk about to go and its spare.
    let about_to_go = 2;
    main_out + spares_out + about_to_go <= window
}

/// Where a chunk's spare restart stands (#381), if anywhere: the guess `j`
/// whose refusal it bets on, the token it bets the check draws there instead
/// (the drafter's runner-up), and the chance of exactly that answer as the
/// drafter sees it — every guess before `j` kept, then its runner-up drawn at
/// `j`: `p₂(j) · Π_{i<j} p₁(i)`. A guess past the rows (the look-ahead, the
/// last) is a candidate only where the far end is told it
/// ([`lookahead_to_send`]): a walk that kept every row and drew another token
/// there is a refusal only then. `None` below `floor`, or where the runner-ups
/// do not answer every guess.
fn spare_at(
    guesses: &[u32],
    runner_ups: &[crate::inference::worker_ipc::RunnerUp],
    lookahead_named: bool,
    floor: f32,
) -> Option<(usize, u32, f32)> {
    if guesses.is_empty() || runner_ups.len() != guesses.len() {
        return None;
    }
    let candidates = if lookahead_named {
        guesses.len()
    } else {
        guesses.len() - 1
    };
    let mut reached = 1.0f32;
    let mut best: Option<(usize, u32, f32)> = None;
    for (j, r) in runner_ups.iter().enumerate().take(candidates) {
        let score = reached * r.p;
        if score.is_finite()
            && score >= floor
            && r.token != guesses[j]
            && best.is_none_or(|(_, _, b)| score > b)
        {
            best = Some((j, r.token, score));
        }
        reached *= r.guess_p;
    }
    best
}

/// The chance a chunk of `guesses` refuses one, by the drafter's reckoning:
/// one less the chance it keeps them all — the look-ahead included, whose
/// miss restarts the stream too ([`StreamRun::take_answer`]). Where the
/// runner-ups do not answer every guess, the reply's recent acceptance.
fn chance_of_a_refusal(
    guesses: &[u32],
    runner_ups: &[crate::inference::worker_ipc::RunnerUp],
    alpha: f64,
) -> f64 {
    let kept_all = if runner_ups.len() == guesses.len() {
        runner_ups
            .iter()
            .map(|r| f64::from(r.guess_p).clamp(0.0, 1.0))
            .product::<f64>()
    } else {
        alpha
            .clamp(0.0, 1.0)
            .powi(i32::try_from(guesses.len()).unwrap_or(i32::MAX))
    };
    if kept_all.is_finite() {
        1.0 - kept_all
    } else {
        1.0
    }
}

/// The wire tree (`LayerForward::draft_parents`) of a near forward carrying a
/// spare (#381): `chain` rows, each after the one before — the rows read
/// again after a spare was taken, then the chunk's — and `spare` rows
/// branching after chain row `after`, each after the one before. Wire form:
/// 0 follows the cache's end, `k` follows row `k − 1`.
fn near_tree(chain: usize, after: usize, spare: usize) -> Vec<u16> {
    let wire = |row: usize| u16::try_from(row + 1).unwrap_or(u16::MAX);
    (0..chain)
        .map(|i| if i == 0 { 0 } else { wire(i - 1) })
        .chain((0..spare).map(|i| {
            if i == 0 {
                wire(after)
            } else {
                wire(chain + i - 1)
            }
        }))
        .collect()
}

/// Rows `from..to` of a near forward's answer of `rows` rows, each the same
/// width.
fn rows_of(bytes: &[u8], rows: usize, from: usize, to: usize) -> Result<Vec<u8>, SwarmError> {
    use crate::inference::tensor_util;
    // Hidden states cross as a shape header and an f32 or Q8_0 payload, never
    // bare rows: the header says how many rows they are (gotcha #133's class).
    let held = tensor_util::activation_positions(bytes);
    if held != u32::try_from(rows).ok() || from >= to || to > rows {
        return Err(SwarmError::Internal(format!(
            "streamed check: {} bytes holding {held:?} rows are not {rows} to take {from}..{to} of",
            bytes.len()
        )));
    }
    if from == 0 && to == rows {
        return Ok(bytes.to_vec());
    }
    if let Some(taken) = tensor_util::take_positions(bytes, from as u32, to as u32) {
        return Ok(taken);
    }
    // A Q8_0 row that ends inside a block cannot be cut in its encoding:
    // narrowed after decoding, it goes as f32, which every reader takes.
    let tensor = tensor_util::bytes_to_tensor(bytes)?;
    let at = tensor.dims().len().saturating_sub(2);
    let narrowed = tensor
        .narrow(at, from, to - from)
        .map_err(SwarmError::internal)?;
    tensor_util::tensor_to_bytes(&narrowed)
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
    /// The far end's time for a chunk at each width, remembered across
    /// requests on the same machines.
    pub far_pace: &'a mut FarPace,
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
    /// Its place among the chunks the far end runs or skips in turn — the
    /// main line's count, which a spare shares with the chunk it stands
    /// beside ([`StreamWidth::note_arrival`] reads two answers as back to back
    /// by it, never by their numbers, which spares leave gaps in).
    order: u32,
    started: std::time::Instant,
    /// When it was sent on, this node's layers done.
    left: std::time::Instant,
}

/// A spare restart out (#381): run at the far end only where chunk `of`'s
/// walk keeps `kept` guesses and draws `token` next; skipped there, unrun,
/// otherwise.
struct SpareOut {
    of: u32,
    kept: u32,
    token: u32,
    /// Its own number.
    seq: u32,
    /// Where its first row — `token` — sits.
    index_pos: u32,
    /// Its rows after the first, then its look-ahead: what the reply stands
    /// on once it is taken.
    guesses: Vec<u32>,
    /// The drafter's chance of exactly its refusal ([`spare_at`]) — logged
    /// when taken, beside the same figure logged when sent, so the floor
    /// can be checked against how often a spare at each score runs.
    score: f32,
    /// When it was sent on, this node's layers done.
    left: std::time::Instant,
    wait: Waiting,
}

/// The chunk [`StreamRun::send_spare`] builds a spare beside: its number and
/// main-line order, where it started, how much of the frontier's history was
/// there then, its guesses with the drafter's runner-ups, whether its
/// look-ahead was named, and where its rows start in this node's caches.
struct ChunkSent<'a> {
    seq: u32,
    order: u32,
    index_pos: u32,
    history: usize,
    guesses: &'a [u32],
    runner_ups: &'a [crate::inference::worker_ipc::RunnerUp],
    named: bool,
    near_from: u32,
}

/// A chunk on its way to the far segment ([`StreamRun::send_to_far`]).
struct FarChunk {
    seq: u32,
    index_pos: u32,
    activations: Vec<u8>,
    drafts: Vec<u32>,
    lookahead: Option<u32>,
    /// The look-ahead as the far end is told it ([`lookahead_to_send`]).
    named: Option<u32>,
    /// The reply through the chunk's first row.
    history: Vec<u32>,
    truncate: Option<u32>,
    branch: Option<crate::types::StreamBranch>,
    alone: bool,
    order: u32,
    started: std::time::Instant,
}

/// A spare restart drafted for a chunk ([`StreamRun::draft_spare`]).
struct SparePlan {
    /// The guess it bets the check refuses: as many are kept before it.
    after_guess: usize,
    /// The token it bets the check draws there — its first row.
    token: u32,
    /// Where that row sits.
    index_pos: u32,
    /// Its rows: `token`, then its guesses but the last.
    rows: Vec<u32>,
    /// Its guesses, the look-ahead last.
    guesses: Vec<u32>,
    /// The reply through `token`.
    history: Vec<u32>,
    /// [`spare_at`]'s chance of exactly this refusal.
    score: f32,
}

/// A chunk, its check's result, and the moment that result arrived
/// ([`stamp_on_arrival`]).
type Answer = (Sent, Result<LayerResult, SwarmError>, std::time::Instant);
type Waiting = Pin<Box<dyn Future<Output = Answer> + Send>>;

/// Run `wait` as its own task and hand back its output with the moment it
/// finished. Dropping the returned future aborts the task, so a restart that
/// drops a chunk's wait still withdraws it ([`StreamWaiter`]).
///
/// A future only runs when it is polled, and the stream polls its waits only
/// between the chunks it builds (`StreamRun::run_with`). A wait polled there
/// read its answer's arrival as late as the end of the build it landed in, up
/// to a whole chunk's drafting and near forward late, and every check time
/// recorded carried that. On the 70B split (2026-10-30) the fit of a row's price
/// from those times read 208, 3.7 and 55 ms after three replies, where the
/// row costs ~40-55 ms, and a chunk's width chosen from it was chosen at random
/// (#375).
fn stamp_on_arrival<T: Send + 'static>(
    wait: impl Future<Output = T> + Send + 'static,
) -> impl Future<Output = Result<(T, std::time::Instant), tokio::task::JoinError>> + Send {
    use tracing::Instrument;
    let task = tokio::spawn(
        async move {
            let out = wait.await;
            (out, std::time::Instant::now())
        }
        .in_current_span(),
    );
    let withdraw = AbortOnDrop(task.abort_handle());
    async move {
        let _withdraw = withdraw;
        task.await
    }
}

/// Aborts the task it guards when dropped.
struct AbortOnDrop(tokio::task::AbortHandle);

impl Drop for AbortOnDrop {
    fn drop(&mut self) {
        self.0.abort();
    }
}

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
    /// What the next chunk is to the far end's turns: built on a taken spare
    /// (`StreamBranch::AfterSpare`), or `None` — follows, or restarts by its
    /// `truncate`.
    branch: Option<crate::types::StreamBranch>,
}

enum Taken {
    /// The chunks out are still good.
    Continue,
    /// Every chunk out was built on a refused guess.
    Restart,
    /// Every chunk out was built on a refused guess, and the spare for that
    /// refusal ran at the far end (#381): its wait is the only one left.
    Spare(Waiting),
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
    /// The least time a chunk took to come back, read as it arrives
    /// ([`stamp_on_arrival`]) — the unit the stream's rounds are counted in
    /// when it ends (`split_rounds::round_trips_waited`). Run in order at the
    /// far end, a chunk may sit behind the one before it; the least is one
    /// that did not.
    least_answer_ms: Option<f64>,
    /// How long answers sat arrived before the loop took them — what a miss
    /// waits behind the chunk being built costs this reply.
    answers_waited_ms: f64,
    /// When the stream began — what each chunk's times are logged from.
    began: std::time::Instant,
    /// The longest conversation every machine of the chain serves for this
    /// model, the drafter here included — `None` where one has not said
    /// ([`StreamRun::positions_served`]).
    positions_served: Option<u32>,
    /// How wide each chunk is.
    width: StreamWidth,
    /// Whether this reply's chunks may carry spare restarts (#381), decided
    /// once at its start ([`StreamRun::spares_possible`]); turned off for the
    /// rest of the reply if a near forward refuses a spare's rows.
    spares_on: bool,
    /// The spares out, at most one a chunk, oldest first.
    spares: Vec<SpareOut>,
    /// Where this node's segments' caches stop holding the reply's line: a
    /// spare's rows sit past it, so the next near forward cuts them there.
    near_cut: Option<u32>,
    /// Below this position only do this node's segments' caches hold the
    /// reply: a taken spare's rows rode in an earlier near forward as a
    /// branch, so they are in no near cache, and the next near forward reads
    /// the reply again from here ([`StreamRun::near_pass`]).
    near_valid_to: Option<u32>,
    spares_sent: u32,
    spares_taken: u32,
    /// Spares [`spare_pays`] priced out.
    spares_declined: u32,
    /// The chunks out on the reply's line, oldest first — what
    /// [`spare_pays`] reads; kept only while spares are on.
    line_out: std::collections::VecDeque<LineOut>,
    /// This reply's time to build a spare: drafting and this node's layers.
    spare_ms: RecentMedian,
    /// Main-line chunks built ([`Sent::order`]).
    main_built: u32,
}

/// A chunk out on the reply's line, as [`spare_pays`] reads it: when the far
/// end should finish its check and its answer be back — `None` until the
/// far end's pace and the round trip are measured — and the drafter's chance
/// it refuses a guess.
struct LineOut {
    seq: u32,
    done: Option<std::time::Instant>,
    due: Option<std::time::Instant>,
    refused: f64,
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
                branch: None,
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
            answers_waited_ms: 0.0,
            began: std::time::Instant::now(),
            positions_served: None,
            width: StreamWidth::new(),
            spares_on: false,
            spares: Vec::new(),
            near_cut: None,
            near_valid_to: None,
            spares_sent: 0,
            spares_taken: 0,
            spares_declined: 0,
            line_out: std::collections::VecDeque::new(),
            spare_ms: RecentMedian::default(),
            main_built: 0,
        };
        run.positions_served = run.positions_served();
        run.spares_on = run.spares_possible();
        let started = std::time::Instant::now();
        let outcome = run.run().await;
        // The rounds the price will read for this reply (FUTURE_WORK #3 (d)):
        // the round trips the stream waited, its chunks overlapping.
        // A spare that was not taken was skipped at the far end, unrun: not
        // a forward the reply waited on.
        let rounds_waited = crate::daemon::state::split_rounds::round_trips_waited(
            started.elapsed().as_secs_f64() * 1000.0,
            run.least_answer_ms,
            run.turn
                .saturating_sub(run.spares_sent.saturating_sub(run.spares_taken)),
        );
        run.state.note_decode_rounds(self.request.id, rounds_waited);
        tracing::info!(
            request_id = %self.request.id,
            shape = shape_name,
            peers,
            chunks = run.chunks,
            rounds_waited,
            least_answer_ms = format_args!("{:.0}", run.least_answer_ms.unwrap_or(0.0)),
            answers_waited_ms = format_args!("{:.0}", run.answers_waited_ms),
            kept_whole = run.kept_whole,
            restarts = run.restarts,
            held = run.held,
            guesses_per_chunk = run.width.narrow,
            wide_guesses = run.width.wide,
            chunks_wide = run.width.sent_wide,
            check_slope_measured = run.reply.check.slope_is_measured(),
            far_narrow_ms = format_args!("{:.0}", run.reply.far_pace.narrow.median().unwrap_or(0.0)),
            far_wide_ms = format_args!("{:.0}", run.reply.far_pace.wide.median().unwrap_or(0.0)),
            window = chunks_in_flight(),
            spares_on = run.spares_on,
            spares_sent = run.spares_sent,
            spares_taken = run.spares_taken,
            spares_declined = run.spares_declined,
            spare_ms = format_args!("{:.0}", run.spare_ms.median().unwrap_or(0.0)),
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
                        let chunk = self.send_chunk(out.len()).await?;
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
                Taken::Spare(wait) => {
                    *out = FuturesOrdered::new();
                    out.push_back(wait);
                }
                Taken::Done => return Ok(()),
            }
        }
    }

    /// Guess, run this node's layers over the chunk, send it to the far
    /// segment, and hand back the wait on its answer — with, where one pays,
    /// the chunk's spare restart sent beside it (#381), its wait kept in
    /// [`StreamRun::spares`].
    async fn send_chunk(&mut self, main_out: usize) -> Result<Waiting, SwarmError> {
        let alone = main_out == 0;
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
            self.reply.far_pace,
            self.reply.draft_cost.median(),
            chunks_in_flight(),
        );
        let want = chunk_guesses(
            width,
            room,
            self.positions_served
                .map(|served| served.saturating_sub(self.frontier.index_pos)),
        );
        let spares_on = self.spares_on;
        let (guesses, runner_ups): (Vec<u32>, Vec<_>) = if self.drafting_off || want == 0 {
            (Vec::new(), Vec::new())
        } else {
            let drafted = if spares_on {
                self.drafter
                    .draft_with_runner_ups(
                        &self.state,
                        want,
                        sampling,
                        &self.frontier.history,
                        self.io.noise.map(|n| n.seed()),
                        super::engine_drafter::stream_confidence_floor(),
                        exec.request.cancel.clone(),
                    )
                    .await
                    .map(|d| (d.tokens, d.runner_ups))
            } else {
                self.drafter
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
                    .map(|guesses| (guesses, Vec::new()))
            };
            match drafted {
                Ok((guesses, runner_ups)) => {
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
                    (guesses, runner_ups)
                }
                Err(e) => {
                    tracing::warn!(
                        %request_id,
                        error = %e,
                        "DSD: the drafter failed — finishing this reply without guessing ahead"
                    );
                    self.drafting_off = true;
                    (Vec::new(), Vec::new())
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
        let branch = self.frontier.branch.take();
        let named = lookahead_to_send(
            &self.shape,
            lookahead,
            names_the_lookahead()
                && self.state.peer_advertises_feature(
                    &self.shape.far.node_id,
                    swarmllm_types::node::features::STREAM_LOOKAHEAD,
                ),
        );
        // A check's cost is timed from here, drafting excluded — the rounds'
        // `CheckCost` measures the same span.
        let checking = std::time::Instant::now();
        let seq = stream_seq::compose(self.attempt, self.turn).ok_or_else(|| {
            SwarmError::Internal("a streamed reply ran past its checks' numbering".into())
        })?;
        self.turn += 1;
        let order = self.main_built;
        self.main_built += 1;
        // No round is counted here: chunks overlap, so the stream counts the
        // round trips it waited when it ends (`split_rounds::
        // round_trips_waited`, FUTURE_WORK #3 (d)).

        // This node's own layers before the peer's, one segment after another.
        let near_started = std::time::Instant::now();
        let (activations, near_from) = self.near_pass(index_pos, truncate, &rows, seq).await?;
        if !self.shape.near.is_empty() {
            self.width.note_near(
                rows.len() as u32,
                near_started.elapsed().as_secs_f64() * 1000.0,
            );
        }

        let history = self.frontier.history.clone();
        let chunk_history = history.len();
        let wait = self
            .send_to_far(FarChunk {
                seq,
                index_pos,
                activations,
                drafts,
                lookahead,
                named,
                history,
                truncate,
                branch,
                alone,
                order,
                started: checking,
            })
            .await?;
        if spares_on {
            let refused = chance_of_a_refusal(&guesses, &runner_ups, self.width.acceptance.alpha());
            let entry = self.line_entry(seq, rows.len(), refused, std::time::Instant::now(), None);
            self.line_out.push_back(entry);
        }
        if lookahead.is_some() {
            self.frontier.index_pos += rows.len() as u32;
            self.frontier.history.extend_from_slice(&guesses);
        }
        if lookahead.is_none() || (guesses.len() as u32) < want {
            self.frontier.held_by = Some(seq);
            self.held += 1;
        }

        // The spare goes after the chunk it stands beside, built while the far
        // end checks that chunk: the far end runs a stream in its order, a
        // spare runs only once that chunk's walk has said which refusal it
        // made, and the chunk itself waits for nothing of the spare's.
        if spares_on && lookahead.is_some() && room_for_a_spare(main_out, self.spares.len()) {
            self.send_spare(ChunkSent {
                seq,
                order,
                index_pos,
                history: chunk_history,
                guesses: &guesses,
                runner_ups: &runner_ups,
                named: named.is_some(),
                near_from,
            })
            .await?;
        }
        Ok(wait)
    }

    /// A chunk out on the reply's line, sent at `sent` with `rows` rows, as
    /// [`spare_pays`] reads it: the far end runs it on arrival or once it
    /// finishes the chunk before it (`after`, else the line's last), at its
    /// measured pace for that width ([`FarPace`]), and its answer is back a
    /// half round trip after that. No times until both are measured.
    fn line_entry(
        &self,
        seq: u32,
        rows: usize,
        refused: f64,
        sent: std::time::Instant,
        after: Option<std::time::Instant>,
    ) -> LineOut {
        let wide =
            u32::try_from(rows).is_ok_and(|r| r == self.width.wide && r != self.width.narrow);
        let pace = &*self.reply.far_pace;
        let far_ms = pace.at(wide).median().or_else(|| pace.at(!wide).median());
        let times = self
            .reply
            .check
            .floor_ms()
            .zip(far_ms)
            .and_then(|(round_trip, far)| {
                let half =
                    std::time::Duration::try_from_secs_f64(round_trip.max(0.0) / 2000.0).ok()?;
                let far = std::time::Duration::try_from_secs_f64(far.max(0.0) / 1000.0).ok()?;
                let arrives = sent + half;
                let before = after.or_else(|| self.line_out.back().and_then(|o| o.done));
                let done = before.map_or(arrives, |b| b.max(arrives)) + far;
                Some((done, done + half))
            });
        LineOut {
            seq,
            done: times.map(|t| t.0),
            due: times.map(|t| t.1),
            refused,
        }
    }

    /// [`spare_pays`] for the spare of the chunk last sent, which `score`
    /// bets on, from what this reply has measured; no spare until the far
    /// end's pace, the round trip and the drafter's cost are known.
    fn spare_pays_now(&self, score: f64) -> bool {
        let now = std::time::Instant::now();
        let ms = |t: std::time::Instant| match t.checked_duration_since(now) {
            Some(ahead) => ahead.as_secs_f64() * 1000.0,
            None => -(now.duration_since(t).as_secs_f64() * 1000.0),
        };
        let Some(this) = self.line_out.back() else {
            return false;
        };
        let Some(this_done) = this.done.map(ms) else {
            return false;
        };
        let mut earlier = Vec::with_capacity(self.line_out.len());
        for out in self.line_out.iter().take(self.line_out.len() - 1) {
            let Some(due) = out.due else {
                return false;
            };
            earlier.push((ms(due), out.refused));
        }
        let (Some(round_trip), Some(draft_ms)) =
            (self.reply.check.floor_ms(), self.reply.draft_cost.median())
        else {
            return false;
        };
        let near = self.width.near_ms[0]
            .median()
            .or_else(|| self.width.near_ms[1].median())
            .unwrap_or(0.0);
        let build = draft_ms * f64::from(self.width.narrow) + near;
        let spare = self
            .spare_ms
            .median()
            .unwrap_or(draft_ms * f64::from(spare_guesses()) + near);
        spare_pays(&SpareTerms {
            score,
            earlier: &earlier,
            this_refused: this.refused,
            this_done,
            room: chunks_in_flight().saturating_sub(self.line_out.len()),
            build,
            spare,
            round_trip,
        })
    }

    /// Build and send the spare restart of the chunk just sent (#381), if
    /// one pays: drafted ([`StreamRun::draft_spare`]), its rows run through
    /// this node's layers as a branch of the chunk's cached rows
    /// ([`StreamRun::near_spare`]), sent to the far end behind the chunk and
    /// kept in [`StreamRun::spares`]. Nothing here fails the chunk: a spare
    /// this node's layers refuse turns spares off for the reply.
    async fn send_spare(&mut self, chunk: ChunkSent<'_>) -> Result<(), SwarmError> {
        let building = std::time::Instant::now();
        let Some(spare) = self
            .draft_spare(
                chunk.guesses,
                chunk.runner_ups,
                chunk.named,
                chunk.index_pos,
                chunk.history,
            )
            .await
        else {
            return Ok(());
        };
        let Some((&spare_lookahead, spare_drafts)) = spare.guesses.split_last() else {
            return Ok(());
        };
        let spare_drafts = spare_drafts.to_vec();
        let checking = std::time::Instant::now();
        // The chunk's rows — its bootstrap and every guess but the
        // look-ahead — end where its look-ahead sits.
        let chunk_end = chunk.index_pos + chunk.guesses.len() as u32;
        let activations = match self
            .near_spare(
                chunk.near_from,
                chunk_end,
                (chunk.index_pos - chunk.near_from) as usize + spare.after_guess,
                &spare.rows,
                chunk.seq,
            )
            .await
        {
            Ok(activations) => activations,
            Err(e) => {
                tracing::warn!(
                    request_id = %self.exec.request.id,
                    error = %e,
                    "DSD: this node's layers refused a spare restart's rows — no more spares this reply"
                );
                self.spares_on = false;
                return Ok(());
            }
        };
        self.spare_ms
            .record(building.elapsed().as_secs_f64() * 1000.0);
        let spare_seq = stream_seq::compose(self.attempt, self.turn).ok_or_else(|| {
            SwarmError::Internal("a streamed reply ran past its checks' numbering".into())
        })?;
        self.turn += 1;
        let spare_named = lookahead_to_send(&self.shape, Some(spare_lookahead), chunk.named);
        let kept = spare.after_guess as u32;
        let wait = self
            .send_to_far(FarChunk {
                seq: spare_seq,
                index_pos: spare.index_pos,
                activations,
                drafts: spare_drafts,
                lookahead: Some(spare_lookahead),
                named: spare_named,
                history: spare.history,
                truncate: Some(spare.index_pos),
                branch: Some(crate::types::StreamBranch::Spare {
                    of: chunk.seq,
                    kept,
                    token: spare.token,
                }),
                alone: false,
                order: chunk.order,
                started: checking,
            })
            .await?;
        tracing::debug!(
            request_id = %self.exec.request.id,
            of = stream_seq::turn(chunk.seq),
            spare = stream_seq::turn(spare_seq),
            kept,
            score = spare.score,
            "DSD: a spare restart sent"
        );
        self.spares.push(SpareOut {
            of: chunk.seq,
            kept,
            token: spare.token,
            seq: spare_seq,
            index_pos: spare.index_pos,
            guesses: spare.guesses,
            score: spare.score,
            left: std::time::Instant::now(),
            wait,
        });
        self.spares_sent += 1;
        Ok(())
    }

    /// Whether this reply's chunks may carry spare restarts (#381): switched
    /// on, the far end a LONE segment that walks (its walk says which refusal
    /// it made; a chain's tail is a hop away) and runs spares
    /// (`features::STREAM_SPARES`), a greedy request (a runner-up bets on a
    /// greedy pick — a sampled check's token is a draw), and, where this node
    /// runs layers before the far end, a family whose layers read a tree:
    /// the spare's rows ride in the chunk's near forward as a branch.
    fn spares_possible(&self) -> bool {
        let reads_trees = || {
            self.state
                .gguf_meta_for(&self.shape.far.shard_id.model_id)
                .is_some_and(|m| {
                    crate::inference::model_arch::ModelArch::from_gguf_arch(&m.architecture)
                        .reads_draft_trees()
                })
        };
        spares_switched_on()
            && self.shape.far_walks()
            && self.shape.chain.is_empty()
            && self.io.noise.is_none()
            && self.state.peer_advertises_feature(
                &self.shape.far.node_id,
                swarmllm_types::node::features::STREAM_SPARES,
            )
            && (self.shape.near.is_empty() || reads_trees())
    }

    /// Draft the spare restart for a chunk of `guesses` (#381), if one pays
    /// ([`spare_at`]): its first row is the drafter's runner-up at the guess
    /// it bets the check refuses, its guesses the drafter's after that. The
    /// drafter's line is put back to the chunk's afterwards: the spare is a
    /// branch the reply takes only if the far end runs it
    /// ([`StreamRun::take_answer`]). `index_pos` is where the chunk started
    /// and `history` how much of the frontier's history was there then: the
    /// chunk has already moved the frontier past itself.
    async fn draft_spare(
        &mut self,
        guesses: &[u32],
        runner_ups: &[crate::inference::worker_ipc::RunnerUp],
        lookahead_named: bool,
        chunk_index_pos: u32,
        history: usize,
    ) -> Option<SparePlan> {
        let fixed = fixed_spare_floor();
        let (j, token, score) =
            spare_at(guesses, runner_ups, lookahead_named, fixed.unwrap_or(0.0))?;
        if fixed.is_none() && !self.spare_pays_now(f64::from(score)) {
            self.spares_declined += 1;
            return None;
        }
        // A reply that would end at the guesses kept, or at the bet token, has
        // nothing after it to restart.
        if guesses[..j]
            .iter()
            .chain(std::iter::once(&token))
            .any(|t| self.io.eos.contains(t))
        {
            return None;
        }
        let index_pos = chunk_index_pos + 1 + j as u32;
        // The drafter's sequence ends with the chunk's guesses; guess `j` sits
        // at the spare's first position.
        let at = self.drafter.tokens().len().checked_sub(guesses.len() - j)?;
        if at != index_pos as usize {
            tracing::debug!(
                request_id = %self.exec.request.id,
                at,
                index_pos,
                "DSD: the drafter's sequence is not where the stream is — no spare"
            );
            return None;
        }
        let mut history = self.frontier.history.get(..history)?.to_vec();
        history.extend_from_slice(&guesses[..j]);
        history.push(token);
        let room = self.io.max_tokens.saturating_sub(history.len() as u32);
        let want = chunk_guesses(
            spare_guesses(),
            room,
            self.positions_served
                .map(|served| served.saturating_sub(index_pos)),
        );
        if want == 0 {
            return None;
        }
        self.drafter.rewind(at);
        self.drafter.push(&[token]);
        let drafted = self
            .drafter
            .draft(
                &self.state,
                want,
                &self.exec.request.sampling_params,
                &history,
                None,
                super::engine_drafter::stream_confidence_floor(),
                self.exec.request.cancel.clone(),
            )
            .await;
        // Back to the chunk's line, which the next chunk is drafted on.
        self.drafter.rewind(at);
        self.drafter.push(&guesses[j..]);
        let spare = match drafted {
            Ok(spare) if !spare.is_empty() => spare,
            Ok(_) => return None,
            Err(e) => {
                tracing::debug!(
                    request_id = %self.exec.request.id,
                    error = %e,
                    "DSD: a spare restart's guesses failed — the chunk goes without one"
                );
                return None;
            }
        };
        let rows = std::iter::once(token)
            .chain(spare[..spare.len() - 1].iter().copied())
            .collect();
        Some(SparePlan {
            after_guess: j,
            token,
            index_pos,
            rows,
            guesses: spare,
            history,
            score,
        })
    }

    /// This node's own segments over a chunk's `rows` from `index_pos`,
    /// handing back what goes to the far end — the chunk's rows — and where
    /// this forward's rows start in this node's caches (`index_pos`, or
    /// earlier where the reply's rows the caches lost to a taken spare are
    /// read again first, [`StreamRun::near_valid_to`]). A spare's rows left
    /// past the line by the forward before are cut ([`StreamRun::near_cut`]).
    /// With no segment of this node's before the far end, the rows go as
    /// token ids.
    async fn near_pass(
        &mut self,
        index_pos: u32,
        truncate: Option<u32>,
        rows: &[u32],
        seq: u32,
    ) -> Result<(Vec<u8>, u32), SwarmError> {
        if self.shape.near.is_empty() {
            return Ok((super::pack_verify_tokens_to_le_bytes(rows), index_pos));
        }
        let near_from = self
            .near_valid_to
            .take()
            .map_or(index_pos, |valid| valid.min(index_pos));
        // The reply's tokens from `near_from` up to the chunk's bootstrap,
        // which ends `history` at `index_pos`.
        let reread: Vec<u32> = {
            let back = (index_pos - near_from) as usize;
            let end = self.frontier.history.len().saturating_sub(1);
            let from = end.checked_sub(back).ok_or_else(|| {
                SwarmError::Internal(format!(
                    "streamed check: {back} rows to read again before a reply of {}",
                    self.frontier.history.len()
                ))
            })?;
            self.frontier.history[from..end].to_vec()
        };
        let cut = if reread.is_empty() {
            [truncate, self.near_cut.take()].into_iter().flatten().min()
        } else {
            self.near_cut = None;
            Some(near_from)
        };
        let mut tokens = reread;
        let reread = tokens.len();
        tokens.extend_from_slice(rows);
        let answered = self
            .near_forwards(near_from, cut, &tokens, None, seq)
            .await?;
        Ok((
            rows_of(&answered, tokens.len(), reread, tokens.len())?,
            near_from,
        ))
    }

    /// This node's own segments over a spare restart's `rows` (#381): a
    /// branch of the chunk whose rows its caches hold from `near_from` to
    /// `chunk_end`, standing under the chunk's row `after` (counted from
    /// `near_from`) — a tree that continues the cached rows
    /// (`LayerForward::draft_tree_from`, the flow's shape), so the chunk went
    /// before it with nothing of the spare's. The spare's rows go to the far
    /// end; in this node's caches they sit past the line, so the next forward
    /// cuts there ([`StreamRun::near_cut`]) whatever this one does.
    async fn near_spare(
        &mut self,
        near_from: u32,
        chunk_end: u32,
        after: usize,
        rows: &[u32],
        seq: u32,
    ) -> Result<Vec<u8>, SwarmError> {
        if self.shape.near.is_empty() {
            return Ok(super::pack_verify_tokens_to_le_bytes(rows));
        }
        self.near_cut = Some(chunk_end);
        let tree = near_tree((chunk_end - near_from) as usize, after, rows.len());
        self.near_forwards(chunk_end, None, rows, Some((&tree, near_from)), seq)
            .await
    }

    /// One forward of `tokens` from `index_pos` through each of this node's
    /// segments before the far end, cut to `cut` first, shaped by `tree`
    /// where one is given; the last segment's activations. `&mut self` only
    /// so the future is `Send` (see [`StreamRun::finish_here`]).
    async fn near_forwards(
        &mut self,
        index_pos: u32,
        cut: Option<u32>,
        tokens: &[u32],
        tree: Option<(&[u16], u32)>,
        seq: u32,
    ) -> Result<Vec<u8>, SwarmError> {
        let exec = self.exec;
        let mut activations = super::pack_verify_tokens_to_le_bytes(tokens);
        for segment in &self.shape.near {
            let mut forward = super::build_spec_verify_forward(
                exec.request.id,
                index_pos,
                activations,
                segment,
                cut,
                None,
                Some(seq),
            );
            if let Some((tree, from)) = tree {
                forward.draft_parents = tree.to_vec();
                forward.draft_tree_from = Some(from);
            }
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
        Ok(activations)
    }

    /// Send one chunk — the main line's or a spare — to the far segment and
    /// hand back the wait on its answer.
    async fn send_to_far(&mut self, chunk: FarChunk) -> Result<Waiting, SwarmError> {
        let exec = self.exec;
        let request_id = exec.request.id;
        let sampling = &exec.request.sampling_params;
        let FarChunk {
            seq,
            index_pos,
            activations,
            drafts,
            lookahead,
            named,
            history,
            truncate,
            branch,
            alone,
            order,
            started,
        } = chunk;
        let rows = drafts.len() + 1;
        // The walk goes to the peer only where it samples. In the boomerang it
        // is sent no guess, no history and no sampler — hidden states alone,
        // as on every other step of a reply that starts and finishes here.
        let walk = TailWalk {
            drafts: &drafts,
            sampling,
            generated: &history,
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
        forward.stream_lookahead = named;
        forward.stream_branch = branch;
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
        let left = std::time::Instant::now();
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

        let sent = Sent {
            seq,
            index_pos,
            rows,
            drafts,
            lookahead,
            history,
            truncate,
            far_input,
            alone,
            order,
            started,
            left,
        };
        let state = self.state.clone();
        let network_tx = exec.network_tx.clone();
        let node = far.node_id.clone();
        let cancel = exec.request.cancel.clone();
        let segment_idx = self.shape.near.len();
        let chained = !self.shape.chain.is_empty();
        let wait = stamp_on_arrival(async move {
            let _waiter = waiter;
            let rebuild = move || forward.clone();
            PipelineExecutor::wait_for_result(
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
            .await
        });
        Ok(Box::pin(async move {
            let (result, arrived) = wait.await.unwrap_or_else(|e| {
                // Only a panic ends it so: an abort comes with this future's drop.
                (
                    Err(SwarmError::Internal(format!(
                        "streamed check's wait failed: {e}"
                    ))),
                    std::time::Instant::now(),
                )
            });
            (sent, result, arrived)
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
    async fn take_answer(&mut self, (sent, result, arrived): Answer) -> Result<Taken, SwarmError> {
        let taken = std::time::Instant::now();
        self.line_out.retain(|out| out.seq != sent.seq);
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
        // The check ran from this node's first layer until its answer arrived,
        // and on through this node's last leg where the chain finishes here —
        // never the time the answer sat waiting for the loop to take it.
        let last_leg_ms = if sent.far_input.is_some() {
            taken.elapsed().as_secs_f64() * 1000.0
        } else {
            0.0
        };
        let answered_ms = arrived
            .saturating_duration_since(sent.started)
            .as_secs_f64()
            * 1000.0
            + last_leg_ms;
        let waited_ms = taken.saturating_duration_since(arrived).as_secs_f64() * 1000.0;
        self.answers_waited_ms += waited_ms;
        self.width.note_arrival(
            sent.order,
            sent.rows,
            sent.left,
            arrived,
            self.reply.check.floor_ms(),
            self.reply.far_pace,
        );
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
        tracing::debug!(
            request_id = %self.exec.request.id,
            turn = stream_seq::turn(sent.seq),
            rows = sent.rows,
            alone = sent.alone,
            check_ms = format_args!("{answered_ms:.0}"),
            waited_ms = format_args!("{waited_ms:.0}"),
            started_at_ms = sent.started.saturating_duration_since(self.began).as_millis(),
            arrived_at_ms = arrived.saturating_duration_since(self.began).as_millis(),
            kept = kept.len() + usize::from(whole),
            of = guessed,
            "DSD: a streamed check answered"
        );
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
            // Its look-ahead was right: the frontier already stands on it,
            // and its spare, if it had one, is skipped at the far end.
            self.spares.retain(|s| s.of != sent.seq);
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
        // The spare for exactly this refusal ran at the far end (#381) — its
        // rule there is this one (`forward_streams`) — and every other spare
        // out stood beside a chunk this refusal ended.
        let taken = self
            .spares
            .iter()
            .position(|s| {
                s.of == sent.seq
                    && s.kept as usize == kept.len()
                    && s.token == last
                    && s.index_pos == next_pos
            })
            .map(|i| self.spares.swap_remove(i));
        self.spares.clear();
        if let Some(spare) = taken {
            return Ok(self.take_spare(spare, next_pos));
        }
        self.frontier = Frontier {
            index_pos: next_pos,
            history: self.reply.generated.clone(),
            truncate: Some(next_pos),
            held_by: None,
            branch: None,
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
                self.line_out.clear();
                Taken::Restart
            }
            // A chunk without a look-ahead had nothing built after it.
            None => Taken::Continue,
        })
    }

    /// The far end ran the spare restart for the refusal just taken (#381):
    /// the reply goes on from its rows as from a chunk sent now, a round trip
    /// sooner than the restart would have reached the far end. The drafter
    /// and the frontier stand on its guesses, the next chunk is built on it
    /// (`StreamBranch::AfterSpare`), and this node's caches — which held the
    /// spare's rows only as a branch — read the reply again from its first
    /// row ([`StreamRun::near_valid_to`]).
    fn take_spare(&mut self, spare: SpareOut, next_pos: u32) -> Taken {
        self.spares_taken += 1;
        self.restarts += 1;
        tracing::debug!(
            request_id = %self.exec.request.id,
            of = stream_seq::turn(spare.of),
            spare = stream_seq::turn(spare.seq),
            kept = spare.kept,
            score = spare.score,
            "DSD: a streamed check refused a guess — its spare restart already ran"
        );
        self.drafter.push(&spare.guesses);
        // The spare is the line's only chunk out now; the far end runs it as
        // it finishes the refused chunk, which it did half a round trip ago.
        self.line_out.clear();
        let finished = self
            .reply
            .check
            .floor_ms()
            .and_then(|rt| std::time::Duration::try_from_secs_f64(rt.max(0.0) / 2000.0).ok())
            .and_then(|half| std::time::Instant::now().checked_sub(half));
        let refused = chance_of_a_refusal(&spare.guesses, &[], self.width.acceptance.alpha());
        let entry = self.line_entry(
            spare.seq,
            spare.guesses.len(),
            refused,
            spare.left,
            finished,
        );
        self.line_out.push_back(entry);
        let mut history = self.reply.generated.clone();
        history.extend_from_slice(&spare.guesses);
        let index_pos = next_pos + spare.guesses.len() as u32;
        self.frontier = Frontier {
            index_pos,
            history,
            truncate: Some(index_pos),
            held_by: None,
            branch: Some(crate::types::StreamBranch::AfterSpare { spare: spare.seq }),
        };
        self.near_valid_to = Some(
            self.near_valid_to
                .map_or(next_pos, |valid| valid.min(next_pos)),
        );
        Taken::Spare(spare.wait)
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

    /// #382: the look-ahead goes only to a lone far segment that walks and
    /// reads it — an older far end, a chained run or the boomerang is told
    fn runner_up(guess_p: f32, token: u32, p: f32) -> crate::inference::worker_ipc::RunnerUp {
        crate::inference::worker_ipc::RunnerUp { guess_p, token, p }
    }

    /// A spare bets on the refusal the drafter thinks likeliest: every guess
    /// before it kept, then its runner-up drawn — not the likeliest runner-up
    /// alone, which a guess deep in the chunk rarely reaches. The look-ahead
    /// is a candidate only where the far end is told it; below the floor, or
    /// with runner-ups that do not answer every guess, there is none.
    #[test]
    fn a_spare_bets_on_the_refusal_its_drafter_thinks_likeliest() {
        let guesses = [10, 11, 12];
        let ups = [
            runner_up(0.9, 20, 0.05),
            runner_up(0.5, 21, 0.3),
            runner_up(0.6, 22, 0.4),
        ];
        // 0.05 at the first; 0.9 x 0.3 = 0.27 at the second; 0.9 x 0.5 x 0.4
        // = 0.18 at the look-ahead, whose runner-up alone is the likeliest.
        let (j, token, score) = spare_at(&guesses, &ups, true, 0.08).unwrap();
        assert_eq!((j, token), (1, 21));
        assert!((score - 0.27).abs() < 1e-6);
        assert_eq!(spare_at(&guesses, &ups, false, 0.08).map(|s| s.0), Some(1));
        assert_eq!(spare_at(&guesses, &ups, true, 0.3), None, "below the floor");
        assert_eq!(spare_at(&guesses, &ups[..2], true, 0.0), None);
        assert_eq!(spare_at(&[], &[], true, 0.0), None);
        // Only the look-ahead clears the floor, and it is not named.
        let ups = [
            runner_up(0.9, 20, 0.0),
            runner_up(0.9, 21, 0.0),
            runner_up(0.9, 22, 0.9),
        ];
        assert_eq!(spare_at(&guesses, &ups, true, 0.5).map(|s| s.0), Some(2));
        assert_eq!(spare_at(&guesses, &ups, false, 0.5), None);
        // A runner-up that is the guess itself is no other outcome.
        let ups = [
            runner_up(0.5, 10, 0.5),
            runner_up(0.9, 21, 0.0),
            runner_up(0.9, 22, 0.0),
        ];
        assert_eq!(spare_at(&guesses, &ups, true, 0.1), None);
    }

    /// The 70B's split as measured (#381): a 400 ms round trip, ~312 ms to
    /// build a chunk here and as much for a spare.
    fn terms(score: f64, earlier: &[(f64, f64)], this_done: f64, room: usize) -> SpareTerms<'_> {
        SpareTerms {
            score,
            earlier,
            this_refused: 0.2,
            this_done,
            room,
            build: 312.0,
            spare: 312.0,
            round_trip: 400.0,
        }
    }

    /// A spare built while this node would only have waited costs nothing,
    /// so any chance of running pays for it: the window is full and the
    /// oldest answer is not due until the spare is long built.
    #[test]
    fn a_spare_built_while_this_node_would_wait_pays_at_any_chance() {
        let earlier = [(600.0, 0.3)];
        assert!(spare_pays(&terms(0.05, &earlier, 1000.0, 0)));
        assert!(!spare_pays(&terms(0.0, &earlier, 1000.0, 0)), "no chance");
    }

    /// A spare holds back the answer already in until it is built: where that
    /// answer is likely a refusal, the restart it calls for waits, and only a
    /// spare likely to run pays for that. The fixed floor this rule replaced
    /// sent the 0.45 spare.
    #[test]
    fn a_spare_that_holds_back_a_likely_refusal_pays_only_if_likely_to_run() {
        // 0.6 x 312 ms of a restart held back; it runs only if the earlier
        // chunk is kept (0.4), and then saves 400 + 312 ms.
        let earlier = [(-10.0, 0.6)];
        assert!(!spare_pays(&terms(0.45, &earlier, 1000.0, 0)));
        assert!(spare_pays(&terms(0.9, &earlier, 1000.0, 0)));
        // The same answer due after the spare is built: nothing held back.
        let later = [(400.0, 0.6)];
        assert!(spare_pays(&terms(0.45, &later, 1000.0, 0)));
    }

    /// Every chunk kept, the far end finishes this one at 600 ms: the next
    /// chunk built now reaches it at 512, built after the spare at 824 — the
    /// far end waits 224 ms four times in five, which a spare must outweigh.
    #[test]
    fn a_spare_that_leaves_the_far_end_waiting_is_charged_the_wait() {
        assert!(!spare_pays(&terms(0.2, &[], 600.0, 1)));
        assert!(spare_pays(&terms(0.3, &[], 600.0, 1)));
        // A far end busy for longer waits for nothing.
        assert!(spare_pays(&terms(0.05, &[], 1200.0, 1)));
    }

    /// A chunk's chance of a refusal is one less the chance its drafter keeps
    /// every guess; where the runner-ups do not answer them all, the reply's
    /// acceptance stands in.
    #[test]
    fn a_chunks_chance_of_a_refusal_is_its_guesses_kept_all_together() {
        let ups = [runner_up(0.9, 20, 0.05), runner_up(0.5, 21, 0.3)];
        assert!((chance_of_a_refusal(&[10, 11], &ups, 0.7) - 0.55).abs() < 1e-6);
        assert!((chance_of_a_refusal(&[10, 11], &ups[..1], 0.7) - 0.51).abs() < 1e-6);
        assert_eq!(chance_of_a_refusal(&[], &[], 0.7), 0.0);
    }

    /// A spare's rows ride in its chunk's near forward as a branch: its first
    /// row follows the row before the guess it stands at, at that guess's
    /// position, and each of its rows the one before — after any rows read
    /// again ahead of the chunk.
    #[test]
    fn a_spares_rows_branch_where_its_guess_stood() {
        use crate::inference::draft_tree::DraftTree;
        // Chunk rows [b, d0, d1]; a spare of 2 at guess 1 (d1's place).
        let wire = near_tree(3, 1, 2);
        assert_eq!(wire, vec![0, 1, 2, 2, 4]);
        let tree = DraftTree::from_wire(100, &wire).unwrap();
        let depths: Vec<usize> = (0..tree.len()).map(|r| tree.depth(r)).collect();
        assert_eq!(
            depths,
            vec![0, 1, 2, 2, 3],
            "the spare's first row sits where d1 does"
        );
        assert_eq!(tree.parent(3), Some(1));
        // Two rows read again first, then the chunk; a spare at the look-ahead.
        let wire = near_tree(5, 2 + 2, 3);
        let tree = DraftTree::from_wire(100, &wire).unwrap();
        let depths: Vec<usize> = (0..tree.len()).map(|r| tree.depth(r)).collect();
        assert_eq!(depths, vec![0, 1, 2, 3, 4, 5, 6, 7]);
        assert_eq!(tree.parent(5), Some(4));
        assert_eq!(tree.parent(6), Some(5));
    }

    /// Spares never take a stream past the window the far end's cap is sized
    /// for: at the default window every chunk may carry one, at the widest
    /// none once the main line fills it.
    #[test]
    fn a_spare_goes_only_where_the_far_end_has_room_for_it() {
        let window = crate::daemon::state::forward_streams::MAX_STREAM_WINDOW as usize;
        // The default window (3): the third chunk and its spare still fit.
        assert!(room_for_a_spare(2, 2));
        assert!(room_for_a_spare(0, 0));
        // The widest window: the last main chunk goes alone.
        assert!(!room_for_a_spare(window - 1, 0));
        assert!(room_for_a_spare(window - 2, 0));
        assert!(!room_for_a_spare(window - 2, 1));
    }

    /// A near forward's answer — hidden states with a shape header, f32 or
    /// Q8_0, as this node's layers send them — is cut into the chunk's rows
    /// and the spare's, each the rows the narrowed tensor holds; a whole
    /// answer passes untouched; a count the header does not hold is refused.
    /// The first version cut BYTES into equal parts, and every 7B check on the
    /// rig (3 rows of 3,584 in Q8_0: 11,444 bytes) was refused.
    #[test]
    fn a_near_answer_is_cut_into_whole_rows() {
        use crate::inference::tensor_util::{
            bytes_to_tensor, tensor_to_bytes, tensor_to_bytes_q8_0,
        };
        use candle_core::{Device, Tensor};
        let hidden = |rows: usize, width: usize| {
            let values: Vec<f32> = (0..rows * width)
                .map(|i| (i % 97) as f32 * 0.03 - (i / width) as f32)
                .collect();
            Tensor::from_vec(values, (1, rows, width), &Device::Cpu).unwrap()
        };
        let rows_in = |bytes: &[u8]| bytes_to_tensor(bytes).unwrap().dims()[1];
        // The rig's shape, and a Q8_0 width that ends inside a block.
        for (width, encode) in [
            (3584, tensor_to_bytes_q8_0 as fn(&Tensor) -> _),
            (3584, tensor_to_bytes),
            (48, tensor_to_bytes_q8_0),
        ] {
            let whole = hidden(3, width);
            let bytes = encode(&whole).unwrap();
            assert_eq!(
                rows_of(&bytes, 3, 0, 3).unwrap(),
                bytes,
                "whole passes as it is"
            );
            let decoded = bytes_to_tensor(&bytes).unwrap();
            for (from, to) in [(0, 2), (2, 3), (1, 3)] {
                let cut = rows_of(&bytes, 3, from, to).unwrap();
                assert_eq!(rows_in(&cut), to - from);
                let want: Vec<f32> = decoded
                    .narrow(1, from, to - from)
                    .unwrap()
                    .flatten_all()
                    .unwrap()
                    .to_vec1()
                    .unwrap();
                let got: Vec<f32> = bytes_to_tensor(&cut)
                    .unwrap()
                    .flatten_all()
                    .unwrap()
                    .to_vec1()
                    .unwrap();
                assert_eq!(got, want, "width {width}, rows {from}..{to}");
            }
            assert!(rows_of(&bytes, 4, 0, 1).is_err(), "the header holds 3");
            assert!(rows_of(&bytes, 3, 2, 4).is_err());
            assert!(rows_of(&bytes, 3, 1, 1).is_err());
        }
        assert!(
            rows_of(&[0u8; 12], 3, 0, 1).is_err(),
            "bare bytes are not rows"
        );
    }

    /// nothing (an older peer would fail to open an encrypted forward whose
    /// AAD carries a trailer it does not know).
    #[test]
    fn the_lookahead_goes_only_to_a_lone_far_end_that_walks_and_reads_it() {
        let every = |_: &NodeId, _: bool| true;
        let shape_over = |nodes: &[NodeId]| {
            let segments = plan(nodes);
            shape_of(&segments, &ME, every, every_hop_chains(&segments)).unwrap()
        };
        let lone = shape_over(&[ME, PEER]);
        assert_eq!(lookahead_to_send(&lone, Some(7), true), Some(7));
        assert_eq!(
            lookahead_to_send(&lone, Some(7), false),
            None,
            "older far end"
        );
        assert_eq!(
            lookahead_to_send(&lone, None, true),
            None,
            "no look-ahead drafted"
        );
        let chained = shape_over(&[ME, PEER, OTHER]);
        assert!(chained.far_walks() && !chained.chain.is_empty());
        assert_eq!(
            lookahead_to_send(&chained, Some(7), true),
            None,
            "the chain's tail is a hop away"
        );
        let boomerang = shape_over(&[ME, PEER, ME]);
        assert!(!boomerang.far_walks());
        assert_eq!(
            lookahead_to_send(&boomerang, Some(7), true),
            None,
            "the walk is ours"
        );
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

    /// A check's answer is timed when it arrives, not when the loop, busy
    /// building the next chunk, gets round to reading it (#375).
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn an_answer_is_timed_when_it_arrives_not_when_it_is_read() {
        let started = std::time::Instant::now();
        let wait = stamp_on_arrival(async {
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        });
        // The loop building a chunk for a second before it reads the answer.
        // Wide margins: a loaded parallel test run must not lose the race
        // (gotcha #270's shape).
        tokio::time::sleep(std::time::Duration::from_millis(1000)).await;
        let ((), arrived) = wait.await.expect("the wait ran");
        let arrived_ms = arrived.duration_since(started).as_millis();
        assert!(
            arrived_ms < 600,
            "stamped {arrived_ms} ms in: the read, not the arrival"
        );
    }

    /// Dropping a chunk's wait — a restart abandoning it — stops the wait, so
    /// what it holds (its `StreamWaiter`) is let go at once.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn dropping_a_timed_wait_withdraws_it() {
        struct Held(Arc<std::sync::atomic::AtomicBool>);
        impl Drop for Held {
            fn drop(&mut self) {
                self.0.store(true, Ordering::Release);
            }
        }
        let let_go = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let held = Held(let_go.clone());
        let wait = stamp_on_arrival(async move {
            let _held = held;
            tokio::time::sleep(std::time::Duration::from_secs(30)).await;
        });
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        drop(wait);
        for _ in 0..300 {
            if let_go.load(Ordering::Acquire) {
                return;
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
        panic!("the abandoned wait still held its waiter 3 s later");
    }

    /// The far end's time is the gap between two answers only where the later
    /// chunk was already waiting there — left here a round trip before the
    /// earlier answer came back — and was the very next chunk of the stream.
    #[test]
    fn the_far_ends_time_is_read_from_answers_that_waited_there() {
        let mut width = StreamWidth::with(3, 5, false);
        let mut far = FarPace::default();
        let t0 = std::time::Instant::now();
        let at = |ms| t0 + std::time::Duration::from_millis(ms);
        let near = |got: Option<f64>, want: f64| got.is_some_and(|g| (g - want).abs() < 1e-6);
        let trip = Some(436.0);
        width.note_arrival(0, 3, at(300), at(1300), trip, &mut far);
        assert!(
            far.narrow.is_empty(),
            "the first answer has nothing before it"
        );
        // Left at 600, a round trip before 1300: it waited there.
        width.note_arrival(1, 3, at(600), at(2060), trip, &mut far);
        assert!(
            near(far.narrow.median(), 760.0),
            "{:?}",
            far.narrow.median()
        );
        // Left at 1700, reached the far end after 2060's answer had left it.
        width.note_arrival(2, 5, at(1700), at(2900), trip, &mut far);
        assert!(far.wide.is_empty(), "the far end sat idle between them");
        // Turn 3 was dropped by a restart: 4's gap is not its own.
        width.note_arrival(4, 5, at(1000), at(3800), trip, &mut far);
        assert!(far.wide.is_empty(), "not the chunk before it");
        width.note_arrival(5, 5, at(2000), at(4640), trip, &mut far);
        assert!(near(far.wide.median(), 840.0), "{:?}", far.wide.median());
        // A chunk held short is neither width; a check with no peer's trip has
        // no far end to time.
        width.note_arrival(6, 2, at(2500), at(5300), trip, &mut far);
        width.note_arrival(7, 3, at(2600), at(6000), None, &mut far);
        assert_eq!((far.narrow.len(), far.wide.len()), (1, 1));
    }

    /// Once both widths have the far end's time, it prices them — at the 70B
    /// split's times (#375) the switch is near α 0.78; until then chunks that
    /// queue try the width without, and a width not chosen is still tried now
    /// and then.
    #[test]
    fn a_chunk_is_as_wide_as_the_far_ends_time_pays() {
        let (narrow, wide) = (3, 5);
        let mut width = StreamWidth::with(narrow, wide, false);
        let check = CheckCost::for_a_plan(Some(420));
        let mut far = FarPace::default();
        for _ in 0..FAR_PACE_SAMPLES {
            far.record(false, 760.0);
        }
        let queued: Vec<u32> = (0..4)
            .map(|_| width.next(false, &check, &far, Some(61.0), 3))
            .collect();
        assert_eq!(
            queued,
            [wide, narrow, wide, narrow],
            "queued, one width timed"
        );
        assert_eq!(
            width.next(true, &check, &far, Some(61.0), 3),
            narrow,
            "alone, one width timed"
        );

        // The 70B split's far end: two more rows cost it ~100 ms.
        for _ in 0..FAR_PACE_SAMPLES {
            far.record(true, 860.0);
        }
        width.note_near(narrow, 295.0);
        width.note_near(wide, 325.0);
        for _ in 0..12 {
            width.note_answer(2, 3);
        }
        assert_eq!(
            width.next(false, &check, &far, Some(61.0), 3),
            narrow,
            "guessed poorly (α ~0.67)"
        );
        for _ in 0..30 {
            width.note_answer(3, 3);
        }
        assert_eq!(
            width.next(false, &check, &far, Some(61.0), 3),
            wide,
            "guessed well"
        );
        let picks: Vec<u32> = (0..PROBE_EVERY)
            .map(|_| width.next(false, &check, &far, Some(61.0), 3))
            .collect();
        assert_eq!(
            picks.iter().filter(|&&w| w == narrow).count(),
            1,
            "one chunk in {PROBE_EVERY} tries the width not chosen"
        );
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
            .map(|_| width.next(true, &check, &FarPace::default(), Some(10.0), 3))
            .collect();
        assert_eq!(
            alone,
            [wide, narrow, wide, narrow],
            "alone, one width fitted"
        );
        assert_eq!(
            width.next(false, &check, &FarPace::default(), Some(10.0), 3),
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
            width.next(false, &check, &FarPace::default(), Some(10.0), 3),
            wide,
            "cheap rows, kept"
        );
        // Dear rows on the same trip: narrow.
        let mut dear = CheckCost::for_a_plan(Some(420));
        dear.record(narrow, 1000.0);
        dear.record(wide, 1800.0);
        assert_eq!(
            width.next(false, &dear, &FarPace::default(), Some(10.0), 3),
            narrow,
            "dear rows"
        );
        assert_eq!(width.sent_wide, 3);
    }
}
