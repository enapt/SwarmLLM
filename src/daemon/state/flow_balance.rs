//! How many of a flowing split's first layers this node should run, learned
//! from the flows themselves (FUTURE_WORK #504).
//!
//! A split whose guesses FLOW keeps several forwards in flight, so it runs at
//! its slowest stage's pace, not at the sum the planner prices it by. The
//! drafting node's stage carries the drafter's time beside its layers (12.5-
//! 14.6 ms of its 34-39 ms a forward on the 14B half and half to Australia,
//! 2026-11-27), and no planner input holds it. Moving three of its 26 layers to
//! the far card made prose 21% and code 8% faster (`real_link_bal313/`) — but
//! moving six made them 26% and 13% SLOWER (`real_link_bal314/`): the far
//! card's forward was flat from 22 to 25 layers and queued at 28. A far stage
//! priced from per-layer figures is a straight line and would have moved
//! about eight. So the balance is not priced: it is MEASURED, one layer at a
//! time.
//!
//! **Which way**: the far end's own time a forward (the median it reported,
//! `far_said_ms`, its queue included) against this node's (`here_ms_a_forward`:
//! drafting, layers and the rest) — under [`SHED_BELOW`] a layer moves to the
//! far end, over [`BACK_ABOVE`] one comes back. **Whether it stays**: the
//! flow's own time a forward (`wall_ms_a_forward`, what a reply's speed is made
//! of), like for like by reply kind: a move whose forwards came [`SLOWER_BY`]
//! slower than the cut it left's is undone, and a cut measured no faster is not
//! gone to again for [`TRUSTED_FOR`]. The direction alone was not enough
//! (`real_link_bal504/`, 2026-11-27): the far's reported time carries its
//! queue, so the ratio read 1.3 at 20 layers here while the flow ran as fast
//! there as at 21, and a noisy step to 22 ran 5% slower with nothing to undo
//! it. Kinds are kept apart because a code reply's forward runs ~6% quicker
//! than prose's at the same cut. Each step waits for [`FLOWS_A_STEP`] flows at
//! one cut, judged by medians: a three-flow median of the wall time a forward
//! spread 4-5% at one cut.
//!
//! Prior art: PipeDream's partitioner minimises the slowest stage from
//! profiled times (arXiv 1806.03377); PipeLive re-partitions a live pipeline
//! as its bottleneck drifts (2604.12171); Petals moves a server's blocks only
//! where the slowest stage gains enough (`MOVE_AT_MOST`, a threshold
//! `auto_manage::coverage::near_chain` copies). TCP BBR (Cardwell et al.,
//! 2016) decides on the delivery rate it MEASURES after a probe rather than on
//! a loss signal — as this keeps or undoes a step on the forward time it
//! measured, not on the far's queue (a first version stepped on the guard's
//! far-window cuts and a 30 min floor, and held a slower cut). None of them
//! steps from a running pipeline's own stage times: what a plan cannot know
//! (the drafter's time, the far card's cliff) is read off the run.
//!
//! **How far a shed goes** (2026-11-28): one layer a step took 28 flows to
//! go from 16 layers here to the 9 the 14B's flow to Australia balanced at
//! (`real_link_t510w/`: code 46.8 tok/s at 12 here, 53.1 at 9; this node
//! 34-36 ms a forward at 12, 30-33 at 9; the far 29-56) — and the balance is
//! memory only, so a node that is restarted or idle an hour started over each
//! time. Where the far end is ONE segment, a shed now moves half the gap's
//! width in layers at once, at most [`MOST_LAYERS_A_SHED`]: the gap is this
//! node's time a forward less the far end's, and a layer's width the sum of
//! the two stages' times a layer (this node's LAYERS' time over its layers,
//! the drafter's left out; the far end's whole time over its layers, its
//! fixed costs in, which overstates it and shortens the move). Halved
//! because the far card is not a straight line (`real_link_bal314/`: six
//! layers at once ran 26% slower); a cut measured slower lately is passed
//! over for a nearer one; the judgement above still undoes a move that ran
//! slower. A layer comes BACK one at a time: the far end's time carries its
//! queue, which overstates the gap that way. A/B
//! `SWARMLLM_FLOW_BALANCE_STEP=one`. Prior art: a proportional controller
//! with its gain halved for a plant it cannot see whole (the far card's
//! cliff); Petals re-balances a whole span at once from measured
//! throughputs, only where its balance quality gains (`should_choose_other_blocks`).
//!
//! **One far node, one drafter**: the balance is the pair's. A flow to another
//! far node, or drafted by another model, starts the model's balance over.
//! Memory only, forgotten [`FORGET_AFTER`] without a flow.

use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

use super::split_rounds::ReplyKind;
use crate::types::NodeId;

/// Flows at one cut before a step is taken.
pub(crate) const FLOWS_A_STEP: usize = 4;

/// The far end's time a forward under this share of this node's: the far end
/// waits on this node, which gives it a layer.
pub(crate) const SHED_BELOW: f64 = 0.97;

/// Over this share: the far end is the slower, and a layer comes back.
pub(crate) const BACK_ABOVE: f64 = 1.10;

/// A move whose cut's forwards ran this much slower than the cut it left's
/// (like for like, the median) is undone.
pub(crate) const SLOWER_BY: f64 = 0.03;

/// How long a cut's measured pace is trusted: a cut measured no faster is not
/// gone to again before, and is tried again after (the far's load changes).
pub(crate) const TRUSTED_FOR: Duration = Duration::from_secs(30 * 60);

/// A model's balance with no flow for this long is forgotten.
pub(crate) const FORGET_AFTER: Duration = Duration::from_secs(60 * 60);

/// A flow of fewer forwards is not counted: a short reply's forwards are
/// mostly its first, which wait for the far end to open its cache.
pub(crate) const LEAST_FORWARDS: u32 = 32;

/// The fewest layers a balance leaves this node.
const LEAST_LAYERS_HERE: u32 = 1;

/// The most layers one shed moves (half the measured gap, at least one).
pub(crate) const MOST_LAYERS_A_SHED: u32 = 4;

/// Whether a shed may move more than one layer: `SWARMLLM_FLOW_BALANCE_STEP=one`
/// keeps the one-layer steps (the A/B inside one binary).
fn sheds_by_the_gap() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        !std::env::var("SWARMLLM_FLOW_BALANCE_STEP").is_ok_and(|v| v.trim() == "one")
    })
}

/// One finished flow this node drove and the far end drew, as
/// `pipeline::dsd_flow` hands it over.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FlowBalanceSample {
    /// The layers this node ran before the far end's (its one leading
    /// segment).
    pub layers_here: u32,
    /// The reply's kind as `pipeline::reply_kind` predicted it: forwards are
    /// compared only with the same kind's.
    pub kind: ReplyKind,
    /// This node's time a forward: drafting, its layers and the rest.
    pub here_ms_a_forward: f64,
    /// Of it, this node's LAYERS' time a forward (the drafter's left out):
    /// what one layer here costs, for how far a shed goes.
    pub near_ms_a_forward: f64,
    /// The median of the far end's own times a forward, as it reported them.
    pub far_said_ms_median: f64,
    /// The flow's wall time over its forwards: what the balance is for.
    pub wall_ms_a_forward: f64,
    /// Forwards the flow sent.
    pub forwards: u32,
    /// The far end's layers where it is ONE segment; `None` for a chain,
    /// which sheds one layer at a time.
    pub layers_far: Option<u32>,
}

/// What a balance holds a flowing split's planner to (`scheduler::parallax`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FlowCap {
    /// This node's leading layers at most.
    pub layers: u32,
    /// The far node the balance was measured with — the cap applies only to a
    /// plan whose next segment is on it.
    pub far: NodeId,
}

/// What a flow's figures moved a balance to — for the log line.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FlowBalanceStep {
    /// Too few flows at this cut yet, or a flow too short to count.
    Waiting,
    /// The stages are near enough: the cut holds.
    Holds,
    /// The far end waits on this node: one layer moves to it.
    Shed,
    /// The far end is the slower: one layer comes back.
    Back,
    /// The last move's forwards ran slower than the cut it left's: back to it.
    Undone,
    /// The stages say move, to a cut measured no faster lately: the cut holds.
    Kept,
}

struct Flow {
    kind: ReplyKind,
    /// The far end's time a forward over this node's.
    ratio: f64,
    wall_ms: f64,
    here_ms: f64,
    near_ms: f64,
    far_ms: f64,
    layers_far: Option<u32>,
}

/// A cut's pace as last measured there: per kind, the median wall time a
/// forward.
struct Settled {
    at: Instant,
    pace: Vec<(ReplyKind, f64)>,
}

struct Balance {
    far: NodeId,
    drafter: String,
    /// The leading layers this node is held to.
    cap: Option<u32>,
    /// The cut the latest move left, until the cut it moved to is judged
    /// against it.
    from: Option<u32>,
    /// Each cut's pace as last measured, while [`TRUSTED_FOR`].
    measured: HashMap<u32, Settled>,
    /// Cuts a step asked for that the plans did not take — the flows after it
    /// still ran at the cut it left (no room here for the drafter beside more
    /// layers, none on the far end: `parallax::cut_for_the_flow`) — and when,
    /// while [`TRUSTED_FOR`]. Not asked for again meanwhile, so no step is
    /// logged that moved nothing (the v0.3.315 pre-release review).
    out_of_reach: HashMap<u32, Instant>,
    /// The cut the flows below ran at.
    cut: u32,
    /// The latest flows at [`Self::cut`], [`FLOWS_A_STEP`] at most.
    at_the_cut: VecDeque<Flow>,
    last_flow: Instant,
}

impl Balance {
    fn new(far: &NodeId, drafter: &str, now: Instant) -> Self {
        Self {
            far: far.clone(),
            drafter: drafter.to_string(),
            cap: None,
            from: None,
            measured: HashMap::new(),
            out_of_reach: HashMap::new(),
            cut: 0,
            at_the_cut: VecDeque::new(),
            last_flow: now,
        }
    }
}

/// Every model's balance. Written ONLY by `pipeline::dsd_flow` at a flow's
/// end; read ONLY by the scheduler's candidate builder.
#[derive(Default)]
pub struct FlowBalances {
    by_model: dashmap::DashMap<String, Balance>,
}

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    let n = values.len();
    if n % 2 == 1 {
        values[n / 2]
    } else {
        (values[n / 2 - 1] + values[n / 2]) / 2.0
    }
}

/// Per kind, the median wall time a forward of `flows`.
fn pace_of(flows: &VecDeque<Flow>) -> Vec<(ReplyKind, f64)> {
    let mut kinds: Vec<ReplyKind> = Vec::new();
    for flow in flows {
        if !kinds.contains(&flow.kind) {
            kinds.push(flow.kind);
        }
    }
    kinds
        .into_iter()
        .map(|kind| {
            let walls = flows.iter().filter(|f| f.kind == kind).map(|f| f.wall_ms);
            (kind, median(walls.collect()))
        })
        .collect()
}

/// How many layers a shed from `k` moves: half the measured gap's width in
/// layers, at least one, at most [`MOST_LAYERS_A_SHED`], never past
/// [`LEAST_LAYERS_HERE`] — one where the far end is a chain or a figure is
/// missing.
fn layers_to_shed(flows: &VecDeque<Flow>, k: u32) -> u32 {
    let room = k.saturating_sub(LEAST_LAYERS_HERE).max(1);
    let one_far = flows
        .iter()
        .map(|f| f.layers_far)
        .collect::<Option<Vec<u32>>>();
    let (true, Some(far_layers)) = (sheds_by_the_gap(), one_far) else {
        return 1;
    };
    let far_layers = median(far_layers.into_iter().map(f64::from).collect());
    let here = median(flows.iter().map(|f| f.here_ms).collect());
    let near = median(flows.iter().map(|f| f.near_ms).collect());
    let far = median(flows.iter().map(|f| f.far_ms).collect());
    let a_layer_here = near / f64::from(k);
    let a_layer_far = far / far_layers;
    let width = a_layer_here + a_layer_far;
    if !(width.is_finite() && width > 0.0 && near > 0.0 && far_layers >= 1.0) {
        return 1;
    }
    let gap_in_layers = (here - far) / width;
    let half = (gap_in_layers / 2.0).round();
    if !half.is_finite() || half < 1.0 {
        return 1;
    }
    (half as u32).clamp(1, MOST_LAYERS_A_SHED).min(room)
}

/// How much slower `flows` ran than `pace`, like for like: the median of each
/// flow's wall time a forward over its kind's there; `None` with no kind in
/// common.
fn slower_than(flows: &VecDeque<Flow>, pace: &[(ReplyKind, f64)]) -> Option<f64> {
    let relative: Vec<f64> = flows
        .iter()
        .filter_map(|flow| {
            let (_, there) = pace.iter().find(|(kind, _)| *kind == flow.kind)?;
            Some(flow.wall_ms / there)
        })
        .collect();
    (!relative.is_empty()).then(|| median(relative))
}

impl FlowBalances {
    /// A flow of `model` to `far`, drafted by `drafter`, finished with
    /// `sample`; what it moved the balance to.
    pub fn record(
        &self,
        model: &str,
        far: &NodeId,
        drafter: &str,
        sample: FlowBalanceSample,
        now: Instant,
    ) -> FlowBalanceStep {
        let timed = |ms: f64| ms.is_finite() && ms > 0.0;
        let counts = sample.forwards >= LEAST_FORWARDS
            && sample.layers_here > 0
            && timed(sample.here_ms_a_forward)
            && timed(sample.far_said_ms_median)
            && timed(sample.wall_ms_a_forward);
        if !counts {
            return FlowBalanceStep::Waiting;
        }
        let mut entry = self
            .by_model
            .entry(model.to_string())
            .or_insert_with(|| Balance::new(far, drafter, now));
        let balance: &mut Balance = &mut entry;
        let stale = now.saturating_duration_since(balance.last_flow) > FORGET_AFTER;
        if stale || &balance.far != far || balance.drafter != drafter {
            *balance = Balance::new(far, drafter, now);
        }
        balance.last_flow = now;
        balance
            .measured
            .retain(|_, settled| now.saturating_duration_since(settled.at) <= TRUSTED_FOR);
        balance
            .out_of_reach
            .retain(|_, at| now.saturating_duration_since(*at) <= TRUSTED_FOR);
        let k = sample.layers_here;
        if balance.cut != k {
            balance.cut = k;
            balance.at_the_cut.clear();
        }
        balance.at_the_cut.push_back(Flow {
            kind: sample.kind,
            ratio: sample.far_said_ms_median / sample.here_ms_a_forward,
            wall_ms: sample.wall_ms_a_forward,
            here_ms: sample.here_ms_a_forward,
            near_ms: sample.near_ms_a_forward,
            far_ms: sample.far_said_ms_median,
            layers_far: sample.layers_far,
        });
        while balance.at_the_cut.len() > FLOWS_A_STEP {
            balance.at_the_cut.pop_front();
        }
        if balance.at_the_cut.len() < FLOWS_A_STEP {
            return FlowBalanceStep::Waiting;
        }
        let settled = Settled {
            at: now,
            pace: pace_of(&balance.at_the_cut),
        };
        balance.measured.insert(k, settled);
        // A move is judged once, against the cut it left.
        match balance.from.take() {
            Some(left) if left != k => {
                let slower = balance
                    .measured
                    .get(&left)
                    .and_then(|there| slower_than(&balance.at_the_cut, &there.pace));
                if slower.is_some_and(|s| s > 1.0 + SLOWER_BY) {
                    balance.cap = Some(left);
                    balance.at_the_cut.clear();
                    return FlowBalanceStep::Undone;
                }
            }
            // The flows after the step still ran where it left: the plans
            // could not take the cut it asked for (still the cap).
            Some(_) => {
                if let Some(asked) = balance.cap {
                    balance.out_of_reach.insert(asked, now);
                }
            }
            None => {}
        }
        balance.cap = Some(k);
        let ratio = median(balance.at_the_cut.iter().map(|f| f.ratio).collect());
        // The cuts a step may go to, the farthest first: a shed's by the
        // measured gap, a layer back one at a time.
        let candidates: Vec<u32> = if ratio > BACK_ABOVE {
            vec![k + 1]
        } else if ratio < SHED_BELOW && k > LEAST_LAYERS_HERE {
            (1..=layers_to_shed(&balance.at_the_cut, k))
                .rev()
                .map(|d| k - d)
                .collect()
        } else {
            return FlowBalanceStep::Holds;
        };
        let mut passed_over_a_slower = false;
        let mut chosen = None;
        for want in candidates {
            if balance.out_of_reach.contains_key(&want) {
                continue;
            }
            // A cut measured lately is gone to only where it ran faster.
            if let Some(there) = balance.measured.get(&want) {
                let faster_there = slower_than(&balance.at_the_cut, &there.pace)
                    .is_some_and(|s| s > 1.0 + SLOWER_BY);
                if !faster_there {
                    passed_over_a_slower = true;
                    continue;
                }
            }
            chosen = Some(want);
            break;
        }
        let Some(want) = chosen else {
            return if passed_over_a_slower {
                FlowBalanceStep::Kept
            } else {
                FlowBalanceStep::Holds
            };
        };
        balance.cap = Some(want);
        balance.from = Some(k);
        balance.at_the_cut.clear();
        if want < k {
            FlowBalanceStep::Shed
        } else {
            FlowBalanceStep::Back
        }
    }

    /// The leading layers a flowing split of `model` is held to here, and the
    /// far node that holds for; `None` before any step, or once forgotten.
    pub fn cap(&self, model: &str, now: Instant) -> Option<FlowCap> {
        let balance = self.by_model.get(model)?;
        if now.saturating_duration_since(balance.last_flow) > FORGET_AFTER {
            return None;
        }
        balance.cap.map(|layers| FlowCap {
            layers,
            far: balance.far.clone(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ReplyKind::{Code, Prose};

    const M: &str = "qwen2.5-14b-instruct-q4-k-m";
    const D: &str = "qwen2.5-0.5b-instruct-q8-0";

    fn far() -> NodeId {
        NodeId([7; 32])
    }

    /// A flow at `k` layers here: its wall time a forward, this node's time a
    /// forward and the far end's.
    fn flow(k: u32, kind: ReplyKind, wall: f64, here: f64, far_said: f64) -> FlowBalanceSample {
        FlowBalanceSample {
            layers_here: k,
            kind,
            here_ms_a_forward: here,
            near_ms_a_forward: 0.0,
            far_said_ms_median: far_said,
            wall_ms_a_forward: wall,
            forwards: 200,
            layers_far: None,
        }
    }

    /// A flow whose far end is one segment of `of - k` layers, this node's
    /// layers taking `near` of its `here` a forward.
    fn flow_to_one(
        k: u32,
        of: u32,
        wall: f64,
        here: f64,
        near: f64,
        far_said: f64,
    ) -> FlowBalanceSample {
        FlowBalanceSample {
            near_ms_a_forward: near,
            layers_far: Some(of - k),
            ..flow(k, Prose, wall, here, far_said)
        }
    }

    /// `real_link_t517ab/` and `real_link_t510w/` (2026-11-28, the 14B to
    /// Australia's 3060, 48 layers): at 16 here this node took ~40 ms a
    /// forward (its layers ~16, the drafter the rest) and the far end ~27 —
    /// the gap is ~7 layers wide, and a shed moves half of it at once; one
    /// step at a time took 28 flows to reach 9.
    #[test]
    fn a_shed_moves_half_the_measured_gap_at_once() {
        let b = FlowBalances::default();
        let t = Instant::now();
        let mut last = FlowBalanceStep::Waiting;
        for _ in 0..FLOWS_A_STEP {
            last = b.record(M, &far(), D, flow_to_one(16, 48, 41.0, 40.0, 16.0, 27.0), t);
        }
        assert_eq!(last, FlowBalanceStep::Shed);
        // a layer here 1.0 ms, there 27/32: 13 ms is ~7 layers; half is 4.
        assert_eq!(layers(&b, t), Some(12));
        // At 12 the gap is narrower: 37.6 against 30 is ~4 layers, 2 move.
        for _ in 0..FLOWS_A_STEP {
            last = b.record(M, &far(), D, flow_to_one(12, 48, 39.0, 37.6, 12.7, 30.0), t);
        }
        assert_eq!(last, FlowBalanceStep::Shed);
        assert_eq!(layers(&b, t), Some(10));
        // A chain, or a figure missing, sheds one.
        let chain = FlowBalances::default();
        for _ in 0..FLOWS_A_STEP {
            chain.record(M, &far(), D, flow(16, Prose, 41.0, 40.0, 27.0), t);
        }
        assert_eq!(layers(&chain, t), Some(15));
    }

    /// A shed never goes to a cut measured slower lately: it passes it over
    /// for a nearer one.
    #[test]
    fn a_shed_passes_over_a_cut_measured_slower() {
        let b = FlowBalances::default();
        let t = Instant::now();
        let at = |k: u32, wall: f64, here: f64, near: f64, far_said: f64| {
            let mut last = FlowBalanceStep::Waiting;
            for _ in 0..FLOWS_A_STEP {
                last = b.record(
                    M,
                    &far(),
                    D,
                    flow_to_one(k, 48, wall, here, near, far_said),
                    t,
                );
            }
            last
        };
        assert_eq!(at(16, 41.0, 40.0, 16.0, 27.0), FlowBalanceStep::Shed);
        assert_eq!(layers(&b, t), Some(12));
        // 12 ran 10% slower than 16: undone, back to 16.
        assert_eq!(at(12, 45.1, 37.6, 12.7, 30.0), FlowBalanceStep::Undone);
        assert_eq!(layers(&b, t), Some(16));
        // The stages still say shed: 12 is passed over for 13.
        assert_eq!(at(16, 41.0, 40.0, 16.0, 27.0), FlowBalanceStep::Shed);
        assert_eq!(layers(&b, t), Some(13));
    }

    fn layers(balances: &FlowBalances, at: Instant) -> Option<u32> {
        balances.cap(M, at).map(|cap| cap.layers)
    }

    /// `real_link_bal504/`'s figures (2026-11-27): at 20 layers here the far
    /// end reported the longer forward, and 21 ran as fast; 22 ran 5% slower
    /// — a step the first version took on a noisy signal and held.
    #[test]
    fn a_step_that_measured_slower_is_undone_and_not_taken_again_while_trusted() {
        let b = FlowBalances::default();
        let t = Instant::now();
        let step = |s: FlowBalanceSample, at| b.record(M, &far(), D, s, at);
        // 20, as arm 1: the far end's forward 1.3-2.3x this node's.
        assert_eq!(
            step(flow(20, Prose, 31.9, 30.5, 42.0), t),
            FlowBalanceStep::Waiting
        );
        step(flow(20, Prose, 31.2, 29.5, 38.0), t);
        step(flow(20, Code, 30.2, 28.0, 64.0), t);
        assert_eq!(layers(&b, t), None, "no step before four flows at a cut");
        assert_eq!(
            step(flow(20, Prose, 33.2, 29.5, 40.0), t),
            FlowBalanceStep::Back
        );
        assert_eq!(layers(&b, t), Some(21));
        // 21, as arm 2: as fast like for like, the stages near: it holds.
        step(flow(21, Prose, 32.0, 31.0, 31.0), t);
        step(flow(21, Prose, 32.4, 30.5, 31.0), t);
        step(flow(21, Code, 30.6, 29.5, 39.0), t);
        assert_eq!(
            step(flow(21, Prose, 32.0, 31.1, 32.0), t),
            FlowBalanceStep::Holds
        );
        assert_eq!(layers(&b, t), Some(21));
        // The far end's reports run long (arm 2's last replies): a step to 22.
        step(flow(21, Prose, 32.1, 31.0, 35.7), t);
        assert_eq!(
            step(flow(21, Prose, 33.0, 31.3, 36.0), t),
            FlowBalanceStep::Back
        );
        assert_eq!(layers(&b, t), Some(22));
        // 22 runs 5% slower like for like (arm 3: 37.3 against 35.5): undone.
        step(flow(22, Prose, 33.9, 34.5, 30.0), t);
        step(flow(22, Prose, 34.1, 34.8, 30.0), t);
        step(flow(22, Code, 32.2, 33.6, 31.0), t);
        assert_eq!(
            step(flow(22, Prose, 33.8, 34.3, 29.0), t),
            FlowBalanceStep::Undone
        );
        assert_eq!(layers(&b, t), Some(21));
        // Back at 21 the far's reports still run long, but 22 measured slower.
        for _ in 0..FLOWS_A_STEP - 1 {
            step(flow(21, Prose, 32.2, 31.0, 36.0), t);
        }
        assert_eq!(
            step(flow(21, Prose, 32.2, 31.0, 36.0), t),
            FlowBalanceStep::Kept
        );
        assert_eq!(layers(&b, t), Some(21));
        // Once that measurement ages, 22 is tried again (the far's load changes).
        let later = t + TRUSTED_FOR + Duration::from_secs(1);
        assert_eq!(
            step(flow(21, Prose, 32.2, 31.0, 36.0), later),
            FlowBalanceStep::Back
        );
        assert_eq!(layers(&b, later), Some(22));
    }

    /// A step the plans cannot take — no room here for the drafter beside one
    /// more layer — leaves the flows at the cut it left. It is not asked for
    /// again while trusted, so no line says a layer moved when none did (the
    /// v0.3.315 pre-release review: it was asked again every four flows).
    #[test]
    fn a_step_the_plans_cannot_take_is_not_asked_for_again_while_trusted() {
        let b = FlowBalances::default();
        let t = Instant::now();
        let at = |k, when| {
            let mut last = FlowBalanceStep::Waiting;
            for _ in 0..FLOWS_A_STEP {
                last = b.record(M, &far(), D, flow(k, Prose, 32.0, 30.0, 40.0), when);
            }
            last
        };
        // The far end the slower: a layer back here is asked for.
        assert_eq!(at(20, t), FlowBalanceStep::Back);
        assert_eq!(layers(&b, t), Some(21));
        // The plans keep 20: it holds there and asks for nothing.
        assert_eq!(at(20, t), FlowBalanceStep::Holds);
        assert_eq!(layers(&b, t), Some(20));
        assert_eq!(at(20, t), FlowBalanceStep::Holds);
        // Once that ages it is asked for again — at the next flow, the four
        // at 20 being measured already: the room may have changed.
        let later = t + TRUSTED_FOR + Duration::from_secs(1);
        assert_eq!(
            b.record(M, &far(), D, flow(20, Prose, 32.0, 30.0, 40.0), later),
            FlowBalanceStep::Back
        );
        assert_eq!(layers(&b, later), Some(21));
    }

    /// `real_link_bal313/` (2026-11-27): at 26 layers here this node is the
    /// slower stage, and each layer moved off it ran faster.
    #[test]
    fn a_flow_sheds_layers_while_each_step_runs_faster() {
        let b = FlowBalances::default();
        let t = Instant::now();
        let at = |k, wall, here, far_said| {
            let mut last = FlowBalanceStep::Waiting;
            for _ in 0..FLOWS_A_STEP {
                last = b.record(M, &far(), D, flow(k, Prose, wall, here, far_said), t);
            }
            last
        };
        assert_eq!(at(26, 38.5, 37.0, 31.0), FlowBalanceStep::Shed);
        assert_eq!(at(25, 36.6, 35.0, 31.0), FlowBalanceStep::Shed);
        assert_eq!(at(24, 35.0, 33.5, 31.5), FlowBalanceStep::Shed);
        assert_eq!(layers(&b, t), Some(23));
        // At 23 the stages meet: it holds.
        assert_eq!(at(23, 33.3, 32.0, 32.0), FlowBalanceStep::Holds);
        // Near enough from the start, a cut holds.
        let near = FlowBalances::default();
        for _ in 0..FLOWS_A_STEP {
            near.record(M, &far(), D, flow(24, Prose, 30.0, 30.0, 30.0), t);
        }
        assert_eq!(layers(&near, t), Some(24));
    }

    /// A code reply's forward runs ~6% quicker than prose's at one cut
    /// (`real_link_bal504/`): a step is never judged across kinds.
    #[test]
    fn forwards_are_compared_only_with_the_same_kinds() {
        let b = FlowBalances::default();
        let t = Instant::now();
        let step = |s: FlowBalanceSample| b.record(M, &far(), D, s, t);
        for _ in 0..FLOWS_A_STEP {
            step(flow(20, Code, 30.6, 29.0, 40.0));
        }
        assert_eq!(layers(&b, t), Some(21));
        // Prose at 21 runs slower than code at 20, which says nothing: kept.
        for _ in 0..FLOWS_A_STEP - 1 {
            step(flow(21, Prose, 32.5, 31.0, 31.0));
        }
        assert_eq!(
            step(flow(21, Prose, 32.5, 31.0, 31.0)),
            FlowBalanceStep::Holds
        );
        assert_eq!(layers(&b, t), Some(21));
        // Nor does a cut known only by another kind's forwards count as
        // measured faster: the stages say shed, but 20 is not gone back to.
        for _ in 0..FLOWS_A_STEP - 1 {
            step(flow(21, Prose, 32.5, 33.0, 30.0));
        }
        assert_eq!(
            step(flow(21, Prose, 32.5, 33.0, 30.0)),
            FlowBalanceStep::Kept
        );
    }

    #[test]
    fn a_balance_is_the_far_node_and_drafter_it_was_measured_with() {
        let b = FlowBalances::default();
        let t = Instant::now();
        let shed = flow(26, Prose, 38.0, 38.0, 31.0);
        for _ in 0..FLOWS_A_STEP {
            b.record(M, &far(), "a", shed, t);
        }
        assert_eq!(layers(&b, t), Some(25));
        // Another drafter's time is not this one's.
        b.record(M, &far(), "b", shed, t);
        assert_eq!(b.cap(M, t), None);
        for _ in 0..FLOWS_A_STEP {
            b.record(M, &far(), "b", shed, t);
        }
        assert!(b.cap(M, t).is_some());
        // Nor another far node's.
        b.record(M, &NodeId([8; 32]), "b", shed, t);
        assert_eq!(b.cap(M, t), None);
        // A flow too short to say what a forward costs is not counted.
        let short = FlowBalances::default();
        for _ in 0..FLOWS_A_STEP + 1 {
            let mut s = shed;
            s.forwards = LEAST_FORWARDS - 1;
            short.record(M, &far(), "a", s, t);
        }
        assert_eq!(short.cap(M, t), None);
        // Nor is a balance kept past an hour without a flow.
        for _ in 0..FLOWS_A_STEP {
            b.record(M, &far(), "b", shed, t);
        }
        assert_eq!(b.cap(M, t + FORGET_AFTER + Duration::from_secs(1)), None);
    }
}
