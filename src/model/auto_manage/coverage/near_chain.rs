//! Nearby machines hold complementary parts of a model none of them can run
//! whole — Stage 3 of `docs/plans/regional_pipelines.md` (2026-10-21).
//!
//! A split is only fast when its machines are close (6.76 tok/s at 18 ms
//! against 0.35 between Thailand and Italy), and nothing placed parts by
//! distance: replication counts copies (`geo_target_replicas`), carrying (#231,
//! the rest of this module) asks only whether SOME holders could run the model,
//! wherever they are. A large model's near chain was complete only by luck.
//!
//! **One plan per neighbourhood, computed alike by every machine in it.** For a
//! model somebody asked for, that no near machine can run whole and the near
//! machines' ceilings could carry between them, each near machine computes the
//! same chain over the near machines — each used once, for one contiguous run
//! of at most its ceiling — with the FEWEST SEGMENTS (within one of the least
//! the ceilings allow), then the fewest parts to fetch, ties broken by
//! rendezvous order, and fetches the parts of ITS segment it lacks. Prune keeps
//! them: [`AutoShardManager::copy_in_near_chain`] is asked by `copy_kept_because`,
//! which the download pass asks before every fetch (gotcha #795).
//!
//! - **In-flight downloads count as held**, so the chain a fetch starts under is
//!   the chain it lands under: the chosen chain's cost falls by everything this
//!   machine fetches, any other chain's by no more.
//! - **A chain already complete is never rebalanced**: while one exists over
//!   what is held, only such a chain is chosen, so nothing is fetched.
//! - **Near** is this node and each connected machine in scope whose measured
//!   windowed-minimum round trip (`metrics.network_coord_samples`) is at most
//!   [`NEAR_RTT_MS`] — the measurement, never the coordinate, which orders near
//!   peers worst (regional_pipelines.md § "The soak reading").
//! - **A blip is not a gap**: a near machine keeps its place, and its parts
//!   count, for [`NEAR_GRACE`] after it was last measured near.
//! - **A named part is kept a day** ([`NAMED_PART_HOLD`]): a neighbour that
//!   flaps must not turn this node's segment into a shed-and-refetch loop —
//!   #876's rule, a moving input held at its window's maximum.
//! - **A named machine keeps its place only by progress**, as a carrier does:
//!   one whose own budget refuses its segment is passed over.
//!
//! Chosen by simulation (`examples/placement_sim.py`): this machine + the LAN
//! box completed in 25 min (one-at-a-time carrying never did — it needs two
//! machines to move at once); 60 machines in six cities in 30 min with 5.0
//! segments and 109 parts against 118 min, 8.5 and 122; a neighbour off 15
//! minutes in every three hours caused no re-fetch. What it cannot do —
//! neighbourhoods that only overlap, which see different near sets — is phase 2
//! (published near lists), in the plan.
//!
//! **Where the near machines cannot carry the model, the same chain is planned
//! over every machine in scope** ([`Reach::Far`], #346, 2026-10-25). A Llama
//! 3.3 70B no machine could run whole had four holders 300-410 ms apart, none
//! near another but this one's LAN box: no near chain, and carrying — which
//! summed the holders' figures, counting a layer once per holder — read 80 of
//! its 80 layers carried while 67 could be placed, so nothing fetched and a
//! holder shed the middle it alone had room for. A far chain is slow, but the
//! only way such a model runs at all; the near chain stays first. It is Petals'
//! placement (Borzunov et al. 2023, arXiv 2312.08361 § 3.2;
//! `petals/server/block_selection.py`): each server one contiguous span sized
//! to its memory, over the blocks covered worst. Every machine sees the same
//! machines only in a swarm where all are connected; where they are not, two
//! plans may name different segments, each fetch bounded by one machine's
//! ceiling, and both settle on what is then held. While a chain is planned,
//! carrying stands down for the model ([`AutoShardManager::chain_planned_for`]):
//! one placement per model.

use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};

use dashmap::DashMap;

use crate::inference::process_pool::{ModelWeights, RangeWeights};
use crate::types::{ModelId, ModelManifest, NodeId, ShardId, MMPROJ_SHARD_INDEX};

use super::super::AutoShardManager;
use super::{rendezvous, CARRIER_PATIENCE, PASSED_OVER_FOR};

/// A machine is near when its measured round trip is at most this. At 18 ms
/// the network was 17% of a token (regional_pipelines.md § "The premise is
/// validated"); a chain of three near machines led by its head pays three legs
/// a pass.
const NEAR_RTT_MS: f32 = 30.0;

/// How long a near machine keeps its place after it was last measured near —
/// gone, or a round trip that blipped. In the simulator a neighbour dropping
/// off 15 minutes in every three hours made one-at-a-time carrying fetch, shed
/// and fetch back 15 parts in two days; counting its parts for 30 minutes
/// ended it.
const NEAR_GRACE: Duration = Duration::from_secs(30 * 60);

/// How long prune keeps a part after this node was last named for it.
const NAMED_PART_HOLD: Duration = Duration::from_secs(24 * 60 * 60);

/// The most machines a chain is planned over: the plan is exact over
/// subsets of them. Those holding most of the model come first.
const MOST_MACHINES: usize = 10;

/// How long a computed chain answers. Prune asks about every part it holds,
/// and the inputs move on gossip's minutes, not per call.
const CHAIN_ANSWERS_FOR: Duration = Duration::from_secs(60);

/// Segments past the fewest the ceilings allow that a plan may use.
const EXTRA_SEGMENTS: usize = 1;

/// The most headers one pass learns for near chains.
const MOST_HEADERS_A_PASS: usize = 2;

/// After this node gives back memory holding layers outside its segment, how
/// long that model is left to the ordinary idle rule. A router that still
/// prefers the old route (a far card faster than the near processors) loads
/// the range again, and is not fought every conversation — the hysteresis
/// Petals' servers keep before moving to other blocks
/// (`block_selection.py::should_choose_other_blocks`).
const OFF_SEGMENT_RELEASE_HOLD: Duration = NEAR_GRACE;

/// What the near chain remembers between passes. Each entry is a DEADLINE,
/// so nothing needs a clock set in the past.
#[derive(Default)]
pub(in crate::model::auto_manage) struct NearChainBook {
    /// Until when each machine keeps its place among the near ones.
    near_until: DashMap<NodeId, Instant>,
    /// Each model's chain, and until when it answers.
    chains: DashMap<ModelId, (Instant, Option<NearChain>)>,
    /// Until when prune keeps each part this node was named for.
    named_until: DashMap<ShardId, Instant>,
    /// A named machine's lease: when it lapses unless renewed, and the layers
    /// it held at the last renewal.
    leases: DashMap<(ModelId, NodeId), (Instant, u32)>,
    /// Until when a machine whose lease lapsed is left out of a model's chain.
    passed_over_until: DashMap<(ModelId, NodeId), Instant>,
    /// Until when a model whose layers outside this node's segment were given
    /// back is left to the ordinary idle rule ([`OFF_SEGMENT_RELEASE_HOLD`]).
    off_segment_released_until: DashMap<ModelId, Instant>,
}

/// The chain a neighbourhood plans for a model.
#[derive(Debug, Clone, PartialEq, Eq)]
struct NearChain {
    /// Each machine's layers, `[start, end)`, in chain order.
    segments: Vec<(NodeId, (u32, u32))>,
    /// Over what its machines hold already: nothing is fetched.
    complete: bool,
    /// Which machines it was planned over.
    reach: Reach,
}

/// Which machines a chain is planned over.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::model::auto_manage) enum Reach {
    /// This node and the machines measured near it.
    Near,
    /// Every machine in scope — the near ones could not carry the model (#346).
    Far,
}

/// Why a part is this node's to keep for a chain
/// ([`AutoShardManager::copy_in_near_chain`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::model::auto_manage) enum ChainPart {
    /// It is in this node's segment of the chain planned now, at this reach.
    PlannedNow(Reach),
    /// A chain named it within [`NAMED_PART_HOLD`] and none does now.
    NamedLately,
}

/// What planning over one [`Reach`] came to.
enum Planned {
    Chain(NearChain),
    /// A machine among them could run the model whole: carrying's to fill.
    OneRunsItWhole,
    /// Their ceilings could not carry it between them, or no chain over them
    /// fits their disks.
    CannotCarry,
}

impl NearChain {
    /// `node:start-end` for each segment, in chain order — for the log.
    fn describe(&self) -> String {
        self.segments
            .iter()
            .map(|(node, (start, end))| format!("{node}:{start}-{end}"))
            .collect::<Vec<_>>()
            .join(" ")
    }

    fn segment_of(&self, node: &NodeId) -> Option<(u32, u32)> {
        self.segments
            .iter()
            .find(|(n, _)| n == node)
            .map(|(_, range)| *range)
    }
}

/// A part of the model, in layer order.
#[derive(Debug, Clone, Copy)]
struct Part {
    index: u32,
    start: u32,
    end: u32,
    size: u64,
}

/// A machine as the plan sees it.
#[derive(Debug, Clone)]
struct Machine {
    node: NodeId,
    /// Layers it could ever hold (`coverage::ceiling_layers`).
    ceiling: u32,
    /// By part position: held, or being fetched.
    has: Vec<bool>,
    /// Layers it holds, fetches not counted — what its lease renews on.
    held_layers: u32,
    /// Layers it holds or is fetching — what ranks it for [`MOST_MACHINES`].
    has_layers: u32,
    /// Fetching a part of the model now.
    fetching: bool,
    /// What its advertised free disk would take.
    disk_bytes: u64,
}

/// A segment: the machine (its position in the plan's list) and its layers.
type Step = (usize, u32, u32);

/// The best way on from a point: (segments, parts fetched, the first step).
type Best = (u32, u32, Step);

/// The model's parts in layer order — when they tile `[0, num_layers)` with
/// whole layers, as every manifest `shard_layout` builds does. Anything else is
/// not planned for.
fn parts_in_order(manifest: &ModelManifest) -> Option<Vec<Part>> {
    let mut parts: Vec<Part> = manifest
        .shards
        .iter()
        .filter(|s| s.index != MMPROJ_SHARD_INDEX)
        .map(|s| Part {
            index: s.index,
            start: s.layer_range.0,
            end: s.layer_range.1,
            size: s.size_bytes,
        })
        .collect();
    parts.sort_by_key(|p| (p.start, p.end));
    let mut at = 0;
    for part in &parts {
        if part.start != at || part.end <= part.start {
            return None;
        }
        at = part.end;
    }
    (!parts.is_empty() && at == manifest.num_layers).then_some(parts)
}

/// The parts a machine running `segment` needs: those its layers touch — the
/// first part carries the embeddings, the last the output head.
fn parts_touching(manifest: &ModelManifest, segment: (u32, u32)) -> Vec<u32> {
    manifest
        .shards
        .iter()
        .filter(|s| {
            s.index != MMPROJ_SHARD_INDEX
                && s.layer_range.0 < segment.1
                && s.layer_range.1 > segment.0
        })
        .map(|s| s.index)
        .collect()
}

/// The chain over `machines` (in rendezvous order) covering `[0, layers)`,
/// each machine used once for one run of at most its ceiling: the fewest
/// segments, then the fewest parts to fetch; of equals, the first in machine
/// order, each machine's furthest end first. `may_fetch: false` plans only
/// over what the machines have. `None`: no such chain.
///
/// `weights` weighs each run as what it loads against the average layer the
/// ceilings count in (`process_pool::RangeWeights`): a run ending the model
/// carries its output head, one starting it its token table — 822 MB and
/// 564 MB of a 70B, a layer and a half and half of one at its average
/// (field report #039). `None` (no header here) counts every run as its
/// width, as before.
fn plan_chain(
    layers: u32,
    parts: &[Part],
    machines: &[Machine],
    may_fetch: bool,
    weights: Option<&RangeWeights>,
) -> Option<Vec<Step>> {
    if machines.is_empty() || machines.len() > 32 || parts.is_empty() {
        return None;
    }
    let max_segments = if may_fetch {
        let mut ceilings: Vec<u32> = machines.iter().map(|m| m.ceiling).collect();
        ceilings.sort_unstable_by(|a, b| b.cmp(a));
        let mut total = 0u32;
        let fewest = ceilings.iter().position(|&c| {
            total = total.saturating_add(c);
            total >= layers
        })? + 1;
        (fewest + EXTRA_SEGMENTS).min(machines.len())
    } else {
        machines.len()
    };
    let lacking = machines
        .iter()
        .map(|m| {
            let mut count = vec![0u32; parts.len() + 1];
            let mut bytes = vec![0u64; parts.len() + 1];
            for (p, part) in parts.iter().enumerate() {
                let lacks = !m.has[p];
                count[p + 1] = count[p] + u32::from(lacks);
                bytes[p + 1] = bytes[p] + if lacks { part.size } else { 0 };
            }
            (count, bytes)
        })
        .collect();
    let mut planner = Planner {
        layers,
        parts,
        machines,
        lacking,
        max_segments: max_segments as u32,
        may_fetch,
        weights,
        memo: HashMap::new(),
    };
    planner.chain()
}

/// The search behind [`plan_chain`]: exact over (position, machines used).
struct Planner<'a> {
    layers: u32,
    parts: &'a [Part],
    machines: &'a [Machine],
    /// Per machine, prefix sums over part positions of the parts it lacks and
    /// their bytes.
    lacking: Vec<(Vec<u32>, Vec<u64>)>,
    max_segments: u32,
    may_fetch: bool,
    /// What a run really loads against the ceilings' average layer.
    weights: Option<&'a RangeWeights>,
    /// (position, machines used) → the best cover of the rest: segments, parts
    /// fetched, and its first step.
    memo: HashMap<(u32, u32), Option<Best>>,
}

impl Planner<'_> {
    /// The part holding `layer`.
    fn part_at(&self, layer: u32) -> usize {
        self.parts.partition_point(|p| p.end <= layer)
    }

    /// Where machine `i` may end a segment it starts at `x`, furthest first.
    /// Its cost changes only where the segment reaches a new part, and a
    /// segment ending further is never worse for the rest of the chain, so
    /// these are the only ends worth trying: its ceiling's reach, and just
    /// short of each part in between. Without fetching: the end of the run it
    /// has, within its reach.
    fn ends(&self, i: usize, x: u32) -> Vec<u32> {
        let machine = &self.machines[i];
        let reach = self.reach(i, x);
        if reach <= x {
            return Vec::new();
        }
        let first = self.part_at(x);
        if !self.may_fetch {
            if !machine.has[first] {
                return Vec::new();
            }
            let mut last = first;
            while self.parts[last].end < reach
                && last + 1 < self.parts.len()
                && machine.has[last + 1]
            {
                last += 1;
            }
            return vec![reach.min(self.parts[last].end)];
        }
        let mut ends = vec![reach];
        let starts: Vec<u32> = self.parts[first + 1..]
            .iter()
            .map(|p| p.start)
            .take_while(|&start| start < reach)
            .collect();
        ends.extend(starts.into_iter().rev());
        ends
    }

    /// The furthest a run from `x` on machine `i` may end: its ceiling's worth
    /// of layers, each run weighed as what it loads where the header is here
    /// (`RangeWeights::width_in_bound`) — a run ending the model is shorter by
    /// its output head. `x` when not one layer fits.
    fn reach(&self, i: usize, x: u32) -> u32 {
        let ceiling = self.machines[i].ceiling;
        let Some(weights) = self.weights else {
            return self.layers.min(x.saturating_add(ceiling));
        };
        // A middle run weighs less than its average layers, so it may reach a
        // little past the ceiling's count — never twice as far.
        let furthest = self.layers.min(x.saturating_add(ceiling.saturating_mul(2)));
        (x + 1..=furthest)
            .rev()
            .find(|&e| weights.width_in_bound((x, e)) <= ceiling)
            .unwrap_or(x)
    }

    /// Parts machine `i` lacks among those `[x, e)` touches, and their bytes.
    fn lacks(&self, i: usize, x: u32, e: u32) -> (u32, u64) {
        let (lo, hi) = (self.part_at(x), self.part_at(e - 1) + 1);
        let (count, bytes) = &self.lacking[i];
        (count[hi] - count[lo], bytes[hi] - bytes[lo])
    }

    /// The best cover of `[x, layers)` by machines not in `used`: (segments,
    /// parts fetched).
    fn best(&mut self, x: u32, used: u32) -> Option<(u32, u32)> {
        if x >= self.layers {
            return Some((0, 0));
        }
        if used.count_ones() >= self.max_segments {
            return None;
        }
        if let Some(known) = self.memo.get(&(x, used)) {
            return known.map(|(segments, fetched, _)| (segments, fetched));
        }
        let mut best: Option<Best> = None;
        for i in 0..self.machines.len() {
            if used & (1 << i) != 0 {
                continue;
            }
            for e in self.ends(i, x) {
                let (fetch, bytes) = self.lacks(i, x, e);
                if bytes > 0 && bytes >= self.machines[i].disk_bytes {
                    continue;
                }
                let Some((segments, fetched)) = self.best(e, used | (1 << i)) else {
                    continue;
                };
                let candidate = (segments + 1, fetched + fetch);
                if best.is_none_or(|(s, f, _)| candidate < (s, f)) {
                    best = Some((candidate.0, candidate.1, (i, x, e)));
                }
            }
        }
        self.memo.insert((x, used), best);
        best.map(|(segments, fetched, _)| (segments, fetched))
    }

    /// The steps of the best cover of the whole model.
    fn chain(&mut self) -> Option<Vec<Step>> {
        self.best(0, 0)?;
        let (mut x, mut used, mut steps) = (0u32, 0u32, Vec::new());
        while x < self.layers {
            let (_, _, (i, _, e)) = self.memo.get(&(x, used)).copied().flatten()?;
            steps.push((i, x, e));
            x = e;
            used |= 1 << i;
        }
        Some(steps)
    }
}

/// Does `machine` lack any part `[x, e)` touches?
/// `SWARMLLM_NEAR_CHAIN=0` plans no near chain, so nothing is fetched or kept
/// for one: the control arm of `examples/near_chain_rig.sh`, in the same
/// binary as the change it measures.
fn near_chains_switched_off() -> bool {
    static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *OFF.get_or_init(|| {
        std::env::var("SWARMLLM_NEAR_CHAIN")
            .map(|v| v == "0" || v.eq_ignore_ascii_case("false"))
            .unwrap_or(false)
    })
}

fn lacks_any(parts: &[Part], machine: &Machine, x: u32, e: u32) -> bool {
    parts
        .iter()
        .zip(&machine.has)
        .any(|(p, &has)| !has && p.start < e && p.end > x)
}

impl AutoShardManager {
    /// The machines near this node, itself left out, in scope — after renewing
    /// the place of each connected one measured near now.
    fn near_peers(&self) -> Vec<NodeId> {
        let state = &self.shared_state;
        let local = state.identity.node_id();
        let allowed = crate::pool::scope::allowed_node_set(state);
        let in_scope = |n: &NodeId| allowed.as_ref().is_none_or(|a| a.contains(n));
        let now = Instant::now();
        let measured_near: Vec<NodeId> = state
            .connected_node_ids
            .iter()
            .map(|n| n.clone())
            .filter(|n| {
                n != local
                    && in_scope(n)
                    && state
                        .metrics
                        .network_coord_samples
                        .get(n)
                        .and_then(|samples| samples.min())
                        .is_some_and(|ms| ms <= NEAR_RTT_MS)
            })
            .collect();
        let book = &self.near_chain_book;
        for node in measured_near {
            book.near_until.insert(node, now + NEAR_GRACE);
        }
        book.near_until.retain(|_, until| now < *until);
        book.near_until
            .iter()
            .map(|e| e.key().clone())
            .filter(|n| n != local && in_scope(n))
            .collect()
    }

    /// Is `node` left out of `model`'s chain because its lease lapsed?
    fn passed_over_for_chain(&self, model: &ModelId, node: &NodeId) -> bool {
        let key = (model.clone(), node.clone());
        let until = self.near_chain_book.passed_over_until.get(&key).map(|u| *u);
        match until {
            Some(until) if Instant::now() < until => true,
            Some(_) => {
                self.near_chain_book.passed_over_until.remove(&key);
                false
            }
            None => false,
        }
    }

    /// The machines `manifest`'s chain is planned over at `reach`, in
    /// rendezvous order: this node and the near ones — or every connected one
    /// in scope — not passed over. `None` when this node, or a near one,
    /// states no ceiling — cannot tell, and unknown is never short; one that
    /// has gone takes no part. A FAR machine stating none yet (connected a
    /// moment ago) is planned without: the far set differs per observer
    /// anyway, and in a large swarm a newcomer every minute would hand the
    /// model back and forth between the chain and carrying.
    fn chain_machines(
        &self,
        manifest: &ModelManifest,
        parts: &[Part],
        curve: (u64, u64),
        reach: Reach,
    ) -> Option<Vec<Machine>> {
        let state = &self.shared_state;
        let local = state.identity.node_id().clone();
        let mut nodes = match reach {
            Reach::Near => {
                let mut nodes = vec![local.clone()];
                nodes.extend(self.near_peers());
                nodes
            }
            Reach::Far => self.machines_in_scope(),
        };
        nodes.retain(|n| !self.passed_over_for_chain(&manifest.id, n));
        let sid = |index| ShardId {
            model_id: manifest.id.clone(),
            index,
        };
        let holders: Vec<Vec<NodeId>> = parts
            .iter()
            .map(|p| state.model_registry.shard_holders(&sid(p.index)))
            .collect();
        let fetching: Vec<Vec<NodeId>> = parts
            .iter()
            .map(|p| {
                let mut nodes: Vec<NodeId> = state
                    .models
                    .peer_shard_downloads
                    .get(&sid(p.index))
                    .map(|v| v.iter().map(|(n, _)| n.clone()).collect())
                    .unwrap_or_default();
                if state.models.is_shard_in_progress(&manifest.id, p.index) {
                    nodes.push(local.clone());
                }
                nodes
            })
            .collect();
        let layers_marked = |marks: &[bool]| -> u32 {
            parts
                .iter()
                .zip(marks)
                .filter(|(_, marked)| **marked)
                .map(|(p, _)| p.end - p.start)
                .sum()
        };
        let mut machines = Vec::with_capacity(nodes.len());
        for node in nodes {
            let Some(ceiling) = self.ceiling_layers(&node, manifest, curve) else {
                if node == local
                    || (reach == Reach::Near && state.connected_node_ids.contains(&node))
                {
                    return None;
                }
                continue;
            };
            let held: Vec<bool> = holders.iter().map(|h| h.contains(&node)).collect();
            let has: Vec<bool> = held
                .iter()
                .zip(&fetching)
                .map(|(&h, f)| h || f.contains(&node))
                .collect();
            let held_layers = layers_marked(&held);
            let has_layers = layers_marked(&has);
            let fetching_now = fetching.iter().any(|f| f.contains(&node));
            let disk_bytes = self
                .advertised(&node)
                .map_or(0, |(_, mb)| mb.saturating_mul(1024 * 1024));
            machines.push(Machine {
                node,
                ceiling,
                has,
                held_layers,
                has_layers,
                fetching: fetching_now,
                disk_bytes,
            });
        }
        let rank = |m: &Machine| rendezvous(&manifest.id, &m.node);
        // A machine that could run the model whole survives the cut whatever
        // it holds: it makes the model carrying's (`plan_over`), and ten
        // holders of a part each must not hide it (v0.3.254's review).
        let whole = |m: &Machine| m.ceiling >= manifest.num_layers;
        machines.sort_by(|a, b| {
            whole(b)
                .cmp(&whole(a))
                .then(b.has_layers.cmp(&a.has_layers))
                .then(b.ceiling.cmp(&a.ceiling))
                .then_with(|| rank(b).cmp(&rank(a)))
        });
        machines.truncate(MOST_MACHINES);
        machines.sort_by_key(|m| std::cmp::Reverse(rank(m)));
        Some(machines)
    }

    /// Does `machine`, named for parts it lacks, keep its place in `model`'s
    /// chain? Its lease is renewed while it gains layers or fetches a part;
    /// after `CARRIER_PATIENCE` with neither it is passed over for
    /// `PASSED_OVER_FOR` — a machine whose own budget or disk refuses its
    /// segment says nothing, and would hold the chain incomplete for ever.
    fn named_machine_keeps_its_place(&self, model: &ModelId, machine: &Machine) -> bool {
        let book = &self.near_chain_book;
        let key = (model.clone(), machine.node.clone());
        let now = Instant::now();
        let renewed = (now + CARRIER_PATIENCE, machine.held_layers);
        let lease = book.leases.get(&key).map(|l| *l);
        let lapsed = match lease {
            Some((_, held)) if machine.held_layers > held || machine.fetching => {
                book.leases.insert(key.clone(), renewed);
                false
            }
            Some((lapses_at, _)) => now >= lapses_at,
            None => {
                book.leases.insert(key.clone(), renewed);
                false
            }
        };
        if lapsed {
            tracing::info!(
                model = %model,
                machine = %machine.node,
                "DIAG: a nearby machine named for part of this model has fetched none of it — \
                 planning the chain without it for a while (Stage 3)"
            );
            book.leases.remove(&key);
            book.passed_over_until.insert(key, now + PASSED_OVER_FOR);
        }
        !lapsed
    }

    /// May this node plan `model_id`'s near chain — asked for, and a model it
    /// would host? One the owner did not choose with its safety training
    /// removed (`SharedState::unchosen_uncensored_model`) is never fetched
    /// here, so it is planned no chain and its header is not fetched for one
    /// (the v0.3.247 security review). The chain and the header ask this.
    fn may_plan_a_near_chain(&self, model_id: &ModelId) -> bool {
        self.asked_for(model_id) && !self.shared_state.unchosen_uncensored_model(model_id)
    }

    /// `manifest`'s chain, computed now: over the near machines, else — where
    /// they cannot carry it — over every machine in scope (#346). `None`:
    /// nobody asked for it, no header here, a machine planned over could run
    /// it whole, they could not carry it between them, or a ceiling is
    /// unknown.
    fn compute_near_chain(&self, manifest: &ModelManifest) -> Option<NearChain> {
        let layers = manifest.num_layers;
        if layers == 0 || near_chains_switched_off() || !self.may_plan_a_near_chain(&manifest.id) {
            return None;
        }
        let curve = self.carry_curve(&manifest.id)?;
        let parts = parts_in_order(manifest)?;
        match self.plan_over(manifest, &parts, curve, Reach::Near)? {
            Planned::Chain(chain) => Some(chain),
            Planned::OneRunsItWhole => None,
            Planned::CannotCarry => match self.plan_over(manifest, &parts, curve, Reach::Far)? {
                Planned::Chain(chain) => Some(chain),
                Planned::OneRunsItWhole | Planned::CannotCarry => None,
            },
        }
    }

    /// Plan `manifest`'s chain over the machines at `reach`, passing over a
    /// named machine whose lease lapsed. `None`: a ceiling is unknown.
    fn plan_over(
        &self,
        manifest: &ModelManifest,
        parts: &[Part],
        curve: (u64, u64),
        reach: Reach,
    ) -> Option<Planned> {
        let layers = manifest.num_layers;
        let mut machines = self.chain_machines(manifest, parts, curve, reach)?;
        // Each run weighed as what it loads, against the average layer the
        // ceilings count in (`carry_curve`'s, per MB): the header's tables at
        // the model's ends (field report #039).
        let weights = self
            .shared_state
            .gguf_meta_for(&manifest.id)
            .and_then(|meta| ModelWeights::for_planning(&meta))
            .map(|table| RangeWeights {
                weights: std::sync::Arc::new(table),
                num_layers: layers,
                average_bytes: manifest.total_size_bytes / u64::from(layers.max(1)),
                layer_price_bytes: curve.1.saturating_mul(1024 * 1024),
                on_card: false,
            });
        loop {
            // A chain is for a model that needs a split: a machine that could
            // run it whole is today's rules' to fill.
            let total: u64 = machines.iter().map(|m| u64::from(m.ceiling)).sum();
            if machines.iter().any(|m| m.ceiling >= layers) {
                return Some(Planned::OneRunsItWhole);
            }
            if total < u64::from(layers) {
                return Some(Planned::CannotCarry);
            }
            let held = plan_chain(layers, parts, &machines, false, weights.as_ref());
            let complete = held.is_some();
            let Some(steps) =
                held.or_else(|| plan_chain(layers, parts, &machines, true, weights.as_ref()))
            else {
                return Some(Planned::CannotCarry);
            };
            let mut named = HashSet::new();
            let mut stalled = None;
            for &(i, x, e) in &steps {
                let machine = &machines[i];
                if lacks_any(parts, machine, x, e) {
                    named.insert(machine.node.clone());
                    if !self.named_machine_keeps_its_place(&manifest.id, machine) {
                        stalled = Some(i);
                        break;
                    }
                }
            }
            if let Some(i) = stalled {
                machines.remove(i);
                continue;
            }
            self.near_chain_book
                .leases
                .retain(|(model, node), _| model != &manifest.id || named.contains(node));
            let chain = NearChain {
                segments: steps
                    .iter()
                    .map(|&(i, x, e)| (machines[i].node.clone(), (x, e)))
                    .collect(),
                complete,
                reach,
            };
            tracing::debug!(
                model = %manifest.id,
                machines = machines.len(),
                segments = chain.segments.len(),
                complete,
                reach = ?reach,
                plan = %chain.describe(),
                "near chain planned (Stage 3)"
            );
            return Some(Planned::Chain(chain));
        }
    }

    /// `manifest`'s near chain, as last computed within
    /// [`CHAIN_ANSWERS_FOR`]. Each computation names this node's parts anew,
    /// to be kept [`NAMED_PART_HOLD`].
    fn near_chain(&self, manifest: &ModelManifest) -> Option<NearChain> {
        let book = &self.near_chain_book;
        let now = Instant::now();
        let cached = book
            .chains
            .get(&manifest.id)
            .filter(|c| now < c.0)
            .map(|c| c.1.clone());
        if let Some(chain) = cached {
            return chain;
        }
        let chain = self.compute_near_chain(manifest);
        let before = book.chains.get(&manifest.id).and_then(|c| c.1.clone());
        if let Some(planned) = chain.as_ref().filter(|c| before.as_ref() != Some(*c)) {
            tracing::info!(
                model = %manifest.id,
                reach = ?planned.reach,
                complete = planned.complete,
                plan = %planned.describe(),
                "DIAG: a chain of machines is planned for a model none of them can run whole \
                 (Stage 3; over every machine in scope where the near ones cannot carry it, #346)"
            );
        }
        book.named_until.retain(|_, until| now < *until);
        let local = self.shared_state.identity.node_id();
        if let Some(segment) = chain.as_ref().and_then(|c| c.segment_of(local)) {
            for index in parts_touching(manifest, segment) {
                book.named_until.insert(
                    ShardId {
                        model_id: manifest.id.clone(),
                        index,
                    },
                    now + NAMED_PART_HOLD,
                );
            }
        }
        book.chains.insert(
            manifest.id.clone(),
            (now + CHAIN_ANSWERS_FOR, chain.clone()),
        );
        chain
    }

    /// Is a chain planned for `manifest` — near or far? Then carrying stands
    /// down for it (`coverage::needs_carrier`, `copy_carries_model`): the chain
    /// places every segment, and a carrier fetching beside it in model order
    /// would fetch what the chain does not use, and keep a holder's every copy
    /// for the few layers it adds.
    pub(in crate::model::auto_manage) fn chain_planned_for(
        &self,
        manifest: &ModelManifest,
    ) -> bool {
        self.near_chain(manifest).is_some()
    }

    /// The parts THIS node fetches as its segment of `manifest`'s near chain:
    /// those it does not hold. None when the chain does not name it.
    pub(in crate::model::auto_manage) fn parts_this_node_plans(
        &self,
        manifest: &ModelManifest,
    ) -> HashSet<u32> {
        let local = self.shared_state.identity.node_id();
        let Some(segment) = self.near_chain(manifest).and_then(|c| c.segment_of(local)) else {
            return HashSet::new();
        };
        parts_touching(manifest, segment)
            .into_iter()
            .filter(|&index| {
                !self
                    .shared_state
                    .model_registry
                    .shard_holders(&ShardId {
                        model_id: manifest.id.clone(),
                        index,
                    })
                    .contains(local)
            })
            .collect()
    }

    /// Is `shard_id` in this node's segment of its model's near chain — named
    /// now, or within [`NAMED_PART_HOLD`]? Prune keeps such a part
    /// (`copy_kept_because`); the download pass, asking the same before it
    /// fetches, fetches it. The chain counts in-flight parts as held, so the
    /// answer before a fetch is the answer after it.
    ///
    /// Prune keeps a part of the chain planned NOW by its reach — a far
    /// chain's at any pressure (it is how a model only a split can run runs at
    /// all, as carrying's copies are — #346), a near chain's below urgent (it
    /// makes a split fast) — and one named only within the day below urgent:
    /// the hold rides out a neighbour that flaps, never a full disk.
    pub(in crate::model::auto_manage) fn copy_in_near_chain(
        &self,
        shard_id: &ShardId,
    ) -> Option<ChainPart> {
        if shard_id.index == MMPROJ_SHARD_INDEX {
            return None;
        }
        if let Some(manifest) = self
            .shared_state
            .model_registry
            .get_manifest(&shard_id.model_id)
        {
            // Asked first: it names this node's parts anew.
            let local = self.shared_state.identity.node_id();
            if let Some(chain) = self.near_chain(&manifest) {
                if chain.segment_of(local).is_some_and(|segment| {
                    parts_touching(&manifest, segment).contains(&shard_id.index)
                }) {
                    return Some(ChainPart::PlannedNow(chain.reach));
                }
            }
        }
        self.near_chain_book
            .named_until
            .get(shard_id)
            .is_some_and(|until| Instant::now() < *until)
            .then_some(ChainPart::NamedLately)
    }

    /// Forget every computed chain, as a minute passing would.
    #[cfg(test)]
    pub(in crate::model::auto_manage) fn forget_planned_chains(&self) {
        self.near_chain_book.chains.clear();
    }

    /// Does this node hold `model_id` loaded over layers OUTSIDE its segment of
    /// a near chain whose every machine holds its segment, this one included
    /// ([`Self::chain_held_by_its_machines`])? Then the idle-unload pass gives
    /// that memory back once the conversation using it is over, not after the
    /// idle window.
    ///
    /// The router holds a peer to its room BESIDE the layers it has loaded
    /// (`scheduler::PublishedRoom`), and an idle worker is retired only when
    /// asked for a range it cannot add. So a near machine still holding the
    /// range a far-reaching route gave it is never planned its own segment:
    /// in the rig, B held layers 0-12 of a chain it was named 12-22 in, and
    /// the next request went B + a far machine rather than A + B. Kept busy,
    /// the idle window never came and the chain never formed. A Petals server
    /// serves only the span it chose, and clients route over what is served;
    /// here the routers price the cold load the chain then costs.
    pub(in crate::model::auto_manage) fn loaded_outside_near_segment(
        &self,
        model_id: &ModelId,
    ) -> bool {
        let held: Vec<(u32, u32)> = self
            .shared_state
            .model_process_pool
            .held_layer_ranges_for_planning(model_id)
            .iter()
            .map(|held| held.range)
            .collect();
        self.held_outside_near_segment(model_id, &held)
    }

    /// [`Self::loaded_outside_near_segment`] for the ranges `held`.
    fn held_outside_near_segment(&self, model_id: &ModelId, held: &[(u32, u32)]) -> bool {
        let book = &self.near_chain_book;
        if book
            .off_segment_released_until
            .get(model_id)
            .is_some_and(|until| Instant::now() < *until)
        {
            return false;
        }
        let Some(manifest) = self.shared_state.model_registry.get_manifest(model_id) else {
            return false;
        };
        let local = self.shared_state.identity.node_id();
        let Some(chain) = self.near_chain(&manifest) else {
            return false;
        };
        let Some(segment) = chain.segment_of(local) else {
            return false;
        };
        if !self.chain_held_by_its_machines(&manifest, &chain) {
            return false;
        }
        held.iter()
            .any(|&(start, end)| start < segment.0 || end > segment.1)
    }

    /// Does every machine of `chain` HOLD its segment — each part touching it
    /// on that machine's disk, as the routers read holdings? Then the chain
    /// can be routed over now. A part being fetched counts toward PLANNING a
    /// chain (`NearChain::complete`), never toward this: no router plans a
    /// part still on its way, and a peer's fetch is only its own word
    /// (`peer_shard_downloads`) — taken here, any near peer could have this
    /// node give back loaded layers after the conversation gap instead of the
    /// idle window (the v0.3.247 security review).
    fn chain_held_by_its_machines(&self, manifest: &ModelManifest, chain: &NearChain) -> bool {
        let registry = &self.shared_state.model_registry;
        chain.segments.iter().all(|(node, segment)| {
            parts_touching(manifest, *segment).into_iter().all(|index| {
                registry
                    .shard_holders(&ShardId {
                        model_id: manifest.id.clone(),
                        index,
                    })
                    .contains(node)
            })
        })
    }

    /// `model_id`'s layers outside this node's segment were given back: leave
    /// it to the ordinary idle rule for [`OFF_SEGMENT_RELEASE_HOLD`].
    pub(in crate::model::auto_manage) fn note_off_segment_release(&self, model_id: &ModelId) {
        self.near_chain_book
            .off_segment_released_until
            .insert(model_id.clone(), Instant::now() + OFF_SEGMENT_RELEASE_HOLD);
    }

    /// The models whose header this node should learn so it can plan their
    /// chain: asked for, no cost curve here (it is read from the header), and
    /// a part already held by a near machine — whose chain may name this one —
    /// or, for a model whose weights alone outgrow every machine in scope's
    /// ceiling, by any machine in scope: its far chain may name this one
    /// (#346). A node fetches a header only to route a model or to fetch a
    /// part of it, so a machine holding none of the model never computed the
    /// plan naming it, never fetched, and was passed over. Only where
    /// HuggingFace can give it (`ensure_model_geometry`'s source).
    fn models_a_near_chain_needs_a_header_for(&self) -> Vec<ModelId> {
        let near = self.near_peers();
        let in_scope = self.machines_in_scope();
        // Unknown for any machine: it might run the model whole, so no far
        // chain is assumed.
        let largest_ceiling_mb = in_scope
            .iter()
            .map(|node| self.advertised(node).and_then(|(ceiling, _)| ceiling))
            .collect::<Option<Vec<u64>>>()
            .and_then(|ceilings| ceilings.into_iter().max());
        let registry = &self.shared_state.model_registry;
        let mut wanted = Vec::new();
        for manifest in registry.models() {
            if wanted.len() >= MOST_HEADERS_A_PASS {
                break;
            }
            if !self.may_plan_a_near_chain(&manifest.id)
                || !self
                    .shared_state
                    .models
                    .hf_sources
                    .contains_key(&manifest.id)
                || self.carry_curve(&manifest.id).is_some()
            {
                continue;
            }
            let layer_parts = || {
                manifest
                    .shards
                    .iter()
                    .filter(|s| s.index != MMPROJ_SHARD_INDEX)
            };
            let held_by = |machines: &[NodeId]| {
                layer_parts().any(|s| {
                    registry
                        .shard_holders(&ShardId {
                            model_id: manifest.id.clone(),
                            index: s.index,
                        })
                        .iter()
                        .any(|h| machines.contains(h))
                })
            };
            let weights_mb = layer_parts().map(|s| s.size_bytes).sum::<u64>() / (1024 * 1024);
            let only_a_split_runs_it = largest_ceiling_mb.is_some_and(|c| weights_mb > c);
            if held_by(&near) || (only_a_split_runs_it && held_by(&in_scope)) {
                wanted.push(manifest.id.clone());
            }
        }
        wanted
    }

    /// Learn the headers [`Self::models_a_near_chain_needs_a_header_for`]
    /// names — a few MB each, the fetch routing makes, bounded by its own
    /// budget and cooldown. Not offline, not on a metered connection.
    pub(in crate::model::auto_manage) async fn learn_headers_near_chains_need(&self) {
        let wanted = self.models_a_near_chain_needs_a_header_for();
        if wanted.is_empty()
            || self
                .shared_state
                .credits
                .offline_mode
                .load(std::sync::atomic::Ordering::Relaxed)
            || self.shared_state.metered_hold().await
        {
            return;
        }
        for model in wanted {
            self.shared_state.ensure_model_geometry(&model).await;
            // The curve was cached as unknown; read it from the header now.
            self.carry_curves.remove(&model);
            self.near_chain_book.chains.remove(&model);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::Ordering::Relaxed;
    use std::sync::Arc;
    use std::time::Instant;

    use super::super::super::test_support::{
        capability, fetched, holds, make_test_manager_with_config, offered, peer_with_disk,
        register_manifest_with_sized_shards, write_sparse_shards,
    };
    use super::super::super::AutoShardManager;
    use super::{plan_chain, Machine, NearChain, Part, Reach};
    use crate::config::Config;
    use crate::daemon::SharedState;
    use crate::types::{ModelId, ModelManifest, NodeId, ShardId};

    const MIB: u64 = 1024 * 1024;
    const PART: u64 = 500 * MIB;
    /// What a holder's admission charges: 300 MB + 200 MB a layer.
    const CURVE: (u64, u64) = (300, 200);
    /// Room for `layers` of it, by that curve.
    fn ceiling_for(layers: u64) -> u64 {
        CURVE.0 + CURVE.1 * layers
    }

    /// `node`'s measured round trip, as the PEX ping records it.
    fn measured(state: &Arc<SharedState>, node: &NodeId, rtt_ms: f32) {
        let now_ms = chrono::Utc::now().timestamp_millis().max(0) as u64;
        state
            .metrics
            .network_coord_samples
            .entry(node.clone())
            .or_default()
            .observe(now_ms, rtt_ms);
    }

    struct Swarm {
        state: Arc<SharedState>,
        manager: AutoShardManager,
        mid: ModelId,
        /// Could run 20 of the 32 layers; holds nothing.
        near: NodeId,
        /// 200 ms away, holds the whole model and could run it.
        far: NodeId,
    }

    impl Swarm {
        /// Forget each model's computed chain, as a minute passing would.
        fn later(&self) {
            self.manager.near_chain_book.chains.clear();
        }

        fn manifest(&self) -> ModelManifest {
            self.state.model_registry.get_manifest(&self.mid).unwrap()
        }

        fn chain(&self) -> Option<NearChain> {
            self.later();
            self.manager.near_chain(&self.manifest())
        }

        fn part(&self, index: u32) -> ShardId {
            ShardId {
                model_id: self.mid.clone(),
                index,
            }
        }

        fn local(&self) -> NodeId {
            self.state.identity.node_id().clone()
        }

        fn offered(&self) -> Vec<u32> {
            self.later();
            offered(&self.manager, &self.state)
        }

        fn sheds(&self, index: u32, holders: &[NodeId], pressure: f64) -> bool {
            self.later();
            self.manager
                .would_shed_copy(&self.part(index), holders, pressure, &[])
        }
    }

    /// A 32-layer model in eight 4-layer parts, asked for in another region. A
    /// FAR peer (200 ms) holds it whole and could run it, so the replica target
    /// is met and nothing needs carrying: only a near chain moves anything.
    /// This node could run 16 of its layers, the peer `near_rtt_ms` away 20.
    fn swarm(near_rtt_ms: f32) -> Swarm {
        let mut config = Config::default();
        config.auto_manage.min_replicas = 1;
        let (state, manager) = make_test_manager_with_config(config);
        state.models.auto_manage_default_model_cap.store(0, Relaxed);
        let ranges: Vec<(u32, u32)> = (0..8).map(|i| (i * 4, i * 4 + 4)).collect();
        let mid = register_manifest_with_sized_shards(&state, "near-me", 32, &ranges, PART);
        state
            .model_process_pool
            .test_cost_curve
            .insert(mid.clone(), CURVE);
        state
            .local_capability
            .store(Some(Arc::new(capability(Some(ceiling_for(16)), 100_000))));
        let near = peer_with_disk(&state, 2, Some(ceiling_for(20)), 100_000);
        measured(&state, &near, near_rtt_ms);
        let far = peer_with_disk(&state, 3, Some(ceiling_for(32)), 100_000);
        measured(&state, &far, 200.0);
        holds(&state, &mid, &far, &[0, 1, 2, 3, 4, 5, 6, 7]);
        state
            .region_demand
            .insert((mid.clone(), "BE".to_string()), 2.0);
        let mut trust = crate::types::ModelTrustInfo::new_discovered();
        trust.trust_level = crate::types::ModelTrustLevel::DemandVerified;
        state.models.model_trust.insert(mid.clone(), trust);
        Swarm {
            state,
            manager,
            mid,
            near,
            far,
        }
    }

    /// [`swarm`] with the near peer holding parts 2-6 — this machine and the
    /// LAN box: the chain is this node on layers 0-12 (parts 0-2) and the
    /// peer on 12-32 (it lacks part 7), four parts against five for any other.
    fn fleet(near_rtt_ms: f32) -> Swarm {
        let s = swarm(near_rtt_ms);
        holds(&s.state, &s.mid, &s.near, &[2, 3, 4, 5, 6]);
        s
    }

    /// The near chain gives this node its segment's parts — which routine
    /// replication, satisfied by the far copy, never would — and it starts
    /// them at once. The far peer is no part of it; with the peer far too,
    /// or nobody asking, nothing is offered (the null controls).
    #[test]
    fn nearby_machines_fetch_complementary_parts_of_a_model_none_can_run_whole() {
        let s = fleet(4.0);
        let chain = s.chain().expect("a near chain");
        assert_eq!(
            chain.segments,
            vec![(s.local(), (0, 12)), (s.near.clone(), (12, 32))]
        );
        assert!(!chain.complete);
        assert_eq!(s.offered(), vec![0, 1, 2]);
        s.later();
        // A cycle starts at most its contribution mode's cap of them (one or
        // two with this few peers), so the next cycle starts the rest.
        let started = fetched(&s.manager, &s.state);
        assert!(
            !started.is_empty() && started.iter().all(|i| [0, 1, 2].contains(i)),
            "started {started:?}"
        );

        let far = fleet(200.0);
        assert!(far.chain().is_none());
        assert!(
            far.offered().is_empty(),
            "no near machine: routine replication is met and nothing moves"
        );

        s.state.region_demand.clear();
        assert!(s.offered().is_empty(), "nobody asked");
    }

    /// Prune keeps this node's segment of a near chain and sheds what lies
    /// outside it, below urgent disk pressure; an urgent disk takes it. A
    /// MIDDLE part: prompt privacy keeps the ends anyway (gotcha #876's
    /// first test passed without its rule that way).
    #[test]
    fn prune_keeps_this_computers_part_of_the_chain_and_sheds_the_rest() {
        let s = fleet(4.0);
        let local = s.local();
        holds(&s.state, &s.mid, &local, &[0, 1, 2, 5]);
        write_sparse_shards(&s.state, &s.mid, [0, 1, 2, 5], PART);
        holds(&s.state, &s.mid, &s.near, &[7]);
        let chain = s.chain().expect("a near chain");
        assert!(chain.complete, "this node 0-12, the peer 12-32");

        // Three copies — a nearby one among them, so no region rule keeps
        // either part: the count alone sheds part 5 and would shed part 1.
        let three = [s.near.clone(), s.far.clone(), local.clone()];
        assert!(
            !s.sheds(1, &three, 0.9),
            "a copy too many by the count — kept: the chain needs it here"
        );
        assert!(
            s.sheds(5, &three, 0.9),
            "part 5 is outside this node's segment"
        );
        assert!(
            s.sheds(1, &three, 0.99),
            "an urgent disk takes it: a near chain makes a split fast, not possible"
        );
    }

    /// Memory holding layers outside this node's segment of a COMPLETE chain,
    /// with the segment's parts here, is given back early — once, then the
    /// model is left to the ordinary idle rule for the hold. Layers inside the
    /// segment, an incomplete chain or a segment still being fetched change
    /// nothing.
    #[test]
    fn layers_loaded_outside_this_computers_segment_are_given_back() {
        let s = fleet(4.0);
        let local = s.local();
        holds(&s.state, &s.mid, &local, &[0, 1, 2]);
        let chain = s.chain().expect("a near chain");
        assert!(!chain.complete, "the peer lacks part 7");
        s.later();
        assert!(
            !s.manager.held_outside_near_segment(&s.mid, &[(12, 20)]),
            "an incomplete chain cannot be routed over yet"
        );

        holds(&s.state, &s.mid, &s.near, &[7]);
        assert!(s.chain().expect("a near chain").complete);
        for inside in [vec![], vec![(0, 12)], vec![(4, 12)]] {
            s.later();
            assert!(
                !s.manager.held_outside_near_segment(&s.mid, &inside),
                "{inside:?} is this node's segment, or nothing"
            );
        }
        for outside in [vec![(12, 20)], vec![(8, 16)], vec![(0, 12), (12, 16)]] {
            s.later();
            assert!(
                s.manager.held_outside_near_segment(&s.mid, &outside),
                "{outside:?} reaches outside 0-12"
            );
        }

        s.manager.note_off_segment_release(&s.mid);
        s.later();
        assert!(
            !s.manager.held_outside_near_segment(&s.mid, &[(12, 20)]),
            "loaded again after a release: the ordinary idle rule, for the hold"
        );
    }

    /// The segment's parts must be HERE: a node still fetching its segment
    /// keeps what it has loaded.
    #[test]
    fn a_segment_still_being_fetched_keeps_what_is_loaded() {
        let s = fleet(4.0);
        let local = s.local();
        holds(&s.state, &s.mid, &local, &[0, 1]);
        holds(&s.state, &s.mid, &s.near, &[7]);
        let _downloading = s
            .state
            .models
            .claim_shard_download(&s.part(2))
            .expect("nobody else is writing part 2");
        assert!(
            s.chain().expect("a near chain").complete,
            "the chain counts part 2, on its way, as held"
        );
        s.later();
        assert!(
            !s.manager.held_outside_near_segment(&s.mid, &[(12, 20)]),
            "part 2 of this node's segment is not here yet"
        );
    }

    /// A chain is routed over only once every machine HOLDS its segment: a
    /// PEER still fetching its part — by its own word, which is all
    /// `peer_shard_downloads` is — leaves this node's layers loaded.
    #[test]
    fn a_peer_segment_still_being_fetched_keeps_what_is_loaded() {
        let s = fleet(4.0);
        let local = s.local();
        holds(&s.state, &s.mid, &local, &[0, 1, 2]);
        s.state
            .models
            .peer_shard_downloads
            .insert(s.part(7), vec![(s.near.clone(), 50)]);
        assert!(
            s.chain().expect("a near chain").complete,
            "the chain counts the peer's part 7, on its way, as held"
        );
        s.later();
        assert!(
            !s.manager.held_outside_near_segment(&s.mid, &[(12, 20)]),
            "the peer says it is fetching part 7; nobody can route over it yet"
        );

        s.state.models.peer_shard_downloads.clear();
        holds(&s.state, &s.mid, &s.near, &[7]);
        s.later();
        assert!(
            s.manager.held_outside_near_segment(&s.mid, &[(12, 20)]),
            "control: the peer holds part 7"
        );
    }

    /// A chain the near machines already complete is never rebalanced, even
    /// where fetching would shorten it: nothing is offered.
    #[test]
    fn a_complete_near_chain_is_never_rebalanced() {
        let s = swarm(4.0);
        let local = s.local();
        holds(&s.state, &s.mid, &local, &[0, 1]);
        holds(&s.state, &s.mid, &s.near, &[2, 3, 4]);
        let other = peer_with_disk(&s.state, 4, Some(ceiling_for(16)), 100_000);
        measured(&s.state, &other, 6.0);
        holds(&s.state, &s.mid, &other, &[5, 6, 7]);
        let chain = s.chain().expect("a near chain");
        assert!(chain.complete);
        assert_eq!(
            chain.segments,
            vec![
                (local, (0, 8)),
                (s.near.clone(), (8, 20)),
                (other, (20, 32))
            ],
            "three segments, though two would do after a few fetches"
        );
        assert!(s.offered().is_empty());
    }

    /// A near machine that drops off keeps its place, and its parts count, for
    /// half an hour: the chain does not change. After that it is gone, and
    /// this node alone cannot carry the model.
    #[test]
    fn a_neighbour_that_drops_off_keeps_its_place_for_half_an_hour() {
        let s = fleet(4.0);
        assert_eq!(s.offered(), vec![0, 1, 2]);

        s.state.connected_node_ids.remove(&s.near);
        assert_eq!(s.offered(), vec![0, 1, 2], "a blip is not a gap");

        s.manager
            .near_chain_book
            .near_until
            .insert(s.near.clone(), Instant::now());
        assert!(s.chain().is_none());
        assert!(s.offered().is_empty());
    }

    /// A part this node was named for is kept a day after the chain stops
    /// naming it — here because the node can now run the model whole — and
    /// shed like any other after that.
    #[test]
    fn a_part_named_for_this_computer_is_kept_a_day() {
        let s = fleet(4.0);
        let local = s.local();
        holds(&s.state, &s.mid, &local, &[0, 1, 2]);
        write_sparse_shards(&s.state, &s.mid, [0, 1, 2], PART);
        holds(&s.state, &s.mid, &s.near, &[7]);
        assert!(s.chain().is_some());
        // Three copies, a nearby one among them: the count alone sheds one.
        let three = [s.near.clone(), s.far.clone(), local];

        s.state
            .local_capability
            .store(Some(Arc::new(capability(Some(ceiling_for(32)), 100_000))));
        assert!(s.chain().is_none(), "it runs here whole: no near chain");
        assert!(!s.sheds(1, &three, 0.9), "named a moment ago: kept");

        s.manager
            .near_chain_book
            .named_until
            .insert(s.part(1), Instant::now());
        assert!(s.sheds(1, &three, 0.9), "a day on: shed like any other");
    }

    /// A machine named for parts it never fetches is passed over after its
    /// patience, and the chain names another; one that gains a part keeps
    /// its place.
    #[test]
    fn a_named_machine_that_makes_no_progress_is_passed_over() {
        let s = swarm(4.0);
        let local = s.local();
        holds(&s.state, &s.mid, &local, &[0, 1, 2, 3]);
        let other = peer_with_disk(&s.state, 4, Some(ceiling_for(20)), 100_000);
        measured(&s.state, &other, 5.0);
        let tail = |s: &Swarm| {
            let chain = s.chain().expect("a near chain");
            assert_eq!(chain.segment_of(&local), Some((0, 16)));
            chain
                .segments
                .iter()
                .find(|(_, range)| *range == (16, 32))
                .map(|(n, _)| n.clone())
                .expect("a machine on layers 16-32")
        };
        let first = tail(&s);
        let second = if first == s.near {
            other.clone()
        } else {
            s.near.clone()
        };
        let age = |s: &Swarm, node: &NodeId| {
            s.manager
                .near_chain_book
                .leases
                .get_mut(&(s.mid.clone(), node.clone()))
                .expect("a lease")
                .0 = Instant::now();
        };

        holds(&s.state, &s.mid, &first, &[7]);
        age(&s, &first);
        assert_eq!(tail(&s), first, "it gained a part");

        age(&s, &first);
        assert_eq!(tail(&s), second, "no progress since: passed over");
        assert_eq!(tail(&s), second, "and the other keeps it");
    }

    /// In private mode a machine outside the pool is not near: no chain.
    #[test]
    fn a_machine_outside_the_pool_is_no_part_of_a_near_chain() {
        let s = fleet(4.0);
        assert_eq!(s.offered(), vec![0, 1, 2], "control: open swarm");
        s.state.credits.private_mode.store(true, Relaxed);
        assert!(s.chain().is_none());
    }

    /// [`swarm`] with the machines far apart and none able to run the model:
    /// this node could run 20 of the 32 layers, the peer 250 ms away 16, the
    /// one 200 ms away 12 — and it holds the model whole.
    fn apart() -> Swarm {
        let s = swarm(250.0);
        s.state
            .local_capability
            .store(Some(Arc::new(capability(Some(ceiling_for(20)), 100_000))));
        peer_with_disk(&s.state, 2, Some(ceiling_for(16)), 100_000);
        peer_with_disk(&s.state, 3, Some(ceiling_for(12)), 100_000);
        s.manager.carry_curves.clear();
        s
    }

    /// A machine connected a moment ago, its ceiling not yet stated, is
    /// planned without: a far chain does not wait on every newcomer. (A NEAR
    /// one stating none still stops the plan:
    /// `a_near_machine_with_no_ceiling_changes_nothing`.)
    #[test]
    fn a_far_chain_is_planned_without_a_machine_that_has_not_stated_its_memory() {
        let s = apart();
        assert_eq!(s.chain().map(|c| c.reach), Some(Reach::Far), "control");
        peer_with_disk(&s.state, 7, None, 100_000);
        assert_eq!(s.chain().map(|c| c.reach), Some(Reach::Far));
    }

    /// A machine that could run the model whole is never cut from the
    /// machines a chain is planned over, however many hold parts of it: the
    /// model is carrying's then, not a far chain's (pre-release review of
    /// v0.3.254 — the cut to `MOST_MACHINES` ranked by layers held, so ten
    /// holders of a part each pushed it out).
    #[test]
    fn a_machine_that_could_run_the_model_whole_is_never_cut_from_the_count() {
        let s = apart();
        // Ten more machines hold a part each and could run 8 layers.
        for byte in 10..20u8 {
            let holder = peer_with_disk(&s.state, byte, Some(ceiling_for(8)), 100_000);
            holds(&s.state, &s.mid, &holder, &[u32::from(byte % 8)]);
        }
        assert_eq!(
            s.chain().map(|c| c.reach),
            Some(Reach::Far),
            "control: no machine could run it whole"
        );
        // One that could, holding nothing.
        peer_with_disk(&s.state, 30, Some(ceiling_for(32)), 100_000);
        assert!(s.chain().is_none());
    }

    /// Where the near machines cannot carry a model only a split can run,
    /// the chain is planned over every machine in scope (#346) — the 70B's
    /// shape: holders far apart, none able to run it, enough between them.
    /// This node is named a segment and offered its parts, prune keeps them,
    /// and carrying — which would fetch in model order beside it — stands
    /// down. With a machine able to run the model whole, or the ceilings
    /// short of it, no chain (the controls).
    #[test]
    fn machines_far_apart_that_only_together_can_run_a_model_plan_one_chain() {
        let s = apart();
        let chain = s.chain().expect("a far chain");
        assert_eq!(chain.reach, Reach::Far);
        assert!(!chain.complete);
        // Fewest segments (20 + 12 = 32), then fewest parts: the holder runs
        // what it holds and this node the rest — five parts, either way round.
        let width = |node: &NodeId| chain.segment_of(node).map(|(a, b)| b - a);
        assert_eq!(chain.segments.len(), 2);
        assert_eq!(width(&s.far), Some(12));
        assert_eq!(width(&s.local()), Some(20));
        let (start, end) = chain.segment_of(&s.local()).unwrap();
        let mine: Vec<u32> = (start / 4..end / 4).collect();
        assert_eq!(s.offered(), mine);
        assert!(
            !s.sheds(mine[2], &[s.far.clone(), s.local()], 0.9),
            "a part of this node's segment is kept below urgent pressure"
        );
        s.later();
        assert!(
            !s.manager.needs_carrier(&s.manifest()),
            "carrying stands down while a chain places the model"
        );

        // Controls: a machine that could run it whole leaves it to carrying,
        // and ceilings short of it plan nothing.
        let whole = swarm(250.0);
        assert!(whole.chain().is_none(), "the 200 ms peer could run all 32");
        let short = swarm(250.0);
        short
            .state
            .local_capability
            .store(Some(Arc::new(capability(Some(ceiling_for(8)), 100_000))));
        peer_with_disk(&short.state, 2, Some(ceiling_for(8)), 100_000);
        peer_with_disk(&short.state, 3, Some(ceiling_for(12)), 100_000);
        short.manager.carry_curves.clear();
        assert!(short.chain().is_none(), "8 + 8 + 12 of 32");
        assert!(short.offered().is_empty());
    }

    /// A near machine stating no ceiling (older than v0.3.230) cannot be
    /// planned around — the plan it computes is unknowable, and planning
    /// without it would part ways with it — so nothing changes, even where
    /// the others could complete a chain without it.
    #[test]
    fn a_near_machine_with_no_ceiling_changes_nothing() {
        let s = fleet(4.0);
        let other = peer_with_disk(&s.state, 4, Some(ceiling_for(20)), 100_000);
        measured(&s.state, &other, 5.0);
        assert!(s.chain().is_some(), "control: every ceiling known");
        s.state
            .peer_registry
            .get_mut(&s.near)
            .unwrap()
            .capability
            .as_mut()
            .unwrap()
            .model_memory_ceiling_mb = None;
        assert!(s.chain().is_none());
        assert!(s.offered().is_empty());
    }

    /// A node with no header for an asked-for model learns it when a near
    /// machine holds part of it — the chain that machine plans may name this
    /// one — and not otherwise.
    #[test]
    fn a_header_is_learned_where_a_near_chain_may_need_it() {
        let without_header = |s: &Swarm| {
            s.state.model_process_pool.test_cost_curve.clear();
            s.state.models.hf_sources.insert(
                s.mid.clone(),
                crate::daemon::state::HfSource {
                    repo_id: "someone/near-me-GGUF".into(),
                    filename: "near-me.gguf".into(),
                    mmproj_filename: None,
                },
            );
        };
        let s = fleet(4.0);
        without_header(&s);
        assert_eq!(
            s.manager.models_a_near_chain_needs_a_header_for(),
            vec![s.mid.clone()]
        );

        let far = fleet(200.0);
        without_header(&far);
        assert!(far
            .manager
            .models_a_near_chain_needs_a_header_for()
            .is_empty());

        // A model with its safety training removed that the owner did not
        // choose is never hosted here: no chain, no header for one.
        s.state.models.hf_sources.insert(
            s.mid.clone(),
            crate::daemon::state::HfSource {
                repo_id: "someone/near-me-abliterated-GGUF".into(),
                filename: "near-me.gguf".into(),
                mmproj_filename: None,
            },
        );
        assert!(s.state.unchosen_uncensored_model(&s.mid));
        assert!(s
            .manager
            .models_a_near_chain_needs_a_header_for()
            .is_empty());
    }

    /// A model the owner did not choose with its safety training removed is
    /// planned no near chain here: this node would never fetch its parts.
    #[test]
    fn an_unchosen_uncensored_model_is_planned_no_near_chain() {
        let s = fleet(4.0);
        assert!(s.chain().is_some(), "control: an ordinary model");
        s.state.models.hf_sources.insert(
            s.mid.clone(),
            crate::daemon::state::HfSource {
                repo_id: "someone/near-me-abliterated-GGUF".into(),
                filename: "near-me.gguf".into(),
                mmproj_filename: None,
            },
        );
        assert!(s.chain().is_none());
        assert!(s.offered().is_empty());
    }

    fn machine(byte: u8, ceiling: u32, has: &[bool]) -> Machine {
        Machine {
            node: NodeId([byte; 32]),
            ceiling,
            has: has.to_vec(),
            held_layers: 0,
            has_layers: 0,
            fetching: false,
            disk_bytes: u64::MAX,
        }
    }

    fn parts(layers_each: &[u32]) -> Vec<Part> {
        let mut at = 0;
        layers_each
            .iter()
            .enumerate()
            .map(|(i, &n)| {
                let part = Part {
                    index: i as u32,
                    start: at,
                    end: at + n,
                    size: 100,
                };
                at += n;
                part
            })
            .collect()
    }

    /// Field report #039: a 70B's chain must not give a machine whose ceiling
    /// holds 7 of its AVERAGE layers the last 7 — they load the 822 MB output
    /// head beside them. Weighed as what each run loads, the tail goes to a
    /// machine with room for it, and every run fits its machine's ceiling.
    #[test]
    fn a_chains_runs_are_weighed_with_the_tables_at_the_models_ends() {
        const MB: u64 = 1024 * 1024;
        let names: Vec<String> = (0..80)
            .map(|i| format!("blk.{i}.ffn_down.weight"))
            .collect();
        let mut tensors: Vec<(&str, u64)> = names.iter().map(|n| (n.as_str(), 500 * MB)).collect();
        tensors.push(("token_embd.weight", 564 * MB));
        tensors.push(("output.weight", 822 * MB));
        let total: u64 = tensors.iter().map(|(_, b)| b).sum();
        let weights = crate::inference::process_pool::RangeWeights {
            weights: std::sync::Arc::new(
                crate::inference::process_pool::ModelWeights::of(&tensors).expect("layers named"),
            ),
            num_layers: 80,
            average_bytes: total / 80,
            layer_price_bytes: total / 80,
            on_card: false,
        };
        let parts = parts(&[1; 80]);
        let holding = |from: u32, to: u32| -> Vec<bool> {
            (0..80).map(|i| (from..to).contains(&i)).collect()
        };
        let machines = [
            machine(1, 73, &holding(0, 73)),
            machine(2, 7, &holding(70, 80)),
            machine(3, 10, &holding(60, 80)),
        ];
        // Null control: counted at the average, the 7-layer machine is given
        // 73-80 — nine layers' worth with the head.
        let unweighed = plan_chain(80, &parts, &machines, true, None).expect("a chain");
        assert!(
            unweighed
                .iter()
                .any(|&(i, x, e)| i == 1 && e == 80 && weights.width_in_bound((x, e)) > 7),
            "{unweighed:?}"
        );
        let weighed = plan_chain(80, &parts, &machines, true, Some(&weights)).expect("a chain");
        let mut at = 0;
        for &(i, x, e) in &weighed {
            assert_eq!(x, at, "{weighed:?}");
            assert!(
                weights.width_in_bound((x, e)) <= machines[i].ceiling,
                "machine {i} planned {x}..{e} past its ceiling: {weighed:?}"
            );
            at = e;
        }
        assert_eq!(at, 80, "{weighed:?}");
    }

    /// The simulator's fleet: 16 one-layer parts, this machine able to run 7,
    /// the LAN box 10 and holding 4-13. Closing it takes both to move at once
    /// — the case one-at-a-time carrying never finished.
    #[test]
    fn the_plan_moves_two_machines_at_once() {
        let parts = parts(&[1; 16]);
        let here = machine(1, 7, &[false; 16]);
        let held: Vec<bool> = (0..16).map(|i| (4..14).contains(&i)).collect();
        let lan_box = machine(2, 10, &held);
        assert_eq!(
            plan_chain(16, &parts, &[here.clone(), lan_box.clone()], true, None),
            Some(vec![(0, 0, 6), (1, 6, 16)]),
            "6 + 2 parts fetched; every other two-segment chain fetches more"
        );
        assert_eq!(plan_chain(16, &parts, &[here, lan_box], false, None), None);
    }

    /// A segment may end inside a part: two machines able to run 12 and 20 of
    /// a 32-layer model in 8-layer parts are a chain, though no cut on a part
    /// boundary fits them — the router splits by layer, and so does the plan.
    #[test]
    fn a_split_inside_a_part_is_a_plan() {
        let parts = parts(&[8, 8, 8, 8]);
        let small = machine(1, 12, &[false; 4]);
        let large = machine(2, 20, &[false; 4]);
        let chain = plan_chain(32, &parts, &[small, large], true, None).expect("a chain");
        assert_eq!(chain.len(), 2);
        assert_eq!(chain.iter().map(|&(_, x, e)| e - x).sum::<u32>(), 32);
    }

    /// A machine whose advertised disk cannot take its segment's parts is not
    /// given that segment.
    #[test]
    fn a_machine_without_the_disk_for_its_segment_is_not_given_it() {
        let parts = parts(&[4; 8]);
        let mut full = machine(1, 20, &[false; 8]);
        full.disk_bytes = 150;
        let roomy = machine(2, 20, &[false; 8]);
        let other = machine(3, 20, &[false; 8]);
        let chain = plan_chain(32, &parts, &[full, roomy, other], true, None).expect("a chain");
        assert!(chain.iter().all(|&(i, _, _)| i != 0), "{chain:?}");
    }
}
