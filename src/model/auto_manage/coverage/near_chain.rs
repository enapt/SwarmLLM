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

use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};

use dashmap::DashMap;

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
}

/// The chain a neighbourhood plans for a model.
#[derive(Debug, Clone, PartialEq, Eq)]
struct NearChain {
    /// Each machine's layers, `[start, end)`, in chain order.
    segments: Vec<(NodeId, (u32, u32))>,
    /// Over what its machines hold already: nothing is fetched.
    complete: bool,
}

impl NearChain {
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
fn plan_chain(
    layers: u32,
    parts: &[Part],
    machines: &[Machine],
    may_fetch: bool,
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
        let reach = self.layers.min(x.saturating_add(machine.ceiling));
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

    /// The machines `manifest`'s near chain is planned over, in rendezvous
    /// order: this node and the near ones not passed over. `None` when a
    /// connected one states no ceiling — cannot tell, and unknown is never
    /// short; one that has gone takes no part.
    fn near_machines(
        &self,
        manifest: &ModelManifest,
        parts: &[Part],
        curve: (u64, u64),
    ) -> Option<Vec<Machine>> {
        let state = &self.shared_state;
        let local = state.identity.node_id().clone();
        let mut nodes = vec![local.clone()];
        nodes.extend(self.near_peers());
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
                if node == local || state.connected_node_ids.contains(&node) {
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
        machines.sort_by(|a, b| {
            b.has_layers
                .cmp(&a.has_layers)
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

    /// `manifest`'s near chain, computed now. `None`: nobody asked for it,
    /// no header here, a near machine could run it whole, the near machines
    /// could not carry it between them, or a ceiling is unknown.
    fn compute_near_chain(&self, manifest: &ModelManifest) -> Option<NearChain> {
        let layers = manifest.num_layers;
        if layers == 0 || !self.asked_for(&manifest.id) {
            return None;
        }
        let curve = self.carry_curve(&manifest.id)?;
        let parts = parts_in_order(manifest)?;
        let mut machines = self.near_machines(manifest, &parts, curve)?;
        loop {
            // Stage 3 is for a model that needs a split: a near machine that
            // could run it whole is today's rules' to fill.
            let total: u64 = machines.iter().map(|m| u64::from(m.ceiling)).sum();
            if machines.iter().any(|m| m.ceiling >= layers) || total < u64::from(layers) {
                return None;
            }
            let held = plan_chain(layers, &parts, &machines, false);
            let complete = held.is_some();
            let steps = held.or_else(|| plan_chain(layers, &parts, &machines, true))?;
            let mut named = HashSet::new();
            let mut stalled = None;
            for &(i, x, e) in &steps {
                let machine = &machines[i];
                if lacks_any(&parts, machine, x, e) {
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
            };
            tracing::debug!(
                model = %manifest.id,
                machines = machines.len(),
                segments = chain.segments.len(),
                complete,
                plan = %chain
                    .segments
                    .iter()
                    .map(|(node, (start, end))| format!("{node}:{start}-{end}"))
                    .collect::<Vec<_>>()
                    .join(" "),
                "near chain planned (Stage 3)"
            );
            return Some(chain);
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
    pub(in crate::model::auto_manage) fn copy_in_near_chain(&self, shard_id: &ShardId) -> bool {
        if shard_id.index == MMPROJ_SHARD_INDEX {
            return false;
        }
        if let Some(manifest) = self
            .shared_state
            .model_registry
            .get_manifest(&shard_id.model_id)
        {
            // Asked first: it names this node's parts anew.
            let _ = self.near_chain(&manifest);
        }
        self.near_chain_book
            .named_until
            .get(shard_id)
            .is_some_and(|until| Instant::now() < *until)
    }

    /// The models whose header this node should learn so it can plan their
    /// near chain: asked for, a part already held by a near machine — whose
    /// chain may name this one — and no cost curve here, which is read from
    /// the header. A node fetches a header only to route a model or to fetch a
    /// part of it, so a near machine holding none of the model never computed
    /// the plan naming it, never fetched, and was passed over. Only where
    /// HuggingFace can give it (`ensure_model_geometry`'s source).
    fn models_a_near_chain_needs_a_header_for(&self) -> Vec<ModelId> {
        let near = self.near_peers();
        if near.is_empty() {
            return Vec::new();
        }
        let registry = &self.shared_state.model_registry;
        let mut wanted = Vec::new();
        for manifest in registry.models() {
            if wanted.len() >= MOST_HEADERS_A_PASS {
                break;
            }
            if !self.asked_for(&manifest.id)
                || !self
                    .shared_state
                    .models
                    .hf_sources
                    .contains_key(&manifest.id)
                || self.carry_curve(&manifest.id).is_some()
            {
                continue;
            }
            let held_near = manifest
                .shards
                .iter()
                .filter(|s| s.index != MMPROJ_SHARD_INDEX)
                .any(|s| {
                    registry
                        .shard_holders(&ShardId {
                            model_id: manifest.id.clone(),
                            index: s.index,
                        })
                        .iter()
                        .any(|h| near.contains(h))
                });
            if held_near {
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
    use super::{plan_chain, Machine, NearChain, Part};
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
            plan_chain(16, &parts, &[here.clone(), lan_box.clone()], true),
            Some(vec![(0, 0, 6), (1, 6, 16)]),
            "6 + 2 parts fetched; every other two-segment chain fetches more"
        );
        assert_eq!(plan_chain(16, &parts, &[here, lan_box], false), None);
    }

    /// A segment may end inside a part: two machines able to run 12 and 20 of
    /// a 32-layer model in 8-layer parts are a chain, though no cut on a part
    /// boundary fits them — the router splits by layer, and so does the plan.
    #[test]
    fn a_split_inside_a_part_is_a_plan() {
        let parts = parts(&[8, 8, 8, 8]);
        let small = machine(1, 12, &[false; 4]);
        let large = machine(2, 20, &[false; 4]);
        let chain = plan_chain(32, &parts, &[small, large], true).expect("a chain");
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
        let chain = plan_chain(32, &parts, &[full, roomy, other], true).expect("a chain");
        assert!(chain.iter().all(|&(i, _, _)| i != 0), "{chain:?}");
    }
}
