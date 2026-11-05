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
//! model somebody asked for, that no near machine's card holds whole and the
//! near machines' ceilings could carry between them, each near machine computes
//! the same chain over the near machines — each used once, for one contiguous
//! run of at most its ceiling — that is CHEAPEST a token by the speeds the
//! machines state (#356; within one segment of the fewest the ceilings, or the
//! cards, allow), then fetches the fewest parts, ties broken by rendezvous
//! order, and fetches the parts of ITS segment it lacks. Prune keeps them:
//! [`AutoShardManager::copy_in_near_chain`] is asked by `copy_kept_because`,
//! which the download pass asks before every fetch (gotcha #795).
//!
//! **Cards first** (#356, 2026-10-26). A machine's ceiling counts the memory
//! beside its card, whose layers run on its processor — several times slower a
//! layer (Llama 3.3 70B here: 9 of this node's 21 layers on the card, 12 on
//! the processor). The price counts each hop — a near segment the reach's
//! round trip; a far chain as the router prices a ring its head leads, half
//! the published coordinates' round trip a hop, the tail's back to the head
//! included ([`Hops`], #318) — and each layer at the speed of the device it
//! runs on, the card taking
//! a machine's first layers as the loader does; so where the cards can carry
//! the model, a chain on cards alone wins once the processor layers it saves
//! outweigh its extra hops, and where they cannot, the processor fills the
//! gap. Machines stating no speeds price every layer alike, and the cheapest
//! chain is then the fewest segments, as before. Petals sizes a server by its
//! accelerator's memory alone (`server.py::_choose_num_blocks`); Parallax and
//! Helix place on GPU memory only — none of them has a processor share to
//! weigh, so the trade against hops is ours.
//!
//! - **In-flight downloads count as held**, so the chain a fetch starts under is
//!   the chain it lands under: the chosen chain's cost falls by everything this
//!   machine fetches, any other chain's by no more.
//! - **A chain already complete is rebalanced only onto cards**: while one
//!   exists over what is held it is kept, unless a chain running fewer layers
//!   on processors costs less and its slowest machine takes at most three
//!   quarters of the held chain's slowest's time ([`worth_moving`], Petals'
//!   `balance_quality` 0.75, which is of the bottleneck too) — fewer hops
//!   alone never move it.
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

use swarmllm_types::netcoord::NetworkCoord;

use crate::inference::process_pool::{ModelWeights, RangeWeights};
use crate::types::{ModelId, ModelManifest, NodeId, ShardId, MMPROJ_SHARD_INDEX};

use super::super::AutoShardManager;
use super::{rendezvous, CARRIER_PATIENCE, PASSED_OVER_FOR};

/// A machine is near when its measured round trip is at most this. At 18 ms
/// the network was 17% of a token (regional_pipelines.md § "The premise is
/// validated"); a chain of three near machines led by its head pays three legs
/// a pass.
const NEAR_RTT_MS: f32 = 30.0;

/// What one more segment of a near chain costs a token, in ms: the reach's
/// round trip — a led chain pays a leg a segment, and the visit's fixed cost
/// beside it, so the bound overstates neither: at most [`NEAR_RTT_MS`].
const NEAR_HOP_MS: u64 = NEAR_RTT_MS as u64;

/// What a far hop costs a token, in ms, where either machine publishes no
/// usable coordinate ([`Hops::Ring`]): the 70B's far holders measured 300-410
/// ms apart (#346), the low end — every far segment's price before hops were
/// read off the coordinates (#318), so a chain of such machines costs what it
/// did.
const FAR_HOP_MS: u64 = 300;

/// The most cards a chain's search takes in where the cards cannot carry the
/// model between them (#318) — a bound on the search's size, which grows
/// with every segment it may use.
const MOST_CARD_SEGMENTS: usize = 3;

/// A complete chain moves to one whose slowest machine takes at most this
/// fraction of its slowest's time a token (numerator, denominator): Petals
/// moves a server only where the swarm's bottleneck throughput before the
/// move is below 0.75 of it after (`block_selection.py::
/// should_choose_other_blocks`: `throughputs.min()` before over after, read
/// 2026-11-05) — the bottleneck, which a stream's pipelined chunks run at,
/// never a sum with every hop in it (#318) — and here a move is a download
/// of parts, not a reload.
const MOVE_AT_MOST: (u64, u64) = (3, 4);

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
    /// What it costs a token, by the machines' stated speeds (#356).
    price: ChainPrice,
}

/// Which machines a chain is planned over.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::model::auto_manage) enum Reach {
    /// This node and the machines measured near it.
    Near,
    /// Every machine in scope — the near ones could not carry the model (#346).
    Far,
}

impl Reach {
    /// What a chain's hops cost a token at this reach.
    fn hops(self) -> Hops {
        match self {
            Reach::Near => Hops::PerSegment(1000 * NEAR_HOP_MS),
            Reach::Far => Hops::Ring {
                unknown_us: 1000 * FAR_HOP_MS,
            },
        }
    }
}

/// What a chain's hops cost a token (#356, #318).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Hops {
    /// Each segment alike, µs: the near machines' bound ([`NEAR_HOP_MS`]).
    /// Their measured round trip is what made them near, and coordinates
    /// predict a near pair poorly (60 ms for a 1 ms LAN link,
    /// `parallax::led_hop_ms`).
    PerSegment(u64),
    /// A ring its head leads, as the router prices one: each hop half the
    /// round trip the two machines' published coordinates predict
    /// (`scheduler::hop_one_way_ms`), the tail's back to the head included;
    /// a lone machine pays none — a whole-model hand-off pays the network once
    /// a request, not a token. `unknown_us` a hop where either machine
    /// publishes no usable coordinate. Every machine reads the same published
    /// coordinates, so every machine still plans the same chain. A flat price
    /// a segment (300 ms) could not tell a US card 236 ms from here from an
    /// Italian processor 371 ms away, and charged the Australian machine
    /// holding a 70B whole a hop it never pays (#318).
    Ring { unknown_us: u64 },
}

/// What a chain costs a token, by the speeds its machines state (#356).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
struct ChainPrice {
    /// Every segment's hop and every layer on its device, µs.
    total_us: u64,
    /// The layers that run on a processor beside a card, or on a machine
    /// without one.
    processor_layers: u32,
    /// The most one machine's layers take a token, µs, hops not counted —
    /// the stage a stream of guesses runs at.
    slowest_us: u64,
}

/// May a complete chain priced `held` give way to `best`, which needs parts
/// fetched? Only onto cards — fewer processor layers — only where it costs
/// less a token, and only where its slowest machine takes at most
/// [`MOVE_AT_MOST`] of the held chain's slowest: fewer hops alone never move a
/// chain that works, and a small gain is not worth the downloads (#356). The
/// fraction is of the bottleneck, as Petals' is (#318): a 70B's third card
/// halved its slowest machine's time and was refused, at 77% of a sum whose
/// round trips both chains pay.
fn worth_moving(held: ChainPrice, best: ChainPrice) -> bool {
    best.processor_layers < held.processor_layers
        && best.total_us < held.total_us
        && best.slowest_us.saturating_mul(MOVE_AT_MOST.1)
            <= held.slowest_us.saturating_mul(MOVE_AT_MOST.0)
}

/// A chain [`plan_chain`] chose, and its price.
#[derive(Debug, Clone, PartialEq, Eq)]
struct ChosenChain {
    steps: Vec<Step>,
    price: ChainPrice,
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
    /// A card among them holds the model whole, or one machine running it
    /// whole costs no more than any chain: carrying's to fill.
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
    /// Of those, the layers its card holds — the first of any run it is given,
    /// as its loader places them. `None`: its models run on its processor.
    card: Option<u32>,
    /// One layer a token on its card, µs, by the speed it states.
    card_us: u64,
    /// One layer a token on its processor, µs, by the speed it states.
    processor_us: u64,
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
    /// Where it publishes itself in the latency space, when usable — what a
    /// far chain's hops are priced by ([`Hops::Ring`]).
    coord: Option<NetworkCoord>,
}

impl Machine {
    /// `node:ceiling/card/card_us/processor_us/has/disk_mb/coord` — what the
    /// plan weighs it by, for the log.
    fn describe(&self) -> String {
        format!(
            "{}:ceiling={},card={},card_us={},processor_us={},has={},disk_mb={},coord={}",
            self.node,
            self.ceiling,
            self.card
                .map_or_else(|| "none".to_string(), |c| c.to_string()),
            self.card_us,
            self.processor_us,
            self.has_layers,
            self.disk_bytes / (1024 * 1024),
            if self.coord.is_some() { "yes" } else { "none" },
        )
    }
}

/// `node:start-end` for each step of `chain` over `machines` — for the log.
fn describe_steps(chain: &ChosenChain, machines: &[Machine]) -> String {
    chain
        .steps
        .iter()
        .map(|&(i, start, end)| format!("{}:{start}-{end}", machines[i].node))
        .collect::<Vec<_>>()
        .join(" ")
}

/// What a machine's devices give a chain of one model (#356).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct Devices {
    /// The layers its card holds. `None`: its models run on its processor.
    card: Option<u32>,
    /// One layer a token on its card, µs.
    card_us: u64,
    /// One layer a token on its processor, µs.
    processor_us: u64,
}

impl Devices {
    /// From what a machine advertises: a card only where its models run on
    /// it (`NodeCapability::models_run_on_card`), holding what the card's
    /// memory takes on the model's card curve; the card at its stated speed
    /// (`est_tokens_per_sec_7b`), its processor at the speed it states beside
    /// the card (`processor_tokens_per_sec_7b`), or — a machine without a card
    /// — at its stated speed. Each at `scheduler::stated_layer_ms`, the
    /// routers' own conversion; a speed it does not state is priced as the
    /// routers price one (`UNKNOWN_COMPUTE_MS`).
    fn of(
        c: &crate::types::NodeCapability,
        num_layers: u32,
        card_curve: Option<(u64, u64)>,
        layer_work: f32,
    ) -> Self {
        let us = |tokens_per_sec_7b: f32| -> u64 {
            let stated = if tokens_per_sec_7b.is_finite() {
                tokens_per_sec_7b
            } else {
                0.0
            };
            (crate::inference::scheduler::stated_layer_ms(stated, layer_work) * 1000.0) as u64
        };
        if !c.models_run_on_card() {
            let processor_us = us(c.est_tokens_per_sec_7b);
            return Devices {
                card: None,
                card_us: processor_us,
                processor_us,
            };
        }
        let card = c
            .gpu
            .as_ref()
            .zip(card_curve)
            .and_then(|(gpu, (fixed_mb, per_layer_mb))| {
                crate::inference::process_pool::layers_that_fit(
                    gpu.vram_total_mb,
                    fixed_mb,
                    per_layer_mb,
                )
            })
            .map(|layers| layers.min(num_layers));
        Devices {
            card: Some(card.unwrap_or(0)),
            card_us: us(c.est_tokens_per_sec_7b),
            processor_us: us(c.processor_tokens_per_sec_7b.unwrap_or(0.0)),
        }
    }
}

/// A segment: the machine (its position in the plan's list) and its layers.
type Step = (usize, u32, u32);

/// The best way on from a point: (price µs, parts fetched, the first step).
type Best = (u64, u32, Step);

/// The model's parts in layer order — when they tile `[0, num_layers)` with
/// whole layers, as every manifest `shard_layout` builds does, maybe with a
/// part past the last layer (a file's next-token-prediction blocks beside the
/// output head, `part_serves_layers`). Anything else is not planned for.
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
    (!parts.is_empty() && at >= manifest.num_layers).then_some(parts)
}

/// The parts a machine running `segment` needs: those its layers touch — the
/// first part carries the embeddings, the last the output head
/// (`part_serves_layers`: a part past the last layer too).
fn parts_touching(manifest: &ModelManifest, segment: (u32, u32)) -> Vec<u32> {
    manifest
        .shards
        .iter()
        .filter(|s| {
            s.index != MMPROJ_SHARD_INDEX
                && crate::inference::split::part_serves_layers(
                    s.layer_range,
                    segment.0,
                    segment.1,
                    manifest.num_layers,
                )
        })
        .map(|s| s.index)
        .collect()
}

/// The chain over `machines` (in rendezvous order) covering `[0, layers)`,
/// each machine used once for one run of at most its ceiling: the cheapest a
/// token (#356) — each hop as `hops` prices it, each layer at the speed of the
/// device it runs on, a machine's card taking the first of its run — then the fewest
/// parts to fetch; of equals, the first in machine order, each machine's
/// furthest end first. Where every machine prices every layer alike, the
/// cheapest is the fewest segments, as it was before prices. `may_fetch:
/// false` plans only over what the machines have. `None`: no such chain.
///
/// `weights` weighs each run as what it loads against the average layer the
/// ceilings count in (`process_pool::RangeWeights`): a run ending the model
/// carries its output head, one starting it its token table — 822 MB and
/// 564 MB of a 70B, a layer and a half and half of one at its average
/// (field report #039). `None` (no header here) counts every run as its
/// width, as before. `hops` prices each hop ([`Reach::hops`]).
fn plan_chain(
    layers: u32,
    parts: &[Part],
    machines: &[Machine],
    may_fetch: bool,
    weights: Option<&RangeWeights>,
    hops: Hops,
) -> Option<ChosenChain> {
    if machines.is_empty() || machines.len() > 32 || parts.is_empty() {
        return None;
    }
    let max_segments = if may_fetch {
        // The fewest machines whose figures sum to the model.
        let fewest = |mut ceilings: Vec<u32>| -> Option<usize> {
            ceilings.sort_unstable_by(|a, b| b.cmp(a));
            let mut total = 0u32;
            ceilings
                .iter()
                .position(|&c| {
                    total = total.saturating_add(c);
                    total >= layers
                })
                .map(|p| p + 1)
        };
        let all = fewest(machines.iter().map(|m| m.ceiling).collect())?;
        // Within reach of the search too: the fewest the CARDS allow, where
        // they can carry it between them (#356) — and where they cannot,
        // every card (up to MOST_CARD_SEGMENTS), the processors filling the
        // rest (#318: a 70B whose three cards hold 59 of its 80 layers was
        // searched over two segments, one machine carrying it whole, and its
        // third card was never tried).
        let cards: Vec<u32> = machines
            .iter()
            .filter_map(|m| m.card)
            .filter(|&c| c > 0)
            .collect();
        let card_count = cards.len();
        let cards = fewest(cards).unwrap_or(card_count.min(MOST_CARD_SEGMENTS));
        (all.max(cards) + EXTRA_SEGMENTS).min(machines.len())
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
        hops,
        memo: HashMap::new(),
    };
    let steps = planner.chain()?;
    let price = planner.price_of(&steps);
    Some(ChosenChain { steps, price })
}

/// Where the search stands: (position, machines used, the machine that ran
/// last, the chain's head) — the last two `u8::MAX` where they do not change
/// a price ([`Planner::key`]).
type State = (u32, u32, u8, u8);

/// The search behind [`plan_chain`]: exact over (position, machines used) —
/// and, for a ring, the machine that ran last and the head, which price its
/// hops.
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
    /// What each hop costs a token.
    hops: Hops,
    /// [`State`] → the best cover of the rest: its price, parts fetched, and
    /// its first step.
    memo: HashMap<State, Option<Best>>,
}

impl Planner<'_> {
    /// The part holding `layer`.
    fn part_at(&self, layer: u32) -> usize {
        self.parts.partition_point(|p| p.end <= layer)
    }

    /// One past the last part a run ending at `e` reads: a run ending the
    /// model reads every part past its last layer too (the output head beside
    /// a file's next-token-prediction blocks, `part_serves_layers`).
    fn parts_end(&self, e: u32) -> usize {
        if e >= self.layers {
            self.parts.len()
        } else {
            self.part_at(e - 1) + 1
        }
    }

    /// Where machine `i` may end a segment it starts at `x`, furthest first.
    /// Its fetch changes only where the segment reaches a new part, and its
    /// price only where its card's layers end (`card_end`, #356): past that
    /// point its layers run on its processor, and the rest of the chain may be
    /// cheaper on another card. So these are the only ends worth trying: its
    /// ceiling's reach, its card's end, and just short of each part in
    /// between. Without fetching: the end of the run it has, within its reach,
    /// and its card's end within that.
    fn ends(&self, i: usize, x: u32, card_end: Option<u32>) -> Vec<u32> {
        let machine = &self.machines[i];
        let reach = self.reach(i, x);
        if reach <= x {
            return Vec::new();
        }
        let card_end = card_end.filter(|&c| c > x && c < reach);
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
            let mut end = reach.min(self.parts[last].end);
            // The model's end only beside every part past it.
            if end >= self.layers && !machine.has[last + 1..].iter().all(|&h| h) {
                end = self.layers - 1;
                if end <= x {
                    return Vec::new();
                }
            }
            let mut ends = vec![end];
            ends.extend(card_end.filter(|&c| c < end));
            return ends;
        }
        let mut ends = vec![reach];
        ends.extend(card_end);
        ends.extend(
            self.parts[first + 1..]
                .iter()
                .map(|p| p.start)
                .take_while(|&start| start < reach),
        );
        ends.sort_unstable_by(|a, b| b.cmp(a));
        ends.dedup();
        ends
    }

    /// The furthest a run from `x` on machine `i` may end: its ceiling's worth
    /// of layers, each run weighed as what it loads where the header is here
    /// (`RangeWeights::width_in_bound`) — a run ending the model is shorter by
    /// its output head. `x` when not one layer fits.
    fn reach(&self, i: usize, x: u32) -> u32 {
        self.reach_within(x, self.machines[i].ceiling)
    }

    /// Where the layers machine `i`'s card holds end, on a run from `x`:
    /// weighed as [`Self::reach`] weighs its ceiling. `None`: no card.
    fn card_end(&self, i: usize, x: u32) -> Option<u32> {
        self.machines[i].card.map(|card| self.reach_within(x, card))
    }

    /// The furthest a run from `x` may end within `ceiling` layers.
    fn reach_within(&self, x: u32, ceiling: u32) -> u32 {
        let Some(weights) = self.weights else {
            return self.layers.min(x.saturating_add(ceiling));
        };
        // A middle run weighs less than its average layers, so it may reach a
        // little past the ceiling's count — never twice as far.
        let furthest = self.layers.min(x.saturating_add(ceiling.saturating_mul(2)));
        (x + 1..=furthest)
            .rev()
            .find(|&e| weights.width_in_bound((x, e), &[]) <= ceiling)
            .unwrap_or(x)
    }

    /// Parts machine `i` lacks among those `[x, e)` touches, and their bytes.
    fn lacks(&self, i: usize, x: u32, e: u32) -> (u32, u64) {
        let (lo, hi) = (self.part_at(x), self.parts_end(e));
        let (count, bytes) = &self.lacking[i];
        (count[hi] - count[lo], bytes[hi] - bytes[lo])
    }

    /// Of `[x, e)` on a machine whose card's layers end at `card_end`: the
    /// layers on its card, and those on its processor.
    fn by_device(x: u32, e: u32, card_end: Option<u32>) -> (u32, u32) {
        let on_card = card_end.map_or(0, |c| c.min(e).saturating_sub(x));
        (on_card, (e - x) - on_card)
    }

    /// What machine `i` running `[x, e)` costs a token on its devices, µs:
    /// each layer on the device it runs on.
    fn compute_us(&self, i: usize, x: u32, e: u32, card_end: Option<u32>) -> u64 {
        let machine = &self.machines[i];
        let (on_card, beside) = Self::by_device(x, e, card_end);
        u64::from(on_card)
            .saturating_mul(machine.card_us)
            .saturating_add(u64::from(beside).saturating_mul(machine.processor_us))
    }

    /// What the hop into machine `to` costs a token, µs, after machine `from`
    /// — `None`: `to` heads the chain.
    fn hop_into_us(&self, from: Option<usize>, to: usize) -> u64 {
        match self.hops {
            Hops::PerSegment(us) => us,
            Hops::Ring { unknown_us } => match from {
                None => 0,
                Some(from) if from == to => 0,
                Some(from) => match (self.machines[from].coord, self.machines[to].coord) {
                    (Some(a), Some(b)) => {
                        (crate::inference::scheduler::hop_one_way_ms(&a, &b) * 1000.0) as u64
                    }
                    _ => unknown_us,
                },
            },
        }
    }

    /// What closing a chain from `tail` back to `head` costs a token, µs: a
    /// ring's last hop; nothing where each segment pays its own.
    fn closing_us(&self, tail: usize, head: usize) -> u64 {
        match self.hops {
            Hops::PerSegment(_) => 0,
            Hops::Ring { .. } => self.hop_into_us(Some(tail), head),
        }
    }

    /// The memo's key: where the chain stands. Only a ring's price depends on
    /// the machine that ran last and the head, so a chain priced a hop a
    /// segment keeps the search it had.
    fn key(&self, x: u32, used: u32, prev: Option<usize>, head: Option<usize>) -> State {
        let index = |m: Option<usize>| m.map_or(u8::MAX, |m| m as u8);
        match self.hops {
            Hops::PerSegment(_) => (x, used, u8::MAX, u8::MAX),
            Hops::Ring { .. } => (x, used, index(prev), index(head)),
        }
    }

    /// What `steps` cost a token.
    fn price_of(&self, steps: &[Step]) -> ChainPrice {
        let mut price = ChainPrice::default();
        let mut prev = None;
        for &(i, x, e) in steps {
            let card_end = self.card_end(i, x);
            let (_, beside) = Self::by_device(x, e, card_end);
            let compute_us = self.compute_us(i, x, e, card_end);
            price.total_us = price
                .total_us
                .saturating_add(self.hop_into_us(prev, i))
                .saturating_add(compute_us);
            price.processor_layers += beside;
            price.slowest_us = price.slowest_us.max(compute_us);
            prev = Some(i);
        }
        if let (Some(&(head, _, _)), Some(tail)) = (steps.first(), prev) {
            price.total_us = price.total_us.saturating_add(self.closing_us(tail, head));
        }
        price
    }

    /// The best cover of `[x, layers)` by machines not in `used`, after
    /// machine `prev` of a chain `head` leads (both `None` at the start):
    /// (price µs, parts fetched).
    fn best(
        &mut self,
        x: u32,
        used: u32,
        prev: Option<usize>,
        head: Option<usize>,
    ) -> Option<(u64, u32)> {
        if x >= self.layers {
            let closing = prev
                .zip(head)
                .map_or(0, |(tail, head)| self.closing_us(tail, head));
            return Some((closing, 0));
        }
        if used.count_ones() >= self.max_segments {
            return None;
        }
        let key = self.key(x, used, prev, head);
        if let Some(known) = self.memo.get(&key) {
            return known.map(|(price, fetched, _)| (price, fetched));
        }
        let mut best: Option<Best> = None;
        for i in 0..self.machines.len() {
            if used & (1 << i) != 0 {
                continue;
            }
            let card_end = self.card_end(i, x);
            for e in self.ends(i, x, card_end) {
                let (fetch, bytes) = self.lacks(i, x, e);
                if bytes > 0 && bytes >= self.machines[i].disk_bytes {
                    continue;
                }
                let Some((price, fetched)) =
                    self.best(e, used | (1 << i), Some(i), head.or(Some(i)))
                else {
                    continue;
                };
                let candidate = (
                    price
                        .saturating_add(self.hop_into_us(prev, i))
                        .saturating_add(self.compute_us(i, x, e, card_end)),
                    fetched + fetch,
                );
                if best.is_none_or(|(p, f, _)| candidate < (p, f)) {
                    best = Some((candidate.0, candidate.1, (i, x, e)));
                }
            }
        }
        self.memo.insert(key, best);
        best.map(|(price, fetched, _)| (price, fetched))
    }

    /// The steps of the best cover of the whole model.
    fn chain(&mut self) -> Option<Vec<Step>> {
        self.best(0, 0, None, None)?;
        let (mut x, mut used, mut steps) = (0u32, 0u32, Vec::new());
        let (mut prev, mut head) = (None, None);
        while x < self.layers {
            let key = self.key(x, used, prev, head);
            let (_, _, (i, _, e)) = self.memo.get(&key).copied().flatten()?;
            steps.push((i, x, e));
            x = e;
            used |= 1 << i;
            prev = Some(i);
            head = head.or(Some(i));
        }
        Some(steps)
    }
}

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

/// Does `machine` lack any part `[x, e)` reads — a part past the last of a
/// `layers`-layer model's layers too, where the run ends it
/// (`part_serves_layers`)?
fn lacks_any(parts: &[Part], machine: &Machine, x: u32, e: u32, layers: u32) -> bool {
    parts.iter().zip(&machine.has).any(|(p, &has)| {
        !has && crate::inference::split::part_serves_layers((p.start, p.end), x, e, layers)
    })
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

    /// What `node`'s devices give a chain of a model (#356), from the
    /// capability it advertises — this node's own last broadcast for itself,
    /// as [`Self::advertised`] reads it, so every machine prices every other
    /// alike. `None`: it advertises none.
    fn advertised_devices(
        &self,
        node: &NodeId,
        num_layers: u32,
        card_curve: Option<(u64, u64)>,
        layer_work: f32,
    ) -> Option<Devices> {
        let devices =
            |c: &crate::types::NodeCapability| Devices::of(c, num_layers, card_curve, layer_work);
        if node == self.shared_state.identity.node_id() {
            return self
                .shared_state
                .local_capability
                .load_full()
                .map(|c| devices(&c));
        }
        self.shared_state
            .peer_registry
            .get(node)
            .and_then(|p| p.capability.as_ref().map(devices))
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
        // Each machine's card and speeds price the chain (#356): the card's
        // layers by the model's card curve, each layer by what it weighs.
        let card_curve = state
            .model_process_pool
            .segment_cost_curve(&manifest.id, true);
        let layer_work = crate::inference::scheduler::layer_work_for(state, &manifest.id);
        let mut nodes = match reach {
            Reach::Near => {
                let mut nodes = vec![local.clone()];
                nodes.extend(self.near_peers());
                nodes
            }
            Reach::Far => self.machines_in_scope(),
        };
        // Every machine left out is named with its reason at debug — the
        // planner's choice cannot otherwise be read from outside (#318: a
        // card the 70B's chain never used, for reasons nothing logged). The
        // scheduler's pattern: kube-scheduler keeps a status per filtered
        // node (`framework.Diagnosis`, "0/5 nodes are available: …").
        let mut passed_over = Vec::new();
        nodes.retain(|n| {
            let keep = !self.passed_over_for_chain(&manifest.id, n);
            if !keep {
                passed_over.push(n.to_string());
            }
            keep
        });
        let mut no_ceiling = Vec::new();
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
                no_ceiling.push(node.to_string());
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
            // What it may be handed to fetch: none where it does not fetch
            // parts (#400) — it still serves the segment it holds.
            let disk_bytes = self.room_to_fetch_bytes(&node);
            let devices = self
                .advertised_devices(&node, manifest.num_layers, card_curve, layer_work)
                .unwrap_or_default();
            // Its published coordinate — this node's own last broadcast for
            // itself — so every machine prices every hop alike (#318).
            let coord = self
                .advertised(&node, |c| c.coord)
                .flatten()
                .filter(NetworkCoord::is_usable);
            machines.push(Machine {
                node,
                ceiling,
                card: devices.card.map(|card| card.min(ceiling)),
                card_us: devices.card_us,
                processor_us: devices.processor_us,
                has,
                held_layers,
                has_layers,
                fetching: fetching_now,
                disk_bytes,
                coord,
            });
        }
        let rank = |m: &Machine| rendezvous(&manifest.id, &m.node);
        // A machine that could run the model whole survives the cut whatever
        // it holds: it may make the model carrying's (`plan_over`, where it
        // is the cheapest plan), and ten
        // holders of a part each must not hide it (v0.3.254's review).
        let whole = |m: &Machine| m.ceiling >= manifest.num_layers;
        machines.sort_by(|a, b| {
            whole(b)
                .cmp(&whole(a))
                .then(b.has_layers.cmp(&a.has_layers))
                .then(b.ceiling.cmp(&a.ceiling))
                .then_with(|| rank(b).cmp(&rank(a)))
        });
        let cut: Vec<String> = machines
            .iter()
            .skip(MOST_MACHINES)
            .map(|m| m.node.to_string())
            .collect();
        machines.truncate(MOST_MACHINES);
        machines.sort_by_key(|m| std::cmp::Reverse(rank(m)));
        tracing::debug!(
            model = %manifest.id,
            reach = ?reach,
            weighed = %machines.iter().map(Machine::describe).collect::<Vec<_>>().join(" "),
            passed_over = %passed_over.join(","),
            no_ceiling = %no_ceiling.join(","),
            cut = %cut.join(","),
            "the machines a chain is planned over, and those left out (Stage 3, #318)"
        );
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

    /// May this node plan `model_id`'s near chain — is it asked for? Any model
    /// may be (#389 (c): nothing automatic judges content); a model nobody runs
    /// is not asked for (`unused::nobody_runs`). The chain and the header ask
    /// this.
    fn may_plan_a_near_chain(&self, model_id: &ModelId) -> bool {
        self.asked_for(model_id)
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
            // A chain is for a model that needs a split: a card that holds
            // it whole is today's rules' to fill. A machine that runs it whole
            // only with its processor's help competes with the chains of cards
            // (#356): carrying's, below, where no chain is cheaper.
            let total: u64 = machines.iter().map(|m| u64::from(m.ceiling)).sum();
            if machines
                .iter()
                .any(|m| m.card.is_some_and(|card| card >= layers))
            {
                return Some(Planned::OneRunsItWhole);
            }
            if total < u64::from(layers) {
                return Some(Planned::CannotCarry);
            }
            let hops = reach.hops();
            let held = plan_chain(layers, parts, &machines, false, weights.as_ref(), hops);
            let best = plan_chain(layers, parts, &machines, true, weights.as_ref(), hops);
            // One machine running it whole is carrying's where no chain costs
            // less — priced whatever its disk, as carrying chooses who
            // fetches. Machines stating no speeds: wherever one can, as
            // before prices.
            let chain_us = [held.as_ref(), best.as_ref()]
                .into_iter()
                .flatten()
                .map(|c| c.price.total_us)
                .min();
            let whole_us = machines
                .iter()
                .filter(|m| m.ceiling >= layers)
                .filter_map(|m| {
                    let alone = Machine {
                        disk_bytes: u64::MAX,
                        ..m.clone()
                    };
                    plan_chain(layers, parts, &[alone], true, weights.as_ref(), hops)
                })
                .map(|c| c.price.total_us)
                .min();
            if whole_us.is_some_and(|w| chain_us.is_none_or(|c| w <= c)) {
                return Some(Planned::OneRunsItWhole);
            }
            // Both chains and the verdict, whichever way it goes: a chain that
            // stays where it is says why as plainly as one that moves (#318).
            let moving = matches!(
                (&held, &best),
                (Some(h), Some(b)) if worth_moving(h.price, b.price)
            );
            let ms = |c: &Option<ChosenChain>| c.as_ref().map(|c| c.price.total_us / 1000);
            let on_processors =
                |c: &Option<ChosenChain>| c.as_ref().map(|c| c.price.processor_layers);
            let slowest_ms =
                |c: &Option<ChosenChain>| c.as_ref().map(|c| c.price.slowest_us / 1000);
            let plan = |c: &Option<ChosenChain>| {
                c.as_ref()
                    .map_or_else(|| "none".to_string(), |c| describe_steps(c, &machines))
            };
            tracing::debug!(
                model = %manifest.id,
                reach = ?reach,
                held_ms = ?ms(&held),
                held_slowest_ms = ?slowest_ms(&held),
                held_processor_layers = ?on_processors(&held),
                held_plan = %plan(&held),
                best_ms = ?ms(&best),
                best_slowest_ms = ?slowest_ms(&best),
                best_processor_layers = ?on_processors(&best),
                best_plan = %plan(&best),
                moving,
                "a chain over what is held, against the cheapest with fetches — a complete \
                 one moves only onto cards, for less, its slowest machine at most 3/4 as slow \
                 (#356, #318)"
            );
            let (chosen, complete) = match (held, best) {
                (Some(_), Some(best)) if moving => (best, false),
                (Some(held), _) => (held, true),
                (None, Some(best)) => (best, false),
                (None, None) => return Some(Planned::CannotCarry),
            };
            let steps = chosen.steps;
            let mut named = HashSet::new();
            let mut stalled = None;
            for &(i, x, e) in &steps {
                let machine = &machines[i];
                if lacks_any(parts, machine, x, e, manifest.num_layers) {
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
                price: chosen.price,
            };
            tracing::debug!(
                model = %manifest.id,
                machines = machines.len(),
                segments = chain.segments.len(),
                complete,
                reach = ?reach,
                price_ms = chain.price.total_us / 1000,
                processor_layers = chain.price.processor_layers,
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
                price_ms = planned.price.total_us / 1000,
                processor_layers = planned.price.processor_layers,
                plan = %planned.describe(),
                "DIAG: a chain of machines is planned for a model no card among them holds whole \
                 (Stage 3; cards first, #356; over every machine in scope where the near ones \
                 cannot carry it, #346)"
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
            .map(|node| self.advertised_ceiling_mb(node))
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
    use super::{
        plan_chain, worth_moving, ChainPrice, Machine, NearChain, NetworkCoord, Part, RangeWeights,
        Reach, Step,
    };
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
        swarm_named("near-me", near_rtt_ms)
    }

    /// [`swarm`] with the model named `name`.
    fn swarm_named(name: &str, near_rtt_ms: f32) -> Swarm {
        let mut config = Config::default();
        config.auto_manage.min_replicas = 1;
        let (state, manager) = make_test_manager_with_config(config);
        state.models.auto_manage_default_model_cap.store(0, Relaxed);
        let ranges: Vec<(u32, u32)> = (0..8).map(|i| (i * 4, i * 4 + 4)).collect();
        let mid = register_manifest_with_sized_shards(&state, name, 32, &ranges, PART);
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
        fleet_named("near-me", near_rtt_ms)
    }

    /// [`fleet`] with the model named `name`.
    fn fleet_named(name: &str, near_rtt_ms: f32) -> Swarm {
        let s = swarm_named(name, near_rtt_ms);
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

    /// #371: a node with no storage left takes its part of the chain its
    /// machines plan. The download pass notes the room the chain's part
    /// lacks, prune gives back a copy of another model three computers hold,
    /// and the next cycle fetches the part. Before, a full node returned
    /// before it looked at the chain: its segment never came, and the chain
    /// waited on it until the node was passed over.
    #[tokio::test]
    async fn a_full_node_makes_room_for_its_part_of_the_chain() {
        use super::super::super::manager::RoomFor;
        use super::super::part_rank_weight;
        let s = fleet(4.0);
        let local = s.local();
        let mut config = Config::clone(&s.state.cfg());
        config.auto_manage.max_storage_mb = 1_000;
        config.node.contribution = crate::types::ContributionMode::Minimal;
        s.state.apply_live_config(config);
        // Two of another model's four parts fill the budget; two computers
        // 300 ms away hold all four — so the region rule, not the count, keeps
        // this node's copies — each ranked above this node on both parts.
        let ranges = [(0, 4), (4, 8), (8, 12), (12, 16)];
        let copies = register_manifest_with_sized_shards(&s.state, "copies", 16, &ranges, PART);
        holds(&s.state, &copies, &local, &[0, 1]);
        write_sparse_shards(&s.state, &copies, [0, 1], PART);
        let part = |index| ShardId {
            model_id: copies.clone(),
            index,
        };
        let above: Vec<NodeId> = (0u32..1_000_000)
            .map(|i| NodeId(*blake3::hash(&i.to_le_bytes()).as_bytes()))
            .filter(|n| {
                (0..2).all(|i| part_rank_weight(&part(i), n) > part_rank_weight(&part(i), &local))
            })
            .take(2)
            .collect();
        for id in &above {
            let mut info = crate::types::PeerInfo {
                node_id: id.clone(),
                ..s.state.peer_registry.get(&s.far).unwrap().clone()
            };
            info.latency_ms = Some(300);
            info.capability = None;
            s.state.peer_registry.insert(id.clone(), info);
            s.state.connected_node_ids.insert(id.clone());
            holds(&s.state, &copies, id, &[0, 1, 2, 3]);
        }
        let parts_of_copies_here = || {
            (0..4)
                .filter(|&i| s.state.shard_store().shard_path(&copies, i).exists())
                .collect::<Vec<u32>>()
        };
        assert_eq!(
            s.manager.remaining_budget(&local).remaining_bytes(),
            0,
            "fixture: no storage left"
        );

        s.later();
        s.manager.evaluate_and_download().await;
        let room = s
            .manager
            .room_wanted_now()
            .expect("this node's part of the chain asks for room");
        assert_eq!(
            (room.why, room.model, room.bytes),
            (RoomFor::Placed, s.mid.clone(), PART)
        );
        s.manager.evaluate_and_prune().await;
        assert_eq!(
            parts_of_copies_here().len(),
            1,
            "one copy given back, for the room the part lacks and no more"
        );

        s.later();
        let report = s.manager.remaining_budget(&local);
        let next: Vec<(ModelId, u32)> = s
            .manager
            .select_within_budget(s.manager.gather_candidates(&local, 0), &report)
            .into_iter()
            .map(|c| (c.model_id, c.shard_index))
            .collect();
        assert_eq!(next.len(), 1, "one part a cycle at this contribution");
        assert!(
            next[0].0 == s.mid && next[0].1 <= 2,
            "the room goes to this node's segment (parts 0-2): {next:?}"
        );
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
    }

    /// A model with its safety training removed is planned a near chain like
    /// any other once it is asked for (#389 (c)): nothing automatic judges
    /// content. Until 2026-11-01 one the owner had not chosen was planned none.
    /// Marked by its NAME: an origin entry would mark it too, but leaves its
    /// upload unvalidated, and `canonical_allows_acquisition` then refuses
    /// every fetch whatever the content rule says.
    #[test]
    fn an_uncensored_model_is_planned_a_near_chain_like_any_other() {
        let s = fleet_named("near-me-abliterated", 4.0);
        assert!(
            s.state.unchosen_uncensored_model(&s.mid),
            "fixture: marked, not chosen"
        );
        assert!(s.state.canonical_allows_acquisition(&s.mid));
        let chain = s.chain().expect("a near chain");
        assert_eq!(
            chain.segments,
            vec![(s.local(), (0, 12)), (s.near.clone(), (12, 32))]
        );
        assert_eq!(s.offered(), vec![0, 1, 2]);
    }

    fn machine(byte: u8, ceiling: u32, has: &[bool]) -> Machine {
        Machine {
            node: NodeId([byte; 32]),
            ceiling,
            card: None,
            card_us: 0,
            processor_us: 0,
            has: has.to_vec(),
            held_layers: 0,
            has_layers: 0,
            fetching: false,
            disk_bytes: u64::MAX,
            coord: None,
        }
    }

    /// [`plan_chain`]'s steps at a near hop. Machines stating no speeds price
    /// a chain by its segments alone: the fewest, as before prices (#356).
    fn plan(
        layers: u32,
        parts: &[Part],
        machines: &[Machine],
        may_fetch: bool,
        weights: Option<&RangeWeights>,
    ) -> Option<Vec<Step>> {
        plan_chain(
            layers,
            parts,
            machines,
            may_fetch,
            weights,
            Reach::Near.hops(),
        )
        .map(|chosen| chosen.steps)
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
        let unweighed = plan(80, &parts, &machines, true, None).expect("a chain");
        assert!(
            unweighed
                .iter()
                .any(|&(i, x, e)| i == 1 && e == 80 && weights.width_in_bound((x, e), &[]) > 7),
            "{unweighed:?}"
        );
        let weighed = plan(80, &parts, &machines, true, Some(&weights)).expect("a chain");
        let mut at = 0;
        for &(i, x, e) in &weighed {
            assert_eq!(x, at, "{weighed:?}");
            assert!(
                weights.width_in_bound((x, e), &[]) <= machines[i].ceiling,
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
            plan(16, &parts, &[here.clone(), lan_box.clone()], true, None),
            Some(vec![(0, 0, 6), (1, 6, 16)]),
            "6 + 2 parts fetched; every other two-segment chain fetches more"
        );
        assert_eq!(plan(16, &parts, &[here, lan_box], false, None), None);
    }

    /// A segment may end inside a part: two machines able to run 12 and 20 of
    /// a 32-layer model in 8-layer parts are a chain, though no cut on a part
    /// boundary fits them — the router splits by layer, and so does the plan.
    #[test]
    fn a_split_inside_a_part_is_a_plan() {
        let parts = parts(&[8, 8, 8, 8]);
        let small = machine(1, 12, &[false; 4]);
        let large = machine(2, 20, &[false; 4]);
        let chain = plan(32, &parts, &[small, large], true, None).expect("a chain");
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
        let chain = plan(32, &parts, &[full, roomy, other], true, None).expect("a chain");
        assert!(chain.iter().all(|&(i, _, _)| i != 0), "{chain:?}");
    }

    /// A machine with a card holding `card` of its `ceiling` layers: a layer on
    /// the card 2 ms a token, on its processor `processor_us`.
    fn carded(byte: u8, ceiling: u32, card: u32, processor_us: u64, layers: usize) -> Machine {
        Machine {
            card: Some(card),
            card_us: 2_000,
            processor_us,
            ..machine(byte, ceiling, &vec![false; layers])
        }
    }

    /// **A chain's last run reads the part past the model's last layer**
    /// (2026-11-02): Ornith-1.5-9B's ninth part is `[32, 33)` — its
    /// next-token-prediction block beside the output head. A machine holding
    /// parts 0-7 holds no complete chain (it cannot end the model), and one
    /// asked to fetch is asked for that part too; the manifest's parts are in
    /// order with it. Before, the chain was "complete" without the head, and
    /// no machine ever fetched it.
    #[test]
    fn a_chain_that_ends_the_model_reads_the_part_past_its_last_layer() {
        let mut with_tail = parts(&[4; 8]);
        with_tail.push(Part {
            index: 8,
            start: 32,
            end: 33,
            size: 100,
        });
        let mut no_head = vec![true; 8];
        no_head.push(false);
        let lacking = [machine(1, 40, &no_head)];
        assert_eq!(
            plan(32, &with_tail, &lacking, false, None),
            None,
            "parts 0-7 are not a complete chain"
        );
        assert!(super::lacks_any(&with_tail, &lacking[0], 0, 32, 32));
        assert!(
            !super::lacks_any(&with_tail, &lacking[0], 0, 31, 32),
            "a run short of the end does not read it"
        );
        let whole = [machine(1, 40, &[true; 9])];
        assert_eq!(
            plan(32, &with_tail, &whole, false, None),
            Some(vec![(0, 0, 32)])
        );
        // Asked to fetch, the machine that holds the head is the one chosen:
        // the other would fetch a part.
        let both = [machine(1, 40, &no_head), machine(2, 40, &[true; 9])];
        let chosen =
            plan_chain(32, &with_tail, &both, true, None, Reach::Near.hops()).expect("a chain");
        assert_eq!(chosen.steps, vec![(1, 0, 32)]);
        let both_reversed = [machine(2, 40, &[true; 9]), machine(1, 40, &no_head)];
        let chosen = plan_chain(
            32,
            &with_tail,
            &both_reversed,
            true,
            None,
            Reach::Near.hops(),
        )
        .expect("a chain");
        assert_eq!(chosen.steps, vec![(0, 0, 32)]);

        let manifest = ModelManifest {
            id: ModelId("ornith-1-5-9b".into()),
            name: "Ornith-1.5-9B".into(),
            architecture: crate::types::ModelArchitecture::Llama,
            num_layers: 32,
            num_params_billions: 9.0,
            quantization: crate::types::Quantization::Q4KM,
            total_size_bytes: 900,
            shard_count: 9,
            shards: with_tail
                .iter()
                .map(|p| crate::types::ShardInfo {
                    index: p.index,
                    layer_range: (p.start, p.end),
                    size_bytes: p.size,
                    hash: [0u8; 32],
                    tensors: Vec::new(),
                })
                .collect(),
            tokenizer_hash: [0u8; 32],
            manifest_hash: [0u8; 32],
            publisher: NodeId([0u8; 32]),
            publish_date: chrono::Utc::now(),
            license: "MIT".into(),
            mmproj: None,
            context_length: None,
        };
        assert_eq!(super::parts_in_order(&manifest).map(|p| p.len()), Some(9));
        assert_eq!(super::parts_touching(&manifest, (24, 32)), vec![6, 7, 8]);
        assert_eq!(
            super::parts_touching(&manifest, (0, 31)),
            (0..8).collect::<Vec<_>>()
        );
    }

    /// #356: where the cards can carry a model, its chain runs on cards alone
    /// — three near machines whose cards hold 10, 12 and 10 of a 32-layer
    /// model, though two of them could carry it with the first one's
    /// processor (20 ms a layer) running 10 layers: 3 hops of 30 ms and 32
    /// card layers (154 ms) against 2 hops and 10 processor layers (304 ms).
    /// Controls: at a far hop (300 ms) the extra hop costs more than the
    /// processor layers, and the two-machine chain wins; machines stating no
    /// speeds plan the fewest segments, as before.
    #[test]
    fn a_chain_runs_on_cards_where_the_cards_can_carry_it() {
        let parts = parts(&[1; 32]);
        let machines = [
            carded(1, 30, 10, 20_000, 32),
            carded(2, 12, 12, 20_000, 32),
            carded(3, 10, 10, 20_000, 32),
        ];
        let near =
            plan_chain(32, &parts, &machines, true, None, Reach::Near.hops()).expect("a chain");
        assert_eq!(near.steps.len(), 3, "{near:?}");
        assert_eq!(near.price.processor_layers, 0, "{near:?}");
        for &(i, x, e) in &near.steps {
            assert!(e - x <= machines[i].card.unwrap(), "{near:?}");
        }
        assert_eq!(near.price.total_us, 3 * 30_000 + 32 * 2_000);

        let far =
            plan_chain(32, &parts, &machines, true, None, Reach::Far.hops()).expect("a chain");
        assert_eq!(
            far.steps.len(),
            2,
            "control: a far hop outweighs them: {far:?}"
        );
        assert_eq!(far.price.processor_layers, 10, "{far:?}");

        let unstated: Vec<Machine> = machines
            .iter()
            .map(|m| Machine {
                card_us: 0,
                processor_us: 0,
                ..m.clone()
            })
            .collect();
        let fewest = plan(32, &parts, &unstated, true, None).expect("a chain");
        assert_eq!(
            fewest.len(),
            2,
            "control: no speeds, fewest segments: {fewest:?}"
        );
    }

    /// #356, #318: a complete chain moves only onto cards — fewer layers on
    /// processors — for less a token, and only where its slowest machine
    /// takes at most three quarters of the held chain's slowest's time.
    #[test]
    fn a_complete_chain_moves_only_onto_cards_and_only_for_a_quarter_off_its_slowest() {
        let held = ChainPrice {
            total_us: 1_000_000,
            processor_layers: 12,
            slowest_us: 800_000,
        };
        let price = |total_us, processor_layers, slowest_us| ChainPrice {
            total_us,
            processor_layers,
            slowest_us,
        };
        assert!(worth_moving(held, price(999_999, 0, 600_000)));
        assert!(worth_moving(held, price(900_000, 11, 600_000)));
        assert!(
            !worth_moving(held, price(900_000, 0, 600_001)),
            "too small a gain at the slowest machine"
        );
        assert!(
            !worth_moving(held, price(1_000_000, 0, 100_000)),
            "never for a chain that costs as much a token"
        );
        assert!(
            !worth_moving(held, price(100_000, 12, 100_000)),
            "fewer hops alone never move a chain that works"
        );
        assert!(!worth_moving(held, price(100_000, 13, 100_000)));
    }

    /// Coordinates `rtt_ms` apart on a line with the two heights at their
    /// least (1 ms each, so the planar distance is `rtt_ms - 2`), settled.
    fn coord_at(x: f32, y: f32) -> NetworkCoord {
        NetworkCoord {
            x,
            y,
            height: 1.0,
            error: 0.1,
        }
    }

    /// **#318, the live 70B on 2026-11-05**: this machine's 8 GB card holds 15
    /// of its 24-layer ceiling (it holds parts 0-28), the Australian 3060
    /// holds the whole model (22 on its card), the US 3080 Ti holds nothing
    /// (22 on its card), at the speeds each stated that day; round trips
    /// here–AU 131 ms, here–US 236, US–AU 200, from published coordinates.
    /// Priced as the router prices a ring its head leads, the chain over what
    /// is held is the route the router ran (here 0-15 + AU, not AU alone —
    /// which a flat 300 ms a segment chose), the cheapest with fetches takes
    /// the US card in for 22 layers (786 ms a token against 1018), and the
    /// move is taken: its slowest machine takes about half the time. Controls: with no coordinates every far hop
    /// costs what a segment did, and the three cards are not the cheapest.
    #[test]
    fn a_far_chain_prices_its_hops_by_the_coordinates_and_takes_in_a_third_card() {
        let parts = parts(&[1; 80]);
        let holding = |from: u32, to: u32| -> Vec<bool> {
            (0..80).map(|i| (from..to).contains(&i)).collect()
        };
        let stated = |byte, ceiling, card, card_us, processor_us, has: Vec<bool>| Machine {
            card: Some(card),
            card_us,
            processor_us,
            ..machine(byte, ceiling, &has)
        };
        // here (0, 0); AU 129 ms along x; US placed 234 from here, 198 from AU.
        let x_us = (234.0f32 * 234.0 - 198.0 * 198.0 + 129.0 * 129.0) / (2.0 * 129.0);
        let y_us = (234.0f32 * 234.0 - x_us * x_us).sqrt();
        let with_coords = [
            Machine {
                coord: Some(coord_at(0.0, 0.0)),
                ..stated(1, 24, 15, 2_126, 22_624, holding(0, 28))
            },
            Machine {
                coord: Some(coord_at(129.0, 0.0)),
                ..stated(2, 80, 22, 2_646, 18_523, holding(0, 80))
            },
            Machine {
                coord: Some(coord_at(x_us, y_us)),
                ..stated(3, 46, 22, 1_044, 21_252, holding(0, 0))
            },
        ];
        let hops = Reach::Far.hops();
        let held = plan_chain(80, &parts, &with_coords, false, None, hops).expect("held");
        assert_eq!(held.steps, vec![(0, 0, 15), (1, 15, 80)], "{held:?}");
        // 15 card layers here, 22 card + 43 processor in AU, one round trip.
        let here_us = 15 * 2_126;
        let au_us = 22 * 2_646 + 43 * 18_523;
        assert_eq!(held.price.slowest_us, au_us);
        assert!(
            held.price.total_us.abs_diff(here_us + au_us + 131_000) < 1_000,
            "{held:?}"
        );
        let alone = plan_chain(80, &parts, &with_coords[1..2], false, None, hops).expect("AU");
        assert_eq!(
            alone.price.total_us,
            22 * 2_646 + 58 * 18_523,
            "a lone machine pays no hop"
        );
        assert!(held.price.total_us < alone.price.total_us);

        let best = plan_chain(80, &parts, &with_coords, true, None, hops).expect("best");
        // A ring costs the same either way round, so which of AU and the US
        // follows this node is a tie; what matters is the US card's 22 layers.
        assert_eq!(best.steps.len(), 3, "{best:?}");
        assert_eq!(best.steps[0], (0, 0, 15), "{best:?}");
        let us: Vec<_> = best.steps.iter().filter(|s| s.0 == 2).collect();
        assert!(matches!(us[..], [&(2, x, e)] if e - x == 22), "{best:?}");
        assert_eq!(best.price.processor_layers, 21);
        // The ring here → AU → US → here (or back): 65.5 + 100 + 118 ms.
        let three_us = here_us + 22 * 1_044 + 22 * 2_646 + 21 * 18_523 + 283_500;
        assert!(best.price.total_us.abs_diff(three_us) < 1_000, "{best:?}");
        assert!(
            worth_moving(held.price, best.price),
            "held {:?}, best {:?}",
            held.price,
            best.price
        );

        // Control: no coordinates — every far hop is the old segment's 300 ms,
        // and three cards cost three of them.
        let unplaced: Vec<Machine> = with_coords
            .iter()
            .map(|m| Machine {
                coord: None,
                ..m.clone()
            })
            .collect();
        let held = plan_chain(80, &parts, &unplaced, false, None, hops).expect("held");
        assert_eq!(held.steps, vec![(1, 0, 80)], "{held:?}");
        let best = plan_chain(80, &parts, &unplaced, true, None, hops).expect("best");
        assert!(best.steps.len() < 3, "{best:?}");
    }

    /// #318: where the cards cannot carry a model between them, the search
    /// still takes every card in, the processors filling the rest. Three near
    /// cards hold 8 of a 32-layer model each and one machine without a card
    /// could carry it alone: on two segments at most (the fewest by ceiling,
    /// one more) 24 layers ran on its processor; now 8 do.
    #[test]
    fn where_the_cards_cannot_carry_a_model_every_card_may_join_its_chain() {
        let parts = parts(&[1; 32]);
        let machines = [
            carded(1, 8, 8, 20_000, 32),
            carded(2, 8, 8, 20_000, 32),
            carded(3, 8, 8, 20_000, 32),
            Machine {
                card_us: 20_000,
                processor_us: 20_000,
                ..machine(4, 32, &[false; 32])
            },
        ];
        let chain =
            plan_chain(32, &parts, &machines, true, None, Reach::Near.hops()).expect("a chain");
        assert_eq!(chain.steps.len(), 4, "{chain:?}");
        assert_eq!(chain.price.processor_layers, 8, "{chain:?}");
        assert_eq!(chain.price.total_us, 4 * 30_000 + 24 * 2_000 + 8 * 20_000);
    }

    /// [`swarm`] with three near machines, each with a card holding 12 of the
    /// model's 32 layers (card 56 tokens/s at 7B), this node and the near peer
    /// holding a complete chain between them that runs 4 layers each on its
    /// processor, and a third near machine holding nothing. `processor` is
    /// the speed each states beside its card.
    fn three_cards(processor: f32) -> (Swarm, crate::types::NodeId) {
        let s = swarm(4.0);
        let local = s.local();
        holds(&s.state, &s.mid, &local, &[0, 1, 2, 3]);
        holds(&s.state, &s.mid, &s.near, &[4, 5, 6, 7]);
        let other = peer_with_disk(&s.state, 4, Some(ceiling_for(16)), 100_000);
        measured(&s.state, &other, 6.0);
        let with_card = |c: &mut crate::types::NodeCapability| {
            c.gpu = Some(crate::types::GpuInfo {
                name: "card".to_string(),
                vram_total_mb: ceiling_for(12),
                vram_available_mb: ceiling_for(12),
                compute_capability: None,
                memory_bandwidth_gbps: 0.0,
            });
            c.est_tokens_per_sec_7b = 56.0;
            c.processor_tokens_per_sec_7b = Some(processor);
        };
        let mut own = (*s.state.local_capability.load_full().unwrap()).clone();
        with_card(&mut own);
        s.state.local_capability.store(Some(Arc::new(own)));
        for peer in [&s.near, &other] {
            with_card(
                s.state
                    .peer_registry
                    .get_mut(peer)
                    .unwrap()
                    .capability
                    .as_mut()
                    .unwrap(),
            );
        }
        (s, other)
    }

    /// #356, the user's aim (2026-10-26): once the swarm has the cards for a
    /// model, its layers leave the processors. A complete chain running 8 of
    /// its 32 layers on processors (1 token/s at 7B: 31 ms a layer) is planned
    /// onto the three cards (108 ms a token against 323): of the chains on
    /// cards alone, the one fetching least — this node keeps 0-12, the near
    /// peer 20-32, and the third machine fetches the two parts between.
    /// At 6.3 tokens/s the cards save a token only 5 ms (108 against 113),
    /// but the slowest machine goes from ~27 ms to 7 — and a stream of checks
    /// runs at its slowest machine — so it moves too (#318: Petals compares
    /// bottlenecks; until 2026-11-05 this case was the control, the bar laid
    /// on the sum). Control: processors at 20 tokens/s make the complete
    /// chain the cheaper a token (~86 ms against 108), and a chain slower for
    /// every reply is never moved to, whatever its slowest machine.
    #[test]
    fn a_complete_chain_moves_onto_cards_that_can_carry_it() {
        for processor in [1.0, 6.3] {
            let (s, other) = three_cards(processor);
            let chain = s.chain().expect("a near chain");
            assert!(!chain.complete, "{processor}: {chain:?}");
            assert_eq!(chain.price.processor_layers, 0, "{chain:?}");
            assert_eq!(
                chain.segments,
                vec![
                    (s.local(), (0, 12)),
                    (other, (12, 20)),
                    (s.near.clone(), (20, 32))
                ]
            );
            assert!(s.offered().is_empty(), "this node holds its run");
        }

        let (s, _) = three_cards(20.0);
        let chain = s.chain().expect("a near chain");
        assert!(
            chain.complete,
            "control: a slower chain is not taken: {chain:?}"
        );
        assert_eq!(chain.price.processor_layers, 8, "{chain:?}");
        assert!(s.offered().is_empty());
    }
}
