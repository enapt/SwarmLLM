//! Card ranges (#506): each machine with a graphics card holds ONE contiguous
//! run of a model's layers, sized to what its card can run, and the runs are
//! staggered so every layer sits on as many cards as the model's replica
//! target asks.
//!
//! The user, 2026-11-27: "improve the pruning and auto manage system to evenly
//! distribute chunks including redundancy". Until this, every rule choosing
//! parts — the score's rarest-first, the download ring, the replica count, the
//! per-part rank prune gave copies back by — weighed ONE PART at a time. A
//! card can serve only a contiguous run of layers, so parts spread one at a
//! time give a model copies to download from and nothing a card can run: the
//! 27B's far chain was `225e6fe7:0-21 aa3d1e66:21-64` on 11-23, complete, and
//! on 11-27 the Australian 3060 held its parts 0, 7, 12, 15 and 30, the Belgian
//! 4050 0-5, 13, 17 and 18 — two cards with nothing to run. Only a chain
//! (`near_chain`) gave a machine a contiguous segment, and only where no
//! machine runs the model whole.
//!
//! **The plan**, computed alike by every machine from what each advertises and
//! holds, for a model somebody asked for and no chain is planned for:
//!
//! - each machine whose models run on its card is weighed with the layers its
//!   card holds of this model (the model's card curve, as a chain weighs it),
//!   within its ceiling and the disk it may fetch into;
//! - a machine that already holds a run of at least half its card's worth is
//!   SETTLED: it is planned first, where most of what it holds lies, so a
//!   machine joining moves nobody (the shortest held runs first — a whole
//!   copy, which could stand anywhere, takes what they leave);
//! - every other machine, in rendezvous order, takes the window of its card's
//!   width whose layers are on fewest cards: most layers on no card, then on
//!   one, … up to the target — the weakest layer lifted first;
//! - a machine whose best window lifts no layer below the target is not
//!   planned: the model has its copies on cards without it.
//!
//! The download pass fetches this node's run past the replica target and the
//! ring, and NOTHING else of the model on a weighed card (a hole excepted).
//! Prune keeps the run below urgent pressure and gives back a part outside it
//! once the target's worth of other machines hold it inside theirs — never a
//! model held whole here (`completes_model_here_counting`). A planned machine
//! that fetches nothing for `CARRIER_PATIENCE` is passed over, as a chain's
//! is. A/B `SWARMLLM_CARD_RANGES=0`.
//!
//! Prior art: Petals' `block_selection.py` — `_choose_best_start` gives each
//! server the window of `num_blocks` consecutive blocks whose sorted
//! throughputs are lexicographically smallest (the weakest block lifted first,
//! spans summed in sorted peer order so every server computes the same
//! floats); `should_choose_other_blocks` keeps a server where it is unless a
//! move lifts the swarm's minimum by 1/0.75, and never leaves a block with
//! nobody. Parallax (arXiv 2509.26182) water-fills layers by capacity into
//! consecutive ranges and slots a joining GPU at the bottleneck layer. Taken:
//! the weakest window first, counted in layers below the target, and
//! incumbency. Not taken: moving a placed server to rebalance (the residual,
//! #506) — a run once held stays, so a join moves nobody (the mod-N cascade
//! of a plan in hash order: Dynamo's and CRUSH's lesson), and Petals'
//! issue #389 shows servers re-choosing at once overshoot. BitTorrent's
//! rarest-first is what scattered the parts; streaming clients (libtorrent's
//! sequential mode) keep it only to choose which end of a run to extend.

use std::collections::HashSet;
use std::sync::Arc;
use std::time::{Duration, Instant};

use dashmap::DashMap;

use crate::types::{ModelId, ModelManifest, NodeId, ShardId, MMPROJ_SHARD_INDEX};

use super::super::AutoShardManager;
use super::near_chain::{parts_in_order, parts_touching, Part};
use super::{rendezvous, CARRIER_PATIENCE, PASSED_OVER_FOR};

/// How long a model's plan answers before it is computed again — a chain's
/// minute (`near_chain::CHAIN_ANSWERS_FOR`).
const PLAN_ANSWERS_FOR: Duration = Duration::from_secs(60);

/// What the card ranges remember between passes. Each entry is a DEADLINE.
#[derive(Default)]
pub(in crate::model::auto_manage) struct CardRangeBook {
    /// Each model's plan, and until when it answers.
    plans: DashMap<ModelId, (Instant, Option<Arc<CardRanges>>)>,
    /// A planned machine's lease: when it lapses unless renewed, and the
    /// layers it held at the last renewal.
    leases: DashMap<(ModelId, NodeId), (Instant, u32)>,
    /// Until when a machine whose lease lapsed is left out of a model's plan.
    passed_over_until: DashMap<(ModelId, NodeId), Instant>,
}

/// A model's card ranges.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(in crate::model::auto_manage) struct CardRanges {
    /// Each planned machine's run, `[start, end)`, in planning order.
    runs: Vec<(NodeId, (u32, u32))>,
    /// Every machine with a card the plan weighed, planned or not.
    weighed: HashSet<NodeId>,
    /// The copies on cards it aims each layer at — the replica target.
    copies: u32,
    /// The fewest cards any layer is planned on.
    least: u32,
}

impl CardRanges {
    fn run_of(&self, node: &NodeId) -> Option<(u32, u32)> {
        self.runs
            .iter()
            .find(|(n, _)| n == node)
            .map(|(_, run)| *run)
    }

    /// `node:start-end` for each run, in planning order — for the log.
    fn describe(&self) -> String {
        self.runs
            .iter()
            .map(|(node, (start, end))| format!("{node}:{start}-{end}"))
            .collect::<Vec<_>>()
            .join(" ")
    }
}

/// What a copy of a part is to the card ranges ([`AutoShardManager::card_range_copy`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::model::auto_manage) enum CardRangeCopy {
    /// No plan for its model, or this node is not a card it weighed.
    NotPlanned,
    /// In this node's run.
    InThisRun,
    /// Outside this node's run, and the target's worth of other machines
    /// hold it inside theirs.
    Surplus,
    /// Outside this node's run, with too few holding it inside theirs.
    OutsideTheRun,
}

/// A machine with a card, as the plan weighs it.
#[derive(Debug, Clone)]
struct Card {
    node: NodeId,
    /// Layers of the model its card holds, within its ceiling.
    layers: u32,
    /// By part position (`parts_in_order`): held, or being fetched.
    has: Vec<bool>,
    /// Layers it holds, fetches not counted — what its lease renews on.
    held_layers: u32,
    /// Fetching a part of the model now.
    fetching: bool,
    /// What it may be handed to fetch (`room_to_fetch_bytes`).
    disk_bytes: u64,
    /// Its rendezvous weight for the model.
    rank: [u8; 32],
}

/// The layers of `[x, e)` the parts marked in `has` cover.
fn layers_held(parts: &[Part], has: &[bool], x: u32, e: u32) -> u32 {
    parts
        .iter()
        .zip(has)
        .filter(|(_, &h)| h)
        .map(|(p, _)| p.end.min(e).saturating_sub(p.start.max(x)))
        .sum()
}

/// The longest run of layers the parts marked in `has` cover without a gap.
fn longest_held_run(parts: &[Part], has: &[bool], layers: u32) -> u32 {
    let (mut best, mut run) = (0, 0);
    for (p, &h) in parts.iter().zip(has) {
        let width = p.end.min(layers).saturating_sub(p.start);
        run = if h { run + width } else { 0 };
        best = best.max(run);
    }
    best
}

/// Every window of whole parts at most `width` layers wide, one a part start:
/// the widest from each. A part wider than `width` starts none.
fn windows(parts: &[Part], layers: u32, width: u32) -> Vec<(u32, u32)> {
    let mut out = Vec::new();
    for (a, first) in parts.iter().enumerate() {
        let x = first.start;
        if x >= layers {
            break;
        }
        let mut e = None;
        for p in &parts[a..] {
            let end = p.end.min(layers);
            if end - x > width {
                break;
            }
            e = Some(end);
            if end >= layers {
                break;
            }
        }
        if let Some(e) = e {
            out.push((x, e));
        }
    }
    out
}

/// The bytes of the parts `[x, e)` reads that `has` lacks — a part past the
/// last layer too where the window ends the model (`part_serves_layers`).
fn lacking_bytes(parts: &[Part], has: &[bool], x: u32, e: u32, layers: u32) -> u64 {
    parts
        .iter()
        .zip(has)
        .filter(|(p, &h)| {
            !h && crate::inference::split::part_serves_layers((p.start, p.end), x, e, layers)
        })
        .map(|(p, _)| p.size)
        .sum()
}

/// How many of `[x, e)`'s layers stand on no card, on one, … up to
/// `copies - 1`: compared lexicographically, the window lifting the weakest
/// layers is the greatest (Petals' sorted-throughput rule, in layer counts).
fn shortfall(coverage: &[u32], x: u32, e: u32, copies: u32) -> Vec<u32> {
    let mut counts = vec![0; copies as usize];
    for &c in &coverage[x as usize..e as usize] {
        if c < copies {
            counts[c as usize] += 1;
        }
    }
    counts
}

/// The plan over `cards`: each card's run, by position, or `None` where it is
/// not planned. See the module's doc for the order and the rule.
fn plan_card_ranges(
    layers: u32,
    parts: &[Part],
    cards: &[Card],
    copies: u32,
) -> Vec<Option<(u32, u32)>> {
    let copies = copies.max(1);
    let mut coverage = vec![0u32; layers as usize];
    let mut runs = vec![None; cards.len()];
    let run_of = |c: &Card| longest_held_run(parts, &c.has, layers);
    let settled = |c: &Card| 2 * run_of(c) >= c.layers.min(layers);
    // Settled first, the shortest held runs first; then the rest. Each in
    // rendezvous order, highest first.
    let mut order: Vec<usize> = (0..cards.len()).filter(|&i| cards[i].layers > 0).collect();
    order.sort_by(|&a, &b| {
        let (ca, cb) = (&cards[a], &cards[b]);
        settled(cb)
            .cmp(&settled(ca))
            .then_with(|| {
                if settled(ca) {
                    run_of(ca).cmp(&run_of(cb))
                } else {
                    std::cmp::Ordering::Equal
                }
            })
            .then_with(|| cb.rank.cmp(&ca.rank))
    });
    for i in order {
        let card = &cards[i];
        let is_settled = settled(card);
        let best = windows(parts, layers, card.layers.min(layers))
            .into_iter()
            .filter(|&(x, e)| lacking_bytes(parts, &card.has, x, e, layers) <= card.disk_bytes)
            .map(|(x, e)| {
                let short = shortfall(&coverage, x, e, copies);
                let held = layers_held(parts, &card.has, x, e);
                ((x, e), short, held)
            })
            .max_by(|(wa, sa, ha), (wb, sb, hb)| {
                let first = if is_settled {
                    ha.cmp(hb).then_with(|| sa.cmp(sb))
                } else {
                    sa.cmp(sb).then_with(|| ha.cmp(hb))
                };
                first
                    .then_with(|| (wa.1 - wa.0).cmp(&(wb.1 - wb.0)))
                    .then_with(|| wb.0.cmp(&wa.0))
            });
        let Some(((x, e), short, _)) = best else {
            continue;
        };
        if short.iter().all(|&n| n == 0) {
            continue;
        }
        for c in &mut coverage[x as usize..e as usize] {
            *c += 1;
        }
        runs[i] = Some((x, e));
    }
    runs
}

/// Card ranges switched off for an A/B (`SWARMLLM_CARD_RANGES=0`).
fn card_ranges_switched_off() -> bool {
    static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *OFF.get_or_init(|| {
        std::env::var("SWARMLLM_CARD_RANGES")
            .map(|v| v == "0" || v.eq_ignore_ascii_case("false"))
            .unwrap_or(false)
    })
}

impl AutoShardManager {
    /// `manifest`'s card ranges, as last computed within [`PLAN_ANSWERS_FOR`].
    fn card_ranges(&self, manifest: &ModelManifest) -> Option<Arc<CardRanges>> {
        let book = &self.card_range_book;
        let now = Instant::now();
        if let Some(plan) = book
            .plans
            .get(&manifest.id)
            .filter(|p| now < p.0)
            .map(|p| p.1.clone())
        {
            return plan;
        }
        let plan = self.compute_card_ranges(manifest).map(Arc::new);
        let before = book.plans.get(&manifest.id).and_then(|p| p.1.clone());
        if let Some(planned) = plan.as_ref().filter(|p| before.as_deref() != Some(&***p)) {
            tracing::info!(
                model = %manifest.id,
                copies = planned.copies,
                least = planned.least,
                weighed = planned.weighed.len(),
                plan = %planned.describe(),
                "DIAG: card ranges planned — each card one run of the model, staggered so every \
                 layer is on the target's cards (#506)"
            );
        }
        book.plans
            .insert(manifest.id.clone(), (now + PLAN_ANSWERS_FOR, plan.clone()));
        plan
    }

    /// Forget every computed plan, as a minute passing would.
    #[cfg(test)]
    pub(in crate::model::auto_manage) fn forget_card_ranges(&self) {
        self.card_range_book.plans.clear();
    }

    /// `manifest`'s card ranges, computed now. `None`: switched off, nobody
    /// asked for it, a chain is planned for it (the chain places its
    /// segments), no header or card curve here, its parts do not tile its
    /// layers, this node states no ceiling, or no machine weighed has a card.
    fn compute_card_ranges(&self, manifest: &ModelManifest) -> Option<CardRanges> {
        let layers = manifest.num_layers;
        if layers == 0
            || card_ranges_switched_off()
            || !self.asked_for(&manifest.id)
            || self.chain_planned_for(manifest)
        {
            return None;
        }
        let curve = self.carry_curve(&manifest.id)?;
        let parts = parts_in_order(manifest)?;
        let state = &self.shared_state;
        let card_curve = state
            .model_process_pool
            .segment_cost_curve(&manifest.id, true)?;
        let layer_work = crate::inference::scheduler::layer_work_for(state, &manifest.id);
        let local = state.identity.node_id().clone();
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
        let mut cards = Vec::new();
        for node in self.machines_in_scope() {
            if self.passed_over_for_card_range(&manifest.id, &node) {
                continue;
            }
            let Some(ceiling) = self.ceiling_layers(&node, manifest, curve) else {
                if node == local {
                    return None;
                }
                continue;
            };
            let card_layers = self
                .advertised_devices(&node, layers, Some(card_curve), layer_work)
                .and_then(|d| d.card)
                .unwrap_or(0)
                .min(ceiling);
            if card_layers == 0 {
                continue;
            }
            let held: Vec<bool> = holders.iter().map(|h| h.contains(&node)).collect();
            let has: Vec<bool> = held
                .iter()
                .zip(&fetching)
                .map(|(&h, f)| h || f.contains(&node))
                .collect();
            cards.push(Card {
                held_layers: layers_held(&parts, &held, 0, layers),
                fetching: fetching.iter().any(|f| f.contains(&node)),
                disk_bytes: self.room_to_fetch_bytes(&node),
                rank: rendezvous(&manifest.id, &node),
                node,
                layers: card_layers,
                has,
            });
        }
        if cards.is_empty() {
            return None;
        }
        let live = state.cfg();
        let pool_size = crate::pool::scope::effective_pool_size(state);
        let copies = self
            .shed_target_replicas(&manifest.id, live.auto_manage.min_replicas, pool_size)
            .max(1);
        loop {
            let runs = plan_card_ranges(layers, &parts, &cards, copies);
            // A planned machine lacking parts of its run keeps its place only
            // while it fetches (`CARRIER_PATIENCE`), as a chain's does.
            let mut named = HashSet::new();
            let stalled = runs.iter().enumerate().find_map(|(i, run)| {
                let (x, e) = (*run)?;
                let card = &cards[i];
                if lacking_bytes(&parts, &card.has, x, e, layers) == 0 {
                    return None;
                }
                named.insert(card.node.clone());
                (!self.planned_card_keeps_its_place(&manifest.id, card)).then_some(i)
            });
            if let Some(i) = stalled {
                cards.remove(i);
                continue;
            }
            self.card_range_book
                .leases
                .retain(|(model, node), _| model != &manifest.id || named.contains(node));
            let mut coverage = vec![0u32; layers as usize];
            for &(x, e) in runs.iter().flatten() {
                for c in &mut coverage[x as usize..e as usize] {
                    *c += 1;
                }
            }
            return Some(CardRanges {
                runs: runs
                    .iter()
                    .zip(&cards)
                    .filter_map(|(run, card)| run.map(|r| (card.node.clone(), r)))
                    .collect(),
                weighed: cards.iter().map(|c| c.node.clone()).collect(),
                copies,
                least: coverage.iter().copied().min().unwrap_or(0),
            });
        }
    }

    /// Is `node` left out of `model`'s card ranges because its lease lapsed?
    fn passed_over_for_card_range(&self, model: &ModelId, node: &NodeId) -> bool {
        let key = (model.clone(), node.clone());
        let book = &self.card_range_book;
        let until = book.passed_over_until.get(&key).map(|u| *u);
        match until {
            Some(until) if Instant::now() < until => true,
            Some(_) => {
                book.passed_over_until.remove(&key);
                false
            }
            None => false,
        }
    }

    /// Does `card`, planned a run it lacks parts of, keep its place? Renewed
    /// while it gains layers or fetches; after [`CARRIER_PATIENCE`] with
    /// neither it is passed over for [`PASSED_OVER_FOR`] — a machine whose
    /// budget or disk refuses its run says nothing, and would hold its layers
    /// "covered" for ever.
    fn planned_card_keeps_its_place(&self, model: &ModelId, card: &Card) -> bool {
        let book = &self.card_range_book;
        let key = (model.clone(), card.node.clone());
        let now = Instant::now();
        let renewed = (now + CARRIER_PATIENCE, card.held_layers);
        let lease = book.leases.get(&key).map(|l| *l);
        let lapsed = match lease {
            Some((_, held)) if card.held_layers > held || card.fetching => {
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
                machine = %card.node,
                "DIAG: a card planned a run of this model has fetched none of it — planning the \
                 card ranges without it for a while (#506)"
            );
            book.leases.remove(&key);
            book.passed_over_until.insert(key, now + PASSED_OVER_FOR);
        }
        !lapsed
    }

    /// The parts of this node's run of `manifest`'s card ranges — empty where
    /// it is weighed and not planned. `None`: no plan, or this node is not a
    /// card it weighed; the model's parts are then today's rules' to place.
    /// The download pass fetches what of it this node lacks, past the replica
    /// target and the ring, and nothing else of the model.
    pub(in crate::model::auto_manage) fn this_nodes_card_run(
        &self,
        manifest: &ModelManifest,
    ) -> Option<HashSet<u32>> {
        let plan = self.card_ranges(manifest)?;
        let local = self.shared_state.identity.node_id();
        if !plan.weighed.contains(local) {
            return None;
        }
        Some(
            plan.run_of(local)
                .map(|run| parts_touching(manifest, run).into_iter().collect())
                .unwrap_or_default(),
        )
    }

    /// What this node's copy of `shard_id` is to its model's card ranges —
    /// asked by prune (`copy_kept_because`) and, through it, by the download
    /// pass before a fetch: the plan counts in-flight parts as held, so the
    /// answer before a fetch is the answer after it.
    pub(in crate::model::auto_manage) fn card_range_copy(
        &self,
        shard_id: &ShardId,
    ) -> CardRangeCopy {
        if shard_id.index == MMPROJ_SHARD_INDEX {
            return CardRangeCopy::NotPlanned;
        }
        let Some(manifest) = self
            .shared_state
            .model_registry
            .get_manifest(&shard_id.model_id)
        else {
            return CardRangeCopy::NotPlanned;
        };
        let Some(plan) = self.card_ranges(&manifest) else {
            return CardRangeCopy::NotPlanned;
        };
        let local = self.shared_state.identity.node_id();
        if !plan.weighed.contains(local) {
            return CardRangeCopy::NotPlanned;
        }
        let touches = |run: (u32, u32)| parts_touching(&manifest, run).contains(&shard_id.index);
        if plan.run_of(local).is_some_and(touches) {
            return CardRangeCopy::InThisRun;
        }
        let holders = self.shared_state.model_registry.shard_holders(shard_id);
        let held_in_runs = plan
            .runs
            .iter()
            .filter(|(node, run)| node != local && holders.contains(node) && touches(*run))
            .count();
        if held_in_runs >= plan.copies as usize {
            CardRangeCopy::Surplus
        } else {
            CardRangeCopy::OutsideTheRun
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{plan_card_ranges, windows, Card, Part};
    use crate::types::NodeId;

    const GB: u64 = 1 << 30;

    /// A 32-layer model in eight 4-layer parts of 1 GB.
    fn parts() -> Vec<Part> {
        (0..8)
            .map(|i| Part {
                index: i,
                start: i * 4,
                end: i * 4 + 4,
                size: GB,
            })
            .collect()
    }

    fn card(byte: u8, layers: u32, has: &[u32]) -> Card {
        let has: Vec<bool> = (0..8).map(|i| has.contains(&i)).collect();
        Card {
            node: NodeId([byte; 32]),
            layers,
            held_layers: has.iter().filter(|&&h| h).count() as u32 * 4,
            has,
            fetching: false,
            disk_bytes: 100 * GB,
            rank: [byte; 32],
        }
    }

    fn coverage(runs: &[Option<(u32, u32)>]) -> Vec<u32> {
        let mut c = vec![0; 32];
        for &(x, e) in runs.iter().flatten() {
            for l in x..e {
                c[l as usize] += 1;
            }
        }
        c
    }

    #[test]
    fn windows_are_whole_parts_no_wider_than_the_card() {
        assert_eq!(
            windows(&parts(), 32, 10),
            vec![
                (0, 8),
                (4, 12),
                (8, 16),
                (12, 20),
                (16, 24),
                (20, 28),
                (24, 32),
                (28, 32)
            ]
        );
        assert!(
            windows(&parts(), 32, 3).is_empty(),
            "a part wider than the card starts none"
        );
    }

    /// Three empty cards of 12 layers and a target of one copy: the runs
    /// stagger and every layer is on a card — where one part at a time (the
    /// score, the ring) gave each card scattered parts.
    #[test]
    fn empty_cards_take_staggered_runs_that_cover_every_layer() {
        let cards = [card(1, 12, &[]), card(2, 12, &[]), card(3, 12, &[])];
        let runs = plan_card_ranges(32, &parts(), &cards, 1);
        assert!(runs.iter().all(Option::is_some), "{runs:?}");
        assert!(coverage(&runs).iter().all(|&c| c >= 1), "{runs:?}");
        // Three cards of 12 cover 32 layers once with 4 to spare: no layer
        // three deep.
        assert!(coverage(&runs).iter().all(|&c| c <= 2), "{runs:?}");
    }

    /// Two copies asked of four 16-layer cards: every layer on two cards.
    /// Control: at one copy two cards suffice and the other two are not
    /// planned — the model has its copies without them.
    #[test]
    fn a_target_of_two_puts_every_layer_on_two_cards() {
        let cards = [
            card(1, 16, &[]),
            card(2, 16, &[]),
            card(3, 16, &[]),
            card(4, 16, &[]),
        ];
        let runs = plan_card_ranges(32, &parts(), &cards, 2);
        assert!(coverage(&runs).iter().all(|&c| c == 2), "{runs:?}");

        let runs = plan_card_ranges(32, &parts(), &cards, 1);
        assert_eq!(runs.iter().flatten().count(), 2, "{runs:?}");
        assert!(coverage(&runs).iter().all(|&c| c == 1), "{runs:?}");
    }

    /// A card holding a run of its card's worth keeps it, whatever its rank:
    /// a machine joining moves nobody. The empty card fills what it leaves.
    #[test]
    fn a_settled_run_keeps_its_place_and_a_newcomer_fills_the_rest() {
        // Ranked below the newcomer, it would otherwise be placed after it.
        let settled = card(1, 16, &[0, 1, 2, 3]);
        let newcomer = card(9, 16, &[]);
        let runs = plan_card_ranges(32, &parts(), &[settled, newcomer], 1);
        assert_eq!(runs, vec![Some((0, 16)), Some((16, 32))]);
    }

    /// A whole copy on a card short of the model is planned AFTER the runs
    /// that can stand in one place only, so it covers what they leave — the
    /// 27B's case: the Belgian 4050's 0-5 and this node's 0-9 at its start,
    /// the US 5080 holding it whole.
    #[test]
    fn a_whole_copy_covers_what_the_settled_runs_leave() {
        let whole = || card(9, 16, &[0, 1, 2, 3, 4, 5, 6, 7]);
        let start = || card(1, 12, &[0, 1, 2]);
        let runs = plan_card_ranges(32, &parts(), &[whole(), start()], 1);
        assert_eq!(runs[1], Some((0, 12)), "{runs:?}");
        let (x, _) = runs[0].expect("the whole copy is planned");
        assert!(x >= 12, "it goes where the start's run is not: {runs:?}");
        // What the two leave, a third card covers.
        let runs = plan_card_ranges(32, &parts(), &[whole(), start(), card(5, 8, &[])], 1);
        assert!(coverage(&runs).iter().all(|&c| c >= 1), "{runs:?}");
    }

    /// The Australian 3060's case: scattered parts are no run, so the card is
    /// a newcomer and takes the window the others leave weakest — what it
    /// holds there counts only as a tie-break.
    #[test]
    fn scattered_parts_do_not_pin_a_card_to_them() {
        let start = card(1, 16, &[0, 1, 2, 3]);
        let scattered = card(2, 16, &[0, 2, 5]);
        let runs = plan_card_ranges(32, &parts(), &[start, scattered], 1);
        assert_eq!(runs[1], Some((16, 32)), "{runs:?}");
    }

    /// The night of 2026-11-27, on the 27B's own parts (64 layers, 32 parts,
    /// the head past the last layer): this node's 8 GB card holding parts
    /// 0-9 and 30-31, the Belgian 4050 0-5, 13, 17 and 18, the US 5080 all
    /// of it, the Australian 3060 parts 0, 7, 12, 15 and 30 — nothing a card
    /// could run but the first two's starts. Card widths as their cards hold
    /// the model (about 22, 15, 46 and 38 layers). Planned, every layer is on
    /// a card, the 5080 covers the end, and the 3060 is given ONE run of its
    /// card's width — 38 layers it can serve, where it served none.
    #[test]
    fn the_27b_field_case_gives_every_card_one_run() {
        let bounds = [
            0, 1, 3, 5, 7, 9, 11, 13, 15, 18, 21, 24, 27, 30, 33, 35, 37, 39, 41, 43, 45, 47, 49,
            51, 53, 55, 57, 59, 61, 62, 63, 64, 65,
        ];
        let parts: Vec<Part> = bounds
            .windows(2)
            .enumerate()
            .map(|(i, w)| Part {
                index: i as u32,
                start: w[0],
                end: w[1],
                size: GB / 2,
            })
            .collect();
        let card = |byte: u8, layers: u32, has: &[u32]| {
            let has: Vec<bool> = (0..32).map(|i| has.contains(&i)).collect();
            Card {
                node: NodeId([byte; 32]),
                layers,
                held_layers: 0,
                has,
                fetching: false,
                disk_bytes: 100 * GB,
                rank: [byte; 32],
            }
        };
        let here = card(1, 22, &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 30, 31]);
        let belgium = card(2, 15, &[0, 1, 2, 3, 4, 5, 13, 17, 18]);
        let us = card(3, 46, &(0..32).collect::<Vec<u32>>());
        let australia = card(4, 38, &[0, 7, 12, 15, 30]);
        let runs = plan_card_ranges(64, &parts, &[here, belgium, us, australia], 3);
        let mut coverage = vec![0; 64];
        for &(x, e) in runs.iter().flatten() {
            for l in x..e {
                coverage[l as usize] += 1;
            }
        }
        assert!(coverage.iter().all(|&c| c >= 1), "{runs:?}");
        assert_eq!(runs[0], Some((0, 21)), "this node keeps what it holds");
        assert_eq!(runs[1], Some((0, 15)), "so does the 4050");
        assert_eq!(
            runs[2].map(|r| r.1),
            Some(64),
            "the whole copy covers the end"
        );
        let (x, e) = runs[3].expect("the 3060 is planned");
        assert!(e - x >= 36, "one run of its card's width: {runs:?}");
    }

    /// A card is planned only where its disk takes what it lacks of the run:
    /// with none, nowhere; with two parts' room, two parts' run — or the four
    /// it holds half of.
    #[test]
    fn a_card_is_planned_only_where_its_disk_takes_the_run() {
        let with_disk = |disk: u64, has: &[u32]| {
            let mut c = card(1, 16, has);
            c.disk_bytes = disk;
            c
        };
        let plan = |c: Card| plan_card_ranges(32, &parts(), &[c], 1);
        assert_eq!(plan(with_disk(0, &[])), vec![None]);
        assert_eq!(plan(with_disk(2 * GB, &[])), vec![Some((24, 32))]);
        assert_eq!(plan(with_disk(2 * GB, &[6, 7])), vec![Some((16, 32))]);
    }
}

#[cfg(test)]
mod manager_tests {
    use std::sync::atomic::Ordering::Relaxed;
    use std::sync::Arc;

    use super::super::super::test_support::{
        capability, holds, make_test_manager_with_config, offered, peer_with_disk,
        register_manifest_with_sized_shards,
    };
    use super::super::super::AutoShardManager;
    use super::CardRangeCopy;
    use crate::config::Config;
    use crate::daemon::SharedState;
    use crate::types::{GpuInfo, ModelId, NodeCapability, NodeId, ShardId};

    const MIB: u64 = 1024 * 1024;
    /// What a holder's admission charges: 300 MB + 200 MB a layer.
    const CURVE: (u64, u64) = (300, 200);
    fn ceiling_for(layers: u64) -> u64 {
        CURVE.0 + CURVE.1 * layers
    }

    struct Fleet {
        state: Arc<SharedState>,
        manager: AutoShardManager,
        mid: ModelId,
        /// Holds the model whole on a card of 12 of its 32 layers.
        whole: NodeId,
    }

    impl Fleet {
        fn part(&self, index: u32) -> ShardId {
            ShardId {
                model_id: self.mid.clone(),
                index,
            }
        }

        fn local(&self) -> NodeId {
            self.state.identity.node_id().clone()
        }

        /// This node's run, computed afresh.
        fn run(&self) -> Option<Vec<u32>> {
            self.manager.forget_card_ranges();
            let manifest = self.state.model_registry.get_manifest(&self.mid).unwrap();
            self.manager.this_nodes_card_run(&manifest).map(|run| {
                let mut run: Vec<u32> = run.into_iter().collect();
                run.sort_unstable();
                run
            })
        }

        fn sheds(&self, index: u32, holders: &[NodeId]) -> bool {
            self.manager.forget_card_ranges();
            self.manager
                .would_shed_copy(&self.part(index), holders, 0.0, &[])
        }
    }

    /// Three cards of 12 of a 32-layer model somebody asked for: one holds it
    /// whole (and runs it whole beside its processor for less than any chain,
    /// so no chain is planned — the 27B's US 5080), this node and another hold
    /// nothing. Nothing is near.
    fn fleet() -> Fleet {
        let mut config = Config::default();
        config.auto_manage.min_replicas = 1;
        let (state, manager) = make_test_manager_with_config(config);
        state.models.auto_manage_default_model_cap.store(0, Relaxed);
        let ranges: Vec<(u32, u32)> = (0..8).map(|i| (i * 4, i * 4 + 4)).collect();
        let mid = register_manifest_with_sized_shards(&state, "ranged", 32, &ranges, 500 * MIB);
        state
            .model_process_pool
            .test_cost_curve
            .insert(mid.clone(), CURVE);
        let with_card = |c: &mut NodeCapability| {
            c.gpu = Some(GpuInfo {
                name: "card".to_string(),
                vram_total_mb: ceiling_for(12),
                vram_available_mb: ceiling_for(12),
                compute_capability: None,
                memory_bandwidth_gbps: 0.0,
            });
            c.est_tokens_per_sec_7b = 56.0;
            c.processor_tokens_per_sec_7b = Some(20.0);
        };
        let mut own = capability(Some(ceiling_for(32)), 100_000);
        with_card(&mut own);
        state.local_capability.store(Some(Arc::new(own)));
        let whole = peer_with_disk(&state, 3, Some(ceiling_for(32)), 100_000);
        let other = peer_with_disk(&state, 2, Some(ceiling_for(32)), 100_000);
        for peer in [&whole, &other] {
            with_card(
                state
                    .peer_registry
                    .get_mut(peer)
                    .unwrap()
                    .capability
                    .as_mut()
                    .unwrap(),
            );
        }
        holds(&state, &mid, &whole, &[0, 1, 2, 3, 4, 5, 6, 7]);
        state
            .region_demand
            .insert((mid.clone(), "BE".to_string()), 2.0);
        let mut trust = crate::types::ModelTrustInfo::new_discovered();
        trust.trust_level = crate::types::ModelTrustLevel::DemandVerified;
        state.models.model_trust.insert(mid.clone(), trust);
        Fleet {
            state,
            manager,
            mid,
            whole,
        }
    }

    /// #506: this node's card is planned ONE run of three whole parts, where
    /// the whole copy's run is not, and the download pass offers exactly that
    /// run — where before it offered nothing (every part's one holder met the
    /// target) or, past the ring, parts one at a time.
    #[test]
    fn a_card_fetches_its_run_of_the_model_and_nothing_else() {
        let f = fleet();
        assert!(
            !f.manager
                .chain_planned_for(&f.state.model_registry.get_manifest(&f.mid).unwrap()),
            "no chain: the whole holder runs it for less"
        );
        let run = f.run().expect("this card is weighed");
        assert_eq!(run.len(), 3, "a 12-layer card, 4-layer parts: {run:?}");
        assert!(
            run.windows(2).all(|w| w[1] == w[0] + 1),
            "one contiguous run: {run:?}"
        );
        assert!(run[0] >= 3, "not where the whole copy's run is: {run:?}");
        f.manager.forget_card_ranges();
        assert_eq!(offered(&f.manager, &f.state), run);
    }

    /// Prune keeps this card's run and gives back a part outside it that the
    /// target's worth of other cards hold inside theirs — the copy the count
    /// alone kept (two holders, a target of one, no disk pressure). A part
    /// outside the run that no other card holds inside its own stays.
    #[test]
    fn prune_keeps_the_run_and_gives_back_a_copy_other_cards_run() {
        let f = fleet();
        let run = f.run().expect("this card is weighed");
        let mut held = run.clone();
        held.push(0);
        holds(&f.state, &f.mid, &f.local(), &held);
        let run_after = f.run().expect("still weighed");
        assert_eq!(run_after, run, "holding its run settles it there");
        for &index in &run {
            assert!(
                !f.sheds(index, &[f.local(), f.whole.clone()]),
                "part {index} of the run is kept"
            );
        }
        assert_eq!(
            f.manager.card_range_copy(&f.part(0)),
            CardRangeCopy::Surplus,
            "part 0 is in the whole copy's run, and held there"
        );
        assert!(
            f.sheds(0, &[f.local(), f.whole.clone()]),
            "a copy no card here runs, that another card runs, goes"
        );
        // A part in no run of a card holding it — the other card's, which it
        // does not hold yet — is no surplus: nobody runs it from a card.
        let manifest = f.state.model_registry.get_manifest(&f.mid).unwrap();
        f.manager.forget_card_ranges();
        let plan = f.manager.card_ranges(&manifest).expect("a plan");
        let (wx, we) = plan.run_of(&f.whole).expect("the whole copy is planned");
        let outside: Vec<u32> = (0..8)
            .filter(|i| !run.contains(i) && (i * 4 >= we || i * 4 + 4 <= wx))
            .collect();
        assert!(!outside.is_empty(), "{plan:?}");
        for index in outside {
            assert_eq!(
                f.manager.card_range_copy(&f.part(index)),
                CardRangeCopy::OutsideTheRun,
                "part {index}: {plan:?}"
            );
        }
    }

    /// A machine whose models run on its processor is weighed by no plan: its
    /// parts stay today's rules' to place.
    #[test]
    fn a_processor_only_machine_is_not_weighed() {
        let f = fleet();
        let mut own = (*f.state.local_capability.load_full().unwrap()).clone();
        own.gpu = None;
        f.state.local_capability.store(Some(Arc::new(own)));
        assert_eq!(f.run(), None);
        assert_eq!(
            f.manager.card_range_copy(&f.part(0)),
            CardRangeCopy::NotPlanned
        );
    }
}
