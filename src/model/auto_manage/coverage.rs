//! A model its holders cannot carry is fetched by a machine that can (#231).
//!
//! A tester's report, 2026-10-07 (v0.3.229): the only computer holding a whole
//! Qwen 3.5 9B was a 6 GB processor-only peer capped at 5200 MB — the model
//! needs 6688 MB at its admission — while three graphics cards (11.6, 8 and
//! 5.7 GB) held none of it. Every part had a holder, so the replica target was
//! met and nothing asked a machine that COULD run the model to fetch it. Since
//! #230 the request is refused at once as "not enough memory in the swarm" —
//! honest, and still no answer.
//!
//! **Carrying** is the question replicas never asked: how many of a model's
//! layers its live holders could ever hold between them — the most layers
//! that can each be given to one holder of it, none past its advertised
//! ceiling (`NodeCapability::model_memory_ceiling_mb`, weighed with the model's
//! own admission curve), the arithmetic the planner's shortage report uses
//! (`scheduler::layers_carried`; a sum of each holder's own figure until #346,
//! which counted a layer once per holder). When that falls short of the model:
//!
//! - **one carrier at a time** fetches it: the machine with the highest
//!   rendezvous weight (`blake3(model ‖ node)`) among those with room for at
//!   least one part they lack and the disk for it, judged from the figures every
//!   node gossips — its own included (`local_capability`) — so all of them pick
//!   the same one. It fetches only what closes the SHORTFALL — whole parts, in
//!   model order — never more than its own ceiling, and if that is not enough
//!   the next carrier is chosen. A machine with room for the whole model is not
//!   asked to hold it: "no machine holds the whole model" (the user,
//!   2026-09-28, `docs/plans/wan_parallel.md`) — the model spreads across
//!   machines able to run their part;
//! - **prune keeps** any copy whose loss would leave the model short
//!   (`would_shed_copy` asks [`AutoShardManager::copy_carries_model`]), so the
//!   download pass — which asks the same function before each fetch — and
//!   prune agree, and a fetched part is not shed again (gotcha #795's loop).
//!
//! Only for a model somebody has ASKED for (regional demand or this node's own
//! requests) and only when the connected swarm's ceilings could carry it at
//! all: a model nobody can run must not pull a copy onto every machine. A model
//! only a split can run is placed by a CHAIN instead (`near_chain`, near
//! machines first, else every machine in scope — #346), and carrying stands
//! down while one is planned.
//! Unknown — no header here, a holder or candidate with no ceiling (older than
//! v0.3.230) — is never "short": everything behaves as before.

use std::collections::{HashMap, HashSet};

use crate::types::{ModelId, ModelManifest, NodeId, ShardId, MMPROJ_SHARD_INDEX};

use super::AutoShardManager;

mod drafter;
mod near_chain;
pub(super) use drafter::DrafterBook;
pub(super) use near_chain::{ChainPart, NearChainBook, Reach};

/// How long a model's cost curve is kept. It changes only with the node's own
/// context override; the header behind it never changes.
const CARRY_CURVE_TTL: std::time::Duration = std::time::Duration::from_secs(600);

/// Multiplies a carrier's score for the parts it fetches toward carrying a
/// model, so they are chosen ahead of routine replication within its budget.
pub(super) const CARRY_BONUS: f64 = 50.0;

/// How long a chosen carrier may make no progress on a model — no part gained,
/// none being fetched — before every node passes it over and the next machine
/// in the same ranking steps in. A carrier whose own storage budget or disk
/// reserve will not take its plan never fetches, and nothing it gossips says
/// so; without a lease it stayed elected for ever (review of #231). A lease
/// renewed by progress is how a stalled leader is replaced elsewhere
/// (Kubernetes' leader lease; CRUSH re-placing data off an "out" OSD). Part
/// downloads take minutes at the peers' serving rate, and one in flight renews
/// the lease, so a working carrier is not passed over.
const CARRIER_PATIENCE: std::time::Duration = std::time::Duration::from_secs(20 * 60);

/// How long a passed-over carrier stays passed over for that model — long
/// enough for the next one to finish, short enough that a machine whose budget
/// has since grown is asked again.
const PASSED_OVER_FOR: std::time::Duration = std::time::Duration::from_secs(2 * 60 * 60);

/// A model's chosen carrier, when its lease lapses unless renewed, and the
/// layers it held at the last renewal.
#[derive(Debug, Clone)]
pub(super) struct CarrierLease {
    node: NodeId,
    lapses_at: std::time::Instant,
    held_layers: u32,
}

impl AutoShardManager {
    /// The model's processor admission curve, `(fixed_mb, per_layer_mb)`, the
    /// same one every node's admission computes from the header (#230).
    fn carry_curve(&self, model_id: &ModelId) -> Option<(u64, u64)> {
        if let Some(entry) = self.carry_curves.get(model_id) {
            if entry.0.elapsed() < CARRY_CURVE_TTL {
                return entry.1;
            }
        }
        let curve = self
            .shared_state
            .model_process_pool
            .segment_cost_curve(model_id, false);
        self.carry_curves
            .insert(model_id.clone(), (std::time::Instant::now(), curve));
        curve
    }

    /// The capability `node` advertises — this node's own last broadcast for
    /// itself, so every node judges every machine from the same figures.
    fn advertised(&self, node: &NodeId) -> Option<(Option<u64>, u64)> {
        if node == self.shared_state.identity.node_id() {
            return self
                .shared_state
                .local_capability
                .load_full()
                .map(|c| (c.model_memory_ceiling_mb, c.disk_available_mb));
        }
        self.shared_state.peer_registry.get(node).and_then(|p| {
            p.capability
                .as_ref()
                .map(|c| (c.model_memory_ceiling_mb, c.disk_available_mb))
        })
    }

    /// How many of `manifest`'s layers `node` could ever hold. `None`: cannot
    /// tell (no ceiling advertised, or no curve).
    fn ceiling_layers(
        &self,
        node: &NodeId,
        manifest: &ModelManifest,
        curve: (u64, u64),
    ) -> Option<u32> {
        let ceiling_mb = self.advertised(node)?.0?;
        crate::inference::process_pool::layers_that_fit(ceiling_mb, curve.0, curve.1)
            .map(|layers| layers.min(manifest.num_layers))
    }

    /// Each live holder of `manifest` (this node included when it holds a
    /// part) with the parts it holds — prune's own live-holder reading
    /// (`live_holders_with_us`, without adding this node unconditionally).
    fn holdings(&self, manifest: &ModelManifest) -> HashMap<NodeId, HashSet<u32>> {
        let allowed = crate::pool::scope::allowed_node_set(&self.shared_state);
        let local = self.shared_state.identity.node_id();
        let mut held: HashMap<NodeId, HashSet<u32>> = HashMap::new();
        for shard in manifest
            .shards
            .iter()
            .filter(|s| s.index != MMPROJ_SHARD_INDEX)
        {
            let sid = ShardId {
                model_id: manifest.id.clone(),
                index: shard.index,
            };
            let holders = self.shared_state.model_registry.shard_holders(&sid);
            for h in crate::pool::scope::filter_allowed_holders(holders, &allowed) {
                if &h == local || self.shared_state.connected_node_ids.contains(&h) {
                    held.entry(h).or_default().insert(shard.index);
                }
            }
        }
        held
    }

    /// Layers of `manifest` its live holders could ever carry between them,
    /// `without` left out. `None` when that cannot be told — never "short".
    pub(super) fn carried_layers(
        &self,
        manifest: &ModelManifest,
        without: Option<&NodeId>,
    ) -> Option<u32> {
        let curve = self.carry_curve(&manifest.id)?;
        let mut holdings = Vec::new();
        for (node, parts) in self.holdings(manifest) {
            if Some(&node) == without {
                continue;
            }
            holdings.push((
                ranges_of(manifest, &parts),
                self.ceiling_layers(&node, manifest, curve)?,
            ));
        }
        Some(crate::inference::scheduler::layers_carried(holdings))
    }

    /// The parts `node` would fetch as `manifest`'s carrier: those it lacks,
    /// in model order, until they close `shortfall` — never past its room (its
    /// ceiling less what it holds). **The one plan** the choice of carrier
    /// reads (a machine that could take nothing is not chosen) and the
    /// carrier's own download pass fetches, so the two cannot disagree. Only
    /// the shortfall: a machine with room for the whole model is not asked to
    /// hold it ("no machine holds the whole model", the user, 2026-09-28).
    fn parts_to_carry(
        &self,
        manifest: &ModelManifest,
        node: &NodeId,
        curve: (u64, u64),
        holdings: &HashMap<NodeId, HashSet<u32>>,
        shortfall: u32,
    ) -> Vec<(u32, u64)> {
        let Some(could) = self.ceiling_layers(node, manifest, curve) else {
            return Vec::new();
        };
        let none = HashSet::new();
        let held = holdings.get(node).unwrap_or(&none);
        let held_layers =
            crate::inference::scheduler::layers_carried([(ranges_of(manifest, held), u32::MAX)]);
        let mut room = could.saturating_sub(held_layers);
        let mut closed = 0u32;
        let mut parts = Vec::new();
        for shard in manifest
            .shards
            .iter()
            .filter(|s| s.index != MMPROJ_SHARD_INDEX && !held.contains(&s.index))
        {
            let layers = shard.layer_range.1.saturating_sub(shard.layer_range.0);
            if closed >= shortfall || layers > room {
                break;
            }
            room -= layers;
            closed = closed.saturating_add(layers);
            parts.push((shard.index, shard.size_bytes));
        }
        parts
    }

    /// Is `model_id` a model being assembled — asked for, a part of it on a
    /// live computer and another part on none? The model whose missing parts
    /// come first (`scoring::fills_a_hole`); prune never gives back a copy of
    /// one to make room for another's (`prune::spare_for_a_hole`). Only live
    /// holders count: a part listed for a computer that is gone is missing.
    pub(super) fn model_has_holes(&self, model_id: &ModelId) -> bool {
        if !self.asked_for(model_id) {
            return false;
        }
        let registry = &self.shared_state.model_registry;
        let Some(manifest) = registry.get_manifest(model_id) else {
            return false;
        };
        let local = self.shared_state.identity.node_id();
        let held: Vec<bool> = manifest
            .shards
            .iter()
            .filter(|s| s.index != MMPROJ_SHARD_INDEX)
            .map(|s| {
                registry
                    .shard_holders(&ShardId {
                        model_id: model_id.clone(),
                        index: s.index,
                    })
                    .iter()
                    .any(|h| h == local || self.shared_state.connected_node_ids.contains(h))
            })
            .collect();
        held.iter().any(|&h| h) && held.iter().any(|&h| !h)
    }

    /// Has anyone asked for this model — regional demand gossiped by any node,
    /// or a request routed by this one since the last decay?
    pub(super) fn asked_for(&self, model_id: &ModelId) -> bool {
        self.shared_state
            .region_demand
            .iter()
            .any(|e| &e.key().0 == model_id && *e.value() > 0.0)
            || self
                .shared_state
                .models
                .model_request_counts
                .get(model_id)
                .is_some_and(|c| c.load(std::sync::atomic::Ordering::Relaxed) > 0)
    }

    /// This node and the connected machines in scope — in private mode, the
    /// pool's only (`pool::scope::allowed_node_set`, which `holdings` applies
    /// too): a machine outside the pool must neither be chosen to carry the
    /// pool's model nor count toward whether it could be carried.
    fn machines_in_scope(&self) -> Vec<NodeId> {
        let allowed = crate::pool::scope::allowed_node_set(&self.shared_state);
        let local = self.shared_state.identity.node_id().clone();
        std::iter::once(local.clone())
            .chain(
                self.shared_state
                    .connected_node_ids
                    .iter()
                    .map(|n| n.clone())
                    .filter(|n| *n != local),
            )
            .filter(|n| allowed.as_ref().is_none_or(|a| a.contains(n)) || *n == local)
            .collect()
    }

    /// Could the machines in scope — this node included — carry the model at
    /// all? Unknown for any machine is "cannot tell", so no.
    fn swarm_could_carry(&self, manifest: &ModelManifest, curve: (u64, u64)) -> bool {
        let mut total = 0u32;
        for node in self.machines_in_scope() {
            match self.ceiling_layers(&node, manifest, curve) {
                Some(layers) => total = total.saturating_add(layers),
                None => return false,
            }
        }
        total >= manifest.num_layers
    }

    /// Does `manifest` need a carrier: asked for, its holders short of it,
    /// the swarm able to carry it, and no chain planned for it — a chain
    /// places a model only a split can run (`near_chain`, #346), and two
    /// placements of one model would fetch past each other.
    pub(super) fn needs_carrier(&self, manifest: &ModelManifest) -> bool {
        if manifest.num_layers == 0
            || !self.asked_for(&manifest.id)
            || self.chain_planned_for(manifest)
        {
            return false;
        }
        let Some(curve) = self.carry_curve(&manifest.id) else {
            return false;
        };
        self.carried_layers(manifest, None)
            .is_some_and(|carried| carried < manifest.num_layers)
            && self.swarm_could_carry(manifest, curve)
    }

    /// The machine that should fetch more of `manifest` now: in rendezvous
    /// order (`blake3(model ‖ node)`), the first machine in scope whose carry
    /// plan (`parts_to_carry`) is not empty and fits the disk it advertises,
    /// and which has not let its lease lapse ([`CARRIER_PATIENCE`]).
    ///
    /// Every node ranks the same machines the same way, so they agree — except
    /// where they do not see the same machines (a partial mesh): two carriers
    /// may then fetch at once, which costs duplicate parts, bounded by the
    /// shortfall, and never a loop.
    pub(super) fn carrier_for(&self, manifest: &ModelManifest) -> Option<NodeId> {
        let curve = self.carry_curve(&manifest.id)?;
        let shortfall = manifest
            .num_layers
            .saturating_sub(self.carried_layers(manifest, None)?);
        let holdings = self.holdings(manifest);
        let mut ranked: Vec<NodeId> = self
            .machines_in_scope()
            .into_iter()
            .filter(|node| {
                let plan = self.parts_to_carry(manifest, node, curve, &holdings, shortfall);
                let bytes: u64 = plan.iter().map(|(_, size)| size).sum();
                !plan.is_empty()
                    && self
                        .advertised(node)
                        .is_some_and(|(_, disk_mb)| disk_mb.saturating_mul(1024 * 1024) > bytes)
            })
            .collect();
        ranked.sort_by_key(|node| std::cmp::Reverse(rendezvous(&manifest.id, node)));
        ranked
            .into_iter()
            .find(|node| self.carrier_keeps_its_lease(manifest, node, &holdings))
    }

    /// Does `node` keep (or take) the lease to carry `manifest`? It is renewed
    /// whenever the node holds more of the model than at the last renewal or
    /// is fetching a part of it; after [`CARRIER_PATIENCE`] with neither, the
    /// node is passed over for [`PASSED_OVER_FOR`].
    fn carrier_keeps_its_lease(
        &self,
        manifest: &ModelManifest,
        node: &NodeId,
        holdings: &HashMap<NodeId, HashSet<u32>>,
    ) -> bool {
        let key = (manifest.id.clone(), node.clone());
        if let Some(when) = self.carriers_passed_over.get(&key).map(|w| *w) {
            if when.elapsed() < PASSED_OVER_FOR {
                return false;
            }
            self.carriers_passed_over.remove(&key);
        }
        let none = HashSet::new();
        let held_layers = crate::inference::scheduler::layers_carried([(
            ranges_of(manifest, holdings.get(node).unwrap_or(&none)),
            u32::MAX,
        )]);
        let now = std::time::Instant::now();
        let renewed = CarrierLease {
            node: node.clone(),
            lapses_at: now + CARRIER_PATIENCE,
            held_layers,
        };
        let lapsed = match self.carrier_leases.get(&manifest.id).map(|l| l.clone()) {
            Some(lease) if &lease.node == node => {
                if held_layers > lease.held_layers || self.is_fetching_part_of(manifest, node) {
                    self.carrier_leases.insert(manifest.id.clone(), renewed);
                    false
                } else {
                    now >= lease.lapses_at
                }
            }
            _ => {
                self.carrier_leases.insert(manifest.id.clone(), renewed);
                false
            }
        };
        if lapsed {
            tracing::info!(
                model = %manifest.id,
                carrier = %node,
                "DIAG: the machine chosen to carry this model has made no progress — \
                 passing it over for the next (#231)"
            );
            self.carrier_leases.remove(&manifest.id);
            self.carriers_passed_over.insert(key, now);
        }
        !lapsed
    }

    /// Is `node` fetching a part of `manifest` right now — this node from its
    /// own download claims, a peer from the progress it gossips?
    fn is_fetching_part_of(&self, manifest: &ModelManifest, node: &NodeId) -> bool {
        let local = self.shared_state.identity.node_id();
        manifest.shards.iter().any(|s| {
            if node == local {
                return self
                    .shared_state
                    .models
                    .is_shard_in_progress(&manifest.id, s.index);
            }
            let sid = ShardId {
                model_id: manifest.id.clone(),
                index: s.index,
            };
            self.shared_state
                .models
                .peer_shard_downloads
                .get(&sid)
                .is_some_and(|v| v.iter().any(|(n, _)| n == node))
        })
    }

    /// The parts THIS node fetches now as `manifest`'s carrier — none unless
    /// the model needs one and this node is the one chosen.
    pub(super) fn parts_this_node_carries(&self, manifest: &ModelManifest) -> HashSet<u32> {
        let local = self.shared_state.identity.node_id();
        if !self.needs_carrier(manifest) || self.carrier_for(manifest).as_ref() != Some(local) {
            return HashSet::new();
        }
        let (Some(curve), Some(carried)) = (
            self.carry_curve(&manifest.id),
            self.carried_layers(manifest, None),
        ) else {
            return HashSet::new();
        };
        let shortfall = manifest.num_layers.saturating_sub(carried);
        self.parts_to_carry(manifest, local, curve, &self.holdings(manifest), shortfall)
            .into_iter()
            .map(|(index, _)| index)
            .collect()
    }

    /// Does `node`'s copy of `shard_id` carry a model somebody asked for?
    /// Prune keeps such a copy (`would_shed_copy`), and the download pass —
    /// asking the same before it fetches — fetches it. Two cases:
    ///
    /// - `node` is the machine chosen to carry the model while it needs one:
    ///   its parts are kept, and so fetched — the question is asked BEFORE the
    ///   part is held, so holdings alone cannot answer it;
    /// - otherwise, `node` adds to what carries the model, and without it the
    ///   model would fall short. A holder adding nothing (no room) is not kept.
    ///
    /// Never for a model a chain is planned for: the chain keeps its own
    /// segments (`copy_in_near_chain`), and this answer — per holder, not per
    /// part — would keep all 62 layers a holder with room for 7 has (#346).
    pub(super) fn copy_carries_model(&self, shard_id: &ShardId, node: &NodeId) -> bool {
        if shard_id.index == MMPROJ_SHARD_INDEX || !self.asked_for(&shard_id.model_id) {
            return false;
        }
        let Some(manifest) = self
            .shared_state
            .model_registry
            .get_manifest(&shard_id.model_id)
        else {
            return false;
        };
        if self.chain_planned_for(&manifest) {
            return false;
        }
        if self.needs_carrier(&manifest) && self.carrier_for(&manifest).as_ref() == Some(node) {
            return true;
        }
        let (Some(with), Some(without)) = (
            self.carried_layers(&manifest, None),
            self.carried_layers(&manifest, Some(node)),
        ) else {
            return false;
        };
        without < manifest.num_layers && with > without
    }

    /// Does this node hold `shard_id`'s model whole — or all but a few of its
    /// parts ([`short_of_whole_by_few`]) — once `shard_id` itself is counted as
    /// held? Prune keeps such a part below urgent disk pressure
    /// (`copy_kept_because`, #286): it is the one computer answering the model
    /// with no hop, or about to be. Asked of a part held (prune) and of one
    /// about to be fetched (the download pass, [`Self::parts_completing_model_here`])
    /// alike — the state after the fetch, as gotcha #795 requires. Read through
    /// `parts_missing_here`, the one count of what this computer holds
    /// (registry and files); a model with no manifest here is not held whole.
    pub(super) fn completes_model_here_counting(&self, shard_id: &ShardId) -> bool {
        if shard_id.index == MMPROJ_SHARD_INDEX {
            return false;
        }
        let Some(manifest) = self
            .shared_state
            .model_registry
            .get_manifest(&shard_id.model_id)
            .filter(|m| !m.shards.is_empty())
        else {
            return false;
        };
        let others = self
            .shared_state
            .parts_missing_here(&shard_id.model_id)
            .iter()
            .filter(|&&index| index != shard_id.index)
            .count();
        short_of_whole_by_few(others, manifest.shards.len())
    }

    /// The parts this node lacks of a model it already holds all but a few of
    /// — fetched to complete it, past the replica target (field report
    /// 2026-10-19: a 30B one part short since the pruner before #286 took it
    /// went through a peer abroad in 121 s, where all 34 parts here answered
    /// in 2.3 s, and the target never let the part back: it had three other
    /// holders). Only while the model is short by few parts BEFORE any of them
    /// lands — so never a model in fewer than eight parts, and never how a
    /// node starts on one. Each part lands in a state
    /// [`Self::completes_model_here_counting`] keeps, as does every part
    /// already held meanwhile, so prune and this pass agree (gotcha #795).
    pub(super) fn parts_completing_model_here(&self, manifest: &ModelManifest) -> HashSet<u32> {
        let missing = self.shared_state.parts_missing_here(&manifest.id);
        if missing.is_empty() || !short_of_whole_by_few(missing.len(), manifest.shards.len()) {
            return HashSet::new();
        }
        missing.into_iter().collect()
    }
}

/// One part in this many may be missing from a model this node still counts
/// as whole-or-completing ([`AutoShardManager::completes_model_here_counting`]):
/// one part of an 8-part 7B, four of a 34-part 30B, none of a model in fewer
/// than eight parts — a small model keeps the replica spread, and a node never
/// completes one it holds half of.
const COMPLETE_WHEN_SHORT_BY_ONE_PART_IN: usize = 8;

/// Is a model of `total` parts with `missing` of them absent here whole, or
/// short of whole by few enough to complete?
fn short_of_whole_by_few(missing: usize, total: usize) -> bool {
    missing * COMPLETE_WHEN_SHORT_BY_ONE_PART_IN <= total
}

/// `node`'s rendezvous weight for `model`, `blake3(model ‖ node)`: the order
/// every node ranks machines in for a model, so they choose alike from the
/// same gossip — the carrier (#231) and the near chain's ties.
fn rendezvous(model: &ModelId, node: &NodeId) -> [u8; 32] {
    let mut h = blake3::Hasher::new();
    h.update(model.0.as_bytes());
    h.update(&node.0);
    *h.finalize().as_bytes()
}

/// `node`'s rendezvous weight for one part, `blake3(model ‖ 0 ‖ index ‖ node)`:
/// per part, so the holders a model's parts keep are spread across machines.
pub(super) fn part_rank_weight(part: &ShardId, node: &NodeId) -> [u8; 32] {
    let mut h = blake3::Hasher::new();
    h.update(part.model_id.0.as_bytes());
    h.update(&[0]);
    h.update(&part.index.to_le_bytes());
    h.update(&node.0);
    *h.finalize().as_bytes()
}

/// Is `node` among the first `keep` of `part`'s `holders` in rendezvous order
/// ([`part_rank_weight`])? Every computer reading the same holders ranks them
/// alike, so when copies are given back for a hole (`prune::spare_for_a_hole`)
/// the first `keep` never give theirs back, and the part keeps them however
/// many holders decide in the same minute. Without it, each of three full
/// holders could count three copies, one to spare, and give its own back at
/// once. Rendezvous (highest-random-weight) hashing is how CRUSH-style
/// placement agrees on replicas with no coordinator.
pub(super) fn kept_by_rank(part: &ShardId, holders: &[NodeId], node: &NodeId, keep: usize) -> bool {
    let mut ranked: Vec<&NodeId> = holders.iter().collect();
    ranked.sort_by_key(|n| std::cmp::Reverse(part_rank_weight(part, n)));
    ranked.dedup();
    ranked.into_iter().take(keep).any(|n| n == node)
}

/// The layer ranges of `parts` of `manifest`.
fn ranges_of(manifest: &ModelManifest, parts: &HashSet<u32>) -> Vec<(u32, u32)> {
    manifest
        .shards
        .iter()
        .filter(|s| parts.contains(&s.index))
        .map(|s| s.layer_range)
        .collect()
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::Ordering::Relaxed;
    use std::sync::Arc;

    use super::super::test_support::{
        capability, fetched, holds, make_test_manager_with_config, offered, peer_with_disk,
        register_manifest_with_sized_shards, write_sparse_shards,
    };
    use super::super::AutoShardManager;
    use crate::config::Config;
    use crate::daemon::SharedState;
    use crate::types::{ModelId, NodeId, ShardId};

    const MIB: u64 = 1024 * 1024;
    const PART: u64 = 500 * MIB;
    /// What a holder's admission charges: 300 MB + 200 MB a layer, 6700 MB for
    /// all 32 — the reported 9B's shape.
    const CURVE: (u64, u64) = (300, 200);
    /// Room for `layers` of it, by that curve.
    fn ceiling_for(layers: u64) -> u64 {
        CURVE.0 + CURVE.1 * layers
    }

    fn peer(state: &Arc<SharedState>, byte: u8, ceiling_mb: Option<u64>) -> NodeId {
        peer_with_disk(state, byte, ceiling_mb, 100_000)
    }

    /// The report's shape: a 32-layer model in four 8-layer parts, held whole by
    /// one peer whose ceiling holds 24 of its layers, and asked for in ANOTHER
    /// region (the reporter's requests, seen by a card elsewhere). Demand here
    /// would raise the routine replica target past one copy; demand elsewhere
    /// leaves it at one, so routine replication is satisfied and only carrying
    /// can move anything.
    fn the_reported_shape(
        local_ceiling_layers: u64,
    ) -> (Arc<SharedState>, AutoShardManager, ModelId, NodeId) {
        let mut config = Config::default();
        config.auto_manage.min_replicas = 1;
        let (state, manager) = make_test_manager_with_config(config);
        state.models.auto_manage_default_model_cap.store(0, Relaxed);
        let mid = register_manifest_with_sized_shards(
            &state,
            "carry-me",
            32,
            &[(0, 8), (8, 16), (16, 24), (24, 32)],
            PART,
        );
        state
            .model_process_pool
            .test_cost_curve
            .insert(mid.clone(), CURVE);
        let small = peer(&state, 2, Some(ceiling_for(24)));
        holds(&state, &mid, &small, &[0, 1, 2, 3]);
        state.local_capability.store(Some(Arc::new(capability(
            Some(ceiling_for(local_ceiling_layers)),
            100_000,
        ))));
        state
            .region_demand
            .insert((mid.clone(), "BE".to_string()), 2.0);
        // Vetted for automatic adoption, as the HuggingFace watcher marks a
        // popular curator's upload (the reported 9B is unsloth's): carrying
        // goes through the trust gate like any other adoption.
        let mut trust = crate::types::ModelTrustInfo::new_discovered();
        trust.trust_level = crate::types::ModelTrustLevel::DemandVerified;
        state.models.model_trust.insert(mid.clone(), trust);
        (state, manager, mid, small)
    }

    /// The machine that can carry more of the model is the one chosen — not
    /// the holder that cannot run it, and not a machine with no room.
    #[test]
    fn a_model_its_holders_cannot_carry_gets_a_carrier_that_can() {
        // This node could hold none of it (its ceiling is under the fixed cost).
        let (state, manager, mid, _small) = the_reported_shape(0);
        let manifest = state.model_registry.get_manifest(&mid).unwrap();
        assert_eq!(manager.carried_layers(&manifest, None), Some(24));
        let big = peer(&state, 3, Some(ceiling_for(32)));
        assert!(manager.needs_carrier(&manifest));
        assert_eq!(manager.carrier_for(&manifest), Some(big));
    }

    /// The chosen carrier is offered what it lacks in model order until the
    /// SHORTFALL is closed — not everything its ceiling has room for ("no
    /// machine holds the whole model") — and never past its room; it starts
    /// fetching at once. With nobody asking for the model nothing is offered
    /// (the null control).
    #[test]
    fn the_chosen_carrier_fetches_only_what_closes_the_shortfall() {
        let (state, manager, mid, small) = the_reported_shape(16);
        // A machine that could run the model whole, as the report's cards
        // could — with no disk, so it is never the one chosen. Without one
        // the model is a split's, placed by a chain (`near_chain`, #346).
        peer_with_disk(&state, 6, Some(ceiling_for(32)), 0);
        assert_eq!(
            offered(&manager, &state),
            vec![0],
            "24 of 32 carried: 8 layers short, so part 0 alone — not the 16 it has room for"
        );
        assert_eq!(fetched(&manager, &state), vec![0]);

        // The holder could carry only 8: 24 short, and this node's room (16)
        // is what bounds it.
        state
            .peer_registry
            .get_mut(&small)
            .unwrap()
            .capability
            .as_mut()
            .unwrap()
            .model_memory_ceiling_mb = Some(ceiling_for(8));
        manager.carry_curves.clear();
        assert_eq!(
            offered(&manager, &state),
            vec![0, 1],
            "room for 16: parts 0 and 1"
        );

        state.region_demand.remove(&(mid, "BE".to_string()));
        assert!(
            offered(&manager, &state).is_empty(),
            "nobody asked for it: routine replication is satisfied, nothing is offered"
        );
    }

    /// Prune keeps a copy without which the model's holders could not carry it,
    /// and sheds it once another holder can — the same answer the download pass
    /// asked before fetching it, so a carried part is never shed and refetched.
    /// Here no machine could run the model whole, so the keep is the far
    /// chain's (this node 0-16, the holder 16-32; #346), at urgent pressure as
    /// carrying's is; once one could, it is carrying's question again.
    #[test]
    fn prune_keeps_the_copies_that_carry_a_model_until_another_holder_can() {
        let (state, manager, mid, small) = the_reported_shape(16);
        let local = state.identity.node_id().clone();
        holds(&state, &mid, &local, &[0, 1]);
        write_sparse_shards(&state, &mid, [0, 1], PART);
        let part0 = ShardId {
            model_id: mid.clone(),
            index: 0,
        };
        // A third holder of those parts that has no room to carry any of them:
        // three copies, which prune would thin like any other part.
        let roomless = peer(&state, 5, Some(ceiling_for(0)));
        holds(&state, &mid, &roomless, &[0, 1]);
        let holders = vec![small.clone(), roomless.clone(), local.clone()];
        assert!(
            !manager.would_shed_copy(&part0, &holders, 0.99, &[]),
            "24 + 16 layers carry the model; 24 alone do not"
        );

        // A second peer that could carry the whole model holds it too.
        let big = peer(&state, 3, Some(ceiling_for(32)));
        holds(&state, &mid, &big, &[0, 1, 2, 3]);
        let holders = vec![small, roomless, big, local];
        manager.forget_planned_chains();
        assert!(
            manager.would_shed_copy(&part0, &holders, 0.99, &[]),
            "the model is carried without this copy: prune treats it like any other"
        );
    }

    /// Unknown is never "short": a holder advertising no ceiling (older than
    /// v0.3.230) leaves everything as it was.
    #[test]
    fn a_holder_with_no_ceiling_changes_nothing() {
        let (state, manager, mid, _small) = the_reported_shape(32);
        let manifest = state.model_registry.get_manifest(&mid).unwrap();
        let older = peer(&state, 4, None);
        holds(&state, &mid, &older, &[0]);
        assert_eq!(manager.carried_layers(&manifest, None), None);
        assert!(!manager.needs_carrier(&manifest));
        assert!(offered(&manager, &state).is_empty());
    }

    /// A carrier that makes no progress — its own budget will not take the
    /// plan, and nothing it gossips says so — is passed over after
    /// `CARRIER_PATIENCE` by the same ranking, and the next machine carries.
    /// One that gains a part keeps its lease.
    #[test]
    fn a_carrier_that_makes_no_progress_is_passed_over() {
        let (state, manager, mid, small) = the_reported_shape(0);
        let manifest = state.model_registry.get_manifest(&mid).unwrap();
        // The holder carries 8 of 32: 24 short, so one part is progress, not
        // the end of carrying.
        state
            .peer_registry
            .get_mut(&small)
            .unwrap()
            .capability
            .as_mut()
            .unwrap()
            .model_memory_ceiling_mb = Some(ceiling_for(8));
        peer(&state, 3, Some(ceiling_for(32)));
        peer(&state, 4, Some(ceiling_for(32)));
        let first = manager.carrier_for(&manifest).expect("a carrier");

        // Progress keeps the lease: it now holds part 0.
        holds(&state, &mid, &first, &[0]);
        // Past its patience, without backdating a clock a just-booted host
        // cannot represent.
        let aged = |manager: &AutoShardManager| {
            let mut lease = manager.carrier_leases.get_mut(&mid).unwrap();
            lease.lapses_at = std::time::Instant::now();
        };
        aged(&manager);
        assert_eq!(
            manager.carrier_for(&manifest).as_ref(),
            Some(&first),
            "it gained a part"
        );

        // No progress since: passed over, and the other machine carries.
        aged(&manager);
        let next = manager.carrier_for(&manifest).expect("the next carrier");
        assert_ne!(next, first);
        assert_eq!(manager.carrier_for(&manifest), Some(next), "and keeps it");
    }

    /// In private mode a machine outside the pool is neither chosen to carry
    /// the pool's model nor counted toward whether it could be carried.
    #[test]
    fn a_machine_outside_the_pool_is_never_the_carrier() {
        let (state, manager, mid, _small) = the_reported_shape(0);
        let manifest = state.model_registry.get_manifest(&mid).unwrap();
        let outsider = peer(&state, 3, Some(ceiling_for(32)));
        assert_eq!(
            manager.carrier_for(&manifest),
            Some(outsider),
            "control: open swarm"
        );

        state.credits.private_mode.store(true, Relaxed);
        assert!(!manager.needs_carrier(&manifest));
        assert_eq!(manager.carrier_for(&manifest), None);
    }
}
