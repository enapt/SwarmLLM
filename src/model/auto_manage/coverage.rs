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
//! layers its live holders could ever hold between them — each counted at the
//! smaller of the layers it holds and its advertised ceiling
//! (`NodeCapability::model_memory_ceiling_mb`, weighed with the model's own
//! admission curve), the arithmetic the planner's shortage report uses
//! (`scheduler::layers_carried`). When that falls short of the model:
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
//! all: a model nobody can run must not pull a copy onto every machine.
//! Unknown — no header here, a holder or candidate with no ceiling (older than
//! v0.3.230) — is never "short": everything behaves as before.

use std::collections::{HashMap, HashSet};

use crate::types::{ModelId, ModelManifest, NodeId, ShardId, MMPROJ_SHARD_INDEX};

use super::AutoShardManager;

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

    /// Has anyone asked for this model — regional demand gossiped by any node,
    /// or a request routed by this one since the last decay?
    fn asked_for(&self, model_id: &ModelId) -> bool {
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
    /// and the swarm able to carry it?
    pub(super) fn needs_carrier(&self, manifest: &ModelManifest) -> bool {
        if manifest.num_layers == 0 || !self.asked_for(&manifest.id) {
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
        ranked.sort_by_key(|node| {
            let mut h = blake3::Hasher::new();
            h.update(manifest.id.0.as_bytes());
            h.update(&node.0);
            std::cmp::Reverse(*h.finalize().as_bytes())
        });
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

    use super::super::scoring::BudgetReport;
    use super::super::test_support::{
        make_test_manager_with_config, register_manifest_with_sized_shards, write_sparse_shards,
    };
    use super::super::{AutoShardManager, DiskSpace, StorageReading};
    use crate::config::Config;
    use crate::daemon::SharedState;
    use crate::types::{ContributionMode, ModelId, NodeCapability, NodeId, ShardId};

    const MIB: u64 = 1024 * 1024;
    const PART: u64 = 500 * MIB;
    /// What a holder's admission charges: 300 MB + 200 MB a layer, 6700 MB for
    /// all 32 — the reported 9B's shape.
    const CURVE: (u64, u64) = (300, 200);
    /// Room for `layers` of it, by that curve.
    fn ceiling_for(layers: u64) -> u64 {
        CURVE.0 + CURVE.1 * layers
    }

    fn capability(ceiling_mb: Option<u64>, disk_mb: u64) -> NodeCapability {
        NodeCapability {
            coord: None,
            node_id: NodeId([0u8; 32]),
            gpu: None,
            cpu: None,
            ram_total_mb: 0,
            ram_available_mb: 0,
            ram_model_budget_mb: None,
            disk_available_mb: disk_mb,
            bandwidth_mbps: 0.0,
            hosted_shards: vec![],
            max_contribution: crate::types::ContributionLevel::Moderate,
            uptime_seconds: 0,
            version: String::new(),
            region: None,
            est_tokens_per_sec_7b: 0.0,
            os: None,
            observed_latencies: vec![],
            relay_capable: false,
            protocol_version: 0,
            features: 0,
            relay_reservations: vec![],
            anchor_mode: false,
            can_serve_inference: true,
            resident_layers: Vec::new(),
            context_ceiling_tokens: None,
            model_load_ms_per_gib: None,
            model_memory_ceiling_mb: ceiling_mb,
        }
    }

    fn peer(state: &Arc<SharedState>, byte: u8, ceiling_mb: Option<u64>) -> NodeId {
        peer_with_disk(state, byte, ceiling_mb, 100_000)
    }

    fn peer_with_disk(
        state: &Arc<SharedState>,
        byte: u8,
        ceiling_mb: Option<u64>,
        disk_mb: u64,
    ) -> NodeId {
        let id = NodeId([byte; 32]);
        state.peer_registry.insert(
            id.clone(),
            crate::types::PeerInfo {
                node_id: id.clone(),
                addresses: vec![],
                capability: Some(capability(ceiling_mb, disk_mb)),
                last_seen: chrono::Utc::now(),
                latency_ms: Some(10),
                trust_score: 0.8,
                peer_id_bytes: None,
                ack_srtt_ms: None,
                active_request_count: 0,
                first_seen: 0,
                verified_transaction_count: 0,
                is_lan_peer: false,
                goodput_bytes_per_sec: None,
                goodput_samples: 0,
            },
        );
        state.connected_node_ids.insert(id.clone());
        id
    }

    fn holds(state: &Arc<SharedState>, mid: &ModelId, node: &NodeId, parts: &[u32]) {
        for &index in parts {
            state.model_registry.record_shard_holder(
                ShardId {
                    model_id: mid.clone(),
                    index,
                },
                node.clone(),
            );
        }
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

    /// The parts the download pass offers, in index order.
    fn offered(manager: &AutoShardManager, state: &Arc<SharedState>) -> Vec<u32> {
        let local = state.identity.node_id().clone();
        let mut parts: Vec<u32> = manager
            .gather_candidates(&local, 0)
            .into_iter()
            .map(|c| c.shard_index)
            .collect();
        parts.sort_unstable();
        parts
    }

    /// The parts one cycle of the download pass actually starts.
    fn fetched(manager: &AutoShardManager, state: &Arc<SharedState>) -> Vec<u32> {
        let reading = StorageReading {
            max_storage_mb: 100_000,
            max_disk_mb: 500_000,
            contribution: ContributionMode::Maximum,
            disk: Some(DiskSpace {
                free_bytes: 400_000 * MIB,
                total_bytes: 1_000_000 * MIB,
            }),
            held_bytes: 0,
        };
        let report = BudgetReport {
            held_bytes: 0,
            held_shards: 0,
            budget: reading.budget(),
            max_shards: 0,
            max_shards_reached: false,
            reading,
        };
        let local = state.identity.node_id().clone();
        manager
            .select_within_budget(manager.gather_candidates(&local, 0), &report, 0)
            .into_iter()
            .map(|c| c.shard_index)
            .collect()
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
        assert_eq!(
            offered(&manager, &state),
            vec![0],
            "24 of 32 carried: 8 layers short, so part 0 alone — not the 16 it has room for"
        );
        assert_eq!(fetched(&manager, &state), vec![0]);

        // The holder could carry only 8: 24 short, and this node's room (16)
        // is what bounds it. A machine with room but no disk keeps the swarm
        // able to carry the model without being the one chosen to.
        peer_with_disk(&state, 6, Some(ceiling_for(32)), 0);
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
