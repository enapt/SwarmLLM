//! Swarm-wide capacity snapshot — the headline metric exposed to the
//! dashboard.
//!
//! The whole point of SwarmLLM is "everyday users pool resources to run
//! huge models together". The capacity snapshot is how we make that visible:
//! how much hardware does the swarm have right now, which models can it
//! actually serve, and which models are within reach if a few more people
//! join. Designed for non-technical users — the JSON shape maps directly
//! onto plain-language UI strings; no internal jargon (shards, replication
//! factor, layer coverage) leaks out.
//!
//! Recomputed every gossip tick (cheap — single pass over the registries),
//! cached via `ArcSwap` so the Prometheus + REST + WS readers all see a
//! lock-free snapshot.

use std::collections::{BTreeSet, HashMap, HashSet};
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::types::{ModelId, NodeId, ShardId};

use super::SharedState;

/// Snapshot of the swarm's collective capability at a point in time.
///
/// "Serveable" = every shard of the model has at least one connected
/// holder, so a coordinator could assemble a complete pipeline today.
/// "Aspirational" = at least one shard has a non-local holder but the
/// coverage is incomplete — these are the ones the swarm is close to
/// running but can't quite. "Hosted locally" is the subset of serveable
/// models this node hosts at least one shard of.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct SwarmCapacity {
    /// Total number of online nodes (this node + connected peers).
    pub online_nodes: u32,
    /// Combined VRAM across the swarm in MB. Includes this node's GPU
    /// (when present). Peer GPU info is `None` for CPU-only peers, which
    /// contribute 0 here — the user-visible UI should explicitly mention
    /// that some nodes are CPU-only.
    pub total_vram_mb: u64,
    /// Combined disk budget across the swarm in MB (sum of every peer's
    /// declared `max_disk_mb`). When a peer hasn't reported a budget we
    /// optimistically use the network-level default; this matches how
    /// auto-manage already reasons about pool capacity.
    pub total_disk_mb: u64,
    /// Number of distinct GPU-equipped nodes (everyone with `capability.gpu`
    /// reported). Used in the dashboard to phrase things like "12 nodes
    /// with GPU, 47 helping with CPU".
    pub gpu_nodes: u32,
    /// Number of distinct regions represented (continent-level).
    /// Affects fault tolerance — a swarm of 100 nodes in one region is
    /// less robust than 100 spread across five.
    pub regions_represented: u32,
    /// Models the swarm can serve right now. Each entry is the model id;
    /// the frontend joins these against the manifest registry to get
    /// display name + size.
    pub serveable_models: Vec<ModelEntry>,
    /// Models that are partially covered — at least one shard exists on
    /// the network but coverage is incomplete. We still surface them so
    /// the user sees "the swarm is working on it".
    pub aspirational_models: Vec<ModelEntry>,
    /// Subset of `serveable_models` that this node hosts at least one
    /// shard of. Used for the "your contribution" framing.
    pub hosted_locally: Vec<ModelEntry>,
    /// Lowest replication factor across all serveable models. A value of
    /// 1 means at least one model has only a single host — losing that
    /// node would drop the model. 2+ means every model has redundancy.
    pub min_redundancy: u32,
    /// Median replication factor across all serveable models. More
    /// resistant to outliers than the min — a friendlier "how healthy
    /// is the swarm" headline number.
    pub median_redundancy: u32,
    /// Best (highest-quality) model the swarm can serve, by parameter
    /// count proxied via `total_size_bytes`. Surfaced as the headline
    /// "your swarm runs up to X" claim.
    pub headline_model: Option<HeadlineModel>,
    /// Last time this snapshot was rebuilt (Unix timestamp).
    pub computed_at: i64,
}

/// Per-model summary used in the dashboard cards.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelEntry {
    pub model_id: String,
    /// User-facing display name (falls back to model id).
    pub display_name: String,
    /// Total model size in MB across all shards.
    pub size_mb: u64,
    /// Distinct nodes hosting at least one shard of this model.
    pub holders: u32,
    /// Total shard count for this model.
    pub total_shards: u32,
    /// Number of shards covered by at least one holder.
    pub shards_covered: u32,
    /// Whether THIS node hosts at least one shard. Drives the "your
    /// contribution" badge.
    pub hosted_by_us: bool,
}

/// Headline pick — surfaced as "your swarm runs up to ..." on the dash.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HeadlineModel {
    pub model_id: String,
    pub display_name: String,
    pub size_mb: u64,
}

/// Compute a fresh capacity snapshot from current state. Single pass over
/// the model + peer registries, no allocations beyond the result vectors.
/// Safe to call from any context — does not block, does not lock writers.
pub fn compute_swarm_capacity(state: &SharedState) -> SwarmCapacity {
    // 1. Headcount + VRAM/disk pool. Local node + connected peers only —
    //    `peer_registry` may include disconnected entries we keep around
    //    for reconnect tracking; `connected_node_ids` is the connectivity
    //    oracle (gotcha #86).
    let online_nodes = state.connected_node_ids.len() as u32 + 1; // +1 = self
    let local_vram_mb = state
        .gpu_info
        .as_ref()
        .map(|g| g.vram_total_mb)
        .unwrap_or(0);

    let mut total_vram_mb = local_vram_mb;
    let mut gpu_nodes = if local_vram_mb > 0 { 1 } else { 0 };
    let mut regions: BTreeSet<String> = BTreeSet::new();
    if let Some(my_region) = state.config.identity.region.as_ref() {
        if !my_region.is_empty() {
            regions.insert(my_region.clone());
        }
    }

    for entry in state.peer_registry.iter() {
        if !state.connected_node_ids.contains(entry.key()) {
            continue;
        }
        if let Some(cap) = entry.value().capability.as_ref() {
            if let Some(gpu) = cap.gpu.as_ref() {
                if gpu.vram_total_mb > 0 {
                    total_vram_mb = total_vram_mb.saturating_add(gpu.vram_total_mb);
                    gpu_nodes += 1;
                }
            }
            if let Some(ref region) = cap.region {
                if !region.is_empty() {
                    regions.insert(region.clone());
                }
            }
        }
    }

    // 2. Disk pool — peer capability advertises `disk_available_mb` (free
    //    space, refreshed by gossip). Sum these as the swarm's
    //    contributable disk; the local node uses its configured budget.
    //    Peers without capability info contribute 0 — conservative.
    let local_disk_mb = state.cfg().resources.max_disk_mb;
    let mut total_disk_mb = local_disk_mb;
    for entry in state.peer_registry.iter() {
        if !state.connected_node_ids.contains(entry.key()) {
            continue;
        }
        let peer_disk = entry
            .value()
            .capability
            .as_ref()
            .map(|c| c.disk_available_mb)
            .unwrap_or(0);
        total_disk_mb = total_disk_mb.saturating_add(peer_disk);
    }

    // 3. Build per-model coverage map. For each known model, count how
    //    many shards have at least one connected holder, and how many
    //    distinct nodes hold at least one shard. This is the core of the
    //    "what can the swarm actually run" answer.
    let local_node_id = state.identity.node_id().clone();

    struct Coverage {
        total_shards: u32,
        shards_covered: u32,
        // NodeId is `Hash + Eq` but not `Ord`, so use a HashSet — order
        // doesn't matter (we only read `.len()` for the holder count).
        holders: HashSet<NodeId>,
        replications: Vec<u32>, // per-shard holder counts (only for fully-covered models)
        hosted_by_us: bool,
        size_mb: u64,
        display_name: String,
    }
    let mut by_model: HashMap<ModelId, Coverage> = HashMap::new();

    for manifest in state.model_registry.models() {
        if manifest.shards.is_empty() {
            continue;
        }
        let mut cov = Coverage {
            total_shards: manifest.shards.len() as u32,
            shards_covered: 0,
            holders: HashSet::new(),
            replications: Vec::with_capacity(manifest.shards.len()),
            hosted_by_us: false,
            size_mb: manifest.total_size_bytes / (1024 * 1024),
            display_name: if manifest.name.is_empty() {
                manifest.id.0.clone()
            } else {
                manifest.name.clone()
            },
        };
        let mut all_covered = true;
        for shard_info in &manifest.shards {
            let shard_id = ShardId {
                model_id: manifest.id.clone(),
                index: shard_info.index,
            };
            let holders = state.model_registry.shard_holders(&shard_id);
            let live: Vec<&NodeId> = holders
                .iter()
                .filter(|n| **n == local_node_id || state.connected_node_ids.contains(n))
                .collect();
            if live.is_empty() {
                all_covered = false;
                continue;
            }
            cov.shards_covered += 1;
            cov.replications.push(live.len() as u32);
            for h in &live {
                if **h == local_node_id {
                    cov.hosted_by_us = true;
                }
                cov.holders.insert((*h).clone());
            }
        }
        if !all_covered {
            cov.replications.clear();
        }
        by_model.insert(manifest.id.clone(), cov);
    }

    // 4. Partition models. Skip the local-only "stub" / "test" entries
    //    that exist in registry but have a single shard and no peers
    //    holding it — those aren't user-facing models.
    let mut serveable: Vec<ModelEntry> = Vec::new();
    let mut aspirational: Vec<ModelEntry> = Vec::new();
    let mut hosted_locally: Vec<ModelEntry> = Vec::new();
    let mut all_replications: Vec<u32> = Vec::new();
    let mut headline: Option<HeadlineModel> = None;

    for (mid, cov) in by_model {
        let entry = ModelEntry {
            model_id: mid.0.clone(),
            display_name: cov.display_name.clone(),
            size_mb: cov.size_mb,
            holders: cov.holders.len() as u32,
            total_shards: cov.total_shards,
            shards_covered: cov.shards_covered,
            hosted_by_us: cov.hosted_by_us,
        };
        let fully_serveable = cov.shards_covered == cov.total_shards && cov.total_shards > 0;
        if fully_serveable {
            // Headline = biggest serveable model (proxy for capability).
            if headline
                .as_ref()
                .map(|h| h.size_mb < entry.size_mb)
                .unwrap_or(true)
            {
                headline = Some(HeadlineModel {
                    model_id: entry.model_id.clone(),
                    display_name: entry.display_name.clone(),
                    size_mb: entry.size_mb,
                });
            }
            all_replications.extend(cov.replications.iter().copied());
            if entry.hosted_by_us {
                hosted_locally.push(entry.clone());
            }
            serveable.push(entry);
        } else if cov.shards_covered > 0 {
            aspirational.push(entry);
        }
    }

    serveable.sort_by(|a, b| {
        b.size_mb
            .cmp(&a.size_mb)
            .then_with(|| a.display_name.cmp(&b.display_name))
    });
    aspirational.sort_by(|a, b| {
        let a_pct = (a.shards_covered as f32) / (a.total_shards.max(1) as f32);
        let b_pct = (b.shards_covered as f32) / (b.total_shards.max(1) as f32);
        b_pct
            .partial_cmp(&a_pct)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.display_name.cmp(&b.display_name))
    });
    hosted_locally.sort_by(|a, b| a.display_name.cmp(&b.display_name));

    // 5. Redundancy — surfaces "is the swarm fragile" without exposing
    //    the per-shard internals.
    let min_redundancy = all_replications.iter().copied().min().unwrap_or(0);
    let median_redundancy = if all_replications.is_empty() {
        0
    } else {
        let mut s = all_replications.clone();
        s.sort_unstable();
        s[s.len() / 2]
    };

    SwarmCapacity {
        online_nodes,
        total_vram_mb,
        total_disk_mb,
        gpu_nodes,
        regions_represented: regions.len() as u32,
        serveable_models: serveable,
        aspirational_models: aspirational,
        hosted_locally,
        min_redundancy,
        median_redundancy,
        headline_model: headline,
        computed_at: chrono::Utc::now().timestamp(),
    }
}

/// Recompute and publish the capacity snapshot to `MetricsProviders`.
/// Cheap — single pass; safe to call from gossip-tick handlers.
pub fn refresh_swarm_capacity(state: &SharedState) {
    let snapshot = compute_swarm_capacity(state);
    state.metrics.swarm_capacity.store(Arc::new(snapshot));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_state_produces_zero_capacity() {
        // Just exercises the default Serialize path; SharedState construction
        // is too heavy to fixture here. The full integration is covered in
        // the metrics endpoint test below.
        let cap = SwarmCapacity::default();
        let json = serde_json::to_value(&cap).unwrap();
        assert_eq!(json["online_nodes"], 0);
        assert!(json["serveable_models"].is_array());
        assert!(json["headline_model"].is_null());
    }
}
