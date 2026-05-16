use std::sync::Arc;

use crate::daemon::SharedState;
use crate::error::SwarmError;
use crate::types::{
    ModelId, ModelManifest, NodeId, PipelineAssignment, PipelineSegment, ShardId,
    TensorParallelGroup,
};

/// PipelineScheduler assembles a distributed inference pipeline
/// by selecting the best nodes for each layer range.
#[derive(Clone)]
pub struct PipelineScheduler {
    shared_state: Arc<SharedState>,
}

/// A candidate node for layer ranges, with scoring metadata.
/// A single node may advertise multiple non-contiguous layer ranges (e.g.,
/// layers [0,2) and [10,14)) when the GGUF's alphabetical tensor ordering
/// scatters layers across byte-range shards.
#[derive(Debug, Clone)]
struct NodeCandidate {
    node_id: NodeId,
    shard_id: ShardId,
    /// All contiguous layer ranges this node can serve for the model.
    available_ranges: Vec<(u32, u32)>,
    latency_ms: u32,
    load: f32,
    trust_score: f32,
    /// True if this node has shard 0 (token_embd.weight, needed for is_first).
    can_be_first: bool,
    /// True if this node has the final shard (output head, needed for is_last).
    can_be_last: bool,
    /// Region proximity score: 1.0 same region, 0.5 adjacent, 0.2 distant, 0.7 unknown.
    region_score: f32,
    /// Estimated tokens/s for a 7B Q4 model (from NodeCapability). 0 = unknown.
    est_tokens_per_sec: f32,
    /// Observed per-layer latency EMA (ms/layer) for remote segments this peer
    /// served for us. None = no samples yet, use `est_tokens_per_sec` as proxy.
    /// Populated by `state.record_peer_segment_latency` after successful
    /// `forward_through_segments` hops. Used by the Parallax routing DP to
    /// replace the static capability estimate with live signal when available.
    observed_latency_ms_per_layer: Option<f32>,
    /// True if this node is in our device pool (preferred for routing — free, trusted, low latency).
    is_pool_member: bool,
}

/// Maximum number of GPUs in a tensor-parallel group. AllReduce communication
/// between layers requires low latency, so groups are bounded to LAN-class
/// peers; 4 keeps the all-reduce ring small enough for sub-millisecond
/// per-token sync on a single switch.
const MAX_TP_GROUP_SIZE: usize = 4;

/// Static adjacency table for adjacent regions (0.5 score).
/// These pairs represent geographically close countries where cross-region
/// latency is typically acceptable for inference.
const ADJACENT_REGIONS: &[(&str, &str)] = &[
    ("US", "CA"),
    ("US", "MX"),
    ("DE", "FR"),
    ("DE", "NL"),
    ("DE", "AT"),
    ("DE", "CH"),
    ("DE", "PL"),
    ("FR", "ES"),
    ("FR", "IT"),
    ("FR", "BE"),
    ("GB", "IE"),
    ("GB", "FR"),
    ("GB", "NL"),
    ("JP", "KR"),
    ("JP", "TW"),
    ("AU", "NZ"),
    ("SE", "NO"),
    ("SE", "FI"),
    ("SE", "DK"),
    ("BR", "AR"),
    ("SG", "MY"),
    ("IN", "BD"),
];

/// Check if two regions are adjacent.
fn regions_adjacent(a: &str, b: &str) -> bool {
    ADJACENT_REGIONS.iter().any(|(x, y)| {
        (x.eq_ignore_ascii_case(a) && y.eq_ignore_ascii_case(b))
            || (x.eq_ignore_ascii_case(b) && y.eq_ignore_ascii_case(a))
    })
}

impl PipelineScheduler {
    pub fn new(shared_state: Arc<SharedState>) -> Self {
        Self { shared_state }
    }

    /// Assemble a pipeline for the given model.
    ///
    /// Algorithm (from spec):
    /// 1. Fetch model manifest from registry
    /// 2. Determine required layer ranges (0..num_layers)
    /// 3. Query model_registry.shard_holders for all nodes hosting shards of this model
    /// 4. For each node, fetch current load and latency from peer_registry
    /// 5. Greedy assignment: sort candidates by (latency ASC, load ASC, trust DESC),
    ///    assign the best available node covering the widest contiguous layer range
    /// 6. If any layer range has no available node -> fail
    /// 7. Identify standby nodes for each segment
    /// 8. Return PipelineAssignment
    #[cfg(test)]
    pub(crate) fn assemble_pipeline(
        &self,
        model_id: &ModelId,
        local_node_id: &NodeId,
    ) -> Result<PipelineAssignment, SwarmError> {
        self.assemble_pipeline_for(model_id, local_node_id, uuid::Uuid::new_v4())
    }

    /// Assemble a pipeline for the given model with a specific request ID.
    pub fn assemble_pipeline_for(
        &self,
        model_id: &ModelId,
        local_node_id: &NodeId,
        request_id: uuid::Uuid,
    ) -> Result<PipelineAssignment, SwarmError> {
        let manifest = self
            .shared_state
            .model_registry
            .get_manifest(model_id)
            .ok_or_else(|| SwarmError::ModelNotAvailable(model_id.clone()))?;

        let num_layers = manifest.num_layers;
        if num_layers == 0 {
            return Err(SwarmError::PipelineError(
                "Model has zero layers".to_string(),
            ));
        }

        let start = std::time::Instant::now();

        // Check if encrypted pipeline is enabled for this model (per-model → global fallback)
        let encrypted = self
            .shared_state
            .encrypted_pipeline_models
            .get(model_id)
            .map(|r| *r.value())
            .unwrap_or(self.shared_state.config.inference.encrypted_pipeline);
        if encrypted {
            tracing::info!(
                model = %model_id,
                "Encrypted pipeline active — forcing first+last segments to local node"
            );
        }

        // Gather all candidates: nodes that have shards for this model
        let candidates = self.gather_candidates(&manifest, local_node_id);
        if candidates.is_empty() {
            // In private mode, give a specific error showing which shards are missing
            if self
                .shared_state
                .credits
                .private_mode
                .load(std::sync::atomic::Ordering::Relaxed)
            {
                // Find which shards no allowed node holds. R134.7: also fold in
                // cross-pool extras so the error message matches what
                // gather_candidates considered eligible.
                let mut allowed =
                    crate::pool::scope::allowed_node_set(&self.shared_state).unwrap_or_default();
                allowed.extend(crate::pool::scope::cross_pool_extras(
                    &self.shared_state,
                    &manifest.id,
                ));
                let missing: Vec<u32> = manifest
                    .shards
                    .iter()
                    .filter(|s| {
                        let sid = ShardId {
                            model_id: manifest.id.clone(),
                            index: s.index,
                        };
                        let holders = self.shared_state.model_registry.shard_holders(&sid);
                        !holders.iter().any(|h| allowed.contains(h))
                    })
                    .map(|s| s.index)
                    .collect();
                return Err(SwarmError::PrivateModeUnavailable {
                    model_id: manifest.name.clone(),
                    missing_shards: missing,
                });
            }
            return Err(SwarmError::InsufficientCapacity(model_id.clone()));
        }

        // Fast path: if the local node has full layer coverage (0..num_layers),
        // run entirely locally without involving remote peers.  This prevents
        // "Segment N failed with no standby" errors caused by remote peers that
        // hold overlapping shards being pulled into the pipeline unnecessarily.
        if let Some(local_cand) = candidates.iter().find(|c| {
            c.node_id == *local_node_id
                && c.available_ranges
                    .iter()
                    .any(|r| r.0 == 0 && r.1 >= num_layers)
        }) {
            tracing::info!(
                model = %model_id,
                num_layers,
                "Local node has full layer coverage — single local segment (no remote peers)"
            );
            let segment = PipelineSegment {
                node_id: local_node_id.clone(),
                shard_id: local_cand.shard_id.clone(),
                layer_range: (0, num_layers),
            };
            // Still detect TP groups — LAN peers covering the same range
            // can participate in tensor parallelism even in single-segment mode.
            let tp_groups = self.detect_tp_groups(std::slice::from_ref(&segment), &candidates);
            return Ok(PipelineAssignment {
                request_id,
                segments: vec![segment],
                standbys: vec![],
                tp_groups,
                supports_speculative: true,
            });
        }

        // Distributed layer assignment: prefer Parallax shortest-path DP when
        // enabled; fall back to greedy on any failure (disjoint ranges, no
        // valid source/sink, etc.) so routing never regresses below greedy.
        let raw_segments = if self.shared_state.config.inference.parallax_routing {
            match parallax::route_shortest_path(num_layers, &candidates, local_node_id, encrypted) {
                Ok(segs) => {
                    tracing::debug!(
                        model = %model_id,
                        segments = segs.len(),
                        "DIAG: parallax routing selected chain"
                    );
                    segs
                }
                Err(e) => {
                    tracing::debug!(
                        model = %model_id,
                        err = %e,
                        "parallax routing unavailable — falling back to greedy"
                    );
                    self.greedy_assign(num_layers, &candidates, encrypted)?
                }
            }
        } else {
            self.greedy_assign(num_layers, &candidates, encrypted)?
        };

        // Merge contiguous segments on the same node into a single segment.
        // This avoids sending multiple LayerForward messages to the same node
        // when it handles its full layer range in one forward pass.
        let segments = Self::merge_contiguous(raw_segments);

        // Identify standby nodes for each segment
        let standbys = self.find_standbys(&segments, &candidates);

        // Detect tensor-parallel opportunities: LAN peers sharing the same layer range.
        // Skip TP when encrypted pipeline is active — no remote node should process
        // tensor data in encrypted mode (defeats the purpose of local-only embedding/sampling).
        let tp_groups = if encrypted {
            vec![]
        } else {
            self.detect_tp_groups(&segments, &candidates)
        };

        tracing::info!(
            request_id = %request_id,
            model = %model_id,
            candidates_count = candidates.len(),
            segments = segments.len(),
            standbys = standbys.len(),
            tp_groups = tp_groups.len(),
            elapsed_ms = start.elapsed().as_millis() as u64,
            "DIAG: assemble_pipeline_for completed"
        );

        Ok(PipelineAssignment {
            request_id,
            segments,
            standbys,
            tp_groups,
            // All current nodes advertise speculative verify-batch support. Will
            // flip to a per-peer capability check once we gate on version.
            supports_speculative: true,
        })
    }

    /// Gather all candidate nodes for the given model's shards.
    ///
    /// Groups shards by node and computes combined layer ranges using actual GGUF
    /// tensor metadata when available, falling back to manifest layer_range otherwise.
    fn gather_candidates(
        &self,
        manifest: &ModelManifest,
        local_node_id: &NodeId,
    ) -> Vec<NodeCandidate> {
        // Private mode: compute allowed node set (None = unrestricted).
        // R134.7: when `allow_cross_pool_inference` is on and the local pool
        // can't serve this model, union the cross-pool extras into the
        // allowed set. No-op when both flags aren't on or when a local pool
        // member already holds the model.
        let allowed_set = {
            let base = crate::pool::scope::allowed_node_set(&self.shared_state);
            let extras = crate::pool::scope::cross_pool_extras(&self.shared_state, &manifest.id);
            match (base, extras.is_empty()) {
                (None, _) => None,
                (Some(set), true) => Some(set),
                (Some(set), false) => {
                    let mut merged = set;
                    merged.extend(extras);
                    Some(merged)
                }
            }
        };

        // Build set of pool member NodeIds for preferred routing.
        // Pool devices are trusted, free (no credit cost), and usually low latency.
        let pool_member_ids: std::collections::HashSet<NodeId> = {
            if let Ok(ps) = self.shared_state.credits.pool_state.try_read() {
                ps.as_ref()
                    .map(|s| s.members.iter().map(|m| m.node_id.clone()).collect())
                    .unwrap_or_default()
            } else {
                std::collections::HashSet::new()
            }
        };

        // First, collect which shard indices each node holds
        let mut node_shards: std::collections::HashMap<NodeId, Vec<u32>> =
            std::collections::HashMap::new();

        for shard in &manifest.shards {
            let shard_id = ShardId {
                model_id: manifest.id.clone(),
                index: shard.index,
            };
            let holders = self.shared_state.model_registry.shard_holders(&shard_id);
            for node_id in holders {
                // Private mode: skip nodes outside the allowed set
                if let Some(ref allowed) = allowed_set {
                    if !allowed.contains(&node_id) {
                        continue;
                    }
                }
                // Skip peers we can't currently reach. Two stale-source paths:
                // (1) The DHT periodically re-injects stale providers (peers
                //     that recently disconnected but whose Kademlia provider
                //     records haven't expired yet) into shard_holders.
                // (2) When a peer disconnects mid-pipeline, peer_registry is
                //     intentionally preserved for reconnect attempts (see
                //     handle_connection_closed `in_active_pipeline` branch),
                //     but the libp2p `connected_node_ids` set is cleared
                //     unconditionally — making it the right liveness oracle.
                // Without this filter, the scheduler picks a dead peer,
                // remote-generate sends to it, and the request hangs until
                // the 120s first-token timeout.
                let is_local = node_id == *local_node_id;
                if !is_local && !self.shared_state.connected_node_ids.contains(&node_id) {
                    continue;
                }
                node_shards.entry(node_id).or_default().push(shard.index);
            }
        }

        let mut candidates = Vec::new();

        for (node_id, mut shard_indices) in node_shards {
            shard_indices.sort();

            // Compute ALL contiguous layer ranges for this node's shards
            // from manifest layer_range data.
            let ranges = {
                let manifest_ranges = crate::inference::split::available_layer_ranges_from_manifest(
                    manifest,
                    &shard_indices,
                );
                if !manifest_ranges.is_empty() {
                    manifest_ranges
                        .into_iter()
                        .map(|(s, e)| (s as u32, e as u32))
                        .collect::<Vec<_>>()
                } else {
                    // Fallback: use manifest layer ranges (approximate, single range)
                    let mut ls = manifest.num_layers as usize;
                    let mut le = 0usize;
                    for &idx in &shard_indices {
                        if let Some(shard) = manifest.shards.iter().find(|s| s.index == idx) {
                            ls = ls.min(shard.layer_range.0 as usize);
                            le = le.max(shard.layer_range.1 as usize);
                        }
                    }
                    if ls < le {
                        vec![(ls as u32, le as u32)]
                    } else {
                        vec![]
                    }
                }
            };

            if ranges.is_empty() {
                continue; // No complete layers on this node
            }

            let first_shard_id = ShardId {
                model_id: manifest.id.clone(),
                index: shard_indices[0],
            };
            let (latency_ms, trust_score) = self.get_peer_metrics(&node_id, local_node_id);

            // Determine if this node can serve as first/last segment
            let can_be_first = shard_indices.contains(&0);
            let last_shard_idx = manifest.shard_count.saturating_sub(1);
            let can_be_last = shard_indices.contains(&last_shard_idx);

            // Use the most up-to-date load info: for local node, use active_pipelines
            // directly. For remote nodes, take the max of health-ping report and local
            // pipeline tracking (health pings can be stale by up to ~5s).
            let active_load = if &node_id == local_node_id {
                self.shared_state.active_pipelines.len() as f32
            } else {
                let health_ping_load = self
                    .shared_state
                    .peer_registry
                    .get(&node_id)
                    .map(|p| p.active_request_count as f32)
                    .unwrap_or(0.0);
                let local_pipeline_load = self
                    .shared_state
                    .active_pipelines
                    .iter()
                    .filter(|entry| entry.value().segments.iter().any(|s| s.node_id == node_id))
                    .count() as f32;
                health_ping_load.max(local_pipeline_load)
            };

            // Compute region_score: 1.0 same, 0.5 adjacent, 0.2 distant, 0.7 unknown.
            let region_score = if &node_id == local_node_id {
                1.0 // Local node is always "same region"
            } else {
                self.compute_region_score(&node_id, local_node_id)
            };

            // Look up speed estimation from capability gossip
            let est_tokens_per_sec = if &node_id == local_node_id {
                // Local: compute directly from our GPU info
                self.shared_state
                    .gpu_info
                    .as_ref()
                    .map(|g| {
                        let bw =
                            crate::model::auto_manage::vram::gpu_memory_bandwidth_gbps(&g.name);
                        crate::model::auto_manage::vram::estimate_tokens_per_sec_7b(bw, true)
                    })
                    .unwrap_or(0.0)
            } else {
                self.shared_state
                    .peer_registry
                    .get(&node_id)
                    .map(|p| {
                        p.capability
                            .as_ref()
                            .map(|c| c.est_tokens_per_sec_7b)
                            .unwrap_or(0.0)
                    })
                    .unwrap_or(0.0)
            };

            let is_pool = pool_member_ids.contains(&node_id);
            let observed_latency_ms_per_layer = if &node_id == local_node_id {
                None
            } else {
                self.shared_state.observed_latency_ms_per_layer(&node_id)
            };
            candidates.push(NodeCandidate {
                node_id,
                shard_id: first_shard_id,
                available_ranges: ranges,
                latency_ms,
                load: active_load,
                trust_score,
                can_be_first,
                can_be_last,
                region_score,
                est_tokens_per_sec,
                observed_latency_ms_per_layer,
                is_pool_member: is_pool,
            });
        }

        // Log candidates for debugging
        for c in &candidates {
            tracing::debug!(
                node = %c.node_id,
                ranges = ?c.available_ranges,
                can_be_first = c.can_be_first,
                can_be_last = c.can_be_last,
                region_score = c.region_score,
                "Pipeline candidate"
            );
        }

        // Sort: latency ASC, region_score DESC (tiebreaker), load ASC, trust DESC.
        // Latency is the primary sort — we never sacrifice speed for region affinity.
        // Region breaks ties between nodes with similar latency, preventing
        // cross-continent routing when same-region alternatives exist.
        // Sort: pool members first (free + trusted), then by latency, region, load, trust, speed
        candidates.sort_by(|a, b| {
            b.is_pool_member
                .cmp(&a.is_pool_member) // true (1) > false (0) → pool members first
                .then_with(|| a.latency_ms.cmp(&b.latency_ms))
                .then_with(|| {
                    b.region_score
                        .partial_cmp(&a.region_score)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .then_with(|| {
                    a.load
                        .partial_cmp(&b.load)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .then_with(|| {
                    b.trust_score
                        .partial_cmp(&a.trust_score)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .then_with(|| {
                    // Speed tie-breaker: faster nodes (higher tokens/s) preferred
                    b.est_tokens_per_sec
                        .partial_cmp(&a.est_tokens_per_sec)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
        });

        tracing::debug!(
            candidates_count = candidates.len(),
            model = %manifest.id,
            latency_range = ?candidates.first().map(|c| c.latency_ms)
                ..=candidates.last().map(|c| c.latency_ms),
            "DIAG: gather_candidates complete"
        );

        candidates
    }

    /// Compute region proximity score for a remote node relative to us.
    /// Returns 1.0 same, 0.5 adjacent, 0.2 distant, 0.7 unknown.
    fn compute_region_score(&self, node_id: &NodeId, _local_node_id: &NodeId) -> f32 {
        // Get our region
        let our_region = if let Ok(guard) = self.shared_state.detected_region.try_read() {
            guard.as_ref().map(|r| r.to_uppercase())
        } else {
            self.shared_state
                .config
                .identity
                .region
                .as_ref()
                .map(|r| r.to_uppercase())
        };
        let our_region = match our_region {
            Some(r) => r,
            None => return 0.7, // Our region unknown
        };

        // Get the candidate node's region
        let peer_region = self.shared_state.peer_registry.get(node_id).and_then(|p| {
            p.capability
                .as_ref()
                .and_then(|c| c.region.as_ref().map(|r| r.to_uppercase()))
        });

        match peer_region {
            Some(ref r) if *r == our_region => 1.0,
            Some(ref r) if regions_adjacent(&our_region, r) => 0.5,
            Some(_) => 0.2,
            None => 0.7, // Unknown region — treat as neutral
        }
    }

    /// Get latency and trust for a peer. Local node gets zero latency and max trust.
    fn get_peer_metrics(&self, node_id: &NodeId, local_node_id: &NodeId) -> (u32, f32) {
        if node_id == local_node_id {
            return (0, 1.0);
        }

        self.shared_state
            .peer_registry
            .get(node_id)
            .map(|peer| (peer.latency_ms.unwrap_or(100), peer.trust_score))
            .unwrap_or((200, 0.3))
    }

    /// Greedy layer assignment: cover all layers 0..num_layers using sorted candidates.
    ///
    /// Starting from layer 0, find the best candidate that covers at least
    /// the current layer, preferring those that cover the widest contiguous range.
    /// A single node may appear multiple times in the pipeline if it has
    /// non-contiguous layer ranges (e.g., layers [0,2) and [10,14)).
    ///
    /// Constraints:
    /// - The first segment (layer 0) must be assigned to a node with `can_be_first`
    ///   (has shard 0 for token_embd.weight)
    /// - The last segment (ending at num_layers) must be assigned to a node with
    ///   `can_be_last` (has the final shard for output.weight)
    /// - When `encrypted_pipeline` is true, both first AND last segments must be
    ///   the local (requesting) node — ensures no remote node sees plaintext.
    fn greedy_assign(
        &self,
        num_layers: u32,
        candidates: &[NodeCandidate],
        encrypted_pipeline: bool,
    ) -> Result<Vec<PipelineSegment>, SwarmError> {
        let mut segments = Vec::new();
        let mut current_layer = 0u32;
        let local_node_id = self.shared_state.identity.node_id();

        while current_layer < num_layers {
            let is_first_segment = current_layer == 0;

            // Find all (candidate, range) pairs that cover current_layer
            let mut options: Vec<(&NodeCandidate, (u32, u32))> = candidates
                .iter()
                .flat_map(|c| {
                    c.available_ranges
                        .iter()
                        .filter(|r| r.0 <= current_layer && r.1 > current_layer)
                        .map(move |r| (c, *r))
                })
                .collect();

            // Encrypted pipeline: first segment MUST be the local (requesting) node
            // so that token embedding happens locally (no remote sees raw tokens).
            if is_first_segment && encrypted_pipeline {
                let local_only: Vec<_> = options
                    .iter()
                    .filter(|(c, _)| c.node_id == *local_node_id && c.can_be_first)
                    .cloned()
                    .collect();
                if local_only.is_empty() {
                    return Err(SwarmError::PipelineError(
                        "Encrypted pipeline requires the requesting node to hold shard 0 \
                         (embedding table). Download the first shard to enable this mode."
                            .to_string(),
                    ));
                }
                options = local_only;
            }
            // First segment must be assigned to a node that can serve as first
            else if is_first_segment {
                let first_capable: Vec<_> = options
                    .iter()
                    .filter(|(c, _)| c.can_be_first)
                    .cloned()
                    .collect();
                if !first_capable.is_empty() {
                    options = first_capable;
                }
                // If no can_be_first candidates, fall through (best-effort)
            }

            // If this range could reach the end, prefer nodes that can be last.
            // But ALWAYS keep the local node as an option — distributed inference
            // should use locally-hosted shards first, forwarding the remainder.
            let any_reaches_end = options.iter().any(|(_, r)| r.1 >= num_layers);
            if any_reaches_end {
                // Encrypted pipeline: last segment MUST be the local node
                // so that token sampling happens locally (no remote sees output).
                if encrypted_pipeline {
                    let local_last: Vec<_> = options
                        .iter()
                        .filter(|(c, r)| {
                            c.node_id == *local_node_id && r.1 >= num_layers && c.can_be_last
                        })
                        .cloned()
                        .collect();
                    if !local_last.is_empty() {
                        options = local_last;
                    } else {
                        // Local node can't finish from this layer, but may have a later
                        // range that reaches the end (A→B→A bounce-back).
                        // Check if the local node has ANY range that finishes the model.
                        let local_can_finish_later = candidates.iter().any(|c| {
                            c.node_id == *local_node_id
                                && c.can_be_last
                                && c.available_ranges.iter().any(|r| r.1 >= num_layers)
                        });
                        if local_can_finish_later {
                            // Find where the local node's finishing range starts, and cap
                            // remote nodes to stop before that so A can take over.
                            let local_finish_start = candidates
                                .iter()
                                .filter(|c| c.node_id == *local_node_id)
                                .flat_map(|c| c.available_ranges.iter())
                                .filter(|r| r.1 >= num_layers)
                                .map(|r| r.0)
                                .min()
                                .unwrap_or(num_layers);
                            // Cap all remote options to end before the local finishing range
                            let capped: Vec<_> = options
                                .iter()
                                .map(|(c, r)| {
                                    if c.node_id != *local_node_id && r.1 > local_finish_start {
                                        (*c, (r.0, local_finish_start))
                                    } else {
                                        (*c, *r)
                                    }
                                })
                                .filter(|(_, r)| r.1 > r.0) // drop zero-width ranges
                                .collect();
                            if !capped.is_empty() {
                                options = capped;
                            } else {
                                return Err(SwarmError::PipelineError(
                                    "Encrypted pipeline requires the requesting node to hold \
                                     the final shard (output head). Download the last shard \
                                     to enable this mode."
                                        .to_string(),
                                ));
                            }
                        } else {
                            // Local node truly can't finish — no range reaches the end
                            let not_reaching_end: Vec<_> = options
                                .iter()
                                .filter(|(_, r)| r.1 < num_layers)
                                .cloned()
                                .collect();
                            if !not_reaching_end.is_empty() {
                                options = not_reaching_end;
                            } else {
                                return Err(SwarmError::PipelineError(
                                    "Encrypted pipeline requires the requesting node to hold \
                                     the final shard (output head). Download the last shard \
                                     to enable this mode."
                                        .to_string(),
                                ));
                            }
                        }
                    }
                } else {
                    let last_capable: Vec<_> = options
                        .iter()
                        .filter(|(c, r)| {
                            (r.1 >= num_layers && c.can_be_last) || c.node_id == *local_node_id
                        })
                        .cloned()
                        .collect();
                    if !last_capable.is_empty() {
                        options = last_capable;
                    }
                    // If no can_be_last candidates reach the end, let others that DON'T
                    // reach the end take over so a can_be_last node can finish later
                    else {
                        let not_reaching_end: Vec<_> = options
                            .iter()
                            .filter(|(c, r)| r.1 < num_layers || c.node_id == *local_node_id)
                            .cloned()
                            .collect();
                        if !not_reaching_end.is_empty() {
                            options = not_reaching_end;
                        }
                    }
                }
            }

            // Pick the best candidate. Prefer the LOCAL node first to minimize
            // network round-trips and maximize use of locally-hosted shards.
            // Among remote candidates, prefer the one covering the most layers,
            // then lower-load nodes for better distribution.
            let best = options.into_iter().max_by(|(ca, ra), (cb, rb)| {
                let cov_a = ra.1 - current_layer;
                let cov_b = rb.1 - current_layer;
                let local_a = if ca.node_id == *local_node_id {
                    1u32
                } else {
                    0u32
                };
                let local_b = if cb.node_id == *local_node_id {
                    1u32
                } else {
                    0u32
                };
                // Local node always preferred — distributed inference should use
                // locally-hosted shards first, forwarding only the remainder.
                local_a
                    .cmp(&local_b)
                    .then_with(|| cov_a.cmp(&cov_b))
                    // Lower load is better → reverse comparison
                    .then_with(|| {
                        cb.load
                            .partial_cmp(&ca.load)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .then_with(|| ca.latency_ms.cmp(&cb.latency_ms).reverse())
            });

            match best {
                Some((candidate, range)) => {
                    segments.push(PipelineSegment {
                        node_id: candidate.node_id.clone(),
                        shard_id: candidate.shard_id.clone(),
                        layer_range: (current_layer, range.1),
                    });
                    current_layer = range.1;
                }
                None => {
                    return Err(SwarmError::PipelineError(format!(
                        "No node available for layer {current_layer}"
                    )));
                }
            }
        }

        Ok(segments)
    }

    /// Merge contiguous segments assigned to the same node into one segment.
    fn merge_contiguous(segments: Vec<PipelineSegment>) -> Vec<PipelineSegment> {
        let mut merged: Vec<PipelineSegment> = Vec::new();
        for seg in segments {
            if let Some(last) = merged.last_mut() {
                if last.node_id == seg.node_id && last.layer_range.1 == seg.layer_range.0 {
                    // Extend the previous segment
                    last.layer_range.1 = seg.layer_range.1;
                    continue;
                }
            }
            merged.push(seg);
        }
        merged
    }

    /// Detect tensor-parallel opportunities among LAN peers.
    ///
    /// For each pipeline segment, check if there are additional LAN peers that
    /// could serve the same layer range. If so, form a TensorParallelGroup
    /// containing the primary node plus LAN peers (up to 4 nodes per group).
    ///
    /// Tensor parallelism is only beneficial on LAN (<5ms latency) because the
    /// AllReduce communication between layers requires low latency.
    fn detect_tp_groups(
        &self,
        segments: &[PipelineSegment],
        candidates: &[NodeCandidate],
    ) -> Vec<TensorParallelGroup> {
        let local_id = self.shared_state.identity.node_id().clone();

        let mut groups = Vec::new();

        for segment in segments {
            // Only form TP groups for segments assigned to us (local node)
            if segment.node_id != local_id {
                continue;
            }

            // Find LAN peers that can serve the same layer range
            let mut tp_nodes = vec![local_id.clone()];

            for candidate in candidates {
                if candidate.node_id == local_id {
                    continue;
                }
                if tp_nodes.len() >= MAX_TP_GROUP_SIZE {
                    break;
                }

                // Must cover the same layer range
                let covers = candidate
                    .available_ranges
                    .iter()
                    .any(|r| r.0 <= segment.layer_range.0 && r.1 >= segment.layer_range.1);
                if !covers {
                    continue;
                }

                // Must be a LAN peer with low latency for AllReduce.
                // Accept peers that are either mDNS-discovered (is_lan_peer) or
                // have measured RTT ≤ 10ms (auto-detected via rr_ping).
                let (is_lan, measured_latency) = self
                    .shared_state
                    .peer_registry
                    .get(&candidate.node_id)
                    .map(|p| (p.is_lan_peer, p.latency_ms))
                    .unwrap_or((false, None));
                let tp_max_ms = self.shared_state.config.inference.tp_max_latency_ms;
                let low_latency = measured_latency.is_some_and(|ms| ms <= tp_max_ms);
                if !is_lan && !low_latency {
                    continue;
                }

                tp_nodes.push(candidate.node_id.clone());
            }

            // Need at least 2 nodes for tensor parallelism
            if tp_nodes.len() >= 2 {
                let shard_ids: Vec<_> = candidates
                    .iter()
                    .filter(|c| tp_nodes.contains(&c.node_id))
                    .map(|c| c.shard_id.clone())
                    .collect();

                tracing::info!(
                    layers = ?(segment.layer_range.0..segment.layer_range.1),
                    tp_size = tp_nodes.len(),
                    nodes = ?tp_nodes.iter().map(|n| n.to_string()).collect::<Vec<_>>(),
                    "Formed tensor-parallel group"
                );

                groups.push(TensorParallelGroup {
                    nodes: tp_nodes,
                    layer_range: segment.layer_range,
                    shard_ids,
                });
            }
        }

        groups
    }

    /// Produce a Parallax-style offline allocation recommendation for the
    /// given model. Snapshots the current peer registry + local node, derives
    /// per-peer layer capacity from known VRAM or capability signals, and
    /// runs the greedy multi-pipeline packer.
    ///
    /// Returns `None` when the manifest is missing or the cluster can't cover
    /// the model's layer count even once. Callers should treat the result as
    /// advisory — in v1 this is not auto-applied to `ShardRebalancer`.
    pub fn allocate_offline(
        &self,
        model_id: &crate::types::ModelId,
        max_pipelines: u32,
    ) -> Option<parallax_allocator::AllocationPlan> {
        let manifest = self.shared_state.model_registry.get_manifest(model_id)?;
        let num_layers = manifest.num_layers;
        if num_layers == 0 {
            return None;
        }
        let local_node_id = self.shared_state.identity.node_id();
        let bytes_per_layer = manifest.total_size_bytes / manifest.num_layers.max(1) as u64;

        let mut peers: Vec<parallax_allocator::PeerCapacity> = Vec::new();
        // Local node. If we have on-disk shards for this model, treat our
        // capacity as the union of their layer ranges; otherwise assume no
        // local capacity (Phase C won't recommend putting layers here).
        let local_tps = self
            .shared_state
            .gpu_info
            .as_ref()
            .map(|g| {
                let bw = crate::model::auto_manage::vram::gpu_memory_bandwidth_gbps(&g.name);
                crate::model::auto_manage::vram::estimate_tokens_per_sec_7b(bw, true)
            })
            .unwrap_or(0.0);
        let local_layer_capacity = manifest_layer_capacity_for_local(&manifest, &self.shared_state);
        peers.push(parallax_allocator::PeerCapacity {
            node_id: local_node_id.clone(),
            layer_capacity: local_layer_capacity,
            tokens_per_sec: local_tps,
            latency_ms: 0,
        });

        for entry in self.shared_state.peer_registry.iter() {
            let peer = entry.value();
            let node_id = peer.node_id.clone();
            if &node_id == local_node_id {
                continue;
            }
            // Prefer VRAM when the peer has a GPU, else fall back to RAM —
            // the worker subprocess can host layers in either.
            let available_mb = peer
                .capability
                .as_ref()
                .map(|c| match &c.gpu {
                    Some(g) => g.vram_available_mb,
                    None => c.ram_available_mb,
                })
                .unwrap_or(0);
            let available_bytes = available_mb.saturating_mul(1_048_576);
            let layer_capacity = available_bytes.checked_div(bytes_per_layer).unwrap_or(0) as u32;
            let tps = peer
                .capability
                .as_ref()
                .map(|c| c.est_tokens_per_sec_7b)
                .unwrap_or(0.0);
            let latency_ms = peer.latency_ms.unwrap_or(200);
            peers.push(parallax_allocator::PeerCapacity {
                node_id,
                layer_capacity,
                tokens_per_sec: tps,
                latency_ms,
            });
        }

        parallax_allocator::recommend_allocation(&peers, num_layers, max_pipelines)
    }

    /// Find standby (backup) nodes for each pipeline segment.
    fn find_standbys(
        &self,
        segments: &[PipelineSegment],
        candidates: &[NodeCandidate],
    ) -> Vec<PipelineSegment> {
        let mut standbys = Vec::new();

        let local_node_id = self.shared_state.identity.node_id();

        for segment in segments {
            // Collect all eligible standbys, then pick the local node first.
            // Preferring local prevents "no standby available" when a remote
            // primary returns an inference error — the local node can always
            // execute the segment if it has full coverage.
            let mut eligible: Vec<&NodeCandidate> = candidates
                .iter()
                .filter(|c| {
                    c.node_id != segment.node_id
                        && c.available_ranges
                            .iter()
                            .any(|r| r.0 <= segment.layer_range.0 && r.1 >= segment.layer_range.1)
                })
                .collect();
            // For standby anti-affinity: prefer DIFFERENT regions from primary
            // so a regional outage doesn't kill both primary and standby.
            // Local node first (most reliable), then different-region, then by latency.
            let primary_region_score = candidates
                .iter()
                .find(|c| c.node_id == segment.node_id)
                .map(|c| c.region_score)
                .unwrap_or(0.7);
            eligible.sort_by(|a, b| {
                let la = u32::from(a.node_id != *local_node_id);
                let lb = u32::from(b.node_id != *local_node_id);
                // Anti-affinity: if primary is same-region (1.0), prefer standbys
                // with lower region_score (different region). If primary is distant,
                // prefer same-region standbys for faster failover.
                let region_cmp = if primary_region_score > 0.8 {
                    // Primary is same-region — prefer different-region standbys
                    a.region_score
                        .partial_cmp(&b.region_score)
                        .unwrap_or(std::cmp::Ordering::Equal)
                } else {
                    // Primary is distant — prefer same-region standbys
                    b.region_score
                        .partial_cmp(&a.region_score)
                        .unwrap_or(std::cmp::Ordering::Equal)
                };
                la.cmp(&lb)
                    .then(region_cmp)
                    .then_with(|| a.latency_ms.cmp(&b.latency_ms))
            });
            if let Some(backup) = eligible.first() {
                standbys.push(PipelineSegment {
                    node_id: backup.node_id.clone(),
                    shard_id: backup.shard_id.clone(),
                    layer_range: segment.layer_range,
                });
            }
        }

        tracing::debug!(
            segment_count = segments.len(),
            standby_count = standbys.len(),
            segments = ?segments.iter().map(|s| format!("{}:{}-{}", s.node_id, s.layer_range.0, s.layer_range.1)).collect::<Vec<_>>(),
            "DIAG: find_standbys complete"
        );

        standbys
    }
}

mod parallax;
pub mod parallax_allocator;

/// Estimate how many layers the local node can reasonably host for `manifest`.
/// Uses the shards already on disk for this model as the primary signal — the
/// union of their GGUF tensor layer ranges — falling back to 0 when the node
/// holds none. This keeps Phase C's recommendations aligned with what the
/// local node is ACTUALLY ready to serve, rather than an aspirational VRAM
/// estimate.
fn manifest_layer_capacity_for_local(manifest: &ModelManifest, shared_state: &SharedState) -> u32 {
    let local_node = shared_state.identity.node_id();
    let mut shard_indices: Vec<u32> = Vec::new();
    for shard in &manifest.shards {
        let shard_id = ShardId {
            model_id: manifest.id.clone(),
            index: shard.index,
        };
        if shared_state
            .model_registry
            .shard_holders(&shard_id)
            .iter()
            .any(|n| n == local_node)
        {
            shard_indices.push(shard.index);
        }
    }
    let ranges =
        crate::inference::split::available_layer_ranges_from_manifest(manifest, &shard_indices);
    ranges
        .iter()
        .map(|(s, e)| (e - s) as u32)
        .sum::<u32>()
        .min(manifest.num_layers)
}

#[cfg(test)]
mod tests;
