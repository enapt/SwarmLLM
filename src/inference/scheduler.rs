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
    /// 3. Query shard_registry for all nodes hosting shards of this model
    /// 4. For each node, fetch current load and latency from peer_registry
    /// 5. Greedy assignment: sort candidates by (latency ASC, load ASC, trust DESC),
    ///    assign the best available node covering the widest contiguous layer range
    /// 6. If any layer range has no available node -> fail
    /// 7. Identify standby nodes for each segment
    /// 8. Return PipelineAssignment
    pub fn assemble_pipeline(
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

        // Gather all candidates: nodes that have shards for this model
        let candidates = self.gather_candidates(&manifest, local_node_id);
        if candidates.is_empty() {
            return Err(SwarmError::InsufficientCapacity(model_id.clone()));
        }

        // Greedy layer assignment
        let raw_segments = self.greedy_assign(num_layers, &candidates)?;

        // Merge contiguous segments on the same node into a single segment.
        // This avoids sending multiple LayerForward messages to the same node
        // when it handles its full layer range in one forward pass.
        let segments = Self::merge_contiguous(raw_segments);

        // Identify standby nodes for each segment
        let standbys = self.find_standbys(&segments, &candidates);

        // Detect tensor-parallel opportunities: LAN peers sharing the same layer range.
        let tp_groups = self.detect_tp_groups(&segments, &candidates, &manifest);

        tracing::info!(
            request_id = %request_id,
            model = %model_id,
            segments = segments.len(),
            standbys = standbys.len(),
            tp_groups = tp_groups.len(),
            "Pipeline assembled"
        );

        Ok(PipelineAssignment {
            request_id,
            segments,
            standbys,
            tp_groups,
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
                node_shards.entry(node_id).or_default().push(shard.index);
            }
        }

        let _shard_size = self.shared_state.config.model.shard_size_bytes();

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

            // Use peer-reported load from health pings when available,
            // fall back to local active_pipelines count for our own node
            let active_load = if &node_id == local_node_id {
                self.shared_state.active_pipelines.len() as f32
            } else {
                self.shared_state
                    .peer_registry
                    .get(&node_id)
                    .map(|p| p.active_request_count as f32)
                    .unwrap_or_else(|| {
                        // Fallback: estimate from active pipelines (pre-health-ping behavior)
                        self.shared_state
                            .active_pipelines
                            .iter()
                            .filter(|entry| {
                                entry.value().segments.iter().any(|s| s.node_id == node_id)
                            })
                            .count() as f32
                    })
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
            });
        }

        // Log candidates for debugging
        for c in &candidates {
            tracing::debug!(
                node = %c.node_id,
                ranges = ?c.available_ranges,
                can_be_first = c.can_be_first,
                can_be_last = c.can_be_last,
                "Pipeline candidate"
            );
        }

        // Sort: latency ASC, load ASC, trust DESC
        candidates.sort_by(|a, b| {
            a.latency_ms
                .cmp(&b.latency_ms)
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
        });

        candidates
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
    fn greedy_assign(
        &self,
        num_layers: u32,
        candidates: &[NodeCandidate],
    ) -> Result<Vec<PipelineSegment>, SwarmError> {
        let mut segments = Vec::new();
        let mut current_layer = 0u32;

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

            // First segment must be assigned to a node that can serve as first
            if is_first_segment {
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
            // Check if any candidate can reach num_layers from current position.
            let any_reaches_end = options.iter().any(|(_, r)| r.1 >= num_layers);
            if any_reaches_end {
                let last_capable: Vec<_> = options
                    .iter()
                    .filter(|(c, r)| r.1 >= num_layers && c.can_be_last)
                    .cloned()
                    .collect();
                if !last_capable.is_empty() {
                    // Prefer candidates that can be the final segment
                    options = last_capable;
                }
                // If no can_be_last candidates reach the end, let others that DON'T
                // reach the end take over so a can_be_last node can finish later
                else {
                    let not_reaching_end: Vec<_> = options
                        .iter()
                        .filter(|(_, r)| r.1 < num_layers)
                        .cloned()
                        .collect();
                    if !not_reaching_end.is_empty() {
                        options = not_reaching_end;
                    }
                }
            }

            // Pick the candidate that covers the most layers. When tied, prefer
            // the local node to avoid unnecessary network round-trips, then
            // prefer lower-load nodes for better distribution.
            let local_node_id = self.shared_state.identity.node_id();
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
                cov_a
                    .cmp(&cov_b)
                    .then_with(|| local_a.cmp(&local_b))
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
        _segments: &[PipelineSegment],
        _candidates: &[NodeCandidate],
        _manifest: &ModelManifest,
    ) -> Vec<TensorParallelGroup> {
        // Multi-node TP requires AllReduce, which is not yet implemented.
        // Forming TP groups would produce wrong hidden states (each node computes
        // a partial result but there is no reduction step to combine them).
        // Disabled until AllReduce lands — always return empty.
        Vec::new()
    }

    /// Find standby (backup) nodes for each pipeline segment.
    fn find_standbys(
        &self,
        segments: &[PipelineSegment],
        candidates: &[NodeCandidate],
    ) -> Vec<PipelineSegment> {
        let mut standbys = Vec::new();

        for segment in segments {
            // Find the next-best candidate for the same layer range
            // that isn't the primary node.  Check if ANY of the candidate's
            // available_ranges fully covers the segment.
            if let Some(backup) = candidates.iter().find(|c| {
                c.node_id != segment.node_id
                    && c.available_ranges
                        .iter()
                        .any(|r| r.0 <= segment.layer_range.0 && r.1 >= segment.layer_range.1)
            }) {
                standbys.push(PipelineSegment {
                    node_id: backup.node_id.clone(),
                    shard_id: backup.shard_id.clone(),
                    layer_range: segment.layer_range,
                });
            }
        }

        standbys
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use crate::identity::Identity;
    use crate::inference::executor::ModelExecutor;
    use crate::storage::db::Database;
    use crate::types::*;
    use std::sync::Arc;
    use tokio::sync::Mutex;

    fn make_shared_state() -> Arc<SharedState> {
        let config = Config::default();
        let identity = Identity::generate();
        let temp = tempfile::tempdir().unwrap();
        let db = Database::open(temp.path()).unwrap();
        let executor = Arc::new(Mutex::new(ModelExecutor::new()));
        let (state, _) = SharedState::new(config, identity, db, executor, None);
        state
    }

    fn make_manifest(model_id: &str, num_layers: u32, shards: Vec<ShardInfo>) -> ModelManifest {
        ModelManifest {
            schema_version: 1,
            id: ModelId(model_id.into()),
            name: "Test Model".into(),
            architecture: ModelArchitecture::Llama,
            num_layers,
            num_params_billions: 7.0,
            quantization: Quantization::Q4KM,
            total_size_bytes: 4_000_000_000,
            shard_count: shards.len() as u32,
            shards,
            tokenizer_hash: [0u8; 32],
            manifest_hash: [0u8; 32],
            publisher: NodeId([0u8; 32]),
            publish_date: chrono::Utc::now(),
            license: "MIT".into(),
        }
    }

    #[test]
    fn assemble_single_node_pipeline() {
        let state = make_shared_state();
        let local_id = state.identity.node_id().clone();

        let shards = vec![ShardInfo {
            index: 0,
            layer_range: (0, 32),
            size_bytes: 4_000_000_000,
            hash: [0u8; 32],
            tensors: vec![],
        }];
        let manifest = make_manifest("test-model", 32, shards);
        state.model_registry.register_manifest(manifest);

        // Register local node as shard holder
        let shard_id = ShardId {
            model_id: ModelId("test-model".into()),
            index: 0,
        };
        state
            .model_registry
            .record_shard_holder(shard_id, local_id.clone());

        let scheduler = PipelineScheduler::new(state);
        let assignment = scheduler
            .assemble_pipeline(&ModelId("test-model".into()), &local_id)
            .unwrap();

        assert_eq!(assignment.segments.len(), 1);
        assert_eq!(assignment.segments[0].layer_range, (0, 32));
        assert_eq!(assignment.segments[0].node_id, local_id);
    }

    #[test]
    fn assemble_multi_node_pipeline() {
        let state = make_shared_state();
        let local_id = state.identity.node_id().clone();
        let node_b = NodeId([2u8; 32]);
        let node_c = NodeId([3u8; 32]);

        let shards = vec![
            ShardInfo {
                index: 0,
                layer_range: (0, 16),
                size_bytes: 2_000_000_000,
                hash: [0u8; 32],
                tensors: vec![],
            },
            ShardInfo {
                index: 1,
                layer_range: (16, 32),
                size_bytes: 2_000_000_000,
                hash: [0u8; 32],
                tensors: vec![],
            },
        ];
        let manifest = make_manifest("test-model", 32, shards);
        state.model_registry.register_manifest(manifest);

        // Node B has shard 0, Node C has shard 1
        state.model_registry.record_shard_holder(
            ShardId {
                model_id: ModelId("test-model".into()),
                index: 0,
            },
            node_b.clone(),
        );
        state.model_registry.record_shard_holder(
            ShardId {
                model_id: ModelId("test-model".into()),
                index: 1,
            },
            node_c.clone(),
        );

        // Add peer info so latencies are known
        state.peer_registry.insert(
            node_b.clone(),
            PeerInfo {
                node_id: node_b.clone(),
                addresses: vec![],
                capability: None,
                last_seen: chrono::Utc::now(),
                latency_ms: Some(10),
                trust_score: 0.8,
                peer_id_bytes: None,
                active_request_count: 0,
                first_seen: 0,
                verified_transaction_count: 0,
                is_lan_peer: false,
            },
        );
        state.peer_registry.insert(
            node_c.clone(),
            PeerInfo {
                node_id: node_c.clone(),
                addresses: vec![],
                capability: None,
                last_seen: chrono::Utc::now(),
                latency_ms: Some(15),
                trust_score: 0.9,
                peer_id_bytes: None,
                active_request_count: 0,
                first_seen: 0,
                verified_transaction_count: 0,
                is_lan_peer: false,
            },
        );

        let scheduler = PipelineScheduler::new(state);
        let assignment = scheduler
            .assemble_pipeline(&ModelId("test-model".into()), &local_id)
            .unwrap();

        assert_eq!(assignment.segments.len(), 2);
        assert_eq!(assignment.segments[0].layer_range, (0, 16));
        assert_eq!(assignment.segments[1].layer_range, (16, 32));
    }

    #[test]
    fn fails_when_model_not_found() {
        let state = make_shared_state();
        let local_id = state.identity.node_id().clone();
        let scheduler = PipelineScheduler::new(state);

        let result = scheduler.assemble_pipeline(&ModelId("nonexistent".into()), &local_id);
        assert!(result.is_err());
    }

    #[test]
    fn fails_when_no_shard_holders() {
        let state = make_shared_state();
        let local_id = state.identity.node_id().clone();

        let manifest = make_manifest(
            "orphan-model",
            32,
            vec![ShardInfo {
                index: 0,
                layer_range: (0, 32),
                size_bytes: 4_000_000_000,
                hash: [0u8; 32],
                tensors: vec![],
            }],
        );
        state.model_registry.register_manifest(manifest);

        let scheduler = PipelineScheduler::new(state);
        let result = scheduler.assemble_pipeline(&ModelId("orphan-model".into()), &local_id);
        assert!(result.is_err());
    }

    #[test]
    fn merge_contiguous_segments_same_node() {
        let node = NodeId([1u8; 32]);
        let shard = ShardId {
            model_id: ModelId("m".into()),
            index: 0,
        };
        let segments = vec![
            PipelineSegment {
                node_id: node.clone(),
                shard_id: shard.clone(),
                layer_range: (0, 2),
            },
            PipelineSegment {
                node_id: node.clone(),
                shard_id: shard.clone(),
                layer_range: (2, 4),
            },
        ];
        let merged = PipelineScheduler::merge_contiguous(segments);
        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0].layer_range, (0, 4));
    }

    #[test]
    fn greedy_assign_multi_range_candidate() {
        // Test that a candidate with multiple non-contiguous ranges can
        // serve multiple pipeline segments for the same model.
        let state = make_shared_state();
        let scheduler = PipelineScheduler::new(state);

        // Candidate A: layers [0,2) and [10,14)
        // Candidate B: layers [2,10)
        let candidates = vec![
            NodeCandidate {
                node_id: NodeId([1u8; 32]),
                shard_id: ShardId {
                    model_id: ModelId("test".into()),
                    index: 0,
                },
                available_ranges: vec![(0, 2), (10, 14)],
                latency_ms: 0,
                load: 0.0,
                trust_score: 1.0,
                can_be_first: true,
                can_be_last: true,
            },
            NodeCandidate {
                node_id: NodeId([2u8; 32]),
                shard_id: ShardId {
                    model_id: ModelId("test".into()),
                    index: 1,
                },
                available_ranges: vec![(2, 10)],
                latency_ms: 10,
                load: 0.0,
                trust_score: 0.8,
                can_be_first: false,
                can_be_last: false,
            },
        ];

        let segments = scheduler.greedy_assign(14, &candidates).unwrap();
        // Should produce 3 segments: [0,2) on A, [2,10) on B, [10,14) on A
        assert_eq!(segments.len(), 3);
        assert_eq!(segments[0].layer_range, (0, 2));
        assert_eq!(segments[0].node_id, NodeId([1u8; 32]));
        assert_eq!(segments[1].layer_range, (2, 10));
        assert_eq!(segments[1].node_id, NodeId([2u8; 32]));
        assert_eq!(segments[2].layer_range, (10, 14));
        assert_eq!(segments[2].node_id, NodeId([1u8; 32]));

        // After merging, same-node contiguous segments collapse
        let merged = PipelineScheduler::merge_contiguous(segments);
        // A's [0,2) and [10,14) are NOT contiguous → no merge → still 3 segments
        assert_eq!(merged.len(), 3);
    }

    #[test]
    fn prefers_lower_load_node() {
        // Two nodes with identical latency and trust but different load.
        // The scheduler should prefer the node with lower load.
        let state = make_shared_state();
        let local_id = state.identity.node_id().clone();
        let node_a = NodeId([10u8; 32]);
        let node_b = NodeId([11u8; 32]);

        let shards = vec![ShardInfo {
            index: 0,
            layer_range: (0, 16),
            size_bytes: 2_000_000_000,
            hash: [0u8; 32],
            tensors: vec![],
        }];
        let manifest = make_manifest("load-test", 16, shards);
        state.model_registry.register_manifest(manifest);

        // Both nodes hold shard 0
        let shard_id = ShardId {
            model_id: ModelId("load-test".into()),
            index: 0,
        };
        state
            .model_registry
            .record_shard_holder(shard_id.clone(), node_a.clone());
        state
            .model_registry
            .record_shard_holder(shard_id, node_b.clone());

        // Same latency and trust, but different load via active_request_count
        state.peer_registry.insert(
            node_a.clone(),
            PeerInfo {
                node_id: node_a.clone(),
                addresses: vec![],
                capability: None,
                last_seen: chrono::Utc::now(),
                latency_ms: Some(20),
                trust_score: 0.8,
                peer_id_bytes: None,
                active_request_count: 10, // high load
                first_seen: 0,
                verified_transaction_count: 0,
                is_lan_peer: false,
            },
        );
        state.peer_registry.insert(
            node_b.clone(),
            PeerInfo {
                node_id: node_b.clone(),
                addresses: vec![],
                capability: None,
                last_seen: chrono::Utc::now(),
                latency_ms: Some(20),
                trust_score: 0.8,
                peer_id_bytes: None,
                active_request_count: 1, // low load
                first_seen: 0,
                verified_transaction_count: 0,
                is_lan_peer: false,
            },
        );

        let scheduler = PipelineScheduler::new(state);
        let assignment = scheduler
            .assemble_pipeline(&ModelId("load-test".into()), &local_id)
            .unwrap();

        // Node B (low load) should be selected over Node A (high load)
        assert_eq!(assignment.segments.len(), 1);
        assert_eq!(assignment.segments[0].node_id, node_b);
    }

    #[test]
    fn detects_tp_group_for_lan_peers() {
        let state = make_shared_state();
        let local_id = state.identity.node_id().clone();
        let node_b = NodeId([20u8; 32]);

        let shards = vec![ShardInfo {
            index: 0,
            layer_range: (0, 32),
            size_bytes: 4_000_000_000,
            hash: [0u8; 32],
            tensors: vec![],
        }];
        let manifest = make_manifest("tp-model", 32, shards);
        state.model_registry.register_manifest(manifest);

        // Both local node and Node B host the same shard
        let shard_id = ShardId {
            model_id: ModelId("tp-model".into()),
            index: 0,
        };
        state
            .model_registry
            .record_shard_holder(shard_id.clone(), local_id.clone());
        state
            .model_registry
            .record_shard_holder(shard_id, node_b.clone());

        // Mark Node B as a LAN peer
        state.peer_registry.insert(
            node_b.clone(),
            PeerInfo {
                node_id: node_b.clone(),
                addresses: vec![],
                capability: None,
                last_seen: chrono::Utc::now(),
                latency_ms: Some(1),
                trust_score: 0.9,
                peer_id_bytes: None,
                active_request_count: 0,
                first_seen: 0,
                verified_transaction_count: 0,
                is_lan_peer: true,
            },
        );

        let scheduler = PipelineScheduler::new(state);
        let assignment = scheduler
            .assemble_pipeline(&ModelId("tp-model".into()), &local_id)
            .unwrap();

        // Pipeline should have 1 segment (local node wins) + 1 TP group with both nodes
        assert_eq!(assignment.segments.len(), 1);
        assert_eq!(assignment.tp_groups.len(), 1);
        assert_eq!(assignment.tp_groups[0].tp_size(), 2);
        assert!(assignment.tp_groups[0].nodes.contains(&local_id));
        assert!(assignment.tp_groups[0].nodes.contains(&node_b));
        assert_eq!(assignment.tp_groups[0].layer_range, (0, 32));
    }

    #[test]
    fn no_tp_group_for_wan_peers() {
        let state = make_shared_state();
        let local_id = state.identity.node_id().clone();
        let node_b = NodeId([21u8; 32]);

        let shards = vec![ShardInfo {
            index: 0,
            layer_range: (0, 32),
            size_bytes: 4_000_000_000,
            hash: [0u8; 32],
            tensors: vec![],
        }];
        let manifest = make_manifest("wan-model", 32, shards);
        state.model_registry.register_manifest(manifest);

        let shard_id = ShardId {
            model_id: ModelId("wan-model".into()),
            index: 0,
        };
        state
            .model_registry
            .record_shard_holder(shard_id.clone(), local_id.clone());
        state
            .model_registry
            .record_shard_holder(shard_id, node_b.clone());

        // Node B is NOT a LAN peer
        state.peer_registry.insert(
            node_b.clone(),
            PeerInfo {
                node_id: node_b,
                addresses: vec![],
                capability: None,
                last_seen: chrono::Utc::now(),
                latency_ms: Some(100),
                trust_score: 0.8,
                peer_id_bytes: None,
                active_request_count: 0,
                first_seen: 0,
                verified_transaction_count: 0,
                is_lan_peer: false,
            },
        );

        let scheduler = PipelineScheduler::new(state);
        let assignment = scheduler
            .assemble_pipeline(&ModelId("wan-model".into()), &local_id)
            .unwrap();

        // Should have segments but NO TP groups (WAN peer)
        assert_eq!(assignment.segments.len(), 1);
        assert!(assignment.tp_groups.is_empty());
    }

    #[test]
    fn tp_group_rank_of() {
        let group = TensorParallelGroup {
            nodes: vec![NodeId([1u8; 32]), NodeId([2u8; 32]), NodeId([3u8; 32])],
            layer_range: (0, 32),
            shard_ids: vec![],
        };
        assert_eq!(group.rank_of(&NodeId([1u8; 32])), Some(0));
        assert_eq!(group.rank_of(&NodeId([2u8; 32])), Some(1));
        assert_eq!(group.rank_of(&NodeId([3u8; 32])), Some(2));
        assert_eq!(group.rank_of(&NodeId([4u8; 32])), None);
        assert_eq!(group.tp_size(), 3);
    }
}
