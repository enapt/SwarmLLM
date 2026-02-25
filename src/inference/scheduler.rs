use std::sync::Arc;

use crate::daemon::SharedState;
use crate::error::SwarmError;
use crate::types::{ModelId, ModelManifest, NodeId, PipelineAssignment, PipelineSegment, ShardId};

/// PipelineScheduler assembles a distributed inference pipeline
/// by selecting the best nodes for each layer range.
#[derive(Clone)]
pub struct PipelineScheduler {
    shared_state: Arc<SharedState>,
}

/// A candidate node for a layer range, with scoring metadata.
#[derive(Debug, Clone)]
struct NodeCandidate {
    node_id: NodeId,
    shard_id: ShardId,
    layer_range: (u32, u32),
    latency_ms: u32,
    load: f32,
    trust_score: f32,
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
        let segments = self.greedy_assign(num_layers, &candidates)?;

        // Identify standby nodes for each segment
        let standbys = self.find_standbys(&segments, &candidates);

        let request_id = uuid::Uuid::new_v4();

        tracing::info!(
            request_id = %request_id,
            model = %model_id,
            segments = segments.len(),
            standbys = standbys.len(),
            "Pipeline assembled"
        );

        Ok(PipelineAssignment {
            request_id,
            segments,
            standbys,
        })
    }

    /// Gather all candidate nodes for the given model's shards.
    fn gather_candidates(
        &self,
        manifest: &ModelManifest,
        local_node_id: &NodeId,
    ) -> Vec<NodeCandidate> {
        let mut candidates = Vec::new();

        for shard in &manifest.shards {
            let shard_id = ShardId {
                model_id: manifest.id.clone(),
                index: shard.index,
            };

            // Check model_registry shard_holders
            let holders = self.shared_state.model_registry.shard_holders(&shard_id);

            for node_id in holders {
                let (latency_ms, trust_score) = self.get_peer_metrics(&node_id, local_node_id);

                candidates.push(NodeCandidate {
                    node_id,
                    shard_id: shard_id.clone(),
                    layer_range: shard.layer_range,
                    latency_ms,
                    load: 0.0, // TODO: track active requests per node
                    trust_score,
                });
            }
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
    fn greedy_assign(
        &self,
        num_layers: u32,
        candidates: &[NodeCandidate],
    ) -> Result<Vec<PipelineSegment>, SwarmError> {
        let mut segments = Vec::new();
        let mut current_layer = 0u32;

        while current_layer < num_layers {
            // Find the best candidate that covers current_layer
            let best = candidates
                .iter()
                .filter(|c| c.layer_range.0 <= current_layer && c.layer_range.1 > current_layer)
                .max_by_key(|c| {
                    // Prefer wider coverage, then better metrics (already sorted)
                    c.layer_range.1 - current_layer
                });

            match best {
                Some(candidate) => {
                    segments.push(PipelineSegment {
                        node_id: candidate.node_id.clone(),
                        shard_id: candidate.shard_id.clone(),
                        layer_range: (current_layer, candidate.layer_range.1),
                    });
                    current_layer = candidate.layer_range.1;
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

    /// Find standby (backup) nodes for each pipeline segment.
    fn find_standbys(
        &self,
        segments: &[PipelineSegment],
        candidates: &[NodeCandidate],
    ) -> Vec<PipelineSegment> {
        let mut standbys = Vec::new();

        for segment in segments {
            // Find the next-best candidate for the same layer range
            // that isn't the primary node
            if let Some(backup) = candidates.iter().find(|c| {
                c.node_id != segment.node_id
                    && c.layer_range.0 <= segment.layer_range.0
                    && c.layer_range.1 >= segment.layer_range.1
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
            },
            ShardInfo {
                index: 1,
                layer_range: (16, 32),
                size_bytes: 2_000_000_000,
                hash: [0u8; 32],
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
            }],
        );
        state.model_registry.register_manifest(manifest);

        let scheduler = PipelineScheduler::new(state);
        let result = scheduler.assemble_pipeline(&ModelId("orphan-model".into()), &local_id);
        assert!(result.is_err());
    }
}
