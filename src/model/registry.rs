use dashmap::DashMap;

use crate::error::SwarmError;
use crate::storage::db::Database;
use crate::types::{ModelId, ModelManifest, NodeId, ShardId};

/// Thread-safe registry of known models and shard locations.
///
/// Uses DashMap for concurrent access from multiple daemon tasks.
pub struct ModelRegistry {
    /// Known model manifests, keyed by model ID.
    manifests: DashMap<ModelId, ModelManifest>,
    /// Shard location tracking: which nodes hold which shards.
    shard_holders: DashMap<ShardId, Vec<NodeId>>,
}

impl ModelRegistry {
    pub fn new() -> Self {
        Self {
            manifests: DashMap::new(),
            shard_holders: DashMap::new(),
        }
    }

    /// Register a model manifest.
    pub fn register_manifest(&self, manifest: ModelManifest) {
        tracing::info!(model = %manifest.id, name = %manifest.name, "Registered model");
        self.manifests.insert(manifest.id.clone(), manifest);
    }

    /// Record that a node holds a specific shard.
    pub fn record_shard_holder(&self, shard_id: ShardId, node_id: NodeId) {
        self.shard_holders
            .entry(shard_id)
            .or_default()
            .push(node_id);
    }

    /// Remove a node from shard holders (e.g., node went offline).
    pub fn remove_shard_holder(&self, shard_id: &ShardId, node_id: &NodeId) {
        if let Some(mut holders) = self.shard_holders.get_mut(shard_id) {
            holders.retain(|id| id != node_id);
        }
    }

    /// Get all nodes that hold a specific shard.
    pub fn shard_holders(&self, shard_id: &ShardId) -> Vec<NodeId> {
        self.shard_holders
            .get(shard_id)
            .map(|v| v.clone())
            .unwrap_or_default()
    }

    /// Get a model manifest by ID.
    pub fn get_manifest(&self, model_id: &ModelId) -> Option<ModelManifest> {
        self.manifests.get(model_id).map(|v| v.clone())
    }

    /// Get all known model manifests.
    pub fn models(&self) -> Vec<ModelManifest> {
        self.manifests.iter().map(|v| v.value().clone()).collect()
    }

    /// List all known model manifests (alias for models()).
    pub fn list_models(&self) -> Vec<ModelManifest> {
        self.models()
    }

    /// Get the number of registered models.
    pub fn model_count(&self) -> usize {
        self.manifests.len()
    }

    /// Get the number of tracked shards.
    pub fn shard_count(&self) -> usize {
        self.shard_holders.len()
    }

    /// Check if a specific shard is tracked.
    pub fn has_shard(&self, shard_id: &ShardId) -> bool {
        self.shard_holders
            .get(shard_id)
            .map(|v| !v.is_empty())
            .unwrap_or(false)
    }

    /// Iterate over all tracked shard entries (shard_id, holders).
    pub fn all_shard_entries(&self) -> Vec<(ShardId, Vec<NodeId>)> {
        self.shard_holders
            .iter()
            .map(|entry| (entry.key().clone(), entry.value().clone()))
            .collect()
    }

    /// Load model and shard metadata from the database.
    pub fn load_from_db(db: &Database) -> Result<Self, SwarmError> {
        let registry = Self::new();

        // Load model manifests from the "model_meta" tree
        let tree = db.tree("model_meta")?;
        for item in tree.iter() {
            let (_, value) = item.map_err(SwarmError::Database)?;
            match serde_json::from_slice::<ModelManifest>(&value) {
                Ok(manifest) => {
                    registry.manifests.insert(manifest.id.clone(), manifest);
                }
                Err(e) => {
                    tracing::warn!(error = %e, "Failed to deserialize model manifest from DB");
                }
            }
        }

        tracing::info!(
            models = registry.manifests.len(),
            "Loaded model registry from DB"
        );

        Ok(registry)
    }

    /// Persist a model manifest to the database.
    pub fn persist_manifest(
        &self,
        db: &Database,
        manifest: &ModelManifest,
    ) -> Result<(), SwarmError> {
        db.put_json("model_meta", &manifest.id.0, manifest)?;
        Ok(())
    }
}

impl Default for ModelRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::*;

    #[test]
    fn register_and_retrieve_manifest() {
        let registry = ModelRegistry::new();
        let manifest = ModelManifest {
            id: ModelId("test".into()),
            name: "Test".into(),
            architecture: ModelArchitecture::Llama,
            num_layers: 2,
            num_params_billions: 0.001,
            quantization: Quantization::Q4KM,
            total_size_bytes: 1024,
            shard_count: 1,
            shards: vec![],
            tokenizer_hash: [0u8; 32],
            manifest_hash: [0u8; 32],
            publisher: NodeId([0u8; 32]),
            publish_date: chrono::Utc::now(),
            license: "MIT".into(),
        };

        registry.register_manifest(manifest.clone());

        let retrieved = registry.get_manifest(&ModelId("test".into()));
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().name, "Test");
    }

    #[test]
    fn shard_holder_tracking() {
        let registry = ModelRegistry::new();
        let shard_id = ShardId {
            model_id: ModelId("test".into()),
            index: 0,
        };
        let node_a = NodeId([1u8; 32]);
        let node_b = NodeId([2u8; 32]);

        registry.record_shard_holder(shard_id.clone(), node_a.clone());
        registry.record_shard_holder(shard_id.clone(), node_b.clone());

        let holders = registry.shard_holders(&shard_id);
        assert_eq!(holders.len(), 2);
    }

    #[test]
    fn remove_shard_holder() {
        let registry = ModelRegistry::new();
        let shard_id = ShardId {
            model_id: ModelId("test".into()),
            index: 0,
        };
        let node_a = NodeId([1u8; 32]);
        let node_b = NodeId([2u8; 32]);

        registry.record_shard_holder(shard_id.clone(), node_a.clone());
        registry.record_shard_holder(shard_id.clone(), node_b.clone());

        registry.remove_shard_holder(&shard_id, &node_a);
        let holders = registry.shard_holders(&shard_id);
        assert_eq!(holders.len(), 1);
        assert_eq!(holders[0], node_b);
    }

    #[test]
    fn models_returns_all() {
        let registry = ModelRegistry::new();
        assert_eq!(registry.model_count(), 0);

        registry.register_manifest(ModelManifest {
            id: ModelId("a".into()),
            name: "A".into(),
            architecture: ModelArchitecture::Llama,
            num_layers: 2,
            num_params_billions: 0.001,
            quantization: Quantization::Q4KM,
            total_size_bytes: 1024,
            shard_count: 1,
            shards: vec![],
            tokenizer_hash: [0u8; 32],
            manifest_hash: [0u8; 32],
            publisher: NodeId([0u8; 32]),
            publish_date: chrono::Utc::now(),
            license: "MIT".into(),
        });

        assert_eq!(registry.model_count(), 1);
        assert_eq!(registry.models().len(), 1);
    }
}
