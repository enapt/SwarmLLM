use std::collections::{HashMap, HashSet};
use std::time::Instant;

use dashmap::DashMap;

use crate::error::SwarmError;
use crate::storage::db::Database;
use crate::types::{ModelId, ModelManifest, NodeId, ShardId, MMPROJ_SHARD_INDEX};

/// Maximum number of holders tracked per shard in the in-memory cache.
/// At 50K nodes, this bounds memory to O(shards × 50) instead of O(shards × nodes).
/// DHT provider queries fill in the rest on demand.
const MAX_HOLDERS_PER_SHARD: usize = 50;

/// Thread-safe registry of known models and shard locations.
///
/// Uses DashMap for concurrent access from multiple daemon tasks.
/// Shard holder tracking is bounded: at most `MAX_HOLDERS_PER_SHARD` holders
/// are cached per shard. The local node is never evicted. For accurate holder
/// counts at scale, use DHT provider queries via `QueryShardProviders`.
pub struct ModelRegistry {
    /// Known model manifests, keyed by model ID.
    manifests: DashMap<ModelId, ModelManifest>,
    /// Shard location tracking: which nodes hold which shards.
    /// HashMap value maps NodeId → last_seen timestamp for LRU eviction.
    /// Bounded at MAX_HOLDERS_PER_SHARD entries per shard.
    shard_holders: DashMap<ShardId, HashMap<NodeId, Instant>>,
    /// Reverse index: which shards each node holds.
    /// Maintained in sync by record_shard_holder() / remove_shard_holder().
    /// Enables O(shards_held) peer departure instead of O(all_shards).
    node_shards: DashMap<NodeId, HashSet<ShardId>>,
    /// Local node ID — never evicted from holder sets.
    local_node_id: Option<NodeId>,
}

impl ModelRegistry {
    pub fn new() -> Self {
        Self {
            manifests: DashMap::new(),
            shard_holders: DashMap::new(),
            node_shards: DashMap::new(),
            local_node_id: None,
        }
    }

    /// Create a registry with a known local node ID.
    /// The local node is never evicted from bounded holder sets.
    pub fn with_local_node(local_node_id: NodeId) -> Self {
        Self {
            manifests: DashMap::new(),
            shard_holders: DashMap::new(),
            node_shards: DashMap::new(),
            local_node_id: Some(local_node_id),
        }
    }

    /// Set the local node ID after construction (e.g., after loading from DB).
    pub fn set_local_node_id(&mut self, node_id: NodeId) {
        self.local_node_id = Some(node_id);
    }

    /// Register a model manifest.
    pub fn register_manifest(&self, manifest: ModelManifest) {
        tracing::info!(
            model = %manifest.id,
            name = %manifest.name,
            schema_version = manifest.schema_version,
            shard_count = manifest.shard_count,
            "DIAG: register_manifest"
        );
        self.manifests.insert(manifest.id.clone(), manifest);
    }

    /// Record that a node holds a specific shard.
    ///
    /// Bounded: if the holder set is at capacity, the oldest non-local holder
    /// is evicted to make room. Maintains reverse index.
    pub fn record_shard_holder(&self, shard_id: ShardId, node_id: NodeId) {
        let mut entry = self.shard_holders.entry(shard_id.clone()).or_default();
        let holders = entry.value_mut();

        // Update timestamp if already present (always succeeds, no eviction needed)
        if holders.contains_key(&node_id) {
            holders.insert(node_id.clone(), Instant::now());
            // Reverse index already has this entry
            return;
        }

        // At capacity — evict oldest non-local holder
        if holders.len() >= MAX_HOLDERS_PER_SHARD {
            let evict_id = holders
                .iter()
                .filter(|(nid, _)| Some(*nid) != self.local_node_id.as_ref())
                .min_by_key(|(_, ts)| *ts)
                .map(|(nid, _)| nid.clone());

            if let Some(evict) = evict_id {
                holders.remove(&evict);
                // Mirror in reverse index
                if let Some(mut shards) = self.node_shards.get_mut(&evict) {
                    shards.remove(&shard_id);
                }
            } else {
                // All holders are local (shouldn't happen) — skip insert
                return;
            }
        }

        holders.insert(node_id.clone(), Instant::now());
        // Drop the DashMap entry guard before taking another lock
        drop(entry);
        self.node_shards
            .entry(node_id)
            .or_default()
            .insert(shard_id);
    }

    /// Remove a node from shard holders (e.g., node went offline).
    /// Maintains reverse index.
    pub fn remove_shard_holder(&self, shard_id: &ShardId, node_id: &NodeId) {
        if let Some(mut holders) = self.shard_holders.get_mut(shard_id) {
            holders.remove(node_id);
        }
        if let Some(mut shards) = self.node_shards.get_mut(node_id) {
            shards.remove(shard_id);
        }
    }

    /// Get nodes that hold the mmproj (vision encoder) for a model.
    pub fn mmproj_holders(&self, model_id: &ModelId) -> Vec<NodeId> {
        self.shard_holders(&ShardId {
            model_id: model_id.clone(),
            index: MMPROJ_SHARD_INDEX,
        })
    }

    /// Get all nodes that hold a specific shard (from the bounded cache).
    pub fn shard_holders(&self, shard_id: &ShardId) -> Vec<NodeId> {
        self.shard_holders
            .get(shard_id)
            .map(|v| v.keys().cloned().collect())
            .unwrap_or_default()
    }

    /// Get the cached holder count for a shard without allocating.
    /// This may undercount at scale — use DHT queries for accurate counts.
    pub fn shard_holder_count(&self, shard_id: &ShardId) -> usize {
        self.shard_holders
            .get(shard_id)
            .map(|v| v.len())
            .unwrap_or(0)
    }

    /// Merge holders discovered via DHT provider queries into the cache.
    /// Updates timestamps for existing entries and adds new ones (with eviction).
    pub fn merge_dht_providers(&self, shard_id: &ShardId, providers: &[NodeId]) {
        for node_id in providers {
            self.record_shard_holder(shard_id.clone(), node_id.clone());
        }
    }

    /// Get a model manifest by ID.
    pub fn get_manifest(&self, model_id: &ModelId) -> Option<ModelManifest> {
        self.manifests.get(model_id).map(|v| v.clone())
    }

    /// Get all known model manifests.
    pub fn models(&self) -> Vec<ModelManifest> {
        self.manifests.iter().map(|v| v.value().clone()).collect()
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

    /// Remove a model manifest from the registry.
    pub fn remove_manifest(&self, model_id: &ModelId) {
        self.manifests.remove(model_id);
    }

    /// Remove all shard holder entries for a given model.
    /// Maintains reverse index: removes affected ShardIds from each holder's node_shards.
    pub fn remove_all_model_shards(&self, model_id: &ModelId) {
        // Collect affected (shard, holders) before removing from shard_holders
        let affected: Vec<(ShardId, Vec<NodeId>)> = self
            .shard_holders
            .iter()
            .filter(|e| &e.key().model_id == model_id)
            .map(|e| (e.key().clone(), e.value().keys().cloned().collect()))
            .collect();

        self.shard_holders
            .retain(|shard_id, _| &shard_id.model_id != model_id);

        // Mirror removal in reverse index
        for (shard_id, holders) in affected {
            for node_id in holders {
                if let Some(mut shards) = self.node_shards.get_mut(&node_id) {
                    shards.remove(&shard_id);
                }
            }
        }
    }

    /// Remove a peer from all shard holder entries (e.g., peer went offline).
    /// Uses reverse index for O(shards_held) instead of O(all_shards).
    pub fn remove_peer_from_all_shards(&self, node_id: &NodeId) {
        if let Some((_, shards)) = self.node_shards.remove(node_id) {
            for shard_id in shards {
                if let Some(mut holders) = self.shard_holders.get_mut(&shard_id) {
                    holders.remove(node_id);
                }
            }
        }
    }

    /// Get all shards held by a specific node (via reverse index).
    pub fn shards_for_node(&self, node_id: &NodeId) -> Vec<ShardId> {
        self.node_shards
            .get(node_id)
            .map(|s| s.iter().cloned().collect())
            .unwrap_or_default()
    }

    /// Get unique model IDs held by a specific node.
    pub fn models_for_node(&self, node_id: &NodeId) -> Vec<ModelId> {
        self.node_shards
            .get(node_id)
            .map(|shards| {
                let mut models: HashSet<ModelId> = HashSet::new();
                for shard in shards.iter() {
                    models.insert(shard.model_id.clone());
                }
                models.into_iter().collect()
            })
            .unwrap_or_default()
    }

    /// Iterate over all tracked shard entries (shard_id, holders).
    pub fn all_shard_entries(&self) -> Vec<(ShardId, Vec<NodeId>)> {
        self.shard_holders
            .iter()
            .map(|entry| (entry.key().clone(), entry.value().keys().cloned().collect()))
            .collect()
    }

    /// Load model and shard metadata from the database.
    pub fn load_from_db(db: &Database) -> Result<Self, SwarmError> {
        let registry = Self::new();

        // Load model manifests from the "model_meta" tree
        let entries = db.iter_raw("model_meta")?;
        for (_key, value) in entries {
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
            manifests_loaded_count = registry.manifests.len(),
            "DIAG: load_from_db complete"
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
            schema_version: 2,
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
            mmproj: None,
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
    fn bounded_eviction() {
        let local = NodeId([0u8; 32]);
        let registry = ModelRegistry::with_local_node(local.clone());
        let shard_id = ShardId {
            model_id: ModelId("test".into()),
            index: 0,
        };

        // Register local node first
        registry.record_shard_holder(shard_id.clone(), local.clone());

        // Fill to capacity
        for i in 1..MAX_HOLDERS_PER_SHARD {
            let mut bytes = [0u8; 32];
            bytes[0] = (i & 0xFF) as u8;
            bytes[1] = ((i >> 8) & 0xFF) as u8;
            registry.record_shard_holder(shard_id.clone(), NodeId(bytes));
        }

        assert_eq!(
            registry.shard_holder_count(&shard_id),
            MAX_HOLDERS_PER_SHARD
        );

        // Insert one more — should evict oldest non-local
        let overflow = NodeId([255u8; 32]);
        registry.record_shard_holder(shard_id.clone(), overflow.clone());

        // Still at cap
        assert_eq!(
            registry.shard_holder_count(&shard_id),
            MAX_HOLDERS_PER_SHARD
        );

        // Local node still present
        assert!(registry.shard_holders(&shard_id).contains(&local));
        // New node present
        assert!(registry.shard_holders(&shard_id).contains(&overflow));
    }

    #[test]
    fn merge_dht_providers() {
        let registry = ModelRegistry::new();
        let shard_id = ShardId {
            model_id: ModelId("test".into()),
            index: 0,
        };
        let nodes: Vec<NodeId> = (1..=5)
            .map(|i| {
                let mut bytes = [0u8; 32];
                bytes[0] = i;
                NodeId(bytes)
            })
            .collect();

        registry.merge_dht_providers(&shard_id, &nodes);
        assert_eq!(registry.shard_holder_count(&shard_id), 5);
    }

    #[test]
    fn models_returns_all() {
        let registry = ModelRegistry::new();
        assert_eq!(registry.model_count(), 0);

        registry.register_manifest(ModelManifest {
            schema_version: 2,
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
            mmproj: None,
        });

        assert_eq!(registry.model_count(), 1);
        assert_eq!(registry.models().len(), 1);
    }

    #[test]
    fn sentinel_shard_is_mmproj() {
        let shard = ShardId {
            model_id: ModelId("test".into()),
            index: MMPROJ_SHARD_INDEX,
        };
        assert!(shard.is_mmproj());

        let regular = ShardId {
            model_id: ModelId("test".into()),
            index: 0,
        };
        assert!(!regular.is_mmproj());
    }

    #[test]
    fn mmproj_for_creates_sentinel() {
        let shard = ShardId::mmproj_for(ModelId("test".into()));
        assert_eq!(shard.index, MMPROJ_SHARD_INDEX);
        assert!(shard.is_mmproj());
        assert_eq!(shard.model_id, ModelId("test".into()));
    }

    #[test]
    fn mmproj_holders_tracking() {
        let registry = ModelRegistry::new();
        let model_id = ModelId("vlm".into());
        let node_a = NodeId([1u8; 32]);
        let node_b = NodeId([2u8; 32]);

        // Initially no holders
        assert!(registry.mmproj_holders(&model_id).is_empty());

        // Register mmproj sentinel shard
        let sentinel = ShardId::mmproj_for(model_id.clone());
        registry.record_shard_holder(sentinel.clone(), node_a.clone());
        assert_eq!(registry.mmproj_holders(&model_id).len(), 1);

        registry.record_shard_holder(sentinel.clone(), node_b.clone());
        assert_eq!(registry.mmproj_holders(&model_id).len(), 2);

        // Remove one holder
        registry.remove_shard_holder(&sentinel, &node_a);
        let holders = registry.mmproj_holders(&model_id);
        assert_eq!(holders.len(), 1);
        assert_eq!(holders[0], node_b);
    }

    #[test]
    fn shard_announce_includes_mmproj_sentinel() {
        let registry = ModelRegistry::new();
        let model_id = ModelId("vlm".into());
        let node = NodeId([1u8; 32]);

        // Register regular shard and mmproj sentinel
        registry.record_shard_holder(
            ShardId {
                model_id: model_id.clone(),
                index: 0,
            },
            node.clone(),
        );
        registry.record_shard_holder(ShardId::mmproj_for(model_id.clone()), node.clone());

        let entries = registry.all_shard_entries();
        assert_eq!(entries.len(), 2);

        let has_mmproj = entries.iter().any(|(sid, _)| sid.is_mmproj());
        assert!(has_mmproj);
    }
}
