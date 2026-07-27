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
    /// Uncapped global holder count from the most recent DHT GetProviders
    /// response, keyed by shard. The bounded `shard_holders` cache above
    /// caps at MAX_HOLDERS_PER_SHARD for routing economy, but the prune
    /// score's redundancy_ratio needs the true swarm-wide count to detect
    /// severely over-replicated shards (gotcha: at 1000-node scale, a 50-cap
    /// cache pegs at 50 regardless of actual replication). Written only by
    /// `record_global_holder_count` from the DHT query result; readers fall
    /// back to the cached count if no DHT data is available.
    global_holder_count: DashMap<ShardId, u32>,
    /// Local node ID — never evicted from holder sets.
    local_node_id: Option<NodeId>,
}

impl ModelRegistry {
    pub fn new() -> Self {
        Self {
            manifests: DashMap::new(),
            shard_holders: DashMap::new(),
            node_shards: DashMap::new(),
            global_holder_count: DashMap::new(),
            local_node_id: None,
        }
    }

    /// Create a registry with a known local node ID.
    /// The local node is never evicted from bounded holder sets.
    #[cfg(test)]
    pub fn with_local_node(local_node_id: NodeId) -> Self {
        Self {
            manifests: DashMap::new(),
            shard_holders: DashMap::new(),
            node_shards: DashMap::new(),
            global_holder_count: DashMap::new(),
            local_node_id: Some(local_node_id),
        }
    }

    /// Set the local node ID after construction (e.g., after loading from DB).
    pub fn set_local_node_id(&mut self, node_id: NodeId) {
        self.local_node_id = Some(node_id);
    }

    /// Register a model manifest.
    pub fn register_manifest(&self, manifest: ModelManifest) {
        // Universal net against copied-folder model names (`<model>.FULLBACKUP`,
        // `<model>.old`). This is the single point every adoption path funnels
        // through — gossip ingress, DB reload on startup, local disk scan,
        // acquisition — so rejecting here keeps a backup-copy identity out of
        // the registry regardless of how it arrived, and stops it being
        // persisted and re-gossiped. See `manifest::is_backup_artifact_id`.
        if crate::model::manifest::is_backup_artifact_id(&manifest.id.0) {
            tracing::warn!(
                model = %manifest.id,
                "Refusing manifest with a backup-copy name — a model's identity \
                 must come from the model, not a copied directory. Rename it to \
                 the real model id if this is genuine."
            );
            return;
        }
        tracing::info!(
            model = %manifest.id,
            name = %manifest.name,
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
        // Update reverse index BEFORE dropping shard_holders guard so concurrent
        // remove_peer_from_all_shards sees a consistent state (no half-update window).
        self.node_shards
            .entry(node_id)
            .or_default()
            .insert(shard_id);
        drop(entry);
    }

    /// Remove a node from shard holders (e.g., node went offline).
    /// Maintains reverse index.
    pub fn remove_shard_holder(&self, shard_id: &ShardId, node_id: &NodeId) {
        if let Some(mut holders) = self.shard_holders.get_mut(shard_id) {
            holders.remove(node_id);
        }
        // Remove empty tombstones atomically — `remove_if` holds the
        // shard lock for both the empty check and the removal, so a
        // concurrent `record_shard_holder` inserting a fresh holder
        // can't be silently dropped between a separate check + remove
        // (the prior two-step pattern lost holders under peer churn).
        self.shard_holders
            .remove_if(shard_id, |_, holders| holders.is_empty());
        if let Some(mut shards) = self.node_shards.get_mut(node_id) {
            shards.remove(shard_id);
        }
    }

    /// Make `node_id`'s holder records for `model_id` exactly `keep`, dropping
    /// any shard of that model it is no longer listed as holding.
    ///
    /// This is the only way a node's claim is ever retracted by someone other
    /// than the node itself. `remove_shard_holder` is always called with the
    /// LOCAL node id — a peer dropping a shard had no way to tell us, so its
    /// stale claim survived for as long as it stayed connected and the
    /// scheduler kept routing layers to a node that could not serve them.
    ///
    /// Scoped to one model on purpose: a `ShardAnnounce` is only ever complete
    /// for the models it declares, so retracting beyond them would delete
    /// records the announcement says nothing about.
    ///
    /// Returns the number of records removed.
    pub fn retain_node_shards_for_model(
        &self,
        model_id: &ModelId,
        node_id: &NodeId,
        keep: &std::collections::HashSet<u32>,
    ) -> usize {
        // Collect first: mutating shard_holders while iterating it can deadlock
        // on the same DashMap shard.
        let stale: Vec<ShardId> = self
            .node_shards
            .get(node_id)
            .map(|shards| {
                shards
                    .iter()
                    .filter(|sid| sid.model_id == *model_id && !keep.contains(&sid.index))
                    .cloned()
                    .collect()
            })
            .unwrap_or_default();

        for shard_id in &stale {
            self.remove_shard_holder(shard_id, node_id);
        }
        stale.len()
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
    #[cfg(test)]
    pub fn shard_holder_count(&self, shard_id: &ShardId) -> usize {
        self.shard_holders
            .get(shard_id)
            .map(|v| v.len())
            .unwrap_or(0)
    }

    /// Get the shard indices held locally by a node for a given model.
    /// Convenience wrapper around `local_shard_indices_in` for call
    /// sites that don't already have the manifest in scope.
    pub fn local_shard_indices(
        &self,
        model_id: &crate::types::ModelId,
        node_id: &NodeId,
    ) -> Vec<u32> {
        self.get_manifest(model_id)
            .map(|m| self.local_shard_indices_in(&m, node_id))
            .unwrap_or_default()
    }

    /// Filter the shards of an already-resolved manifest down to indices
    /// held by `node_id`. Lets hot-path callers (per-token LayerForward)
    /// reuse a manifest they already fetched rather than paying for a
    /// second `get_manifest` clone.
    pub fn local_shard_indices_in(
        &self,
        manifest: &crate::types::ModelManifest,
        node_id: &NodeId,
    ) -> Vec<u32> {
        manifest
            .shards
            .iter()
            .filter(|s| {
                let sid = ShardId {
                    model_id: manifest.id.clone(),
                    index: s.index,
                };
                self.shard_holders(&sid).contains(node_id)
            })
            .map(|s| s.index)
            .collect()
    }

    /// Merge holders discovered via DHT provider queries into the cache.
    /// Updates timestamps for existing entries and adds new ones (with eviction).
    pub fn merge_dht_providers(&self, shard_id: &ShardId, providers: &[NodeId]) {
        for node_id in providers {
            self.record_shard_holder(shard_id.clone(), node_id.clone());
        }
    }

    /// Record the swarm-wide provider count from a DHT GetProviders response.
    /// The count comes from the raw PeerId set (before NodeId resolution drops
    /// any) so it reflects what the DHT reports, independent of our local cache
    /// cap. Overwrites any previous reading.
    pub fn record_global_holder_count(&self, shard_id: ShardId, count: u32) {
        self.global_holder_count.insert(shard_id, count);
    }

    /// Best-effort uncapped holder count for a shard. Returns the most recent
    /// DHT-reported count if any query has resolved, else `None` — caller
    /// should fall back to the cached `shard_holders().len()` for the local
    /// view.
    pub fn global_holder_count(&self, shard_id: &ShardId) -> Option<u32> {
        self.global_holder_count.get(shard_id).map(|v| *v)
    }

    /// Get a model manifest by ID.
    pub fn get_manifest(&self, model_id: &ModelId) -> Option<ModelManifest> {
        self.manifests.get(model_id).map(|v| v.clone())
    }

    /// Shard ids whose layer ranges overlap `layer_range`.
    ///
    /// **This is the answer to "which shards does a pipeline segment actually
    /// touch?", and it is not `segment.shard_id`.** That field holds a single
    /// shard — the first one of the candidate that served the segment — while a
    /// segment routinely spans several. Reading it as though it named the whole
    /// span has already caused one bug: a segment failing on `blk.10` (shard 2)
    /// retracted shard 0, which the holder genuinely had. Sub-range routing
    /// makes multi-shard segments the normal case rather than an edge case, so
    /// every consumer that needs the span must come through here.
    ///
    /// Returns an empty Vec when the manifest is unknown; callers guarding a
    /// destructive operation must treat that as "cannot prove it is unused"
    /// rather than as "unused".
    pub fn shards_overlapping_layers(
        &self,
        model_id: &ModelId,
        layer_range: (u32, u32),
    ) -> Vec<crate::types::ShardId> {
        let Some(manifest) = self.get_manifest(model_id) else {
            return Vec::new();
        };
        manifest
            .shards
            .iter()
            // Half-open ranges: [a,b) overlaps [c,d) iff a < d && c < b.
            .filter(|s| s.layer_range.0 < layer_range.1 && layer_range.0 < s.layer_range.1)
            .map(|s| crate::types::ShardId {
                model_id: model_id.clone(),
                index: s.index,
            })
            .collect()
    }

    /// Shards a pipeline segment actually reads. Falls back to the segment's own
    /// `shard_id` when the manifest is unknown, so a caller never sees an empty
    /// span for a segment that is demonstrably executing.
    pub fn shards_spanned_by_segment(
        &self,
        segment: &swarmllm_types::PipelineSegment,
    ) -> Vec<crate::types::ShardId> {
        let spanned =
            self.shards_overlapping_layers(&segment.shard_id.model_id, segment.layer_range);
        if spanned.is_empty() {
            vec![segment.shard_id.clone()]
        } else {
            spanned
        }
    }

    /// Resolve a manifest for a loaded model by trying slug, display name, and manifest name field.
    pub fn resolve_manifest_by_name(&self, display_name: &str) -> Option<ModelManifest> {
        let slug = crate::types::slugify_model_name(display_name);
        self.get_manifest(&ModelId(slug))
            .or_else(|| self.get_manifest(&ModelId(display_name.to_string())))
            .or_else(|| self.models().into_iter().find(|m| m.name == display_name))
    }

    /// Human-friendly display name for a model (falls back to raw model ID).
    pub fn display_name(&self, model_id: &ModelId) -> String {
        self.manifests
            .get(model_id)
            .map(|m| m.name.clone())
            .unwrap_or_else(|| model_id.0.clone())
    }

    /// Get all known model manifests.
    pub fn models(&self) -> Vec<ModelManifest> {
        self.manifests.iter().map(|v| v.value().clone()).collect()
    }

    /// Build a "model not found" error carrying the list of known models.
    ///
    /// Callers get `SwarmError::ModelNotAvailable` (mapped to HTTP 404 by the
    /// API layer). The message varies based on whether the registry is empty
    /// to give users a more actionable hint.
    pub fn model_not_found_error(&self, model_id: &ModelId) -> SwarmError {
        let available: Vec<String> = self.models().iter().map(|m| m.id.0.clone()).collect();
        let msg = if available.is_empty() {
            format!(
                "Model '{}' not found. No models are available — download shards first.",
                model_id.0
            )
        } else {
            format!(
                "Model '{}' not found. Available models: {}",
                model_id.0,
                available.join(", ")
            )
        };
        SwarmError::ModelNotAvailable(ModelId(msg))
    }

    /// Get the number of registered models.
    #[cfg(test)]
    pub fn model_count(&self) -> usize {
        self.manifests.len()
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
        // Drop the uncapped DHT-derived global count too — stale entries here
        // would falsely inflate a future shard's redundancy_ratio if the same
        // ShardId got reused.
        self.global_holder_count
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
        let mut purge_backup_ids: Vec<String> = Vec::new();
        for (_key, value) in entries {
            match serde_json::from_slice::<ModelManifest>(&value) {
                Ok(manifest) => {
                    // A backup-copy id (`<model>.FULLBACKUP`, `<model>.old`)
                    // persisted before the register_manifest guard existed must
                    // NOT be re-adopted on reload — otherwise the phantom model
                    // resurrects on every restart. This is the one manifest-entry
                    // path that bypasses register_manifest, so it needs its own
                    // guard. Skip it and purge the stale row so it stops coming
                    // back. See `manifest::is_backup_artifact_id`.
                    if crate::model::manifest::is_backup_artifact_id(&manifest.id.0) {
                        tracing::warn!(
                            model = %manifest.id,
                            "Dropping backup-copy manifest from DB on load (and purging the row)"
                        );
                        purge_backup_ids.push(manifest.id.0.clone());
                        continue;
                    }
                    registry.manifests.insert(manifest.id.clone(), manifest);
                }
                Err(e) => {
                    tracing::warn!(error = %e, "Failed to deserialize model manifest from DB");
                }
            }
        }
        for id in purge_backup_ids {
            if let Err(e) = db.remove("model_meta", &id) {
                tracing::debug!(model = %id, error = %e, "Failed to purge backup-copy manifest row");
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

    fn test_manifest(id: &str, name: &str) -> ModelManifest {
        ModelManifest {
            id: ModelId(id.into()),
            name: name.into(),
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
        }
    }

    #[test]
    fn register_and_retrieve_manifest() {
        let registry = ModelRegistry::new();
        let manifest = test_manifest("test", "Test");

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

    /// The reported failure: a peer deleted shards 0-4 and restarted, but the
    /// scheduler kept assigning it layers 0..32 because nothing could retract
    /// a still-connected peer's claim.
    #[test]
    fn retain_node_shards_drops_what_a_peer_no_longer_holds() {
        let registry = ModelRegistry::new();
        let model = ModelId("llama-3.1-8b".into());
        let peer = NodeId([7u8; 32]);

        for idx in 0..9u32 {
            registry.record_shard_holder(
                ShardId {
                    model_id: model.clone(),
                    index: idx,
                },
                peer.clone(),
            );
        }

        // Peer re-announces holding only 5-8.
        let keep: std::collections::HashSet<u32> = (5..9).collect();
        let dropped = registry.retain_node_shards_for_model(&model, &peer, &keep);
        assert_eq!(dropped, 5, "shards 0-4 should be retracted");

        for idx in 0..5u32 {
            assert!(
                registry
                    .shard_holders(&ShardId {
                        model_id: model.clone(),
                        index: idx
                    })
                    .is_empty(),
                "shard {idx} should have no holders left"
            );
        }
        for idx in 5..9u32 {
            assert!(
                registry
                    .shard_holders(&ShardId {
                        model_id: model.clone(),
                        index: idx
                    })
                    .contains(&peer),
                "shard {idx} should still be held"
            );
        }
    }

    /// Retraction is scoped to one model — an announce complete for model A
    /// says nothing about model B, and must not delete it.
    #[test]
    fn retain_node_shards_leaves_other_models_alone() {
        let registry = ModelRegistry::new();
        let (a, b) = (ModelId("model-a".into()), ModelId("model-b".into()));
        let peer = NodeId([7u8; 32]);

        registry.record_shard_holder(
            ShardId {
                model_id: a.clone(),
                index: 0,
            },
            peer.clone(),
        );
        registry.record_shard_holder(
            ShardId {
                model_id: b.clone(),
                index: 0,
            },
            peer.clone(),
        );

        // Complete for A with nothing held → drops A only.
        let dropped = registry.retain_node_shards_for_model(&a, &peer, &Default::default());
        assert_eq!(dropped, 1);
        assert!(registry
            .shard_holders(&ShardId {
                model_id: a,
                index: 0
            })
            .is_empty());
        assert!(registry
            .shard_holders(&ShardId {
                model_id: b,
                index: 0
            })
            .contains(&peer));
    }

    /// One peer retracting must not evict another peer's holding.
    #[test]
    fn retain_node_shards_only_touches_the_announcing_node() {
        let registry = ModelRegistry::new();
        let model = ModelId("m".into());
        let (peer_a, peer_b) = (NodeId([1u8; 32]), NodeId([2u8; 32]));
        let sid = ShardId {
            model_id: model.clone(),
            index: 0,
        };

        registry.record_shard_holder(sid.clone(), peer_a.clone());
        registry.record_shard_holder(sid.clone(), peer_b.clone());

        registry.retain_node_shards_for_model(&model, &peer_a, &Default::default());
        let holders = registry.shard_holders(&sid);
        assert!(!holders.contains(&peer_a));
        assert!(holders.contains(&peer_b), "peer B must be untouched");
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
    fn remove_last_shard_holder_cleans_tombstone() {
        // R142.9: empty-tombstone cleanup must be atomic with the empty
        // check (remove_if). Verify a single-holder removal clears the
        // entry entirely (not just empties the inner HashSet) so a
        // subsequent record_shard_holder starts from a clean slot.
        let registry = ModelRegistry::new();
        let shard_id = ShardId {
            model_id: ModelId("test".into()),
            index: 0,
        };
        let node_a = NodeId([1u8; 32]);
        registry.record_shard_holder(shard_id.clone(), node_a.clone());
        assert_eq!(registry.shard_holder_count(&shard_id), 1);

        registry.remove_shard_holder(&shard_id, &node_a);
        assert_eq!(registry.shard_holder_count(&shard_id), 0);
        assert!(registry.shard_holders(&shard_id).is_empty());

        // Re-inserting a fresh holder after the last was removed must work.
        let node_b = NodeId([2u8; 32]);
        registry.record_shard_holder(shard_id.clone(), node_b.clone());
        assert_eq!(registry.shard_holders(&shard_id), vec![node_b]);
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
    fn global_holder_count_overrides_local_cap() {
        let registry = ModelRegistry::new();
        let shard_id = ShardId {
            model_id: ModelId("test".into()),
            index: 0,
        };

        // No DHT data yet — global_holder_count returns None.
        assert!(registry.global_holder_count(&shard_id).is_none());

        // Cache has 3 holders, DHT reports 247 — the uncapped figure wins
        // for redundancy_ratio purposes.
        for i in 1u8..=3 {
            let mut bytes = [0u8; 32];
            bytes[0] = i;
            registry.record_shard_holder(shard_id.clone(), NodeId(bytes));
        }
        registry.record_global_holder_count(shard_id.clone(), 247);

        assert_eq!(registry.shard_holders(&shard_id).len(), 3);
        assert_eq!(registry.global_holder_count(&shard_id), Some(247));

        // remove_all_model_shards must drop the global entry too — a future
        // shard reusing the ShardId must not inherit a stale 247.
        registry.remove_all_model_shards(&ModelId("test".into()));
        assert!(registry.global_holder_count(&shard_id).is_none());
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

        registry.register_manifest(test_manifest("a", "A"));

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
