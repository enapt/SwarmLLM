use crate::model::registry::ModelRegistry;
use crate::types::{ModelManifest, ShardId};

/// Select the rarest shards for a model (fewest known holders).
///
/// This implements the "rarest first" strategy similar to BitTorrent,
/// prioritizing shards that have the fewest replicas in the network.
///
/// If `local_shards` is provided, already-held shard indices are excluded
/// from the result so callers don't re-download shards they already have.
pub fn select_rarest_shards(manifest: &ModelManifest, registry: &ModelRegistry) -> Vec<ShardId> {
    select_rarest_shards_excluding(manifest, registry, None)
}

/// Select the rarest shards, optionally excluding already-held shard indices.
pub fn select_rarest_shards_excluding(
    manifest: &ModelManifest,
    registry: &ModelRegistry,
    local_shards: Option<&[u32]>,
) -> Vec<ShardId> {
    let mut shard_counts: Vec<(ShardId, usize)> = manifest
        .shards
        .iter()
        .filter(|shard| {
            // Exclude shards we already hold locally
            match local_shards {
                Some(held) => !held.contains(&shard.index),
                None => true,
            }
        })
        .map(|shard| {
            let shard_id = ShardId {
                model_id: manifest.id.clone(),
                index: shard.index,
            };
            let holders = registry.shard_holders(&shard_id);
            (shard_id, holders.len())
        })
        .collect();

    // Sort by holder count ascending (rarest first)
    shard_counts.sort_by_key(|(_, count)| *count);

    tracing::debug!(
        model = %manifest.id,
        candidates = shard_counts.len(),
        rarest_holders = shard_counts.first().map(|(_, c)| *c).unwrap_or(0),
        "DIAG: select_rarest_shards"
    );

    shard_counts.into_iter().map(|(id, _)| id).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::*;

    fn test_manifest() -> ModelManifest {
        ModelManifest {
            schema_version: 2,
            id: ModelId("test-model".into()),
            name: "Test Model".into(),
            architecture: ModelArchitecture::Llama,
            num_layers: 4,
            num_params_billions: 0.001,
            quantization: Quantization::Q4KM,
            total_size_bytes: 4096,
            shard_count: 2,
            shards: vec![
                ShardInfo {
                    index: 0,
                    layer_range: (0, 2),
                    size_bytes: 2048,
                    hash: [0u8; 32],
                    tensors: vec![],
                },
                ShardInfo {
                    index: 1,
                    layer_range: (2, 4),
                    size_bytes: 2048,
                    hash: [0u8; 32],
                    tensors: vec![],
                },
            ],
            tokenizer_hash: [0u8; 32],
            manifest_hash: [0u8; 32],
            publisher: NodeId([0u8; 32]),
            publish_date: chrono::Utc::now(),
            license: "MIT".into(),
            mmproj: None,
        }
    }

    #[test]
    fn select_rarest_shards_orders_by_holder_count() {
        let manifest = test_manifest();
        let registry = ModelRegistry::new();

        // Add holders for shard 0 but not shard 1
        registry.record_shard_holder(
            ShardId {
                model_id: ModelId("test-model".into()),
                index: 0,
            },
            NodeId([1u8; 32]),
        );
        registry.record_shard_holder(
            ShardId {
                model_id: ModelId("test-model".into()),
                index: 0,
            },
            NodeId([2u8; 32]),
        );

        let rarest = select_rarest_shards(&manifest, &registry);
        assert_eq!(rarest.len(), 2);
        // Shard 1 should be first (0 holders < 2 holders)
        assert_eq!(rarest[0].index, 1);
        assert_eq!(rarest[1].index, 0);
    }
}
