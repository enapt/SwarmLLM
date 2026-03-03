use std::path::Path;

use crate::error::SwarmError;
use crate::types::{Blake3Hash, ModelManifest};

impl ModelManifest {
    /// Load a manifest from a model directory.
    ///
    /// Reads `manifest.json` from the given directory and deserializes it.
    pub fn load_from_dir(dir: &Path) -> Result<Self, SwarmError> {
        let manifest_path = dir.join("manifest.json");
        if !manifest_path.exists() {
            return Err(SwarmError::Config(format!(
                "Manifest not found: {}",
                manifest_path.display()
            )));
        }

        let contents = std::fs::read_to_string(&manifest_path).map_err(SwarmError::Io)?;
        let manifest: ModelManifest =
            serde_json::from_str(&contents).map_err(SwarmError::Serialization)?;

        // Reject legacy manifests at load time — v2 tensor entries are required
        if let Err(e) = manifest.validate_version() {
            return Err(SwarmError::Config(format!(
                "Rejecting manifest {}: {e}. Delete and re-download shards.",
                manifest_path.display()
            )));
        }

        tracing::debug!(
            model = %manifest.id,
            shards = manifest.shard_count,
            "Loaded manifest"
        );

        Ok(manifest)
    }

    /// Verify the manifest hash by recomputing it from the manifest content
    /// (excluding the manifest_hash field itself).
    ///
    /// Allows zero-hash manifests (not yet computed, e.g. from local HF downloads).
    /// For network-received manifests, use `verify_hash_strict()` instead.
    pub fn verify_hash(&self) -> Result<(), SwarmError> {
        // Allow manifests with a zero hash (not yet computed, e.g. from partial
        // HF downloads before hash is set). Local-only — see verify_hash_strict.
        if self.manifest_hash == [0u8; 32] {
            return Ok(());
        }
        let computed = self.compute_hash();
        if computed != self.manifest_hash {
            return Err(SwarmError::ShardIntegrity {
                expected: hex::encode(self.manifest_hash),
                actual: hex::encode(computed),
            });
        }
        Ok(())
    }

    /// Strict hash verification for network-received manifests.
    /// Rejects zero-hash manifests to prevent gossip-based poisoning.
    pub fn verify_hash_strict(&self) -> Result<(), SwarmError> {
        if self.manifest_hash == [0u8; 32] {
            return Err(SwarmError::ShardIntegrity {
                expected: "non-zero manifest hash".into(),
                actual: "zero hash (unsigned manifest)".into(),
            });
        }
        let computed = self.compute_hash();
        if computed != self.manifest_hash {
            return Err(SwarmError::ShardIntegrity {
                expected: hex::encode(self.manifest_hash),
                actual: hex::encode(computed),
            });
        }
        Ok(())
    }

    /// Compute the BLAKE3 hash of the manifest content.
    ///
    /// Hashes a canonical representation that excludes the manifest_hash field.
    pub fn compute_hash(&self) -> Blake3Hash {
        let mut hasher = blake3::Hasher::new();
        hasher.update(self.id.0.as_bytes());
        hasher.update(self.name.as_bytes());
        // Include publisher, license, architecture, quantization in hash
        hasher.update(&self.publisher.0);
        hasher.update(self.license.as_bytes());
        hasher.update(format!("{:?}", self.architecture).as_bytes());
        hasher.update(format!("{:?}", self.quantization).as_bytes());
        hasher.update(&self.num_layers.to_le_bytes());
        hasher.update(&self.num_params_billions.to_le_bytes());
        hasher.update(&self.total_size_bytes.to_le_bytes());
        hasher.update(self.publish_date.to_rfc3339().as_bytes());
        hasher.update(&self.shard_count.to_le_bytes());
        for shard in &self.shards {
            hasher.update(&shard.index.to_le_bytes());
            hasher.update(&shard.layer_range.0.to_le_bytes());
            hasher.update(&shard.layer_range.1.to_le_bytes());
            hasher.update(&shard.size_bytes.to_le_bytes());
            hasher.update(&shard.hash);
            for entry in &shard.tensors {
                hasher.update(entry.name.as_bytes());
                hasher.update(&entry.gguf_offset.to_le_bytes());
                hasher.update(&entry.shard_offset.to_le_bytes());
                hasher.update(&entry.size.to_le_bytes());
            }
        }
        hasher.update(&self.tokenizer_hash);
        *hasher.finalize().as_bytes()
    }

    /// Save the manifest to a model directory as `manifest.json`.
    pub fn save_to_dir(&self, dir: &Path) -> Result<(), SwarmError> {
        std::fs::create_dir_all(dir).map_err(SwarmError::Io)?;
        let manifest_path = dir.join("manifest.json");
        let json = serde_json::to_string_pretty(self).map_err(SwarmError::Serialization)?;
        std::fs::write(manifest_path, json).map_err(SwarmError::Io)?;
        Ok(())
    }

    /// Get the path to a specific shard file within the data directory.
    /// Sanitizes model_id to prevent path traversal attacks.
    pub fn shard_path(data_dir: &Path, model_id: &str, index: u32) -> std::path::PathBuf {
        let safe_id = model_id.replace(['/', '\\'], "_").replace("..", "_");
        data_dir
            .join("models")
            .join(&safe_id)
            .join(format!("shard_{index:03}.bin"))
    }
}

/// Build ShardInfo entries from `LayerShardLayout` computed by `compute_layer_shard_layouts`.
///
/// For each layout, hashes the on-disk shard file (if present) and builds the
/// `ShardTensorEntry` list from the layout's tensor data.
pub fn build_shard_infos_from_layouts(
    model_dir: &Path,
    layouts: &[crate::inference::split::LayerShardLayout],
) -> Vec<crate::types::ShardInfo> {
    layouts
        .iter()
        .map(|layout| {
            let shard_path = model_dir.join(format!("shard_{:03}.bin", layout.index));
            let hash = if shard_path.exists() {
                match std::fs::read(&shard_path) {
                    Ok(data) => *blake3::hash(&data).as_bytes(),
                    Err(_) => [0u8; 32],
                }
            } else {
                [0u8; 32]
            };
            let file_size = std::fs::metadata(&shard_path)
                .map(|m| m.len())
                .unwrap_or(layout.size_bytes);

            // Build tensor entries with sequential shard-local offsets
            let mut shard_offset = 0u64;
            let tensors: Vec<crate::types::ShardTensorEntry> = layout
                .tensors
                .iter()
                .map(|(name, gguf_offset, size)| {
                    let entry = crate::types::ShardTensorEntry {
                        name: name.clone(),
                        gguf_offset: *gguf_offset,
                        shard_offset,
                        size: *size,
                    };
                    shard_offset += size;
                    entry
                })
                .collect();

            crate::types::ShardInfo {
                index: layout.index,
                layer_range: (layout.layer_start, layout.layer_end),
                size_bytes: file_size,
                hash,
                tensors,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use crate::types::*;

    fn test_manifest() -> ModelManifest {
        ModelManifest {
            schema_version: 2,
            id: ModelId("test-model".into()),
            name: "Test Model".into(),
            architecture: ModelArchitecture::Llama,
            num_layers: 2,
            num_params_billions: 0.001,
            quantization: Quantization::Q4KM,
            total_size_bytes: 1024,
            shard_count: 1,
            shards: vec![ShardInfo {
                index: 0,
                layer_range: (0, 2),
                size_bytes: 1024,
                hash: [0u8; 32],
                tensors: vec![],
            }],
            tokenizer_hash: [0u8; 32],
            manifest_hash: [0u8; 32],
            publisher: NodeId([0u8; 32]),
            publish_date: chrono::Utc::now(),
            license: "MIT".into(),
        }
    }

    #[test]
    fn compute_hash_is_deterministic() {
        let manifest = test_manifest();
        let hash1 = manifest.compute_hash();
        let hash2 = manifest.compute_hash();
        assert_eq!(hash1, hash2);
    }

    #[test]
    fn save_and_load_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let mut manifest = test_manifest();
        manifest.manifest_hash = manifest.compute_hash();

        manifest.save_to_dir(dir.path()).unwrap();
        let loaded = ModelManifest::load_from_dir(dir.path()).unwrap();

        assert_eq!(loaded.id, manifest.id);
        assert_eq!(loaded.name, manifest.name);
        assert_eq!(loaded.shard_count, manifest.shard_count);
    }

    #[test]
    fn verify_hash_with_correct_hash() {
        let mut manifest = test_manifest();
        manifest.manifest_hash = manifest.compute_hash();
        assert!(manifest.verify_hash().is_ok());
    }

    #[test]
    fn verify_hash_with_zero_hash_allowed() {
        let manifest = test_manifest(); // manifest_hash is all zeros
                                        // Zero hash should be allowed (manifest not yet signed)
        assert!(manifest.verify_hash().is_ok());
    }

    #[test]
    fn verify_hash_with_wrong_nonzero_hash() {
        let mut manifest = test_manifest();
        manifest.manifest_hash = [1u8; 32]; // non-zero but wrong
        let result = manifest.verify_hash();
        assert!(result.is_err());
    }

    #[test]
    fn shard_path_format() {
        let path = ModelManifest::shard_path(std::path::Path::new("/data"), "llama3-70b", 5);
        assert_eq!(
            path.to_string_lossy(),
            "/data/models/llama3-70b/shard_005.bin"
        );
    }
}
