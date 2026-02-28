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

        tracing::debug!(
            model = %manifest.id,
            shards = manifest.shard_count,
            "Loaded manifest"
        );

        Ok(manifest)
    }

    /// Verify the manifest hash by recomputing it from the manifest content
    /// (excluding the manifest_hash field itself).
    pub fn verify_hash(&self) -> Result<(), SwarmError> {
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
            hasher.update(&shard.byte_start.unwrap_or(0).to_le_bytes());
            hasher.update(&shard.byte_end.unwrap_or(0).to_le_bytes());
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
    pub fn shard_path(data_dir: &Path, model_id: &str, index: u32) -> std::path::PathBuf {
        data_dir
            .join("models")
            .join(model_id)
            .join(format!("shard_{index:03}.bin"))
    }
}

/// Build `ShardInfo` entries from on-disk shard files and GGUF tensor metadata.
///
/// Detects v2 layer-aligned shards (repacked files matching layout sizes) and
/// falls back to v1 byte-range shards otherwise. Used by both daemon startup
/// manifest regeneration and the admin API manifest generation.
pub fn build_shard_infos(
    model_dir: &Path,
    meta: &crate::inference::split::GgufTensorMeta,
    shard_count: u32,
    shard_size: u64,
    total_size: u64,
) -> Vec<crate::types::ShardInfo> {
    let layouts = crate::inference::split::compute_layer_shard_layouts(meta, shard_count);
    let v2_matches = !layouts.is_empty()
        && layouts.iter().all(|layout| {
            let shard_path = model_dir.join(format!("shard_{:03}.bin", layout.index));
            if let Ok(file_meta) = std::fs::metadata(&shard_path) {
                let diff = (file_meta.len() as i64 - layout.size_bytes as i64).unsigned_abs();
                diff <= layout.size_bytes / 100 + 1
            } else {
                false
            }
        });

    if v2_matches {
        // V2 layer-aligned shards
        tracing::info!(shard_count, "Building v2 layer-aligned shard infos");
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
                crate::types::ShardInfo {
                    index: layout.index,
                    layer_range: (layout.layer_start, layout.layer_end),
                    size_bytes: file_size,
                    hash,
                    byte_start: Some(0),
                    byte_end: Some(file_size),
                }
            })
            .collect()
    } else {
        // V1 byte-range shards (legacy, needed for partial downloads)
        (0..shard_count)
            .map(|idx| {
                let shard_path = model_dir.join(format!("shard_{idx:03}.bin"));
                let expected_size = if idx == shard_count - 1 {
                    total_size - (idx as u64) * shard_size
                } else {
                    shard_size
                };
                let file_size = std::fs::metadata(&shard_path)
                    .map(|m| m.len())
                    .unwrap_or(expected_size);
                let (ls, le) = crate::inference::split::compute_local_layer_range(
                    meta, shard_size, &[idx],
                );
                let hash = if shard_path.exists() {
                    match std::fs::read(&shard_path) {
                        Ok(data) => *blake3::hash(&data).as_bytes(),
                        Err(_) => [0u8; 32],
                    }
                } else {
                    [0u8; 32]
                };
                crate::types::ShardInfo {
                    index: idx,
                    layer_range: (ls as u32, le as u32),
                    size_bytes: file_size,
                    hash,
                    byte_start: None,
                    byte_end: None,
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use crate::types::*;

    fn test_manifest() -> ModelManifest {
        ModelManifest {
            schema_version: 1,
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
                byte_start: None,
                byte_end: None,
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
    fn verify_hash_with_wrong_hash() {
        let manifest = test_manifest(); // manifest_hash is all zeros
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
