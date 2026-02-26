use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use crate::error::SwarmError;
use crate::types::{ModelId, ShardInfo};

/// Sanitize a path component to prevent path traversal attacks.
/// Strips directory separators and rejects `..` sequences.
fn sanitize_path_component(s: &str) -> String {
    s.replace(['/', '\\'], "_")
        .replace("..", "_")
}

/// Manages shard files on disk — loading, verification, and storage.
pub struct ShardStore {
    data_dir: PathBuf,
}

impl ShardStore {
    pub fn new(data_dir: &Path) -> Self {
        Self {
            data_dir: data_dir.to_path_buf(),
        }
    }

    /// Get the path to a specific shard file.
    pub fn shard_path(&self, model_id: &ModelId, index: u32) -> PathBuf {
        // SECURITY: Sanitize model_id to prevent path traversal
        let safe_id = sanitize_path_component(&model_id.0);
        self.data_dir
            .join("models")
            .join(&safe_id)
            .join(format!("shard_{index:03}.bin"))
    }

    /// Get the models directory path.
    pub fn models_dir(&self) -> PathBuf {
        self.data_dir.join("models")
    }

    /// Verify a shard's BLAKE3 hash matches the expected value.
    pub fn verify_shard(&self, model_id: &ModelId, info: &ShardInfo) -> Result<(), SwarmError> {
        let path = self.shard_path(model_id, info.index);
        if !path.exists() {
            return Err(SwarmError::ShardNotFound(crate::types::ShardId {
                model_id: model_id.clone(),
                index: info.index,
            }));
        }

        let mut file = std::fs::File::open(&path).map_err(SwarmError::Io)?;
        let mut hasher = blake3::Hasher::new();
        let mut buf = [0u8; 64 * 1024]; // 64KB buffer

        loop {
            let n = file.read(&mut buf).map_err(SwarmError::Io)?;
            if n == 0 {
                break;
            }
            hasher.update(&buf[..n]);
        }

        let actual = *hasher.finalize().as_bytes();
        if actual != info.hash {
            // Quarantine the bad shard
            let quarantine_path = path.with_extension("bin.quarantine");
            let _ = std::fs::rename(&path, &quarantine_path);
            tracing::warn!(
                model = %model_id,
                shard = info.index,
                "Shard failed verification, quarantined"
            );

            return Err(SwarmError::ShardIntegrity {
                expected: hex::encode(info.hash),
                actual: hex::encode(actual),
            });
        }

        Ok(())
    }

    /// Scan the models directory and return all locally available, verified shards.
    ///
    /// Security: Only loads shards that have a valid manifest with a verified
    /// BLAKE3 hash. Shards without a manifest are rejected — this prevents
    /// arbitrary files on disk from being absorbed into the network.
    /// Each shard's content hash is verified against the manifest before inclusion.
    pub fn load_all_local(&self) -> Result<Vec<(ModelId, ShardInfo)>, SwarmError> {
        let models_dir = self.models_dir();
        if !models_dir.exists() {
            return Ok(vec![]);
        }

        let mut shards = Vec::new();
        let mut rejected = 0u32;

        let entries = std::fs::read_dir(&models_dir).map_err(SwarmError::Io)?;
        for entry in entries {
            let entry = entry.map_err(SwarmError::Io)?;
            let model_dir = entry.path();
            if !model_dir.is_dir() {
                continue;
            }

            let model_id_str = model_dir
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("")
                .to_string();
            let model_id = ModelId(model_id_str);

            // SECURITY: Require a manifest — reject directories without one.
            // This prevents arbitrary shard files from being absorbed into the network.
            let manifest_path = model_dir.join("manifest.json");
            if !manifest_path.exists() {
                tracing::warn!(
                    model = %model_id,
                    "Skipping model directory without manifest (unverified shards rejected)"
                );
                rejected += 1;
                continue;
            }

            match crate::types::ModelManifest::load_from_dir(&model_dir) {
                Ok(manifest) => {
                    // SECURITY: Verify the manifest's own integrity hash
                    if let Err(e) = manifest.verify_hash() {
                        tracing::warn!(
                            model = %model_id,
                            error = %e,
                            "Manifest hash verification failed — skipping (possible tampering)"
                        );
                        rejected += 1;
                        continue;
                    }

                    // Verify each shard's content hash against the manifest
                    for shard_info in &manifest.shards {
                        let shard_path = self.shard_path(&model_id, shard_info.index);
                        if !shard_path.exists() {
                            continue;
                        }

                        match self.verify_shard(&model_id, shard_info) {
                            Ok(()) => {
                                shards.push((model_id.clone(), shard_info.clone()));
                            }
                            Err(e) => {
                                tracing::warn!(
                                    model = %model_id,
                                    shard = shard_info.index,
                                    error = %e,
                                    "Shard verification failed on startup — quarantined"
                                );
                                rejected += 1;
                            }
                        }
                    }
                }
                Err(e) => {
                    tracing::warn!(
                        model = %model_id,
                        error = %e,
                        "Failed to load manifest, skipping"
                    );
                    rejected += 1;
                }
            }
        }

        if rejected > 0 {
            tracing::warn!(
                rejected,
                "Rejected unverified or corrupt model data on startup"
            );
        }
        tracing::info!(
            count = shards.len(),
            "Loaded verified local shard inventory"
        );
        Ok(shards)
    }

    /// Write a chunk of shard data to disk (for progressive downloads).
    pub fn write_chunk(
        &self,
        model_id: &ModelId,
        index: u32,
        offset: u64,
        data: &[u8],
    ) -> Result<(), SwarmError> {
        let path = self.shard_path(model_id, index);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(SwarmError::Io)?;
        }

        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(false)
            .open(&path)
            .map_err(SwarmError::Io)?;

        file.seek(SeekFrom::Start(offset)).map_err(SwarmError::Io)?;
        file.write_all(data).map_err(SwarmError::Io)?;

        Ok(())
    }

    /// Reconstruct a full GGUF file by concatenating shard files in order.
    ///
    /// Byte-range shards are contiguous slices of the original GGUF,
    /// so concatenating shard_000.bin + shard_001.bin + ... recreates the exact file.
    /// Returns the path to the reconstructed GGUF.
    pub fn reconstruct_gguf(
        &self,
        model_id: &ModelId,
        manifest: &crate::types::ModelManifest,
    ) -> Result<PathBuf, SwarmError> {
        let model_dir = self.models_dir().join(&model_id.0);
        let gguf_path = model_dir.join("model.gguf");

        // Skip if already reconstructed
        if gguf_path.exists() {
            let meta = std::fs::metadata(&gguf_path).map_err(SwarmError::Io)?;
            if meta.len() == manifest.total_size_bytes {
                tracing::info!(
                    model = %model_id,
                    "GGUF already reconstructed, skipping"
                );
                return Ok(gguf_path);
            }
        }

        tracing::info!(
            model = %model_id,
            shards = manifest.shard_count,
            total_bytes = manifest.total_size_bytes,
            "Reconstructing GGUF from shards"
        );

        let mut out = std::fs::File::create(&gguf_path).map_err(SwarmError::Io)?;
        let mut total_written: u64 = 0;

        // Shards must be concatenated in order
        let mut sorted_shards = manifest.shards.clone();
        sorted_shards.sort_by_key(|s| s.index);

        for shard_info in &sorted_shards {
            let shard_path = self.shard_path(model_id, shard_info.index);
            if !shard_path.exists() {
                return Err(SwarmError::ShardNotFound(crate::types::ShardId {
                    model_id: model_id.clone(),
                    index: shard_info.index,
                }));
            }

            let mut input = std::fs::File::open(&shard_path).map_err(SwarmError::Io)?;
            let copied = std::io::copy(&mut input, &mut out).map_err(SwarmError::Io)?;
            total_written += copied;

            tracing::debug!(
                model = %model_id,
                shard = shard_info.index,
                bytes = copied,
                "Appended shard to GGUF"
            );
        }

        tracing::info!(
            model = %model_id,
            path = %gguf_path.display(),
            bytes = total_written,
            "GGUF reconstruction complete"
        );

        Ok(gguf_path)
    }

    /// Delete a shard file from disk.
    pub fn delete_shard(&self, model_id: &ModelId, index: u32) -> Result<(), SwarmError> {
        let path = self.shard_path(model_id, index);
        if path.exists() {
            std::fs::remove_file(&path).map_err(SwarmError::Io)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shard_path_format() {
        let store = ShardStore::new(Path::new("/data"));
        let path = store.shard_path(&ModelId("test-model".into()), 5);
        assert_eq!(
            path.to_string_lossy(),
            "/data/models/test-model/shard_005.bin"
        );
    }

    #[test]
    fn write_and_verify_shard() {
        let dir = tempfile::tempdir().unwrap();
        let store = ShardStore::new(dir.path());
        let model_id = ModelId("test".into());
        let data = b"test shard data for verification";

        // Write shard
        store.write_chunk(&model_id, 0, 0, data).unwrap();

        // Compute expected hash
        let expected_hash = *blake3::hash(data).as_bytes();

        let info = ShardInfo {
            index: 0,
            layer_range: (0, 1),
            size_bytes: data.len() as u64,
            hash: expected_hash,
        };

        // Verify should succeed
        assert!(store.verify_shard(&model_id, &info).is_ok());
    }

    #[test]
    fn verify_corrupt_shard_fails() {
        let dir = tempfile::tempdir().unwrap();
        let store = ShardStore::new(dir.path());
        let model_id = ModelId("test".into());
        let data = b"test shard data";

        store.write_chunk(&model_id, 0, 0, data).unwrap();

        let info = ShardInfo {
            index: 0,
            layer_range: (0, 1),
            size_bytes: data.len() as u64,
            hash: [0xFF; 32], // Wrong hash
        };

        assert!(store.verify_shard(&model_id, &info).is_err());

        // Verify file was quarantined
        let quarantine = store
            .shard_path(&model_id, 0)
            .with_extension("bin.quarantine");
        assert!(quarantine.exists());
    }

    #[test]
    fn load_all_local_empty() {
        let dir = tempfile::tempdir().unwrap();
        let store = ShardStore::new(dir.path());
        let shards = store.load_all_local().unwrap();
        assert!(shards.is_empty());
    }

    #[test]
    fn delete_shard_removes_file() {
        let dir = tempfile::tempdir().unwrap();
        let store = ShardStore::new(dir.path());
        let model_id = ModelId("test".into());

        store.write_chunk(&model_id, 0, 0, b"data").unwrap();
        assert!(store.shard_path(&model_id, 0).exists());

        store.delete_shard(&model_id, 0).unwrap();
        assert!(!store.shard_path(&model_id, 0).exists());
    }
}
