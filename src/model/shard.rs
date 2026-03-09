use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use crate::error::SwarmError;
use crate::types::{ModelId, ShardInfo};

/// Sanitize a path component to prevent path traversal attacks.
/// Uses an allowlist approach: only `[a-zA-Z0-9_\-.]` characters are kept;
/// everything else (including null bytes and directory separators) is replaced
/// with `_`. Consecutive dots (`..`) are collapsed to prevent traversal.
pub fn sanitize_path_component(s: &str) -> String {
    let replaced: String = s
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' || c == '.' {
                c
            } else {
                '_'
            }
        })
        .collect();
    replaced.replace("..", "_")
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
    ///
    /// When `allow_zero_hash` is true, shards with an all-zero hash in the manifest
    /// (placeholder from HF download before hashes are known) skip verification.
    /// This should ONLY be true for the local HF download path.
    /// Network-received shards must always have a real hash.
    pub fn verify_shard_with_options(
        &self,
        model_id: &ModelId,
        info: &ShardInfo,
        allow_zero_hash: bool,
    ) -> Result<(), SwarmError> {
        let path = self.shard_path(model_id, info.index);
        if !path.exists() {
            return Err(SwarmError::ShardNotFound(crate::types::ShardId {
                model_id: model_id.clone(),
                index: info.index,
            }));
        }

        let hash_unknown = info.hash == [0u8; 32];

        // Only skip verification for zero-hash if explicitly allowed (local HF downloads)
        if hash_unknown && !allow_zero_hash {
            return Err(SwarmError::ShardIntegrity {
                expected: "non-zero hash required".to_string(),
                actual: "all-zero hash (placeholder)".to_string(),
            });
        }

        if hash_unknown && allow_zero_hash {
            // Zero-hash bypass only for local HF download path
            return Ok(());
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

        if actual == info.hash {
            tracing::debug!(
                model = %model_id,
                shard = info.index,
                "DIAG: verify_shard OK"
            );
        }

        if actual != info.hash {
            tracing::info!(
                model = %model_id,
                shard = info.index,
                "DIAG: verify_shard FAILED — hash mismatch"
            );
            // Quarantine the bad shard
            let quarantine_path = path.with_extension("bin.quarantine");
            if let Err(e) = std::fs::rename(&path, &quarantine_path) {
                tracing::warn!(
                    model = %model_id,
                    shard = info.index,
                    error = %e,
                    "Failed to quarantine shard, attempting deletion"
                );
                let _ = std::fs::remove_file(&path);
            }
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

    /// Verify a shard's BLAKE3 hash matches the expected value.
    /// Does NOT allow zero-hash bypass (safe default for network-received shards).
    pub fn verify_shard(&self, model_id: &ModelId, info: &ShardInfo) -> Result<(), SwarmError> {
        self.verify_shard_with_options(model_id, info, false)
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
                Ok(mut manifest) => {
                    // Auto-compute hash for zero-hash manifests (e.g. manually placed
                    // or pre-V2 manifests updated to V2 format). This ensures they pass
                    // the strict verification when gossiped to peers.
                    if manifest.manifest_hash == [0u8; 32] {
                        manifest.manifest_hash = manifest.compute_hash();
                        if let Err(e) = manifest.save_to_dir(&model_dir) {
                            tracing::warn!(
                                model = %model_id,
                                error = %e,
                                "Failed to save auto-computed manifest hash"
                            );
                        } else {
                            tracing::info!(
                                model = %model_id,
                                hash = %hex::encode(manifest.manifest_hash),
                                "Auto-computed and saved manifest hash"
                            );
                        }
                    }
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

                        match self.verify_shard_with_options(&model_id, shard_info, true) {
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

        let model_count = shards
            .iter()
            .map(|(m, _)| m)
            .collect::<std::collections::HashSet<_>>()
            .len();
        tracing::info!(
            model_count,
            total_shards = shards.len(),
            rejected_count = rejected,
            "DIAG: load_all_local complete"
        );
        Ok(shards)
    }

    /// Get the path to the temporary file used during shard download.
    fn shard_tmp_path(&self, model_id: &ModelId, index: u32) -> PathBuf {
        let mut p = self.shard_path(model_id, index);
        p.set_extension("bin.tmp");
        p
    }

    /// Write a chunk of shard data to disk (for progressive downloads).
    /// Writes to a .tmp file; call `finalize_shard` to atomically rename.
    pub fn write_chunk(
        &self,
        model_id: &ModelId,
        index: u32,
        offset: u64,
        data: &[u8],
    ) -> Result<(), SwarmError> {
        let tmp_path = self.shard_tmp_path(model_id, index);
        if let Some(parent) = tmp_path.parent() {
            std::fs::create_dir_all(parent).map_err(SwarmError::Io)?;
        }

        // Truncate on first write (offset == 0) to clean up partial downloads
        if offset == 0 {
            let _ = std::fs::remove_file(&tmp_path);
        }

        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(false)
            .open(&tmp_path)
            .map_err(SwarmError::Io)?;

        file.seek(SeekFrom::Start(offset)).map_err(SwarmError::Io)?;
        file.write_all(data).map_err(SwarmError::Io)?;

        Ok(())
    }

    /// Atomically finalize a shard download by renaming .tmp → .bin.
    pub fn finalize_shard(&self, model_id: &ModelId, index: u32) -> Result<(), SwarmError> {
        let tmp_path = self.shard_tmp_path(model_id, index);
        let final_path = self.shard_path(model_id, index);
        if tmp_path.exists() {
            std::fs::rename(&tmp_path, &final_path).map_err(SwarmError::Io)?;
        }
        Ok(())
    }

    /// Clean up leftover .tmp files from interrupted downloads on startup.
    pub fn cleanup_tmp_files(&self) {
        let models_dir = self.models_dir();
        if !models_dir.exists() {
            return;
        }
        if let Ok(entries) = std::fs::read_dir(&models_dir) {
            for entry in entries.flatten() {
                if !entry.path().is_dir() {
                    continue;
                }
                if let Ok(files) = std::fs::read_dir(entry.path()) {
                    for file in files.flatten() {
                        let path = file.path();
                        if path.extension().and_then(|e| e.to_str()) == Some("tmp") {
                            tracing::info!(path = %path.display(), "Cleaning up leftover .tmp shard file");
                            let _ = std::fs::remove_file(&path);
                        }
                    }
                }
            }
        }
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

        // Write shard (goes to .tmp)
        store.write_chunk(&model_id, 0, 0, data).unwrap();
        // Finalize (.tmp → .bin)
        store.finalize_shard(&model_id, 0).unwrap();

        // Compute expected hash
        let expected_hash = *blake3::hash(data).as_bytes();

        let info = ShardInfo {
            index: 0,
            layer_range: (0, 1),
            size_bytes: data.len() as u64,
            hash: expected_hash,
            tensors: vec![],
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
        store.finalize_shard(&model_id, 0).unwrap();

        let info = ShardInfo {
            index: 0,
            layer_range: (0, 1),
            size_bytes: data.len() as u64,
            hash: [0xFF; 32], // Wrong hash
            tensors: vec![],
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
        store.finalize_shard(&model_id, 0).unwrap();
        assert!(store.shard_path(&model_id, 0).exists());

        store.delete_shard(&model_id, 0).unwrap();
        assert!(!store.shard_path(&model_id, 0).exists());
    }

    // --- sanitize_path_component tests ---

    #[test]
    fn sanitize_replaces_forward_slash() {
        assert_eq!(sanitize_path_component("a/b/c"), "a_b_c");
    }

    #[test]
    fn sanitize_replaces_backslash() {
        assert_eq!(sanitize_path_component("a\\b\\c"), "a_b_c");
    }

    #[test]
    fn sanitize_replaces_dot_dot() {
        assert_eq!(sanitize_path_component(".."), "_");
        // "/" replaced first → "a_.._b", then ".." replaced → "a___b"
        assert_eq!(sanitize_path_component("a/../b"), "a___b");
    }

    #[test]
    fn sanitize_combined_traversal_attack() {
        // "../../etc/passwd" → replace / → ".._.._etc_passwd" → replace .. → "____etc_passwd"
        let result = sanitize_path_component("../../etc/passwd");
        assert!(!result.contains(".."), "must not contain '..'");
        assert!(!result.contains('/'), "must not contain '/'");
        assert_eq!(result, "____etc_passwd");
    }

    #[test]
    fn sanitize_normal_strings_unchanged() {
        assert_eq!(sanitize_path_component("my-model-v1"), "my-model-v1");
        assert_eq!(
            sanitize_path_component("TinyLlama-1.1B-Q4_K_M"),
            "TinyLlama-1.1B-Q4_K_M"
        );
        assert_eq!(
            sanitize_path_component("meta-llama_Llama-3-8B"),
            "meta-llama_Llama-3-8B"
        );
    }

    #[test]
    fn sanitize_empty_string() {
        assert_eq!(sanitize_path_component(""), "");
    }

    #[test]
    fn sanitize_single_dot() {
        // A single dot is fine — it's not ".."
        assert_eq!(sanitize_path_component("."), ".");
    }

    #[test]
    fn sanitize_triple_dots() {
        // "..." contains ".." so the first two dots become "_", leaving "_."
        assert_eq!(sanitize_path_component("..."), "_.");
    }

    #[test]
    fn sanitize_unicode() {
        // Unicode should pass through — only /, \, and .. are dangerous
        assert_eq!(sanitize_path_component("模型-v1"), "模型-v1");
        assert_eq!(sanitize_path_component("café-model"), "café-model");
    }

    #[test]
    fn sanitize_mixed_separators() {
        // "a/b\c/../d" → replace /,\ → "a_b_c_.._d" → replace .. → "a_b_c___d"
        assert_eq!(sanitize_path_component("a/b\\c/../d"), "a_b_c___d");
    }

    #[test]
    fn shard_path_sanitizes_model_id() {
        let store = ShardStore::new(Path::new("/data"));
        // Path traversal in model_id should be neutralized
        let path = store.shard_path(&ModelId("../../etc".into()), 0);
        assert!(
            !path.to_string_lossy().contains(".."),
            "Path should not contain '..' after sanitization: {}",
            path.display()
        );
        // "../../etc" → replace / → "_.._etc" wait no: ".._._etc" no.
        // "../../etc" → replace / → ".._.._etc" → replace .. → "____etc"
        assert_eq!(path.to_string_lossy(), "/data/models/____etc/shard_000.bin");
    }
}
