use std::sync::Arc;

use crate::model::shard::ShardStore;

use super::map_gguf_architecture;
use super::state::{LoadedModelInfo, SharedState};

/// Generate a ModelManifest for a locally loaded GGUF file and register it.
///
/// This solves the "bootstrap deadlock" — without a manifest, peers can't discover
/// or request the model. By generating a manifest from the loaded GGUF at startup,
/// we can broadcast it to the network so other nodes can acquire shards.
pub fn generate_and_register_local_manifest(
    shared_state: &Arc<SharedState>,
    info: &LoadedModelInfo,
    model_path: &std::path::Path,
) {
    // Use a filesystem-safe slug for the model ID.
    // Lowercase, replace spaces/special chars with hyphens, collapse runs.
    let slug = info
        .name
        .to_lowercase()
        .replace(|c: char| !c.is_alphanumeric() && c != '-' && c != '.', "-")
        .split('-')
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join("-");
    let model_id = crate::types::ModelId(slug);

    // Check if we already have a manifest for this model (e.g. persisted from a previous run).
    // Even if the manifest exists, we must still register ourselves as shard holders.
    if let Some(existing) = shared_state.model_registry.get_manifest(&model_id) {
        tracing::debug!(model = %model_id, "Manifest already registered, registering shard holders");
        let node_id = shared_state.identity.node_id().clone();
        let shard_range = shared_state.config.inference.shard_range;
        for shard_info in &existing.shards {
            let in_range = match shard_range {
                Some((start, end)) => shard_info.index >= start && shard_info.index <= end,
                None => true,
            };
            if in_range {
                let shard_id = crate::types::ShardId {
                    model_id: model_id.clone(),
                    index: shard_info.index,
                };
                shared_state
                    .model_registry
                    .record_shard_holder(shard_id, node_id.clone());
            }
        }
        // Also load GGUF metadata if not already cached
        if !shared_state.gguf_meta.contains_key(&model_id) {
            let path = std::path::Path::new(model_path);
            if let Ok(meta) = crate::inference::split::GgufTensorMeta::from_gguf_file(path) {
                shared_state.gguf_meta.insert(model_id.clone(), meta);
            }
        }
        return;
    }

    let path = std::path::Path::new(model_path);
    if !path.exists() {
        tracing::warn!(path = %model_path.display(), "Model file not found, cannot generate manifest");
        return;
    }

    let file_size = std::fs::metadata(path)
        .map(|m| m.len())
        .unwrap_or(info.size_bytes);

    // Split model into shards for torrent-style distribution.
    // Shard size is configurable via [model].shard_size_mb (default 512MB).
    let shard_size: u64 = shared_state.config.model.shard_size_bytes();
    let node_id = shared_state.identity.node_id().clone();

    // Extract model metadata from GGUF header (num_layers, architecture, etc.)
    // and compute layer-aligned shard layouts. The layout count determines shard_count
    // (NOT file_size / shard_size, which can differ from the actual layout count).
    let (num_layers, architecture, shard_count, shards) =
        match crate::inference::split::GgufTensorMeta::from_gguf_file(path) {
            Ok(meta) => {
                let num_layers = meta.block_count as u32;
                // Estimate shard count from file size for layout computation
                let estimated_count = file_size.div_ceil(shard_size).max(1) as u32;
                let layouts =
                    crate::inference::split::compute_layer_shard_layouts(&meta, estimated_count);
                let actual_shard_count = layouts.len() as u32;
                tracing::info!(
                    model = %model_id,
                    num_layers,
                    embedding_length = meta.embedding_length,
                    shard_count = actual_shard_count,
                    "Extracted GGUF metadata for manifest"
                );

                // Build shard infos from layouts (handles hashing, tensor entries, layer ranges)
                let model_dir =
                    crate::model::shard::ShardStore::new(&shared_state.config.node.data_dir)
                        .models_dir()
                        .join(&model_id.0);
                let shards =
                    crate::model::manifest::build_shard_infos_from_layouts(&model_dir, &layouts);

                // Store the metadata for later use in layer range computation
                shared_state.gguf_meta.insert(model_id.clone(), meta);
                // Map GGUF general.architecture string to our ModelArchitecture enum
                let arch = map_gguf_architecture(path);
                (num_layers, arch, actual_shard_count, shards)
            }
            Err(e) => {
                tracing::warn!(error = %e, "Failed to extract GGUF metadata, using defaults");
                let shard_count = file_size.div_ceil(shard_size).max(1) as u32;
                let shards = vec![];
                (
                    0u32,
                    crate::types::ModelArchitecture::Llama,
                    shard_count,
                    shards,
                )
            }
        };

    let mut manifest = crate::types::ModelManifest {
        schema_version: 2,
        id: model_id.clone(),
        name: info.name.clone(),
        architecture,
        num_layers,
        num_params_billions: 0.0,
        quantization: crate::types::Quantization::Q4KM,
        total_size_bytes: file_size,
        shard_count,
        shards,
        tokenizer_hash: [0u8; 32],
        manifest_hash: [0u8; 32],
        publisher: node_id.clone(),
        publish_date: chrono::Utc::now(),
        license: "Unknown".to_string(),
        mmproj: None,
    };
    manifest.manifest_hash = manifest.compute_hash();

    // Store the source GGUF path so the shard server can read byte ranges from it.
    // We write a small metadata file alongside the manifest.
    let shard_store = ShardStore::new(&shared_state.config.node.data_dir);
    let model_dir = shard_store.models_dir().join(&model_id.0);
    let _ = std::fs::create_dir_all(&model_dir);

    // Write a source_path file so the shard server knows where the original GGUF lives
    if let Ok(canonical) = path.canonicalize() {
        let source_path_file = model_dir.join("source_path");
        if let Err(e) = std::fs::write(&source_path_file, canonical.to_string_lossy().as_bytes()) {
            tracing::warn!(error = %e, "Failed to write source_path file");
        }
    }

    // Save GGUF header for shard-only operation.
    // This allows nodes without the full model file to use ShardReader.
    let header_path = model_dir.join("gguf_header.bin");
    if !header_path.exists() {
        if let Err(e) = crate::inference::split::save_gguf_header(path, &header_path) {
            tracing::warn!(error = %e, "Failed to save GGUF header (shard-only mode won't work)");
        }
    }

    // Save manifest to disk
    if let Err(e) = manifest.save_to_dir(&model_dir) {
        tracing::warn!(error = %e, "Failed to save generated manifest");
        return;
    }

    // If shards live in a differently-named directory (e.g. from HF download),
    // also save manifest + header there so shard scanning finds them.
    let shard0_in_model_dir = model_dir.join("shard_000.bin");
    if !shard0_in_model_dir.exists() {
        // Shards might be in a different directory — scan for them
        let models_dir = shard_store.models_dir();
        if let Ok(entries) = std::fs::read_dir(&models_dir) {
            for entry in entries.flatten() {
                let dir = entry.path();
                if dir.is_dir() && dir != model_dir && dir.join("shard_000.bin").exists() {
                    // Found shards in a different directory — save manifest + header there too
                    if !dir.join("manifest.json").exists() {
                        if let Err(e) = manifest.save_to_dir(&dir) {
                            tracing::warn!(error = %e, path = %dir.display(), "Failed to save manifest to shard dir");
                        } else {
                            tracing::info!(
                                model = %model_id,
                                shard_dir = %dir.display(),
                                "Also saved manifest to shard directory"
                            );
                        }
                    }
                    let alt_header = dir.join("gguf_header.bin");
                    if !alt_header.exists() {
                        if let Err(e) = crate::inference::split::save_gguf_header(path, &alt_header)
                        {
                            tracing::warn!(error = %e, "Failed to save GGUF header to shard dir");
                        }
                    }
                }
            }
        }
    }

    // Register in model_registry
    shared_state
        .model_registry
        .register_manifest(manifest.clone());

    // Register ourselves as holder of our shards.
    // If --shards range is set, only claim those indices; otherwise claim all.
    // Only register shards that actually exist on disk.
    let shard_range = shared_state.config.inference.shard_range;
    let shard_store_check = ShardStore::new(&shared_state.config.node.data_dir);
    for shard_info in &manifest.shards {
        let in_range = match shard_range {
            Some((start, end)) => shard_info.index >= start && shard_info.index <= end,
            None => true,
        };
        if !in_range {
            continue;
        }
        // Verify file exists on disk before registering
        let shard_path = shard_store_check.shard_path(&model_id, shard_info.index);
        if !shard_path.exists() {
            tracing::warn!(
                model = %model_id,
                shard = shard_info.index,
                "Shard file missing on disk — skipping registration"
            );
            continue;
        }
        let shard_id = crate::types::ShardId {
            model_id: model_id.clone(),
            index: shard_info.index,
        };
        shared_state
            .model_registry
            .record_shard_holder(shard_id, node_id.clone());
    }
    if let Some((s, e)) = shard_range {
        tracing::info!(
            model = %model_id,
            shard_start = s,
            shard_end = e,
            "Registered as holder of shard range only"
        );
    }

    // Persist to DB
    if let Err(e) = shared_state
        .model_registry
        .persist_manifest(&shared_state.db, &manifest)
    {
        tracing::warn!(error = %e, "Failed to persist manifest to DB");
    }

    tracing::info!(
        model = %model_id,
        size = file_size,
        shards = shard_count,
        "Generated and registered multi-shard manifest for local model"
    );
}

/// Regenerate a manifest from GGUF header metadata and on-disk shard files.
/// Used when manifest.json is missing but gguf_header.bin + shards exist.
pub(super) fn regenerate_manifest_from_header(
    model_id: &crate::types::ModelId,
    model_dir: &std::path::Path,
    meta: &crate::inference::split::GgufTensorMeta,
    config: &crate::config::Config,
) -> Option<crate::types::ModelManifest> {
    let shard_size = config.model.shard_size_bytes();

    // Compute total GGUF file size from tensor metadata (header + all tensor data).
    // This is the REAL total, even when we only have a subset of shards locally.
    let total_size = {
        let max_end = meta
            .tensors
            .values()
            .map(|loc| meta.tensor_data_offset + loc.offset + loc.size)
            .max()
            .unwrap_or(meta.tensor_data_offset);
        // Round up to alignment (GGUF tensors are 32-byte aligned)
        (max_end + 31) & !31
    };

    // Check if this is actually a single full GGUF file stored as shard_000.bin
    // (not a real byte-range shard). If shard_000 exists and its size >= total_size,
    // treat as a 1-shard model.
    let shard0_path = model_dir.join("shard_000.bin");
    let is_single_full_gguf = shard0_path.exists()
        && !model_dir.join("shard_001.bin").exists()
        && shard0_path
            .metadata()
            .map(|m| m.len() >= total_size.saturating_sub(4096))
            .unwrap_or(false);

    let (shard_count, shards) = if is_single_full_gguf {
        // Single full GGUF — 1 shard containing all layers
        let file_size = shard0_path.metadata().map(|m| m.len()).unwrap_or(0);
        let hash: crate::types::Blake3Hash = {
            let data = std::fs::read(&shard0_path).unwrap_or_default();
            blake3::hash(&data).into()
        };
        let shard_info = crate::types::ShardInfo {
            index: 0,
            layer_range: (0, meta.block_count as u32),
            size_bytes: file_size,
            hash,
            tensors: Vec::new(),
        };
        (1u32, vec![shard_info])
    } else {
        let estimated_count = total_size.div_ceil(shard_size).max(1) as u32;
        let layouts = crate::inference::split::compute_layer_shard_layouts(meta, estimated_count);
        let sc = layouts.len() as u32;
        let sh = crate::model::manifest::build_shard_infos_from_layouts(model_dir, &layouts);
        (sc, sh)
    };

    let model_name = meta
        .model_name
        .clone()
        .unwrap_or_else(|| model_id.0.clone());

    // Map GGUF architecture string to our ModelArchitecture enum
    let architecture = match meta.architecture.as_str() {
        "qwen2" | "qwen3" | "qwen2moe" => crate::types::ModelArchitecture::Qwen2,
        "qwen35" => crate::types::ModelArchitecture::Qwen35,
        "qwen35moe" | "qwen3_5moe" => crate::types::ModelArchitecture::Qwen35Moe {
            num_experts: 0,
            experts_per_token: 0,
        },
        "mistral" => crate::types::ModelArchitecture::Mistral,
        "phi" | "phi3" => crate::types::ModelArchitecture::Phi,
        _ => crate::types::ModelArchitecture::Llama,
    };

    let mut manifest = crate::types::ModelManifest {
        schema_version: 2,
        id: model_id.clone(),
        name: model_name,
        architecture,
        num_layers: meta.block_count as u32,
        num_params_billions: 0.0,
        quantization: crate::types::Quantization::Q4KM,
        total_size_bytes: total_size,
        shard_count,
        shards,
        tokenizer_hash: [0u8; 32],
        manifest_hash: [0u8; 32],
        publisher: crate::types::NodeId([0u8; 32]),
        publish_date: chrono::Utc::now(),
        license: "Unknown".to_string(),
        mmproj: None,
    };
    manifest.manifest_hash = manifest.compute_hash();

    // Save to disk
    if let Err(e) = manifest.save_to_dir(model_dir) {
        tracing::warn!(model = %model_id, error = %e, "Failed to save regenerated manifest");
    } else {
        tracing::info!(
            model = %model_id,
            shard_count,
            num_layers = meta.block_count,
            "Regenerated and saved manifest with accurate layer ranges"
        );
    }

    Some(manifest)
}

/// Extract `tied_output_weight.bin` from shard_000.bin for weight-tied models.
///
/// Weight-tied models (like Gemma-2) reuse `token_embd.weight` as the output head.
/// In distributed inference, a node may have the last shard but not shard_000.
/// This function extracts the raw tensor bytes from shard_000 so any node can load it.
pub(super) fn extract_tied_output_weight(
    shard0_path: &std::path::Path,
    model_dir: &std::path::Path,
    meta: &crate::inference::split::GgufTensorMeta,
) -> Result<(), String> {
    let embd_loc = meta
        .tensors
        .get("token_embd.weight")
        .ok_or("token_embd.weight not found in tensor metadata")?;

    // token_embd.weight is in shard_000 — its offset in the GGUF is tensor_data_offset + embd_loc.offset.
    // In shard_000.bin, the header is preserved so the absolute offset is the same.
    let abs_offset = meta.tensor_data_offset + embd_loc.offset;
    let size = embd_loc.size;

    let shard_data =
        std::fs::read(shard0_path).map_err(|e| format!("Failed to read shard_000.bin: {e}"))?;

    let end = (abs_offset + size) as usize;
    if end > shard_data.len() {
        return Err(format!(
            "token_embd.weight extends beyond shard_000.bin (need {end} bytes, have {})",
            shard_data.len()
        ));
    }

    let tensor_bytes = &shard_data[abs_offset as usize..end];
    let dest_path = model_dir.join("tied_output_weight.bin");
    std::fs::write(&dest_path, tensor_bytes)
        .map_err(|e| format!("Failed to write tied_output_weight.bin: {e}"))?;

    tracing::info!(
        size = tensor_bytes.len(),
        path = %dest_path.display(),
        "Extracted tied_output_weight.bin from shard_000.bin"
    );
    Ok(())
}
