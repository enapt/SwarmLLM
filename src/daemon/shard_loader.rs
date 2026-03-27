use crate::error::SwarmError;
use crate::model::shard::ShardStore;

/// Handle an incoming LayerForward from a remote peer: run the local split model
/// segment and send back a LayerResult with either logits (last segment) or
/// hidden-state activations (intermediate segment).
/// Parameters for shard-based model loading.
pub struct ShardLoadParams<'a> {
    pub model_dir: &'a std::path::Path,
    pub shard_store: &'a ShardStore,
    pub model_id: &'a crate::types::ModelId,
    pub layer_start: usize,
    pub layer_end: usize,
    pub is_first: bool,
    pub is_last: bool,
    /// Manifest for this model — provides tensor entries and total size.
    pub manifest: &'a crate::types::ModelManifest,
}

/// Try to load a SplitModel from shard files + gguf_header.bin.
/// This is the shard-only loading path — no full GGUF needed.
pub fn try_load_from_shards(
    params: &ShardLoadParams<'_>,
) -> Result<crate::inference::split::SplitModel, SwarmError> {
    let model_dir = params.model_dir;
    let shard_store = params.shard_store;
    let model_id = params.model_id;
    let layer_start = params.layer_start;
    let layer_end = params.layer_end;
    let is_first = params.is_first;
    let is_last = params.is_last;

    // Reject legacy v1 manifests — they lack tensor entries, so ShardReader
    // would silently produce an empty tensor_map and fail at read time.
    if params.manifest.schema_version < 2 {
        return Err(SwarmError::ModelNotAvailable(crate::types::ModelId(
            format!(
                "{} (schema_version {} — v2 required, re-download shards)",
                model_id, params.manifest.schema_version
            ),
        )));
    }

    // Ensure GGUF header exists (extract from shard_000 if needed)
    if let Err(e) = crate::inference::split::ensure_gguf_header(model_dir) {
        return Err(SwarmError::ModelNotAvailable(crate::types::ModelId(
            format!("Cannot load from shards: {e}"),
        )));
    }

    // Collect available shard files for this model
    // Scan all possible shard indices — don't stop on gaps (sparse sets are valid)
    let mut shard_files: Vec<(u32, std::path::PathBuf)> = Vec::new();
    let scan_limit = params.manifest.shard_count.max(1);
    for i in 0u32..scan_limit {
        let path = shard_store.shard_path(model_id, i);
        if path.exists() {
            shard_files.push((i, path));
        }
    }

    if shard_files.is_empty() {
        return Err(SwarmError::Internal(format!(
            "No shard files found for model {} in {}",
            model_id,
            model_dir.display()
        )));
    }

    // Build tensor entries for each shard file from manifest data.
    // The order must match shard_files (which is sorted by shard index).
    let tensor_entries: Vec<Vec<crate::types::ShardTensorEntry>> = shard_files
        .iter()
        .map(|(idx, _)| {
            params
                .manifest
                .shards
                .iter()
                .find(|s| s.index == *idx)
                .map(|s| s.tensors.clone())
                .unwrap_or_default()
        })
        .collect();

    tracing::info!(
        model = %model_id,
        shards = shard_files.len(),
        layers = format!("[{layer_start}..{layer_end})"),
        "Loading split model from shard files (no full GGUF)"
    );

    let result = crate::inference::split::SplitModel::load_from_shards(
        model_dir,
        shard_files.clone(),
        &tensor_entries,
        params.manifest.total_size_bytes,
        layer_start,
        layer_end,
        is_first,
        is_last,
    );

    // GPU OOM fallback: retry on CPU so the model is still usable (slower but functional)
    match &result {
        Err(e) if e.to_string().contains("OUT_OF_MEMORY") => {
            tracing::warn!(
                model = %model_id,
                "GPU OOM — retrying model load on CPU"
            );
            crate::inference::split::SplitModel::load_from_shards_cpu(
                model_dir,
                shard_files,
                &tensor_entries,
                params.manifest.total_size_bytes,
                layer_start,
                layer_end,
                is_first,
                is_last,
            )
        }
        _ => result,
    }
}
