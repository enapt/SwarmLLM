use axum::extract::{Path, State};
use axum::Json;
use serde::Deserialize;

use crate::api::server::AppState;
use crate::error::ApiError;
use crate::model::manifest::ModelManifestExt;

/// Heuristic shard sizing constants for HF search scoring.
const EST_SHARD_SIZE_BYTES: u64 = 800 * 1024 * 1024;
const EST_SHARD_COUNT_MIN: u64 = 2;
const EST_SHARD_COUNT_MAX: u64 = 16;
const BOOMERANG_SIZE_NUMERATOR: u64 = 12;
const BOOMERANG_SIZE_DENOMINATOR: u64 = 5;
const MODEL_SIZE_SCORE_MAX_GB: f64 = 8.0;

/// Count unique peers holding shards of the given model IDs.
fn count_unique_shard_holders(
    registry: &crate::model::registry::ModelRegistry,
    model_ids: &[crate::types::ModelId],
) -> usize {
    let mut unique = std::collections::HashSet::new();
    for (shard_id, holders) in registry.all_shard_entries() {
        if model_ids.contains(&shard_id.model_id) {
            for h in &holders {
                unique.insert(h.clone());
            }
        }
    }
    unique.len()
}

/// Slow-download detection threshold: 100 KB/s.
const SLOW_DOWNLOAD_SPEED_THRESHOLD: u64 = 102400;
/// Duration in seconds before emitting a slow-download warning.
const SLOW_DOWNLOAD_WARN_SECS: f64 = 30.0;

/// Spawn a background task that reads download progress events and updates acquisition_progress.
fn spawn_progress_updater(
    shared: std::sync::Arc<crate::daemon::state::SharedState>,
    mid: crate::types::ModelId,
    mut prx: tokio::sync::mpsc::Receiver<crate::model::huggingface::DownloadProgress>,
) {
    let mut shutdown_rx = shared.shutdown_rx();
    tokio::spawn(async move {
        let mut last_bytes = 0u64;
        let mut last_time = std::time::Instant::now();
        let mut slow_since: Option<std::time::Instant> = None;
        let mut throttle_warned = false;
        loop {
            tokio::select! {
                prog = prx.recv() => {
                    let Some(prog) = prog else { break };
                    if let Some(mut entry) = shared.models.acquisition_progress.get_mut(&mid) {
                        entry.downloaded_bytes = prog.downloaded_bytes;
                        entry.total_bytes = prog.total_bytes;
                        let now = std::time::Instant::now();
                        let dt = now.duration_since(last_time).as_secs_f64();
                        if dt > 0.5 {
                            let speed =
                                (prog.downloaded_bytes.saturating_sub(last_bytes) as f64 / dt) as u64;
                            entry.speed_bytes_per_sec = speed;
                            last_bytes = prog.downloaded_bytes;
                            last_time = now;

                            // Slow-download detection: warn once after sustained slow speed
                            if speed > 0 && speed < SLOW_DOWNLOAD_SPEED_THRESHOLD {
                                let since = *slow_since.get_or_insert(now);
                                if !throttle_warned && now.duration_since(since).as_secs_f64() > SLOW_DOWNLOAD_WARN_SECS {
                                    throttle_warned = true;
                                    let speed_str = format!("{:.1} KB/s", speed as f64 / 1024.0);
                                    shared.emit_activity(
                                        crate::daemon::state::ActivityEvent::new(
                                            "model", "download_slow",
                                            format!("Download is slow ({speed_str}) — this can happen with popular models. It will keep going."),
                                        )
                                        .with_model(mid.0.clone())
                                        .with_detail_str(speed_str)
                                        .with_toast("warning", 10000),
                                    );
                                }
                            } else {
                                slow_since = None;
                            }
                        }
                    }
                }
                _ = shutdown_rx.changed() => break,
            }
        }
    });
}

/// SEC: Validate HuggingFace repo_id format — delegates to the canonical validator.
fn is_valid_hf_repo_id(repo_id: &str) -> bool {
    crate::model::huggingface::validate_hf_repo_id(repo_id).is_ok()
}

/// SEC: Validate HuggingFace filename format.
/// Only allows alphanumeric, hyphens, dots, underscores. Must end with .gguf.
fn is_valid_hf_filename(filename: &str) -> bool {
    !filename.is_empty()
        && filename.len() <= 256
        && filename.ends_with(".gguf")
        && filename
            .chars()
            .all(|c| c.is_alphanumeric() || c == '-' || c == '_' || c == '.')
        && !filename.contains("..")
}

/// Convert a GGUF filename to a model ID slug.
/// Strips .gguf suffix, lowercases, replaces non-alphanumeric chars with hyphens,
/// and collapses consecutive hyphens.
fn gguf_filename_to_model_id(filename: &str) -> String {
    filename
        .trim_end_matches(".gguf")
        .to_lowercase()
        .replace(|c: char| !c.is_alphanumeric() && c != '-' && c != '.', "-")
        .split('-')
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join("-")
}

/// Validate HF repo_id and filename inputs, returning ApiError on failure.
fn validate_hf_inputs(repo_id: &str, filename: &str) -> Result<(), ApiError> {
    if repo_id.is_empty() || filename.is_empty() {
        return Err(ApiError(crate::error::SwarmError::Validation(
            "repo_id and filename are required".into(),
        )));
    }
    if !is_valid_hf_repo_id(repo_id) {
        return Err(ApiError(crate::error::SwarmError::Validation(
            "Invalid repo_id format. Expected: owner/repo (alphanumeric, hyphens, dots, underscores)"
                .into(),
        )));
    }
    if !is_valid_hf_filename(filename) {
        return Err(ApiError(crate::error::SwarmError::Validation(
            "Invalid filename. Must be alphanumeric with hyphens, dots, underscores, ending in .gguf"
                .into(),
        )));
    }
    Ok(())
}

/// Extract EOS token IDs from a GGUF file, with architecture-specific fallbacks.
fn extract_eos_token_ids(path: &std::path::Path, arch: &str) -> Vec<u32> {
    match crate::inference::split::GgufTokenizerMeta::from_gguf_file(path) {
        Ok(tok) => tok.eos_tokens_with_arch_fallback(arch),
        Err(_) => vec![2],
    }
}
// ---- HuggingFace Endpoints ----

/// GET /api/admin/hf/search?q=... — Search HuggingFace for GGUF models.
pub async fn hf_search(
    State(state): State<AppState>,
    axum::extract::Query(params): axum::extract::Query<HfSearchParams>,
) -> Result<Json<Vec<serde_json::Value>>, ApiError> {
    let query = params.query.unwrap_or_default();
    if query.is_empty() {
        return Ok(Json(vec![]));
    }
    if query.len() > 256 {
        return Err(ApiError(crate::error::SwarmError::Validation(
            "Search query too long (max 256 chars)".into(),
        )));
    }

    let results = crate::model::huggingface::search_gguf_models(&query)
        .await
        .map_err(|e| ApiError(crate::error::SwarmError::ServiceUnavailable(e)))?;

    // Available VRAM for fits_vram check (pool VRAM or local GPU)
    let available_vram_bytes: u64 = state
        .shared_state
        .gpu_info
        .as_ref()
        .map(|g| g.vram_free_mb * 1024 * 1024)
        .unwrap_or(0);

    // Group results by repo_id with quant variants (preserve HF API order = by downloads)
    let mut repo_order: Vec<String> = Vec::new();
    let mut repo_map: std::collections::HashMap<
        String,
        Vec<crate::model::huggingface::HfModelResult>,
    > = std::collections::HashMap::new();
    for r in results {
        if !repo_map.contains_key(&r.repo_id) {
            repo_order.push(r.repo_id.clone());
        }
        repo_map.entry(r.repo_id.clone()).or_default().push(r);
    }

    let mut values: Vec<serde_json::Value> = repo_order
        .into_iter()
        .filter_map(|repo_id| {
            let files = repo_map.remove(&repo_id)?;
            Some((repo_id, files))
        })
        .map(|(repo_id, files)| {
            let downloads = files.first().map(|f| f.downloads).unwrap_or(0);
            let likes = files.first().map(|f| f.likes).unwrap_or(0);

            let variants: Vec<serde_json::Value> = files
                .iter()
                .map(|f| {
                    let quant = crate::model::huggingface::extract_quant_tag(&f.filename)
                        .unwrap_or_else(|| "unknown".into());
                    serde_json::json!({
                        "filename": f.filename,
                        "size_bytes": f.size_bytes,
                        "quant": quant,
                    })
                })
                .collect();

            // Recommended variant: prefer Q4_K_M, else smallest Q4+, else first
            let recommended = files
                .iter()
                .find(|f| {
                    crate::model::huggingface::extract_quant_tag(&f.filename)
                        .is_some_and(|q| q == "Q4_K_M")
                })
                .or_else(|| {
                    files
                        .iter()
                        .filter(|f| {
                            crate::model::huggingface::extract_quant_tag(&f.filename)
                                .is_some_and(|q| q.starts_with("Q4"))
                        })
                        .min_by_key(|f| f.size_bytes)
                })
                .or(files.first());

            let recommended_variant = recommended
                .and_then(|f| crate::model::huggingface::extract_quant_tag(&f.filename))
                .unwrap_or_else(|| "unknown".into());

            // VRAM fit levels: full model, boomerang (first+last shard), single shard
            let rec_size = recommended
                .map(|f| f.size_bytes)
                .unwrap_or(files.iter().map(|f| f.size_bytes).min().unwrap_or(u64::MAX));
            let est_shards =
                (rec_size / EST_SHARD_SIZE_BYTES).clamp(EST_SHARD_COUNT_MIN, EST_SHARD_COUNT_MAX);
            let est_shard_size = rec_size / est_shards;
            // Boomerang: first + last shard (~2.4x one shard due to embedding/output weights)
            let est_boomerang_size =
                est_shard_size * BOOMERANG_SIZE_NUMERATOR / BOOMERANG_SIZE_DENOMINATOR;

            let fits_full = available_vram_bytes > 0 && rec_size < available_vram_bytes;
            let fits_boomerang =
                available_vram_bytes > 0 && est_boomerang_size < available_vram_bytes;
            let fits_shard = available_vram_bytes > 0 && est_shard_size < available_vram_bytes;
            // True if any participation mode fits
            let fits_vram = fits_full || fits_boomerang || fits_shard;

            // Network replication: count unique peers holding shards of any variant of this repo
            let variant_ids: Vec<crate::types::ModelId> = files
                .iter()
                .map(|f| crate::types::ModelId(gguf_filename_to_model_id(&f.filename)))
                .collect();
            let network_replicas =
                count_unique_shard_holders(&state.shared_state.model_registry, &variant_ids);

            // Composite score: surfaces small, popular, scarce, VRAM-fitting models
            let quality = (downloads as f64 + 10.0).log10() / 7.0; // 0-1 popularity proxy
            let fit = if fits_boomerang {
                1.0
            } else if fits_shard {
                0.6
            } else {
                0.1
            };
            let demand = if network_replicas == 0 {
                1.5
            } else if network_replicas < 3 {
                1.2
            } else if network_replicas < 10 {
                1.0
            } else {
                0.7
            };
            let shard_gb = rec_size as f64 / (1024.0 * 1024.0 * 1024.0);
            let size_factor = (1.0 - shard_gb / MODEL_SIZE_SCORE_MAX_GB).clamp(0.1, 1.0);
            let composite_score = (quality * fit * demand * size_factor * 100.0) as u32;

            serde_json::json!({
                "repo_id": repo_id,
                "downloads": downloads,
                "likes": likes,
                "variants": variants,
                "recommended_variant": recommended_variant,
                "fits_vram": fits_vram,
                "fits_boomerang": fits_boomerang,
                "fits_shard": fits_shard,
                "est_shard_size": est_shard_size,
                "est_boomerang_size": est_boomerang_size,
                "network_replicas": network_replicas,
                "composite_score": composite_score,
                "score_breakdown": {
                    "quality": (quality * 100.0) as u32,
                    "fit": (fit * 100.0) as u32,
                    "demand": (demand * 100.0) as u32,
                    "size": (size_factor * 100.0) as u32,
                },
            })
        })
        .collect();

    // Sort by composite score descending (best-fit models first)
    values.sort_by(|a, b| {
        let sa = a
            .get("composite_score")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        let sb = b
            .get("composite_score")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        sb.cmp(&sa)
    });

    Ok(Json(values))
}

/// POST /api/admin/hf/download — Start downloading a GGUF model from HuggingFace.
pub async fn hf_download(
    State(state): State<AppState>,
    Json(body): Json<HfDownloadRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let repo_id = body.repo_id;
    let filename = body.filename;

    validate_hf_inputs(&repo_id, &filename)?;

    let dest_dir = crate::model::shard::model_dir(&state.config.node.data_dir, &repo_id);

    tracing::info!(repo = %repo_id, file = %filename, "Starting HuggingFace download");

    // Spawn download in background
    let repo_id = repo_id.clone();
    let filename = filename.clone();
    let shared = state.shared_state.clone();
    let model_id_str = format!("hf:{}/{}", repo_id, filename);
    let mid = crate::types::ModelId(model_id_str.clone());

    // Create initial acquisition progress entry
    let status = crate::model::acquisition::AcquisitionStatus::new_downloading(
        mid.clone(),
        1,
        0,
        "huggingface",
        "user",
        format!("Downloading {} from HuggingFace...", filename),
    );
    shared
        .models
        .acquisition_progress
        .insert(mid.clone(), status);

    // Register cancellation flag for this download
    let hf_cancel_flag = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    shared
        .models
        .download_cancel_flags
        .insert(mid.clone(), hf_cancel_flag.clone());

    tokio::spawn(async move {
        let mut shutdown_rx = shared.shutdown_rx();
        let (ptx, prx) =
            tokio::sync::mpsc::channel::<crate::model::huggingface::DownloadProgress>(64);

        let download_mid = mid.clone();
        let download_shared = shared.clone();

        spawn_progress_updater(shared.clone(), mid.clone(), prx);

        let download_result = tokio::select! {
            result = crate::model::huggingface::download_model(
                &repo_id,
                &filename,
                &dest_dir,
                Some(ptx),
            ) => Some(result),
            _ = shutdown_rx.wait_for(|v| *v) => {
                tracing::info!(model = %download_mid, "Download cancelled by shutdown");
                if let Some(mut entry) = download_shared.models.acquisition_progress.get_mut(&download_mid) {
                    entry.state = crate::model::acquisition::AcquisitionState::Failed {
                        reason: "Cancelled by daemon shutdown".into(),
                    };
                    entry.log_push("Cancelled by daemon shutdown".into());
                }
                None
            }
        };
        let Some(download_result) = download_result else {
            shared.models.download_cancel_flags.remove(&download_mid);
            return;
        };
        match download_result {
            Ok(path) => {
                tracing::info!(path = %path.display(), "HuggingFace download complete");
                if let Some(mut entry) = download_shared
                    .models
                    .acquisition_progress
                    .get_mut(&download_mid)
                {
                    entry.state = crate::model::acquisition::AcquisitionState::Complete;
                    entry.downloaded_shards = 1;
                    entry.verified_shards = 1;
                    entry
                        .log
                        .push(format!("Download complete: {}", path.display()));
                }

                // Try to load the downloaded model
                let executor = download_shared.executor.clone();
                let gpu_layers = download_shared.config.inference.gpu_layers;
                let model_name = format!("{}/{}", repo_id, filename);

                let mut exec = executor.lock().await;
                match exec.load_model(&path, gpu_layers) {
                    Ok(()) => {
                        let size = exec.model_size_bytes().unwrap_or(0);
                        let gguf_meta = crate::inference::executor::extract_gguf_metadata(&path);
                        let arch = gguf_meta
                            .as_ref()
                            .map(|m| m.architecture.as_str())
                            .unwrap_or("llama");
                        let eos_tokens = extract_eos_token_ids(&path, arch);
                        *download_shared.loaded_model_info.write().await =
                            Some(crate::daemon::LoadedModelInfo {
                                name: model_name.clone(),
                                size_bytes: size,
                                eos_tokens,
                                chat_template: gguf_meta
                                    .as_ref()
                                    .and_then(|m| m.chat_template.clone()),
                                bos_token: gguf_meta
                                    .as_ref()
                                    .map(|m| m.bos_token.clone())
                                    .unwrap_or_default(),
                                eos_token: gguf_meta
                                    .as_ref()
                                    .map(|m| m.eos_token.clone())
                                    .unwrap_or_default(),
                            });
                        download_shared
                            .model_loaded
                            .store(true, std::sync::atomic::Ordering::Release);
                        if let Some(mut entry) = download_shared
                            .models
                            .acquisition_progress
                            .get_mut(&download_mid)
                        {
                            entry.log_push(format!("Model loaded: {}", model_name));
                        }
                        tracing::info!(model = %model_name, "HF model loaded for inference");
                    }
                    Err(e) => {
                        if let Some(mut entry) = download_shared
                            .models
                            .acquisition_progress
                            .get_mut(&download_mid)
                        {
                            entry.log_push(format!("Model load failed: {}", e));
                        }
                        tracing::error!(error = %e, "Failed to load HF model");
                    }
                }
            }
            Err(e) => {
                tracing::error!(error = %e, "HuggingFace download failed");
                if let Some(mut entry) = download_shared
                    .models
                    .acquisition_progress
                    .get_mut(&download_mid)
                {
                    entry.state =
                        crate::model::acquisition::AcquisitionState::Failed { reason: e.clone() };
                    entry.failed_shards = 1;
                    entry.log_push(format!("Download failed: {}", e));
                }
                download_shared.emit_activity(
                    crate::daemon::state::ActivityEvent::new(
                        "download",
                        "hf_download_failed",
                        format!("Download failed: {}", e),
                    )
                    .with_model(download_mid.0.clone())
                    .with_detail_str(e)
                    .with_toast("error", 8000),
                );
            }
        }

        // Clean up cancel flag
        download_shared
            .models
            .download_cancel_flags
            .remove(&download_mid);

        // Clean up acquisition_progress after a delay so the frontend sees
        // the final state and triggers a re-render before we remove it.
        download_shared.schedule_acquisition_cleanup(download_mid.clone());
    });

    Ok(Json(serde_json::json!({
        "status": "started",
        "model_id": model_id_str,
    })))
}

/// POST /api/admin/shutdown — Gracefully shut down the node.
/// Only accepts requests from localhost (127.0.0.1 or ::1) for safety.
// shutdown_node lives in admin.rs (not here) to avoid duplicate symbol from glob re-export

#[derive(Debug, Deserialize)]
pub struct HfSearchParams {
    pub query: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct HfDownloadRequest {
    pub repo_id: String,
    pub filename: String,
    #[serde(default)]
    pub mode: String,
}

#[derive(Debug, Deserialize)]
pub struct HfShardDownloadRequest {
    pub repo_id: String,
    pub filename: String,
    /// Which shard indices to download (e.g. [0,1,2] for the first 3 shards).
    /// If empty, the server will probe the file and return shard info without downloading.
    #[serde(default)]
    pub shards: Vec<u32>,
    /// Optional: target an existing model_id so downloaded shards merge into its directory.
    /// If omitted, a new model_id is derived from the filename.
    #[serde(default)]
    pub model_id: Option<String>,
    /// When true AND `shards` is empty: compute a deterministic fair share of shards
    /// based on the node's identity and peer count. Each node claims `ceil(shard_count / (peers + 1))`
    /// shards, with assignment determined by BLAKE3(node_id || model_id) for consistency.
    /// Peers with auto-manage enabled will auto-acquire the remaining shards.
    #[serde(default)]
    pub peer_fair_share: bool,
}

/// GET /api/admin/hf/probe — Probe a remote GGUF file to get shard info.
pub async fn hf_probe(
    State(state): State<AppState>,
    axum::extract::Query(params): axum::extract::Query<HfProbeParams>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let repo_id = params.repo_id.unwrap_or_default();
    let filename = params.filename.unwrap_or_default();

    validate_hf_inputs(&repo_id, &filename)?;

    let shard_size = state.config.model.shard_size_bytes();
    match crate::model::huggingface::probe_gguf_file(&repo_id, &filename, shard_size).await {
        Ok(info) => {
            // Cache probe result so the frontend can look up HF source later
            let mid = crate::types::ModelId(gguf_filename_to_model_id(&filename));
            let probe_info = crate::daemon::HfProbeInfo {
                repo_id: repo_id.clone(),
                filename: filename.clone(),
                shard_count: info.shard_count(),
                total_size_bytes: info.total_size,
                probed_at: chrono::Utc::now(),
            };
            // Count unique peers hosting shards of this model
            let network_replicas = count_unique_shard_holders(
                &state.shared_state.model_registry,
                std::slice::from_ref(&mid),
            );

            // Cap probe cache at 1000 entries — evict oldest by probed_at.
            // Note: len() check + insert is not atomic, so under concurrent admin
            // requests the cache may briefly exceed MAX_PROBE_CACHE. This is bounded
            // by the number of concurrent hf_probe requests (admin-only, typically 1).
            const MAX_PROBE_CACHE: usize = 1_000;
            if state.shared_state.models.hf_probe_cache.len() >= MAX_PROBE_CACHE {
                // Clone key before remove to avoid holding DashMap Ref across remove()
                let oldest = state
                    .shared_state
                    .models
                    .hf_probe_cache
                    .iter()
                    .min_by_key(|entry| entry.value().probed_at)
                    .map(|entry| entry.key().clone());
                if let Some(key) = oldest {
                    state.shared_state.models.hf_probe_cache.remove(&key);
                }
            }
            state
                .shared_state
                .models
                .hf_probe_cache
                .insert(mid, probe_info);

            let arch_str = &info.tensor_meta.architecture;
            let model_arch = crate::inference::split::ModelArch::from_gguf_arch(arch_str);

            Ok(Json(serde_json::json!({
                "status": "ok",
                "total_size": info.total_size,
                "header_size": info.header_size,
                "shard_count": info.shard_count(),
                "architecture": arch_str,
                "architecture_supported": model_arch.is_supported(),
                "network_replicas": network_replicas,
            })))
        }
        Err(e) => Err(ApiError(crate::error::SwarmError::ServiceUnavailable(e))),
    }
}

#[derive(Debug, Deserialize)]
pub struct HfProbeParams {
    pub repo_id: Option<String>,
    pub filename: Option<String>,
}

/// POST /api/admin/hf/download-shards — Download specific shards of a GGUF from HuggingFace.
///
/// Instead of downloading the full multi-GB GGUF file, this downloads only the
/// GGUF header (~6MB) plus the requested shard byte ranges (~512MB each).
/// After download, it generates a manifest and registers the shards.
pub async fn hf_download_shards(
    State(state): State<AppState>,
    Json(body): Json<HfShardDownloadRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let repo_id = body.repo_id;
    let filename = body.filename;
    let shard_indices = body.shards;
    let peer_fair_share = body.peer_fair_share;

    validate_hf_inputs(&repo_id, &filename)?;

    if shard_indices.is_empty() && !peer_fair_share {
        return Err(ApiError(crate::error::SwarmError::Validation(
            "shards array is required (e.g. [0, 1, 2])".into(),
        )));
    }

    if shard_indices.len() > 256 {
        return Err(ApiError(crate::error::SwarmError::Validation(
            "Too many shards requested (max 256)".into(),
        )));
    }

    tracing::info!(
        repo_id = %repo_id,
        filename = %filename,
        shard_count = shard_indices.len(),
        peer_fair_share,
        "DIAG: hf_download_shards handler"
    );

    // peer_fair_share: compute shard assignment deterministically.
    // Deferred until after probe (we need shard_count), so store the peer count now.
    let fair_share_peer_count = if peer_fair_share && shard_indices.is_empty() {
        Some(state.shared_state.peer_registry.len())
    } else {
        None
    };
    let fair_share_node_id = state.shared_state.identity.node_id().clone();

    // Use provided model_id if it matches an existing model, otherwise derive from filename.
    // Always sanitize to prevent path traversal.
    let safe_name = if let Some(ref mid) = body.model_id {
        if mid.len() > 256 {
            return Err(ApiError(crate::error::SwarmError::Validation(
                "model_id must be 256 characters or fewer".into(),
            )));
        }
        let sanitized = crate::model::shard::sanitize_path_component(mid);
        if crate::model::shard::model_dir(&state.config.node.data_dir, &sanitized).exists() {
            sanitized
        } else {
            gguf_filename_to_model_id(&filename)
        }
    } else {
        gguf_filename_to_model_id(&filename)
    };

    let dest_dir = crate::model::shard::model_dir(&state.config.node.data_dir, &safe_name);

    tracing::info!(
        repo = %repo_id,
        file = %filename,
        shards = ?shard_indices,
        dest = %dest_dir.display(),
        "Starting HuggingFace shard download"
    );

    let model_id_str = safe_name.clone();
    let mid = crate::types::ModelId(model_id_str.clone());

    // ── Trust: pin this model as user-approved ──────────────────────────
    // User explicitly chose to download → set Pinned trust level so auto-manage
    // will propagate shards for this model across the network.
    {
        let mut trust = state
            .shared_state
            .models
            .model_trust
            .entry(mid.clone())
            .or_insert_with(crate::types::ModelTrustInfo::new_pinned);
        if !trust.pinned_by_user {
            trust.pinned_by_user = true;
            if trust.trust_level < crate::types::ModelTrustLevel::Pinned {
                trust.trust_level = crate::types::ModelTrustLevel::Pinned;
            }
        }
        let _ = state
            .shared_state
            .db
            .put_json("model_trust", &mid.0, trust.value());
    }

    // ── Synchronous probe + architecture check ──────────────────────────
    // Probe before spawning the download task so we can return an immediate
    // HTTP error for unsupported architectures (fast: reads ~few KB header).
    let configured_shard_size = state.shared_state.config.model.shard_size_bytes();
    let info =
        crate::model::huggingface::probe_gguf_file(&repo_id, &filename, configured_shard_size)
            .await
            .map_err(|e| {
                tracing::error!(error = %e, "HuggingFace probe failed");
                ApiError(crate::error::SwarmError::ServiceUnavailable(format!(
                    "HuggingFace probe failed: {e}"
                )))
            })?;

    let arch_str = &info.tensor_meta.architecture;
    let model_arch = crate::inference::split::ModelArch::from_gguf_arch(arch_str);
    if !model_arch.is_supported() {
        let msg = format!(
            "Unsupported architecture '{}'. Supported: {}",
            arch_str,
            crate::inference::split::ModelArch::supported_list().join(", ")
        );
        tracing::warn!(%arch_str, "Refusing download: unsupported architecture");
        return Err(ApiError(crate::error::SwarmError::Validation(msg)));
    }

    // Create initial acquisition progress entry with per-shard progress so that
    // auto-manage can detect these downloads are already in flight and skip them.
    let log_msg = if peer_fair_share && shard_indices.is_empty() {
        format!("Computing fair share of {} from HuggingFace...", filename)
    } else {
        format!(
            "Downloading shards {:?} of {} from HuggingFace...",
            shard_indices, filename
        )
    };
    let mut initial_shard_progress = std::collections::HashMap::new();
    for &idx in &shard_indices {
        initial_shard_progress.insert(
            idx,
            crate::model::acquisition::ShardProgress {
                index: idx,
                total_bytes: 0,
                downloaded_bytes: 0,
                state: crate::model::acquisition::ShardState::Downloading,
            },
        );
    }
    let mut status = crate::model::acquisition::AcquisitionStatus::new_downloading(
        mid.clone(),
        shard_indices.len() as u32,
        0,
        "huggingface",
        "user",
        log_msg,
    );
    status.shard_progress = initial_shard_progress;
    let shared = state.shared_state.clone();
    shared
        .models
        .acquisition_progress
        .insert(mid.clone(), status);

    // Clone values needed both in the spawn and the response
    let response_model_id = model_id_str.clone();
    let response_shards = shard_indices.clone();

    // Register cancellation flag for this download
    let cancel_flag = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    shared
        .models
        .download_cancel_flags
        .insert(mid.clone(), cancel_flag.clone());

    // Capture network_tx for broadcasting HfSourceGossip + ModelManifest after download
    let network_tx = state.network_tx.clone();

    tokio::spawn(async move {
        let shutdown_rx = shared.shutdown_rx();
        let download_mid = mid.clone();
        let download_shared = shared.clone();

        // peer_fair_share: download just ONE seed shard. Auto-manage handles the rest.
        // Each node picks a deterministic shard (based on node_id hash) so that
        // different nodes seed different shards when they add the same model.
        let shard_indices = if let Some(peer_count) = fair_share_peer_count {
            let total_shards = info.shard_count() as u32;

            // Deterministic shard selection: hash(node_id || model_id) → shard index
            let mut hasher = blake3::Hasher::new();
            hasher.update(fair_share_node_id.0.as_ref());
            hasher.update(model_id_str.as_bytes());
            let hash = hasher.finalize();
            let seed_shard = u32::from_le_bytes([
                hash.as_bytes()[0],
                hash.as_bytes()[1],
                hash.as_bytes()[2],
                hash.as_bytes()[3],
            ]) % total_shards;

            let assigned = vec![seed_shard];

            tracing::info!(
                total_shards,
                peers = peer_count,
                seed_shard,
                "peer_fair_share: seeding 1 shard (auto-manage will acquire more as needed)"
            );

            // Update acquisition progress with the single seed shard
            if let Some(mut entry) = download_shared
                .models
                .acquisition_progress
                .get_mut(&download_mid)
            {
                entry.total_shards = 1;
                entry.log_push(format!(
                    "Seeding shard {seed_shard}/{total_shards} — auto-manage will acquire more as peers join"
                ));
                entry.shard_progress.insert(
                    seed_shard,
                    crate::model::acquisition::ShardProgress {
                        index: seed_shard,
                        total_bytes: 0,
                        downloaded_bytes: 0,
                        state: crate::model::acquisition::ShardState::Downloading,
                    },
                );
            }
            assigned
        } else {
            shard_indices
        };

        if let Some(mut entry) = download_shared
            .models
            .acquisition_progress
            .get_mut(&download_mid)
        {
            // Set total_bytes to the sum of requested shards only (not full model size)
            let requested_bytes: u64 = shard_indices
                .iter()
                .filter_map(|&idx| info.layouts.get(idx as usize))
                .map(|l| l.size_bytes)
                .sum();
            entry.total_bytes = requested_bytes;
            // Don't overwrite total_shards — keep as the requested count, not the full model count
            entry.log_push(format!(
                "Probed: {} shards, {:.1} MB total",
                info.shard_count(),
                info.total_size as f64 / (1024.0 * 1024.0)
            ));
            // Set per-shard total_bytes now that we know sizes from the probe
            for &idx in &shard_indices {
                if let Some(layout) = info.layouts.get(idx as usize) {
                    if let Some(sp) = entry.shard_progress.get_mut(&idx) {
                        sp.total_bytes = layout.size_bytes;
                    }
                }
            }
        }

        // Download GGUF header (~6MB) — needed for manifest generation
        if let Err(e) = crate::model::huggingface::download_gguf_header(
            &repo_id,
            &filename,
            &dest_dir,
            info.header_size,
        )
        .await
        {
            tracing::error!(error = %e, "GGUF header download failed");
            if let Some(mut entry) = download_shared
                .models
                .acquisition_progress
                .get_mut(&download_mid)
            {
                entry.state =
                    crate::model::acquisition::AcquisitionState::Failed { reason: e.clone() };
                entry.log_push(format!("Header download failed: {}", e));
            }
            download_shared.emit_activity(
                crate::daemon::state::ActivityEvent::new(
                    "download",
                    "hf_download_failed",
                    format!("GGUF header download failed: {}", e),
                )
                .with_model(download_mid.0.clone())
                .with_detail_str(e)
                .with_toast("error", 6000),
            );
            return;
        }

        // Download tied output weight if model is weight-tied (no output.weight tensor).
        // This is needed by the last node in distributed inference for logit projection.
        if let Err(e) = crate::model::huggingface::download_tied_output_weight(
            &repo_id,
            &filename,
            &dest_dir,
            &info.tensor_meta,
        )
        .await
        {
            tracing::warn!(error = %e, "Tied output weight download failed (non-fatal)");
        }

        // Generate manifest from header BEFORE downloading shard data.
        // Pass empty shard_indices — no shards to register yet (they don't exist on disk).
        let header_path = dest_dir.join(crate::model::shard::HEADER_FILENAME);
        let manifest_result = generate_manifest_from_header(&ManifestGenParams {
            header_path: &header_path,
            model_id_str: &model_id_str,
            filename: &filename,
            total_size: info.total_size,
            shard_count: info.shard_count(),
            shard_indices: &[],
            shared: &download_shared,
            precomputed_layouts: Some(&info.layouts),
        });

        if let Err(e) = &manifest_result {
            tracing::error!(error = %e, "Manifest generation failed (early broadcast skipped)");
            if let Some(mut entry) = download_shared
                .models
                .acquisition_progress
                .get_mut(&download_mid)
            {
                entry.log_push(format!("Manifest generation failed: {e}"));
            }
            // Continue with downloads anyway — manifest can be regenerated later
        }

        // Record HF source so auto-manager (and peers) know where to fetch shards
        let hf_source = crate::daemon::HfSource {
            repo_id: repo_id.clone(),
            filename: filename.clone(),
            mmproj_filename: None,
        };
        download_shared.models.hf_sources.insert(
            crate::types::ModelId(model_id_str.clone()),
            hf_source.clone(),
        );
        let _ = download_shared
            .db
            .put_json("hf_sources", &model_id_str, &hf_source);
        let hf_source_path = dest_dir.join(crate::model::shard::HF_SOURCE_FILENAME);
        let hf_source_json = serde_json::to_string_pretty(&hf_source).unwrap_or_default();
        let _ =
            tokio::task::spawn_blocking(move || std::fs::write(&hf_source_path, hf_source_json))
                .await;

        // Broadcast HfSourceGossip + ModelManifest EARLY so peers can start
        // auto-acquiring shards immediately (before our shard data downloads finish).
        if let Some(ref ntx) = network_tx {
            let gossip_msg =
                crate::types::SwarmMessage::HfSourceGossip(crate::types::HfSourceGossip {
                    model_id: crate::types::ModelId(model_id_str.clone()),
                    repo_id: repo_id.clone(),
                    filename: filename.clone(),
                    publisher: download_shared.identity.node_id().clone(),
                    mmproj_filename: None,
                });
            let _ = ntx
                .send(crate::types::NetworkCommand::Broadcast(gossip_msg))
                .await;

            if let Some(manifest) = download_shared
                .model_registry
                .get_manifest(&crate::types::ModelId(model_id_str.clone()))
            {
                let _ = ntx
                    .send(crate::types::NetworkCommand::Broadcast(
                        crate::types::SwarmMessage::ModelManifest(manifest),
                    ))
                    .await;
            }

            tracing::info!(model = %model_id_str, "Broadcast manifest + HF source EARLY (before shard downloads)");

            // Broadcast download intent for each shard so peers know we're
            // working on them and auto-manage won't duplicate the download.
            let our_node_id = download_shared.identity.node_id().clone();
            for &idx in &shard_indices {
                let intent_msg = crate::types::SwarmMessage::ShardDownloadProgress(
                    crate::types::ShardDownloadProgress {
                        node_id: our_node_id.clone(),
                        shard_id: crate::types::ShardId {
                            model_id: crate::types::ModelId(model_id_str.clone()),
                            index: idx,
                        },
                        progress_pct: 0,
                        state: crate::types::DownloadState::Downloading,
                    },
                );
                let _ = ntx
                    .send(crate::types::NetworkCommand::Broadcast(intent_msg))
                    .await;
            }
        }

        // NOTE: Do NOT wake auto-manage here. Shards aren't downloaded yet,
        // so holder_count == 0 and auto-manage would race to download them
        // from HF. The notify happens AFTER downloads complete (line ~1666).

        // ── Phase 2: Download shard data ────────────────────────────────

        let (ptx, prx) =
            tokio::sync::mpsc::channel::<crate::model::huggingface::DownloadProgress>(64);

        spawn_progress_updater(shared.clone(), mid.clone(), prx);

        // Download individual layer-aligned shards
        let total_shard_bytes: u64 = shard_indices
            .iter()
            .filter_map(|&idx| info.layouts.get(idx as usize))
            .map(|layout| layout.size_bytes)
            .sum();

        let mut cumulative_downloaded: u64 = 0;
        let mut failed = false;

        for &shard_idx in &shard_indices {
            // Check cancellation flag and shutdown before each shard download
            if cancel_flag.load(std::sync::atomic::Ordering::Acquire) || *shutdown_rx.borrow() {
                let reason = if *shutdown_rx.borrow() {
                    "Cancelled by daemon shutdown"
                } else {
                    "Cancelled by user"
                };
                tracing::info!(model = %model_id_str, reason, "Download cancelled");
                if let Some(mut entry) = download_shared
                    .models
                    .acquisition_progress
                    .get_mut(&download_mid)
                {
                    entry.state = crate::model::acquisition::AcquisitionState::Failed {
                        reason: reason.to_string(),
                    };
                    entry.log_push(reason.to_string());
                }
                // Clean up cancel flag
                download_shared
                    .models
                    .download_cancel_flags
                    .remove(&download_mid);
                return;
            }

            let layout = match info.layouts.get(shard_idx as usize) {
                Some(l) => l,
                None => {
                    tracing::error!(
                        shard_idx,
                        max = info.layouts.len().saturating_sub(1),
                        "Shard index out of range"
                    );
                    failed = true;
                    break;
                }
            };

            let (shard_tx, mut shard_rx) =
                tokio::sync::mpsc::channel::<crate::model::huggingface::DownloadProgress>(64);
            let progress_tx = ptx.clone();
            let base_downloaded = cumulative_downloaded;
            let total = total_shard_bytes;
            let shard_progress_shared = shared.clone();
            let shard_progress_mid = mid.clone();
            let gossip_ntx = network_tx.clone();
            let gossip_node_id = shared.identity.node_id().clone();
            let gossip_model_id = model_id_str.clone();
            let progress_task = tokio::spawn(async move {
                let mut last_broadcast_pct: u32 = 0;
                while let Some(prog) = shard_rx.recv().await {
                    // Forward cumulative bytes to the overall progress updater
                    let _ = progress_tx.try_send(crate::model::huggingface::DownloadProgress {
                        downloaded_bytes: base_downloaded + prog.downloaded_bytes,
                        total_bytes: total,
                    });
                    // Update per-shard progress directly
                    if let Some(mut entry) = shard_progress_shared
                        .models
                        .acquisition_progress
                        .get_mut(&shard_progress_mid)
                    {
                        if let Some(sp) = entry.shard_progress.get_mut(&shard_idx) {
                            sp.downloaded_bytes = prog.downloaded_bytes;
                            if sp.total_bytes == 0 {
                                sp.total_bytes = prog.total_bytes;
                            }
                        }
                    }
                    // Broadcast progress to peers every ~2% so they see near real-time updates
                    let pct = crate::model::acquisition::shard_pct(
                        prog.downloaded_bytes,
                        prog.total_bytes,
                    );
                    if let Some(ref ntx) = gossip_ntx {
                        let sid = crate::types::ShardId {
                            model_id: crate::types::ModelId(gossip_model_id.clone()),
                            index: shard_idx,
                        };
                        last_broadcast_pct =
                            crate::model::acquisition::maybe_broadcast_shard_progress(
                                ntx,
                                &gossip_node_id,
                                &sid,
                                pct,
                                last_broadcast_pct,
                                2,
                            );
                    }
                }
            });

            match crate::model::huggingface::download_shard(
                &repo_id,
                &filename,
                &dest_dir,
                layout,
                Some(shard_tx),
                Some(cancel_flag.as_ref()),
            )
            .await
            {
                Ok(_shard_path) => {
                    progress_task.abort();
                    cumulative_downloaded += layout.size_bytes;

                    if let Some(mut entry) = download_shared
                        .models
                        .acquisition_progress
                        .get_mut(&download_mid)
                    {
                        entry.downloaded_shards += 1;
                        entry.log_push(format!("Shard {} downloaded", shard_idx));
                        // Mark this shard's progress as complete so check_and_load_model
                        // won't skip it as "still downloading"
                        if let Some(sp) = entry.shard_progress.get_mut(&shard_idx) {
                            sp.state = crate::model::acquisition::ShardState::Complete;
                            sp.downloaded_bytes = sp.total_bytes;
                        }
                    }

                    // Register + announce the shard to the network
                    let shard_id = crate::types::ShardId {
                        model_id: crate::types::ModelId(model_id_str.clone()),
                        index: shard_idx,
                    };
                    if let Some(ref ntx) = network_tx {
                        download_shared.announce_shard_acquired(ntx, &shard_id);
                    } else {
                        // No network channel — just register locally
                        download_shared.model_registry.record_shard_holder(
                            shard_id,
                            download_shared.identity.node_id().clone(),
                        );
                    }
                }
                Err(e) => {
                    progress_task.abort();
                    tracing::error!(error = %e, shard_idx, "Shard download failed");
                    if let Some(mut entry) = download_shared
                        .models
                        .acquisition_progress
                        .get_mut(&download_mid)
                    {
                        entry.failed_shards += 1;
                        entry.log_push(format!("Shard {} failed: {}", shard_idx, e));
                    }
                    download_shared.emit_activity(
                        crate::daemon::state::ActivityEvent::new(
                            "download",
                            "shard_download_failed",
                            format!("Shard {} download failed: {}", shard_idx + 1, e),
                        )
                        .with_model(download_mid.0.clone())
                        .with_detail_num(shard_idx as i64)
                        .with_detail_str(e.to_string())
                        .with_toast("error", 6000),
                    );
                    failed = true;
                    break;
                }
            }
        }

        // Drop the progress sender so the updater task exits
        drop(ptx);

        // Clean up cancel flag
        download_shared
            .models
            .download_cancel_flags
            .remove(&download_mid);

        if failed {
            if let Some(mut entry) = download_shared
                .models
                .acquisition_progress
                .get_mut(&download_mid)
            {
                entry.state = crate::model::acquisition::AcquisitionState::Failed {
                    reason: "One or more shard downloads failed".to_string(),
                };
            }
        } else {
            tracing::info!(
                model = %model_id_str,
                shards = ?shard_indices,
                "All shard downloads complete"
            );

            // Regenerate manifest with correct BLAKE3 hashes now that shard files
            // exist on disk. The early manifest had [0u8; 32] placeholders.
            if let Err(e) = generate_manifest_from_header(&ManifestGenParams {
                header_path: &header_path,
                model_id_str: &model_id_str,
                filename: &filename,
                total_size: info.total_size,
                shard_count: info.shard_count(),
                shard_indices: &shard_indices,
                shared: &download_shared,
                precomputed_layouts: Some(&info.layouts),
            }) {
                tracing::error!(error = %e, model = %model_id_str, "Final manifest regeneration failed after shard download");
            }

            if let Some(mut entry) = download_shared
                .models
                .acquisition_progress
                .get_mut(&download_mid)
            {
                entry.state = crate::model::acquisition::AcquisitionState::Complete;
                entry.verified_shards = shard_indices.len() as u32;
                entry
                    .log
                    .push("All shards downloaded and registered".to_string());
            }

            // Load available shards for inference (partial is fine)
            let vram_budget = crate::model::auto_manage::compute_vram_budget(&download_shared);
            crate::model::auto_manage::check_and_load_model(
                &download_shared,
                &crate::types::ModelId(model_id_str.clone()),
                vram_budget,
            )
            .await;

            // Notify dashboard that models have changed
            let _ = download_shared
                .events
                .dashboard_tx
                .send(crate::daemon::state::DashboardSignal::ModelsChanged);

            // Emit activity event for HF download completion
            download_shared.emit_activity(
                crate::daemon::state::ActivityEvent::new(
                    "download",
                    "hf_download_complete",
                    format!("HuggingFace download complete: {}", model_id_str),
                )
                .with_model(model_id_str.clone())
                .with_detail_str("huggingface".to_string())
                .with_toast("success", 8000),
            );

            // Wake auto-manage again to re-evaluate (maybe download more shards)
            download_shared.models.auto_manage_notify.notify_one();

            // Clean up acquisition_progress after a delay so the frontend sees
            // the "complete" state and triggers a re-render before we remove it.
            download_shared.schedule_acquisition_cleanup(download_mid.clone());
        }
    });

    Ok(Json(serde_json::json!({
        "status": "started",
        "model_id": response_model_id,
        "shards": response_shards,
        "peer_fair_share": peer_fair_share,
    })))
}

struct ManifestGenParams<'a> {
    header_path: &'a std::path::Path,
    model_id_str: &'a str,
    filename: &'a str,
    total_size: u64,
    shard_count: u32,
    shard_indices: &'a [u32],
    shared: &'a std::sync::Arc<crate::daemon::SharedState>,
    /// Pre-computed layouts from probe. When provided, these are used directly
    /// instead of recomputing (avoids shard_count mismatch between probe and manifest).
    precomputed_layouts: Option<&'a [crate::inference::split::LayerShardLayout]>,
}

/// Generate a manifest from a downloaded GGUF header and register shards.
fn generate_manifest_from_header(params: &ManifestGenParams<'_>) -> Result<(), String> {
    use crate::inference::split::GgufTensorMeta;

    let header_path = params.header_path;
    let total_size = params.total_size;
    let shard_count = params.shard_count;

    // Parse model metadata from the GGUF header
    let meta = GgufTensorMeta::from_gguf_file(header_path)
        .map_err(|e| format!("Failed to parse GGUF header: {e}"))?;

    let model_id = crate::types::ModelId(params.model_id_str.to_string());
    let num_layers = meta.block_count as u32;

    // Build a friendly model name from the GGUF metadata or filename
    let model_name = meta
        .model_name
        .clone()
        .unwrap_or_else(|| params.filename.trim_end_matches(".gguf").to_string());

    // Architecture already extracted by GgufTensorMeta above — no need to re-read the file
    let architecture = match meta.architecture.as_str() {
        "qwen2" | "qwen3" | "qwen2moe" => crate::types::ModelArchitecture::Qwen2,
        "mistral" => crate::types::ModelArchitecture::Mistral,
        "phi" | "phi3" => crate::types::ModelArchitecture::Phi,
        _ => crate::types::ModelArchitecture::Llama,
    };

    let model_dir = header_path
        .parent()
        .ok_or_else(|| "GGUF header path has no parent directory".to_string())?;

    let computed_layouts;
    let layouts: &[crate::inference::split::LayerShardLayout] = if let Some(precomputed) =
        params.precomputed_layouts
    {
        precomputed
    } else {
        computed_layouts = crate::inference::split::compute_layer_shard_layouts(&meta, shard_count);
        &computed_layouts
    };
    let shards = crate::model::manifest::build_shard_infos_from_layouts(model_dir, layouts);

    let node_id = params.shared.identity.node_id().clone();

    let mut manifest = crate::types::ModelManifest {
        id: model_id.clone(),
        name: model_name,
        architecture,
        num_layers,
        num_params_billions: 0.0,
        quantization: crate::types::Quantization::Q4KM,
        total_size_bytes: total_size,
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

    // Save manifest to disk
    manifest.save_to_dir(model_dir).map_err(|e| e.to_string())?;

    // Register manifest in the model registry
    params
        .shared
        .model_registry
        .register_manifest(manifest.clone());

    // Store GGUF metadata
    params.shared.gguf_meta.insert(model_id.clone(), meta);

    // Register this node as holder of the downloaded shards
    for &shard_idx in params.shard_indices {
        let shard_id = crate::types::ShardId {
            model_id: model_id.clone(),
            index: shard_idx,
        };
        params
            .shared
            .model_registry
            .record_shard_holder(shard_id, node_id.clone());
    }

    tracing::info!(
        model = %model_id,
        shards_registered = params.shard_indices.len(),
        num_layers,
        "Generated manifest and registered shards from HF download"
    );

    Ok(())
}
/// POST /api/admin/downloads/:model_id/cancel — Cancel an in-progress HF download.
///
/// Sets the cancellation flag so the download loop aborts. Cleans up partial .tmp files.
/// Returns 200 on success, 404 if no active download for that model.
pub async fn cancel_download(
    State(state): State<AppState>,
    Path(model_id): Path<String>,
) -> Result<Json<serde_json::Value>, ApiError> {
    crate::api::admin_models::validate_model_id(&model_id)?;
    let mid = crate::types::ModelId(model_id.clone());
    let shared = &state.shared_state;

    // Check if there's an active download for this model
    let has_active = shared
        .models
        .acquisition_progress
        .get(&mid)
        .map(|entry| {
            matches!(
                entry.state,
                crate::model::acquisition::AcquisitionState::Downloading
                    | crate::model::acquisition::AcquisitionState::AwaitingManifest
            )
        })
        .unwrap_or(false);

    if !has_active {
        return Err(ApiError(crate::error::SwarmError::Validation(format!(
            "No active download found for model '{}'",
            model_id
        ))));
    }

    // Set the cancel flag (the download loop checks this)
    if let Some(flag) = shared.models.download_cancel_flags.get(&mid) {
        flag.store(true, std::sync::atomic::Ordering::Release);
    }

    // Mark the acquisition as failed/cancelled
    if let Some(mut entry) = shared.models.acquisition_progress.get_mut(&mid) {
        entry.state = crate::model::acquisition::AcquisitionState::Failed {
            reason: "Cancelled by user".to_string(),
        };
        entry.log_push("Download cancelled by user".to_string());
    }

    // Clean up partial .tmp files in the model directory
    let model_dir = crate::model::shard::model_dir(&state.config.node.data_dir, &model_id);
    let md = model_dir.clone();
    let _ = tokio::task::spawn_blocking(move || {
        if md.exists() {
            if let Ok(entries) = std::fs::read_dir(&md) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if path.extension().and_then(|e| e.to_str()) == Some("tmp") {
                        tracing::info!(path = %path.display(), "Removing partial download file");
                        let _ = std::fs::remove_file(&path);
                    }
                }
            }
        }
    })
    .await;

    tracing::info!(model = %model_id, "Download cancelled");

    Ok(Json(serde_json::json!({
        "status": "cancelled",
        "model_id": model_id,
    })))
}
/// GET /api/admin/hf/source/:model_id — Look up HuggingFace source for a model.
///
/// Returns the repo_id and filename needed to trigger per-shard downloads.
/// Checks both hf_sources (downloaded models) and hf_probe_cache (probed models).
pub async fn hf_source(
    State(state): State<AppState>,
    Path(model_id): Path<String>,
) -> Result<Json<serde_json::Value>, ApiError> {
    crate::api::admin_models::validate_model_id(&model_id)?;
    let mid = crate::types::ModelId(model_id.clone());

    if let Some(src) = state.shared_state.models.hf_sources.get(&mid) {
        return Ok(Json(serde_json::json!({
            "model_id": model_id,
            "repo_id": src.repo_id,
            "filename": src.filename,
        })));
    }

    if let Some(probe) = state.shared_state.models.hf_probe_cache.get(&mid) {
        return Ok(Json(serde_json::json!({
            "model_id": model_id,
            "repo_id": probe.repo_id,
            "filename": probe.filename,
        })));
    }

    // Fallback: try to auto-discover HF source by searching HuggingFace.
    // The model_id is a slug derived from the GGUF filename (lowercase, hyphens).
    // Strip the quant suffix to get a cleaner search query.
    let search_query = {
        let mut q = model_id.clone();
        // Remove common quant suffixes for a better search
        for suffix in &[
            ".q4-k-m", ".q4-k-s", ".q5-k-m", ".q5-k-s", ".q6-k", ".q8-0", ".q4-0", ".q4-1",
            ".q5-0", ".q5-1", ".q3-k-m", ".q3-k-s", ".q2-k", ".iq4-xs", ".f16", ".f32", ".bf16",
            "-q4-k-m", "-q4-k-s", "-q5-k-m", "-q5-k-s", "-q6-k", "-q8-0", "-q4-0", "-q4-1",
            "-q5-0", "-q5-1", "-q3-k-m", "-q3-k-s", "-q2-k", "-iq4-xs", "-f16", "-f32", "-bf16",
        ] {
            if let Some(stripped) = q.strip_suffix(suffix) {
                q = stripped.to_string();
                break;
            }
        }
        q
    };

    tracing::info!(
        model = %model_id,
        query = %search_query,
        "Auto-discovering HF source for model"
    );

    match crate::model::huggingface::search_gguf_models(&search_query).await {
        Ok(results) => {
            // Find the result whose filename slug matches our model_id
            if let Some(hit) = results
                .iter()
                .find(|r| gguf_filename_to_model_id(&r.filename) == model_id)
            {
                // Cache the discovered source for future lookups
                let source = crate::daemon::HfSource {
                    repo_id: hit.repo_id.clone(),
                    filename: hit.filename.clone(),
                    mmproj_filename: None,
                };
                state
                    .shared_state
                    .models
                    .hf_sources
                    .insert(mid.clone(), source);
                let _ = state.db.put_json(
                    "hf_sources",
                    &model_id,
                    &crate::daemon::HfSource {
                        repo_id: hit.repo_id.clone(),
                        filename: hit.filename.clone(),
                        mmproj_filename: None,
                    },
                );

                // Also write hf_source.json to disk for future startups
                let model_dir =
                    crate::model::shard::model_dir(&state.config.node.data_dir, &model_id);
                if model_dir.is_dir() {
                    let hf_path = model_dir.join(crate::model::shard::HF_SOURCE_FILENAME);
                    let json_str = serde_json::to_string_pretty(&serde_json::json!({
                        "repo_id": hit.repo_id,
                        "filename": hit.filename,
                    }))
                    .unwrap_or_default();
                    let _ = tokio::task::spawn_blocking(move || std::fs::write(&hf_path, json_str))
                        .await;
                }

                tracing::info!(
                    model = %model_id,
                    repo = %hit.repo_id,
                    file = %hit.filename,
                    "Auto-discovered HF source"
                );

                return Ok(Json(serde_json::json!({
                    "model_id": model_id,
                    "repo_id": hit.repo_id,
                    "filename": hit.filename,
                    "auto_discovered": true,
                })));
            }
        }
        Err(e) => {
            tracing::debug!(model = %model_id, error = %e, "HF auto-discovery search failed");
        }
    }

    Err(ApiError(crate::error::SwarmError::Validation(format!(
        "No HuggingFace source found for model '{}'",
        model_id
    ))))
}
