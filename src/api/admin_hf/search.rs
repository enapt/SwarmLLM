use axum::extract::State;
use axum::Json;
use serde::Deserialize;

use crate::api::server::AppState;
use crate::error::ApiError;

use super::{count_unique_shard_holders, gguf_filename_to_model_id};

/// Heuristic shard sizing constants for HF search scoring.
const EST_SHARD_SIZE_BYTES: u64 = 800 * 1024 * 1024;
const EST_SHARD_COUNT_MIN: u64 = 2;
const EST_SHARD_COUNT_MAX: u64 = 16;
const BOOMERANG_SIZE_NUMERATOR: u64 = 12;
const BOOMERANG_SIZE_DENOMINATOR: u64 = 5;
const MODEL_SIZE_SCORE_MAX_GB: f64 = 8.0;

#[derive(Debug, Deserialize)]
pub struct HfSearchParams {
    #[serde(rename = "q")]
    pub query: Option<String>,
}

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
        .map_err(|e| {
            ApiError(crate::error::SwarmError::ServiceUnavailable(
                crate::api::scrub_truncate_error(&e),
            ))
        })?;

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
