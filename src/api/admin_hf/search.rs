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
    /// R114: optional task filter — comma-separated tokens
    /// (chat / code / vision / multilingual / reasoning). Results that
    /// don't match any of the requested tokens are dropped. Empty/missing
    /// = no filter (returns everything).
    #[serde(default)]
    pub tasks: Option<String>,
}

pub async fn hf_search(
    State(state): State<AppState>,
    axum::extract::Query(params): axum::extract::Query<HfSearchParams>,
) -> Result<Json<Vec<serde_json::Value>>, ApiError> {
    let query = params.query.unwrap_or_default();
    if query.is_empty() {
        return Ok(Json(vec![]));
    }
    // chars().count() so non-ASCII queries don't get rejected by byte
    // counting when the user-facing limit is character-count (R93/R94
    // pool-name pattern).
    if query.chars().count() > 256 {
        return Err(ApiError(crate::error::SwarmError::Validation(
            "Search query too long (max 256 chars)".into(),
        )));
    }

    let results = crate::model::huggingface::search_gguf_models(&query)
        .await
        .map_err(|e| {
            // Upstream HuggingFace failure → ProviderError, NOT
            // ServiceUnavailable (which is for local-server outages).
            // Matches R93's probe.rs fix; same error-contract rule.
            ApiError(crate::error::SwarmError::ProviderError {
                status: 502,
                body: crate::api::scrub_truncate_error(&e),
            })
        })?;

    // R114: task-filter parsing. Tokens are case-insensitive; unknown
    // tokens are silently ignored so a future filter chip the backend
    // doesn't recognise yet doesn't break the response.
    if let Some(ref t) = params.tasks {
        if t.len() > 512 {
            return Err(ApiError(crate::error::SwarmError::Validation(
                "tasks filter too long (max 512 bytes)".into(),
            )));
        }
    }
    let task_filter: Option<std::collections::HashSet<String>> = params.tasks.as_ref().map(|s| {
        s.split(',')
            .map(|t| t.trim().to_lowercase())
            .filter(|t| !t.is_empty())
            .collect()
    });

    // Map repo_id → task tags from the HfWatcher's trending cache. For
    // non-trending repos we infer tags from the repo name (best-effort
    // — tags help filtering, not correctness, so a miss just falls
    // back to "uncategorised").
    let trending_snapshot = state.shared_state.models.hf_trending_cache.load_full();
    let mut tags_by_repo: std::collections::HashMap<String, Vec<String>> =
        std::collections::HashMap::new();
    for e in &trending_snapshot.entries {
        tags_by_repo.insert(e.repo_id.clone(), e.task_tags.clone());
    }

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
        .filter_map(|(repo_id, files)| {
            // R114: task tags + filter. Tags from the HfWatcher trending
            // cache (authoritative); fall back to a tiny name-based heuristic
            // for non-trending repos so the chips still do something useful.
            let task_tags: Vec<String> = tags_by_repo
                .get(&repo_id)
                .cloned()
                .unwrap_or_else(|| infer_task_tags_from_repo_name(&repo_id));
            if let Some(ref filter) = task_filter {
                if !filter.is_empty()
                    && !task_tags.iter().any(|t| filter.contains(&t.to_lowercase()))
                {
                    return None;
                }
            }
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

            // R114: status-driven CTA. Drives a single button per result
            // instead of forcing the user to interpret a 0..100 composite
            // score. The mapping mirrors the wishlist's status taxonomy
            // (Hosting / Serveable / Aspirational / Unreachable / Blocked)
            // so non-technical users see consistent language across views.
            //
            // Bug fix (multi-node test): when the local node can't fit *any*
            // shard but peers host the model, the old logic flagged it as
            // "unreachable" — which is wrong. If `network_replicas > 0` the
            // model IS reachable via remote inference through those peers,
            // so report `swarm_serveable` instead.
            let status = if !fits_full && !fits_boomerang && !fits_shard {
                if network_replicas > 0 {
                    "swarm_serveable"
                } else {
                    "unreachable"
                }
            } else if network_replicas == 0 && (fits_boomerang || fits_full) {
                "be_first_host"
            } else if network_replicas > 0 && network_replicas < 3 {
                "needs_more_hosts"
            } else if network_replicas >= 3 {
                "well_replicated"
            } else {
                "downloadable"
            };

            Some(serde_json::json!({
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
                "task_tags": task_tags,
                "swarm_cta_status": status,
            }))
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

/// R114: tiny name-based tag inference for non-trending repos. The
/// HfWatcher trending cache is the authoritative source; this fallback
/// just makes the chips do something useful for repos HfWatcher hasn't
/// seen. Best-effort; users can always remove a filter to see everything.
fn infer_task_tags_from_repo_name(repo_id: &str) -> Vec<String> {
    let lower = repo_id.to_lowercase();
    let mut tags: Vec<String> = Vec::new();
    if lower.contains("code")
        || lower.contains("coder")
        || lower.contains("starcoder")
        || lower.contains("granite-code")
    {
        tags.push("code".to_string());
    }
    if lower.contains("vision")
        || lower.contains("vlm")
        || lower.contains("llava")
        || lower.contains("multimodal")
    {
        tags.push("vision".to_string());
    }
    if lower.contains("math")
        || lower.contains("reasoning")
        || lower.contains("o1")
        || lower.contains("deepseek-r1")
    {
        tags.push("reasoning".to_string());
    }
    if lower.contains("multilingual")
        || lower.contains("aya")
        || lower.contains("nllb")
        || lower.contains("madlad")
    {
        tags.push("multilingual".to_string());
    }
    // Default fallback so the chips always do something. Most GGUFs
    // are chat / instruct fine-tunes anyway.
    if tags.is_empty() || lower.contains("chat") || lower.contains("instruct") {
        tags.push("chat".to_string());
    }
    tags
}
