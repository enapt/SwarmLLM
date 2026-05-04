use axum::extract::{Path, State};
use axum::Json;

use crate::api::server::AppState;
use crate::error::ApiError;

use super::gguf_filename_to_model_id;

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
                let model_dir = state.model_dir(&model_id);
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

    Err(ApiError(crate::error::SwarmError::NotFound(format!(
        "No HuggingFace source found for model '{}'",
        model_id
    ))))
}
