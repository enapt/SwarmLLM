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
            // EVERY upload whose file name gives this id is a claim, and the
            // choice among them is the swarm-wide rule — not the search's
            // order, which follows download counts and so changes over time
            // and from node to node. Taking the first hit is how two nodes
            // asking an hour apart came to fetch two different files.
            for hit in results
                .iter()
                .filter(|r| gguf_filename_to_model_id(&r.filename) == model_id)
            {
                state.shared_state.note_origin_claim(
                    &mid,
                    crate::daemon::HfSource {
                        repo_id: hit.repo_id.clone(),
                        filename: hit.filename.clone(),
                        mmproj_filename: None,
                    },
                );
            }
            if let Some(best) = state
                .shared_state
                .origin_candidates(&mid)
                .into_iter()
                .next()
            {
                tracing::info!(
                    model = %model_id,
                    repo = %best.repo_id,
                    file = %best.filename,
                    "Auto-discovered HF source"
                );
                return Ok(Json(serde_json::json!({
                    "model_id": model_id,
                    "repo_id": best.repo_id,
                    "filename": best.filename,
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
