use axum::extract::State;
use axum::Json;
use serde::Deserialize;

use crate::api::server::AppState;
use crate::error::ApiError;

use super::{count_unique_shard_holders, gguf_filename_to_model_id, validate_hf_inputs};

#[derive(Debug, Deserialize)]
pub struct HfProbeParams {
    pub repo_id: Option<String>,
    pub filename: Option<String>,
}

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
        // A wrong name is the caller's to fix, not a gateway failure. Reporting
        // it as 502 says "this server is broken" about a typo.
        Err(e) if crate::model::huggingface::probe_failure_is_user_fixable(&e) => Err(ApiError(
            crate::error::SwarmError::NotFound(crate::api::scrub_truncate_error(&e)),
        )),
        Err(e) => Err(ApiError(crate::error::SwarmError::ProviderError {
            status: 502,
            body: crate::api::scrub_truncate_error(&e),
        })),
    }
}
