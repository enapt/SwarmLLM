use crate::api::server::JsonBody;
use axum::extract::{Path, State};
use axum::Json;
use serde::Deserialize;

use crate::api::server::AppState;
use crate::error::ApiError;

#[derive(Deserialize)]
pub struct RegisterAdapterRequest {
    pub id: Option<String>,
    pub name: String,
    pub base_model: String,
    pub rank: usize,
    pub alpha: f32,
    /// Path to the safetensors file (relative to data_dir/adapters or absolute).
    pub path: String,
}

/// POST /api/admin/adapters — Register a LoRA adapter.
pub async fn register_adapter(
    State(state): State<AppState>,
    JsonBody(body): JsonBody<RegisterAdapterRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let adapter_id = body.id.unwrap_or_else(|| uuid::Uuid::new_v4().to_string());

    let path = std::path::PathBuf::from(&body.path);
    let adapter_dir = state.shared_state.adapter_registry.adapter_dir();
    let resolved = if path.is_absolute() {
        path
    } else {
        adapter_dir.join(&path)
    };

    // Reject path traversal attempts (e.g. "../../../etc/passwd")
    for component in resolved.components() {
        if matches!(component, std::path::Component::ParentDir) {
            return Err(ApiError(crate::error::SwarmError::Validation(
                "Path traversal not allowed in adapter path".into(),
            )));
        }
    }

    if !resolved.exists() {
        return Err(ApiError(crate::error::SwarmError::Validation(format!(
            "Adapter file not found: {}",
            resolved.display()
        ))));
    }

    // Confine resolved path to the adapter directory (symlinks + absolute paths).
    let canonical = resolved.canonicalize().map_err(|_| {
        ApiError(crate::error::SwarmError::Validation(
            "Adapter path could not be resolved".into(),
        ))
    })?;
    let canonical_root = adapter_dir
        .canonicalize()
        .unwrap_or_else(|_| adapter_dir.to_path_buf());
    if !canonical.starts_with(&canonical_root) {
        return Err(ApiError(crate::error::SwarmError::Validation(
            "Adapter path must be within the adapter directory".into(),
        )));
    }

    let device = candle_core::Device::Cpu;
    let metadata = state.shared_state.adapter_registry.register(
        &adapter_id,
        &body.name,
        &body.base_model,
        body.rank,
        body.alpha,
        &resolved,
        &device,
    )?;

    Ok(Json(serde_json::json!({
        "status": "ok",
        "adapter": metadata,
    })))
}

/// GET /api/admin/adapters — List all registered adapters.
pub async fn list_adapters(
    State(state): State<AppState>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let adapters = state.shared_state.adapter_registry.list();
    Ok(Json(serde_json::json!({
        "adapters": adapters,
    })))
}

/// DELETE /api/admin/adapters/:id — Remove a registered adapter.
pub async fn delete_adapter(
    State(state): State<AppState>,
    Path(adapter_id): Path<String>,
) -> Result<Json<serde_json::Value>, ApiError> {
    if state.shared_state.adapter_registry.remove(&adapter_id) {
        Ok(Json(serde_json::json!({
            "status": "ok",
            "message": format!("Adapter '{adapter_id}' removed"),
        })))
    } else {
        Err(ApiError(crate::error::SwarmError::Validation(format!(
            "Adapter '{}' not found",
            adapter_id
        ))))
    }
}
