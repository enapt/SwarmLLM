use axum::extract::{Path, State};
use axum::Json;

use crate::api::server::AppState;
use crate::error::ApiError;

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
    shared.models.update_acquisition(&mid, |s| {
        s.state = crate::model::acquisition::AcquisitionState::Failed {
            reason: "Cancelled by user".to_string(),
        };
        s.log_push("Download cancelled by user".to_string());
    });

    // Clean up partial .tmp files in the model directory
    let model_dir = state.model_dir(&model_id);
    let md = model_dir.clone();
    let _ = tokio::task::spawn_blocking(move || {
        crate::model::shard::ShardStore::cleanup_tmp_files_in_dir(&md);
    })
    .await;

    tracing::info!(model = %model_id, "Download cancelled");

    Ok(Json(serde_json::json!({
        "status": "cancelled",
        "model_id": model_id,
    })))
}
