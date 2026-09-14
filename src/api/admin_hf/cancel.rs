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

    // Set the cancel flag. `live_cancel_flag` creates one if this model has
    // none, so the flag exists for the next writer even if the current ones
    // started before it — a download that begins between here and the next
    // request sees a flag that is already set and stops immediately, which is
    // what the user asked for.
    shared
        .models
        .live_cancel_flag(&mid)
        .store(true, std::sync::atomic::Ordering::Release);

    // Mark the acquisition as failed/cancelled
    shared.models.update_acquisition(&mid, |s| {
        s.state = crate::model::acquisition::AcquisitionState::Failed {
            reason: "Cancelled by user".to_string(),
        };
        s.log_push("Download cancelled by user".to_string());
    });

    // Clean up partial .tmp files — but ONLY the ones nothing is writing.
    //
    // This used to delete every `*.tmp` in the model directory unconditionally.
    // The downloads that had just been told to stop had not noticed yet (the
    // flag is read once per chunk), so their files were removed out from under
    // them: each writer carried on into an unlinked inode, then checked the
    // size at a path that now held somebody else's file, failed, and deleted
    // that one too. It is the same collision that made a shard re-download
    // from byte zero for ever, triggered by pressing Cancel.
    //
    // A live download cleans up its own `.tmp` and layout sidecar when it sees
    // the flag — as a unit, which a directory sweep cannot do. So all this has
    // to remove is what a PREVIOUS run left behind.
    let model_dir = state.model_dir(&model_id);
    let claims = shared.models.shard_download_claims.clone();
    let cancel_mid = mid.clone();
    let removed = tokio::task::spawn_blocking(move || {
        crate::model::shard::cleanup_tmp_files_no_one_is_writing(&model_dir, &cancel_mid, &claims)
    })
    .await
    .unwrap_or(0);

    tracing::info!(
        model = %model_id,
        stale_tmp_removed = removed,
        "Download cancelled"
    );

    Ok(Json(serde_json::json!({
        "status": "cancelled",
        "model_id": model_id,
    })))
}
