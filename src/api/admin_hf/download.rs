use axum::extract::State;
use axum::Json;
use serde::Deserialize;

use crate::api::server::AppState;
use crate::error::ApiError;

use super::{extract_eos_token_ids, progress::spawn_progress_updater, validate_hf_inputs};

#[derive(Debug, Deserialize)]
pub struct HfDownloadRequest {
    pub repo_id: String,
    pub filename: String,
}

pub async fn hf_download(
    State(state): State<AppState>,
    Json(body): Json<HfDownloadRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let repo_id = body.repo_id;
    let filename = body.filename;

    validate_hf_inputs(&repo_id, &filename)?;

    let dest_dir = state.model_dir(&repo_id);

    tracing::info!(repo = %repo_id, file = %filename, "Starting HuggingFace download");

    // Spawn download in background
    let repo_id = repo_id.clone();
    let filename = filename.clone();
    let shared = state.shared_state.clone();
    let model_id_str = format!("hf:{}/{}", repo_id, filename);
    let mid = crate::types::ModelId(model_id_str.clone());

    // Register the download atomically: AcquisitionStatus + cancel flag.
    let status = crate::model::acquisition::AcquisitionStatus::new_downloading(
        mid.clone(),
        1,
        0,
        "huggingface",
        "user",
        format!("Downloading {} from HuggingFace...", filename),
    );
    let _hf_cancel_flag = shared.models.begin_download(mid.clone(), status);

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
                download_shared.models.update_acquisition(&download_mid, |s| {
                    s.state = crate::model::acquisition::AcquisitionState::Failed {
                        reason: "Cancelled by daemon shutdown".into(),
                    };
                    s.log_push("Cancelled by daemon shutdown".into());
                });
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
                download_shared.models.set_acquisition_complete_single(
                    &download_mid,
                    format!("Download complete: {}", path.display()),
                );

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
                        download_shared
                            .models
                            .update_acquisition(&download_mid, |s| {
                                s.log_push(format!("Model loaded: {}", model_name));
                            });
                        tracing::info!(model = %model_name, "HF model loaded for inference");
                    }
                    Err(e) => {
                        download_shared
                            .models
                            .update_acquisition(&download_mid, |s| {
                                s.log_push(format!("Model load failed: {}", e));
                            });
                        tracing::error!(error = %e, "Failed to load HF model");
                    }
                }
            }
            Err(e) => {
                tracing::error!(error = %e, "HuggingFace download failed");
                download_shared
                    .models
                    .set_acquisition_failed(&download_mid, e.clone());
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
