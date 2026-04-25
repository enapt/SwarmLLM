use std::sync::Arc;

use tokio::sync::mpsc;

use crate::types::{NetworkCommand, SwarmMessage};

use super::super::state::SharedState;
use super::ZSTD_COMPRESS_LEVEL;

pub(super) async fn handle_vision_encode_request(
    shared_state: Arc<SharedState>,
    network_tx: mpsc::Sender<NetworkCommand>,
    req: crate::types::VisionEncodeRequest,
) {
    let model_id = &req.model_id;

    // SEC: Reject oversized image payloads BEFORE loading vision module to prevent
    // a malicious peer from triggering expensive module loading with large payloads.
    const MAX_IMAGE_BYTES: usize = 20 * 1024 * 1024; // 20 MB
    if req.image_data.len() > MAX_IMAGE_BYTES {
        tracing::warn!(
            request_id = %req.request_id,
            size = req.image_data.len(),
            max = MAX_IMAGE_BYTES,
            "VisionEncodeRequest image_data too large — rejecting"
        );
        return;
    }

    tracing::info!(
        request_id = %req.request_id,
        model = %model_id,
        image_bytes = req.image_data.len(),
        "Handling VisionEncodeRequest"
    );

    // Load or get the vision module
    let vision_module = if let Some(entry) = shared_state.vision_modules.get(model_id) {
        entry.value().clone()
    } else {
        // Try to load mmproj on-demand
        let model_dir = shared_state.model_dir(&model_id.0);
        let mmproj_path = model_dir.join(crate::model::shard::MMPROJ_FILENAME);
        if !mmproj_path.exists() {
            tracing::warn!(
                request_id = %req.request_id,
                model = %model_id,
                "VisionEncodeRequest received but no mmproj.gguf found"
            );
            return;
        }
        match crate::inference::vision::load_from_mmproj_gguf(
            &mmproj_path,
            &candle_core::Device::Cpu,
        ) {
            Ok(module) => {
                let module = Arc::new(module);
                shared_state
                    .vision_modules
                    .insert(model_id.clone(), module.clone());
                module
            }
            Err(e) => {
                tracing::warn!(
                    request_id = %req.request_id,
                    error = %e,
                    "Failed to load mmproj for VisionEncodeRequest"
                );
                return;
            }
        }
    };

    // Decode JPEG image into ImageData. (The image_data byte cap was checked
    // at the top of this function — req.image_data is immutable, so a
    // re-check here would be unreachable.)
    let img = match image::load_from_memory(&req.image_data) {
        Ok(dyn_img) => {
            let rgb = dyn_img.to_rgb8();
            let (w, h) = rgb.dimensions();
            crate::types::ImageData {
                rgb_bytes: rgb.into_raw(),
                width: w,
                height: h,
            }
        }
        Err(e) => {
            tracing::warn!(
                request_id = %req.request_id,
                error = %e,
                "Failed to decode image in VisionEncodeRequest"
            );
            return;
        }
    };

    // Encode image to vision embeddings (CPU-bound)
    let encode_result = tokio::task::block_in_place(|| vision_module.encode_images(&[img]));
    match encode_result {
        Ok(embeddings) => {
            // Compress embeddings with zstd for wire transfer
            let (num_tokens, hidden_dim) = embeddings.dims2().unwrap_or((0, 0));
            let raw_bytes: Vec<u8> = embeddings
                .to_dtype(candle_core::DType::F16)
                .and_then(|t| t.to_vec2::<half::f16>())
                .map(|v: Vec<Vec<half::f16>>| {
                    let mut bytes = Vec::with_capacity(num_tokens * hidden_dim * 2);
                    for row in v {
                        for f in row {
                            bytes.extend_from_slice(&f.to_le_bytes());
                        }
                    }
                    bytes
                })
                .unwrap_or_default();
            let compressed =
                zstd::encode_all(std::io::Cursor::new(&raw_bytes), ZSTD_COMPRESS_LEVEL)
                    .unwrap_or(raw_bytes);

            let response = crate::types::VisionEncodeResponse {
                request_id: req.request_id,
                embeddings: compressed,
                num_tokens: num_tokens as u32,
                hidden_dim: hidden_dim as u32,
            };

            tracing::info!(
                request_id = %req.request_id,
                num_tokens,
                hidden_dim,
                compressed_bytes = response.embeddings.len(),
                "VisionEncodeRequest completed, sending response"
            );

            let msg = if let Some(target_bytes) = &req.sender_peer_bytes {
                NetworkCommand::SendDirectMessage {
                    target_peer_bytes: target_bytes.clone(),
                    message: SwarmMessage::VisionEncodeResponse(response),
                }
            } else {
                tracing::warn!(request_id = %req.request_id, "VisionEncodeResponse has no sender — dropping");
                return;
            };
            if let Err(e) = network_tx.send(msg).await {
                tracing::warn!(error = %e, "Failed to send VisionEncodeResponse");
            }
        }
        Err(e) => {
            tracing::warn!(
                request_id = %req.request_id,
                error = %e,
                "Vision encoding failed"
            );
        }
    }
}
