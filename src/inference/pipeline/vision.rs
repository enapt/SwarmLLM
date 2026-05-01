//! Vision-embedding pre-computation for multimodal inference. Encodes
//! image inputs either locally (if an mmproj is loaded/available) or by
//! offloading to a remote peer that holds the mmproj shard.

use crate::error::SwarmError;
use crate::types::{NetworkCommand, SwarmMessage};

use super::{PipelineExecutor, VISION_ENCODE_TIMEOUT_SECS};

impl PipelineExecutor {
    /// T14: Pre-compute vision embeddings before the text pipeline.
    /// Encodes images locally if this node has mmproj, otherwise sends to a remote node.
    /// Returns zstd-compressed FP16 bytes, or None if no images / no encoder available.
    pub(super) async fn precompute_vision_embeddings(&self) -> Result<Option<Vec<u8>>, SwarmError> {
        let images: Vec<crate::types::ImageData> =
            crate::inference::vision::collect_images(&self.request.messages)
                .into_iter()
                .cloned()
                .collect();
        if images.is_empty() {
            return Ok(None);
        }

        let model_id = &self.request.model_id;

        // T15: Select vision node — prefer local > first-segment node > any holder
        let local_node_id = self.shared_state.identity.node_id().clone();
        let mmproj_holders = self.shared_state.model_registry.mmproj_holders(model_id);

        // Check if we have mmproj locally
        let has_local = mmproj_holders.contains(&local_node_id)
            || self.shared_state.vision_modules.contains_key(model_id);

        if has_local {
            // Encode locally
            let vision_module = if let Some(vm) = self.shared_state.vision_modules.get(model_id) {
                vm.value().clone()
            } else {
                let model_dir = crate::model::shard::model_dir(
                    &self.shared_state.config.node.data_dir,
                    &model_id.0,
                );
                let mmproj_path = model_dir.join(crate::model::shard::MMPROJ_FILENAME);
                let vm = crate::inference::vision::load_from_mmproj_gguf(
                    &mmproj_path,
                    &candle_core::Device::Cpu,
                )?;
                let vm = std::sync::Arc::new(vm);
                self.shared_state
                    .vision_modules
                    .insert(model_id.clone(), vm.clone());
                vm
            };

            let embeddings = tokio::task::block_in_place(|| vision_module.encode_images(&images))?;
            let compressed = self.compress_vision_embeddings(&embeddings)?;

            tracing::info!(
                request_id = %self.request.id,
                image_count = images.len(),
                compressed_bytes = compressed.len(),
                "DIAG: precompute_vision_embeddings local"
            );
            return Ok(Some(compressed));
        }

        // No local mmproj — try remote encoding
        if mmproj_holders.is_empty() {
            return Err(SwarmError::VisionEncoderUnavailable(model_id.clone()));
        }

        // Pick the best remote node: prefer first-segment node, then any
        let first_seg_node = self.assignment.segments.first().map(|s| &s.node_id);
        let remote_node = if let Some(first) = first_seg_node {
            if mmproj_holders.contains(first) {
                first.clone()
            } else {
                mmproj_holders[0].clone()
            }
        } else {
            mmproj_holders[0].clone()
        };

        tracing::info!(
            request_id = %self.request.id,
            remote_node = %remote_node,
            "DIAG: precompute_vision_embeddings remote"
        );

        // Remote vision encoding only supports single images — multi-image requires local mmproj
        if images.len() > 1 {
            return Err(SwarmError::Validation(
                "Multi-image VLM requires local mmproj — remote encoding only supports single images"
                    .into(),
            ));
        }

        // Compress image as JPEG for wire transfer
        let first_image = &images[0];
        let jpeg_bytes = self.compress_image_jpeg(first_image)?;

        // Register response channel with expected responder for auth verification.
        // Use entry().or_insert() to detect collision rather than silently
        // overwriting a live waiter — request_id reuse from a crashed/restarted
        // session that didn't unregister would otherwise leak the original
        // sender and the replaced entry's task would hang forever.
        let (tx, rx) = tokio::sync::oneshot::channel();
        match self
            .shared_state
            .pending_vision_results
            .entry(self.request.id)
        {
            dashmap::mapref::entry::Entry::Vacant(slot) => {
                slot.insert((remote_node.clone(), tx));
            }
            dashmap::mapref::entry::Entry::Occupied(_) => {
                return Err(SwarmError::Internal(format!(
                    "duplicate vision request_id {} — refusing to overwrite live waiter",
                    self.request.id
                )));
            }
        }

        // Send VisionEncodeRequest directly to the selected remote node (not broadcast)
        let req = crate::types::VisionEncodeRequest {
            request_id: self.request.id,
            model_id: model_id.clone(),
            image_data: jpeg_bytes,
            sender_peer_bytes: None,
        };
        let target_peer_bytes = self
            .shared_state
            .peer_id_map
            .get(&remote_node)
            .map(|r| r.value().clone())
            .or_else(|| {
                self.shared_state
                    .peer_registry
                    .get(&remote_node)
                    .and_then(|p| p.peer_id_bytes.clone())
            })
            .ok_or_else(|| {
                self.shared_state
                    .pending_vision_results
                    .remove(&self.request.id);
                SwarmError::Network(format!("No peer_id_bytes for vision node {}", remote_node))
            })?;
        let msg = NetworkCommand::SendDirectMessage {
            target_peer_bytes,
            message: SwarmMessage::VisionEncodeRequest(req),
            delivery_request_id: None,
        };
        if let Err(e) = self.network_tx.send(msg).await {
            self.shared_state
                .pending_vision_results
                .remove(&self.request.id);
            return Err(SwarmError::Network(format!(
                "Failed to send VisionEncodeRequest: {e}"
            )));
        }

        // Wait for response with timeout
        let timeout = std::time::Duration::from_secs(VISION_ENCODE_TIMEOUT_SECS);
        match tokio::time::timeout(timeout, rx).await {
            Ok(Ok(resp)) => {
                self.shared_state
                    .pending_vision_results
                    .remove(&self.request.id);
                tracing::info!(
                    request_id = %self.request.id,
                    num_tokens = resp.num_tokens,
                    hidden_dim = resp.hidden_dim,
                    compressed_bytes = resp.embeddings.len(),
                    "Received VisionEncodeResponse from remote node"
                );
                Ok(Some(resp.embeddings))
            }
            Ok(Err(_)) => {
                self.shared_state
                    .pending_vision_results
                    .remove(&self.request.id);
                Err(SwarmError::Inference(
                    "Vision encode channel dropped".into(),
                ))
            }
            Err(_) => {
                self.shared_state
                    .pending_vision_results
                    .remove(&self.request.id);
                Err(SwarmError::InferenceTimeout(VISION_ENCODE_TIMEOUT_SECS))
            }
        }
    }

    /// Compress vision embeddings tensor to zstd-compressed FP16 bytes.
    fn compress_vision_embeddings(
        &self,
        embeddings: &candle_core::Tensor,
    ) -> Result<Vec<u8>, SwarmError> {
        let dims = embeddings.dims();
        let (num_tokens, hidden_dim) = if dims.len() == 2 {
            (dims[0] as u32, dims[1] as u32)
        } else {
            (1u32, embeddings.elem_count() as u32)
        };
        let fp16 = embeddings
            .to_dtype(candle_core::DType::F16)
            .map_err(|e| SwarmError::Inference(format!("FP16 conversion: {e}")))?;
        let data: Vec<half::f16> = fp16
            .flatten_all()
            .map_err(|e| SwarmError::Inference(format!("Flatten: {e}")))?
            .to_vec1()
            .map_err(|e| SwarmError::Inference(format!("to_vec1: {e}")))?;
        let raw_bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
        let compressed = zstd::encode_all(std::io::Cursor::new(&raw_bytes), 3)
            .map_err(|e| SwarmError::Inference(format!("zstd compress: {e}")))?;
        // Prepend 8-byte header: num_tokens (u32 LE) + hidden_dim (u32 LE)
        // so the worker can reconstruct the exact tensor shape without heuristics.
        let mut result = Vec::with_capacity(8 + compressed.len());
        result.extend_from_slice(&num_tokens.to_le_bytes());
        result.extend_from_slice(&hidden_dim.to_le_bytes());
        result.extend_from_slice(&compressed);
        Ok(result)
    }

    /// Compress an ImageData to JPEG for wire transfer.
    fn compress_image_jpeg(&self, img: &crate::types::ImageData) -> Result<Vec<u8>, SwarmError> {
        use image::ImageEncoder;
        let rgb_image = image::RgbImage::from_raw(img.width, img.height, img.rgb_bytes.clone())
            .ok_or_else(|| SwarmError::Inference("Invalid image dimensions".into()))?;
        let mut buf = Vec::new();
        let encoder = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut buf, 85);
        encoder
            .write_image(
                &rgb_image,
                img.width,
                img.height,
                image::ExtendedColorType::Rgb8,
            )
            .map_err(|e| SwarmError::Inference(format!("JPEG encode: {e}")))?;
        Ok(buf)
    }
}
