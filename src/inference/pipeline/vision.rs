//! Vision-embedding pre-computation for multimodal inference. Encodes
//! image inputs either locally (if an mmproj is loaded/available) or by
//! offloading to a remote peer that holds the mmproj shard.

use crate::error::SwarmError;
use crate::types::{NetworkCommand, SwarmMessage};

use super::{PipelineExecutor, VISION_ENCODE_TIMEOUT_SECS};

impl PipelineExecutor {
    /// The picture spans a forward starting at `index_pos` carries: all of
    /// them while any picture ends after `index_pos` — the prompt pass, and a
    /// stand-in's replay from position 0 — none for a forward past every
    /// picture, which reads in order anyway. The ONE answer for every send of
    /// this request, local, remote and stand-in.
    pub(super) fn image_spans_from(&self, index_pos: usize) -> Vec<crate::types::PromptSpan> {
        if self.image_spans.iter().any(|s| s.end as usize > index_pos) {
            self.image_spans.clone()
        } else {
            Vec::new()
        }
    }

    /// May an image prompt's first forward go to `node`: this node, or a peer
    /// advertising `features::IMAGES_AT_MARKERS` — one that tokenizes the
    /// prompt with its markers and puts each image's rows there, as this node
    /// counted it. An older peer prepends the rows and reads the markers as
    /// text, so every position after the prompt would be off.
    ///
    /// The same bit says a peer reads the picture spans (the `0x0F` trailer,
    /// #115 (2b)), so where a prompt carries them this is asked of EVERY
    /// segment and stand-in, not only the first: each one's layers apply them,
    /// and an older peer would read its part of the picture in order without
    /// a word.
    pub(super) fn may_send_image_prompt_to(&self, node: &crate::types::NodeId) -> bool {
        node == self.shared_state.identity.node_id()
            || self
                .shared_state
                .peer_advertises_feature(node, swarmllm_types::node::features::IMAGES_AT_MARKERS)
    }

    /// The positions each picture's rows take in `prompt`, for a family whose
    /// picture rows attend both ways (`vision::rows_attend_both_ways`, #115
    /// (2b)); empty for every other family and for a prompt with no picture.
    ///
    /// Counted from the tokenization the first segment builds its input from
    /// (`vision::encode_prompt_with_images`) and the rows the payload states,
    /// so the spans and the splice describe one prompt — the first segment
    /// checks that it does. Without a tokenizer here the rows are read in
    /// order, as before, and the log says so. `SWARMLLM_IMAGE_BOTH_WAYS=0` is
    /// the control arm: every picture read in order, as before #115 (2b).
    pub(super) fn image_spans_for(
        &self,
        prompt: &str,
        image_rows: usize,
    ) -> Vec<crate::types::PromptSpan> {
        static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let switched_off = *OFF.get_or_init(|| {
            std::env::var("SWARMLLM_IMAGE_BOTH_WAYS")
                .map(|v| v == "0" || v.eq_ignore_ascii_case("false"))
                .unwrap_or(false)
        });
        if image_rows == 0 || switched_off {
            return Vec::new();
        }
        let model_id = &self.request.model_id;
        let architecture = self
            .shared_state
            .gguf_meta_for(model_id)
            .map(|m| m.architecture.clone());
        if !crate::inference::vision::rows_attend_both_ways(architecture.as_deref()) {
            return Vec::new();
        }
        let Some(tokenizer) = self.shared_state.standalone_tokenizer(model_id) else {
            tracing::warn!(
                request_id = %self.request.id,
                model = %model_id,
                "no tokenizer for this model here, so its pictures are read in order \
                 rather than as a whole — the reply may describe them less well"
            );
            return Vec::new();
        };
        let ids = crate::inference::vision::encode_prompt_with_images(&tokenizer, prompt);
        let spans: Vec<crate::types::PromptSpan> =
            crate::inference::vision::image_spans(&ids, image_rows)
                .into_iter()
                .map(|s| crate::types::PromptSpan {
                    start: s.start as u32,
                    end: s.end as u32,
                })
                .collect();
        tracing::info!(
            request_id = %self.request.id,
            pictures = spans.len(),
            spans = ?spans.iter().map(|s| (s.start, s.end)).collect::<Vec<_>>(),
            "DIAG: picture rows attend both ways"
        );
        spans
    }

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
        // The image goes only where this request may go
        // (`pool::scope::request_scope`, the scope the plan was built under).
        // Until 2026-10-12 this picked ANY holder when the plan's first node
        // had no encoder, so a private-mode node could hand the user's image
        // to a computer outside its pool.
        let mmproj_holders = in_request_scope(
            mmproj_holders,
            crate::pool::scope::request_scope(&self.shared_state, model_id, self.request.keep)
                .as_ref(),
        );
        if mmproj_holders.is_empty() {
            if let Some(kept) =
                crate::pool::scope::kept_here(&self.shared_state, model_id, self.request.keep)
            {
                return Err(
                    kept.refusal(model_id.0.clone(), vec![crate::types::MMPROJ_SHARD_INDEX])
                );
            }
            return Err(SwarmError::PrivateModeUnavailable {
                model_id: model_id.0.clone(),
                missing_shards: vec![crate::types::MMPROJ_SHARD_INDEX],
            });
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

        // Wait for response with timeout.
        //
        // Watched, not merely bounded: two minutes is long enough for a client
        // to give up, and until this was added nothing noticed — the remote
        // went on encoding an image for a request that no longer existed. Same
        // rule as every other long wait in a request's life (`inference::
        // cancel`); the cleanup below is shared with the timeout arm because a
        // cancelled request must not leave a waiter behind either.
        let timeout = std::time::Duration::from_secs(VISION_ENCODE_TIMEOUT_SECS);
        let waited = crate::inference::cancel::unless_cancelled(
            async { Ok(tokio::time::timeout(timeout, rx).await) },
            self.request.cancel.as_ref(),
        )
        .await;
        let waited = match waited {
            Ok(w) => w,
            Err(e) => {
                self.shared_state
                    .pending_vision_results
                    .remove(&self.request.id);
                return Err(e);
            }
        };
        match waited {
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

/// The holders a request's image may be sent to: every one when nothing
/// restricts the request (`allowed` is `None`), else those inside the scope.
fn in_request_scope(
    holders: Vec<crate::types::NodeId>,
    allowed: Option<&std::collections::HashSet<crate::types::NodeId>>,
) -> Vec<crate::types::NodeId> {
    match allowed {
        None => holders,
        Some(scope) => holders.into_iter().filter(|n| scope.contains(n)).collect(),
    }
}

#[cfg(test)]
mod scope_tests {
    use super::*;
    use crate::types::NodeId;

    /// Private mode: a holder outside the pool never receives the image, and
    /// with no holder inside it the request is refused as private mode's,
    /// naming the encoder rather than a shard number.
    #[test]
    fn an_image_is_never_sent_outside_private_modes_scope() {
        let pool_member = NodeId([1u8; 32]);
        let stranger = NodeId([2u8; 32]);
        let scope: std::collections::HashSet<NodeId> = [pool_member.clone()].into();
        assert_eq!(
            in_request_scope(vec![stranger.clone(), pool_member.clone()], Some(&scope)),
            vec![pool_member]
        );
        assert!(in_request_scope(vec![stranger.clone()], Some(&scope)).is_empty());
        assert_eq!(
            in_request_scope(vec![stranger.clone()], None),
            vec![stranger]
        );

        let refusal = SwarmError::PrivateModeUnavailable {
            model_id: "llava".into(),
            missing_shards: vec![crate::types::MMPROJ_SHARD_INDEX],
        }
        .to_string();
        assert!(refusal.contains("its image encoder"), "{refusal}");
        assert!(!refusal.contains(&u32::MAX.to_string()), "{refusal}");
    }
}
