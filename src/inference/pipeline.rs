use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::mpsc;

use crate::daemon::SharedState;
use crate::error::SwarmError;
use crate::inference::chat_template;
use crate::inference::router::{InferenceOutput, StreamingTokenEvent, StreamingTokenTx};
use crate::types::{
    InferenceError, InferenceRequest, LayerForward, LayerResult, NetworkCommand,
    NetworkFinishReason, PipelineAssignment, PipelineSegment, SwarmMessage, TensorFormat,
};

/// Extract chat template, BOS, and EOS strings from a GGUF header file on disk.
/// Used by distributed-only nodes that have the probe but no loaded model.
pub fn template_from_header(
    header_path: &std::path::Path,
) -> Option<(Option<String>, String, String)> {
    use candle_core::quantized::gguf_file;

    let header_bytes = std::fs::read(header_path).ok()?;
    let mut cursor = std::io::Cursor::new(&header_bytes);
    let ct = gguf_file::Content::read(&mut cursor).ok()?;

    let chat_template = ct
        .metadata
        .get("tokenizer.chat_template")
        .and_then(|v| v.to_string().ok().cloned());

    let vocab: Vec<String> = ct
        .metadata
        .get("tokenizer.ggml.tokens")
        .and_then(|v| v.to_vec().ok())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.to_string().ok().cloned())
                .collect()
        })
        .unwrap_or_default();

    let bos_id = ct
        .metadata
        .get("tokenizer.ggml.bos_token_id")
        .and_then(|v| v.to_u32().ok())
        .unwrap_or(1) as usize;
    let eos_id = ct
        .metadata
        .get("tokenizer.ggml.eos_token_id")
        .and_then(|v| v.to_u32().ok())
        .unwrap_or(2) as usize;

    let bos = vocab.get(bos_id).cloned().unwrap_or_default();
    let eos = vocab.get(eos_id).cloned().unwrap_or_default();

    Some((chat_template, bos, eos))
}

/// Cached vocabulary and tokenizer state for lock-free token decoding during streaming.
/// Extracted once from the model under the mutex, then used for all subsequent decoding
/// without re-acquiring the lock.
struct CachedDecoder {
    vocab: Vec<String>,
    byte_decoder: HashMap<char, u8>,
    is_sentencepiece: bool,
    has_tokenizer: bool,
}

impl CachedDecoder {
    fn decode_tokens(&self, token_ids: &[u32]) -> String {
        if self.has_tokenizer {
            let mut bytes = Vec::new();
            for &id in token_ids {
                if let Some(token_str) = self.vocab.get(id as usize) {
                    bytes.extend(self.decode_token_bytes(token_str));
                }
            }
            String::from_utf8_lossy(&bytes).to_string()
        } else if !self.vocab.is_empty() {
            let mut raw = String::new();
            for &id in token_ids {
                if let Some(token_str) = self.vocab.get(id as usize) {
                    raw.push_str(token_str);
                } else {
                    raw.push_str(&format!("[{id}]"));
                }
            }
            decode_bpe_text(&raw)
        } else {
            token_ids
                .iter()
                .map(|id| format!("[{id}]"))
                .collect::<String>()
        }
    }

    fn decode_token_bytes(&self, token_str: &str) -> Vec<u8> {
        if self.is_sentencepiece {
            if token_str.starts_with("<0x") && token_str.ends_with('>') && token_str.len() == 6 {
                if let Ok(byte) = u8::from_str_radix(&token_str[3..5], 16) {
                    return vec![byte];
                }
            }
            if token_str.starts_with('<') && token_str.ends_with('>') {
                return vec![];
            }
            token_str.replace('\u{2581}', " ").into_bytes()
        } else {
            token_str
                .chars()
                .filter_map(|c| self.byte_decoder.get(&c).copied())
                .collect()
        }
    }
}

/// Timeout for a single layer forward pass across the network.
/// Used in full distributed execution (Phase 6+ with Cap'n Proto protocol).
const _LAYER_FORWARD_TIMEOUT: Duration = Duration::from_secs(30);

/// Maximum number of failover attempts per segment.
/// Used in full distributed execution (Phase 6+ with robust retry logic).
const _MAX_FAILOVER_ATTEMPTS: u32 = 2;

/// Executes a distributed inference pipeline across multiple nodes.
///
/// The pipeline is a sequence of segments, each assigned to a node.
/// Activation tensors are forwarded between nodes in order.
/// If a segment fails, the executor attempts failover to a standby node.
pub struct PipelineExecutor {
    shared_state: Arc<SharedState>,
    network_tx: mpsc::Sender<NetworkCommand>,
    request: InferenceRequest,
    assignment: PipelineAssignment,
}

impl PipelineExecutor {
    pub fn new(
        shared_state: Arc<SharedState>,
        network_tx: mpsc::Sender<NetworkCommand>,
        request: InferenceRequest,
        assignment: PipelineAssignment,
    ) -> Self {
        Self {
            shared_state,
            network_tx,
            request,
            assignment,
        }
    }

    /// T14: Pre-compute vision embeddings before the text pipeline.
    /// Encodes images locally if this node has mmproj, otherwise sends to a remote node.
    /// Returns zstd-compressed FP16 bytes, or None if no images / no encoder available.
    async fn precompute_vision_embeddings(&self) -> Result<Option<Vec<u8>>, SwarmError> {
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
                let model_dir = self
                    .shared_state
                    .config
                    .node
                    .data_dir
                    .join("models")
                    .join(&model_id.0);
                let mmproj_path = model_dir.join("mmproj.gguf");
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
                "Pre-computed vision embeddings locally"
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
            "Sending VisionEncodeRequest to remote mmproj holder"
        );

        // Compress image as JPEG for wire transfer
        let first_image = &images[0];
        let jpeg_bytes = self.compress_image_jpeg(first_image)?;

        // Register response channel
        let (tx, rx) = tokio::sync::oneshot::channel();
        self.shared_state
            .pending_vision_results
            .insert(self.request.id, tx);

        // Send VisionEncodeRequest
        let req = crate::types::VisionEncodeRequest {
            request_id: self.request.id,
            model_id: model_id.clone(),
            image_data: jpeg_bytes,
        };
        let msg = NetworkCommand::Broadcast(SwarmMessage::VisionEncodeRequest(req));
        self.network_tx
            .send(msg)
            .await
            .map_err(|e| SwarmError::Network(format!("Failed to send VisionEncodeRequest: {e}")))?;

        // Wait for response with timeout
        let timeout = std::time::Duration::from_secs(120);
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
            Ok(Err(_)) => Err(SwarmError::Inference(
                "Vision encode channel dropped".into(),
            )),
            Err(_) => {
                self.shared_state
                    .pending_vision_results
                    .remove(&self.request.id);
                Err(SwarmError::InferenceTimeout(120))
            }
        }
    }

    /// Compress vision embeddings tensor to zstd-compressed FP16 bytes.
    fn compress_vision_embeddings(
        &self,
        embeddings: &candle_core::Tensor,
    ) -> Result<Vec<u8>, SwarmError> {
        let fp16 = embeddings
            .to_dtype(candle_core::DType::F16)
            .map_err(|e| SwarmError::Inference(format!("FP16 conversion: {e}")))?;
        let data: Vec<half::f16> = fp16
            .flatten_all()
            .map_err(|e| SwarmError::Inference(format!("Flatten: {e}")))?
            .to_vec1()
            .map_err(|e| SwarmError::Inference(format!("to_vec1: {e}")))?;
        let raw_bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
        zstd::encode_all(std::io::Cursor::new(&raw_bytes), 3)
            .map_err(|e| SwarmError::Inference(format!("zstd compress: {e}")))
    }

    /// Decompress zstd-compressed FP16 vision embeddings back to a Tensor.
    fn decompress_vision_embeddings(
        &self,
        compressed: &[u8],
    ) -> Result<candle_core::Tensor, SwarmError> {
        let raw_bytes = zstd::decode_all(std::io::Cursor::new(compressed))
            .map_err(|e| SwarmError::Inference(format!("zstd decompress vision: {e}")))?;
        // Raw bytes are FP16 LE — convert to f16 values
        let num_f16 = raw_bytes.len() / 2;
        // We need to know the shape. For LLaVA: 577 tokens × 4096 hidden dim.
        // Infer hidden_dim from common sizes, fall back to sqrt-ish heuristic.
        let hidden_dim = if num_f16 % 4096 == 0 {
            4096
        } else if num_f16 % 2048 == 0 {
            2048
        } else if num_f16 % 1024 == 0 {
            1024
        } else {
            return Err(SwarmError::Inference(format!(
                "Cannot infer vision embedding shape from {} values",
                num_f16
            )));
        };
        let num_tokens = num_f16 / hidden_dim;
        // Convert FP16 LE bytes → f32 in a single pass (avoids intermediate Vec<f16>)
        let f32_values: Vec<f32> = raw_bytes
            .chunks_exact(2)
            .map(|b| half::f16::from_le_bytes([b[0], b[1]]).to_f32())
            .collect();
        candle_core::Tensor::from_vec(
            f32_values,
            &[num_tokens, hidden_dim],
            &candle_core::Device::Cpu,
        )
        .map_err(|e| SwarmError::Inference(format!("Vision tensor from vec: {e}")))
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

    /// Build chat prompt using the template from the loaded model (if available).
    async fn build_prompt(&self) -> String {
        // Try to get template from the specific model's GGUF header (not the singleton loaded_model_info
        // which may hold a different model's info).
        let model_id = &self.request.model_id;
        let shard_store =
            crate::model::shard::ShardStore::new(&self.shared_state.config.node.data_dir);
        let header_path = shard_store
            .models_dir()
            .join(&model_id.0)
            .join("gguf_header.bin");
        if let Some((tmpl, bos, eos)) = template_from_header(&header_path) {
            let prompt = chat_template::build_prompt_with_model(
                &self.request.messages,
                tmpl.as_deref(),
                &bos,
                &eos,
                Some(&model_id.0),
            );
            tracing::info!(
                model = %model_id,
                prompt_len = prompt.len(),
                prompt_preview = %&prompt[..prompt.len().min(200)],
                "DIAG: build_prompt from header"
            );
            return prompt;
        }
        // Fall back to loaded_model_info (singleton, may be wrong model)
        let info = self.shared_state.loaded_model_info.read().await;
        match info.as_ref() {
            Some(i) => chat_template::build_prompt_with_model(
                &self.request.messages,
                i.chat_template.as_deref(),
                &i.bos_token,
                &i.eos_token,
                Some(&model_id.0),
            ),
            None => chat_template::chatml_fallback(&self.request.messages),
        }
    }

    /// Execute the pipeline end-to-end.
    ///
    /// For each token:
    /// 1. Send initial prompt activations to the first segment
    /// 2. Each segment processes its layers and forwards to the next
    /// 3. The last segment samples a token and returns a LayerResult
    /// 4. Repeat until stop condition or max_tokens
    ///
    /// If `token_tx` is provided, tokens are sent incrementally for SSE streaming.
    pub async fn execute(
        &mut self,
        token_tx: Option<StreamingTokenTx>,
    ) -> Result<InferenceOutput, SwarmError> {
        let request_id = self.request.id;
        let num_segments = self.assignment.segments.len();

        tracing::info!(
            request_id = %request_id,
            segments = num_segments,
            "Starting pipeline execution"
        );

        if num_segments == 0 {
            return Err(SwarmError::PipelineError(
                "Pipeline has no segments".to_string(),
            ));
        }

        // Check if this is a single-node pipeline on the local node
        // AND we have the llama-cpp executor loaded (full GGUF path).
        // If the model was loaded from shards (auto-manage), the executor
        // won't be loaded — fall through to execute_distributed which uses
        // the split model via process_local_segment.
        // Use the atomic flag to avoid locking the executor mutex just to check.
        if num_segments == 1
            && self.assignment.segments[0].node_id == *self.shared_state.identity.node_id()
            && self
                .shared_state
                .model_loaded
                .load(std::sync::atomic::Ordering::Acquire)
        {
            return self.execute_local().await;
        }

        // Distributed execution path (also handles single-node split-model execution)
        self.execute_distributed(token_tx).await
    }

    /// Execute entirely on the local node (we have all layers).
    ///
    /// If speculative decoding is enabled and a draft model is loaded,
    /// uses the draft-verify-accept loop for higher throughput.
    async fn execute_local(&self) -> Result<InferenceOutput, SwarmError> {
        let prompt = self.build_prompt().await;

        // Check if speculative decoding is available
        if self.shared_state.config.inference.speculative_decoding {
            let mut draft = self.shared_state.draft_executor.lock().await;
            if draft.is_loaded() {
                let gamma = self.shared_state.config.inference.speculative_gamma;
                let mut executor = self.shared_state.executor.lock().await;
                if !executor.is_loaded() {
                    return Err(SwarmError::NoModelLoaded);
                }
                let mut content = String::new();
                let (gen_result, spec_state) = executor.generate_speculative(
                    &mut draft,
                    &prompt,
                    &self.request.sampling_params,
                    gamma,
                    |token| {
                        content.push_str(token);
                        true
                    },
                )?;
                tracing::info!(
                    acceptance_rate = %spec_state.acceptance_rate(),
                    "Speculative decoding acceptance rate"
                );
                return Ok(InferenceOutput {
                    request_id: self.request.id,
                    content,
                    prompt_tokens: gen_result.prompt_tokens,
                    completion_tokens: gen_result.completion_tokens,
                    finish_reason: gen_result.finish_reason.as_str().to_string(),
                    session_id: self.request.session_id.clone(),
                });
            }
        }

        // Standard (non-speculative) local inference
        let mut executor = self.shared_state.executor.lock().await;
        if !executor.is_loaded() {
            return Err(SwarmError::NoModelLoaded);
        }
        let (content, gen_result) = executor.generate(&prompt, &self.request.sampling_params)?;

        Ok(InferenceOutput {
            request_id: self.request.id,
            content,
            prompt_tokens: gen_result.prompt_tokens,
            completion_tokens: gen_result.completion_tokens,
            finish_reason: gen_result.finish_reason.as_str().to_string(),
            session_id: self.request.session_id.clone(),
        })
    }

    /// Execute across multiple network nodes.
    ///
    /// In this phase, we implement the protocol for forwarding activations:
    /// 1. Build initial activation tensor from the prompt
    /// 2. Send LayerForward to each segment in sequence
    /// 3. Wait for the result from the last segment
    /// 4. Collect tokens until finish condition
    ///
    /// If `token_tx` is provided, each decoded token is sent on the channel
    /// as it arrives, enabling true SSE streaming for distributed inference.
    async fn execute_distributed(
        &mut self,
        token_tx: Option<StreamingTokenTx>,
    ) -> Result<InferenceOutput, SwarmError> {
        let request_id = self.request.id;
        let max_tokens = self.request.sampling_params.max_tokens;

        // Build the initial prompt representation
        let prompt = self.build_prompt().await;
        let prompt_bytes = prompt.as_bytes().to_vec();

        let mut generated_tokens: Vec<u32> = Vec::new();
        let mut finish_reason = String::new();

        // Cumulative position for RoPE / KV-cache
        let mut index_pos: usize = 0;
        // Will be set after the first forward pass (once the split model is loaded with tokenizer)
        let mut prompt_token_count: Option<usize> = None;

        // Cached EOS tokens and decoder — extracted once after prefill under a single
        // model lock acquisition. Avoids per-token mutex + DashMap scan.
        let mut cached_eos: Option<Vec<u32>> = None;
        let mut cached_decoder: Option<CachedDecoder> = None;
        let is_streaming = token_tx.is_some();
        // For streaming: accumulate decoded text to avoid redundant final decode
        let mut streamed_text = if is_streaming {
            Some(String::new())
        } else {
            None
        };
        // Text-based stop sequences from the chat template (e.g. "<|user|>")
        let stop_strings = {
            let info = self.shared_state.loaded_model_info.read().await;
            let tmpl = info.as_ref().and_then(|i| i.chat_template.as_deref());
            chat_template::extract_stop_strings(tmpl)
        };
        // Accumulate decoded text for stop-string matching (both streaming and non-streaming)
        let mut accumulated_text = String::new();

        // T14: Pre-compute vision embeddings before the token generation loop.
        // This decouples vision encoding from the text pipeline — any node with
        // mmproj can encode, and the embeddings travel with LayerForward.
        let precomputed_vision: Option<Vec<u8>> =
            if !crate::inference::vision::collect_images(&self.request.messages).is_empty() {
                match self.precompute_vision_embeddings().await {
                    Ok(Some(bytes)) => Some(bytes),
                    Ok(None) => None,
                    Err(e) => {
                        tracing::warn!(
                            request_id = %request_id,
                            error = %e,
                            "Vision pre-computation failed, proceeding without images"
                        );
                        None
                    }
                }
            } else {
                None
            };

        // Token generation loop
        let mut prompt_bytes_opt = Some(prompt_bytes);
        for seq_num in 0..max_tokens {
            let activations = if seq_num == 0 {
                prompt_bytes_opt.take().unwrap()
            } else {
                // For subsequent tokens, encode the last generated token ID as i64 LE bytes
                // so the first segment can embed it directly.
                let last_token = generated_tokens.last().copied().unwrap_or(0) as i64;
                last_token.to_le_bytes().to_vec()
            };

            tracing::info!(
                request_id = %request_id,
                seq_num,
                index_pos,
                activation_bytes = activations.len(),
                generated_so_far = generated_tokens.len(),
                "DIAG: starting forward_through_segments"
            );

            // Forward through each segment
            let fwd_start = std::time::Instant::now();
            // Attach pre-computed vision on first forward only
            let vision_for_forward = if seq_num == 0 {
                precomputed_vision.clone()
            } else {
                None
            };
            match self
                .forward_through_segments(
                    request_id,
                    seq_num,
                    index_pos,
                    activations,
                    vision_for_forward,
                )
                .await
            {
                Ok(result) => {
                    tracing::info!(
                        request_id = %request_id,
                        seq_num,
                        fwd_ms = fwd_start.elapsed().as_millis() as u64,
                        tokens = result.token_ids.len(),
                        activations_bytes = result.activations.len(),
                        finish = ?result.finish_reason,
                        "DIAG: forward_through_segments returned OK"
                    );
                    // After the first forward pass, extract everything we need from the model
                    // in a SINGLE lock acquisition: prompt token count, EOS tokens, and
                    // cached decoder for lock-free per-token decoding.
                    if seq_num == 0 {
                        let (ptc, eos, decoder) = self.extract_model_cache(&prompt).await;
                        // For VLM: the <image> token (1 tok) was replaced by N vision
                        // tokens per image. The vision module produces
                        // (image_size/patch_size)^2 + 1 tokens per image. Look up the
                        // actual count from the cached vision module if available.
                        let has_images =
                            crate::inference::vision::has_images(&self.request.messages);
                        let vision_expand = if has_images {
                            let model_id = &self.assignment.segments[0].shard_id.model_id;
                            self.shared_state
                                .vision_modules
                                .get(model_id)
                                .map(|vm| {
                                    let num_patches = vm.value().num_image_tokens();
                                    let num_images: usize =
                                        self.request.messages.iter().map(|m| m.images.len()).sum();
                                    // Each <image> token (1) is replaced by num_patches tokens
                                    num_patches * num_images - num_images
                                })
                                .unwrap_or(0)
                        } else {
                            0
                        };
                        index_pos = ptc + vision_expand;
                        prompt_token_count = Some(ptc + vision_expand);
                        cached_eos = Some(eos);
                        cached_decoder = Some(decoder);
                    } else {
                        index_pos += 1;
                    }

                    generated_tokens.extend(&result.token_ids);

                    // Decode and stream each non-EOS token, checking for stop strings.
                    let eos = cached_eos.as_deref().unwrap_or(&[2]);
                    let decoder = cached_decoder.as_ref();
                    let mut hit_stop_string = false;
                    for &tid in &result.token_ids {
                        if !eos.contains(&tid) {
                            let text = match decoder {
                                Some(d) => d.decode_tokens(&[tid]),
                                None => format!("[{tid}]"),
                            };
                            accumulated_text.push_str(&text);

                            // Check if accumulated text contains a stop string
                            if let Some(stop) = stop_strings
                                .iter()
                                .find(|s| accumulated_text.contains(s.as_str()))
                            {
                                // Trim everything from the stop string onwards
                                if let Some(pos) = accumulated_text.find(stop.as_str()) {
                                    let trimmed = accumulated_text[pos..].to_string();
                                    accumulated_text.truncate(pos);
                                    if let Some(ref mut st) = streamed_text {
                                        // Remove the stop string from streamed text too
                                        if let Some(spos) = st.rfind(stop.as_str()) {
                                            st.truncate(spos);
                                        } else if st.len() >= trimmed.len() {
                                            st.truncate(st.len() - trimmed.len());
                                        }
                                    }
                                }
                                hit_stop_string = true;
                                break;
                            }

                            if let Some(ref tx) = token_tx {
                                if let Some(ref mut st) = streamed_text {
                                    st.push_str(&text);
                                }
                                if tx
                                    .send(StreamingTokenEvent {
                                        text,
                                        finish_reason: None,
                                    })
                                    .await
                                    .is_err()
                                {
                                    // Client disconnected — stop generating tokens
                                    tracing::info!(
                                        request_id = %request_id,
                                        seq_num,
                                        "Streaming client disconnected — stopping generation"
                                    );
                                    finish_reason = "stop".to_string();
                                    break;
                                }
                            }
                        }
                    }

                    // Client disconnect already set finish_reason — break outer loop
                    if !finish_reason.is_empty() {
                        break;
                    }

                    if hit_stop_string {
                        finish_reason = "stop".to_string();
                        if let Some(ref tx) = token_tx {
                            let _ = tx
                                .send(StreamingTokenEvent {
                                    text: String::new(),
                                    finish_reason: Some("stop".to_string()),
                                })
                                .await;
                        }
                        break;
                    }

                    if let Some(reason) = result.finish_reason {
                        finish_reason = match reason {
                            NetworkFinishReason::Stop => "stop".to_string(),
                            NetworkFinishReason::MaxTokens => "length".to_string(),
                            NetworkFinishReason::Error(e) => {
                                return Err(SwarmError::Inference(e));
                            }
                        };
                        // Send finish event on streaming channel
                        if let Some(ref tx) = token_tx {
                            let _ = tx
                                .send(StreamingTokenEvent {
                                    text: String::new(),
                                    finish_reason: Some(finish_reason.clone()),
                                })
                                .await;
                        }
                        break;
                    }
                }
                Err(e) => {
                    tracing::warn!(
                        request_id = %request_id,
                        error = %e,
                        seq_num,
                        "Pipeline segment failed, attempting failover"
                    );
                    return Err(e);
                }
            }
        }

        // If we ran out of tokens without a stop signal
        if generated_tokens.len() as u32 >= max_tokens {
            finish_reason = "length".to_string();
            if let Some(ref tx) = token_tx {
                let _ = tx
                    .send(StreamingTokenEvent {
                        text: String::new(),
                        finish_reason: Some("length".to_string()),
                    })
                    .await;
            }
        }

        // Strip EOS tokens before decoding (loaded from GGUF metadata)
        let eos_tokens = cached_eos.unwrap_or_else(|| vec![2]);
        let clean_tokens: Vec<u32> = generated_tokens
            .iter()
            .copied()
            .filter(|t| !eos_tokens.contains(t))
            .collect();

        // For streaming: use already-decoded text. For non-streaming: use accumulated_text
        // (which has stop strings already trimmed), falling back to full decode.
        let generated_text = if let Some(text) = streamed_text {
            text
        } else if !accumulated_text.is_empty() {
            accumulated_text
        } else {
            match cached_decoder.as_ref() {
                Some(d) => d.decode_tokens(&clean_tokens),
                None => self.decode_tokens(&clean_tokens).await,
            }
        };

        // Batch credit write — one DB persist for the entire request instead of per-token.
        // The CreditLedger task also persists periodically, so this is durable enough.
        let total_tokens = generated_tokens.len() as i64;
        if total_tokens > 0 {
            let local_layers: i64 = self
                .assignment
                .segments
                .iter()
                .filter(|s| s.node_id == *self.shared_state.identity.node_id())
                .map(|s| (s.layer_range.1 - s.layer_range.0) as i64)
                .sum();
            let total_earned =
                crate::credit::ledger::RATE_INFERENCE_SERVE * local_layers * total_tokens;
            if total_earned > 0 {
                if let Err(e) = crate::credit::ledger::apply_credit_direct(
                    &self.shared_state.credit_balance,
                    &self.shared_state.db,
                    total_earned,
                    true,
                )
                .await
                {
                    tracing::warn!(error = %e, "Failed to persist batched credit earn");
                }
            }
        }

        Ok(InferenceOutput {
            request_id,
            content: generated_text,
            prompt_tokens: prompt_token_count.unwrap_or_else(|| prompt.chars().count() / 4) as u32,
            completion_tokens: clean_tokens.len() as u32,
            finish_reason: if finish_reason.is_empty() {
                "stop".to_string()
            } else {
                finish_reason
            },
            session_id: self.request.session_id.clone(),
        })
    }

    /// Decode token IDs to text using the GGUF vocabulary from the split model.
    async fn decode_tokens(&self, token_ids: &[u32]) -> String {
        let model_id = &self.assignment.segments[0].shard_id.model_id;
        let model_arc = self
            .shared_state
            .split_models
            .iter()
            .find(|e| e.key().0 == *model_id)
            .map(|e| e.value().model.clone());
        if let Some(model_arc) = model_arc {
            let model = model_arc.lock().await;
            if let Some(vocab) = model.vocab() {
                // If we have a BPE tokenizer, use its byte decoder for proper decoding
                if let Some(tokenizer) = model.tokenizer() {
                    let mut bytes = Vec::new();
                    for &id in token_ids {
                        if let Some(token_str) = vocab.get(id as usize) {
                            bytes.extend(tokenizer.decode_token(token_str));
                        }
                    }
                    return String::from_utf8_lossy(&bytes).to_string();
                }
                // Fallback: raw vocab concatenation with GPT-2 byte decode
                let mut raw = String::new();
                for &id in token_ids {
                    if let Some(token_str) = vocab.get(id as usize) {
                        raw.push_str(token_str);
                    } else {
                        raw.push_str(&format!("[{id}]"));
                    }
                }
                return decode_bpe_text(&raw);
            }
        }
        // Last fallback: render token IDs
        token_ids
            .iter()
            .map(|id| format!("[{id}]"))
            .collect::<String>()
    }

    /// Extract prompt token count, EOS tokens, and a cached decoder in a SINGLE
    /// model lock acquisition. This replaces three separate calls to
    /// compute_prompt_token_count + get_eos_tokens + (per-token decode_tokens),
    /// each of which previously acquired the model mutex independently.
    async fn extract_model_cache(&self, prompt: &str) -> (usize, Vec<u32>, CachedDecoder) {
        let model_id = &self.assignment.segments[0].shard_id.model_id;
        let model_arc = self
            .shared_state
            .split_models
            .iter()
            .find(|e| e.key().0 == *model_id)
            .map(|e| e.value().model.clone());

        if let Some(model_arc) = model_arc {
            let model = model_arc.lock().await;

            // 1. Prompt token count (excluding <image> sub-tokens which get replaced)
            let image_placeholder = crate::inference::chat_template::IMAGE_PLACEHOLDER;
            let ptc = if let Some(tokenizer) = model.tokenizer() {
                if let Some(img_pos) = prompt.find(image_placeholder) {
                    // VLM: count tokens for before + after parts (excluding <image> text)
                    let before = &prompt[..img_pos];
                    let after = &prompt[img_pos + image_placeholder.len()..];
                    tokenizer.encode(before).len() + tokenizer.encode(after).len()
                } else {
                    tokenizer.encode(prompt).len()
                }
            } else {
                prompt.chars().count() / 4
            };

            // 2. EOS tokens
            let eos = model.eos_tokens().to_vec();

            // 3. Cached decoder — clone vocab + byte_decoder for lock-free decoding
            let decoder = if let Some(vocab) = model.vocab() {
                let (byte_decoder, is_sentencepiece, has_tokenizer) =
                    if let Some(tokenizer) = model.tokenizer() {
                        (tokenizer.byte_decoder(), tokenizer.is_sentencepiece(), true)
                    } else {
                        (HashMap::new(), false, false)
                    };
                CachedDecoder {
                    vocab: vocab.clone(),
                    byte_decoder,
                    is_sentencepiece,
                    has_tokenizer,
                }
            } else {
                CachedDecoder {
                    vocab: Vec::new(),
                    byte_decoder: HashMap::new(),
                    is_sentencepiece: false,
                    has_tokenizer: false,
                }
            };

            (ptc, eos, decoder)
        } else {
            // No model loaded — try loading vocab from GGUF header on disk.
            // The header is always available from the probe/manifest exchange.
            let header_path = self
                .shared_state
                .config
                .node
                .data_dir
                .join("models")
                .join(model_id.0.as_str())
                .join("gguf_header.bin");
            if header_path.exists() {
                match Self::decoder_from_header(&header_path) {
                    Some((eos, decoder, tokenizer_opt)) => {
                        let ptc = if let Some(ref tok) = tokenizer_opt {
                            tok.encode(prompt).len()
                        } else {
                            prompt.chars().count() / 4
                        };
                        tracing::debug!(
                            model = %model_id,
                            vocab_size = decoder.vocab.len(),
                            eos_count = eos.len(),
                            "Built decoder from GGUF header (no local model)"
                        );
                        (ptc, eos, decoder)
                    }
                    None => {
                        let ptc = prompt.chars().count() / 4;
                        (
                            ptc,
                            vec![2],
                            CachedDecoder {
                                vocab: Vec::new(),
                                byte_decoder: HashMap::new(),
                                is_sentencepiece: false,
                                has_tokenizer: false,
                            },
                        )
                    }
                }
            } else {
                // No header on disk — try fetching from HuggingFace on-demand
                if let Some(hf_source) = self.shared_state.hf_sources.get(model_id) {
                    let model_dir = self
                        .shared_state
                        .config
                        .node
                        .data_dir
                        .join("models")
                        .join(&model_id.0);
                    tracing::info!(
                        model = %model_id,
                        repo = %hf_source.repo_id,
                        "Fetching GGUF header from HuggingFace for remote model"
                    );
                    let probe_result = crate::model::huggingface::probe_gguf_file(
                        &hf_source.repo_id,
                        &hf_source.filename,
                        self.shared_state.config.model.shard_size_bytes(),
                    )
                    .await;
                    if let Ok(info) = probe_result {
                        if let Ok(path) = crate::model::huggingface::download_gguf_header(
                            &hf_source.repo_id,
                            &hf_source.filename,
                            &model_dir,
                            info.header_size,
                        )
                        .await
                        {
                            if let Some((eos, decoder, tokenizer_opt)) =
                                Self::decoder_from_header(&path)
                            {
                                let ptc = if let Some(ref tok) = tokenizer_opt {
                                    tok.encode(prompt).len()
                                } else {
                                    prompt.chars().count() / 4
                                };
                                return (ptc, eos, decoder);
                            }
                        }
                    }
                }
                let ptc = prompt.chars().count() / 4;
                (
                    ptc,
                    vec![2],
                    CachedDecoder {
                        vocab: Vec::new(),
                        byte_decoder: HashMap::new(),
                        is_sentencepiece: false,
                        has_tokenizer: false,
                    },
                )
            }
        }
    }

    /// Build a CachedDecoder + EOS tokens from a GGUF header file on disk.
    /// Used when the node has probe data but no loaded model.
    fn decoder_from_header(
        header_path: &std::path::Path,
    ) -> Option<(
        Vec<u32>,
        CachedDecoder,
        Option<crate::inference::split::SplitTokenizer>,
    )> {
        use candle_core::quantized::gguf_file;

        let header_bytes = std::fs::read(header_path).ok()?;
        let mut cursor = std::io::Cursor::new(&header_bytes);
        let ct = gguf_file::Content::read(&mut cursor).ok()?;

        // Extract vocabulary
        let vocab: Vec<String> = ct
            .metadata
            .get("tokenizer.ggml.tokens")
            .and_then(|v| v.to_vec().ok())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.to_string().ok().cloned())
                    .collect()
            })?;

        // Extract EOS tokens
        let mut eos_tokens = Vec::new();
        if let Some(eos_id) = ct
            .metadata
            .get("tokenizer.ggml.eos_token_id")
            .and_then(|v| v.to_u32().ok())
        {
            eos_tokens.push(eos_id);
        }

        // Build BPE tokenizer if merges available
        let merges_raw = ct
            .metadata
            .get("tokenizer.ggml.merges")
            .and_then(|v| v.to_vec().ok())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.to_string().ok().cloned())
                    .collect::<Vec<String>>()
            })
            .unwrap_or_default();
        let pre_type = ct
            .metadata
            .get("tokenizer.ggml.pre")
            .and_then(|v| v.to_string().ok().cloned())
            .unwrap_or_else(|| "gpt2".to_string());
        let tokenizer_model = ct
            .metadata
            .get("tokenizer.ggml.model")
            .and_then(|v| v.to_string().ok().cloned())
            .unwrap_or_else(|| "gpt2".to_string());

        let tokenizer = if !merges_raw.is_empty() {
            Some(crate::inference::split::SplitTokenizer::from_bpe(
                &vocab,
                &merges_raw,
                &pre_type,
                &tokenizer_model,
            ))
        } else if tokenizer_model == "llama" {
            // Sentencepiece model — build HF Unigram tokenizer from scores
            let scores: Vec<f32> = ct
                .metadata
                .get("tokenizer.ggml.scores")
                .and_then(|v| v.to_vec().ok())
                .map(|arr| arr.iter().filter_map(|v| v.to_f32().ok()).collect())
                .unwrap_or_default();
            if !scores.is_empty() {
                Some(crate::inference::split::SplitTokenizer::from_sentencepiece(
                    &vocab, &scores,
                ))
            } else {
                None
            }
        } else {
            None
        };

        let decoder = if let Some(ref tok) = tokenizer {
            CachedDecoder {
                vocab: vocab.clone(),
                byte_decoder: tok.byte_decoder(),
                is_sentencepiece: tok.is_sentencepiece(),
                has_tokenizer: true,
            }
        } else {
            CachedDecoder {
                vocab,
                byte_decoder: HashMap::new(),
                is_sentencepiece: false,
                has_tokenizer: false,
            }
        };

        Some((eos_tokens, decoder, tokenizer))
    }

    /// Forward activation data through all pipeline segments in order.
    ///
    /// If tensor-parallel groups are available for a segment's layer range,
    /// the executor uses layer-by-layer AllReduce across the TP group instead
    /// of sending the full layer range to a single node.
    async fn forward_through_segments(
        &mut self,
        request_id: uuid::Uuid,
        sequence_num: u32,
        index_pos: usize,
        initial_activations: Vec<u8>,
        precomputed_vision: Option<Vec<u8>>,
    ) -> Result<LayerResult, SwarmError> {
        let mut activations = initial_activations;
        let num_segments = self.assignment.segments.len();
        let pipeline_start = std::time::Instant::now();

        for idx in 0..num_segments {
            let is_last = idx == num_segments - 1;
            let segment = &self.assignment.segments[idx];

            // Check if this segment has a tensor-parallel group
            let tp_group = self
                .assignment
                .tp_groups
                .iter()
                .find(|g| {
                    g.layer_range.0 <= segment.layer_range.0
                        && g.layer_range.1 >= segment.layer_range.1
                })
                .cloned();

            if let Some(ref group) = tp_group {
                // Tensor-parallel execution: layer-by-layer with AllReduce
                activations = self
                    .execute_tp_segment(
                        request_id,
                        sequence_num,
                        index_pos,
                        &activations,
                        segment,
                        group,
                        is_last,
                    )
                    .await?;
                if is_last {
                    // For the last segment, activations contains the serialized LayerResult
                    // We already handled sampling in execute_tp_segment
                    // Return a synthetic result
                    return Ok(LayerResult {
                        request_id,
                        token_ids: vec![], // filled by execute_tp_segment
                        finish_reason: None,
                        activations,
                    });
                }
                continue;
            }

            // Standard pipeline execution (no TP)
            let segment_start = std::time::Instant::now();
            // If this is the local node, process locally (no clone needed)
            if segment.node_id == *self.shared_state.identity.node_id() {
                let result = self
                    .process_local_segment(
                        segment,
                        sequence_num,
                        index_pos,
                        &activations,
                        if idx == 0 {
                            precomputed_vision.as_deref()
                        } else {
                            None
                        },
                    )
                    .await?;
                tracing::debug!(
                    request_id = %request_id,
                    segment = idx,
                    segment_ms = segment_start.elapsed().as_millis() as u64,
                    activation_bytes = result.activations.len(),
                    "DIAG: local segment complete"
                );
                if is_last {
                    tracing::info!(
                        request_id = %request_id,
                        num_segments,
                        pipeline_ms = pipeline_start.elapsed().as_millis() as u64,
                        "DIAG: forward_through_segments completed (last segment local)"
                    );
                    return Ok(result);
                }
                // Use hidden-state activations for the next segment
                activations = result.activations;
            } else {
                // Only clone activations when sending over the network
                // T17: Attach vision embeddings on first forward (seq_num==0, first segment)
                let vision_for_wire = if idx == 0 && sequence_num == 0 {
                    precomputed_vision.clone()
                } else {
                    None
                };
                let forward = LayerForward {
                    request_id,
                    sequence_num,
                    index_pos: index_pos as u32,
                    activations: activations.clone(),
                    format: TensorFormat::FP32,
                    model_id: segment.shard_id.model_id.clone(),
                    layer_range: segment.layer_range,
                    vision_embeddings: vision_for_wire,
                    sender_peer_bytes: None,
                    tp_meta: None,
                };

                // Look up the peer's libp2p PeerId bytes. Use peer_id_map (persistent,
                // survives disconnects) first, fall back to peer_registry.
                let target_peer_bytes = self
                    .shared_state
                    .peer_id_map
                    .get(&segment.node_id)
                    .map(|r| r.value().clone())
                    .or_else(|| {
                        self.shared_state
                            .peer_registry
                            .get(&segment.node_id)
                            .and_then(|p| p.peer_id_bytes.clone())
                    })
                    .ok_or_else(|| {
                        SwarmError::Network(format!(
                            "No peer_id_bytes for node {}",
                            segment.node_id
                        ))
                    })?;

                // Register the result channel BEFORE sending so we never miss
                // a fast response.
                let (tx, rx) = tokio::sync::oneshot::channel();
                self.shared_state
                    .pending_layer_results
                    .insert(request_id, tx);

                tracing::info!(
                    request_id = %request_id,
                    seq = sequence_num,
                    segment = idx,
                    node = %segment.node_id,
                    activation_bytes = activations.len(),
                    "Sending LayerForward to remote segment"
                );

                if self
                    .network_tx
                    .send(NetworkCommand::SendTensor {
                        target_peer_bytes: target_peer_bytes.clone(),
                        forward,
                    })
                    .await
                    .is_err()
                {
                    self.shared_state.pending_layer_results.remove(&request_id);
                    return Err(SwarmError::Network(
                        "Failed to send LayerForward".to_string(),
                    ));
                }

                let num_layers = segment.layer_range.1 - segment.layer_range.0;
                let result = Self::wait_for_result(
                    rx,
                    request_id,
                    idx,
                    &segment.node_id,
                    num_layers,
                    activations.len(),
                )
                .await;

                match result {
                    Ok(result) => {
                        // Check if the remote node returned an error — if so, failover
                        if let Some(NetworkFinishReason::Error(ref err_msg)) = result.finish_reason
                        {
                            tracing::warn!(
                                request_id = %request_id,
                                segment = idx,
                                node = %segment.node_id,
                                error = %err_msg,
                                "Remote segment returned error, attempting failover"
                            );
                            let failover_result = self
                                .failover_segment(
                                    idx,
                                    request_id,
                                    sequence_num,
                                    index_pos,
                                    &activations,
                                    is_last,
                                )
                                .await?;
                            if is_last {
                                return Ok(failover_result);
                            }
                            activations = failover_result.activations;
                        } else {
                            tracing::debug!(
                                request_id = %request_id,
                                segment = idx,
                                segment_ms = segment_start.elapsed().as_millis() as u64,
                                activation_bytes = result.activations.len(),
                                "DIAG: remote segment complete"
                            );
                            if is_last {
                                tracing::info!(
                                    request_id = %request_id,
                                    num_segments,
                                    pipeline_ms = pipeline_start.elapsed().as_millis() as u64,
                                    "DIAG: forward_through_segments completed (last segment remote)"
                                );
                                return Ok(result);
                            }
                            activations = result.activations;
                        }
                    }
                    Err(e) => {
                        // Timeout or channel drop — remove stale entry and failover
                        self.shared_state.pending_layer_results.remove(&request_id);
                        tracing::warn!(
                            request_id = %request_id,
                            segment = idx,
                            node = %segment.node_id,
                            error = %e,
                            seq = sequence_num,
                            segment_ms = segment_start.elapsed().as_millis() as u64,
                            "Remote segment timed out, attempting failover"
                        );
                        let failover_result = self
                            .failover_segment(
                                idx,
                                request_id,
                                sequence_num,
                                index_pos,
                                &activations,
                                is_last,
                            )
                            .await?;
                        if is_last {
                            return Ok(failover_result);
                        }
                        activations = failover_result.activations;
                    }
                }
            }
        }

        Err(SwarmError::PipelineError(
            "Pipeline completed without producing a result".to_string(),
        ))
    }

    /// Process a pipeline segment locally using the split inference engine.
    ///
    /// Loads the split model (layer range) from the local GGUF if not already cached,
    /// then runs the forward pass on the activation tensor.
    async fn process_local_segment(
        &self,
        segment: &PipelineSegment,
        sequence_num: u32,
        index_pos: usize,
        activation_bytes: &[u8],
        precomputed_vision_bytes: Option<&[u8]>,
    ) -> Result<LayerResult, SwarmError> {
        use crate::inference::split::{self, SplitModel};

        let model_id = &segment.shard_id.model_id;
        let (layer_start, layer_end) = (
            segment.layer_range.0 as usize,
            segment.layer_range.1 as usize,
        );

        // Ensure the split model is loaded for this model's layer range.
        // Note: concurrent requests may both enter this block and double-load;
        // the entry().or_insert() at the end ensures only one survives in the map.
        // This is acceptable since the discarded model is freed immediately.
        let split_key = (model_id.clone(), layer_start, layer_end);
        if !self.shared_state.split_models.contains_key(&split_key) {
            let shard_store =
                crate::model::shard::ShardStore::new(&self.shared_state.config.node.data_dir);
            let model_dir = shard_store.models_dir().join(&model_id.0);

            let manifest = self
                .shared_state
                .model_registry
                .get_manifest(model_id)
                .ok_or_else(|| SwarmError::Internal("No manifest for model".into()))?;
            let total_layers = manifest.num_layers as usize;

            // Determine is_first/is_last with shard-awareness
            let local_node_id = self.shared_state.identity.node_id().clone();
            let local_shards: Vec<u32> = manifest
                .shards
                .iter()
                .filter(|s| {
                    let sid = crate::types::ShardId {
                        model_id: model_id.clone(),
                        index: s.index,
                    };
                    self.shared_state
                        .model_registry
                        .shard_holders(&sid)
                        .contains(&local_node_id)
                })
                .map(|s| s.index)
                .collect();
            let has_shard_0 = local_shards.contains(&0);
            let last_shard_idx = manifest.shard_count.saturating_sub(1);
            let has_last_shard = local_shards.contains(&last_shard_idx);
            let is_first = layer_start == 0 && has_shard_0;
            let is_last = layer_end >= total_layers && has_last_shard;

            // Try loading: GGUF file → source_path → shard files
            let gguf_path = model_dir.join("model.gguf");
            let source_path_file = model_dir.join("source_path");

            let load_result = if gguf_path.exists() {
                tracing::info!(
                    model = %model_id,
                    layers = format!("[{layer_start}..{layer_end})"),
                    "Loading split model from GGUF"
                );
                SplitModel::load_from_gguf(&gguf_path, layer_start, layer_end, is_first, is_last)
            } else if source_path_file.exists() {
                let p = std::fs::read_to_string(&source_path_file).map_err(SwarmError::Io)?;
                let path = std::path::PathBuf::from(p.trim());
                tracing::info!(
                    model = %model_id,
                    layers = format!("[{layer_start}..{layer_end})"),
                    "Loading split model from source_path"
                );
                SplitModel::load_from_gguf(&path, layer_start, layer_end, is_first, is_last)
            } else {
                // Shard-only loading path
                let params = crate::daemon::ShardLoadParams {
                    model_dir: &model_dir,
                    shard_store: &shard_store,
                    model_id,
                    layer_start,
                    layer_end,
                    is_first,
                    is_last,
                    manifest: &manifest,
                };
                tracing::info!(
                    model = %model_id,
                    layers = format!("[{layer_start}..{layer_end})"),
                    "Loading split model from shard files"
                );
                crate::daemon::try_load_from_shards(&params)
            };

            let split_model = load_result?;
            // VRAM-aware eviction before inserting new model
            let max_batch = self.shared_state.config.inference.max_batch_size as usize;
            let batch_timeout = std::time::Duration::from_millis(
                self.shared_state.config.inference.batch_timeout_ms,
            );
            let new_entry = if max_batch > 1 {
                crate::inference::split::SplitModelEntry::new_with_batching(
                    split_model,
                    self.shared_state.kv_cache_store.clone(),
                    max_batch,
                    batch_timeout,
                )
            } else {
                crate::inference::split::SplitModelEntry::new(split_model)
            };
            let vram_budget = crate::model::auto_manage::compute_vram_budget(&self.shared_state)
                .or(self.shared_state.config.inference.max_split_model_memory_mb);
            if let Some(budget_mb) = vram_budget {
                crate::inference::split::evict_split_models_lru(
                    &self.shared_state.split_models,
                    &self.shared_state.active_pipelines,
                    budget_mb,
                    new_entry.estimated_vram_mb,
                );
            }
            // Re-check before inserting to handle concurrent loaders
            self.shared_state
                .split_models
                .entry(split_key.clone())
                .or_insert(new_entry);
        }

        // Get model entry and extract what we need
        let (split_model_ref, batch_forwarder, cached_eos_tokens) = {
            let entry = self
                .shared_state
                .split_models
                .get(&split_key)
                .ok_or_else(|| SwarmError::Internal("Split model not found after load".into()))?;
            entry.value().touch();
            (
                entry.value().model.clone(),
                entry.value().batch_forwarder.clone(),
                entry.value().eos_tokens.clone(),
            )
        };

        let request_id_str = self.request.id.to_string();

        // Try batch path for decode steps (seq_num > 0) when batching is enabled
        // and this is NOT the first segment (which needs tokenization under the model lock).
        // LoRA adapters require per-request weight deltas, incompatible with batched MLP.
        let use_batch =
            batch_forwarder.is_some() && sequence_num > 0 && self.request.lora_adapter.is_none();

        if use_batch {
            let forwarder = batch_forwarder.unwrap();

            // Build input tensor without holding the model lock
            let input_tensor = if activation_bytes.len() == 8 {
                // First segment, decode step: single token ID as i64 LE
                let token_id = i64::from_le_bytes(activation_bytes[..8].try_into().unwrap());
                candle_core::Tensor::from_vec(vec![token_id], &[1, 1], &candle_core::Device::Cpu)
                    .map_err(|e| SwarmError::Internal(format!("Tensor creation failed: {e}")))?
            } else {
                // Non-first segment or hidden states
                split::bytes_to_tensor(activation_bytes)?
            };

            // Submit to batch forwarder — will be grouped with other concurrent requests
            let output = forwarder
                .submit(input_tensor, index_pos, request_id_str.clone())
                .await?;

            // Track stats (credit persistence is batched at end of request)
            if let Ok(mut stats) = self.shared_state.node_stats.try_write() {
                stats.forwards_served += 1;
            }

            // Determine is_last from segment info (no model lock needed)
            let is_last = {
                let manifest = self
                    .shared_state
                    .model_registry
                    .get_manifest(model_id)
                    .ok_or_else(|| SwarmError::Internal("No manifest for model".into()))?;
                layer_end >= manifest.num_layers as usize
            };

            if is_last {
                let token_id =
                    split::sample_token_with_params(&output, &self.request.sampling_params)?;
                let finish = if cached_eos_tokens.contains(&token_id) {
                    Some(NetworkFinishReason::Stop)
                } else {
                    None
                };
                return Ok(LayerResult {
                    request_id: self.request.id,
                    token_ids: vec![token_id],
                    finish_reason: finish,
                    activations: vec![],
                });
            } else {
                let activation_bytes = split::tensor_to_bytes(&output)?;
                return Ok(LayerResult {
                    request_id: self.request.id,
                    token_ids: vec![],
                    finish_reason: None,
                    activations: activation_bytes,
                });
            }
        }

        // Sequential path: prefill or batching disabled
        let mut split_model = split_model_ref.lock().await;

        // Clear per-request KV-cache at the start of a new request (prefill).
        if sequence_num == 0 {
            let model_key = format!(
                "{}-{}-{}",
                split_model.layer_start, split_model.layer_end, split_model.total_layers
            );
            self.shared_state
                .kv_cache_store
                .clear_request(&model_key, &request_id_str);
        }

        let is_first = split_model.layer_start == 0;
        let is_last = split_model.layer_end >= split_model.total_layers;

        // Convert activation bytes to a candle Tensor, with prefix cache support.
        // For prefill (sequence_num==0, is_first): try to reuse cached KV state
        // for the system prompt prefix, only computing the new (suffix) tokens.
        let (input_tensor, effective_index_pos) = if is_first {
            if sequence_num == 0 {
                // Prefill: activation_bytes contain the prompt text from execute_distributed.
                let prompt = String::from_utf8_lossy(activation_bytes);
                let image_placeholder = crate::inference::chat_template::IMAGE_PLACEHOLDER;
                let all_tokens: Vec<i64> = if let Some(tokenizer) = split_model.tokenizer() {
                    if let Some(img_pos) = prompt.find(image_placeholder) {
                        // VLM: split prompt at <image>, tokenize parts, insert -1 marker
                        let before = &prompt[..img_pos];
                        let after = &prompt[img_pos + image_placeholder.len()..];
                        let mut tokens = tokenizer.encode(before);
                        tokens.push(-1); // marker for vision embedding insertion
                        tokens.extend(tokenizer.encode(after));
                        tokens
                    } else {
                        tokenizer.encode(&prompt)
                    }
                } else {
                    prompt.bytes().map(|b| b as i64).collect()
                };

                // Try prefix caching for system prompt
                let model_key = format!(
                    "{}-{}-{}",
                    split_model.layer_start, split_model.layer_end, split_model.total_layers
                );
                let num_layers = split_model.num_layers();
                let prefix_cache_max = self.shared_state.config.inference.prefix_cache_max_entries;

                let prefix_result = if prefix_cache_max > 0 && split_model.tokenizer().is_some() {
                    self.try_prefix_lookup(&split_model, &all_tokens)
                } else {
                    None
                };

                if let Some((prefix_hash, prefix_len)) = prefix_result {
                    let mut cache_guard = self
                        .shared_state
                        .prefix_cache
                        .lock()
                        .unwrap_or_else(|e| e.into_inner());

                    if let Some((layer_kv, cached_prefix_len)) =
                        cache_guard.get(&prefix_hash, &model_key)
                    {
                        // HIT: pre-populate KV cache store with cached prefix KV
                        let prefix_len = cached_prefix_len;
                        let layer_kv_cloned: Vec<(candle_core::Tensor, candle_core::Tensor)> =
                            layer_kv
                                .iter()
                                .map(|(k, v)| (k.clone(), v.clone()))
                                .collect();
                        drop(cache_guard);

                        {
                            let mut store_entry = self.shared_state.kv_cache_store.get_or_create(
                                &model_key,
                                &request_id_str,
                                num_layers,
                            );
                            for (i, (cached_k, cached_v)) in layer_kv_cloned.iter().enumerate() {
                                let mut kv =
                                    candle_nn::kv_cache::KvCache::new(2, split_model.max_seq_len());
                                let _ = kv.append(cached_k, cached_v).map_err(|e| {
                                    SwarmError::Internal(format!("Prefix cache restore: {e}"))
                                })?;
                                store_entry.layers[i] = Some(kv);
                            }
                            store_entry.last_accessed = std::time::Instant::now();
                        }

                        tracing::info!(
                            request_id = %self.request.id,
                            prefix_len,
                            suffix_len = all_tokens.len() - prefix_len,
                            "Prefix cache HIT — skipping prefix prefill"
                        );

                        let suffix_tokens = all_tokens[prefix_len..].to_vec();
                        let tensor = candle_core::Tensor::from_vec(
                            suffix_tokens.clone(),
                            &[1, suffix_tokens.len()],
                            &candle_core::Device::Cpu,
                        )
                        .map_err(|e| SwarmError::Internal(format!("Tensor: {e}")))?;
                        (tensor, prefix_len)
                    } else {
                        drop(cache_guard);

                        // MISS: process prefix first, cache KV, then process suffix
                        let prefix_tokens = all_tokens[..prefix_len].to_vec();
                        let suffix_tokens = all_tokens[prefix_len..].to_vec();

                        let prefix_tensor = candle_core::Tensor::from_vec(
                            prefix_tokens,
                            &[1, prefix_len],
                            &candle_core::Device::Cpu,
                        )
                        .map_err(|e| SwarmError::Internal(format!("Tensor: {e}")))?;

                        // Forward prefix through model (populates KV cache).
                        // block_in_place: CPU-bound inference must not starve yamux.
                        let _prefix_out = tokio::task::block_in_place(|| {
                            split_model.forward(
                                &prefix_tensor,
                                0,
                                &self.shared_state.kv_cache_store,
                                &request_id_str,
                            )
                        })?;

                        // Extract and cache prefix KV state
                        {
                            let entry = self.shared_state.kv_cache_store.get_or_create(
                                &model_key,
                                &request_id_str,
                                num_layers,
                            );
                            let layer_kv: Vec<(candle_core::Tensor, candle_core::Tensor)> = entry
                                .layers
                                .iter()
                                .filter_map(|c: &Option<candle_nn::kv_cache::KvCache>| {
                                    c.as_ref().and_then(|cache: &candle_nn::kv_cache::KvCache| {
                                        let k = cache.k().ok()??;
                                        let v = cache.v().ok()??;
                                        Some((k.clone(), v.clone()))
                                    })
                                })
                                .collect();

                            if layer_kv.len() == num_layers {
                                let mut cache_guard = self
                                    .shared_state
                                    .prefix_cache
                                    .lock()
                                    .unwrap_or_else(|e| e.into_inner());
                                cache_guard.insert(
                                    prefix_hash,
                                    model_key.clone(),
                                    layer_kv,
                                    prefix_len,
                                );
                                tracing::info!(
                                    request_id = %self.request.id,
                                    prefix_len,
                                    suffix_len = suffix_tokens.len(),
                                    "Prefix cache MISS — cached for future reuse"
                                );
                            }
                        }

                        let tensor = candle_core::Tensor::from_vec(
                            suffix_tokens.clone(),
                            &[1, suffix_tokens.len()],
                            &candle_core::Device::Cpu,
                        )
                        .map_err(|e| SwarmError::Internal(format!("Tensor: {e}")))?;
                        (tensor, prefix_len)
                    }
                } else {
                    // No prefix-cacheable system prompt — normal path
                    let tensor = candle_core::Tensor::from_vec(
                        all_tokens.clone(),
                        &[1, all_tokens.len()],
                        &candle_core::Device::Cpu,
                    )
                    .map_err(|e| SwarmError::Internal(format!("Tensor creation failed: {e}")))?;
                    (tensor, index_pos)
                }
            } else {
                // Decode step: activation_bytes contains a single i64 token ID (8 bytes LE)
                let token_id = if activation_bytes.len() >= 8 {
                    i64::from_le_bytes(activation_bytes[..8].try_into().unwrap())
                } else {
                    0i64
                };
                let tensor = candle_core::Tensor::from_vec(
                    vec![token_id],
                    &[1, 1],
                    &candle_core::Device::Cpu,
                )
                .map_err(|e| SwarmError::Internal(format!("Tensor creation failed: {e}")))?;
                (tensor, index_pos)
            }
        } else {
            // Non-first segment: input is hidden states from previous segment
            (split::bytes_to_tensor(activation_bytes)?, index_pos)
        };

        // Look up LoRA adapter if requested
        let lora_adapter = self
            .request
            .lora_adapter
            .as_ref()
            .and_then(|id| self.shared_state.adapter_registry.get(id));

        // T16: VLM vision embeddings — use pre-computed if available, fall back to local encoding
        let all_images: Vec<&crate::types::ImageData> =
            crate::inference::vision::collect_images(&self.request.messages);
        let vision_embeddings = if is_first && sequence_num == 0 && !all_images.is_empty() {
            if let Some(compressed_bytes) = precomputed_vision_bytes {
                // Decompress pre-computed embeddings from wire format (zstd FP16)
                match self.decompress_vision_embeddings(compressed_bytes) {
                    Ok(tensor) => {
                        tracing::info!(
                            request_id = %self.request.id,
                            shape = ?tensor.dims(),
                            "Using pre-computed vision embeddings"
                        );
                        Some(tensor)
                    }
                    Err(e) => {
                        tracing::warn!(
                            request_id = %self.request.id,
                            error = %e,
                            "Failed to decompress pre-computed vision embeddings, falling back to local"
                        );
                        None
                    }
                }
            } else {
                // Fall back to local encoding (original behavior)
                let model_id = &segment.shard_id.model_id;
                let vision_mod = if let Some(vm) = self.shared_state.vision_modules.get(model_id) {
                    Some(vm.value().clone())
                } else {
                    let shard_store = crate::model::shard::ShardStore::new(
                        &self.shared_state.config.node.data_dir,
                    );
                    let model_dir = shard_store.models_dir().join(&model_id.0);
                    let mmproj_path = model_dir.join("mmproj.gguf");
                    if mmproj_path.exists() {
                        let device = split_model.device().clone();
                        match crate::inference::vision::load_from_mmproj_gguf(&mmproj_path, &device)
                        {
                            Ok(vm) => {
                                let vm = std::sync::Arc::new(vm);
                                self.shared_state
                                    .vision_modules
                                    .insert(model_id.clone(), vm.clone());
                                tracing::info!(model = %model_id, "Loaded VLM vision module from mmproj.gguf");
                                Some(vm)
                            }
                            Err(e) => {
                                tracing::warn!(model = %model_id, error = %e, "Failed to load mmproj.gguf");
                                None
                            }
                        }
                    } else {
                        None
                    }
                };
                if let Some(vm) = vision_mod {
                    let owned_images: Vec<crate::types::ImageData> =
                        all_images.iter().map(|img| (*img).clone()).collect();
                    let embeddings =
                        tokio::task::block_in_place(|| vm.encode_images(&owned_images))?;
                    tracing::info!(
                        request_id = %self.request.id,
                        image_count = owned_images.len(),
                        embedding_shape = ?embeddings.dims(),
                        "VLM: encoded images for multimodal forward (local fallback)"
                    );
                    Some(embeddings)
                } else {
                    tracing::warn!(
                        request_id = %self.request.id,
                        "Request has images but no VLM module available — ignoring images"
                    );
                    None
                }
            }
        } else {
            None
        };

        // Run the forward pass with per-request KV-cache isolation.
        // CRITICAL: block_in_place() tells Tokio to move other tasks off this thread
        // before the CPU-bound forward pass (~700-1000ms). Without this, the blocked
        // thread starves yamux, preventing outbound substream opens for tensor forwards.
        let output = tokio::task::block_in_place(|| {
            if let Some(ref vis_emb) = vision_embeddings {
                split_model.forward_multimodal(
                    &input_tensor,
                    effective_index_pos,
                    &self.shared_state.kv_cache_store,
                    &request_id_str,
                    Some(vis_emb),
                )
            } else {
                split_model.forward_with_lora(
                    &input_tensor,
                    effective_index_pos,
                    &self.shared_state.kv_cache_store,
                    &request_id_str,
                    lora_adapter.as_deref(),
                )
            }
        })?;

        // Track stats (credit persistence is batched at end of request)
        if let Ok(mut stats) = self.shared_state.node_stats.try_write() {
            stats.forwards_served += 1;
        }

        if is_last {
            // Last segment: output is logits → sample token
            // Debug: log top-5 logits for VLM diagnostics
            if vision_embeddings.is_some() || sequence_num == 0 {
                if let Ok(logits_1d) = output.flatten_all() {
                    let logits_vec: Vec<f32> = logits_1d.to_vec1().unwrap_or_default();
                    if logits_vec.len() > 10 {
                        let mut indexed: Vec<(usize, f32)> =
                            logits_vec.iter().copied().enumerate().collect();
                        indexed.sort_by(|a, b| {
                            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
                        });
                        let top5: Vec<String> = indexed
                            .iter()
                            .take(5)
                            .map(|(i, v)| format!("{i}:{v:.2}"))
                            .collect();
                        tracing::info!(
                            request_id = %self.request.id,
                            top5 = %top5.join(", "),
                            vocab_size = logits_vec.len(),
                            "DIAG: VLM top logits before sampling"
                        );
                    }
                }
            }
            let token_id = split::sample_token_with_params(&output, &self.request.sampling_params)?;

            // EOS detection: use tokens loaded from GGUF metadata
            let eos_tokens = split_model.eos_tokens();
            let finish = if eos_tokens.contains(&token_id) {
                Some(NetworkFinishReason::Stop)
            } else {
                None
            };

            Ok(LayerResult {
                request_id: self.request.id,
                token_ids: vec![token_id],
                finish_reason: finish,
                activations: vec![],
            })
        } else {
            // Intermediate segment: return hidden states for next segment
            let activation_bytes = split::tensor_to_bytes(&output)?;
            Ok(LayerResult {
                request_id: self.request.id,
                token_ids: vec![],
                finish_reason: None,
                activations: activation_bytes,
            })
        }
    }

    /// Try to identify a prefix-cacheable system prompt in the request.
    ///
    /// Returns `Some((blake3_hash, prefix_token_count))` if the request has system
    /// messages whose tokens align with the start of the full prompt tokens.
    fn try_prefix_lookup(
        &self,
        model: &crate::inference::split::SplitModel,
        all_tokens: &[i64],
    ) -> Option<([u8; 32], usize)> {
        let prefix_text =
            crate::inference::prefix_cache::build_system_prefix(&self.request.messages)?;

        let tokenizer = model.tokenizer()?;
        let prefix_tokens = tokenizer.encode(&prefix_text);
        let prefix_len = prefix_tokens.len();

        // Verify: prefix must be non-empty, shorter than full prompt, and tokens align
        if prefix_len == 0 || prefix_len >= all_tokens.len() {
            return None;
        }
        if all_tokens[..prefix_len] != prefix_tokens[..] {
            tracing::debug!(
                request_id = %self.request.id,
                "Prefix cache: token alignment mismatch, skipping"
            );
            return None;
        }

        let hash = crate::inference::prefix_cache::hash_token_ids(&prefix_tokens);
        Some((hash, prefix_len))
    }

    /// Compute a reasonable timeout for a remote segment based on workload.
    ///
    /// Prefill (large activation = many input tokens) is much slower than decode
    /// (single token). Budget 15s/layer for prefill, 2s/layer for decode, with
    /// a 30s floor and 600s ceiling.
    fn compute_segment_timeout(num_layers: u32, activation_bytes: usize) -> Duration {
        // Heuristic: activation > 100KB means prefill, otherwise decode
        let is_prefill = activation_bytes > 100_000;
        let per_layer_secs: u64 = if is_prefill { 15 } else { 2 };
        let base = (num_layers as u64) * per_layer_secs;
        let timeout = base.clamp(30, 600);
        Duration::from_secs(timeout)
    }

    /// Wait for a remote segment to return its result via the oneshot channel.
    async fn wait_for_result(
        rx: tokio::sync::oneshot::Receiver<LayerResult>,
        request_id: uuid::Uuid,
        segment_idx: usize,
        node_id: &crate::types::NodeId,
        num_layers: u32,
        activation_bytes: usize,
    ) -> Result<LayerResult, SwarmError> {
        let timeout = Self::compute_segment_timeout(num_layers, activation_bytes);
        let send_time = std::time::Instant::now();
        tracing::info!(
            request_id = %request_id,
            segment = segment_idx,
            node = %node_id,
            timeout_secs = timeout.as_secs(),
            num_layers,
            activation_bytes,
            is_prefill = activation_bytes > 100_000,
            "DIAG: waiting for remote segment result"
        );
        match tokio::time::timeout(timeout, rx).await {
            Ok(Ok(result)) => {
                let elapsed = send_time.elapsed();
                tracing::info!(
                    request_id = %request_id,
                    segment = segment_idx,
                    node = %node_id,
                    elapsed_ms = elapsed.as_millis() as u64,
                    tokens = result.token_ids.len(),
                    activations_bytes = result.activations.len(),
                    finish = ?result.finish_reason,
                    "DIAG: segment result received"
                );
                Ok(result)
            }
            Ok(Err(_)) => {
                let elapsed = send_time.elapsed();
                tracing::error!(
                    request_id = %request_id,
                    segment = segment_idx,
                    node = %node_id,
                    elapsed_ms = elapsed.as_millis() as u64,
                    "DIAG: response channel DROPPED — sender gone before result"
                );
                Err(SwarmError::PipelineError("Response channel dropped".into()))
            }
            Err(_) => {
                tracing::error!(
                    request_id = %request_id,
                    segment = segment_idx,
                    node = %node_id,
                    timeout_secs = timeout.as_secs(),
                    num_layers,
                    activation_bytes,
                    "DIAG: segment TIMED OUT — no result received"
                );
                Err(SwarmError::PipelineError(format!(
                    "Timed out waiting for segment result ({}s, {} layers)",
                    timeout.as_secs(),
                    num_layers
                )))
            }
        }
    }

    /// Attempt failover to a standby node for a failed segment.
    async fn failover_segment(
        &mut self,
        failed_idx: usize,
        request_id: uuid::Uuid,
        sequence_num: u32,
        index_pos: usize,
        activations: &[u8],
        _is_last: bool,
    ) -> Result<LayerResult, SwarmError> {
        let failed_segment = &self.assignment.segments[failed_idx];

        // Find a standby for this segment's layer range
        let standby = self
            .assignment
            .standbys
            .iter()
            .find(|s| {
                s.layer_range.0 <= failed_segment.layer_range.0
                    && s.layer_range.1 >= failed_segment.layer_range.1
                    && s.node_id != failed_segment.node_id
            })
            .cloned();

        match standby {
            Some(backup) => {
                tracing::warn!(
                    request_id = %request_id,
                    failed_node = %failed_segment.node_id,
                    backup_node = %backup.node_id,
                    failed_layer_range = ?failed_segment.layer_range,
                    backup_layer_range = ?backup.layer_range,
                    segment = failed_idx,
                    total_segments = self.assignment.segments.len(),
                    total_standbys = self.assignment.standbys.len(),
                    "DIAG: failing over to standby node"
                );

                // Register a response channel BEFORE sending the request
                let (tx, rx) = tokio::sync::oneshot::channel();
                self.shared_state
                    .pending_layer_results
                    .insert(request_id, tx);

                // Send to backup node via directed tensor protocol
                let forward = LayerForward {
                    request_id,
                    sequence_num,
                    index_pos: index_pos as u32,
                    activations: activations.to_vec(),
                    format: TensorFormat::FP32,
                    model_id: backup.shard_id.model_id.clone(),
                    layer_range: backup.layer_range,
                    tp_meta: None,
                    vision_embeddings: None,
                    sender_peer_bytes: None,
                };

                let target_peer_bytes = match self
                    .shared_state
                    .peer_registry
                    .get(&backup.node_id)
                    .and_then(|p| p.peer_id_bytes.clone())
                {
                    Some(b) => b,
                    None => {
                        self.shared_state.pending_layer_results.remove(&request_id);
                        return Err(SwarmError::Network(format!(
                            "No peer_id_bytes for backup node {}",
                            backup.node_id
                        )));
                    }
                };
                if self
                    .network_tx
                    .send(NetworkCommand::SendTensor {
                        target_peer_bytes,
                        forward,
                    })
                    .await
                    .is_err()
                {
                    self.shared_state.pending_layer_results.remove(&request_id);
                    return Err(SwarmError::Network(
                        "Failed to send to standby node".to_string(),
                    ));
                }

                // Wait for standby response via the oneshot channel
                let failed_segment = &self.assignment.segments[failed_idx];
                let num_layers = failed_segment.layer_range.1 - failed_segment.layer_range.0;
                let result = Self::wait_for_result(
                    rx,
                    request_id,
                    failed_idx,
                    &backup.node_id,
                    num_layers,
                    activations.len(),
                )
                .await?;

                // Update the assignment so subsequent tokens use the standby
                // directly, avoiding repeated failover + 30s timeout per token.
                self.assignment.segments[failed_idx].node_id = backup.node_id;
                self.assignment.segments[failed_idx].layer_range = backup.layer_range;

                Ok(result)
            }
            None => {
                tracing::error!(
                    request_id = %request_id,
                    failed_segment = failed_idx,
                    failed_node = %failed_segment.node_id,
                    failed_layer_range = ?failed_segment.layer_range,
                    total_standbys = self.assignment.standbys.len(),
                    standby_nodes = ?self.assignment.standbys.iter().map(|s| format!("{}[{:?}]", s.node_id, s.layer_range)).collect::<Vec<_>>(),
                    "DIAG: NO standby available for failed segment — pipeline will fail"
                );
                Err(SwarmError::PipelineError(format!(
                    "Segment {failed_idx} failed with no standby available"
                )))
            }
        }
    }

    /// Execute a pipeline segment using tensor parallelism across a TP group.
    ///
    /// Instead of sending the full layer range to one node, this executes
    /// layer-by-layer across all nodes in the TP group:
    ///
    /// For each layer:
    /// 1. Send activations + TP metadata to all TP nodes
    /// 2. Each node computes its fraction (head-parallel attn + column-parallel MLP)
    /// 3. Collect partial results from all nodes
    /// 4. AllReduce (sum) partial results + add residual
    /// 5. Use result as input for next layer
    ///
    /// The local node's TP computation is done inline; remote nodes receive
    /// LayerForward messages with TensorParallelMeta.
    #[allow(clippy::too_many_arguments)]
    async fn execute_tp_segment(
        &self,
        request_id: uuid::Uuid,
        sequence_num: u32,
        index_pos: usize,
        activation_bytes: &[u8],
        segment: &PipelineSegment,
        tp_group: &crate::types::TensorParallelGroup,
        _is_last: bool,
    ) -> Result<Vec<u8>, SwarmError> {
        use crate::inference::split;

        let model_id = &segment.shard_id.model_id;
        let (layer_start, layer_end) = (
            segment.layer_range.0 as usize,
            segment.layer_range.1 as usize,
        );
        let local_node_id = self.shared_state.identity.node_id().clone();
        let tp_size = tp_group.tp_size();

        // Find our rank in the TP group
        let local_tp_rank = tp_group.rank_of(&local_node_id);

        tracing::info!(
            request_id = %request_id,
            tp_size,
            local_rank = ?local_tp_rank,
            layers = ?(layer_start..layer_end),
            "Starting tensor-parallel segment execution"
        );

        // Ensure split model is loaded for this layer range.
        // Reuse the same loading logic as process_local_segment.
        let split_key = (model_id.clone(), layer_start, layer_end);
        if !self.shared_state.split_models.contains_key(&split_key) {
            let shard_store =
                crate::model::shard::ShardStore::new(&self.shared_state.config.node.data_dir);
            let model_dir = shard_store.models_dir().join(&model_id.0);
            let manifest = self
                .shared_state
                .model_registry
                .get_manifest(model_id)
                .ok_or_else(|| SwarmError::Internal("No manifest for model".into()))?;
            let total_layers = manifest.num_layers as usize;

            let local_shards: Vec<u32> = manifest
                .shards
                .iter()
                .filter(|s| {
                    let sid = crate::types::ShardId {
                        model_id: model_id.clone(),
                        index: s.index,
                    };
                    self.shared_state
                        .model_registry
                        .shard_holders(&sid)
                        .contains(&local_node_id)
                })
                .map(|s| s.index)
                .collect();
            let has_first_shard = local_shards.contains(&0);
            let last_idx = manifest.shard_count.saturating_sub(1);
            let has_last_shard = local_shards.contains(&last_idx);
            let is_first = layer_start == 0 && has_first_shard;
            let is_last_segment = layer_end >= total_layers && has_last_shard;

            // Load via the standard shard loading path
            let params = crate::daemon::ShardLoadParams {
                model_dir: &model_dir,
                shard_store: &shard_store,
                model_id,
                layer_start,
                layer_end,
                is_first,
                is_last: is_last_segment,
                manifest: &manifest,
            };
            let split_model = crate::daemon::try_load_from_shards(&params)?;
            let new_entry = split::SplitModelEntry::new(split_model);
            self.shared_state
                .split_models
                .entry(split_key.clone())
                .or_insert(new_entry);
        }

        // Get model reference
        let model_arc = self
            .shared_state
            .split_models
            .get(&split_key)
            .ok_or_else(|| SwarmError::Internal("Split model not loaded after insert".into()))?
            .model
            .clone();

        // Parse input activations
        let kv_cache_store = &self.shared_state.kv_cache_store;
        let req_id_str = request_id.to_string();

        // Build the initial hidden states tensor
        let mut current_activations = if sequence_num == 0 {
            // First token: input is the prompt (text bytes)
            let model = model_arc.lock().await;
            let prompt = String::from_utf8_lossy(activation_bytes);
            let input = model
                .tokenize_and_embed(&prompt)
                .map_err(|e| SwarmError::Internal(format!("Tokenize+embed: {e}")))?;
            input
        } else {
            // Subsequent tokens: input is last token ID as i64 LE bytes
            let model = model_arc.lock().await;
            let token_id = if activation_bytes.len() >= 8 {
                i64::from_le_bytes(activation_bytes[..8].try_into().unwrap()) as u32
            } else {
                0u32
            };
            model
                .embed_token(token_id)
                .map_err(|e| SwarmError::Internal(format!("Embed token: {e}")))?
        };

        // Layer-by-layer tensor-parallel execution
        for abs_layer in layer_start..layer_end {
            let residual = current_activations.clone();

            if let Some(tp_rank) = local_tp_rank {
                // We are in the TP group — compute our partial result
                let partial = {
                    let mut model = model_arc.lock().await;
                    model.forward_tp_layer(
                        &current_activations,
                        abs_layer,
                        index_pos,
                        tp_rank,
                        tp_size,
                        kv_cache_store,
                        &req_id_str,
                    )?
                };

                // Compress partial tensor for AllReduce
                let partial_bytes = split::tensor_to_bytes(&partial)
                    .map_err(|e| SwarmError::Internal(format!("Serialize TP partial: {e}")))?;
                let shape: Vec<u32> = partial.dims().iter().map(|&d| d as u32).collect();
                let compressed = zstd::encode_all(std::io::Cursor::new(&partial_bytes), 1)
                    .map_err(|e| SwarmError::Internal(format!("Compress TP partial: {e}")))?;

                // AllReduce: send partial to coordinator, wait for reduced result
                let resp = crate::inference::allreduce::allreduce_sum(
                    &self.shared_state,
                    &self.network_tx,
                    &self.shared_state.allreduce_registry,
                    request_id,
                    abs_layer as u32,
                    tp_group,
                    tp_rank,
                    compressed,
                    shape,
                )
                .await?;

                // Decompress reduced tensor
                let reduced_bytes = zstd::decode_all(std::io::Cursor::new(&resp.reduced_data))
                    .map_err(|e| SwarmError::Internal(format!("Decompress AllReduce: {e}")))?;
                let reduced = split::bytes_to_tensor(&reduced_bytes)
                    .map_err(|e| SwarmError::Internal(format!("Deserialize AllReduce: {e}")))?;

                // Add residual connection
                current_activations = (reduced + &residual)
                    .map_err(|e| SwarmError::Internal(format!("Residual add: {e}")))?;
            } else {
                // We're not in the TP group — run full (non-TP) layer
                let mut model = model_arc.lock().await;
                let result = tokio::task::block_in_place(|| {
                    model.forward(&current_activations, index_pos, kv_cache_store, &req_id_str)
                })?;
                current_activations = result;
                // Skip remaining layers since full forward processes all of them
                break;
            }
        }

        // Serialize the output
        let result_bytes = split::tensor_to_bytes(&current_activations)
            .map_err(|e| SwarmError::Internal(format!("Serialize TP output: {e}")))?;

        Ok(result_bytes)
    }
}

/// Decode BPE byte-level encoded text.
/// GPT-2/Qwen2 BPE uses Unicode characters to represent bytes:
/// - Ġ (U+0120) → space (0x20)
/// - Ċ (U+010A) → newline (0x0A)
/// - Other mapped bytes per GPT-2 byte encoder table
fn decode_bpe_text(text: &str) -> String {
    // GPT-2 byte encoder maps bytes 0-255 to Unicode chars.
    // The printable ASCII range (33-126) and some others map to themselves.
    // Others are shifted: byte 0x00 → U+0100 (Ā), 0x01 → U+0101 (ā), etc.
    // Space (0x20) → U+0120 (Ġ), newline (0x0A) → U+010A (Ċ), etc.
    let mut bytes = Vec::with_capacity(text.len());
    for ch in text.chars() {
        let cp = ch as u32;
        // Printable ASCII and some others map directly
        match cp {
            // Standard printable ASCII
            33..=126 | 161..=172 | 174..=255 => {
                bytes.push(cp as u8);
            }
            // GPT-2 mapped range: U+0100..U+01FF → bytes 0..255
            0x0100..=0x01FF => {
                // The GPT-2 byte encoder maps non-printable/special bytes to U+0100+offset
                // We need to reverse this mapping
                let byte_val = gpt2_unicode_to_byte(cp);
                bytes.push(byte_val);
            }
            _ => {
                // Fallback: try UTF-8 encoding of the character
                let mut buf = [0u8; 4];
                let s = ch.encode_utf8(&mut buf);
                bytes.extend_from_slice(s.as_bytes());
            }
        }
    }
    String::from_utf8_lossy(&bytes).to_string()
}

/// Reverse the GPT-2 byte-to-unicode mapping for a Unicode codepoint.
fn gpt2_unicode_to_byte(cp: u32) -> u8 {
    use std::sync::LazyLock;
    static LOOKUP: LazyLock<Vec<u8>> = LazyLock::new(|| {
        // Build the reverse mapping once: the GPT-2 encoder assigns unicode codepoints
        // to bytes that aren't in the "printable" set. The mapping is:
        // printable bytes (33-126, 161-172, 174-255) → themselves
        // remaining bytes 0-32, 127-160, 173 → 256, 257, ... (U+0100, U+0101, ...)
        let mut non_printable = Vec::new();
        for b in 0u16..=255 {
            let is_printable =
                (33..=126).contains(&b) || (161..=172).contains(&b) || (174..=255).contains(&b);
            if !is_printable {
                non_printable.push(b as u8);
            }
        }
        non_printable
    });
    let table = &*LOOKUP;

    // non_printable[i] maps to U+0100+i
    let offset = cp.wrapping_sub(0x0100) as usize;
    if offset < table.len() {
        table[offset]
    } else {
        b'?'
    }
}

/// Notify all pipeline participants of an error.
pub async fn broadcast_pipeline_error(
    network_tx: &mpsc::Sender<NetworkCommand>,
    request_id: uuid::Uuid,
    error: &str,
) {
    let error_msg = SwarmMessage::InferenceError(InferenceError {
        request_id,
        error: error.to_string(),
        recoverable: false,
    });

    if let Err(e) = network_tx.send(NetworkCommand::Broadcast(error_msg)).await {
        tracing::warn!(error = %e, "Failed to broadcast pipeline error");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use crate::identity::Identity;
    use crate::inference::executor::ModelExecutor;
    use crate::storage::db::Database;
    use crate::types::*;
    use std::sync::Arc;
    use tokio::sync::Mutex;

    fn make_test_state() -> Arc<SharedState> {
        let config = Config::default();
        let identity = Identity::generate();
        let temp = tempfile::tempdir().unwrap();
        let db = Database::open(temp.path()).unwrap();
        let executor = Arc::new(Mutex::new(ModelExecutor::new()));
        let (state, _) = SharedState::new(config, identity, db, executor, None);
        state
    }

    fn make_test_request(state: &SharedState) -> InferenceRequest {
        InferenceRequest {
            id: uuid::Uuid::new_v4(),
            model_id: ModelId("test".into()),
            messages: vec![ChatMessage {
                role: Role::User,
                content: "hello".into(),
                images: vec![],
            }],
            sampling_params: SamplingParams::default(),
            stream: false,
            requester: state.identity.node_id().clone(),
            priority: PriorityTier::Silver,
            created_at: chrono::Utc::now(),
            session_id: None,
            lora_adapter: None,
        }
    }

    #[tokio::test]
    async fn empty_pipeline_fails() {
        let state = make_test_state();
        let (tx, _rx) = mpsc::channel::<NetworkCommand>(64);
        let request = make_test_request(&state);
        let assignment = PipelineAssignment {
            request_id: request.id,
            segments: vec![],
            standbys: vec![],
            tp_groups: vec![],
        };

        let mut executor = PipelineExecutor::new(state, tx, request, assignment);
        let result = executor.execute(None).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn local_pipeline_returns_stub() {
        let state = make_test_state();
        let local_id = state.identity.node_id().clone();
        let (tx, _rx) = mpsc::channel::<NetworkCommand>(64);
        let request = make_test_request(&state);
        let request_id = request.id;

        let assignment = PipelineAssignment {
            request_id,
            segments: vec![PipelineSegment {
                node_id: local_id,
                shard_id: ShardId {
                    model_id: ModelId("test".into()),
                    index: 0,
                },
                layer_range: (0, 32),
            }],
            standbys: vec![],
            tp_groups: vec![],
        };

        let mut executor = PipelineExecutor::new(state, tx, request, assignment);
        let result = executor.execute(None).await;
        // Without a loaded model, this should return a stub result
        // The local path falls through to the stub
        assert!(result.is_err()); // NoModelLoaded
    }
}
