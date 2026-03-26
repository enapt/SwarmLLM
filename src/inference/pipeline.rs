use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::mpsc;

use crate::daemon::SharedState;
use crate::error::SwarmError;
use crate::inference::chat_template;
use crate::inference::router::{
    InferenceOutput, StreamingTokenEvent, StreamingTokenTx, TokenLogProbEntry,
};
use crate::inference::split;
use crate::types::{
    InferenceError, InferenceRequest, LayerForward, LayerResult, ModelId, NetworkCommand,
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
    /// Collected per-token logprobs during token generation.
    /// Uses Mutex because process_local_segment takes &self.
    collected_logprobs: std::sync::Mutex<Vec<TokenLogProbEntry>>,
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
            collected_logprobs: std::sync::Mutex::new(Vec::new()),
        }
    }

    /// Ensure the split model metadata entry exists in SharedState.
    ///
    /// Lightweight — reads GGUF header only, no GPU loading.
    /// Creates the entry if missing, handles VRAM budget eviction.
    fn ensure_split_model_entry(
        &self,
        model_id: &ModelId,
        layer_start: usize,
        layer_end: usize,
    ) -> Result<(ModelId, usize, usize), SwarmError> {
        let split_key = (model_id.clone(), layer_start, layer_end);
        if self.shared_state.split_models.contains_key(&split_key) {
            return Ok(split_key);
        }

        let shard_store =
            crate::model::shard::ShardStore::new(&self.shared_state.config.node.data_dir);
        let model_dir = shard_store.models_dir().join(&model_id.0);
        let manifest = self
            .shared_state
            .model_registry
            .get_manifest(model_id)
            .ok_or_else(|| SwarmError::Internal("No manifest for model".into()))?;
        let total_layers = manifest.num_layers as usize;

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
        let has_first = local_shards.contains(&0);
        let has_last = local_shards.contains(&manifest.shard_count.saturating_sub(1));
        let is_first = layer_start == 0 && has_first;
        let is_last = layer_end >= total_layers && has_last;

        let header_path = model_dir.join("gguf_header.bin");
        let vram_estimate = crate::daemon::estimate_vram_from_shard_dir(
            &model_dir,
            layer_start,
            layer_end,
            total_layers,
        );
        let new_entry = split::SplitModelEntry::from_header(
            &header_path,
            layer_start,
            layer_end,
            is_first,
            is_last,
            vram_estimate,
        );

        let vram_budget = crate::model::auto_manage::compute_vram_budget(&self.shared_state)
            .or(self.shared_state.config.inference.max_split_model_memory_mb);
        if let Some(budget_mb) = vram_budget {
            split::evict_split_models_lru(
                &self.shared_state.split_models,
                &self.shared_state.active_pipelines,
                budget_mb,
                new_entry.estimated_vram_mb,
            );
        }
        self.shared_state
            .split_models
            .entry(split_key.clone())
            .or_insert(new_entry);

        Ok(split_key)
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
            return Err(SwarmError::Internal(
                "Multi-image VLM requires local mmproj — remote encoding only supports single images"
                    .into(),
            ));
        }

        // Compress image as JPEG for wire transfer
        let first_image = &images[0];
        let jpeg_bytes = self.compress_image_jpeg(first_image)?;

        // Register response channel with expected responder for auth verification
        let (tx, rx) = tokio::sync::oneshot::channel();
        self.shared_state
            .pending_vision_results
            .insert(self.request.id, (remote_node.clone(), tx));

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
                    token_logprobs: vec![],
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
            token_logprobs: vec![],
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

        if max_tokens == 0 {
            return Ok(InferenceOutput {
                request_id,
                content: String::new(),
                prompt_tokens: 0,
                completion_tokens: 0,
                finish_reason: "length".to_string(),
                session_id: self.request.session_id.clone(),
                token_logprobs: vec![],
            });
        }

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
        // Use the specific model's GGUF header first, fall back to loaded_model_info singleton.
        let stop_strings = {
            let model_id = &self.request.model_id;
            let shard_store =
                crate::model::shard::ShardStore::new(&self.shared_state.config.node.data_dir);
            let header_path = shard_store
                .models_dir()
                .join(&model_id.0)
                .join("gguf_header.bin");
            if let Some((tmpl, _, _)) = template_from_header(&header_path) {
                chat_template::extract_stop_strings(tmpl.as_deref())
            } else {
                let info = self.shared_state.loaded_model_info.read().await;
                let tmpl = info.as_ref().and_then(|i| i.chat_template.as_deref());
                chat_template::extract_stop_strings(tmpl)
            }
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

        // Local embedding privacy: check if we should embed locally before sending
        // activations to the first pipeline segment. This prevents remote nodes from
        // seeing raw token IDs — they only receive hidden-state activation tensors.
        // Auto-enabled when encrypted_pipeline is active (it requires both ends local).
        let model_id = &self.assignment.segments[0].shard_id.model_id;
        let encrypted_for_model = self
            .shared_state
            .encrypted_pipeline_models
            .get(model_id)
            .map(|r| *r.value())
            .unwrap_or(self.shared_state.config.inference.encrypted_pipeline);
        let use_local_embedding =
            self.shared_state.config.inference.local_embedding_privacy || encrypted_for_model;
        let local_embedder = if use_local_embedding {
            self.shared_state
                .local_embedders
                .get(model_id)
                .map(|e| e.value().clone())
        } else {
            None
        };

        // Token generation loop
        let mut prompt_bytes_opt = Some(prompt_bytes);
        for seq_num in 0..max_tokens {
            let (activations, pre_embedded) = if let Some(ref embedder) = local_embedder {
                // Local embedding privacy: embed locally, never send raw tokens
                if seq_num == 0 {
                    let prompt =
                        std::str::from_utf8(prompt_bytes_opt.as_ref().unwrap()).unwrap_or("");
                    let (bytes, token_count) = embedder.embed_prompt(prompt)?;
                    // Set prompt_token_count from local tokenization
                    if prompt_token_count.is_none() {
                        prompt_token_count = Some(token_count);
                        index_pos = token_count;
                    }
                    prompt_bytes_opt.take();
                    (bytes, true)
                } else {
                    let last_token = generated_tokens.last().copied().unwrap_or(0);
                    let bytes = embedder.embed_token(last_token)?;
                    (bytes, true)
                }
            } else if seq_num == 0 {
                (
                    prompt_bytes_opt
                        .take()
                        .expect("seq_num==0 implies prompt_bytes set"),
                    false,
                )
            } else {
                // For subsequent tokens, encode the last generated token ID as i64 LE bytes
                // so the first segment can embed it directly.
                let last_token = generated_tokens.last().copied().unwrap_or(0) as i64;
                (last_token.to_le_bytes().to_vec(), false)
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
                    pre_embedded,
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
                                    accumulated_text.truncate(pos);
                                    if let Some(ref mut st) = streamed_text {
                                        // Remove the stop string from streamed text too
                                        // Use find (not rfind) to match the first occurrence,
                                        // consistent with accumulated_text truncation above.
                                        if let Some(spos) = st.find(stop.as_str()) {
                                            st.truncate(spos);
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

                    // Check for EOS tokens in the result — the worker may return EOS
                    // as a token ID without setting finish_reason explicitly.
                    if result.token_ids.iter().any(|t| eos.contains(t)) {
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
        if generated_tokens.len() as u32 >= max_tokens && finish_reason.is_empty() {
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
        let mut generated_text = if let Some(text) = streamed_text {
            text
        } else if !accumulated_text.is_empty() {
            accumulated_text
        } else {
            match cached_decoder.as_ref() {
                Some(d) => d.decode_tokens(&clean_tokens),
                None => self.decode_tokens(&clean_tokens).await,
            }
        };

        // Strip trailing partial stop strings (e.g. "<|user" when stop is "<|user|>").
        // The token-by-token check above only catches complete matches, so a partial
        // stop string at the very end of generation can leak into the output.
        'stop_trim: for stop in &stop_strings {
            for end_len in (1..stop.len()).rev() {
                let prefix = &stop[..end_len];
                if generated_text.ends_with(prefix) {
                    generated_text.truncate(generated_text.len() - end_len);
                    break 'stop_trim; // Only trim once — don't cascade across stop strings
                }
            }
        }

        // Batch credit write — one DB persist for the entire request instead of per-token.
        // Formula: rate * tokens (no layer multiplier — balanced with consume side).
        let total_tokens = generated_tokens.len() as i64;
        if total_tokens > 0 {
            let has_local_segment = self
                .assignment
                .segments
                .iter()
                .any(|s| s.node_id == *self.shared_state.identity.node_id());
            if has_local_segment {
                let rate = self.shared_state.config.pool.credit_rates.inference_serve;
                let total_earned = rate.saturating_mul(total_tokens);
                if let Err(e) = crate::credit::ledger::apply_credit_direct(
                    &self.shared_state.credits.credit_balance,
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
            completion_tokens: generated_tokens.len() as u32,
            finish_reason: if finish_reason.is_empty() {
                "stop".to_string()
            } else {
                finish_reason
            },
            session_id: self.request.session_id.clone(),
            token_logprobs: self
                .collected_logprobs
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .drain(..)
                .collect(),
        })
    }

    /// Decode token IDs to text using cached vocabulary from the split model metadata.
    async fn decode_tokens(&self, token_ids: &[u32]) -> String {
        let model_id = &self.assignment.segments[0].shard_id.model_id;
        let entry = self
            .shared_state
            .split_models
            .iter()
            .find(|e| e.key().0 == *model_id);
        if let Some(entry) = entry {
            if let Some(ref vocab) = entry.value().vocab {
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

    /// Extract prompt token count, EOS tokens, and a cached decoder from metadata.
    /// No model lock needed — uses cached metadata from SplitModelEntry.
    async fn extract_model_cache(&self, prompt: &str) -> (usize, Vec<u32>, CachedDecoder) {
        let model_id = &self.assignment.segments[0].shard_id.model_id;
        let entry = self
            .shared_state
            .split_models
            .iter()
            .find(|e| e.key().0 == *model_id);

        if let Some(entry) = entry {
            let eos = entry.value().eos_tokens.clone();
            let vocab = entry.value().vocab.clone().unwrap_or_default();

            // Approximate prompt token count (no tokenizer in-process)
            // Rough estimate: chars / 4 (average BPE token length), minimum 1
            let ptc = (prompt.chars().count() / 4).max(1);

            let decoder = CachedDecoder {
                vocab: vocab.clone(),
                byte_decoder: HashMap::new(),
                is_sentencepiece: false,
                has_tokenizer: false,
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
                            (prompt.chars().count() / 4).max(1)
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
                        let ptc = (prompt.chars().count() / 4).max(1);
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
                if let Some(hf_source) = self.shared_state.models.hf_sources.get(model_id) {
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
            let add_space_prefix = ct
                .metadata
                .get("tokenizer.ggml.add_space_prefix")
                .and_then(|v| v.to_bool().ok())
                .unwrap_or(true);
            let add_bos_token = ct
                .metadata
                .get("tokenizer.ggml.add_bos_token")
                .and_then(|v| v.to_bool().ok())
                .unwrap_or(false);
            if !scores.is_empty() {
                Some(crate::inference::split::SplitTokenizer::from_sentencepiece(
                    &vocab,
                    &scores,
                    add_space_prefix,
                    add_bos_token,
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
    /// Pipeline sealing: unseal a LayerResult if it contains sealed token IDs.
    /// Recovers the real token_ids from the sealed envelope using this node's X25519 secret.
    fn unseal_result(&self, mut result: LayerResult) -> LayerResult {
        if let Some(ref sealed_bytes) = result.sealed_token_ids {
            match serde_json::from_slice::<crate::types::SealedPrompt>(sealed_bytes) {
                Ok(sealed) => {
                    let local_secret = crate::crypto::session::ed25519_to_x25519_secret(
                        &self.shared_state.identity.signing_key_bytes(),
                    );
                    match crate::crypto::pipeline_seal::open_prompt(&sealed, &local_secret) {
                        Ok(plaintext) => {
                            // Deserialize token IDs from the decrypted payload
                            if let Ok(token_ids) = serde_json::from_slice::<Vec<u32>>(&plaintext) {
                                tracing::debug!(
                                    request_id = %result.request_id,
                                    num_tokens = token_ids.len(),
                                    "Pipeline seal: unsealed token IDs from final segment"
                                );
                                result.token_ids = token_ids;
                                result.sealed_token_ids = None;
                            } else {
                                tracing::warn!(
                                    request_id = %result.request_id,
                                    "Pipeline seal: failed to deserialize unsealed token IDs"
                                );
                            }
                        }
                        Err(e) => {
                            tracing::error!(
                                request_id = %result.request_id,
                                error = %e,
                                "Pipeline seal: failed to unseal result — rejecting (no plaintext fallback)"
                            );
                            // Do NOT fall through to plaintext — clear tokens to prevent
                            // accepting unverified data as legitimate decrypted output
                            result.token_ids.clear();
                        }
                    }
                }
                Err(e) => {
                    tracing::warn!(
                        request_id = %result.request_id,
                        error = %e,
                        "Pipeline seal: failed to parse SealedPrompt from result"
                    );
                }
            }
        }
        result
    }

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
        pre_embedded: bool,
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
                let tp_result = self
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

                // Parse the tagged result: 0x01 prefix = sampled token, 0x00 = raw activations
                if !tp_result.is_empty() && tp_result[0] == 0x01 {
                    // Last segment returned a sampled token ID
                    let token_id = if tp_result.len() >= 9 {
                        i64::from_le_bytes(tp_result[1..9].try_into().unwrap()) as u32
                    } else {
                        0u32
                    };
                    // Check EOS
                    let eos_tokens = self
                        .shared_state
                        .split_models
                        .get(&(
                            segment.shard_id.model_id.clone(),
                            segment.layer_range.0 as usize,
                            segment.layer_range.1 as usize,
                        ))
                        .map(|e| e.value().eos_tokens.clone())
                        .unwrap_or_default();
                    let finish = if eos_tokens.contains(&token_id) {
                        Some(NetworkFinishReason::Stop)
                    } else {
                        None
                    };
                    return Ok(LayerResult {
                        request_id,
                        token_ids: vec![token_id],
                        finish_reason: finish,
                        activations: vec![],
                        sealed_token_ids: None,
                    });
                } else {
                    // Intermediate segment: strip the 0x00 tag and continue
                    activations = if !tp_result.is_empty() {
                        tp_result[1..].to_vec()
                    } else {
                        tp_result
                    };
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
                        pre_embedded && idx == 0,
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
                    // Pipeline sealing: attach our node ID so the final segment
                    // can seal the result tokens for our X25519 key.
                    requester_node_id: Some(self.shared_state.identity.node_id().0),
                    // Local embedding privacy: only the first segment of the first
                    // forward needs this flag (subsequent segments receive hidden states anyway).
                    pre_embedded: pre_embedded && idx == 0,
                    adapter_id: None,
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
                            // Remove stale pending entry before failover inserts a new one
                            self.shared_state.pending_layer_results.remove(&request_id);
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
                                // Pipeline sealing: unseal token IDs if the final node sealed them
                                let result = self.unseal_result(result);
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
        pre_embedded: bool,
    ) -> Result<LayerResult, SwarmError> {
        let model_id = &segment.shard_id.model_id;
        let (layer_start, layer_end) = (
            segment.layer_range.0 as usize,
            segment.layer_range.1 as usize,
        );

        let split_key = self.ensure_split_model_entry(model_id, layer_start, layer_end)?;

        // Touch the metadata entry and extract cached EOS tokens
        let _cached_eos_tokens = {
            let entry = self
                .shared_state
                .split_models
                .get(&split_key)
                .ok_or_else(|| SwarmError::Internal("Split model not found".into()))?;
            entry.value().touch();
            entry.value().eos_tokens.clone()
        };

        // Build a LayerForward and route to the worker subprocess
        let layer_forward = crate::types::LayerForward {
            request_id: self.request.id,
            sequence_num,
            index_pos: index_pos as u32,
            activations: activation_bytes.to_vec(),
            format: crate::types::TensorFormat::FP16,
            model_id: model_id.clone(),
            layer_range: (layer_start as u32, layer_end as u32),
            tp_meta: None,
            vision_embeddings: precomputed_vision_bytes.map(|b| b.to_vec()),
            sender_peer_bytes: None,
            requester_node_id: None,
            pre_embedded,
            adapter_id: None,
        };
        let layer_result = self
            .shared_state
            .model_process_pool
            .forward(layer_forward)
            .await?;

        // Track stats
        if let Ok(mut stats) = self.shared_state.metrics.node_stats.try_write() {
            stats.forwards_served += 1;
        }

        Ok(layer_result)
    }

    /// Try to identify a prefix-cacheable system prompt in the request.
    ///
    /// Returns `Some((blake3_hash, prefix_token_count))` if the request has system
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
                    requester_node_id: Some(self.shared_state.identity.node_id().0),
                    pre_embedded: false,
                    adapter_id: None,
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

        let split_key = self.ensure_split_model_entry(model_id, layer_start, layer_end)?;

        // Touch the metadata entry
        if let Some(entry) = self.shared_state.split_models.get(&split_key) {
            entry.value().touch();
        }

        // TP execution: layer-by-layer with AllReduce coordination in main process.
        // Each layer's partial computation is routed to the worker subprocess via
        // LayerForward with tp_meta set. AllReduce network communication stays here.
        if let Some(tp_rank) = local_tp_rank {
            // We are in the TP group — per-layer partial computation via subprocess + AllReduce
            let mut current_activations_bytes = activation_bytes.to_vec();

            for abs_layer in layer_start..layer_end {
                // 2-phase TP protocol: AttnOnly → AllReduce → FfnOnly → AllReduce
                // This ensures FFN norm is applied to the full post-attention tensor,
                // not the partial pre-AllReduce output.

                // Phase 1: AttnOnly — norm → head-sliced attention → partial output
                let attn_forward = crate::types::LayerForward {
                    request_id,
                    sequence_num,
                    index_pos: index_pos as u32,
                    activations: current_activations_bytes.clone(),
                    format: crate::types::TensorFormat::FP16,
                    model_id: model_id.clone(),
                    layer_range: (layer_start as u32, layer_end as u32),
                    tp_meta: Some(crate::types::TensorParallelMeta {
                        tp_rank: tp_rank as u8,
                        tp_size: tp_size as u8,
                        single_layer: abs_layer as u32,
                        phase: crate::types::TpPhase::AttnOnly,
                    }),
                    vision_embeddings: None,
                    sender_peer_bytes: None,
                    requester_node_id: None,
                    pre_embedded: false,
                    adapter_id: None,
                };
                let attn_partial = self
                    .shared_state
                    .model_process_pool
                    .forward(attn_forward)
                    .await?;

                // AllReduce attention partials → full post-attention output
                let attn_compressed =
                    zstd::encode_all(std::io::Cursor::new(&attn_partial.activations), 1)
                        .map_err(|e| SwarmError::Internal(format!("Compress attn partial: {e}")))?;
                let attn_tensor = split::bytes_to_tensor(&attn_partial.activations)
                    .map_err(|e| SwarmError::Internal(format!("Deserialize attn partial: {e}")))?;
                let attn_shape: Vec<u32> = attn_tensor.dims().iter().map(|&d| d as u32).collect();

                // Use layer*2 as AllReduce step ID for attn phase, layer*2+1 for FFN phase
                let attn_resp = crate::inference::allreduce::allreduce_sum(
                    &self.shared_state,
                    &self.network_tx,
                    &self.shared_state.allreduce_registry,
                    request_id,
                    abs_layer as u32 * 2,
                    tp_group,
                    tp_rank,
                    attn_compressed,
                    attn_shape,
                )
                .await?;

                // Decompress full post-attention tensor + residual add happens in worker
                let post_attn_bytes =
                    zstd::decode_all(std::io::Cursor::new(&attn_resp.reduced_data))
                        .map_err(|e| SwarmError::Internal(format!("Decompress attn AR: {e}")))?;

                // Phase 2: FfnOnly — ffn_norm → column-sliced FFN → partial output
                let ffn_forward = crate::types::LayerForward {
                    request_id,
                    sequence_num,
                    index_pos: index_pos as u32,
                    activations: post_attn_bytes,
                    format: crate::types::TensorFormat::FP16,
                    model_id: model_id.clone(),
                    layer_range: (layer_start as u32, layer_end as u32),
                    tp_meta: Some(crate::types::TensorParallelMeta {
                        tp_rank: tp_rank as u8,
                        tp_size: tp_size as u8,
                        single_layer: abs_layer as u32,
                        phase: crate::types::TpPhase::FfnOnly,
                    }),
                    vision_embeddings: None,
                    sender_peer_bytes: None,
                    requester_node_id: None,
                    pre_embedded: false,
                    adapter_id: None,
                };
                let ffn_partial = self
                    .shared_state
                    .model_process_pool
                    .forward(ffn_forward)
                    .await?;

                // AllReduce FFN partials → full post-FFN output
                let ffn_compressed =
                    zstd::encode_all(std::io::Cursor::new(&ffn_partial.activations), 1)
                        .map_err(|e| SwarmError::Internal(format!("Compress ffn partial: {e}")))?;
                let ffn_tensor = split::bytes_to_tensor(&ffn_partial.activations)
                    .map_err(|e| SwarmError::Internal(format!("Deserialize ffn partial: {e}")))?;
                let ffn_shape: Vec<u32> = ffn_tensor.dims().iter().map(|&d| d as u32).collect();

                let ffn_resp = crate::inference::allreduce::allreduce_sum(
                    &self.shared_state,
                    &self.network_tx,
                    &self.shared_state.allreduce_registry,
                    request_id,
                    abs_layer as u32 * 2 + 1,
                    tp_group,
                    tp_rank,
                    ffn_compressed,
                    ffn_shape,
                )
                .await?;

                // Decompress full layer output — becomes input for next layer
                current_activations_bytes =
                    zstd::decode_all(std::io::Cursor::new(&ffn_resp.reduced_data))
                        .map_err(|e| SwarmError::Internal(format!("Decompress ffn AR: {e}")))?;
            }

            if _is_last {
                // Last segment: the final activations need token sampling
                let final_tensor = split::bytes_to_tensor(&current_activations_bytes)
                    .map_err(|e| SwarmError::Internal(format!("Deserialize TP output: {e}")))?;
                let token_id =
                    split::sample_token_with_params(&final_tensor, &self.request.sampling_params)?;
                let mut result = vec![0x01];
                result.extend_from_slice(&(token_id as i64).to_le_bytes());
                Ok(result)
            } else {
                // Intermediate segment: prefix raw activations with 0x00 tag
                let mut result = vec![0x00];
                result.extend(current_activations_bytes);
                Ok(result)
            }
        } else {
            // Not in TP group — run full forward via subprocess (non-TP path)
            let layer_forward = crate::types::LayerForward {
                request_id,
                sequence_num,
                index_pos: index_pos as u32,
                activations: activation_bytes.to_vec(),
                format: crate::types::TensorFormat::FP16,
                model_id: model_id.clone(),
                layer_range: (layer_start as u32, layer_end as u32),
                tp_meta: None,
                vision_embeddings: None,
                sender_peer_bytes: None,
                requester_node_id: None,
                pre_embedded: false,
                adapter_id: None,
            };
            let layer_result = self
                .shared_state
                .model_process_pool
                .forward(layer_forward)
                .await?;

            if _is_last {
                let final_tensor = split::bytes_to_tensor(&layer_result.activations)
                    .map_err(|e| SwarmError::Internal(format!("Deserialize output: {e}")))?;
                let token_id =
                    split::sample_token_with_params(&final_tensor, &self.request.sampling_params)?;
                let mut result = vec![0x01];
                result.extend_from_slice(&(token_id as i64).to_le_bytes());
                Ok(result)
            } else {
                let mut result = vec![0x00];
                result.extend(layer_result.activations);
                Ok(result)
            }
        }
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
        let (state, _, _) = SharedState::new(config, identity, db, executor, None);
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
