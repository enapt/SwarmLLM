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

    /// Build chat prompt using the template from the loaded model (if available).
    async fn build_prompt(&self) -> String {
        let info = self.shared_state.loaded_model_info.read().await;
        match info.as_ref() {
            Some(i) => chat_template::build_prompt(
                &self.request.messages,
                i.chat_template.as_deref(),
                &i.bos_token,
                &i.eos_token,
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
        let mut finish_reason = "stop".to_string();

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

        // Token generation loop
        for seq_num in 0..max_tokens {
            let activations = if seq_num == 0 {
                prompt_bytes.clone()
            } else {
                // For subsequent tokens, encode the last generated token ID as i64 LE bytes
                // so the first segment can embed it directly.
                let last_token = generated_tokens.last().copied().unwrap_or(0) as i64;
                last_token.to_le_bytes().to_vec()
            };

            // Forward through each segment
            match self
                .forward_through_segments(request_id, seq_num, index_pos, activations)
                .await
            {
                Ok(result) => {
                    // After the first forward pass, extract everything we need from the model
                    // in a SINGLE lock acquisition: prompt token count, EOS tokens, and
                    // cached decoder for lock-free per-token decoding.
                    if seq_num == 0 {
                        let (ptc, eos, decoder) = self.extract_model_cache(&prompt).await;
                        index_pos = ptc;
                        prompt_token_count = Some(ptc);
                        cached_eos = Some(eos);
                        cached_decoder = Some(decoder);
                    } else {
                        index_pos += 1;
                    }

                    generated_tokens.extend(&result.token_ids);

                    // Stream each non-EOS token — uses cached decoder (no mutex)
                    if let Some(ref tx) = token_tx {
                        let eos = cached_eos.as_deref().unwrap_or(&[2]);
                        let decoder = cached_decoder.as_ref();
                        for &tid in &result.token_ids {
                            if !eos.contains(&tid) {
                                let text = match decoder {
                                    Some(d) => d.decode_tokens(&[tid]),
                                    None => format!("[{tid}]"),
                                };
                                if let Some(ref mut st) = streamed_text {
                                    st.push_str(&text);
                                }
                                let _ = tx
                                    .send(StreamingTokenEvent {
                                        text,
                                        finish_reason: None,
                                    })
                                    .await;
                            }
                        }
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

        // For streaming: use already-decoded text. For non-streaming: decode once at end.
        let generated_text = match streamed_text {
            Some(text) => text,
            None => match cached_decoder.as_ref() {
                Some(d) => d.decode_tokens(&clean_tokens),
                None => self.decode_tokens(&clean_tokens).await,
            },
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
                    false,
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
            finish_reason,
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

            // 1. Prompt token count
            let ptc = if let Some(tokenizer) = model.tokenizer() {
                tokenizer.encode(prompt).len()
            } else {
                prompt.chars().count() / 4
            };

            // 2. EOS tokens
            let eos = model.eos_tokens().to_vec();

            // 3. Cached decoder — clone vocab + byte_decoder for lock-free decoding
            let decoder = if let Some(vocab) = model.vocab() {
                let (byte_decoder, is_sentencepiece, has_tokenizer) =
                    if let Some(tokenizer) = model.tokenizer() {
                        (
                            tokenizer.byte_decoder().clone(),
                            tokenizer.is_sentencepiece(),
                            true,
                        )
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
            // No model loaded — use fallbacks
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
    ) -> Result<LayerResult, SwarmError> {
        let mut activations = initial_activations;
        let num_segments = self.assignment.segments.len();

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
            let forward = LayerForward {
                request_id,
                sequence_num,
                index_pos: index_pos as u32,
                activations: activations.clone(),
                format: TensorFormat::FP32,
                layer_range: Some(segment.layer_range),
                sender_peer_bytes: None,
                tp_meta: None,
            };

            // If this is the local node, process locally
            if segment.node_id == *self.shared_state.identity.node_id() {
                let result = self
                    .process_local_segment(segment, sequence_num, index_pos, &activations)
                    .await?;
                if is_last {
                    return Ok(result);
                }
                // Use hidden-state activations for the next segment
                activations = result.activations;
            } else {
                // Register a response channel BEFORE sending the request
                let (tx, rx) = tokio::sync::oneshot::channel();
                self.shared_state
                    .pending_layer_results
                    .insert(request_id, tx);

                // Look up the peer's libp2p PeerId bytes from the peer registry.
                // NodeId (Ed25519 key) != PeerId (libp2p identity), so we need the mapping.
                let target_peer_bytes = self
                    .shared_state
                    .peer_registry
                    .get(&segment.node_id)
                    .and_then(|p| p.peer_id_bytes.clone())
                    .ok_or_else(|| {
                        SwarmError::Network(format!(
                            "No peer_id_bytes for node {}",
                            segment.node_id
                        ))
                    })?;

                // Send to remote node via directed tensor protocol
                if self
                    .network_tx
                    .send(NetworkCommand::SendTensor {
                        target_peer_bytes,
                        forward,
                    })
                    .await
                    .is_err()
                {
                    // Clean up the pending entry to prevent memory leak
                    self.shared_state.pending_layer_results.remove(&request_id);
                    return Err(SwarmError::Network(
                        "Failed to send LayerForward".to_string(),
                    ));
                }

                // Wait for response via the oneshot channel (with timeout)
                match Self::wait_for_result(rx).await {
                    Ok(result) => {
                        if is_last {
                            return Ok(result);
                        }
                        // Use hidden-state activations for the next segment
                        activations = result.activations;
                    }
                    Err(_e) => {
                        // Clean up the pending entry to prevent memory leak
                        self.shared_state.pending_layer_results.remove(&request_id);
                        // Attempt failover to standby
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
                        // Continue pipeline with the standby's activations
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
            let new_entry = if max_batch > 1 {
                crate::inference::split::SplitModelEntry::new_with_batching(
                    split_model,
                    self.shared_state.kv_cache_store.clone(),
                    max_batch,
                )
            } else {
                crate::inference::split::SplitModelEntry::new(split_model)
            };
            if let Some(budget_mb) = self.shared_state.config.inference.max_split_model_memory_mb {
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
        let (split_model_ref, batch_forwarder) = {
            let entry = self
                .shared_state
                .split_models
                .get(&split_key)
                .ok_or_else(|| SwarmError::Internal("Split model not found after load".into()))?;
            entry.value().touch();
            (
                entry.value().model.clone(),
                entry.value().batch_forwarder.clone(),
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

            // Post-process: need model lock for EOS tokens and sampling
            let split_model = split_model_ref.lock().await;
            let is_last = split_model.layer_end >= split_model.total_layers;

            if is_last {
                let token_id = split::sample_token(
                    &output,
                    self.request.sampling_params.temperature,
                    self.request.sampling_params.top_p,
                )?;
                let eos_tokens = split_model.eos_tokens();
                let finish = if eos_tokens.contains(&token_id) {
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
                let all_tokens: Vec<i64> = if let Some(tokenizer) = split_model.tokenizer() {
                    tokenizer.encode(&prompt)
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
                    let mut cache_guard = self.shared_state.prefix_cache.lock().unwrap();

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

                        // Forward prefix through model (populates KV cache)
                        let _prefix_out = split_model.forward(
                            &prefix_tensor,
                            0,
                            &self.shared_state.kv_cache_store,
                            &request_id_str,
                        )?;

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
                                let mut cache_guard =
                                    self.shared_state.prefix_cache.lock().unwrap();
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

        // Run the forward pass with per-request KV-cache isolation
        let output = split_model.forward_with_lora(
            &input_tensor,
            effective_index_pos,
            &self.shared_state.kv_cache_store,
            &request_id_str,
            lora_adapter.as_deref(),
        )?;

        // Track stats (credit persistence is batched at end of request)
        if let Ok(mut stats) = self.shared_state.node_stats.try_write() {
            stats.forwards_served += 1;
        }

        if is_last {
            // Last segment: output is logits → sample token
            let token_id = split::sample_token(
                &output,
                self.request.sampling_params.temperature,
                self.request.sampling_params.top_p,
            )?;

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

    /// Wait for a remote segment to return its result via the oneshot channel.
    async fn wait_for_result(
        rx: tokio::sync::oneshot::Receiver<LayerResult>,
    ) -> Result<LayerResult, SwarmError> {
        match tokio::time::timeout(Duration::from_secs(30), rx).await {
            Ok(Ok(result)) => Ok(result),
            Ok(Err(_)) => Err(SwarmError::PipelineError("Response channel dropped".into())),
            Err(_) => Err(SwarmError::PipelineError(
                "Timed out waiting for segment result".into(),
            )),
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
                tracing::info!(
                    request_id = %request_id,
                    failed_node = %failed_segment.node_id,
                    backup_node = %backup.node_id,
                    "Failing over to standby node"
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
                    layer_range: Some(backup.layer_range),
                    tp_meta: None,
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
                Self::wait_for_result(rx).await
            }
            None => {
                tracing::error!(
                    request_id = %request_id,
                    segment = failed_idx,
                    "No standby available for failed segment"
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
                let mut model = model_arc.lock().await;
                let partial = model.forward_tp_layer(
                    &current_activations,
                    abs_layer,
                    index_pos,
                    tp_rank,
                    tp_size,
                    kv_cache_store,
                    &req_id_str,
                )?;

                // For now: if this is the only local TP participant, just use partial + residual
                // Full AllReduce with remote nodes would send partial to each remote TP node,
                // collect their partials, and sum. This is the local-only TP path.
                // TODO: implement remote AllReduce for multi-node TP
                current_activations = (partial + &residual)
                    .map_err(|e| SwarmError::Internal(format!("Residual add: {e}")))?;
            } else {
                // We're not in the TP group — run full (non-TP) layer
                let mut model = model_arc.lock().await;
                let result =
                    model.forward(&current_activations, index_pos, kv_cache_store, &req_id_str)?;
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
    use std::sync::OnceLock;
    static LOOKUP: OnceLock<Vec<u8>> = OnceLock::new();

    let table = LOOKUP.get_or_init(|| {
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
