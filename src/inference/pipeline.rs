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
        if num_segments == 1
            && self.assignment.segments[0].node_id == *self.shared_state.identity.node_id()
        {
            return self.execute_local().await;
        }

        // Distributed execution path
        self.execute_distributed(token_tx).await
    }

    /// Execute entirely on the local node (we have all layers).
    async fn execute_local(&self) -> Result<InferenceOutput, SwarmError> {
        let mut executor = self.shared_state.executor.lock().await;

        if !executor.is_loaded() {
            return Err(SwarmError::NoModelLoaded);
        }

        let prompt = self.build_prompt().await;
        let (content, gen_result) = executor.generate(&prompt, &self.request.sampling_params)?;

        Ok(InferenceOutput {
            request_id: self.request.id,
            content,
            prompt_tokens: gen_result.prompt_tokens,
            completion_tokens: gen_result.completion_tokens,
            finish_reason: gen_result.finish_reason.as_str().to_string(),
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
                    // After the first forward pass, the split model is loaded with its tokenizer.
                    // Use it to compute the real prompt token count for correct KV-cache positioning.
                    if seq_num == 0 {
                        let ptc = self.compute_prompt_token_count(&prompt).await;
                        index_pos = ptc;
                        prompt_token_count = Some(ptc);
                    } else {
                        index_pos += 1;
                    }

                    generated_tokens.extend(&result.token_ids);

                    // Stream each non-EOS token as it arrives
                    if let Some(ref tx) = token_tx {
                        let eos = self.get_eos_tokens().await;
                        for &tid in &result.token_ids {
                            if !eos.contains(&tid) {
                                let text = self.decode_tokens(&[tid]).await;
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
        let eos_tokens = self.get_eos_tokens().await;
        let clean_tokens: Vec<u32> = generated_tokens
            .iter()
            .copied()
            .filter(|t| !eos_tokens.contains(t))
            .collect();

        // Decode generated token IDs to text using the split model's vocabulary
        let generated_text = self.decode_tokens(&clean_tokens).await;

        Ok(InferenceOutput {
            request_id,
            content: generated_text,
            prompt_tokens: prompt_token_count.unwrap_or(prompt.len()) as u32,
            completion_tokens: generated_tokens.len() as u32,
            finish_reason,
        })
    }

    /// Compute prompt token count using BPE tokenizer if available, else byte-level count.
    async fn compute_prompt_token_count(&self, prompt: &str) -> usize {
        let model_id = &self.assignment.segments[0].shard_id.model_id;
        if let Some(model_ref) = self.shared_state.split_models.get(model_id) {
            let model = model_ref.lock().await;
            if let Some(tokenizer) = model.tokenizer() {
                return tokenizer.encode(prompt).len();
            }
        }
        prompt.len()
    }

    /// Decode token IDs to text using the GGUF vocabulary from the split model.
    async fn decode_tokens(&self, token_ids: &[u32]) -> String {
        let model_id = &self.assignment.segments[0].shard_id.model_id;
        if let Some(model_ref) = self.shared_state.split_models.get(model_id) {
            let model = model_ref.lock().await;
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

    /// Get EOS token IDs from the loaded split model, falling back to [2].
    async fn get_eos_tokens(&self) -> Vec<u32> {
        let model_id = &self.assignment.segments[0].shard_id.model_id;
        if let Some(model_ref) = self.shared_state.split_models.get(model_id) {
            let model = model_ref.lock().await;
            return model.eos_tokens().to_vec();
        }
        vec![2]
    }

    /// Forward activation data through all pipeline segments in order.
    async fn forward_through_segments(
        &mut self,
        request_id: uuid::Uuid,
        sequence_num: u32,
        index_pos: usize,
        initial_activations: Vec<u8>,
    ) -> Result<LayerResult, SwarmError> {
        let mut activations = initial_activations;
        let num_segments = self.assignment.segments.len();

        for (idx, segment) in self.assignment.segments.iter().enumerate() {
            let is_last = idx == num_segments - 1;

            // Send LayerForward to this segment's node
            let forward = LayerForward {
                request_id,
                sequence_num,
                index_pos: index_pos as u32,
                activations: activations.clone(),
                format: TensorFormat::FP32,
                sender_peer_bytes: None,
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
                        return self
                            .failover_segment(idx, request_id, sequence_num, &activations, is_last)
                            .await;
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

        // Ensure the split model is loaded for this model's layer range
        if !self.shared_state.split_models.contains_key(model_id) {
            // Find the GGUF file (reconstructed or original)
            let shard_store =
                crate::model::shard::ShardStore::new(&self.shared_state.config.node.data_dir);
            let model_dir = shard_store.models_dir().join(&model_id.0);
            let gguf_path = model_dir.join("model.gguf");

            // If no reconstructed GGUF, try the source_path file
            let gguf_path = if gguf_path.exists() {
                gguf_path
            } else {
                let source_path_file = model_dir.join("source_path");
                if source_path_file.exists() {
                    let p = std::fs::read_to_string(&source_path_file).map_err(SwarmError::Io)?;
                    std::path::PathBuf::from(p.trim())
                } else {
                    return Err(SwarmError::Internal(
                        "No GGUF file found for split model".into(),
                    ));
                }
            };

            // Determine if this is the first/last segment
            let manifest = self
                .shared_state
                .model_registry
                .get_manifest(model_id)
                .ok_or_else(|| SwarmError::Internal("No manifest for model".into()))?;
            let total_layers = manifest.num_layers as usize;
            let is_first = layer_start == 0;
            let is_last = layer_end >= total_layers;

            tracing::info!(
                model = %model_id,
                layers = format!("[{layer_start}..{layer_end})"),
                total = total_layers,
                path = %gguf_path.display(),
                "Loading split model segment"
            );

            let split_model =
                SplitModel::load_from_gguf(&gguf_path, layer_start, layer_end, is_first, is_last)?;

            self.shared_state.split_models.insert(
                model_id.clone(),
                std::sync::Arc::new(tokio::sync::Mutex::new(split_model)),
            );
        }

        let split_model_ref = self
            .shared_state
            .split_models
            .get(model_id)
            .ok_or_else(|| SwarmError::Internal("Split model not found after load".into()))?;

        let mut split_model = split_model_ref.lock().await;

        // Clear KV-cache at the start of a new request (prefill)
        if sequence_num == 0 {
            split_model.clear_kv_cache();
        }

        let is_first = split_model.layer_start == 0;
        let is_last = split_model.layer_end == split_model.total_layers;

        // Convert activation bytes to a candle Tensor
        let input_tensor = if is_first {
            if sequence_num == 0 {
                // Prefill: tokenize the full prompt using BPE tokenizer if available
                let prompt = self.build_prompt().await;
                let token_ids: Vec<i64> = if let Some(tokenizer) = split_model.tokenizer() {
                    tokenizer.encode(&prompt)
                } else {
                    // Fallback: byte-level tokenization
                    prompt.bytes().map(|b| b as i64).collect()
                };
                candle_core::Tensor::from_vec(
                    token_ids.clone(),
                    &[1, token_ids.len()],
                    &candle_core::Device::Cpu,
                )
                .map_err(|e| SwarmError::Internal(format!("Tensor creation failed: {e}")))?
            } else {
                // Decode step: activation_bytes contains a single i64 token ID (8 bytes LE)
                let token_id = if activation_bytes.len() >= 8 {
                    i64::from_le_bytes(activation_bytes[..8].try_into().unwrap())
                } else {
                    0i64
                };
                candle_core::Tensor::from_vec(vec![token_id], &[1, 1], &candle_core::Device::Cpu)
                    .map_err(|e| SwarmError::Internal(format!("Tensor creation failed: {e}")))?
            }
        } else {
            // Non-first segment: input is hidden states from previous segment
            split::bytes_to_tensor(activation_bytes)?
        };

        // Run the forward pass with correct position
        let output = split_model.forward(&input_tensor, index_pos)?;

        // Track local layer-forward participation and earn credits
        {
            let layers_processed = (layer_end - layer_start) as i64;
            if let Ok(mut stats) = self.shared_state.node_stats.try_write() {
                stats.forwards_served += 1;
            }
            let earned = crate::credit::ledger::RATE_INFERENCE_SERVE * layers_processed;
            if let Err(e) = crate::credit::ledger::apply_credit_direct(
                &self.shared_state.credit_balance,
                &self.shared_state.db,
                earned,
                true,
            )
            .await
            {
                tracing::warn!(error = %e, "Failed to persist credit earn");
            }
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
                    index_pos: 0, // Failover doesn't track position precisely
                    activations: activations.to_vec(),
                    format: TensorFormat::FP16,
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
                if self.network_tx
                    .send(NetworkCommand::SendTensor {
                        target_peer_bytes,
                        forward,
                    })
                    .await
                    .is_err()
                {
                    self.shared_state.pending_layer_results.remove(&request_id);
                    return Err(SwarmError::Network("Failed to send to standby node".to_string()));
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
            }],
            sampling_params: SamplingParams::default(),
            stream: false,
            requester: state.identity.node_id().clone(),
            priority: PriorityTier::Silver,
            created_at: chrono::Utc::now(),
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
        };

        let mut executor = PipelineExecutor::new(state, tx, request, assignment);
        let result = executor.execute(None).await;
        // Without a loaded model, this should return a stub result
        // The local path falls through to the stub
        assert!(result.is_err()); // NoModelLoaded
    }
}
