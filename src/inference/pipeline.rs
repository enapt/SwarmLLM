use std::sync::Arc;
use std::time::Duration;

use tokio::sync::mpsc;

use crate::daemon::SharedState;
use crate::error::SwarmError;
use crate::inference::executor::build_chat_prompt;
use crate::inference::router::InferenceOutput;
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

    /// Execute the pipeline end-to-end.
    ///
    /// For each token:
    /// 1. Send initial prompt activations to the first segment
    /// 2. Each segment processes its layers and forwards to the next
    /// 3. The last segment samples a token and returns a LayerResult
    /// 4. Repeat until stop condition or max_tokens
    pub async fn execute(&mut self) -> Result<InferenceOutput, SwarmError> {
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
        self.execute_distributed().await
    }

    /// Execute entirely on the local node (we have all layers).
    async fn execute_local(&self) -> Result<InferenceOutput, SwarmError> {
        let mut executor = self.shared_state.executor.lock().await;

        if !executor.is_loaded() {
            return Err(SwarmError::NoModelLoaded);
        }

        let prompt = build_chat_prompt(&self.request.messages);
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
    async fn execute_distributed(&mut self) -> Result<InferenceOutput, SwarmError> {
        let request_id = self.request.id;
        let max_tokens = self.request.sampling_params.max_tokens;

        // Build the initial prompt representation
        let prompt = build_chat_prompt(&self.request.messages);
        let prompt_bytes = prompt.as_bytes().to_vec();
        let prompt_tokens = (prompt.len() / 4).max(1) as u32;

        let mut generated_tokens: Vec<u32> = Vec::new();
        let mut generated_text = String::new();
        let mut finish_reason = "stop".to_string();

        // Token generation loop
        for seq_num in 0..max_tokens {
            let activations = if seq_num == 0 {
                prompt_bytes.clone()
            } else {
                // For subsequent tokens, send the last generated token as activation
                let last_token = generated_tokens.last().copied().unwrap_or(0);
                last_token.to_le_bytes().to_vec()
            };

            // Forward through each segment
            match self
                .forward_through_segments(request_id, seq_num, activations)
                .await
            {
                Ok(result) => {
                    generated_tokens.extend(&result.token_ids);

                    // Decode tokens (stub: use token IDs as ASCII chars)
                    for &token_id in &result.token_ids {
                        if token_id < 128 {
                            generated_text.push(token_id as u8 as char);
                        } else {
                            generated_text.push_str(&format!("[{token_id}]"));
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
                        break;
                    }
                }
                Err(e) => {
                    // Pipeline error — try failover
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
        }

        Ok(InferenceOutput {
            request_id,
            content: generated_text,
            prompt_tokens,
            completion_tokens: generated_tokens.len() as u32,
            finish_reason,
        })
    }

    /// Forward activation data through all pipeline segments in order.
    async fn forward_through_segments(
        &mut self,
        request_id: uuid::Uuid,
        sequence_num: u32,
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
                activations: activations.clone(),
                format: TensorFormat::FP16,
            };

            // If this is the local node, process locally
            if segment.node_id == *self.shared_state.identity.node_id() {
                let result = self.process_local_segment(segment, &activations).await?;
                if is_last {
                    return Ok(result);
                }
                // Use the result's token_ids as activations for next segment
                activations = result
                    .token_ids
                    .iter()
                    .flat_map(|t| t.to_le_bytes())
                    .collect();
            } else {
                // Send to remote node via directed tensor protocol
                let target_peer_bytes = segment.node_id.0.to_vec();
                self.network_tx
                    .send(NetworkCommand::SendTensor {
                        target_peer_bytes,
                        forward,
                    })
                    .await
                    .map_err(|_| SwarmError::Network("Failed to send LayerForward".to_string()))?;

                // Wait for response (with timeout)
                // In a full implementation, this would use a response channel
                // keyed by (request_id, segment_idx). For now, simulate with timeout.
                match self.wait_for_segment_result(request_id, idx, is_last).await {
                    Ok(result) => {
                        if is_last {
                            return Ok(result);
                        }
                        activations = result
                            .token_ids
                            .iter()
                            .flat_map(|t| t.to_le_bytes())
                            .collect();
                    }
                    Err(_e) => {
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

    /// Process a pipeline segment locally (this node has the shard).
    async fn process_local_segment(
        &self,
        _segment: &PipelineSegment,
        _activations: &[u8],
    ) -> Result<LayerResult, SwarmError> {
        // In a full implementation, this would:
        // 1. Load the shard for segment.layer_range
        // 2. Run the forward pass on the activation tensor
        // 3. Return the output activations (or final tokens if last segment)
        //
        // For Phase 3 stub: use the executor to generate tokens
        let mut executor = self.shared_state.executor.lock().await;
        if executor.is_loaded() {
            let prompt = build_chat_prompt(&self.request.messages);
            let (content, gen_result) =
                executor.generate(&prompt, &self.request.sampling_params)?;

            let token_ids: Vec<u32> = content.bytes().map(|b| b as u32).collect();
            let finish = match gen_result.finish_reason {
                crate::inference::executor::FinishReason::Stop => Some(NetworkFinishReason::Stop),
                crate::inference::executor::FinishReason::MaxTokens => {
                    Some(NetworkFinishReason::MaxTokens)
                }
            };

            return Ok(LayerResult {
                request_id: self.request.id,
                token_ids,
                finish_reason: finish,
            });
        }

        // Stub: return a placeholder result
        Ok(LayerResult {
            request_id: self.request.id,
            token_ids: vec![72, 105], // "Hi"
            finish_reason: Some(NetworkFinishReason::Stop),
        })
    }

    /// Wait for a segment to return its result.
    async fn wait_for_segment_result(
        &self,
        request_id: uuid::Uuid,
        _segment_idx: usize,
        _is_last: bool,
    ) -> Result<LayerResult, SwarmError> {
        // In a full implementation, this would listen on a channel
        // for LayerResult messages matching (request_id, segment_idx).
        // For Phase 3: use a timeout-based stub that simulates network latency.

        tokio::time::sleep(Duration::from_millis(100)).await;

        // Stub: return placeholder tokens
        Ok(LayerResult {
            request_id,
            token_ids: vec![72, 105], // "Hi"
            finish_reason: Some(NetworkFinishReason::Stop),
        })
    }

    /// Attempt failover to a standby node for a failed segment.
    async fn failover_segment(
        &mut self,
        failed_idx: usize,
        request_id: uuid::Uuid,
        sequence_num: u32,
        activations: &[u8],
        is_last: bool,
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

                // Send to backup node via directed tensor protocol
                let forward = LayerForward {
                    request_id,
                    sequence_num,
                    activations: activations.to_vec(),
                    format: TensorFormat::FP16,
                };

                let target_peer_bytes = backup.node_id.0.to_vec();
                self.network_tx
                    .send(NetworkCommand::SendTensor {
                        target_peer_bytes,
                        forward,
                    })
                    .await
                    .map_err(|_| {
                        SwarmError::Network("Failed to send to standby node".to_string())
                    })?;

                // Wait for standby response
                self.wait_for_segment_result(request_id, failed_idx, is_last)
                    .await
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
        let (state, _) = SharedState::new(config, identity, db, executor);
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
        let result = executor.execute().await;
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
        let result = executor.execute().await;
        // Without a loaded model, this should return a stub result
        // The local path falls through to the stub
        assert!(result.is_err()); // NoModelLoaded
    }
}
