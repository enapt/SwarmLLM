//! Distributed inference pipeline. Orchestrates per-token generation across
//! pipeline segments (on-node + remote peers), with tensor-parallel, vision
//! pre-computation, streaming, and failover support.
//!
//! The `PipelineExecutor` struct lives here; per-phase methods live in sibling
//! files (`local`, `distributed`, `vision`, `prompt`, `tensor_parallel`).

mod distributed;
mod dsd;
mod local;
mod prompt;
mod remote_generate;
mod speculative;
mod tensor_parallel;
mod vision;

use std::sync::Arc;

use tokio::sync::mpsc;

use crate::daemon::SharedState;
use crate::error::SwarmError;
use crate::inference::router::{InferenceOutput, StreamingTokenTx, TokenLogProbEntry};
use crate::types::{
    InferenceError, InferenceRequest, ModelId, NetworkCommand, PipelineAssignment, SwarmMessage,
};

pub use prompt::template_from_header;

// Timeout constants for remote operations
const VISION_ENCODE_TIMEOUT_SECS: u64 = 120;
const PREFILL_SECS_PER_LAYER: u64 = 15;
const DECODE_SECS_PER_LAYER: u64 = 2;
const SEGMENT_TIMEOUT_MIN_SECS: u64 = 30;
const SEGMENT_TIMEOUT_MAX_SECS: u64 = 600;
/// Cap pending layer results to prevent OOM under sustained load.
const MAX_PENDING_LAYER_RESULTS: usize = 1024;

/// RAII guard that removes a `pending_layer_results` entry on drop unless
/// `disarm()` has been called first. Shared by every coordinator that
/// inserts a oneshot before awaiting a remote tensor result — without it
/// a bare `?` propagation from the wait site leaves a stale entry per
/// failed inference, eventually exhausting `MAX_PENDING_LAYER_RESULTS`.
/// Per gotcha #45 in `memory/MEMORY.md`.
pub(super) struct PendingLayerResultGuard<'a> {
    pub(super) map:
        &'a dashmap::DashMap<uuid::Uuid, tokio::sync::oneshot::Sender<crate::types::LayerResult>>,
    pub(super) id: uuid::Uuid,
    pub(super) armed: bool,
}
impl<'a> PendingLayerResultGuard<'a> {
    pub(super) fn new(
        map: &'a dashmap::DashMap<
            uuid::Uuid,
            tokio::sync::oneshot::Sender<crate::types::LayerResult>,
        >,
        id: uuid::Uuid,
    ) -> Self {
        Self {
            map,
            id,
            armed: true,
        }
    }
    pub(super) fn disarm(&mut self) {
        self.armed = false;
    }
}

/// Cap-check + insert + RAII-guard for a pending layer-result oneshot.
/// Shared by the canonical 3 sites (speculative prefill, speculative
/// verify batch, DSD verify) — `distributed.rs` keeps its inline form
/// because one branch needs `&mut self` access and another skips the
/// cap check during failover. Returns `ServiceUnavailable` when the
/// pending map is at `MAX_PENDING_LAYER_RESULTS`.
pub(super) fn register_pending_layer_result(
    map: &dashmap::DashMap<uuid::Uuid, tokio::sync::oneshot::Sender<crate::types::LayerResult>>,
    request_id: uuid::Uuid,
) -> Result<
    (
        tokio::sync::oneshot::Receiver<crate::types::LayerResult>,
        PendingLayerResultGuard<'_>,
    ),
    crate::error::SwarmError,
> {
    if map.len() >= MAX_PENDING_LAYER_RESULTS {
        return Err(crate::error::SwarmError::ServiceUnavailable(
            "Pipeline overloaded — too many pending layer results".into(),
        ));
    }
    let (tx, rx) = tokio::sync::oneshot::channel();
    map.insert(request_id, tx);
    let guard = PendingLayerResultGuard::new(map, request_id);
    Ok((rx, guard))
}
impl<'a> Drop for PendingLayerResultGuard<'a> {
    fn drop(&mut self) {
        if self.armed {
            self.map.remove(&self.id);
        }
    }
}
/// Fallback EOS token ID when GGUF metadata is unavailable. Matches LLaMA family;
/// other architectures (Qwen2, Phi-3, Gemma) have different EOS tokens.
/// A warning is emitted when this fallback is used.
pub(crate) const LLAMA_FALLBACK_EOS_TOKEN: u32 = 2;
pub(crate) const PREFILL_ACTIVATION_THRESHOLD_BYTES: usize = 100_000;

/// Pack a slice of u32 token IDs as i64 little-endian bytes — the
/// activation byte format the worker expects for first-segment
/// multi-token decode (DSD Phase 4 / Item 12). Shared by
/// `speculative.rs::send_verify_batch` and
/// `dsd.rs::forward_verify_through_segments` so a wire-format change
/// has a single source of truth.
pub(super) fn pack_verify_tokens_to_le_bytes(tokens: &[u32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(tokens.len() * 8);
    for &t in tokens {
        out.extend_from_slice(&(t as i64).to_le_bytes());
    }
    out
}

/// Build the `LayerForward` envelope for a speculative-verify send.
/// Shared by `speculative.rs::send_verify_batch` (single-segment) and
/// `dsd.rs::forward_verify_through_segments` (multi-segment) so adding
/// a `LayerForward` field can't drift between the two paths. The
/// `spec_logits_requested` flag is set uniformly; the receiver gates
/// emission on `is_last`.
pub(super) fn build_spec_verify_forward(
    request_id: uuid::Uuid,
    index_pos: u32,
    activations: Vec<u8>,
    segment: &crate::types::PipelineSegment,
    requester_node_id_bytes: [u8; 32],
    truncate_kv_to: Option<u32>,
) -> crate::types::LayerForward {
    crate::types::LayerForward {
        request_id,
        sequence_num: 1, // not prefill
        index_pos,
        activations,
        format: crate::types::TensorFormat::FP32,
        model_id: segment.shard_id.model_id.clone(),
        layer_range: segment.layer_range,
        vision_embeddings: None,
        sender_peer_bytes: None,
        tp_meta: None,
        requester_node_id: Some(requester_node_id_bytes),
        pre_embedded: false,
        generated_ids: Vec::new(),
        adapter_id: None,
        // The receiver gates spec-logits emission on
        // `spec_logits_requested && is_last`, not on `draft_tokens`. The
        // draft IDs are also already encoded in `activations`, so leaving
        // this empty saves an allocation per spec round at no cost.
        draft_tokens: Vec::new(),
        spec_logits_requested: true,
        truncate_kv_to,
    }
}

/// Build the `LayerForward` envelope for a stop-sequence KV-truncate
/// signal sent to a remote segment. Empty activations + no compute,
/// `truncate_kv_to: Some(truncate_to)` is the only signal — the receiver
/// trims its KV cache and acknowledges. Shares the same field order as
/// `build_spec_verify_forward` so a `LayerForward` field addition has a
/// single source of truth in this module.
pub(super) fn build_kv_truncate_forward(
    request_id: uuid::Uuid,
    segment: &crate::types::PipelineSegment,
    truncate_to: u32,
    requester_node_id_bytes: [u8; 32],
) -> crate::types::LayerForward {
    crate::types::LayerForward {
        request_id,
        sequence_num: 1, // not prefill
        index_pos: truncate_to,
        activations: Vec::new(), // truncate-only, no compute
        format: crate::types::TensorFormat::FP32,
        model_id: segment.shard_id.model_id.clone(),
        layer_range: segment.layer_range,
        vision_embeddings: None,
        sender_peer_bytes: None,
        tp_meta: None,
        requester_node_id: Some(requester_node_id_bytes),
        pre_embedded: false,
        generated_ids: Vec::new(),
        adapter_id: None,
        draft_tokens: Vec::new(),
        spec_logits_requested: false,
        truncate_kv_to: Some(truncate_to),
    }
}

/// Request-level disqualifiers shared by every "fast path" coordinator:
/// remote-generate (`remote_generate.rs`), distributed-speculative
/// (`speculative.rs`), and DSD multi-segment (`dsd.rs`). Returns
/// `true` when ANY disqualifier applies — callers short-circuit their
/// own `eligible()` check on a true return.
///
/// Each path additionally has its own segment-shape and flag-config
/// preconditions; those stay in the per-path `eligible()` because the
/// shapes are subtly divergent (1-segment / 2+-segment / per-model
/// encrypted-pipeline gate).
pub(super) fn fastpath_request_disqualified(exec: &PipelineExecutor) -> bool {
    if !exec.assignment.tp_groups.is_empty() {
        return true;
    }
    if exec.request.lora_adapter.is_some() {
        return true;
    }
    if !crate::inference::vision::collect_images(&exec.request.messages).is_empty() {
        return true;
    }
    false
}

/// Common preconditions for any speculative-decoding fast path
/// (`speculative.rs`'s single-segment Item 2 and `dsd.rs`'s
/// multi-segment Item 12). Returns `true` when the request is
/// eligible *so far* — callers add their own segment-shape check on
/// top. Greedy temperature, draft model availability, and the
/// non-encryption / non-LoRA / non-vision baseline are required by
/// both paths; the bool flag config is per-path so it stays inline.
pub(super) fn speculative_common_eligible(exec: &PipelineExecutor) -> bool {
    let cfg = &exec.shared_state.config.inference;
    if !cfg.speculative_decoding {
        return false;
    }
    if !exec.assignment.supports_speculative {
        return false;
    }
    if fastpath_request_disqualified(exec) {
        return false;
    }
    if exec.request.sampling_params.temperature != 0.0 {
        return false;
    }
    if cfg.draft_model_path.is_none() {
        return false;
    }
    true
}

/// Executes a distributed inference pipeline across multiple nodes.
///
/// The pipeline is a sequence of segments, each assigned to a node.
/// Activation tensors are forwarded between nodes in order.
/// If a segment fails, the executor attempts failover to a standby node.
pub struct PipelineExecutor {
    pub(super) shared_state: Arc<SharedState>,
    pub(super) network_tx: mpsc::Sender<NetworkCommand>,
    pub(super) request: InferenceRequest,
    pub(super) assignment: PipelineAssignment,
    /// Collected per-token logprobs during token generation.
    /// Uses Mutex because process_local_segment takes &self.
    pub(super) collected_logprobs: std::sync::Mutex<Vec<TokenLogProbEntry>>,
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
    pub(super) fn ensure_split_model_entry(
        &self,
        model_id: &ModelId,
        layer_start: usize,
        layer_end: usize,
    ) -> Result<(ModelId, usize, usize), SwarmError> {
        let manifest = self
            .shared_state
            .model_registry
            .get_manifest(model_id)
            .ok_or_else(|| SwarmError::ModelNotAvailable(model_id.clone()))?;
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
        let (is_first, is_last) = crate::model::shard::compute_first_last(
            &local_shards,
            manifest.shard_count,
            layer_start,
            layer_end,
            total_layers,
        );

        let split_key = self.shared_state.ensure_split_model_entry(
            model_id,
            layer_start,
            layer_end,
            is_first,
            is_last,
            total_layers,
        );
        Ok(split_key)
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
}

/// Broadcast a pipeline error to the network so peers can update shard availability.
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
            cancel: None,
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
            supports_speculative: false,
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
            supports_speculative: false,
        };

        let mut executor = PipelineExecutor::new(state, tx, request, assignment);
        let result = executor.execute(None).await;
        // Without a loaded model, this should return a stub result
        // The local path falls through to the stub
        assert!(result.is_err()); // NoModelLoaded
    }
}
