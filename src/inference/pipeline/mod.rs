//! Distributed inference pipeline. Orchestrates per-token generation across
//! pipeline segments (on-node + remote peers), with tensor-parallel, vision
//! pre-computation, streaming, and failover support.
//!
//! The `PipelineExecutor` struct lives here; per-phase methods live in sibling
//! files (`local`, `distributed`, `vision`, `prompt`, `tensor_parallel`).

mod distributed;
mod dsd;
mod hedge_dispatch;
mod local;
mod ngram_only_spec;
mod prompt;
pub(crate) mod remote_generate;
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

/// Does this remote error mean the holder does not have the shard data it
/// advertised?
///
/// A holder's claim is gossiped, so our registry can be stale — a peer that has
/// deleted or pruned a shard keeps receiving requests for it until its retraction
/// announcement reaches us (`delete_shard` re-announces immediately with
/// `complete_for_models`, but that is a GossipSub broadcast and a NAT'd peer may
/// not get it promptly). Recognising the failure lets us drop the stale claim
/// locally and re-route on the spot, instead of failing every request until the
/// next announcement lands.
///
/// Deliberately narrow. This must match ONLY "you asked me for bytes I do not
/// have", never a transient or compute failure — dropping a healthy holder's
/// claim would push work off a good peer and, at worst, empty the holder set for
/// a shard nobody else has. `ShardReader`'s missing-region error is the
/// authoritative signal: it is raised when a byte range maps to a shard file the
/// node does not hold. Observed live 2026-07-26 as
/// `blk.0.attn_q: ShardReader: position 345977248 is in a missing region`.
pub(crate) fn remote_error_means_missing_shard(msg: &str) -> bool {
    // Both spellings appear: the reader's own message, and `ShardNotFound`
    // rendered through `Display` when a load never got as far as reading.
    msg.contains("is in a missing region")
        || msg.contains("missing shard region")
        || msg.contains("Shard not found")
}

/// Pack a slice of u32 token IDs as i64 little-endian bytes — the
/// activation byte format the worker expects for first-segment
/// multi-token decode (DSD Phase 4 / Item 12). Shared by
/// `speculative.rs::send_verify_batch` and
/// `super::forward_verify_through_segments` (multi-segment) so a wire-format change
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
/// `super::forward_verify_through_segments` (multi-segment) so adding
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
        chunk_meta: None,
    }
}

/// Propagate a γ-token verify request through every pipeline segment in
/// order. First segment receives `verify_tokens` encoded as i64 LE bytes
/// (`8 × verify_tokens.len()` bytes). Intermediate segments receive the
/// previous segment's `[1, γ, hidden]` activations. The final segment
/// returns γ+1 logit vectors via `LayerResult.spec_logits`.
///
/// Local segments (`peer_id_for_segment[idx] == None`) dispatch directly
/// to the local `model_process_pool` worker.
///
/// Shared by DSD multi-segment spec (`dsd.rs::try_dsd_distributed`)
/// and SWARM-SPEC L1 n-gram-only spec (`ngram_only_spec.rs`). Extracted
/// from dsd.rs (R136 Layer 1 multi-segment) so it's available without
/// the `llama` feature gate.
#[allow(clippy::too_many_arguments)]
pub(super) async fn forward_verify_through_segments(
    shared_state: &Arc<SharedState>,
    network_tx: &mpsc::Sender<NetworkCommand>,
    request_id: uuid::Uuid,
    index_pos: u32,
    segments: &[crate::types::PipelineSegment],
    peer_id_for_segment: &[Option<Vec<u8>>],
    verify_tokens: &[u32],
    truncate_kv_to: Option<u32>,
) -> Result<Vec<Vec<f32>>, SwarmError> {
    let num_segments = segments.len();
    debug_assert_eq!(num_segments, peer_id_for_segment.len());

    let mut activation_bytes: Vec<u8> = pack_verify_tokens_to_le_bytes(verify_tokens);

    for (idx, segment) in segments.iter().enumerate() {
        let is_last = idx == num_segments - 1;
        let target_peer_bytes = &peer_id_for_segment[idx];

        let forward = build_spec_verify_forward(
            request_id,
            index_pos,
            activation_bytes.clone(),
            segment,
            shared_state.identity.node_id().0,
            truncate_kv_to,
        );

        let result = if let Some(peer_bytes) = target_peer_bytes {
            let (rx, mut pending_guard) =
                register_pending_layer_result(&shared_state.pending_layer_results, request_id)?;

            if network_tx
                .send(NetworkCommand::SendTensor {
                    target_peer_bytes: peer_bytes.clone(),
                    forward,
                })
                .await
                .is_err()
            {
                // Disarm BEFORE the inline remove so the guard's Drop
                // doesn't double-remove on return (the guard would
                // otherwise fire its own remove when this stack frame
                // unwinds). Exposed by L2 hedging which doubles the
                // call rate through this function.
                pending_guard.disarm();
                shared_state.pending_layer_results.remove(&request_id);
                return Err(SwarmError::Network(
                    "fwd_verify_through_segments: send dropped".into(),
                ));
            }

            let num_layers = segment.layer_range.1 - segment.layer_range.0;
            let result = PipelineExecutor::wait_for_result(
                rx,
                request_id,
                idx,
                &segment.node_id,
                num_layers,
                activation_bytes.len(),
            )
            .await?;
            pending_guard.disarm();
            result
        } else {
            shared_state.model_process_pool.forward(forward).await?
        };

        if let Some(crate::types::NetworkFinishReason::Error(msg)) = &result.finish_reason {
            return Err(SwarmError::Inference(format!(
                "spec verify segment {idx}: {msg}"
            )));
        }

        if is_last {
            if result.spec_logits.is_empty() {
                return Err(SwarmError::Inference(
                    "spec verify: last segment returned no spec_logits".into(),
                ));
            }
            return Ok(result.spec_logits);
        }

        // Intermediate: feed hidden state to next segment. SEC: validate
        // intermediate-to-intermediate shape preservation. The first
        // segment's input is token IDs (8 bytes/position via
        // pack_verify_tokens_to_le_bytes) but its OUTPUT is the hidden
        // state (hidden_dim × bytes_per_elem per position) — those don't
        // match, so equality only applies for idx >= 1.
        //
        // For idx == 0 we instead apply an absolute upper-bound sanity
        // check. The wire-level MAX_ACTIVATION_SIZE (128 MB at
        // network/protocol/mod.rs) already caps malicious sends, but
        // the per-segment hidden-state activation should be
        // (γ+1) × hidden_dim × bytes/elem — well under 64 MB even for
        // huge models with γ=64 and hidden_dim=12288 fp32. A larger
        // response from segment 0 indicates a broken / malicious peer
        // and would crash the next worker if forwarded.
        const MAX_INTERMEDIATE_ACTIVATION_BYTES: usize = 64 * 1024 * 1024;
        if idx == 0 && result.activations.len() > MAX_INTERMEDIATE_ACTIVATION_BYTES {
            return Err(SwarmError::Inference(format!(
                "spec verify segment 0 returned oversized activation: {} bytes (max {})",
                result.activations.len(),
                MAX_INTERMEDIATE_ACTIVATION_BYTES
            )));
        }
        if idx > 0 && result.activations.len() != activation_bytes.len() {
            return Err(SwarmError::Inference(format!(
                "spec verify segment {idx} returned wrong activation shape: got {} bytes, expected {}",
                result.activations.len(),
                activation_bytes.len()
            )));
        }
        activation_bytes = result.activations;
    }
    unreachable!("loop returns on the last segment")
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
        chunk_meta: None,
    }
}

/// Send a single token's decoded text down the streaming channel,
/// ignoring any send error (matches the "first-token" pattern that
/// fires before the per-round loop). Used by speculative / DSD /
/// ngram-only-spec where the first token is always emitted
/// fire-and-forget — disconnection at this point is fine, the per-round
/// loop will detect it on the next emit.
pub(in crate::inference::pipeline) async fn emit_first_streaming_token(
    token_tx: &Option<StreamingTokenTx>,
    decoder: &prompt::CachedDecoder,
    token: u32,
) {
    if let Some(tx) = token_tx {
        let text = decoder.decode_tokens(&[token]);
        let _ = tx
            .send(crate::inference::router::StreamingTokenEvent {
                text,
                finish_reason: None,
                matched_stop_sequence: None,
            })
            .await;
    }
}

/// Emit a slice of accepted tokens to the streaming channel. On a
/// channel-closed error (client disconnect), stamp `finish_reason =
/// "stop"` and break. Returns whether a disconnect happened so the
/// caller can bail out of further bookkeeping. Shared by
/// speculative / DSD / ngram-only-spec.
pub(in crate::inference::pipeline) async fn emit_streaming_batch(
    token_tx: &Option<StreamingTokenTx>,
    decoder: &prompt::CachedDecoder,
    tokens: &[u32],
    finish_reason: &mut String,
) -> bool {
    let tx = match token_tx {
        Some(tx) => tx,
        None => return false,
    };
    for &t in tokens {
        let text = decoder.decode_tokens(&[t]);
        if tx
            .send(crate::inference::router::StreamingTokenEvent {
                text,
                finish_reason: None,
                matched_stop_sequence: None,
            })
            .await
            .is_err()
        {
            *finish_reason = "stop".to_string();
            return true;
        }
    }
    false
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

    /// Resolve a `peer_id_bytes` for each segment in `self.assignment`.
    /// `None` for local segments (dispatched to the worker subprocess),
    /// `Some(bytes)` for remote segments. Returns `Ok(None)` when any
    /// remote segment can't be resolved — the caller (DSD, ngram-only
    /// spec, …) treats that as "fall back to the standard distributed
    /// loop" without surfacing a hard error. `label` flows into the
    /// debug log for fall-back attribution.
    pub(super) fn resolve_peer_id_for_segments(
        &self,
        request_id: uuid::Uuid,
        label: &str,
    ) -> Option<Vec<Option<Vec<u8>>>> {
        let local_node_id = self.shared_state.identity.node_id().clone();
        let mut out: Vec<Option<Vec<u8>>> = Vec::with_capacity(self.assignment.segments.len());
        for segment in &self.assignment.segments {
            if segment.node_id == local_node_id {
                out.push(None);
                continue;
            }
            match self.shared_state.resolve_peer_id_bytes(&segment.node_id) {
                Some(p) => out.push(Some(p)),
                None => {
                    tracing::debug!(
                        %request_id,
                        node = %segment.node_id,
                        "{}: missing peer_id_bytes — falling back",
                        label
                    );
                    return None;
                }
            }
        }
        Some(out)
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

        // Check if this is a single-node pipeline on the local node AND the
        // llama-cpp executor holds THE REQUESTED model (full GGUF path).
        // If the model was loaded from shards (auto-manage), the executor
        // won't be loaded — fall through to execute_distributed which uses
        // the split model via process_local_segment.
        //
        // The model check is not optional: `execute_local` generates from the
        // singleton executor without consulting `request.model_id`, so testing
        // only the global `model_loaded` flag here served requests for other
        // models from whichever one was resident. Same defect, same fix as
        // `router::batch::execute_batch`.
        if num_segments == 1
            && self.assignment.segments[0].node_id == *self.shared_state.identity.node_id()
            && self
                .shared_state
                .local_executor_serves(&self.request.model_id)
                .await
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

    /// `model_loaded` is global — "a model is loaded", not "this one". Both
    /// local fast paths (`pipeline::execute` and `router::batch::execute_batch`)
    /// generate from the singleton executor WITHOUT consulting
    /// `request.model_id`, so gating them on the bare flag answered a request
    /// for one model with a different model's weights and chat template, and
    /// reported it as a success. These pin the identity check that replaced it.
    #[tokio::test]
    async fn local_executor_serves_requires_the_flag() {
        let state = make_test_state();
        // Nothing loaded: the flag is false and no info is cached.
        assert!(
            !state
                .local_executor_serves(&ModelId("anything".into()))
                .await
        );
    }

    #[tokio::test]
    async fn local_executor_serves_matches_name_and_slug_only() {
        let state = make_test_state();
        *state.loaded_model_info.write().await = Some(crate::daemon::state::LoadedModelInfo {
            name: "Qwen2.5 Coder 7B Instruct".into(),
            size_bytes: 0,
            eos_tokens: vec![],
            chat_template: None,
            bos_token: String::new(),
            eos_token: String::new(),
        });
        state
            .model_loaded
            .store(true, std::sync::atomic::Ordering::Release);

        // The display name and its slug are both legitimate spellings of the
        // resident model — `resolve_model_for_inference` can hand the router
        // either one.
        assert!(
            state
                .local_executor_serves(&ModelId("Qwen2.5 Coder 7B Instruct".into()))
                .await
        );
        assert!(
            state
                .local_executor_serves(&ModelId(crate::types::slugify_model_name(
                    "Qwen2.5 Coder 7B Instruct"
                )))
                .await
        );

        // A DIFFERENT model must not be served by this executor, however
        // plausible the name. This is the whole point: before the fix every
        // one of these was answered with Qwen's weights.
        for other in [
            "tinyllama-1.1b-chat-v1.0.q4-k-m",
            "llama-3.2-3b-instruct-q4-k-m",
            "Qwen2.5 Coder 14B Instruct",
            "qwen2.5-coder-7b",
            "",
        ] {
            assert!(
                !state.local_executor_serves(&ModelId(other.into())).await,
                "{other} must not be served by the resident Qwen executor"
            );
        }
    }

    /// `is_shard_in_vram`'s legacy fallback matched with `contains`, so a node
    /// with "Llama 3.2" resident reported every `llama-3.2-*` id as being in
    /// VRAM — other quantisations, other parameter counts, any id sharing the
    /// prefix. It now shares the dispatch path's exact rule.
    #[tokio::test]
    async fn shard_in_vram_does_not_substring_match_the_loaded_model() {
        let state = make_test_state();
        *state.loaded_model_info.write().await = Some(crate::daemon::state::LoadedModelInfo {
            name: "Llama 3.2".into(),
            size_bytes: 0,
            eos_tokens: vec![],
            chat_template: None,
            bos_token: String::new(),
            eos_token: String::new(),
        });
        state
            .model_loaded
            .store(true, std::sync::atomic::Ordering::Release);

        // Nothing is registered and no split segment is loaded, so the legacy
        // fallback is the only thing that can answer here.
        for other in [
            "llama-3.2-3b-instruct-q4-k-m",
            "llama-3.2-1b-instruct-q8-0",
            "llama-3.2-90b-vision",
        ] {
            assert!(
                !state.is_shard_in_vram(&ModelId(other.into()), 0),
                "{other} shares a prefix with the resident model but is not it"
            );
        }
        // The resident model itself still reports correctly, by slug.
        assert!(state.is_shard_in_vram(&ModelId("llama-3.2".into()), 0));
    }

    /// The flag can be true while the cached info is absent (unload clears the
    /// info under a separate lock). Say no rather than guessing.
    #[tokio::test]
    async fn local_executor_serves_says_no_without_model_info() {
        let state = make_test_state();
        state
            .model_loaded
            .store(true, std::sync::atomic::Ordering::Release);
        assert!(state.loaded_model_info.read().await.is_none());
        assert!(
            !state
                .local_executor_serves(&ModelId("anything".into()))
                .await
        );
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

    /// R137 (closes R136 test-coverage deferral, partial): the wire-format
    /// helpers `pack_verify_tokens_to_le_bytes`, `build_spec_verify_forward`,
    /// and `build_kv_truncate_forward` are pure and unit-testable. Full
    /// `forward_verify_through_segments` orchestration still needs worker
    /// subprocess infrastructure, but the building blocks now have direct
    /// coverage so a wire-format drift fails a fast test before integration.
    /// Retracting a holder's shard claim pushes work off that peer and can empty
    /// a shard's holder set, so this classifier must fire on "I don't have that
    /// data" and on nothing else. Both directions are pinned.
    #[test]
    fn missing_shard_classifier_matches_the_real_reader_errors() {
        // Observed live 2026-07-26 on a node deliberately holding only the tail.
        assert!(remote_error_means_missing_shard(
            "Inference error: Internal error: blk.0.attn_q: ShardReader: position 345977248 is in a missing region"
        ));
        // The output-head spelling from the same reader.
        assert!(remote_error_means_missing_shard(
            "Worker: Inference error: Internal error: Failed to load output head: ShardReader: position 7827872 is in a missing region (total_size=1321079200)"
        ));
        assert!(remote_error_means_missing_shard(
            "ShardReader: position is in a missing shard region"
        ));
        assert!(remote_error_means_missing_shard("Shard not found: model/2"));
    }

    #[test]
    fn missing_shard_classifier_ignores_everything_else() {
        // A healthy holder must never lose its claim over a transient or
        // compute failure — that would move work off a good peer.
        for msg in [
            "peer never acknowledged the request",
            "remote-generate timed out after 120s",
            "OutboundFailure: DialFailure",
            "Worker: Inference error: CUDA out of memory",
            "Pipeline assembly failed: Segment 1 failed with no standby available",
            "Timed out waiting for segment result (30s, 6 layers)",
            "decrypt FAILED — possible AAD mismatch",
            "",
        ] {
            assert!(
                !remote_error_means_missing_shard(msg),
                "must not retract a holder for: {msg}"
            );
        }
    }

    #[test]
    fn pack_verify_tokens_to_le_bytes_packs_i64_le() {
        // Empty input → empty output (no allocator panic).
        assert!(pack_verify_tokens_to_le_bytes(&[]).is_empty());
        // u32 → i64 widening preserves value; LE means the low byte is first.
        let bytes = pack_verify_tokens_to_le_bytes(&[1u32, 0xFFFFFFFFu32]);
        assert_eq!(bytes.len(), 16);
        assert_eq!(&bytes[..8], &[1, 0, 0, 0, 0, 0, 0, 0]);
        // 0xFFFFFFFF widened to i64 is 0x00000000FFFFFFFF (positive — u32 is
        // unsigned). Verifies we're using `as i64` (zero-extend) not a signed
        // re-interpret that would produce 0xFFFFFFFFFFFFFFFF.
        assert_eq!(&bytes[8..], &[0xFF, 0xFF, 0xFF, 0xFF, 0, 0, 0, 0]);
    }

    #[test]
    fn build_spec_verify_forward_carries_requested_fields() {
        let request_id = uuid::Uuid::new_v4();
        let segment = PipelineSegment {
            node_id: NodeId([7u8; 32]),
            shard_id: ShardId {
                model_id: ModelId("test-model".into()),
                index: 3,
            },
            layer_range: (4, 8),
        };
        let activations = vec![1u8, 2, 3, 4];
        let requester = [9u8; 32];
        let fwd = build_spec_verify_forward(
            request_id,
            42,
            activations.clone(),
            &segment,
            requester,
            Some(100),
        );
        assert_eq!(fwd.request_id, request_id);
        assert_eq!(fwd.index_pos, 42);
        assert_eq!(fwd.sequence_num, 1, "spec verify is never prefill");
        assert_eq!(fwd.activations, activations);
        assert!(matches!(fwd.format, TensorFormat::FP32));
        assert_eq!(fwd.model_id.0, "test-model");
        assert_eq!(fwd.layer_range, (4, 8));
        assert!(fwd.vision_embeddings.is_none());
        assert!(fwd.sender_peer_bytes.is_none());
        assert!(fwd.tp_meta.is_none());
        assert_eq!(fwd.requester_node_id, Some(requester));
        assert!(!fwd.pre_embedded);
        assert!(fwd.generated_ids.is_empty());
        assert!(fwd.adapter_id.is_none());
        assert!(
            fwd.draft_tokens.is_empty(),
            "spec verify packs draft tokens in activations, not draft_tokens"
        );
        assert!(
            fwd.spec_logits_requested,
            "spec verify always sets the flag"
        );
        assert_eq!(fwd.truncate_kv_to, Some(100));
    }

    #[test]
    fn build_kv_truncate_forward_uniquely_identified_by_empty_activations() {
        let request_id = uuid::Uuid::new_v4();
        let segment = PipelineSegment {
            node_id: NodeId([3u8; 32]),
            shard_id: ShardId {
                model_id: ModelId("trunc-model".into()),
                index: 1,
            },
            layer_range: (0, 4),
        };
        let requester = [5u8; 32];
        let fwd = build_kv_truncate_forward(request_id, &segment, 50, requester);
        // Three invariants the receiver uses to identify a truncate-only:
        assert!(
            fwd.activations.is_empty(),
            "truncate signals MUST carry no compute payload"
        );
        assert!(
            !fwd.spec_logits_requested,
            "truncate signals MUST NOT request spec logits"
        );
        assert_eq!(fwd.truncate_kv_to, Some(50), "truncate target MUST be set");
        // And the index_pos carries the truncation point (the receiver uses
        // this to size its retain window).
        assert_eq!(fwd.index_pos, 50);
    }

    /// R137 (partial closure of R136 test deferral): the network-send-failure
    /// arm in `forward_verify_through_segments` disarms the
    /// `PendingLayerResultGuard` AND removes the pending entry inline so the
    /// guard's Drop doesn't double-remove. Verify the failure path without
    /// needing a worker subprocess — close the network_tx side, dispatch a
    /// remote segment, and assert the error surfaces + pending_layer_results
    /// is clean post-call.
    #[tokio::test]
    async fn forward_verify_through_segments_disarms_guard_on_network_drop() {
        let state = make_test_state();
        let (tx, rx) = mpsc::channel::<NetworkCommand>(64);
        drop(rx); // close the receive side so any send fails.

        let request_id = uuid::Uuid::new_v4();
        // Single remote segment pointing at a fake peer; the send will fail
        // synchronously because the rx is closed.
        let segments = vec![PipelineSegment {
            node_id: NodeId([42u8; 32]),
            shard_id: ShardId {
                model_id: ModelId("test-model".into()),
                index: 0,
            },
            layer_range: (0, 8),
        }];
        let peer_id_for_segment: Vec<Option<Vec<u8>>> = vec![Some(vec![1, 2, 3, 4])];
        let verify_tokens: Vec<u32> = vec![100, 101, 102];

        let result = forward_verify_through_segments(
            &state,
            &tx,
            request_id,
            10,
            &segments,
            &peer_id_for_segment,
            &verify_tokens,
            None,
        )
        .await;
        assert!(result.is_err(), "closed-channel send must surface as Err");
        // CRITICAL: pending_layer_results must be empty — proves the guard
        // disarm + inline remove worked. A double-remove would not break
        // this assertion but a missed remove WOULD leave an entry behind,
        // leaking the oneshot. We check the leak direction.
        assert!(
            state.pending_layer_results.is_empty(),
            "pending_layer_results leak after network-drop failure path"
        );
    }
}
