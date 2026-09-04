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
    InferenceError, InferenceRequest, LayerResult, ModelId, NetworkCommand, NetworkFinishReason,
    PipelineAssignment, SwarmMessage,
};

pub use prompt::template_from_header;

/// Recover a peer's failure from a completed [`LayerResult`].
///
/// A serving node reports why it refused a forward in
/// `finish_reason: NetworkFinishReason::Error(msg)`, and
/// [`LayerResult::error`] leaves `token_ids` empty when it does. A caller that
/// tests only `token_ids.is_empty()` therefore throws the peer's stated reason
/// away and substitutes one of its own — which is how the SAME over-long prompt
/// came back as `400 invalid_request_error` with an actionable message when the
/// model was local, and `500 server_error: ngram-only: prefill returned no
/// tokens` when it was reached over the network (measured on the live swarm
/// 2026-08-30). That is gotcha #304's shape: whose fault a mistake was decided
/// by which machine happened to hold the model.
///
/// It also mis-attributes blame. `failure_is_penalty_worthy` exempts
/// `Validation`, but never sees one when the class has been flattened into
/// `Inference`, so the peer is docked a serve-failure penalty for the caller's
/// mistake.
///
/// Returns `None` when the result carries no error, so a caller keeps its own
/// "the peer returned nothing and said nothing" message for the genuinely
/// silent case.
/// Would every other holder refuse this forward in exactly the same way?
///
/// A peer's `Validation` describes the REQUEST — a prompt longer than the
/// model's context, a malformed argument — so every machine holding that model
/// reproduces it identically. Failing over then spends a round trip per standby
/// collecting the same refusal, and when the standbys run out the caller is
/// handed `SegmentFailoverExhausted`, whose hint says this model "is being held
/// by too few machines right now" and suggests `swarmllm get-model`. That
/// advice cannot work: the prompt is simply too long, and fetching the model
/// changes nothing. Measured on the live swarm 2026-08-30, where a 12041-token
/// prompt against an 8192-token model produced exactly that 503.
///
/// This is the retry half of gotcha #295 — a permanent failure given advice to
/// retry — and the reason the check belongs at the failover DECISION rather
/// than at the point the error is finally rendered.
///
/// Deliberately narrow: only `Validation` is provably the caller's own input.
/// A missing shard or an unresponsive worker says nothing about the next
/// holder, and those must still fail over.
pub(super) fn every_holder_would_refuse(err_msg: &str) -> Option<SwarmError> {
    match crate::error::reclassify_flattened_error(err_msg) {
        Some(e @ SwarmError::Validation(_)) => Some(e),
        _ => None,
    }
}

pub(super) fn peer_error_from_result(result: &LayerResult) -> Option<SwarmError> {
    match &result.finish_reason {
        Some(NetworkFinishReason::Error(msg)) => Some(
            crate::error::reclassify_flattened_error(msg)
                .unwrap_or_else(|| SwarmError::Inference(msg.clone())),
        ),
        _ => None,
    }
}

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
    pub(super) map: &'a dashmap::DashMap<uuid::Uuid, crate::daemon::state::PendingLayerResult>,
    pub(super) id: uuid::Uuid,
    pub(super) armed: bool,
}
impl<'a> PendingLayerResultGuard<'a> {
    pub(super) fn new(
        map: &'a dashmap::DashMap<uuid::Uuid, crate::daemon::state::PendingLayerResult>,
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
///
/// `awaiting` pins the waiter to the node the forward is being sent to, so a
/// late notification about a DIFFERENT node's forward for the same
/// `request_id` cannot resolve it. Pass `Some(node)` whenever the target is
/// known; `None` accepts a result from any sender.
pub(super) fn register_pending_layer_result(
    map: &dashmap::DashMap<uuid::Uuid, crate::daemon::state::PendingLayerResult>,
    request_id: uuid::Uuid,
    awaiting: Option<crate::types::NodeId>,
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
    map.insert(
        request_id,
        crate::daemon::state::PendingLayerResult {
            tx,
            awaiting,
            // This helper serves the speculative and DSD paths, which never
            // chain — they build their own forwards and drive them per token.
            chain_members: Vec::new(),
        },
    );
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
pub(crate) const PREFILL_ACTIVATION_THRESHOLD_BYTES: usize = 100_000;

/// Which half of inference a forward represents, from its sequence number.
///
/// `sequence_num == 0` is the prompt pass; every later one is a single-token
/// decode step (`speculative.rs` spells this out at its own construction site
/// with `sequence_num: 1, // not prefill`). This is authoritative, unlike the
/// `PREFILL_ACTIVATION_THRESHOLD_BYTES` size heuristic, which misclassifies a
/// short prompt on a narrow model. Prefill and decode cost ~2 orders of
/// magnitude apart, so anything sizing a budget from them must not confuse the
/// two — see `daemon::state::peer_speed`.
pub(super) fn work_kind_for(sequence_num: u32) -> crate::daemon::state::WorkKind {
    if sequence_num == 0 {
        crate::daemon::state::WorkKind::Prefill
    } else {
        crate::daemon::state::WorkKind::Decode
    }
}

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
        // The remote-generate fast path's own refusal: the peer checked
        // `can_serve_layer_range` for the WHOLE model and said no. That is the
        // same fact — our holder record for it is stale — arriving one hop
        // earlier, before any bytes were asked for. Until 2026-09-01 it matched
        // nothing here, so the claim stayed, nothing was retried, and a
        // request for a model four other peers held failed on the strength of
        // one peer's honest "not me" (gotcha #433, measured live: a 14B with
        // five candidates answered `Inference error: model not hosted on
        // target` in 0.8 s while the non-streaming sibling a minute earlier
        // had been served by a two-node chain).
        || msg.contains(crate::daemon::dispatch::remote_generate::REMOTE_GENERATE_NOT_HOSTED)
        // What a peer ACTUALLY sends. `sanitize_peer_facing_error` replaces
        // every shard-related error with this before it leaves the serving
        // node, so the wordings above — which are what the local code produces
        // — never reach a coordinator over the network. Without this arm the
        // detector was blind to the only form it ever sees.
        || msg.contains(crate::daemon::dispatch::layer_forward::PEER_FACING_MISSING_SHARDS)
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
    _requester_node_id_bytes: [u8; 32],
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
        chain: Vec::new(),
        sender_peer_bytes: None,
        tp_meta: None,
        // Unchained: the receiver answers its sender, which is us. Since
        // 2026-08-21 a `Some` here would put the 0x07 reply-to trailer on the
        // wire, which only chained runs carry and older peers do not expect.
        // (The seal this once fed never ran for a remote segment — the id was
        // not on the wire — so nothing is lost; `docs/FUTURE_WORK.md`.)
        requester_node_id: None,
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
        sampling: None,
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
/// Run one speculative-verify round through the pipeline's segments.
///
/// **The target peer is resolved HERE, from `segment.node_id`, on every send.**
/// It used to be passed in as a parallel `peer_id_for_segment` array that the
/// caller resolved once, before its decode loop — and mid-request failover
/// rewrites `assignment.segments[i].node_id` in place
/// (`distributed.rs`, "failing over to standby node"). The array then still
/// named the FAILED node while `register_pending_layer_result` below pinned the
/// waiter to the new one, so every subsequent round sent the work to the node
/// that had just failed and waited for an answer from a node that was never
/// asked. The abandoned node's reply arrived and was correctly discarded as
/// "from a node this request is no longer waiting on", and the request then sat
/// until its segment timeout — 284s, measured live 2026-08-04, on a request
/// whose first token had already succeeded via failover in 243ms.
///
/// Deriving both the send target and the `awaiting` pin from the same
/// `segment.node_id` makes that disagreement unrepresentable.
pub(super) async fn forward_verify_through_segments(
    shared_state: &Arc<SharedState>,
    network_tx: &mpsc::Sender<NetworkCommand>,
    request_id: uuid::Uuid,
    index_pos: u32,
    segments: &[crate::types::PipelineSegment],
    verify_tokens: &[u32],
    truncate_kv_to: Option<u32>,
) -> Result<Vec<Vec<f32>>, SwarmError> {
    let num_segments = segments.len();
    let local_node_id = shared_state.identity.node_id().clone();

    let mut activation_bytes: Vec<u8> = pack_verify_tokens_to_le_bytes(verify_tokens);

    for (idx, segment) in segments.iter().enumerate() {
        let is_last = idx == num_segments - 1;
        // Resolved fresh from the CURRENT assignment, never cached across rounds.
        let target_peer_bytes: Option<Vec<u8>> = if segment.node_id == local_node_id {
            None
        } else {
            match shared_state.resolve_peer_id_bytes(&segment.node_id) {
                Some(p) => Some(p),
                None => {
                    return Err(SwarmError::Inference(format!(
                        "verify round: no route to segment holder {} — it left mid-request",
                        segment.node_id
                    )));
                }
            }
        };
        let target_peer_bytes = target_peer_bytes.as_ref();

        let forward = build_spec_verify_forward(
            request_id,
            index_pos,
            activation_bytes.clone(),
            segment,
            shared_state.identity.node_id().0,
            truncate_kv_to,
        );

        let result = if let Some(peer_bytes) = target_peer_bytes {
            let (rx, mut pending_guard) = register_pending_layer_result(
                &shared_state.pending_layer_results,
                request_id,
                Some(segment.node_id.clone()),
            )?;

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
            let budget = local::SegmentBudget::for_forward(
                shared_state,
                &segment.node_id,
                &segment.shard_id.model_id,
                // Verify batches carry a few packed tokens, not a prompt pass.
                crate::daemon::state::WorkKind::Decode,
                num_layers,
                activation_bytes.len(),
                // Segment 0 gets packed token ids; later hops get hidden states.
                if idx == 0 {
                    local::ActivationUnits::PromptBytes
                } else {
                    local::ActivationUnits::HiddenStates
                },
            );
            let seg_start = std::time::Instant::now();
            let result = PipelineExecutor::wait_for_result(
                rx,
                request_id,
                idx,
                &segment.node_id,
                num_layers,
                activation_bytes.len(),
                budget,
                // A verify step is a few tokens' work; the loop that issues it
                // reads the cancel flag between steps.
                None,
            )
            .await?;
            pending_guard.disarm();
            // A SINGLE-token verify is an ordinary decode step wearing a
            // verify's name — `ngram_only_spec` sends `vec![last_token]` on a
            // cascade MISS — so it is exactly the sample `peer_speed` wants,
            // and this is the one place both verify callers funnel through
            // (`hedge_dispatch` reaches it on all five of its return paths).
            //
            // Without it the scheduler learned NOTHING from a speculative
            // request, however long it took. Measured 2026-08-30: four
            // consecutive requests to a peer-held model took 244/210/200/182 s
            // while a peer on the LAN served the identical request in 0.80 s,
            // and every one of them was ranked on the slow peer's own
            // ADVERTISED 2.0 tok/s because `observed_ms_per_layer` was `None`
            // every time. The prefill that DID reach a recorder carried
            // `activations.len() == 0`, which `peer_speed::observe` rejects for
            // a coefficient that divides by it — so nothing was ever learned.
            //
            // A MULTI-token batch is deliberately NOT recorded: `observe`
            // divides by `layers` alone, so K drafts through the same layers
            // would inflate ms/layer by roughly K and corrupt the same EWMA
            // that sizes segment timeouts — a worse failure than the one this
            // fixes. See `docs/FUTURE_WORK.md`.
            if verify_tokens.len() == 1 {
                shared_state.record_peer_segment_latency(
                    &segment.node_id,
                    &segment.shard_id.model_id,
                    crate::daemon::state::WorkKind::Decode,
                    seg_start.elapsed().as_millis() as u64,
                    num_layers,
                    result.activations.len(),
                );
            }
            result
        } else {
            shared_state.model_process_pool.forward(forward).await?
        };

        if let Some(crate::types::NetworkFinishReason::Error(msg)) = &result.finish_reason {
            // Same recovery as the distributed and remote-generate siblings:
            // the class does not survive the wire, and without it a peer's
            // `Validation` is reported to the caller as this server breaking
            // AND the peer is charged a serve-failure penalty for a mistake
            // that was the caller's (gotcha #304).
            return Err(
                crate::error::reclassify_flattened_error(msg).unwrap_or_else(|| {
                    SwarmError::Inference(format!("spec verify segment {idx}: {msg}"))
                }),
            );
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
    _requester_node_id_bytes: [u8; 32],
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
        chain: Vec::new(),
        sender_peer_bytes: None,
        tp_meta: None,
        // Unchained: the receiver answers its sender, which is us. Since
        // 2026-08-21 a `Some` here would put the 0x07 reply-to trailer on the
        // wire, which only chained runs carry and older peers do not expect.
        // (The seal this once fed never ran for a remote segment — the id was
        // not on the wire — so nothing is lost; `docs/FUTURE_WORK.md`.)
        requester_node_id: None,
        pre_embedded: false,
        generated_ids: Vec::new(),
        adapter_id: None,
        draft_tokens: Vec::new(),
        spec_logits_requested: false,
        truncate_kv_to: Some(truncate_to),
        chunk_meta: None,
        sampling: None,
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
    eos: &std::collections::HashSet<u32>,
) {
    // `eos` is REQUIRED, with no convenience wrapper that omits it, because
    // every one of these call sites emitted the token BEFORE testing whether
    // it was end-of-turn. See `emit_streaming_batch` for what that produced.
    if eos.contains(&token) {
        return;
    }
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
    eos: &std::collections::HashSet<u32>,
    finish_reason: &mut String,
) -> bool {
    let tx = match token_tx {
        Some(tx) => tx,
        None => return false,
    };
    for &t in tokens {
        // End-of-turn is a CONTROL token: it ends the reply, it is not part of
        // it. Every caller keeps it in its own `emitted` vector so the decode
        // loop still stops on it, and every caller passed that vector straight
        // here — so all three speculative coordinators streamed `<|eot_id|>` to
        // the client as reply text, while `finish_speculative` filtered it out
        // of the non-streaming content. The same answer therefore differed by
        // transport (gotcha #414, measured on the live swarm 2026-08-30).
        //
        // Filtering HERE rather than at the call sites is deliberate: the
        // R105 fix truncated each caller's vector to `eos_at + 1` — stopping
        // post-EOS junk but keeping EOS itself — and was copied into all three
        // files, so the defect was copied with it.
        if eos.contains(&t) {
            continue;
        }
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
    /// Set the first time a chained run fails for this request. From then on
    /// every segment of this request is sent unchained — the plain path that
    /// names its culprit and fails over per segment. Per request, not global:
    /// one bad hand-off says nothing about the next request's peers.
    pub(super) chaining_disabled: bool,
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
            chaining_disabled: false,
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

    pub(super) fn make_test_state() -> Arc<SharedState> {
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

    /// A model's prompt must never be built with a DIFFERENT model's template.
    ///
    /// `loaded_model_info` holds whichever model was loaded last, and
    /// `resolve_chat_template` returned it for any requested model — so on a
    /// node with one model resident, every other model was prompted in the
    /// resident one's format, with its BOS and EOS. That produces a plausible
    /// wrong answer, not an error (gotcha #169), and only on the non-split
    /// route, so the same model looked correct when served locally.
    #[tokio::test]
    async fn the_resident_models_template_is_not_lent_to_other_models() {
        let state = make_test_state();
        *state.loaded_model_info.write().await = Some(crate::daemon::state::LoadedModelInfo {
            name: "Qwen2.5 Coder 7B Instruct".into(),
            size_bytes: 0,
            eos_tokens: vec![],
            chat_template: Some("QWEN-TEMPLATE".into()),
            bos_token: "<qwen-bos>".into(),
            eos_token: "<qwen-eos>".into(),
        });

        // The resident model, by display name and by slug, is itself.
        for id in ["Qwen2.5 Coder 7B Instruct", "qwen2.5-coder-7b-instruct"] {
            assert!(
                state
                    .loaded_model_info_is_for(&crate::types::ModelId(id.into()))
                    .await,
                "{id} names the resident model"
            );
        }

        // Anything else is not, and must fall through to its own header.
        for id in [
            "llama-3.2-3b-instruct-q4-k-m",
            "qwen2.5-0.5b-instruct-fp16",
            // Same family and prefix, different model — matching must not be a
            // substring test.
            "qwen2.5-coder-7b-instruct-q4-k-m-EXTRA",
        ] {
            assert!(
                !state
                    .loaded_model_info_is_for(&crate::types::ModelId(id.into()))
                    .await,
                "{id} must NOT inherit the resident model's template"
            );
        }
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

    /// What a simulated peer answers a forward with.
    enum PeerReply {
        Error(&'static str),
        Activations(Vec<u8>),
        Token(u32),
    }

    /// Play every peer in an assignment: answer each `SendTensor` the executor
    /// emits, as the node it was addressed to, and record who was sent what.
    /// Ends when the executor (and so the channel's sender) is dropped.
    fn spawn_peers(
        state: Arc<SharedState>,
        mut rx: mpsc::Receiver<NetworkCommand>,
        peers: std::collections::HashMap<Vec<u8>, NodeId>,
        reply: impl Fn(&NodeId) -> PeerReply + Send + 'static,
    ) -> tokio::task::JoinHandle<Vec<(NodeId, Vec<u8>)>> {
        tokio::spawn(async move {
            let mut sent = Vec::new();
            while let Some(cmd) = rx.recv().await {
                let NetworkCommand::SendTensor {
                    target_peer_bytes,
                    forward,
                } = cmd
                else {
                    continue;
                };
                let node = peers[&target_peer_bytes].clone();
                sent.push((node.clone(), forward.activations.clone()));
                let base = LayerResult::error(forward.request_id, "");
                let result = match reply(&node) {
                    PeerReply::Error(m) => LayerResult::error(forward.request_id, m),
                    PeerReply::Activations(a) => LayerResult {
                        activations: a,
                        finish_reason: None,
                        ..base
                    },
                    PeerReply::Token(t) => LayerResult {
                        token_ids: vec![t],
                        finish_reason: None,
                        ..base
                    },
                };
                assert!(
                    state.resolve_pending_layer_result(Some(&node), result),
                    "the executor registers its waiter before it sends"
                );
            }
            sent
        })
    }

    fn remote_segment(node: &NodeId, range: (u32, u32)) -> PipelineSegment {
        PipelineSegment {
            node_id: node.clone(),
            shard_id: ShardId {
                model_id: ModelId("test".into()),
                index: range.0,
            },
            layer_range: range,
        }
    }

    /// The standby the scheduler prefers is THIS node, and it was the one
    /// standby that could not work.
    ///
    /// `find_standbys` sorts the local node first on purpose — a node holding
    /// every shard is the most reliable fallback there is. `failover_segment`
    /// only knew how to DIAL, and the local node has no `peer_id_bytes`, so the
    /// most-preferred standby ended the request with
    /// `Network error: No peer_id_bytes for backup node` one line after the
    /// scheduler correctly chose the machine that could have answered
    /// (gotcha #458). Reported live on a 48-layer model over an 8-segment
    /// route, after the primary died with a driver-level out-of-memory.
    ///
    /// Here the local node has no loaded model, so running in-process fails —
    /// which is the point: the failure must be that standby's, and the NEXT
    /// standby must still be tried. Before the fix the request ended on the
    /// local entry with a network error and never reached the second one.
    #[tokio::test]
    async fn a_local_standby_is_run_here_and_a_failed_standby_does_not_end_the_request() {
        let state = make_test_state();
        let local = state.identity.node_id().clone();
        let (tx, rx) = mpsc::channel::<NetworkCommand>(64);
        let request = make_test_request(&state);
        let request_id = request.id;
        let (a, b, d) = (NodeId([0xA1; 32]), NodeId([0xB2; 32]), NodeId([0xD4; 32]));
        let mut peers = std::collections::HashMap::new();
        for (node, byte) in [(&a, 0xA1u8), (&b, 0xB2), (&d, 0xD4)] {
            state.peer_id_map.insert(node.clone(), vec![byte]);
            peers.insert(vec![byte], node.clone());
        }
        // Deliberately NO peer_id_map entry for `local` — that is what the node
        // running this code looks like, and what the old code tripped over.
        let assignment = PipelineAssignment {
            request_id,
            segments: vec![remote_segment(&a, (0, 16)), remote_segment(&d, (16, 32))],
            // Local first, exactly as `find_standbys` orders them.
            standbys: vec![remote_segment(&local, (0, 16)), remote_segment(&b, (0, 16))],
            tp_groups: vec![],
            supports_speculative: false,
        };
        let mut executor = PipelineExecutor::new(state.clone(), tx, request, assignment);
        let a2 = a.clone();
        let peers_task = spawn_peers(state.clone(), rx, peers, move |node| {
            if *node == a2 {
                PeerReply::Error("Worker: Service unavailable: worker fatal error: out of memory")
            } else {
                // The second standby answers, and so does segment 1.
                PeerReply::Token(7)
            }
        });

        let result = executor
            .forward_through_segments(request_id, 0, 0, b"hello".to_vec(), None, false, &[])
            .await;
        drop(executor);
        let sent = peers_task.await.unwrap();

        if let Err(ref e) = result {
            let msg = e.to_string();
            assert!(
                !msg.contains("No peer_id_bytes"),
                "the local node is run in-process, never dialled: {msg}"
            );
        }
        // The whole point: the local standby's own failure did not end the
        // failover, so the next standby was reached.
        assert!(
            sent.iter().any(|(n, _)| *n == b),
            "the second standby must still be tried after the local one fails"
        );
        assert!(
            !sent.iter().any(|(n, _)| *n == local),
            "nothing is ever sent to ourselves over the network"
        );
    }

    /// With no standby at all — which is every single-peer delegation, by
    /// design — the failure must still say WHY the segment failed.
    ///
    /// `last_failure` was seeded to `None` and only written inside the standby
    /// loop, so a plan carrying zero standbys never entered that loop and the
    /// caller got a bare "no standby available". Reported live on v0.3.154: a
    /// delegated segment lost its connection mid-stream
    /// (`OutboundFailure: IO error on outbound stream: connection lost`) and
    /// the operator had to correlate two log lines to learn that. It also cost
    /// the retry, which matches "OutboundFailure" in the message text.
    #[tokio::test]
    async fn a_segment_with_no_standby_still_reports_why_it_failed() {
        let state = make_test_state();
        let (tx, rx) = mpsc::channel::<NetworkCommand>(64);
        let request = make_test_request(&state);
        let request_id = request.id;
        let (a, d) = (NodeId([0xA1; 32]), NodeId([0xD4; 32]));
        let mut peers = std::collections::HashMap::new();
        for (node, byte) in [(&a, 0xA1u8), (&d, 0xD4)] {
            state.peer_id_map.insert(node.clone(), vec![byte]);
            peers.insert(vec![byte], node.clone());
        }
        let assignment = PipelineAssignment {
            request_id,
            segments: vec![remote_segment(&a, (0, 16)), remote_segment(&d, (16, 32))],
            // The shape single-peer delegation always produces.
            standbys: vec![],
            tp_groups: vec![],
            supports_speculative: false,
        };
        let mut executor = PipelineExecutor::new(state.clone(), tx, request, assignment);
        let a2 = a.clone();
        let peers = spawn_peers(state, rx, peers, move |node| {
            if *node == a2 {
                PeerReply::Error("OutboundFailure: IO error on outbound stream: connection lost")
            } else {
                PeerReply::Token(7)
            }
        });

        let result = executor
            .forward_through_segments(request_id, 0, 0, b"hello".to_vec(), None, false, &[])
            .await;
        drop(executor);
        let _ = peers.await.unwrap();

        let err = result.expect_err("segment 0 failed and nothing could take it over");
        let msg = err.to_string();
        assert!(
            matches!(err, SwarmError::SegmentFailoverExhausted(_)),
            "wrong class: {msg}"
        );
        assert!(
            msg.contains("connection lost"),
            "the failure must name the cause, not only that nothing could take over: {msg}"
        );
        // And carrying the cause is what puts it back in front of the retry.
        assert!(
            crate::inference::router::is_transient_remote_failure_for_test(&err),
            "a lost connection is retryable, and the message is how the router sees it: {msg}"
        );
    }

    /// The live failure of 2026-09-01 (gotcha #435), in miniature: the first
    /// segment's holder fails, its only standby REFUSES the segment, and the
    /// refusal's empty activations used to be forwarded to segment 1 as though
    /// the standby had computed them — which failed there as `Tensor bytes
    /// too short` and reported segment 1 as the one with no standby. A
    /// standby's error is a failure of that standby.
    #[tokio::test]
    async fn a_standby_that_refuses_is_a_failure_not_the_segments_output() {
        let state = make_test_state();
        let (tx, rx) = mpsc::channel::<NetworkCommand>(64);
        let request = make_test_request(&state);
        let request_id = request.id;
        let (a, b, d) = (NodeId([0xA1; 32]), NodeId([0xB2; 32]), NodeId([0xD4; 32]));
        let mut peers = std::collections::HashMap::new();
        for (node, byte) in [(&a, 0xA1u8), (&b, 0xB2), (&d, 0xD4)] {
            state.peer_id_map.insert(node.clone(), vec![byte]);
            peers.insert(vec![byte], node.clone());
        }
        let assignment = PipelineAssignment {
            request_id,
            segments: vec![remote_segment(&a, (0, 16)), remote_segment(&d, (16, 32))],
            standbys: vec![remote_segment(&b, (0, 16))],
            tp_groups: vec![],
            supports_speculative: false,
        };
        let mut executor = PipelineExecutor::new(state.clone(), tx, request, assignment);
        let (a2, b2) = (a.clone(), b.clone());
        let peers = spawn_peers(state, rx, peers, move |node| {
            if *node == a2 || *node == b2 {
                PeerReply::Error(
                    "Worker: Service unavailable: needs about 10362 MB of memory: 8566 MB of weights",
                )
            } else {
                PeerReply::Token(7)
            }
        });

        let result = executor
            .forward_through_segments(request_id, 0, 0, b"hello".to_vec(), None, false, &[])
            .await;
        drop(executor);
        let sent = peers.await.unwrap();

        let err = result.expect_err("nobody computed segment 0");
        let msg = err.to_string();
        assert!(
            matches!(err, SwarmError::SegmentFailoverExhausted(_)),
            "wrong class: {msg}"
        );
        assert!(
            msg.contains("Segment 0") && msg.contains("10362 MB"),
            "the failure names the segment that failed and why the last standby refused it: {msg}"
        );
        let asked: Vec<&NodeId> = sent.iter().map(|(n, _)| n).collect();
        assert_eq!(asked, vec![&a, &b], "holder, then its standby, then nobody");
        assert!(
            !sent.iter().any(|(n, _)| *n == d),
            "segment 1 must never be sent the refusal's empty activations"
        );
    }

    /// With more than one standby, a refusal moves the segment on to the next
    /// one, and the loop resumes with what THAT standby computed; the segment
    /// is re-pointed at it for the rest of the request.
    #[tokio::test]
    async fn a_second_standby_is_tried_when_the_first_fails() {
        let state = make_test_state();
        let (tx, rx) = mpsc::channel::<NetworkCommand>(64);
        let request = make_test_request(&state);
        let request_id = request.id;
        let (a, b, c, d) = (
            NodeId([0xA1; 32]),
            NodeId([0xB2; 32]),
            NodeId([0xC3; 32]),
            NodeId([0xD4; 32]),
        );
        let mut peers = std::collections::HashMap::new();
        for (node, byte) in [(&a, 0xA1u8), (&b, 0xB2), (&c, 0xC3), (&d, 0xD4)] {
            state.peer_id_map.insert(node.clone(), vec![byte]);
            peers.insert(vec![byte], node.clone());
        }
        let assignment = PipelineAssignment {
            request_id,
            segments: vec![remote_segment(&a, (0, 16)), remote_segment(&d, (16, 32))],
            standbys: vec![remote_segment(&b, (0, 16)), remote_segment(&c, (0, 16))],
            tp_groups: vec![],
            supports_speculative: false,
        };
        let mut executor = PipelineExecutor::new(state.clone(), tx, request, assignment);
        let (a2, b2, c2) = (a.clone(), b.clone(), c.clone());
        let computed = vec![0x5Au8; 64];
        let computed2 = computed.clone();
        let peers = spawn_peers(state, rx, peers, move |node| {
            if *node == a2 || *node == b2 {
                PeerReply::Error("Worker: Service unavailable: out of memory")
            } else if *node == c2 {
                PeerReply::Activations(computed2.clone())
            } else {
                PeerReply::Token(7)
            }
        });

        let result = executor
            .forward_through_segments(request_id, 0, 0, b"hello".to_vec(), None, false, &[])
            .await;
        let now_serving = executor.assignment.segments[0].node_id.clone();
        drop(executor);
        let sent = peers.await.unwrap();

        let out = result.expect("the second standby answered");
        assert_eq!(out.token_ids, vec![7]);
        let asked: Vec<&NodeId> = sent.iter().map(|(n, _)| n).collect();
        assert_eq!(asked, vec![&a, &b, &c, &d]);
        let to_d = &sent.iter().find(|(n, _)| *n == d).unwrap().1;
        assert_eq!(
            to_d, &computed,
            "segment 1 receives what the standby that answered computed"
        );
        assert_eq!(
            now_serving, c,
            "the segment is re-pointed at the standby that answered"
        );
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
        // The fast path's own refusal, as the coordinator receives it — flattened
        // to text and re-wrapped, so the classifier sees a prefix in front of it.
        assert!(
            remote_error_means_missing_shard(&format!(
                "Inference error: {}",
                crate::daemon::dispatch::remote_generate::REMOTE_GENERATE_NOT_HOSTED
            )),
            "a peer saying it does not host the model is a stale holder claim (gotcha #433)"
        );
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
        assert_eq!(
            fwd.requester_node_id, None,
            "a spec-verify forward is unchained, so it names no reply-to"
        );
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
        // The peer must be RESOLVABLE, or the function now fails earlier on
        // routing and this stops testing the closed-channel arm it exists for.
        state
            .peer_id_map
            .insert(NodeId([42u8; 32]), vec![1, 2, 3, 4]);
        let verify_tokens: Vec<u32> = vec![100, 101, 102];

        let result = forward_verify_through_segments(
            &state,
            &tx,
            request_id,
            10,
            &segments,
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

    /// Drive one verify round to completion and answer it, so
    /// `forward_verify_through_segments` reaches its `wait_for_result`.
    ///
    /// Returns how long the round was measured at, which the caller ignores —
    /// what matters is the side effect on `peer_speed`.
    async fn run_one_verify_round(
        state: &Arc<SharedState>,
        segments: &[PipelineSegment],
        verify_tokens: Vec<u32>,
    ) {
        let (tx, mut rx) = mpsc::channel::<NetworkCommand>(8);
        let request_id = uuid::Uuid::new_v4();
        let s2 = state.clone();
        let segs = segments.to_vec();
        let handle = tokio::spawn(async move {
            forward_verify_through_segments(&s2, &tx, request_id, 0, &segs, &verify_tokens, None)
                .await
        });

        // The forward is dispatched, then the waiter blocks on its oneshot.
        let _cmd = tokio::time::timeout(std::time::Duration::from_secs(5), rx.recv())
            .await
            .expect("a forward must be dispatched")
            .expect("channel open");

        // Answer as the peer would: no error, one logit row for the single
        // verified position.
        let mut reply = LayerResult::error(request_id, "unused");
        reply.finish_reason = None;
        reply.spec_logits = vec![vec![0.0f32; 4]];
        // The waiter registers before the send, but yield anyway so this is
        // not a race on a loaded machine.
        for _ in 0..50 {
            if state.pending_layer_results.contains_key(&request_id) {
                break;
            }
            tokio::task::yield_now().await;
        }
        // The waiter is pinned to the node the forward went to
        // (`awaiting: Some(..)`), and `accepts` rejects a `None` sender
        // outright — answering as "nobody" is exactly the late-result case
        // gotcha #229 exists to stop.
        assert!(
            state.resolve_pending_layer_result(Some(&segments[0].node_id), reply),
            "the pending waiter must still be there to answer"
        );
        handle
            .await
            .expect("task joined")
            .expect("verify round must succeed");
    }

    /// A single-token verify must teach the scheduler what that peer cost.
    ///
    /// **The regression test for four consecutive 200-second requests.**
    /// Measured 2026-08-30: a peer-held model answered in 244/210/200/182 s
    /// while a peer on the LAN served the identical request in 0.80 s. The LAN
    /// peer was a candidate every time and lost every time on the slow peer's
    /// own ADVERTISED speed, because `observed_ms_per_layer` was `None` — the
    /// speculative path never recorded anything, so a three-minute answer
    /// taught the scheduler nothing and it re-picked the same peer at the same
    /// wrong price.
    ///
    /// Two rounds because the first is COLD: `peer_speed::observe` deliberately
    /// drops a cold decode sample ("a load time wearing a compute figure's
    /// clothes") and the same call marks the peer warm, so the second round is
    /// the one that can be believed. That is the production sequence too — the
    /// prefill marks it warm before any verify round runs.
    #[tokio::test]
    async fn a_single_token_verify_teaches_the_scheduler_what_the_peer_cost() {
        let state = make_test_state();
        let peer = NodeId([42u8; 32]);
        state.peer_id_map.insert(peer.clone(), vec![1, 2, 3, 4]);
        let segments = vec![PipelineSegment {
            node_id: peer.clone(),
            shard_id: ShardId {
                model_id: ModelId("test-model".into()),
                index: 0,
            },
            layer_range: (0, 8),
        }];

        assert!(
            state.observed_latency_ms_per_layer(&peer).is_none(),
            "nothing measured yet — if this starts as Some the test proves nothing"
        );

        run_one_verify_round(&state, &segments, vec![7]).await;
        run_one_verify_round(&state, &segments, vec![7]).await;

        assert!(
            state.observed_latency_ms_per_layer(&peer).is_some(),
            "a single-token verify is an ordinary decode step and MUST be \
             recorded — without this the scheduler ranks a peer on the speed it \
             advertises about itself no matter how slowly it actually answered"
        );
    }

    /// ...but a MULTI-token batch must NOT be, and that is deliberate.
    ///
    /// `peer_speed::observe` divides by `layers` alone, so K drafts through the
    /// same layers would inflate ms/layer by roughly K — corrupting the same
    /// EWMA that sizes segment timeouts, which is a worse failure than the one
    /// the sibling test above pins. See `docs/FUTURE_WORK.md`.
    #[tokio::test]
    async fn a_speculative_batch_is_deliberately_not_recorded_as_a_decode_step() {
        let state = make_test_state();
        let peer = NodeId([43u8; 32]);
        state.peer_id_map.insert(peer.clone(), vec![9, 9, 9, 9]);
        let segments = vec![PipelineSegment {
            node_id: peer.clone(),
            shard_id: ShardId {
                model_id: ModelId("test-model".into()),
                index: 0,
            },
            layer_range: (0, 8),
        }];

        // Same two rounds as the sibling, so "cold sample dropped" cannot be
        // the reason this one stays empty.
        run_one_verify_round(&state, &segments, vec![7, 8, 9]).await;
        run_one_verify_round(&state, &segments, vec![7, 8, 9]).await;

        assert!(
            state.observed_latency_ms_per_layer(&peer).is_none(),
            "a batch of drafts is not one decode step; recording it as one \
             overstates ms/layer by roughly the draft width"
        );
    }
}

#[cfg(test)]
mod failover_retarget_tests {
    use super::tests::make_test_state;
    use super::*;
    use crate::types::{ModelId, NodeId, PipelineSegment, ShardId};

    fn seg(node: NodeId) -> PipelineSegment {
        PipelineSegment {
            node_id: node,
            shard_id: ShardId {
                model_id: ModelId("m".into()),
                index: 0,
            },
            layer_range: (0, 22),
        }
    }

    /// **The regression test for the 284-second hang after a failover.**
    ///
    /// A verify round must send to whichever node the assignment names *now*.
    /// `distributed.rs` rewrites `segments[i].node_id` in place when it fails
    /// over to a standby, so anything the caller resolved before its decode
    /// loop is stale from that moment on.
    ///
    /// Measured live 2026-08-04: token 1 succeeded via failover in 243 ms, then
    /// every later round was sent to the FAILED node while the waiter was
    /// pinned to the standby. The failed node's reply was discarded as "from a
    /// node this request is no longer waiting on" and the request sat until its
    /// 284 s segment timeout — 12 failovers, 10 timeouts on one node.
    ///
    /// The property: the peer the round dispatches to is derived from
    /// `segment.node_id`, the same field `register_pending_layer_result` pins
    /// the waiter to, so the two cannot disagree.
    #[tokio::test]
    async fn a_verify_round_targets_the_node_the_assignment_names_now() {
        let state = make_test_state();
        let failed = NodeId([1u8; 32]);
        let standby = NodeId([2u8; 32]);
        state.peer_id_map.insert(failed.clone(), vec![0xFA, 0x11]);
        state.peer_id_map.insert(standby.clone(), vec![0x5B, 0x22]);

        let (tx, mut rx) = mpsc::channel::<NetworkCommand>(8);

        // Assignment as it looks AFTER failover has rewritten it.
        let segments = vec![seg(standby.clone())];
        let request_id = uuid::Uuid::new_v4();

        tokio::spawn(async move {
            let _ = forward_verify_through_segments(
                &state,
                &tx,
                request_id,
                0,
                &segments,
                &[7u32],
                None,
            )
            .await;
        });

        let cmd = tokio::time::timeout(std::time::Duration::from_secs(5), rx.recv())
            .await
            .expect("a forward must be dispatched")
            .expect("channel open");

        match cmd {
            NetworkCommand::SendTensor {
                target_peer_bytes, ..
            } => {
                assert_eq!(
                    target_peer_bytes,
                    vec![0x5B, 0x22],
                    "must dispatch to the STANDBY the assignment now names, not the \
                     node it failed over from — sending to the old node is what hung \
                     the request for 284s"
                );
            }
            other => panic!("expected SendTensor, got {other:?}"),
        }
    }
}

/// How far can we chain, starting at segment `idx`?
///
/// Returns the ordered hops to hand to segment `idx` — that is,
/// `segments[idx+1 ..= j]` for the longest run of consecutive segments that can
/// pass activations to each other directly. Empty means "send this one the old
/// way and wait for it to come home", which is always correct.
///
/// **Only a run of consecutive REMOTE segments can be chained**, and that is
/// what makes prompt privacy survive: with `encrypted_pipeline` the coordinator
/// holds the first and last segments itself, so the run is the middle, the ends
/// stay here, and no remote node ever sees the prompt or the sampled token. A
/// local segment simply ends the run.
///
/// Every hop must advertise `features::PIPELINE_CHAIN`, because a node without
/// it ignores the field and returns its result here — which is correct but
/// would leave the coordinator waiting on the wrong node.
pub(crate) fn plan_chain(
    segments: &[crate::types::PipelineSegment],
    idx: usize,
    local_node: &crate::types::NodeId,
    can_chain: impl Fn(&crate::types::NodeId) -> bool,
    max_hops: usize,
) -> Vec<crate::types::ChainHop> {
    if max_hops == 0 || idx >= segments.len() {
        return Vec::new();
    }
    // The segment we are about to send to must itself be a remote node that can
    // forward; otherwise there is nothing to chain FROM.
    let head = &segments[idx];
    if head.node_id == *local_node || !can_chain(&head.node_id) {
        return Vec::new();
    }
    let mut hops = Vec::new();
    for seg in segments.iter().skip(idx + 1) {
        if seg.node_id == *local_node || !can_chain(&seg.node_id) {
            break;
        }
        // A hop that does not begin where the previous one ended is not a
        // pipeline — refuse rather than forward activations into a gap.
        let prev_end = hops
            .last()
            .map(|h: &crate::types::ChainHop| h.layer_range.1)
            .unwrap_or(head.layer_range.1);
        if seg.layer_range.0 != prev_end {
            break;
        }
        hops.push(crate::types::ChainHop {
            node_id: seg.node_id.clone(),
            layer_range: seg.layer_range,
        });
        if hops.len() >= max_hops {
            break;
        }
    }
    hops
}

#[cfg(test)]
mod chain_planning_tests {
    use super::plan_chain;
    use crate::types::{ModelId, NodeId, PipelineSegment, ShardId};

    fn seg(node: u8, start: u32, end: u32) -> PipelineSegment {
        PipelineSegment {
            node_id: NodeId([node; 32]),
            shard_id: ShardId {
                model_id: ModelId("m".into()),
                index: 0,
            },
            layer_range: (start, end),
        }
    }

    const LOCAL: NodeId = NodeId([0u8; 32]);
    fn all_can_chain(_: &NodeId) -> bool {
        true
    }

    /// The whole point: a run of remote segments is handed over once, so the
    /// coordinator's round trips stop scaling with the number of shards.
    #[test]
    fn a_run_of_remote_segments_is_chained_in_order() {
        let segs = [seg(1, 0, 8), seg(2, 8, 16), seg(3, 16, 24)];
        let hops = plan_chain(&segs, 0, &LOCAL, all_can_chain, 8);
        assert_eq!(hops.len(), 2, "two nodes follow the head");
        assert_eq!(hops[0].node_id, NodeId([2u8; 32]));
        assert_eq!(hops[1].node_id, NodeId([3u8; 32]));
        assert_eq!(hops[1].layer_range, (16, 24));
    }

    /// Prompt privacy survives chaining because a local segment ends the run.
    /// With `encrypted_pipeline` the coordinator holds the first and last
    /// segments, so the ends stay here and only the middle is chained.
    #[test]
    fn a_local_segment_ends_the_run_so_the_boomerang_still_holds() {
        // local, remote, remote, local — the privacy-preserving shape.
        let segs = [seg(0, 0, 4), seg(1, 4, 12), seg(2, 12, 20), seg(0, 20, 24)];
        let hops = plan_chain(&segs, 1, &LOCAL, all_can_chain, 8);
        assert_eq!(hops.len(), 1, "the run stops before the local tail");
        assert_eq!(hops[0].node_id, NodeId([2u8; 32]));
    }

    /// A chained run finishes the pipeline when its TAIL is the last segment,
    /// not when its head is. The coordinator asks that question to decide
    /// whether it has the final answer in its hand, so getting it wrong is a
    /// hang rather than an error: the reply arrives and is walked past.
    #[test]
    fn a_run_ending_at_the_last_segment_is_recognised_as_finishing() {
        let segs = [seg(1, 0, 8), seg(2, 8, 16), seg(3, 16, 24)];
        let num_segments = segs.len();

        // Chaining all three from index 0: the head is segment 0, which is NOT
        // the last, but the run ends at segment 2, which is.
        let head = 0usize;
        let hops = plan_chain(&segs, head, &LOCAL, all_can_chain, 8);
        assert_eq!(hops.len(), 2);
        assert_eq!(
            head + hops.len(),
            num_segments - 1,
            "the run reaches the final segment"
        );
        assert_ne!(
            head,
            num_segments - 1,
            "and the head alone does not, which is the bug"
        );

        // A run that stops short must NOT claim to finish the pipeline.
        let short = plan_chain(&segs, head, &LOCAL, all_can_chain, 1);
        assert_eq!(short.len(), 1);
        assert_ne!(head + short.len(), num_segments - 1);
    }

    #[test]
    fn nothing_is_chained_from_a_local_segment() {
        let segs = [seg(0, 0, 8), seg(1, 8, 16)];
        assert!(plan_chain(&segs, 0, &LOCAL, all_can_chain, 8).is_empty());
    }

    #[test]
    fn the_last_segment_has_nobody_to_chain_to() {
        let segs = [seg(1, 0, 8), seg(2, 8, 16)];
        assert!(plan_chain(&segs, 1, &LOCAL, all_can_chain, 8).is_empty());
    }

    /// A node that does not advertise the feature ignores the field and replies
    /// to the coordinator, so chaining THROUGH it would leave us waiting on the
    /// wrong node. The run stops at it.
    #[test]
    fn a_peer_without_the_feature_ends_the_run() {
        let segs = [seg(1, 0, 8), seg(2, 8, 16), seg(3, 16, 24)];
        let hops = plan_chain(&segs, 0, &LOCAL, |n| n.0[0] != 3, 8);
        assert_eq!(hops.len(), 1, "stops before the node that cannot chain");
        assert_eq!(hops[0].node_id, NodeId([2u8; 32]));

        // And the head itself must be able to forward.
        assert!(plan_chain(&segs, 0, &LOCAL, |n| n.0[0] != 1, 8).is_empty());
    }

    /// Layers must be contiguous. A gap means the activations would be fed to a
    /// segment expecting different inputs, which produces a confident wrong
    /// answer rather than an error.
    #[test]
    fn a_gap_in_the_layer_ranges_ends_the_run() {
        let segs = [seg(1, 0, 8), seg(2, 12, 20)];
        assert!(plan_chain(&segs, 0, &LOCAL, all_can_chain, 8).is_empty());
    }

    #[test]
    fn the_hop_count_is_bounded() {
        let segs: Vec<_> = (1u8..=6)
            .map(|i| seg(i, (i as u32 - 1) * 4, i as u32 * 4))
            .collect();
        assert_eq!(plan_chain(&segs, 0, &LOCAL, all_can_chain, 2).len(), 2);
        assert!(plan_chain(&segs, 0, &LOCAL, all_can_chain, 0).is_empty());
    }
}

#[cfg(test)]
mod missing_shard_over_the_wire_tests {
    use super::remote_error_means_missing_shard;
    use crate::daemon::dispatch::layer_forward::sanitize_peer_facing_error;

    /// **The test that would have caught this.** The detector and the
    /// sanitiser live in different files and had to agree on a phrase; one of
    /// them was rewriting it. Testing either alone passes happily — the bug is
    /// only visible when a real error is put through the transformation it
    /// actually undergoes before a coordinator sees it.
    ///
    /// Live consequence, 2026-08-26: one peer was segment 1 of eleven
    /// consecutive failed pipelines in half an hour, refusing each in ~7 ms
    /// with "Required shards not available", and its stale claim was never
    /// retracted because the detector did not recognise the sanitised form.
    #[test]
    fn a_missing_shard_error_is_still_recognised_after_sanitising() {
        // REAL wordings only. Every one is produced somewhere in the tree:
        // `ShardReader` emits the two "missing region" forms
        // (`inference/split/shard_reader.rs`), and `layer_forward` emits
        // "No local shards for model". An invented fixture proves nothing —
        // an earlier draft of this test used "No local shards for this layer
        // range", which exists nowhere, and its failure looked like a second
        // bug until the string was grepped for.
        for raw in [
            "ShardReader: position is in a missing shard region",
            "Internal error: blk.0.attn_q: ShardReader: position 345977248 is in a missing region",
            "No local shards for model",
        ] {
            let over_the_wire = sanitize_peer_facing_error(raw);
            assert!(
                remote_error_means_missing_shard(&over_the_wire),
                "detector must recognise what the peer actually sends. \
                 raw {raw:?} was sanitised to {over_the_wire:?}"
            );
        }
        // The same wordings unsanitised, which is what a LOCAL segment failure
        // produces — that path never goes through the sanitiser.
        for raw in [
            "ShardReader: position is in a missing shard region",
            "Internal error: blk.0.attn_q: ShardReader: position 345977248 is in a missing region",
        ] {
            assert!(
                remote_error_means_missing_shard(raw),
                "detector must recognise the local form: {raw:?}"
            );
        }
    }

    /// The converse still holds: an unrelated failure must not cost a healthy
    /// peer its claim. Retracting on the wrong error moves work off a good
    /// node, which is the failure this whole mechanism has to avoid.
    #[test]
    fn an_unrelated_failure_never_retracts_a_claim() {
        for raw in [
            "CUDA out of memory",
            "peer never acknowledged the request",
            "Timed out waiting for segment result (52s, 26 layers)",
            "",
        ] {
            assert!(!remote_error_means_missing_shard(raw), "raw: {raw:?}");
            assert!(
                !remote_error_means_missing_shard(&sanitize_peer_facing_error(raw)),
                "sanitised: {raw:?}"
            );
        }
    }
}

#[cfg(test)]
mod peer_error_recovery_tests {
    use super::*;

    fn result_with(finish: Option<NetworkFinishReason>) -> LayerResult {
        LayerResult {
            request_id: uuid::Uuid::nil(),
            token_ids: Vec::new(),
            finish_reason: finish,
            activations: Vec::new(),
            sealed_token_ids: None,
            spec_logits: Vec::new(),
            matched_stop_sequence: None,
            token_logprobs: Vec::new(),
        }
    }

    /// The defect this exists for: a prompt too long for a PEER-held model came
    /// back as `500 server_error` naming an internal mechanism, while the same
    /// prompt on a LOCAL model was a `400 invalid_request_error` that said how
    /// much to cut. Measured on the live swarm 2026-08-30.
    #[test]
    fn an_over_long_prompt_refused_by_a_peer_stays_the_callers_mistake() {
        // Verbatim shape of what the serving node put on the wire.
        let from_peer = "Worker: Validation error: This conversation is 12041 tokens, \
                         longer than the model's limit of 8192";
        let err = peer_error_from_result(&result_with(Some(NetworkFinishReason::Error(
            from_peer.to_string(),
        ))))
        .expect("an Error finish_reason must produce an error");

        let (status, _msg, kind) = crate::error::classify_error(&err);
        assert_eq!(
            status,
            axum::http::StatusCode::BAD_REQUEST,
            "the caller's own over-long prompt must not be reported as this server breaking; got {err:?}"
        );
        assert_eq!(kind, "invalid_request_error");

        // The second consequence: blame. `failure_is_penalty_worthy` exempts
        // `Validation` but not `Inference`, so flattening the class docks a
        // peer for a mistake that was the caller's.
        assert!(
            matches!(err, SwarmError::Validation(_)),
            "must stay Validation so the peer is not penalised; got {err:?}"
        );
    }

    #[test]
    fn a_result_that_carries_no_error_is_left_alone() {
        assert!(peer_error_from_result(&result_with(None)).is_none());
        assert!(peer_error_from_result(&result_with(Some(NetworkFinishReason::Stop))).is_none());
        assert!(
            peer_error_from_result(&result_with(Some(NetworkFinishReason::MaxTokens))).is_none(),
            "a normal finish must not be turned into a failure"
        );
    }

    /// A peer that returns nothing and says nothing keeps the caller's own
    /// message — the helper must not invent a reason it was not given.
    #[test]
    fn an_unexplained_empty_result_is_not_given_a_reason() {
        assert!(peer_error_from_result(&result_with(None)).is_none());
    }

    /// Failing over cannot help when the request itself is the problem: every
    /// holder refuses it identically, and the caller ends up told the model is
    /// under-replicated. Measured on the live swarm 2026-08-30.
    #[test]
    fn a_refusal_of_the_request_itself_does_not_fail_over() {
        let from_peer = "Worker: Validation error: This conversation is 12041 tokens, \
                         longer than the model's limit of 8192";
        let err = every_holder_would_refuse(from_peer)
            .expect("a peer's Validation must stop the failover search");
        assert!(matches!(err, SwarmError::Validation(_)), "got {err:?}");
        assert_eq!(
            crate::error::classify_error(&err).0,
            axum::http::StatusCode::BAD_REQUEST
        );
    }

    /// The control, and the more important half: a peer that is missing the
    /// shard or has died says NOTHING about the next holder, so those must
    /// still fail over. Narrowing this predicate too far would disable
    /// failover itself.
    #[test]
    fn a_failure_that_is_the_peers_own_still_fails_over() {
        for from_peer in [
            "Worker: Shard not found: shard 3 of llama-3.2-3b",
            "Service unavailable: worker died",
            "Internal error: index out of bounds",
            "the card fell out",
            "Model not available: llama-3.2-3b",
        ] {
            assert!(
                every_holder_would_refuse(from_peer).is_none(),
                "{from_peer:?} must remain failover-eligible"
            );
        }
    }

    /// An error whose class cannot be recovered still surfaces the peer's own
    /// words rather than a message written at the call site.
    #[test]
    fn an_unrecognised_peer_error_still_carries_what_the_peer_said() {
        let err = peer_error_from_result(&result_with(Some(NetworkFinishReason::Error(
            "the card fell out".to_string(),
        ))))
        .expect("an Error finish_reason must produce an error");
        assert!(err.to_string().contains("the card fell out"), "got {err:?}");
    }
}
