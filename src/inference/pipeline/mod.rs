//! Distributed inference pipeline. Orchestrates per-token generation across
//! pipeline segments (on-node + remote peers), with tensor-parallel, vision
//! pre-computation, streaming, and failover support.
//!
//! The `PipelineExecutor` struct lives here; per-phase methods live in sibling
//! files (`local`, `distributed`, `vision`, `prompt`, `tensor_parallel`).

pub(crate) mod distributed;
mod dsd;
mod dsd_flow;
mod dsd_stream;
mod dsd_tree;
mod engine_drafter;
pub(crate) mod handoff_prompt_cache;
#[cfg(test)]
pub(crate) use engine_drafter::note_split_wanting_a_drafter;
#[cfg(test)]
pub(crate) use engine_drafter::DrafterStep;
pub(crate) use engine_drafter::{
    drafter_to_acquire, drafter_to_leave_room_for, DrafterToAcquire, DRAFTER_WANTED_FOR,
};
mod local;
pub(crate) use local::LoadAllowance;
mod local_generate;
mod ngram_only_spec;
mod prompt;
mod prompt_admission;
mod prompt_chunks;
pub(crate) mod remote_generate;
mod requester_guesses;
mod speculative;
mod split_prompt_cache;
pub(crate) use split_prompt_cache::{believes_any_of, held_openings, HeldOpening};
mod tensor_parallel;
mod vision;

use std::sync::Arc;

use tokio::sync::mpsc;

use crate::daemon::state::WaiterKey;
use crate::daemon::SharedState;
use crate::error::SwarmError;
use crate::inference::router::{InferenceOutput, StreamingTokenTx, TokenLogProbEntry};
use crate::types::{
    InferenceRequest, LayerResult, ModelId, NetworkCommand, NetworkFinishReason, PipelineAssignment,
};

pub(crate) use prompt::render_prompt_from_header;
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
///
/// **Except a prompt longer than THAT node serves.** How long a conversation a
/// node serves is the model's declared context capped by the node's OWN
/// ceiling (`max_seq_len_override`, else the shipped 8192), so a machine on the
/// default refuses what one raised to 65536 would run. Reported from the field
/// 2026-09-25: a segment with a standby was abandoned because an 8192 peer
/// refused an 8560-token prompt the coordinator itself served at 65536. Such a
/// refusal is one only when its limit is the MODEL's (`declared_context`) — no
/// node serves more than that — and when the model's limit is unknown it is
/// given the benefit of the doubt: a standby that refuses the same way costs a
/// round trip, and `failover_segment` still ends on the caller's own 400 rather
/// than the "too few machines" 503 this function exists to prevent.
pub(super) fn every_holder_would_refuse(
    err_msg: &str,
    declared_context: Option<usize>,
) -> Option<SwarmError> {
    match crate::error::reclassify_flattened_error(err_msg) {
        Some(e @ SwarmError::Validation(_)) => {
            match crate::error::served_context_refusal(err_msg) {
                Some(refusal) if declared_context.is_none_or(|d| refusal.limit < d) => None,
                _ => Some(e),
            }
        }
        _ => None,
    }
}

/// Why a CHAINED run of a check failed, from what its wait came back with —
/// `None` for an unchained forward, or an answer. Any failure counts: the tail
/// answers for the whole run, so none of them says which hop it was.
fn chained_run_failure(
    chain: &[crate::types::ChainHop],
    waited: &Result<LayerResult, SwarmError>,
) -> Option<String> {
    if chain.is_empty() {
        return None;
    }
    match waited {
        Err(e) => Some(e.to_string()),
        Ok(result) => match &result.finish_reason {
            Some(NetworkFinishReason::Error(msg)) => Some(msg.clone()),
            _ => None,
        },
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
/// The most a coordinator waits for one warm forward's answer — also how long a
/// serving node holds a piece of a prompt pass for the piece before it
/// (`forward_streams::PIECE_TURN_WAIT`).
pub(crate) const SEGMENT_TIMEOUT_MAX_SECS: u64 = 600;
/// Cap pending layer results to prevent OOM under sustained load.
const MAX_PENDING_LAYER_RESULTS: usize = 1024;

/// RAII guard that removes a `pending_layer_results` entry on drop unless
/// `disarm()` has been called first. Shared by every coordinator that
/// inserts a oneshot before awaiting a remote tensor result — without it
/// a bare `?` propagation from the wait site leaves a stale entry per
/// failed inference, eventually exhausting `MAX_PENDING_LAYER_RESULTS`.
/// Per gotcha #45 in `memory/MEMORY.md`.
pub(super) struct PendingLayerResultGuard<'a> {
    pub(super) map: &'a dashmap::DashMap<WaiterKey, crate::daemon::state::PendingLayerResult>,
    pub(super) id: WaiterKey,
    pub(super) armed: bool,
}
impl<'a> PendingLayerResultGuard<'a> {
    pub(super) fn new(
        map: &'a dashmap::DashMap<WaiterKey, crate::daemon::state::PendingLayerResult>,
        id: WaiterKey,
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
///
/// `expects_step` is the forward about to be sent — its position and layer
/// range: a result naming anything else is a stale copy, or another segment's
/// answer, and cannot resolve this waiter (#113). Required, so a new caller
/// has to say which forward it waits on.
pub(super) fn register_pending_layer_result(
    map: &dashmap::DashMap<WaiterKey, crate::daemon::state::PendingLayerResult>,
    key: WaiterKey,
    awaiting: Option<crate::types::NodeId>,
    // Every hop of a chained check besides the tail that answers — any of them
    // may report a failure it cannot recover from. Empty for an unchained one.
    chain_members: Vec<crate::types::NodeId>,
    expects_step: Option<crate::daemon::state::ExpectedStep>,
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
        key,
        crate::daemon::state::PendingLayerResult {
            tx,
            awaiting,
            chain_members,
            expects_step,
        },
    );
    let guard = PendingLayerResultGuard::new(map, key);
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

/// Who a split's speculative checks reach (`PipelineExecutor::exposure_of_a_check`).
pub(super) struct CheckExposure {
    /// A guess the check would send — naming more than it reaches, never
    /// fewer: as token ids to a remote first segment (packed in its input), to
    /// the tail that walks them (`LayerForward::draft_tokens`) and, where the
    /// check may chain, to every hop of the run ending there that hands them
    /// on (`verify_onward`) — every remote segment after the last one run
    /// here.
    pub shown: crate::inference::owner_replies::Exposure,
    /// The reply — naming fewer than it reached, never more: as ids only a
    /// remote first segment, sent every token but the last as its input
    /// (unless prompt privacy embeds them here), and a tail that walks
    /// (`peer_walks_at_tail`): it is sent every guess and draws every other
    /// token, the first on the prompt pass. A tail that cannot walk returns
    /// distributions and is not counted. Never a middle segment: hidden
    /// states only, outside a chained check's guesses.
    pub seen: crate::inference::owner_replies::Exposure,
}

/// What the segment that samples needs to walk a verify's drafts itself
/// (`LayerForward::spec_walk_at_tail`): the drafts, the caller's sampler, and
/// the history its penalties read. The walk IS sampling, so these travel
/// together or not at all.
#[derive(Clone, Copy)]
pub(super) struct TailWalk<'a> {
    pub drafts: &'a [u32],
    pub sampling: &'a crate::types::SamplingParams,
    pub generated: &'a [u32],
    /// The request's shared noise seed, when its drafts were drawn with it
    /// (`inference::coupled_noise`); only a tail advertising
    /// `features::COUPLED_SAMPLING` is sent it.
    pub coupling: Option<u64>,
    /// A tree of guesses' shape (#134): each row's parent in the wire form of
    /// `LayerForward::draft_parents`, one per row of the forward; empty for a
    /// chain. It travels with the walk because only a tail that walks may be
    /// sent a tree (a vocabulary per row would be tens of megabytes), and
    /// every segment is sent it, since every segment's layers read the shape.
    pub parents: &'a [u16],
}

/// What a verify round came back with. A tail that could walk answers with
/// the tokens it kept; one that could not (an older peer, or a caller that
/// asked for none) with a vocabulary per position. [`VerifyReply::accept`] is
/// the one place either becomes accepted tokens.
#[derive(Debug)]
pub(super) enum VerifyReply {
    Logits(Vec<Vec<f32>>),
    /// The drafts the tail kept, then the token it sampled where its own
    /// sample first disagreed (or the bonus after all of them).
    Walked(Vec<u32>),
}

impl VerifyReply {
    /// `(accepted drafts, next token, all accepted)` — by
    /// `sampling::sampled_accept_reject` here for logits, or read off the
    /// tail's walk, which ran that same function where the logits were.
    ///
    /// Refuses what no honest tail sends, rather than emitting it: logits that
    /// are not finite (the rule used to hand back token 0 as the "bonus" and a
    /// caller streamed it), too few positions, or a walk that claims to have
    /// kept a token that was never drafted.
    ///
    /// `coupling` is the shared noise and the absolute position row 0 predicts,
    /// for a reply whose logits are walked HERE — the same draw the tail makes.
    pub(super) fn accept(
        self,
        drafts: &[u32],
        sampling: &crate::types::SamplingParams,
        generated: &[u32],
        coupling: Option<(&crate::inference::coupled_noise::CoupledNoise, u64)>,
    ) -> Result<(Vec<u32>, u32, bool), SwarmError> {
        match self {
            VerifyReply::Logits(rows) => {
                if rows.len() < drafts.len() + 1 {
                    return Err(SwarmError::Inference(format!(
                        "spec verify: {} logit rows for {} drafts",
                        rows.len(),
                        drafts.len()
                    )));
                }
                if !crate::inference::sampling::verify_rows_are_finite(drafts.len(), &rows) {
                    return Err(SwarmError::Inference(
                        "spec verify: the last segment returned non-finite logits".into(),
                    ));
                }
                Ok(crate::inference::sampling::sampled_accept_reject(
                    drafts, &rows, sampling, generated, coupling,
                ))
            }
            VerifyReply::Walked(tokens) => {
                let Some((&next, kept)) = tokens.split_last() else {
                    return Err(SwarmError::Inference(
                        "spec verify: the last segment walked and returned no token".into(),
                    ));
                };
                if kept.len() > drafts.len() || kept != &drafts[..kept.len()] {
                    return Err(SwarmError::Inference(format!(
                        "spec verify: the last segment kept {} tokens that were not the {} drafted",
                        kept.len(),
                        drafts.len()
                    )));
                }
                Ok((kept.to_vec(), next, kept.len() == drafts.len()))
            }
        }
    }
}

/// Build the `LayerForward` envelope for a speculative-verify send.
/// Shared by `speculative.rs::send_verify_batch` (single-segment) and
/// `super::forward_verify_through_segments` (multi-segment) so adding
/// a `LayerForward` field can't drift between the two paths. The
/// `spec_logits_requested` flag is set uniformly; the receiver gates
/// emission on `is_last`. `walk` is for the LAST segment only, and only one
/// that can walk (`forward_verify_through_segments` decides).
pub(super) fn build_spec_verify_forward(
    request_id: uuid::Uuid,
    index_pos: u32,
    activations: Vec<u8>,
    segment: &crate::types::PipelineSegment,
    truncate_kv_to: Option<u32>,
    walk: Option<&TailWalk<'_>>,
    stream_seq: Option<u32>,
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
        // The penalties' history, for a tail that samples — and only when the
        // sampler reads it, so a request without penalties sends nothing.
        generated_ids: walk
            .filter(|w| crate::inference::sampling::sampler_reads_history(w.sampling))
            .map(|w| w.generated.to_vec())
            .unwrap_or_default(),
        adapter_id: None,
        // The receiver gates spec-logits emission on
        // `spec_logits_requested && is_last`, not on `draft_tokens`. The
        // draft IDs are also already encoded in `activations` for segment 0,
        // but a tail past it receives hidden states, so a walk names them.
        draft_tokens: walk.map(|w| w.drafts.to_vec()).unwrap_or_default(),
        spec_logits_requested: true,
        spec_walk_at_tail: walk.is_some(),
        coupling_seed: walk.and_then(|w| w.coupling),
        // Only a streamed check (`dsd_stream`) numbers its forwards, and only
        // for a peer advertising `features::STREAMED_VERIFY`.
        stream_seq,
        truncate_kv_to,
        // A check never carries a prompt pass's hint: it is not a prompt pass.
        prompt_cache: None,
        chunk_meta: None,
        sampling: walk.map(|w| w.sampling.clone()),
        prompt_span: None,
        image_spans: Vec::new(),
        draft_parents: Vec::new(),
        draft_tree_from: None,
        stream_lookahead: None,
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
    walk: Option<TailWalk<'_>>,
    requester: crate::inference::process_pool::Requester,
    // May a run of remote segments take the check straight from each other
    // (`PipelineExecutor::verify_may_chain`)? Each hop must also advertise
    // `features::CHAINED_VERIFY`; anything else is sent the old way.
    may_chain: bool,
    // Set when a chained run of this check failed. The check is re-run
    // unchained here, and the caller passes its `chaining_disabled` so the
    // rest of the request is not chained again.
    chain_failed: &mut bool,
) -> Result<VerifyReply, SwarmError> {
    let num_segments = segments.len();
    let local_node_id = shared_state.identity.node_id().clone();
    let mut may_chain = may_chain;
    // Segments up to this index re-run a chained run that failed: the hops
    // that ran wrote this check's rows, so each is cut back to `index_pos`.
    let mut rerun_through: Option<usize> = None;
    // A guess check — or an n-gram miss's one-token step — is one round of
    // the reply (FUTURE_WORK #3 (d)).
    if !verify_tokens.is_empty() {
        shared_state.note_decode_round(request_id);
    }
    // A tree of guesses goes to every segment, so every one must read it and
    // the tail must walk it (`tree_check_possible`, which the caller asked
    // before growing one): refused here, before anything is sent, if not.
    let tree_parents: &[u16] = walk.map_or(&[], |w| w.parents);
    if !tree_parents.is_empty() && !tree_check_possible(shared_state, segments) {
        return Err(SwarmError::Validation(
            "a tree of guesses was asked of segments that cannot all check one".into(),
        ));
    }

    let mut activation_bytes: Vec<u8> = pack_verify_tokens_to_le_bytes(verify_tokens);

    let mut idx = 0usize;
    while idx < num_segments {
        let segment = &segments[idx];
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
        let local = target_peer_bytes.is_none();
        // A run of remote segments after this one that take the check from
        // each other: one trip around them instead of one per segment through
        // here (`features::CHAINED_VERIFY`). The run's TAIL answers, and when it
        // is the model's last segment it walks the guesses. Empty — the old
        // way, always correct — for a local segment, a hop without the bit, or
        // when the caller may not chain.
        let chain: Vec<crate::types::ChainHop> = if local || !may_chain {
            Vec::new()
        } else {
            plan_chain(
                segments,
                idx,
                &local_node_id,
                |n| {
                    shared_state.peer_supports_pipeline_chain(n)
                        && shared_state.peer_advertises_feature(
                            n,
                            swarmllm_types::node::features::CHAINED_VERIFY,
                        )
                },
                shared_state.cfg().inference.max_chain_hops as usize,
            )
        };
        let run_end = idx + chain.len();
        let is_last = run_end == num_segments - 1;
        // The node that ANSWERS this forward: the run's tail.
        let answering = chain
            .last()
            .map(|h| h.node_id.clone())
            .unwrap_or_else(|| segment.node_id.clone());
        // The sampler walks where the logits are — our own worker always can,
        // a peer only when it says so. Otherwise it sends every position's
        // vocabulary back and `VerifyReply::accept` walks them here.
        let tail_walk = walk
            .filter(|_| is_last && (local || peer_walks_at_tail(shared_state, &answering)))
            // The seed only to a tail that samples with it; any other walks with
            // its own draw — still exact, just less often in agreement.
            .map(|w| TailWalk {
                coupling: w.coupling.filter(|_| {
                    local
                        || shared_state.peer_advertises_feature(
                            &answering,
                            swarmllm_types::node::features::COUPLED_SAMPLING,
                        )
                }),
                ..w
            });

        let truncate_here = if rerun_through.is_some_and(|end| idx <= end) {
            Some(index_pos)
        } else {
            truncate_kv_to
        };
        // Rebuildable: a peer that refuses this unopened is sent it again once
        // the link is re-keyed (`local::ResendOnRefusal`).
        let rebuild_forward = || {
            let mut forward = build_spec_verify_forward(
                request_id,
                index_pos,
                activation_bytes.clone(),
                segment,
                truncate_here,
                tail_walk.as_ref(),
                None,
            );
            forward.draft_parents = tree_parents.to_vec();
            if !chain.is_empty() {
                forward.chain = chain.clone();
                // The reply-to the tail answers (the `0x07` trailer), as on a
                // chained decode step: unchained, a receiver answers its
                // sender, and in a chain that is the hop before it.
                forward.requester_node_id = Some(local_node_id.0);
            }
            forward
        };
        let forward = rebuild_forward();

        let result = if let Some(peer_bytes) = target_peer_bytes {
            let (rx, mut pending_guard) = register_pending_layer_result(
                &shared_state.pending_layer_results,
                WaiterKey::request(request_id),
                Some(answering.clone()),
                // Any hop may refuse; the head too, which `awaiting` does not
                // cover once a chain is planned (`distributed.rs`'s rule).
                if chain.is_empty() {
                    Vec::new()
                } else {
                    std::iter::once(segment.node_id.clone())
                        .chain(chain.iter().map(|h| h.node_id.clone()))
                        .collect()
                },
                // Every hop forwards this same position; the tail answers with
                // ITS range and any hop may refuse with its own.
                Some(crate::daemon::state::ExpectedStep {
                    index_pos,
                    layer_ranges: std::iter::once(segment.layer_range)
                        .chain(chain.iter().map(|h| h.layer_range))
                        .collect(),
                }),
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
                // unwinds).
                pending_guard.disarm();
                shared_state
                    .pending_layer_results
                    .remove(&WaiterKey::request(request_id));
                return Err(SwarmError::Network(
                    "fwd_verify_through_segments: send dropped".into(),
                ));
            }

            // The whole run's layers: its tail answers for all of them.
            let num_layers = std::iter::once(segment.layer_range)
                .chain(chain.iter().map(|h| h.layer_range))
                .map(|(a, b)| b - a)
                .sum::<u32>();
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
            let waited = PipelineExecutor::wait_for_result(
                shared_state,
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
                // A chained run's refusal may come from any hop
                // (`arch-scheduling.md` § "A forward the peer could not open").
                if chain.is_empty() {
                    local::ResendOnRefusal::SameForward {
                        network_tx,
                        target_peer_bytes: peer_bytes,
                        rebuild: &rebuild_forward,
                    }
                } else {
                    local::ResendOnRefusal::Never(
                        "a chained check's refusal may be any hop's, and the run is not re-sent",
                    )
                },
            )
            .await;
            // A chained run that failed is re-run unchained, as the plain decode
            // loop re-runs a chained step (`distributed.rs`): it does not say
            // WHICH hop failed — a middle one not yet connected to the next is
            // the common case, before the swarm has introduced them — and
            // unchained, each segment answers for itself. The run consumed
            // nothing here: `activation_bytes` is still its input. Ending the
            // reply instead lost every chained check's request to a missing
            // connection (the 3-machine rig, 2026-10-20, #318).
            if let Some(reason) = chained_run_failure(&chain, &waited) {
                drop(pending_guard);
                tracing::warn!(
                    %request_id,
                    segment = idx,
                    hops = chain.len(),
                    error = %reason,
                    "chained check failed — re-running it unchained for the rest of the request"
                );
                *chain_failed = true;
                may_chain = false;
                rerun_through = Some(run_end);
                continue;
            }
            let result = waited?;
            pending_guard.disarm();
            // A SINGLE-token verify is an ordinary decode step wearing a
            // verify's name — `ngram_only_spec` sends `vec![last_token]` on a
            // cascade MISS — so it is exactly the sample `peer_speed` wants,
            // and this is the one place both verify callers funnel through.
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
            // One peer's sample only: a chained run's time is several peers'.
            if verify_tokens.len() == 1 && chain.is_empty() {
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
            // A verify step of a request this node coordinates: the owner's,
            // or the peer's that delegated it (`worker_requester`).
            let seg_start = std::time::Instant::now();
            let result = shared_state
                .model_process_pool
                .forward_for_request(forward, None, requester)
                .await?;
            // Measure ourselves too, exactly as the branch above measures a
            // peer and `forward_through_segments_inner` measures a local
            // segment. This loop is the default path for a split, and without
            // this our own node was never timed on a decode step there — its
            // price came only from the prefill fallback, and one cold sample of
            // that priced an RTX 3070 at 196 ms a layer (2026-09-27). Same
            // single-token rule, for the same reason.
            if verify_tokens.len() == 1 {
                shared_state.record_peer_segment_latency(
                    &segment.node_id,
                    &segment.shard_id.model_id,
                    crate::daemon::state::WorkKind::Decode,
                    seg_start.elapsed().as_millis() as u64,
                    segment.layer_range.1 - segment.layer_range.0,
                    result.activations.len(),
                );
            }
            result
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
            // Read the shape the tail ANSWERED in, not the one we asked for:
            // either is a correct reply, and `accept` handles both.
            if !result.spec_logits.is_empty() {
                return Ok(VerifyReply::Logits(result.spec_logits));
            }
            if tail_walk.is_some() && !result.token_ids.is_empty() {
                return Ok(VerifyReply::Walked(result.token_ids));
            }
            return Err(SwarmError::Inference(
                "spec verify: last segment returned neither logits nor a walk".into(),
            ));
        }

        // Intermediate: feed hidden state to next segment.
        check_intermediate_activations(idx, &activation_bytes, &result.activations)
            .map_err(|e| SwarmError::Inference(format!("spec verify {e}")))?;
        activation_bytes = result.activations;
        idx = run_end + 1;
    }
    unreachable!("loop returns on the last segment")
}

/// A check's answer from a segment that is not the last, before it is fed to
/// the next: `Err` names what is wrong with it. Asked by the rounds
/// (`forward_verify_through_segments`) and the stream's boomerang
/// (`dsd_stream`) alike — one rule for what a middle segment may hand on.
///
/// SEC: segment `idx` was sent `sent`. The first segment's input is token ids
/// (8 bytes a position, `pack_verify_tokens_to_le_bytes`) but its OUTPUT is
/// hidden states, so the shapes only have to agree past it. For segment 0 an
/// absolute bound applies instead: the wire's `MAX_ACTIVATION_SIZE` (128 MB)
/// already caps a malicious send, but a check's hidden states are
/// (γ+1) × hidden_dim × bytes/element — well under 64 MB even with γ = 64 and
/// hidden_dim = 12288 in f32. A larger answer is a broken or malicious peer,
/// and would crash the next worker if fed on.
pub(super) fn check_intermediate_activations(
    idx: usize,
    sent: &[u8],
    returned: &[u8],
) -> Result<(), String> {
    const MAX_INTERMEDIATE_ACTIVATION_BYTES: usize = 64 * 1024 * 1024;
    if idx == 0 && returned.len() > MAX_INTERMEDIATE_ACTIVATION_BYTES {
        return Err(format!(
            "segment 0 returned oversized activation: {} bytes (max {MAX_INTERMEDIATE_ACTIVATION_BYTES})",
            returned.len()
        ));
    }
    // By declared SHAPE, never byte length — the same rule and helper as the
    // standard forward loop: an honest peer on the other
    // `activation_compression` setting answers f32 where we sent Q8_0 (#98).
    if idx > 0 && !crate::inference::tensor_util::activation_shape_matches(sent, returned) {
        return Err(format!(
            "segment {idx} returned wrong activation shape: got {} bytes, expected {}",
            returned.len(),
            sent.len()
        ));
    }
    Ok(())
}

/// May this peer be asked to walk a verify's drafts itself? It must read the
/// walk flag, and the trailers the walk samples with — the caller's sampling
/// and the penalties' history. A peer with the walk bit is newer than both, but
/// the three are asked together so no reading of this depends on that order.
/// May a check of a tree of guesses (#134) go through these segments? Every
/// peer's segment must read the tree (`features::DRAFT_TREES`) and the last
/// must walk it — this node's own segments always can (this build). The one
/// answer the round loop asks before growing a tree and
/// `forward_verify_through_segments` asks before sending one; the family is
/// asked separately (`ModelArch::reads_draft_trees`), and a segment whose
/// layers still cannot read one refuses before it runs anything.
pub(super) fn tree_check_possible(
    shared_state: &SharedState,
    segments: &[crate::types::PipelineSegment],
) -> bool {
    let local = shared_state.identity.node_id();
    let Some(last) = segments.last() else {
        return false;
    };
    segments.iter().all(|s| {
        s.node_id == *local
            || shared_state
                .peer_advertises_feature(&s.node_id, swarmllm_types::node::features::DRAFT_TREES)
    }) && (last.node_id == *local || peer_walks_at_tail(shared_state, &last.node_id))
}

/// Does every machine of this plan run its layers of `model_id` whole on a
/// graphics card? There a row of a check costs almost nothing beside the
/// round trip, so a tree's extra rows are nearly free: on the TH↔AU link
/// (2026-10-21) a check cost 497-520 ms plus 2-3 ms a row with the head's
/// layers on its card, against 10-19 ms a row on its processor.
/// `dsd_tree::trees_wanted` reads it. This node answers from its live worker
/// (`ModelProcessPool::runs_whole_on_the_card`), a peer from what it
/// advertises (`NodeCapability::models_run_on_card`); a peer that says
/// nothing is not on a card.
pub(super) fn every_segment_runs_on_a_card(
    shared_state: &SharedState,
    segments: &[crate::types::PipelineSegment],
    model_id: &crate::types::ModelId,
) -> bool {
    every_segment_on_a_card(
        segments,
        shared_state.identity.node_id(),
        || {
            shared_state
                .model_process_pool
                .runs_whole_on_the_card(model_id)
        },
        |node| {
            shared_state.peer_registry.get(node).is_some_and(|p| {
                p.capability
                    .as_ref()
                    .is_some_and(|c| c.models_run_on_card())
            })
        },
    )
}

/// [`every_segment_runs_on_a_card`] over its two readings: this node's part
/// and each peer's.
fn every_segment_on_a_card(
    segments: &[crate::types::PipelineSegment],
    local: &crate::types::NodeId,
    local_on_card: impl Fn() -> bool,
    peer_on_card: impl Fn(&crate::types::NodeId) -> bool,
) -> bool {
    !segments.is_empty()
        && segments.iter().all(|s| {
            if s.node_id == *local {
                local_on_card()
            } else {
                peer_on_card(&s.node_id)
            }
        })
}

/// How far away the farthest peer segment of a plan is (ms): the least round
/// trip recently measured to each peer — the window's minimum
/// (`metrics.network_coord_samples`, as the network coordinate and the near
/// chains read distance), never `PeerInfo::round_trip_ms`, which is a COST:
/// its smoothed figure (fed forwards up to 256 KB, transfer included, gotcha
/// #446) read 2,500-5,000 ms on a 20 ms rig link through a node's first
/// request, and a short link would have been taken for a long one.
/// `dsd_tree::trees_wanted` reads it — trees pay over a long link and lose
/// over a short one. `None` where the plan has no peer segment, or any peer
/// segment is unmeasured: unknown is neither near nor far.
pub(super) fn farthest_segment_round_trip_ms(
    shared_state: &SharedState,
    segments: &[crate::types::PipelineSegment],
) -> Option<u32> {
    farthest_round_trip(segments, shared_state.identity.node_id(), |node| {
        let least_ms = shared_state
            .metrics
            .network_coord_samples
            .get(node)
            .and_then(|samples| samples.min())?;
        Some(least_ms.round() as u32)
    })
}

/// [`farthest_segment_round_trip_ms`] over its one reading.
fn farthest_round_trip(
    segments: &[crate::types::PipelineSegment],
    local: &crate::types::NodeId,
    round_trip: impl Fn(&crate::types::NodeId) -> Option<u32>,
) -> Option<u32> {
    segments
        .iter()
        .filter(|s| s.node_id != *local)
        .try_fold(None, |farthest: Option<u32>, s| {
            Some(farthest.max(Some(round_trip(&s.node_id)?)))
        })
        .flatten()
}

fn peer_walks_at_tail(shared_state: &SharedState, node_id: &crate::types::NodeId) -> bool {
    use swarmllm_types::node::features::{
        FORWARD_GENERATED_IDS, FORWARD_SAMPLING, SPEC_WALK_AT_TAIL,
    };
    shared_state.peer_advertises_feature(
        node_id,
        SPEC_WALK_AT_TAIL | FORWARD_SAMPLING | FORWARD_GENERATED_IDS,
    )
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
        spec_walk_at_tail: false,
        coupling_seed: None,
        stream_seq: None,
        truncate_kv_to: Some(truncate_to),
        prompt_cache: None,
        chunk_meta: None,
        sampling: None,
        prompt_span: None,
        image_spans: Vec::new(),
        draft_parents: Vec::new(),
        draft_tree_from: None,
        stream_lookahead: None,
    }
}

/// What a request has generated so far, kept so a failure can still hand back
/// the work already done.
///
/// **It is filled by the shared emit helpers**, the one place all three
/// speculative coordinators turn accepted tokens into reply text — and a place
/// `streamed_reply_text_goes_through_the_shared_emit_helpers` in
/// `tests/repo_consistency.rs` obliges a new coordinator to use. So nothing has
/// to remember to record a partial: recording is what emitting IS, and a fourth
/// coordinator inherits it. Read at one choke point,
/// `PipelineExecutor::keeping_the_partial`.
///
/// The text is accumulated as it is emitted rather than decoded once at the
/// end, because the reader of a salvage has no decoder in hand — so the
/// salvaged reply is exactly what a streaming caller was shown. **It also holds
/// the UTF-8 carry** (`decode_next`) that every emitted token is decoded
/// through: a character whose bytes span several tokens is held back until it
/// is whole, instead of streaming as U+FFFD once per byte, which is what
/// decoding each token on its own did on all three speculative coordinators.
///
/// The token ids are kept beside the text for two different jobs: they are the
/// honest `completion_tokens` (the trace logged `tokens=0` for these failures
/// while ~220 had been generated), and they are what `may_salvage` asks about.
#[derive(Default)]
pub(super) struct PartialReply {
    inner: std::sync::Mutex<PartialReplyState>,
}

#[derive(Default)]
struct PartialReplyState {
    text: String,
    ids: Vec<u32>,
    prompt_tokens: u32,
    /// Bytes of a character not yet completed by the tokens emitted so far.
    utf8_carry: Vec<u8>,
}

impl PartialReply {
    /// The text the client should be shown for the next emitted token, decoded
    /// through this reply's UTF-8 carry. Empty while a character is still
    /// arriving; every consumer of a streamed event already skips empty text.
    fn decode_next(&self, decoder: &prompt::CachedDecoder, token: u32) -> String {
        let mut state = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        decoder.decode_tokens_streaming(&[token], &mut state.utf8_carry)
    }

    /// Add an emitted batch. `text` is what the client was shown for it.
    fn record(&self, text: &str, ids: &[u32]) {
        let mut state = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        state.text.push_str(text);
        state.ids.extend_from_slice(ids);
    }

    /// How many positions the prompt occupied, for the salvage's usage figure.
    /// Recorded by `extract_model_cache`, which every speculative coordinator
    /// calls — missing it costs the usage number, never the reply.
    fn note_prompt_tokens(&self, prompt_tokens: u32) {
        self.inner
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .prompt_tokens = prompt_tokens;
    }

    /// The reply so far, or `None` if nothing has been generated.
    fn taken(&self) -> Option<(String, Vec<u32>, u32)> {
        let mut state = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        if state.ids.is_empty() {
            return None;
        }
        // A character the last tokens did not finish is rendered as a
        // whole-reply decode renders it, U+FFFD — never dropped. The standard
        // loop flushes its own carry the same way when a reply ends.
        let tail = crate::inference::tokenizer::flush_utf8_carry(&mut state.utf8_carry);
        state.text.push_str(&tail);
        Some((
            std::mem::take(&mut state.text),
            std::mem::take(&mut state.ids),
            state.prompt_tokens,
        ))
    }
}

/// Send a single token's decoded text down the streaming channel,
/// ignoring any send error (matches the "first-token" pattern that
/// fires before the per-round loop). Used by speculative / DSD /
/// ngram-only-spec where the first token is always emitted
/// fire-and-forget — disconnection at this point is fine, the per-round
/// loop will detect it on the next emit.
pub(in crate::inference::pipeline) async fn emit_first_streaming_token(
    partial: &PartialReply,
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
    let text = partial.decode_next(decoder, token);
    // Recorded BEFORE the streaming check, and whatever the channel does with
    // it: a salvage is for the NON-streamed request, so the one case that
    // needs this is the one where `token_tx` is `None` and the old code
    // returned here having done nothing.
    partial.record(&text, &[token]);
    if let Some(tx) = token_tx {
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
    partial: &PartialReply,
    token_tx: &Option<StreamingTokenTx>,
    decoder: &prompt::CachedDecoder,
    tokens: &[u32],
    eos: &std::collections::HashSet<u32>,
    finish_reason: &mut String,
) -> bool {
    // Recorded for every caller, before the streaming check below returns:
    // a non-streamed request is exactly the one a salvage exists for, and it
    // is the one that used to leave here having recorded nothing. EOS is
    // dropped from the text for the same reason it is dropped from the wire,
    // and kept among the ids so the count matches what the decode produced.
    //
    // Each kept token is decoded ONCE, through the reply's UTF-8 carry, and
    // the same pieces are both recorded and streamed — a character split
    // across tokens (or across two batches) arrives whole in either.
    let pieces: Vec<String> = tokens
        .iter()
        .filter(|t| !eos.contains(t))
        .map(|&t| partial.decode_next(decoder, t))
        .collect();
    partial.record(&pieces.concat(), tokens);
    let tx = match token_tx {
        Some(tx) => tx,
        None => return false,
    };
    let mut pieces = pieces.into_iter();
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
        let text = pieces.next().unwrap_or_default();
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
/// What a speculative round hands the reply: the guesses the check kept and
/// the token it sampled after them, cut at the first end-of-reply token (R105:
/// nothing past it reaches anyone) and at the room `max_tokens` leaves.
///
/// The ONE place a round's tokens are trimmed — the n-gram loop, DSD, the
/// continuous stream and the single-segment speculative path all call it. A
/// round's guesses are sized to the room, but the token the check samples after
/// them is not, so a round that kept every guess one short of the budget went
/// one past it: `completion_tokens: 121` for `max_tokens: 120` on the n-gram
/// loop, the default split path (2026-10-02 rig). Only the stream had a room
/// check, inline; the other three had the EOS cut and nothing else.
pub(super) fn round_tokens_for_reply(
    kept: &[u32],
    sampled: u32,
    eos: &std::collections::HashSet<u32>,
    room: usize,
) -> Vec<u32> {
    let mut emitted: Vec<u32> = kept
        .iter()
        .copied()
        .chain(std::iter::once(sampled))
        .collect();
    if let Some(at) = emitted.iter().position(|t| eos.contains(t)) {
        emitted.truncate(at + 1);
    }
    emitted.truncate(room.max(1));
    emitted
}

/// Whose work a LOCAL segment of `request` is, for the worker's admission and
/// thread width: the owner's — a request made to this node's own API — or a
/// peer's, which this node serves within its contribution level.
///
/// Until the delegated split the router only ever received this node's own
/// API requests, so four call sites hardcoded the owner. A split a peer
/// DELEGATED to us (`RoutePlanOverride::lead_here`, `docs/FUTURE_WORK.md` #143)
/// runs through the same router and the same paths, and it is the PEER's: run
/// as the owner's, a peer could have this machine read its prompts on every
/// core the owner is entitled to. The whole-model hand-off it generalises has
/// always run as `Swarm`.
pub(crate) fn worker_requester(
    request: &crate::types::InferenceRequest,
) -> crate::inference::process_pool::Requester {
    requester_for_override(request.route_override.as_ref())
}

/// [`worker_requester`] for a reader that holds the request's override rather
/// than the request — the scheduler bounding this node by the memory cap the
/// spawn will weigh the same request against (#266).
pub(crate) fn requester_for_override(
    route_override: Option<&crate::inference::route_override::RoutePlanOverride>,
) -> crate::inference::process_pool::Requester {
    if route_override.is_some_and(|o| o.lead_here) {
        crate::inference::process_pool::Requester::Swarm
    } else {
        crate::inference::process_pool::Requester::Owner
    }
}

pub(super) fn fastpath_request_disqualified(exec: &PipelineExecutor) -> bool {
    if !exec.assignment.tp_groups.is_empty() {
        return true;
    }
    // Prompt privacy (`inference.local_embedding_privacy`): every fast path
    // sends the first segment TOKEN IDS — none embeds the prompt here — so
    // none may run a plan whose first layers another computer may read. Only
    // the plain loop embeds (`execute_distributed`), and it refuses where it
    // cannot (#316: the n-gram path served every such request it was offered).
    if exec.shared_state.local_embedding_privacy() && exec.segment_zero_may_leave_this_node() {
        return true;
    }
    request_disqualifies_fast_paths(&exec.request)
}

impl PipelineExecutor {
    /// Could the first segment's input — the prompt — be handed to another
    /// computer under this plan: the first segment runs elsewhere, or a
    /// stand-in elsewhere could take over layer 0. The question prompt privacy
    /// (`inference.local_embedding_privacy`) asks before it lets token ids go.
    pub(super) fn segment_zero_may_leave_this_node(&self) -> bool {
        let me = self.shared_state.identity.node_id();
        self.assignment
            .segments
            .first()
            .is_some_and(|first| first.node_id != *me)
            || self
                .assignment
                .standbys
                .iter()
                .any(|s| s.layer_range.0 == 0 && s.node_id != *me)
    }

    /// The embedder the prompt pass uses, if this plan needs one — the one
    /// answer `execute_distributed` asks (FUTURE_WORK #316).
    ///
    /// Only prompt privacy (`inference.local_embedding_privacy`) embeds, and
    /// only where the first segment's input could reach another computer
    /// (`segment_zero_may_leave_this_node`). Where every first layer runs
    /// here, the token ids never reach a wire and the worker reads them as it
    /// always has: an embedded first segment would skip what only the token
    /// path does — the split prompt cache, the prompt read in pieces, and,
    /// before #316's review, a LoRA adapter and the tensor-parallel embed step.
    /// Where the first layers may leave and the embedder cannot be loaded here
    /// (this computer does not hold the model's first part), the request is
    /// refused: sending token ids instead would work perfectly and break the
    /// one promise the setting makes.
    ///
    /// "Start and finish on this computer" never embeds: its first segment is
    /// this node by construction. It read `local_embedders` until #316, which
    /// no released build ever filled; filled now by prompt privacy, an entry
    /// left after that setting is turned off would have embedded a first
    /// segment running here.
    pub(super) async fn prompt_embedder(
        &self,
        request_id: uuid::Uuid,
    ) -> Result<Option<std::sync::Arc<crate::inference::local_embedder::LocalEmbedder>>, SwarmError>
    {
        if !self.shared_state.local_embedding_privacy() || !self.segment_zero_may_leave_this_node()
        {
            return Ok(None);
        }
        let model_id = &self.assignment.segments[0].shard_id.model_id;
        match crate::inference::local_embedder::local_embedder_for(&self.shared_state, model_id)
            .await
        {
            Ok(embedder) => Ok(Some(embedder)),
            Err(e) => {
                tracing::warn!(
                    request_id = %request_id,
                    model = %model_id,
                    error = %e,
                    "Refusing prompt privacy: this computer cannot embed the prompt \
                     itself, and the model's first layers may run on another computer"
                );
                Err(SwarmError::EmbeddingPrivacyUnavailable {
                    model_id: model_id.0.clone(),
                })
            }
        }
    }

    /// The tensor-parallel group that runs `segment`, if any. Not for a
    /// request whose pictures attend both ways (#115 (2b)), for any of its
    /// forwards: a group's layers read the prompt in order, and its primary is
    /// this node, which runs the range plainly instead. Not for a first
    /// segment whose input is already embedded (#316): both of the group's
    /// paths read their input as the prompt's TEXT — the embed step a member
    /// runs, and the plain forward a non-member hands its worker — so it goes
    /// the ordinary way, which says the input is embedded.
    pub(super) fn tensor_parallel_group_for(
        &self,
        segment: &crate::types::PipelineSegment,
        idx: usize,
        pre_embedded: bool,
    ) -> Option<crate::types::TensorParallelGroup> {
        if !self.image_spans.is_empty() || (idx == 0 && pre_embedded) {
            return None;
        }
        self.assignment
            .tp_groups
            .iter()
            .find(|g| {
                g.layer_range.0 <= segment.layer_range.0 && g.layer_range.1 >= segment.layer_range.1
            })
            .cloned()
    }
}

/// [`fastpath_request_disqualified`]'s half that is about the REQUEST, not
/// the plan — what the planner can ask before a plan exists
/// (`remote_generate::request_may_be_led_by_its_head`).
pub(super) fn request_disqualifies_fast_paths(request: &crate::types::InferenceRequest) -> bool {
    !FastPathNeeds::of(&request.messages, request.lora_adapter.as_deref()).fast_paths_serve()
}

/// What a request carries that no fast path can serve: a picture (only the
/// pipeline's first segment puts image rows in the prompt,
/// `vision::splice_image_rows`; the worker's whole-model `Generate` has no
/// image path) or a LoRA adapter (that `Generate` has none, FUTURE_WORK #110).
///
/// The one rule for the pipeline's fast paths ([`request_disqualifies_fast_paths`])
/// AND the API surfaces' local fast path (`SharedState::local_fast_path_for`,
/// where it is a REQUIRED argument). Until 2026-10-19 that gate never saw the
/// request: the OpenAI surface checked the adapter beside it, neither surface
/// checked for pictures, and a picture sent to a node holding the whole model
/// reached it as the text `<__media__>` — the model then described an image it
/// was never shown (#115 (2a), gotcha #901).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FastPathNeeds {
    images: bool,
    adapter: bool,
}

impl FastPathNeeds {
    /// What THIS request needs, read off its messages and adapter.
    pub fn of(messages: &[crate::types::ChatMessage], lora_adapter: Option<&str>) -> Self {
        Self {
            images: crate::inference::vision::has_images(messages),
            adapter: lora_adapter.is_some(),
        }
    }

    /// A question about the model with no request behind it (the model list's
    /// "runs here"): nothing a request could carry is asked about.
    pub const NO_REQUEST: Self = Self {
        images: false,
        adapter: false,
    };

    /// May a fast path serve it?
    pub fn fast_paths_serve(self) -> bool {
        !self.images && !self.adapter
    }
}

#[cfg(test)]
mod fast_path_needs_tests {
    use super::FastPathNeeds;
    use crate::types::{ChatMessage, ImageData, Role};

    fn turn(images: Vec<ImageData>) -> ChatMessage {
        ChatMessage {
            role: Role::User,
            content: "look".into(),
            images,
            tool: Default::default(),
        }
    }

    /// A picture anywhere in the conversation, or an adapter, keeps a request
    /// off every fast path — the API's local one included, which once sent a
    /// picture to the whole-model `Generate` as the text `<__media__>`.
    #[test]
    fn a_picture_or_an_adapter_keeps_a_request_off_the_fast_paths() {
        let picture = ImageData {
            rgb_bytes: vec![0; 3],
            width: 1,
            height: 1,
        };
        assert!(FastPathNeeds::of(&[turn(vec![])], None).fast_paths_serve());
        assert!(!FastPathNeeds::of(&[turn(vec![]), turn(vec![picture])], None).fast_paths_serve());
        assert!(!FastPathNeeds::of(&[turn(vec![])], Some("adapter")).fast_paths_serve());
        assert!(FastPathNeeds::NO_REQUEST.fast_paths_serve());
    }
}

/// May a split of this request that this node drives guess ahead at all — is
/// any guess-check path switched on (the drafter's, `dsd`; the n-gram loop's,
/// which needs no drafter), is the request free of what stands every fast
/// path aside, and can the model take a refused guess back — any model by the
/// n-gram loop, one without recurrent state by the drafter's too
/// (`speculation_can_roll_back`)? What each path then asks of the PLAN and
/// the drafter is its own (`checks_can_be_taken_back`), and the yield it got
/// is measured, not assumed.
///
/// The ONE request-side answer for FUTURE_WORK #3 (d): the planner prices a
/// split by its model's measured tokens per round only for a request this
/// answers yes for (`PromptPlan::with_guessing_ahead`), and
/// `PipelineExecutor::execute` records a split's rounds only for one — so the
/// figure describes the requests it prices. A request with an image or an
/// adapter runs one round a token and would otherwise lower it.
pub(crate) fn request_may_guess_ahead(
    state: &SharedState,
    request: &crate::types::InferenceRequest,
) -> bool {
    let cfg = state.cfg();
    let drafter = cfg.inference.speculative_decoding
        && cfg.inference.decentralized_spec_decoding
        && distributed::speculation_can_roll_back(
            state
                .gguf_meta_for(&request.model_id)
                .map(|m| m.architecture.clone())
                .as_deref(),
        );
    (drafter || cfg.inference.ngram_lookup_enabled) && !request_disqualifies_fast_paths(request)
}

/// May a check of guesses on this plan be taken back INTO — truncated to its
/// last accepted token? Always for a model without recurrent state. One with
/// (Qwen 3.5's DeltaNet) takes back only its last check, by replaying the rows
/// it kept (`SsmState::rewind_to`, #228 (5)), and only on a node that does:
/// every peer segment must advertise `features::RECURRENT_REWIND` — an older
/// one refuses the truncate, and the forward after it. Asked before the n-gram
/// loop and at each of its rounds, of the LIVE segments, since a stand-in may
/// not advertise it.
pub(super) fn checks_can_be_taken_back(exec: &PipelineExecutor) -> bool {
    let architecture = exec
        .shared_state
        .gguf_meta_for(&exec.request.model_id)
        .map(|m| m.architecture.clone());
    distributed::speculation_can_roll_back(architecture.as_deref())
        || every_peer_segment(
            &exec.assignment.segments,
            exec.shared_state.identity.node_id(),
            |node| {
                exec.shared_state
                    .peer_advertises_feature(node, swarmllm_types::node::features::RECURRENT_REWIND)
            },
        )
}

/// Does every segment of a plan that is not this node's pass `peer`?
fn every_peer_segment(
    segments: &[crate::types::PipelineSegment],
    local: &crate::types::NodeId,
    peer: impl Fn(&crate::types::NodeId) -> bool,
) -> bool {
    segments
        .iter()
        .all(|s| s.node_id == *local || peer(&s.node_id))
}

/// Common preconditions for any speculative-decoding fast path
/// (`speculative.rs`'s single-segment Item 2 and `dsd.rs`'s
/// multi-segment Item 12). Returns `true` when the request is
/// eligible *so far* — callers add their own segment-shape check on
/// top. Draft model availability and the non-encryption / non-LoRA /
/// non-vision baseline are required by both paths; the bool flag config
/// is per-path so it stays inline. Temperature is NOT common: Item 2
/// accepts by argmax and requires 0 itself, DSD walks with the sampler.
/// Read live (`cfg()`), so a setting changed in Settings applies to the
/// next request rather than the next restart (#281).
pub(super) fn speculative_common_eligible(exec: &PipelineExecutor) -> bool {
    speculation_allowed(exec) && exec.shared_state.cfg().inference.draft_model_path.is_some()
}

/// [`speculative_common_eligible`] short of WHICH drafter guesses: the switch,
/// the peers' support, and the request's own disqualifiers. DSD asks this and
/// then chooses between llama.cpp's drafter and a model this node holds
/// (`pipeline::engine_drafter`); Item 2 can only use llama.cpp's.
pub(super) fn speculation_allowed(exec: &PipelineExecutor) -> bool {
    let cfg = exec.shared_state.cfg();
    if !cfg.inference.speculative_decoding {
        return false;
    }
    if !exec.assignment.supports_speculative {
        return false;
    }
    if fastpath_request_disqualified(exec) {
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
    /// This request's complete stop set — see `reply_stops`. Warmed by
    /// `build_prompt_with_header`, the choke point every prompt passes
    /// through, so it always describes the template the prompt was actually
    /// built from and finalising costs no second header parse.
    pub(super) reply_stops: tokio::sync::OnceCell<Vec<String>>,
    /// What this request has generated so far — see [`PartialReply`]. Written
    /// by the shared emit helpers, read once by `keeping_the_partial`.
    pub(super) partial_reply: PartialReply,
    /// Content tokens a hand-off has delivered to this request so far. A
    /// delegated split that failed before delivering any is run here instead
    /// (`try_delegated_split`); one that delivered some failed the request.
    pub(super) hand_off_emitted: usize,
    /// While a split's prompt pass keeps the prompt between turns
    /// (`split_prompt_cache`, FUTURE_WORK #10): what every segment's forward of
    /// that pass carries. Set and cleared around the pass by
    /// `forward_through_segments`; `None` otherwise.
    pub(super) prompt_cache_hint: Option<crate::types::PromptCacheHint>,
    /// The positions each picture's rows take in this request's prompt, for a
    /// family that reads them both ways (`image_spans_for`, #115 (2b)). Set
    /// once before the prompt pass; every forward of that pass carries it, to
    /// every segment — each one's layers need it.
    pub(super) image_spans: Vec<crate::types::PromptSpan>,
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
            reply_stops: tokio::sync::OnceCell::new(),
            partial_reply: PartialReply::default(),
            hand_off_emitted: 0,
            prompt_cache_hint: None,
            image_spans: Vec::new(),
        }
    }

    /// May this request's speculative CHECKS travel a chain of remote segments
    /// (`forward_verify_through_segments`' `may_chain`)? What a chained decode
    /// step needs — chaining switched on, no chained run failed earlier in this
    /// request — and `SWARMLLM_CHAIN_VERIFY=0` is the control arm that keeps
    /// them star-shaped. A check's history travels WITH it (the head's forward
    /// carries the tail's `generated_ids`, handed on by every hop), so unlike a
    /// decode step it may chain with penalties set.
    pub(super) fn verify_may_chain(&self) -> bool {
        static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let switched_off = *OFF.get_or_init(|| {
            std::env::var("SWARMLLM_CHAIN_VERIFY")
                .map(|v| v == "0" || v.eq_ignore_ascii_case("false"))
                .unwrap_or(false)
        });
        !switched_off
            && !self.chaining_disabled
            && self.shared_state.cfg().inference.pipeline_chaining
    }

    /// Who a speculative check's guesses reach, and who this reply has
    /// reached, from the LIVE segments (failover rewrites them) — what the
    /// owner's earlier replies are copied and kept by (`owner_replies`, #337).
    /// The flow's own is `dsd_flow::flow_exposure`.
    pub(super) fn exposure_of_a_check(&self) -> CheckExposure {
        let local = self.shared_state.identity.node_id();
        let segments = &self.assignment.segments;
        let peers: Vec<crate::types::NodeId> = segments
            .iter()
            .filter(|s| s.node_id != *local)
            .map(|s| s.node_id.clone())
            .collect();
        let remote_id = |s: Option<&crate::types::PipelineSegment>| {
            s.filter(|s| s.node_id != *local).map(|s| s.node_id.clone())
        };
        let first = remote_id(segments.first());
        let tail = remote_id(segments.last());
        // Drafts go as ids only to the first segment and to the run that
        // ends at the last one. Chained, that run is any of the remote
        // segments back to the nearest one run here (`plan_chain` never
        // crosses this node), so `[me, a, me]`'s `a` is never named.
        let shown_ids: Vec<crate::types::NodeId> = if self.verify_may_chain() {
            first
                .iter()
                .cloned()
                .chain(
                    segments
                        .iter()
                        .rev()
                        .take_while(|s| s.node_id != *local)
                        .map(|s| s.node_id.clone()),
                )
                .collect()
        } else {
            first.iter().chain(tail.iter()).cloned().collect()
        };
        // A tail that cannot walk (an older peer) sends its vocabularies
        // here and draws none of a check's tokens.
        let seen_ids = first
            .filter(|_| !self.shared_state.local_embedding_privacy())
            .into_iter()
            .chain(tail.filter(|t| peer_walks_at_tail(&self.shared_state, t)));
        // A tensor-parallel group's other machines share a segment's work:
        // counted as reached by a guess, never as having seen the reply.
        let grouped = self
            .assignment
            .tp_groups
            .iter()
            .flat_map(|g| g.nodes.iter())
            .filter(|n| *n != local)
            .cloned();
        CheckExposure {
            shown: crate::inference::owner_replies::Exposure::new(
                peers.iter().cloned().chain(grouped),
                shown_ids,
            ),
            seen: crate::inference::owner_replies::Exposure::new(peers, seen_ids),
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
    ///
    /// Whatever the outcome, the peers this attempt ran segments on — as
    /// planned, and any that took a segment over — are recorded for the end of
    /// the REQUEST, when they are told it is over (#238). Not now: a retry of
    /// the same request id may still use them (gotcha #749).
    pub async fn execute(
        &mut self,
        token_tx: Option<StreamingTokenTx>,
    ) -> Result<InferenceOutput, SwarmError> {
        let planned: Vec<crate::types::NodeId> = self
            .assignment
            .segments
            .iter()
            .map(|s| s.node_id.clone())
            .collect();
        let rounds_before = self.shared_state.decode_rounds_of(self.request.id);
        let outcome = self.execute_attempt(token_tx).await;
        if let (Ok(out), Some(before)) = (&outcome, rounds_before) {
            self.note_split_rounds(out, before);
        }
        let ended_with = self.assignment.segments.iter().map(|s| s.node_id.clone());
        self.shared_state
            .note_request_peers(self.request.id, planned.into_iter().chain(ended_with));
        outcome
    }

    /// What this attempt's split yielded a round, for the price of the next
    /// one (`state.metrics.split_rounds`, FUTURE_WORK #3 (d)): its tokens over
    /// the rounds its decode sent through the first segment — counted where
    /// each round leaves, whichever path ran (`RequestTrace::
    /// note_decode_round`) — plus the prompt pass, which yields the first
    /// token. Only a split of two or more segments, for a request that may
    /// guess ahead (`request_may_guess_ahead`, the planner's own question),
    /// whose decode ran HERE: a split led by its head, a hand-off or a local
    /// generation sends no round from here and records nothing.
    fn note_split_rounds(&self, out: &InferenceOutput, rounds_before: u32) {
        let Some(after) = self.shared_state.decode_rounds_of(self.request.id) else {
            return;
        };
        let rounds = after.saturating_sub(rounds_before);
        if rounds == 0
            || self.assignment.segments.len() < 2
            || !request_may_guess_ahead(&self.shared_state, &self.request)
        {
            return;
        }
        self.shared_state.metrics.split_rounds.record(
            &self.request.model_id.0,
            out.completion_tokens,
            rounds.saturating_add(1),
        );
    }

    async fn execute_attempt(
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
            // OURS, so `Internal` and not `PipelineError`. The scheduler handed
            // back a plan with nothing in it; no peer, no setting and no model
            // shape can cause that. As a `PipelineError` it carried that
            // variant's hint — "the model is missing a piece, fetch it with
            // `swarmllm get-model <name> --all`" — which is advice for a
            // condition this is not, so following it could not help
            // (`docs/FUTURE_WORK.md` #86, gotcha #295). `Internal` has no hint
            // by design: there is nothing the reader can do about our bug.
            return Err(SwarmError::Internal("Pipeline has no segments".to_string()));
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
        // `router::batch::execute_batch`. And it is asked with the REQUEST:
        // a LoRA plan is exactly "one segment, on this node", and
        // `execute_local` applies no adapter (#110).
        if num_segments == 1
            && self.assignment.segments[0].node_id == *self.shared_state.identity.node_id()
            && self.shared_state.local_executor_serves(&self.request).await
        {
            return self.execute_local().await;
        }

        // Distributed execution path (also handles single-node split-model execution)
        self.execute_distributed(token_tx).await
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
                tool: Default::default(),
            }],
            sampling_params: SamplingParams::default(),
            stream: false,
            requester: state.identity.node_id().clone(),
            priority: PriorityTier::Silver,
            created_at: chrono::Utc::now(),
            session_id: None,
            lora_adapter: None,
            tools: None,
            cancel: None,
            route_override: None,
            continuation: None,
            keep: Default::default(),
            requested_by: Default::default(),
        }
    }

    /// `model_loaded` is global — "a model is loaded", not "this one". Both
    /// local fast paths (`pipeline::execute` and `router::batch::execute_batch`)
    /// generate from the singleton executor WITHOUT consulting
    /// `request.model_id`, so gating them on the bare flag answered a request
    /// for one model with a different model's weights and chat template, and
    /// reported it as a success. These pin the identity check that replaced it.
    /// A request for `model`, as the router would hand it over.
    fn request_for(state: &SharedState, model: &str) -> InferenceRequest {
        InferenceRequest {
            model_id: ModelId(model.into()),
            ..make_test_request(state)
        }
    }

    #[tokio::test]
    async fn local_executor_serves_requires_the_flag() {
        let state = make_test_state();
        // Nothing loaded: the flag is false and no info is cached.
        assert!(
            !state
                .local_executor_serves(&request_for(&state, "anything"))
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
                .local_executor_serves(&request_for(&state, "Qwen2.5 Coder 7B Instruct"))
                .await
        );
        assert!(
            state
                .local_executor_serves(&request_for(
                    &state,
                    &crate::types::slugify_model_name("Qwen2.5 Coder 7B Instruct")
                ))
                .await
        );

        // The RIGHT model, but naming a LoRA adapter: the singleton executor
        // applies none, so it must not answer — all three gates that hand it
        // work ask this one predicate (#110).
        let with_adapter = InferenceRequest {
            lora_adapter: Some("coder-v1".into()),
            ..request_for(&state, "Qwen2.5 Coder 7B Instruct")
        };
        assert!(!state.local_executor_serves(&with_adapter).await);

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
                !state
                    .local_executor_serves(&request_for(&state, other))
                    .await,
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
                .local_executor_serves(&request_for(&state, "anything"))
                .await
        );
    }

    /// A continuation is honoured in the ONE place every path's prompt comes
    /// from (FUTURE_WORK #236): the reply so far follows the rendered prompt,
    /// inside the assistant turn, character for character. And the singleton
    /// executor, which renders elsewhere, stands aside for it (the delegated
    /// split's half is in `a_split_none_of_which_is_ours_is_led_by_its_head_and_only_that`).
    #[tokio::test]
    async fn a_continuation_follows_the_rendered_prompt_and_nothing_else_renders_it() {
        let state = make_test_state();
        let (tx, _rx) = mpsc::channel::<NetworkCommand>(64);
        let assignment = |id| PipelineAssignment {
            request_id: id,
            segments: vec![],
            standbys: vec![],
            tp_groups: vec![],
            supports_speculative: false,
        };
        let plain = make_test_request(&state);
        let rendered = PipelineExecutor::new(
            state.clone(),
            tx.clone(),
            plain.clone(),
            assignment(plain.id),
        )
        .build_prompt_with_header(None)
        .await;

        let mut continued = plain.clone();
        continued.continuation = Some(crate::types::ReplyContinuation {
            text: "The capital of Fr".into(),
            tokens: 5,
        });
        let exec = PipelineExecutor::new(
            state.clone(),
            tx.clone(),
            continued.clone(),
            assignment(plain.id),
        );
        assert_eq!(
            exec.build_prompt_with_header(None).await,
            format!("{rendered}The capital of Fr")
        );

        // A reply held to JSON is held from where that text leaves off — the
        // prompt ends there, and a grammar starting fresh would open a second
        // document (pre-release review of v0.3.262).
        let grammar = crate::inference::reply_grammar::settle_json_object();
        let mut held = continued.clone();
        held.sampling_params.reply_grammar = Some(grammar.clone());
        let sampling_for = |request: crate::types::InferenceRequest| {
            let exec =
                PipelineExecutor::new(state.clone(), tx.clone(), request, assignment(plain.id));
            async move { exec.build_prompt_and_stops().await.1 }
        };
        assert_eq!(
            sampling_for(held.clone())
                .await
                .reply_grammar
                .as_deref()
                .map(|g| g.written.as_str()),
            Some("The capital of Fr")
        );
        held.continuation = None;
        assert_eq!(
            sampling_for(held).await.reply_grammar.as_deref(),
            Some(&*grammar),
            "a reply that starts fresh is held from its start"
        );

        // The singleton executor renders its own prompt too: never handed one.
        *state.loaded_model_info.write().await = Some(crate::daemon::state::LoadedModelInfo {
            name: "test".into(),
            size_bytes: 0,
            eos_tokens: vec![],
            chat_template: None,
            bos_token: String::new(),
            eos_token: String::new(),
        });
        state
            .model_loaded
            .store(true, std::sync::atomic::Ordering::Release);
        assert!(
            state.local_executor_serves(&plain).await,
            "fixture: it serves the plain request"
        );
        assert!(!state.local_executor_serves(&continued).await);
    }

    /// Every attempt records the peers it ran segments on — as planned and as
    /// it ended — for the request's end, where they are told it is over (#238).
    #[tokio::test]
    async fn an_attempt_records_the_peers_it_was_planned_on() {
        let state = make_test_state();
        let (tx, _rx) = mpsc::channel::<NetworkCommand>(64);
        let request = make_test_request(&state);
        let peer = crate::types::NodeId([0x7E; 32]);
        let assignment = PipelineAssignment {
            request_id: request.id,
            segments: vec![PipelineSegment {
                node_id: peer.clone(),
                shard_id: ShardId {
                    model_id: request.model_id.clone(),
                    index: 0,
                },
                layer_range: (0, 1),
            }],
            standbys: vec![],
            tp_groups: vec![],
            supports_speculative: false,
        };
        let rid = request.id;
        let mut executor = PipelineExecutor::new(state.clone(), tx, request, assignment);
        // It fails — the peer is unknown here — and records the peer anyway.
        let _ = executor.execute(None).await;
        assert_eq!(state.take_request_peers(rid), vec![peer]);
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

    /// #337: who a split's checks reach, for the owner's earlier replies. A
    /// remote first segment is sent the guesses' ids and knows the reply's;
    /// the tail is sent the guesses' ids to walk and draws every token; a
    /// middle peer sees hidden states — counted as sent the guesses' ids where
    /// a check may chain (it hands them on), never as knowing the reply's.
    #[test]
    fn a_checks_guesses_and_its_reply_are_counted_for_every_peer_they_reach() {
        use crate::inference::owner_replies::Exposure;
        let state = make_test_state();
        let me = state.identity.node_id().clone();
        let (a, b) = (
            crate::types::NodeId([0xA1; 32]),
            crate::types::NodeId([0xB2; 32]),
        );
        let executor = |nodes: &[&crate::types::NodeId]| {
            let request = make_test_request(&state);
            let assignment = PipelineAssignment {
                request_id: request.id,
                segments: nodes
                    .iter()
                    .enumerate()
                    .map(|(i, n)| PipelineSegment {
                        node_id: (*n).clone(),
                        shard_id: ShardId {
                            model_id: request.model_id.clone(),
                            index: i as u32,
                        },
                        layer_range: (i as u32, i as u32 + 1),
                    })
                    .collect(),
                standbys: vec![],
                tp_groups: vec![],
                supports_speculative: true,
            };
            let (tx, _rx) = mpsc::channel::<NetworkCommand>(8);
            PipelineExecutor::new(state.clone(), tx, request, assignment)
        };
        let both = || [a.clone(), b.clone()];
        // Peers that walk a check at the tail, as every current build does.
        let walks = swarmllm_types::node::features::SPEC_WALK_AT_TAIL
            | swarmllm_types::node::features::FORWARD_SAMPLING
            | swarmllm_types::node::features::FORWARD_GENERATED_IDS;
        for n in both() {
            state
                .peer_registry
                .insert(n.clone(), peer_advertising(&n, walks));
        }

        // This node first, one peer the rest: its tail.
        let e = executor(&[&me, &a]).exposure_of_a_check();
        assert_eq!(e.shown, Exposure::new([a.clone()], [a.clone()]));
        assert_eq!(e.seen, Exposure::new([a.clone()], [a.clone()]));
        // Both remote: the first is sent ids and knows the reply's, as the tail does.
        let e = executor(&[&a, &b]).exposure_of_a_check();
        assert_eq!(e.seen, Exposure::new(both(), both()));
        // This node last: the peer before it is the first segment.
        let e = executor(&[&a, &me]).exposure_of_a_check();
        assert_eq!(e.seen, Exposure::new([a.clone()], [a.clone()]));

        // A middle peer knows the reply as hidden states ...
        let mut middle = executor(&[&me, &a, &b]);
        let e = middle.exposure_of_a_check();
        assert_eq!(e.seen, Exposure::new(both(), [b.clone()]));
        // ... and is sent the guesses' ids only where the check may chain.
        assert_eq!(e.shown, Exposure::new(both(), both()), "a chained check");
        middle.chaining_disabled = true;
        assert_eq!(
            middle.exposure_of_a_check().shown,
            Exposure::new(both(), [b.clone()])
        );
        // A chain never runs through this node: with both ends here, the
        // peer between them is sent hidden states only, chained or not.
        let e = executor(&[&me, &a, &me]).exposure_of_a_check();
        assert_eq!(e.shown, Exposure::new([a.clone()], []), "a boomerang");
        assert_eq!(e.seen, Exposure::new([a.clone()], []));

        // An older tail that cannot walk sends its distributions here: sent
        // the guesses, it is never counted as knowing the reply's ids.
        let old = crate::types::NodeId([0xC3; 32]);
        let e = executor(&[&me, &old]).exposure_of_a_check();
        assert_eq!(e.shown, Exposure::new([old.clone()], [old.clone()]));
        assert_eq!(e.seen, Exposure::new([old.clone()], []), "an older tail");

        // Under prompt privacy this node embeds what the first peer reads.
        let mut cfg = crate::config::Config::clone(&state.cfg());
        cfg.inference.local_embedding_privacy = true;
        state.apply_live_config(cfg);
        let e = executor(&[&a, &b]).exposure_of_a_check();
        assert_eq!(e.seen, Exposure::new(both(), [b.clone()]));
        assert_eq!(e.shown, Exposure::new(both(), both()));
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
                // These stand in for what a PEER sent, so they must look like
                // it: `LayerResult::error` marks a result this process
                // manufactured, and a harness that left that set would be
                // simulating our own ACK sweep rather than the peer replying.
                let base = LayerResult {
                    locally_constructed: false,
                    ..LayerResult::error(forward.request_id, "")
                };
                let result = match reply(&node) {
                    PeerReply::Error(m) => LayerResult {
                        locally_constructed: false,
                        ..LayerResult::error(forward.request_id, m)
                    },
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

    /// A check of a model with recurrent state is taken back into only where
    /// EVERY peer segment can replay it (`features::RECURRENT_REWIND`): one
    /// older peer refuses the truncate, so the plan reads one token at a time.
    /// This node's own segments are its own build and never asked.
    #[test]
    fn a_check_is_taken_back_into_only_where_every_peer_segment_can() {
        let (here, a, b) = (NodeId([1; 32]), NodeId([2; 32]), NodeId([3; 32]));
        let segments = vec![
            remote_segment(&here, (0, 8)),
            remote_segment(&a, (8, 16)),
            remote_segment(&b, (16, 24)),
        ];
        assert!(every_peer_segment(&segments, &here, |_| true));
        assert!(!every_peer_segment(&segments, &here, |n| n == &a));
        assert!(!every_peer_segment(&segments, &here, |n| n == &b));
        // Only this node's segments: nothing to ask.
        assert!(every_peer_segment(&segments[..1], &here, |_| false));
        assert!(!every_peer_segment(&segments, &here, |n| n == &here));
    }

    /// Trees are wanted only where EVERY machine of the plan runs on a card:
    /// one segment on a processor makes each row of a check cost what that
    /// processor takes to read it.
    #[test]
    fn a_plan_is_on_cards_only_where_every_segment_is() {
        let here = NodeId([1; 32]);
        let card_peer = NodeId([2; 32]);
        let processor_peer = NodeId([3; 32]);
        let peer_on_card = |n: &NodeId| *n == card_peer;
        let plan = |nodes: &[&NodeId]| -> Vec<PipelineSegment> {
            nodes
                .iter()
                .enumerate()
                .map(|(i, n)| remote_segment(n, (i as u32 * 14, (i as u32 + 1) * 14)))
                .collect()
        };

        assert!(every_segment_on_a_card(
            &plan(&[&here, &card_peer]),
            &here,
            || true,
            peer_on_card
        ));
        assert!(
            !every_segment_on_a_card(&plan(&[&here, &card_peer]), &here, || false, peer_on_card),
            "this node's part on its processor"
        );
        assert!(
            !every_segment_on_a_card(
                &plan(&[&here, &processor_peer]),
                &here,
                || true,
                peer_on_card
            ),
            "a peer's part on its processor"
        );
        assert!(
            every_segment_on_a_card(&plan(&[&card_peer]), &here, || false, peer_on_card),
            "a led plan this node is not in is judged by its peers alone"
        );
        assert!(
            !every_segment_on_a_card(&[], &here, || true, peer_on_card),
            "no plan, no trees"
        );
    }

    /// The link a plan's checks cross is its FARTHEST peer's: a check reaches
    /// every segment, so the near one's 20 ms does not make a 420 ms plan
    /// near. This node's own segment is no exchange at all, and one unmeasured
    /// peer leaves the whole plan unknown — never read as near or far.
    #[test]
    fn a_plan_is_as_far_as_its_farthest_peer() {
        let here = NodeId([1; 32]);
        let near = NodeId([2; 32]);
        let far = NodeId([3; 32]);
        let unmeasured = NodeId([4; 32]);
        let rtt = |n: &NodeId| match n.0[0] {
            2 => Some(20),
            3 => Some(420),
            _ => None,
        };
        let plan = |nodes: &[&NodeId]| -> Vec<PipelineSegment> {
            nodes
                .iter()
                .enumerate()
                .map(|(i, n)| remote_segment(n, (i as u32 * 7, (i as u32 + 1) * 7)))
                .collect()
        };
        assert_eq!(
            farthest_round_trip(&plan(&[&here, &near, &far]), &here, rtt),
            Some(420)
        );
        assert_eq!(
            farthest_round_trip(&plan(&[&far, &near]), &here, rtt),
            Some(420),
            "the order of the hops does not matter"
        );
        assert_eq!(
            farthest_round_trip(&plan(&[&here, &near]), &here, rtt),
            Some(20)
        );
        assert_eq!(
            farthest_round_trip(&plan(&[&here, &far, &unmeasured]), &here, rtt),
            None,
            "one unmeasured peer"
        );
        assert_eq!(
            farthest_round_trip(&plan(&[&here]), &here, rtt),
            None,
            "no peer at all"
        );
    }

    /// The rig's first request on a 20 ms link (2026-10-21): the smoothed
    /// forwarding cost read 2,500 ms while the far machine loaded the model,
    /// and a gate reading it took the short link for a long one. The distance
    /// is the least round trip measured, whatever the cost reads.
    #[test]
    fn a_plan_is_as_far_as_the_least_round_trip_measured_not_its_smoothed_cost() {
        let state = make_test_state();
        let here = state.identity.node_id().clone();
        let peer = NodeId([7; 32]);
        let mut info = peer_advertising(&peer, 0);
        info.ack_srtt_ms = Some(2_500);
        info.latency_ms = Some(2_400);
        state.peer_registry.insert(peer.clone(), info);
        let segments = vec![
            remote_segment(&here, (0, 14)),
            remote_segment(&peer, (14, 28)),
        ];
        assert_eq!(
            farthest_segment_round_trip_ms(&state, &segments),
            None,
            "a peer with no round trip measured is unknown, not 2.5 s away"
        );
        {
            let mut samples = state
                .metrics
                .network_coord_samples
                .entry(peer.clone())
                .or_default();
            for (at, ms) in [(1_000, 2_600.0), (2_000, 20.6), (3_000, 31.0)] {
                samples.observe(at, ms);
            }
        }
        assert_eq!(farthest_segment_round_trip_ms(&state, &segments), Some(21));
    }

    /// Play every peer as a SERVING NODE answers a piece of a prompt pass:
    /// each answer names the step it answers (`answers_step`, the stream
    /// number included), which is how a streamed wait is found. Records every
    /// forward sent, in order.
    fn spawn_piece_peers(
        state: Arc<SharedState>,
        mut rx: mpsc::Receiver<NetworkCommand>,
        peers: std::collections::HashMap<Vec<u8>, NodeId>,
        reply: impl Fn(&NodeId, &LayerForward) -> PeerReply + Send + 'static,
    ) -> tokio::task::JoinHandle<Vec<(NodeId, LayerForward)>> {
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
                let answers = Some(forward.step());
                let base = LayerResult {
                    locally_constructed: false,
                    answers_step: answers,
                    ..LayerResult::error(forward.request_id, "")
                };
                let result = match reply(&node, &forward) {
                    PeerReply::Error(m) => LayerResult {
                        locally_constructed: false,
                        answers_step: answers,
                        ..LayerResult::error(forward.request_id, m)
                    },
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
                state.resolve_pending_layer_result(Some(&node), result);
                sent.push((node, forward));
            }
            sent
        })
    }

    /// FUTURE_WORK #171: a prompt pass read in pieces sends every segment its
    /// pieces in order, each naming the whole pass's span, each peer's as a
    /// stream of its own, hands the head's output for each piece to the tail,
    /// and answers with the tail's answer to the FINAL piece.
    #[tokio::test]
    async fn a_prompt_read_in_pieces_runs_every_piece_through_every_segment_in_order() {
        use crate::types::inference::stream_seq;
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
            standbys: vec![],
            tp_groups: vec![],
            supports_speculative: false,
        };
        let executor = PipelineExecutor::new(state.clone(), tx, request, assignment);
        let head = a.clone();
        let peers_task = spawn_piece_peers(state.clone(), rx, peers, move |node, f| {
            if *node == head {
                // The head's output names the piece it came from.
                PeerReply::Activations(f.index_pos.to_le_bytes().to_vec())
            } else {
                PeerReply::Token(f.index_pos)
            }
        });
        let ids: Vec<u32> = (0..1100).collect();
        let result = executor
            .prompt_pass_in_pieces(request_id, &ids, 0, &[])
            .await
            .expect("every piece answered");
        drop(executor);
        let sent = peers_task.await.unwrap();

        assert_eq!(
            result.token_ids,
            vec![1024],
            "the tail's answer to the FINAL piece"
        );
        let span = crate::types::PromptSpan {
            start: 0,
            end: 1100,
        };
        assert!(sent
            .iter()
            .all(|(_, f)| f.prompt_span == Some(span) && f.sequence_num == 0));
        for node in [&a, &d] {
            let mine: Vec<&LayerForward> = sent
                .iter()
                .filter(|(n, _)| n == node)
                .map(|(_, f)| f)
                .collect();
            assert_eq!(
                mine.iter().map(|f| f.index_pos).collect::<Vec<_>>(),
                vec![0, 512, 1024],
                "each segment runs every piece, in order"
            );
            let seqs: Vec<u32> = mine.iter().map(|f| f.stream_seq.unwrap()).collect();
            assert_eq!(
                seqs.iter()
                    .map(|&s| stream_seq::turn(s))
                    .collect::<Vec<_>>(),
                vec![0, 1, 2]
            );
            assert!(seqs
                .iter()
                .all(|&s| stream_seq::attempt(s) == stream_seq::attempt(seqs[0])));
        }
        let attempt_of = |node: &NodeId| {
            sent.iter()
                .find(|(n, _)| n == node)
                .map(|(_, f)| stream_seq::attempt(f.stream_seq.unwrap()))
        };
        assert_ne!(
            attempt_of(&a),
            attempt_of(&d),
            "each segment's pieces are a stream of their own"
        );
        // The tail was handed the head's output for the same piece.
        for (_, f) in sent.iter().filter(|(n, _)| *n == d) {
            assert_eq!(f.activations, f.index_pos.to_le_bytes().to_vec());
        }
        assert_eq!(
            state.pending_layer_results.len(),
            0,
            "no wait is left behind"
        );
    }

    /// The review of #171 (2026-10-10): a piece whose wait TIMES OUT may still be
    /// running at its peer, or the next one queued there, and the whole pass
    /// that a fallback sends goes straight to that peer's worker — which routes
    /// replies by request id (gotcha #180). So a silent peer ends the pass
    /// with `every_piece_answered: false`, and the caller hands the error to
    /// the router instead. Also: a peer is sent ONE piece until it answers one
    /// (the first may load the model), so the silent tail never had two.
    #[tokio::test(start_paused = true)]
    async fn a_piece_nobody_answered_forbids_reading_the_pass_whole_beside_it() {
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
            standbys: vec![],
            tp_groups: vec![],
            supports_speculative: false,
        };
        let executor = PipelineExecutor::new(state.clone(), tx, request, assignment);
        let (head, tail) = (a.clone(), d.clone());
        // The tail never answers anything: a peer that took the work and went quiet.
        let peers_task = tokio::spawn(async move {
            let mut rx = rx;
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
                if node == head {
                    let answer = LayerResult {
                        locally_constructed: false,
                        finish_reason: None,
                        activations: vec![1, 2, 3, 4],
                        answers_step: Some(forward.step()),
                        ..LayerResult::error(forward.request_id, "")
                    };
                    state.resolve_pending_layer_result(Some(&node), answer);
                }
                sent.push((node, forward.index_pos));
            }
            sent
        });
        let ids: Vec<u32> = (0..1100).collect();
        let failed = executor
            .prompt_pass_in_pieces(request_id, &ids, 0, &[])
            .await
            .expect_err("the tail never answered");
        drop(executor);
        let sent = peers_task.await.unwrap();
        assert!(
            !failed.every_piece_answered,
            "a piece may still be live at the tail: the pass must not be read whole beside it"
        );
        assert_eq!(
            sent.iter().filter(|(n, _)| *n == tail).count(),
            1,
            "one piece to a peer until it answers one"
        );
    }

    /// A piece that fails ends the pass with ITS error — never the knock-on of
    /// the segments that stopped because of it — and only after every piece
    /// already out has been answered, so none reaches a worker beside the
    /// whole pass the caller runs next.
    #[tokio::test]
    async fn a_piece_that_fails_ends_the_pass_with_its_own_error_and_nothing_left_out() {
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
            standbys: vec![],
            tp_groups: vec![],
            supports_speculative: false,
        };
        let executor = PipelineExecutor::new(state.clone(), tx, request, assignment);
        let head = a.clone();
        let peers_task = spawn_piece_peers(state.clone(), rx, peers, move |node, f| {
            if *node == head {
                PeerReply::Activations(vec![1, 2, 3, 4])
            } else if f.index_pos == 512 {
                PeerReply::Error("Worker: Service unavailable: out of memory")
            } else {
                PeerReply::Token(1)
            }
        });
        // Six pieces: 0, 512, ..., 2560.
        let ids: Vec<u32> = (0..2600).collect();
        let err = executor
            .prompt_pass_in_pieces(request_id, &ids, 0, &[])
            .await
            .expect_err("the tail failed a piece");
        drop(executor);
        let sent = peers_task.await.unwrap();
        assert!(
            err.error.to_string().contains("out of memory"),
            "the cause, not a knock-on: {err:?}"
        );
        assert!(
            err.every_piece_answered,
            "a refusal is an answer: nothing is left live, so the pass may be read whole"
        );
        // The piece after the failed one may already be out (two to a peer
        // that has answered one); nothing past that window is sent.
        assert!(
            !sent.iter().any(|(n, f)| *n == d && f.index_pos >= 1536),
            "nothing past the window is sent to the failed segment"
        );
        assert_eq!(
            state.pending_layer_results.len(),
            0,
            "every piece out was waited for"
        );
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

    /// The first live run of a composite takeover (`split_rig.sh failover`,
    /// 2026-09-25) found this: the LAST segment fails on the prompt pass, no
    /// single standby holds its layers, and two cover them between them. The
    /// first part answered, and its hidden states came back as the pipeline's
    /// ANSWER — the caller's `is_last` was computed before the splice — so the
    /// second part never saw the prompt and the next token failed on it.
    #[tokio::test]
    async fn a_failed_last_segment_taken_over_by_several_nodes_runs_every_part_before_answering() {
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
            segments: vec![remote_segment(&a, (0, 16)), remote_segment(&b, (16, 32))],
            // Neither holds all of B's range; together they cover it.
            standbys: vec![remote_segment(&c, (16, 24)), remote_segment(&d, (24, 32))],
            tp_groups: vec![],
            supports_speculative: false,
        };
        let mut executor = PipelineExecutor::new(state.clone(), tx, request, assignment);
        let c_out = vec![0xCC; 64];
        let (a2, b2, c2, c_out2) = (a.clone(), b.clone(), c.clone(), c_out.clone());
        let peers_task = spawn_peers(state.clone(), rx, peers, move |node| {
            if *node == a2 {
                PeerReply::Activations(vec![0xAA; 64])
            } else if *node == b2 {
                PeerReply::Error(
                    "Worker: Service unavailable: worker closed connection before reply",
                )
            } else if *node == c2 {
                PeerReply::Activations(c_out2.clone())
            } else {
                PeerReply::Token(7)
            }
        });

        let result = executor
            .forward_through_segments(request_id, 0, 0, b"hello".to_vec(), None, false, &[])
            .await;
        drop(executor);
        let sent = peers_task.await.unwrap();

        let result = result.expect("the cover covers the whole range, so the prompt pass succeeds");
        assert_eq!(
            result.token_ids,
            vec![7],
            "the answer must come from the part that SAMPLES — the last one — not the first \
             part's hidden states"
        );
        assert!(
            sent.iter().any(|(n, act)| *n == d && *act == c_out),
            "the second part must be sent the first part's output on the prompt pass: {:?}",
            sent.iter().map(|(n, _)| n.to_string()).collect::<Vec<_>>()
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

    /// The shared emit helper streams a character split across tokens — and
    /// across TWO emitted batches — whole, and records the same text for a
    /// salvage. All three speculative coordinators stream through it, and it
    /// used to decode each token on its own: "🎉" arrived as four U+FFFD.
    #[tokio::test]
    async fn a_character_split_across_two_batches_streams_whole() {
        let decoder = super::prompt::decoder_tests::byte_fallback_decoder();
        let partial = PartialReply::default();
        let (tx, mut rx) = crate::inference::router::StreamingTokenTx::channel(16);
        let token_tx = Some(tx);
        let eos = std::collections::HashSet::new();
        let mut finish = String::new();
        // " Party", then the emoji's first two bytes | its last two.
        for batch in [&[1u32, 2, 3][..], &[4, 5][..]] {
            assert!(
                !emit_streaming_batch(&partial, &token_tx, &decoder, batch, &eos, &mut finish)
                    .await
            );
        }
        drop(token_tx);
        let mut streamed = String::new();
        while let Some(ev) = rx.recv().await {
            streamed.push_str(&ev.text);
        }
        assert_eq!(streamed, " Party\u{1F389}", "streamed");
        let (recorded, ids, _) = partial.taken().expect("recorded");
        assert_eq!(recorded, " Party\u{1F389}", "recorded for a salvage");
        assert_eq!(ids, vec![1, 2, 3, 4, 5]);

        // Cut two bytes into the next emoji: the fragment is not lost from
        // the salvage — it is the U+FFFD a whole-reply decode would show.
        let (tx, _rx) = crate::inference::router::StreamingTokenTx::channel(16);
        let token_tx = Some(tx);
        assert!(
            !emit_streaming_batch(&partial, &token_tx, &decoder, &[1, 2, 3], &eos, &mut finish)
                .await
        );
        let (recorded, ids, _) = partial.taken().expect("recorded");
        assert_eq!(
            recorded, " Party\u{FFFD}",
            "a cut character is rendered, not dropped"
        );
        assert_eq!(ids, vec![1, 2, 3]);
    }

    /// A registry entry for `node` advertising exactly `features` — what the
    /// sender-side trailer gates read.
    pub(super) fn peer_advertising(node: &NodeId, features: u64) -> PeerInfo {
        PeerInfo {
            node_id: node.clone(),
            addresses: vec![],
            capability: Some(NodeCapability {
                coord: None,
                node_id: node.clone(),
                gpu: None,
                cpu: None,
                ram_total_mb: 0,
                ram_available_mb: 0,
                ram_model_budget_mb: None,
                disk_available_mb: 0,
                bandwidth_mbps: 0.0,
                hosted_shards: vec![],
                max_contribution: ContributionLevel::Moderate,
                uptime_seconds: 0,
                version: String::new(),
                region: None,
                est_tokens_per_sec_7b: 0.0,
                os: None,
                observed_latencies: vec![],
                relay_capable: false,
                protocol_version: 0,
                features,
                relay_reservations: vec![],
                anchor_mode: false,
                can_serve_inference: true,
                resident_layers: Vec::new(),
                context_ceiling_tokens: None,
                model_load_ms_per_gib: None,
                model_memory_ceiling_mb: None,
                machine_room_beside_card_mb: None,
                processor_tokens_per_sec_7b: None,
            }),
            last_seen: chrono::Utc::now(),
            latency_ms: Some(10),
            trust_score: 0.9,
            peer_id_bytes: None,
            ack_srtt_ms: None,
            active_request_count: 0,
            first_seen: 0,
            verified_transaction_count: 0,
            is_lan_peer: false,
            goodput_bytes_per_sec: None,
            goodput_samples: 0,
        }
    }

    /// The reply so far travels to the peer that samples only when its sampler
    /// will READ it — i.e. when a repetition penalty is set. With the default
    /// (none) it was shipped on every decode step anyway: the caller always
    /// passes the whole completion, and the send gated only on "is it
    /// non-empty", which is true for every step after the first. The positive
    /// control — penalties on, history arrives — is what proves the harness
    /// can see the field at all.
    #[tokio::test]
    async fn a_forward_carries_the_reply_so_far_only_when_penalties_will_read_it() {
        async fn history_sent_to_the_sampling_peer(frequency_penalty: f32) -> Vec<u32> {
            let state = make_test_state();
            let (tx, mut rx) = mpsc::channel::<NetworkCommand>(64);
            let mut request = make_test_request(&state);
            request.sampling_params.frequency_penalty = frequency_penalty;
            let request_id = request.id;
            let (a, d) = (NodeId([0xA1; 32]), NodeId([0xD4; 32]));
            let mut peers = std::collections::HashMap::new();
            for (node, byte) in [(&a, 0xA1u8), (&d, 0xD4)] {
                state.peer_id_map.insert(node.clone(), vec![byte]);
                peers.insert(vec![byte], node.clone());
                // A peer that CAN read the history (the `0x08` trailer), so
                // the feature gate is not what keeps it off the wire.
                state.peer_registry.insert(
                    node.clone(),
                    peer_advertising(node, swarmllm_types::node::features::FORWARD_GENERATED_IDS),
                );
            }
            let assignment = PipelineAssignment {
                request_id,
                segments: vec![remote_segment(&a, (0, 16)), remote_segment(&d, (16, 32))],
                standbys: vec![],
                tp_groups: vec![],
                supports_speculative: false,
            };
            let mut executor = PipelineExecutor::new(state.clone(), tx, request, assignment);
            let sampling_peer = d.clone();
            let harness_state = state.clone();
            let harness = tokio::spawn(async move {
                let mut to_sampler = None;
                while let Some(cmd) = rx.recv().await {
                    let NetworkCommand::SendTensor {
                        target_peer_bytes,
                        forward,
                    } = cmd
                    else {
                        continue;
                    };
                    let node = peers[&target_peer_bytes].clone();
                    let base = LayerResult {
                        locally_constructed: false,
                        ..LayerResult::error(forward.request_id, "")
                    };
                    let result = if node == sampling_peer {
                        to_sampler = Some(forward.generated_ids.clone());
                        LayerResult {
                            token_ids: vec![7],
                            finish_reason: None,
                            ..base
                        }
                    } else {
                        LayerResult {
                            activations: forward.activations.clone(),
                            finish_reason: None,
                            ..base
                        }
                    };
                    assert!(harness_state.resolve_pending_layer_result(Some(&node), result));
                }
                to_sampler.expect("the sampling peer was sent a forward")
            });
            // A decode step four tokens into the reply.
            executor
                .forward_through_segments(
                    request_id,
                    4,
                    9,
                    vec![0x11; 64],
                    None,
                    false,
                    &[5, 6, 7, 8],
                )
                .await
                .expect("both segments answered");
            drop(executor);
            harness.await.unwrap()
        }

        assert_eq!(
            history_sent_to_the_sampling_peer(0.5).await,
            vec![5, 6, 7, 8],
            "control: with a penalty set, the sampling peer must still get the reply so far"
        );
        assert!(
            history_sent_to_the_sampling_peer(0.0).await.is_empty(),
            "with no penalty the sampler never reads the history, so it must not be sent"
        );
    }

    /// A standby that takes over the sampling segment is a DIFFERENT peer from
    /// the one planned, so the `0x08` trailer is gated on the standby's own
    /// features — the ordinary send checks the planned peer's. The failover
    /// forward carried the history blind: an older standby rebuilds the seal's
    /// AAD from the trailers it parses, so every encrypted forward to it failed
    /// to open, and the failover failed too.
    ///
    /// Staged on the prompt pass, where a stand-in takes the segment over
    /// without a replay (mid-reply it needs the retained history, which this
    /// harness does not encode); the history is passed as a decode step passes
    /// it, and the gate is the same expression on either.
    #[tokio::test]
    async fn a_standby_is_sent_the_reply_so_far_only_when_it_can_read_it() {
        async fn history_sent_to_the_standby(standby_features: u64) -> Vec<u32> {
            use swarmllm_types::node::features::FORWARD_GENERATED_IDS;
            let state = make_test_state();
            let (tx, mut rx) = mpsc::channel::<NetworkCommand>(64);
            let mut request = make_test_request(&state);
            request.sampling_params.frequency_penalty = 0.5;
            let request_id = request.id;
            let (a, c, d) = (NodeId([0xA1; 32]), NodeId([0xC3; 32]), NodeId([0xD4; 32]));
            let mut peers = std::collections::HashMap::new();
            for (node, byte, features) in [
                (&a, 0xA1u8, FORWARD_GENERATED_IDS),
                (&d, 0xD4, FORWARD_GENERATED_IDS),
                (&c, 0xC3, standby_features),
            ] {
                state.peer_id_map.insert(node.clone(), vec![byte]);
                peers.insert(vec![byte], node.clone());
                state
                    .peer_registry
                    .insert(node.clone(), peer_advertising(node, features));
            }
            let assignment = PipelineAssignment {
                request_id,
                segments: vec![remote_segment(&a, (0, 16)), remote_segment(&d, (16, 32))],
                standbys: vec![remote_segment(&c, (16, 32))],
                tp_groups: vec![],
                supports_speculative: false,
            };
            let mut executor = PipelineExecutor::new(state.clone(), tx, request, assignment);
            let (planned_sampler, standby) = (d.clone(), c.clone());
            let harness_state = state.clone();
            let harness = tokio::spawn(async move {
                let mut to_standby = None;
                while let Some(cmd) = rx.recv().await {
                    let NetworkCommand::SendTensor {
                        target_peer_bytes,
                        forward,
                    } = cmd
                    else {
                        continue;
                    };
                    let node = peers[&target_peer_bytes].clone();
                    let base = LayerResult {
                        locally_constructed: false,
                        ..LayerResult::error(forward.request_id, "")
                    };
                    let result = if node == planned_sampler {
                        LayerResult {
                            locally_constructed: false,
                            ..LayerResult::error(
                                forward.request_id,
                                "Worker: Service unavailable: out of memory",
                            )
                        }
                    } else if node == standby {
                        to_standby = Some(forward.generated_ids.clone());
                        LayerResult {
                            token_ids: vec![7],
                            finish_reason: None,
                            ..base
                        }
                    } else {
                        LayerResult {
                            activations: forward.activations.clone(),
                            finish_reason: None,
                            ..base
                        }
                    };
                    assert!(harness_state.resolve_pending_layer_result(Some(&node), result));
                }
                to_standby.expect("the standby was sent a forward")
            });
            executor
                .forward_through_segments(
                    request_id,
                    0,
                    0,
                    vec![0x11; 64],
                    None,
                    false,
                    &[5, 6, 7, 8],
                )
                .await
                .expect("the standby answered");
            drop(executor);
            harness.await.unwrap()
        }

        assert_eq!(
            history_sent_to_the_standby(swarmllm_types::node::features::FORWARD_GENERATED_IDS)
                .await,
            vec![5, 6, 7, 8],
            "control: a standby that reads the trailer is sent the history"
        );
        assert!(
            history_sent_to_the_standby(0).await.is_empty(),
            "a standby that cannot parse the `0x08` trailer must not be sent one"
        );
    }

    /// Every hop of a chain builds the NEXT hop's forward, so a hop that cannot
    /// read and hand down the `0x0A` trailer leaves the tail sampling at its
    /// defaults — #106 one hop removed. Such a peer is never put in a chain.
    #[test]
    fn a_hop_that_cannot_hand_down_the_callers_sampling_is_never_chained() {
        use swarmllm_types::node::features::{FORWARD_SAMPLING, PIPELINE_CHAIN_V2};
        let state = make_test_state();
        let (old, new) = (NodeId([0x01; 32]), NodeId([0x02; 32]));
        state
            .peer_registry
            .insert(old.clone(), peer_advertising(&old, PIPELINE_CHAIN_V2));
        state.peer_registry.insert(
            new.clone(),
            peer_advertising(&new, PIPELINE_CHAIN_V2 | FORWARD_SAMPLING),
        );
        assert!(!state.peer_supports_pipeline_chain(&old));
        assert!(state.peer_supports_pipeline_chain(&new));
    }

    /// FUTURE_WORK #106: a REMOTE sampling segment used the worker's defaults
    /// (0.7 / 0.9 / 40) whatever the caller asked, because `sampling` never
    /// reached the wire. Now the caller's parameters go to the segment that
    /// samples — only to one that reads the `0x0A` trailer, and not to a
    /// segment that does not sample. Both sends are gated: the planned peer's
    /// and, on failover, the stand-in's own features (gotcha #703).
    #[tokio::test]
    async fn the_callers_sampling_reaches_the_segment_that_samples_when_it_can_read_it() {
        use swarmllm_types::node::features::FORWARD_SAMPLING;
        use swarmllm_types::SamplingParams;
        // (planned sampler fails?, sampler/stand-in features) ->
        //     (sent to segment 0, sent to whoever sampled)
        async fn sent(
            fail_planned: bool,
            sampler_features: u64,
        ) -> (Option<SamplingParams>, Option<SamplingParams>) {
            let state = make_test_state();
            let (tx, mut rx) = mpsc::channel::<NetworkCommand>(64);
            let mut request = make_test_request(&state);
            request.sampling_params.temperature = 0.0;
            request.sampling_params.top_k = 1;
            request.sampling_params.presence_penalty = 0.5;
            let request_id = request.id;
            let (a, c, d) = (NodeId([0xA1; 32]), NodeId([0xC3; 32]), NodeId([0xD4; 32]));
            let mut peers = std::collections::HashMap::new();
            let planned_features = if fail_planned {
                FORWARD_SAMPLING
            } else {
                sampler_features
            };
            for (node, byte, features) in [
                (&a, 0xA1u8, FORWARD_SAMPLING),
                (&d, 0xD4, planned_features),
                (&c, 0xC3, sampler_features),
            ] {
                state.peer_id_map.insert(node.clone(), vec![byte]);
                peers.insert(vec![byte], node.clone());
                state
                    .peer_registry
                    .insert(node.clone(), peer_advertising(node, features));
            }
            let assignment = PipelineAssignment {
                request_id,
                segments: vec![remote_segment(&a, (0, 16)), remote_segment(&d, (16, 32))],
                standbys: vec![remote_segment(&c, (16, 32))],
                tp_groups: vec![],
                supports_speculative: false,
            };
            let mut executor = PipelineExecutor::new(state.clone(), tx, request, assignment);
            let (first, planned, standby) = (a.clone(), d.clone(), c.clone());
            let harness_state = state.clone();
            let harness = tokio::spawn(async move {
                let (mut to_first, mut to_sampler) = (None, None);
                while let Some(cmd) = rx.recv().await {
                    let NetworkCommand::SendTensor {
                        target_peer_bytes,
                        forward,
                    } = cmd
                    else {
                        continue;
                    };
                    let node = peers[&target_peer_bytes].clone();
                    let base = LayerResult {
                        locally_constructed: false,
                        ..LayerResult::error(forward.request_id, "")
                    };
                    let result = if node == first {
                        to_first = Some(forward.sampling.clone());
                        LayerResult {
                            activations: forward.activations.clone(),
                            finish_reason: None,
                            ..base
                        }
                    } else if node == planned && fail_planned {
                        LayerResult {
                            locally_constructed: false,
                            ..LayerResult::error(
                                forward.request_id,
                                "Worker: Service unavailable: out of memory",
                            )
                        }
                    } else {
                        assert!(node == planned || node == standby);
                        to_sampler = Some(forward.sampling.clone());
                        LayerResult {
                            token_ids: vec![7],
                            finish_reason: None,
                            ..base
                        }
                    };
                    assert!(harness_state.resolve_pending_layer_result(Some(&node), result));
                }
                (
                    to_first.flatten(),
                    to_sampler.expect("a sampler was sent a forward"),
                )
            });
            executor
                .forward_through_segments(request_id, 0, 0, vec![0x11; 64], None, false, &[])
                .await
                .expect("the sampling segment answered");
            drop(executor);
            harness.await.unwrap()
        }

        for fail_planned in [false, true] {
            let (to_first, to_sampler) = sent(fail_planned, FORWARD_SAMPLING).await;
            let got = to_sampler.expect("a sampler that reads the trailer is sent the sampling");
            assert_eq!(
                (got.temperature, got.top_k, got.presence_penalty),
                (0.0, 1, 0.5),
                "the CALLER's parameters, not the worker's 0.7 / 0.9 / 40 (failover: {fail_planned})"
            );
            assert!(
                to_first.is_none(),
                "a segment that does not sample is not sent them"
            );
            assert!(
                sent(fail_planned, 0).await.1.is_none(),
                "an older peer is never sent a trailer it cannot parse (failover: {fail_planned})"
            );
        }
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
        let fwd = build_spec_verify_forward(
            request_id,
            42,
            activations.clone(),
            &segment,
            Some(100),
            None,
            None,
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
            None,
            crate::inference::process_pool::Requester::Owner,
            false,
            &mut false,
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

    /// FUTURE_WORK #3 (d): a round of a split's reply is counted where it
    /// leaves — a plain step through `forward_through_segments`, a guess check
    /// of any length through `forward_verify_through_segments` — once,
    /// whatever then becomes of it. The prompt pass is not counted there: the
    /// executor adds it when it records the attempt.
    #[tokio::test]
    async fn a_round_of_a_split_is_counted_where_it_leaves_and_the_prompt_pass_is_not() {
        let state = make_test_state();
        // Every send fails at once: what is counted is the round LEAVING.
        let (tx, rx) = mpsc::channel::<NetworkCommand>(8);
        drop(rx);
        let request = make_test_request(&state);
        let request_id = request.id;
        let trace = Arc::new(crate::inference::trace::RequestTrace::new(
            request_id, "test", "chat",
        ));
        state.active_traces.insert(request_id, trace.clone());
        let (a, d) = (NodeId([0xA1; 32]), NodeId([0xD4; 32]));
        state.peer_id_map.insert(a.clone(), vec![0xA1]);
        state.peer_id_map.insert(d.clone(), vec![0xD4]);
        let segments = vec![remote_segment(&a, (0, 16)), remote_segment(&d, (16, 32))];
        let assignment = PipelineAssignment {
            request_id,
            segments: segments.clone(),
            standbys: vec![],
            tp_groups: vec![],
            supports_speculative: true,
        };
        let mut executor = PipelineExecutor::new(state.clone(), tx.clone(), request, assignment);
        let within = std::time::Duration::from_secs(10);

        let _ = tokio::time::timeout(
            within,
            executor.forward_through_segments(
                request_id,
                0,
                0,
                b"hello".to_vec(),
                None,
                false,
                &[],
            ),
        )
        .await
        .expect("a failed send answers at once");
        assert_eq!(
            trace.decode_rounds(),
            0,
            "the prompt pass is not a decode round"
        );

        let _ = tokio::time::timeout(
            within,
            executor.forward_through_segments(
                request_id,
                1,
                5,
                vec![7, 0, 0, 0],
                None,
                false,
                &[7],
            ),
        )
        .await
        .expect("a failed send answers at once");
        assert_eq!(trace.decode_rounds(), 1, "a plain step is one round");

        let _ = tokio::time::timeout(
            within,
            forward_verify_through_segments(
                &state,
                &tx,
                request_id,
                6,
                &segments,
                &[7, 8, 9],
                None,
                None,
                crate::inference::process_pool::Requester::Owner,
                false,
                &mut false,
            ),
        )
        .await
        .expect("a failed send answers at once");
        assert_eq!(
            trace.decode_rounds(),
            2,
            "a check of three guesses is ONE round"
        );
    }

    /// FUTURE_WORK #3 (d): a finished split is recorded at its tokens over
    /// its rounds plus the prompt pass — and only a split (two or more
    /// segments) of a request that may guess ahead, the planner's own
    /// question, whose rounds ran here.
    #[test]
    fn a_finished_split_is_recorded_at_its_tokens_a_round_only_where_it_is_priced_so() {
        let state = make_test_state();
        let (tx, _rx) = mpsc::channel::<NetworkCommand>(8);
        let (a, d) = (NodeId([0xA1; 32]), NodeId([0xD4; 32]));
        let executor_for = |request: InferenceRequest, segments: Vec<PipelineSegment>| {
            let assignment = PipelineAssignment {
                request_id: request.id,
                segments,
                standbys: vec![],
                tp_groups: vec![],
                supports_speculative: true,
            };
            PipelineExecutor::new(state.clone(), tx.clone(), request, assignment)
        };
        let split = || vec![remote_segment(&a, (0, 16)), remote_segment(&d, (16, 32))];
        // An attempt whose decode sent `rounds` rounds and wrote `tokens`.
        let finish = |executor: &PipelineExecutor, rounds: u32, tokens: u32| {
            let id = executor.request.id;
            let trace = Arc::new(crate::inference::trace::RequestTrace::new(
                id, "test", "chat",
            ));
            state.active_traces.insert(id, trace.clone());
            for _ in 0..rounds {
                trace.note_decode_round();
            }
            let out = InferenceOutput {
                request_id: id,
                content: String::new(),
                reasoning: None,
                prompt_tokens: 5,
                completion_tokens: tokens,
                finish_reason: "stop".into(),
                session_id: None,
                token_logprobs: vec![],
                matched_stop_sequence: None,
                trace: None,
            };
            executor.note_split_rounds(&out, 0);
        };
        let per_round = || state.metrics.split_rounds.tokens_per_round("test");

        // Not a split, no round here, an adapter (one round a token, which
        // would lower the figure for the requests it prices): none recorded.
        let mut with_adapter = make_test_request(&state);
        with_adapter.lora_adapter = Some("an-adapter".into());
        for _ in 0..crate::daemon::state::recent_by_model::BEFORE_TRUSTED {
            finish(
                &executor_for(make_test_request(&state), vec![remote_segment(&a, (0, 32))]),
                9,
                30,
            );
            finish(&executor_for(make_test_request(&state), split()), 0, 30);
            let mut adapted = with_adapter.clone();
            adapted.id = uuid::Uuid::new_v4();
            finish(&executor_for(adapted, split()), 9, 30);
        }
        assert_eq!(
            per_round(),
            None,
            "only a split this node drove may be recorded"
        );

        // 30 tokens in 9 rounds and the prompt pass: 3 a round.
        for _ in 0..crate::daemon::state::recent_by_model::BEFORE_TRUSTED {
            finish(&executor_for(make_test_request(&state), split()), 9, 30);
        }
        assert_eq!(per_round(), Some(3.0));
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
            forward_verify_through_segments(
                &s2,
                &tx,
                request_id,
                0,
                &segs,
                &verify_tokens,
                None,
                None,
                crate::inference::process_pool::Requester::Owner,
                false,
                &mut false,
            )
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
            if state
                .pending_layer_results
                .contains_key(&WaiterKey::request(request_id))
            {
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
            state
                .observed_latency_ms_per_layer(&peer, &ModelId("test-model".into()))
                .is_none(),
            "nothing measured yet — if this starts as Some the test proves nothing"
        );

        run_one_verify_round(&state, &segments, vec![7]).await;
        run_one_verify_round(&state, &segments, vec![7]).await;

        assert!(
            state
                .observed_latency_ms_per_layer(&peer, &ModelId("test-model".into()))
                .is_some(),
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
            state
                .observed_latency_ms_per_layer(&peer, &ModelId("test-model".into()))
                .is_none(),
            "a batch of drafts is not one decode step; recording it as one \
             overstates ms/layer by roughly the draft width"
        );
    }

    /// A node with `inference.local_embedding_privacy` on and its data in a
    /// directory of its own (no model files: an embedder cannot be loaded
    /// from it, only inserted).
    fn privacy_state(on: bool) -> (Arc<SharedState>, tempfile::TempDir) {
        let temp = tempfile::tempdir().unwrap();
        let mut config = Config::default();
        config.node.data_dir = temp.path().to_path_buf();
        config.inference.local_embedding_privacy = on;
        let db = Database::open(temp.path()).unwrap();
        let executor = Arc::new(Mutex::new(ModelExecutor::new()));
        let (state, _, _) = SharedState::new(config, Identity::generate(), db, executor, None);
        (state, temp)
    }

    /// One peer, `0xB2`, holding every layer and told embedded prompts apart.
    fn privacy_peer(state: &SharedState) -> (NodeId, std::collections::HashMap<Vec<u8>, NodeId>) {
        use swarmllm_types::node::features::FORWARD_PRE_EMBEDDED;
        let peer = NodeId([0xB2; 32]);
        state.peer_id_map.insert(peer.clone(), vec![0xB2]);
        state
            .peer_registry
            .insert(peer.clone(), peer_advertising(&peer, FORWARD_PRE_EMBEDDED));
        let peers = std::collections::HashMap::from([(vec![0xB2], peer.clone())]);
        (peer, peers)
    }

    /// FUTURE_WORK #316: under prompt privacy the prompt reaches another
    /// computer's first segment EMBEDDED, and its pass is read from position
    /// 0 like every other — it was sent at the prompt's length, so every row
    /// was rotated that far and the reply decoded from before them.
    #[tokio::test]
    async fn a_prompt_embedded_here_is_sent_embedded_and_read_from_position_zero() {
        let (state, _dir) = privacy_state(true);
        let (peer, peers) = privacy_peer(&state);
        // One row per byte: with no tokenizer the embedder reads byte by byte.
        let table = candle_core::Tensor::arange(0f32, 1024f32, &candle_core::Device::Cpu)
            .and_then(|t| t.reshape((256, 4)))
            .unwrap();
        state.local_embedders.insert(
            ModelId("test".into()),
            Arc::new(crate::inference::local_embedder::LocalEmbedder::from_table(
                table,
            )),
        );
        let (tx, rx) = mpsc::channel::<NetworkCommand>(64);
        let mut request = make_test_request(&state);
        request.sampling_params.max_tokens = 2;
        let assignment = PipelineAssignment {
            request_id: request.id,
            segments: vec![remote_segment(&peer, (0, 32))],
            standbys: vec![],
            tp_groups: vec![],
            supports_speculative: false,
        };
        let mut executor = PipelineExecutor::new(state.clone(), tx, request, assignment);
        let peers_task = spawn_piece_peers(state.clone(), rx, peers, |_, _| PeerReply::Token(7));
        let _ = executor.execute(None).await;
        drop(executor);
        let sent = peers_task.await.unwrap();

        let (to, prompt) = sent.first().expect("the prompt pass was sent");
        assert_eq!(*to, peer);
        assert_eq!(prompt.sequence_num, 0);
        assert!(
            prompt.pre_embedded,
            "the prompt goes embedded, never as text"
        );
        assert_eq!(
            prompt.index_pos, 0,
            "the prompt pass is read from position 0"
        );
        assert!(
            sent.iter().all(|(_, f)| f.pre_embedded),
            "every step's token goes embedded too"
        );
    }

    /// FUTURE_WORK #316: where this node cannot embed the prompt (it does not
    /// hold the model's first part) and the first segment is another
    /// computer's, the request is refused — nothing is sent. It used to go
    /// out as text: the embedder never loaded, and nothing checked.
    #[tokio::test]
    async fn a_prompt_that_cannot_be_embedded_here_is_never_sent_to_another_computer() {
        let (state, _dir) = privacy_state(true);
        let (peer, peers) = privacy_peer(&state);
        let (tx, rx) = mpsc::channel::<NetworkCommand>(64);
        let mut request = make_test_request(&state);
        request.sampling_params.max_tokens = 2;
        let assignment = PipelineAssignment {
            request_id: request.id,
            segments: vec![remote_segment(&peer, (0, 32))],
            standbys: vec![],
            tp_groups: vec![],
            supports_speculative: false,
        };
        let mut executor = PipelineExecutor::new(state.clone(), tx, request, assignment);
        let peers_task = spawn_piece_peers(state.clone(), rx, peers, |_, _| PeerReply::Token(7));
        let result = executor.execute(None).await;
        drop(executor);
        let sent = peers_task.await.unwrap();

        assert!(
            matches!(result, Err(SwarmError::EmbeddingPrivacyUnavailable { .. })),
            "{result:?}"
        );
        assert!(sent.is_empty(), "nothing reached the peer: {sent:?}");
    }

    /// #316's review: the prompt is embedded only where the first segment's
    /// input could reach another computer. With every first layer here the
    /// worker reads token ids as it always has — embedding them anyway dropped
    /// a LoRA adapter, handed the tensor-parallel embed step floats as text and
    /// skipped the split prompt cache. Where the first layers may leave and
    /// nothing can be embedded here, the refusal names THIS setting.
    #[tokio::test]
    async fn the_prompt_is_embedded_only_where_the_first_layers_may_leave() {
        async fn embeds(
            privacy: bool,
            first_here: bool,
            remote_standby: bool,
            loadable: bool,
        ) -> Result<bool, SwarmError> {
            let (state, _dir) = privacy_state(privacy);
            let (peer, _) = privacy_peer(&state);
            if loadable {
                let table = candle_core::Tensor::zeros(
                    (256, 4),
                    candle_core::DType::F32,
                    &candle_core::Device::Cpu,
                )
                .unwrap();
                state.local_embedders.insert(
                    ModelId("test".into()),
                    Arc::new(crate::inference::local_embedder::LocalEmbedder::from_table(
                        table,
                    )),
                );
            }
            let me = state.identity.node_id().clone();
            let (tx, _rx) = mpsc::channel::<NetworkCommand>(4);
            let request = make_test_request(&state);
            let assignment = PipelineAssignment {
                request_id: request.id,
                segments: vec![remote_segment(
                    if first_here { &me } else { &peer },
                    (0, 32),
                )],
                standbys: if remote_standby {
                    vec![remote_segment(&peer, (0, 16))]
                } else {
                    vec![]
                },
                tp_groups: vec![],
                supports_speculative: false,
            };
            let executor = PipelineExecutor::new(state.clone(), tx, request, assignment);
            executor
                .prompt_embedder(uuid::Uuid::new_v4())
                .await
                .map(|e| e.is_some())
        }
        assert!(
            matches!(embeds(true, true, false, true).await, Ok(false)),
            "every first layer here: token ids, as before"
        );
        assert!(
            matches!(embeds(true, false, false, true).await, Ok(true)),
            "first segment elsewhere"
        );
        assert!(
            matches!(embeds(true, true, true, true).await, Ok(true)),
            "a stand-in elsewhere for layer 0"
        );
        assert!(
            matches!(embeds(false, false, false, true).await, Ok(false)),
            "privacy off"
        );
        assert!(
            matches!(embeds(true, true, false, false).await, Ok(false)),
            "nothing can be embedded and nothing leaves: served"
        );
        assert!(
            matches!(
                embeds(true, false, false, false).await,
                Err(SwarmError::EmbeddingPrivacyUnavailable { .. })
            ),
            "nothing can be embedded and the first layers leave: refused"
        );
    }

    /// #316's review: a tensor-parallel group reads its input as the prompt's
    /// text on both of its paths, so an embedded first segment goes the
    /// ordinary way, which says it is embedded. A first segment of token ids,
    /// and every later segment, keep their group.
    #[test]
    fn an_embedded_first_segment_is_never_run_by_a_tensor_parallel_group() {
        let (state, _dir) = privacy_state(true);
        let (peer, _) = privacy_peer(&state);
        let me = state.identity.node_id().clone();
        let (tx, _rx) = mpsc::channel::<NetworkCommand>(4);
        let request = make_test_request(&state);
        let first = remote_segment(&peer, (0, 16));
        let second = remote_segment(&me, (16, 32));
        let group = |layer_range: (u32, u32)| crate::types::TensorParallelGroup {
            nodes: vec![me.clone(), peer.clone()],
            layer_range,
            shard_ids: vec![],
        };
        let assignment = PipelineAssignment {
            request_id: request.id,
            segments: vec![first.clone(), second.clone()],
            standbys: vec![],
            tp_groups: vec![group((0, 16)), group((16, 32))],
            supports_speculative: false,
        };
        let executor = PipelineExecutor::new(state.clone(), tx, request, assignment);
        assert!(
            executor
                .tensor_parallel_group_for(&first, 0, true)
                .is_none(),
            "an embedded first segment"
        );
        assert!(
            executor
                .tensor_parallel_group_for(&first, 0, false)
                .is_some(),
            "control: token ids, the group runs it"
        );
        assert!(
            executor
                .tensor_parallel_group_for(&second, 1, true)
                .is_some(),
            "a later segment keeps its group"
        );
    }

    /// FUTURE_WORK #316: every fast path sends the first segment token ids, so
    /// under prompt privacy none takes a plan whose first layers another
    /// computer may read — the first segment, or a stand-in for layer 0.
    #[test]
    fn prompt_privacy_keeps_the_fast_paths_off_a_plan_whose_first_layers_may_leave() {
        let disqualified = |privacy: bool, first_here: bool, remote_standby: bool| {
            let (state, _dir) = privacy_state(privacy);
            let (peer, _) = privacy_peer(&state);
            let me = state.identity.node_id().clone();
            let first = if first_here {
                remote_segment(&me, (0, 32))
            } else {
                remote_segment(&peer, (0, 32))
            };
            let (tx, _rx) = mpsc::channel::<NetworkCommand>(4);
            let request = make_test_request(&state);
            let assignment = PipelineAssignment {
                request_id: request.id,
                segments: vec![first],
                standbys: if remote_standby {
                    vec![remote_segment(&peer, (0, 16))]
                } else {
                    vec![]
                },
                tp_groups: vec![],
                supports_speculative: true,
            };
            let executor = PipelineExecutor::new(state, tx, request, assignment);
            fastpath_request_disqualified(&executor)
        };
        assert!(disqualified(true, false, false), "first segment elsewhere");
        assert!(
            disqualified(true, true, true),
            "a stand-in elsewhere for layer 0"
        );
        assert!(!disqualified(true, true, false), "every first layer here");
        assert!(!disqualified(false, false, false), "privacy off: as before");
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
                None,
                crate::inference::process_pool::Requester::Owner,
                false,
                &mut false,
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

    /// The forward a one-segment verify round sends to `peer`, asked to walk.
    async fn verify_forward_sent_to(
        state: Arc<SharedState>,
        peer: NodeId,
        sampling: crate::types::SamplingParams,
    ) -> crate::types::LayerForward {
        state.peer_id_map.insert(peer.clone(), vec![0xAB]);
        let (tx, mut rx) = mpsc::channel::<NetworkCommand>(8);
        tokio::spawn(async move {
            let _ = forward_verify_through_segments(
                &state,
                &tx,
                uuid::Uuid::new_v4(),
                0,
                &[seg(peer)],
                &[7u32, 8, 9],
                None,
                Some(TailWalk {
                    drafts: &[8, 9],
                    sampling: &sampling,
                    generated: &[3, 7],
                    coupling: Some(0x5EED),
                    parents: &[],
                }),
                crate::inference::process_pool::Requester::Owner,
                false,
                &mut false,
            )
            .await;
        });
        match tokio::time::timeout(std::time::Duration::from_secs(5), rx.recv())
            .await
            .expect("a forward must be dispatched")
            .expect("channel open")
        {
            NetworkCommand::SendTensor { forward, .. } => forward,
            other => panic!("expected SendTensor, got {other:?}"),
        }
    }

    /// **A round never takes the reply past `max_tokens`, nor past its end.**
    /// The overshoot was the sampled token after a full set of kept guesses.
    #[test]
    fn a_round_is_cut_at_the_budget_and_at_the_end_of_the_reply() {
        let eos: std::collections::HashSet<u32> = [2].into_iter().collect();
        assert_eq!(
            super::round_tokens_for_reply(&[10, 11, 12], 13, &eos, 3),
            vec![10, 11, 12]
        );
        assert_eq!(
            super::round_tokens_for_reply(&[10, 11, 12], 13, &eos, 9),
            vec![10, 11, 12, 13]
        );
        assert_eq!(
            super::round_tokens_for_reply(&[10, 2, 12], 13, &eos, 9),
            vec![10, 2]
        );
        assert_eq!(super::round_tokens_for_reply(&[], 13, &eos, 1), vec![13]);
        assert_eq!(
            super::round_tokens_for_reply(&[], 13, &eos, 0),
            vec![13],
            "a round always hands back the token it sampled"
        );
    }

    /// The first forward a check over `segments` sends, with `may_chain`.
    async fn first_check_forward(
        state: Arc<SharedState>,
        segments: Vec<PipelineSegment>,
        may_chain: bool,
    ) -> (Vec<u8>, crate::types::LayerForward) {
        let (tx, mut rx) = mpsc::channel::<NetworkCommand>(8);
        let sampling = crate::types::SamplingParams::default();
        tokio::spawn(async move {
            let _ = forward_verify_through_segments(
                &state,
                &tx,
                uuid::Uuid::new_v4(),
                5,
                &segments,
                &[7u32, 8, 9],
                Some(4),
                Some(TailWalk {
                    drafts: &[8, 9],
                    sampling: &sampling,
                    generated: &[3, 7],
                    coupling: Some(0x5EED),
                    parents: &[],
                }),
                crate::inference::process_pool::Requester::Owner,
                may_chain,
                &mut false,
            )
            .await;
        });
        match tokio::time::timeout(std::time::Duration::from_secs(5), rx.recv())
            .await
            .expect("a forward must be dispatched")
            .expect("channel open")
        {
            NetworkCommand::SendTensor {
                target_peer_bytes,
                forward,
            } => (target_peer_bytes, forward),
            other => panic!("expected SendTensor, got {other:?}"),
        }
    }

    /// **A check runs a chain of remote segments in ONE trip**
    /// (`features::CHAINED_VERIFY`): the first hop is handed the rest of the
    /// run and the reply-to, and the forward carries what the TAIL walks with
    /// — guesses, walk flag, seed, rewind — which every hop hands on. Star-
    /// shaped, a plan of N remote segments paid N of this node's round trips
    /// per check. The controls: a caller that may not chain, and a hop
    /// without the bit (it would hand the tail a plain decode step), both get
    /// the old one-segment forward.
    #[tokio::test]
    async fn a_check_travels_a_chain_of_peers_that_can_carry_it() {
        use swarmllm_types::node::features::{
            CHAINED_VERIFY, COUPLED_SAMPLING, FORWARD_GENERATED_IDS, FORWARD_SAMPLING,
            PIPELINE_CHAIN, PIPELINE_CHAIN_V2, SPEC_WALK_AT_TAIL,
        };
        let base = PIPELINE_CHAIN
            | PIPELINE_CHAIN_V2
            | FORWARD_SAMPLING
            | FORWARD_GENERATED_IDS
            | SPEC_WALK_AT_TAIL
            | COUPLED_SAMPLING;
        let (b, c) = (NodeId([0x1b; 32]), NodeId([0x1c; 32]));
        let plan = |b: &NodeId, c: &NodeId| {
            vec![
                PipelineSegment {
                    node_id: b.clone(),
                    shard_id: ShardId {
                        model_id: ModelId("m".into()),
                        index: 0,
                    },
                    layer_range: (0, 14),
                },
                PipelineSegment {
                    node_id: c.clone(),
                    shard_id: ShardId {
                        model_id: ModelId("m".into()),
                        index: 2,
                    },
                    layer_range: (14, 28),
                },
            ]
        };
        let state_with = |c_features: u64| {
            let state = super::tests::make_test_state();
            state.peer_registry.insert(
                b.clone(),
                super::tests::peer_advertising(&b, base | CHAINED_VERIFY),
            );
            state
                .peer_registry
                .insert(c.clone(), super::tests::peer_advertising(&c, c_features));
            state.peer_id_map.insert(b.clone(), vec![0xB1]);
            state.peer_id_map.insert(c.clone(), vec![0xC1]);
            state
        };

        let state = state_with(base | CHAINED_VERIFY);
        let me = state.identity.node_id().0;
        let (to, fwd) = first_check_forward(state, plan(&b, &c), true).await;
        assert_eq!(to, vec![0xB1], "sent to the run's head");
        assert_eq!(fwd.chain.len(), 1, "the rest of the run rides with it");
        assert_eq!(fwd.chain[0].node_id, c);
        assert_eq!(fwd.chain[0].layer_range, (14, 28));
        assert_eq!(fwd.requester_node_id, Some(me), "the tail answers us");
        assert!(fwd.spec_logits_requested && fwd.spec_walk_at_tail);
        assert_eq!(fwd.draft_tokens, vec![8, 9], "the tail's guesses travel");
        assert_eq!(
            fwd.coupling_seed,
            Some(0x5EED),
            "the tail samples with the seed"
        );
        assert_eq!(fwd.truncate_kv_to, Some(4), "every hop rewinds");

        let (to, fwd) =
            first_check_forward(state_with(base | CHAINED_VERIFY), plan(&b, &c), false).await;
        assert_eq!(to, vec![0xB1]);
        assert!(
            fwd.chain.is_empty(),
            "a caller that may not chain sends the old way"
        );
        assert_eq!(fwd.requester_node_id, None);

        let (_, fwd) = first_check_forward(state_with(base), plan(&b, &c), true).await;
        assert!(
            fwd.chain.is_empty(),
            "a tail without CHAINED_VERIFY is never handed a check by its predecessor"
        );
        assert!(
            !fwd.spec_walk_at_tail,
            "and the head, not being the last segment, is not asked to walk"
        );
    }

    /// **A chained check that fails is re-run unchained, not the reply's end**
    /// (#318): the 3-machine rig's middle machine had not met the last, every
    /// chained check came back refused, and every reply ended after its first
    /// token. The re-run goes to the run's head alone, cuts back the rows the
    /// hops that ran wrote, and tells the caller to chain no more.
    #[tokio::test]
    async fn a_chained_check_that_fails_is_re_run_unchained() {
        use swarmllm_types::node::features::{
            CHAINED_VERIFY, FORWARD_SAMPLING, PIPELINE_CHAIN, PIPELINE_CHAIN_V2,
        };
        let every = PIPELINE_CHAIN | PIPELINE_CHAIN_V2 | FORWARD_SAMPLING | CHAINED_VERIFY;
        let (b, c) = (NodeId([0x2b; 32]), NodeId([0x2c; 32]));
        let state = super::tests::make_test_state();
        for (node, peer) in [(&b, 0xB2u8), (&c, 0xC2)] {
            state
                .peer_registry
                .insert(node.clone(), super::tests::peer_advertising(node, every));
            state.peer_id_map.insert(node.clone(), vec![peer]);
            state.connected_node_ids.insert(node.clone());
        }
        let segments = vec![
            PipelineSegment {
                node_id: b.clone(),
                shard_id: ShardId {
                    model_id: ModelId("m".into()),
                    index: 0,
                },
                layer_range: (0, 14),
            },
            PipelineSegment {
                node_id: c.clone(),
                shard_id: ShardId {
                    model_id: ModelId("m".into()),
                    index: 2,
                },
                layer_range: (14, 28),
            },
        ];
        let request_id = uuid::Uuid::new_v4();
        let (tx, mut rx) = mpsc::channel::<NetworkCommand>(8);
        let checking = {
            let state = state.clone();
            tokio::spawn(async move {
                let mut chain_failed = false;
                let _ = forward_verify_through_segments(
                    &state,
                    &tx,
                    request_id,
                    5,
                    &segments,
                    &[7u32],
                    None,
                    None,
                    crate::inference::process_pool::Requester::Owner,
                    true,
                    &mut chain_failed,
                )
                .await;
                chain_failed
            })
        };
        async fn wait_for_send(
            rx: &mut mpsc::Receiver<NetworkCommand>,
        ) -> (Vec<u8>, crate::types::LayerForward) {
            match tokio::time::timeout(std::time::Duration::from_secs(5), rx.recv())
                .await
                .expect("a forward must be dispatched")
            {
                Some(NetworkCommand::SendTensor {
                    target_peer_bytes,
                    forward,
                }) => (target_peer_bytes, forward),
                other => panic!("expected SendTensor, got {other:?}"),
            }
        }
        let (to, chained) = wait_for_send(&mut rx).await;
        assert_eq!(to, vec![0xB2]);
        assert_eq!(chained.chain.len(), 1, "precondition: the check chained");
        // The middle machine cannot reach the last.
        let refused = crate::types::LayerResult::error(
            request_id,
            "chained run could not reach the next segment",
        )
        .answering_step(chained.step());
        assert!(state.resolve_pending_layer_result(Some(&b), refused));

        let (to, rerun) = wait_for_send(&mut rx).await;
        assert_eq!(to, vec![0xB2], "the run's head again");
        assert!(rerun.chain.is_empty(), "unchained");
        assert_eq!(rerun.index_pos, 5);
        assert_eq!(
            rerun.truncate_kv_to,
            Some(5),
            "the head ran the chained check: its rows are cut back"
        );
        // Unchained, the head's own failure is the check's.
        let failed = crate::types::LayerResult::error(request_id, "worker gone")
            .answering_step(rerun.step());
        assert!(state.resolve_pending_layer_result(Some(&b), failed));
        assert!(
            tokio::time::timeout(std::time::Duration::from_secs(5), checking)
                .await
                .expect("the check ends")
                .unwrap(),
            "the caller is told to chain no more"
        );
        assert!(rx.try_recv().is_err(), "and nothing is re-run twice");
    }

    /// Only a peer that reads the walk flag AND the trailers the walk samples
    /// with is asked to walk. An older peer rebuilds the seal's AAD from the
    /// flags it parsed, so sending it bit 1 would fail every encrypted verify;
    /// it gets today's logits request, byte for byte.
    #[tokio::test]
    async fn only_a_peer_that_can_walk_is_asked_to_and_it_gets_the_callers_sampler() {
        use swarmllm_types::node::features::{
            FORWARD_GENERATED_IDS, FORWARD_SAMPLING, SPEC_WALK_AT_TAIL,
        };
        let penalised = crate::types::SamplingParams {
            temperature: 0.7,
            frequency_penalty: 0.5,
            ..Default::default()
        };

        let state = super::tests::make_test_state();
        let new_peer = NodeId([3u8; 32]);
        state.peer_registry.insert(
            new_peer.clone(),
            super::tests::peer_advertising(
                &new_peer,
                SPEC_WALK_AT_TAIL | FORWARD_SAMPLING | FORWARD_GENERATED_IDS,
            ),
        );
        let fwd = verify_forward_sent_to(state, new_peer, penalised.clone()).await;
        assert!(fwd.spec_walk_at_tail && fwd.spec_logits_requested);
        assert_eq!(
            fwd.coupling_seed, None,
            "the seed only to a peer that samples with it"
        );
        assert_eq!(
            fwd.draft_tokens,
            vec![8, 9],
            "a tail past segment 0 sees no token ids"
        );
        assert_eq!(fwd.sampling.as_ref().map(|s| s.temperature), Some(0.7));
        assert_eq!(
            fwd.generated_ids,
            vec![3, 7],
            "the penalties' history travels"
        );

        // A peer that also samples with shared noise is sent the seed.
        let state = super::tests::make_test_state();
        let coupled = NodeId([5u8; 32]);
        state.peer_registry.insert(
            coupled.clone(),
            super::tests::peer_advertising(
                &coupled,
                SPEC_WALK_AT_TAIL
                    | FORWARD_SAMPLING
                    | FORWARD_GENERATED_IDS
                    | swarmllm_types::node::features::COUPLED_SAMPLING,
            ),
        );
        let fwd = verify_forward_sent_to(state, coupled, penalised.clone()).await;
        assert_eq!(fwd.coupling_seed, Some(0x5EED));

        // Sampling and history but not the walk bit: an older peer.
        let state = super::tests::make_test_state();
        let old_peer = NodeId([4u8; 32]);
        state.peer_registry.insert(
            old_peer.clone(),
            super::tests::peer_advertising(&old_peer, FORWARD_SAMPLING | FORWARD_GENERATED_IDS),
        );
        let fwd = verify_forward_sent_to(state, old_peer, penalised).await;
        assert!(fwd.spec_logits_requested);
        assert!(
            !fwd.spec_walk_at_tail,
            "an older peer must never see the walk bit"
        );
        assert!(
            fwd.draft_tokens.is_empty() && fwd.sampling.is_none() && fwd.generated_ids.is_empty()
        );
    }

    /// `accept` is the one place a verify reply becomes tokens, and what no
    /// honest tail sends is refused rather than emitted.
    #[test]
    fn a_verify_reply_is_accepted_the_same_way_whichever_side_walked() {
        let greedy = crate::types::SamplingParams {
            temperature: 0.0,
            ..Default::default()
        };
        let rows = vec![
            vec![0.1, 5.0, 0.2],
            vec![7.0, 0.5, 0.3],
            vec![0.0, 0.1, 9.0],
        ];
        // The coordinator walking the logits and a tail that walked them agree.
        let here = VerifyReply::Logits(rows.clone())
            .accept(&[1, 2], &greedy, &[], None)
            .unwrap();
        assert_eq!(here, (vec![1], 0, false));
        let (mut kept, next, _) =
            crate::inference::sampling::sampled_accept_reject(&[1, 2], &rows, &greedy, &[], None);
        kept.push(next);
        assert_eq!(
            VerifyReply::Walked(kept)
                .accept(&[1, 2], &greedy, &[], None)
                .unwrap(),
            here
        );
        // All kept: the bonus follows.
        assert_eq!(
            VerifyReply::Walked(vec![1, 2, 5])
                .accept(&[1, 2], &greedy, &[], None)
                .unwrap(),
            (vec![1, 2], 5, true)
        );

        // Refused: a "kept" token that was never drafted, more kept than
        // drafted, an empty walk, too few rows, and non-finite logits — which
        // the rule used to answer with token 0 as the bonus.
        assert!(VerifyReply::Walked(vec![9, 4])
            .accept(&[1, 2], &greedy, &[], None)
            .is_err());
        assert!(VerifyReply::Walked(vec![1, 2, 3, 4])
            .accept(&[1, 2], &greedy, &[], None)
            .is_err());
        assert!(VerifyReply::Walked(vec![])
            .accept(&[1, 2], &greedy, &[], None)
            .is_err());
        assert!(VerifyReply::Logits(rows[..2].to_vec())
            .accept(&[1, 2], &greedy, &[], None)
            .is_err());
        let mut poisoned = rows;
        poisoned[1][0] = f32::NAN;
        assert!(VerifyReply::Logits(poisoned)
            .accept(&[1, 2], &greedy, &[], None)
            .is_err());
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

    /// A peer whose header and tensor table describe different uploads refuses
    /// at load (FUTURE_WORK #156). What reaches the coordinator must read as
    /// missing shard data, so the peer's claim is retracted, it is barred from
    /// the retry and the request re-routes — the same fact (it cannot give the
    /// bytes it advertised), and the response that keeps it out of every later
    /// plan until it announces again. The raw form is the worker's own
    /// `Display`, wrapped the way the IPC hop wraps it.
    #[test]
    fn a_peers_mixed_copy_refusal_reaches_the_coordinator_as_missing_shards() {
        let raw = format!(
            "Inference error: Worker: {}",
            crate::error::SwarmError::MixedModelCopy(
                "blk.3.attn_q.weight is 4608 bytes at byte 7003808 in the header, but 4608 \
                 bytes at byte 7000000 in the table its part was cut by"
                    .into()
            )
        );
        let over_the_wire = sanitize_peer_facing_error(&raw);
        assert!(
            remote_error_means_missing_shard(&over_the_wire),
            "raw {raw:?} was sanitised to {over_the_wire:?}"
        );
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
            locally_constructed: false,
            refusal: None,
            answers_step: None,
            prompt_blocks_stored: None,
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
        let err = every_holder_would_refuse(from_peer, Some(32768))
            .expect("a peer's Validation must stop the failover search");
        assert!(matches!(err, SwarmError::Validation(_)), "got {err:?}");
        assert_eq!(
            crate::error::classify_error(&err).0,
            axum::http::StatusCode::BAD_REQUEST
        );
    }

    /// The whole-model worker's refusal is judged by the same rule (#111): its
    /// "limit" is the window THAT node loaded, so below the model's declared
    /// context it is the peer's and another machine may serve it — and AT the
    /// declared context every holder would refuse, so it stays the caller's.
    #[test]
    fn a_whole_model_refusal_below_the_models_own_limit_is_the_peers() {
        let from_delegate = "Validation error: This conversation is too long for \
             qwen2.5-coder-7b: 9000 tokens of prompt against a limit of 8192, which leaves \
             no room for a reply. Shorten it by at least 808 tokens (roughly 606 words), or \
             start a new conversation.";
        assert!(
            every_holder_would_refuse(from_delegate, Some(32768)).is_none(),
            "8192 on a model declaring 32768 is that peer's ceiling"
        );
        assert!(
            every_holder_would_refuse(from_delegate, None).is_none(),
            "a coordinator without the header cannot rule another holder out"
        );
        assert!(
            every_holder_would_refuse(from_delegate, Some(8192)).is_some(),
            "at the model's own limit no machine serves more"
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
                every_holder_would_refuse(from_peer, Some(32768)).is_none(),
                "{from_peer:?} must remain failover-eligible"
            );
        }
    }

    /// A prompt longer than ONE peer serves is that peer's limit, and another
    /// holder may serve more — the field report of 2026-09-25, where an 8192
    /// peer's refusal ended a request the coordinator itself served at 65536.
    /// Only at the MODEL's own limit is it every holder's answer.
    #[test]
    fn a_prompt_longer_than_one_peer_serves_fails_over() {
        let from_peer = format!("Worker: {}", crate::error::longer_than_served(8560, 8192));
        assert!(
            every_holder_would_refuse(&from_peer, Some(32768)).is_none(),
            "8192 is the peer's ceiling, not the model's 32768: another holder may serve it"
        );
        assert!(
            every_holder_would_refuse(&from_peer, None).is_none(),
            "an unknown model limit must not end the search on one peer's say-so"
        );
        let at_the_models_limit = every_holder_would_refuse(&from_peer, Some(8192))
            .expect("no node serves more than the model declares");
        assert_eq!(
            crate::error::classify_error(&at_the_models_limit).0,
            axum::http::StatusCode::BAD_REQUEST
        );
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

#[cfg(test)]
mod failover_state_tests {
    use super::distributed::{cannot_resume_message, failover_can_restore_state};

    /// A stand-in can only reproduce the failed machine on the PROMPT PASS.
    ///
    /// After it, the KV cache the failed machine accumulated is gone: the
    /// forward carries the current step alone, the cache is keyed by
    /// `(layer range, request id)` so a machine that has not served this
    /// segment holds nothing, and `split::executor` then reads `kv_offset` from
    /// the cache rather than from `index_pos`. Measured in
    /// `examples/failover_kv_probe.rs`: replacing 4 of 28 layers takes the
    /// probability of the healthy machine's own token from 0.997 to 0.119.
    #[test]
    fn a_stand_in_can_only_reproduce_the_failed_machine_on_the_prompt_pass() {
        assert!(
            failover_can_restore_state(0),
            "the prompt pass hands the stand-in the whole prompt, so it builds \
             its own cache — this is the case standbys exist for and must keep \
             working"
        );

        // And every step after it, equally. There is deliberately no "recent
        // enough to be safe" window: what the stand-in is missing is the
        // PROMPT, so one token in is measurably worse than twenty-four
        // (P = 0.0000 against 0.119), not better.
        for seq in [1u32, 2, 8, 24, 500] {
            assert!(
                !failover_can_restore_state(seq),
                "a reply already under way (step {seq}) cannot be continued by a \
                 machine that holds none of its state"
            );
        }
    }

    /// The refusal must not borrow the wording of the case where no standby
    /// existed — here one may well have.
    #[test]
    fn the_refusal_says_the_stand_in_cannot_continue_not_that_none_was_free() {
        let m = cannot_resume_message(3, 412, Some("connection closed"));
        assert!(m.contains("Segment 3"), "{m}");
        assert!(
            m.contains("412"),
            "the reply's length is what makes this cost: {m}"
        );
        assert!(
            m.contains("cannot continue"),
            "the reason must name the missing state: {m}"
        );
        assert!(
            !m.contains("no standby available"),
            "that is the OTHER cause, and sends an operator hunting for capacity \
             they already have: {m}"
        );
        // The originating failure still travels, as it does on the sibling.
        assert!(m.contains("connection closed"), "{m}");

        // And it stands alone when there is no cause to report.
        let bare = cannot_resume_message(0, 1, None);
        assert!(!bare.contains("cause:"), "{bare}");
    }
}
