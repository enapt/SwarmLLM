//! Coordinator-side "remote-generate" fast path.
//!
//! When a distributed inference request has a single-segment pipeline (one
//! remote peer holds the entire layer range), we can bypass the per-token
//! coordinator/remote round trip entirely. The coordinator sends ONE
//! `RemoteGenerateRequest` to the holder, the holder runs the whole decode
//! loop in its local worker subprocess, and streams tokens back as
//! `StreamingToken` messages.
//!
//! This eliminates the ~140ms/token overhead (libp2p substream + JSON IPC)
//! that dominates the per-token path on loopback, leaving just compute +
//! single-frame network transit (~20-30ms/token). ~5-7x single-user speedup
//! for the common single-segment case.
//!
//! The same request also carries the DELEGATED SPLIT (`docs/FUTURE_WORK.md`
//! #143): a plan of several segments, none of them ours, is handed to the
//! holder of its first layers, which leads it among its own peers and streams
//! the reply back the same way. Only the request's contents and the
//! requester's reading of a failure differ — [`HandOff`].

use std::time::Duration;

use crate::error::SwarmError;
use crate::inference::router::{InferenceOutput, StreamingTokenEvent, StreamingTokenTx};
use crate::types::{NetworkCommand, NetworkFinishReason, RemoteGenerateRequest};

use super::PipelineExecutor;

/// May this completed request be used as a measurement of the peer's speed?
///
/// Extracted so the rule is testable without a network: the decision is made
/// deep inside a streaming loop, and every input that decides it is already
/// known by the time the reply is assembled.
///
/// A truncated stream is disqualified, and taking such a sample is worse than
/// taking none. Both halves of the arithmetic are corrupted in the same
/// direction — the elapsed time includes the deadline spent waiting for tokens
/// that never came, while the token count is the truncated one — so the
/// division yields the give-up timeout over the handful that survived rather
/// than a decode rate.
///
/// Measured live 2026-08-20: three requests to an RTX 4050 lost tokens in
/// transit and returned 3, 3 and 18 of 60. Those recorded the card at 345
/// ms/layer against the 3.1 it had measured minutes earlier on the same model,
/// and since a delegated observation outranks a peer's advertised speed, it
/// demoted the swarm's only GPU behind a laptop CPU for the ten minutes such a
/// figure takes to expire. A transport failure was being recorded as a fact
/// about the hardware.
///
/// The single-token case is excluded for the older reason: with one token
/// `total - ttft` is zero and says nothing about decode speed at all.
fn delegated_sample_is_usable(truncated: bool, completion_tokens: u32, layers: u32) -> bool {
    !truncated && completion_tokens > 1 && layers > 0
}

/// Burst budget for one full generation before backpressure applies to the
/// remote-generate token stream. Sized to comfortably hold a long completion's
/// worth of tokens without blocking the inbound dispatch task.
const REMOTE_GENERATE_TOKEN_CHANNEL_CAP: usize = 256;

/// Reassembles the reply stream of a remote generation.
///
/// Each `StreamingToken` is an independent request_response send — one
/// substream apiece — so the network gives no ordering guarantee between them
/// and the terminal "done" token can arrive before content tokens still in
/// flight. The coordinator used to stop at the first token carrying a
/// `finish_reason` and discard the rest, which truncated replies from distant
/// peers: measured 2026-08-09 against a peer at ~6s RTT, the same two-token
/// answer came back as "", "ch" and "Cherry" on successive attempts, while
/// `usage.completion_tokens` correctly said 2 every time (it rides on the done
/// token, which always arrives).
///
/// `token_id` numbers the content tokens and the done token carries the total,
/// so "the stream ended" and "the end overtook the middle" become
/// distinguishable. A server too old to fill it in sends zeros throughout;
/// `sequenced` stays false and delivery degrades to arrival order, which is
/// exactly the previous behaviour.
pub(super) struct StreamReassembler {
    /// True once any token has carried a non-zero id — i.e. the peer sequences.
    sequenced: bool,
    /// Next sequence number that may be emitted.
    next_seq: u32,
    /// Tokens that arrived ahead of their turn.
    pending: std::collections::BTreeMap<u32, swarmllm_types::StreamingToken>,
    /// Content-token count from the done token, once it has arrived.
    expected_total: Option<u32>,
}

impl StreamReassembler {
    pub(super) fn new() -> Self {
        Self {
            sequenced: false,
            next_seq: 0,
            pending: std::collections::BTreeMap::new(),
            expected_total: None,
        }
    }

    /// Accept a content token; returns whatever is now emittable, in order.
    ///
    /// Only the consecutive run from `next_seq` is released. Emitting past a
    /// gap would silently reorder the reply, which is worse than delaying it.
    pub(super) fn push_content(
        &mut self,
        tok: swarmllm_types::StreamingToken,
    ) -> Vec<swarmllm_types::StreamingToken> {
        if tok.token_id > 0 {
            self.sequenced = true;
        }
        // A token already emitted arriving again — a resend that raced the
        // original, or the original arriving after a resend filled its slot.
        // Without this guard it would sit in `pending` for ever, counted as
        // buffered and never drained.
        if self.sequenced && tok.token_id < self.next_seq {
            return Vec::new();
        }
        let slot = if self.sequenced {
            tok.token_id
        } else {
            self.next_seq
        };
        self.pending.insert(slot, tok);

        let mut ready = Vec::new();
        while let Some(t) = self.pending.remove(&self.next_seq) {
            self.next_seq = self.next_seq.saturating_add(1);
            ready.push(t);
        }
        ready
    }

    /// Record the done token's content-token total.
    pub(super) fn mark_done(&mut self, total: u32) {
        if total > 0 {
            self.sequenced = true;
        }
        self.expected_total = Some(total);
    }

    /// Has the done token arrived?
    pub(super) fn done_seen(&self) -> bool {
        self.expected_total.is_some()
    }

    /// Every token accounted for — safe to finish.
    ///
    /// An unsequenced peer cannot tell us what to wait for, so its done token
    /// completes the stream immediately, as it always did.
    pub(super) fn is_complete(&self) -> bool {
        match self.expected_total {
            None => false,
            Some(total) => !self.sequenced || self.next_seq >= total,
        }
    }

    /// Tokens the peer says it sent that have not been emitted.
    pub(super) fn missing(&self) -> u32 {
        self.expected_total
            .map(|t| t.saturating_sub(self.next_seq))
            .unwrap_or(0)
    }

    /// Did tokens the peer actually SENT fail to arrive?
    ///
    /// The only honest truncation oracle, and deliberately NOT
    /// `usage.completion_tokens > emitted()`. That figure counts tokens the
    /// model GENERATED, while the serving node skips forwarding any whose text
    /// is empty (`dispatch/remote_generate.rs`: `if evt.text.is_empty()
    /// { continue; }`) — so the two differ BY CONSTRUCTION on any reply
    /// containing tokens that decode to nothing. The done token carries
    /// `streamed_count`, which is what this compares against.
    ///
    /// The tokens that decode to nothing are MULTI-BYTE CHARACTERS.
    /// `decode_token` accumulates bytes in a `carry` buffer and returns empty
    /// until the codepoint completes, so one CJK character or emoji is several
    /// generated tokens and ONE sent token. Pure ASCII has almost none, which
    /// is why the counts agreed in testing and the fault stayed hidden.
    ///
    /// Measured against the released v0.3.135, same data dir, only the binary
    /// changed (gotcha #416): "one sentence in Chinese" answered
    /// `503 Reply truncated in transit: 1 of 3 tokens arrived`, and five emoji
    /// answered `9 of 12`. Both replies had arrived COMPLETE. The hint told the
    /// user their answer was lost and to try a different machine — advice that
    /// could never help, for a reply that was already correct. The product
    /// ships 21 locales and most are multi-byte, so this refused a large share
    /// of non-English replies from peer-held models.
    ///
    /// An unsequenced peer cannot say what to expect, so it can never be judged
    /// truncated — the same degradation `is_complete` makes.
    pub(super) fn truncated(&self) -> bool {
        self.sequenced && self.missing() > 0
    }

    /// How many content tokens the peer says it sent. Falls back to what
    /// arrived, so an error message can never read "3 of 0".
    pub(super) fn sent_by_peer(&self) -> u32 {
        self.expected_total.unwrap_or_else(|| self.emitted())
    }

    pub(super) fn emitted(&self) -> u32 {
        self.next_seq
    }

    pub(super) fn buffered(&self) -> usize {
        self.pending.len()
    }

    /// Is the next token to emit one the peer has (or says it has) sent, but
    /// that has not arrived? True when later tokens are buffered behind it,
    /// or when the done token promised more than has been emitted. Only ever
    /// true for a sequenced peer — an unsequenced one has no holes, only
    /// arrival order.
    pub(super) fn has_hole(&self) -> bool {
        self.sequenced && (!self.pending.is_empty() || self.missing() > 0)
    }

    /// The range `[from, to)` to ask the peer to resend: from the next token
    /// to emit up to the first token already buffered, or to the total the
    /// done token announced. `None` when there is nothing to ask for.
    pub(super) fn resend_range(&self) -> Option<(u32, u32)> {
        if !self.has_hole() {
            return None;
        }
        let from = self.next_seq;
        let to = self
            .pending
            .keys()
            .next()
            .copied()
            .or(self.expected_total)?;
        (to > from).then_some((from, to))
    }
}

/// Asks for a resend per reply before the requester falls back to waiting the
/// old deadlines out. Each ask costs one round trip; four cover a hole that
/// is itself lossy without letting a dead peer hold the request for long.
const MAX_RESEND_ASKS: u32 = 4;

/// Whether, and how often, to ask the serving peer to fill a hole.
///
/// Pure so the policy can be pinned without a network: it decides only
/// "may I ask again", given what the peer advertised. `SWARMLLM_RESEND_TOKENS=0`
/// answers no unconditionally — the A/B switch that shows the truncation
/// coming back inside one binary.
pub(super) struct ResendPolicy {
    supported: bool,
    asks: u32,
}

impl ResendPolicy {
    pub(super) fn new(peer_supports_resend: bool) -> Self {
        Self {
            supported: peer_supports_resend && !resend_disabled_by_env(),
            asks: 0,
        }
    }

    /// May an ask be made now? True while the peer supports it and the
    /// budget is not spent.
    pub(super) fn can_ask(&self) -> bool {
        self.supported && self.asks < MAX_RESEND_ASKS
    }

    /// Spend one ask. Returns false, spending nothing, when none is allowed.
    pub(super) fn ask(&mut self) -> bool {
        if !self.can_ask() {
            return false;
        }
        self.asks += 1;
        true
    }

    pub(super) fn asks(&self) -> u32 {
        self.asks
    }
}

/// `SWARMLLM_RESEND_TOKENS=0` disables asking for resends on this node, so a
/// lost token truncates the reply the way it did before — the control arm
/// for `examples/dropped_token_test.sh`. Read once.
fn resend_disabled_by_env() -> bool {
    use std::sync::OnceLock;
    static OFF: OnceLock<bool> = OnceLock::new();
    *OFF.get_or_init(|| {
        std::env::var("SWARMLLM_RESEND_TOKENS")
            .map(|v| v == "0" || v.eq_ignore_ascii_case("false"))
            .unwrap_or(false)
    })
}

/// How long to wait on a hole before asking for a resend: a few round trips
/// to the peer, so ordinary reordering settles on its own, bounded so a lost
/// token costs seconds rather than the 15 s straggler wait.
const HOLE_WAIT_MIN: Duration = Duration::from_secs(1);
const HOLE_WAIT_MAX: Duration = Duration::from_secs(5);

pub(super) fn hole_wait(peer_latency_ms: Option<u32>) -> Duration {
    Duration::from_millis(u64::from(peer_latency_ms.unwrap_or(0)) * 4)
        .clamp(HOLE_WAIT_MIN, HOLE_WAIT_MAX)
}

/// Which hand-off a `RemoteGenerateRequest` makes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum HandOff {
    /// The whole model, to the one peer holding it ([`eligible`]).
    WholeModel,
    /// A split none of which is ours, to the holder of its first layers, which
    /// LEADS it among its own peers ([`delegation_eligible`]).
    ///
    /// The requester reads a failure differently from a whole-model one: the
    /// error may be about a peer the delegate chose, so it retracts no claim
    /// and bars nobody, and the reply's speed is the delegate's PLAN's, not
    /// the delegate's — so it is not recorded as the delegate's.
    DelegatedSplit,
}

/// `SWARMLLM_DELEGATE_SPLIT=0` keeps every split coordinated here — the
/// control arm for measuring the delegated split in one binary. Read once.
fn delegation_switched_off() -> bool {
    use std::sync::OnceLock;
    static OFF: OnceLock<bool> = OnceLock::new();
    *OFF.get_or_init(|| {
        std::env::var("SWARMLLM_DELEGATE_SPLIT")
            .map(|v| v == "0" || v.eq_ignore_ascii_case("false"))
            .unwrap_or(false)
    })
}

/// Will the delegated split take this plan? The single answer
/// (`docs/FUTURE_WORK.md` #143).
///
/// The shape: several segments, NONE of them this node's. Here the
/// coordinator sits inside every token's round trip while holding no layer —
/// the tail answers it and it starts the next token at the head — so a
/// requester far from machines that are close to EACH OTHER paid its own
/// distance per token (measured 0.35 tok/s Thailand↔Italy against 6.76 at
/// 18 ms). Handed to the head, the loop runs at the holders' distance and
/// only the finished tokens cross the requester's.
///
/// What must hold for it to be the SAME request, served as safely:
/// - the head advertises `features::DELEGATED_SPLIT` (an older node would
///   refuse it as a whole model it does not hold);
/// - prompt privacy is off — the raw prompt reaches the head either way, but
///   privacy keeps layer 0 HERE, which a node holding nothing cannot honour
///   anyway, and the boomerang is never this shape;
/// - this node's private mode is off: its scope is "who may serve MY
///   request", and a delegate plans among peers this node never checked;
/// - no `swarm_route` override: it is an instruction to THIS node's planner,
///   and the delegate could not follow it;
/// - the request is not itself one a peer delegated to us (never handed on).
pub(crate) fn delegation_eligible(exec: &PipelineExecutor) -> bool {
    if delegation_switched_off() {
        return false;
    }
    if super::fastpath_request_disqualified(exec) {
        return false;
    }
    let segments = &exec.assignment.segments;
    if segments.len() < 2 {
        return false;
    }
    let me = exec.shared_state.identity.node_id();
    if segments.iter().any(|s| &s.node_id == me) {
        return false;
    }
    if exec.request.route_override.is_some() {
        return false;
    }
    let head = &segments[0];
    if head.layer_range.0 != 0 {
        return false;
    }
    if !exec
        .shared_state
        .peer_advertises_feature(&head.node_id, crate::types::features::DELEGATED_SPLIT)
    {
        return false;
    }
    let encrypted = exec
        .shared_state
        .encrypted_pipeline_for_request(&exec.request.model_id, exec.request.id);
    if encrypted || exec.shared_state.config.inference.local_embedding_privacy {
        return false;
    }
    if crate::pool::scope::allowed_node_set(&exec.shared_state).is_some() {
        return false;
    }
    true
}

/// Preconditions for the fast path. All checks are local and cheap.
/// Will the whole-model hand-off take this plan? The single answer — the n-gram
/// loop, which runs earlier in `execute_distributed`, asks it too and stands
/// aside for every plan this says yes to.
pub(super) fn eligible(exec: &PipelineExecutor) -> bool {
    // Shared disqualifiers: TP, LoRA adapter, vision images.
    if super::fastpath_request_disqualified(exec) {
        return false;
    }
    // Single segment.
    if exec.assignment.segments.len() != 1 {
        return false;
    }
    // The sole segment must be remote. Local inference is handled by
    // `execute_local` which has its own faster path.
    if exec.assignment.segments[0].node_id == *exec.shared_state.identity.node_id() {
        return false;
    }
    let model_id = &exec.request.model_id;
    // Ask the one accessor. Re-deriving it from the per-model map missed the
    // automatic case (`encrypted_pipeline_auto`, the default), so a model with
    // privacy in force did not disqualify the fast path here — the path that
    // puts the RAW PROMPT on the wire to a peer.
    //
    // The REQUEST form, which is the one the scheduler planned with: under a
    // `swarm_route` override that releases this node's shards, the automatic
    // default steps aside (an explicit per-model or global setting still
    // applies — `encrypted_pipeline_for_request`). Asking the model-only form
    // here disagreed with the plan: the scheduler handed the whole model to one
    // peer, and this refused the hand-off, so the request ran one round trip
    // per token instead (2026-09-27, the spread benchmark's peer arm).
    let encrypted_for_model = exec
        .shared_state
        .encrypted_pipeline_for_request(model_id, exec.request.id);
    // `encrypted_pipeline` forces local embedding (no raw tokens on wire).
    // `local_embedding_privacy` is similar. Both bypass the fast path.
    if encrypted_for_model || exec.shared_state.config.inference.local_embedding_privacy {
        return false;
    }
    true
}

/// Base budget for the first token: connection, queueing and the decode of a
/// short prompt on a peer believed WARM. A peer that may first have to load the
/// model is given `LoadAllowance` on top — this base used to claim the load as
/// well, and 120 s did not cover one on a busy machine (FUTURE_WORK #129).
const FIRST_TOKEN_TIMEOUT: Duration = Duration::from_secs(120);
/// Extra first-token budget per estimated prompt token.
///
/// This budget was originally flat, sized against how long *generation* takes.
/// It ignored prefill, which is linear in prompt length: a ~600-token prompt
/// measured 285s on a 6-core CPU node that was working perfectly normally, so
/// the flat 120s budget expired mid-prefill, the request retried, and failed
/// again — a long prompt could not succeed on a modest node at all.
const PREFILL_ALLOWANCE_PER_TOKEN: Duration = Duration::from_millis(500);
/// Ceiling on the first-token budget, so a genuinely dead peer is still
/// detected in bounded time no matter how long the prompt is.
const FIRST_TOKEN_TIMEOUT_MAX: Duration = Duration::from_secs(600);
/// Fallback characters-per-token divisor, used only when the model's tokenizer
/// isn't available locally (the coordinator of a remote generate often holds no
/// shard of the model, so it has no header to load one from).
///
/// Deliberately pessimistic. Latin prose runs about 4 characters per token, but
/// this must not under-budget the scripts this project ships locales for —
/// Chinese and Japanese are close to 1 character per token, so a divisor tuned
/// for English would silently reintroduce the premature timeout for exactly
/// those users. Overestimating only lengthens the wait, and the ceiling bounds
/// the damage.
const PROMPT_CHARS_PER_TOKEN: usize = 2;

/// First-token budget for a prompt of `prompt_tokens` tokens.
///
/// Only ever extends the base budget, never shortens it, so short prompts keep
/// exactly the previous behaviour.
///
/// `load` is what the answer may wait behind before any of that starts: a
/// machine that must first read the model into memory. Added ON TOP of the
/// ceiling, as the segment path does — a cold load is not prompt work and must
/// not be clipped away by the prompt's cap. Required, so no caller can wait on
/// a peer's first answer without having asked whether that peer is cold.
pub(crate) fn first_token_timeout(prompt_tokens: usize, load: super::LoadAllowance) -> Duration {
    let tokens = u32::try_from(prompt_tokens).unwrap_or(u32::MAX);
    FIRST_TOKEN_TIMEOUT
        .saturating_add(PREFILL_ALLOWANCE_PER_TOKEN.saturating_mul(tokens))
        .min(FIRST_TOKEN_TIMEOUT_MAX)
        .saturating_add(load.duration())
}

/// How often a wait for a peer's first token looks up whether the peer is
/// still connected.
///
/// A serving node aborts a hand-off the moment its last connection to the
/// requester closes — there is no route left to send tokens back on
/// (`connections::handle_connection_closed`). The requester kept waiting out
/// the whole first-token budget all the same, and that budget now includes a
/// cold-load allowance of minutes. Waking this often turns "the peer left" into
/// a retry within seconds, without touching the budget a working peer gets.
const PEER_PRESENCE_CHECK: Duration = Duration::from_secs(5);

/// What a wait for a peer's first token does when one of its slices runs out.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FirstTokenWait {
    /// Still inside the budget and the peer is still connected: wait again,
    /// at most this long.
    Again(Duration),
    /// The peer's last connection has closed — it has dropped the request.
    PeerLeft,
    /// The budget is spent.
    Expired,
}

/// Decide [`FirstTokenWait`] at `now`. Pure, so the rule is tested without a
/// network.
fn first_token_wait(
    now: std::time::Instant,
    deadline: std::time::Instant,
    peer_connected: bool,
) -> FirstTokenWait {
    if now >= deadline {
        FirstTokenWait::Expired
    } else if !peer_connected {
        FirstTokenWait::PeerLeft
    } else {
        FirstTokenWait::Again((deadline - now).min(PEER_PRESENCE_CHECK))
    }
}

/// Estimate the prompt's token count from characters, for when the tokenizer
/// isn't loadable. Counts `chars()`, not bytes — a byte-length divisor would
/// under-count multi-byte scripts by the very factor that makes them expensive.
pub(crate) fn estimate_prompt_tokens(prompt: &str) -> usize {
    prompt.chars().count().div_ceil(PROMPT_CHARS_PER_TOKEN)
}
/// Between-token timeout once generation has started. Generous to accommodate
/// slow prefill-then-decode transitions on big models.
const INTER_TOKEN_TIMEOUT: Duration = Duration::from_secs(60);

/// How long to keep waiting for content tokens that the server has already
/// sent but that have not arrived yet.
///
/// Every token is an independent request_response send, so the "done" token can
/// overtake tokens still in flight. Once done arrives the server has finished,
/// the stragglers are already on the wire, and this only has to cover transit —
/// generous at any real RTT (the worst peer observed was ~6s), while bounding
/// the wait when a send was genuinely dropped.
const STRAGGLER_TIMEOUT: Duration = Duration::from_secs(15);

impl PipelineExecutor {
    /// Try the remote-generate fast path. Returns `Ok(None)` if preconditions
    /// aren't met (caller falls back to `execute_distributed`'s standard
    /// loop). Returns `Ok(Some(_))` on success.
    /// The peer running the whole model refused `err_msg` as longer than the
    /// context it serves — and that is below the model's own, so another
    /// machine, this one included, may serve it: bar the peer from this
    /// request and answer the variant the router re-plans. `None` for any other
    /// error, and for a refusal at the model's declared limit, which every
    /// holder would give (`every_holder_would_refuse` decides both, for
    /// segments and here).
    ///
    /// The whole-model half of #111. A delegate on the shipped 8192 default
    /// answered a 9000-token prompt with its own ceiling, the coordinator
    /// returned it as the request's 400, and a model that declares 32768 was
    /// never tried anywhere else.
    fn longer_than_this_peer_serves(
        &self,
        err_msg: &str,
        segment: &crate::types::PipelineSegment,
    ) -> Option<SwarmError> {
        let refusal = crate::error::served_context_refusal(err_msg)?;
        let declared = self
            .shared_state
            .model_declared_context(&segment.shard_id.model_id);
        if super::every_holder_would_refuse(err_msg, declared).is_some() {
            return None;
        }
        self.shared_state
            .blacklist_holder_for_request(self.request.id, &segment.node_id);
        tracing::info!(
            request_id = %self.request.id,
            peer = %segment.node_id,
            tokens = refusal.tokens,
            peer_limit = refusal.limit,
            declared = ?declared,
            "remote-generate: the peer serves a shorter conversation than this one \
             — re-planning without it"
        );
        Some(SwarmError::LongerThanPeerServes(
            super::distributed::longer_than_the_swarm_serves_text(
                refusal.tokens,
                refusal.limit,
                segment.layer_range,
            ),
        ))
    }

    pub(super) async fn try_remote_generate_fastpath(
        &mut self,
        token_tx: Option<StreamingTokenTx>,
    ) -> Result<Option<InferenceOutput>, SwarmError> {
        if !eligible(self) {
            return Ok(None);
        }
        let wire_id = self.request.id;
        self.run_hand_off(token_tx, HandOff::WholeModel, wire_id)
            .await
    }

    /// Hand a split none of which is ours to the holder of its first layers,
    /// which leads it and streams the reply back (`delegation_eligible`).
    ///
    /// `Ok(None)` — run this same plan here instead — when the plan is not
    /// that shape, and when the delegate failed BEFORE a single token of the
    /// reply arrived: it declined (it no longer holds the first layers, its
    /// router could not plan, its queue was full) or it never answered. Every
    /// one of those leaves this node able to coordinate the plan it already
    /// has, and nothing has been shown to the caller twice. A failure after
    /// tokens arrived is the request's, exactly as for a whole-model hand-off.
    pub(super) async fn try_delegated_split(
        &mut self,
        token_tx: Option<StreamingTokenTx>,
    ) -> Result<Option<InferenceOutput>, SwarmError> {
        if !delegation_eligible(self) {
            return Ok(None);
        }
        let head = self.assignment.segments[0].node_id.clone();
        tracing::info!(
            request_id = %self.request.id,
            delegate = %head,
            segments = self.assignment.segments.len(),
            "delegated split: this node holds none of the plan — handing it to the \
             holder of its first layers to lead"
        );
        self.hand_off_emitted = 0;
        // Its own id on the wire, per ATTEMPT. After a fallback this node sends
        // its OWN forwards to the delegate under the request id, so a cancel
        // of the abandoned hand-off — or a late token from it — must not be
        // able to name them; and a router retry that delegates again must not
        // collect the first attempt's stragglers (`arch-scheduling.md` § "State
        // that belongs to an ATTEMPT").
        let wire_id = uuid::Uuid::new_v4();
        let outcome = self
            .run_hand_off(token_tx, HandOff::DelegatedSplit, wire_id)
            .await;
        if outcome.is_err() {
            // Whatever the delegate is still doing is for nobody now: a reply
            // this node has failed, or a plan it is about to run itself.
            self.shared_state.streaming_token_txs.remove(&wire_id);
            if let Some(target) = self.shared_state.resolve_peer_id_bytes(&head) {
                let _ = self
                    .network_tx
                    .send(NetworkCommand::SendDirectMessage {
                        target_peer_bytes: target,
                        message: crate::types::SwarmMessage::CancelInference(
                            swarmllm_types::CancelInference {
                                request_id: wire_id,
                            },
                        ),
                        delivery_request_id: None,
                    })
                    .await;
            }
        }
        match outcome {
            Err(e) if self.hand_off_emitted == 0 => {
                tracing::warn!(
                    request_id = %self.request.id,
                    delegate = %head,
                    error = %e,
                    "delegated split: the delegate did not serve it — coordinating the \
                     same plan from here instead"
                );
                Ok(None)
            }
            other => other,
        }
    }

    /// Send a `RemoteGenerateRequest` to the head of the plan and turn the
    /// tokens it streams back into this request's reply. One protocol for
    /// both hand-offs; `hand_off` says which, and so how the request is built
    /// and how a failure is read.
    async fn run_hand_off(
        &mut self,
        token_tx: Option<StreamingTokenTx>,
        hand_off: HandOff,
        // The id the request, its tokens, resend asks and cancels travel
        // under. A whole model's is the request's own, as it always was.
        wire_id: uuid::Uuid,
    ) -> Result<Option<InferenceOutput>, SwarmError> {
        let whole_model = hand_off == HandOff::WholeModel;
        let request_id = self.request.id;
        // Between two tokens a delegate may be waiting out ITS segment's
        // deadline and failing over — the recovery that delegation moves with
        // the loop. A gap shorter than that turns its recovery into this
        // node's failure, so it is the usual gap plus one decode deadline for
        // the whole model (`compute_segment_timeout`'s rule), sized by the
        // model and capped with it.
        let inter_token_timeout = match hand_off {
            HandOff::WholeModel => INTER_TOKEN_TIMEOUT,
            HandOff::DelegatedSplit => {
                let layers = self
                    .assignment
                    .segments
                    .last()
                    .map(|s| s.layer_range.1)
                    .unwrap_or(0);
                INTER_TOKEN_TIMEOUT
                    + Duration::from_secs((u64::from(layers) * super::DECODE_SECS_PER_LAYER).clamp(
                        super::SEGMENT_TIMEOUT_MIN_SECS,
                        super::SEGMENT_TIMEOUT_MAX_SECS,
                    ))
            }
        };
        let segment = self.assignment.segments[0].clone();
        let target_peer_bytes = self
            .shared_state
            .resolve_peer_id_bytes(&segment.node_id)
            .ok_or_else(|| {
                SwarmError::Network(format!("No peer_id_bytes for node {}", segment.node_id))
            })?;

        // Build the chat-templated prompt AND the stop strings that template
        // implies. Taking the prompt without the stops is the bug this pairing
        // exists to prevent: the serving node only truncates the stop list it
        // is handed, so a marker we do not send is one nothing will ever match.
        let (prompt, sampling) = self.build_prompt_and_stops().await;
        // Sized from the prompt we are about to send, before it is moved.
        // Prefer the model's own tokenizer: a character heuristic mis-sizes the
        // budget by several-fold across scripts. It is often unavailable here —
        // this path exists precisely because the model runs elsewhere — so fall
        // back to the estimate.
        let prompt_tokens_est = self
            .shared_state
            .standalone_tokenizer(&self.request.model_id)
            .map(|tk| tk.encode(&prompt).len())
            .unwrap_or_else(|| estimate_prompt_tokens(&prompt));
        // Whom the first token waits behind: the one peer for a whole model,
        // every segment of the plan for a delegate that will lead it.
        let waited_on = match hand_off {
            HandOff::WholeModel => &self.assignment.segments[..1],
            HandOff::DelegatedSplit => &self.assignment.segments[..],
        };
        let load = super::LoadAllowance::for_segments(&self.shared_state, waited_on);
        let first_token_budget = first_token_timeout(prompt_tokens_est, load);

        // Register an inbound StreamingToken channel before sending the
        // request so we never miss an early token.
        let (stream_tx, mut stream_rx) = tokio::sync::mpsc::channel::<crate::types::StreamingToken>(
            REMOTE_GENERATE_TOKEN_CHANNEL_CAP,
        );
        self.shared_state.streaming_token_txs.insert(
            wire_id,
            crate::daemon::state::StreamingTokenSink {
                tx: stream_tx,
                // A retry keeps the request id, so the peer is what
                // distinguishes this attempt from the one it replaced.
                expected_peer: segment.node_id.clone(),
            },
        );

        // Send the RemoteGenerateRequest. A delegated split names the WHOLE
        // model and carries the conversation itself: the delegate renders it
        // through its own router, so every path it can choose builds the
        // prompt the one way it always does. `prompt` rides along as sent
        // today, informational for a delegate.
        let (layer_range, delegation) = match hand_off {
            HandOff::WholeModel => (segment.layer_range, None),
            HandOff::DelegatedSplit => (
                (
                    0,
                    self.assignment
                        .segments
                        .last()
                        .map(|s| s.layer_range.1)
                        .unwrap_or(segment.layer_range.1),
                ),
                Some(crate::types::DelegatedSplit {
                    messages: self.request.messages.clone(),
                    tools: self.request.tools.clone(),
                }),
            ),
        };
        let msg = crate::types::SwarmMessage::RemoteGenerateRequest(RemoteGenerateRequest {
            request_id: wire_id,
            model_id: segment.shard_id.model_id.clone(),
            layer_range,
            prompt,
            sampling,
            session_id: self.request.session_id.clone(),
            delegation,
            sender_peer_bytes: None,
        });
        if self
            .network_tx
            .send(NetworkCommand::SendDirectMessage {
                target_peer_bytes: target_peer_bytes.clone(),
                message: msg,
                // Opt into ACK-timeout tracking. If libp2p rr silently drops
                // the request (observed under load), the daemon closes
                // streaming_token_txs[request_id] within RR_ACK_TIMEOUT_SECS
                // (10s) so we fail fast instead of waiting 120s.
                delivery_request_id: Some(wire_id),
            })
            .await
            .is_err()
        {
            self.shared_state.streaming_token_txs.remove(&wire_id);
            return Err(SwarmError::Network(
                "RemoteGenerateRequest send dropped".into(),
            ));
        }

        tracing::info!(
            %request_id,
            %wire_id,
            target = %segment.node_id,
            ?hand_off,
            first_token_budget_s = first_token_budget.as_secs(),
            cold_loads = load.cold_loads(),
            "remote-generate fast path: request sent"
        );

        // Collect streamed tokens. First token gets a longer timeout
        // (prefill time); subsequent tokens get the inter-token timeout.
        let mut content = String::new();
        let mut finish_reason = String::new();
        let mut prompt_tokens = 0u32;
        let mut completion_tokens = 0u32;
        let mut matched_stop_seq: Option<String> = None;
        let mut token_logprobs: Vec<swarmllm_types::TokenLogProbEntry> = Vec::new();
        let mut first = true;
        // Timing for the delegated-speed observation recorded at the end.
        //
        // This path measured NOTHING until 2026-08-18, and it is the path most
        // single-model requests take — so a node whose traffic all went this
        // way never learned a thing about its peers and only ever had figures
        // gossiped to it second-hand. That blindness is what made the routing
        // decision this feeds a guess rather than a measurement.
        let sent_at = std::time::Instant::now();
        // Absolute, so the slices the wait is taken in cannot restart it.
        let first_token_deadline = sent_at + first_token_budget;
        let mut first_token_at: Option<std::time::Instant> = None;

        // Reassembly of an unordered stream — see `StreamReassembler`.
        let mut stream = StreamReassembler::new();
        // A hole in that stream can be FILLED if the peer keeps what it sent
        // (`features::RESEND_TOKENS`); before that existed one lost token
        // cost the whole tail of the reply (gotcha #438).
        let mut resend = ResendPolicy::new(
            self.shared_state
                .peer_advertises_feature(&segment.node_id, crate::types::features::RESEND_TOKENS),
        );
        let hole_wait = hole_wait(
            self.shared_state
                .peer_registry
                .get(&segment.node_id)
                .and_then(|p| p.latency_ms),
        );
        // When the current hole opened. An ask is made once it has stood for
        // `hole_wait`, whether or not later tokens keep arriving behind it —
        // measured with a token dropped on purpose, waiting for silence meant
        // the ask came only after the whole rest of the reply had streamed
        // in, ten seconds after the loss.
        let mut hole_since: Option<std::time::Instant> = None;

        loop {
            // Honor external cancel — same pattern as execute_distributed
            // line 174. Without this, a cancelled request keeps draining
            // tokens until INTER_TOKEN_TIMEOUT (60s) per gap.
            if self.request.is_cancelled() {
                tracing::info!(
                    %request_id,
                    "DIAG: remote-generate cancelled externally"
                );
                // Tell the remote to stop streaming wasted tokens too. Best
                // effort — if the send drops, the remote will hit its own
                // timeout/EOS naturally.
                let _ = self
                    .network_tx
                    .send(NetworkCommand::SendDirectMessage {
                        target_peer_bytes: target_peer_bytes.clone(),
                        message: crate::types::SwarmMessage::CancelInference(
                            swarmllm_types::CancelInference {
                                request_id: wire_id,
                            },
                        ),
                        delivery_request_id: None,
                    })
                    .await;
                self.shared_state.streaming_token_txs.remove(&wire_id);
                finish_reason = "stop".to_string();
                break;
            }
            if stream.has_hole() {
                let since = *hole_since.get_or_insert_with(std::time::Instant::now);
                if since.elapsed() >= hole_wait && resend.can_ask() {
                    if let Some((from, to)) = stream.resend_range() {
                        resend.ask();
                        hole_since = Some(std::time::Instant::now());
                        tracing::info!(
                            %request_id,
                            peer = %segment.node_id,
                            from,
                            to,
                            ask = resend.asks(),
                            emitted = stream.emitted(),
                            buffered = stream.buffered(),
                            "DIAG: remote-generate: tokens missing — asking the peer to resend"
                        );
                        let _ = self
                            .network_tx
                            .send(NetworkCommand::SendDirectMessage {
                                target_peer_bytes: target_peer_bytes.clone(),
                                message: crate::types::SwarmMessage::ResendTokens(
                                    swarmllm_types::ResendTokens {
                                        request_id: wire_id,
                                        from_token_id: from,
                                        to_token_id: to,
                                    },
                                ),
                                delivery_request_id: None,
                            })
                            .await;
                    }
                }
            } else {
                hole_since = None;
            }
            let timeout_dur = if stream.has_hole() && resend.can_ask() {
                hole_wait
            } else if stream.done_seen() {
                STRAGGLER_TIMEOUT
            } else if first {
                // In slices, so a peer that has left is noticed in seconds
                // rather than at the end of a budget that may hold a cold load.
                match first_token_wait(std::time::Instant::now(), first_token_deadline, true) {
                    FirstTokenWait::Again(slice) => slice,
                    FirstTokenWait::PeerLeft | FirstTokenWait::Expired => Duration::ZERO,
                }
            } else {
                inter_token_timeout
            };
            // Watched, not merely checked beforehand. The first-token budget
            // is prompt-scaled and reaches ten minutes, so a client that gave
            // up during it went unnoticed for that whole time while the peer
            // generated a reply nobody would read — the same shape as gotchas
            // #445 and #459, on the path a peer-served whole model takes. A
            // cancel here goes round the loop and the check above does the
            // real work: tell the peer, drop the channel, stop.
            let maybe = match crate::inference::cancel::unless_cancelled(
                async { Ok(tokio::time::timeout(timeout_dur, stream_rx.recv()).await) },
                self.request.cancel.as_ref(),
            )
            .await
            {
                Ok(m) => m,
                Err(_) => continue,
            };
            // A wait on a hole ran out in silence: go round and ask (the check
            // at the top of the loop does it, since the hole has now stood
            // for at least `hole_wait`), rather than treating a lost send as
            // the end of the reply. Bounded by the policy; a peer that never
            // answers still reaches the deadlines below.
            if maybe.is_err() && resend.can_ask() && stream.resend_range().is_some() {
                continue;
            }
            // A slice of the first-token wait ran out: wait again while the
            // budget lasts and the peer is still connected to us.
            let mut peer_left = false;
            if maybe.is_err() && first {
                match first_token_wait(
                    std::time::Instant::now(),
                    first_token_deadline,
                    self.shared_state
                        .connected_node_ids
                        .contains(&segment.node_id),
                ) {
                    FirstTokenWait::Again(_) => continue,
                    FirstTokenWait::PeerLeft => peer_left = true,
                    FirstTokenWait::Expired => {}
                }
            }
            let tok = match maybe {
                Ok(Some(t)) => t,
                Ok(None) => {
                    // Channel closed by the daemon's ACK-timeout sweep
                    // (libp2p rr silent-drop) or by an OutboundFailure event.
                    // If nothing was EMITTED, surface as an explicit error so
                    // the caller can retry; otherwise treat as graceful
                    // end-of-stream and return what we have.
                    //
                    // **Emitted, not "a token arrived".** Since tokens are
                    // reassembled by `token_id`, one can arrive and be BUFFERED
                    // rather than released — a reply whose first token is lost
                    // but whose later tokens land has `first == false` and
                    // nothing to show. Keying off arrival therefore returned an
                    // empty reply as a SUCCESS: billed, no error, no retry.
                    // Reported 2026-08-11 as intermittent empty answers on a
                    // node that routes remotely, ~50% of calls, 35-39s each —
                    // the delay being this loop waiting for a token that never
                    // came. Regression from the reassembly fix (#282), which
                    // made arrival and emission different events.
                    if stream.emitted() == 0 {
                        tracing::warn!(%request_id, "remote-generate: token channel closed before any token (likely send failure)");
                        // `PeerUnresponsive`, not `PipelineError`: the peer
                        // went quiet, so the caller gets a 503 that invites
                        // the retry that helps, and the peer is eligible for
                        // the serve-failure penalty (`PipelineError` is
                        // exempt as a local scheduling problem).
                        self.shared_state.record_peer_delivery(
                            &segment.node_id,
                            Some(&segment.shard_id.model_id),
                            false,
                        );
                        // The retry this invites re-plans; without this it
                        // can re-pick the peer that just went quiet and wait
                        // the same silence out again (Envoy's `previous_hosts`
                        // rule). Scoped to this request only. A delegate quiet
                        // before any token is instead run as a segment of the
                        // plan this node already holds (`try_delegated_split`),
                        // where its silence is the segment deadline's to judge.
                        if whole_model || self.hand_off_emitted > 0 {
                            self.shared_state
                                .blacklist_holder_for_request(request_id, &segment.node_id);
                        }
                        return Err(SwarmError::PeerUnresponsive(format!(
                            "remote-generate: peer never acknowledged request_id={request_id} (silent drop or disconnect)"
                        )));
                    }
                    tracing::warn!(%request_id, "remote-generate: token channel closed mid-stream");
                    break;
                }
                Err(_) => {
                    // Waiting on stragglers after a done token is not a failed
                    // request — the answer is complete up to the gap, and the
                    // caller is better served by the prefix than by an error.
                    // Only the consecutive run is kept: emitting past a hole
                    // would silently reorder the reply, which is worse than a
                    // short one.
                    // ...but "the prefix" has to BE something. When the hole is
                    // at the very start there is no prefix, only an empty reply
                    // that would be returned as a success and charged for. That
                    // is a failed request and must say so.
                    if stream.done_seen() && stream.emitted() > 0 {
                        tracing::warn!(
                            %request_id,
                            peer = %segment.node_id,
                            emitted = stream.emitted(),
                            missing = stream.missing(),
                            buffered = stream.buffered(),
                            resend_asks = resend.asks(),
                            "remote-generate: gave up on tokens that never arrived — returning what did"
                        );
                        break;
                    }
                    self.shared_state.streaming_token_txs.remove(&wire_id);
                    // A delivery that yielded nothing usable. Recorded so the
                    // router learns to prefer a peer whose answers arrive —
                    // this is also how the peer's own terminal ERROR frame
                    // going missing shows up, since that frame is a single
                    // unacknowledged send like any other token. A tester saw
                    // exactly that: a precise "needs more memory than my
                    // budget allows" refusal, produced immediately and twice,
                    // reaching the caller as a 143-second silence.
                    self.shared_state.record_peer_delivery(
                        &segment.node_id,
                        Some(&segment.shard_id.model_id),
                        false,
                    );
                    // Same reclassification as the never-acknowledged arm
                    // above: the peer went quiet past its deadline — and the
                    // same bar from this request's retry. A delegate quiet
                    // before any token is not barred: the same plan runs here.
                    if whole_model || self.hand_off_emitted > 0 {
                        self.shared_state
                            .blacklist_holder_for_request(request_id, &segment.node_id);
                    }
                    if peer_left {
                        tracing::warn!(
                            %request_id,
                            peer = %segment.node_id,
                            waited_ms = sent_at.elapsed().as_millis() as u64,
                            "remote-generate: the peer disconnected before its first token — \
                             it has dropped the request, so retrying now"
                        );
                        return Err(SwarmError::PeerUnresponsive(
                            "remote-generate: the peer disconnected before its first token".into(),
                        ));
                    }
                    return Err(SwarmError::PeerUnresponsive(format!(
                        "remote-generate timed out waiting for token (first={first})"
                    )));
                }
            };
            if first {
                first_token_at = Some(std::time::Instant::now());
            }
            first = false;

            if let Some(ref reason) = tok.finish_reason {
                finish_reason = match reason {
                    NetworkFinishReason::Stop => "stop".to_string(),
                    NetworkFinishReason::MaxTokens => "length".to_string(),
                    NetworkFinishReason::Error(e) if !whole_model => {
                        // A delegate's error may name a peer IT chose — a
                        // missing shard, a refusal, a context limit — so
                        // nothing here is evidence about the delegate's own
                        // holdings or limits: retract nothing, bar nobody.
                        // Before any token, `try_delegated_split` runs the plan
                        // here; after, it is the request's failure.
                        self.shared_state.streaming_token_txs.remove(&wire_id);
                        return Err(crate::error::reclassify_flattened_error(e)
                            .unwrap_or_else(|| SwarmError::Inference(e.clone())));
                    }
                    NetworkFinishReason::Error(e) => {
                        // Same stale-claim retraction as the multi-segment path.
                        // This fast path has no failover, so without it a peer
                        // whose shard set shrank keeps being chosen and every
                        // request fails until its retraction gossip arrives —
                        // which is exactly the case reported on 2026-07-26.
                        if super::remote_error_means_missing_shard(e) {
                            self.shared_state.retract_shard_holder_claims_for_range(
                                &segment.shard_id.model_id,
                                &segment.node_id,
                                segment.layer_range,
                                "remote reported the shard data as missing",
                            );
                            // Make the retraction stick for the retry: the DHT still
                            // advertises this holder, so the next assembly would
                            // otherwise re-learn the claim and pick it again.
                            self.shared_state
                                .blacklist_holder_for_request(request_id, &segment.node_id);
                        } else if crate::inference::router::message_means_peer_cannot_serve(e) {
                            // The peer holds the shards but cannot run them — its
                            // worker failed to start, died, or dropped the
                            // connection. Retracting its claims would be wrong,
                            // the data really is there; but the retry must not
                            // come straight back to it. Without this the retry
                            // re-picked the same broken peer and the request
                            // failed twice: observed 2026-07-27 as `assemblies=2`
                            // with the same node id on both attempts, after a
                            // node whose binary had been replaced underneath it
                            // lost the ability to start any worker at all.
                            self.shared_state
                                .blacklist_holder_for_request(request_id, &segment.node_id);
                        }
                        self.shared_state.streaming_token_txs.remove(&wire_id);
                        // A conversation longer than THIS peer serves is its
                        // limit, not the request's, unless it is the model's own
                        // (#111): bar it and let the router re-plan, as a
                        // segment would fail over.
                        if let Some(err) = self.longer_than_this_peer_serves(e, &segment) {
                            return Err(err);
                        }
                        // A peer's error arrives as text, so its class is
                        // gone unless we recover it. Without this a prompt too
                        // long for a peer-held model answered 500
                        // `server_error` while the identical request on a local
                        // model answered 400 — and, because the peer is blamed
                        // for anything that is not a local-only variant, it was
                        // also docked for a mistake that was the caller's.
                        return Err(crate::error::reclassify_flattened_error(e)
                            .unwrap_or_else(|| SwarmError::Inference(e.clone())));
                    }
                };
                if let Some(usage) = tok.usage {
                    prompt_tokens = usage.prompt_tokens;
                    completion_tokens = usage.completion_tokens;
                }
                if let Some(ms) = tok.matched_stop_sequence.clone() {
                    matched_stop_seq = Some(ms);
                }
                if let Some(lp) = tok.logprob.clone() {
                    token_logprobs.push(lp);
                }
                // The done token says how many content tokens were sent. If any
                // are still in flight, keep waiting rather than discarding
                // them — stopping here is exactly the truncation this solves.
                stream.mark_done(tok.token_id);
                if !stream.is_complete() {
                    tracing::debug!(
                        %request_id,
                        emitted = stream.emitted(),
                        missing = stream.missing(),
                        "remote-generate: done arrived before all tokens — waiting for stragglers"
                    );
                    continue;
                }
                if let Some(ref tx) = token_tx {
                    let _ = tx
                        .send(StreamingTokenEvent {
                            text: String::new(),
                            finish_reason: Some(finish_reason.clone()),
                            matched_stop_sequence: matched_stop_seq.clone(),
                        })
                        .await;
                }
                break;
            }

            // Content token. Buffer by sequence and emit only the consecutive
            // run starting at `next_seq`, so the text the user sees is in
            // generation order even when the network delivers out of order.
            // An unsequenced server has arrival order as its only order, so its
            // tokens slot in at `next_seq` and emit immediately — unchanged
            // behaviour.
            let mut client_gone = false;
            for t in stream.push_content(tok) {
                if let Some(lp) = t.logprob.clone() {
                    token_logprobs.push(lp);
                }
                if t.text.is_empty() {
                    continue;
                }
                self.hand_off_emitted += 1;
                content.push_str(&t.text);
                if let Some(ref tx) = token_tx {
                    if tx
                        .send(StreamingTokenEvent {
                            text: t.text,
                            finish_reason: None,
                            matched_stop_sequence: None,
                        })
                        .await
                        .is_err()
                    {
                        client_gone = true;
                        break;
                    }
                }
            }

            if client_gone {
                tracing::info!(
                    %request_id,
                    "remote-generate: client disconnected — sending CancelInference"
                );
                // Tell the remote to stop its decode immediately so it
                // doesn't keep streaming tokens we'll discard.
                let _ = self
                    .network_tx
                    .send(NetworkCommand::SendDirectMessage {
                        target_peer_bytes: target_peer_bytes.clone(),
                        message: crate::types::SwarmMessage::CancelInference(
                            swarmllm_types::CancelInference {
                                request_id: wire_id,
                            },
                        ),
                        delivery_request_id: None,
                    })
                    .await;
                finish_reason = "stop".to_string();
                break;
            }

            // A done token that arrived early completes the request once the
            // tokens it was waiting on have landed.
            if stream.is_complete() {
                if let Some(ref tx) = token_tx {
                    let _ = tx
                        .send(StreamingTokenEvent {
                            text: String::new(),
                            finish_reason: Some(finish_reason.clone()),
                            matched_stop_sequence: matched_stop_seq.clone(),
                        })
                        .await;
                }
                break;
            }
        }

        self.shared_state.streaming_token_txs.remove(&wire_id);

        if finish_reason.is_empty() {
            finish_reason = "stop".to_string();
        }

        // Report the tokens the caller actually RECEIVED, not the count the peer
        // says it generated.
        //
        // `usage` rides on the done token, which always arrives, while content
        // tokens can be lost — so a truncated reply reported the full count
        // beside a short answer. That mismatch is precisely the signal used to
        // diagnose the truncation bug in the first place (#282): a token count
        // disagreeing with the text means tokens went missing. Passing it on as
        // truth hands clients the same misleading figure, and it is also what
        // the request is BILLED on — settlement multiplies this number — so a
        // reply that lost half its tokens was charged in full for them.
        //
        // Clamping down only. A peer under-reporting is its own problem and not
        // one to paper over by inventing usage the caller cannot see.
        let delivered = stream.emitted();
        // Tokens went missing in transit. Remembered because it also disqualifies
        // this request as a speed measurement — see the sample below.
        let stream_was_truncated = stream.truncated();
        // What the peer says it PUT ON THE WIRE, which is what `delivered` is
        // comparable to. Not `completion_tokens` — see `truncated()`.
        let claimed_by_peer = stream.sent_by_peer();
        if stream_was_truncated {
            tracing::warn!(
                %request_id,
                peer = %segment.node_id,
                claimed = claimed_by_peer,
                delivered,
                generated = completion_tokens,
                "remote-generate: tokens the peer sent did not all arrive — \
                 reporting and billing the delivered count"
            );
            completion_tokens = delivered;
        }

        // What this peer is actually worth when handed a whole model.
        //
        // Deliberately EXCLUDES the time to the first token: that is prefill
        // plus any model load and queueing on the peer, and folding it in would
        // make a peer look slow for being cold. What is left is the steady
        // decode rate, divided by the tokens it produced and the layers it ran
        // — the same shape as every other sample, so it composes with them.
        //
        // Needs at least two tokens: with one, `total - ttft` is zero and says
        // nothing about decode speed at all.
        // A truncated stream is not a speed measurement, and taking one is worse
        // than taking none.
        //
        // Both halves of the arithmetic are corrupted, in the same direction.
        // The elapsed time includes the deadline we spent waiting for tokens
        // that never came, and the token count is the truncated one we clamped
        // to just above — so a division meant to yield "ms per token of steady
        // decoding" instead yields the give-up timeout divided by the handful
        // that survived.
        //
        // Measured live 2026-08-20: three requests to an RTX 4050 lost their
        // tokens in transit and returned 3, 3 and 18 of 60. That recorded the
        // card at **345 ms/layer** — against the 3.1 ms/layer the same peer had
        // measured minutes earlier on the same model — and since a delegated
        // observation outranks a peer's advertised speed, it demoted the only
        // GPU in the swarm behind a laptop CPU for the ten minutes the figure
        // takes to expire. A transport failure was being recorded as a fact
        // about the hardware.
        //
        // Same family as the cold-sample rule above it: a cold sample is a load
        // time wearing a compute figure's clothes, and this is a timeout wearing
        // the same disguise.
        // Reliability of the PATH to this peer, recorded whether or not the
        // reply survived — a figure built only from intact replies measures
        // nothing. This is what steers traffic away from a lossy link now that
        // truncation no longer (wrongly) does so through the speed EMA.
        self.shared_state.record_peer_delivery(
            &segment.node_id,
            Some(&segment.shard_id.model_id),
            !stream_was_truncated,
        );
        if !stream_was_truncated {
            // The peer saw this request through: whatever it failed before is
            // forgiven (`peer_outliers`).
            self.shared_state
                .note_peer_completed_request(&segment.node_id, &segment.shard_id.model_id);
        }

        // What arrived is not the reply that was generated, so it must not be
        // handed over as one.
        //
        // This used to return the surviving prefix with `finish_reason: "stop"`
        // — the model chose to stop after three tokens — which a client cannot
        // distinguish from a real answer. Measured 2026-08-20: 3, 3 and 18
        // tokens of 60, each reported as a clean completion.
        //
        // Safe to raise here in both modes, and the placement is the reason.
        // The give-up arm breaks WITHOUT sending a terminal event to the
        // caller's token channel, so nothing has been finished yet: a
        // non-streaming caller gets an error instead of a short answer, and a
        // streaming one gets the partial content it already received followed
        // by an explicit error frame rather than a false ending. The variant is
        // deliberately absent from `is_transient_remote_failure`, because that
        // retry reuses this same token channel and would emit the reply twice.
        if stream_was_truncated {
            return Err(SwarmError::ReplyTruncated(format!(
                "{} of {} tokens arrived from {}",
                delivered, claimed_by_peer, segment.node_id
            )));
        }

        // A delegated split's pace is its PLAN's — the delegate and every peer
        // it chose — and recorded as the delegate's it would misprice it.
        if let Some(ttft) = first_token_at.filter(|_| whole_model) {
            let steady = sent_at
                .elapsed()
                .saturating_sub(ttft.duration_since(sent_at));
            let layers = segment.layer_range.1.saturating_sub(segment.layer_range.0);
            if delegated_sample_is_usable(stream_was_truncated, completion_tokens, layers) {
                let per_token_ms = steady.as_millis() as u64 / u64::from(completion_tokens - 1);
                self.shared_state.record_peer_segment_latency(
                    &segment.node_id,
                    &self.request.model_id,
                    crate::daemon::state::WorkKind::Delegated,
                    per_token_ms,
                    layers,
                    0,
                );
            }
        }

        // Reply text is finalised in exactly one place — see
        // `finalize_reply_text` — and this path reached the user without ever
        // calling it. A whole model served by ONE peer is the commonest
        // distributed shape there is, so a reasoning model asked over the swarm
        // answered with its raw `<think>` scratchpad while the identical request
        // answered locally came back clean. Verified 2026-09-17 against this
        // node's OWN peer on the same build, which rules out an older peer:
        // asked directly it stripped the block, asked as a peer it did not.
        //
        // Finalising HERE rather than on the serving side is deliberate: the
        // coordinator is the only place that covers every peer, including ones
        // running a build that never learned to strip anything. The helper is
        // documented idempotent, so a peer that already finalised loses nothing.
        //
        // Empty stops, and that is the whole point of passing them: the peer
        // generated the text and already applied both the caller's stop
        // sequences and its own template's, reporting the result in
        // `matched_stop_seq`. Re-running that decision against a stop set this
        // node derived for a model it may not even hold could truncate a reply
        // the peer correctly kept. What is left — the control-token scrub, the
        // leading reasoning block, the stranded newlines — is the part no peer
        // can have done on our behalf.
        crate::inference::finalize_reply_text(&mut content, &[]);

        crate::inference::report_short_reply(
            &request_id,
            completion_tokens,
            self.request.sampling_params.max_tokens,
            matched_stop_seq.as_deref(),
        );
        Ok(Some(InferenceOutput {
            request_id,
            content,
            prompt_tokens,
            completion_tokens,
            finish_reason,
            session_id: self.request.session_id.clone(),
            token_logprobs,
            // Captured from the terminal StreamingToken above; the remote
            // worker carries the user-provided matched sequence on the
            // final token so the API layer can surface it to Anthropic
            // clients.
            matched_stop_sequence: matched_stop_seq,
            trace: None,
        }))
    }
}

#[cfg(test)]
mod first_token_budget_tests {
    use super::*;

    fn warm() -> super::super::LoadAllowance {
        super::super::LoadAllowance::none()
    }

    /// The cold load is added ON TOP of the prompt's cap, as on the segment
    /// path: a load is not prompt work.
    #[test]
    fn a_cold_load_is_added_beyond_the_prompt_ceiling() {
        use crate::inference::pipeline::LoadAllowance;
        // Two machines to load, via the only constructor a caller can reach.
        let state = test_state();
        let cold = LoadAllowance::for_segments(&state, &[segment([1u8; 32]), segment([2u8; 32])]);
        assert_eq!(cold.cold_loads(), 2);
        assert_eq!(
            first_token_timeout(usize::MAX, cold),
            FIRST_TOKEN_TIMEOUT_MAX + cold.duration()
        );
        assert_eq!(
            first_token_timeout(0, cold),
            FIRST_TOKEN_TIMEOUT + cold.duration()
        );
    }

    #[test]
    fn a_first_token_wait_is_taken_in_slices_that_never_pass_the_deadline() {
        let now = std::time::Instant::now();
        let far = now + Duration::from_secs(300);
        assert_eq!(
            first_token_wait(now, far, true),
            FirstTokenWait::Again(PEER_PRESENCE_CHECK)
        );
        let near = now + Duration::from_secs(2);
        assert_eq!(
            first_token_wait(now, near, true),
            FirstTokenWait::Again(Duration::from_secs(2))
        );
        assert_eq!(first_token_wait(near, near, true), FirstTokenWait::Expired);
    }

    /// The serving node drops the request when its last connection to us
    /// closes, so waiting out a budget that holds a cold load would be minutes
    /// spent on an answer that cannot come.
    #[test]
    fn a_peer_that_left_ends_the_wait_before_its_budget_does() {
        let now = std::time::Instant::now();
        let far = now + Duration::from_secs(372);
        assert_eq!(first_token_wait(now, far, false), FirstTokenWait::PeerLeft);
        // At the deadline the budget is what ran out, whatever the link.
        assert_eq!(first_token_wait(far, far, false), FirstTokenWait::Expired);
    }

    fn segment(node: [u8; 32]) -> crate::types::PipelineSegment {
        crate::types::PipelineSegment {
            node_id: crate::types::NodeId(node),
            shard_id: crate::types::ShardId {
                model_id: crate::types::ModelId("m".into()),
                index: 0,
            },
            layer_range: (0, 32),
        }
    }

    fn test_state() -> std::sync::Arc<crate::daemon::SharedState> {
        use crate::inference::executor::ModelExecutor;
        let temp = tempfile::tempdir().unwrap();
        let db = crate::storage::db::Database::open(temp.path()).unwrap();
        let (state, _, _) = crate::daemon::SharedState::new(
            crate::config::Config::default(),
            crate::identity::Identity::generate(),
            db,
            std::sync::Arc::new(tokio::sync::Mutex::new(ModelExecutor::new())),
            None,
        );
        state
    }

    #[test]
    fn short_prompt_keeps_the_original_budget() {
        // A handful of tokens must not shift the long-standing default.
        assert_eq!(first_token_timeout(0, warm()), FIRST_TOKEN_TIMEOUT);
        assert!(first_token_timeout(10, warm()) < FIRST_TOKEN_TIMEOUT + Duration::from_secs(6));
    }

    /// CJK sits near one token per character, so a divisor tuned for Latin
    /// prose would under-budget it — the case this fallback must not get wrong.
    #[test]
    fn fallback_estimate_does_not_undercount_multibyte_scripts() {
        let cjk = "\u{4f60}\u{597d}\u{4e16}\u{754c}".repeat(50); // 200 chars, ~200 tokens
        assert_eq!(cjk.chars().count(), 200);
        assert!(
            estimate_prompt_tokens(&cjk) >= 100,
            "multi-byte prompt under-counted: {}",
            estimate_prompt_tokens(&cjk)
        );
        // Byte length would have inflated this to ~600; chars keeps it honest.
        assert!(estimate_prompt_tokens(&cjk) <= 200);
    }

    #[test]
    fn fallback_estimate_never_returns_zero_for_a_nonempty_prompt() {
        assert_eq!(estimate_prompt_tokens(""), 0);
        assert!(
            estimate_prompt_tokens("a") >= 1,
            "a prompt must cost >= 1 token"
        );
    }

    #[test]
    fn budget_never_shrinks_below_the_base() {
        for tokens in [0, 1, 10, 100, 1_000, 100_000] {
            assert!(
                first_token_timeout(tokens, warm()) >= FIRST_TOKEN_TIMEOUT,
                "prompt of {tokens} tokens shortened the budget"
            );
        }
    }

    /// The measured live failure: a 613-token prompt needed 285s of prefill on
    /// a 6-core CPU node and was cut off by the flat 120s budget.
    #[test]
    fn covers_the_prompt_that_timed_out_live() {
        let budget = first_token_timeout(613, warm());
        assert!(
            budget > Duration::from_secs(285),
            "budget {budget:?} still cuts off the prompt that measured 285s"
        );
    }

    /// A second live measurement, on a longer prompt: 1322 tokens took 319s.
    #[test]
    fn covers_the_longer_measured_prompt() {
        let budget = first_token_timeout(1322, warm());
        assert!(
            budget > Duration::from_secs(319),
            "budget {budget:?} cuts off a prompt measured at 319s"
        );
    }

    #[test]
    fn budget_is_monotonic_in_prompt_length() {
        let mut prev = Duration::ZERO;
        for tokens in [0, 100, 500, 1_000, 5_000, 50_000] {
            let b = first_token_timeout(tokens, warm());
            assert!(b >= prev, "budget went backwards at {tokens} tokens");
            prev = b;
        }
    }

    #[test]
    fn absurd_prompt_is_capped_not_overflowed() {
        // A dead peer must still be detected in bounded time.
        assert_eq!(
            first_token_timeout(usize::MAX, warm()),
            FIRST_TOKEN_TIMEOUT_MAX
        );
        assert_eq!(
            first_token_timeout(10_000_000, warm()),
            FIRST_TOKEN_TIMEOUT_MAX
        );
    }
}

#[cfg(test)]
mod stream_reassembly_tests {
    use super::{
        hole_wait, ResendPolicy, StreamReassembler, HOLE_WAIT_MAX, HOLE_WAIT_MIN, MAX_RESEND_ASKS,
    };
    use std::time::Duration;
    use swarmllm_types::StreamingToken;

    fn content(id: u32, text: &str) -> StreamingToken {
        StreamingToken {
            request_id: uuid::Uuid::nil(),
            token_id: id,
            finish_reason: None,
            text: text.to_string(),
            usage: None,
            matched_stop_sequence: None,
            logprob: None,
        }
    }

    fn joined(toks: Vec<StreamingToken>) -> String {
        toks.into_iter().map(|t| t.text).collect()
    }

    /// The measured failure this oracle exists for. One emoji is several
    /// generated tokens and ONE sent token, because `decode_token` returns
    /// empty until the multi-byte codepoint completes and the serving node
    /// skips forwarding an empty-text event. Judging arrivals against
    /// `usage.completion_tokens` therefore called a perfectly delivered reply
    /// truncated: on the released v0.3.135, five emoji came back as
    /// `503 Reply truncated in transit: 9 of 12 tokens arrived` (gotcha #416).
    #[test]
    fn a_multi_byte_reply_is_not_truncated_just_because_it_generated_more_tokens() {
        let mut s = StreamReassembler::new();
        // What actually goes on the wire for "🚀 ⭐ 🔥 ❤️ 🌕": one send per
        // completed codepoint, densely numbered.
        for (i, t) in ["🚀", " ⭐", " 🔥", " ❤️", " 🌕"].iter().enumerate() {
            s.push_content(content(i as u32, t));
        }
        // The peer sent 5 content tokens and says so, though it GENERATED 12.
        s.mark_done(5);
        assert_eq!(s.emitted(), 5);
        // The control: the predicate this replaced DOES fire here, so the test
        // is not passing merely because the situation is benign.
        let generated_by_the_model = 12u32;
        assert!(
            generated_by_the_model > s.emitted(),
            "the old `completion_tokens > delivered` check fires on this input"
        );
        assert!(
            !s.truncated(),
            "every token the peer sent arrived — the model generating more that \
             decoded to nothing is not a delivery failure"
        );
        assert_eq!(
            s.sent_by_peer(),
            5,
            "must report what was SENT, not generated"
        );
    }

    /// The control, and the reason the oracle cannot simply be deleted: a real
    /// loss must still be caught. Without this the fix above would read as
    /// "never report truncation", which is the failure mode #282 exists for.
    #[test]
    fn a_genuinely_lost_token_is_still_truncated() {
        let mut s = StreamReassembler::new();
        s.push_content(content(0, "a"));
        s.push_content(content(1, "b"));
        // Token 2 never arrives.
        s.mark_done(4);
        assert!(s.truncated(), "two of four tokens arrived — that is a loss");
        assert_eq!(s.missing(), 2);
        assert_eq!(s.sent_by_peer(), 4);
    }

    // ── Filling a hole (gotcha #438) ──

    /// The shape of the measured failure: a token early in the reply is lost,
    /// later ones arrive and wait behind it. The reassembler must say so and
    /// name exactly the range to ask for.
    #[test]
    fn a_hole_names_the_range_to_ask_for() {
        let mut s = StreamReassembler::new();
        s.push_content(content(0, "a"));
        s.push_content(content(1, "b"));
        assert!(!s.has_hole(), "nothing missing yet");
        assert_eq!(s.resend_range(), None);
        // 2 is lost; 3 and 4 land.
        s.push_content(content(3, "d"));
        s.push_content(content(4, "e"));
        assert!(s.has_hole());
        assert_eq!(s.resend_range(), Some((2, 3)));
        // The done token widens what is known to be missing.
        s.mark_done(7);
        assert_eq!(
            s.resend_range(),
            Some((2, 3)),
            "ask for the hole, not the tail"
        );
        // The resend arrives: the run drains through the buffered tokens.
        let released = s.push_content(content(2, "c"));
        assert_eq!(released.len(), 3);
        assert_eq!(s.emitted(), 5);
        // 5 and 6 were promised and are still missing — a hole with nothing
        // buffered behind it is bounded by the total instead.
        assert!(s.has_hole());
        assert_eq!(s.resend_range(), Some((5, 7)));
    }

    /// A resend that races the original — or an original arriving after its
    /// resend — must not be buffered for ever as a phantom hole.
    #[test]
    fn a_duplicate_of_an_emitted_token_is_ignored() {
        let mut s = StreamReassembler::new();
        s.push_content(content(0, "a"));
        s.push_content(content(1, "b"));
        assert!(s.push_content(content(1, "b")).is_empty());
        assert!(s.push_content(content(0, "a")).is_empty());
        assert_eq!(s.buffered(), 0);
        assert!(!s.has_hole());
        assert_eq!(s.emitted(), 2);
    }

    /// An unsequenced peer has arrival order and nothing else: no holes, no
    /// asks, exactly the behaviour it always had.
    #[test]
    fn an_unsequenced_peer_never_has_a_hole() {
        let mut s = StreamReassembler::new();
        s.push_content(content(0, "a"));
        s.push_content(content(0, "b"));
        s.mark_done(0);
        assert!(!s.has_hole());
        assert_eq!(s.resend_range(), None);
    }

    /// The policy is what bounds the wait: a peer that never answers an ask
    /// runs the budget out and the old deadlines take over.
    #[test]
    fn asks_are_budgeted_and_need_the_peers_support() {
        let mut p = ResendPolicy::new(true);
        for _ in 0..MAX_RESEND_ASKS {
            assert!(p.can_ask());
            assert!(p.ask());
        }
        assert!(!p.can_ask());
        assert!(!p.ask(), "an exhausted budget spends nothing");
        assert_eq!(p.asks(), MAX_RESEND_ASKS);

        let mut old_peer = ResendPolicy::new(false);
        assert!(!old_peer.can_ask());
        assert!(!old_peer.ask());
        assert_eq!(old_peer.asks(), 0);
    }

    /// A few round trips, never less than a second, never more than five.
    #[test]
    fn the_hole_wait_scales_with_the_peer_and_stays_bounded() {
        assert_eq!(hole_wait(None), HOLE_WAIT_MIN);
        assert_eq!(hole_wait(Some(0)), HOLE_WAIT_MIN);
        assert_eq!(hole_wait(Some(300)), Duration::from_millis(1200));
        assert_eq!(hole_wait(Some(10_000)), HOLE_WAIT_MAX);
    }

    /// A peer too old to sequence sends zeros throughout and cannot tell us
    /// what to expect, so it must never be judged truncated — the same
    /// degradation `is_complete` makes for mixed-version swarms.
    #[test]
    fn an_unsequenced_peer_is_never_called_truncated() {
        let mut s = StreamReassembler::new();
        s.push_content(content(0, "a"));
        s.push_content(content(0, "b"));
        s.mark_done(0);
        assert!(!s.truncated());
    }

    /// The measured failure. A peer at ~6s RTT answered "Cherry" as two tokens,
    /// and the done token — a separate request_response send with no ordering
    /// relative to them — overtook both. The coordinator stopped there and
    /// returned "", while `usage` correctly reported 2 completion tokens.
    #[test]
    fn a_done_token_that_overtakes_the_content_does_not_end_the_stream() {
        let mut s = StreamReassembler::new();
        s.mark_done(2);
        assert!(
            !s.is_complete(),
            "the peer said it sent 2 tokens and none have arrived — finishing here \
             is what truncated the reply"
        );
        assert_eq!(s.missing(), 2);

        assert_eq!(joined(s.push_content(content(0, "Ch"))), "Ch");
        assert!(!s.is_complete(), "still one token outstanding");

        assert_eq!(joined(s.push_content(content(1, "erry"))), "erry");
        assert!(s.is_complete(), "both tokens in — now the stream is done");
    }

    /// Content tokens race each other too, so arrival order is not text order.
    #[test]
    fn out_of_order_content_is_emitted_in_generation_order() {
        let mut s = StreamReassembler::new();

        // Token 1 wins the race; nothing may be emitted yet or the reply reads
        // "erryCh".
        assert!(s.push_content(content(1, "erry")).is_empty());
        // Token 0 lands and releases both, in order.
        assert_eq!(joined(s.push_content(content(0, "Ch"))), "Cherry");
    }

    /// A token arriving is NOT the same as a token being shown, and the
    /// difference decides whether a request is a success or a failure.
    ///
    /// This is the state the caller has to distinguish: the peer's later tokens
    /// landed, the FIRST one never did, so the reassembler is holding content it
    /// cannot release. `emitted()` is 0 while tokens have definitely arrived.
    ///
    /// The loop in `stream_remote_tokens` used to decide "did we get anything?"
    /// from whether a token had arrived (`first`), which is true here — so a
    /// closed channel or a straggler timeout took the graceful end-of-stream
    /// path and returned an EMPTY reply as a success. Reported 2026-08-11 as
    /// intermittent empty answers, ~50% of remote calls, each taking 35-39s,
    /// charged for and never refunded. It has to key off `emitted()`.
    #[test]
    fn tokens_can_arrive_while_nothing_can_be_shown() {
        let mut s = StreamReassembler::new();
        s.mark_done(3);

        // Tokens 1 and 2 arrive; token 0 is lost.
        assert!(s.push_content(content(1, "ell")).is_empty());
        assert!(s.push_content(content(2, "o")).is_empty());

        assert_eq!(
            s.emitted(),
            0,
            "nothing can be released while the first token is missing"
        );
        assert_eq!(s.buffered(), 2, "but tokens HAVE arrived");
        assert!(!s.is_complete());
        // `missing()` counts what has not been EMITTED, not what is absent from
        // the wire: 3, even though only token 0 is actually lost and the other
        // two are sitting in the buffer. Worth stating, because reading it as
        // "tokens the peer still owes us" is wrong and this test asserted that
        // first.
        assert_eq!(s.missing(), 3);

        // Emitting only the consecutive run stays correct once the hole fills.
        assert_eq!(joined(s.push_content(content(0, "H"))), "Hello");
        assert_eq!(s.emitted(), 3);
    }

    /// The converse, so the guard cannot be "always error on a gap": a reply
    /// that lost a token in the MIDDLE has a real prefix, and the caller is
    /// better served by it than by an error.
    #[test]
    fn a_hole_after_the_start_still_leaves_something_to_return() {
        let mut s = StreamReassembler::new();
        s.mark_done(3);
        assert_eq!(joined(s.push_content(content(0, "Hi"))), "Hi");
        assert!(s.push_content(content(2, "!")).is_empty());
        assert_eq!(s.emitted(), 1, "the prefix is real and worth returning");
        assert!(!s.is_complete());
    }

    /// The count reported to the caller must describe what they RECEIVED.
    ///
    /// `usage` rides on the done token, which always arrives; content tokens
    /// can be lost. So a reply that lost tokens carried the peer's full count
    /// next to a short answer — the exact disagreement that identifies a
    /// delivery failure (#282), handed to clients as fact and, worse, used as
    /// the quantity the request is billed on.
    ///
    /// `emitted()` is the honest number and is what the caller now sees.
    #[test]
    fn delivered_token_count_is_what_the_caller_received() {
        let mut s = StreamReassembler::new();
        s.mark_done(4);
        assert_eq!(joined(s.push_content(content(0, "a"))), "a");
        assert_eq!(joined(s.push_content(content(1, "b"))), "b");
        // Tokens 2 and 3 never arrive.
        assert_eq!(
            s.emitted(),
            2,
            "two tokens reached the caller; the peer claims four"
        );
        assert!(!s.is_complete(), "the reply is short and known to be short");
    }

    /// ...and a complete reply must not be clamped: emitted equals claimed, so
    /// the guard is invisible on the normal path.
    #[test]
    fn a_complete_reply_reports_every_token_it_generated() {
        let mut s = StreamReassembler::new();
        s.mark_done(3);
        for (i, t) in [(0, "x"), (1, "y"), (2, "z")] {
            assert!(!s.push_content(content(i, t)).is_empty());
        }
        assert!(s.is_complete());
        assert_eq!(s.emitted(), 3, "nothing to clamp on a complete reply");
    }

    /// A late token releases everything buffered behind it in one go.
    #[test]
    fn a_single_late_token_releases_the_whole_run_behind_it() {
        let mut s = StreamReassembler::new();
        assert!(s.push_content(content(3, "d")).is_empty());
        assert!(s.push_content(content(1, "b")).is_empty());
        assert!(s.push_content(content(2, "c")).is_empty());
        assert_eq!(joined(s.push_content(content(0, "a"))), "abcd");
    }

    /// A peer on an older build sends `token_id: 0` for every token. It cannot
    /// say what to wait for, so arrival order is the only order available and
    /// its done token must finish the stream immediately — the behaviour that
    /// shipped before sequencing existed.
    #[test]
    fn an_unsequenced_peer_still_streams_in_arrival_order_and_completes() {
        let mut s = StreamReassembler::new();
        assert_eq!(joined(s.push_content(content(0, "Ch"))), "Ch");
        assert_eq!(joined(s.push_content(content(0, "erry"))), "erry");

        s.mark_done(0);
        assert!(
            s.is_complete(),
            "an old peer reports 0; waiting for tokens it will never number \
             would hang every request it serves"
        );
    }

    /// The first token legitimately carries id 0, so a stream is only known to
    /// be sequenced once something non-zero shows up — including the done
    /// token, which is what identifies a modern peer that sent exactly one.
    #[test]
    fn a_single_token_reply_is_recognised_as_sequenced_by_its_done_token() {
        let mut s = StreamReassembler::new();
        assert_eq!(joined(s.push_content(content(0, "Cherry"))), "Cherry");
        s.mark_done(1);
        assert!(s.is_complete());
        assert_eq!(s.missing(), 0);
    }

    /// A modern peer that generated nothing at all still terminates.
    #[test]
    fn an_empty_reply_completes() {
        let mut s = StreamReassembler::new();
        s.mark_done(0);
        assert!(s.is_complete());
    }

    /// A duplicate or already-emitted id must not stall the stream. It lands in
    /// the buffer at a slot the drain has passed and stays there until the
    /// request ends — bounded by one reply's token count — while completion
    /// still turns on `next_seq`, so nothing waits on it.
    #[test]
    fn a_late_duplicate_does_not_stall_completion() {
        let mut s = StreamReassembler::new();
        assert_eq!(joined(s.push_content(content(0, "a"))), "a");
        assert_eq!(joined(s.push_content(content(1, "b"))), "b");
        // Token 0 again, after both have been emitted.
        assert!(s.push_content(content(0, "a")).is_empty());
        s.mark_done(2);
        assert!(
            s.is_complete(),
            "a re-delivered token must not hold the stream open"
        );
    }

    /// Losing a token must not reorder what survives: the consecutive prefix is
    /// returned and the tokens stranded behind the gap are reported, not
    /// spliced in.
    #[test]
    fn a_permanent_gap_leaves_a_prefix_rather_than_a_scrambled_reply() {
        let mut s = StreamReassembler::new();
        assert_eq!(joined(s.push_content(content(0, "a"))), "a");
        assert!(s.push_content(content(2, "c")).is_empty());
        s.mark_done(3);

        assert!(!s.is_complete());
        assert_eq!(s.emitted(), 1);
        assert_eq!(s.missing(), 2);
        assert_eq!(
            s.buffered(),
            1,
            "token 2 is held, never emitted out of order"
        );
    }

    /// A reply that lost tokens in transit must not set the peer's speed. This
    /// is the fix for a live incident: a transport failure recorded an RTX 4050
    /// at 345 ms/layer against its real 3.1, which demoted the swarm's only GPU
    /// behind a laptop CPU until the figure expired.
    #[test]
    fn a_truncated_stream_is_not_a_speed_measurement() {
        assert!(
            !super::delegated_sample_is_usable(true, 60, 16),
            "a stream that lost tokens must not be measured, however many arrived"
        );
        assert!(
            !super::delegated_sample_is_usable(true, 3, 16),
            "the badly truncated case is the one that poisoned the EMA"
        );
        assert!(
            super::delegated_sample_is_usable(false, 60, 16),
            "an intact reply is exactly what we want to measure"
        );
    }

    /// The pre-existing guards still hold: one token cannot describe a decode
    /// rate, and a zero-layer segment cannot be normalised.
    #[test]
    fn a_sample_still_needs_more_than_one_token_and_some_layers() {
        assert!(!super::delegated_sample_is_usable(false, 1, 16));
        assert!(!super::delegated_sample_is_usable(false, 0, 16));
        assert!(!super::delegated_sample_is_usable(false, 60, 0));
    }
}
