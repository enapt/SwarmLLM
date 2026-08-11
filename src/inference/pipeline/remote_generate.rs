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

use std::time::Duration;

use crate::error::SwarmError;
use crate::inference::router::{InferenceOutput, StreamingTokenEvent, StreamingTokenTx};
use crate::types::{NetworkCommand, NetworkFinishReason, RemoteGenerateRequest};

use super::PipelineExecutor;

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

    pub(super) fn emitted(&self) -> u32 {
        self.next_seq
    }

    pub(super) fn buffered(&self) -> usize {
        self.pending.len()
    }
}

/// Preconditions for the fast path. All checks are local and cheap.
fn eligible(exec: &PipelineExecutor) -> bool {
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
    let encrypted_for_model = exec
        .shared_state
        .encrypted_pipeline_models
        .get(model_id)
        .map(|r| *r.value())
        .unwrap_or(exec.shared_state.config.inference.encrypted_pipeline);
    // `encrypted_pipeline` forces local embedding (no raw tokens on wire).
    // `local_embedding_privacy` is similar. Both bypass the fast path.
    if encrypted_for_model || exec.shared_state.config.inference.local_embedding_privacy {
        return false;
    }
    true
}

/// Base budget for the first token: connection, model load, and the decode of
/// a short prompt. Ample for 2048-token generations on CPU with a 7B model.
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
pub(crate) fn first_token_timeout(prompt_tokens: usize) -> Duration {
    let tokens = u32::try_from(prompt_tokens).unwrap_or(u32::MAX);
    FIRST_TOKEN_TIMEOUT
        .saturating_add(PREFILL_ALLOWANCE_PER_TOKEN.saturating_mul(tokens))
        .min(FIRST_TOKEN_TIMEOUT_MAX)
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
    pub(super) async fn try_remote_generate_fastpath(
        &mut self,
        token_tx: Option<StreamingTokenTx>,
    ) -> Result<Option<InferenceOutput>, SwarmError> {
        if !eligible(self) {
            return Ok(None);
        }

        let request_id = self.request.id;
        let segment = self.assignment.segments[0].clone();
        let target_peer_bytes = self
            .shared_state
            .resolve_peer_id_bytes(&segment.node_id)
            .ok_or_else(|| {
                SwarmError::Network(format!("No peer_id_bytes for node {}", segment.node_id))
            })?;

        // Build the chat-templated prompt (same as the standard path).
        let prompt = self.build_prompt().await;
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
        let first_token_budget = first_token_timeout(prompt_tokens_est);

        // Register an inbound StreamingToken channel before sending the
        // request so we never miss an early token.
        let (stream_tx, mut stream_rx) = tokio::sync::mpsc::channel::<crate::types::StreamingToken>(
            REMOTE_GENERATE_TOKEN_CHANNEL_CAP,
        );
        self.shared_state
            .streaming_token_txs
            .insert(request_id, stream_tx);

        // Send the RemoteGenerateRequest.
        let msg = crate::types::SwarmMessage::RemoteGenerateRequest(RemoteGenerateRequest {
            request_id,
            model_id: segment.shard_id.model_id.clone(),
            layer_range: segment.layer_range,
            prompt,
            sampling: self.request.sampling_params.clone(),
            session_id: self.request.session_id.clone(),
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
                delivery_request_id: Some(request_id),
            })
            .await
            .is_err()
        {
            self.shared_state.streaming_token_txs.remove(&request_id);
            return Err(SwarmError::Network(
                "RemoteGenerateRequest send dropped".into(),
            ));
        }

        tracing::info!(
            %request_id,
            target = %segment.node_id,
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

        // Reassembly of an unordered stream — see `StreamReassembler`.
        let mut stream = StreamReassembler::new();

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
                            swarmllm_types::CancelInference { request_id },
                        ),
                        delivery_request_id: None,
                    })
                    .await;
                self.shared_state.streaming_token_txs.remove(&request_id);
                finish_reason = "stop".to_string();
                break;
            }
            let timeout_dur = if stream.done_seen() {
                STRAGGLER_TIMEOUT
            } else if first {
                first_token_budget
            } else {
                INTER_TOKEN_TIMEOUT
            };
            let maybe = tokio::time::timeout(timeout_dur, stream_rx.recv()).await;
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
                        return Err(SwarmError::PipelineError(format!(
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
                            emitted = stream.emitted(),
                            missing = stream.missing(),
                            buffered = stream.buffered(),
                            "remote-generate: gave up on tokens that never arrived — returning what did"
                        );
                        break;
                    }
                    self.shared_state.streaming_token_txs.remove(&request_id);
                    return Err(SwarmError::PipelineError(format!(
                        "remote-generate timed out waiting for token (first={first})"
                    )));
                }
            };
            first = false;

            if let Some(ref reason) = tok.finish_reason {
                finish_reason = match reason {
                    NetworkFinishReason::Stop => "stop".to_string(),
                    NetworkFinishReason::MaxTokens => "length".to_string(),
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
                        self.shared_state.streaming_token_txs.remove(&request_id);
                        return Err(SwarmError::Inference(e.clone()));
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
                            swarmllm_types::CancelInference { request_id },
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

        self.shared_state.streaming_token_txs.remove(&request_id);

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
        if completion_tokens > delivered {
            tracing::warn!(
                %request_id,
                claimed = completion_tokens,
                delivered,
                "remote-generate: peer reported more tokens than were delivered — \
                 reporting and billing the delivered count"
            );
            completion_tokens = delivered;
        }

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

    #[test]
    fn short_prompt_keeps_the_original_budget() {
        // A handful of tokens must not shift the long-standing default.
        assert_eq!(first_token_timeout(0), FIRST_TOKEN_TIMEOUT);
        assert!(first_token_timeout(10) < FIRST_TOKEN_TIMEOUT + Duration::from_secs(6));
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
                first_token_timeout(tokens) >= FIRST_TOKEN_TIMEOUT,
                "prompt of {tokens} tokens shortened the budget"
            );
        }
    }

    /// The measured live failure: a 613-token prompt needed 285s of prefill on
    /// a 6-core CPU node and was cut off by the flat 120s budget.
    #[test]
    fn covers_the_prompt_that_timed_out_live() {
        let budget = first_token_timeout(613);
        assert!(
            budget > Duration::from_secs(285),
            "budget {budget:?} still cuts off the prompt that measured 285s"
        );
    }

    /// A second live measurement, on a longer prompt: 1322 tokens took 319s.
    #[test]
    fn covers_the_longer_measured_prompt() {
        let budget = first_token_timeout(1322);
        assert!(
            budget > Duration::from_secs(319),
            "budget {budget:?} cuts off a prompt measured at 319s"
        );
    }

    #[test]
    fn budget_is_monotonic_in_prompt_length() {
        let mut prev = Duration::ZERO;
        for tokens in [0, 100, 500, 1_000, 5_000, 50_000] {
            let b = first_token_timeout(tokens);
            assert!(b >= prev, "budget went backwards at {tokens} tokens");
            prev = b;
        }
    }

    #[test]
    fn absurd_prompt_is_capped_not_overflowed() {
        // A dead peer must still be detected in bounded time.
        assert_eq!(first_token_timeout(usize::MAX), FIRST_TOKEN_TIMEOUT_MAX);
        assert_eq!(first_token_timeout(10_000_000), FIRST_TOKEN_TIMEOUT_MAX);
    }
}

#[cfg(test)]
mod stream_reassembly_tests {
    use super::StreamReassembler;
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
}
