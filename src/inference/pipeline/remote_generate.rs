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
            let timeout_dur = if first {
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
                    // If no tokens arrived yet, surface as an explicit error
                    // so the caller can retry; otherwise treat as graceful
                    // end-of-stream and return what we have.
                    if first {
                        tracing::warn!(%request_id, "remote-generate: token channel closed before any token (likely send failure)");
                        return Err(SwarmError::PipelineError(format!(
                            "remote-generate: peer never acknowledged request_id={request_id} (silent drop or disconnect)"
                        )));
                    }
                    tracing::warn!(%request_id, "remote-generate: token channel closed mid-stream");
                    break;
                }
                Err(_) => {
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

            if let Some(lp) = tok.logprob.clone() {
                token_logprobs.push(lp);
            }

            // Streaming token: append text + forward to SSE client.
            if !tok.text.is_empty() {
                content.push_str(&tok.text);
                if let Some(ref tx) = token_tx {
                    if tx
                        .send(StreamingTokenEvent {
                            text: tok.text,
                            finish_reason: None,
                            matched_stop_sequence: None,
                        })
                        .await
                        .is_err()
                    {
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
                }
            }
        }

        self.shared_state.streaming_token_txs.remove(&request_id);

        if finish_reason.is_empty() {
            finish_reason = "stop".to_string();
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
