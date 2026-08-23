//! Distributed speculative decoding coordinator loop.
//!
//! See `docs/plans/archive/distributed_inference_speedup.md` § Item 2.
//!
//! # Flow
//!
//! 1. Acquire the local draft model (llama-cpp), prefill it with the prompt.
//! 2. Do one normal non-speculative forward to get the first token + prime the
//!    remote KV state. (Without this, the very first spec round has no
//!    `bootstrap` token to carry between "draft generated" and "target verify".)
//! 3. For each round, while under `max_tokens`:
//!    - Draft γ tokens locally (greedy argmax) using the draft model.
//!    - Send a `LayerForward` with `draft_tokens = [last_token, q_1..q_γ]`
//!      (γ+1 tokens) and `spec_logits_requested = true` to the single remote
//!      segment holder. `truncate_kv_to = Some(prev_expected_kv_len)` carries
//!      any fixup from a prior partial-accept round.
//!    - Remote multi-position forward returns γ+1 logit vectors.
//!      `spec_logits[i]` is target's distribution AFTER seeing input position
//!      i — i.e., target's prediction for position (current_pos + i + 1).
//!    - Greedy accept/reject: accept q_i iff argmax(spec_logits[i-1]) == q_i,
//!      stopping at the first mismatch. The bonus token is the argmax of the
//!      distribution at the rejection point (or spec_logits[γ] if all accepted).
//!    - Emit [q_1..q_k, bonus] (k+1 tokens). Resync the draft KV to match
//!      what the target's KV will look like after truncation.
//!    - Record expected KV length for next round's `truncate_kv_to`.
//!
//! # Constraints (MVP)
//!
//! - Single-segment pipeline (one remote peer holds the full model). TP and
//!   multi-hop pipelines fall back to non-speculative.
//! - Greedy only (`temperature == 0`). Non-greedy accept-reject needs draft
//!   probabilities; the existing `speculative::accept_reject` wants per-position
//!   draft distributions which we don't pay to transmit in greedy mode.
//! - Non-pre-embedded, non-vision, non-LoRA requests.
//! - Requires draft model loaded (`config.inference.draft_model_path` set and
//!   `SharedState.draft_executor` holds a loaded model).
//!
//! # Correctness notes
//!
//! Greedy speculative decoding produces output IDENTICAL to greedy
//! non-speculative decoding, by construction (we accept draft tokens only
//! when they match target's argmax at that position). So regressions show up
//! as output divergence on fixed prompts with temperature=0.

use std::sync::Arc;

use crate::daemon::SharedState;
use crate::error::SwarmError;
use crate::inference::router::{InferenceOutput, StreamingTokenEvent, StreamingTokenTx};
use crate::types::{LayerForward, LayerResult, NetworkCommand, NetworkFinishReason, TensorFormat};
use tokio::sync::mpsc;

use super::prompt::CachedDecoder;
use super::PipelineExecutor;

/// Fast-path preconditions for the greedy distributed speculative loop.
fn eligible(exec: &PipelineExecutor) -> bool {
    // Path-specific flag.
    if !exec.shared_state.config.inference.speculative_distributed {
        return false;
    }
    // Common speculative-path baseline (greedy temp, draft model, etc.).
    if !super::speculative_common_eligible(exec) {
        return false;
    }
    // Single segment only — multi-segment is DSD's path (Item 12).
    if exec.assignment.segments.len() != 1 {
        return false;
    }
    // The single segment must be remote. Local-only inference is handled by
    // `execute_local`'s own speculative path.
    if exec.assignment.segments[0].node_id == *exec.shared_state.identity.node_id() {
        return false;
    }
    true
}

impl PipelineExecutor {
    /// Try the greedy distributed speculative path. Returns `Ok(None)` if
    /// preconditions aren't met at runtime (caller falls back to normal
    /// `execute_distributed`). Returns `Ok(Some(_))` on success.
    pub(super) async fn try_speculative_distributed(
        &mut self,
        token_tx: Option<StreamingTokenTx>,
    ) -> Result<Option<InferenceOutput>, SwarmError> {
        if !eligible(self) {
            return Ok(None);
        }

        let request_id = self.request.id;
        let max_tokens = self.request.sampling_params.max_tokens;
        let gamma = self.shared_state.config.inference.speculative_gamma.max(1);

        // Peer / segment info (single-segment path).
        let segment = self.assignment.segments[0].clone();
        let target_peer_bytes = self
            .shared_state
            .resolve_peer_id_bytes(&segment.node_id)
            .ok_or_else(|| {
                SwarmError::Network(format!("No peer_id_bytes for node {}", segment.node_id))
            })?;

        // Prompt + cached EOS/decoder extraction.
        let prompt = self.build_prompt().await;

        // Acquire draft model lock for the entire request.
        let mut draft = self.shared_state.draft_executor.lock().await;
        if !draft.is_loaded() {
            tracing::debug!(%request_id, "speculative: draft model not loaded — falling back");
            return Ok(None);
        }

        // Phase 1: do one normal forward to get the first token and prime the
        // remote KV. After this the remote KV holds [0..N-1] from prefill
        // plus nothing for the generated token (we sample it, don't feed it
        // back yet). The first_token will ride as the "bootstrap" on round 1.
        let prompt_bytes = prompt.as_bytes().to_vec();
        let prompt_byte_len = prompt_bytes.len();
        let (first_token, prompt_token_count, eos_tokens, decoder) = {
            // Register response channel. Cap-checked + RAII-guarded so an
            // early Err propagation from wait_for_result or the empty-tokens
            // check below doesn't leak the pending entry — see
            // PendingLayerResultGuard / gotcha #45.
            let (rx, mut prefill_guard) = super::register_pending_layer_result(
                &self.shared_state.pending_layer_results,
                request_id,
                Some(segment.node_id.clone()),
            )?;

            let forward = LayerForward {
                request_id,
                sequence_num: 0,
                index_pos: 0,
                activations: prompt_bytes,
                format: TensorFormat::FP32,
                model_id: segment.shard_id.model_id.clone(),
                layer_range: segment.layer_range,
                vision_embeddings: None,
                chain: Vec::new(),
                sender_peer_bytes: None,
                tp_meta: None,
                // Unchained: the receiver answers its sender, which is us. Naming
                // ourselves would add the 0x07 reply-to trailer to a frame older
                // peers do not expect.
                requester_node_id: None,
                pre_embedded: false,
                generated_ids: Vec::new(),
                adapter_id: None,
                draft_tokens: Vec::new(),
                spec_logits_requested: false,
                truncate_kv_to: None,
                chunk_meta: None,
            };
            if self
                .network_tx
                .send(NetworkCommand::SendTensor {
                    target_peer_bytes: target_peer_bytes.clone(),
                    forward,
                })
                .await
                .is_err()
            {
                // Drop guard removes the pending entry on return.
                return Err(SwarmError::Network(
                    "Failed to send prefill LayerForward".into(),
                ));
            }
            let num_layers = segment.layer_range.1 - segment.layer_range.0;
            // Pass the real prompt byte length for adaptive timeout. Passing 0
            // would classify this as a decode pass (DECODE_SECS_PER_LAYER=2)
            // and yield the 30s SEGMENT_TIMEOUT_MIN_SECS floor regardless of
            // model size — too tight for long prompts on slow hardware.
            // forward_through_segments already does this for the standard
            // path; mirror here.
            let budget = super::local::SegmentBudget::for_forward(
                &self.shared_state,
                &segment.node_id,
                &segment.shard_id.model_id,
                crate::daemon::state::WorkKind::Prefill,
                num_layers,
                prompt_byte_len,
                // The prompt itself, not hidden states.
                super::local::ActivationUnits::PromptBytes,
            );
            let prefill_result = Self::wait_for_result(
                rx,
                request_id,
                0,
                &segment.node_id,
                num_layers,
                prompt_byte_len,
                budget,
            )
            .await?;
            // Result delivered (the dispatcher already removed the entry when
            // it consumed the oneshot); disarm the guard so it doesn't double-
            // remove on drop.
            prefill_guard.disarm();

            if prefill_result.token_ids.is_empty() {
                return Err(SwarmError::Inference(
                    "speculative: prefill returned no tokens".into(),
                ));
            }
            let first_token = prefill_result.token_ids[0];
            // Extract model cache (EOS, decoder, ptc) — must happen after the
            // first forward has populated split_models / executor state.
            let (ptc, eos, decoder) = self.extract_model_cache(&prompt).await;
            let eos_set: std::collections::HashSet<u32> = eos.into_iter().collect();
            (first_token, ptc, eos_set, decoder)
        };

        // Prefill draft model and advance it by the first_token so its KV
        // matches target's (prompt tokens processed + nothing, but we'll
        // feed first_token into draft momentarily via the first spec round's
        // draft-phase bootstrap).
        //
        // Use the existing generate_stream helper wouldn't work — we need
        // precise token-by-token control. Prefill by directly tokenizing and
        // stepping the draft context.
        let draft_prefill = tokio::task::block_in_place(|| draft_prefill(&mut draft, &prompt));
        let mut draft_state = match draft_prefill {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!(%request_id, error = %e, "speculative: draft prefill failed — falling back");
                return Ok(None);
            }
        };

        // Generated tokens (for max_tokens accounting and final output).
        let mut generated: Vec<u32> = vec![first_token];
        let mut current_pos = prompt_token_count;
        let mut last_token = first_token;
        // KV length expected on the remote BEFORE the next forward runs.
        // After prefill + 0 generated forwards, remote KV = prompt_token_count.
        let mut expected_kv_len: u32 = prompt_token_count as u32;
        let mut pending_truncate: Option<u32> = None;

        // Stream the first token if we have a streaming channel.
        super::emit_first_streaming_token(&token_tx, &decoder, first_token).await;

        let mut acceptance_proposed: u32 = 0;
        let mut acceptance_accepted: u32 = 0;
        let mut finish_reason = String::new();

        if eos_tokens.contains(&first_token) {
            finish_reason = "stop".to_string();
        }

        // Speculative round loop.
        while finish_reason.is_empty() && (generated.len() as u32) < max_tokens {
            // Honor external cancel between rounds — same pattern as
            // execute_distributed line 174. Each spec round emits γ+1
            // tokens, so worst-case observation latency is one round.
            if self.request.is_cancelled() {
                tracing::info!(
                    request_id = %request_id,
                    "DIAG: speculative inference cancelled externally"
                );
                finish_reason = "stop".to_string();
                break;
            }
            let remaining = max_tokens - generated.len() as u32;
            let this_gamma = gamma.min(remaining).max(1);

            // SWARM-SPEC Layer 1.1+1.2: try n-gram lookup first. On hit
            // we skip the draft-model forward entirely (sub-ms hash
            // lookup vs ~5-20ms draft decode). On miss we fall back to
            // the draft model. Drafts are still verified by the target,
            // so there's no quality risk — only acceptance rate
            // varies. The literature (PROMTEC, prompt-lookup-decoding)
            // reports 2.4-4.2× speedup on input-grounded workloads
            // (code, RAG, summarisation) — exactly SwarmLLM's primary
            // workload (Claude Code subs, MCP tool use).
            let ngram_drafts = if self.shared_state.config.inference.ngram_lookup_enabled {
                let d = ngram_lookup_drafts(
                    &self.shared_state.config.inference,
                    &draft_state.prompt_tokens,
                    &generated,
                    this_gamma,
                );
                // R137: bump lifetime hit/miss counters for visibility via
                // /api/admin/stats → swarm_spec.ngram.
                if d.is_empty() {
                    self.shared_state
                        .metrics
                        .ngram_misses
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                } else {
                    self.shared_state
                        .metrics
                        .ngram_hits
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
                d
            } else {
                Vec::new()
            };

            let (drafts, used_ngram) = if !ngram_drafts.is_empty() {
                // N-gram fast-path hit. Still feed the proposed tokens
                // through the draft model so its KV stays in sync with
                // what the target will commit (if accepted). The draft
                // forward is unavoidable for KV sync but we save the
                // sampling step.
                let sync_outcome = tokio::task::block_in_place(|| {
                    draft_sync_tokens(&mut draft_state, &mut draft, last_token, &ngram_drafts)
                });
                if let Err(e) = sync_outcome {
                    tracing::warn!(%request_id, error = %e, "speculative: draft KV sync after n-gram lookup failed — falling back to draft sample");
                    let draft_outcome = tokio::task::block_in_place(|| {
                        draft_next_gamma(&mut draft_state, &mut draft, last_token, this_gamma)
                    });
                    match draft_outcome {
                        Ok(d) => (d, false),
                        Err(e2) => {
                            tracing::warn!(%request_id, error = %e2, "speculative: draft step failed after n-gram fallback — partial");
                            return Ok(Some(self.finish_speculative(
                                request_id,
                                generated,
                                &decoder,
                                &eos_tokens,
                                prompt_token_count as u32,
                                "stop".into(),
                            )));
                        }
                    }
                } else {
                    (ngram_drafts, true)
                }
            } else {
                // Draft phase — sync, llama-cpp.
                let draft_outcome = tokio::task::block_in_place(|| {
                    draft_next_gamma(&mut draft_state, &mut draft, last_token, this_gamma)
                });
                let d = match draft_outcome {
                    Ok(d) => d,
                    Err(e) => {
                        tracing::warn!(%request_id, error = %e, "speculative: draft step failed — falling back");
                        // Return the partial output as a failed-fallback signal.
                        return Ok(Some(self.finish_speculative(
                            request_id,
                            generated,
                            &decoder,
                            &eos_tokens,
                            prompt_token_count as u32,
                            "stop".into(),
                        )));
                    }
                };
                (d, false)
            };

            tracing::trace!(
                %request_id,
                drafts = drafts.len(),
                from_ngram = used_ngram,
                "speculative round draft source"
            );

            if drafts.is_empty() {
                break;
            }

            // Build the verify batch: [last_token, q_1, ..., q_γ].
            let mut verify_tokens: Vec<u32> = Vec::with_capacity(drafts.len() + 1);
            verify_tokens.push(last_token);
            verify_tokens.extend_from_slice(&drafts);

            // Send verify and await spec_logits.
            let spec_result = send_verify_batch(
                &self.shared_state,
                &self.network_tx,
                request_id,
                current_pos as u32,
                &segment,
                &target_peer_bytes,
                &verify_tokens,
                pending_truncate,
            )
            .await;
            let spec_logits = match spec_result {
                Ok(r) => r,
                Err(e) => {
                    tracing::warn!(%request_id, error = %e, "speculative: verify failed — returning partial");
                    return Ok(Some(self.finish_speculative(
                        request_id,
                        generated,
                        &decoder,
                        &eos_tokens,
                        prompt_token_count as u32,
                        "stop".into(),
                    )));
                }
            };
            // Remote forwarded verify_tokens.len() = γ+1 positions →
            // expected spec_logits.len() == γ+1. greedy_accept_reject indexes
            // `spec_logits[drafts.len()]` (= γ) on ALL-ACCEPTED, so we need
            // strict `< drafts.len() + 1` to avoid OOB on a corrupt response.
            if spec_logits.len() < drafts.len() + 1 {
                tracing::warn!(
                    %request_id,
                    got = spec_logits.len(),
                    want_min = drafts.len() + 1,
                    "speculative: insufficient spec_logits — returning partial"
                );
                return Ok(Some(self.finish_speculative(
                    request_id,
                    generated,
                    &decoder,
                    &eos_tokens,
                    prompt_token_count as u32,
                    "stop".into(),
                )));
            }

            // After this forward, remote KV was grown by verify_tokens.len().
            let kv_after_forward = expected_kv_len + verify_tokens.len() as u32;

            // Greedy accept-reject — see `greedy_accept_reject` doc.
            //   spec_logits[i] verifies drafts[i]; bonus comes from the
            //   first mismatch or from spec_logits[γ] when all accepted.
            let (accepted, bonus, _all_accepted) = greedy_accept_reject(&drafts, &spec_logits);

            acceptance_proposed += drafts.len() as u32;
            acceptance_accepted += accepted.len() as u32;

            // Emit accepted + bonus.
            let mut emitted: Vec<u32> = accepted
                .iter()
                .copied()
                .chain(std::iter::once(bonus))
                .collect();

            // BUG-FIX (R105): truncate emitted at the FIRST EOS, before any
            // downstream consumer (streaming, generated buffer, KV bookkeeping)
            // sees post-EOS tokens. Previously the EOS check happened AFTER
            // both streaming and `generated.extend`, so junk tokens following
            // an EOS in a multi-token speculative round (e.g. [a, EOS, b, c])
            // were streamed to the client and appended to `generated`. Same
            // pattern applied in dsd.rs.
            if let Some(eos_at) = emitted.iter().position(|t| eos_tokens.contains(t)) {
                emitted.truncate(eos_at + 1);
            }

            // Streaming per token.
            super::emit_streaming_batch(&token_tx, &decoder, &emitted, &mut finish_reason).await;

            // Bail before the per-round bookkeeping (KV tracking,
            // `draft_sync_after_round` which holds a `block_in_place`) when the
            // client has disconnected. The inner `break` above only exits the
            // streaming for-loop; without this gate the round still pays for
            // the synchronous draft sync before the outer `while` re-checks
            // `finish_reason.is_empty()`.
            if !finish_reason.is_empty() {
                break;
            }

            generated.extend(&emitted);

            // Track KV state.
            // Target KV grew by verify_tokens.len() entries, but we only want
            // the entries corresponding to [last_token, q_1..q_k]. That's
            // (accepted.len() + 1) positions. Bonus is NOT in target KV
            // (we sampled it, never forwarded). So after this round, target
            // KV should be = expected_kv_len + accepted.len() + 1.
            let new_expected_kv = expected_kv_len + accepted.len() as u32 + 1;
            if new_expected_kv < kv_after_forward {
                pending_truncate = Some(new_expected_kv);
            } else {
                pending_truncate = None;
            }
            expected_kv_len = new_expected_kv;

            // Draft-side KV sync — make the draft's KV length match target's.
            tokio::task::block_in_place(|| {
                draft_sync_after_round(&mut draft_state, &mut draft, &drafts, &accepted, bonus)
            })?;

            current_pos += emitted.len();
            last_token = *emitted.last().unwrap();

            // Stop conditions. R105's truncation at the first EOS guarantees
            // that if `emitted` contains an EOS token it must be the last
            // element, so checking `last_token` is sufficient.
            if eos_tokens.contains(&last_token) {
                finish_reason = "stop".to_string();
                break;
            }
        }

        if finish_reason.is_empty() {
            if (generated.len() as u32) >= max_tokens {
                finish_reason = "length".to_string();
            } else {
                finish_reason = "stop".to_string();
            }
        }

        // Send final SSE event.
        if let Some(ref tx) = token_tx {
            let _ = tx
                .send(StreamingTokenEvent {
                    text: String::new(),
                    finish_reason: Some(finish_reason.clone()),
                    matched_stop_sequence: None,
                })
                .await;
        }

        tracing::info!(
            %request_id,
            proposed = acceptance_proposed,
            accepted = acceptance_accepted,
            acceptance_rate = format_args!(
                "{:.2}",
                if acceptance_proposed > 0 {
                    acceptance_accepted as f32 / acceptance_proposed as f32
                } else {
                    0.0
                }
            ),
            "speculative: round stats"
        );

        Ok(Some(self.finish_speculative(
            request_id,
            generated,
            &decoder,
            &eos_tokens,
            prompt_token_count as u32,
            finish_reason,
        )))
    }

    pub(super) fn finish_speculative(
        &self,
        request_id: uuid::Uuid,
        generated: Vec<u32>,
        decoder: &CachedDecoder,
        eos_tokens: &std::collections::HashSet<u32>,
        prompt_tokens: u32,
        finish_reason: String,
    ) -> InferenceOutput {
        let clean: Vec<u32> = generated
            .into_iter()
            .filter(|t| !eos_tokens.contains(t))
            .collect();
        let completion_tokens = clean.len() as u32;
        let content = decoder.decode_tokens(&clean);
        InferenceOutput {
            request_id,
            content,
            prompt_tokens,
            completion_tokens,
            finish_reason,
            session_id: self.request.session_id.clone(),
            token_logprobs: vec![],
            // Speculative path: matched stop string isn't tracked here today.
            matched_stop_sequence: None,
            trace: None,
        }
    }
}

// ─── Network helper ────────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
pub(super) async fn send_verify_batch(
    shared_state: &Arc<SharedState>,
    network_tx: &mpsc::Sender<NetworkCommand>,
    request_id: uuid::Uuid,
    index_pos: u32,
    segment: &crate::types::PipelineSegment,
    target_peer_bytes: &[u8],
    verify_tokens: &[u32],
    truncate_kv_to: Option<u32>,
) -> Result<Vec<Vec<f32>>, SwarmError> {
    // Register oneshot for the result. Cap-checked + RAII-guarded so a
    // wait_for_result Err propagation doesn't leak the pending entry —
    // see PendingLayerResultGuard / gotcha #45.
    let (rx, mut verify_guard) = super::register_pending_layer_result(
        &shared_state.pending_layer_results,
        request_id,
        Some(segment.node_id.clone()),
    )?;

    // Build the LayerForward. As of DSD Phase 4 (Item 12) the worker
    // unifies speculative and standard input paths through the first-segment
    // multi-token decode branch, which reads γ token IDs from `activations`
    // (γ × 8 bytes LE). Pack all verify_tokens, not just the first.
    let activations = super::pack_verify_tokens_to_le_bytes(verify_tokens);
    let forward = super::build_spec_verify_forward(
        request_id,
        index_pos,
        activations,
        segment,
        shared_state.identity.node_id().0,
        truncate_kv_to,
    );
    if network_tx
        .send(NetworkCommand::SendTensor {
            target_peer_bytes: target_peer_bytes.to_vec(),
            forward,
        })
        .await
        .is_err()
    {
        // Drop guard removes the pending entry on return.
        return Err(SwarmError::Network("verify send dropped".into()));
    }
    let num_layers = segment.layer_range.1 - segment.layer_range.0;
    let budget = super::local::SegmentBudget::for_forward(
        shared_state,
        &segment.node_id,
        &segment.shard_id.model_id,
        // A verify batch is a handful of tokens, not a prompt pass.
        crate::daemon::state::WorkKind::Decode,
        num_layers,
        verify_tokens.len() * 4,
        super::local::ActivationUnits::PromptBytes,
    );
    let result: LayerResult = PipelineExecutor::wait_for_result(
        rx,
        request_id,
        0,
        &segment.node_id,
        num_layers,
        verify_tokens.len() * 4,
        budget,
    )
    .await?;
    // Result delivered (the dispatcher already removed the entry); disarm
    // the guard so we don't double-remove on drop.
    verify_guard.disarm();
    if let Some(NetworkFinishReason::Error(msg)) = &result.finish_reason {
        // The peer's error crossed the wire as text, so its class has to be
        // recovered before it reaches a caller — see the sibling in
        // `pipeline::forward_verify_through_segments`.
        return Err(crate::error::reclassify_flattened_error(msg)
            .unwrap_or_else(|| SwarmError::Inference(msg.clone())));
    }
    if result.spec_logits.is_empty() {
        return Err(SwarmError::Inference(
            "speculative verify returned no spec_logits".into(),
        ));
    }
    Ok(result.spec_logits)
}

pub(super) fn argmax(logits: &[f32]) -> u32 {
    let mut best_idx: usize = 0;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best_idx = i;
        }
    }
    best_idx as u32
}

/// Greedy accept-reject for distributed speculative decoding.
///
/// `spec_logits[i]` is the target's predicted distribution AFTER seeing
/// input position `i`, so it verifies `drafts[i]`. Walks the draft tokens:
/// accept while the target's argmax matches; on first mismatch, take the
/// target's argmax as the bonus and stop. If all drafts are accepted,
/// take `spec_logits[drafts.len()]` (the target's prediction beyond the
/// last accepted token) as the bonus.
///
/// Returns `(accepted, bonus, all_accepted)`. Shared by Item 2's
/// `try_speculative_distributed` (single-segment) and Item 12's
/// `try_dsd_distributed` (multi-segment) — both paths run bit-identical
/// arithmetic; what differs around the call is round bookkeeping
/// (gamma controller updates, partial-accept fixup) and token emission.
pub(super) fn greedy_accept_reject(
    drafts: &[u32],
    spec_logits: &[Vec<f32>],
) -> (Vec<u32>, u32, bool) {
    // SEC: Reject NaN/Inf from a peer-supplied target segment. NaN comparisons
    // in IEEE 754 are non-deterministic in argmax (`partial_cmp` returns
    // `Equal` for any NaN), letting a malicious peer steer accepted tokens.
    // We treat any non-finite row as an all-rejected decision with bonus = 0,
    // forcing the caller to fall back / finish the request safely.
    let nonfinite = spec_logits
        .iter()
        .take(drafts.len() + 1)
        .any(|row| row.iter().any(|v| !v.is_finite()));
    if nonfinite {
        return (Vec::new(), 0, false);
    }
    let mut accepted: Vec<u32> = Vec::with_capacity(drafts.len());
    let mut bonus: u32 = 0;
    for (i, &q) in drafts.iter().enumerate() {
        let target_pick = argmax(&spec_logits[i]);
        if target_pick == q {
            accepted.push(q);
        } else {
            bonus = target_pick;
            break;
        }
    }
    let all_accepted = accepted.len() == drafts.len();
    if all_accepted {
        bonus = argmax(&spec_logits[drafts.len()]);
    }
    (accepted, bonus, all_accepted)
}

/// Accept-reject for a DRAFT THAT CARRIES NO DISTRIBUTION, honouring the
/// caller's sampling parameters.
///
/// An n-gram draft proposes a token with no probability behind it — `q = δ_x` —
/// and speculative sampling then reduces to: accept with probability
/// `min(1, p(x)/q(x)) = p(x)`, otherwise draw from the residual
/// `norm((p − q)₊)`, i.e. `p` with `x` removed and renormalised. "Draw `t ~ p`;
/// keep the draft iff `t == x`" has exactly those two branches, so sampling each
/// position through the real sampler and keeping a match IS that rule — at any
/// temperature, with no separate machinery.
/// `accepting_only_on_a_match_preserves_the_sampled_distribution` in
/// `inference::sampling` pins it, with a control that fails if the metric could
/// not detect a bias.
///
/// **Why this exists beside `greedy_accept_reject`.** That one takes the raw
/// argmax, which ignores temperature, top-k, top-p AND the repetition
/// penalties. It is correct for the paths that are greedy-only by construction
/// (a draft MODEL needs the draft's own probabilities to do this properly, which
/// is a different algorithm), but it meant the same request got a different
/// answer depending on whether it was served locally or across peers — the local
/// worker has always sampled properly. It also meant n-gram speculation could
/// only ever run at temperature 0, which is not what any client asks for: the
/// OpenAI surface defaults to 0.7 and the Anthropic one to 1.0.
///
/// `generated` is the history the penalties apply against, and it grows as
/// drafts are accepted, so each position sees what it would have seen had the
/// tokens been produced one at a time.
///
/// The non-finite guard is NOT optional and is the reason this cannot simply
/// call the sampler in a loop: a peer supplies these logits, and NaN comparisons
/// are non-deterministic, so a malicious segment could otherwise steer which
/// tokens get accepted. Same treatment as the greedy sibling — reject the whole
/// round.
pub(super) fn sampled_accept_reject(
    drafts: &[u32],
    spec_logits: &[Vec<f32>],
    params: &crate::types::SamplingParams,
    generated: &[u32],
) -> (Vec<u32>, u32, bool) {
    let nonfinite = spec_logits
        .iter()
        .take(drafts.len() + 1)
        .any(|row| row.iter().any(|v| !v.is_finite()));
    if nonfinite {
        return (Vec::new(), 0, false);
    }
    let vocab = spec_logits.first().map(|r| r.len()).unwrap_or(0);
    let mut ctx = crate::inference::sampling::SamplingContext::new(vocab);
    let mut history: Vec<u32> = generated.to_vec();
    let mut accepted: Vec<u32> = Vec::with_capacity(drafts.len());
    let mut bonus: u32 = 0;
    for (i, &q) in drafts.iter().enumerate() {
        let mut row = spec_logits[i].clone();
        let pick = crate::inference::sampling::sample_token_with_history(
            &mut row, params, &history, &mut ctx,
        );
        if pick == q {
            accepted.push(q);
            history.push(q);
        } else {
            bonus = pick;
            break;
        }
    }
    let all_accepted = accepted.len() == drafts.len();
    if all_accepted {
        let mut row = spec_logits[drafts.len()].clone();
        bonus = crate::inference::sampling::sample_token_with_history(
            &mut row, params, &history, &mut ctx,
        );
    }
    (accepted, bonus, all_accepted)
}

// ─── Draft-model driver (llama-cpp) ────────────────────────────────────────

#[cfg(feature = "llama")]
pub(crate) struct DraftState {
    pub ctx: llama_cpp_2::context::LlamaContext<'static>,
    pub batch: llama_cpp_2::llama_batch::LlamaBatch<'static>,
    pub pos: usize,
    pub n_vocab: usize,
    /// SWARM-SPEC Layer 1: prompt tokens captured at prefill time, used
    /// by the n-gram lookup fast-path to find candidate drafts without
    /// running a draft-model forward pass.
    pub prompt_tokens: Vec<u32>,
}

#[cfg(not(feature = "llama"))]
pub(crate) struct DraftState {
    /// SWARM-SPEC Layer 1: prompt tokens (always present, even without
    /// llama feature, so the n-gram lookup module can compile cleanly).
    /// In non-llama builds this remains empty and the n-gram path
    /// auto-skips.
    pub prompt_tokens: Vec<u32>,
}

// SAFETY: We carefully avoid letting this state escape the mutex-locked
// ModelExecutor. The 'static lifetime is a workaround for storing the context
// inline — in practice its real lifetime is bounded by the locked executor.
#[cfg(feature = "llama")]
unsafe impl Send for DraftState {}

/// Prefill the draft model with `prompt`. Returns a `DraftState` holding the
/// llama-cpp context positioned at `prompt_tokens.len()`.
#[cfg(feature = "llama")]
pub(super) fn draft_prefill(
    draft: &mut crate::inference::executor::ModelExecutor,
    prompt: &str,
) -> Result<DraftState, SwarmError> {
    use llama_cpp_2::context::params::LlamaContextParams;
    use llama_cpp_2::llama_batch::LlamaBatch;
    use llama_cpp_2::model::AddBos;
    use std::num::NonZeroU32;

    let model = draft
        .raw_model()
        .ok_or_else(|| SwarmError::Inference("draft model not loaded".into()))?;
    let backend = draft
        .raw_backend()
        .ok_or_else(|| SwarmError::Inference("draft backend not initialized".into()))?;

    let n_ctx = model.n_ctx_train();
    let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(n_ctx));
    // SAFETY: We cast the context's lifetime to 'static. The real lifetime is
    // bounded by the backend+model references inside the locked ModelExecutor.
    // As long as the DraftState doesn't outlive the executor's mutex guard,
    // this is sound. `try_speculative_distributed` holds the guard for the
    // entire request, so the invariant holds.
    let ctx = model
        .new_context(backend, ctx_params)
        .map_err(|e| SwarmError::Inference(format!("draft ctx: {e}")))?;
    let ctx: llama_cpp_2::context::LlamaContext<'static> = unsafe { std::mem::transmute(ctx) };

    let tokens = model
        .str_to_token(prompt, AddBos::Always)
        .map_err(|e| SwarmError::Inference(format!("draft tokenize: {e}")))?;
    let mut batch = LlamaBatch::new(n_ctx as usize, 1);
    for (i, token) in tokens.iter().enumerate() {
        let is_last = i == tokens.len() - 1;
        batch
            .add(*token, i as i32, &[0], is_last)
            .map_err(|e| SwarmError::Inference(format!("draft batch add: {e}")))?;
    }
    let mut ctx = ctx;
    ctx.decode(&mut batch)
        .map_err(|e| SwarmError::Inference(format!("draft prefill decode: {e}")))?;

    let n_vocab = model.n_vocab() as usize;
    let prompt_tokens: Vec<u32> = tokens.iter().map(|t| t.0 as u32).collect();
    Ok(DraftState {
        ctx,
        batch,
        pos: tokens.len(),
        n_vocab,
        prompt_tokens,
    })
}

#[cfg(not(feature = "llama"))]
pub(super) fn draft_prefill(
    _draft: &mut crate::inference::executor::ModelExecutor,
    _prompt: &str,
) -> Result<DraftState, SwarmError> {
    Err(SwarmError::Inference(
        "speculative requires llama feature".into(),
    ))
}

/// Advance the draft model by feeding `bootstrap` (the last accepted target
/// token) then greedily sampling γ tokens from the draft. Returns the γ
/// tokens. Draft KV ends γ+1 positions ahead of where it started.
#[cfg(feature = "llama")]
pub(super) fn draft_next_gamma(
    state: &mut DraftState,
    _draft: &mut crate::inference::executor::ModelExecutor,
    bootstrap: u32,
    gamma: u32,
) -> Result<Vec<u32>, SwarmError> {
    use llama_cpp_2::token::LlamaToken;

    // Feed bootstrap.
    state.batch.clear();
    state
        .batch
        .add(LlamaToken(bootstrap as i32), state.pos as i32, &[0], true)
        .map_err(|e| SwarmError::Inference(format!("draft bootstrap batch: {e}")))?;
    state
        .ctx
        .decode(&mut state.batch)
        .map_err(|e| SwarmError::Inference(format!("draft bootstrap decode: {e}")))?;
    state.pos += 1;

    let mut drafts = Vec::with_capacity(gamma as usize);
    for _ in 0..gamma {
        let logits: &[f32] = &state.ctx.get_logits()[..state.n_vocab];
        let t = argmax(logits);
        drafts.push(t);

        state.batch.clear();
        state
            .batch
            .add(LlamaToken(t as i32), state.pos as i32, &[0], true)
            .map_err(|e| SwarmError::Inference(format!("draft step batch: {e}")))?;
        state
            .ctx
            .decode(&mut state.batch)
            .map_err(|e| SwarmError::Inference(format!("draft step decode: {e}")))?;
        state.pos += 1;
    }
    Ok(drafts)
}

#[cfg(not(feature = "llama"))]
pub(super) fn draft_next_gamma(
    _state: &mut DraftState,
    _draft: &mut crate::inference::executor::ModelExecutor,
    _bootstrap: u32,
    _gamma: u32,
) -> Result<Vec<u32>, SwarmError> {
    Err(SwarmError::Inference(
        "speculative requires llama feature".into(),
    ))
}

/// SWARM-SPEC Layer 1 helper: try n-gram lookup for the next γ draft tokens.
/// Returns an empty Vec when no match is found OR when n-gram lookup is
/// disabled. Caps the returned candidate at `this_gamma` so the cascading
/// caller can use the existing verify wire format unchanged.
///
/// `prompt_tokens` is the tokenized prompt (captured at draft prefill time).
/// `generated` is the running list of tokens emitted so far. The cascade
/// searches both for matches of the recent context tail.
pub(super) fn ngram_lookup_drafts(
    cfg: &crate::config::InferenceConfig,
    prompt_tokens: &[u32],
    generated: &[u32],
    this_gamma: u32,
) -> Vec<u32> {
    use crate::inference::ngram_lookup::{cascade_find_candidate, NgramLookupConfig};
    if !cfg.ngram_lookup_enabled {
        return Vec::new();
    }
    let mut context: Vec<u32> = Vec::with_capacity(prompt_tokens.len() + generated.len());
    context.extend_from_slice(prompt_tokens);
    context.extend_from_slice(generated);
    let prompt_len = prompt_tokens.len();
    let lookup_cfg = NgramLookupConfig {
        max_ngram_size: cfg.ngram_max_size as usize,
        min_ngram_size: crate::inference::ngram_lookup::DEFAULT_MIN_NGRAM_SIZE,
        num_pred_tokens: cfg.ngram_num_pred_tokens as usize,
    };
    // Recent-generation window is the last 500 generated tokens — captures
    // "model is in a repeating-pattern groove" without slowing the lookup
    // on long conversations.
    let (cand, _source) = cascade_find_candidate(&context, prompt_len, 500, lookup_cfg);
    // Cap at this_gamma so the verify wire format stays sized as expected.
    if cand.len() <= this_gamma as usize {
        cand
    } else {
        cand[..this_gamma as usize].to_vec()
    }
}

/// SWARM-SPEC Layer 1 helper: feed `[bootstrap, tokens...]` through the
/// draft model to keep its KV cache in sync with the target's expected
/// KV state, but DON'T sample (n-gram already provided the drafts). After
/// this call, `state.pos` advances by `1 + tokens.len()`. Mirrors the KV
/// advancement that `draft_next_gamma` would have done.
#[cfg(feature = "llama")]
pub(super) fn draft_sync_tokens(
    state: &mut DraftState,
    _draft: &mut crate::inference::executor::ModelExecutor,
    bootstrap: u32,
    tokens: &[u32],
) -> Result<(), SwarmError> {
    use llama_cpp_2::token::LlamaToken;

    state.batch.clear();
    state
        .batch
        .add(LlamaToken(bootstrap as i32), state.pos as i32, &[0], true)
        .map_err(|e| SwarmError::Inference(format!("ngram-sync bootstrap batch: {e}")))?;
    state
        .ctx
        .decode(&mut state.batch)
        .map_err(|e| SwarmError::Inference(format!("ngram-sync bootstrap decode: {e}")))?;
    state.pos += 1;

    for &t in tokens {
        state.batch.clear();
        state
            .batch
            .add(LlamaToken(t as i32), state.pos as i32, &[0], true)
            .map_err(|e| SwarmError::Inference(format!("ngram-sync step batch: {e}")))?;
        state
            .ctx
            .decode(&mut state.batch)
            .map_err(|e| SwarmError::Inference(format!("ngram-sync step decode: {e}")))?;
        state.pos += 1;
    }
    Ok(())
}

#[cfg(not(feature = "llama"))]
pub(super) fn draft_sync_tokens(
    _state: &mut DraftState,
    _draft: &mut crate::inference::executor::ModelExecutor,
    _bootstrap: u32,
    _tokens: &[u32],
) -> Result<(), SwarmError> {
    Err(SwarmError::Inference(
        "speculative requires llama feature".into(),
    ))
}

/// Sync draft KV after a round. Rewinds the draft's KV cache to match the
/// target's KV length (= pre-round length + accepted.len() + 1 for bonus).
/// After this call, draft_state.pos matches target_pos and the draft is
/// ready to bootstrap the next round from `bonus`.
#[cfg(feature = "llama")]
pub(super) fn draft_sync_after_round(
    state: &mut DraftState,
    _draft: &mut crate::inference::executor::ModelExecutor,
    drafts: &[u32],
    accepted: &[u32],
    _bonus: u32,
) -> Result<(), SwarmError> {
    // Before this fn: state.pos = pos_at_round_start + 1 (bootstrap) + gamma (drafts).
    // After accept-reject: target keeps pos_at_round_start + 1 (bootstrap in target KV)
    //                      + accepted.len() entries for accepted draft tokens.
    //                      Bonus is NOT in target KV.
    // So draft should rewind to: current pos - (drafts.len() - accepted.len())
    // Then we DON'T feed bonus into draft — it rides as the bootstrap for the
    // NEXT round (which will feed it via draft_next_gamma's bootstrap step).
    let num_rejected = drafts.len().saturating_sub(accepted.len());
    if num_rejected > 0 {
        let target_draft_pos = state.pos - num_rejected;
        let _ = state
            .ctx
            .clear_kv_cache_seq(Some(0), Some(target_draft_pos as u32), None);
        state.pos = target_draft_pos;
    }
    Ok(())
}

#[cfg(not(feature = "llama"))]
pub(super) fn draft_sync_after_round(
    _state: &mut DraftState,
    _draft: &mut crate::inference::executor::ModelExecutor,
    _drafts: &[u32],
    _accepted: &[u32],
    _bonus: u32,
) -> Result<(), SwarmError> {
    Err(SwarmError::Inference(
        "speculative requires llama feature".into(),
    ))
}

#[cfg(test)]
mod sampled_accept_reject_tests {
    use super::{greedy_accept_reject, sampled_accept_reject};
    use crate::types::SamplingParams;

    fn greedy_params() -> SamplingParams {
        SamplingParams {
            temperature: 0.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            ..Default::default()
        }
    }

    fn logits(rows: &[&[f32]]) -> Vec<Vec<f32>> {
        rows.iter().map(|r| r.to_vec()).collect()
    }

    /// **The compatibility guarantee.** This path shipped greedy-only, so at
    /// temperature 0 with no penalties the new sampler must reproduce the old
    /// argmax decision exactly — otherwise switching it changes answers for
    /// every request that was already using it.
    #[test]
    fn at_temperature_zero_it_agrees_with_the_argmax_it_replaces() {
        let rows = logits(&[
            &[0.1, 5.0, 0.2], // argmax 1
            &[7.0, 0.5, 0.3], // argmax 0
            &[0.0, 0.1, 9.0], // argmax 2
            &[3.0, 0.2, 0.1], // argmax 0
        ]);
        for drafts in [
            vec![],
            vec![1u32],
            vec![1, 0],
            vec![1, 0, 2],
            vec![1, 9], // second draft mismatches
            vec![9],    // first draft mismatches
        ] {
            let want = greedy_accept_reject(&drafts, &rows);
            let got = sampled_accept_reject(&drafts, &rows, &greedy_params(), &[]);
            assert_eq!(got, want, "drafts {drafts:?}");
        }
    }

    /// A peer supplies these logits. NaN comparisons are non-deterministic in
    /// argmax, so a malicious segment could otherwise steer which drafts are
    /// accepted — the guard must survive the change of sampler.
    #[test]
    fn non_finite_logits_from_a_peer_reject_the_whole_round() {
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let rows = logits(&[&[0.1, bad, 0.2], &[1.0, 2.0, 3.0]]);
            let got = sampled_accept_reject(&[1], &rows, &greedy_params(), &[]);
            assert_eq!(got, (Vec::new(), 0, false), "value {bad}");
            // Same verdict as the greedy sibling, so the two cannot drift.
            assert_eq!(got, greedy_accept_reject(&[1], &rows));
        }
    }

    /// The behaviour change that comes with sampling properly: repetition
    /// penalties now apply here. They always applied on the local worker, so
    /// before this the same request got a different answer depending on whether
    /// it was served locally or across peers.
    #[test]
    fn repetition_penalties_now_reach_this_path() {
        // Token 1 wins on raw logits, but has been generated already.
        let rows = logits(&[&[4.0, 5.0, 0.0]]);
        let mut penalised = greedy_params();
        penalised.frequency_penalty = 2.0;

        let (_, unpenalised_bonus, _) =
            sampled_accept_reject(&[], &rows, &greedy_params(), &[1, 1, 1]);
        let (_, penalised_bonus, _) = sampled_accept_reject(&[], &rows, &penalised, &[1, 1, 1]);
        assert_eq!(
            unpenalised_bonus, 1,
            "without a penalty the raw argmax wins"
        );
        assert_ne!(
            penalised_bonus, 1,
            "a frequency penalty against an already-generated token must move the pick"
        );
        // And the greedy helper it replaces cannot express this at all.
        assert_eq!(greedy_accept_reject(&[], &rows).1, 1);
    }
}
