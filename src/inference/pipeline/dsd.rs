//! Decentralized Speculative Decoding coordinator loop (Item 12 / DSD).
//!
//! See `docs/plans/archive/distributed_inference_speedup.md` § Item 12 for the
//! design and arxiv 2511.11733 / 2511.21669 for the source papers.
//!
//! # What this fixes that Item 2 doesn't
//!
//! Item 2's `try_speculative_distributed` requires the entire model to live
//! on a single remote peer (`is_first && is_last`). DSD generalizes the
//! verify pass to a multi-segment pipeline: the coordinator drafts γ tokens
//! locally, then propagates the γ-token batch through every pipeline segment
//! in one round trip. Each intermediate segment processes
//! `[1, γ, hidden]` activations through its layer range; the last segment
//! runs `forward_verify_all_positions_pre_embedded` and returns γ+1 logit
//! vectors.
//!
//! # Time-cost analysis (paper 1)
//!
//! Per-token cost without DSD: `T_std = γ · (t0 + (N-1)·t1)`
//! Per-token cost with DSD:    `T_DSD = γ · t0 + (N-1) · t1`
//!
//! The savings `(N-1)·t1·(γ-1)` grow with both pipeline depth `N` and
//! per-link RTT `t1`. Paper's regime (`3·t0 < t1 < 10·t0`) matches
//! SwarmLLM's WAN deployments (50–150 ms RTT vs 10–100 ms candle compute).
//!
//! # Eligibility (MVP)
//!
//! - `decentralized_spec_decoding && speculative_decoding` config flags both on
//! - Pipeline has 2+ segments AND no TP groups (single-segment is Item 2's job)
//! - Greedy temperature == 0
//! - Draft model loaded
//! - No vision or LoRA
//!
//! # Correctness
//!
//! Greedy speculative decoding produces output bit-identical to greedy
//! non-speculative decoding because we only accept draft tokens whose ID
//! matches the target's argmax at the same position. Regressions show up as
//! output divergence on a fixed prompt with `temperature=0`.

use crate::error::SwarmError;
#[cfg(feature = "llama")]
use crate::inference::router::StreamingTokenEvent;
use crate::inference::router::{InferenceOutput, StreamingTokenTx};

#[cfg(feature = "llama")]
use super::speculative::{
    draft_next_gamma, draft_prefill, draft_sync_after_round, draft_sync_tokens, ngram_lookup_drafts,
};
use super::PipelineExecutor;
#[cfg(feature = "llama")]
use crate::inference::dsd_controller::GammaController;

/// Fast-path preconditions for the DSD coordinator loop.
#[cfg(feature = "llama")]
fn eligible(exec: &PipelineExecutor) -> bool {
    // Path-specific flag.
    if !exec
        .shared_state
        .config
        .inference
        .decentralized_spec_decoding
    {
        return false;
    }
    // Common speculative-path baseline (greedy temp, draft model, etc.).
    if !super::speculative_common_eligible(exec) {
        return false;
    }
    // Multi-segment pipeline only — single-segment falls through to Item 2.
    if exec.assignment.segments.len() < 2 {
        return false;
    }
    true
}

#[cfg(not(feature = "llama"))]
impl PipelineExecutor {
    /// DSD requires the `llama` feature for the local draft model. Without
    /// it the eligibility check above also fails (`draft_model_path` is None
    /// because no draft can be loaded), but stub the entry point regardless
    /// so the dispatch site in `execute_distributed` compiles unconditionally.
    pub(super) async fn try_dsd_distributed(
        &mut self,
        _token_tx: Option<StreamingTokenTx>,
    ) -> Result<Option<InferenceOutput>, SwarmError> {
        Ok(None)
    }
}

#[cfg(feature = "llama")]
impl PipelineExecutor {
    /// Try the distributed multi-segment DSD path. Returns `Ok(None)` if any
    /// runtime precondition fails; caller falls back to standard
    /// `execute_distributed`.
    pub(super) async fn try_dsd_distributed(
        &mut self,
        token_tx: Option<StreamingTokenTx>,
    ) -> Result<Option<InferenceOutput>, SwarmError> {
        if !eligible(self) {
            return Ok(None);
        }

        let request_id = self.request.id;
        let max_tokens = self.request.sampling_params.max_tokens;
        let initial_gamma = self.shared_state.config.inference.speculative_gamma.max(2);
        let mut controller = GammaController::new(initial_gamma);

        // Resolve peer IDs upfront. Local segments push None and dispatch to
        // the worker subprocess in `forward_verify_through_segments`; remote
        // segments need a resolved peer_id_bytes so we can fall through
        // cleanly if any are missing.
        let peer_id_for_segment = match self.resolve_peer_id_for_segments(request_id, "DSD") {
            Some(v) => v,
            None => return Ok(None),
        };

        // Build prompt and confirm a draft model is loaded BEFORE any pipeline
        // forward — we want to fail fast and fall back cleanly. The lock is
        // released before the (mutable-self) prefill forward to avoid an
        // overlapping borrow on `self.shared_state.draft_executor`.
        let prompt = self.build_prompt().await;
        {
            let draft = self.shared_state.draft_executor.lock().await;
            if !draft.is_loaded() {
                tracing::debug!(%request_id, "DSD: draft model not loaded — falling back");
                return Ok(None);
            }
        }

        // Phase 1: standard prefill through the pipeline to produce the first
        // token AND prime every segment's KV with the prompt. We reuse the
        // existing forward_through_segments path. The first token bootstraps
        // the spec round loop.
        let prompt_bytes = prompt.as_bytes().to_vec();
        // Empty `generated_ids` for prefill — no tokens generated yet.
        let prefill_result = self
            .forward_through_segments(request_id, 0, 0, prompt_bytes.clone(), None, false, &[])
            .await?;
        if prefill_result.token_ids.is_empty() {
            return Err(SwarmError::Inference(
                "DSD: prefill returned no tokens".into(),
            ));
        }
        let first_token = prefill_result.token_ids[0];
        let (prompt_token_count, eos_tokens, decoder) = self.extract_model_cache(&prompt).await;
        let eos_set: std::collections::HashSet<u32> = eos_tokens.into_iter().collect();

        // Phase 2: re-acquire the draft lock for the rest of the request,
        // prefill the draft model, sync to target's KV state.
        let mut draft = self.shared_state.draft_executor.lock().await;
        let draft_state = tokio::task::block_in_place(|| draft_prefill(&mut draft, &prompt));
        let mut draft_state = match draft_state {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!(%request_id, error = %e, "DSD: draft prefill failed — falling back");
                return Ok(None);
            }
        };

        let mut generated: Vec<u32> = vec![first_token];
        let mut current_pos = prompt_token_count;
        let mut last_token = first_token;
        // KV length expected on every remote segment BEFORE the next forward.
        // After prefill + 0 generated forwards, remote KV = prompt_token_count
        // (the sampled first_token came out of the prefill's last logit and
        // is NOT yet written to KV — that happens on the next forward when
        // it's used as input). Mirror Item 2's baseline at speculative.rs:205.
        let mut expected_kv_len: u32 = prompt_token_count as u32;
        let mut pending_truncate: Option<u32> = None;

        super::emit_first_streaming_token(&token_tx, &decoder, first_token).await;

        let mut acceptance_proposed: u32 = 0;
        let mut acceptance_accepted: u32 = 0;
        let mut finish_reason = String::new();

        if eos_set.contains(&first_token) {
            finish_reason = "stop".to_string();
        }

        // Spec round loop.
        while finish_reason.is_empty() && (generated.len() as u32) < max_tokens {
            // Honor external cancel between rounds — same pattern as
            // execute_distributed line 174 / speculative.rs.
            if self.request.is_cancelled() {
                tracing::info!(
                    %request_id,
                    "DIAG: DSD inference cancelled externally"
                );
                finish_reason = "stop".to_string();
                break;
            }
            let remaining = max_tokens - generated.len() as u32;
            let gamma = controller.current_gamma().min(remaining).max(1);

            // SWARM-SPEC Layer 1 cascade (same pattern as single-segment
            // speculative.rs): try n-gram lookup first; on miss fall back
            // to the draft model. On hit, still sync draft KV via
            // draft_sync_tokens so subsequent rounds remain consistent.
            let ngram_drafts = ngram_lookup_drafts(
                &self.shared_state.config.inference,
                &draft_state.prompt_tokens,
                &generated,
                gamma,
            );
            let drafts = if !ngram_drafts.is_empty() {
                let sync_outcome = tokio::task::block_in_place(|| {
                    draft_sync_tokens(&mut draft_state, &mut draft, last_token, &ngram_drafts)
                });
                match sync_outcome {
                    Ok(()) => ngram_drafts,
                    Err(e) => {
                        tracing::warn!(%request_id, error = %e, "DSD: ngram-sync failed — falling back to draft sample");
                        let draft_outcome = tokio::task::block_in_place(|| {
                            draft_next_gamma(&mut draft_state, &mut draft, last_token, gamma)
                        });
                        match draft_outcome {
                            Ok(d) => d,
                            Err(e2) => {
                                tracing::warn!(%request_id, error = %e2, "DSD: draft step failed");
                                finish_reason = "stop".to_string();
                                break;
                            }
                        }
                    }
                }
            } else {
                // Draft phase — sync, llama-cpp.
                let draft_outcome = tokio::task::block_in_place(|| {
                    draft_next_gamma(&mut draft_state, &mut draft, last_token, gamma)
                });
                match draft_outcome {
                    Ok(d) => d,
                    Err(e) => {
                        tracing::warn!(%request_id, error = %e, "DSD: draft step failed");
                        finish_reason = "stop".to_string();
                        break;
                    }
                }
            };
            if drafts.is_empty() {
                break;
            }

            // verify_tokens = [bootstrap, q_1..q_γ]
            let mut verify_tokens: Vec<u32> = Vec::with_capacity(drafts.len() + 1);
            verify_tokens.push(last_token);
            verify_tokens.extend_from_slice(&drafts);

            // Multi-segment verify forward. Returns γ+1 logit vectors from
            // the LAST segment.
            let spec_logits = match super::forward_verify_through_segments(
                &self.shared_state,
                &self.network_tx,
                request_id,
                current_pos as u32,
                &self.assignment.segments,
                &peer_id_for_segment,
                &verify_tokens,
                pending_truncate,
            )
            .await
            {
                Ok(v) => v,
                Err(e) => {
                    tracing::warn!(%request_id, error = %e, "DSD: pipeline verify failed — returning partial");
                    finish_reason = "stop".to_string();
                    break;
                }
            };

            // Need γ+1 logit vectors: γ for verifying drafts[i] vs target's
            // pick at position i, plus the bonus at position γ for the
            // ALL-ACCEPTED branch. greedy_accept_reject indexes
            // `spec_logits[drafts.len()]` when all drafts accepted, so a
            // strict `< drafts.len() + 1` guard is required to avoid OOB.
            if spec_logits.len() < drafts.len() + 1 {
                tracing::warn!(
                    %request_id,
                    got = spec_logits.len(),
                    want_min = drafts.len() + 1,
                    "DSD: insufficient spec_logits — returning partial"
                );
                finish_reason = "stop".to_string();
                break;
            }

            let kv_after_forward = expected_kv_len + verify_tokens.len() as u32;

            // Greedy accept-reject — shared with Item 2 via `greedy_accept_reject`.
            let (accepted, bonus, _all_accepted) =
                super::speculative::greedy_accept_reject(&drafts, &spec_logits);

            acceptance_proposed += drafts.len() as u32;
            acceptance_accepted += accepted.len() as u32;

            let mut emitted: Vec<u32> = accepted
                .iter()
                .copied()
                .chain(std::iter::once(bonus))
                .collect();

            // BUG-FIX (R105): truncate at first EOS before any consumer sees
            // post-EOS tokens. See speculative.rs for the same fix and rationale.
            if let Some(eos_at) = emitted.iter().position(|t| eos_set.contains(t)) {
                emitted.truncate(eos_at + 1);
            }

            super::emit_streaming_batch(&token_tx, &decoder, &emitted, &mut finish_reason).await;

            // Bail before the per-round bookkeeping when the client has
            // disconnected. Mirrors speculative.rs — the inner `break` only
            // exits the streaming for-loop, leaving `controller.record_round`
            // and the synchronous `draft_sync_after_round` to run before the
            // outer `while` notices the disconnect.
            if !finish_reason.is_empty() {
                break;
            }

            generated.extend(&emitted);

            // After this round, every remote KV grew by verify_tokens.len()
            // entries. Only (accepted.len() + 1) of those are valid (the
            // bootstrap token + accepted drafts; the bonus is sampled by the
            // coordinator from the target's logits and never lands in target
            // KV via this round).
            let new_expected_kv = expected_kv_len + accepted.len() as u32 + 1;
            pending_truncate = if new_expected_kv < kv_after_forward {
                Some(new_expected_kv)
            } else {
                None
            };
            expected_kv_len = new_expected_kv;

            // Update γ controller for next round.
            controller.record_round(accepted.len() as u32, drafts.len() as u32);

            // Sync draft KV.
            tokio::task::block_in_place(|| {
                draft_sync_after_round(&mut draft_state, &mut draft, &drafts, &accepted, bonus)
            })?;

            current_pos += emitted.len();
            last_token = *emitted.last().unwrap();

            // R105's truncation at the first EOS guarantees that if `emitted`
            // contains an EOS token it must be the last element; checking
            // `last_token` is sufficient and lets us break out of the outer
            // `while` directly instead of waiting for the next iteration.
            if eos_set.contains(&last_token) {
                finish_reason = "stop".to_string();
                break;
            }
        }

        if finish_reason.is_empty() {
            finish_reason = if (generated.len() as u32) >= max_tokens {
                "length".to_string()
            } else {
                "stop".to_string()
            };
        }

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
            segments = self.assignment.segments.len(),
            proposed = acceptance_proposed,
            accepted = acceptance_accepted,
            final_gamma = controller.current_gamma(),
            accept_ema = format_args!("{:.2}", controller.accept_ema()),
            "DSD: request complete"
        );

        Ok(Some(self.finish_speculative(
            request_id,
            generated,
            &decoder,
            &eos_set,
            prompt_token_count as u32,
            finish_reason,
        )))
    }
}

// forward_verify_through_segments moved to pipeline/mod.rs (R136 Layer 1
// multi-segment) so it's reachable without the `llama` feature gate.
// DSD calls super::forward_verify_through_segments now.
