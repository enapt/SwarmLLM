//! SWARM-SPEC Layer 1 (draft-free): n-gram-only distributed
//! speculative decoding.
//!
//! Unlocks the n-gram lookup cascade for users who DON'T have a draft
//! model configured — i.e. the vast majority of SwarmLLM deployments.
//! The existing `try_speculative_distributed` requires
//! `inference.draft_model_path` to be set. This path works without
//! one by tokenising locally via the standalone tokenizer cache
//! (loaded lazily from `gguf_header.bin`).
//!
//! # Flow
//!
//! 1. Build standalone tokenizer for the model. Bail if unavailable.
//! 2. Tokenise the prompt locally (target-vocab compatible).
//! 3. Do one normal prefill forward to prime remote KV + get first_token.
//! 4. For each round: try cascade_find_candidate over
//!    (prompt_tokens + generated); on HIT send the draft batch via
//!    send_verify_batch + accept-reject; on MISS send a single-token
//!    verify as the fallback path.
//! 5. Stop on EOS / max_tokens / cancel.
//!
//! # Correctness
//!
//! Greedy speculative decoding is bit-identical to greedy non-spec
//! decoding by construction: a draft token is only accepted when its
//! id matches the target's argmax at that position. N-gram lookup
//! shifts only the DRAFT SOURCE, not the verification — so this path
//! produces the same output a non-spec greedy decode would on the
//! same prompt.

use crate::error::SwarmError;
use crate::inference::router::{InferenceOutput, StreamingTokenEvent, StreamingTokenTx};

use super::PipelineExecutor;

/// Fast-path preconditions for the draft-free n-gram-only spec loop.
fn eligible(exec: &PipelineExecutor) -> bool {
    let cfg = &exec.shared_state.config.inference;
    if !cfg.ngram_lookup_enabled {
        return false;
    }
    // Only when no draft model is configured — when one IS configured,
    // try_speculative_distributed handles the cascade with n-gram fast-path.
    if cfg.draft_model_path.is_some() {
        return false;
    }
    if exec.request.sampling_params.temperature != 0.0 {
        return false;
    }
    if exec.assignment.segments.is_empty() {
        return false;
    }
    // Don't take over pure-local pipelines — execute_local handles those.
    let local_node_id = exec.shared_state.identity.node_id();
    if exec
        .assignment
        .segments
        .iter()
        .all(|s| s.node_id == *local_node_id)
    {
        return false;
    }
    if super::fastpath_request_disqualified(exec) {
        return false;
    }
    if exec
        .shared_state
        .standalone_tokenizer(&exec.request.model_id)
        .is_none()
    {
        return false;
    }
    true
}

impl PipelineExecutor {
    /// Try the draft-free n-gram-only speculative path. Returns
    /// `Ok(None)` when ineligible (caller falls back to standard
    /// non-spec loop in execute_distributed). Returns `Ok(Some(_))`
    /// on success.
    pub(super) async fn try_ngram_only_distributed(
        &mut self,
        token_tx: Option<StreamingTokenTx>,
    ) -> Result<Option<InferenceOutput>, SwarmError> {
        let cfg = &self.shared_state.config.inference;
        tracing::debug!(
            request_id = %self.request.id,
            ngram_lookup_enabled = cfg.ngram_lookup_enabled,
            draft_model_path_is_some = cfg.draft_model_path.is_some(),
            temperature = self.request.sampling_params.temperature,
            num_segments = self.assignment.segments.len(),
            "DIAG: try_ngram_only_distributed entry"
        );
        if !eligible(self) {
            tracing::debug!(
                request_id = %self.request.id,
                "DIAG: try_ngram_only_distributed ineligible — skipping"
            );
            return Ok(None);
        }
        tracing::info!(
            request_id = %self.request.id,
            "DIAG: try_ngram_only_distributed ELIGIBLE — entering n-gram-only path"
        );

        let request_id = self.request.id;
        let max_tokens = self.request.sampling_params.max_tokens;

        // Resolve per-segment peer ids upfront (None for local segments).
        let peer_id_for_segment = match self.resolve_peer_id_for_segments(request_id, "ngram-only")
        {
            Some(v) => v,
            None => return Ok(None),
        };

        // ── Tokenise prompt locally for n-gram lookup ──
        let prompt = self.build_prompt().await;
        let tokenizer = match self
            .shared_state
            .standalone_tokenizer(&self.request.model_id)
        {
            Some(t) => t,
            None => {
                return Ok(None);
            }
        };
        let prompt_tokens: Vec<u32> = tokenizer
            .encode(&prompt)
            .into_iter()
            .map(|i| i as u32)
            .collect();
        if prompt_tokens.is_empty() {
            return Ok(None);
        }

        tracing::info!(
            request_id = %request_id,
            prompt_tokens_local = prompt_tokens.len(),
            num_segments = self.assignment.segments.len(),
            "SWARM-SPEC L1 ngram-only: starting"
        );

        // ── Phase 1: prefill via standard multi-segment forward ──
        // Uses existing forward_through_segments which handles N-segment
        // pipelines + local-vs-remote dispatch + retries.
        let prompt_bytes = prompt.as_bytes().to_vec();
        let prefill_result = self
            .forward_through_segments(request_id, 0, 0, prompt_bytes, None, false, &[])
            .await?;
        if prefill_result.token_ids.is_empty() {
            return Err(SwarmError::Inference(
                "ngram-only: prefill returned no tokens".into(),
            ));
        }
        let first_token = prefill_result.token_ids[0];
        let (_extract_ptc, eos_tokens, decoder) = self.extract_model_cache(&prompt).await;
        let eos_tokens: std::collections::HashSet<u32> = eos_tokens.into_iter().collect();
        // Use the standalone tokenizer's count (consistent with the n-gram
        // history and with the remote worker, since both derive from the
        // same gguf_header.bin). extract_model_cache may estimate when
        // the local split_model entry isn't populated (all-remote pipelines).
        let prompt_token_count = prompt_tokens.len();

        // ── Phase 2: decode loop ──
        let mut generated: Vec<u32> = vec![first_token];
        // After prefill: remote KV holds `prompt_token_count` positions
        // (0..prompt_token_count-1). The first verify batch ships
        // `[last_token, drafts...]` and the remote appends each to KV,
        // growing it by `verify_tokens.len()` per round.
        let mut current_pos = prompt_token_count;
        let mut last_token = first_token;
        let mut finish_reason = String::new();
        let mut ngram_rounds: u32 = 0;
        let mut fallback_rounds: u32 = 0;

        // Stream first token
        super::emit_first_streaming_token(&token_tx, &decoder, first_token).await;
        if eos_tokens.contains(&first_token) {
            finish_reason = "stop".into();
        }

        let cfg = &self.shared_state.config.inference;
        let max_draft = cfg.ngram_num_pred_tokens.min(cfg.speculative_gamma);
        // SWARM-SPEC L1: pending truncate carries over to the NEXT verify
        // call. When a verify round partially rejects drafts, the remote
        // KV still holds k+1 positions from this round, but only
        // accepted+1 are valid. The NEXT verify must include
        // truncate_kv_to=Some(valid_pos) so the worker rewinds its KV
        // before applying the new tokens. Mirrors DSD's pattern.
        let mut pending_truncate: Option<u32> = None;

        while finish_reason.is_empty() && (generated.len() as u32) < max_tokens {
            if self.request.is_cancelled() {
                finish_reason = "stop".into();
                break;
            }
            let remaining = max_tokens - generated.len() as u32;
            let this_gamma = max_draft.min(remaining).max(1);

            // ── N-gram cascade ──
            let drafts = super::speculative::ngram_lookup_drafts(
                cfg,
                &prompt_tokens,
                &generated,
                this_gamma,
            );

            if drafts.is_empty() {
                // ── Fallback: single-token verify (γ=0 batch = 1 position) ──
                fallback_rounds += 1;
                // R137: bump lifetime miss counter for /api/admin/stats →
                // swarm_spec.ngram visibility.
                self.shared_state
                    .metrics
                    .ngram_misses
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let verify_tokens = vec![last_token];
                let truncate_for_this_round = pending_truncate.take();
                let spec_logits = super::forward_verify_through_segments(
                    &self.shared_state,
                    &self.network_tx,
                    request_id,
                    current_pos as u32,
                    &self.assignment.segments,
                    &peer_id_for_segment,
                    &verify_tokens,
                    truncate_for_this_round,
                )
                .await?;
                if spec_logits.is_empty() {
                    finish_reason = "stop".into();
                    break;
                }
                let bonus = super::speculative::argmax(&spec_logits[0]);
                last_token = bonus;
                generated.push(bonus);
                current_pos += 1;
                if let Some(ref tx) = token_tx {
                    let text = decoder.decode_tokens(&[bonus]);
                    let _ = tx
                        .send(StreamingTokenEvent {
                            text,
                            finish_reason: None,
                            matched_stop_sequence: None,
                        })
                        .await;
                }
                if eos_tokens.contains(&bonus) {
                    finish_reason = "stop".into();
                    break;
                }
                continue;
            }

            // ── N-gram HIT path ──
            ngram_rounds += 1;
            // R137: bump lifetime hit counter (counts hit-rounds, not
            // accepted tokens — accept count is in the spec_logits result).
            self.shared_state
                .metrics
                .ngram_hits
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let mut verify_tokens: Vec<u32> = Vec::with_capacity(drafts.len() + 1);
            verify_tokens.push(last_token);
            verify_tokens.extend_from_slice(&drafts);

            let truncate_for_this_round = pending_truncate.take();
            // SWARM-SPEC Layer 2: route through hedge wrapper. On
            // single-segment pipelines with hedge_enabled, races the
            // primary against a duplicate to alt holder; on
            // multi-segment, falls back to direct forward.
            let hedge_key = crate::inference::hedging::HedgeKey {
                model_id: self.request.model_id.clone(),
                segment_idx: 0,
                holder: self.assignment.segments[0].node_id.clone(),
            };
            let hedge_cfg = crate::inference::hedging::HedgeConfig {
                enabled: self.shared_state.config.inference.hedge_enabled,
                after_factor: self.shared_state.config.inference.hedge_after_factor,
                max_rate: self.shared_state.config.inference.hedge_max_rate,
                min_samples: self.shared_state.config.inference.hedge_min_samples,
            };
            let spec_logits = super::hedge_dispatch::forward_verify_with_hedge(
                &self.shared_state,
                &self.network_tx,
                request_id,
                current_pos as u32,
                &self.assignment.segments,
                &peer_id_for_segment,
                &verify_tokens,
                truncate_for_this_round,
                hedge_key,
                hedge_cfg,
            )
            .await?;
            if spec_logits.len() < drafts.len() + 1 {
                tracing::warn!(
                    %request_id,
                    got = spec_logits.len(),
                    want = drafts.len() + 1,
                    "ngram-only: insufficient spec_logits, returning partial"
                );
                break;
            }

            let (accepted, bonus, _all) =
                super::speculative::greedy_accept_reject(&drafts, &spec_logits);
            let mut emitted: Vec<u32> = accepted
                .iter()
                .copied()
                .chain(std::iter::once(bonus))
                .collect();
            if let Some(eos_at) = emitted.iter().position(|t| eos_tokens.contains(t)) {
                emitted.truncate(eos_at + 1);
            }

            // Emit (stream + accumulate)
            super::emit_streaming_batch(&token_tx, &decoder, &emitted, &mut finish_reason).await;
            for &t in &emitted {
                generated.push(t);
                if eos_tokens.contains(&t) {
                    finish_reason = "stop".into();
                    break;
                }
            }
            // Remote KV grew by verify_tokens.len() positions. If we
            // partially rejected (emitted.len() < verify_tokens.len()),
            // the trailing rejected positions hold incorrect content
            // (the rejected drafts ran through the forward but our
            // coordinator state diverged). Set pending_truncate to the
            // first invalid position so the NEXT verify call rewinds the
            // remote KV before applying new tokens — same pattern as DSD.
            current_pos += verify_tokens.len();
            let valid_kv_len = (current_pos - verify_tokens.len()) + emitted.len();
            if emitted.len() < verify_tokens.len() {
                pending_truncate = Some(valid_kv_len as u32);
                // After truncate fires, current_pos will reflect the
                // actual remote KV length. Adjust now so the next
                // iteration's index_pos matches.
                current_pos = valid_kv_len;
            }
            last_token = *emitted.last().unwrap_or(&last_token);

            if (generated.len() as u32) >= max_tokens {
                finish_reason = "length".into();
            }
        }

        if finish_reason.is_empty() {
            finish_reason = "length".into();
        }

        tracing::info!(
            request_id = %request_id,
            generated_tokens = generated.len(),
            ngram_rounds,
            fallback_rounds,
            ngram_hit_rate = format!(
                "{:.1}%",
                100.0 * ngram_rounds as f32 / (ngram_rounds + fallback_rounds).max(1) as f32
            ),
            "SWARM-SPEC L1 ngram-only: complete"
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
}
