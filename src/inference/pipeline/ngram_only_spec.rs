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

/// Observed payoff of this loop, as accepted tokens per round x 100.
/// `0` means unknown — nothing has run yet and the next request finds out.
static DIST_SPEC_PAYOFF_X100: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

/// The payoff at which speculating over the network is worth what it costs.
///
/// **A round on this path is not free the way a local one is.** Every round —
/// including one where the n-gram found nothing to draft — goes back through
/// the coordinator asking for `spec_logits`, which is a full-vocabulary f32
/// vector per position: ~513 KB for a 128k-vocab model, where an ordinary
/// decode step through `forward_through_segments` returns a sampled token id in
/// four bytes and can be chained straight down the pipeline.
///
/// So a round has to return meaningfully more than the one token a plain step
/// would have returned before it pays for the vocabulary it drags back. 1.3
/// tokens per round is a deliberately modest bar: it clears a workload that is
/// copying its input, which is what this path was built for, and fails one that
/// is writing prose, which is where it was measured doing nothing at all.
///
/// Measured on the live node before this gate existed: three requests at
/// `ngram_hit_rate="0.0%"` — every round a miss, every miss a full vocabulary
/// for one token — and one at 30.5%, whose 41 miss rounds moved roughly 21 MB
/// to produce a 60-token reply.
const PAYOFF_WORTH_THE_WIRE_X100: u32 = 130;

/// Whether the loop has earned its bandwidth. Pure, so the policy is testable
/// without a pipeline.
fn payoff_justifies_the_wire(seen_x100: u32) -> bool {
    seen_x100 == 0 || seen_x100 >= PAYOFF_WORTH_THE_WIRE_X100
}

/// Fold one request's result into the running figure.
///
/// Blended rather than replaced, for the same reason every other EMA here is:
/// one short reply that happened to copy its prompt should not switch the path
/// back on for every request after it, and one that happened not to should not
/// switch it off forever. `accepted` counts tokens produced by the loop, so a
/// round that drafted nothing contributes its single fallback token — which is
/// exactly the comparison being made.
fn record_payoff(accepted_tokens: u32, rounds: u32) {
    if rounds == 0 {
        return;
    }
    let observed = (accepted_tokens as u64 * 100 / rounds as u64) as u32;
    let prev = DIST_SPEC_PAYOFF_X100.load(std::sync::atomic::Ordering::Relaxed);
    let blended = if prev == 0 {
        observed.max(1)
    } else {
        ((prev as u64 * 7 + observed as u64 * 3) / 10) as u32
    };
    DIST_SPEC_PAYOFF_X100.store(blended.max(1), std::sync::atomic::Ordering::Relaxed);
}

/// Fast-path preconditions for the draft-free n-gram-only spec loop.
fn eligible(exec: &PipelineExecutor) -> bool {
    // The LIVE config, not the boot snapshot. Every read here used to come from
    // `shared_state.config`, so turning n-gram lookup off in Settings changed
    // nothing until the daemon was restarted (gotcha #281).
    let live = exec.shared_state.cfg();
    let cfg = &live.inference;
    if !cfg.ngram_lookup_enabled {
        return false;
    }
    // Only when no draft model is configured — when one IS configured,
    // try_speculative_distributed handles the cascade with n-gram fast-path.
    if cfg.draft_model_path.is_some() {
        return false;
    }
    // Temperature is deliberately NOT a condition. It was one, because this path
    // took the target's raw argmax and a draft compared against an argmax
    // verifies nothing once sampling is on. It now samples every position through
    // the real sampler and keeps a draft only on a match, which IS the
    // speculative-sampling rejection rule for a draft with no distribution
    // behind it — see `speculative::sampled_accept_reject`.
    //
    // The gate mattered: no client asks for greedy by default (0.7 on the
    // OpenAI surface, 1.0 on the Anthropic one), so peer-served requests never
    // reached this path at all.
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
    // Has speculating over the wire actually been paying? Unknown lets one
    // request find out, and its answer steers the rest — the same shape as
    // `spec_payoff_justifies_diverting` on the local speculator, which learned
    // this lesson on 2026-08-23. This path never had it, so a workload it
    // cannot help paid the full cost on every token indefinitely.
    if !payoff_justifies_the_wire(DIST_SPEC_PAYOFF_X100.load(std::sync::atomic::Ordering::Relaxed))
    {
        tracing::debug!(
            "SWARM-SPEC L1: skipping the n-gram wire — it has not been accepting \
             enough tokens per round to pay for the logits it returns"
        );
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
        let live = self.shared_state.cfg();
        let cfg = &live.inference;
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
        // Eligibility pre-check only. The resolved list is deliberately NOT
        // kept: failover rewrites `assignment.segments[i].node_id` mid-request,
        // so a list captured here would name the failed node for every later
        // round. The send path resolves from the live segment instead.
        if self
            .resolve_peer_id_for_segments(request_id, "ngram-only")
            .is_none()
        {
            return Ok(None);
        }

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

        let live = self.shared_state.cfg();
        let cfg = &live.inference;
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
                    &verify_tokens,
                    truncate_for_this_round,
                )
                .await?;
                if spec_logits.is_empty() {
                    finish_reason = "stop".into();
                    break;
                }
                // Sample, not argmax: this is an ordinary decode step that
                // happens to have gone through the verify wire, and it must
                // honour the same sampling parameters every other step does.
                let (_, bonus, _) = super::speculative::sampled_accept_reject(
                    &[],
                    &spec_logits,
                    &self.request.sampling_params,
                    &generated,
                );
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
            let hedge_live = self.shared_state.cfg();
            let hedge_cfg = crate::inference::hedging::HedgeConfig {
                enabled: hedge_live.inference.hedge_enabled,
                after_factor: hedge_live.inference.hedge_after_factor,
                max_rate: hedge_live.inference.hedge_max_rate,
                min_samples: hedge_live.inference.hedge_min_samples,
            };
            let spec_logits = super::hedge_dispatch::forward_verify_with_hedge(
                &self.shared_state,
                &self.network_tx,
                request_id,
                current_pos as u32,
                &self.assignment.segments,
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

            let (accepted, bonus, _all) = super::speculative::sampled_accept_reject(
                &drafts,
                &spec_logits,
                &self.request.sampling_params,
                &generated,
            );
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

        record_payoff(generated.len() as u32, ngram_rounds + fallback_rounds);

        tracing::info!(
            request_id = %request_id,
            generated_tokens = generated.len(),
            ngram_rounds,
            fallback_rounds,
            payoff_x100 = DIST_SPEC_PAYOFF_X100.load(std::sync::atomic::Ordering::Relaxed),
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

#[cfg(test)]
mod payoff_tests {
    use super::{payoff_justifies_the_wire, PAYOFF_WORTH_THE_WIRE_X100};

    /// Unknown must try. A gate that refused on no evidence would never
    /// collect any, and the path would be dead on arrival for every workload
    /// including the one it helps.
    #[test]
    fn unknown_payoff_lets_one_request_find_out() {
        assert!(payoff_justifies_the_wire(0));
    }

    /// The measured failure: every round a miss, so the loop returned exactly
    /// the one token a plain decode step would have returned — while dragging
    /// back a full vocabulary to do it.
    #[test]
    fn one_token_per_round_does_not_pay_for_a_vocabulary() {
        assert!(
            !payoff_justifies_the_wire(100),
            "a round that yields one token is a plain decode step wearing a \
             513 KB hat"
        );
    }

    /// The workload this path exists for: copying its input back, several
    /// tokens accepted per round.
    #[test]
    fn a_copying_workload_still_speculates() {
        assert!(payoff_justifies_the_wire(880));
        assert!(payoff_justifies_the_wire(PAYOFF_WORTH_THE_WIRE_X100));
    }

    /// Just under the bar must be refused, or the constant is decorative.
    #[test]
    fn just_below_the_bar_is_refused() {
        assert!(!payoff_justifies_the_wire(PAYOFF_WORTH_THE_WIRE_X100 - 1));
    }
}
