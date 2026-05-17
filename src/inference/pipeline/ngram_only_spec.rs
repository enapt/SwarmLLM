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
use crate::types::{LayerForward, NetworkCommand, TensorFormat};

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
    if exec.assignment.segments.len() != 1 {
        return false;
    }
    let local_node_id = exec.shared_state.identity.node_id();
    if exec.assignment.segments[0].node_id == *local_node_id {
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
        if !eligible(self) {
            return Ok(None);
        }

        let request_id = self.request.id;
        let max_tokens = self.request.sampling_params.max_tokens;
        let segment = self.assignment.segments[0].clone();
        let target_peer_bytes = self
            .shared_state
            .resolve_peer_id_bytes(&segment.node_id)
            .ok_or_else(|| {
                SwarmError::Network(format!("No peer_id_bytes for {}", segment.node_id))
            })?;

        // ── Tokenise prompt locally for n-gram lookup ──
        let prompt = self.build_prompt().await;
        let tokenizer = match self
            .shared_state
            .standalone_tokenizer(&self.request.model_id)
        {
            Some(t) => t,
            None => {
                // Should be unreachable given eligibility; defensive fallthrough.
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
            target_peer = %segment.node_id,
            "SWARM-SPEC L1 ngram-only: starting"
        );

        // ── Phase 1: prefill (standard forward) ──
        let prompt_bytes = prompt.as_bytes().to_vec();
        let prompt_byte_len = prompt_bytes.len();
        let num_layers = segment.layer_range.1 - segment.layer_range.0;
        let (first_token, prompt_token_count, eos_tokens, decoder) = {
            let (rx, mut prefill_guard) = super::register_pending_layer_result(
                &self.shared_state.pending_layer_results,
                request_id,
            )?;
            let forward = LayerForward {
                request_id,
                sequence_num: 0,
                index_pos: 0,
                activations: prompt_bytes,
                format: TensorFormat::FP32,
                model_id: self.request.model_id.clone(),
                layer_range: segment.layer_range,
                tp_meta: None,
                vision_embeddings: None,
                sender_peer_bytes: None,
                requester_node_id: Some(self.shared_state.identity.node_id().0),
                pre_embedded: false,
                generated_ids: Vec::new(),
                adapter_id: None,
                draft_tokens: Vec::new(),
                spec_logits_requested: false,
                truncate_kv_to: None,
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
                return Err(SwarmError::Network("ngram-only prefill send failed".into()));
            }
            let prefill_result = Self::wait_for_result(
                rx,
                request_id,
                0,
                &segment.node_id,
                num_layers,
                prompt_byte_len,
            )
            .await?;
            prefill_guard.disarm();
            if prefill_result.token_ids.is_empty() {
                return Err(SwarmError::Inference(
                    "ngram-only: prefill returned no tokens".into(),
                ));
            }
            let first_token = prefill_result.token_ids[0];
            let (ptc, eos, decoder) = self.extract_model_cache(&prompt).await;
            let eos_set: std::collections::HashSet<u32> = eos.into_iter().collect();
            (first_token, ptc, eos_set, decoder)
        };

        // ── Phase 2: decode loop ──
        let mut generated: Vec<u32> = vec![first_token];
        let mut current_pos = prompt_token_count;
        let mut last_token = first_token;
        let mut finish_reason = String::new();
        let mut ngram_rounds: u32 = 0;
        let mut fallback_rounds: u32 = 0;

        // Stream first token
        if let Some(ref tx) = token_tx {
            let text = decoder.decode_tokens(&[first_token]);
            let _ = tx
                .send(StreamingTokenEvent {
                    text,
                    finish_reason: None,
                    matched_stop_sequence: None,
                })
                .await;
        }
        if eos_tokens.contains(&first_token) {
            finish_reason = "stop".into();
        }

        let cfg = &self.shared_state.config.inference;
        let max_draft = cfg.ngram_num_pred_tokens.min(cfg.speculative_gamma);

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
                let verify_tokens = vec![last_token];
                let spec_logits = super::speculative::send_verify_batch(
                    &self.shared_state,
                    &self.network_tx,
                    request_id,
                    current_pos as u32,
                    &segment,
                    &target_peer_bytes,
                    &verify_tokens,
                    None,
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
            let mut verify_tokens: Vec<u32> = Vec::with_capacity(drafts.len() + 1);
            verify_tokens.push(last_token);
            verify_tokens.extend_from_slice(&drafts);

            let spec_logits = super::speculative::send_verify_batch(
                &self.shared_state,
                &self.network_tx,
                request_id,
                current_pos as u32,
                &segment,
                &target_peer_bytes,
                &verify_tokens,
                None,
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
            if let Some(ref tx) = token_tx {
                for &t in &emitted {
                    let text = decoder.decode_tokens(&[t]);
                    if tx
                        .send(StreamingTokenEvent {
                            text,
                            finish_reason: None,
                            matched_stop_sequence: None,
                        })
                        .await
                        .is_err()
                    {
                        // Client disconnected
                        finish_reason = "stop".into();
                        break;
                    }
                }
            }
            for &t in &emitted {
                generated.push(t);
                if eos_tokens.contains(&t) {
                    finish_reason = "stop".into();
                    break;
                }
            }
            // Remote KV grew by verify_tokens.len() positions
            current_pos += verify_tokens.len();
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
