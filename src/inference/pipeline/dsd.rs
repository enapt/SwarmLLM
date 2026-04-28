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
//! - All segments remote (a local segment in the pipeline would need a
//!   different code path — future work)
//! - No vision, LoRA, or encryption
//!
//! # Correctness
//!
//! Greedy speculative decoding produces output bit-identical to greedy
//! non-speculative decoding because we only accept draft tokens whose ID
//! matches the target's argmax at the same position. Regressions show up as
//! output divergence on a fixed prompt with `temperature=0`.

#[cfg(feature = "llama")]
use std::sync::Arc;

#[cfg(feature = "llama")]
use crate::daemon::SharedState;
use crate::error::SwarmError;
#[cfg(feature = "llama")]
use crate::inference::router::StreamingTokenEvent;
use crate::inference::router::{InferenceOutput, StreamingTokenTx};
#[cfg(feature = "llama")]
use crate::types::{LayerForward, NetworkCommand, NetworkFinishReason, TensorFormat};
#[cfg(feature = "llama")]
use tokio::sync::mpsc;

#[cfg(feature = "llama")]
use super::speculative::{draft_next_gamma, draft_prefill, draft_sync_after_round};
use super::PipelineExecutor;
#[cfg(feature = "llama")]
use super::MAX_PENDING_LAYER_RESULTS;
#[cfg(feature = "llama")]
use crate::inference::dsd_controller::GammaController;

/// Fast-path preconditions for the DSD coordinator loop.
#[cfg(feature = "llama")]
pub(super) fn eligible(exec: &PipelineExecutor) -> bool {
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
    // All segments must be remote — see ARCHITECTURE.md § Deferred Items.
    let local_node_id = exec.shared_state.identity.node_id();
    if exec
        .assignment
        .segments
        .iter()
        .any(|s| s.node_id == *local_node_id)
    {
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

        // Resolve all peer IDs upfront. If any segment's peer can't be
        // located, fall through cleanly.
        let mut peer_id_for_segment: Vec<Vec<u8>> =
            Vec::with_capacity(self.assignment.segments.len());
        for segment in &self.assignment.segments {
            match self.shared_state.resolve_peer_id_bytes(&segment.node_id) {
                Some(p) => peer_id_for_segment.push(p),
                None => {
                    tracing::debug!(%request_id, node = %segment.node_id, "DSD: missing peer_id_bytes — falling back");
                    return Ok(None);
                }
            }
        }

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
        let prefill_result = self
            .forward_through_segments(request_id, 0, 0, prompt_bytes.clone(), None, false)
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

        if let Some(ref tx) = token_tx {
            let text = decoder.decode_tokens(&[first_token]);
            let _ = tx
                .send(StreamingTokenEvent {
                    text,
                    finish_reason: None,
                })
                .await;
        }

        let mut acceptance_proposed: u32 = 0;
        let mut acceptance_accepted: u32 = 0;
        let mut finish_reason = String::new();

        if eos_set.contains(&first_token) {
            finish_reason = "stop".to_string();
        }

        // Spec round loop.
        while finish_reason.is_empty() && (generated.len() as u32) < max_tokens {
            let remaining = max_tokens - generated.len() as u32;
            let gamma = controller.current_gamma().min(remaining).max(1);

            // Draft phase — sync, llama-cpp.
            let draft_outcome = tokio::task::block_in_place(|| {
                draft_next_gamma(&mut draft_state, &mut draft, last_token, gamma)
            });
            let drafts = match draft_outcome {
                Ok(d) => d,
                Err(e) => {
                    tracing::warn!(%request_id, error = %e, "DSD: draft step failed");
                    finish_reason = "stop".to_string();
                    break;
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
            let spec_logits = match forward_verify_through_segments(
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

            let emitted: Vec<u32> = accepted
                .iter()
                .copied()
                .chain(std::iter::once(bonus))
                .collect();

            if let Some(ref tx) = token_tx {
                for &t in &emitted {
                    let text = decoder.decode_tokens(&[t]);
                    if tx
                        .send(StreamingTokenEvent {
                            text,
                            finish_reason: None,
                        })
                        .await
                        .is_err()
                    {
                        finish_reason = "stop".to_string();
                        break;
                    }
                }
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

            for t in &emitted {
                if eos_set.contains(t) {
                    finish_reason = "stop".to_string();
                    break;
                }
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

        let clean: Vec<u32> = generated
            .into_iter()
            .filter(|t| !eos_set.contains(t))
            .collect();
        let completion_tokens = clean.len() as u32;
        let content = decoder.decode_tokens(&clean);
        Ok(Some(InferenceOutput {
            request_id,
            content,
            prompt_tokens: prompt_token_count as u32,
            completion_tokens,
            finish_reason,
            session_id: self.request.session_id.clone(),
            token_logprobs: vec![],
        }))
    }
}

/// Propagate a γ-token verify request through every pipeline segment in
/// order. The first segment receives `verify_tokens` encoded as i64 LE bytes
/// (`8 × verify_tokens.len()` bytes). Intermediate segments receive the
/// previous segment's `[1, γ, hidden]` activations. The final segment
/// returns γ+1 logit vectors via `LayerResult.spec_logits`. Every
/// `LayerForward` carries `truncate_kv_to`, `draft_tokens` (informational),
/// and `spec_logits_requested = true` — only the last segment actually emits
/// `spec_logits` (the worker gates emission on `is_last`).
#[cfg(feature = "llama")]
#[allow(clippy::too_many_arguments)]
async fn forward_verify_through_segments(
    shared_state: &Arc<SharedState>,
    network_tx: &mpsc::Sender<NetworkCommand>,
    request_id: uuid::Uuid,
    index_pos: u32,
    segments: &[crate::types::PipelineSegment],
    peer_id_for_segment: &[Vec<u8>],
    verify_tokens: &[u32],
    truncate_kv_to: Option<u32>,
) -> Result<Vec<Vec<f32>>, SwarmError> {
    let num_segments = segments.len();
    debug_assert_eq!(num_segments, peer_id_for_segment.len());

    // First-segment activations: γ+1 token IDs as i64 LE bytes.
    let mut activation_bytes: Vec<u8> = Vec::with_capacity(verify_tokens.len() * 8);
    for &t in verify_tokens {
        activation_bytes.extend_from_slice(&(t as i64).to_le_bytes());
    }

    for (idx, segment) in segments.iter().enumerate() {
        let is_last = idx == num_segments - 1;
        let target_peer_bytes = &peer_id_for_segment[idx];

        if shared_state.pending_layer_results.len() >= MAX_PENDING_LAYER_RESULTS {
            return Err(SwarmError::ServiceUnavailable(
                "DSD: pipeline overloaded — too many pending layer results".into(),
            ));
        }
        let (tx, rx) = tokio::sync::oneshot::channel();
        shared_state.pending_layer_results.insert(request_id, tx);

        // RAII: ensure the pending_layer_results entry is removed on every
        // exit path from this iteration, including `?` propagation from
        // wait_for_result. Without this, a non-final-segment timeout/network
        // error leaves a permanent stale entry that consumes capacity (the
        // MAX_PENDING_LAYER_RESULTS check at the loop head would fail under
        // load) and silently swallows any late-arriving response.
        struct PendingGuard<'a> {
            map: &'a dashmap::DashMap<
                uuid::Uuid,
                tokio::sync::oneshot::Sender<crate::types::LayerResult>,
            >,
            id: uuid::Uuid,
            armed: bool,
        }
        impl<'a> PendingGuard<'a> {
            fn disarm(&mut self) {
                self.armed = false;
            }
        }
        impl<'a> Drop for PendingGuard<'a> {
            fn drop(&mut self) {
                if self.armed {
                    self.map.remove(&self.id);
                }
            }
        }
        let mut pending_guard = PendingGuard {
            map: &shared_state.pending_layer_results,
            id: request_id,
            armed: true,
        };

        let forward = LayerForward {
            request_id,
            sequence_num: 1, // not prefill
            index_pos,
            activations: activation_bytes.clone(),
            format: TensorFormat::FP32,
            model_id: segment.shard_id.model_id.clone(),
            layer_range: segment.layer_range,
            vision_embeddings: None,
            sender_peer_bytes: None,
            tp_meta: None,
            requester_node_id: Some(shared_state.identity.node_id().0),
            pre_embedded: false,
            adapter_id: None,
            // Carry draft_tokens informationally so the receiver can log/trace
            // the verify request, even though the input shape is sourced from
            // `activations` after Phase 1.
            draft_tokens: verify_tokens.to_vec(),
            // Only the last segment will actually populate spec_logits — but
            // setting the flag uniformly makes the protocol symmetric.
            spec_logits_requested: true,
            truncate_kv_to,
        };

        if network_tx
            .send(NetworkCommand::SendTensor {
                target_peer_bytes: target_peer_bytes.clone(),
                forward,
            })
            .await
            .is_err()
        {
            shared_state.pending_layer_results.remove(&request_id);
            return Err(SwarmError::Network("DSD: verify send dropped".into()));
        }

        let num_layers = segment.layer_range.1 - segment.layer_range.0;
        let result = PipelineExecutor::wait_for_result(
            rx,
            request_id,
            idx,
            &segment.node_id,
            num_layers,
            activation_bytes.len(),
        )
        .await?;

        if let Some(NetworkFinishReason::Error(msg)) = &result.finish_reason {
            return Err(SwarmError::Inference(format!("DSD segment {idx}: {msg}")));
        }

        if is_last {
            if result.spec_logits.is_empty() {
                return Err(SwarmError::Inference(
                    "DSD: last segment returned no spec_logits".into(),
                ));
            }
            return Ok(result.spec_logits);
        }

        // Intermediate: feed the segment's hidden state output to the next.
        // SEC: Validate that the returned activation byte length matches what we
        // sent in. Transformer layers preserve [seq, hidden] shape; a malicious
        // peer returning a different length would crash the next worker (gotcha #20).
        if result.activations.len() != activation_bytes.len() {
            return Err(SwarmError::Inference(format!(
                "DSD segment {idx} returned wrong activation shape: got {} bytes, expected {}",
                result.activations.len(),
                activation_bytes.len()
            )));
        }
        activation_bytes = result.activations;
    }
    unreachable!("loop returns on the last segment")
}
