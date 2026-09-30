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
//! - A drafter: llama.cpp's from `draft_model_path` (the `llama` build), or a
//!   small model this node holds that shares the target's vocabulary
//!   (`inference.draft_model`, run in this engine — `pipeline::engine_drafter`)
//! - No vision or LoRA
//!
//! # Correctness
//!
//! A draft is kept only while the TARGET's own sample, drawn with the
//! caller's sampling parameters, equals it — SpecExec's walk
//! (arXiv 2406.02532), run by `sampling::sampled_accept_reject` at the last
//! segment or, for a tail that cannot walk, here on the logits it returns.
//! The drafter proposes its argmax: a draft with no distribution behind it,
//! for which accepting with probability `p(x)` and otherwise drawing from the
//! remainder IS the speculative-sampling rule. So this path is exact at any
//! temperature; it was greedy-only while it compared argmaxes.
//!
//! **That is not the same as being bit-identical to non-speculative decoding,
//! and this comment used to claim it was** (gotcha #370). A verify forward
//! computes several positions at once and reassociates, so the logits differ
//! in their last bits; where two candidates are near-tied that can flip an
//! argmax and the sampled token with it. Rare, expected, and not a bug.
//!
//! So do NOT treat divergence on a fixed prompt at `temperature = 0` as a
//! regression signal on its own — that advice was here and it points at
//! behaviour the design permits. A real regression is a distributional one
//! or a systematic drift, not one token in a long reply.

use crate::error::SwarmError;
use crate::inference::router::StreamingTokenEvent;
use crate::inference::router::{InferenceOutput, StreamingTokenTx};

use super::engine_drafter::{engine_drafter_for, EngineDrafter};
use super::speculative::{
    draft_next_gamma, draft_prefill, draft_sync_after_round, draft_sync_tokens,
    ngram_lookup_drafts, DraftState,
};
use super::PipelineExecutor;
use crate::inference::dsd_controller::{
    best_gamma_for_check, AcceptanceEstimate, CheckCost, BEST_GAMMA_MAX,
};

/// Which model guesses for this request.
enum Drafter {
    /// llama.cpp from `inference.draft_model_path` — a build with the `llama`
    /// feature, holding the draft executor for the whole request.
    Llama {
        exec: tokio::sync::OwnedMutexGuard<crate::inference::executor::ModelExecutor>,
        state: DraftState,
    },
    /// A small model this node holds, run in this engine by its own worker
    /// (`engine_drafter`, `inference.draft_model`).
    Engine(EngineDrafter),
}

/// Fast-path preconditions for the DSD coordinator loop, short of which
/// drafter will guess — that is decided in the loop, where the llama.cpp
/// executor can be asked whether it holds a model and this node's held models
/// searched for one that can draft (`engine_drafter_for`).
fn eligible(exec: &PipelineExecutor) -> bool {
    let cfg = exec.shared_state.cfg();
    cfg.inference.decentralized_spec_decoding
        // Any temperature: acceptance walks with the caller's sampler.
        && super::speculation_allowed(exec)
        // Multi-segment pipeline only — single-segment falls through to Item 2.
        && exec.assignment.segments.len() >= 2
}

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
        let initial_gamma = self.shared_state.cfg().inference.guesses_per_check().max(2);
        // γ is chosen each round from MEASURED costs (Leviathan et al. §3.4):
        // the verify's cost as a line in the positions it reads — its round
        // trip, plus what the far layers charge per position, which is ~0 on a
        // card and dominant on a processor (`CheckCost`) — and the time per
        // drafted guess, against the running acceptance. A long link makes long
        // runs pay; a free round trip, or a far node checking on its processor,
        // makes them waste. The multiplicative controller this replaced could
        // never leave γ = 4; the constant-cost one after it ran a processor
        // check at γ = 14-16 and 1.1-2.8 s a round.
        //
        // A request starts from what the last one on the same model and
        // machines learned (`dsd_controller::recall`) — including γ = 0, "plain
        // rounds", where guessing does not pay — rather than re-proving it from
        // the configured length every time.
        let learned_key = format!(
            "{}|{}",
            self.request.model_id.0,
            self.assignment
                .segments
                .iter()
                .map(|s| s.node_id.to_string())
                .collect::<Vec<_>>()
                .join(",")
        );
        let recalled = crate::inference::dsd_controller::recall(&learned_key);
        if recalled.as_ref().is_some_and(|l| {
            crate::inference::dsd_controller::steps_aside(l, std::time::Instant::now())
        }) {
            tracing::debug!(
                %request_id,
                "DSD: guessing did not pay on these machines last time — the ordinary loop runs this request"
            );
            return Ok(None);
        }
        let (mut acceptance, mut check, mut draft_cost, mut gamma_now) = match recalled {
            Some(l) => (
                l.acceptance,
                l.check,
                crate::inference::dsd_controller::RecentMedian::seeded(l.draft_ms_each),
                l.gamma,
            ),
            None => (
                AcceptanceEstimate::new(),
                CheckCost::default(),
                crate::inference::dsd_controller::RecentMedian::default(),
                initial_gamma,
            ),
        };

        // Resolve peer IDs upfront. Local segments push None and dispatch to
        // the worker subprocess in `forward_verify_through_segments`; remote
        // segments need a resolved peer_id_bytes so we can fall through
        // cleanly if any are missing.
        // Eligibility pre-check only. The resolved list is deliberately NOT
        // kept: failover rewrites `assignment.segments[i].node_id` mid-request,
        // so a list captured here would name the failed node for every later
        // round. The send path resolves from the live segment instead.
        if self
            .resolve_peer_id_for_segments(request_id, "DSD")
            .is_none()
        {
            return Ok(None);
        }

        // Build prompt and settle WHICH drafter guesses BEFORE any pipeline
        // forward — we want to fail fast and fall back cleanly. llama.cpp's,
        // where the build can load `draft_model_path`; otherwise a model this
        // node holds (`inference.draft_model`), which also needs the prompt as
        // the TARGET tokenized it, since its guesses are checked as ids.
        let prompt = self.build_prompt().await;
        let llama_loaded = self.shared_state.cfg().inference.draft_model_path.is_some()
            && self.shared_state.draft_executor.lock().await.is_loaded();
        let engine = if llama_loaded {
            None
        } else {
            let target = self.request.model_id.clone();
            let Some(spec) = engine_drafter_for(&self.shared_state, &target) else {
                tracing::debug!(%request_id, "DSD: no drafter available — falling back");
                return Ok(None);
            };
            let Some(tokenizer) = self.shared_state.standalone_tokenizer(&target) else {
                tracing::debug!(%request_id, "DSD: no tokenizer for the target — falling back");
                return Ok(None);
            };
            let ids: Vec<u32> = tokenizer
                .encode(&prompt)
                .into_iter()
                .map(|t| t as u32)
                .collect();
            Some((spec, ids))
        };

        // Our engine's drafter reads the prompt while the target does, not
        // after it — see `engine_drafter::read_ahead` — but only once this
        // node's own segments of the target are loaded. The drafter is a guest
        // (`process_pool::Tenancy::Guest`): it takes graphics memory only as it
        // stands free, and on a cold start that is exactly the memory those
        // segments are about to be loaded into. Loaded first, it would push
        // them onto the processor; waiting, it takes what they leave. The first
        // round then reads the prompt itself.
        let target_segments_loaded = {
            let me = self.shared_state.identity.node_id();
            let pool = &self.shared_state.model_process_pool;
            self.assignment
                .segments
                .iter()
                .filter(|s| s.node_id == *me)
                .all(|s| pool.holds_segment(&self.request.model_id, s.layer_range))
        };
        // The attempt's own key for the drafter's cache — never the request's
        // id, which a router retry reuses (`engine_drafter::draft_key`).
        let draft_key = super::engine_drafter::draft_key();
        let read_ahead = engine
            .as_ref()
            .filter(|_| target_segments_loaded)
            .map(|(spec, ids)| {
                super::engine_drafter::read_ahead(
                    self.shared_state.clone(),
                    spec,
                    draft_key,
                    ids.clone(),
                    &self.request.sampling_params,
                    self.request.cancel.clone(),
                )
            });

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

        // Phase 2: the drafter reads the prompt. llama.cpp re-acquires its lock
        // for the rest of the request and prefills now; our engine's drafter
        // reads it with its first round, together with the first reply token.
        let (mut drafter, prompt_tokens) = match engine {
            Some((spec, ids)) => {
                if ids.len() != prompt_token_count {
                    // The drafter's positions would then disagree with the
                    // target's — the shared noise is keyed by position.
                    tracing::warn!(
                        %request_id,
                        drafter_tokens = ids.len(),
                        target_tokens = prompt_token_count,
                        "DSD: the drafter's prompt is not the target's — falling back"
                    );
                    return Ok(None);
                }
                let mut e = EngineDrafter::new(
                    spec,
                    draft_key,
                    ids.clone(),
                    super::engine_drafter::release_of(self.shared_state.clone(), draft_key),
                );
                if let Some(h) = read_ahead {
                    // On failure the first round reads the prompt instead; a
                    // drafter that cannot read at all is caught there, with
                    // the fallback.
                    match h.finish().await {
                        None => {}
                        Some(Ok(Ok(_))) => e.prompt_read(),
                        Some(Ok(Err(err))) => tracing::debug!(
                            %request_id,
                            error = %err,
                            "DSD: the drafter's read-ahead failed — the first round reads the prompt"
                        ),
                        Some(Err(err)) => tracing::debug!(
                            %request_id,
                            error = %err,
                            "DSD: the drafter's read-ahead task ended — the first round reads the prompt"
                        ),
                    }
                }
                e.push(&[first_token]);
                (Drafter::Engine(e), ids)
            }
            None => {
                let mut exec = self.shared_state.draft_executor.clone().lock_owned().await;
                let prefilled = tokio::task::block_in_place(|| draft_prefill(&mut exec, &prompt));
                match prefilled {
                    Ok(state) => {
                        let prompt_tokens = state.prompt_tokens.clone();
                        (Drafter::Llama { exec, state }, prompt_tokens)
                    }
                    Err(e) => {
                        tracing::warn!(%request_id, error = %e, "DSD: draft prefill failed — falling back");
                        return Ok(None);
                    }
                }
            }
        };
        let drafter_name = match &drafter {
            Drafter::Llama { .. } => "llama.cpp".to_string(),
            Drafter::Engine(e) => e.model_id().0.clone(),
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

        super::emit_first_streaming_token(
            &self.partial_reply,
            &token_tx,
            &decoder,
            first_token,
            &eos_set,
        )
        .await;

        // Shared noise (`inference::coupled_noise`, split_speculation.md Phase 4c):
        // the drafter draws each guess with the same noise the sampling segment
        // will use, so a close drafter reproduces the target's SAMPLE, not just
        // its argmax — at temperature 1.0 a 3-bit copy of a 7B's far half agreed
        // 92.5% of the time this way against 71.3% for a fixed guess. Greedy
        // requests need none: both sides take the argmax.
        // `SWARMLLM_SPEC_COUPLING=0` drafts fixed guesses instead, for an A/B
        // inside one binary.
        let noise = (self.request.sampling_params.temperature > 0.0
            && std::env::var("SWARMLLM_SPEC_COUPLING").as_deref() != Ok("0"))
        .then(|| crate::inference::coupled_noise::CoupledNoise::new(rand::random::<u64>()));

        let mut acceptance_proposed: u32 = 0;
        let mut acceptance_accepted: u32 = 0;
        let mut finish_reason = String::new();
        // Set when the drafter fails: the reply is finished with rounds that
        // guess nothing — a check of the one token already sampled, which is a
        // plain decode step through the same path — rather than ended where the
        // drafter broke. A zero-guess round is walked at the tail like any
        // other (`walk_verified_positions` with no drafts answers the sample),
        // and an older tail answers one row of logits, which `accept` reads.
        let mut drafting_off = false;
        // The drafter's FIRST call in a request is not a guess's cost: it reads
        // the prompt (when the read-ahead did not) and whatever it missed, and
        // on a cold worker loads — measured 902 ms "per guess" on the rig
        // (2026-09-29), which, remembered across requests and never re-measured
        // while γ sat at 0, would have kept guessing off even where it pays.
        let mut drafter_warm = false;

        if eos_set.contains(&first_token) {
            finish_reason = "stop".to_string();
        }

        // The continuous stream (`dsd_stream`): where every segment but the far
        // one is this node's own and that peer serves a stream, the next chunk
        // of guesses goes out while the last is still being checked, instead
        // of the rounds below waiting a whole round trip between them.
        // Opt-in until measured over a real link: `SWARMLLM_SPEC_STREAM=1`.
        let mut streamed = false;
        if let Drafter::Engine(engine) = &mut drafter {
            let tail = super::dsd_stream::stream_requested()
                .then(|| {
                    super::dsd_stream::stream_tail(&self.shared_state, &self.assignment.segments)
                        .cloned()
                })
                .flatten();
            if let Some(tail) = tail.filter(|_| finish_reason.is_empty()) {
                self.stream_checks(
                    tail,
                    engine,
                    current_pos as u32,
                    super::dsd_stream::StreamIo {
                        token_tx: &token_tx,
                        decoder: &decoder,
                        eos: &eos_set,
                        noise,
                        max_tokens,
                    },
                    super::dsd_stream::StreamReply {
                        generated: &mut generated,
                        finish_reason: &mut finish_reason,
                        acceptance: &mut acceptance,
                        check: &mut check,
                        draft_cost: &mut draft_cost,
                        proposed: &mut acceptance_proposed,
                        accepted: &mut acceptance_accepted,
                    },
                )
                .await?;
                streamed = true;
            }
        }

        // Spec round loop.
        while !streamed && finish_reason.is_empty() && (generated.len() as u32) < max_tokens {
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
            if let Some(each) = draft_cost.median() {
                gamma_now = best_gamma_for_check(
                    acceptance.alpha(),
                    &check,
                    each,
                    gamma_now,
                    BEST_GAMMA_MAX,
                );
            }
            // 0 = a plain round: the controller's answer where guessing costs
            // more than the round trips it saves (`best_gamma_for_check`).
            let gamma = gamma_now.min(remaining);
            let round_start = std::time::Instant::now();

            // Guess k lands at absolute position current_pos + 1 + k: the verify
            // writes `last_token` at `current_pos`, and its row k predicts the
            // token after it — the positions the sampling segment keys by.
            let pick = noise.as_ref().map(|n| super::speculative::DraftPick {
                noise: n,
                first: current_pos as u64 + 1,
                params: &self.request.sampling_params,
                history: &generated,
            });

            // SWARM-SPEC Layer 1 cascade (same pattern as single-segment
            // speculative.rs): try n-gram lookup first; on miss fall back
            // to the draft model. On hit, llama.cpp's draft KV is synced via
            // draft_sync_tokens so subsequent rounds remain consistent; our
            // engine's drafter reads what it missed with its next call.
            // Not with shared noise: an n-gram guess is a FIXED token, accepted
            // with probability p(guess) at temperature > 0, while the drafter
            // drawing with the shared noise reproduces the sample itself.
            // Measured on the split rig (llama-3.2-3b, 3-bit far-half shadow,
            // T=0.7, γ=4): 4.11 tokens per round with n-gram lookup skipped,
            // 3.47 with it tried first.
            let ngram_drafts = if pick.is_some() || gamma == 0 {
                Vec::new()
            } else {
                ngram_lookup_drafts(
                    &self.shared_state.config.inference,
                    &prompt_tokens,
                    &generated,
                    gamma,
                )
            };
            let outcome: Result<Vec<u32>, SwarmError> = match &mut drafter {
                _ if drafting_off => Ok(Vec::new()),
                // A plain round guesses nothing. llama.cpp's drafter still takes
                // the round's first token into its cache — each drafting call
                // begins by feeding it, and a round that skipped it would leave
                // a gap in the drafter's context. Our engine's drafter reads
                // what it missed with its next call.
                Drafter::Llama { exec, state } if gamma == 0 => {
                    tokio::task::block_in_place(|| draft_sync_tokens(state, exec, last_token, &[]))
                        .map(|()| Vec::new())
                }
                Drafter::Engine(_) if gamma == 0 => Ok(Vec::new()),
                Drafter::Llama { exec, state } => {
                    let synced = if ngram_drafts.is_empty() {
                        None
                    } else {
                        match tokio::task::block_in_place(|| {
                            draft_sync_tokens(state, exec, last_token, &ngram_drafts)
                        }) {
                            Ok(()) => Some(ngram_drafts),
                            Err(e) => {
                                tracing::warn!(%request_id, error = %e, "DSD: ngram-sync failed — falling back to draft sample");
                                None
                            }
                        }
                    };
                    match synced {
                        Some(d) => Ok(d),
                        None => tokio::task::block_in_place(|| {
                            draft_next_gamma(state, exec, last_token, gamma, pick.as_ref())
                        }),
                    }
                }
                Drafter::Engine(_) if !ngram_drafts.is_empty() => Ok(ngram_drafts),
                Drafter::Engine(e) => {
                    e.draft(
                        &self.shared_state,
                        gamma,
                        &self.request.sampling_params,
                        &generated,
                        noise.map(|n| n.seed()),
                        self.request.cancel.clone(),
                    )
                    .await
                }
            };
            let drafts = match outcome {
                Ok(d) => d,
                Err(e) => {
                    tracing::warn!(
                        %request_id,
                        drafter = %drafter_name,
                        error = %e,
                        "DSD: the drafter failed — finishing this reply without guessing ahead"
                    );
                    drafting_off = true;
                    Vec::new()
                }
            };

            let drafted_at = std::time::Instant::now();
            // verify_tokens = [bootstrap, q_1..q_γ]
            let mut verify_tokens: Vec<u32> = Vec::with_capacity(drafts.len() + 1);
            verify_tokens.push(last_token);
            verify_tokens.extend_from_slice(&drafts);

            // Multi-segment verify forward. The last segment walks the drafts
            // with the caller's sampler where it can and answers with the
            // tokens it kept; otherwise it returns γ+1 logit vectors and
            // `accept` walks them here — the same rule either way.
            let reply = match super::forward_verify_through_segments(
                &self.shared_state,
                &self.network_tx,
                request_id,
                current_pos as u32,
                &self.assignment.segments,
                &verify_tokens,
                pending_truncate,
                Some(super::TailWalk {
                    drafts: &drafts,
                    sampling: &self.request.sampling_params,
                    generated: &generated,
                    coupling: noise.map(|n| n.seed()),
                }),
            )
            .await
            {
                Ok(v) => v,
                // A check that did not come back is the request's failure, as
                // it is in the n-gram loop (`ngram_only_spec`, the default split
                // path): `keeping_the_partial` hands back what was produced,
                // reported as the failure it is, and the router retries a
                // request that has streamed nothing. Ending the reply here with
                // `stop` — as this did — reported a dropped connection as a
                // finished one-token answer (2026-09-28, the far node's link
                // dropping mid-reply on the TH↔BE split).
                Err(e) => {
                    tracing::warn!(%request_id, error = %e, "DSD: pipeline verify failed");
                    return Err(e);
                }
            };

            let kv_after_forward = expected_kv_len + verify_tokens.len() as u32;

            // SpecExec's walk: keep a draft while the target's own SAMPLE
            // agrees. The drafter proposes its argmax, a draft with no
            // distribution behind it, for which this is exactly the
            // speculative-sampling rule at any temperature.
            let (accepted, bonus, _all_accepted) = match reply.accept(
                &drafts,
                &self.request.sampling_params,
                &generated,
                noise.as_ref().map(|n| (n, current_pos as u64 + 1)),
            ) {
                Ok(decided) => decided,
                // A reply that cannot be read — too few rows, non-finite
                // logits, a walk claiming tokens that were never guessed — is
                // a failure of the check, reported as one exactly as a check
                // that never came back (above): ending here with `stop` would
                // hand the caller a truncated reply as a finished one.
                Err(e) => {
                    tracing::warn!(%request_id, error = %e, "DSD: unusable verify reply");
                    return Err(e);
                }
            };

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

            super::emit_streaming_batch(
                &self.partial_reply,
                &token_tx,
                &decoder,
                &emitted,
                &eos_set,
                &mut finish_reason,
            )
            .await;

            // Bail before the per-round bookkeeping when the client has
            // disconnected. Mirrors speculative.rs — the inner `break` only
            // exits the streaming for-loop, leaving the acceptance bookkeeping
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

            // What this round cost and kept, for the next round's γ. A round
            // that guessed nothing says nothing about guessing.
            if !drafts.is_empty() {
                acceptance.record(accepted.len() as u32, drafts.len() as u32);
                let draft_ms = (drafted_at - round_start).as_secs_f64() * 1000.0;
                if drafter_warm {
                    draft_cost.record(draft_ms / drafts.len() as f64);
                }
                drafter_warm = true;
            }
            check.record(
                verify_tokens.len() as u32,
                drafted_at.elapsed().as_secs_f64() * 1000.0,
            );

            // Bring the drafter's cache in line with what the check kept.
            match &mut drafter {
                _ if drafting_off => {}
                Drafter::Llama { exec, state } => tokio::task::block_in_place(|| {
                    draft_sync_after_round(state, exec, &drafts, &accepted, bonus)
                })?,
                Drafter::Engine(e) => {
                    e.settle(accepted.len());
                    e.push(&emitted);
                }
            }

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

        crate::inference::dsd_controller::remember(
            learned_key,
            crate::inference::dsd_controller::Learned {
                acceptance: acceptance.clone(),
                check: check.clone(),
                draft_ms_each: draft_cost.median(),
                gamma: gamma_now,
                at: std::time::Instant::now(),
            },
        );
        tracing::info!(
            %request_id,
            segments = self.assignment.segments.len(),
            drafter = %drafter_name,
            proposed = acceptance_proposed,
            accepted = acceptance_accepted,
            final_gamma = gamma_now,
            alpha = format_args!("{:.3}", acceptance.alpha()),
            check_fixed_ms = format_args!("{:.1}", check.fit().map_or(0.0, |f| f.0)),
            check_ms_per_position = format_args!("{:.1}", check.fit().map_or(0.0, |f| f.1)),
            draft_ms_each = format_args!("{:.1}", draft_cost.median().unwrap_or(0.0)),
            "DSD: request complete"
        );

        Ok(Some(
            self.finish_speculative(
                request_id,
                generated,
                &decoder,
                &eos_set,
                prompt_token_count as u32,
                finish_reason,
            )
            .await,
        ))
    }
}

// forward_verify_through_segments moved to pipeline/mod.rs (R136 Layer 1
// multi-segment) so it's reachable without the `llama` feature gate.
// DSD calls super::forward_verify_through_segments now.
