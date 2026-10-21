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
//! This path preserves the SAMPLED DISTRIBUTION, at any temperature. A
//! draft position is sampled through the real sampler
//! (`sampling::sampled_accept_reject`) and the draft is kept only
//! when the sampler independently produced the same token — which IS
//! the speculative-sampling rejection rule for a draft with no
//! distribution behind it, since accepting with probability `p(x)` and
//! otherwise drawing from the renormalised remainder is exactly what
//! "draw `t ~ p`, keep the draft iff `t == x`" does. N-gram lookup
//! shifts only the DRAFT SOURCE, not the verification.
//!
//! **It is NOT bit-identical to non-speculative decoding** (gotcha
//! #370), and this comment used to say it was — twice over: it also
//! described an argmax comparison the code stopped doing when the
//! greedy-only gate was removed. A verify forward reassociates, so the
//! logits differ in the last bits even at temperature 0. Describing
//! this as bit-identical invites someone to hunt a bug in a difference
//! that is expected.
//!
//! # Cost
//!
//! A tail that advertises `features::SPEC_WALK_AT_TAIL` walks the drafts
//! where the logits are and answers with token ids, so a round — hit or
//! miss — costs a few bytes back. An older tail still returns a
//! full-vocabulary f32 vector per position (~513 KB on a 128k-vocab model),
//! including on a MISS round; `payoff_justifies_the_wire` is what stops a
//! workload that never hits from paying that to such a peer indefinitely.

use crate::error::SwarmError;
use crate::inference::router::{InferenceOutput, StreamingTokenEvent, StreamingTokenTx};

use super::PipelineExecutor;
use crate::daemon::state::requester_guesses::GuessSlot;

/// Observed payoff of this loop, as accepted tokens per round x 100.
/// `0` means unknown — nothing has run yet and the next request finds out.
static DIST_SPEC_PAYOFF_X100: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

/// The payoff at which speculating over the network is worth what it costs,
/// when we cannot price the link.
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

/// What a round has to return before speculating beats not speculating, given
/// what the link actually costs.
///
/// **The flat constant above ignores the round trip it saves, and that is the
/// wrong calculus on exactly the path where speculation matters most.** A plain
/// decode step on a multi-segment pipeline also pays a full round trip, so
/// speculating trades `(k - 1)` round trips for one payload of logits:
///
/// ```text
///   speculating wins  <=>  ship(logits) < (k - 1) * rtt
///   i.e.  k > 1 + ship(logits) / rtt
/// ```
///
/// Worked at 513 KB (a 128k-vocab miss round) and 200 ms RTT: on a 10 Mbps
/// uplink the payload costs ~410 ms, so a round must return more than 3.05
/// tokens; on 100 Mbps it costs ~41 ms and 1.21 is enough. A single flat bar
/// cannot express that, and it lands conservative on the slow link and
/// permissive on the fast one — the opposite of useful.
///
/// It matters most for **boomerang** (`inference.encrypted_pipeline`), which is
/// multi-segment by construction and therefore takes this path. Boomerang
/// decode is latency-bound at about one round trip per token — a ceiling near
/// 5 tok/s at 200 ms and 1.25 at 800 ms whatever the hardware does — and
/// amortising that trip over several tokens is the only lever that raises it.
///
/// Returns `None` when the link cannot be priced (no bandwidth advertised, no
/// latency sample), and the caller falls back to the flat constant — unknown
/// must not become a number.
fn required_tokens_per_round_x100(
    logits_bytes: u64,
    bandwidth_mbps: Option<f32>,
    rtt_ms: Option<u32>,
) -> Option<u32> {
    let bw = bandwidth_mbps.filter(|b| *b > 0.0 && b.is_finite())?;
    let rtt = rtt_ms.filter(|r| *r > 0)? as f32;
    // bytes -> bits -> ms at Mbit/s: (bytes * 8) / (mbps * 1000)
    let ship_ms = (logits_bytes as f32 * 8.0) / (bw * 1000.0);
    if !ship_ms.is_finite() {
        return None;
    }
    let required = 1.0 + ship_ms / rtt;
    // A round can only ever return so many tokens; past the draft width the bar
    // is unreachable and the honest answer is "never worth it", which the
    // caller gets by comparing against it.
    Some((required * 100.0).min(u32::MAX as f32) as u32)
}

/// Whether the loop has earned its bandwidth. Pure, so the policy is testable
/// without a pipeline.
///
/// `required_x100` is the link-priced bar where the link could be priced; the
/// flat constant otherwise.
fn payoff_justifies_the_wire(seen_x100: u32, required_x100: Option<u32>) -> bool {
    seen_x100 == 0 || seen_x100 >= required_x100.unwrap_or(PAYOFF_WORTH_THE_WIRE_X100)
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

/// The requester's guesses for this reply (#318): a split a peer handed over
/// asking to guess for it (`daemon::state::requester_guesses`), led here from
/// its first token. A continued reply takes none — its positions are not the
/// ones the requester follows. (A node whose owner turned speculation off opens
/// no slot: `dispatch::remote_generate::handle_delegated_split`.)
fn requester_guess_slot(exec: &PipelineExecutor) -> Option<std::sync::Arc<GuessSlot>> {
    if exec.request.continuation.is_some() {
        return None;
    }
    exec.shared_state
        .requester_guesses
        .for_led(&exec.request.id)
}

/// How many of the requester's guesses the next check takes: the measured
/// choice (`dsd_controller::best_gamma_for_check`) over this reply's checks,
/// with no drafting cost — the requester drafted them — and
/// `guesses_per_check` as the ceiling.
fn choose_guess_gamma(
    acceptance: &crate::inference::dsd_controller::AcceptanceEstimate,
    check: &crate::inference::dsd_controller::CheckCost,
    current: u32,
    cfg: &crate::config::InferenceConfig,
) -> u32 {
    next_guess_gamma(
        acceptance,
        check,
        current,
        cfg.guesses_per_check(),
        fixed_guess_gamma(),
    )
}

fn next_guess_gamma(
    acceptance: &crate::inference::dsd_controller::AcceptanceEstimate,
    check: &crate::inference::dsd_controller::CheckCost,
    current: u32,
    max: u32,
    fixed: bool,
) -> u32 {
    if fixed {
        return max;
    }
    crate::inference::dsd_controller::best_gamma_for_check(
        acceptance.alpha(),
        check,
        0.0,
        current,
        max,
    )
}

/// `SWARMLLM_HEAD_GUESS_GAMMA=fixed` checks as many of the requester's guesses
/// as `guesses_per_check` allows every time — the rule before the head chose
/// (#318) — for A/B inside one binary.
fn fixed_guess_gamma() -> bool {
    static FIXED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FIXED.get_or_init(|| std::env::var("SWARMLLM_HEAD_GUESS_GAMMA").as_deref() == Ok("fixed"))
}

/// Up to `max` of the requester's guesses that go on from `generated`, cut at
/// the first id outside the model's vocabulary: a window comes off the wire,
/// and an id this node cannot embed would fail the reply.
fn requester_drafts(slot: &GuessSlot, generated: &[u32], max: usize, vocab: usize) -> Vec<u32> {
    let mut guesses = slot.guesses(generated, max);
    if let Some(bad) = guesses.iter().position(|&t| t as usize >= vocab) {
        guesses.truncate(bad);
    }
    guesses
}

/// Fast-path preconditions for the draft-free n-gram-only spec loop.
///
/// `guessed`: the requester guesses for this reply ([`requester_guess_slot`]).
/// Its guesses are checked whether or not prompt lookup is switched on, and
/// whatever prompt lookup has been earning: the payoff below measures prompt
/// lookup's guesses, not a model's.
pub(super) fn eligible(exec: &PipelineExecutor, guessed: bool) -> bool {
    // The LIVE config, not the boot snapshot. Every read here used to come from
    // `shared_state.config`, so turning n-gram lookup off in Settings changed
    // nothing until the daemon was restarted (gotcha #281).
    let live = exec.shared_state.cfg();
    let cfg = &live.inference;
    if !cfg.ngram_lookup_enabled && !guessed {
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
    // behind it — see `sampling::sampled_accept_reject`.
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
    // Nor a plan the whole-model hand-off will run. `remote_generate` gives the
    // one peer holding the model the whole generation: it decodes on its own
    // card — speculating there with its OWN n-gram lookup
    // (`model_worker::ngram_spec_eligible`), where a miss costs nothing on the
    // wire — and streams the reply back. This loop pays a round trip per round,
    // so over a single remote segment it can only lose. Measured on the live
    // node against one peer 443 ms away (2026-09-27): this loop 0.78 tok/s,
    // the hand-off 10.7-11.1. Asked through the hand-off's own predicate, so
    // the two cannot disagree about which plans it takes.
    if super::remote_generate::eligible(exec) {
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
    // Price the actual link where we can. The tail is what returns the logits,
    // so its bandwidth and our round trip to it are the two numbers that decide
    // whether speculating beats a plain step — see
    // `required_tokens_per_round_x100`.
    let required = (|| {
        let tail = exec.assignment.segments.last()?;
        if &tail.node_id == local_node_id {
            // A local tail returns its logits over a Unix socket; the payload
            // is not what limits this and the flat bar is the honest answer.
            return None;
        }
        let vocab = exec
            .shared_state
            .standalone_tokenizer(&exec.request.model_id)?
            .as_ref()
            .vocab_size() as u64;
        let peer = exec.shared_state.peer_registry.get(&tail.node_id)?;
        let bandwidth = peer.capability.as_ref().map(|c| c.bandwidth_mbps);
        required_tokens_per_round_x100(vocab * 4, bandwidth, peer.latency_ms)
    })();

    let seen = DIST_SPEC_PAYOFF_X100.load(std::sync::atomic::Ordering::Relaxed);
    if !guessed && !payoff_justifies_the_wire(seen, required) {
        tracing::debug!(
            seen_x100 = seen,
            required_x100 = ?required,
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
        let guess_slot = requester_guess_slot(self);
        if !eligible(self, guess_slot.is_some()) {
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
        let (_extract_ptc, eos_tokens, decoder) = self.extract_model_cache(&prompt, 0).await;
        let eos_tokens: std::collections::HashSet<u32> = eos_tokens.into_iter().collect();
        // Use the standalone tokenizer's count (consistent with the n-gram
        // history and with the remote worker, since both derive from the
        // same gguf_header.bin). extract_model_cache may estimate when
        // the local split_model entry isn't populated (all-remote pipelines).
        let prompt_token_count = prompt_tokens.len();
        let vocab = tokenizer.vocab_size();

        // The requester's shared noise (`inference::coupled_noise`): its
        // drafter draws each guess with the noise this reply's every sample is
        // drawn with here, so at a temperature above zero a close drafter
        // reproduces the SAMPLE (dsd.rs measured 92.5% against 71.3% fixed at
        // 1.0). Keyed by absolute position, so only where both tokenized the
        // prompt alike; otherwise its guesses are checked uncoupled — still
        // exact, agreeing less often. Prompt lookup's fixed guesses lose
        // nothing to it: each is kept with its probability either way.
        let noise = guess_slot.as_ref().and_then(|slot| {
            let asked = slot.asked();
            let seed = asked.coupling_seed?;
            if self.request.sampling_params.temperature <= 0.0 {
                return None;
            }
            if asked.prompt_tokens as usize != prompt_token_count {
                tracing::info!(
                    %request_id,
                    requester_prompt_tokens = asked.prompt_tokens,
                    prompt_tokens = prompt_token_count,
                    "requester guesses: the requester read the prompt as another length — \
                     its guesses are checked without shared noise"
                );
                return None;
            }
            Some(crate::inference::coupled_noise::CoupledNoise::new(seed))
        });

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
        // Rounds that checked the requester's guesses, the guesses they
        // checked and the ones they kept.
        let mut guess_rounds: u32 = 0;
        let mut guesses_checked: u32 = 0;
        let mut guesses_kept: u32 = 0;

        // Every token goes into the slot BEFORE it is emitted, so the task
        // streaming the reply finds its id beside its text.
        if let Some(slot) = &guess_slot {
            slot.record(0, &[first_token]);
        }
        // Stream first token
        super::emit_first_streaming_token(
            &self.partial_reply,
            &token_tx,
            &decoder,
            first_token,
            &eos_tokens,
        )
        .await;
        if eos_tokens.contains(&first_token) {
            finish_reason = "stop".into();
        }

        let live = self.shared_state.cfg();
        let cfg = &live.inference;
        let max_draft = cfg.ngram_num_pred_tokens.min(cfg.guesses_per_check());
        // How many of the requester's guesses a check takes (#318): the choice
        // the requester makes when it drives (`dsd_controller::best_gamma_for_check`),
        // from this reply's own checks — what one costs by the positions it
        // reads (every check feeds the line, prompt lookup's and one-token
        // checks too) and how often a guess is kept. Guesses drafted on the
        // requester cost this node nothing, so only the check is priced. Zero
        // is an answer: on processors each position a check reads cost about
        // half a one-token pass, and guesses kept 59% of the time won nothing
        // (`~/swarmllm-318/guess318.sh`, 10-20).
        let mut guess_gamma = cfg.guesses_per_check();
        let mut check_cost = crate::inference::dsd_controller::CheckCost::default();
        let mut guess_acceptance = crate::inference::dsd_controller::AcceptanceEstimate::new();
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

            // ── The requester's guesses, then the n-gram cascade ──
            let guessed = guess_slot
                .as_deref()
                .map(|slot| {
                    requester_drafts(slot, &generated, guess_gamma.min(remaining) as usize, vocab)
                })
                .unwrap_or_default();
            let from_requester = !guessed.is_empty();
            let drafts = if from_requester {
                guessed
            } else if cfg.ngram_lookup_enabled {
                super::speculative::ngram_lookup_drafts(cfg, &prompt_tokens, &generated, this_gamma)
            } else {
                Vec::new()
            };

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
                let checked_at = std::time::Instant::now();
                // Sample, not argmax: this is an ordinary decode step that
                // happens to have gone through the verify wire, and it must
                // honour the same sampling parameters every other step does.
                // Walked at the tail where it can be, so a miss round costs a
                // token id on the wire, not a vocabulary.
                let reply = super::forward_verify_through_segments(
                    &self.shared_state,
                    &self.network_tx,
                    request_id,
                    current_pos as u32,
                    &self.assignment.segments,
                    &verify_tokens,
                    truncate_for_this_round,
                    Some(super::TailWalk {
                        drafts: &[],
                        sampling: &self.request.sampling_params,
                        generated: &generated,
                        coupling: noise.map(|n| n.seed()),
                        parents: &[],
                    }),
                    super::worker_requester(&self.request),
                    self.verify_may_chain(),
                    &mut self.chaining_disabled,
                )
                .await?;
                // An unusable reply fails the request, exactly as the hit arm
                // below does and as a check that never came back does (the `?`
                // above) — the same failure must not stream an error on a miss
                // round and a clean finish on a hit round. `keeping_the_partial`
                // hands back what was produced as the failure it is; ending
                // with what it had here reported a truncated reply as a
                // finished one (gotcha #746, review of 2026-09-28).
                let (_, bonus, _) = match reply.accept(
                    &[],
                    &self.request.sampling_params,
                    &generated,
                    noise.as_ref().map(|n| (n, current_pos as u64 + 1)),
                ) {
                    Ok(decided) => decided,
                    Err(e) => {
                        tracing::warn!(%request_id, error = %e, "ngram-only: unusable verify reply");
                        return Err(e);
                    }
                };
                if guess_slot.is_some() {
                    check_cost.record(1, checked_at.elapsed().as_secs_f64() * 1000.0);
                    guess_gamma =
                        choose_guess_gamma(&guess_acceptance, &check_cost, guess_gamma, cfg);
                }
                last_token = bonus;
                if let Some(slot) = &guess_slot {
                    slot.record(generated.len(), &[bonus]);
                }
                generated.push(bonus);
                current_pos += 1;
                // Through the shared helper rather than a direct send, so this
                // arm inherits the end-of-turn filter like every other emission
                // point. A hand-rolled send here is exactly how a control token
                // reaches a client (gotcha #414).
                super::emit_streaming_batch(
                    &self.partial_reply,
                    &token_tx,
                    &decoder,
                    &[bonus],
                    &eos_tokens,
                    &mut finish_reason,
                )
                .await;
                if eos_tokens.contains(&bonus) {
                    finish_reason = "stop".into();
                    break;
                }
                continue;
            }

            // ── HIT path: the requester's guesses or the n-gram's ──
            if from_requester {
                guess_rounds += 1;
                guesses_checked += drafts.len() as u32;
            } else {
                ngram_rounds += 1;
                // R137: bump lifetime hit counter (counts hit-rounds, not
                // accepted tokens — accept count is in the spec_logits result).
                self.shared_state
                    .metrics
                    .ngram_hits
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            let mut verify_tokens: Vec<u32> = Vec::with_capacity(drafts.len() + 1);
            verify_tokens.push(last_token);
            verify_tokens.extend_from_slice(&drafts);

            let truncate_for_this_round = pending_truncate.take();
            let checked_at = std::time::Instant::now();
            let reply = super::forward_verify_through_segments(
                &self.shared_state,
                &self.network_tx,
                request_id,
                current_pos as u32,
                &self.assignment.segments,
                &verify_tokens,
                truncate_for_this_round,
                Some(super::TailWalk {
                    drafts: &drafts,
                    sampling: &self.request.sampling_params,
                    generated: &generated,
                    coupling: noise.map(|n| n.seed()),
                    parents: &[],
                }),
                super::worker_requester(&self.request),
                self.verify_may_chain(),
                &mut self.chaining_disabled,
            )
            .await?;
            let (accepted, bonus, _all) = match reply.accept(
                &drafts,
                &self.request.sampling_params,
                &generated,
                noise.as_ref().map(|n| (n, current_pos as u64 + 1)),
            ) {
                Ok(decided) => decided,
                Err(e) => {
                    tracing::warn!(%request_id, error = %e, "ngram-only: unusable verify reply");
                    return Err(e);
                }
            };
            let emitted = super::round_tokens_for_reply(
                &accepted,
                bonus,
                &eos_tokens,
                (max_tokens as usize).saturating_sub(generated.len()),
            );
            if from_requester {
                guesses_kept += accepted.len() as u32;
                guess_acceptance.record(accepted.len() as u32, drafts.len() as u32);
            }
            if guess_slot.is_some() {
                check_cost.record(
                    verify_tokens.len() as u32,
                    checked_at.elapsed().as_secs_f64() * 1000.0,
                );
                guess_gamma = choose_guess_gamma(&guess_acceptance, &check_cost, guess_gamma, cfg);
            }
            if let Some(slot) = &guess_slot {
                slot.record(generated.len(), &emitted);
            }

            // Emit (stream + accumulate). The EOS token stays in `emitted` so
            // the accumulator below still sees it and stops; `emit_streaming_batch`
            // is what keeps it off the wire, for every path at once.
            super::emit_streaming_batch(
                &self.partial_reply,
                &token_tx,
                &decoder,
                &emitted,
                &eos_tokens,
                &mut finish_reason,
            )
            .await;
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

        // The payoff is prompt lookup's: a reply the requester guessed for
        // says nothing about what prompt lookup earns the next one.
        if guess_rounds == 0 {
            record_payoff(generated.len() as u32, ngram_rounds + fallback_rounds);
        }

        tracing::info!(
            request_id = %request_id,
            generated_tokens = generated.len(),
            ngram_rounds,
            fallback_rounds,
            requester_guesses = guess_slot.is_some(),
            guess_rounds,
            guesses_checked,
            guesses_kept,
            guess_gamma,
            check_fit = ?check_cost.fit().map(|(fixed, per)| (fixed.round(), per.round())),
            coupled = noise.is_some(),
            payoff_x100 = DIST_SPEC_PAYOFF_X100.load(std::sync::atomic::Ordering::Relaxed),
            ngram_hit_rate = format!(
                "{:.1}%",
                100.0 * ngram_rounds as f32 / (ngram_rounds + fallback_rounds).max(1) as f32
            ),
            "SWARM-SPEC L1 ngram-only: complete"
        );

        // Every other pipeline path sends this, and the API layer depends on
        // it: without a finish event `got_finish` stays false, so the SSE
        // encoder takes its "pipeline never streamed" fallback and emits the
        // WHOLE reply again as one delta — on top of the tokens already sent.
        // Measured on the live swarm 2026-08-30: "1\n2\n3" came back as
        // "1\n2\n3<|eot_id|>1\n2\n3".
        if let Some(ref tx) = token_tx {
            let _ = tx
                .send(StreamingTokenEvent {
                    text: String::new(),
                    finish_reason: Some(finish_reason.clone()),
                    matched_stop_sequence: None,
                })
                .await;
        }

        Ok(Some(
            self.finish_speculative(
                request_id,
                generated,
                &decoder,
                &eos_tokens,
                prompt_token_count as u32,
                finish_reason,
            )
            .await,
        ))
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
        assert!(payoff_justifies_the_wire(0, None));
    }

    /// The measured failure: every round a miss, so the loop returned exactly
    /// the one token a plain decode step would have returned — while dragging
    /// back a full vocabulary to do it.
    #[test]
    fn one_token_per_round_does_not_pay_for_a_vocabulary() {
        assert!(
            !payoff_justifies_the_wire(100, None),
            "a round that yields one token is a plain decode step wearing a \
             513 KB hat"
        );
    }

    /// The workload this path exists for: copying its input back, several
    /// tokens accepted per round.
    #[test]
    fn a_copying_workload_still_speculates() {
        assert!(payoff_justifies_the_wire(880, None));
        assert!(payoff_justifies_the_wire(PAYOFF_WORTH_THE_WIRE_X100, None));
    }

    /// Just under the bar must be refused, or the constant is decorative.
    #[test]
    fn just_below_the_bar_is_refused() {
        assert!(!payoff_justifies_the_wire(
            PAYOFF_WORTH_THE_WIRE_X100 - 1,
            None
        ));
    }
}

#[cfg(test)]
mod link_price_tests {
    use super::{payoff_justifies_the_wire, required_tokens_per_round_x100};

    /// 128k vocab x 4 bytes — one miss round on llama-3.2-3b.
    const MISS_BYTES: u64 = 128_256 * 4;

    /// The whole point: the same payload is a bargain on a fast link and a
    /// waste on a slow one, at identical round-trip time. A flat threshold
    /// cannot express that.
    #[test]
    fn the_same_payload_costs_differently_on_different_links() {
        let slow = required_tokens_per_round_x100(MISS_BYTES, Some(10.0), Some(200)).unwrap();
        let fast = required_tokens_per_round_x100(MISS_BYTES, Some(100.0), Some(200)).unwrap();
        // ~410ms of shipping against a 200ms trip -> needs >3 tokens a round.
        assert!(
            (290..=320).contains(&slow),
            "10 Mbps at 200ms should need ~3.05 tokens/round, got {slow}"
        );
        // ~41ms against the same trip -> 1.2 is enough.
        assert!(
            (115..=125).contains(&fast),
            "100 Mbps at 200ms should need ~1.21 tokens/round, got {fast}"
        );
        assert!(slow > fast, "a slower link must demand more per round");
    }

    /// A longer round trip makes speculating MORE attractive, because each
    /// saved trip is worth more. This is the boomerang case.
    #[test]
    fn a_worse_round_trip_lowers_the_bar() {
        let near = required_tokens_per_round_x100(MISS_BYTES, Some(100.0), Some(60)).unwrap();
        let far = required_tokens_per_round_x100(MISS_BYTES, Some(100.0), Some(800)).unwrap();
        assert!(
            far < near,
            "a distant peer should need FEWER tokens per round to justify \
             speculating, not more: far={far} near={near}"
        );
        assert!(far >= 100, "the bar can never fall below one token a round");
    }

    /// Unknown must stay unknown. A peer that advertises no bandwidth, or that
    /// we have never timed, falls back to the flat constant rather than being
    /// assigned a number — the same rule the capacity bound follows.
    #[test]
    fn an_unpriceable_link_returns_none() {
        assert_eq!(
            required_tokens_per_round_x100(MISS_BYTES, None, Some(200)),
            None
        );
        assert_eq!(
            required_tokens_per_round_x100(MISS_BYTES, Some(100.0), None),
            None
        );
        assert_eq!(
            required_tokens_per_round_x100(MISS_BYTES, Some(0.0), Some(200)),
            None
        );
        assert_eq!(
            required_tokens_per_round_x100(MISS_BYTES, Some(f32::NAN), Some(200)),
            None
        );
    }

    /// And unknown-payoff still gets to try, whatever the link says.
    #[test]
    fn a_priced_link_still_lets_an_unmeasured_workload_find_out() {
        let required = required_tokens_per_round_x100(MISS_BYTES, Some(1.0), Some(50));
        assert!(
            required.unwrap() > 1000,
            "a 1 Mbps link should demand a lot"
        );
        assert!(
            payoff_justifies_the_wire(0, required),
            "nothing measured yet must still get one request to find out"
        );
        assert!(
            !payoff_justifies_the_wire(400, required),
            "4 tokens a round cannot pay for 513 KB on a 1 Mbps link"
        );
    }
}

#[cfg(test)]
mod requester_guess_tests {
    use super::*;
    use crate::daemon::state::requester_guesses::RequesterGuessSlots;
    use crate::types::{GuessesAsked, RequesterGuesses};

    /// A guess the model has no row for is never sent through the layers: the
    /// window is cut there, and what precedes it is still checked.
    #[test]
    fn a_guess_outside_the_vocabulary_is_never_checked() {
        let slots = std::sync::Arc::new(RequesterGuessSlots::default());
        let (wire, led) = (uuid::Uuid::new_v4(), uuid::Uuid::new_v4());
        let _open = slots.open(
            wire,
            led,
            vec![1],
            GuessesAsked {
                coupling_seed: None,
                prompt_tokens: 3,
            },
        );
        let slot = slots.for_led(&led).unwrap();
        slot.record(0, &[5]);
        assert!(slot.take_unsent().is_some());
        slots.offer_from(
            &[1],
            RequesterGuesses {
                request_id: wire,
                after: 1,
                tokens: vec![7, 8, 99_999, 9],
            },
        );
        assert_eq!(requester_drafts(&slot, &[5], 8, 1000), vec![7, 8]);
        assert_eq!(requester_drafts(&slot, &[5], 1, 1000), vec![7]);
        assert_eq!(requester_drafts(&slot, &[5, 7], 8, 1000), vec![8]);
    }

    /// The head checks as many of the requester's guesses as pay for their
    /// positions. A check that grows with every position it reads — a
    /// processor's, as the 10-20 rig measured (~0.84 s at one position,
    /// ~1.95 s at four) — with guesses kept less than half the time takes
    /// none; a check that costs the same at any length (a card's, a long
    /// link) takes all it may.
    #[test]
    fn the_head_checks_as_many_guesses_as_pay_for_their_positions() {
        use crate::inference::dsd_controller::{AcceptanceEstimate, CheckCost};
        let mut seldom = AcceptanceEstimate::new();
        for i in 0..20 {
            seldom.record(u32::from(i % 2 == 0), 4);
        }
        let mut processor = CheckCost::default();
        for i in 0..8 {
            if i % 2 == 0 {
                processor.record(1, 840.0);
            } else {
                processor.record(4, 1950.0);
            }
        }
        let mut gamma = 4;
        for _ in 0..4 {
            gamma = next_guess_gamma(&seldom, &processor, gamma, 4, false);
        }
        assert_eq!(gamma, 0, "guessing that does not pay steps aside");

        let mut often = AcceptanceEstimate::new();
        for _ in 0..20 {
            often.record(3, 4);
        }
        let mut card = CheckCost::default();
        for i in 0..8 {
            if i % 2 == 0 {
                card.record(1, 450.0);
            } else {
                card.record(5, 470.0);
            }
        }
        let mut gamma = 2;
        for _ in 0..4 {
            gamma = next_guess_gamma(&often, &card, gamma, 4, false);
        }
        assert_eq!(
            gamma, 4,
            "a check that costs the same at any length takes all it may"
        );
        assert_eq!(
            next_guess_gamma(&seldom, &processor, 0, 4, true),
            4,
            "the A/B switch keeps the old rule"
        );
    }
}
