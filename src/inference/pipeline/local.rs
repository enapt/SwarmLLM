//! Local (same-node) execution paths: `execute_local` (full GGUF fast path)
//! and `process_local_segment` (split-model forward for one segment), plus
//! remote-segment timeout computation and result-await helpers.

use std::time::Duration;

use crate::error::SwarmError;
use crate::inference::router::InferenceOutput;
use crate::types::{LayerResult, PipelineSegment};

use super::{
    PipelineExecutor, DECODE_SECS_PER_LAYER, PREFILL_ACTIVATION_THRESHOLD_BYTES,
    PREFILL_SECS_PER_LAYER, SEGMENT_TIMEOUT_MAX_SECS, SEGMENT_TIMEOUT_MIN_SECS,
};
use crate::daemon::state::WorkKind;
use crate::daemon::SharedState;

/// Headroom over the predicted time before a healthy peer is given up on.
///
/// A prediction is an EMA over a peer whose real speed moves with whatever
/// else it is doing. Cutting off at the prediction would abandon roughly half
/// of all forwards. 3× is generous enough to ride out ordinary variance while
/// still detecting a peer that has genuinely stopped, and a wrong guess now
/// costs one failover rather than the request.
const SEGMENT_TIMEOUT_SAFETY_FACTOR: f32 = 3.0;

/// Extra budget granted when a peer may still have to LOAD the model.
///
/// This is the case that produced the 2026-08-01 failure: a CPU peer took
/// ~120s to load an 8B model and was cut off by a flat 120s deadline, despite
/// computing the segment itself in 10s. Loading is not proportional to
/// anything the prediction models, so it is added rather than scaled.
///
/// Being generous here does NOT slow down detection of a peer that is simply
/// gone: an unreachable peer is failed by `RR_ACK_TIMEOUT_SECS` (10s) on the
/// send path, which this deadline never gets to influence.
const COLD_MODEL_LOAD_ALLOWANCE_SECS: u64 = 240;

/// How long after a successful forward we still assume the peer has the model
/// resident. Comfortably longer than the idle-unload sweep, so a model we
/// believe is warm generally still is.
const PEER_MODEL_WARM_TTL_SECS: u64 = 900;

/// What `activation_bytes` is counting.
///
/// The prefill coefficient in `daemon::state::peer_speed` is measured in
/// hidden-state bytes. A first-segment forward instead carries raw prompt or
/// token bytes, which is a completely different scale for the same amount of
/// work — feeding those into the same average, or predicting from it, would
/// silently corrupt the estimate. So the unit is explicit at every call site
/// and the measured path is used only when they match.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ActivationUnits {
    /// Hidden states flowing between segments — the unit the coefficient uses.
    HiddenStates,
    /// Raw prompt or packed token bytes handed to the first segment.
    PromptBytes,
}

/// How long to wait for one remote segment.
///
/// Deliberately opaque and constructible only through
/// [`SegmentBudget::for_forward`]. `wait_for_result` takes this rather than a
/// bare `Duration` so that a new call site cannot invent its own deadline and
/// quietly bypass measured-speed sizing — the recurring failure mode described
/// in `.claude/rules/architecture.md` § "One invariant, N paths".
///
/// `Copy` so a resend after a repaired link (`ResendOnRefusal`) waits under the
/// same deadline as the forward it replaces; still constructible only here.
#[derive(Clone, Copy)]
pub(super) struct SegmentBudget {
    duration: Duration,
    /// Why this budget is what it is — logged so an operator can tell a
    /// measured deadline from a fallback one.
    basis: &'static str,
    /// Whether this forward was budgeted as a prefill. Carried rather than
    /// re-derived so the DIAG line cannot contradict the deadline it reports.
    prefill: bool,
}

impl SegmentBudget {
    /// Size the deadline for a specific forward to a specific peer.
    ///
    /// Uses what we have measured of this peer where the units allow, adds a
    /// cold-load allowance when the peer may not hold the model yet, and falls
    /// back to the peer-agnostic constants when we have never seen it do this
    /// kind of work.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn for_forward(
        state: &SharedState,
        node_id: &crate::types::NodeId,
        model_id: &crate::types::ModelId,
        kind: WorkKind,
        num_layers: u32,
        activation_bytes: usize,
        units: ActivationUnits,
    ) -> Self {
        let measured = match (kind, units) {
            // The prefill coefficient is ms per layer per HIDDEN-STATE byte; a
            // prediction from it over raw prompt bytes would be meaningless
            // rather than merely imprecise.
            (WorkKind::Prefill, ActivationUnits::PromptBytes) => None,
            // The decode coefficient is ms per layer and never reads the byte
            // count, so what a decode step happens to carry — one token id to
            // segment 0, hidden states to every later segment — cannot
            // corrupt it. Until 2026-09-02 this arm fell into the one above
            // whenever the units were `PromptBytes`, and the fallback beneath
            // it then asked the units rather than the kind: every decode step
            // of a remote first segment was budgeted as a PREFILL, 15 s a
            // layer, however fast the peer had been measured answering the
            // identical step a moment earlier (gotcha #434).
            _ => state.predict_segment_ms(node_id, kind, num_layers, activation_bytes),
        };

        let (base, basis) = match measured {
            Some(ms) => (
                Duration::from_millis((ms * SEGMENT_TIMEOUT_SAFETY_FACTOR) as u64),
                "measured",
            ),
            None => (
                PipelineExecutor::compute_segment_timeout(
                    kind,
                    num_layers,
                    activation_bytes,
                    units,
                ),
                "default",
            ),
        };

        let warm = state.peer_model_is_warm(
            node_id,
            model_id,
            Duration::from_secs(PEER_MODEL_WARM_TTL_SECS),
        );
        let (total, basis) = if warm {
            (base, basis)
        } else {
            (
                base + Duration::from_secs(COLD_MODEL_LOAD_ALLOWANCE_SECS),
                if basis == "measured" {
                    "measured+coldload"
                } else {
                    "default+coldload"
                },
            )
        };

        // A cold load can legitimately exceed the warm ceiling, so the
        // allowance is added on top of it rather than being clipped away by it.
        let ceiling = Duration::from_secs(SEGMENT_TIMEOUT_MAX_SECS)
            + Duration::from_secs(if warm {
                0
            } else {
                COLD_MODEL_LOAD_ALLOWANCE_SECS
            });
        // ...but never beyond what the transport will hold the request open
        // for. `RR_REQUEST_TIMEOUT_SECS` is the request_response protocol
        // timeout: past it libp2p fails the send regardless, so a larger budget
        // here would only wait for a result that can no longer arrive. Waiting
        // longer than the layer beneath you is not patience, it is a hang.
        let transport_ceiling =
            Duration::from_secs(crate::network::behaviour::RR_REQUEST_TIMEOUT_SECS);

        Self {
            duration: total
                .clamp(Duration::from_secs(SEGMENT_TIMEOUT_MIN_SECS), ceiling)
                .min(transport_ceiling),
            basis,
            prefill: PipelineExecutor::forward_is_prefill(kind, activation_bytes, units),
        }
    }

    pub(super) fn duration(&self) -> Duration {
        self.duration
    }

    pub(super) fn basis(&self) -> &'static str {
        self.basis
    }

    pub(super) fn is_prefill(&self) -> bool {
        self.prefill
    }

    /// A budget with an explicit deadline, for tests that need the wait to
    /// actually expire.
    ///
    /// `for_forward` clamps to `SEGMENT_TIMEOUT_MIN_SECS` (30 s), so no
    /// combination of measured latencies can produce a short one — which is why
    /// the timeout arm of `wait_for_result` had no end-to-end coverage, and why
    /// the helper that claimed to provide it ("a tiny measured latency, so a
    /// timeout can be provoked without sleeping") could not have worked. The
    /// production invariant is unchanged: outside tests a budget still comes
    /// only from `for_forward`.
    #[cfg(test)]
    pub(super) fn with_deadline_for_test(duration: Duration, prefill: bool) -> Self {
        Self {
            duration,
            basis: "test",
            prefill,
        }
    }
}

impl PipelineExecutor {
    /// Execute entirely on the local node (we have all layers).
    ///
    /// If speculative decoding is enabled and a draft model is loaded,
    /// uses the draft-verify-accept loop for higher throughput.
    pub(super) async fn execute_local(&self) -> Result<InferenceOutput, SwarmError> {
        let prompt = self.build_prompt().await;

        // Check if speculative decoding is available
        if self.shared_state.config.inference.speculative_decoding {
            let mut draft = self.shared_state.draft_executor.lock().await;
            if draft.is_loaded() {
                let gamma = self.shared_state.config.inference.speculative_gamma;
                let mut executor = self.shared_state.executor.lock().await;
                if !executor.is_loaded() {
                    return Err(SwarmError::NoModelLoaded);
                }
                let mut content = String::new();
                let (gen_result, spec_state) =
                    crate::inference::executor::without_starving_the_runtime(|| {
                        executor.generate_speculative(
                            &mut draft,
                            &prompt,
                            &self.request.sampling_params,
                            gamma,
                            |token| {
                                content.push_str(token);
                                true
                            },
                        )
                    })?;
                tracing::info!(
                    acceptance_rate = %spec_state.acceptance_rate(),
                    "Speculative decoding acceptance rate"
                );
                return Ok(InferenceOutput::from_gen_result(
                    self.request.id,
                    self.request.session_id.clone(),
                    content,
                    gen_result.finish_reason.as_str().to_string(),
                    &gen_result,
                ));
            }
        }

        // Standard (non-speculative) local inference
        let mut executor = self.shared_state.executor.lock().await;
        if !executor.is_loaded() {
            return Err(SwarmError::NoModelLoaded);
        }
        let (content, gen_result) =
            crate::inference::executor::without_starving_the_runtime(|| {
                executor.generate(&prompt, &self.request.sampling_params)
            })?;

        Ok(InferenceOutput::from_gen_result(
            self.request.id,
            self.request.session_id.clone(),
            content,
            gen_result.finish_reason.as_str().to_string(),
            &gen_result,
        ))
    }

    /// Process a pipeline segment locally using the split inference engine.
    ///
    /// Loads the split model (layer range) from the local GGUF if not already cached,
    /// then runs the forward pass on the activation tensor.
    /// Run one local segment of a distributed pipeline.
    ///
    /// `activation_bytes` is taken by value (not `&[u8]`) so the caller can
    /// `std::mem::take` the previous segment's buffer instead of forcing a
    /// `to_vec()` copy on every iteration of the segment loop. The buffer
    /// flows directly into `LayerForward.activations`.
    #[allow(clippy::too_many_arguments)]
    pub(super) async fn process_local_segment(
        &self,
        segment: &PipelineSegment,
        sequence_num: u32,
        index_pos: usize,
        activation_bytes: Vec<u8>,
        precomputed_vision_bytes: Option<&[u8]>,
        pre_embedded: bool,
        generated_ids: &[u32],
    ) -> Result<LayerResult, SwarmError> {
        let model_id = &segment.shard_id.model_id;
        let (layer_start, layer_end) = (
            segment.layer_range.0 as usize,
            segment.layer_range.1 as usize,
        );

        let split_key = self.ensure_split_model_entry(model_id, layer_start, layer_end)?;

        // Touch the metadata entry and extract cached EOS tokens
        let _cached_eos_tokens = {
            let entry = self
                .shared_state
                .split_models
                .get(&split_key)
                .ok_or_else(|| {
                    SwarmError::ServiceUnavailable(
                        "Split model was evicted during request — please retry".into(),
                    )
                })?;
            entry.value().touch();
            entry.value().eos_tokens.clone()
        };

        // R108: only ship `generated_ids` to the worker when the sampler
        // actually needs it — i.e. when frequency_penalty or
        // presence_penalty is non-zero. Otherwise the worker silently
        // ignores it but we still pay for the per-segment Vec<u32> copy
        // and the JSON-array serialization (the field is annotated
        // `skip_serializing_if = "Vec::is_empty"`). This comment used to say
        // "the distributed path already gates this" — it had stopped doing so,
        // and shipped the whole history to remote peers on every step until
        // both paths were made to ask the same predicate (2026-09-24).
        let needs_generated_ids =
            crate::inference::sampling::sampler_reads_history(&self.request.sampling_params);
        let generated_ids_for_worker = if needs_generated_ids {
            generated_ids.to_vec()
        } else {
            Vec::new()
        };

        // Build a LayerForward and route to the worker subprocess
        let layer_forward = crate::types::LayerForward {
            request_id: self.request.id,
            sequence_num,
            index_pos: index_pos as u32,
            activations: activation_bytes,
            format: crate::types::TensorFormat::FP16,
            model_id: model_id.clone(),
            layer_range: (layer_start as u32, layer_end as u32),
            tp_meta: None,
            vision_embeddings: precomputed_vision_bytes.map(|b| b.to_vec()),
            chain: Vec::new(),
            sender_peer_bytes: None,
            requester_node_id: None,
            pre_embedded,
            generated_ids: generated_ids_for_worker,
            // The request's LoRA adapter, which only ever runs HERE: a LoRA
            // request is planned as one local segment over every layer
            // (`router::distributed_exec::lora_local_assignment`), because the
            // adapter is registered on this computer alone.
            adapter_id: self.request.lora_adapter.clone(),
            draft_tokens: Vec::new(),
            spec_logits_requested: false,
            truncate_kv_to: None,
            chunk_meta: None,
            // The segment that samples needs the caller's parameters. Without
            // this the worker fell back to `SamplingParams::default()` —
            // 0.7/0.9/40 — so a request's temperature, top_p, top_k and stop
            // strings were silently discarded whenever it took the pipeline
            // path, which is every COLD-START request.
            sampling: Some(self.request.sampling_params.clone()),
        };
        // Watched against the request's cancel flag: a client that leaves
        // during a minutes-long prompt pass on this node's processor ends this
        // wait, and dropping it tells the worker to stop (gotcha #445).
        let layer_result = self
            .shared_state
            .model_process_pool
            .forward_for_request(layer_forward, self.request.cancel.clone())
            .await?;

        // Deliberately NOT counted as a forward served. This is our own segment
        // inside a pipeline we coordinate — work done for ourselves. The
        // dashboard tile reads "computations your computer did as part of the
        // network (earns credits)", and counting local work there told a user
        // whose only traffic was their own chat that they had served the swarm.
        // Serving is recorded at `SharedState::record_peer_serve`, reached only
        // from the two inbound paths.

        Ok(layer_result)
    }

    /// Compute a reasonable timeout for a remote segment based on workload,
    /// with no knowledge of the peer. The fallback used when we have not
    /// measured this peer doing this kind of work.
    ///
    /// Prefill (the whole prompt at once) is much slower than decode (a single
    /// token), so the two get different per-layer rates. Which one this forward
    /// is doing is decided by [`PipelineExecutor::forward_is_prefill`] — by the
    /// work KIND first and then the activation UNITS, never by the byte count;
    /// read that before assuming a large payload means prefill or a small one
    /// means decode.
    pub(super) fn compute_segment_timeout(
        kind: WorkKind,
        num_layers: u32,
        activation_bytes: usize,
        units: ActivationUnits,
    ) -> Duration {
        let per_layer_secs: u64 = if Self::forward_is_prefill(kind, activation_bytes, units) {
            PREFILL_SECS_PER_LAYER
        } else {
            DECODE_SECS_PER_LAYER
        };
        let base = (num_layers as u64) * per_layer_secs;
        let timeout = base.clamp(SEGMENT_TIMEOUT_MIN_SECS, SEGMENT_TIMEOUT_MAX_SECS);
        Duration::from_secs(timeout)
    }

    /// Is this forward doing a prefill? The single answer, for the deadline and
    /// for the DIAG that reports it.
    ///
    /// **The kind decides first.** `WorkKind` comes from the sequence number
    /// (`work_kind_for`): 0 is the prompt pass, everything after it is a
    /// single-token decode step — and a decode step is a decode step whatever
    /// it carries. Segment 0 of a decode step is handed the sampled token as
    /// `PromptBytes`, because that is the unit the first segment takes, and
    /// asking the units alone answered "prefill" for it: every decode step of
    /// a remote first segment was budgeted at 15 s a layer. Measured on the
    /// live swarm 2026-09-01 (gotcha #434): a 16-layer segment 0 that had just
    /// answered in 4.9 s took a decode step and went silent, and the pipeline
    /// waited 240 s for it — with a standby idle the whole time — where the
    /// decode budget is 32 s and its measured speed would have allowed ~30.
    ///
    /// **Within the prompt pass, the units decide, not the size.** A
    /// `PromptBytes` payload is the prompt itself — that is what the unit
    /// means — so the forward carrying it performs the whole prefill by
    /// construction, however short the prompt. Only `HiddenStates` can be
    /// classified by size, because `PREFILL_ACTIVATION_THRESHOLD_BYTES` is a
    /// hidden-state scale: one token of hidden state is thousands of bytes,
    /// whereas one token of prompt is a few. Comparing prompt bytes against it
    /// asks a question in the wrong units and answers "decode" for anything
    /// under ~100 KB of text — roughly 25k tokens, i.e. essentially every real
    /// prompt.
    ///
    /// Measured on the live swarm 2026-08-29: a 4728-token prompt reached
    /// segment 0 as `activation_bytes=24045`, was budgeted `2s/layer`, and its
    /// holder was abandoned after 32 s of a job that needs minutes. The request
    /// then succeeded on the standby, so the cost was a wasted 32 s of a 176 s
    /// request plus a needless failover — not a failure, which is why it went
    /// unnoticed. The peer-agnostic constants stay peer-agnostic; this only
    /// stops the fallback asking the wrong question.
    fn forward_is_prefill(kind: WorkKind, activation_bytes: usize, units: ActivationUnits) -> bool {
        match kind {
            WorkKind::Decode => false,
            WorkKind::Prefill | WorkKind::Delegated => match units {
                ActivationUnits::PromptBytes => true,
                ActivationUnits::HiddenStates => {
                    activation_bytes > PREFILL_ACTIVATION_THRESHOLD_BYTES
                }
            },
        }
    }

    /// Wait for a remote segment to return its result via the oneshot channel.
    ///
    /// `cancel` is the request's cancel flag when the caller has one: the wait
    /// ends with `inference::cancel::request_abandoned` the moment it flips,
    /// instead of running out the segment deadline for an answer nobody will
    /// read. The caller then owes the peer a `CancelInference`.
    /// `state` is required rather than optional because this is the ONE place a
    /// remote segment's outcome is observed, and the reliability figure it
    /// feeds is worthless if a call site can decline to report. See the
    /// recording arms below.
    ///
    /// `resend` is what this wait may do when the peer refuses the forward
    /// before any of it ran (`ForwardRefusal`) — see [`ResendOnRefusal`]. It is
    /// here, in the one wait every remote segment goes through, so that no
    /// path can be the one that forgot: that is FUTURE_WORK #92, a request lost
    /// to a link that had repaired itself a second later.
    // Nine primitives that name one wait; a struct would rename them at the
    // eight call sites without saying anything new.
    #[allow(clippy::too_many_arguments)]
    pub(super) async fn wait_for_result(
        state: &SharedState,
        rx: tokio::sync::oneshot::Receiver<LayerResult>,
        request_id: uuid::Uuid,
        segment_idx: usize,
        node_id: &crate::types::NodeId,
        num_layers: u32,
        activation_bytes: usize,
        budget: SegmentBudget,
        cancel: Option<&std::sync::Arc<std::sync::atomic::AtomicBool>>,
        resend: ResendOnRefusal<'_>,
    ) -> Result<LayerResult, SwarmError> {
        let mut rx = rx;
        // No later than the refused forward was sealed: the caller has handed
        // it to the network task already, and sealing happens there.
        let mut sent_at = std::time::Instant::now();
        let mut resent = false;
        loop {
            let result = Self::wait_for_one_result(
                state,
                rx,
                request_id,
                segment_idx,
                node_id,
                num_layers,
                activation_bytes,
                budget,
                cancel,
            )
            .await?;
            if result.refusal != Some(crate::types::ForwardRefusal::Undecryptable) {
                return Ok(result);
            }
            let (network_tx, target_peer_bytes, rebuild) = match resend {
                ResendOnRefusal::SameForward {
                    network_tx,
                    target_peer_bytes,
                    rebuild,
                } if !resent => (network_tx, target_peer_bytes, rebuild),
                ResendOnRefusal::SameForward { .. } => {
                    tracing::warn!(
                        request_id = %request_id,
                        segment = segment_idx,
                        node = %node_id,
                        "DIAG: peer could not decrypt the forward AGAIN after the link was \
                         re-keyed — not resending a second time"
                    );
                    return Ok(result);
                }
                ResendOnRefusal::Never(why) => {
                    tracing::info!(
                        request_id = %request_id,
                        segment = segment_idx,
                        node = %node_id,
                        why,
                        "DIAG: peer could not decrypt the forward — this path does not resend"
                    );
                    return Ok(result);
                }
            };
            let Some(new_rx) = resend_after_link_repair(
                state,
                request_id,
                segment_idx,
                node_id,
                sent_at,
                cancel,
                network_tx,
                target_peer_bytes,
                rebuild,
            )
            .await
            else {
                return Ok(result);
            };
            rx = new_rx;
            sent_at = std::time::Instant::now();
            resent = true;
        }
    }

    /// One wait on one forward — the body `wait_for_result` had before it could
    /// resend. Separate so a resend waits under exactly the same rules,
    /// recording and deadline as the first attempt.
    #[allow(clippy::too_many_arguments)]
    async fn wait_for_one_result(
        state: &SharedState,
        rx: tokio::sync::oneshot::Receiver<LayerResult>,
        request_id: uuid::Uuid,
        segment_idx: usize,
        node_id: &crate::types::NodeId,
        num_layers: u32,
        activation_bytes: usize,
        budget: SegmentBudget,
        cancel: Option<&std::sync::Arc<std::sync::atomic::AtomicBool>>,
    ) -> Result<LayerResult, SwarmError> {
        // Never about ourselves. A local segment runs in-process and does not
        // come through here, but the guard keeps a future caller from docking
        // this node's own reliability — `peer_expected_attempts` prices the
        // LOCAL candidate too, so a stray sample would make us route away from
        // ourselves.
        let is_remote = node_id != state.identity.node_id();
        let timeout = budget.duration();
        let send_time = std::time::Instant::now();
        tracing::info!(
            request_id = %request_id,
            segment = segment_idx,
            node = %node_id,
            timeout_secs = timeout.as_secs(),
            timeout_basis = budget.basis(),
            num_layers,
            activation_bytes,
            is_prefill = budget.is_prefill(),
            "DIAG: waiting for remote segment result"
        );
        let timed = crate::inference::cancel::unless_cancelled(
            async { Ok(tokio::time::timeout(timeout, rx).await) },
            cancel,
        )
        .await?;
        match timed {
            Ok(Ok(result)) => {
                let elapsed = send_time.elapsed();
                tracing::info!(
                    request_id = %request_id,
                    segment = segment_idx,
                    node = %node_id,
                    elapsed_ms = elapsed.as_millis() as u64,
                    tokens = result.token_ids.len(),
                    activations_bytes = result.activations.len(),
                    finish = ?result.finish_reason,
                    "DIAG: segment result received"
                );
                // Something arrived — but not necessarily from the peer.
                //
                // A peer REFUSING (out of memory, a missing shard) lands here
                // carrying its reason in `finish_reason`, and that IS an intact
                // delivery: the distinction is compute against transport, the
                // same one `failure_is_penalty_worthy` draws, and pricing a
                // refusal as a lossy link would steer traffic away from a peer
                // whose network is fine.
                //
                // What also lands here is a result THIS NODE manufactured to
                // end the wait — the ACK fast-fail sweep, a departed peer, a
                // closed pipeline stream. Those are the transport failures the
                // reliability figure exists for, and they are the common case:
                // the ACK deadline is 10-90 s inside a segment budget that runs
                // to 300 s, so a peer whose link is dead is abandoned here long
                // before the timeout arm below can fire. `locally_constructed`
                // is the structural discriminator — it cannot survive either
                // codec, so anything the peer actually sent reads false.
                let outcome = if result.locally_constructed {
                    SegmentOutcome::AbandonedLocally
                } else {
                    SegmentOutcome::Returned
                };
                note_segment_delivery(state, is_remote, node_id, outcome, budget.is_prefill());
                Ok(result)
            }
            Ok(Err(_)) => {
                let elapsed = send_time.elapsed();
                tracing::error!(
                    request_id = %request_id,
                    segment = segment_idx,
                    node = %node_id,
                    elapsed_ms = elapsed.as_millis() as u64,
                    "DIAG: response channel DROPPED — sender gone before result"
                );
                // Reported, and classified as saying NOTHING — the
                // suppression is `segment_delivery_verdict`'s `SenderDropped =>
                // None`, not an absence of a call here. (It read "deliberately
                // NOT recorded" directly above the call that records it, which
                // is precisely the misdirection that would defeat an audit of
                // where deliveries are observed.)
                //
                // This node's own machinery drops the sender — a failover
                // resolving elsewhere, a cancelled request, the stale-forward
                // sweep — so the peer may have done nothing wrong, and charging
                // it for our bookkeeping is the mis-attribution
                // `failure_is_penalty_worthy` exists to avoid.
                note_segment_delivery(
                    state,
                    is_remote,
                    node_id,
                    SegmentOutcome::SenderDropped,
                    budget.is_prefill(),
                );
                // OURS, by the same reasoning the comment above gives for not
                // charging the peer: this node's own machinery dropped the
                // sender. `Internal` keeps that — it is in
                // `failure_is_penalty_worthy`'s local-only list exactly as
                // `PipelineError` was, and the message is unchanged, so neither
                // the penalty nor the retry moves.
                //
                // **Deliberately NOT `ServiceUnavailable`**, which reads as
                // "a peer could not serve" (`router::remote_peer_could_not_serve`)
                // and would bar that peer from the retry — the mis-attribution
                // this site already refuses to make. What changes is the HINT:
                // `PipelineError` told the reader to fetch a missing model part,
                // which cannot help when we cancelled or failed over elsewhere
                // (`docs/FUTURE_WORK.md` #86, gotcha #295).
                Err(SwarmError::Internal("Response channel dropped".into()))
            }
            Err(_) => {
                tracing::error!(
                    request_id = %request_id,
                    segment = segment_idx,
                    node = %node_id,
                    timeout_secs = timeout.as_secs(),
                    num_layers,
                    activation_bytes,
                    "DIAG: segment TIMED OUT — no result received"
                );
                // The peer took the forward and said nothing inside a deadline
                // sized for it. That is the delivery failure the reliability
                // figure is FOR, and `segment_timeout_error` already classifies
                // it as penalty-worthy for the same reason.
                note_segment_delivery(
                    state,
                    is_remote,
                    node_id,
                    SegmentOutcome::TimedOut,
                    budget.is_prefill(),
                );
                // And bar it from THIS request's retry. A re-plan that can
                // re-pick the peer that just sat on a forward for the whole
                // deadline waits that deadline again — Envoy's `previous_hosts`
                // retry predicate exists for exactly this: a host that just
                // failed is likely to keep failing for a while, so the retry
                // goes elsewhere or nowhere. `failover_segment` bars the failed
                // node at each of its exits already; this covers the paths
                // that never reach it — the speculative verify rounds, which
                // propagate this error straight to the router. Scoped to the
                // request, released with its state, never about ourselves.
                if is_remote {
                    state.blacklist_holder_for_request(request_id, node_id);
                }
                Err(segment_timeout_error(timeout.as_secs(), num_layers))
            }
        }
    }
}

/// What [`PipelineExecutor::wait_for_result`] may do when the peer refuses a
/// forward before any of it ran ([`crate::types::ForwardRefusal`]).
///
/// Required, not an `Option` that defaults to "never": a wait that must not
/// resend says so at its call site, with the reason in words
/// (`architecture.md`, "make the wrong call unrepresentable").
///
/// **Why the same node, and why that is safe.** A forward the peer could not
/// decrypt never reached its worker, so the peer's KV cache for the
/// conversation is exactly as it was — sending the same step again is correct
/// at any point in a reply, where handing the segment to ANOTHER machine is
/// not (`failover_can_restore_state`). gRPC makes the same call for an RPC the
/// server application never saw (gRFC A6, "transparent retries": retried once,
/// outside the retry policy, not counted as a failure). And the refusing node
/// has already armed the repair, so the one thing that made the forward fail
/// is being fixed as we wait — WireGuard likewise holds a packet it cannot
/// send until the handshake completes, then sends it (`staged_packet_queue`).
pub(crate) enum ResendOnRefusal<'a> {
    /// Build the same forward again and send it to the same node, once, when
    /// the link has been re-keyed. A closure rather than a stored copy so the
    /// forward is rebuilt from what the caller already holds and nothing is
    /// spent on the path where no refusal ever comes.
    SameForward {
        network_tx: &'a tokio::sync::mpsc::Sender<crate::types::NetworkCommand>,
        target_peer_bytes: &'a [u8],
        rebuild: &'a (dyn Fn() -> crate::types::LayerForward + Send + Sync),
    },
    /// This wait must not resend; the reason is logged if a refusal arrives.
    Never(&'static str),
}

/// Floor on how long to wait for a refusing peer's repair to land. The repair
/// is one ephemeral exchange plus its confirmation — a round trip and a half
/// — started by the peer's rotation task the moment the refusal armed it.
const LINK_REPAIR_WAIT_MIN: Duration = Duration::from_secs(2);
/// Ceiling on the same wait. Past it the link is not repairing, and the
/// refusal goes to the caller's ordinary handling, which is where it went
/// before any of this existed.
const LINK_REPAIR_WAIT_MAX: Duration = Duration::from_secs(10);
/// The wait when we have no round-trip figure for the peer at all.
const LINK_REPAIR_WAIT_UNMEASURED: Duration = Duration::from_secs(5);
/// How often the wait looks for the new key.
const LINK_REPAIR_POLL: Duration = Duration::from_millis(20);

/// How long to wait for the link to `node` to be re-keyed.
///
/// Scaled by the peer's measured round trip rather than fixed (`architecture.md`
/// § Timeouts, "bound what actually varies"): the repair costs round trips, and
/// this fleet spans ~1 ms LAN links and ~350 ms intercontinental ones.
fn link_repair_wait(state: &SharedState, node: &crate::types::NodeId) -> Duration {
    let srtt_ms = state.peer_registry.get(node).and_then(|p| p.ack_srtt_ms);
    match srtt_ms {
        Some(ms) => (LINK_REPAIR_WAIT_MIN + Duration::from_millis(4 * u64::from(ms)))
            .min(LINK_REPAIR_WAIT_MAX),
        None => LINK_REPAIR_WAIT_UNMEASURED,
    }
}

/// Wait for the link to `node_id` to be re-keyed, then send the refused
/// forward to it again. `Some` is the new result channel, registered before the
/// send; `None` means nothing was sent and the refusal stands.
#[allow(clippy::too_many_arguments)]
async fn resend_after_link_repair(
    state: &SharedState,
    request_id: uuid::Uuid,
    segment_idx: usize,
    node_id: &crate::types::NodeId,
    sent_at: std::time::Instant,
    cancel: Option<&std::sync::Arc<std::sync::atomic::AtomicBool>>,
    network_tx: &tokio::sync::mpsc::Sender<crate::types::NetworkCommand>,
    target_peer_bytes: &[u8],
    rebuild: &(dyn Fn() -> crate::types::LayerForward + Send + Sync),
) -> Option<tokio::sync::oneshot::Receiver<LayerResult>> {
    let wait = link_repair_wait(state, node_id);
    // Tokio's clock, so a test can run the whole bound in paused time.
    let started = tokio::time::Instant::now();
    tracing::info!(
        request_id = %request_id,
        segment = segment_idx,
        node = %node_id,
        wait_ms = wait.as_millis() as u64,
        "DIAG: peer could not decrypt the forward — waiting for the link repair to send it again"
    );
    // Sealed after this, the forward goes out under the key the peer just
    // agreed; sealed before, under the one it refused (`rekeyed_since`).
    while !state.session_manager.rekeyed_since(node_id, sent_at) {
        if cancel.is_some_and(|c| c.load(std::sync::atomic::Ordering::Relaxed)) {
            return None;
        }
        if started.elapsed() >= wait {
            tracing::warn!(
                request_id = %request_id,
                segment = segment_idx,
                node = %node_id,
                waited_ms = started.elapsed().as_millis() as u64,
                "DIAG: link to the peer was not re-keyed in time — not resending"
            );
            return None;
        }
        tokio::time::sleep(LINK_REPAIR_POLL).await;
    }
    // The refusal consumed the caller's waiter. Pinned to the same node, as
    // the caller's was: this forward is for it and for nobody else.
    let forward = rebuild();
    let (tx, rx) = tokio::sync::oneshot::channel();
    state.pending_layer_results.insert(
        request_id,
        crate::daemon::state::PendingLayerResult {
            tx,
            awaiting: Some(node_id.clone()),
            chain_members: Vec::new(),
            expects_step: Some(crate::daemon::state::ExpectedStep::one(
                forward.index_pos,
                forward.layer_range,
            )),
        },
    );
    if network_tx
        .send(crate::types::NetworkCommand::SendTensor {
            target_peer_bytes: target_peer_bytes.to_vec(),
            forward,
        })
        .await
        .is_err()
    {
        state.pending_layer_results.remove(&request_id);
        return None;
    }
    tracing::info!(
        request_id = %request_id,
        segment = segment_idx,
        node = %node_id,
        repaired_after_ms = started.elapsed().as_millis() as u64,
        "DIAG: link re-keyed — sent the refused forward again to the same node"
    );
    Some(rx)
}

/// Record what one remote segment outcome says about a peer's link, if
/// anything. The ONE place a segment forward feeds the reliability figure.
fn note_segment_delivery(
    state: &SharedState,
    is_remote: bool,
    node_id: &crate::types::NodeId,
    outcome: SegmentOutcome,
    is_prefill: bool,
) {
    if !is_remote {
        return;
    }
    let Some(intact) = segment_delivery_verdict(outcome) else {
        return;
    };
    // An INTACT sample is taken only on the forward that actually tests the
    // link — the prompt pass. A failure is always taken, whenever it happens.
    //
    // Two reasons, and the first is why the figure was inert as shipped. This
    // runs once per segment PER TOKEN, while the whole-model fast path
    // (`remote_generate`) records once per reply; both feed one EMA at
    // `ALPHA = 0.3`, where `1 - 0.7^n` passes 0.99 by fifteen samples. A
    // 150-token reply over three segments is 450 of them, so a peer that fails
    // once per request after streaming a hundred tokens was priced at ~1.0 and
    // the multiplier stayed inert on exactly the path it was added for. Taking
    // the prompt pass gives one sample per request per peer, which is the same
    // event the fast path counts.
    //
    // The second is cost: this is the per-token forward path, and an
    // unconditional `record_peer_delivery` is a DashMap exclusive shard lock
    // plus a `NodeId` clone per segment per token, contended across concurrent
    // requests to the same peer.
    //
    // Nothing is lost by it. The prompt pass is the largest forward a request
    // makes and the one whose delivery says most about the link, and a failure
    // at any point is still recorded in full.
    if intact && !is_prefill {
        return;
    }
    state.record_peer_delivery(node_id, intact);
}

/// How a remote segment forward ended, as far as the PEER'S LINK is concerned.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum SegmentOutcome {
    /// A `LayerResult` came back FROM THE PEER — including one carrying a
    /// refusal, which is still a delivery.
    Returned,
    /// A `LayerResult` came back, but this node manufactured it to end the
    /// wait: the ACK fast-fail sweep, a peer whose connection closed and whose
    /// re-dial failed, or a closed pipeline stream. The peer sent nothing.
    AbandonedLocally,
    /// The deadline expired with nothing.
    TimedOut,
    /// Our own sender was dropped before a result arrived.
    SenderDropped,
}

/// Is this outcome a statement about the peer's link, and if so what?
///
/// Extracted rather than written inline at the three arms for the reason
/// [`segment_timeout_error`] was: the classification is the whole point and it
/// is invisible at the call site — and here the alternative is a test that must
/// wait out a real deadline (30 s, measured) to assert one boolean.
///
/// - `Returned` is intact even when the payload is a refusal. A peer that
///   answers "out of memory" has delivered perfectly; the distinction is
///   compute against transport, the one `failure_is_penalty_worthy` draws.
///   Pricing a refusal as a lossy link steers traffic away from a peer whose
///   network is fine.
/// - `AbandonedLocally` is a delivery failure and is the COMMON one. The three
///   paths that abandon a forward hand the waiter a manufactured
///   `LayerResult::error` rather than letting it expire, so before this outcome
///   existed they were all scored `Returned` — a peer whose link had died was
///   credited with a perfect delivery, and the reliability term could observe
///   almost nothing. `locally_constructed` is what tells the two apart.
/// - `TimedOut` is the same failure reached the slow way — the peer took the
///   forward and went quiet with nothing abandoning it on our side. Rarer than
///   it looks: the ACK deadline is well inside the segment budget.
/// - `SenderDropped` says NOTHING about the peer. This node's own machinery
///   drops the sender: a failover resolving elsewhere, a cancelled request, the
///   stale-forward sweep. Charging the peer for our bookkeeping is precisely
///   the mis-attribution to avoid.
pub(super) fn segment_delivery_verdict(outcome: SegmentOutcome) -> Option<bool> {
    match outcome {
        SegmentOutcome::Returned => Some(true),
        SegmentOutcome::AbandonedLocally | SegmentOutcome::TimedOut => Some(false),
        SegmentOutcome::SenderDropped => None,
    }
}

/// The error a segment forward gets when the peer holding it says nothing
/// before the deadline.
///
/// **`PeerUnresponsive`, not `PipelineError`.** The peer accepted the forward
/// and then went quiet, which is the exact case that variant exists for. Worn
/// as a `PipelineError` it was wrong in four ways at once: the caller got
/// **500 `server_error`** — a peer going quiet on the other side of the world
/// reported as a bug in the node they are talking to; this node logged it at
/// `ERROR`, which means "I am broken", for the same reason; it inherited
/// `PipelineError`'s exemption from `failure_is_penalty_worthy`, so the silent
/// peer was never docked even though that function's own comment names
/// "timeouts waiting on a peer" as what the penalty is for; and the hint was
/// the generic one rather than the "a fresh request usually routes to a
/// different holder" that actually helps.
///
/// Observed live 2026-08-25 on `gemma-2-2b-it`: the single holder answered the
/// prefill in 5.9 s, then said nothing for the whole 52 s decode deadline, and
/// the request surfaced to the caller as a 500 with the peer un-penalised.
///
/// A named function rather than an inline `format!` so the variant is pinned by
/// a test — the classification is the whole point, and it is invisible at the
/// call site.
fn segment_timeout_error(timeout_secs: u64, num_layers: u32) -> SwarmError {
    SwarmError::PeerUnresponsive(format!(
        "Timed out waiting for segment result ({timeout_secs}s, {num_layers} layers)"
    ))
}

#[cfg(test)]
mod segment_budget_tests {
    use super::*;
    use crate::types::{ModelId, NodeId};
    use std::sync::Arc;

    fn test_state() -> Arc<SharedState> {
        use crate::identity::Identity;
        use crate::inference::executor::ModelExecutor;
        use crate::storage::db::Database;
        use tokio::sync::Mutex;

        let identity = Identity::generate();
        let temp = tempfile::tempdir().unwrap();
        let db = Database::open(temp.path()).unwrap();
        let executor = Arc::new(Mutex::new(ModelExecutor::new()));
        let (state, _, _) = SharedState::new(
            crate::config::Config::default(),
            identity,
            db,
            executor,
            None,
        );
        state
    }

    /// A budget for a forward that resolves immediately. The deadline is
    /// floored at 30 s by `for_forward`, which is fine for every arm except a
    /// real expiry — see [`SegmentBudget::with_deadline_for_test`].
    fn budget(
        state: &Arc<SharedState>,
        node: &NodeId,
        model: &ModelId,
        kind: WorkKind,
    ) -> SegmentBudget {
        state.record_peer_segment_latency(node, model, kind, 1, 1, 8);
        SegmentBudget::for_forward(
            state,
            node,
            model,
            kind,
            1,
            8,
            match kind {
                WorkKind::Prefill => ActivationUnits::PromptBytes,
                _ => ActivationUnits::HiddenStates,
            },
        )
    }

    /// What a peer actually sent, as opposed to something this node built.
    /// `locally_constructed` cannot survive either codec, so a result off the
    /// wire always reads false — this reproduces that shape in-process.
    fn as_if_from_the_peer(mut r: LayerResult) -> LayerResult {
        r.locally_constructed = false;
        r
    }

    async fn wait_on(
        state: &Arc<SharedState>,
        node: &NodeId,
        budget: SegmentBudget,
        result: Option<LayerResult>,
    ) -> Result<LayerResult, SwarmError> {
        wait_on_request(state, uuid::Uuid::new_v4(), node, budget, result).await
    }

    /// [`wait_on`] for a NAMED request, so what the wait left behind under
    /// that id can be inspected.
    async fn wait_on_request(
        state: &Arc<SharedState>,
        request_id: uuid::Uuid,
        node: &NodeId,
        budget: SegmentBudget,
        result: Option<LayerResult>,
    ) -> Result<LayerResult, SwarmError> {
        let (tx, rx) = tokio::sync::oneshot::channel::<LayerResult>();
        match result {
            Some(r) => tx.send(r).unwrap(),
            // Held open so the deadline is what ends the wait, not a dropped
            // sender — those are different outcomes and only one of them is
            // the peer's fault.
            None => std::mem::forget(tx),
        }
        PipelineExecutor::wait_for_result(
            state,
            rx,
            request_id,
            0,
            node,
            1,
            8,
            budget,
            None,
            ResendOnRefusal::Never("test"),
        )
        .await
    }

    /// The defect this pins: `record_peer_delivery` was called ONLY from
    /// `remote_generate`, the whole-model fast path, so a peer serving pipeline
    /// SEGMENTS produced no reliability samples at all — and the
    /// `expected_attempts` multiplier in `parallax::vertex_cost`, which exists
    /// precisely to steer chains away from a lossy link, was inert on the one
    /// path that routes chains.
    ///
    /// Found by a contributor's netem measurements (issue #21): a peer at 60 ms
    /// with 3% loss out-sorts one at 81 ms with none while being 2.9x slower on
    /// a 513 KB round trip. Loss is what a health-check ping cannot see.
    ///
    /// The COUNT is the assertion, not the multiplier: an all-intact peer and a
    /// peer nothing is recording for both price at 1.0, which is why this went
    /// unnoticed.
    #[tokio::test]
    async fn a_returned_segment_is_recorded_as_an_intact_delivery() {
        let state = test_state();
        let node = NodeId([7u8; 32]);
        let model = ModelId("m".into());
        let budget = budget(&state, &node, &model, WorkKind::Prefill);
        assert_eq!(
            state.peer_delivery_samples(&node),
            0,
            "nothing recorded yet"
        );

        wait_on(
            &state,
            &node,
            budget,
            Some(as_if_from_the_peer(LayerResult::error(
                uuid::Uuid::new_v4(),
                "out of memory",
            ))),
        )
        .await
        .expect("a result that arrived is delivered, whatever it says");

        assert_eq!(
            state.peer_delivery_samples(&node),
            1,
            "the segment path must feed the reliability figure"
        );
        assert_eq!(
            state.peer_expected_attempts(&node),
            1.0,
            "and a peer that ANSWERED — even to refuse — is not a lossy link"
        );
    }

    /// The half that made the figure inert in v0.3.164.
    ///
    /// The ACK fast-fail sweep, a departed peer and a closed pipeline stream
    /// all end the wait by handing it a manufactured `LayerResult::error`
    /// rather than letting the deadline expire. Those arrive in the SAME arm as
    /// a peer's own refusal, so they were all scored as intact deliveries — and
    /// since the ACK deadline (10-90 s) sits well inside the segment budget
    /// (300 s), that was the normal way a dead link was observed. The peer was
    /// credited with a perfect delivery every time.
    #[tokio::test]
    async fn a_forward_this_node_abandoned_is_not_credited_to_the_peer() {
        let state = test_state();
        let node = NodeId([8u8; 32]);
        let model = ModelId("m".into());
        let budget = budget(&state, &node, &model, WorkKind::Prefill);

        // Exactly what `fail_tensor_forward` puts on the waiter.
        let abandoned = LayerResult::error(uuid::Uuid::new_v4(), "peer never acknowledged");
        assert!(
            abandoned.locally_constructed,
            "the discriminator must be set by the constructor those paths use"
        );
        let _ = wait_on(&state, &node, budget, Some(abandoned)).await;

        assert_eq!(state.peer_delivery_samples(&node), 1);
        assert!(
            state.peer_expected_attempts(&node) > 1.0,
            "a forward we gave up on must cost the peer, or the term cannot see \
             the failure it exists for"
        );
    }

    /// The cadence rule. An intact sample is taken on the prompt pass only:
    /// this runs once per segment per token, while the whole-model fast path
    /// records once per reply, and at `ALPHA = 0.3` a hundred decode samples
    /// bury the one failure that ended the request.
    #[tokio::test]
    async fn an_intact_decode_forward_adds_no_sample() {
        let state = test_state();
        let node = NodeId([9u8; 32]);
        let model = ModelId("m".into());
        let budget = budget(&state, &node, &model, WorkKind::Decode);
        assert!(!budget.is_prefill(), "control: this is a decode forward");

        wait_on(
            &state,
            &node,
            budget,
            Some(as_if_from_the_peer(LayerResult::error(
                uuid::Uuid::new_v4(),
                "fine",
            ))),
        )
        .await
        .unwrap();

        assert_eq!(
            state.peer_delivery_samples(&node),
            0,
            "a good decode step is not a fresh statement about the link"
        );
    }

    /// ...and the control in the other direction: a FAILURE is always recorded,
    /// whenever it happens. Suppressing those would be the bug this rule exists
    /// to avoid rather than a cheaper version of it.
    #[tokio::test]
    async fn a_failed_decode_forward_is_still_recorded() {
        let state = test_state();
        let node = NodeId([10u8; 32]);
        let model = ModelId("m".into());
        let budget = budget(&state, &node, &model, WorkKind::Decode);

        let _ = wait_on(
            &state,
            &node,
            budget,
            Some(LayerResult::error(uuid::Uuid::new_v4(), "stream closed")),
        )
        .await;

        assert_eq!(state.peer_delivery_samples(&node), 1);
        assert!(state.peer_expected_attempts(&node) > 1.0);
    }

    /// The timeout arm, end to end. It had no coverage at all: `for_forward`
    /// floors every budget at `SEGMENT_TIMEOUT_MIN_SECS` (30 s), so the helper
    /// that claimed to provoke a timeout "without sleeping" could not have.
    #[tokio::test]
    async fn a_deadline_that_expires_counts_against_the_peer() {
        let state = test_state();
        let node = NodeId([11u8; 32]);
        let budget = SegmentBudget::with_deadline_for_test(Duration::from_millis(20), true);

        let err = wait_on(&state, &node, budget, None)
            .await
            .expect_err("nothing was ever sent");
        assert!(
            matches!(err, SwarmError::PeerUnresponsive(_)),
            "a peer that took the forward and went quiet, not our bug: {err:?}"
        );
        assert_eq!(state.peer_delivery_samples(&node), 1);
        assert!(state.peer_expected_attempts(&node) > 1.0);
    }

    /// The peer that sat on the forward for the whole deadline is barred from
    /// this request's retry — otherwise the re-plan can pick it again and the
    /// request waits the same deadline twice. Scoped to the request: another
    /// request id is free to use it.
    #[tokio::test]
    async fn a_deadline_that_expires_bars_the_peer_from_this_request() {
        let state = test_state();
        let node = NodeId([13u8; 32]);
        let request_id = uuid::Uuid::new_v4();
        let budget = SegmentBudget::with_deadline_for_test(Duration::from_millis(20), true);

        let err = wait_on_request(&state, request_id, &node, budget, None)
            .await
            .expect_err("nothing was ever sent");
        assert!(matches!(err, SwarmError::PeerUnresponsive(_)));
        assert!(
            state.holder_blacklisted_for_request(request_id, &node),
            "the silent peer must not be re-picked by this request's retry"
        );
        assert!(
            !state.holder_blacklisted_for_request(uuid::Uuid::new_v4(), &node),
            "barred for this request only — it may be healthy for everyone else"
        );
    }

    /// The control: our own dropped sender says nothing about the peer, so it
    /// is not barred either — a failover resolving elsewhere or a cancelled
    /// request must not cost a healthy holder the retry.
    #[tokio::test]
    async fn a_dropped_response_channel_does_not_bar_the_peer() {
        let state = test_state();
        let node = NodeId([14u8; 32]);
        let model = ModelId("m".into());
        let request_id = uuid::Uuid::new_v4();
        let budget = budget(&state, &node, &model, WorkKind::Prefill);

        let (tx, rx) = tokio::sync::oneshot::channel::<LayerResult>();
        drop(tx);
        let err = PipelineExecutor::wait_for_result(
            &state,
            rx,
            request_id,
            0,
            &node,
            1,
            8,
            budget,
            None,
            ResendOnRefusal::Never("test"),
        )
        .await
        .expect_err("the sender was dropped");
        assert!(!matches!(err, SwarmError::PeerUnresponsive(_)));
        assert!(
            !state.holder_blacklisted_for_request(request_id, &node),
            "a dropped sender is our bookkeeping, not the peer's silence"
        );
    }

    /// Every verdict in one place, so the timeout arm is pinned without a test
    /// that waits out a real deadline (30 s, measured).
    #[test]
    fn only_a_deadline_that_expired_counts_against_the_peer() {
        assert_eq!(
            segment_delivery_verdict(SegmentOutcome::Returned),
            Some(true)
        );
        assert_eq!(
            segment_delivery_verdict(SegmentOutcome::TimedOut),
            Some(false)
        );
        assert_eq!(
            segment_delivery_verdict(SegmentOutcome::SenderDropped),
            None,
            "our own dropped sender says nothing about the peer"
        );
    }

    /// The attribution control. This node's OWN machinery drops the sender — a
    /// failover resolving elsewhere, a cancelled request, the stale-forward
    /// sweep — so the peer may have done nothing wrong. Charging it for our
    /// bookkeeping is the mis-attribution `failure_is_penalty_worthy` exists to
    /// prevent.
    #[tokio::test]
    async fn a_dropped_response_channel_is_not_charged_to_the_peer() {
        let state = test_state();
        let node = NodeId([12u8; 32]);
        let model = ModelId("m".into());
        let budget = budget(&state, &node, &model, WorkKind::Prefill);

        let (tx, rx) = tokio::sync::oneshot::channel::<LayerResult>();
        drop(tx);
        let err = PipelineExecutor::wait_for_result(
            &state,
            rx,
            uuid::Uuid::new_v4(),
            0,
            &node,
            1,
            8,
            budget,
            None,
            ResendOnRefusal::Never("test"),
        )
        .await
        .expect_err("a dropped sender is still a failure for this request");
        // Ours, so `Internal` — this node's own machinery dropped the sender.
        // It was `PipelineError` until 2026-09-19, which handed the reader that
        // variant's hint: fetch the model part that is missing. Nothing is
        // missing here (FUTURE_WORK #86).
        assert!(matches!(err, SwarmError::Internal(_)), "{err:?}");
        assert!(
            crate::error::error_hint(&err).is_none(),
            "our own bug has no advice to offer, and the hint it used to carry \
             pointed at a condition this is not: {err:?}"
        );

        assert_eq!(
            state.peer_delivery_samples(&node),
            0,
            "our own dropped sender must not even be counted as a sample"
        );
        assert_eq!(state.peer_expected_attempts(&node), 1.0);
    }

    /// The 2026-08-01 failure: a peer needing ~120s to load an 8B model was cut
    /// off by a flat 120s deadline even though it computed the segment in 10s.
    /// An unseen (peer, model) pair must be given room for that load.
    #[test]
    fn a_cold_peer_gets_room_to_load_the_model() {
        let state = test_state();
        let node = NodeId([1u8; 32]);
        let model = ModelId("meta-llama-3.1-8b-instruct-q4-k-m".into());

        let cold = SegmentBudget::for_forward(
            &state,
            &node,
            &model,
            WorkKind::Prefill,
            8,
            213_268,
            ActivationUnits::HiddenStates,
        );
        assert!(
            cold.duration() > Duration::from_secs(130),
            "a cold peer must outlast the ~120s model load that broke this, got {:?}",
            cold.duration()
        );
        assert!(cold.basis().contains("coldload"));
    }

    /// Once the peer has served the model and been measured, the deadline
    /// tightens to its actual speed rather than a worst-case constant.
    #[test]
    fn a_measured_warm_peer_gets_a_deadline_matched_to_its_speed() {
        let state = test_state();
        let node = NodeId([2u8; 32]);
        let model = ModelId("m".into());

        // Peer prefilled this shape in 10.2s — the real measurement.
        state.record_peer_segment_latency(&node, &model, WorkKind::Prefill, 10_199, 8, 213_268);

        let warm = SegmentBudget::for_forward(
            &state,
            &node,
            &model,
            WorkKind::Prefill,
            8,
            213_268,
            ActivationUnits::HiddenStates,
        );
        assert_eq!(warm.basis(), "measured");
        // 10.2s × 3 safety = ~30.6s, and nothing like the 240s cold budget.
        assert!(
            warm.duration() >= Duration::from_secs(30) && warm.duration() < Duration::from_secs(60),
            "expected a deadline near 3x the measured 10.2s, got {:?}",
            warm.duration()
        );
    }

    /// A slower peer must be given proportionally longer, which is the whole
    /// point — the flat budget is what cut off a healthy CPU node.
    #[test]
    fn a_slower_peer_is_given_longer_than_a_faster_one() {
        let state = test_state();
        let model = ModelId("m".into());
        let slow = NodeId([3u8; 32]);
        let fast = NodeId([4u8; 32]);

        state.record_peer_segment_latency(&slow, &model, WorkKind::Prefill, 60_000, 8, 213_268);
        state.record_peer_segment_latency(&fast, &model, WorkKind::Prefill, 2_000, 8, 213_268);

        let mk = |n: &NodeId| {
            SegmentBudget::for_forward(
                &state,
                n,
                &model,
                WorkKind::Prefill,
                8,
                213_268,
                ActivationUnits::HiddenStates,
            )
            .duration()
        };
        assert!(
            mk(&slow) > mk(&fast),
            "slow {:?} should outrank fast {:?}",
            mk(&slow),
            mk(&fast)
        );
    }

    /// A forward carrying the prompt performs the whole prefill, however few
    /// bytes the prompt is — so it must not be budgeted at the decode rate.
    ///
    /// Measured on the live swarm 2026-08-29: a 4728-token prompt reached
    /// segment 0 as `activation_bytes=24045`, fell under
    /// `PREFILL_ACTIVATION_THRESHOLD_BYTES` (a hidden-state scale), and was
    /// given 16 x 2s. Its holder was abandoned 32 s into a job needing minutes.
    #[test]
    fn a_forward_carrying_the_prompt_is_budgeted_as_a_prefill() {
        let state = test_state();
        let node = NodeId([9u8; 32]);
        let model = ModelId("m".into());
        // The exact shape observed live.
        let (prompt_bytes, layers) = (24_045usize, 16u32);

        let mk = |units, kind| {
            SegmentBudget::for_forward(&state, &node, &model, kind, layers, prompt_bytes, units)
        };
        let prompt = mk(ActivationUnits::PromptBytes, WorkKind::Prefill);
        assert!(
            prompt.is_prefill(),
            "a forward handed the prompt itself IS the prefill"
        );

        // Control: the SAME byte count as hidden states really is about one
        // token's worth, and must still be budgeted as a decode. This is what
        // makes the assertion above about the units rather than the size.
        let hidden = mk(ActivationUnits::HiddenStates, WorkKind::Decode);
        assert!(!hidden.is_prefill());
        assert!(
            prompt.duration() > hidden.duration(),
            "prefill budget {:?} must exceed the decode budget {:?} at identical \
             byte counts — before the fix they were equal",
            prompt.duration(),
            hidden.duration()
        );
    }

    /// The coefficient is measured in hidden-state bytes. A first-segment
    /// forward carries raw prompt bytes, so predicting from the coefficient
    /// would be a unit error, not an approximation.
    #[test]
    fn prompt_byte_forwards_do_not_use_the_hidden_state_coefficient() {
        let state = test_state();
        let node = NodeId([5u8; 32]);
        let model = ModelId("m".into());
        state.record_peer_segment_latency(&node, &model, WorkKind::Prefill, 10_199, 8, 213_268);

        let b = SegmentBudget::for_forward(
            &state,
            &node,
            &model,
            WorkKind::Prefill,
            8,
            256, // a short prompt in raw bytes
            ActivationUnits::PromptBytes,
        );
        assert_eq!(
            b.basis(),
            "default",
            "raw-prompt units must fall back rather than mis-apply the coefficient"
        );
    }

    /// A decode step of a remote FIRST segment carries the sampled token as
    /// `PromptBytes` — that is the unit segment 0 takes — and was budgeted as
    /// a prefill for it: 15 s a layer instead of 2. Measured on the live swarm
    /// 2026-09-01 (gotcha #434): a 16-layer segment 0 took a decode step, went
    /// silent, and the pipeline waited 240 s with a standby idle throughout.
    #[test]
    fn a_decode_step_to_the_first_segment_is_budgeted_as_a_decode() {
        let state = test_state();
        let node = NodeId([11u8; 32]);
        let model = ModelId("m".into());
        // A prefill sample marks the peer warm (so the cold-load allowance does
        // not blur the number) while leaving its decode speed unmeasured, which
        // is what sends the budget to the peer-agnostic fallback.
        state.record_peer_segment_latency(&node, &model, WorkKind::Prefill, 4_900, 16, 146);
        // The exact shape observed live: one packed token id, 16 layers.
        let (token_bytes, layers) = (8usize, 16u32);
        let mk = |kind| {
            SegmentBudget::for_forward(
                &state,
                &node,
                &model,
                kind,
                layers,
                token_bytes,
                ActivationUnits::PromptBytes,
            )
        };

        let decode = mk(WorkKind::Decode);
        assert!(
            !decode.is_prefill(),
            "a decode step is a decode step whatever it carries"
        );
        assert_eq!(decode.basis(), "default");
        assert_eq!(
            decode.duration(),
            Duration::from_secs(layers as u64 * DECODE_SECS_PER_LAYER),
            "16 layers at the decode rate — before the fix this was 240 s"
        );

        // Control: the prompt pass to the same segment, carrying the same
        // units, is still the prefill. The kind is what changed the answer.
        let prefill = mk(WorkKind::Prefill);
        assert!(prefill.is_prefill());
        assert!(prefill.duration() > decode.duration());
    }

    /// The decode coefficient is ms per layer and never reads the byte count,
    /// so a decode step to segment 0 can be sized from what this peer has
    /// actually been measured doing — the same "measured" basis every later
    /// segment already gets. Before the fix the `PromptBytes` units sent it to
    /// the fallback unconditionally, and the measurement was never consulted.
    #[test]
    fn a_decode_step_to_the_first_segment_uses_the_measured_decode_speed() {
        let state = test_state();
        let node = NodeId([12u8; 32]);
        let model = ModelId("m".into());
        // 16 layers in 600 ms, as a decode step: ~37 ms a layer. Recorded
        // twice because the FIRST sample finds the peer cold and is dropped —
        // a cold decode sample is a load time wearing a compute figure's
        // clothes (`PeerSpeed::observe`) — while marking the peer warm, so
        // the second is the one that counts. Exactly production's order.
        state.record_peer_segment_latency(&node, &model, WorkKind::Decode, 600, 16, 8);
        state.record_peer_segment_latency(&node, &model, WorkKind::Decode, 600, 16, 8);

        let decode = SegmentBudget::for_forward(
            &state,
            &node,
            &model,
            WorkKind::Decode,
            16,
            8,
            ActivationUnits::PromptBytes,
        );
        assert_eq!(
            decode.basis(),
            "measured",
            "a measured decode speed applies to the first segment as it does to the rest"
        );
        assert!(!decode.is_prefill());

        // Control: the prompt pass in the same units still refuses to predict
        // from a coefficient that is not in those units.
        let prefill = SegmentBudget::for_forward(
            &state,
            &node,
            &model,
            WorkKind::Prefill,
            16,
            146,
            ActivationUnits::PromptBytes,
        );
        assert_eq!(prefill.basis(), "default");
    }

    /// The transport reaps an outstanding forward at `RR_REQUEST_TIMEOUT_SECS`
    /// and libp2p fails the send at the same point, so a budget larger than
    /// that waits for a result that can no longer arrive.
    ///
    /// This is the regression v0.3.60 shipped: the pipeline learned to size
    /// deadlines from measured peer speed and could allow 600s+, while the
    /// network manager still reaped the same forward on a `layers x 15s`
    /// formula — 120s for an 8-layer segment. A legitimately slow peer was
    /// killed at 130s against a 600s budget, twice, and the empty result
    /// surfaced as "Tensor bytes too short". Both sides are now bounded by the
    /// transport, and neither recomputes the other's number.
    #[test]
    fn no_budget_outlives_the_transport_that_carries_it() {
        let state = test_state();
        let model = ModelId("m".into());
        let transport = Duration::from_secs(crate::network::behaviour::RR_REQUEST_TIMEOUT_SECS);

        // Cold + unmeasured is the largest budget this can produce.
        let cold = SegmentBudget::for_forward(
            &state,
            &NodeId([9u8; 32]),
            &model,
            WorkKind::Prefill,
            80,
            4_000_000,
            ActivationUnits::HiddenStates,
        );
        assert!(
            cold.duration() <= transport,
            "budget {:?} exceeds the transport ceiling {:?} — the forward would be \
             reaped before this deadline and the wait could never succeed",
            cold.duration(),
            transport
        );

        // And a measured-but-very-slow peer likewise.
        state.record_peer_segment_latency(
            &NodeId([10u8; 32]),
            &model,
            WorkKind::Prefill,
            600_000,
            8,
            213_268,
        );
        let slow = SegmentBudget::for_forward(
            &state,
            &NodeId([10u8; 32]),
            &model,
            WorkKind::Prefill,
            8,
            213_268,
            ActivationUnits::HiddenStates,
        );
        assert!(
            slow.duration() <= transport,
            "measured-slow budget must also be capped"
        );
    }

    /// Never return a deadline below the floor, however fast the peer looks.
    #[test]
    fn the_floor_still_applies_to_a_very_fast_peer() {
        let state = test_state();
        let node = NodeId([6u8; 32]);
        let model = ModelId("m".into());
        state.record_peer_segment_latency(&node, &model, WorkKind::Decode, 5, 8, 16_384);

        let b = SegmentBudget::for_forward(
            &state,
            &node,
            &model,
            WorkKind::Decode,
            8,
            16_384,
            ActivationUnits::HiddenStates,
        );
        assert!(b.duration() >= Duration::from_secs(SEGMENT_TIMEOUT_MIN_SECS));
    }

    // ── A forward the peer could not open (FUTURE_WORK #92) ──────────────────

    use crate::crypto::session::{KeyAdoption, SessionManager, SESSION_CONFIRM_MARKER};
    use crate::types::{ForwardRefusal, LayerForward, NetworkCommand, TensorFormat};

    fn a_forward(request_id: uuid::Uuid) -> LayerForward {
        LayerForward {
            request_id,
            sequence_num: 7,
            index_pos: 41,
            activations: vec![4, 2, 4, 2],
            format: TensorFormat::FP32,
            model_id: ModelId("m".into()),
            layer_range: (0, 8),
            tp_meta: None,
            vision_embeddings: None,
            chain: Vec::new(),
            sender_peer_bytes: None,
            requester_node_id: None,
            pre_embedded: false,
            generated_ids: Vec::new(),
            adapter_id: None,
            draft_tokens: Vec::new(),
            spec_logits_requested: false,
            truncate_kv_to: None,
            chunk_meta: None,
            sampling: None,
        }
    }

    fn refused(request_id: uuid::Uuid) -> LayerResult {
        as_if_from_the_peer(
            LayerResult::error(request_id, "Could not decrypt forward")
                .with_refusal(ForwardRefusal::Undecryptable),
        )
    }

    /// This node and `node` with a session each way, as a coordinator and the
    /// peer it forwards to.
    fn linked(state: &SharedState, node: &NodeId) -> (SessionManager, NodeId) {
        let peer = SessionManager::from_ed25519_key(&[5u8; 32]);
        let us = state.identity.node_id().clone();
        assert!(state
            .session_manager
            .establish_session(node, *peer.local_public_key()));
        assert!(peer.establish_session(&us, *state.session_manager.local_public_key()));
        (peer, us)
    }

    /// The repair a refusal arms, carried out as it is live: the refusing peer
    /// initiates, this node answers, and the peer's confirmation opens here.
    fn repair(state: &SharedState, peer: &SessionManager, node: &NodeId, us: &NodeId) {
        let theirs = peer.initiate_ephemeral_exchange(us);
        let ours = state.session_manager.accept_ephemeral_exchange(
            node,
            &theirs,
            KeyAdoption::OnConfirmation,
        );
        let confirm = peer.complete_ephemeral_session(us, &ours).unwrap();
        state
            .session_manager
            .open(node, &confirm, SESSION_CONFIRM_MARKER)
            .unwrap();
    }

    /// The request #92 lost: the peer could not open a forward, armed the
    /// repair, and the link was fine a round trip later — but the coordinator
    /// failed over, found no standby, and ended the request. Now the same
    /// forward goes to the same peer once the link is re-keyed, and not before.
    #[tokio::test]
    async fn a_forward_the_peer_could_not_open_goes_to_it_again_once_the_link_is_rekeyed() {
        let state = test_state();
        let node = NodeId([7u8; 32]);
        let (peer, us) = linked(&state, &node);
        let request_id = uuid::Uuid::new_v4();
        let b = budget(&state, &node, &ModelId("m".into()), WorkKind::Decode);
        let (net_tx, mut net_rx) = tokio::sync::mpsc::channel(4);
        let sent = a_forward(request_id);
        let rebuild = || sent.clone();
        let (tx, rx) = tokio::sync::oneshot::channel();
        tx.send(refused(request_id)).unwrap();

        let peer_side = async {
            tokio::time::sleep(Duration::from_millis(60)).await;
            assert!(
                net_rx.try_recv().is_err(),
                "nothing may be resent before the link is re-keyed — it would go out \
                 under the key the peer just refused"
            );
            repair(&state, &peer, &node, &us);
            let Ok(Some(NetworkCommand::SendTensor {
                target_peer_bytes,
                forward,
            })) = tokio::time::timeout(Duration::from_secs(20), net_rx.recv()).await
            else {
                panic!("the refused forward must be sent again");
            };
            assert_eq!(target_peer_bytes, vec![9, 9], "to the same peer");
            assert_eq!(
                (
                    forward.index_pos,
                    forward.sequence_num,
                    &forward.activations
                ),
                (41, 7, &vec![4, 2, 4, 2]),
                "the SAME step — the peer never ran it"
            );
            let mut answer = LayerResult::error(request_id, "");
            answer.finish_reason = None;
            answer.activations = vec![1, 2, 3];
            assert!(state.resolve_pending_layer_result(Some(&node), as_if_from_the_peer(answer)));
        };
        let (got, ()) = tokio::join!(
            PipelineExecutor::wait_for_result(
                &state,
                rx,
                request_id,
                0,
                &node,
                1,
                8,
                b,
                None,
                ResendOnRefusal::SameForward {
                    network_tx: &net_tx,
                    target_peer_bytes: &[9, 9],
                    rebuild: &rebuild,
                },
            ),
            peer_side
        );
        let got = got.unwrap();
        assert_eq!(got.refusal, None);
        assert_eq!(
            got.activations,
            vec![1, 2, 3],
            "the resend's answer is the result"
        );
        assert!(net_rx.try_recv().is_err(), "sent again exactly once");
    }

    /// Where the link does not repair, nothing is resent and the refusal goes to
    /// the caller's ordinary handling — exactly where it went before.
    #[tokio::test(start_paused = true)]
    async fn a_link_that_is_never_rekeyed_gets_no_resend_and_the_refusal_stands() {
        let state = test_state();
        let node = NodeId([8u8; 32]);
        let _link = linked(&state, &node);
        let request_id = uuid::Uuid::new_v4();
        let b = budget(&state, &node, &ModelId("m".into()), WorkKind::Decode);
        let (net_tx, mut net_rx) = tokio::sync::mpsc::channel(4);
        let sent = a_forward(request_id);
        let rebuild = || sent.clone();
        let (tx, rx) = tokio::sync::oneshot::channel();
        tx.send(refused(request_id)).unwrap();

        let got = PipelineExecutor::wait_for_result(
            &state,
            rx,
            request_id,
            0,
            &node,
            1,
            8,
            b,
            None,
            ResendOnRefusal::SameForward {
                network_tx: &net_tx,
                target_peer_bytes: &[9, 9],
                rebuild: &rebuild,
            },
        )
        .await
        .unwrap();
        assert_eq!(got.refusal, Some(ForwardRefusal::Undecryptable));
        assert!(net_rx.try_recv().is_err());
        assert!(
            !state.pending_layer_results.contains_key(&request_id),
            "no waiter is left behind for a forward that was never sent"
        );
    }

    /// The resend is for the TYPED refusal on a path that allows it, once. The
    /// link IS repaired during each of these waits, so the only thing standing
    /// between them and a resend is the rule under test — a wrong resend would
    /// find the new key and send, rather than time out and look correct.
    #[tokio::test]
    async fn only_a_typed_refusal_on_a_path_that_allows_it_is_resent_and_only_once() {
        let state = test_state();
        let node = NodeId([9u8; 32]);
        let (peer, us) = linked(&state, &node);
        let b = budget(&state, &node, &ModelId("m".into()), WorkKind::Decode);
        let (net_tx, mut net_rx) = tokio::sync::mpsc::channel(4);

        // (1) The same words with no type: a peer on an older build, or any
        //     other error. Handled as before.
        // (2) The type, on a path that has said it must not resend.
        for (label, answer, may) in [
            (
                "untyped",
                as_if_from_the_peer(LayerResult::error(
                    uuid::Uuid::nil(),
                    "Could not decrypt forward",
                )),
                true,
            ),
            ("never", refused(uuid::Uuid::nil()), false),
        ] {
            let request_id = uuid::Uuid::new_v4();
            let mut answer = answer;
            answer.request_id = request_id;
            let sent = a_forward(request_id);
            let rebuild = || sent.clone();
            let resend = if may {
                ResendOnRefusal::SameForward {
                    network_tx: &net_tx,
                    target_peer_bytes: &[9, 9],
                    rebuild: &rebuild,
                }
            } else {
                ResendOnRefusal::Never("test: this path must not resend")
            };
            let (tx, rx) = tokio::sync::oneshot::channel();
            tx.send(answer).unwrap();
            let (got, ()) = tokio::join!(
                PipelineExecutor::wait_for_result(
                    &state, rx, request_id, 0, &node, 1, 8, b, None, resend
                ),
                async {
                    tokio::time::sleep(Duration::from_millis(30)).await;
                    repair(&state, &peer, &node, &us);
                }
            );
            assert!(
                matches!(
                    got.unwrap().finish_reason,
                    Some(crate::types::NetworkFinishReason::Error(_))
                ),
                "{label}: the error is handed back as it came"
            );
            assert!(net_rx.try_recv().is_err(), "{label}: nothing is resent");
        }

        // (3) Refused, resent, refused AGAIN: the second refusal stands — with
        //     the link repaired a second time, so a second resend would happen
        //     if the rule allowed one.
        let request_id = uuid::Uuid::new_v4();
        let sent = a_forward(request_id);
        let rebuild = || sent.clone();
        let (tx, rx) = tokio::sync::oneshot::channel();
        tx.send(refused(request_id)).unwrap();
        let (got, ()) = tokio::join!(
            PipelineExecutor::wait_for_result(
                &state,
                rx,
                request_id,
                0,
                &node,
                1,
                8,
                b,
                None,
                ResendOnRefusal::SameForward {
                    network_tx: &net_tx,
                    target_peer_bytes: &[9, 9],
                    rebuild: &rebuild,
                },
            ),
            async {
                tokio::time::sleep(Duration::from_millis(30)).await;
                repair(&state, &peer, &node, &us);
                assert!(matches!(
                    tokio::time::timeout(Duration::from_secs(20), net_rx.recv()).await,
                    Ok(Some(NetworkCommand::SendTensor { .. }))
                ));
                assert!(state.resolve_pending_layer_result(Some(&node), refused(request_id)));
                tokio::time::sleep(Duration::from_millis(30)).await;
                repair(&state, &peer, &node, &us);
            }
        );
        assert_eq!(got.unwrap().refusal, Some(ForwardRefusal::Undecryptable));
        assert!(
            net_rx.try_recv().is_err(),
            "one resend per forward, never two"
        );
    }
}

#[cfg(test)]
mod segment_timeout_classification_tests {
    use super::*;

    /// A peer that took the forward and went silent is not this node breaking.
    /// Pins all three user-visible consequences of the variant at once, because
    /// each of them was wrong while the error was a `PipelineError`.
    #[test]
    fn a_silent_peer_is_reported_as_a_peer_failure_not_as_our_bug() {
        let err = segment_timeout_error(52, 26);
        let (status, _msg, kind) = crate::error::classify_error(&err);
        assert_eq!(
            status,
            axum::http::StatusCode::SERVICE_UNAVAILABLE,
            "a peer going quiet must not be reported as a 500 from this node"
        );
        assert_eq!(kind, "server_error");
        assert_eq!(
            crate::error::failure_log_level(&err),
            crate::error::FailureLevel::Warn,
            "ERROR means this node is broken; a quiet peer is not that"
        );
    }

    /// The advice has to be advice the caller can act on. The generic
    /// `PipelineError` hint was picked by substring-matching prose, which is
    /// how three failures came to be told to retry something retrying could not
    /// fix (gotcha #295).
    #[test]
    fn the_hint_points_at_a_different_holder() {
        let hint = crate::error::error_hint(&segment_timeout_error(52, 26))
            .expect("a silent peer must carry a hint");
        assert!(
            !hint.is_empty(),
            "a caller with no next step is a caller who retries blindly"
        );
    }

    /// The deadline and the layer count stay in the message: an operator
    /// reading the log needs to know WHICH wait expired and how long it was.
    #[test]
    fn the_message_still_names_the_deadline_and_the_span() {
        let msg = segment_timeout_error(52, 26).to_string();
        assert!(msg.contains("52s"), "{msg}");
        assert!(msg.contains("26 layers"), "{msg}");
    }
}
