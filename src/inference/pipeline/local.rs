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
                let (gen_result, spec_state) = executor.generate_speculative(
                    &mut draft,
                    &prompt,
                    &self.request.sampling_params,
                    gamma,
                    |token| {
                        content.push_str(token);
                        true
                    },
                )?;
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
        let (content, gen_result) = executor.generate(&prompt, &self.request.sampling_params)?;

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
        // `skip_serializing_if = "Vec::is_empty"`). The distributed path
        // already gates this; the local path was unconditional.
        let needs_generated_ids = self.request.sampling_params.frequency_penalty != 0.0
            || self.request.sampling_params.presence_penalty != 0.0;
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
            adapter_id: None,
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
        let layer_result = self
            .shared_state
            .model_process_pool
            .forward(layer_forward)
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
    pub(super) async fn wait_for_result(
        rx: tokio::sync::oneshot::Receiver<LayerResult>,
        request_id: uuid::Uuid,
        segment_idx: usize,
        node_id: &crate::types::NodeId,
        num_layers: u32,
        activation_bytes: usize,
        budget: SegmentBudget,
    ) -> Result<LayerResult, SwarmError> {
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
        match tokio::time::timeout(timeout, rx).await {
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
                Err(SwarmError::PipelineError("Response channel dropped".into()))
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
                Err(segment_timeout_error(timeout.as_secs(), num_layers))
            }
        }
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
