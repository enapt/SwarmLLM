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
        let measured = match units {
            ActivationUnits::HiddenStates => {
                state.predict_segment_ms(node_id, kind, num_layers, activation_bytes)
            }
            // The coefficient is not in these units; a prediction from it would
            // be meaningless rather than merely imprecise.
            ActivationUnits::PromptBytes => None,
        };

        let (base, basis) = match measured {
            Some(ms) => (
                Duration::from_millis((ms * SEGMENT_TIMEOUT_SAFETY_FACTOR) as u64),
                "measured",
            ),
            None => (
                PipelineExecutor::compute_segment_timeout(num_layers, activation_bytes),
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
        }
    }

    pub(super) fn duration(&self) -> Duration {
        self.duration
    }

    pub(super) fn basis(&self) -> &'static str {
        self.basis
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
    /// Prefill (large activation = many input tokens) is much slower than decode
    /// (single token). Budget per-layer time with a floor and ceiling.
    pub(super) fn compute_segment_timeout(num_layers: u32, activation_bytes: usize) -> Duration {
        let is_prefill = activation_bytes > PREFILL_ACTIVATION_THRESHOLD_BYTES;
        let per_layer_secs: u64 = if is_prefill {
            PREFILL_SECS_PER_LAYER
        } else {
            DECODE_SECS_PER_LAYER
        };
        let base = (num_layers as u64) * per_layer_secs;
        let timeout = base.clamp(SEGMENT_TIMEOUT_MIN_SECS, SEGMENT_TIMEOUT_MAX_SECS);
        Duration::from_secs(timeout)
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
            is_prefill = activation_bytes > super::PREFILL_ACTIVATION_THRESHOLD_BYTES,
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
                Err(SwarmError::PipelineError(format!(
                    "Timed out waiting for segment result ({}s, {} layers)",
                    timeout.as_secs(),
                    num_layers
                )))
            }
        }
    }
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
