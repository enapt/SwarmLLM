//! Core types exposed by the inference router: commands, queued requests,
//! streaming events, and the output struct returned to API callers.

use tokio::sync::{mpsc, oneshot};

use crate::error::SwarmError;
use crate::inference::executor::GenerationResult;
use crate::types::{InferenceRequest, SwarmMessage};

/// Result channel for returning inference output to API callers.
pub type InferenceResultTx = oneshot::Sender<Result<InferenceOutput, SwarmError>>;

/// Sender for incremental streaming tokens from distributed inference.
///
/// A newtype around the channel rather than a bare `mpsc::Sender` so that
/// **time-to-first-token is stamped at the one place every path funnels
/// through**. Tokens are emitted from seven sites across `local_exec`,
/// `process_pool`, `dsd`, `speculative`, `ngram_only_spec` and
/// `pipeline/mod`; stamping TTFT at each is exactly the "one invariant, N
/// paths" defect in `.claude/rules/architecture.md` — a new emit site would
/// silently report no TTFT. Here a new site inherits it for free.
///
/// The API surface mirrors `mpsc::Sender` (`send`, `try_send`, `is_closed`,
/// `closed`, `Clone`) so call sites read unchanged.
#[derive(Clone)]
pub struct StreamingTokenTx {
    inner: mpsc::Sender<StreamingTokenEvent>,
    /// `None` on paths with no trace (tests, internal fan-out). The wrapper
    /// still forwards; it just records nothing.
    trace: Option<std::sync::Arc<crate::inference::trace::RequestTrace>>,
}

impl StreamingTokenTx {
    /// Create the channel. Preferred over building an `mpsc::channel` by hand
    /// so the sender is always the traced wrapper.
    pub fn channel(capacity: usize) -> (Self, mpsc::Receiver<StreamingTokenEvent>) {
        let (inner, rx) = mpsc::channel(capacity);
        (Self { inner, trace: None }, rx)
    }

    /// Attach the trace whose TTFT this channel should stamp.
    pub fn with_trace(
        mut self,
        trace: std::sync::Arc<crate::inference::trace::RequestTrace>,
    ) -> Self {
        self.trace = Some(trace);
        self
    }

    /// Stamp on the first event that carries actual text. The terminal
    /// `finish_reason` event has an empty `text` and must not count as the
    /// first token — otherwise a zero-token response would report a TTFT.
    #[inline]
    fn stamp(&self, event: &StreamingTokenEvent) {
        if !event.text.is_empty() {
            if let Some(ref t) = self.trace {
                t.mark_first_token();
            }
        }
    }

    pub async fn send(
        &self,
        event: StreamingTokenEvent,
    ) -> Result<(), mpsc::error::SendError<StreamingTokenEvent>> {
        self.stamp(&event);
        self.inner.send(event).await
    }

    pub fn try_send(
        &self,
        event: StreamingTokenEvent,
    ) -> Result<(), mpsc::error::TrySendError<StreamingTokenEvent>> {
        self.stamp(&event);
        self.inner.try_send(event)
    }

    pub fn is_closed(&self) -> bool {
        self.inner.is_closed()
    }

    /// Resolves when the RECEIVER is dropped — consumer-liveness watch.
    pub async fn closed(&self) {
        self.inner.closed().await
    }
}

/// A queued inference request with its result channel and priority ordering.
pub(super) struct QueuedRequest {
    pub(super) request: InferenceRequest,
    pub(super) result_tx: InferenceResultTx,
    /// If set, tokens are sent incrementally for SSE streaming.
    pub(super) token_tx: Option<StreamingTokenTx>,
    /// Routing + performance record. Created at enqueue so queue wait is
    /// measured from admission rather than from dispatch.
    pub(super) trace: std::sync::Arc<crate::inference::trace::RequestTrace>,
}

impl Eq for QueuedRequest {}
impl PartialEq for QueuedRequest {
    fn eq(&self, other: &Self) -> bool {
        self.request.priority == other.request.priority
            && self.request.created_at == other.request.created_at
    }
}

impl PartialOrd for QueuedRequest {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for QueuedRequest {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Higher priority first, then earlier created_at (FIFO within same tier)
        self.request
            .priority
            .cmp(&other.request.priority)
            .then_with(|| other.request.created_at.cmp(&self.request.created_at))
    }
}

/// Output from a completed inference request.
#[derive(Debug, Clone)]
pub struct InferenceOutput {
    pub request_id: uuid::Uuid,
    pub content: String,
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub finish_reason: String,
    /// The session ID for multi-turn KV-cache reuse. Echoed back from the
    /// request or auto-generated if the router created one.
    pub session_id: Option<String>,
    /// Per-token log probabilities (populated when logprobs=true in request).
    pub token_logprobs: Vec<TokenLogProbEntry>,
    /// The user-provided stop sequence that triggered termination, if any.
    /// Populated only when `finish_reason == "stop"` AND a sequence from
    /// `SamplingParams.stop` matched the accumulated text. Anthropic's
    /// `/v1/messages` response contract requires this in the
    /// `stop_sequence` field; OpenAI doesn't expose it but it's harmless
    /// extra metadata for compatible clients.
    pub matched_stop_sequence: Option<String>,
}

impl InferenceOutput {
    /// Build an `InferenceOutput` from a local executor's `GenerationResult`.
    /// Used by the six local-execution sites (single-node + the streaming and
    /// non-streaming variants of local-exec and the distributed-exec early-
    /// return path) that all pin `token_logprobs: vec![]`. When per-token
    /// logprob plumbing reaches these paths, swap the `vec![]` here for the
    /// real value.
    pub(crate) fn from_gen_result(
        request_id: uuid::Uuid,
        session_id: Option<String>,
        content: String,
        finish_reason: String,
        gen_result: &GenerationResult,
    ) -> Self {
        Self {
            request_id,
            content,
            prompt_tokens: gen_result.prompt_tokens,
            completion_tokens: gen_result.completion_tokens,
            finish_reason,
            session_id,
            token_logprobs: vec![],
            matched_stop_sequence: gen_result.matched_stop_sequence.clone(),
        }
    }
}

/// A single token's log probability info for the logprobs response field.
/// Canonical definition lives in `swarmllm-types` so it can also be carried
/// in `LayerResult` over the distributed-pipeline wire.
pub use swarmllm_types::TokenLogProbEntry;

/// A single token event sent during streaming distributed inference.
#[derive(Debug, Clone)]
pub struct StreamingTokenEvent {
    pub text: String,
    pub finish_reason: Option<String>,
    /// Set on the final event (`finish_reason: Some("stop")`) when a
    /// user-provided stop sequence matched. The Anthropic SSE handler
    /// reads this to populate `message_delta.delta.stop_sequence`.
    /// Empty/intermediate token events leave this as `None`.
    pub matched_stop_sequence: Option<String>,
}

/// Command sent to the InferenceRouter from the API layer or network.
pub enum RouterCommand {
    /// Submit a new inference request with a channel for the result.
    Submit {
        request: InferenceRequest,
        result_tx: InferenceResultTx,
    },
    /// Submit a streaming inference request. Tokens are sent incrementally
    /// on `token_tx`. The final `InferenceOutput` is still sent on `result_tx`
    /// for stats/credit accounting.
    StreamSubmit {
        request: InferenceRequest,
        result_tx: InferenceResultTx,
        token_tx: StreamingTokenTx,
    },
    /// A network message relevant to inference (LayerForward, LayerResult, etc.)
    NetworkMessage(SwarmMessage),
    /// Update multi-turn KV-cache token count after inference completes.
    UpdateCacheTokens {
        session_id: String,
        total_tokens: u32,
        prompt: String,
    },
}
