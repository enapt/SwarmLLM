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
    /// Route and timing, attached at the router's completion arm so the API
    /// layer can render response headers.
    ///
    /// `None` on paths that never reached the router — the cloud proxy, and the
    /// rejections that fail before dispatch. Those have no swarm route to
    /// report, so the headers are omitted rather than guessed at.
    pub trace: Option<crate::inference::trace::TraceSnapshot>,
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
            trace: None,
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

/// Hand a finished request's result to whoever is waiting for it, and — when
/// nobody is — say truthfully which of the two reasons applies.
///
/// A `oneshot` send fails whenever the receiver has been dropped, and that
/// covers two situations a log reader needs to tell apart:
///
/// - **The client went away.** For every non-streaming caller
///   (`api::submit_to_router`, and so ordinary chat, the Anthropic surface,
///   MCP, the Responses background task and the fan-out tools) the future
///   holding `result_rx` IS the client's connection, so a drop there really is
///   a disconnect and worth a warning.
/// - **Nobody needed it.** A streaming caller that has already delivered every
///   token and its `finish_reason` reads `result_rx` only to answer
///   `stream_options.include_usage`. With usage off — the default, and what
///   most OpenAI-compatible clients send — the SSE bridge simply returns, and
///   the receiver is gone by design. Every side effect the result carries
///   (`finalize_request`, `release_request_state`, the active-count decrement
///   and the queue wake) has already run at all three call sites, so nothing
///   is lost.
///
/// The old message asserted the first cause for both, and a successful stream
/// was therefore indistinguishable in the log from a client dropping the
/// connection mid-answer. Measured on the live node, 2 of 28 router-path
/// streams tripped it — it is a race between the SSE bridge returning and this
/// send, so it is intermittent, which is worse than constant: a reader cannot
/// even learn to discount it.
///
/// `InferenceRequest::cancel` is the discriminator and costs nothing — it is
/// already on the request at every send site, and the SSE loop sets it ONLY on
/// a genuine disconnect (`sse_tx.closed()`). So a cancelled request keeps the
/// warning; an uncancelled one is the ordinary case and logs at debug.
pub(super) fn deliver_result(
    request: &InferenceRequest,
    result_tx: InferenceResultTx,
    output: Result<InferenceOutput, SwarmError>,
    path: &'static str,
) {
    if result_tx.send(output).is_ok() {
        return;
    }
    if request.is_cancelled() {
        tracing::warn!(
            request_id = %request.id,
            path,
            "DIAG: result_tx receiver dropped after the client disconnected — \
             the answer was computed and had nowhere to go"
        );
    } else {
        tracing::debug!(
            request_id = %request.id,
            path,
            "DIAG: result_tx receiver dropped without a cancellation — the caller \
             had everything it asked for (a stream that did not request usage \
             stats does not read this channel)"
        );
    }
}

#[cfg(test)]
mod deliver_result_tests {
    use super::*;
    use crate::types::ModelId;
    use std::sync::atomic::AtomicBool;
    use std::sync::{Arc, Mutex};

    /// Records the LEVEL of every event emitted, which is the whole assertion:
    /// the message is chosen from the cancel flag, and a reader distinguishes
    /// the two cases by severity.
    #[derive(Clone)]
    struct Levels(Arc<Mutex<Vec<tracing::Level>>>);

    impl tracing::subscriber::Subscriber for Levels {
        fn enabled(&self, _: &tracing::Metadata<'_>) -> bool {
            true
        }
        fn new_span(&self, _: &tracing::span::Attributes<'_>) -> tracing::Id {
            tracing::Id::from_u64(1)
        }
        fn record(&self, _: &tracing::Id, _: &tracing::span::Record<'_>) {}
        fn record_follows_from(&self, _: &tracing::Id, _: &tracing::Id) {}
        fn event(&self, e: &tracing::Event<'_>) {
            self.0.lock().unwrap().push(*e.metadata().level());
        }
        fn enter(&self, _: &tracing::Id) {}
        fn exit(&self, _: &tracing::Id) {}
    }

    fn request(cancelled: bool) -> InferenceRequest {
        let mut r = InferenceRequest::local(
            ModelId("m".into()),
            Vec::new(),
            Default::default(),
            false,
            None,
            None,
        );
        if cancelled {
            r.cancel = Some(Arc::new(AtomicBool::new(true)));
        }
        r
    }

    fn levels_when(cancelled: bool, take_the_result: bool) -> Vec<tracing::Level> {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let sub = Levels(seen.clone());
        let req = request(cancelled);
        let (tx, rx) = oneshot::channel();
        // The receiver is held for the whole send when the caller is still
        // waiting, and gone before it when nobody is — which is the only
        // difference the helper can see.
        let held = if take_the_result {
            Some(rx)
        } else {
            drop(rx);
            None
        };
        tracing::subscriber::with_default(sub, || {
            deliver_result(&req, tx, Err(SwarmError::NoModelLoaded), "test");
        });
        drop(held);
        let out = seen.lock().unwrap().clone();
        out
    }

    /// A stream that delivered every token and simply did not ask for usage
    /// stats drops the receiver by design. Warning about it says the client
    /// disconnected, which it did not — and a reader watching a healthy node
    /// sees a failure on every ordinary request.
    #[test]
    fn a_result_nobody_asked_for_is_not_reported_as_a_disconnect() {
        let levels = levels_when(false, false);
        assert_eq!(levels, vec![tracing::Level::DEBUG], "{levels:?}");
    }

    /// The control: the case the warning was written for still warns. Without
    /// the cancel flag being consulted, this and the test above are the same
    /// line at the same level, which is exactly the defect.
    #[test]
    fn a_result_that_had_nowhere_to_go_because_the_client_left_still_warns() {
        let levels = levels_when(true, false);
        assert_eq!(levels, vec![tracing::Level::WARN], "{levels:?}");
    }

    /// The common path stays silent — nothing is logged when the result is
    /// delivered, whatever the cancel flag says.
    #[test]
    fn a_delivered_result_logs_nothing() {
        assert!(levels_when(false, true).is_empty());
        assert!(levels_when(true, true).is_empty());
    }
}
