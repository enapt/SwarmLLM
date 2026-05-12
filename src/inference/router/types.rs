//! Core types exposed by the inference router: commands, queued requests,
//! streaming events, and the output struct returned to API callers.

use tokio::sync::{mpsc, oneshot};

use crate::error::SwarmError;
use crate::types::{InferenceRequest, SwarmMessage};

/// Result channel for returning inference output to API callers.
pub type InferenceResultTx = oneshot::Sender<Result<InferenceOutput, SwarmError>>;

/// Sender for incremental streaming tokens from distributed inference.
pub type StreamingTokenTx = mpsc::Sender<StreamingTokenEvent>;

/// A queued inference request with its result channel and priority ordering.
pub(super) struct QueuedRequest {
    pub(super) request: InferenceRequest,
    pub(super) result_tx: InferenceResultTx,
    /// If set, tokens are sent incrementally for SSE streaming.
    pub(super) token_tx: Option<StreamingTokenTx>,
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

/// A single token's log probability info for the logprobs response field.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TokenLogProbEntry {
    /// The token text.
    pub token: String,
    /// Log probability of this token.
    pub logprob: f32,
    /// Top-N alternative tokens with their logprobs.
    pub top_logprobs: Vec<(String, f32)>,
}

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
