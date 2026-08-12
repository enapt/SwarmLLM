//! Provider-neutral streaming primitives shared by SSE endpoints.
//!
//! OpenAI and Anthropic formats are different enough that the final SSE
//! serialization lives in each provider module, but the intermediate event
//! type used between the inference loop and the SSE encoder is the same
//! shape for OpenAI-style streams and is kept here for reuse.

use tokio::sync::mpsc;

// ---- Raw SSE framing for non-axum-Sse streams ----
//
// Some handlers (Claude subscription, Claude Code sessions) produce a pre-
// formatted byte stream wrapped in `axum::body::Body::from_stream` rather
// than routing through `axum::response::Sse`. These helpers keep the `data:`
// and `event:` framing uniform across those handlers.

/// Format a value as an SSE `data:` frame (for byte streams).
/// Accepts any `serde::Serialize` — typed response structs or raw JSON values.
/// On serialization failure (e.g. NaN in a logprob) emits a structured error
/// event rather than an empty `data:` line — empty data is a valid SSE
/// no-op event and would silently hang clients waiting for a specific
/// payload (e.g. `[DONE]` or `message_stop`).
pub fn data_frame<S: serde::Serialize>(value: &S) -> bytes::Bytes {
    match serde_json::to_string(value) {
        Ok(json) => bytes::Bytes::from(format!("data: {json}\n\n")),
        Err(e) => {
            tracing::error!(error = %e, "SSE data_frame: serialization failed");
            bytes::Bytes::from_static(b"data: {\"error\":\"serialization_failed\"}\n\n")
        }
    }
}

/// Format a named SSE event frame (`event: ...\ndata: ...`) for byte streams.
pub fn event_frame<S: serde::Serialize>(event_type: &str, value: &S) -> bytes::Bytes {
    match serde_json::to_string(value) {
        Ok(json) => bytes::Bytes::from(format!("event: {event_type}\ndata: {json}\n\n")),
        Err(e) => {
            tracing::error!(error = %e, event_type, "SSE event_frame: serialization failed");
            bytes::Bytes::from(format!(
                "event: {event_type}\ndata: {{\"error\":\"serialization_failed\"}}\n\n"
            ))
        }
    }
}

/// Terminal `data: [DONE]` frame used by OpenAI-compatible streams.
pub fn done_frame() -> bytes::Bytes {
    bytes::Bytes::from_static(b"data: [DONE]\n\n")
}

/// Intermediate stream event emitted by the inference loop and consumed by
/// the OpenAI-format SSE encoder.
pub enum StreamEvent {
    Delta {
        content: Option<String>,
        role: Option<String>,
        finish_reason: Option<String>,
    },
    /// A failure, in the same terms the non-streaming sibling would report it.
    ///
    /// `error_type` is a required field rather than an encoder-side default
    /// precisely so a new call site has to say what kind of failure this is.
    /// The encoder used to stamp every one of them `server_error`, which told
    /// the caller that this server had broken when in fact their prompt was too
    /// long. Fill it from `crate::error::classify_error`, never by hand.
    Error {
        message: String,
        error_type: &'static str,
    },
    /// A complete set of tool calls recovered from a local model's output.
    ///
    /// Additive variant rather than a field on `Delta`, which has a dozen
    /// construction sites that have nothing to do with tools.
    ///
    /// Emitted as ONE event carrying whole calls rather than the fragment
    /// sequence a cloud provider streams. A local model's tool call can only be
    /// recognised once its text is complete — mid-stream we cannot tell a tool
    /// call from prose that happens to start with a brace — so fragments would
    /// mean emitting text we might have to retract. Clients that concatenate
    /// streamed `tool_calls` deltas handle a single complete delta correctly,
    /// since the index/id/name/arguments fields are all present at once.
    ToolCalls {
        calls: Vec<crate::api::openai::StreamToolCall>,
    },
    /// OpenAI 2024+ spec: when the request includes
    /// `stream_options: {"include_usage": true}`, an extra terminal chunk
    /// is emitted right before `[DONE]` with `choices: []` and the usage
    /// object populated. This event carries the token counts to the
    /// encoder; emit it ONLY when the request opted in.
    Usage {
        prompt_tokens: u32,
        completion_tokens: u32,
    },
    Done,
}

/// Send the initial `role: "assistant"` delta that opens the streaming response.
/// Returns `false` if the client has already disconnected.
pub async fn send_role_preamble(tx: &mpsc::Sender<StreamEvent>) -> bool {
    tx.send(StreamEvent::Delta {
        content: None,
        role: Some("assistant".into()),
        finish_reason: None,
    })
    .await
    .is_ok()
}

/// One-line human-readable progress note for an SSE comment frame.
///
/// Deliberately plain text, not JSON: it is a comment, so nothing parses it —
/// its only reader is a person watching a stream that would otherwise look
/// dead. Says "still reading" rather than naming an internal phase, and omits
/// the ETA entirely rather than inventing one before the rate is known.
pub fn format_progress_comment(s: &crate::inference::trace::ProgressSnapshot) -> String {
    let what = match s.phase {
        "loading_model" => "loading model".to_string(),
        "prefill" => match s.percent {
            Some(pct) => format!("reading prompt {pct}% ({}/{} tokens)", s.done, s.total),
            None => "reading prompt".to_string(),
        },
        other => other.to_string(),
    };
    match s.eta_ms {
        Some(ms) if ms >= 1000 => format!("{what}, about {}s left", ms / 1000),
        _ => what,
    }
}
