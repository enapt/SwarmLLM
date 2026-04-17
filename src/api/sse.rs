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

/// Format a JSON value as an SSE `data:` frame (for byte streams).
pub fn data_frame(value: &serde_json::Value) -> bytes::Bytes {
    bytes::Bytes::from(format!("data: {value}\n\n"))
}

/// Format a named SSE event frame (`event: ...\ndata: ...`) for byte streams.
pub fn event_frame(event_type: &str, value: &serde_json::Value) -> bytes::Bytes {
    bytes::Bytes::from(format!("event: {event_type}\ndata: {value}\n\n"))
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
    Error {
        message: String,
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
