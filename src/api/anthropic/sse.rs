//! Anthropic SSE event types + serialization + stream-response builder.

use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::IntoResponse;
use std::convert::Infallible;
use tokio_stream::StreamExt;

use crate::api::SSE_KEEPALIVE_INTERVAL_SECS;

/// Internal SSE event types for Anthropic streaming.
pub(super) enum AnthropicSseEvent {
    MessageStart {
        id: String,
        model: String,
    },
    ContentBlockStart {
        index: u32,
    },
    /// Open a `tool_use` content block. Used by the OpenAI→Anthropic
    /// streaming translator to surface upstream `tool_calls` chunks.
    ContentBlockStartToolUse {
        index: u32,
        id: String,
        name: String,
    },
    ContentBlockDelta {
        index: u32,
        text: String,
    },
    /// `input_json_delta` for an open `tool_use` block. Anthropic streams
    /// the tool-call arguments as a sequence of partial JSON fragments
    /// the client concatenates and parses at content_block_stop.
    ContentBlockInputJsonDelta {
        index: u32,
        partial_json: String,
    },
    ContentBlockStop {
        index: u32,
    },
    MessageDelta {
        stop_reason: String,
        /// Anthropic spec: when `stop_reason == "stop_sequence"`, the matched
        /// custom stop string is reported here so clients can route on which
        /// sequence fired. `None` for `end_turn` / `max_tokens` reasons.
        stop_sequence: Option<String>,
        output_tokens: u32,
    },
    MessageStop,
}

/// Serialize an Anthropic SSE event to (event_type, data_json).
pub(super) fn serialize_anthropic_event(event: &AnthropicSseEvent) -> (&'static str, String) {
    match event {
        AnthropicSseEvent::MessageStart { id, model } => (
            "message_start",
            serde_json::json!({
                "type": "message_start",
                "message": {
                    "id": id,
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": model,
                    "stop_reason": null,
                    "stop_sequence": null,
                    "usage": { "input_tokens": 0, "output_tokens": 0 }
                }
            })
            .to_string(),
        ),
        AnthropicSseEvent::ContentBlockStart { index } => (
            "content_block_start",
            serde_json::json!({
                "type": "content_block_start",
                "index": index,
                "content_block": { "type": "text", "text": "" }
            })
            .to_string(),
        ),
        AnthropicSseEvent::ContentBlockStartToolUse { index, id, name } => (
            "content_block_start",
            serde_json::json!({
                "type": "content_block_start",
                "index": index,
                "content_block": {
                    "type": "tool_use",
                    "id": id,
                    "name": name,
                    "input": {}
                }
            })
            .to_string(),
        ),
        AnthropicSseEvent::ContentBlockDelta { index, text } => (
            "content_block_delta",
            serde_json::json!({
                "type": "content_block_delta",
                "index": index,
                "delta": { "type": "text_delta", "text": text }
            })
            .to_string(),
        ),
        AnthropicSseEvent::ContentBlockInputJsonDelta {
            index,
            partial_json,
        } => (
            "content_block_delta",
            serde_json::json!({
                "type": "content_block_delta",
                "index": index,
                "delta": { "type": "input_json_delta", "partial_json": partial_json }
            })
            .to_string(),
        ),
        AnthropicSseEvent::ContentBlockStop { index } => (
            "content_block_stop",
            serde_json::json!({
                "type": "content_block_stop",
                "index": index
            })
            .to_string(),
        ),
        AnthropicSseEvent::MessageDelta {
            stop_reason,
            stop_sequence,
            output_tokens,
        } => (
            "message_delta",
            serde_json::json!({
                "type": "message_delta",
                "delta": { "stop_reason": stop_reason, "stop_sequence": stop_sequence },
                "usage": { "output_tokens": output_tokens }
            })
            .to_string(),
        ),
        AnthropicSseEvent::MessageStop => (
            "message_stop",
            serde_json::json!({ "type": "message_stop" }).to_string(),
        ),
    }
}

/// Build an SSE response from an Anthropic event channel.
pub(super) fn build_anthropic_sse_response(
    sse_rx: tokio::sync::mpsc::Receiver<AnthropicSseEvent>,
) -> axum::response::Response {
    let stream = tokio_stream::wrappers::ReceiverStream::new(sse_rx).map(move |event| {
        let (event_type, data) = serialize_anthropic_event(&event);
        Ok::<_, Infallible>(Event::default().event(event_type).data(data))
    });
    Sse::new(stream)
        .keep_alive(
            KeepAlive::new().interval(std::time::Duration::from_secs(SSE_KEEPALIVE_INTERVAL_SECS)),
        )
        .into_response()
}

/// Send SSE preamble: message_start + content_block_start.
pub(super) async fn send_sse_preamble(
    sse_tx: &tokio::sync::mpsc::Sender<AnthropicSseEvent>,
    request_id: &str,
    model: &str,
) {
    let _ = sse_tx
        .send(AnthropicSseEvent::MessageStart {
            id: request_id.to_string(),
            model: model.to_string(),
        })
        .await;
    let _ = sse_tx
        .send(AnthropicSseEvent::ContentBlockStart { index: 0 })
        .await;
}

/// Send SSE epilogue: content_block_stop + message_delta + message_stop.
pub(super) async fn send_sse_epilogue(
    sse_tx: &tokio::sync::mpsc::Sender<AnthropicSseEvent>,
    stop_reason: String,
    output_tokens: u32,
) {
    send_sse_epilogue_with_stop(sse_tx, stop_reason, None, output_tokens).await
}

/// Variant that carries the matched `stop_sequence` string in the
/// `message_delta` event when `stop_reason == "stop_sequence"`.
pub(super) async fn send_sse_epilogue_with_stop(
    sse_tx: &tokio::sync::mpsc::Sender<AnthropicSseEvent>,
    stop_reason: String,
    stop_sequence: Option<String>,
    output_tokens: u32,
) {
    let _ = sse_tx
        .send(AnthropicSseEvent::ContentBlockStop { index: 0 })
        .await;
    let _ = sse_tx
        .send(AnthropicSseEvent::MessageDelta {
            stop_reason,
            stop_sequence,
            output_tokens,
        })
        .await;
    let _ = sse_tx.send(AnthropicSseEvent::MessageStop).await;
}
