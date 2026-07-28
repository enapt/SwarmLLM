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
/// `progress` is an optional (state, request uuid) pair used to interleave SSE
/// comment frames describing a not-yet-streaming request — the same treatment
/// the OpenAI encoder gives, so a long prefill reads as progress on this API
/// too rather than as an idle socket. Pass `None` where no trace exists (a
/// cloud proxy stream, for instance, whose latency is not ours to explain).
pub(super) fn build_anthropic_sse_response(
    sse_rx: tokio::sync::mpsc::Receiver<AnthropicSseEvent>,
    progress: Option<(std::sync::Arc<crate::daemon::SharedState>, uuid::Uuid)>,
) -> axum::response::Response {
    let finished = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let finished_for_map = finished.clone();
    let stream = tokio_stream::wrappers::ReceiverStream::new(sse_rx).map(move |event| {
        let (event_type, data) = serialize_anthropic_event(&event);
        // `message_stop` is Anthropic's terminal frame; after it the ticker
        // must end or `merge` would hold the response open forever.
        if event_type == "message_stop" {
            finished_for_map.store(true, std::sync::atomic::Ordering::Relaxed);
        }
        Ok::<_, Infallible>(Event::default().event(event_type).data(data))
    });
    let ticker = futures::stream::unfold((progress, finished), |(p, finished)| async move {
        tokio::time::sleep(std::time::Duration::from_secs(SSE_KEEPALIVE_INTERVAL_SECS)).await;
        if finished.load(std::sync::atomic::Ordering::Relaxed) {
            return None;
        }
        let text = p
            .as_ref()
            .and_then(|(state, rid)| state.active_traces.get(rid).and_then(|t| t.progress()))
            .map(|s| crate::api::sse::format_progress_comment(&s))
            .unwrap_or_default();
        Some((Ok(Event::default().comment(text)), (p, finished)))
    });
    Sse::new(tokio_stream::StreamExt::merge(stream, ticker))
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
///
/// For a response with no tool blocks, so the opening text block is still open.
pub(super) async fn send_sse_epilogue(
    sse_tx: &tokio::sync::mpsc::Sender<AnthropicSseEvent>,
    stop_reason: String,
    output_tokens: u32,
) {
    send_sse_epilogue_with_stop(sse_tx, stop_reason, None, output_tokens, TextBlock::Open).await
}

/// Whether the opening text block (index 0) is still open when the epilogue
/// runs.
///
/// Anthropic's streaming contract is that content blocks are SEQUENTIAL: each
/// has a start, its deltas, and a stop, and block N is closed before block N+1
/// opens. Emitting tool blocks while block 0 was still open produced
/// start(0) → start(1) → stop(1) → stop(0), which is nested rather than
/// sequential and can desynchronise a client that tracks a current block
/// instead of indexing into a map (live 2026-07-26).
///
/// An explicit argument rather than a default, so each call site has to say
/// which shape it produced — the compiler then catches a new streaming path
/// that forgets.
#[derive(Clone, Copy, PartialEq)]
pub(super) enum TextBlock {
    /// Block 0 still open — the epilogue closes it.
    Open,
    /// Block 0 already closed, by the code that opened a later block.
    AlreadyClosed,
}

/// Variant that carries the matched `stop_sequence` string in the
/// `message_delta` event when `stop_reason == "stop_sequence"`.
pub(super) async fn send_sse_epilogue_with_stop(
    sse_tx: &tokio::sync::mpsc::Sender<AnthropicSseEvent>,
    stop_reason: String,
    stop_sequence: Option<String>,
    output_tokens: u32,
    text_block: TextBlock,
) {
    if text_block == TextBlock::Open {
        let _ = sse_tx
            .send(AnthropicSseEvent::ContentBlockStop { index: 0 })
            .await;
    }
    let _ = sse_tx
        .send(AnthropicSseEvent::MessageDelta {
            stop_reason,
            stop_sequence,
            output_tokens,
        })
        .await;
    let _ = sse_tx.send(AnthropicSseEvent::MessageStop).await;
}

#[cfg(test)]
mod block_sequencing_tests {
    use super::*;

    /// Collect the (event name, index) pairs a stream would put on the wire.
    fn seq(events: &[AnthropicSseEvent]) -> Vec<(String, Option<i64>)> {
        events
            .iter()
            .map(|e| {
                let (name, data) = serialize_anthropic_event(e);
                let idx = serde_json::from_str::<serde_json::Value>(&data)
                    .ok()
                    .and_then(|v| v.get("index").and_then(|i| i.as_i64()));
                (name.to_string(), idx)
            })
            .collect()
    }

    /// Anthropic's contract: content blocks are SEQUENTIAL. Block N must stop
    /// before block N+1 starts. We previously emitted
    /// start(0) → start(1) → stop(1) → stop(0), nesting the tool block inside
    /// the still-open text block (live 2026-07-26), which can desynchronise a
    /// client that tracks a current block rather than indexing into a map.
    #[test]
    fn tool_stream_blocks_are_sequential_not_nested() {
        // The order the handler now produces: preamble, close text, tool block,
        // then an epilogue told the text block is already closed.
        let events = vec![
            AnthropicSseEvent::MessageStart {
                id: "m".into(),
                model: "x".into(),
            },
            AnthropicSseEvent::ContentBlockStart { index: 0 },
            AnthropicSseEvent::ContentBlockStop { index: 0 },
            AnthropicSseEvent::ContentBlockStartToolUse {
                index: 1,
                id: "call_1".into(),
                name: "get_weather".into(),
            },
            AnthropicSseEvent::ContentBlockInputJsonDelta {
                index: 1,
                partial_json: "{\"city\":\"Paris\"}".into(),
            },
            AnthropicSseEvent::ContentBlockStop { index: 1 },
        ];

        let mut open: Option<i64> = None;
        for (name, idx) in seq(&events) {
            match name.as_str() {
                "content_block_start" => {
                    assert!(
                        open.is_none(),
                        "block {idx:?} started while {open:?} was still open — blocks must be sequential"
                    );
                    open = idx;
                }
                "content_block_delta" => {
                    assert_eq!(open, idx, "delta for a block that is not the open one");
                }
                "content_block_stop" => {
                    assert_eq!(open, idx, "stop for a block that is not the open one");
                    open = None;
                }
                _ => {}
            }
        }
        assert!(open.is_none(), "a content block was left open: {open:?}");
    }

    /// The epilogue must not close block 0 a second time once the tool path has
    /// closed it — a duplicate stop is as malformed as a missing one.
    #[tokio::test]
    async fn epilogue_skips_the_text_block_when_already_closed() {
        let (tx, mut rx) = tokio::sync::mpsc::channel::<AnthropicSseEvent>(8);
        send_sse_epilogue_with_stop(&tx, "tool_use".into(), None, 5, TextBlock::AlreadyClosed)
            .await;
        drop(tx);

        let mut names = Vec::new();
        while let Some(e) = rx.recv().await {
            names.push(serialize_anthropic_event(&e).0);
        }
        assert!(
            !names.contains(&"content_block_stop"),
            "epilogue closed an already-closed block: {names:?}"
        );
        assert_eq!(names, vec!["message_delta", "message_stop"]);
    }

    /// A plain text response still gets its block closed by the epilogue.
    #[tokio::test]
    async fn epilogue_closes_the_text_block_when_still_open() {
        let (tx, mut rx) = tokio::sync::mpsc::channel::<AnthropicSseEvent>(8);
        send_sse_epilogue_with_stop(&tx, "end_turn".into(), None, 5, TextBlock::Open).await;
        drop(tx);

        let mut names = Vec::new();
        while let Some(e) = rx.recv().await {
            names.push(serialize_anthropic_event(&e).0);
        }
        assert_eq!(
            names,
            vec!["content_block_stop", "message_delta", "message_stop"]
        );
    }
}
