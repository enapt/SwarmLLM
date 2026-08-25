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
        /// Prompt tokens, when this path knows them.
        ///
        /// Anthropic reports input tokens on `message_start`, which this server
        /// sends before generation begins — at which point the prompt has not
        /// been tokenised and the number does not exist yet. It was therefore
        /// hardcoded to 0 and never corrected, so a STREAMING client could not
        /// see prompt usage at all while the non-streaming sibling reported it
        /// correctly (measured 2026-08-10: 0 against a true 59). Reporting it
        /// here, where it IS known, is what the official SDKs accumulate.
        input_tokens: Option<u32>,
    },
    MessageStop,
    /// A failure, in the shape the Anthropic API actually uses for one.
    ///
    /// This surface had no way to say "that went wrong", and the two streaming
    /// paths invented one each: the router path reported a failed request as a
    /// normal empty turn (`stop_reason: "end_turn"`, HTTP 200 — a policy refusal
    /// or a dead pipeline arrived as the model choosing to say nothing), and the
    /// split path wrote the reason into the assistant's own text as
    /// `[inference failed: …]` and set `stop_reason: "error"`, which is not a
    /// value the API defines — so an SDK deserialising that enum sees a reply
    /// the model never made. Both measured 2026-08-12.
    ///
    /// `error` is terminal: `build_anthropic_sse_response` ends the stream on it
    /// exactly as it does on `message_stop`, so no epilogue follows.
    Error {
        error_type: &'static str,
        message: String,
    },
}

/// Translate our canonical error type into one the Anthropic API defines.
///
/// The canonical set (`crate::error::classify_error`) is OpenAI-flavoured
/// because that is the older surface; Anthropic names some of the same things
/// differently and clients match on the name. The overlapping types pass
/// through unchanged and everything else becomes `api_error`, which is
/// Anthropic's generic server-side failure — never a made-up type, because an
/// unknown one deserialises no better than the `"error"` stop_reason did.
pub(super) fn anthropic_error_type(error_type: &str) -> &'static str {
    match error_type {
        "invalid_request_error" => "invalid_request_error",
        "authentication_error" => "authentication_error",
        "not_found_error" => "not_found_error",
        "permission_error" => "permission_error",
        "rate_limit_error" => "rate_limit_error",
        "request_too_large" => "request_too_large",
        "overloaded_error" => "overloaded_error",
        _ => "api_error",
    }
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
            input_tokens,
        } => ("message_delta", {
            let mut usage = serde_json::json!({ "output_tokens": output_tokens });
            // Omitted rather than sent as 0 when unknown: a client that sums
            // usage across events must not be handed a confident zero.
            if let Some(input) = input_tokens {
                usage["input_tokens"] = (*input).into();
            }
            serde_json::json!({
                "type": "message_delta",
                "delta": { "stop_reason": stop_reason, "stop_sequence": stop_sequence },
                "usage": usage
            })
            .to_string()
        }),
        AnthropicSseEvent::MessageStop => (
            "message_stop",
            serde_json::json!({ "type": "message_stop" }).to_string(),
        ),
        AnthropicSseEvent::Error {
            error_type,
            message,
        } => (
            "error",
            serde_json::json!({
                "type": "error",
                "error": { "type": anthropic_error_type(error_type), "message": message }
            })
            .to_string(),
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
    // `watch` rather than an `AtomicBool`: the ticker waits on this — see
    // `api::sse::progress_ticker`.
    let (finished_tx, finished_rx) = tokio::sync::watch::channel(false);
    let stream = tokio_stream::wrappers::ReceiverStream::new(sse_rx).map(move |event| {
        let (event_type, data) = serialize_anthropic_event(&event);
        // `message_stop` is Anthropic's terminal frame; after it the ticker
        // must end or `merge` would hold the response open forever. `error` is
        // terminal too — upstream ends the stream there rather than following it
        // with an epilogue, and a frame that ends the stream without ending the
        // ticker would leave the client hanging on a request it has already been
        // told failed.
        if event_type == "message_stop" || event_type == "error" {
            let _ = finished_tx.send(true);
        }
        Ok::<_, Infallible>(Event::default().event(event_type).data(data))
    });
    let ticker = crate::api::sse::progress_ticker(
        progress,
        finished_rx,
        std::time::Duration::from_secs(SSE_KEEPALIVE_INTERVAL_SECS),
    );
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
    input_tokens: Option<u32>,
) {
    send_sse_epilogue_with_stop(
        sse_tx,
        stop_reason,
        None,
        output_tokens,
        TextBlock::Open,
        input_tokens,
    )
    .await
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
    input_tokens: Option<u32>,
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
            input_tokens,
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

    /// A failure goes on the wire as Anthropic's `error` event, never as text.
    ///
    /// The split stream used to emit the reason as a `content_block_delta`, so
    /// the model appeared to have said `[inference failed: …]` — indistinguishable
    /// from a real reply, and persisted into conversation history as an assistant
    /// turn. Measured on the released v0.3.95 binary 2026-08-12.
    #[test]
    fn a_failure_is_an_error_event_not_assistant_text() {
        let (name, data) = serialize_anthropic_event(&AnthropicSseEvent::Error {
            error_type: "invalid_request_error",
            message: "This conversation is too long".into(),
        });
        assert_eq!(name, "error", "must use the SSE `error` event name");
        let v: serde_json::Value = serde_json::from_str(&data).expect("valid JSON");
        assert_eq!(v["type"], "error");
        assert_eq!(v["error"]["type"], "invalid_request_error");
        assert_eq!(v["error"]["message"], "This conversation is too long");
        // The reason must not be reachable as assistant content.
        assert!(
            v.get("delta").is_none() && v.get("content_block").is_none(),
            "an error must not carry content: {data}"
        );
    }

    /// `stop_reason` may only carry values the API defines.
    ///
    /// `"error"` is not one of them — the split path used to send it alongside
    /// the fake assistant text, so an SDK deserialising that enum was handed a
    /// value it has no variant for.
    #[test]
    fn stop_reason_is_never_a_value_the_api_does_not_define() {
        // Per the Messages API: end_turn | max_tokens | stop_sequence |
        // tool_use | pause_turn | refusal | model_context_window_exceeded.
        const DEFINED: &[&str] = &[
            "end_turn",
            "max_tokens",
            "stop_sequence",
            "tool_use",
            "pause_turn",
            "refusal",
            "model_context_window_exceeded",
        ];
        for reason in ["end_turn", "max_tokens", "tool_use", "stop_sequence"] {
            let (_n, data) = serialize_anthropic_event(&AnthropicSseEvent::MessageDelta {
                stop_reason: reason.to_string(),
                stop_sequence: None,
                output_tokens: 1,
                input_tokens: None,
            });
            let v: serde_json::Value = serde_json::from_str(&data).unwrap();
            let got = v["delta"]["stop_reason"].as_str().unwrap_or_default();
            assert!(
                DEFINED.contains(&got),
                "`{got}` is not a stop_reason the Anthropic API defines"
            );
        }
    }

    /// Our canonical error types are OpenAI-flavoured; the Anthropic surface
    /// must only ever name a type Anthropic defines, falling back to
    /// `api_error` rather than inventing one.
    #[test]
    fn error_types_are_translated_into_anthropics_own_set() {
        const ANTHROPIC_DEFINED: &[&str] = &[
            "invalid_request_error",
            "authentication_error",
            "permission_error",
            "not_found_error",
            "request_too_large",
            "rate_limit_error",
            "api_error",
            "overloaded_error",
        ];
        // Every type `classify_error` can produce, including the ones with no
        // Anthropic equivalent.
        for ours in [
            "invalid_request_error",
            "not_found_error",
            "authentication_error",
            "server_error",
            "network_error",
            "insufficient_credits",
            "private_mode_error",
            "prompt_privacy_error",
            "service_unavailable",
        ] {
            let mapped = anthropic_error_type(ours);
            assert!(
                ANTHROPIC_DEFINED.contains(&mapped),
                "`{ours}` mapped to `{mapped}`, which Anthropic does not define"
            );
        }
        // A caller's own mistake must not be laundered into a server fault.
        assert_eq!(
            anthropic_error_type("invalid_request_error"),
            "invalid_request_error"
        );
        assert_eq!(anthropic_error_type("server_error"), "api_error");
    }

    /// The epilogue must not close block 0 a second time once the tool path has
    /// closed it — a duplicate stop is as malformed as a missing one.
    #[tokio::test]
    async fn epilogue_skips_the_text_block_when_already_closed() {
        let (tx, mut rx) = tokio::sync::mpsc::channel::<AnthropicSseEvent>(8);
        send_sse_epilogue_with_stop(
            &tx,
            "tool_use".into(),
            None,
            5,
            TextBlock::AlreadyClosed,
            None,
        )
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
        send_sse_epilogue_with_stop(&tx, "end_turn".into(), None, 5, TextBlock::Open, None).await;
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

    /// Prompt usage must reach a STREAMING client.
    ///
    /// `message_start` is emitted before the prompt is tokenised, so its
    /// `input_tokens` is structurally 0; the count is known by the time the
    /// epilogue runs and belongs there. Measured 2026-08-10: a streaming
    /// request reported 0 for a prompt the non-streaming sibling reported as
    /// 59, so a client tracking cost or context from the stream could not see
    /// prompt usage at all.
    #[tokio::test]
    async fn the_epilogue_reports_prompt_tokens_when_known() {
        let (tx, mut rx) = tokio::sync::mpsc::channel::<AnthropicSseEvent>(8);
        send_sse_epilogue_with_stop(&tx, "end_turn".into(), None, 5, TextBlock::Open, Some(38))
            .await;
        drop(tx);

        let mut delta = None;
        while let Some(e) = rx.recv().await {
            let (name, json) = serialize_anthropic_event(&e);
            if name == "message_delta" {
                delta = Some(serde_json::from_str::<serde_json::Value>(&json).unwrap());
            }
        }
        let d = delta.expect("message_delta emitted");
        assert_eq!(d["usage"]["input_tokens"], 38);
        assert_eq!(d["usage"]["output_tokens"], 5);
    }

    /// ...and is OMITTED, not zeroed, on a path that does not know it. A client
    /// summing usage across events must never be handed a confident zero.
    #[tokio::test]
    async fn an_unknown_prompt_count_is_omitted_rather_than_zero() {
        let (tx, mut rx) = tokio::sync::mpsc::channel::<AnthropicSseEvent>(8);
        send_sse_epilogue_with_stop(&tx, "end_turn".into(), None, 5, TextBlock::Open, None).await;
        drop(tx);

        let mut delta = None;
        while let Some(e) = rx.recv().await {
            let (name, json) = serialize_anthropic_event(&e);
            if name == "message_delta" {
                delta = Some(serde_json::from_str::<serde_json::Value>(&json).unwrap());
            }
        }
        let d = delta.expect("message_delta emitted");
        assert!(
            d["usage"].get("input_tokens").is_none(),
            "unknown must be absent, not 0: {}",
            d["usage"]
        );
    }
}
