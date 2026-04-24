//! SSE streaming for `POST /v1/responses` (local inference path).
//!
//! Strategy: translate to a streaming Chat Completions request, call the
//! existing handler, parse its SSE body, and re-emit each chat chunk as
//! the appropriate Responses events with a monotonic `sequence_number`.
//!
//! Events emitted in the text path:
//! 1. `response.created` — minimal response metadata.
//! 2. `response.in_progress` — model is running.
//! 3. `response.output_item.added` — assistant message item, empty content.
//! 4. `response.content_part.added` — `output_text` part, empty text.
//! 5. `response.output_text.delta` — one per content delta.
//! 6. `response.output_text.done` — full accumulated text.
//! 7. `response.content_part.done` — finalized part.
//! 8. `response.output_item.done` — finalized message item.
//! 9. `response.completed` — full response object.
//!
//! Tool-call path (assistant returns tool_calls instead of text):
//! - `response.output_item.added` per tool call (`function_call` item).
//! - `response.function_call_arguments.delta` per accumulated args fragment.
//! - `response.function_call_arguments.done` per call with full args.
//! - `response.output_item.done` per call.
//! - `response.completed`.
//!
//! Cloud-proxy requests never hit this module — they stream verbatim via
//! the provider's own SSE. This path is local-inference only.

use std::collections::HashMap;
use std::convert::Infallible;

use axum::body::Body;
use axum::extract::State;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use bytes::{Buf, BytesMut};
use futures::{Stream, StreamExt};

use super::translate;
use super::types::*;
use crate::api::server::{AppState, JsonBody};
use crate::error::ApiError;

const SSE_KEEPALIVE_INTERVAL_SECS: u64 = 15;

/// Entry point for streaming `/v1/responses` on the local-inference path.
pub async fn run_streaming(
    state: AppState,
    headers: axum::http::HeaderMap,
    req: ResponsesRequest,
) -> Result<Response, ApiError> {
    let mut chat_req = translate::request_to_chat(&req)?;
    chat_req.stream = true;

    let chat_response = crate::api::openai::chat_completions(
        State(state.clone()),
        headers.clone(),
        JsonBody(chat_req),
    )
    .await?;

    if !chat_response.status().is_success() {
        return Ok(chat_response);
    }

    let response_id = format!("resp_{}", uuid::Uuid::new_v4().simple());
    let item_id = format!("msg_{}", uuid::Uuid::new_v4().simple());
    let created_at = chrono::Utc::now().timestamp();

    let initial_response = build_initial_response(&req, &response_id, created_at);

    let chat_body: Body = chat_response.into_body();
    let chat_stream = chat_body.into_data_stream();

    let sse_stream = build_response_event_stream(
        chat_stream,
        req,
        initial_response,
        response_id,
        item_id,
        created_at,
    );

    Ok(Sse::new(sse_stream)
        .keep_alive(
            KeepAlive::new().interval(std::time::Duration::from_secs(SSE_KEEPALIVE_INTERVAL_SECS)),
        )
        .into_response())
}

// ============================================================================
// Stream transformation
// ============================================================================

/// Extract complete SSE `data:` payloads from the accumulator. Consumes
/// bytes up to and including each `\n\n` boundary. Lines that aren't
/// `data:` (e.g. `event:`, `:keepalive`) are ignored.
fn drain_sse_data_payloads(buf: &mut BytesMut) -> Vec<String> {
    let mut out = Vec::new();
    while let Some(pos) = find_subslice(buf, b"\n\n") {
        let block = buf[..pos].to_vec();
        for line in block.split(|&b| b == b'\n') {
            let line = line.strip_prefix(b"data:").unwrap_or(&[]);
            let line = line.strip_prefix(b" ").unwrap_or(line);
            if line.is_empty() {
                continue;
            }
            if let Ok(s) = std::str::from_utf8(line) {
                out.push(s.to_string());
            }
        }
        buf.advance(pos + 2);
    }
    out
}

fn find_subslice(hay: &[u8], needle: &[u8]) -> Option<usize> {
    hay.windows(needle.len()).position(|w| w == needle)
}

/// Map a helper SSE event name + JSON body into an axum `Event`.
fn sse_event(name: &str, body: serde_json::Value) -> Event {
    Event::default()
        .event(name)
        .data(serde_json::to_string(&body).unwrap_or_default())
}

/// Accumulated state for streaming tool calls (chat streams arguments as
/// fragments; we need to track the current fragment per `index`).
#[derive(Default, Clone)]
struct StreamingToolCall {
    id: String,
    name: String,
    arguments_so_far: String,
}

fn build_response_event_stream<S>(
    chat_stream: S,
    original: ResponsesRequest,
    initial_response: ResponsesResponse,
    response_id: String,
    item_id: String,
    created_at: i64,
) -> impl Stream<Item = Result<Event, Infallible>>
where
    S: Stream<Item = Result<bytes::Bytes, axum::Error>> + Send + 'static,
{
    async_stream::stream! {
        let mut seq: u64 = 0;
        let mut buf = BytesMut::new();

        let mut item_added = false;
        let mut content_part_added = false;
        let mut accumulated_text = String::new();
        let mut finish_reason: Option<String> = None;
        // Keyed by tool_call `index` (chat emits fragments under a stable index).
        let mut tool_calls: HashMap<u64, StreamingToolCall> = HashMap::new();
        let mut tool_call_order: Vec<u64> = Vec::new();
        let mut model_from_chunk: Option<String> = None;

        // ---- response.created ----
        yield Ok::<_, Infallible>(sse_event(
            "response.created",
            serde_json::json!({
                "type": "response.created",
                "sequence_number": seq,
                "response": initial_response,
            }),
        ));
        seq += 1;

        // ---- response.in_progress ----
        yield Ok(sse_event(
            "response.in_progress",
            serde_json::json!({
                "type": "response.in_progress",
                "sequence_number": seq,
                "response": initial_response,
            }),
        ));
        seq += 1;

        let mut chat_stream = Box::pin(chat_stream);
        'outer: while let Some(chunk_result) = chat_stream.next().await {
            let chunk_bytes = match chunk_result {
                Ok(b) => b,
                Err(e) => {
                    tracing::warn!(error = %e, "responses stream: chat body error");
                    finish_reason = Some("error".into());
                    break 'outer;
                }
            };
            buf.extend_from_slice(&chunk_bytes);

            for data in drain_sse_data_payloads(&mut buf) {
                if data == "[DONE]" {
                    break 'outer;
                }
                let chunk: serde_json::Value = match serde_json::from_str(&data) {
                    Ok(v) => v,
                    Err(e) => {
                        tracing::warn!(error = %e, data = %data, "responses stream: malformed chat chunk");
                        continue;
                    }
                };

                if model_from_chunk.is_none() {
                    if let Some(m) = chunk.get("model").and_then(|v| v.as_str()) {
                        model_from_chunk = Some(m.to_string());
                    }
                }

                let choice = match chunk.get("choices").and_then(|c| c.get(0)) {
                    Some(c) => c,
                    None => continue,
                };
                let delta = choice.get("delta");

                // Open the message item on the first role or content delta.
                if !item_added {
                    let has_role_or_content = delta
                        .map(|d| {
                            d.get("role").is_some()
                                || d.get("content").and_then(|v| v.as_str()).is_some()
                                || d.get("tool_calls").is_some()
                        })
                        .unwrap_or(false);
                    if has_role_or_content {
                        let opening_item = serde_json::json!({
                            "type": "message",
                            "id": item_id,
                            "role": "assistant",
                            "status": "in_progress",
                            "content": [],
                        });
                        yield Ok(sse_event(
                            "response.output_item.added",
                            serde_json::json!({
                                "type": "response.output_item.added",
                                "sequence_number": seq,
                                "output_index": 0,
                                "item": opening_item,
                            }),
                        ));
                        seq += 1;
                        item_added = true;
                    }
                }

                // Content delta.
                if let Some(text) = delta.and_then(|d| d.get("content")).and_then(|c| c.as_str()) {
                    if !text.is_empty() {
                        if !content_part_added {
                            let part_shell = serde_json::json!({
                                "type": "output_text",
                                "text": "",
                                "annotations": [],
                            });
                            yield Ok(sse_event(
                                "response.content_part.added",
                                serde_json::json!({
                                    "type": "response.content_part.added",
                                    "sequence_number": seq,
                                    "output_index": 0,
                                    "content_index": 0,
                                    "item_id": item_id,
                                    "part": part_shell,
                                }),
                            ));
                            seq += 1;
                            content_part_added = true;
                        }

                        accumulated_text.push_str(text);
                        yield Ok(sse_event(
                            "response.output_text.delta",
                            serde_json::json!({
                                "type": "response.output_text.delta",
                                "sequence_number": seq,
                                "output_index": 0,
                                "content_index": 0,
                                "item_id": item_id,
                                "delta": text,
                            }),
                        ));
                        seq += 1;
                    }
                }

                // Tool-call deltas. Chat streams each call by `index`, with
                // partial `arguments` strings that we concatenate. The
                // first fragment for an index carries `id` + `function.name`.
                if let Some(tcs) = delta.and_then(|d| d.get("tool_calls")).and_then(|v| v.as_array()) {
                    for tc in tcs {
                        let index = tc.get("index").and_then(|v| v.as_u64()).unwrap_or(0);
                        let entry = tool_calls.entry(index).or_insert_with(|| {
                            tool_call_order.push(index);
                            StreamingToolCall::default()
                        });
                        if let Some(id) = tc.get("id").and_then(|v| v.as_str()) {
                            if entry.id.is_empty() {
                                entry.id = id.to_string();
                            }
                        }
                        if let Some(name) = tc
                            .get("function")
                            .and_then(|f| f.get("name"))
                            .and_then(|v| v.as_str())
                        {
                            if entry.name.is_empty() {
                                entry.name = name.to_string();
                            }
                        }
                        if let Some(args) = tc
                            .get("function")
                            .and_then(|f| f.get("arguments"))
                            .and_then(|v| v.as_str())
                        {
                            entry.arguments_so_far.push_str(args);
                        }
                    }
                }

                if let Some(fr) = choice.get("finish_reason").and_then(|v| v.as_str()) {
                    finish_reason = Some(fr.to_string());
                }
            }
        }

        // ---- Finalize text path ----
        if content_part_added {
            yield Ok(sse_event(
                "response.output_text.done",
                serde_json::json!({
                    "type": "response.output_text.done",
                    "sequence_number": seq,
                    "output_index": 0,
                    "content_index": 0,
                    "item_id": item_id,
                    "text": accumulated_text,
                }),
            ));
            seq += 1;
            yield Ok(sse_event(
                "response.content_part.done",
                serde_json::json!({
                    "type": "response.content_part.done",
                    "sequence_number": seq,
                    "output_index": 0,
                    "content_index": 0,
                    "item_id": item_id,
                    "part": {
                        "type": "output_text",
                        "text": accumulated_text,
                        "annotations": [],
                    },
                }),
            ));
            seq += 1;
        }

        // Build final list of output items for response.completed.
        let mut output: Vec<OutputItem> = Vec::new();

        if item_added {
            // Emit output_item.done for the message (may have empty content
            // if the assistant went straight to tool calls).
            let message_item = serde_json::json!({
                "type": "message",
                "id": item_id,
                "role": "assistant",
                "status": "completed",
                "content": if accumulated_text.is_empty() {
                    serde_json::json!([])
                } else {
                    serde_json::json!([{
                        "type": "output_text",
                        "text": accumulated_text,
                        "annotations": [],
                    }])
                },
            });

            // Only emit a message output item when we actually produced text.
            // An empty assistant message alongside function_call items would
            // break the "pure function_call output" shape.
            if !accumulated_text.is_empty() {
                yield Ok(sse_event(
                    "response.output_item.done",
                    serde_json::json!({
                        "type": "response.output_item.done",
                        "sequence_number": seq,
                        "output_index": 0,
                        "item": message_item.clone(),
                    }),
                ));
                seq += 1;

                output.push(OutputItem::Typed(TypedOutputItem::Message(
                    OutputMessageItem {
                        id: item_id.clone(),
                        role: "assistant".into(),
                        status: Some("completed".into()),
                        content: vec![OutputContentPart::Typed(TypedOutputContentPart::Text {
                            text: accumulated_text.clone(),
                            annotations: Vec::new(),
                            logprobs: None,
                            extras: HashMap::new(),
                        })],
                        extras: HashMap::new(),
                    },
                )));
            }
        }

        // ---- Tool-call items ----
        let mut current_output_index: u64 = if output.is_empty() { 0 } else { 1 };
        for idx in &tool_call_order {
            let tc = match tool_calls.get(idx) {
                Some(v) if !v.id.is_empty() && !v.name.is_empty() => v.clone(),
                _ => continue,
            };
            let fc_item_id = format!("fc_{}", tc.id);
            let opening = serde_json::json!({
                "type": "function_call",
                "id": fc_item_id,
                "call_id": tc.id,
                "name": tc.name,
                "arguments": "",
                "status": "in_progress",
            });
            yield Ok(sse_event(
                "response.output_item.added",
                serde_json::json!({
                    "type": "response.output_item.added",
                    "sequence_number": seq,
                    "output_index": current_output_index,
                    "item": opening,
                }),
            ));
            seq += 1;

            // Emit the full accumulated arguments as a single delta. Chat's
            // per-fragment deltas are collapsed here because the Responses
            // event semantics want arguments-as-a-string-stream, which we
            // could forward fragment-by-fragment in a future refinement.
            if !tc.arguments_so_far.is_empty() {
                yield Ok(sse_event(
                    "response.function_call_arguments.delta",
                    serde_json::json!({
                        "type": "response.function_call_arguments.delta",
                        "sequence_number": seq,
                        "output_index": current_output_index,
                        "item_id": fc_item_id,
                        "delta": tc.arguments_so_far,
                    }),
                ));
                seq += 1;
            }

            yield Ok(sse_event(
                "response.function_call_arguments.done",
                serde_json::json!({
                    "type": "response.function_call_arguments.done",
                    "sequence_number": seq,
                    "output_index": current_output_index,
                    "item_id": fc_item_id,
                    "arguments": tc.arguments_so_far,
                }),
            ));
            seq += 1;

            let completed = serde_json::json!({
                "type": "function_call",
                "id": fc_item_id,
                "call_id": tc.id,
                "name": tc.name,
                "arguments": tc.arguments_so_far,
                "status": "completed",
            });
            yield Ok(sse_event(
                "response.output_item.done",
                serde_json::json!({
                    "type": "response.output_item.done",
                    "sequence_number": seq,
                    "output_index": current_output_index,
                    "item": completed,
                }),
            ));
            seq += 1;

            output.push(OutputItem::Typed(TypedOutputItem::FunctionCall(
                FunctionCallItem {
                    call_id: tc.id.clone(),
                    name: tc.name.clone(),
                    arguments: tc.arguments_so_far.clone(),
                    id: Some(fc_item_id),
                    status: Some("completed".into()),
                    extras: HashMap::new(),
                },
            )));
            current_output_index += 1;
        }

        // ---- Final response object ----
        let status = match finish_reason.as_deref() {
            Some("length") => ResponseStatus::Incomplete,
            Some("content_filter") | Some("error") => ResponseStatus::Failed,
            _ => ResponseStatus::Completed,
        };

        let incomplete_details = (status == ResponseStatus::Incomplete).then(|| {
            IncompleteDetails {
                reason: "max_output_tokens".into(),
                extras: HashMap::new(),
            }
        });

        let final_model = model_from_chunk.unwrap_or_else(|| original.model.clone());
        let final_response = ResponsesResponse {
            id: response_id.clone(),
            object: "response".into(),
            created_at,
            status,
            model: final_model,
            output,
            output_text: Some(accumulated_text.clone()),
            // Streaming paths don't have final usage counts without a
            // provider echoing them. Leave zero; upstream callers that
            // need totals can use the non-streaming path.
            usage: ResponsesUsage::default(),
            error: None,
            incomplete_details,
            previous_response_id: original.previous_response_id.clone(),
            instructions: original.instructions.clone(),
            tools: original.tools.clone(),
            tool_choice: original.tool_choice.clone(),
            parallel_tool_calls: original.parallel_tool_calls,
            temperature: Some(original.temperature.unwrap_or(0.7)),
            top_p: Some(original.top_p.unwrap_or(0.9)),
            max_output_tokens: Some(original.max_output_tokens.unwrap_or(2048)),
            truncation: original.truncation.clone(),
            metadata: original.metadata.clone(),
            user: original.user.clone(),
            reasoning: original.reasoning.clone(),
            text: original.text.clone(),
            modalities: original.modalities.clone(),
            service_tier: original.service_tier.clone(),
            background: original.background,
            extras: HashMap::new(),
        };

        let terminal_event = if status == ResponseStatus::Failed {
            "response.failed"
        } else if status == ResponseStatus::Incomplete {
            "response.incomplete"
        } else {
            "response.completed"
        };
        yield Ok(sse_event(
            terminal_event,
            serde_json::json!({
                "type": terminal_event,
                "sequence_number": seq,
                "response": final_response,
            }),
        ));
    }
}

/// Build the minimal response object emitted with the `response.created` /
/// `response.in_progress` events. It carries request metadata only — the
/// output array fills in during streaming, and `response.completed` carries
/// the full populated form.
fn build_initial_response(
    req: &ResponsesRequest,
    response_id: &str,
    created_at: i64,
) -> ResponsesResponse {
    ResponsesResponse {
        id: response_id.into(),
        object: "response".into(),
        created_at,
        status: ResponseStatus::InProgress,
        model: req.model.clone(),
        output: Vec::new(),
        output_text: None,
        usage: ResponsesUsage::default(),
        error: None,
        incomplete_details: None,
        previous_response_id: req.previous_response_id.clone(),
        instructions: req.instructions.clone(),
        tools: req.tools.clone(),
        tool_choice: req.tool_choice.clone(),
        parallel_tool_calls: req.parallel_tool_calls,
        temperature: Some(req.temperature.unwrap_or(0.7)),
        top_p: Some(req.top_p.unwrap_or(0.9)),
        max_output_tokens: Some(req.max_output_tokens.unwrap_or(2048)),
        truncation: req.truncation.clone(),
        metadata: req.metadata.clone(),
        user: req.user.clone(),
        reasoning: req.reasoning.clone(),
        text: req.text.clone(),
        modalities: req.modalities.clone(),
        service_tier: req.service_tier.clone(),
        background: req.background,
        extras: HashMap::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn drain_extracts_two_events() {
        let mut buf = BytesMut::from(&b"data: {\"a\":1}\n\ndata: {\"b\":2}\n\n"[..]);
        let events = drain_sse_data_payloads(&mut buf);
        assert_eq!(
            events,
            vec!["{\"a\":1}".to_string(), "{\"b\":2}".to_string()]
        );
        assert!(buf.is_empty());
    }

    #[test]
    fn drain_leaves_partial_event() {
        let mut buf = BytesMut::from(&b"data: {\"a\":1}\n\ndata: {\"b\":"[..]);
        let events = drain_sse_data_payloads(&mut buf);
        assert_eq!(events, vec!["{\"a\":1}".to_string()]);
        // Partial second event remains in buffer.
        assert!(!buf.is_empty());
    }

    #[test]
    fn drain_handles_done_marker() {
        let mut buf = BytesMut::from(&b"data: [DONE]\n\n"[..]);
        let events = drain_sse_data_payloads(&mut buf);
        assert_eq!(events, vec!["[DONE]".to_string()]);
    }

    #[test]
    fn drain_ignores_non_data_lines() {
        let mut buf = BytesMut::from(&b": keepalive\nevent: foo\ndata: {\"x\":1}\n\n"[..]);
        let events = drain_sse_data_payloads(&mut buf);
        assert_eq!(events, vec!["{\"x\":1}".to_string()]);
    }

    #[test]
    fn drain_handles_data_without_space() {
        // Spec allows `data:X` (no space) — some upstreams omit it.
        let mut buf = BytesMut::from(&b"data:{\"a\":1}\n\n"[..]);
        let events = drain_sse_data_payloads(&mut buf);
        assert_eq!(events, vec!["{\"a\":1}".to_string()]);
    }

    #[test]
    fn drain_progressive_feed() {
        // Simulate bytes arriving in small chunks.
        let mut buf = BytesMut::new();
        buf.extend_from_slice(b"data: {\"");
        assert!(drain_sse_data_payloads(&mut buf).is_empty());
        buf.extend_from_slice(b"a\":1}\n");
        assert!(drain_sse_data_payloads(&mut buf).is_empty());
        buf.extend_from_slice(b"\ndata: [DONE]\n\n");
        let events = drain_sse_data_payloads(&mut buf);
        assert_eq!(events, vec!["{\"a\":1}".to_string(), "[DONE]".to_string()]);
    }
}
