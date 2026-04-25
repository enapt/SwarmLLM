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

use axum::body::to_bytes;
use axum::extract::State;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use bytes::{Buf, BytesMut};
use futures::{Stream, StreamExt};

use super::background::BufferedEvent;
use super::store;
use super::translate;
use super::types::*;
use crate::api::server::{AppState, JsonBody};
use crate::error::ApiError;
use crate::storage::db::Database;

/// Keep-alive interval for SSE responses on `/v1/responses`. Shared with
/// `background.rs` so the resume-stream and completed-replay paths use
/// the same heartbeat cadence as the live local-inference stream.
pub(super) const SSE_KEEPALIVE_INTERVAL_SECS: u64 = 15;

/// Cap on a non-success chat response body when surfacing it as the
/// message of a `response.failed` SSE event. Mirrors `MAX_CHAT_RESPONSE_BYTES`
/// in the non-streaming path.
const MAX_CHAT_ERROR_BODY_BYTES: usize = 16 * 1024 * 1024;

/// Entry point for streaming `/v1/responses` on the local-inference path.
///
/// V1 (responses_api_v2): the chat-completions handler is *not* awaited
/// before the SSE stream opens. We yield `response.created` and
/// `response.in_progress` immediately, then await the inference future
/// inside the generator. This shaves the chat handler's preflight latency
/// (model resolution, worker probe, template build, etc.) off the
/// first-token timing — `response.created` arrives within a few ms of the
/// HTTP handler entering the route.
///
/// Errors from chat_completions (whether sync `ApiError` or non-success
/// `Response`) surface as `response.failed` events in the open SSE stream
/// rather than HTTP error responses, since by the time the future resolves
/// the SSE response has already been sent.
pub async fn run_streaming(
    state: AppState,
    headers: axum::http::HeaderMap,
    req: ResponsesRequest,
    prior: Option<store::ResponsesRecord>,
) -> Result<Response, ApiError> {
    // Translate is sync and validation-only — keep it before the SSE opens
    // so a malformed request still returns a normal 4xx instead of an SSE
    // 200 with response.failed inside.
    let mut chat_req = translate::request_to_chat(&req, prior.as_ref())?;
    chat_req.stream = true;

    let response_id = format!("resp_{}", uuid::Uuid::new_v4().simple());
    let item_id = format!("msg_{}", uuid::Uuid::new_v4().simple());
    let created_at = chrono::Utc::now().timestamp();

    let initial_response = build_initial_response(&req, &response_id, created_at);

    // M7: streaming responses are persisted at stream-end when store=true
    // (OpenAI default). Pass both into the generator so it can write after
    // the final_response is built but before the terminal event yields.
    let store_db = req.store.unwrap_or(true).then(|| state.db.clone());

    // The chat_future is awaited *inside* the SSE generator so the early
    // lifecycle events flush before any inference setup blocks.
    let state_for_chat = state.clone();
    let headers_for_chat = headers.clone();
    let chat_future = async move {
        crate::api::openai::chat_completions(
            State(state_for_chat),
            headers_for_chat,
            JsonBody(chat_req),
        )
        .await
    };

    let buffered_stream = build_response_event_stream(
        chat_future,
        req,
        initial_response,
        response_id,
        item_id,
        created_at,
        store_db,
    );

    // Live SSE: wrap the BufferedEvent stream as axum Events.
    let sse_stream = buffered_stream.map(|ev| Ok::<_, Infallible>(buffered_to_event(&ev)));

    Ok(Sse::new(sse_stream)
        .keep_alive(
            KeepAlive::new().interval(std::time::Duration::from_secs(SSE_KEEPALIVE_INTERVAL_SECS)),
        )
        .into_response())
}

/// V8 (responses_api_v2): public entry point the background-streaming
/// driver uses to obtain the raw `BufferedEvent` stream (no SSE wrap).
/// Invokes the same generator that `run_streaming` uses, so the
/// streaming behavior stays bit-for-bit identical between direct-SSE
/// and background-buffered callers.
pub fn run_streaming_buffered<F>(
    chat_future: F,
    req: ResponsesRequest,
    response_id: String,
    item_id: String,
    created_at: i64,
    store_db: Option<Database>,
) -> impl Stream<Item = BufferedEvent>
where
    F: std::future::Future<Output = Result<Response, ApiError>> + Send + 'static,
{
    let initial_response = build_initial_response(&req, &response_id, created_at);
    build_response_event_stream(
        chat_future,
        req,
        initial_response,
        response_id,
        item_id,
        created_at,
        store_db,
    )
}

/// Convert a buffered event back into the axum SSE Event representation.
fn buffered_to_event(ev: &BufferedEvent) -> Event {
    Event::default()
        .event(&ev.event_name)
        .data(serde_json::to_string(&ev.data).unwrap_or_default())
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

/// Accumulated state for streaming tool calls (chat streams arguments as
/// fragments; we need to track the current fragment per `index`).
#[derive(Default, Clone)]
struct StreamingToolCall {
    id: String,
    name: String,
    arguments_so_far: String,
}

fn build_response_event_stream<F>(
    chat_future: F,
    original: ResponsesRequest,
    initial_response: ResponsesResponse,
    response_id: String,
    item_id: String,
    created_at: i64,
    store_db: Option<Database>,
) -> impl Stream<Item = BufferedEvent>
where
    F: std::future::Future<Output = Result<Response, ApiError>> + Send + 'static,
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

        // ---- response.created (V1: emitted before chat_future is awaited) ----
        yield BufferedEvent {
            sequence_number: seq,
            event_name: "response.created".into(),
            data:
            serde_json::json!({
                "type": "response.created",
                "sequence_number": seq,
                "response": initial_response,
            }),
        };
        seq += 1;

        // ---- response.in_progress ----
        yield BufferedEvent {
            sequence_number: seq,
            event_name: "response.in_progress".into(),
            data:
            serde_json::json!({
                "type": "response.in_progress",
                "sequence_number": seq,
                "response": initial_response,
            }),
        };
        seq += 1;

        // ---- Now await the inference setup. Errors and non-success
        // responses become response.failed events because the SSE response
        // is already in flight.
        let chat_response = match chat_future.await {
            Ok(r) => r,
            Err(e) => {
                let error = ResponseError {
                    code: classify_error_code(&e),
                    message: e.0.to_string(),
                    extras: HashMap::new(),
                };
                let failed = build_failed_response(
                    &original,
                    &response_id,
                    created_at,
                    error.clone(),
                );
                if let Some(db) = store_db.as_ref() {
                    let record = store::ResponsesRecord::new(
                        original.clone(),
                        failed.clone(),
                        created_at,
                        store::DEFAULT_TTL_SECS,
                    );
                    if let Err(e) = store::store(db, &record) {
                        tracing::warn!(error = %e, id = %response_id, "responses stream store failed (early error)");
                    }
                }
                yield BufferedEvent {
            sequence_number: seq,
            event_name: "response.failed".into(),
            data:
                    serde_json::json!({
                        "type": "response.failed",
                        "sequence_number": seq,
                        "response": failed,
                    }),
        };
                return;
            }
        };

        if !chat_response.status().is_success() {
            let status_code = chat_response.status();
            let bytes = match to_bytes(chat_response.into_body(), MAX_CHAT_ERROR_BODY_BYTES).await {
                Ok(b) => b,
                Err(e) => {
                    let error = ResponseError {
                        code: "internal_error".into(),
                        message: format!("buffer chat error body: {e}"),
                        extras: HashMap::new(),
                    };
                    let failed = build_failed_response(
                        &original,
                        &response_id,
                        created_at,
                        error,
                    );
                    yield BufferedEvent {
            sequence_number: seq,
            event_name: "response.failed".into(),
            data:
                        serde_json::json!({
                            "type": "response.failed",
                            "sequence_number": seq,
                            "response": failed,
                        }),
        };
                    return;
                }
            };
            let message = parse_error_message(&bytes);
            let code = if status_code.is_client_error() {
                "invalid_request_error"
            } else {
                "upstream_error"
            };
            let error = ResponseError {
                code: code.into(),
                message,
                extras: HashMap::new(),
            };
            let failed = build_failed_response(
                &original,
                &response_id,
                created_at,
                error,
            );
            if let Some(db) = store_db.as_ref() {
                let record = store::ResponsesRecord::new(
                    original.clone(),
                    failed.clone(),
                    created_at,
                    store::DEFAULT_TTL_SECS,
                );
                if let Err(e) = store::store(db, &record) {
                    tracing::warn!(error = %e, id = %response_id, "responses stream store failed (chat error)");
                }
            }
            yield BufferedEvent {
            sequence_number: seq,
            event_name: "response.failed".into(),
            data:
                serde_json::json!({
                    "type": "response.failed",
                    "sequence_number": seq,
                    "response": failed,
                }),
        };
            return;
        }

        let chat_body = chat_response.into_body();
        let chat_stream = chat_body.into_data_stream();
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
                        yield BufferedEvent {
            sequence_number: seq,
            event_name: "response.output_item.added".into(),
            data:
                            serde_json::json!({
                                "type": "response.output_item.added",
                                "sequence_number": seq,
                                "output_index": 0,
                                "item": opening_item,
                            }),
        };
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
                            yield BufferedEvent {
            sequence_number: seq,
            event_name: "response.content_part.added".into(),
            data:
                                serde_json::json!({
                                    "type": "response.content_part.added",
                                    "sequence_number": seq,
                                    "output_index": 0,
                                    "content_index": 0,
                                    "item_id": item_id,
                                    "part": part_shell,
                                }),
        };
                            seq += 1;
                            content_part_added = true;
                        }

                        accumulated_text.push_str(text);
                        yield BufferedEvent {
            sequence_number: seq,
            event_name: "response.output_text.delta".into(),
            data:
                            serde_json::json!({
                                "type": "response.output_text.delta",
                                "sequence_number": seq,
                                "output_index": 0,
                                "content_index": 0,
                                "item_id": item_id,
                                "delta": text,
                            }),
        };
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
            yield BufferedEvent {
            sequence_number: seq,
            event_name: "response.output_text.done".into(),
            data:
                serde_json::json!({
                    "type": "response.output_text.done",
                    "sequence_number": seq,
                    "output_index": 0,
                    "content_index": 0,
                    "item_id": item_id,
                    "text": accumulated_text,
                }),
        };
            seq += 1;
            yield BufferedEvent {
            sequence_number: seq,
            event_name: "response.content_part.done".into(),
            data:
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
        };
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
                yield BufferedEvent {
            sequence_number: seq,
            event_name: "response.output_item.done".into(),
            data:
                    serde_json::json!({
                        "type": "response.output_item.done",
                        "sequence_number": seq,
                        "output_index": 0,
                        "item": message_item.clone(),
                    }),
        };
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
            yield BufferedEvent {
            sequence_number: seq,
            event_name: "response.output_item.added".into(),
            data:
                serde_json::json!({
                    "type": "response.output_item.added",
                    "sequence_number": seq,
                    "output_index": current_output_index,
                    "item": opening,
                }),
        };
            seq += 1;

            // Emit the full accumulated arguments as a single delta. Chat's
            // per-fragment deltas are collapsed here because the Responses
            // event semantics want arguments-as-a-string-stream, which we
            // could forward fragment-by-fragment in a future refinement.
            if !tc.arguments_so_far.is_empty() {
                yield BufferedEvent {
            sequence_number: seq,
            event_name: "response.function_call_arguments.delta".into(),
            data:
                    serde_json::json!({
                        "type": "response.function_call_arguments.delta",
                        "sequence_number": seq,
                        "output_index": current_output_index,
                        "item_id": fc_item_id,
                        "delta": tc.arguments_so_far,
                    }),
        };
                seq += 1;
            }

            yield BufferedEvent {
            sequence_number: seq,
            event_name: "response.function_call_arguments.done".into(),
            data:
                serde_json::json!({
                    "type": "response.function_call_arguments.done",
                    "sequence_number": seq,
                    "output_index": current_output_index,
                    "item_id": fc_item_id,
                    "arguments": tc.arguments_so_far,
                }),
        };
            seq += 1;

            let completed = serde_json::json!({
                "type": "function_call",
                "id": fc_item_id,
                "call_id": tc.id,
                "name": tc.name,
                "arguments": tc.arguments_so_far,
                "status": "completed",
            });
            yield BufferedEvent {
            sequence_number: seq,
            event_name: "response.output_item.done".into(),
            data:
                serde_json::json!({
                    "type": "response.output_item.done",
                    "sequence_number": seq,
                    "output_index": current_output_index,
                    "item": completed,
                }),
        };
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
            max_output_tokens: Some(
                original
                    .max_output_tokens
                    .unwrap_or(super::DEFAULT_MAX_OUTPUT_TOKENS),
            ),
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

        // Persist the fully-assembled response before emitting the
        // terminal event so a subsequent GET sees the same record the
        // caller just observed close the stream.
        if let Some(db) = store_db.as_ref() {
            let record = store::ResponsesRecord::new(
                original.clone(),
                final_response.clone(),
                created_at,
                store::DEFAULT_TTL_SECS,
            );
            if let Err(e) = store::store(db, &record) {
                tracing::warn!(error = %e, id = %response_id, "responses stream store failed");
            }
        }

        let terminal_event = if status == ResponseStatus::Failed {
            "response.failed"
        } else if status == ResponseStatus::Incomplete {
            "response.incomplete"
        } else {
            "response.completed"
        };
        yield BufferedEvent {
            sequence_number: seq,
            event_name: terminal_event.into(),
            data:
            serde_json::json!({
                "type": terminal_event,
                "sequence_number": seq,
                "response": final_response,
            }),
        };
    }
}

/// Build a `response.failed` final response carrying the given error.
/// Used when chat_completions errors after the SSE stream has already
/// opened, so the caller still receives a structurally-valid Responses
/// object instead of a closed stream.
fn build_failed_response(
    req: &ResponsesRequest,
    response_id: &str,
    created_at: i64,
    error: ResponseError,
) -> ResponsesResponse {
    let mut resp = build_initial_response(req, response_id, created_at);
    resp.status = ResponseStatus::Failed;
    resp.error = Some(error);
    resp
}

/// Pick a coarse error code for an `ApiError` raised during chat-completions
/// setup. The chat handler emits SwarmError variants that fan out to
/// roughly two buckets (request validation vs. internal/upstream); this
/// matches them onto the OpenAI error-code naming used in non-streaming
/// failures.
fn classify_error_code(err: &ApiError) -> String {
    use crate::error::SwarmError;
    match &err.0 {
        SwarmError::Validation(_) => "invalid_request_error".into(),
        SwarmError::ModelNotAvailable(_) | SwarmError::ShardNotFound(_) => "not_found".into(),
        SwarmError::ProviderError { .. } => "upstream_error".into(),
        _ => "internal_error".into(),
    }
}

/// Extract the OpenAI-style `error.message` from a non-success chat
/// response body, falling back to the raw body text.
fn parse_error_message(bytes: &[u8]) -> String {
    if let Ok(v) = serde_json::from_slice::<serde_json::Value>(bytes) {
        if let Some(msg) = v
            .get("error")
            .and_then(|e| e.get("message"))
            .and_then(|m| m.as_str())
        {
            return msg.to_string();
        }
    }
    String::from_utf8_lossy(bytes).to_string()
}

/// Build the minimal response object emitted with the `response.created` /
/// `response.in_progress` events. It carries request metadata only — the
/// output array fills in during streaming, and `response.completed` carries
/// the full populated form. Thin wrapper around the shared
/// `super::build_response_skeleton` so all four "build a response object
/// from a request" sites stay in lockstep.
fn build_initial_response(
    req: &ResponsesRequest,
    response_id: &str,
    created_at: i64,
) -> ResponsesResponse {
    super::build_response_skeleton(req, response_id, created_at, ResponseStatus::InProgress)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_request() -> ResponsesRequest {
        ResponsesRequest {
            model: "test-model".into(),
            input: ResponsesInput::Text("hi".into()),
            instructions: None,
            previous_response_id: None,
            max_output_tokens: None,
            temperature: None,
            top_p: None,
            frequency_penalty: None,
            presence_penalty: None,
            stop: None,
            seed: None,
            user: None,
            metadata: None,
            stream: None,
            store: None,
            background: None,
            parallel_tool_calls: None,
            truncation: None,
            service_tier: None,
            modalities: None,
            include: None,
            tools: None,
            tool_choice: None,
            reasoning: None,
            text: None,
            conversation: None,
            context_management: None,
            extras: HashMap::new(),
        }
    }

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
    fn classify_error_code_buckets_swarmerror_variants() {
        use crate::error::SwarmError;
        use crate::types::{ModelId, ShardId};
        assert_eq!(
            classify_error_code(&ApiError(SwarmError::Validation("bad".into()))),
            "invalid_request_error"
        );
        assert_eq!(
            classify_error_code(&ApiError(SwarmError::ModelNotAvailable(ModelId(
                "x".into()
            )))),
            "not_found"
        );
        assert_eq!(
            classify_error_code(&ApiError(SwarmError::ShardNotFound(ShardId {
                model_id: ModelId("x".into()),
                index: 0,
            }))),
            "not_found"
        );
        assert_eq!(
            classify_error_code(&ApiError(SwarmError::ProviderError {
                status: 502,
                body: "upstream".into(),
            })),
            "upstream_error"
        );
        assert_eq!(
            classify_error_code(&ApiError(SwarmError::Internal("boom".into()))),
            "internal_error"
        );
    }

    #[test]
    fn parse_error_message_pulls_openai_shape() {
        let body = br#"{"error":{"message":"model not found","type":"invalid_request_error"}}"#;
        assert_eq!(parse_error_message(body), "model not found");
    }

    #[test]
    fn parse_error_message_falls_back_to_raw_body() {
        let body = b"plain text 503";
        assert_eq!(parse_error_message(body), "plain text 503");
    }

    #[test]
    fn parse_error_message_no_error_key_returns_raw() {
        let body = br#"{"foo":"bar"}"#;
        assert_eq!(parse_error_message(body), r#"{"foo":"bar"}"#);
    }

    #[test]
    fn build_failed_response_carries_error_and_status() {
        let req = test_request();
        let err = ResponseError {
            code: "invalid_request_error".into(),
            message: "bad model".into(),
            extras: HashMap::new(),
        };
        let resp = build_failed_response(&req, "resp_test123", 99, err);
        assert_eq!(resp.id, "resp_test123");
        assert_eq!(resp.status, ResponseStatus::Failed);
        assert_eq!(resp.created_at, 99);
        let e = resp.error.expect("error populated");
        assert_eq!(e.code, "invalid_request_error");
        assert_eq!(e.message, "bad model");
    }

    /// V1 contract: feeding the generator a chat_future that errors
    /// immediately still yields response.created + response.in_progress
    /// before response.failed. The point of the fix is that those two
    /// events do not depend on the future resolving.
    #[tokio::test]
    async fn early_error_still_emits_lifecycle_then_failed() {
        use crate::error::SwarmError;
        use futures::StreamExt;

        let req = test_request();
        let initial = build_initial_response(&req, "resp_x", 0);
        let chat_future =
            async { Err::<Response, ApiError>(ApiError(SwarmError::Validation("bad".into()))) };
        let stream = build_response_event_stream(
            chat_future,
            req,
            initial,
            "resp_x".into(),
            "msg_x".into(),
            0,
            None,
        );
        let events: Vec<_> = stream.collect().await;
        // Three events expected: created, in_progress, failed.
        assert_eq!(events.len(), 3, "expected 3 events, got {}", events.len());
        // We don't introspect axum::Event payloads here, but we can
        // confirm the stream finalizes (no extra events after failed)
        // and produces exactly the right count.
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
