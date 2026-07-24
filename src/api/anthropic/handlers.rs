use axum::response::IntoResponse;
use axum::Json;
use serde_json::{json, Value};

use crate::api::providers;
use crate::api::server::AppState;
use crate::error::ApiError;
use crate::inference::router::RouterCommand;
use crate::types::{ChatMessage, InferenceRequest, ModelId, SamplingParams};

use super::convert::map_finish_reason;
use super::sse::{
    build_anthropic_sse_response, send_sse_epilogue, send_sse_epilogue_with_stop,
    send_sse_preamble, AnthropicSseEvent,
};
use super::types::{
    AnthropicContent, AnthropicUsage, ContentBlock, MessagesRequest, MessagesResponse,
    ResponseContentBlock,
};
use crate::inference::router::InferenceOutput;

/// Build the final Anthropic `MessagesResponse` from a router-produced
/// `InferenceOutput`. Centralises the `map_finish_reason_with_match` +
/// `MessagesResponse::text_with_stop` sequence shared by both the
/// distributed-router and split-model non-streaming paths.
fn build_messages_response(
    request_id: String,
    model: String,
    output: InferenceOutput,
) -> MessagesResponse {
    let stop_reason = super::convert::map_finish_reason_with_match(
        &output.finish_reason,
        output.matched_stop_sequence.as_deref(),
    );
    MessagesResponse::text_with_stop(
        request_id,
        model,
        output.content,
        stop_reason,
        output.matched_stop_sequence,
        output.prompt_tokens,
        output.completion_tokens,
    )
}

/// Tool `type` strings on the Anthropic side that designate hosted server
/// tools (executed by Anthropic, not by the caller). These have no OpenAI
/// function-calling equivalent.
fn is_anthropic_server_tool(kind: &str) -> bool {
    matches!(
        kind,
        "web_search_20250305"
            | "web_search"
            | "code_execution_20250522"
            | "code_execution"
            | "computer_20241022"
            | "computer_20250124"
            | "computer"
            | "bash_20241022"
            | "bash_20250124"
            | "bash"
            | "text_editor_20241022"
            | "text_editor_20250124"
            | "text_editor"
    ) || kind.starts_with("web_search_")
        || kind.starts_with("code_execution_")
        || kind.starts_with("computer_")
        || kind.starts_with("bash_")
        || kind.starts_with("text_editor_")
        || kind.starts_with("tool_search_")
}

pub(super) async fn anthropic_non_stream(
    router_tx: tokio::sync::mpsc::Sender<RouterCommand>,
    messages: Vec<ChatMessage>,
    params: SamplingParams,
    request_id: String,
    model: String,
) -> Result<axum::response::Response, ApiError> {
    let inference_req =
        InferenceRequest::local(ModelId(model.clone()), messages, params, false, None, None);

    let output = crate::api::submit_to_router(&router_tx, inference_req).await?;
    let response = build_messages_response(request_id, model, output);
    Ok(Json(response).into_response())
}

/// Streaming inference via router, returning Anthropic SSE format.
pub(super) async fn anthropic_stream(
    router_tx: tokio::sync::mpsc::Sender<RouterCommand>,
    messages: Vec<ChatMessage>,
    params: SamplingParams,
    request_id: String,
    model: String,
) -> Result<axum::response::Response, ApiError> {
    let (result_rx, mut token_rx) = crate::api::openai::submit_stream_to_router(
        &router_tx,
        ModelId(model.clone()),
        messages,
        params,
        None,
        None,
        None, // anthropic /v1/messages doesn't have a cancel-by-token wire yet
    )
    .await?;

    let (sse_tx, sse_rx) = tokio::sync::mpsc::channel::<AnthropicSseEvent>(64);

    let rid = request_id.clone();
    let model = model.clone();

    tokio::spawn(async move {
        send_sse_preamble(&sse_tx, &rid, &model).await;

        // Stream tokens — count events as a fallback estimate
        let mut got_finish = false;
        let mut client_disconnected = false;
        let mut streamed_token_count = 0u32;
        let mut finish_stop_reason = String::new();
        let mut finish_matched_stop: Option<String> = None;
        loop {
            let event = tokio::select! {
                biased;
                // Client dropped the connection — cancel the instant the SSE
                // body's receiver is gone, not only when the next token's send
                // fails (a slow, mostly-CPU generation would otherwise keep a
                // worker busy for tens of seconds after the client left).
                _ = sse_tx.closed() => {
                    tracing::warn!(
                        token_count = streamed_token_count,
                        "Anthropic SSE client disconnected (connection closed) — cancelling pipeline"
                    );
                    client_disconnected = true;
                    break;
                }
                maybe = token_rx.recv() => match maybe {
                    Some(e) => e,
                    None => break,
                },
            };
            if let Some(ref reason) = event.finish_reason {
                got_finish = true;
                if !event.text.is_empty() {
                    streamed_token_count += 1;
                    let _ = sse_tx
                        .send(AnthropicSseEvent::ContentBlockDelta {
                            index: 0,
                            text: event.text,
                        })
                        .await;
                }
                finish_matched_stop = event.matched_stop_sequence.clone();
                finish_stop_reason = super::convert::map_finish_reason_with_match(
                    reason,
                    finish_matched_stop.as_deref(),
                )
                .into();
                break;
            }
            if !event.text.is_empty() {
                streamed_token_count += 1;
                if sse_tx
                    .send(AnthropicSseEvent::ContentBlockDelta {
                        index: 0,
                        text: event.text,
                    })
                    .await
                    .is_err()
                {
                    tracing::warn!(
                        token_count = streamed_token_count,
                        "Anthropic SSE client disconnected mid-stream — cancelling pipeline"
                    );
                    client_disconnected = true;
                    break;
                }
            }
        }

        // Client disconnected: drop token_rx to signal the pipeline to stop
        // generating, and skip the result_rx await so we don't block on a
        // now-useless pipeline holding the handler open.
        if client_disconnected {
            drop(token_rx);
            return;
        }

        // Get authoritative token count from the result when available
        let result = result_rx.await;
        if got_finish {
            let (output_tokens, matched_from_result) = match &result {
                Ok(Ok(output)) => (
                    output.completion_tokens,
                    output.matched_stop_sequence.clone(),
                ),
                _ => (streamed_token_count, None),
            };
            // Stream event takes precedence; result.matched_stop_sequence is
            // the authoritative fallback when the token stream didn't carry
            // it (e.g. distributed pipeline with no stop-string plumbing).
            let matched = finish_matched_stop.or(matched_from_result);
            send_sse_epilogue_with_stop(&sse_tx, finish_stop_reason, matched, output_tokens).await;
        } else {
            // Fallback: pipeline finished without streaming events
            match result {
                Ok(Ok(output)) => {
                    if !output.content.is_empty() {
                        let _ = sse_tx
                            .send(AnthropicSseEvent::ContentBlockDelta {
                                index: 0,
                                text: output.content,
                            })
                            .await;
                    }
                    let stop_reason = super::convert::map_finish_reason_with_match(
                        &output.finish_reason,
                        output.matched_stop_sequence.as_deref(),
                    )
                    .to_string();
                    send_sse_epilogue_with_stop(
                        &sse_tx,
                        stop_reason,
                        output.matched_stop_sequence,
                        output.completion_tokens,
                    )
                    .await;
                }
                _ => {
                    send_sse_epilogue(&sse_tx, "end_turn".into(), streamed_token_count).await;
                }
            }
        }
    });

    Ok(build_anthropic_sse_response(sse_rx))
}

/// Direct split-model non-streaming generation for Anthropic Messages API.
///
/// Bypasses the distributed pipeline and generates tokens directly via
/// SplitModel.forward() for maximum local inference speed.
pub(super) async fn anthropic_split_non_stream(
    state: &AppState,
    messages: &[ChatMessage],
    params: SamplingParams,
    request_id: String,
    model: String,
) -> Result<axum::response::Response, ApiError> {
    let requested_mid = crate::types::ModelId(model.clone());
    let output = crate::api::openai::run_split_generate(
        state,
        &requested_mid,
        messages,
        params.clone(),
        &request_id,
    )
    .await?;

    let response = build_messages_response(request_id, model, output);
    Ok(Json(response).into_response())
}

/// Direct split-model streaming generation for Anthropic Messages API.
///
/// Same fast path as anthropic_split_non_stream but streams tokens via SSE
/// in Anthropic's streaming format.
pub(super) async fn anthropic_split_stream(
    state: &AppState,
    messages: Vec<ChatMessage>,
    params: SamplingParams,
    request_id: String,
    model: String,
) -> Result<axum::response::Response, ApiError> {
    let (sse_tx, sse_rx) = tokio::sync::mpsc::channel::<AnthropicSseEvent>(64);

    let state = state.clone();
    let rid = request_id.clone();
    let model_for_lookup = model.clone();
    let model = model.clone();

    tokio::spawn(async move {
        send_sse_preamble(&sse_tx, &rid, &model).await;

        let requested_mid = crate::types::ModelId(model_for_lookup);
        let mut token_rx = match crate::api::openai::spawn_split_stream(
            &state,
            &requested_mid,
            &messages,
            params,
            &rid,
        ) {
            Some(rx) => rx,
            None => {
                send_sse_epilogue(&sse_tx, "end_turn".into(), 0).await;
                return;
            }
        };

        let mut total_output_tokens = 0u32;
        let mut stop_reason = "max_tokens".to_string();
        let mut matched_stop_sequence: Option<String> = None;

        while let Some(event) = token_rx.recv().await {
            if let Some(fr) = &event.finish_reason {
                stop_reason = super::convert::map_finish_reason_with_match(
                    fr,
                    event.matched_stop_sequence.as_deref(),
                )
                .to_string();
                matched_stop_sequence = event.matched_stop_sequence.clone();
                break;
            }
            total_output_tokens += 1;
            if sse_tx
                .send(AnthropicSseEvent::ContentBlockDelta {
                    index: 0,
                    text: event.text,
                })
                .await
                .is_err()
            {
                return; // Client disconnected
            }
        }

        send_sse_epilogue_with_stop(
            &sse_tx,
            stop_reason,
            matched_stop_sequence,
            total_output_tokens,
        )
        .await;
    });

    Ok(build_anthropic_sse_response(sse_rx))
}

/// Translate an Anthropic Messages API request to OpenAI chat completions
/// format and proxy it to a non-Anthropic cloud provider. Tools, tool_use,
/// tool_result, images, and system messages survive the round-trip; the
/// upstream's `tool_calls` come back as `tool_use` content blocks (and
/// `stop_reason: "tool_use"`) so multi-turn function-calling flows work.
pub(super) async fn anthropic_to_openai_proxy(
    req: &MessagesRequest,
    base_url: &str,
    api_key: &str,
) -> Result<axum::response::Response, ApiError> {
    if let Some(ref tools) = req.tools {
        for tool in tools {
            let kind = tool.get("type").and_then(|v| v.as_str()).unwrap_or("");
            if is_anthropic_server_tool(kind) {
                return Err(ApiError(crate::error::SwarmError::Validation(format!(
                    "Tool type `{kind}` is an Anthropic-hosted server tool and cannot be \
                     translated to OpenAI function-calling. Route this request to an \
                     Anthropic-compatible provider, or remove the server tool."
                ))));
            }
        }
    }

    let body = build_openai_request_body(req)?;

    let url = format!("{}/chat/completions", base_url.trim_end_matches('/'));
    let client = providers::get_provider_client();

    let resp = client
        .post(&url)
        .header("Authorization", format!("Bearer {api_key}"))
        .header("Content-Type", "application/json")
        .json(&body)
        .send()
        .await
        .map_err(|e| {
            ApiError(crate::error::SwarmError::Network(format!(
                "Cloud provider proxy failed: {e}"
            )))
        })?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        return Err(crate::api::providers::extract_provider_error(
            &body,
            status,
            "anthropic-proxy",
            crate::api::providers::ANTHROPIC_ERROR_KEYS,
        ));
    }

    if req.stream {
        let (sse_tx, sse_rx) = tokio::sync::mpsc::channel::<AnthropicSseEvent>(64);
        let model_clone = req.model.clone();

        tokio::spawn(async move {
            stream_openai_to_anthropic(resp, sse_tx, model_clone).await;
        });

        return Ok(build_anthropic_sse_response(sse_rx));
    }

    // Non-streaming: read full JSON response.
    let openai_resp: Value = resp.json().await.map_err(|e| {
        ApiError(crate::error::SwarmError::ProviderError {
            status: 502,
            body: format!("Cloud provider returned malformed JSON: {e}"),
        })
    })?;

    let response = openai_response_to_anthropic(&openai_resp, &req.model);
    Ok(Json(response).into_response())
}

/// Build the OpenAI Chat Completions request body from an Anthropic
/// Messages request, preserving tool_use / tool_result / image blocks
/// and translating tool definitions + tool_choice.
fn build_openai_request_body(req: &MessagesRequest) -> Result<Value, ApiError> {
    let messages = anthropic_messages_to_openai(req)?;
    let mut body = json!({
        "model": req.model,
        "messages": messages,
        "max_tokens": req.max_tokens,
        "stream": req.stream,
    });
    if let Some(t) = req.temperature {
        body["temperature"] = json!(t);
    }
    if let Some(tp) = req.top_p {
        body["top_p"] = json!(tp);
    }
    if let Some(ref stops) = req.stop_sequences {
        body["stop"] = json!(stops);
    }
    if let Some(ref tools) = req.tools {
        let openai_tools = anthropic_tools_to_openai(tools)?;
        if !openai_tools.is_empty() {
            body["tools"] = Value::Array(openai_tools);
        }
    }
    if let Some(ref tc) = req.tool_choice {
        if let Some(translated) = anthropic_tool_choice_to_openai(tc) {
            body["tool_choice"] = translated;
        }
    }
    Ok(body)
}

/// Walk the Anthropic messages array and emit OpenAI-shaped messages.
/// `system` is hoisted to a leading `system` role message; assistant
/// `tool_use` blocks become `tool_calls` on the assistant message; user
/// `tool_result` blocks each become their own `role: "tool"` message
/// (OpenAI requires one tool message per tool_call_id).
fn anthropic_messages_to_openai(req: &MessagesRequest) -> Result<Vec<Value>, ApiError> {
    let mut out = Vec::new();

    if let Some(ref system) = req.system {
        let text = system.to_plain_text();
        if !text.is_empty() {
            out.push(json!({"role": "system", "content": text}));
        }
    }

    for msg in &req.messages {
        let role = msg.role.as_str();
        match &msg.content {
            AnthropicContent::Text(s) => {
                out.push(json!({"role": role, "content": s}));
            }
            AnthropicContent::Blocks(blocks) => {
                out.extend(translate_block_message(role, blocks));
            }
        }
    }

    Ok(out)
}

/// Translate a single Anthropic message (role + content blocks) into one
/// or more OpenAI-shaped messages. A user message containing N
/// `tool_result` blocks expands to N `tool` messages plus optionally a
/// trailing `user` message for any text/image blocks.
fn translate_block_message(role: &str, blocks: &[ContentBlock]) -> Vec<Value> {
    if role == "assistant" {
        let mut text_buf = String::new();
        let mut tool_calls: Vec<Value> = Vec::new();
        for block in blocks {
            match block {
                ContentBlock::Text { text } => text_buf.push_str(text),
                ContentBlock::ToolUse { id, name, input } => {
                    tool_calls.push(json!({
                        "id": id,
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": serde_json::to_string(input)
                                .unwrap_or_else(|_| "{}".into()),
                        }
                    }));
                }
                // Thinking, redacted thinking, server-tool variants, and
                // citation blocks have no OpenAI assistant-side equivalent.
                _ => {}
            }
        }
        let mut msg = serde_json::Map::new();
        msg.insert("role".into(), Value::String("assistant".into()));
        if !tool_calls.is_empty() {
            // OpenAI accepts `content: null` when tool_calls are present.
            // Emit text content alongside tool_calls only if non-empty.
            if text_buf.is_empty() {
                msg.insert("content".into(), Value::Null);
            } else {
                msg.insert("content".into(), Value::String(text_buf));
            }
            msg.insert("tool_calls".into(), Value::Array(tool_calls));
        } else {
            msg.insert("content".into(), Value::String(text_buf));
        }
        return vec![Value::Object(msg)];
    }

    // User role (or fallback for unknown roles): split tool_result blocks
    // out as their own role:"tool" messages.
    let mut out = Vec::new();
    let mut user_parts: Vec<Value> = Vec::new();
    let mut user_has_image = false;

    let flush_user = |out: &mut Vec<Value>, parts: Vec<Value>, has_image: bool| {
        if parts.is_empty() {
            return;
        }
        if !has_image {
            // Collapse text-only parts into a single string content.
            let combined = parts
                .iter()
                .filter_map(|p| p.get("text").and_then(|v| v.as_str()))
                .collect::<Vec<_>>()
                .join("\n");
            out.push(json!({"role": "user", "content": combined}));
        } else {
            out.push(json!({"role": "user", "content": parts}));
        }
    };

    for block in blocks {
        match block {
            ContentBlock::Text { text } => {
                user_parts.push(json!({"type": "text", "text": text}));
            }
            ContentBlock::Image { source } => {
                if let Some(url) = anthropic_image_source_to_url(source) {
                    user_has_image = true;
                    user_parts.push(json!({"type": "image_url", "image_url": {"url": url}}));
                }
            }
            ContentBlock::ToolResult {
                tool_use_id,
                content,
                is_error: _,
            } => {
                // Flush any pending user content first so message ordering
                // matches Anthropic's intent.
                let parts = std::mem::take(&mut user_parts);
                let has_image = std::mem::replace(&mut user_has_image, false);
                flush_user(&mut out, parts, has_image);

                let content_str = tool_result_to_openai_string(content.as_ref());
                out.push(json!({
                    "role": "tool",
                    "tool_call_id": tool_use_id,
                    "content": content_str,
                }));
            }
            ContentBlock::ToolUse { name, input, .. } => {
                // Anthropic spec only allows tool_use on assistant role,
                // but be lenient for echoed conversation histories that
                // misroute the block — render as text so OpenAI sees it.
                user_parts.push(json!({
                    "type": "text",
                    "text": format!("[Tool call: {name}({input})]"),
                }));
            }
            ContentBlock::Thinking { thinking } => {
                user_parts.push(json!({
                    "type": "text",
                    "text": format!("<thinking>{thinking}</thinking>"),
                }));
            }
            // Server-tool result/use blocks and citation blocks: no
            // OpenAI equivalent; drop. The proxy never receives these on
            // the user side under normal Claude Code traffic.
            _ => {}
        }
    }

    flush_user(&mut out, user_parts, user_has_image);
    out
}

/// Map an Anthropic image `source` object to an OpenAI `image_url.url`
/// string. Returns `None` for unsupported source shapes.
fn anthropic_image_source_to_url(source: &Value) -> Option<String> {
    let kind = source.get("type").and_then(|v| v.as_str()).unwrap_or("");
    match kind {
        "base64" => {
            let media_type = source
                .get("media_type")
                .and_then(|v| v.as_str())
                .unwrap_or("image/png");
            let data = source.get("data").and_then(|v| v.as_str()).unwrap_or("");
            Some(format!("data:{media_type};base64,{data}"))
        }
        "url" => source.get("url").and_then(|v| v.as_str()).map(String::from),
        _ => None,
    }
}

/// Anthropic tool_result `content` is either a string or an array of
/// content blocks (text/image). OpenAI tool messages take a string;
/// flatten by extracting text and stubbing images.
fn tool_result_to_openai_string(content: Option<&Value>) -> String {
    match content {
        None => String::new(),
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(arr)) => arr
            .iter()
            .filter_map(|item| {
                let kind = item.get("type").and_then(|v| v.as_str()).unwrap_or("");
                match kind {
                    "text" => item.get("text").and_then(|v| v.as_str()).map(String::from),
                    "image" => Some("[image]".into()),
                    _ => None,
                }
            })
            .collect::<Vec<_>>()
            .join("\n"),
        Some(other) => other.to_string(),
    }
}

/// Translate Anthropic tool definitions (`{name, description?, input_schema}`)
/// to OpenAI tool definitions (`{type: "function", function: {...}}`).
fn anthropic_tools_to_openai(tools: &[Value]) -> Result<Vec<Value>, ApiError> {
    let mut out = Vec::with_capacity(tools.len());
    for tool in tools {
        let name = tool.get("name").and_then(|v| v.as_str()).unwrap_or("");
        if name.is_empty() {
            return Err(ApiError(crate::error::SwarmError::Validation(
                "Tool definition missing required `name` field".into(),
            )));
        }
        let mut function = serde_json::Map::new();
        function.insert("name".into(), Value::String(name.into()));
        if let Some(d) = tool.get("description").and_then(|v| v.as_str()) {
            function.insert("description".into(), Value::String(d.into()));
        }
        let schema = tool
            .get("input_schema")
            .cloned()
            .unwrap_or_else(|| json!({"type": "object", "properties": {}}));
        function.insert("parameters".into(), schema);
        out.push(json!({"type": "function", "function": Value::Object(function)}));
    }
    Ok(out)
}

/// Map Anthropic tool_choice (`{type: "auto" | "any" | "none" | "tool", name?}`)
/// to OpenAI tool_choice (string mode or `{type: "function", function: {name}}`).
/// Anthropic's `any` (force a tool, any tool) maps to OpenAI's `required`.
fn anthropic_tool_choice_to_openai(tc: &Value) -> Option<Value> {
    let kind = tc.get("type").and_then(|v| v.as_str())?;
    match kind {
        "auto" => Some(json!("auto")),
        "any" => Some(json!("required")),
        "none" => Some(json!("none")),
        "tool" => {
            let name = tc.get("name").and_then(|v| v.as_str())?;
            Some(json!({"type": "function", "function": {"name": name}}))
        }
        _ => None,
    }
}

/// Translate an OpenAI Chat Completions response (non-streaming) into
/// an Anthropic MessagesResponse. `tool_calls` become `tool_use` content
/// blocks; `finish_reason: "tool_calls"` maps to `stop_reason: "tool_use"`.
fn openai_response_to_anthropic(openai_resp: &Value, model: &str) -> MessagesResponse {
    let choice = openai_resp["choices"]
        .as_array()
        .and_then(|c| c.first())
        .unwrap_or(&Value::Null);
    let message = &choice["message"];
    let finish = choice["finish_reason"].as_str().unwrap_or("stop");

    let mut content_blocks: Vec<ResponseContentBlock> = Vec::new();
    if let Some(text) = message.get("content").and_then(|v| v.as_str()) {
        if !text.is_empty() {
            content_blocks.push(ResponseContentBlock::Text { text: text.into() });
        }
    }
    if let Some(tool_calls) = message.get("tool_calls").and_then(|v| v.as_array()) {
        for tc in tool_calls {
            let id = tc
                .get("id")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let function = tc.get("function").cloned().unwrap_or(Value::Null);
            let name = function
                .get("name")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let args_str = function
                .get("arguments")
                .and_then(|v| v.as_str())
                .unwrap_or("{}");
            let input: Value = serde_json::from_str(args_str)
                .unwrap_or_else(|_| json!({"_raw_arguments": args_str}));
            content_blocks.push(ResponseContentBlock::ToolUse { id, name, input });
        }
    }
    if content_blocks.is_empty() {
        // Anthropic always returns at least one block; emit empty text.
        content_blocks.push(ResponseContentBlock::Text {
            text: String::new(),
        });
    }

    let stop_reason = match finish {
        "tool_calls" => "tool_use",
        "length" => "max_tokens",
        "stop" => "end_turn",
        other => map_finish_reason(other),
    };

    let input_tokens = openai_resp["usage"]["prompt_tokens"].as_u64().unwrap_or(0) as u32;
    let output_tokens = openai_resp["usage"]["completion_tokens"]
        .as_u64()
        .unwrap_or(0) as u32;

    MessagesResponse {
        id: format!("msg_{}", uuid::Uuid::new_v4().simple()),
        response_type: "message",
        role: "assistant",
        content: content_blocks,
        model: model.to_string(),
        stop_reason: Some(stop_reason.into()),
        stop_sequence: None,
        usage: AnthropicUsage {
            input_tokens,
            output_tokens,
            ..Default::default()
        },
    }
}

/// Stream an OpenAI Chat Completions SSE response, translating each
/// chunk into Anthropic SSE events. Tracks tool_calls by upstream index
/// and opens an Anthropic content block (`tool_use`) per call.
async fn stream_openai_to_anthropic(
    mut resp: reqwest::Response,
    sse_tx: tokio::sync::mpsc::Sender<AnthropicSseEvent>,
    model: String,
) {
    use std::collections::HashMap;

    let msg_id = format!("msg_{}", uuid::Uuid::new_v4().simple());
    let _ = sse_tx
        .send(AnthropicSseEvent::MessageStart { id: msg_id, model })
        .await;

    let mut buffer = String::new();
    let mut output_tokens: u32 = 0;
    let mut text_block_open = false;
    // OpenAI tool_call upstream-index → (anthropic content block index)
    let mut tool_blocks: HashMap<u64, u32> = HashMap::new();
    // anthropic content block indices currently open and in need of stop
    let mut open_block_indices: Vec<u32> = Vec::new();
    // Allocator for anthropic content block indices. Index 0 is reserved
    // for the text block; tool blocks start at 1.
    let mut next_block_idx: u32 = 1;
    let mut finish_reason: Option<String> = None;

    'outer: while let Ok(Some(chunk)) = resp.chunk().await {
        buffer.push_str(&String::from_utf8_lossy(&chunk));

        while let Some(line_end) = buffer.find('\n') {
            let line = buffer[..line_end].trim().to_string();
            buffer = buffer[line_end + 1..].to_string();

            let Some(data) = line.strip_prefix("data: ") else {
                continue;
            };
            if data == "[DONE]" {
                break 'outer;
            }
            let Ok(event) = serde_json::from_str::<Value>(data) else {
                continue;
            };

            let choice = &event["choices"][0];
            if let Some(fr) = choice.get("finish_reason").and_then(|v| v.as_str()) {
                if !fr.is_empty() {
                    finish_reason = Some(fr.to_string());
                }
            }

            let delta = &choice["delta"];

            // Text content delta.
            if let Some(text) = delta.get("content").and_then(|v| v.as_str()) {
                if !text.is_empty() {
                    if !text_block_open {
                        if sse_tx
                            .send(AnthropicSseEvent::ContentBlockStart { index: 0 })
                            .await
                            .is_err()
                        {
                            return;
                        }
                        text_block_open = true;
                        open_block_indices.push(0);
                    }
                    output_tokens += 1;
                    if sse_tx
                        .send(AnthropicSseEvent::ContentBlockDelta {
                            index: 0,
                            text: text.to_string(),
                        })
                        .await
                        .is_err()
                    {
                        return;
                    }
                }
            }

            // Tool-call deltas.
            if let Some(tcs) = delta.get("tool_calls").and_then(|v| v.as_array()) {
                for tc in tcs {
                    let upstream_idx = tc.get("index").and_then(|v| v.as_u64()).unwrap_or(0);
                    let function = tc.get("function").cloned().unwrap_or(Value::Null);
                    let block_idx = match tool_blocks.entry(upstream_idx) {
                        std::collections::hash_map::Entry::Occupied(e) => *e.get(),
                        std::collections::hash_map::Entry::Vacant(e) => {
                            // First chunk for this tool call: must carry id
                            // + function.name. Open a new Anthropic block.
                            let id = tc
                                .get("id")
                                .and_then(|v| v.as_str())
                                .unwrap_or("")
                                .to_string();
                            let name = function
                                .get("name")
                                .and_then(|v| v.as_str())
                                .unwrap_or("")
                                .to_string();
                            let idx = next_block_idx;
                            next_block_idx += 1;
                            e.insert(idx);
                            open_block_indices.push(idx);
                            if sse_tx
                                .send(AnthropicSseEvent::ContentBlockStartToolUse {
                                    index: idx,
                                    id,
                                    name,
                                })
                                .await
                                .is_err()
                            {
                                return;
                            }
                            idx
                        }
                    };
                    if let Some(args) = function.get("arguments").and_then(|v| v.as_str()) {
                        if !args.is_empty()
                            && sse_tx
                                .send(AnthropicSseEvent::ContentBlockInputJsonDelta {
                                    index: block_idx,
                                    partial_json: args.to_string(),
                                })
                                .await
                                .is_err()
                        {
                            return;
                        }
                    }
                }
            }
        }
    }

    // Close every open content block in order.
    for idx in open_block_indices {
        let _ = sse_tx
            .send(AnthropicSseEvent::ContentBlockStop { index: idx })
            .await;
    }

    let stop_reason = match finish_reason.as_deref() {
        Some("tool_calls") => "tool_use",
        Some("length") => "max_tokens",
        Some("stop") | None => "end_turn",
        Some(other) => map_finish_reason(other),
    };
    send_sse_epilogue(&sse_tx, stop_reason.into(), output_tokens).await;
}

#[cfg(test)]
mod tests {
    use super::super::types::{AnthropicMessage, SystemContent};
    use super::*;

    fn base_req() -> MessagesRequest {
        MessagesRequest {
            model: "deepseek-chat".into(),
            max_tokens: 100,
            messages: Vec::new(),
            system: None,
            stream: false,
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: None,
            tools: None,
            tool_choice: None,
            metadata: None,
            thinking: None,
            extras: std::collections::HashMap::new(),
        }
    }

    #[test]
    fn system_prompt_becomes_leading_system_message() {
        let mut req = base_req();
        req.system = Some(SystemContent::Text("Be terse.".into()));
        req.messages = vec![AnthropicMessage {
            role: "user".into(),
            content: AnthropicContent::Text("hi".into()),
        }];
        let body = build_openai_request_body(&req).unwrap_or_else(|_| panic!("translation failed"));
        let messages = body["messages"].as_array().unwrap();
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0]["role"], "system");
        assert_eq!(messages[0]["content"], "Be terse.");
        assert_eq!(messages[1]["role"], "user");
        assert_eq!(messages[1]["content"], "hi");
    }

    #[test]
    fn assistant_tool_use_block_becomes_tool_calls() {
        let mut req = base_req();
        req.messages = vec![
            AnthropicMessage {
                role: "user".into(),
                content: AnthropicContent::Text("call foo".into()),
            },
            AnthropicMessage {
                role: "assistant".into(),
                content: AnthropicContent::Blocks(vec![
                    ContentBlock::Text {
                        text: "Sure, calling.".into(),
                    },
                    ContentBlock::ToolUse {
                        id: "tu_1".into(),
                        name: "lookup".into(),
                        input: json!({"q": "rust"}),
                    },
                ]),
            },
        ];
        let body = build_openai_request_body(&req).unwrap_or_else(|_| panic!("translation failed"));
        let messages = body["messages"].as_array().unwrap();
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[1]["role"], "assistant");
        assert_eq!(messages[1]["content"], "Sure, calling.");
        let tcs = messages[1]["tool_calls"].as_array().unwrap();
        assert_eq!(tcs.len(), 1);
        assert_eq!(tcs[0]["id"], "tu_1");
        assert_eq!(tcs[0]["type"], "function");
        assert_eq!(tcs[0]["function"]["name"], "lookup");
        // arguments must be a JSON-encoded string, not an object.
        let args_str = tcs[0]["function"]["arguments"].as_str().unwrap();
        let parsed: Value = serde_json::from_str(args_str).unwrap();
        assert_eq!(parsed["q"], "rust");
    }

    #[test]
    fn assistant_tool_use_only_has_null_content() {
        let mut req = base_req();
        req.messages = vec![AnthropicMessage {
            role: "assistant".into(),
            content: AnthropicContent::Blocks(vec![ContentBlock::ToolUse {
                id: "tu_1".into(),
                name: "lookup".into(),
                input: json!({}),
            }]),
        }];
        let body = build_openai_request_body(&req).unwrap_or_else(|_| panic!("translation failed"));
        let messages = body["messages"].as_array().unwrap();
        assert_eq!(messages[0]["content"], Value::Null);
        assert_eq!(messages[0]["tool_calls"].as_array().unwrap().len(), 1);
    }

    #[test]
    fn user_tool_result_block_becomes_role_tool_message() {
        let mut req = base_req();
        req.messages = vec![AnthropicMessage {
            role: "user".into(),
            content: AnthropicContent::Blocks(vec![
                ContentBlock::ToolResult {
                    tool_use_id: "tu_1".into(),
                    content: Some(json!("result-text")),
                    is_error: None,
                },
                ContentBlock::Text {
                    text: "now answer".into(),
                },
            ]),
        }];
        let body = build_openai_request_body(&req).unwrap_or_else(|_| panic!("translation failed"));
        let messages = body["messages"].as_array().unwrap();
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0]["role"], "tool");
        assert_eq!(messages[0]["tool_call_id"], "tu_1");
        assert_eq!(messages[0]["content"], "result-text");
        assert_eq!(messages[1]["role"], "user");
        assert_eq!(messages[1]["content"], "now answer");
    }

    #[test]
    fn tool_result_block_array_content_flattens_to_string() {
        let mut req = base_req();
        req.messages = vec![AnthropicMessage {
            role: "user".into(),
            content: AnthropicContent::Blocks(vec![ContentBlock::ToolResult {
                tool_use_id: "tu_2".into(),
                content: Some(json!([
                    {"type": "text", "text": "line1"},
                    {"type": "text", "text": "line2"},
                ])),
                is_error: None,
            }]),
        }];
        let body = build_openai_request_body(&req).unwrap_or_else(|_| panic!("translation failed"));
        let messages = body["messages"].as_array().unwrap();
        assert_eq!(messages[0]["content"], "line1\nline2");
    }

    #[test]
    fn user_image_block_becomes_image_url_content_part() {
        let mut req = base_req();
        req.messages = vec![AnthropicMessage {
            role: "user".into(),
            content: AnthropicContent::Blocks(vec![
                ContentBlock::Text {
                    text: "what is this?".into(),
                },
                ContentBlock::Image {
                    source: json!({
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": "iVBORw==",
                    }),
                },
            ]),
        }];
        let body = build_openai_request_body(&req).unwrap_or_else(|_| panic!("translation failed"));
        let parts = body["messages"][0]["content"].as_array().unwrap();
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0]["type"], "text");
        assert_eq!(parts[1]["type"], "image_url");
        assert_eq!(
            parts[1]["image_url"]["url"],
            "data:image/jpeg;base64,iVBORw=="
        );
    }

    #[test]
    fn tools_translate_to_openai_function_shape() {
        let mut req = base_req();
        req.tools = Some(vec![json!({
            "name": "lookup",
            "description": "Find a thing",
            "input_schema": {"type": "object", "properties": {"q": {"type": "string"}}},
        })]);
        req.messages = vec![AnthropicMessage {
            role: "user".into(),
            content: AnthropicContent::Text("hi".into()),
        }];
        let body = build_openai_request_body(&req).unwrap_or_else(|_| panic!("translation failed"));
        let tools = body["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0]["type"], "function");
        assert_eq!(tools[0]["function"]["name"], "lookup");
        assert_eq!(tools[0]["function"]["description"], "Find a thing");
        assert_eq!(tools[0]["function"]["parameters"]["type"], "object");
    }

    #[test]
    fn tool_choice_any_maps_to_required() {
        let mut req = base_req();
        req.tool_choice = Some(json!({"type": "any"}));
        req.messages = vec![AnthropicMessage {
            role: "user".into(),
            content: AnthropicContent::Text("hi".into()),
        }];
        let body = build_openai_request_body(&req).unwrap_or_else(|_| panic!("translation failed"));
        assert_eq!(body["tool_choice"], "required");
    }

    #[test]
    fn tool_choice_named_tool_maps_to_function_object() {
        let mut req = base_req();
        req.tool_choice = Some(json!({"type": "tool", "name": "lookup"}));
        req.messages = vec![AnthropicMessage {
            role: "user".into(),
            content: AnthropicContent::Text("hi".into()),
        }];
        let body = build_openai_request_body(&req).unwrap_or_else(|_| panic!("translation failed"));
        assert_eq!(body["tool_choice"]["type"], "function");
        assert_eq!(body["tool_choice"]["function"]["name"], "lookup");
    }

    #[test]
    fn server_tool_in_tools_is_rejected_via_proxy_check() {
        // The handler-level guard runs in `anthropic_to_openai_proxy` before
        // build_openai_request_body. Verify the detection function is the
        // canonical check.
        assert!(is_anthropic_server_tool("web_search_20250305"));
    }

    #[test]
    fn openai_response_with_tool_calls_becomes_tool_use_blocks() {
        let upstream = json!({
            "choices": [{
                "message": {
                    "content": "Looking that up.",
                    "tool_calls": [{
                        "id": "call_abc",
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "arguments": "{\"q\":\"hello\"}"
                        }
                    }]
                },
                "finish_reason": "tool_calls"
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20}
        });
        let resp = openai_response_to_anthropic(&upstream, "deepseek-chat");
        assert_eq!(resp.stop_reason.as_deref(), Some("tool_use"));
        assert_eq!(resp.content.len(), 2);
        match &resp.content[0] {
            ResponseContentBlock::Text { text } => assert_eq!(text, "Looking that up."),
            _ => panic!("expected text block first"),
        }
        match &resp.content[1] {
            ResponseContentBlock::ToolUse { id, name, input } => {
                assert_eq!(id, "call_abc");
                assert_eq!(name, "lookup");
                assert_eq!(input["q"], "hello");
            }
            _ => panic!("expected tool_use block"),
        }
        assert_eq!(resp.usage.input_tokens, 10);
        assert_eq!(resp.usage.output_tokens, 20);
    }

    #[test]
    fn openai_response_text_only_maps_finish_stop_to_end_turn() {
        let upstream = json!({
            "choices": [{
                "message": {"content": "hello"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1}
        });
        let resp = openai_response_to_anthropic(&upstream, "deepseek-chat");
        assert_eq!(resp.stop_reason.as_deref(), Some("end_turn"));
        assert_eq!(resp.content.len(), 1);
    }

    #[test]
    fn openai_response_finish_length_maps_to_max_tokens() {
        let upstream = json!({
            "choices": [{
                "message": {"content": "partial"},
                "finish_reason": "length"
            }],
            "usage": {"prompt_tokens": 1, "completion_tokens": 100}
        });
        let resp = openai_response_to_anthropic(&upstream, "deepseek-chat");
        assert_eq!(resp.stop_reason.as_deref(), Some("max_tokens"));
    }

    #[test]
    fn empty_response_still_emits_one_block() {
        // Anthropic clients expect content to be non-empty.
        let upstream = json!({
            "choices": [{"message": {}, "finish_reason": "stop"}],
            "usage": {}
        });
        let resp = openai_response_to_anthropic(&upstream, "deepseek-chat");
        assert_eq!(resp.content.len(), 1);
        match &resp.content[0] {
            ResponseContentBlock::Text { text } => assert!(text.is_empty()),
            _ => panic!("expected empty text block"),
        }
    }

    #[test]
    fn detects_anthropic_server_tools() {
        // Exact known types.
        assert!(is_anthropic_server_tool("web_search_20250305"));
        assert!(is_anthropic_server_tool("code_execution_20250522"));
        assert!(is_anthropic_server_tool("bash_20250124"));
        assert!(is_anthropic_server_tool("text_editor_20250124"));
        assert!(is_anthropic_server_tool("computer_20250124"));

        // Future-dated variants via prefix match.
        assert!(is_anthropic_server_tool("web_search_99991231"));
        assert!(is_anthropic_server_tool("tool_search_20260101"));

        // Caller-defined function tools must NOT trip the filter — these
        // are legitimately translatable to OpenAI function-calling.
        assert!(!is_anthropic_server_tool("function"));
        assert!(!is_anthropic_server_tool("custom"));
        assert!(!is_anthropic_server_tool(""));
        assert!(!is_anthropic_server_tool("my_tool"));
    }
}
