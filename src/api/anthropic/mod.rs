use axum::extract::State;
use axum::response::IntoResponse;
use axum::Json;
use serde::Serialize;

use crate::api::providers;
use crate::api::server::AppState;
use crate::error::ApiError;
use crate::inference::chat_template;
use crate::inference::router::RouterCommand;
use crate::types::{ChatMessage, InferenceRequest, ModelId, Role, SamplingParams};

mod convert;
mod sse;
mod types;

// Re-export public wire-format types so they remain part of the crate's
// external surface (mirrors the pre-split `pub enum` visibility).
pub use types::{AnthropicUsage, MessagesResponse, ResponseContentBlock};

use convert::{
    is_connectivity_probe, map_finish_reason, resolve_model, to_internal_messages,
    to_sampling_params,
};
#[cfg(test)]
use sse::serialize_anthropic_event;
use sse::{build_anthropic_sse_response, send_sse_epilogue, send_sse_preamble, AnthropicSseEvent};
use types::{
    AnthropicContent, AnthropicMessage, ContentBlock, MessagesRequest, SystemBlock, SystemContent,
};

// ---- Handler ----

/// POST /v1/messages — Anthropic Messages API endpoint.
pub async fn messages(
    State(state): State<AppState>,
    crate::api::server::JsonBody(req): crate::api::server::JsonBody<MessagesRequest>,
) -> Result<axum::response::Response, ApiError> {
    super::validate_common_params(
        req.model.len(),
        req.messages.len(),
        req.temperature.unwrap_or(1.0).into(),
    )?;

    if let Some(ref stops) = req.stop_sequences {
        super::validate_stop_sequences(stops)?;
    }

    if let Some(ref tools) = req.tools {
        super::validate_tools(
            tools,
            |t| t.get("name").and_then(|v| v.as_str()),
            |t| t.get("description").and_then(|v| v.as_str()),
            |t| t.get("input_schema").map(|s| s.to_string().len()),
        )?;
    }

    // SEC: Cap individual message content size and total prompt size
    super::validate_content_size(req.messages.iter().map(|msg| {
        match &msg.content {
            AnthropicContent::Text(s) => s.len(),
            AnthropicContent::Blocks(blocks) => blocks
                .iter()
                .map(|b| match b {
                    ContentBlock::Text { text } => text.len(),
                    ContentBlock::Image { source } => source.to_string().len(),
                    ContentBlock::ToolUse {
                        input, name, id, ..
                    } => name.len() + id.len() + input.to_string().len(),
                    ContentBlock::ToolResult { content, .. } => {
                        content.as_ref().map(|c| c.to_string().len()).unwrap_or(0)
                    }
                    ContentBlock::Thinking { thinking } => thinking.len(),
                    ContentBlock::RedactedThinking { data } => data.len(),
                })
                .sum(),
        }
    }))?;

    let request_id = format!("msg_{}", uuid::Uuid::new_v4().simple());
    let model = resolve_model(&req.model).to_string();

    // Track requests made by this node
    super::increment_requests_made(&state.shared_state);

    tracing::info!(
        request_id = %request_id,
        model = %model,
        messages = req.messages.len(),
        stream = req.stream,
        max_tokens = req.max_tokens,
        "DIAG: anthropic messages request"
    );

    // Fast-path: connectivity probes (Claude Code sends max_tokens=1 pings)
    if is_connectivity_probe(&req) {
        tracing::debug!(request_id = %request_id, "DIAG: anthropic connectivity probe — fast path");
        let response = MessagesResponse::text(request_id, model, "ok".into(), "end_turn", 1, 1);
        return Ok(Json(response).into_response());
    }

    let internal_messages = to_internal_messages(&req);
    let sampling_params = to_sampling_params(&req);

    // Resolve model alias (display name → registry ID, "auto" → first available).
    let model = crate::api::openai::resolve_model_for_inference(&state, &model).await;

    // Check if network has all shards for this model
    let network_available = crate::api::openai::all_shards_available(&state, &model);

    // Fast path: if we have a complete local split model for the REQUESTED model, generate directly.
    // Match by model ID — not just "any loaded model" (compare sends different model IDs).
    let requested_mid = crate::types::ModelId(model.clone());
    let has_local_split_model = state.shared_state.has_complete_split_model(&requested_mid);

    tracing::debug!(
        request_id = %request_id,
        has_local_split_model,
        network_available,
        "DIAG: anthropic inference path resolution"
    );

    if has_local_split_model {
        if req.stream {
            return anthropic_split_stream(
                &state,
                internal_messages,
                sampling_params,
                request_id,
                model,
            )
            .await;
        } else {
            return anthropic_split_non_stream(
                &state,
                &internal_messages,
                sampling_params,
                request_id,
                model,
            )
            .await;
        }
    }

    // Check if the requested model is actually available (locally or on network)
    let model_locally_available = {
        let info = state.shared_state.loaded_model_info.read().await;
        info.as_ref()
            .map(|i| crate::api::openai::model_matches_loaded(&state, &i.name, &model))
            .unwrap_or(false)
    };

    if model_locally_available || network_available {
        // Distributed inference available
        if let Some(router_tx) = &state.router_tx {
            if req.stream {
                return anthropic_stream(
                    router_tx.clone(),
                    &state,
                    &req,
                    internal_messages,
                    sampling_params,
                    request_id,
                    model,
                )
                .await;
            } else {
                return anthropic_non_stream(
                    router_tx.clone(),
                    &req,
                    internal_messages,
                    sampling_params,
                    request_id,
                    model,
                )
                .await;
            }
        }

        // Direct executor fallback (single-node, no router)
        if model_locally_available {
            let (tmpl, bos, eos) = super::resolve_chat_template(&state, &model).await;
            let prompt =
                chat_template::build_prompt(&internal_messages, tmpl.as_deref(), &bos, &eos);

            let mut executor = state.executor.lock().await;
            let (content, result) = executor
                .generate(&prompt, &sampling_params)
                .map_err(ApiError)?;

            let response = MessagesResponse::text(
                request_id,
                model,
                content,
                map_finish_reason(result.finish_reason.as_str()),
                result.prompt_tokens,
                result.completion_tokens,
            );
            return Ok(Json(response).into_response());
        }
    }

    // No local model — try proxying to cloud providers
    let lower_model = req.model.to_lowercase();

    // Claude subscription: proxy through local CLI subprocess (higher priority than API key)
    #[cfg(feature = "claude-subscription")]
    if let Some(sub_config) =
        crate::api::claude_sub::try_get_claude_subscription(&state, &req.model).await
    {
        tracing::info!(model = %req.model, "DIAG: anthropic proxying via claude subscription subprocess");
        // Build a minimal JSON for the subprocess handler (MessagesRequest isn't Serialize)
        let body = serde_json::json!({
            "model": req.model,
            "max_tokens": req.max_tokens,
            "messages": req.messages.iter().map(|m| {
                serde_json::json!({
                    "role": m.role,
                    "content": match &m.content {
                        AnthropicContent::Text(s) => serde_json::Value::String(s.clone()),
                        AnthropicContent::Blocks(blocks) => serde_json::Value::Array(
                            blocks.iter().map(|b| match b {
                                ContentBlock::Text { text } => serde_json::json!({"type": "text", "text": text}),
                                _ => serde_json::json!({"type": "text", "text": "[non-text content]"}),
                            }).collect()
                        ),
                    }
                })
            }).collect::<Vec<_>>(),
            "stream": req.stream,
            "system": match &req.system {
                Some(SystemContent::Text(s)) => serde_json::Value::String(s.clone()),
                Some(SystemContent::Blocks(blocks)) => serde_json::Value::Array(
                    blocks.iter().map(|b| serde_json::json!({"type": b.block_type, "text": b.text})).collect()
                ),
                None => serde_json::Value::Null,
            },
        });
        return crate::api::claude_sub::proxy_via_subprocess_anthropic(&sub_config, &body).await;
    }

    // Claude models → Anthropic cloud API (full pass-through, preserves tools/thinking)
    if lower_model.starts_with("claude") {
        let config = state.shared_state.metrics.providers_config.read().await;
        if let Some(ref entry) = config.anthropic {
            let api_key = entry.api_key.clone();
            drop(config);

            tracing::debug!(model = %req.model, "DIAG: anthropic proxying to cloud API");
            let body = serde_json::to_value(&ProxyMessagesRequest {
                model: &req.model,
                max_tokens: req.max_tokens,
                messages: &req.messages,
                system: &req.system,
                stream: req.stream,
                temperature: req.temperature,
                top_p: req.top_p,
                top_k: req.top_k,
                stop_sequences: &req.stop_sequences,
                tools: &req.tools,
                tool_choice: &req.tool_choice,
                metadata: &req.metadata,
                thinking: &req.thinking,
            })
            .map_err(|e| {
                ApiError(crate::error::SwarmError::Validation(format!(
                    "Failed to serialize request: {e}"
                )))
            })?;

            return providers::proxy_to_anthropic(&api_key, &body, req.stream).await;
        }
    }

    // Non-Claude models → translate Anthropic format to OpenAI and proxy through cloud providers
    {
        let config = state.shared_state.metrics.providers_config.read().await;
        if let Some(provider) = providers::resolve_provider(&req.model, &config) {
            let provider_name = provider.name.clone();
            let provider_url = provider.base_url.clone();
            let provider_key = provider.api_key.clone();
            drop(config);
            tracing::info!(
                model = %req.model,
                provider = %provider_name,
                "DIAG: anthropic→openai translation proxy to cloud provider"
            );
            return anthropic_to_openai_proxy(
                &req,
                &internal_messages,
                &provider_url,
                &provider_key,
            )
            .await;
        }
    }

    // No local model, no cloud provider configured
    Err(ApiError(crate::error::SwarmError::NoModelLoaded))
}

/// Serializable proxy request (borrows from the original request).
/// Includes all Claude Code fields for full pass-through to Anthropic cloud.
#[derive(Serialize)]
struct ProxyMessagesRequest<'a> {
    model: &'a str,
    max_tokens: u32,
    messages: &'a [AnthropicMessage],
    #[serde(skip_serializing_if = "Option::is_none")]
    system: &'a Option<SystemContent>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_k: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop_sequences: &'a Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: &'a Option<Vec<serde_json::Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: &'a Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    metadata: &'a Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking: &'a Option<serde_json::Value>,
}

// We need Serialize for AnthropicMessage/Content to proxy them
impl Serialize for AnthropicMessage {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeMap;
        let mut map = serializer.serialize_map(Some(2))?;
        map.serialize_entry("role", &self.role)?;
        map.serialize_entry("content", &self.content)?;
        map.end()
    }
}

impl Serialize for AnthropicContent {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            AnthropicContent::Text(s) => serializer.serialize_str(s),
            AnthropicContent::Blocks(blocks) => blocks.serialize(serializer),
        }
    }
}

impl Serialize for ContentBlock {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeMap;
        match self {
            ContentBlock::Text { text } => {
                let mut map = serializer.serialize_map(Some(2))?;
                map.serialize_entry("type", "text")?;
                map.serialize_entry("text", text)?;
                map.end()
            }
            ContentBlock::Image { source } => {
                let mut map = serializer.serialize_map(Some(2))?;
                map.serialize_entry("type", "image")?;
                map.serialize_entry("source", source)?;
                map.end()
            }
            ContentBlock::ToolUse { id, name, input } => {
                let mut map = serializer.serialize_map(Some(4))?;
                map.serialize_entry("type", "tool_use")?;
                map.serialize_entry("id", id)?;
                map.serialize_entry("name", name)?;
                map.serialize_entry("input", input)?;
                map.end()
            }
            ContentBlock::ToolResult {
                tool_use_id,
                content,
                is_error,
            } => {
                let mut map = serializer.serialize_map(Some(4))?;
                map.serialize_entry("type", "tool_result")?;
                map.serialize_entry("tool_use_id", tool_use_id)?;
                if let Some(c) = content {
                    map.serialize_entry("content", c)?;
                }
                if let Some(e) = is_error {
                    map.serialize_entry("is_error", e)?;
                }
                map.end()
            }
            ContentBlock::Thinking { thinking } => {
                let mut map = serializer.serialize_map(Some(2))?;
                map.serialize_entry("type", "thinking")?;
                map.serialize_entry("thinking", thinking)?;
                map.end()
            }
            ContentBlock::RedactedThinking { data } => {
                let mut map = serializer.serialize_map(Some(2))?;
                map.serialize_entry("type", "redacted_thinking")?;
                map.serialize_entry("data", data)?;
                map.end()
            }
        }
    }
}

impl Serialize for SystemContent {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            SystemContent::Text(s) => serializer.serialize_str(s),
            SystemContent::Blocks(blocks) => blocks.serialize(serializer),
        }
    }
}

impl Serialize for SystemBlock {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeMap;
        let mut map = serializer.serialize_map(None)?;
        map.serialize_entry("type", &self.block_type)?;
        if let Some(ref text) = self.text {
            map.serialize_entry("text", text)?;
        }
        if let Some(ref cc) = self.cache_control {
            map.serialize_entry("cache_control", cc)?;
        }
        map.end()
    }
}

/// Non-streaming inference via router, returning Anthropic format.
async fn anthropic_non_stream(
    router_tx: tokio::sync::mpsc::Sender<RouterCommand>,
    _req: &MessagesRequest,
    messages: Vec<ChatMessage>,
    params: SamplingParams,
    request_id: String,
    model: String,
) -> Result<axum::response::Response, ApiError> {
    let inference_req =
        InferenceRequest::local(ModelId(model.clone()), messages, params, false, None, None);

    let output = super::submit_to_router(&router_tx, inference_req).await?;

    let response = MessagesResponse::text(
        request_id,
        model,
        output.content,
        map_finish_reason(&output.finish_reason),
        output.prompt_tokens,
        output.completion_tokens,
    );

    Ok(Json(response).into_response())
}

/// Streaming inference via router, returning Anthropic SSE format.
async fn anthropic_stream(
    router_tx: tokio::sync::mpsc::Sender<RouterCommand>,
    _state: &AppState,
    _req: &MessagesRequest,
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
        while let Some(event) = token_rx.recv().await {
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
                finish_stop_reason = map_finish_reason(reason).into();
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
            let output_tokens = match &result {
                Ok(Ok(output)) => output.completion_tokens,
                _ => streamed_token_count,
            };
            send_sse_epilogue(&sse_tx, finish_stop_reason, output_tokens).await;
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
                    send_sse_epilogue(
                        &sse_tx,
                        map_finish_reason(&output.finish_reason).into(),
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
async fn anthropic_split_non_stream(
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

    let response = MessagesResponse::text(
        request_id,
        model,
        output.content,
        map_finish_reason(&output.finish_reason),
        output.prompt_tokens,
        output.completion_tokens,
    );

    Ok(Json(response).into_response())
}

/// Direct split-model streaming generation for Anthropic Messages API.
///
/// Same fast path as anthropic_split_non_stream but streams tokens via SSE
/// in Anthropic's streaming format.
async fn anthropic_split_stream(
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

        while let Some(event) = token_rx.recv().await {
            if let Some(fr) = &event.finish_reason {
                stop_reason = map_finish_reason(fr).to_string();
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

        send_sse_epilogue(&sse_tx, stop_reason, total_output_tokens).await;
    });

    Ok(build_anthropic_sse_response(sse_rx))
}

/// Translate an Anthropic Messages API request to OpenAI chat completions format
/// and proxy it to a non-Anthropic cloud provider. Translates the response back
/// to Anthropic Messages format.
async fn anthropic_to_openai_proxy(
    req: &MessagesRequest,
    messages: &[ChatMessage],
    base_url: &str,
    api_key: &str,
) -> Result<axum::response::Response, ApiError> {
    // Build OpenAI-compatible request body
    let openai_messages: Vec<serde_json::Value> = messages
        .iter()
        .map(|m| {
            let role = match m.role {
                Role::System => "system",
                Role::User => "user",
                Role::Assistant => "assistant",
                Role::Tool => "tool",
            };
            serde_json::json!({
                "role": role,
                "content": m.content,
            })
        })
        .collect();

    let mut body = serde_json::json!({
        "model": req.model,
        "messages": openai_messages,
        "max_tokens": req.max_tokens,
        "stream": req.stream,
    });
    if let Some(t) = req.temperature {
        body["temperature"] = serde_json::json!(t);
    }
    if let Some(tp) = req.top_p {
        body["top_p"] = serde_json::json!(tp);
    }
    if let Some(ref stops) = req.stop_sequences {
        body["stop"] = serde_json::json!(stops);
    }

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
        let truncated = super::scrub_truncate_error(&body);
        return Err(ApiError(crate::error::SwarmError::ProviderError {
            status: status.as_u16(),
            body: truncated,
        }));
    }

    // Handle streaming vs non-streaming responses
    if req.stream {
        // Upstream was told to stream (stream: true in body), so it returns SSE.
        // Stream the response back, translating OpenAI SSE events to Anthropic format.
        let (sse_tx, sse_rx) = tokio::sync::mpsc::channel::<AnthropicSseEvent>(64);
        let model_clone = req.model.clone();

        tokio::spawn(async move {
            let mut buffer = String::new();
            let mut content_so_far = String::new();
            let mut output_tokens: u32 = 0;

            // Send message_start
            let msg_id = format!("msg_{}", uuid::Uuid::new_v4().simple());
            send_sse_preamble(&sse_tx, &msg_id, &model_clone).await;

            // Read the upstream response body in chunks
            let mut resp = resp;
            while let Ok(Some(chunk)) = resp.chunk().await {
                buffer.push_str(&String::from_utf8_lossy(&chunk));

                // Process complete SSE lines
                while let Some(line_end) = buffer.find('\n') {
                    let line = buffer[..line_end].trim().to_string();
                    buffer = buffer[line_end + 1..].to_string();

                    if let Some(data) = line.strip_prefix("data: ") {
                        if data == "[DONE]" {
                            continue;
                        }
                        if let Ok(event) = serde_json::from_str::<serde_json::Value>(data) {
                            // Extract delta content from OpenAI SSE
                            if let Some(delta_text) =
                                event["choices"][0]["delta"]["content"].as_str()
                            {
                                if !delta_text.is_empty() {
                                    content_so_far.push_str(delta_text);
                                    output_tokens += 1;
                                    if sse_tx
                                        .send(AnthropicSseEvent::ContentBlockDelta {
                                            index: 0,
                                            text: delta_text.to_string(),
                                        })
                                        .await
                                        .is_err()
                                    {
                                        return; // client disconnected
                                    }
                                }
                            }
                        }
                    }
                }
            }

            send_sse_epilogue(&sse_tx, "end_turn".into(), output_tokens).await;
        });

        return Ok(build_anthropic_sse_response(sse_rx));
    }

    // Non-streaming: read full JSON response
    let openai_resp: serde_json::Value = resp.json().await.map_err(|e| {
        ApiError(crate::error::SwarmError::ProviderError {
            status: 502,
            body: format!("Cloud provider returned malformed JSON: {e}"),
        })
    })?;

    // Translate OpenAI response → Anthropic Messages format
    let choice = openai_resp["choices"]
        .as_array()
        .and_then(|c| c.first())
        .unwrap_or(&serde_json::Value::Null);
    let content_text = choice["message"]["content"]
        .as_str()
        .unwrap_or("")
        .to_string();
    let finish = choice["finish_reason"].as_str().unwrap_or("stop");

    let input_tokens = openai_resp["usage"]["prompt_tokens"].as_u64().unwrap_or(0) as u32;
    let output_tokens = openai_resp["usage"]["completion_tokens"]
        .as_u64()
        .unwrap_or(0) as u32;

    let response = MessagesResponse::text(
        format!("msg_{}", uuid::Uuid::new_v4().simple()),
        req.model.clone(),
        content_text,
        map_finish_reason(finish),
        input_tokens,
        output_tokens,
    );

    Ok(Json(response).into_response())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deserialize_text_content() {
        let json = r#"{"role":"user","content":"Hello"}"#;
        let msg: AnthropicMessage = serde_json::from_str(json).unwrap();
        assert_eq!(msg.role, "user");
        match msg.content {
            AnthropicContent::Text(t) => assert_eq!(t, "Hello"),
            _ => panic!("Expected text content"),
        }
    }

    #[test]
    fn deserialize_block_content() {
        let json = r#"{"role":"user","content":[{"type":"text","text":"Hello world"}]}"#;
        let msg: AnthropicMessage = serde_json::from_str(json).unwrap();
        match msg.content {
            AnthropicContent::Blocks(blocks) => {
                assert_eq!(blocks.len(), 1);
                match &blocks[0] {
                    ContentBlock::Text { text } => assert_eq!(text, "Hello world"),
                    _ => panic!("Expected text block"),
                }
            }
            _ => panic!("Expected blocks content"),
        }
    }

    #[test]
    fn system_prompt_to_internal() {
        let req = MessagesRequest {
            model: "local-model".into(),
            max_tokens: 100,
            messages: vec![AnthropicMessage {
                role: "user".into(),
                content: AnthropicContent::Text("Hi".into()),
            }],
            system: Some(SystemContent::Text("You are helpful.".into())),
            stream: false,
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: None,
            tools: None,
            tool_choice: None,
            metadata: None,
            thinking: None,
        };
        let msgs = to_internal_messages(&req);
        assert_eq!(msgs.len(), 2);
        assert!(matches!(msgs[0].role, Role::System));
        assert_eq!(msgs[0].content, "You are helpful.");
        assert!(matches!(msgs[1].role, Role::User));
        assert_eq!(msgs[1].content, "Hi");
    }

    #[test]
    fn system_blocks_to_internal() {
        let req = MessagesRequest {
            model: "local-model".into(),
            max_tokens: 100,
            messages: vec![],
            system: Some(SystemContent::Blocks(vec![
                SystemBlock {
                    block_type: "text".into(),
                    text: Some("Line 1".into()),
                    cache_control: None,
                },
                SystemBlock {
                    block_type: "text".into(),
                    text: Some("Line 2".into()),
                    cache_control: None,
                },
            ])),
            stream: false,
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: None,
            tools: None,
            tool_choice: None,
            metadata: None,
            thinking: None,
        };
        let msgs = to_internal_messages(&req);
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].content, "Line 1\nLine 2");
    }

    #[test]
    fn no_system_prompt() {
        let req = MessagesRequest {
            model: "test".into(),
            max_tokens: 100,
            messages: vec![AnthropicMessage {
                role: "user".into(),
                content: AnthropicContent::Text("Hello".into()),
            }],
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
        };
        let msgs = to_internal_messages(&req);
        assert_eq!(msgs.len(), 1);
        assert!(matches!(msgs[0].role, Role::User));
    }

    #[test]
    fn finish_reason_mapping() {
        assert_eq!(map_finish_reason("stop"), "end_turn");
        assert_eq!(map_finish_reason("length"), "max_tokens");
        assert_eq!(map_finish_reason("unknown"), "end_turn");
    }

    #[test]
    fn connectivity_probe_detection() {
        let probe = MessagesRequest {
            model: "claude-opus-4-7".into(),
            max_tokens: 1,
            messages: vec![AnthropicMessage {
                role: "user".into(),
                content: AnthropicContent::Text("Hi".into()),
            }],
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
        };
        assert!(is_connectivity_probe(&probe));

        let normal = MessagesRequest {
            model: "claude-opus-4-7".into(),
            max_tokens: 1024,
            messages: vec![AnthropicMessage {
                role: "user".into(),
                content: AnthropicContent::Text("Hi".into()),
            }],
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
        };
        assert!(!is_connectivity_probe(&normal));
    }

    #[test]
    fn sampling_params_conversion() {
        let req = MessagesRequest {
            model: "test".into(),
            max_tokens: 500,
            messages: vec![],
            system: None,
            stream: false,
            temperature: Some(0.5),
            top_p: Some(0.95),
            top_k: Some(50),
            stop_sequences: Some(vec!["STOP".into()]),
            tools: None,
            tool_choice: None,
            metadata: None,
            thinking: None,
        };
        let params = to_sampling_params(&req);
        assert!((params.temperature - 0.5).abs() < f32::EPSILON);
        assert!((params.top_p - 0.95).abs() < f32::EPSILON);
        assert_eq!(params.top_k, 50);
        assert_eq!(params.max_tokens, 500);
        assert_eq!(params.stop, vec!["STOP".to_string()]);
    }

    #[test]
    fn model_resolution() {
        assert_eq!(resolve_model("claude-opus-4-7"), "claude-opus-4-7");
        assert_eq!(
            resolve_model("anthropic:claude-opus-4-7"),
            "claude-opus-4-7"
        );
        assert_eq!(resolve_model("local:my-model"), "my-model");
    }

    #[test]
    fn sse_event_serialization() {
        let (event_type, data) = serialize_anthropic_event(&AnthropicSseEvent::MessageStart {
            id: "msg_test".into(),
            model: "claude-3".into(),
        });
        assert_eq!(event_type, "message_start");
        let v: serde_json::Value = serde_json::from_str(&data).unwrap();
        assert_eq!(v["type"], "message_start");
        assert_eq!(v["message"]["id"], "msg_test");

        let (event_type, data) = serialize_anthropic_event(&AnthropicSseEvent::ContentBlockDelta {
            index: 0,
            text: "hello".into(),
        });
        assert_eq!(event_type, "content_block_delta");
        let v: serde_json::Value = serde_json::from_str(&data).unwrap();
        assert_eq!(v["delta"]["text"], "hello");

        let (event_type, _) = serialize_anthropic_event(&AnthropicSseEvent::MessageStop);
        assert_eq!(event_type, "message_stop");
    }

    #[test]
    fn deserialize_full_request() {
        let json = r#"{
            "model": "claude-opus-4-7",
            "max_tokens": 1024,
            "system": "You are helpful.",
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
                {"role": "user", "content": "How are you?"}
            ]
        }"#;
        let req: MessagesRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.model, "claude-opus-4-7");
        assert_eq!(req.max_tokens, 1024);
        assert_eq!(req.messages.len(), 3);
        assert!(matches!(req.system, Some(SystemContent::Text(_))));
    }

    #[test]
    fn deserialize_tool_use_content() {
        let json = r#"{"role":"assistant","content":[
            {"type":"text","text":"I'll read the file."},
            {"type":"tool_use","id":"toolu_123","name":"Read","input":{"file_path":"/tmp/test.rs"}}
        ]}"#;
        let msg: AnthropicMessage = serde_json::from_str(json).unwrap();
        match msg.content {
            AnthropicContent::Blocks(blocks) => {
                assert_eq!(blocks.len(), 2);
                match &blocks[1] {
                    ContentBlock::ToolUse { id, name, input } => {
                        assert_eq!(id, "toolu_123");
                        assert_eq!(name, "Read");
                        assert_eq!(input["file_path"], "/tmp/test.rs");
                    }
                    _ => panic!("Expected ToolUse block"),
                }
            }
            _ => panic!("Expected blocks content"),
        }
    }

    #[test]
    fn deserialize_tool_result_content() {
        let json = r#"{"role":"user","content":[
            {"type":"tool_result","tool_use_id":"toolu_123","content":"file contents here"}
        ]}"#;
        let msg: AnthropicMessage = serde_json::from_str(json).unwrap();
        match msg.content {
            AnthropicContent::Blocks(blocks) => {
                assert_eq!(blocks.len(), 1);
                match &blocks[0] {
                    ContentBlock::ToolResult {
                        tool_use_id,
                        content,
                        ..
                    } => {
                        assert_eq!(tool_use_id, "toolu_123");
                        assert_eq!(
                            content.as_ref().unwrap().as_str().unwrap(),
                            "file contents here"
                        );
                    }
                    _ => panic!("Expected ToolResult block"),
                }
            }
            _ => panic!("Expected blocks content"),
        }
    }

    #[test]
    fn deserialize_thinking_content() {
        let json = r#"{"role":"assistant","content":[
            {"type":"thinking","thinking":"Let me analyze this..."},
            {"type":"text","text":"Here's my answer."}
        ]}"#;
        let msg: AnthropicMessage = serde_json::from_str(json).unwrap();
        match msg.content {
            AnthropicContent::Blocks(blocks) => {
                assert_eq!(blocks.len(), 2);
                match &blocks[0] {
                    ContentBlock::Thinking { thinking } => {
                        assert_eq!(thinking, "Let me analyze this...");
                    }
                    _ => panic!("Expected Thinking block"),
                }
            }
            _ => panic!("Expected blocks content"),
        }
    }

    #[test]
    fn deserialize_request_with_tools() {
        let json = r#"{
            "model": "claude-opus-4-7",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "Read /tmp/test.rs"}],
            "tools": [{"name": "Read", "description": "Read a file", "input_schema": {"type": "object"}}],
            "tool_choice": {"type": "auto"},
            "thinking": {"type": "enabled", "budget_tokens": 5000}
        }"#;
        let req: MessagesRequest = serde_json::from_str(json).unwrap();
        assert!(req.tools.is_some());
        assert_eq!(req.tools.as_ref().unwrap().len(), 1);
        assert!(req.tool_choice.is_some());
        assert!(req.thinking.is_some());
        assert_eq!(req.thinking.as_ref().unwrap()["type"], "enabled");
    }

    #[test]
    fn tool_use_to_internal_text() {
        let req = MessagesRequest {
            model: "test".into(),
            max_tokens: 100,
            messages: vec![AnthropicMessage {
                role: "assistant".into(),
                content: AnthropicContent::Blocks(vec![
                    ContentBlock::Text {
                        text: "I'll read it.".into(),
                    },
                    ContentBlock::ToolUse {
                        id: "t1".into(),
                        name: "Read".into(),
                        input: serde_json::json!({"path": "/tmp/x"}),
                    },
                ]),
            }],
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
        };
        let msgs = to_internal_messages(&req);
        assert_eq!(msgs.len(), 1);
        assert!(msgs[0].content.contains("I'll read it."));
        assert!(msgs[0].content.contains("[Tool call: Read("));
    }
}
