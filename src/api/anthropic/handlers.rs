use axum::response::IntoResponse;
use axum::Json;

use crate::api::providers;
use crate::api::server::AppState;
use crate::error::ApiError;
use crate::inference::router::RouterCommand;
use crate::types::{ChatMessage, InferenceRequest, ModelId, Role, SamplingParams};

use super::convert::map_finish_reason;
use super::sse::{
    build_anthropic_sse_response, send_sse_epilogue, send_sse_preamble, AnthropicSseEvent,
};
use super::types::{MessagesRequest, MessagesResponse};

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
    _req: &MessagesRequest,
    messages: Vec<ChatMessage>,
    params: SamplingParams,
    request_id: String,
    model: String,
) -> Result<axum::response::Response, ApiError> {
    let inference_req =
        InferenceRequest::local(ModelId(model.clone()), messages, params, false, None, None);

    let output = crate::api::submit_to_router(&router_tx, inference_req).await?;

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
pub(super) async fn anthropic_stream(
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
pub(super) async fn anthropic_to_openai_proxy(
    req: &MessagesRequest,
    messages: &[ChatMessage],
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
        let truncated = crate::api::scrub_truncate_error(&body);
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
