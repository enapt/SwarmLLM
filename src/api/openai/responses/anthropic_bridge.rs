//! V3 (responses_api_v2): translate between OpenAI Responses and
//! Anthropic Messages wire formats so a caller can `POST /v1/responses`
//! with `model=claude-*` and get back a Responses-shaped object.
//!
//! Why translate on the wire rather than proxy verbatim: the OpenAI
//! Responses endpoint and the Anthropic Messages endpoint have different
//! request shapes (`input` vs `messages`, `instructions` vs `system`,
//! `tools` function schema nesting, `reasoning.effort` vs `thinking`).
//! The caller speaks OpenAI SDK; Anthropic speaks its own API. A cloud
//! provider check happens before we translate so non-Anthropic models
//! never hit this path.
//!
//! Scope:
//! - Non-streaming request + response translation (full surface).
//! - Streaming SSE event mapping (text / tool_use / thinking deltas).
//! - Cloud API path (via `proxy_to_anthropic`) and subprocess path
//!   (via `proxy_via_subprocess_anthropic` under `claude-subscription`).
//!
//! Unscoped:
//! - Server-side tool blocks (web_search_tool_result etc.) are passed
//!   through as opaque content but not re-shaped into Responses items.

use std::collections::HashMap;

use axum::body::to_bytes;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::{json, Value};

use super::store;
use super::types::*;
use crate::api::providers::{proxy_to_anthropic, resolve_provider};
use crate::api::server::AppState;
use crate::error::{ApiError, SwarmError};

use super::MAX_UPSTREAM_BODY_BYTES;

/// Anthropic's `max_tokens` is a required field. Mirror the shared
/// Responses default if the caller didn't set `max_output_tokens`.
const DEFAULT_MAX_TOKENS: u32 = super::DEFAULT_MAX_OUTPUT_TOKENS;

// ============================================================================
// Request: Responses → Messages
// ============================================================================

/// Build an Anthropic Messages request JSON from a Responses request.
///
/// Caller supplies `stream` separately because the Responses field
/// mirrors `stream=true` only on explicit SSE requests — the non-
/// streaming proxy wants the Anthropic body without the stream flag.
pub fn responses_to_messages(req: &ResponsesRequest, stream: bool) -> Result<Value, SwarmError> {
    let mut obj = serde_json::Map::new();
    obj.insert("model".into(), Value::String(req.model.clone()));
    obj.insert(
        "max_tokens".into(),
        Value::Number(req.max_output_tokens.unwrap_or(DEFAULT_MAX_TOKENS).into()),
    );

    // `instructions` → `system` (string form — Anthropic also accepts
    // block array, but caller-supplied plain text is the common case).
    if let Some(system) = req.instructions.as_ref() {
        if !system.is_empty() {
            obj.insert("system".into(), Value::String(system.clone()));
        }
    }

    // Build messages array from input.
    let messages = input_to_anthropic_messages(&req.input)?;
    obj.insert("messages".into(), Value::Array(messages));

    if let Some(t) = req.temperature {
        obj.insert("temperature".into(), json!(t));
    }
    if let Some(p) = req.top_p {
        obj.insert("top_p".into(), json!(p));
    }
    if let Some(stop) = req.stop.as_ref() {
        let arr = match stop {
            StopField::One(s) => vec![Value::String(s.clone())],
            StopField::Many(v) => v.iter().map(|s| Value::String(s.clone())).collect(),
        };
        obj.insert("stop_sequences".into(), Value::Array(arr));
    }

    if let Some(metadata) = req.metadata.as_ref() {
        obj.insert(
            "metadata".into(),
            serde_json::to_value(metadata).unwrap_or(Value::Null),
        );
    }

    // tools[] — only `function` tools translate; built-in tools were
    // rejected in the handler before we got here.
    if let Some(tools) = req.tools.as_deref() {
        let anthropic_tools = translate_tools(tools)?;
        if !anthropic_tools.is_empty() {
            obj.insert("tools".into(), Value::Array(anthropic_tools));
        }
    }

    if let Some(tc) = req.tool_choice.as_ref() {
        obj.insert("tool_choice".into(), translate_tool_choice(tc));
    }

    // reasoning.effort → thinking { type: enabled, budget_tokens }
    if let Some(reasoning) = req.reasoning.as_ref() {
        if let Some(effort) = reasoning.effort.as_deref() {
            let budget = match effort {
                "minimal" => 1024,
                "low" => 2048,
                "medium" => 8192,
                "high" => 16384,
                _ => 8192, // Unknown efforts default to medium — conservative.
            };
            obj.insert(
                "thinking".into(),
                json!({ "type": "enabled", "budget_tokens": budget }),
            );
        }
    }

    if stream {
        obj.insert("stream".into(), Value::Bool(true));
    }

    Ok(Value::Object(obj))
}

/// Convert Responses `input` into an Anthropic `messages` array. Each
/// message has role + content (string or blocks).
fn input_to_anthropic_messages(input: &ResponsesInput) -> Result<Vec<Value>, SwarmError> {
    let mut out = Vec::new();

    match input {
        ResponsesInput::Text(s) => {
            out.push(json!({ "role": "user", "content": s }));
        }
        ResponsesInput::Items(items) => {
            for item in items {
                match item {
                    InputItem::Typed(TypedInputItem::Message(m)) => {
                        // Anthropic only models user|assistant roles.
                        // system/developer roles on Responses get hoisted
                        // into the top-level `system` field on request
                        // building — when they appear mid-conversation
                        // we flatten to a user-role synthetic note.
                        let role = match m.role.as_str() {
                            "user" => "user",
                            "assistant" => "assistant",
                            // Developer / system messages mid-conversation:
                            // we can't attach another top-level system to
                            // the body, so inline them as a user note
                            // (matches how chat_completions handles a
                            // mid-conversation system).
                            "system" | "developer" => "user",
                            other => {
                                return Err(SwarmError::Validation(format!(
                                    "Unsupported message role `{other}` on /v1/responses for Anthropic model"
                                )));
                            }
                        };
                        let content = message_content_to_anthropic(&m.content)?;
                        out.push(json!({ "role": role, "content": content }));
                    }
                    InputItem::Typed(TypedInputItem::FunctionCall(fc)) => {
                        // Prior assistant tool call — Anthropic encodes
                        // this as an assistant message with a `tool_use`
                        // content block. `input` must be a JSON object;
                        // we parse `arguments` on a best-effort basis.
                        let input_val: Value = serde_json::from_str(&fc.arguments)
                            .unwrap_or_else(|_| json!({"_raw_arguments": fc.arguments}));
                        out.push(json!({
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "tool_use",
                                    "id": fc.call_id,
                                    "name": fc.name,
                                    "input": input_val,
                                }
                            ],
                        }));
                    }
                    InputItem::Typed(TypedInputItem::FunctionCallOutput(out_item)) => {
                        // Tool result — Anthropic models this as a user
                        // message with a tool_result block referencing
                        // the tool_use id.
                        out.push(json!({
                            "role": "user",
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": out_item.call_id,
                                    "content": out_item.output,
                                }
                            ],
                        }));
                    }
                    InputItem::Typed(TypedInputItem::Reasoning(_)) => {
                        // Reasoning input items are cloud-side state; we
                        // don't echo them back on the Anthropic path
                        // (they belong to the OpenAI reasoning surface).
                    }
                    InputItem::Raw(value) => {
                        let kind = value
                            .get("type")
                            .and_then(|x| x.as_str())
                            .unwrap_or("<unknown>");
                        return Err(SwarmError::Validation(format!(
                            "Input item type `{kind}` is not supported on /v1/responses \
                             for Anthropic models. Supported: message, function_call, \
                             function_call_output."
                        )));
                    }
                }
            }
        }
    }

    // Anthropic requires at least one message.
    if out.is_empty() {
        return Err(SwarmError::Validation(
            "/v1/responses requires at least one input message for Anthropic models".into(),
        ));
    }

    Ok(out)
}

/// Translate an InputMessageContent (Text or Parts) into Anthropic's
/// content value (string or array of content blocks).
fn message_content_to_anthropic(content: &InputMessageContent) -> Result<Value, SwarmError> {
    match content {
        InputMessageContent::Text(s) => Ok(Value::String(s.clone())),
        InputMessageContent::Parts(parts) => {
            let blocks = parts
                .iter()
                .filter_map(|p| match input_part_to_anthropic_block(p).transpose() {
                    Some(Ok(v)) => Some(Ok(v)),
                    Some(Err(e)) => Some(Err(e)),
                    None => None,
                })
                .collect::<Result<Vec<_>, SwarmError>>()?;
            if blocks.is_empty() {
                // Empty content is valid but Anthropic prefers a string;
                // we emit an empty string to keep the shape compact.
                Ok(Value::String(String::new()))
            } else {
                Ok(Value::Array(blocks))
            }
        }
    }
}

/// Map one Responses input content part to an Anthropic content block
/// value. Returns `Ok(None)` when the part should be dropped (e.g. an
/// empty text part); `Err` for parts we can't translate.
fn input_part_to_anthropic_block(part: &InputContentPart) -> Result<Option<Value>, SwarmError> {
    let typed = match part {
        InputContentPart::Typed(t) => t,
        InputContentPart::Raw(v) => {
            let kind = v
                .get("type")
                .and_then(|x| x.as_str())
                .unwrap_or("<unknown>");
            return Err(SwarmError::Validation(format!(
                "Input content part type `{kind}` is not supported on /v1/responses \
                 for Anthropic models."
            )));
        }
    };

    match typed {
        TypedInputContentPart::Text { text, .. } => {
            if text.is_empty() {
                Ok(None)
            } else {
                Ok(Some(json!({ "type": "text", "text": text })))
            }
        }
        TypedInputContentPart::Image {
            image_url, file_id, ..
        } => {
            if file_id.is_some() {
                return Err(SwarmError::Validation(
                    "input_image file_id references are not supported for Anthropic \
                     models. Inline the image via image_url as a base64 data URI."
                        .into(),
                ));
            }
            let url = image_url.as_ref().ok_or_else(|| {
                SwarmError::Validation(
                    "input_image requires image_url (base64 data URI) for Anthropic models".into(),
                )
            })?;
            // Anthropic image source: { type: "base64", media_type, data }
            // or { type: "url", url }. Reuse the URL as-is for data URIs.
            let source = if let Some(data) = url.strip_prefix("data:") {
                // Parse "data:<media_type>;base64,<data>" minimally.
                let (meta, body) = data.split_once(',').ok_or_else(|| {
                    SwarmError::Validation("input_image image_url is a malformed data URI".into())
                })?;
                let media_type = meta
                    .split(';')
                    .next()
                    .filter(|s| !s.is_empty())
                    .unwrap_or("image/png")
                    .to_string();
                json!({
                    "type": "base64",
                    "media_type": media_type,
                    "data": body,
                })
            } else {
                // Remote URL form — Anthropic added native url support in
                // their 2024-09 API shape. Pass through.
                json!({ "type": "url", "url": url })
            };
            Ok(Some(json!({ "type": "image", "source": source })))
        }
        TypedInputContentPart::File { .. } => Err(SwarmError::Validation(
            "input_file is not supported on /v1/responses for Anthropic models. \
             Use input_text for text content or input_image for images."
                .into(),
        )),
        TypedInputContentPart::Audio { .. } => Err(SwarmError::Validation(
            "input_audio is not supported on /v1/responses for Anthropic models.".into(),
        )),
    }
}

/// Anthropic tools are flat `{ name, description, input_schema }`, not
/// nested under `function`. Translate each function tool.
fn translate_tools(tools: &[ToolDef]) -> Result<Vec<Value>, SwarmError> {
    let mut out = Vec::with_capacity(tools.len());
    for t in tools {
        match t {
            ToolDef::Typed(TypedToolDef::Function {
                name,
                description,
                parameters,
                strict: _,
                extras: _,
            }) => {
                let mut obj = serde_json::Map::new();
                obj.insert("name".into(), Value::String(name.clone()));
                if let Some(d) = description {
                    obj.insert("description".into(), Value::String(d.clone()));
                }
                if let Some(p) = parameters {
                    obj.insert("input_schema".into(), p.clone());
                } else {
                    // Anthropic requires input_schema; empty-object is the
                    // defined "no-args" shape.
                    obj.insert(
                        "input_schema".into(),
                        json!({ "type": "object", "properties": {} }),
                    );
                }
                out.push(Value::Object(obj));
            }
            ToolDef::Raw(value) => {
                let kind = super::types::raw_tool_kind_or_unknown(value);
                return Err(SwarmError::Validation(format!(
                    "Tool type `{kind}` is not supported on /v1/responses for \
                     Anthropic models. Only `function` tools translate."
                )));
            }
        }
    }
    Ok(out)
}

/// `tool_choice` semantics:
/// - "auto" / "any" / "none" / "required" → Anthropic `{"type":"auto"}` /
///   `{"type":"any"}` / `{"type":"none"}` / `{"type":"any"}` (Anthropic
///   has no "required"; "any" is the closest — force a tool call of
///   any kind).
/// - `{"type":"function","name":"X"}` → `{"type":"tool","name":"X"}`.
fn translate_tool_choice(tc: &ToolChoice) -> Value {
    match tc {
        ToolChoice::Mode(s) => match s.as_str() {
            "auto" => json!({"type": "auto"}),
            "none" => json!({"type": "none"}),
            "any" => json!({"type": "any"}),
            "required" => json!({"type": "any"}),
            // Pass unknown strings through as auto — safer than rejecting.
            _ => json!({"type": "auto"}),
        },
        ToolChoice::Object(obj) if obj.kind == "function" => {
            if let Some(name) = &obj.name {
                json!({ "type": "tool", "name": name })
            } else {
                json!({"type": "auto"})
            }
        }
        ToolChoice::Object(_) => json!({"type": "auto"}),
    }
}

// ============================================================================
// Response: Messages → Responses
// ============================================================================

/// Parse an Anthropic MessagesResponse (as JSON Value) and produce a
/// ResponsesResponse. `req` carries the original Responses request for
/// fields the upstream doesn't echo back (`instructions`, `tools`, etc.).
pub fn messages_to_responses(
    msg: &Value,
    req: &ResponsesRequest,
    response_id: &str,
    created_at: i64,
) -> Result<ResponsesResponse, SwarmError> {
    let model = msg
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or(req.model.as_str())
        .to_string();

    let stop_reason = msg.get("stop_reason").and_then(|v| v.as_str());
    let (status, incomplete_details) = match stop_reason {
        Some("max_tokens") => (
            ResponseStatus::Incomplete,
            Some(IncompleteDetails {
                reason: "max_output_tokens".into(),
                extras: HashMap::new(),
            }),
        ),
        Some("refusal") => (ResponseStatus::Failed, None),
        // end_turn | tool_use | stop_sequence | pause_turn → completed.
        Some(_) | None => (ResponseStatus::Completed, None),
    };

    let content = msg
        .get("content")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();

    // Walk content blocks. Anthropic's order mirrors the assistant's
    // actual turn structure so we preserve it directly: an assistant
    // message item holds every text block's concatenated content, and
    // each tool_use / thinking block produces its own output item.
    let mut output_items: Vec<OutputItem> = Vec::new();
    let mut message_text_parts: Vec<OutputContentPart> = Vec::new();
    let mut accumulated_text = String::new();

    for block in &content {
        let kind = block.get("type").and_then(|v| v.as_str()).unwrap_or("");
        match kind {
            "text" => {
                let text = block
                    .get("text")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                if !text.is_empty() {
                    accumulated_text.push_str(&text);
                    message_text_parts.push(OutputContentPart::Typed(
                        TypedOutputContentPart::Text {
                            text,
                            annotations: Vec::new(),
                            logprobs: None,
                            extras: HashMap::new(),
                        },
                    ));
                }
            }
            "tool_use" => {
                let id = block
                    .get("id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let name = block
                    .get("name")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let input = block.get("input").cloned().unwrap_or(Value::Null);
                let arguments = serde_json::to_string(&input).unwrap_or_else(|_| "{}".into());
                output_items.push(OutputItem::Typed(TypedOutputItem::FunctionCall(
                    FunctionCallItem {
                        call_id: id.clone(),
                        name,
                        arguments,
                        id: Some(format!("fc_{id}")),
                        status: Some("completed".into()),
                        extras: HashMap::new(),
                    },
                )));
            }
            "thinking" => {
                let text = block
                    .get("thinking")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                output_items.push(OutputItem::Typed(TypedOutputItem::Reasoning(
                    ReasoningItem {
                        summary: Some(vec![ReasoningSummaryPart::SummaryText {
                            text,
                            extras: HashMap::new(),
                        }]),
                        id: Some(format!("rs_{}", uuid::Uuid::new_v4().simple())),
                        status: Some("completed".into()),
                        encrypted_content: None,
                        extras: HashMap::new(),
                    },
                )));
            }
            _ => {
                // redacted_thinking and server-tool blocks aren't part of
                // the OpenAI Responses surface. Drop them rather than
                // synthesizing a cosmetic wrapper.
            }
        }
    }

    // If we collected any text blocks, emit a leading message item. We
    // place it before tool_use / reasoning items so non-tool-calling
    // turns remain a single-item output (matches OpenAI's shape).
    let message_item = if !message_text_parts.is_empty() {
        Some(OutputMessageItem {
            id: crate::api::openai::responses::new_message_id(),
            role: "assistant".into(),
            status: Some("completed".into()),
            content: message_text_parts,
            extras: HashMap::new(),
        })
    } else {
        None
    };

    // OutputItems ordering: message (if any) first, then tool_use,
    // then reasoning — the iteration above already interleaved
    // function_calls and reasoning in document order, so we prepend
    // message rather than reinserting.
    let mut final_output: Vec<OutputItem> = Vec::new();
    if let Some(mi) = message_item {
        final_output.push(OutputItem::Typed(TypedOutputItem::Message(mi)));
    }
    final_output.extend(output_items);

    // Usage translation. Anthropic gives input_tokens / output_tokens /
    // cache_creation_input_tokens / cache_read_input_tokens. OpenAI's
    // Responses shape nests cached under `input_tokens_details`.
    let usage_val = msg.get("usage");
    let input_tokens = usage_val
        .and_then(|u| u.get("input_tokens"))
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as u32;
    let output_tokens = usage_val
        .and_then(|u| u.get("output_tokens"))
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as u32;
    let cache_read = usage_val
        .and_then(|u| u.get("cache_read_input_tokens"))
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as u32;
    // cache_creation_input_tokens is billed at ~25× standard rate by
    // Anthropic — must surface in total_tokens for accurate cost tracking.
    // Stash in input_tokens_details.extras so callers who want the
    // breakdown can read it.
    let cache_creation = usage_val
        .and_then(|u| u.get("cache_creation_input_tokens"))
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as u32;
    let mut details_extras = HashMap::new();
    if cache_creation > 0 {
        details_extras.insert("cache_creation_input_tokens".into(), json!(cache_creation));
    }
    let usage = ResponsesUsage {
        input_tokens: input_tokens + cache_creation,
        output_tokens,
        total_tokens: input_tokens + cache_creation + output_tokens,
        input_tokens_details: if cache_read > 0 || cache_creation > 0 {
            Some(InputTokensDetails {
                cached_tokens: Some(cache_read),
                extras: details_extras,
            })
        } else {
            None
        },
        output_tokens_details: None,
    };

    Ok(ResponsesResponse {
        id: response_id.into(),
        object: "response".into(),
        created_at,
        status,
        model,
        output: final_output,
        output_text: Some(accumulated_text),
        usage,
        error: None,
        incomplete_details,
        previous_response_id: req.previous_response_id.clone(),
        instructions: req.instructions.clone(),
        tools: req.tools.clone(),
        tool_choice: req.tool_choice.clone(),
        parallel_tool_calls: req.parallel_tool_calls,
        temperature: Some(req.temperature.unwrap_or(super::DEFAULT_TEMPERATURE)),
        top_p: Some(req.top_p.unwrap_or(super::DEFAULT_TOP_P)),
        max_output_tokens: Some(req.max_output_tokens.unwrap_or(DEFAULT_MAX_TOKENS)),
        truncation: req.truncation.clone(),
        metadata: req.metadata.clone(),
        user: req.user.clone(),
        reasoning: req.reasoning.clone(),
        text: req.text.clone(),
        modalities: req.modalities.clone(),
        service_tier: req.service_tier.clone(),
        background: req.background,
        extras: HashMap::new(),
    })
}

// ============================================================================
// Orchestration: resolve Anthropic provider → translate → forward → translate
// ============================================================================

/// V3: equivalent of `try_proxy_openai_responses` for Anthropic
/// providers. Resolves the model, translates the Responses request to
/// an Anthropic Messages request, forwards via `proxy_to_anthropic`
/// (cloud API) or `proxy_via_subprocess_anthropic` (claude-subscription),
/// then translates the response back.
///
/// Returns `Ok(None)` when the model doesn't resolve to an Anthropic /
/// subprocess provider (so the caller falls through to local inference).
pub async fn try_proxy_anthropic_responses(
    state: &AppState,
    headers: &axum::http::HeaderMap,
    req: &ResponsesRequest,
) -> Result<Option<Response>, ApiError> {
    let stream = req.stream.unwrap_or(false);

    let config = state.shared_state.metrics.providers_config.read().await;
    let provider = match resolve_provider(&req.model, &config) {
        Some(p) if p.is_anthropic || p.is_subprocess => p,
        _ => return Ok(None),
    };
    // Snapshot the subscription config for subprocess routing before we
    // drop the read lock.
    #[cfg(feature = "claude-subscription")]
    let subscription_config = config.claude_subscription.clone();
    drop(config);

    tracing::info!(
        provider = %provider.name,
        model = %req.model,
        stream,
        "Proxying /v1/responses request to Anthropic provider (translated)"
    );

    let messages_body = responses_to_messages(req, stream).map_err(ApiError)?;

    if stream {
        return proxy_anthropic_responses_stream(
            &provider,
            #[cfg(feature = "claude-subscription")]
            subscription_config.as_ref(),
            headers,
            &messages_body,
            req,
        )
        .await
        .map(Some);
    }

    // Non-streaming: forward, buffer the upstream body, translate.
    let upstream = if provider.is_subprocess {
        #[cfg(feature = "claude-subscription")]
        {
            let sub_config = subscription_config.as_ref().ok_or_else(|| {
                ApiError(SwarmError::Validation(
                    "claude-subscription provider resolved but configuration is missing".into(),
                ))
            })?;
            crate::api::claude_sub::proxy_via_subprocess_anthropic(sub_config, &messages_body)
                .await?
        }
        #[cfg(not(feature = "claude-subscription"))]
        {
            return Err(ApiError(SwarmError::Validation(
                "Model resolved to a subprocess provider, but this build was \
                 compiled without the claude-subscription feature."
                    .into(),
            )));
        }
    } else {
        let beta = headers.get("anthropic-beta").and_then(|v| v.to_str().ok());
        let version = headers
            .get("anthropic-version")
            .and_then(|v| v.to_str().ok());
        proxy_to_anthropic(&provider.api_key, &messages_body, false, beta, version).await?
    };

    if !upstream.status().is_success() {
        // Forward the upstream error body to the caller verbatim — the
        // shape is Anthropic's `{error: {type, message}}`, which most
        // OpenAI SDKs surface intact.
        return Ok(Some(upstream));
    }

    let (parts, body) = upstream.into_parts();
    // Failures here are caused by the upstream provider's response, not by
    // local logic — surface as ProviderError (502) so the caller sees a
    // gateway-class status, not a generic 500.
    let bytes = to_bytes(body, MAX_UPSTREAM_BODY_BYTES).await.map_err(|e| {
        ApiError(SwarmError::ProviderError {
            status: 502,
            body: format!("Failed to buffer Anthropic upstream body: {e}"),
        })
    })?;
    let msg_value: Value = serde_json::from_slice(&bytes).map_err(|e| {
        ApiError(SwarmError::ProviderError {
            status: 502,
            body: format!("Failed to parse Anthropic upstream JSON: {e}"),
        })
    })?;

    let response_id = crate::api::openai::responses::new_response_id();
    let created_at = chrono::Utc::now().timestamp();
    let responses_resp =
        messages_to_responses(&msg_value, req, &response_id, created_at).map_err(ApiError)?;

    // M7: persist when store=true (OpenAI default).
    if req.store.unwrap_or(true) {
        let record = store::ResponsesRecord::new(
            req.clone(),
            responses_resp.clone(),
            created_at,
            store::DEFAULT_TTL_SECS,
        );
        if let Err(e) = store::store(&state.db, &record) {
            tracing::warn!(error = %e, id = %responses_resp.id,
                "Anthropic responses bridge store failed");
        }
    }

    let mut out = (StatusCode::OK, Json(responses_resp)).into_response();
    for (name, value) in parts.headers.iter() {
        if name == axum::http::header::CONTENT_TYPE || name == axum::http::header::CONTENT_LENGTH {
            continue;
        }
        out.headers_mut().insert(name.clone(), value.clone());
    }
    Ok(Some(out))
}

/// Translate an Anthropic streaming response into a Responses SSE
/// stream. Maps message_start / content_block_start / content_block_delta
/// / message_delta / message_stop → response.* events with monotonic
/// sequence numbers.
async fn proxy_anthropic_responses_stream(
    provider: &crate::api::providers::ProviderInfo,
    #[cfg(feature = "claude-subscription")] subscription_config: Option<
        &crate::api::claude_sub::ClaudeSubscriptionConfig,
    >,
    headers: &axum::http::HeaderMap,
    messages_body: &Value,
    req: &ResponsesRequest,
) -> Result<Response, ApiError> {
    let upstream = if provider.is_subprocess {
        #[cfg(feature = "claude-subscription")]
        {
            let sub_config = subscription_config.ok_or_else(|| {
                ApiError(SwarmError::Validation(
                    "claude-subscription provider resolved but configuration is missing".into(),
                ))
            })?;
            crate::api::claude_sub::proxy_via_subprocess_anthropic(sub_config, messages_body)
                .await?
        }
        #[cfg(not(feature = "claude-subscription"))]
        {
            return Err(ApiError(SwarmError::Validation(
                "Model resolved to a subprocess provider, but this build was \
                 compiled without the claude-subscription feature."
                    .into(),
            )));
        }
    } else {
        let beta = headers.get("anthropic-beta").and_then(|v| v.to_str().ok());
        let version = headers
            .get("anthropic-version")
            .and_then(|v| v.to_str().ok());
        proxy_to_anthropic(&provider.api_key, messages_body, true, beta, version).await?
    };

    if !upstream.status().is_success() {
        return Ok(upstream);
    }

    let response_id = crate::api::openai::responses::new_response_id();
    let created_at = chrono::Utc::now().timestamp();
    let req_cloned = req.clone();

    let sse_stream = stream_anthropic_to_responses(upstream, req_cloned, response_id, created_at);

    Ok(axum::response::sse::Sse::new(sse_stream)
        .keep_alive(
            axum::response::sse::KeepAlive::new().interval(std::time::Duration::from_secs(
                super::stream::SSE_KEEPALIVE_INTERVAL_SECS,
            )),
        )
        .into_response())
}

/// SSE event generator mapping Anthropic streaming events to Responses
/// streaming events. Shares the V1 shape (emit response.created +
/// response.in_progress first, then translate deltas as they arrive).
fn stream_anthropic_to_responses(
    upstream: Response,
    req: ResponsesRequest,
    response_id: String,
    created_at: i64,
) -> impl futures::Stream<Item = Result<axum::response::sse::Event, std::convert::Infallible>> {
    use bytes::{Buf, BytesMut};
    use futures::StreamExt;

    let body = upstream.into_body();
    let mut byte_stream = Box::pin(body.into_data_stream());

    let initial = build_initial_response(&req, &response_id, created_at);

    async_stream::stream! {
        let mut seq: u64 = 0;
        let mut buf = BytesMut::new();

        // Early lifecycle events (V1 pattern — match the local-path
        // shape so SDKs don't care about the hop).
        yield Ok::<_, std::convert::Infallible>(
            sse_event("response.created", json!({
                "type": "response.created",
                "sequence_number": seq,
                "response": initial,
            }))
        );
        seq += 1;
        yield Ok(sse_event("response.in_progress", json!({
            "type": "response.in_progress",
            "sequence_number": seq,
            "response": initial,
        })));
        seq += 1;

        // Per-block state. Anthropic sends content_block_start with an
        // index + block type; we track the active block per index so
        // deltas route to the right Responses event.
        #[derive(Default)]
        struct BlockState {
            kind: String,      // "text" | "tool_use" | "thinking"
            item_id: String,   // "msg_..." / "fc_..." / "rs_..."
            tool_id: String,   // only for tool_use
            tool_name: String, // only for tool_use
            text_so_far: String,
            args_so_far: String,
            content_part_opened: bool,
            output_item_opened: bool,
        }
        let mut blocks: std::collections::HashMap<u64, BlockState> = std::collections::HashMap::new();
        let mut output_index: u64 = 0;
        let mut block_output_index: std::collections::HashMap<u64, u64> = std::collections::HashMap::new();
        let mut final_output: Vec<OutputItem> = Vec::new();
        let mut accumulated_text = String::new();
        let mut stop_reason: Option<String> = None;
        let mut input_tokens: u32 = 0;
        let mut output_tokens: u32 = 0;
        let mut cache_read: u32 = 0;
        let mut model_from_upstream: Option<String> = None;

        'outer: while let Some(chunk) = byte_stream.next().await {
            let chunk = match chunk {
                Ok(b) => b,
                Err(e) => {
                    tracing::warn!(error = %e, "anthropic_bridge: upstream body error");
                    stop_reason = Some("error".into());
                    break 'outer;
                }
            };
            buf.extend_from_slice(&chunk);

            // Drain complete SSE events separated by blank lines.
            while let Some(pos) = super::stream::find_subslice(&buf, b"\n\n") {
                let block = buf[..pos].to_vec();
                buf.advance(pos + 2);

                // Extract `data:` payload(s) from the block via the shared
                // SSE parser. Anthropic emits one JSON body per event so we
                // only consume the first line.
                let data_lines = super::stream::parse_sse_block_data_lines(&block);
                let data = match data_lines.first() {
                    Some(d) => d.clone(),
                    None => continue,
                };
                if data == "[DONE]" {
                    break 'outer;
                }
                let evt: Value = match serde_json::from_str(&data) {
                    Ok(v) => v,
                    Err(e) => {
                        tracing::warn!(error = %e, "anthropic_bridge: malformed event");
                        continue;
                    }
                };

                let kind = evt.get("type").and_then(|v| v.as_str()).unwrap_or("");
                match kind {
                    "message_start" => {
                        if let Some(m) = evt.get("message").and_then(|v| v.get("model")).and_then(|v| v.as_str()) {
                            model_from_upstream = Some(m.into());
                        }
                        if let Some(u) = evt.get("message").and_then(|v| v.get("usage")) {
                            input_tokens = u.get("input_tokens").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
                            cache_read = u.get("cache_read_input_tokens").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
                        }
                    }
                    "content_block_start" => {
                        let index = evt.get("index").and_then(|v| v.as_u64()).unwrap_or(0);
                        let block_val = evt.get("content_block").cloned().unwrap_or(Value::Null);
                        let block_type = block_val.get("type").and_then(|v| v.as_str()).unwrap_or("").to_string();
                        let mut state = BlockState {
                            kind: block_type.clone(),
                            ..Default::default()
                        };

                        let my_output_index = output_index;
                        output_index += 1;
                        block_output_index.insert(index, my_output_index);

                        match block_type.as_str() {
                            "text" => {
                                state.item_id = crate::api::openai::responses::new_message_id();
                                // Emit output_item.added (message, empty content)
                                yield Ok(sse_event("response.output_item.added", json!({
                                    "type": "response.output_item.added",
                                    "sequence_number": seq,
                                    "output_index": my_output_index,
                                    "item": {
                                        "type": "message",
                                        "id": state.item_id,
                                        "role": "assistant",
                                        "status": "in_progress",
                                        "content": [],
                                    }
                                })));
                                seq += 1;
                                // Emit content_part.added (output_text)
                                yield Ok(sse_event("response.content_part.added", json!({
                                    "type": "response.content_part.added",
                                    "sequence_number": seq,
                                    "output_index": my_output_index,
                                    "content_index": 0,
                                    "item_id": state.item_id,
                                    "part": { "type": "output_text", "text": "", "annotations": [] }
                                })));
                                seq += 1;
                                state.content_part_opened = true;
                                state.output_item_opened = true;
                            }
                            "tool_use" => {
                                let tid = block_val.get("id").and_then(|v| v.as_str()).unwrap_or("").to_string();
                                let tname = block_val.get("name").and_then(|v| v.as_str()).unwrap_or("").to_string();
                                state.tool_id = tid.clone();
                                state.tool_name = tname.clone();
                                state.item_id = format!("fc_{tid}");
                                yield Ok(sse_event("response.output_item.added", json!({
                                    "type": "response.output_item.added",
                                    "sequence_number": seq,
                                    "output_index": my_output_index,
                                    "item": {
                                        "type": "function_call",
                                        "id": state.item_id,
                                        "call_id": tid,
                                        "name": tname,
                                        "arguments": "",
                                        "status": "in_progress",
                                    }
                                })));
                                seq += 1;
                                state.output_item_opened = true;
                            }
                            "thinking" => {
                                state.item_id = format!("rs_{}", uuid::Uuid::new_v4().simple());
                                // Responses doesn't have a dedicated
                                // "reasoning.added" lifecycle event shape
                                // in the current public schema — we emit
                                // output_item.added with the reasoning
                                // item in its initial form and accumulate
                                // deltas internally.
                                yield Ok(sse_event("response.output_item.added", json!({
                                    "type": "response.output_item.added",
                                    "sequence_number": seq,
                                    "output_index": my_output_index,
                                    "item": {
                                        "type": "reasoning",
                                        "id": state.item_id,
                                        "summary": [{"type": "summary_text", "text": ""}],
                                        "status": "in_progress",
                                    }
                                })));
                                seq += 1;
                                state.output_item_opened = true;
                            }
                            _ => {}
                        }

                        blocks.insert(index, state);
                    }
                    "content_block_delta" => {
                        let index = evt.get("index").and_then(|v| v.as_u64()).unwrap_or(0);
                        let delta = evt.get("delta").cloned().unwrap_or(Value::Null);
                        let delta_type = delta.get("type").and_then(|v| v.as_str()).unwrap_or("");
                        let my_output_index = *block_output_index.get(&index).unwrap_or(&0);
                        if let Some(state) = blocks.get_mut(&index) {
                            match delta_type {
                                "text_delta" => {
                                    let text = delta.get("text").and_then(|v| v.as_str()).unwrap_or("");
                                    if !text.is_empty() {
                                        state.text_so_far.push_str(text);
                                        accumulated_text.push_str(text);
                                        yield Ok(sse_event("response.output_text.delta", json!({
                                            "type": "response.output_text.delta",
                                            "sequence_number": seq,
                                            "output_index": my_output_index,
                                            "content_index": 0,
                                            "item_id": state.item_id,
                                            "delta": text,
                                        })));
                                        seq += 1;
                                    }
                                }
                                "input_json_delta" => {
                                    let fragment = delta.get("partial_json").and_then(|v| v.as_str()).unwrap_or("");
                                    if !fragment.is_empty() {
                                        state.args_so_far.push_str(fragment);
                                        yield Ok(sse_event("response.function_call_arguments.delta", json!({
                                            "type": "response.function_call_arguments.delta",
                                            "sequence_number": seq,
                                            "output_index": my_output_index,
                                            "item_id": state.item_id,
                                            "delta": fragment,
                                        })));
                                        seq += 1;
                                    }
                                }
                                "thinking_delta" => {
                                    let text = delta.get("thinking").and_then(|v| v.as_str()).unwrap_or("");
                                    if !text.is_empty() {
                                        state.text_so_far.push_str(text);
                                        // Responses doesn't yet define a
                                        // reasoning_summary_text.delta
                                        // event in the public schema, so
                                        // we accumulate silently and emit
                                        // the finalized summary at
                                        // content_block_stop.
                                    }
                                }
                                _ => {}
                            }
                        }
                    }
                    "content_block_stop" => {
                        let index = evt.get("index").and_then(|v| v.as_u64()).unwrap_or(0);
                        let my_output_index = *block_output_index.get(&index).unwrap_or(&0);
                        if let Some(state) = blocks.remove(&index) {
                            match state.kind.as_str() {
                                "text" => {
                                    if state.content_part_opened {
                                        yield Ok(sse_event("response.output_text.done", json!({
                                            "type": "response.output_text.done",
                                            "sequence_number": seq,
                                            "output_index": my_output_index,
                                            "content_index": 0,
                                            "item_id": state.item_id,
                                            "text": state.text_so_far,
                                        })));
                                        seq += 1;
                                        yield Ok(sse_event("response.content_part.done", json!({
                                            "type": "response.content_part.done",
                                            "sequence_number": seq,
                                            "output_index": my_output_index,
                                            "content_index": 0,
                                            "item_id": state.item_id,
                                            "part": {
                                                "type": "output_text",
                                                "text": state.text_so_far,
                                                "annotations": [],
                                            }
                                        })));
                                        seq += 1;
                                    }
                                    let done_item = json!({
                                        "type": "message",
                                        "id": state.item_id,
                                        "role": "assistant",
                                        "status": "completed",
                                        "content": [{
                                            "type": "output_text",
                                            "text": state.text_so_far,
                                            "annotations": [],
                                        }],
                                    });
                                    yield Ok(sse_event("response.output_item.done", json!({
                                        "type": "response.output_item.done",
                                        "sequence_number": seq,
                                        "output_index": my_output_index,
                                        "item": done_item.clone(),
                                    })));
                                    seq += 1;
                                    final_output.push(OutputItem::Typed(TypedOutputItem::Message(
                                        OutputMessageItem {
                                            id: state.item_id.clone(),
                                            role: "assistant".into(),
                                            status: Some("completed".into()),
                                            content: vec![OutputContentPart::Typed(TypedOutputContentPart::Text {
                                                text: state.text_so_far,
                                                annotations: Vec::new(),
                                                logprobs: None,
                                                extras: HashMap::new(),
                                            })],
                                            extras: HashMap::new(),
                                        }
                                    )));
                                }
                                "tool_use" => {
                                    yield Ok(sse_event("response.function_call_arguments.done", json!({
                                        "type": "response.function_call_arguments.done",
                                        "sequence_number": seq,
                                        "output_index": my_output_index,
                                        "item_id": state.item_id,
                                        "arguments": state.args_so_far,
                                    })));
                                    seq += 1;
                                    let done_item = json!({
                                        "type": "function_call",
                                        "id": state.item_id,
                                        "call_id": state.tool_id,
                                        "name": state.tool_name,
                                        "arguments": state.args_so_far,
                                        "status": "completed",
                                    });
                                    yield Ok(sse_event("response.output_item.done", json!({
                                        "type": "response.output_item.done",
                                        "sequence_number": seq,
                                        "output_index": my_output_index,
                                        "item": done_item.clone(),
                                    })));
                                    seq += 1;
                                    final_output.push(OutputItem::Typed(TypedOutputItem::FunctionCall(
                                        FunctionCallItem {
                                            call_id: state.tool_id,
                                            name: state.tool_name,
                                            arguments: state.args_so_far,
                                            id: Some(state.item_id),
                                            status: Some("completed".into()),
                                            extras: HashMap::new(),
                                        }
                                    )));
                                }
                                "thinking" => {
                                    let done_item = json!({
                                        "type": "reasoning",
                                        "id": state.item_id,
                                        "summary": [{"type": "summary_text", "text": state.text_so_far}],
                                        "status": "completed",
                                    });
                                    yield Ok(sse_event("response.output_item.done", json!({
                                        "type": "response.output_item.done",
                                        "sequence_number": seq,
                                        "output_index": my_output_index,
                                        "item": done_item,
                                    })));
                                    seq += 1;
                                    final_output.push(OutputItem::Typed(TypedOutputItem::Reasoning(
                                        ReasoningItem {
                                            id: Some(state.item_id),
                                            summary: Some(vec![ReasoningSummaryPart::SummaryText {
                                                text: state.text_so_far,
                                                extras: HashMap::new(),
                                            }]),
                                            encrypted_content: None,
                                            status: Some("completed".into()),
                                            extras: HashMap::new(),
                                        }
                                    )));
                                }
                                _ => {}
                            }
                        }
                    }
                    "message_delta" => {
                        if let Some(sr) = evt.get("delta").and_then(|d| d.get("stop_reason")).and_then(|v| v.as_str()) {
                            stop_reason = Some(sr.to_string());
                        }
                        if let Some(u) = evt.get("usage") {
                            let ot = u.get("output_tokens").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
                            if ot > 0 { output_tokens = ot; }
                        }
                    }
                    "message_stop" => {
                        break 'outer;
                    }
                    _ => {}
                }
            }
        }

        // Build final response + terminal event.
        let (status, incomplete_details) = match stop_reason.as_deref() {
            Some("max_tokens") => (ResponseStatus::Incomplete, Some(IncompleteDetails {
                reason: "max_output_tokens".into(),
                extras: HashMap::new(),
            })),
            Some("error") => (ResponseStatus::Failed, None),
            _ => (ResponseStatus::Completed, None),
        };

        let final_response = ResponsesResponse {
            id: response_id.clone(),
            object: "response".into(),
            created_at,
            status,
            model: model_from_upstream.unwrap_or_else(|| req.model.clone()),
            output: final_output,
            output_text: Some(accumulated_text),
            usage: ResponsesUsage {
                input_tokens,
                output_tokens,
                total_tokens: input_tokens + output_tokens,
                input_tokens_details: if cache_read > 0 {
                    Some(InputTokensDetails {
                        cached_tokens: Some(cache_read),
                        extras: HashMap::new(),
                    })
                } else {
                    None
                },
                output_tokens_details: None,
            },
            error: None,
            incomplete_details,
            previous_response_id: req.previous_response_id.clone(),
            instructions: req.instructions.clone(),
            tools: req.tools.clone(),
            tool_choice: req.tool_choice.clone(),
            parallel_tool_calls: req.parallel_tool_calls,
            temperature: Some(req.temperature.unwrap_or(super::DEFAULT_TEMPERATURE)),
            top_p: Some(req.top_p.unwrap_or(super::DEFAULT_TOP_P)),
            max_output_tokens: Some(req.max_output_tokens.unwrap_or(DEFAULT_MAX_TOKENS)),
            truncation: req.truncation.clone(),
            metadata: req.metadata.clone(),
            user: req.user.clone(),
            reasoning: req.reasoning.clone(),
            text: req.text.clone(),
            modalities: req.modalities.clone(),
            service_tier: req.service_tier.clone(),
            background: req.background,
            extras: HashMap::new(),
        };

        let terminal = match status {
            ResponseStatus::Failed => "response.failed",
            ResponseStatus::Incomplete => "response.incomplete",
            _ => "response.completed",
        };
        yield Ok(sse_event(terminal, json!({
            "type": terminal,
            "sequence_number": seq,
            "response": final_response,
        })));
    }
}

/// Build the minimal Responses object for response.created /
/// response.in_progress events in the streaming path. Thin wrapper around
/// the shared `super::build_response_skeleton` to keep all four
/// response-skeleton sites in lockstep.
fn build_initial_response(
    req: &ResponsesRequest,
    response_id: &str,
    created_at: i64,
) -> ResponsesResponse {
    super::build_response_skeleton(req, response_id, created_at, ResponseStatus::InProgress)
}

fn sse_event(name: &str, body: Value) -> axum::response::sse::Event {
    axum::response::sse::Event::default()
        .event(name)
        .data(serde_json::to_string(&body).unwrap_or_default())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_req() -> ResponsesRequest {
        ResponsesRequest {
            model: "claude-sonnet-4-6".into(),
            input: ResponsesInput::Text("hello".into()),
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
    fn text_input_maps_to_user_message_with_string_content() {
        let req = base_req();
        let body = responses_to_messages(&req, false).unwrap();
        assert_eq!(body["model"], "claude-sonnet-4-6");
        assert_eq!(body["max_tokens"], 2048);
        let messages = body["messages"].as_array().unwrap();
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0]["role"], "user");
        assert_eq!(messages[0]["content"], "hello");
    }

    #[test]
    fn instructions_become_top_level_system() {
        let mut req = base_req();
        req.instructions = Some("You are terse.".into());
        let body = responses_to_messages(&req, false).unwrap();
        assert_eq!(body["system"], "You are terse.");
    }

    #[test]
    fn max_output_tokens_maps_to_max_tokens() {
        let mut req = base_req();
        req.max_output_tokens = Some(512);
        let body = responses_to_messages(&req, false).unwrap();
        assert_eq!(body["max_tokens"], 512);
    }

    #[test]
    fn stop_field_maps_to_stop_sequences_array() {
        let mut req = base_req();
        req.stop = Some(StopField::One("</end>".into()));
        let body = responses_to_messages(&req, false).unwrap();
        assert_eq!(body["stop_sequences"], json!(["</end>"]));

        let mut req = base_req();
        req.stop = Some(StopField::Many(vec!["a".into(), "b".into()]));
        let body = responses_to_messages(&req, false).unwrap();
        assert_eq!(body["stop_sequences"], json!(["a", "b"]));
    }

    #[test]
    fn reasoning_effort_maps_to_thinking_budget() {
        let mut req = base_req();
        req.reasoning = Some(ReasoningOpts {
            effort: Some("high".into()),
            summary: None,
            generate_summary: None,
            extras: HashMap::new(),
        });
        let body = responses_to_messages(&req, false).unwrap();
        assert_eq!(body["thinking"]["type"], "enabled");
        assert!(body["thinking"]["budget_tokens"].as_u64().unwrap() >= 2048);
    }

    #[test]
    fn array_input_with_function_call_and_output_maps_to_tool_blocks() {
        let mut req = base_req();
        req.input = serde_json::from_value(json!([
            {"type": "message", "role": "user", "content": "call foo"},
            {"type": "function_call", "call_id": "c1", "name": "foo", "arguments": "{\"x\":1}"},
            {"type": "function_call_output", "call_id": "c1", "output": "42"},
        ]))
        .unwrap();
        let body = responses_to_messages(&req, false).unwrap();
        let messages = body["messages"].as_array().unwrap();
        assert_eq!(messages.len(), 3);
        // First: user text.
        assert_eq!(messages[0]["role"], "user");
        // Second: assistant with tool_use block.
        assert_eq!(messages[1]["role"], "assistant");
        let blocks = messages[1]["content"].as_array().unwrap();
        assert_eq!(blocks[0]["type"], "tool_use");
        assert_eq!(blocks[0]["id"], "c1");
        assert_eq!(blocks[0]["name"], "foo");
        assert_eq!(blocks[0]["input"], json!({"x": 1}));
        // Third: user with tool_result.
        assert_eq!(messages[2]["role"], "user");
        let blocks = messages[2]["content"].as_array().unwrap();
        assert_eq!(blocks[0]["type"], "tool_result");
        assert_eq!(blocks[0]["tool_use_id"], "c1");
        assert_eq!(blocks[0]["content"], "42");
    }

    #[test]
    fn function_tool_flattens_parameters_into_input_schema() {
        let mut req = base_req();
        req.tools = Some(
            serde_json::from_value(json!([
                {
                    "type": "function",
                    "name": "lookup",
                    "description": "find a thing",
                    "parameters": {"type": "object", "properties": {"q": {"type": "string"}}}
                }
            ]))
            .unwrap(),
        );
        let body = responses_to_messages(&req, false).unwrap();
        let tools = body["tools"].as_array().unwrap();
        assert_eq!(tools[0]["name"], "lookup");
        assert_eq!(tools[0]["description"], "find a thing");
        assert_eq!(tools[0]["input_schema"]["type"], "object");
        assert_eq!(
            tools[0]["input_schema"]["properties"]["q"]["type"],
            "string"
        );
        // Should NOT have a nested "function" key (that's chat shape).
        assert!(tools[0].get("function").is_none());
    }

    #[test]
    fn input_image_base64_data_uri_maps_to_anthropic_base64_source() {
        let mut req = base_req();
        req.input = serde_json::from_value(json!([
            {"type": "message", "role": "user", "content": [
                {"type": "input_text", "text": "what is this?"},
                {"type": "input_image", "image_url": "data:image/jpeg;base64,iVBORw=="},
            ]}
        ]))
        .unwrap();
        let body = responses_to_messages(&req, false).unwrap();
        let blocks = body["messages"][0]["content"].as_array().unwrap();
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0]["type"], "text");
        assert_eq!(blocks[1]["type"], "image");
        assert_eq!(blocks[1]["source"]["type"], "base64");
        assert_eq!(blocks[1]["source"]["media_type"], "image/jpeg");
        assert_eq!(blocks[1]["source"]["data"], "iVBORw==");
    }

    #[test]
    fn tool_choice_function_maps_to_anthropic_tool_name() {
        let mut req = base_req();
        req.tool_choice = Some(ToolChoice::Object(ToolChoiceObject {
            kind: "function".into(),
            name: Some("lookup".into()),
            extras: HashMap::new(),
        }));
        let body = responses_to_messages(&req, false).unwrap();
        assert_eq!(body["tool_choice"]["type"], "tool");
        assert_eq!(body["tool_choice"]["name"], "lookup");
    }

    #[test]
    fn messages_response_text_becomes_output_message_item() {
        let msg = json!({
            "model": "claude-sonnet-4-6",
            "content": [{"type": "text", "text": "hi there"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 3},
        });
        let req = base_req();
        let resp = messages_to_responses(&msg, &req, "resp_abc", 100).unwrap();
        assert_eq!(resp.id, "resp_abc");
        assert_eq!(resp.status, ResponseStatus::Completed);
        assert_eq!(resp.output.len(), 1);
        assert_eq!(resp.output_text.as_deref(), Some("hi there"));
        assert_eq!(resp.usage.input_tokens, 10);
        assert_eq!(resp.usage.output_tokens, 3);
    }

    #[test]
    fn messages_response_tool_use_becomes_function_call_item() {
        let msg = json!({
            "content": [
                {"type": "text", "text": "Looking that up."},
                {"type": "tool_use", "id": "tu_1", "name": "search", "input": {"q": "hello"}},
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 20, "output_tokens": 5},
        });
        let req = base_req();
        let resp = messages_to_responses(&msg, &req, "resp_xyz", 0).unwrap();
        assert_eq!(resp.status, ResponseStatus::Completed);
        assert_eq!(resp.output.len(), 2);
        // Message first.
        match &resp.output[0] {
            OutputItem::Typed(TypedOutputItem::Message(_)) => {}
            _ => panic!("expected message item first"),
        }
        match &resp.output[1] {
            OutputItem::Typed(TypedOutputItem::FunctionCall(fc)) => {
                assert_eq!(fc.call_id, "tu_1");
                assert_eq!(fc.name, "search");
                assert_eq!(fc.arguments, r#"{"q":"hello"}"#);
            }
            _ => panic!("expected function_call item"),
        }
    }

    #[test]
    fn messages_max_tokens_stop_reason_marks_incomplete() {
        let msg = json!({
            "content": [{"type": "text", "text": "partial"}],
            "stop_reason": "max_tokens",
            "usage": {"input_tokens": 5, "output_tokens": 2},
        });
        let req = base_req();
        let resp = messages_to_responses(&msg, &req, "resp_1", 0).unwrap();
        assert_eq!(resp.status, ResponseStatus::Incomplete);
        assert_eq!(
            resp.incomplete_details.as_ref().map(|d| d.reason.as_str()),
            Some("max_output_tokens")
        );
    }

    #[test]
    fn messages_thinking_block_becomes_reasoning_item() {
        let msg = json!({
            "content": [
                {"type": "thinking", "thinking": "deliberating..."},
                {"type": "text", "text": "answer"},
            ],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 5, "output_tokens": 10},
        });
        let req = base_req();
        let resp = messages_to_responses(&msg, &req, "resp_1", 0).unwrap();
        // Output order: message (text), then reasoning (thinking).
        assert_eq!(resp.output.len(), 2);
        match &resp.output[0] {
            OutputItem::Typed(TypedOutputItem::Message(_)) => {}
            _ => panic!("expected message first"),
        }
        match &resp.output[1] {
            OutputItem::Typed(TypedOutputItem::Reasoning(r)) => {
                let summary = r.summary.as_ref().expect("summary populated");
                assert_eq!(summary.len(), 1);
                match &summary[0] {
                    ReasoningSummaryPart::SummaryText { text, .. } => {
                        assert_eq!(text, "deliberating...");
                    }
                }
            }
            _ => panic!("expected reasoning item"),
        }
    }

    #[test]
    fn cache_read_tokens_populate_input_tokens_details() {
        let msg = json!({
            "content": [{"type": "text", "text": "ok"}],
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": 100,
                "output_tokens": 5,
                "cache_read_input_tokens": 80,
            },
        });
        let req = base_req();
        let resp = messages_to_responses(&msg, &req, "resp_1", 0).unwrap();
        let cached = resp
            .usage
            .input_tokens_details
            .as_ref()
            .and_then(|d| d.cached_tokens);
        assert_eq!(cached, Some(80));
    }

    #[test]
    fn empty_input_is_rejected() {
        let mut req = base_req();
        req.input = ResponsesInput::Items(Vec::new());
        let err = responses_to_messages(&req, false).unwrap_err().to_string();
        assert!(err.contains("at least one input message"), "{err}");
    }
}
