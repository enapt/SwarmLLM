use axum::extract::State;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::IntoResponse;
use axum::Json;
use serde::{Deserialize, Serialize};
use std::convert::Infallible;
use tokio_stream::StreamExt;

use crate::api::providers;
use crate::api::server::AppState;
use crate::error::ApiError;
use crate::inference::chat_template;
use crate::inference::router::{RouterCommand, StreamingTokenEvent};
use crate::types::{
    ChatMessage, InferenceRequest, ModelId, NodeId, PriorityTier, Role, SamplingParams,
};

/// SSE keep-alive interval for streaming responses (seconds).
const SSE_KEEPALIVE_INTERVAL_SECS: u64 = 15;

// ---- Anthropic Messages API types ----

#[derive(Debug, Deserialize)]
pub struct MessagesRequest {
    pub model: String,
    pub max_tokens: u32,
    pub messages: Vec<AnthropicMessage>,
    #[serde(default)]
    pub system: Option<SystemContent>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default = "default_temperature")]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub top_k: Option<u32>,
    #[serde(default)]
    pub stop_sequences: Option<Vec<String>>,
    /// Tool definitions for function calling (Claude Code sends these).
    #[serde(default)]
    pub tools: Option<Vec<serde_json::Value>>,
    /// Tool choice: "auto", "any", "none", or {"type":"tool","name":"..."}
    #[serde(default)]
    pub tool_choice: Option<serde_json::Value>,
    /// Request metadata (e.g. user_id for abuse tracking).
    #[serde(default)]
    pub metadata: Option<serde_json::Value>,
    /// Extended thinking configuration: {"type":"enabled","budget_tokens":N}
    #[serde(default)]
    pub thinking: Option<serde_json::Value>,
}

fn default_temperature() -> Option<f32> {
    Some(1.0)
}

/// System content: either a plain string or an array of blocks.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum SystemContent {
    Text(String),
    Blocks(Vec<SystemBlock>),
}

#[derive(Debug, Deserialize)]
pub struct SystemBlock {
    #[serde(rename = "type")]
    pub block_type: String,
    #[serde(default)]
    pub text: Option<String>,
    /// Cache control hint (Anthropic prompt caching).
    #[serde(default)]
    pub cache_control: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
pub struct AnthropicMessage {
    pub role: String,
    pub content: AnthropicContent,
}

/// Content field: either a plain string or an array of content blocks.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum AnthropicContent {
    Text(String),
    Blocks(Vec<ContentBlock>),
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
pub enum ContentBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image")]
    Image { source: serde_json::Value },
    /// Tool use request from assistant (Claude Code tool calls).
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    /// Tool result from user (response to tool call).
    #[serde(rename = "tool_result")]
    ToolResult {
        tool_use_id: String,
        #[serde(default)]
        content: Option<serde_json::Value>,
        #[serde(default)]
        is_error: Option<bool>,
    },
    /// Thinking block (extended thinking / chain-of-thought).
    #[serde(rename = "thinking")]
    Thinking { thinking: String },
    /// Redacted thinking block.
    #[serde(rename = "redacted_thinking")]
    RedactedThinking { data: String },
}

// ---- Response types ----

#[derive(Debug, Serialize)]
pub struct MessagesResponse {
    pub id: String,
    #[serde(rename = "type")]
    pub response_type: &'static str,
    pub role: &'static str,
    pub content: Vec<ResponseContentBlock>,
    pub model: String,
    pub stop_reason: Option<String>,
    pub stop_sequence: Option<String>,
    pub usage: AnthropicUsage,
}

/// Response content block — supports text, tool_use, and thinking.
#[derive(Debug, Serialize)]
#[serde(tag = "type")]
pub enum ResponseContentBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    #[serde(rename = "thinking")]
    Thinking { thinking: String },
}

#[derive(Debug, Serialize)]
pub struct AnthropicUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
}

// ---- Helpers ----

/// Check if a request is a connectivity probe (Claude Code sends these to test the endpoint).
/// Narrowed to also check message content to avoid false-positiving on legitimate max_tokens=1 requests.
fn is_connectivity_probe(req: &MessagesRequest) -> bool {
    if req.max_tokens != 1 || req.messages.len() != 1 || req.stream {
        return false;
    }
    // Check if the single message has very short content (probes are typically <20 chars)
    let content_len = match &req.messages[0].content {
        AnthropicContent::Text(s) => s.len(),
        AnthropicContent::Blocks(blocks) => blocks
            .iter()
            .map(|b| match b {
                ContentBlock::Text { text } => text.len(),
                _ => 100, // non-text content = not a probe
            })
            .sum(),
    };
    content_len <= 20
}

/// Convert Anthropic messages to internal ChatMessage format.
fn to_internal_messages(req: &MessagesRequest) -> Vec<ChatMessage> {
    let mut messages = Vec::new();

    // System prompt → System role
    if let Some(ref system) = req.system {
        let text = match system {
            SystemContent::Text(s) => s.clone(),
            SystemContent::Blocks(blocks) => blocks
                .iter()
                .filter_map(|b| b.text.as_ref())
                .cloned()
                .collect::<Vec<_>>()
                .join("\n"),
        };
        if !text.is_empty() {
            messages.push(ChatMessage {
                role: Role::System,
                content: text,
                images: vec![],
            });
        }
    }

    for msg in &req.messages {
        let role = match msg.role.as_str() {
            "user" => Role::User,
            "assistant" => Role::Assistant,
            _ => Role::User,
        };
        let text = match &msg.content {
            AnthropicContent::Text(s) => s.clone(),
            AnthropicContent::Blocks(blocks) => {
                let mut texts = Vec::new();
                for b in blocks {
                    match b {
                        ContentBlock::Text { text } => texts.push(text.clone()),
                        ContentBlock::Image { .. } => {
                            // Image handling: VLM images handled via openai.rs path
                        }
                        ContentBlock::ToolResult { content, .. } => {
                            // Include tool result text in conversation for local inference
                            if let Some(c) = content {
                                if let Some(s) = c.as_str() {
                                    texts.push(s.to_string());
                                } else if let Some(arr) = c.as_array() {
                                    for item in arr {
                                        if let Some(t) = item.get("text").and_then(|v| v.as_str()) {
                                            texts.push(t.to_string());
                                        }
                                    }
                                }
                            }
                        }
                        ContentBlock::ToolUse { name, input, .. } => {
                            // For local inference, represent tool calls as text
                            texts.push(format!("[Tool call: {name}({input})]"));
                        }
                        ContentBlock::Thinking { thinking } => {
                            // Include thinking in context for local models
                            texts.push(format!("<thinking>{thinking}</thinking>"));
                        }
                        ContentBlock::RedactedThinking { .. } => {}
                    }
                }
                texts.join("\n")
            }
        };
        messages.push(ChatMessage {
            role,
            content: text,
            images: vec![],
        });
    }

    messages
}

/// Convert Anthropic request to SamplingParams.
fn to_sampling_params(req: &MessagesRequest) -> SamplingParams {
    SamplingParams {
        temperature: req.temperature.unwrap_or(1.0).clamp(0.0, 2.0),
        top_p: req.top_p.unwrap_or(0.9).clamp(f32::EPSILON, 1.0),
        top_k: req.top_k.unwrap_or(40),
        max_tokens: req.max_tokens.clamp(1, 32768),
        stop: req.stop_sequences.clone().unwrap_or_default(),
        frequency_penalty: 0.0,
        presence_penalty: 0.0,
        logprobs: false,
        top_logprobs: 0,
    }
}

/// Map internal finish reason to Anthropic stop_reason.
fn map_finish_reason(reason: &str) -> &'static str {
    match reason {
        "stop" => "end_turn",
        "length" => "max_tokens",
        _ => "end_turn",
    }
}

/// Resolve model name: strip `provider:` prefix if present.
fn resolve_model(model: &str) -> &str {
    if let Some((_provider, model_name)) = model.split_once(':') {
        model_name
    } else {
        model
    }
}

// ---- Handler ----

/// POST /v1/messages — Anthropic Messages API endpoint.
pub async fn messages(
    State(state): State<AppState>,
    crate::api::server::JsonBody(req): crate::api::server::JsonBody<MessagesRequest>,
) -> Result<axum::response::Response, ApiError> {
    // Validate model field length to prevent DashMap memory exhaustion
    if req.model.len() > 256 {
        return Err(ApiError(crate::error::SwarmError::Validation(
            "model name too long (max 256 chars)".into(),
        )));
    }

    // Validate messages array is not empty
    if req.messages.is_empty() {
        return Err(ApiError(crate::error::SwarmError::Validation(
            "messages array must not be empty".into(),
        )));
    }

    // Limit message count to prevent excessive prompt construction overhead
    if req.messages.len() > 4096 {
        return Err(ApiError(crate::error::SwarmError::Validation(
            "Too many messages (max 4096)".into(),
        )));
    }

    // Validate stop sequences
    if let Some(ref stops) = req.stop_sequences {
        if stops.len() > 16 {
            return Err(ApiError(crate::error::SwarmError::Validation(
                "Too many stop sequences (max 16)".into(),
            )));
        }
        if stops.iter().any(|s| s.is_empty() || s.len() > 256) {
            return Err(ApiError(crate::error::SwarmError::Validation(
                "Stop sequences must be 1–256 chars each".into(),
            )));
        }
    }

    // Limit tools array — count and per-tool field sizes
    if let Some(ref tools) = req.tools {
        if tools.len() > 128 {
            return Err(ApiError(crate::error::SwarmError::Validation(
                "Too many tools (max 128)".into(),
            )));
        }
        for tool in tools {
            if let Some(name) = tool.get("name").and_then(|v| v.as_str()) {
                if name.len() > 256 {
                    return Err(ApiError(crate::error::SwarmError::Validation(format!(
                        "Tool name too long: {} (max 256)",
                        name.len()
                    ))));
                }
            }
            if let Some(desc) = tool.get("description").and_then(|v| v.as_str()) {
                if desc.len() > 4096 {
                    return Err(ApiError(crate::error::SwarmError::Validation(format!(
                        "Tool description too long: {} (max 4096)",
                        desc.len()
                    ))));
                }
            }
            if let Some(schema) = tool.get("input_schema") {
                let schema_size = schema.to_string().len();
                if schema_size > 65536 {
                    return Err(ApiError(crate::error::SwarmError::Validation(format!(
                        "Tool input_schema too large: {} bytes (max 64KB)",
                        schema_size
                    ))));
                }
            }
        }
    }

    let request_id = format!("msg_{}", uuid::Uuid::new_v4().simple());
    let model = resolve_model(&req.model).to_string();

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
        let response = MessagesResponse {
            id: request_id,
            response_type: "message",
            role: "assistant",
            content: vec![ResponseContentBlock::Text { text: "ok".into() }],
            model,
            stop_reason: Some("end_turn".into()),
            stop_sequence: None,
            usage: AnthropicUsage {
                input_tokens: 1,
                output_tokens: 1,
            },
        };
        return Ok(Json(response).into_response());
    }

    let internal_messages = to_internal_messages(&req);
    let sampling_params = to_sampling_params(&req);

    // Try local inference first — resolve model to registry ID using shared helper.
    let model = {
        let info = state.shared_state.loaded_model_info.read().await;
        if let Some(i) = info.as_ref() {
            if !crate::api::openai::all_shards_available(&state, &model) {
                let resolved =
                    crate::api::openai::resolve_loaded_model_registry_id(&state, &i.name);
                // Only use resolved ID if request matches. Never silently substitute.
                if model == "auto" || model == resolved || model == i.name {
                    resolved
                } else {
                    model
                }
            } else {
                model
            }
        } else {
            model
        }
    };

    // Check if network has all shards for this model
    let network_available = crate::api::openai::all_shards_available(&state, &model);

    // Fast path: if we have a complete local split model for the REQUESTED model, generate directly.
    // Match by model ID — not just "any loaded model" (compare sends different model IDs).
    let requested_mid = crate::types::ModelId(model.clone());
    let has_local_split_model = state
        .shared_state
        .split_models
        .iter()
        .any(|e| e.key().0 == requested_mid && e.value().is_complete);

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
            let (tmpl, bos, eos) = {
                let info = state.shared_state.loaded_model_info.read().await;
                match info.as_ref() {
                    Some(i) => (
                        i.chat_template.clone(),
                        i.bos_token.clone(),
                        i.eos_token.clone(),
                    ),
                    None => (None, String::new(), String::new()),
                }
            };
            let prompt =
                chat_template::build_prompt(&internal_messages, tmpl.as_deref(), &bos, &eos);

            let mut executor = state.executor.lock().await;
            let (content, result) = executor
                .generate(&prompt, &sampling_params)
                .map_err(ApiError)?;

            let response = MessagesResponse {
                id: request_id,
                response_type: "message",
                role: "assistant",
                content: vec![ResponseContentBlock::Text { text: content }],
                model,
                stop_reason: Some(map_finish_reason(result.finish_reason.as_str()).into()),
                stop_sequence: None,
                usage: AnthropicUsage {
                    input_tokens: result.prompt_tokens,
                    output_tokens: result.completion_tokens,
                },
            };
            return Ok(Json(response).into_response());
        }
    }

    // No local model — try proxying to cloud providers
    let lower_model = req.model.to_lowercase();

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
    let (result_tx, result_rx) = tokio::sync::oneshot::channel();

    let inference_req = InferenceRequest {
        id: uuid::Uuid::new_v4(),
        model_id: ModelId(model.clone()),
        messages,
        sampling_params: params,
        stream: false,
        requester: NodeId([0u8; 32]),
        priority: PriorityTier::Silver,
        created_at: chrono::Utc::now(),
        session_id: None,
        lora_adapter: None,
    };

    router_tx
        .send(RouterCommand::Submit {
            request: inference_req,
            result_tx,
        })
        .await
        .map_err(|_| {
            ApiError(crate::error::SwarmError::ServiceUnavailable(
                "Router unavailable".into(),
            ))
        })?;

    let output = result_rx.await.map_err(|_| {
        ApiError(crate::error::SwarmError::ServiceUnavailable(
            "Router dropped the request".into(),
        ))
    })??;

    let response = MessagesResponse {
        id: request_id,
        response_type: "message",
        role: "assistant",
        content: vec![ResponseContentBlock::Text {
            text: output.content,
        }],
        model,
        stop_reason: Some(map_finish_reason(&output.finish_reason).into()),
        stop_sequence: None,
        usage: AnthropicUsage {
            input_tokens: output.prompt_tokens,
            output_tokens: output.completion_tokens,
        },
    };

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
        // message_start
        let _ = sse_tx
            .send(AnthropicSseEvent::MessageStart {
                id: rid.clone(),
                model: model.clone(),
            })
            .await;

        // content_block_start
        let _ = sse_tx
            .send(AnthropicSseEvent::ContentBlockStart { index: 0 })
            .await;

        // Stream tokens — count events as a fallback estimate
        let mut got_finish = false;
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
                let _ = sse_tx
                    .send(AnthropicSseEvent::ContentBlockStop { index: 0 })
                    .await;
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
                    break;
                }
            }
        }

        // Get authoritative token count from the result when available
        let result = result_rx.await;
        if got_finish {
            // Use completion_tokens from result if available, else fall back to event count
            let output_tokens = match &result {
                Ok(Ok(output)) => output.completion_tokens,
                _ => streamed_token_count,
            };
            let _ = sse_tx
                .send(AnthropicSseEvent::MessageDelta {
                    stop_reason: finish_stop_reason,
                    output_tokens,
                })
                .await;
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
                    let _ = sse_tx
                        .send(AnthropicSseEvent::ContentBlockStop { index: 0 })
                        .await;
                    let _ = sse_tx
                        .send(AnthropicSseEvent::MessageDelta {
                            stop_reason: map_finish_reason(&output.finish_reason).into(),
                            output_tokens: output.completion_tokens,
                        })
                        .await;
                }
                _ => {
                    let _ = sse_tx
                        .send(AnthropicSseEvent::ContentBlockStop { index: 0 })
                        .await;
                    let _ = sse_tx
                        .send(AnthropicSseEvent::MessageDelta {
                            stop_reason: "end_turn".into(),
                            output_tokens: streamed_token_count,
                        })
                        .await;
                }
            }
        }
        let _ = sse_tx.send(AnthropicSseEvent::MessageStop).await;
    });

    let stream = tokio_stream::wrappers::ReceiverStream::new(sse_rx).map(move |event| {
        let (event_type, data) = serialize_anthropic_event(&event);
        Ok::<_, Infallible>(Event::default().event(event_type).data(data))
    });

    Ok(Sse::new(stream)
        .keep_alive(
            KeepAlive::new().interval(std::time::Duration::from_secs(SSE_KEEPALIVE_INTERVAL_SECS)),
        )
        .into_response())
}

/// Internal SSE event types for Anthropic streaming.
enum AnthropicSseEvent {
    MessageStart {
        id: String,
        model: String,
    },
    ContentBlockStart {
        index: u32,
    },
    ContentBlockDelta {
        index: u32,
        text: String,
    },
    ContentBlockStop {
        index: u32,
    },
    MessageDelta {
        stop_reason: String,
        output_tokens: u32,
    },
    MessageStop,
}

/// Serialize an Anthropic SSE event to (event_type, data_json).
fn serialize_anthropic_event(event: &AnthropicSseEvent) -> (&'static str, String) {
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
        AnthropicSseEvent::ContentBlockDelta { index, text } => (
            "content_block_delta",
            serde_json::json!({
                "type": "content_block_delta",
                "index": index,
                "delta": { "type": "text_delta", "text": text }
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
            output_tokens,
        } => (
            "message_delta",
            serde_json::json!({
                "type": "message_delta",
                "delta": { "stop_reason": stop_reason, "stop_sequence": null },
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
    let meta = crate::api::openai::get_split_model_meta(state, &requested_mid)
        .ok_or(ApiError(crate::error::SwarmError::NoModelLoaded))?;
    let (chat_tmpl, bos, eos_str, layer_range) = (
        meta.chat_template,
        meta.bos_token,
        meta.eos_token_str,
        meta.layer_range,
    );

    let prompt = chat_template::build_prompt(messages, chat_tmpl.as_deref(), &bos, &eos_str);

    let rid = uuid::Uuid::parse_str(&request_id).unwrap_or_else(|_| uuid::Uuid::new_v4());
    let output = state
        .shared_state
        .model_process_pool
        .generate(
            &requested_mid,
            layer_range,
            prompt,
            params.clone(),
            rid,
            None,
            None,
        )
        .await
        .map_err(ApiError)?;

    let stop_reason = if output.finish_reason == "stop" {
        "end_turn"
    } else {
        "max_tokens"
    };

    let response = MessagesResponse {
        id: request_id,
        response_type: "message",
        role: "assistant",
        content: vec![ResponseContentBlock::Text {
            text: output.content,
        }],
        model,
        stop_reason: Some(stop_reason.into()),
        stop_sequence: None,
        usage: AnthropicUsage {
            input_tokens: output.prompt_tokens,
            output_tokens: output.completion_tokens,
        },
    };

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
        // message_start
        let _ = sse_tx
            .send(AnthropicSseEvent::MessageStart {
                id: rid.clone(),
                model,
            })
            .await;

        // content_block_start
        let _ = sse_tx
            .send(AnthropicSseEvent::ContentBlockStart { index: 0 })
            .await;

        let requested_mid = crate::types::ModelId(model_for_lookup);
        let (chat_tmpl, bos, eos_str, layer_range) =
            match crate::api::openai::get_split_model_meta(&state, &requested_mid) {
                Some(m) => (m.chat_template, m.bos_token, m.eos_token_str, m.layer_range),
                None => {
                    let _ = sse_tx
                        .send(AnthropicSseEvent::ContentBlockStop { index: 0 })
                        .await;
                    let _ = sse_tx
                        .send(AnthropicSseEvent::MessageDelta {
                            stop_reason: "end_turn".into(),
                            output_tokens: 0,
                        })
                        .await;
                    let _ = sse_tx.send(AnthropicSseEvent::MessageStop).await;
                    return;
                }
            };

        // Build prompt using cached metadata
        let prompt = chat_template::build_prompt(&messages, chat_tmpl.as_deref(), &bos, &eos_str);

        let gen_rid = uuid::Uuid::parse_str(&rid).unwrap_or_else(|_| uuid::Uuid::new_v4());

        // Stream tokens via worker subprocess
        let (token_tx, mut token_rx) = tokio::sync::mpsc::channel::<StreamingTokenEvent>(64);

        let pool = state.shared_state.model_process_pool.clone();
        let requested_mid = requested_mid.clone();
        let params = params.clone();
        tokio::spawn(async move {
            let _ = pool
                .generate(
                    &requested_mid,
                    layer_range,
                    prompt,
                    params,
                    gen_rid,
                    None,
                    Some(token_tx),
                )
                .await;
        });

        let mut total_output_tokens = 0u32;
        let mut stop_reason = "max_tokens".to_string();

        while let Some(event) = token_rx.recv().await {
            if let Some(fr) = &event.finish_reason {
                stop_reason = if fr == "stop" {
                    "end_turn".to_string()
                } else {
                    fr.clone()
                };
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

        // content_block_stop + message_delta + message_stop
        let _ = sse_tx
            .send(AnthropicSseEvent::ContentBlockStop { index: 0 })
            .await;
        let _ = sse_tx
            .send(AnthropicSseEvent::MessageDelta {
                stop_reason,
                output_tokens: total_output_tokens,
            })
            .await;
        let _ = sse_tx.send(AnthropicSseEvent::MessageStop).await;
    });

    let stream = tokio_stream::wrappers::ReceiverStream::new(sse_rx).map(move |event| {
        let (event_type, data) = serialize_anthropic_event(&event);
        Ok::<_, Infallible>(Event::default().event(event_type).data(data))
    });

    Ok(Sse::new(stream)
        .keep_alive(
            KeepAlive::new().interval(std::time::Duration::from_secs(SSE_KEEPALIVE_INTERVAL_SECS)),
        )
        .into_response())
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
            ApiError(crate::error::SwarmError::Internal(format!(
                "Cloud provider proxy failed: {e}"
            )))
        })?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        let scrubbed = crate::crypto::scrub_api_keys(&body);
        // Truncate to prevent large error bodies from leaking to API callers
        let truncated = if scrubbed.len() > 512 {
            let safe_end: String = scrubbed.chars().take(500).collect();
            format!("{safe_end}…[truncated]")
        } else {
            scrubbed
        };
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
            let _ = sse_tx
                .send(AnthropicSseEvent::MessageStart {
                    id: format!("msg_{}", uuid::Uuid::new_v4().simple()),
                    model: model_clone,
                })
                .await;
            let _ = sse_tx
                .send(AnthropicSseEvent::ContentBlockStart { index: 0 })
                .await;

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

            let _ = sse_tx
                .send(AnthropicSseEvent::ContentBlockStop { index: 0 })
                .await;
            let _ = sse_tx
                .send(AnthropicSseEvent::MessageDelta {
                    stop_reason: "end_turn".to_string(),
                    output_tokens,
                })
                .await;
            let _ = sse_tx.send(AnthropicSseEvent::MessageStop).await;
        });

        let stream = tokio_stream::wrappers::ReceiverStream::new(sse_rx).map(move |event| {
            let (event_type, data) = serialize_anthropic_event(&event);
            Ok::<_, std::convert::Infallible>(
                axum::response::sse::Event::default()
                    .event(event_type)
                    .data(data),
            )
        });

        return Ok(axum::response::sse::Sse::new(stream)
            .keep_alive(
                axum::response::sse::KeepAlive::new()
                    .interval(std::time::Duration::from_secs(SSE_KEEPALIVE_INTERVAL_SECS)),
            )
            .into_response());
    }

    // Non-streaming: read full JSON response
    let openai_resp: serde_json::Value = resp.json().await.map_err(|e| {
        ApiError(crate::error::SwarmError::Internal(format!(
            "Failed to parse cloud response: {e}"
        )))
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

    let response = MessagesResponse {
        id: format!("msg_{}", uuid::Uuid::new_v4().simple()),
        response_type: "message",
        role: "assistant",
        content: vec![ResponseContentBlock::Text { text: content_text }],
        model: req.model.clone(),
        stop_reason: Some(map_finish_reason(finish).into()),
        stop_sequence: None,
        usage: AnthropicUsage {
            input_tokens,
            output_tokens,
        },
    };

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
            model: "claude-opus-4-6".into(),
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
            model: "claude-opus-4-6".into(),
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
        assert_eq!(resolve_model("claude-opus-4-6"), "claude-opus-4-6");
        assert_eq!(
            resolve_model("anthropic:claude-opus-4-6"),
            "claude-opus-4-6"
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
            "model": "claude-opus-4-6",
            "max_tokens": 1024,
            "system": "You are helpful.",
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
                {"role": "user", "content": "How are you?"}
            ]
        }"#;
        let req: MessagesRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.model, "claude-opus-4-6");
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
            "model": "claude-opus-4-6",
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

    #[test]
    fn response_content_block_serialization() {
        let text = ResponseContentBlock::Text {
            text: "hello".into(),
        };
        let json = serde_json::to_value(&text).unwrap();
        assert_eq!(json["type"], "text");
        assert_eq!(json["text"], "hello");

        let tool = ResponseContentBlock::ToolUse {
            id: "t1".into(),
            name: "Read".into(),
            input: serde_json::json!({"path": "/tmp"}),
        };
        let json = serde_json::to_value(&tool).unwrap();
        assert_eq!(json["type"], "tool_use");
        assert_eq!(json["name"], "Read");

        let think = ResponseContentBlock::Thinking {
            thinking: "hmm".into(),
        };
        let json = serde_json::to_value(&think).unwrap();
        assert_eq!(json["type"], "thinking");
        assert_eq!(json["thinking"], "hmm");
    }
}
