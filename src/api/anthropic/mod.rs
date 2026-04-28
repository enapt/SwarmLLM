use axum::extract::State;
use axum::http::HeaderMap;
use axum::response::IntoResponse;
use axum::Json;

use crate::api::providers;
use crate::api::server::AppState;
use crate::error::ApiError;

mod convert;
mod handlers;
mod proxy;
mod sse;
mod types;

// Re-export public wire-format types so they remain part of the crate's
// external surface (mirrors the pre-split `pub enum` visibility).
pub use types::{AnthropicUsage, MessagesResponse, ResponseContentBlock};

use crate::inference::chat_template;
use convert::{
    is_connectivity_probe, map_finish_reason, resolve_model, to_internal_messages,
    to_sampling_params,
};
#[cfg(test)]
use sse::{serialize_anthropic_event, AnthropicSseEvent};
use types::{AnthropicContent, ContentBlock, MessagesRequest};

/// Hard cap on `max_tokens`. Matches the local sampling-params clamp ceiling
/// (`build_sampling_params` clamps to DEFAULT_MAX_TOKENS=32768). Anything
/// larger lands as a clean 400 at ingress instead of being silently clamped
/// for local inference and forwarded raw to upstream proxies.
const MAX_TOKENS_HARD_CAP: u32 = 32768;

// ---- Handler ----

/// POST /v1/messages — Anthropic Messages API endpoint.
pub async fn messages(
    State(state): State<AppState>,
    headers: HeaderMap,
    crate::api::server::JsonBody(req): crate::api::server::JsonBody<MessagesRequest>,
) -> Result<axum::response::Response, ApiError> {
    // Capture Anthropic beta/version headers for forwarding on the proxy path.
    // `anthropic-beta` is the big one: Claude Code + SDK users enable features
    // like advanced-tool-use-*, context-1m-*, token-efficient-tools-*,
    // code-execution-* through this header. Silently dropping it means those
    // features degrade to vanilla 2023-06-01 behaviour without any error.
    let proxy_beta = headers
        .get("anthropic-beta")
        .and_then(|v| v.to_str().ok())
        .map(str::to_owned);
    let proxy_version = headers
        .get("anthropic-version")
        .and_then(|v| v.to_str().ok())
        .map(str::to_owned);
    super::validate_common_params(
        req.model.len(),
        req.messages.len(),
        req.temperature.unwrap_or(1.0).into(),
    )?;

    // Cap max_tokens at the same upper bound the local sampling clamp uses
    // (`build_sampling_params` clamps to DEFAULT_MAX_TOKENS=32768). Without
    // this, callers can send max_tokens=u32::MAX which (a) confuses upstream
    // proxy targets with a value larger than the model context, and
    // (b) lands raw in our DIAG log fields. The clamp still protects local
    // inference; this just produces a clean 400 at ingress.
    if req.max_tokens == 0 || req.max_tokens > MAX_TOKENS_HARD_CAP {
        return Err(ApiError(crate::error::SwarmError::Validation(format!(
            "max_tokens must be 1..={MAX_TOKENS_HARD_CAP}"
        ))));
    }

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
                    // Server-tool variants contribute their serialized size —
                    // approximates the echo-back weight of a multi-turn
                    // Claude Code conversation with server-tool results.
                    ContentBlock::ServerToolUse { input, name, id } => {
                        name.len() + id.len() + input.to_string().len()
                    }
                    ContentBlock::WebSearchToolResult { content, .. }
                    | ContentBlock::CodeExecutionToolResult { content, .. }
                    | ContentBlock::BashToolResult { content, .. }
                    | ContentBlock::TextEditorToolResult { content, .. } => {
                        content.to_string().len()
                    }
                    ContentBlock::Document {
                        source, citations, ..
                    } => {
                        source.to_string().len()
                            + citations.as_ref().map(|c| c.to_string().len()).unwrap_or(0)
                    }
                    ContentBlock::SearchResult {
                        source, citations, ..
                    } => {
                        source.as_ref().map(|s| s.to_string().len()).unwrap_or(0)
                            + citations.as_ref().map(|c| c.to_string().len()).unwrap_or(0)
                    }
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
            return handlers::anthropic_split_stream(
                &state,
                internal_messages,
                sampling_params,
                request_id,
                model,
            )
            .await;
        } else {
            return handlers::anthropic_split_non_stream(
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
                return handlers::anthropic_stream(
                    router_tx.clone(),
                    internal_messages,
                    sampling_params,
                    request_id,
                    model,
                )
                .await;
            } else {
                return handlers::anthropic_non_stream(
                    router_tx.clone(),
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
        // Use the same ProxyMessagesRequest serializer as the cloud path so
        // tool_use / tool_result / thinking blocks survive the subprocess
        // hop. The previous hand-serialization replaced every non-text
        // ContentBlock with a "[non-text content]" placeholder, which broke
        // multi-turn function-calling conversations because the assistant's
        // tool_use blocks (and the user's tool_result blocks) were stripped
        // before the subprocess ever saw them.
        let body = serde_json::to_value(&proxy::ProxyMessagesRequest {
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
            extras: &req.extras,
        })
        .map_err(|e| {
            ApiError(crate::error::SwarmError::Internal(format!(
                "serialize request for proxy: {e}"
            )))
        })?;
        return crate::api::claude_sub::proxy_via_subprocess_anthropic(&sub_config, &body).await;
    }

    // Claude models → Anthropic cloud API (full pass-through, preserves tools/thinking)
    if lower_model.starts_with("claude") {
        let config = state.shared_state.metrics.providers_config.read().await;
        if let Some(ref entry) = config.anthropic {
            let api_key = entry.api_key.clone();
            drop(config);

            tracing::debug!(model = %req.model, "DIAG: anthropic proxying to cloud API");
            let body = serde_json::to_value(&proxy::ProxyMessagesRequest {
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
                extras: &req.extras,
            })
            .map_err(|e| {
                ApiError(crate::error::SwarmError::Internal(format!(
                    "serialize request for proxy: {e}"
                )))
            })?;

            return providers::proxy_to_anthropic(
                &api_key,
                &body,
                req.stream,
                proxy_beta.as_deref(),
                proxy_version.as_deref(),
            )
            .await;
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
            return handlers::anthropic_to_openai_proxy(&req, &provider_url, &provider_key).await;
        }
    }

    // No local model, no cloud provider configured
    Err(ApiError(crate::error::SwarmError::NoModelLoaded))
}

#[cfg(test)]
mod tests {
    use super::types::{AnthropicMessage, SystemBlock, SystemContent};
    use super::*;
    use crate::types::Role;

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
            extras: std::collections::HashMap::new(),
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
            extras: std::collections::HashMap::new(),
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
            extras: std::collections::HashMap::new(),
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
            extras: std::collections::HashMap::new(),
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
            extras: std::collections::HashMap::new(),
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
            extras: std::collections::HashMap::new(),
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
            extras: std::collections::HashMap::new(),
        };
        let msgs = to_internal_messages(&req);
        assert_eq!(msgs.len(), 1);
        assert!(msgs[0].content.contains("I'll read it."));
        assert!(msgs[0].content.contains("[Tool call: Read("));
    }

    #[test]
    fn unknown_anthropic_fields_preserved_for_proxy() {
        // Caller-supplied fields our struct doesn't model (service_tier,
        // container, hypothetical future knob) must round-trip through the
        // ProxyMessagesRequest serializer verbatim. Regression for the audit
        // finding that these were silently dropped.
        let json = r#"{
            "model": "claude-opus-4-7",
            "max_tokens": 128,
            "messages": [{"role": "user", "content": "Hi"}],
            "service_tier": "standard_only",
            "container": "container_abc123",
            "extra_future_field": {"nested": true}
        }"#;
        let req: MessagesRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.extras.get("service_tier").unwrap(), "standard_only");
        assert_eq!(req.extras.get("container").unwrap(), "container_abc123");
        assert!(req.extras.contains_key("extra_future_field"));

        let proxy = proxy::ProxyMessagesRequest {
            model: &req.model,
            max_tokens: req.max_tokens,
            messages: &req.messages,
            system: &req.system,
            stream: false,
            temperature: req.temperature,
            top_p: req.top_p,
            top_k: req.top_k,
            stop_sequences: &req.stop_sequences,
            tools: &req.tools,
            tool_choice: &req.tool_choice,
            metadata: &req.metadata,
            thinking: &req.thinking,
            extras: &req.extras,
        };
        let v = serde_json::to_value(&proxy).unwrap();
        assert_eq!(v["service_tier"], "standard_only");
        assert_eq!(v["container"], "container_abc123");
        assert_eq!(v["extra_future_field"]["nested"], true);
    }
}
