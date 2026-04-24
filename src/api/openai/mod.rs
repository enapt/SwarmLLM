use axum::extract::State;
use axum::response::IntoResponse;
use axum::Json;

use crate::api::server::AppState;
use crate::error::ApiError;
use crate::inference::chat_template;

mod peer_forward;
mod resolver;
pub mod responses;
mod streaming;
mod types;

pub use resolver::{
    all_shards_available, get_split_model_meta, model_matches_loaded,
    resolve_loaded_model_registry_id, resolve_model_for_inference, resolve_model_name,
    SplitModelMeta,
};
pub use streaming::{run_split_generate, spawn_split_stream, submit_stream_to_router};
pub use types::*;

use peer_forward::forward_to_peer;
use resolver::find_peer_with_model;
use streaming::{
    dispatch_inference, router_inference, split_non_stream_response, split_stream_response,
    stream_response,
};
#[cfg(test)]
use types::format_tool_system_prompt;

use super::DEFAULT_TOP_K;

/// Maximum cold-start wait time before returning 503 (seconds).
const COLD_START_WAIT_SECS: u32 = 10;

/// Maximum number of concurrent requests in the cold-start polling loop.
/// Prevents unbounded task pile-up when many requests arrive for unavailable models.
const MAX_COLD_START_SLOTS: usize = 5;
static COLD_START_SEMAPHORE: std::sync::LazyLock<tokio::sync::Semaphore> =
    std::sync::LazyLock::new(|| tokio::sync::Semaphore::new(MAX_COLD_START_SLOTS));

// ---- Handlers ----

/// POST /v1/chat/completions
/// Validate and sanitize a chat completion request.
/// Checks session_id, model, messages, temperature, content sizes, tools, stop sequences.
/// Binds session_id to caller's API key to prevent cross-user KV-cache hijacking.
fn validate_chat_request(
    req: &mut ChatCompletionRequest,
    headers: &axum::http::HeaderMap,
) -> Result<(), ApiError> {
    // Validate session_id length to prevent memory abuse
    if let Some(ref sid) = req.session_id {
        if sid.len() > 256 {
            return Err(ApiError(crate::error::SwarmError::Validation(
                "session_id too long (max 256 chars)".into(),
            )));
        }
    }

    // Bind session_id to the caller's API key so different users cannot
    // hijack each other's KV-cache sessions by guessing session IDs.
    if let Some(ref mut sid) = req.session_id {
        let api_key = super::extract_bearer_token(headers);
        let key_hash = &blake3::hash(api_key.as_bytes()).to_hex()[..16];
        *sid = format!("{}:{}", key_hash, sid);
    }

    super::validate_common_params(req.model.len(), req.messages.len(), req.temperature.into())?;

    // SEC: Cap individual message content size and total prompt size
    super::validate_content_size(req.messages.iter().map(|msg| {
        match &msg.content {
            MessageContent::Text(s) => s.len(),
            MessageContent::Parts(parts) => parts
                .iter()
                .map(|p| match p {
                    ContentPart::Text { text } => text.len(),
                    ContentPart::ImageUrl { image_url } => image_url.url.len(),
                })
                .sum(),
        }
    }))?;

    if let Some(ref tools) = req.tools {
        super::validate_tools(
            tools,
            |t| Some(t.function.name.as_str()),
            |t| t.function.description.as_deref(),
            |t| t.function.parameters.as_ref().map(|p| p.to_string().len()),
        )?;
    }

    if let Some(ref adapter) = req.lora_adapter {
        if adapter.len() > 256 {
            return Err(ApiError(crate::error::SwarmError::Validation(
                "lora_adapter name too long (max 256 chars)".into(),
            )));
        }
        if adapter.contains("..") || adapter.contains('/') || adapter.contains('\\') {
            return Err(ApiError(crate::error::SwarmError::Validation(
                "lora_adapter contains invalid characters".into(),
            )));
        }
    }

    match &req.stop {
        Some(StopSequence::Multiple(v)) => {
            super::validate_stop_sequences(v)?;
        }
        Some(StopSequence::Single(s)) => {
            super::validate_stop_sequences(std::slice::from_ref(s))?;
        }
        None => {}
    }

    Ok(())
}

/// Try proxying a chat completion request to a configured cloud provider.
async fn try_cloud_proxy(
    state: &AppState,
    req: &ChatCompletionRequest,
) -> Result<Option<axum::response::Response>, ApiError> {
    // Claude subscription: proxy through local CLI subprocess (higher priority than API key)
    #[cfg(feature = "claude-subscription")]
    if let Some(sub_config) =
        crate::api::claude_sub::try_get_claude_subscription(state, &req.model).await
    {
        tracing::info!(model = %req.model, "DIAG: openai proxying via claude subscription subprocess");
        let body = serde_json::to_value(req).map_err(|e| {
            ApiError(crate::error::SwarmError::Validation(format!(
                "serialize request: {e}"
            )))
        })?;
        return crate::api::claude_sub::proxy_via_subprocess_openai(&sub_config, &body)
            .await
            .map(Some);
    }

    let body = serde_json::to_value(req).map_err(|e| {
        ApiError(crate::error::SwarmError::Validation(format!(
            "serialize request: {e}"
        )))
    })?;
    crate::api::providers::try_proxy_openai(state, &body, req.stream).await
}

pub async fn chat_completions(
    State(state): State<AppState>,
    headers: axum::http::HeaderMap,
    crate::api::server::JsonBody(mut req): crate::api::server::JsonBody<ChatCompletionRequest>,
) -> Result<axum::response::Response, ApiError> {
    validate_chat_request(&mut req, &headers)?;

    // Convert API messages to internal format (decode base64 images if present)
    let internal_messages = req.to_internal_messages().map_err(ApiError)?;

    let request_id = format!("swarm-{}", uuid::Uuid::new_v4().simple());
    let created = chrono::Utc::now().timestamp();

    // Track requests made by this node
    super::increment_requests_made(&state.shared_state);

    // Emit activity event for inference request
    {
        let mname = state
            .shared_state
            .model_registry
            .get_manifest(&crate::types::ModelId(req.model.clone()))
            .map(|m| m.name.clone());
        let display = mname.as_deref().unwrap_or(&req.model);
        let max_tok = req.max_tokens;
        let msg_count = req.messages.len();
        let prompt_preview: String = req
            .messages
            .last()
            .map(|m| {
                let content = match &m.content {
                    MessageContent::Text(s) => s.clone(),
                    MessageContent::Parts(parts) => parts
                        .iter()
                        .filter_map(|p| match p {
                            ContentPart::Text { text } => Some(text.as_str()),
                            _ => None,
                        })
                        .collect::<Vec<_>>()
                        .join(" "),
                };
                if content.len() > 60 {
                    format!(
                        "{}...",
                        &content[..content
                            .char_indices()
                            .take_while(|(i, _)| *i < 57)
                            .last()
                            .map(|(i, c)| i + c.len_utf8())
                            .unwrap_or(57)]
                    )
                } else {
                    content
                }
            })
            .unwrap_or_default();
        state.shared_state.emit_activity(
            crate::daemon::state::ActivityEvent::new(
                "inference",
                "inference_request",
                format!(
                    "Inference request on {} — {} message{}, max {} tokens{}",
                    display,
                    msg_count,
                    if msg_count != 1 { "s" } else { "" },
                    max_tok,
                    if !prompt_preview.is_empty() {
                        format!(": \"{}\"", prompt_preview)
                    } else {
                        String::new()
                    }
                ),
            )
            .with_model(req.model.clone())
            .with_detail_num(max_tok as i64),
        );
    }

    // Resolve "auto" alias and display-name → registry-ID mapping.
    {
        let resolved = resolve_model_for_inference(&state, &req.model).await;
        if resolved != req.model {
            tracing::info!(original = %req.model, resolved_model = %resolved, "Resolved model alias");
            req.model = resolved;
        }
    }

    let image_count: usize = internal_messages.iter().map(|m| m.images.len()).sum();
    tracing::info!(
        request_id = %request_id,
        model = %req.model,
        messages = internal_messages.len(),
        images = image_count,
        stream = req.stream,
        "Chat completion request"
    );

    // Get model name — try loaded_model_info cache first, then manifest registry.
    let model_name = resolve_model_name(&state, &req.model).await;

    // No local full-model executor — use distributed inference or forward.
    // Nodes are NOT required to have all shards. Any node can initiate inference
    // as long as the network collectively covers all layers.
    // The `x-swarm-forwarded` header prevents infinite forwarding loops between nodes.
    // Only trust this header from internal requests (authenticated with internal token).
    // Middleware already validates that x-swarm-forwarded only passes from known peer IPs
    // or loopback with valid internal token, so we just check presence here.
    let is_forwarded = headers.get("x-swarm-forwarded").is_some();

    if model_name.is_none() {
        // Priority 1: Check if all layers are covered across the network for
        // distributed inference. The local node may have zero, some, or all shards —
        // it doesn't matter as long as the network covers every layer.
        if all_shards_available(&state, &req.model) {
            tracing::info!(
                request_id = %request_id,
                model = %req.model,
                stream = req.stream,
                "All layers covered across network — using distributed inference"
            );

            if let Some(router_tx) = &state.router_tx {
                return dispatch_inference(
                    router_tx.clone(),
                    &state,
                    &req,
                    internal_messages.clone(),
                    request_id,
                    created,
                )
                .await;
            } else {
                return Err(ApiError(crate::error::SwarmError::NoModelLoaded));
            }
        }

        // Priority 2: Forward to a peer that hosts shards for this model.
        // That peer can handle inference locally or build its own pipeline.
        if !is_forwarded {
            if let Some(peer_url) = find_peer_with_model(&state, &req.model) {
                tracing::info!(
                    request_id = %request_id,
                    peer_url = %peer_url,
                    "Forwarding request to peer"
                );
                return forward_to_peer(&peer_url, &req, req.stream).await;
            }
        }

        // Cloud provider fast-path: if the model matches a configured cloud provider,
        // route immediately without cold-start waiting. Cloud models are never local.
        if let Some(response) = try_cloud_proxy(&state, &req).await? {
            return Ok(response);
        }

        // Fast-reject: if the model has no manifest but other models ARE registered,
        // the user likely mistyped. Return ModelNotAvailable immediately instead of
        // wasting 10 seconds on cold-start polling. When NO models are registered,
        // fall through to the cold-start wait (node may still be starting up).
        {
            let model_id = crate::types::ModelId(req.model.clone());
            let registry = &state.shared_state.model_registry;
            if registry.get_manifest(&model_id).is_none() && !registry.models().is_empty() {
                return Err(ApiError(registry.model_not_found_error(&model_id)));
            }
        }

        // Cold-start wait: shard announcements may still be propagating.
        // Poll for up to 10 seconds before giving up.
        // Semaphore prevents unbounded pile-up under burst traffic.
        let _cold_permit = match COLD_START_SEMAPHORE.try_acquire() {
            Ok(permit) => permit,
            Err(_) => {
                tracing::warn!(
                    request_id = %request_id,
                    model = %req.model,
                    "Cold-start wait slots full — returning 503 immediately"
                );
                return Err(ApiError(crate::error::SwarmError::NoModelLoaded));
            }
        };
        let max_polls = COLD_START_WAIT_SECS * 2; // 500ms intervals
        for attempt in 1..=max_polls {
            tokio::time::sleep(std::time::Duration::from_millis(500)).await;
            // Re-check distributed inference availability
            if all_shards_available(&state, &req.model) {
                tracing::info!(
                    request_id = %request_id,
                    model = %req.model,
                    wait_ms = attempt * 500,
                    "Model became available after cold-start wait"
                );
                if let Some(router_tx) = &state.router_tx {
                    return dispatch_inference(
                        router_tx.clone(),
                        &state,
                        &req,
                        internal_messages.clone(),
                        request_id,
                        created,
                    )
                    .await;
                }
                break;
            }
            // Re-check peer forwarding
            if !is_forwarded {
                if let Some(peer_url) = find_peer_with_model(&state, &req.model) {
                    tracing::info!(
                        request_id = %request_id,
                        peer_url = %peer_url,
                        wait_ms = attempt * 500,
                        "Found peer after cold-start wait"
                    );
                    return forward_to_peer(&peer_url, &req, req.stream).await;
                }
            }
        }

        // Cloud provider fallback: proxy to configured cloud provider if model matches
        if let Some(response) = try_cloud_proxy(&state, &req).await? {
            return Ok(response);
        }

        return Err(ApiError(crate::error::SwarmError::NoModelLoaded));
    }

    let model_name =
        model_name.expect("model_name guaranteed Some — None case returns early above");
    let params = req.to_sampling_params();

    // Fast path: if we have a complete local split model (all layers), generate directly.
    // This avoids the distributed pipeline overhead (per-token segment coordination,
    // activation serialization, mutex per token). ~5-10x faster for local inference.
    // Uses the pre-computed is_complete flag — no model mutex needed.
    // Look up by the REQUESTED model ID, not just the first entry.
    // NOTE: prompt is built inside split_*_response using the model's own chat template,
    // avoiding template mismatch (e.g. `<|assistant|>` prefix leak) and the unnecessary
    // loaded_model_info.read().await before the split path.
    let requested_mid = crate::types::ModelId(req.model.clone());
    let has_local_split_model = state.shared_state.has_complete_split_model(&requested_mid);

    if has_local_split_model {
        if req.stream {
            return Ok(split_stream_response(
                state,
                request_id,
                created,
                model_name,
                internal_messages.clone(),
                params,
                requested_mid.clone(),
            )
            .await
            .into_response());
        } else {
            return split_non_stream_response(
                state,
                request_id,
                created,
                model_name,
                internal_messages.clone(),
                params,
                requested_mid.clone(),
            )
            .await;
        }
    }

    // Build prompt for non-split paths (distributed inference + direct executor).
    // Uses loaded_model_info first; falls back to GGUF header on disk for
    // distributed-only nodes that have no local model but do have the probe.
    let prompt = {
        let (tmpl, bos, eos) = super::resolve_chat_template(&state, &req.model).await;
        chat_template::build_prompt(&internal_messages, tmpl.as_deref(), &bos, &eos)
    };

    // Distributed inference: network covers all layers across multiple nodes.
    let peers_have_shards = all_shards_available(&state, &req.model)
        || state.shared_state.config.inference.shard_range.is_some();
    if peers_have_shards {
        if let Some(router_tx) = &state.router_tx {
            return dispatch_inference(
                router_tx.clone(),
                &state,
                &req,
                internal_messages.clone(),
                request_id,
                created,
            )
            .await;
        }
    }

    if req.stream {
        // Streaming: use direct executor path for real token-by-token SSE
        Ok(
            stream_response(state, request_id, created, model_name, prompt, params)
                .await
                .into_response(),
        )
    } else if let Some(router_tx) = &state.router_tx {
        // Non-streaming: route through InferenceRouter for priority queueing
        router_inference(
            router_tx.clone(),
            &req,
            internal_messages,
            request_id,
            created,
        )
        .await
    } else {
        Err(crate::error::ApiError(
            crate::error::SwarmError::ServiceUnavailable(
                "Inference router not available".to_string(),
            ),
        ))
    }
}

/// GET /v1/models
///
/// Lists models usable for inference. A model is usable when all its layers
/// are covered by at least one node in the network — no single node needs
/// the full shard set. Models still propagating across the network (some
/// layers uncovered) are excluded here but visible in the admin dashboard.
pub async fn list_models(State(state): State<AppState>) -> Json<ModelListResponse> {
    let mut data = vec![];
    let mut seen = std::collections::HashSet::new();

    // Use cached model info (lock-free, no executor contention)
    if let Some(info) = state.shared_state.loaded_model_info.read().await.as_ref() {
        seen.insert(info.name.clone());
        let slug = crate::types::slugify_model_name(&info.name);
        seen.insert(slug.clone());

        // Find the registry manifest for this model so we can use its canonical ID
        // and mark it as seen (prevents duplicates in section 2).
        let manifest = state
            .shared_state
            .model_registry
            .resolve_manifest_by_name(&info.name);

        let model_id = if let Some(ref m) = manifest {
            seen.insert(m.id.0.clone());
            m.id.0.clone()
        } else {
            slug
        };

        data.push(ModelInfo {
            id: model_id,
            object: "model",
            created: chrono::Utc::now().timestamp(),
            owned_by: "local".into(),
        });
    }

    // Include models from the registry if all layers are covered network-wide
    for manifest in state.shared_state.model_registry.models() {
        let id = manifest.id.0.clone();
        if seen.contains(&id) {
            continue;
        }
        if all_shards_available(&state, &id) {
            seen.insert(id.clone());
            data.push(ModelInfo {
                id,
                object: "model",
                created: manifest.publish_date.timestamp(),
                owned_by: "network".into(),
            });
        }
    }

    Json(ModelListResponse {
        object: "list",
        data,
    })
}

/// POST /v1/embeddings — OpenAI-compatible embeddings endpoint.
///
/// Not available with subprocess inference (Phase 17) — models run in isolated
/// worker processes without in-process tensor access. Use a dedicated embedding
/// model or cloud provider instead.
pub async fn embeddings(
    State(_state): State<AppState>,
    crate::api::server::JsonBody(_req): crate::api::server::JsonBody<EmbeddingRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    Err(ApiError(crate::error::SwarmError::ServiceUnavailable(
        "Embeddings API not available with subprocess inference. Use a dedicated embedding model or provider.".into(),
    )))
}

/// GET /v1/status — SwarmLLM extension endpoint
pub async fn status(State(state): State<AppState>) -> Json<serde_json::Value> {
    let info = state.shared_state.loaded_model_info.read().await;
    let local_model = info.is_some();
    let model_name = info.as_ref().map(|i| i.name.clone()).unwrap_or_default();
    drop(info);

    // Count network-available models from peers
    let mut network_models = Vec::new();
    for entry in state.shared_state.peer_registry.iter() {
        if let Some(ref cap) = entry.value().capability {
            for shard in &cap.hosted_shards {
                network_models.push(shard.model_id.0.clone());
            }
        }
    }
    network_models.sort();
    network_models.dedup();

    Json(serde_json::json!({
        "status": "ok",
        "version": env!("CARGO_PKG_VERSION"),
        "model_loaded": local_model,
        "model_name": model_name,
        "network_models": network_models,
        "peers": state.shared_state.peer_registry.len(),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tool_call_request_deserializes() {
        let json = r#"{
            "model": "test",
            "messages": [{"role": "user", "content": "What's the weather?"}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather",
                    "parameters": {"type": "object", "properties": {"location": {"type": "string"}}}
                }
            }],
            "tool_choice": "auto"
        }"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert!(req.tools.is_some());
        let tools = req.tools.unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].function.name, "get_weather");
        assert_eq!(req.tool_choice, Some(serde_json::json!("auto")));
    }

    #[test]
    fn tool_role_message_deserializes() {
        let json = r#"{
            "role": "tool",
            "content": "{\"temperature\": 72}",
            "tool_call_id": "call_abc123"
        }"#;
        let msg: ApiChatMessage = serde_json::from_str(json).unwrap();
        assert!(matches!(msg.role, crate::types::Role::Tool));
        assert_eq!(msg.tool_call_id, Some("call_abc123".into()));
    }

    #[test]
    fn assistant_tool_calls_message_deserializes() {
        let json = r#"{
            "role": "assistant",
            "content": null,
            "tool_calls": [{
                "id": "call_abc123",
                "type": "function",
                "function": {"name": "get_weather", "arguments": "{\"location\":\"NYC\"}"}
            }]
        }"#;
        let msg: ApiChatMessage = serde_json::from_str(json).unwrap();
        let tc = msg.tool_calls.unwrap();
        assert_eq!(tc.len(), 1);
        assert_eq!(tc[0].id, "call_abc123");
        assert_eq!(tc[0].function.name, "get_weather");
    }

    #[test]
    fn logprobs_request_fields() {
        let json = r#"{
            "model": "test",
            "messages": [{"role": "user", "content": "hi"}],
            "logprobs": true,
            "top_logprobs": 5
        }"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert!(req.logprobs);
        assert_eq!(req.top_logprobs, Some(5));
        let params = req.to_sampling_params();
        assert!(params.logprobs);
        assert_eq!(params.top_logprobs, 5);
    }

    #[test]
    fn logprobs_response_serializes() {
        let choice = ChatChoice {
            index: 0,
            message: ChatMessageResponse {
                role: "assistant".into(),
                content: Some("hello".into()),
                tool_calls: None,
            },
            finish_reason: "stop".into(),
            logprobs: Some(ChoiceLogProbs {
                content: vec![TokenLogProb {
                    token: "hello".into(),
                    logprob: -0.5,
                    bytes: None,
                    top_logprobs: vec![
                        TopLogProb {
                            token: "hello".into(),
                            logprob: -0.5,
                            bytes: None,
                        },
                        TopLogProb {
                            token: "hi".into(),
                            logprob: -1.2,
                            bytes: None,
                        },
                    ],
                }],
            }),
        };
        let json = serde_json::to_string(&choice).unwrap();
        assert!(json.contains("\"logprobs\""));
        assert!(json.contains("\"top_logprobs\""));
        assert!(json.contains("-0.5"));
    }

    #[test]
    fn tool_calls_response_serializes() {
        let choice = ChatChoice {
            index: 0,
            message: ChatMessageResponse {
                role: "assistant".into(),
                content: None,
                tool_calls: Some(vec![ToolCall {
                    id: "call_123".into(),
                    tool_type: "function".into(),
                    function: FunctionCall {
                        name: "get_weather".into(),
                        arguments: r#"{"location":"NYC"}"#.into(),
                    },
                }]),
            },
            finish_reason: "tool_calls".into(),
            logprobs: None,
        };
        let json = serde_json::to_string(&choice).unwrap();
        assert!(json.contains("\"tool_calls\""));
        assert!(json.contains("\"call_123\""));
        assert!(json.contains("\"tool_calls\"")); // finish_reason
                                                  // content should be absent when None
        assert!(!json.contains("\"content\""));
    }

    #[test]
    fn request_without_tools_works() {
        let json = r#"{
            "model": "test",
            "messages": [{"role": "user", "content": "hi"}]
        }"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert!(req.tools.is_none());
        assert!(!req.logprobs);
        assert!(req.top_logprobs.is_none());
    }

    #[test]
    fn format_tool_system_prompt_output() {
        let tools = vec![ToolDefinition {
            tool_type: "function".into(),
            function: FunctionDefinition {
                name: "get_weather".into(),
                description: Some("Get weather info".into()),
                parameters: Some(serde_json::json!({"type": "object"})),
            },
        }];
        let prompt = format_tool_system_prompt(&tools);
        assert!(prompt.contains("get_weather"));
        assert!(prompt.contains("Get weather info"));
        assert!(prompt.contains("Parameters:"));
    }

    #[test]
    fn stream_delta_tool_calls_serializes() {
        let delta = Delta {
            role: Some("assistant".into()),
            content: None,
            tool_calls: Some(vec![StreamToolCall {
                index: 0,
                id: Some("call_1".into()),
                tool_type: Some("function".into()),
                function: Some(StreamFunctionCall {
                    name: Some("get_weather".into()),
                    arguments: Some("{".into()),
                }),
            }]),
        };
        let json = serde_json::to_string(&delta).unwrap();
        assert!(json.contains("\"tool_calls\""));
        assert!(json.contains("\"call_1\""));
    }

    #[test]
    fn response_format_json_object_deserializes() {
        let json = r#"{
            "model": "test",
            "messages": [{"role": "user", "content": "hi"}],
            "response_format": {"type": "json_object"}
        }"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert!(matches!(
            req.response_format,
            Some(ResponseFormat::JsonObject)
        ));
    }

    #[test]
    fn response_format_json_schema_deserializes() {
        let json = r#"{
            "model": "test",
            "messages": [{"role": "user", "content": "hi"}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "person",
                    "schema": {"type": "object", "properties": {"name": {"type": "string"}}},
                    "strict": true
                }
            }
        }"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        match req.response_format {
            Some(ResponseFormat::JsonSchema { ref json_schema }) => {
                assert_eq!(json_schema.name, "person");
                assert!(json_schema.strict);
            }
            _ => panic!("expected JsonSchema"),
        }
    }

    #[test]
    fn response_format_text_deserializes() {
        let json = r#"{
            "model": "test",
            "messages": [{"role": "user", "content": "hi"}],
            "response_format": {"type": "text"}
        }"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert!(matches!(req.response_format, Some(ResponseFormat::Text)));
    }

    #[test]
    fn response_format_absent_is_none() {
        let json = r#"{
            "model": "test",
            "messages": [{"role": "user", "content": "hi"}]
        }"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert!(req.response_format.is_none());
    }

    #[test]
    fn cache_control_request_deserializes() {
        let json = r#"{
            "model": "test",
            "messages": [{"role": "user", "content": "hi"}],
            "cache_control": {"type": "ephemeral", "prefix_messages": 2}
        }"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        let cc = req.cache_control.unwrap();
        assert_eq!(cc.r#type.as_deref(), Some("ephemeral"));
        assert_eq!(cc.prefix_messages, Some(2));
    }

    #[test]
    fn cache_control_persistent_type() {
        let json = r#"{
            "model": "test",
            "messages": [{"role": "user", "content": "hi"}],
            "cache_control": {"type": "persistent"}
        }"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        let cc = req.cache_control.unwrap();
        assert_eq!(cc.r#type.as_deref(), Some("persistent"));
        assert!(cc.prefix_messages.is_none());
    }

    #[test]
    fn cache_control_absent_is_none() {
        let json = r#"{
            "model": "test",
            "messages": [{"role": "user", "content": "hi"}]
        }"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert!(req.cache_control.is_none());
    }

    #[test]
    fn per_message_cache_control_deserializes() {
        let json = r#"{
            "role": "system",
            "content": "You are helpful",
            "cache_control": {"type": "ephemeral"}
        }"#;
        let msg: ApiChatMessage = serde_json::from_str(json).unwrap();
        let cc = msg.cache_control.unwrap();
        assert_eq!(cc.r#type.as_deref(), Some("ephemeral"));
    }

    #[test]
    fn usage_cache_stats_serialize_when_present() {
        let usage = Usage {
            prompt_tokens: 100,
            completion_tokens: 50,
            total_tokens: 150,
            cache_creation_input_tokens: Some(80),
            cache_read_input_tokens: Some(0),
        };
        let json = serde_json::to_string(&usage).unwrap();
        assert!(json.contains("\"cache_creation_input_tokens\":80"));
        assert!(json.contains("\"cache_read_input_tokens\":0"));
    }

    #[test]
    fn usage_cache_stats_omitted_when_none() {
        let usage = Usage {
            prompt_tokens: 100,
            completion_tokens: 50,
            total_tokens: 150,
            cache_creation_input_tokens: None,
            cache_read_input_tokens: None,
        };
        let json = serde_json::to_string(&usage).unwrap();
        assert!(!json.contains("cache_creation_input_tokens"));
        assert!(!json.contains("cache_read_input_tokens"));
    }

    #[test]
    fn json_schema_injects_system_prompt() {
        let json = r#"{
            "model": "test",
            "messages": [{"role": "user", "content": "hi"}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "result",
                    "schema": {"type": "object"}
                }
            }
        }"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        let msgs = req.to_internal_messages().unwrap();
        // Should have 2 messages: injected system + original user
        assert_eq!(msgs.len(), 2);
        assert!(matches!(msgs[0].role, crate::types::Role::System));
        assert!(msgs[0].content.contains("valid JSON"));
        assert!(msgs[0].content.contains("result"));
    }

    #[test]
    fn max_completion_tokens_alias_parses() {
        let json = r#"{
            "model": "o3-mini",
            "messages": [{"role": "user", "content": "hi"}],
            "max_completion_tokens": 5000
        }"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.max_tokens, 5000);
    }

    #[test]
    fn unknown_openai_fields_preserved_for_proxy() {
        // Reasoning-model / Responses-era fields SwarmLLM doesn't parse:
        // reasoning_effort, service_tier, seed, store, metadata,
        // parallel_tool_calls, stream_options, prediction.
        // They should survive the deserialize → serialize round-trip so
        // the cloud-proxy path doesn't silently drop them.
        let json = r#"{
            "model": "gpt-5",
            "messages": [{"role": "user", "content": "hi"}],
            "reasoning_effort": "high",
            "service_tier": "priority",
            "seed": 42,
            "metadata": {"run": "eval-batch-1"},
            "parallel_tool_calls": false,
            "stream_options": {"include_usage": true}
        }"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        let round_tripped = serde_json::to_value(&req).unwrap();
        assert_eq!(round_tripped["reasoning_effort"], "high");
        assert_eq!(round_tripped["service_tier"], "priority");
        assert_eq!(round_tripped["seed"], 42);
        assert_eq!(round_tripped["metadata"]["run"], "eval-batch-1");
        assert_eq!(round_tripped["parallel_tool_calls"], false);
        assert_eq!(round_tripped["stream_options"]["include_usage"], true);
    }
}
