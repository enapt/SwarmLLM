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
use convert::{is_connectivity_probe, resolve_model, to_internal_messages, to_sampling_params};
#[cfg(test)]
use sse::{serialize_anthropic_event, AnthropicSseEvent};
use types::{AnthropicContent, ContentBlock, MessagesRequest};

/// Hard cap on `max_tokens`. Matches the local sampling-params clamp ceiling
/// (`build_sampling_params` clamps to DEFAULT_MAX_TOKENS=32768). Anything
/// larger lands as a clean 400 at ingress instead of being silently clamped
/// for local inference and forwarded raw to upstream proxies.
const MAX_TOKENS_HARD_CAP: u32 = 32768;

/// Largest error body this layer will buffer before giving up and passing the
/// response through untouched. Error envelopes are a few hundred bytes; the cap
/// only exists so a pathological body cannot be held in memory.
const MAX_ERROR_BODY_BYTES: usize = 64 * 1024;

/// The one route whose failures wear Anthropic's envelope. Kept next to the
/// layer so the guard and the router cannot disagree about which path that is.
const ANTHROPIC_MESSAGES_PATH: &str = "/v1/messages";

/// Rewrite a failure on `/v1/messages` into the envelope Anthropic clients
/// actually parse.
///
/// Our canonical envelope is OpenAI-shaped — `{"error": {...}}` with an
/// OpenAI-flavoured `type` — because that is the older surface and
/// [`crate::error::classify_error`] is its single source of truth. Anthropic's
/// is different in two ways that matter to a client:
///
/// ```text
/// {"type": "error", "error": {"type": "not_found_error", "message": "..."}}
/// ```
///
/// The top-level `"type": "error"` is the discriminator SDKs branch on, and the
/// inner `type` must come from Anthropic's own set — `invalid_request_error`,
/// `authentication_error`, `permission_error`, `not_found_error`,
/// `request_too_large`, `rate_limit_error`, `overloaded_error`, `api_error`.
///
/// [`sse::anthropic_error_type`] already did this translation, but **only on the
/// streaming path** (gotcha #302). Measured against the live node 2026-08-18: a
/// prompt-privacy refusal came back as `{"error": {"type":
/// "prompt_privacy_error", ...}}` with no top-level `type` — a made-up type in
/// the wrong envelope, which is the exact failure that gotcha warns about, on
/// the sibling path. This is the codebase's recurring "one invariant, N paths"
/// shape, so the translation now lives where the response leaves the route
/// rather than in any handler.
///
/// **A layer rather than a handler-side error type, deliberately.** Most
/// failures on this route never reach the handler: `JsonBody`'s rejection for a
/// missing `max_tokens` comes from the extractor, and an unauthenticated or
/// rate-limited request is refused before either. A wrapper error type could not
/// reach any of them.
///
/// **It is therefore layered OUTSIDE auth and rate limiting**, not on the route.
/// Layered on the route it sat inside both, so the single most likely failure a
/// new user meets — a wrong API key — still came back in the OpenAI envelope.
/// Anthropic reports authentication failures in its ordinary error shape, so
/// that is the shape they must arrive in. Note this differs from MCP, whose
/// authorization spec puts auth failures at the transport level (HTTP status
/// plus `WWW-Authenticate`) and says nothing about the body — which is why
/// `/mcp` is deliberately not given the same treatment.
///
/// Because it now sees every request, it checks the path FIRST and returns
/// anything else untouched before reading a byte.
///
/// Success responses are returned untouched and are never buffered — the
/// streaming path is a 200 carrying an open SSE stream, and reading it here
/// would defeat streaming entirely.
pub async fn anthropic_error_envelope(
    req: axum::extract::Request,
    next: axum::middleware::Next,
) -> axum::response::Response {
    let is_anthropic_surface = req.uri().path() == ANTHROPIC_MESSAGES_PATH;
    let response = next.run(req).await;
    if !is_anthropic_surface
        || !(response.status().is_client_error() || response.status().is_server_error())
    {
        return response;
    }
    let (mut parts, body) = response.into_parts();
    let Ok(bytes) = axum::body::to_bytes(body, MAX_ERROR_BODY_BYTES).await else {
        // The body is gone either way at this point; an empty one with the
        // original status still tells the client the request failed.
        return (parts, axum::body::Body::empty()).into_response();
    };
    let Some(rewritten) = to_anthropic_error_body(&bytes) else {
        return (parts, axum::body::Body::from(bytes)).into_response();
    };
    // The body changed length, so the old value would truncate it.
    parts.headers.remove(axum::http::header::CONTENT_LENGTH);
    (parts, axum::body::Body::from(rewritten)).into_response()
}

/// Convert one canonical error body into Anthropic's, or `None` to leave it be.
///
/// Returning `None` for anything that is not our envelope is what keeps this
/// safe to apply to a whole route: a failure that never went through
/// `classify_error` passes through unchanged rather than being reshaped into a
/// claim about its own type.
fn to_anthropic_error_body(bytes: &[u8]) -> Option<Vec<u8>> {
    let parsed: serde_json::Value = serde_json::from_slice(bytes).ok()?;
    let err = parsed.get("error")?.as_object()?;
    let message = err.get("message")?.as_str()?;
    let mapped = sse::anthropic_error_type(err.get("type").and_then(|t| t.as_str()).unwrap_or(""));

    let mut inner = serde_json::Map::new();
    inner.insert("type".into(), serde_json::Value::String(mapped.to_string()));
    inner.insert(
        "message".into(),
        serde_json::Value::String(message.to_string()),
    );
    // The hint and its translation key are ours, not Anthropic's, and SDKs
    // ignore fields they do not know. Dropping them would cost the caller the
    // one part of the message that says what to do next.
    for extra in ["hint", "hint_key"] {
        if let Some(v) = err.get(extra) {
            inner.insert(extra.into(), v.clone());
        }
    }
    serde_json::to_vec(&serde_json::json!({ "type": "error", "error": inner })).ok()
}

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

    // R108: validate top_p range parity with the OpenAI handler (which calls
    // `validate_optional_sampling`). Without this, top_p out of [0,1] was
    // silently clamped by `build_sampling_params`, hiding client bugs.
    // Anthropic doesn't expose presence/frequency penalties or top_logprobs,
    // so only top_p applies here.
    super::validate_optional_sampling(req.top_p.map(|t| t as f64), None, None, None)?;

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
    // One predicate for both API surfaces — see `SharedState::local_fast_path_for`.
    let has_local_split_model = state.shared_state.local_fast_path_for(&requested_mid);

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
                req.tools.as_ref().is_some_and(|t| !t.is_empty()),
            )
            .await;
        } else {
            return handlers::anthropic_split_non_stream(
                &state,
                &internal_messages,
                sampling_params,
                request_id,
                model,
                req.tools.as_ref().is_some_and(|t| !t.is_empty()),
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
                    &state,
                    router_tx.clone(),
                    internal_messages,
                    sampling_params,
                    request_id,
                    model,
                    req.tools.as_ref().is_some_and(|t| !t.is_empty()),
                )
                .await;
            } else {
                return handlers::anthropic_non_stream(
                    router_tx.clone(),
                    internal_messages,
                    sampling_params,
                    request_id,
                    model,
                    req.tools.as_ref().is_some_and(|t| !t.is_empty()),
                )
                .await;
            }
        }

        // Direct executor fallback (single-node, no router)
        if model_locally_available {
            let (tmpl, bos, eos) = super::resolve_chat_template(&state, &model).await;
            let prompt = chat_template::build_prompt(
                &internal_messages,
                tmpl.as_deref(),
                &bos,
                &eos,
                Some(model.as_str()),
            );

            let mut executor = state.executor.lock().await;
            let (content, result) = executor
                .generate(&prompt, &sampling_params)
                .map_err(ApiError)?;

            let stop_reason = crate::api::anthropic::convert::map_finish_reason_with_match(
                result.finish_reason.as_str(),
                result.matched_stop_sequence.as_deref(),
            );
            let response = MessagesResponse::text_with_stop(
                request_id,
                model,
                content,
                stop_reason,
                result.matched_stop_sequence,
                result.prompt_tokens,
                result.completion_tokens,
            );
            return Ok(Json(response).into_response());
        }
    }

    // No local model — try proxying to cloud providers.
    //
    // `provider:model` selects the provider HERE; the provider itself has never
    // heard of the prefix, so it must not travel upstream — DeepSeek rejects
    // `deepseek:deepseek-v4-flash` outright. v0.3.27 stripped it on the
    // OpenAI-compatible proxy but not here, so every cloud path below still
    // forwarded it (live-confirmed 2026-07-26).
    //
    // Routing reads the STRIPPED name: `anthropic:claude-opus-4-8` does not
    // start with "claude", so the prefixed form skipped both the subscription
    // and Anthropic-cloud branches and fell through to the OpenAI translation
    // path — an Anthropic request reshaped into OpenAI form. The explicit
    // prefix still wins over the name heuristic, so `openai:claude-x` is not
    // hijacked by the Anthropic branch below.
    let (upstream_model, anthropic_allowed) = route_model(&req.model);
    let lower_model = upstream_model.to_lowercase();

    // Claude subscription: proxy through local CLI subprocess (higher priority than API key).
    // SEC: pass `is_forwarded` so peer-forwarded requests don't burn the
    // local operator's personal Claude subscription quota (gotcha #115).
    #[cfg(feature = "claude-subscription")]
    let is_forwarded = headers.get("x-swarm-forwarded").is_some();
    // An explicit non-Anthropic prefix must not be captured by the subscription
    // path: `openai:claude-x` names OpenAI, whatever the model is called. The
    // empty name matches no provider, so the lookup declines.
    #[cfg(feature = "claude-subscription")]
    let subscription_candidate = if anthropic_allowed {
        upstream_model
    } else {
        ""
    };
    #[cfg(feature = "claude-subscription")]
    if let Some(sub_config) = crate::api::claude_sub::try_get_claude_subscription(
        &state,
        subscription_candidate,
        is_forwarded,
    )
    .await
    {
        tracing::info!(model = %upstream_model, "DIAG: anthropic proxying via claude subscription subprocess");
        // Use the same ProxyMessagesRequest serializer as the cloud path so
        // tool_use / tool_result / thinking blocks survive the subprocess
        // hop. The previous hand-serialization replaced every non-text
        // ContentBlock with a "[non-text content]" placeholder, which broke
        // multi-turn function-calling conversations because the assistant's
        // tool_use blocks (and the user's tool_result blocks) were stripped
        // before the subprocess ever saw them.
        let body = serde_json::to_value(&proxy::ProxyMessagesRequest {
            model: upstream_model,
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
    if anthropic_allowed && lower_model.starts_with("claude") {
        let config = state.shared_state.metrics.providers_config.read().await;
        if let Some(ref entry) = config.anthropic {
            let api_key = entry.api_key.clone();
            drop(config);

            tracing::debug!(model = %upstream_model, "DIAG: anthropic proxying to cloud API");
            let body = serde_json::to_value(&proxy::ProxyMessagesRequest {
                model: upstream_model,
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
                model = %upstream_model,
                provider = %provider_name,
                "DIAG: anthropic→openai translation proxy to cloud provider"
            );
            return handlers::anthropic_to_openai_proxy(
                &req,
                upstream_model,
                &provider_url,
                &provider_key,
            )
            .await;
        }
    }

    // Nothing local and no cloud provider matched.
    //
    // Distinguish "you asked for a model that does not exist" from "this node
    // has nothing loaded", because they need opposite things from the user. A
    // typo'd or unavailable id is `ModelNotAvailable` → **404**, carrying the
    // list of ids that ARE available; only a node with no models at all is
    // `NoModelLoaded` → 503, whose hint tells the user to go and download one.
    //
    // The same rule already lived in the OpenAI handler ("the user likely
    // mistyped"), reached through `model_not_found_error` — the single builder
    // for this 404. This path never called it, so `/v1/messages` answered a
    // misspelled model with 503 and "No model is loaded yet. Go to the
    // dashboard and select a model", advice that is wrong when eight models are
    // loaded and only the name was off. That surface is what Claude Code talks
    // to, where a model id is typed by hand.
    let model_id = crate::types::ModelId(req.model.clone());
    if let Some(err) = state
        .shared_state
        .model_registry
        .reject_if_unknown_model(&model_id)
    {
        return Err(ApiError(err));
    }
    Err(ApiError(crate::error::SwarmError::NoModelLoaded))
}

/// Split a requested model name into the name to send upstream and whether an
/// Anthropic-shaped route is still permitted.
///
/// Two separate things go wrong if the `provider:` prefix is treated as part of
/// the model name, and both were live on `/v1/messages` until 2026-07-26:
///
/// - It travels upstream, where the provider has never heard of it and rejects
///   the request (`deepseek:deepseek-v4-flash`).
/// - It defeats the `starts_with("claude")` routing test, so
///   `anthropic:claude-opus-4-8` skipped the Anthropic branches and fell
///   through to the OpenAI translation path.
///
/// An explicit prefix outranks the name heuristic, so `openai:claude-x` names
/// OpenAI and is not captured by the Anthropic branch.
///
/// The upstream name comes from [`resolve_model`], so bare family aliases are
/// expanded here too. The cloud paths previously read `req.model` directly and
/// so never saw that expansion, despite `resolve_model`'s own documentation
/// citing this routing test as the reason it exists — meaning Claude Code
/// 2.1's default bare `sonnet` was neither routed to Anthropic nor sent as a
/// name any provider recognises.
fn route_model(model: &str) -> (&str, bool) {
    let anthropic_allowed = match model.split_once(':') {
        Some((provider, _)) => provider.eq_ignore_ascii_case("anthropic"),
        None => true,
    };
    (resolve_model(model), anthropic_allowed)
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
        use convert::{map_finish_reason, map_finish_reason_with_match};
        assert_eq!(map_finish_reason("stop"), "end_turn");
        assert_eq!(map_finish_reason("length"), "max_tokens");
        assert_eq!(map_finish_reason("unknown"), "end_turn");
        // R109: matched stop sequence overrides the default `end_turn`.
        assert_eq!(
            map_finish_reason_with_match("stop", Some("\n\nHuman:")),
            "stop_sequence"
        );
        assert_eq!(map_finish_reason_with_match("stop", None), "end_turn");
        assert_eq!(
            map_finish_reason_with_match("length", Some("xxx")),
            "max_tokens"
        );
    }

    #[test]
    fn connectivity_probe_detection() {
        let probe = MessagesRequest {
            model: "claude-opus-4-8".into(),
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
            model: "claude-opus-4-8".into(),
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
        assert_eq!(resolve_model("claude-opus-4-8"), "claude-opus-4-8");
        assert_eq!(
            resolve_model("anthropic:claude-opus-4-8"),
            "claude-opus-4-8"
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
    fn sse_message_delta_with_stop_sequence_serializes_correctly() {
        // R142.10: the split-stream handler now propagates
        // matched_stop_sequence into the MessageDelta. Pin the JSON
        // wire format so a regression here would silently violate the
        // Anthropic spec contract clients (incl. Claude Code) use to
        // detect which user-provided stop sequence fired.
        let (event_type, data) = serialize_anthropic_event(&AnthropicSseEvent::MessageDelta {
            stop_reason: "stop_sequence".into(),
            stop_sequence: Some("\n\nHuman:".into()),
            output_tokens: 42,
            input_tokens: None,
        });
        assert_eq!(event_type, "message_delta");
        let v: serde_json::Value = serde_json::from_str(&data).unwrap();
        assert_eq!(v["delta"]["stop_reason"], "stop_sequence");
        assert_eq!(v["delta"]["stop_sequence"], "\n\nHuman:");
        assert_eq!(v["usage"]["output_tokens"], 42);
    }

    #[test]
    fn sse_message_delta_without_stop_sequence_is_null() {
        let (_, data) = serialize_anthropic_event(&AnthropicSseEvent::MessageDelta {
            stop_reason: "end_turn".into(),
            stop_sequence: None,
            output_tokens: 10,
            input_tokens: None,
        });
        let v: serde_json::Value = serde_json::from_str(&data).unwrap();
        assert_eq!(v["delta"]["stop_reason"], "end_turn");
        assert!(v["delta"]["stop_sequence"].is_null());
    }

    #[test]
    fn deserialize_full_request() {
        let json = r#"{
            "model": "claude-opus-4-8",
            "max_tokens": 1024,
            "system": "You are helpful.",
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
                {"role": "user", "content": "How are you?"}
            ]
        }"#;
        let req: MessagesRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.model, "claude-opus-4-8");
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
            "model": "claude-opus-4-8",
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
            "model": "claude-opus-4-8",
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

    /// `provider:model` picks the provider locally and must not travel
    /// upstream, and the prefix must not defeat Anthropic routing.
    #[test]
    fn route_model_strips_prefix_and_keeps_anthropic_routable() {
        // Bare names are unchanged and stay eligible for the Claude branch.
        assert_eq!(route_model("claude-opus-4-8"), ("claude-opus-4-8", true));
        assert_eq!(
            route_model("deepseek-v4-flash"),
            ("deepseek-v4-flash", true)
        );

        // An `anthropic:` prefix is stripped and still routes to Anthropic —
        // previously it failed `starts_with("claude")` and fell through to the
        // OpenAI translation path.
        assert_eq!(
            route_model("anthropic:claude-opus-4-8"),
            ("claude-opus-4-8", true)
        );

        // A non-Anthropic prefix is stripped and blocks the Anthropic branch,
        // so an explicit provider choice outranks the model-name heuristic.
        assert_eq!(
            route_model("deepseek:deepseek-v4-flash"),
            ("deepseek-v4-flash", false)
        );
        assert_eq!(route_model("openai:claude-x"), ("claude-x", false));

        // Bare family aliases expand — Claude Code 2.1 sends `sonnet` by
        // default, and the cloud paths used to forward it verbatim.
        assert_eq!(route_model("sonnet"), ("claude-sonnet-5", true));
        assert_eq!(route_model("anthropic:opus"), ("claude-opus-4-8", true));
    }
}

#[cfg(test)]
mod anthropic_error_envelope_tests {
    use super::to_anthropic_error_body;

    fn convert(body: &str) -> Option<serde_json::Value> {
        to_anthropic_error_body(body.as_bytes()).map(|b| serde_json::from_slice(&b).unwrap())
    }

    /// The two things an Anthropic client needs and our canonical envelope does
    /// not carry: the top-level `"type": "error"` discriminator, and an inner
    /// type drawn from Anthropic's own set.
    ///
    /// Measured on the live node 2026-08-18 — a prompt-privacy refusal reached
    /// the caller as `{"error": {"type": "prompt_privacy_error", ...}}`, which
    /// is a type Anthropic does not define, in an envelope it does not use.
    /// `anthropic_error_type` already existed; only the streaming path called it
    /// (gotcha #302).
    #[test]
    fn a_refusal_reaches_an_anthropic_client_in_anthropics_own_shape() {
        let out = convert(
            r#"{"error":{"code":"prompt_privacy_error","message":"Prompt privacy is on",
                "param":null,"type":"prompt_privacy_error"}}"#,
        )
        .expect("our envelope must convert");

        assert_eq!(out["type"], "error", "the discriminator SDKs branch on");
        assert_eq!(
            out["error"]["type"], "api_error",
            "a type Anthropic does not define must become its generic server-side one, \
             never be invented"
        );
        assert_eq!(out["error"]["message"], "Prompt privacy is on");
    }

    /// A type both APIs share keeps its name — the translation must not flatten
    /// every failure into `api_error` and lose what actually went wrong.
    #[test]
    fn a_shared_type_survives_translation() {
        let out = convert(
            r#"{"error":{"code":"not_found_error","message":"Model not available",
                "type":"not_found_error"}}"#,
        )
        .unwrap();
        assert_eq!(out["error"]["type"], "not_found_error");

        let out = convert(
            r#"{"error":{"code":"invalid_request_error","message":"bad","type":"invalid_request_error"}}"#,
        )
        .unwrap();
        assert_eq!(out["error"]["type"], "invalid_request_error");
    }

    /// The hint is the one part of the message that says what to do next, and it
    /// is translated. Anthropic clients ignore fields they do not know, so there
    /// is no reason to drop it.
    #[test]
    fn the_actionable_hint_and_its_translation_key_are_kept() {
        let out = convert(
            r#"{"error":{"code":"not_found_error","message":"Model not available",
                "type":"not_found_error","hint":"Open the Models tab",
                "hint_key":"model_not_available"}}"#,
        )
        .unwrap();
        assert_eq!(out["error"]["hint"], "Open the Models tab");
        assert_eq!(out["error"]["hint_key"], "model_not_available");
    }

    /// Anything that is not our envelope is left alone. The layer covers a whole
    /// route, so a body that never went through `classify_error` must pass
    /// through rather than be reshaped into a claim about its own type.
    #[test]
    fn a_body_that_is_not_ours_is_left_untouched() {
        assert!(convert("not json at all").is_none());
        assert!(convert(r#"{"detail":"something else"}"#).is_none());
        assert!(
            convert(r#"{"error":{"code":"x"}}"#).is_none(),
            "no message means we cannot build a valid Anthropic error"
        );
        assert!(
            convert(r#"{"error":"a bare string"}"#).is_none(),
            "the error field must be an object"
        );
    }
}
