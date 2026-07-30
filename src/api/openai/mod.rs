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
                "session_id too long (max 256 bytes)".into(),
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
    super::validate_optional_sampling(
        Some(req.top_p as f64),
        req.top_logprobs,
        Some(req.presence_penalty as f64),
        Some(req.frequency_penalty as f64),
    )?;

    // R107: reject max_tokens=0 and out-of-range values explicitly. The
    // local sampling clamp at api/mod.rs::build_sampling_params would
    // silently coerce 0→1 and >32768→32768, but OpenAI's spec requires
    // max_tokens>0 and clients deserve explicit error feedback rather
    // than getting a one-token response. The Anthropic /v1/messages
    // handler enforces the same range — keep the two paths in sync.
    if req.max_tokens == 0 || req.max_tokens > super::DEFAULT_MAX_TOKENS {
        return Err(ApiError(crate::error::SwarmError::Validation(format!(
            "max_tokens must be 1..={}",
            super::DEFAULT_MAX_TOKENS
        ))));
    }

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
                "lora_adapter name too long (max 256 bytes)".into(),
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

    // SEC: cap response_format.json_schema sizes. The schema is serialized to a
    // string and injected into the system prompt; without a cap, an attacker
    // can submit a 10 MB schema that bypasses validate_content_size (which
    // only inspects messages, not response_format).
    if let Some(crate::api::openai::types::ResponseFormat::JsonSchema { ref json_schema }) =
        req.response_format
    {
        const MAX_SCHEMA_NAME_BYTES: usize = 256;
        const MAX_SCHEMA_BYTES: usize = 64 * 1024;
        if json_schema.name.len() > MAX_SCHEMA_NAME_BYTES {
            return Err(ApiError(crate::error::SwarmError::Validation(format!(
                "response_format.json_schema.name exceeds {MAX_SCHEMA_NAME_BYTES} bytes"
            ))));
        }
        let schema_str = json_schema.schema.to_string();
        if schema_str.len() > MAX_SCHEMA_BYTES {
            return Err(ApiError(crate::error::SwarmError::Validation(format!(
                "response_format.json_schema.schema exceeds {MAX_SCHEMA_BYTES} bytes"
            ))));
        }
    }

    Ok(())
}

/// Try proxying a chat completion request to a configured cloud provider.
///
/// `is_forwarded` is consulted only by the `claude-subscription` feature
/// path; when that feature is disabled the parameter is unused.
#[cfg_attr(not(feature = "claude-subscription"), allow(unused_variables))]
async fn try_cloud_proxy(
    state: &AppState,
    req: &ChatCompletionRequest,
    is_forwarded: bool,
) -> Result<Option<axum::response::Response>, ApiError> {
    // Claude subscription: proxy through local CLI subprocess (higher priority than API key)
    #[cfg(feature = "claude-subscription")]
    if let Some(sub_config) =
        crate::api::claude_sub::try_get_claude_subscription(state, &req.model, is_forwarded).await
    {
        tracing::info!(model = %req.model, "DIAG: openai proxying via claude subscription subprocess");
        let body = serde_json::to_value(req).map_err(|e| {
            ApiError(crate::error::SwarmError::Internal(format!(
                "serialize request for proxy: {e}"
            )))
        })?;
        return crate::api::claude_sub::proxy_via_subprocess_openai(&sub_config, &body)
            .await
            .map(Some);
    }

    let body = serde_json::to_value(req).map_err(|e| {
        ApiError(crate::error::SwarmError::Internal(format!(
            "serialize request for proxy: {e}"
        )))
    })?;
    crate::api::providers::try_proxy_openai(state, &body, req.stream).await
}

/// Cancels an in-flight request if this guard drops before `disarm` is called.
///
/// A client that simply closes its connection cancels nothing on its own:
/// cancellation was only ever wired to an explicit `x-swarmllm-cancel-token`
/// header, which only the `/v1/responses` background runner sets. So a client
/// that sent a long prompt and went away left the request running, holding the
/// executor for the whole generation and blocking every later request to that
/// model — reported 2026-07-29, where the next trivial request stayed blocked
/// and only killing the process recovered it.
///
/// Axum drops the handler future when the client disconnects, so a drop guard
/// is the signal. It is armed ONLY on the non-streaming path: a streaming
/// handler returns as soon as the SSE body is constructed and the generation
/// continues afterwards, so cancelling on its return would kill every stream.
struct CancelOnDisconnect(Option<std::sync::Arc<std::sync::atomic::AtomicBool>>);

impl CancelOnDisconnect {
    /// The request finished normally — do not cancel on drop.
    fn disarm(mut self) {
        self.0 = None;
    }
}

impl Drop for CancelOnDisconnect {
    fn drop(&mut self) {
        if let Some(flag) = self.0.take() {
            flag.store(true, std::sync::atomic::Ordering::Release);
            tracing::info!("DIAG: client disconnected before completion — cancelling request");
        }
    }
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

    // Cancellation: callers (e.g. /v1/responses background runner) set a
    // pre-registered cancel token via the `x-swarmllm-cancel-token` header.
    // The pipeline executor checks `request.is_cancelled()` per token in the
    // decode loop, so flipping the AtomicBool from the cancel API stops the
    // generation on the next iteration.
    let cancel_token: Option<std::sync::Arc<std::sync::atomic::AtomicBool>> = headers
        .get("x-swarmllm-cancel-token")
        .and_then(|v| v.to_str().ok())
        .and_then(|tok| {
            state
                .shared_state
                .cancel_signals
                .get(tok)
                .map(|r| r.value().clone())
        });

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
        // PRIVACY: do NOT include prompt content. The activity event bus
        // broadcasts every emit to all authenticated dashboard subscribers
        // and replays activity_history to new connections — leaking even
        // the first ~60 characters of a prompt across tenants on a
        // multi-user node would be a privacy regression. Operational
        // visibility (model + msg count + max_tokens) is sufficient.
        state.shared_state.emit_activity(
            crate::daemon::state::ActivityEvent::new(
                "inference",
                "inference_request",
                format!(
                    "Inference request on {} — {} message{}, max {} tokens",
                    display,
                    msg_count,
                    if msg_count != 1 { "s" } else { "" },
                    max_tok,
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
    //
    // The `x-swarm-forwarded` header prevents infinite forwarding loops between
    // nodes. The legitimate setter is `forward_to_peer`, which adds it when a
    // peer node forwards an inference request (auth carried via the verbatim
    // Bearer — see gotcha #30). Any Bearer-authenticated caller can set this
    // header on their own request, but the only effect is forcing this node
    // to skip peer-forwarding and attempt the request locally; that's a
    // routing-only side channel that provides no privilege gain (they already
    // have full inference rights via Bearer) and only hurts the caller (they
    // get a 503 if local shards aren't available). Not worth additional
    // middleware enforcement.
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
                    &state,
                    router_tx.clone(),
                    &req,
                    internal_messages.clone(),
                    request_id,
                    created,
                    cancel_token.clone(),
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
                let auth = headers
                    .get(axum::http::header::AUTHORIZATION)
                    .and_then(|v| v.to_str().ok());
                return forward_to_peer(&peer_url, &req, req.stream, auth).await;
            }
        }

        // Cloud provider fast-path: if the model matches a configured cloud provider,
        // route immediately without cold-start waiting. Cloud models are never local.
        if let Some(response) = try_cloud_proxy(&state, &req, is_forwarded).await? {
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
                        &state,
                        router_tx.clone(),
                        &req,
                        internal_messages.clone(),
                        request_id,
                        created,
                        cancel_token.clone(),
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
                    let auth = headers
                        .get(axum::http::header::AUTHORIZATION)
                        .and_then(|v| v.to_str().ok());
                    return forward_to_peer(&peer_url, &req, req.stream, auth).await;
                }
            }
        }

        // Cloud provider fallback: proxy to configured cloud provider if model matches
        if let Some(response) = try_cloud_proxy(&state, &req, is_forwarded).await? {
            return Ok(response);
        }

        return Err(ApiError(crate::error::SwarmError::NoModelLoaded));
    }

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
        // Echo the model the client actually requested (`req.model`), NOT the
        // manifest's display name — they can diverge (e.g. id
        // `qwen2.5-0.5b-instruct-fp16` vs name `qwen2.5-0.5b-instruct`), and an
        // OpenAI-compatible client/router that re-routes on the response `model`
        // field must see back what it sent. (Anthropic already echoes verbatim.)
        if req.stream {
            return Ok(split_stream_response(
                state,
                request_id,
                created,
                req.model.clone(),
                internal_messages.clone(),
                params,
                requested_mid.clone(),
                req.tools.as_ref().is_some_and(|t| !t.is_empty()),
                req.extras
                    .get("stream_options")
                    .and_then(|v| v.get("include_usage"))
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false),
            )
            .await
            .into_response());
        } else {
            return split_non_stream_response(
                state,
                request_id,
                created,
                req.model.clone(),
                internal_messages.clone(),
                params,
                requested_mid.clone(),
                req.tools.as_ref().is_some_and(|t| !t.is_empty()),
            )
            .await;
        }
    }

    // Build prompt for non-split paths (distributed inference + direct executor).
    // Uses loaded_model_info first; falls back to GGUF header on disk for
    // distributed-only nodes that have no local model but do have the probe.
    let prompt = {
        let (tmpl, bos, eos) = super::resolve_chat_template(&state, &req.model).await;
        chat_template::build_prompt(
            &internal_messages,
            tmpl.as_deref(),
            &bos,
            &eos,
            Some(req.model.as_str()),
        )
    };

    // Distributed inference: network covers all layers across multiple nodes.
    let peers_have_shards = all_shards_available(&state, &req.model)
        || state.shared_state.config.inference.shard_range.is_some();
    if peers_have_shards {
        if let Some(router_tx) = &state.router_tx {
            return dispatch_inference(
                &state,
                router_tx.clone(),
                &req,
                internal_messages.clone(),
                request_id,
                created,
                cancel_token.clone(),
            )
            .await;
        }
    }

    if req.stream {
        // Streaming: use direct executor path for real token-by-token SSE.
        // Echo `req.model` (the requested id), not the manifest display name —
        // consistent with the split fast path and `router_inference` above.
        Ok(stream_response(
            state,
            request_id,
            created,
            req.model.clone(),
            prompt,
            params,
        )
        .await
        .into_response())
    } else if let Some(router_tx) = &state.router_tx {
        // Non-streaming: route through InferenceRouter for priority queueing.
        //
        // Give the request a cancel flag even when the caller supplied none, so
        // that a client going away actually stops the work instead of leaving
        // the model held for the full generation.
        let effective_cancel = cancel_token.clone().or_else(|| {
            Some(std::sync::Arc::new(std::sync::atomic::AtomicBool::new(
                false,
            )))
        });
        let disconnect_guard = CancelOnDisconnect(effective_cancel.clone());
        let result = router_inference(
            router_tx.clone(),
            &req,
            internal_messages,
            request_id,
            created,
            effective_cancel,
        )
        .await;
        disconnect_guard.disarm();
        result
    } else {
        Err(crate::error::ApiError(
            crate::error::SwarmError::ServiceUnavailable(
                "Inference router not available".to_string(),
            ),
        ))
    }
}

/// How a model is served from this node's point of view: `local` when every
/// shard is here, `hybrid` when some are, `network` when none are.
///
/// Derived from shard possession. It previously came from `loaded_model_info`,
/// the most-recently-loaded singleton, which made the flag effectively inverted:
/// a tester saw four completely-held models reported `network` while the only
/// partially-held one (2/8) was reported `local`, because it was the last one
/// touched. Clients pick models by this field to avoid network round trips, so
/// wrong is worse than absent. `hybrid` is reported honestly rather than being
/// rounded to `local` — holding some shards is not the same as being able to
/// answer alone.
fn owned_by_for(state: &AppState, model_id: &str) -> String {
    let Some(m) = state
        .shared_state
        .model_registry
        .get_manifest(&crate::types::ModelId(model_id.to_string()))
    else {
        // No manifest: it is servable (we listed it) but we cannot count shards.
        return "network".into();
    };
    let (local, _reachable) = crate::api::count_shard_availability(&m, &state.shared_state);
    if m.shard_count > 0 && local == m.shard_count as usize {
        "local".into()
    } else if local > 0 {
        "hybrid".into()
    } else {
        "network".into()
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

        // Try to use the manifest's publish date; fall back to the registry's
        // local-load timestamp if available; only fall back to 0 if neither.
        // Returning Utc::now() per call (the prior behaviour) made `created`
        // change every /v1/models response, breaking client-side caches that
        // key on (id, created).
        let created = state
            .shared_state
            .model_registry
            .get_manifest(&crate::types::ModelId(model_id.clone()))
            .map(|m| m.publish_date.timestamp())
            .unwrap_or(0);
        // `owned_by` must describe shard possession, not which model happened
        // to be loaded last. See `api::count_shard_availability`.
        let owned_by = owned_by_for(&state, &model_id);
        data.push(ModelInfo {
            id: model_id,
            object: "model",
            created,
            owned_by,
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
            let owned_by = owned_by_for(&state, &id);
            data.push(ModelInfo {
                id,
                object: "model",
                created: manifest.publish_date.timestamp(),
                owned_by,
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

    // `model_loaded` only ever meant "this node has *a* model loaded" — it is
    // derived from a singleton and says nothing about the model you asked for,
    // or about one still downloading. Reported 2026-07-29 by a user who saw
    // `model_loaded: true` while shards were still arriving, trusted it, and
    // got "No node available for layer 3". The field is kept for compatibility;
    // `models_downloading` is the honest signal for readiness.
    let mut models_downloading = Vec::new();
    for entry in state.shared_state.models.acquisition_progress.iter() {
        let st = entry.value();
        if matches!(
            st.state,
            crate::model::acquisition::AcquisitionState::Downloading
                | crate::model::acquisition::AcquisitionState::AwaitingManifest
        ) {
            models_downloading.push(serde_json::json!({
                "model": st.model_id.0,
                "downloaded_shards": st.downloaded_shards,
                "total_shards": st.total_shards,
                "state": match st.state {
                    crate::model::acquisition::AcquisitionState::AwaitingManifest => "awaiting_manifest",
                    _ => "downloading",
                },
            }));
        }
    }
    models_downloading.sort_by(|a, b| a["model"].as_str().cmp(&b["model"].as_str()));

    Json(serde_json::json!({
        "status": "ok",
        "version": env!("CARGO_PKG_VERSION"),
        "model_loaded": local_model,
        "model_name": model_name,
        "network_models": network_models,
        // Non-empty means shards are still arriving: models listed here are NOT
        // ready to serve, regardless of `model_loaded`.
        "models_downloading": models_downloading,
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

#[cfg(test)]
mod cancel_on_disconnect_tests {
    use super::CancelOnDisconnect;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    /// Dropping without disarming means the client went away — cancel.
    #[test]
    fn drop_without_disarm_cancels() {
        let flag = Arc::new(AtomicBool::new(false));
        {
            let _g = CancelOnDisconnect(Some(flag.clone()));
        }
        assert!(
            flag.load(Ordering::Acquire),
            "an abandoned request must be cancelled so it stops holding the model"
        );
    }

    /// A request that completed normally must NOT be marked cancelled.
    #[test]
    fn disarm_prevents_cancellation() {
        let flag = Arc::new(AtomicBool::new(false));
        CancelOnDisconnect(Some(flag.clone())).disarm();
        assert!(
            !flag.load(Ordering::Acquire),
            "a completed request must not be flagged cancelled"
        );
    }
}
