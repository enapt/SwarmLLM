//! OpenAI `/v1/responses` endpoint — request/response types, translation
//! to/from Chat Completions, and HTTP handler.
//!
//! Milestone scope:
//! - **M1**: types + serde roundtrip.
//! - **M2**: route wired, built-in-tool rejection, 501 stub.
//! - **M3**: plain-text local inference via Chat translation.
//! - **M4**: function tools and tool_choice translation.
//! - **M5**: cloud-proxy verbatim path.
//! - **M6**: SSE streaming.
//! - **M7**: redb persistence (store=true + retrieve + delete).
//! - **M8**: previous_response_id chaining.
//! - **M9**: background=true + POST .../cancel.
//! - **V1 (v2 plan)**: streaming first-token latency fix.
//! - **V2 (v2 plan)**: multimodal input parts.
//! - **V3 (v2 plan)**: Claude → Anthropic Messages translation.
//! - **V4 (v2 plan)**: input_items pagination endpoint.

pub mod anthropic_bridge;
pub mod store;
pub mod stream;
pub mod translate;
pub mod types;

pub use types::*;

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use axum::body::to_bytes;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use dashmap::DashMap;

use crate::api::server::{AppState, JsonBody};
use crate::error::{ApiError, SwarmError};

/// Tool `type` strings that map to OpenAI-hosted infrastructure SwarmLLM
/// does not run.
pub(crate) const BUILTIN_TOOL_TYPES: &[&str] = &[
    "web_search",
    "file_search",
    "computer_use_preview",
    "code_interpreter",
    "image_generation",
    "mcp",
    "custom",
];

/// Cap on the body size we'll buffer when forwarding a Chat Completions
/// response into the translation layer. 16 MiB is a generous bound — local
/// inference responses are normally well under 1 MiB; the cap exists to
/// prevent an unbounded internal allocation if something goes sideways.
const MAX_CHAT_RESPONSE_BYTES: usize = 16 * 1024 * 1024;

/// In-flight map of background response ids to their cancel flags.
///
/// The cancel handler flips the flag; the background worker checks it at
/// completion time — if set, the worker discards its inference result
/// instead of overwriting the stored `cancelled` record. Real-time token-
/// level interruption is out of scope: the chat handler owns its own
/// cancellation surface (per-request, not per-response).
///
/// Keys are removed when the background worker finishes (whether it wrote
/// a completed result or was cancelled).
static BACKGROUND_CANCEL: std::sync::LazyLock<DashMap<String, Arc<AtomicBool>>> =
    std::sync::LazyLock::new(DashMap::new);

/// Walk a tools array and return the first built-in tool type encountered.
pub(crate) fn first_builtin_tool(tools: &[ToolDef]) -> Option<&'static str> {
    for t in tools {
        let kind = t.type_str()?;
        for &builtin in BUILTIN_TOOL_TYPES {
            if kind == builtin {
                return Some(builtin);
            }
        }
    }
    None
}

/// `POST /v1/responses`.
///
/// Routing order:
/// 1. Cloud proxy: if the model resolves to an OpenAI-compatible provider,
///    proxy the request body verbatim to the upstream `/responses`
///    endpoint. Built-in tools, streaming, background, reasoning effort,
///    text.verbosity, include[], previous_response_id, and any future
///    field all round-trip via `#[serde(flatten)] extras` and the upstream
///    handles them. Anthropic / subprocess providers return a clear 400
///    pointing the caller at `/v1/messages`.
/// 2. Local inference: built-in-tool gate, M6/M8/M9 stubs, then translate
///    to Chat Completions and run the local model.
pub async fn create_response(
    State(state): State<AppState>,
    headers: axum::http::HeaderMap,
    JsonBody(req): JsonBody<ResponsesRequest>,
) -> Result<Response, ApiError> {
    // ---- 1a. Cloud proxy passthrough (M5 / V3). ----
    // Serialize the request struct back to JSON so flatten-extras and any
    // unmodeled OpenAI knobs reach the upstream verbatim.
    let body_value = serde_json::to_value(&req).map_err(|e| {
        ApiError(SwarmError::Internal(format!(
            "Failed to serialize Responses request for cloud proxy: {e}"
        )))
    })?;
    let stream = req.stream.unwrap_or(false);
    if let Some(response) =
        crate::api::providers::try_proxy_openai_responses(&state, &body_value, stream).await?
    {
        return Ok(response);
    }

    // ---- 1b. Anthropic-translated passthrough (V3). ----
    // When the model resolves to an Anthropic provider (or the
    // claude-subscription subprocess), translate the Responses request
    // to an Anthropic Messages request, forward, and translate back.
    // `try_proxy_anthropic_responses` returns Ok(None) when the model
    // isn't Anthropic, letting us fall through to local inference.
    if let Some(response) =
        anthropic_bridge::try_proxy_anthropic_responses(&state, &headers, &req).await?
    {
        return Ok(response);
    }

    // ---- 2. Local inference path. ----
    // Built-in tool gate. Built-ins require backing infra we don't run;
    // `function` tools translate through to Chat (M4).
    if let Some(tools) = req.tools.as_deref() {
        if let Some(builtin) = first_builtin_tool(tools) {
            return Err(ApiError(SwarmError::Validation(format!(
                "Built-in tool `{builtin}` is not supported by /v1/responses on \
                 this server. Only `function` tools are accepted; OpenAI-hosted \
                 tools (web_search, file_search, computer_use_preview, \
                 code_interpreter, image_generation, mcp, custom) require backing \
                 infrastructure SwarmLLM does not run."
            ))));
        }
    }

    // M8: load the prior record when the caller is chaining. Both the
    // streaming and non-streaming paths below pass it through to
    // translate::request_to_chat so the prior turn's messages prepend
    // to the current input. (Cloud proxy already returned above with
    // the field forwarded verbatim.)
    let prior = match req.previous_response_id.as_ref() {
        Some(prev_id) => match store::load(&state.db, prev_id).map_err(ApiError)? {
            Some(record) => Some(record),
            None => {
                return Err(ApiError(SwarmError::Validation(format!(
                    "previous_response_id `{prev_id}` not found or expired. \
                     Either pass the prior turn's messages inline via `input` \
                     or re-run the original call with store=true (default)."
                ))));
            }
        },
        None => None,
    };

    // Streaming (M6) — local-inference SSE. Cloud proxy already streamed
    // above if it matched.
    if stream {
        return stream::run_streaming(state, headers, req, prior).await;
    }

    // Background mode (M9). stream=true + background=true requires
    // resumable-SSE plumbing (scope-out in the plan); the non-stream
    // background path is what M9 ships.
    if req.background.unwrap_or(false) {
        return start_background(state, headers, req, prior).await;
    }

    // Translate to a Chat Completions request and call the existing
    // handler. Translation failures bubble up as 400 via Validation.
    let chat_req = translate::request_to_chat(&req, prior.as_ref())?;

    let chat_response = crate::api::openai::chat_completions(
        State(state.clone()),
        headers.clone(),
        JsonBody(chat_req),
    )
    .await?;

    // 7. If the chat handler returned an error response, pass it through
    //    verbatim — error JSON has the same shape both APIs use.
    if !chat_response.status().is_success() {
        return Ok(chat_response);
    }

    // 8. Parse the chat response body and translate to a Responses shape.
    let (parts, body) = chat_response.into_parts();
    let bytes = to_bytes(body, MAX_CHAT_RESPONSE_BYTES).await.map_err(|e| {
        ApiError(SwarmError::Internal(format!(
            "Failed to buffer chat response body: {e}"
        )))
    })?;
    let chat_value: serde_json::Value = serde_json::from_slice(&bytes).map_err(|e| {
        ApiError(SwarmError::Internal(format!(
            "Failed to parse chat response JSON: {e}"
        )))
    })?;

    let response_id = format!("resp_{}", uuid::Uuid::new_v4().simple());
    let created_at = chrono::Utc::now().timestamp();
    let resp = translate::chat_response_to_responses(&chat_value, &req, &response_id, created_at)?;

    // M7: persist the completed response when store=true (the OpenAI default).
    if req.store.unwrap_or(true) {
        let record = store::ResponsesRecord::new(
            req.clone(),
            resp.clone(),
            created_at,
            store::DEFAULT_TTL_SECS,
        );
        if let Err(e) = store::store(&state.db, &record) {
            // Failure to persist should not kill the response — log and
            // return the generated answer. Caller can retry a GET if they
            // care (they'll get 404 and know to pass the full turn inline).
            tracing::warn!(error = %e, id = %resp.id, "responses store failed");
        }
    }

    let mut out = (StatusCode::OK, Json(resp)).into_response();
    // Preserve any non-content headers the chat handler set (rate-limit
    // headers, custom auth echoes, etc.). Keep our own status.
    for (name, value) in parts.headers.iter() {
        if name == axum::http::header::CONTENT_TYPE || name == axum::http::header::CONTENT_LENGTH {
            continue;
        }
        out.headers_mut().insert(name.clone(), value.clone());
    }
    Ok(out)
}

/// Spawn a background inference task and return a queued placeholder
/// response immediately. The task runs the same translate → chat path as
/// the synchronous handler; the cancel flag is checked right before the
/// completed record is written so a cancel request mid-inference leaves
/// `status="cancelled"` as the final stored state.
async fn start_background(
    state: AppState,
    headers: axum::http::HeaderMap,
    req: ResponsesRequest,
    prior: Option<store::ResponsesRecord>,
) -> Result<Response, ApiError> {
    let response_id = format!("resp_{}", uuid::Uuid::new_v4().simple());
    let created_at = chrono::Utc::now().timestamp();

    let cancel_flag = Arc::new(AtomicBool::new(false));
    BACKGROUND_CANCEL.insert(response_id.clone(), cancel_flag.clone());

    // Seed redb with a queued placeholder so a GET before inference runs
    // returns meaningful state (id, model, queued).
    let queued = ResponsesResponse {
        id: response_id.clone(),
        object: "response".into(),
        created_at,
        status: ResponseStatus::Queued,
        model: req.model.clone(),
        output: Vec::new(),
        output_text: None,
        usage: ResponsesUsage::default(),
        error: None,
        incomplete_details: None,
        previous_response_id: req.previous_response_id.clone(),
        instructions: req.instructions.clone(),
        tools: req.tools.clone(),
        tool_choice: req.tool_choice.clone(),
        parallel_tool_calls: req.parallel_tool_calls,
        temperature: Some(req.temperature.unwrap_or(0.7)),
        top_p: Some(req.top_p.unwrap_or(0.9)),
        max_output_tokens: Some(req.max_output_tokens.unwrap_or(2048)),
        truncation: req.truncation.clone(),
        metadata: req.metadata.clone(),
        user: req.user.clone(),
        reasoning: req.reasoning.clone(),
        text: req.text.clone(),
        modalities: req.modalities.clone(),
        service_tier: req.service_tier.clone(),
        background: Some(true),
        extras: HashMap::new(),
    };
    let record = store::ResponsesRecord::new(
        req.clone(),
        queued.clone(),
        created_at,
        store::DEFAULT_TTL_SECS,
    );
    if let Err(e) = store::store(&state.db, &record) {
        BACKGROUND_CANCEL.remove(&response_id);
        return Err(ApiError(e));
    }

    let state_bg = state.clone();
    let headers_bg = headers.clone();
    let req_bg = req.clone();
    let prior_bg = prior;
    let response_id_bg = response_id.clone();
    let flag_bg = cancel_flag.clone();
    tokio::spawn(async move {
        run_background_inference(
            state_bg,
            headers_bg,
            req_bg,
            prior_bg,
            response_id_bg,
            created_at,
            flag_bg,
        )
        .await;
    });

    Ok((StatusCode::OK, Json(queued)).into_response())
}

/// The tokio task body for a background response. Runs translate →
/// chat_completions → translate-back, then writes the final record
/// (unless the cancel flag was flipped in the meantime). Errors are
/// captured into `status="failed"` with an `error` object so a polling
/// caller gets a stable shape.
async fn run_background_inference(
    state: AppState,
    headers: axum::http::HeaderMap,
    req: ResponsesRequest,
    prior: Option<store::ResponsesRecord>,
    response_id: String,
    created_at: i64,
    cancel_flag: Arc<AtomicBool>,
) {
    // Mark in_progress before running so pollers see the transition.
    if let Ok(Some(mut rec)) = store::load(&state.db, &response_id) {
        rec.response.status = ResponseStatus::InProgress;
        let _ = store::store(&state.db, &rec);
    }

    let finalize = |status: ResponseStatus,
                    output: Vec<OutputItem>,
                    output_text: Option<String>,
                    usage: ResponsesUsage,
                    error: Option<ResponseError>| {
        // Cancel-wins policy: if the flag flipped during inference, keep
        // the cancelled record that POST .../cancel wrote; drop our
        // finalized result on the floor.
        if cancel_flag.load(Ordering::SeqCst) {
            BACKGROUND_CANCEL.remove(&response_id);
            return;
        }
        if let Ok(Some(mut rec)) = store::load(&state.db, &response_id) {
            rec.response.status = status;
            rec.response.output = output;
            rec.response.output_text = output_text;
            rec.response.usage = usage;
            rec.response.error = error;
            if let Err(e) = store::store(&state.db, &rec) {
                tracing::warn!(error = %e, id = %response_id, "background finalize store failed");
            }
        }
        BACKGROUND_CANCEL.remove(&response_id);
    };

    let chat_req = match translate::request_to_chat(&req, prior.as_ref()) {
        Ok(c) => c,
        Err(e) => {
            finalize(
                ResponseStatus::Failed,
                Vec::new(),
                None,
                ResponsesUsage::default(),
                Some(ResponseError {
                    code: "invalid_request_error".into(),
                    message: e.to_string(),
                    extras: HashMap::new(),
                }),
            );
            return;
        }
    };

    let chat_response = match crate::api::openai::chat_completions(
        State(state.clone()),
        headers,
        JsonBody(chat_req),
    )
    .await
    {
        Ok(r) => r,
        Err(e) => {
            finalize(
                ResponseStatus::Failed,
                Vec::new(),
                None,
                ResponsesUsage::default(),
                Some(ResponseError {
                    code: "internal_error".into(),
                    message: e.0.to_string(),
                    extras: HashMap::new(),
                }),
            );
            return;
        }
    };

    if !chat_response.status().is_success() {
        let bytes = match to_bytes(chat_response.into_body(), MAX_CHAT_RESPONSE_BYTES).await {
            Ok(b) => b,
            Err(e) => {
                finalize(
                    ResponseStatus::Failed,
                    Vec::new(),
                    None,
                    ResponsesUsage::default(),
                    Some(ResponseError {
                        code: "internal_error".into(),
                        message: format!("buffer error: {e}"),
                        extras: HashMap::new(),
                    }),
                );
                return;
            }
        };
        let msg = String::from_utf8_lossy(&bytes).to_string();
        finalize(
            ResponseStatus::Failed,
            Vec::new(),
            None,
            ResponsesUsage::default(),
            Some(ResponseError {
                code: "upstream_error".into(),
                message: msg,
                extras: HashMap::new(),
            }),
        );
        return;
    }

    let bytes = match to_bytes(chat_response.into_body(), MAX_CHAT_RESPONSE_BYTES).await {
        Ok(b) => b,
        Err(e) => {
            finalize(
                ResponseStatus::Failed,
                Vec::new(),
                None,
                ResponsesUsage::default(),
                Some(ResponseError {
                    code: "internal_error".into(),
                    message: format!("buffer error: {e}"),
                    extras: HashMap::new(),
                }),
            );
            return;
        }
    };
    let chat_value: serde_json::Value = match serde_json::from_slice(&bytes) {
        Ok(v) => v,
        Err(e) => {
            finalize(
                ResponseStatus::Failed,
                Vec::new(),
                None,
                ResponsesUsage::default(),
                Some(ResponseError {
                    code: "internal_error".into(),
                    message: format!("parse chat JSON: {e}"),
                    extras: HashMap::new(),
                }),
            );
            return;
        }
    };

    let resp =
        match translate::chat_response_to_responses(&chat_value, &req, &response_id, created_at) {
            Ok(r) => r,
            Err(e) => {
                finalize(
                    ResponseStatus::Failed,
                    Vec::new(),
                    None,
                    ResponsesUsage::default(),
                    Some(ResponseError {
                        code: "internal_error".into(),
                        message: e.to_string(),
                        extras: HashMap::new(),
                    }),
                );
                return;
            }
        };

    finalize(resp.status, resp.output, resp.output_text, resp.usage, None);
}

/// `GET /v1/responses/:id` — retrieve a stored response.
pub async fn get_response(
    State(state): State<AppState>,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> Result<Response, ApiError> {
    match store::load(&state.db, &id).map_err(ApiError)? {
        Some(record) => Ok((StatusCode::OK, Json(record.response)).into_response()),
        None => Err(ApiError(SwarmError::Validation(format!(
            "Response `{id}` not found or expired. Retention is 30 days; \
             pass store=false to opt out of persistence."
        )))),
    }
}

/// `POST /v1/responses/:id/cancel` — flip the cancel flag and mark the
/// stored record as `cancelled`. Idempotent: a second call is a no-op.
/// Returns the updated response. 400 if the id is unknown.
pub async fn cancel_response(
    State(state): State<AppState>,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> Result<Response, ApiError> {
    // Signal the in-flight task (if any) to drop its result.
    if let Some(entry) = BACKGROUND_CANCEL.get(&id) {
        entry.store(true, Ordering::SeqCst);
    }

    match store::load(&state.db, &id).map_err(ApiError)? {
        Some(mut record) => {
            record.response.status = ResponseStatus::Cancelled;
            store::store(&state.db, &record).map_err(ApiError)?;
            Ok((StatusCode::OK, Json(record.response)).into_response())
        }
        None => Err(ApiError(SwarmError::Validation(format!(
            "Response `{id}` not found or expired. Cancel only applies to \
             background responses with store=true (the default)."
        )))),
    }
}

/// Query parameters for `GET /v1/responses/:id/input_items`.
#[derive(Debug, serde::Deserialize)]
pub struct ListInputItemsParams {
    /// Cursor: return items after the one with this id.
    #[serde(default)]
    pub after: Option<String>,
    /// Page size. Defaults to 20 (matches OpenAI's default). Capped at 100.
    #[serde(default)]
    pub limit: Option<u32>,
    /// `"asc"` (default) returns items in the order they appeared in the
    /// original request; `"desc"` reverses.
    #[serde(default)]
    pub order: Option<String>,
    /// Forward-compat: OpenAI SDKs pass `before` for reverse-cursor
    /// pagination. We accept and ignore it for now (single-direction
    /// cursor is sufficient for the shapes our callers use).
    #[serde(default)]
    pub before: Option<String>,
    /// Forward-compat for `include[reasoning.encrypted_content]` etc.
    /// Currently unused on the local path (we don't generate reasoning
    /// input items; cloud-proxied responses are served from the verbatim
    /// stored body).
    #[serde(default)]
    pub include: Option<String>,
}

/// `GET /v1/responses/:id/input_items` — paginated list of the input
/// items that were sent as the request body. V4 (responses_api_v2):
/// small bookkeeping endpoint hit by OpenAI SDKs in retried-tool-call
/// flows.
///
/// Synthetic ids: input items don't carry stable ids on the wire, so we
/// emit `item_{n}` where `n` is the zero-based position in the original
/// request. A `Text` input produces a single synthetic `message` item.
/// Cursor (`after`) matches by the synthetic id.
pub async fn list_input_items(
    State(state): State<AppState>,
    axum::extract::Path(id): axum::extract::Path<String>,
    axum::extract::Query(params): axum::extract::Query<ListInputItemsParams>,
) -> Result<Response, ApiError> {
    let record = store::load(&state.db, &id)
        .map_err(ApiError)?
        .ok_or_else(|| {
            ApiError(SwarmError::Validation(format!(
                "Response `{id}` not found or expired. Retention is 30 days; \
                 pass store=false to opt out of persistence."
            )))
        })?;

    let body = build_input_items_page(&record.request.input, &params);
    Ok((StatusCode::OK, Json(body)).into_response())
}

/// Pure pagination helper: turn the stored `ResponsesInput` into the
/// OpenAI list-object JSON shape. Separated from the handler so cursor
/// + limit + order logic can be unit-tested without the full AppState.
pub(crate) fn build_input_items_page(
    input: &ResponsesInput,
    params: &ListInputItemsParams,
) -> serde_json::Value {
    let mut items_with_ids: Vec<(String, serde_json::Value)> = match input {
        ResponsesInput::Text(s) => {
            let item_id = "item_0".to_string();
            let v = serde_json::json!({
                "id": item_id,
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": s}],
            });
            vec![(item_id, v)]
        }
        ResponsesInput::Items(items) => items
            .iter()
            .enumerate()
            .map(|(i, item)| {
                let item_id = format!("item_{i}");
                let mut v = serde_json::to_value(item).unwrap_or(serde_json::Value::Null);
                if let Some(obj) = v.as_object_mut() {
                    obj.insert("id".into(), serde_json::Value::String(item_id.clone()));
                }
                (item_id, v)
            })
            .collect(),
    };

    // `order=desc` reverses the stable list before cursoring so `after`
    // semantics still mean "the next page of results" regardless of
    // direction.
    if matches!(params.order.as_deref(), Some("desc")) {
        items_with_ids.reverse();
    }

    let start = match params.after.as_deref() {
        Some(cursor) => items_with_ids
            .iter()
            .position(|(i, _)| i == cursor)
            .map(|i| i + 1)
            .unwrap_or(items_with_ids.len()),
        None => 0,
    };

    let limit = params.limit.unwrap_or(20).clamp(1, 100) as usize;
    let total = items_with_ids.len();
    let end = start.saturating_add(limit).min(total);
    let page: Vec<serde_json::Value> = items_with_ids[start..end]
        .iter()
        .map(|(_, v)| v.clone())
        .collect();

    let first_id = page
        .first()
        .and_then(|v| v.get("id").and_then(|x| x.as_str()))
        .map(String::from);
    let last_id = page
        .last()
        .and_then(|v| v.get("id").and_then(|x| x.as_str()))
        .map(String::from);
    let has_more = end < total;

    serde_json::json!({
        "object": "list",
        "data": page,
        "first_id": first_id,
        "last_id": last_id,
        "has_more": has_more,
    })
}

/// `DELETE /v1/responses/:id` — remove a stored response.
pub async fn delete_response(
    State(state): State<AppState>,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> Result<Response, ApiError> {
    // Check existence so we can return a meaningful success body.
    let existed = store::load(&state.db, &id).map_err(ApiError)?.is_some();
    store::delete(&state.db, &id).map_err(ApiError)?;
    let body = serde_json::json!({
        "id": id,
        "object": "response.deleted",
        "deleted": existed,
    });
    Ok((StatusCode::OK, Json(body)).into_response())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn tools_from(json_value: serde_json::Value) -> Vec<ToolDef> {
        serde_json::from_value(json_value).unwrap()
    }

    #[test]
    fn function_only_returns_none() {
        let tools = tools_from(json!([
            {"type": "function", "name": "f", "parameters": {"type": "object"}},
        ]));
        assert_eq!(first_builtin_tool(&tools), None);
    }

    #[test]
    fn detects_each_builtin_tool() {
        for &kind in BUILTIN_TOOL_TYPES {
            let tools = tools_from(json!([{"type": kind}]));
            assert_eq!(
                first_builtin_tool(&tools),
                Some(kind),
                "expected {kind} to be flagged"
            );
        }
    }

    #[test]
    fn detects_builtin_when_mixed_with_function() {
        let tools = tools_from(json!([
            {"type": "function", "name": "f", "parameters": {"type": "object"}},
            {"type": "web_search"},
            {"type": "function", "name": "g", "parameters": {"type": "object"}},
        ]));
        assert_eq!(first_builtin_tool(&tools), Some("web_search"));
    }

    // ------------------------------------------------------------------
    // V4: input_items pagination
    // ------------------------------------------------------------------

    fn empty_params() -> ListInputItemsParams {
        ListInputItemsParams {
            after: None,
            limit: None,
            order: None,
            before: None,
            include: None,
        }
    }

    #[test]
    fn input_items_text_input_produces_single_synthetic_message() {
        let input = ResponsesInput::Text("hi there".into());
        let page = build_input_items_page(&input, &empty_params());
        assert_eq!(page["object"], "list");
        assert_eq!(page["has_more"], false);
        let data = page["data"].as_array().unwrap();
        assert_eq!(data.len(), 1);
        assert_eq!(data[0]["id"], "item_0");
        assert_eq!(data[0]["type"], "message");
        assert_eq!(data[0]["role"], "user");
        assert_eq!(data[0]["content"][0]["type"], "input_text");
        assert_eq!(data[0]["content"][0]["text"], "hi there");
        assert_eq!(page["first_id"], "item_0");
        assert_eq!(page["last_id"], "item_0");
    }

    #[test]
    fn input_items_array_input_paginates_by_limit() {
        let input: ResponsesInput = serde_json::from_value(json!([
            {"type": "message", "role": "user", "content": "one"},
            {"type": "message", "role": "user", "content": "two"},
            {"type": "message", "role": "user", "content": "three"},
            {"type": "message", "role": "user", "content": "four"},
        ]))
        .unwrap();

        let mut params = empty_params();
        params.limit = Some(2);
        let page = build_input_items_page(&input, &params);
        assert_eq!(page["data"].as_array().unwrap().len(), 2);
        assert_eq!(page["first_id"], "item_0");
        assert_eq!(page["last_id"], "item_1");
        assert_eq!(page["has_more"], true);
    }

    #[test]
    fn input_items_after_cursor_returns_next_page() {
        let input: ResponsesInput = serde_json::from_value(json!([
            {"type": "message", "role": "user", "content": "one"},
            {"type": "message", "role": "user", "content": "two"},
            {"type": "message", "role": "user", "content": "three"},
            {"type": "message", "role": "user", "content": "four"},
        ]))
        .unwrap();

        let mut params = empty_params();
        params.limit = Some(2);
        params.after = Some("item_1".into());
        let page = build_input_items_page(&input, &params);
        let data = page["data"].as_array().unwrap();
        assert_eq!(data.len(), 2);
        assert_eq!(data[0]["id"], "item_2");
        assert_eq!(data[1]["id"], "item_3");
        assert_eq!(page["has_more"], false);
    }

    #[test]
    fn input_items_after_cursor_at_end_returns_empty_page() {
        let input: ResponsesInput = serde_json::from_value(json!([
            {"type": "message", "role": "user", "content": "only"},
        ]))
        .unwrap();
        let mut params = empty_params();
        params.after = Some("item_0".into());
        let page = build_input_items_page(&input, &params);
        assert!(page["data"].as_array().unwrap().is_empty());
        assert_eq!(page["has_more"], false);
        assert_eq!(page["first_id"], serde_json::Value::Null);
        assert_eq!(page["last_id"], serde_json::Value::Null);
    }

    #[test]
    fn input_items_order_desc_reverses_iteration() {
        let input: ResponsesInput = serde_json::from_value(json!([
            {"type": "message", "role": "user", "content": "a"},
            {"type": "message", "role": "user", "content": "b"},
            {"type": "message", "role": "user", "content": "c"},
        ]))
        .unwrap();
        let mut params = empty_params();
        params.order = Some("desc".into());
        let page = build_input_items_page(&input, &params);
        let data = page["data"].as_array().unwrap();
        // All three present because default limit=20 > 3.
        assert_eq!(data.len(), 3);
        // Items returned in reverse order (last first).
        assert_eq!(data[0]["id"], "item_2");
        assert_eq!(data[1]["id"], "item_1");
        assert_eq!(data[2]["id"], "item_0");
    }

    #[test]
    fn input_items_limit_capped_at_100() {
        let input = ResponsesInput::Text("hi".into());
        let mut params = empty_params();
        params.limit = Some(10_000);
        let page = build_input_items_page(&input, &params);
        // 1 item so only 1 returned, but the clamp path shouldn't panic.
        assert_eq!(page["data"].as_array().unwrap().len(), 1);
    }

    #[test]
    fn input_items_function_call_items_preserve_fields_and_add_id() {
        let input: ResponsesInput = serde_json::from_value(json!([
            {"type": "message", "role": "user", "content": "call X"},
            {"type": "function_call", "call_id": "c1", "name": "lookup", "arguments": "{\"q\":\"x\"}"},
            {"type": "function_call_output", "call_id": "c1", "output": "{\"result\":42}"},
        ]))
        .unwrap();
        let page = build_input_items_page(&input, &empty_params());
        let data = page["data"].as_array().unwrap();
        assert_eq!(data.len(), 3);
        assert_eq!(data[0]["id"], "item_0");
        assert_eq!(data[1]["id"], "item_1");
        assert_eq!(data[1]["type"], "function_call");
        assert_eq!(data[1]["name"], "lookup");
        assert_eq!(data[2]["id"], "item_2");
        assert_eq!(data[2]["type"], "function_call_output");
        assert_eq!(data[2]["call_id"], "c1");
    }

    #[test]
    fn unknown_tool_type_does_not_trigger_rejection() {
        // Future / unmodeled tool types round-trip via Raw and should be
        // forwarded (cloud proxy in M5) rather than 400ing here.
        let tools = tools_from(json!([{"type": "future_tool_type_2027"}]));
        assert_eq!(first_builtin_tool(&tools), None);
    }
}
