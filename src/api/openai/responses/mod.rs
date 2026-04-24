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
//! - **M9 (current)**: background=true + POST .../cancel.

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
    // ---- 1. Cloud proxy passthrough (M5). ----
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

    #[test]
    fn unknown_tool_type_does_not_trigger_rejection() {
        // Future / unmodeled tool types round-trip via Raw and should be
        // forwarded (cloud proxy in M5) rather than 400ing here.
        let tools = tools_from(json!([{"type": "future_tool_type_2027"}]));
        assert_eq!(first_builtin_tool(&tools), None);
    }
}
