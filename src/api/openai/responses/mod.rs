//! OpenAI `/v1/responses` endpoint — request/response types, translation
//! to/from Chat Completions, and HTTP handler.
//!
//! Milestone scope:
//! - **M1**: types + serde roundtrip.
//! - **M2**: route wired, built-in-tool rejection, 501 stub.
//! - **M3**: plain-text local inference via Chat translation.
//! - **M4 (current)**: function tools and tool_choice translation.
//! - **M5**: cloud-proxy verbatim path.
//! - **M6**: SSE streaming.
//! - **M7-M9**: persistence, chaining, background.

pub mod store;
pub mod stream;
pub mod translate;
pub mod types;

pub use types::*;

use axum::body::to_bytes;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;

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

/// Build a 501 JSON response with a stable shape.
fn not_implemented(message: &str) -> Response {
    let body = serde_json::json!({
        "error": {
            "message": message,
            "type": "not_implemented",
            "param": null,
            "code": "not_implemented",
        }
    });
    (StatusCode::NOT_IMPLEMENTED, Json(body)).into_response()
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

    // Streaming (M6) — local-inference SSE. Cloud proxy already streamed
    // above if it matched.
    if stream {
        return stream::run_streaming(state, headers, req).await;
    }

    // Background mode — wired in M9.
    if req.background.unwrap_or(false) {
        return Ok(not_implemented(
            "background=true on /v1/responses is not yet implemented \
             (planned for M9).",
        ));
    }

    // previous_response_id — wired in M8 for the local path. (Cloud proxy
    // already returned above with the field forwarded verbatim.)
    if req.previous_response_id.is_some() {
        return Ok(not_implemented(
            "previous_response_id chaining is not yet implemented for local \
             models (planned for M8). Pass the prior turn's messages directly \
             via `input` for now.",
        ));
    }

    // Translate to a Chat Completions request and call the existing
    // handler. Translation failures bubble up as 400 via Validation.
    let chat_req = translate::request_to_chat(&req)?;

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
