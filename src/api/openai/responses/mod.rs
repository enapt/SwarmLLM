//! OpenAI `/v1/responses` endpoint — request/response types and (in later
//! milestones) handlers, translation, streaming, and persistence.
//!
//! Milestone 2 wires the route + the built-in-tool rejection. Anything past
//! the rejection currently returns 501 Not Implemented; M3 fills in local
//! inference, M5 adds the cloud-proxy passthrough.

pub mod types;

pub use types::*;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;

use crate::api::server::{AppState, JsonBody};
use crate::error::{ApiError, SwarmError};

/// Tool `type` strings that map to OpenAI-hosted infrastructure SwarmLLM
/// does not run. Listed in the plan under "Tool types we accept and pass
/// through to local inference as `function`: just the `function` tool."
///
/// Order matters only for the error message; we surface the first one we
/// find so the caller gets a single, specific name to fix.
pub(crate) const BUILTIN_TOOL_TYPES: &[&str] = &[
    "web_search",
    "file_search",
    "computer_use_preview",
    "code_interpreter",
    "image_generation",
    "mcp",
    "custom",
];

/// Walk a tools array and return the first built-in tool type encountered,
/// or `None` if every entry is a `function` (or unknown — those round-trip
/// via Raw and we don't preemptively reject).
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

/// `POST /v1/responses` — Milestone 2 stub.
///
/// Parses the request, rejects built-in tools with a clear 400, and
/// returns 501 for everything else. M3 replaces the 501 path with local
/// inference via the Chat Completions translation.
pub async fn create_response(
    State(_state): State<AppState>,
    _headers: axum::http::HeaderMap,
    JsonBody(req): JsonBody<ResponsesRequest>,
) -> Result<Response, ApiError> {
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

    let body = serde_json::json!({
        "error": {
            "message": "/v1/responses is not yet implemented on this server. \
                        Use /v1/chat/completions for OpenAI-compatible inference.",
            "type": "not_implemented",
            "param": null,
            "code": "not_implemented",
        }
    });
    Ok((StatusCode::NOT_IMPLEMENTED, Json(body)).into_response())
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
