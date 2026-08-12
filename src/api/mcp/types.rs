//! JSON-RPC 2.0 protocol types + MCP error codes.

use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Deserialize)]
pub struct JsonRpcRequest {
    pub jsonrpc: String,
    pub id: Option<Value>,
    pub method: String,
    #[serde(default)]
    pub params: Value,
}

#[derive(Debug, Serialize)]
pub struct JsonRpcResponse {
    pub jsonrpc: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<JsonRpcError>,
}

#[derive(Debug, Serialize)]
pub struct JsonRpcError {
    pub code: i64,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
}

impl JsonRpcResponse {
    pub(super) fn success(id: Option<Value>, result: Value) -> Self {
        Self {
            jsonrpc: "2.0",
            id,
            result: Some(result),
            error: None,
        }
    }

    pub(super) fn error(id: Option<Value>, code: i64, message: impl Into<String>) -> Self {
        Self {
            jsonrpc: "2.0",
            id,
            result: None,
            error: Some(JsonRpcError {
                code,
                message: message.into(),
                data: None,
            }),
        }
    }
}

// JSON-RPC error codes
/// Invalid JSON was received — the body could not be parsed at all.
pub(super) const PARSE_ERROR: i64 = -32700;
/// The JSON parsed, but is not a valid Request object (e.g. wrong `jsonrpc`).
/// Distinct from [`PARSE_ERROR`], which the version check used to return even
/// though the JSON itself was fine.
pub(super) const INVALID_REQUEST: i64 = -32600;
pub(super) const METHOD_NOT_FOUND: i64 = -32601;
pub(super) const INVALID_PARAMS: i64 = -32602;
pub(super) const INTERNAL_ERROR: i64 = -32603;
/// Application-level error: resource unavailable (no models loaded).
pub(super) const RESOURCE_UNAVAILABLE: i64 = -32000;

/// The JSON-RPC code for a failure raised while serving a tool call.
///
/// Derived from `crate::error::classify_error`, so the code follows the same
/// judgement the HTTP surfaces make about the same failure instead of a
/// separate one made here.
///
/// Every inference failure used to be `INTERNAL_ERROR`: a model name that does
/// not exist and a prompt longer than the context both came back as "the server
/// had an internal error", when this file was already careful to answer
/// `INVALID_PARAMS` for its OWN argument checks a few lines earlier (measured
/// 2026-08-12). An MCP client reads `-32603` as "the tool is broken" and
/// `-32602` as "fix the call" — the difference decides whether it retries
/// forever or tells the user.
pub(super) fn tool_error_code(err: &crate::error::SwarmError) -> i64 {
    // The class may already have been flattened to text by a worker or network
    // hop, in which case recover it first — same reason as everywhere else.
    let recovered = crate::error::reclassify_flattened_error(&err.to_string());
    let effective = recovered.as_ref().unwrap_or(err);
    let (status, _msg, _ty) = crate::error::classify_error(effective);
    if status == axum::http::StatusCode::SERVICE_UNAVAILABLE {
        // The node cannot serve this right now — the caller's request is fine.
        RESOURCE_UNAVAILABLE
    } else if status.is_client_error() {
        INVALID_PARAMS
    } else {
        INTERNAL_ERROR
    }
}

#[cfg(test)]
mod tool_error_code_tests {
    use super::*;
    use crate::error::SwarmError;
    use crate::types::ModelId;

    /// A caller's own mistake must reach an MCP client as "fix the call", not
    /// "the tool is broken" — the two lead a client to opposite behaviour.
    #[test]
    fn a_callers_mistake_is_invalid_params_not_internal_error() {
        for err in [
            SwarmError::Validation("prompt too long".into()),
            SwarmError::ModelNotAvailable(ModelId("nope".into())),
            SwarmError::NotFound("session".into()),
        ] {
            assert_eq!(
                tool_error_code(&err),
                INVALID_PARAMS,
                "{err} must be invalid-params"
            );
        }
    }

    /// "Not right now" is neither the caller's fault nor a bug.
    #[test]
    fn a_node_that_cannot_serve_is_resource_unavailable() {
        assert_eq!(
            tool_error_code(&SwarmError::ServiceUnavailable("worker died".into())),
            RESOURCE_UNAVAILABLE
        );
    }

    /// A genuine bug must still say so, or it hides.
    #[test]
    fn an_actual_fault_stays_internal_error() {
        assert_eq!(
            tool_error_code(&SwarmError::Internal("boom".into())),
            INTERNAL_ERROR
        );
    }

    /// The class survives a boundary that flattened it to text — the same
    /// recovery every other surface now does.
    #[test]
    fn a_flattened_class_is_recovered_before_coding() {
        let flattened = SwarmError::Inference(
            "Validation error: Sequence length (10036) exceeds model context window (4096)".into(),
        );
        assert_eq!(tool_error_code(&flattened), INVALID_PARAMS);
    }
}
