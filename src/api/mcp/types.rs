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
    /// Serialized even when absent, as an explicit `null`.
    ///
    /// JSON-RPC 2.0 §5 makes `id` REQUIRED on a Response object, and says that
    /// when it could not be determined from the Request — a parse error, an
    /// unusable Request object — it MUST be Null. Skipping it omitted the key
    /// entirely on exactly those paths, so a client that expects a Response to
    /// always carry an `id` did not find one.
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

/// A tool result carrying one text block — the shape every MCP tool returns.
pub(super) fn tool_text_result(id: Option<Value>, text: impl Into<String>) -> JsonRpcResponse {
    tool_text_result_with_route(id, text, None)
}

/// The `_meta` key a tool result names its route under (#251 (3)) — which
/// computers served the model call behind it. Reverse-DNS, as MCP asks of a
/// `_meta` key (2025-11-25 § `_meta`), and the twin of the request-side
/// `com.swarmllm/keep_on_this_computer` (`keep_here::MCP_META_KEY`).
pub(super) const ROUTE_META_KEY: &str = "com.swarmllm/route";

/// [`tool_text_result`] for a tool whose answer is ONE model call's: its route
/// goes in the result's `_meta`, for the program driving the tool rather than
/// in the text the model reads. A JSON-RPC answer has no headers of its own, so
/// this is the only place an MCP caller can learn where a call ran. `None`
/// (no swarm route: a cloud model, a refusal) leaves `_meta` out.
pub(super) fn tool_text_result_with_route(
    id: Option<Value>,
    text: impl Into<String>,
    route: Option<crate::inference::trace::RouteSummary>,
) -> JsonRpcResponse {
    let mut result = serde_json::json!({ "content": [{ "type": "text", "text": text.into() }] });
    if let Some(route) = route {
        let mut meta = serde_json::Map::new();
        meta.insert(
            ROUTE_META_KEY.to_string(),
            serde_json::to_value(route).unwrap_or_default(),
        );
        result["_meta"] = Value::Object(meta);
    }
    JsonRpcResponse::success(id, result)
}

/// The `_meta` key a failed tool result carries its classified error under
/// (#196): the API's own error object (`crate::error::error_body` — `message`,
/// `type`, `code`, `hint`, `hint_key`), for the program driving the tool. The
/// text block is what the model reads; this is what code can branch on.
pub(super) const ERROR_META_KEY: &str = "com.swarmllm/error";

/// A tool that RAN and failed (#196): a JSON-RPC SUCCESS whose result says
/// `isError: true`, the failure in a text block the model reads.
///
/// MCP 2025-11-25 § "Error Handling" splits failures in two. A protocol error
/// (a JSON-RPC `error`) is for an unknown tool, a request that does not fit
/// `CallToolRequest`, and server faults. Everything a model can act on — input
/// validation, API failures, business logic — is a tool execution error, which
/// "clients SHOULD provide … to language models to enable self-correction",
/// while a protocol error MAY never reach the model at all. SEP-1303 moved input
/// validation here for exactly that reason, and the official TypeScript and
/// Python SDKs turn every failure inside a tool into this shape.
///
/// Every tool failure here used to be a JSON-RPC error whose CODE said who
/// should act (`tool_error_code`: `-32602` "fix the call", `-32000` "not right
/// now", `-32603` "broken" — 2026-08-12, when a missing model and an over-long
/// prompt both said `-32603`). That judgement is kept, in a form both readers
/// get: the class goes in [`ERROR_META_KEY`] as the HTTP surfaces' own error
/// object, and its advice goes in the text. The message is the client-safe one
/// every other surface sends: an internal fault reads "An internal error
/// occurred", and the full error goes to this node's log.
pub(super) fn tool_failed(
    id: Option<Value>,
    doing: Option<&str>,
    err: &crate::error::SwarmError,
) -> JsonRpcResponse {
    // A worker or network hop may have flattened the class to text, re-wrapped
    // as a 500 — recover it then, and only then, so a typed error keeps its own.
    let internal =
        crate::error::classify_error(err).0 == axum::http::StatusCode::INTERNAL_SERVER_ERROR;
    let recovered = internal
        .then(|| crate::error::reclassify_flattened_error(&err.to_string()))
        .flatten();
    let err = recovered.as_ref().unwrap_or(err);
    crate::log_failure!(err, error = %err, "MCP tool call failed");

    let body = crate::error::error_body(err);
    let message = body["message"].as_str().unwrap_or_default();
    let mut text = match doing {
        Some(doing) => format!("{doing}: {message}"),
        None => message.to_string(),
    };
    if let Some(hint) = body["hint"].as_str() {
        text.push_str("\n\n");
        text.push_str(hint);
    }
    let mut response = tool_failed_text(id, text);
    if let Some(result) = response.result.as_mut() {
        let mut meta = serde_json::Map::new();
        meta.insert(ERROR_META_KEY.to_string(), body);
        result["_meta"] = Value::Object(meta);
    }
    response
}

/// [`tool_failed`] for an argument the model can correct: a missing field, a
/// value past a limit.
pub(super) fn invalid_input(id: Option<Value>, why: impl Into<String>) -> JsonRpcResponse {
    tool_failed(id, None, &crate::error::SwarmError::Validation(why.into()))
}

/// The bare `isError` result, for a failure that reached the tool only as TEXT
/// — `delegate`'s call to this node's own API, whose error body the text then
/// carries whole. No class is guessed for it: absent is honest where a class
/// read back out of prose would not be.
pub(super) fn tool_failed_text(id: Option<Value>, text: impl Into<String>) -> JsonRpcResponse {
    JsonRpcResponse::success(
        id,
        serde_json::json!({
            "content": [{ "type": "text", "text": text.into() }],
            "isError": true,
        }),
    )
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
/// A `uri` that is well-formed but names no resource this server has. The MCP
/// resources spec names this code specifically, and it is a different situation
/// from [`INVALID_PARAMS`], which says the caller's argument was malformed —
/// asking correctly for something that is not here is not a malformed ask.
pub(super) const RESOURCE_NOT_FOUND: i64 = -32002;

#[cfg(test)]
mod tool_failed_tests {
    use super::*;
    use crate::error::SwarmError;
    use crate::types::ModelId;

    fn as_json(r: JsonRpcResponse) -> Value {
        serde_json::to_value(r).unwrap()
    }

    /// **A tool that ran and failed is a RESULT the model reads, not a
    /// protocol error** (#196, MCP 2025-11-25 § Error Handling): no JSON-RPC
    /// `error`, `isError: true`, the reason in the text — and the class the
    /// HTTP surfaces give the same failure, for the program.
    #[test]
    fn a_failed_tool_is_an_is_error_result_carrying_its_class() {
        let v = as_json(tool_failed(
            Some(json_id()),
            None,
            &SwarmError::ModelNotAvailable(ModelId("nope".into())),
        ));
        assert!(v.get("error").is_none(), "{v}");
        assert_eq!(v["id"], json_id());
        assert_eq!(v["result"]["isError"], true, "{v}");
        let text = v["result"]["content"][0]["text"].as_str().unwrap();
        assert!(text.starts_with("Model not available: nope"), "{text}");
        let class = &v["result"]["_meta"][ERROR_META_KEY];
        assert_eq!(class["type"], "not_found_error", "{v}");
        assert_eq!(class["hint_key"], "model_not_available");
        assert!(
            text.contains(class["hint"].as_str().unwrap()),
            "the advice reaches the model too: {text}"
        );
    }

    /// "Fix the call", "not right now" and "broken" stay three different
    /// classes, as the JSON-RPC codes they replace were.
    #[test]
    fn who_should_act_is_still_told_apart() {
        let class = |e: SwarmError| {
            as_json(tool_failed(None, None, &e))["result"]["_meta"][ERROR_META_KEY]["type"].clone()
        };
        assert_eq!(
            class(SwarmError::Validation("prompt too long".into())),
            "invalid_request_error"
        );
        assert_eq!(
            class(SwarmError::ServiceUnavailable("worker died".into())),
            "server_error"
        );
        assert_eq!(class(SwarmError::Internal("boom".into())), "server_error");
        assert_eq!(
            as_json(invalid_input(None, "Missing required field: model"))["result"]["content"][0]
                ["text"],
            "Validation error: Missing required field: model"
        );
    }

    /// An internal fault is reported with the client-safe message every other
    /// surface sends — the detail stays in this node's log.
    #[test]
    fn an_internal_fault_does_not_reach_the_model_in_detail() {
        let v = as_json(tool_failed(
            None,
            None,
            &SwarmError::Internal("/home/someone/.local/share/swarmllm/db locked".into()),
        ));
        let text = v["result"]["content"][0]["text"].as_str().unwrap();
        assert!(text.starts_with("An internal error occurred"), "{v}");
        assert!(!text.contains("/home"), "{v}");
    }

    /// The class survives a boundary that flattened it to text — and a typed
    /// error is not re-read from its own prose.
    #[test]
    fn a_flattened_class_is_recovered_and_a_typed_one_kept() {
        let flattened = SwarmError::Inference(
            "Validation error: Sequence length (10036) exceeds model context window (4096)".into(),
        );
        let v = as_json(tool_failed(None, Some("Delegating failed"), &flattened));
        assert_eq!(
            v["result"]["_meta"][ERROR_META_KEY]["type"],
            "invalid_request_error"
        );
        assert!(v["result"]["content"][0]["text"]
            .as_str()
            .unwrap()
            .starts_with("Delegating failed: Validation error: Sequence length (10036)"));
        // Its Display carries an earlier marker, so re-reading it would make it
        // a `Validation` error.
        let typed = SwarmError::ModelNotAvailable(ModelId("Validation error: x".into()));
        assert_eq!(
            as_json(tool_failed(None, None, &typed))["result"]["_meta"][ERROR_META_KEY]["type"],
            "not_found_error"
        );
    }

    /// A failure that arrived as text carries no guessed class.
    #[test]
    fn a_text_failure_has_no_class() {
        let v = as_json(tool_failed_text(None, "HTTP 502: upstream"));
        assert_eq!(v["result"]["isError"], true);
        assert!(v["result"].get("_meta").is_none(), "{v}");
    }

    fn json_id() -> Value {
        serde_json::json!(7)
    }
}
