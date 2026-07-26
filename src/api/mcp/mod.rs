//! MCP (Model Context Protocol) server — Streamable HTTP transport.
//!
//! Implements the MCP 2025-11-25 specification over Streamable HTTP:
//! - POST for client→server JSON-RPC requests and notifications
//! - Notifications (no `id`) return HTTP 202 with no body
//! - GET for server→client SSE stream (returns 405 — not needed for tool-only servers)
//! - DELETE for session termination (returns 200)
//!
//! Exposes SwarmLLM as an MCP server so Claude Code, Cursor, VS Code Copilot,
//! and other MCP clients can use local/swarm models directly.

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::Json;
use serde_json::{json, Value};

use crate::api::server::AppState;

mod dispatch;
mod resources;
mod tools;
mod tools_list;
mod types;

use resources::{handle_resources_list, handle_resources_read};
use tools::handle_tools_call;
use tools_list::handle_tools_list;
use types::{JsonRpcRequest, JsonRpcResponse, METHOD_NOT_FOUND, PARSE_ERROR};

const MCP_PROTOCOL_VERSION: &str = "2025-11-25";

/// Revisions this server can speak, newest first.
///
/// The features we implement — `tools/list`, `tools/call`, `resources/list`,
/// `resources/read` — are identical across these revisions, so a client pinned
/// to an older one is fully served.
const SUPPORTED_PROTOCOL_VERSIONS: &[&str] =
    &["2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05"];
const SERVER_NAME: &str = "swarmllm";
const SERVER_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Server instructions for MCP Tool Search discovery.
/// Claude Code reads this to know when to load our tools.
const SERVER_INSTRUCTIONS: &str = "SwarmLLM is a decentralized LLM inference network. \
Use these tools when you need to: run inference on local/network models, \
compare multiple models side-by-side, fan out research questions to many models in parallel, \
execute batch prompts across different models concurrently, or check node status. \
The 'research' tool is especially useful for getting diverse perspectives from multiple \
models without using expensive API tokens. The 'batch_prompts' tool lets you offload \
independent subtasks to specific models in parallel (e.g., summarize with one model, \
translate with another, review code with a third).";

/// Maximum prompt/question length for MCP tool inputs (4 MB, matches HTTP validation).
pub(crate) const MCP_MAX_PROMPT_BYTES: usize = 4 * 1024 * 1024;
/// Maximum length (bytes) of a model id supplied to an MCP tool (matches the
/// `tool_chat` guard at `tools.rs:45`). Without this cap the same string in
/// `batch_prompts` would be cloned per-task before being rejected downstream.
pub(crate) const MCP_MAX_MODEL_ID_BYTES: usize = 256;
/// Maximum length (bytes) of a caller-supplied task id in `batch_prompts`. The
/// id is embedded in the response and cloned into per-task closures, so an
/// uncapped string is a per-batch heap-amplification vector.
pub(crate) const MCP_MAX_TASK_ID_BYTES: usize = 256;

// ---- MCP handlers ----

/// POST /mcp — handles JSON-RPC requests and notifications.
/// Per Streamable HTTP spec: notifications (no `id`) get HTTP 202 with no body.
pub async fn handle_mcp(
    State(state): State<AppState>,
    Json(req): Json<JsonRpcRequest>,
) -> impl IntoResponse {
    if req.jsonrpc != "2.0" {
        return (
            StatusCode::OK,
            Json(Some(JsonRpcResponse::error(
                req.id,
                PARSE_ERROR,
                "Invalid JSON-RPC version",
            ))),
        );
    }

    // JSON-RPC notifications have no `id` — per MCP spec, return HTTP 202 with no body
    let is_notification = req.id.is_none();

    let response = match req.method.as_str() {
        "initialize" => handle_initialize(req.id, &req.params),
        "notifications/initialized" | "notifications/cancelled" => {
            if is_notification {
                return (StatusCode::ACCEPTED, Json(None));
            }
            JsonRpcResponse::success(req.id, json!({}))
        }
        "tools/list" => handle_tools_list(req.id),
        "tools/call" => handle_tools_call(&state, req.id, req.params).await,
        "resources/list" => handle_resources_list(req.id),
        "resources/read" => handle_resources_read(&state, req.id, req.params).await,
        "ping" => JsonRpcResponse::success(req.id, json!({})),
        // Explicit error arm for `sampling/createMessage`: clients
        // (Claude Code, Cursor) sometimes try this method to ask the
        // server to perform completions. SwarmLLM is a tools-only MCP
        // server — inference is exposed via tools/call, not sampling.
        // A bare METHOD_NOT_FOUND with the literal method name is hard
        // to debug client-side; spell it out instead.
        "sampling/createMessage" => JsonRpcResponse::error(
            req.id,
            METHOD_NOT_FOUND,
            "sampling/createMessage is not implemented. SwarmLLM is a \
             tools-only MCP server — invoke inference via the `tools/call` \
             method (use `tools/list` to discover available tools).",
        ),
        _ if is_notification => {
            // Unknown notification — silently accept per spec
            return (StatusCode::ACCEPTED, Json(None));
        }
        _ => {
            // SEC: cap the reflected method string. The body is bounded by
            // the global 32 MB request cap, but echoing 32 MB of caller-
            // controlled UTF-8 in error responses (and structured logs)
            // is amplification we can avoid cheaply.
            const REFLECT_CAP: usize = 64;
            let mut shown = req.method.clone();
            if shown.len() > REFLECT_CAP {
                let mut end = REFLECT_CAP;
                while end > 0 && !shown.is_char_boundary(end) {
                    end -= 1;
                }
                shown.truncate(end);
                shown.push_str("...");
            }
            JsonRpcResponse::error(
                req.id,
                METHOD_NOT_FOUND,
                format!("Method not found: {shown}"),
            )
        }
    };

    (StatusCode::OK, Json(Some(response)))
}

/// GET /mcp — SSE stream for server-initiated messages.
/// We don't initiate server→client messages, so return 405.
pub async fn handle_mcp_get() -> impl IntoResponse {
    StatusCode::METHOD_NOT_ALLOWED
}

/// DELETE /mcp — session termination.
/// We're stateless per-request, so just acknowledge.
pub async fn handle_mcp_delete() -> impl IntoResponse {
    StatusCode::OK
}

// ---- Protocol handlers ----

/// Answer `initialize`, echoing the client's protocol revision when we speak it.
///
/// The spec is a MUST: "If the server supports the requested protocol version,
/// it MUST respond with the same version. Otherwise, the server MUST respond
/// with another protocol version it supports" — and a client that receives a
/// version it does not support SHOULD DISCONNECT. Answering with our own
/// newest revision unconditionally therefore turned away every client pinned to
/// an older one, even though our tool surface is identical across all of them
/// (live 2026-07-26).
fn handle_initialize(id: Option<Value>, params: &Value) -> JsonRpcResponse {
    let requested = params.get("protocolVersion").and_then(Value::as_str);
    let version = match requested {
        Some(v) if SUPPORTED_PROTOCOL_VERSIONS.contains(&v) => v,
        // Unknown or absent: answer with our newest, as the spec directs.
        _ => MCP_PROTOCOL_VERSION,
    };
    JsonRpcResponse::success(
        id,
        json!({
            "protocolVersion": version,
            "serverInfo": {
                "name": SERVER_NAME,
                "version": SERVER_VERSION,
            },
            "capabilities": {
                "tools": { "listChanged": false },
                "resources": { "listChanged": false },
            },
            "instructions": SERVER_INSTRUCTIONS,
        }),
    )
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jsonrpc_request_deserializes() {
        let json = r#"{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}"#;
        let req: JsonRpcRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.method, "initialize");
        assert_eq!(req.id, Some(Value::Number(1.into())));
    }

    #[test]
    fn jsonrpc_request_deserializes_string_id() {
        let json = r#"{"jsonrpc": "2.0", "id": "abc-123", "method": "ping", "params": {}}"#;
        let req: JsonRpcRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.id, Some(Value::String("abc-123".into())));
    }

    #[test]
    fn jsonrpc_request_deserializes_no_id() {
        let json = r#"{"jsonrpc": "2.0", "method": "ping"}"#;
        let req: JsonRpcRequest = serde_json::from_str(json).unwrap();
        assert!(req.id.is_none());
        assert_eq!(req.params, Value::Null);
    }

    #[test]
    fn success_response_serializes() {
        let resp = JsonRpcResponse::success(Some(Value::Number(1.into())), json!({"ok": true}));
        let s = serde_json::to_string(&resp).unwrap();
        assert!(s.contains("\"jsonrpc\":\"2.0\""));
        assert!(s.contains("\"result\""));
        assert!(!s.contains("\"error\""));
    }

    #[test]
    fn error_response_serializes() {
        let resp =
            JsonRpcResponse::error(Some(Value::Number(1.into())), -32601, "Method not found");
        let s = serde_json::to_string(&resp).unwrap();
        assert!(s.contains("\"error\""));
        assert!(s.contains("-32601"));
        assert!(!s.contains("\"result\""));
    }

    #[test]
    fn initialize_response_structure() {
        let resp = handle_initialize(Some(Value::Number(1.into())), &Value::Null);
        let result = resp.result.unwrap();
        assert_eq!(result["protocolVersion"], MCP_PROTOCOL_VERSION);
        assert_eq!(result["serverInfo"]["name"], SERVER_NAME);
        assert!(result["capabilities"]["tools"].is_object());
        assert!(result["capabilities"]["resources"].is_object());
        // Server instructions for Tool Search
        assert!(result["instructions"]
            .as_str()
            .unwrap()
            .contains("SwarmLLM"));
    }

    #[test]
    fn tools_list_response_structure() {
        let resp = handle_tools_list(Some(Value::Number(1.into())));
        let result = resp.result.unwrap();
        let tools = result["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 7); // chat, models, compare, research, batch_prompts, delegate, node_info

        let chat_tool = &tools[0];
        assert_eq!(chat_tool["name"], "chat");
        assert!(chat_tool["inputSchema"]["properties"]["model"].is_object());
        assert!(chat_tool["inputSchema"]["properties"]["messages"].is_object());
        assert_eq!(chat_tool["annotations"]["readOnlyHint"], true);

        let models_tool = &tools[1];
        assert_eq!(models_tool["name"], "models");

        let compare_tool = &tools[2];
        assert_eq!(compare_tool["name"], "compare");
        assert!(compare_tool["inputSchema"]["properties"]["prompt"].is_object());
        assert!(compare_tool["inputSchema"]["properties"]["models"].is_object());

        let research_tool = &tools[3];
        assert_eq!(research_tool["name"], "research");
        assert!(research_tool["inputSchema"]["properties"]["question"].is_object());

        let batch_tool = &tools[4];
        assert_eq!(batch_tool["name"], "batch_prompts");
        assert!(batch_tool["inputSchema"]["properties"]["tasks"].is_object());

        let delegate_tool = &tools[5];
        assert_eq!(delegate_tool["name"], "delegate");
        assert!(delegate_tool["inputSchema"]["properties"]["tier"].is_object());

        let node_info_tool = &tools[6];
        assert_eq!(node_info_tool["name"], "node_info");
    }

    #[test]
    fn resources_list_response_structure() {
        let resp = handle_resources_list(Some(Value::Number(1.into())));
        let result = resp.result.unwrap();
        let resources = result["resources"].as_array().unwrap();
        assert_eq!(resources.len(), 3); // status, models, peers
        assert_eq!(resources[0]["uri"], "swarmllm://status");
        assert_eq!(resources[1]["uri"], "swarmllm://models");
        assert_eq!(resources[2]["uri"], "swarmllm://peers");
    }
}

#[cfg(test)]
mod version_negotiation_tests {
    use super::*;

    fn init_with(version: Option<&str>) -> String {
        let params = match version {
            Some(v) => json!({ "protocolVersion": v, "capabilities": {} }),
            None => json!({ "capabilities": {} }),
        };
        let resp = handle_initialize(Some(json!(1)), &params);
        serde_json::to_value(&resp).unwrap()["result"]["protocolVersion"]
            .as_str()
            .unwrap()
            .to_string()
    }

    /// The spec is a MUST: a supported version is echoed back verbatim.
    /// Answering with our own newest instead makes a client pinned to an older
    /// revision disconnect, per the same section.
    #[test]
    fn supported_client_version_is_echoed() {
        for v in SUPPORTED_PROTOCOL_VERSIONS {
            assert_eq!(&init_with(Some(v)), v, "must echo supported version {v}");
        }
    }

    /// An unknown version gets our newest, which the spec directs and which
    /// lets the client decide whether to proceed.
    #[test]
    fn unknown_client_version_gets_our_newest() {
        assert_eq!(init_with(Some("1.0.0")), MCP_PROTOCOL_VERSION);
        assert_eq!(init_with(Some("2099-01-01")), MCP_PROTOCOL_VERSION);
    }

    /// A missing protocolVersion must not panic — answer with our newest.
    #[test]
    fn absent_client_version_gets_our_newest() {
        assert_eq!(init_with(None), MCP_PROTOCOL_VERSION);
    }

    /// Our advertised newest must itself be in the supported set, or the echo
    /// logic and the fallback disagree.
    #[test]
    fn advertised_version_is_in_the_supported_set() {
        assert!(SUPPORTED_PROTOCOL_VERSIONS.contains(&MCP_PROTOCOL_VERSION));
        assert_eq!(
            SUPPORTED_PROTOCOL_VERSIONS[0], MCP_PROTOCOL_VERSION,
            "supported list must be newest-first"
        );
    }
}
