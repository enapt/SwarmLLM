//! MCP (Model Context Protocol) server — JSON-RPC 2.0 over HTTP POST.
//!
//! Exposes SwarmLLM as an MCP server so Claude Code, Cursor, VS Code Copilot,
//! and other MCP clients can use local/swarm models directly.

use axum::extract::State;
use axum::response::IntoResponse;
use axum::Json;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::api::server::AppState;

const MCP_PROTOCOL_VERSION: &str = "2024-11-05";
const SERVER_NAME: &str = "swarmllm";
const SERVER_VERSION: &str = env!("CARGO_PKG_VERSION");

// ---- JSON-RPC 2.0 types ----

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
    fn success(id: Option<Value>, result: Value) -> Self {
        Self {
            jsonrpc: "2.0",
            id,
            result: Some(result),
            error: None,
        }
    }

    fn error(id: Option<Value>, code: i64, message: impl Into<String>) -> Self {
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
const PARSE_ERROR: i64 = -32700;
const METHOD_NOT_FOUND: i64 = -32601;
const INVALID_PARAMS: i64 = -32602;
const INTERNAL_ERROR: i64 = -32603;

// ---- MCP handler ----

pub async fn handle_mcp(
    State(state): State<AppState>,
    Json(req): Json<JsonRpcRequest>,
) -> impl IntoResponse {
    if req.jsonrpc != "2.0" {
        return Json(JsonRpcResponse::error(
            req.id,
            PARSE_ERROR,
            "Invalid JSON-RPC version",
        ));
    }

    let response = match req.method.as_str() {
        "initialize" => handle_initialize(req.id),
        "notifications/initialized" => JsonRpcResponse::success(req.id, json!({})),
        "tools/list" => handle_tools_list(req.id),
        "tools/call" => handle_tools_call(&state, req.id, req.params).await,
        "resources/list" => handle_resources_list(req.id),
        "resources/read" => handle_resources_read(&state, req.id, req.params).await,
        "ping" => JsonRpcResponse::success(req.id, json!({})),
        _ => JsonRpcResponse::error(
            req.id,
            METHOD_NOT_FOUND,
            format!("Method not found: {}", req.method),
        ),
    };

    Json(response)
}

// ---- Protocol handlers ----

fn handle_initialize(id: Option<Value>) -> JsonRpcResponse {
    JsonRpcResponse::success(
        id,
        json!({
            "protocolVersion": MCP_PROTOCOL_VERSION,
            "serverInfo": {
                "name": SERVER_NAME,
                "version": SERVER_VERSION,
            },
            "capabilities": {
                "tools": {},
                "resources": {},
            },
        }),
    )
}

fn handle_tools_list(id: Option<Value>) -> JsonRpcResponse {
    JsonRpcResponse::success(
        id,
        json!({
            "tools": [
                {
                    "name": "chat",
                    "description": "Send a chat completion request to the SwarmLLM inference engine",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "model": {
                                "type": "string",
                                "description": "Model ID to use for inference"
                            },
                            "messages": {
                                "type": "array",
                                "description": "Array of chat messages",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "role": { "type": "string", "enum": ["system", "user", "assistant"] },
                                        "content": { "type": "string" }
                                    },
                                    "required": ["role", "content"]
                                }
                            },
                            "temperature": {
                                "type": "number",
                                "description": "Sampling temperature (0.0-2.0)"
                            },
                            "max_tokens": {
                                "type": "integer",
                                "description": "Maximum tokens to generate"
                            }
                        },
                        "required": ["model", "messages"]
                    }
                },
                {
                    "name": "models",
                    "description": "List available models on this SwarmLLM node and network",
                    "inputSchema": {
                        "type": "object",
                        "properties": {}
                    }
                },
                {
                    "name": "compare",
                    "description": "Send the same prompt to multiple models concurrently and return all responses side-by-side for comparison. Supports local, network, and cloud models.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "prompt": {
                                "type": "string",
                                "description": "The prompt to send to all models"
                            },
                            "system": {
                                "type": "string",
                                "description": "Optional system prompt"
                            },
                            "models": {
                                "type": "array",
                                "description": "Array of model IDs to compare (e.g. [\"qwen2.5-coder-7b\", \"gpt-4o\", \"claude-sonnet-4-6\"])",
                                "items": { "type": "string" }
                            },
                            "temperature": {
                                "type": "number",
                                "description": "Sampling temperature (0.0-2.0, default 0.7)"
                            },
                            "max_tokens": {
                                "type": "integer",
                                "description": "Maximum tokens per response (default 1024)"
                            }
                        },
                        "required": ["prompt", "models"]
                    }
                }
            ]
        }),
    )
}

async fn handle_tools_call(state: &AppState, id: Option<Value>, params: Value) -> JsonRpcResponse {
    let tool_name = params.get("name").and_then(|v| v.as_str()).unwrap_or("");
    let arguments = params.get("arguments").cloned().unwrap_or(json!({}));

    match tool_name {
        "chat" => tool_chat(state, id, arguments).await,
        "models" => tool_models(state, id).await,
        "compare" => tool_compare(state, id, arguments).await,
        _ => JsonRpcResponse::error(id, INVALID_PARAMS, format!("Unknown tool: {tool_name}")),
    }
}

fn handle_resources_list(id: Option<Value>) -> JsonRpcResponse {
    JsonRpcResponse::success(
        id,
        json!({
            "resources": [
                {
                    "uri": "swarmllm://status",
                    "name": "Node Status",
                    "description": "Current SwarmLLM node status including loaded models and peer count",
                    "mimeType": "application/json"
                }
            ]
        }),
    )
}

async fn handle_resources_read(
    state: &AppState,
    id: Option<Value>,
    params: Value,
) -> JsonRpcResponse {
    let uri = params.get("uri").and_then(|v| v.as_str()).unwrap_or("");

    match uri {
        "swarmllm://status" => resource_status(state, id).await,
        _ => JsonRpcResponse::error(id, INVALID_PARAMS, format!("Unknown resource: {uri}")),
    }
}

// ---- Tool implementations ----

async fn tool_chat(state: &AppState, id: Option<Value>, args: Value) -> JsonRpcResponse {
    let router_tx = match &state.router_tx {
        Some(tx) => tx.clone(),
        None => {
            return JsonRpcResponse::error(id, INTERNAL_ERROR, "Inference router not available");
        }
    };

    let model = match args.get("model").and_then(|v| v.as_str()) {
        Some(m) => m.to_string(),
        None => {
            return JsonRpcResponse::error(id, INVALID_PARAMS, "Missing required field: model");
        }
    };

    let messages = match args.get("messages").and_then(|v| v.as_array()) {
        Some(msgs) => msgs.clone(),
        None => {
            return JsonRpcResponse::error(id, INVALID_PARAMS, "Missing required field: messages");
        }
    };

    let temperature = args
        .get("temperature")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.7) as f32;
    let max_tokens = args
        .get("max_tokens")
        .and_then(|v| v.as_u64())
        .unwrap_or(512) as u32;

    let chat_messages: Vec<crate::types::ChatMessage> = messages
        .iter()
        .filter_map(|m| {
            let role_str = m.get("role")?.as_str()?;
            let role = match role_str {
                "system" => crate::types::Role::System,
                "user" => crate::types::Role::User,
                "assistant" => crate::types::Role::Assistant,
                "tool" => crate::types::Role::Tool,
                _ => return None,
            };
            let content = m.get("content")?.as_str()?.to_string();
            Some(crate::types::ChatMessage {
                role,
                content,
                images: vec![],
            })
        })
        .collect();

    if chat_messages.is_empty() {
        return JsonRpcResponse::error(id, INVALID_PARAMS, "No valid messages provided");
    }

    let request = crate::types::InferenceRequest {
        id: uuid::Uuid::new_v4(),
        model_id: crate::types::ModelId(model),
        messages: chat_messages,
        sampling_params: crate::types::SamplingParams {
            temperature,
            top_p: 0.9,
            top_k: 40,
            max_tokens,
            stop: vec![],
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            logprobs: false,
            top_logprobs: 0,
        },
        stream: false,
        priority: crate::types::PriorityTier::Silver,
        requester: crate::types::NodeId([0u8; 32]),
        created_at: chrono::Utc::now(),
        session_id: None,
        lora_adapter: None,
    };

    let (result_tx, result_rx) = tokio::sync::oneshot::channel();
    let cmd = crate::inference::router::RouterCommand::Submit { request, result_tx };

    if router_tx.send(cmd).await.is_err() {
        return JsonRpcResponse::error(id, INTERNAL_ERROR, "Failed to send inference request");
    }

    match tokio::time::timeout(std::time::Duration::from_secs(120), result_rx).await {
        Ok(Ok(Ok(response))) => JsonRpcResponse::success(
            id,
            json!({
                "content": [
                    {
                        "type": "text",
                        "text": response.content
                    }
                ]
            }),
        ),
        Ok(Ok(Err(e))) => {
            JsonRpcResponse::error(id, INTERNAL_ERROR, format!("Inference error: {e}"))
        }
        Ok(Err(_)) => JsonRpcResponse::error(id, INTERNAL_ERROR, "Inference channel closed"),
        Err(_) => JsonRpcResponse::error(id, INTERNAL_ERROR, "Inference request timed out"),
    }
}

async fn tool_models(state: &AppState, id: Option<Value>) -> JsonRpcResponse {
    let mut models = vec![];
    let mut seen = std::collections::HashSet::new();

    // Local loaded model
    if let Some(info) = state.shared_state.loaded_model_info.read().await.as_ref() {
        let slug = info
            .name
            .to_lowercase()
            .replace(' ', "-")
            .replace(|c: char| !c.is_alphanumeric() && c != '-' && c != '.', "");
        seen.insert(slug.clone());
        seen.insert(info.name.clone());
        models.push(json!({
            "id": slug,
            "name": info.name,
            "source": "local",
        }));
    }

    // Registry models
    for manifest in state.shared_state.model_registry.models() {
        if !seen.contains(&manifest.id.0) {
            seen.insert(manifest.id.0.clone());
            models.push(json!({
                "id": manifest.id.0,
                "name": manifest.name,
                "source": "network",
            }));
        }
    }

    // Cloud provider models
    for entry in state.shared_state.provider_model_map.iter() {
        let model_id = entry.key().clone();
        if !seen.contains(&model_id) {
            seen.insert(model_id.clone());
            models.push(json!({
                "id": model_id,
                "name": model_id,
                "source": "cloud",
            }));
        }
    }

    JsonRpcResponse::success(
        id,
        json!({
            "content": [
                {
                    "type": "text",
                    "text": serde_json::to_string_pretty(&models).unwrap_or_default()
                }
            ]
        }),
    )
}

/// Compare tool: sends the same prompt to multiple models concurrently.
async fn tool_compare(state: &AppState, id: Option<Value>, args: Value) -> JsonRpcResponse {
    let prompt = match args.get("prompt").and_then(|v| v.as_str()) {
        Some(p) => p.to_string(),
        None => {
            return JsonRpcResponse::error(id, INVALID_PARAMS, "Missing required field: prompt");
        }
    };

    let models: Vec<String> = match args.get("models").and_then(|v| v.as_array()) {
        Some(arr) => arr
            .iter()
            .filter_map(|v| v.as_str().map(|s| s.to_string()))
            .collect(),
        None => {
            return JsonRpcResponse::error(id, INVALID_PARAMS, "Missing required field: models");
        }
    };

    if models.is_empty() {
        return JsonRpcResponse::error(id, INVALID_PARAMS, "models array must not be empty");
    }
    if models.len() > 10 {
        return JsonRpcResponse::error(id, INVALID_PARAMS, "Maximum 10 models per comparison");
    }

    let system = args
        .get("system")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());
    let temperature = args
        .get("temperature")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.7) as f32;
    let max_tokens = args
        .get("max_tokens")
        .and_then(|v| v.as_u64())
        .unwrap_or(1024) as u32;

    // Build the Anthropic Messages API request body for each model
    let mut messages = Vec::new();
    if let Some(ref sys) = system {
        messages.push(json!({"role": "system", "content": sys}));
    }
    messages.push(json!({"role": "user", "content": prompt}));

    // Fire all model requests concurrently using the internal /v1/messages endpoint
    let client = crate::api::providers::get_provider_client();
    let port = state.config.node.listen_port;
    let base = format!("http://127.0.0.1:{port}");

    // Get the auth token for self-requests
    let api_key = state.shared_state.api_key.clone();

    let mut handles = Vec::new();
    for model_id in &models {
        let url = format!("{base}/v1/messages");
        let client = client.clone();
        let model_id = model_id.clone();
        let api_key = api_key.clone();
        let system_val = system.clone();
        let prompt_clone = prompt.clone();

        let handle = tokio::spawn(async move {
            let start = std::time::Instant::now();

            let mut body = json!({
                "model": model_id,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": [{"role": "user", "content": prompt_clone}],
                "stream": false,
            });
            if let Some(sys) = system_val {
                body["system"] = json!(sys);
            }

            let result = client
                .post(&url)
                .header("Authorization", format!("Bearer {api_key}"))
                .header("Content-Type", "application/json")
                .json(&body)
                .send()
                .await;

            let elapsed_ms = start.elapsed().as_millis() as u64;

            match result {
                Ok(resp) if resp.status().is_success() => {
                    let resp_body: serde_json::Value =
                        resp.json().await.unwrap_or(json!({"error": "parse failed"}));
                    // Extract text from Anthropic response
                    let content = resp_body["content"]
                        .as_array()
                        .and_then(|arr| {
                            arr.iter()
                                .filter_map(|b| b.get("text").and_then(|t| t.as_str()))
                                .next()
                        })
                        .unwrap_or("")
                        .to_string();
                    let input_tokens = resp_body["usage"]["input_tokens"].as_u64().unwrap_or(0);
                    let output_tokens = resp_body["usage"]["output_tokens"].as_u64().unwrap_or(0);

                    json!({
                        "model": model_id,
                        "content": content,
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "latency_ms": elapsed_ms,
                        "status": "ok",
                    })
                }
                Ok(resp) => {
                    let status = resp.status().as_u16();
                    let body = resp.text().await.unwrap_or_default();
                    json!({
                        "model": model_id,
                        "error": format!("HTTP {status}: {body}"),
                        "latency_ms": elapsed_ms,
                        "status": "error",
                    })
                }
                Err(e) => {
                    json!({
                        "model": model_id,
                        "error": format!("{e}"),
                        "latency_ms": elapsed_ms,
                        "status": "error",
                    })
                }
            }
        });
        handles.push(handle);
    }

    // Collect all results
    let mut results = Vec::new();
    for handle in handles {
        match handle.await {
            Ok(result) => results.push(result),
            Err(e) => results.push(json!({"error": format!("Task failed: {e}"), "status": "error"})),
        }
    }

    let summary = json!({
        "prompt": prompt,
        "models_compared": models.len(),
        "results": results,
    });

    JsonRpcResponse::success(
        id,
        json!({
            "content": [
                {
                    "type": "text",
                    "text": serde_json::to_string_pretty(&summary).unwrap_or_default()
                }
            ]
        }),
    )
}

// ---- Resource implementations ----

async fn resource_status(state: &AppState, id: Option<Value>) -> JsonRpcResponse {
    let info = state.shared_state.loaded_model_info.read().await;
    let model_name = info.as_ref().map(|i| i.name.clone()).unwrap_or_default();
    let model_loaded = info.is_some();
    drop(info);

    let peer_count = state.shared_state.peer_registry.len();

    let status = json!({
        "status": "ok",
        "version": env!("CARGO_PKG_VERSION"),
        "model_loaded": model_loaded,
        "model_name": model_name,
        "peers": peer_count,
    });

    JsonRpcResponse::success(
        id,
        json!({
            "contents": [
                {
                    "uri": "swarmllm://status",
                    "mimeType": "application/json",
                    "text": serde_json::to_string_pretty(&status).unwrap_or_default()
                }
            ]
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
        let resp = handle_initialize(Some(Value::Number(1.into())));
        let result = resp.result.unwrap();
        assert_eq!(result["protocolVersion"], MCP_PROTOCOL_VERSION);
        assert_eq!(result["serverInfo"]["name"], SERVER_NAME);
        assert!(result["capabilities"]["tools"].is_object());
        assert!(result["capabilities"]["resources"].is_object());
    }

    #[test]
    fn tools_list_response_structure() {
        let resp = handle_tools_list(Some(Value::Number(1.into())));
        let result = resp.result.unwrap();
        let tools = result["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 3);

        let chat_tool = &tools[0];
        assert_eq!(chat_tool["name"], "chat");
        assert!(chat_tool["inputSchema"]["properties"]["model"].is_object());
        assert!(chat_tool["inputSchema"]["properties"]["messages"].is_object());

        let models_tool = &tools[1];
        assert_eq!(models_tool["name"], "models");

        let compare_tool = &tools[2];
        assert_eq!(compare_tool["name"], "compare");
        assert!(compare_tool["inputSchema"]["properties"]["prompt"].is_object());
        assert!(compare_tool["inputSchema"]["properties"]["models"].is_object());
    }

    #[test]
    fn resources_list_response_structure() {
        let resp = handle_resources_list(Some(Value::Number(1.into())));
        let result = resp.result.unwrap();
        let resources = result["resources"].as_array().unwrap();
        assert_eq!(resources.len(), 1);
        assert_eq!(resources[0]["uri"], "swarmllm://status");
    }
}
