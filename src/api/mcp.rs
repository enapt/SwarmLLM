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
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use super::{DEFAULT_MAX_TOKENS, DEFAULT_TOP_K};
use crate::api::server::AppState;

/// Collect results from spawned JoinHandles, converting join errors to error JSON.
/// Per-task timeout for MCP multi-model calls (matches tool_chat's 120s).
const MCP_TASK_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(120);

async fn collect_handle_results(
    handles: Vec<tokio::task::JoinHandle<serde_json::Value>>,
) -> Vec<serde_json::Value> {
    let mut results = Vec::with_capacity(handles.len());
    for handle in handles {
        match tokio::time::timeout(MCP_TASK_TIMEOUT, handle).await {
            Ok(Ok(result)) => results.push(result),
            Ok(Err(e)) => {
                results.push(json!({"error": format!("Task failed: {e}"), "status": "error"}))
            }
            Err(_) => results.push(json!({"error": "Request timed out (120s)", "status": "error"})),
        }
    }
    results
}

/// Extract text content and token usage from an Anthropic Messages API response body.
fn extract_anthropic_response(body: &serde_json::Value) -> (String, u64, u64) {
    let content = body["content"]
        .as_array()
        .and_then(|arr| {
            arr.iter()
                .filter_map(|b| b.get("text").and_then(|t| t.as_str()))
                .next()
        })
        .unwrap_or("")
        .to_string();
    let input_tokens = body["usage"]["input_tokens"].as_u64().unwrap_or(0);
    let output_tokens = body["usage"]["output_tokens"].as_u64().unwrap_or(0);
    (content, input_tokens, output_tokens)
}

/// Result of a single model dispatch call used by MCP compare/research/batch tools.
struct ModelCallResult {
    pub content: String,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub elapsed_ms: u64,
    /// None on success, Some(message) on error.
    pub error: Option<String>,
}

/// Send a prompt to a model endpoint and return the parsed result.
///
/// Shared core for tool_compare, tool_research, and tool_batch_prompts.
#[allow(clippy::too_many_arguments)]
async fn dispatch_model_call(
    client: &reqwest::Client,
    url: &str,
    api_key: &str,
    model_id: &str,
    prompt: &str,
    system: Option<&str>,
    temperature: f32,
    max_tokens: u32,
) -> ModelCallResult {
    let start = std::time::Instant::now();

    let mut body = json!({
        "model": model_id,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [{"role": "user", "content": prompt}],
        "stream": false,
    });
    if let Some(sys) = system {
        body["system"] = json!(sys);
    }

    let result = client
        .post(url)
        .header("Authorization", format!("Bearer {api_key}"))
        .header("Content-Type", "application/json")
        .json(&body)
        .send()
        .await;

    let elapsed_ms = start.elapsed().as_millis() as u64;

    match result {
        Ok(resp) if resp.status().is_success() => {
            let resp_body: serde_json::Value = resp
                .json()
                .await
                .unwrap_or(json!({"error": "parse failed"}));
            let (content, input_tokens, output_tokens) = extract_anthropic_response(&resp_body);
            ModelCallResult {
                content,
                input_tokens,
                output_tokens,
                elapsed_ms,
                error: None,
            }
        }
        Ok(resp) => {
            let status = resp.status().as_u16();
            let body = resp.text().await.unwrap_or_default();
            let truncated = super::scrub_truncate_error(&body);
            ModelCallResult {
                content: String::new(),
                input_tokens: 0,
                output_tokens: 0,
                elapsed_ms,
                error: Some(format!("HTTP {status}: {truncated}")),
            }
        }
        Err(e) => ModelCallResult {
            content: String::new(),
            input_tokens: 0,
            output_tokens: 0,
            elapsed_ms,
            error: Some(format!("{e}")),
        },
    }
}

const MCP_PROTOCOL_VERSION: &str = "2025-11-25";
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
/// Application-level error: resource unavailable (no models loaded).
const RESOURCE_UNAVAILABLE: i64 = -32000;
/// Maximum prompt/question length for MCP tool inputs (4 MB, matches HTTP validation).
const MCP_MAX_PROMPT_BYTES: usize = 4 * 1024 * 1024;

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
        "initialize" => handle_initialize(req.id),
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
        _ if is_notification => {
            // Unknown notification — silently accept per spec
            return (StatusCode::ACCEPTED, Json(None));
        }
        _ => JsonRpcResponse::error(
            req.id,
            METHOD_NOT_FOUND,
            format!("Method not found: {}", req.method),
        ),
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
                "tools": { "listChanged": false },
                "resources": { "listChanged": false },
            },
            "instructions": SERVER_INSTRUCTIONS,
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
                    "annotations": {
                        "title": "Chat Completion",
                        "readOnlyHint": true,
                        "destructiveHint": false,
                        "idempotentHint": false,
                        "openWorldHint": true
                    },
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
                    "annotations": {
                        "title": "List Models",
                        "readOnlyHint": true,
                        "destructiveHint": false,
                        "idempotentHint": true,
                        "openWorldHint": false
                    },
                    "inputSchema": {
                        "type": "object",
                        "properties": {}
                    }
                },
                {
                    "name": "compare",
                    "description": "Send the same prompt to multiple models concurrently and return all responses side-by-side for comparison. Supports local, network, and cloud models.",
                    "annotations": {
                        "title": "Compare Models",
                        "readOnlyHint": true,
                        "destructiveHint": false,
                        "idempotentHint": false,
                        "openWorldHint": true
                    },
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
                },
                {
                    "name": "research",
                    "description": "Fan out a research question to multiple models in parallel and collect all responses. Designed for knowledge gathering — send a question to cheap/fast models to get diverse perspectives without using expensive model tokens. Each model's response is returned separately with latency and token usage.",
                    "annotations": {
                        "title": "Research (Multi-Model)",
                        "readOnlyHint": true,
                        "destructiveHint": false,
                        "idempotentHint": false,
                        "openWorldHint": true
                    },
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "question": {
                                "type": "string",
                                "description": "The research question to send to all models"
                            },
                            "system": {
                                "type": "string",
                                "description": "Optional system prompt to guide research focus"
                            },
                            "models": {
                                "type": "array",
                                "description": "Array of model IDs to query. If omitted, uses all available models (local + cloud).",
                                "items": { "type": "string" }
                            },
                            "max_models": {
                                "type": "integer",
                                "description": "Maximum number of models to query when models is omitted (default 5)"
                            },
                            "max_tokens": {
                                "type": "integer",
                                "description": "Maximum tokens per response (default 2048)"
                            }
                        },
                        "required": ["question"]
                    }
                },
                {
                    "name": "batch_prompts",
                    "description": "Execute multiple independent prompts in parallel, each targeting a specific model. Returns all results once complete. Ideal for offloading parallel subtasks — e.g., ask one model to summarize, another to translate, another to review code, all at once.",
                    "annotations": {
                        "title": "Batch Prompts (Parallel)",
                        "readOnlyHint": true,
                        "destructiveHint": false,
                        "idempotentHint": false,
                        "openWorldHint": true
                    },
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "tasks": {
                                "type": "array",
                                "description": "Array of independent prompt tasks to execute in parallel",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "id": {
                                            "type": "string",
                                            "description": "Caller-defined ID for this task (returned in results for matching)"
                                        },
                                        "model": {
                                            "type": "string",
                                            "description": "Model ID to use for this task"
                                        },
                                        "prompt": {
                                            "type": "string",
                                            "description": "The prompt to send"
                                        },
                                        "system": {
                                            "type": "string",
                                            "description": "Optional system prompt"
                                        },
                                        "max_tokens": {
                                            "type": "integer",
                                            "description": "Max tokens for this task (default 1024)"
                                        },
                                        "temperature": {
                                            "type": "number",
                                            "description": "Temperature for this task (default 0.7)"
                                        }
                                    },
                                    "required": ["id", "model", "prompt"]
                                }
                            }
                        },
                        "required": ["tasks"]
                    }
                },
                {
                    "name": "delegate",
                    "description": "Offload a task to the most appropriate model based on a tier preference. Tiers: 'fast' picks the lowest-latency local model, 'cheap' picks a small/free model, 'smart' picks the most capable available model (may use cloud). Saves your subscription tokens by routing routine work to local/cheap models automatically.",
                    "annotations": {
                        "title": "Delegate Task",
                        "readOnlyHint": true,
                        "destructiveHint": false,
                        "idempotentHint": false,
                        "openWorldHint": true
                    },
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "prompt": {
                                "type": "string",
                                "description": "The task/prompt to delegate"
                            },
                            "tier": {
                                "type": "string",
                                "enum": ["fast", "cheap", "smart"],
                                "description": "Model selection strategy: 'fast' = lowest latency, 'cheap' = smallest/free model, 'smart' = most capable"
                            },
                            "system": {
                                "type": "string",
                                "description": "Optional system prompt"
                            },
                            "max_tokens": {
                                "type": "integer",
                                "description": "Maximum tokens to generate (default 1024)"
                            }
                        },
                        "required": ["prompt", "tier"]
                    }
                },
                {
                    "name": "node_info",
                    "description": "Get detailed information about this SwarmLLM node: loaded models, connected peers, VRAM/disk usage, credit balance, available cloud providers, and network status.",
                    "annotations": {
                        "title": "Node Information",
                        "readOnlyHint": true,
                        "destructiveHint": false,
                        "idempotentHint": true,
                        "openWorldHint": false
                    },
                    "inputSchema": {
                        "type": "object",
                        "properties": {}
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
        "research" => tool_research(state, id, arguments).await,
        "batch_prompts" => tool_batch_prompts(state, id, arguments).await,
        "delegate" => tool_delegate(state, id, arguments).await,
        "node_info" => tool_node_info(state, id).await,
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
                },
                {
                    "uri": "swarmllm://models",
                    "name": "Available Models",
                    "description": "All models available for inference: local, network, and cloud providers with capabilities and status",
                    "mimeType": "application/json"
                },
                {
                    "uri": "swarmllm://peers",
                    "name": "Connected Peers",
                    "description": "Currently connected P2P peers with latency, trust, load, and shard info",
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
        "swarmllm://models" => resource_models(state, id).await,
        "swarmllm://peers" => resource_peers(state, id).await,
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
    if model.len() > 256 {
        return JsonRpcResponse::error(id, INVALID_PARAMS, "Model name too long");
    }

    let messages = match args.get("messages").and_then(|v| v.as_array()) {
        Some(msgs) => msgs.clone(),
        None => {
            return JsonRpcResponse::error(id, INVALID_PARAMS, "Missing required field: messages");
        }
    };
    if messages.len() > 4096 {
        return JsonRpcResponse::error(id, INVALID_PARAMS, "Too many messages (max 4096)");
    }

    let temperature = args
        .get("temperature")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.7)
        .clamp(0.0, 2.0) as f32;
    let max_tokens = args
        .get("max_tokens")
        .and_then(|v| v.as_u64())
        .unwrap_or(512)
        .min(DEFAULT_MAX_TOKENS as u64) as u32;

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

    let request = crate::types::InferenceRequest::local(
        crate::types::ModelId(model),
        chat_messages,
        crate::types::SamplingParams {
            temperature,
            top_p: 0.9,
            top_k: DEFAULT_TOP_K,
            max_tokens,
            stop: vec![],
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            logprobs: false,
            top_logprobs: 0,
        },
        false,
        None,
        None,
    );

    match tokio::time::timeout(
        std::time::Duration::from_secs(120),
        crate::api::submit_to_router(&router_tx, request),
    )
    .await
    {
        Ok(Ok(response)) => JsonRpcResponse::success(
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
        Ok(Err(e)) => {
            JsonRpcResponse::error(id, INTERNAL_ERROR, format!("Inference error: {}", e.0))
        }
        Err(_) => JsonRpcResponse::error(id, INTERNAL_ERROR, "Inference request timed out"),
    }
}

/// Enumerate all available models across sources (local, network, cloud), deduped by ID.
async fn enumerate_models(state: &AppState) -> Vec<(String, String, &'static str)> {
    let mut results = vec![];
    let mut seen = std::collections::HashSet::new();

    // Local loaded model
    if let Some(info) = state.shared_state.loaded_model_info.read().await.as_ref() {
        let slug = crate::types::slugify_model_name(&info.name);
        seen.insert(slug.clone());
        seen.insert(info.name.clone());
        results.push((slug, info.name.clone(), "local"));
    }

    // Registry models
    for manifest in state.shared_state.model_registry.models() {
        if !seen.contains(&manifest.id.0) {
            seen.insert(manifest.id.0.clone());
            results.push((manifest.id.0.clone(), manifest.name.clone(), "network"));
        }
    }

    // Cloud provider models
    let mut cloud: Vec<String> = state
        .shared_state
        .metrics
        .provider_model_map
        .iter()
        .map(|e| e.key().clone())
        .collect();
    cloud.sort();
    for model_id in cloud {
        if !seen.contains(&model_id) {
            seen.insert(model_id.clone());
            results.push((model_id.clone(), model_id, "cloud"));
        }
    }

    results
}

async fn tool_models(state: &AppState, id: Option<Value>) -> JsonRpcResponse {
    let models: Vec<Value> = enumerate_models(state)
        .await
        .into_iter()
        .map(|(id, name, source)| json!({ "id": id, "name": name, "source": source }))
        .collect();

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
        Some(p) if p.len() <= MCP_MAX_PROMPT_BYTES => p.to_string(),
        Some(_) => {
            return JsonRpcResponse::error(id, INVALID_PARAMS, "prompt exceeds maximum length");
        }
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
    if system
        .as_ref()
        .is_some_and(|s| s.len() > MCP_MAX_PROMPT_BYTES)
    {
        return JsonRpcResponse::error(id, INVALID_PARAMS, "system prompt exceeds maximum length");
    }
    let temperature = args
        .get("temperature")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.7)
        .clamp(0.0, 2.0) as f32;
    let max_tokens = args
        .get("max_tokens")
        .and_then(|v| v.as_u64())
        .unwrap_or(1024)
        .min(DEFAULT_MAX_TOKENS as u64) as u32;

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
        let prompt = prompt.clone();

        let handle = tokio::spawn(async move {
            let r = dispatch_model_call(
                &client,
                &url,
                &api_key,
                &model_id,
                &prompt,
                system_val.as_deref(),
                temperature,
                max_tokens,
            )
            .await;
            match r.error {
                None => json!({
                    "model": model_id,
                    "content": r.content,
                    "input_tokens": r.input_tokens,
                    "output_tokens": r.output_tokens,
                    "latency_ms": r.elapsed_ms,
                    "status": "ok",
                }),
                Some(err) => json!({
                    "model": model_id,
                    "error": err,
                    "latency_ms": r.elapsed_ms,
                    "status": "error",
                }),
            }
        });
        handles.push(handle);
    }

    let results = collect_handle_results(handles).await;

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

/// Research tool: fan-out a question to multiple models, collect all responses.
/// If no models specified, auto-selects available models (local first, then cloud).
async fn tool_research(state: &AppState, id: Option<Value>, args: Value) -> JsonRpcResponse {
    let question = match args.get("question").and_then(|v| v.as_str()) {
        Some(q) if q.len() <= MCP_MAX_PROMPT_BYTES => q.to_string(),
        Some(_) => {
            return JsonRpcResponse::error(id, INVALID_PARAMS, "question exceeds maximum length");
        }
        None => {
            return JsonRpcResponse::error(id, INVALID_PARAMS, "Missing required field: question");
        }
    };

    let max_models = args
        .get("max_models")
        .and_then(|v| v.as_u64())
        .unwrap_or(5)
        .min(20) as usize;
    let max_tokens = args
        .get("max_tokens")
        .and_then(|v| v.as_u64())
        .unwrap_or(2048)
        .min(DEFAULT_MAX_TOKENS as u64) as u32;
    let system = args
        .get("system")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());
    if system
        .as_ref()
        .is_some_and(|s| s.len() > MCP_MAX_PROMPT_BYTES)
    {
        return JsonRpcResponse::error(id, INVALID_PARAMS, "system prompt exceeds maximum length");
    }

    // Determine which models to query
    let models: Vec<String> = if let Some(arr) = args.get("models").and_then(|v| v.as_array()) {
        arr.iter()
            .filter_map(|v| v.as_str().map(|s| s.to_string()))
            .collect()
    } else {
        // Auto-select: local models only when no explicit models specified.
        // SEC: Do NOT auto-discover cloud providers — prevents unintended cost drain
        // from MCP clients triggering paid API calls without explicit model selection.
        let mut auto_models = Vec::new();
        if let Some(info) = state.shared_state.loaded_model_info.read().await.as_ref() {
            let slug = crate::types::slugify_model_name(&info.name);
            auto_models.push(slug);
        }
        // Only add network-available models (not cloud providers)
        for entry in state.shared_state.split_models.iter() {
            if auto_models.len() >= max_models {
                break;
            }
            let model_id = &entry.key().0;
            if !auto_models.iter().any(|m| m == &model_id.0) {
                auto_models.push(model_id.0.clone());
            }
        }
        auto_models
    };

    if models.is_empty() {
        return JsonRpcResponse::error(
            id,
            RESOURCE_UNAVAILABLE,
            "No models available for research",
        );
    }
    if models.len() > 20 {
        return JsonRpcResponse::error(id, INVALID_PARAMS, "Maximum 20 models per research query");
    }

    // Reuse the compare infrastructure but with research-focused framing
    let client = crate::api::providers::get_provider_client();
    let port = state.config.node.listen_port;
    let base = format!("http://127.0.0.1:{port}");
    let api_key = state.shared_state.api_key.clone();

    let mut handles = Vec::new();
    for model_id in &models {
        let url = format!("{base}/v1/messages");
        let client = client.clone();
        let model_id = model_id.clone();
        let api_key = api_key.clone();
        let system_val = system.clone();
        let question = question.clone();

        let handle = tokio::spawn(async move {
            let r = dispatch_model_call(
                &client,
                &url,
                &api_key,
                &model_id,
                &question,
                system_val.as_deref(),
                0.7,
                max_tokens,
            )
            .await;
            match r.error {
                None => json!({
                    "model": model_id,
                    "response": r.content,
                    "input_tokens": r.input_tokens,
                    "output_tokens": r.output_tokens,
                    "latency_ms": r.elapsed_ms,
                    "status": "ok",
                }),
                Some(err) => json!({
                    "model": model_id,
                    "error": err,
                    "latency_ms": r.elapsed_ms,
                    "status": "error",
                }),
            }
        });
        handles.push(handle);
    }

    let results = collect_handle_results(handles).await;

    let total_tokens: u64 = results
        .iter()
        .map(|r| r["input_tokens"].as_u64().unwrap_or(0) + r["output_tokens"].as_u64().unwrap_or(0))
        .sum();
    let successful = results.iter().filter(|r| r["status"] == "ok").count();

    let summary = json!({
        "question": question,
        "models_queried": models.len(),
        "successful_responses": successful,
        "total_tokens_used": total_tokens,
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

/// Batch prompts tool: execute multiple independent prompts in parallel.
async fn tool_batch_prompts(state: &AppState, id: Option<Value>, args: Value) -> JsonRpcResponse {
    let tasks = match args.get("tasks").and_then(|v| v.as_array()) {
        Some(arr) => arr.clone(),
        None => {
            return JsonRpcResponse::error(id, INVALID_PARAMS, "Missing required field: tasks");
        }
    };

    if tasks.is_empty() {
        return JsonRpcResponse::error(id, INVALID_PARAMS, "tasks array must not be empty");
    }
    if tasks.len() > 20 {
        return JsonRpcResponse::error(id, INVALID_PARAMS, "Maximum 20 tasks per batch");
    }

    let client = crate::api::providers::get_provider_client();
    let port = state.config.node.listen_port;
    let base = format!("http://127.0.0.1:{port}");
    let api_key = state.shared_state.api_key.clone();

    let mut handles = Vec::new();
    for task in &tasks {
        let task_id = task
            .get("id")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown")
            .to_string();
        let model_id = match task.get("model").and_then(|v| v.as_str()) {
            Some(m) => m.to_string(),
            None => {
                handles.push(tokio::spawn(async move {
                    json!({
                        "task_id": task_id,
                        "error": "Missing required field: model",
                        "status": "error",
                    })
                }));
                continue;
            }
        };
        let prompt = match task.get("prompt").and_then(|v| v.as_str()) {
            Some(p) if p.len() <= MCP_MAX_PROMPT_BYTES => p.to_string(),
            Some(_) => {
                handles.push(tokio::spawn(async move {
                    json!({
                        "task_id": task_id,
                        "error": "prompt exceeds maximum length",
                        "status": "error",
                    })
                }));
                continue;
            }
            None => {
                handles.push(tokio::spawn(async move {
                    json!({
                        "task_id": task_id,
                        "error": "Missing required field: prompt",
                        "status": "error",
                    })
                }));
                continue;
            }
        };
        let system = task
            .get("system")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        if system
            .as_ref()
            .is_some_and(|s| s.len() > MCP_MAX_PROMPT_BYTES)
        {
            handles.push(tokio::spawn(async move {
                json!({
                    "task_id": task_id,
                    "error": "system prompt exceeds maximum length",
                    "status": "error",
                })
            }));
            continue;
        }
        let max_tokens = task
            .get("max_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(1024)
            .min(DEFAULT_MAX_TOKENS as u64) as u32;
        let temperature = task
            .get("temperature")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.7)
            .clamp(0.0, 2.0) as f32;

        let url = format!("{base}/v1/messages");
        let client = client.clone();
        let api_key = api_key.clone();

        let handle = tokio::spawn(async move {
            let r = dispatch_model_call(
                &client,
                &url,
                &api_key,
                &model_id,
                &prompt,
                system.as_deref(),
                temperature,
                max_tokens,
            )
            .await;
            match r.error {
                None => json!({
                    "task_id": task_id,
                    "model": model_id,
                    "content": r.content,
                    "input_tokens": r.input_tokens,
                    "output_tokens": r.output_tokens,
                    "latency_ms": r.elapsed_ms,
                    "status": "ok",
                }),
                Some(err) => json!({
                    "task_id": task_id,
                    "model": model_id,
                    "error": err,
                    "latency_ms": r.elapsed_ms,
                    "status": "error",
                }),
            }
        });
        handles.push(handle);
    }

    let results = collect_handle_results(handles).await;

    let successful = results.iter().filter(|r| r["status"] == "ok").count();

    let summary = json!({
        "tasks_submitted": tasks.len(),
        "tasks_completed": successful,
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

/// Delegate tool: auto-select a model by tier and run inference.
/// Tiers: fast (lowest latency local), cheap (smallest/free), smart (most capable).
async fn tool_delegate(state: &AppState, id: Option<Value>, args: Value) -> JsonRpcResponse {
    let prompt = match args.get("prompt").and_then(|v| v.as_str()) {
        Some(p) if p.len() <= MCP_MAX_PROMPT_BYTES => p.to_string(),
        Some(_) => {
            return JsonRpcResponse::error(id, INVALID_PARAMS, "prompt exceeds maximum length");
        }
        None => {
            return JsonRpcResponse::error(id, INVALID_PARAMS, "Missing required field: prompt");
        }
    };
    let tier = match args.get("tier").and_then(|v| v.as_str()) {
        Some(t) => t.to_string(),
        None => {
            return JsonRpcResponse::error(id, INVALID_PARAMS, "Missing required field: tier");
        }
    };
    let system = args
        .get("system")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());
    let max_tokens = args
        .get("max_tokens")
        .and_then(|v| v.as_u64())
        .unwrap_or(1024)
        .min(DEFAULT_MAX_TOKENS as u64) as u32;

    // Collect available models with metadata for tier selection
    let mut candidates: Vec<(String, &str, u64)> = Vec::new(); // (model_id, source, size_hint)

    // Local loaded model — always fastest
    if let Some(info) = state.shared_state.loaded_model_info.read().await.as_ref() {
        let slug = crate::types::slugify_model_name(&info.name);
        candidates.push((slug, "local", info.size_bytes));
    }

    // Network models from split_models
    for entry in state.shared_state.split_models.iter() {
        let model_id = &entry.key().0;
        if !candidates.iter().any(|(id, _, _)| id == &model_id.0) {
            candidates.push((model_id.0.clone(), "network", 0));
        }
    }

    // Cloud models (only for "smart" tier — avoid surprise costs)
    if tier == "smart" {
        // Prefer known-capable cloud models
        let smart_prefixes = ["claude", "gpt-4", "o1", "o3", "gemini-2"];
        for entry in state.shared_state.metrics.provider_model_map.iter() {
            let model_id = entry.key().clone();
            let is_smart = smart_prefixes
                .iter()
                .any(|p| model_id.to_lowercase().contains(p));
            if is_smart && !candidates.iter().any(|(id, _, _)| id == &model_id) {
                candidates.push((model_id, "cloud", u64::MAX));
            }
        }
    }

    if candidates.is_empty() {
        return JsonRpcResponse::error(
            id,
            RESOURCE_UNAVAILABLE,
            "No models available for delegation",
        );
    }

    // Select based on tier
    let selected = match tier.as_str() {
        "fast" => {
            // Prefer local, then network (smallest = fastest)
            candidates
                .iter()
                .find(|(_, src, _)| *src == "local")
                .or_else(|| candidates.iter().find(|(_, src, _)| *src == "network"))
                .or(candidates.first())
                .cloned()
        }
        "cheap" => {
            // Prefer smallest local model, then network, avoid cloud
            let mut non_cloud: Vec<_> = candidates
                .iter()
                .filter(|(_, src, _)| *src != "cloud")
                .cloned()
                .collect();
            // Sort by size ascending (0 = unknown, sort last)
            non_cloud.sort_by_key(|(_, _, size)| if *size == 0 { u64::MAX } else { *size });
            non_cloud.first().cloned().or(candidates.first().cloned())
        }
        "smart" => {
            // Prefer cloud > largest local > network
            candidates
                .iter()
                .find(|(_, src, _)| *src == "cloud")
                .or_else(|| {
                    // Largest local model
                    candidates
                        .iter()
                        .filter(|(_, src, _)| *src == "local" || *src == "network")
                        .max_by_key(|(_, _, size)| *size)
                })
                .or(candidates.first())
                .cloned()
        }
        _ => {
            return JsonRpcResponse::error(
                id,
                INVALID_PARAMS,
                "Invalid tier: must be 'fast', 'cheap', or 'smart'",
            );
        }
    };

    let (model_id, source, _) = match selected {
        Some(s) => s,
        None => {
            return JsonRpcResponse::error(id, INTERNAL_ERROR, "No suitable model found for tier");
        }
    };

    // Route through our own /v1/messages endpoint
    let client = crate::api::providers::get_provider_client();
    let port = state.config.node.listen_port;
    let url = format!("http://127.0.0.1:{port}/v1/messages");
    let api_key = state.shared_state.api_key.clone();
    let start = std::time::Instant::now();

    let mut body = json!({
        "model": model_id,
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "messages": [{"role": "user", "content": prompt}],
        "stream": false,
    });
    if let Some(sys) = system {
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
            let resp_body: Value = resp
                .json()
                .await
                .unwrap_or(json!({"error": "parse failed"}));
            let (content, input_tokens, output_tokens) = extract_anthropic_response(&resp_body);

            let result = json!({
                "model": model_id,
                "source": source,
                "tier": tier,
                "content": content,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "latency_ms": elapsed_ms,
            });

            JsonRpcResponse::success(
                id,
                json!({
                    "content": [
                        {
                            "type": "text",
                            "text": serde_json::to_string_pretty(&result).unwrap_or_default()
                        }
                    ]
                }),
            )
        }
        Ok(resp) => {
            let status = resp.status().as_u16();
            let body = resp.text().await.unwrap_or_default();
            let scrubbed = crate::crypto::scrub_api_keys(&body);
            JsonRpcResponse::error(
                id,
                INTERNAL_ERROR,
                format!("Delegate failed (HTTP {status}): {scrubbed}"),
            )
        }
        Err(e) => JsonRpcResponse::error(id, INTERNAL_ERROR, format!("Delegate failed: {e}")),
    }
}

/// Node info tool: detailed node status, models, peers, resources.
async fn tool_node_info(state: &AppState, id: Option<Value>) -> JsonRpcResponse {
    // Loaded model
    let info = state.shared_state.loaded_model_info.read().await;
    let loaded_model = info.as_ref().map(|i| {
        json!({
            "name": i.name,
            "size_bytes": i.size_bytes,
        })
    });
    drop(info);

    // Peers
    let peer_count = state.shared_state.peer_registry.len();
    let mut peers_summary = Vec::new();
    for entry in state.shared_state.peer_registry.iter() {
        let peer = entry.value();
        peers_summary.push(json!({
            "node_id_short": entry.key().to_string(),
            "latency_ms": peer.latency_ms,
            "is_lan": peer.is_lan_peer,
            "trust_score": peer.trust_score,
            "active_requests": peer.active_request_count,
        }));
    }

    // Registry models
    let mut registry_models = Vec::new();
    for manifest in state.shared_state.model_registry.models() {
        let shard_count = manifest.shards.len();
        registry_models.push(json!({
            "id": manifest.id.0,
            "name": manifest.name,
            "shards": shard_count,
            "architecture": format!("{:?}", manifest.architecture),
        }));
    }

    // Cloud providers
    let mut cloud_models: Vec<String> = state
        .shared_state
        .metrics
        .provider_model_map
        .iter()
        .map(|e| e.key().clone())
        .collect();
    cloud_models.sort();

    // Node stats
    let stats = state.shared_state.metrics.node_stats.read().await;
    let node_stats = json!({
        "requests_served": stats.requests_served,
        "requests_made": stats.requests_made,
        "forwards_served": stats.forwards_served,
        "bytes_uploaded": stats.bytes_uploaded,
        "bytes_downloaded": stats.bytes_downloaded,
        "uptime_seconds": chrono::Utc::now().signed_duration_since(stats.uptime_start).num_seconds(),
    });
    drop(stats);

    // Credit balance
    let credit_balance = state.shared_state.credits.credit_balance.read().await;
    let credits = crate::api::credit_summary_json(&credit_balance);
    drop(credit_balance);

    let node_info = json!({
        "version": env!("CARGO_PKG_VERSION"),
        "loaded_model": loaded_model,
        "registry_models": registry_models,
        "cloud_models_available": cloud_models.len(),
        "cloud_models": cloud_models,
        "peers": {
            "count": peer_count,
            "details": peers_summary,
        },
        "stats": node_stats,
        "credits": credits,
    });

    JsonRpcResponse::success(
        id,
        json!({
            "content": [
                {
                    "type": "text",
                    "text": serde_json::to_string_pretty(&node_info).unwrap_or_default()
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

async fn resource_models(state: &AppState, id: Option<Value>) -> JsonRpcResponse {
    let base = enumerate_models(state).await;
    let mut models = Vec::with_capacity(base.len());

    for (model_id, name, source) in base {
        let mut entry = json!({ "id": model_id, "name": name, "source": source });
        match source {
            "local" => {
                if let Some(info) = state.shared_state.loaded_model_info.read().await.as_ref() {
                    if let Some(obj) = entry.as_object_mut() {
                        obj.insert("size_bytes".to_string(), json!(info.size_bytes));
                        obj.insert("status".to_string(), json!("loaded"));
                    }
                }
            }
            "network" => {
                if let Some(manifest) = state
                    .shared_state
                    .model_registry
                    .get_manifest(&crate::types::ModelId(model_id))
                {
                    if let Some(obj) = entry.as_object_mut() {
                        obj.insert("shards".to_string(), json!(manifest.shards.len()));
                        obj.insert(
                            "architecture".to_string(),
                            json!(format!("{:?}", manifest.architecture)),
                        );
                    }
                }
            }
            _ => {}
        }
        models.push(entry);
    }

    JsonRpcResponse::success(
        id,
        json!({
            "contents": [
                {
                    "uri": "swarmllm://models",
                    "mimeType": "application/json",
                    "text": serde_json::to_string_pretty(&models).unwrap_or_default()
                }
            ]
        }),
    )
}

async fn resource_peers(state: &AppState, id: Option<Value>) -> JsonRpcResponse {
    let mut peers = Vec::new();
    for entry in state.shared_state.peer_registry.iter() {
        let peer = entry.value();
        let region = peer
            .capability
            .as_ref()
            .and_then(|c| c.region.as_deref())
            .unwrap_or("unknown");
        peers.push(json!({
            "node_id_short": entry.key().to_string(),
            "latency_ms": peer.latency_ms,
            "is_lan": peer.is_lan_peer,
            "trust_score": peer.trust_score,
            "active_requests": peer.active_request_count,
            "region": region,
        }));
    }

    JsonRpcResponse::success(
        id,
        json!({
            "contents": [
                {
                    "uri": "swarmllm://peers",
                    "mimeType": "application/json",
                    "text": serde_json::to_string_pretty(&peers).unwrap_or_default()
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
