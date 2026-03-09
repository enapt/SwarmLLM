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

const MCP_PROTOCOL_VERSION: &str = "2025-11-05";
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
                    let resp_body: serde_json::Value = resp
                        .json()
                        .await
                        .unwrap_or(json!({"error": "parse failed"}));
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
            Err(e) => {
                results.push(json!({"error": format!("Task failed: {e}"), "status": "error"}))
            }
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

/// Research tool: fan-out a question to multiple models, collect all responses.
/// If no models specified, auto-selects available models (local first, then cloud).
async fn tool_research(state: &AppState, id: Option<Value>, args: Value) -> JsonRpcResponse {
    let question = match args.get("question").and_then(|v| v.as_str()) {
        Some(q) => q.to_string(),
        None => {
            return JsonRpcResponse::error(id, INVALID_PARAMS, "Missing required field: question");
        }
    };

    let max_models = args.get("max_models").and_then(|v| v.as_u64()).unwrap_or(5) as usize;
    let max_tokens = args
        .get("max_tokens")
        .and_then(|v| v.as_u64())
        .unwrap_or(2048) as u32;
    let system = args
        .get("system")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());

    // Determine which models to query
    let models: Vec<String> = if let Some(arr) = args.get("models").and_then(|v| v.as_array()) {
        arr.iter()
            .filter_map(|v| v.as_str().map(|s| s.to_string()))
            .collect()
    } else {
        // Auto-select: local models first, then cloud providers
        let mut auto_models = Vec::new();
        if let Some(info) = state.shared_state.loaded_model_info.read().await.as_ref() {
            let slug = info
                .name
                .to_lowercase()
                .replace(' ', "-")
                .replace(|c: char| !c.is_alphanumeric() && c != '-' && c != '.', "");
            auto_models.push(slug);
        }
        for entry in state.shared_state.provider_model_map.iter() {
            if auto_models.len() >= max_models {
                break;
            }
            auto_models.push(entry.key().clone());
        }
        auto_models
    };

    if models.is_empty() {
        return JsonRpcResponse::error(id, INTERNAL_ERROR, "No models available for research");
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
        let question_clone = question.clone();

        let handle = tokio::spawn(async move {
            let start = std::time::Instant::now();

            let mut body = json!({
                "model": model_id,
                "max_tokens": max_tokens,
                "temperature": 0.7,
                "messages": [{"role": "user", "content": question_clone}],
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
                    let resp_body: serde_json::Value = resp
                        .json()
                        .await
                        .unwrap_or(json!({"error": "parse failed"}));
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
                        "response": content,
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

    let mut results = Vec::new();
    for handle in handles {
        match handle.await {
            Ok(result) => results.push(result),
            Err(e) => {
                results.push(json!({"error": format!("Task failed: {e}"), "status": "error"}))
            }
        }
    }

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
            Some(p) => p.to_string(),
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
        let max_tokens = task
            .get("max_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(1024) as u32;
        let temperature = task
            .get("temperature")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.7) as f32;

        let url = format!("{base}/v1/messages");
        let client = client.clone();
        let api_key = api_key.clone();

        let handle = tokio::spawn(async move {
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
                .post(&url)
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
                        "task_id": task_id,
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
                        "task_id": task_id,
                        "model": model_id,
                        "error": format!("HTTP {status}: {body}"),
                        "latency_ms": elapsed_ms,
                        "status": "error",
                    })
                }
                Err(e) => {
                    json!({
                        "task_id": task_id,
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

    let mut results = Vec::new();
    for handle in handles {
        match handle.await {
            Ok(result) => results.push(result),
            Err(e) => {
                results.push(json!({"error": format!("Task failed: {e}"), "status": "error"}))
            }
        }
    }

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
        .provider_model_map
        .iter()
        .map(|e| e.key().clone())
        .collect();
    cloud_models.sort();

    // Node stats
    let stats = state.shared_state.node_stats.read().await;
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
    let credit_balance = state.shared_state.credit_balance.read().await;
    let credits = json!({
        "balance": credit_balance.balance,
        "lifetime_earned": credit_balance.lifetime_earned,
        "lifetime_spent": credit_balance.lifetime_spent,
    });
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
        assert_eq!(tools.len(), 6);

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

        let node_info_tool = &tools[5];
        assert_eq!(node_info_tool["name"], "node_info");
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
