use serde_json::{json, Value};

use super::dispatch::{
    collect_handle_results, dispatch_model_call, spawn_model_call_task, MCP_TASK_TIMEOUT,
};
use super::resources::mcp_peer_json;
use super::types::{JsonRpcResponse, INTERNAL_ERROR, INVALID_PARAMS, RESOURCE_UNAVAILABLE};
use super::{MCP_MAX_MODEL_ID_BYTES, MCP_MAX_PROMPT_BYTES, MCP_MAX_TASK_ID_BYTES};
use crate::api::server::AppState;
use crate::api::{DEFAULT_MAX_TOKENS, DEFAULT_TOP_K};

pub(super) async fn handle_tools_call(
    state: &AppState,
    id: Option<Value>,
    params: Value,
) -> JsonRpcResponse {
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
    if model.len() > MCP_MAX_MODEL_ID_BYTES {
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

    // SEC: cap per-message and total prompt size before submitting to the
    // router. The OpenAI and Anthropic handlers run this check; the MCP
    // handler had only a message-count cap, so a client could send 4096
    // messages of 2 MB each (8 GiB total) and the router would OOM trying
    // to assemble the prompt.
    if let Err(e) = crate::api::validate_content_size(chat_messages.iter().map(|m| m.content.len()))
    {
        return JsonRpcResponse::error(id, INVALID_PARAMS, format!("{}", e.0));
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
        MCP_TASK_TIMEOUT,
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
pub(super) async fn enumerate_models(state: &AppState) -> Vec<(String, String, &'static str)> {
    let mut results = vec![];
    let mut seen = std::collections::HashSet::new();

    // Every locally-known model comes from the registry, classified by what this
    // node actually holds.
    //
    // There used to be a separate "local loaded model" entry built by slugifying
    // `loaded_model_info.name`. That name is GGUF-internal
    // (`tinyllama_tinyllama-1.1b-chat-v1.0`), and slugifying strips the
    // underscore, so the advertised id was `tinyllamatinyllama-1.1b-chat-v1.0` —
    // **not a model id**, and rejected with "Model not available" by every tool
    // it was passed to. The real model was listed separately in the same
    // response, labelled `network`. An MCP client had no way to tell which of
    // the two to use.
    for manifest in state.shared_state.model_registry.models() {
        if !seen.contains(&manifest.id.0) {
            seen.insert(manifest.id.0.clone());
            let source = crate::api::model_source_for(&state.shared_state, &manifest.id.0);
            results.push((manifest.id.0.clone(), manifest.name.clone(), source));
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
        handles.push(spawn_model_call_task(
            client.clone(),
            &base,
            api_key.clone(),
            model_id.clone(),
            prompt.clone(),
            system.clone(),
            temperature,
            max_tokens,
            |mid, r| match r.error {
                None => json!({
                    "model": mid,
                    "content": r.content,
                    "input_tokens": r.input_tokens,
                    "output_tokens": r.output_tokens,
                    "latency_ms": r.elapsed_ms,
                    "status": "ok",
                }),
                Some(err) => json!({
                    "model": mid,
                    "error": err,
                    "latency_ms": r.elapsed_ms,
                    "status": "error",
                }),
            },
        ));
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
        // Registry ids, not a slug of the loaded model's GGUF name — that slug
        // is not a usable id, so auto-selection used to pick a model every
        // subsequent call would reject.
        for manifest in state.shared_state.model_registry.models() {
            if auto_models.len() >= max_models {
                break;
            }
            if crate::api::model_source_for(&state.shared_state, &manifest.id.0) == "local" {
                auto_models.push(manifest.id.0.clone());
            }
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
        handles.push(spawn_model_call_task(
            client.clone(),
            &base,
            api_key.clone(),
            model_id.clone(),
            question.clone(),
            system.clone(),
            0.7,
            max_tokens,
            |mid, r| match r.error {
                None => json!({
                    "model": mid,
                    "response": r.content,
                    "input_tokens": r.input_tokens,
                    "output_tokens": r.output_tokens,
                    "latency_ms": r.elapsed_ms,
                    "status": "ok",
                }),
                Some(err) => json!({
                    "model": mid,
                    "error": err,
                    "latency_ms": r.elapsed_ms,
                    "status": "error",
                }),
            },
        ));
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
        let raw_task_id = task.get("id").and_then(|v| v.as_str()).unwrap_or("unknown");
        if raw_task_id.len() > MCP_MAX_TASK_ID_BYTES {
            // Match every other oversize-input arm in this fn — return an
            // explicit error rather than silently truncating, so the caller
            // can correlate the error response back to a known task ID
            // (R94: silent truncation broke that correlation).
            let preview: String = raw_task_id
                .char_indices()
                .take_while(|(i, _)| *i < MCP_MAX_TASK_ID_BYTES)
                .map(|(_, c)| c)
                .collect();
            handles.push(tokio::spawn(async move {
                json!({
                    "task_id": preview,
                    "error": "task_id too long",
                    "status": "error",
                })
            }));
            continue;
        }
        let task_id = raw_task_id.to_string();
        let model_id = match task.get("model").and_then(|v| v.as_str()) {
            Some(m) if m.len() <= MCP_MAX_MODEL_ID_BYTES => m.to_string(),
            Some(_) => {
                handles.push(tokio::spawn(async move {
                    json!({
                        "task_id": task_id,
                        "error": "model name too long",
                        "status": "error",
                    })
                }));
                continue;
            }
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

        let task_id_for_shape = task_id.clone();
        handles.push(spawn_model_call_task(
            client.clone(),
            &base,
            api_key.clone(),
            model_id,
            prompt,
            system,
            temperature,
            max_tokens,
            move |mid, r| match r.error {
                None => json!({
                    "task_id": task_id_for_shape,
                    "model": mid,
                    "content": r.content,
                    "input_tokens": r.input_tokens,
                    "output_tokens": r.output_tokens,
                    "latency_ms": r.elapsed_ms,
                    "status": "ok",
                }),
                Some(err) => json!({
                    "task_id": task_id_for_shape,
                    "model": mid,
                    "error": err,
                    "latency_ms": r.elapsed_ms,
                    "status": "error",
                }),
            },
        ));
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
/// Sort key for the `fast` delegation tier: lower is preferred.
///
/// Time to an answer is dominated by whether the model is ALREADY IN MEMORY.
/// Loading a cold one costs tens of seconds — measured 57s when a cold 3.8B
/// model was picked while a 3B sat loaded and would have answered in under a
/// second. So that comes first, then local over network, then smallest.
///
/// A size of 0 means "unknown" (network models carry no size hint) and sorts
/// last rather than first, so an unmeasured model never beats a known small one.
fn fast_tier_rank(source: &str, size_bytes: u64, loaded: bool) -> (u8, u8, u64) {
    let source_rank = match source {
        "local" => 0,
        "network" => 1,
        _ => 2,
    };
    (
        u8::from(!loaded),
        source_rank,
        if size_bytes == 0 {
            u64::MAX
        } else {
            size_bytes
        },
    )
}

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

    // Fully-held models — always fastest. Registry ids for the same reason as
    // above: a slug of the loaded model's GGUF name is not a usable id.
    for manifest in state.shared_state.model_registry.models() {
        if crate::api::model_source_for(&state.shared_state, &manifest.id.0) == "local" {
            candidates.push((manifest.id.0.clone(), "local", manifest.total_size_bytes));
        }
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
        let smart_prefixes = ["claude", "gpt-", "o3", "o4", "gemini-", "kimi", "deepseek"];
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
            // Lowest time to an answer, which is dominated by whether the model
            // is ALREADY IN MEMORY. Loading a cold one costs tens of seconds —
            // measured 57s picking a cold 3.8B model while a 3B sat loaded and
            // would have answered in under a second. Size only breaks ties among
            // models in the same state.
            //
            // This used to take the first local candidate in registry order,
            // under a comment claiming "smallest = fastest" that the code never
            // implemented; `cheap` below is the one that actually sorts by size.
            let pool = &state.shared_state.model_process_pool;
            candidates
                .iter()
                .min_by_key(|(id, src, size)| {
                    fast_tier_rank(
                        src,
                        *size,
                        pool.is_loaded(&crate::types::ModelId(id.clone())),
                    )
                })
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

    let call = dispatch_model_call(
        client,
        &url,
        &api_key,
        &model_id,
        &prompt,
        system.as_deref(),
        0.7,
        max_tokens,
    )
    .await;

    if let Some(err) = call.error {
        return JsonRpcResponse::error(id, INTERNAL_ERROR, format!("Delegate failed: {err}"));
    }

    let result = json!({
        "model": model_id,
        "source": source,
        "tier": tier,
        "content": call.content,
        "input_tokens": call.input_tokens,
        "output_tokens": call.output_tokens,
        "latency_ms": call.elapsed_ms,
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
        peers_summary.push(mcp_peer_json(&entry));
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
        "requests_served": state.shared_state.metrics.requests_served_atomic.load(std::sync::atomic::Ordering::Relaxed),
        "requests_made": stats.requests_made,
        "forwards_served": state.shared_state.metrics.forwards_served_atomic.load(std::sync::atomic::Ordering::Relaxed),
        "bytes_uploaded": state.shared_state.shard_bytes_served.load(std::sync::atomic::Ordering::Relaxed),
        "relay_bytes_forwarded": state.shared_state.relay_inference_bytes.load(std::sync::atomic::Ordering::Relaxed),
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

#[cfg(test)]
mod fast_tier_tests {
    use super::fast_tier_rank;

    /// The `fast` tier used to take the first local model in registry order,
    /// under a comment claiming "smallest = fastest" that the code never
    /// implemented. Measured live: it chose a cold 3.8B model and took 57
    /// seconds while an already-loaded 3B would have answered in under one.
    #[test]
    fn an_already_loaded_model_wins_however_big_it_is() {
        let loaded_big = fast_tier_rank("local", 8_000_000_000, true);
        let cold_tiny = fast_tier_rank("local", 500_000_000, false);
        assert!(
            loaded_big < cold_tiny,
            "loading a model costs far more than its size saves"
        );
    }

    #[test]
    fn among_equals_local_beats_network_and_smaller_beats_bigger() {
        assert!(
            fast_tier_rank("local", 2_000_000_000, false)
                < fast_tier_rank("network", 2_000_000_000, false),
            "a model on this machine avoids a network hop"
        );
        assert!(
            fast_tier_rank("local", 1_000_000_000, false)
                < fast_tier_rank("local", 4_000_000_000, false),
            "with both cold, the smaller one loads sooner"
        );
        assert!(
            fast_tier_rank("local", 500_000_000, true)
                < fast_tier_rank("local", 1_000_000_000, true),
            "with both loaded, the smaller one decodes faster"
        );
    }

    /// Network candidates carry no size hint. Treating 0 as "smallest" would
    /// make every unmeasured model beat every measured one.
    #[test]
    fn an_unknown_size_does_not_masquerade_as_the_smallest() {
        assert!(
            fast_tier_rank("network", 4_000_000_000, false) < fast_tier_rank("network", 0, false),
            "unknown size must sort last, not first"
        );
    }
}
