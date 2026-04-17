//! Shared dispatch helpers used by the compare/research/batch_prompts tools.
//!
//! Each of these tools fans out the same HTTP call to N models and collects
//! results back. Helpers live here so tools.rs (or mod.rs) stays focused on
//! per-tool shape + parameter validation.

use serde_json::json;

/// Per-task timeout for MCP multi-model calls (matches tool_chat's 120s).
pub(super) const MCP_TASK_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(120);

/// Collect results from spawned JoinHandles, converting join errors and
/// timeouts into structured error JSON objects (one per handle, preserving
/// ordering so callers can correlate with their input model list).
pub(super) async fn collect_handle_results(
    handles: Vec<tokio::task::JoinHandle<serde_json::Value>>,
) -> Vec<serde_json::Value> {
    let mut results = Vec::with_capacity(handles.len());
    for handle in handles {
        match tokio::time::timeout(MCP_TASK_TIMEOUT, handle).await {
            Ok(Ok(result)) => results.push(result),
            Ok(Err(e)) => {
                results.push(json!({"error": format!("Task failed: {e}"), "status": "error"}))
            }
            Err(_) => results.push(json!({"error": format!("Request timed out ({}s)", MCP_TASK_TIMEOUT.as_secs()), "status": "error"})),
        }
    }
    results
}

/// Extract text content and token usage from an Anthropic Messages API response body.
pub(super) fn extract_anthropic_response(body: &serde_json::Value) -> (String, u64, u64) {
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
pub(super) struct ModelCallResult {
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
pub(super) async fn dispatch_model_call(
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
            let truncated = crate::api::scrub_truncate_error(&body);
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

/// Spawn a single model dispatch call as a detached task, with caller-supplied
/// JSON shaping applied to the result. Shared plumbing for tool_compare,
/// tool_research, and tool_batch_prompts, each of which uses slightly different
/// output keys (`content` vs `response`, with or without `task_id`).
#[allow(clippy::too_many_arguments)]
pub(super) fn spawn_model_call_task<F>(
    client: reqwest::Client,
    base_url: &str,
    api_key: String,
    model_id: String,
    prompt: String,
    system: Option<String>,
    temperature: f32,
    max_tokens: u32,
    shape: F,
) -> tokio::task::JoinHandle<serde_json::Value>
where
    F: FnOnce(&str, ModelCallResult) -> serde_json::Value + Send + 'static,
{
    let url = format!("{base_url}/v1/messages");
    tokio::spawn(async move {
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
        shape(&model_id, r)
    })
}
