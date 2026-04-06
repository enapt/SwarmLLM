//! Claude Subscription Provider — subprocess proxy for Claude CLI.
//!
//! Routes Claude model requests through a locally-authenticated `claude` CLI
//! subprocess (Pro/Max/Team/Enterprise subscription). Isolated behind the
//! `claude-subscription` cargo feature flag for easy removal.
//!
//! Architecture: spawn `claude -p --output-format stream-json` per request,
//! parse NDJSON stdout lines, translate to Anthropic/OpenAI SSE or JSON.

use std::sync::LazyLock;

use axum::response::IntoResponse;
use serde::{Deserialize, Serialize};
use tokio::io::AsyncBufReadExt;
use tokio::process::Command;
use tokio::sync::Semaphore;

use crate::error::ApiError;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the Claude subscription subprocess provider.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ClaudeSubscriptionConfig {
    /// Whether to route Claude model requests through the local CLI.
    #[serde(default)]
    pub enabled: bool,
    /// Path to the `claude` binary (default: "claude" — found via PATH).
    #[serde(default)]
    pub claude_binary: Option<String>,
    /// Override model for all requests (default: use request's model field).
    #[serde(default)]
    pub default_model: Option<String>,
    /// Max concurrent subprocess invocations (default: 3).
    #[serde(default)]
    pub max_concurrent: Option<usize>,
    /// Timeout in seconds per request (default: 300 = 5 min).
    #[serde(default)]
    pub timeout_secs: Option<u64>,
    /// Working directory for the CLI subprocess (default: system temp dir).
    /// Set to a project path to give Claude project context (CLAUDE.md, etc.).
    /// Use "none" or leave empty for a clean context (recommended for API proxy use).
    #[serde(default)]
    pub working_dir: Option<String>,
}

impl ClaudeSubscriptionConfig {
    pub fn binary(&self) -> &str {
        self.claude_binary.as_deref().unwrap_or("claude")
    }

    pub fn timeout(&self) -> std::time::Duration {
        std::time::Duration::from_secs(self.timeout_secs.unwrap_or(300))
    }

    pub fn concurrency_limit(&self) -> usize {
        self.max_concurrent.unwrap_or(3)
    }
}

// ---------------------------------------------------------------------------
// Concurrency limiter (global semaphore)
// ---------------------------------------------------------------------------

/// Default concurrency limit. Overridden at runtime by config, but the
/// semaphore is sized generously — actual limiting is via `try_acquire`.
static SUBPROCESS_SEMAPHORE: LazyLock<Semaphore> = LazyLock::new(|| Semaphore::new(8));

fn acquire_permit(
    config: &ClaudeSubscriptionConfig,
) -> Result<tokio::sync::SemaphorePermit<'static>, ApiError> {
    let limit = config.concurrency_limit();
    // Enforce configured limit by checking available permits.
    let available = SUBPROCESS_SEMAPHORE.available_permits();
    if available <= 8usize.saturating_sub(limit) {
        return Err(ApiError(crate::error::SwarmError::ServiceUnavailable(
            "Claude subscription: too many concurrent requests, try again later".into(),
        )));
    }
    SUBPROCESS_SEMAPHORE.try_acquire().map_err(|_| {
        ApiError(crate::error::SwarmError::ServiceUnavailable(
            "Claude subscription: concurrency limit reached".into(),
        ))
    })
}

// ---------------------------------------------------------------------------
// Session management for multi-turn conversations
// ---------------------------------------------------------------------------

/// Maps an external conversation key to a Claude CLI session_id for multi-turn.
/// Key format: "{model}:{hash_of_messages_prefix}" or caller-provided session ID.
static SESSION_CACHE: LazyLock<dashmap::DashMap<String, SessionEntry>> =
    LazyLock::new(dashmap::DashMap::new);

struct SessionEntry {
    session_id: String,
    last_used: std::time::Instant,
}

/// Session TTL — expire after 10 minutes of inactivity (matches KV-cache TTL).
const SESSION_TTL: std::time::Duration = std::time::Duration::from_secs(600);

/// Maximum entries in SESSION_CACHE before forced cleanup.
const MAX_SESSION_CACHE_SIZE: usize = 200;

/// Maximum per-line size in subprocess output (1MB — matches claude_session.rs).
const MAX_RESPONSE_LINE: usize = 1024 * 1024;

/// Get a cached session ID for multi-turn, if one exists and is fresh.
fn get_session(key: &str) -> Option<String> {
    if let Some(entry) = SESSION_CACHE.get(key) {
        if entry.last_used.elapsed() < SESSION_TTL {
            return Some(entry.session_id.clone());
        }
        drop(entry);
        SESSION_CACHE.remove(key);
    }
    None
}

/// Store a session ID for multi-turn reuse.
fn put_session(key: String, session_id: String) {
    // Lazy cleanup on every insert to bound cache size
    cleanup_expired_sessions();
    if SESSION_CACHE.len() >= MAX_SESSION_CACHE_SIZE {
        // Emergency eviction: remove oldest entries
        let mut oldest: Vec<(String, std::time::Instant)> = SESSION_CACHE
            .iter()
            .map(|e| (e.key().clone(), e.value().last_used))
            .collect();
        oldest.sort_by_key(|(_, t)| *t);
        for (k, _) in oldest.iter().take(oldest.len() / 2) {
            SESSION_CACHE.remove(k);
        }
    }
    SESSION_CACHE.insert(
        key,
        SessionEntry {
            session_id,
            last_used: std::time::Instant::now(),
        },
    );
}

/// Build a session cache key from model + conversation history prefix.
/// Only produces a key for multi-turn conversations (2+ messages).
/// Single-turn requests return None — no session reuse.
/// Uses BLAKE3 (cryptographic) to prevent collision-based session confusion.
fn session_key(model: &str, messages: &[serde_json::Value]) -> Option<String> {
    if messages.len() < 2 {
        return None;
    }
    let mut hasher = blake3::Hasher::new();
    for msg in &messages[..messages.len() - 1] {
        hasher.update(msg.to_string().as_bytes());
    }
    hasher.update(model.as_bytes());
    Some(format!(
        "claude_sub:{model}:{}",
        &hasher.finalize().to_hex()[..16]
    ))
}

/// Periodic cleanup of expired sessions — called lazily from `put_session()`.
fn cleanup_expired_sessions() {
    SESSION_CACHE.retain(|_, entry| entry.last_used.elapsed() < SESSION_TTL);
}

/// Validate model name: must be ≤128 chars, safe characters only.
fn validate_model_name(model: &str) -> Result<(), ApiError> {
    if model.len() > 128
        || !model
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '.' || c == '_' || c == ':')
    {
        return Err(ApiError(crate::error::SwarmError::Validation(
            "Invalid model name for Claude subscription".into(),
        )));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// CLI detection
// ---------------------------------------------------------------------------

/// CLI status information returned by the detection endpoint.
#[derive(Serialize)]
pub struct ClaudeCliStatus {
    pub cli_installed: bool,
    pub cli_version: Option<String>,
    pub authenticated: bool,
    pub subscription_type: Option<String>,
    pub rate_limit_tier: Option<String>,
}

/// Detect whether the `claude` CLI is installed and authenticated.
pub async fn detect_cli(binary: &str) -> ClaudeCliStatus {
    // Check version
    let version_output = Command::new(binary).arg("--version").output().await;

    let (installed, version) = match version_output {
        Ok(output) if output.status.success() => {
            let ver = String::from_utf8_lossy(&output.stdout).trim().to_string();
            (true, Some(ver))
        }
        _ => {
            return ClaudeCliStatus {
                cli_installed: false,
                cli_version: None,
                authenticated: false,
                subscription_type: None,
                rate_limit_tier: None,
            }
        }
    };

    // Check credentials file for subscription info
    let (authenticated, sub_type, rate_tier) = read_credential_info();

    ClaudeCliStatus {
        cli_installed: installed,
        cli_version: version,
        authenticated,
        subscription_type: sub_type,
        rate_limit_tier: rate_tier,
    }
}

/// Read subscription info from ~/.claude/.credentials.json (read-only, display only).
fn read_credential_info() -> (bool, Option<String>, Option<String>) {
    let home = match dirs::home_dir() {
        Some(h) => h,
        None => return (false, None, None),
    };
    let cred_path = home.join(".claude").join(".credentials.json");
    let content = match std::fs::read_to_string(&cred_path) {
        Ok(c) => c,
        Err(_) => return (false, None, None),
    };
    let json: serde_json::Value = match serde_json::from_str(&content) {
        Ok(v) => v,
        Err(_) => return (false, None, None),
    };
    let oauth = &json["claudeAiOauth"];
    if oauth.is_null() {
        return (false, None, None);
    }
    let has_token = oauth["accessToken"].is_string();
    let sub_type = oauth["subscriptionType"].as_str().map(String::from);
    let rate_tier = oauth["rateLimitTier"].as_str().map(String::from);
    (has_token, sub_type, rate_tier)
}

// ---------------------------------------------------------------------------
// Subprocess spawning and NDJSON parsing
// ---------------------------------------------------------------------------

/// Build CLI args for `claude -p`.
fn build_cli_args(
    model: &str,
    prompt: &str,
    system_prompt: Option<&str>,
    session_id: Option<&str>,
    stream: bool,
) -> Vec<String> {
    let mut args = vec![
        "--print".to_string(),
        "--output-format".to_string(),
        if stream {
            "stream-json".to_string()
        } else {
            "json".to_string()
        },
        "--model".to_string(),
        model.to_string(),
    ];
    if stream {
        args.push("--verbose".to_string());
        args.push("--include-partial-messages".to_string());
    }
    if let Some(sid) = session_id {
        args.push("--resume".to_string());
        args.push(sid.to_string());
    } else {
        args.push("--no-session-persistence".to_string());
    }
    if let Some(sys) = system_prompt {
        if !sys.is_empty() {
            args.push("--system-prompt".to_string());
            args.push(sys.to_string());
        }
    }
    // Prompt is the final positional argument
    args.push(prompt.to_string());
    args
}

/// Extract text content from a message value (handles both string and array formats).
fn extract_content_text(msg: &serde_json::Value) -> String {
    match msg.get("content") {
        Some(serde_json::Value::String(s)) => s.clone(),
        Some(serde_json::Value::Array(parts)) => parts
            .iter()
            .filter_map(|p| {
                if p["type"].as_str() == Some("text") {
                    p["text"].as_str().map(String::from)
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
            .join("\n"),
        _ => String::new(),
    }
}

/// Serialize messages to a single prompt string for `claude -p`.
///
/// Uses the production-proven format from claude-max-api-proxy-rs:
/// - system messages → `<system>...</system>` (also extracted as --system-prompt)
/// Serialize a conversation to a single CLI prompt string.
///
/// Message roles are tagged with XML:
/// - system → `<system>...</system>`
/// - assistant → `<previous_response>...</previous_response>`
/// - user/tool/function → bare text
///
/// `system_override` provides a pre-extracted system prompt (Anthropic format);
/// when `None`, system prompts are extracted from inline `"system"` role messages (OpenAI format).
fn extract_prompt(
    messages: &[serde_json::Value],
    system_override: Option<&serde_json::Value>,
) -> (String, Option<String>) {
    // Extract system from top-level field (Anthropic) if provided
    let mut system_prompt = system_override.and_then(|s| match s {
        serde_json::Value::String(text) => Some(text.clone()),
        serde_json::Value::Array(blocks) => Some(
            blocks
                .iter()
                .filter_map(|b| b["text"].as_str())
                .collect::<Vec<_>>()
                .join("\n"),
        ),
        _ => None,
    });

    let mut parts: Vec<String> = Vec::new();

    // If system was provided via override, include it first
    if let Some(ref sys) = system_prompt {
        parts.push(format!("<system>\n{sys}\n</system>"));
    }

    for msg in messages {
        let role = msg["role"].as_str().unwrap_or("user");
        let text = extract_content_text(msg);
        if text.is_empty() {
            continue;
        }
        match role {
            "system" if system_prompt.is_none() => {
                // OpenAI format: system is an inline message role
                system_prompt = Some(text.clone());
                parts.push(format!("<system>\n{text}\n</system>"));
            }
            "assistant" => {
                parts.push(format!("<previous_response>\n{text}\n</previous_response>"));
            }
            _ => {
                parts.push(text);
            }
        }
    }

    let prompt = parts.join("\n\n").trim().to_string();
    (prompt, system_prompt)
}

/// Spawn the claude subprocess and return a stream of parsed NDJSON events.
async fn spawn_and_stream(
    config: &ClaudeSubscriptionConfig,
    model: &str,
    prompt: &str,
    system_prompt: Option<&str>,
    session_id: Option<&str>,
    stream: bool,
) -> Result<
    (
        tokio::process::Child,
        tokio::io::Lines<tokio::io::BufReader<tokio::process::ChildStdout>>,
    ),
    ApiError,
> {
    let binary = config.binary();
    let args = build_cli_args(model, prompt, system_prompt, session_id, stream);

    tracing::info!(
        binary = binary,
        model = model,
        session_id = session_id,
        stream = stream,
        "DIAG: claude_sub spawning subprocess"
    );

    // Default to /tmp to avoid loading the current project's CLAUDE.md, hooks,
    // MCP servers, and skills — we just want raw inference, not agent mode.
    // Users can override via working_dir config to give Claude project context.
    let cwd = config
        .working_dir
        .as_ref()
        .filter(|d| !d.is_empty() && *d != "none")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(std::env::temp_dir);

    let mut child = Command::new(binary)
        .args(&args)
        .current_dir(&cwd)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .map_err(|e| {
            ApiError(crate::error::SwarmError::Internal(format!(
                "Failed to spawn claude CLI: {e}"
            )))
        })?;

    let stdout = child.stdout.take().ok_or_else(|| {
        ApiError(crate::error::SwarmError::Internal(
            "claude subprocess: no stdout".into(),
        ))
    })?;
    let reader = tokio::io::BufReader::new(stdout);
    let lines = reader.lines();

    Ok((child, lines))
}

// ---------------------------------------------------------------------------
// OpenAI-format proxy
// ---------------------------------------------------------------------------

/// Common setup for proxy_via_subprocess_* functions.
struct ProxySetup {
    child: tokio::process::Child,
    lines: tokio::io::Lines<tokio::io::BufReader<tokio::process::ChildStdout>>,
    model: String,
    stream: bool,
    system_prompt: Option<String>,
    sess_key: Option<String>,
    timeout: std::time::Duration,
    _permit: tokio::sync::SemaphorePermit<'static>,
}

async fn prepare_proxy(
    config: &ClaudeSubscriptionConfig,
    req: &serde_json::Value,
    system_override: Option<&serde_json::Value>,
) -> Result<ProxySetup, ApiError> {
    let permit = acquire_permit(config)?;

    let model = req["model"].as_str().unwrap_or("claude-sonnet-4-6");
    let model = config.default_model.as_deref().unwrap_or(model);
    validate_model_name(model)?;
    let stream = req["stream"].as_bool().unwrap_or(false);
    let messages: Vec<serde_json::Value> = req["messages"].as_array().cloned().unwrap_or_default();

    let (prompt, system_prompt) = extract_prompt(&messages, system_override);

    let sess_key = session_key(model, &messages);
    let cached_session = sess_key.as_ref().and_then(|k| get_session(k));

    let (child, lines) = spawn_and_stream(
        config,
        model,
        &prompt,
        system_prompt.as_deref(),
        cached_session.as_deref(),
        stream,
    )
    .await?;

    Ok(ProxySetup {
        child,
        lines,
        model: model.to_string(),
        stream,
        system_prompt,
        sess_key,
        timeout: config.timeout(),
        _permit: permit,
    })
}

/// Proxy a ChatCompletionRequest through the Claude CLI, returning OpenAI format.
pub async fn proxy_via_subprocess_openai(
    config: &ClaudeSubscriptionConfig,
    req: &serde_json::Value,
) -> Result<axum::response::Response, ApiError> {
    let ProxySetup {
        mut child,
        mut lines,
        model,
        stream,
        system_prompt: _,
        sess_key,
        timeout,
        _permit,
    } = prepare_proxy(config, req, None).await?;

    let request_id = format!("chatcmpl-{}", uuid::Uuid::new_v4().simple());
    let created = chrono::Utc::now().timestamp();

    if stream {
        // Streaming: translate NDJSON → OpenAI SSE
        let model_owned = model.clone();
        let rid = request_id.clone();
        let sk = sess_key.clone();

        let stream = async_stream::stream! {
            let deadline = tokio::time::Instant::now() + timeout;
            loop {
                let line_result = tokio::time::timeout_at(deadline, lines.next_line()).await;
                match line_result {
                    Ok(Ok(Some(line))) => {
                        let trimmed = line.trim();
                        if trimmed.is_empty() || !trimmed.starts_with('{') {
                            continue;
                        }
                        let parsed: serde_json::Value = match serde_json::from_str(trimmed) {
                            Ok(v) => v,
                            Err(_) => continue,
                        };
                        let event_type = parsed["type"].as_str().unwrap_or("");
                        match event_type {
                            "stream_event" => {
                                let inner_type = parsed["event"]["type"].as_str().unwrap_or("");
                                match inner_type {
                                    "content_block_delta" => {
                                        let delta_type = parsed["event"]["delta"]["type"].as_str().unwrap_or("");
                                        if delta_type == "text_delta" {
                                            let text = parsed["event"]["delta"]["text"].as_str().unwrap_or("");
                                            let chunk = serde_json::json!({
                                                "id": rid,
                                                "object": "chat.completion.chunk",
                                                "created": created,
                                                "model": model_owned,
                                                "choices": [{
                                                    "index": 0,
                                                    "delta": { "content": text },
                                                    "finish_reason": null
                                                }]
                                            });
                                            yield Ok::<_, std::io::Error>(
                                                bytes::Bytes::from(format!("data: {}\n\n", chunk))
                                            );
                                        } else if delta_type == "thinking_delta" {
                                            // Extended thinking — pass through as a custom field
                                            let text = parsed["event"]["delta"]["thinking"].as_str().unwrap_or("");
                                            let chunk = serde_json::json!({
                                                "id": rid,
                                                "object": "chat.completion.chunk",
                                                "created": created,
                                                "model": model_owned,
                                                "choices": [{
                                                    "index": 0,
                                                    "delta": { "content": text },
                                                    "finish_reason": null
                                                }]
                                            });
                                            yield Ok(bytes::Bytes::from(format!("data: {}\n\n", chunk)));
                                        }
                                    }
                                    "message_delta" => {
                                        let stop = parsed["event"]["delta"]["stop_reason"].as_str();
                                        if stop.is_some() {
                                            let chunk = serde_json::json!({
                                                "id": rid,
                                                "object": "chat.completion.chunk",
                                                "created": created,
                                                "model": model_owned,
                                                "choices": [{
                                                    "index": 0,
                                                    "delta": {},
                                                    "finish_reason": "stop"
                                                }]
                                            });
                                            yield Ok(bytes::Bytes::from(format!("data: {}\n\n", chunk)));
                                        }
                                    }
                                    _ => {} // content_block_start, content_block_stop, message_start, message_stop — skip
                                }
                            }
                            "result" => {
                                // Cache session_id for multi-turn
                                if let (Some(ref key), Some(sid)) = (&sk, parsed["session_id"].as_str()) {
                                    put_session(key.clone(), sid.to_string());
                                }
                                // Usage info (optional — not all clients use it)
                                let usage = &parsed["usage"];
                                if !usage.is_null() {
                                    let input = usage["input_tokens"].as_u64().unwrap_or(0)
                                        + usage["cache_read_input_tokens"].as_u64().unwrap_or(0)
                                        + usage["cache_creation_input_tokens"].as_u64().unwrap_or(0);
                                    let output = usage["output_tokens"].as_u64().unwrap_or(0);
                                    let usage_chunk = serde_json::json!({
                                        "id": rid,
                                        "object": "chat.completion.chunk",
                                        "created": created,
                                        "model": model_owned,
                                        "choices": [],
                                        "usage": {
                                            "prompt_tokens": input,
                                            "completion_tokens": output,
                                            "total_tokens": input + output
                                        }
                                    });
                                    yield Ok(bytes::Bytes::from(format!("data: {}\n\n", usage_chunk)));
                                }
                                yield Ok(bytes::Bytes::from("data: [DONE]\n\n"));
                                break;
                            }
                            _ => {} // system, rate_limit_event, assistant — skip
                        }
                    }
                    Ok(Ok(None)) => {
                        // EOF
                        yield Ok(bytes::Bytes::from("data: [DONE]\n\n"));
                        break;
                    }
                    Ok(Err(e)) => {
                        tracing::error!(error = %e, "claude_sub: stdout read error");
                        break;
                    }
                    Err(_) => {
                        tracing::error!(model = %model_owned, "claude_sub: subprocess timeout");
                        let _ = child.kill().await;
                        break;
                    }
                }
            }
            let _ = child.wait().await;
        };

        let body = axum::body::Body::from_stream(stream);
        axum::response::Response::builder()
            .header("content-type", "text/event-stream")
            .header("cache-control", "no-cache")
            .body(body)
            .map_err(|e| {
                ApiError(crate::error::SwarmError::Internal(format!(
                    "Failed to build SSE response: {e}"
                )))
            })
    } else {
        // Non-streaming: collect full response
        let result = collect_result(&mut lines, timeout, &mut child).await?;

        // Cache session for multi-turn
        if let (Some(key), Some(sid)) =
            (sess_key, result.get("session_id").and_then(|v| v.as_str()))
        {
            put_session(key, sid.to_string());
        }

        let content = result["result"].as_str().unwrap_or("");
        let usage = &result["usage"];
        let input_tokens = usage["input_tokens"].as_u64().unwrap_or(0)
            + usage["cache_read_input_tokens"].as_u64().unwrap_or(0)
            + usage["cache_creation_input_tokens"].as_u64().unwrap_or(0);
        let output_tokens = usage["output_tokens"].as_u64().unwrap_or(0);

        let response = serde_json::json!({
            "id": request_id,
            "object": "chat.completion",
            "created": created,
            "model": model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens
            }
        });

        Ok(axum::Json(response).into_response())
    }
}

// ---------------------------------------------------------------------------
// Anthropic-format proxy
// ---------------------------------------------------------------------------

/// Proxy a MessagesRequest through the Claude CLI, returning Anthropic format.
pub async fn proxy_via_subprocess_anthropic(
    config: &ClaudeSubscriptionConfig,
    req: &serde_json::Value,
) -> Result<axum::response::Response, ApiError> {
    let ProxySetup {
        mut child,
        mut lines,
        model,
        stream,
        system_prompt: _,
        sess_key,
        timeout,
        _permit,
    } = prepare_proxy(config, req, req.get("system")).await?;

    if stream {
        // Streaming: translate NDJSON → Anthropic SSE
        // Claude CLI stream_events are already in Anthropic format — near pass-through
        let model_owned = model.to_string();
        let sk = sess_key.clone();

        let stream = async_stream::stream! {
            let deadline = tokio::time::Instant::now() + timeout;

            // Emit message_start event
            let msg_start = serde_json::json!({
                "type": "message_start",
                "message": {
                    "id": format!("msg_{}", uuid::Uuid::new_v4().simple()),
                    "type": "message",
                    "role": "assistant",
                    "model": model_owned,
                    "content": [],
                    "stop_reason": null,
                    "usage": { "input_tokens": 0, "output_tokens": 0 }
                }
            });
            yield Ok::<_, std::io::Error>(
                bytes::Bytes::from(format!("event: message_start\ndata: {}\n\n", msg_start))
            );

            let mut emitted_block_start = false;

            loop {
                let line_result = tokio::time::timeout_at(deadline, lines.next_line()).await;
                match line_result {
                    Ok(Ok(Some(line))) => {
                        let trimmed = line.trim();
                        if trimmed.is_empty() || !trimmed.starts_with('{') { continue; }
                        let parsed: serde_json::Value = match serde_json::from_str(trimmed) {
                            Ok(v) => v,
                            Err(_) => continue,
                        };
                        let event_type = parsed["type"].as_str().unwrap_or("");
                        match event_type {
                            "stream_event" => {
                                let inner = &parsed["event"];
                                let inner_type = inner["type"].as_str().unwrap_or("");
                                match inner_type {
                                    "content_block_start" | "content_block_delta"
                                    | "content_block_stop" | "message_delta" => {
                                        // Ensure we emit content_block_start before first delta
                                        if inner_type == "content_block_delta" && !emitted_block_start {
                                            let block_start = serde_json::json!({
                                                "type": "content_block_start",
                                                "index": 0,
                                                "content_block": { "type": "text", "text": "" }
                                            });
                                            yield Ok(bytes::Bytes::from(format!(
                                                "event: content_block_start\ndata: {}\n\n", block_start
                                            )));
                                            emitted_block_start = true;
                                        }
                                        if inner_type == "content_block_start" {
                                            emitted_block_start = true;
                                        }
                                        yield Ok(bytes::Bytes::from(format!(
                                            "event: {}\ndata: {}\n\n", inner_type, inner
                                        )));
                                    }
                                    "message_stop" => {
                                        yield Ok(bytes::Bytes::from(format!(
                                            "event: message_stop\ndata: {}\n\n", inner
                                        )));
                                    }
                                    _ => {} // message_start from CLI — we already emitted our own
                                }
                            }
                            "result" => {
                                if let (Some(ref key), Some(sid)) = (&sk, parsed["session_id"].as_str()) {
                                    put_session(key.clone(), sid.to_string());
                                }
                                break;
                            }
                            _ => {}
                        }
                    }
                    Ok(Ok(None)) => break,
                    Ok(Err(e)) => {
                        tracing::error!(error = %e, "claude_sub: stdout read error");
                        break;
                    }
                    Err(_) => {
                        tracing::error!(model = %model_owned, "claude_sub: subprocess timeout");
                        let _ = child.kill().await;
                        break;
                    }
                }
            }
            let _ = child.wait().await;
        };

        let body = axum::body::Body::from_stream(stream);
        axum::response::Response::builder()
            .header("content-type", "text/event-stream")
            .header("cache-control", "no-cache")
            .body(body)
            .map_err(|e| {
                ApiError(crate::error::SwarmError::Internal(format!(
                    "Failed to build SSE response: {e}"
                )))
            })
    } else {
        // Non-streaming: collect and return Anthropic JSON
        let result = collect_result(&mut lines, timeout, &mut child).await?;

        if let (Some(key), Some(sid)) =
            (sess_key, result.get("session_id").and_then(|v| v.as_str()))
        {
            put_session(key, sid.to_string());
        }

        let content = result["result"].as_str().unwrap_or("");
        let usage = &result["usage"];
        let input_tokens = usage["input_tokens"].as_u64().unwrap_or(0);
        let cache_read = usage["cache_read_input_tokens"].as_u64().unwrap_or(0);
        let cache_creation = usage["cache_creation_input_tokens"].as_u64().unwrap_or(0);
        let output_tokens = usage["output_tokens"].as_u64().unwrap_or(0);

        let response = serde_json::json!({
            "id": format!("msg_{}", uuid::Uuid::new_v4().simple()),
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [{
                "type": "text",
                "text": content,
            }],
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cache_read_input_tokens": cache_read,
                "cache_creation_input_tokens": cache_creation,
            }
        });

        Ok(axum::Json(response).into_response())
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Collect the final `result` event from a non-streaming NDJSON response.
async fn collect_result(
    lines: &mut tokio::io::Lines<tokio::io::BufReader<tokio::process::ChildStdout>>,
    timeout: std::time::Duration,
    child: &mut tokio::process::Child,
) -> Result<serde_json::Value, ApiError> {
    let deadline = tokio::time::Instant::now() + timeout;

    loop {
        let line_result = tokio::time::timeout_at(deadline, lines.next_line()).await;
        match line_result {
            Ok(Ok(Some(line))) => {
                let trimmed = line.trim();
                if trimmed.is_empty() || !trimmed.starts_with('{') {
                    continue;
                }
                // Security: cap per-line size to prevent OOM from misbehaving subprocess
                if line.len() > MAX_RESPONSE_LINE {
                    let _ = child.kill().await;
                    return Err(ApiError(crate::error::SwarmError::Internal(
                        "Claude CLI response line too large (>1MB)".into(),
                    )));
                }
                let parsed: serde_json::Value = match serde_json::from_str(trimmed) {
                    Ok(v) => v,
                    Err(_) => continue,
                };
                if parsed["type"].as_str() == Some("result") {
                    let _ = child.wait().await;
                    if parsed["is_error"].as_bool() == Some(true) {
                        let err_msg = parsed["result"]
                            .as_str()
                            .unwrap_or("Claude CLI returned an error");
                        return Err(ApiError(crate::error::SwarmError::Internal(format!(
                            "Claude subscription error: {err_msg}"
                        ))));
                    }
                    return Ok(parsed);
                }
            }
            Ok(Ok(None)) => {
                // EOF without result — check stderr
                let _ = child.wait().await;
                let stderr_output = if let Some(mut stderr) = child.stderr.take() {
                    let mut buf = String::new();
                    let _ = tokio::io::AsyncReadExt::read_to_string(&mut stderr, &mut buf).await;
                    buf
                } else {
                    String::new()
                };
                return Err(ApiError(crate::error::SwarmError::Internal(format!(
                    "Claude CLI exited without result. stderr: {}",
                    stderr_output.chars().take(500).collect::<String>()
                ))));
            }
            Ok(Err(e)) => {
                return Err(ApiError(crate::error::SwarmError::Internal(format!(
                    "Claude CLI stdout error: {e}"
                ))));
            }
            Err(_) => {
                let _ = child.kill().await;
                return Err(ApiError(crate::error::SwarmError::Internal(
                    "Claude CLI subprocess timed out".into(),
                )));
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Admin status handler
// ---------------------------------------------------------------------------

/// Handler for GET /api/admin/claude-subscription/status
pub async fn get_status(
    axum::extract::State(state): axum::extract::State<crate::api::server::AppState>,
) -> Result<axum::Json<ClaudeCliStatus>, ApiError> {
    let config = state.shared_state.metrics.providers_config.read().await;
    let binary = config
        .claude_subscription
        .as_ref()
        .map(|c| c.binary())
        .unwrap_or("claude");
    let binary = binary.to_string();
    drop(config);

    let status = detect_cli(&binary).await;

    // Emit activity event with detection result
    if status.cli_installed && status.authenticated {
        let plan = status.subscription_type.as_deref().unwrap_or("unknown");
        state.shared_state.emit_activity(
            crate::daemon::state::ActivityEvent::new(
                "provider",
                "claude_detected",
                format!(
                    "Claude Code CLI detected — {} plan, {}",
                    plan,
                    status.cli_version.as_deref().unwrap_or("unknown version")
                ),
            )
            .with_toast("info", 3000),
        );
    } else if status.cli_installed {
        state.shared_state.emit_activity(
            crate::daemon::state::ActivityEvent::new(
                "provider",
                "claude_detected",
                "Claude Code CLI found but not authenticated — run 'claude login'".to_string(),
            )
            .with_toast("warning", 4000),
        );
    }

    Ok(axum::Json(status))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_cli_args_basic() {
        let args = build_cli_args("claude-sonnet-4-6", "Hello", None, None, true);
        assert!(args.contains(&"--print".to_string()));
        assert!(args.contains(&"stream-json".to_string()));
        assert!(args.contains(&"--model".to_string()));
        assert!(args.contains(&"claude-sonnet-4-6".to_string()));
        assert!(args.contains(&"--no-session-persistence".to_string()));
        assert!(args.contains(&"Hello".to_string()));
        assert!(!args.contains(&"--resume".to_string()));
    }

    #[test]
    fn test_build_cli_args_with_session() {
        let args = build_cli_args(
            "claude-opus-4-6",
            "Follow up",
            Some("Be helpful"),
            Some("abc-123"),
            false,
        );
        assert!(args.contains(&"--resume".to_string()));
        assert!(args.contains(&"abc-123".to_string()));
        assert!(args.contains(&"--system-prompt".to_string()));
        assert!(args.contains(&"Be helpful".to_string()));
        assert!(args.contains(&"json".to_string())); // non-streaming
        assert!(!args.contains(&"--no-session-persistence".to_string()));
        assert!(!args.contains(&"--verbose".to_string())); // non-streaming
    }

    #[test]
    fn test_extract_prompt_single_turn() {
        let messages = vec![serde_json::json!({"role": "user", "content": "Hello"})];
        let (prompt, system) = extract_prompt(&messages, None);
        assert_eq!(prompt, "Hello");
        assert!(system.is_none());
    }

    #[test]
    fn test_extract_prompt_multi_turn() {
        let messages = vec![
            serde_json::json!({"role": "system", "content": "You are helpful"}),
            serde_json::json!({"role": "user", "content": "What is 2+2?"}),
            serde_json::json!({"role": "assistant", "content": "4"}),
            serde_json::json!({"role": "user", "content": "And 3+3?"}),
        ];
        let (prompt, system) = extract_prompt(&messages, None);
        assert!(prompt.contains("<system>"));
        assert!(prompt.contains("You are helpful"));
        assert!(prompt.contains("<previous_response>"));
        assert!(prompt.contains("4"));
        assert!(prompt.contains("And 3+3?"));
        assert_eq!(system.unwrap(), "You are helpful");
    }

    #[test]
    fn test_session_key_stability() {
        let msgs1 = vec![
            serde_json::json!({"role": "user", "content": "Hi"}),
            serde_json::json!({"role": "assistant", "content": "Hello"}),
            serde_json::json!({"role": "user", "content": "New question"}),
        ];
        let msgs2 = vec![
            serde_json::json!({"role": "user", "content": "Hi"}),
            serde_json::json!({"role": "assistant", "content": "Hello"}),
            serde_json::json!({"role": "user", "content": "Different question"}),
        ];
        let key1 = session_key("claude-sonnet-4-6", &msgs1);
        let key2 = session_key("claude-sonnet-4-6", &msgs2);
        // Same prefix (first 2 messages), different last message — same session key
        assert_eq!(key1, key2);
        // Both should be Some (multi-turn)
        assert!(key1.is_some());
    }

    #[test]
    fn test_session_key_single_turn_none() {
        let msgs = vec![serde_json::json!({"role": "user", "content": "Hi"})];
        let key = session_key("claude-sonnet-4-6", &msgs);
        // Single-turn requests should NOT produce a session key
        assert!(key.is_none());
    }

    #[test]
    fn test_session_cache_put_get() {
        put_session("test-key".into(), "session-abc".into());
        assert_eq!(get_session("test-key"), Some("session-abc".to_string()));
        assert_eq!(get_session("nonexistent"), None);
    }

    #[test]
    fn test_config_defaults() {
        let config = ClaudeSubscriptionConfig::default();
        assert_eq!(config.binary(), "claude");
        assert_eq!(config.timeout(), std::time::Duration::from_secs(300));
        assert_eq!(config.concurrency_limit(), 3);
    }
}
