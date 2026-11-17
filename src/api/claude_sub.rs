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

/// Check if a model should be routed through Claude subscription and return the config if so.
///
/// Returns `Some(config)` when the model starts with "claude", a subscription
/// is enabled in provider config, and the request originated locally
/// (`is_forwarded == false`). Both openai.rs and anthropic.rs call this to
/// avoid duplicating the check logic.
///
/// SEC: peer-forwarded inference requests must NEVER consume the local
/// operator's personal Claude subscription. The subscription is bound to
/// `~/.claude/.credentials.json` on this host — burning it for cluster
/// peers exhausts the operator's Pro/Max quota and is an unintended
/// resource transfer. Local-operator requests never carry
/// `x-swarm-forwarded`; that header is set by `peer_forward.rs` on
/// peer-to-peer hops only.
pub async fn try_get_claude_subscription(
    state: &crate::api::server::AppState,
    model: &str,
    is_forwarded: bool,
) -> Option<ClaudeSubscriptionConfig> {
    if is_forwarded {
        return None;
    }
    let lower = model.to_lowercase();
    if !lower.starts_with("claude") {
        return None;
    }
    let config = state.shared_state.metrics.providers_config.read().await;
    let sub = config.claude_subscription.as_ref()?;
    if !sub.enabled {
        return None;
    }
    Some(sub.clone())
}

// ---------------------------------------------------------------------------
// Concurrency limiter (global semaphore)
// ---------------------------------------------------------------------------

/// Maximum concurrent Claude subprocess sessions.
const SUBPROCESS_SEMAPHORE_CAPACITY: usize = 8;
/// Concurrency limiter for Claude subprocess sessions.
static SUBPROCESS_SEMAPHORE: LazyLock<Semaphore> =
    LazyLock::new(|| Semaphore::new(SUBPROCESS_SEMAPHORE_CAPACITY));

fn acquire_permit(
    config: &ClaudeSubscriptionConfig,
) -> Result<tokio::sync::SemaphorePermit<'static>, ApiError> {
    let limit = config
        .concurrency_limit()
        .min(SUBPROCESS_SEMAPHORE_CAPACITY);
    // Enforce configured limit by checking available permits.
    let available = SUBPROCESS_SEMAPHORE.available_permits();
    if available <= SUBPROCESS_SEMAPHORE_CAPACITY.saturating_sub(limit) {
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

/// A conversation state ([`Conversation::key`]) → the CLI session holding
/// it, so the caller's next turn resumes it (FUTURE_WORK #471).
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
    // SEC: cap the cached session_id length. The value is parsed from the
    // `claude` subprocess's NDJSON stdout and used as the `--resume <sid>`
    // CLI argument on the next invocation. A compromised/replaced binary
    // could emit megabyte-scale session IDs to fill up to 200 cache
    // entries with large strings — bounded targeted memory exhaustion.
    // Real session IDs are short UUIDs.
    const MAX_SESSION_ID_BYTES: usize = 512;
    if session_id.len() > MAX_SESSION_ID_BYTES {
        tracing::warn!(
            len = session_id.len(),
            cap = MAX_SESSION_ID_BYTES,
            "claude_sub: session_id exceeds cap — not caching"
        );
        return;
    }
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

/// A conversation as a CLI session holds it (FUTURE_WORK #471): the model, the
/// folder the CLI runs in (it finds a session under its folder), the system
/// text, and each message's role and text as [`extract_prompt`] reads them —
/// a message with no text is skipped there and here. Two requests can share a
/// session exactly when these agree, whatever shape the client gave the
/// content (a string or text parts) and whatever order its JSON keys came in.
///
/// **Why sessions are kept by the state they HOLD.** The key was the
/// conversation before its last message, stored for the session that answered
/// it, and every call still sent the whole history. A normal next turn never
/// matched (its earlier messages now include the reply), so the resume fired
/// only on an edit or a regenerate — into a session holding the question the
/// caller took back, with the history sent a second time. And the first call
/// ran with `--no-session-persistence`, so that resume found nothing and the
/// request failed ("No conversation found", checked 2026-11-18 on CLI
/// 2.1.326). Now each reply stores the session under the conversation
/// INCLUDING the reply; the next turn finds it by the conversation before its
/// last message, forks it (`--fork-session`, so the stored state is never
/// changed and a regenerate resumes the same one) and sends only that message.
/// Measured on Haiku, a second turn wrote 102 tokens to the prompt cache and
/// read 31,722, against 17,426 written and 14,328 read for the whole history
/// sent fresh — cache writes cost about twelve times reads.
///
/// Prior art: OpenAI's Responses API `previous_response_id` — each reply is
/// an immutable state and a request names the one it continues, so an edit
/// branches instead of overwriting; the CLI's `--resume` + `--fork-session`
/// is that model for a CLI session.
#[derive(Clone, Debug, PartialEq)]
struct Conversation {
    model: String,
    folder: std::path::PathBuf,
    system: Option<String>,
    turns: Vec<(String, String)>,
}

impl Conversation {
    fn of(
        model: &str,
        folder: &std::path::Path,
        system: Option<&str>,
        messages: &[serde_json::Value],
    ) -> Self {
        let turns = messages
            .iter()
            .filter_map(|msg| {
                let text = extract_content_text(msg);
                let text = text.trim();
                (!text.is_empty()).then(|| {
                    let role = msg["role"].as_str().unwrap_or("user");
                    (role.to_string(), text.to_string())
                })
            })
            .collect();
        Self {
            model: model.to_string(),
            folder: folder.to_path_buf(),
            system: system.map(str::to_string),
            turns,
        }
    }

    /// The session cache's key: BLAKE3 (cryptographic, so two conversations
    /// never share a session by collision) over every field, each
    /// length-prefixed so no two conversations share an encoding.
    fn key(&self) -> String {
        fn field(hasher: &mut blake3::Hasher, text: &str) {
            hasher.update(&(text.len() as u64).to_le_bytes());
            hasher.update(text.as_bytes());
        }
        let mut hasher = blake3::Hasher::new();
        field(&mut hasher, &self.model);
        field(&mut hasher, &self.folder.to_string_lossy());
        match &self.system {
            Some(system) => {
                field(&mut hasher, "1");
                field(&mut hasher, system);
            }
            None => field(&mut hasher, "0"),
        }
        for (role, text) in &self.turns {
            field(&mut hasher, role);
            field(&mut hasher, text);
        }
        format!(
            "claude_sub:{}:{}",
            self.model,
            &hasher.finalize().to_hex()[..32]
        )
    }

    /// The conversation before its last message, and that message — only
    /// where the last message is the caller's (not the system text, not an
    /// assistant turn to continue) and something precedes it.
    fn before_last(&self) -> Option<(Self, &str)> {
        let ((role, text), before) = self.turns.split_last()?;
        if before.is_empty() || role == "assistant" || role == "system" {
            return None;
        }
        let before = Self {
            model: self.model.clone(),
            folder: self.folder.clone(),
            system: self.system.clone(),
            turns: before.to_vec(),
        };
        Some((before, text))
    }

    /// This conversation once `reply` answered it — what the CLI's session
    /// holds after the call, and what the caller sends back next turn.
    fn with_reply(&self, reply: &str) -> Self {
        let mut next = self.clone();
        let reply = reply.trim();
        if !reply.is_empty() {
            next.turns
                .push(("assistant".to_string(), reply.to_string()));
        }
        next
    }
}

/// After a reply, the conversation with it is the state the call's session
/// holds: kept for the caller's next turn ([`Conversation`]). Never after a
/// failure, whose session holds no reply.
fn remember_reply(conversation: &Conversation, result: &serde_json::Value) {
    if result["is_error"].as_bool() == Some(true) {
        return;
    }
    if let (Some(reply), Some(sid)) = (result["result"].as_str(), result["session_id"].as_str()) {
        put_session(conversation.with_reply(reply).key(), sid.to_string());
    }
}

/// A stored session the CLI could not resume is never handed to it again.
fn forget_session(key: &str) {
    SESSION_CACHE.remove(key);
}

/// Periodic cleanup of expired sessions — called lazily from `put_session()`.
fn cleanup_expired_sessions() {
    SESSION_CACHE.retain(|_, entry| entry.last_used.elapsed() < SESSION_TTL);
}

/// Extract total input tokens from a CLI usage JSON object, summing base +
/// cache_read + cache_creation components.
/// Build an OpenAI-format streaming chunk with a single text delta.
/// Shared between `text_delta`, `thinking_delta`, and terminal `stop` frames
/// so all chat.completion.chunk emissions flow through the same typed struct.
fn build_content_chunk(
    id: &str,
    created: i64,
    model: &str,
    content: String,
    finish_reason: Option<String>,
) -> super::openai::ChatCompletionChunk {
    let delta = super::openai::Delta {
        role: None,
        content: if content.is_empty() && finish_reason.is_some() {
            None
        } else {
            Some(content)
        },
        reasoning_content: None,
        tool_calls: None,
    };
    super::openai::ChatCompletionChunk {
        id: id.to_string(),
        object: "chat.completion.chunk",
        created,
        model: model.to_string(),
        choices: vec![super::openai::ChunkChoice {
            index: 0,
            delta,
            finish_reason,
            logprobs: None,
        }],
        session_id: None,
        usage: None,
        // A Claude subscription's reply: a cloud provider, no swarm route.
        swarmllm: None,
    }
}

fn extract_total_input_tokens(usage: &serde_json::Value) -> u64 {
    usage["input_tokens"].as_u64().unwrap_or(0)
        + usage["cache_read_input_tokens"].as_u64().unwrap_or(0)
        + usage["cache_creation_input_tokens"].as_u64().unwrap_or(0)
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
    // Every session is saved, so the caller's next turn can resume it; a
    // resume forks, so the state it was stored as is never changed and a
    // regenerate finds it again ([`Conversation`], #471).
    if let Some(sid) = session_id {
        args.push("--resume".to_string());
        args.push(sid.to_string());
        args.push("--fork-session".to_string());
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
///
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

    let cwd = cli_folder(config);

    let mut child = Command::new(binary)
        .args(&args)
        .current_dir(&cwd)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .map_err(|e| {
            ApiError(crate::error::SwarmError::ServiceUnavailable(format!(
                "Failed to spawn claude CLI: {e}"
            )))
        })?;

    let stdout = child.stdout.take().ok_or_else(|| {
        ApiError(crate::error::SwarmError::ServiceUnavailable(
            "claude subprocess: no stdout".into(),
        ))
    })?;
    let reader = tokio::io::BufReader::new(stdout);
    let lines = reader.lines();

    Ok((child, lines))
}

/// The folder the CLI runs in — the temp dir by default, so the current
/// project's CLAUDE.md, hooks, MCP servers and skills stay out (raw inference,
/// not agent mode); `working_dir` gives Claude a project's context. A session
/// is saved under its folder, so it is part of [`Conversation`].
fn cli_folder(config: &ClaudeSubscriptionConfig) -> std::path::PathBuf {
    config
        .working_dir
        .as_ref()
        .filter(|d| !d.is_empty() && *d != "none")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(std::env::temp_dir)
}

/// The CLI's output lines, with any line read ahead of the caller handed out
/// first ([`Self::resume_refused`]).
struct CliLines {
    ahead: std::collections::VecDeque<String>,
    rest: tokio::io::Lines<tokio::io::BufReader<tokio::process::ChildStdout>>,
}

impl CliLines {
    fn new(rest: tokio::io::Lines<tokio::io::BufReader<tokio::process::ChildStdout>>) -> Self {
        Self {
            ahead: std::collections::VecDeque::new(),
            rest,
        }
    }

    async fn next_line(&mut self) -> std::io::Result<Option<String>> {
        match self.ahead.pop_front() {
            Some(line) => Ok(Some(line)),
            None => self.rest.next_line().await,
        }
    }

    /// Did the CLI refuse the session it was told to resume? Reads ahead to
    /// the first event past its start-up ones (`system`): a refusal is an
    /// error result there before any turn ran (`--output-format stream-json`),
    /// or no output at all (`json`, whose refusal goes to stderr) — "No
    /// conversation found with session ID", 2026-11-18 on CLI 2.1.326. Every
    /// line read is kept for the caller. A read error or the deadline is left
    /// for the caller to meet.
    async fn resume_refused(&mut self, deadline: tokio::time::Instant) -> bool {
        loop {
            match tokio::time::timeout_at(deadline, self.rest.next_line()).await {
                Ok(Ok(Some(line))) => {
                    let event = serde_json::from_str::<serde_json::Value>(line.trim()).ok();
                    self.ahead.push_back(line);
                    let Some(event) = event else { continue };
                    match event["type"].as_str() {
                        Some("system") => continue,
                        Some("result") => {
                            return event["is_error"].as_bool() == Some(true)
                                && event["num_turns"].as_u64().unwrap_or(0) == 0;
                        }
                        _ => return false,
                    }
                }
                Ok(Ok(None)) => return true,
                Ok(Err(_)) | Err(_) => return false,
            }
        }
    }
}

// ---------------------------------------------------------------------------
// OpenAI-format proxy
// ---------------------------------------------------------------------------

/// Common setup for proxy_via_subprocess_* functions.
struct ProxySetup {
    child: tokio::process::Child,
    lines: CliLines,
    model: String,
    stream: bool,
    /// The conversation this call answers; with the reply, the state its
    /// session holds after ([`remember_reply`]).
    conversation: Conversation,
    /// The call's whole deadline, set before the CLI was started — a resume
    /// it refused and the call that replaced it included.
    deadline: tokio::time::Instant,
    _permit: tokio::sync::SemaphorePermit<'static>,
}

async fn prepare_proxy(
    config: &ClaudeSubscriptionConfig,
    req: &serde_json::Value,
    system_override: Option<&serde_json::Value>,
) -> Result<ProxySetup, ApiError> {
    let permit = acquire_permit(config)?;

    let model = req["model"].as_str().unwrap_or("claude-sonnet-5-5");
    let model = config.default_model.as_deref().unwrap_or(model);
    validate_model_name(model)?;
    let stream = req["stream"].as_bool().unwrap_or(false);
    let messages: Vec<serde_json::Value> = req["messages"].as_array().cloned().unwrap_or_default();

    let deadline = tokio::time::Instant::now() + config.timeout();

    let (whole_prompt, system_prompt) = extract_prompt(&messages, system_override);
    let conversation = Conversation::of(
        model,
        &cli_folder(config),
        system_prompt.as_deref(),
        &messages,
    );
    // A session holding the conversation up to its last message: forked, and
    // handed that message alone ([`Conversation`], #471).
    let resume = conversation.before_last().and_then(|(before, last)| {
        let key = before.key();
        get_session(&key).map(|sid| (key, sid, last.to_string()))
    });
    let prompt = resume
        .as_ref()
        .map_or(whole_prompt.as_str(), |(_, _, last)| last.as_str());

    let (mut child, lines) = spawn_and_stream(
        config,
        model,
        prompt,
        system_prompt.as_deref(),
        resume.as_ref().map(|(_, sid, _)| sid.as_str()),
        stream,
    )
    .await?;
    let mut lines = CliLines::new(lines);

    if let Some((key, sid, _)) = &resume {
        if lines.resume_refused(deadline).await {
            tracing::warn!(
                model,
                session_id = %sid,
                "claude_sub: the CLI could not resume this conversation's session — \
                 sending the whole conversation instead"
            );
            forget_session(key);
            let _ = child.kill().await;
            let _ = child.wait().await;
            let (fresh, fresh_lines) = spawn_and_stream(
                config,
                model,
                &whole_prompt,
                system_prompt.as_deref(),
                None,
                stream,
            )
            .await?;
            child = fresh;
            lines = CliLines::new(fresh_lines);
        }
    }

    Ok(ProxySetup {
        child,
        lines,
        model: model.to_string(),
        stream,
        conversation,
        deadline,
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

        conversation,
        deadline,
        _permit,
    } = prepare_proxy(config, req, None).await?;

    let request_id = format!("chatcmpl-{}", uuid::Uuid::new_v4().simple());
    let created = chrono::Utc::now().timestamp();

    if stream {
        // Streaming: translate NDJSON → OpenAI SSE
        let model_owned = model.clone();
        let rid = request_id.clone();

        let stream = async_stream::stream! {
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
                                        // PRIVACY: drop `thinking_delta` from the OpenAI-format
                                        // proxy. Anthropic's extended thinking carries the model's
                                        // chain-of-thought; the OpenAI streaming chunk schema has
                                        // no `reasoning` field, so the previous code concatenated
                                        // it into `choices[0].delta.content`. Clients see internal
                                        // reasoning indistinguishable from the actual answer —
                                        // a privacy regression vs. talking to Anthropic directly.
                                        // The Anthropic-format proxy below preserves typed
                                        // thinking blocks for clients that explicitly support them.
                                        let text_opt = if delta_type == "text_delta" {
                                            Some(parsed["event"]["delta"]["text"].as_str().unwrap_or("").to_string())
                                        } else {
                                            None
                                        };
                                        if let Some(text) = text_opt {
                                            let chunk = build_content_chunk(&rid, created, &model_owned, text, None);
                                            yield Ok::<_, std::io::Error>(crate::api::sse::data_frame(&chunk));
                                        }
                                    }
                                    "message_delta"
                                        if parsed["event"]["delta"]["stop_reason"].as_str().is_some() =>
                                    {
                                        let chunk = build_content_chunk(&rid, created, &model_owned, String::new(), Some("stop".into()));
                                        yield Ok(crate::api::sse::data_frame(&chunk));
                                    }
                                    _ => {} // content_block_start, content_block_stop, message_start, message_stop — skip
                                }
                            }
                            "result" => {
                                // The state this session now holds, for the next turn (#471).
                                remember_reply(&conversation, &parsed);
                                // Usage info (optional — not all clients use it)
                                let usage = &parsed["usage"];
                                if !usage.is_null() {
                                    let input = extract_total_input_tokens(usage);
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
                                    yield Ok(crate::api::sse::data_frame(&usage_chunk));
                                }
                                yield Ok(crate::api::sse::done_frame());
                                break;
                            }
                            _ => {} // system, rate_limit_event, assistant — skip
                        }
                    }
                    Ok(Ok(None)) => {
                        // EOF
                        yield Ok(crate::api::sse::done_frame());
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
        super::providers::build_sse_response(body)
    } else {
        // Non-streaming: collect full response
        let result = collect_result(&mut lines, deadline, &mut child).await?;
        remember_reply(&conversation, &result);

        let content = result["result"].as_str().unwrap_or("");
        let usage = &result["usage"];
        let input_tokens = extract_total_input_tokens(usage);
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
    // Boundary defence — see `providers::strip_prefix_in_body`. The Claude CLI
    // is as unaware of `anthropic:` as any cloud provider.
    let req = &*crate::api::providers::strip_prefix_in_body(req);
    let ProxySetup {
        mut child,
        mut lines,
        model,
        stream,

        conversation,
        deadline,
        _permit,
    } = prepare_proxy(config, req, req.get("system")).await?;

    if stream {
        // Streaming: translate NDJSON → Anthropic SSE
        // Claude CLI stream_events are already in Anthropic format — near pass-through
        let model_owned = model.to_string();

        let stream = async_stream::stream! {

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
                crate::api::sse::event_frame("message_start", &msg_start)
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
                                remember_reply(&conversation, &parsed);
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
        super::providers::build_sse_response(body)
    } else {
        // Non-streaming: collect and return Anthropic JSON
        let result = collect_result(&mut lines, deadline, &mut child).await?;
        remember_reply(&conversation, &result);

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
    lines: &mut CliLines,
    deadline: tokio::time::Instant,
    child: &mut tokio::process::Child,
) -> Result<serde_json::Value, ApiError> {
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
                    return Err(ApiError(crate::error::SwarmError::ServiceUnavailable(
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
                        return Err(ApiError(crate::error::SwarmError::ServiceUnavailable(
                            format!("Claude subscription error: {err_msg}"),
                        )));
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
                return Err(ApiError(crate::error::SwarmError::ServiceUnavailable(
                    format!(
                        "Claude CLI exited without result. stderr: {}",
                        stderr_output.chars().take(500).collect::<String>()
                    ),
                )));
            }
            Ok(Err(e)) => {
                return Err(ApiError(crate::error::SwarmError::ServiceUnavailable(
                    format!("Claude CLI stdout error: {e}"),
                )));
            }
            Err(_) => {
                let _ = child.kill().await;
                return Err(ApiError(crate::error::SwarmError::ServiceUnavailable(
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

    // Deliberately emits NO ActivityEvent.
    //
    // This is a GET, and it used to append to the user's activity list and
    // raise a toast on EVERY call — measured 2026-09-14 on an isolated node:
    // three requests took the list from one entry to four. `loadClaudeSubStatus`
    // is called by the header strip, the dashboard panel and the settings panel,
    // so opening the dashboard was enough, repeatedly, into a 100-entry ring
    // that is also the replay a dashboard opens on and the "recent activity" of
    // the pasteable report.
    //
    // Nothing was lost by removing it. All three callers render `status`
    // themselves from this very response, so the event and the toast restated
    // what the panel that asked was already about to show; no frontend code
    // referenced the `claude_detected` kind at all. A read endpoint should not
    // write to the activity list — and where an event about a detection IS
    // wanted, it belongs on the transition, not on the read
    // (`docs/invariants/network.md` § "The activity list reports a TRANSITION").
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
        let args = build_cli_args("claude-sonnet-5", "Hello", None, None, true);
        assert!(args.contains(&"--print".to_string()));
        assert!(args.contains(&"stream-json".to_string()));
        assert!(args.contains(&"--model".to_string()));
        assert!(args.contains(&"claude-sonnet-5".to_string()));
        assert!(args.contains(&"Hello".to_string()));
        assert!(!args.contains(&"--resume".to_string()));
        assert!(!args.contains(&"--fork-session".to_string()));
        // Saved, so the caller's next turn can resume it (#471).
        assert!(!args.contains(&"--no-session-persistence".to_string()));
    }

    #[test]
    fn test_build_cli_args_with_session() {
        let args = build_cli_args(
            "claude-opus-4-8",
            "Follow up",
            Some("Be helpful"),
            Some("abc-123"),
            false,
        );
        assert!(args.contains(&"--resume".to_string()));
        assert!(args.contains(&"abc-123".to_string()));
        // Forked: the stored state stays what it was, for a regenerate (#471).
        assert!(args.contains(&"--fork-session".to_string()));
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

    fn conversation(messages: &[serde_json::Value]) -> Conversation {
        let (_, system) = extract_prompt(messages, None);
        Conversation::of(
            "claude-haiku-5-5",
            std::path::Path::new("/tmp"),
            system.as_deref(),
            messages,
        )
    }

    /// #471: a reply is stored under the conversation INCLUDING it, and the
    /// caller's next turn finds it by its messages before the new one —
    /// whatever shape the client sends the reply back in.
    #[test]
    fn a_reply_is_found_by_the_callers_next_turn() {
        let first = conversation(&[serde_json::json!({"role": "user", "content": "Hi"})]);
        let stored = first.with_reply("Hello").key();
        // The reply back as text parts, padded, keys in another order.
        let next = conversation(&[
            serde_json::json!({"role": "user", "content": "Hi"}),
            serde_json::json!({"content": [{"text": "Hello\n", "type": "text"}], "role": "assistant"}),
            serde_json::json!({"role": "user", "content": "And now?"}),
        ]);
        let (before, last) = next.before_last().expect("the caller's own last message");
        assert_eq!(before.key(), stored);
        assert_eq!(last, "And now?");
    }

    /// #471: an edit or a regenerate of the last question resumes the state
    /// before it — never one holding the question it replaces.
    #[test]
    fn an_edit_resumes_the_state_before_the_question_it_replaces() {
        let history = [
            serde_json::json!({"role": "user", "content": "Hi"}),
            serde_json::json!({"role": "assistant", "content": "Hello"}),
        ];
        let asked = |q: &str| {
            let mut messages = history.to_vec();
            messages.push(serde_json::json!({"role": "user", "content": q}));
            conversation(&messages)
        };
        let (one, two) = (asked("Question one"), asked("Question two"));
        let (first, _) = one.before_last().unwrap();
        let (edited, last) = two.before_last().unwrap();
        assert_eq!(first.key(), edited.key());
        assert_eq!(last, "Question two");
        // The state after the first question was answered is another key.
        let answered = asked("Question one").with_reply("An answer");
        assert_ne!(answered.key(), edited.key());
    }

    /// A session is never shared across a model, a folder (the CLI finds a
    /// session under its folder) or a system text.
    #[test]
    fn a_session_is_never_shared_across_model_folder_or_system() {
        let messages = [serde_json::json!({"role": "user", "content": "Hi"})];
        let folder = std::path::Path::new("/tmp");
        let base = Conversation::of("claude-haiku-5-5", folder, None, &messages);
        let others = [
            Conversation::of("claude-sonnet-5-5", folder, None, &messages),
            Conversation::of(
                "claude-haiku-5-5",
                std::path::Path::new("/srv"),
                None,
                &messages,
            ),
            Conversation::of("claude-haiku-5-5", folder, Some("Be brief"), &messages),
            Conversation::of("claude-haiku-5-5", folder, Some(""), &messages),
        ];
        for other in &others {
            assert_ne!(base.key(), other.key(), "{other:?}");
        }
    }

    /// Only the caller's own last message is ever sent alone: an assistant
    /// turn to continue, a system text, or a first message has no session
    /// before it.
    #[test]
    fn only_the_callers_own_last_message_is_sent_alone() {
        let lone = conversation(&[serde_json::json!({"role": "user", "content": "Hi"})]);
        assert!(lone.before_last().is_none());
        let continued = conversation(&[
            serde_json::json!({"role": "user", "content": "Hi"}),
            serde_json::json!({"role": "assistant", "content": "Hel"}),
        ]);
        assert!(continued.before_last().is_none());
        let system_last = conversation(&[
            serde_json::json!({"role": "user", "content": "Hi"}),
            serde_json::json!({"role": "system", "content": "Be brief"}),
        ]);
        assert!(system_last.before_last().is_none());
        // A message with no text is skipped, as `extract_prompt` skips it.
        let tool_result = conversation(&[
            serde_json::json!({"role": "user", "content": "Hi"}),
            serde_json::json!({"role": "assistant", "content": "Hello"}),
            serde_json::json!({"role": "user", "content": ""}),
        ]);
        assert!(tool_result.before_last().is_none());
    }

    /// A failed call stores nothing; a reply is stored for the next turn.
    #[test]
    fn a_failed_call_is_never_remembered() {
        let asked = conversation(&[serde_json::json!({
            "role": "user", "content": "a_failed_call_is_never_remembered"
        })]);
        let failed = serde_json::json!({
            "type": "result", "is_error": true, "result": "boom", "session_id": "s-failed"
        });
        remember_reply(&asked, &failed);
        assert_eq!(get_session(&asked.with_reply("boom").key()), None);
        let answered = serde_json::json!({
            "type": "result", "is_error": false, "result": "Fine", "session_id": "s-answered"
        });
        remember_reply(&asked, &answered);
        assert_eq!(
            get_session(&asked.with_reply("Fine").key()),
            Some("s-answered".to_string())
        );
    }

    #[cfg(unix)]
    async fn lines_of(output: &str) -> (tokio::process::Child, CliLines) {
        let mut child = tokio::process::Command::new("sh")
            .arg("-c")
            .arg("printf '%s' \"$0\"")
            .arg(output)
            .stdout(std::process::Stdio::piped())
            .spawn()
            .expect("sh");
        let stdout = child.stdout.take().unwrap();
        (
            child,
            CliLines::new(tokio::io::BufReader::new(stdout).lines()),
        )
    }

    /// #471: a refused resume is seen before the caller reads a line —
    /// stream-json's error result ahead of any turn, or json's silence — and
    /// every line read ahead still reaches the caller, in order.
    #[cfg(unix)]
    #[tokio::test]
    async fn a_refused_resume_is_seen_before_the_caller_reads() {
        let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(20);
        let refused = "{\"type\":\"system\",\"subtype\":\"init\"}\n\
                       {\"type\":\"result\",\"is_error\":true,\"num_turns\":0}\n";
        let (mut child, mut lines) = lines_of(refused).await;
        assert!(lines.resume_refused(deadline).await);
        let _ = child.wait().await;

        let (mut child, mut lines) = lines_of("").await;
        assert!(
            lines.resume_refused(deadline).await,
            "json's refusal: no output"
        );
        let _ = child.wait().await;

        // A turn that ran and then failed is the caller's error, not a refusal.
        let ran = "{\"type\":\"result\",\"is_error\":true,\"num_turns\":1}\n";
        let (mut child, mut lines) = lines_of(ran).await;
        assert!(!lines.resume_refused(deadline).await);
        let _ = child.wait().await;

        let answering = "{\"type\":\"system\",\"subtype\":\"init\"}\n\
                         {\"type\":\"stream_event\",\"event\":{}}\n\
                         {\"type\":\"result\",\"is_error\":false}\n";
        let (mut child, mut lines) = lines_of(answering).await;
        assert!(!lines.resume_refused(deadline).await);
        let mut seen = Vec::new();
        while let Some(line) = lines.next_line().await.unwrap() {
            seen.push(line);
        }
        assert_eq!(seen.len(), 3);
        assert!(seen[0].contains("init") && seen[1].contains("stream_event"));
        let _ = child.wait().await;
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
