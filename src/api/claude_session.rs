//! Claude Code Session Manager — long-lived bidirectional subprocess sessions.
//!
//! Manages persistent Claude CLI subprocesses using `--input-format stream-json`
//! for bidirectional NDJSON communication. Each chat session maps to a long-lived
//! subprocess that preserves project context, tool history, and conversation state.
//!
//! Feature-gated behind `claude-subscription`.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::LazyLock;
use std::time::Instant;

use axum::extract::State;
use axum::Json;
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::Command;
use tokio::sync::Mutex;

use crate::api::server::AppState;
use crate::error::ApiError;

/// Allowed permission modes.
const ALLOWED_PERMISSION_MODES: &[&str] = &[
    "default",
    "acceptEdits",
    "auto",
    "plan",
    "bypassPermissions",
    "dontAsk",
];

/// Maximum concurrent sessions (hard ceiling regardless of config).
const MAX_SESSIONS_HARD_LIMIT: usize = 20;

/// Maximum JSON buffer size (1MB) — matches the official SDK.
const MAX_JSON_BUFFER: usize = 1024 * 1024;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Default idle timeout before a session subprocess is gracefully suspended.
const DEFAULT_IDLE_TIMEOUT_SECS: u64 = 4 * 3600; // 4 hours
/// Timeout for the initial Claude CLI subprocess handshake.
const CLAUDE_INIT_TIMEOUT_SECS: u64 = 120;

/// Grace period for each graceful-shutdown step in `suspend()` — used for the
/// initial stdin-EOF wait and the post-SIGTERM wait before escalating to SIGKILL.
const SUSPEND_GRACEFUL_TIMEOUT_SECS: u64 = 5;
/// Warning sent to frontend this many seconds before idle timeout.
const IDLE_WARNING_BEFORE_SECS: u64 = 15 * 60; // 15 minutes

// ---------------------------------------------------------------------------
// Session state
// ---------------------------------------------------------------------------

/// Session lifecycle states.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SessionState {
    /// Subprocess is starting up (waiting for init message).
    Creating,
    /// Subprocess is alive and ready for messages.
    Active,
    /// Subprocess is alive but no recent activity.
    Idle,
    /// Subprocess was stopped gracefully — can be resumed via `--resume`.
    Suspended,
    /// Session is too old or CLI session file deleted — cannot resume.
    Expired,
}

/// Shared stdin handle — can be written to without holding the session lock.
/// This is critical for the permission flow: the SSE loop holds the session
/// lock while reading stdout, and the permission handler must write to stdin
/// concurrently.
pub type StdinHandle = std::sync::Arc<Mutex<Option<tokio::process::ChildStdin>>>;

/// A single Claude Code session backed by a live subprocess.
pub struct ClaudeSession {
    /// SwarmLLM session ID (from frontend chat.js).
    pub id: String,
    /// Claude CLI's internal session_id (from the `system/init` NDJSON event).
    pub claude_session_id: Option<String>,
    /// Subprocess handle.
    child: Option<tokio::process::Child>,
    /// Stdin writer — behind its own lock to allow concurrent writes
    /// while the SSE loop holds the session lock for stdout reads.
    stdin: StdinHandle,
    /// Buffered stdout reader for NDJSON events.
    stdout: Option<tokio::io::Lines<BufReader<tokio::process::ChildStdout>>>,
    /// Working directory the subprocess runs in.
    pub working_dir: PathBuf,
    /// Model used for this session.
    pub model: String,
    /// Current lifecycle state.
    pub state: SessionState,
    /// When the session was created.
    pub created: Instant,
    /// Last time a message was sent or received.
    pub last_active: Instant,
    /// Tools available (populated from init message).
    pub tools: Vec<String>,
    /// Configured idle timeout in seconds.
    pub idle_timeout_secs: u64,
    /// Tracks the last process exit error (for better diagnostics).
    exit_error: Option<String>,
}

impl ClaudeSession {
    /// Check if the subprocess is still alive.
    pub fn is_alive(&self) -> bool {
        if self.state == SessionState::Suspended || self.state == SessionState::Expired {
            return false;
        }
        // Check if child process has exited
        if let Some(ref child) = self.child {
            child.id().is_some() // None means already exited
        } else {
            false
        }
    }

    /// Touch the session to reset idle timer.
    pub fn touch(&mut self) {
        self.last_active = Instant::now();
        if self.state == SessionState::Idle {
            self.state = SessionState::Active;
        }
    }

    /// Seconds since last activity.
    pub fn idle_secs(&self) -> u64 {
        self.last_active.elapsed().as_secs()
    }

    /// Send a user message to the subprocess via stdin.
    pub async fn send_user_message(&mut self, content: &str) -> Result<(), ApiError> {
        let session_id = self.claude_session_id.clone().unwrap_or_default();
        let msg = serde_json::json!({
            "type": "user",
            "message": {
                "role": "user",
                "content": content
            },
            "parent_tool_use_id": null,
            "session_id": session_id
        });

        write_to_stdin(&self.stdin, &msg).await?;
        self.touch();
        Ok(())
    }

    /// Read the next NDJSON event from stdout. Returns None on EOF.
    ///
    /// Matches SDK behavior: skips non-JSON lines (e.g. `[SandboxDebug]`),
    /// accumulates partial JSON across lines with a 1MB buffer limit.
    pub async fn read_event(&mut self) -> Option<serde_json::Value> {
        let stdout = self.stdout.as_mut()?;
        let mut json_buffer = String::new();
        loop {
            match stdout.next_line().await {
                Ok(Some(line)) => {
                    let trimmed = line.trim();
                    if trimmed.is_empty() {
                        continue;
                    }
                    // Skip non-JSON lines when not mid-parse (matches SDK behavior).
                    // Lines like [SandboxDebug] would corrupt the buffer.
                    if json_buffer.is_empty() && !trimmed.starts_with('{') {
                        tracing::trace!("Skipping non-JSON line from CLI stdout");
                        continue;
                    }
                    // If mid-parse and a non-JSON line arrives, it would corrupt
                    // the buffer (e.g., [SandboxDebug] interleaved with JSON).
                    // Discard the partial buffer and skip the offending line.
                    if !json_buffer.is_empty()
                        && !trimmed.starts_with('{')
                        && !trimmed.starts_with('"')
                        && !trimmed.starts_with('[')
                        && !trimmed.starts_with('}')
                    {
                        tracing::warn!(
                            buffer_len = json_buffer.len(),
                            "Non-JSON line interleaved mid-parse, discarding partial buffer"
                        );
                        json_buffer.clear();
                        continue;
                    }
                    json_buffer.push_str(trimmed);
                    if json_buffer.len() > MAX_JSON_BUFFER {
                        tracing::warn!(
                            "JSON buffer exceeded {}B limit, discarding",
                            MAX_JSON_BUFFER
                        );
                        json_buffer.clear();
                        continue;
                    }
                    match serde_json::from_str(&json_buffer) {
                        Ok(val) => {
                            json_buffer.clear();
                            self.touch();
                            return Some(val);
                        }
                        Err(_) => {
                            // If the buffer started as a valid JSON beginning but
                            // this line looks like a fresh JSON object, the previous
                            // buffer was likely a truncated event. Start fresh.
                            if trimmed.starts_with('{') && json_buffer.len() > trimmed.len() {
                                if let Ok(val) = serde_json::from_str::<serde_json::Value>(trimmed)
                                {
                                    json_buffer.clear();
                                    self.touch();
                                    return Some(val);
                                }
                            }
                            // Speculatively accumulate — may be partial JSON
                            continue;
                        }
                    }
                }
                Ok(None) => {
                    // EOF — subprocess exited
                    self.state = SessionState::Expired;
                    self.exit_error = Some("Claude CLI subprocess exited (EOF on stdout)".into());
                    return None;
                }
                Err(e) => {
                    self.state = SessionState::Expired;
                    self.exit_error = Some(format!("Error reading CLI stdout: {e}"));
                    return None;
                }
            }
        }
    }

    /// Gracefully stop the subprocess.
    ///
    /// Matches the official SDK shutdown sequence:
    /// 1. Close stdin (signal EOF — lets CLI flush session files)
    /// 2. Wait 5s for graceful exit
    /// 3. SIGTERM (not available on all platforms, fall back to kill)
    /// 4. Wait 5s
    /// 5. SIGKILL (force)
    pub async fn suspend(&mut self) {
        // Step 1: close stdin
        {
            let mut stdin_guard = self.stdin.lock().await;
            if let Some(stdin) = stdin_guard.take() {
                drop(stdin);
            }
        }
        if let Some(mut child) = self.child.take() {
            // Step 2: wait SUSPEND_GRACEFUL_TIMEOUT_SECS for graceful exit after stdin EOF
            let grace = std::time::Duration::from_secs(SUSPEND_GRACEFUL_TIMEOUT_SECS);
            match tokio::time::timeout(grace, child.wait()).await {
                Ok(_) => {} // exited gracefully
                Err(_) => {
                    // Step 3: start_kill is async-safe (no subprocess spawn)
                    let _ = child.start_kill();
                    // Step 4: wait SUSPEND_GRACEFUL_TIMEOUT_SECS after SIGTERM
                    match tokio::time::timeout(grace, child.wait()).await {
                        Ok(_) => {}
                        Err(_) => {
                            // Step 5: force kill
                            let _ = child.kill().await;
                            let _ = child.wait().await;
                        }
                    }
                }
            }
        }
        self.stdout = None;
        self.state = SessionState::Suspended;
        tracing::info!(
            session_id = %self.id,
            claude_session_id = ?self.claude_session_id,
            "Claude session suspended"
        );
    }

    /// Kill the subprocess immediately.
    pub async fn kill(&mut self) {
        {
            let mut stdin_guard = self.stdin.lock().await;
            *stdin_guard = None;
        }
        self.stdout = None;
        if let Some(mut child) = self.child.take() {
            let _ = child.kill().await;
        }
        self.state = SessionState::Expired;
    }
}

/// Write a JSON message to the subprocess stdin handle.
///
/// Matches SDK behavior: acquires lock, checks readiness, writes + flushes.
/// On write failure, marks the handle as closed (takes it) so subsequent
/// writes fail immediately with a clear error.
async fn write_to_stdin(handle: &StdinHandle, msg: &serde_json::Value) -> Result<(), ApiError> {
    let mut guard = handle.lock().await;
    let stdin = guard.as_mut().ok_or_else(|| {
        ApiError(crate::error::SwarmError::Internal(
            "Claude session: subprocess stdin not available (process may have exited)".into(),
        ))
    })?;

    let mut line = serde_json::to_string(msg).map_err(|e| {
        ApiError(crate::error::SwarmError::Internal(format!(
            "Failed to serialize message: {e}"
        )))
    })?;
    line.push('\n');

    if let Err(e) = stdin.write_all(line.as_bytes()).await {
        // Mark stdin as dead so future writes fail immediately
        *guard = None;
        return Err(ApiError(crate::error::SwarmError::Internal(format!(
            "Failed to write to subprocess stdin: {e}"
        ))));
    }
    if let Err(e) = stdin.flush().await {
        *guard = None;
        return Err(ApiError(crate::error::SwarmError::Internal(format!(
            "Failed to flush subprocess stdin: {e}"
        ))));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Session Manager
// ---------------------------------------------------------------------------

/// Global session manager — manages all active Claude Code sessions.
static SESSION_MANAGER: LazyLock<SessionManager> = LazyLock::new(SessionManager::new);

pub struct SessionManager {
    sessions: DashMap<String, std::sync::Arc<Mutex<ClaudeSession>>>,
    /// Stdin handles indexed by session ID — accessible without locking the
    /// session mutex, which prevents deadlock when the SSE stream loop holds
    /// the session lock for stdout reads while the permission handler writes
    /// to stdin concurrently.
    stdin_handles: DashMap<String, StdinHandle>,
}

impl SessionManager {
    fn new() -> Self {
        Self {
            sessions: DashMap::new(),
            stdin_handles: DashMap::new(),
        }
    }

    /// Get the global session manager.
    pub fn global() -> &'static SessionManager {
        &SESSION_MANAGER
    }

    /// Number of active (non-suspended) sessions.
    pub fn active_count(&self) -> usize {
        self.sessions
            .iter()
            .filter(|e| {
                // Try to check state without blocking — if locked, count as active
                match e.value().try_lock() {
                    Ok(s) => s.is_alive(),
                    Err(_) => true,
                }
            })
            .count()
    }

    /// Create a new session by spawning a Claude CLI subprocess.
    pub async fn create_session(
        &self,
        session_id: String,
        model: String,
        working_dir: PathBuf,
        resume_claude_id: Option<String>,
        permission_mode: Option<String>,
        config: &super::claude_sub::ClaudeSubscriptionConfig,
        mcp_url: Option<String>,
        mcp_api_key: Option<String>,
    ) -> Result<(), ApiError> {
        // Check concurrent limit
        if self.active_count() >= config.concurrency_limit() {
            return Err(ApiError(crate::error::SwarmError::ServiceUnavailable(
                "Too many active Claude Code sessions. Close or suspend one first.".into(),
            )));
        }

        // Remove existing session if present
        if let Some((_, old)) = self.sessions.remove(&session_id) {
            self.stdin_handles.remove(&session_id);
            let mut old = old.lock().await;
            old.kill().await;
        }

        let binary = config.binary();
        let permission_mode = permission_mode.unwrap_or_else(|| "acceptEdits".to_string());
        let is_bypass = permission_mode == "bypassPermissions";
        // -p "" is required to trigger the CLI to start a session and emit system/init.
        // Without it, the CLI in --input-format stream-json mode just waits silently.
        // The SDK's new agent protocol uses control_request/initialize instead, but
        // -p "" is simpler and proven to work with subscription auth.
        let mut args = vec![
            "-p".to_string(),
            String::new(),
            "--output-format".to_string(),
            "stream-json".to_string(),
            "--verbose".to_string(),
            // Stream partial messages so frontend gets text deltas as they're generated
            // instead of waiting for complete assistant turns (prevents "Thinking..." stalls)
            "--include-partial-messages".to_string(),
            "--model".to_string(),
            model.clone(),
            "--permission-mode".to_string(),
            permission_mode,
        ];

        // bypassPermissions requires the explicit --dangerously-skip-permissions flag
        if is_bypass {
            args.push("--dangerously-skip-permissions".to_string());
        } else {
            // Route permission prompts through stdin/stdout so the frontend can
            // display approve/deny UI. Not needed for bypassPermissions since
            // no prompts are generated. Without this flag the CLI auto-denies in
            // non-interactive mode and control_request events are never emitted.
            args.push("--permission-prompt-tool".to_string());
            args.push("stdio".to_string());
        }

        // Connect SwarmLLM's MCP server so Claude can query other models
        if let Some(ref url) = mcp_url {
            let mut server_config = serde_json::json!({
                "type": "http",
                "url": url
            });
            // Add Bearer auth so the MCP server accepts our requests
            if let Some(ref key) = mcp_api_key {
                server_config["headers"] = serde_json::json!({
                    "Authorization": format!("Bearer {key}")
                });
            }
            let mcp_config = serde_json::json!({
                "mcpServers": {
                    "swarmllm": server_config
                }
            });
            args.push("--mcp-config".to_string());
            args.push(mcp_config.to_string());
        }

        // Resume from a previous CLI session if available
        if let Some(ref claude_id) = resume_claude_id {
            args.push("--resume".to_string());
            args.push(claude_id.clone());
        }

        // SDK adds --input-format last (always uses streaming mode)
        args.push("--input-format".to_string());
        args.push("stream-json".to_string());

        tracing::info!(
            session_id = %session_id,
            model = %model,
            working_dir = %working_dir.display(),
            resume = ?resume_claude_id,
            "Creating Claude Code session"
        );

        // Build environment matching the official SDK:
        // - Set CLAUDE_CODE_ENTRYPOINT to identify our spawned processes
        // - Filter out CLAUDECODE to prevent nested subprocess confusion
        let mut env: HashMap<String, String> = std::env::vars()
            .filter(|(k, _)| k != "CLAUDECODE")
            .collect();
        env.insert("CLAUDE_CODE_ENTRYPOINT".to_string(), "swarmllm".to_string());

        let mut child = Command::new(binary)
            .args(&args)
            .current_dir(&working_dir)
            .envs(&env)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .map_err(|e| {
                ApiError(crate::error::SwarmError::Internal(format!(
                    "Failed to spawn Claude CLI: {e}"
                )))
            })?;

        let stdin = child.stdin.take().ok_or_else(|| {
            ApiError(crate::error::SwarmError::Internal(
                "Claude session: no stdin".into(),
            ))
        })?;
        let stdout = child.stdout.take().ok_or_else(|| {
            ApiError(crate::error::SwarmError::Internal(
                "Claude session: no stdout".into(),
            ))
        })?;

        let reader = BufReader::new(stdout);
        let lines = reader.lines();

        let now = Instant::now();
        let stdin_handle: StdinHandle = std::sync::Arc::new(Mutex::new(Some(stdin)));
        let session = ClaudeSession {
            id: session_id.clone(),
            claude_session_id: resume_claude_id,
            child: Some(child),
            stdin: stdin_handle.clone(),
            stdout: Some(lines),
            working_dir,
            model,
            state: SessionState::Creating,
            created: now,
            last_active: now,
            tools: Vec::new(),
            idle_timeout_secs: config.timeout_secs.unwrap_or(DEFAULT_IDLE_TIMEOUT_SECS),
            exit_error: None,
        };

        self.stdin_handles.insert(session_id.clone(), stdin_handle);
        self.sessions
            .insert(session_id, std::sync::Arc::new(Mutex::new(session)));

        Ok(())
    }

    /// Get a session by ID.
    pub fn get_session(&self, session_id: &str) -> Option<std::sync::Arc<Mutex<ClaudeSession>>> {
        self.sessions.get(session_id).map(|e| e.value().clone())
    }

    /// Get a session's stdin handle by ID (no session lock needed).
    pub fn get_stdin_handle(&self, session_id: &str) -> Option<StdinHandle> {
        self.stdin_handles
            .get(session_id)
            .map(|e| e.value().clone())
    }

    /// Remove a session, killing its subprocess.
    pub async fn close_session(&self, session_id: &str) {
        self.stdin_handles.remove(session_id);
        if let Some((_, session)) = self.sessions.remove(session_id) {
            let mut s = session.lock().await;
            let working_dir = s.working_dir.clone();
            s.kill().await;
            drop(s);
            // Clean up temp directories created for quick chats
            let tmp_prefix = std::env::temp_dir().join("swarmllm-chat-");
            if working_dir.starts_with(&tmp_prefix.parent().unwrap_or(&tmp_prefix))
                && working_dir
                    .file_name()
                    .and_then(|n| n.to_str())
                    .map_or(false, |n| n.starts_with("swarmllm-chat-"))
            {
                let _ = std::fs::remove_dir_all(&working_dir);
                tracing::debug!(dir = %working_dir.display(), "Cleaned up temp session directory");
            }
        }
    }

    /// Suspend a session (graceful stop, preserving CLI session for resume).
    pub async fn suspend_session(&self, session_id: &str) {
        if let Some(session) = self.get_session(session_id) {
            let mut s = session.lock().await;
            s.suspend().await;
        }
    }

    /// List all sessions with their current state.
    pub fn list_sessions(&self) -> Vec<SessionInfo> {
        self.sessions
            .iter()
            .filter_map(|entry| {
                let session = entry.value().try_lock().ok()?;
                Some(SessionInfo {
                    id: session.id.clone(),
                    claude_session_id: session.claude_session_id.clone(),
                    model: session.model.clone(),
                    working_dir: session.working_dir.display().to_string(),
                    state: session.state,
                    created_secs_ago: session.created.elapsed().as_secs(),
                    idle_secs: session.idle_secs(),
                    tools_count: session.tools.len(),
                })
            })
            .collect()
    }

    /// Run periodic cleanup — suspend idle sessions, prune dead entries, warn about upcoming timeouts.
    pub async fn cleanup_stale(&self, shared_state: &crate::daemon::state::SharedState) {
        let mut to_suspend = Vec::new();
        let mut to_warn = Vec::new();
        let mut to_remove = Vec::new();

        for entry in self.sessions.iter() {
            if let Ok(session) = entry.value().try_lock() {
                if session.state == SessionState::Expired {
                    to_remove.push(session.id.clone());
                    continue;
                }
                if !session.is_alive() {
                    continue;
                }
                let idle = session.idle_secs();
                if idle >= session.idle_timeout_secs {
                    to_suspend.push(session.id.clone());
                } else if idle
                    >= session
                        .idle_timeout_secs
                        .saturating_sub(IDLE_WARNING_BEFORE_SECS)
                {
                    to_warn.push(session.id.clone());
                }
            }
        }

        // Remove expired/dead entries from the map
        for id in &to_remove {
            self.stdin_handles.remove(id.as_str());
            self.sessions.remove(id.as_str());
        }

        for id in to_warn {
            shared_state.emit_activity(
                crate::daemon::state::ActivityEvent::new(
                    "claude_code",
                    "idle_warning",
                    "Claude Code session will suspend soon due to inactivity".to_string(),
                )
                .with_toast("warning", 10000),
            );
            tracing::info!(session_id = %id, "Claude Code session idle warning");
        }

        for id in to_suspend {
            self.suspend_session(&id).await;
            shared_state.emit_activity(
                crate::daemon::state::ActivityEvent::new(
                    "claude_code",
                    "session_suspended",
                    "Claude Code session suspended (idle timeout) \u{2014} will resume on next message".to_string(),
                )
                .with_toast("info", 5000),
            );
        }
    }

    /// Shut down all active sessions — called during daemon shutdown.
    pub async fn shutdown_all(&self) {
        let ids: Vec<String> = self.sessions.iter().map(|e| e.key().clone()).collect();
        for id in ids {
            self.close_session(&id).await;
        }
        tracing::info!("All Claude Code sessions shut down");
    }
}

// ---------------------------------------------------------------------------
// API types
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize)]
pub struct SessionInfo {
    pub id: String,
    pub claude_session_id: Option<String>,
    pub model: String,
    pub working_dir: String,
    pub state: SessionState,
    pub created_secs_ago: u64,
    pub idle_secs: u64,
    pub tools_count: usize,
}

#[derive(Debug, Deserialize)]
pub struct CreateSessionRequest {
    /// Client-provided session ID. If omitted, the server generates a UUID.
    #[serde(default)]
    pub session_id: Option<String>,
    pub model: String,
    #[serde(default)]
    pub working_dir: Option<String>,
    /// If set, resume a previous CLI session instead of starting fresh.
    #[serde(default)]
    pub resume_claude_session_id: Option<String>,
    /// Permission mode: "default", "acceptEdits", "auto", "plan", "bypassPermissions", "dontAsk".
    /// Default: "acceptEdits". All modes require a valid API key.
    #[serde(default = "default_permission_mode")]
    pub permission_mode: String,
}

fn default_permission_mode() -> String {
    "acceptEdits".to_string()
}

#[derive(Debug, Deserialize)]
pub struct SendMessageRequest {
    pub content: String,
}

#[derive(Debug, Deserialize)]
pub struct PermissionRequest {
    pub request_id: String,
    pub allow: bool,
    #[serde(default)]
    pub message: Option<String>,
    /// Optional modified input to pass back when allowing (SDK `updatedInput`).
    #[serde(default)]
    pub updated_input: Option<serde_json::Value>,
}

// ---------------------------------------------------------------------------
// API handlers
// ---------------------------------------------------------------------------

/// POST /api/claude-code/session — Create a new Claude Code session.
pub async fn create_session_handler(
    State(state): State<AppState>,
    Json(req): Json<CreateSessionRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let config = state.shared_state.metrics.providers_config.read().await;
    let sub_config = config
        .claude_subscription
        .as_ref()
        .ok_or_else(|| {
            ApiError(crate::error::SwarmError::Validation(
                "Claude subscription not configured".into(),
            ))
        })?
        .clone();
    drop(config);

    if !sub_config.enabled {
        return Err(ApiError(crate::error::SwarmError::Validation(
            "Claude subscription is not enabled".into(),
        )));
    }

    // --- Security: validate permission_mode against allowlist ---
    if !ALLOWED_PERMISSION_MODES.contains(&req.permission_mode.as_str()) {
        return Err(ApiError(crate::error::SwarmError::Validation(format!(
            "Invalid permission_mode '{}'. Allowed: {}",
            req.permission_mode,
            ALLOWED_PERMISSION_MODES.join(", ")
        ))));
    }

    // --- Security: validate model name format ---
    if req.model.len() > 128
        || !req
            .model
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '.' || c == '_' || c == ':')
    {
        return Err(ApiError(crate::error::SwarmError::Validation(
            "Invalid model name: must be ≤128 chars, alphanumeric with -._: only".into(),
        )));
    }

    // Generate or validate session_id
    let session_id = if let Some(ref id) = req.session_id {
        // Client-provided: validate format (prevent path traversal)
        if id.len() > 128 || id.contains('/') || id.contains('\\') || id.contains("..") {
            return Err(ApiError(crate::error::SwarmError::Validation(
                "Invalid session_id: must be ≤128 chars, no path separators".into(),
            )));
        }
        id.clone()
    } else {
        // Server-generated UUID — prevents session ID guessing
        uuid::Uuid::new_v4().to_string()
    };

    // --- Security: validate resume_claude_session_id as UUID ---
    if let Some(ref resume_id) = req.resume_claude_session_id {
        if uuid::Uuid::parse_str(resume_id).is_err() {
            return Err(ApiError(crate::error::SwarmError::Validation(
                "resume_claude_session_id must be a valid UUID".into(),
            )));
        }
    }

    // --- Security: enforce hard session limit ---
    if SessionManager::global().sessions.len() >= MAX_SESSIONS_HARD_LIMIT {
        return Err(ApiError(crate::error::SwarmError::Validation(
            "Maximum session limit reached. Close existing sessions first.".into(),
        )));
    }

    let working_dir = if let Some(dir) = req.working_dir.as_deref().filter(|d| !d.is_empty()) {
        let p = PathBuf::from(dir);
        if !p.is_dir() {
            return Err(ApiError(crate::error::SwarmError::Validation(format!(
                "Working directory does not exist: {}",
                p.display()
            ))));
        }
        // --- Security: restrict working_dir to user home subtree ---
        // Reject system directories, root, and paths outside home.
        let canonical = p.canonicalize().unwrap_or_else(|_| p.clone());
        let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("/nonexistent"));
        let tmp = std::env::temp_dir();
        if !canonical.starts_with(&home) && !canonical.starts_with(&tmp) {
            return Err(ApiError(crate::error::SwarmError::Validation(
                "working_dir must be under the user's home directory or temp directory".into(),
            )));
        }
        p
    } else {
        // Create a unique temp directory per session to avoid collisions
        let dir = std::env::temp_dir().join(format!("swarmllm-chat-{}", &session_id));
        if !dir.exists() {
            std::fs::create_dir_all(&dir).map_err(|e| {
                ApiError(crate::error::SwarmError::Internal(format!(
                    "Failed to create temp directory: {e}"
                )))
            })?;
        }
        dir
    };

    // Build MCP URL + auth for SwarmLLM's MCP server
    let listen_port = state.shared_state.config.node.listen_port;
    let mcp_url = format!("http://127.0.0.1:{listen_port}/mcp");
    let mcp_api_key = state.shared_state.api_key.clone();

    SessionManager::global()
        .create_session(
            session_id.clone(),
            req.model,
            working_dir.clone(),
            req.resume_claude_session_id,
            Some(req.permission_mode),
            &sub_config,
            Some(mcp_url),
            Some(mcp_api_key),
        )
        .await?;

    // The CLI with -p "" + --input-format stream-json waits for a stdin message
    // before emitting system/init. Write an empty user message to trigger init.
    let session_arc = SessionManager::global()
        .get_session(&session_id)
        .ok_or_else(|| {
            ApiError(crate::error::SwarmError::Internal(
                "Session created but not found".into(),
            ))
        })?;

    // Write empty user message to trigger CLI initialization
    {
        let session = session_arc.lock().await;
        let trigger_msg = serde_json::json!({
            "type": "user",
            "message": { "role": "user", "content": "" },
            "session_id": "",
            "parent_tool_use_id": null
        });
        write_to_stdin(&session.stdin, &trigger_msg).await?;
    }

    // Take stdout reader out of session (avoids holding lock during init wait)
    let mut stdout_reader = {
        let mut session = session_arc.lock().await;
        session.stdout.take()
    };

    let mut init_info = serde_json::json!({
        "status": "created",
        "session_id": session_id,
        "working_dir": working_dir.display().to_string(),
    });

    // Read events until we get system/init, then drain through the `result` event
    // from the empty -p "" prompt. Without this drain, the buffered result event
    // would be read by the first SSE stream, which would interpret it as the
    // user's response and immediately close with [DONE].
    let deadline =
        tokio::time::Instant::now() + std::time::Duration::from_secs(CLAUDE_INIT_TIMEOUT_SECS);
    let init_result: Result<serde_json::Value, ApiError> = async {
        let reader = stdout_reader.as_mut().ok_or_else(|| {
            ApiError(crate::error::SwarmError::Internal(
                "Claude session: stdout not available".into(),
            ))
        })?;
        let mut init_evt: Option<serde_json::Value> = None;
        loop {
            let line = tokio::time::timeout_at(deadline, reader.next_line()).await;
            match line {
                Ok(Ok(Some(text))) => {
                    let trimmed = text.trim();
                    if trimmed.is_empty() {
                        continue;
                    }
                    if !trimmed.starts_with('{') {
                        continue;
                    }
                    let evt: serde_json::Value = match serde_json::from_str(trimmed) {
                        Ok(v) => v,
                        Err(_) => continue,
                    };
                    let evt_type = evt["type"].as_str().unwrap_or("");
                    let evt_subtype = evt["subtype"].as_str().unwrap_or("");

                    if evt_type == "system" && evt_subtype == "init" {
                        init_evt = Some(evt);
                        // Don't return yet — keep draining until we consume the
                        // `result` event from the empty -p "" prompt.
                        continue;
                    }

                    if init_evt.is_some() && evt_type == "result" {
                        // Consumed the empty prompt's result. Session is now
                        // fully idle and ready for user messages.
                        tracing::debug!("Drained empty-prompt result event during init");
                        return Ok(init_evt.unwrap());
                    }
                    // Skip hook messages, assistant messages from empty prompt, etc.
                }
                Ok(Ok(None)) => {
                    // EOF — if we got init, proceed (CLI may have exited after empty prompt)
                    if let Some(evt) = init_evt {
                        return Ok(evt);
                    }
                    return Err(ApiError(crate::error::SwarmError::Internal(
                        "Claude CLI exited before sending init message".into(),
                    )));
                }
                Ok(Err(e)) => {
                    return Err(ApiError(crate::error::SwarmError::Internal(format!(
                        "Error reading Claude CLI stdout: {e}"
                    ))));
                }
                Err(_) => {
                    // Timeout — if we have init, proceed (result may not come)
                    if let Some(evt) = init_evt {
                        tracing::warn!(
                            "Timed out waiting for empty-prompt result, proceeding with init"
                        );
                        return Ok(evt);
                    }
                    return Err(ApiError(crate::error::SwarmError::Internal(
                        format!(
                            "Timeout waiting for Claude CLI init ({}s)",
                            CLAUDE_INIT_TIMEOUT_SECS
                        )
                        .into(),
                    )));
                }
            }
        }
    }
    .await;

    // Step 4: Put stdout reader back and update session state
    {
        let mut session = session_arc.lock().await;
        session.stdout = stdout_reader;

        match init_result {
            Ok(evt) => {
                let cli_session_id = evt["session_id"].as_str().map(String::from);
                session.claude_session_id = cli_session_id.clone();
                session.state = SessionState::Active;

                if let Some(tools) = evt["tools"].as_array() {
                    session.tools = tools
                        .iter()
                        .filter_map(|t| t.as_str().map(String::from))
                        .collect();
                }

                init_info["claude_session_id"] = serde_json::json!(cli_session_id);
                init_info["tools"] = serde_json::json!(session.tools);
                init_info["model"] = serde_json::json!(evt["model"]);
                init_info["slash_commands"] = evt["slash_commands"].clone();
                init_info["status"] = serde_json::json!("active");
                let has_mcp = session.tools.iter().any(|t| t.contains("swarmllm"));
                init_info["mcp_connected"] = serde_json::json!(has_mcp);
            }
            Err(e) => {
                session.state = SessionState::Expired;
                session.kill().await;
                return Err(e);
            }
        }
    }

    // Emit activity event
    state.shared_state.emit_activity(
        crate::daemon::state::ActivityEvent::new(
            "claude_code",
            "session_created",
            format!("Claude Code session started in {}", working_dir.display()),
        )
        .with_toast("info", 3000),
    );

    Ok(Json(init_info))
}

/// POST /api/claude-code/session/:id/message — Send a message, returns SSE stream.
pub async fn send_message_handler(
    State(_state): State<AppState>,
    axum::extract::Path(session_id): axum::extract::Path<String>,
    Json(req): Json<SendMessageRequest>,
) -> Result<axum::response::Response, ApiError> {
    let session_arc = SessionManager::global()
        .get_session(&session_id)
        .ok_or_else(|| {
            ApiError(crate::error::SwarmError::Validation(format!(
                "No Claude Code session with id '{session_id}'"
            )))
        })?;

    // Check if session needs to be resumed
    {
        let session = session_arc.lock().await;
        if session.state == SessionState::Suspended {
            drop(session);
            return Err(ApiError(crate::error::SwarmError::Validation(
                "Session is suspended. Send a create request with resume_claude_session_id to resume.".into(),
            )));
        }
    }

    // Send the user message
    {
        let mut session = session_arc.lock().await;
        session.send_user_message(&req.content).await?;
    }

    // Stream events back as SSE.
    // The SDK spec confirms: exactly one `result` event per query.
    // Break immediately on `result` — it's always the final event.
    let stream = async_stream::stream! {
        loop {
            let event = {
                let mut session = session_arc.lock().await;
                session.read_event().await
            };

            match event {
                Some(evt) => {
                    let evt_type = evt["type"].as_str().unwrap_or("");

                    // Log control_request events for debugging permission flow
                    if evt_type == "control_request" {
                        tracing::info!(
                            target: "swarmllm::api::claude_session",
                            "Control request received: {}",
                            serde_json::to_string(&evt).unwrap_or_default()
                        );
                    }

                    // Forward the raw NDJSON event as an SSE data line
                    let data = serde_json::to_string(&evt).unwrap_or_default();
                    yield Ok::<_, std::io::Error>(
                        bytes::Bytes::from(format!("data: {}\n\n", data))
                    );

                    // result = query complete (one per query, always final)
                    if evt_type == "result" {
                        yield Ok(bytes::Bytes::from("data: [DONE]\n\n"));
                        break;
                    }
                    // control_request (permission prompt): keep the SSE stream open.
                }
                None => {
                    // EOF — subprocess exited
                    let err = serde_json::json!({
                        "type": "error",
                        "message": "Claude CLI subprocess exited unexpectedly"
                    });
                    yield Ok(bytes::Bytes::from(format!("data: {}\n\n", err)));
                    yield Ok(bytes::Bytes::from("data: [DONE]\n\n"));
                    break;
                }
            }
        }
    };

    let body = axum::body::Body::from_stream(stream);
    super::providers::build_sse_response(body)
}

/// POST /api/claude-code/session/:id/permission — Respond to a tool permission prompt.
///
/// Uses the stdin handle directly (not the session lock) to avoid deadlock:
/// the SSE loop holds the session lock while reading stdout, and this handler
/// must write to stdin concurrently.
pub async fn permission_handler(
    axum::extract::Path(session_id): axum::extract::Path<String>,
    Json(req): Json<PermissionRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    // Get the stdin handle directly — no session lock needed.
    // This avoids a deadlock: the SSE stream loop holds the session lock
    // while blocking on read_event(), so locking the session here would
    // block forever.
    let stdin_handle = SessionManager::global()
        .get_stdin_handle(&session_id)
        .ok_or_else(|| {
            ApiError(crate::error::SwarmError::Validation(format!(
                "No Claude Code session with id '{session_id}'"
            )))
        })?;

    // SDK protocol: response envelope wraps request_id + subtype inside .response.
    // When allowing, updatedInput passes through the (possibly modified) tool input.
    let inner = if req.allow {
        let mut allow_obj = serde_json::json!({ "behavior": "allow" });
        if let Some(updated) = &req.updated_input {
            allow_obj["updatedInput"] = updated.clone();
        }
        allow_obj
    } else {
        serde_json::json!({
            "behavior": "deny",
            "message": req.message.as_deref().unwrap_or("User denied this action")
        })
    };
    let response = serde_json::json!({
        "type": "control_response",
        "response": {
            "subtype": "success",
            "request_id": req.request_id,
            "response": inner
        }
    });

    write_to_stdin(&stdin_handle, &response).await?;

    Ok(Json(serde_json::json!({"status": "ok"})))
}

/// DELETE /api/claude-code/session/:id — Close and destroy a session.
pub async fn close_session_handler(
    axum::extract::Path(session_id): axum::extract::Path<String>,
) -> Json<serde_json::Value> {
    SessionManager::global().close_session(&session_id).await;
    Json(serde_json::json!({"status": "closed"}))
}

/// GET /api/claude-code/session/:id — Get session info.
pub async fn get_session_handler(
    axum::extract::Path(session_id): axum::extract::Path<String>,
) -> Result<Json<SessionInfo>, ApiError> {
    let session_arc = SessionManager::global()
        .get_session(&session_id)
        .ok_or_else(|| {
            ApiError(crate::error::SwarmError::Validation(format!(
                "No Claude Code session with id '{session_id}'"
            )))
        })?;

    let session = session_arc.lock().await;
    Ok(Json(SessionInfo {
        id: session.id.clone(),
        claude_session_id: session.claude_session_id.clone(),
        model: session.model.clone(),
        working_dir: session.working_dir.display().to_string(),
        state: session.state,
        created_secs_ago: session.created.elapsed().as_secs(),
        idle_secs: session.idle_secs(),
        tools_count: session.tools.len(),
    }))
}

/// GET /api/claude-code/sessions — List all sessions.
pub async fn list_sessions_handler() -> Json<serde_json::Value> {
    let sessions = SessionManager::global().list_sessions();
    Json(serde_json::json!({ "sessions": sessions }))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_session_state_serialization() {
        let json = serde_json::to_string(&SessionState::Active).unwrap();
        assert_eq!(json, "\"active\"");
        let json = serde_json::to_string(&SessionState::Suspended).unwrap();
        assert_eq!(json, "\"suspended\"");
    }

    #[test]
    fn test_session_manager_global() {
        let mgr = SessionManager::global();
        assert_eq!(mgr.active_count(), 0);
        assert!(mgr.list_sessions().is_empty());
    }
}
