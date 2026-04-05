//! Claude Code Session Manager — long-lived bidirectional subprocess sessions.
//!
//! Manages persistent Claude CLI subprocesses using `--input-format stream-json`
//! for bidirectional NDJSON communication. Each chat session maps to a long-lived
//! subprocess that preserves project context, tool history, and conversation state.
//!
//! Feature-gated behind `claude-subscription`.

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

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Default idle timeout before a session subprocess is gracefully suspended.
const DEFAULT_IDLE_TIMEOUT_SECS: u64 = 4 * 3600; // 4 hours
/// Warning sent to frontend this many seconds before idle timeout.
const IDLE_WARNING_BEFORE_SECS: u64 = 15 * 60; // 15 minutes
/// Maximum concurrent active subprocesses (used by concurrency_limit fallback).
const _DEFAULT_MAX_ACTIVE_SESSIONS: usize = 3;

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
}

impl ClaudeSession {
    /// Check if the subprocess is still alive.
    pub fn is_alive(&self) -> bool {
        self.child.is_some()
            && self.state != SessionState::Suspended
            && self.state != SessionState::Expired
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

    /// Get a cloned handle to the stdin writer (for concurrent access).
    pub fn stdin_handle(&self) -> StdinHandle {
        self.stdin.clone()
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

    /// Send a permission response (allow/deny) to the subprocess.
    pub async fn send_permission_response(
        &mut self,
        request_id: &str,
        allow: bool,
        deny_message: Option<&str>,
    ) -> Result<(), ApiError> {
        let response = if allow {
            serde_json::json!({
                "type": "control_response",
                "request_id": request_id,
                "response": {
                    "behavior": "allow"
                }
            })
        } else {
            serde_json::json!({
                "type": "control_response",
                "request_id": request_id,
                "response": {
                    "behavior": "deny",
                    "message": deny_message.unwrap_or("User denied this action")
                }
            })
        };

        write_to_stdin(&self.stdin, &response).await?;
        self.touch();
        Ok(())
    }

    /// Read the next NDJSON event from stdout. Returns None on EOF.
    pub async fn read_event(&mut self) -> Option<serde_json::Value> {
        let stdout = self.stdout.as_mut()?;
        loop {
            match stdout.next_line().await {
                Ok(Some(line)) => {
                    let trimmed = line.trim();
                    if trimmed.is_empty() {
                        continue;
                    }
                    match serde_json::from_str(trimmed) {
                        Ok(val) => {
                            self.touch();
                            return Some(val);
                        }
                        Err(_) => continue, // skip non-JSON lines
                    }
                }
                Ok(None) => {
                    // EOF — subprocess exited unexpectedly
                    self.state = SessionState::Expired;
                    self.claude_session_id = None;
                    return None;
                }
                Err(_) => {
                    self.state = SessionState::Expired;
                    self.claude_session_id = None;
                    return None;
                }
            }
        }
    }

    /// Gracefully stop the subprocess (close stdin, wait for exit).
    pub async fn suspend(&mut self) {
        {
            let mut stdin_guard = self.stdin.lock().await;
            if let Some(stdin) = stdin_guard.take() {
                drop(stdin); // closes stdin → signals EOF to subprocess
            }
        }
        if let Some(mut child) = self.child.take() {
            // Give it a few seconds to exit gracefully
            let _ = tokio::time::timeout(std::time::Duration::from_secs(5), child.wait()).await;
            let _ = child.kill().await; // force kill if still alive
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
async fn write_to_stdin(handle: &StdinHandle, msg: &serde_json::Value) -> Result<(), ApiError> {
    let mut guard = handle.lock().await;
    let stdin = guard.as_mut().ok_or_else(|| {
        ApiError(crate::error::SwarmError::Internal(
            "Claude session: subprocess stdin not available".into(),
        ))
    })?;

    let mut line = serde_json::to_string(msg).map_err(|e| {
        ApiError(crate::error::SwarmError::Internal(format!(
            "Failed to serialize message: {e}"
        )))
    })?;
    line.push('\n');

    stdin.write_all(line.as_bytes()).await.map_err(|e| {
        ApiError(crate::error::SwarmError::Internal(format!(
            "Failed to write to subprocess stdin: {e}"
        )))
    })?;
    stdin.flush().await.map_err(|e| {
        ApiError(crate::error::SwarmError::Internal(format!(
            "Failed to flush subprocess stdin: {e}"
        )))
    })?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Session Manager
// ---------------------------------------------------------------------------

/// Global session manager — manages all active Claude Code sessions.
static SESSION_MANAGER: LazyLock<SessionManager> = LazyLock::new(SessionManager::new);

pub struct SessionManager {
    sessions: DashMap<String, std::sync::Arc<Mutex<ClaudeSession>>>,
}

impl SessionManager {
    fn new() -> Self {
        Self {
            sessions: DashMap::new(),
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
    ) -> Result<(), ApiError> {
        // Check concurrent limit
        if self.active_count() >= config.concurrency_limit() {
            return Err(ApiError(crate::error::SwarmError::Internal(
                "Too many active Claude Code sessions. Close or suspend one first.".into(),
            )));
        }

        // Remove existing session if present
        if let Some((_, old)) = self.sessions.remove(&session_id) {
            let mut old = old.lock().await;
            old.kill().await;
        }

        let binary = config.binary();
        let permission_mode = permission_mode.unwrap_or_else(|| "acceptEdits".to_string());
        let mut args = vec![
            "-p".to_string(),
            String::new(), // empty initial prompt — messages sent via stdin
            "--input-format".to_string(),
            "stream-json".to_string(),
            "--output-format".to_string(),
            "stream-json".to_string(),
            "--verbose".to_string(),
            // Skip slash commands and auto-discovered MCP servers for faster init.
            // We provide our MCP config explicitly via --mcp-config below.
            // Note: NOT using --bare because it skips OAuth (breaks subscription auth).
            "--disable-slash-commands".to_string(),
            "--model".to_string(),
            model.clone(),
            "--permission-mode".to_string(),
            permission_mode,
        ];

        // Connect SwarmLLM's MCP server so Claude can query other models
        if let Some(ref url) = mcp_url {
            let mcp_config = serde_json::json!({
                "mcpServers": {
                    "swarmllm": {
                        "type": "http",
                        "url": url
                    }
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

        tracing::info!(
            session_id = %session_id,
            model = %model,
            working_dir = %working_dir.display(),
            resume = ?resume_claude_id,
            "Creating Claude Code session"
        );

        let mut child = Command::new(binary)
            .args(&args)
            .current_dir(&working_dir)
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
        let session = ClaudeSession {
            id: session_id.clone(),
            claude_session_id: resume_claude_id,
            child: Some(child),
            stdin: std::sync::Arc::new(Mutex::new(Some(stdin))),
            stdout: Some(lines),
            working_dir,
            model,
            state: SessionState::Creating,
            created: now,
            last_active: now,
            tools: Vec::new(),
            idle_timeout_secs: config.timeout_secs.unwrap_or(DEFAULT_IDLE_TIMEOUT_SECS),
        };

        self.sessions
            .insert(session_id, std::sync::Arc::new(Mutex::new(session)));

        Ok(())
    }

    /// Get a session by ID.
    pub fn get_session(&self, session_id: &str) -> Option<std::sync::Arc<Mutex<ClaudeSession>>> {
        self.sessions.get(session_id).map(|e| e.value().clone())
    }

    /// Remove a session, killing its subprocess.
    pub async fn close_session(&self, session_id: &str) {
        if let Some((_, session)) = self.sessions.remove(session_id) {
            let mut s = session.lock().await;
            s.kill().await;
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

    /// Run periodic cleanup — suspend idle sessions, warn about upcoming timeouts.
    pub async fn cleanup_stale(&self, shared_state: &crate::daemon::state::SharedState) {
        let mut to_suspend = Vec::new();
        let mut to_warn = Vec::new();

        for entry in self.sessions.iter() {
            if let Ok(session) = entry.value().try_lock() {
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

        for id in to_warn {
            // Send idle warning via activity event (frontend can show a keepalive prompt)
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
    pub session_id: String,
    pub model: String,
    #[serde(default)]
    pub working_dir: Option<String>,
    /// If set, resume a previous CLI session instead of starting fresh.
    #[serde(default)]
    pub resume_claude_session_id: Option<String>,
    /// Permission mode: "default", "acceptEdits", "bypassPermissions", "plan".
    /// Default: "acceptEdits".
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

    let working_dir = req
        .working_dir
        .as_deref()
        .filter(|d| !d.is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(std::env::temp_dir);

    // Validate working directory exists
    if !working_dir.is_dir() {
        return Err(ApiError(crate::error::SwarmError::Validation(format!(
            "Working directory does not exist: {}",
            working_dir.display()
        ))));
    }

    // Build MCP URL for SwarmLLM's MCP server
    let listen_port = state.shared_state.config.node.listen_port;
    let mcp_url = format!("http://127.0.0.1:{listen_port}/mcp");

    SessionManager::global()
        .create_session(
            req.session_id.clone(),
            req.model,
            working_dir.clone(),
            req.resume_claude_session_id,
            Some(req.permission_mode),
            &sub_config,
            Some(mcp_url),
        )
        .await?;

    // Wait for init message to capture claude_session_id.
    // Take the stdout reader out of the session so we don't hold the session
    // lock during the potentially long init wait (hooks can take 30-60s+).
    let session_arc = SessionManager::global()
        .get_session(&req.session_id)
        .ok_or_else(|| {
            ApiError(crate::error::SwarmError::Internal(
                "Session created but not found".into(),
            ))
        })?;

    let mut stdout_reader = {
        let mut session = session_arc.lock().await;
        session.stdout.take()
    };

    let mut init_info = serde_json::json!({
        "status": "created",
        "session_id": req.session_id,
        "working_dir": working_dir.display().to_string(),
    });

    // Read events without the session lock held.
    // Hooks (SessionStart) can take 30-60s+, so use a generous timeout.
    let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(120);
    let init_result: Result<serde_json::Value, ApiError> = async {
        let reader = stdout_reader.as_mut().ok_or_else(|| {
            ApiError(crate::error::SwarmError::Internal(
                "Claude session: stdout not available".into(),
            ))
        })?;
        loop {
            let line = tokio::time::timeout_at(deadline, reader.next_line()).await;
            match line {
                Ok(Ok(Some(text))) => {
                    let trimmed = text.trim();
                    if trimmed.is_empty() {
                        continue;
                    }
                    let evt: serde_json::Value = match serde_json::from_str(trimmed) {
                        Ok(v) => v,
                        Err(_) => continue,
                    };
                    let evt_type = evt["type"].as_str().unwrap_or("");
                    let evt_subtype = evt["subtype"].as_str().unwrap_or("");
                    if evt_type == "system" && evt_subtype == "init" {
                        return Ok(evt);
                    }
                    // Skip hook messages, continue waiting for init
                }
                Ok(Ok(None)) => {
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
                    return Err(ApiError(crate::error::SwarmError::Internal(
                        "Timeout waiting for Claude CLI init (120s)".into(),
                    )));
                }
            }
        }
    }
    .await;

    // Put the stdout reader back and update session state
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
                init_info["status"] = serde_json::json!("active");
                let has_mcp = session.tools.iter().any(|t| t.contains("swarmllm"));
                init_info["mcp_connected"] = serde_json::json!(has_mcp);
            }
            Err(e) => {
                session.state = SessionState::Expired;
                // Kill the subprocess on init failure
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
            // TODO: auto-resume by re-creating with --resume
            return Err(ApiError(crate::error::SwarmError::Internal(
                "Session is suspended. Send a create request with resume_claude_session_id to resume.".into(),
            )));
        }
    }

    // Send the user message
    {
        let mut session = session_arc.lock().await;
        session.send_user_message(&req.content).await?;
    }

    // Stream events back as SSE
    let stream = async_stream::stream! {
        loop {
            let event = {
                let mut session = session_arc.lock().await;
                session.read_event().await
            };

            match event {
                Some(evt) => {
                    let evt_type = evt["type"].as_str().unwrap_or("");

                    // Forward the raw NDJSON event as an SSE data line
                    let data = serde_json::to_string(&evt).unwrap_or_default();
                    yield Ok::<_, std::io::Error>(
                        bytes::Bytes::from(format!("data: {}\n\n", data))
                    );

                    // Stop streaming on result event (turn complete)
                    if evt_type == "result" {
                        yield Ok(bytes::Bytes::from("data: [DONE]\n\n"));
                        break;
                    }

                    // control_request (permission prompt): keep the SSE stream open.
                    // The subprocess blocks waiting for the permission response on stdin.
                    // The frontend sends a concurrent POST to /permission which writes
                    // to stdin, the subprocess resumes, and events continue flowing
                    // through this still-open stream.
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
    axum::response::Response::builder()
        .header("content-type", "text/event-stream")
        .header("cache-control", "no-cache")
        .body(body)
        .map_err(|e| {
            ApiError(crate::error::SwarmError::Internal(format!(
                "Failed to build SSE response: {e}"
            )))
        })
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
    let session_arc = SessionManager::global()
        .get_session(&session_id)
        .ok_or_else(|| {
            ApiError(crate::error::SwarmError::Validation(format!(
                "No Claude Code session with id '{session_id}'"
            )))
        })?;

    // Grab the stdin handle without holding the session lock.
    let stdin_handle = {
        let session = session_arc.lock().await;
        session.stdin_handle()
    };

    let response = if req.allow {
        serde_json::json!({
            "type": "control_response",
            "request_id": req.request_id,
            "response": {
                "behavior": "allow"
            }
        })
    } else {
        serde_json::json!({
            "type": "control_response",
            "request_id": req.request_id,
            "response": {
                "behavior": "deny",
                "message": req.message.as_deref().unwrap_or("User denied this action")
            }
        })
    };

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
