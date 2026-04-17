use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::LazyLock;
use std::time::Instant;

use dashmap::DashMap;
use serde::Serialize;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command;
use tokio::sync::Mutex;

use crate::error::ApiError;

use super::session::{ClaudeSession, SessionState, StdinHandle, DEFAULT_IDLE_TIMEOUT_SECS};

/// Warning sent to frontend this many seconds before idle timeout.
const IDLE_WARNING_BEFORE_SECS: u64 = 15 * 60; // 15 minutes

/// Maximum concurrent sessions (hard ceiling regardless of config).
pub(super) const MAX_SESSIONS_HARD_LIMIT: usize = 20;

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

    /// Total session count (active + suspended).
    pub(super) fn sessions_len(&self) -> usize {
        self.sessions.len()
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
    #[allow(clippy::too_many_arguments)]
    pub async fn create_session(
        &self,
        session_id: String,
        model: String,
        working_dir: PathBuf,
        resume_claude_id: Option<String>,
        permission_mode: Option<String>,
        config: &crate::api::claude_sub::ClaudeSubscriptionConfig,
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
            if working_dir.starts_with(tmp_prefix.parent().unwrap_or(&tmp_prefix))
                && working_dir
                    .file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n.starts_with("swarmllm-chat-"))
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
            let short = &id[..8.min(id.len())];
            shared_state.emit_activity(
                crate::daemon::state::ActivityEvent::new(
                    "claude_code",
                    "idle_warning",
                    format!(
                        "Claude Code session {} will suspend soon due to inactivity",
                        short
                    ),
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
