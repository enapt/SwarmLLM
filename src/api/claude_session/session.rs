use std::path::PathBuf;
use std::time::Instant;

use serde::Serialize;
use tokio::io::{AsyncWriteExt, BufReader};
use tokio::sync::Mutex;

use crate::error::ApiError;

/// Maximum JSON buffer size (1MB) — matches the official SDK.
pub(super) const MAX_JSON_BUFFER: usize = 1024 * 1024;

/// Default idle timeout before a session subprocess is gracefully suspended.
pub(super) const DEFAULT_IDLE_TIMEOUT_SECS: u64 = 4 * 3600; // 4 hours
/// Timeout for the initial Claude CLI subprocess handshake.
pub(super) const CLAUDE_INIT_TIMEOUT_SECS: u64 = 120;
/// Grace period for each graceful-shutdown step in `suspend()` — used for the
/// initial stdin-EOF wait and the post-SIGTERM wait before escalating to SIGKILL.
const SUSPEND_GRACEFUL_TIMEOUT_SECS: u64 = 5;

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
    pub(super) child: Option<tokio::process::Child>,
    /// Stdin writer — behind its own lock to allow concurrent writes
    /// while the SSE loop holds the session lock for stdout reads.
    pub(super) stdin: StdinHandle,
    /// Buffered stdout reader for NDJSON events.
    pub(super) stdout: Option<tokio::io::Lines<BufReader<tokio::process::ChildStdout>>>,
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
    pub(super) exit_error: Option<String>,
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
pub(super) async fn write_to_stdin(
    handle: &StdinHandle,
    msg: &serde_json::Value,
) -> Result<(), ApiError> {
    let mut guard = handle.lock().await;
    let stdin = guard.as_mut().ok_or_else(|| {
        ApiError(crate::error::SwarmError::ServiceUnavailable(
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
        return Err(ApiError(crate::error::SwarmError::ServiceUnavailable(
            format!("Failed to write to subprocess stdin: {e}"),
        )));
    }
    if let Err(e) = stdin.flush().await {
        *guard = None;
        return Err(ApiError(crate::error::SwarmError::ServiceUnavailable(
            format!("Failed to flush subprocess stdin: {e}"),
        )));
    }
    Ok(())
}
