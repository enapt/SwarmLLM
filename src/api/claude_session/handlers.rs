use std::path::PathBuf;

use axum::extract::State;
use axum::Json;
use serde::Deserialize;

use crate::api::server::AppState;
use crate::error::ApiError;

use super::manager::{SessionInfo, SessionManager, MAX_SESSIONS_HARD_LIMIT};
use super::session::{write_to_stdin, SessionState, CLAUDE_INIT_TIMEOUT_SECS};

/// Allowed permission modes.
const ALLOWED_PERMISSION_MODES: &[&str] = &[
    "default",
    "acceptEdits",
    "auto",
    "plan",
    "bypassPermissions",
    "dontAsk",
];

// ---------------------------------------------------------------------------
// API types
// ---------------------------------------------------------------------------

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
    if SessionManager::global().sessions_len() >= MAX_SESSIONS_HARD_LIMIT {
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
                ApiError(crate::error::SwarmError::ServiceUnavailable(format!(
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

                    if let Some(evt) = init_evt.take() {
                        if evt_type == "result" {
                            // Consumed the empty prompt's result. Session is now
                            // fully idle and ready for user messages.
                            tracing::debug!("Drained empty-prompt result event during init");
                            return Ok(evt);
                        }
                        init_evt = Some(evt);
                    }
                    // Skip hook messages, assistant messages from empty prompt, etc.
                }
                Ok(Ok(None)) => {
                    // EOF — if we got init, proceed (CLI may have exited after empty prompt)
                    if let Some(evt) = init_evt {
                        return Ok(evt);
                    }
                    return Err(ApiError(crate::error::SwarmError::ServiceUnavailable(
                        "Claude CLI exited before sending init message".into(),
                    )));
                }
                Ok(Err(e)) => {
                    return Err(ApiError(crate::error::SwarmError::ServiceUnavailable(
                        format!("Error reading Claude CLI stdout: {e}"),
                    )));
                }
                Err(_) => {
                    // Timeout — if we have init, proceed (result may not come)
                    if let Some(evt) = init_evt {
                        tracing::warn!(
                            "Timed out waiting for empty-prompt result, proceeding with init"
                        );
                        return Ok(evt);
                    }
                    return Err(ApiError(crate::error::SwarmError::ServiceUnavailable(
                        format!(
                            "Timeout waiting for Claude CLI init ({}s)",
                            CLAUDE_INIT_TIMEOUT_SECS
                        ),
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
    // Per-field size cap — the global 32 MiB body limit exists for VLM image
    // payloads and is far too generous for a text message written to the
    // Claude CLI subprocess stdin. A multi-MB message blocks the stdin writer
    // or OOMs the CLI; 1 MiB is well above any realistic prompt.
    const MAX_MESSAGE_BYTES: usize = 1_000_000;
    if req.content.len() > MAX_MESSAGE_BYTES {
        return Err(ApiError(crate::error::SwarmError::Validation(format!(
            "Message content too large: {} bytes (max {} bytes)",
            req.content.len(),
            MAX_MESSAGE_BYTES
        ))));
    }

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
                    yield Ok::<_, std::io::Error>(crate::api::sse::data_frame(&evt));

                    // result = query complete (one per query, always final)
                    if evt_type == "result" {
                        yield Ok(crate::api::sse::done_frame());
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
                    yield Ok(crate::api::sse::data_frame(&err));
                    yield Ok(crate::api::sse::done_frame());
                    break;
                }
            }
        }
    };

    let body = axum::body::Body::from_stream(stream);
    crate::api::providers::build_sse_response(body)
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
