//! Claude Code Session Manager — long-lived bidirectional subprocess sessions.
//!
//! Manages persistent Claude CLI subprocesses using `--input-format stream-json`
//! for bidirectional NDJSON communication. Each chat session maps to a long-lived
//! subprocess that preserves project context, tool history, and conversation state.
//!
//! Feature-gated behind `claude-subscription`.

mod handlers;
mod manager;
mod session;

pub use handlers::{
    close_session_handler, create_session_handler, get_session_handler, list_sessions_handler,
    permission_handler, send_message_handler, CreateSessionRequest, PermissionRequest,
    SendMessageRequest,
};
pub use manager::{SessionInfo, SessionManager};
pub use session::{ClaudeSession, SessionState, StdinHandle};

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
