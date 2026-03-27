/// Scrub API keys from an error body and truncate to 512 chars (char-boundary safe).
pub(crate) fn scrub_truncate_error(body: &str) -> String {
    let scrubbed = crate::crypto::scrub_api_keys(body);
    if scrubbed.len() > 512 {
        let mut idx = 512;
        while !scrubbed.is_char_boundary(idx) {
            idx -= 1;
        }
        format!("{}…[truncated]", &scrubbed[..idx])
    } else {
        scrubbed
    }
}

// Shared validation limits for API request parameters.
// Used by both openai.rs and anthropic.rs handlers.
pub(crate) const MAX_TOOLS: usize = 128;
pub(crate) const MAX_TOOL_NAME_LEN: usize = 256;
pub(crate) const MAX_TOOL_DESCRIPTION_LEN: usize = 4096;
pub(crate) const MAX_STOP_SEQUENCES: usize = 16;

pub mod admin;
pub mod admin_hf;
pub mod admin_models;
pub mod admin_providers;
pub mod anthropic;
pub mod identity;
pub mod mcp;
pub mod metrics;
pub mod middleware;
pub mod openai;
pub mod pool;
pub mod providers;
pub mod server;
pub mod websocket;
