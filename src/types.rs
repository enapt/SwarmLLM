//! Re-export all types from the swarmllm-types crate.
//!
//! Also re-exports extension traits for types that have methods
//! defined in the main crate.

pub use swarmllm_types::*;

/// Normalize a model name into a URL/ID-safe slug.
///
/// Used across API handlers (OpenAI, Anthropic, MCP, admin) to match
/// user-provided model names against registry entries.
/// Lowercase, spaces → dashes, strip non-alphanumeric (keep `-` and `.`).
pub fn slugify_model_name(name: &str) -> String {
    name.to_lowercase()
        .replace(' ', "-")
        .replace(|c: char| !c.is_alphanumeric() && c != '-' && c != '.', "")
}

/// Current time as Unix milliseconds.
#[inline]
pub fn unix_now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

/// Current time as Unix seconds.
#[inline]
pub fn unix_now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

// Extension traits for types defined in swarmllm-types
pub use crate::identity::nickname::NicknameRecordExt;
pub use crate::model::manifest::ModelManifestExt;
pub use crate::pool::types::BlindedPoolInvitationExt;
