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

/// BLAKE3 over the concatenation of `parts`, folded to a `u32` from the first
/// four bytes (little-endian).
///
/// This is the project's deterministic-placement primitive: consistent-hash
/// ring positions, replica-to-node assignment, and fair-share seed-shard
/// selection all need the same node/model/index inputs to map to the same
/// number on every node, or peers disagree about who should hold what.
///
/// Callers pass their inputs in order; BLAKE3's `update` is streaming, so
/// `hash_parts_to_u32(&[a, b])` is byte-identical to `update(a); update(b)`.
/// Changing the byte order, the number of bytes taken, or the endianness here
/// silently re-shuffles shard placement across the entire swarm — treat this
/// function as a wire format, not an implementation detail.
#[inline]
pub fn hash_parts_to_u32(parts: &[&[u8]]) -> u32 {
    let mut hasher = blake3::Hasher::new();
    for part in parts {
        hasher.update(part);
    }
    let hash = hasher.finalize();
    let b = hash.as_bytes();
    u32::from_le_bytes([b[0], b[1], b[2], b[3]])
}

// Extension traits for types defined in swarmllm-types
pub use crate::identity::nickname::NicknameRecordExt;
pub use crate::model::manifest::ModelManifestExt;
pub use crate::pool::types::BlindedPoolInvitationExt;
