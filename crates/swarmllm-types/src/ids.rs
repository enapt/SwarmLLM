//! Newtype identifiers used throughout SwarmLLM.

use serde::{Deserialize, Serialize};
use std::fmt;

/// 32-byte BLAKE3 digest.
pub type Blake3Hash = [u8; 32];

/// Wrapper around Ed25519 public key. This IS the node's identity.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct NodeId(pub [u8; 32]);

impl fmt::Display for NodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", hex::encode(&self.0[..8]))
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ModelId(pub String);

impl fmt::Display for ModelId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // SEC: Sanitize control characters to prevent log injection via network-supplied model IDs.
        // Newlines, carriage returns, and null bytes are replaced with their escape sequences.
        for ch in self.0.chars() {
            match ch {
                '\n' => write!(f, "\\n")?,
                '\r' => write!(f, "\\r")?,
                '\0' => write!(f, "\\0")?,
                c if c.is_control() => write!(f, "\\x{:02x}", c as u32)?,
                c => write!(f, "{c}")?,
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct ShardId {
    pub model_id: ModelId,
    pub index: u32,
}

/// Sentinel shard index for mmproj (vision encoder) files.
/// Using u32::MAX avoids collisions with text model shard indices (0..N).
pub const MMPROJ_SHARD_INDEX: u32 = u32::MAX;

impl ShardId {
    /// Whether this shard ID refers to a mmproj (vision encoder) file.
    pub fn is_mmproj(&self) -> bool {
        self.index == MMPROJ_SHARD_INDEX
    }

    /// Human-friendly display label for a shard index (1-based, or "mmproj").
    pub fn display_index(index: u32) -> String {
        if index == MMPROJ_SHARD_INDEX {
            "mmproj".to_string()
        } else {
            format!("shard {}", index + 1)
        }
    }

    /// Safe 1-based index for display, returning "mmproj" for MMPROJ_SHARD_INDEX.
    /// Use this instead of `index + 1` to avoid u32 overflow on mmproj shards.
    pub fn display_index_short(index: u32) -> String {
        if index == MMPROJ_SHARD_INDEX {
            "mmproj".to_string()
        } else {
            format!("{}", index + 1)
        }
    }

    /// Create a ShardId for the mmproj of a given model.
    pub fn mmproj_for(model_id: ModelId) -> Self {
        Self {
            model_id,
            index: MMPROJ_SHARD_INDEX,
        }
    }
}
