//! Node identity and privacy types shared across the crate.

use serde::{Deserialize, Serialize};

use crate::ids::NodeId;

/// A sealed (encrypted) inference prompt for E2E privacy.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SealedPrompt {
    pub request_id: uuid::Uuid,
    /// ChaCha20-Poly1305 encrypted prompt bytes.
    pub encrypted_prompt: Vec<u8>,
    /// 12-byte nonce used for prompt encryption.
    pub nonce: [u8; 12],
    /// Ephemeral X25519 public key (32 bytes) of the sealer.
    pub ephemeral_pub: [u8; 32],
    /// The request_key encrypted for the first pipeline node's X25519 key.
    pub key_envelope: Vec<u8>,
}

/// A signed nickname record for a node.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NicknameRecord {
    pub node_id: NodeId,
    pub nickname: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Ed25519 signature over the signing payload.
    pub signature: Vec<u8>,
}

/// Nickname announcement gossiped across the network.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NicknameGossip {
    pub record: NicknameRecord,
}
