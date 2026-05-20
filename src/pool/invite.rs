//! Pool invite code v2 — `swarmpool://...` shareable bootstrap blob.
//!
//! The original 8-character invite code (still kept inside this blob as the
//! actual join token) only worked once both nodes were already on the same
//! libp2p swarm — it carried no information about how to *find* the inviter.
//! In a fully decentralized network that's fine (DHT bootstrap), but during
//! the bootstrap-before-decentralization phase the pool owner is often the
//! only address the joiner can reach. v2 fixes that by bundling the owner's
//! reachable listen multiaddrs alongside the join token, so a fresh node
//! anywhere with the code can:
//!
//! 1. Decode it.
//! 2. Dial one of the owner's multiaddrs over the wire (Tailscale, LAN,
//!    public IP, whatever's in the bundle).
//! 3. Once a libp2p connection lands, broadcast the existing
//!    `PoolMessage::JoinRequest { code_hash, ... }` over GossipSub.
//!
//! The owner's handler still matches `code_hash` against its
//! `invite_codes` map exactly as before — there is **no wire-protocol
//! change** for the pool-join itself. v2 is purely the rendezvous wrapper.
//!
//! ## Wire format
//!
//! - prefix: `swarmpool://`
//! - payload: base64url(no-pad) of `key (32B) || nonce (12B) || ciphertext`
//! - ciphertext: ChaCha20-Poly1305 of `serde_json::to_vec(&InviteCodePayload)`
//!
//! Encryption is here to prevent casual IP harvesting from anyone glancing
//! at a code pasted into a chat window. The key is embedded in the code, so
//! anyone with the full code can decrypt — same threat model as the existing
//! `swarm://` network code in `src/network/discovery.rs`.

use base64::Engine;
use chacha20poly1305::aead::{Aead, KeyInit, OsRng};
use chacha20poly1305::{ChaCha20Poly1305, Nonce};
use rand::RngCore;
use serde::{Deserialize, Serialize};

use crate::error::SwarmError;
use crate::types::NodeId;

/// `swarmpool://...` is the v2 prefix. Separate from `swarm://` (network-only
/// invite) so callers can tell at a glance which flow a pasted code drives.
pub const INVITE_PREFIX: &str = "swarmpool://";

/// Current wire-format version. Bump when changing the inner payload.
pub const INVITE_VERSION: u8 = 1;

/// Hard cap on the encoded length we'll attempt to decode. A typical blob is
/// ~300-500 chars; anything bigger is either garbage or an attempted DoS.
const MAX_ENCODED_LEN: usize = 2048;

/// Inner payload of a v2 invite code — what the recipient gets after
/// decryption.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InviteCodePayload {
    /// Wire-format version. Must equal `INVITE_VERSION` or decode fails fast.
    pub version: u8,
    /// The pool ID. Equals the owner's `NodeId`; the libp2p `PeerId` is
    /// derivable from this via `peer_id_to_node_id`'s inverse.
    pub pool_id: NodeId,
    /// Optional human-friendly pool name for the joining UI. Empty string if
    /// the owner hasn't named the pool.
    pub pool_name: String,
    /// The owner's reachable listen multiaddrs (each terminated with
    /// `/p2p/<peer_id>` so a remote dialer can verify the target identity).
    /// The joiner dials each in order until one connects.
    pub multiaddrs: Vec<String>,
    /// The 8-char short code that still drives the actual join wire protocol.
    /// The joiner hashes this and broadcasts `JoinRequest { code_hash, ... }`
    /// over GossipSub — same as the legacy 8-char path.
    pub code: String,
    /// Expiry as a unix-seconds timestamp. Mirrored from the underlying
    /// `PoolInviteCode.expires_at` so decoders can fail fast before dialing.
    pub expires_at_unix: i64,
}

impl InviteCodePayload {
    pub fn is_expired(&self) -> bool {
        chrono::Utc::now().timestamp() > self.expires_at_unix
    }
}

/// Encode an invite code into the `swarmpool://...` wire form.
///
/// Returns `Err` only if JSON serialization of the payload fails — in
/// practice that means a bug, since all field types are infallibly
/// serializable.
pub fn encode_invite_code(payload: &InviteCodePayload) -> Result<String, SwarmError> {
    let plaintext = serde_json::to_vec(payload)
        .map_err(|e| SwarmError::Internal(format!("invite payload serialize: {e}")))?;

    let mut key_bytes = [0u8; 32];
    let mut nonce_bytes = [0u8; 12];
    OsRng.fill_bytes(&mut key_bytes);
    OsRng.fill_bytes(&mut nonce_bytes);

    let cipher = ChaCha20Poly1305::new_from_slice(&key_bytes).expect("valid key size");
    let nonce = Nonce::from_slice(&nonce_bytes);
    let ciphertext = cipher
        .encrypt(nonce, plaintext.as_slice())
        .expect("ChaCha20Poly1305 encryption cannot fail on a valid key");

    let mut packed = Vec::with_capacity(32 + 12 + ciphertext.len());
    packed.extend_from_slice(&key_bytes);
    packed.extend_from_slice(&nonce_bytes);
    packed.extend_from_slice(&ciphertext);
    let encoded = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(&packed);
    Ok(format!("{INVITE_PREFIX}{encoded}"))
}

/// Decode a `swarmpool://...` blob back to its inner payload.
///
/// Returns `SwarmError::Validation` for malformed input, expired codes, or
/// version mismatches — anything the user might paste that we can recognize
/// and report cleanly. Decryption failure also surfaces as `Validation`
/// (rather than `Internal`) because the most likely cause is a truncated
/// or mistyped code, not a daemon bug.
pub fn decode_invite_code(raw: &str) -> Result<InviteCodePayload, SwarmError> {
    let trimmed = raw.trim();
    let Some(encoded) = trimmed.strip_prefix(INVITE_PREFIX) else {
        return Err(SwarmError::Validation(format!(
            "Invite code must start with '{INVITE_PREFIX}'"
        )));
    };

    if encoded.is_empty() {
        return Err(SwarmError::Validation("Invite code is empty".into()));
    }
    if encoded.len() > MAX_ENCODED_LEN {
        return Err(SwarmError::Validation(format!(
            "Invite code too long (max {MAX_ENCODED_LEN} chars)"
        )));
    }

    let packed = base64::engine::general_purpose::URL_SAFE_NO_PAD
        .decode(encoded.as_bytes())
        .map_err(|e| SwarmError::Validation(format!("Invite code is not valid base64url: {e}")))?;

    if packed.len() < 32 + 12 + 16 {
        return Err(SwarmError::Validation(
            "Invite code is too short to be valid".into(),
        ));
    }

    let key_bytes = &packed[..32];
    let nonce_bytes = &packed[32..44];
    let ciphertext = &packed[44..];

    let cipher = ChaCha20Poly1305::new_from_slice(key_bytes)
        .map_err(|e| SwarmError::Validation(format!("Invite code key invalid: {e}")))?;
    let nonce = Nonce::from_slice(nonce_bytes);
    let plaintext = cipher.decrypt(nonce, ciphertext).map_err(|_| {
        SwarmError::Validation(
            "Invite code is corrupt or was edited — ask the inviter to regenerate".into(),
        )
    })?;

    let payload: InviteCodePayload = serde_json::from_slice(&plaintext).map_err(|e| {
        SwarmError::Validation(format!(
            "Invite code payload is not valid (newer daemon version?): {e}"
        ))
    })?;

    if payload.version != INVITE_VERSION {
        return Err(SwarmError::Validation(format!(
            "Unsupported invite code version {} (this daemon supports v{})",
            payload.version, INVITE_VERSION
        )));
    }
    if payload.is_expired() {
        return Err(SwarmError::Validation(
            "Invite code has expired — ask the inviter to generate a new one".into(),
        ));
    }
    if payload.code.len() != 8 {
        return Err(SwarmError::Validation(
            "Invite code payload is malformed (join token wrong length)".into(),
        ));
    }
    Ok(payload)
}

/// Quick sniff: does this string look like a v2 code? Used by the API layer
/// to route a pasted blob to either the v2 path or the legacy 8-char path.
pub fn looks_like_v2(raw: &str) -> bool {
    raw.trim().starts_with(INVITE_PREFIX)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_payload() -> InviteCodePayload {
        InviteCodePayload {
            version: INVITE_VERSION,
            pool_id: NodeId([7u8; 32]),
            pool_name: "Test Pool".into(),
            multiaddrs: vec![
                "/ip4/100.64.0.5/tcp/8810/p2p/12D3KooWFakeXXX".into(),
                "/ip4/192.168.1.5/udp/8800/quic-v1/p2p/12D3KooWFakeXXX".into(),
            ],
            code: "A3F7K2M9".into(),
            expires_at_unix: chrono::Utc::now().timestamp() + 3600,
        }
    }

    #[test]
    fn roundtrip_preserves_payload() {
        let original = sample_payload();
        let encoded = encode_invite_code(&original).unwrap();
        assert!(encoded.starts_with(INVITE_PREFIX));

        let decoded = decode_invite_code(&encoded).unwrap();
        assert_eq!(decoded.version, original.version);
        assert_eq!(decoded.pool_id, original.pool_id);
        assert_eq!(decoded.pool_name, original.pool_name);
        assert_eq!(decoded.multiaddrs, original.multiaddrs);
        assert_eq!(decoded.code, original.code);
        assert_eq!(decoded.expires_at_unix, original.expires_at_unix);
    }

    #[test]
    fn whitespace_around_code_is_tolerated() {
        let original = sample_payload();
        let encoded = encode_invite_code(&original).unwrap();
        let padded = format!("  \n{encoded}\t  ");
        let decoded = decode_invite_code(&padded).unwrap();
        assert_eq!(decoded.code, original.code);
    }

    #[test]
    fn missing_prefix_fails() {
        let encoded = encode_invite_code(&sample_payload()).unwrap();
        let stripped = encoded.strip_prefix(INVITE_PREFIX).unwrap();
        let err = decode_invite_code(stripped).unwrap_err();
        assert!(matches!(err, SwarmError::Validation(_)));
    }

    #[test]
    fn truncated_code_fails_cleanly() {
        let encoded = encode_invite_code(&sample_payload()).unwrap();
        let truncated = &encoded[..encoded.len() - 5];
        let err = decode_invite_code(truncated).unwrap_err();
        assert!(matches!(err, SwarmError::Validation(_)));
    }

    #[test]
    fn tampered_ciphertext_fails_authentication() {
        let encoded = encode_invite_code(&sample_payload()).unwrap();
        // Flip the last char of the base64url body to a different valid char.
        let mut bytes: Vec<char> = encoded.chars().collect();
        let last = bytes.last_mut().unwrap();
        *last = if *last == 'A' { 'B' } else { 'A' };
        let tampered: String = bytes.into_iter().collect();
        let err = decode_invite_code(&tampered).unwrap_err();
        assert!(matches!(err, SwarmError::Validation(_)));
    }

    #[test]
    fn expired_code_rejected() {
        let mut payload = sample_payload();
        payload.expires_at_unix = chrono::Utc::now().timestamp() - 60;
        let encoded = encode_invite_code(&payload).unwrap();
        let err = decode_invite_code(&encoded).unwrap_err();
        match err {
            SwarmError::Validation(msg) => assert!(msg.contains("expired"), "got: {msg}"),
            other => panic!("expected Validation, got {other:?}"),
        }
    }

    #[test]
    fn version_mismatch_rejected() {
        let mut payload = sample_payload();
        payload.version = INVITE_VERSION + 7;
        let encoded = encode_invite_code(&payload).unwrap();
        let err = decode_invite_code(&encoded).unwrap_err();
        match err {
            SwarmError::Validation(msg) => assert!(msg.contains("version"), "got: {msg}"),
            other => panic!("expected Validation, got {other:?}"),
        }
    }

    #[test]
    fn malformed_join_token_length_rejected() {
        let mut payload = sample_payload();
        payload.code = "TOOSHORT".into(); // 8 — should pass
        let encoded = encode_invite_code(&payload).unwrap();
        decode_invite_code(&encoded).unwrap();

        payload.code = "WAYTOOLONG".into(); // 10
        let encoded = encode_invite_code(&payload).unwrap();
        let err = decode_invite_code(&encoded).unwrap_err();
        assert!(matches!(err, SwarmError::Validation(_)));
    }

    #[test]
    fn oversized_input_rejected_before_decode() {
        let huge = format!("{INVITE_PREFIX}{}", "A".repeat(MAX_ENCODED_LEN + 1));
        let err = decode_invite_code(&huge).unwrap_err();
        match err {
            SwarmError::Validation(msg) => assert!(msg.contains("too long"), "got: {msg}"),
            other => panic!("expected Validation, got {other:?}"),
        }
    }

    #[test]
    fn looks_like_v2_sniff() {
        assert!(looks_like_v2("swarmpool://abc"));
        assert!(looks_like_v2("  swarmpool://xyz  "));
        assert!(!looks_like_v2("A3F7K2M9"));
        assert!(!looks_like_v2("swarm://abc"));
        assert!(!looks_like_v2(""));
    }
}
