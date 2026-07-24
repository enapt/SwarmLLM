//! NETWORKING_PLAN Phase 1 — end-to-end sealing for the application-level
//! inference relay.
//!
//! A relay envelope carries a cleartext routing header (`relay_to`, `origin`,
//! `request_id`) that a relay forwards on, plus a `sealed` inner `SwarmMessage`
//! that is ephemeral-sealed for `relay_to`'s X25519 key — derived from its
//! NodeId (Ed25519 pubkey) via the birational map, so NO prior session setup
//! with the target is required. Only the target can open it; the relay is a
//! dumb pipe. The AAD binds `origin || relay_to || request_id`, so a relay that
//! re-addresses, replays, or cross-wires an envelope fails Poly1305 at the
//! recipient before the inner message is ever parsed.

use x25519_dalek::StaticSecret;

use crate::crypto::session::{ed25519_pubkey_to_x25519, ephemeral_open, ephemeral_seal};
use crate::error::SwarmError;
use crate::types::{NodeId, RelayedEnvelope, SwarmMessage};

/// Upper bound on a sealed inner message (pre-seal plaintext). Inference relay
/// payloads are prompts and single tokens — small. This caps what a relay will
/// forward and what a recipient will open, so the channel can't be abused to
/// push large blobs through the anchor.
pub const MAX_RELAY_INNER_BYTES: usize = 256 * 1024;

/// AAD binding `origin || relay_to || request_id`. Any change to these fields
/// on the wire invalidates the seal.
fn relay_aad(origin: &NodeId, relay_to: &NodeId, request_id: &uuid::Uuid) -> Vec<u8> {
    let mut aad = Vec::with_capacity(32 + 32 + 16);
    aad.extend_from_slice(&origin.0);
    aad.extend_from_slice(&relay_to.0);
    aad.extend_from_slice(request_id.as_bytes());
    aad
}

/// Seal an inner `SwarmMessage` end-to-end for `relay_to`, producing an
/// envelope a relay can route but not read. `origin` is the local node.
pub fn seal_relayed_message(
    origin: NodeId,
    relay_to: NodeId,
    request_id: uuid::Uuid,
    inner: &SwarmMessage,
) -> Result<RelayedEnvelope, SwarmError> {
    let target_x = ed25519_pubkey_to_x25519(&relay_to.0)
        .ok_or_else(|| SwarmError::Encryption("relay target key not convertible".into()))?;
    let plaintext = serde_json::to_vec(inner)
        .map_err(|e| SwarmError::Encryption(format!("serialize relayed inner: {e}")))?;
    if plaintext.len() > MAX_RELAY_INNER_BYTES {
        return Err(SwarmError::Encryption(format!(
            "relayed inner too large: {} bytes (max {MAX_RELAY_INNER_BYTES})",
            plaintext.len()
        )));
    }
    let aad = relay_aad(&origin, &relay_to, &request_id);
    let (ephemeral_pub, sealed) = ephemeral_seal(&target_x, &plaintext, &aad)?;
    Ok(RelayedEnvelope {
        relay_to,
        origin,
        request_id,
        ephemeral_pub,
        sealed,
    })
}

/// Open a relayed envelope addressed to this node. `local_secret` is this
/// node's X25519 static secret (`ed25519_to_x25519_secret` of its signing key).
/// The caller MUST have already checked `env.relay_to == local_node_id`.
pub fn open_relayed_message(
    local_secret: &StaticSecret,
    env: &RelayedEnvelope,
) -> Result<SwarmMessage, SwarmError> {
    if env.sealed.len() > MAX_RELAY_INNER_BYTES + 64 {
        return Err(SwarmError::Encryption("relayed envelope too large".into()));
    }
    let aad = relay_aad(&env.origin, &env.relay_to, &env.request_id);
    let plaintext = ephemeral_open(local_secret, &env.ephemeral_pub, &env.sealed, &aad)?;
    serde_json::from_slice(&plaintext)
        .map_err(|e| SwarmError::Encryption(format!("deserialize relayed inner: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crypto::session::ed25519_to_x25519_secret;
    use crate::identity::Identity;

    fn inner_msg() -> SwarmMessage {
        SwarmMessage::CancelInference(crate::types::CancelInference {
            request_id: uuid::Uuid::new_v4(),
        })
    }

    #[test]
    fn seal_open_roundtrip() {
        let origin = Identity::generate();
        let target = Identity::generate();
        let target_secret = ed25519_to_x25519_secret(&target.signing_key_bytes());

        let rid = uuid::Uuid::new_v4();
        let msg = inner_msg();
        let env = seal_relayed_message(
            origin.node_id().clone(),
            target.node_id().clone(),
            rid,
            &msg,
        )
        .unwrap();

        assert_eq!(&env.relay_to, target.node_id());
        assert_eq!(&env.origin, origin.node_id());
        assert!(!env.sealed.is_empty());

        let opened = open_relayed_message(&target_secret, &env).unwrap();
        assert!(matches!(opened, SwarmMessage::CancelInference(_)));
    }

    #[test]
    fn wrong_target_cannot_open() {
        let origin = Identity::generate();
        let target = Identity::generate();
        let eavesdropper = Identity::generate();
        let eve_secret = ed25519_to_x25519_secret(&eavesdropper.signing_key_bytes());

        let env = seal_relayed_message(
            origin.node_id().clone(),
            target.node_id().clone(),
            uuid::Uuid::new_v4(),
            &inner_msg(),
        )
        .unwrap();

        // A relay (or anyone who is not `relay_to`) cannot open the payload.
        assert!(open_relayed_message(&eve_secret, &env).is_err());
    }

    #[test]
    fn tampered_routing_header_fails_seal() {
        let origin = Identity::generate();
        let target = Identity::generate();
        let target_secret = ed25519_to_x25519_secret(&target.signing_key_bytes());

        let mut env = seal_relayed_message(
            origin.node_id().clone(),
            target.node_id().clone(),
            uuid::Uuid::new_v4(),
            &inner_msg(),
        )
        .unwrap();

        // A relay swapping the origin in the cleartext header breaks the AAD.
        env.origin = Identity::generate().node_id().clone();
        assert!(open_relayed_message(&target_secret, &env).is_err());
    }

    #[test]
    fn oversized_inner_rejected() {
        // A serialized inner larger than the cap is refused at seal time.
        let origin = Identity::generate();
        let target = Identity::generate();
        // ShardDownloadProgress is tiny; instead force the cap via a giant
        // nickname-like message. Use PeerExchangeResponse with huge strings.
        let huge = "x".repeat(MAX_RELAY_INNER_BYTES);
        let msg = SwarmMessage::PeerExchangeResponse(crate::types::PeerExchangeResponse {
            peers: vec![huge],
        });
        let err = seal_relayed_message(
            origin.node_id().clone(),
            target.node_id().clone(),
            uuid::Uuid::new_v4(),
            &msg,
        );
        assert!(err.is_err());
    }
}
