use libp2p::identity::Keypair;
use libp2p::PeerId;

use crate::error::SwarmError;
use crate::types::NodeId;

/// Convert an Ed25519 signing key (32 bytes) to a libp2p Keypair.
pub fn ed25519_to_libp2p_keypair(signing_key_bytes: [u8; 32]) -> Result<Keypair, SwarmError> {
    let mut key_bytes = signing_key_bytes;
    Keypair::ed25519_from_bytes(&mut key_bytes)
        .map_err(|e| SwarmError::Network(format!("Failed to create libp2p keypair: {e}")))
}

/// Convert a NodeId (Ed25519 public key bytes) to a libp2p PeerId.
///
/// Both NodeId and PeerId derive from the same Ed25519 key, so the conversion
/// is deterministic. Returns None if the bytes are not a valid Ed25519 public key.
///
/// Was test-only until the diagnostics endpoint needed the local PeerId to
/// filter the peer cache the same way the network manager does — that filter
/// drops any address routing through our own id, so it needs the id.
pub fn node_id_to_peer_id(node_id: &NodeId) -> Option<PeerId> {
    let ed_pk = libp2p::identity::ed25519::PublicKey::try_from_bytes(&node_id.0).ok()?;
    let pk = libp2p::identity::PublicKey::from(ed_pk);
    Some(pk.to_peer_id())
}

/// Extract NodeId (Ed25519 public key bytes) from a libp2p PeerId.
///
/// This only works for Ed25519-based PeerIds where the public key is inlined
/// in the multihash (identity hash). Returns None for other PeerId types.
pub fn peer_id_to_node_id(peer_id: &PeerId) -> Option<NodeId> {
    // PeerId for Ed25519 keys uses identity multihash, so the public key
    // can be extracted directly from the PeerId bytes.
    // .get(2..) — return None on a malformed/short PeerId rather than
    // panicking on an unchecked slice (R96).
    let bytes = peer_id.to_bytes();
    let pk = libp2p::identity::PublicKey::try_decode_protobuf(bytes.get(2..)?).ok()?;
    let ed_pk = pk.try_into_ed25519().ok()?;
    let bytes = ed_pk.to_bytes();
    Some(NodeId(bytes))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keypair_conversion() {
        use ed25519_dalek::SigningKey;
        use rand::rngs::OsRng;

        let signing_key = SigningKey::generate(&mut OsRng);
        let result = ed25519_to_libp2p_keypair(signing_key.to_bytes());
        assert!(result.is_ok());

        let keypair = result.unwrap();
        assert!(keypair
            .public()
            .to_peer_id()
            .to_string()
            .starts_with("12D3"));
    }

    #[test]
    fn node_id_peer_id_roundtrip() {
        use ed25519_dalek::SigningKey;
        use rand::rngs::OsRng;

        let signing_key = SigningKey::generate(&mut OsRng);
        let node_id = NodeId(signing_key.verifying_key().to_bytes());

        // NodeId → PeerId
        let peer_id = node_id_to_peer_id(&node_id).expect("should convert");

        // PeerId → NodeId
        let recovered = peer_id_to_node_id(&peer_id).expect("should recover");
        assert_eq!(node_id, recovered);
    }

    #[test]
    fn node_id_to_peer_id_matches_keypair() {
        use ed25519_dalek::SigningKey;
        use rand::rngs::OsRng;

        let signing_key = SigningKey::generate(&mut OsRng);
        let node_id = NodeId(signing_key.verifying_key().to_bytes());

        // Convert via keypair (existing path)
        let keypair = ed25519_to_libp2p_keypair(signing_key.to_bytes()).unwrap();
        let peer_id_from_keypair = keypair.public().to_peer_id();

        // Convert via node_id (new path)
        let peer_id_from_node_id = node_id_to_peer_id(&node_id).unwrap();

        assert_eq!(peer_id_from_keypair, peer_id_from_node_id);
    }
}
