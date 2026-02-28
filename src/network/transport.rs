use libp2p::identity::Keypair;

use crate::error::SwarmError;

/// Convert an Ed25519 signing key (32 bytes) to a libp2p Keypair.
pub fn ed25519_to_libp2p_keypair(signing_key_bytes: [u8; 32]) -> Result<Keypair, SwarmError> {
    let mut key_bytes = signing_key_bytes;
    Keypair::ed25519_from_bytes(&mut key_bytes)
        .map_err(|e| SwarmError::Network(format!("Failed to create libp2p keypair: {e}")))
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
}
