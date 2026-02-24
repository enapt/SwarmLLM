use libp2p::core::upgrade;
use libp2p::identity::Keypair;
use libp2p::{noise, tcp, yamux, PeerId, Transport};

use crate::error::SwarmError;

/// Build a TCP+Noise+Yamux transport as fallback for QUIC.
///
/// The primary transport (QUIC) is configured directly in the swarm builder.
/// This function provides a TCP fallback transport with Noise encryption
/// and Yamux multiplexing.
pub fn build_tcp_transport(
    keypair: &Keypair,
) -> Result<
    libp2p::core::transport::Boxed<(PeerId, libp2p::core::muxing::StreamMuxerBox)>,
    SwarmError,
> {
    let transport = tcp::tokio::Transport::new(tcp::Config::default().nodelay(true))
        .upgrade(upgrade::Version::V1Lazy)
        .authenticate(
            noise::Config::new(keypair)
                .map_err(|e| SwarmError::Network(format!("Noise config error: {e}")))?,
        )
        .multiplex(yamux::Config::default())
        .boxed();

    Ok(transport)
}

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

    #[test]
    fn build_tcp_transport_succeeds() {
        let keypair = Keypair::generate_ed25519();
        let result = build_tcp_transport(&keypair);
        assert!(result.is_ok());
    }
}
