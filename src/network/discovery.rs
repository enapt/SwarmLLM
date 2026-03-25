use std::time::Duration;

use libp2p::kad;
use libp2p::kad::RecordKey;
use libp2p::swarm::Swarm;
use libp2p::Multiaddr;

use ed25519_dalek::Verifier;

use crate::error::SwarmError;
use crate::network::behaviour::SwarmBehaviour;
use crate::types::ShardId;

/// Parse and dial bootstrap peers.
///
/// Takes a list of multiaddr strings from the config, parses them,
/// and dials each peer to join the network.
pub fn bootstrap_peers(
    swarm: &mut Swarm<SwarmBehaviour>,
    addrs: &[String],
) -> Result<usize, SwarmError> {
    let mut dialed = 0;

    for addr_str in addrs {
        match addr_str.parse::<Multiaddr>() {
            Ok(addr) => {
                // Extract peer ID from the multiaddr if present
                let maybe_peer_id = addr.iter().find_map(|proto| {
                    if let libp2p::multiaddr::Protocol::P2p(pid) = proto {
                        Some(pid)
                    } else {
                        None
                    }
                });

                if let Some(ref peer_id) = maybe_peer_id {
                    // Skip if already connected to this peer
                    if swarm.is_connected(peer_id) {
                        continue;
                    }
                    swarm
                        .behaviour_mut()
                        .kademlia
                        .add_address(peer_id, addr.clone());
                }

                match swarm.dial(addr.clone()) {
                    Ok(_) => {
                        tracing::info!(addr = %addr, peer_id = ?maybe_peer_id, "Dialing bootstrap peer");
                        dialed += 1;
                    }
                    Err(e) => {
                        tracing::debug!(addr = %addr, peer_id = ?maybe_peer_id, error = %e, "DIAG: bootstrap dial failed");
                    }
                }
            }
            Err(e) => {
                tracing::warn!(addr = %addr_str, error = %e, "Invalid bootstrap address");
            }
        }
    }

    Ok(dialed)
}

/// Trigger a Kademlia bootstrap query to discover new peers.
pub fn trigger_bootstrap(swarm: &mut Swarm<SwarmBehaviour>) -> Result<(), SwarmError> {
    match swarm.behaviour_mut().kademlia.bootstrap() {
        Ok(query_id) => {
            tracing::debug!(?query_id, "Kademlia bootstrap query started");
            Ok(())
        }
        Err(e) => {
            tracing::warn!(
                error = %e,
                connected_peers = swarm.connected_peers().count(),
                "DIAG: Kademlia bootstrap failed — no known peers in routing table"
            );
            Ok(())
        }
    }
}

/// Subscribe to the standard GossipSub topics.
pub fn subscribe_topics(swarm: &mut Swarm<SwarmBehaviour>) -> Result<(), SwarmError> {
    use crate::network::protocol::{
        TOPIC_CREDITS, TOPIC_HEALTH, TOPIC_IDENTITY, TOPIC_MODELS, TOPIC_POOLS, TOPIC_REGIONS,
    };
    use libp2p::gossipsub::IdentTopic;

    let topics = [
        TOPIC_MODELS,
        TOPIC_HEALTH,
        TOPIC_CREDITS,
        TOPIC_IDENTITY,
        TOPIC_POOLS,
        TOPIC_REGIONS,
    ];

    for topic_str in &topics {
        let topic = IdentTopic::new(*topic_str);
        swarm
            .behaviour_mut()
            .gossipsub
            .subscribe(&topic)
            .map_err(|e| SwarmError::Network(format!("Failed to subscribe to {topic_str}: {e}")))?;
        tracing::info!(topic = %topic_str, "Subscribed to GossipSub topic");
    }

    Ok(())
}

/// Verify a signed DHT record value. Returns (node_id_bytes, payload) on success.
pub fn verify_dht_value(signed: &[u8]) -> Result<([u8; 32], &[u8]), SwarmError> {
    if signed.len() < 32 + 64 {
        return Err(SwarmError::InvalidSignature);
    }
    let pubkey_bytes: [u8; 32] = signed[..32]
        .try_into()
        .map_err(|_| SwarmError::InvalidSignature)?;
    let sig_bytes: [u8; 64] = signed[32..96]
        .try_into()
        .map_err(|_| SwarmError::InvalidSignature)?;
    let payload = &signed[96..];

    let vk = ed25519_dalek::VerifyingKey::from_bytes(&pubkey_bytes)
        .map_err(|_| SwarmError::InvalidSignature)?;
    let sig = ed25519_dalek::Signature::from_bytes(&sig_bytes);
    vk.verify(payload, &sig)
        .map_err(|_| SwarmError::InvalidSignature)?;

    Ok((pubkey_bytes, payload))
}

/// Discovery interval for periodic peer discovery.
pub const DISCOVERY_INTERVAL: Duration = Duration::from_secs(300);

/// Peer cache save interval.
pub const PEER_CACHE_SAVE_INTERVAL: Duration = Duration::from_secs(300);

/// Network invite code prefix.
const INVITE_PREFIX: &str = "swarm://";

/// Encode a node's listening address into a shareable network invite code.
///
/// Format: `swarm://<base64url(random_key || nonce || encrypted_multiaddr)>`
///
/// The multiaddr is encrypted with ChaCha20Poly1305 using a random key so
/// the IP address is not visible in the code. The key is embedded in the code
/// itself — anyone with the full code can decode it, but you can't extract
/// the IP by just looking at the code (prevents casual IP harvesting).
pub fn encode_network_code(addr: &Multiaddr) -> String {
    use base64::Engine;
    use chacha20poly1305::{
        aead::{Aead, KeyInit, OsRng},
        ChaCha20Poly1305, Nonce,
    };
    use rand::RngCore;

    let plaintext = addr.to_string();

    // Generate random 32-byte key and 12-byte nonce
    let mut key_bytes = [0u8; 32];
    let mut nonce_bytes = [0u8; 12];
    OsRng.fill_bytes(&mut key_bytes);
    OsRng.fill_bytes(&mut nonce_bytes);

    let cipher = ChaCha20Poly1305::new_from_slice(&key_bytes).expect("valid key size");
    let nonce = Nonce::from_slice(&nonce_bytes);

    match cipher.encrypt(nonce, plaintext.as_bytes()) {
        Ok(ciphertext) => {
            // Pack: key (32) + nonce (12) + ciphertext (variable)
            let mut packed = Vec::with_capacity(32 + 12 + ciphertext.len());
            packed.extend_from_slice(&key_bytes);
            packed.extend_from_slice(&nonce_bytes);
            packed.extend_from_slice(&ciphertext);
            let encoded = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(&packed);
            format!("{INVITE_PREFIX}{encoded}")
        }
        Err(e) => {
            unreachable!("ChaCha20Poly1305 encryption cannot fail on valid key: {e}");
        }
    }
}

/// Decode a network invite code back into a multiaddr string.
///
/// Accepts either:
/// - `swarm://<encrypted_base64url>` (invite code format, encrypted)
/// - A raw multiaddr string (passthrough for advanced users)
pub fn decode_network_code(code: &str) -> Result<String, SwarmError> {
    let trimmed = code.trim();

    if let Some(encoded) = trimmed.strip_prefix(INVITE_PREFIX) {
        // SEC: Reject oversized invite codes before base64 decoding to prevent
        // large heap allocation from malicious input (valid codes are ~200 chars max)
        if encoded.len() > 512 {
            return Err(SwarmError::Network(
                "Invite code too long (max 512 chars)".into(),
            ));
        }
        use base64::Engine;
        let packed = base64::engine::general_purpose::URL_SAFE_NO_PAD
            .decode(encoded.as_bytes())
            .map_err(|e| SwarmError::Network(format!("Invalid invite code encoding: {e}")))?;

        // Encrypted format: key (32) + nonce (12) + ciphertext (16+ for tag)
        if packed.len() >= 32 + 12 + 16 {
            use chacha20poly1305::{aead::Aead, aead::KeyInit, ChaCha20Poly1305, Nonce};

            let key_bytes = &packed[..32];
            let nonce_bytes = &packed[32..44];
            let ciphertext = &packed[44..];

            let cipher = ChaCha20Poly1305::new_from_slice(key_bytes)
                .map_err(|e| SwarmError::Network(format!("Invalid key in invite code: {e}")))?;
            let nonce = Nonce::from_slice(nonce_bytes);

            match cipher.decrypt(nonce, ciphertext) {
                Ok(plaintext) => {
                    let addr_str = String::from_utf8(plaintext).map_err(|e| {
                        SwarmError::Network(format!("Invalid invite code UTF-8: {e}"))
                    })?;
                    addr_str.parse::<Multiaddr>().map_err(|e| {
                        SwarmError::Network(format!("Invalid multiaddr in invite code: {e}"))
                    })?;
                    return Ok(addr_str);
                }
                Err(_) => {
                    // Encrypted payload but decryption failed — do not fall through
                    // to legacy plain path (would accept raw crypto bytes as multiaddr)
                    return Err(SwarmError::Network(
                        "Invite code decryption failed — invalid or expired code".into(),
                    ));
                }
            }
        }

        // Payload too short for valid encrypted format — reject
        Err(SwarmError::Network(
            "Invite code too short — all invite codes must be encrypted".into(),
        ))
    } else if trimmed.starts_with('/') {
        // Raw multiaddr — validate and pass through
        trimmed
            .parse::<Multiaddr>()
            .map_err(|e| SwarmError::Network(format!("Invalid multiaddr: {e}")))?;
        Ok(trimmed.to_string())
    } else {
        Err(SwarmError::Network(
            "Invalid network code: must start with 'swarm://' or '/'".to_string(),
        ))
    }
}

// ── DHT Provider Records (S5: DHT-based shard holder resolution) ──

/// Generate the Kademlia provider key for a specific shard.
/// Format: `/swarm/provide/<model_id>/<shard_index>`
pub fn shard_provider_key(shard_id: &ShardId) -> RecordKey {
    RecordKey::new(&format!(
        "/swarm/provide/{}/{}",
        shard_id.model_id, shard_id.index
    ))
}

/// Register this node as a Kademlia provider for the given shards.
///
/// Provider records are automatically republished and expired by Kademlia.
/// This is used alongside GossipSub ShardAnnounce for scalable shard
/// discovery: gossip for nearby/active peers, DHT providers for cold lookups.
pub fn start_providing_shards(
    swarm: &mut Swarm<SwarmBehaviour>,
    shards: &[ShardId],
) -> Result<(), SwarmError> {
    let mut provided = 0;
    for shard_id in shards {
        let key = shard_provider_key(shard_id);
        match swarm.behaviour_mut().kademlia.start_providing(key) {
            Ok(_query_id) => {
                provided += 1;
            }
            Err(e) => {
                tracing::debug!(
                    shard = ?shard_id,
                    error = %e,
                    "Failed to start providing shard (no Kademlia peers?)"
                );
            }
        }
    }
    if provided > 0 {
        tracing::info!(
            count = provided,
            total = shards.len(),
            "Registered as DHT provider for shards"
        );
    }
    Ok(())
}

/// Stop providing specific shards via Kademlia (e.g., after shard deletion).
pub fn stop_providing_shards(swarm: &mut Swarm<SwarmBehaviour>, shards: &[ShardId]) {
    for shard_id in shards {
        let key = shard_provider_key(shard_id);
        swarm.behaviour_mut().kademlia.stop_providing(&key);
    }
    if !shards.is_empty() {
        tracing::info!(count = shards.len(), "Stopped providing shards via DHT");
    }
}

/// Query DHT for providers of a specific shard.
///
/// Returns the Kademlia query ID. Results arrive asynchronously via
/// `KademliaEvent::OutboundQueryProgressed { result: GetProviders(..) }`.
pub fn query_shard_providers(
    swarm: &mut Swarm<SwarmBehaviour>,
    shard_id: &ShardId,
) -> Result<kad::QueryId, SwarmError> {
    let key = shard_provider_key(shard_id);
    Ok(swarm.behaviour_mut().kademlia.get_providers(key))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shard_provider_key_format() {
        let sid = ShardId {
            model_id: crate::types::ModelId("test-model".into()),
            index: 3,
        };
        let key = shard_provider_key(&sid);
        // Key should contain model ID and shard index
        let key_bytes = key.as_ref();
        let key_str = std::str::from_utf8(key_bytes).unwrap();
        assert!(key_str.contains("test-model"));
        assert!(key_str.contains("/3"));
    }

    #[test]
    fn encode_decode_roundtrip() {
        let addr: Multiaddr = "/ip4/203.0.113.5/udp/8800/quic-v1".parse().unwrap();
        let code = encode_network_code(&addr);
        assert!(code.starts_with("swarm://"));

        let decoded = decode_network_code(&code).unwrap();
        assert_eq!(decoded, addr.to_string());
    }

    #[test]
    fn decode_raw_multiaddr() {
        let raw = "/ip4/192.168.1.1/udp/8800/quic-v1";
        let decoded = decode_network_code(raw).unwrap();
        assert_eq!(decoded, raw);
    }

    #[test]
    fn decode_invalid_code() {
        assert!(decode_network_code("http://example.com").is_err());
        assert!(decode_network_code("swarm://!!!invalid!!!").is_err());
    }

    #[test]
    fn decode_with_whitespace() {
        let addr: Multiaddr = "/ip4/10.0.0.1/udp/8800/quic-v1".parse().unwrap();
        let code = format!("  {}  ", encode_network_code(&addr));
        let decoded = decode_network_code(&code).unwrap();
        assert_eq!(decoded, addr.to_string());
    }
}
