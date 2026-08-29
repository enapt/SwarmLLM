use std::time::Duration;

use libp2p::kad;
use libp2p::kad::RecordKey;
use libp2p::swarm::Swarm;
use libp2p::Multiaddr;

use crate::error::SwarmError;
use crate::network::behaviour::SwarmBehaviour;
use crate::types::ShardId;

/// One dial's worth of bootstrap work: a peer, and every address we know for it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BootstrapDial {
    /// The peer these addresses lead to, when the addresses name one.
    pub peer: Option<libp2p::PeerId>,
    /// Every known address for that peer, in the order they were supplied.
    pub addrs: Vec<Multiaddr>,
}

/// Group bootstrap and cached addresses by the peer they lead to — **one dial
/// per peer, never one per address**.
///
/// This used to issue a separate bare `swarm.dial(addr)` for every string in
/// the list. A bare-address dial carries no `PeerCondition`, so libp2p's
/// per-peer dedup cannot see it, and nothing stopped N addresses for one peer
/// becoming N simultaneous dials. Each that completed became another
/// connection, up to `network.max_connections_per_peer`. Measured on the live
/// swarm: a peer cached at two addresses (TCP and QUIC) reached the cap of
/// three routinely, after which request_response spread sends across all three
/// and whichever had quietly died swallowed its share (gotcha #405).
///
/// libp2p already does the right thing when asked properly: a dial by peer id
/// carrying several addresses RACES them concurrently within one attempt
/// (`libp2p_swarm::connection::pool::concurrent_dial`, a `FuturesUnordered`
/// bounded by `dial_concurrency_factor`) and yields exactly ONE connection —
/// the first to succeed, with the rest dropped. So this costs no connection
/// latency against the per-address version; it just stops keeping the losers.
///
/// The peer is read from the LAST `/p2p/` component. A relay circuit is
/// `…/p2p/<relay>/p2p-circuit/p2p/<target>` and the first component is the
/// relay, so reading that grouped a peer's circuit address under the relay and
/// asked "are we connected?" about the wrong node.
///
/// Addresses that name no peer keep their own entry with `peer: None` — the
/// caller can only dial those bare, which is correct for genuine discovery of
/// somebody we have never seen.
pub fn plan_bootstrap_dials(addrs: &[String]) -> Vec<BootstrapDial> {
    let mut plan: Vec<BootstrapDial> = Vec::new();
    for addr_str in addrs {
        let addr = match addr_str.parse::<Multiaddr>() {
            Ok(a) => a,
            Err(e) => {
                tracing::warn!(addr = %addr_str, error = %e, "Invalid bootstrap address");
                continue;
            }
        };
        let peer = crate::network::helpers::target_peer_from_address(&addr);
        match plan.iter_mut().find(|d| d.peer.is_some() && d.peer == peer) {
            Some(existing) => existing.addrs.push(addr),
            None => plan.push(BootstrapDial {
                peer,
                addrs: vec![addr],
            }),
        }
    }
    plan
}

/// Trigger a Kademlia bootstrap query to discover new peers.
pub fn trigger_bootstrap(swarm: &mut Swarm<SwarmBehaviour>) -> Result<(), SwarmError> {
    match swarm.behaviour_mut().kademlia.bootstrap() {
        Ok(query_id) => {
            tracing::debug!(?query_id, "Kademlia bootstrap query started");
            Ok(())
        }
        Err(e) => {
            // Empty routing table at startup is expected — only warn once peers are connected.
            let connected = swarm.connected_peers().count();
            if connected > 0 {
                tracing::warn!(
                    error = %e,
                    connected_peers = connected,
                    "DIAG: Kademlia bootstrap failed despite connected peers"
                );
            } else {
                tracing::debug!(
                    error = %e,
                    "Kademlia bootstrap skipped — routing table empty (expected at startup)"
                );
            }
            Ok(())
        }
    }
}

/// Subscribe to the standard GossipSub topics for this node's network.
///
/// `network_id` is `None` on the public swarm, which keeps the topic names
/// exactly as every existing node knows them. A configured private network gets
/// its own scoped topics — see `protocol::topic_for_network`.
pub fn subscribe_topics(
    swarm: &mut Swarm<SwarmBehaviour>,
    network_id: Option<&str>,
) -> Result<(), SwarmError> {
    use crate::network::protocol::{
        topic_for_network, TOPIC_CREDITS, TOPIC_HEALTH, TOPIC_IDENTITY, TOPIC_MODELS, TOPIC_POOLS,
        TOPIC_REGIONS,
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

    for base in &topics {
        let topic_str = topic_for_network(base, network_id);
        let topic = IdentTopic::new(topic_str.clone());
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
    let sig_bytes = &signed[32..96];
    let payload = &signed[96..];

    let vk = ed25519_dalek::VerifyingKey::from_bytes(&pubkey_bytes)
        .map_err(|_| SwarmError::InvalidSignature)?;
    crate::crypto::verify_ed25519_sig(sig_bytes, payload, &vk)?;

    Ok((pubkey_bytes, payload))
}

/// Discovery interval for periodic peer discovery.
pub const DISCOVERY_INTERVAL: Duration = Duration::from_secs(300);

/// Build a list of loopback TCP multiaddrs to probe for same-machine SwarmLLM peers.
///
/// On WSL2 (and other environments with broken multicast), mDNS cannot find a
/// peer running on the same host, and Kademlia bootstrap has no entry point on
/// a fresh install with an empty peer cache and no configured bootstrap peers.
/// This cheaply scans a small range of common SwarmLLM HTTP API ports on
/// 127.0.0.1, producing TCP P2P multiaddrs (api_port+10) to dial. libp2p rejects
/// dials to our own PeerId and silently drops failed connections, so probing a
/// port range on loopback is a benign O(N) operation.
///
/// Returns addresses for api_port in `own_api_port-10 ..= own_api_port+10` (clamped),
/// excluding the caller's own port. The TCP offset is api_port + 10 to match
/// `NetworkManager::run()`.
fn loopback_candidate_addrs(own_api_port: u16) -> Vec<Multiaddr> {
    let mut ports: std::collections::BTreeSet<u16> = std::collections::BTreeSet::new();
    // Adjacent ports: handles tight test layouts like 8800/8801/8802.
    let lo: u16 = own_api_port.saturating_sub(10).max(1024);
    let hi: u16 = own_api_port.saturating_add(10);
    for p in lo..=hi {
        ports.insert(p);
    }
    // Common multi-node test bases: 8800/8900/9000/9100 layouts (100 apart)
    // and 9800-series used by some dev configs. Covers the standard SwarmLLM
    // multi-node setup where loopback probing needs to cross larger gaps.
    for base in [8800u16, 8900, 9000, 9100, 9200, 9800, 9900] {
        ports.insert(base);
    }
    ports.remove(&own_api_port);
    let mut addrs = Vec::with_capacity(ports.len());
    for api_port in ports {
        let tcp_port = api_port.saturating_add(10);
        if let Ok(addr) = format!("/ip4/127.0.0.1/tcp/{tcp_port}").parse::<Multiaddr>() {
            addrs.push(addr);
        }
    }
    addrs
}

/// Dial each loopback candidate once. Harmless no-ops for ports with no listener
/// (libp2p reports DialFailure and moves on). Logs at debug to avoid log spam.
pub fn probe_loopback_peers(swarm: &mut Swarm<SwarmBehaviour>, own_api_port: u16) -> usize {
    let candidates = loopback_candidate_addrs(own_api_port);
    let mut dialed = 0;
    for addr in candidates {
        match swarm.dial(addr.clone()) {
            Ok(_) => {
                dialed += 1;
                tracing::debug!(%addr, "Loopback probe dial");
            }
            Err(e) => {
                tracing::trace!(%addr, error = %e, "Loopback probe dial rejected");
            }
        }
    }
    if dialed > 0 {
        tracing::debug!(count = dialed, "Loopback discovery: dialed candidate ports");
    }
    dialed
}

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
                    // to plain path (would accept raw crypto bytes as multiaddr)
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
        // Never announce ourselves as a DHT provider for a backup-copy model
        // (`<model>.FULLBACKUP`) — the callers filter these out too, but this is
        // the single choke point every StartProviding flows through, so it can't
        // leak a copied-folder name into the swarm's provider records.
        if crate::model::manifest::is_backup_artifact_id(&shard_id.model_id.0) {
            continue;
        }
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

/// Fixed Kademlia key under which relay-capable nodes register themselves, so a
/// node that has lost its relay(s) can discover fresh ones from the DHT —
/// decentralizing the relay role past the single bootstrap anchor
/// (NETWORKING_PLAN Phase 3). Versioned so the record namespace can evolve.
pub fn relay_service_key() -> RecordKey {
    RecordKey::new(&"/swarm/relay-service/v1")
}

/// Register this node as a DHT provider of relay service. Kademlia auto-
/// republishes and expires the record. Idempotent and safe to retry until
/// Kademlia has peers (returns false, e.g. at cold start with no routing table).
pub fn start_providing_relay_service(swarm: &mut Swarm<SwarmBehaviour>) -> bool {
    match swarm
        .behaviour_mut()
        .kademlia
        .start_providing(relay_service_key())
    {
        Ok(_qid) => {
            tracing::info!("Registered as a DHT relay-service provider");
            true
        }
        Err(e) => {
            tracing::debug!(error = %e, "start_providing relay service failed (no Kademlia peers yet?)");
            false
        }
    }
}

/// Query the DHT for relay-service providers. Results arrive as a
/// `GetProviders` event; the caller tracks the returned `QueryId` to recognize
/// them and dial the discovered relays.
pub fn query_relay_providers(swarm: &mut Swarm<SwarmBehaviour>) -> kad::QueryId {
    swarm
        .behaviour_mut()
        .kademlia
        .get_providers(relay_service_key())
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
    fn relay_service_key_is_stable() {
        // This key is the network-wide rendezvous point for relay discovery
        // (NETWORKING_PLAN Phase 3). Changing it silently partitions relay
        // discovery across versions — pin it, and bump the `/v1` suffix
        // deliberately if the namespace ever must change.
        let key = relay_service_key();
        assert_eq!(key.as_ref(), b"/swarm/relay-service/v1");
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

    /// The bug, directly: a peer cached at two addresses used to become two
    /// simultaneous unconditional dials, and each that landed was another
    /// connection — up to the per-peer cap, routinely, on the live swarm.
    /// libp2p tries several addresses within ONE dial when asked by peer id.
    #[test]
    fn every_address_for_one_peer_becomes_a_single_dial() {
        let pid = libp2p::PeerId::random();
        let plan = plan_bootstrap_dials(&[
            format!("/ip4/81.241.51.1/tcp/8810/p2p/{pid}"),
            format!("/ip4/81.241.51.1/udp/8800/quic-v1/p2p/{pid}"),
        ]);
        assert_eq!(
            plan.len(),
            1,
            "one peer must produce one dial, not one per address"
        );
        assert_eq!(plan[0].peer, Some(pid));
        assert_eq!(
            plan[0].addrs.len(),
            2,
            "both addresses ride along on that single dial"
        );
    }

    /// A relay circuit names the relay FIRST and the destination last. Reading
    /// the first grouped a peer's circuit address under the relay, so the
    /// already-connected check asked about the wrong node entirely.
    #[test]
    fn a_relay_circuit_groups_under_the_destination_not_the_relay() {
        let relay = libp2p::PeerId::random();
        let target = libp2p::PeerId::random();
        let plan = plan_bootstrap_dials(&[
            format!("/ip4/81.241.51.1/tcp/8810/p2p/{target}"),
            format!("/dns4/relay.example/tcp/8810/p2p/{relay}/p2p-circuit/p2p/{target}"),
        ]);
        assert_eq!(
            plan.len(),
            1,
            "the direct address and the circuit both lead to the same peer"
        );
        assert_eq!(plan[0].peer, Some(target));
    }

    /// Distinct peers must not be merged, or dialling one would carry
    /// another's addresses.
    #[test]
    fn distinct_peers_stay_distinct() {
        let a = libp2p::PeerId::random();
        let b = libp2p::PeerId::random();
        let plan = plan_bootstrap_dials(&[
            format!("/ip4/10.0.0.1/tcp/8810/p2p/{a}"),
            format!("/ip4/10.0.0.2/tcp/8810/p2p/{b}"),
            format!("/ip4/10.0.0.3/tcp/8810/p2p/{a}"),
        ]);
        assert_eq!(plan.len(), 2);
        assert_eq!(plan[0].peer, Some(a));
        assert_eq!(plan[0].addrs.len(), 2);
        assert_eq!(plan[1].peer, Some(b));
    }

    /// An address naming nobody keeps its own entry and is never merged with
    /// another unnamed one — they may be different machines, and there is no
    /// evidence either way. This is genuine first contact.
    #[test]
    fn unnamed_addresses_are_never_merged_with_each_other() {
        let plan = plan_bootstrap_dials(&[
            "/ip4/10.0.0.1/tcp/8810".to_string(),
            "/ip4/10.0.0.2/tcp/8810".to_string(),
        ]);
        assert_eq!(
            plan.len(),
            2,
            "two unnamed addresses are two unknowns, not one peer"
        );
        assert!(plan.iter().all(|d| d.peer.is_none()));
    }

    /// A malformed entry is skipped, not fatal — the list comes from a config
    /// file and a saved cache.
    #[test]
    fn an_unparseable_address_is_skipped() {
        let pid = libp2p::PeerId::random();
        let plan = plan_bootstrap_dials(&[
            "not-a-multiaddr".to_string(),
            format!("/ip4/10.0.0.1/tcp/8810/p2p/{pid}"),
        ]);
        assert_eq!(plan.len(), 1);
        assert_eq!(plan[0].peer, Some(pid));
    }
}
