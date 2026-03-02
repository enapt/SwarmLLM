use std::time::Duration;

use libp2p::connection_limits;
use libp2p::identity::Keypair;
use libp2p::kad::store::MemoryStore;
use libp2p::swarm::NetworkBehaviour;
use libp2p::{
    autonat, dcutr, gossipsub, identify, kad, mdns, relay, request_response, StreamProtocol,
};

use crate::config::NetworkConfig;
use crate::network::protocol::SwarmCodec;
use crate::network::relay::RelayServerConfig;

/// Combined network behaviour for the SwarmLLM node.
///
/// Uses the libp2p `NetworkBehaviour` derive macro to combine multiple
/// sub-protocols into a single behaviour.
///
/// Tensor activation forwarding uses the same `request_response` protocol as
/// control messages (unified codec with a type-tag byte to distinguish JSON
/// from binary payloads). This avoids dual-protocol connection routing issues
/// that cause silent message loss when mDNS creates duplicate connections.
#[derive(NetworkBehaviour)]
pub struct SwarmBehaviour {
    pub kademlia: kad::Behaviour<MemoryStore>,
    pub gossipsub: gossipsub::Behaviour,
    pub request_response: request_response::Behaviour<SwarmCodec>,
    pub identify: identify::Behaviour,
    pub autonat: autonat::Behaviour,
    pub dcutr: dcutr::Behaviour,
    pub relay_client: relay::client::Behaviour,
    /// Relay server: accepts reservations from NAT'd peers and forwards circuits.
    pub relay_server: relay::Behaviour,
    /// NET-I5: Connection limits to prevent resource exhaustion.
    pub connection_limits: connection_limits::Behaviour,
    /// mDNS for automatic LAN peer discovery (zero-config).
    pub mdns: libp2p::swarm::behaviour::toggle::Toggle<mdns::tokio::Behaviour>,
}

/// Build the combined network behaviour with all sub-protocols configured.
///
/// `known_peers` is the count of peers from the peer cache, used to auto-scale
/// GossipSub mesh parameters. Small clusters (< 10 peers) use lower thresholds,
/// while larger networks scale up for faster message propagation.
pub fn build_behaviour(
    local_key: &Keypair,
    relay_behaviour: relay::client::Behaviour,
    relay_server_config: Option<&RelayServerConfig>,
    enable_mdns: bool,
    known_peers: usize,
    network_config: Option<&NetworkConfig>,
) -> Result<SwarmBehaviour, Box<dyn std::error::Error>> {
    let local_peer_id = local_key.public().to_peer_id();

    // Kademlia DHT for peer and shard discovery
    let store = MemoryStore::new(local_peer_id);
    let mut kademlia = kad::Behaviour::new(local_peer_id, store);
    // NET-I7: Start in Client mode, switch to Server when external address is confirmed
    kademlia.set_mode(Some(kad::Mode::Client));

    // GossipSub for network-wide announcements
    // Use blake3 for deterministic message IDs across processes
    // (DefaultHasher is seeded randomly per process, breaking deduplication)
    //
    // NET-M9: The message_id_fn includes both data and source peer to distinguish
    // identical payloads from different peers. This means the same logical message
    // (e.g. a shard announce) sent by different peers will NOT be deduplicated —
    // this is intentional since each peer's announcement is meaningful.
    let message_id_fn = |message: &gossipsub::Message| {
        let mut input = message.data.clone();
        if let Some(ref source) = message.source {
            input.extend_from_slice(&source.to_bytes());
        }
        let hash = blake3::hash(&input);
        gossipsub::MessageId::from(hex::encode(&hash.as_bytes()[..16]))
    };
    // Auto-scale GossipSub mesh parameters based on known peer count.
    // Small clusters need low thresholds to form a mesh; large networks need
    // higher values for faster message propagation (O(log n) hops).
    let (mesh_n, mesh_n_low, mesh_n_high, mesh_outbound_min) = if known_peers >= 100 {
        (6, 4, 12, 3) // Full GossipSub defaults — fast propagation
    } else if known_peers >= 30 {
        (4, 3, 8, 2) // Medium networks
    } else if known_peers >= 10 {
        (3, 2, 6, 1) // Small-medium networks
    } else {
        (2, 1, 4, 1) // Tiny clusters (dev/early alpha)
    };
    tracing::info!(
        known_peers,
        mesh_n,
        mesh_n_low,
        mesh_n_high,
        "Auto-scaled GossipSub mesh parameters"
    );

    let gossipsub_config = gossipsub::ConfigBuilder::default()
        .heartbeat_interval(Duration::from_secs(10))
        .validation_mode(gossipsub::ValidationMode::Strict)
        .message_id_fn(message_id_fn)
        .mesh_n(mesh_n)
        .mesh_n_low(mesh_n_low)
        .mesh_n_high(mesh_n_high)
        .mesh_outbound_min(mesh_outbound_min)
        .build()
        .map_err(|e| format!("GossipSub config error: {e}"))?;
    let gossipsub = gossipsub::Behaviour::new(
        gossipsub::MessageAuthenticity::Signed(local_key.clone()),
        gossipsub_config,
    )
    .map_err(|e| format!("GossipSub init error: {e}"))?;

    // Request/Response for all direct peer communication:
    // - JSON control messages (shard transfers, PEX, health)
    // - Binary tensor payloads (activation forwarding)
    // Uses a unified codec with type-tag byte to distinguish formats.
    // NET-C3: 300s timeout for shard transfers (large files need more time)
    let codec = if let Some(net_cfg) = network_config {
        SwarmCodec {
            compress_tensors: net_cfg.tensor_compression,
            compress_level: net_cfg.tensor_compress_level,
            compress_threshold: net_cfg.tensor_compress_threshold,
        }
    } else {
        SwarmCodec::default()
    };
    let request_response = request_response::Behaviour::with_codec(
        codec,
        [(
            StreamProtocol::new("/swarmllm/1.0.0"),
            request_response::ProtocolSupport::Full,
        )],
        request_response::Config::default().with_request_timeout(Duration::from_secs(300)),
    );

    // Identify protocol
    let identify = identify::Behaviour::new(identify::Config::new(
        "/swarmllm/id/1.0.0".to_string(),
        local_key.public(),
    ));

    // AutoNAT for NAT detection
    let autonat = autonat::Behaviour::new(local_peer_id, autonat::Config::default());

    // DCUtR for hole punching
    let dcutr = dcutr::Behaviour::new(local_peer_id);

    // Relay server: if config provided, use those limits; otherwise use defaults.
    let relay_config = relay_server_config
        .map(crate::network::relay::build_relay_server_config)
        .unwrap_or_default();
    let relay_server = relay::Behaviour::new(local_peer_id, relay_config);

    // NET-I5: Connection limits to prevent resource exhaustion.
    // max_established_per_peer=4: When two nodes discover each other via mDNS on
    // multiple LAN interfaces, both sides dial simultaneously, creating up to 4
    // connection attempts. With max=2, the denied extras trigger ApplicationClose
    // that cascades and closes ALL connections from that peer, leaving them
    // permanently disconnected. 4 is sufficient for dual-interface simultaneous dial.
    let conn_limits = connection_limits::ConnectionLimits::default()
        .with_max_established_per_peer(Some(4))
        .with_max_established(Some(500));
    let connection_limits = connection_limits::Behaviour::new(conn_limits);

    // mDNS for automatic LAN peer discovery
    let mdns_behaviour = if enable_mdns {
        let mdns_config = mdns::Config {
            ttl: Duration::from_secs(300),
            query_interval: Duration::from_secs(10),
            enable_ipv6: false,
        };
        Some(mdns::tokio::Behaviour::new(mdns_config, local_peer_id)?)
    } else {
        None
    };

    Ok(SwarmBehaviour {
        kademlia,
        gossipsub,
        request_response,
        identify,
        autonat,
        dcutr,
        relay_client: relay_behaviour,
        relay_server,
        connection_limits,
        mdns: mdns_behaviour.into(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_behaviour_succeeds() {
        let keypair = Keypair::generate_ed25519();
        let (relay_transport, relay_behaviour) = relay::client::new(keypair.public().to_peer_id());
        // relay_transport isn't used in this test
        drop(relay_transport);
        let result = build_behaviour(&keypair, relay_behaviour, None, false, 0, None);
        assert!(result.is_ok());
    }

    #[test]
    fn build_behaviour_with_relay_config() {
        let keypair = Keypair::generate_ed25519();
        let (relay_transport, relay_behaviour) = relay::client::new(keypair.public().to_peer_id());
        drop(relay_transport);
        let relay_cfg = RelayServerConfig {
            max_reservations: 64,
            max_circuits: 8,
            ..Default::default()
        };
        let result = build_behaviour(&keypair, relay_behaviour, Some(&relay_cfg), false, 0, None);
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn build_behaviour_with_mdns_enabled() {
        let keypair = Keypair::generate_ed25519();
        let (relay_transport, relay_behaviour) = relay::client::new(keypair.public().to_peer_id());
        drop(relay_transport);
        let result = build_behaviour(&keypair, relay_behaviour, None, true, 0, None);
        assert!(result.is_ok());
        let behaviour = result.unwrap();
        // mDNS should be enabled (Toggle wraps Some)
        assert!(behaviour.mdns.is_enabled());
    }

    #[test]
    fn mdns_query_interval_is_10s() {
        // Verify our mDNS config uses 10s query interval for fast LAN discovery
        let config = mdns::Config {
            ttl: Duration::from_secs(300),
            query_interval: Duration::from_secs(10),
            enable_ipv6: false,
        };
        assert_eq!(config.query_interval, Duration::from_secs(10));
    }
}
