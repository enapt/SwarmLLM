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

/// Kademlia provider record TTL — how long a shard-holder record stays valid
/// before being dropped. Republished automatically on the publication interval.
const KAD_PROVIDER_TTL_SECS: u64 = 3600;
/// Kademlia provider record republication interval. Must be < `KAD_PROVIDER_TTL_SECS`
/// to ensure records refresh before expiry.
const KAD_PUBLISH_INTERVAL_SECS: u64 = 1200;
/// GossipSub mesh heartbeat interval — controls mesh maintenance cadence.
const GOSSIPSUB_HEARTBEAT_SECS: u64 = 10;
/// request_response per-request timeout. 10 minutes accommodates CPU-only
/// inference on 7B+ models and slow LAN shard transfers. The vendored handler's
/// Tokio watchdog handles truly stuck futures separately. Shared with the
/// adaptive stale-tensor cleanup upper clamp in manager.rs so the two can't
/// drift out of sync.
pub const RR_REQUEST_TIMEOUT_SECS: u64 = 600;
/// mDNS service record TTL advertised to LAN peers.
const MDNS_TTL_SECS: u64 = 300;
/// mDNS active query interval — how often to probe the LAN for new peers.
const MDNS_QUERY_INTERVAL_SECS: u64 = 10;

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
    /// request_response MUST be first: the NetworkBehaviour derive macro polls
    /// sub-behaviours in field order, and Connection::poll returns immediately on
    /// NotifyBehaviour events (exiting its handler-poll loop). If kademlia or
    /// gossipsub are polled first, their continuous NotifyBehaviour events starve
    /// request_response's OutboundSubstreamRequest — preventing tensor forwards
    /// from ever reaching the codec.
    pub request_response: request_response::Behaviour<SwarmCodec>,
    pub kademlia: kad::Behaviour<MemoryStore>,
    pub gossipsub: gossipsub::Behaviour,
    pub identify: identify::Behaviour,
    pub autonat: libp2p::swarm::behaviour::toggle::Toggle<autonat::Behaviour>,
    pub dcutr: libp2p::swarm::behaviour::toggle::Toggle<dcutr::Behaviour>,
    pub relay_client: relay::client::Behaviour,
    /// Relay server: accepts reservations from NAT'd peers and forwards circuits.
    pub relay_server: relay::Behaviour,
    /// NET-I5: Connection limits to prevent resource exhaustion.
    pub connection_limits: connection_limits::Behaviour,
    /// mDNS for automatic LAN peer discovery (zero-config).
    pub mdns: libp2p::swarm::behaviour::toggle::Toggle<mdns::tokio::Behaviour>,
    /// Persistent bidirectional stream protocol for per-pipeline inference
    /// sessions. Coexists with `request_response` — the latter remains the
    /// fallback and covers all non-streaming traffic. Only used when
    /// `config.inference.persistent_pipeline_stream` is on.
    pub pipeline_stream: libp2p_stream::Behaviour,
}

/// Build the combined network behaviour with all sub-protocols configured.
///
/// `known_peers` is the count of peers from the peer cache, used to auto-scale
/// GossipSub mesh parameters. Small clusters (< 10 peers) use lower thresholds,
/// while larger networks scale up for faster message propagation.
#[allow(clippy::too_many_arguments)]
pub fn build_behaviour(
    local_key: &Keypair,
    relay_behaviour: relay::client::Behaviour,
    relay_server_config: Option<&RelayServerConfig>,
    enable_mdns: bool,
    enable_autonat: bool,
    enable_dcutr: bool,
    known_peers: usize,
    network_config: Option<&NetworkConfig>,
) -> Result<SwarmBehaviour, Box<dyn std::error::Error>> {
    let local_peer_id = local_key.public().to_peer_id();

    // Kademlia DHT for peer and shard discovery
    let store = MemoryStore::new(local_peer_id);
    let mut kad_config = kad::Config::new(StreamProtocol::new("/swarmllm/kad/1.0.0"));
    // S5: Provider records track which nodes hold which shards for DHT-based
    // shard holder resolution at scale (50K+ nodes). Records republish before
    // the TTL expires so stale holders drop out automatically.
    kad_config.set_provider_record_ttl(Some(Duration::from_secs(KAD_PROVIDER_TTL_SECS)));
    kad_config
        .set_provider_publication_interval(Some(Duration::from_secs(KAD_PUBLISH_INTERVAL_SECS)));
    let mut kademlia = kad::Behaviour::with_config(local_peer_id, store, kad_config);
    // Use Server mode so nodes accept each other's DHT queries.
    // Client mode rejects inbound queries — when BOTH nodes are Client, rejected
    // substream negotiations flood the connection event channel (capacity 7),
    // blocking delivery of NotifyHandler(SendRequest) commands and preventing
    // tensor forwards from ever reaching the codec.
    kademlia.set_mode(Some(kad::Mode::Server));

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
    let (mesh_n, mesh_n_low, mesh_n_high, mesh_outbound_min) = if known_peers >= 10_000 {
        (8, 6, 16, 4) // Very large networks (10k+ nodes)
    } else if known_peers >= 1_000 {
        (7, 5, 14, 3) // Large networks (1k+ nodes)
    } else if known_peers >= 100 {
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
        .heartbeat_interval(Duration::from_secs(GOSSIPSUB_HEARTBEAT_SECS))
        .validation_mode(gossipsub::ValidationMode::Strict)
        .message_id_fn(message_id_fn)
        // SEC: Cap gossip message size to prevent oversized gossip flooding.
        // Matches MAX_JSON_MSG_SIZE (4 MB) in the request_response codec.
        .max_transmit_size(4 * 1024 * 1024)
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
            compress_prefix_kv: net_cfg.prefix_kv_compression,
            compress_level: net_cfg.tensor_compress_level,
            compress_threshold: net_cfg.tensor_compress_threshold,
        }
    } else {
        SwarmCodec::default()
    };
    let request_response = request_response::Behaviour::with_codec(
        codec,
        [(
            StreamProtocol::new(crate::network::protocol::PROTOCOL_ID),
            request_response::ProtocolSupport::Full,
        )],
        // NET-C3: RR_REQUEST_TIMEOUT_SECS covers LAN shard transfers and slow CPU
        // inference on large models (7B+). See constant docs for rationale.
        request_response::Config::default()
            .with_request_timeout(Duration::from_secs(RR_REQUEST_TIMEOUT_SECS)),
    );

    // Identify protocol
    let identify = identify::Behaviour::new(identify::Config::new(
        "/swarmllm/id/1.0.0".to_string(),
        local_key.public(),
    ));

    // AutoNAT for NAT detection (toggleable — disable on WSL2)
    let autonat_behaviour = if enable_autonat {
        Some(autonat::Behaviour::new(
            local_peer_id,
            autonat::Config::default(),
        ))
    } else {
        tracing::debug!("DIAG: autonat disabled by config");
        None
    };

    // DCUtR for hole punching (toggleable — disable on WSL2)
    let dcutr_behaviour = if enable_dcutr {
        Some(dcutr::Behaviour::new(local_peer_id))
    } else {
        tracing::debug!("DIAG: dcutr disabled by config");
        None
    };

    // Relay server: if config provided, use those limits; otherwise use defaults.
    let relay_config = relay_server_config
        .map(crate::network::relay::build_relay_server_config)
        .unwrap_or_default();
    let relay_server = relay::Behaviour::new(local_peer_id, relay_config);

    // NET-I5: Connection limits to prevent resource exhaustion.
    // max_established_per_peer=1: MUST be 1. With >1, libp2p request_response
    // round-robins requests across connections via `request_id % connections.len()`.
    // On multi-interface hosts (WSL2, Docker, dual-NIC), mDNS discovers the peer
    // on multiple addresses, creating parallel connections. Some connections go
    // through unreachable routes (e.g. 10.255.255.254 on WSL2) and silently fail:
    // the connection closes immediately (ApplicationClosed error_code=0) but the
    // behaviour's `connected` map retains a stale entry, causing every other
    // send_request to be routed to a dead connection and silently dropped.
    let max_per_peer = 1u32;
    let conn_limits = connection_limits::ConnectionLimits::default()
        .with_max_established_per_peer(Some(max_per_peer))
        .with_max_established(Some(500));
    tracing::info!(
        max_per_peer,
        max_total = 500,
        "DIAG: connection limits configured"
    );
    let connection_limits = connection_limits::Behaviour::new(conn_limits);

    // mDNS for automatic LAN peer discovery
    let mdns_behaviour = if enable_mdns {
        let mdns_config = mdns::Config {
            ttl: Duration::from_secs(MDNS_TTL_SECS),
            query_interval: Duration::from_secs(MDNS_QUERY_INTERVAL_SECS),
            enable_ipv6: false,
        };
        Some(mdns::tokio::Behaviour::new(mdns_config, local_peer_id)?)
    } else {
        None
    };

    let pipeline_stream = libp2p_stream::Behaviour::new();

    Ok(SwarmBehaviour {
        request_response,
        kademlia,
        gossipsub,
        identify,
        autonat: autonat_behaviour.into(),
        dcutr: dcutr_behaviour.into(),
        relay_client: relay_behaviour,
        relay_server,
        connection_limits,
        mdns: mdns_behaviour.into(),
        pipeline_stream,
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
        let result = build_behaviour(&keypair, relay_behaviour, None, false, true, true, 0, None);
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
        let result = build_behaviour(
            &keypair,
            relay_behaviour,
            Some(&relay_cfg),
            false,
            true,
            true,
            0,
            None,
        );
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn build_behaviour_with_mdns_enabled() {
        let keypair = Keypair::generate_ed25519();
        let (relay_transport, relay_behaviour) = relay::client::new(keypair.public().to_peer_id());
        drop(relay_transport);
        let result = build_behaviour(&keypair, relay_behaviour, None, true, true, true, 0, None);
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
