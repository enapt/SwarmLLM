use std::time::Duration;

use libp2p::connection_limits;
use libp2p::identity::Keypair;
use libp2p::kad::store::MemoryStore;
use libp2p::swarm::NetworkBehaviour;
use libp2p::{autonat, dcutr, gossipsub, identify, kad, relay, request_response, StreamProtocol};

use crate::network::protocol::{SwarmCodec, TensorCodec};
use crate::network::relay::RelayServerConfig;

/// Combined network behaviour for the SwarmLLM node.
///
/// Uses the libp2p `NetworkBehaviour` derive macro to combine multiple
/// sub-protocols into a single behaviour.
#[derive(NetworkBehaviour)]
pub struct SwarmBehaviour {
    pub kademlia: kad::Behaviour<MemoryStore>,
    pub gossipsub: gossipsub::Behaviour,
    pub request_response: request_response::Behaviour<SwarmCodec>,
    /// Cap'n Proto tensor protocol for zero-copy activation forwarding.
    pub tensor_rr: request_response::Behaviour<TensorCodec>,
    pub identify: identify::Behaviour,
    pub autonat: autonat::Behaviour,
    pub dcutr: dcutr::Behaviour,
    pub relay_client: relay::client::Behaviour,
    /// Relay server: accepts reservations from NAT'd peers and forwards circuits.
    pub relay_server: relay::Behaviour,
    /// NET-I5: Connection limits to prevent resource exhaustion.
    pub connection_limits: connection_limits::Behaviour,
}

/// Build the combined network behaviour with all sub-protocols configured.
pub fn build_behaviour(
    local_key: &Keypair,
    relay_behaviour: relay::client::Behaviour,
    relay_server_config: Option<&RelayServerConfig>,
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
    let gossipsub_config = gossipsub::ConfigBuilder::default()
        .heartbeat_interval(Duration::from_secs(10))
        .validation_mode(gossipsub::ValidationMode::Strict)
        .message_id_fn(message_id_fn)
        // Lower mesh thresholds so small clusters (2-3 nodes) can form a mesh.
        // Defaults are mesh_n=6, mesh_n_low=4, mesh_n_high=12 — too high for
        // dev/small deployments.
        .mesh_n(2)
        .mesh_n_low(1)
        .mesh_n_high(4)
        // NET-M2: At least 1 outbound mesh peer for message delivery
        .mesh_outbound_min(1)
        .build()
        .map_err(|e| format!("GossipSub config error: {e}"))?;
    let gossipsub = gossipsub::Behaviour::new(
        gossipsub::MessageAuthenticity::Signed(local_key.clone()),
        gossipsub_config,
    )
    .map_err(|e| format!("GossipSub init error: {e}"))?;

    // Request/Response for direct peer communication (shard transfers, control messages)
    // NET-C3: 300s timeout for shard transfers (large files need more time)
    let request_response = request_response::Behaviour::new(
        [(
            StreamProtocol::new("/swarmllm/1.0.0"),
            request_response::ProtocolSupport::Full,
        )],
        request_response::Config::default().with_request_timeout(Duration::from_secs(300)),
    );

    // Tensor request/response for zero-copy activation forwarding (Cap'n Proto)
    let tensor_rr = request_response::Behaviour::new(
        [(
            StreamProtocol::new("/swarmllm/tensor/1.0.0"),
            request_response::ProtocolSupport::Full,
        )],
        request_response::Config::default().with_request_timeout(Duration::from_secs(120)),
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

    // NET-I5: Connection limits to prevent resource exhaustion
    let conn_limits = connection_limits::ConnectionLimits::default()
        .with_max_established_per_peer(Some(2))
        .with_max_established(Some(500));
    let connection_limits = connection_limits::Behaviour::new(conn_limits);

    Ok(SwarmBehaviour {
        kademlia,
        gossipsub,
        request_response,
        tensor_rr,
        identify,
        autonat,
        dcutr,
        relay_client: relay_behaviour,
        relay_server,
        connection_limits,
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
        let result = build_behaviour(&keypair, relay_behaviour, None);
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
        let result = build_behaviour(&keypair, relay_behaviour, Some(&relay_cfg));
        assert!(result.is_ok());
    }
}
