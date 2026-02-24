use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::time::Duration;

use libp2p::identity::Keypair;
use libp2p::kad::store::MemoryStore;
use libp2p::swarm::NetworkBehaviour;
use libp2p::{autonat, dcutr, gossipsub, identify, kad, relay, request_response, StreamProtocol};

use crate::network::protocol::SwarmCodec;

/// Combined network behaviour for the SwarmLLM node.
///
/// Uses the libp2p `NetworkBehaviour` derive macro to combine multiple
/// sub-protocols into a single behaviour.
#[derive(NetworkBehaviour)]
pub struct SwarmBehaviour {
    pub kademlia: kad::Behaviour<MemoryStore>,
    pub gossipsub: gossipsub::Behaviour,
    pub request_response: request_response::Behaviour<SwarmCodec>,
    pub identify: identify::Behaviour,
    pub autonat: autonat::Behaviour,
    pub dcutr: dcutr::Behaviour,
    pub relay_client: relay::client::Behaviour,
}

/// Build the combined network behaviour with all sub-protocols configured.
pub fn build_behaviour(
    local_key: &Keypair,
    relay_behaviour: relay::client::Behaviour,
) -> Result<SwarmBehaviour, Box<dyn std::error::Error>> {
    let local_peer_id = local_key.public().to_peer_id();

    // Kademlia DHT for peer and shard discovery
    let store = MemoryStore::new(local_peer_id);
    let mut kademlia = kad::Behaviour::new(local_peer_id, store);
    kademlia.set_mode(Some(kad::Mode::Server));

    // GossipSub for network-wide announcements
    let message_id_fn = |message: &gossipsub::Message| {
        let mut hasher = DefaultHasher::new();
        message.data.hash(&mut hasher);
        message.source.hash(&mut hasher);
        gossipsub::MessageId::from(hasher.finish().to_string())
    };
    let gossipsub_config = gossipsub::ConfigBuilder::default()
        .heartbeat_interval(Duration::from_secs(10))
        .validation_mode(gossipsub::ValidationMode::Strict)
        .message_id_fn(message_id_fn)
        .build()
        .map_err(|e| format!("GossipSub config error: {e}"))?;
    let gossipsub = gossipsub::Behaviour::new(
        gossipsub::MessageAuthenticity::Signed(local_key.clone()),
        gossipsub_config,
    )
    .map_err(|e| format!("GossipSub init error: {e}"))?;

    // Request/Response for direct peer communication (shard transfers, inference pipeline)
    let request_response = request_response::Behaviour::new(
        [(
            StreamProtocol::new("/swarmllm/1.0.0"),
            request_response::ProtocolSupport::Full,
        )],
        request_response::Config::default(),
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

    Ok(SwarmBehaviour {
        kademlia,
        gossipsub,
        request_response,
        identify,
        autonat,
        dcutr,
        relay_client: relay_behaviour,
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
        let result = build_behaviour(&keypair, relay_behaviour);
        assert!(result.is_ok());
    }
}
