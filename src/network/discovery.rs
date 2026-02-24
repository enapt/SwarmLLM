use std::time::Duration;

use libp2p::kad;
use libp2p::kad::RecordKey;
use libp2p::swarm::Swarm;
use libp2p::Multiaddr;

use crate::error::SwarmError;
use crate::network::behaviour::SwarmBehaviour;
use crate::types::{NodeCapability, NodeId, ShardId};

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
                if let Some(libp2p::multiaddr::Protocol::P2p(peer_id)) = addr.iter().last() {
                    swarm
                        .behaviour_mut()
                        .kademlia
                        .add_address(&peer_id, addr.clone());
                }

                match swarm.dial(addr.clone()) {
                    Ok(_) => {
                        tracing::info!(addr = %addr, "Dialing bootstrap peer");
                        dialed += 1;
                    }
                    Err(e) => {
                        tracing::warn!(addr = %addr, error = %e, "Failed to dial bootstrap peer");
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
            tracing::debug!(error = %e, "Kademlia bootstrap failed (no known peers)");
            Ok(())
        }
    }
}

/// Subscribe to the standard GossipSub topics.
pub fn subscribe_topics(swarm: &mut Swarm<SwarmBehaviour>) -> Result<(), SwarmError> {
    use crate::network::protocol::{TOPIC_GOVERNANCE, TOPIC_HEALTH, TOPIC_MODELS};
    use libp2p::gossipsub::IdentTopic;

    let topics = [TOPIC_MODELS, TOPIC_GOVERNANCE, TOPIC_HEALTH];

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

/// Announce local node capability to the DHT.
pub fn announce_capability(
    swarm: &mut Swarm<SwarmBehaviour>,
    node_id: &NodeId,
    capability: &NodeCapability,
) -> Result<(), SwarmError> {
    let key = RecordKey::new(&format!("/swarm/node/{node_id}"));
    let value = serde_json::to_vec(capability).map_err(|e| SwarmError::Network(e.to_string()))?;

    let record = kad::Record {
        key,
        value,
        publisher: None,
        expires: None,
    };

    swarm
        .behaviour_mut()
        .kademlia
        .put_record(record, kad::Quorum::One)
        .map_err(|e| SwarmError::Network(format!("Failed to put capability record: {e}")))?;

    tracing::debug!(node_id = %node_id, "Announced capability to DHT");
    Ok(())
}

/// Announce shard holdings to the DHT.
pub fn announce_shards(
    swarm: &mut Swarm<SwarmBehaviour>,
    node_id: &NodeId,
    shards: &[ShardId],
) -> Result<(), SwarmError> {
    for shard in shards {
        let key = RecordKey::new(&format!("/swarm/shard/{}/{}", shard.model_id, shard.index));
        let value = serde_json::to_vec(node_id).map_err(|e| SwarmError::Network(e.to_string()))?;

        let record = kad::Record {
            key,
            value,
            publisher: None,
            expires: None,
        };

        swarm
            .behaviour_mut()
            .kademlia
            .put_record(record, kad::Quorum::One)
            .map_err(|e| SwarmError::Network(format!("Failed to announce shard: {e}")))?;
    }

    if !shards.is_empty() {
        tracing::info!(count = shards.len(), "Announced shards to DHT");
    }

    Ok(())
}

/// Discovery interval for periodic peer discovery.
pub const DISCOVERY_INTERVAL: Duration = Duration::from_secs(60);
