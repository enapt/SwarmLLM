use std::sync::Arc;

use futures::StreamExt;
use libp2p::gossipsub::{self, IdentTopic};
use libp2p::request_response;
use libp2p::swarm::SwarmEvent;
use libp2p::{Multiaddr, Swarm, SwarmBuilder};
use tokio::sync::{mpsc, watch};

use crate::config::Config;
use crate::daemon::SharedState;
use crate::error::SwarmError;
use crate::identity::Identity;
use crate::network::behaviour::{self, SwarmBehaviour, SwarmBehaviourEvent};
use crate::network::discovery;
use crate::network::protocol::{self, SwarmRequest, SwarmResponse, TOPIC_MODELS};
use crate::network::transport;
use crate::types::{PeerInfo, SwarmMessage};

/// NetworkManager owns the libp2p Swarm and is the sole interface to the P2P network.
pub struct NetworkManager {
    shared_state: Arc<SharedState>,
    swarm: Swarm<SwarmBehaviour>,
    inbound_rx: mpsc::Receiver<SwarmMessage>,
    outbound_tx: mpsc::Sender<SwarmMessage>,
    shutdown_rx: watch::Receiver<bool>,
}

impl NetworkManager {
    /// Create a new NetworkManager and initialize the libp2p Swarm.
    pub fn new(
        shared_state: Arc<SharedState>,
        identity: &Identity,
        _config: &Config,
        inbound_rx: mpsc::Receiver<SwarmMessage>,
        outbound_tx: mpsc::Sender<SwarmMessage>,
        shutdown_rx: watch::Receiver<bool>,
    ) -> Result<Self, SwarmError> {
        let keypair = transport::ed25519_to_libp2p_keypair(identity.signing_key_bytes())?;
        let peer_id = keypair.public().to_peer_id();

        tracing::info!(%peer_id, "Initializing network");

        let swarm = SwarmBuilder::with_existing_identity(keypair.clone())
            .with_tokio()
            .with_quic()
            .with_relay_client(libp2p::noise::Config::new, libp2p::yamux::Config::default)
            .map_err(|e| SwarmError::Network(format!("Relay client error: {e}")))?
            .with_behaviour(|_key, relay_behaviour| {
                behaviour::build_behaviour(&keypair, relay_behaviour).map_err(|e| {
                    Box::new(std::io::Error::other(e.to_string()))
                        as Box<dyn std::error::Error + Send + Sync>
                })
            })
            .map_err(|e| SwarmError::Network(format!("Behaviour error: {e}")))?
            .with_swarm_config(|c| {
                c.with_idle_connection_timeout(std::time::Duration::from_secs(60))
            })
            .build();

        Ok(Self {
            shared_state,
            swarm,
            inbound_rx,
            outbound_tx,
            shutdown_rx,
        })
    }

    /// Start the network manager event loop.
    pub async fn run(mut self) -> Result<(), SwarmError> {
        let config = self.shared_state.config.clone();
        let port = config.node.listen_port;

        // Listen on QUIC and TCP
        let quic_addr: Multiaddr = format!("/ip4/0.0.0.0/udp/{port}/quic-v1")
            .parse()
            .map_err(|e| SwarmError::Network(format!("Invalid QUIC address: {e}")))?;
        let tcp_addr: Multiaddr = format!("/ip4/0.0.0.0/tcp/{port}")
            .parse()
            .map_err(|e| SwarmError::Network(format!("Invalid TCP address: {e}")))?;

        self.swarm
            .listen_on(quic_addr.clone())
            .map_err(|e| SwarmError::Network(format!("Failed to listen on QUIC: {e}")))?;
        self.swarm
            .listen_on(tcp_addr.clone())
            .map_err(|e| SwarmError::Network(format!("Failed to listen on TCP: {e}")))?;

        tracing::info!(%quic_addr, %tcp_addr, "Listening for P2P connections");

        // Subscribe to GossipSub topics
        discovery::subscribe_topics(&mut self.swarm)?;

        // Bootstrap with configured peers
        let bootstrap_count =
            discovery::bootstrap_peers(&mut self.swarm, &config.network.bootstrap_peers)?;
        if bootstrap_count > 0 {
            discovery::trigger_bootstrap(&mut self.swarm)?;
        }

        // Periodic discovery timer
        let mut discovery_interval = tokio::time::interval(discovery::DISCOVERY_INTERVAL);
        discovery_interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

        tracing::info!("NetworkManager running");

        loop {
            tokio::select! {
                // Shutdown signal
                _ = self.shutdown_rx.changed() => {
                    if *self.shutdown_rx.borrow() {
                        tracing::info!("NetworkManager shutting down");
                        break;
                    }
                }
                // Swarm events from the network
                event = self.swarm.select_next_some() => {
                    self.handle_swarm_event(event).await;
                }
                // Outbound messages from other daemon tasks
                msg = self.inbound_rx.recv() => {
                    match msg {
                        Some(msg) => self.handle_outbound(msg).await,
                        None => {
                            tracing::info!("Inbound channel closed, shutting down");
                            break;
                        }
                    }
                }
                // Periodic discovery
                _ = discovery_interval.tick() => {
                    let _ = discovery::trigger_bootstrap(&mut self.swarm);
                    self.update_peer_count();
                }
            }
        }

        Ok(())
    }

    async fn handle_swarm_event(&mut self, event: SwarmEvent<SwarmBehaviourEvent>) {
        match event {
            SwarmEvent::Behaviour(SwarmBehaviourEvent::Gossipsub(gossipsub::Event::Message {
                propagation_source,
                message,
                ..
            })) => match protocol::decode_message(&message.data) {
                Ok(msg) => {
                    tracing::debug!(
                        source = %propagation_source,
                        "Received GossipSub message"
                    );
                    if let Err(e) = self.outbound_tx.send(msg).await {
                        tracing::warn!(error = %e, "Failed to forward gossipsub message");
                    }
                }
                Err(e) => {
                    tracing::debug!(error = %e, "Failed to decode gossipsub message");
                }
            },

            SwarmEvent::Behaviour(SwarmBehaviourEvent::RequestResponse(
                request_response::Event::Message { peer, message },
            )) => match message {
                request_response::Message::Request {
                    request, channel, ..
                } => {
                    tracing::debug!(%peer, "Received request");
                    self.handle_request(peer, request, channel).await;
                }
                request_response::Message::Response { response, .. } => {
                    tracing::debug!(%peer, "Received response");
                    self.handle_response(peer, response).await;
                }
            },

            SwarmEvent::Behaviour(SwarmBehaviourEvent::Identify(
                libp2p::identify::Event::Received {
                    peer_id,
                    info,
                    connection_id: _,
                },
            )) => {
                tracing::debug!(
                    %peer_id,
                    protocol_version = %info.protocol_version,
                    "Identified peer"
                );
                // Add addresses to Kademlia
                for addr in &info.listen_addrs {
                    self.swarm
                        .behaviour_mut()
                        .kademlia
                        .add_address(&peer_id, addr.clone());
                }
                // Update peer info — use a deterministic NodeId from the peer_id bytes
                let peer_bytes = peer_id.to_bytes();
                let mut node_id_bytes = [0u8; 32];
                let copy_len = peer_bytes.len().min(32);
                node_id_bytes[..copy_len].copy_from_slice(&peer_bytes[..copy_len]);
                let node_id = crate::types::NodeId(node_id_bytes);

                let peer_info = PeerInfo {
                    node_id: node_id.clone(),
                    addresses: info.listen_addrs.iter().map(|a| a.to_string()).collect(),
                    capability: None,
                    last_seen: chrono::Utc::now(),
                    latency_ms: None,
                    trust_score: 0.5,
                };
                self.shared_state.peer_registry.insert(node_id, peer_info);
            }

            SwarmEvent::Behaviour(SwarmBehaviourEvent::Kademlia(
                libp2p::kad::Event::OutboundQueryProgressed { result, .. },
            )) => {
                tracing::debug!(?result, "Kademlia query progressed");
            }

            SwarmEvent::ConnectionEstablished { peer_id, .. } => {
                tracing::info!(%peer_id, "Connection established");
                self.update_peer_count();
            }

            SwarmEvent::ConnectionClosed { peer_id, cause, .. } => {
                tracing::info!(%peer_id, ?cause, "Connection closed");
                self.update_peer_count();
            }

            SwarmEvent::NewListenAddr { address, .. } => {
                tracing::info!(%address, "New listen address");
            }

            _ => {}
        }
    }

    async fn handle_request(
        &mut self,
        peer: libp2p::PeerId,
        request: SwarmRequest,
        channel: request_response::ResponseChannel<SwarmResponse>,
    ) {
        match request {
            SwarmRequest::Message(msg) => {
                tracing::debug!(%peer, "Handling protocol message request");
                if let Err(e) = self.outbound_tx.send(msg).await {
                    tracing::warn!(error = %e, "Failed to forward request message");
                }
                // Send ACK
                let _ = self
                    .swarm
                    .behaviour_mut()
                    .request_response
                    .send_response(channel, SwarmResponse::Ack);
            }
            SwarmRequest::ShardTransfer(shard_req) => {
                tracing::info!(
                    %peer,
                    model = %shard_req.shard_id.model_id,
                    index = shard_req.shard_id.index,
                    "Shard transfer request"
                );
                // Shard transfer is handled by reading from disk
                // For now, respond with empty data (full implementation in model/distribution)
                let _ = self.swarm.behaviour_mut().request_response.send_response(
                    channel,
                    SwarmResponse::ShardData(crate::types::ShardResponse {
                        data: vec![],
                        total_size: 0,
                    }),
                );
            }
        }
    }

    async fn handle_response(&mut self, peer: libp2p::PeerId, response: SwarmResponse) {
        match response {
            SwarmResponse::Message(msg) => {
                if let Err(e) = self.outbound_tx.send(msg).await {
                    tracing::warn!(%peer, error = %e, "Failed to forward response message");
                }
            }
            SwarmResponse::ShardData(data) => {
                tracing::info!(%peer, size = data.total_size, "Received shard data");
                // Route to model distribution handler
            }
            SwarmResponse::Ack => {
                tracing::debug!(%peer, "Received ACK");
            }
        }
    }

    async fn handle_outbound(&mut self, msg: SwarmMessage) {
        // Publish to GossipSub topic
        let topic = match &msg {
            SwarmMessage::ShardAnnounce(_) | SwarmMessage::NodeCapabilityUpdate(_) => TOPIC_MODELS,
            SwarmMessage::ModelVote(_) => crate::network::protocol::TOPIC_GOVERNANCE,
            SwarmMessage::HealthPing { .. } | SwarmMessage::HealthPong { .. } => {
                crate::network::protocol::TOPIC_HEALTH
            }
            // Inference and credit messages go via request_response, not gossipsub
            _ => return,
        };

        match protocol::encode_message(&msg) {
            Ok(data) => {
                let gossip_topic = IdentTopic::new(topic);
                match self
                    .swarm
                    .behaviour_mut()
                    .gossipsub
                    .publish(gossip_topic, data)
                {
                    Ok(_) => tracing::debug!(topic, "Published message to GossipSub"),
                    Err(e) => tracing::debug!(topic, error = %e, "Failed to publish to GossipSub"),
                }
            }
            Err(e) => {
                tracing::warn!(error = %e, "Failed to encode outbound message");
            }
        }
    }

    fn update_peer_count(&self) {
        let count = self.shared_state.peer_registry.len() as u32;
        if let Ok(mut stats) = self.shared_state.node_stats.try_write() {
            stats.peers_connected = count;
        }
    }
}
