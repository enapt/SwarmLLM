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
use crate::network::protocol::{
    self, SwarmRequest, SwarmResponse, TensorRequest, TensorResponse, TOPIC_MODELS,
};
use crate::network::relay::RelayServerConfig;
use crate::network::transport;
use crate::types::{NetworkCommand, PeerInfo, SwarmMessage};

/// NetworkManager owns the libp2p Swarm and is the sole interface to the P2P network.
pub struct NetworkManager {
    shared_state: Arc<SharedState>,
    swarm: Swarm<SwarmBehaviour>,
    /// Receives commands from daemon tasks (broadcast, send tensor, etc.)
    inbound_rx: mpsc::Receiver<NetworkCommand>,
    /// Sends decoded network messages to the dispatcher for routing.
    outbound_tx: mpsc::Sender<SwarmMessage>,
    shutdown_rx: watch::Receiver<bool>,
}

impl NetworkManager {
    /// Create a new NetworkManager and initialize the libp2p Swarm.
    pub fn new(
        shared_state: Arc<SharedState>,
        identity: &Identity,
        config: &Config,
        inbound_rx: mpsc::Receiver<NetworkCommand>,
        outbound_tx: mpsc::Sender<SwarmMessage>,
        shutdown_rx: watch::Receiver<bool>,
    ) -> Result<Self, SwarmError> {
        let keypair = transport::ed25519_to_libp2p_keypair(identity.signing_key_bytes())?;
        let peer_id = keypair.public().to_peer_id();

        tracing::info!(%peer_id, "Initializing network");

        // Build relay server config if relay serving is enabled
        let relay_server_config = if config.network.enable_relay {
            Some(RelayServerConfig {
                max_circuits: config.network.relay_max_circuits,
                max_circuit_duration: std::time::Duration::from_secs(
                    config.network.relay_max_circuit_duration_secs,
                ),
                ..Default::default()
            })
        } else {
            None
        };

        let kp_clone = keypair.clone();
        let relay_cfg = relay_server_config;
        let swarm = SwarmBuilder::with_existing_identity(keypair)
            .with_tokio()
            .with_quic()
            .with_relay_client(libp2p::noise::Config::new, libp2p::yamux::Config::default)
            .map_err(|e| SwarmError::Network(format!("Relay client error: {e}")))?
            .with_behaviour(|_key, relay_behaviour| {
                behaviour::build_behaviour(&kp_clone, relay_behaviour, relay_cfg.as_ref()).map_err(
                    |e| {
                        Box::new(std::io::Error::other(e.to_string()))
                            as Box<dyn std::error::Error + Send + Sync>
                    },
                )
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

        // TCP listen is best-effort — may not be available depending on transport config
        match self.swarm.listen_on(tcp_addr.clone()) {
            Ok(_) => tracing::info!(%quic_addr, %tcp_addr, "Listening for P2P connections"),
            Err(e) => {
                tracing::warn!(%quic_addr, error = %e, "TCP listen unavailable, using QUIC only");
            }
        }

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
                // Outbound commands from other daemon tasks
                cmd = self.inbound_rx.recv() => {
                    match cmd {
                        Some(cmd) => self.handle_outbound_command(cmd).await,
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
            // ── GossipSub messages ──
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

            // ── JSON request/response (control messages, shard transfers) ──
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

            // ── Tensor request/response (Cap'n Proto, zero-copy) ──
            SwarmEvent::Behaviour(SwarmBehaviourEvent::TensorRr(
                request_response::Event::Message { peer, message },
            )) => match message {
                request_response::Message::Request {
                    request, channel, ..
                } => {
                    tracing::debug!(%peer, "Received tensor request");
                    match protocol::decode_layer_forward(&request.payload) {
                        Ok(forward) => {
                            let msg = SwarmMessage::LayerForward(forward);
                            let _ = self.outbound_tx.send(msg).await;
                            // ACK the tensor request
                            let resp = TensorResponse {
                                payload: protocol::encode_ack(),
                            };
                            let _ = self
                                .swarm
                                .behaviour_mut()
                                .tensor_rr
                                .send_response(channel, resp);
                        }
                        Err(e) => {
                            tracing::warn!(error = %e, "Failed to decode tensor forward");
                        }
                    }
                }
                request_response::Message::Response { response, .. } => {
                    // Decode LayerResult from the tensor response
                    if response.payload.len() > 1 {
                        if let Ok(result) = protocol::decode_layer_result(&response.payload) {
                            let _ = self
                                .outbound_tx
                                .send(SwarmMessage::LayerResult(result))
                                .await;
                        }
                    }
                    // Single byte = ACK, ignore
                }
            },

            // ── Relay server events ──
            SwarmEvent::Behaviour(SwarmBehaviourEvent::RelayServer(event)) => {
                crate::network::relay::handle_relay_server_event(event);
            }

            // ── AutoNAT status changes ──
            SwarmEvent::Behaviour(SwarmBehaviourEvent::Autonat(
                libp2p::autonat::Event::StatusChanged { old, new },
            )) => {
                tracing::info!(?old, ?new, "AutoNAT status changed");
                if let Ok(mut stats) = self.shared_state.node_stats.try_write() {
                    stats.nat_status = Some(format!("{new:?}"));
                }
            }

            // ── Identify ──
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
                // Derive NodeId from the peer's Ed25519 public key (32 bytes)
                // per spec: NodeId(verifying_key.to_bytes())
                let node_id = if let Ok(ed_key) = info.public_key.clone().try_into_ed25519() {
                    crate::types::NodeId(ed_key.to_bytes())
                } else {
                    // Fallback for non-Ed25519 keys: hash the peer_id
                    let hash = blake3::hash(&peer_id.to_bytes());
                    crate::types::NodeId(*hash.as_bytes())
                };

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

            // ── Kademlia ──
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
                if let Err(e) = self.outbound_tx.send(*msg).await {
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
                if let Err(e) = self.outbound_tx.send(*msg).await {
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

    /// Handle outbound commands from daemon tasks.
    async fn handle_outbound_command(&mut self, cmd: NetworkCommand) {
        match cmd {
            NetworkCommand::Broadcast(msg) => {
                self.handle_broadcast(msg).await;
            }
            NetworkCommand::SendTensor {
                target_peer_bytes,
                forward,
            } => {
                self.handle_send_tensor(target_peer_bytes, forward);
            }
            NetworkCommand::SendTensorResult {
                target_peer_bytes,
                result,
            } => {
                self.handle_send_tensor_result(target_peer_bytes, result);
            }
        }
    }

    /// Broadcast a message via GossipSub.
    async fn handle_broadcast(&mut self, msg: SwarmMessage) {
        let topic = match &msg {
            SwarmMessage::ShardAnnounce(_) | SwarmMessage::NodeCapabilityUpdate(_) => TOPIC_MODELS,
            SwarmMessage::CreditGossip(_) => crate::network::protocol::TOPIC_CREDITS,
            SwarmMessage::ModelVote(_) => crate::network::protocol::TOPIC_GOVERNANCE,
            SwarmMessage::HealthPing { .. } | SwarmMessage::HealthPong { .. } => {
                crate::network::protocol::TOPIC_HEALTH
            }
            // Governance Phase 7 messages
            SwarmMessage::Proposal(_)
            | SwarmMessage::ProposalAmendment(_)
            | SwarmMessage::ProposalStatusChange(_) => {
                crate::network::protocol::TOPIC_GOV_PROPOSALS
            }
            SwarmMessage::ProposalVote(_) => crate::network::protocol::TOPIC_GOV_VOTES,
            SwarmMessage::Issue(_)
            | SwarmMessage::IssueComment(_)
            | SwarmMessage::IssueStatusChange(_)
            | SwarmMessage::IssueUpvote(_) => crate::network::protocol::TOPIC_GOV_ISSUES,
            SwarmMessage::ReleaseCandidate(_)
            | SwarmMessage::TestReport(_)
            | SwarmMessage::ReleaseApproval(_) => crate::network::protocol::TOPIC_GOV_RELEASES,
            SwarmMessage::ChangelogEntry(_) => crate::network::protocol::TOPIC_GOV_CHANGELOG,
            // Inference and credit transaction messages go via request_response, not gossipsub
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

    /// Send a tensor forward to a specific peer via the Cap'n Proto tensor protocol.
    fn handle_send_tensor(
        &mut self,
        target_peer_bytes: Vec<u8>,
        forward: crate::types::LayerForward,
    ) {
        let peer_id = match libp2p::PeerId::from_bytes(&target_peer_bytes) {
            Ok(id) => id,
            Err(e) => {
                tracing::warn!(error = %e, "Invalid peer ID bytes for tensor send");
                return;
            }
        };

        match protocol::encode_layer_forward(&forward) {
            Ok(payload) => {
                let req = TensorRequest { payload };
                self.swarm
                    .behaviour_mut()
                    .tensor_rr
                    .send_request(&peer_id, req);
                tracing::debug!(
                    %peer_id,
                    request_id = %forward.request_id,
                    seq = forward.sequence_num,
                    "Sent tensor forward"
                );
            }
            Err(e) => {
                tracing::warn!(error = %e, "Failed to encode tensor forward");
            }
        }
    }

    /// Send a tensor result to a specific peer via the Cap'n Proto tensor protocol.
    fn handle_send_tensor_result(
        &mut self,
        target_peer_bytes: Vec<u8>,
        result: crate::types::LayerResult,
    ) {
        let peer_id = match libp2p::PeerId::from_bytes(&target_peer_bytes) {
            Ok(id) => id,
            Err(e) => {
                tracing::warn!(error = %e, "Invalid peer ID bytes for tensor result");
                return;
            }
        };

        match protocol::encode_layer_result(&result) {
            Ok(payload) => {
                let req = TensorRequest { payload };
                self.swarm
                    .behaviour_mut()
                    .tensor_rr
                    .send_request(&peer_id, req);
                tracing::debug!(
                    %peer_id,
                    request_id = %result.request_id,
                    "Sent tensor result"
                );
            }
            Err(e) => {
                tracing::warn!(error = %e, "Failed to encode tensor result");
            }
        }
    }

    fn update_peer_count(&mut self) {
        let count = self.swarm.connected_peers().count() as u32;
        if let Ok(mut stats) = self.shared_state.node_stats.try_write() {
            stats.peers_connected = count;
        }
    }
}
