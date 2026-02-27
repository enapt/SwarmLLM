use std::collections::HashMap;
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
use crate::model::acquisition::AcquisitionCommand;
use crate::model::shard::ShardStore;
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
    /// Sends shard data to the AcquisitionManager when received from peers.
    acquisition_tx: Option<mpsc::Sender<AcquisitionCommand>>,
    /// Shard store for serving shard data to peers.
    shard_store: ShardStore,
    /// Maps peer_id → shard_id for in-flight shard download requests.
    pending_shard_requests: HashMap<libp2p::PeerId, crate::types::ShardId>,
    /// Tracks bytes downloaded so far per shard for chunked transfers.
    shard_download_progress: HashMap<crate::types::ShardId, u64>,
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
        acquisition_tx: Option<mpsc::Sender<AcquisitionCommand>>,
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

        let shard_store = ShardStore::new(&config.node.data_dir);

        Ok(Self {
            shared_state,
            swarm,
            inbound_rx,
            outbound_tx,
            acquisition_tx,
            shard_store,
            pending_shard_requests: HashMap::new(),
            shard_download_progress: HashMap::new(),
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
            })) => {
                // Try to unseal gossip (encrypted), then try plaintext JSON decode.
                // This handles key mismatches between bootstrap and joining nodes
                // as well as pre-encryption upgrade nodes.
                let decoded = self
                    .shared_state
                    .gossip_sealer
                    .open(&message.data)
                    .and_then(|plaintext| {
                        protocol::decode_message(&plaintext).map_err(|e| e.into())
                    })
                    .or_else(|_| protocol::decode_message(&message.data));

                match decoded {
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
                }
            }

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
                    // Dispatch based on the message type tag (first byte)
                    let tag = request.payload.first().copied().unwrap_or(0);
                    match tag {
                        protocol::TENSOR_TAG_FORWARD => {
                            tracing::debug!(%peer, "Received tensor LayerForward");
                            match protocol::decode_layer_forward(&request.payload) {
                                Ok(mut forward) => {
                                    // Attach sender's PeerId so the dispatcher can route back the result
                                    forward.sender_peer_bytes = Some(peer.to_bytes());
                                    let msg = SwarmMessage::LayerForward(forward);
                                    let _ = self.outbound_tx.send(msg).await;
                                }
                                Err(e) => {
                                    tracing::warn!(error = %e, "Failed to decode tensor forward");
                                }
                            }
                        }
                        protocol::TENSOR_TAG_RESULT => {
                            tracing::debug!(%peer, "Received tensor LayerResult");
                            match protocol::decode_layer_result(&request.payload) {
                                Ok(result) => {
                                    let _ = self
                                        .outbound_tx
                                        .send(SwarmMessage::LayerResult(result))
                                        .await;
                                }
                                Err(e) => {
                                    tracing::warn!(error = %e, "Failed to decode tensor result");
                                }
                            }
                        }
                        protocol::TENSOR_TAG_ENCRYPTED => {
                            tracing::debug!(%peer, "Received encrypted tensor");
                            match protocol::decode_layer_forward_encrypted(&request.payload) {
                                Ok((mut forward, sealed, aad)) => {
                                    // Find the sender's NodeId to decrypt
                                    let sender_node_id = self.find_node_id_for_peer(&peer);
                                    if let Some(node_id) = sender_node_id {
                                        match self
                                            .shared_state
                                            .session_manager
                                            .open(&node_id, &sealed, &aad)
                                        {
                                            Ok(plaintext) => {
                                                forward.activations = plaintext;
                                                forward.sender_peer_bytes = Some(peer.to_bytes());
                                                let _ = self
                                                    .outbound_tx
                                                    .send(SwarmMessage::LayerForward(forward))
                                                    .await;
                                            }
                                            Err(e) => {
                                                tracing::warn!(
                                                    error = %e,
                                                    %peer,
                                                    "Failed to decrypt tensor — dropping"
                                                );
                                            }
                                        }
                                    } else {
                                        tracing::warn!(
                                            %peer,
                                            "Encrypted tensor from unknown peer — dropping"
                                        );
                                    }
                                }
                                Err(e) => {
                                    tracing::warn!(error = %e, "Failed to decode encrypted tensor");
                                }
                            }
                        }
                        _ => {
                            tracing::warn!(%peer, tag, "Unknown tensor message tag");
                        }
                    }
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
                request_response::Message::Response { response, .. } => {
                    // Check for LayerResult in response (legacy path)
                    if response.payload.len() > 1 {
                        let tag = response.payload[0];
                        if tag == protocol::TENSOR_TAG_RESULT {
                            if let Ok(result) = protocol::decode_layer_result(&response.payload) {
                                let _ = self
                                    .outbound_tx
                                    .send(SwarmMessage::LayerResult(result))
                                    .await;
                            }
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

                // Establish encryption session from the peer's Ed25519 public key
                if let Ok(ed_key) = info.public_key.clone().try_into_ed25519() {
                    if let Some(x25519_pub) =
                        crate::crypto::session::ed25519_pubkey_to_x25519(&ed_key.to_bytes())
                    {
                        self.shared_state
                            .session_manager
                            .establish_session(&node_id, x25519_pub);
                    }
                }

                let peer_info = PeerInfo {
                    node_id: node_id.clone(),
                    addresses: info.listen_addrs.iter().map(|a| a.to_string()).collect(),
                    capability: None,
                    last_seen: chrono::Utc::now(),
                    latency_ms: None,
                    trust_score: 0.5,
                    peer_id_bytes: Some(peer_id.to_bytes()),
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
                    offset = shard_req.chunk_offset,
                    chunk_size = shard_req.chunk_size,
                    "Shard transfer request"
                );

                let response = self.serve_shard_data(&shard_req);

                let _ = self
                    .swarm
                    .behaviour_mut()
                    .request_response
                    .send_response(channel, response);
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
                if data.data.is_empty() {
                    tracing::debug!(%peer, "Received empty shard response (peer doesn't have it)");
                    return;
                }
                tracing::info!(
                    %peer,
                    bytes = data.data.len(),
                    total_size = data.total_size,
                    "Received shard data chunk"
                );
                // Route to AcquisitionManager if we have a pending request
                if let Some(ref acq_tx) = self.acquisition_tx {
                    // Look up which shard this peer was sending us from pending requests
                    if let Some(shard_id) = self.pending_shard_requests.remove(&peer) {
                        let offset = self
                            .shard_download_progress
                            .get(&shard_id)
                            .copied()
                            .unwrap_or(0);
                        let chunk_len = data.data.len() as u64;

                        if let Err(e) = acq_tx
                            .send(AcquisitionCommand::ShardDataReceived {
                                shard_id: shard_id.clone(),
                                offset,
                                data: data.data,
                                total_size: data.total_size,
                            })
                            .await
                        {
                            tracing::warn!(error = %e, "Failed to forward shard data to acquisition");
                        }

                        // Update progress tracking
                        let new_offset = offset + chunk_len;
                        if new_offset < data.total_size {
                            // More chunks needed — re-register and request next chunk
                            self.shard_download_progress
                                .insert(shard_id.clone(), new_offset);
                            self.pending_shard_requests.insert(peer, shard_id.clone());

                            let next_req = crate::types::ShardRequest {
                                shard_id,
                                chunk_offset: new_offset,
                                chunk_size: 32 * 1024 * 1024, // 32MB chunks
                            };
                            let req = SwarmRequest::ShardTransfer(next_req);
                            self.swarm
                                .behaviour_mut()
                                .request_response
                                .send_request(&peer, req);
                        } else {
                            // Download complete for this shard
                            self.shard_download_progress.remove(&shard_id);
                            tracing::info!(
                                model = %shard_id.model_id,
                                index = shard_id.index,
                                "Shard download complete"
                            );
                        }
                    } else {
                        tracing::warn!(
                            %peer,
                            "Received shard data but no pending request found for peer"
                        );
                    }
                }
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
            NetworkCommand::SendStreamingToken {
                target_peer_bytes,
                token,
            } => {
                self.handle_send_streaming_token(target_peer_bytes, token);
            }
            NetworkCommand::SendShardRequest {
                target_peer_bytes,
                request,
            } => {
                self.handle_send_shard_request(target_peer_bytes, request);
            }
        }
    }

    /// Broadcast a message via GossipSub.
    async fn handle_broadcast(&mut self, msg: SwarmMessage) {
        let topic = match &msg {
            SwarmMessage::ShardAnnounce(_)
            | SwarmMessage::NodeCapabilityUpdate(_)
            | SwarmMessage::ModelManifest(_)
            | SwarmMessage::ShardDownloadProgress(_)
            | SwarmMessage::HfSourceGossip(_) => TOPIC_MODELS,
            SwarmMessage::CreditGossip(_) => crate::network::protocol::TOPIC_CREDITS,
            SwarmMessage::ModelVote(_) => crate::network::protocol::TOPIC_GOVERNANCE,
            SwarmMessage::HealthPing { .. } | SwarmMessage::HealthPong { .. } => {
                crate::network::protocol::TOPIC_HEALTH
            }
            SwarmMessage::NicknameGossip(_) => crate::network::protocol::TOPIC_IDENTITY,
            SwarmMessage::PoolMessage(_) => crate::network::protocol::TOPIC_POOLS,
            // Inference and credit transaction messages go via request_response, not gossipsub
            _ => return,
        };

        match protocol::encode_message(&msg) {
            Ok(data) => {
                // Publish plaintext JSON — gossip messages (shard announces, health,
                // nicknames) are inherently public. Unicast messages use pairwise
                // session encryption for privacy.
                let publish_data = data;

                let gossip_topic = IdentTopic::new(topic);
                match self
                    .swarm
                    .behaviour_mut()
                    .gossipsub
                    .publish(gossip_topic, publish_data)
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
    /// Encrypts activations when an encryption session exists, falls back to plaintext.
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

        // Try to find the peer's NodeId for encryption
        let peer_node_id = self.find_node_id_for_peer(&peer_id);
        let use_encryption = peer_node_id
            .as_ref()
            .is_some_and(|nid| self.shared_state.session_manager.has_session(nid));

        let payload = if use_encryption {
            let node_id = peer_node_id.unwrap();
            // Build the AAD from the cleartext header fields
            let mut aad = Vec::with_capacity(25);
            aad.extend_from_slice(forward.request_id.as_bytes());
            aad.extend_from_slice(&forward.sequence_num.to_le_bytes());
            aad.extend_from_slice(&forward.index_pos.to_le_bytes());
            let fmt_tag: u8 = match forward.format {
                crate::types::TensorFormat::FP16 => 0,
                crate::types::TensorFormat::FP32 => 1,
                crate::types::TensorFormat::INT8 => 2,
            };
            aad.push(fmt_tag);

            match self
                .shared_state
                .session_manager
                .seal(&node_id, &forward.activations, &aad)
            {
                Ok(sealed) => match protocol::encode_layer_forward_encrypted(&forward, sealed) {
                    Ok(p) => p,
                    Err(e) => {
                        tracing::warn!(error = %e, "Failed to encode encrypted tensor");
                        return;
                    }
                },
                Err(e) => {
                    tracing::debug!(error = %e, "Encryption failed, falling back to plaintext");
                    match protocol::encode_layer_forward(&forward) {
                        Ok(p) => p,
                        Err(e) => {
                            tracing::warn!(error = %e, "Failed to encode tensor forward");
                            return;
                        }
                    }
                }
            }
        } else {
            match protocol::encode_layer_forward(&forward) {
                Ok(p) => p,
                Err(e) => {
                    tracing::warn!(error = %e, "Failed to encode tensor forward");
                    return;
                }
            }
        };

        let req = TensorRequest { payload };
        self.swarm
            .behaviour_mut()
            .tensor_rr
            .send_request(&peer_id, req);
        tracing::debug!(
            %peer_id,
            request_id = %forward.request_id,
            seq = forward.sequence_num,
            encrypted = use_encryption,
            "Sent tensor forward"
        );
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

    /// Serve shard data from disk. Supports two modes:
    /// 1. Individual shard files (shard_NNN.bin) — for nodes that downloaded shards
    /// 2. Source GGUF file with byte-range mapping — for the original model host
    ///
    /// The shard's `chunk_offset` is relative to the shard itself (not the source file).
    fn serve_shard_data(&self, req: &crate::types::ShardRequest) -> SwarmResponse {
        use std::io::{Read, Seek, SeekFrom};

        let model_id = &req.shard_id.model_id;
        let shard_index = req.shard_id.index;

        // First try: individual shard file on disk
        let shard_path = self.shard_store.shard_path(model_id, shard_index);
        if shard_path.exists() {
            return self.read_file_chunk(
                &shard_path,
                req.chunk_offset,
                req.chunk_size,
                model_id,
                shard_index,
            );
        }

        // Second try: read byte range from the source GGUF file
        // The source_path file tells us where the original GGUF lives
        let model_dir = self.shard_store.models_dir().join(&model_id.0);
        let source_path_file = model_dir.join("source_path");
        if source_path_file.exists() {
            if let Ok(source_path_str) = std::fs::read_to_string(&source_path_file) {
                let source_path = std::path::Path::new(source_path_str.trim());
                if source_path.exists() {
                    // Look up the shard's size from the manifest to compute byte offset
                    let manifest = self.shared_state.model_registry.get_manifest(model_id);
                    if let Some(manifest) = manifest {
                        if let Some(shard_info) =
                            manifest.shards.iter().find(|s| s.index == shard_index)
                        {
                            // Compute the byte offset in the source file for this shard
                            let shard_file_offset: u64 = manifest
                                .shards
                                .iter()
                                .filter(|s| s.index < shard_index)
                                .map(|s| s.size_bytes)
                                .sum();
                            let total_shard_size = shard_info.size_bytes;

                            // chunk_offset is relative to this shard
                            let file_offset = shard_file_offset + req.chunk_offset;
                            let chunk_size = req.chunk_size.min(32 * 1024 * 1024);
                            let remaining_in_shard =
                                total_shard_size.saturating_sub(req.chunk_offset);
                            let read_len = chunk_size.min(remaining_in_shard) as usize;

                            if read_len == 0 {
                                return SwarmResponse::ShardData(crate::types::ShardResponse {
                                    data: vec![],
                                    total_size: total_shard_size,
                                });
                            }

                            match std::fs::File::open(source_path) {
                                Ok(mut file) => {
                                    let _ = file.seek(SeekFrom::Start(file_offset));
                                    let mut buf = vec![0u8; read_len];
                                    match file.read_exact(&mut buf) {
                                        Ok(()) => {
                                            tracing::info!(
                                                model = %model_id,
                                                shard = shard_index,
                                                bytes = buf.len(),
                                                shard_size = total_shard_size,
                                                "Serving shard from source GGUF"
                                            );
                                            return SwarmResponse::ShardData(
                                                crate::types::ShardResponse {
                                                    data: buf,
                                                    total_size: total_shard_size,
                                                },
                                            );
                                        }
                                        Err(e) => {
                                            tracing::warn!(error = %e, "Failed to read from source GGUF");
                                        }
                                    }
                                }
                                Err(e) => {
                                    tracing::warn!(error = %e, "Failed to open source GGUF");
                                }
                            }
                        }
                    }
                }
            }
        }

        tracing::debug!(
            model = %model_id,
            shard = shard_index,
            "Shard not available locally"
        );
        SwarmResponse::ShardData(crate::types::ShardResponse {
            data: vec![],
            total_size: 0,
        })
    }

    /// Read a chunk from a file (individual shard file on disk).
    fn read_file_chunk(
        &self,
        path: &std::path::Path,
        offset: u64,
        chunk_size: u64,
        model_id: &crate::types::ModelId,
        shard_index: u32,
    ) -> SwarmResponse {
        use std::io::{Read, Seek, SeekFrom};

        match std::fs::File::open(path) {
            Ok(mut file) => {
                let total_size = file.metadata().map(|m| m.len()).unwrap_or(0);
                let chunk_size = chunk_size.min(32 * 1024 * 1024);
                let _ = file.seek(SeekFrom::Start(offset));
                let read_len = chunk_size.min(total_size.saturating_sub(offset)) as usize;
                let mut buf = vec![0u8; read_len];
                match file.read_exact(&mut buf) {
                    Ok(()) => {
                        tracing::info!(
                            model = %model_id,
                            shard = shard_index,
                            bytes = buf.len(),
                            total_size,
                            "Serving shard chunk from file"
                        );
                        SwarmResponse::ShardData(crate::types::ShardResponse {
                            data: buf,
                            total_size,
                        })
                    }
                    Err(e) => {
                        tracing::warn!(error = %e, "Failed to read shard file");
                        SwarmResponse::ShardData(crate::types::ShardResponse {
                            data: vec![],
                            total_size: 0,
                        })
                    }
                }
            }
            Err(e) => {
                tracing::warn!(error = %e, "Failed to open shard file");
                SwarmResponse::ShardData(crate::types::ShardResponse {
                    data: vec![],
                    total_size: 0,
                })
            }
        }
    }

    /// Send a shard transfer request to a specific peer.
    fn handle_send_shard_request(
        &mut self,
        target_peer_bytes: Vec<u8>,
        request: crate::types::ShardRequest,
    ) {
        let peer_id = match libp2p::PeerId::from_bytes(&target_peer_bytes) {
            Ok(id) => id,
            Err(e) => {
                tracing::warn!(error = %e, "Invalid peer ID bytes for shard request");
                return;
            }
        };

        tracing::info!(
            %peer_id,
            model = %request.shard_id.model_id,
            index = request.shard_id.index,
            offset = request.chunk_offset,
            "Sending shard transfer request to peer"
        );

        // Track this request so we know which shard the response belongs to
        self.pending_shard_requests
            .insert(peer_id, request.shard_id.clone());

        let req = SwarmRequest::ShardTransfer(request);
        self.swarm
            .behaviour_mut()
            .request_response
            .send_request(&peer_id, req);
    }

    /// Send a StreamingToken to a specific peer via the JSON request_response protocol.
    fn handle_send_streaming_token(
        &mut self,
        target_peer_bytes: Vec<u8>,
        token: crate::types::StreamingToken,
    ) {
        let peer_id = match libp2p::PeerId::from_bytes(&target_peer_bytes) {
            Ok(id) => id,
            Err(e) => {
                tracing::warn!(error = %e, "Invalid peer ID bytes for streaming token");
                return;
            }
        };

        let msg = SwarmMessage::StreamingToken(token);
        let req = SwarmRequest::Message(Box::new(msg));
        self.swarm
            .behaviour_mut()
            .request_response
            .send_request(&peer_id, req);
    }

    fn update_peer_count(&mut self) {
        let count = self.swarm.connected_peers().count() as u32;
        if let Ok(mut stats) = self.shared_state.node_stats.try_write() {
            stats.peers_connected = count;
        }
    }

    /// Look up the NodeId for a libp2p PeerId by searching the peer registry.
    fn find_node_id_for_peer(&self, peer_id: &libp2p::PeerId) -> Option<crate::types::NodeId> {
        let peer_bytes = peer_id.to_bytes();
        for entry in self.shared_state.peer_registry.iter() {
            if let Some(ref stored_bytes) = entry.value().peer_id_bytes {
                if stored_bytes == &peer_bytes {
                    return Some(entry.key().clone());
                }
            }
        }
        None
    }
}
