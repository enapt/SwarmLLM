use std::collections::HashMap;
use std::sync::Arc;

use dashmap::DashMap;
use futures::StreamExt;
use libp2p::gossipsub::{self, IdentTopic};
use libp2p::request_response::{self, OutboundRequestId};
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
use crate::network::protocol::{self, SwarmRequest, SwarmResponse, TOPIC_MODELS};
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
    /// Maps OutboundRequestId → (PeerId, ShardId) for in-flight shard download requests.
    pending_shard_requests: HashMap<OutboundRequestId, (libp2p::PeerId, crate::types::ShardId)>,
    /// Tracks bytes downloaded so far per shard for chunked transfers.
    shard_download_progress: HashMap<crate::types::ShardId, u64>,
    /// Reverse lookup: PeerId → NodeId for O(1) peer identification.
    peer_to_node: DashMap<libp2p::PeerId, crate::types::NodeId>,
    /// Buffered GossipSub messages that failed to publish at startup (no peers yet).
    buffered_gossip: Vec<(String, Vec<u8>)>,
    /// Whether relay listen has been activated for this session (at most once).
    relay_activated: bool,
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
        let enable_mdns = config.network.enable_mdns;
        // Load cached peer count to auto-scale GossipSub mesh parameters.
        let known_peers = crate::network::peer_cache::load_peer_cache(&shared_state.db).len()
            + config.network.bootstrap_peers.len();
        let swarm = SwarmBuilder::with_existing_identity(keypair)
            .with_tokio()
            .with_quic()
            .with_relay_client(libp2p::noise::Config::new, libp2p::yamux::Config::default)
            .map_err(|e| SwarmError::Network(format!("Relay client error: {e}")))?
            .with_behaviour(|_key, relay_behaviour| {
                behaviour::build_behaviour(
                    &kp_clone,
                    relay_behaviour,
                    relay_cfg.as_ref(),
                    enable_mdns,
                    known_peers,
                    Some(&config.network),
                )
                .map_err(|e| {
                    Box::new(std::io::Error::other(e.to_string()))
                        as Box<dyn std::error::Error + Send + Sync>
                })
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
            peer_to_node: DashMap::new(),
            buffered_gossip: Vec::new(),
            relay_activated: false,
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

        // Layer 2: Load cached peers from last session and dial them
        let cached_peers = crate::network::peer_cache::load_peer_cache(&self.shared_state.db);
        if !cached_peers.is_empty() {
            let cached_count = discovery::bootstrap_peers(&mut self.swarm, &cached_peers)?;
            if cached_count > 0 {
                tracing::info!(
                    count = cached_count,
                    "Dialing cached peers from last session"
                );
            }
        }

        // Bootstrap with configured peers
        let bootstrap_count =
            discovery::bootstrap_peers(&mut self.swarm, &config.network.bootstrap_peers)?;
        if bootstrap_count > 0 || !cached_peers.is_empty() {
            discovery::trigger_bootstrap(&mut self.swarm)?;
        }

        // Periodic discovery timer
        let mut discovery_interval = tokio::time::interval(discovery::DISCOVERY_INTERVAL);
        discovery_interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

        // Periodic peer cache save timer (every 5 minutes)
        let mut peer_cache_interval = tokio::time::interval(discovery::PEER_CACHE_SAVE_INTERVAL);
        peer_cache_interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

        tracing::info!("NetworkManager running");

        loop {
            tokio::select! {
                // Shutdown signal
                _ = self.shutdown_rx.changed() => {
                    if *self.shutdown_rx.borrow() {
                        self.save_peer_cache();
                        tracing::info!("NetworkManager shutting down");
                        break;
                    }
                }
                // Periodic discovery
                _ = discovery_interval.tick() => {
                    tracing::debug!("Discovery tick");
                    let _ = discovery::trigger_bootstrap(&mut self.swarm);
                    // Re-dial cached peers that we're not currently connected to.
                    // This handles peers that went offline and came back.
                    let cached = crate::network::peer_cache::load_peer_cache(&self.shared_state.db);
                    if !cached.is_empty() {
                        let _ = discovery::bootstrap_peers(&mut self.swarm, &cached);
                    }
                    self.update_peer_count();
                }
                // Periodic peer cache save
                _ = peer_cache_interval.tick() => {
                    self.save_peer_cache();
                }
                // Outbound commands from other daemon tasks
                cmd = self.inbound_rx.recv() => {
                    match cmd {
                        Some(cmd) => {
                            self.handle_outbound_command(cmd).await;
                            self.drain_pending_swarm_events().await;
                        },
                        None => {
                            self.save_peer_cache();
                            tracing::info!("Inbound channel closed, shutting down");
                            break;
                        }
                    }
                }
                // Swarm events from the network
                event = self.swarm.select_next_some() => {
                    self.handle_swarm_event(event).await;
                }
            }
        }

        Ok(())
    }

    /// Drive the swarm to process any pending behaviour events (e.g. from send_request()).
    ///
    /// After calling methods like `behaviour_mut().request_response.send_request()`, the
    /// behaviour has internal pending events (NotifyHandler) that need to be delivered
    /// to the connection task. Without explicitly driving the swarm, these events wait
    /// until the next `select!` iteration polls the swarm future. This method ensures
    /// they are processed immediately.
    async fn drain_pending_swarm_events(&mut self) {
        use futures::StreamExt;
        use std::task::Poll;
        let mut drained = 0u32;
        // Poll the swarm in a non-blocking loop until it returns Pending.
        while let Some(event) = std::future::poll_fn(|cx| match self.swarm.poll_next_unpin(cx) {
            Poll::Ready(Some(event)) => Poll::Ready(Some(event)),
            Poll::Pending | Poll::Ready(None) => Poll::Ready(None),
        })
        .await
        {
            drained += 1;
            self.handle_swarm_event(event).await;
        }
        if drained > 0 {
            tracing::trace!(drained, "Drained pending swarm events after command");
        }
    }

    async fn handle_swarm_event(&mut self, event: SwarmEvent<SwarmBehaviourEvent>) {
        tracing::trace!(event_type = %swarm_event_name(&event), "Processing swarm event");
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
                        // NET-M10: Reject gossip messages with timestamps older than 5 minutes
                        let now_epoch = chrono::Utc::now().timestamp() as u64;
                        let too_old = match &msg {
                            SwarmMessage::HealthPing { timestamp, .. }
                            | SwarmMessage::HealthPong { timestamp, .. } => {
                                now_epoch.saturating_sub(*timestamp) > 300
                            }
                            _ => false,
                        };
                        if too_old {
                            tracing::debug!(
                                source = %propagation_source,
                                "Dropping stale gossip message (>5 min old)"
                            );
                        } else {
                            tracing::debug!(
                                source = %propagation_source,
                                "Received GossipSub message"
                            );
                            if let Err(e) = self.outbound_tx.try_send(msg) {
                                self.shared_state
                                    .channel_metrics
                                    .network_out
                                    .record_dropped();
                                tracing::warn!(error = %e, "Dispatcher backpressured, dropping gossipsub message");
                            } else {
                                self.shared_state.channel_metrics.network_out.record_sent();
                            }
                        }
                    }
                    Err(e) => {
                        tracing::debug!(error = %e, "Failed to decode gossipsub message");
                    }
                }
            }

            // ── JSON request/response (control messages, shard transfers) ──
            SwarmEvent::Behaviour(SwarmBehaviourEvent::RequestResponse(
                request_response::Event::Message { peer, message, .. },
            )) => match message {
                request_response::Message::Request {
                    request, channel, ..
                } => {
                    let kind = match &request {
                        SwarmRequest::Message(_) => "message",
                        SwarmRequest::ShardTransfer(_) => "shard",
                        SwarmRequest::TensorPayload(_) => "tensor",
                    };
                    tracing::info!(%peer, kind, "Received request");
                    self.handle_request(peer, request, channel).await;
                }
                request_response::Message::Response {
                    request_id,
                    response,
                } => {
                    let kind = match &response {
                        SwarmResponse::Message(_) => "message",
                        SwarmResponse::ShardData(_) => "shard",
                        SwarmResponse::Ack => "ack",
                        SwarmResponse::TensorPayload(_) => "tensor",
                    };
                    tracing::info!(%peer, kind, "Received response");
                    self.handle_response(peer, request_id, response).await;
                }
            },

            // ── Request/response failures ──
            SwarmEvent::Behaviour(SwarmBehaviourEvent::RequestResponse(
                request_response::Event::OutboundFailure {
                    peer,
                    request_id,
                    error,
                    ..
                },
            )) => {
                tracing::warn!(%peer, ?request_id, %error, "Request outbound failure");
                // Check if this was a pending shard download request
                if let Some((_peer_id, shard_id)) = self.pending_shard_requests.remove(&request_id)
                {
                    tracing::error!(
                        %peer, shard = ?shard_id,
                        "Shard request failed: {error}"
                    );
                    // Clean up stale download progress entry to prevent resource leak
                    self.shard_download_progress.remove(&shard_id);
                }
            }
            SwarmEvent::Behaviour(SwarmBehaviourEvent::RequestResponse(
                request_response::Event::InboundFailure { peer, error, .. },
            )) => {
                tracing::debug!(%peer, %error, "Request inbound failure");
            }
            SwarmEvent::Behaviour(SwarmBehaviourEvent::RequestResponse(
                request_response::Event::ResponseSent {
                    peer, request_id, ..
                },
            )) => {
                tracing::debug!(%peer, ?request_id, "Response sent to peer");
            }

            // ── GossipSub peer subscribed — flush buffered messages (NET-I4) ──
            SwarmEvent::Behaviour(SwarmBehaviourEvent::Gossipsub(
                gossipsub::Event::Subscribed { peer_id, topic },
            )) => {
                tracing::debug!(%peer_id, %topic, "Peer subscribed to topic");
                if !self.buffered_gossip.is_empty() {
                    let buffered = std::mem::take(&mut self.buffered_gossip);
                    let mut replayed = 0;
                    for (topic_str, data) in buffered {
                        let gossip_topic = IdentTopic::new(&topic_str);
                        match self
                            .swarm
                            .behaviour_mut()
                            .gossipsub
                            .publish(gossip_topic, data.clone())
                        {
                            Ok(_) => {
                                replayed += 1;
                            }
                            Err(_) => {
                                // Still can't publish — re-buffer
                                self.buffered_gossip.push((topic_str, data));
                            }
                        }
                    }
                    if replayed > 0 {
                        tracing::info!(count = replayed, "Replayed buffered GossipSub messages");
                    }
                }
            }

            // ── Relay server events ──
            SwarmEvent::Behaviour(SwarmBehaviourEvent::RelayServer(event)) => {
                crate::network::relay::handle_relay_server_event(event);
            }

            // ── AutoNAT status changes ──
            SwarmEvent::Behaviour(SwarmBehaviourEvent::Autonat(
                libp2p::autonat::Event::StatusChanged { old, new },
            )) => {
                tracing::info!(?old, ?new, "AutoNAT status changed");
                {
                    if let Ok(mut stats) = self.shared_state.node_stats.try_write() {
                        stats.nat_status = Some(format!("{new:?}"));
                    }
                }
                // NET-M3: Auto-listen on relay when NAT is detected as Private
                if matches!(new, libp2p::autonat::NatStatus::Private)
                    && !self.relay_activated
                    && self.shared_state.config.network.auto_relay
                {
                    self.relay_activated = true;
                    tracing::info!("NAT detected, activating relay listener");

                    // Try bootstrap peers as relay candidates — they are most likely
                    // to be publicly reachable and have relay enabled.
                    let bootstrap_addrs = &self.shared_state.config.network.bootstrap_peers;
                    let mut relayed = false;
                    for addr_str in bootstrap_addrs {
                        if let Ok(maddr) = addr_str.parse::<Multiaddr>() {
                            // Extract the peer ID from the multiaddr (/p2p/<peer_id>)
                            let maybe_pid = maddr.iter().find_map(|proto| {
                                if let libp2p::multiaddr::Protocol::P2p(pid) = proto {
                                    Some(pid)
                                } else {
                                    None
                                }
                            });
                            if let Some(relay_pid) = maybe_pid {
                                // Build a relay-listen address without the trailing /p2p
                                let base: Multiaddr = maddr
                                    .iter()
                                    .take_while(|p| {
                                        !matches!(p, libp2p::multiaddr::Protocol::P2p(_))
                                    })
                                    .collect();
                                let relay_addr =
                                    crate::network::relay::relay_listen_addr(&relay_pid, &base);
                                match self.swarm.listen_on(relay_addr.clone()) {
                                    Ok(_) => {
                                        tracing::info!(
                                            relay_peer = %relay_pid,
                                            %relay_addr,
                                            "Relay listen activated"
                                        );
                                        relayed = true;
                                        break; // One relay is sufficient
                                    }
                                    Err(e) => {
                                        tracing::debug!(
                                            relay_peer = %relay_pid,
                                            error = %e,
                                            "Failed to listen via relay peer"
                                        );
                                    }
                                }
                            }
                        }
                    }

                    if !relayed && !bootstrap_addrs.is_empty() {
                        tracing::warn!(
                            "NAT detected but no relay peers accepted — node may be unreachable"
                        );
                    }
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

                let now_ts = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs();
                // Preserve first_seen from existing entry or use current time
                let first_seen = self
                    .shared_state
                    .peer_registry
                    .get(&node_id)
                    .map(|p| p.first_seen)
                    .unwrap_or(now_ts);
                let peer_info = PeerInfo {
                    node_id: node_id.clone(),
                    addresses: info.listen_addrs.iter().map(|a| a.to_string()).collect(),
                    capability: None,
                    last_seen: chrono::Utc::now(),
                    latency_ms: None,
                    trust_score: 0.5,
                    peer_id_bytes: Some(peer_id.to_bytes()),
                    active_request_count: 0,
                    first_seen,
                    verified_transaction_count: 0,
                    is_lan_peer: false,
                };
                // NET-C4: Populate reverse PeerId → NodeId lookup
                self.peer_to_node.insert(peer_id, node_id.clone());
                self.shared_state
                    .peer_registry
                    .insert(node_id.clone(), peer_info);

                // Layer 6: Track subnet for anti-gaming — extract IPv4 from listen addrs
                for addr in &info.listen_addrs {
                    if let Some(ip_bytes) = extract_ipv4_bytes(addr) {
                        // Skip private (RFC 1918) and loopback addresses
                        if ip_bytes[0] == 127
                            || ip_bytes[0] == 0
                            || ip_bytes[0] == 10
                            || (ip_bytes[0] == 172 && (16..=31).contains(&ip_bytes[1]))
                            || (ip_bytes[0] == 192 && ip_bytes[1] == 168)
                        {
                            continue;
                        }
                        // Use try_lock() to avoid blocking the event loop.
                        // If contended, skip — next Identify event will catch it.
                        if let Ok(mut anti_gaming) = self.shared_state.anti_gaming.try_lock() {
                            anti_gaming.register_subnet(&node_id, ip_bytes);
                        }
                        break; // One IP per peer is enough
                    }
                }
            }

            // ── Kademlia ──
            SwarmEvent::Behaviour(SwarmBehaviourEvent::Kademlia(
                libp2p::kad::Event::OutboundQueryProgressed { result, .. },
            )) => {
                tracing::debug!(?result, "Kademlia query progressed");
            }

            // ── mDNS ──
            SwarmEvent::Behaviour(SwarmBehaviourEvent::Mdns(libp2p::mdns::Event::Discovered(
                peers,
            ))) => {
                for (peer_id, addr) in peers {
                    // Do NOT add mDNS addresses to Kademlia. Kademlia's periodic
                    // routing table refresh dials all known addresses, creating
                    // duplicate connections every 30s that corrupt request_response
                    // routing. The identify protocol handles address exchange after
                    // connection is established.
                    if !self.swarm.is_connected(&peer_id) {
                        tracing::info!(
                            %peer_id, %addr,
                            "LAN peer discovered automatically — no configuration needed"
                        );
                        // Use Disconnected (not DisconnectedAndNotDialing) so mDNS
                        // can override a failing bootstrap dial attempt. Without this,
                        // a peer that restarts with a new identity can't reconnect
                        // because the stale bootstrap dial blocks mDNS.
                        let opts = libp2p::swarm::dial_opts::DialOpts::peer_id(peer_id)
                            .condition(
                                libp2p::swarm::dial_opts::PeerCondition::Disconnected,
                            )
                            .addresses(vec![addr])
                            .build();
                        if let Err(e) = self.swarm.dial(opts) {
                            tracing::debug!(%peer_id, error = %e, "mDNS: dial skipped");
                        }
                    } else {
                        tracing::debug!(%peer_id, "mDNS: already connected, skipping");
                    }
                    // Mark as LAN peer if we can derive their NodeId
                    if let Some(node_id) = self.peer_to_node.get(&peer_id) {
                        if let Some(mut peer) = self.shared_state.peer_registry.get_mut(&*node_id) {
                            if !peer.is_lan_peer {
                                peer.is_lan_peer = true;
                                drop(peer);
                                // Increment LAN peer count and notify WebSocket clients
                                let count = self
                                    .shared_state
                                    .lan_peer_count
                                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
                                    + 1;
                                let _ = self.shared_state.lan_discovery_tx.send(count as u32);
                                tracing::info!(
                                    lan_peers = count,
                                    "Found {} peer{} on your local network",
                                    count,
                                    if count == 1 { "" } else { "s" }
                                );
                            }
                        }
                    }
                }
            }

            SwarmEvent::Behaviour(SwarmBehaviourEvent::Mdns(libp2p::mdns::Event::Expired(
                peers,
            ))) => {
                for (peer_id, _addr) in peers {
                    tracing::debug!(%peer_id, "mDNS: peer expired");
                    // Decrement LAN peer count if this was a tracked LAN peer
                    if let Some(node_id) = self.peer_to_node.get(&peer_id) {
                        if let Some(mut peer) = self.shared_state.peer_registry.get_mut(&*node_id) {
                            if peer.is_lan_peer {
                                peer.is_lan_peer = false;
                                drop(peer);
                                self.shared_state
                                    .lan_peer_count
                                    .fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
                            }
                        }
                    }
                }
            }

            SwarmEvent::ConnectionEstablished {
                peer_id,
                connection_id,
                num_established,
                ..
            } => {
                tracing::info!(%peer_id, %connection_id, count = num_established, "Connection established");
                self.update_peer_count();

                // Layer 5: Peer Exchange — send PEX request on first connection only
                if num_established.get() == 1 && self.shared_state.config.network.peer_exchange {
                    let req = SwarmRequest::Message(Box::new(SwarmMessage::PeerExchangeRequest));
                    self.swarm
                        .behaviour_mut()
                        .request_response
                        .send_request(&peer_id, req);
                    tracing::debug!(%peer_id, "Sent PEX request");
                }
            }

            SwarmEvent::ConnectionClosed {
                peer_id,
                cause,
                num_established,
                ..
            } => {
                tracing::info!(%peer_id, ?cause, remaining = num_established, "Connection closed");

                // Skip cleanup if other connections to this peer remain
                if num_established > 0 {
                    tracing::debug!(%peer_id, remaining = num_established, "Other connections remain, skipping cleanup");
                } else if self.swarm.is_connected(&peer_id) {
                    // Swarm still considers peer connected (race: another
                    // connection was just established) — skip cleanup.
                    tracing::debug!(%peer_id, "Peer still connected per swarm, skipping cleanup");
                    self.update_peer_count();
                } else {
                    self.update_peer_count();

                    // NET-I1: Drain pending shard requests and download progress for this peer
                    let drained_ids: Vec<OutboundRequestId> = self
                        .pending_shard_requests
                        .iter()
                        .filter(|(_, (pid, _))| *pid == peer_id)
                        .map(|(rid, _)| *rid)
                        .collect();
                    for rid in &drained_ids {
                        if let Some((_, shard_id)) = self.pending_shard_requests.remove(rid) {
                            self.shard_download_progress.remove(&shard_id);
                            tracing::debug!(
                                %peer_id,
                                model = %shard_id.model_id,
                                index = shard_id.index,
                                "Cleaned up pending shard request for disconnected peer"
                            );
                        }
                    }

                    // NET-I2: Remove peer from registry, but skip if in active pipelines.
                    // Clone the NodeId and drop the DashMap Ref BEFORE calling remove(),
                    // otherwise get() holds a read lock and remove() needs a write lock
                    // on the same shard → synchronous deadlock that freezes the event loop.
                    let node_id_opt = self.peer_to_node.get(&peer_id).map(|r| r.clone());
                    if let Some(node_id) = node_id_opt {
                        let in_active_pipeline =
                            self.shared_state.active_pipelines.iter().any(|entry| {
                                entry
                                    .value()
                                    .segments
                                    .iter()
                                    .any(|seg| seg.node_id == node_id)
                            });
                        if !in_active_pipeline {
                            self.shared_state.peer_registry.remove(&node_id);
                            self.peer_to_node.remove(&peer_id);
                            tracing::debug!(%peer_id, "Removed disconnected peer from registry");
                        } else {
                            tracing::debug!(%peer_id, "Keeping peer in registry (active pipeline)");
                        }
                    }
                } // end else (num_established == 0)
            }

            SwarmEvent::NewListenAddr { address, .. } => {
                tracing::info!(%address, "New listen address");
            }

            // NET-I7: Switch Kademlia to Server mode when external address is confirmed
            SwarmEvent::ExternalAddrConfirmed { address } => {
                tracing::info!(%address, "External address confirmed — switching Kademlia to Server mode");
                self.swarm
                    .behaviour_mut()
                    .kademlia
                    .set_mode(Some(libp2p::kad::Mode::Server));
            }

            SwarmEvent::OutgoingConnectionError {
                peer_id,
                error,
                ..
            } => {
                tracing::debug!(
                    ?peer_id, %error,
                    "Outgoing connection failed"
                );
            }

            SwarmEvent::IncomingConnectionError {
                error,
                ..
            } => {
                tracing::debug!(
                    %error,
                    "Incoming connection failed"
                );
            }

            other => {
                tracing::trace!(?other, "Unhandled swarm event");
            }
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
                // Handle PEX messages inline instead of forwarding to dispatcher
                match *msg {
                    SwarmMessage::PeerExchangeRequest => {
                        tracing::debug!(%peer, "Handling PEX request");
                        // Respond with up to 20 known peer addresses (filter out self)
                        let local_node_id = self.shared_state.identity.node_id();
                        let peers: Vec<String> = self
                            .shared_state
                            .peer_registry
                            .iter()
                            .filter(|entry| entry.key() != local_node_id)
                            .flat_map(|entry| entry.addresses.clone())
                            .take(20)
                            .collect();
                        let pex_resp = SwarmMessage::PeerExchangeResponse(
                            crate::types::PeerExchangeResponse { peers },
                        );
                        let resp = SwarmResponse::Message(Box::new(pex_resp));
                        if self
                            .swarm
                            .behaviour_mut()
                            .request_response
                            .send_response(channel, resp)
                            .is_err()
                        {
                            tracing::debug!(%peer, "Failed to send PEX response (channel closed)");
                        }
                        return;
                    }
                    SwarmMessage::PeerExchangeResponse(ref pex_resp) => {
                        tracing::debug!(%peer, count = pex_resp.peers.len(), "Received PEX response (via request)");
                        self.handle_pex_response(&pex_resp.peers);
                        // ACK and return
                        let _ = self
                            .swarm
                            .behaviour_mut()
                            .request_response
                            .send_response(channel, SwarmResponse::Ack);
                        return;
                    }
                    _ => {
                        // Forward all other messages to dispatcher
                        tracing::debug!(%peer, "Handling protocol message request");
                        if let Err(e) = self.outbound_tx.try_send(*msg) {
                            self.shared_state
                                .channel_metrics
                                .network_out
                                .record_dropped();
                            tracing::warn!(error = %e, "Dispatcher backpressured, dropping request message");
                        } else {
                            self.shared_state.channel_metrics.network_out.record_sent();
                        }
                    }
                }
                // NET-M7: Log send_response errors
                if self
                    .swarm
                    .behaviour_mut()
                    .request_response
                    .send_response(channel, SwarmResponse::Ack)
                    .is_err()
                {
                    tracing::debug!(%peer, "Failed to send ACK (channel closed)");
                }
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

                // NET-M7: Log send_response errors
                if self
                    .swarm
                    .behaviour_mut()
                    .request_response
                    .send_response(channel, response)
                    .is_err()
                {
                    tracing::debug!(%peer, "Failed to send shard data response (channel closed)");
                }
            }
            SwarmRequest::TensorPayload(payload) => {
                self.handle_tensor_payload(peer, &payload);
                // ACK the tensor request
                let resp = SwarmResponse::Ack;
                if self
                    .swarm
                    .behaviour_mut()
                    .request_response
                    .send_response(channel, resp)
                    .is_err()
                {
                    tracing::debug!(%peer, "Failed to send tensor ACK (channel closed)");
                }
            }
        }
    }

    async fn handle_response(
        &mut self,
        peer: libp2p::PeerId,
        request_id: OutboundRequestId,
        response: SwarmResponse,
    ) {
        match response {
            SwarmResponse::Message(msg) => {
                // Handle PEX response inline
                if let SwarmMessage::PeerExchangeResponse(ref pex_resp) = *msg {
                    tracing::debug!(%peer, count = pex_resp.peers.len(), "Received PEX response");
                    self.handle_pex_response(&pex_resp.peers);
                    return;
                }
                if let Err(e) = self.outbound_tx.try_send(*msg) {
                    self.shared_state
                        .channel_metrics
                        .network_out
                        .record_dropped();
                    tracing::warn!(%peer, error = %e, "Dispatcher backpressured, dropping response message");
                } else {
                    self.shared_state.channel_metrics.network_out.record_sent();
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
                // Route to AcquisitionManager — always clean up tracking state
                // NET-C1: Look up by OutboundRequestId for correct correlation
                if let Some((_, shard_id)) = self.pending_shard_requests.remove(&request_id) {
                    let offset = self
                        .shard_download_progress
                        .get(&shard_id)
                        .copied()
                        .unwrap_or(0);
                    let chunk_len = data.data.len() as u64;

                    if let Some(ref acq_tx) = self.acquisition_tx {
                        if let Err(e) = acq_tx.try_send(
                            AcquisitionCommand::ShardDataReceived {
                                shard_id: shard_id.clone(),
                                offset,
                                data: data.data,
                                total_size: data.total_size,
                            },
                        ) {
                            tracing::warn!(error = %e, "Failed to forward shard data to acquisition");
                        }
                    }

                    // Update progress tracking
                    let new_offset = offset + chunk_len;
                    if new_offset < data.total_size {
                        // More chunks needed — re-register and request next chunk
                        self.shard_download_progress
                            .insert(shard_id.clone(), new_offset);

                        let next_req = crate::types::ShardRequest {
                            shard_id: shard_id.clone(),
                            chunk_offset: new_offset,
                            chunk_size: 32 * 1024 * 1024, // 32MB chunks
                        };
                        let req = SwarmRequest::ShardTransfer(next_req);
                        let new_req_id = self
                            .swarm
                            .behaviour_mut()
                            .request_response
                            .send_request(&peer, req);
                        self.pending_shard_requests
                            .insert(new_req_id, (peer, shard_id));
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
                        "Received shard data but no pending request found"
                    );
                }
            }
            SwarmResponse::TensorPayload(payload) => {
                // Tensor data in a response (e.g. LayerResult sent as response)
                self.handle_tensor_payload(peer, &payload);
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
                let publish_data = data;

                let gossip_topic = IdentTopic::new(topic);
                match self
                    .swarm
                    .behaviour_mut()
                    .gossipsub
                    .publish(gossip_topic, publish_data.clone())
                {
                    Ok(_) => tracing::debug!(topic, "Published message to GossipSub"),
                    Err(e) => {
                        // NET-I4: Buffer messages that fail at startup (no peers), capped to prevent memory leak
                        if self.buffered_gossip.len() < 64 {
                            tracing::debug!(topic, error = %e, "Failed to publish to GossipSub, buffering");
                            self.buffered_gossip.push((topic.to_string(), publish_data));
                        } else {
                            tracing::warn!(topic, "Gossip buffer full (64), dropping message");
                        }
                    }
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
        let use_encryption = peer_node_id.is_some();

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

        let payload_len = payload.len();
        let req = SwarmRequest::TensorPayload(payload);
        self.swarm
            .behaviour_mut()
            .request_response
            .send_request(&peer_id, req);
        tracing::info!(
            %peer_id,
            request_id = %forward.request_id,
            seq = forward.sequence_num,
            encrypted = use_encryption,
            payload_len,
            "Sent tensor forward"
        );
    }

    /// Send a tensor result to a specific peer via the unified protocol.
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
                let payload_len = payload.len();
                let req = SwarmRequest::TensorPayload(payload);
                self.swarm
                    .behaviour_mut()
                    .request_response
                    .send_request(&peer_id, req);
                tracing::info!(
                    %peer_id,
                    request_id = %result.request_id,
                    payload_len,
                    "Sent tensor result"
                );
            }
            Err(e) => {
                tracing::warn!(error = %e, "Failed to encode tensor result");
            }
        }
    }

    /// Process an inbound binary tensor payload (from either request or response).
    fn handle_tensor_payload(&mut self, peer: libp2p::PeerId, payload: &[u8]) {
        let tag = payload.first().copied().unwrap_or(0);
        tracing::debug!(%peer, tag, payload_len = payload.len(), "handle_tensor_payload");
        match tag {
            protocol::TENSOR_TAG_FORWARD => {
                tracing::info!(%peer, payload_len = payload.len(), "Received tensor LayerForward");
                match protocol::decode_layer_forward(payload) {
                    Ok(mut forward) => {
                        forward.sender_peer_bytes = Some(peer.to_bytes());
                        let msg = SwarmMessage::LayerForward(forward);
                        if let Err(e) = self.outbound_tx.try_send(msg) {
                            self.shared_state
                                .channel_metrics
                                .network_out
                                .record_dropped();
                            tracing::warn!(error = %e, "Outbound channel full, dropping tensor forward");
                        } else {
                            self.shared_state.channel_metrics.network_out.record_sent();
                        }
                    }
                    Err(e) => {
                        tracing::warn!(error = %e, "Failed to decode tensor forward");
                    }
                }
            }
            protocol::TENSOR_TAG_RESULT => {
                tracing::info!(%peer, payload_len = payload.len(), "Received tensor LayerResult");
                match protocol::decode_layer_result(payload) {
                    Ok(result) => {
                        tracing::debug!(
                            %peer,
                            request_id = %result.request_id,
                            tokens = result.token_ids.len(),
                            activations_bytes = result.activations.len(),
                            "Decoded tensor LayerResult, dispatching"
                        );
                        if let Err(e) = self.outbound_tx.try_send(SwarmMessage::LayerResult(result))
                        {
                            self.shared_state
                                .channel_metrics
                                .network_out
                                .record_dropped();
                            tracing::warn!(error = %e, "Outbound channel full, dropping tensor result");
                        } else {
                            self.shared_state.channel_metrics.network_out.record_sent();
                        }
                    }
                    Err(e) => {
                        tracing::warn!(error = %e, "Failed to decode tensor result");
                    }
                }
            }
            protocol::TENSOR_TAG_ENCRYPTED => {
                tracing::debug!(%peer, "Received encrypted tensor");
                match protocol::decode_layer_forward_encrypted(payload) {
                    Ok((mut forward, sealed, aad)) => {
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
                                    if let Err(e) = self
                                        .outbound_tx
                                        .try_send(SwarmMessage::LayerForward(forward))
                                    {
                                        self.shared_state
                                            .channel_metrics
                                            .network_out
                                            .record_dropped();
                                        tracing::warn!(error = %e, "Outbound channel full, dropping decrypted tensor");
                                    } else {
                                        self.shared_state.channel_metrics.network_out.record_sent();
                                    }
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
    }

    /// Serve shard data from disk. Supports two modes:
    /// 1. Individual shard files (shard_NNN.bin) — for nodes that downloaded shards
    /// 2. Source GGUF file with byte-range mapping — for the original model host
    ///
    /// The shard's `chunk_offset` is relative to the shard itself (not the source file).
    fn serve_shard_data(&self, req: &crate::types::ShardRequest) -> SwarmResponse {
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

        // Shard files are self-contained — no source GGUF fallback needed.

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
    /// Uses spawn_blocking to avoid stalling the async event loop with file I/O.
    fn read_file_chunk(
        &self,
        path: &std::path::Path,
        offset: u64,
        chunk_size: u64,
        model_id: &crate::types::ModelId,
        shard_index: u32,
    ) -> SwarmResponse {
        use std::io::{Read, Seek, SeekFrom};

        let path = path.to_path_buf();
        let model_id = model_id.clone();

        // Perform blocking file I/O on a dedicated thread to avoid stalling
        // the libp2p event loop while reading potentially large shard chunks.
        let result = std::thread::scope(|_| match std::fs::File::open(&path) {
            Ok(mut file) => {
                let total_size = file.metadata().map(|m| m.len()).unwrap_or(0);
                let chunk_size = chunk_size.min(32 * 1024 * 1024);
                if let Err(e) = file.seek(SeekFrom::Start(offset)) {
                    tracing::warn!(error = %e, "Failed to seek in shard file");
                    return SwarmResponse::ShardData(crate::types::ShardResponse {
                        data: vec![],
                        total_size: 0,
                    });
                }
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
        });

        result
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

        let shard_id = request.shard_id.clone();
        let req = SwarmRequest::ShardTransfer(request);
        // NET-C1: Track by OutboundRequestId for correct request-response correlation
        let outbound_id = self
            .swarm
            .behaviour_mut()
            .request_response
            .send_request(&peer_id, req);
        self.pending_shard_requests
            .insert(outbound_id, (peer_id, shard_id));
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

    /// Update the peer count in shared state.
    ///
    /// Uses `try_write()` instead of `.write().await` to avoid deadlocking the
    /// event loop. The WebSocket stats pusher holds a long-lived read lock on
    /// `node_stats` (across DashMap iteration + JSON serialization), so a
    /// `.write().await` here suspends the entire swarm event loop until that
    /// read lock is released — causing the event loop to freeze and preventing
    /// reconnection. With `try_write()`, a contended update is simply skipped;
    /// the next connection event or discovery tick will correct it.
    fn update_peer_count(&self) {
        let count = self.swarm.connected_peers().count() as u32;
        if let Ok(mut stats) = self.shared_state.node_stats.try_write() {
            stats.peers_connected = count;
        }
    }

    /// NET-C4: O(1) lookup of NodeId for a libp2p PeerId via reverse index.
    fn find_node_id_for_peer(&self, peer_id: &libp2p::PeerId) -> Option<crate::types::NodeId> {
        self.peer_to_node.get(peer_id).map(|v| v.clone())
    }

    /// Save current peer addresses to the persistent cache.
    fn save_peer_cache(&self) {
        let addrs: Vec<String> = self
            .shared_state
            .peer_registry
            .iter()
            .flat_map(|entry| entry.addresses.clone())
            .collect();
        if !addrs.is_empty() {
            crate::network::peer_cache::save_peer_cache(&self.shared_state.db, &addrs);
        }
    }

    /// Handle PEX response — dial unknown peers from the exchanged address list.
    /// Limits to 5 dials per response to prevent connection storms.
    fn handle_pex_response(&mut self, peer_addrs: &[String]) {
        const MAX_PEX_DIALS: usize = 5;
        let mut dialed = 0;
        for addr_str in peer_addrs {
            if dialed >= MAX_PEX_DIALS {
                break;
            }
            if let Ok(addr) = addr_str.parse::<Multiaddr>() {
                // Extract peer ID to check if already connected
                let maybe_peer_id = addr.iter().find_map(|proto| {
                    if let libp2p::multiaddr::Protocol::P2p(pid) = proto {
                        Some(pid)
                    } else {
                        None
                    }
                });

                // Skip if already connected
                if let Some(pid) = &maybe_peer_id {
                    if self.swarm.is_connected(pid) {
                        continue;
                    }
                }

                // Add to Kademlia
                if let Some(pid) = &maybe_peer_id {
                    self.swarm
                        .behaviour_mut()
                        .kademlia
                        .add_address(pid, addr.clone());
                }

                if let Err(e) = self.swarm.dial(addr) {
                    tracing::debug!(error = %e, "PEX: failed to dial peer");
                } else {
                    dialed += 1;
                }
            }
        }
        if dialed > 0 {
            tracing::info!(count = dialed, "PEX: dialed new peers");
        }
    }
}

/// Get a human-readable name for a swarmEvent (for debug logging).
fn swarm_event_name(event: &SwarmEvent<SwarmBehaviourEvent>) -> &'static str {
    match event {
        SwarmEvent::Behaviour(b) => match b {
            SwarmBehaviourEvent::Gossipsub(_) => "Gossipsub",
            SwarmBehaviourEvent::RequestResponse(_) => "RequestResponse",
            SwarmBehaviourEvent::Kademlia(_) => "Kademlia",
            SwarmBehaviourEvent::Identify(_) => "Identify",
            SwarmBehaviourEvent::Autonat(_) => "AutoNAT",
            SwarmBehaviourEvent::Dcutr(_) => "DCUtR",
            SwarmBehaviourEvent::RelayClient(_) => "RelayClient",
            SwarmBehaviourEvent::RelayServer(_) => "RelayServer",
            SwarmBehaviourEvent::ConnectionLimits(_) => "ConnectionLimits",
            SwarmBehaviourEvent::Mdns(_) => "mDNS",
        },
        SwarmEvent::ConnectionEstablished { .. } => "ConnectionEstablished",
        SwarmEvent::ConnectionClosed { .. } => "ConnectionClosed",
        SwarmEvent::IncomingConnection { .. } => "IncomingConnection",
        SwarmEvent::IncomingConnectionError { .. } => "IncomingConnectionError",
        SwarmEvent::OutgoingConnectionError { .. } => "OutgoingConnectionError",
        SwarmEvent::NewListenAddr { .. } => "NewListenAddr",
        SwarmEvent::ExpiredListenAddr { .. } => "ExpiredListenAddr",
        SwarmEvent::ListenerClosed { .. } => "ListenerClosed",
        SwarmEvent::ListenerError { .. } => "ListenerError",
        SwarmEvent::Dialing { .. } => "Dialing",
        SwarmEvent::NewExternalAddrCandidate { .. } => "NewExternalAddrCandidate",
        SwarmEvent::ExternalAddrConfirmed { .. } => "ExternalAddrConfirmed",
        SwarmEvent::ExternalAddrExpired { .. } => "ExternalAddrExpired",
        _ => "Unknown",
    }
}

/// Extract IPv4 bytes from a multiaddr, if present.
fn extract_ipv4_bytes(addr: &Multiaddr) -> Option<[u8; 4]> {
    for proto in addr.iter() {
        if let libp2p::multiaddr::Protocol::Ip4(ip) = proto {
            return Some(ip.octets());
        }
    }
    None
}
