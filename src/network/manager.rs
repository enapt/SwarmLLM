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

/// Maximum in-flight shard chunk requests before dropping new ones.
const MAX_PENDING_SHARD_REQUESTS: usize = 1024;
/// Maximum lifetime (seconds) for a tensor channel or outbound forward before eviction.
/// Used for both channel cleanup and adaptive timeout upper clamp.
const MAX_TENSOR_FORWARD_SECS: u64 = 600;
/// libp2p swarm idle connection timeout. Connections with no traffic for this long are closed.
const IDLE_CONNECTION_TIMEOUT_SECS: u64 = 120;
/// Interval for periodic PEX ping health checks. Keeps the outbound queue shallow so
/// tensor forwards get immediate service instead of queueing behind stale requests.
const RR_PING_INTERVAL_SECS: u64 = 120;
/// Interval for stale tensor forward cleanup — catches requests silently dropped by libp2p
/// (no OutboundFailure event) due to stale ConnectionIds or handler starvation.
const STALE_TENSOR_CLEANUP_SECS: u64 = 10;
/// Retention cutoff (seconds) for ping_sent_times entries when pruning under pressure.
const PING_SENT_TIMES_CUTOFF_SECS: u64 = 120;
/// Staleness threshold for PEX health-ping bookkeeping. Entries older than this
/// are dropped from `ping_sent_times` at each rr_ping tick because their
/// response is never arriving. Distinct from the 120s storm-guard cutoff —
/// this is the normal-operation liveness window.
const PEX_PING_STALENESS_SECS: u64 = 30;
/// Sliding window for inbound PEX rate limiting.
const PEX_WINDOW: std::time::Duration = std::time::Duration::from_secs(60);
/// Maximum inbound PEX requests allowed per window before dropping.
const PEX_MAX_PER_WINDOW: usize = 50;
/// Maximum pending tensor channels before rejecting new ones (memory exhaustion guard).
const MAX_PENDING_TENSOR_CHANNELS: usize = 256;
/// Maximum pending provider model-list queries before pruning oldest.
const MAX_PENDING_PROVIDER_QUERIES: usize = 500;
/// Maximum queued redial attempts for recently-disconnected peers.
const MAX_PENDING_REDIAL: usize = 50;
/// Maximum buffered gossip messages when no peers are connected at startup.
const MAX_BUFFERED_GOSSIP: usize = 64;
/// Maximum entries in connection_addrs before half-eviction of oldest ConnectionIds.
const MAX_CONNECTION_ADDRS: usize = 1024;
/// Maximum entries in ping_sent_times before pruning stale entries.
const MAX_PING_ENTRIES: usize = 2048;

/// Check if a multiaddr string contains a private/loopback/link-local/CGN IP.
/// Used for PEX filtering to prevent leaking internal topology.
fn is_non_public_addr(addr_str: &str) -> bool {
    if let Ok(addr) = addr_str.parse::<Multiaddr>() {
        addr.iter().any(|proto| match proto {
            libp2p::multiaddr::Protocol::Ip4(ip) => {
                ip.is_private()
                    || ip.is_loopback()
                    || ip.is_link_local()
                    || ip.is_unspecified()
                    // RFC 6598 CGN / Tailscale 100.64.0.0/10
                    || (ip.octets()[0] == 100 && (64..128).contains(&ip.octets()[1]))
                    // link-local metadata
                    || ip == std::net::Ipv4Addr::new(169, 254, 169, 254)
            }
            libp2p::multiaddr::Protocol::Ip6(ip) => {
                ip.is_loopback()
                    || (ip.segments()[0] & 0xffc0) == 0xfe80 // link-local
                    || (ip.segments()[0] & 0xfe00) == 0xfc00 // unique local (fd/fc)
            }
            _ => false,
        })
    } else {
        true // unparseable addresses are not public
    }
}

/// NetworkManager owns the libp2p Swarm and is the sole interface to the P2P network.
pub struct NetworkManager {
    shared_state: Arc<SharedState>,
    swarm: Swarm<SwarmBehaviour>,
    /// Receives commands from daemon tasks (broadcast, send tensor, etc.)
    inbound_rx: mpsc::Receiver<NetworkCommand>,
    /// Sends decoded network messages to the dispatcher for routing.
    /// Each message carries the transport-authenticated sender identity.
    outbound_tx: mpsc::Sender<crate::types::AuthenticatedMessage>,
    /// Sends shard data to the AcquisitionManager when received from peers.
    acquisition_tx: Option<mpsc::Sender<AcquisitionCommand>>,
    /// Deferred broadcasts queued during swarm event handling (can't publish gossip inline).
    deferred_broadcasts: Vec<crate::types::SwarmMessage>,
    /// Shard store for serving shard data to peers.
    shard_store: ShardStore,
    /// Maps OutboundRequestId → (PeerId, ShardId) for in-flight shard download requests.
    pending_shard_requests: HashMap<OutboundRequestId, (libp2p::PeerId, crate::types::ShardId)>,
    /// Tracks bytes downloaded so far per shard for chunked transfers.
    shard_download_progress: HashMap<crate::types::ShardId, u64>,
    /// P2P retry count per shard (max 5 before HF fallback).
    shard_p2p_retries: HashMap<crate::types::ShardId, u32>,
    /// Reverse lookup: PeerId → NodeId for O(1) peer identification.
    peer_to_node: DashMap<libp2p::PeerId, crate::types::NodeId>,
    /// Buffered GossipSub messages that failed to publish at startup (no peers yet).
    buffered_gossip: Vec<(String, Vec<u8>)>,
    /// Whether relay listen has been activated for this session (at most once).
    relay_activated: bool,
    /// Maps OutboundRequestId → (inference UUID, send time, target PeerId, num_layers, activation_bytes)
    /// for tensor forwards. Used to notify the pipeline on OutboundFailure.
    /// The Instant + workload info are used for adaptive stale tensor cleanup.
    pending_tensor_outbound:
        HashMap<OutboundRequestId, (uuid::Uuid, std::time::Instant, libp2p::PeerId, u32, usize)>,
    /// Holds ResponseChannels for pending tensor forwards, keyed by inference UUID.
    /// When a LayerForward arrives, we store the channel here instead of ACK-ing immediately.
    /// When the computed LayerResult comes back via NetworkCommand::SendTensorResult,
    /// we send the result as the response on the original channel — single substream per token.
    pending_tensor_channels: HashMap<
        uuid::Uuid,
        (
            std::time::Instant,
            request_response::ResponseChannel<SwarmResponse>,
        ),
    >,
    /// Maps ConnectionId → remote Multiaddr for each established connection.
    /// Used by the Identify handler to add only the *connected* address to Kademlia,
    /// not all listen_addrs (which causes redundant connections to the same peer on
    /// different addresses, leading to request_response round-robin routing failures).
    connection_addrs: HashMap<libp2p::swarm::ConnectionId, Multiaddr>,
    /// Tracks rr_ping send times per OutboundRequestId for RTT measurement.
    /// When the PEX response arrives, RTT is calculated and stored as `latency_ms`
    /// on the peer's PeerInfo, enabling tensor-parallelism group detection.
    ping_sent_times: HashMap<OutboundRequestId, (libp2p::PeerId, std::time::Instant)>,
    shutdown_rx: watch::Receiver<bool>,
    /// Peers to re-dial after a short delay (e.g., mDNS simultaneous-dial race).
    /// Stores (peer_id, address, scheduled_time). Checked every second in the event loop.
    pending_redial: Vec<(libp2p::PeerId, Multiaddr, std::time::Instant)>,
    /// S5: Receives model IDs for DHT provider queries from scheduler/auto-manage.
    dht_query_rx: mpsc::Receiver<crate::types::ModelId>,
    /// S5: Maps Kademlia QueryId → ShardId for routing GetProviders results.
    pending_provider_queries: HashMap<libp2p::kad::QueryId, crate::types::ShardId>,
    /// Aggregate PEX rate limiter: timestamps of recent inbound PEX requests.
    /// Bounded to a sliding window — rejects requests when the budget is exhausted.
    pex_inbound_timestamps: Vec<std::time::Instant>,
}

impl NetworkManager {
    /// Create a new NetworkManager and initialize the libp2p Swarm.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        shared_state: Arc<SharedState>,
        identity: &Identity,
        config: &Config,
        inbound_rx: mpsc::Receiver<NetworkCommand>,
        outbound_tx: mpsc::Sender<crate::types::AuthenticatedMessage>,
        shutdown_rx: watch::Receiver<bool>,
        acquisition_tx: Option<mpsc::Sender<AcquisitionCommand>>,
        dht_query_rx: mpsc::Receiver<crate::types::ModelId>,
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

        let keypair_for_behaviour = keypair.clone();
        let relay_cfg = relay_server_config;
        let enable_mdns = config.network.enable_mdns;
        let enable_autonat = config.network.enable_autonat;
        let enable_dcutr = config.network.enable_dcutr;
        // Load cached peer count to auto-scale GossipSub mesh parameters.
        let known_peers = crate::network::peer_cache::load_peer_cache(&shared_state.db).len()
            + config.network.bootstrap_peers.len();
        let swarm = SwarmBuilder::with_existing_identity(keypair)
            .with_tokio()
            .with_tcp(
                libp2p::tcp::Config::default().nodelay(true),
                libp2p::noise::Config::new,
                // Use yamux 0.13 defaults (auto-tuned windows, 1 GiB max connection window).
                // NOTE: Do NOT call set_receive_window_size or set_max_buffer_size — those
                // are deprecated and silently downgrade to yamux 0.12 which has severe
                // substream opening delays (~30s between successful outbound requests).
                libp2p::yamux::Config::default,
            )
            .map_err(|e| SwarmError::Network(format!("TCP transport error: {e}")))?
            .with_quic()
            .with_relay_client(libp2p::noise::Config::new, libp2p::yamux::Config::default)
            .map_err(|e| SwarmError::Network(format!("Relay client error: {e}")))?
            .with_behaviour(|_key, relay_behaviour| {
                behaviour::build_behaviour(
                    &keypair_for_behaviour,
                    relay_behaviour,
                    relay_cfg.as_ref(),
                    enable_mdns,
                    enable_autonat,
                    enable_dcutr,
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
                c.with_idle_connection_timeout(std::time::Duration::from_secs(
                    IDLE_CONNECTION_TIMEOUT_SECS,
                ))
                .with_notify_handler_buffer_size(std::num::NonZeroUsize::new(256).expect("256 > 0"))
                // Increase connection→swarm event buffer from default 7 to 64.
                // With many sub-behaviours (identify, kademlia, gossipsub, mdns),
                // the default 7-slot buffer fills during post-connect bursts,
                // blocking the connection task at events.send().await and preventing
                // it from processing inbound NotifyHandler commands (tensor forwards).
                .with_per_connection_event_buffer_size(64)
            })
            .build();

        let shard_store = ShardStore::new(&config.node.data_dir);

        Ok(Self {
            shared_state,
            swarm,
            inbound_rx,
            outbound_tx,
            acquisition_tx,
            deferred_broadcasts: Vec::new(),
            shard_store,
            pending_shard_requests: HashMap::new(),
            shard_download_progress: HashMap::new(),
            shard_p2p_retries: HashMap::new(),
            peer_to_node: DashMap::new(),
            buffered_gossip: Vec::new(),
            relay_activated: false,
            pending_tensor_outbound: HashMap::new(),
            pending_tensor_channels: HashMap::new(),
            connection_addrs: HashMap::new(),
            ping_sent_times: HashMap::new(),
            shutdown_rx,
            pending_redial: Vec::new(),
            dht_query_rx,
            pending_provider_queries: HashMap::new(),
            pex_inbound_timestamps: Vec::new(),
        })
    }

    /// Resolve a PeerId to a NodeId using the peer_to_node mapping.
    fn peer_to_node_id(&self, peer: &libp2p::PeerId) -> Option<crate::types::NodeId> {
        self.peer_to_node.get(peer).map(|r| r.value().clone())
    }

    /// Refresh `last_seen` on any inbound request/response traffic. Health monitor
    /// evicts peers at PING_INTERVAL × MAX_MISSED_PINGS (90s) of silence. On slow
    /// transports (e.g. WSL2 QUIC substream negotiation at 14–25s) PEX replies can
    /// arrive successfully but after specific dispatch handlers already declared
    /// the peer stale. Any rr activity proves liveness — treat it as a heartbeat.
    fn refresh_peer_last_seen(&self, peer: &libp2p::PeerId) {
        if let Some(node_id) = self.peer_to_node.get(peer) {
            if let Some(mut peer_info) = self.shared_state.peer_registry.get_mut(&*node_id) {
                peer_info.last_seen = chrono::Utc::now();
            }
        }
    }

    /// Send a SwarmMessage to the dispatcher with transport-authenticated sender.
    #[allow(clippy::result_large_err)]
    fn dispatch_authenticated(
        &self,
        sender_peer: Option<&libp2p::PeerId>,
        msg: SwarmMessage,
    ) -> Result<(), mpsc::error::TrySendError<crate::types::AuthenticatedMessage>> {
        let sender = sender_peer.and_then(|p| self.peer_to_node_id(p));
        self.outbound_tx
            .try_send(crate::types::AuthenticatedMessage {
                sender,
                message: msg,
            })
    }

    /// Start the network manager event loop.
    /// Send an error LayerResult to the pipeline for a failed tensor forward.
    fn fail_tensor_forward(
        &mut self,
        request_id: uuid::Uuid,
        peer: &libp2p::PeerId,
        reason: String,
    ) {
        let error_result = crate::types::LayerResult {
            request_id,
            token_ids: vec![],
            finish_reason: Some(crate::types::NetworkFinishReason::Error(reason)),
            activations: vec![],
            sealed_token_ids: None,
        };
        if let Err(e) =
            self.dispatch_authenticated(Some(peer), SwarmMessage::LayerResult(error_result))
        {
            tracing::warn!(error = %e, "Failed to send error LayerResult to pipeline");
        }
    }

    pub async fn run(mut self) -> Result<(), SwarmError> {
        let config = self.shared_state.config.clone();
        let port = config.node.listen_port;

        // Listen on QUIC and TCP
        // TCP P2P uses port+10 to avoid conflicting with the HTTP API server on the same TCP port.
        let listen_ip = &config.network.listen_address;
        let tcp_port = port + 10;
        let tcp_addr: Multiaddr = format!("/ip4/{listen_ip}/tcp/{tcp_port}")
            .parse()
            .map_err(|e| SwarmError::Network(format!("Invalid TCP address: {e}")))?;

        if config.network.enable_quic {
            let quic_addr: Multiaddr = format!("/ip4/{listen_ip}/udp/{port}/quic-v1")
                .parse()
                .map_err(|e| SwarmError::Network(format!("Invalid QUIC address: {e}")))?;
            self.swarm
                .listen_on(quic_addr.clone())
                .map_err(|e| SwarmError::Network(format!("Failed to listen on QUIC: {e}")))?;

            match self.swarm.listen_on(tcp_addr.clone()) {
                Ok(_) => tracing::info!(%quic_addr, %tcp_addr, "Listening for P2P connections"),
                Err(e) => {
                    tracing::warn!(%quic_addr, error = %e, "TCP listen unavailable, using QUIC only");
                }
            }
        } else {
            self.swarm
                .listen_on(tcp_addr.clone())
                .map_err(|e| SwarmError::Network(format!("Failed to listen on TCP: {e}")))?;
            tracing::info!(%tcp_addr, "Listening for P2P connections (QUIC disabled)");
        }

        // Subscribe to GossipSub topics
        discovery::subscribe_topics(&mut self.swarm)?;

        // Layer 2: Load cached peers from last session and dial them
        if config.pool.offline_mode {
            tracing::info!(
                "Offline LAN mode — skipping bootstrap peers and peer cache, mDNS discovery only"
            );
        } else {
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
        }

        // Periodic discovery timer
        let mut discovery_interval = tokio::time::interval(discovery::DISCOVERY_INTERVAL);
        discovery_interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

        // Periodic peer cache save timer (every 5 minutes)
        let mut peer_cache_interval = tokio::time::interval(discovery::PEER_CACHE_SAVE_INTERVAL);
        peer_cache_interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

        // Periodic send_request health check via PeerExchangeRequest.
        // Keeps the outbound queue nearly empty so tensor forwards get immediate service.
        let mut rr_ping_interval =
            tokio::time::interval(std::time::Duration::from_secs(RR_PING_INTERVAL_SECS));
        rr_ping_interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        let mut rr_ping_seq: u64 = 0;

        // Periodic stale tensor forward cleanup — catches requests that are silently dropped
        // by libp2p (no OutboundFailure event) due to stale ConnectionIds or handler starvation.
        let mut stale_tensor_interval =
            tokio::time::interval(std::time::Duration::from_secs(STALE_TENSOR_CLEANUP_SECS));
        stale_tensor_interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

        // Periodic redial check — processes pending_redial queue for peers that failed
        // initial connection (e.g., mDNS simultaneous-dial race).
        let mut redial_interval = tokio::time::interval(std::time::Duration::from_secs(1));
        redial_interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

        // Liveness heartbeat — every 30s, refresh `last_seen` on every peer
        // libp2p still reports as connected. The HealthMonitor evicts peers at
        // 90s of silence, but rr_ping fires every 120s — so a peer with no
        // other inbound traffic in its first 90s would be evicted while still
        // having a live TCP/QUIC connection (especially on WSL2 where QUIC
        // substream negotiation is slow). This tick is the floor.
        let mut liveness_interval = tokio::time::interval(std::time::Duration::from_secs(30));
        liveness_interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

        tracing::info!(
            target: "swarmllm::network::manager",
            port = self.shared_state.config.node.listen_port,
            "NetworkManager running"
        );

        loop {
            tokio::select! {
                // Shutdown signal
                _ = self.shutdown_rx.changed() => {
                    if *self.shutdown_rx.borrow() {
                        self.save_peer_cache();
                        tracing::info!(target: "swarmllm::network::manager", "NetworkManager shutting down");
                        break;
                    }
                }
                // Periodic discovery — skip when tensor forwards are in-flight to avoid
                // Kademlia event bursts that create back-pressure in the connection task's
                // event channel, delaying NotifyHandler delivery for tensor forwards.
                _ = discovery_interval.tick() => {
                    if !self.pending_tensor_outbound.is_empty() {
                        tracing::debug!(
                            pending = self.pending_tensor_outbound.len(),
                            "Skipping discovery tick — tensor forwards pending"
                        );
                    } else if self.shared_state.credits.offline_mode.load(std::sync::atomic::Ordering::Relaxed) {
                        // Offline mode: skip bootstrap/cache redials, rely on mDNS only
                    } else {
                        tracing::debug!("Discovery tick");
                        let _ = discovery::trigger_bootstrap(&mut self.swarm);
                        // Re-dial cached peers that we're not currently connected to.
                        // This handles peers that went offline and came back.
                        let cached = crate::network::peer_cache::load_peer_cache(&self.shared_state.db);
                        if !cached.is_empty() {
                            let _ = discovery::bootstrap_peers(&mut self.swarm, &cached);
                        }
                    }
                    self.update_peer_count();
                }
                // Periodic peer cache save
                _ = peer_cache_interval.tick() => {
                    self.save_peer_cache();
                }
                // Liveness heartbeat — refresh `last_seen` for every peer
                // libp2p still considers connected. Prevents the HealthMonitor
                // from evicting peers whose only proof of life is their TCP/QUIC
                // connection (no rr / gossip / identify traffic in the window).
                _ = liveness_interval.tick() => {
                    let connected: Vec<libp2p::PeerId> =
                        self.swarm.connected_peers().cloned().collect();
                    for peer_id in &connected {
                        self.refresh_peer_last_seen(peer_id);
                    }
                }
                // Periodic request_response health ping.
                // Skip when tensor forwards are pending — each ping consumes a substream
                // slot that could carry a tensor forward, and on slow QUIC transports the
                // queue backlog can exceed the pipeline timeout (30s).
                _ = rr_ping_interval.tick() => {
                    if !self.pending_tensor_outbound.is_empty() {
                        tracing::debug!(
                            pending = self.pending_tensor_outbound.len(),
                            "Skipping health ping — tensor forwards pending"
                        );
                    } else {
                        // Clean up stale ping_sent_times (response never arrived)
                        let cutoff = std::time::Instant::now()
                            - std::time::Duration::from_secs(PEX_PING_STALENESS_SECS);
                        self.ping_sent_times.retain(|_, (_, sent_at)| *sent_at > cutoff);

                        let peers: Vec<libp2p::PeerId> = self.swarm.connected_peers().cloned().collect();
                        if !peers.is_empty() {
                            rr_ping_seq += 1;
                            for peer_id in &peers {
                                let rr_connected = self.swarm.behaviour().request_response.is_connected(peer_id);
                                let req = SwarmRequest::Message(Box::new(SwarmMessage::PeerExchangeRequest));
                                let outbound_id = self.swarm.behaviour_mut().request_response.send_request(peer_id, req);
                                self.ping_sent_times.insert(outbound_id, (*peer_id, std::time::Instant::now()));
                                tracing::info!(
                                    %peer_id,
                                    ?outbound_id,
                                    seq = rr_ping_seq,
                                    rr_connected,
                                    total_peers = peers.len(),
                                    pending_tensor_out = self.pending_tensor_outbound.len(),
                                    "DIAG: rr_ping sent (health check)"
                                );
                            }
                        }
                    }
                }
                // Stale tensor forward cleanup — catches requests stuck in the handler
                // due to yamux/connection-task stalls where the libp2p SubstreamRequested
                // timeout (futures_timer::Delay) fails to fire. When detected, we disconnect
                // the stale peer to force a fresh TCP+yamux session on the next exchange.
                _ = stale_tensor_interval.tick() => {
                    // Sweep stale pending_tensor_channels independently of outbound state.
                    // On serving nodes, pending_tensor_outbound is empty but channels can still leak.
                    self.pending_tensor_channels.retain(|_uuid, (inserted, _chan)| {
                        inserted.elapsed().as_secs() < MAX_TENSOR_FORWARD_SECS
                    });
                    if !self.pending_tensor_outbound.is_empty() {
                        let now = std::time::Instant::now();
                        let mut stale: Vec<(OutboundRequestId, uuid::Uuid, libp2p::PeerId)> = Vec::new();
                        for (req_id, (uuid, sent_at, target_peer, num_layers, activation_bytes)) in &self.pending_tensor_outbound {
                            let age = now.duration_since(*sent_at);
                            // Adaptive timeout: 15s/layer for prefill, 2s/layer for decode,
                            // clamped to [30s, 600s]. Matches pipeline.rs logic.
                            let is_prefill = *activation_bytes > crate::inference::pipeline::PREFILL_ACTIVATION_THRESHOLD_BYTES;
                            let per_layer = if is_prefill { 15u64 } else { 2 };
                            let timeout_secs = ((*num_layers as u64) * per_layer).clamp(30, MAX_TENSOR_FORWARD_SECS);
                            let is_rr_pending = self.swarm.behaviour()
                                .request_response.is_pending_outbound(target_peer, req_id);
                            let is_connected = self.swarm.is_connected(target_peer);
                            let rr_connected = self.swarm.behaviour()
                                .request_response.is_connected(target_peer);
                            tracing::debug!(
                                ?req_id,
                                request_id = %uuid,
                                %target_peer,
                                age_secs = age.as_secs(),
                                timeout_secs,
                                is_rr_pending,
                                is_connected,
                                rr_connected,
                                "DIAG: pending tensor forward status"
                            );
                            if age.as_secs() > timeout_secs {
                                stale.push((*req_id, *uuid, *target_peer));
                            }
                        }
                        // Collect unique stale peers for disconnection
                        let mut stale_peers: std::collections::HashSet<libp2p::PeerId> = std::collections::HashSet::new();
                        for (req_id, uuid, target_peer) in stale {
                            tracing::warn!(
                                ?req_id,
                                request_id = %uuid,
                                %target_peer,
                                "DIAG: stale tensor forward — notifying pipeline + disconnecting peer"
                            );
                            self.pending_tensor_outbound.remove(&req_id);
                            stale_peers.insert(target_peer);
                            self.fail_tensor_forward(uuid, &target_peer, "Tensor forward timed out".into());
                            // Also clean up the inbound response channel for this request
                            // to prevent unbounded memory leak on timed-out distributed inference
                            if self.pending_tensor_channels.remove(&uuid).is_some() {
                                tracing::debug!(
                                    request_id = %uuid,
                                    "Cleaned up stale pending_tensor_channel"
                                );
                            }
                        }
                        // Disconnect stale peers to reset the yamux session.
                        // The connection task may be stuck (handler not polled, SubstreamRequested
                        // timeout not firing). Disconnecting kills the stale TCP+yamux session.
                        // The peer will be reconnected on the next send_request() or Kademlia
                        // bootstrap (60s interval).
                        for peer in &stale_peers {
                            if self.swarm.is_connected(peer) {
                                tracing::warn!(
                                    %peer,
                                    "DIAG: disconnecting stale peer to reset yamux session"
                                );
                                let _ = self.swarm.disconnect_peer_id(*peer);
                            }
                        }
                    }
                }
                // Process pending re-dials (mDNS simultaneous-dial race recovery).
                // When both sides discover each other via mDNS at the same time, both dial,
                // and with max_established_per_peer=1, the loser's connection is immediately
                // closed. We schedule a re-dial with random jitter (2-5s) so one side wins.
                _ = redial_interval.tick() => {
                    if !self.pending_redial.is_empty() {
                        let now = std::time::Instant::now();
                        let ready: Vec<_> = self.pending_redial
                            .iter()
                            .enumerate()
                            .filter(|(_, (_, _, scheduled))| now >= *scheduled)
                            .map(|(i, (peer_id, addr, _))| (i, *peer_id, addr.clone()))
                            .collect();
                        // Remove in reverse order to preserve indices
                        for (i, peer_id, addr) in ready.iter().rev() {
                            self.pending_redial.remove(*i);
                            if !self.swarm.is_connected(peer_id) {
                                let opts = libp2p::swarm::dial_opts::DialOpts::peer_id(*peer_id)
                                    .condition(libp2p::swarm::dial_opts::PeerCondition::Disconnected)
                                    .addresses(vec![addr.clone()])
                                    .build();
                                match self.swarm.dial(opts) {
                                    Ok(()) => tracing::info!(
                                        %peer_id, %addr,
                                        "Re-dialing peer after connection race"
                                    ),
                                    Err(e) => tracing::debug!(
                                        %peer_id, error = %e,
                                        "Re-dial skipped"
                                    ),
                                }
                            }
                        }
                        // Cap queue to prevent unbounded growth
                        if self.pending_redial.len() > MAX_PENDING_REDIAL {
                            self.pending_redial.truncate(MAX_PENDING_REDIAL);
                        }
                    }
                }
                // S5: DHT provider queries from scheduler/auto-manage
                Some(model_id) = self.dht_query_rx.recv() => {
                    self.handle_dht_provider_query(&model_id);
                }
                // Outbound commands from other daemon tasks
                cmd = self.inbound_rx.recv() => {
                    match cmd {
                        Some(cmd) => {
                            self.handle_outbound_command(cmd).await;
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
                    // Process any broadcasts queued during event handling
                    if !self.deferred_broadcasts.is_empty() {
                        let msgs: Vec<_> = self.deferred_broadcasts.drain(..).collect();
                        for msg in msgs {
                            self.handle_broadcast(msg).await;
                        }
                    }
                }
            }
        }

        Ok(())
    }

    async fn handle_swarm_event(&mut self, event: SwarmEvent<SwarmBehaviourEvent>) {
        tracing::debug!(event_type = %swarm_event_name(&event), "DIAG: processing swarm event");
        match event {
            // ── GossipSub messages ──
            SwarmEvent::Behaviour(SwarmBehaviourEvent::Gossipsub(gossipsub::Event::Message {
                propagation_source,
                message,
                ..
            })) => {
                // SEC: All gossip MUST be signed + sealed. No unsigned fallback.
                let decoded = self
                    .shared_state
                    .gossip_sealer
                    .open_signed(&message.data)
                    .map_err(|e| {
                        tracing::warn!(
                            source = ?message.source,
                            error = %e,
                            "Rejecting unsigned/invalid gossip message"
                        );
                        e
                    })
                    .and_then(|(sender_pub, plaintext)| {
                        let msg = protocol::decode_message(&plaintext)?;
                        Ok((crate::types::NodeId(sender_pub), msg))
                    });

                match decoded {
                    Ok((sender_node_id, msg)) => {
                        // NET-M10: Reject gossip messages with timestamps older than 5 minutes
                        let now_epoch = chrono::Utc::now().timestamp() as u64;
                        let too_old = match &msg {
                            SwarmMessage::HealthPing { timestamp, .. }
                            | SwarmMessage::HealthPong { timestamp, .. } => {
                                now_epoch.saturating_sub(*timestamp) > 300
                                    || timestamp.saturating_sub(now_epoch) > 300
                            }
                            SwarmMessage::ShardAnnounce(ann) => {
                                let ts = ann.timestamp.timestamp() as u64;
                                now_epoch.saturating_sub(ts) > 300
                                    || ts.saturating_sub(now_epoch) > 300
                            }
                            SwarmMessage::CreditGossip(gossip) => {
                                let ts = gossip.timestamp.timestamp() as u64;
                                now_epoch.saturating_sub(ts) > 300
                                    || ts.saturating_sub(now_epoch) > 300
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
                                sender = %sender_node_id,
                                "Received signed GossipSub message"
                            );
                            let authed = crate::types::AuthenticatedMessage {
                                sender: Some(sender_node_id),
                                message: msg,
                            };
                            if let Err(e) = self.outbound_tx.try_send(authed) {
                                self.shared_state
                                    .metrics
                                    .channel_metrics
                                    .network_out
                                    .record_dropped();
                                tracing::warn!(error = %e, "Dispatcher backpressured, dropping gossipsub message");
                            } else {
                                self.shared_state
                                    .metrics
                                    .channel_metrics
                                    .network_out
                                    .record_sent();
                            }
                        }
                    }
                    Err(e) => {
                        tracing::debug!(error = %e, "Failed to decode/verify gossipsub message");
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
                    let was_tensor = self.pending_tensor_outbound.contains_key(&request_id);
                    tracing::info!(
                        %peer,
                        kind,
                        ?request_id,
                        was_tensor_forward = was_tensor,
                        pending_tensor_out = self.pending_tensor_outbound.len(),
                        "DIAG: received response"
                    );
                    // Clean up tensor outbound tracking (response received = not a failure)
                    self.pending_tensor_outbound.remove(&request_id);
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
                tracing::warn!(
                    %peer,
                    ?request_id,
                    %error,
                    is_connected = self.swarm.is_connected(&peer),
                    pending_tensor_out = self.pending_tensor_outbound.len(),
                    pending_channels = self.pending_tensor_channels.len(),
                    "DIAG: OutboundFailure"
                );
                // Check if this was a pending tensor forward — notify the pipeline
                if let Some((inference_uuid, sent_at, _target, _, _)) =
                    self.pending_tensor_outbound.remove(&request_id)
                {
                    let age_ms = sent_at.elapsed().as_millis();
                    tracing::error!(
                        %peer,
                        inference_request_id = %inference_uuid,
                        %error,
                        age_ms,
                        "Tensor forward OutboundFailure — notifying pipeline"
                    );
                    // Send an error LayerResult so the pipeline can failover immediately
                    self.fail_tensor_forward(
                        inference_uuid,
                        &peer,
                        format!("OutboundFailure: {error}"),
                    );
                }
                // Check if this was a pending shard download request
                if let Some((_peer_id, shard_id)) = self.pending_shard_requests.remove(&request_id)
                {
                    let progress = self
                        .shard_download_progress
                        .get(&shard_id)
                        .copied()
                        .unwrap_or(0);
                    tracing::error!(
                        %peer,
                        model = %shard_id.model_id,
                        shard_index = shard_id.index,
                        %error,
                        bytes_downloaded = progress,
                        "DIAG: shard download OutboundFailure"
                    );
                    {
                        let mname = self
                            .shared_state
                            .model_registry
                            .get_manifest(&shard_id.model_id)
                            .map(|m| m.name.clone());
                        self.shared_state.emit_activity(
                            crate::daemon::state::ActivityEvent::new(
                                "download",
                                "shard_transfer_failed",
                                format!(
                                    "P2P transfer failed: shard {} of {} — {} ({}B received)",
                                    crate::types::ShardId::display_index_short(shard_id.index),
                                    mname.as_deref().unwrap_or(&shard_id.model_id.0),
                                    error,
                                    progress
                                ),
                            )
                            .with_model(shard_id.model_id.0.clone())
                            .with_node(format!("{}", peer))
                            .with_detail_num(shard_id.index as i64)
                            .with_detail_str(format!("{}", error))
                            .with_toast("warning", 6000),
                        );
                    }
                    // Clean up stale download progress entry to prevent resource leak
                    self.shard_download_progress.remove(&shard_id);
                }
            }
            SwarmEvent::Behaviour(SwarmBehaviourEvent::RequestResponse(
                request_response::Event::InboundFailure {
                    peer,
                    request_id,
                    error,
                    ..
                },
            )) => {
                // Note: pending_tensor_channels is keyed by Uuid (from the parsed
                // message), not InboundRequestId — we can't directly remove the entry
                // here. The stale timeout cleanup (every 30s) handles orphaned channels.
                tracing::warn!(
                    %peer,
                    ?request_id,
                    %error,
                    pending_channels = self.pending_tensor_channels.len(),
                    "DIAG: InboundFailure — response send may have failed, stale cleanup will reclaim"
                );
            }
            SwarmEvent::Behaviour(SwarmBehaviourEvent::RequestResponse(
                request_response::Event::ResponseSent {
                    peer, request_id, ..
                },
            )) => {
                tracing::info!(%peer, ?request_id, "DIAG: ResponseSent event — response written to wire");
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
                crate::network::relay::handle_relay_server_event(event, &self.shared_state);
            }

            // ── AutoNAT status changes ──
            SwarmEvent::Behaviour(SwarmBehaviourEvent::Autonat(
                libp2p::autonat::Event::StatusChanged { old, new },
            )) => {
                tracing::info!(?old, ?new, "AutoNAT status changed");
                {
                    if let Ok(mut stats) = self.shared_state.metrics.node_stats.try_write() {
                        stats.nat_status = Some(format!("{new:?}"));
                    }
                }
                // NET-M3: Auto-listen on relay when NAT is detected as Private
                if matches!(new, libp2p::autonat::NatStatus::Private)
                    && !self.relay_activated
                    && self.shared_state.config.network.auto_relay
                {
                    self.relay_activated = true;
                    tracing::info!(target: "swarmllm::network::manager", "NAT detected, activating relay listener");

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
                    connection_id,
                },
            )) => {
                self.handle_identify_received(peer_id, info, connection_id)
                    .await;
            }

            // ── Kademlia ──
            SwarmEvent::Behaviour(SwarmBehaviourEvent::Kademlia(
                libp2p::kad::Event::OutboundQueryProgressed { id, result, .. },
            )) => {
                use libp2p::kad::QueryResult;
                match result {
                    QueryResult::GetRecord(Ok(libp2p::kad::GetRecordOk::FoundRecord(
                        peer_record,
                    ))) => {
                        // Verify Ed25519 signature on DHT records before trusting
                        match crate::network::discovery::verify_dht_value(&peer_record.record.value)
                        {
                            Ok((pubkey, payload)) => {
                                tracing::debug!(
                                    key = ?peer_record.record.key,
                                    signer = %hex::encode(&pubkey[..8]),
                                    payload_len = payload.len(),
                                    "DHT record verified"
                                );
                                // Process verified payload: deserialize NodeCapability
                                // and update peer registry with the advertised capabilities.
                                let key_bytes = peer_record.record.key.as_ref();
                                let key_str = String::from_utf8_lossy(key_bytes);
                                if key_str.starts_with("/swarm/node/") {
                                    if let Ok(cap) = serde_json::from_slice::<
                                        crate::types::NodeCapability,
                                    >(payload)
                                    {
                                        let node_id = crate::types::NodeId(pubkey);
                                        if let Some(mut entry) =
                                            self.shared_state.peer_registry.get_mut(&node_id)
                                        {
                                            entry.capability = Some(cap);
                                        }
                                    }
                                }
                            }
                            Err(_) => {
                                tracing::warn!(
                                    key = ?peer_record.record.key,
                                    "DHT record failed signature verification — ignoring"
                                );
                            }
                        }
                    }
                    // S5: DHT provider query results — merge discovered holders
                    // into the bounded shard_holders cache.
                    QueryResult::GetProviders(Ok(
                        libp2p::kad::GetProvidersOk::FoundProviders { providers, .. },
                    )) => {
                        self.handle_dht_providers_found(id, &providers);
                    }
                    QueryResult::GetProviders(Ok(
                        libp2p::kad::GetProvidersOk::FinishedWithNoAdditionalRecord { .. },
                    )) => {
                        // Query finished — clean up tracking
                        self.pending_provider_queries.remove(&id);
                    }
                    QueryResult::GetProviders(Err(ref e)) => {
                        tracing::debug!(error = ?e, "DHT provider query failed — cleaning up");
                        self.pending_provider_queries.remove(&id);
                    }
                    _ => {
                        tracing::debug!(?result, "Kademlia query progressed");
                    }
                }
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
                            .condition(libp2p::swarm::dial_opts::PeerCondition::Disconnected)
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
                                // Increment LAN peer count and notify via unified activity event
                                let count = self
                                    .shared_state
                                    .lan_peer_count
                                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
                                    + 1;
                                let msg = format!(
                                    "Found {} peer{} on your local network",
                                    count,
                                    if count == 1 { "" } else { "s" }
                                );
                                tracing::info!(lan_peers = count, message = %msg, "LAN peer discovery update");
                                self.shared_state.emit_activity(
                                    crate::daemon::state::ActivityEvent::new(
                                        "network",
                                        "lan_peer_discovered",
                                        msg,
                                    )
                                    .with_detail_num(count as i64)
                                    .with_toast("success", 8000),
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
                                let _ = self.shared_state.lan_peer_count.fetch_update(
                                    std::sync::atomic::Ordering::Relaxed,
                                    std::sync::atomic::Ordering::Relaxed,
                                    |v| v.checked_sub(1),
                                );
                            }
                        }
                    }
                }
            }

            SwarmEvent::ConnectionEstablished {
                peer_id,
                connection_id,
                num_established,
                endpoint,
                ..
            } => {
                self.handle_connection_established(
                    peer_id,
                    connection_id,
                    num_established,
                    &endpoint,
                );
            }

            SwarmEvent::ConnectionClosed {
                peer_id,
                connection_id,
                cause,
                num_established,
                ..
            } => {
                self.handle_connection_closed(peer_id, connection_id, cause, num_established);
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

            SwarmEvent::OutgoingConnectionError { peer_id, error, .. } => {
                tracing::debug!(
                    ?peer_id, %error,
                    "Outgoing connection failed"
                );
            }

            SwarmEvent::IncomingConnectionError { error, .. } => {
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

    /// Handle Identify protocol — peer identified, establish encryption, register in peer_registry.
    async fn handle_identify_received(
        &mut self,
        peer_id: libp2p::PeerId,
        info: libp2p::identify::Info,
        connection_id: libp2p::swarm::ConnectionId,
    ) {
        tracing::debug!(
            %peer_id,
            protocol_version = %info.protocol_version,
            listen_addrs = ?info.listen_addrs,
            "Identified peer"
        );
        // Add ONLY the connected address to Kademlia — not all listen_addrs.
        // Adding all addresses causes Kademlia to route DHT queries through
        // addresses we haven't connected on, triggering redundant dials that
        // create multiple connections per peer. request_response round-robins
        // across connections, and degraded connections silently drop messages.
        if let Some(connected_addr) = self.connection_addrs.get(&connection_id) {
            self.swarm
                .behaviour_mut()
                .kademlia
                .add_address(&peer_id, connected_addr.clone());
            tracing::debug!(
                %peer_id,
                addr = %connected_addr,
                "Added connected address to Kademlia (skipped {} other listen_addrs)",
                info.listen_addrs.len().saturating_sub(1)
            );
        } else if let Some(addr) = info.listen_addrs.first() {
            // Fallback: connection_id not tracked (shouldn't happen)
            self.swarm
                .behaviour_mut()
                .kademlia
                .add_address(&peer_id, addr.clone());
            tracing::warn!(
                %peer_id,
                "No tracked connection address for Identify, used first listen_addr"
            );
        }
        // Verify announced key matches the authenticated PeerId from Noise handshake
        // to prevent NodeId spoofing via forged Identify messages.
        let announced_peer_id = info.public_key.to_peer_id();
        if announced_peer_id != peer_id {
            tracing::warn!(
                %peer_id,
                announced = %announced_peer_id,
                "Peer announced mismatched public key in Identify — ignoring"
            );
            return;
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
                tracing::info!(
                    %peer_id,
                    node_id = %node_id,
                    session_type = "static",
                    "DIAG: key exchange initiated"
                );
                self.shared_state
                    .session_manager
                    .establish_session(&node_id, x25519_pub);
                tracing::info!(
                    %peer_id,
                    node_id = %node_id,
                    session_type = "static",
                    session_count = self.shared_state.session_manager.session_count(),
                    "DIAG: encryption session established"
                );
            }
        }

        let now_ts = crate::types::unix_now_secs();
        // Preserve first_seen from existing entry or use current time
        let first_seen = self
            .shared_state
            .peer_registry
            .get(&node_id)
            .map(|p| p.first_seen)
            .unwrap_or(now_ts);
        // Preserve trust, capability, and verified count from existing entry
        let existing = self.shared_state.peer_registry.get(&node_id);
        let trust_score = existing.as_ref().map(|p| p.trust_score).unwrap_or(0.5);
        let capability = existing.as_ref().and_then(|p| p.capability.clone());
        let vtc = existing
            .as_ref()
            .map(|p| p.verified_transaction_count)
            .unwrap_or(0);
        let is_lan = existing.as_ref().map(|p| p.is_lan_peer).unwrap_or(false);
        drop(existing);
        let peer_info = PeerInfo {
            node_id: node_id.clone(),
            addresses: info
                .listen_addrs
                .iter()
                .take(8)
                .map(|a| a.to_string())
                .collect(),
            capability,
            last_seen: chrono::Utc::now(),
            latency_ms: None,
            trust_score,
            peer_id_bytes: Some(peer_id.to_bytes()),
            active_request_count: 0,
            first_seen,
            verified_transaction_count: vtc,
            is_lan_peer: is_lan,
        };
        // Insert peer_registry BEFORE peer_to_node to prevent TOCTOU race
        // where dispatch can resolve NodeId from peer_to_node but peer_registry
        // check fails because insert hasn't happened yet.
        self.shared_state
            .peer_registry
            .insert(node_id.clone(), peer_info);
        // Restore persisted trust score from DB (survives restarts)
        let persisted_trust = self.shared_state.credits.trust_manager.get_trust(&node_id);
        if (persisted_trust - 0.5_f32).abs() > f32::EPSILON {
            if let Some(mut peer) = self.shared_state.peer_registry.get_mut(&node_id) {
                peer.trust_score = persisted_trust;
            }
        }
        self.shared_state
            .signal_dashboard(crate::daemon::state::DashboardSignal::PeersChanged);

        // Emit activity event for peer connection
        {
            let label = crate::identity::nickname::short_display_name(
                &node_id,
                &self.shared_state.nickname_registry,
            );
            let gpu_name = self.shared_state.peer_registry.get(&node_id).and_then(|p| {
                p.capability
                    .as_ref()
                    .and_then(|c| c.gpu.as_ref().map(|g| g.name.clone()))
            });
            let detail = if is_lan { "LAN" } else { "WAN" };
            self.shared_state.emit_activity(
                crate::daemon::state::ActivityEvent::new(
                    "network",
                    "peer_connected",
                    format!(
                        "Peer connected: {}{}",
                        label,
                        gpu_name
                            .as_ref()
                            .map(|g| format!(" ({})", g))
                            .unwrap_or_default()
                    ),
                )
                .with_node(format!("{}", node_id))
                .with_detail_str(detail.to_string()),
            );
        }

        // S3: Cap peer_registry to prevent unbounded growth at 10K+ nodes.
        // Evict highest-latency non-LAN non-pipeline peer when over limit.
        const MAX_PEER_REGISTRY: usize = 200;
        if self.shared_state.peer_registry.len() > MAX_PEER_REGISTRY {
            // Find the worst peer to evict: highest latency, not LAN, not in active pipeline
            let active_pipeline_nodes: std::collections::HashSet<_> = {
                let segments: Vec<_> = self
                    .shared_state
                    .active_pipelines
                    .iter()
                    .flat_map(|e| {
                        e.value()
                            .segments
                            .iter()
                            .map(|s| s.node_id.clone())
                            .collect::<Vec<_>>()
                    })
                    .collect();
                segments.into_iter().collect()
            };
            let evict_candidate = self
                .shared_state
                .peer_registry
                .iter()
                .filter(|e| {
                    !e.is_lan_peer
                        && !active_pipeline_nodes.contains(e.key())
                        && *e.key() != node_id
                })
                // Prefer evicting peers with known high latency over unmeasured peers.
                // Unmeasured peers (None) get 0 so they survive until measured.
                .max_by_key(|e| e.latency_ms.unwrap_or(0))
                .map(|e| e.key().clone());
            if let Some(evict_id) = evict_candidate {
                self.shared_state.peer_registry.remove(&evict_id);
                // Also remove from peer_to_node and disconnect
                let evict_peer = self
                    .peer_to_node
                    .iter()
                    .find(|e| *e.value() == evict_id)
                    .map(|e| *e.key());
                if let Some(pid) = evict_peer {
                    self.peer_to_node.remove(&pid);
                    let _ = self.swarm.disconnect_peer_id(pid);
                }
                tracing::debug!(
                    evicted = %evict_id,
                    registry_size = self.shared_state.peer_registry.len(),
                    "Evicted distant peer to stay under registry cap"
                );
            }
        }

        // NET-C4: Populate reverse PeerId → NodeId lookup (capped)
        const MAX_PEER_TO_NODE: usize = 10_000;
        if self.peer_to_node.len() < MAX_PEER_TO_NODE || self.peer_to_node.contains_key(&peer_id) {
            self.peer_to_node.insert(peer_id, node_id.clone());
        }
        // Persistent NodeId → PeerId mapping (survives disconnects, same cap)
        if self.shared_state.peer_id_map.len() < MAX_PEER_TO_NODE
            || self.shared_state.peer_id_map.contains_key(&node_id)
        {
            self.shared_state
                .peer_id_map
                .insert(node_id.clone(), peer_id.to_bytes());
        }

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
                if let Ok(mut anti_gaming) = self.shared_state.credits.anti_gaming.try_lock() {
                    anti_gaming.register_subnet(&node_id, ip_bytes);
                }
                break; // One IP per peer is enough
            }
        }
    }

    /// Handle new peer connection — track address, send PEX request.
    fn handle_connection_established(
        &mut self,
        peer_id: libp2p::PeerId,
        connection_id: libp2p::swarm::ConnectionId,
        num_established: std::num::NonZeroU32,
        endpoint: &libp2p::core::ConnectedPoint,
    ) {
        let remote_addr = endpoint.get_remote_address();
        let is_loopback = remote_addr.iter().any(|proto| {
            matches!(proto, libp2p::multiaddr::Protocol::Ip4(ip) if ip.is_loopback())
                || matches!(proto, libp2p::multiaddr::Protocol::Ip6(ip) if ip.is_loopback())
        });
        tracing::info!(
            %peer_id, %connection_id, count = num_established,
            remote_addr = %remote_addr,
            is_loopback,
            is_dialer = endpoint.is_dialer(),
            total_established = self.swarm.network_info().connection_counters().num_established(),
            total_peers = self.swarm.connected_peers().count(),
            pending_tensor_forwards = self.pending_tensor_outbound.len(),
            "DIAG: connection established"
        );
        // Track which address each connection uses — the Identify handler
        // uses this to add only the connected address to Kademlia.
        // SEC: Cap connection_addrs to prevent unbounded memory growth.
        if self.connection_addrs.len() >= MAX_CONNECTION_ADDRS {
            // Evict oldest half — stale ConnectionIds from missed close events.
            let mut ids: Vec<_> = self.connection_addrs.keys().cloned().collect();
            ids.sort();
            for id in ids.iter().take(MAX_CONNECTION_ADDRS / 2) {
                self.connection_addrs.remove(id);
            }
        }
        self.connection_addrs
            .insert(connection_id, remote_addr.clone());
        self.update_peer_count();

        // Layer 5: Peer Exchange — send PEX request on first connection only
        if num_established.get() == 1 && self.shared_state.config.network.peer_exchange {
            // SEC: Cap ping_sent_times to prevent unbounded growth from connection storms.
            // Prune stale entries before inserting.
            if self.ping_sent_times.len() >= MAX_PING_ENTRIES {
                let cutoff = std::time::Instant::now()
                    - std::time::Duration::from_secs(PING_SENT_TIMES_CUTOFF_SECS);
                self.ping_sent_times
                    .retain(|_, (_, sent_at)| *sent_at > cutoff);
            }
            let req = SwarmRequest::Message(Box::new(SwarmMessage::PeerExchangeRequest));
            let outbound_id = self
                .swarm
                .behaviour_mut()
                .request_response
                .send_request(&peer_id, req);
            // Track send time for RTT measurement
            self.ping_sent_times
                .insert(outbound_id, (peer_id, std::time::Instant::now()));
            tracing::debug!(%peer_id, "Sent PEX request");
        }
    }

    /// Handle peer disconnection — cleanup registry, sessions, downloads.
    fn handle_connection_closed(
        &mut self,
        peer_id: libp2p::PeerId,
        connection_id: libp2p::swarm::ConnectionId,
        cause: Option<libp2p::swarm::ConnectionError>,
        num_established: u32,
    ) {
        let closed_addr = self.connection_addrs.remove(&connection_id);
        // Check if any in-flight tensor forwards are affected
        let affected_tensors: Vec<_> = self
            .pending_tensor_outbound
            .values()
            .map(|(u, _, _, _, _)| u.to_string())
            .collect();
        tracing::warn!(
            %peer_id, %connection_id, ?cause, remaining = num_established,
            pending_tensor_forwards = self.pending_tensor_outbound.len(),
            affected_request_ids = ?affected_tensors.iter().take(5).collect::<Vec<_>>(),
            total_peers = self.swarm.connected_peers().count(),
            "DIAG: connection closed"
        );

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
                    self.shard_p2p_retries.remove(&shard_id);
                    tracing::debug!(
                        %peer_id,
                        model = %shard_id.model_id,
                        index = shard_id.index,
                        "Cleaned up pending shard request for disconnected peer"
                    );
                }
            }

            // NET-I3: Clean up peer_shard_downloads for disconnected peer.
            // Entries for peers that disconnect mid-download would otherwise
            // be orphaned permanently, accumulating stale data.
            let node_id_for_cleanup = self.peer_to_node.get(&peer_id).map(|r| r.clone());
            if let Some(ref nid) = node_id_for_cleanup {
                self.shared_state
                    .models
                    .peer_shard_downloads
                    .retain(|_shard_id, peers| {
                        peers.retain(|(n, _)| n != nid);
                        !peers.is_empty()
                    });

                // NET-I4: Clean up stale peer_credit_balances entry.
                // Prevents unbounded growth and stale entries skewing priority tier percentiles.
                self.shared_state.credits.peer_credit_balances.remove(nid);
            }

            // NET-I2: Remove peer from registry, but skip if in active pipelines.
            // Clone the NodeId and drop the DashMap Ref BEFORE calling remove(),
            // otherwise get() holds a read lock and remove() needs a write lock
            // on the same shard → synchronous deadlock that freezes the event loop.
            let node_id_opt = node_id_for_cleanup;
            if let Some(node_id) = node_id_opt {
                let in_active_pipeline = self.shared_state.active_pipelines.iter().any(|entry| {
                    entry
                        .value()
                        .segments
                        .iter()
                        .any(|seg| seg.node_id == node_id)
                });
                // Clear encryption session on full disconnect to
                // prevent epoch desync after reconnection.
                // Only remove if no new connection has been established
                // (prevents race where reconnect arrives before close is processed).
                // Keep the session alive if the peer is in an active pipeline —
                // reconnection will refresh it, and removing it mid-pipeline
                // causes "seal() failed" on pending TP forwards.
                if !self.swarm.is_connected(&peer_id) && !in_active_pipeline {
                    self.shared_state.session_manager.remove_session(&node_id);
                }

                if !in_active_pipeline {
                    // Capture info before removing
                    let label = crate::identity::nickname::short_display_name(
                        &node_id,
                        &self.shared_state.nickname_registry,
                    );

                    // Remove peer_to_node BEFORE peer_registry to prevent
                    // dispatch from resolving NodeId for a peer that's being removed
                    self.peer_to_node.remove(&peer_id);
                    self.shared_state.peer_registry.remove(&node_id);
                    self.shared_state
                        .signal_dashboard(crate::daemon::state::DashboardSignal::PeersChanged);

                    self.shared_state.emit_activity(
                        crate::daemon::state::ActivityEvent::new(
                            "network",
                            "peer_disconnected",
                            format!("Peer disconnected: {}", label),
                        )
                        .with_node(format!("{}", node_id)),
                    );
                    tracing::debug!(%peer_id, "Removed disconnected peer from registry");
                } else {
                    tracing::info!(%peer_id, "Keeping peer in registry (active pipeline) — scheduling reconnect");
                    // Active pipeline needs this peer — reconnect immediately.
                    // Use peer_id_map to find the address, or fall back to closed_addr.
                    if let Some(addr) = closed_addr.clone() {
                        let already_queued = self
                            .pending_redial
                            .iter()
                            .any(|(pid, _, _)| *pid == peer_id);
                        if !already_queued {
                            let scheduled =
                                std::time::Instant::now() + std::time::Duration::from_millis(500);
                            self.pending_redial.push((peer_id, addr, scheduled));
                        }
                    }
                }
            } else {
                // Peer was never registered (connection died before Identify).
                // This typically happens during mDNS simultaneous-dial race.
                // Schedule a re-dial with random jitter to break symmetry.
                if let Some(addr) = closed_addr {
                    // Only re-dial if not already in the queue
                    let already_queued = self
                        .pending_redial
                        .iter()
                        .any(|(pid, _, _)| *pid == peer_id);
                    if !already_queued && self.pending_redial.len() < MAX_PENDING_REDIAL {
                        use std::hash::{Hash, Hasher};
                        let mut hasher = std::collections::hash_map::DefaultHasher::new();
                        peer_id.hash(&mut hasher);
                        let jitter_ms = 2000 + (hasher.finish() % 3000); // 2-5s
                        let scheduled =
                            std::time::Instant::now() + std::time::Duration::from_millis(jitter_ms);
                        tracing::info!(
                            %peer_id, %addr, jitter_ms,
                            "Scheduling re-dial after connection race"
                        );
                        self.pending_redial.push((peer_id, addr, scheduled));
                    }
                }
            }
        } // end else (num_established == 0)
    }
    async fn handle_request(
        &mut self,
        peer: libp2p::PeerId,
        request: SwarmRequest,
        channel: request_response::ResponseChannel<SwarmResponse>,
    ) {
        self.refresh_peer_last_seen(&peer);
        match request {
            SwarmRequest::Message(mut msg) => {
                // Handle PEX messages inline instead of forwarding to dispatcher
                match *msg {
                    SwarmMessage::PeerExchangeRequest => {
                        let now_pex = std::time::Instant::now();
                        self.pex_inbound_timestamps
                            .retain(|t| now_pex.duration_since(*t) < PEX_WINDOW);
                        if self.pex_inbound_timestamps.len() >= PEX_MAX_PER_WINDOW {
                            tracing::debug!(%peer, limit = PEX_MAX_PER_WINDOW, window_secs = PEX_WINDOW.as_secs(), "PEX rate limit exceeded, dropping request");
                            let _ = self
                                .swarm
                                .behaviour_mut()
                                .request_response
                                .send_response(channel, SwarmResponse::Ack);
                            return;
                        }
                        self.pex_inbound_timestamps.push(now_pex);
                        tracing::debug!(%peer, "Handling PEX request");
                        // Respond with up to 20 known peer addresses (filter out self)
                        let local_node_id = self.shared_state.identity.node_id();
                        let peers: Vec<String> = self
                            .shared_state
                            .peer_registry
                            .iter()
                            .filter(|entry| entry.key() != local_node_id)
                            .flat_map(|entry| entry.addresses.clone())
                            .filter(|addr| !is_non_public_addr(addr))
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
                        // SEC: Only process PEX from registered peers to prevent topology manipulation
                        let is_registered = self
                            .peer_to_node
                            .get(&peer)
                            .is_some_and(|n| self.shared_state.peer_registry.contains_key(&*n));
                        if !is_registered {
                            tracing::debug!(%peer, "Ignoring PEX response from unregistered peer");
                        } else {
                            tracing::debug!(%peer, count = pex_resp.peers.len(), "Received PEX response (via request)");
                            self.handle_pex_response(&pex_resp.peers);
                        }
                        // ACK and return
                        let _ = self
                            .swarm
                            .behaviour_mut()
                            .request_response
                            .send_response(channel, SwarmResponse::Ack);
                        return;
                    }
                    _ => {
                        // Attach sender peer identity for messages that need it
                        match &mut *msg {
                            SwarmMessage::VisionEncodeRequest(ref mut req) => {
                                req.sender_peer_bytes = Some(peer.to_bytes());
                            }
                            SwarmMessage::TpAllReduceRequest(ref mut req) => {
                                req.sender_peer_bytes = Some(peer.to_bytes());
                            }
                            _ => {}
                        }
                        // Forward all other messages to dispatcher with authenticated sender
                        tracing::debug!(%peer, "Handling protocol message request");
                        if let Err(e) = self.dispatch_authenticated(Some(&peer), *msg) {
                            self.shared_state
                                .metrics
                                .channel_metrics
                                .network_out
                                .record_dropped();
                            tracing::warn!(error = %e, "Dispatcher backpressured, dropping request message");
                        } else {
                            self.shared_state
                                .metrics
                                .channel_metrics
                                .network_out
                                .record_sent();
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
                // SEC: Only serve shards to peers in our peer_registry (authenticated + known).
                // Without this, any node that completes a Noise handshake can exfiltrate shards.
                let peer_node_id = self.peer_to_node.get(&peer).map(|r| r.clone());
                if let Some(ref nid) = peer_node_id {
                    if !self.shared_state.peer_registry.contains_key(nid) {
                        tracing::warn!(%peer, "Shard transfer from unknown peer — rejecting");
                        let _ = self.swarm.behaviour_mut().request_response.send_response(
                            channel,
                            SwarmResponse::ShardData(crate::types::ShardResponse {
                                data: vec![],
                                total_size: 0,
                            }),
                        );
                        return;
                    }
                } else {
                    tracing::warn!(%peer, "Shard transfer from unmapped peer — rejecting");
                    let _ = self.swarm.behaviour_mut().request_response.send_response(
                        channel,
                        SwarmResponse::ShardData(crate::types::ShardResponse {
                            data: vec![],
                            total_size: 0,
                        }),
                    );
                    return;
                }

                tracing::info!(
                    %peer,
                    model = %shard_req.shard_id.model_id,
                    index = shard_req.shard_id.index,
                    offset = shard_req.chunk_offset,
                    chunk_size = shard_req.chunk_size,
                    "Shard transfer request"
                );

                // Extract path info from self (sync), then do blocking I/O
                // via spawn_blocking without holding &self across the await.
                let prepared = self.prepare_shard_read(&shard_req);
                let bw_limit = self.shared_state.config.resources.max_bandwidth_mbps;
                let response = match prepared {
                    Some((path, offset, chunk_size, model_id, shard_index)) => {
                        let resp =
                            read_shard_chunk_async(path, offset, chunk_size, model_id, shard_index)
                                .await;
                        // Enforce upload bandwidth cap: delay proportional to chunk size.
                        // 0 = unlimited (default). Only throttles shard serving, not tensor forwards.
                        if bw_limit > 0 {
                            if let SwarmResponse::ShardData(ref sr) = resp {
                                if !sr.data.is_empty() {
                                    let bytes = sr.data.len() as u64;
                                    let limit_bytes_per_sec = bw_limit * 125_000; // Mbps → bytes/s
                                    let delay_ms = (bytes * 1000) / limit_bytes_per_sec;
                                    if delay_ms > 0 {
                                        tokio::time::sleep(std::time::Duration::from_millis(
                                            delay_ms,
                                        ))
                                        .await;
                                    }
                                }
                            }
                        }
                        resp
                    }
                    None => SwarmResponse::ShardData(crate::types::ShardResponse {
                        data: vec![],
                        total_size: 0,
                    }),
                };

                // Track bytes served for seeding credits
                let bytes_served = match &response {
                    SwarmResponse::ShardData(ref sr) => sr.data.len() as u64,
                    _ => 0,
                };

                // NET-M7: Log send_response errors
                if self
                    .swarm
                    .behaviour_mut()
                    .request_response
                    .send_response(channel, response)
                    .is_ok()
                {
                    // Only credit bytes if the response was actually sent
                    if bytes_served > 0 {
                        self.shared_state
                            .shard_bytes_served
                            .fetch_add(bytes_served, std::sync::atomic::Ordering::Relaxed);
                    }
                } else {
                    tracing::debug!(%peer, "Failed to send shard data response (channel closed)");
                }
            }
            SwarmRequest::TensorPayload(payload) => {
                // SEC: Only accept tensor payloads from authenticated peers in peer_registry.
                // Without this, any node completing a Noise handshake can inject activations
                // into in-flight inference pipelines, corrupting output silently.
                let peer_node_id = self.peer_to_node.get(&peer).map(|r| r.clone());
                let is_authenticated = match &peer_node_id {
                    Some(nid) => self.shared_state.peer_registry.contains_key(nid),
                    None => false,
                };
                if !is_authenticated {
                    tracing::warn!(%peer, "TensorPayload from unauthenticated peer — rejecting");
                    let _ = self
                        .swarm
                        .behaviour_mut()
                        .request_response
                        .send_response(channel, SwarmResponse::Ack);
                    return;
                }

                tracing::info!(
                    %peer,
                    payload_len = payload.len(),
                    "DIAG: inbound TensorPayload request"
                );
                if let Some(request_id) = self.handle_tensor_payload(peer, &payload) {
                    // Store the ResponseChannel so we can send the computed result as
                    // the actual response (single substream per token, no separate request).
                    if self.pending_tensor_channels.len() >= MAX_PENDING_TENSOR_CHANNELS {
                        tracing::warn!(%peer, "pending_tensor_channels full — rejecting with ACK");
                        // Send ACK to avoid leaving requester hung, then skip storing
                        let _ = self
                            .swarm
                            .behaviour_mut()
                            .request_response
                            .send_response(channel, SwarmResponse::Ack);
                    } else {
                        self.pending_tensor_channels
                            .insert(request_id, (std::time::Instant::now(), channel));
                        tracing::info!(
                            %peer,
                            %request_id,
                            pending_channels = self.pending_tensor_channels.len(),
                            "DIAG: stored ResponseChannel for tensor forward"
                        );
                    }
                } else {
                    // LayerResult or decode failure — just ACK since there's no round-trip
                    if self
                        .swarm
                        .behaviour_mut()
                        .request_response
                        .send_response(channel, SwarmResponse::Ack)
                        .is_err()
                    {
                        tracing::debug!(%peer, "Failed to send tensor ACK (channel closed)");
                    }
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
        self.refresh_peer_last_seen(&peer);
        match response {
            SwarmResponse::Message(msg) => {
                // Handle PEX response inline
                if let SwarmMessage::PeerExchangeResponse(ref pex_resp) = *msg {
                    // Measure RTT from rr_ping send time
                    if let Some((_sent_peer, sent_at)) = self.ping_sent_times.remove(&request_id) {
                        let rtt_ms = sent_at.elapsed().as_millis() as u32;
                        if let Some(node_id) = self.peer_to_node.get(&peer) {
                            if let Some(mut peer_info) =
                                self.shared_state.peer_registry.get_mut(&*node_id)
                            {
                                peer_info.latency_ms = Some(rtt_ms);
                                // Auto-detect LAN peer from low latency (< 5ms)
                                if rtt_ms < 5 && !peer_info.is_lan_peer {
                                    peer_info.is_lan_peer = true;
                                    drop(peer_info);
                                    let count = self
                                        .shared_state
                                        .lan_peer_count
                                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
                                        + 1;
                                    let msg = format!(
                                        "Found {} peer{} on your local network",
                                        count,
                                        if count == 1 { "" } else { "s" }
                                    );
                                    tracing::info!(%peer, rtt_ms, lan_peers = count, message = %msg, "LAN peer discovery update");
                                    self.shared_state.emit_activity(
                                        crate::daemon::state::ActivityEvent::new(
                                            "network",
                                            "lan_peer_discovered",
                                            msg,
                                        )
                                        .with_detail_num(count as i64)
                                        .with_toast("success", 8000),
                                    );
                                } else {
                                    tracing::debug!(%peer, rtt_ms, "Peer RTT measured");
                                }
                            }
                        }
                    }
                    // SEC: Only process PEX from registered peers
                    let is_registered = self
                        .peer_to_node
                        .get(&peer)
                        .is_some_and(|n| self.shared_state.peer_registry.contains_key(&*n));
                    if is_registered {
                        tracing::debug!(%peer, count = pex_resp.peers.len(), "Received PEX response");
                        self.handle_pex_response(&pex_resp.peers);
                    } else {
                        tracing::debug!(%peer, "Ignoring PEX response from unregistered peer");
                    }
                    return;
                }
                if let Err(e) = self.dispatch_authenticated(Some(&peer), *msg) {
                    self.shared_state
                        .metrics
                        .channel_metrics
                        .network_out
                        .record_dropped();
                    tracing::warn!(%peer, error = %e, "Dispatcher backpressured, dropping response message");
                } else {
                    self.shared_state
                        .metrics
                        .channel_metrics
                        .network_out
                        .record_sent();
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
                    // Cap total_size to prevent unbounded download loops from malicious peers
                    let max_shard_bytes =
                        self.shared_state.config.model.shard_size_mb * 1024 * 1024 * 2;
                    if data.total_size > max_shard_bytes {
                        tracing::warn!(
                            %peer,
                            total_size = data.total_size,
                            max = max_shard_bytes,
                            "Rejecting shard download — total_size exceeds limit"
                        );
                        self.shard_download_progress.remove(&shard_id);
                        return;
                    }
                    let offset = self
                        .shard_download_progress
                        .get(&shard_id)
                        .copied()
                        .unwrap_or(0);
                    let chunk_len = data.data.len() as u64;

                    // Empty response = peer doesn't have the shard — retry with next peer or HF
                    if data.total_size == 0 || data.data.is_empty() {
                        tracing::warn!(
                            %peer,
                            model = %shard_id.model_id,
                            shard = shard_id.index,
                            "Peer returned empty shard data — trying next peer or HF fallback"
                        );
                        self.shard_download_progress.remove(&shard_id);

                        // Track retry count — max 5 P2P attempts before HF fallback
                        const MAX_P2P_RETRIES: u32 = 5;
                        let retries = self.shard_p2p_retries.entry(shard_id.clone()).or_insert(0);
                        *retries += 1;
                        let retry_num = *retries;

                        if retry_num > MAX_P2P_RETRIES {
                            self.shard_p2p_retries.remove(&shard_id);
                            // Fall through to HF fallback below
                        } else {
                            // Try next peer that holds this shard (excluding the failed one)
                            let local_nid = self.shared_state.identity.node_id().clone();
                            let failed_peer_nid = self.peer_to_node.get(&peer).map(|r| r.clone());
                            let other_holders: Vec<_> = self
                                .shared_state
                                .model_registry
                                .shard_holders(&shard_id)
                                .into_iter()
                                .filter(|n| {
                                    if *n == local_nid {
                                        return false;
                                    }
                                    match &failed_peer_nid {
                                        Some(fp) => n != fp,
                                        None => true,
                                    }
                                })
                                .collect();

                            if !other_holders.is_empty() {
                                // Retry with best remaining peer
                                let next_target =
                                    self.shared_state.select_best_peer(&other_holders);

                                if let Some(next_bytes) = self
                                    .shared_state
                                    .peer_registry
                                    .get(&next_target)
                                    .and_then(|p| p.peer_id_bytes.clone())
                                {
                                    let mname = self
                                        .shared_state
                                        .model_registry
                                        .get_manifest(&shard_id.model_id)
                                        .map(|m| m.name.clone());
                                    self.shared_state.emit_activity(
                                        crate::daemon::state::ActivityEvent::new(
                                            "download",
                                            "shard_download_p2p",
                                            format!(
                                                "Retrying shard {} of {} from another peer",
                                                crate::types::ShardId::display_index_short(
                                                    shard_id.index
                                                ),
                                                mname.as_deref().unwrap_or(&shard_id.model_id.0),
                                            ),
                                        )
                                        .with_model(shard_id.model_id.0.clone())
                                        .with_node(format!("{}", next_target))
                                        .with_detail_num(shard_id.index as i64)
                                        .with_detail_str("retry".to_string()),
                                    );
                                    let retry_req = crate::types::ShardRequest {
                                        shard_id: shard_id.clone(),
                                        chunk_offset: 0,
                                        chunk_size: crate::network::protocol::SHARD_CHUNK_SIZE,
                                    };
                                    self.handle_send_shard_request(next_bytes, retry_req);
                                    return;
                                }
                            }
                        } // end retry_num <= MAX_P2P_RETRIES

                        // Retries exhausted or no more peers — fall back to HuggingFace
                        self.shard_p2p_retries.remove(&shard_id);
                        if let Some(hf_src) =
                            self.shared_state.models.hf_sources.get(&shard_id.model_id)
                        {
                            let mname = self
                                .shared_state
                                .model_registry
                                .get_manifest(&shard_id.model_id)
                                .map(|m| m.name.clone());
                            self.shared_state.emit_activity(
                                crate::daemon::state::ActivityEvent::new(
                                    "download",
                                    "shard_download_started",
                                    format!(
"P2P failed after {} retries — falling back to HuggingFace for shard {} of {}",
retry_num,
crate::types::ShardId::display_index_short(shard_id.index),
mname.as_deref().unwrap_or(&shard_id.model_id.0),
),
                                )
                                .with_model(shard_id.model_id.0.clone())
                                .with_detail_num(shard_id.index as i64)
                                .with_detail_str("hf_fallback".to_string()),
                            );
                            // Wake auto-manage to pick up HF download
                            self.shared_state.models.auto_manage_notify.notify_one();
                            drop(hf_src);
                        } else {
                            let mname = self
                                .shared_state
                                .model_registry
                                .get_manifest(&shard_id.model_id)
                                .map(|m| m.name.clone());
                            self.shared_state.emit_activity(
                                crate::daemon::state::ActivityEvent::new(
                                    "download",
                                    "shard_transfer_failed",
                                    format!(
                                        "No peers or HF source for shard {} of {}",
                                        crate::types::ShardId::display_index_short(shard_id.index),
                                        mname.as_deref().unwrap_or(&shard_id.model_id.0),
                                    ),
                                )
                                .with_model(shard_id.model_id.0.clone())
                                .with_detail_num(shard_id.index as i64)
                                .with_detail_str("no_source".to_string())
                                .with_toast("warning", 6000),
                            );
                        }

                        // Clean up acquisition progress
                        if let Some(mut entry) = self
                            .shared_state
                            .models
                            .acquisition_progress
                            .get_mut(&shard_id.model_id)
                        {
                            entry.state = crate::model::acquisition::AcquisitionState::Failed {
                                reason: "P2P transfer failed".into(),
                            };
                        }
                        self.shared_state
                            .schedule_acquisition_cleanup(shard_id.model_id.clone());
                        return;
                    }

                    // Forward to AcquisitionManager if it has a job for this model
                    // (user-initiated downloads via the "Download" button).
                    // Auto-manage P2P downloads don't have jobs, so we also write directly.
                    if let Some(ref acq_tx) = self.acquisition_tx {
                        let _ = acq_tx.try_send(AcquisitionCommand::ShardDataReceived {
                            shard_id: shard_id.clone(),
                            offset,
                            data: data.data.clone(),
                            total_size: data.total_size,
                        });
                    }

                    // Also write directly (handles auto-manage P2P downloads that
                    // bypass the AcquisitionManager job registry).
                    if let Err(e) = self.shard_store.write_chunk(
                        &shard_id.model_id,
                        shard_id.index,
                        offset,
                        &data.data,
                    ) {
                        tracing::error!(
                            model = %shard_id.model_id,
                            shard = shard_id.index,
                            error = %e,
                            "Failed to write P2P shard chunk to disk"
                        );
                        self.shared_state.emit_activity(
                            crate::daemon::state::ActivityEvent::new(
                                "download",
                                "shard_write_failed",
                                format!(
                                    "Failed to write shard {} of {} to disk: {}",
                                    crate::types::ShardId::display_index_short(shard_id.index),
                                    shard_id.model_id,
                                    e
                                ),
                            )
                            .with_model(shard_id.model_id.0.clone())
                            .with_detail_num(shard_id.index as i64)
                            .with_toast("error", 6000),
                        );
                    }

                    // Update acquisition_progress so the frontend download bar moves
                    if let Some(mut entry) = self
                        .shared_state
                        .models
                        .acquisition_progress
                        .get_mut(&shard_id.model_id)
                    {
                        entry.downloaded_bytes = entry.downloaded_bytes.saturating_add(chunk_len);
                        if let Some(sp) = entry.shard_progress.get_mut(&shard_id.index) {
                            sp.downloaded_bytes = sp.downloaded_bytes.saturating_add(chunk_len);
                        }
                        // Estimate speed from chunk timing
                        if chunk_len > 0 {
                            entry.speed_bytes_per_sec = chunk_len; // rough per-chunk estimate
                        }
                    }

                    // Update progress tracking
                    let new_offset = offset + chunk_len;
                    if new_offset < data.total_size {
                        // More chunks needed — re-register and request next chunk
                        // SEC: enforce cap even for continuation requests to prevent unbounded growth
                        if self.pending_shard_requests.len() >= MAX_PENDING_SHARD_REQUESTS {
                            tracing::warn!(
                                %peer,
                                model = %shard_id.model_id,
                                index = shard_id.index,
                                "Shard download continuation dropped — pending_shard_requests at cap"
                            );
                            self.shard_download_progress.remove(&shard_id);
                            self.shard_p2p_retries.remove(&shard_id);
                        } else {
                            self.shard_download_progress
                                .insert(shard_id.clone(), new_offset);

                            let next_req = crate::types::ShardRequest {
                                shard_id: shard_id.clone(),
                                chunk_offset: new_offset,
                                chunk_size: crate::network::protocol::SHARD_CHUNK_SIZE, // 32MB chunks
                            };
                            let req = SwarmRequest::ShardTransfer(next_req);
                            let new_req_id = self
                                .swarm
                                .behaviour_mut()
                                .request_response
                                .send_request(&peer, req);
                            self.pending_shard_requests
                                .insert(new_req_id, (peer, shard_id));
                        }
                    } else {
                        // Download complete for this shard
                        self.shard_download_progress.remove(&shard_id);
                        self.shard_p2p_retries.remove(&shard_id);

                        // Finalize: rename .tmp → .bin atomically
                        if let Err(e) = self
                            .shard_store
                            .finalize_shard(&shard_id.model_id, shard_id.index)
                        {
                            tracing::error!(
                                model = %shard_id.model_id,
                                shard = shard_id.index,
                                error = %e,
                                "Failed to finalize P2P shard (.tmp → .bin rename)"
                            );
                            self.shared_state.emit_activity(
                                crate::daemon::state::ActivityEvent::new(
                                    "download",
                                    "shard_finalize_failed",
                                    format!(
                                        "Failed to finalize shard {} of {}: {}",
                                        crate::types::ShardId::display_index_short(shard_id.index),
                                        shard_id.model_id,
                                        e
                                    ),
                                )
                                .with_model(shard_id.model_id.0.clone())
                                .with_detail_num(shard_id.index as i64)
                                .with_toast("error", 6000),
                            );
                        }

                        // Mark acquisition as complete so frontend clears the download bar
                        if let Some(mut entry) = self
                            .shared_state
                            .models
                            .acquisition_progress
                            .get_mut(&shard_id.model_id)
                        {
                            entry.downloaded_shards = entry.downloaded_shards.saturating_add(1);
                            entry.downloaded_bytes = entry.total_bytes;
                            entry.state = crate::model::acquisition::AcquisitionState::Complete;
                            if let Some(sp) = entry.shard_progress.get_mut(&shard_id.index) {
                                sp.state = crate::model::acquisition::ShardState::Complete;
                                sp.downloaded_bytes = sp.total_bytes;
                            }
                        }
                        // Remove the acquisition entry after a delay so UI sees the completion
                        self.shared_state
                            .schedule_acquisition_cleanup(shard_id.model_id.clone());

                        // Register ourselves as a holder of this shard
                        let local_node_id = self.shared_state.identity.node_id().clone();
                        self.shared_state
                            .model_registry
                            .record_shard_holder(shard_id.clone(), local_node_id.clone());

                        // Queue shard announce — can't call handle_broadcast inline
                        // because we're inside a swarm event handler (causes re-entrant panic).
                        // The announce will be sent on the next event loop iteration.
                        self.deferred_broadcasts
                            .push(crate::types::SwarmMessage::ShardAnnounce(
                                crate::types::ShardAnnounce {
                                    node_id: local_node_id,
                                    shards: vec![shard_id.clone()],
                                    timestamp: chrono::Utc::now(),
                                },
                            ));

                        // Load the model with the new shard (spawned async — can't block event loop)
                        {
                            let load_shared = self.shared_state.clone();
                            let load_mid = shard_id.model_id.clone();
                            tokio::spawn(async move {
                                let vram_budget =
                                    crate::model::auto_manage::vram::compute_vram_budget(
                                        &load_shared,
                                    );
                                // check_and_load_model emits model_loaded activity event
                                crate::model::auto_manage::scan::check_and_load_model(
                                    &load_shared,
                                    &load_mid,
                                    vram_budget,
                                )
                                .await;
                                load_shared.signal_dashboard(
                                    crate::daemon::state::DashboardSignal::ModelsChanged,
                                );
                            });
                        }
                        self.shared_state.models.auto_manage_notify.notify_one();

                        tracing::info!(
                            model = %shard_id.model_id,
                            index = shard_id.index,
                            "P2P shard download complete — registered and announced"
                        );
                        let mname = self
                            .shared_state
                            .model_registry
                            .get_manifest(&shard_id.model_id)
                            .map(|m| m.name.clone());
                        let peer_node_id = self.peer_to_node.get(&peer).map(|r| r.clone());
                        let peer_label = peer_node_id
                            .as_ref()
                            .and_then(|nid| {
                                self.shared_state
                                    .nickname_registry
                                    .get(nid)
                                    .map(|r| r.nickname.clone())
                            })
                            .or_else(|| {
                                peer_node_id
                                    .as_ref()
                                    .map(|nid| format!("{}", nid).chars().take(8).collect())
                            })
                            .unwrap_or_else(|| format!("{}", peer).chars().take(12).collect());
                        self.shared_state.emit_activity(
                            crate::daemon::state::ActivityEvent::new(
                                "download",
                                "shard_p2p_complete",
                                format!(
                                    "Shard {} of {} downloaded from peer {}",
                                    crate::types::ShardId::display_index_short(shard_id.index),
                                    mname.as_deref().unwrap_or(&shard_id.model_id.0),
                                    peer_label
                                ),
                            )
                            .with_model(shard_id.model_id.0.clone())
                            .with_node(format!("{}", peer))
                            .with_detail_num(shard_id.index as i64),
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
                tracing::info!(
                    %peer,
                    payload_len = payload.len(),
                    "DIAG: received TensorPayload response"
                );
                let _ = self.handle_tensor_payload(peer, &payload);
            }
            SwarmResponse::Ack => {
                tracing::debug!(%peer, "Received ACK");
            }
        }
    }

    /// Handle outbound commands from daemon tasks.
    async fn handle_outbound_command(&mut self, cmd: NetworkCommand) {
        let cmd_name = match &cmd {
            NetworkCommand::Broadcast(_) => "Broadcast",
            NetworkCommand::SendTensor { .. } => "SendTensor",
            NetworkCommand::SendTensorResult { .. } => "SendTensorResult",
            NetworkCommand::SendStreamingToken { .. } => "SendStreamingToken",
            NetworkCommand::SendShardRequest { .. } => "SendShardRequest",
            NetworkCommand::SendAllReduceRequest { .. } => "SendAllReduceRequest",
            NetworkCommand::SendAllReduceResponse { .. } => "SendAllReduceResponse",
            NetworkCommand::SendRingChunk { .. } => "SendRingChunk",
            NetworkCommand::SendDirectMessage { .. } => "SendDirectMessage",
            NetworkCommand::DialAddress(_) => "DialAddress",
            NetworkCommand::StartProviding(_) => "StartProviding",
            NetworkCommand::StopProviding(_) => "StopProviding",
        };
        tracing::debug!(cmd = cmd_name, "DIAG: handling outbound command");
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
            NetworkCommand::SendAllReduceRequest {
                target_peer_bytes,
                request,
            } => {
                self.handle_send_rr_message(
                    target_peer_bytes,
                    SwarmMessage::TpAllReduceRequest(request),
                    "AllReduceRequest",
                );
            }
            NetworkCommand::SendAllReduceResponse {
                target_peer_bytes,
                response,
            } => {
                self.handle_send_rr_message(
                    target_peer_bytes,
                    SwarmMessage::TpAllReduceResponse(response),
                    "AllReduceResponse",
                );
            }
            NetworkCommand::SendRingChunk {
                target_peer_bytes,
                chunk,
            } => {
                self.handle_send_rr_message(
                    target_peer_bytes,
                    SwarmMessage::TpRingChunk(chunk),
                    "RingChunk",
                );
            }
            NetworkCommand::SendDirectMessage {
                target_peer_bytes,
                message,
            } => {
                self.handle_send_rr_message(target_peer_bytes, message, "DirectMessage");
            }
            NetworkCommand::DialAddress(addr_str) => {
                match addr_str.parse::<libp2p::Multiaddr>() {
                    Ok(addr) => {
                        tracing::info!(%addr, "Dialing peer from invite code");
                        // Dial the original address
                        if let Err(e) = self.swarm.dial(addr.clone()) {
                            tracing::warn!(%addr, error = %e, "Failed to dial invite peer");
                        }
                        // Also try localhost variant for same-machine peers
                        let addr_s = addr.to_string();
                        if !addr_s.contains("/ip4/127.0.0.1/") {
                            let localhost_addr = addr_s
                                .split("/ip4/")
                                .enumerate()
                                .map(|(i, part)| {
                                    if i == 1 {
                                        // Replace the IP portion
                                        let rest = part.split_once('/').map(|x| x.1).unwrap_or("");
                                        format!("127.0.0.1/{rest}")
                                    } else {
                                        part.to_string()
                                    }
                                })
                                .collect::<Vec<_>>()
                                .join("/ip4/");
                            if let Ok(lo_addr) = localhost_addr.parse::<libp2p::Multiaddr>() {
                                tracing::debug!(%lo_addr, "Also trying localhost variant");
                                let _ = self.swarm.dial(lo_addr);
                            }
                        }
                    }
                    Err(e) => {
                        tracing::warn!(addr = %addr_str, error = %e, "Invalid multiaddr in DialAddress");
                    }
                }
            }
            NetworkCommand::StartProviding(shards) => {
                let _ = crate::network::discovery::start_providing_shards(&mut self.swarm, &shards);
            }
            NetworkCommand::StopProviding(shards) => {
                crate::network::discovery::stop_providing_shards(&mut self.swarm, &shards);
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
            SwarmMessage::ModelVote(_) => TOPIC_MODELS, // wire-compat: retained for older peers, payload discarded on receive
            SwarmMessage::HealthPing { .. } | SwarmMessage::HealthPong { .. } => {
                crate::network::protocol::TOPIC_HEALTH
            }
            SwarmMessage::NicknameGossip(_) => crate::network::protocol::TOPIC_IDENTITY,
            SwarmMessage::PoolMessage(_) => crate::network::protocol::TOPIC_POOLS,
            SwarmMessage::RegionShardSummary(_) | SwarmMessage::ModelDemandGossip(_) => {
                crate::network::protocol::TOPIC_REGIONS
            }
            // AllReduce responses broadcast to TP group via gossip (small group, LAN-local)
            SwarmMessage::TpAllReduceResponse(_) => crate::network::protocol::TOPIC_HEALTH,
            // Inference and credit transaction messages go via request_response, not gossipsub
            _ => return,
        };

        match protocol::encode_message(&msg) {
            Ok(data) => {
                // SEC: Sign and seal gossip with Ed25519 identity + epoch PSK.
                // Unsigned gossip allows any peer to forge messages (shard announcements,
                // model manifests, health pings, credit gossip) under arbitrary NodeIds.
                let publish_data = match self
                    .shared_state
                    .gossip_sealer
                    .seal_signed(&data, &self.shared_state.identity)
                {
                    Ok(sealed) => sealed,
                    Err(e) => {
                        tracing::warn!(error = %e, "Failed to seal gossip message, dropping");
                        return;
                    }
                };

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
                        if self.buffered_gossip.len() < MAX_BUFFERED_GOSSIP {
                            tracing::debug!(topic, error = %e, "Failed to publish to GossipSub, buffering");
                            self.buffered_gossip.push((topic.to_string(), publish_data));
                        } else {
                            tracing::warn!(
                                topic,
                                cap = MAX_BUFFERED_GOSSIP,
                                "Gossip buffer full, dropping message"
                            );
                        }
                    }
                }
            }
            Err(e) => {
                tracing::warn!(error = %e, "Failed to encode outbound message");
            }
        }
    }

    /// Send a tensor forward to a specific peer via the unified binary tensor protocol.
    /// Uses WIRE_TAG_TENSOR (0x01) framing. Encrypts activations when an encryption
    /// session exists, falls back to plaintext.
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
        let peer_node_id = self.peer_to_node_id(&peer_id);
        let use_encryption =
            self.shared_state.config.network.enable_encryption && peer_node_id.is_some();

        let payload = if use_encryption {
            let node_id = match peer_node_id {
                Some(n) => n,
                None => {
                    tracing::warn!(%peer_id, "Encryption enabled but no NodeId for peer");
                    return;
                }
            };
            // Build the AAD from the cleartext header fields — must match
            // decode_layer_forward_encrypted's AAD (uuid + seq + idx_pos + fmt + layer_range + model_id)
            let model_id_bytes = forward.model_id.0.as_bytes();
            let mut aad = Vec::with_capacity(35 + model_id_bytes.len());
            aad.extend_from_slice(forward.request_id.as_bytes());
            aad.extend_from_slice(&forward.sequence_num.to_le_bytes());
            aad.extend_from_slice(&forward.index_pos.to_le_bytes());
            let fmt_tag: u8 = match forward.format {
                crate::types::TensorFormat::FP16 => 0,
                crate::types::TensorFormat::FP32 => 1,
                crate::types::TensorFormat::INT8 => 2,
            };
            aad.push(fmt_tag);
            let (layer_start, layer_end) = forward.layer_range;
            aad.extend_from_slice(&layer_start.to_le_bytes());
            aad.extend_from_slice(&layer_end.to_le_bytes());
            aad.extend_from_slice(&(model_id_bytes.len() as u16).to_le_bytes());
            aad.extend_from_slice(model_id_bytes);

            tracing::debug!(
                request_id = %forward.request_id,
                %peer_id,
                node_id = %node_id,
                aad_len = aad.len(),
                activation_len = forward.activations.len(),
                has_session = self.shared_state.session_manager.has_session(&node_id),
                session_count = self.shared_state.session_manager.session_count(),
                "DIAG: encrypting tensor forward"
            );

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
                    // SEC: Never fall back to plaintext — fail the forward instead.
                    // Plaintext fallback would silently strip encryption, allowing
                    // eavesdroppers to read intermediate tensor activations.
                    tracing::warn!(
                        error = %e,
                        request_id = %forward.request_id,
                        %peer_id,
                        node_id = %node_id,
                        aad_len = aad.len(),
                        has_session = self.shared_state.session_manager.has_session(&node_id),
                        "DIAG: seal() failed — dropping forward (no plaintext fallback)"
                    );
                    // Notify the pipeline immediately so it fails fast
                    // instead of waiting for the AllReduce timeout.
                    let request_id = forward.request_id;
                    if let Some((_, channel)) = self.pending_tensor_channels.remove(&request_id) {
                        let _ = self
                            .swarm
                            .behaviour_mut()
                            .request_response
                            .send_response(channel, SwarmResponse::Ack);
                    }
                    let error_result = crate::types::LayerResult {
                        request_id,
                        token_ids: vec![],
                        finish_reason: Some(crate::types::NetworkFinishReason::Error(
                            "Encryption session lost — reconnecting".into(),
                        )),
                        activations: vec![],
                        sealed_token_ids: None,
                    };
                    if let Some((_, tx)) =
                        self.shared_state.pending_layer_results.remove(&request_id)
                    {
                        let _ = tx.send(error_result);
                    }
                    return;
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
        let is_connected = self.swarm.is_connected(&peer_id);
        // DIAG: Check if request_response behaviour thinks this peer is connected
        let rr_is_connected = self
            .swarm
            .behaviour()
            .request_response
            .is_connected(&peer_id);
        // Count total established connections (all peers) for diagnostics
        let total_conn_count = self
            .swarm
            .network_info()
            .connection_counters()
            .num_established();
        tracing::info!(
            %peer_id,
            request_id = %forward.request_id,
            rr_is_connected,
            swarm_is_connected = is_connected,
            "DIAG: PRE-send_request state"
        );
        // CRITICAL: If the swarm says the peer is not connected, fail immediately.
        // The rr behaviour may have a stale connection entry (rr_connected=true)
        // after a disconnect, causing send_request() to target a dead ConnectionId
        // whose NotifyHandler is silently dropped by the swarm pool.
        if !is_connected {
            tracing::warn!(
                %peer_id,
                request_id = %forward.request_id,
                rr_is_connected,
                "Peer not connected — failing tensor forward immediately"
            );
            self.fail_tensor_forward(forward.request_id, &peer_id, "Peer not connected".into());
            return;
        }
        let req = SwarmRequest::TensorPayload(payload);
        let outbound_id = self
            .swarm
            .behaviour_mut()
            .request_response
            .send_request(&peer_id, req);
        // Track OutboundRequestId → (UUID, time, peer, layers, activation_size)
        // so we can notify pipeline on OutboundFailure and compute adaptive stale timeouts.
        let num_layers = forward.layer_range.1.saturating_sub(forward.layer_range.0);
        let activation_bytes = forward.activations.len();
        self.pending_tensor_outbound.insert(
            outbound_id,
            (
                forward.request_id,
                std::time::Instant::now(),
                peer_id,
                num_layers,
                activation_bytes,
            ),
        );
        // DIAG: check is_pending_outbound immediately — confirms the request was registered
        let is_rr_pending = self
            .swarm
            .behaviour()
            .request_response
            .is_pending_outbound(&peer_id, &outbound_id);
        // DIAG: enumerate connection IDs for this peer to detect stale conn_id issues
        let peer_established_count = self
            .swarm
            .connected_peers()
            .filter(|p| **p == peer_id)
            .count();
        let all_conn_ids: Vec<_> = self
            .connection_addrs
            .iter()
            .map(|(cid, addr)| format!("{cid:?}→{addr}"))
            .collect();
        tracing::info!(
            %peer_id,
            request_id = %forward.request_id,
            seq = forward.sequence_num,
            encrypted = use_encryption,
            payload_len,
            is_connected,
            total_connections = total_conn_count,
            peer_established_count,
            is_rr_pending,
            pending_tensor_count = self.pending_tensor_outbound.len(),
            ?outbound_id,
            tracked_connections = ?all_conn_ids,
            "DIAG: sent tensor forward via send_request"
        );
    }

    /// Send a tensor result back to the requesting peer.
    ///
    /// Prefers sending the result as a **response** on the original forward's
    /// ResponseChannel (stored in `pending_tensor_channels`). This keeps the
    /// entire forward→result exchange on a single QUIC substream, halving
    /// substream usage and preventing the stall that occurred when results
    /// were sent as separate requests.
    ///
    /// Falls back to a new request if no stored channel is found (e.g. timeout
    /// or the forward arrived via gossip).
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

        let is_connected = self.swarm.is_connected(&peer_id);
        if !is_connected {
            tracing::error!(
                %peer_id,
                request_id = %result.request_id,
                "DIAG: cannot send tensor result — peer NOT connected"
            );
        }

        // Try to send as response on the original forward's channel
        if let Some((_inserted, channel)) = self.pending_tensor_channels.remove(&result.request_id)
        {
            match protocol::encode_layer_result(&result) {
                Ok(payload) => {
                    let payload_len = payload.len();
                    let resp = SwarmResponse::TensorPayload(payload);
                    if self
                        .swarm
                        .behaviour_mut()
                        .request_response
                        .send_response(channel, resp)
                        .is_err()
                    {
                        tracing::warn!(
                            %peer_id,
                            request_id = %result.request_id,
                            "ResponseChannel closed, falling back to new request"
                        );
                        // Channel was closed (timeout/disconnect) — fall back to new request
                        self.send_tensor_result_as_request(&peer_id, &result, is_connected);
                    } else {
                        tracing::info!(
                            %peer_id,
                            request_id = %result.request_id,
                            payload_len,
                            is_connected,
                            tokens = result.token_ids.len(),
                            activations_bytes = result.activations.len(),
                            finish = ?result.finish_reason,
                            pending_channels = self.pending_tensor_channels.len(),
                            "DIAG: sent tensor result as response (same substream)"
                        );
                    }
                }
                Err(e) => {
                    tracing::warn!(error = %e, request_id = %result.request_id, "Failed to encode tensor result");
                }
            }
        } else {
            // No stored ResponseChannel — peer may have reconnected or sent a result for a different substream
            tracing::debug!(
                %peer_id,
                request_id = %result.request_id,
                "No stored ResponseChannel, sending result as new request"
            );
            self.send_tensor_result_as_request(&peer_id, &result, is_connected);
        }
    }

    /// Timeout recovery: send tensor result as a new outbound request when the
    /// original response channel was closed (peer disconnect or timeout).
    fn send_tensor_result_as_request(
        &mut self,
        peer_id: &libp2p::PeerId,
        result: &crate::types::LayerResult,
        is_connected: bool,
    ) {
        match protocol::encode_layer_result(result) {
            Ok(payload) => {
                let payload_len = payload.len();
                let req = SwarmRequest::TensorPayload(payload);
                let outbound_id = self
                    .swarm
                    .behaviour_mut()
                    .request_response
                    .send_request(peer_id, req);
                tracing::info!(
                    %peer_id,
                    request_id = %result.request_id,
                    payload_len,
                    is_connected,
                    tokens = result.token_ids.len(),
                    activations_bytes = result.activations.len(),
                    finish = ?result.finish_reason,
                    ?outbound_id,
                    "DIAG: sent tensor result as new request (fallback)"
                );
            }
            Err(e) => {
                tracing::warn!(error = %e, request_id = %result.request_id, "Failed to encode tensor result (fallback)");
            }
        }
    }

    /// Process an inbound binary tensor payload (from either request or response).
    /// Returns `Some(request_id)` if this was a LayerForward (so the caller can
    /// save the ResponseChannel for sending the result back as a response).
    fn handle_tensor_payload(
        &mut self,
        peer: libp2p::PeerId,
        payload: &[u8],
    ) -> Option<uuid::Uuid> {
        let tag = payload.first().copied().unwrap_or(0);
        tracing::debug!(%peer, tag, payload_len = payload.len(), "handle_tensor_payload");
        match tag {
            protocol::TENSOR_TAG_FORWARD => {
                tracing::info!(%peer, payload_len = payload.len(), "Received tensor LayerForward");
                match protocol::decode_layer_forward(payload) {
                    Ok(mut forward) => {
                        let request_id = forward.request_id;
                        let is_tp = forward.tp_meta.is_some();
                        forward.sender_peer_bytes = Some(peer.to_bytes());
                        let msg = SwarmMessage::LayerForward(forward);
                        if let Err(e) = self.dispatch_authenticated(Some(&peer), msg) {
                            self.shared_state
                                .metrics
                                .channel_metrics
                                .network_out
                                .record_dropped();
                            tracing::warn!(error = %e, "Outbound channel full, dropping tensor forward");
                            return None;
                        } else {
                            self.shared_state
                                .metrics
                                .channel_metrics
                                .network_out
                                .record_sent();
                        }
                        // TP forwards send their response via separate TpAllReduceRequest,
                        // NOT through the original request_response channel. Return None
                        // so the caller ACKs immediately instead of storing the channel.
                        if is_tp {
                            return None;
                        }
                        Some(request_id)
                    }
                    Err(e) => {
                        tracing::warn!(error = %e, "Failed to decode tensor forward");
                        None
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
                        if let Err(e) = self
                            .dispatch_authenticated(Some(&peer), SwarmMessage::LayerResult(result))
                        {
                            self.shared_state
                                .metrics
                                .channel_metrics
                                .network_out
                                .record_dropped();
                            tracing::warn!(error = %e, "Outbound channel full, dropping tensor result");
                        } else {
                            self.shared_state
                                .metrics
                                .channel_metrics
                                .network_out
                                .record_sent();
                        }
                    }
                    Err(e) => {
                        tracing::warn!(error = %e, "Failed to decode tensor result");
                    }
                }
                None
            }
            protocol::TENSOR_TAG_ENCRYPTED => {
                tracing::info!(
                    %peer,
                    payload_len = payload.len(),
                    "DIAG: Received encrypted tensor"
                );
                match protocol::decode_layer_forward_encrypted(payload) {
                    Ok((mut forward, sealed, aad)) => {
                        let sender_node_id = self.peer_to_node_id(&peer);
                        tracing::debug!(
                            %peer,
                            request_id = %forward.request_id,
                            sender_node_id = ?sender_node_id.as_ref().map(|n| format!("{}", n)),
                            aad_len = aad.len(),
                            sealed_len = sealed.len(),
                            has_session = sender_node_id.as_ref().is_some_and(|n| self.shared_state.session_manager.has_session(n)),
                            "DIAG: decrypting tensor"
                        );
                        if let Some(node_id) = sender_node_id {
                            match self
                                .shared_state
                                .session_manager
                                .open(&node_id, &sealed, &aad)
                            {
                                Ok(plaintext) => {
                                    let request_id = forward.request_id;
                                    let is_tp = forward.tp_meta.is_some();
                                    forward.activations = plaintext;
                                    forward.sender_peer_bytes = Some(peer.to_bytes());
                                    if let Err(e) = self.dispatch_authenticated(
                                        Some(&peer),
                                        SwarmMessage::LayerForward(forward),
                                    ) {
                                        self.shared_state
                                            .metrics
                                            .channel_metrics
                                            .network_out
                                            .record_dropped();
                                        tracing::warn!(error = %e, "Outbound channel full, dropping decrypted tensor");
                                        return None;
                                    } else {
                                        self.shared_state
                                            .metrics
                                            .channel_metrics
                                            .network_out
                                            .record_sent();
                                    }
                                    // TP forwards respond via TpAllReduceRequest, not this channel
                                    if is_tp {
                                        return None;
                                    }
                                    return Some(request_id);
                                }
                                Err(e) => {
                                    tracing::error!(
                                        error = %e,
                                        %peer,
                                        request_id = %forward.request_id,
                                        node_id = %node_id,
                                        aad_len = aad.len(),
                                        sealed_len = sealed.len(),
                                        model_id = %forward.model_id,
                                        layer_range = ?forward.layer_range,
                                        seq = forward.sequence_num,
                                        "DIAG: decrypt FAILED — possible AAD mismatch, key mismatch, or corruption"
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
                None
            }
            _ => {
                tracing::warn!(%peer, tag, "Unknown tensor message tag");
                None
            }
        }
    }

    /// Check if a shard is available locally and return the path + request params.
    /// Returns None if the shard is not available. This is sync to avoid holding
    /// `&self` across async boundaries (NetworkManager is not Send).
    fn prepare_shard_read(
        &self,
        req: &crate::types::ShardRequest,
    ) -> Option<(std::path::PathBuf, u64, u64, crate::types::ModelId, u32)> {
        let model_id = &req.shard_id.model_id;
        let shard_index = req.shard_id.index;

        let shard_path = self.shard_store.shard_path(model_id, shard_index);
        if shard_path.exists() {
            return Some((
                shard_path,
                req.chunk_offset,
                req.chunk_size,
                model_id.clone(),
                shard_index,
            ));
        }

        tracing::debug!(
            model = %model_id,
            shard = shard_index,
            "Shard not available locally"
        );
        None
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

        // SEC: Cap pending shard requests to prevent memory exhaustion from
        // malicious peers that send partial chunks in an infinite loop.
        if self.pending_shard_requests.len() >= MAX_PENDING_SHARD_REQUESTS {
            tracing::warn!(
                count = self.pending_shard_requests.len(),
                "Pending shard requests at capacity — dropping new request"
            );
            return;
        }

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

    /// Send an arbitrary SwarmMessage to a specific peer via request_response.
    /// Used for AllReduce and other point-to-point messages.
    fn handle_send_rr_message(
        &mut self,
        target_peer_bytes: Vec<u8>,
        msg: SwarmMessage,
        label: &str,
    ) {
        let peer_id = match libp2p::PeerId::from_bytes(&target_peer_bytes) {
            Ok(id) => id,
            Err(e) => {
                tracing::warn!(error = %e, label, "Invalid peer ID bytes for rr message");
                return;
            }
        };

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
        if let Ok(mut stats) = self.shared_state.metrics.node_stats.try_write() {
            stats.peers_connected = count;
        }
    }

    /// Save current peer addresses to the persistent cache.
    /// Appends `/p2p/<peer_id>` to each address so that `bootstrap_peers` can
    /// skip already-connected peers via the `is_connected()` check.
    fn save_peer_cache(&self) {
        let addrs: Vec<String> = self
            .shared_state
            .peer_registry
            .iter()
            .flat_map(|entry| {
                // Resolve PeerId from the reverse index so we can append /p2p/<id>
                let peer_id = entry
                    .peer_id_bytes
                    .as_ref()
                    .and_then(|b| libp2p::PeerId::from_bytes(b).ok());
                entry
                    .addresses
                    .iter()
                    .map(move |addr| {
                        if let Some(ref pid) = peer_id {
                            // Only append if not already present
                            if !addr.contains("/p2p/") {
                                return format!("{addr}/p2p/{pid}");
                            }
                        }
                        addr.clone()
                    })
                    .collect::<Vec<_>>()
            })
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
                // SEC: Filter out private/link-local/loopback/CGN IPs to prevent SSRF
                if is_non_public_addr(addr_str) {
                    tracing::debug!(addr = %addr_str, "PEX: skipping private/loopback address");
                    continue;
                }

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

    // ── S5: DHT-based shard holder resolution ──

    /// Issue DHT provider queries for all shards of a model.
    /// Results arrive asynchronously via GetProviders events and are merged
    /// into the model_registry's bounded shard_holders cache.
    fn handle_dht_provider_query(&mut self, model_id: &crate::types::ModelId) {
        // Dedup: skip if we already have pending queries for any shard of this model
        let already_querying = self
            .pending_provider_queries
            .values()
            .any(|sid| &sid.model_id == model_id);
        if already_querying {
            tracing::debug!(model = %model_id, "DHT query skipped — already querying this model");
            return;
        }

        let manifest = match self.shared_state.model_registry.get_manifest(model_id) {
            Some(m) => m,
            None => {
                tracing::debug!(model = %model_id, "DHT query skipped — manifest not found");
                return;
            }
        };

        let mut queried = 0;
        for shard_info in &manifest.shards {
            let shard_id = crate::types::ShardId {
                model_id: model_id.clone(),
                index: shard_info.index,
            };
            match crate::network::discovery::query_shard_providers(&mut self.swarm, &shard_id) {
                Ok(query_id) => {
                    self.pending_provider_queries.insert(query_id, shard_id);
                    queried += 1;
                }
                Err(e) => {
                    tracing::debug!(error = %e, "DHT provider query failed");
                }
            }
        }

        if queried > 0 {
            tracing::info!(
                model = %model_id,
                shards_queried = queried,
                "Issued DHT provider queries for shard holders"
            );
        }

        // Cap pending queries to prevent unbounded growth
        if self.pending_provider_queries.len() > MAX_PENDING_PROVIDER_QUERIES {
            let excess = self.pending_provider_queries.len() - MAX_PENDING_PROVIDER_QUERIES;
            let keys: Vec<_> = self
                .pending_provider_queries
                .keys()
                .take(excess)
                .cloned()
                .collect();
            for k in keys {
                self.pending_provider_queries.remove(&k);
            }
        }
    }

    /// Handle DHT provider results — convert PeerIds to NodeIds and merge
    /// into the model_registry's bounded shard_holders cache.
    fn handle_dht_providers_found(
        &mut self,
        query_id: libp2p::kad::QueryId,
        providers: &std::collections::HashSet<libp2p::PeerId>,
    ) {
        let shard_id = match self.pending_provider_queries.get(&query_id) {
            Some(sid) => sid.clone(),
            None => return, // Unknown query, ignore
        };

        let mut resolved = Vec::new();
        for peer_id in providers {
            // Try local reverse map first (fast)
            if let Some(node_id) = self.peer_to_node_id(peer_id) {
                resolved.push(node_id);
            } else if let Some(node_id) = crate::network::transport::peer_id_to_node_id(peer_id) {
                // Derive from PeerId directly (works for Ed25519 identity-hashed PeerIds)
                resolved.push(node_id);
            }
        }

        if !resolved.is_empty() {
            tracing::debug!(
                shard = ?shard_id,
                providers = resolved.len(),
                "Merging DHT providers into shard holders cache"
            );
            self.shared_state
                .model_registry
                .merge_dht_providers(&shard_id, &resolved);
        }
    }
}

/// Read a shard chunk from disk using spawn_blocking to avoid stalling the
/// async event loop. This is a free function (not on NetworkManager) so
/// `&self` is not captured across the await point.
async fn read_shard_chunk_async(
    path: std::path::PathBuf,
    offset: u64,
    chunk_size: u64,
    model_id: crate::types::ModelId,
    shard_index: u32,
) -> SwarmResponse {
    let result = tokio::task::spawn_blocking(move || {
        use std::io::{Read, Seek, SeekFrom};
        match std::fs::File::open(&path) {
            Ok(mut file) => {
                let total_size = file.metadata().map(|m| m.len()).unwrap_or(0);
                let chunk_size = chunk_size.min(crate::network::protocol::SHARD_CHUNK_SIZE);
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
        }
    })
    .await;

    match result {
        Ok(response) => response,
        Err(e) => {
            tracing::warn!(error = %e, "spawn_blocking panicked during shard read");
            SwarmResponse::ShardData(crate::types::ShardResponse {
                data: vec![],
                total_size: 0,
            })
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
