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
use crate::network::protocol::{self, SwarmRequest, SwarmResponse};
use crate::network::relay::RelayServerConfig;
use crate::network::transport;
use crate::types::{NetworkCommand, SwarmMessage};

mod commands;
mod connections;
mod dht;
mod identify;
mod requests;
mod shard_transfer;
mod tensors;

/// Maximum in-flight shard chunk requests before dropping new ones.
const MAX_PENDING_SHARD_REQUESTS: usize = 1024;
/// Maximum lifetime (seconds) for a tensor channel or outbound forward before eviction.
/// Used for both channel cleanup and adaptive timeout upper clamp. Must match
/// the libp2p request_response timeout so our cleanup fires at or just before
/// libp2p's own failure notification — not before (spurious double-failures)
/// and not after (stuck entries outliving the transport).
const MAX_TENSOR_FORWARD_SECS: u64 = behaviour::RR_REQUEST_TIMEOUT_SECS;
/// libp2p swarm idle connection timeout. Connections with no traffic for this long are closed.
const IDLE_CONNECTION_TIMEOUT_SECS: u64 = 120;
/// Interval for periodic PEX ping health checks. Keeps the outbound queue shallow so
/// tensor forwards get immediate service instead of queueing behind stale requests.
const RR_PING_INTERVAL_SECS: u64 = 120;
/// Interval for stale tensor forward cleanup — catches requests silently dropped by libp2p
/// (no OutboundFailure event) due to stale ConnectionIds or handler starvation.
const STALE_TENSOR_CLEANUP_SECS: u64 = 10;
/// Cancel a shard download after this many seconds without any chunk progress.
/// Catches silent connection drops and handler starvation where no OutboundFailure fires.
const SHARD_STALL_SECS: u64 = 30;
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
/// Maximum concurrent inbound prefix-KV fetch requests before replying miss.
const MAX_INBOUND_PREFIX_FETCHES: usize = 256;
/// Maximum entries in connection_addrs before half-eviction of oldest ConnectionIds.
const MAX_CONNECTION_ADDRS: usize = 1024;
/// Maximum entries in ping_sent_times before pruning stale entries.
const MAX_PING_ENTRIES: usize = 2048;
/// How often the run loop wakes to process the pending_redial queue.
const REDIAL_CHECK_INTERVAL_SECS: u64 = 1;
/// Polling cadence for the Kademlia bootstrap backoff schedule.
const BOOTSTRAP_POLL_INTERVAL_SECS: u64 = 5;
/// Cadence for the "no peers connected" WARN log while isolated.
const NO_PEERS_WARN_INTERVAL_SECS: u64 = 30;
/// Cadence for the liveness heartbeat that refreshes `last_seen` on connected peers.
const LIVENESS_INTERVAL_SECS: u64 = 30;
/// Minimum redial delay added to peers that never completed Identify handshake.
/// Jitter breaks mDNS simultaneous-dial race symmetry.
const REDIAL_JITTER_MIN_MS: u64 = 2000;
/// Random window added on top of `REDIAL_JITTER_MIN_MS` (effective delay 2-5s).
const REDIAL_JITTER_RANGE_MS: u64 = 3000;

use super::helpers::{extract_ipv4_bytes, is_non_public_ipv4_bytes, swarm_event_name};

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
    /// Last time a shard download made forward progress (chunk received or request dispatched).
    /// Used by the stale shard watchdog to detect stalled downloads that never fire
    /// OutboundFailure (e.g., silent connection drops, handler starvation).
    shard_last_progress_at: HashMap<crate::types::ShardId, std::time::Instant>,
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
    /// Maps OutboundRequestId → inference UUID for tensor *results* sent via the
    /// fallback request path (`send_tensor_result_as_request`). Used purely for
    /// observability: on OutboundFailure we can log which result UUID failed
    /// to reach the upstream requester. We cannot notify the upstream's
    /// pipeline from here (it lives on the other peer) — it handles its own
    /// timeout via its own `pending_tensor_outbound` watchdog.
    pending_tensor_result_outbound: HashMap<OutboundRequestId, (uuid::Uuid, std::time::Instant)>,
    /// Observability: track streaming-token + rr-message sends so OutboundFailure
    /// events can be attributed to a label. Purely for logging — the upstream
    /// protocol handles its own timeouts.
    pending_rr_observability: HashMap<OutboundRequestId, (String, std::time::Instant)>,
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
    /// Maps PeerId → most recent connected remote Multiaddr. Used by the Identify
    /// handler to mark peers as LAN when their connected address is loopback/private,
    /// even if their advertised listen_addrs are empty or public.
    peer_remote_addrs: HashMap<libp2p::PeerId, Multiaddr>,
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
    /// Item 8 Phase 2: maps libp2p `OutboundRequestId` (minted on
    /// `send_request`) to the fetcher's `request_id` Uuid. On
    /// `SwarmResponse::PrefixKvData` arrival we pop the mapping, look up
    /// the caller-installed oneshot in `state.pending_prefix_kv_fetches`,
    /// and fulfil it with the payload. On `OutboundFailure` we resolve
    /// with `None`.
    pending_prefix_kv_outbound: HashMap<OutboundRequestId, uuid::Uuid>,
    /// Item 8 Phase 2b: inbound `SwarmRequest::PrefixKvFetch` reply
    /// channels. Manager stashes the `ResponseChannel` here keyed by a
    /// fresh `ticket` Uuid, spawns a task that fetches from the local
    /// worker, and emits `NetworkCommand::DeliverPrefixKvResponse {
    /// ticket, ... }` when the worker replies. Manager pops the stored
    /// channel and sends the response on its substream.
    pending_prefix_kv_inbound: HashMap<
        uuid::Uuid,
        (
            uuid::Uuid,
            std::time::Instant,
            request_response::ResponseChannel<SwarmResponse>,
        ),
    >,
    /// Item 8 Phase 2b: internal sender used by the manager's own spawned
    /// tasks to push commands back into the event loop (e.g. the
    /// `DeliverPrefixKvResponse` reply after the serving-side worker
    /// IPC completes). Bounded — full queue means the manager is
    /// overloaded; callers (internal tasks) drop their reply rather than
    /// block.
    internal_cmd_tx: mpsc::Sender<NetworkCommand>,
    internal_cmd_rx: mpsc::Receiver<NetworkCommand>,
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
        let (internal_cmd_tx, internal_cmd_rx) = mpsc::channel::<NetworkCommand>(256);

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
            shard_last_progress_at: HashMap::new(),
            peer_to_node: DashMap::new(),
            buffered_gossip: Vec::new(),
            relay_activated: false,
            pending_tensor_outbound: HashMap::new(),
            pending_tensor_result_outbound: HashMap::new(),
            pending_rr_observability: HashMap::new(),
            pending_tensor_channels: HashMap::new(),
            connection_addrs: HashMap::new(),
            peer_remote_addrs: HashMap::new(),
            ping_sent_times: HashMap::new(),
            shutdown_rx,
            pending_redial: Vec::new(),
            dht_query_rx,
            pending_provider_queries: HashMap::new(),
            pending_prefix_kv_outbound: HashMap::new(),
            pending_prefix_kv_inbound: HashMap::new(),
            internal_cmd_tx,
            internal_cmd_rx,
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
            spec_logits: Vec::new(),
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

        // Persistent pipeline stream: obtain a Control, register the protocol
        // acceptor, publish the client to SharedState, and spawn the accept
        // loop. Off-switch is `config.inference.persistent_pipeline_stream`,
        // enforced at the call site in `forward_through_segments` — the
        // protocol always registers so remote peers can connect regardless of
        // who flipped the flag first.
        {
            let mut control = self.swarm.behaviour().pipeline_stream.new_control();
            let incoming = control
                .accept(libp2p::StreamProtocol::new(
                    crate::network::pipeline_stream::PROTOCOL_PIPELINE,
                ))
                .map_err(|e| SwarmError::Network(format!("pipeline_stream accept: {e}")))?;
            let client = std::sync::Arc::new(
                crate::network::pipeline_stream::PipelineStreamClient::new(control),
            );
            if self
                .shared_state
                .pipeline_stream_client
                .set(client)
                .is_err()
            {
                tracing::warn!("pipeline_stream_client already set — ignoring");
            }
            crate::network::pipeline_stream::spawn_accept_loop(
                incoming,
                self.shared_state.clone(),
                self.outbound_tx.clone(),
            );
            tracing::info!(
                protocol = crate::network::pipeline_stream::PROTOCOL_PIPELINE,
                "pipeline_stream behaviour armed"
            );
        }

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

            // Loopback probe: find same-host peers when mDNS is off
            // (WSL2 default) and the peer cache / bootstrap list is empty.
            // Cheap — only fires a handful of TCP dials on 127.0.0.1.
            discovery::probe_loopback_peers(
                &mut self.swarm,
                self.shared_state.config.node.listen_port,
            );
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
        let mut redial_interval =
            tokio::time::interval(std::time::Duration::from_secs(REDIAL_CHECK_INTERVAL_SECS));
        redial_interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

        // Kademlia bootstrap backoff — retries at 10s, 30s, 60s, 120s after a
        // no-peers startup instead of waiting the full 300s discovery tick.
        // `bootstrap_backoff_secs` is the next retry delay; resets to 10s once
        // any peer connects. Works in tandem with the loopback probe and
        // discovery_interval.
        let backoff_schedule: [u64; 4] = [10, 30, 60, 120];
        let mut backoff_idx: usize = 0;
        let mut bootstrap_retry_deadline: Option<std::time::Instant> = if config.pool.offline_mode {
            None
        } else {
            Some(std::time::Instant::now() + std::time::Duration::from_secs(backoff_schedule[0]))
        };
        let mut bootstrap_retry_interval =
            tokio::time::interval(std::time::Duration::from_secs(BOOTSTRAP_POLL_INTERVAL_SECS));
        bootstrap_retry_interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

        // "No peers" WARN loop — surfaces the "node is running but totally
        // isolated" state (happens on fresh WSL2 installs with mDNS disabled
        // and no bootstrap peers configured). Fires every 30s while
        // connected_peers == 0.
        let mut no_peers_interval = tokio::time::interval_at(
            tokio::time::Instant::now()
                + std::time::Duration::from_secs(NO_PEERS_WARN_INTERVAL_SECS),
            std::time::Duration::from_secs(NO_PEERS_WARN_INTERVAL_SECS),
        );
        no_peers_interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        let startup_instant = std::time::Instant::now();

        // Liveness heartbeat — every 30s, refresh `last_seen` on every peer
        // libp2p still reports as connected. The HealthMonitor evicts peers at
        // 90s of silence, but rr_ping fires every 120s — so a peer with no
        // other inbound traffic in its first 90s would be evicted while still
        // having a live TCP/QUIC connection (especially on WSL2 where QUIC
        // substream negotiation is slow). This tick is the floor.
        let mut liveness_interval =
            tokio::time::interval(std::time::Duration::from_secs(LIVENESS_INTERVAL_SECS));
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
                // Kademlia bootstrap retry with exponential backoff.
                // Checks every 5s whether we've passed the next retry deadline
                // AND are still peerless; fires bootstrap + loopback probe if so,
                // then advances the backoff schedule. Resets to the first step
                // once any peer connects.
                _ = bootstrap_retry_interval.tick() => {
                    let connected = self.swarm.connected_peers().count();
                    if connected > 0 {
                        backoff_idx = 0;
                        bootstrap_retry_deadline = None;
                    } else if let Some(deadline) = bootstrap_retry_deadline {
                        if std::time::Instant::now() >= deadline
                            && !self.shared_state.credits.offline_mode.load(std::sync::atomic::Ordering::Relaxed)
                        {
                            tracing::info!(
                                attempt_delay_secs = backoff_schedule[backoff_idx],
                                "Bootstrap retry — still no peers, re-triggering Kademlia bootstrap + loopback probe"
                            );
                            let _ = discovery::trigger_bootstrap(&mut self.swarm);
                            // Re-dial bootstrap + cached peers
                            let _ = discovery::bootstrap_peers(
                                &mut self.swarm,
                                &self.shared_state.config.network.bootstrap_peers,
                            );
                            let cached = crate::network::peer_cache::load_peer_cache(
                                &self.shared_state.db,
                            );
                            if !cached.is_empty() {
                                let _ = discovery::bootstrap_peers(&mut self.swarm, &cached);
                            }
                            discovery::probe_loopback_peers(
                                &mut self.swarm,
                                self.shared_state.config.node.listen_port,
                            );
                            backoff_idx = (backoff_idx + 1).min(backoff_schedule.len() - 1);
                            bootstrap_retry_deadline = Some(
                                std::time::Instant::now()
                                    + std::time::Duration::from_secs(backoff_schedule[backoff_idx]),
                            );
                        }
                    }
                }
                // "No peers" visibility loop — every 30s while isolated, log a
                // WARN that lists every discovery path with its state so the
                // operator can see *why* no peers are being found.
                _ = no_peers_interval.tick() => {
                    let connected = self.swarm.connected_peers().count();
                    if connected == 0 {
                        let cfg = &self.shared_state.config.network;
                        let age_secs = startup_instant.elapsed().as_secs();
                        let cached = crate::network::peer_cache::load_peer_cache(
                            &self.shared_state.db,
                        );
                        let offline = self.shared_state.credits.offline_mode
                            .load(std::sync::atomic::Ordering::Relaxed);
                        tracing::warn!(
                            age_secs,
                            mdns = cfg.enable_mdns,
                            quic = cfg.enable_quic,
                            autonat = cfg.enable_autonat,
                            dcutr = cfg.enable_dcutr,
                            bootstrap_peers = cfg.bootstrap_peers.len(),
                            cached_peers = cached.len(),
                            offline_mode = offline,
                            listen_addr = %cfg.listen_address,
                            "No peers connected — all active discovery paths listed. If all counters are 0 and mDNS is off, configure bootstrap_peers in config.toml or share an invite code."
                        );
                    }
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
                    // Also sweep the observability-only result-fallback map.
                    self.pending_tensor_result_outbound.retain(|_id, (_, inserted)| {
                        inserted.elapsed().as_secs() < MAX_TENSOR_FORWARD_SECS
                    });
                    self.pending_rr_observability.retain(|_id, (_, inserted)| {
                        inserted.elapsed().as_secs() < MAX_TENSOR_FORWARD_SECS
                    });
                    // Sweep pending_prefix_kv_inbound tickets whose serving task
                    // panicked or whose DeliverPrefixKvResponse command was dropped
                    // (internal_cmd_tx full). Without this, under load the 256-entry
                    // cap silently fills with orphans and all new fetches reply miss.
                    let before_prefix = self.pending_prefix_kv_inbound.len();
                    self.pending_prefix_kv_inbound.retain(|_ticket, (_, inserted, _chan)| {
                        inserted.elapsed().as_secs() < MAX_TENSOR_FORWARD_SECS
                    });
                    let removed_prefix = before_prefix - self.pending_prefix_kv_inbound.len();
                    if removed_prefix > 0 {
                        tracing::warn!(
                            removed = removed_prefix,
                            remaining = self.pending_prefix_kv_inbound.len(),
                            "Swept stale pending_prefix_kv_inbound tickets"
                        );
                    }
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

                    // Stale shard download watchdog — catches downloads that stopped
                    // making progress without firing OutboundFailure (silent connection
                    // drops, handler starvation). After SHARD_STALL_SECS of no chunks,
                    // cancel and retry via retry_shard_or_fallback.
                    let now = std::time::Instant::now();
                    let stalled: Vec<(crate::types::ShardId, libp2p::PeerId, OutboundRequestId)> = self
                        .pending_shard_requests
                        .iter()
                        .filter_map(|(req_id, (peer_id, shard_id))| {
                            let last = self
                                .shard_last_progress_at
                                .get(shard_id)
                                .copied()
                                .unwrap_or(now);
                            if now.duration_since(last).as_secs() > SHARD_STALL_SECS {
                                Some((shard_id.clone(), *peer_id, *req_id))
                            } else {
                                None
                            }
                        })
                        .collect();
                    for (shard_id, peer_id, req_id) in stalled {
                        tracing::warn!(
                            %peer_id,
                            model = %shard_id.model_id,
                            shard = shard_id.index,
                            stall_secs = SHARD_STALL_SECS,
                            "DIAG: stalled shard download — cancelling + retrying"
                        );
                        self.pending_shard_requests.remove(&req_id);
                        self.retry_shard_or_fallback(
                            shard_id,
                            peer_id,
                            &format!("stalled (no progress >{SHARD_STALL_SECS}s)"),
                        );
                    }
                }
                // Process pending re-dials (mDNS simultaneous-dial race recovery).
                // When both sides discover each other via mDNS at the same time, both dial,
                // and with max_established_per_peer=1, the loser's connection is immediately
                // closed. We schedule a re-dial with random jitter (2-5s) so one side wins.
                _ = redial_interval.tick() => {
                    // Cap queue unconditionally — push sites may race past the
                    // soft check or skip it entirely. Truncating before dispatch
                    // evicts oldest (front) first in FIFO fashion: reverse,
                    // truncate, reverse so the newest entries survive.
                    if self.pending_redial.len() > MAX_PENDING_REDIAL {
                        self.pending_redial.reverse();
                        self.pending_redial.truncate(MAX_PENDING_REDIAL);
                        self.pending_redial.reverse();
                    }
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
                // Item 8 Phase 2b: internal commands from spawned serve
                // tasks (e.g. `DeliverPrefixKvResponse` after an inbound
                // fetch has been served by the local worker).
                Some(cmd) = self.internal_cmd_rx.recv() => {
                    self.handle_outbound_command(cmd).await;
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
                        SwarmRequest::PrefixKvFetch(_) => "prefix_kv_fetch",
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
                        SwarmResponse::PrefixKvData(_) => "prefix_kv_data",
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
                    self.pending_tensor_result_outbound.remove(&request_id);
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
                // Log result-send fallback failures with UUID context.
                // We can't notify the upstream requester from here — their pipeline
                // has its own timeout via their pending_tensor_outbound watchdog.
                if let Some((result_uuid, _)) =
                    self.pending_tensor_result_outbound.remove(&request_id)
                {
                    tracing::error!(
                        %peer,
                        inference_request_id = %result_uuid,
                        %error,
                        "Tensor result fallback OutboundFailure — upstream will timeout"
                    );
                }
                if let Some((label, _)) = self.pending_rr_observability.remove(&request_id) {
                    tracing::warn!(
                        %peer,
                        label,
                        %error,
                        "rr-message OutboundFailure — upstream will handle via its own timeout"
                    );
                }
                // Item 8 Phase 2: unblock a pending prefix-KV fetch on failure.
                if let Some(uuid) = self.pending_prefix_kv_outbound.remove(&request_id) {
                    if let Some((_, tx)) = self.shared_state.pending_prefix_kv_fetches.remove(&uuid)
                    {
                        let _ = tx.send(None);
                    }
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
                        "DIAG: shard download OutboundFailure — attempting peer failover"
                    );
                    // Try another peer; fall back to HF only after retries exhausted.
                    self.retry_shard_or_fallback(shard_id, peer, &format!("{error}"));
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

            // ── GossipSub peer subscribed — flush matching buffered messages (NET-I4) ──
            //
            // Only the just-subscribed topic is eligible for replay — a
            // Subscribed{peer_id, topic=X} event tells us the mesh now has
            // at least one peer on topic X, but says nothing about topic Y.
            // Before this filter, ANY Subscribed event iterated the whole
            // buffer and called publish() on every entry; publish() would
            // still return Err for topics with no subscribers (gossipsub
            // routes correctly, so this wasn't an info leak), but the
            // entry got re-buffered, wasting a full O(buffer) pass per
            // Subscribed event on a multi-topic mesh.
            SwarmEvent::Behaviour(SwarmBehaviourEvent::Gossipsub(
                gossipsub::Event::Subscribed { peer_id, topic },
            )) => {
                tracing::debug!(%peer_id, %topic, "Peer subscribed to topic");
                let subscribed_topic_str = topic.to_string();
                let has_match = self
                    .buffered_gossip
                    .iter()
                    .any(|(t, _)| t == &subscribed_topic_str);
                if has_match {
                    let mut remaining = Vec::with_capacity(self.buffered_gossip.len());
                    let mut replayed = 0;
                    for (topic_str, data) in std::mem::take(&mut self.buffered_gossip) {
                        if topic_str != subscribed_topic_str {
                            remaining.push((topic_str, data));
                            continue;
                        }
                        let gossip_topic = IdentTopic::new(&topic_str);
                        match self
                            .swarm
                            .behaviour_mut()
                            .gossipsub
                            .publish(gossip_topic, data.clone())
                        {
                            Ok(_) => replayed += 1,
                            Err(_) => remaining.push((topic_str, data)),
                        }
                    }
                    self.buffered_gossip = remaining;
                    if replayed > 0 {
                        tracing::info!(
                            topic = %subscribed_topic_str,
                            count = replayed,
                            "Replayed buffered GossipSub messages for newly-subscribed topic"
                        );
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
                self.handle_identify_received(peer_id, info, connection_id);
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
}
