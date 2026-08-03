use std::collections::HashMap;
use std::sync::Arc;

use dashmap::DashMap;
use futures::StreamExt;
use libp2p::request_response::{self, OutboundRequestId};
use libp2p::{Multiaddr, Swarm, SwarmBuilder};
use tokio::sync::{mpsc, watch};

use crate::config::Config;
use crate::daemon::SharedState;
use crate::error::SwarmError;
use crate::identity::Identity;
use crate::model::acquisition::AcquisitionCommand;
use crate::model::shard::ShardStore;
use crate::network::behaviour::{self, SwarmBehaviour};
use crate::network::discovery;
use crate::network::protocol::{SwarmRequest, SwarmResponse};
use crate::network::relay::RelayServerConfig;
use crate::network::transport;
use crate::types::{NetworkCommand, SwarmMessage};

mod commands;
mod connections;
mod dht;
mod events;
/// Shared with `network::peer_cache` so the advertise path and the cache path
/// apply one reachability predicate rather than two that can drift.
pub(crate) use events::addr_is_remotely_reachable;
mod identify;
mod relay;
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
/// Per-message ACK deadline for streaming-tracked rr sends (`SendDirectMessage`
/// with `delivery_request_id = Some(_)`). When elapsed without a Response or
/// OutboundFailure event, treat as silent-drop and close the streaming
/// caller's channel so it fails fast instead of waiting FIRST_TOKEN_TIMEOUT.
/// 10s is generous on LAN (sub-millisecond ACKs in practice) but short enough
/// to convert a 2-minute hang into a 10-second retry window.
const RR_ACK_TIMEOUT_SECS: u64 = 10;
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
///
/// **Layering with `model/auto_manage/manager.rs::P2P_PERMIT_STALL_SECS` (180s):**
/// This is the network-layer first-line guard — fires on a single peer-shard
/// transfer that stalls, triggering `retry_shard_or_fallback` (try another peer
/// for this shard). The auto-manage permit sweep is the second-line escalation
/// — after 180s of no progress on the whole acquisition it releases the
/// semaphore permit so the HF fallback path can run. Keep this strictly less
/// than `P2P_PERMIT_STALL_SECS` so per-peer retries get a chance before the
/// outer permit gives up.
const SHARD_STALL_SECS: u64 = 30;
/// NETWORKING_PLAN Phase 3 — target number of connected relay-capable peers.
/// While a node has fewer than this, each discovery tick queries the DHT for
/// more relays to dial, so relaying survives the loss of any single relay
/// (including the bootstrap anchor). Low, since a couple of relay paths suffice.
const MIN_RELAY_CONNECTIONS: usize = 2;
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
/// Maximum concurrent inbound shard-transfer requests being served off the
/// swarm event loop. Beyond this cap, new requests reply with an empty
/// chunk so the requester can retry / fall back rather than waiting on the
/// serving task to drain. Bound is generous because each ticket only holds
/// a `ResponseChannel` (cheap) — the actual disk + throttle work is in the
/// spawned task.
const MAX_INBOUND_SHARD_FETCHES: usize = 256;
/// Maximum age of a stale `pending_shard_responses` ticket before the sweep
/// drops it. Sized to the worst-case shard chunk read + bandwidth-throttle
/// at the cluster's slowest cap (4 MB at 1 Mbps ≈ 32s, plus disk latency
/// headroom). Beyond this, the serving task likely panicked or was
/// orphaned; the channel will close on drop.
const PENDING_SHARD_TICKET_TTL_SECS: u64 = 60;
/// Maximum entries in connection_addrs before half-eviction of oldest ConnectionIds.
const MAX_CONNECTION_ADDRS: usize = 1024;
/// Maximum entries in peer_remote_addrs before half-eviction of stale peers.
/// Disconnected peers' entries are removed in `handle_connection_closed`, but
/// the cap defends against missed close events leaking entries into the map.
const MAX_PEER_REMOTE_ADDRS: usize = 1024;
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
/// Grace period after startup before the belt-and-suspenders relay fallback fires
/// (gives UPnP + AutoNAT a chance to confirm a direct address first). Checked on
/// the liveness tick, so the effective first check lands at the next 30s tick past
/// this delay.
const RELAY_FALLBACK_DELAY_SECS: u64 = 45;
/// Minimum redial delay added to peers that never completed Identify handshake.
/// Jitter breaks mDNS simultaneous-dial race symmetry.
const REDIAL_JITTER_MIN_MS: u64 = 2000;
/// Random window added on top of `REDIAL_JITTER_MIN_MS` (effective delay 2-5s).
const REDIAL_JITTER_RANGE_MS: u64 = 3000;
/// How many times a dial to a peer we were previously connected to is retried
/// before we accept it has left. A single attempt was not enough: a re-dial that
/// lands while the peer is still rebooting fails, and a failed dial raises
/// `OutgoingConnectionError` rather than `ConnectionClosed`, so nothing
/// re-enqueued it and the peer stayed forgotten until it announced itself.
const MAX_REDIAL_ATTEMPTS: u32 = 5;
/// Backoff before each successive retry. Reaches ~8 minutes in total, which
/// covers a peer reboot without becoming a re-dial storm against one that has
/// genuinely gone: attempts are capped and only peers we have actually been
/// connected to are ever retried.
const REDIAL_BACKOFF_MS: [u64; MAX_REDIAL_ATTEMPTS as usize] =
    [5_000, 15_000, 45_000, 120_000, 300_000];
/// Cap on `redial_attempts` so a churn storm cannot grow it without bound.
const MAX_REDIAL_TRACKED_PEERS: usize = 256;

/// Consecutive request/response failures to one peer before the connection is
/// closed as unusable.
///
/// libp2p keeps a TCP+yamux connection open long after the peer stopped
/// answering: blocking one direction of a peer's traffic left every request
/// timing out for 200s with `is_connected=true` the whole time, so it stayed in
/// `connected_node_ids` — the scheduler's liveness oracle — and kept being
/// offered work it could not do. A QUIC connection in the same state dropped in
/// ~30s, so which transport a peer happens to be on decided whether this was a
/// half-minute problem or an unbounded one.
///
/// Any successful response resets the count, so this only fires on a peer that
/// has answered NOTHING across the whole run.
///
/// The number is measured, not guessed. Counting the worst consecutive-failure
/// run per peer across a full day of this node's logs: the **anchor — a healthy,
/// critical relay — reached 5**, while peers that were genuinely gone reached
/// 34, 40, 56 and 121. A threshold of 5 would therefore have disconnected the
/// relay during normal operation, which is a far worse outcome than the bug
/// being fixed. 20 sits in the gap: four times the worst healthy run observed,
/// and well under the shortest dead-peer run.
///
/// Re-measure before changing it. The cost of being wrong is asymmetric — too
/// low disconnects working peers (and the relay a NAT'd node depends on), too
/// high just delays eviction of one that has already stopped answering.
const MAX_CONSECUTIVE_RR_FAILURES: u32 = 20;

/// NetworkManager owns the libp2p Swarm and is the sole interface to the P2P network.
pub struct NetworkManager {
    shared_state: Arc<SharedState>,
    swarm: Swarm<SwarmBehaviour>,
    /// Receives commands from daemon tasks (broadcast, send tensor, etc.)
    inbound_rx: mpsc::Receiver<NetworkCommand>,
    /// A sender into our OWN `inbound_rx`, for spawned tasks that must post a
    /// command back into the event loop. The loop owns state a `tokio::spawn`
    /// cannot borrow (`pending_tensor_channels` in particular), so a detached
    /// task answers a peer by queueing a command rather than touching it.
    self_command_tx: mpsc::Sender<NetworkCommand>,
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
    /// Last time we attempted relay activation, to rate-limit retries until one
    /// succeeds (AutoNAT-unreachable results + the startup fallback both trigger
    /// `try_activate_relay`).
    last_relay_attempt: Option<std::time::Instant>,
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
    /// Track outbound rr-message sends. Three uses:
    /// 1. Attribute OutboundFailure events to a label for logging.
    /// 2. Stale-sweep: any entry older than `RR_ACK_TIMEOUT_SECS` indicates
    ///    libp2p never delivered the message (silent-drop case observed
    ///    under load — neither Response nor OutboundFailure fires).
    /// 3. When the third value is `Some(uuid)`, the entry corresponds to a
    ///    streaming caller (typically remote-generate fast path) whose
    ///    `streaming_token_txs[uuid]` should be closed on stale/failure so
    ///    the caller fails fast instead of waiting 120s for first-token.
    pending_rr_observability:
        HashMap<OutboundRequestId, (String, std::time::Instant, Option<uuid::Uuid>)>,
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
    /// PeerId → set of DIRECT (non-relay-circuit) connection ids. Populated on
    /// ConnectionEstablished when the remote address is not a `/p2p-circuit`,
    /// cleared on ConnectionClosed. Lets the relay send path (NETWORKING_PLAN
    /// Phase 1) prefer the application-level relay over a flaky relay *circuit*
    /// when the circuit is the only path to a peer — a peer "connected" only via
    /// a circuit still can't reliably round-trip request_response.
    peer_direct_conns:
        HashMap<libp2p::PeerId, std::collections::HashSet<libp2p::swarm::ConnectionId>>,
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
    /// Peers to re-dial, with the addresses to try. The address list may be
    /// empty — the dial then goes by peer id alone and the behaviours supply
    /// addresses. See `handle_connection_closed`.
    pending_redial: Vec<(libp2p::PeerId, Vec<Multiaddr>, std::time::Instant)>,
    /// Re-dial attempts spent on peers we have previously been connected to,
    /// with the addresses to keep trying. An entry exists only once
    /// `try_enqueue_redial` has run for that peer, which is what confines the
    /// retry to peers we actually know rather than every failed dial target.
    /// Cleared when a connection to the peer is established.
    redial_attempts: HashMap<libp2p::PeerId, (Vec<Multiaddr>, u32)>,
    /// Consecutive request/response failures per peer, reset by any success.
    /// At `MAX_CONSECUTIVE_RR_FAILURES` the connection is closed so the
    /// re-dial path can replace it. See that constant.
    rr_failures: HashMap<libp2p::PeerId, u32>,
    /// S5: Receives model IDs for DHT provider queries from scheduler/auto-manage.
    dht_query_rx: mpsc::Receiver<crate::types::ModelId>,
    /// S5: Maps Kademlia QueryId → ShardId for routing GetProviders results.
    pending_provider_queries: HashMap<libp2p::kad::QueryId, crate::types::ShardId>,
    /// NETWORKING_PLAN Phase 3 — set once this node has registered as a DHT
    /// relay-service provider (only if it forwards relay traffic). Retried each
    /// discovery tick until Kademlia has peers, then latched.
    relay_provider_registered: bool,
    /// NETWORKING_PLAN Phase 3 — the in-flight `get_providers` query for the
    /// relay-service key, so its `GetProviders` results are recognized as relay
    /// discovery (dial the peers) rather than shard-holder resolution.
    pending_relay_provider_query: Option<libp2p::kad::QueryId>,
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
    /// Inbound shard-transfer reply channels. Manager stashes the
    /// `ResponseChannel` here keyed by a fresh ticket Uuid, spawns a task
    /// that does the disk read + bandwidth-throttle sleep OFF the swarm
    /// event loop (per gotcha #11), and emits
    /// `NetworkCommand::DeliverShardResponse { ticket, ... }` when the
    /// task completes. Manager then pops the stored channel and sends the
    /// response on its substream. Without this indirection a 4 MB chunk
    /// at a 1 Mbps cap would suspend the swarm task for ~32s.
    pending_shard_responses: HashMap<
        uuid::Uuid,
        (
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

/// Build the startup error for a failed `listen_on`.
///
/// Two things made the original message useless in the most common case.
/// libp2p's `TransportError::Other` renders its inner error, and for the QUIC
/// transport that can be EMPTY — a node whose port was taken exited with the
/// bare text "Failed to listen on QUIC: " and nothing after the colon
/// (reproduced 2026-07-26 by starting a second node on a port already in use,
/// which is the single most likely way a first-time user fails to start).
///
/// So: never emit an empty detail, and when the port really is occupied, say
/// so in terms the user can act on rather than making them interpret a
/// transport error.
fn listen_failure_message<E: std::fmt::Display + std::fmt::Debug>(
    transport: &str,
    listen_ip: &str,
    port: u16,
    err: &E,
) -> String {
    let detail = {
        let shown = format!("{err}");
        if shown.trim().is_empty() {
            format!("{err:?}")
        } else {
            shown
        }
    };
    if port_is_taken(transport, listen_ip, port) {
        format!(
            "Port {port} is already in use, so this node cannot start. \
             Another SwarmLLM node is most likely already running — stop it first, \
             or start this one on a different port with `--port <number>`. \
             ({transport} listen failed: {detail})"
        )
    } else {
        format!("Failed to listen on {transport} port {port}: {detail}")
    }
}

/// Whether `port` is already bound for the transport that just failed.
///
/// Probes the protocol that actually failed: QUIC runs over UDP, so a QUIC
/// failure must test a UDP bind. Testing both and requiring both to fail —
/// the obvious-looking version — reports nothing in the common case where UDP
/// is taken but TCP is free, which is precisely the QUIC failure this exists
/// to explain.
///
/// Only used to phrase an error that has already occurred, so a false negative
/// costs nothing beyond a less specific message.
fn port_is_taken(transport: &str, listen_ip: &str, port: u16) -> bool {
    use std::net::{TcpListener, UdpSocket};
    let host = if listen_ip.is_empty() {
        "0.0.0.0"
    } else {
        listen_ip
    };
    let addr = format!("{host}:{port}");
    if transport.eq_ignore_ascii_case("quic") {
        UdpSocket::bind(&addr).is_err()
    } else {
        TcpListener::bind(&addr).is_err()
    }
}

impl NetworkManager {
    /// Create a new NetworkManager and initialize the libp2p Swarm.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        shared_state: Arc<SharedState>,
        identity: &Identity,
        config: &Config,
        inbound_rx: mpsc::Receiver<NetworkCommand>,
        self_command_tx: mpsc::Sender<NetworkCommand>,
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
        let enable_upnp = config.network.enable_upnp;
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
            // Wrap the transports with DNS resolution so `/dns4` / `/dns6` /
            // `/dnsaddr` multiaddrs are dialable — without this, dialing a
            // DNS-named peer (e.g. the default `swarmllm.duckdns.org` bootstrap
            // anchor, or any `network.external_addresses` DNS entry) fails with
            // "Multiaddr is not supported". Uses the system resolver.
            .with_dns()
            .map_err(|e| SwarmError::Network(format!("DNS transport error: {e}")))?
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
                    enable_upnp,
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
            self_command_tx,
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
            last_relay_attempt: None,
            pending_tensor_outbound: HashMap::new(),
            pending_tensor_result_outbound: HashMap::new(),
            pending_rr_observability: HashMap::new(),
            pending_tensor_channels: HashMap::new(),
            connection_addrs: HashMap::new(),
            peer_direct_conns: HashMap::new(),
            peer_remote_addrs: HashMap::new(),
            ping_sent_times: HashMap::new(),
            shutdown_rx,
            pending_redial: Vec::new(),
            redial_attempts: HashMap::new(),
            rr_failures: HashMap::new(),
            dht_query_rx,
            pending_provider_queries: HashMap::new(),
            relay_provider_registered: false,
            pending_relay_provider_query: None,
            pending_prefix_kv_outbound: HashMap::new(),
            pending_prefix_kv_inbound: HashMap::new(),
            pending_shard_responses: HashMap::new(),
            internal_cmd_tx,
            internal_cmd_rx,
            // Pre-allocate at the rate-limit cap so the Vec never grows past
            // PEX_MAX_PER_WINDOW (R93 — capacity creep otherwise persists
            // across bursts).
            pex_inbound_timestamps: Vec::with_capacity(PEX_MAX_PER_WINDOW),
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
        self.dispatch_authenticated_as(sender, msg)
    }

    /// Inject a message into the dispatch feed with an EXPLICIT logical sender.
    /// Used by the relay-unwrap path (NETWORKING_PLAN Phase 1): an inner message
    /// extracted from a `RelayedEnvelope` is dispatched as though `origin` sent
    /// it directly, even though the transport-authenticated sender was the
    /// relay. All the dispatch handlers' "requires authenticated sender" gates
    /// then see `origin`, exactly as a direct send would.
    #[allow(clippy::result_large_err)]
    fn dispatch_authenticated_as(
        &self,
        sender: Option<crate::types::NodeId>,
        msg: SwarmMessage,
    ) -> Result<(), mpsc::error::TrySendError<crate::types::AuthenticatedMessage>> {
        self.outbound_tx
            .try_send(crate::types::AuthenticatedMessage {
                sender,
                message: msg,
            })
    }

    /// Decode peer ID bytes; logs and returns None on failure.
    /// Used by `tensors.rs`, `commands.rs`, and `shard_transfer.rs` —
    /// keeping it on the parent so the call sites read uniformly as
    /// `Self::resolve_peer_id(...)` regardless of which sibling file they're in.
    fn resolve_peer_id(bytes: &[u8], label: &str) -> Option<libp2p::PeerId> {
        match libp2p::PeerId::from_bytes(bytes) {
            Ok(id) => Some(id),
            Err(e) => {
                tracing::warn!(error = %e, label, "Invalid peer ID bytes");
                None
            }
        }
    }

    /// Send an error LayerResult to the pipeline for a failed tensor forward.
    fn fail_tensor_forward(
        &mut self,
        request_id: uuid::Uuid,
        peer: &libp2p::PeerId,
        reason: String,
    ) {
        let error_result = crate::types::LayerResult::error(request_id, reason);
        if let Err(e) =
            self.dispatch_authenticated(Some(peer), SwarmMessage::LayerResult(error_result))
        {
            tracing::warn!(error = %e, "Failed to send error LayerResult to pipeline");
        }
    }

    /// Start the network manager event loop. Listens for inbound libp2p
    /// `SwarmEvent`s, daemon `NetworkCommand`s, internal commands, and
    /// `dht_query_rx` model IDs, dispatching each to the appropriate
    /// per-protocol handler in the sibling modules (events.rs, requests.rs,
    /// identify.rs, connections.rs, commands.rs, dht.rs, tensors.rs,
    /// shard_transfer.rs). Returns when the shutdown signal fires or the
    /// inbound channel closes.
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
            self.swarm.listen_on(quic_addr.clone()).map_err(|e| {
                SwarmError::Network(listen_failure_message("QUIC", listen_ip, port, &e))
            })?;

            match self.swarm.listen_on(tcp_addr.clone()) {
                Ok(_) => tracing::info!(%quic_addr, %tcp_addr, "Listening for P2P connections"),
                Err(e) => {
                    tracing::warn!(%quic_addr, error = %e, "TCP listen unavailable, using QUIC only");
                }
            }
        } else {
            self.swarm.listen_on(tcp_addr.clone()).map_err(|e| {
                SwarmError::Network(listen_failure_message("TCP", listen_ip, tcp_port, &e))
            })?;
            tracing::info!(%tcp_addr, "Listening for P2P connections (QUIC disabled)");
        }

        // Manual external-address override: a node that already knows how it is
        // reachable from the internet — a port-forwarded box, a VPS, or a
        // dynamic-DNS anchor — declares it via `network.external_addresses`.
        // Confirm each with the swarm so it flows into identify, the DHT, and
        // every invite code this node mints. List both transports (TCP + QUIC)
        // to advertise a readable DNS name over each. Load-bearing path for a
        // self-hosted anchor node behind CGNAT-free hosting.
        for ext in &config.network.external_addresses.0 {
            let ext = ext.trim();
            if ext.is_empty() {
                continue;
            }
            match ext.parse::<Multiaddr>() {
                Ok(mut maddr) => {
                    // The swarm tracks external addresses without our own peer
                    // id; strip a trailing /p2p if the user added one.
                    if matches!(
                        maddr.iter().last(),
                        Some(libp2p::multiaddr::Protocol::P2p(_))
                    ) {
                        maddr.pop();
                    }
                    self.swarm.add_external_address(maddr.clone());
                    tracing::info!(%maddr, "Declared external address from config (network.external_addresses)");
                }
                Err(e) => {
                    tracing::warn!(
                        external_address = %ext,
                        error = %e,
                        "network.external_addresses entry is not a valid multiaddr — ignoring. \
                         Expected e.g. /dns4/anchor.example.net/tcp/8810 or /ip4/203.0.113.5/tcp/8810"
                    );
                }
            }
        }

        // listen_on returns before NewListenAddr fires (the listener task hasn't
        // finished binding the socket yet), so the listen_multiaddrs snapshot
        // stays empty until the event loop pumps. NewListenAddr will refresh
        // again with the bound address shortly — but if an invite code is
        // minted in that tiny window we'd hand out an empty address list. Seed
        // the snapshot now from the swarm's known listeners; usually still
        // empty here, but the cost is one ArcSwap store.
        self.refresh_listen_multiaddrs();

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
                self.shutdown_rx.clone(),
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
            let cached_peers = self.dialable_peer_cache();
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
                        let cached = self.dialable_peer_cache();
                        if !cached.is_empty() {
                            let _ = discovery::bootstrap_peers(&mut self.swarm, &cached);
                        }
                        // NETWORKING_PLAN Phase 3 — DHT relay discovery.
                        // Register ourselves as a relay provider (once Kademlia
                        // has peers), and — when short on relay connections —
                        // query the DHT for more relays to dial, so the relay
                        // role decentralizes past the bootstrap anchor.
                        if self.is_relay_forwarder() && !self.relay_provider_registered {
                            self.relay_provider_registered =
                                discovery::start_providing_relay_service(&mut self.swarm);
                        }
                        if self.count_connected_relays() < MIN_RELAY_CONNECTIONS {
                            self.pending_relay_provider_query =
                                Some(discovery::query_relay_providers(&mut self.swarm));
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
                            let cached = self.dialable_peer_cache();
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
                        // Report the dialable count, not the raw one — an
                        // operator acting on this line needs the number of
                        // addresses actually being tried.
                        let cached = self.dialable_peer_cache();
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
                    // Belt-and-suspenders relay fallback: if, well after startup,
                    // this node still has NO internet-reachable address (no public
                    // listener, no UPnP/AutoNAT-confirmed external addr, no relay
                    // circuit yet), reserve a relay proactively — don't wait for
                    // AutoNAT to produce a conclusive "unreachable", which can fail
                    // to fire if no AutoNAT servers are reachable. `try_activate_relay`
                    // is idempotent + rate-limited and no-ops once we're reachable.
                    if !self.relay_activated
                        && startup_instant.elapsed().as_secs() >= RELAY_FALLBACK_DELAY_SECS
                    {
                        let reachable = crate::pool::invite::any_internet_reachable(
                            self.shared_state.listen_multiaddrs.load().as_ref(),
                        );
                        if !reachable {
                            self.try_activate_relay("no internet-reachable address after startup");
                        }
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
                    // Streaming-tracked entries (`delivery_request_id = Some`)
                    // get the much shorter `RR_ACK_TIMEOUT_SECS` window so the
                    // remote-generate fast path fails fast on libp2p rr
                    // silent-drops. On expiry we close the caller's channel
                    // so it sees Err immediately. Untracked entries (label-
                    // only) keep the existing long sweep window.
                    let now = std::time::Instant::now();
                    let mut closed_streaming: Vec<uuid::Uuid> = Vec::new();
                    self.pending_rr_observability
                        .retain(|_id, (_label, inserted, delivery_uuid)| {
                            let age = now.duration_since(*inserted).as_secs();
                            match delivery_uuid {
                                Some(uuid) => {
                                    if age >= RR_ACK_TIMEOUT_SECS {
                                        closed_streaming.push(*uuid);
                                        false
                                    } else {
                                        true
                                    }
                                }
                                None => age < MAX_TENSOR_FORWARD_SECS,
                            }
                        });
                    for uuid in closed_streaming {
                        if self.shared_state.streaming_token_txs.remove(&uuid).is_some() {
                            tracing::warn!(
                                request_id = %uuid,
                                ack_timeout_secs = RR_ACK_TIMEOUT_SECS,
                                "DIAG: rr ACK timeout — closing streaming caller (silent-drop suspected)"
                            );
                        }
                    }
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
                    // Same defence for shard-serve tickets: if the spawned
                    // task panicked or its DeliverShardResponse was dropped
                    // because internal_cmd_tx was full, the channel here
                    // would otherwise sit until the request_response timeout
                    // (30s) closes the substream — but the entry counts
                    // against MAX_INBOUND_SHARD_FETCHES until then.
                    let before_shard = self.pending_shard_responses.len();
                    self.pending_shard_responses.retain(|_ticket, (inserted, _chan)| {
                        inserted.elapsed().as_secs() < PENDING_SHARD_TICKET_TTL_SECS
                    });
                    let removed_shard = before_shard - self.pending_shard_responses.len();
                    if removed_shard > 0 {
                        tracing::warn!(
                            removed = removed_shard,
                            remaining = self.pending_shard_responses.len(),
                            "Swept stale pending_shard_responses tickets"
                        );
                    }
                    if !self.pending_tensor_outbound.is_empty() {
                        let now = std::time::Instant::now();
                        let mut stale: Vec<(OutboundRequestId, uuid::Uuid, libp2p::PeerId)> = Vec::new();
                        for (req_id, (uuid, sent_at, target_peer, num_layers, activation_bytes)) in &self.pending_tensor_outbound {
                            let age = now.duration_since(*sent_at);
                            let _ = (num_layers, activation_bytes);
                            // This sweep is a BACKSTOP for sends libp2p dropped
                            // silently — not a second, competing deadline. It
                            // must therefore never fire before the pipeline's
                            // own `SegmentBudget`, which is the deadline that
                            // knows how fast this peer is and whether it still
                            // has to load the model.
                            //
                            // It used to recompute `layers x 15s` here and a
                            // comment claimed it "matches pipeline.rs logic".
                            // That stopped being true the moment the pipeline
                            // learned to size deadlines from measured peer
                            // speed (v0.3.60): the pipeline would allow 600s
                            // while this reaped the same forward at 120s and
                            // synthesised "Tensor forward timed out", so the
                            // whole point of measuring peers was defeated for
                            // exactly the slow peers it was built for.
                            // Observed live on v0.3.61 — a 2000-token prefill
                            // over 8 layers, killed at 130s against a 600s
                            // budget, twice, and surfaced as the misleading
                            // "Tensor bytes too short".
                            //
                            // `MAX_TENSOR_FORWARD_SECS` is the request_response
                            // protocol's own timeout, so libp2p reports a real
                            // `OutboundFailure` at that point anyway; this is
                            // the correct ceiling and there is nothing to gain
                            // from guessing a tighter one.
                            let timeout_secs = MAX_TENSOR_FORWARD_SECS;
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
                            .map(|(i, (peer_id, addrs, _))| (i, *peer_id, addrs.clone()))
                            .collect();
                        // Remove in reverse order to preserve indices
                        for (i, peer_id, addrs) in ready.iter().rev() {
                            self.pending_redial.remove(*i);
                            if !self.swarm.is_connected(peer_id) {
                                // `extend_addresses_through_behaviour` lets
                                // Kademlia and the identify address book add
                                // what they know on top of our hints, so a peer
                                // whose recorded address has gone stale is still
                                // reachable. With an empty hint list it is the
                                // only source — which is the inbound-connection
                                // case that previously got no dial at all.
                                let opts = libp2p::swarm::dial_opts::DialOpts::peer_id(*peer_id)
                                    .addresses(addrs.clone())
                                    .condition(libp2p::swarm::dial_opts::PeerCondition::Disconnected)
                                    .extend_addresses_through_behaviour()
                                    .build();
                                match self.swarm.dial(opts) {
                                    Ok(()) => tracing::info!(
                                        %peer_id, addr_count = addrs.len(),
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
    /// Cached peer addresses worth dialling, with unreachable and
    /// self-referencing entries removed. Filtering on *read* (not only on
    /// write) is what repairs a cache poisoned by an older build — the write
    /// path is skipped entirely while no peers are connected, so a bad cache
    /// would otherwise outlive every restart.
    fn dialable_peer_cache(&self) -> Vec<String> {
        let cached = crate::network::peer_cache::load_peer_cache(&self.shared_state.db);
        let local_addrs = self.shared_state.listen_multiaddrs.load();
        crate::network::peer_cache::filter_dialable(
            &cached,
            self.swarm.local_peer_id(),
            &local_addrs,
        )
    }

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
        // Carry forward addresses already in the cache. `peer_registry` holds
        // only CURRENTLY connected peers — `handle_connection_closed` removes a
        // peer the moment its last connection drops — and this save replaces the
        // whole tree, so building it from the registry alone erased a peer from
        // the cache within one save interval of it going quiet. The cache exists
        // so "two machines in one house find each other again after a reboot"
        // (see `filter_storable`), and that promise does not hold if a peer has
        // to be connected at save time to stay in it.
        //
        // Connected peers go FIRST so that when `save_peer_cache` truncates at
        // MAX_CACHED_PEERS it is peers that have genuinely left that fall off,
        // not live ones. `filter_storable` still runs over the union, so a bad
        // address is evicted on the next save exactly as before — merging keeps
        // stale-but-valid entries, it does not make the cache unpurgeable.
        let merged = crate::network::peer_cache::merge_for_save(
            addrs,
            crate::network::peer_cache::load_peer_cache(&self.shared_state.db),
        );
        // Filter before persisting so the cache never grows entries the read
        // path would just discard.
        // Storable, not dialable: keep a peer's LAN addresses even when this
        // node currently has no use for them. Whether they are worth dialling
        // is a question about where we are now, and that is asked on read.
        let addrs =
            crate::network::peer_cache::filter_storable(&merged, self.swarm.local_peer_id());
        if !addrs.is_empty() {
            crate::network::peer_cache::save_peer_cache(&self.shared_state.db, &addrs);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_peer_id_round_trips_valid_bytes() {
        // Generate a random PeerId and verify the bytes form round-trips through the helper.
        let kp = libp2p::identity::Keypair::generate_ed25519();
        let pid = kp.public().to_peer_id();
        let bytes = pid.to_bytes();
        let resolved = NetworkManager::resolve_peer_id(&bytes, "test")
            .expect("valid PeerId bytes should resolve");
        assert_eq!(resolved, pid);
    }

    #[test]
    fn resolve_peer_id_returns_none_for_garbage() {
        // Random bytes can decode but not always — explicitly bad bytes (too short)
        // must return None without panicking.
        let bad: [u8; 3] = [0xff, 0x01, 0x02];
        assert!(NetworkManager::resolve_peer_id(&bad, "test").is_none());

        let empty: [u8; 0] = [];
        assert!(NetworkManager::resolve_peer_id(&empty, "test").is_none());
    }
}

#[cfg(test)]
mod listen_failure_tests {
    use super::*;

    /// libp2p's QUIC transport error can render as an empty string, which
    /// produced the bare "Failed to listen on QUIC: " a user actually saw. A
    /// message must never end at the colon.
    #[test]
    fn empty_transport_error_still_produces_a_detail() {
        #[derive(Debug)]
        struct Blank;
        impl std::fmt::Display for Blank {
            fn fmt(&self, _f: &mut std::fmt::Formatter) -> std::fmt::Result {
                Ok(())
            }
        }
        // Port 0 is never bound, so this takes the non-occupied branch.
        let msg = listen_failure_message("QUIC", "127.0.0.1", 0, &Blank);
        assert!(msg.contains("Blank"), "must fall back to Debug: {msg}");
        assert!(
            !msg.trim_end().ends_with(':'),
            "message ends at the colon: {msg}"
        );
    }

    /// When the port really is taken, say that in terms the user can act on —
    /// this is the most common way a first run fails.
    #[test]
    fn occupied_tcp_port_is_reported_as_such() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();

        let msg = listen_failure_message("TCP", "127.0.0.1", port, &"whatever");
        assert!(msg.contains("already in use"), "got {msg}");
        assert!(msg.contains("--port"), "must suggest the fix: {msg}");
        assert!(msg.contains(&port.to_string()), "must name the port: {msg}");
    }

    /// A QUIC failure must probe UDP, not TCP. Requiring both to be unbindable
    /// would stay silent here — a UDP port taken while TCP is free is exactly
    /// how the live failure presented.
    #[test]
    fn occupied_udp_port_is_reported_for_quic() {
        let sock = std::net::UdpSocket::bind("127.0.0.1:0").unwrap();
        let port = sock.local_addr().unwrap().port();
        // TCP on the same number is deliberately left free.
        let msg = listen_failure_message("QUIC", "127.0.0.1", port, &"whatever");
        assert!(msg.contains("already in use"), "got {msg}");
        assert!(msg.contains(&port.to_string()), "got {msg}");
    }

    /// A free port keeps the plain message — no misleading "in use" claim.
    #[test]
    fn free_port_keeps_the_plain_message() {
        let msg = listen_failure_message("QUIC", "127.0.0.1", 0, &"some cause");
        assert!(msg.contains("some cause"));
        assert!(!msg.contains("already in use"), "got {msg}");
    }
}
