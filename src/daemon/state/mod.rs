use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64};
use std::sync::Arc;

use dashmap::DashMap;
use tokio::sync::{broadcast, mpsc, watch, RwLock};

use crate::config::Config;
use crate::identity::Identity;
use crate::inference::executor::SharedExecutor;
use crate::model::registry::ModelRegistry;
use crate::storage::db::Database;
use crate::types::{NodeId, NodeStats, PeerInfo, PipelineAssignment};

use super::helpers::resolve_api_key;

mod activity;
mod capacity;
mod capacity_plan;
mod credits;
mod events;
mod hf;
mod metrics;
mod models;
mod perf_history;
mod relay;
mod tp_allreduce;

pub use activity::{ActivityEvent, DashboardSignal, LoadedModelInfo};
pub use capacity::{
    compute_swarm_capacity, refresh_swarm_capacity, HeadlineModel, ModelEntry, SwarmCapacity,
};
pub use capacity_plan::{compute_capacity_plan, CapacityPlan, CapacityScenario, HeadlineTarget};
pub use credits::CreditPool;
pub use events::EventBus;
pub use hf::{HfProbeInfo, HfSource};
pub use metrics::{ChannelCounters, ChannelMetricsSet, MetricsProviders};
pub use models::{ModelMgmt, FOREIGN_WISHLIST_MAX_AGE_MS, MAX_FOREIGN_WISHLIST_ENTRIES};
pub use relay::{RelayForwardCounter, RelayProvenFeatures, RelayRoute};
pub use tp_allreduce::TpAllReduceCollector;

// ---- Main SharedState ----

/// How many recent inference failures to retain for diagnostics. Enough to show
/// a pattern (one flaky peer, one bad model) without turning into a log.
pub const MAX_RECENT_FAILURES: usize = 20;

/// Completed traces kept for `GET /api/admin/diagnostics`. Larger than the
/// failure ring because successful requests are the baseline you compare a
/// slow one against, and they are cheap (no strings beyond node ids).
pub const MAX_RECENT_TRACES: usize = 50;

/// One failed inference, retained for `GET /api/admin/diagnostics`.
#[derive(Debug, Clone)]
pub struct RequestFailure {
    pub at: chrono::DateTime<chrono::Utc>,
    pub request_id: String,
    pub model: String,
    /// Where it ran: `None` for local, else the peer that served the first
    /// segment. This is the field that distinguishes "this node is broken" from
    /// "one peer is broken", which is the distinction we have repeatedly had to
    /// reconstruct by hand from two sides' logs.
    pub served_by: Option<String>,
    pub error: String,
    pub elapsed_ms: u64,
}

/// One peer's serving performance, joined from the health-ping RTT, the
/// per-layer EMA and the hedge tracker's EWMA. Rendered by
/// `GET /api/admin/diagnostics` and the swarm dashboard.
#[derive(Clone, Debug, serde::Serialize)]
pub struct PeerPerformanceRow {
    pub node_id: String,
    /// Health-ping round trip.
    pub rtt_ms: Option<u32>,
    /// Observed per-layer forward cost — comparable across peers serving
    /// differently-sized segments, unlike raw segment time.
    pub ms_per_layer: Option<f32>,
    /// Sample-weighted EWMA of segment latency across all (model, segment)
    /// pairs this peer served for us.
    pub ewma_ms: Option<f32>,
    /// Observations behind `ewma_ms`. A one-sample average is not evidence.
    pub samples: u32,
    pub region: Option<String>,
}

/// What a node is currently doing on behalf of other peers, per model.
#[derive(Debug, Clone)]
pub struct ServingState {
    /// Requests from peers currently being computed for this model.
    pub in_flight: u32,
    /// When this node last finished serving this model to a peer.
    pub last_served_at: std::time::Instant,
}

/// RAII marker for one peer-served request. Increments the model's in-flight
/// count on construction and decrements on drop, so a serving path that panics
/// or is cancelled cannot leave the model looking permanently busy.
pub struct ServingGuard {
    state: std::sync::Arc<SharedState>,
    model_id: crate::types::ModelId,
}

impl ServingGuard {
    pub fn new(state: &std::sync::Arc<SharedState>, model_id: crate::types::ModelId) -> Self {
        state
            .serving_models
            .entry(model_id.clone())
            .and_modify(|s| {
                s.in_flight = s.in_flight.saturating_add(1);
                s.last_served_at = std::time::Instant::now();
            })
            .or_insert_with(|| ServingState {
                in_flight: 1,
                last_served_at: std::time::Instant::now(),
            });
        Self {
            state: state.clone(),
            model_id,
        }
    }
}

impl Drop for ServingGuard {
    fn drop(&mut self) {
        if let Some(mut e) = self.state.serving_models.get_mut(&self.model_id) {
            e.in_flight = e.in_flight.saturating_sub(1);
            // Stamped on the way out too: the useful question for idleness is
            // "when did work last finish", not "when did it last start".
            e.last_served_at = std::time::Instant::now();
        }
    }
}

pub struct SharedState {
    // Core infrastructure (accessed by nearly every subsystem)
    pub config: Config,
    pub identity: Identity,
    pub db: Database,
    pub api_key: String,
    pub internal_auth_token: String,
    pub is_ready: AtomicBool,
    pub config_watch_tx: watch::Sender<crate::config::OperationalParams>,

    // Cross-cutting registries (too widely accessed to sub-struct)
    pub peer_registry: DashMap<NodeId, PeerInfo>,
    pub model_registry: ModelRegistry,
    pub nickname_registry: DashMap<NodeId, crate::identity::nickname::NicknameRecord>,
    pub peer_id_map: DashMap<NodeId, Vec<u8>>,
    /// NodeIds currently connected at the libp2p transport layer.
    /// Populated by NetworkManager on Identify-Received, removed on ConnectionClosed
    /// when num_established transitions to 0. HealthMonitor uses this as ground truth
    /// to avoid evicting peer_registry entries for peers that are still connected but
    /// momentarily silent (e.g. slow WSL2 QUIC substream negotiation, no recent gossip).
    pub connected_node_ids: dashmap::DashSet<NodeId>,
    /// NETWORKING_PLAN Phase 1 — learned reverse relay routes, keyed by the
    /// TARGET's libp2p peer-id bytes. When an inference message for us arrives
    /// wrapped in a `RelayedEnvelope` via relay R, we record "to reach the
    /// envelope's `origin`, send through R". The directed-send path in
    /// `network/manager/commands.rs` consults this whenever a target peer is not
    /// directly connected, so replies and subsequent turns flow back the same
    /// way without any routing decision leaking into daemon code. Bounded by
    /// distinct target peers; entries expire after `RELAY_ROUTE_TTL_SECS` and
    /// are swept on the HealthMonitor tick via `sweep_stale_relay_state`.
    pub relay_routes: DashMap<Vec<u8>, RelayRoute>,
    /// NETWORKING_PLAN Phase 1 — per-origin relay-forward rate counters (relay
    /// side only), keyed by the origin's peer-id bytes. Bounds how fast one peer
    /// can push traffic through us as a relay so the role can't be abused to
    /// exhaust our uplink. Swept alongside `relay_routes`.
    pub relay_forward_counters: DashMap<Vec<u8>, RelayForwardCounter>,
    /// NETWORKING_PLAN — relay features a peer has demonstrably used by relaying
    /// to us (keyed by NodeId). Direct proof that sidesteps the capability-gossip
    /// cold-start window on the relay send path's feature gates, so a computed
    /// result is never refused a return relay to a coordinator that just relayed
    /// a forward to us. Bounded by distinct relay peers; swept on the
    /// HealthMonitor tick alongside `relay_routes` (`RELAY_ROUTE_TTL_SECS`).
    pub relay_proven_features: DashMap<NodeId, RelayProvenFeatures>,

    // Inference engine
    pub executor: SharedExecutor,
    pub draft_executor: SharedExecutor,
    pub loaded_model_info: RwLock<Option<LoadedModelInfo>>,
    pub gpu_info: Option<crate::inference::executor::GpuInfo>,
    pub model_loaded: std::sync::atomic::AtomicBool,
    pub active_pipelines: DashMap<uuid::Uuid, PipelineAssignment>,
    /// Abort handles for in-flight remote-generate decodes serving requests
    /// from peers. Keyed by `request_id`; the value carries the AbortHandle and
    /// the originator's peer bytes. When the originator broadcasts a
    /// `SwarmMessage::CancelInference`, OR when the originator's connection
    /// drops (a NAT'd coordinator can't be re-dialed to receive tokens, so the
    /// decode is pure waste — external report 2026-07-23), the matching task is
    /// aborted so the worker stops streaming tokens nobody will receive. Cleared
    /// when the decode finishes naturally.
    pub inbound_generate_aborts: DashMap<uuid::Uuid, (tokio::task::AbortHandle, Vec<u8>)>,
    /// Per-cancel-token cancel signals. The HTTP entry for `chat_completions`
    /// looks up an `Arc<AtomicBool>` by token (passed via the
    /// `x-swarmllm-cancel-token` header) and attaches it to the
    /// `InferenceRequest`. The pipeline executor reads `request.is_cancelled()`
    /// per token and stops the decode loop when tripped. Background paths
    /// (e.g. `/v1/responses/{id}/cancel`) flip the bool to interrupt
    /// in-flight inference. Cleared on request completion.
    pub cancel_signals: DashMap<String, Arc<std::sync::atomic::AtomicBool>>,
    pub split_models:
        DashMap<crate::inference::split::SplitModelKey, crate::inference::split::SplitModelEntry>,
    /// Secondary index: model_id → loaded segment ranges for O(1) lookup by model.
    pub split_model_index: DashMap<crate::types::ModelId, Vec<(usize, usize)>>,
    pub kv_cache_store: Arc<crate::inference::split::KvCacheStore>,
    pub gguf_meta: DashMap<crate::types::ModelId, crate::inference::split::GgufTensorMeta>,
    /// Distributed streaming token routing (pipeline_id → sender).
    /// Consumer: dispatch handler + health monitor cleanup. Producer: pipeline.rs.
    pub streaming_token_txs: DashMap<uuid::Uuid, mpsc::Sender<crate::types::StreamingToken>>,
    pub pending_layer_results:
        DashMap<uuid::Uuid, tokio::sync::oneshot::Sender<crate::types::LayerResult>>,
    /// R139 Tier 4K — receiver-side assembly state for STREAM-chunked
    /// activation forwards. Keyed by `request_id`. Each chunk arriving on
    /// the wire (LayerForward with `chunk_meta = Some(_)`) gets inserted at
    /// its `chunk_idx` slot; once all `total_chunks` slots are filled, the
    /// dispatch path concatenates and forwards a single reassembled
    /// LayerForward to the worker. The 0x05 trailer is bound into the AAD
    /// via `build_layer_forward_aad`, so reorder, truncation, and
    /// cross-transfer-substitution attempts fail Poly1305 before reaching
    /// this state. A periodic TTL sweep evicts incomplete assemblies older
    /// than `STREAM_CHUNK_ASSEMBLY_TTL_SECS` so a stuck/abandoned sender
    /// cannot leak memory.
    pub pending_activation_chunks: DashMap<uuid::Uuid, crate::types::ChunkAssemblyState>,
    /// Remote-side (segment holder) pending result routes for forwards received
    /// on a persistent pipeline stream. Keyed by request_id; the handler task
    /// registers a oneshot before dispatch, and NetworkManager delivers the
    /// result here (instead of the request_response path) when a match is found.
    pub pending_stream_result_routes:
        DashMap<uuid::Uuid, tokio::sync::oneshot::Sender<crate::types::LayerResult>>,
    /// Item 8 Phase 2: in-flight cross-node prefix KV fetches keyed by the
    /// fetcher's `request_id`. Daemon caller installs the oneshot BEFORE
    /// sending `NetworkCommand::SendPrefixKvFetch`; NetworkManager resolves
    /// it with `Some(bytes)` on hit or `None` on miss/failure. RAII guard on
    /// the caller side removes the entry on drop so a cancelled fetch
    /// doesn't leak.
    pub pending_prefix_kv_fetches:
        DashMap<uuid::Uuid, tokio::sync::oneshot::Sender<Option<Vec<u8>>>>,
    pub pending_vision_results: DashMap<
        uuid::Uuid,
        (
            crate::types::NodeId,
            tokio::sync::oneshot::Sender<crate::types::VisionEncodeResponse>,
        ),
    >,
    pub pending_tp_partials: DashMap<(uuid::Uuid, u32), TpAllReduceCollector>,
    pub allreduce_registry: Arc<crate::inference::allreduce::AllReduceRegistry>,
    pub ring_chunk_registry: Arc<crate::inference::allreduce::RingChunkRegistry>,
    pub vision_modules: DashMap<crate::types::ModelId, Arc<crate::inference::vision::VisionModule>>,
    pub encrypted_pipeline_models: DashMap<crate::types::ModelId, bool>,
    pub local_embedders:
        DashMap<crate::types::ModelId, Arc<crate::inference::local_embedder::LocalEmbedder>>,
    /// R136 Layer 1 / Layer 3 follow-on: lightweight tokenizer cache
    /// keyed by ModelId. Lazily populated from the model's
    /// `gguf_header.bin` (no shard files needed). Unlocks coordinator-
    /// side tokenization without loading the full model — required for:
    /// (1) n-gram-only spec path (no draft model needed), (2) Layer 3
    /// first-token observation. Loaded on first request per model that
    /// needs it; cached for subsequent reuse. Sized: typically a few
    /// hundred KB per model (vocab + merges).
    pub standalone_tokenizers:
        DashMap<crate::types::ModelId, Arc<crate::inference::split::SplitTokenizer>>,
    pub adapter_registry: Arc<crate::model::lora::AdapterRegistry>,
    pub model_process_pool: Arc<crate::inference::process_pool::ModelProcessPool>,
    // Network & crypto
    pub session_manager: Arc<crate::crypto::SessionManager>,
    pub gossip_sealer: Arc<crate::crypto::GossipSealer>,
    pub lan_peer_count: std::sync::atomic::AtomicUsize,
    /// Live snapshot of the swarm's current listen multiaddrs, each terminated
    /// with `/p2p/<local_peer_id>` so they can be dialed directly. Updated by
    /// NetworkManager on `NewListenAddr` / `ExpiredListenAddr` events. Read by
    /// the pool invite-code generator so a freshly-minted code carries every
    /// address a remote peer might reach this node on.
    pub listen_multiaddrs: arc_swap::ArcSwap<Vec<String>>,
    /// Runtime mirror of `config.api.dashboard_trust_lan` — whether a browser
    /// on a private/LAN address is handed the dashboard's API key.
    ///
    /// `state.config` is startup-frozen, and this is a setting the user flips
    /// *because* their dashboard is currently unreachable. Requiring a daemon
    /// restart to apply it would mean the fix for "I can't reach this node"
    /// is "go to the machine you can't reach and restart it". Written by
    /// `PUT /api/admin/config`, read by `api::dashboard_trust::classify` on
    /// the next page load. New code that gates on LAN dashboard trust MUST
    /// read this atomic, not `state.config.api.dashboard_trust_lan`.
    pub dashboard_trust_lan: std::sync::atomic::AtomicBool,
    /// NETWORKING_PLAN Phase 3 — whether this node has been observed to be
    /// reachable from the open internet, and may therefore donate itself as an
    /// application-level inference relay.
    ///
    /// `state.config` is startup-frozen, but public reachability is only learned
    /// at runtime (UPnP mapping, AutoNAT confirmation, a manually declared
    /// external address), so this atomic is the runtime signal. Written by
    /// `NetworkManager::refresh_listen_multiaddrs` alongside `listen_multiaddrs`;
    /// read via `SharedState::relay_forwarding_enabled()`.
    ///
    /// Deliberately NOT derived from `listen_multiaddrs` at read time: that list
    /// counts a `/p2p-circuit` address as reachable (correct for invite codes),
    /// but a node reachable only *through* a relay is itself NAT'd and must
    /// never advertise itself as one.
    pub publicly_reachable: std::sync::atomic::AtomicBool,
    /// DCUtR hole-punch outcomes since start. Surfaced by
    /// `GET /api/admin/diagnostics` so "did this node ever get off the relay?"
    /// is answerable without scraping logs — the question that matters most
    /// when a user reports slow or failing remote inference.
    pub hole_punch_successes: AtomicU64,
    pub hole_punch_failures: AtomicU64,
    /// The most recent inference failures, newest last, capped at
    /// [`MAX_RECENT_FAILURES`].
    ///
    /// Exists so "why did my request fail?" is answerable from a single
    /// diagnostics paste. Previously it required the user to have been running
    /// with `-v`, reproduce the failure, and send logs — a multi-round exchange
    /// that usually lost the original occurrence. A bounded in-memory ring costs
    /// nothing and turns most reports into one command.
    pub recent_failures: std::sync::Mutex<std::collections::VecDeque<RequestFailure>>,
    /// Completed request traces — route, timings and per-segment attribution.
    /// The successful sibling of `recent_failures`: together they answer both
    /// "why did that break" and "why was that slow" without a log excerpt.
    pub recent_traces:
        std::sync::Mutex<std::collections::VecDeque<crate::inference::trace::TraceSnapshot>>,
    /// In-flight traces, keyed by request id.
    ///
    /// Deep pipeline code (`pipeline/distributed.rs`) knows a segment's elapsed
    /// time but has no trace handle, and threading one through every executor
    /// signature would be invasive. This map has **exactly the same lifetime as
    /// `active_pipelines`** — inserted and removed at the same sites — so it
    /// inherits that mechanism's already-correct cleanup, including the panic
    /// path via `ActivePipelineGuard::drop`.
    pub active_traces: DashMap<uuid::Uuid, std::sync::Arc<crate::inference::trace::RequestTrace>>,
    /// Per-model record of work this node is doing **for other peers**:
    /// `(requests currently in flight, when we last served one)`.
    ///
    /// **`active_pipelines` does not cover this.** That map is the
    /// *coordinator's* view — it holds pipelines this node assembled and
    /// originated. A node answering a peer's `RemoteGenerateRequest` or
    /// `LayerForward` never appears in it, so anything that consults only
    /// `active_pipelines` believes a pure-server node is doing nothing.
    ///
    /// That is not hypothetical: the idle-VRAM unload's active-pipeline guard
    /// consulted only `active_pipelines`, so on 2026-07-28 a soak caught it
    /// killing a worker with two peer requests mid-generate — the node had
    /// served nothing of its own, so it looked idle for 12x the window and the
    /// hard-unload ceiling fired. `record_request` has the same outbound-only
    /// blind spot (v0.3.38), which is why "regional demand" was standing in for
    /// it; this map is the real signal that proxy was approximating.
    ///
    /// Anything asking "is this model in use?" MUST consult this as well as
    /// `active_pipelines`, or it is only asking about half the node's work.
    pub serving_models: DashMap<crate::types::ModelId, ServingState>,
    /// The most recent `NodeCapability` this node broadcast about itself.
    ///
    /// The HealthMonitor builds one every tick and sends it straight to the
    /// network, so nothing local could answer "what do we advertise about
    /// ourselves?" — the leaderboard could describe every peer's hardware but
    /// not our own. Retained here purely so this node can be rendered on the
    /// same footing as everyone else.
    pub local_capability: arc_swap::ArcSwapOption<crate::types::NodeCapability>,
    /// Hourly performance rollups, persisted so a trend survives a restart.
    /// Aggregates only — per-request detail stays in `recent_traces`, in memory.
    pub perf_history: perf_history::PerfHistory,
    /// Holders this request must not be routed to again, keyed by request id.
    ///
    /// Retracting a holder's claim locally is not enough on its own: the DHT
    /// still advertises it as a provider, so the very next assembly — especially
    /// one that waits for DHT results — re-learns the claim and picks the same
    /// dead holder. Observed live 2026-07-26: retraction fired correctly, the
    /// retry re-added the holder from the DHT and failed identically.
    ///
    /// Scoped to one request so a peer that fails once is not banned globally on
    /// a single data point. Same lifetime as `active_traces`.
    pub request_holder_blacklist: DashMap<uuid::Uuid, std::collections::HashSet<NodeId>>,
    pub detected_region: RwLock<Option<String>>,
    pub shard_bytes_served: AtomicU64,
    pub relay_seconds_served: AtomicU64,
    /// NETWORKING_PLAN Phase 3 — bytes this node has forwarded as an
    /// application-level inference relay (`RelayedEnvelope`). Drained by the
    /// CreditLedger tick and converted to credit at the same byte-rate as shard
    /// seeding, so donating relay capacity earns like serving does — making
    /// public relay supply economically self-sustaining (incentive-aligned).
    pub relay_inference_bytes: AtomicU64,
    pub active_relay_circuits: DashMap<(libp2p::PeerId, libp2p::PeerId), std::time::Instant>,
    pub region_shard_summaries:
        DashMap<(String, crate::types::ModelId), crate::types::RegionShardSummary>,
    pub region_demand: DashMap<(crate::types::ModelId, String), f64>,
    pub dht_query_tx: mpsc::Sender<crate::types::ModelId>,

    /// Persistent pipeline-stream client handle, installed by NetworkManager at
    /// startup when `config.inference.persistent_pipeline_stream` is enabled.
    /// When absent, distributed forwards fall back to the request_response path.
    pub pipeline_stream_client:
        tokio::sync::OnceCell<Arc<crate::network::pipeline_stream::PipelineStreamClient>>,

    // Sub-structs (logically grouped fields)
    pub events: EventBus,
    pub credits: CreditPool,
    pub models: ModelMgmt,
    pub metrics: MetricsProviders,

    shutdown_tx: watch::Sender<bool>,
}

impl SharedState {
    pub fn new(
        config: Config,
        identity: Identity,
        db: Database,
        executor: SharedExecutor,
        gpu_info: Option<crate::inference::executor::GpuInfo>,
    ) -> (
        Arc<Self>,
        watch::Receiver<bool>,
        mpsc::Receiver<crate::types::ModelId>,
    ) {
        // Resolve API key: config > persisted in DB > generate new
        let api_key = resolve_api_key(&config, &db);
        let (shutdown_tx, shutdown_rx) = watch::channel(false);
        // S5: DHT shard provider query channel — subsystems send ModelId,
        // NetworkManager issues kademlia.get_providers() and merges results.
        let (dht_query_tx, dht_query_rx) = mpsc::channel::<crate::types::ModelId>(64);

        let mut model_registry = ModelRegistry::load_from_db(&db).unwrap_or_default();
        model_registry.set_local_node_id(identity.node_id().clone());

        // Hydrate nickname registry from DB
        let nickname_registry = DashMap::new();
        let nick_store = crate::identity::nickname::NicknameStore::new(db.clone());
        if let Ok(records) = nick_store.load_all() {
            for record in records {
                nickname_registry.insert(record.node_id.clone(), record);
            }
        }

        let node_id = identity.node_id().clone();

        // Initialize E2E encryption subsystem
        let session_manager = Arc::new(crate::crypto::SessionManager::from_ed25519_key(
            &identity.signing_key_bytes(),
        ));
        // Use a fixed network ID so all public nodes share gossip encryption keys.
        // Private networks can override via config: gossip_network_id = "my-private-net"
        let gossip_network_id = config
            .network
            .gossip_network_id
            .as_deref()
            .unwrap_or("swarmllm-mainnet-v1")
            .as_bytes()
            .to_vec();
        let gossip_sealer = Arc::new(crate::crypto::GossipSealer::new(&gossip_network_id));

        // Load persisted HF sources from database
        let hf_sources = {
            let map = DashMap::new();
            if let Ok(entries) = db.iter_raw("hf_sources") {
                for (key, value) in entries {
                    if let (Ok(model_id_str), Ok(source)) = (
                        std::str::from_utf8(&key),
                        serde_json::from_slice::<HfSource>(&value),
                    ) {
                        map.insert(crate::types::ModelId(model_id_str.to_string()), source);
                    }
                }
            }
            map
        };

        let auto_manage_enabled = config.auto_manage.enabled;
        let default_model_shard_cap = config.auto_manage.default_model_shard_cap;
        let kv_cache_ttl_secs = config.inference.kv_cache_ttl_secs.unwrap_or(600);
        let initial_ops = crate::config::OperationalParams::from_config(&config);
        let (config_watch_tx, _) = watch::channel(initial_ops);
        let trust_manager = crate::credit::trust::TrustManager::new(db.clone());
        let hydrated = trust_manager.hydrate_cache();
        if hydrated > 0 {
            tracing::debug!(
                trust_entries = hydrated,
                "DIAG: TrustManager cache hydrated from db"
            );
        }
        let escrow_manager = Arc::new(crate::credit::escrow::EscrowManager::new(
            db.clone(),
            crate::credit::escrow::DEFAULT_ESCROW_THRESHOLD,
        ));

        // Hydrate per-model auto-manage policies from database + config
        let model_auto_manage_policies = {
            let map = DashMap::new();
            if let Ok(entries) = db.iter_raw("model_auto_manage_policies") {
                for (key, value) in entries {
                    if let (Ok(model_id_str), Ok(policy)) = (
                        std::str::from_utf8(&key),
                        serde_json::from_slice::<crate::config::ModelAutoManagePolicy>(&value),
                    ) {
                        map.insert(crate::types::ModelId(model_id_str.to_string()), policy);
                    }
                }
            }
            for (model_id, policy) in &config.auto_manage.model_policies {
                map.entry(crate::types::ModelId(model_id.clone()))
                    .or_insert_with(|| policy.clone());
            }
            map
        };
        // Grab signing key bytes before identity is moved into the struct.
        // SEC: wrap in `Zeroizing` so the local copy is scrubbed on drop —
        // this lives across the whole `Arc::new(Self { ... })` block and
        // through `decrypt_config` below.
        let signing_key_bytes: zeroize::Zeroizing<[u8; 32]> =
            zeroize::Zeroizing::new(identity.signing_key_bytes());
        let state = Arc::new(Self {
            config: config.clone(),
            identity,
            db: db.clone(),
            peer_registry: DashMap::new(),
            model_registry,
            active_pipelines: DashMap::new(),
            inbound_generate_aborts: DashMap::new(),
            cancel_signals: DashMap::new(),
            metrics: MetricsProviders {
                node_stats: RwLock::new(NodeStats::default()),
                inference_requests_total: AtomicU64::new(0),
                requests_served_atomic: AtomicU64::new(0),
                forwards_served_atomic: AtomicU64::new(0),
                inference_latency_samples: std::sync::RwLock::new(std::collections::VecDeque::new()),
                inference_latency_total_count: AtomicU64::new(0),
                inference_latency_total_micros: AtomicU64::new(0),
                ttft_samples: std::sync::RwLock::new(std::collections::VecDeque::new()),
                ttft_total_count: AtomicU64::new(0),
                ttft_total_micros: AtomicU64::new(0),
                tpot_samples: std::sync::RwLock::new(std::collections::VecDeque::new()),
                tpot_total_count: AtomicU64::new(0),
                tpot_total_micros: AtomicU64::new(0),
                requests_by_route: DashMap::new(),
                segments_served: AtomicU64::new(0),
                layers_served: AtomicU64::new(0),
                segment_serve_micros: AtomicU64::new(0),
                segment_bytes_out: AtomicU64::new(0),
                channel_metrics: ChannelMetricsSet::new(),
                ws_connection_count: std::sync::atomic::AtomicUsize::new(0),
                providers_config: RwLock::new({
                    let stored = db
                        .get_json::<crate::config::ProvidersConfig>("providers", "config")
                        .ok()
                        .flatten();
                    let mut pc = match stored {
                        Some(cfg) => crate::crypto::decrypt_config(&cfg, &signing_key_bytes)
                            .unwrap_or_else(|e| {
                                tracing::warn!(
                                    error = %e,
                                    "Failed to decrypt stored provider keys, using config file"
                                );
                                config.providers.clone()
                            }),
                        None => config.providers.clone(),
                    };
                    pc.fill_from_env();
                    pc
                }),
                provider_model_map: DashMap::new(),
                provider_models_cache: RwLock::new((Vec::new(), std::time::Instant::now())),
                stats_cache: parking_lot::Mutex::new(None),
                stats_building: std::sync::atomic::AtomicBool::new(false),
                peer_segment_latency_ms_per_layer: DashMap::new(),
                swarm_capacity: arc_swap::ArcSwap::from_pointee(SwarmCapacity::default()),
                hedge_tracker: Arc::new(crate::inference::hedging::HedgeTracker::new()),
                prefetch_orchestrator: Arc::new(
                    crate::inference::prefetch::PrefetchOrchestrator::new(),
                ),
                ngram_hits: std::sync::atomic::AtomicU64::new(0),
                ngram_misses: std::sync::atomic::AtomicU64::new(0),
            },
            credits: CreditPool {
                credit_balance: Arc::new(RwLock::new(crate::types::CreditBalance {
                    node_id,
                    balance: 0,
                    lifetime_earned: 0,
                    lifetime_spent: 0,
                    lifetime_refunded: 0,
                    last_updated: chrono::Utc::now(),
                })),
                pending_credit_earn: std::sync::atomic::AtomicI64::new(0),
                pool_state: RwLock::new(None),
                pool_registry: DashMap::new(),
                pool_tx: RwLock::new(None),
                pool_credit_rates: DashMap::new(),
                trust_manager,
                escrow_manager,
                anti_gaming: tokio::sync::Mutex::new(crate::credit::anti_gaming::AntiGaming::new()),
                peer_credit_balances: DashMap::new(),
                credit_percentile_cache: parking_lot::Mutex::new((std::time::Instant::now(), 0.5)),
                private_mode: std::sync::atomic::AtomicBool::new({
                    // Restore from DB (with R138 legacy-path migration handled
                    // inside `restore_node_mode`), fall back to config default.
                    crate::pool::manager::restore_node_mode(
                        &db,
                        crate::pool::manager::KEY_PRIVATE_MODE,
                    )
                    .unwrap_or(config.pool.private_mode)
                }),
                offline_mode: std::sync::atomic::AtomicBool::new({
                    crate::pool::manager::restore_node_mode(
                        &db,
                        crate::pool::manager::KEY_OFFLINE_MODE,
                    )
                    .unwrap_or(config.pool.offline_mode)
                }),
                foreign_pool_catalog: DashMap::new(),
                allow_cross_pool_inference: std::sync::atomic::AtomicBool::new(
                    config.pool.allow_cross_pool_inference,
                ),
                share_model_catalog: std::sync::atomic::AtomicBool::new(
                    config.pool.share_model_catalog,
                ),
            },
            models: ModelMgmt {
                acquisition_progress: DashMap::new(),
                hf_sources,
                auto_manage_notify: Arc::new(tokio::sync::Notify::new()),
                auto_manage_enabled: std::sync::atomic::AtomicBool::new(auto_manage_enabled),
                contribution_auto: std::sync::atomic::AtomicBool::new(
                    config.node.contribution_auto,
                ),
                auto_manage_default_model_cap: AtomicU32::new(default_model_shard_cap),
                model_auto_manage_policies,
                hf_probe_cache: DashMap::new(),
                peer_shard_downloads: DashMap::new(),
                download_cancel_flags: DashMap::new(),
                model_trust: {
                    let map = DashMap::new();
                    if let Ok(pairs) =
                        db.get_all_json::<crate::types::ModelTrustInfo>("model_trust")
                    {
                        for (key, info) in pairs {
                            map.insert(crate::types::ModelId(key), info);
                        }
                    }
                    map
                },
                loading_models: DashMap::new(),
                locked_shards: {
                    let map = DashMap::new();
                    if let Ok(entries) = db.iter_raw("locked_shards") {
                        for (key, _value) in entries {
                            if let Ok(key_str) = std::str::from_utf8(&key) {
                                if let Ok(shard_id) =
                                    serde_json::from_str::<crate::types::ShardId>(key_str)
                                {
                                    map.insert(shard_id, true);
                                }
                            }
                        }
                    }
                    map
                },
                model_request_counts: DashMap::new(),
                resource_schedule: RwLock::new(config.resources.schedule.clone()),
                prune_history: RwLock::new(VecDeque::new()),
                shard_p2p_failed: dashmap::DashSet::new(),
                shard_download_backoff: DashMap::new(),
                parallax_stability: DashMap::new(),
                cross_node_prefix_index: DashMap::new(),
                peer_prefix_blocks: DashMap::new(),
                p2p_download_permits: DashMap::new(),
                wishlist: arc_swap::ArcSwap::from_pointee(
                    crate::model::auto_manage::wishlist::Wishlist::default(),
                ),
                hf_trending_cache: arc_swap::ArcSwap::from_pointee(
                    crate::model::huggingface::HfTrendingSnapshot::default(),
                ),
                foreign_wishlist: DashMap::new(),
                quant_recommendations: arc_swap::ArcSwap::from_pointee(
                    crate::model::auto_manage::quant::QuantRecommendations::default(),
                ),
            },
            events: EventBus {
                dashboard_tx: broadcast::channel(32).0,
                update_state: Arc::new(RwLock::new(crate::update::UpdateState::default())),
                activity_tx: broadcast::channel(256).0,
                activity_history: parking_lot::Mutex::new(VecDeque::new()),
                ws_tickets: DashMap::new(),
            },
            // Root-level fields (not sub-structed)
            executor,
            draft_executor: Arc::new(tokio::sync::Mutex::new(
                crate::inference::executor::ModelExecutor::new(),
            )),
            loaded_model_info: RwLock::new(None),
            gpu_info,
            pending_layer_results: DashMap::new(),
            pending_activation_chunks: DashMap::new(),
            pending_stream_result_routes: DashMap::new(),
            pending_prefix_kv_fetches: DashMap::new(),
            pipeline_stream_client: tokio::sync::OnceCell::new(),
            split_models: DashMap::new(),
            split_model_index: DashMap::new(),
            kv_cache_store: Arc::new(crate::inference::split::KvCacheStore::new(
                std::time::Duration::from_secs(kv_cache_ttl_secs),
            )),
            gguf_meta: DashMap::new(),
            nickname_registry,
            peer_id_map: DashMap::new(),
            connected_node_ids: dashmap::DashSet::new(),
            relay_routes: DashMap::new(),
            relay_forward_counters: DashMap::new(),
            relay_proven_features: DashMap::new(),
            session_manager,
            gossip_sealer,
            api_key,
            internal_auth_token: {
                use rand::RngCore;
                let mut bytes = [0u8; 16];
                rand::rngs::OsRng.fill_bytes(&mut bytes);
                hex::encode(bytes)
            },
            model_loaded: std::sync::atomic::AtomicBool::new(false),
            streaming_token_txs: DashMap::new(),
            is_ready: AtomicBool::new(false),
            config_watch_tx,
            detected_region: RwLock::new(None),
            adapter_registry: Arc::new(crate::model::lora::AdapterRegistry::new(
                &config.node.data_dir,
            )),
            lan_peer_count: std::sync::atomic::AtomicUsize::new(0),
            listen_multiaddrs: arc_swap::ArcSwap::from_pointee(Vec::new()),
            dashboard_trust_lan: std::sync::atomic::AtomicBool::new(config.api.dashboard_trust_lan),
            publicly_reachable: std::sync::atomic::AtomicBool::new(false),
            hole_punch_successes: AtomicU64::new(0),
            hole_punch_failures: AtomicU64::new(0),
            recent_failures: std::sync::Mutex::new(std::collections::VecDeque::new()),
            recent_traces: std::sync::Mutex::new(std::collections::VecDeque::new()),
            active_traces: DashMap::new(),
            serving_models: DashMap::new(),
            local_capability: arc_swap::ArcSwapOption::empty(),
            perf_history: perf_history::PerfHistory::load(&db),
            request_holder_blacklist: DashMap::new(),
            vision_modules: DashMap::new(),
            encrypted_pipeline_models: {
                let map = DashMap::new();
                if let Ok(pairs) = db.get_all_json::<bool>("encrypted_pipeline_models") {
                    for (key, enabled) in pairs {
                        map.insert(crate::types::ModelId(key), enabled);
                    }
                }
                map
            },
            local_embedders: DashMap::new(),
            standalone_tokenizers: DashMap::new(),
            pending_vision_results: DashMap::new(),
            pending_tp_partials: DashMap::new(),
            allreduce_registry: Arc::new(crate::inference::allreduce::AllReduceRegistry::new()),
            ring_chunk_registry: Arc::new(crate::inference::allreduce::RingChunkRegistry::new()),
            shard_bytes_served: AtomicU64::new(0),
            relay_inference_bytes: AtomicU64::new(0),
            relay_seconds_served: AtomicU64::new(0),
            active_relay_circuits: DashMap::new(),
            model_process_pool: Arc::new(crate::inference::process_pool::ModelProcessPool::new(
                config.node.data_dir.clone(),
            )),
            region_shard_summaries: DashMap::new(),
            region_demand: DashMap::new(),
            dht_query_tx,
            shutdown_tx,
        });

        // Wire activity_tx and KV-cache TTL into the process pool
        state
            .model_process_pool
            .set_activity_tx(state.events.activity_tx.clone());
        state
            .model_process_pool
            .set_kv_cache_ttl(state.config.inference.kv_cache_ttl_secs.unwrap_or(600));
        state.model_process_pool.set_prefix_cache_config(
            state.config.inference.prefix_cache_enabled,
            state.config.inference.prefix_cache_max_entries,
            state.config.inference.prefix_cache_max_prompt_tokens,
            state.config.inference.prefix_cache_block_tokens,
            state.config.inference.prefix_cache_min_tokens,
        );
        state.model_process_pool.set_swift_config(
            state.config.inference.swift_self_speculative,
            state.config.inference.swift_calibration_tokens,
            state.config.inference.swift_gamma,
            state.config.inference.swift_skip_ratio,
        );
        state
            .model_process_pool
            .set_gpu_layers(state.config.inference.gpu_layers);
        state
            .model_process_pool
            .set_force_standard_attn(state.config.inference.force_standard_attn);
        state
            .model_process_pool
            .set_max_seq_len_override(state.config.inference.max_seq_len_override);
        state
            .model_process_pool
            .set_activation_compression(state.config.inference.activation_compression);
        // Admission budget. Same source as the split-model eviction budget, so
        // the two cannot disagree about how much GPU memory this node may use.
        if let Some(budget) = crate::model::auto_manage::compute_vram_budget(&state) {
            state.model_process_pool.set_vram_budget_mb(budget);
        }
        state
            .model_process_pool
            .set_continuous_batching(state.config.inference.continuous_batching);
        state.model_process_pool.set_batch_params(
            state.config.inference.batch_collection_ms,
            state.config.inference.max_concurrent_decode_batch,
        );
        state
            .model_process_pool
            .set_prefill_chunk_tokens(state.config.inference.prefill_chunk_tokens);
        state
            .model_process_pool
            .set_prefill_target_ms(state.config.inference.prefill_target_ms);
        state
            .model_process_pool
            .set_batched_prefill_forward(state.config.inference.batched_prefill_forward);
        state
            .model_process_pool
            .start_batch_scheduler(shutdown_rx.clone());

        (state, shutdown_rx, dht_query_rx)
    }

    /// Signal all tasks to shut down.
    pub fn shutdown(&self) {
        let _ = self.shutdown_tx.send(true);
    }

    /// Get a receiver for the shutdown watch channel.
    pub fn shutdown_rx(&self) -> watch::Receiver<bool> {
        self.shutdown_tx.subscribe()
    }

    /// Subscribe to config hot-reload notifications.
    pub fn config_watch_rx(&self) -> watch::Receiver<crate::config::OperationalParams> {
        self.config_watch_tx.subscribe()
    }

    /// Apply hot-reloaded operational params and notify subscribers.
    pub fn apply_config_reload(&self, params: crate::config::OperationalParams) {
        let _ = self.config_watch_tx.send(params);
    }

    /// Resolve the on-disk directory for a model: `<data_dir>/models/<sanitized-id>`.
    /// Preferred over `model::shard::model_dir(&self.config.node.data_dir, id)` —
    /// removes the reach-through into `config.node.data_dir` at call sites.
    pub fn model_dir(&self, model_id: &str) -> std::path::PathBuf {
        crate::model::shard::model_dir(&self.config.node.data_dir, model_id)
    }

    /// Update the observed per-layer latency EMA for a peer after a successful
    /// R139 Tier 4K — receiver-side assembly entry point for a chunked
    /// activation forward. Called from the decrypt-dispatch path
    /// (`tensors.rs::handle_tensor_payload` TENSOR_TAG_ENCRYPTED, and the
    /// `pipeline_stream.rs` reader) when a decoded LayerForward carries
    /// `chunk_meta = Some(_)`.
    ///
    /// Returns:
    /// - `Ok(Some(complete_forward))` when this chunk completes the assembly.
    ///   Caller dispatches the returned `LayerForward` through the normal
    ///   `SwarmMessage::LayerForward` path (worker IPC). The `chunk_meta`
    ///   field on the returned forward is cleared.
    /// - `Ok(None)` when the chunk was accepted but the assembly is not yet
    ///   complete. Caller should NOT dispatch.
    /// - `Err(_)` for protocol violations (mismatched `total_chunks` across
    ///   chunks of the same `request_id`, duplicate chunk_idx, etc.). Caller
    ///   should drop the forward and log.
    ///
    /// AAD-bound metadata (chunk_idx, total_chunks) was already verified by
    /// the AEAD open() step — this function only enforces the orthogonal
    /// integrity invariants that ride on top of authentication (e.g., a peer
    /// authenticating each chunk individually but with internally inconsistent
    /// counts across chunks).
    pub fn try_assemble_chunked_forward(
        &self,
        forward: crate::types::LayerForward,
        sender_peer_bytes: Vec<u8>,
    ) -> Result<Option<crate::types::LayerForward>, crate::error::SwarmError> {
        let cm = forward.chunk_meta.ok_or_else(|| {
            crate::error::SwarmError::Network("chunked assembly called without chunk_meta".into())
        })?;
        let request_id = forward.request_id;
        let chunk_idx = cm.chunk_idx as usize;
        let total_chunks = cm.total_chunks;
        // AAD-validated by decrypt; defence-in-depth re-check.
        if total_chunks == 0 || chunk_idx >= total_chunks as usize {
            return Err(crate::error::SwarmError::Network(format!(
                "Invalid chunk_meta: chunk_idx={chunk_idx}, total_chunks={total_chunks}"
            )));
        }
        // Memory cap: total_chunks × chunk_size ≤ MAX_ACTIVATION_SIZE. The
        // received chunk's `activations.len()` is already capped by
        // `decode_layer_forward(_encrypted)` ≤ MAX_ACTIVATION_SIZE / chunk;
        // here we additionally bound `total_chunks` so a peer can't allocate
        // a huge `Vec<Option<...>>` slot table even if each chunk is tiny.
        const MAX_TOTAL_CHUNKS: u32 = 4096;
        if total_chunks > MAX_TOTAL_CHUNKS {
            return Err(crate::error::SwarmError::Network(format!(
                "total_chunks={total_chunks} exceeds cap {MAX_TOTAL_CHUNKS}"
            )));
        }

        // Insert under entry-lock so two concurrent chunks for the same
        // request_id can't race on completion check.
        let mut completion: Option<crate::types::LayerForward> = None;
        let mut error: Option<crate::error::SwarmError> = None;
        self.pending_activation_chunks
            .entry(request_id)
            .and_modify(|state| {
                if state.total_chunks != total_chunks {
                    error = Some(crate::error::SwarmError::Network(format!(
                        "chunk total_chunks mismatch: stored={}, got={}",
                        state.total_chunks, total_chunks
                    )));
                    return;
                }
                if state.sender_peer_bytes != sender_peer_bytes {
                    error = Some(crate::error::SwarmError::Network(
                        "chunk sender peer changed mid-transfer".into(),
                    ));
                    return;
                }
                if state.received[chunk_idx].is_some() {
                    error = Some(crate::error::SwarmError::Network(format!(
                        "duplicate chunk_idx={chunk_idx} for request_id={request_id}"
                    )));
                    return;
                }
                state.received[chunk_idx] = Some(forward.activations.clone());
                state.filled += 1;
                state.last_update_at = std::time::Instant::now();
            })
            .or_insert_with(|| {
                // First chunk: capture the cleartext template (we'll clone
                // activations onto it after reassembly). Strip chunk_meta
                // before inserting into the slot table so the template field
                // doesn't shadow assembly state.
                let mut template = forward.clone();
                template.activations = Vec::new();
                template.chunk_meta = None;
                let mut state = crate::types::ChunkAssemblyState::new(
                    total_chunks,
                    template,
                    sender_peer_bytes.clone(),
                );
                state.received[chunk_idx] = Some(forward.activations.clone());
                state.filled = 1;
                state
            });

        if let Some(e) = error {
            return Err(e);
        }

        // Completion check + remove must be atomic — otherwise two
        // concurrent decrypt-spawn tasks for the same request_id can
        // BOTH observe `is_complete()` true (the get and remove were
        // separate locks), each assemble a copy, and double-dispatch the
        // reassembled LayerForward to the worker. `remove_if` performs
        // the check + remove under one DashMap shard lock; only the
        // winning thread observes Some, the loser sees None and
        // gracefully returns Ok(None).
        if let Some((_, state)) = self
            .pending_activation_chunks
            .remove_if(&request_id, |_, s| s.is_complete())
        {
            let assembled = state.assemble();
            let mut out = (*state.template).clone();
            out.activations = assembled;
            out.sender_peer_bytes = Some(state.sender_peer_bytes.clone());
            completion = Some(out);
        }
        Ok(completion)
    }

    /// R139 Tier 4K — TTL sweep for stale chunk assemblies. Called from the
    /// HealthMonitor periodic tick. Evicts entries whose last chunk arrived
    /// more than `ttl_secs` ago, preventing a stuck/abandoned sender from
    /// leaking `pending_activation_chunks` slots. Returns the number of
    /// assemblies evicted (for diagnostics).
    pub fn sweep_stale_chunk_assemblies(&self, ttl_secs: u64) -> usize {
        let cutoff = std::time::Instant::now()
            .checked_sub(std::time::Duration::from_secs(ttl_secs))
            .unwrap_or_else(std::time::Instant::now);
        let stale: Vec<uuid::Uuid> = self
            .pending_activation_chunks
            .iter()
            .filter_map(|entry| {
                if entry.value().last_update_at < cutoff {
                    Some(*entry.key())
                } else {
                    None
                }
            })
            .collect();
        let count = stale.len();
        for key in stale {
            self.pending_activation_chunks.remove(&key);
        }
        count
    }

    /// remote segment. `segment_ms` is the wall-clock round-trip; `layers` is
    /// the number of transformer layers this segment covered. Per-layer
    /// normalisation lets later lookups scale the cost to arbitrary widths.
    /// α = 0.3 (30% weight on the new sample) — responsive but smoothed. Skips
    /// zero-layer segments (shouldn't happen, defensive).
    pub fn record_peer_segment_latency(
        &self,
        node_id: &crate::types::NodeId,
        segment_ms: u64,
        layers: u32,
    ) {
        if layers == 0 {
            return;
        }
        let sample = segment_ms as f32 / layers as f32;
        let mut entry = self
            .metrics
            .peer_segment_latency_ms_per_layer
            .entry(node_id.clone())
            .or_insert(sample);
        // EMA: new = α·sample + (1−α)·old. Use get/set via deref to avoid entry API lock-in.
        const ALPHA: f32 = 0.3;
        *entry = ALPHA * sample + (1.0 - ALPHA) * (*entry);
    }

    /// R136 Layer 1 / Layer 3 follow-on: get or lazy-load a standalone
    /// tokenizer for `model_id` from the local `gguf_header.bin`.
    /// Returns `None` when the header file isn't on disk (either the
    /// model hasn't been registered locally OR auto-manage's catalog
    /// has the manifest but no header yet). Read-side is lock-free
    /// (DashMap shard); load happens at most once per model.
    /// Pure precedence rule behind [`SharedState::encrypted_pipeline_for`], split
    /// out so it can be tested without building a whole node.
    ///
    /// `holds_both_ends` is a closure because it costs a registry lookup and is
    /// only needed when nothing explicit has decided the question.
    #[allow(clippy::needless_pass_by_value)]
    fn resolve_encrypted_pipeline_inner(
        explicit_per_model: Option<bool>,
        global_explicit_on: bool,
        auto_enabled: bool,
        holds_both_ends: impl FnOnce() -> bool,
    ) -> bool {
        if let Some(explicit) = explicit_per_model {
            return explicit;
        }
        if global_explicit_on {
            return true;
        }
        auto_enabled && holds_both_ends()
    }

    /// Whether this node holds BOTH the first and last shard of a model — the
    /// precondition for prompt privacy (`encrypted_pipeline`), which forces the
    /// first and last pipeline segments to run locally so no peer ever sees the
    /// prompt or the sampled tokens.
    pub fn holds_both_model_ends(&self, model_id: &crate::types::ModelId) -> bool {
        let Some(manifest) = self.model_registry.get_manifest(model_id) else {
            return false;
        };
        let me = self.identity.node_id();
        let holds = |index: u32| {
            self.model_registry
                .shard_holders(&crate::types::ShardId {
                    model_id: model_id.clone(),
                    index,
                })
                .contains(me)
        };
        // A single-shard model has one piece that is both ends at once.
        holds(0) && holds(manifest.shard_count.saturating_sub(1))
    }

    /// Effective prompt-privacy setting for a model.
    ///
    /// Precedence, most explicit first:
    /// 1. A per-model choice the user made — always respected, including OFF.
    /// 2. An explicit global `encrypted_pipeline = true`.
    /// 3. Otherwise ON when this node holds both ends of the model, unless
    ///    `encrypted_pipeline_auto` has been turned off.
    ///
    /// Step 3 exists because prompt privacy is the only thing stopping the
    /// machine that answers you from reading your prompt, and it was previously
    /// off unless a user knew to look for it. It is safe to default on ONLY where
    /// both ends are already local, because otherwise the pipeline has no legal
    /// route and the request fails outright.
    ///
    /// **This is the single answer to "is prompt privacy on for this model".**
    /// The scheduler and the admin API both read it here; computing it separately
    /// is how they would drift.
    pub fn encrypted_pipeline_for(&self, model_id: &crate::types::ModelId) -> bool {
        Self::resolve_encrypted_pipeline_inner(
            self.encrypted_pipeline_models
                .get(model_id)
                .map(|r| *r.value()),
            self.config.inference.encrypted_pipeline,
            self.config.inference.encrypted_pipeline_auto,
            // Only consulted when nothing explicit decides it, so the registry
            // lookup is skipped in the common explicit cases.
            || self.holds_both_model_ends(model_id),
        )
    }

    pub fn standalone_tokenizer(
        &self,
        model_id: &crate::types::ModelId,
    ) -> Option<Arc<crate::inference::split::SplitTokenizer>> {
        if let Some(t) = self.standalone_tokenizers.get(model_id) {
            return Some(t.value().clone());
        }
        let header_path = self
            .model_dir(&model_id.0)
            .join(crate::model::shard::HEADER_FILENAME);
        if !header_path.exists() {
            return None;
        }
        let meta = match crate::inference::split::GgufTokenizerMeta::from_gguf_file(&header_path) {
            Ok(m) => m,
            Err(e) => {
                tracing::debug!(
                    model_id = %model_id.0,
                    error = %e,
                    "standalone_tokenizer: failed to read gguf header"
                );
                return None;
            }
        };
        let tokenizer = meta.build_tokenizer()?;
        let arc = Arc::new(tokenizer);
        self.standalone_tokenizers
            .insert(model_id.clone(), arc.clone());
        tracing::info!(
            model_id = %model_id.0,
            "standalone_tokenizer: loaded from gguf_header.bin"
        );
        Some(arc)
    }

    /// SWARM-SPEC Layer 2: record a successful forward observation
    /// against the hedge tracker. Keyed on (model, segment, holder)
    /// rather than just holder because different models/segments have
    /// very different latency profiles on the same physical peer.
    /// `latency_ms` is the wall-clock time the forward took
    /// end-to-end (not per-layer).
    ///
    /// Also performs a post-hoc "would have hedged" dry-run check: if
    /// the latency exceeded the configured hedge threshold AND the
    /// rate budget allowed it, increments the would-fire counter and
    /// emits a tracing::info event. Lets operators see hedge potential
    /// even when running with `hedge_enabled = false`. True duplicate
    /// dispatch (race-then-discard) ships in
    /// `pipeline/hedge_dispatch.rs::forward_verify_with_hedge` for
    /// single-segment pipelines without a wire-format change — uses a
    /// fresh Uuid for the hedge so `pending_layer_results` doesn't
    /// collide with the primary. Multi-segment hedging remains deferred.
    pub fn record_hedge_observation(
        &self,
        model_id: &crate::types::ModelId,
        segment_idx: u8,
        holder: &crate::types::NodeId,
        latency_ms: f32,
    ) {
        let key = crate::inference::hedging::HedgeKey {
            model_id: model_id.clone(),
            segment_idx,
            holder: holder.clone(),
        };
        // Post-hoc dry-run hedge decision: would we have fired a hedge
        // for this forward if dispatch were wired? Check BEFORE the
        // observe call so the EWMA reflects the same baseline the
        // pre-completion decision would have used.
        let cfg = crate::inference::hedging::HedgeConfig {
            enabled: true, // always evaluate the would-have-fired branch
            after_factor: self.config.inference.hedge_after_factor,
            max_rate: self.config.inference.hedge_max_rate,
            min_samples: self.config.inference.hedge_min_samples,
        };
        if self
            .metrics
            .hedge_tracker
            .should_hedge(&key, latency_ms, cfg)
        {
            tracing::info!(
                model_id = %model_id.0,
                segment_idx,
                holder = %holder,
                latency_ms,
                p99_estimate = ?self.metrics.hedge_tracker.get(&key).map(|s| s.p99_estimate_ms()),
                hedge_dispatch_enabled = self.config.inference.hedge_enabled,
                "DIAG: hedge would have fired (dry-run; dispatch needs wire-format follow-up)"
            );
            // Count the decision so operators can compute the would-hedge
            // rate. record_decision(true, false) increments hedges_fired;
            // when actual dispatch lands, the winner-flag will be wired.
            self.metrics.hedge_tracker.record_decision(true, false);
        } else {
            self.metrics.hedge_tracker.record_decision(false, false);
        }
        self.metrics.hedge_tracker.observe(key, latency_ms);
    }

    /// Read the observed per-layer latency EMA for a peer. Returns None when
    /// this peer has no observed samples yet (caller falls back to static
    /// capability estimate).
    pub fn observed_latency_ms_per_layer(&self, node_id: &crate::types::NodeId) -> Option<f32> {
        self.metrics
            .peer_segment_latency_ms_per_layer
            .get(node_id)
            .map(|r| *r.value())
    }

    /// Item 8 Phase 2: longest-prefix cross-node cache lookup. Walks the
    /// chained BLAKE3 manifest of `prompt_tokens` at `block_size` granularity
    /// from longest → shortest, returning the first `(peer, block_hash,
    /// token_count)` triple whose peer set contains a non-self member. The
    /// caller then dispatches a KV-fetch to that peer.
    ///
    /// Self-entries recorded by the loopback forwarder are SKIPPED here —
    /// local hits are already served by the in-process `PrefixCache`, so a
    /// "remote" fetch to ourselves would only waste a round trip.
    ///
    /// Observed per-peer latency (via `observed_latency_ms_per_layer`) breaks
    /// ties at the same block_hash so the fastest peer wins when multiple
    /// holders are available. Peers without observed samples tie-break by
    /// NodeId for determinism.
    pub fn best_cross_node_prefix_match(
        &self,
        model_id: &crate::types::ModelId,
        prompt_tokens: &[u32],
        block_size: usize,
    ) -> Option<(crate::types::NodeId, [u8; 32], u32)> {
        if block_size == 0 || prompt_tokens.is_empty() {
            return None;
        }
        let manifest = crate::inference::split::compute_block_hashes(prompt_tokens, block_size);
        if manifest.is_empty() {
            return None;
        }
        let our_id = self.identity.node_id();
        let model_index = self.models.cross_node_prefix_index.get(model_id)?;
        // Walk longest-first. First peer that isn't self wins.
        for entry in manifest.iter().rev() {
            let Some(holders) = model_index.get(&entry.block_hash) else {
                continue;
            };
            let candidates: Vec<crate::types::NodeId> = holders
                .iter()
                .map(|r| r.clone())
                .filter(|n| n != our_id)
                .collect();
            if candidates.is_empty() {
                continue;
            }
            // Pick the peer with the lowest observed per-layer latency; if
            // none have observed samples, fall back to the NodeId that sorts
            // first (deterministic).
            let best = candidates.into_iter().min_by(|a, b| {
                let la = self
                    .observed_latency_ms_per_layer(a)
                    .unwrap_or(f32::INFINITY);
                let lb = self
                    .observed_latency_ms_per_layer(b)
                    .unwrap_or(f32::INFINITY);
                la.partial_cmp(&lb)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.0.cmp(&b.0))
            });
            if let Some(peer) = best {
                return Some((peer, entry.block_hash, entry.token_count));
            }
        }
        None
    }

    /// Item 8 Phase 2: end-to-end helper used by the admit-time KV-fetch
    /// path. Looks up the longest cross-node prefix match for `prompt_tokens`,
    /// installs a oneshot, dispatches `NetworkCommand::SendPrefixKvFetch` to
    /// the best peer, awaits the reply with `timeout_ms`, BLAKE3-verifies
    /// the returned snapshot tokens match the requested block hash, then
    /// deserializes the snapshot onto `device`.
    ///
    /// Returns `Ok(None)` for any "no hit" outcome (no index entry, peer
    /// miss, timeout, verification failure) so callers can unconditionally
    /// fall through to normal prefill. Returns `Ok(Some((snap, token_count)))`
    /// when a trusted KV snapshot is available — caller hydrates.
    pub async fn try_fetch_cross_node_prefix(
        &self,
        network_tx: &tokio::sync::mpsc::Sender<crate::types::NetworkCommand>,
        model_id: &crate::types::ModelId,
        prompt_tokens: &[u32],
        block_size: usize,
        device: &candle_core::Device,
        timeout_ms: u64,
    ) -> Result<Option<(crate::inference::split::KvSnapshot, usize)>, crate::error::SwarmError>
    {
        let Some((peer, block_hash, token_count)) =
            self.best_cross_node_prefix_match(model_id, prompt_tokens, block_size)
        else {
            return Ok(None);
        };
        let Some(peer_bytes) = self.peer_id_map.get(&peer).map(|r| r.clone()) else {
            tracing::debug!(%peer, "prefix-kv fetch: no peer_id_bytes in map — skipping");
            return Ok(None);
        };
        let fetch_id = uuid::Uuid::new_v4();
        let (tx, rx) = tokio::sync::oneshot::channel::<Option<Vec<u8>>>();
        self.pending_prefix_kv_fetches.insert(fetch_id, tx);
        // RAII cleanup: if the caller is cancelled before the response
        // lands, the oneshot drops automatically; the manager's later
        // resolve attempt becomes a no-op. Also remove ourselves so we
        // don't leak the DashMap entry on the cancellation path.
        struct FetchGuard<'a> {
            state: &'a SharedState,
            fetch_id: uuid::Uuid,
        }
        impl<'a> Drop for FetchGuard<'a> {
            fn drop(&mut self) {
                self.state.pending_prefix_kv_fetches.remove(&self.fetch_id);
            }
        }
        let _guard = FetchGuard {
            state: self,
            fetch_id,
        };
        let cmd = crate::types::NetworkCommand::SendPrefixKvFetch {
            target_peer_bytes: peer_bytes,
            request_id: fetch_id,
            model_id: model_id.clone(),
            block_hash,
        };
        if let Err(e) = network_tx.send(cmd).await {
            tracing::debug!(error = %e, "prefix-kv fetch: network_tx send failed");
            return Ok(None);
        }
        let bytes_opt =
            match tokio::time::timeout(std::time::Duration::from_millis(timeout_ms), rx).await {
                Ok(Ok(payload)) => payload,
                Ok(Err(_)) => {
                    tracing::debug!("prefix-kv fetch: oneshot dropped before response");
                    None
                }
                Err(_) => {
                    tracing::debug!(timeout_ms, "prefix-kv fetch: timed out");
                    None
                }
            };
        let Some(bytes) = bytes_opt else {
            return Ok(None);
        };
        let (snap, tokens_from_peer) =
            match crate::inference::split::deserialize_snapshot(&bytes, device) {
                Ok(x) => x,
                Err(e) => {
                    tracing::warn!(%peer, error = %e, "prefix-kv: deserialize failed");
                    return Ok(None);
                }
            };
        // BLAKE3 verify: re-hash the peer-provided tokens and compare against
        // the hash we asked for. Untrusted peers can't forge a matching chain.
        if !crate::inference::split::verify_token_hash_chain(
            &tokens_from_peer,
            block_size,
            snap.token_count,
            &block_hash,
        ) {
            tracing::warn!(
                %peer,
                fetch_id = %fetch_id,
                "prefix-kv: BLAKE3 verify failed — peer returned mismatched tokens (rejected)"
            );
            return Ok(None);
        }
        // Also require that `tokens_from_peer` is a strict prefix of OUR
        // prompt — otherwise the peer could hand us a VALID snapshot from
        // some other prompt that happens to have the same prefix length.
        // Chained-hash property already guarantees this, but belt + braces.
        if tokens_from_peer.len() > prompt_tokens.len()
            || &prompt_tokens[..tokens_from_peer.len()] != tokens_from_peer.as_slice()
        {
            tracing::warn!(
                %peer,
                "prefix-kv: peer tokens don't prefix-match our prompt (rejected)"
            );
            return Ok(None);
        }
        tracing::info!(
            %peer,
            matched_tokens = token_count,
            snapshot_bytes = bytes.len(),
            "DIAG: prefix-kv fetch HIT"
        );
        Ok(Some((snap, token_count as usize)))
    }

    /// Merge a foreign observation (received via `NodeCapabilityUpdate`
    /// gossip) into the local EMA, weighted by the gossip sender's trust.
    /// `weight` is the sender's trust score clamped to `[0, 1]`; the
    /// effective α collapses to `0.3 * weight` so trust=0 is a no-op and
    /// trust=1 matches a direct local sample. When no local entry exists
    /// yet, only moderately-trusted sources (weight ≥ 0.3) may seed it
    /// with the raw sample — this prevents a low-trust peer from painting
    /// us an out-of-band picture of a peer we've never observed.
    pub fn merge_peer_segment_latency(
        &self,
        node_id: &crate::types::NodeId,
        sample_ms_per_layer: f32,
        weight: f32,
    ) {
        if weight <= 0.0 || !sample_ms_per_layer.is_finite() || sample_ms_per_layer <= 0.0 {
            return;
        }
        let weight = weight.clamp(0.0, 1.0);
        const BASE_ALPHA: f32 = 0.3;
        const SEED_THRESHOLD: f32 = 0.3;
        let effective_alpha = BASE_ALPHA * weight;
        use dashmap::mapref::entry::Entry;
        match self
            .metrics
            .peer_segment_latency_ms_per_layer
            .entry(node_id.clone())
        {
            Entry::Occupied(mut e) => {
                let old = *e.get();
                e.insert(effective_alpha * sample_ms_per_layer + (1.0 - effective_alpha) * old);
            }
            Entry::Vacant(v) => {
                if weight >= SEED_THRESHOLD {
                    v.insert(sample_ms_per_layer);
                }
            }
        }
    }

    /// Returns "VRAM" if a GPU is available, "RAM" otherwise.
    pub fn memory_type_label(&self) -> &'static str {
        if self.gpu_info.is_some() {
            "VRAM"
        } else {
            "RAM"
        }
    }

    /// Look up the first loaded segment key for a model by its model_id. O(1) via secondary index.
    /// Falls back gracefully if the index is stale (e.g., after LRU eviction).
    pub fn find_split_model_key(
        &self,
        model_id: &crate::types::ModelId,
    ) -> Option<crate::inference::split::SplitModelKey> {
        if let Some(ranges) = self.split_model_index.get(model_id) {
            for &(s, e) in ranges.iter() {
                let key = (model_id.clone(), s, e);
                if self.split_models.contains_key(&key) {
                    return Some(key);
                }
            }
        }
        None
    }

    /// Check if any segment of a model is loaded. O(1) via secondary index.
    pub fn has_split_model(&self, model_id: &crate::types::ModelId) -> bool {
        if let Some(ranges) = self.split_model_index.get(model_id) {
            for &(s, e) in ranges.iter() {
                if self.split_models.contains_key(&(model_id.clone(), s, e)) {
                    return true;
                }
            }
        }
        false
    }

    /// This node's region for reporting/geo purposes: the explicitly configured
    /// `identity.region` if set, otherwise the IP-geolocated `detected_region`.
    ///
    /// `identity.region` is `None` unless the operator set it in config.toml, and
    /// the auto-detected value lives in `detected_region` — so any site that reads
    /// `config.identity.region` directly reports "no region" on the common
    /// auto-detected node, which is why the network map placed every such node
    /// nowhere (or defaulted them to the viewer's own country). Reporting paths
    /// (capacity announcement, WS region counts, region gossip) MUST use this.
    pub async fn effective_region(&self) -> Option<String> {
        if let Some(r) = self.config.identity.region.as_ref() {
            if !r.is_empty() {
                return Some(r.clone());
            }
        }
        self.detected_region.read().await.clone()
    }

    /// Non-blocking sibling of [`effective_region`] for synchronous callers
    /// (the scheduler's region-score, auto-manage's `our_region`). Uses
    /// `try_read` so it never blocks; if the `detected_region` lock is
    /// momentarily held it falls through to `None` rather than stalling a hot
    /// path. Same precedence as the async version — configured region wins,
    /// else detected — so the two never disagree on the same node (they used to:
    /// the scheduler read detected-first with no config fallback while
    /// auto-manage fell back to config, so one path could score a node's region
    /// as "unknown" while the other resolved it).
    pub fn effective_region_sync(&self) -> Option<String> {
        if let Some(r) = self.config.identity.region.as_ref() {
            if !r.is_empty() {
                return Some(r.clone());
            }
        }
        self.detected_region.try_read().ok().and_then(|g| g.clone())
    }

    /// Does the legacy single-model executor hold **this** model?
    ///
    /// **`model_loaded` on its own must never be used to pick the local
    /// inference path.** It is a global `AtomicBool` meaning "*a* model has
    /// been loaded at least once" — set when `inference.model_path` is
    /// configured at startup, or after a full-GGUF download via
    /// `/api/admin/hf/download`, and never cleared per-model. It says nothing
    /// about the model the caller actually asked for.
    ///
    /// `execute_local_batch` reads neither `request.model_id` nor anything
    /// derived from it: it generates from the singleton `executor` using the
    /// singleton `loaded_model_info`'s chat template. So dispatching on the
    /// bare flag meant that once a node had any full-GGUF model resident, a
    /// request for a *different* model was answered by the resident one —
    /// wrong weights and wrong prompt template, reported as a success, with
    /// the requested name echoed back in the response. Silently serving the
    /// wrong model is worse than failing, because nothing surfaces it.
    ///
    /// Matching mirrors what `resolve_model_for_inference` can hand the
    /// router: the display name, its slug, or a registry id whose manifest
    /// carries that same display name. Anything else is not this model.
    ///
    /// Note the deliberate asymmetry in the false case: answering "no" costs a
    /// trip through the distributed path, which handles a locally-complete
    /// model perfectly well. Answering "yes" wrongly produces a confident
    /// wrong answer. When in doubt, say no.
    pub async fn local_executor_serves(&self, model_id: &crate::types::ModelId) -> bool {
        if !self.model_loaded.load(std::sync::atomic::Ordering::Acquire) {
            return false;
        }
        let info = self.loaded_model_info.read().await;
        info.as_ref()
            .is_some_and(|loaded| self.model_id_names(model_id, &loaded.name))
    }

    /// Does `model_id` name the model whose GGUF display name is `loaded_name`?
    ///
    /// The one rule for "is this identifier that model", shared by
    /// [`Self::local_executor_serves`] (dispatch) and [`Self::is_shard_in_vram`]
    /// (reporting). The three accepted spellings are exactly what
    /// `resolve_model_for_inference` can produce: the display name, its slug,
    /// or a registry id whose manifest carries that same display name.
    ///
    /// **Matching is exact, never a substring.** `is_shard_in_vram` used to ask
    /// `model_id.0.contains(slug)`, so a node with "Llama 3.2" resident
    /// reported every `llama-3.2-*` variant as being in VRAM — different
    /// quantisations, different parameter counts, any id that merely started
    /// the same way. The registry clause covers the case a substring test was
    /// really reaching for (an id carrying a quant suffix the display name
    /// lacks) without matching unrelated models, because it requires the
    /// manifest's OWN name to equal the loaded one.
    fn model_id_names(&self, model_id: &crate::types::ModelId, loaded_name: &str) -> bool {
        if model_id.0 == loaded_name || model_id.0 == crate::types::slugify_model_name(loaded_name)
        {
            return true;
        }
        self.model_registry
            .get_manifest(model_id)
            .is_some_and(|m| m.name == loaded_name)
    }

    /// Whether any holder in `holders` can actually serve a shard *right now* —
    /// the local node, or a currently-connected peer.
    ///
    /// This is the reporting-side mirror of the scheduler's liveness oracle
    /// (`inference::scheduler::gather_candidates` filters shard holders by
    /// `connected_node_ids`, skipping the local node which is always available).
    /// Readiness/availability shown on the dashboard MUST use this rather than
    /// "the holder set is non-empty": `peer_registry`/`shard_holders` retain a
    /// peer's announces across disconnects, so a stale announce from a peer that
    /// has since left would otherwise mark a model "ready" that the scheduler
    /// can never assemble — the exact "dashboard says ready but it isn't" gap.
    pub fn any_holder_reachable(&self, holders: &[crate::types::NodeId]) -> bool {
        let local = self.identity.node_id();
        holders
            .iter()
            .any(|h| h == local || self.connected_node_ids.contains(h))
    }

    /// Check if a local shard is currently loaded in VRAM (process pool, split model, or legacy executor).
    pub fn is_shard_in_vram(&self, model_id: &crate::types::ModelId, shard_index: u32) -> bool {
        let window = self.model_process_pool.get_shard_window(model_id);
        match window {
            Some(w) => w.contains(&shard_index),
            None => {
                // Check if the process pool has this model loaded (without shard window info)
                if self.model_process_pool.is_loaded(model_id) {
                    return true;
                }
                // Check if the shard's layer range overlaps with any loaded split model segment
                if let Some(manifest) = self.model_registry.get_manifest(model_id) {
                    if let Some(shard_info) =
                        manifest.shards.iter().find(|s| s.index == shard_index)
                    {
                        let (sl, se) = shard_info.layer_range;
                        let (sl, se) = (sl as usize, se as usize);
                        if let Some(ranges) = self.split_model_index.get(model_id) {
                            return ranges.iter().any(|&(ls, le)| {
                                // Shard's layer range overlaps with this loaded segment
                                sl < le && se > ls
                            });
                        }
                    }
                }
                // Legacy fallback: a model loaded whole via `--model`. Uses the
                // shared identity rule — `try_read` rather than `.await`
                // because this is a sync reporting path; a write lock in
                // flight means a load/unload is happening, and "not in VRAM"
                // is the honest answer during that window.
                self.model_loaded.load(std::sync::atomic::Ordering::Relaxed)
                    && self
                        .loaded_model_info
                        .try_read()
                        .map(|info| {
                            info.as_ref()
                                .is_some_and(|i| self.model_id_names(model_id, &i.name))
                        })
                        .unwrap_or(false)
            }
        }
    }

    /// Check if a complete (all layers covered) split model is loaded.
    pub fn has_complete_split_model(&self, model_id: &crate::types::ModelId) -> bool {
        self.split_models
            .iter()
            .any(|e| e.key().0 == *model_id && e.value().is_complete)
    }

    /// Ensure a split model metadata entry exists for the given key.
    /// Creates the entry from GGUF header, runs VRAM-aware eviction, and inserts.
    /// Returns the split key. No-op if the entry already exists.
    pub fn ensure_split_model_entry(
        &self,
        model_id: &crate::types::ModelId,
        layer_start: usize,
        layer_end: usize,
        is_first: bool,
        is_last: bool,
        total_layers: usize,
    ) -> crate::inference::split::SplitModelKey {
        let split_key = (model_id.clone(), layer_start, layer_end);
        if self.split_models.contains_key(&split_key) {
            return split_key;
        }

        let shard_store = self.shard_store();
        let model_dir = shard_store.model_dir(model_id);
        let header_path = model_dir.join(crate::model::shard::HEADER_FILENAME);
        let vram_estimate = crate::daemon::estimate_vram_from_shard_dir(
            &model_dir,
            layer_start,
            layer_end,
            total_layers,
        );
        let entry = crate::inference::split::SplitModelEntry::from_header(
            &header_path,
            layer_start,
            layer_end,
            is_first,
            is_last,
            vram_estimate,
        );

        let vram_budget = crate::model::auto_manage::compute_vram_budget(self)
            .or(self.config.inference.max_split_model_memory_mb);
        if let Some(budget_mb) = vram_budget {
            let evicted = self.evict_split_models_and_free_vram(budget_mb, entry.estimated_vram_mb);
            if !evicted.is_empty() {
                tracing::info!(
                    evicted = evicted.len(),
                    budget_mb,
                    "Evicted LRU split models for VRAM budget"
                );
            }
        }
        self.index_split_model_insert(model_id, layer_start, layer_end);
        self.split_models.entry(split_key.clone()).or_insert(entry);
        split_key
    }

    /// Register a split model segment in the secondary index.
    pub fn index_split_model_insert(
        &self,
        model_id: &crate::types::ModelId,
        layer_start: usize,
        layer_end: usize,
    ) {
        self.split_model_index
            .entry(model_id.clone())
            .or_default()
            .push((layer_start, layer_end));
    }

    /// Evict split-model entries for the VRAM budget **and actually free the
    /// VRAM**.
    ///
    /// `evict_split_models_lru` only removes daemon-side metadata from
    /// `split_models`. The memory itself belongs to the model worker
    /// **subprocess**, and the only thing that returns it to the OS is killing
    /// that child (`ModelProcessPool::unload_model`). Every eviction site
    /// evicted and purged the index but never unloaded, so the budget was
    /// enforced against a phantom: the daemon decided it had freed 2 GB, loaded
    /// another model on top of memory that was never released, and the worker
    /// hit a real `CUDA_ERROR_OUT_OF_MEMORY`. `classify_worker_error` then
    /// pinned that model to the CPU for the rest of the run — a ~10x throughput
    /// loss with nothing in the API response to show for it. Reported across
    /// v0.3.53 and v0.3.54 as "GPU silently falls back to CPU from the daemon's
    /// own background churn"; open since 2026-07-21.
    ///
    /// **A model is unloaded only when no split-model entry references it any
    /// more.** Entries are keyed per layer range, so a node holding two
    /// segments of one model shares a single worker; unloading on the first
    /// eviction would kill a worker still serving the second.
    ///
    /// **And never while it is in use.** `active_pipelines` is the
    /// COORDINATOR's map and never holds peer-served work, so `serving_models`
    /// has to be checked too — otherwise this kills a worker mid-answer for a
    /// peer, which is precisely the mistake gotcha #194 records.
    ///
    /// The unload is spawned rather than awaited because the callers are sync
    /// and on the model-load path; the kill is a signal plus a process reap, so
    /// the VRAM comes back promptly without blocking the loader.
    pub fn evict_split_models_and_free_vram(
        &self,
        budget_mb: u64,
        needed_mb: u64,
    ) -> Vec<crate::inference::split::SplitModelKey> {
        let evicted = crate::inference::split::evict_split_models_lru(
            &self.split_models,
            &self.active_pipelines,
            budget_mb,
            needed_mb,
        );
        if evicted.is_empty() {
            return evicted;
        }
        self.purge_split_model_index_entries(&evicted);

        let mut to_unload: Vec<crate::types::ModelId> = Vec::new();
        for key in &evicted {
            let model_id = &key.0;
            if to_unload.contains(model_id) {
                continue;
            }
            // Another segment of the same model still resident → shared worker.
            if self.split_models.iter().any(|e| &e.key().0 == model_id) {
                continue;
            }
            if self.serving_models.contains_key(model_id) {
                tracing::debug!(model = %model_id, "Evicted metadata but worker is serving a peer — leaving it loaded");
                continue;
            }
            to_unload.push(model_id.clone());
        }

        if !to_unload.is_empty() {
            let pool = self.model_process_pool.clone();
            tokio::spawn(async move {
                for model_id in to_unload {
                    tracing::info!(model = %model_id, "Unloading evicted model to actually free its GPU memory");
                    pool.unload_model(&model_id).await;
                }
            });
        }
        evicted
    }

    /// Purge a list of evicted split-model keys from the secondary index.
    /// Pair with `evict_split_models_lru`'s return value to keep the
    /// `(model_id, layer_start, layer_end)` index in lockstep with the
    /// primary `split_models` map; without this, the index grows
    /// unboundedly under sustained VRAM pressure.
    pub fn purge_split_model_index_entries(
        &self,
        evicted: &[crate::inference::split::SplitModelKey],
    ) {
        for (model_id, layer_start, layer_end) in evicted {
            if let Some(mut ranges) = self.split_model_index.get_mut(model_id) {
                ranges.retain(|(s, e)| (s, e) != (layer_start, layer_end));
            }
        }
    }

    /// Remove all segments for a model from the secondary index.
    pub fn index_split_model_remove_all(&self, model_id: &crate::types::ModelId) {
        self.split_model_index.remove(model_id);
    }

    /// Clear cached split model entries for a model (e.g., after shard load/unload/delete).
    /// Call this whenever the set of loaded shards changes so inference re-evaluates segments.
    pub fn evict_split_models(&self, model_id: &crate::types::ModelId) {
        self.split_models.retain(|key, _| key.0 != *model_id);
        self.index_split_model_remove_all(model_id);
    }

    /// Evict cached split model entries AND kill the worker subprocess for a model.
    /// Use this when fully unloading a model (delete, unload, shard removal with no remaining shards).
    pub async fn evict_and_unload(&self, model_id: &crate::types::ModelId) {
        self.evict_split_models(model_id);
        self.model_process_pool
            .unload_and_clear_window(model_id)
            .await;
    }

    /// Emit a rich activity event to the dashboard.
    /// Lightweight fire-and-forget — if no WebSocket subscribers, the event is dropped.
    pub fn emit_activity(&self, event: ActivityEvent) {
        // Store in history for replay to new WS clients.
        {
            let mut history = self.events.activity_history.lock();
            history.push_back(event.clone());
            if history.len() > 100 {
                history.pop_front();
            }
        }
        let _ = self.events.activity_tx.send(event);
    }

    /// Signal the dashboard to refresh (peers changed, models changed, update available).
    /// Fire-and-forget — if no WebSocket subscribers, the signal is dropped.
    pub fn signal_dashboard(&self, signal: DashboardSignal) {
        let _ = self.events.dashboard_tx.send(signal);
    }

    /// Register a shard as locally held and announce it to the network.
    ///
    /// Steps: record in model_registry, broadcast ShardAnnounce, start DHT providing,
    /// signal dashboard ModelsChanged. Uses `try_send` on the network channel.
    pub fn announce_shard_acquired(
        &self,
        net_tx: &mpsc::Sender<crate::types::NetworkCommand>,
        shard_id: &crate::types::ShardId,
    ) {
        let node_id = self.identity.node_id().clone();
        self.model_registry
            .record_shard_holder(shard_id.clone(), node_id.clone());
        let announce = crate::types::SwarmMessage::ShardAnnounce(crate::types::ShardAnnounce {
            node_id,
            shards: vec![shard_id.clone()],
            timestamp: chrono::Utc::now(),
            // Incremental: one shard we just acquired says nothing about the
            // rest, so declaring completeness here would delete every other
            // shard of this model we hold.
            complete_for_models: Vec::new(),
        });
        let _ = net_tx.try_send(crate::types::NetworkCommand::Broadcast(announce));
        let _ = net_tx.try_send(crate::types::NetworkCommand::StartProviding(vec![
            shard_id.clone()
        ]));
        self.signal_dashboard(DashboardSignal::ModelsChanged);
    }

    /// Convenience accessor for a `ShardStore` rooted at this node's data dir.
    pub fn shard_store(&self) -> crate::model::shard::ShardStore {
        crate::model::shard::ShardStore::new(&self.config.node.data_dir)
    }

    /// Select the best peer from a set of holders based on LAN proximity, latency, and trust.
    /// Returns the first holder as fallback if no peers are in the registry.
    pub fn select_best_peer(&self, holders: &[crate::types::NodeId]) -> crate::types::NodeId {
        let mut scored: Vec<_> = holders
            .iter()
            .filter_map(|nid| {
                self.peer_registry.get(nid).map(|p| {
                    let is_lan = if p.is_lan_peer { 0u64 } else { 1 };
                    let latency = p.latency_ms.unwrap_or(9999) as u64;
                    let trust = (10000.0 - p.trust_score * 100.0) as u64;
                    (nid.clone(), is_lan * 100_000 + latency * 100 + trust)
                })
            })
            .collect();
        scored.sort_by_key(|(_, score)| *score);
        scored
            .first()
            .map(|(nid, _)| nid.clone())
            .unwrap_or_else(|| holders[0].clone())
    }

    /// Schedule deferred removal of an acquisition_progress entry after 5 seconds.
    pub fn schedule_acquisition_cleanup(self: &std::sync::Arc<Self>, mid: crate::types::ModelId) {
        let shared = self.clone();
        tokio::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_secs(5)).await;
            shared.models.acquisition_progress.remove(&mid);
        });
    }

    /// Reverse lookup: PeerId → NodeId, via `peer_id_map`. O(N_peers) scan, but
    /// peer counts are tiny in practice and this is only called on stream open.
    pub fn peer_to_node_id_from_registry(
        &self,
        peer: &libp2p::PeerId,
    ) -> Option<crate::types::NodeId> {
        let peer_bytes = peer.to_bytes();
        self.peer_id_map.iter().find_map(|entry| {
            if entry.value() == &peer_bytes {
                Some(entry.key().clone())
            } else {
                None
            }
        })
    }

    /// Resolve a `NodeId` to its libp2p `PeerId` bytes. The persistent
    /// `peer_id_map` (indexed at first connect, survives disconnects) is the
    /// primary source; `peer_registry` is a fallback for nodes seen via
    /// gossip but not yet connected. Returns `None` only when neither source
    /// has a record — typically a fresh node we've never observed.
    ///
    /// Replaces the duplicated 8-line lookup pattern that lived in 5 inference
    /// pipeline call sites; keep the fallback order in sync here.
    pub fn resolve_peer_id_bytes(&self, node_id: &crate::types::NodeId) -> Option<Vec<u8>> {
        self.peer_id_map
            .get(node_id)
            .map(|r| r.value().clone())
            .or_else(|| {
                self.peer_registry
                    .get(node_id)
                    .and_then(|p| p.peer_id_bytes.clone())
            })
    }

    /// Resolve a `NodeId` to its `PeerId` bytes **only while we hold a live
    /// libp2p connection to it**. Use this for any message that
    /// `network::manager::relay::is_relay_eligible` refuses — i.e. everything
    /// except `RemoteGenerateRequest` / `StreamingToken` / `CancelInference`.
    /// For those direct-only messages "reachable" means "connected", so a
    /// plain [`Self::resolve_peer_id_bytes`] hands back a target the send path
    /// can only drop.
    ///
    /// **Why this exists.** `peer_id_map` is deliberately persistent — it is
    /// indexed at first connect and survives disconnects, and its only eviction
    /// (`cleanup_stale_peer_id_map`) is gated behind an 8,000-entry soft cap
    /// that never trips on a small swarm. Meanwhile a departed peer keeps
    /// reaching us through the **gossipsub mesh**, relayed by other peers, long
    /// after its direct connection is gone. Gossip reachability is not
    /// request_response reachability, so replying to gossip by resolving
    /// through `peer_id_map` alone produced an unbounded 30s loop of
    /// undeliverable sends (one departed peer accounted for 45% of a night's
    /// log volume).
    ///
    /// `connected_node_ids` is the liveness oracle (see
    /// `.claude/rules/architecture.md` § Scheduler Liveness Oracle);
    /// `peer_registry` is explicitly NOT, as it is preserved across
    /// disconnects for reconnect purposes.
    pub fn resolve_connected_peer_id_bytes(
        &self,
        node_id: &crate::types::NodeId,
    ) -> Option<Vec<u8>> {
        if !self.connected_node_ids.contains(node_id) {
            return None;
        }
        self.resolve_peer_id_bytes(node_id)
    }
}

#[cfg(test)]
mod connected_peer_resolution_tests {
    use crate::types::NodeId;

    fn test_state() -> std::sync::Arc<crate::daemon::SharedState> {
        use crate::identity::Identity;
        use crate::inference::executor::ModelExecutor;
        use crate::storage::db::Database;
        use tokio::sync::Mutex;

        let identity = Identity::generate();
        let temp = tempfile::tempdir().unwrap();
        let db = Database::open(temp.path()).unwrap();
        let executor = std::sync::Arc::new(Mutex::new(ModelExecutor::new()));
        let (state, _, _) = crate::daemon::SharedState::new(
            crate::config::Config::default(),
            identity,
            db,
            executor,
            None,
        );
        state
    }

    fn peer_bytes() -> Vec<u8> {
        libp2p::PeerId::random().to_bytes()
    }

    /// A connected peer resolves exactly as the ungated helper does — the gate
    /// must not cost us reachable targets.
    #[test]
    fn a_connected_peer_still_resolves() {
        let state = test_state();
        let node = NodeId([7u8; 32]);
        let bytes = peer_bytes();
        state.peer_id_map.insert(node.clone(), bytes.clone());
        state.connected_node_ids.insert(node.clone());

        assert_eq!(
            state.resolve_connected_peer_id_bytes(&node),
            Some(bytes),
            "a live peer must still be reachable"
        );
    }

    /// The defect this helper exists for: `peer_id_map` deliberately survives
    /// disconnects, so the ungated lookup keeps handing back a target the send
    /// path can only drop. Observed live as one departed-but-still-gossiping
    /// peer drawing an undeliverable HealthPong every 30s for hours.
    #[test]
    fn a_departed_peer_does_not_resolve_even_though_the_map_retains_it() {
        let state = test_state();
        let node = NodeId([9u8; 32]);
        state.peer_id_map.insert(node.clone(), peer_bytes());
        // Never connected, or connected and since dropped — same state.
        assert!(!state.connected_node_ids.contains(&node));

        assert!(
            state.resolve_peer_id_bytes(&node).is_some(),
            "precondition: the persistent map still holds the mapping"
        );
        assert_eq!(
            state.resolve_connected_peer_id_bytes(&node),
            None,
            "an unreachable peer must not be handed to a direct-only send"
        );
    }

    /// Disconnecting must actually take effect — the gate reads live state
    /// rather than caching a verdict from first resolution.
    #[test]
    fn resolution_stops_the_moment_the_peer_disconnects() {
        let state = test_state();
        let node = NodeId([11u8; 32]);
        state.peer_id_map.insert(node.clone(), peer_bytes());
        state.connected_node_ids.insert(node.clone());
        assert!(state.resolve_connected_peer_id_bytes(&node).is_some());

        state.connected_node_ids.remove(&node);
        assert_eq!(state.resolve_connected_peer_id_bytes(&node), None);
    }

    /// `peer_registry` is preserved across disconnects for reconnect purposes
    /// and is explicitly NOT the liveness oracle, so it must not resurrect a
    /// departed peer through the fallback arm of `resolve_peer_id_bytes`.
    #[test]
    fn the_peer_registry_fallback_is_not_a_liveness_backdoor() {
        let state = test_state();
        let node = NodeId([13u8; 32]);
        state.peer_registry.insert(
            node.clone(),
            crate::types::PeerInfo {
                node_id: node.clone(),
                addresses: vec![],
                capability: None,
                last_seen: chrono::Utc::now(),
                latency_ms: Some(50),
                trust_score: 0.5,
                peer_id_bytes: Some(peer_bytes()),
                active_request_count: 0,
                first_seen: 0,
                verified_transaction_count: 0,
                is_lan_peer: false,
            },
        );

        assert!(
            state.resolve_peer_id_bytes(&node).is_some(),
            "precondition: the registry fallback resolves"
        );
        assert_eq!(state.resolve_connected_peer_id_bytes(&node), None);
    }
}

#[cfg(test)]
mod encrypted_pipeline_precedence_tests {
    use super::SharedState;

    fn resolve(explicit: Option<bool>, global: bool, auto: bool, ends: bool) -> bool {
        SharedState::resolve_encrypted_pipeline_inner(explicit, global, auto, || ends)
    }

    /// The point of the change: privacy applies automatically wherever it can,
    /// because it is the only thing stopping the machine answering you from
    /// reading your prompt, and it was off unless a user knew to look.
    #[test]
    fn on_automatically_when_this_node_holds_both_ends() {
        assert!(resolve(None, false, true, true));
    }

    /// And never where it would break: without both ends an encrypted pipeline
    /// has no legal route, so turning it on would fail the request outright.
    #[test]
    fn off_when_the_node_cannot_hold_both_ends() {
        assert!(!resolve(None, false, true, false));
    }

    /// A user who turned it OFF for a model must keep it off — even though the
    /// node could support it. This is the case a naive "default on" breaks.
    #[test]
    fn an_explicit_per_model_off_is_respected() {
        assert!(!resolve(Some(false), false, true, true));
        assert!(
            !resolve(Some(false), true, true, true),
            "even against a global on"
        );
    }

    /// And an explicit per-model ON survives everything.
    #[test]
    fn an_explicit_per_model_on_is_respected() {
        assert!(resolve(Some(true), false, false, false));
    }

    /// Opting out of the automatic behaviour restores off-unless-asked.
    #[test]
    fn auto_can_be_turned_off_entirely() {
        assert!(!resolve(None, false, false, true));
    }

    /// The pre-existing global switch keeps working regardless of shard layout;
    /// the scheduler still refuses the route if it cannot be honoured.
    #[test]
    fn explicit_global_on_still_applies() {
        assert!(resolve(None, true, false, false));
        assert!(resolve(None, true, true, false));
    }
}
