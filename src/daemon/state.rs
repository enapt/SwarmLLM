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
use crate::types::{CreditBalance, NodeId, NodeStats, PeerInfo, PipelineAssignment};

use super::resolve_api_key;

/// Unified activity event for the dashboard — the single event bus.
/// Pushed over WebSocket as `activity_event` messages. Replaces the former
/// separate `prune_event`, `lan_peer_discovered`, and `system_notification`
/// WS message types (all now flow through this struct).
#[derive(Clone, Debug, serde::Serialize)]
pub struct ActivityEvent {
    /// Event category for frontend grouping/filtering.
    pub category: &'static str,
    /// Machine-readable event kind.
    pub kind: &'static str,
    /// Human-readable description (English; frontend may i18n-override).
    pub message: String,
    /// Optional model ID for per-model ticker routing.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_id: Option<String>,
    /// Optional model display name.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_name: Option<String>,
    /// Optional peer/node ID (short hex).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub node_id: Option<String>,
    /// Optional numeric detail (e.g. shard index, credit amount, latency).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail_num: Option<i64>,
    /// Optional string detail (e.g. reason, source, error message).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail_str: Option<String>,
    /// If set, the frontend shows a toast at this level ("success", "info", "warning", "error").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub toast_level: Option<&'static str>,
    /// Toast auto-dismiss duration in ms (default 5000 if toast_level is set but this is None).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub toast_duration_ms: Option<u32>,
    /// Shard index (for prune/shard events that need structured data beyond detail_num).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub shard_index: Option<u32>,
    /// Bytes freed (for prune events).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub freed_bytes: Option<u64>,
    /// Holder count before an operation (for prune events).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub holder_count_before: Option<usize>,
    /// Holder count after an operation (for prune events).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub holder_count_after: Option<usize>,
    /// Remaining local shards after an operation (for prune events).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub remaining_local_shards: Option<u32>,
    /// ISO 8601 timestamp for events that need a backend-authoritative time (e.g. prune).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timestamp: Option<String>,
}

impl ActivityEvent {
    /// Create an event with only the core fields; all extended fields default to None.
    pub fn new(category: &'static str, kind: &'static str, message: String) -> Self {
        Self {
            category,
            kind,
            message,
            model_id: None,
            model_name: None,
            node_id: None,
            detail_num: None,
            detail_str: None,
            toast_level: None,
            toast_duration_ms: None,
            shard_index: None,
            freed_bytes: None,
            holder_count_before: None,
            holder_count_after: None,
            remaining_local_shards: None,
            timestamp: None,
        }
    }

    /// Builder: set model_id.
    pub fn with_model(mut self, model_id: impl Into<String>) -> Self {
        self.model_id = Some(model_id.into());
        self
    }

    /// Builder: set model_name.
    pub fn with_model_name(mut self, name: impl Into<String>) -> Self {
        self.model_name = Some(name.into());
        self
    }

    /// Builder: set node_id.
    pub fn with_node(mut self, node_id: impl Into<String>) -> Self {
        self.node_id = Some(node_id.into());
        self
    }

    /// Builder: set detail_num.
    pub fn with_detail_num(mut self, n: i64) -> Self {
        self.detail_num = Some(n);
        self
    }

    /// Builder: set detail_str.
    pub fn with_detail_str(mut self, s: impl Into<String>) -> Self {
        self.detail_str = Some(s.into());
        self
    }

    /// Builder: request a frontend toast.
    pub fn with_toast(mut self, level: &'static str, duration_ms: u32) -> Self {
        self.toast_level = Some(level);
        self.toast_duration_ms = Some(duration_ms);
        self
    }

    pub fn with_shard_index(mut self, idx: u32) -> Self {
        self.shard_index = Some(idx);
        self
    }

    pub fn with_freed_bytes(mut self, bytes: u64) -> Self {
        self.freed_bytes = Some(bytes);
        self
    }

    pub fn with_holders(mut self, before: usize, after: usize) -> Self {
        self.holder_count_before = Some(before);
        self.holder_count_after = Some(after);
        self
    }

    pub fn with_remaining_local(mut self, n: u32) -> Self {
        self.remaining_local_shards = Some(n);
        self
    }

    pub fn with_timestamp(mut self, ts: impl Into<String>) -> Self {
        self.timestamp = Some(ts.into());
        self
    }
}

/// Signal enum for dashboard-targeted WS pushes.
/// Consolidates peer_list_changed, models_changed, and update_available
/// into a single broadcast channel to reduce channel proliferation.
#[derive(Clone, Debug)]
pub enum DashboardSignal {
    /// Peer registry changed — push full peer list to dashboard.
    PeersChanged,
    /// Model state changed (shard download, load, prune) — frontend should re-fetch models.
    ModelsChanged,
    /// Software update available — push banner to dashboard.
    UpdateAvailable(crate::update::UpdateInfo),
}

/// Thread-safe shared state accessible by all daemon tasks.
/// Cached info about a locally loaded model (lock-free reads).
#[derive(Clone, Debug)]
pub struct LoadedModelInfo {
    pub name: String,
    pub size_bytes: u64,
    /// EOS token IDs loaded from GGUF metadata.
    pub eos_tokens: Vec<u32>,
    /// Chat template from GGUF `tokenizer.chat_template` metadata (Jinja2 format).
    pub chat_template: Option<String>,
    /// BOS token string from GGUF metadata.
    pub bos_token: String,
    /// EOS token string from GGUF metadata.
    pub eos_token: String,
}

// ---- Sub-structs for logical field grouping ----

/// Event bus: activity events + dashboard signals + update state.
pub struct EventBus {
    pub activity_tx: broadcast::Sender<ActivityEvent>,
    pub activity_history: std::sync::Mutex<VecDeque<ActivityEvent>>,
    pub dashboard_tx: broadcast::Sender<DashboardSignal>,
    pub update_state: Arc<RwLock<crate::update::UpdateState>>,
}

impl EventBus {
    /// Remove stale `model_loaded` history entries for a model (e.g., after load/unload/delete).
    pub fn clear_model_load_history(&self, model_id: &str) {
        if let Ok(mut history) = self.activity_history.lock() {
            history
                .retain(|e| !(e.kind == "model_loaded" && e.model_id.as_deref() == Some(model_id)));
        }
    }
}

/// Credit & pool: balances, pool membership, escrow, trust, anti-gaming.
pub struct CreditPool {
    pub credit_balance: Arc<RwLock<CreditBalance>>,
    pub pending_credit_earn: std::sync::atomic::AtomicI64,
    pub pool_state: RwLock<Option<crate::pool::types::PoolState>>,
    pub pool_registry: DashMap<crate::pool::types::PoolId, crate::pool::types::PoolState>,
    pub pool_tx: RwLock<Option<mpsc::Sender<crate::pool::types::PoolCommand>>>,
    pub pool_credit_rates: DashMap<NodeId, crate::config::CreditRateConfig>,
    pub trust_manager: crate::credit::trust::TrustManager,
    pub escrow_manager: Arc<crate::credit::escrow::EscrowManager>,
    pub anti_gaming: tokio::sync::Mutex<crate::credit::anti_gaming::AntiGaming>,
    pub peer_credit_balances: DashMap<NodeId, i64>,
    /// Private mode: restrict inference + auto-manage to pool members (+ optional LAN peers).
    pub private_mode: std::sync::atomic::AtomicBool,
}

/// Model management: shard acquisition, auto-manage, trust gating, pruning.
pub struct ModelMgmt {
    pub acquisition_progress:
        DashMap<crate::types::ModelId, crate::model::acquisition::AcquisitionStatus>,
    pub hf_sources: DashMap<crate::types::ModelId, HfSource>,
    pub auto_manage_notify: Arc<tokio::sync::Notify>,
    pub auto_manage_enabled: std::sync::atomic::AtomicBool,
    pub auto_manage_default_model_cap: AtomicU32,
    pub model_auto_manage_policies:
        DashMap<crate::types::ModelId, crate::config::ModelAutoManagePolicy>,
    pub hf_probe_cache: DashMap<crate::types::ModelId, HfProbeInfo>,
    pub peer_shard_downloads: DashMap<crate::types::ShardId, Vec<(NodeId, u32)>>,
    pub download_cancel_flags: DashMap<crate::types::ModelId, Arc<AtomicBool>>,
    pub model_trust: DashMap<crate::types::ModelId, crate::types::ModelTrustInfo>,
    pub loading_models: DashMap<crate::types::ModelId, Arc<tokio::sync::Notify>>,
    pub locked_shards: DashMap<crate::types::ShardId, bool>,
    pub model_request_counts: DashMap<crate::types::ModelId, AtomicU64>,
    pub resource_schedule: RwLock<crate::config::ResourceSchedule>,
    pub prune_history: RwLock<VecDeque<crate::types::PruneEvent>>,
}

impl ModelMgmt {
    /// Check if a shard is currently being downloaded, pending, or verifying.
    /// Prevents races where multiple subsystems try to download the same shard.
    pub fn is_shard_in_progress(&self, model_id: &crate::types::ModelId, shard_index: u32) -> bool {
        self.acquisition_progress
            .get(model_id)
            .map(|entry| {
                entry
                    .shard_progress
                    .get(&shard_index)
                    .map(|sp| {
                        matches!(
                            sp.state,
                            crate::model::acquisition::ShardState::Downloading
                                | crate::model::acquisition::ShardState::Pending
                                | crate::model::acquisition::ShardState::Verifying
                        )
                    })
                    .unwrap_or(false)
            })
            .unwrap_or(false)
    }
}

/// Metrics, stats, and provider configuration.
pub struct MetricsProviders {
    pub inference_requests_total: AtomicU64,
    pub inference_latency_samples: std::sync::RwLock<std::collections::VecDeque<f64>>,
    pub channel_metrics: ChannelMetricsSet,
    pub ws_connection_count: std::sync::atomic::AtomicUsize,
    pub node_stats: RwLock<NodeStats>,
    pub providers_config: RwLock<crate::config::ProvidersConfig>,
    pub provider_model_map: DashMap<String, String>,
    pub provider_models_cache: RwLock<(Vec<serde_json::Value>, std::time::Instant)>,
}

// ---- Main SharedState ----

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

    // Inference engine
    pub executor: SharedExecutor,
    pub draft_executor: SharedExecutor,
    pub loaded_model_info: RwLock<Option<LoadedModelInfo>>,
    pub gpu_info: Option<crate::inference::executor::GpuInfo>,
    pub model_loaded: std::sync::atomic::AtomicBool,
    pub active_pipelines: DashMap<uuid::Uuid, PipelineAssignment>,
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
    pub adapter_registry: Arc<crate::model::lora::AdapterRegistry>,
    pub model_process_pool: Arc<crate::inference::process_pool::ModelProcessPool>,
    // Network & crypto
    pub session_manager: Arc<crate::crypto::SessionManager>,
    pub gossip_sealer: Arc<crate::crypto::GossipSealer>,
    pub lan_peer_count: std::sync::atomic::AtomicUsize,
    pub detected_region: RwLock<Option<String>>,
    pub shard_bytes_served: AtomicU64,
    pub relay_seconds_served: AtomicU64,
    pub active_relay_circuits: DashMap<(libp2p::PeerId, libp2p::PeerId), std::time::Instant>,
    pub region_shard_summaries:
        DashMap<(String, crate::types::ModelId), crate::types::RegionShardSummary>,
    pub region_demand: DashMap<(crate::types::ModelId, String), f64>,
    pub dht_query_tx: mpsc::Sender<crate::types::ModelId>,

    // Sub-structs (logically grouped fields)
    pub events: EventBus,
    pub credits: CreditPool,
    pub models: ModelMgmt,
    pub metrics: MetricsProviders,

    shutdown_tx: watch::Sender<bool>,
}

/// Atomic counters for a single mpsc channel.
pub struct ChannelCounters {
    pub capacity: u32,
    pub sent: AtomicU64,
    pub dropped: AtomicU64,
}

impl ChannelCounters {
    pub fn new(capacity: u32) -> Self {
        Self {
            capacity,
            sent: AtomicU64::new(0),
            dropped: AtomicU64::new(0),
        }
    }

    pub fn record_sent(&self) {
        self.sent.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    pub fn record_dropped(&self) {
        self.dropped
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
}

/// Backpressure metrics for all daemon mpsc channels.
pub struct ChannelMetricsSet {
    pub network_cmd: Arc<ChannelCounters>,
    pub network_out: Arc<ChannelCounters>,
    pub router_cmd: Arc<ChannelCounters>,
    pub rebalance: Arc<ChannelCounters>,
    pub acquisition: Arc<ChannelCounters>,
    pub pool_cmd: Arc<ChannelCounters>,
}

impl ChannelMetricsSet {
    fn new() -> Self {
        Self {
            network_cmd: Arc::new(ChannelCounters::new(1024)),
            network_out: Arc::new(ChannelCounters::new(1024)),
            router_cmd: Arc::new(ChannelCounters::new(256)),
            rebalance: Arc::new(ChannelCounters::new(64)),
            acquisition: Arc::new(ChannelCounters::new(64)),
            pool_cmd: Arc::new(ChannelCounters::new(64)),
        }
    }
}

/// Tracks the HuggingFace origin of a model for re-downloading shards.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct HfSource {
    pub repo_id: String,
    pub filename: String,
    /// Filename of the mmproj GGUF on HuggingFace (for VLM models).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mmproj_filename: Option<String>,
}

/// Cached result from probing a HuggingFace GGUF file.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct HfProbeInfo {
    pub repo_id: String,
    pub filename: String,
    pub shard_count: u32,
    pub total_size_bytes: u64,
    pub probed_at: chrono::DateTime<chrono::Utc>,
}

/// Collects partial AllReduce tensors from TP ranks for a single (request, layer).
/// When all `tp_size` partials arrive, the coordinator sums them and responds.
pub struct TpAllReduceCollector {
    pub tp_size: u32,
    /// Collected partials indexed by tp_rank.
    pub partials: Vec<Option<crate::types::TpAllReduceRequest>>,
    /// Sender peer bytes for responding to each rank.
    pub sender_peers: Vec<Option<Vec<u8>>>,
    pub created_at: std::time::Instant,
}

impl TpAllReduceCollector {
    pub fn new(tp_size: u32) -> Self {
        // Clamp tp_size to [1, 32] to prevent panics from empty partials vec
        // and bound memory allocation from malicious requests
        let safe_size = tp_size.clamp(1, 32) as usize;
        Self {
            tp_size,
            partials: vec![None; safe_size],
            sender_peers: vec![None; safe_size],
            created_at: std::time::Instant::now(),
        }
    }

    /// Insert a partial. Returns true when all partials have arrived.
    pub fn insert(
        &mut self,
        req: crate::types::TpAllReduceRequest,
        sender_peer: Option<Vec<u8>>,
    ) -> bool {
        let rank = req.tp_rank as usize;
        // Validate tp_rank is within bounds and tp_size matches collector's expected size
        if rank >= self.partials.len() {
            tracing::warn!(
                rank,
                tp_size = self.tp_size,
                "AllReduce: tp_rank out of bounds — ignoring"
            );
            return false;
        }
        if req.tp_size != self.tp_size {
            tracing::warn!(
                req_tp_size = req.tp_size,
                collector_tp_size = self.tp_size,
                "AllReduce: tp_size mismatch — ignoring"
            );
            return false;
        }
        if self.partials[rank].is_some() {
            tracing::warn!(rank, "AllReduce: duplicate partial for rank — overwriting");
        }
        self.sender_peers[rank] = sender_peer;
        self.partials[rank] = Some(req);
        self.partials.iter().all(|p| p.is_some())
    }

    /// Sum all partial tensors (f32) and return the reduced bytes + shape.
    pub fn reduce_sum(&self) -> Result<(Vec<u8>, Vec<u32>), crate::error::SwarmError> {
        let first = self.partials[0].as_ref().ok_or_else(|| {
            crate::error::SwarmError::Internal("AllReduce: missing rank 0 partial".into())
        })?;
        let shape = first.shape.clone();
        let elem_count: usize = shape
            .iter()
            .try_fold(1usize, |acc, &s| acc.checked_mul(s as usize))
            .ok_or_else(|| {
                crate::error::SwarmError::Internal("AllReduce: shape overflow".into())
            })?;
        // Cap at 256MB worth of f32 elements (64M floats)
        if elem_count > 64 * 1024 * 1024 {
            return Err(crate::error::SwarmError::Internal(
                "AllReduce: tensor too large".into(),
            ));
        }

        // Decompress first partial (cap decompressed size to prevent zip-bomb)
        let max_decompressed = elem_count * 4 + 1024; // expected size + small margin
        let decompressed = {
            let mut decoder = zstd::Decoder::new(std::io::Cursor::new(&first.partial_data))
                .map_err(|e| crate::error::SwarmError::Internal(format!("zstd init: {e}")))?;
            let mut buf = Vec::with_capacity(elem_count * 4);
            use std::io::Read;
            decoder
                .by_ref()
                .take(max_decompressed as u64)
                .read_to_end(&mut buf)
                .map_err(|e| crate::error::SwarmError::Internal(format!("zstd decompress: {e}")))?;
            buf
        };
        let mut sum = vec![0.0f32; elem_count];
        if decompressed.len() == elem_count * 4 {
            for (i, chunk) in decompressed.chunks_exact(4).enumerate() {
                sum[i] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            }
        }

        // Add remaining partials
        for (i, partial) in self.partials[1..].iter().enumerate() {
            let req = partial.as_ref().ok_or_else(|| {
                crate::error::SwarmError::Internal(format!(
                    "AllReduce: missing rank {} partial",
                    i + 1
                ))
            })?;
            let dec = {
                let mut decoder = zstd::Decoder::new(std::io::Cursor::new(&req.partial_data))
                    .map_err(|e| crate::error::SwarmError::Internal(format!("zstd init: {e}")))?;
                let mut buf = Vec::with_capacity(elem_count * 4);
                use std::io::Read;
                decoder
                    .by_ref()
                    .take(max_decompressed as u64)
                    .read_to_end(&mut buf)
                    .map_err(|e| {
                        crate::error::SwarmError::Internal(format!("zstd decompress: {e}"))
                    })?;
                buf
            };
            if dec.len() == elem_count * 4 {
                for (j, chunk) in dec.chunks_exact(4).enumerate() {
                    sum[j] += f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                }
            } else {
                return Err(crate::error::SwarmError::Internal(format!(
                    "AllReduce: rank {} partial size mismatch ({} != {})",
                    i + 1,
                    dec.len(),
                    elem_count * 4
                )));
            }
        }

        // Check for NaN/Inf in reduced result (possible tensor poisoning)
        if sum.iter().any(|v| !v.is_finite()) {
            return Err(crate::error::SwarmError::Internal(
                "AllReduce result contains NaN/Inf — possible tensor poisoning".into(),
            ));
        }

        // Compress reduced result
        let raw: Vec<u8> = sum.iter().flat_map(|f| f.to_le_bytes()).collect();
        let compressed = zstd::encode_all(std::io::Cursor::new(&raw), 1)
            .map_err(|e| crate::error::SwarmError::Internal(format!("zstd compress: {e}")))?;
        Ok((compressed, shape))
    }
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
        // Grab signing key bytes before identity is moved into the struct
        let signing_key_bytes = identity.signing_key_bytes();
        let state = Arc::new(Self {
            config: config.clone(),
            identity,
            db: db.clone(),
            peer_registry: DashMap::new(),
            model_registry,
            active_pipelines: DashMap::new(),
            metrics: MetricsProviders {
                node_stats: RwLock::new(NodeStats::default()),
                inference_requests_total: AtomicU64::new(0),
                inference_latency_samples: std::sync::RwLock::new(std::collections::VecDeque::new()),
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
            },
            credits: CreditPool {
                credit_balance: Arc::new(RwLock::new(CreditBalance {
                    node_id,
                    balance: 0,
                    lifetime_earned: 0,
                    lifetime_spent: 0,
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
                private_mode: std::sync::atomic::AtomicBool::new({
                    // Restore from DB, fall back to config
                    db.get_json::<bool>("pool_state", "private_mode")
                        .ok()
                        .flatten()
                        .unwrap_or(config.pool.private_mode)
                }),
            },
            models: ModelMgmt {
                acquisition_progress: DashMap::new(),
                hf_sources,
                auto_manage_notify: Arc::new(tokio::sync::Notify::new()),
                auto_manage_enabled: std::sync::atomic::AtomicBool::new(auto_manage_enabled),
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
            },
            events: EventBus {
                dashboard_tx: broadcast::channel(32).0,
                update_state: Arc::new(RwLock::new(crate::update::UpdateState::default())),
                activity_tx: broadcast::channel(256).0,
                activity_history: std::sync::Mutex::new(VecDeque::new()),
            },
            // Root-level fields (not sub-structed)
            executor,
            draft_executor: Arc::new(tokio::sync::Mutex::new(
                crate::inference::executor::ModelExecutor::new(),
            )),
            loaded_model_info: RwLock::new(None),
            gpu_info,
            pending_layer_results: DashMap::new(),
            split_models: DashMap::new(),
            split_model_index: DashMap::new(),
            kv_cache_store: Arc::new(crate::inference::split::KvCacheStore::new(
                std::time::Duration::from_secs(kv_cache_ttl_secs),
            )),
            gguf_meta: DashMap::new(),
            nickname_registry,
            peer_id_map: DashMap::new(),
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
            pending_vision_results: DashMap::new(),
            pending_tp_partials: DashMap::new(),
            allreduce_registry: Arc::new(crate::inference::allreduce::AllReduceRegistry::new()),
            ring_chunk_registry: Arc::new(crate::inference::allreduce::RingChunkRegistry::new()),
            shard_bytes_served: AtomicU64::new(0),
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
                // Legacy fallback: check loaded_model_info for --model flag loaded models
                self.model_loaded.load(std::sync::atomic::Ordering::Relaxed)
                    && self
                        .loaded_model_info
                        .try_read()
                        .map(|info| {
                            info.as_ref().is_some_and(|i| {
                                model_id
                                    .0
                                    .contains(&i.name.to_lowercase().replace([' ', '_'], "-"))
                            })
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
            let evicted = crate::inference::split::evict_split_models_lru(
                &self.split_models,
                &self.active_pipelines,
                budget_mb,
                entry.estimated_vram_mb,
            );
            if evicted > 0 {
                tracing::info!(
                    evicted,
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
        // Store in history for replay to new WS clients
        if let Ok(mut history) = self.events.activity_history.lock() {
            history.push_back(event.clone());
            if history.len() > 100 {
                history.pop_front();
            }
        }
        let _ = self.events.activity_tx.send(event);
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
        });
        let _ = net_tx.try_send(crate::types::NetworkCommand::Broadcast(announce));
        let _ = net_tx.try_send(crate::types::NetworkCommand::StartProviding(vec![
            shard_id.clone()
        ]));
        let _ = self
            .events
            .dashboard_tx
            .send(DashboardSignal::ModelsChanged);
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
}
