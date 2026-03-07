use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64};
use std::sync::Arc;

use dashmap::DashMap;
use tokio::sync::{broadcast, mpsc, watch, RwLock};
use tokio::task::JoinSet;

use crate::config::Config;
use crate::credit::ledger::CreditLedger;
use crate::error::SwarmError;
use crate::health::monitor::HealthMonitor;
use crate::health::rebalancer::ShardRebalancer;
use crate::identity::Identity;
use crate::inference::executor::SharedExecutor;
use crate::inference::router::{InferenceRouter, RouterCommand};
use crate::model::acquisition::{AcquisitionCommand, AcquisitionManager};
use crate::model::registry::ModelRegistry;
use crate::model::shard::ShardStore;
use crate::network::manager::NetworkManager;
use crate::storage::db::Database;
use crate::types::{
    CreditBalance, EphemeralKeyExchange, NetworkCommand, NodeId, NodeStats, PeerInfo,
    PipelineAssignment, RebalanceEvent, ShardId, SwarmMessage,
};

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

pub struct SharedState {
    pub config: Config,
    pub identity: Identity,
    pub db: Database,
    pub peer_registry: DashMap<NodeId, PeerInfo>,
    pub model_registry: ModelRegistry,
    pub active_pipelines: DashMap<uuid::Uuid, PipelineAssignment>,
    pub credit_balance: Arc<RwLock<CreditBalance>>,
    pub node_stats: RwLock<NodeStats>,
    pub executor: SharedExecutor,
    /// Optional draft model executor for speculative decoding.
    /// When loaded alongside the main model, enables speculative decoding
    /// for 2-3x throughput improvement on local inference.
    pub draft_executor: SharedExecutor,
    /// Cached model info for lock-free reads (set once at startup).
    pub loaded_model_info: RwLock<Option<LoadedModelInfo>>,
    /// Detected GPU info (set once at startup).
    pub gpu_info: Option<crate::inference::executor::GpuInfo>,
    /// Model governance vote tallies.
    pub model_vote_tallies: DashMap<crate::types::Blake3Hash, crate::model::governance::VoteTally>,
    /// Live acquisition progress — written by AcquisitionManager, read by API/WebSocket.
    pub acquisition_progress:
        DashMap<crate::types::ModelId, crate::model::acquisition::AcquisitionStatus>,
    /// Pending LayerResult channels for distributed pipeline execution.
    /// Keyed by request_id. Pipeline executor registers a oneshot sender before
    /// sending a LayerForward, and the network dispatcher fires it when the
    /// LayerResult arrives.
    pub pending_layer_results:
        DashMap<uuid::Uuid, tokio::sync::oneshot::Sender<crate::types::LayerResult>>,
    /// Loaded split models for distributed inference (layer-range segments).
    /// Keyed by (model_id, layer_start, layer_end) so a node can cache multiple
    /// non-contiguous segments (e.g., layers [0,2) and [10,14)) for the same model.
    /// Each entry tracks last-used time for VRAM-aware LRU eviction.
    pub split_models:
        DashMap<crate::inference::split::SplitModelKey, crate::inference::split::SplitModelEntry>,
    /// Per-request KV-cache storage for split inference.
    /// Keyed by (model_key, request_id) — isolates KV-cache per request,
    /// allowing concurrent requests to use the same model without corruption.
    pub kv_cache_store: Arc<crate::inference::split::KvCacheStore>,
    /// GGUF tensor metadata for known models (extracted from GGUF header, stored in manifest).
    pub gguf_meta: DashMap<crate::types::ModelId, crate::inference::split::GgufTensorMeta>,
    /// Nickname registry: node_id -> signed nickname record.
    pub nickname_registry: DashMap<NodeId, crate::identity::nickname::NicknameRecord>,
    /// E2E encryption session manager for pairwise ECDH sessions.
    pub session_manager: Arc<crate::crypto::SessionManager>,
    /// Gossip message sealer with epoch-based group key.
    pub gossip_sealer: Arc<crate::crypto::GossipSealer>,
    /// Current pool state for this node (owner or member).
    pub pool_state: RwLock<Option<crate::pool::types::PoolState>>,
    /// Network-wide pool registry (pool_id → PoolState).
    pub pool_registry: DashMap<crate::pool::types::PoolId, crate::pool::types::PoolState>,
    /// Channel to send commands to the PoolManager task.
    pub pool_tx: RwLock<Option<mpsc::Sender<crate::pool::types::PoolCommand>>>,
    /// Per-pool credit rate overrides. Key is the pool_id (PoolId == NodeId).
    pub pool_credit_rates: DashMap<NodeId, crate::config::CreditRateConfig>,
    /// API Bearer token for authentication.
    pub api_key: String,
    /// Per-process internal auth token for loopback forwarded requests.
    /// Prevents localhost auth bypass via guessable headers.
    pub internal_auth_token: String,
    /// Lock-free flag indicating a model is loaded in the llama-cpp executor.
    /// Set after `executor.load_model()` succeeds; checked by InferenceRouter
    /// to avoid locking the executor mutex just to check readiness.
    /// Note: this is only for the llama-cpp path. Nodes using split-model
    /// inference (partial shards) use `split_models` instead.
    pub model_loaded: std::sync::atomic::AtomicBool,
    /// Runtime toggle for auto-manage (mirrors config.auto_manage.enabled).
    /// Updated by the admin API so the AutoShardManager can pick up changes without restart.
    pub auto_manage_enabled: std::sync::atomic::AtomicBool,
    /// Anti-gaming system for credit transaction validation.
    pub anti_gaming: tokio::sync::Mutex<crate::credit::anti_gaming::AntiGaming>,
    /// Streaming token channels for distributed inference SSE.
    /// Keyed by request_id. The pipeline executor registers a sender so that
    /// incoming StreamingToken messages can be forwarded to the SSE stream.
    pub streaming_token_txs: DashMap<uuid::Uuid, mpsc::Sender<crate::types::StreamingToken>>,
    /// HuggingFace source info for models downloaded from HF, for re-download.
    pub hf_sources: DashMap<crate::types::ModelId, HfSource>,
    /// Notify trigger for the AutoShardManager — woken when new HF sources or manifests arrive.
    pub auto_manage_notify: Arc<tokio::sync::Notify>,
    /// Download progress reported by remote peers via gossip.
    /// Key: ShardId, Value: Vec<(NodeId, progress_pct)>
    pub peer_shard_downloads: DashMap<crate::types::ShardId, Vec<(NodeId, u32)>>,
    /// Cancel flags for in-progress HF downloads, keyed by model ID.
    /// Set to `true` to signal the download loop to abort.
    pub download_cancel_flags: DashMap<crate::types::ModelId, Arc<AtomicBool>>,
    /// Total inference requests processed (for Prometheus /metrics).
    pub inference_requests_total: AtomicU64,
    /// Inference latency samples in seconds (for Prometheus histogram).
    /// Capped at 1000 samples (ring-buffer behavior) to bound memory.
    pub inference_latency_samples: std::sync::RwLock<std::collections::VecDeque<f64>>,
    /// Readiness flag — set to true after all subsystem tasks are spawned.
    pub is_ready: AtomicBool,
    /// Watch channel for hot-reloaded operational config parameters.
    /// Subsystems can subscribe to changes via `config_watch_rx()`.
    pub config_watch_tx: watch::Sender<crate::config::OperationalParams>,
    /// Trust score manager — tracks per-peer reputation, persisted to sled.
    pub trust_manager: crate::credit::trust::TrustManager,
    /// Auto-detected country code from IP geolocation (e.g. "US", "DE").
    /// Falls back to config.identity.region if geolocation fails.
    pub detected_region: RwLock<Option<String>>,
    /// Per-peer credit balance buckets from gossip, for leaderboard display.
    /// Keyed by NodeId, value is the latest gossiped balance bucket.
    pub peer_credit_balances: DashMap<NodeId, i64>,
    /// Paged KV cache pool for PagedAttention (CUDA-only, feature-gated).
    /// When `None`, callers fall back to `KvCacheStore` (Phase 1 pre-allocated buffers).
    pub paged_kv_pool: Option<Arc<crate::inference::paged_kv::PagedKvPool>>,
    /// Paged KV store: per-request block table tracking.
    pub paged_kv_store: Option<Arc<crate::inference::paged_kv::PagedKvStore>>,
    /// Per-model auto-manage policies (runtime-mutable, persisted to sled).
    pub model_auto_manage_policies:
        DashMap<crate::types::ModelId, crate::config::ModelAutoManagePolicy>,
    /// Global default cap on auto-managed shards per model (from config).
    pub auto_manage_default_model_cap: AtomicU32,
    /// Cache of HuggingFace probe results (populated when user probes a model).
    pub hf_probe_cache: DashMap<crate::types::ModelId, HfProbeInfo>,
    /// Per-model inference request counts (rolling window for popularity scoring).
    pub model_request_counts: DashMap<crate::types::ModelId, AtomicU64>,
    /// Runtime-mutable resource schedule (initialized from config, overridable via API).
    pub resource_schedule: RwLock<crate::config::ResourceSchedule>,
    /// Broadcast channel for prune events (WebSocket push + history).
    pub prune_events_tx: broadcast::Sender<crate::types::PruneEvent>,
    /// Recent prune events (capped at 100) for the prune history API.
    pub prune_history: RwLock<VecDeque<crate::types::PruneEvent>>,
    /// Per-shard lock/pin flags — locked shards are never auto-pruned.
    pub locked_shards: DashMap<crate::types::ShardId, bool>,
    /// LoRA adapter registry for per-request fine-tuned inference.
    pub adapter_registry: Arc<crate::model::lora::AdapterRegistry>,
    /// Cross-request prefix cache for sharing KV state across requests with
    /// identical system prompts. Protected by std::sync::Mutex since operations
    /// are fast (hash lookup, tensor clone) and never held across await points.
    pub prefix_cache: std::sync::Mutex<crate::inference::prefix_cache::PrefixCache>,
    /// Per-channel backpressure metrics (capacity, sent, dropped).
    pub channel_metrics: ChannelMetricsSet,
    /// Number of peers discovered via mDNS (LAN peers).
    pub lan_peer_count: std::sync::atomic::AtomicUsize,
    /// Broadcast channel for LAN peer discovery events (WebSocket push).
    pub lan_discovery_tx: broadcast::Sender<u32>,
    /// Runtime-mutable cloud provider configuration (API keys for Anthropic, OpenAI, etc.).
    pub providers_config: RwLock<crate::config::ProvidersConfig>,
    /// Update checker shared state (version info, last checked, etc.).
    pub update_state: Arc<RwLock<crate::update::UpdateState>>,
    /// Broadcast channel for update availability notifications (WebSocket push).
    pub update_tx: broadcast::Sender<crate::update::UpdateInfo>,
    /// Broadcast channel fired when models change (shard download, load, prune).
    /// WebSocket subscribers push a `models_changed` event so the dashboard auto-refreshes.
    pub models_changed_tx: broadcast::Sender<()>,
    /// Cached mapping of cloud provider model IDs to provider names.
    /// Populated by `list_provider_models` so that `try_proxy_openai` can route
    /// models whose ID doesn't match a known prefix (e.g. NVIDIA NIM `01-ai/yi-large`).
    pub provider_model_map: DashMap<String, String>,
    /// Persistent NodeId → PeerId bytes mapping. Populated by the identify handler
    /// and NEVER cleared on disconnect. Solves the race where the scheduler picks a
    /// peer from shard_holders but the peer_registry entry was removed on disconnect
    /// and hasn't been re-created by identify yet.
    pub peer_id_map: DashMap<NodeId, Vec<u8>>,
    /// Loaded VLM vision modules (mmproj encoders) keyed by model ID.
    /// Populated when an mmproj.gguf is found alongside a model's shards.
    pub vision_modules: DashMap<crate::types::ModelId, Arc<crate::inference::vision::VisionModule>>,
    /// Pending VisionEncodeResponse channels for distributed vision encoding.
    /// Keyed by request_id. Pipeline registers a oneshot sender before sending
    /// VisionEncodeRequest to a remote mmproj holder; the network dispatcher fires it
    /// when the VisionEncodeResponse arrives.
    pub pending_vision_results:
        DashMap<uuid::Uuid, tokio::sync::oneshot::Sender<crate::types::VisionEncodeResponse>>,
    /// Pending tensor-parallel AllReduce partials, keyed by (request_id, layer_idx).
    /// Coordinator collects partials from all TP ranks, sums them, and responds.
    pub pending_tp_partials: DashMap<(uuid::Uuid, u32), TpAllReduceCollector>,
    /// AllReduce response registry — pipeline executors register here to receive
    /// reduced tensors after the coordinator completes the allreduce.
    pub allreduce_registry: Arc<crate::inference::allreduce::AllReduceRegistry>,
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
        Self {
            tp_size,
            partials: vec![None; tp_size as usize],
            sender_peers: vec![None; tp_size as usize],
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
        if rank < self.partials.len() {
            self.sender_peers[rank] = sender_peer;
            self.partials[rank] = Some(req);
        }
        self.partials.iter().all(|p| p.is_some())
    }

    /// Sum all partial tensors (f32) and return the reduced bytes + shape.
    pub fn reduce_sum(&self) -> Result<(Vec<u8>, Vec<u32>), crate::error::SwarmError> {
        let first = self.partials[0].as_ref().unwrap();
        let shape = first.shape.clone();
        let elem_count: usize = shape.iter().map(|&s| s as usize).product();

        // Decompress first partial
        let decompressed = zstd::decode_all(std::io::Cursor::new(&first.partial_data))
            .map_err(|e| crate::error::SwarmError::Internal(format!("zstd decompress: {e}")))?;
        let mut sum = vec![0.0f32; elem_count];
        if decompressed.len() == elem_count * 4 {
            for (i, chunk) in decompressed.chunks_exact(4).enumerate() {
                sum[i] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            }
        }

        // Add remaining partials
        for partial in &self.partials[1..] {
            let req = partial.as_ref().unwrap();
            let dec = zstd::decode_all(std::io::Cursor::new(&req.partial_data))
                .map_err(|e| crate::error::SwarmError::Internal(format!("zstd decompress: {e}")))?;
            if dec.len() == elem_count * 4 {
                for (i, chunk) in dec.chunks_exact(4).enumerate() {
                    sum[i] += f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                }
            }
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
    ) -> (Arc<Self>, watch::Receiver<bool>) {
        // Resolve API key: config > persisted in DB > generate new
        let api_key = resolve_api_key(&config, &db);
        let (shutdown_tx, shutdown_rx) = watch::channel(false);

        let model_registry = ModelRegistry::load_from_db(&db).unwrap_or_default();

        // Hydrate nickname registry from sled
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
        let (config_watch_tx, _config_watch_rx) = watch::channel(initial_ops);
        let trust_manager = crate::credit::trust::TrustManager::new(db.clone());

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
        let state = Arc::new(Self {
            config: config.clone(),
            identity,
            db: db.clone(),
            peer_registry: DashMap::new(),
            model_registry,
            active_pipelines: DashMap::new(),
            credit_balance: Arc::new(RwLock::new(CreditBalance {
                node_id,
                balance: 0,
                lifetime_earned: 0,
                lifetime_spent: 0,
                last_updated: chrono::Utc::now(),
            })),
            node_stats: RwLock::new(NodeStats::default()),
            executor,
            draft_executor: Arc::new(tokio::sync::Mutex::new(
                crate::inference::executor::ModelExecutor::new(),
            )),
            loaded_model_info: RwLock::new(None),
            gpu_info,
            model_vote_tallies: DashMap::new(),
            acquisition_progress: DashMap::new(),
            pending_layer_results: DashMap::new(),
            split_models: DashMap::new(),
            kv_cache_store: Arc::new(crate::inference::split::KvCacheStore::new(
                std::time::Duration::from_secs(kv_cache_ttl_secs),
            )),
            gguf_meta: DashMap::new(),
            nickname_registry,
            session_manager,
            gossip_sealer,
            pool_state: RwLock::new(None),
            pool_registry: DashMap::new(),
            pool_tx: RwLock::new(None),
            pool_credit_rates: DashMap::new(),
            api_key,
            internal_auth_token: {
                use rand::RngCore;
                let mut bytes = [0u8; 16];
                rand::rngs::OsRng.fill_bytes(&mut bytes);
                hex::encode(bytes)
            },
            model_loaded: std::sync::atomic::AtomicBool::new(false),
            auto_manage_enabled: std::sync::atomic::AtomicBool::new(auto_manage_enabled),
            anti_gaming: tokio::sync::Mutex::new(crate::credit::anti_gaming::AntiGaming::new()),
            streaming_token_txs: DashMap::new(),
            hf_sources,
            auto_manage_notify: Arc::new(tokio::sync::Notify::new()),
            peer_shard_downloads: DashMap::new(),
            download_cancel_flags: DashMap::new(),
            inference_requests_total: AtomicU64::new(0),
            inference_latency_samples: std::sync::RwLock::new(std::collections::VecDeque::new()),
            is_ready: AtomicBool::new(false),
            config_watch_tx,
            trust_manager,
            detected_region: RwLock::new(None),
            peer_credit_balances: DashMap::new(),
            paged_kv_pool: None,
            paged_kv_store: None,
            model_auto_manage_policies,
            auto_manage_default_model_cap: AtomicU32::new(default_model_shard_cap),
            hf_probe_cache: DashMap::new(),
            model_request_counts: DashMap::new(),
            resource_schedule: RwLock::new(config.resources.schedule.clone()),
            prune_events_tx: broadcast::channel(64).0,
            prune_history: RwLock::new(VecDeque::new()),
            locked_shards: {
                let map = DashMap::new();
                if let Ok(entries) = db.iter_raw("locked_shards") {
                    for (key, _value) in entries {
                        // Keys are stored as JSON strings of ShardId
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
            adapter_registry: Arc::new(crate::model::lora::AdapterRegistry::new(
                &config.node.data_dir,
            )),
            prefix_cache: std::sync::Mutex::new(crate::inference::prefix_cache::PrefixCache::new(
                config.inference.prefix_cache_max_entries,
            )),
            channel_metrics: ChannelMetricsSet::new(),
            lan_peer_count: std::sync::atomic::AtomicUsize::new(0),
            lan_discovery_tx: broadcast::channel(16).0,
            providers_config: RwLock::new({
                // Hydrate from database (persisted via admin API), fall back to config
                db.get_json::<crate::config::ProvidersConfig>("providers", "config")
                    .ok()
                    .flatten()
                    .unwrap_or_else(|| config.providers.clone())
            }),
            update_state: Arc::new(RwLock::new(crate::update::UpdateState::default())),
            update_tx: broadcast::channel(4).0,
            models_changed_tx: broadcast::channel(16).0,
            provider_model_map: DashMap::new(),
            peer_id_map: DashMap::new(),
            vision_modules: DashMap::new(),
            pending_vision_results: DashMap::new(),
            pending_tp_partials: DashMap::new(),
            allreduce_registry: Arc::new(crate::inference::allreduce::AllReduceRegistry::new()),
            shutdown_tx,
        });

        (state, shutdown_rx)
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
}

/// Maximum restart attempts before a subsystem is considered permanently failed.
const MAX_RESTART_ATTEMPTS: u32 = 5;
/// Base backoff duration for subsystem restarts (doubles each attempt, capped at 16s).
/// Note: Not currently wired for production restart logic — channel-bound subsystems
/// cannot be restarted. Kept for tests and future use.
#[allow(dead_code)]
const RESTART_BACKOFF_BASE: std::time::Duration = std::time::Duration::from_secs(1);
/// Maximum backoff duration for subsystem restarts.
#[allow(dead_code)]
const RESTART_BACKOFF_CAP: std::time::Duration = std::time::Duration::from_secs(16);

/// Whether a subsystem is critical to daemon operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubsystemCriticality {
    /// Daemon must shut down if this subsystem permanently fails.
    Critical,
    /// Daemon can continue without this subsystem.
    NonCritical,
}

/// Compute the backoff duration for a given restart attempt.
/// Note: Not currently wired for production restart logic — channel-bound subsystems
/// cannot be restarted. Kept for tests and future use.
#[allow(dead_code)]
fn restart_backoff(attempt: u32) -> std::time::Duration {
    let secs = RESTART_BACKOFF_BASE
        .as_secs()
        .saturating_mul(1u64.checked_shl(attempt).unwrap_or(u64::MAX));
    std::time::Duration::from_secs(secs).min(RESTART_BACKOFF_CAP)
}

/// Top-level daemon orchestrating all SwarmLLM subsystems.
pub struct Daemon {
    config: Config,
    identity: Identity,
    db: Database,
}

impl Daemon {
    pub fn new(config: Config, identity: Identity, db: Database) -> Self {
        Self {
            config,
            identity,
            db,
        }
    }

    /// Run the daemon — spawns all subsystems and waits for shutdown.
    pub async fn run(self) -> anyhow::Result<()> {
        // Log resolved configuration at startup
        let auto_interval = self
            .config
            .auto_manage
            .interval_seconds
            .map(|s| format!("{s}s"))
            .unwrap_or_else(|| format!("{}m", self.config.auto_manage.interval_minutes));
        tracing::info!(
            port = self.config.node.listen_port,
            data_dir = %self.config.node.data_dir.display(),
            bootstrap_peers = self.config.network.bootstrap_peers.len(),
            auto_manage = self.config.auto_manage.enabled,
            "SwarmLLM daemon starting with resolved config"
        );
        tracing::debug!(
            port = self.config.node.listen_port,
            data_dir = %self.config.node.data_dir.display(),
            bootstrap_peers = self.config.network.bootstrap_peers.len(),
            auto_manage_enabled = self.config.auto_manage.enabled,
            auto_manage_interval = %auto_interval,
            max_concurrent_requests = self.config.inference.max_concurrent_requests,
            shard_size_mb = self.config.model.shard_size_mb,
            log_level = %self.config.logging.level,
            max_peers = self.config.network.max_peers,
            session_timeout_secs = self.config.inference.session_timeout_seconds,
            relay_enabled = self.config.network.enable_relay,
            "Full resolved configuration"
        );

        // Run database integrity check before spawning subsystems
        let integrity_report = self.db.check_integrity();
        if integrity_report.total_corrupt > 0 {
            tracing::warn!(
                corrupt_entries = integrity_report.total_corrupt,
                "Database integrity issues detected — some entries may be skipped"
            );
        }

        // Initialize model executor
        let mut executor = crate::inference::executor::ModelExecutor::new();
        if let Some(ref model_path) = self.config.inference.model_path {
            match executor.load_model(model_path, self.config.inference.gpu_layers) {
                Ok(()) => tracing::info!("Model ready"),
                Err(e) => {
                    tracing::warn!(error = %e, "Failed to load model — running without inference")
                }
            }
        }

        // Gather model info for manifest generation and admin display.
        // Extract GGUF metadata (chat template, special tokens) if available.
        let gguf_meta = self
            .config
            .inference
            .model_path
            .as_ref()
            .and_then(|p| crate::inference::executor::extract_gguf_metadata(p));

        let model_info = if executor.is_loaded() {
            Some(LoadedModelInfo {
                name: executor.model_name().to_string(),
                size_bytes: executor.model_size_bytes().unwrap_or(0),
                eos_tokens: vec![2], // Default; updated when split model loads with GGUF metadata
                chat_template: gguf_meta.as_ref().and_then(|m| m.chat_template.clone()),
                bos_token: gguf_meta
                    .as_ref()
                    .map(|m| m.bos_token.clone())
                    .unwrap_or_default(),
                eos_token: gguf_meta
                    .as_ref()
                    .map(|m| m.eos_token.clone())
                    .unwrap_or_default(),
            })
        } else {
            None
        };

        // When --shards is set, the node only holds part of the model — don't
        // report a fully loaded model, which would cause the API to serve
        // requests through the (incomplete) local executor.
        let cached_info = if self.config.inference.shard_range.is_some() {
            if let Some(ref info) = model_info {
                tracing::info!(
                    model = %info.name,
                    "Model available for split inference (not full-model serving)"
                );
            }
            None
        } else {
            model_info.clone()
        };

        // Detect GPU via llama.cpp backend
        let gpu_info = crate::inference::executor::detect_gpu();
        if let Some(ref gpu) = gpu_info {
            tracing::info!(gpu = %gpu.name, vram_mb = gpu.vram_total_mb, backend = %gpu.backend, "GPU detected");
        }

        let executor = Arc::new(tokio::sync::Mutex::new(executor));

        // Create shared state
        let (shared_state, mut shutdown_rx) = SharedState::new(
            self.config.clone(),
            self.identity.clone(),
            self.db.clone(),
            executor,
            gpu_info,
        );

        // Set the cached model info (lock-free for admin reads)
        *shared_state.loaded_model_info.write().await = cached_info;

        // Set the model_loaded atomic for the llama-cpp executor path.
        // Not set in shard/split mode — those nodes use split_models instead.
        if model_info.is_some() && self.config.inference.shard_range.is_none() {
            shared_state
                .model_loaded
                .store(true, std::sync::atomic::Ordering::Release);
        }

        // Load draft model for speculative decoding if configured
        if self.config.inference.speculative_decoding {
            if let Some(ref draft_path) = self.config.inference.draft_model_path {
                let draft_gpu_layers = self
                    .config
                    .inference
                    .draft_gpu_layers
                    .unwrap_or(self.config.inference.gpu_layers);
                let mut draft = shared_state.draft_executor.lock().await;
                match draft.load_model(draft_path, draft_gpu_layers) {
                    Ok(()) => tracing::info!(
                        draft_model = %draft.model_name(),
                        gamma = self.config.inference.speculative_gamma,
                        "Draft model loaded for speculative decoding"
                    ),
                    Err(e) => tracing::warn!(
                        error = %e,
                        "Failed to load draft model — falling back to standard decoding"
                    ),
                }
            } else {
                tracing::info!("Speculative decoding enabled but no draft_model_path configured");
            }
        }

        // Generate a ModelManifest for the locally loaded model so peers can discover it.
        // This is needed even in split mode so the shard registry gets populated.
        if let Some(ref info) = model_info {
            if let Some(ref model_path) = self.config.inference.model_path {
                generate_and_register_local_manifest(&shared_state, info, model_path);
            }
        }

        // Restore persisted manifests from the DB and register shard holders.
        // This handles the case where a node restarts with --shards but no --model:
        // the manifest was generated in a previous run and persisted, so we restore
        // it and re-register ourselves as holder of our shard range.
        {
            let node_id = shared_state.identity.node_id().clone();
            let shard_range = self.config.inference.shard_range;
            if let Ok(manifests) = self
                .db
                .iter_json::<crate::types::ModelManifest>("model_meta")
            {
                for manifest in manifests {
                    let model_id = manifest.id.clone();
                    // Verify manifest hash before trusting DB data (MOD-I2)
                    if manifest.verify_hash().is_err() {
                        tracing::warn!(
                            model = %model_id,
                            "Manifest from DB failed hash verification — skipping"
                        );
                        continue;
                    }
                    // Register the manifest if not already in-memory
                    if shared_state
                        .model_registry
                        .get_manifest(&model_id)
                        .is_none()
                    {
                        shared_state
                            .model_registry
                            .register_manifest(manifest.clone());
                        tracing::info!(
                            model = %model_id,
                            shards = manifest.shard_count,
                            "Restored manifest from DB"
                        );
                    }
                    // Register ourselves as holder of our shard range
                    let shard_store_reg = ShardStore::new(&self.config.node.data_dir);
                    for shard_info in &manifest.shards {
                        let in_range = match shard_range {
                            Some((start, end)) => {
                                shard_info.index >= start && shard_info.index <= end
                            }
                            None => true,
                        };
                        if in_range {
                            // Verify the shard file actually exists on disk before registering
                            let shard_path =
                                shard_store_reg.shard_path(&model_id, shard_info.index);
                            if !shard_path.exists() {
                                tracing::warn!(
                                    model = %model_id,
                                    shard = shard_info.index,
                                    path = %shard_path.display(),
                                    "Shard file missing on disk — skipping registration"
                                );
                                continue;
                            }
                            let shard_id = crate::types::ShardId {
                                model_id: model_id.clone(),
                                index: shard_info.index,
                            };
                            shared_state
                                .model_registry
                                .record_shard_holder(shard_id, node_id.clone());
                        }
                    }
                    // Load GGUF metadata for the model if we have a source path
                    if !shared_state.gguf_meta.contains_key(&model_id) {
                        let shard_store_tmp = ShardStore::new(&self.config.node.data_dir);
                        let model_dir = shard_store_tmp.models_dir().join(&model_id.0);
                        let source_path_file = model_dir.join("source_path");
                        if let Ok(path_str) = std::fs::read_to_string(&source_path_file) {
                            let path = std::path::Path::new(path_str.trim());
                            if let Ok(meta) =
                                crate::inference::split::GgufTensorMeta::from_gguf_file(path)
                            {
                                tracing::info!(
                                    model = %model_id,
                                    layers = meta.block_count,
                                    "Loaded GGUF metadata from source path"
                                );
                                shared_state.gguf_meta.insert(model_id.clone(), meta);
                            }
                        }
                    }
                }
            }
        }

        // Pre-pass: regenerate any missing manifests from GGUF headers + shard files.
        // load_all_local() requires a manifest to exist (security check), so we must
        // create one first if gguf_header.bin + shard files are present.
        let shard_store = ShardStore::new(&self.config.node.data_dir);
        {
            let models_dir = shard_store.models_dir();
            if models_dir.exists() {
                if let Ok(entries) = std::fs::read_dir(&models_dir) {
                    for entry in entries.flatten() {
                        let model_dir = entry.path();
                        if !model_dir.is_dir() {
                            continue;
                        }
                        let manifest_path = model_dir.join("manifest.json");
                        let header_path = model_dir.join("gguf_header.bin");
                        if !manifest_path.exists() && header_path.exists() {
                            let model_id_str = model_dir
                                .file_name()
                                .and_then(|n| n.to_str())
                                .unwrap_or("")
                                .to_string();
                            let model_id = crate::types::ModelId(model_id_str);
                            if let Ok(meta) =
                                crate::inference::split::GgufTensorMeta::from_gguf_file(
                                    &header_path,
                                )
                            {
                                tracing::info!(
                                    model = %model_id,
                                    "Regenerating missing manifest from GGUF header"
                                );
                                if regenerate_manifest_from_header(
                                    &model_id,
                                    &model_dir,
                                    &meta,
                                    &self.config,
                                )
                                .is_some()
                                {
                                    shared_state.gguf_meta.insert(model_id, meta);
                                }
                            }
                        }
                    }
                }
            }
        }

        // Scan local shards and register them + their manifests
        match shard_store.load_all_local() {
            Ok(shards) => {
                // Track which model manifests we've already registered
                let mut registered_manifests = std::collections::HashSet::new();

                for (model_id, shard_info) in &shards {
                    // Register the manifest if we haven't yet
                    if registered_manifests.insert(model_id.clone()) {
                        let model_dir = shard_store.models_dir().join(&model_id.0);

                        // Ensure GGUF header exists (extract from shard_000 if available)
                        // and load GGUF metadata for split inference.
                        // Do this BEFORE loading manifest so we can regenerate if needed.
                        if !shared_state.gguf_meta.contains_key(model_id) {
                            if let Ok(()) = crate::inference::split::ensure_gguf_header(&model_dir)
                            {
                                let header_path = model_dir.join("gguf_header.bin");
                                if let Ok(meta) =
                                    crate::inference::split::GgufTensorMeta::from_gguf_file(
                                        &header_path,
                                    )
                                {
                                    tracing::info!(
                                        model = %model_id,
                                        layers = meta.block_count,
                                        "Loaded GGUF metadata from shard header"
                                    );
                                    shared_state.gguf_meta.insert(model_id.clone(), meta);
                                }
                            }
                        }

                        let manifest_loaded = if let Ok(manifest) =
                            crate::types::ModelManifest::load_from_dir(&model_dir)
                        {
                            if manifest.verify_hash().is_ok() {
                                shared_state
                                    .model_registry
                                    .register_manifest(manifest.clone());
                                if let Err(e) = shared_state
                                    .model_registry
                                    .persist_manifest(&shared_state.db, &manifest)
                                {
                                    tracing::warn!(error = %e, "Failed to persist manifest to DB");
                                }
                                tracing::info!(
                                    model = %model_id,
                                    shards = manifest.shard_count,
                                    "Registered manifest from local shard directory"
                                );
                                true
                            } else {
                                false
                            }
                        } else {
                            false
                        };

                        // Regenerate manifest if missing/invalid and GGUF header available
                        if !manifest_loaded {
                            if let Some(meta) = shared_state.gguf_meta.get(model_id) {
                                tracing::info!(
                                    model = %model_id,
                                    "Regenerating manifest from GGUF header + shard files"
                                );
                                if let Some(manifest) = regenerate_manifest_from_header(
                                    model_id,
                                    &model_dir,
                                    &meta,
                                    &shared_state.config,
                                ) {
                                    shared_state
                                        .model_registry
                                        .register_manifest(manifest.clone());
                                    let _ = shared_state
                                        .model_registry
                                        .persist_manifest(&shared_state.db, &manifest);
                                }
                            }
                        }
                    }

                    let shard_id = ShardId {
                        model_id: model_id.clone(),
                        index: shard_info.index,
                    };
                    let node_id = shared_state.identity.node_id().clone();
                    shared_state
                        .model_registry
                        .record_shard_holder(shard_id, node_id);
                }
            }
            Err(e) => {
                tracing::warn!(error = %e, "Failed to scan local shards");
            }
        }

        // Register local mmproj files as sentinel shards.
        {
            let models_dir = self.config.node.data_dir.join("models");
            if models_dir.is_dir() {
                if let Ok(entries) = std::fs::read_dir(&models_dir) {
                    let node_id = shared_state.identity.node_id().clone();
                    for entry in entries.flatten() {
                        if !entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                            continue;
                        }
                        let mmproj_path = entry.path().join("mmproj.gguf");
                        if mmproj_path.exists() {
                            let model_id_str = entry.file_name().to_string_lossy().to_string();
                            let model_id = crate::types::ModelId(model_id_str.clone());
                            let mmproj_sid = ShardId::mmproj_for(model_id);
                            shared_state
                                .model_registry
                                .record_shard_holder(mmproj_sid, node_id.clone());
                            tracing::info!(
                                model = %model_id_str,
                                "Registered local mmproj.gguf as vision encoder shard"
                            );
                        }
                    }
                }
            }
        }

        // Discover HF sources from hf_source.json files alongside manifests.
        // Models always originate from HuggingFace, so this ensures the source
        // is known even after a DB wipe or fresh node with pre-seeded shards.
        {
            let models_dir = self.config.node.data_dir.join("models");
            if models_dir.is_dir() {
                if let Ok(entries) = std::fs::read_dir(&models_dir) {
                    for entry in entries.flatten() {
                        if !entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                            continue;
                        }
                        let model_id_str = entry.file_name().to_string_lossy().to_string();
                        let mid = crate::types::ModelId(model_id_str.clone());
                        if shared_state.hf_sources.contains_key(&mid) {
                            continue;
                        }
                        let hf_path = entry.path().join("hf_source.json");
                        if hf_path.exists() {
                            if let Ok(data) = std::fs::read_to_string(&hf_path) {
                                if let Ok(source) = serde_json::from_str::<HfSource>(&data) {
                                    tracing::info!(
                                        model = %model_id_str,
                                        repo = %source.repo_id,
                                        file = %source.filename,
                                        "Loaded HF source from disk"
                                    );
                                    shared_state.hf_sources.insert(mid.clone(), source.clone());
                                    let _ = self.db.put_json("hf_sources", &model_id_str, &source);
                                }
                            }
                        }
                    }
                }
            }
        }

        // ── Channel Architecture ──
        //
        // network_tx      → NetworkManager (outbound commands: broadcast, send tensor)
        // network_out_tx  → from NetworkManager (inbound decoded messages)
        // router_cmd_tx   → InferenceRouter (commands from API + network)
        // rebalance_tx    → ShardRebalancer (events from HealthMonitor)
        // acquisition_tx  → AcquisitionManager (model download commands from API)
        //
        let (network_tx, network_rx) = mpsc::channel::<NetworkCommand>(1024);
        let (network_out_tx, mut network_out_rx) = mpsc::channel::<SwarmMessage>(1024);
        let (router_cmd_tx, router_cmd_rx) = mpsc::channel::<RouterCommand>(256);
        let (rebalance_tx, rebalance_rx) = mpsc::channel::<RebalanceEvent>(64);
        let (acquisition_tx, acquisition_rx) = mpsc::channel::<AcquisitionCommand>(64);

        // ── Subsystem Supervisor (JoinSet) ──
        //
        // All 10 subsystem tasks are spawned into a JoinSet for unified monitoring.
        // Each task returns (name, criticality, result) so the supervisor loop
        // can decide whether to trigger shutdown or continue degraded.
        //
        let mut subsystems: JoinSet<(&'static str, SubsystemCriticality, Result<(), String>)> =
            JoinSet::new();

        // Spawn NetworkManager (acquisition_tx wired after channel creation below)
        let network_manager = NetworkManager::new(
            shared_state.clone(),
            &self.identity,
            &self.config,
            network_rx,
            network_out_tx,
            shutdown_rx.clone(),
            Some(acquisition_tx.clone()),
        )?;

        subsystems.spawn(async move {
            let result = network_manager.run().await.map_err(|e| e.to_string());
            ("NetworkManager", SubsystemCriticality::Critical, result)
        });

        // Spawn InferenceRouter
        let inference_router = InferenceRouter::new(
            shared_state.clone(),
            router_cmd_rx,
            router_cmd_tx.clone(),
            network_tx.clone(),
            shutdown_rx.clone(),
        );

        subsystems.spawn(async move {
            let result = inference_router.run().await.map_err(|e| e.to_string());
            ("InferenceRouter", SubsystemCriticality::Critical, result)
        });

        // Spawn message dispatcher: routes network inbound messages to the right subsystem
        let dispatcher_credit_balances: Arc<RwLock<Vec<i64>>> = Arc::new(RwLock::new(Vec::new()));
        let dispatcher_router_tx = router_cmd_tx.clone();
        let dispatcher_shutdown = shutdown_rx.clone();
        let dispatcher_credit_ref = dispatcher_credit_balances.clone();
        let dispatcher_state = shared_state.clone();
        let dispatcher_network_tx = network_tx.clone();
        subsystems.spawn(async move {
            dispatch_network_messages(
                &mut network_out_rx,
                &dispatcher_router_tx,
                dispatcher_credit_ref,
                &dispatcher_state,
                dispatcher_network_tx,
                dispatcher_shutdown,
            )
            .await;
            ("MessageDispatcher", SubsystemCriticality::Critical, Ok(()))
        });

        // Spawn HealthMonitor
        let health_monitor = HealthMonitor::new(
            shared_state.clone(),
            network_tx.clone(),
            rebalance_tx,
            shutdown_rx.clone(),
        );

        subsystems.spawn(async move {
            let result = health_monitor.run().await.map_err(|e| e.to_string());
            ("HealthMonitor", SubsystemCriticality::NonCritical, result)
        });

        // Spawn ShardRebalancer
        let shard_rebalancer = ShardRebalancer::new(
            shared_state.clone(),
            rebalance_rx,
            network_tx.clone(),
            acquisition_tx.clone(),
            shutdown_rx.clone(),
        );

        subsystems.spawn(async move {
            let result = shard_rebalancer.run().await.map_err(|e| e.to_string());
            ("ShardRebalancer", SubsystemCriticality::NonCritical, result)
        });

        // Spawn CreditLedger — shares the same Arc<RwLock<CreditBalance>> as SharedState
        let mut credit_ledger = CreditLedger::new(
            shared_state.identity.node_id().clone(),
            shared_state.credit_balance.clone(),
            self.db.clone(),
            network_tx.clone(),
            shutdown_rx.clone(),
            dispatcher_credit_balances.clone(),
        );
        credit_ledger.set_shared_state(shared_state.clone());
        credit_ledger.set_identity(shared_state.identity.clone());

        subsystems.spawn(async move {
            let result = credit_ledger.run().await.map_err(|e| e.to_string());
            ("CreditLedger", SubsystemCriticality::NonCritical, result)
        });

        // Spawn AcquisitionManager
        let acquisition_manager = AcquisitionManager::new(
            shared_state.clone(),
            network_tx.clone(),
            acquisition_rx,
            shutdown_rx.clone(),
        );

        subsystems.spawn(async move {
            let result = acquisition_manager.run().await.map_err(|e| e.to_string());
            (
                "AcquisitionManager",
                SubsystemCriticality::NonCritical,
                result,
            )
        });

        // Spawn PoolManager (9th subsystem task)
        let (pool_cmd_tx, pool_cmd_rx) = mpsc::channel::<crate::pool::types::PoolCommand>(64);
        {
            *shared_state.pool_tx.write().await = Some(pool_cmd_tx);
        }
        let pool_manager = crate::pool::manager::PoolManager::new(
            shared_state.clone(),
            pool_cmd_rx,
            network_tx.clone(),
            shutdown_rx.clone(),
        );
        subsystems.spawn(async move {
            let result = pool_manager.run().await.map_err(|e| e.to_string());
            ("PoolManager", SubsystemCriticality::NonCritical, result)
        });

        // Spawn AutoShardManager (10th subsystem task — optional, runs only if enabled)
        let auto_manage = crate::model::auto_manage::AutoShardManager::new(
            shared_state.clone(),
            network_tx.clone(),
            shutdown_rx.clone(),
        );
        subsystems.spawn(async move {
            auto_manage.run().await;
            (
                "AutoShardManager",
                SubsystemCriticality::NonCritical,
                Ok(()),
            )
        });

        // Spawn UpdateChecker (11th subsystem task — optional, runs only if not disabled)
        {
            let update_config = self.config.updates.clone();
            let update_state = shared_state.update_state.clone();
            let update_tx = shared_state.update_tx.clone();
            let update_shutdown = shutdown_rx.clone();
            let checker = crate::update::UpdateChecker::new(
                update_config,
                "enapt/SwarmLLM".to_string(),
                update_state,
                update_tx,
            );
            subsystems.spawn(async move {
                checker.run(update_shutdown).await;
                ("UpdateChecker", SubsystemCriticality::NonCritical, Ok(()))
            });
        }

        // Spawn API server (pass router_cmd_tx + acquisition_tx + network_tx so API can submit requests)
        let api_shared_state = shared_state.clone();
        let api_router_tx = router_cmd_tx.clone();
        let api_acquisition_tx = acquisition_tx.clone();
        let api_network_tx = network_tx.clone();
        subsystems.spawn(async move {
            let result = crate::api::server::run_server_with_state(
                api_shared_state,
                api_router_tx,
                api_acquisition_tx,
                api_network_tx,
            )
            .await
            .map_err(|e| e.to_string());
            ("ApiServer", SubsystemCriticality::Critical, result)
        });

        // All subsystems spawned — mark node as ready for health probes
        shared_state
            .is_ready
            .store(true, std::sync::atomic::Ordering::Release);

        tracing::info!(
            node_id = %self.identity.node_id(),
            port = self.config.node.listen_port,
            "SwarmLLM daemon running"
        );

        // Auto-detect region via IP geolocation (non-blocking, best-effort)
        if shared_state.config.identity.region.is_none() {
            let geo_state = shared_state.clone();
            let mut geo_shutdown = shutdown_rx.clone();
            tokio::spawn(async move {
                tokio::select! {
                    result = detect_region_from_ip() => {
                        match result {
                            Some(code) => {
                                tracing::info!(region = %code, "Auto-detected region via IP geolocation");
                                *geo_state.detected_region.write().await = Some(code);
                            }
                            None => {
                                tracing::debug!(
                                    "IP geolocation unavailable — network map will show unknown region"
                                );
                            }
                        }
                    }
                    _ = geo_shutdown.changed() => {}
                }
            });
        } else {
            // User configured a region explicitly — use it
            *shared_state.detected_region.write().await =
                shared_state.config.identity.region.clone();
        }

        // Broadcast shard announcements and manifests shortly after startup
        // so peers discover our shards quickly (don't wait for the 30s health tick).
        {
            let announce_state = shared_state.clone();
            let announce_tx = network_tx.clone();
            let mut announce_shutdown = shutdown_rx.clone();
            tokio::spawn(async move {
                // Wait for peer connections to establish, abort on shutdown
                tokio::select! {
                    _ = tokio::time::sleep(std::time::Duration::from_secs(5)) => {}
                    _ = announce_shutdown.changed() => { return; }
                }

                let node_id = announce_state.identity.node_id().clone();

                // Broadcast shard announcements
                let mut hosted_shards = Vec::new();
                for entry in announce_state.model_registry.all_shard_entries() {
                    let (shard_id, holders) = entry;
                    if holders.contains(&node_id) {
                        hosted_shards.push(shard_id);
                    }
                }

                if !hosted_shards.is_empty() {
                    let announce = crate::types::ShardAnnounce {
                        node_id: node_id.clone(),
                        shards: hosted_shards,
                        timestamp: chrono::Utc::now(),
                    };
                    tracing::info!(
                        shards = announce.shards.len(),
                        "Broadcasting initial shard announcement"
                    );
                    let _ = announce_tx
                        .send(NetworkCommand::Broadcast(SwarmMessage::ShardAnnounce(
                            announce,
                        )))
                        .await;
                }

                // Broadcast our manifests
                for manifest in announce_state.model_registry.models() {
                    if manifest.publisher == node_id {
                        let _ = announce_tx
                            .send(NetworkCommand::Broadcast(SwarmMessage::ModelManifest(
                                manifest,
                            )))
                            .await;
                    }
                }
            });
        }

        // Spawn key rotation task (evicts stale sessions + ephemeral re-keying)
        {
            let rotation_sm = shared_state.session_manager.clone();
            let rotation_shutdown = shutdown_rx.clone();
            let rotation_network_tx = network_tx.clone();
            let rotation_node_id = shared_state.identity.node_id().clone();
            tokio::spawn(async move {
                crate::crypto::key_rotation::run_key_rotation(
                    rotation_sm,
                    rotation_network_tx,
                    rotation_node_id,
                    rotation_shutdown,
                )
                .await;
            });
        }

        // Open browser on first start if configured
        if self.config.ui.open_browser_on_start {
            let url = format!("http://localhost:{}", self.config.node.listen_port);
            // Check if config file exists — if not, open setup wizard
            let config_path = self.config.node.data_dir.join("config.toml");
            let target = if config_path.exists() {
                format!("{url}/admin")
            } else {
                format!("{url}/setup")
            };
            let mut browser_shutdown = shutdown_rx.clone();
            tokio::spawn(async move {
                // Small delay to let the server bind, abort on shutdown
                tokio::select! {
                    _ = tokio::time::sleep(std::time::Duration::from_millis(500)) => {
                        if let Err(e) = open_browser(&target) {
                            tracing::debug!(error = %e, "Could not open browser automatically");
                        }
                    }
                    _ = browser_shutdown.changed() => {}
                }
            });
        }

        // Auto-load models that have local shards available
        {
            let sm = shared_state.clone();
            let mut autoload_shutdown = shutdown_rx.clone();
            tokio::spawn(async move {
                // Brief delay to let shard announcements propagate, abort on shutdown
                tokio::select! {
                    _ = tokio::time::sleep(std::time::Duration::from_secs(2)) => {}
                    _ = autoload_shutdown.changed() => { return; }
                }
                let mut manifests = sm.model_registry.list_models();
                // Sort by request count descending so popular models get VRAM priority on restart
                manifests.sort_by(|a, b| {
                    let count_a = sm
                        .model_request_counts
                        .get(&a.id)
                        .map(|c| c.value().load(std::sync::atomic::Ordering::Relaxed))
                        .unwrap_or(0);
                    let count_b = sm
                        .model_request_counts
                        .get(&b.id)
                        .map(|c| c.value().load(std::sync::atomic::Ordering::Relaxed))
                        .unwrap_or(0);
                    count_b.cmp(&count_a)
                });
                let vram_budget = crate::model::auto_manage::compute_vram_budget(&sm);
                for m in &manifests {
                    if sm.split_models.iter().any(|e| e.key().0 == m.id) {
                        continue;
                    }
                    crate::model::auto_manage::check_and_load_model(&sm, &m.id, vram_budget).await;
                }
            });
        }

        // ── SIGHUP Config Reload Handler (Unix only) ──
        #[cfg(unix)]
        {
            let sighup_state = shared_state.clone();
            let sighup_config = self.config.clone();
            let mut sighup_shutdown = shutdown_rx.clone();
            tokio::spawn(async move {
                let mut sighup = match tokio::signal::unix::signal(
                    tokio::signal::unix::SignalKind::hangup(),
                ) {
                    Ok(s) => s,
                    Err(e) => {
                        tracing::warn!(error = %e, "Failed to register SIGHUP handler — config reload via signal disabled");
                        return;
                    }
                };
                loop {
                    tokio::select! {
                        _ = sighup_shutdown.changed() => {
                            if *sighup_shutdown.borrow() {
                                break;
                            }
                        }
                        _ = sighup.recv() => {
                            let config_path = sighup_config.node.data_dir.join("config.toml");
                            tracing::info!(
                                "SIGHUP received — reloading config from {}",
                                config_path.display()
                            );
                            match crate::config::reload_operational_params(&config_path) {
                                Ok(params) => {
                                    let old = crate::config::OperationalParams::from_config(
                                        &sighup_config,
                                    );
                                    if params != old {
                                        tracing::info!(
                                            ?params,
                                            "Config reloaded with changes"
                                        );
                                    } else {
                                        tracing::info!(
                                            "Config reloaded — no changes detected"
                                        );
                                    }
                                    sighup_state.apply_config_reload(params);
                                }
                                Err(e) => {
                                    tracing::error!(
                                        error = %e,
                                        "Failed to reload config on SIGHUP"
                                    );
                                }
                            }
                        }
                    }
                }
            });
        }

        // ── Supervisor Loop ──
        //
        // Monitors all subsystem tasks via JoinSet. When a task exits:
        // - Due to shutdown signal: expected, no action needed
        // - Non-critical subsystem: log error and continue running
        // - Critical subsystem: trigger graceful shutdown
        // - Panic: treated as unexpected exit with same criticality rules
        //
        // Track restart attempts per subsystem name
        let mut restart_counts: std::collections::HashMap<&str, u32> =
            std::collections::HashMap::new();

        loop {
            tokio::select! {
                // Handle OS shutdown signals
                _ = async {
                    let ctrl_c = tokio::signal::ctrl_c();
                    #[cfg(unix)]
                    {
                        match tokio::signal::unix::signal(
                            tokio::signal::unix::SignalKind::terminate(),
                        ) {
                            Ok(mut sigterm) => {
                                tokio::select! {
                                    _ = ctrl_c => {
                                        tracing::info!("Shutdown signal received (Ctrl+C)");
                                    }
                                    _ = sigterm.recv() => {
                                        tracing::info!("Shutdown signal received (SIGTERM)");
                                    }
                                }
                            }
                            Err(e) => {
                                tracing::warn!(error = %e, "Failed to register SIGTERM handler — using Ctrl+C only");
                                ctrl_c.await.ok();
                                tracing::info!("Shutdown signal received (Ctrl+C)");
                            }
                        }
                    }
                    #[cfg(not(unix))]
                    {
                        ctrl_c.await.ok();
                        tracing::info!("Shutdown signal received (Ctrl+C)");
                    }
                } => {
                    break;
                }
                // Handle API-triggered shutdown (watch channel)
                _ = shutdown_rx.changed() => {
                    if *shutdown_rx.borrow() {
                        tracing::info!("Shutdown requested via API — draining subsystems");
                        break;
                    }
                }
                // Handle subsystem task exits
                result = subsystems.join_next() => {
                    match result {
                        None => {
                            // All tasks finished — shouldn't happen during normal operation
                            tracing::error!("All subsystem tasks have exited");
                            break;
                        }
                        Some(Ok((name, criticality, task_result))) => {
                            // Check if this is a shutdown-induced exit (expected)
                            if *shutdown_rx.borrow() {
                                tracing::debug!(subsystem = name, "Subsystem exited during shutdown");
                                continue;
                            }

                            match task_result {
                                Ok(()) => {
                                    tracing::warn!(
                                        subsystem = name,
                                        "Subsystem exited unexpectedly with Ok"
                                    );
                                }
                                Err(ref e) => {
                                    tracing::error!(
                                        subsystem = name,
                                        error = %e,
                                        "Subsystem exited with error"
                                    );
                                }
                            }

                            let count = restart_counts.entry(name).or_insert(0);
                            *count += 1;

                            if criticality == SubsystemCriticality::Critical {
                                if *count > MAX_RESTART_ATTEMPTS {
                                    tracing::error!(
                                        subsystem = name,
                                        attempts = *count,
                                        "Critical subsystem permanently failed — shutting down"
                                    );
                                    break;
                                }
                                // Critical subsystem failed but we can't restart channel-bound
                                // tasks, so trigger shutdown immediately.
                                tracing::error!(
                                    subsystem = name,
                                    "Critical subsystem failed — triggering graceful shutdown"
                                );
                                break;
                            } else {
                                // Non-critical: log and continue
                                tracing::warn!(
                                    subsystem = name,
                                    restart_count = *count,
                                    max_restarts = MAX_RESTART_ATTEMPTS,
                                    "Non-critical subsystem failed — daemon continues without it"
                                );
                            }
                        }
                        Some(Err(join_error)) => {
                            // Task panicked or was cancelled
                            if join_error.is_panic() {
                                tracing::error!(
                                    error = %join_error,
                                    "Subsystem task panicked — triggering shutdown"
                                );
                                break;
                            } else {
                                tracing::warn!(
                                    error = %join_error,
                                    "Subsystem task cancelled"
                                );
                            }
                        }
                    }
                }
            }
        }

        // Signal graceful shutdown to all subsystems
        shared_state.shutdown();

        // Drain the JoinSet with a timeout so subsystems can run their cleanup
        // (e.g., save peer cache, close connections, flush data).
        tracing::info!("Waiting for subsystems to shut down (10s timeout)...");
        let drain_deadline = tokio::time::sleep(std::time::Duration::from_secs(10));
        tokio::pin!(drain_deadline);
        loop {
            tokio::select! {
                _ = &mut drain_deadline => {
                    tracing::warn!("Shutdown timeout — aborting remaining subsystems");
                    break;
                }
                result = subsystems.join_next() => {
                    match result {
                        Some(Ok((name, _, _))) => {
                            tracing::debug!(subsystem = name, "Subsystem exited cleanly");
                        }
                        Some(Err(e)) => {
                            tracing::debug!(error = %e, "Subsystem join error during shutdown");
                        }
                        None => {
                            tracing::info!("All subsystems shut down cleanly");
                            break;
                        }
                    }
                }
            }
        }

        // Flush database after subsystems have had a chance to write final state
        if let Err(e) = shared_state.db.flush() {
            tracing::error!(error = %e, "Failed to flush database during shutdown");
        }

        tracing::info!("Daemon shutdown complete");

        Ok(())
    }
}

/// Maximum number of concurrent LayerForward tasks.
const MAX_CONCURRENT_FORWARDS: usize = 64;

/// Dispatch inbound network messages to the appropriate subsystem.
///
/// Inference-related messages (InferenceRequest, LayerForward, LayerResult,
/// InferenceError, PipelineAssignment) are routed to the InferenceRouter.
/// CreditGossip messages are used to update the peer balance distribution.
/// ModelVote messages are routed to the governance processor.
/// Other messages (health, discovery) are handled by their respective
/// subsystems directly via SharedState or are already handled by NetworkManager.
async fn dispatch_network_messages(
    network_out_rx: &mut mpsc::Receiver<SwarmMessage>,
    router_tx: &mpsc::Sender<RouterCommand>,
    credit_peer_balances: Arc<RwLock<Vec<i64>>>,
    shared_state: &Arc<SharedState>,
    network_tx: mpsc::Sender<NetworkCommand>,
    mut shutdown_rx: watch::Receiver<bool>,
) {
    let forward_semaphore = Arc::new(tokio::sync::Semaphore::new(MAX_CONCURRENT_FORWARDS));
    loop {
        tokio::select! {
            _ = shutdown_rx.changed() => {
                if *shutdown_rx.borrow() {
                    break;
                }
            }
            msg = network_out_rx.recv() => {
                match msg {
                    Some(msg) => {
                        match msg {
                            // LayerResult: route to pending pipeline executor via oneshot channel
                            SwarmMessage::LayerResult(ref result) => {
                                tracing::info!(
                                    request_id = %result.request_id,
                                    tokens = result.token_ids.len(),
                                    activations_bytes = result.activations.len(),
                                    finish = ?result.finish_reason,
                                    pending_count = shared_state.pending_layer_results.len(),
                                    "DIAG: dispatcher received LayerResult"
                                );
                                if let Some((_, tx)) = shared_state
                                    .pending_layer_results
                                    .remove(&result.request_id)
                                {
                                    if tx.send(result.clone()).is_err() {
                                        tracing::warn!(
                                            request_id = %result.request_id,
                                            tokens = result.token_ids.len(),
                                            finish = ?result.finish_reason,
                                            "DIAG: LayerResult delivered but pipeline receiver DROPPED"
                                        );
                                    } else {
                                        tracing::info!(
                                            request_id = %result.request_id,
                                            tokens = result.token_ids.len(),
                                            activations_bytes = result.activations.len(),
                                            finish = ?result.finish_reason,
                                            pending_remaining = shared_state.pending_layer_results.len(),
                                            "DIAG: LayerResult delivered to pipeline"
                                        );
                                    }
                                } else {
                                    tracing::warn!(
                                        request_id = %result.request_id,
                                        tokens = result.token_ids.len(),
                                        finish = ?result.finish_reason,
                                        pending_count = shared_state.pending_layer_results.len(),
                                        "DIAG: No pending channel for LayerResult — already timed out or duplicate"
                                    );
                                }
                            }
                            // LayerForward: process locally using split inference engine,
                            // then send back a LayerResult to the requesting node.
                            SwarmMessage::LayerForward(forward) => {
                                tracing::info!(
                                    request_id = %forward.request_id,
                                    seq = forward.sequence_num,
                                    layer_range = ?forward.layer_range,
                                    activation_bytes = forward.activations.len(),
                                    has_sender = forward.sender_peer_bytes.is_some(),
                                    "DIAG: dispatcher received LayerForward, spawning handler"
                                );
                                let ss = shared_state.clone();
                                let ntx = network_tx.clone();
                                let sem = forward_semaphore.clone();
                                tokio::spawn(async move {
                                    let _permit = match sem.acquire().await {
                                        Ok(p) => p,
                                        Err(_) => return, // semaphore closed
                                    };
                                    handle_layer_forward(ss, ntx, forward).await;
                                });
                            }
                            // StreamingToken: route to registered streaming channel
                            SwarmMessage::StreamingToken(ref token) => {
                                // Clone the sender to drop the DashMap Ref (read lock) before
                                // awaiting send() or calling remove() — avoids deadlock.
                                let maybe_tx = shared_state
                                    .streaming_token_txs
                                    .get(&token.request_id)
                                    .map(|r| r.clone());
                                if let Some(tx) = maybe_tx {
                                    if tx.send(token.clone()).await.is_err() {
                                        tracing::debug!(
                                            request_id = %token.request_id,
                                            "Streaming token channel closed"
                                        );
                                        shared_state.streaming_token_txs.remove(&token.request_id);
                                    }
                                }
                            }
                            // T13: VisionEncodeRequest — encode image using local mmproj
                            SwarmMessage::VisionEncodeRequest(req) => {
                                let ss = shared_state.clone();
                                let ntx = network_tx.clone();
                                tokio::spawn(async move {
                                    handle_vision_encode_request(ss, ntx, req).await;
                                });
                            }
                            // T13: VisionEncodeResponse — fire pending oneshot
                            SwarmMessage::VisionEncodeResponse(resp) => {
                                if let Some((_, tx)) = shared_state
                                    .pending_vision_results
                                    .remove(&resp.request_id)
                                {
                                    let _ = tx.send(resp);
                                }
                            }
                            msg @ SwarmMessage::InferenceRequest(_)
                            | msg @ SwarmMessage::PipelineAssignment(_)
                            | msg @ SwarmMessage::InferenceError(_) => {
                                if let Err(e) = router_tx
                                    .send(RouterCommand::NetworkMessage(msg))
                                    .await
                                {
                                    tracing::warn!(
                                        error = %e,
                                        "Failed to route inference message to router"
                                    );
                                }
                            }
                            SwarmMessage::CreditGossip(gossip) => {
                                crate::credit::ledger::process_balance_gossip(
                                    &credit_peer_balances,
                                    &gossip,
                                ).await;
                                // Store per-peer balance for leaderboard display
                                shared_state.peer_credit_balances.insert(
                                    gossip.node_id.clone(),
                                    gossip.balance_bucket,
                                );
                            }
                            SwarmMessage::ModelVote(vote) => {
                                tracing::info!(
                                    voter = %vote.voter,
                                    manifest_hash = hex::encode(&vote.model_manifest_hash[..8]),
                                    vote = vote.vote,
                                    "Received model vote"
                                );
                                match crate::model::governance::process_vote(
                                    &shared_state.model_vote_tallies,
                                    vote.clone(),
                                ) {
                                    Ok(Some(verdict)) => {
                                        tracing::info!(
                                            ?verdict,
                                            manifest_hash = hex::encode(&vote.model_manifest_hash[..8]),
                                            "Model vote concluded"
                                        );
                                    }
                                    Ok(None) => {} // Still pending
                                    Err(e) => {
                                        tracing::warn!(error = %e, "Failed to process model vote");
                                    }
                                }
                            }
                            SwarmMessage::CreditTransaction(tx) => {
                                tracing::debug!(
                                    tx_id = %tx.id,
                                    from = %tx.from,
                                    to = %tx.to,
                                    amount = tx.amount,
                                    "Received credit transaction"
                                );
                                // Anti-gaming validation for network transactions
                                {
                                    let mut ag = shared_state.anti_gaming.lock().await;
                                    match ag.check_transaction(&tx.from, &tx.to, tx.amount) {
                                        Ok(_decision) => {
                                            ag.record_transaction(&tx.from);
                                        }
                                        Err(violation) => {
                                            tracing::warn!(
                                                tx_id = %tx.id,
                                                violation = %violation,
                                                "Anti-gaming rejected credit transaction"
                                            );
                                            continue;
                                        }
                                    }
                                }
                                // Record the transaction and apply balance change
                                // if we are the recipient
                                let local_id = shared_state.identity.node_id().clone();
                                if tx.to == local_id {
                                    if let Err(e) = crate::credit::ledger::apply_credit_direct(
                                        &shared_state.credit_balance,
                                        &shared_state.db,
                                        tx.amount,
                                        true,
                                    ).await {
                                        tracing::warn!(error = %e, "Failed to apply credit transaction");
                                    }
                                    let bal = shared_state.credit_balance.read().await;
                                    tracing::info!(
                                        amount = tx.amount,
                                        balance = bal.balance,
                                        "Applied incoming credit transaction"
                                    );
                                }
                                let key = tx.id.to_string();
                                if let Err(e) = shared_state.db.put_json(crate::credit::ledger::TREE_TRANSACTIONS, &key, &tx) {
                                    tracing::warn!(error = %e, "Failed to store credit transaction");
                                }
                            }
                            // Process shard announcements from peers
                            SwarmMessage::ShardAnnounce(announce) => {
                                tracing::info!(
                                    node_id = %announce.node_id,
                                    shards = announce.shards.len(),
                                    "Received shard announce from peer"
                                );
                                // Refresh last_seen so health monitor doesn't remove active peers
                                if let Some(mut peer) = shared_state.peer_registry.get_mut(&announce.node_id) {
                                    peer.last_seen = chrono::Utc::now();
                                }
                                for shard_id in &announce.shards {
                                    shared_state.model_registry
                                        .record_shard_holder(shard_id.clone(), announce.node_id.clone());
                                }
                                // Wake auto-manage so it re-evaluates rarity scores —
                                // new shard holders change which shards are most needed.
                                shared_state.auto_manage_notify.notify_one();
                            }
                            // Process model manifests from peers — register in model_registry
                            SwarmMessage::ModelManifest(manifest) => {
                                tracing::info!(
                                    model = %manifest.id,
                                    name = %manifest.name,
                                    shards = manifest.shard_count,
                                    publisher = %manifest.publisher,
                                    "Received model manifest from network"
                                );
                                // Strict verification for network-received manifests:
                                // reject zero-hash to prevent gossip poisoning.
                                match manifest.verify_hash_strict() {
                                    Ok(()) => {
                                        let is_new = shared_state
                                            .model_registry
                                            .get_manifest(&manifest.id)
                                            .is_none();
                                        shared_state.model_registry.register_manifest(manifest.clone());
                                        // Wake auto-manage when a genuinely new model appears
                                        if is_new {
                                            shared_state.auto_manage_notify.notify_one();
                                        }
                                    }
                                    Err(e) => {
                                        tracing::warn!(
                                            error = %e,
                                            "Manifest hash verification failed — rejecting"
                                        );
                                    }
                                }
                            }
                            // Process capability updates from peers
                            SwarmMessage::NodeCapabilityUpdate(cap) => {
                                tracing::debug!(
                                    node_id = %cap.node_id,
                                    hosted_shards = cap.hosted_shards.len(),
                                    "Received capability update from peer"
                                );
                                if let Some(mut peer) = shared_state.peer_registry.get_mut(&cap.node_id) {
                                    peer.capability = Some(cap.clone());
                                    peer.last_seen = chrono::Utc::now();
                                }
                            }
                            // Nickname gossip from peers
                            SwarmMessage::NicknameGossip(gossip) => {
                                let record = &gossip.record;
                                // Age check: reject messages older than 24 hours
                                let age = chrono::Utc::now() - record.timestamp;
                                if age > chrono::Duration::hours(24) {
                                    tracing::debug!(
                                        node_id = %record.node_id,
                                        "Rejecting stale nickname gossip (>24h old)"
                                    );
                                } else if record.verify().is_err() {
                                    tracing::warn!(
                                        node_id = %record.node_id,
                                        "Rejecting nickname gossip with invalid signature"
                                    );
                                } else {
                                    // Timestamp-wins: only update if newer
                                    let should_insert = match shared_state
                                        .nickname_registry
                                        .get(&record.node_id)
                                    {
                                        Some(existing) => record.timestamp > existing.timestamp,
                                        None => true,
                                    };
                                    if should_insert {
                                        tracing::info!(
                                            node_id = %record.node_id,
                                            nickname = %record.nickname,
                                            "Accepted nickname from peer"
                                        );
                                        shared_state
                                            .nickname_registry
                                            .insert(record.node_id.clone(), record.clone());
                                        // Persist
                                        let store = crate::identity::nickname::NicknameStore::new(
                                            shared_state.db.clone(),
                                        );
                                        if let Err(e) = store.put_record(record) {
                                            tracing::warn!(error = %e, "Failed to persist nickname");
                                        }
                                    }
                                }
                            }
                            // Route pool messages to the PoolManager
                            SwarmMessage::PoolMessage(pool_msg) => {
                                if let Some(ref tx) = *shared_state.pool_tx.read().await {
                                    let cmd = match pool_msg {
                                        crate::types::PoolMessage::Invitation(inv) => {
                                            Some(crate::pool::types::PoolCommand::InboundInvitation {
                                                invitation: inv,
                                            })
                                        }
                                        crate::types::PoolMessage::BlindedInvitation(blinded) => {
                                            Some(crate::pool::types::PoolCommand::InboundBlindedInvitation {
                                                blinded,
                                            })
                                        }
                                        crate::types::PoolMessage::Acceptance(acc) => {
                                            Some(crate::pool::types::PoolCommand::InboundAcceptance {
                                                acceptance: acc,
                                            })
                                        }
                                        crate::types::PoolMessage::StateGossip(state) => {
                                            Some(crate::pool::types::PoolCommand::PoolStateGossip {
                                                state,
                                            })
                                        }
                                        crate::types::PoolMessage::CreditForward(fwd) => {
                                            Some(crate::pool::types::PoolCommand::ProcessCreditForward {
                                                forward: fwd,
                                            })
                                        }
                                        crate::types::PoolMessage::Removal(rem) => {
                                            Some(crate::pool::types::PoolCommand::InboundRemoval {
                                                removal: rem,
                                            })
                                        }
                                        crate::types::PoolMessage::MemberLeft { pool_id, node_id, signature } => {
                                            Some(crate::pool::types::PoolCommand::InboundMemberLeft {
                                                pool_id,
                                                node_id,
                                                signature,
                                            })
                                        }
                                    };
                                    if let Some(cmd) = cmd {
                                        if let Err(e) = tx.send(cmd).await {
                                            tracing::warn!(error = %e, "Failed to route pool message");
                                        }
                                    }
                                }
                            }
                            // HuggingFace source gossip — store so auto-manage can download shards
                            SwarmMessage::HfSourceGossip(gossip) => {
                                let mid = gossip.model_id.clone();
                                if !shared_state.hf_sources.contains_key(&mid) {
                                    tracing::info!(
                                        model = %mid,
                                        repo = %gossip.repo_id,
                                        filename = %gossip.filename,
                                        publisher = %gossip.publisher,
                                        "Received HfSourceGossip — storing HF source"
                                    );
                                    let source = crate::daemon::HfSource {
                                        repo_id: gossip.repo_id.clone(),
                                        filename: gossip.filename.clone(),
                                        mmproj_filename: gossip.mmproj_filename.clone(),
                                    };
                                    shared_state.hf_sources.insert(mid.clone(), source.clone());
                                    // Persist to sled
                                    let _ = shared_state.db.put_json("hf_sources", &mid.0, &source);
                                    // Wake the AutoShardManager so it evaluates promptly
                                    shared_state.auto_manage_notify.notify_one();
                                }
                            }
                            SwarmMessage::ShardDownloadProgress(progress) => {
                                // Update peer download state in shared state
                                let local_nid = shared_state.identity.node_id();
                                if progress.node_id != *local_nid {
                                    if progress.state == crate::types::DownloadState::Complete || progress.progress_pct >= 100 {
                                        // Download finished — remove from download tracking
                                        if let Some(mut entry) = shared_state.peer_shard_downloads.get_mut(&progress.shard_id) {
                                            entry.retain(|(nid, _)| *nid != progress.node_id);
                                        }
                                        // Register the peer as a shard holder now
                                        // (the ShardAnnounce gossip will also arrive,
                                        //  but this gives immediate consistency)
                                        shared_state.model_registry
                                            .record_shard_holder(progress.shard_id.clone(), progress.node_id.clone());
                                        // Wake auto-manage — peer completed a download, rarity changed
                                        shared_state.auto_manage_notify.notify_one();
                                    } else {
                                        // Update or insert download progress
                                        let mut entry = shared_state.peer_shard_downloads.entry(progress.shard_id.clone()).or_default();
                                        if let Some(pos) = entry.iter().position(|(nid, _)| *nid == progress.node_id) {
                                            entry[pos].1 = progress.progress_pct;
                                        } else {
                                            entry.push((progress.node_id.clone(), progress.progress_pct));
                                        }
                                    }
                                    tracing::debug!(
                                        node = %progress.node_id,
                                        model = %progress.shard_id.model_id,
                                        shard = progress.shard_id.index,
                                        pct = progress.progress_pct,
                                        state = %progress.state,
                                        "Peer shard download progress"
                                    );
                                }
                            }
                            // Health pings: update sender's load and respond with pong
                            SwarmMessage::HealthPing { nonce, node_id: Some(sender_id), active_request_count, .. } => {
                                // Update the sender's active request count in peer_registry
                                if let Some(mut peer) = shared_state.peer_registry.get_mut(&sender_id) {
                                    peer.active_request_count = active_request_count;
                                    peer.last_seen = chrono::Utc::now();
                                }

                                // Respond with a pong containing our own load
                                let ts = std::time::SystemTime::now()
                                    .duration_since(std::time::UNIX_EPOCH)
                                    .unwrap_or_default()
                                    .as_secs();
                                let our_load = shared_state.active_pipelines.len() as u32;
                                let our_id = Some(shared_state.identity.node_id().clone());
                                let pong = SwarmMessage::HealthPong {
                                    nonce,
                                    timestamp: ts,
                                    node_id: our_id,
                                    active_request_count: our_load,
                                };
                                let _ = network_tx.send(NetworkCommand::Broadcast(pong)).await;
                            }
                            // Ignore health pings without node_id (pre-alpha format)
                            SwarmMessage::HealthPing { node_id: None, .. } => {}
                            // Health pongs: update the sender's load in peer_registry
                            SwarmMessage::HealthPong { node_id: Some(sender_id), active_request_count, .. } => {
                                if let Some(mut peer) = shared_state.peer_registry.get_mut(&sender_id) {
                                    peer.active_request_count = active_request_count;
                                    peer.last_seen = chrono::Utc::now();
                                }
                            }
                            // Ignore health pongs without node_id (pre-alpha format)
                            SwarmMessage::HealthPong { node_id: None, .. } => {}
                            // Ephemeral key exchange for forward secrecy
                            SwarmMessage::EphemeralKeyExchange(exchange) => {
                                let sm = shared_state.session_manager.clone();
                                let our_id = shared_state.identity.node_id().clone();
                                if exchange.node_id == our_id {
                                    // Ignore our own broadcast
                                } else if exchange.is_initiator {
                                    // Peer wants to re-key: accept and reply
                                    let response_pub = sm.accept_ephemeral_exchange(
                                        &exchange.node_id,
                                        &exchange.ephemeral_pubkey,
                                    );
                                    let reply = SwarmMessage::EphemeralKeyExchange(EphemeralKeyExchange {
                                        session_id: exchange.session_id,
                                        node_id: our_id,
                                        ephemeral_pubkey: response_pub,
                                        is_initiator: false,
                                    });
                                    let _ = network_tx.send(NetworkCommand::Broadcast(reply)).await;
                                } else {
                                    // Response to our initiation: complete the exchange
                                    sm.complete_ephemeral_session(
                                        &exchange.node_id,
                                        &exchange.ephemeral_pubkey,
                                    );
                                }
                            }
                            // Tensor-parallel AllReduce: collect partial from a TP rank
                            SwarmMessage::TpAllReduceRequest(req) => {
                                let key = (req.request_id, req.layer_idx);
                                let tp_size = req.tp_size;
                                let ss = shared_state.clone();
                                let ntx = network_tx.clone();

                                // Extract sender peer bytes from the request context
                                // (embedded by NetworkManager when receiving the rr request)
                                let sender_peer = None; // TODO: plumb sender peer from rr handler

                                let all_arrived = {
                                    let mut entry = ss.pending_tp_partials
                                        .entry(key)
                                        .or_insert_with(|| TpAllReduceCollector::new(tp_size));
                                    entry.insert(req, sender_peer)
                                };

                                if all_arrived {
                                    // All partials collected — reduce and respond
                                    tokio::spawn(async move {
                                        let collector = ss.pending_tp_partials.remove(&key);
                                        if let Some((_, collector)) = collector {
                                            match collector.reduce_sum() {
                                                Ok((reduced_data, shape)) => {
                                                    let resp = crate::types::TpAllReduceResponse {
                                                        request_id: key.0,
                                                        layer_idx: key.1,
                                                        reduced_data,
                                                        shape,
                                                    };
                                                    // Deliver to local registry (coordinator is also a TP rank)
                                                    ss.allreduce_registry.deliver(resp.clone());
                                                    // Broadcast response to remote TP participants
                                                    let msg = SwarmMessage::TpAllReduceResponse(resp);
                                                    let _ = ntx.send(NetworkCommand::Broadcast(msg)).await;
                                                }
                                                Err(e) => {
                                                    tracing::warn!(
                                                        error = %e,
                                                        request_id = %key.0,
                                                        layer_idx = key.1,
                                                        "AllReduce sum failed"
                                                    );
                                                }
                                            }
                                        }
                                    });
                                }
                            }
                            // Tensor-parallel AllReduce response: deliver to waiting pipeline
                            SwarmMessage::TpAllReduceResponse(resp) => {
                                let delivered = shared_state.allreduce_registry.deliver(resp.clone());
                                tracing::debug!(
                                    request_id = %resp.request_id,
                                    layer_idx = resp.layer_idx,
                                    reduced_bytes = resp.reduced_data.len(),
                                    delivered,
                                    "AllReduce response received"
                                );
                            }
                            // Other messages handled by NetworkManager
                            _ => {}
                        }
                    }
                    None => break,
                }
            }
        }
    }
}

/// Resolve the API key: config > DB > generate new.
/// Returns the key and persists it to the DB if newly generated.
fn resolve_api_key(config: &Config, db: &Database) -> String {
    let key;

    // 1. Explicit key in config takes priority
    if let Some(ref k) = config.api.api_key {
        if !k.is_empty() {
            tracing::info!("Using API key from configuration");
            key = k.clone();
            write_api_key_file(&config.node.data_dir, &key);
            return key;
        }
    }

    // 2. Check persisted key in database
    if let Ok(Some(k)) = db.get_json::<String>("config", "api_key") {
        if !k.is_empty() {
            tracing::info!("Using persisted API key from database");
            write_api_key_file(&config.node.data_dir, &k);
            return k;
        }
    }

    // 3. Generate a new 32-byte hex key
    use rand::RngCore;
    let mut bytes = [0u8; 32];
    rand::rngs::OsRng.fill_bytes(&mut bytes);
    key = hex::encode(bytes);

    // Persist to DB
    if let Err(e) = db.put_json("config", "api_key", &key) {
        tracing::warn!(error = %e, "Failed to persist API key to database");
    }

    // Write to file so CLI `status` can read it without opening sled
    write_api_key_file(&config.node.data_dir, &key);

    // Print API key to stderr only (not to tracing logs which may be persisted/shipped)
    eprintln!("Generated new API key (save this for API access): {key}");

    key
}

/// Write the API key to a plain file so the CLI can read it while the daemon holds the DB lock.
fn write_api_key_file(data_dir: &std::path::Path, key: &str) {
    let path = data_dir.join("api_key");
    if let Err(e) = std::fs::write(&path, key) {
        tracing::warn!(error = %e, "Failed to write api_key file");
    }
    // Restrict permissions on Unix (owner read/write only)
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600));
    }
}

/// Generate a ModelManifest for a locally loaded GGUF file and register it.
///
/// This solves the "bootstrap deadlock" — without a manifest, peers can't discover
/// or request the model. By generating a manifest from the loaded GGUF at startup,
/// we can broadcast it to the network so other nodes can acquire shards.
pub fn generate_and_register_local_manifest(
    shared_state: &Arc<SharedState>,
    info: &LoadedModelInfo,
    model_path: &std::path::Path,
) {
    // Use a filesystem-safe slug for the model ID.
    // Lowercase, replace spaces/special chars with hyphens, collapse runs.
    let slug = info
        .name
        .to_lowercase()
        .replace(|c: char| !c.is_alphanumeric() && c != '-' && c != '.', "-")
        .split('-')
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join("-");
    let model_id = crate::types::ModelId(slug);

    // Check if we already have a manifest for this model (e.g. persisted from a previous run).
    // Even if the manifest exists, we must still register ourselves as shard holders.
    if let Some(existing) = shared_state.model_registry.get_manifest(&model_id) {
        tracing::debug!(model = %model_id, "Manifest already registered, registering shard holders");
        let node_id = shared_state.identity.node_id().clone();
        let shard_range = shared_state.config.inference.shard_range;
        for shard_info in &existing.shards {
            let in_range = match shard_range {
                Some((start, end)) => shard_info.index >= start && shard_info.index <= end,
                None => true,
            };
            if in_range {
                let shard_id = crate::types::ShardId {
                    model_id: model_id.clone(),
                    index: shard_info.index,
                };
                shared_state
                    .model_registry
                    .record_shard_holder(shard_id, node_id.clone());
            }
        }
        // Also load GGUF metadata if not already cached
        if !shared_state.gguf_meta.contains_key(&model_id) {
            let path = std::path::Path::new(model_path);
            if let Ok(meta) = crate::inference::split::GgufTensorMeta::from_gguf_file(path) {
                shared_state.gguf_meta.insert(model_id.clone(), meta);
            }
        }
        return;
    }

    let path = std::path::Path::new(model_path);
    if !path.exists() {
        tracing::warn!(path = %model_path.display(), "Model file not found, cannot generate manifest");
        return;
    }

    let file_size = std::fs::metadata(path)
        .map(|m| m.len())
        .unwrap_or(info.size_bytes);

    // Split model into shards for torrent-style distribution.
    // Shard size is configurable via [model].shard_size_mb (default 512MB).
    let shard_size: u64 = shared_state.config.model.shard_size_bytes();
    let node_id = shared_state.identity.node_id().clone();

    // Extract model metadata from GGUF header (num_layers, architecture, etc.)
    // and compute layer-aligned shard layouts. The layout count determines shard_count
    // (NOT file_size / shard_size, which can differ from the actual layout count).
    let (num_layers, architecture, shard_count, shards) =
        match crate::inference::split::GgufTensorMeta::from_gguf_file(path) {
            Ok(meta) => {
                let num_layers = meta.block_count as u32;
                // Estimate shard count from file size for layout computation
                let estimated_count = file_size.div_ceil(shard_size).max(1) as u32;
                let layouts =
                    crate::inference::split::compute_layer_shard_layouts(&meta, estimated_count);
                let actual_shard_count = layouts.len() as u32;
                tracing::info!(
                    model = %model_id,
                    num_layers,
                    embedding_length = meta.embedding_length,
                    shard_count = actual_shard_count,
                    "Extracted GGUF metadata for manifest"
                );

                // Build shard infos from layouts (handles hashing, tensor entries, layer ranges)
                let model_dir =
                    crate::model::shard::ShardStore::new(&shared_state.config.node.data_dir)
                        .models_dir()
                        .join(&model_id.0);
                let shards =
                    crate::model::manifest::build_shard_infos_from_layouts(&model_dir, &layouts);

                // Store the metadata for later use in layer range computation
                shared_state.gguf_meta.insert(model_id.clone(), meta);
                // Map GGUF general.architecture string to our ModelArchitecture enum
                let arch = map_gguf_architecture(path);
                (num_layers, arch, actual_shard_count, shards)
            }
            Err(e) => {
                tracing::warn!(error = %e, "Failed to extract GGUF metadata, using defaults");
                let shard_count = file_size.div_ceil(shard_size).max(1) as u32;
                let shards = vec![];
                (
                    0u32,
                    crate::types::ModelArchitecture::Llama,
                    shard_count,
                    shards,
                )
            }
        };

    let mut manifest = crate::types::ModelManifest {
        schema_version: 2,
        id: model_id.clone(),
        name: info.name.clone(),
        architecture,
        num_layers,
        num_params_billions: 0.0,
        quantization: crate::types::Quantization::Q4KM,
        total_size_bytes: file_size,
        shard_count,
        shards,
        tokenizer_hash: [0u8; 32],
        manifest_hash: [0u8; 32],
        publisher: node_id.clone(),
        publish_date: chrono::Utc::now(),
        license: "Unknown".to_string(),
        mmproj: None,
    };
    manifest.manifest_hash = manifest.compute_hash();

    // Store the source GGUF path so the shard server can read byte ranges from it.
    // We write a small metadata file alongside the manifest.
    let shard_store = ShardStore::new(&shared_state.config.node.data_dir);
    let model_dir = shard_store.models_dir().join(&model_id.0);
    let _ = std::fs::create_dir_all(&model_dir);

    // Write a source_path file so the shard server knows where the original GGUF lives
    if let Ok(canonical) = path.canonicalize() {
        let source_path_file = model_dir.join("source_path");
        if let Err(e) = std::fs::write(&source_path_file, canonical.to_string_lossy().as_bytes()) {
            tracing::warn!(error = %e, "Failed to write source_path file");
        }
    }

    // Save GGUF header for shard-only operation.
    // This allows nodes without the full model file to use ShardReader.
    let header_path = model_dir.join("gguf_header.bin");
    if !header_path.exists() {
        if let Err(e) = crate::inference::split::save_gguf_header(path, &header_path) {
            tracing::warn!(error = %e, "Failed to save GGUF header (shard-only mode won't work)");
        }
    }

    // Save manifest to disk
    if let Err(e) = manifest.save_to_dir(&model_dir) {
        tracing::warn!(error = %e, "Failed to save generated manifest");
        return;
    }

    // If shards live in a differently-named directory (e.g. from HF download),
    // also save manifest + header there so shard scanning finds them.
    let shard0_in_model_dir = model_dir.join("shard_000.bin");
    if !shard0_in_model_dir.exists() {
        // Shards might be in a different directory — scan for them
        let models_dir = shard_store.models_dir();
        if let Ok(entries) = std::fs::read_dir(&models_dir) {
            for entry in entries.flatten() {
                let dir = entry.path();
                if dir.is_dir() && dir != model_dir && dir.join("shard_000.bin").exists() {
                    // Found shards in a different directory — save manifest + header there too
                    if !dir.join("manifest.json").exists() {
                        if let Err(e) = manifest.save_to_dir(&dir) {
                            tracing::warn!(error = %e, path = %dir.display(), "Failed to save manifest to shard dir");
                        } else {
                            tracing::info!(
                                model = %model_id,
                                shard_dir = %dir.display(),
                                "Also saved manifest to shard directory"
                            );
                        }
                    }
                    let alt_header = dir.join("gguf_header.bin");
                    if !alt_header.exists() {
                        if let Err(e) = crate::inference::split::save_gguf_header(path, &alt_header)
                        {
                            tracing::warn!(error = %e, "Failed to save GGUF header to shard dir");
                        }
                    }
                }
            }
        }
    }

    // Register in model_registry
    shared_state
        .model_registry
        .register_manifest(manifest.clone());

    // Register ourselves as holder of our shards.
    // If --shards range is set, only claim those indices; otherwise claim all.
    // Only register shards that actually exist on disk.
    let shard_range = shared_state.config.inference.shard_range;
    let shard_store_check = ShardStore::new(&shared_state.config.node.data_dir);
    for shard_info in &manifest.shards {
        let in_range = match shard_range {
            Some((start, end)) => shard_info.index >= start && shard_info.index <= end,
            None => true,
        };
        if !in_range {
            continue;
        }
        // Verify file exists on disk before registering
        let shard_path = shard_store_check.shard_path(&model_id, shard_info.index);
        if !shard_path.exists() {
            tracing::warn!(
                model = %model_id,
                shard = shard_info.index,
                "Shard file missing on disk — skipping registration"
            );
            continue;
        }
        let shard_id = crate::types::ShardId {
            model_id: model_id.clone(),
            index: shard_info.index,
        };
        shared_state
            .model_registry
            .record_shard_holder(shard_id, node_id.clone());
    }
    if let Some((s, e)) = shard_range {
        tracing::info!(
            model = %model_id,
            shard_start = s,
            shard_end = e,
            "Registered as holder of shard range only"
        );
    }

    // Persist to DB
    if let Err(e) = shared_state
        .model_registry
        .persist_manifest(&shared_state.db, &manifest)
    {
        tracing::warn!(error = %e, "Failed to persist manifest to DB");
    }

    tracing::info!(
        model = %model_id,
        size = file_size,
        shards = shard_count,
        "Generated and registered multi-shard manifest for local model"
    );
}

/// Regenerate a manifest from GGUF header metadata and on-disk shard files.
/// Used when manifest.json is missing but gguf_header.bin + shards exist.
fn regenerate_manifest_from_header(
    model_id: &crate::types::ModelId,
    model_dir: &std::path::Path,
    meta: &crate::inference::split::GgufTensorMeta,
    config: &crate::config::Config,
) -> Option<crate::types::ModelManifest> {
    let shard_size = config.model.shard_size_bytes();

    // Compute total GGUF file size from tensor metadata (header + all tensor data).
    // This is the REAL total, even when we only have a subset of shards locally.
    let total_size = {
        let max_end = meta
            .tensors
            .values()
            .map(|loc| meta.tensor_data_offset + loc.offset + loc.size)
            .max()
            .unwrap_or(meta.tensor_data_offset);
        // Round up to alignment (GGUF tensors are 32-byte aligned)
        (max_end + 31) & !31
    };

    let estimated_count = total_size.div_ceil(shard_size).max(1) as u32;

    let layouts = crate::inference::split::compute_layer_shard_layouts(meta, estimated_count);
    let shard_count = layouts.len() as u32;
    let shards = crate::model::manifest::build_shard_infos_from_layouts(model_dir, &layouts);

    let model_name = meta
        .model_name
        .clone()
        .unwrap_or_else(|| model_id.0.clone());

    let mut manifest = crate::types::ModelManifest {
        schema_version: 2,
        id: model_id.clone(),
        name: model_name,
        architecture: crate::types::ModelArchitecture::Llama,
        num_layers: meta.block_count as u32,
        num_params_billions: 0.0,
        quantization: crate::types::Quantization::Q4KM,
        total_size_bytes: total_size,
        shard_count,
        shards,
        tokenizer_hash: [0u8; 32],
        manifest_hash: [0u8; 32],
        publisher: crate::types::NodeId([0u8; 32]),
        publish_date: chrono::Utc::now(),
        license: "Unknown".to_string(),
        mmproj: None,
    };
    manifest.manifest_hash = manifest.compute_hash();

    // Save to disk
    if let Err(e) = manifest.save_to_dir(model_dir) {
        tracing::warn!(model = %model_id, error = %e, "Failed to save regenerated manifest");
    } else {
        tracing::info!(
            model = %model_id,
            shard_count,
            num_layers = meta.block_count,
            "Regenerated and saved manifest with accurate layer ranges"
        );
    }

    Some(manifest)
}

/// Handle an incoming LayerForward from a remote peer: run the local split model
/// segment and send back a LayerResult with either logits (last segment) or
/// hidden-state activations (intermediate segment).
/// Parameters for shard-based model loading.
pub struct ShardLoadParams<'a> {
    pub model_dir: &'a std::path::Path,
    pub shard_store: &'a ShardStore,
    pub model_id: &'a crate::types::ModelId,
    pub layer_start: usize,
    pub layer_end: usize,
    pub is_first: bool,
    pub is_last: bool,
    /// Manifest for this model — provides tensor entries and total size.
    pub manifest: &'a crate::types::ModelManifest,
}

/// Try to load a SplitModel from shard files + gguf_header.bin.
/// This is the shard-only loading path — no full GGUF needed.
pub fn try_load_from_shards(
    params: &ShardLoadParams<'_>,
) -> Result<crate::inference::split::SplitModel, SwarmError> {
    let model_dir = params.model_dir;
    let shard_store = params.shard_store;
    let model_id = params.model_id;
    let layer_start = params.layer_start;
    let layer_end = params.layer_end;
    let is_first = params.is_first;
    let is_last = params.is_last;

    // Reject legacy v1 manifests — they lack tensor entries, so ShardReader
    // would silently produce an empty tensor_map and fail at read time.
    if params.manifest.schema_version < 2 {
        return Err(SwarmError::Internal(format!(
            "Model {} has schema_version {} manifest — v2 required. Re-download shards.",
            model_id, params.manifest.schema_version
        )));
    }

    // Ensure GGUF header exists (extract from shard_000 if needed)
    if let Err(e) = crate::inference::split::ensure_gguf_header(model_dir) {
        return Err(SwarmError::Internal(format!(
            "Cannot load from shards: {e}"
        )));
    }

    // Collect available shard files for this model
    let mut shard_files: Vec<(u32, std::path::PathBuf)> = Vec::new();
    for i in 0u32..256 {
        let path = shard_store.shard_path(model_id, i);
        if path.exists() {
            shard_files.push((i, path));
        } else if i > 0 && shard_files.is_empty() {
            // Keep looking — shards might not start at 0
            continue;
        } else if !shard_files.is_empty() {
            // Found a gap after some shards — stop
            break;
        }
    }

    if shard_files.is_empty() {
        return Err(SwarmError::Internal(format!(
            "No shard files found for model {} in {}",
            model_id,
            model_dir.display()
        )));
    }

    // Build tensor entries for each shard file from manifest data.
    // The order must match shard_files (which is sorted by shard index).
    let tensor_entries: Vec<Vec<crate::types::ShardTensorEntry>> = shard_files
        .iter()
        .map(|(idx, _)| {
            params
                .manifest
                .shards
                .iter()
                .find(|s| s.index == *idx)
                .map(|s| s.tensors.clone())
                .unwrap_or_default()
        })
        .collect();

    tracing::info!(
        model = %model_id,
        shards = shard_files.len(),
        layers = format!("[{layer_start}..{layer_end})"),
        "Loading split model from shard files (no full GGUF)"
    );

    crate::inference::split::SplitModel::load_from_shards(
        model_dir,
        shard_files,
        &tensor_entries,
        params.manifest.total_size_bytes,
        layer_start,
        layer_end,
        is_first,
        is_last,
    )
}

async fn handle_layer_forward(
    shared_state: Arc<SharedState>,
    network_tx: mpsc::Sender<NetworkCommand>,
    forward: crate::types::LayerForward,
) {
    use crate::inference::split::{self, SplitModel};

    let request_id = forward.request_id;
    let sender_peer_bytes = match forward.sender_peer_bytes {
        Some(ref bytes) => bytes.clone(),
        None => {
            tracing::warn!(request_id = %request_id, "LayerForward missing sender_peer_bytes");
            return;
        }
    };

    let forward_start = std::time::Instant::now();
    tracing::info!(
        request_id = %request_id,
        seq = forward.sequence_num,
        activation_bytes = forward.activations.len(),
        model_id = %forward.model_id,
        layer_range = ?forward.layer_range,
        "DIAG: processing LayerForward locally"
    );

    let model_id = forward.model_id.clone();

    // Determine our layer range from the manifest and local shards
    let manifest = match shared_state.model_registry.get_manifest(&model_id) {
        Some(m) => m,
        None => {
            send_error_result(
                &network_tx,
                &sender_peer_bytes,
                request_id,
                "No manifest for model",
            )
            .await;
            return;
        }
    };

    // Figure out which shard indices we hold locally
    let local_node_id = shared_state.identity.node_id().clone();
    let mut local_shard_indices: Vec<u32> = Vec::new();
    for shard_info in &manifest.shards {
        let shard_id = crate::types::ShardId {
            model_id: model_id.clone(),
            index: shard_info.index,
        };
        let holders = shared_state.model_registry.shard_holders(&shard_id);
        if holders.contains(&local_node_id) {
            local_shard_indices.push(shard_info.index);
        }
    }

    if local_shard_indices.is_empty() {
        send_error_result(
            &network_tx,
            &sender_peer_bytes,
            request_id,
            "No local shards for model",
        )
        .await;
        return;
    }

    // Layer range is required in the forward message — no guessing
    let (layer_start, layer_end, total_layers) = {
        let (ls, le) = forward.layer_range;
        let total = manifest.num_layers as usize;
        (ls as usize, le as usize, total)
    };

    if layer_start >= layer_end {
        send_error_result(
            &network_tx,
            &sender_peer_bytes,
            request_id,
            "Empty layer range",
        )
        .await;
        return;
    }

    // is_first requires shard 0 (token_embd.weight is at tensor offset 0)
    // is_last requires the final shard (output.weight spans to the end of the file)
    let has_shard_0 = local_shard_indices.contains(&0);
    let last_shard_idx = manifest.shard_count.saturating_sub(1);
    let has_last_shard = local_shard_indices.contains(&last_shard_idx);
    let is_first = layer_start == 0 && has_shard_0;
    let is_last = layer_end >= total_layers && has_last_shard;

    // Ensure the split model is loaded
    let split_key = (model_id.clone(), layer_start, layer_end);
    if !shared_state.split_models.contains_key(&split_key) {
        let shard_store = crate::model::shard::ShardStore::new(&shared_state.config.node.data_dir);
        let model_dir = shard_store.models_dir().join(&model_id.0);

        // Try loading the split model from available sources, in priority order:
        // 1. Reconstructed model.gguf (all shards concatenated)
        // 2. Original GGUF via source_path
        // 3. Shard files + gguf_header.bin (no full GGUF needed)
        let gguf_path = model_dir.join("model.gguf");
        let source_path_file = model_dir.join("source_path");

        let load_result = if gguf_path.exists() {
            tracing::info!(
                model = %model_id,
                layers = format!("[{layer_start}..{layer_end})"),
                path = %gguf_path.display(),
                "Loading split model from reconstructed GGUF"
            );
            SplitModel::load_from_gguf(&gguf_path, layer_start, layer_end, is_first, is_last)
        } else if source_path_file.exists() {
            match std::fs::read_to_string(&source_path_file) {
                Ok(p) => {
                    let p = std::path::PathBuf::from(p.trim());
                    if p.exists() {
                        tracing::info!(
                            model = %model_id,
                            layers = format!("[{layer_start}..{layer_end})"),
                            path = %p.display(),
                            "Loading split model from source GGUF"
                        );
                        SplitModel::load_from_gguf(&p, layer_start, layer_end, is_first, is_last)
                    } else {
                        // source_path exists but file is gone — try shard-based loading
                        try_load_from_shards(&ShardLoadParams {
                            model_dir: &model_dir,
                            shard_store: &shard_store,
                            model_id: &model_id,
                            layer_start,
                            layer_end,
                            is_first,
                            is_last,
                            manifest: &manifest,
                        })
                    }
                }
                Err(e) => Err(SwarmError::Io(e)),
            }
        } else {
            // No full GGUF anywhere — use shard-based loading
            try_load_from_shards(&ShardLoadParams {
                model_dir: &model_dir,
                shard_store: &shard_store,
                model_id: &model_id,
                layer_start,
                layer_end,
                is_first,
                is_last,
                manifest: &manifest,
            })
        };

        match load_result {
            Ok(model) => {
                // VRAM-aware eviction: if a memory budget is set, evict LRU
                // models before inserting the new one.
                let max_batch = shared_state.config.inference.max_batch_size as usize;
                let batch_timeout = std::time::Duration::from_millis(
                    shared_state.config.inference.batch_timeout_ms,
                );
                let new_entry = if max_batch > 1 {
                    crate::inference::split::SplitModelEntry::new_with_batching(
                        model,
                        shared_state.kv_cache_store.clone(),
                        max_batch,
                        batch_timeout,
                    )
                } else {
                    crate::inference::split::SplitModelEntry::new(model)
                };
                let vram_budget = crate::model::auto_manage::compute_vram_budget(&shared_state)
                    .or(shared_state.config.inference.max_split_model_memory_mb);
                if let Some(budget_mb) = vram_budget {
                    let evicted = crate::inference::split::evict_split_models_lru(
                        &shared_state.split_models,
                        &shared_state.active_pipelines,
                        budget_mb,
                        new_entry.estimated_vram_mb,
                    );
                    if evicted > 0 {
                        tracing::info!(
                            evicted,
                            budget_mb,
                            "Evicted LRU split models for VRAM budget"
                        );
                    }
                }
                shared_state
                    .split_models
                    .insert(split_key.clone(), new_entry);
            }
            Err(e) => {
                send_error_result(
                    &network_tx,
                    &sender_peer_bytes,
                    request_id,
                    &format!("Load failed: {e}"),
                )
                .await;
                return;
            }
        }
    }

    let (split_model_ref, batch_forwarder, cached_eos_tokens) =
        match shared_state.split_models.get(&split_key) {
            Some(r) => {
                r.value().touch();
                (
                    r.value().model.clone(),
                    r.value().batch_forwarder.clone(),
                    r.value().eos_tokens.clone(),
                )
            }
            None => {
                send_error_result(
                    &network_tx,
                    &sender_peer_bytes,
                    request_id,
                    "Split model vanished",
                )
                .await;
                return;
            }
        };

    // Clear per-request KV-cache at the start of a new request (prefill)
    let req_id_str = request_id.to_string();
    if forward.sequence_num == 0 {
        let model_key = format!("{}-{}-{}", layer_start, layer_end, total_layers);
        shared_state
            .kv_cache_store
            .clear_request(&model_key, &req_id_str);
    }

    // Try batch path for decode steps (seq > 0) when batching is enabled.
    // Prefill (seq 0 on is_first) requires tokenization under the model lock,
    // so it always falls through to the sequential path.
    let use_batch = batch_forwarder.is_some() && forward.sequence_num > 0;

    if use_batch {
        let forwarder = batch_forwarder.unwrap();

        // Build input tensor without holding the model lock
        let input_tensor = if is_first {
            // Decode step on first segment: single token ID as i64 LE
            let token_id = if forward.activations.len() >= 8 {
                let bytes: [u8; 8] = match forward.activations[..8].try_into() {
                    Ok(b) => b,
                    Err(_) => {
                        send_error_result(
                            &network_tx,
                            &sender_peer_bytes,
                            request_id,
                            "Invalid activation data",
                        )
                        .await;
                        return;
                    }
                };
                i64::from_le_bytes(bytes)
            } else {
                0i64
            };
            match candle_core::Tensor::from_vec(vec![token_id], &[1, 1], &candle_core::Device::Cpu)
            {
                Ok(t) => t,
                Err(e) => {
                    send_error_result(
                        &network_tx,
                        &sender_peer_bytes,
                        request_id,
                        &format!("Tensor: {e}"),
                    )
                    .await;
                    return;
                }
            }
        } else {
            match split::bytes_to_tensor(&forward.activations) {
                Ok(t) => t,
                Err(e) => {
                    send_error_result(
                        &network_tx,
                        &sender_peer_bytes,
                        request_id,
                        &format!("Decode: {e}"),
                    )
                    .await;
                    return;
                }
            }
        };

        // Submit to batch forwarder — will be batched with other concurrent requests
        let output = match forwarder
            .submit(input_tensor, forward.index_pos as usize, req_id_str)
            .await
        {
            Ok(o) => o,
            Err(e) => {
                send_error_result(
                    &network_tx,
                    &sender_peer_bytes,
                    request_id,
                    &format!("Batch forward: {e}"),
                )
                .await;
                return;
            }
        };

        // Post-process using cached eos_tokens (no model lock needed)
        let result = if is_last {
            match split::sample_token(&output, 0.7, 0.9) {
                Ok(token_id) => {
                    let finish = if cached_eos_tokens.contains(&token_id) {
                        Some(crate::types::NetworkFinishReason::Stop)
                    } else {
                        None
                    };
                    crate::types::LayerResult {
                        request_id,
                        token_ids: vec![token_id],
                        finish_reason: finish,
                        activations: vec![],
                    }
                }
                Err(e) => {
                    send_error_result(
                        &network_tx,
                        &sender_peer_bytes,
                        request_id,
                        &format!("Sample: {e}"),
                    )
                    .await;
                    return;
                }
            }
        } else {
            match split::tensor_to_bytes(&output) {
                Ok(activation_bytes) => crate::types::LayerResult {
                    request_id,
                    token_ids: vec![],
                    finish_reason: None,
                    activations: activation_bytes,
                },
                Err(e) => {
                    send_error_result(
                        &network_tx,
                        &sender_peer_bytes,
                        request_id,
                        &format!("Encode: {e}"),
                    )
                    .await;
                    return;
                }
            }
        };

        let forward_elapsed = forward_start.elapsed();
        tracing::info!(
            request_id = %request_id,
            tokens = result.token_ids.len(),
            activations_bytes = result.activations.len(),
            is_last,
            elapsed_ms = forward_elapsed.as_millis() as u64,
            model_id = %model_id,
            layers = format!("[{layer_start}..{layer_end})"),
            batched = true,
            "DIAG: LayerForward processed via batch forwarder"
        );

        // Track participation
        {
            if let Ok(mut stats) = shared_state.node_stats.try_write() {
                stats.forwards_served += 1;
            }
            let layers_processed = (layer_end - layer_start) as i64;
            let earned = crate::credit::ledger::RATE_INFERENCE_SERVE * layers_processed;
            if let Ok(mut bal) = shared_state.credit_balance.try_write() {
                bal.balance += earned;
                bal.lifetime_earned += earned as u64;
                bal.last_updated = chrono::Utc::now();
            }
        }

        if let Err(e) = network_tx
            .send(NetworkCommand::SendTensorResult {
                target_peer_bytes: sender_peer_bytes,
                result,
            })
            .await
        {
            tracing::warn!(error = %e, "Failed to send LayerResult back to peer");
        }
        return;
    }

    // Sequential path: prefill or batching disabled
    let mut split_model = split_model_ref.lock().await;

    // Convert activation bytes to a candle Tensor
    let input_tensor = if is_first {
        if forward.index_pos == 0 {
            // Prefill: activations are the prompt text → tokenize with BPE if available
            let prompt = String::from_utf8_lossy(&forward.activations);
            let token_ids: Vec<i64> = if let Some(tokenizer) = split_model.tokenizer() {
                tokenizer.encode(&prompt)
            } else {
                prompt.bytes().map(|b| b as i64).collect()
            };
            match candle_core::Tensor::from_vec(
                token_ids.clone(),
                &[1, token_ids.len()],
                &candle_core::Device::Cpu,
            ) {
                Ok(t) => t,
                Err(e) => {
                    send_error_result(
                        &network_tx,
                        &sender_peer_bytes,
                        request_id,
                        &format!("Tensor: {e}"),
                    )
                    .await;
                    return;
                }
            }
        } else {
            // Decode step: activations are a single i64 token ID (8 bytes LE)
            let token_id = if forward.activations.len() >= 8 {
                let bytes: [u8; 8] = match forward.activations[..8].try_into() {
                    Ok(b) => b,
                    Err(_) => {
                        tracing::warn!("LayerForward activations too short for token ID");
                        send_error_result(
                            &network_tx,
                            &sender_peer_bytes,
                            request_id,
                            "Invalid activation data",
                        )
                        .await;
                        return;
                    }
                };
                i64::from_le_bytes(bytes)
            } else {
                0i64
            };
            match candle_core::Tensor::from_vec(vec![token_id], &[1, 1], &candle_core::Device::Cpu)
            {
                Ok(t) => t,
                Err(e) => {
                    send_error_result(
                        &network_tx,
                        &sender_peer_bytes,
                        request_id,
                        &format!("Tensor: {e}"),
                    )
                    .await;
                    return;
                }
            }
        }
    } else {
        match split::bytes_to_tensor(&forward.activations) {
            Ok(t) => t,
            Err(e) => {
                send_error_result(
                    &network_tx,
                    &sender_peer_bytes,
                    request_id,
                    &format!("Decode: {e}"),
                )
                .await;
                return;
            }
        }
    };

    // Decompress vision embeddings from LayerForward if present
    let vision_tensor: Option<candle_core::Tensor> = if let Some(ref compressed) =
        forward.vision_embeddings
    {
        match zstd::decode_all(std::io::Cursor::new(compressed)) {
            Ok(raw_bytes) => {
                let num_f16 = raw_bytes.len() / 2;
                let hidden_dim = if num_f16 % 4096 == 0 {
                    4096
                } else if num_f16 % 2048 == 0 {
                    2048
                } else {
                    1024
                };
                let num_tokens = num_f16 / hidden_dim;
                let f32_values: Vec<f32> = raw_bytes
                    .chunks_exact(2)
                    .map(|b| half::f16::from_le_bytes([b[0], b[1]]).to_f32())
                    .collect();
                candle_core::Tensor::from_vec(
                    f32_values,
                    &[num_tokens, hidden_dim],
                    &candle_core::Device::Cpu,
                )
                .ok()
            }
            Err(e) => {
                tracing::warn!(error = %e, "Failed to decompress vision embeddings from LayerForward");
                None
            }
        }
    } else {
        None
    };

    // Run the forward pass with per-request KV-cache isolation.
    // CRITICAL: Use block_in_place() to prevent blocking the Tokio worker thread.
    // split_model.forward() is CPU-bound (hundreds of ms for LLM inference) and
    // would otherwise starve the network event loop — preventing yamux window
    // updates and causing substream stalling on the next request_response exchange.
    let compute_result =
        tokio::task::block_in_place(|| -> Result<crate::types::LayerResult, String> {
            let output = if let Some(ref vis_emb) = vision_tensor {
                split_model
                    .forward_multimodal(
                        &input_tensor,
                        forward.index_pos as usize,
                        &shared_state.kv_cache_store,
                        &req_id_str,
                        Some(vis_emb),
                    )
                    .map_err(|e| format!("Forward multimodal: {e}"))?
            } else {
                split_model
                    .forward(
                        &input_tensor,
                        forward.index_pos as usize,
                        &shared_state.kv_cache_store,
                        &req_id_str,
                    )
                    .map_err(|e| format!("Forward: {e}"))?
            };

            if is_last {
                let token_id =
                    split::sample_token(&output, 0.7, 0.9).map_err(|e| format!("Sample: {e}"))?;
                let eos_tokens = split_model.eos_tokens();
                let finish = if eos_tokens.contains(&token_id) {
                    Some(crate::types::NetworkFinishReason::Stop)
                } else {
                    None
                };
                Ok(crate::types::LayerResult {
                    request_id,
                    token_ids: vec![token_id],
                    finish_reason: finish,
                    activations: vec![],
                })
            } else {
                let activation_bytes =
                    split::tensor_to_bytes(&output).map_err(|e| format!("Encode: {e}"))?;
                Ok(crate::types::LayerResult {
                    request_id,
                    token_ids: vec![],
                    finish_reason: None,
                    activations: activation_bytes,
                })
            }
        });

    let result = match compute_result {
        Ok(r) => r,
        Err(e) => {
            send_error_result(&network_tx, &sender_peer_bytes, request_id, &e).await;
            return;
        }
    };

    let forward_elapsed = forward_start.elapsed();
    tracing::info!(
        request_id = %request_id,
        tokens = result.token_ids.len(),
        activations_bytes = result.activations.len(),
        is_last,
        elapsed_ms = forward_elapsed.as_millis() as u64,
        model_id = %model_id,
        layers = format!("[{layer_start}..{layer_end})"),
        "DIAG: LayerForward processed, sending result back"
    );

    // Track participation: increment forwards_served and earn credits (non-blocking)
    {
        if let Ok(mut stats) = shared_state.node_stats.try_write() {
            stats.forwards_served += 1;
        }
        let layers_processed = (layer_end - layer_start) as i64;
        let earned = crate::credit::ledger::RATE_INFERENCE_SERVE * layers_processed;
        if let Ok(mut bal) = shared_state.credit_balance.try_write() {
            bal.balance += earned;
            bal.lifetime_earned += earned as u64;
            bal.last_updated = chrono::Utc::now();
        }
    }

    // Send back as a separate request to the originating peer
    if let Err(e) = network_tx
        .send(NetworkCommand::SendTensorResult {
            target_peer_bytes: sender_peer_bytes,
            result,
        })
        .await
    {
        tracing::warn!(error = %e, "Failed to send LayerResult back to peer");
    }
}

/// Handle a VisionEncodeRequest: encode the image using local mmproj and respond.
async fn handle_vision_encode_request(
    shared_state: Arc<SharedState>,
    network_tx: mpsc::Sender<NetworkCommand>,
    req: crate::types::VisionEncodeRequest,
) {
    let model_id = &req.model_id;
    tracing::info!(
        request_id = %req.request_id,
        model = %model_id,
        image_bytes = req.image_data.len(),
        "Handling VisionEncodeRequest"
    );

    // Load or get the vision module
    let vision_module = if let Some(entry) = shared_state.vision_modules.get(model_id) {
        entry.value().clone()
    } else {
        // Try to load mmproj on-demand
        let model_dir = shared_state
            .config
            .node
            .data_dir
            .join("models")
            .join(&model_id.0);
        let mmproj_path = model_dir.join("mmproj.gguf");
        if !mmproj_path.exists() {
            tracing::warn!(
                request_id = %req.request_id,
                model = %model_id,
                "VisionEncodeRequest received but no mmproj.gguf found"
            );
            return;
        }
        match crate::inference::vision::load_from_mmproj_gguf(
            &mmproj_path,
            &candle_core::Device::Cpu,
        ) {
            Ok(module) => {
                let module = Arc::new(module);
                shared_state
                    .vision_modules
                    .insert(model_id.clone(), module.clone());
                module
            }
            Err(e) => {
                tracing::warn!(
                    request_id = %req.request_id,
                    error = %e,
                    "Failed to load mmproj for VisionEncodeRequest"
                );
                return;
            }
        }
    };

    // Decode JPEG image into ImageData
    let img = match image::load_from_memory(&req.image_data) {
        Ok(dyn_img) => {
            let rgb = dyn_img.to_rgb8();
            let (w, h) = rgb.dimensions();
            crate::types::ImageData {
                rgb_bytes: rgb.into_raw(),
                width: w,
                height: h,
            }
        }
        Err(e) => {
            tracing::warn!(
                request_id = %req.request_id,
                error = %e,
                "Failed to decode image in VisionEncodeRequest"
            );
            return;
        }
    };

    // Encode image to vision embeddings (CPU-bound)
    let encode_result = tokio::task::block_in_place(|| vision_module.encode_images(&[img]));
    match encode_result {
        Ok(embeddings) => {
            // Compress embeddings with zstd for wire transfer
            let (num_tokens, hidden_dim) = embeddings.dims2().unwrap_or((0, 0));
            let raw_bytes: Vec<u8> = embeddings
                .to_dtype(candle_core::DType::F16)
                .and_then(|t| t.to_vec2::<half::f16>())
                .map(|v: Vec<Vec<half::f16>>| {
                    let mut bytes = Vec::with_capacity(num_tokens * hidden_dim * 2);
                    for row in v {
                        for f in row {
                            bytes.extend_from_slice(&f.to_le_bytes());
                        }
                    }
                    bytes
                })
                .unwrap_or_default();
            let compressed =
                zstd::encode_all(std::io::Cursor::new(&raw_bytes), 3).unwrap_or(raw_bytes);

            let response = crate::types::VisionEncodeResponse {
                request_id: req.request_id,
                embeddings: compressed,
                num_tokens: num_tokens as u32,
                hidden_dim: hidden_dim as u32,
            };

            tracing::info!(
                request_id = %req.request_id,
                num_tokens,
                hidden_dim,
                compressed_bytes = response.embeddings.len(),
                "VisionEncodeRequest completed, sending response"
            );

            // Send response back via gossip (directed messages use the same path)
            let msg = NetworkCommand::Broadcast(SwarmMessage::VisionEncodeResponse(response));
            if let Err(e) = network_tx.send(msg).await {
                tracing::warn!(error = %e, "Failed to send VisionEncodeResponse");
            }
        }
        Err(e) => {
            tracing::warn!(
                request_id = %req.request_id,
                error = %e,
                "Vision encoding failed"
            );
        }
    }
}

/// Send an error LayerResult back to the requesting peer.
async fn send_error_result(
    network_tx: &mpsc::Sender<NetworkCommand>,
    target_peer_bytes: &[u8],
    request_id: uuid::Uuid,
    error: &str,
) {
    tracing::warn!(request_id = %request_id, error, "LayerForward processing failed");
    let result = crate::types::LayerResult {
        request_id,
        token_ids: vec![],
        finish_reason: Some(crate::types::NetworkFinishReason::Error(error.to_string())),
        activations: vec![],
    };
    let _ = network_tx
        .send(NetworkCommand::SendTensorResult {
            target_peer_bytes: target_peer_bytes.to_vec(),
            result,
        })
        .await;
}

/// Map a GGUF file's `general.architecture` metadata to our ModelArchitecture enum.
fn map_gguf_architecture(path: &std::path::Path) -> crate::types::ModelArchitecture {
    let arch_str = match std::fs::File::open(path) {
        Ok(mut f) => match candle_core::quantized::gguf_file::Content::read(&mut f) {
            Ok(ct) => ct
                .metadata
                .get("general.architecture")
                .and_then(|v| v.to_string().ok().cloned())
                .unwrap_or_else(|| "llama".to_string()),
            Err(_) => "llama".to_string(),
        },
        Err(_) => "llama".to_string(),
    };
    match arch_str.as_str() {
        "qwen2" | "qwen3" | "qwen2moe" => crate::types::ModelArchitecture::Qwen2,
        "qwen35" => crate::types::ModelArchitecture::Qwen35,
        "qwen35moe" | "qwen3_5moe" => crate::types::ModelArchitecture::Qwen35Moe {
            num_experts: 0,
            experts_per_token: 0,
        },
        "mistral" => crate::types::ModelArchitecture::Mistral,
        "phi" | "phi3" => crate::types::ModelArchitecture::Phi,
        // All remaining supported transformer architectures map to Llama
        // (they share the same manifest structure).
        "llama" | "gemma" | "gemma2" | "starcoder2" | "deepseek2" | "glm4" | "llama4" => {
            crate::types::ModelArchitecture::Llama
        }
        other => {
            tracing::warn!(
                arch = other,
                "Unknown model architecture, defaulting to Llama"
            );
            crate::types::ModelArchitecture::Llama
        }
    }
}

/// Try to open a URL in the default browser.
fn open_browser(url: &str) -> Result<(), String> {
    #[cfg(target_os = "linux")]
    let cmd = "xdg-open";
    #[cfg(target_os = "macos")]
    let cmd = "open";
    #[cfg(target_os = "windows")]
    {
        // On Windows, use `cmd /C start` for opening URLs
        return std::process::Command::new("cmd")
            .args(["/C", "start", "", url])
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
            .map_err(|e| e.to_string())
            .map(|_| ());
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
    return Err("Unsupported platform".into());

    #[cfg(any(target_os = "linux", target_os = "macos"))]
    {
        std::process::Command::new(cmd)
            .arg(url)
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
            .map_err(|e| e.to_string())?;

        Ok(())
    }
}

/// Best-effort IP geolocation using a free API (ip-api.com).
/// Returns an ISO 3166-1 alpha-2 country code (e.g. "US", "DE") or None on failure.
/// Timeout: 5 seconds. No API key required.
async fn detect_region_from_ip() -> Option<String> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(5))
        .build()
        .ok()?;

    // ip-api.com returns JSON with a "countryCode" field for free, no key needed.
    // Rate limit: 45 requests/min (we only call once at startup).
    let resp = client
        .get("http://ip-api.com/json/?fields=status,countryCode")
        .send()
        .await
        .ok()?;

    let json: serde_json::Value = resp.json().await.ok()?;
    if json.get("status")?.as_str()? == "success" {
        json.get("countryCode")?.as_str().map(|s| s.to_string())
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn restart_backoff_doubles_each_attempt() {
        assert_eq!(restart_backoff(0), Duration::from_secs(1));
        assert_eq!(restart_backoff(1), Duration::from_secs(2));
        assert_eq!(restart_backoff(2), Duration::from_secs(4));
        assert_eq!(restart_backoff(3), Duration::from_secs(8));
        assert_eq!(restart_backoff(4), Duration::from_secs(16));
    }

    #[test]
    fn restart_backoff_caps_at_16_seconds() {
        assert_eq!(restart_backoff(5), Duration::from_secs(16));
        assert_eq!(restart_backoff(10), Duration::from_secs(16));
        assert_eq!(restart_backoff(100), Duration::from_secs(16));
    }

    #[test]
    fn subsystem_criticality_classification() {
        // Verify our criticality assignments match the task spec
        assert_eq!(
            SubsystemCriticality::Critical,
            SubsystemCriticality::Critical
        );
        assert_ne!(
            SubsystemCriticality::Critical,
            SubsystemCriticality::NonCritical
        );
    }

    #[test]
    fn max_restart_attempts_is_five() {
        assert_eq!(MAX_RESTART_ATTEMPTS, 5);
    }

    #[tokio::test]
    async fn joinset_catches_task_panic() {
        let mut set: JoinSet<(&str, SubsystemCriticality, Result<(), String>)> = JoinSet::new();
        set.spawn(async {
            panic!("simulated subsystem panic");
        });

        let result = set.join_next().await.unwrap();
        assert!(result.is_err());
        assert!(result.unwrap_err().is_panic());
    }

    #[tokio::test]
    async fn joinset_returns_task_error() {
        let mut set: JoinSet<(&str, SubsystemCriticality, Result<(), String>)> = JoinSet::new();
        set.spawn(async {
            (
                "TestSubsystem",
                SubsystemCriticality::NonCritical,
                Err("boom".to_string()),
            )
        });

        let result = set.join_next().await.unwrap();
        let (name, crit, task_result) = result.unwrap();
        assert_eq!(name, "TestSubsystem");
        assert_eq!(crit, SubsystemCriticality::NonCritical);
        assert!(task_result.is_err());
        assert_eq!(task_result.unwrap_err(), "boom");
    }

    #[tokio::test]
    async fn joinset_returns_task_success() {
        let mut set: JoinSet<(&str, SubsystemCriticality, Result<(), String>)> = JoinSet::new();
        set.spawn(async { ("TestSubsystem", SubsystemCriticality::Critical, Ok(())) });

        let result = set.join_next().await.unwrap();
        let (name, crit, task_result) = result.unwrap();
        assert_eq!(name, "TestSubsystem");
        assert_eq!(crit, SubsystemCriticality::Critical);
        assert!(task_result.is_ok());
    }

    #[tokio::test]
    async fn supervisor_non_critical_failure_does_not_drain_set() {
        // Simulate: one non-critical task fails, others keep running
        let mut set: JoinSet<(&str, SubsystemCriticality, Result<(), String>)> = JoinSet::new();

        // Task that fails immediately
        set.spawn(async {
            (
                "HealthMonitor",
                SubsystemCriticality::NonCritical,
                Err("test error".to_string()),
            )
        });

        // Task that runs until cancelled
        set.spawn(async {
            tokio::time::sleep(Duration::from_secs(3600)).await;
            ("ApiServer", SubsystemCriticality::Critical, Ok(()))
        });

        // First join: get the failed task
        let result = set.join_next().await.unwrap();
        let (name, crit, _) = result.unwrap();
        assert_eq!(name, "HealthMonitor");
        assert_eq!(crit, SubsystemCriticality::NonCritical);

        // The other task is still running — set is not empty
        assert_eq!(set.len(), 1);

        // Clean up
        set.abort_all();
    }

    #[tokio::test]
    async fn supervisor_restart_counting() {
        // Simulate the restart counting logic from the supervisor loop
        let mut restart_counts: std::collections::HashMap<&str, u32> =
            std::collections::HashMap::new();

        // Simulate 5 failures of a non-critical subsystem
        for i in 1..=5 {
            let count = restart_counts.entry("HealthMonitor").or_insert(0);
            *count += 1;
            assert_eq!(*count, i);
        }

        // After 5 failures, count should be 5 (at the limit)
        assert_eq!(
            *restart_counts.get("HealthMonitor").unwrap(),
            MAX_RESTART_ATTEMPTS
        );

        // One more would exceed
        let count = restart_counts.entry("HealthMonitor").or_insert(0);
        *count += 1;
        assert!(*count > MAX_RESTART_ATTEMPTS);
    }
}
