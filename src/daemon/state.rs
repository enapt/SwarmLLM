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

pub struct SharedState {
    pub config: Config,
    pub identity: Identity,
    pub db: Database,
    pub peer_registry: DashMap<NodeId, PeerInfo>,
    pub model_registry: ModelRegistry,
    pub active_pipelines: DashMap<uuid::Uuid, PipelineAssignment>,
    pub credit_balance: Arc<RwLock<CreditBalance>>,
    /// Atomic accumulator for credits earned during forward participation.
    /// Avoids lock contention on credit_balance during high-concurrency forwards.
    /// Flushed to credit_balance by the CreditLedger periodic persist (every 60s).
    pub pending_credit_earn: std::sync::atomic::AtomicI64,
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
    /// Trust score manager — tracks per-peer reputation, persisted to redb.
    pub trust_manager: crate::credit::trust::TrustManager,
    /// Credit escrow manager — holds credits during large inference requests.
    pub escrow_manager: Arc<crate::credit::escrow::EscrowManager>,
    /// Auto-detected country code from IP geolocation (e.g. "US", "DE").
    /// Falls back to config.identity.region if geolocation fails.
    pub detected_region: RwLock<Option<String>>,
    /// Per-peer credit balance buckets from gossip, for leaderboard display.
    /// Keyed by NodeId, value is the latest gossiped balance bucket.
    pub peer_credit_balances: DashMap<NodeId, i64>,
    /// Paged KV cache pool for PagedAttention (CUDA-only, feature-gated).
    /// When `None`, callers fall back to `KvCacheStore` (Phase 1 pre-allocated buffers).
    #[cfg(feature = "paged-attn")]
    pub paged_kv_pool: Option<Arc<crate::inference::paged_kv::PagedKvPool>>,
    /// Paged KV store: per-request block table tracking.
    #[cfg(feature = "paged-attn")]
    pub paged_kv_store: Option<Arc<crate::inference::paged_kv::PagedKvStore>>,
    /// Per-model auto-manage policies (runtime-mutable, persisted to redb).
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
    // NOTE: prune_events_tx removed — prune events now flow through activity_tx.
    // PruneEvent struct kept for prune_history storage and the REST API.
    /// Recent prune events (capped at 100) for the prune history API.
    pub prune_history: RwLock<VecDeque<crate::types::PruneEvent>>,
    /// Per-shard lock/pin flags — locked shards are never auto-pruned.
    pub locked_shards: DashMap<crate::types::ShardId, bool>,
    /// Per-model trust metadata for auto-manage gating.
    /// Models must reach `DemandVerified` or be `Pinned` before auto-manage will
    /// download their shards. Prevents trash model propagation.
    pub model_trust: DashMap<crate::types::ModelId, crate::types::ModelTrustInfo>,
    /// Coordination for on-demand model loading. When an inference request arrives
    /// for a model with shards on disk but not loaded, only one task loads it;
    /// others wait on the Notify.
    pub loading_models: DashMap<crate::types::ModelId, Arc<tokio::sync::Notify>>,
    /// LoRA adapter registry for per-request fine-tuned inference.
    pub adapter_registry: Arc<crate::model::lora::AdapterRegistry>,
    /// Cross-request prefix cache for sharing KV state across requests with
    /// identical system prompts. Protected by std::sync::Mutex since operations
    /// Per-channel backpressure metrics (capacity, sent, dropped).
    pub channel_metrics: ChannelMetricsSet,
    /// Number of peers discovered via mDNS (LAN peers).
    pub lan_peer_count: std::sync::atomic::AtomicUsize,
    /// Broadcast channel fired whenever the peer_registry changes (connect/disconnect).
    /// WebSocket subscribers push a full `peer_list` message so the dashboard updates live.
    pub peer_list_changed_tx: broadcast::Sender<()>,
    /// Runtime-mutable cloud provider configuration (API keys for Anthropic, OpenAI, etc.).
    pub providers_config: RwLock<crate::config::ProvidersConfig>,
    /// Update checker shared state (version info, last checked, etc.).
    pub update_state: Arc<RwLock<crate::update::UpdateState>>,
    /// Broadcast channel for update availability notifications (WebSocket push).
    pub update_tx: broadcast::Sender<crate::update::UpdateInfo>,
    /// Broadcast channel fired when models change (shard download, load, prune).
    /// WebSocket subscribers push a `models_changed` event so the dashboard auto-refreshes.
    pub models_changed_tx: broadcast::Sender<()>,
    // NOTE: system_notify_tx removed — system notifications now flow through activity_tx
    // with toast_level set.
    /// Broadcast channel for rich activity events (verbose dashboard activity log).
    /// All subsystems emit events here; WebSocket pushes them as `activity_event` messages.
    pub activity_tx: broadcast::Sender<ActivityEvent>,
    /// Recent activity events for replaying to new WebSocket clients.
    /// Capped at 100 entries (ring buffer). New clients receive the full history
    /// on connect so they see startup events even after page refresh.
    pub activity_history: std::sync::Mutex<VecDeque<ActivityEvent>>,
    /// Cached mapping of cloud provider model IDs to provider names.
    /// Populated by `list_provider_models` so that `try_proxy_openai` can route
    /// models whose ID doesn't match a known prefix (e.g. NVIDIA NIM `01-ai/yi-large`).
    pub provider_model_map: DashMap<String, String>,
    /// Cached provider model list — avoids hitting every provider API on each page load.
    /// Tuple: (models JSON array, timestamp of last successful fetch).
    pub provider_models_cache: RwLock<(Vec<serde_json::Value>, std::time::Instant)>,
    /// Active WebSocket connection count — capped to prevent resource exhaustion.
    pub ws_connection_count: std::sync::atomic::AtomicUsize,
    /// Persistent NodeId → PeerId bytes mapping. Populated by the identify handler
    /// and NEVER cleared on disconnect. Solves the race where the scheduler picks a
    /// peer from shard_holders but the peer_registry entry was removed on disconnect
    /// and hasn't been re-created by identify yet.
    pub peer_id_map: DashMap<NodeId, Vec<u8>>,
    /// Loaded VLM vision modules (mmproj encoders) keyed by model ID.
    /// Populated when an mmproj.gguf is found alongside a model's shards.
    pub vision_modules: DashMap<crate::types::ModelId, Arc<crate::inference::vision::VisionModule>>,
    /// Per-model encrypted pipeline toggle. When enabled for a model, the pipeline
    /// scheduler forces both first and last segments to be the requesting (local) node,
    /// ensuring no remote node sees plaintext input or output. The global
    /// `config.inference.encrypted_pipeline` serves as a fallback when no per-model
    /// override exists. Persisted to DB under "encrypted_pipeline_models".
    pub encrypted_pipeline_models: DashMap<crate::types::ModelId, bool>,
    /// Local embedding tables for privacy mode. When available, the requesting node
    /// performs token→embedding locally so remote first-segment nodes never see raw tokens.
    /// Keyed by model ID, loaded from shard_000.bin during startup scan.
    pub local_embedders:
        DashMap<crate::types::ModelId, Arc<crate::inference::local_embedder::LocalEmbedder>>,
    /// Pending VisionEncodeResponse channels for distributed vision encoding.
    /// Keyed by request_id. Pipeline registers a (expected_responder, oneshot sender) before
    /// sending VisionEncodeRequest to a remote mmproj holder; the network dispatcher fires it
    /// when the VisionEncodeResponse arrives from the expected responder.
    pub pending_vision_results: DashMap<
        uuid::Uuid,
        (
            crate::types::NodeId,
            tokio::sync::oneshot::Sender<crate::types::VisionEncodeResponse>,
        ),
    >,
    /// Pending tensor-parallel AllReduce partials, keyed by (request_id, layer_idx).
    /// Coordinator collects partials from all TP ranks, sums them, and responds.
    pub pending_tp_partials: DashMap<(uuid::Uuid, u32), TpAllReduceCollector>,
    /// AllReduce response registry — pipeline executors register here to receive
    /// reduced tensors after the coordinator completes the allreduce.
    pub allreduce_registry: Arc<crate::inference::allreduce::AllReduceRegistry>,
    /// Cumulative bytes served for shard transfers since last credit tick.
    /// NetworkManager increments on each chunk served; CreditLedger drains periodically.
    pub shard_bytes_served: AtomicU64,
    /// Cumulative relay circuit seconds since last credit tick.
    /// Incremented when relay circuits close; CreditLedger drains periodically.
    pub relay_seconds_served: AtomicU64,
    /// Active relay circuits: maps (src_peer, dst_peer) to start time.
    pub active_relay_circuits: DashMap<(libp2p::PeerId, libp2p::PeerId), std::time::Instant>,
    /// Model process pool: one subprocess per loaded model for GPU memory isolation.
    pub model_process_pool: Arc<crate::inference::process_pool::ModelProcessPool>,
    /// Regional shard summaries from gossip. Key: (region, model_id).
    /// Updated by RegionShardSummary gossip dispatch + local broadcast.
    pub region_shard_summaries:
        DashMap<(String, crate::types::ModelId), crate::types::RegionShardSummary>,
    /// Per-(model, region) EMA-decayed request rate from demand gossip.
    /// Updated by ModelDemandGossip dispatch + local popularity decay tick.
    pub region_demand: DashMap<(crate::types::ModelId, String), f64>,
    /// S5: Channel for requesting DHT shard provider queries.
    /// Subsystems (scheduler, auto-manage) send ModelId here to trigger
    /// `kademlia.get_providers()` for all shards of that model. Results
    /// are merged into `model_registry.shard_holders` automatically.
    pub dht_query_tx: mpsc::Sender<crate::types::ModelId>,
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
            credit_balance: Arc::new(RwLock::new(CreditBalance {
                node_id,
                balance: 0,
                lifetime_earned: 0,
                lifetime_spent: 0,
                last_updated: chrono::Utc::now(),
            })),
            pending_credit_earn: std::sync::atomic::AtomicI64::new(0),
            node_stats: RwLock::new(NodeStats::default()),
            executor,
            draft_executor: Arc::new(tokio::sync::Mutex::new(
                crate::inference::executor::ModelExecutor::new(),
            )),
            loaded_model_info: RwLock::new(None),
            gpu_info,
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
            escrow_manager,
            detected_region: RwLock::new(None),
            peer_credit_balances: DashMap::new(),
            #[cfg(feature = "paged-attn")]
            paged_kv_pool: None,
            model_auto_manage_policies,
            auto_manage_default_model_cap: AtomicU32::new(default_model_shard_cap),
            hf_probe_cache: DashMap::new(),
            model_request_counts: DashMap::new(),
            resource_schedule: RwLock::new(config.resources.schedule.clone()),
            // prune_events_tx removed (unified into activity_tx)
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
            model_trust: {
                // Load persisted trust info from database
                let map = DashMap::new();
                if let Ok(pairs) = db.get_all_json::<crate::types::ModelTrustInfo>("model_trust") {
                    for (key, info) in pairs {
                        map.insert(crate::types::ModelId(key), info);
                    }
                }
                map
            },
            loading_models: DashMap::new(),
            adapter_registry: Arc::new(crate::model::lora::AdapterRegistry::new(
                &config.node.data_dir,
            )),
            channel_metrics: ChannelMetricsSet::new(),
            lan_peer_count: std::sync::atomic::AtomicUsize::new(0),
            // lan_discovery_tx removed (unified into activity_tx)
            peer_list_changed_tx: broadcast::channel(4).0,
            providers_config: RwLock::new({
                // Hydrate from database (persisted via admin API), fall back to config.
                // Database values may be encrypted — decrypt using the node's signing key.
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
                // Fill any unconfigured providers from environment variables (.env or shell)
                pc.fill_from_env();
                pc
            }),
            update_state: Arc::new(RwLock::new(crate::update::UpdateState::default())),
            update_tx: broadcast::channel(4).0,
            models_changed_tx: broadcast::channel(16).0,
            // system_notify_tx removed (unified into activity_tx)
            activity_tx: broadcast::channel(256).0,
            activity_history: std::sync::Mutex::new(VecDeque::new()),
            provider_model_map: DashMap::new(),
            provider_models_cache: RwLock::new((Vec::new(), std::time::Instant::now())),
            ws_connection_count: std::sync::atomic::AtomicUsize::new(0),
            peer_id_map: DashMap::new(),
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

        // Wire activity_tx into the process pool for spawn/unload notifications
        state
            .model_process_pool
            .set_activity_tx(state.activity_tx.clone());

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

    /// Emit a rich activity event to the dashboard.
    /// Lightweight fire-and-forget — if no WebSocket subscribers, the event is dropped.
    pub fn emit_activity(&self, event: ActivityEvent) {
        // Store in history for replay to new WS clients
        if let Ok(mut history) = self.activity_history.lock() {
            history.push_back(event.clone());
            if history.len() > 100 {
                history.pop_front();
            }
        }
        let _ = self.activity_tx.send(event);
    }
}
