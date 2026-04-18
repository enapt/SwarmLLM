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
mod credits;
mod events;
mod hf;
mod metrics;
mod models;
mod tp_allreduce;

pub use activity::{ActivityEvent, DashboardSignal, LoadedModelInfo};
pub use credits::CreditPool;
pub use events::EventBus;
pub use hf::{HfProbeInfo, HfSource};
pub use metrics::{ChannelCounters, ChannelMetricsSet, MetricsProviders};
pub use models::ModelMgmt;
pub use tp_allreduce::TpAllReduceCollector;

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
    /// NodeIds currently connected at the libp2p transport layer.
    /// Populated by NetworkManager on Identify-Received, removed on ConnectionClosed
    /// when num_established transitions to 0. HealthMonitor uses this as ground truth
    /// to avoid evicting peer_registry entries for peers that are still connected but
    /// momentarily silent (e.g. slow WSL2 QUIC substream negotiation, no recent gossip).
    pub connected_node_ids: dashmap::DashSet<NodeId>,

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
    /// Remote-side (segment holder) pending result routes for forwards received
    /// on a persistent pipeline stream. Keyed by request_id; the handler task
    /// registers a oneshot before dispatch, and NetworkManager delivers the
    /// result here (instead of the request_response path) when a match is found.
    pub pending_stream_result_routes:
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
                stats_cache: parking_lot::Mutex::new(None),
                stats_building: std::sync::atomic::AtomicBool::new(false),
            },
            credits: CreditPool {
                credit_balance: Arc::new(RwLock::new(crate::types::CreditBalance {
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
                credit_percentile_cache: parking_lot::Mutex::new((std::time::Instant::now(), 0.5)),
                private_mode: std::sync::atomic::AtomicBool::new({
                    // Restore from DB, fall back to config
                    db.get_json::<bool>("pool_state", "private_mode")
                        .ok()
                        .flatten()
                        .unwrap_or(config.pool.private_mode)
                }),
                offline_mode: std::sync::atomic::AtomicBool::new({
                    db.get_json::<bool>("pool_state", "offline_mode")
                        .ok()
                        .flatten()
                        .unwrap_or(config.pool.offline_mode)
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
                shard_p2p_failed: dashmap::DashSet::new(),
            },
            events: EventBus {
                dashboard_tx: broadcast::channel(32).0,
                update_state: Arc::new(RwLock::new(crate::update::UpdateState::default())),
                activity_tx: broadcast::channel(256).0,
                activity_history: parking_lot::Mutex::new(VecDeque::new()),
            },
            // Root-level fields (not sub-structed)
            executor,
            draft_executor: Arc::new(tokio::sync::Mutex::new(
                crate::inference::executor::ModelExecutor::new(),
            )),
            loaded_model_info: RwLock::new(None),
            gpu_info,
            pending_layer_results: DashMap::new(),
            pending_stream_result_routes: DashMap::new(),
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
            .set_force_standard_attn(state.config.inference.force_standard_attn);
        state
            .model_process_pool
            .set_max_seq_len_override(state.config.inference.max_seq_len_override);

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
}
