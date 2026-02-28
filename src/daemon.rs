use std::sync::Arc;

use dashmap::DashMap;
use tokio::sync::{mpsc, watch, RwLock};

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
    CreditBalance, NetworkCommand, NodeId, NodeStats, PeerInfo, PipelineAssignment, RebalanceEvent,
    ShardId, SwarmMessage,
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
    /// Which nodes have which shards — spec-required top-level field.
    pub shard_registry: DashMap<ShardId, Vec<NodeId>>,
    pub active_pipelines: DashMap<uuid::Uuid, PipelineAssignment>,
    pub credit_balance: Arc<RwLock<CreditBalance>>,
    pub node_stats: RwLock<NodeStats>,
    pub executor: SharedExecutor,
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
    pub split_models: DashMap<
        (crate::types::ModelId, usize, usize),
        Arc<tokio::sync::Mutex<crate::inference::split::SplitModel>>,
    >,
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
    /// API Bearer token for authentication.
    pub api_key: String,
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
    shutdown_tx: watch::Sender<bool>,
}

/// Tracks the HuggingFace origin of a model for re-downloading shards.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct HfSource {
    pub repo_id: String,
    pub filename: String,
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
        // Derive gossip network ID from sorted bootstrap addresses so nodes on the
        // same network share gossip encryption keys. Standalone nodes use the default.
        let gossip_network_id = {
            let mut bootstraps: Vec<String> = config
                .network
                .bootstrap_peers
                .iter()
                .map(|s| s.to_string())
                .collect();
            bootstraps.sort();
            if bootstraps.is_empty() {
                b"swarmllm-default-network".to_vec()
            } else {
                let mut h = blake3::Hasher::new();
                h.update(b"swarmllm-network-id-v1");
                for b in &bootstraps {
                    h.update(b.as_bytes());
                }
                h.finalize().as_bytes().to_vec()
            }
        };
        let gossip_sealer = Arc::new(crate::crypto::GossipSealer::new(&gossip_network_id));

        // Load persisted HF sources from sled
        let hf_sources = {
            let map = DashMap::new();
            if let Ok(tree) = db.tree("hf_sources") {
                for (key, value) in tree.iter().flatten() {
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
        let state = Arc::new(Self {
            config,
            identity,
            db,
            peer_registry: DashMap::new(),
            model_registry,
            shard_registry: DashMap::new(),
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
            loaded_model_info: RwLock::new(None),
            gpu_info,
            model_vote_tallies: DashMap::new(),
            acquisition_progress: DashMap::new(),
            pending_layer_results: DashMap::new(),
            split_models: DashMap::new(),
            gguf_meta: DashMap::new(),
            nickname_registry,
            session_manager,
            gossip_sealer,
            pool_state: RwLock::new(None),
            pool_registry: DashMap::new(),
            pool_tx: RwLock::new(None),
            api_key,
            model_loaded: std::sync::atomic::AtomicBool::new(false),
            auto_manage_enabled: std::sync::atomic::AtomicBool::new(auto_manage_enabled),
            anti_gaming: tokio::sync::Mutex::new(crate::credit::anti_gaming::AntiGaming::new()),
            streaming_token_txs: DashMap::new(),
            hf_sources,
            auto_manage_notify: Arc::new(tokio::sync::Notify::new()),
            peer_shard_downloads: DashMap::new(),
            shutdown_tx,
        });

        (state, shutdown_rx)
    }

    /// Signal all tasks to shut down.
    pub fn shutdown(&self) {
        let _ = self.shutdown_tx.send(true);
    }
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
        let (shared_state, shutdown_rx) = SharedState::new(
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
                                .record_shard_holder(shard_id.clone(), node_id.clone());
                            let mut holders =
                                shared_state.shard_registry.entry(shard_id).or_default();
                            if !holders.contains(&node_id) {
                                holders.push(node_id.clone());
                            }
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
                        .record_shard_holder(shard_id.clone(), node_id.clone());
                    let mut holders = shared_state.shard_registry.entry(shard_id).or_default();
                    if !holders.contains(&node_id) {
                        holders.push(node_id);
                    }
                }
            }
            Err(e) => {
                tracing::warn!(error = %e, "Failed to scan local shards");
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

        let network_handle = tokio::spawn(async move {
            if let Err(e) = network_manager.run().await {
                tracing::error!(error = %e, "NetworkManager exited with error");
            }
        });

        // Spawn InferenceRouter
        let inference_router = InferenceRouter::new(
            shared_state.clone(),
            router_cmd_rx,
            network_tx.clone(),
            shutdown_rx.clone(),
        );

        let inference_handle = tokio::spawn(async move {
            if let Err(e) = inference_router.run().await {
                tracing::error!(error = %e, "InferenceRouter exited with error");
            }
        });

        // Spawn message dispatcher: routes network inbound messages to the right subsystem
        let dispatcher_credit_balances: Arc<RwLock<Vec<i64>>> = Arc::new(RwLock::new(Vec::new()));
        let dispatcher_router_tx = router_cmd_tx.clone();
        let dispatcher_shutdown = shutdown_rx.clone();
        let dispatcher_credit_ref = dispatcher_credit_balances.clone();
        let dispatcher_state = shared_state.clone();
        let dispatcher_network_tx = network_tx.clone();
        let dispatcher_handle = tokio::spawn(async move {
            dispatch_network_messages(
                &mut network_out_rx,
                &dispatcher_router_tx,
                dispatcher_credit_ref,
                &dispatcher_state,
                dispatcher_network_tx,
                dispatcher_shutdown,
            )
            .await;
        });

        // Spawn HealthMonitor
        let health_monitor = HealthMonitor::new(
            shared_state.clone(),
            network_tx.clone(),
            rebalance_tx,
            shutdown_rx.clone(),
        );

        let health_handle = tokio::spawn(async move {
            if let Err(e) = health_monitor.run().await {
                tracing::error!(error = %e, "HealthMonitor exited with error");
            }
        });

        // Spawn ShardRebalancer
        let shard_rebalancer = ShardRebalancer::new(
            shared_state.clone(),
            rebalance_rx,
            network_tx.clone(),
            acquisition_tx.clone(),
            shutdown_rx.clone(),
        );

        let rebalancer_handle = tokio::spawn(async move {
            if let Err(e) = shard_rebalancer.run().await {
                tracing::error!(error = %e, "ShardRebalancer exited with error");
            }
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

        let credit_handle = tokio::spawn(async move {
            if let Err(e) = credit_ledger.run().await {
                tracing::error!(error = %e, "CreditLedger exited with error");
            }
        });

        // Spawn AcquisitionManager
        let acquisition_manager = AcquisitionManager::new(
            shared_state.clone(),
            network_tx.clone(),
            acquisition_rx,
            shutdown_rx.clone(),
        );

        let acquisition_handle = tokio::spawn(async move {
            if let Err(e) = acquisition_manager.run().await {
                tracing::error!(error = %e, "AcquisitionManager exited with error");
            }
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
        let pool_handle = tokio::spawn(async move {
            if let Err(e) = pool_manager.run().await {
                tracing::error!(error = %e, "PoolManager exited with error");
            }
        });

        // Spawn AutoShardManager (10th subsystem task — optional, runs only if enabled)
        let auto_manage = crate::model::auto_manage::AutoShardManager::new(
            shared_state.clone(),
            network_tx.clone(),
            shutdown_rx.clone(),
        );
        let auto_manage_handle = tokio::spawn(async move {
            auto_manage.run().await;
        });

        // Spawn API server (pass router_cmd_tx + acquisition_tx + network_tx so API can submit requests)
        let api_shared_state = shared_state.clone();
        let api_router_tx = router_cmd_tx.clone();
        let api_acquisition_tx = acquisition_tx.clone();
        let api_network_tx = network_tx.clone();
        let api_handle = tokio::spawn(async move {
            if let Err(e) = crate::api::server::run_server_with_state(
                api_shared_state,
                api_router_tx,
                api_acquisition_tx,
                api_network_tx,
            )
            .await
            {
                tracing::error!(error = %e, "API server exited with error");
            }
        });

        tracing::info!(
            node_id = %self.identity.node_id(),
            port = self.config.node.listen_port,
            "SwarmLLM daemon running"
        );

        // Broadcast shard announcements and manifests shortly after startup
        // so peers discover our shards quickly (don't wait for the 30s health tick).
        {
            let announce_state = shared_state.clone();
            let announce_tx = network_tx.clone();
            tokio::spawn(async move {
                // Wait for peer connections to establish
                tokio::time::sleep(std::time::Duration::from_secs(5)).await;

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

        // Spawn key rotation task (evicts stale encryption sessions)
        {
            let rotation_sm = shared_state.session_manager.clone();
            let rotation_shutdown = shutdown_rx.clone();
            tokio::spawn(async move {
                crate::crypto::key_rotation::run_key_rotation(rotation_sm, rotation_shutdown).await;
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
            tokio::spawn(async move {
                // Small delay to let the server bind
                tokio::time::sleep(std::time::Duration::from_millis(500)).await;
                if let Err(e) = open_browser(&target) {
                    tracing::debug!(error = %e, "Could not open browser automatically");
                }
            });
        }

        // Auto-load models that have local shards available
        {
            let sm = shared_state.clone();
            tokio::spawn(async move {
                // Brief delay to let shard announcements propagate
                tokio::time::sleep(std::time::Duration::from_secs(2)).await;
                let manifests = sm.model_registry.list_models();
                for m in &manifests {
                    if sm.split_models.iter().any(|e| e.key().0 == m.id) {
                        continue;
                    }
                    crate::model::auto_manage::check_and_load_model(&sm, &m.id).await;
                }
            });
        }

        // Wait for shutdown signal or task exit
        tokio::select! {
            _ = async {
                let ctrl_c = tokio::signal::ctrl_c();
                #[cfg(unix)]
                {
                    let mut sigterm = tokio::signal::unix::signal(
                        tokio::signal::unix::SignalKind::terminate(),
                    ).expect("Failed to register SIGTERM handler");
                    tokio::select! {
                        _ = ctrl_c => {
                            tracing::info!("Shutdown signal received (Ctrl+C)");
                        }
                        _ = sigterm.recv() => {
                            tracing::info!("Shutdown signal received (SIGTERM)");
                        }
                    }
                }
                #[cfg(not(unix))]
                {
                    ctrl_c.await.ok();
                    tracing::info!("Shutdown signal received (Ctrl+C)");
                }
            } => {}
            result = network_handle => {
                tracing::error!(?result, "NetworkManager task exited");
            }
            result = inference_handle => {
                tracing::error!(?result, "InferenceRouter task exited");
            }
            result = health_handle => {
                tracing::error!(?result, "HealthMonitor task exited");
            }
            result = rebalancer_handle => {
                tracing::error!(?result, "ShardRebalancer task exited");
            }
            result = credit_handle => {
                tracing::error!(?result, "CreditLedger task exited");
            }
            result = acquisition_handle => {
                tracing::error!(?result, "AcquisitionManager task exited");
            }
            result = pool_handle => {
                tracing::error!(?result, "PoolManager task exited");
            }
            result = api_handle => {
                tracing::error!(?result, "API server task exited");
            }
            result = dispatcher_handle => {
                tracing::error!(?result, "Message dispatcher task exited");
            }
            result = auto_manage_handle => {
                tracing::error!(?result, "AutoShardManager task exited");
            }
        }

        // Signal graceful shutdown
        shared_state.shutdown();

        // Flush database before exiting to ensure all pending writes are persisted
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
                                tracing::debug!(
                                    request_id = %result.request_id,
                                    tokens = result.token_ids.len(),
                                    activations_bytes = result.activations.len(),
                                    "Received LayerResult from remote segment"
                                );
                                if let Some((_, tx)) = shared_state
                                    .pending_layer_results
                                    .remove(&result.request_id)
                                {
                                    let _ = tx.send(result.clone());
                                } else {
                                    tracing::warn!(
                                        request_id = %result.request_id,
                                        "No pending channel for LayerResult — dropped"
                                    );
                                }
                            }
                            // LayerForward: process locally using split inference engine,
                            // then send back a LayerResult to the requesting node.
                            SwarmMessage::LayerForward(forward) => {
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
                                if let Some(tx) = shared_state
                                    .streaming_token_txs
                                    .get(&token.request_id)
                                {
                                    if tx.send(token.clone()).await.is_err() {
                                        tracing::debug!(
                                            request_id = %token.request_id,
                                            "Streaming token channel closed"
                                        );
                                        shared_state.streaming_token_txs.remove(&token.request_id);
                                    }
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
                                tracing::debug!(
                                    peer = %gossip.node_id,
                                    bucket = gossip.balance_bucket,
                                    "Received credit gossip"
                                );
                                let mut balances = credit_peer_balances.write().await;
                                balances.push(gossip.balance_bucket);
                                if balances.len() > 1000 {
                                    let excess = balances.len() - 1000;
                                    balances.drain(..excess);
                                }
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
                                    {
                                        let mut holders = shared_state.shard_registry
                                            .entry(shard_id.clone())
                                            .or_default();
                                        if !holders.contains(&announce.node_id) {
                                            holders.push(announce.node_id.clone());
                                        }
                                    }
                                    // Also register in model_registry so auto-acquire
                                    // can see shard coverage across the network
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
                                // Verify the manifest hash before trusting it
                                match manifest.verify_hash() {
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
                                        {
                                            let mut holders = shared_state.shard_registry
                                                .entry(progress.shard_id.clone())
                                                .or_default();
                                            if !holders.contains(&progress.node_id) {
                                                holders.push(progress.node_id.clone());
                                            }
                                        }
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
    // 1. Explicit key in config takes priority
    if let Some(ref key) = config.api.api_key {
        if !key.is_empty() {
            tracing::info!("Using API key from configuration");
            return key.clone();
        }
    }

    // 2. Check persisted key in database
    if let Ok(Some(key)) = db.get_json::<String>("config", "api_key") {
        if !key.is_empty() {
            tracing::info!("Using persisted API key from database");
            return key;
        }
    }

    // 3. Generate a new 32-byte hex key
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let mut bytes = [0u8; 32];
    rng.fill(&mut bytes);
    let key = hex::encode(bytes);

    // Persist to DB
    if let Err(e) = db.put_json("config", "api_key", &key) {
        tracing::warn!(error = %e, "Failed to persist API key to database");
    }

    // Print API key to stderr only (not to tracing logs which may be persisted/shipped)
    eprintln!("Generated new API key (save this for API access): {key}");

    key
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
                    .record_shard_holder(shard_id.clone(), node_id.clone());
                let mut holders = shared_state.shard_registry.entry(shard_id).or_default();
                if !holders.contains(&node_id) {
                    holders.push(node_id.clone());
                }
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
    let shard_count = file_size.div_ceil(shard_size).max(1) as u32;

    // Compute per-shard BLAKE3 hashes by reading byte ranges (blocking I/O)
    let hash_path = path.to_path_buf();
    let mut shards = match tokio::task::block_in_place(|| {
        compute_shard_hashes(&hash_path, file_size, shard_size)
    }) {
        Ok(s) => s,
        Err(e) => {
            tracing::warn!(error = %e, "Failed to compute shard hashes");
            return;
        }
    };

    // Extract model metadata from GGUF header (num_layers, architecture, etc.)
    // This MUST happen before computing per-shard layer ranges below.
    let (num_layers, architecture) =
        match crate::inference::split::GgufTensorMeta::from_gguf_file(path) {
            Ok(meta) => {
                let num_layers = meta.block_count as u32;
                tracing::info!(
                    model = %model_id,
                    num_layers,
                    embedding_length = meta.embedding_length,
                    "Extracted GGUF metadata for manifest"
                );
                // Assign layer ranges per shard using actual GGUF tensor positions.
                // The naive approach (num_layers / shard_count) is inaccurate because
                // layer tensors don't align to shard byte boundaries.
                for shard in &mut shards {
                    let (ls, le) = crate::inference::split::compute_local_layer_range(
                        &meta,
                        shard_size,
                        &[shard.index],
                    );
                    shard.layer_range = (ls as u32, le as u32);
                }
                // Store the metadata for later use in layer range computation
                shared_state.gguf_meta.insert(model_id.clone(), meta);
                // Map GGUF general.architecture string to our ModelArchitecture enum
                let arch = map_gguf_architecture(path);
                (num_layers, arch)
            }
            Err(e) => {
                tracing::warn!(error = %e, "Failed to extract GGUF metadata, using defaults");
                (0u32, crate::types::ModelArchitecture::Llama)
            }
        };

    let mut manifest = crate::types::ModelManifest {
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
            .record_shard_holder(shard_id.clone(), node_id.clone());
        let mut holders = shared_state.shard_registry.entry(shard_id).or_default();
        if !holders.contains(&node_id) {
            holders.push(node_id.clone());
        }
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

    let shard_count = total_size.div_ceil(shard_size).max(1) as u32;

    let mut shards = Vec::new();
    for idx in 0..shard_count {
        let shard_path = model_dir.join(format!("shard_{idx:03}.bin"));
        // For shards we don't have locally, compute expected size
        let expected_size = if idx == shard_count - 1 {
            total_size - (idx as u64) * shard_size
        } else {
            shard_size
        };
        let file_size = std::fs::metadata(&shard_path)
            .map(|m| m.len())
            .unwrap_or(expected_size);

        let (ls, le) = crate::inference::split::compute_local_layer_range(meta, shard_size, &[idx]);

        let hash = if shard_path.exists() {
            match std::fs::read(&shard_path) {
                Ok(data) => *blake3::hash(&data).as_bytes(),
                Err(_) => [0u8; 32],
            }
        } else {
            [0u8; 32]
        };

        shards.push(crate::types::ShardInfo {
            index: idx,
            layer_range: (ls as u32, le as u32),
            size_bytes: file_size,
            hash,
        });
    }

    let model_name = meta
        .model_name
        .clone()
        .unwrap_or_else(|| model_id.0.clone());

    let mut manifest = crate::types::ModelManifest {
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

/// Split a file into byte-range shards and compute BLAKE3 hash for each.
fn compute_shard_hashes(
    path: &std::path::Path,
    file_size: u64,
    shard_size: u64,
) -> Result<Vec<crate::types::ShardInfo>, SwarmError> {
    use std::io::{Read, Seek, SeekFrom};

    let mut file = std::fs::File::open(path).map_err(SwarmError::Io)?;
    let shard_count = file_size.div_ceil(shard_size).max(1);
    let mut shards = Vec::with_capacity(shard_count as usize);

    for i in 0..shard_count {
        let offset = i * shard_size;
        let this_shard_size = shard_size.min(file_size - offset);

        file.seek(SeekFrom::Start(offset)).map_err(SwarmError::Io)?;

        let mut hasher = blake3::Hasher::new();
        let mut remaining = this_shard_size;
        let mut buf = [0u8; 64 * 1024];

        while remaining > 0 {
            let to_read = (remaining as usize).min(buf.len());
            let n = file.read(&mut buf[..to_read]).map_err(SwarmError::Io)?;
            if n == 0 {
                break;
            }
            hasher.update(&buf[..n]);
            remaining -= n as u64;
        }

        shards.push(crate::types::ShardInfo {
            index: i as u32,
            layer_range: (0, 0), // Byte-range shards, not layer-based
            size_bytes: this_shard_size,
            hash: *hasher.finalize().as_bytes(),
        });
    }

    tracing::info!(
        shards = shards.len(),
        shard_size_mb = shard_size / (1024 * 1024),
        "Computed shard hashes"
    );

    Ok(shards)
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
    /// Configured shard size in bytes (from [model].shard_size_mb config).
    /// Used for byte-range calculations instead of actual file sizes.
    pub shard_size_bytes: u64,
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

    // Use configured shard size rather than file size on disk.
    // File sizes may differ from configured shard_size when downloaded from HF
    // (CDN edge caching, partial downloads, etc.). The configured value is what
    // the manifest and layer-range computations are based on.
    let shard_size: u64 = params.shard_size_bytes;

    tracing::info!(
        model = %model_id,
        shards = shard_files.len(),
        shard_size_mb = shard_size / (1024 * 1024),
        layers = format!("[{layer_start}..{layer_end})"),
        "Loading split model from shard files (no full GGUF)"
    );

    crate::inference::split::SplitModel::load_from_shards(
        model_dir,
        shard_files,
        shard_size,
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

    tracing::info!(
        request_id = %request_id,
        seq = forward.sequence_num,
        activation_bytes = forward.activations.len(),
        "Processing LayerForward locally"
    );

    // Find which model we have shards for. For now, pick the first model with a split model
    // or the first model we have local shards for.
    let model_id = {
        // Check if we already have a cached split model
        if let Some(entry) = shared_state.split_models.iter().next() {
            entry.key().0.clone()
        } else {
            // Find a model we have local shards for
            match shared_state.shard_registry.iter().next() {
                Some(entry) => entry.key().model_id.clone(),
                None => {
                    tracing::warn!(request_id = %request_id, "No local shards to process LayerForward");
                    send_error_result(
                        &network_tx,
                        &sender_peer_bytes,
                        request_id,
                        "No local shards",
                    )
                    .await;
                    return;
                }
            }
        }
    };

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

    // Determine layer range.  If the sender specified a layer_range in the forward
    // message (new protocol), use that directly — it tells us exactly which segment
    // to process.  Otherwise fall back to computing from GGUF metadata / manifest.
    let shard_size = shared_state.config.model.shard_size_bytes();
    let (layer_start, layer_end, total_layers) = if let Some((ls, le)) = forward.layer_range {
        let total = shared_state
            .gguf_meta
            .get(&model_id)
            .map(|m| m.block_count)
            .unwrap_or(manifest.num_layers as usize);
        (ls as usize, le as usize, total)
    } else if let Some(meta) = shared_state.gguf_meta.get(&model_id) {
        let (ls, le) = split::compute_local_layer_range(&meta, shard_size, &local_shard_indices);
        (ls, le, meta.block_count)
    } else if manifest.num_layers > 0 {
        // Fallback: use manifest layer_range (approximate)
        let mut ls = manifest.num_layers as usize;
        let mut le = 0usize;
        for shard_info in &manifest.shards {
            if local_shard_indices.contains(&shard_info.index) {
                ls = ls.min(shard_info.layer_range.0 as usize);
                le = le.max(shard_info.layer_range.1 as usize);
            }
        }
        (ls, le, manifest.num_layers as usize)
    } else {
        send_error_result(
            &network_tx,
            &sender_peer_bytes,
            request_id,
            "Cannot determine layer range",
        )
        .await;
        return;
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
                            shard_size_bytes: shared_state.config.model.shard_size_bytes(),
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
                shard_size_bytes: shared_state.config.model.shard_size_bytes(),
            })
        };

        match load_result {
            Ok(model) => {
                shared_state.split_models.insert(
                    split_key.clone(),
                    std::sync::Arc::new(tokio::sync::Mutex::new(model)),
                );
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

    let split_model_ref = match shared_state.split_models.get(&split_key) {
        Some(r) => r.clone(),
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

    // Clear KV-cache at the start of a new request (prefill)
    if forward.sequence_num == 0 {
        split_model.clear_kv_cache();
    }

    // Run the forward pass
    let output = match split_model.forward(&input_tensor, forward.index_pos as usize) {
        Ok(o) => o,
        Err(e) => {
            send_error_result(
                &network_tx,
                &sender_peer_bytes,
                request_id,
                &format!("Forward: {e}"),
            )
            .await;
            return;
        }
    };

    let result = if is_last {
        // Sample a token from logits
        let token_id = match split::sample_token(&output, 0.7, 0.9) {
            Ok(t) => t,
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
        };
        // EOS detection: use tokens loaded from GGUF metadata
        let eos_tokens = split_model.eos_tokens();
        let finish = if eos_tokens.contains(&token_id) {
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
    } else {
        // Intermediate segment: serialize hidden states
        let activation_bytes = match split::tensor_to_bytes(&output) {
            Ok(b) => b,
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
        };
        crate::types::LayerResult {
            request_id,
            token_ids: vec![],
            finish_reason: None,
            activations: activation_bytes,
        }
    };

    tracing::info!(
        request_id = %request_id,
        tokens = result.token_ids.len(),
        activations_bytes = result.activations.len(),
        is_last,
        "LayerForward processed, sending result back"
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

    // Send back via tensor protocol
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
        Ok(mut f) => {
            match candle_core::quantized::gguf_file::Content::read(&mut f) {
                Ok(ct) => ct
                    .metadata
                    .get("general.architecture")
                    .and_then(|v| v.to_string().ok().cloned())
                    .unwrap_or_else(|| "llama".to_string()),
                Err(_) => "llama".to_string(),
            }
        }
        Err(_) => "llama".to_string(),
    };
    match arch_str.as_str() {
        "qwen2" => crate::types::ModelArchitecture::Qwen2,
        "mistral" => crate::types::ModelArchitecture::Mistral,
        "phi" | "phi3" => crate::types::ModelArchitecture::Phi,
        _ => crate::types::ModelArchitecture::Llama,
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
