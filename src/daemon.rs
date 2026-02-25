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
    CreditBalance, GovernanceParams, NetworkCommand, NetworkStats, NodeId, NodeStats, PeerInfo,
    PipelineAssignment, RebalanceEvent, ShardId, SwarmMessage,
};

/// Thread-safe shared state accessible by all daemon tasks.
/// Cached info about a locally loaded model (lock-free reads).
#[derive(Clone, Debug)]
pub struct LoadedModelInfo {
    pub name: String,
    pub size_bytes: u64,
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
    /// Governance parameters (tunable via GovernanceChange proposals).
    pub governance_params: RwLock<GovernanceParams>,
    /// Network-wide statistics for governance role calculation.
    pub network_stats: RwLock<NetworkStats>,
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
    /// Keyed by model_id. Each node loads only its assigned layers.
    pub split_models: DashMap<
        crate::types::ModelId,
        Arc<tokio::sync::Mutex<crate::inference::split::SplitModel>>,
    >,
    /// GGUF tensor metadata for known models (extracted from GGUF header, stored in manifest).
    pub gguf_meta: DashMap<crate::types::ModelId, crate::inference::split::GgufTensorMeta>,
    shutdown_tx: watch::Sender<bool>,
}

impl SharedState {
    pub fn new(
        config: Config,
        identity: Identity,
        db: Database,
        executor: SharedExecutor,
        gpu_info: Option<crate::inference::executor::GpuInfo>,
    ) -> (Arc<Self>, watch::Receiver<bool>) {
        let (shutdown_tx, shutdown_rx) = watch::channel(false);

        let model_registry = ModelRegistry::load_from_db(&db).unwrap_or_default();

        let node_id = identity.node_id().clone();
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
            governance_params: RwLock::new(
                // Load from DB or use defaults (genesis params if early network)
                GovernanceParams::default(),
            ),
            network_stats: RwLock::new(NetworkStats::default()),
            acquisition_progress: DashMap::new(),
            pending_layer_results: DashMap::new(),
            split_models: DashMap::new(),
            gguf_meta: DashMap::new(),
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
        let model_info = if executor.is_loaded() {
            Some(LoadedModelInfo {
                name: executor.model_name().to_string(),
                size_bytes: executor.model_size_bytes().unwrap_or(0),
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

        // Generate a ModelManifest for the locally loaded model so peers can discover it.
        // This is needed even in split mode so the shard registry gets populated.
        if let Some(ref info) = model_info {
            if let Some(ref model_path) = self.config.inference.model_path {
                generate_and_register_local_manifest(&shared_state, info, model_path);
            }
        }

        // Scan local shards and register them + their manifests
        let shard_store = ShardStore::new(&self.config.node.data_dir);
        match shard_store.load_all_local() {
            Ok(shards) => {
                // Track which model manifests we've already registered
                let mut registered_manifests = std::collections::HashSet::new();

                for (model_id, shard_info) in &shards {
                    // Register the manifest if we haven't yet
                    if registered_manifests.insert(model_id.clone()) {
                        let model_dir = shard_store.models_dir().join(&model_id.0);
                        if let Ok(manifest) = crate::types::ModelManifest::load_from_dir(&model_dir)
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
                    shared_state
                        .shard_registry
                        .entry(shard_id)
                        .or_default()
                        .push(node_id);
                }
            }
            Err(e) => {
                tracing::warn!(error = %e, "Failed to scan local shards");
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
            shutdown_rx.clone(),
        );

        let rebalancer_handle = tokio::spawn(async move {
            if let Err(e) = shard_rebalancer.run().await {
                tracing::error!(error = %e, "ShardRebalancer exited with error");
            }
        });

        // Spawn CreditLedger — shares the same Arc<RwLock<CreditBalance>> as SharedState
        let credit_ledger = CreditLedger::new(
            shared_state.identity.node_id().clone(),
            shared_state.credit_balance.clone(),
            self.db.clone(),
            network_tx.clone(),
            shutdown_rx.clone(),
            dispatcher_credit_balances.clone(),
        );

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

        // Spawn API server (pass router_cmd_tx + acquisition_tx so API can submit requests)
        let api_shared_state = shared_state.clone();
        let api_router_tx = router_cmd_tx.clone();
        let api_acquisition_tx = acquisition_tx.clone();
        let api_handle = tokio::spawn(async move {
            if let Err(e) = crate::api::server::run_server_with_state(
                api_shared_state,
                api_router_tx,
                api_acquisition_tx,
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

        // Wait for shutdown signal or task exit
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {
                tracing::info!("Shutdown signal received (Ctrl+C)");
            }
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
            result = api_handle => {
                tracing::error!(?result, "API server task exited");
            }
            result = dispatcher_handle => {
                tracing::error!(?result, "Message dispatcher task exited");
            }
        }

        // Signal graceful shutdown
        shared_state.shutdown();
        tracing::info!("Daemon shutdown complete");

        Ok(())
    }
}

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
                                tokio::spawn(async move {
                                    handle_layer_forward(ss, ntx, forward).await;
                                });
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
                            // Governance Phase 7 messages — store in DB
                            SwarmMessage::Proposal(proposal) => {
                                tracing::info!(
                                    hash = hex::encode(&proposal.hash[..8]),
                                    title = %proposal.title,
                                    "Received governance proposal"
                                );
                                let key = hex::encode(proposal.hash);
                                if let Err(e) = shared_state.db.put_json("proposals", &key, &proposal) {
                                    tracing::warn!(error = %e, "Failed to store proposal");
                                }
                            }
                            SwarmMessage::ProposalVote(vote) => {
                                tracing::debug!(
                                    voter = %vote.voter,
                                    proposal = hex::encode(&vote.proposal_hash[..8]),
                                    "Received proposal vote"
                                );
                                let key = format!(
                                    "{}/{}",
                                    hex::encode(vote.proposal_hash),
                                    hex::encode(&vote.voter.0[..8])
                                );
                                if let Err(e) = shared_state.db.put_json("proposal_votes", &key, &vote) {
                                    tracing::warn!(error = %e, "Failed to store proposal vote");
                                }
                            }
                            SwarmMessage::Issue(issue) => {
                                tracing::info!(
                                    hash = hex::encode(&issue.hash[..8]),
                                    title = %issue.title,
                                    "Received governance issue"
                                );
                                let key = hex::encode(issue.hash);
                                if let Err(e) = shared_state.db.put_json("issues", &key, &issue) {
                                    tracing::warn!(error = %e, "Failed to store issue");
                                }
                            }
                            SwarmMessage::IssueComment(comment) => {
                                let key = format!(
                                    "{}/{}",
                                    hex::encode(comment.issue_hash),
                                    comment.created_at.timestamp_millis()
                                );
                                if let Err(e) = shared_state.db.put_json("issue_comments", &key, &comment) {
                                    tracing::warn!(error = %e, "Failed to store issue comment");
                                }
                            }
                            SwarmMessage::IssueUpvote(upvote) => {
                                let key = format!(
                                    "{}/{}",
                                    hex::encode(upvote.issue_hash),
                                    hex::encode(&upvote.voter.0[..8])
                                );
                                if let Err(e) = shared_state.db.put_json("issue_upvotes", &key, &upvote) {
                                    tracing::warn!(error = %e, "Failed to store issue upvote");
                                }
                            }
                            SwarmMessage::ReleaseCandidate(rc) => {
                                tracing::info!(
                                    version = %rc.version,
                                    builder = %rc.builder,
                                    "Received release candidate"
                                );
                                let key = rc.version.to_key();
                                if let Err(e) = shared_state.db.put_json("releases", &key, &rc) {
                                    tracing::warn!(error = %e, "Failed to store release candidate");
                                }
                            }
                            SwarmMessage::ReleaseApproval(approval) => {
                                let key = format!(
                                    "{}/{}",
                                    approval.release_version.to_key(),
                                    hex::encode(&approval.approver.0[..8])
                                );
                                if let Err(e) = shared_state.db.put_json("release_approvals", &key, &approval) {
                                    tracing::warn!(error = %e, "Failed to store release approval");
                                }
                            }
                            SwarmMessage::TestReport(report) => {
                                let key = format!(
                                    "{}/{}",
                                    report.release_version.to_key(),
                                    hex::encode(&report.tester.0[..8])
                                );
                                if let Err(e) = shared_state.db.put_json("test_reports", &key, &report) {
                                    tracing::warn!(error = %e, "Failed to store test report");
                                }
                            }
                            SwarmMessage::ChangelogEntry(ref entry) => {
                                if let Err(e) = crate::governance::changelog::store_changelog(&shared_state.db, entry) {
                                    tracing::warn!(error = %e, "Failed to store changelog");
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
                                // Record the transaction and apply balance change
                                // if we are the recipient
                                let local_id = shared_state.identity.node_id().clone();
                                if tx.to == local_id {
                                    let mut bal = shared_state.credit_balance.write().await;
                                    bal.balance += tx.amount;
                                    bal.lifetime_earned += tx.amount as u64;
                                    bal.last_updated = chrono::Utc::now();
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
                                for shard_id in &announce.shards {
                                    shared_state.shard_registry
                                        .entry(shard_id.clone())
                                        .or_default()
                                        .push(announce.node_id.clone());
                                    // Also register in model_registry so auto-acquire
                                    // can see shard coverage across the network
                                    shared_state.model_registry
                                        .record_shard_holder(shard_id.clone(), announce.node_id.clone());
                                }
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
                                        shared_state.model_registry.register_manifest(manifest.clone());
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
    let model_id = crate::types::ModelId(info.name.clone());

    // Check if we already have a manifest for this model
    if shared_state
        .model_registry
        .get_manifest(&model_id)
        .is_some()
    {
        tracing::debug!(model = %model_id, "Manifest already registered, skipping generation");
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

    // Split model into multiple shards (~512MB each) for torrent-style distribution.
    // Each shard is a byte range of the original GGUF file.
    const SHARD_SIZE: u64 = 512 * 1024 * 1024; // 512MB per shard
    let node_id = shared_state.identity.node_id().clone();
    let shard_count = file_size.div_ceil(SHARD_SIZE).max(1) as u32;

    // Compute per-shard BLAKE3 hashes by reading byte ranges
    let mut shards = match compute_shard_hashes(path, file_size, SHARD_SIZE) {
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
                // Assign layer ranges proportionally across shards.
                // Simple linear distribution: divide num_layers evenly.
                // This avoids gaps caused by tensor byte offsets crossing shard boundaries.
                let n = shards.len() as u32;
                let layers_per_shard = num_layers / n;
                let remainder = num_layers % n;
                let mut layer_cursor = 0u32;
                for shard in &mut shards {
                    let extra = if shard.index < remainder { 1 } else { 0 };
                    let shard_layers = layers_per_shard + extra;
                    shard.layer_range = (layer_cursor, layer_cursor + shard_layers);
                    layer_cursor += shard_layers;
                }
                // Store the metadata for later use in layer range computation
                shared_state.gguf_meta.insert(model_id.clone(), meta);
                (num_layers, crate::types::ModelArchitecture::Llama)
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

    // Save manifest to disk
    if let Err(e) = manifest.save_to_dir(&model_dir) {
        tracing::warn!(error = %e, "Failed to save generated manifest");
        return;
    }

    // Register in model_registry
    shared_state
        .model_registry
        .register_manifest(manifest.clone());

    // Register ourselves as holder of our shards.
    // If --shards range is set, only claim those indices; otherwise claim all.
    let shard_range = shared_state.config.inference.shard_range;
    for shard_info in &manifest.shards {
        let in_range = match shard_range {
            Some((start, end)) => shard_info.index >= start && shard_info.index <= end,
            None => true,
        };
        if !in_range {
            continue;
        }
        let shard_id = crate::types::ShardId {
            model_id: model_id.clone(),
            index: shard_info.index,
        };
        shared_state
            .model_registry
            .record_shard_holder(shard_id.clone(), node_id.clone());
        shared_state
            .shard_registry
            .entry(shard_id)
            .or_default()
            .push(node_id.clone());
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
            entry.key().clone()
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

    // Use GgufTensorMeta to compute our layer range, or fall back to manifest layer_range
    let (layer_start, layer_end, total_layers) = if let Some(meta) =
        shared_state.gguf_meta.get(&model_id)
    {
        let shard_size = if manifest.shard_count > 0 {
            manifest.total_size_bytes / manifest.shard_count as u64
        } else {
            manifest.total_size_bytes
        };
        let (ls, le) = split::compute_local_layer_range(&meta, shard_size, &local_shard_indices);
        (ls, le, meta.block_count)
    } else if manifest.num_layers > 0 {
        // Use shard layer_range from manifest
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

    let is_first = layer_start == 0;
    let is_last = layer_end >= total_layers;

    // Ensure the split model is loaded
    if !shared_state.split_models.contains_key(&model_id) {
        let shard_store = crate::model::shard::ShardStore::new(&shared_state.config.node.data_dir);
        let model_dir = shard_store.models_dir().join(&model_id.0);
        let gguf_path = model_dir.join("model.gguf");
        let gguf_path = if gguf_path.exists() {
            gguf_path
        } else {
            let source_path_file = model_dir.join("source_path");
            if source_path_file.exists() {
                match std::fs::read_to_string(&source_path_file) {
                    Ok(p) => std::path::PathBuf::from(p.trim()),
                    Err(e) => {
                        send_error_result(
                            &network_tx,
                            &sender_peer_bytes,
                            request_id,
                            &format!("IO: {e}"),
                        )
                        .await;
                        return;
                    }
                }
            } else {
                send_error_result(
                    &network_tx,
                    &sender_peer_bytes,
                    request_id,
                    "No GGUF file found",
                )
                .await;
                return;
            }
        };

        tracing::info!(
            model = %model_id,
            layers = format!("[{layer_start}..{layer_end})"),
            total = total_layers,
            path = %gguf_path.display(),
            "Loading split model for LayerForward handling"
        );

        match SplitModel::load_from_gguf(&gguf_path, layer_start, layer_end, is_first, is_last) {
            Ok(model) => {
                shared_state.split_models.insert(
                    model_id.clone(),
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

    let split_model_ref = match shared_state.split_models.get(&model_id) {
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
        // First segment: activations are the prompt text → tokenize
        let prompt = String::from_utf8_lossy(&forward.activations);
        let token_ids: Vec<i64> = prompt.bytes().map(|b| b as i64).collect();
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

    // Run the forward pass
    let output = match split_model.forward(&input_tensor, forward.sequence_num as usize) {
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
        let finish = if token_id == 2 || token_id == 0 {
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
