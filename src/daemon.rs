use std::sync::Arc;

use dashmap::DashMap;
use tokio::sync::{mpsc, watch, RwLock};

use crate::config::Config;
use crate::credit::ledger::CreditLedger;
use crate::health::monitor::HealthMonitor;
use crate::health::rebalancer::ShardRebalancer;
use crate::identity::Identity;
use crate::inference::executor::SharedExecutor;
use crate::inference::router::{InferenceRouter, RouterCommand};
use crate::model::registry::ModelRegistry;
use crate::model::shard::ShardStore;
use crate::network::manager::NetworkManager;
use crate::storage::db::Database;
use crate::types::{
    CreditBalance, GovernanceParams, NetworkCommand, NetworkStats, NodeId, NodeStats, PeerInfo,
    PipelineAssignment, RebalanceEvent, ShardId, SwarmMessage,
};

/// Thread-safe shared state accessible by all daemon tasks.
pub struct SharedState {
    pub config: Config,
    pub identity: Identity,
    pub db: Database,
    pub peer_registry: DashMap<NodeId, PeerInfo>,
    pub model_registry: ModelRegistry,
    pub active_pipelines: DashMap<uuid::Uuid, PipelineAssignment>,
    pub credit_balance: RwLock<CreditBalance>,
    pub node_stats: RwLock<NodeStats>,
    pub executor: SharedExecutor,
    /// Model governance vote tallies.
    pub model_vote_tallies: DashMap<crate::types::Blake3Hash, crate::model::governance::VoteTally>,
    /// Governance parameters (tunable via GovernanceChange proposals).
    pub governance_params: RwLock<GovernanceParams>,
    /// Network-wide statistics for governance role calculation.
    pub network_stats: RwLock<NetworkStats>,
    shutdown_tx: watch::Sender<bool>,
}

impl SharedState {
    pub fn new(
        config: Config,
        identity: Identity,
        db: Database,
        executor: SharedExecutor,
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
            active_pipelines: DashMap::new(),
            credit_balance: RwLock::new(CreditBalance {
                node_id,
                balance: 0,
                lifetime_earned: 0,
                lifetime_spent: 0,
                last_updated: chrono::Utc::now(),
            }),
            node_stats: RwLock::new(NodeStats::default()),
            executor,
            model_vote_tallies: DashMap::new(),
            governance_params: RwLock::new(
                // Load from DB or use defaults (genesis params if early network)
                GovernanceParams::default(),
            ),
            network_stats: RwLock::new(NetworkStats::default()),
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
    executor: SharedExecutor,
}

impl Daemon {
    pub fn new(config: Config, identity: Identity, db: Database, executor: SharedExecutor) -> Self {
        Self {
            config,
            identity,
            db,
            executor,
        }
    }

    /// Run the daemon — spawns all subsystems and waits for shutdown.
    pub async fn run(self) -> anyhow::Result<()> {
        // Create shared state
        let (shared_state, shutdown_rx) = SharedState::new(
            self.config.clone(),
            self.identity.clone(),
            self.db.clone(),
            self.executor.clone(),
        );

        // Scan local shards and register them in the model_registry
        let shard_store = ShardStore::new(&self.config.node.data_dir);
        match shard_store.load_all_local() {
            Ok(shards) => {
                for (model_id, shard_info) in &shards {
                    let shard_id = ShardId {
                        model_id: model_id.clone(),
                        index: shard_info.index,
                    };
                    shared_state
                        .model_registry
                        .record_shard_holder(shard_id, shared_state.identity.node_id().clone());
                }
            }
            Err(e) => {
                tracing::warn!(error = %e, "Failed to scan local shards");
            }
        }

        // ── Channel Architecture ──
        //
        // network_tx  → NetworkManager (outbound commands: broadcast, send tensor)
        // network_out_tx → from NetworkManager (inbound decoded messages)
        // router_cmd_tx  → InferenceRouter (commands from API + network)
        // rebalance_tx   → ShardRebalancer (events from HealthMonitor)
        //
        let (network_tx, network_rx) = mpsc::channel::<NetworkCommand>(1024);
        let (network_out_tx, mut network_out_rx) = mpsc::channel::<SwarmMessage>(1024);
        let (router_cmd_tx, router_cmd_rx) = mpsc::channel::<RouterCommand>(256);
        let (rebalance_tx, rebalance_rx) = mpsc::channel::<RebalanceEvent>(64);

        // Spawn NetworkManager
        let network_manager = NetworkManager::new(
            shared_state.clone(),
            &self.identity,
            &self.config,
            network_rx,
            network_out_tx,
            shutdown_rx.clone(),
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
        let dispatcher_handle = tokio::spawn(async move {
            dispatch_network_messages(
                &mut network_out_rx,
                &dispatcher_router_tx,
                dispatcher_credit_ref,
                &dispatcher_state,
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

        // Spawn CreditLedger
        let credit_balance_arc = Arc::new(RwLock::new(
            shared_state.credit_balance.read().await.clone(),
        ));
        let credit_ledger = CreditLedger::new(
            shared_state.identity.node_id().clone(),
            credit_balance_arc,
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

        // Spawn API server (pass router_cmd_tx so API can submit inference requests)
        let api_shared_state = shared_state.clone();
        let api_router_tx = router_cmd_tx.clone();
        let api_handle = tokio::spawn(async move {
            if let Err(e) =
                crate::api::server::run_server_with_state(api_shared_state, api_router_tx).await
            {
                tracing::error!(error = %e, "API server exited with error");
            }
        });

        tracing::info!(
            node_id = %self.identity.node_id(),
            port = self.config.node.listen_port,
            "SwarmLLM daemon running"
        );

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
                        match &msg {
                            SwarmMessage::InferenceRequest(_)
                            | SwarmMessage::PipelineAssignment(_)
                            | SwarmMessage::LayerForward(_)
                            | SwarmMessage::LayerResult(_)
                            | SwarmMessage::InferenceError(_) => {
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
                            SwarmMessage::ChangelogEntry(entry) => {
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
                                if let Err(e) = shared_state.db.put_json("credit_transactions", &key, &tx) {
                                    tracing::warn!(error = %e, "Failed to store credit transaction");
                                }
                            }
                            // Discovery, health, and status change messages
                            // are handled by NetworkManager or their respective subsystems
                            _ => {}
                        }
                    }
                    None => break,
                }
            }
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
