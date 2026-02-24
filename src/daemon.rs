use std::sync::Arc;

use dashmap::DashMap;
use tokio::sync::{mpsc, watch, RwLock};

use crate::config::Config;
use crate::health::monitor::HealthMonitor;
use crate::identity::Identity;
use crate::inference::executor::SharedExecutor;
use crate::model::registry::ModelRegistry;
use crate::model::shard::ShardStore;
use crate::network::manager::NetworkManager;
use crate::storage::db::Database;
use crate::types::{CreditBalance, NodeId, NodeStats, PeerInfo, PipelineAssignment, ShardId};

/// Thread-safe shared state accessible by all daemon tasks.
pub struct SharedState {
    pub config: Config,
    pub identity: Identity,
    pub db: Database,
    pub peer_registry: DashMap<NodeId, PeerInfo>,
    pub model_registry: ModelRegistry,
    pub shard_registry: DashMap<ShardId, Vec<NodeId>>,
    pub active_pipelines: DashMap<uuid::Uuid, PipelineAssignment>,
    pub credit_balance: RwLock<CreditBalance>,
    pub node_stats: RwLock<NodeStats>,
    pub executor: SharedExecutor,
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
            shard_registry: DashMap::new(),
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

        // Scan local shards
        let shard_store = ShardStore::new(&self.config.node.data_dir);
        match shard_store.load_all_local() {
            Ok(shards) => {
                for (model_id, shard_info) in &shards {
                    let shard_id = ShardId {
                        model_id: model_id.clone(),
                        index: shard_info.index,
                    };
                    shared_state
                        .shard_registry
                        .entry(shard_id)
                        .or_default()
                        .push(shared_state.identity.node_id().clone());
                }
            }
            Err(e) => {
                tracing::warn!(error = %e, "Failed to scan local shards");
            }
        }

        // Create channels for inter-task communication
        let (network_tx, network_rx) = mpsc::channel::<crate::types::SwarmMessage>(1024);
        let (network_out_tx, _network_out_rx) = mpsc::channel::<crate::types::SwarmMessage>(1024);

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

        // Spawn HealthMonitor
        let health_monitor = HealthMonitor::new(
            shared_state.clone(),
            network_tx.clone(),
            shutdown_rx.clone(),
        );

        let health_handle = tokio::spawn(async move {
            if let Err(e) = health_monitor.run().await {
                tracing::error!(error = %e, "HealthMonitor exited with error");
            }
        });

        // Spawn API server
        let api_shared_state = shared_state.clone();
        let api_handle = tokio::spawn(async move {
            if let Err(e) = crate::api::server::run_server_with_state(api_shared_state).await {
                tracing::error!(error = %e, "API server exited with error");
            }
        });

        tracing::info!(
            node_id = %self.identity.node_id(),
            port = self.config.node.listen_port,
            "SwarmLLM daemon running"
        );

        // Wait for shutdown signal or task exit
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {
                tracing::info!("Shutdown signal received (Ctrl+C)");
            }
            result = network_handle => {
                tracing::error!(?result, "NetworkManager task exited");
            }
            result = health_handle => {
                tracing::error!(?result, "HealthMonitor task exited");
            }
            result = api_handle => {
                tracing::error!(?result, "API server task exited");
            }
        }

        // Signal graceful shutdown
        shared_state.shutdown();
        tracing::info!("Daemon shutdown complete");

        Ok(())
    }
}
