use std::sync::Arc;

use tokio::sync::mpsc;
use tokio::task::JoinSet;

use crate::config::Config;
use crate::credit::ledger::CreditLedger;
use crate::health::monitor::HealthMonitor;
use crate::health::rebalancer::ShardRebalancer;
use crate::identity::Identity;
use crate::inference::router::{InferenceRouter, RouterCommand};
use crate::model::acquisition::{AcquisitionCommand, AcquisitionManager};
use crate::network::manager::NetworkManager;
use crate::storage::db::Database;
use crate::types::{AuthenticatedMessage, NetworkCommand, RebalanceEvent};

mod background;
pub(crate) mod dispatch;
mod helpers;
pub mod manifest;
pub mod shard_loader;
mod startup;
pub mod state;
mod supervisor;

// Re-export public types so callers use crate::daemon::SharedState etc.
pub use dispatch::estimate_vram_from_shard_dir;
pub use helpers::SubsystemCriticality;
pub use manifest::generate_and_register_local_manifest;
pub use shard_loader::{try_load_from_shards, ShardLoadParams};
pub use state::*;

use dispatch::dispatch_network_messages;

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
        // .env loading happens in main() before the Tokio runtime spawns
        // worker threads — `std::env::set_var` is unsound in a multi-threaded
        // process. By the time we reach here, env vars are already populated.

        // Log detected provider API keys from environment
        let env_keys = crate::config::ProvidersConfig::detect_env_keys();
        if !env_keys.is_empty() {
            let names: Vec<&str> = env_keys.iter().map(|(_, name)| *name).collect();
            tracing::info!(
                providers = ?names,
                count = env_keys.len(),
                "Detected provider API keys in environment"
            );
        }

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
                Ok(()) => tracing::info!(path = %model_path.display(), "Model ready"),
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

        // Detect GPU via llama.cpp backend; fall back to candle CUDA probe
        let gpu_info = {
            let llama_gpu = crate::inference::executor::detect_gpu();
            #[cfg(feature = "candle-cuda")]
            let gpu_info = llama_gpu.or_else(|| {
                let cuda_ok = candle_core::Device::cuda_if_available(0)
                    .map(|d| d.is_cuda())
                    .unwrap_or(false);
                if cuda_ok {
                    let (name, vram_mb) = crate::api::admin::detect_gpu_nvidia_smi();
                    Some(crate::inference::executor::GpuInfo {
                        name: name.unwrap_or_else(|| "NVIDIA GPU".to_string()),
                        vram_total_mb: vram_mb.unwrap_or(0),
                        vram_free_mb: 0,
                        backend: "CUDA".to_string(),
                    })
                } else {
                    None
                }
            });
            #[cfg(not(feature = "candle-cuda"))]
            let gpu_info = llama_gpu;
            gpu_info
        };
        if let Some(ref gpu) = gpu_info {
            tracing::info!(gpu = %gpu.name, vram_mb = gpu.vram_total_mb, backend = %gpu.backend, "GPU detected");
            // Loud, because this combination used to be a silent lie: the
            // shipped default was `gpu_layers = 0` documented as "CPU only",
            // and a CUDA build ran on the GPU anyway. Now that the setting is
            // honoured, someone carrying that value forward in their
            // config.toml would quietly lose GPU inference — so say it.
            if self.config.inference.gpu_layers == 0 {
                tracing::warn!(
                    gpu = %gpu.name,
                    "inference.gpu_layers = 0 — running CPU-only despite the detected GPU. \
                     Set gpu_layers = -1 (auto) to use it."
                );
            }
        }

        let executor = Arc::new(tokio::sync::Mutex::new(executor));

        // Create shared state
        let (shared_state, shutdown_rx, dht_query_rx) = SharedState::new(
            self.config.clone(),
            self.identity.clone(),
            self.db.clone(),
            executor,
            gpu_info,
        );

        // Anchor mode: a pure bootstrap/relay/AutoNAT node. The inference and
        // model-management subsystems below are skipped so no models load, no
        // HuggingFace polling / shard acquisition runs, and no inference
        // subprocess ever spawns. The node still runs NetworkManager (relay
        // server, AutoNAT, DCUtR, UPnP, DHT, gossip), the dispatcher, health,
        // credit ledger, and pool manager, so it fully participates in P2P
        // discovery without exposing any inference surface. See `--anchor`.
        let anchor = self.config.node.anchor_mode;

        *shared_state.loaded_model_info.write().await = cached_info;

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

        startup::restore_persistent_state(&shared_state, &self.config, &self.db).await;

        // ── Channel Architecture ──
        //
        // network_tx      → NetworkManager (outbound commands: broadcast, send tensor)
        // network_out_tx  → from NetworkManager (inbound decoded messages)
        // router_cmd_tx   → InferenceRouter (commands from API + network)
        // rebalance_tx    → ShardRebalancer (events from HealthMonitor)
        // acquisition_tx  → AcquisitionManager (model download commands from API)
        //
        let (network_tx, network_rx) = mpsc::channel::<NetworkCommand>(1024);
        let (network_out_tx, mut network_out_rx) = mpsc::channel::<AuthenticatedMessage>(1024);
        let (router_cmd_tx, router_cmd_rx) = mpsc::channel::<RouterCommand>(256);
        let (rebalance_tx, rebalance_rx) = mpsc::channel::<RebalanceEvent>(64);
        let (acquisition_tx, acquisition_rx) = mpsc::channel::<AcquisitionCommand>(64);

        // ── Subsystem Supervisor (JoinSet) ──
        //
        // All 12 subsystem tasks are spawned into a JoinSet for unified monitoring.
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
            dht_query_rx,
        )?;

        subsystems.spawn(async move {
            let result = network_manager.run().await.map_err(|e| e.to_string());
            ("NetworkManager", SubsystemCriticality::Critical, result)
        });

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
        let dispatcher_credit_balances: Arc<arc_swap::ArcSwap<Vec<i64>>> =
            Arc::new(arc_swap::ArcSwap::from_pointee(Vec::new()));
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
            shared_state.credits.credit_balance.clone(),
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

        // R112: HfWatcher — background task that polls HuggingFace's
        // trending GGUF feed every hour. Non-critical: a HF outage or a
        // network partition keeps the rest of the daemon running. The
        // watcher exits cleanly when `auto_manage.hf_watcher_enabled` is
        // false; the spawned task is still useful as a one-shot "is HF
        // reachable?" probe in that case (run() returns immediately).
        if !anchor {
            let hf_watcher = crate::model::huggingface::HfWatcher::new(
                shared_state.clone(),
                shutdown_rx.clone(),
            );
            subsystems.spawn(async move {
                let result = hf_watcher.run().await.map_err(|e| e.to_string());
                ("HfWatcher", SubsystemCriticality::NonCritical, result)
            });
        }

        let (pool_cmd_tx, pool_cmd_rx) = mpsc::channel::<crate::pool::types::PoolCommand>(64);
        {
            *shared_state.credits.pool_tx.write().await = Some(pool_cmd_tx);
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

        if !anchor {
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
        }

        // Spawn UpdateChecker (11th subsystem task — optional, runs only if not disabled).
        // When disabled, skip the spawn entirely — otherwise the supervisor logs a
        // misleading "Subsystem exited unexpectedly with Ok" warning at startup.
        if self.config.updates.auto_update != crate::config::AutoUpdateMode::Disabled {
            let update_config = self.config.updates.clone();
            let update_state = shared_state.events.update_state.clone();
            let dash_tx = shared_state.events.dashboard_tx.clone();
            let update_shutdown = shutdown_rx.clone();
            let checker = crate::update::UpdateChecker::new(
                update_config,
                crate::update::SWARMLLM_GITHUB_REPO.to_string(),
                update_state,
                dash_tx,
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

        // Plain stderr banner — fires regardless of tracing log level or
        // log redirection. New users running `swarmllm run` need to see the
        // dashboard URL and API key path unmissably; tracing format is too
        // noisy for "what URL do I open" first-run guidance.
        let port = self.config.node.listen_port;
        let api_key_path = self.config.node.data_dir.join("api_key");
        eprintln!();
        eprintln!("============================================================");
        if anchor {
            eprintln!("  SwarmLLM ANCHOR is running (bootstrap / relay only)");
            eprintln!("  Inference disabled — no models, no downloads.");
            eprintln!("  Dashboard:  http://127.0.0.1:{port}  (loopback only)");
            eprintln!("  P2P:        TCP {}  +  UDP {} (QUIC)", port + 10, port);
        } else {
            eprintln!("  SwarmLLM is running");
            eprintln!("  Dashboard:  http://localhost:{port}");
            eprintln!("  API key:    {}", api_key_path.display());
            eprintln!("  OpenAI API: http://localhost:{port}/v1/chat/completions");
        }
        eprintln!("============================================================");
        eprintln!();

        shared_state.emit_activity(
            crate::daemon::state::ActivityEvent::new(
                "system",
                "daemon_started",
                format!("SwarmLLM started on port {}", self.config.node.listen_port),
            )
            .with_node(format!("{}", self.identity.node_id()))
            .with_detail_num(self.config.node.listen_port as i64),
        );

        // Best-effort background tasks share a `JoinSet` so panics surface
        // (logged in the drain phase below) and pending work has a chance
        // to land before the daemon exits. Pre-2026-04-24 these used bare
        // `tokio::spawn`, which silently swallowed panics.
        let mut background_tasks: background::BackgroundTasks = tokio::task::JoinSet::new();

        // Item 8 Phase 1: install the prefix-cache manifest channel + spawn
        // the forwarder. Worker processes emit `WorkerMsg::PrefixManifestUpdate`
        // each time they snapshot a new prefix into their local cache; the
        // forwarder turns those into gossip + folds them into the cross-node
        // index (also recording our own NodeId for loopback verification).
        let (prefix_manifest_tx, prefix_manifest_rx) =
            mpsc::channel::<crate::inference::process_pool::PrefixManifestEvent>(256);
        shared_state
            .model_process_pool
            .set_prefix_manifest_tx(prefix_manifest_tx);
        background::spawn_prefix_announce_forwarder(
            &mut background_tasks,
            shared_state.clone(),
            network_tx.clone(),
            prefix_manifest_rx,
            shutdown_rx.clone(),
        );

        // Pre-first-token progress (prefill / model load). Workers emit one per
        // prefill chunk; the forwarder stamps it onto the in-flight trace so a
        // long prompt reads as progress rather than as a hang.
        let (progress_tx, progress_rx) =
            mpsc::channel::<crate::inference::process_pool::ProgressEvent>(256);
        shared_state.model_process_pool.set_progress_tx(progress_tx);
        background::spawn_progress_forwarder(
            &mut background_tasks,
            shared_state.clone(),
            progress_rx,
            shutdown_rx.clone(),
        );

        // Item 8 Phase 2b: worker-initiated fetch probes go here.
        let (prefix_probe_tx, prefix_probe_rx) =
            mpsc::channel::<crate::inference::process_pool::PrefixProbeEvent>(256);
        shared_state
            .model_process_pool
            .set_prefix_probe_tx(prefix_probe_tx);
        background::spawn_prefix_probe_handler(
            &mut background_tasks,
            shared_state.clone(),
            network_tx.clone(),
            prefix_probe_rx,
            shutdown_rx.clone(),
        );

        background::spawn_shard_verification(
            &mut background_tasks,
            shared_state.clone(),
            self.config.node.data_dir.clone(),
            shutdown_rx.clone(),
        );
        background::spawn_region_detection(
            &mut background_tasks,
            shared_state.clone(),
            shutdown_rx.clone(),
        );
        background::spawn_initial_announcements(
            &mut background_tasks,
            shared_state.clone(),
            network_tx.clone(),
            shutdown_rx.clone(),
        );
        background::spawn_key_rotation(
            &mut background_tasks,
            shared_state.clone(),
            network_tx.clone(),
            shutdown_rx.clone(),
        );
        background::spawn_browser_open(&mut background_tasks, &self.config, shutdown_rx.clone());
        // Anchor nodes never load models — skip autoload entirely (the main
        // memory win: no candle model weights in RAM).
        if !anchor {
            background::spawn_model_autoload(
                &mut background_tasks,
                shared_state.clone(),
                shutdown_rx.clone(),
            );
        }
        background::spawn_sighup_handler(
            &mut background_tasks,
            shared_state.clone(),
            self.config.clone(),
            shutdown_rx.clone(),
        );
        background::spawn_responses_sweep(
            &mut background_tasks,
            self.db.clone(),
            shutdown_rx.clone(),
        );

        supervisor::run(subsystems, shutdown_rx, shared_state).await;

        // Drain the background JoinSet so panics surface and tasks get a
        // brief window to run their own cleanup paths before exit.
        background::drain(background_tasks).await;

        // redb writes are durable on commit — no flush needed

        // Shut down Claude Code sessions (kill subprocesses, clean temp dirs)
        #[cfg(feature = "claude-subscription")]
        {
            crate::api::claude_session::SessionManager::global()
                .shutdown_all()
                .await;
        }

        tracing::info!("Daemon shutdown complete");

        Ok(())
    }
}
