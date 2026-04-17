//! `swarmllm run` — start the long-lived daemon.

use std::path::PathBuf;

use swarmllm::config::Config;
use swarmllm::daemon::Daemon;
use swarmllm::identity::Identity;
use swarmllm::storage::db::Database;

/// Flat args bundle for `run_daemon` so `main.rs` doesn't have to expose its
/// clap-derived `Cli` struct to submodules.
pub struct DaemonArgs {
    pub config: Option<PathBuf>,
    pub port: Option<u16>,
    pub data_dir: Option<PathBuf>,
    pub model: Option<PathBuf>,
    pub gpu_layers: Option<u32>,
    pub bootstrap: Vec<String>,
    pub shards: Option<String>,
    pub no_update_check: bool,
}

pub async fn run_daemon(args: DaemonArgs) -> anyhow::Result<()> {
    tracing::debug!(version = env!("CARGO_PKG_VERSION"), "DIAG: daemon starting");

    // Load config
    let mut config = Config::load_or_create(
        args.config.as_deref(),
        args.port,
        args.data_dir.as_deref(),
        args.model.as_deref(),
        args.gpu_layers,
        args.bootstrap,
    )?;

    // Parse --shards range (e.g. "0-4" → (0, 4)) — hidden dev flag
    if let Some(ref shard_str) = args.shards {
        if let Some((start, end)) = shard_str.split_once('-') {
            if let (Ok(s), Ok(e)) = (start.parse::<u32>(), end.parse::<u32>()) {
                if s > e {
                    anyhow::bail!("Invalid --shards range: start ({s}) must be <= end ({e})");
                }
                config.inference.shard_range = Some((s, e));
                tracing::info!(shard_start = s, shard_end = e, "Node claiming shard range");
            } else {
                anyhow::bail!("Invalid --shards format: expected 'START-END' (e.g. '0-4')");
            }
        } else {
            anyhow::bail!("Invalid --shards format: expected 'START-END' (e.g. '0-4')");
        }
    }

    // CLI --no-update-check overrides config
    if args.no_update_check {
        config.updates.auto_update = swarmllm::config::AutoUpdateMode::Disabled;
    }

    // Ensure data directory exists
    std::fs::create_dir_all(&config.node.data_dir)?;

    // Load or generate node identity
    let identity = Identity::load_or_generate(&config.node.data_dir)?;

    // Open database
    let db = Database::open(&config.node.data_dir)?;

    // Persist or restore shard range: CLI flag takes priority, else load from DB
    if let Some((s, e)) = config.inference.shard_range {
        if let Err(err) = db.save_shard_range(s, e) {
            tracing::warn!(error = %err, "Failed to persist shard range to database");
        }
    } else {
        match db.load_shard_range() {
            Ok(Some((s, e))) => {
                config.inference.shard_range = Some((s, e));
                tracing::info!(
                    shard_start = s,
                    shard_end = e,
                    "Restored shard range from previous session"
                );
            }
            Ok(None) => {}
            Err(err) => {
                tracing::warn!(error = %err, "Failed to load shard range from database");
            }
        }
    }

    // Build and run daemon (spawns network, health, API tasks)
    let daemon = Daemon::new(config, identity, db);
    daemon.run().await
}
