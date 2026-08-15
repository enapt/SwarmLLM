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
    pub gpu_layers: Option<i32>,
    pub bootstrap: Vec<String>,
    pub shards: Option<String>,
    pub no_update_check: bool,
    pub anchor: bool,
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

    // Parse --shards range (e.g. "0-4" → (0, 4)) — hidden dev flag. Setting a
    // range persists it, and `--shards all` is how you undo that; without that
    // escape a node told once to hold half a model held half of it forever.
    //
    // `file_shard_range` is captured BEFORE the flag can overwrite it, because
    // the two sources must stay distinguishable — only the flag is sticky. Once
    // merged into one `Option`, `resolve_shard_range` could not tell them apart
    // and treated a config-file value as though the flag had set it: writing it
    // to the database, then restoring it from there after the line was deleted.
    // Editing a config file is expected to be undone by un-editing it; that made
    // it a one-way door whose only escape was a flag nothing pointed you at
    // (measured on a live node 2026-08-15, gotcha #306).
    let file_shard_range = config.inference.shard_range;
    let mut cli_shard_range: Option<(u32, u32)> = None;
    let mut clear_shard_range = false;
    if let Some(ref shard_str) = args.shards {
        if matches!(shard_str.trim(), "all" | "none") {
            clear_shard_range = true;
            tracing::info!("Clearing any saved shard range — this node claims every shard");
        } else if let Some((start, end)) = shard_str.split_once('-') {
            if let (Ok(s), Ok(e)) = (start.parse::<u32>(), end.parse::<u32>()) {
                if s > e {
                    anyhow::bail!("Invalid --shards range: start ({s}) must be <= end ({e})");
                }
                cli_shard_range = Some((s, e));
                config.inference.shard_range = Some((s, e));
                tracing::info!(shard_start = s, shard_end = e, "Node claiming shard range");
            } else {
                anyhow::bail!("Invalid --shards format: expected 'START-END' (e.g. '0-4')");
            }
        } else {
            anyhow::bail!(
                "Invalid --shards format: expected 'START-END' (e.g. '0-4'), or 'all' to \
                 clear a previously saved range"
            );
        }
    }

    // CLI --no-update-check overrides config
    if args.no_update_check {
        config.updates.auto_update = swarmllm::config::AutoUpdateMode::Disabled;
    }

    // --anchor makes the flag self-sufficient: a bootstrap/relay node that
    // never runs inference. `apply_anchor_mode` forces every inference/model
    // knob off (no model load, HF poll, shard acquisition, auto-manage, or
    // browser pop). The API bind is narrowed to loopback in the server.
    if args.anchor || config.node.anchor_mode {
        config.apply_anchor_mode();
    }

    // Ensure data directory exists
    std::fs::create_dir_all(&config.node.data_dir)?;

    // Load or generate node identity
    let identity = Identity::load_or_generate(&config.node.data_dir)?;

    // Open database
    let db = Database::open(&config.node.data_dir)?;

    let decision =
        resolve_shard_range(cli_shard_range, file_shard_range, clear_shard_range, || {
            db.load_shard_range().unwrap_or(None)
        });
    config.inference.shard_range = decision.effective;
    match decision.persist {
        ShardRangePersist::Clear => {
            if let Err(err) = db.clear_shard_range() {
                tracing::warn!(error = %err, "Failed to clear saved shard range");
            }
        }
        ShardRangePersist::Save(s, e) => {
            if let Err(err) = db.save_shard_range(s, e) {
                tracing::warn!(error = %err, "Failed to persist shard range to database");
            }
        }
        ShardRangePersist::Leave => {}
    }
    if let Some((s, e)) = decision.effective {
        tracing::info!(
            shard_start = s,
            shard_end = e,
            source = decision.source,
            "Node claiming shard range"
        );
    }

    // Build and run daemon (spawns network, health, API tasks)
    let daemon = Daemon::new(config, identity, db);
    daemon.run().await
}

/// What to do with the shard range saved in the database.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum ShardRangePersist {
    /// Forget the saved range (`--shards all`).
    Clear,
    /// Remember this range for future starts (`--shards A-B`).
    Save(u32, u32),
    /// Touch nothing — the saved range is neither used nor overwritten.
    Leave,
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct ShardRangeDecision {
    pub effective: Option<(u32, u32)>,
    pub persist: ShardRangePersist,
    pub source: &'static str,
}

/// Resolve which shard range this node claims, and whether to remember it.
///
/// Precedence: `--shards` flag > config file > the range a previous flag saved.
///
/// **Only the flag is sticky.** That is what it is for — a dev splitting a model
/// across machines runs it once and expects it to survive restarts, with
/// `--shards all` to undo it. A CONFIG-FILE value must not behave that way: the
/// two arrived as one `Option`, so a file value was written to the database and
/// then restored from there once the line was deleted, which made editing a
/// config file a one-way door whose only escape was a hidden flag nothing
/// pointed you at. Measured on a live node: `shard_range = [0, 7]` removed,
/// config clean, two restarts, file touched, explicit rescan — the shard stayed
/// unclaimed and 709 MB sat on disk advertised to nobody (gotcha #306).
pub(crate) fn resolve_shard_range(
    cli: Option<(u32, u32)>,
    file: Option<(u32, u32)>,
    clear: bool,
    saved: impl FnOnce() -> Option<(u32, u32)>,
) -> ShardRangeDecision {
    if clear {
        return ShardRangeDecision {
            effective: None,
            persist: ShardRangePersist::Clear,
            source: "cleared",
        };
    }
    if let Some((s, e)) = cli {
        return ShardRangeDecision {
            effective: Some((s, e)),
            persist: ShardRangePersist::Save(s, e),
            source: "flag",
        };
    }
    if let Some((s, e)) = file {
        // Not saved, and the saved value is not consulted: the file is the one
        // place this is expressed, so deleting the line undoes it.
        return ShardRangeDecision {
            effective: Some((s, e)),
            persist: ShardRangePersist::Leave,
            source: "config file",
        };
    }
    match saved() {
        Some((s, e)) => ShardRangeDecision {
            effective: Some((s, e)),
            persist: ShardRangePersist::Leave,
            source: "saved by a previous --shards flag",
        },
        None => ShardRangeDecision {
            effective: None,
            persist: ShardRangePersist::Leave,
            source: "unset",
        },
    }
}

#[cfg(test)]
mod shard_range_precedence_tests {
    use super::*;

    /// The bug: a config-file value was persisted, so deleting the line left the
    /// node still restricted with no way back short of a hidden flag.
    #[test]
    fn a_config_file_range_is_never_written_to_the_database() {
        let d = resolve_shard_range(None, Some((0, 2)), false, || None);
        assert_eq!(d.effective, Some((0, 2)));
        assert_eq!(d.persist, ShardRangePersist::Leave, "must not persist");
    }

    /// And removing the line must take effect even when an old flag saved one.
    #[test]
    fn removing_the_config_line_undoes_it() {
        // Config file no longer sets it, nothing saved -> unrestricted.
        let d = resolve_shard_range(None, None, false, || None);
        assert_eq!(d.effective, None);
        // A range saved by an earlier FLAG still applies — that one is sticky
        // by design, and `--shards all` is its documented undo.
        let d = resolve_shard_range(None, None, false, || Some((0, 1)));
        assert_eq!(d.effective, Some((0, 1)));
    }

    /// The config file beats a stale saved range, rather than the reverse.
    #[test]
    fn the_config_file_wins_over_a_previously_saved_range() {
        let d = resolve_shard_range(None, Some((5, 9)), false, || Some((0, 1)));
        assert_eq!(d.effective, Some((5, 9)));
        assert_eq!(d.persist, ShardRangePersist::Leave);
    }

    /// The flag still wins over everything, and is still remembered.
    #[test]
    fn the_flag_wins_and_stays_sticky() {
        let d = resolve_shard_range(Some((2, 3)), Some((0, 1)), false, || Some((7, 8)));
        assert_eq!(d.effective, Some((2, 3)));
        assert_eq!(d.persist, ShardRangePersist::Save(2, 3));
    }

    /// `--shards all` clears the saved range and restores full claiming.
    #[test]
    fn clearing_wins_over_every_source() {
        let d = resolve_shard_range(None, Some((0, 1)), true, || Some((0, 1)));
        assert_eq!(d.effective, None);
        assert_eq!(d.persist, ShardRangePersist::Clear);
    }
}
