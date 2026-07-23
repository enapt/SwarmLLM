//! `swarmllm update` — check for and optionally apply GitHub releases.

use std::sync::Arc;

use swarmllm::update::{UpdateChecker, UpdateState, SWARMLLM_GITHUB_REPO};
use tokio::sync::RwLock;

pub async fn run_update_command(check_only: bool) -> anyhow::Result<()> {
    println!(
        "SwarmLLM {} — checking for updates...",
        env!("CARGO_PKG_VERSION")
    );

    // A manual `swarmllm update` check must report the newest available build,
    // including pre-releases. This repo ships alpha/beta tags AS GitHub
    // pre-releases, so the default `Disabled`/`Stable` filter hides every
    // release published so far — making the check always answer "you're on the
    // latest" even when several versions behind (field report, 2026-07-23). The
    // auto-update MODE only governs auto-APPLYING in the background; an explicit
    // check the user typed should show whatever is actually out there.
    let mut config = swarmllm::config::UpdateConfig::default();
    config.auto_update = swarmllm::config::AutoUpdateMode::All;
    let state = Arc::new(RwLock::new(UpdateState::default()));
    let (dashboard_tx, _) =
        tokio::sync::broadcast::channel::<swarmllm::daemon::state::DashboardSignal>(16);
    let checker = UpdateChecker::new(
        config,
        SWARMLLM_GITHUB_REPO.to_string(),
        state,
        dashboard_tx,
    );

    match checker.check_for_update().await {
        Ok(Some(info)) => {
            println!(
                "Update available: v{} -> v{}",
                info.current_version, info.latest_version
            );
            println!("Published: {}", info.published_at);
            if !info.changelog.is_empty() {
                println!("\nChangelog:\n{}", info.changelog);
            }

            if check_only {
                return Ok(());
            }

            println!("\nDownloading...");
            match checker.download_update(&info).await {
                Ok(tmp_path) => {
                    println!("Downloaded to: {}", tmp_path.display());
                    println!("Applying update...");
                    match checker.apply_update(
                        &tmp_path,
                        &info.latest_version,
                        info.checksum_sha256.as_deref(),
                    ) {
                        Ok(()) => {
                            println!(
                                "Update applied successfully! Restart SwarmLLM to use v{}.",
                                info.latest_version
                            );
                        }
                        Err(e) => {
                            eprintln!("Failed to apply update: {e}");
                            std::process::exit(1);
                        }
                    }
                }
                Err(e) => {
                    eprintln!("Failed to download update: {e}");
                    std::process::exit(1);
                }
            }
        }
        Ok(None) => {
            println!(
                "You are running the latest version (v{}).",
                env!("CARGO_PKG_VERSION")
            );
        }
        Err(e) => {
            eprintln!("Failed to check for updates: {e}");
            std::process::exit(1);
        }
    }

    Ok(())
}
