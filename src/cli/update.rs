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
    // including pre-releases — this repo ships alpha tags AS GitHub
    // pre-releases, so filtering them hides every release published so far
    // (field report, 2026-07-23). `include_prereleases` now defaults true for
    // exactly that reason, so the default config is already right here.
    let config = swarmllm::config::UpdateConfig::default();
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

            // Ask whether this install CAN replace its own binary before
            // fetching ~1 GB to find out. It could not on a Mac with the
            // binary in `/Applications`, and the tester learnt that from a
            // bare "Permission denied" after the whole download (reported
            // 2026-09-03). The check is a file create-and-delete beside the
            // binary, and the message names the folder and the fix.
            if let Some(blocker) = checker.self_update_blocker().await {
                eprintln!();
                eprintln!("Cannot install this update.");
                eprintln!();
                eprintln!("{}", blocker.advice());
                eprintln!();
                eprintln!(
                    "Nothing was downloaded and nothing was changed. This node stays on v{}.",
                    info.current_version
                );
                std::process::exit(1);
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
                            println!("\nUpdate applied — v{} is on disk.", info.latest_version);
                            report_restart_needed(&info.latest_version).await;
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

/// Say — unmissably — that a daemon already running is STILL on the old build.
///
/// Replacing the file on disk does not change a running process: it keeps
/// executing the old image and keeps reporting the old version. A tester hit
/// exactly this on 2026-07-28 — ran `swarmllm update`, saw it succeed, and the
/// node went on answering as the previous version until they killed it by hand.
/// The old wording ("Restart SwarmLLM to use vX") was true but read as a
/// footnote to a success message, so it did not land.
///
/// Only shouts when a daemon is actually up; updating with nothing running
/// needs no ceremony.
async fn report_restart_needed(version: &str) {
    if !daemon_is_running().await {
        println!("Start SwarmLLM when you're ready — it will come up on v{version}.");
        return;
    }

    let how = if std::path::Path::new("/run/systemd/system").exists()
        && std::path::Path::new("/usr/lib/systemd/system/swarmllm.service").exists()
    {
        "  sudo systemctl restart swarmllm"
    } else {
        "  stop the running node (Ctrl-C or `swarmllm stop`), then start it again"
    };

    println!();
    println!("  ┌──────────────────────────────────────────────────────────────┐");
    println!("  │  RESTART REQUIRED                                            │");
    println!("  └──────────────────────────────────────────────────────────────┘");
    println!("  SwarmLLM is still RUNNING the previous version. Replacing the");
    println!("  file does not change a process that already started, so this");
    println!("  node keeps serving the old build until you restart it:");
    println!();
    println!("{how}");
    println!();
}

/// Is a node answering on the local API?
///
/// `/health` needs no credentials, so this works without reading the API key.
async fn daemon_is_running() -> bool {
    let port = std::env::var("SWARMLLM_PORT")
        .ok()
        .and_then(|p| p.parse::<u16>().ok())
        .unwrap_or(8800);
    let client = match reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(2))
        .build()
    {
        Ok(c) => c,
        Err(_) => return false,
    };
    client
        .get(format!("http://127.0.0.1:{port}/health"))
        .send()
        .await
        .map(|r| r.status().is_success())
        .unwrap_or(false)
}
