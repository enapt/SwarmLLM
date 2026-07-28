//! Draining and restarting into a freshly-applied binary.
//!
//! Replacing the binary on disk does not change the running process: it goes on
//! executing the old image, and `current_exe()` starts reporting
//! `".../swarmllm (deleted)"` (gotcha #188). Until v0.3.39 that also broke every
//! inference on the node, because worker spawning used that path — so a node
//! that had "updated" kept advertising its shards while being unable to serve
//! any of them. The fix made the daemon survive it; this module removes the
//! reason to be in that state at all.
//!
//! A tester hit the visible half of this on 2026-07-28: `swarmllm update`
//! replaced the binary, the daemon kept reporting the old version, and only a
//! manual SIGTERM and relaunch actually applied it. The dashboard button said
//! "Apply & Restart" while the code deliberately did not restart.

use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::daemon::SharedState;

/// How long to wait for in-flight work to finish before restarting anyway.
///
/// Generous because the work being waited on is somebody's answer: a long
/// prompt on a CPU node can legitimately take minutes (prefill is ~99% of a
/// long request), and cutting it off to install an update is a worse outcome
/// than installing a few minutes later. Bounded because a wedged request must
/// not defer an update forever.
pub const DRAIN_TIMEOUT: Duration = Duration::from_secs(600);

/// How often to re-check whether the node has gone idle.
const DRAIN_POLL: Duration = Duration::from_secs(2);

/// Wait until this node is neither running nor serving any inference.
///
/// Returns `true` if the node went idle, `false` if [`DRAIN_TIMEOUT`] expired
/// first (the caller decides whether that is still worth restarting for).
///
/// Consults BOTH `active_pipelines` and `serving_models`, and the second is the
/// one that is easy to forget: `active_pipelines` is the *coordinator's* map
/// and never contains work this node is doing on a peer's behalf, so a node
/// that does nothing but answer other people looks permanently idle through it
/// alone (gotcha #194 — that exact blind spot got a worker killed mid-answer).
pub async fn drain(state: &Arc<SharedState>) -> bool {
    let started = Instant::now();
    loop {
        let coordinating = state.active_pipelines.len();
        let serving = state.serving_models.len();
        if coordinating == 0 && serving == 0 {
            return true;
        }
        if started.elapsed() >= DRAIN_TIMEOUT {
            tracing::warn!(
                coordinating,
                serving,
                timeout_secs = DRAIN_TIMEOUT.as_secs(),
                "Still busy after the drain window — restarting into the update anyway"
            );
            return false;
        }
        tokio::time::sleep(DRAIN_POLL).await;
    }
}

/// Replace this process with the binary at `exe`, keeping the original argv.
///
/// On Unix this is `execv`: same PID, same parent, same file descriptors, so it
/// works identically under systemd (no `Restart=` policy involved — the service
/// never exits), under a plain `swarmllm run` in a terminal, and under a
/// process supervisor. Exiting and hoping something restarts us would only work
/// for the first of those, and the packaged unit is `Restart=on-failure`, which
/// deliberately does not restart a clean exit.
///
/// Only returns on failure — success never returns.
pub fn exec_into(exe: &std::path::Path) -> std::io::Error {
    let args: Vec<std::ffi::OsString> = std::env::args_os().skip(1).collect();
    tracing::info!(exe = %exe.display(), "Restarting into the updated binary");

    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        std::process::Command::new(exe).args(&args).exec()
    }

    #[cfg(not(unix))]
    {
        // Windows has no exec: spawn a replacement and let this process exit.
        // The new process inherits the console, so an interactive user keeps
        // their window. A service wrapper sees the old process exit cleanly.
        match std::process::Command::new(exe).args(&args).spawn() {
            Ok(_) => {
                tracing::info!("Replacement process spawned — exiting");
                std::process::exit(0);
            }
            Err(e) => e,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn drain_timeout_is_long_enough_for_a_real_request() {
        // Prefill on a modest CPU node has been measured at 285-320s for a
        // ~600-1300 token prompt (gotcha #181). A drain window shorter than
        // that would routinely cut off a healthy request to install an update.
        assert!(
            DRAIN_TIMEOUT >= Duration::from_secs(320),
            "drain window must outlast a legitimate long prefill"
        );
    }
}
