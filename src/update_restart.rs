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

/// Carries the process id of the build that spawned this one, so a replacement
/// can wait for its predecessor to let go before it tries to take over.
///
/// Only ever set by [`exec_into`] on Windows, and always to the CURRENT process
/// id — never inherited unchanged, or a node that updated twice would wait on
/// the id of a process two generations back (and, after id reuse, on whatever
/// holds it now).
pub const HANDOFF_PID_VAR: &str = "SWARMLLM_UPDATE_HANDOFF_PID";

/// Longest a replacement waits for its predecessor. Generous relative to what
/// it is waiting for — a process that has already called `exit` — and bounded
/// because waiting forever on a predecessor that will not die is worse than
/// starting and reporting the port conflict.
const HANDOFF_WAIT: Duration = Duration::from_secs(30);

/// How often to re-check.
const HANDOFF_POLL: Duration = Duration::from_millis(100);

/// Wait for the build that spawned this one during an update to exit.
///
/// **Unix does not need this and never calls it.** There, `exec_into` replaces
/// the process image: same pid, same open files, so nothing is ever held twice.
/// Windows has no `exec`, so the replacement is a SECOND process that starts
/// while the first is still exiting — and the two things it needs, the API port
/// and the redb lock, are both exclusive and neither is retried. Losing that
/// race costs the user their node: the predecessor has gone, the replacement
/// refuses to start, and nothing tries again.
///
/// Ordinary starts are unaffected — with no handoff variable set this returns
/// immediately, so the `Port … is already in use` message still means what it
/// has always meant.
pub fn await_predecessor_exit() {
    let Ok(raw) = std::env::var(HANDOFF_PID_VAR) else {
        return;
    };
    // Deliberately NOT removed from the environment afterwards. It is tempting
    // — worker subprocesses inherit it and it means nothing to them — but
    // `std::env::remove_var` mutates a global while every Tokio worker thread
    // is already running, which is the hazard that made it `unsafe` in edition
    // 2024. It buys nothing here either: `exec_into` writes this variable
    // afresh on every handoff, so a value from an earlier generation can never
    // be the one acted on.
    let Ok(pid) = raw.trim().parse::<u32>() else {
        tracing::warn!(value = %raw, "Ignoring an unreadable update handoff process id");
        return;
    };

    tracing::info!(
        predecessor_pid = pid,
        "Started by an update — waiting for the previous version to finish exiting"
    );
    let waited = wait_while_alive(
        || process_is_running(pid),
        HANDOFF_WAIT,
        HANDOFF_POLL,
        std::time::Instant::now,
    );
    match waited {
        Some(elapsed) => tracing::info!(
            waited_ms = elapsed.as_millis() as u64,
            "The previous version has exited — starting"
        ),
        None => tracing::warn!(
            predecessor_pid = pid,
            timeout_secs = HANDOFF_WAIT.as_secs(),
            "The previous version is still running. Starting anyway; if it still holds the API \
             port or the database this node will say so and stop, and starting it again by hand \
             will work"
        ),
    }
}

/// Is a process with this id running?
fn process_is_running(pid: u32) -> bool {
    let mut sys = sysinfo::System::new();
    let pid = sysinfo::Pid::from_u32(pid);
    sys.refresh_processes(sysinfo::ProcessesToUpdate::Some(&[pid]), true);
    sys.process(pid).is_some()
}

/// Poll `alive` until it answers false, and say how long that took — or `None`
/// if it never did.
///
/// The clock is a parameter so this can be tested without one: what is being
/// asserted is the loop's shape, and a test that waits out real timeouts to
/// check a timeout is a test nobody runs twice.
fn wait_while_alive(
    mut alive: impl FnMut() -> bool,
    timeout: Duration,
    poll: Duration,
    now: impl Fn() -> Instant,
) -> Option<Duration> {
    let started = now();
    loop {
        if !alive() {
            return Some(now().saturating_duration_since(started));
        }
        if now().saturating_duration_since(started) >= timeout {
            return None;
        }
        std::thread::sleep(poll);
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
        //
        // Unlike `exec`, this leaves two processes alive for an instant, both
        // wanting the API port and the redb lock. `HANDOFF_PID_VAR` is how the
        // replacement knows to wait for this one — see `await_predecessor_exit`.
        match std::process::Command::new(exe)
            .args(&args)
            .env(HANDOFF_PID_VAR, std::process::id().to_string())
            .spawn()
        {
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

    /// The predecessor going away ends the wait, and the answer says how long
    /// it took rather than merely that it happened.
    #[test]
    fn the_wait_ends_when_the_previous_version_exits() {
        let mut polls = 0;
        let waited = wait_while_alive(
            || {
                polls += 1;
                polls < 3
            },
            Duration::from_secs(30),
            Duration::from_millis(1),
            Instant::now,
        );
        assert!(
            waited.is_some(),
            "a predecessor that exits must end the wait"
        );
        assert_eq!(polls, 3);
    }

    /// A predecessor that never exits must not hold the replacement for ever.
    /// Starting and reporting the port conflict is the better failure: the node
    /// says something, and starting it again by hand then works.
    ///
    /// Driven by a fake clock — a test that waits out a 30-second timeout to
    /// check a 30-second timeout is one nobody runs twice.
    #[test]
    fn a_predecessor_that_never_exits_does_not_block_startup_for_ever() {
        let start = Instant::now();
        let ticks = std::cell::Cell::new(0u32);
        let waited = wait_while_alive(
            || true,
            Duration::from_secs(30),
            Duration::from_millis(1),
            || {
                let t = ticks.get();
                ticks.set(t + 1);
                start + Duration::from_secs(u64::from(t) * 10)
            },
        );
        assert!(
            waited.is_none(),
            "the wait must give up rather than hang a node's startup on a stuck predecessor"
        );
    }

    /// An ordinary start — no update, no variable — must not wait at all, or
    /// every node start pays for a case that only arises on Windows after an
    /// update.
    #[test]
    fn an_ordinary_start_does_not_wait() {
        assert!(
            std::env::var(HANDOFF_PID_VAR).is_err(),
            "test environment must not carry a handoff id"
        );
        let before = Instant::now();
        await_predecessor_exit();
        assert!(before.elapsed() < Duration::from_secs(1));
    }

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
