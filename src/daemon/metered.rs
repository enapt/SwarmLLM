//! Is this computer's internet connection metered — charged by the amount of
//! data, like a phone's hotspot or a mobile plan?
//!
//! Asked by the automatic download pass just before it fetches
//! (`AutoShardManager::evaluate_and_download`), as Microsoft tells an app to:
//! re-check the cost just before a sizable transfer, and gate large optional
//! downloads on a `Fixed` or `Variable` connection
//! (`Windows.Networking.Connectivity.ConnectionCost`, "adaptive behavior").
//! A tester's laptop took 1.4 GB over a phone's tether in five minutes, and
//! 9 GB in its first 44 (field report 2026-10-15).
//!
//! - **Linux**: NetworkManager's `Metered` property on its manager object — the
//!   primary connection's — where `YES` (1) and `GUESS_YES` (3) are metered.
//!   NetworkManager guesses yes for an Android hotspot, whose DHCP server sends
//!   `ANDROID_METERED`; its own advice is to treat a guess as the answer.
//! - **Windows, and Linux under WSL2** (where NetworkManager is absent but the
//!   Windows host knows): the internet connection's cost, through PowerShell.
//!   `Fixed` and `Variable` are metered, and so is a connection roaming or over
//!   its data limit.
//! - **Anything else** — macOS, a server without NetworkManager, a container —
//!   is unknown, which pauses nothing. `auto_manage.on_metered_connection =
//!   "pause"` is the setting for a node the system cannot tell about.
//!
//! Both readings run a program through `BoundedCommand`: bounded, never two at
//! once, and off the runtime worker.
//!
//! **`SharedState::automatic_downloads_wait` is the one answer** to "may this
//! node start a download nobody asked for now?" — asked by every path that
//! starts one on its own: the download pass (`evaluate_and_download`), which
//! the rebalancer wakes when a holder leaves rather than fetching itself. Guard:
//! `every_automatic_download_asks_whether_the_connection_is_metered`. What it
//! leaves alone, deliberately: a download the owner asks for, a part already
//! held being repaired, and a download already under way.

use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use crate::config::MeteredDownloads;
use crate::daemon::state::{ActivityEvent, SharedState};
#[cfg(any(target_os = "linux", windows))]
use crate::model::auto_manage::vram::BoundedCommand;

/// What the system says about the connection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Metered {
    Yes,
    No,
    /// The system could not be asked, or did not say. Never a reason to pause.
    Unknown,
}

/// The longest either reading may take. Both answer in well under a second
/// (PowerShell 0.3 s from WSL2, measured 2026-10-15).
#[cfg(any(target_os = "linux", windows))]
const ASK_TIMEOUT: Duration = Duration::from_secs(5);

/// Ask the system now. Blocking: it runs a program.
pub fn read() -> Metered {
    #[cfg(target_os = "linux")]
    {
        let said = network_manager();
        if said != Metered::Unknown || !crate::config::network::is_wsl2() {
            return said;
        }
        windows_cost()
    }
    #[cfg(windows)]
    {
        windows_cost()
    }
    #[cfg(not(any(target_os = "linux", windows)))]
    {
        Metered::Unknown
    }
}

#[cfg(target_os = "linux")]
fn network_manager() -> Metered {
    static BUSCTL: BoundedCommand = BoundedCommand::new("busctl", ASK_TIMEOUT);
    BUSCTL
        .run(&[
            "get-property",
            "org.freedesktop.NetworkManager",
            "/org/freedesktop/NetworkManager",
            "org.freedesktop.NetworkManager",
            "Metered",
        ])
        .map_or(Metered::Unknown, |out| from_network_manager(&out))
}

/// `busctl get-property` prints a `u` property as `u 3`. `NMMetered`:
/// 0 unknown, 1 yes, 2 no, 3 guess yes, 4 guess no.
#[cfg(any(target_os = "linux", test))]
pub(crate) fn from_network_manager(out: &str) -> Metered {
    match out
        .trim()
        .strip_prefix("u ")
        .and_then(|n| n.trim().parse::<u32>().ok())
    {
        Some(1 | 3) => Metered::Yes,
        Some(2 | 4) => Metered::No,
        _ => Metered::Unknown,
    }
}

#[cfg(any(target_os = "linux", windows))]
fn windows_cost() -> Metered {
    static POWERSHELL: BoundedCommand = BoundedCommand::new("powershell.exe", ASK_TIMEOUT);
    POWERSHELL
        .run(&["-NoProfile", "-NonInteractive", "-Command", COST_SCRIPT])
        .map_or(Metered::Unknown, |out| from_windows_cost(&out))
}

/// Prints the internet connection's `NetworkCostType`, then whether it is
/// roaming and whether it is over its data limit: `Unrestricted False False`.
/// No internet connection prints nothing.
#[cfg(any(target_os = "linux", windows))]
const COST_SCRIPT: &str = "$c = [Windows.Networking.Connectivity.NetworkInformation,\
Windows.Networking.Connectivity,ContentType=WindowsRuntime]::GetInternetConnectionProfile()\
.GetConnectionCost(); \"$($c.NetworkCostType) $($c.Roaming) $($c.OverDataLimit)\"";

#[cfg(any(target_os = "linux", windows, test))]
pub(crate) fn from_windows_cost(out: &str) -> Metered {
    let mut words = out.split_whitespace();
    let cost = words.next();
    let roaming = words.next() == Some("True");
    let over_limit = words.next() == Some("True");
    match cost {
        Some("Fixed" | "Variable") => Metered::Yes,
        Some("Unrestricted") if roaming || over_limit => Metered::Yes,
        Some("Unrestricted") => Metered::No,
        _ => Metered::Unknown,
    }
}

/// How long a reading answers for. The download pass runs every few minutes,
/// and again whenever a departure or a setting wakes it; a minute keeps those
/// from starting a program each, and is well inside the time a person takes to switch from a
/// hotspot to Wi-Fi and back.
const READING_ANSWERS_FOR: Duration = Duration::from_secs(60);

/// What this node last learned about its connection (`state.models.metered`).
#[derive(Debug, Default)]
pub struct MeteredWatch {
    /// The last reading and when it was taken. An async lock, held while the
    /// system is asked, so a second caller waits for that answer instead of
    /// finding the program busy (`BoundedCommand` runs one copy at a time, and
    /// a busy one reads as unknown).
    reading: tokio::sync::Mutex<Option<(Instant, Metered)>>,
    /// The last thing the system DID say, yes or no. An unknown reading
    /// answers with it: a check that times out on a hotspot is not a reason to
    /// start downloading.
    last_known: AtomicBool,
    /// Whether automatic downloads wait now — so the owner is told once per
    /// change, not once per pass.
    waiting: AtomicBool,
}

impl MeteredWatch {
    /// Whether automatic downloads were waiting when last asked — for display
    /// (the diagnostics report); a decision asks
    /// `SharedState::automatic_downloads_wait`.
    pub fn waiting(&self) -> bool {
        self.waiting.load(Ordering::Relaxed)
    }
}

impl SharedState {
    /// May this node start a download nobody asked for, now? `true` = wait.
    /// The one answer (see the module); asked just before such a download
    /// starts, as Microsoft's guidance for a metered connection says.
    pub async fn automatic_downloads_wait(&self) -> bool {
        let policy = self.cfg().auto_manage.on_metered_connection;
        self.automatic_downloads_wait_asking(policy, read).await
    }

    /// [`Self::automatic_downloads_wait`], asking the system with `ask`.
    pub(crate) async fn automatic_downloads_wait_asking(
        &self,
        policy: MeteredDownloads,
        ask: fn() -> Metered,
    ) -> bool {
        let watch = &self.models.metered;
        let wait = match policy {
            MeteredDownloads::Allow => false,
            MeteredDownloads::Pause => true,
            MeteredDownloads::Auto => {
                let mut reading = watch.reading.lock().await;
                let metered = match *reading {
                    Some((at, metered)) if at.elapsed() < READING_ANSWERS_FOR => metered,
                    _ => {
                        let metered = tokio::task::spawn_blocking(ask)
                            .await
                            .unwrap_or(Metered::Unknown);
                        *reading = Some((Instant::now(), metered));
                        metered
                    }
                };
                match metered {
                    Metered::Yes => watch.last_known.store(true, Ordering::Relaxed),
                    Metered::No => watch.last_known.store(false, Ordering::Relaxed),
                    Metered::Unknown => {}
                }
                watch.last_known.load(Ordering::Relaxed)
            }
        };
        if watch.waiting.swap(wait, Ordering::Relaxed) != wait {
            let (kind, message) = if wait {
                (
                    "metered_paused",
                    "Automatic downloads paused: this connection is metered. \
                     Models you download yourself still download.",
                )
            } else {
                (
                    "metered_resumed",
                    "Automatic downloads resumed: this connection is no longer metered.",
                )
            };
            tracing::info!(?policy, "{message}");
            self.emit_activity(
                ActivityEvent::new("auto_manage", kind, message.to_string())
                    .with_toast("info", 8000),
            );
        }
        wait
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state() -> std::sync::Arc<SharedState> {
        let temp = tempfile::tempdir().expect("tempdir").keep();
        let db = crate::storage::db::Database::open(&temp).expect("db");
        let mut config = crate::config::Config::default();
        config.node.data_dir = temp;
        let executor = std::sync::Arc::new(tokio::sync::Mutex::new(
            crate::inference::executor::ModelExecutor::new(),
        ));
        let identity = crate::identity::Identity::generate();
        SharedState::new(config, identity, db, executor, None).0
    }

    fn told(state: &SharedState) -> Vec<&'static str> {
        state
            .events
            .activity_history
            .lock()
            .iter()
            .filter(|e| e.kind.starts_with("metered_"))
            .map(|e| e.kind)
            .collect()
    }

    #[tokio::test]
    async fn automatic_downloads_wait_on_a_metered_connection_and_the_owner_is_told_once() {
        let state = state();
        let metered = || Metered::Yes;
        assert!(
            state
                .automatic_downloads_wait_asking(MeteredDownloads::Auto, metered)
                .await
        );
        // The reading answers for a minute: the system is not asked again.
        assert!(
            state
                .automatic_downloads_wait_asking(MeteredDownloads::Auto, || panic!("asked again"))
                .await
        );
        assert_eq!(told(&state), ["metered_paused"], "told once, not per pass");
        // The owner says downloads may go ahead whatever the connection.
        assert!(
            !state
                .automatic_downloads_wait_asking(MeteredDownloads::Allow, metered)
                .await
        );
        assert_eq!(told(&state), ["metered_paused", "metered_resumed"]);
        *state.models.metered.reading.lock().await = None;
        assert!(
            !state
                .automatic_downloads_wait_asking(MeteredDownloads::Auto, || Metered::No)
                .await
        );
    }

    #[tokio::test]
    async fn a_system_that_cannot_say_pauses_nothing_and_a_failed_check_changes_nothing() {
        let state = state();
        let unknown = || Metered::Unknown;
        assert!(
            !state
                .automatic_downloads_wait_asking(MeteredDownloads::Auto, unknown)
                .await,
            "macOS, no NetworkManager"
        );
        assert!(
            state
                .automatic_downloads_wait_asking(MeteredDownloads::Pause, unknown)
                .await,
            "the setting for a system that cannot tell"
        );
        *state.models.metered.reading.lock().await = None;
        assert!(
            state
                .automatic_downloads_wait_asking(MeteredDownloads::Auto, || Metered::Yes)
                .await
        );
        // A hotspot whose next check times out is still a hotspot.
        *state.models.metered.reading.lock().await = None;
        assert!(
            state
                .automatic_downloads_wait_asking(MeteredDownloads::Auto, unknown)
                .await
        );
    }

    #[test]
    fn network_managers_answer_is_read_guesses_included() {
        assert_eq!(from_network_manager("u 1\n"), Metered::Yes);
        assert_eq!(
            from_network_manager("u 3\n"),
            Metered::Yes,
            "an Android hotspot"
        );
        assert_eq!(from_network_manager("u 2\n"), Metered::No);
        assert_eq!(from_network_manager("u 4\n"), Metered::No);
        assert_eq!(from_network_manager("u 0\n"), Metered::Unknown);
        assert_eq!(from_network_manager(""), Metered::Unknown);
    }

    #[test]
    fn windows_cost_is_metered_when_charged_roaming_or_over_the_limit() {
        assert_eq!(
            from_windows_cost("Unrestricted False False\r\n"),
            Metered::No
        );
        assert_eq!(from_windows_cost("Fixed False False"), Metered::Yes);
        assert_eq!(from_windows_cost("Variable False False"), Metered::Yes);
        assert_eq!(from_windows_cost("Unrestricted True False"), Metered::Yes);
        assert_eq!(from_windows_cost("Unrestricted False True"), Metered::Yes);
        assert_eq!(from_windows_cost("Unknown False False"), Metered::Unknown);
        assert_eq!(
            from_windows_cost(""),
            Metered::Unknown,
            "no internet connection"
        );
    }
}
