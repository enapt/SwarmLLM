use std::path::PathBuf;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use tokio::sync::{watch, RwLock};

use crate::config::UpdateConfig;
use crate::error::SwarmError;

/// Canonical GitHub repo slug used by both the daemon's UpdateChecker
/// subsystem and the standalone `swarmllm update` CLI. Single source of
/// truth — keep `swarmllm::update::SWARMLLM_GITHUB_REPO` in sync if the
/// repository moves.
pub const SWARMLLM_GITHUB_REPO: &str = "enapt/SwarmLLM";

/// HTTP timeout for update-check requests (small GitHub API call).
const UPDATE_CHECK_TIMEOUT_SECS: u64 = 15;
/// HTTP timeout for the update-download request (binary transfer).
const UPDATE_DOWNLOAD_TIMEOUT_SECS: u64 = 300;
/// Delay between daemon start and the first update check — lets the rest of
/// the node finish initializing before we touch the network.
const UPDATE_STARTUP_DELAY_SECS: u64 = 30;

static UPDATE_CHECK_CLIENT: std::sync::LazyLock<reqwest::Client> = std::sync::LazyLock::new(|| {
    crate::http::build_client(|b| {
        b.user_agent(concat!("SwarmLLM/", env!("CARGO_PKG_VERSION")))
            .timeout(std::time::Duration::from_secs(UPDATE_CHECK_TIMEOUT_SECS))
    })
});

static UPDATE_DOWNLOAD_CLIENT: std::sync::LazyLock<reqwest::Client> =
    std::sync::LazyLock::new(|| {
        crate::http::build_client(|b| {
            b.user_agent(concat!("SwarmLLM/", env!("CARGO_PKG_VERSION")))
                .timeout(std::time::Duration::from_secs(UPDATE_DOWNLOAD_TIMEOUT_SECS))
        })
    });

/// Information about an available update.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct UpdateInfo {
    pub latest_version: String,
    pub current_version: String,
    pub download_url: String,
    pub changelog: String,
    pub published_at: String,
    /// SHA256 checksum (hex) if a .sha256 sidecar asset exists.
    pub checksum_sha256: Option<String>,
    /// Whether the update binary has been downloaded and is ready to apply.
    #[serde(default)]
    pub downloaded: bool,
}

/// State for the update checker, stored in SharedState.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct UpdateState {
    pub update_available: Option<UpdateInfo>,
    pub last_checked: Option<String>,
    pub last_error: Option<String>,
}

/// Performs update checks against the GitHub releases API.
pub struct UpdateChecker {
    config: UpdateConfig,
    /// GitHub repo in "owner/repo" format (e.g. "enapt/SwarmLLM").
    repo: String,
    /// Path to the running binary.
    binary_path: PathBuf,
    /// Shared update state.
    state: Arc<RwLock<UpdateState>>,
    /// Dashboard signal sender for update notifications.
    dashboard_tx: tokio::sync::broadcast::Sender<crate::daemon::state::DashboardSignal>,
}

/// GitHub release API response (subset of fields we need).
#[derive(Debug, Deserialize)]
struct GitHubRelease {
    tag_name: String,
    body: Option<String>,
    published_at: Option<String>,
    assets: Vec<GitHubAsset>,
}

#[derive(Debug, Deserialize)]
struct GitHubAsset {
    name: String,
    browser_download_url: String,
}

impl UpdateChecker {
    pub fn new(
        config: UpdateConfig,
        repo: String,
        state: Arc<RwLock<UpdateState>>,
        dashboard_tx: tokio::sync::broadcast::Sender<crate::daemon::state::DashboardSignal>,
    ) -> Self {
        // current_exe() can fail in sandboxed environments (seccomp, certain
        // container runtimes, missing /proc/self/exe). The "swarmllm" fallback
        // resolves against CWD at apply-time, which is almost never the install
        // dir — log loudly so operators know auto-update will fail.
        let binary_path = std::env::current_exe().unwrap_or_else(|e| {
            tracing::warn!(
                error = %e,
                "current_exe() failed — auto-update disabled (binary path unknown)"
            );
            PathBuf::from("swarmllm")
        });
        Self {
            config,
            repo,
            binary_path,
            state,
            dashboard_tx,
        }
    }

    /// Check GitHub for a newer release. Returns `Some(UpdateInfo)` if an update is available.
    pub async fn check_for_update(&self) -> Result<Option<UpdateInfo>, SwarmError> {
        tracing::debug!("DIAG: check_for_update starting");
        let current = env!("CARGO_PKG_VERSION");
        let url = format!("https://api.github.com/repos/{}/releases/latest", self.repo);

        let resp = UPDATE_CHECK_CLIENT
            .get(&url)
            .send()
            .await
            .map_err(|e| SwarmError::Network(format!("GitHub API request failed: {e}")))?;

        if !resp.status().is_success() {
            let status = resp.status();
            return Err(SwarmError::Network(format!("GitHub API returned {status}")));
        }

        let release: GitHubRelease = resp
            .json()
            .await
            .map_err(|e| SwarmError::Network(format!("Failed to parse release JSON: {e}")))?;

        let latest_tag = release.tag_name.trim_start_matches('v').to_string();

        tracing::debug!(current, latest = %latest_tag, "DIAG: check_for_update version compare");

        if !is_newer_version(current, &latest_tag) {
            // Update last_checked even when no update is found
            let mut state = self.state.write().await;
            state.last_checked = Some(chrono::Utc::now().to_rfc3339());
            state.last_error = None;
            return Ok(None);
        }

        // Find the binary asset for this platform
        let (os_str, arch_str) = platform_strings();
        let asset_name = if cfg!(target_os = "windows") {
            format!("swarmllm-{os_str}-{arch_str}.exe")
        } else {
            format!("swarmllm-{os_str}-{arch_str}")
        };

        let binary_asset = release.assets.iter().find(|a| a.name == asset_name);

        let download_url = match binary_asset {
            Some(asset) => asset.browser_download_url.clone(),
            None => {
                tracing::warn!(
                    expected = %asset_name,
                    available = ?release.assets.iter().map(|a| &a.name).collect::<Vec<_>>(),
                    "No matching binary asset found for this platform"
                );
                return Ok(None);
            }
        };

        // Look for a .sha256 checksum sidecar
        let checksum_sha256 = if let Some(sha_asset) = release
            .assets
            .iter()
            .find(|a| a.name == format!("{asset_name}.sha256"))
        {
            match UPDATE_CHECK_CLIENT
                .get(&sha_asset.browser_download_url)
                .send()
                .await
            {
                Ok(resp) if resp.status().is_success() => {
                    resp.text().await.ok().map(|t| t.trim().to_string())
                }
                _ => None,
            }
        } else {
            None
        };

        let info = UpdateInfo {
            latest_version: latest_tag,
            current_version: current.to_string(),
            download_url,
            changelog: release.body.unwrap_or_default(),
            published_at: release.published_at.unwrap_or_default(),
            checksum_sha256,
            downloaded: false,
        };

        Ok(Some(info))
    }

    /// Download the update binary to a temp file alongside the current binary.
    /// Path that `download_update` will stage to when the install dir is
    /// writable — same filesystem as the running binary, so the atomic
    /// rename in `apply_update` succeeds.
    pub fn preferred_tmp_path(&self) -> PathBuf {
        self.binary_path.with_extension("update.tmp")
    }

    pub async fn download_update(&self, info: &UpdateInfo) -> Result<PathBuf, SwarmError> {
        // SECURITY: Only allow downloads from GitHub to prevent SSRF via poisoned API response
        if !info.download_url.starts_with("https://github.com/")
            && !info
                .download_url
                .starts_with("https://objects.githubusercontent.com/")
        {
            return Err(SwarmError::Validation(format!(
                "Update rejected: download URL is not from GitHub: {}",
                info.download_url
            )));
        }

        // Pick a writable location for the staging file. The natural choice is
        // alongside the binary so apply_update's atomic rename stays on the same
        // filesystem. But systemd-installed deb/rpm packages run as user
        // `swarmllm` with no write access to /usr/bin/, so File::create EPERMs.
        // Probe once, then fall back to the OS temp dir on PermissionDenied so
        // the download still completes (apply_update will fail loudly with the
        // real permission error or EXDEV across filesystems — that's the user's
        // signal to update via their package manager instead).
        let preferred_tmp = self.binary_path.with_extension("update.tmp");
        let tmp_path = match tokio::fs::File::create(&preferred_tmp).await {
            Ok(f) => {
                drop(f);
                preferred_tmp
            }
            Err(e) if e.kind() == std::io::ErrorKind::PermissionDenied => {
                let pid = std::process::id();
                let fallback = std::env::temp_dir().join(format!("swarmllm-{pid}.update.tmp"));
                tracing::warn!(
                    install_dir = %self.binary_path.parent().map(|p| p.display().to_string()).unwrap_or_default(),
                    fallback = %fallback.display(),
                    "Install dir not writable for daemon user — staging update in temp dir. \
                     `apply_update` will likely fail; consider updating via your package manager (deb/rpm) instead."
                );
                fallback
            }
            Err(e) => return Err(SwarmError::Io(e)),
        };

        let client = &*UPDATE_DOWNLOAD_CLIENT;

        tracing::info!(
            url = %info.download_url,
            dest = %tmp_path.display(),
            "Downloading update"
        );

        let resp = client
            .get(&info.download_url)
            .send()
            .await
            .map_err(|e| SwarmError::Network(format!("Download request failed: {e}")))?;

        if !resp.status().is_success() {
            return Err(SwarmError::Network(format!(
                "Download failed with status {}",
                resp.status()
            )));
        }

        // Check Content-Length to reject absurdly large downloads before buffering
        const MAX_UPDATE_SIZE: u64 = 500 * 1024 * 1024; // 500 MB
        if let Some(content_length) = resp.content_length() {
            if content_length > MAX_UPDATE_SIZE {
                return Err(SwarmError::Validation(format!(
                    "Update binary too large: {} bytes (max {} bytes)",
                    content_length, MAX_UPDATE_SIZE
                )));
            }
        }

        // Stream the body to disk while incrementally hashing.
        // Avoids buffering up to MAX_UPDATE_SIZE in RAM and lets us abort early
        // on oversize bodies that omit Content-Length.
        use sha2::{Digest, Sha256};
        use tokio::io::AsyncWriteExt;
        let mut hasher = Sha256::new();
        let mut total: u64 = 0;
        let mut file = tokio::fs::File::create(&tmp_path)
            .await
            .map_err(SwarmError::Io)?;
        {
            use futures::StreamExt;
            let mut stream = resp.bytes_stream();
            let cleanup = |tp: &std::path::Path| {
                let tp = tp.to_path_buf();
                tokio::spawn(async move {
                    let _ = tokio::fs::remove_file(&tp).await;
                });
            };
            while let Some(chunk) = stream.next().await {
                let chunk = match chunk {
                    Ok(c) => c,
                    Err(e) => {
                        drop(file);
                        cleanup(&tmp_path);
                        return Err(SwarmError::Network(format!(
                            "Failed to read response body: {e}"
                        )));
                    }
                };
                total = total.saturating_add(chunk.len() as u64);
                if total > MAX_UPDATE_SIZE {
                    drop(file);
                    let _ = tokio::fs::remove_file(&tmp_path).await;
                    return Err(SwarmError::Validation(format!(
                        "Update binary too large (>{} bytes) — aborting download",
                        MAX_UPDATE_SIZE
                    )));
                }
                hasher.update(&chunk);
                if let Err(e) = file.write_all(&chunk).await {
                    drop(file);
                    cleanup(&tmp_path);
                    return Err(SwarmError::Io(e));
                }
            }
            if let Err(e) = file.flush().await {
                drop(file);
                cleanup(&tmp_path);
                return Err(SwarmError::Io(e));
            }
        }
        drop(file);

        // Verify SHA256 checksum — MANDATORY for security.
        // Reject updates without a .sha256 sidecar to prevent accepting unverified binaries.
        match info.checksum_sha256 {
            Some(ref expected_hash) => {
                let actual_hash = hex::encode(hasher.finalize());
                // The .sha256 file may contain "hash  filename" format
                let expected_trimmed = expected_hash
                    .split_whitespace()
                    .next()
                    .unwrap_or(expected_hash);
                if actual_hash != expected_trimmed {
                    let _ = tokio::fs::remove_file(&tmp_path).await;
                    return Err(SwarmError::ShardIntegrity {
                        expected: expected_trimmed.to_string(),
                        actual: actual_hash,
                    });
                }
                tracing::info!("Update checksum verified (SHA256)");
            }
            None => {
                let _ = tokio::fs::remove_file(&tmp_path).await;
                return Err(SwarmError::Validation(
                    "Update rejected: no SHA256 checksum available. Release must include a .sha256 sidecar file.".to_string(),
                ));
            }
        }

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let tp = tmp_path.clone();
            tokio::task::spawn_blocking(move || -> Result<(), std::io::Error> {
                std::fs::set_permissions(&tp, std::fs::Permissions::from_mode(0o755))
            })
            .await
            .map_err(|e| SwarmError::Internal(format!("spawn_blocking: {e}")))?
            .map_err(SwarmError::Io)?;
        }

        tracing::info!(
            bytes = total,
            path = %tmp_path.display(),
            "Update binary downloaded"
        );

        Ok(tmp_path)
    }

    /// Apply the downloaded update: atomic rename of binaries.
    /// Does NOT restart the daemon — the user must restart manually.
    ///
    /// `latest_version` must be strictly newer than the running version.
    /// This guards against downgrade-by-replay: a stored `UpdateInfo`
    /// pointing at an older release must not be silently re-applied even
    /// if the SHA256 still matches.
    ///
    /// `expected_checksum_sha256` re-verifies the staged file's hash before
    /// the rename. Between download (where the hash was first verified)
    /// and apply, the staging file sits on disk for an unbounded interval
    /// (the dashboard "check / apply" buttons are separate calls). A
    /// process running as the same user can swap the staging file during
    /// that window. Re-hashing here closes that TOCTOU.
    pub fn apply_update(
        &self,
        tmp_path: &std::path::Path,
        latest_version: &str,
        expected_checksum_sha256: Option<&str>,
    ) -> Result<(), SwarmError> {
        self.apply_update_with_version(tmp_path, Some(latest_version), expected_checksum_sha256)
    }

    fn apply_update_with_version(
        &self,
        tmp_path: &std::path::Path,
        latest_version: Option<&str>,
        expected_checksum_sha256: Option<&str>,
    ) -> Result<(), SwarmError> {
        tracing::debug!(path = %tmp_path.display(), "DIAG: apply_update starting");
        if !tmp_path.exists() {
            return Err(SwarmError::ServiceUnavailable(
                "Update file not found — download first".to_string(),
            ));
        }

        // SEC: re-verify the version is strictly newer than the running build at
        // apply time. The version was checked in `check_for_update`, but the
        // UpdateInfo can sit in shared state for arbitrary time and a downgrade
        // would otherwise bypass the version gate.
        if let Some(target) = latest_version {
            let current = env!("CARGO_PKG_VERSION");
            if !is_newer_version(current, target) {
                return Err(SwarmError::Validation(format!(
                    "Refusing to apply update: target version {target} is not newer than running {current}"
                )));
            }
        }

        // SEC: re-hash the staged file before rename. Closes the TOCTOU
        // between download (which verifies hash) and apply (which until
        // now only checked tmp_path.exists()). A local process can swap
        // the staged file during the gap; without re-verifying we'd then
        // rename adversary-supplied bytes onto the binary path.
        if let Some(expected) = expected_checksum_sha256 {
            use sha2::{Digest, Sha256};
            let bytes = std::fs::read(tmp_path)
                .map_err(|e| SwarmError::ServiceUnavailable(format!("read staged file: {e}")))?;
            let actual = hex::encode(Sha256::digest(&bytes));
            // Sidecar files often have the form "<hash>  <filename>"; take
            // only the first whitespace-delimited token.
            let expected_trimmed = expected.split_whitespace().next().unwrap_or(expected);
            if !actual.eq_ignore_ascii_case(expected_trimmed) {
                let _ = std::fs::remove_file(tmp_path);
                return Err(SwarmError::Validation(format!(
                    "Staged update file SHA256 mismatch (expected {expected_trimmed}, got {actual}) — staging file rejected"
                )));
            }
        }

        // Windows locks the running .exe — rename fails with ACCESS_DENIED.
        // Reject early with a clear message instead of a confusing I/O error.
        #[cfg(target_os = "windows")]
        {
            return Err(SwarmError::Validation(
                "Auto-update apply is not supported on Windows. Download the new version manually and replace the binary after stopping the daemon.".to_string(),
            ));
        }

        #[cfg(not(target_os = "windows"))]
        {
            let backup_path = self.binary_path.with_extension("old");

            // Step 1: Rename current binary to .old (backup)
            if self.binary_path.exists() {
                std::fs::rename(&self.binary_path, &backup_path).map_err(|e| {
                    SwarmError::ServiceUnavailable(format!("Failed to backup current binary: {e}"))
                })?;
            }

            // Step 2: Rename .update.tmp to current binary path
            if let Err(e) = std::fs::rename(tmp_path, &self.binary_path) {
                // Rollback: restore backup
                let _ = std::fs::rename(&backup_path, &self.binary_path);
                return Err(SwarmError::ServiceUnavailable(format!(
                    "Failed to install update (rolled back): {e}"
                )));
            }

            tracing::info!(
                new = %self.binary_path.display(),
                backup = %backup_path.display(),
                "Update applied — restart required to use new version"
            );

            Ok(())
        }
    }

    /// Background update loop — checks periodically and stores results in shared state.
    pub async fn run(&self, mut shutdown_rx: watch::Receiver<bool>) {
        use crate::config::AutoUpdateMode;

        if self.config.auto_update == AutoUpdateMode::Disabled {
            tracing::info!("Update checking disabled");
            return;
        }

        let interval =
            std::time::Duration::from_secs(self.config.check_interval_hours as u64 * 3600);

        // Initial check after a short delay (let daemon finish starting)
        tokio::select! {
            _ = tokio::time::sleep(std::time::Duration::from_secs(UPDATE_STARTUP_DELAY_SECS)) => {}
            _ = shutdown_rx.changed() => {
                if *shutdown_rx.borrow() { return; }
            }
        }

        loop {
            match self.check_for_update().await {
                Ok(Some(info)) => {
                    tracing::info!(
                        current = %info.current_version,
                        latest = %info.latest_version,
                        "Update available"
                    );

                    // Auto-download if auto_update is enabled
                    // Stable mode skips pre-release versions (tags containing '-')
                    let mut info = info;
                    let is_prerelease = info.latest_version.contains('-');
                    let should_download = match self.config.auto_update {
                        crate::config::AutoUpdateMode::Disabled => false,
                        crate::config::AutoUpdateMode::Stable => !is_prerelease,
                        crate::config::AutoUpdateMode::All => true,
                    };
                    if should_download {
                        match self.download_update(&info).await {
                            Ok(path) => {
                                // Only mark `downloaded = true` if the staging
                                // file is alongside the running binary — that
                                // path is on the same filesystem so the atomic
                                // rename in apply_update will succeed. The
                                // EPERM-fallback to temp_dir typically lives
                                // on a different filesystem; apply will fail
                                // with EXDEV. The dashboard "ready to apply"
                                // banner would otherwise mislead the operator.
                                let preferred = self.binary_path.with_extension("update.tmp");
                                let appliable = path == preferred;
                                info.downloaded = appliable;
                                if appliable {
                                    tracing::info!("Update downloaded and ready to apply");
                                } else {
                                    tracing::warn!(
                                        path = %path.display(),
                                        "Update staged in temp dir — apply via package manager required"
                                    );
                                }
                            }
                            Err(e) => {
                                tracing::warn!(error = %e, "Failed to auto-download update");
                            }
                        }
                    }

                    // Store in shared state and notify WebSocket clients
                    {
                        let mut state = self.state.write().await;
                        state.update_available = Some(info.clone());
                        state.last_checked = Some(chrono::Utc::now().to_rfc3339());
                        state.last_error = None;
                    }
                    let _ = self
                        .dashboard_tx
                        .send(crate::daemon::state::DashboardSignal::UpdateAvailable(info));
                }
                Ok(None) => {
                    tracing::debug!("No update available");
                }
                Err(e) => {
                    tracing::warn!(error = %e, "Update check failed");
                    let mut state = self.state.write().await;
                    state.last_checked = Some(chrono::Utc::now().to_rfc3339());
                    state.last_error = Some(e.to_string());
                }
            }

            // Wait for next check interval or shutdown
            tokio::select! {
                _ = tokio::time::sleep(interval) => {}
                _ = shutdown_rx.changed() => {
                    if *shutdown_rx.borrow() {
                        tracing::debug!("Update checker shutting down");
                        return;
                    }
                }
            }
        }
    }
}

/// Compare two semver strings. Returns true if `latest` is newer than `current`.
pub(crate) fn is_newer_version(current: &str, latest: &str) -> bool {
    let parse = |s: &str| -> (u64, u64, u64) {
        let parts: Vec<&str> = s.split('.').collect();
        let major = parts.first().and_then(|p| p.parse().ok()).unwrap_or(0);
        let minor = parts.get(1).and_then(|p| p.parse().ok()).unwrap_or(0);
        let patch = parts
            .get(2)
            .and_then(|p| {
                // Handle pre-release suffixes like "1.0.0-rc1"
                p.split('-').next().and_then(|n| n.parse().ok())
            })
            .unwrap_or(0);
        (major, minor, patch)
    };

    let (c_major, c_minor, c_patch) = parse(current);
    let (l_major, l_minor, l_patch) = parse(latest);

    if (l_major, l_minor, l_patch) > (c_major, c_minor, c_patch) {
        return true;
    }
    // Same base version: promote pre-release to stable (e.g., 0.2.0-rc1 → 0.2.0)
    if (l_major, l_minor, l_patch) == (c_major, c_minor, c_patch) {
        let current_is_prerelease = current.contains('-');
        let latest_is_stable = !latest.contains('-');
        if current_is_prerelease && latest_is_stable {
            return true;
        }
    }
    false
}

/// Return (os, arch) strings matching GitHub release asset naming.
fn platform_strings() -> (&'static str, &'static str) {
    let os = if cfg!(target_os = "linux") {
        "linux"
    } else if cfg!(target_os = "macos") {
        "macos"
    } else if cfg!(target_os = "windows") {
        "windows"
    } else {
        "unknown"
    };

    let arch = if cfg!(target_arch = "x86_64") {
        "x86_64"
    } else if cfg!(target_arch = "aarch64") {
        "aarch64"
    } else {
        "unknown"
    };

    (os, arch)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn semver_newer() {
        assert!(is_newer_version("0.1.0", "0.2.0"));
        assert!(is_newer_version("0.1.0", "1.0.0"));
        assert!(is_newer_version("1.0.0", "1.0.1"));
        assert!(is_newer_version("0.9.9", "1.0.0"));
    }

    #[test]
    fn semver_same_or_older() {
        assert!(!is_newer_version("0.1.0", "0.1.0"));
        assert!(!is_newer_version("1.0.0", "0.9.0"));
        assert!(!is_newer_version("2.0.0", "1.99.99"));
    }

    #[test]
    fn semver_with_v_prefix() {
        // The tag is stripped of 'v' before calling is_newer_version
        assert!(is_newer_version("0.1.0", "0.2.0"));
    }

    #[test]
    fn semver_prerelease_stripped() {
        assert!(is_newer_version("0.1.0", "0.2.0-rc1"));
        assert!(!is_newer_version("0.2.0", "0.2.0-rc1"));
    }

    #[test]
    fn platform_detection() {
        let (os, arch) = platform_strings();
        // Just verify they return non-"unknown" on common platforms
        #[cfg(target_os = "linux")]
        assert_eq!(os, "linux");
        #[cfg(target_arch = "x86_64")]
        assert_eq!(arch, "x86_64");
        // Suppress warnings on other platforms
        let _ = (os, arch);
    }

    #[test]
    fn update_config_defaults() {
        let config = UpdateConfig::default();
        assert_eq!(config.auto_update, crate::config::AutoUpdateMode::Disabled);
        assert_eq!(config.check_interval_hours, 6);
    }

    #[test]
    fn update_info_serde_roundtrip() {
        let info = UpdateInfo {
            latest_version: "1.0.0".into(),
            current_version: "0.1.0".into(),
            download_url: "https://example.com/bin".into(),
            changelog: "Bug fixes".into(),
            published_at: "2026-01-01T00:00:00Z".into(),
            checksum_sha256: Some("abc123".into()),
            downloaded: false,
        };
        let json = serde_json::to_string(&info).unwrap();
        let parsed: UpdateInfo = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.latest_version, "1.0.0");
        assert_eq!(parsed.checksum_sha256, Some("abc123".into()));
    }

    #[test]
    fn update_state_default() {
        let state = UpdateState::default();
        assert!(state.update_available.is_none());
        assert!(state.last_checked.is_none());
        assert!(state.last_error.is_none());
    }
}
