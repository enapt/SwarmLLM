use std::path::PathBuf;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use tokio::sync::{watch, RwLock};

use crate::config::UpdateConfig;
use crate::error::SwarmError;

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
    /// Broadcast sender for update notifications (WebSocket will subscribe).
    update_tx: tokio::sync::broadcast::Sender<UpdateInfo>,
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
        update_tx: tokio::sync::broadcast::Sender<UpdateInfo>,
    ) -> Self {
        let binary_path = std::env::current_exe().unwrap_or_else(|_| PathBuf::from("swarmllm"));
        Self {
            config,
            repo,
            binary_path,
            state,
            update_tx,
        }
    }

    /// Check GitHub for a newer release. Returns `Some(UpdateInfo)` if an update is available.
    pub async fn check_for_update(&self) -> Result<Option<UpdateInfo>, SwarmError> {
        tracing::debug!("DIAG: check_for_update starting");
        let current = env!("CARGO_PKG_VERSION");
        let url = format!("https://api.github.com/repos/{}/releases/latest", self.repo);

        let client = reqwest::Client::builder()
            .user_agent(format!("SwarmLLM/{current}"))
            .timeout(std::time::Duration::from_secs(15))
            .build()
            .map_err(|e| SwarmError::Network(format!("HTTP client error: {e}")))?;

        let resp = client
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
            match client.get(&sha_asset.browser_download_url).send().await {
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
    pub async fn download_update(&self, info: &UpdateInfo) -> Result<PathBuf, SwarmError> {
        // SECURITY: Only allow downloads from GitHub to prevent SSRF via poisoned API response
        if !info.download_url.starts_with("https://github.com/")
            && !info
                .download_url
                .starts_with("https://objects.githubusercontent.com/")
        {
            return Err(SwarmError::Internal(format!(
                "Update rejected: download URL is not from GitHub: {}",
                info.download_url
            )));
        }

        let tmp_path = self.binary_path.with_extension("update.tmp");

        let client = reqwest::Client::builder()
            .user_agent(format!("SwarmLLM/{}", env!("CARGO_PKG_VERSION")))
            .timeout(std::time::Duration::from_secs(300))
            .build()
            .map_err(|e| SwarmError::Network(format!("HTTP client error: {e}")))?;

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
                return Err(SwarmError::Internal(format!(
                    "Update binary too large: {} bytes (max {} bytes)",
                    content_length, MAX_UPDATE_SIZE
                )));
            }
        }

        let bytes = resp
            .bytes()
            .await
            .map_err(|e| SwarmError::Network(format!("Failed to read response body: {e}")))?;

        if bytes.len() as u64 > MAX_UPDATE_SIZE {
            return Err(SwarmError::Internal(format!(
                "Update binary too large: {} bytes (max {} bytes)",
                bytes.len(),
                MAX_UPDATE_SIZE
            )));
        }

        // Verify SHA256 checksum — MANDATORY for security.
        // Reject updates without a .sha256 sidecar to prevent accepting unverified binaries.
        match info.checksum_sha256 {
            Some(ref expected_hash) => {
                use sha2::{Digest, Sha256};
                let mut hasher = Sha256::new();
                hasher.update(&bytes);
                let actual_hash = hex::encode(hasher.finalize());
                // The .sha256 file may contain "hash  filename" format
                let expected_trimmed = expected_hash
                    .split_whitespace()
                    .next()
                    .unwrap_or(expected_hash);
                if actual_hash != expected_trimmed {
                    return Err(SwarmError::ShardIntegrity {
                        expected: expected_trimmed.to_string(),
                        actual: actual_hash,
                    });
                }
                tracing::info!("Update checksum verified (SHA256)");
            }
            None => {
                return Err(SwarmError::Internal(
                    "Update rejected: no SHA256 checksum available. Release must include a .sha256 sidecar file.".to_string(),
                ));
            }
        }

        std::fs::write(&tmp_path, &bytes).map_err(SwarmError::Io)?;

        // Set executable permission on Unix
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let perms = std::fs::Permissions::from_mode(0o755);
            std::fs::set_permissions(&tmp_path, perms).map_err(SwarmError::Io)?;
        }

        tracing::info!(
            bytes = bytes.len(),
            path = %tmp_path.display(),
            "Update binary downloaded"
        );

        Ok(tmp_path)
    }

    /// Apply the downloaded update: atomic rename of binaries.
    /// Does NOT restart the daemon — the user must restart manually.
    pub fn apply_update(&self, tmp_path: &std::path::Path) -> Result<(), SwarmError> {
        tracing::debug!(path = %tmp_path.display(), "DIAG: apply_update starting");
        if !tmp_path.exists() {
            return Err(SwarmError::Internal(
                "Update file not found — download first".to_string(),
            ));
        }

        let backup_path = self.binary_path.with_extension("old");

        // Step 1: Rename current binary to .old (backup)
        if self.binary_path.exists() {
            std::fs::rename(&self.binary_path, &backup_path).map_err(|e| {
                SwarmError::Internal(format!("Failed to backup current binary: {e}"))
            })?;
        }

        // Step 2: Rename .update.tmp to current binary path
        if let Err(e) = std::fs::rename(tmp_path, &self.binary_path) {
            // Rollback: restore backup
            let _ = std::fs::rename(&backup_path, &self.binary_path);
            return Err(SwarmError::Internal(format!(
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
            _ = tokio::time::sleep(std::time::Duration::from_secs(30)) => {}
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
                            Ok(_path) => {
                                info.downloaded = true;
                                tracing::info!("Update downloaded and ready to apply");
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
                    let _ = self.update_tx.send(info);
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
pub fn is_newer_version(current: &str, latest: &str) -> bool {
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
        assert_eq!(config.auto_update, crate::config::AutoUpdateMode::Stable);
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
