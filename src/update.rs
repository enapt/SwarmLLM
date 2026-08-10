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
/// How long the update download may go SILENT before we give up.
///
/// This was a *total* timeout of 300s, which quietly decided the minimum
/// connection speed a user needs to update at all: the GPU build is ~933 MB, so
/// finishing inside five minutes takes a sustained ~3.1 MB/s. Anyone slower
/// than that could never complete an update — it would fail at the same point
/// every time, with no indication that speed was the reason. The size cap was
/// raised to 2 GB for exactly these binaries without revisiting the clock.
///
/// Measuring silence instead bounds a download that has genuinely stalled while
/// letting a slow but healthy one finish, however long it takes.
const UPDATE_DOWNLOAD_STALL_SECS: u64 = 120;
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
                .read_timeout(std::time::Duration::from_secs(UPDATE_DOWNLOAD_STALL_SECS))
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
    /// Whether this installation can replace its own binary at all.
    ///
    /// False for every managed install: a `.deb`/`.rpm` service runs under
    /// `ProtectSystem=strict` with only its data directory writable, and the
    /// hardened anchor unit is the same but stricter. Those are updated by the
    /// package manager or, for an anchor, by the root-run
    /// `swarmllm-update.timer` in `deploy/anchor/` — which is the correct
    /// design, not a limitation to work around. Surfaced so the dashboard can
    /// say how THIS node updates instead of offering a button that cannot work.
    #[serde(default)]
    pub self_update_supported: bool,
}

/// State for the update checker, stored in SharedState.
/// redb tree holding update bookkeeping.
pub const TREE_UPDATE: &str = "update";
/// Last version the updater reported installing successfully.
pub const KEY_INSTALLED_VERSION: &str = "installed_version";

/// Compare what was last installed against what is running.
///
/// `None` when they agree, when nothing has been installed yet, or when the
/// running version is NEWER than the record (a manual upgrade, or a rollback
/// followed by a fresh install — either way not a stuck restart).
///
/// Deliberately a pure function of two strings so the comparison is testable
/// without a daemon, a disk or an update.
pub fn restart_required_for(running: &str, installed: Option<&str>) -> Option<RestartRequired> {
    let installed = installed?;
    if installed == running {
        return None;
    }
    // Only flag the case that matters: installed something, still running the
    // older image. A deliberate rollback leaves `installed` behind `running`
    // and must not nag.
    if !is_newer_version(running, installed) {
        return None;
    }
    Some(RestartRequired {
        running: running.to_string(),
        installed: installed.to_string(),
    })
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct UpdateState {
    pub update_available: Option<UpdateInfo>,
    pub last_checked: Option<String>,
    pub last_error: Option<String>,
    /// Set when a newer version was installed on disk but this process is still
    /// running an older image — i.e. the restart into it did not happen.
    ///
    /// An operator reported (2026-08-09, again 2026-08-10) believing their node
    /// had silently missed eight releases. The install path `exec`s into the new
    /// binary, which KEEPS the process id and the kernel's start time, so `ps`
    /// shows the original launch either way and cannot distinguish "updated in
    /// place" from "never restarted" (gotcha #277). Nothing else could either:
    /// the node knew what it had installed and what it was running and never
    /// compared them. This is that comparison, so the question stops needing
    /// forensics.
    pub restart_required: Option<RestartRequired>,
}

/// A newer version is installed on disk than the one running.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
pub struct RestartRequired {
    /// Version this process is actually running (compiled in — cannot be stale).
    pub running: String,
    /// Version the updater last reported installing successfully.
    pub installed: String,
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
    /// Present only when running inside the daemon. The standalone
    /// `swarmllm update` CLI has no node to drain, so it stays `None`.
    shared: Option<Arc<crate::daemon::SharedState>>,
}

/// GitHub release API response (subset of fields we need).
#[derive(Debug, Deserialize)]
struct GitHubRelease {
    tag_name: String,
    body: Option<String>,
    published_at: Option<String>,
    assets: Vec<GitHubAsset>,
    /// Pre-release flag (alpha/beta/rc). `/releases/latest` never returns these,
    /// which is why we list `/releases` and filter by mode instead.
    #[serde(default)]
    prerelease: bool,
    /// Draft releases are only visible to authenticated requests; filtered out
    /// defensively so an unpublished release is never selected.
    #[serde(default)]
    draft: bool,
}

#[derive(Debug, Deserialize)]
struct GitHubAsset {
    name: String,
    browser_download_url: String,
}

/// Pick the newest applicable release from a `/releases` list (GitHub returns
/// them newest-first). Always skips drafts; skips pre-releases unless
/// `include_prereleases` (i.e. `AutoUpdateMode::All`). Returns `None` when no
/// release qualifies — e.g. a fresh repo with only draft/pre-release tags in
/// Stable mode.
fn select_target_release(
    releases: &[GitHubRelease],
    include_prereleases: bool,
) -> Option<&GitHubRelease> {
    releases
        .iter()
        .find(|r| !r.draft && (include_prereleases || !r.prerelease))
}

/// Whether a discovered update should be staged to disk.
///
/// **The single rule, because there are two callers and they had diverged.**
/// The background check loop and `POST /api/admin/update/check` both decide
/// this; the manual endpoint used to decide it from the legacy
/// `updates.auto_update` field while the loop used `effective_mode()`, and its
/// comment said it mirrored the loop. It did not.
///
/// The consequence was that the modern setting did nothing on that path: a user
/// who set `mode = "download"` still has `auto_update` at its `Disabled`
/// default, so pressing "check for updates" in the dashboard refused to stage
/// anything while the background loop happily staged the same release. Two
/// answers to one question, and the one the user triggered was the wrong one.
///
/// A managed install (deb/rpm, hardened anchor) can never replace its own
/// binary, so it does not download at all — otherwise it re-fetches the release
/// on every check, forever.
pub(crate) fn should_stage_download(mode: crate::config::UpdateMode, info: &UpdateInfo) -> bool {
    mode >= crate::config::UpdateMode::Download && info.self_update_supported
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
        let binary_path = crate::current_exe_path().unwrap_or_else(|e| {
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
            shared: None,
        }
    }

    /// Attach the node's state so an automatic install can wait for in-flight
    /// work to finish first. Without it, `Install` mode declines to restart
    /// rather than interrupting a request it cannot see.
    pub fn with_shared_state(mut self, shared: Arc<crate::daemon::SharedState>) -> Self {
        self.shared = Some(shared);
        self
    }

    /// Check GitHub for a newer release. Returns `Some(UpdateInfo)` if an update is available.
    pub async fn check_for_update(&self) -> Result<Option<UpdateInfo>, SwarmError> {
        tracing::debug!("DIAG: check_for_update starting");
        let current = env!("CARGO_PKG_VERSION");
        // `/releases/latest` only ever returns the newest STABLE, non-draft
        // release — it silently skips pre-releases. SwarmLLM ships alpha/beta
        // tags AS pre-releases, so we list `/releases` (returned newest-first)
        // and select ourselves; otherwise auto-update would never see an alpha.
        let url = format!(
            "https://api.github.com/repos/{}/releases?per_page=30",
            self.repo
        );

        let resp = UPDATE_CHECK_CLIENT
            .get(&url)
            .send()
            .await
            .map_err(|e| SwarmError::Network(format!("GitHub API request failed: {e}")))?;

        if !resp.status().is_success() {
            let status = resp.status();
            return Err(SwarmError::Network(format!("GitHub API returned {status}")));
        }

        let releases: Vec<GitHubRelease> = resp
            .json()
            .await
            .map_err(|e| SwarmError::Network(format!("Failed to parse releases JSON: {e}")))?;

        // Defaults true: every release this project publishes is tagged
        // `-alpha`, so filtering pre-releases out means finding nothing, ever.
        let include_prereleases = self.config.include_prereleases;
        let release = match select_target_release(&releases, include_prereleases) {
            Some(r) => r,
            None => {
                let mut state = self.state.write().await;
                state.last_checked = Some(chrono::Utc::now().to_rfc3339());
                state.last_error = None;
                return Ok(None);
            }
        };

        let latest_tag = release.tag_name.trim_start_matches('v').to_string();

        tracing::debug!(current, latest = %latest_tag, "DIAG: check_for_update version compare");

        if !is_newer_version(current, &latest_tag) {
            // Update last_checked even when no update is found
            let mut state = self.state.write().await;
            state.last_checked = Some(chrono::Utc::now().to_rfc3339());
            state.last_error = None;
            return Ok(None);
        }

        // Find the binary asset matching this platform AND build variant.
        //
        // Assets are built for x86-64-v3 (AVX2). A processor older than that
        // is redirected to the `-baseline` asset — see `host_asset_name`.
        let default_asset = update_asset_name();
        let asset_name = host_asset_name(&default_asset);
        if asset_name != default_asset {
            tracing::warn!(
                asset = %asset_name,
                "This processor does not support AVX2, so it needs the baseline build. \
                 The ordinary download is compiled for processors from 2013 onwards and \
                 would not start here. If no baseline build is published for this \
                 platform the update is skipped, which leaves this node on its current \
                 working version"
            );
        }

        // A CPU build on a GPU machine will keep resolving the CPU asset for
        // ever. Say so here, where the user is about to be handed one, rather
        // than leaving them to notice the GPU is idle.
        if let Some(gpu) = cpu_build_on_gpu_host() {
            let cuda_asset = asset_name_for(
                std::env::consts::OS,
                std::env::consts::ARCH,
                "-cuda",
                cfg!(windows),
            );
            tracing::warn!(
                gpu = %gpu,
                running_variant = %asset_name,
                gpu_variant = %cuda_asset,
                "This is the CPU build, but a GPU is present — the update will keep it \
                 on CPU, because updates never switch build variant. To use the GPU, \
                 install the {cuda_asset} asset from the release page once; updates \
                 after that will stay on the GPU build"
            );
        }

        let binary_asset = release.assets.iter().find(|a| a.name == asset_name);

        let download_url = match binary_asset {
            Some(asset) => asset.browser_download_url.clone(),
            None => {
                // Offer nothing rather than fall back to a same-platform asset
                // of a different variant — that path silently swaps a GPU
                // build for a CPU one. Releases before the variant-suffixed
                // assets existed legitimately land here on GPU builds.
                tracing::warn!(
                    expected = %asset_name,
                    available = ?release.assets.iter().map(|a| &a.name).collect::<Vec<_>>(),
                    "No matching binary asset for this platform and build variant — \
                     skipping update rather than installing a different variant"
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
            changelog: release.body.clone().unwrap_or_default(),
            published_at: release.published_at.clone().unwrap_or_default(),
            checksum_sha256,
            downloaded: false,
            self_update_supported: self.can_self_update().await,
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

    /// Can this installation replace its own binary?
    ///
    /// Probes by creating and removing the staging file, because the question
    /// is not "am I root" but "is this exact path writable by this exact
    /// process" — under `ProtectSystem=strict` even root cannot write
    /// `/usr/bin`, and a `.deb` service runs as `swarmllm` besides. Every
    /// managed install answers false: packages update through apt/dnf, and the
    /// hardened anchor updates through the root-run `swarmllm-update.timer`.
    ///
    /// Checked BEFORE downloading rather than after, so a node that can never
    /// apply an update does not fetch ~1 GB every hour to stage a file it will
    /// then refuse to use.
    pub async fn can_self_update(&self) -> bool {
        let probe = self.binary_path.with_extension("update.probe");
        match tokio::fs::File::create(&probe).await {
            Ok(_) => {
                let _ = tokio::fs::remove_file(&probe).await;
                true
            }
            Err(_) => false,
        }
    }

    /// Build a checker with an explicit binary path, for tests that need to
    /// control where the staging file lives.
    #[cfg(test)]
    fn for_test(binary_path: PathBuf) -> Self {
        let (tx, _rx) = tokio::sync::broadcast::channel(4);
        Self {
            config: UpdateConfig::default(),
            repo: "example/repo".to_string(),
            binary_path,
            state: Arc::new(RwLock::new(UpdateState::default())),
            dashboard_tx: tx,
            shared: None,
        }
    }

    pub async fn download_update(&self, info: &UpdateInfo) -> Result<PathBuf, SwarmError> {
        // A verified binary for this exact release may already be staged beside
        // the running one — reuse it rather than fetching ~1 GB again.
        //
        // Nothing in the periodic loop remembers what is already on disk, so
        // without this a node in `download` mode re-downloaded the SAME release
        // on every check (hourly by default) for as long as the update went
        // unapplied. `download` is precisely the mode that leaves an update
        // unapplied indefinitely — it stages and stops by design, pending
        // binary signing — so this repeats for that mode rather than being an
        // edge case.
        //
        // Decided by hashing the file, never by its name, size or mtime: a
        // truncated or superseded staging file must not be mistaken for a good
        // one, and a release whose sidecar checksum is missing is not
        // reusable at all (the download path rejects those outright).
        //
        // This must stay ABOVE the writability probe below, which uses
        // `File::create` and therefore TRUNCATES an existing staging file
        // before anything has had a chance to look at it.
        if let Some(expected) = info.checksum_sha256.as_deref() {
            let staged = self.preferred_tmp_path();
            if staged_file_matches(&staged, expected).await {
                tracing::info!(
                    version = %info.latest_version,
                    path = %staged.display(),
                    "Update for this release is already staged and verified — reusing it \
                     rather than downloading again"
                );
                return Ok(staged);
            }
        }

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

        // GitHub's release-asset CDN can return a transient 504 (or 502/503/429)
        // while it warms up on a freshly-uploaded binary — the minutes right
        // after a release is cut. Retry a few times on transient failures so an
        // auto-update doesn't fail just because it ran seconds after the release.
        const DOWNLOAD_RETRIES: u32 = 5;
        const DOWNLOAD_RETRY_DELAY: std::time::Duration = std::time::Duration::from_secs(15);
        let mut attempt = 0u32;
        let resp = loop {
            attempt += 1;
            match client.get(&info.download_url).send().await {
                Ok(r) if r.status().is_success() => break r,
                Ok(r) => {
                    let status = r.status();
                    let transient = status.is_server_error() || status.as_u16() == 429;
                    if transient && attempt < DOWNLOAD_RETRIES {
                        tracing::warn!(%status, attempt, "update download got a transient status — retrying");
                        tokio::time::sleep(DOWNLOAD_RETRY_DELAY).await;
                        continue;
                    }
                    return Err(SwarmError::Network(format!(
                        "Download failed with status {status}"
                    )));
                }
                Err(e) => {
                    if attempt < DOWNLOAD_RETRIES {
                        tracing::warn!(error = %e, attempt, "update download request failed — retrying");
                        tokio::time::sleep(DOWNLOAD_RETRY_DELAY).await;
                        continue;
                    }
                    return Err(SwarmError::Network(format!("Download request failed: {e}")));
                }
            }
        };

        // Check Content-Length to reject absurdly large downloads before buffering.
        // The CUDA/GPU release binary bundles candle + llama.cpp CUDA kernels and
        // is ~1 GB (978 MB as of v0.3.19); the old 500 MB cap silently broke
        // `swarmllm update` for every CUDA user (external report 2026-07-25).
        // 2 GB leaves headroom for further kernel/arch growth while still
        // rejecting a runaway download.
        const MAX_UPDATE_SIZE: u64 = 2 * 1024 * 1024 * 1024; // 2 GB
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
                if !checksum_matches(expected_hash, &actual_hash) {
                    let _ = tokio::fs::remove_file(&tmp_path).await;
                    return Err(SwarmError::ShardIntegrity {
                        expected: sidecar_hash(expected_hash).to_string(),
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
        self.apply_update_with_version(tmp_path, latest_version, expected_checksum_sha256)
    }

    fn apply_update_with_version(
        &self,
        tmp_path: &std::path::Path,
        latest_version: &str,
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
        let current = env!("CARGO_PKG_VERSION");
        if !is_newer_version(current, latest_version) {
            return Err(SwarmError::Validation(format!(
                "Refusing to apply update: target version {latest_version} is not newer than running {current}"
            )));
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
            let expected_trimmed = sidecar_hash(expected);
            if !checksum_matches(expected, &actual) {
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
            Err(SwarmError::Validation(
                "Auto-update apply is not supported on Windows. Download the new version manually and replace the binary after stopping the daemon.".to_string(),
            ))
        }

        #[cfg(not(target_os = "windows"))]
        {
            let backup_path = self.binary_path.with_extension("old");

            // Keep a rollback copy WITHOUT moving the original out of the way,
            // then replace the binary in a single atomic step. See
            // `preserve_current_binary` for why the order matters.
            if self.binary_path.exists() {
                preserve_current_binary(&self.binary_path, &backup_path).map_err(|e| {
                    SwarmError::ServiceUnavailable(format!("Failed to back up current binary: {e}"))
                })?;
            }

            if let Err(e) = swap_binary_into_place(tmp_path, &self.binary_path) {
                // Nothing to roll back: the binary was never moved. It is still
                // the old version, which is exactly what a failed update should
                // leave behind.
                return Err(SwarmError::ServiceUnavailable(format!(
                    "Failed to install update (binary left untouched): {e}"
                )));
            }

            tracing::info!(
                new = %self.binary_path.display(),
                backup = %backup_path.display(),
                version = %latest_version,
                "Update applied — restart required to use new version"
            );
            // Remember what we installed. If the restart does not take effect,
            // the next start compares this against the version compiled into
            // the running image and says so, instead of leaving an operator to
            // deduce it from process ids.
            if let Some(shared) = self.shared.as_ref() {
                if let Err(e) =
                    shared
                        .db
                        .put_json(TREE_UPDATE, KEY_INSTALLED_VERSION, &latest_version)
                {
                    tracing::warn!(error = %e, "Could not record the installed version");
                }
            }

            Ok(())
        }
    }

    /// Wait for the node to go quiet, swap the binary, and restart into it.
    ///
    /// Only reached in `Install` mode. Everything here is best-effort: a node
    /// that cannot install must carry on serving on the old version, never fall
    /// over. Each failure therefore logs and returns, leaving the dashboard
    /// banner in place so a human can act.
    async fn install_and_restart(&self, info: &UpdateInfo) {
        let Some(shared) = self.shared.as_ref() else {
            tracing::warn!("Automatic install skipped — no node state to drain against");
            return;
        };

        shared.emit_activity(
            crate::daemon::state::ActivityEvent::new(
                "update",
                "installing",
                format!(
                    "Installing v{} — finishing current work first",
                    info.latest_version
                ),
            )
            .with_toast("info", 8000),
        );

        // Waiting out in-flight work is the whole point: replacing the binary
        // under a running request is what made an "updated" node fail every
        // inference until someone restarted it by hand.
        let idle = crate::update_restart::drain(shared).await;
        tracing::info!(
            drained_cleanly = idle,
            version = %info.latest_version,
            "Applying update"
        );

        let staged = self.preferred_tmp_path();
        if let Err(e) = self.apply_update(
            &staged,
            &info.latest_version,
            info.checksum_sha256.as_deref(),
        ) {
            tracing::error!(error = %e, "Update apply failed — staying on the current version");
            shared.emit_activity(
                crate::daemon::state::ActivityEvent::new(
                    "update",
                    "install_failed",
                    format!("Could not install v{}: {e}", info.latest_version),
                )
                .with_toast("error", 10000),
            );
            return;
        }

        let err = crate::update_restart::exec_into(&self.binary_path);
        // exec only returns on failure. The binary IS updated at this point, so
        // the node keeps running the old image until something restarts it —
        // say so loudly rather than leaving it looking healthy (gotcha #188).
        tracing::error!(
            error = %err,
            "Update installed but restarting into it failed — restart this node manually"
        );
    }

    /// Compare the last recorded install against the running image, and say so
    /// loudly if a newer version is sitting on disk unused.
    async fn report_pending_restart(&self) {
        let Some(shared) = self.shared.as_ref() else {
            return;
        };
        let installed: Option<String> = shared
            .db
            .get_json::<String>(TREE_UPDATE, KEY_INSTALLED_VERSION)
            .ok()
            .flatten();
        let running = env!("CARGO_PKG_VERSION");
        let Some(pending) = restart_required_for(running, installed.as_deref()) else {
            return;
        };

        tracing::warn!(
            running = %pending.running,
            installed = %pending.installed,
            "A newer version is installed on disk than the one running — the restart into it \
             did not take effect. Restart this node to pick up v{}. (Process id and start time \
             are NOT evidence either way: an in-place update keeps both.)",
            pending.installed
        );
        shared.emit_activity(
            crate::daemon::state::ActivityEvent::new(
                "update",
                "restart_required",
                format!(
                    "v{} is installed but this node is still running v{} — restart to use it",
                    pending.installed, pending.running
                ),
            )
            .with_toast("warning", 15000),
        );
        self.state.write().await.restart_required = Some(pending);
    }

    /// Background update loop — checks periodically and stores results in shared state.
    pub async fn run(&self, mut shutdown_rx: watch::Receiver<bool>) {
        use crate::config::UpdateMode;

        // Did the last install actually take effect? The updater `exec`s into
        // the new binary, which keeps the process id and the kernel's start
        // time, so `ps` cannot tell "updated in place" from "never restarted"
        // (gotcha #277) — and an operator reported twice that they believed
        // their node had missed eight releases on exactly that evidence. Only
        // the node can answer it: it knows what it installed and what it is
        // running. Say so at every start rather than leaving it to forensics.
        self.report_pending_restart().await;

        let mode = self.config.effective_mode();
        // Say the resolved mode out loud, every start.
        //
        // An operator reported on 2026-08-09 that their node installed two
        // versions while they believed it was set to notify. Whether that was a
        // mistyped key, a `[update]` section that does not exist, or something
        // else could not be established from the outside, because nothing ever
        // stated what the node had actually concluded. The setting is resolved
        // from two fields (`mode`, falling back to the legacy `auto_update`),
        // which is exactly the kind of derivation worth printing rather than
        // leaving someone to infer from behaviour.
        tracing::info!(
            ?mode,
            configured_mode = ?self.config.mode,
            legacy_auto_update = ?self.config.auto_update,
            "Update mode resolved — 'install' is the only value that installs by itself"
        );
        if mode == UpdateMode::Off {
            tracing::info!("Update checking disabled (updates.mode = \"off\")");
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

                    // Download from `Download` up; `Notify` only surfaces it.
                    // A managed install (package / hardened anchor) can never
                    // apply what it downloads, so it does not download at all —
                    // otherwise it re-fetches the release every check forever.
                    let mut info = info;
                    let should_download = should_stage_download(mode, &info);
                    if mode >= UpdateMode::Download && !info.self_update_supported {
                        tracing::info!(
                            latest = %info.latest_version,
                            "Update available, but this installation cannot replace its own \
                             binary — update via your package manager (deb/rpm) or, on an \
                             anchor, swarmllm-update.timer"
                        );
                    }
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
                    let _ = self.dashboard_tx.send(
                        crate::daemon::state::DashboardSignal::UpdateAvailable(info.clone()),
                    );

                    // Install mode finishes the job. Announce first (above) so
                    // the dashboard shows what is happening before the node
                    // goes away for a few seconds.
                    if mode == UpdateMode::Install && info.downloaded {
                        self.install_and_restart(&info).await;
                    }
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

/// Keep a rollback copy of the running binary at `backup`, **without removing
/// the original**.
///
/// The distinction is the whole point. The previous implementation *renamed*
/// the binary to `.old` and then renamed the download into place, which leaves
/// a window — however short — where the path systemd invokes does not exist. If
/// the machine dies in that window the service is bricked: systemd reports
/// `status=203/EXEC` in a restart loop with no binary on disk.
///
/// **That is not hypothetical.** Reported on 0.3.57 → 0.3.58 (Debian 13 LXC,
/// `mode = "install"`): the host became unresponsive mid-swap, leaving a valid
/// `swarmllm.old` (0.3.57) and a valid `swarmllm.update.tmp` (0.3.58) and
/// nothing at `swarmllm`. It took **two days** to surface, because the running
/// process kept serving from its open inode and reported the new version over
/// the API the entire time — so the failure appeared long after the update that
/// caused it, pointing at nothing.
///
/// Hard-links first: the release binary is ~1 GB, and a link is instant and
/// costs no space while still pinning the old inode as a rollback target. Falls
/// back to a copy on filesystems that refuse links.
#[cfg(not(target_os = "windows"))]
fn preserve_current_binary(
    binary: &std::path::Path,
    backup: &std::path::Path,
) -> std::io::Result<()> {
    // `hard_link` fails if the destination exists, and a previous update will
    // have left one.
    if backup.exists() {
        std::fs::remove_file(backup)?;
    }
    match std::fs::hard_link(binary, backup) {
        Ok(()) => Ok(()),
        Err(_) => std::fs::copy(binary, backup).map(|_| ()),
    }
}

/// Replace `binary` with `staged` in one atomic step.
///
/// `rename(2)` over an existing destination is atomic: any observer sees either
/// the old file or the new one, never nothing. That is what keeps the service
/// startable if the machine dies mid-update.
///
/// Permissions are set on the staged file BEFORE the rename, so the destination
/// is never briefly non-executable either. `download_update` already chmods what
/// it writes; this covers a file staged by an older build or restored by hand.
#[cfg(not(target_os = "windows"))]
fn swap_binary_into_place(
    staged: &std::path::Path,
    binary: &std::path::Path,
) -> std::io::Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(staged, std::fs::Permissions::from_mode(0o755))?;
    }
    std::fs::rename(staged, binary)
}

/// Extract the bare hash from a `.sha256` sidecar body.
///
/// Sidecars are written as `"<hash>  <filename>"` by both `shasum -a 256`
/// (Linux/macOS) and the PowerShell step in `release.yml` (Windows), so take
/// only the first whitespace-delimited token.
fn sidecar_hash(sidecar_body: &str) -> &str {
    sidecar_body
        .split_whitespace()
        .next()
        .unwrap_or(sidecar_body)
}

/// Whether a freshly computed hex digest matches a `.sha256` sidecar body.
///
/// Single source of truth for the checksum contract, shared by the download
/// path and the pre-rename re-verification. Two things it must get right:
///
/// - **Sidecar format**: `"<hash>  <filename>"`, not a bare hash.
/// - **Case**: hex is case-insensitive, and the sidecars are produced by a
///   different tool per platform (`shasum` lowercases; PowerShell's
///   `Get-FileHash` uppercases and `release.yml` re-lowercases it). Comparing
///   case-sensitively would make every Windows update fail with a
///   tamper-looking integrity error the day that `.ToLower()` is dropped.
fn checksum_matches(sidecar_body: &str, actual_hash: &str) -> bool {
    actual_hash.eq_ignore_ascii_case(sidecar_hash(sidecar_body))
}

/// Whether an already-staged file is exactly the binary `expected` describes.
///
/// Streams the file rather than reading it whole: the CUDA release binary is
/// ~1 GB, and this runs on a node that may be busy serving. Any read failure —
/// no such file, unreadable, a directory — answers `false`, which is the safe
/// direction: the caller then downloads, rather than trusting something it
/// could not check.
async fn staged_file_matches(staged: &std::path::Path, expected: &str) -> bool {
    use sha2::{Digest, Sha256};
    use tokio::io::AsyncReadExt;

    let Ok(mut file) = tokio::fs::File::open(staged).await else {
        return false;
    };
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1024 * 1024];
    loop {
        match file.read(&mut buf).await {
            Ok(0) => break,
            Ok(n) => hasher.update(&buf[..n]),
            Err(_) => return false,
        }
    }
    checksum_matches(expected, &hex::encode(hasher.finalize()))
}

/// Build-variant suffix for release asset names (`""` for a CPU build).
///
/// The OS/arch pair alone does not identify a release binary: a GPU build and a
/// CPU build share both. Without this, a CUDA node matched
/// `swarmllm-linux-x86_64` — the *CPU* asset — and updated itself into a
/// GPU-less binary, silently losing the only capability that made the machine
/// worth running. The variant is a compile-time property of the running
/// binary, so read it from cfg rather than probing the host for a GPU: what
/// matters is which artefact this binary was built from, not what hardware it
/// happens to be sitting on.
///
/// Keep in sync with the `bare_asset` names in `.github/workflows/release.yml`.
/// `candle-cuda` is included here, not just `cuda`. The published Linux GPU
/// asset is built `--features cuda`, which pulls in `candle-cuda` — but the
/// documented local GPU build is `--features candle-cuda` alone, and the split
/// engine (which is what actually runs inference) is GPU-capable with just
/// that. Keying only on `cuda` classified those builds as CPU, so they resolved
/// the CPU asset and updated their own GPU support away.
fn build_variant_suffix() -> &'static str {
    if cfg!(any(feature = "cuda", feature = "candle-cuda")) {
        "-cuda"
    } else if cfg!(feature = "windows-gpu") {
        "-gpu"
    } else {
        ""
    }
}

/// Is this a CPU-variant build sitting on a machine that has a usable GPU?
///
/// The variant is a compile-time property, deliberately: what may be installed
/// is decided by which artefact is running, not by what hardware it happens to
/// find. That is right, but it makes the CPU variant a **one-way trapdoor** —
/// a node that lands on the CPU binary once will resolve the CPU asset forever
/// after, because the running binary is the thing being asked. It keeps
/// updating "successfully" while the GPU sits idle, and nothing says so.
///
/// Reported 2026-07-29 by a tester who had been manually reinstalling the
/// `-cuda` asset after every update and could find nothing in the UI or logs
/// explaining why it kept reverting.
///
/// So: detect the mismatch and say it out loud. We do NOT silently switch
/// variants — a GPU being present is not proof its driver stack can run the
/// CUDA build, and installing an unusable binary is worse than a slow one.
pub fn cpu_build_on_gpu_host() -> Option<String> {
    if !build_variant_suffix().is_empty() {
        return None;
    }
    let (name, _vram) = crate::model::auto_manage::vram::detect_gpu_nvidia_smi();
    name
}

/// Compose a bare-binary release asset name. Pure so every published variant
/// can be pinned by tests, not just the one this binary happens to be.
fn asset_name_for(os: &str, arch: &str, variant: &str, windows: bool) -> String {
    let ext = if windows { ".exe" } else { "" };
    format!("swarmllm-{os}-{arch}{variant}{ext}")
}

/// Does this processor support AVX2?
///
/// A genuine runtime check of the CPU, unlike the GPU question — there is no
/// equivalent of "the hardware is present but its driver stack cannot use it".
/// If the instruction set is reported, a binary compiled for it will run.
fn host_has_avx2() -> bool {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        std::arch::is_x86_feature_detected!("avx2")
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        false
    }
}

/// Redirect a pre-AVX2 processor to the `-baseline` asset.
///
/// **Every x86-64 asset is built for the `x86-64-v3` baseline**, which requires
/// AVX2 (Intel Haswell 2013, AMD Excavator 2015). That is what makes the default
/// download fast: candle gates its hand-written quantized kernels on
/// `#[cfg(target_feature = "avx2")]`, so at the old default target they were
/// compiled out and every processor ran a scalar fallback — measured 3.09x
/// slower. Upstream candle knows (huggingface/candle#1818) and has not fixed it.
///
/// A processor older than that cannot execute those binaries at all, so it gets
/// a `-baseline` asset built with no raised target.
///
/// **When no baseline asset exists the returned name will not resolve and the
/// caller skips the update.** That is deliberate and is the safe direction:
/// staying on a working older binary beats installing one that dies on its first
/// instruction, which for a node that updates itself is unrecoverable (gotcha
/// #246 — a node was once left with no working binary at all). It is why this
/// asks `is_x86_feature_detected!` rather than inferring anything.
///
/// GPU builds get the same treatment. No `-cuda-baseline` is published — a
/// pre-2013 processor with a modern GPU is vanishingly rare — so such a host
/// stops updating and says so, rather than being handed something broken.
fn host_asset_name(default_name: &str) -> String {
    if host_has_avx2() {
        return default_name.to_string();
    }
    baseline_asset_name(default_name)
}

/// The `-baseline` sibling of an asset name.
///
/// Split out from [`host_asset_name`] so it can be tested on any machine: the
/// redirect only fires on a processor without AVX2, which is exactly the
/// hardware a developer or CI runner is least likely to have — a test routed
/// through the host check silently asserts nothing on almost every machine that
/// runs it. This is the safety-critical half: a wrong name here means an old
/// processor asks for an asset nobody publishes and stops updating.
fn baseline_asset_name(default_name: &str) -> String {
    // Insert before any extension: `...x86_64.exe` -> `...x86_64-baseline.exe`.
    match default_name.rsplit_once('.') {
        Some((stem, ext)) => format!("{stem}-baseline.{ext}"),
        None => format!("{default_name}-baseline"),
    }
}

/// Name of the bare-binary release asset this build should update itself from.
fn update_asset_name() -> String {
    let (os, arch) = platform_strings();
    asset_name_for(
        os,
        arch,
        build_variant_suffix(),
        cfg!(target_os = "windows"),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rel(tag: &str, prerelease: bool, draft: bool) -> GitHubRelease {
        GitHubRelease {
            tag_name: tag.to_string(),
            body: None,
            published_at: None,
            assets: vec![],
            prerelease,
            draft,
        }
    }

    #[test]
    fn select_target_release_respects_mode() {
        // Newest-first, as GitHub returns them: a draft alpha, a published
        // alpha, then a stable release.
        let releases = vec![
            rel("v0.4.0-alpha", true, true),  // draft — never selected
            rel("v0.3.0-alpha", true, false), // published pre-release
            rel("v0.2.0", false, false),      // stable
        ];

        // Stable/Disabled (include_prereleases = false) → newest *stable*.
        assert_eq!(
            select_target_release(&releases, false).unwrap().tag_name,
            "v0.2.0"
        );
        // All (include_prereleases = true) → newest non-draft, incl. pre-release.
        assert_eq!(
            select_target_release(&releases, true).unwrap().tag_name,
            "v0.3.0-alpha"
        );
    }

    #[test]
    fn select_target_release_none_when_only_prereleases_in_stable() {
        // A fresh repo whose only published releases are alphas: Stable mode
        // finds nothing (so auto-update stays quiet instead of downgrading).
        let releases = vec![
            rel("v0.3.0-alpha", true, false),
            rel("v0.2.0-alpha", true, false),
        ];
        assert!(select_target_release(&releases, false).is_none());
        assert_eq!(
            select_target_release(&releases, true).unwrap().tag_name,
            "v0.3.0-alpha"
        );
    }

    #[test]
    fn select_target_release_skips_drafts() {
        let releases = vec![rel("v0.3.0", false, true), rel("v0.2.0", false, false)];
        // Even in "include prereleases" mode, drafts are skipped.
        assert_eq!(
            select_target_release(&releases, true).unwrap().tag_name,
            "v0.2.0"
        );
    }

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

    /// Pins the bare-binary asset names against
    /// `.github/workflows/release.yml`'s `bare_asset` matrix fields. If a name
    /// changes on either side without the other, auto-update silently stops
    /// finding a binary — the failure is a quiet "no update available", not an
    /// error, so nothing else would catch the drift.
    #[test]
    fn asset_names_match_the_published_release_matrix() {
        for (os, arch, variant, windows, expected) in [
            ("linux", "x86_64", "", false, "swarmllm-linux-x86_64"),
            (
                "linux",
                "x86_64",
                "-cuda",
                false,
                "swarmllm-linux-x86_64-cuda",
            ),
            ("macos", "aarch64", "", false, "swarmllm-macos-aarch64"),
            ("windows", "x86_64", "", true, "swarmllm-windows-x86_64.exe"),
            (
                "windows",
                "x86_64",
                "-gpu",
                true,
                "swarmllm-windows-x86_64-gpu.exe",
            ),
        ] {
            assert_eq!(asset_name_for(os, arch, variant, windows), expected);
        }
    }

    /// Sidecars ship as `"<hash>  <filename>"`, so a bare-hash assumption
    /// would compare the digest against the whole line and reject every
    /// update. Body text is the real v0.3.20-alpha Linux CUDA sidecar.
    #[test]
    fn sidecar_hash_ignores_the_trailing_filename() {
        let real = "284cf42ebb952c76d41c083de26bf247fe919ca39e727593b198ee892da987bd  swarmllm-linux-x86_64-cuda";
        assert_eq!(
            sidecar_hash(real),
            "284cf42ebb952c76d41c083de26bf247fe919ca39e727593b198ee892da987bd"
        );
        // A bare hash with no filename must survive unchanged.
        let bare = "284cf42ebb952c76d41c083de26bf247fe919ca39e727593b198ee892da987bd";
        assert_eq!(sidecar_hash(bare), bare);
        // Trailing newline is normal for `shasum > file`.
        assert_eq!(sidecar_hash("abc123  name\n"), "abc123");
    }

    /// Hex is case-insensitive. `shasum` lowercases and PowerShell's
    /// `Get-FileHash` uppercases (release.yml re-lowercases it) — a
    /// case-sensitive compare would fail every Windows update with a
    /// tamper-looking integrity error if that `.ToLower()` were dropped.
    #[test]
    fn checksum_comparison_is_case_insensitive_both_ways() {
        let lower = "284cf42ebb952c76d41c083de26bf247fe919ca39e727593b198ee892da987bd";
        let upper = lower.to_ascii_uppercase();

        // Sidecar and computed digest, in every case combination.
        assert!(checksum_matches(lower, lower));
        assert!(checksum_matches(&upper, lower));
        assert!(checksum_matches(lower, &upper));
        // …and with the filename suffix the real sidecars carry.
        assert!(checksum_matches(
            &format!("{upper}  swarmllm-windows-x86_64-gpu.exe"),
            lower
        ));
    }

    /// The security property that matters more than any of the above: a
    /// genuinely different binary must still be rejected.
    #[test]
    fn checksum_mismatch_is_still_rejected() {
        let expected =
            "284cf42ebb952c76d41c083de26bf247fe919ca39e727593b198ee892da987bd  swarmllm-linux-x86_64-cuda";
        // One nibble different.
        let tampered = "284cf42ebb952c76d41c083de26bf247fe919ca39e727593b198ee892da987be";
        assert!(!checksum_matches(expected, tampered));
        // Truncated digest must not pass as a prefix match.
        assert!(!checksum_matches(expected, "284cf42e"));
        assert!(!checksum_matches(expected, ""));
    }

    /// A GPU build must never resolve to the CPU asset: same os/arch, no GPU.
    #[test]
    fn gpu_variants_do_not_collide_with_the_cpu_asset() {
        let cpu = asset_name_for("linux", "x86_64", "", false);
        let cuda = asset_name_for("linux", "x86_64", "-cuda", false);
        assert_ne!(cpu, cuda);

        let win_cpu = asset_name_for("windows", "x86_64", "", true);
        let win_gpu = asset_name_for("windows", "x86_64", "-gpu", true);
        assert_ne!(win_cpu, win_gpu);
    }

    #[test]
    fn build_variant_matches_this_builds_features() {
        let variant = build_variant_suffix();
        // Exactly one of the three known variants, and it must agree with the
        // features this test binary was compiled with.
        //
        // `candle-cuda` counts as a GPU build, not just `cuda`. The published
        // Linux GPU asset is built `--features cuda` (which implies
        // candle-cuda), but the documented LOCAL GPU build is
        // `--features candle-cuda` alone and is genuinely GPU-capable through
        // the split engine. Classifying those as CPU made them resolve the CPU
        // asset and update their own GPU support away.
        if cfg!(any(feature = "cuda", feature = "candle-cuda")) {
            assert_eq!(variant, "-cuda");
        } else if cfg!(feature = "windows-gpu") {
            assert_eq!(variant, "-gpu");
        } else {
            assert_eq!(variant, "");
        }
        // The composed name always starts with the product prefix.
        assert!(update_asset_name().starts_with("swarmllm-"));
    }

    /// The variant is compile-time on purpose, so a CPU build never installs
    /// itself a GPU binary it may not be able to run. The cost is that landing
    /// on the CPU asset is a one-way trapdoor, so the mismatch has to be
    /// *reported* — a tester spent weeks manually reinstalling the `-cuda`
    /// asset with nothing in the logs explaining why it kept reverting.
    #[test]
    fn cpu_build_on_gpu_host_only_fires_for_cpu_builds() {
        if build_variant_suffix().is_empty() {
            // CPU build: the answer depends on whether this machine has a GPU,
            // so only the shape is assertable here.
            let _ = cpu_build_on_gpu_host();
        } else {
            assert!(
                cpu_build_on_gpu_host().is_none(),
                "a GPU build must never report itself as a CPU build on a GPU host"
            );
        }
    }

    #[test]
    fn update_config_defaults() {
        let config = UpdateConfig::default();
        // Automatic INSTALLING stays opt-in — these are unsigned binaries
        // verified only by a published SHA256 (deferred item C1, minisign).
        assert_eq!(config.auto_update, crate::config::AutoUpdateMode::Disabled);
        // ...but a fresh install must at least find out an update exists, and
        // often enough to matter when several releases can ship in one day.
        assert_eq!(config.effective_mode(), crate::config::UpdateMode::Notify);
        assert!(config.check_interval_hours <= 1);
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
            self_update_supported: true,
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

#[cfg(test)]
mod staged_reuse_tests {
    use super::*;

    fn sha256_hex(bytes: &[u8]) -> String {
        use sha2::{Digest, Sha256};
        let mut h = Sha256::new();
        h.update(bytes);
        hex::encode(h.finalize())
    }

    fn info_for(checksum: Option<String>) -> UpdateInfo {
        UpdateInfo {
            latest_version: "0.9.9".to_string(),
            current_version: "0.9.8".to_string(),
            // A syntactically valid GitHub URL that would fail to connect.
            // Reaching it at all is the failure this test detects.
            download_url: "https://github.com/example/repo/releases/download/v0/x".to_string(),
            changelog: String::new(),
            published_at: String::new(),
            checksum_sha256: checksum,
            downloaded: false,
            self_update_supported: true,
        }
    }

    #[tokio::test]
    async fn a_matching_staged_file_is_recognised() {
        let dir = tempfile::tempdir().unwrap();
        let staged = dir.path().join("swarmllm.update.tmp");
        std::fs::write(&staged, b"pretend binary").unwrap();
        assert!(staged_file_matches(&staged, &sha256_hex(b"pretend binary")).await);
    }

    /// Everything that is not a verified match must answer false, because the
    /// caller treats false as "download it". Trusting a stale or truncated
    /// staging file would install the wrong binary.
    #[tokio::test]
    async fn anything_unverified_is_not_reused() {
        let dir = tempfile::tempdir().unwrap();
        let staged = dir.path().join("swarmllm.update.tmp");

        // Missing.
        assert!(!staged_file_matches(&staged, &sha256_hex(b"x")).await);

        // Present but stale — the shape left behind when a newer release
        // supersedes a staged one.
        std::fs::write(&staged, b"an older release").unwrap();
        assert!(!staged_file_matches(&staged, &sha256_hex(b"the new release")).await);

        // Truncated.
        std::fs::write(&staged, b"pretend bin").unwrap();
        assert!(!staged_file_matches(&staged, &sha256_hex(b"pretend binary")).await);

        // A directory in the way.
        let as_dir = dir.path().join("dir.update.tmp");
        std::fs::create_dir(&as_dir).unwrap();
        assert!(!staged_file_matches(&as_dir, &sha256_hex(b"")).await);
    }

    /// The sidecar is `"<hash>  <filename>"` and its case varies by platform,
    /// so reuse has to accept the same spellings the download path does —
    /// otherwise a Windows node re-downloads every hour forever.
    #[tokio::test]
    async fn sidecar_format_and_case_are_accepted() {
        let dir = tempfile::tempdir().unwrap();
        let staged = dir.path().join("swarmllm.update.tmp");
        std::fs::write(&staged, b"pretend binary").unwrap();
        let hash = sha256_hex(b"pretend binary");

        assert!(staged_file_matches(&staged, &format!("{hash}  swarmllm-linux-x86_64")).await);
        assert!(staged_file_matches(&staged, &hash.to_uppercase()).await);
    }

    /// The behaviour that matters: with the release already staged and
    /// verified, `download_update` returns it WITHOUT going to the network.
    ///
    /// Before this, nothing in the periodic loop remembered what was on disk,
    /// so a node in `download` mode re-fetched the same ~1 GB release every
    /// check — hourly by default — for as long as the update stayed unapplied,
    /// which in that mode is indefinitely.
    ///
    /// The URL points at a host this test cannot reach, so if the reuse path
    /// were removed this would attempt a real download and fail rather than
    /// silently passing.
    #[tokio::test]
    async fn an_already_staged_release_is_not_downloaded_again() {
        let dir = tempfile::tempdir().unwrap();
        let binary = dir.path().join("swarmllm");
        std::fs::write(&binary, b"running binary").unwrap();

        let checker = UpdateChecker::for_test(binary.clone());
        let staged = checker.preferred_tmp_path();
        std::fs::write(&staged, b"pretend binary").unwrap();

        let info = info_for(Some(sha256_hex(b"pretend binary")));
        let got = checker.download_update(&info).await.expect("reuses staged");
        assert_eq!(got, staged);
        assert_eq!(
            std::fs::read(&staged).unwrap(),
            b"pretend binary",
            "the staged binary must survive — the writability probe truncates, \
             so the reuse check has to run before it"
        );
    }
}

#[cfg(all(test, not(target_os = "windows")))]
mod atomic_replace_tests {
    use super::{preserve_current_binary, swap_binary_into_place};

    /// **The regression test for the bricked-service report.**
    ///
    /// A node updating on 0.3.57 → 0.3.58 was left with no binary at all: the
    /// host died between "move the old one aside" and "put the new one in
    /// place", and systemd then failed with `203/EXEC` in a restart loop.
    ///
    /// So the property is not about the end state — the old code reached the
    /// same end state — it is that **the canonical path is never absent part
    /// way through**. This asserts exactly that, at the point the reporter's
    /// machine stopped: after the backup exists and before the swap.
    ///
    /// Implementing `preserve_current_binary` as a rename again makes this fail.
    #[test]
    fn the_binary_is_never_absent_part_way_through_an_update() {
        let dir = tempfile::tempdir().unwrap();
        let binary = dir.path().join("swarmllm");
        let backup = dir.path().join("swarmllm.old");
        let staged = dir.path().join("swarmllm.update.tmp");
        std::fs::write(&binary, b"old version 0.3.57").unwrap();
        std::fs::write(&staged, b"new version 0.3.58").unwrap();

        preserve_current_binary(&binary, &backup).unwrap();

        // The crash point. systemd must still find something to exec here.
        assert!(
            binary.exists(),
            "the binary must never be moved aside — a crash here bricks the service"
        );
        assert_eq!(std::fs::read(&binary).unwrap(), b"old version 0.3.57");
        assert!(backup.exists(), "a rollback target must exist by now");

        swap_binary_into_place(&staged, &binary).unwrap();

        assert_eq!(std::fs::read(&binary).unwrap(), b"new version 0.3.58");
        assert_eq!(
            std::fs::read(&backup).unwrap(),
            b"old version 0.3.57",
            "the backup must still hold the OLD build for rollback"
        );
        assert!(
            !staged.exists(),
            "the staged file is consumed by the rename"
        );
    }

    /// A backup left by a previous update must not block this one. `hard_link`
    /// refuses an existing destination, so a stale `.old` would otherwise fail
    /// every update after the first.
    #[test]
    fn a_stale_backup_from_a_previous_update_is_replaced() {
        let dir = tempfile::tempdir().unwrap();
        let binary = dir.path().join("swarmllm");
        let backup = dir.path().join("swarmllm.old");
        std::fs::write(&binary, b"current").unwrap();
        std::fs::write(&backup, b"ancient").unwrap();

        preserve_current_binary(&binary, &backup).unwrap();

        assert_eq!(std::fs::read(&backup).unwrap(), b"current");
        assert!(binary.exists());
    }

    /// If the staged file is missing or unreadable the running binary must be
    /// left exactly as it was — a failed update is not allowed to cost the node
    /// its working build.
    #[test]
    fn a_failed_swap_leaves_the_running_binary_in_place() {
        let dir = tempfile::tempdir().unwrap();
        let binary = dir.path().join("swarmllm");
        let backup = dir.path().join("swarmllm.old");
        std::fs::write(&binary, b"old version").unwrap();

        preserve_current_binary(&binary, &backup).unwrap();
        let missing = dir.path().join("swarmllm.update.tmp");
        assert!(swap_binary_into_place(&missing, &binary).is_err());

        assert!(binary.exists(), "binary must survive a failed update");
        assert_eq!(std::fs::read(&binary).unwrap(), b"old version");
    }

    /// The replacement must be executable the instant it lands, not a moment
    /// later — the destination is what systemd execs.
    #[cfg(unix)]
    #[test]
    fn the_installed_binary_is_executable() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        let binary = dir.path().join("swarmllm");
        let staged = dir.path().join("swarmllm.update.tmp");
        std::fs::write(&binary, b"old").unwrap();
        std::fs::write(&staged, b"new").unwrap();
        std::fs::set_permissions(&staged, std::fs::Permissions::from_mode(0o600)).unwrap();

        swap_binary_into_place(&staged, &binary).unwrap();

        let mode = std::fs::metadata(&binary).unwrap().permissions().mode();
        assert_eq!(mode & 0o777, 0o755, "installed binary must be executable");
    }
}

#[cfg(test)]
mod cpu_baseline_asset_tests {
    use super::*;

    /// The safety-critical half, asserted against the REAL function rather than
    /// a copy of it. The suffix must land before the extension, or Windows asks
    /// for `swarmllm-windows-x86_64.exe-baseline`, which nobody publishes — and
    /// the update is skipped on exactly the machines that most need one that
    /// runs. These are the names `release.yml` publishes.
    #[test]
    fn the_suffix_lands_before_a_file_extension() {
        assert_eq!(
            baseline_asset_name("swarmllm-linux-x86_64"),
            "swarmllm-linux-x86_64-baseline"
        );
        assert_eq!(
            baseline_asset_name("swarmllm-windows-x86_64.exe"),
            "swarmllm-windows-x86_64-baseline.exe"
        );
    }

    /// Every x86-64 asset is built for x86-64-v3, so a processor without AVX2
    /// cannot run any of them and must be sent to `-baseline`.
    #[test]
    fn the_host_check_picks_the_right_side() {
        let picked = host_asset_name("swarmllm-linux-x86_64");
        if host_has_avx2() {
            assert_eq!(picked, "swarmllm-linux-x86_64");
        } else {
            assert_eq!(picked, "swarmllm-linux-x86_64-baseline");
        }
    }

    /// A modern processor must never be diverted to the slow asset — that would
    /// silently undo a measured 3x on every processor-side inference.
    #[test]
    fn an_avx2_processor_is_never_diverted_to_baseline() {
        if !host_has_avx2() {
            return;
        }
        for name in [
            "swarmllm-linux-x86_64",
            "swarmllm-linux-x86_64-cuda",
            "swarmllm-windows-x86_64.exe",
            "swarmllm-windows-x86_64-gpu.exe",
        ] {
            assert_eq!(host_asset_name(name), name);
        }
    }
}

#[cfg(test)]
mod pending_restart_tests {
    use super::restart_required_for;

    /// The reported case: a newer version installed, an older one still running.
    /// Nothing could previously distinguish this from a healthy node, because
    /// an in-place update keeps the process id and start time (gotcha #277) and
    /// `ps` is therefore silent either way.
    #[test]
    fn an_installed_but_unused_version_is_reported() {
        let r = restart_required_for("0.3.81-alpha", Some("0.3.88-alpha"))
            .expect("installed newer than running must be reported");
        assert_eq!(r.running, "0.3.81-alpha");
        assert_eq!(r.installed, "0.3.88-alpha");
    }

    /// The normal case after a successful restart: they agree, say nothing.
    #[test]
    fn agreement_is_silent() {
        assert!(restart_required_for("0.3.89-alpha", Some("0.3.89-alpha")).is_none());
    }

    /// A node that has never installed an update has nothing to compare.
    #[test]
    fn no_record_is_silent() {
        assert!(restart_required_for("0.3.89-alpha", None).is_none());
    }

    /// A deliberate rollback leaves the record ahead of nothing — the running
    /// binary is NEWER than the last recorded install (e.g. installed manually,
    /// or a later release was put in place by hand). Nagging about that would
    /// be telling someone to undo a choice they made.
    #[test]
    fn a_newer_running_version_is_not_a_pending_restart() {
        assert!(restart_required_for("0.3.89-alpha", Some("0.3.85-alpha")).is_none());
    }
}
