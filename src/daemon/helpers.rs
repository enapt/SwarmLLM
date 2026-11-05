//! Small utility functions and constants shared between the daemon's
//! startup, spawn, and supervisor layers.

use crate::config::Config;
use crate::storage::db::Database;

/// Maximum number of times a non-critical subsystem may exit (with Ok or
/// Err) before the supervisor treats it as permanently failed and shuts
/// the daemon down.
///
/// Naming note: nothing actually re-spawns a failed subsystem — each one
/// is launched once at startup. This counter exists so a subsystem that
/// somehow re-enters the JoinSet (e.g. via a future redesign that does
/// re-spawn) won't loop forever. Today the count effectively reaches 1
/// per name in practice.
pub(super) const MAX_NONCRITICAL_FAILURES: u32 = 5;

/// Whether a subsystem is critical to daemon operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubsystemCriticality {
    /// Daemon must shut down if this subsystem permanently fails.
    Critical,
    /// Daemon can continue without this subsystem.
    NonCritical,
}

pub(super) fn resolve_api_key(config: &Config, db: &Database) -> String {
    // 1. Explicit key in config takes priority
    if let Some(ref k) = config.api.api_key {
        if !k.is_empty() {
            tracing::info!(source = "config", "Using API key from configuration");
            return k.clone();
        }
    }

    // 2. Check persisted key in database
    if let Ok(Some(k)) = db.get_json::<String>("config", "api_key") {
        if !k.is_empty() {
            tracing::info!(source = "database", "Using persisted API key from database");
            return k;
        }
    }

    // 3. Generate a new 32-byte hex key
    use rand::RngCore;
    let mut bytes = [0u8; 32];
    rand::rngs::OsRng.fill_bytes(&mut bytes);
    let key = hex::encode(bytes);

    // Persist to DB
    if let Err(e) = db.put_json("config", "api_key", &key) {
        tracing::warn!(error = %e, "Failed to persist API key to database");
    }

    // The file is written by `publish_api_key_file` from the daemon's own
    // startup, NOT here — resolving a key must not write to the user's data
    // directory, or every test that builds a SharedState overwrites the key of
    // a node that happens to be running.

    // Print API key to stderr — visually distinct so first-run users don't
    // miss it. Stderr only (NOT tracing) so it never lands in shipped logs.
    //
    // It must NOT claim the file has been written, because the comment above
    // explains that this function deliberately does not write it. It said
    // "Saved to: <path>" and "Recover anytime: cat <path>" for a file only
    // `publish_api_key_file` creates, moments later and only on the daemon's
    // own startup path. Every test that builds a `SharedState` therefore
    // announced that it had just overwritten the api_key of whatever node the
    // developer had running — which is the exact bug
    // `tests/api_key_side_effects.rs` was written for after it happened twice
    // for real, so the message is indistinguishable from the regression. It
    // cost a real investigation on 2026-08-30. The banner had simply outlived
    // the fix beneath it.
    let key_path = config.node.data_dir.join("api_key");
    eprintln!();
    eprintln!("============================================================");
    eprintln!("  Generated new API key");
    eprintln!("  KEY:        {key}");
    eprintln!("  Stored in:  this node's database");
    eprintln!("  Once the node is running: cat {}", key_path.display());
    eprintln!("============================================================");
    eprintln!();

    key
}

/// Write the API key to a plain file so the CLI can read it while the daemon holds the DB lock.
///
/// SEC: open with mode 0o600 atomically rather than `fs::write` + `set_permissions`.
/// The two-step variant left a TOCTOU window where the file existed with the
/// process-umask-derived permissions (typically 0o644 — world-readable) before
/// the chmod tightened it. Mirrors the identity.key write at keypair.rs:62.
/// Test builds never touch the real data dir.
///
/// `SharedState::new` resolves the API key, and a test building one from
/// `Config::default()` inherits the REAL `data_dir` — `~/.local/share/swarmllm`
/// — even though its database is a tempdir. The key it generated was written
/// over the live node's `api_key` file while that node kept using the one in
/// its own database, so `cargo test` on a machine with a node running silently
/// broke the dashboard, the CLI and every saved token, presenting as an
/// unexplained 401 with nothing in the log (reproduced 2026-07-31).
///
/// **`#[cfg(test)]` was NOT enough.** It applies only while compiling this
/// crate as a test binary — i.e. `cargo test --lib`. An *integration* test in
/// `tests/` links the library compiled WITHOUT `cfg(test)`, so the real writer
/// was still live there and `cargo test` kept clobbering the file (hit again
/// 2026-08-01, after the guard was believed to have fixed it).
///
/// So the write is no longer a side effect of resolving the key at all. Only
/// the daemon's own startup calls [`publish_api_key_file`], and nothing that
/// merely builds a `SharedState` can touch the user's data directory.
pub fn publish_api_key_file(data_dir: &std::path::Path, key: &str) {
    write_api_key_file(data_dir, key)
}

fn write_api_key_file(data_dir: &std::path::Path, key: &str) {
    let path = data_dir.join("api_key");
    #[cfg(unix)]
    {
        use std::io::Write;
        use std::os::unix::fs::OpenOptionsExt;
        let mut opts = std::fs::OpenOptions::new();
        opts.write(true).create(true).truncate(true).mode(0o600);
        match opts.open(&path) {
            Ok(mut f) => {
                if let Err(e) = f.write_all(key.as_bytes()) {
                    tracing::warn!(error = %e, "Failed to write api_key file");
                }
                // SEC: fsync. On a power loss between write_all and the
                // kernel page-cache flush, the file lands empty/partial.
                // If the DB persist also failed (warn-only path in
                // resolve_api_key), the next startup generates a NEW key
                // and the operator's saved tokens stop working — silently.
                // Mirrors identity.key write (keypair.rs:71).
                if let Err(e) = f.sync_all() {
                    tracing::warn!(error = %e, "Failed to fsync api_key file");
                }
            }
            Err(e) => tracing::warn!(error = %e, "Failed to open api_key file"),
        }
    }
    #[cfg(not(unix))]
    {
        if let Err(e) = std::fs::write(&path, key) {
            tracing::warn!(error = %e, "Failed to write api_key file");
        }
    }
}

pub(super) fn map_gguf_architecture(path: &std::path::Path) -> crate::types::ModelArchitecture {
    let arch_str = match crate::inference::split::read_gguf_header(path) {
        Ok(ct) => crate::inference::split::gguf_arch_str(&ct),
        Err(_) => "llama".to_string(),
    };
    crate::model::manifest::gguf_arch_to_model_architecture(&arch_str)
}

/// Try to open a URL in the default browser.
pub(super) fn open_browser(url: &str) -> Result<(), String> {
    #[cfg(target_os = "linux")]
    let cmd = "xdg-open";
    #[cfg(target_os = "macos")]
    let cmd = "open";
    #[cfg(target_os = "windows")]
    {
        // On Windows, use `cmd /C start` for opening URLs
        std::process::Command::new("cmd")
            .args(["/C", "start", "", url])
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
            .map_err(|e| e.to_string())
            .map(|_| ())
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
    return Err("Unsupported platform".into());

    #[cfg(any(target_os = "linux", target_os = "macos"))]
    {
        std::process::Command::new(cmd)
            .arg(url)
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
            .map_err(|e| e.to_string())?;

        Ok(())
    }
}

/// Best-effort IP geolocation: this computer's country as an ISO 3166-1
/// alpha-2 code ("US", "DE"), or `None` when no service answered with one.
///
/// Asks ip-api.com, then Cloudflare's trace page — a second, unrelated
/// service, so one that is blocked on this network, down, or rate-limited on a
/// shared address does not leave the node without a country. It used to be one
/// ip-api.com call at start-up and nothing after a failure (2026-11-05). 5 s
/// each, no key; the caller asks again on a backoff
/// (`background::region_retry_after`).
pub(super) async fn detect_region_from_ip() -> Option<String> {
    let client = crate::http::client_with_timeout(std::time::Duration::from_secs(5));
    if let Some(code) = ask_ip_api(&client).await {
        return Some(code);
    }
    ask_cloudflare_trace(&client).await
}

/// ip-api.com's free endpoint (plain HTTP; 45 requests a minute an address).
async fn ask_ip_api(client: &reqwest::Client) -> Option<String> {
    let resp = client
        .get("http://ip-api.com/json/?fields=status,countryCode")
        .send()
        .await
        .ok()?;
    let json: serde_json::Value = resp.json().await.ok()?;
    country_from_ip_api(&json)
}

/// Cloudflare's trace page (HTTPS, no key): `key=value` lines, the caller's
/// country in `loc`.
async fn ask_cloudflare_trace(client: &reqwest::Client) -> Option<String> {
    let body = client
        .get("https://www.cloudflare.com/cdn-cgi/trace")
        .send()
        .await
        .ok()?
        .text()
        .await
        .ok()?;
    country_from_cloudflare_trace(&body)
}

/// Each answer is read by `api::admin::country_code`, the one reading of a
/// country: two letters, never Cloudflare's `XX` ("not known"; its `T1`, Tor,
/// fails the letters already).
fn country_from_ip_api(json: &serde_json::Value) -> Option<String> {
    if json.get("status")?.as_str()? != "success" {
        return None;
    }
    crate::api::admin::country_code(json.get("countryCode")?.as_str())
}

fn country_from_cloudflare_trace(body: &str) -> Option<String> {
    crate::api::admin::country_code(
        body.lines()
            .find_map(|line| line.trim().strip_prefix("loc=")),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn subsystem_criticality_variants_are_distinct() {
        assert_ne!(
            SubsystemCriticality::Critical,
            SubsystemCriticality::NonCritical
        );
    }

    #[test]
    fn max_noncritical_failures_is_five() {
        assert_eq!(MAX_NONCRITICAL_FAILURES, 5);
    }

    /// Each service's answer is read as the country it names, and nothing
    /// else reaches the capability peers are told.
    #[test]
    fn a_location_service_answer_is_read_as_a_two_letter_country_or_nothing() {
        let ip_api = |v: serde_json::Value| country_from_ip_api(&v);
        assert_eq!(
            ip_api(serde_json::json!({"status": "success", "countryCode": "au"})),
            Some("AU".to_string())
        );
        assert_eq!(
            ip_api(serde_json::json!({"status": "fail", "countryCode": "AU"})),
            None
        );
        assert_eq!(ip_api(serde_json::json!({"status": "success"})), None);

        // Cloudflare's trace page, as it answered on 2026-11-05.
        let trace = "fl=1\nh=www.cloudflare.com\nip=192.0.2.1\nts=1.2\nvisit_scheme=https\n\
                     loc=TH\ntls=TLSv1.3\n";
        assert_eq!(country_from_cloudflare_trace(trace), Some("TH".to_string()));
        assert_eq!(
            country_from_cloudflare_trace("h=x\nloc=XX\n"),
            None,
            "unknown"
        );
        assert_eq!(country_from_cloudflare_trace("h=x\nloc=T1\n"), None, "Tor");
        assert_eq!(country_from_cloudflare_trace("h=x\n"), None);
        assert_eq!(country_from_cloudflare_trace("<html>blocked</html>"), None);
    }
}
