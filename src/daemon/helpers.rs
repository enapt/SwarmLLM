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
    let key;

    // 1. Explicit key in config takes priority
    if let Some(ref k) = config.api.api_key {
        if !k.is_empty() {
            tracing::info!(source = "config", "Using API key from configuration");
            key = k.clone();
            write_api_key_file(&config.node.data_dir, &key);
            return key;
        }
    }

    // 2. Check persisted key in database
    if let Ok(Some(k)) = db.get_json::<String>("config", "api_key") {
        if !k.is_empty() {
            tracing::info!(source = "database", "Using persisted API key from database");
            write_api_key_file(&config.node.data_dir, &k);
            return k;
        }
    }

    // 3. Generate a new 32-byte hex key
    use rand::RngCore;
    let mut bytes = [0u8; 32];
    rand::rngs::OsRng.fill_bytes(&mut bytes);
    key = hex::encode(bytes);

    // Persist to DB
    if let Err(e) = db.put_json("config", "api_key", &key) {
        tracing::warn!(error = %e, "Failed to persist API key to database");
    }

    // Write to file so CLI `status` can read it without opening the database
    write_api_key_file(&config.node.data_dir, &key);

    // Print API key to stderr — visually distinct so first-run users don't
    // miss it. Stderr only (NOT tracing) so it never lands in shipped logs.
    let key_path = config.node.data_dir.join("api_key");
    eprintln!();
    eprintln!("============================================================");
    eprintln!("  Generated new API key");
    eprintln!("  KEY:        {key}");
    eprintln!("  Saved to:   {}", key_path.display());
    eprintln!("  Recover anytime: cat {}", key_path.display());
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
    let arch_str = match std::fs::File::open(path) {
        Ok(mut f) => match candle_core::quantized::gguf_file::Content::read(&mut f) {
            Ok(ct) => crate::inference::split::gguf_arch_str(&ct),
            Err(_) => "llama".to_string(),
        },
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
        return std::process::Command::new("cmd")
            .args(["/C", "start", "", url])
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
            .map_err(|e| e.to_string())
            .map(|_| ());
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

/// Best-effort IP geolocation using a free API (ip-api.com).
/// Returns an ISO 3166-1 alpha-2 country code (e.g. "US", "DE") or None on failure.
/// Timeout: 5 seconds. No API key required.
pub(super) async fn detect_region_from_ip() -> Option<String> {
    let client = crate::http::client_with_timeout(std::time::Duration::from_secs(5));

    // ip-api.com returns JSON with a "countryCode" field for free, no key needed.
    // Rate limit: 45 requests/min (we only call once at startup).
    let resp = client
        .get("http://ip-api.com/json/?fields=status,countryCode")
        .send()
        .await
        .ok()?;

    let json: serde_json::Value = resp.json().await.ok()?;
    if json.get("status")?.as_str()? == "success" {
        json.get("countryCode")?.as_str().map(|s| s.to_string())
    } else {
        None
    }
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
}
