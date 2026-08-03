//! Subcommand implementations for the SwarmLLM CLI.
//!
//! Each submodule owns one top-level subcommand (or family of subcommands).
//! The `main.rs` binary parses args via clap, then dispatches to the public
//! entry point in the matching submodule.

pub mod bench;
pub mod chat;
pub mod get_model;
pub mod peers;
pub mod pool;
pub mod privacy;
pub mod remove_model;
pub mod run;
pub mod split_test;
pub mod status;
pub mod update;

/// Read the API key from the data dir. Shared helper used by CLI commands
/// that talk to a running daemon over HTTP.
pub(crate) fn read_api_key(data_dir: &std::path::Path) -> Option<String> {
    let key_path = data_dir.join("api_key");
    std::fs::read_to_string(key_path)
        .ok()
        .map(|s| s.trim().to_string())
}

/// Bail with the canonical "daemon not running (no API key)" message.
/// Shared by CLI commands that need an API key to call the local HTTP API.
pub(crate) fn bail_if_no_api_key(api_key: &str, data_dir: &std::path::Path) -> anyhow::Result<()> {
    if api_key.is_empty() {
        anyhow::bail!(
            "SwarmLLM is not running (no API key at {}).\n  Start the daemon first: swarmllm run",
            data_dir.join("api_key").display()
        );
    }
    Ok(())
}

/// Print the canonical "daemon unreachable on port N" guidance and
/// `exit(1)`. Shared by CLI commands that print errors instead of bailing.
pub(crate) fn exit_daemon_unreachable(port: u16) -> ! {
    eprintln!("Error: SwarmLLM daemon is not running on port {port}.");
    eprintln!("  Start it: swarmllm run");
    eprintln!("  Or if it's on a different port: --port <N> or set SWARMLLM_NODE_LISTEN_PORT");
    std::process::exit(1);
}

/// Explain a rejected API key and `exit(1)`.
///
/// A 401 from your own daemon is almost never "wrong password" — it is the CLI
/// reading a key from a different place than the daemon wrote it. Two ways that
/// happens, and the raw `authentication_error` names neither:
///
/// - The daemon runs with `-d /some/dir` (needed for a systemd unit, since the
///   default XDG path is wrong there) and the CLI, invoked without `-d`, looks
///   under the default data dir. Reported 2026-07-29.
/// - The key file has diverged from the key the daemon actually accepts, so a
///   file exists and is simply stale.
///
/// Both are fixed by pointing the CLI at the right data dir, so say so.
pub(crate) fn exit_api_key_rejected(data_dir: &std::path::Path, port: u16) -> ! {
    eprintln!("Error: the daemon on port {port} rejected this API key.");
    eprintln!("  Key read from: {}", data_dir.join("api_key").display());
    eprintln!();
    eprintln!("  This usually means the CLI and the daemon disagree about the data directory.");
    eprintln!("  If you started the daemon with -d/--data-dir, pass the SAME one here:");
    eprintln!("      swarmllm -d <dir> <command>");
    eprintln!("  or set SWARMLLM_NODE_DATA_DIR so both agree.");
    eprintln!();
    eprintln!("  If the path above IS correct, the stored key is stale — restart the daemon.");
    std::process::exit(1);
}

/// Whether a daemon response body is an authentication failure.
pub(crate) fn body_is_auth_error(body: &str) -> bool {
    serde_json::from_str::<serde_json::Value>(body)
        .ok()
        .and_then(|v| {
            v.get("error")
                .and_then(|e| e.get("type"))
                .and_then(|t| t.as_str())
                .map(|t| t == "authentication_error")
        })
        .unwrap_or(false)
}

/// Resolve a model id: explicit override wins, otherwise pick the first
/// listing from the daemon's `/v1/models`.
pub(crate) async fn discover_model(
    client: &reqwest::Client,
    base: &str,
    api_key: &str,
    model_override: Option<String>,
) -> anyhow::Result<String> {
    if let Some(m) = model_override {
        return Ok(m);
    }
    let models_resp: serde_json::Value = client
        .get(format!("{base}/v1/models"))
        .header("Authorization", format!("Bearer {api_key}"))
        .send()
        .await?
        .json()
        .await?;
    models_resp["data"][0]["id"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("No models available — load a model first"))
        .map(str::to_string)
}
