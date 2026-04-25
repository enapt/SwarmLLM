//! Subcommand implementations for the SwarmLLM CLI.
//!
//! Each submodule owns one top-level subcommand (or family of subcommands).
//! The `main.rs` binary parses args via clap, then dispatches to the public
//! entry point in the matching submodule.

pub mod bench;
pub mod chat;
pub mod peers;
pub mod pool;
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
