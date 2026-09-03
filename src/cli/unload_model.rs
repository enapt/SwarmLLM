//! `swarmllm unload <model>` — stop a model's worker and free its memory,
//! keeping the downloaded pieces.
//!
//! The dashboard and the HTTP API could always do this; a terminal could not,
//! and a tester whose processor-only node kept a worker at full load on a
//! request its client had abandoned had `kill -9` as the only tool (gotcha
//! #445). `swarmllm status` now names such a worker; this is the command that
//! retires it. Same endpoint as the dashboard's "unload", so the daemon drains
//! the worker, stops it, releases its memory budget and refreshes every surface
//! together — nothing is left half done the way a hand-killed process is.

/// Ask the daemon to unload `model` from memory (the files stay).
pub async fn unload_model(
    port: u16,
    data_dir: &std::path::Path,
    model: &str,
) -> anyhow::Result<()> {
    let api_key = super::read_api_key(data_dir).unwrap_or_default();
    super::bail_if_no_api_key(&api_key, data_dir)?;

    let client = reqwest::Client::new();
    let url = format!("http://127.0.0.1:{port}/api/admin/models/{model}/unload");
    let resp = match client.post(&url).bearer_auth(&api_key).send().await {
        Ok(r) => r,
        Err(_) => super::exit_daemon_unreachable(port),
    };

    let status = resp.status();
    super::exit_if_api_key_rejected(status, data_dir, port);
    let body: serde_json::Value = resp.json().await.unwrap_or(serde_json::Value::Null);

    if !status.is_success() {
        let msg = body
            .get("error")
            .and_then(|e| e.get("message"))
            .and_then(|m| m.as_str())
            .unwrap_or("request failed");
        if status.as_u16() == 404 {
            anyhow::bail!(
                "No model called {model} on this node. `swarmllm status` lists what is here."
            );
        }
        anyhow::bail!("Could not unload {model}: {msg}");
    }

    let freed_mb = body
        .get("estimated_freed_mb")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    println!("Unloaded {model} — about {freed_mb} MB of memory freed; the files are still here.");
    println!("It loads again on the next request for it.");
    Ok(())
}
