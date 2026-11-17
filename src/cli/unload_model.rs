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
    access: &super::DaemonAccess,
    model: &str,
) -> anyhow::Result<()> {
    let data_dir = access.data_dir.as_path();
    let api_key = access.api_key.clone().unwrap_or_default();
    super::bail_if_no_api_key(&api_key, data_dir)?;

    let url = format!(
        "{}/api/admin/models/{model}/unload",
        super::daemon_base(port)
    );
    let request = super::daemon_client().post(&url);
    let (status, body) = super::admin_call(request, &api_key, data_dir, port).await;

    if !status.is_success() {
        let msg = super::api_error_message(&body);
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
