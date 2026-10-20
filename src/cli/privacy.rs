//! `swarmllm privacy <model>` — make prompt privacy possible for a model.
//!
//! Prompt privacy (`inference.encrypted_pipeline`, "Start and finish on this
//! computer" in the UI) forces the first and last pipeline segments to run
//! locally, so no peer is handed the prompt text or the sampled tokens. Peers
//! still see the hidden states between, which are partly invertible back to
//! text — it is structural, not cryptographic (`docs/ARCHITECTURE.md` §
//! Pipeline Privacy Model). It requires holding the first AND last piece of
//! the model.
//!
//! This asks the daemon to fetch exactly those pieces. It deliberately does not
//! set a flag — privacy turns itself on once both ends are present, so there is
//! no window where a flag is set but the shards have not arrived, which would
//! fail every request for the model until the download finished.

/// Ask the daemon to fetch the first and last shard of `model`.
pub async fn enable_privacy(
    port: u16,
    access: &super::DaemonAccess,
    model: &str,
) -> anyhow::Result<()> {
    let data_dir = access.data_dir.as_path();
    let api_key = access.api_key.clone().unwrap_or_default();
    super::bail_if_no_api_key(&api_key, data_dir)?;

    let client = super::daemon_client();
    let url = format!(
        "{}/api/admin/models/{model}/enable-privacy",
        super::daemon_base(port)
    );
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
        anyhow::bail!("Could not enable prompt privacy for {model}: {msg}");
    }

    match body.get("status").and_then(|s| s.as_str()) {
        Some("already_available") => {
            let on = body
                .get("encrypted_pipeline")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            println!("{model}: both ends are already on this machine.");
            if on {
                println!("Start and finish on this computer is ON — other computers never");
                println!("receive your text or the reply, but the numbers they work on can");
                println!("still reveal much of it.");
            } else {
                println!(
                    "Start and finish on this computer is off for this model because it was turned off for it."
                );
                println!("Turn it back on in the dashboard, or via the per-model setting.");
            }
        }
        Some("downloading") => {
            let shards = body
                .get("needed_shards")
                .and_then(|v| v.as_array())
                .map(|a| {
                    a.iter()
                        .filter_map(|v| v.as_u64())
                        .map(|v| v.to_string())
                        .collect::<Vec<_>>()
                        .join(", ")
                })
                .unwrap_or_default();
            println!("{model}: fetching part(s) {shards}.");
            println!("Start and finish on this computer turns on by itself once they arrive.");
            println!("Progress:  swarmllm status");
        }
        Some("no_download_source") => {
            println!("{model}: the missing part(s) cannot be fetched directly —");
            println!("no HuggingFace source is recorded for this model.");
            println!("They may still arrive from other computers; starting and finishing");
            println!("on this computer turns on by itself once both ends are here.");
        }
        _ => {
            // Unrecognised shape: show it rather than pretending to interpret it.
            println!(
                "{}",
                serde_json::to_string_pretty(&body).unwrap_or_default()
            );
        }
    }

    Ok(())
}
