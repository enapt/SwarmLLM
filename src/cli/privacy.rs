//! `swarmllm privacy <model>` — make prompt privacy possible for a model.
//!
//! Prompt privacy (`inference.encrypted_pipeline`) keeps prompts and answers on
//! this machine: it forces the first and last pipeline segments to run locally,
//! so no peer ever sees the prompt text or the sampled tokens. It therefore
//! requires holding the first AND last piece of the model.
//!
//! This asks the daemon to fetch exactly those pieces. It deliberately does not
//! set a flag — privacy turns itself on once both ends are present, so there is
//! no window where a flag is set but the shards have not arrived, which would
//! fail every request for the model until the download finished.

/// Ask the daemon to fetch the first and last shard of `model`.
pub async fn enable_privacy(
    port: u16,
    data_dir: &std::path::Path,
    model: &str,
) -> anyhow::Result<()> {
    let api_key = super::read_api_key(data_dir).unwrap_or_default();
    super::bail_if_no_api_key(&api_key, data_dir)?;

    let client = reqwest::Client::new();
    let url = format!("http://127.0.0.1:{port}/api/admin/models/{model}/enable-privacy");
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
                println!("Prompt privacy is ON — prompts and answers stay here.");
            } else {
                println!(
                    "Prompt privacy is off for this model because it was turned off explicitly."
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
            println!("{model}: fetching piece(s) {shards}.");
            println!("Prompt privacy turns on by itself once they arrive.");
            println!("Progress:  swarmllm status");
        }
        Some("no_download_source") => {
            println!("{model}: the missing piece(s) cannot be fetched directly —");
            println!("no HuggingFace source is recorded for this model.");
            println!("They may still arrive from peers; privacy turns on by itself once");
            println!("both ends are present.");
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
