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
//! This asks the daemon to fetch exactly those pieces. Where both are already
//! here it switches the setting on for the model. Where one is missing it sets
//! no flag (a flag before the parts arrive fails every request for the model
//! until the download finishes) and says whether a setting will switch it on
//! once they arrive or the owner must — the automatic rule is off by default
//! since 2026-11-09 (`inference.encrypted_pipeline_auto`).

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
            println!("{model}: both ends are already on this computer.");
            println!("Start and finish on this computer is ON — other computers never");
            println!("receive your text or the reply, but the numbers they work on can");
            println!("still reveal much of it.");
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
            print_when_it_turns_on(model, &body);
            println!("Progress:  swarmllm status");
        }
        Some("no_download_source") => {
            println!("{model}: the missing part(s) cannot be fetched directly —");
            println!("no HuggingFace source is recorded for this model.");
            println!("They may still arrive from other computers.");
            print_when_it_turns_on(model, &body);
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

/// Whether the setting switches on by itself once the parts arrive — only
/// where the model's own choice, the global switch or the automatic rule says
/// so (`turns_on_when_they_arrive`); otherwise how to switch it on then.
fn print_when_it_turns_on(model: &str, body: &serde_json::Value) {
    let by_itself = body
        .get("turns_on_when_they_arrive")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    if by_itself {
        println!("Start and finish on this computer turns on by itself once they arrive.");
    } else {
        println!("Once they have arrived, run `swarmllm privacy {model}` again, or switch");
        println!("it on from the model's card on the Dashboard, to turn it on.");
    }
}
