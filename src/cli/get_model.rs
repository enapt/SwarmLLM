//! `swarmllm get-model` — opt in to a pinned reference / test model.
//!
//! The reference models (`docs/REFERENCE_MODELS.md`) exist so results from
//! different machines are comparable, and so a fresh node has a known-good
//! thing to run. This is the terminal path — the same opt-in the dashboard's
//! "Testing & Diagnostics" panel offers, but usable on a headless box (a test
//! VPS, a server) with no browser.
//!
//! Nothing downloads on its own: listing needs no daemon, and a fetch only
//! happens because someone ran this command.

use swarmllm::model::reference::REFERENCE_MODELS;

/// Entry point for `swarmllm get-model [TIER] [--all]`.
///
/// With no `tier`, prints the available tiers (works without a running daemon).
/// With a `tier`, asks the local daemon to download that reference model —
/// this node's fair share of the shards by default, or every shard with `--all`.
pub async fn get_model(
    port: u16,
    data_dir: &std::path::Path,
    tier: Option<String>,
    all: bool,
) -> anyhow::Result<()> {
    // No tier → discovery. Static list, so it works before the daemon is up.
    let Some(tier) = tier else {
        print!("{}", format_tier_table());
        println!();
        println!("Fetch one:  swarmllm get-model <tier> [--all]");
        println!("  (default)  download only this node's fair share of the shards");
        println!("  --all      download every shard (needs room for the whole model)");
        return Ok(());
    };

    let Some(model) = REFERENCE_MODELS.iter().find(|m| m.tier == tier) else {
        eprintln!("Unknown tier: {tier}\n");
        eprint!("{}", format_tier_table());
        std::process::exit(1);
    };

    let api_key = super::read_api_key(data_dir).unwrap_or_default();
    super::bail_if_no_api_key(&api_key, data_dir)?;

    // Fair-share and explicit-shards are mutually exclusive on the endpoint.
    let payload = if all {
        let indices: Vec<u32> = (0..model.shards).collect();
        serde_json::json!({
            "repo_id": model.repo_id,
            "filename": model.filename,
            "shards": indices,
        })
    } else {
        serde_json::json!({
            "repo_id": model.repo_id,
            "filename": model.filename,
            "shards": [],
            "peer_fair_share": true,
        })
    };
    let what = if all {
        format!("all {} shards (~{} MB)", model.shards, model.size_mb)
    } else {
        format!("this node's fair share of {} shards", model.shards)
    };

    println!("Tier:  {}", model.tier);
    println!("Model: {}", model.model_id);
    println!("Fetch: {what}");
    println!();

    let base = format!("http://localhost:{port}");
    let client = reqwest::Client::new();
    let resp = match client
        .post(format!("{base}/api/admin/hf/download-shards"))
        .header("Authorization", format!("Bearer {api_key}"))
        .json(&payload)
        .send()
        .await
    {
        Ok(r) => r,
        Err(_) => super::exit_daemon_unreachable(port),
    };

    let status = resp.status();
    let body: serde_json::Value = resp.json().await.unwrap_or(serde_json::json!({}));
    if !status.is_success() {
        let msg = body
            .get("error")
            .and_then(|e| e.as_str())
            .map(str::to_string)
            .unwrap_or_else(|| body.to_string());
        anyhow::bail!("Download request failed ({status}): {msg}");
    }

    println!("Started — downloads run in the background.");
    println!("  Watch it:  swarmllm status   (or the dashboard at http://localhost:{port})");
    if !all {
        // A fair share is only usable if other nodes hold the rest. They are
        // expected to pick the remainder up via auto-manage, but nothing
        // guarantees they have, so say so rather than letting the first
        // inference fail with "No node available for layer 0".
        println!();
        println!("  Note: a fair share is only part of this model. Answering needs the");
        println!("  other shards to be held by connected peers — usually automatic, but");
        println!("  not instant, and not guaranteed.");
        println!("  If requests fail with \"No node available for layer ...\", fetch the");
        println!("  rest yourself:  swarmllm get-model {} --all", model.tier);
    }
    Ok(())
}

/// Render the reference tiers as an aligned table. Pure — no daemon needed —
/// so `get-model` (list) always works.
fn format_tier_table() -> String {
    let mut out = format!(
        "{:<10} {:>9} {:>7}  {}\n",
        "TIER", "DOWNLOAD", "SHARDS", "MODEL"
    );
    for m in REFERENCE_MODELS {
        out.push_str(&format!(
            "{:<10} {:>9} {:>7}  {}\n",
            m.tier,
            format!("{} MB", m.size_mb),
            m.shards,
            m.model_id,
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_lists_every_tier() {
        let t = format_tier_table();
        assert!(t.contains("TIER") && t.contains("SHARDS"));
        for m in REFERENCE_MODELS {
            assert!(t.contains(m.tier), "table missing tier {}", m.tier);
            assert!(t.contains(m.model_id), "table missing model {}", m.model_id);
        }
        // The three shipped tiers are the discovery surface for new testers.
        assert!(t.contains("smoke") && t.contains("standard") && t.contains("stress"));
    }
}
