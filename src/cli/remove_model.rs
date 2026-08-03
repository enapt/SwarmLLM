//! `swarmllm remove-model <model>` — delete a model and tell the network.
//!
//! Until this existed there was no way to free a model from a terminal. The
//! dashboard and the HTTP API could do it properly, but a headless box had only
//! `rm -rf ~/.local/share/swarmllm/models/<id>/` — which removes the files and
//! nothing else. The node went on listing the model, offering its pieces to
//! other machines and failing when they were asked for, until it was restarted.
//! (A reconciliation pass now limits that to one announcement cycle, but the
//! user still could not cleanly do a thing the software supports.)
//!
//! This calls the same endpoint the dashboard uses, so the files, the records,
//! the network advertisement and the shared directory all go together.

/// Ask the daemon to delete `model` and retract it from the network.
pub async fn remove_model(
    port: u16,
    data_dir: &std::path::Path,
    model: &str,
    yes: bool,
) -> anyhow::Result<()> {
    let api_key = super::read_api_key(data_dir).unwrap_or_default();
    super::bail_if_no_api_key(&api_key, data_dir)?;

    // Deleting a model can mean re-downloading many gigabytes, so it is worth
    // one question unless the caller has already answered it. Scripts and
    // non-interactive shells pass `--yes`.
    if !yes {
        use std::io::Write;
        print!("Remove {model} and its downloaded pieces? [y/N] ");
        std::io::stdout().flush().ok();
        let mut answer = String::new();
        if std::io::stdin().read_line(&mut answer).is_err()
            || !answer.trim().eq_ignore_ascii_case("y")
        {
            println!("Left alone.");
            return Ok(());
        }
    }

    let client = reqwest::Client::new();
    let url = format!("http://127.0.0.1:{port}/api/admin/models/{model}");
    let resp = match client.delete(&url).bearer_auth(&api_key).send().await {
        Ok(r) => r,
        Err(_) => super::exit_daemon_unreachable(port),
    };

    let status = resp.status();
    let body: serde_json::Value = resp.json().await.unwrap_or(serde_json::Value::Null);

    if !status.is_success() {
        let msg = body
            .get("error")
            .and_then(|e| e.get("message"))
            .and_then(|m| m.as_str())
            .unwrap_or("request failed");
        // 503 here is the active-pipeline guard, which is a "try again", not a
        // failure the user needs to debug — say which it is.
        if status.as_u16() == 503 {
            anyhow::bail!("{model} is serving a request right now. Try again in a moment.");
        }
        anyhow::bail!("Could not remove {model}: {msg}");
    }

    let files = body
        .get("files_removed")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    println!("Removed {model} ({files} file(s)).");
    println!("Other machines have been told it is no longer here.");
    Ok(())
}
