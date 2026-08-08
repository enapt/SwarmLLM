//! `swarmllm status` — query a running daemon's /v1/status endpoint.

use super::read_api_key;

pub async fn query_status(
    port: u16,
    data_dir: &std::path::Path,
    as_json: bool,
) -> anyhow::Result<()> {
    let api_key = read_api_key(data_dir).unwrap_or_default();

    if api_key.is_empty() {
        eprintln!(
            "Warning: no API key found at {}/api_key",
            data_dir.display()
        );
        eprintln!("         (is the daemon running with this data directory?)");
    }

    let url = format!("http://localhost:{port}/v1/status");
    // Progress chatter belongs on stderr, so `swarmllm status --json` can be
    // piped into `jq` without the first line breaking the parse.
    eprintln!("Querying daemon at {url}...");

    let client = reqwest::Client::new();
    let mut req = client.get(&url);
    if !api_key.is_empty() {
        req = req.header("Authorization", format!("Bearer {api_key}"));
    }

    match req.send().await {
        Ok(resp) => {
            let body = resp.text().await?;
            if crate::cli::body_is_auth_error(&body) {
                crate::cli::exit_api_key_rejected(data_dir, port);
            }
            match serde_json::from_str::<serde_json::Value>(&body) {
                Ok(json) if as_json => println!("{}", serde_json::to_string_pretty(&json)?),
                Ok(json) => print_summary(&json),
                // Not JSON at all — show whatever came back rather than hiding it.
                Err(_) => println!("{body}"),
            }
        }
        Err(_) => super::exit_daemon_unreachable(port),
    }

    Ok(())
}

/// Render the status payload the way `swarmllm peers` renders peers.
///
/// **Why this exists.** `status` is the first command most people run to check
/// their node is alive, and it printed a raw JSON object — while its sibling
/// `peers`, two lines away in the same CLI, printed a formatted table. For a
/// tool aimed at people who are not engineers, answering "is my node working?"
/// with a JSON blob is a worse answer than the same facts in a sentence.
///
/// `--json` keeps the exact previous output for anyone scripting against it,
/// and the "Querying…" line moved to stderr so that form pipes cleanly.
fn print_summary(json: &serde_json::Value) {
    let s = |k: &str| json.get(k).and_then(|v| v.as_str()).unwrap_or("unknown");
    let n = |k: &str| json.get(k).and_then(|v| v.as_u64());
    let list = |k: &str| -> Vec<String> {
        json.get(k)
            .and_then(|v| v.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|x| x.as_str().map(str::to_string))
                    .collect()
            })
            .unwrap_or_default()
    };

    println!("Node:      {} (v{})", s("status"), s("version"));
    println!("Node id:   {}", s("node_id"));
    // Named "Peer id" so `swarmllm status | grep -i "peer id"` — which
    // docs/NETWORKING.md tells anchor operators to run — finds it.
    match json.get("peer_id").and_then(|v| v.as_str()) {
        Some(p) => println!("Peer id:   {p}"),
        None => println!("Peer id:   not bound yet (still starting up)"),
    }

    match json.get("model_loaded").and_then(|v| v.as_bool()) {
        Some(true) => println!("Model:     {} — loaded and ready", s("model_name")),
        // Not an error: a node with no model still relays and serves peers.
        Some(false) => println!("Model:     none loaded (this node still helps the swarm)"),
        None => println!("Model:     unknown"),
    }

    match n("peers") {
        Some(0) => println!(
            "Peers:     none connected — this node is on its own for now, which is \
             normal for the first minute or two after starting"
        ),
        Some(1) => println!("Peers:     1 connected"),
        Some(p) => println!("Peers:     {p} connected"),
        None => println!("Peers:     unknown"),
    }

    let downloading = list("models_downloading");
    if !downloading.is_empty() {
        println!("Fetching:  {}", downloading.join(", "));
    }

    let available = list("network_models");
    if available.is_empty() {
        println!("Available: no models seen on the network yet");
    } else {
        println!("Available: {} model(s) on the network", available.len());
        for m in &available {
            println!("             {m}");
        }
    }

    println!("\n(run with --json for the raw response)");
}
