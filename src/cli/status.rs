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

    // What the machine is computing right now. A worker still busy for a
    // client that has gone used to be visible only in `ps`, and the only way
    // to stop it was `kill -9` (gotcha #445).
    let workers: Vec<serde_json::Value> = json
        .get("workers")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();
    if workers.is_empty() {
        println!("Workers:   none running (a model starts its worker on first use)");
    } else {
        println!("Workers:   {} running", workers.len());
        for w in &workers {
            println!("             {}", describe_worker(w));
        }
        if workers
            .iter()
            .any(|w| w.get("in_flight").and_then(|v| v.as_u64()).unwrap_or(0) > 0)
        {
            println!(
                "           (a worker still computing for a client that has gone can be retired \
                 with POST /api/admin/models/<model>/unload — the daemon stops it and frees \
                 its memory)"
            );
        }
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

/// One worker, one line: what it runs, where, how busy, how long.
///
/// A function rather than inline formatting so the wording — the part a
/// person reads — can be pinned by a test.
fn describe_worker(w: &serde_json::Value) -> String {
    let s = |k: &str| w.get(k).and_then(|v| v.as_str()).unwrap_or("?");
    let n = |k: &str| w.get(k).and_then(|v| v.as_u64()).unwrap_or(0);
    let pid = w
        .get("pid")
        .and_then(|v| v.as_u64())
        .map(|p| format!("pid {p}"))
        .unwrap_or_else(|| "pid unknown".to_string());
    let device = match (s("device"), w.get("cpu_reason").and_then(|v| v.as_str())) {
        (_, Some(reason)) => format!("processor ({})", cpu_reason_in_words(reason)),
        (d, None) => d.to_string(),
    };
    let busy = match n("in_flight") {
        0 => "idle".to_string(),
        1 => "1 request in flight".to_string(),
        k => format!("{k} requests in flight"),
    };
    let dead = if w.get("dead").and_then(|v| v.as_bool()).unwrap_or(false) {
        "  (exiting)"
    } else {
        ""
    };
    format!(
        "{}  {pid}  {device}  {busy}  idle {}  up {}{dead}",
        s("model"),
        human_secs(n("idle_secs")),
        human_secs(n("age_secs")),
    )
}

/// The daemon's stable machine tags (`CpuReason::as_str`) in plain words. An
/// unknown tag is shown as it is rather than hidden.
fn cpu_reason_in_words(tag: &str) -> String {
    match tag {
        "not_enough_vram" => "not enough graphics memory for it".to_string(),
        "configured_cpu_only" => "configured to use the processor".to_string(),
        "gpu_too_old_for_this_build" => "graphics card too old for this build".to_string(),
        other => other.to_string(),
    }
}

fn human_secs(secs: u64) -> String {
    match secs {
        s if s < 60 => format!("{s}s"),
        s if s < 3600 => format!("{}m", s / 60),
        s => format!("{}h {}m", s / 3600, (s % 3600) / 60),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The line a person reads for a worker stuck on a request nobody is
    /// waiting for: the model, the process, where it runs and why, how busy.
    #[test]
    fn a_busy_processor_worker_is_described_in_plain_words() {
        let w = serde_json::json!({
            "model": "qwen2.5-14b-instruct-q4-k-m",
            "pid": 2466031,
            "device": "processor",
            "cpu_reason": "not_enough_vram",
            "in_flight": 1,
            "idle_secs": 0,
            "age_secs": 1200,
            "dead": false,
        });
        let line = describe_worker(&w);
        assert!(line.starts_with("qwen2.5-14b-instruct-q4-k-m  pid 2466031  processor (not enough graphics memory for it)  1 request in flight"), "{line}");
        assert!(line.ends_with("idle 0s  up 20m"), "{line}");
    }

    /// An idle card worker, and a worker whose process is already going.
    #[test]
    fn an_idle_card_worker_and_a_dying_one_read_as_such() {
        let idle = serde_json::json!({
            "model": "llama-3.2-3b", "pid": 7, "device": "graphics card",
            "cpu_reason": null, "in_flight": 0, "idle_secs": 190, "age_secs": 7300, "dead": false,
        });
        assert_eq!(
            describe_worker(&idle),
            "llama-3.2-3b  pid 7  graphics card  idle  idle 3m  up 2h 1m"
        );
        let dying = serde_json::json!({
            "model": "m", "pid": null, "device": "processor", "cpu_reason": "configured_cpu_only",
            "in_flight": 2, "idle_secs": 5, "age_secs": 5, "dead": true,
        });
        let line = describe_worker(&dying);
        assert!(
            line.contains("pid unknown")
                && line.contains("2 requests in flight")
                && line.ends_with("(exiting)"),
            "{line}"
        );
    }

    /// Every tag the daemon can emit has words; the mapping is pinned against
    /// the enum itself so a renamed tag cannot silently fall through.
    #[test]
    fn every_processor_reason_has_words() {
        use swarmllm::inference::process_pool::CpuReason;
        for r in [
            CpuReason::Configured,
            CpuReason::GpuTooOld,
            CpuReason::NotEnoughVram,
        ] {
            let tag = r.as_str();
            assert_ne!(cpu_reason_in_words(tag), tag, "no words for {tag}");
        }
    }
}
