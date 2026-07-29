//! `swarmllm peers` — list connected peers from a running daemon.

use super::read_api_key;

pub async fn query_peers(
    port: u16,
    data_dir: &std::path::Path,
    json_output: bool,
) -> anyhow::Result<()> {
    let api_key = read_api_key(data_dir).unwrap_or_default();
    if api_key.is_empty() {
        eprintln!(
            "Warning: no API key found at {}",
            data_dir.join("api_key").display()
        );
    }

    let url = format!("http://localhost:{port}/api/admin/peers");
    let client = reqwest::Client::new();
    let mut req = client.get(&url);
    if !api_key.is_empty() {
        req = req.header("Authorization", format!("Bearer {api_key}"));
    }

    match req.send().await {
        Ok(resp) => {
            let body = resp.text().await?;
            if json_output {
                if let Ok(json) = serde_json::from_str::<serde_json::Value>(&body) {
                    println!("{}", serde_json::to_string_pretty(&json)?);
                } else {
                    println!("{body}");
                }
            } else {
                let peers: Vec<serde_json::Value> = match serde_json::from_str(&body) {
                    Ok(p) => p,
                    Err(_) => {
                        if crate::cli::body_is_auth_error(&body) {
                            crate::cli::exit_api_key_rejected(data_dir, port);
                        }
                        eprintln!("Unexpected response from daemon:\n{body}");
                        std::process::exit(1);
                    }
                };
                if peers.is_empty() {
                    println!("No connected peers.");
                } else {
                    let header = format!(
                        "{:<18} {:>8} {:>6} {:>7} {}",
                        "NODE ID", "LATENCY", "TRUST", "STATUS", "MODELS"
                    );
                    println!("{header}");
                    println!("{}", "-".repeat(70));
                    for p in &peers {
                        let node_id = p["node_id"].as_str().unwrap_or("?");
                        let latency = p["latency_ms"]
                            .as_u64()
                            .map(|l| format!("{l}ms"))
                            .unwrap_or_else(|| "—".to_string());
                        let trust = p["trust_score"]
                            .as_f64()
                            .map(|t| format!("{t:.2}"))
                            .unwrap_or_else(|| "—".to_string());
                        let healthy = if p["healthy"].as_bool().unwrap_or(false) {
                            "OK"
                        } else {
                            "DOWN"
                        };
                        let models: Vec<&str> = p["hosted_models"]
                            .as_array()
                            .map(|arr| arr.iter().filter_map(|v| v.as_str()).collect::<Vec<_>>())
                            .unwrap_or_default();
                        let model_str = if models.is_empty() {
                            "—".to_string()
                        } else {
                            models.join(", ")
                        };
                        println!(
                            "{:<18} {:>8} {:>6} {:>7} {}",
                            node_id, latency, trust, healthy, model_str
                        );
                    }
                    println!("\n{} peer(s) connected.", peers.len());
                }
            }
        }
        Err(_) => super::exit_daemon_unreachable(port),
    }

    Ok(())
}
