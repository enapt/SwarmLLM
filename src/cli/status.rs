//! `swarmllm status` — query a running daemon's /v1/status endpoint.

use super::read_api_key;

pub async fn query_status(port: u16, data_dir: &std::path::Path) -> anyhow::Result<()> {
    let api_key = read_api_key(data_dir).unwrap_or_default();

    if api_key.is_empty() {
        eprintln!(
            "Warning: no API key found at {}/api_key",
            data_dir.display()
        );
        eprintln!("         (is the daemon running with this data directory?)");
    }

    let url = format!("http://localhost:{port}/v1/status");
    println!("Querying daemon at {url}...");

    let client = reqwest::Client::new();
    let mut req = client.get(&url);
    if !api_key.is_empty() {
        req = req.header("Authorization", format!("Bearer {api_key}"));
    }

    match req.send().await {
        Ok(resp) => {
            let body = resp.text().await?;
            if let Ok(json) = serde_json::from_str::<serde_json::Value>(&body) {
                println!("{}", serde_json::to_string_pretty(&json)?);
            } else {
                println!("{body}");
            }
        }
        Err(_) => super::exit_daemon_unreachable(port),
    }

    Ok(())
}
