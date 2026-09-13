//! `swarmllm pool` — device-pool management commands.

use clap::Subcommand;

use super::read_api_key;

#[derive(Subcommand, Debug)]
pub enum PoolAction {
    /// Create a new device pool (this node becomes the owner/master)
    Create {
        /// Pool name
        #[arg(long)]
        name: String,
    },
    /// Generate an invite code to share with your other devices
    InviteCode,
    /// Join a pool using an invite code from your master device
    Join {
        /// The invite code (e.g., A3F7K2M9)
        code: String,
    },
    /// Show pool status and the devices in it
    Status,
    /// Leave the current pool
    Leave,
}

/// POST to a pool endpoint and return the parsed body, exiting with status 1
/// on an `error` field.
///
/// Four of the five subcommands repeated the same shape: send, parse JSON,
/// check for `error`, print it and `exit(1)`, else print a success message.
/// Only the path, the optional request body, and the success text differed —
/// so each arm now supplies just those and keeps its own output.
async fn pool_post(
    client: &reqwest::Client,
    base: &str,
    auth: &str,
    path: &str,
    body: Option<serde_json::Value>,
    data_dir: &std::path::Path,
    port: u16,
) -> anyhow::Result<serde_json::Value> {
    let mut req = client.post(format!("{base}{path}")).bearer_auth(auth);
    if let Some(json) = body {
        req = req.json(&json);
    }
    let resp = req.send().await?;
    // Otherwise a rejected key prints the daemon's raw error object, which
    // names the transport and not the data-directory mismatch behind it.
    super::exit_if_api_key_rejected(resp.status(), data_dir, port);
    let parsed: serde_json::Value = resp.json().await?;
    if let Some(err) = parsed.get("error") {
        eprintln!("Error: {err}");
        std::process::exit(1);
    }
    Ok(parsed)
}

pub async fn run_pool_command(
    port: u16,
    data_dir: &std::path::Path,
    action: PoolAction,
) -> anyhow::Result<()> {
    let api_key = read_api_key(data_dir);
    let base = format!("http://127.0.0.1:{port}");
    let client = reqwest::Client::new();

    let auth_header = api_key.as_deref().unwrap_or("");

    match action {
        PoolAction::Create { name } => {
            let body = pool_post(
                &client,
                &base,
                auth_header,
                "/api/pool/create",
                Some(serde_json::json!({ "name": name })),
                data_dir,
                port,
            )
            .await?;
            println!("Pool created: {}", body["name"].as_str().unwrap_or(&name));
            println!(
                "Pool ID: {}",
                body.get("pool_id").and_then(|v| v.as_str()).unwrap_or("?")
            );
            println!("\nNext: Run 'swarmllm pool invite-code' to generate a code for your other devices.");
        }
        PoolAction::InviteCode => {
            let body = pool_post(
                &client,
                &base,
                auth_header,
                "/api/pool/generate-code",
                None,
                data_dir,
                port,
            )
            .await?;
            if let Some(code) = body.get("code").and_then(|v| v.as_str()) {
                println!("Invite Code: {code}");
                println!();
                println!("Share this code with your other devices.");
                println!("On each device, run: swarmllm pool join {code}");
                println!();
                println!("The code expires in 24 hours and can only be used once.");
            }
        }
        PoolAction::Join { code } => {
            pool_post(
                &client,
                &base,
                auth_header,
                "/api/pool/join",
                Some(serde_json::json!({ "code": code })),
                data_dir,
                port,
            )
            .await?;
            println!("Join request sent! Your device will be added to the pool");
            println!("once the owner's node processes the request.");
            // Deliberately says what the pool DOES rather than what it pays.
            // This used to promise "all credits earned by this device will be
            // forwarded to the pool owner" — a claim of value transfer about a
            // figure that is self-minted and reconciled with nobody, which is
            // exactly what `docs/CREDITS_DESIGN.md` § 4 switched off. The
            // dashboard was cleaned; the CLI was never enumerated.
            println!("\nThis device will serve requests as part of the pool, and the");
            println!("pool's owner can pin models to it.");
        }
        PoolAction::Status => {
            let resp = client
                .get(format!("{base}/api/pool/state"))
                .bearer_auth(auth_header)
                .send()
                .await?;
            // The one pool subcommand that does not go through `pool_post`;
            // without this a rejected key reports "Not in a device pool."
            super::exit_if_api_key_rejected(resp.status(), data_dir, port);
            let body: serde_json::Value = resp.json().await?;
            if body.get("in_pool").and_then(|v| v.as_bool()) == Some(true) {
                println!(
                    "Pool: {}",
                    body.get("name").and_then(|v| v.as_str()).unwrap_or("?")
                );
                println!(
                    "Pool ID: {}",
                    body.get("pool_id").and_then(|v| v.as_str()).unwrap_or("?")
                );
                // No credit total, and no CREDITS column below. Credits are
                // dormant and gate nothing (`MIN_BALANCE_FOR_INFERENCE = 0`,
                // `calculate_tier` ignores the balance); the exit criteria in
                // `docs/CREDITS_DESIGN.md` § 6 say they do not become visible
                // again until a credit provably moves between two machines. The
                // dashboard and the leaderboard stopped showing a balance; this
                // did not, because "every surface" was read as "every surface
                // the browser reaches".
                println!();
                if let Some(members) = body.get("members").and_then(|v| v.as_array()) {
                    let header = format!("  {:<20} {:>8} {}", "DEVICE", "CONTRIB", "JOINED");
                    println!("{header}");
                    println!("{}", "-".repeat(45));
                    for m in members {
                        let nid = m.get("node_id").and_then(|v| v.as_str()).unwrap_or("?");
                        let short_id = if nid.len() > 12 { &nid[..12] } else { nid };
                        let display = m
                            .get("device_name")
                            .and_then(|v| v.as_str())
                            .unwrap_or(short_id);
                        let level = m
                            .get("contribution_level")
                            .and_then(|v| v.as_u64())
                            .unwrap_or(100);
                        let joined = m
                            .get("joined_at")
                            .and_then(|v| v.as_str())
                            .map(|s| &s[..10])
                            .unwrap_or("?");
                        let use_ansi = std::env::var("NO_COLOR").is_err()
                            && std::io::IsTerminal::is_terminal(&std::io::stdout());
                        let is_online = m.get("online").and_then(|v| v.as_bool()).unwrap_or(false);
                        let online = match (is_online, use_ansi) {
                            (true, true) => "\x1b[32m●\x1b[0m",
                            (false, true) => "\x1b[90m○\x1b[0m",
                            (true, false) => "[on] ",
                            (false, false) => "[off]",
                        };
                        println!("{online} {display:<18} {level:>5}% {joined}");
                    }
                }
            } else {
                println!("Not in a device pool.");
                println!("\nTo create one: swarmllm pool create --name \"My Devices\"");
                println!("To join one:   swarmllm pool join <INVITE_CODE>");
            }
        }
        PoolAction::Leave => {
            pool_post(
                &client,
                &base,
                auth_header,
                "/api/pool/leave",
                None,
                data_dir,
                port,
            )
            .await?;
            println!("Left the device pool. This device now serves the swarm on its own.");
        }
    }

    Ok(())
}
