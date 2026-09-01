//! `swarmllm diagnostics` — print the node report a maintainer needs, in a
//! form that is safe to paste in public.
//!
//! **Why this is a command and not a documented `curl`.** The report has been
//! available at `GET /api/admin/diagnostics` for many releases and the README
//! calls it "the single most useful thing to include in a bug report" — but
//! reaching it meant knowing the endpoint, finding the API key file, and
//! writing an `Authorization` header. The people most likely to be asked for
//! it are the ones least likely to do that: someone running a node who has
//! noticed something is wrong. The dashboard has had a button for it; a node
//! on a headless machine had nothing.
//!
//! Addresses are redacted by the daemon unless `--full` is passed, so "run
//! `swarmllm diagnostics` and paste the output" is advice that stays safe to
//! give without a caveat attached.

use super::read_api_key;

pub async fn print_diagnostics(
    port: u16,
    data_dir: &std::path::Path,
    full: bool,
) -> anyhow::Result<()> {
    let api_key = read_api_key(data_dir).unwrap_or_default();
    super::bail_if_no_api_key(&api_key, data_dir)?;

    let url = format!(
        "http://localhost:{port}/api/admin/diagnostics{}",
        if full { "?full=1" } else { "" }
    );
    // Everything explanatory goes to stderr so `swarmllm diagnostics > report.txt`
    // captures the report and nothing else.
    eprintln!("Collecting diagnostics from {url}...");

    let client = reqwest::Client::new();
    let resp = match client
        .get(&url)
        .header("Authorization", format!("Bearer {api_key}"))
        .send()
        .await
    {
        Ok(resp) => resp,
        Err(_) => super::exit_daemon_unreachable(port),
    };

    let body = resp.text().await?;
    if crate::cli::body_is_auth_error(&body) {
        crate::cli::exit_api_key_rejected(data_dir, port);
    }
    println!("{body}");

    if full {
        eprintln!();
        eprintln!("This report includes network addresses — yours and your peers'.");
        eprintln!("Run without --full for a version that is safe to post publicly.");
    } else {
        eprintln!();
        eprintln!("Safe to paste: no API key, no invite code, no file paths, and every");
        eprintln!("network address replaced with a placeholder naming only its kind.");
    }

    Ok(())
}
