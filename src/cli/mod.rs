//! Subcommand implementations for the SwarmLLM CLI.
//!
//! Each submodule owns one top-level subcommand (or family of subcommands).
//! The `main.rs` binary parses args via clap, then dispatches to the public
//! entry point in the matching submodule.

pub mod bench;
pub mod chat;
pub mod diagnostics;
pub mod get_model;
pub mod peers;
pub mod pool;
pub mod privacy;
pub mod remove_model;
pub mod run;
pub mod split_test;
pub mod status;
pub mod unload_model;
pub mod update;

/// Where every client command reaches the daemon on `port`: the IPv4
/// loopback address it listens on (`api::server::api_bind_addr`), never
/// `localhost`. `localhost` usually resolves to `::1` first, where the daemon
/// does not listen and any local program may — so a command sent its key to
/// whichever program held `[::1]:port`, and the key proof (`client_access`)
/// would have been checked at one address and the key sent to another.
pub(crate) fn daemon_base(port: u16) -> String {
    format!("http://127.0.0.1:{port}")
}

/// The HTTP client a command reaches the daemon with: never through a proxy.
/// `reqwest` takes `HTTP_PROXY` from the environment, and through one the key
/// went to the proxy, which connected to its OWN host's port — the proof was
/// checked on one path and the key sent on another (the pre-release review,
/// 2026-10-20).
pub(crate) fn daemon_client_builder() -> reqwest::ClientBuilder {
    reqwest::Client::builder().no_proxy()
}

/// [`daemon_client_builder`], built. As `reqwest::Client::new`, it fails only
/// where no HTTP client can be built at all.
pub(crate) fn daemon_client() -> reqwest::Client {
    daemon_client_builder()
        .build()
        .expect("an HTTP client builds")
}

/// What a client command reaches the daemon with: where its data is, and the
/// API key read from there ONCE (#255). A command never reads the key file
/// itself — a second read is not the bytes the proof checked, and a program
/// that named the directory could swap the file between the two (guard
/// `no_command_reads_the_key_file_itself`).
pub(crate) struct DaemonAccess {
    pub(crate) data_dir: std::path::PathBuf,
    pub(crate) api_key: Option<String>,
}

/// The daemon's data directory and key for a client command (#255).
///
/// `-d` or `SWARMLLM_NODE_DATA_DIR` names the directory when either is given.
/// Otherwise the daemon on `port` is asked — `/health` names its data
/// directory to a request from this machine. A packaged service keeps its data
/// in `/var/lib/swarmllm`, which is not the default a user's command looks in,
/// so `swarmllm status` beside a running service said it was "not running"
/// and to start it, and a stale key left in the default directory was
/// rejected. A daemon that does not answer, answers without a directory (an
/// older one), or names one that does not exist here (a container's own path)
/// leaves the default.
///
/// **The named directory's key is sent only to a daemon that proves it holds
/// it** ([`choose_access`]). Without the proof, any local program listening
/// on the port while the node is down could name `/var/lib/swarmllm` and be
/// handed the service's key by `sudo swarmllm status`. While the node runs it
/// holds the port, and a program on this machine can already be handed its
/// key the way the dashboard is (`api::dashboard_trust`), so the proof guards
/// the time the node is down. A command that refuses, or cannot read the key,
/// says why and exits, having sent nothing.
///
/// Research: Jupyter finds its servers through runtime files in the user's
/// runtime directory, which a command run by another user — the packaged
/// service runs as `swarmllm` — never sees; the Docker and Tailscale clients
/// ask the running daemon at the address they already use. This is the second:
/// the port every client command talks to anyway.
pub(crate) async fn client_access(
    cli_data_dir: Option<&std::path::Path>,
    port: u16,
) -> DaemonAccess {
    let access = if cli_data_dir.is_some() || std::env::var_os("SWARMLLM_NODE_DATA_DIR").is_some() {
        DaemonAccess::read_from(swarmllm::config::resolve_data_dir(cli_data_dir))
    } else {
        let nonce = health_nonce();
        let answer = ask_health(port, &nonce).await;
        choose_access(
            swarmllm::config::resolve_data_dir(None),
            answer,
            &nonce,
            port,
        )
    };
    access.unwrap_or_else(|why| {
        eprintln!("Error: {why}");
        std::process::exit(1);
    })
}

impl DaemonAccess {
    /// `dir` and the key in it, unproven: a directory the user named, or the
    /// default when no daemon named one.
    fn read_from(dir: std::path::PathBuf) -> Result<Self, String> {
        let api_key = read_key_file(&dir)?;
        Ok(Self {
            data_dir: dir,
            api_key,
        })
    }
}

/// What `/health` said about the daemon's data directory.
#[derive(Debug, Default, serde::Deserialize)]
struct HealthAnswer {
    data_dir: Option<std::path::PathBuf>,
    key_proof: Option<String>,
}

/// A fresh challenge for `/health`: 32 random bytes, hex.
fn health_nonce() -> String {
    use rand::RngCore;
    let mut bytes = [0u8; 32];
    rand::rngs::OsRng.fill_bytes(&mut bytes);
    hex::encode(bytes)
}

/// The directory and key, given what the daemon on `port` answered.
///
/// The default where the daemon named no directory that exists here. Where it
/// did, its key is read — once, here — and that directory and key are the
/// answer only if `key_proof` is [`swarmllm::api::server::health_key_proof`]
/// of this `nonce` and `port` under that key; otherwise a refusal, because the
/// program on the port is not the node that keeps its data there. The port is
/// in the proof so a program on this port cannot pass on the challenge to a
/// node on another and hand back its answer. No key there means nothing to
/// send; a key this user cannot read is an error.
fn choose_access(
    default: std::path::PathBuf,
    answer: Option<HealthAnswer>,
    nonce: &str,
    port: u16,
) -> Result<DaemonAccess, String> {
    let named = answer.and_then(|a| {
        let proof = a.key_proof;
        a.data_dir.filter(|d| d.is_dir()).map(|d| (d, proof))
    });
    let Some((dir, proof)) = named else {
        return DaemonAccess::read_from(default);
    };
    let api_key = read_key_file(&dir)?;
    let proven = |key: &str| {
        let expected = swarmllm::api::server::health_key_proof(key, port, nonce);
        // `blake3::Hash`'s equality is constant-time.
        proof
            .as_deref()
            .and_then(|p| blake3::Hash::from_hex(p).ok())
            .is_some_and(|p| p == expected)
    };
    match api_key.as_deref() {
        None | Some("") => {}
        Some(key) if proven(key) => {}
        Some(_) => {
            return Err(format!(
                "the program answering on port {port} says its data is in {}, but it \
                 could not prove it holds the access key kept there, so the key was not sent.\n  \
                 If you rotated the key, restart SwarmLLM. Otherwise another program may \
                 be using port {port}: stop it, or name the folder with -d.",
                dir.display()
            ))
        }
    }
    Ok(DaemonAccess {
        data_dir: dir,
        api_key,
    })
}

/// Ask the daemon on `port` where it keeps its data, with a challenge for its
/// key. `None` when nothing answers within two seconds or the answer is not
/// one.
async fn ask_health(port: u16, nonce: &str) -> Option<HealthAnswer> {
    let client = daemon_client_builder()
        .timeout(std::time::Duration::from_secs(2))
        .build()
        .ok()?;
    client
        .get(format!("{}/health?nonce={nonce}", daemon_base(port)))
        .send()
        .await
        .ok()?
        .json()
        .await
        .ok()
}

/// The API key in `data_dir`: `None` where there is no key file, an error
/// where there is one this user cannot read. Read only by [`client_access`].
fn read_key_file(data_dir: &std::path::Path) -> Result<Option<String>, String> {
    let key_path = data_dir.join("api_key");
    match std::fs::read_to_string(&key_path) {
        Ok(s) => Ok(Some(s.trim().to_string())),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        // The file is THERE and we cannot read it. Folding that into the
        // same `None` as "no file" sends the caller down
        // `bail_if_no_api_key`, which says the daemon is not running and
        // tells them to start it — advice that cannot work and that they
        // will follow, because the daemon IS running. The usual cause is
        // running the CLI as a different user from the daemon.
        Err(e) => Err(format!(
            "cannot read {} ({e}).\n  \
             The file is there but this user cannot open it — SwarmLLM is \
             probably running as a different user.\n  \
             Check its owner and permissions, or run the command as that user{}.",
            key_path.display(),
            // A packaged service runs as its own user with a private
            // directory; root reads it, and finds it by asking the daemon
            // (`client_access`), so the plain advice works.
            if cfg!(unix) { " (or with sudo)" } else { "" }
        )),
    }
}

/// Bail with the canonical "daemon not running (no API key)" message.
/// Shared by CLI commands that need an API key to call the local HTTP API.
pub(crate) fn bail_if_no_api_key(api_key: &str, data_dir: &std::path::Path) -> anyhow::Result<()> {
    if api_key.is_empty() {
        anyhow::bail!(
            "SwarmLLM is not running (no API key at {}).\n  Start the daemon first: swarmllm run",
            data_dir.join("api_key").display()
        );
    }
    Ok(())
}

/// Print the canonical "daemon unreachable on port N" guidance and
/// `exit(1)`. Shared by CLI commands that print errors instead of bailing.
pub(crate) fn exit_daemon_unreachable(port: u16) -> ! {
    eprintln!("Error: SwarmLLM daemon is not running on port {port}.");
    eprintln!("  Start it: swarmllm run");
    eprintln!("  Or if it's on a different port: --port <N> or set SWARMLLM_NODE_LISTEN_PORT");
    std::process::exit(1);
}

/// Explain a rejected API key and `exit(1)`.
///
/// A 401 from your own daemon is almost never "wrong password" — it is the CLI
/// reading a key from a different place than the daemon wrote it. Two ways that
/// happens, and the raw `authentication_error` names neither:
///
/// - The daemon runs with `-d /some/dir` (needed for a systemd unit, since the
///   default XDG path is wrong there) and the CLI, invoked without `-d`, looks
///   under the default data dir. Reported 2026-07-29.
/// - The key file has diverged from the key the daemon actually accepts, so a
///   file exists and is simply stale.
///
/// Both are fixed by pointing the CLI at the right data dir, so say so.
pub(crate) fn exit_api_key_rejected(data_dir: &std::path::Path, port: u16) -> ! {
    eprintln!("Error: the daemon on port {port} rejected this API key.");
    eprintln!("  Key read from: {}", data_dir.join("api_key").display());
    eprintln!();
    eprintln!("  This usually means the CLI and the daemon disagree about the data directory.");
    eprintln!("  If you started the daemon with -d/--data-dir, pass the SAME one here:");
    eprintln!("      swarmllm -d <dir> <command>");
    eprintln!("  or set SWARMLLM_NODE_DATA_DIR so both agree.");
    eprintln!();
    eprintln!("  If the path above IS correct, the stored key is stale — restart the daemon.");
    std::process::exit(1);
}

/// Explain a rejected API key if that is what this response is, otherwise do
/// nothing.
///
/// **Every CLI command that sends an authenticated request must call this on the
/// response status before interpreting the body.** The daemon answers a bad key
/// with 401 and a JSON `authentication_error`, which each command was then
/// rendering in its own words — "Download request failed (401 Unauthorized)",
/// "Could not remove <model>: request failed", or reqwest's raw
/// `error_for_status` text. All of them describe the transport and none of them
/// name the cause, which is almost always the CLI and the daemon disagreeing
/// about the data directory.
///
/// Only 401 is treated this way. The daemon uses it exclusively for
/// authentication (`api/middleware.rs`); a 403 or 503 means something else and
/// must keep its own message — notably the 503 that `remove-model` maps to
/// "serving a request right now".
///
/// Enforced by `cli_commands_explain_a_rejected_key` in
/// `tests/repo_consistency.rs`, so a new command inherits the requirement
/// instead of having to remember it.
pub(crate) fn exit_if_api_key_rejected(
    status: reqwest::StatusCode,
    data_dir: &std::path::Path,
    port: u16,
) {
    if status == reqwest::StatusCode::UNAUTHORIZED {
        exit_api_key_rejected(data_dir, port);
    }
}

/// Whether a daemon response body is an authentication failure.
pub(crate) fn body_is_auth_error(body: &str) -> bool {
    serde_json::from_str::<serde_json::Value>(body)
        .ok()
        .and_then(|v| {
            v.get("error")
                .and_then(|e| e.get("type"))
                .and_then(|t| t.as_str())
                .map(|t| t == "authentication_error")
        })
        .unwrap_or(false)
}

/// Resolve a model id: explicit override wins, otherwise pick the first
/// listing from the daemon's `/v1/models`.
pub(crate) async fn discover_model(
    client: &reqwest::Client,
    base: &str,
    api_key: &str,
    model_override: Option<String>,
    data_dir: &std::path::Path,
    port: u16,
) -> anyhow::Result<String> {
    if let Some(m) = model_override {
        return Ok(m);
    }
    let resp = client
        .get(format!("{base}/v1/models"))
        .header("Authorization", format!("Bearer {api_key}"))
        .send()
        .await?;
    // A rejected key answers with an error body that has no `data` array, so
    // without this the fallback below reported "No models available — load a
    // model first" — sending the user to download a model to fix an auth
    // problem.
    exit_if_api_key_rejected(resp.status(), data_dir, port);
    let models_resp: serde_json::Value = resp.json().await?;
    models_resp["data"][0]["id"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("No models available — load a model first"))
        .map(str::to_string)
}

#[cfg(test)]
mod tests {
    use super::*;

    const PORT: u16 = 8800;

    fn proof(key: &str, port: u16, nonce: &str) -> Option<String> {
        Some(
            swarmllm::api::server::health_key_proof(key, port, nonce)
                .to_hex()
                .to_string(),
        )
    }

    /// A data directory holding `key`, as a daemon publishes it.
    fn dir_with_key(key: &str) -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("api_key"), format!("{key}\n")).unwrap();
        dir
    }

    fn answer(dir: &std::path::Path, proof: Option<String>) -> Option<HealthAnswer> {
        Some(HealthAnswer {
            data_dir: Some(dir.to_path_buf()),
            key_proof: proof,
        })
    }

    fn chosen(access: Result<DaemonAccess, String>) -> (std::path::PathBuf, Option<String>) {
        let access = access.expect("an answer, not a refusal");
        (access.data_dir, access.api_key)
    }

    #[test]
    fn the_daemons_data_dir_is_used_only_where_it_exists_here() {
        let default = std::path::PathBuf::from("/nonexistent/default/swarmllm");
        let nonce = health_nonce();
        let here = dir_with_key("k1");
        assert_eq!(
            chosen(choose_access(
                default.clone(),
                answer(here.path(), proof("k1", PORT, &nonce)),
                &nonce,
                PORT
            )),
            (here.path().to_path_buf(), Some("k1".to_string())),
            "a running daemon's directory is where its key is, and that key is the one carried"
        );
        let elsewhere = std::path::Path::new("/data/inside/a/container");
        assert_eq!(
            chosen(choose_access(
                default.clone(),
                answer(elsewhere, None),
                &nonce,
                PORT
            )),
            (default.clone(), None),
            "a path that is not on this machine is not the answer"
        );
        assert_eq!(
            chosen(choose_access(default.clone(), None, &nonce, PORT)),
            (default.clone(), None)
        );
        assert_eq!(
            chosen(choose_access(
                default.clone(),
                Some(HealthAnswer::default()),
                &nonce,
                PORT
            )),
            (default, None),
            "an older daemon names no directory"
        );
    }

    /// The squatter of the pre-release review (2026-10-20): a program on the
    /// port names a directory whose key this user can read, and cannot prove
    /// it holds that key. Nothing is chosen, so nothing is sent.
    #[test]
    fn a_daemon_that_cannot_prove_it_holds_the_key_is_refused() {
        let default = std::path::PathBuf::from("/nonexistent/default/swarmllm");
        let nonce = health_nonce();
        let service = dir_with_key("the-services-key");
        for (what, given) in [
            ("no proof", None),
            ("not hex", Some("not hex".to_string())),
            ("a guessed key", proof("a-guess", PORT, &nonce)),
            (
                "an earlier challenge's proof",
                proof("the-services-key", PORT, &health_nonce()),
            ),
            // A program on this port that passed the challenge on to the node
            // on another port, and handed back its answer.
            (
                "another port's proof",
                proof("the-services-key", PORT + 1, &nonce),
            ),
        ] {
            let refusal =
                choose_access(default.clone(), answer(service.path(), given), &nonce, PORT)
                    .err()
                    .unwrap_or_else(|| panic!("{what} proves nothing"));
            assert!(
                refusal.contains("not sent") && refusal.contains("-d"),
                "{refusal}"
            );
        }
    }

    /// A key file that is not there is no key to send; the command then says
    /// the node is not running, as it did.
    #[test]
    fn a_directory_without_a_key_carries_none() {
        let default = std::path::PathBuf::from("/nonexistent/default/swarmllm");
        let empty = tempfile::tempdir().unwrap();
        assert_eq!(
            chosen(choose_access(
                default,
                answer(empty.path(), None),
                &health_nonce(),
                PORT
            )),
            (empty.path().to_path_buf(), None)
        );
    }

    /// Whether `source` reads the key file: names the reader, or the file
    /// other than to print its path. Whitespace is dropped first, so a read
    /// rustfmt has wrapped is still one read (gotcha #413).
    fn reads_the_key_file(source: &str) -> bool {
        let flat: String = source.split_whitespace().collect();
        flat.contains("read_key_file")
            || flat.contains("read_api_key")
            || flat
                .match_indices("\"api_key\"")
                .any(|(at, _)| !flat[at..].starts_with("\"api_key\").display()"))
    }

    /// The review's FIFO-and-rename attack needs a command to read the key file
    /// a second time, after the proof checked the first read. Only
    /// `client_access` reads it; a command is handed that key.
    #[test]
    fn no_command_reads_the_key_file_itself() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/cli");
        let mut scanned = 0;
        for entry in std::fs::read_dir(&dir).unwrap() {
            let path = entry.unwrap().path();
            if path.file_name().is_some_and(|n| n == "mod.rs")
                || path.extension().is_none_or(|x| x != "rs")
            {
                continue;
            }
            scanned += 1;
            assert!(
                !reads_the_key_file(&std::fs::read_to_string(&path).unwrap()),
                "{} reads the key file; take the key from `DaemonAccess`",
                path.display()
            );
        }
        assert!(scanned >= 10, "the scan found only {scanned} command files");
    }

    #[test]
    fn the_key_file_scan_sees_every_way_to_read_it() {
        for planted in [
            "let api_key = super::read_api_key(data_dir);",
            "let k = super::read_key_file(&dir)?;",
            "let k = std::fs::read_to_string(data_dir.join(\"api_key\"))?;",
            "let k = std::fs::read_to_string(\n        access\n            .data_dir\n            .join(\"api_key\"),\n    )?;",
            "let k = std::fs::read(dir.join(\"api_key\"));",
        ] {
            assert!(reads_the_key_file(planted), "missed: {planted}");
        }
        assert!(!reads_the_key_file(
            "eprintln!(\"no key at {}\", data_dir\n    .join(\"api_key\")\n    .display());"
        ));
    }

    /// The command asks the daemon on its port with a challenge, and checks the
    /// answer `/health` gives (`api::server::health`), when nothing names a
    /// directory.
    #[tokio::test]
    async fn a_command_without_a_data_dir_reads_the_running_daemons() {
        if std::env::var_os("SWARMLLM_NODE_DATA_DIR").is_some() {
            return; // the variable is an explicit answer; nothing is asked
        }
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let daemons = dir_with_key("daemon-key");
        let dir = daemons.path().to_path_buf();
        let app = axum::Router::new().route(
            "/health",
            axum::routing::get(move |axum::extract::RawQuery(q): axum::extract::RawQuery| {
                let dir = dir.clone();
                async move {
                    let nonce = q
                        .unwrap_or_default()
                        .trim_start_matches("nonce=")
                        .to_string();
                    axum::Json(serde_json::json!({
                        "status": "ok",
                        "data_dir": dir,
                        "key_proof": proof("daemon-key", port, &nonce),
                    }))
                }
            }),
        );
        tokio::spawn(async move { axum::serve(listener, app).await });

        let access = client_access(None, port).await;
        assert_eq!(access.data_dir, daemons.path());
        assert_eq!(access.api_key.as_deref(), Some("daemon-key"));
        let explicit = tempfile::tempdir().unwrap();
        assert_eq!(
            client_access(Some(explicit.path()), port).await.data_dir,
            explicit.path(),
            "-d is the answer, whatever the daemon says"
        );
    }

    #[tokio::test]
    async fn no_daemon_leaves_the_default() {
        if std::env::var_os("SWARMLLM_NODE_DATA_DIR").is_some() {
            return;
        }
        // A port nothing listens on: bind one, then free it.
        let port = std::net::TcpListener::bind("127.0.0.1:0")
            .unwrap()
            .local_addr()
            .unwrap()
            .port();
        assert_eq!(
            client_access(None, port).await.data_dir,
            swarmllm::config::resolve_data_dir(None)
        );
    }
}
