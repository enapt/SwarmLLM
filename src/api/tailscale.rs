//! Asking Tailscale who a connection belongs to, instead of guessing from IPs.
//!
//! `tailscaled` serves an HTTP API — the LocalAPI — over a Unix socket, and its
//! `whois` endpoint maps a source address to the tailnet node and user behind
//! it. That is the authoritative answer to the question the dashboard actually
//! needs: *is this browser a member of my tailnet?*
//!
//! It replaces address-range inference, which cannot answer it. Tailscale's
//! IPv4 range is RFC 6598 shared space that internet providers also hand out,
//! so "this address looks like Tailscale" is a guess that a carrier-NAT
//! deployment can satisfy by coincidence — the defect an external security
//! review found in the first version of this trust check. Asking `tailscaled`
//! removes the guess from both sides at once: if the daemon is not there we are
//! not on a tailnet, and if it does not recognise the address then whoever is
//! calling is not on ours.
//!
//! Deliberately hand-rolled over `tokio::net::UnixStream`: the request is one
//! HTTP/1.1 GET with no body, and `reqwest` cannot speak to a Unix socket
//! without pulling in another transport stack for it.

use std::net::IpAddr;
use std::time::Duration;

/// Socket paths `tailscaled` is known to listen on.
///
/// Linux uses `/run`, macOS `/var/run`, and some appliance builds (Synology,
/// QNAP) place it under their own package root. Probing a short list costs a
/// failed `connect` on a missing path and keeps this dependency-free.
const SOCKET_PATHS: &[&str] = &[
    "/run/tailscale/tailscaled.sock",
    "/var/run/tailscale/tailscaled.sock",
    "/var/packages/Tailscale/etc/tailscaled.sock",
];

/// The LocalAPI is local-only, so this bounds a hung daemon rather than a
/// network round trip. Kept short because it sits in the dashboard's page-load
/// path: being slow to answer would be worse than falling back.
const LOCALAPI_TIMEOUT: Duration = Duration::from_millis(400);

/// Cap on the response we will read. A `whois` reply is a few KB of JSON;
/// this only stops a compromised or malfunctioning local daemon from
/// streaming unboundedly into our memory.
#[cfg(unix)]
const MAX_RESPONSE_BYTES: usize = 256 * 1024;

/// What `tailscaled` said about a source address.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WhoIs {
    /// The address belongs to a node on this tailnet.
    Member,
    /// `tailscaled` answered and does not recognise the address.
    NotAMember,
    /// No `tailscaled` to ask — not installed, not running, or its socket is
    /// not readable by this process. Says nothing about the address itself.
    Unavailable,
}

/// Ask `tailscaled` whether `ip` belongs to a node on this tailnet.
///
/// Returns [`WhoIs::Unavailable`] for every local failure — missing socket,
/// permission denied, timeout, malformed reply. A caller must treat that as "no
/// information", never as proof of membership: this is a security decision, and
/// an unreadable socket is exactly what a sandboxed service looks like.
pub async fn whois(ip: IpAddr) -> WhoIs {
    for path in SOCKET_PATHS {
        match tokio::time::timeout(LOCALAPI_TIMEOUT, query_socket(path, ip)).await {
            Ok(Ok(result)) => return result,
            // Wrong path, no permission, or the daemon hung: try the next
            // candidate, then give up as Unavailable.
            Ok(Err(_)) | Err(_) => continue,
        }
    }
    WhoIs::Unavailable
}

/// Windows has no Unix sockets, and `tailscaled` there exposes the LocalAPI over
/// a loopback TCP port guarded by a token file instead. That is a different
/// mechanism, and implementing it without a Windows machine to test against
/// would be guessing at a security boundary — the exact mistake that produced
/// the finding this module exists to fix. So Windows reports
/// [`WhoIs::Unavailable`] and falls back to the narrowed address check and the
/// paste-the-key path, which is correct-but-less-convenient rather than wrong.
/// Noted in `docs/FUTURE_WORK.md`.
#[cfg(not(unix))]
async fn query_socket(_path: &str, _ip: IpAddr) -> std::io::Result<WhoIs> {
    Err(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        "LocalAPI over a Unix socket is not available on this platform",
    ))
}

#[cfg(unix)]
async fn query_socket(path: &str, ip: IpAddr) -> std::io::Result<WhoIs> {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    let mut stream = tokio::net::UnixStream::connect(path).await?;

    // The port is optional for `whois`, and we do not reliably have the
    // browser's source port at the point the trust decision is made.
    //
    // `Host` is required by HTTP/1.1 and `tailscaled` rejects requests whose
    // Host is not one it expects, so send the value its own client uses.
    let request = format!(
        "GET /localapi/v0/whois?addr={ip} HTTP/1.1\r\n\
         Host: local-tailscaled.sock\r\n\
         Connection: close\r\n\
         \r\n"
    );
    stream.write_all(request.as_bytes()).await?;
    stream.flush().await?;

    let mut response = Vec::new();
    let mut chunk = [0u8; 8192];
    loop {
        let n = stream.read(&mut chunk).await?;
        if n == 0 {
            break;
        }
        response.extend_from_slice(&chunk[..n]);
        if response.len() > MAX_RESPONSE_BYTES {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "whois response too large",
            ));
        }
    }

    Ok(classify_response(&response))
}

/// Turn a raw HTTP response into a verdict.
///
/// Split out so the parsing is testable without a running `tailscaled`.
/// `tailscaled` answers 200 with a JSON body for a known address and a 4xx for
/// an unknown one. A 200 whose body carries no node is treated as NOT a member:
/// this decides whether to hand out an API key, so anything we cannot
/// affirmatively read as membership must not count as membership.
#[cfg(unix)]
pub(crate) fn classify_response(raw: &[u8]) -> WhoIs {
    let text = String::from_utf8_lossy(raw);
    let mut lines = text.split("\r\n");
    let Some(status_line) = lines.next() else {
        return WhoIs::Unavailable;
    };

    // "HTTP/1.1 200 OK"
    let mut parts = status_line.split_whitespace();
    let _version = parts.next();
    let status: u16 = match parts.next().and_then(|c| c.parse().ok()) {
        Some(c) => c,
        None => return WhoIs::Unavailable,
    };

    if status == 404 || status == 400 {
        // tailscaled answered and does not know this address.
        return WhoIs::NotAMember;
    }
    if status != 200 {
        // 403 (not permitted to ask), 500, anything else: no information.
        return WhoIs::Unavailable;
    }

    let Some(body_start) = text.find("\r\n\r\n") else {
        return WhoIs::Unavailable;
    };
    let body = &text[body_start + 4..];

    // A successful whois always carries a Node; UserProfile accompanies it.
    // Parse rather than substring-match so a field appearing in an error
    // message cannot be mistaken for a result.
    match serde_json::from_str::<serde_json::Value>(body.trim()) {
        Ok(v) => {
            if v.get("Node").is_some_and(|n| !n.is_null()) {
                WhoIs::Member
            } else {
                WhoIs::NotAMember
            }
        }
        Err(_) => WhoIs::Unavailable,
    }
}

#[cfg(all(test, unix))]
mod parse_tests {
    use super::*;

    fn response(status: &str, body: &str) -> Vec<u8> {
        format!("HTTP/1.1 {status}\r\nContent-Type: application/json\r\n\r\n{body}").into_bytes()
    }

    #[test]
    fn a_known_node_is_a_member() {
        let raw = response(
            "200 OK",
            r#"{"Node":{"ID":1,"Name":"laptop.tail1234.ts.net"},"UserProfile":{"LoginName":"a@b.c"}}"#,
        );
        assert_eq!(classify_response(&raw), WhoIs::Member);
    }

    #[test]
    fn an_unknown_address_is_not_a_member() {
        assert_eq!(
            classify_response(&response("404 Not Found", "no match for IP")),
            WhoIs::NotAMember
        );
        assert_eq!(
            classify_response(&response("400 Bad Request", "invalid addr")),
            WhoIs::NotAMember
        );
    }

    /// The distinction that matters: "I could not ask" must never read as
    /// "yes". Anything short of an affirmative answer denies the key handout.
    #[test]
    fn anything_unreadable_is_unavailable_not_member() {
        assert_eq!(classify_response(b""), WhoIs::Unavailable);
        assert_eq!(classify_response(b"garbage"), WhoIs::Unavailable);
        assert_eq!(
            classify_response(&response("403 Forbidden", "")),
            WhoIs::Unavailable
        );
        assert_eq!(
            classify_response(&response("500 Internal Server Error", "")),
            WhoIs::Unavailable
        );
        // 200 with a body we cannot parse is not evidence of anything.
        assert_eq!(
            classify_response(&response("200 OK", "not json")),
            WhoIs::Unavailable
        );
    }

    /// A 200 carrying no Node is an answer, and the answer is no.
    #[test]
    fn a_success_without_a_node_is_not_a_member() {
        assert_eq!(
            classify_response(&response("200 OK", r#"{"UserProfile":{}}"#)),
            WhoIs::NotAMember
        );
        assert_eq!(
            classify_response(&response("200 OK", r#"{"Node":null}"#)),
            WhoIs::NotAMember
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Probing a daemon that is not there must fail closed, on every platform.
    /// On Windows there is no Unix socket at all, so this also pins that the
    /// unsupported path reports Unavailable rather than claiming membership.
    #[tokio::test]
    async fn a_missing_daemon_is_never_a_member() {
        let verdict = whois("100.101.102.103".parse().unwrap()).await;
        // A developer machine may genuinely run Tailscale; CI does not. What
        // must never happen is Member from a socket that answered nothing.
        assert!(matches!(verdict, WhoIs::Unavailable | WhoIs::NotAMember));
    }
}
