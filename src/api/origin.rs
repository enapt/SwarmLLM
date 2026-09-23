//! Is the person making this request sitting at this machine?
//!
//! Several things are granted to that person and nobody else: the automatic
//! API-key handout, `/metrics` without a key, checking for and applying an
//! update, shutting the node down, prompt previews in the responses list, and
//! the admin rate-limit exemption. Each used to decide it with
//! `addr.ip().is_loopback()` — which really means "the last TCP hop began in
//! this network namespace". A reverse proxy on the same host (nginx, Caddy,
//! `tailscale serve`, Funnel) is exactly that, on behalf of anyone who can
//! reach the proxy, so one same-host proxy handed all of them to the internet
//! at once. The book's own nginx example did it (fixed 2026-09-23).
//!
//! [`RequestOrigin::is_this_machine`] is the one answer, and asks three things:
//!
//! 1. the connection is loopback;
//! 2. the `Host` header names this machine — `localhost`, a `*.localhost`
//!    name, or a loopback IP. A proxy passes on the name the visitor used, and
//!    so does a DNS-rebinding page. This is Jupyter's check (`check_host`,
//!    `ServerApp.local_hostnames`);
//! 3. no forwarding header — `Forwarded` (RFC 7239), `X-Forwarded-*`,
//!    `X-Real-IP`, `Via`, or Tailscale Serve's `Tailscale-User-Login`. A
//!    browser at this machine sends none of them; Caddy and `tailscale serve`
//!    add them by default. Home Assistant draws the same line: forwarding
//!    headers from a proxy nobody declared trusted earn nothing.
//!
//! **This is defence in depth, not proof.** A proxy can be set up to send
//! neither a public `Host` nor a forwarding header — nginx's defaults do
//! exactly that — so the docs still say never to proxy over 127.0.0.1. What it
//! guarantees is the safe direction: every signal here can only TAKE a
//! privilege away. A spoofed header costs its sender the automatic key and
//! grants nothing, which is why none of them needs to be trusted.

use std::net::{IpAddr, SocketAddr};

use axum::extract::{ConnectInfo, FromRequestParts};
use axum::http::request::Parts;
use axum::http::HeaderMap;

/// Where a request came from, for the privileges above.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RequestOrigin {
    /// The TCP peer's address — the proxy's own, when there is one.
    pub ip: IpAddr,
    /// A forwarding header was present, or a loopback request named some host
    /// other than this machine.
    pub via_proxy: bool,
}

impl RequestOrigin {
    pub fn new(peer: IpAddr, headers: &HeaderMap) -> Self {
        let via_proxy = carries_forwarding_header(headers)
            || (peer.is_loopback() && !host_names_this_machine(headers));
        Self {
            ip: peer,
            via_proxy,
        }
    }

    /// The single answer to "is the person making this request at this
    /// machine?". Read this, never the socket address.
    pub fn is_this_machine(&self) -> bool {
        self.ip.is_loopback() && !self.via_proxy
    }
}

impl<S> FromRequestParts<S> for RequestOrigin
where
    S: Send + Sync,
{
    type Rejection = <ConnectInfo<SocketAddr> as FromRequestParts<S>>::Rejection;

    async fn from_request_parts(parts: &mut Parts, state: &S) -> Result<Self, Self::Rejection> {
        let ConnectInfo(addr) = ConnectInfo::<SocketAddr>::from_request_parts(parts, state).await?;
        Ok(Self::new(addr.ip(), &parts.headers))
    }
}

/// Headers a proxy adds and a browser never does.
const FORWARDING_HEADERS: &[&str] = &[
    "forwarded",
    "x-forwarded-for",
    "x-forwarded-host",
    "x-forwarded-proto",
    "x-real-ip",
    "via",
    "tailscale-user-login",
];

fn carries_forwarding_header(headers: &HeaderMap) -> bool {
    FORWARDING_HEADERS.iter().any(|h| headers.contains_key(*h))
}

/// Does `Host` name this machine? An absent header is not evidence of a proxy
/// — every proxy sends one — so it counts as yes.
fn host_names_this_machine(headers: &HeaderMap) -> bool {
    let Some(value) = headers.get(axum::http::header::HOST) else {
        return true;
    };
    let Ok(host) = value.to_str() else {
        return false;
    };
    host_is_local(host)
}

fn host_is_local(host: &str) -> bool {
    let host = host.trim();
    // `[::1]:8800`, `[::1]`
    let name = if let Some(rest) = host.strip_prefix('[') {
        match rest.split_once(']') {
            Some((inner, _)) => inner,
            None => return false,
        }
    } else {
        // `name:port` — only strip a trailing all-digit port.
        match host.rsplit_once(':') {
            Some((name, port)) if !port.is_empty() && port.bytes().all(|b| b.is_ascii_digit()) => {
                name
            }
            _ => host,
        }
    };
    let name = name.trim_end_matches('.').to_ascii_lowercase();
    if let Ok(ip) = name.parse::<IpAddr>() {
        return ip.is_loopback();
    }
    name == "localhost" || name.ends_with(".localhost")
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::HeaderValue;

    fn headers(pairs: &[(&'static str, &str)]) -> HeaderMap {
        let mut h = HeaderMap::new();
        for (k, v) in pairs {
            h.insert(*k, HeaderValue::from_str(v).unwrap());
        }
        h
    }

    fn local() -> IpAddr {
        "127.0.0.1".parse().unwrap()
    }

    /// The browser at this machine, however it spells the address.
    #[test]
    fn a_browser_at_this_machine_is_this_machine() {
        for host in [
            "localhost:8800",
            "127.0.0.1:8800",
            "[::1]:8800",
            "LOCALHOST",
            "localhost.",
            "app.localhost:8800",
        ] {
            let o = RequestOrigin::new(local(), &headers(&[("host", host)]));
            assert!(o.is_this_machine(), "{host}");
        }
        // A client that sends no Host at all (HTTP/1.0) is not a proxy.
        assert!(RequestOrigin::new(local(), &HeaderMap::new()).is_this_machine());
        let v6: IpAddr = "::1".parse().unwrap();
        assert!(RequestOrigin::new(v6, &headers(&[("host", "[::1]:8800")])).is_this_machine());
    }

    /// The hazard: a proxy on this host reaches us over loopback, carrying the
    /// name the visitor used or a forwarding header. Each alone is enough.
    #[test]
    fn a_proxy_on_this_machine_is_not_this_machine() {
        let cases: &[&[(&'static str, &str)]] = &[
            // nginx `proxy_set_header Host $host;` — the book's own example.
            &[("host", "swarmllm.example.com")],
            // Caddy's defaults.
            &[
                ("host", "localhost:8800"),
                ("x-forwarded-for", "203.0.113.9"),
            ],
            // tailscale serve / Funnel.
            &[
                ("host", "node.tailnet.ts.net"),
                ("tailscale-user-login", "alice@example.com"),
            ],
            &[("host", "127.0.0.1:8800"), ("forwarded", "for=203.0.113.9")],
            &[("host", "127.0.0.1:8800"), ("x-real-ip", "203.0.113.9")],
            &[("host", "127.0.0.1:8800"), ("via", "1.1 proxy")],
            // DNS rebinding: a public name resolved to 127.0.0.1.
            &[("host", "evil.example:8800")],
            // Not a local name, however local it looks.
            &[("host", "localhost.evil.example")],
            &[("host", "127.0.0.1.nip.io")],
        ];
        for h in cases {
            let o = RequestOrigin::new(local(), &headers(h));
            assert!(o.via_proxy, "{h:?}");
            assert!(!o.is_this_machine(), "{h:?}");
        }
    }

    /// A LAN client is never "this machine", and a forwarding header marks a
    /// LAN proxy too — so LAN trust cannot be extended to whoever is behind it.
    #[test]
    fn a_lan_address_is_never_this_machine() {
        let lan: IpAddr = "192.168.1.10".parse().unwrap();
        let direct = RequestOrigin::new(lan, &headers(&[("host", "192.168.1.10:8800")]));
        assert!(!direct.is_this_machine());
        assert!(!direct.via_proxy, "a plain LAN browser is not a proxy");
        let proxied = RequestOrigin::new(
            lan,
            &headers(&[
                ("host", "swarmllm.example.com"),
                ("x-forwarded-for", "8.8.8.8"),
            ]),
        );
        assert!(proxied.via_proxy);
    }
}
