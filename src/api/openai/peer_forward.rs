use axum::response::IntoResponse;

use crate::error::ApiError;

use super::types::ChatCompletionRequest;

/// Timeout for peer-forwarded inference requests (seconds).
const INFERENCE_FORWARD_TIMEOUT_SECS: u64 = 120;

/// TCP connect timeout for peer HTTP forwarding (seconds).
const PEER_FORWARD_CONNECT_TIMEOUT_SECS: u64 = 10;

pub(super) fn peer_http_url(peer: &crate::types::PeerInfo) -> Option<String> {
    // Prefer UDP port (QUIC port == HTTP API port per convention),
    // fall back to TCP port - 10 (P2P TCP = HTTP + 10).
    let mut best_ip = None;
    let mut best_port = None;
    let mut have_udp = false;

    for addr in &peer.addresses {
        let parts: Vec<&str> = addr.split('/').collect();
        let mut ip = None;
        let mut udp_port = None;
        let mut tcp_port = None;
        for i in 0..parts.len() {
            if parts[i] == "ip4" && i + 1 < parts.len() {
                ip = Some(parts[i + 1]);
            }
            if parts[i] == "udp" && i + 1 < parts.len() {
                udp_port = Some(parts[i + 1]);
            }
            if parts[i] == "tcp" && i + 1 < parts.len() {
                tcp_port = Some(parts[i + 1]);
            }
        }
        if let Some(ip_str) = ip {
            if let Ok(parsed) = ip_str.parse::<std::net::Ipv4Addr>() {
                // Skip loopback, unspecified, and private IP ranges to prevent
                // SSRF via gossip-controlled peer addresses
                if parsed.is_loopback()
                    || parsed.is_unspecified()
                    || parsed.is_private()
                    || parsed.is_link_local()
                {
                    continue;
                }
            }
            // UDP port == HTTP API port (preferred)
            if let Some(port_str) = udp_port {
                if !have_udp {
                    best_ip = Some(ip_str.to_string());
                    best_port = Some(port_str.to_string());
                    have_udp = true;
                }
            }
            // TCP port = HTTP + 10, so HTTP = TCP - 10
            if let Some(port_str) = tcp_port {
                if !have_udp {
                    if let Ok(p) = port_str.parse::<u16>() {
                        best_ip = Some(ip_str.to_string());
                        best_port = Some(p.saturating_sub(10).to_string());
                    }
                }
            }
        }
    }
    match (best_ip, best_port) {
        (Some(ip), Some(port)) => Some(format!("http://{}:{}", ip, port)),
        _ => None,
    }
}

/// Lazily-initialized shared reqwest client for peer forwarding.
/// Avoids creating a new TLS + connection pool on every request.
static PEER_CLIENT: std::sync::LazyLock<reqwest::Client> = std::sync::LazyLock::new(|| {
    crate::http::build_client(|b| {
        b.connect_timeout(std::time::Duration::from_secs(
            PEER_FORWARD_CONNECT_TIMEOUT_SECS,
        ))
        .timeout(std::time::Duration::from_secs(
            INFERENCE_FORWARD_TIMEOUT_SECS,
        ))
    })
});

fn get_peer_client() -> &'static reqwest::Client {
    &PEER_CLIENT
}

/// Forward a chat completion request to a peer's HTTP API.
///
/// The receiving daemon's auth middleware requires Bearer auth for non-loopback
/// peer-forwarded requests — the `internal_auth_token` is per-process random
/// and not shareable across nodes. So we forward the originating request's
/// Authorization header verbatim. In the standard SwarmLLM cluster
/// deployment all daemons share the same API key (set via env or data dir),
/// so the receiver's Bearer check passes. If the originator didn't send an
/// Authorization header (e.g. unauthed local probe) we still fail loudly at
/// the receiver — that's correct behavior, not a regression.
pub(super) async fn forward_to_peer(
    peer_url: &str,
    req: &ChatCompletionRequest,
    stream: bool,
    auth_header: Option<&str>,
) -> Result<axum::response::Response, ApiError> {
    let client = get_peer_client();
    let url = format!("{}/v1/chat/completions", peer_url);

    let mut builder = client
        .post(&url)
        .header("x-swarm-forwarded", "true")
        .json(req);
    if let Some(auth) = auth_header {
        builder = builder.header(reqwest::header::AUTHORIZATION, auth);
    }
    let peer_resp = builder.send().await.map_err(|e| {
        tracing::warn!(error = %e, url = %url, "Failed to forward to peer");
        ApiError(crate::error::SwarmError::Network(format!(
            "Peer forwarding failed: {e}"
        )))
    })?;

    if !peer_resp.status().is_success() {
        let status = peer_resp.status();
        let raw_body = peer_resp.text().await.unwrap_or_default();
        return Err(crate::api::providers::extract_provider_error(
            &raw_body,
            status,
            "peer-forward",
            crate::api::providers::OPENAI_ERROR_KEYS,
        ));
    }

    let response = crate::api::providers::build_passthrough_response(peer_resp, stream).await?;
    Ok(response.into_response())
}
