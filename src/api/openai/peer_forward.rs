use axum::response::IntoResponse;

use crate::error::ApiError;

use super::types::ChatCompletionRequest;

/// TCP connect timeout for peer HTTP forwarding (seconds).
const PEER_FORWARD_CONNECT_TIMEOUT_SECS: u64 = 10;

/// How long a STREAMED reply from a pool peer may go silent before the forward
/// gives up (FUTURE_WORK #229).
///
/// The peer's own stream sends a keep-alive every `SSE_KEEPALIVE_INTERVAL_SECS`
/// while it loads, reads the prompt or writes, so four missed in a row mean it
/// has stopped. An inactivity timeout, as the provider proxy uses (gotcha #190):
/// the first-token budget used to be reqwest's REQUEST timeout here, which in
/// reqwest 0.12 runs "until the response body has finished", and cut a long
/// reply off mid-stream. reqwest's `read_timeout` also covers the wait for the
/// response headers, which a streaming peer sends at once.
const PEER_STREAM_IDLE_SECS: u64 = 4 * crate::api::SSE_KEEPALIVE_INTERVAL_SECS;

// The silence tolerated must outlast several keep-alives, or a healthy peer
// reading a long prompt is cut; the compiler refuses a build where it does not.
const _: () = assert!(PEER_STREAM_IDLE_SECS >= 3 * crate::api::SSE_KEEPALIVE_INTERVAL_SECS);

/// Slowest decode a NON-streamed forward allows for, per reply token. A
/// non-streamed reply sends nothing — not even its headers — until it is
/// complete, so silence proves nothing there; its bound is the first-token
/// budget plus the reply's length at this speed (0.5 tok/s, a slow processor
/// node), capped by [`NON_STREAMED_REPLY_CEILING_SECS`].
const NON_STREAMED_SECS_PER_TOKEN: u64 = 2;

/// Most a non-streamed reply's writing is allowed, whatever its `max_tokens` —
/// absent, a reply may run to the model's context.
const NON_STREAMED_REPLY_CEILING_SECS: u64 = 3600;

/// How long a forward may take: `None` for a streamed reply, which is bounded by
/// inactivity instead ([`PEER_STREAM_IDLE_SECS`], on [`PEER_STREAM_CLIENT`]);
/// for a non-streamed one, the time to its first token plus its writing at
/// [`NON_STREAMED_SECS_PER_TOKEN`].
fn forward_deadline(
    stream: bool,
    first_token_budget: std::time::Duration,
    max_tokens: Option<u32>,
) -> Option<std::time::Duration> {
    if stream {
        return None;
    }
    let writing = max_tokens
        .map(|t| u64::from(t).saturating_mul(NON_STREAMED_SECS_PER_TOKEN))
        .unwrap_or(NON_STREAMED_REPLY_CEILING_SECS)
        .min(NON_STREAMED_REPLY_CEILING_SECS);
    Some(first_token_budget + std::time::Duration::from_secs(writing))
}

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

/// Lazily-initialized shared reqwest client for NON-streamed peer forwards —
/// no client-wide total: each request carries [`forward_deadline`].
/// Avoids creating a new TLS + connection pool on every request.
static PEER_CLIENT: std::sync::LazyLock<reqwest::Client> = std::sync::LazyLock::new(|| {
    crate::http::build_client(|b| {
        b.connect_timeout(std::time::Duration::from_secs(
            PEER_FORWARD_CONNECT_TIMEOUT_SECS,
        ))
    })
});

/// The same for STREAMED forwards, bounded by inactivity alone. A second
/// client because reqwest sets `read_timeout` per client, and on the other one
/// it would cut a non-streamed reply, which is silent until it is complete.
static PEER_STREAM_CLIENT: std::sync::LazyLock<reqwest::Client> = std::sync::LazyLock::new(|| {
    crate::http::build_client(|b| {
        b.connect_timeout(std::time::Duration::from_secs(
            PEER_FORWARD_CONNECT_TIMEOUT_SECS,
        ))
        .read_timeout(std::time::Duration::from_secs(PEER_STREAM_IDLE_SECS))
    })
});

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
    peer: &super::resolver::PeerTarget,
    req: &ChatCompletionRequest,
    stream: bool,
    auth_header: Option<&str>,
) -> Result<axum::response::Response, ApiError> {
    let client: &reqwest::Client = if stream {
        &PEER_STREAM_CLIENT
    } else {
        &PEER_CLIENT
    };
    let url = format!("{}/v1/chat/completions", peer.url);

    // Reading the prompt (prefill) dominates the wait and grows with the prompt,
    // so a flat timeout here fails long prompts against a peer that is working
    // perfectly — the same defect that had to be fixed for the first-token
    // budget and again for the HTTP request timeout. Share that budget rather
    // than inventing a third rule. Bounded by its own ceiling, so a peer that
    // has genuinely gone away is still given up on.
    // Serialized length stands in for prompt size: it covers text and image
    // parts alike without another shape-matching helper, and it over-estimates
    // (JSON punctuation, base64) — the safe direction here, since the budget is
    // capped anyway and an image prompt genuinely is the expensive kind.
    let prompt_chars = serde_json::to_string(req).map(|s| s.len()).unwrap_or(0);
    let budget = crate::inference::pipeline::remote_generate::first_token_timeout(
        prompt_chars.div_ceil(2),
        peer.load,
    );

    let mut builder = client
        .post(&url)
        .header("x-swarm-forwarded", "true")
        .json(req);
    if let Some(deadline) = forward_deadline(stream, budget, req.max_tokens) {
        builder = builder.timeout(deadline);
    }
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

#[cfg(test)]
mod forward_deadline_tests {
    use super::*;
    use std::time::Duration;

    /// A streamed reply carries no total: the peer keeps it alive, and a total
    /// cut long replies off mid-stream (#229).
    #[test]
    fn a_streamed_forward_has_no_total_deadline() {
        assert_eq!(
            forward_deadline(true, Duration::from_secs(132), Some(16)),
            None
        );
        assert_eq!(forward_deadline(true, Duration::from_secs(132), None), None);
    }

    /// A non-streamed reply is silent until complete, so it is given its
    /// first token AND its writing — the budget alone cut a long reply.
    #[test]
    fn a_non_streamed_forward_is_given_time_to_write_its_reply() {
        let budget = Duration::from_secs(132);
        assert_eq!(
            forward_deadline(false, budget, Some(16)),
            Some(budget + Duration::from_secs(32))
        );
        // 1000 tokens on a slow node: the old bound (the budget) was 132 s.
        assert!(
            forward_deadline(false, budget, Some(1_000)).unwrap() >= Duration::from_secs(2_000)
        );
    }

    /// Bounded however the reply is sized, so a peer that never answers is
    /// still given up on.
    #[test]
    fn a_non_streamed_forward_is_capped() {
        let budget = Duration::from_secs(132);
        let cap = budget + Duration::from_secs(NON_STREAMED_REPLY_CEILING_SECS);
        assert_eq!(forward_deadline(false, budget, None), Some(cap));
        assert_eq!(forward_deadline(false, budget, Some(u32::MAX)), Some(cap));
    }
}
