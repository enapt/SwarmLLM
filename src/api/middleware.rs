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
    let mut origins: Vec<HeaderValue> = Vec::with_capacity(2);
    if let Ok(v) = format!("http://localhost:{port}").parse::<HeaderValue>() {
        origins.push(v);
    }
    if let Ok(v) = format!("http://127.0.0.1:{port}").parse::<HeaderValue>() {
        origins.push(v);
    }
    CorsLayer::new()
        .allow_origin(origins)
        .allow_methods([Method::GET, Method::POST, Method::PUT, Method::DELETE])
        .allow_headers([
            axum::http::header::CONTENT_TYPE,
            axum::http::header::AUTHORIZATION,
            axum::http::header::ACCEPT,
            axum::http::HeaderName::from_static("x-api-key"),
            axum::http::HeaderName::from_static("anthropic-version"),
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
    headers.insert(
        axum::http::header::CONTENT_SECURITY_POLICY,
        HeaderValue::from_static(
            "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; connect-src 'self' ws: wss:; img-src 'self' data: blob:; font-src 'self'; frame-ancestors 'none'; base-uri 'self'; form-action 'self'",
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
}

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
        let (kind, limit) = if path == "/api/admin/provider-model-status"
            || ((path == "/api/admin/providers" || path == "/api/admin/api-key") && is_mutating)
        {
            (BucketKind::SensitiveAdmin, 5)
        } else if path.starts_with("/api/admin/") {
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
        // When full, evict the oldest entry (LRU) instead of denying new clients.
        const MAX_RATE_BUCKETS: usize = 50_000;
        if !self.buckets.contains_key(&key) && self.buckets.len() >= MAX_RATE_BUCKETS {
            // SEC: Deny new clients when at capacity instead of O(n) LRU scan.
            // Previous O(n) DashMap iteration was a DoS amplification vector:
            // attacker with >50K unique IPs would cause O(n^2) total work.
            tracing::warn!(
                "Rate limiter at capacity ({MAX_RATE_BUCKETS} buckets) — denying new client"
            );
            return false;
        }

        let mut entry = self.buckets.entry(key).or_insert((limit, now));
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

/// Extract the IP address from a multiaddr string.
/// Handles both `/ip4/<addr>/...` and `/ip6/<addr>/...` formats.
fn extract_ip_from_multiaddr(multiaddr: &str) -> Option<String> {
    let parts: Vec<&str> = multiaddr.split('/').collect();
    // Multiaddr format: /ip4/<ip>/tcp/... → ["", "ip4", "<ip>", "tcp", ...]
    // or /ip6/<ip>/tcp/... → ["", "ip6", "<ip>", "tcp", ...]
    if parts.len() >= 3 && (parts[1] == "ip4" || parts[1] == "ip6") {
        Some(parts[2].to_string())
    } else {
        None
    }
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

    // /metrics is exempt but only from localhost to prevent remote recon
    if path == "/metrics" && is_loopback {
        return true;
    }

    // Read-only dashboard data endpoints — GET only, LOCALHOST only.
    // Remote clients (other machines on the network) must authenticate.
    // This prevents unauthenticated network reconnaissance.
    if *method == Method::GET && is_loopback {
        return matches!(
            path,
            "/api/admin/stats"
                | "/api/admin/config"
                | "/api/admin/models"
                | "/api/admin/peers"
                | "/api/admin/shard-storage"
                | "/api/admin/hf/search"
                | "/api/admin/hf/probe"
                | "/api/admin/network-map"
                | "/api/admin/schedule"
        ) || path.starts_with("/api/admin/hf/source/")
            || (path.starts_with("/api/admin/models/") && path.ends_with("/auto-manage"))
            || (path.starts_with("/api/admin/models/") && path.ends_with("/encrypted-pipeline"))
            || path.starts_with("/api/identity/")
            || path.starts_with("/api/pool/");
    }

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

    // Exempt frontend routes, health, read-only dashboard endpoints (loopback-gated)
    if is_exempt_request(&path, &method, addr.ip().is_loopback()) {
        return next.run(req).await;
    }

    // Exempt API key retrieval — loopback only.
    // The dashboard needs the key to bootstrap auth for all other requests.
    // Only accessible from localhost to prevent remote key theft.
    if path == "/api/admin/api-key" && method == Method::GET && addr.ip().is_loopback() {
        return next.run(req).await;
    }

    // Exempt WebSocket upgrade requests at /api/admin/ws — loopback only
    if path == "/api/admin/ws"
        && addr.ip().is_loopback()
        && req
            .headers()
            .get(axum::http::header::UPGRADE)
            .and_then(|v| v.to_str().ok())
            .map(|v| v.eq_ignore_ascii_case("websocket"))
            .unwrap_or(false)
    {
        return next.run(req).await;
    }

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
            // Loopback without valid token: fall through to normal Bearer auth
        } else if is_inference_path {
            // Non-loopback peer-forwarded requests: require BOTH known peer IP
            // AND valid internal auth token to prevent auth bypass via P2P membership
            let peer_ip = addr.ip().to_string();
            let is_known_peer = state.shared_state.peer_registry.iter().any(|entry| {
                entry.value().addresses.iter().any(|a| {
                    extract_ip_from_multiaddr(a)
                        .map(|ip| ip == peer_ip)
                        .unwrap_or(false)
                })
            });
            if is_known_peer {
                // Also require internal auth token for peer-forwarded requests
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
            // Unknown IP or missing internal token: fall through to normal auth
        }
    }

    let expected_key = &state.shared_state.api_key;

    // Extract Bearer token from Authorization header, or fall back to x-api-key header
    // (Anthropic SDK sends credentials via x-api-key instead of Authorization: Bearer)
    let auth_header = req
        .headers()
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok());

    let token = auth_header
        .and_then(|h| h.strip_prefix("Bearer "))
        .or_else(|| req.headers().get("x-api-key").and_then(|v| v.to_str().ok()));

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
                    "type": "auth_error",
                    "code": 401
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
    fn extract_ip_from_multiaddr_ipv4() {
        assert_eq!(
            extract_ip_from_multiaddr("/ip4/192.168.1.1/tcp/8810"),
            Some("192.168.1.1".to_string())
        );
    }

    #[test]
    fn extract_ip_from_multiaddr_ipv6() {
        assert_eq!(
            extract_ip_from_multiaddr("/ip6/::1/tcp/8810"),
            Some("::1".to_string())
        );
        assert_eq!(
            extract_ip_from_multiaddr("/ip6/fe80::1/tcp/8810/p2p/12D3abc"),
            Some("fe80::1".to_string())
        );
    }

    #[test]
    fn extract_ip_from_multiaddr_invalid() {
        assert_eq!(extract_ip_from_multiaddr("/dns4/example.com/tcp/80"), None);
        assert_eq!(extract_ip_from_multiaddr(""), None);
    }

    #[test]
    fn exempt_get_requests_loopback() {
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
        // /metrics exempt only from loopback
        assert!(is_exempt_request("/metrics", &get, true));
        assert!(!is_exempt_request("/metrics", &get, false));
        // Read-only dashboard endpoints — GET exempt from loopback only
        assert!(is_exempt_request("/api/admin/stats", &get, true));
        assert!(is_exempt_request("/api/admin/config", &get, true));
        assert!(is_exempt_request("/api/admin/models", &get, true));
        assert!(is_exempt_request("/api/admin/peers", &get, true));
        assert!(is_exempt_request("/api/admin/shard-storage", &get, true));
        assert!(is_exempt_request("/api/admin/hf/search", &get, true));
        assert!(is_exempt_request("/api/admin/hf/probe", &get, true));
        assert!(is_exempt_request("/api/admin/network-map", &get, true));
        assert!(is_exempt_request("/api/admin/schedule", &get, true));
        // provider-models requires auth (makes live API calls with stored keys)
        assert!(!is_exempt_request("/api/admin/provider-models", &get, true));
        assert!(is_exempt_request("/api/identity/nickname", &get, true));
        assert!(is_exempt_request("/api/pool/state", &get, true));
        // Same endpoints NOT exempt from remote IPs
        assert!(!is_exempt_request("/api/admin/stats", &get, false));
        assert!(!is_exempt_request("/api/admin/peers", &get, false));
        assert!(!is_exempt_request("/api/admin/network-map", &get, false));
        assert!(!is_exempt_request("/api/admin/models", &get, false));
        // Sensitive endpoints require auth even from loopback
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
    fn exempt_new_read_only_endpoints() {
        let get = Method::GET;
        // GET /api/admin/hf/source/:id is exempt from loopback
        assert!(is_exempt_request(
            "/api/admin/hf/source/test-model",
            &get,
            true
        ));
        assert!(!is_exempt_request(
            "/api/admin/hf/source/test-model",
            &get,
            false
        ));
        // GET /api/admin/models/:id/auto-manage is exempt from loopback
        assert!(is_exempt_request(
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
}
