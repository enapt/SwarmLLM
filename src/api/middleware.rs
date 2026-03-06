use axum::extract::{ConnectInfo, Request, State};
use axum::http::{HeaderValue, Method, StatusCode};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use axum::Json;
use tower_http::cors::CorsLayer;

use crate::api::server::AppState;

/// Constant-time byte comparison to prevent timing side-channel attacks on API key validation.
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
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
    response
}

/// Request logging middleware using tracing.
pub async fn request_logger(req: Request, next: Next) -> Response {
    let method = req.method().clone();
    let uri = req.uri().clone();
    let start = std::time::Instant::now();

    let response = next.run(req).await;

    let elapsed = start.elapsed();
    tracing::info!(
        method = %method,
        uri = %uri,
        status = %response.status(),
        latency_ms = elapsed.as_millis(),
        "Request handled"
    );

    response
}

/// Check whether a request is exempt from Bearer token authentication.
///
/// Only GET requests to read-only dashboard endpoints are exempt.
/// All state-mutating operations (POST/PUT/DELETE) require auth, with
/// the exception of `/api/admin/join-network` (POST, but non-destructive
/// and needed by the frontend without auth).
///
/// Frontend routes, health checks, and static assets are always exempt.
/// The Bearer token protects external-facing endpoints (`/v1/...`)
/// and all sensitive/destructive operations.
fn is_exempt_request(path: &str, method: &Method) -> bool {
    // Frontend routes, health checks, static assets — always exempt
    if matches!(
        path,
        "/"
            | "/health"
            | "/health/ready"
            | "/metrics"
            | "/admin"
            | "/chat"
            | "/setup"
            | "/favicon.ico"
    ) || path.starts_with("/static/")
    {
        return true;
    }

    // Read-only dashboard endpoints — GET only
    if *method == Method::GET {
        return matches!(
            path,
            "/api/admin/stats"
                | "/api/admin/config"
                | "/api/admin/models"
                | "/api/admin/peers"
                | "/api/admin/credits"
                | "/api/admin/shard-storage"
                | "/api/admin/hf/search"
                | "/api/admin/hf/probe"
                | "/api/admin/network-map"
                | "/api/admin/network-code"
                | "/api/admin/providers"
                | "/api/admin/provider-models"
                | "/api/admin/schedule"
        ) || path.starts_with("/api/admin/hf/source/")
            || (path.starts_with("/api/admin/models/") && path.ends_with("/auto-manage"))
            || path.starts_with("/api/identity/")
            || path.starts_with("/api/pool/");
    }

    // POST /api/admin/join-network writes to peer cache — require auth
    // (frontend sends the API key from localStorage after setup)

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

    // Exempt frontend routes, health, read-only dashboard endpoints
    if is_exempt_request(&path, &method) {
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
    // Replaces the old x-swarm-forwarded header check which was guessable by any localhost process.
    if addr.ip().is_loopback() {
        if let Some(token) = req
            .headers()
            .get("x-swarm-internal-token")
            .and_then(|v| v.to_str().ok())
        {
            if token == state.shared_state.internal_auth_token {
                return next.run(req).await;
            }
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
    fn exempt_get_requests() {
        let get = Method::GET;
        // Frontend routes, health, static — always exempt
        assert!(is_exempt_request("/", &get));
        assert!(is_exempt_request("/health", &get));
        assert!(is_exempt_request("/health/ready", &get));
        assert!(is_exempt_request("/metrics", &get));
        assert!(is_exempt_request("/admin", &get));
        assert!(is_exempt_request("/chat", &get));
        assert!(is_exempt_request("/setup", &get));
        assert!(is_exempt_request("/static/css/style.css", &get));
        assert!(is_exempt_request("/static/js/app.js", &get));
        assert!(is_exempt_request("/favicon.ico", &get));
        // Read-only dashboard endpoints — GET exempt
        assert!(is_exempt_request("/api/admin/stats", &get));
        assert!(is_exempt_request("/api/admin/config", &get));
        assert!(is_exempt_request("/api/admin/models", &get));
        assert!(is_exempt_request("/api/admin/peers", &get));
        assert!(is_exempt_request("/api/admin/credits", &get));
        assert!(is_exempt_request("/api/admin/shard-storage", &get));
        assert!(is_exempt_request("/api/admin/hf/search", &get));
        assert!(is_exempt_request("/api/admin/hf/probe", &get));
        assert!(is_exempt_request("/api/admin/network-map", &get));
        assert!(is_exempt_request("/api/admin/network-code", &get));
        assert!(is_exempt_request("/api/admin/schedule", &get));
        assert!(is_exempt_request("/api/admin/provider-models", &get));
        assert!(is_exempt_request("/api/identity/nickname", &get));
        assert!(is_exempt_request("/api/pool/state", &get));
    }

    #[test]
    fn non_exempt_post_join_network() {
        // POST /api/admin/join-network requires auth (writes to peer cache)
        assert!(!is_exempt_request("/api/admin/join-network", &Method::POST));
    }

    #[test]
    fn non_exempt_mutations() {
        let put = Method::PUT;
        let post = Method::POST;
        let delete = Method::DELETE;
        // PUT /api/admin/config requires auth (C4 fix)
        assert!(!is_exempt_request("/api/admin/config", &put));
        // POST /api/pool/* require auth (C5 fix)
        assert!(!is_exempt_request("/api/pool/create", &post));
        assert!(!is_exempt_request("/api/pool/invite", &post));
        assert!(!is_exempt_request("/api/pool/leave", &post));
        assert!(!is_exempt_request("/api/pool/remove", &post));
        assert!(!is_exempt_request("/api/pool/accept", &post));
        // PUT /api/identity/nickname requires auth
        assert!(!is_exempt_request("/api/identity/nickname", &put));
        assert!(!is_exempt_request("/api/identity/nickname", &delete));
        // API key not in general exempt list (loopback-only exemption in auth_middleware)
        assert!(!is_exempt_request("/api/admin/api-key", &Method::GET));
        assert!(!is_exempt_request("/api/admin/shutdown", &post));
        assert!(!is_exempt_request("/api/admin/hf/download", &post));
        assert!(!is_exempt_request("/api/admin/hf/download-shards", &post));
        // OpenAI API always requires auth
        assert!(!is_exempt_request("/v1/models", &Method::GET));
        assert!(!is_exempt_request("/v1/chat/completions", &post));
        // PUT auto-manage and DELETE shard require auth
        assert!(!is_exempt_request(
            "/api/admin/models/test/auto-manage",
            &put
        ));
        assert!(!is_exempt_request(
            "/api/admin/models/test/shards/0",
            &delete
        ));
    }

    #[test]
    fn exempt_new_read_only_endpoints() {
        let get = Method::GET;
        // GET /api/admin/hf/source/:id is exempt (read-only)
        assert!(is_exempt_request("/api/admin/hf/source/test-model", &get));
        // GET /api/admin/models/:id/auto-manage is exempt (read-only)
        assert!(is_exempt_request(
            "/api/admin/models/test-model/auto-manage",
            &get
        ));
    }
}
