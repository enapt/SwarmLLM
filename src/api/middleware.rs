use axum::extract::{ConnectInfo, Request, State};
use axum::http::{HeaderValue, Method, StatusCode};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use axum::Json;
use tower_http::cors::CorsLayer;

use crate::api::server::AppState;

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
        "/" | "/health" | "/health/ready" | "/metrics" | "/admin" | "/chat" | "/setup"
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
                | "/api/admin/api-key"
        ) || path.starts_with("/api/identity/")
            || path.starts_with("/api/pool/");
    }

    // POST /api/admin/join-network is exempt (frontend needs it, non-destructive)
    if *method == Method::POST && path == "/api/admin/join-network" {
        return true;
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

    // Exempt frontend routes, health, read-only dashboard endpoints
    if is_exempt_request(&path, &method) {
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

    // Exempt peer-forwarded requests — loopback only.
    // In production, peers forward via HTTP to the local API server.
    // Only trust x-swarm-forwarded from loopback to prevent external auth bypass.
    if addr.ip().is_loopback()
        && req
            .headers()
            .get("x-swarm-forwarded")
            .and_then(|v| v.to_str().ok())
            .map(|v| v == "true")
            .unwrap_or(false)
    {
        return next.run(req).await;
    }

    let expected_key = &state.shared_state.api_key;

    // Extract Bearer token from Authorization header
    let auth_header = req
        .headers()
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok());

    let token = auth_header.and_then(|h| h.strip_prefix("Bearer "));

    match token {
        Some(t) if t == expected_key => next.run(req).await,
        _ => {
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
        assert!(is_exempt_request("/api/identity/nickname", &get));
        assert!(is_exempt_request("/api/pool/state", &get));
    }

    #[test]
    fn exempt_post_join_network() {
        // POST /api/admin/join-network is exempt (frontend needs it)
        assert!(is_exempt_request("/api/admin/join-network", &Method::POST));
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
        // API key is readable from dashboard (CORS-restricted to localhost)
        assert!(is_exempt_request("/api/admin/api-key", &Method::GET));
        assert!(!is_exempt_request("/api/admin/shutdown", &post));
        assert!(!is_exempt_request("/api/admin/hf/download", &post));
        assert!(!is_exempt_request("/api/admin/hf/download-shards", &post));
        // OpenAI API always requires auth
        assert!(!is_exempt_request("/v1/models", &Method::GET));
        assert!(!is_exempt_request("/v1/chat/completions", &post));
        assert!(!is_exempt_request("/v1/completions", &post));
    }
}
