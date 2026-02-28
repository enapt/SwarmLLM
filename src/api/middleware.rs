use axum::extract::{Request, State};
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

/// Paths exempt from Bearer token authentication.
/// Frontend routes, health checks, static assets, and admin dashboard APIs
/// are exempt — they're for the embedded UI and already protected by CORS
/// (localhost only). The Bearer token protects external-facing endpoints
/// like the OpenAI-compatible inference API (`/v1/...`).
/// Note: `/api/admin/api-key` requires auth (returns the raw key).
fn is_exempt_path(path: &str) -> bool {
    // Frontend routes, health checks, static assets are exempt.
    // All /v1/* routes require Bearer token (OpenAI-compatible API).
    // Sensitive /api/admin/* endpoints require auth; only safe read-only
    // dashboard endpoints are exempt.
    matches!(
        path,
        "/" | "/health" | "/health/ready" | "/metrics" | "/admin" | "/chat" | "/setup"
    ) || path.starts_with("/static/")
        || matches!(
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
        )
        || path.starts_with("/api/identity/")
        || path.starts_with("/api/pool/")
}

/// Bearer token authentication middleware.
///
/// Checks the `Authorization: Bearer <token>` header against the stored API key.
/// Exempt paths (frontend routes, health, static assets, WebSocket upgrades) are
/// passed through without authentication.
pub async fn auth_middleware(State(state): State<AppState>, req: Request, next: Next) -> Response {
    let path = req.uri().path().to_string();

    // Exempt frontend routes, health, and static assets
    if is_exempt_path(&path) {
        return next.run(req).await;
    }

    // Exempt WebSocket upgrade requests at /api/admin/ws
    if path == "/api/admin/ws"
        && req
            .headers()
            .get(axum::http::header::UPGRADE)
            .and_then(|v| v.to_str().ok())
            .map(|v| v.eq_ignore_ascii_case("websocket"))
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
    fn exempt_paths() {
        assert!(is_exempt_path("/"));
        assert!(is_exempt_path("/health"));
        assert!(is_exempt_path("/admin"));
        assert!(is_exempt_path("/chat"));
        assert!(is_exempt_path("/setup"));
        assert!(is_exempt_path("/static/css/style.css"));
        assert!(is_exempt_path("/static/js/app.js"));
        assert!(is_exempt_path("/api/admin/stats"));
        assert!(is_exempt_path("/api/admin/config"));
        assert!(is_exempt_path("/api/admin/shard-storage"));
        assert!(is_exempt_path("/api/admin/peers"));
        assert!(is_exempt_path("/api/admin/credits"));
        assert!(is_exempt_path("/api/admin/network-map"));
        assert!(is_exempt_path("/api/admin/hf/search"));
        assert!(is_exempt_path("/api/admin/hf/probe"));
        assert!(is_exempt_path("/api/identity/nickname"));
        assert!(is_exempt_path("/api/pool/state"));
        assert!(is_exempt_path("/health/ready"));
        assert!(is_exempt_path("/metrics"));
    }

    #[test]
    fn non_exempt_paths() {
        assert!(!is_exempt_path("/api/admin/api-key")); // sensitive — requires auth
        assert!(!is_exempt_path("/api/admin/shutdown")); // destructive — requires auth
        assert!(!is_exempt_path("/api/admin/hf/download")); // write op — requires auth
        assert!(!is_exempt_path("/api/admin/hf/download-shards")); // write op — requires auth
        assert!(!is_exempt_path("/v1/models")); // OpenAI API — requires auth
        assert!(!is_exempt_path("/v1/chat/completions")); // OpenAI API — requires auth
        assert!(!is_exempt_path("/v1/completions")); // OpenAI API — requires auth
    }
}
