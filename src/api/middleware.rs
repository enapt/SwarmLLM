use axum::extract::{Request, State};
use axum::http::{HeaderValue, Method, StatusCode};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use axum::Json;
use tower_http::cors::CorsLayer;

use crate::api::server::AppState;

/// Create a CORS layer restricted to localhost origins and specific methods/headers.
pub fn cors_layer() -> CorsLayer {
    let origins = [
        "http://localhost:8800".parse::<HeaderValue>().unwrap(),
        "http://127.0.0.1:8800".parse::<HeaderValue>().unwrap(),
        "http://localhost:8801".parse::<HeaderValue>().unwrap(),
        "http://127.0.0.1:8801".parse::<HeaderValue>().unwrap(),
        "http://localhost:8802".parse::<HeaderValue>().unwrap(),
        "http://127.0.0.1:8802".parse::<HeaderValue>().unwrap(),
        "http://localhost:8803".parse::<HeaderValue>().unwrap(),
        "http://127.0.0.1:8803".parse::<HeaderValue>().unwrap(),
    ];
    CorsLayer::new()
        .allow_origin(origins.to_vec())
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
    // api-key endpoint must require auth — it returns the raw key
    if path == "/api/admin/api-key" {
        return false;
    }
    matches!(
        path,
        "/" | "/health" | "/admin" | "/chat" | "/setup"
    ) || path.starts_with("/v1/")
        || path.starts_with("/static/")
        || path.starts_with("/api/admin/")
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
        assert!(is_exempt_path("/api/identity/nickname"));
        assert!(is_exempt_path("/api/pool/state"));
        assert!(is_exempt_path("/v1/models"));
        assert!(is_exempt_path("/v1/chat/completions"));
        assert!(is_exempt_path("/v1/completions"));
    }

    #[test]
    fn non_exempt_paths() {
        assert!(!is_exempt_path("/api/admin/api-key")); // sensitive — requires auth
    }
}
