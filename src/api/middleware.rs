use axum::extract::Request;
use axum::middleware::Next;
use axum::response::Response;
use tower_http::cors::CorsLayer;

/// Create a permissive CORS layer.
/// SwarmLLM serves on localhost so permissive CORS is acceptable.
pub fn cors_layer() -> CorsLayer {
    CorsLayer::permissive()
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
