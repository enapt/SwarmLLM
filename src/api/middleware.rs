use axum::extract::Request;
use axum::http::{HeaderValue, Method};
use axum::middleware::Next;
use axum::response::Response;
use tower_http::cors::CorsLayer;

/// Create a CORS layer restricted to localhost origins and specific methods/headers.
pub fn cors_layer() -> CorsLayer {
    let origins = [
        "http://localhost:8800".parse::<HeaderValue>().unwrap(),
        "http://127.0.0.1:8800".parse::<HeaderValue>().unwrap(),
        "http://localhost:8801".parse::<HeaderValue>().unwrap(),
        "http://127.0.0.1:8801".parse::<HeaderValue>().unwrap(),
        "http://localhost:8802".parse::<HeaderValue>().unwrap(),
        "http://127.0.0.1:8802".parse::<HeaderValue>().unwrap(),
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
