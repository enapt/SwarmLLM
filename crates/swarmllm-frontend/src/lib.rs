//! SwarmLLM frontend asset serving.
//!
//! Two modes:
//! - **embedded** (default): Assets compiled into binary via `include_dir!`
//! - **dev**: Assets served from disk at runtime (zero recompile on CSS/JS changes)
//!
//! Usage:
//! ```bash
//! # Release: frontend baked into binary
//! cargo build --release
//!
//! # Development: frontend changes are instant (just refresh browser)
//! cargo run --features dev --no-default-features -- run
//! ```

use axum::http::{header, StatusCode};
use axum::response::{Html, IntoResponse, Response};

// ── Embedded mode (default): compile assets into binary ──

#[cfg(feature = "embedded")]
mod embedded_mode {
    use include_dir::{include_dir, Dir};
    pub(super) static FRONTEND_DIR: Dir<'_> = include_dir!("$CARGO_MANIFEST_DIR/../../frontend");
}

// ── Serve dashboard (index.html) ──

#[cfg(feature = "embedded")]
pub async fn serve_dashboard() -> Html<&'static str> {
    Html(get_file_content_embedded("index.html").unwrap_or("<!-- missing index.html -->"))
}

#[cfg(all(feature = "dev", not(feature = "embedded")))]
pub async fn serve_dashboard() -> Html<String> {
    let path = frontend_dir().join("index.html");
    match tokio::fs::read_to_string(&path).await {
        Ok(content) => Html(content),
        Err(e) => {
            tracing::warn!(path = %path.display(), error = %e, "Failed to read index.html from disk");
            Html("<!-- index.html not found — check SWARMLLM_FRONTEND_DIR -->".to_string())
        }
    }
}

// ── SPA catch-all ──

#[cfg(feature = "embedded")]
pub async fn serve_dashboard_catchall(
    axum::extract::Path(_path): axum::extract::Path<String>,
) -> Html<&'static str> {
    Html(get_file_content_embedded("index.html").unwrap_or("<!-- missing index.html -->"))
}

#[cfg(all(feature = "dev", not(feature = "embedded")))]
pub async fn serve_dashboard_catchall(
    axum::extract::Path(_path): axum::extract::Path<String>,
) -> Html<String> {
    serve_dashboard().await
}

// ── Static files ──

#[cfg(feature = "embedded")]
pub async fn serve_static(axum::extract::Path(path): axum::extract::Path<String>) -> Response {
    let path = path.trim_start_matches('/');
    match embedded_mode::FRONTEND_DIR.get_file(path) {
        Some(file) => {
            let mime = mime_type_for(path);
            (
                StatusCode::OK,
                [(header::CONTENT_TYPE, mime)],
                file.contents(),
            )
                .into_response()
        }
        None => (StatusCode::NOT_FOUND, "Not found").into_response(),
    }
}

#[cfg(all(feature = "dev", not(feature = "embedded")))]
pub async fn serve_static(axum::extract::Path(path): axum::extract::Path<String>) -> Response {
    let path_str = path.trim_start_matches('/');
    let full_path = frontend_dir().join(path_str);
    match tokio::fs::read(&full_path).await {
        Ok(bytes) => {
            let mime = mime_type_for(path_str);
            (StatusCode::OK, [(header::CONTENT_TYPE, mime)], bytes).into_response()
        }
        Err(_) => (StatusCode::NOT_FOUND, "Not found").into_response(),
    }
}

// ── Helpers ──

#[cfg(feature = "embedded")]
fn get_file_content_embedded(path: &str) -> Option<&'static str> {
    embedded_mode::FRONTEND_DIR
        .get_file(path)
        .and_then(|f| f.contents_utf8())
}

#[cfg(feature = "dev")]
fn frontend_dir() -> std::path::PathBuf {
    std::env::var("SWARMLLM_FRONTEND_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| {
            // Default: workspace root / frontend
            let manifest = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
            manifest.join("../../frontend")
        })
}

fn mime_type_for(path: &str) -> &'static str {
    if path.ends_with(".css") {
        "text/css; charset=utf-8"
    } else if path.ends_with(".js") {
        "application/javascript; charset=utf-8"
    } else if path.ends_with(".html") {
        "text/html; charset=utf-8"
    } else if path.ends_with(".json") {
        "application/json"
    } else if path.ends_with(".svg") {
        "image/svg+xml"
    } else if path.ends_with(".png") {
        "image/png"
    } else if path.ends_with(".ico") {
        "image/x-icon"
    } else {
        "application/octet-stream"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "embedded")]
    #[test]
    fn embedded_files_exist() {
        assert!(embedded_mode::FRONTEND_DIR.get_file("index.html").is_some());
        assert!(embedded_mode::FRONTEND_DIR
            .get_file("css/style.css")
            .is_some());
        assert!(embedded_mode::FRONTEND_DIR.get_file("js/app.js").is_some());
    }

    #[test]
    fn mime_types_correct() {
        assert_eq!(mime_type_for("style.css"), "text/css; charset=utf-8");
        assert_eq!(
            mime_type_for("app.js"),
            "application/javascript; charset=utf-8"
        );
        assert_eq!(mime_type_for("index.html"), "text/html; charset=utf-8");
        assert_eq!(mime_type_for("data.json"), "application/json");
    }
}
