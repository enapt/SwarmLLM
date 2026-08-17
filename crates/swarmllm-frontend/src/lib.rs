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
use axum::response::{IntoResponse, Response};

// ── Embedded mode (default): compile assets into binary ──

#[cfg(feature = "embedded")]
mod embedded_mode {
    use include_dir::{include_dir, Dir};
    pub(super) static FRONTEND_DIR: Dir<'_> = include_dir!("$CARGO_MANIFEST_DIR/../../frontend");
}

// ── Serve dashboard (index.html) ──
//
// All dashboard routes go through `api/server.rs::serve_dashboard_with_nonce`
// in the main crate, which substitutes a per-page bootstrap nonce into the
// served HTML. That handler reads the raw HTML via `dashboard_html_owned`
// below; the previous `serve_dashboard` / `serve_dashboard_catchall`
// handlers (which returned the HTML unchanged) are obsolete.

/// Read the dashboard HTML as an owned `String`. Used by the main crate's
/// nonce-injecting wrapper to template a per-page bootstrap value into
/// the `<meta name="bootstrap-nonce">` tag before responding.
#[cfg(feature = "embedded")]
pub async fn dashboard_html_owned() -> String {
    get_file_content_embedded("index.html")
        .unwrap_or("<!-- missing index.html -->")
        .to_string()
}

#[cfg(all(feature = "dev", not(feature = "embedded")))]
pub async fn dashboard_html_owned() -> String {
    let path = frontend_dir().join("index.html");
    match tokio::fs::read_to_string(&path).await {
        Ok(content) => content,
        Err(e) => {
            tracing::warn!(path = %path.display(), error = %e, "Failed to read index.html from disk");
            "<!-- index.html not found — check SWARMLLM_FRONTEND_DIR -->".to_string()
        }
    }
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
    // SEC: this route is auth-exempt (see middleware.rs `is_exempt_request`
    // for `/static/`). Without containment, joining the caller-controlled
    // `path` onto `frontend_dir()` lets `..` segments escape the workspace
    // and read arbitrary files (e.g. `/static/../../../etc/passwd`). axum's
    // `Path` extractor does NOT canonicalize `..`. Reject paths whose
    // components include `..`, absolute roots, or NUL — and double-check
    // by canonicalizing the result and verifying it stays under
    // `frontend_dir()`. Embedded mode (production) is immune because
    // `include_dir!` resolves at compile time against a fixed tree.
    let path_str = path.trim_start_matches('/');
    if path_str.is_empty()
        || path_str.contains("..")
        || path_str.contains('\0')
        || path_str.starts_with('/')
        || std::path::Path::new(path_str).is_absolute()
        || std::path::Path::new(path_str).components().any(|c| {
            matches!(
                c,
                std::path::Component::ParentDir | std::path::Component::RootDir
            )
        })
    {
        return (StatusCode::NOT_FOUND, "Not found").into_response();
    }
    let base = match frontend_dir().canonicalize() {
        Ok(b) => b,
        Err(_) => return (StatusCode::NOT_FOUND, "Not found").into_response(),
    };
    let full_path = base.join(path_str);
    let resolved = match full_path.canonicalize() {
        Ok(p) => p,
        Err(_) => return (StatusCode::NOT_FOUND, "Not found").into_response(),
    };
    if !resolved.starts_with(&base) {
        return (StatusCode::NOT_FOUND, "Not found").into_response();
    }
    match tokio::fs::read(&resolved).await {
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
    } else if path.ends_with(".woff2") {
        "font/woff2"
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
        assert!(embedded_mode::FRONTEND_DIR.get_file("js/init.js").is_some());
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
        // Served as octet-stream a font still loads in every current browser,
        // so a wrong type here fails silently rather than visibly — which is
        // exactly why it is asserted.
        assert_eq!(
            mime_type_for("fonts/ibm-plex-mono-latin-400-normal.woff2"),
            "font/woff2"
        );
    }

    /// The bundled typefaces are the whole reason the dashboard renders the
    /// same on every machine. Losing one silently falls back to a system font
    /// and the regression looks like "it just looks a bit different".
    #[cfg(feature = "embedded")]
    #[test]
    fn bundled_fonts_are_embedded() {
        for f in [
            "fonts/ibm-plex-sans-latin-wght-normal.woff2",
            "fonts/ibm-plex-mono-latin-400-normal.woff2",
            "fonts/ibm-plex-mono-latin-600-normal.woff2",
            "fonts/ibm-plex-mono-latin-700-normal.woff2",
        ] {
            assert!(
                embedded_mode::FRONTEND_DIR.get_file(f).is_some(),
                "{f} is referenced by style.css but is not embedded in the binary"
            );
        }
    }
}
