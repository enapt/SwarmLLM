use axum::http::{header, StatusCode};
use axum::response::{Html, IntoResponse, Response};
use include_dir::{include_dir, Dir};

/// Embedded frontend directory (compiled into the binary).
static FRONTEND_DIR: Dir<'_> = include_dir!("$CARGO_MANIFEST_DIR/frontend");

/// Serve the main single-page application (index.html).
/// All page routes (/, /admin, /chat, /setup) serve the same HTML.
pub async fn serve_dashboard() -> Html<&'static str> {
    Html(get_file_content("index.html").unwrap_or("<!-- missing index.html -->"))
}

/// Serve static files from the embedded frontend directory.
/// Handles paths like `/static/css/style.css` or `/static/js/app.js`.
pub async fn serve_static(axum::extract::Path(path): axum::extract::Path<String>) -> Response {
    let path = path.trim_start_matches('/');

    match FRONTEND_DIR.get_file(path) {
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

fn get_file_content(path: &str) -> Option<&'static str> {
    FRONTEND_DIR.get_file(path).and_then(|f| f.contents_utf8())
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

    #[test]
    fn embedded_files_exist() {
        assert!(FRONTEND_DIR.get_file("index.html").is_some());
        assert!(FRONTEND_DIR.get_file("css/style.css").is_some());
        assert!(FRONTEND_DIR.get_file("js/app.js").is_some());
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
