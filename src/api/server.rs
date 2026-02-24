use axum::routing::{get, post};
use axum::Router;

use crate::api::{middleware, openai};
use crate::config::Config;
use crate::inference::executor::SharedExecutor;
use crate::storage::db::Database;

/// Shared application state passed to all Axum handlers.
#[derive(Clone)]
pub struct AppState {
    pub config: Config,
    pub db: Database,
    pub executor: SharedExecutor,
}

/// Build the Axum router with all Phase 1 routes.
pub fn build_router(state: AppState) -> Router {
    Router::new()
        // OpenAI-compatible API
        .route("/v1/chat/completions", post(openai::chat_completions))
        .route("/v1/models", get(openai::list_models))
        // SwarmLLM extensions
        .route("/v1/status", get(openai::status))
        // Health check
        .route("/health", get(health))
        // Middleware
        .layer(middleware::cors_layer())
        .with_state(state)
}

async fn health() -> &'static str {
    "ok"
}

/// Start the Axum HTTP server on the configured port.
pub async fn run_server(
    config: Config,
    db: Database,
    executor: SharedExecutor,
) -> anyhow::Result<()> {
    let port = config.node.listen_port;
    let state = AppState {
        config,
        db,
        executor,
    };

    let app = build_router(state);
    let addr = std::net::SocketAddr::from(([0, 0, 0, 0], port));

    tracing::info!(%addr, "Starting HTTP server");

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
