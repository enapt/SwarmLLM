use std::sync::Arc;

use axum::response::Redirect;
use axum::routing::{get, post};
use axum::Router;
use tokio::sync::mpsc;

use crate::api::{admin, middleware, openai, websocket};
use crate::config::Config;
use crate::daemon::SharedState;
use crate::inference::executor::SharedExecutor;
use crate::inference::router::RouterCommand;
use crate::model::acquisition::AcquisitionCommand;
use crate::storage::db::Database;
use crate::ui::assets;

/// Shared application state passed to all Axum handlers.
#[derive(Clone)]
pub struct AppState {
    pub config: Config,
    pub db: Database,
    pub executor: SharedExecutor,
    /// Channel to submit inference requests to the InferenceRouter.
    pub router_tx: Option<mpsc::Sender<RouterCommand>>,
    /// Channel to submit model acquisition requests.
    pub acquisition_tx: Option<mpsc::Sender<AcquisitionCommand>>,
    /// Full daemon shared state for admin endpoints.
    pub shared_state: Arc<SharedState>,
}

/// Build the Axum router with all routes.
pub fn build_router(state: AppState) -> Router {
    Router::new()
        // OpenAI-compatible API
        .route("/v1/chat/completions", post(openai::chat_completions))
        .route("/v1/completions", post(openai::completions))
        .route("/v1/models", get(openai::list_models))
        // SwarmLLM extensions
        .route("/v1/status", get(openai::status))
        // Admin API
        .route("/api/admin/stats", get(admin::stats))
        .route(
            "/api/admin/config",
            get(admin::get_config).put(admin::update_config),
        )
        .route("/api/admin/models", get(admin::list_models))
        .route("/api/admin/models/:id/add", post(admin::add_model_interest))
        .route(
            "/api/admin/models/:id/status",
            get(admin::model_acquisition_status),
        )
        .route("/api/admin/peers", get(admin::list_peers))
        .route("/api/admin/credits", get(admin::credit_info))
        // Governance: Issues
        .route(
            "/api/admin/issues",
            get(admin::list_issues).post(admin::create_issue),
        )
        .route("/api/admin/issues/:hash", get(admin::get_issue))
        .route(
            "/api/admin/issues/:hash/comment",
            post(admin::add_issue_comment),
        )
        .route("/api/admin/issues/:hash/upvote", post(admin::upvote_issue))
        // Governance: Proposals
        .route(
            "/api/admin/proposals",
            get(admin::list_proposals).post(admin::create_proposal),
        )
        .route("/api/admin/proposals/:hash", get(admin::get_proposal))
        .route(
            "/api/admin/proposals/:hash/vote",
            post(admin::vote_proposal),
        )
        // Governance: Releases
        .route("/api/admin/releases", get(admin::list_releases))
        .route("/api/admin/releases/latest", get(admin::get_latest_release))
        // Governance: Role & Params
        .route("/api/admin/governance/role", get(admin::governance_role))
        .route(
            "/api/admin/governance/params",
            get(admin::governance_params),
        )
        // WebSocket
        .route("/api/admin/ws", get(websocket::handler))
        // Static files (embedded frontend)
        .route("/admin", get(assets::serve_dashboard))
        .route("/chat", get(assets::serve_chat))
        .route("/setup", get(assets::serve_setup))
        .route("/static/*path", get(assets::serve_static))
        // Root redirect
        .route("/", get(|| async { Redirect::to("/admin") }))
        // Health check
        .route("/health", get(health))
        // Middleware
        .layer(middleware::cors_layer())
        .with_state(state)
}

async fn health() -> &'static str {
    "ok"
}

/// Start the Axum HTTP server on the configured port (standalone mode).
pub async fn run_server(
    config: Config,
    db: Database,
    executor: SharedExecutor,
) -> anyhow::Result<()> {
    let port = config.node.listen_port;

    // Create a minimal SharedState for standalone mode
    let identity = crate::identity::Identity::generate();
    let (shared_state, _shutdown_rx) =
        SharedState::new(config.clone(), identity, db.clone(), executor.clone());

    let state = AppState {
        config,
        db,
        executor,
        router_tx: None,
        acquisition_tx: None,
        shared_state,
    };

    let app = build_router(state);
    let addr = std::net::SocketAddr::from(([0, 0, 0, 0], port));

    tracing::info!(%addr, "Starting HTTP server");

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

/// Start the Axum HTTP server using SharedState from the daemon.
pub async fn run_server_with_state(
    shared_state: Arc<SharedState>,
    router_tx: mpsc::Sender<RouterCommand>,
    acquisition_tx: mpsc::Sender<AcquisitionCommand>,
) -> anyhow::Result<()> {
    let port = shared_state.config.node.listen_port;
    let state = AppState {
        config: shared_state.config.clone(),
        db: shared_state.db.clone(),
        executor: shared_state.executor.clone(),
        router_tx: Some(router_tx),
        acquisition_tx: Some(acquisition_tx),
        shared_state,
    };

    let app = build_router(state);
    let addr = std::net::SocketAddr::from(([0, 0, 0, 0], port));

    tracing::info!(%addr, "Starting HTTP server");

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
