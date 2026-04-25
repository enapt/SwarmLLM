use std::sync::Arc;

use axum::extract::rejection::JsonRejection;
use axum::extract::{DefaultBodyLimit, FromRequest, Request};
use axum::response::{IntoResponse, Redirect};
use axum::routing::{delete, get, post, put};
use axum::Json;
use axum::Router;
use serde::de::DeserializeOwned;
use tokio::sync::mpsc;

use crate::api::{
    admin, anthropic, identity, mcp, metrics, middleware, openai, pool, providers, websocket,
};
use crate::config::Config;
use crate::daemon::SharedState;
use crate::inference::executor::SharedExecutor;
use crate::inference::router::RouterCommand;
use crate::model::acquisition::AcquisitionCommand;
use crate::storage::db::Database;
use crate::types::NetworkCommand;
use swarmllm_frontend as assets;

/// HTTP request processing timeout (kills stalled connections).
const REQUEST_TIMEOUT_SECS: u64 = 300;
/// Rate-limiter cleanup interval.
const RATE_LIMIT_CLEANUP_INTERVAL_SECS: u64 = 300;
/// Rate-limiter bucket TTL — entries older than this are evicted.
const RATE_LIMIT_BUCKET_TTL_SECS: u64 = 600;

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
    /// Channel to broadcast messages to the P2P network.
    pub network_tx: Option<mpsc::Sender<NetworkCommand>>,
    /// Full daemon shared state for admin endpoints.
    pub shared_state: Arc<SharedState>,
    /// IP-based rate limiter.
    pub rate_limiter: middleware::RateLimiter,
}

impl AppState {
    /// Resolve the on-disk directory for a model (proxy to
    /// [`SharedState::model_dir`]). Keeps handler code free of the
    /// `state.config.node.data_dir` reach-through.
    pub fn model_dir(&self, model_id: &str) -> std::path::PathBuf {
        self.shared_state.model_dir(model_id)
    }
}

/// Custom JSON extractor that returns OpenAI-format error responses on parse failure.
///
/// Axum's built-in `Json<T>` returns raw text on deserialization errors.
/// This wrapper converts those into proper `{"error": {...}}` JSON responses.
pub struct JsonBody<T>(pub T);

impl<S, T> FromRequest<S> for JsonBody<T>
where
    T: DeserializeOwned,
    S: Send + Sync,
    axum::Json<T>: FromRequest<S, Rejection = JsonRejection>,
{
    type Rejection = axum::response::Response;

    async fn from_request(req: Request, state: &S) -> Result<Self, Self::Rejection> {
        match axum::Json::<T>::from_request(req, state).await {
            Ok(axum::Json(value)) => Ok(JsonBody(value)),
            Err(rejection) => {
                let message = rejection.body_text();
                let body = serde_json::json!({
                    "error": {
                        "message": message,
                        "type": "invalid_request_error",
                        "param": null,
                        "code": "invalid_request_error"
                    }
                });
                Err((axum::http::StatusCode::BAD_REQUEST, axum::Json(body)).into_response())
            }
        }
    }
}

/// Build the Axum router with all routes.
pub fn build_router(state: AppState) -> Router {
    Router::new()
        // OpenAI-compatible API
        .route("/v1/chat/completions", post(openai::chat_completions))
        .route("/v1/embeddings", post(openai::embeddings))
        .route("/v1/models", get(openai::list_models))
        // OpenAI Responses API (gpt-5 / o-series default)
        .route("/v1/responses", post(openai::responses::create_response))
        .route(
            "/v1/responses/{id}",
            get(openai::responses::background::get_response_maybe_stream)
                .delete(openai::responses::delete_response),
        )
        .route(
            "/v1/responses/{id}/cancel",
            post(openai::responses::cancel_response),
        )
        .route(
            "/v1/responses/{id}/input_items",
            get(openai::responses::list_input_items),
        )
        // Anthropic Messages API
        .route("/v1/messages", post(anthropic::messages))
        // Provider listing
        .route("/v1/providers", get(providers::list_providers))
        // SwarmLLM extensions
        .route("/v1/status", get(openai::status))
        // Admin API
        .route("/api/admin/stats", get(admin::stats))
        .route("/api/admin/responses", get(admin::list_responses))
        .route(
            "/api/admin/config",
            get(admin::get_config).put(admin::update_config),
        )
        .route("/api/admin/config/reload", post(admin::reload_config))
        .route("/api/admin/models", get(admin::list_models))
        .route(
            "/api/admin/models/{id}/add",
            post(admin::add_model_interest),
        )
        .route(
            "/api/admin/models/{id}/status",
            get(admin::model_acquisition_status),
        )
        .route("/api/admin/peers", get(admin::list_peers))
        .route("/api/admin/credits", get(admin::credit_info))
        // Shard storage info
        .route("/api/admin/shard-storage", get(admin::shard_storage))
        // HuggingFace model browsing
        .route("/api/admin/hf/search", get(admin::hf_search))
        .route("/api/admin/hf/probe", get(admin::hf_probe))
        .route("/api/admin/hf/download", post(admin::hf_download))
        .route(
            "/api/admin/hf/download-shards",
            post(admin::hf_download_shards),
        )
        // Rescan local shard files (hot-reload without restart)
        .route("/api/admin/rescan-shards", post(admin::rescan_shards))
        // Network map (heatmap data)
        .route("/api/admin/network-map", get(admin::network_map))
        // Download management
        .route("/api/admin/downloads", get(admin::download_queue))
        .route(
            "/api/admin/downloads/{model_id}/cancel",
            post(admin::cancel_download),
        )
        // GGUF metadata browser
        .route(
            "/api/admin/models/{id}/metadata",
            get(admin::model_metadata),
        )
        .route(
            "/api/admin/models/{id}/pipeline-plan",
            get(admin::pipeline_plan),
        )
        // Single-shard management
        .route(
            "/api/admin/models/{id}/shards/{index}",
            delete(admin::delete_shard),
        )
        .route(
            "/api/admin/models/{id}/shards/{index}/download",
            post(admin::download_shard),
        )
        // Per-model auto-manage policy
        .route(
            "/api/admin/models/{id}/auto-manage",
            get(admin::get_model_auto_manage).put(admin::set_model_auto_manage),
        )
        // Per-model encrypted pipeline toggle
        .route(
            "/api/admin/models/{id}/encrypted-pipeline",
            get(admin::get_model_encrypted_pipeline).put(admin::set_model_encrypted_pipeline),
        )
        // Resource schedule
        .route(
            "/api/admin/schedule",
            get(admin::get_schedule).put(admin::update_schedule),
        )
        // Prune history
        .route("/api/admin/prune-history", get(admin::prune_history))
        // Shard lock
        .route(
            "/api/admin/models/{id}/shards/{index}/lock",
            put(admin::lock_shard),
        )
        // Shard unload/load (memory management, keep files)
        .route(
            "/api/admin/models/{id}/shards/{index}/unload",
            post(admin::unload_shard),
        )
        .route(
            "/api/admin/models/{id}/shards/{index}/load",
            post(admin::load_shard),
        )
        // HF source lookup
        .route("/api/admin/hf/source/{model_id}", get(admin::hf_source))
        // Model management
        .route("/api/admin/models/{id}", delete(admin::delete_model))
        .route("/api/admin/models/{id}/unload", post(admin::unload_model))
        // LoRA adapters
        .route(
            "/api/admin/adapters",
            get(admin::list_adapters).post(admin::register_adapter),
        )
        .route("/api/admin/adapters/{id}", delete(admin::delete_adapter))
        // Cloud provider configuration
        .route(
            "/api/admin/providers",
            get(admin::get_providers).put(admin::update_providers),
        )
        .route(
            "/api/admin/provider-models",
            get(admin::list_provider_models),
        )
        .route("/api/admin/provider-health", get(admin::provider_health))
        .route(
            "/api/admin/provider-model-status",
            post(admin::provider_model_status),
        )
        // Claude subscription status (feature-gated)
        .route(
            "/api/admin/claude-subscription/status",
            get({
                #[cfg(feature = "claude-subscription")]
                {
                    crate::api::claude_sub::get_status
                }
                #[cfg(not(feature = "claude-subscription"))]
                {
                    || async {
                        axum::Json(
                            serde_json::json!({"error": "claude-subscription feature not enabled"}),
                        )
                    }
                }
            }),
        )
        // Claude Code sessions (feature-gated — all routes added only when feature is enabled)
        .merge({
            #[cfg(feature = "claude-subscription")]
            {
                Router::new()
                    .route(
                        "/api/claude-code/sessions",
                        get(crate::api::claude_session::list_sessions_handler),
                    )
                    .route(
                        "/api/claude-code/session",
                        post(crate::api::claude_session::create_session_handler),
                    )
                    .route(
                        "/api/claude-code/session/{id}",
                        get(crate::api::claude_session::get_session_handler)
                            .delete(crate::api::claude_session::close_session_handler),
                    )
                    .route(
                        "/api/claude-code/session/{id}/message",
                        post(crate::api::claude_session::send_message_handler),
                    )
                    .route(
                        "/api/claude-code/session/{id}/permission",
                        post(crate::api::claude_session::permission_handler),
                    )
                    .with_state(state.clone())
            }
            #[cfg(not(feature = "claude-subscription"))]
            {
                Router::new()
            }
        })
        // Version & Updates
        .route("/api/admin/version", get(admin::version_info))
        .route("/api/admin/update/check", post(admin::check_update))
        .route("/api/admin/update/apply", post(admin::apply_update))
        // Shutdown
        .route("/api/admin/shutdown", post(admin::shutdown_node))
        // API key (requires auth)
        .route("/api/admin/api-key", get(admin::get_api_key))
        // Network discovery (invite codes)
        .route("/api/admin/network-code", get(admin::network_code))
        .route("/api/admin/join-network", post(admin::join_network))
        // Identity & Nickname
        .route(
            "/api/identity/nickname",
            get(identity::get_nickname)
                .put(identity::set_nickname)
                .delete(identity::delete_nickname),
        )
        .route("/api/identity/leaderboard", get(identity::leaderboard))
        .route("/api/identity/peers", get(identity::peers_with_names))
        // Device Pool
        .route("/api/pool/state", get(pool::pool_state))
        .route("/api/pool/create", post(pool::pool_create))
        .route("/api/pool/invite", post(pool::pool_invite))
        .route("/api/pool/accept", post(pool::pool_accept))
        .route("/api/pool/remove", post(pool::pool_remove))
        .route("/api/pool/leave", post(pool::pool_leave))
        .route("/api/pool/invitations", get(pool::pool_invitations))
        .route("/api/pool/leaderboard", get(pool::pool_leaderboard))
        .route("/api/pool/generate-code", post(pool::pool_generate_code))
        .route("/api/pool/join", post(pool::pool_join))
        .route("/api/pool/device-name", post(pool::pool_set_device_name))
        .route("/api/pool/credit-split", put(pool::pool_set_credit_split))
        .route("/api/pool/contribution", put(pool::pool_set_contribution))
        // Private mode
        .route(
            "/api/pool/private-mode",
            get(pool::get_private_mode).put(pool::set_private_mode),
        )
        .route("/api/pool/coverage", get(pool::pool_coverage))
        // Shard pinning
        .route("/api/pool/pins", get(pool::pool_pins))
        .route(
            "/api/pool/pin",
            post(pool::pool_add_pin).delete(pool::pool_remove_pin),
        )
        // Pool credit rates
        .route(
            "/api/admin/pools/{id}/rates",
            get(pool::pool_rates_get).put(pool::pool_rates_set),
        )
        // WebSocket (Bearer-authed via short-lived ticket in ?t=<hex>)
        .route("/api/admin/ws", get(websocket::handler))
        .route("/api/admin/ws-ticket", post(websocket::issue_ticket))
        // Static files (embedded frontend)
        // SPA catch-all: serve index.html for all frontend sub-routes
        // so direct URL access (bookmarks, refresh) works
        .route("/admin", get(assets::serve_dashboard))
        .route("/admin/{*path}", get(assets::serve_dashboard_catchall))
        .route("/chat", get(assets::serve_dashboard))
        .route("/chat/{*path}", get(assets::serve_dashboard_catchall))
        .route("/setup", get(assets::serve_dashboard))
        .route("/static/{*path}", get(assets::serve_static))
        // Root redirect
        .route("/", get(|| async { Redirect::to("/admin") }))
        // Health check
        .route("/health", get(health))
        .route("/health/ready", get(metrics::health_ready))
        // MCP (Model Context Protocol) server — Streamable HTTP transport
        .route(
            "/mcp",
            post(mcp::handle_mcp)
                .get(mcp::handle_mcp_get)
                .delete(mcp::handle_mcp_delete),
        )
        // Prometheus metrics
        .route("/metrics", get(metrics::metrics))
        // Middleware (layers run bottom-to-top: timeout, CORS, security headers, rate limit, auth, body limit, handler)
        .layer(DefaultBodyLimit::max(32 * 1024 * 1024)) // 32MB request body limit (VLM images can be 20MB+)
        // Request timeout: kill idle/stalled connections after 5 minutes.
        // Streaming responses (SSE) are not affected — the timeout applies to the
        // initial request processing, not the response stream. Long inference
        // requests complete within this window; the 30s inference timeout fires first.
        .layer(tower_http::timeout::TimeoutLayer::with_status_code(
            axum::http::StatusCode::REQUEST_TIMEOUT,
            std::time::Duration::from_secs(REQUEST_TIMEOUT_SECS),
        ))
        .layer(axum::middleware::from_fn_with_state(
            state.clone(),
            middleware::auth_middleware,
        ))
        .layer(axum::middleware::from_fn_with_state(
            state.clone(),
            middleware::rate_limit_middleware,
        ))
        .layer(middleware::cors_layer(state.config.node.listen_port))
        .layer(axum::middleware::from_fn(middleware::security_headers))
        .with_state(state)
}

async fn health() -> Json<serde_json::Value> {
    Json(serde_json::json!({"status": "ok"}))
}

/// Start the Axum HTTP server using SharedState from the daemon.
pub async fn run_server_with_state(
    shared_state: Arc<SharedState>,
    router_tx: mpsc::Sender<RouterCommand>,
    acquisition_tx: mpsc::Sender<AcquisitionCommand>,
    network_tx: mpsc::Sender<NetworkCommand>,
) -> anyhow::Result<()> {
    let port = shared_state.config.node.listen_port;
    let mut shutdown_rx = shared_state.shutdown_rx();
    let state = AppState {
        rate_limiter: middleware::RateLimiter::new(
            shared_state.config.api.rate_limit_rpm.unwrap_or(60),
            shared_state.config.api.rate_limit_admin_rpm.unwrap_or(200),
        ),
        config: shared_state.config.clone(),
        db: shared_state.db.clone(),
        executor: shared_state.executor.clone(),
        router_tx: Some(router_tx),
        acquisition_tx: Some(acquisition_tx),
        network_tx: Some(network_tx),
        shared_state,
    };

    // Periodically clean up stale rate-limiter entries to prevent memory exhaustion
    // from unique IPs accumulating over time (DashMap never shrinks otherwise).
    let cleanup_limiter = state.rate_limiter.clone();
    let mut cleanup_shutdown = state.shared_state.shutdown_rx();
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(std::time::Duration::from_secs(
            RATE_LIMIT_CLEANUP_INTERVAL_SECS,
        ));
        loop {
            tokio::select! {
                _ = interval.tick() => {
                    cleanup_limiter.cleanup(std::time::Duration::from_secs(RATE_LIMIT_BUCKET_TTL_SECS));
                }
                _ = cleanup_shutdown.changed() => break,
            }
        }
    });

    let app = build_router(state);
    let addr = std::net::SocketAddr::from(([0, 0, 0, 0], port));

    tracing::debug!(%addr, "DIAG: server startup");

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(
        listener,
        app.into_make_service_with_connect_info::<std::net::SocketAddr>(),
    )
    .with_graceful_shutdown(async move {
        // Wait until the shutdown watch channel signals true
        let _ = shutdown_rx.wait_for(|v| *v).await;
        tracing::info!("API server shutting down gracefully");
    })
    .await?;

    Ok(())
}
