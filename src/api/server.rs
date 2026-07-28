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

/// HTTP request processing timeout for everything EXCEPT running a model.
///
/// Generation has no bounded duration — see `generation_routes` in
/// [`build_router`], which is deliberately excluded from this.
const REQUEST_TIMEOUT_SECS: u64 = 300;
/// Maximum request body. VLM image payloads run to 20MB+.
const MAX_REQUEST_BODY_BYTES: usize = 32 * 1024 * 1024;
/// Rate-limiter cleanup interval.
const RATE_LIMIT_CLEANUP_INTERVAL_SECS: u64 = 300;
/// Rate-limiter bucket TTL — entries older than this are evicted.
const RATE_LIMIT_BUCKET_TTL_SECS: u64 = 600;

/// TTL for bootstrap nonces. Long enough that a slow page load can still
/// consume the nonce, short enough that an attacker who scrapes /admin in
/// the background can't accumulate a large pool of valid nonces. The
/// dashboard's bootstrap fetch happens within ~1s of HTML parse on every
/// platform we've measured.
pub(crate) const BOOTSTRAP_NONCE_TTL_SECS: u64 = 60;

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
    /// Per-page bootstrap nonces, one-time-use. Issued when serving the
    /// dashboard HTML, consumed by the loopback `GET /api/admin/api-key`
    /// gate so a curl-style local attacker can no longer bypass auth by
    /// setting `Sec-Fetch-Site: same-origin` themselves. Entries are
    /// expired lazily on issue and on consume.
    pub bootstrap_nonces: Arc<dashmap::DashMap<String, std::time::Instant>>,
}

impl AppState {
    /// Resolve the on-disk directory for a model (proxy to
    /// [`SharedState::model_dir`]). Keeps handler code free of the
    /// `state.config.node.data_dir` reach-through.
    pub fn model_dir(&self, model_id: &str) -> std::path::PathBuf {
        self.shared_state.model_dir(model_id)
    }

    /// Generate a fresh 32-byte bootstrap nonce, register it with TTL, and
    /// return the base64url-encoded string for embedding in dashboard HTML.
    /// Lazy-cleans expired entries on every issue so the map can't grow
    /// unbounded if the dashboard is opened repeatedly.
    pub(crate) fn issue_bootstrap_nonce(&self) -> String {
        issue_bootstrap_nonce_into(
            &self.bootstrap_nonces,
            std::time::Duration::from_secs(BOOTSTRAP_NONCE_TTL_SECS),
        )
    }

    /// Validate and consume a bootstrap nonce. Returns `true` iff the
    /// nonce was registered AND not expired AT THE TIME OF CHECK. Removes
    /// the entry on any outcome (one-time-use) so a leaked nonce can't be
    /// replayed even within its TTL.
    pub(crate) fn consume_bootstrap_nonce(&self, nonce: &str) -> bool {
        consume_bootstrap_nonce_from(&self.bootstrap_nonces, nonce)
    }
}

/// Standalone form of [`AppState::issue_bootstrap_nonce`] that takes the
/// DashMap and TTL directly. Split out so tests can exercise the
/// expiry / GC semantics without constructing a full `AppState`.
fn issue_bootstrap_nonce_into(
    nonces: &dashmap::DashMap<String, std::time::Instant>,
    ttl: std::time::Duration,
) -> String {
    use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
    use rand::RngCore;
    let mut bytes = [0u8; 32];
    rand::rngs::OsRng.fill_bytes(&mut bytes);
    let nonce = URL_SAFE_NO_PAD.encode(bytes);
    let now = std::time::Instant::now();
    // Lazy GC — bounded work because issuance frequency is low (one per
    // dashboard page load) and the map size is implicitly capped by TTL.
    nonces.retain(|_, exp| *exp > now);
    nonces.insert(nonce.clone(), now + ttl);
    nonce
}

/// Standalone form of [`AppState::consume_bootstrap_nonce`].
fn consume_bootstrap_nonce_from(
    nonces: &dashmap::DashMap<String, std::time::Instant>,
    nonce: &str,
) -> bool {
    match nonces.remove(nonce) {
        Some((_, expires_at)) => expires_at > std::time::Instant::now(),
        None => false,
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

/// Token in `frontend/index.html` that the dashboard handler replaces with
/// a freshly-issued bootstrap nonce. The dashboard JS reads the nonce from
/// the `<meta name="bootstrap-nonce">` tag and sends it as `X-Dashboard-Nonce`
/// on its `/api/admin/api-key` bootstrap fetch. Single-use, 60-second TTL.
const BOOTSTRAP_NONCE_PLACEHOLDER: &str = "__SWARMLLM_BOOTSTRAP_NONCE__";

/// Tokens carrying what the daemon observed about this request's origin: the
/// source address it actually saw, and how `api::dashboard_trust` classified
/// it. Without these the page cannot tell "the key handout refused me" from
/// any other 401, and the user cannot see the one address they would need to
/// allow — behind a NAT or a Tailscale subnet router it is not an address the
/// browser knows about.
const CLIENT_ADDR_PLACEHOLDER: &str = "__SWARMLLM_CLIENT_ADDR__";
const CLIENT_TRUST_PLACEHOLDER: &str = "__SWARMLLM_CLIENT_TRUST__";

/// Render the dashboard HTML for one page load.
///
/// The single place the per-page placeholders are substituted. Both dashboard
/// routes go through it so a new SPA entry point cannot ship a page that is
/// missing the nonce (no key) or the trust markers (an unexplainable 401) —
/// the substitutions belong together and were previously duplicated per
/// handler.
fn render_dashboard(state: &AppState, html: String, client_ip: std::net::IpAddr) -> String {
    let nonce = state.issue_bootstrap_nonce();
    let trust = crate::api::dashboard_trust::classify(&state.shared_state, client_ip);
    html.replace(BOOTSTRAP_NONCE_PLACEHOLDER, &nonce)
        .replace(CLIENT_ADDR_PLACEHOLDER, &client_ip.to_string())
        .replace(CLIENT_TRUST_PLACEHOLDER, trust.as_str())
}

/// Wrapper handler for the dashboard HTML. Issues a fresh per-page
/// bootstrap nonce and substitutes it for the placeholder before
/// returning. Replaces the bare `assets::serve_dashboard` so the
/// admin-key bootstrap is gated by a value the legitimate dashboard JS
/// must read out of the served HTML, raising the bar against curl-style
/// attackers that previously bypassed via `Sec-Fetch-Site`.
async fn serve_dashboard_with_nonce(
    axum::extract::ConnectInfo(addr): axum::extract::ConnectInfo<std::net::SocketAddr>,
    axum::extract::State(state): axum::extract::State<AppState>,
) -> axum::response::Html<String> {
    let html = assets::dashboard_html_owned().await;
    axum::response::Html(render_dashboard(&state, html, addr.ip()))
}

/// SPA catch-all variant of [`serve_dashboard_with_nonce`]. Any path that
/// the SPA owns (e.g. `/admin/models`, `/chat/abc`) returns the same
/// dashboard HTML with a fresh nonce so client-side routing resolves to
/// the same shell.
async fn serve_dashboard_catchall_with_nonce(
    axum::extract::ConnectInfo(addr): axum::extract::ConnectInfo<std::net::SocketAddr>,
    axum::extract::State(state): axum::extract::State<AppState>,
    axum::extract::Path(_path): axum::extract::Path<String>,
) -> axum::response::Html<String> {
    let html = assets::dashboard_html_owned().await;
    axum::response::Html(render_dashboard(&state, html, addr.ip()))
}

/// Build the Axum router with all routes.
pub fn build_router(state: AppState) -> Router {
    // Routes that can run a model, and therefore have no bounded duration.
    //
    // These are deliberately kept out of `REQUEST_TIMEOUT_SECS`. How long a
    // generation legitimately takes is set by the prompt and the answer, not by
    // the clock: reading a long prompt (prefill) is ~99% of the wait and scales
    // with its length, so on a modest CPU node a few thousand prompt tokens can
    // exceed five minutes before the first token is even produced. A blanket
    // timeout cut those requests off mid-prefill while the node was working
    // perfectly — the same mistake as the flat first-token budget, one layer up,
    // and it silently capped that fix at 300s no matter what the node could do.
    //
    // What actually bounds these requests is better suited to the job: the
    // first-token budget scales with the prompt, the client going away is
    // detected and cancels the work, and TCP keepalive notices a client that
    // vanished without saying so. Every other layer — auth, rate limiting, CORS,
    // security headers — still applies, because the merge happens before them.
    let generation_routes = Router::new()
        .route("/v1/chat/completions", post(openai::chat_completions))
        .route("/v1/responses", post(openai::responses::create_response))
        .route("/v1/messages", post(anthropic::messages))
        .route(
            "/mcp",
            post(mcp::handle_mcp)
                .get(mcp::handle_mcp_get)
                .delete(mcp::handle_mcp_delete),
        )
        .layer(DefaultBodyLimit::max(MAX_REQUEST_BODY_BYTES));

    Router::new()
        // OpenAI-compatible API
        .route("/v1/embeddings", post(openai::embeddings))
        .route("/v1/models", get(openai::list_models))
        // OpenAI Responses API (gpt-5 / o-series default)
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
        // Provider listing
        .route("/v1/providers", get(providers::list_providers))
        // SwarmLLM extensions
        .route("/v1/status", get(openai::status))
        // Admin API
        .route("/api/admin/stats", get(admin::stats))
        .route("/api/admin/swarm/capacity", get(admin::swarm_capacity))
        .route(
            "/api/admin/swarm/capacity-plan",
            get(admin::swarm_capacity_plan),
        )
        .route("/api/admin/wishlist", get(admin::wishlist))
        .route("/api/admin/diagnostics", get(admin::diagnostics))
        .route("/api/admin/performance", get(admin::performance))
        .route("/api/admin/reference-models", get(admin::reference_models))
        .route(
            "/api/admin/quant-recommendations",
            get(admin::quant_recommendations),
        )
        .route(
            "/api/admin/foreign-pool-catalog",
            get(admin::foreign_pool_catalog),
        )
        .route("/api/admin/hf/trending", get(admin::hf_trending))
        .route(
            "/api/admin/storage/breakdown",
            get(admin::storage_breakdown),
        )
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
        // One action to make prompt privacy possible: fetch the first and last
        // pieces of a model. Privacy then engages on its own.
        .route(
            "/api/admin/models/{id}/enable-privacy",
            post(admin::enable_model_privacy),
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
        .route("/admin", get(serve_dashboard_with_nonce))
        .route("/admin/{*path}", get(serve_dashboard_catchall_with_nonce))
        .route("/chat", get(serve_dashboard_with_nonce))
        .route("/chat/{*path}", get(serve_dashboard_catchall_with_nonce))
        .route("/setup", get(serve_dashboard_with_nonce))
        .route("/static/{*path}", get(assets::serve_static))
        // Root redirect
        .route("/", get(|| async { Redirect::to("/admin") }))
        // Health check
        .route("/health", get(health))
        .route("/health/ready", get(metrics::health_ready))
        // MCP (Model Context Protocol) server — Streamable HTTP transport
        // Prometheus metrics
        .route("/metrics", get(metrics::metrics))
        // Middleware (layers run bottom-to-top: timeout, CORS, security headers, rate limit, auth, body limit, handler)
        .layer(DefaultBodyLimit::max(MAX_REQUEST_BODY_BYTES))
        // Request timeout: kill stalled connections after 5 minutes. Applies to
        // everything that is NOT running a model — those routes are merged in
        // below, outside this layer, for the reasons given at `generation_routes`.
        .layer(tower_http::timeout::TimeoutLayer::with_status_code(
            axum::http::StatusCode::REQUEST_TIMEOUT,
            std::time::Duration::from_secs(REQUEST_TIMEOUT_SECS),
        ))
        // Merged here, after the timeout and before auth, so generation keeps
        // every other protection while escaping the clock.
        .merge(generation_routes)
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

/// Idle time before the OS starts probing whether an HTTP client is still there.
///
/// The platform default is two hours, which makes keepalive useless for this:
/// the cost being avoided is a CPU node computing a long answer for a client
/// that no longer exists. Detection lands at roughly
/// `IDLE + INTERVAL * RETRIES` (~90s with these values), which is short enough
/// to stop wasting a worker and long enough never to interrupt a healthy client
/// mid-prefill — SSE keep-alive comments go out far more often than this.
const HTTP_KEEPALIVE_IDLE_SECS: u64 = 60;
/// Gap between probes once the connection has gone quiet.
const HTTP_KEEPALIVE_INTERVAL_SECS: u64 = 10;
/// Unanswered probes before the connection is declared dead.
#[cfg(not(target_os = "windows"))]
const HTTP_KEEPALIVE_RETRIES: u32 = 3;

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
        bootstrap_nonces: Arc::new(dashmap::DashMap::new()),
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

    // Anchor nodes bind the dashboard/API to loopback only — the P2P ports are
    // the only thing that should be reachable off-box. Normal nodes bind all
    // interfaces so the dashboard is reachable on the LAN.
    let anchor = state.config.node.anchor_mode;
    let app = build_router(state);
    let bind_ip = if anchor { [127, 0, 0, 1] } else { [0, 0, 0, 0] };
    let addr = std::net::SocketAddr::from((bind_ip, port));

    tracing::debug!(%addr, "DIAG: server startup");

    let listener = tokio::net::TcpListener::bind(addr).await.map_err(|e| {
        anyhow::anyhow!(
            "Failed to bind API server to port {port}: {e}\n  Is another SwarmLLM instance already running? Try: swarmllm status\n  Or start on a different port: swarmllm run --port <N>"
        )
    })?;
    // Ask the OS to tell us when a client has silently gone away.
    //
    // Disconnect detection previously relied entirely on the transport
    // reporting a closed connection — `sse_tx.closed()` fires when hyper drops
    // the response body. That covers a client that closes cleanly, but NOT one
    // that vanishes without a FIN or RST: a killed process whose socket is held
    // by something else, a machine losing power, a network partition, or a
    // firewall silently dropping the flow. In those cases the request kept
    // computing for nobody. An external report measured a decode running about
    // six minutes past the point its client had died.
    //
    // Time-bounding the request is not an option — a long prompt on a CPU node
    // legitimately takes minutes, which is the whole point of the prefill
    // budget. Liveness has to be answered by the transport instead, so probes
    // are enabled explicitly: the platform default idle time is two hours,
    // which is far too long to be useful here.
    let keepalive = socket2::TcpKeepalive::new()
        .with_time(std::time::Duration::from_secs(HTTP_KEEPALIVE_IDLE_SECS))
        .with_interval(std::time::Duration::from_secs(HTTP_KEEPALIVE_INTERVAL_SECS));
    #[cfg(not(target_os = "windows"))]
    let keepalive = keepalive.with_retries(HTTP_KEEPALIVE_RETRIES);
    let listener = {
        use axum::serve::ListenerExt;
        listener.tap_io(move |stream| {
            if let Err(e) = socket2::SockRef::from(&*stream).set_tcp_keepalive(&keepalive) {
                // Not fatal: without probes we simply fall back to the previous
                // behaviour of noticing only clean disconnects.
                tracing::debug!(error = %e, "could not enable TCP keepalive on connection");
            }
        })
    };
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nonce_round_trip_succeeds_within_ttl() {
        let map = dashmap::DashMap::new();
        let nonce = issue_bootstrap_nonce_into(&map, std::time::Duration::from_secs(60));
        // 32 random bytes encoded URL-SAFE-NO-PAD = 43 chars (32 * 4 / 3 rounded up,
        // minus the 1 char of padding skipped).
        assert_eq!(nonce.len(), 43);
        assert!(consume_bootstrap_nonce_from(&map, &nonce));
        // One-time use: the same nonce must not validate again.
        assert!(!consume_bootstrap_nonce_from(&map, &nonce));
    }

    #[test]
    fn nonce_with_zero_ttl_rejects_immediately() {
        // TTL=0 → expires_at == now → strict `>` comparison rejects.
        // Defends against the legitimate-but-expired race the GC sweep
        // might otherwise let through.
        let map = dashmap::DashMap::new();
        let nonce = issue_bootstrap_nonce_into(&map, std::time::Duration::from_secs(0));
        assert!(!consume_bootstrap_nonce_from(&map, &nonce));
    }

    #[test]
    fn nonce_unknown_value_rejected() {
        let map = dashmap::DashMap::new();
        // An attacker who guesses or fabricates a nonce that was never
        // issued must not bypass the gate.
        assert!(!consume_bootstrap_nonce_from(&map, "fake-nonce-value"));
    }

    #[test]
    fn nonce_issuance_gc_drops_expired_entries() {
        let map = dashmap::DashMap::new();
        // Stuff an already-expired entry directly into the map.
        let stale = "stale-nonce-from-an-old-session".to_string();
        map.insert(stale.clone(), std::time::Instant::now());
        // Sleep briefly so the entry is strictly in the past, not "now".
        std::thread::sleep(std::time::Duration::from_millis(2));
        // Issuing a new nonce sweeps expired entries.
        let _fresh = issue_bootstrap_nonce_into(&map, std::time::Duration::from_secs(60));
        assert!(!map.contains_key(&stale));
    }
}
