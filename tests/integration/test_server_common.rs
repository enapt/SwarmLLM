//! Shared test-server harness for the integration test binaries.
//!
//! `api_test.rs` (the `integration` binary) and `test_metrics_health.rs` (a
//! module of the `integration_phase10_11` binary) both need to boot a real
//! Axum server. They live in different `[[test]]` targets, so an ordinary
//! `mod common;` can't reach across — each binary pulls this file in with
//! `#[path = "test_server_common.rs"]` instead, compiling its own copy.
//!
//! Before this existed both files carried ~65 lines of byte-identical setup
//! that had to be edited in lockstep. The R86 readiness-probe fix was applied
//! to both correctly, but nothing enforced that.
//!
//! Each binary uses a different subset, so unused-item warnings are expected
//! and suppressed at the module level.
#![allow(dead_code)]

use std::net::SocketAddr;
use std::sync::Arc;

use tokio::sync::Mutex;

/// Spawn a test API server on a random available port.
///
/// Returns `(base_url, api_key)`. The server task is detached and lives until
/// the test binary exits.
pub async fn spawn_test_server() -> (String, String) {
    let config = swarmllm::config::Config::default();
    let identity = swarmllm::identity::Identity::generate();
    let db = swarmllm::storage::db::Database::open_temp().expect("temp db");
    let executor = Arc::new(Mutex::new(
        swarmllm::inference::executor::ModelExecutor::new(),
    ));

    let (shared_state, _shutdown_rx, _dht_rx) = swarmllm::daemon::SharedState::new(
        config.clone(),
        identity,
        db.clone(),
        executor.clone(),
        None,
    );

    let api_key = shared_state.api_key.clone();

    let state = swarmllm::api::server::AppState {
        rate_limiter: swarmllm::api::middleware::RateLimiter::new(
            config.api.rate_limit_rpm.unwrap_or(60),
            config.api.rate_limit_admin_rpm.unwrap_or(200),
        ),
        config,
        db,
        executor,
        router_tx: None,
        acquisition_tx: None,
        network_tx: None,
        shared_state,
        bootstrap_nonces: std::sync::Arc::new(dashmap::DashMap::new()),
    };

    let app = swarmllm::api::server::build_router(state);
    let addr = SocketAddr::from(([127, 0, 0, 1], 0));
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    let port = listener.local_addr().unwrap().port();

    tokio::spawn(async move {
        axum::serve(
            listener,
            app.into_make_service_with_connect_info::<std::net::SocketAddr>(),
        )
        .await
        .unwrap();
    });

    // Probe TCP accept readiness instead of a fixed 50ms sleep. On slow CI
    // runners 50ms wasn't always enough; on fast machines it's wasted time
    // multiplied across every test. Bound: 2s with 1ms backoff.
    let probe_addr = format!("127.0.0.1:{port}");
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
    while std::time::Instant::now() < deadline {
        match tokio::net::TcpStream::connect(&probe_addr).await {
            Ok(_) => break,
            Err(_) => tokio::time::sleep(std::time::Duration::from_millis(1)).await,
        }
    }

    (format!("http://127.0.0.1:{port}"), api_key)
}

/// Create a reqwest client with the Bearer token pre-configured.
pub fn auth_client(api_key: &str) -> reqwest::Client {
    let mut headers = reqwest::header::HeaderMap::new();
    headers.insert(
        reqwest::header::AUTHORIZATION,
        reqwest::header::HeaderValue::from_str(&format!("Bearer {api_key}")).unwrap(),
    );
    reqwest::Client::builder()
        .default_headers(headers)
        .build()
        .unwrap()
}
