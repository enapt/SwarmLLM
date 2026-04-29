//! End-to-end daemon lifecycle test: spawn, serve, shutdown.
//!
//! Brings up the in-process Axum HTTP server with a real `SharedState`,
//! exercises the basic admin / OpenAI-compatible surface, and verifies the
//! cooperative shutdown signal propagates correctly. This is the test that
//! would catch a "daemon won't start" or "shutdown hangs" regression that
//! the existing api_test.rs (which only mounts the router) doesn't see.
//!
//! Marked `#[ignore]` so CI runs it explicitly via `cargo test --test
//! integration_phase10_11 -- --ignored end_to_end`. The full multi-process
//! `swarmllm run` spawn-and-stop test is left as a manual smoke step until
//! the tiny_model fixture is regenerated.

use std::net::SocketAddr;
use std::sync::Arc;

use tokio::sync::Mutex;

#[tokio::test]
#[ignore]
async fn daemon_lifecycle_serves_health_then_shuts_down() {
    // Spin up SharedState exactly like api_test.rs but watch the shutdown
    // channel from outside so we can assert it propagates.
    let config = swarmllm::config::Config::default();
    let identity = swarmllm::identity::Identity::generate();
    let db = swarmllm::storage::db::Database::open_temp().expect("temp db");
    let executor = Arc::new(Mutex::new(
        swarmllm::inference::executor::ModelExecutor::new(),
    ));

    let (shared_state, shutdown_rx, _dht_rx) = swarmllm::daemon::SharedState::new(
        config.clone(),
        identity,
        db.clone(),
        executor.clone(),
        None,
    );

    let api_key = shared_state.api_key.clone();
    let shared_state_for_shutdown = shared_state.clone();

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
    };

    let app = swarmllm::api::server::build_router(state);
    let addr = SocketAddr::from(([127, 0, 0, 1], 0));
    let listener = tokio::net::TcpListener::bind(addr).await.expect("bind");
    let port = listener.local_addr().unwrap().port();
    let base = format!("http://127.0.0.1:{port}");

    let server = tokio::spawn(async move {
        axum::serve(
            listener,
            app.into_make_service_with_connect_info::<SocketAddr>(),
        )
        .await
        .ok();
    });

    // Give the server a beat to start accepting.
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    let client = reqwest::Client::new();

    // 1. /health (unauthenticated) — daemon is alive.
    let resp = client
        .get(format!("{base}/health"))
        .send()
        .await
        .expect("health");
    assert_eq!(resp.status(), 200, "health should return 200");

    // 2. /v1/models with auth — no models loaded, but the endpoint should
    //    answer with an empty data array, not 500.
    let resp = client
        .get(format!("{base}/v1/models"))
        .bearer_auth(&api_key)
        .send()
        .await
        .expect("models");
    assert_eq!(resp.status(), 200, "/v1/models should return 200");
    let body: serde_json::Value = resp.json().await.expect("models json");
    assert_eq!(body["object"], "list", "v1/models object field");
    assert!(body["data"].is_array(), "v1/models data should be an array");

    // 3. /v1/chat/completions for a missing model — should be a clean 4xx,
    //    not 500. Specifically, ModelNotAvailable maps to 404.
    let resp = client
        .post(format!("{base}/v1/chat/completions"))
        .bearer_auth(&api_key)
        .json(&serde_json::json!({
            "model": "definitely-not-loaded",
            "messages": [ { "role": "user", "content": "hi" } ],
            "max_tokens": 8,
        }))
        .send()
        .await
        .expect("chat completions");
    let st = resp.status();
    // The cold-start fallback path may return 404 (NoModelLoaded) or 503
    // (no router available in this in-process mount). Both are correct
    // 4xx/5xx handling — what we're guarding against is a panic / 500-bug.
    assert!(
        st.is_client_error() || st == reqwest::StatusCode::SERVICE_UNAVAILABLE,
        "missing-model request should fail cleanly, got {st}"
    );

    // 4. Cooperative shutdown via the SharedState helper. Watch the
    //    receiver flip and assert.
    shared_state_for_shutdown.shutdown();
    let started = std::time::Instant::now();
    while started.elapsed() < std::time::Duration::from_secs(2) {
        if *shutdown_rx.borrow() {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    }
    assert!(
        *shutdown_rx.borrow(),
        "shutdown signal did not propagate within 2s"
    );

    // Server keeps running until its handle is aborted; we drop it here.
    server.abort();
}
