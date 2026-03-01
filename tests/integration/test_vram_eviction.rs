//! Integration tests for Prometheus metrics and health readiness endpoints (Phase 11).
//!
//! Tests that the /metrics endpoint returns valid Prometheus text-format output
//! and that the /health/ready endpoint reports subsystem status correctly.

use std::net::SocketAddr;
use std::sync::Arc;

use tokio::sync::Mutex;

use swarmllm::api::server::{build_router, AppState};
use swarmllm::config::Config;
use swarmllm::daemon::SharedState;
use swarmllm::identity::Identity;
use swarmllm::inference::executor::ModelExecutor;
use swarmllm::storage::db::Database;

/// Spawn a test API server on a random port. Returns (base_url, api_key).
async fn spawn_test_server() -> (String, String) {
    let config = Config::default();
    let identity = Identity::generate();
    let db = Database::open_temp().expect("temp db");
    let executor = Arc::new(Mutex::new(ModelExecutor::new()));

    let (shared_state, _shutdown_rx) =
        SharedState::new(config.clone(), identity, db.clone(), executor.clone(), None);

    let api_key = shared_state.api_key.clone();

    let state = AppState {
        config,
        db,
        executor,
        router_tx: None,
        acquisition_tx: None,
        network_tx: None,
        shared_state,
    };

    let app = build_router(state);
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

    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    (format!("http://127.0.0.1:{port}"), api_key)
}

/// Test that the Prometheus /metrics endpoint returns valid text-format output
/// with the expected metric names.
#[tokio::test]
async fn test_prometheus_metrics_endpoint() {
    let (base, _key) = spawn_test_server().await;

    // Metrics endpoint should NOT require auth (Prometheus convention)
    let resp = reqwest::get(format!("{base}/metrics")).await.unwrap();
    assert_eq!(resp.status(), 200);

    let content_type = resp
        .headers()
        .get("content-type")
        .unwrap()
        .to_str()
        .unwrap()
        .to_string();
    assert!(
        content_type.contains("text/plain"),
        "Metrics should return text/plain, got: {content_type}"
    );

    let body = resp.text().await.unwrap();

    // Verify expected Prometheus metrics are present
    assert!(
        body.contains("swarmllm_peers_connected"),
        "Missing peers_connected metric"
    );
    assert!(
        body.contains("swarmllm_inference_requests_total"),
        "Missing inference_requests_total"
    );
    assert!(
        body.contains("swarmllm_credits_balance"),
        "Missing credits_balance"
    );
    assert!(
        body.contains("swarmllm_shards_hosted"),
        "Missing shards_hosted"
    );
    assert!(
        body.contains("swarmllm_inference_latency_seconds"),
        "Missing latency histogram"
    );

    // Verify Prometheus metadata format
    assert!(body.contains("# HELP"), "Missing HELP comments");
    assert!(body.contains("# TYPE"), "Missing TYPE comments");
    assert!(body.contains("gauge") || body.contains("counter") || body.contains("histogram"));

    // Histogram should have bucket entries
    assert!(body.contains("_bucket{le="), "Missing histogram buckets");
    assert!(body.contains("_sum"), "Missing histogram sum");
    assert!(body.contains("_count"), "Missing histogram count");
}

/// Test that the health readiness probe returns appropriate status.
/// A freshly started server (without full daemon) should report not-ready.
#[tokio::test]
async fn test_health_ready_endpoint() {
    let (base, _key) = spawn_test_server().await;

    let resp = reqwest::get(format!("{base}/health/ready")).await.unwrap();
    // Fresh server hasn't set is_ready=true, so should be 503
    let status = resp.status().as_u16();
    let body: serde_json::Value = resp.json().await.unwrap();

    // Should have readiness status and subsystem breakdown
    assert!(body["ready"].is_boolean());
    assert!(body["subsystems"].is_object());

    // In test mode (no full daemon), should be not ready
    assert_eq!(body["ready"], false);
    assert_eq!(status, 503);

    // All subsystems should be listed
    let subsystems = body["subsystems"].as_object().unwrap();
    assert!(subsystems.contains_key("network"));
    assert!(subsystems.contains_key("inference_router"));
    assert!(subsystems.contains_key("api_server"));
}

/// Test that the download cancel endpoint rejects requests when no download is active.
#[tokio::test]
async fn test_download_cancel_no_active_download() {
    let (base, key) = spawn_test_server().await;
    let client = auth_client(&key);

    // Try to cancel a download that doesn't exist
    let resp = client
        .post(format!(
            "{base}/api/admin/downloads/nonexistent-model/cancel"
        ))
        .send()
        .await
        .unwrap();

    // Should fail — no active download
    assert_ne!(resp.status(), 200);
}

fn auth_client(api_key: &str) -> reqwest::Client {
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
