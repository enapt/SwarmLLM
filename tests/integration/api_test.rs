//! Integration tests for the HTTP API server.
//!
//! These tests start a real Axum server and make HTTP requests against it.
//! Run with: cargo test --test integration -- --test-threads=1

use std::net::SocketAddr;
use std::sync::Arc;

use tokio::sync::Mutex;

/// Spawn a test API server on a random available port. Returns (base_url, api_key).
async fn spawn_test_server() -> (String, String) {
    let config = swarmllm::config::Config::default();
    let identity = swarmllm::identity::Identity::generate();
    let db = swarmllm::storage::db::Database::open_temp().expect("temp db");
    let executor = Arc::new(Mutex::new(
        swarmllm::inference::executor::ModelExecutor::new(),
    ));

    let (shared_state, _shutdown_rx) = swarmllm::daemon::SharedState::new(
        config.clone(),
        identity,
        db.clone(),
        executor.clone(),
        None,
    );

    let api_key = shared_state.api_key.clone();

    let state = swarmllm::api::server::AppState {
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

    // Small delay to let server bind
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    (format!("http://127.0.0.1:{port}"), api_key)
}

/// Create a reqwest client with the Bearer token pre-configured.
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

#[tokio::test]
async fn health_endpoint_returns_ok() {
    let (base, _key) = spawn_test_server().await;
    let resp = reqwest::get(format!("{base}/health")).await.unwrap();
    assert_eq!(resp.status(), 200);
    assert_eq!(resp.text().await.unwrap(), "ok");
}

#[tokio::test]
async fn status_endpoint_returns_json() {
    let (base, key) = spawn_test_server().await;
    let client = auth_client(&key);
    let resp = client
        .get(format!("{base}/v1/status"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["status"], "ok");
    assert_eq!(body["model_loaded"], false);
}

#[tokio::test]
async fn models_endpoint_returns_empty_list() {
    let (base, key) = spawn_test_server().await;
    let client = auth_client(&key);
    let resp = client
        .get(format!("{base}/v1/models"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["object"], "list");
    assert!(body["data"].as_array().unwrap().is_empty());
}

#[tokio::test]
async fn chat_completions_without_model_returns_503() {
    let (base, key) = spawn_test_server().await;
    let client = auth_client(&key);
    let resp = client
        .post(format!("{base}/v1/chat/completions"))
        .json(&serde_json::json!({
            "model": "test",
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 503);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert!(body["error"]["message"]
        .as_str()
        .unwrap()
        .contains("No model loaded"));
}

#[tokio::test]
async fn admin_stats_returns_hardware_info() {
    let (base, key) = spawn_test_server().await;
    let client = auth_client(&key);
    let resp = client
        .get(format!("{base}/api/admin/stats"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    let body: serde_json::Value = resp.json().await.unwrap();
    assert!(body["node_id"].is_string());
    assert!(body["hardware"]["cpu_cores"].as_u64().unwrap() > 0);
    assert!(body["hardware"]["total_ram_mb"].as_u64().unwrap() > 0);
}

#[tokio::test]
async fn admin_credits_returns_tier() {
    let (base, key) = spawn_test_server().await;
    let client = auth_client(&key);
    let resp = client
        .get(format!("{base}/api/admin/credits"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    let body: serde_json::Value = resp.json().await.unwrap();
    assert!(body["tier"].is_string());
    assert!(body["balance"].is_number());
}

#[tokio::test]
async fn admin_peers_returns_empty_array() {
    let (base, key) = spawn_test_server().await;
    let client = auth_client(&key);
    let resp = client
        .get(format!("{base}/api/admin/peers"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    let body: Vec<serde_json::Value> = resp.json().await.unwrap();
    assert!(body.is_empty());
}

#[tokio::test]
async fn root_redirects_to_admin() {
    let (base, _key) = spawn_test_server().await;
    let client = reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .unwrap();

    let resp = client.get(format!("{base}/")).send().await.unwrap();
    assert_eq!(resp.status(), 303);
    assert!(resp
        .headers()
        .get("location")
        .unwrap()
        .to_str()
        .unwrap()
        .contains("/admin"));
}

#[tokio::test]
async fn admin_dashboard_serves_html() {
    let (base, _key) = spawn_test_server().await;
    let resp = reqwest::get(format!("{base}/admin")).await.unwrap();
    assert_eq!(resp.status(), 200);
    let text = resp.text().await.unwrap();
    assert!(text.contains("<!DOCTYPE html>") || text.contains("<html"));
}

#[tokio::test]
async fn identity_nickname_get_returns_anonymous() {
    let (base, key) = spawn_test_server().await;
    let client = auth_client(&key);
    let resp = client
        .get(format!("{base}/api/identity/nickname"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["visibility"], "anonymous");
}

#[tokio::test]
async fn identity_leaderboard_returns_ok() {
    let (base, key) = spawn_test_server().await;
    let client = auth_client(&key);
    let resp = client
        .get(format!("{base}/api/identity/leaderboard"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    let body: serde_json::Value = resp.json().await.unwrap();
    // Should return valid JSON (array or object with entries)
    assert!(body.is_array() || body.is_object());
}

#[tokio::test]
async fn pool_state_returns_null_when_not_in_pool() {
    let (base, key) = spawn_test_server().await;
    let client = auth_client(&key);
    let resp = client
        .get(format!("{base}/api/pool/state"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    let body: serde_json::Value = resp.json().await.unwrap();
    assert!(body["pool"].is_null());
}

#[tokio::test]
async fn completions_without_model_returns_503() {
    let (base, key) = spawn_test_server().await;
    let client = auth_client(&key);
    let resp = client
        .post(format!("{base}/v1/completions"))
        .json(&serde_json::json!({
            "model": "test",
            "prompt": "Hello"
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 503);
}

#[tokio::test]
async fn unauthenticated_openai_endpoint_returns_401() {
    let (base, _key) = spawn_test_server().await;
    // No Bearer token — should get 401 on OpenAI-compatible endpoints
    let resp = reqwest::Client::new()
        .get(format!("{base}/v1/models"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 401);

    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["error"]["code"], 401);
}

#[tokio::test]
async fn wrong_api_key_returns_401() {
    let (base, _key) = spawn_test_server().await;
    let client = auth_client("wrong-key-12345");
    // Wrong Bearer token — should get 401 on auth-required endpoints
    let resp = client
        .get(format!("{base}/v1/models"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 401);
}

#[tokio::test]
async fn api_key_endpoint_returns_key() {
    let (base, key) = spawn_test_server().await;
    let client = auth_client(&key);
    let resp = client
        .get(format!("{base}/api/admin/api-key"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["api_key"].as_str().unwrap(), key);
}
