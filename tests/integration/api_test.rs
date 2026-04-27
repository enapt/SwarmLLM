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
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["status"], "ok");
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
    // Should return an object with a leaderboard array
    assert!(
        body["leaderboard"].is_array(),
        "Leaderboard should contain an array, got: {body}"
    );
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
    assert_eq!(body["error"]["code"], "authentication_error");
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

/// V3: Concurrent load test — fire 8 simultaneous requests, verify all complete.
/// Without a model loaded, all should return 503, but the key assertion is that
/// the server handles all requests concurrently without deadlock or dropped connections.
#[tokio::test]
async fn concurrent_requests_all_complete() {
    let (base, key) = spawn_test_server().await;
    let client = auth_client(&key);
    let num_requests = 8;

    let mut handles = Vec::new();
    for i in 0..num_requests {
        let client = client.clone();
        let url = format!("{base}/v1/chat/completions");
        handles.push(tokio::spawn(async move {
            let resp = client
                .post(&url)
                .json(&serde_json::json!({
                    "model": "test",
                    "messages": [{"role": "user", "content": format!("request {i}")}]
                }))
                .send()
                .await
                .unwrap();
            (i, resp.status().as_u16())
        }));
    }

    let mut results = Vec::new();
    for handle in handles {
        results.push(handle.await.unwrap());
    }

    // All 8 requests should complete (503 = no model loaded, not a server error)
    assert_eq!(results.len(), num_requests);
    for (i, status) in &results {
        assert_eq!(
            *status, 503,
            "Request {i} returned {status}, expected 503 (no model)"
        );
    }
}

/// V3: Concurrent requests with mixed endpoints — tests server doesn't deadlock
/// under concurrent load across different endpoint types.
#[tokio::test]
async fn concurrent_mixed_endpoints() {
    let (base, key) = spawn_test_server().await;
    let client = auth_client(&key);

    let mut handles = Vec::new();

    // 4 chat requests
    for _ in 0..4 {
        let c = client.clone();
        let url = format!("{base}/v1/chat/completions");
        handles.push(tokio::spawn(async move {
            c.post(&url)
                .json(&serde_json::json!({
                    "model": "test",
                    "messages": [{"role": "user", "content": "hi"}]
                }))
                .send()
                .await
                .unwrap()
                .status()
                .as_u16()
        }));
    }

    // 4 model list requests
    for _ in 0..4 {
        let c = client.clone();
        let url = format!("{base}/v1/models");
        handles.push(tokio::spawn(async move {
            c.get(&url).send().await.unwrap().status().as_u16()
        }));
    }

    // 4 status requests
    for _ in 0..4 {
        let c = client.clone();
        let url = format!("{base}/v1/status");
        handles.push(tokio::spawn(async move {
            c.get(&url).send().await.unwrap().status().as_u16()
        }));
    }

    let mut results = Vec::new();
    for handle in handles {
        results.push(handle.await.unwrap());
    }

    assert_eq!(results.len(), 12);
    // Chat: 503 (no model), Models: 200, Status: 200
    for (i, status) in results.iter().enumerate() {
        if i < 4 {
            assert_eq!(*status, 503, "Chat request {i}: expected 503");
        } else {
            assert_eq!(*status, 200, "Request {i}: expected 200");
        }
    }
}

/// V7: tool_calls request accepted by the API (returns 503 without model, but parses OK).
#[tokio::test]
async fn tool_calls_request_accepted() {
    let (base, key) = spawn_test_server().await;
    let client = auth_client(&key);
    let resp = client
        .post(format!("{base}/v1/chat/completions"))
        .json(&serde_json::json!({
            "model": "test",
            "messages": [
                {"role": "user", "content": "What's the weather in NYC?"}
            ],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"}
                        },
                        "required": ["location"]
                    }
                }
            }],
            "tool_choice": "auto"
        }))
        .send()
        .await
        .unwrap();
    // Should be 503 (no model) not 400/422 (parse error)
    assert_eq!(resp.status(), 503);
}

/// V7: logprobs request accepted by the API.
#[tokio::test]
async fn logprobs_request_accepted() {
    let (base, key) = spawn_test_server().await;
    let client = auth_client(&key);
    let resp = client
        .post(format!("{base}/v1/chat/completions"))
        .json(&serde_json::json!({
            "model": "test",
            "messages": [{"role": "user", "content": "hi"}],
            "logprobs": true,
            "top_logprobs": 5
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 503);
}

/// C10: response_format json_object accepted.
#[tokio::test]
async fn response_format_json_object_accepted() {
    let (base, key) = spawn_test_server().await;
    let client = auth_client(&key);
    let resp = client
        .post(format!("{base}/v1/chat/completions"))
        .json(&serde_json::json!({
            "model": "test",
            "messages": [{"role": "user", "content": "Return a JSON object with name and age"}],
            "response_format": {"type": "json_object"}
        }))
        .send()
        .await
        .unwrap();
    // 503 = no model (not 400/422 = parsed OK)
    assert_eq!(resp.status(), 503);
}

/// C10: response_format json_schema accepted.
#[tokio::test]
async fn response_format_json_schema_accepted() {
    let (base, key) = spawn_test_server().await;
    let client = auth_client(&key);
    let resp = client
        .post(format!("{base}/v1/chat/completions"))
        .json(&serde_json::json!({
            "model": "test",
            "messages": [{"role": "user", "content": "Give me a person"}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "person",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "age": {"type": "integer"}
                        },
                        "required": ["name", "age"]
                    }
                }
            }
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 503);
}

/// V7: tool role messages parse correctly.
// ============================================================================
// /v1/responses (OpenAI Responses API) integration tests
//
// The test scaffold has no router/inference, so well-formed requests for
// inference paths return 503 (no model). These tests verify the
// auth/validation/translation surface of the Responses endpoint —
// malformed bodies → 400/422, unauthenticated → 401, well-formed →
// 503 (proves the request body parsed and reached the inference layer).
// ============================================================================

#[tokio::test]
async fn responses_endpoint_requires_authentication() {
    let (base, _key) = spawn_test_server().await;
    let resp = reqwest::Client::new()
        .post(format!("{base}/v1/responses"))
        .json(&serde_json::json!({"model": "test", "input": "hi"}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 401, "responses POST without bearer must 401");
}

#[tokio::test]
async fn responses_endpoint_text_input_parses() {
    let (base, key) = spawn_test_server().await;
    let client = auth_client(&key);
    // Plain text input form (string-shaped `input`).
    let resp = client
        .post(format!("{base}/v1/responses"))
        .json(&serde_json::json!({
            "model": "test",
            "input": "hello world"
        }))
        .send()
        .await
        .unwrap();
    // 503 = no model (request body parsed and routed); NOT 400/422.
    assert_eq!(
        resp.status(),
        503,
        "well-formed text input must reach inference path (no model → 503)"
    );
}

#[tokio::test]
async fn responses_endpoint_array_input_with_function_call_parses() {
    let (base, key) = spawn_test_server().await;
    let client = auth_client(&key);
    // Multi-turn input array: user message + prior function_call +
    // function_call_output. Exercises the translate.rs roundtrip
    // for tool-calling conversations.
    let resp = client
        .post(format!("{base}/v1/responses"))
        .json(&serde_json::json!({
            "model": "test",
            "input": [
                {"type": "message", "role": "user", "content": "What's the weather?"},
                {"type": "function_call", "call_id": "c1", "name": "get_weather", "arguments": "{\"loc\":\"NYC\"}"},
                {"type": "function_call_output", "call_id": "c1", "output": "72F sunny"},
            ]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(
        resp.status(),
        503,
        "multi-turn tool input must parse + route"
    );
}

#[tokio::test]
async fn responses_endpoint_with_tools_definition_parses() {
    let (base, key) = spawn_test_server().await;
    let client = auth_client(&key);
    let resp = client
        .post(format!("{base}/v1/responses"))
        .json(&serde_json::json!({
            "model": "test",
            "input": "use the tool",
            "tools": [{
                "type": "function",
                "name": "lookup",
                "description": "find a thing",
                "parameters": {"type": "object", "properties": {"q": {"type": "string"}}}
            }],
            "tool_choice": "auto"
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 503, "tools definition must parse");
}

#[tokio::test]
async fn responses_endpoint_get_unknown_id_404() {
    let (base, key) = spawn_test_server().await;
    let client = auth_client(&key);
    let resp = client
        .get(format!("{base}/v1/responses/resp_does_not_exist_12345"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 404);
}

#[tokio::test]
async fn responses_endpoint_cancel_unknown_id_404() {
    let (base, key) = spawn_test_server().await;
    let client = auth_client(&key);
    let resp = client
        .post(format!(
            "{base}/v1/responses/resp_does_not_exist_12345/cancel"
        ))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 404);
}

#[tokio::test]
async fn responses_endpoint_rejects_missing_input() {
    let (base, key) = spawn_test_server().await;
    let client = auth_client(&key);
    // No `input` and no `model` is invalid per spec.
    let resp = client
        .post(format!("{base}/v1/responses"))
        .json(&serde_json::json!({"model": "test"}))
        .send()
        .await
        .unwrap();
    // Missing-field rejection: 400 (validation) or 422 (json parse).
    let status = resp.status().as_u16();
    assert!(
        status == 400 || status == 422,
        "missing `input` must 400/422, got {status}"
    );
}

#[tokio::test]
async fn responses_endpoint_reasoning_effort_passes_through() {
    let (base, key) = spawn_test_server().await;
    let client = auth_client(&key);
    // reasoning.effort is a known field; must parse and route.
    let resp = client
        .post(format!("{base}/v1/responses"))
        .json(&serde_json::json!({
            "model": "test",
            "input": "think hard",
            "reasoning": {"effort": "high"}
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 503);
}

#[tokio::test]
async fn tool_role_multi_turn_request() {
    let (base, key) = spawn_test_server().await;
    let client = auth_client(&key);
    let resp = client
        .post(format!("{base}/v1/chat/completions"))
        .json(&serde_json::json!({
            "model": "test",
            "messages": [
                {"role": "user", "content": "What's the weather?"},
                {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": "{\"location\":\"NYC\"}"}
                    }]
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "content": "{\"temp\": 72, \"condition\": \"sunny\"}"
                }
            ]
        }))
        .send()
        .await
        .unwrap();
    // 503 = no model, not 400/422 = parsing worked
    assert_eq!(resp.status(), 503);
}
