//! Integration tests for config hot-reload (Phase 11).
//!
//! Tests that operational parameters can be reloaded from a config file
//! at runtime and that invalid configs are rejected gracefully.

use std::sync::Arc;

use tokio::sync::Mutex;

use swarmllm::config::{reload_operational_params, Config, OperationalParams};
use swarmllm::daemon::SharedState;
use swarmllm::identity::Identity;
use swarmllm::inference::executor::ModelExecutor;
use swarmllm::storage::db::Database;

/// Test that modifying a config file and calling reload picks up changes.
/// Simulates the daemon's hot-reload path: write new TOML → reload → verify.
#[tokio::test]
async fn test_config_hot_reload() {
    let dir = tempfile::tempdir().unwrap();
    let config_path = dir.path().join("config.toml");

    // Write initial config
    let initial_toml = r#"
[inference]
max_concurrent_requests = 8
max_batch_size = 4
session_timeout_seconds = 600

[auto_manage]
interval_minutes = 30

[network]
max_peers = 50
"#;
    std::fs::write(&config_path, initial_toml).unwrap();

    let params1 = reload_operational_params(&config_path).unwrap();
    assert_eq!(params1.max_concurrent_requests, 8);
    assert_eq!(params1.max_batch_size, 4);
    assert_eq!(params1.max_peers, 50);

    // Update the config file with new values
    let updated_toml = r#"
[inference]
max_concurrent_requests = 16
max_batch_size = 8
session_timeout_seconds = 1200

[auto_manage]
interval_minutes = 15

[network]
max_peers = 100
"#;
    std::fs::write(&config_path, updated_toml).unwrap();

    // Reload should pick up the new values
    let params2 = reload_operational_params(&config_path).unwrap();
    assert_eq!(params2.max_concurrent_requests, 16);
    assert_eq!(params2.max_batch_size, 8);
    assert_eq!(params2.max_peers, 100);
    assert_eq!(params2.auto_manage_interval_minutes, 15);
    assert_eq!(params2.session_timeout_secs, 1200);

    // Params should differ from the original
    assert_ne!(params1, params2);
}

/// Test that an invalid config file is rejected gracefully without panic.
#[tokio::test]
async fn test_config_reload_invalid_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let config_path = dir.path().join("config.toml");

    // Write invalid TOML
    std::fs::write(&config_path, "this is not valid { toml [[[").unwrap();

    let result = reload_operational_params(&config_path);
    assert!(result.is_err());

    // Missing file should also fail
    let missing_path = dir.path().join("nonexistent.toml");
    let result = reload_operational_params(&missing_path);
    assert!(result.is_err());
}

/// Test that apply_config_reload notifies subscribers via the watch channel.
#[tokio::test]
async fn test_config_reload_notifies_subscribers() {
    let config = Config::default();
    let identity = Identity::generate();
    let db = Database::open_temp().unwrap();
    let executor = Arc::new(Mutex::new(ModelExecutor::new()));

    let (shared_state, _shutdown_rx) =
        SharedState::new(config.clone(), identity, db, executor, None);

    // Subscribe to config changes
    let mut rx = shared_state.config_watch_rx();

    // Get the initial value
    let initial = rx.borrow().clone();

    // Apply new operational params
    let new_params = OperationalParams {
        max_concurrent_requests: 32,
        auto_manage_interval_minutes: 5,
        max_batch_size: 16,
        max_peers: 200,
        session_timeout_secs: 3600,
    };

    shared_state.apply_config_reload(new_params.clone());

    // Subscriber should see the update
    rx.changed().await.unwrap();
    let received = rx.borrow().clone();
    assert_eq!(received.max_concurrent_requests, 32);
    assert_eq!(received.max_batch_size, 16);
    assert_eq!(received.max_peers, 200);
    assert_ne!(received, initial);
}
