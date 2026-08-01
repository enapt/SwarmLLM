//! Building a `SharedState` must not write to the data directory.
//!
//! **This test is an integration test on purpose.** The bug it guards against
//! was "fixed" with a `#[cfg(test)]` no-op writer, which only applies while the
//! library is compiled as its own test binary. An integration test links the
//! library built WITHOUT `cfg(test)`, so the real writer stayed live here and
//! `cargo test` kept overwriting a running node's `api_key` file — the daemon
//! carried on using the key in its database, and every CLI call, dashboard
//! request and saved token started returning 401 with nothing in the log.
//!
//! Hit on 2026-07-31, believed fixed, then hit again on 2026-08-01. Putting the
//! guard in a `--lib` test would reproduce exactly the blind spot that let it
//! recur, so it lives here.

use std::sync::Arc;

use swarmllm::config::Config;
use swarmllm::daemon::SharedState;
use swarmllm::identity::Identity;
use swarmllm::inference::executor::ModelExecutor;
use swarmllm::storage::db::Database;
use tokio::sync::Mutex;

#[test]
fn constructing_shared_state_does_not_write_an_api_key_file() {
    let data_dir = tempfile::tempdir().expect("tempdir");
    let db_dir = tempfile::tempdir().expect("db tempdir");

    let mut config = Config::default();
    config.node.data_dir = data_dir.path().to_path_buf();

    let identity = Identity::generate();
    let db = Database::open(db_dir.path()).expect("open db");
    let executor = Arc::new(Mutex::new(ModelExecutor::new()));

    let (state, _shutdown_rx, _dht_rx) = SharedState::new(config, identity, db, executor, None);

    // A key is still resolved — the daemon needs one.
    assert!(
        !state.api_key.is_empty(),
        "a key must still be resolved for the daemon to use"
    );

    // ...but resolving it must not have touched the filesystem. Only the
    // daemon's own startup publishes the file, via `publish_api_key_file`.
    let key_file = data_dir.path().join("api_key");
    assert!(
        !key_file.exists(),
        "SharedState::new wrote {} — on a machine running a node this is the \
         REAL data dir, and the write silently breaks its API key",
        key_file.display()
    );
}
