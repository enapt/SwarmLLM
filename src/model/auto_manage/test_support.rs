//! Shared fixtures for auto-manage unit tests: a `SharedState` + manager over
//! a throwaway database, and a manifest registered from a list of layer ranges.

use std::sync::Arc;
use tokio::sync::{mpsc, watch, Mutex};

use super::AutoShardManager;
use crate::config::Config;
use crate::daemon::SharedState;
use crate::identity::Identity;
use crate::inference::executor::ModelExecutor;
use crate::storage::db::Database;
use crate::types::{ModelArchitecture, ModelId, ModelManifest, NodeId, Quantization, ShardInfo};

pub(super) fn make_test_manager() -> (Arc<SharedState>, AutoShardManager) {
    make_test_manager_with_config(Config::default())
}

pub(super) fn make_test_manager_with_config(
    mut config: Config,
) -> (Arc<SharedState>, AutoShardManager) {
    let identity = Identity::generate();
    // `keep()` rather than holding the guard: the directory has to outlive this
    // function, and the fixture returns only the state and the manager. A few
    // empty directories per test run in the OS temp area is the price.
    //
    // **`data_dir` MUST point at it.** It did not until 2026-09-17, so
    // `Config::default()` left every fixture pointing at the developer's REAL
    // `~/.local/share/swarmllm`. That was harmless only for as long as nothing
    // in the tested path touched the filesystem; `held_disk_bytes` measures the
    // models directory, so a unit test asserting on a node "holding 14 GB"
    // silently read this machine's actual 33 GB instead. A fixture that names a
    // real directory is a test that depends on the machine it runs on.
    let temp = tempfile::tempdir().unwrap().keep();
    let db = Database::open(&temp).unwrap();
    config.node.data_dir = temp;
    let executor = Arc::new(Mutex::new(ModelExecutor::new()));
    let (state, _, _) = SharedState::new(config, identity, db, executor, None);
    let (net_tx, _net_rx) = mpsc::channel(16);
    let (_shutdown_tx, shutdown_rx) = watch::channel(false);
    let manager = AutoShardManager::new(state.clone(), net_tx, shutdown_rx);
    (state, manager)
}

pub(super) fn register_manifest_with_shards(
    state: &Arc<SharedState>,
    model_id: &str,
    num_layers: u32,
    shard_ranges: &[(u32, u32)],
) -> ModelId {
    register_manifest_with_sized_shards(state, model_id, num_layers, shard_ranges, 100_000_000)
}

/// Put `size_bytes`-long shard files on disk for a model this node holds.
///
/// SPARSE — `set_len` sizes the file without writing blocks, so a test can hold
/// "14 GB" for nothing. `held_disk_bytes` reads `metadata().len()`, which is the
/// logical length, so this is indistinguishable from real holdings to the code
/// under test and costs no disk.
///
/// Needed because the storage budget measures the DIRECTORY, not the manifest:
/// registering a shard in the registry no longer makes the node "hold" bytes.
pub(super) fn write_sparse_shards(
    state: &Arc<SharedState>,
    model_id: &ModelId,
    indices: impl IntoIterator<Item = u32>,
    size_bytes: u64,
) {
    let dir = crate::model::shard::model_dir(&state.config.node.data_dir, &model_id.0);
    std::fs::create_dir_all(&dir).unwrap();
    for index in indices {
        let path = dir.join(crate::model::shard::shard_filename(index));
        let f = std::fs::File::create(&path).unwrap();
        f.set_len(size_bytes).unwrap();
    }
}

/// Like [`register_manifest_with_shards`], with every shard `size_bytes` long —
/// for a test that needs the node to HOLD a known number of bytes.
pub(super) fn register_manifest_with_sized_shards(
    state: &Arc<SharedState>,
    model_id: &str,
    num_layers: u32,
    shard_ranges: &[(u32, u32)],
    size_bytes: u64,
) -> ModelId {
    let shards: Vec<ShardInfo> = shard_ranges
        .iter()
        .enumerate()
        .map(|(i, &(start, end))| ShardInfo {
            index: i as u32,
            layer_range: (start, end),
            size_bytes,
            hash: [0u8; 32],
            tensors: vec![],
        })
        .collect();
    let manifest = ModelManifest {
        id: ModelId(model_id.into()),
        name: format!("Test {model_id}"),
        architecture: ModelArchitecture::Llama,
        num_layers,
        num_params_billions: 1.0,
        quantization: Quantization::Q4KM,
        total_size_bytes: shards.iter().map(|s| s.size_bytes).sum(),
        shard_count: shards.len() as u32,
        shards,
        tokenizer_hash: [0u8; 32],
        manifest_hash: [0u8; 32],
        publisher: NodeId([0u8; 32]),
        publish_date: chrono::Utc::now(),
        license: "MIT".into(),
        mmproj: None,
    };
    let id = manifest.id.clone();
    state.model_registry.register_manifest(manifest);
    id
}
