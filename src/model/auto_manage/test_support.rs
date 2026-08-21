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
    config: Config,
) -> (Arc<SharedState>, AutoShardManager) {
    let identity = Identity::generate();
    let temp = tempfile::tempdir().unwrap();
    let db = Database::open(temp.path()).unwrap();
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
    let shards: Vec<ShardInfo> = shard_ranges
        .iter()
        .enumerate()
        .map(|(i, &(start, end))| ShardInfo {
            index: i as u32,
            layer_range: (start, end),
            size_bytes: 100_000_000,
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
