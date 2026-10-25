//! Shared fixtures for auto-manage unit tests: a `SharedState` + manager over
//! a throwaway database, and a manifest registered from a list of layer ranges.

use std::sync::Arc;
use tokio::sync::{mpsc, watch, Mutex};

use super::scoring::BudgetReport;
use super::{AutoShardManager, DiskSpace, StorageReading};
use crate::config::Config;
use crate::daemon::SharedState;
use crate::identity::Identity;
use crate::inference::executor::ModelExecutor;
use crate::storage::db::Database;
use crate::types::{
    ContributionMode, ModelArchitecture, ModelId, ModelManifest, NodeCapability, NodeId,
    Quantization, ShardId, ShardInfo,
};

const MIB: u64 = 1024 * 1024;

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
        context_length: None,
    };
    let id = manifest.id.clone();
    state.model_registry.register_manifest(manifest);
    id
}

/// A capability advertising `ceiling_mb` for a model's layers and `disk_mb`
/// free — the two figures the shard manager judges a machine by.
pub(super) fn capability(ceiling_mb: Option<u64>, disk_mb: u64) -> NodeCapability {
    NodeCapability {
        coord: None,
        node_id: NodeId([0u8; 32]),
        gpu: None,
        cpu: None,
        ram_total_mb: 0,
        ram_available_mb: 0,
        ram_model_budget_mb: None,
        disk_available_mb: disk_mb,
        bandwidth_mbps: 0.0,
        hosted_shards: vec![],
        max_contribution: crate::types::ContributionLevel::Moderate,
        uptime_seconds: 0,
        version: String::new(),
        region: None,
        est_tokens_per_sec_7b: 0.0,
        os: None,
        observed_latencies: vec![],
        relay_capable: false,
        protocol_version: 0,
        features: 0,
        relay_reservations: vec![],
        anchor_mode: false,
        can_serve_inference: true,
        resident_layers: Vec::new(),
        context_ceiling_tokens: None,
        model_load_ms_per_gib: None,
        model_memory_ceiling_mb: ceiling_mb,
        machine_room_beside_card_mb: None,
        processor_tokens_per_sec_7b: None,
    }
}

/// A connected peer `[byte; 32]` advertising [`capability`].
pub(super) fn peer_with_disk(
    state: &Arc<SharedState>,
    byte: u8,
    ceiling_mb: Option<u64>,
    disk_mb: u64,
) -> NodeId {
    let id = NodeId([byte; 32]);
    state.peer_registry.insert(
        id.clone(),
        crate::types::PeerInfo {
            node_id: id.clone(),
            addresses: vec![],
            capability: Some(capability(ceiling_mb, disk_mb)),
            last_seen: chrono::Utc::now(),
            latency_ms: Some(10),
            trust_score: 0.8,
            peer_id_bytes: None,
            ack_srtt_ms: None,
            active_request_count: 0,
            first_seen: 0,
            verified_transaction_count: 0,
            is_lan_peer: false,
            goodput_bytes_per_sec: None,
            goodput_samples: 0,
        },
    );
    state.connected_node_ids.insert(id.clone());
    id
}

/// Record `node` as a holder of `parts` of `mid`.
pub(super) fn holds(state: &Arc<SharedState>, mid: &ModelId, node: &NodeId, parts: &[u32]) {
    for &index in parts {
        state.model_registry.record_shard_holder(
            ShardId {
                model_id: mid.clone(),
                index,
            },
            node.clone(),
        );
    }
}

/// The parts the download pass offers, in index order.
pub(super) fn offered(manager: &AutoShardManager, state: &Arc<SharedState>) -> Vec<u32> {
    let local = state.identity.node_id().clone();
    let mut parts: Vec<u32> = manager
        .gather_candidates(&local, 0)
        .into_iter()
        .map(|c| c.shard_index)
        .collect();
    parts.sort_unstable();
    parts
}

/// The parts one cycle of the download pass actually starts.
pub(super) fn fetched(manager: &AutoShardManager, state: &Arc<SharedState>) -> Vec<u32> {
    fetched_with_pool_vram(manager, state, 0)
        .into_iter()
        .map(|(_, index)| index)
        .collect()
}

/// [`fetched`] with the swarm's graphics memory at `pool_vram_mb`, naming each
/// part's model — for a cycle choosing between models.
pub(super) fn fetched_with_pool_vram(
    manager: &AutoShardManager,
    state: &Arc<SharedState>,
    pool_vram_mb: u64,
) -> Vec<(ModelId, u32)> {
    let reading = StorageReading {
        max_storage_mb: 100_000,
        max_disk_mb: 500_000,
        contribution: ContributionMode::Maximum,
        disk: Some(DiskSpace {
            free_bytes: 400_000 * MIB,
            total_bytes: 1_000_000 * MIB,
        }),
        held_bytes: 0,
    };
    let report = BudgetReport {
        held_bytes: 0,
        held_shards: 0,
        budget: reading.budget(),
        max_shards: 0,
        max_shards_reached: false,
        reading,
    };
    let local = state.identity.node_id().clone();
    manager
        .select_within_budget(manager.gather_candidates(&local, pool_vram_mb), &report, 0)
        .into_iter()
        .map(|c| (c.model_id, c.shard_index))
        .collect()
}
