use super::*;
use crate::config::Config;
use crate::identity::Identity;
use crate::inference::executor::ModelExecutor;
use crate::storage::db::Database;
use crate::types::*;
use std::sync::Arc;
use tokio::sync::Mutex;

fn make_shared_state() -> Arc<SharedState> {
    make_shared_state_with(|_| {})
}

/// `make_shared_state` with a hook to tweak the config before the state is
/// built. Needed for tensor parallelism, which is opt-in (`inference.
/// tensor_parallel`, default false) since R146.
fn make_shared_state_with(tweak: impl FnOnce(&mut Config)) -> Arc<SharedState> {
    let mut config = Config::default();
    tweak(&mut config);
    let identity = Identity::generate();
    let temp = tempfile::tempdir().unwrap();
    let db = Database::open(temp.path()).unwrap();
    let executor = Arc::new(Mutex::new(ModelExecutor::new()));
    let (state, _, _) = SharedState::new(config, identity, db, executor, None);
    state
}

fn make_manifest(model_id: &str, num_layers: u32, shards: Vec<ShardInfo>) -> ModelManifest {
    ModelManifest {
        id: ModelId(model_id.into()),
        name: "Test Model".into(),
        architecture: ModelArchitecture::Llama,
        num_layers,
        num_params_billions: 7.0,
        quantization: Quantization::Q4KM,
        total_size_bytes: 4_000_000_000,
        shard_count: shards.len() as u32,
        shards,
        tokenizer_hash: [0u8; 32],
        manifest_hash: [0u8; 32],
        publisher: NodeId([0u8; 32]),
        publish_date: chrono::Utc::now(),
        license: "MIT".into(),
        mmproj: None,
    }
}

#[test]
fn assemble_single_node_pipeline() {
    let state = make_shared_state();
    let local_id = state.identity.node_id().clone();

    let shards = vec![ShardInfo {
        index: 0,
        layer_range: (0, 32),
        size_bytes: 4_000_000_000,
        hash: [0u8; 32],
        tensors: vec![],
    }];
    let manifest = make_manifest("test-model", 32, shards);
    state.model_registry.register_manifest(manifest);

    // Register local node as shard holder
    let shard_id = ShardId {
        model_id: ModelId("test-model".into()),
        index: 0,
    };
    state
        .model_registry
        .record_shard_holder(shard_id, local_id.clone());

    let scheduler = PipelineScheduler::new(state);
    let assignment = scheduler
        .assemble_pipeline(&ModelId("test-model".into()), &local_id)
        .unwrap();

    assert_eq!(assignment.segments.len(), 1);
    assert_eq!(assignment.segments[0].layer_range, (0, 32));
    assert_eq!(assignment.segments[0].node_id, local_id);
}

#[test]
fn assemble_multi_node_pipeline() {
    let state = make_shared_state();
    let local_id = state.identity.node_id().clone();
    let node_b = NodeId([2u8; 32]);
    let node_c = NodeId([3u8; 32]);

    let shards = vec![
        ShardInfo {
            index: 0,
            layer_range: (0, 16),
            size_bytes: 2_000_000_000,
            hash: [0u8; 32],
            tensors: vec![],
        },
        ShardInfo {
            index: 1,
            layer_range: (16, 32),
            size_bytes: 2_000_000_000,
            hash: [0u8; 32],
            tensors: vec![],
        },
    ];
    let manifest = make_manifest("test-model", 32, shards);
    state.model_registry.register_manifest(manifest);

    // Node B has shard 0, Node C has shard 1
    state.model_registry.record_shard_holder(
        ShardId {
            model_id: ModelId("test-model".into()),
            index: 0,
        },
        node_b.clone(),
    );
    state.model_registry.record_shard_holder(
        ShardId {
            model_id: ModelId("test-model".into()),
            index: 1,
        },
        node_c.clone(),
    );

    // Add peer info so latencies are known
    state.peer_registry.insert(
        node_b.clone(),
        PeerInfo {
            node_id: node_b.clone(),
            addresses: vec![],
            capability: None,
            last_seen: chrono::Utc::now(),
            latency_ms: Some(10),
            trust_score: 0.8,
            peer_id_bytes: None,
            active_request_count: 0,
            first_seen: 0,
            verified_transaction_count: 0,
            is_lan_peer: false,
        },
    );
    state.connected_node_ids.insert(node_b.clone());
    state.peer_registry.insert(
        node_c.clone(),
        PeerInfo {
            node_id: node_c.clone(),
            addresses: vec![],
            capability: None,
            last_seen: chrono::Utc::now(),
            latency_ms: Some(15),
            trust_score: 0.9,
            peer_id_bytes: None,
            active_request_count: 0,
            first_seen: 0,
            verified_transaction_count: 0,
            is_lan_peer: false,
        },
    );
    state.connected_node_ids.insert(node_c.clone());

    let scheduler = PipelineScheduler::new(state);
    let assignment = scheduler
        .assemble_pipeline(&ModelId("test-model".into()), &local_id)
        .unwrap();

    assert_eq!(assignment.segments.len(), 2);
    assert_eq!(assignment.segments[0].layer_range, (0, 16));
    assert_eq!(assignment.segments[1].layer_range, (16, 32));
}

#[test]
fn fails_when_model_not_found() {
    let state = make_shared_state();
    let local_id = state.identity.node_id().clone();
    let scheduler = PipelineScheduler::new(state);

    let result = scheduler.assemble_pipeline(&ModelId("nonexistent".into()), &local_id);
    assert!(result.is_err());
}

#[test]
fn fails_when_no_shard_holders() {
    let state = make_shared_state();
    let local_id = state.identity.node_id().clone();

    let manifest = make_manifest(
        "orphan-model",
        32,
        vec![ShardInfo {
            index: 0,
            layer_range: (0, 32),
            size_bytes: 4_000_000_000,
            hash: [0u8; 32],
            tensors: vec![],
        }],
    );
    state.model_registry.register_manifest(manifest);

    let scheduler = PipelineScheduler::new(state);
    let result = scheduler.assemble_pipeline(&ModelId("orphan-model".into()), &local_id);
    assert!(result.is_err());
}

#[test]
fn merge_contiguous_segments_same_node() {
    let node = NodeId([1u8; 32]);
    let shard = ShardId {
        model_id: ModelId("m".into()),
        index: 0,
    };
    let segments = vec![
        PipelineSegment {
            node_id: node.clone(),
            shard_id: shard.clone(),
            layer_range: (0, 2),
        },
        PipelineSegment {
            node_id: node.clone(),
            shard_id: shard.clone(),
            layer_range: (2, 4),
        },
    ];
    let merged = PipelineScheduler::merge_contiguous(segments);
    assert_eq!(merged.len(), 1);
    assert_eq!(merged[0].layer_range, (0, 4));
}

#[test]
fn greedy_assign_multi_range_candidate() {
    // Test that a candidate with multiple non-contiguous ranges can
    // serve multiple pipeline segments for the same model.
    let state = make_shared_state();
    let scheduler = PipelineScheduler::new(state);

    // Candidate A: layers [0,2) and [10,14)
    // Candidate B: layers [2,10)
    let candidates = vec![
        NodeCandidate {
            node_id: NodeId([1u8; 32]),
            shard_id: ShardId {
                model_id: ModelId("test".into()),
                index: 0,
            },
            available_ranges: vec![(0, 2), (10, 14)],
            latency_ms: 0,
            load: 0.0,
            trust_score: 1.0,
            can_be_first: true,
            can_be_last: true,
            region_score: 1.0,
            est_tokens_per_sec: 0.0,
            observed_latency_ms_per_layer: None,
            is_pool_member: false,
        },
        NodeCandidate {
            node_id: NodeId([2u8; 32]),
            shard_id: ShardId {
                model_id: ModelId("test".into()),
                index: 1,
            },
            available_ranges: vec![(2, 10)],
            latency_ms: 10,
            load: 0.0,
            trust_score: 0.8,
            can_be_first: false,
            can_be_last: false,
            region_score: 0.7,
            est_tokens_per_sec: 0.0,
            observed_latency_ms_per_layer: None,
            is_pool_member: false,
        },
    ];

    let segments = scheduler.greedy_assign(14, &candidates, false).unwrap();
    // Should produce 3 segments: [0,2) on A, [2,10) on B, [10,14) on A
    assert_eq!(segments.len(), 3);
    assert_eq!(segments[0].layer_range, (0, 2));
    assert_eq!(segments[0].node_id, NodeId([1u8; 32]));
    assert_eq!(segments[1].layer_range, (2, 10));
    assert_eq!(segments[1].node_id, NodeId([2u8; 32]));
    assert_eq!(segments[2].layer_range, (10, 14));
    assert_eq!(segments[2].node_id, NodeId([1u8; 32]));

    // After merging, same-node contiguous segments collapse
    let merged = PipelineScheduler::merge_contiguous(segments);
    // A's [0,2) and [10,14) are NOT contiguous → no merge → still 3 segments
    assert_eq!(merged.len(), 3);
}

#[test]
fn prefers_lower_load_node() {
    // Two nodes with identical latency and trust but different load.
    // The scheduler should prefer the node with lower load.
    let state = make_shared_state();
    let local_id = state.identity.node_id().clone();
    let node_a = NodeId([10u8; 32]);
    let node_b = NodeId([11u8; 32]);

    let shards = vec![ShardInfo {
        index: 0,
        layer_range: (0, 16),
        size_bytes: 2_000_000_000,
        hash: [0u8; 32],
        tensors: vec![],
    }];
    let manifest = make_manifest("load-test", 16, shards);
    state.model_registry.register_manifest(manifest);

    // Both nodes hold shard 0
    let shard_id = ShardId {
        model_id: ModelId("load-test".into()),
        index: 0,
    };
    state
        .model_registry
        .record_shard_holder(shard_id.clone(), node_a.clone());
    state
        .model_registry
        .record_shard_holder(shard_id, node_b.clone());

    // Same latency and trust, but different load via active_request_count
    state.peer_registry.insert(
        node_a.clone(),
        PeerInfo {
            node_id: node_a.clone(),
            addresses: vec![],
            capability: None,
            last_seen: chrono::Utc::now(),
            latency_ms: Some(20),
            trust_score: 0.8,
            peer_id_bytes: None,
            active_request_count: 10, // high load
            first_seen: 0,
            verified_transaction_count: 0,
            is_lan_peer: false,
        },
    );
    state.connected_node_ids.insert(node_a.clone());
    state.peer_registry.insert(
        node_b.clone(),
        PeerInfo {
            node_id: node_b.clone(),
            addresses: vec![],
            capability: None,
            last_seen: chrono::Utc::now(),
            latency_ms: Some(20),
            trust_score: 0.8,
            peer_id_bytes: None,
            active_request_count: 1, // low load
            first_seen: 0,
            verified_transaction_count: 0,
            is_lan_peer: false,
        },
    );
    state.connected_node_ids.insert(node_b.clone());

    let scheduler = PipelineScheduler::new(state);
    let assignment = scheduler
        .assemble_pipeline(&ModelId("load-test".into()), &local_id)
        .unwrap();

    // Node B (low load) should be selected over Node A (high load)
    assert_eq!(assignment.segments.len(), 1);
    assert_eq!(assignment.segments[0].node_id, node_b);
}

/// Two-shard topology in which the local node holds only the first half of
/// the model: local + `node_b` cover layers 0..16, `node_c` covers 16..32.
///
/// This is the only shape where a tensor-parallel group is worth considering.
/// When the local node covers every layer it can serve alone, and pulling a
/// peer in can only add latency and a failure mode.
fn setup_tp_split_topology(
    state: &Arc<SharedState>,
    model_id: &str,
    node_b: &NodeId,
    node_b_is_lan: bool,
    node_b_latency_ms: u32,
) {
    let local_id = state.identity.node_id().clone();
    let node_c = NodeId([99u8; 32]);

    let shards = vec![
        ShardInfo {
            index: 0,
            layer_range: (0, 16),
            size_bytes: 2_000_000_000,
            hash: [0u8; 32],
            tensors: vec![],
        },
        ShardInfo {
            index: 1,
            layer_range: (16, 32),
            size_bytes: 2_000_000_000,
            hash: [1u8; 32],
            tensors: vec![],
        },
    ];
    state
        .model_registry
        .register_manifest(make_manifest(model_id, 32, shards));

    let shard0 = ShardId {
        model_id: ModelId(model_id.into()),
        index: 0,
    };
    let shard1 = ShardId {
        model_id: ModelId(model_id.into()),
        index: 1,
    };
    state
        .model_registry
        .record_shard_holder(shard0.clone(), local_id);
    state
        .model_registry
        .record_shard_holder(shard0, node_b.clone());
    state
        .model_registry
        .record_shard_holder(shard1, node_c.clone());

    for (node, is_lan, latency) in [
        (node_b.clone(), node_b_is_lan, node_b_latency_ms),
        (node_c, false, 50),
    ] {
        state.peer_registry.insert(
            node.clone(),
            PeerInfo {
                node_id: node.clone(),
                addresses: vec![],
                capability: None,
                last_seen: chrono::Utc::now(),
                latency_ms: Some(latency),
                trust_score: 0.9,
                peer_id_bytes: None,
                active_request_count: 0,
                first_seen: 0,
                verified_transaction_count: 0,
                is_lan_peer: is_lan,
            },
        );
        state.connected_node_ids.insert(node);
    }
}

#[test]
fn no_tp_group_when_local_node_covers_every_layer() {
    // Regression (user bug report 2026-07-21 #1): a fully-replicated model on
    // two LAN peers used to form a tensor-parallel group anyway. The local
    // node could have answered alone; instead the request died with
    // "AllReduce timeout after 10s for layer 0" when the peer went quiet.
    // Full local coverage must take the single-local-segment fast path with
    // NO TP group — even with tensor parallelism explicitly enabled.
    let state = make_shared_state_with(|c| c.inference.tensor_parallel = true);
    let local_id = state.identity.node_id().clone();
    let node_b = NodeId([20u8; 32]);

    let shards = vec![ShardInfo {
        index: 0,
        layer_range: (0, 32),
        size_bytes: 4_000_000_000,
        hash: [0u8; 32],
        tensors: vec![],
    }];
    state
        .model_registry
        .register_manifest(make_manifest("tp-full-model", 32, shards));

    let shard_id = ShardId {
        model_id: ModelId("tp-full-model".into()),
        index: 0,
    };
    state
        .model_registry
        .record_shard_holder(shard_id.clone(), local_id.clone());
    state
        .model_registry
        .record_shard_holder(shard_id, node_b.clone());

    state.peer_registry.insert(
        node_b.clone(),
        PeerInfo {
            node_id: node_b.clone(),
            addresses: vec![],
            capability: None,
            last_seen: chrono::Utc::now(),
            latency_ms: Some(1),
            trust_score: 0.9,
            peer_id_bytes: None,
            active_request_count: 0,
            first_seen: 0,
            verified_transaction_count: 0,
            is_lan_peer: true,
        },
    );
    state.connected_node_ids.insert(node_b);

    let scheduler = PipelineScheduler::new(state);
    let assignment = scheduler
        .assemble_pipeline(&ModelId("tp-full-model".into()), &local_id)
        .unwrap();

    assert_eq!(assignment.segments.len(), 1);
    assert_eq!(assignment.segments[0].node_id, local_id);
    assert!(
        assignment.tp_groups.is_empty(),
        "local node holds every layer — no peer should be pulled into the request"
    );
}

#[test]
fn detects_tp_group_for_lan_peers_when_enabled() {
    let state = make_shared_state_with(|c| c.inference.tensor_parallel = true);
    let local_id = state.identity.node_id().clone();
    let node_b = NodeId([20u8; 32]);
    setup_tp_split_topology(&state, "tp-lan-model", &node_b, true, 1);

    let scheduler = PipelineScheduler::new(state);
    let assignment = scheduler
        .assemble_pipeline(&ModelId("tp-lan-model".into()), &local_id)
        .unwrap();

    assert_eq!(assignment.tp_groups.len(), 1);
    assert_eq!(assignment.tp_groups[0].nodes.len(), 2);
    assert!(assignment.tp_groups[0].nodes.contains(&local_id));
    assert!(assignment.tp_groups[0].nodes.contains(&node_b));
    assert_eq!(assignment.tp_groups[0].layer_range, (0, 16));
}

#[test]
fn no_tp_group_when_tensor_parallel_disabled() {
    // Same topology as the test above, but with the default config. TP is
    // opt-in: per-layer AllReduce over Ethernet costs more than the compute
    // it splits for anything short of a large model on a very fast LAN.
    let state = make_shared_state();
    assert!(
        !state.config.inference.tensor_parallel,
        "tensor parallelism must default to off"
    );
    let local_id = state.identity.node_id().clone();
    let node_b = NodeId([20u8; 32]);
    setup_tp_split_topology(&state, "tp-off-model", &node_b, true, 1);

    let scheduler = PipelineScheduler::new(state);
    let assignment = scheduler
        .assemble_pipeline(&ModelId("tp-off-model".into()), &local_id)
        .unwrap();

    assert!(assignment.tp_groups.is_empty());
}

#[test]
fn no_tp_group_for_wan_peers() {
    // Even with TP enabled and a genuinely split model, a high-latency peer
    // is excluded — AllReduce round trips would dominate.
    let state = make_shared_state_with(|c| c.inference.tensor_parallel = true);
    let local_id = state.identity.node_id().clone();
    let node_b = NodeId([21u8; 32]);
    setup_tp_split_topology(&state, "wan-model", &node_b, false, 100);

    let scheduler = PipelineScheduler::new(state);
    let assignment = scheduler
        .assemble_pipeline(&ModelId("wan-model".into()), &local_id)
        .unwrap();

    assert!(assignment.tp_groups.is_empty());
}

#[test]
fn tp_group_from_low_latency_only() {
    // TP group should form when the peer has low measured latency but was NOT
    // discovered via mDNS (is_lan_peer = false).
    let state = make_shared_state_with(|c| c.inference.tensor_parallel = true);
    let local_id = state.identity.node_id().clone();
    let node_b = NodeId([22u8; 32]);
    setup_tp_split_topology(&state, "latency-tp-model", &node_b, false, 2);

    let scheduler = PipelineScheduler::new(state);
    let assignment = scheduler
        .assemble_pipeline(&ModelId("latency-tp-model".into()), &local_id)
        .unwrap();

    assert_eq!(assignment.tp_groups.len(), 1);
    assert_eq!(assignment.tp_groups[0].nodes.len(), 2);
}

#[test]
fn tp_group_rank_of() {
    let group = TensorParallelGroup {
        nodes: vec![NodeId([1u8; 32]), NodeId([2u8; 32]), NodeId([3u8; 32])],
        layer_range: (0, 32),
        shard_ids: vec![],
    };
    assert_eq!(group.rank_of(&NodeId([1u8; 32])), Some(0));
    assert_eq!(group.rank_of(&NodeId([2u8; 32])), Some(1));
    assert_eq!(group.rank_of(&NodeId([3u8; 32])), Some(2));
    assert_eq!(group.rank_of(&NodeId([4u8; 32])), None);
    assert_eq!(group.tp_size(), 3);
}

#[test]
fn encrypted_pipeline_forces_local_first_and_last() {
    // With encrypted_pipeline=true, the local node MUST be assigned
    // both the first and last segments (boomerang topology).
    let state = make_shared_state();
    let local_id = state.identity.node_id().clone();
    let node_b = NodeId([30u8; 32]);

    // 3 shards: local has shard 0 + shard 2, remote has shard 1
    let shards = vec![
        ShardInfo {
            index: 0,
            layer_range: (0, 10),
            size_bytes: 1_000_000_000,
            hash: [0u8; 32],
            tensors: vec![],
        },
        ShardInfo {
            index: 1,
            layer_range: (10, 20),
            size_bytes: 1_000_000_000,
            hash: [0u8; 32],
            tensors: vec![],
        },
        ShardInfo {
            index: 2,
            layer_range: (20, 30),
            size_bytes: 1_000_000_000,
            hash: [0u8; 32],
            tensors: vec![],
        },
    ];
    let manifest = make_manifest("encrypted-model", 30, shards);
    state.model_registry.register_manifest(manifest);

    // Local node has first and last shards
    state.model_registry.record_shard_holder(
        ShardId {
            model_id: ModelId("encrypted-model".into()),
            index: 0,
        },
        local_id.clone(),
    );
    state.model_registry.record_shard_holder(
        ShardId {
            model_id: ModelId("encrypted-model".into()),
            index: 2,
        },
        local_id.clone(),
    );
    // Remote node has middle shard
    state.model_registry.record_shard_holder(
        ShardId {
            model_id: ModelId("encrypted-model".into()),
            index: 1,
        },
        node_b.clone(),
    );

    state.peer_registry.insert(
        node_b.clone(),
        PeerInfo {
            node_id: node_b.clone(),
            addresses: vec![],
            capability: None,
            last_seen: chrono::Utc::now(),
            latency_ms: Some(10),
            trust_score: 0.8,
            peer_id_bytes: None,
            active_request_count: 0,
            first_seen: 0,
            verified_transaction_count: 0,
            is_lan_peer: false,
        },
    );
    state.connected_node_ids.insert(node_b.clone());

    // Enable encrypted pipeline for this model
    state
        .encrypted_pipeline_models
        .insert(ModelId("encrypted-model".into()), true);

    let scheduler = PipelineScheduler::new(state);
    let assignment = scheduler
        .assemble_pipeline(&ModelId("encrypted-model".into()), &local_id)
        .unwrap();

    // Pipeline should be: local [0,10) → remote [10,20) → local [20,30)
    assert_eq!(assignment.segments.len(), 3);
    assert_eq!(assignment.segments[0].node_id, local_id);
    assert_eq!(assignment.segments[0].layer_range, (0, 10));
    assert_eq!(assignment.segments[1].node_id, node_b);
    assert_eq!(assignment.segments[1].layer_range, (10, 20));
    assert_eq!(assignment.segments[2].node_id, local_id);
    assert_eq!(assignment.segments[2].layer_range, (20, 30));
}

#[test]
fn encrypted_pipeline_fails_without_first_shard() {
    // If the local node doesn't have shard 0, encrypted pipeline should fail.
    let state = make_shared_state();
    let local_id = state.identity.node_id().clone();
    let node_b = NodeId([31u8; 32]);

    let shards = vec![
        ShardInfo {
            index: 0,
            layer_range: (0, 16),
            size_bytes: 2_000_000_000,
            hash: [0u8; 32],
            tensors: vec![],
        },
        ShardInfo {
            index: 1,
            layer_range: (16, 32),
            size_bytes: 2_000_000_000,
            hash: [0u8; 32],
            tensors: vec![],
        },
    ];
    let manifest = make_manifest("enc-fail", 32, shards);
    state.model_registry.register_manifest(manifest);

    // Remote has both shards, local has neither
    state.model_registry.record_shard_holder(
        ShardId {
            model_id: ModelId("enc-fail".into()),
            index: 0,
        },
        node_b.clone(),
    );
    state.model_registry.record_shard_holder(
        ShardId {
            model_id: ModelId("enc-fail".into()),
            index: 1,
        },
        node_b.clone(),
    );

    state.peer_registry.insert(
        node_b.clone(),
        PeerInfo {
            node_id: node_b.clone(),
            addresses: vec![],
            capability: None,
            last_seen: chrono::Utc::now(),
            latency_ms: Some(10),
            trust_score: 0.8,
            peer_id_bytes: None,
            active_request_count: 0,
            first_seen: 0,
            verified_transaction_count: 0,
            is_lan_peer: false,
        },
    );
    state.connected_node_ids.insert(node_b.clone());

    state
        .encrypted_pipeline_models
        .insert(ModelId("enc-fail".into()), true);

    let scheduler = PipelineScheduler::new(state);
    let result = scheduler.assemble_pipeline(&ModelId("enc-fail".into()), &local_id);

    assert!(result.is_err());
    let err_msg = format!("{}", result.unwrap_err());
    assert!(
        err_msg.contains("shard 0"),
        "Error should mention shard 0: {err_msg}"
    );
}

#[test]
fn parallax_flag_picks_low_latency_peer_end_to_end() {
    // Integration: flag enabled → scheduler uses parallax routing and picks
    // the lower-latency peer when two remotes both cover all layers.
    let mut config = Config::default();
    config.inference.parallax_routing = true;
    let identity = Identity::generate();
    let temp = tempfile::tempdir().unwrap();
    let db = Database::open(temp.path()).unwrap();
    let executor = Arc::new(Mutex::new(ModelExecutor::new()));
    let (state, _, _) = SharedState::new(config, identity, db, executor, None);
    let local_id = state.identity.node_id().clone();

    let shards = vec![ShardInfo {
        index: 0,
        layer_range: (0, 32),
        size_bytes: 4_000_000_000,
        hash: [0u8; 32],
        tensors: vec![],
    }];
    let manifest = make_manifest("test-model", 32, shards);
    state.model_registry.register_manifest(manifest);

    let slow = NodeId([2u8; 32]);
    let fast = NodeId([3u8; 32]);
    state.model_registry.record_shard_holder(
        ShardId {
            model_id: ModelId("test-model".into()),
            index: 0,
        },
        slow.clone(),
    );
    state.model_registry.record_shard_holder(
        ShardId {
            model_id: ModelId("test-model".into()),
            index: 0,
        },
        fast.clone(),
    );

    state.peer_registry.insert(
        slow.clone(),
        PeerInfo {
            node_id: slow.clone(),
            addresses: vec![],
            capability: None,
            last_seen: chrono::Utc::now(),
            latency_ms: Some(200),
            trust_score: 0.8,
            peer_id_bytes: None,
            active_request_count: 0,
            first_seen: 0,
            verified_transaction_count: 0,
            is_lan_peer: false,
        },
    );
    state.connected_node_ids.insert(slow.clone());
    state.peer_registry.insert(
        fast.clone(),
        PeerInfo {
            node_id: fast.clone(),
            addresses: vec![],
            capability: None,
            last_seen: chrono::Utc::now(),
            latency_ms: Some(10),
            trust_score: 0.8,
            peer_id_bytes: None,
            active_request_count: 0,
            first_seen: 0,
            verified_transaction_count: 0,
            is_lan_peer: false,
        },
    );
    state.connected_node_ids.insert(fast.clone());

    let scheduler = PipelineScheduler::new(state);
    let assignment = scheduler
        .assemble_pipeline(&ModelId("test-model".into()), &local_id)
        .unwrap();
    assert_eq!(assignment.segments.len(), 1);
    assert_eq!(assignment.segments[0].node_id, fast);
}

#[test]
fn peer_segment_latency_ema_math() {
    // Verify the EMA formula: new = 0.3 * sample + 0.7 * old.
    let state = make_shared_state();
    let node = NodeId([7u8; 32]);
    // First sample: 20 ms over 4 layers → 5 ms/layer. No prior → EMA = 5.
    state.record_peer_segment_latency(&node, 20, 4);
    let v1 = state.observed_latency_ms_per_layer(&node).unwrap();
    assert!((v1 - 5.0).abs() < 1e-5, "first sample EMA = {v1}");
    // Second sample: 40 ms over 4 layers → 10 ms/layer. EMA = 0.3*10 + 0.7*5 = 6.5.
    state.record_peer_segment_latency(&node, 40, 4);
    let v2 = state.observed_latency_ms_per_layer(&node).unwrap();
    assert!((v2 - 6.5).abs() < 1e-5, "second sample EMA = {v2}");
    // Width-normalised: a 2-layer segment at 20 ms → 10 ms/layer.
    state.record_peer_segment_latency(&node, 20, 2);
    let v3 = state.observed_latency_ms_per_layer(&node).unwrap();
    // EMA = 0.3*10 + 0.7*6.5 = 7.55.
    assert!((v3 - 7.55).abs() < 1e-5, "third sample EMA = {v3}");
    // Zero-layer guard: no panic, no update.
    state.record_peer_segment_latency(&node, 100, 0);
    let v4 = state.observed_latency_ms_per_layer(&node).unwrap();
    assert!((v4 - 7.55).abs() < 1e-5, "zero-layer update = {v4}");
    // Unknown peer: None.
    let unknown = NodeId([8u8; 32]);
    assert!(state.observed_latency_ms_per_layer(&unknown).is_none());
}

#[test]
fn merge_peer_segment_latency_zero_trust_is_noop() {
    // A zero-trust (or negative) sender cannot influence our EMA in either
    // direction — neither seeding a new entry nor moving an existing one.
    let state = make_shared_state();
    let node = NodeId([11u8; 32]);

    // No entry yet + weight 0 → no insert.
    state.merge_peer_segment_latency(&node, 100.0, 0.0);
    assert!(state.observed_latency_ms_per_layer(&node).is_none());

    // Seed a direct observation, then try to poison it with weight 0.
    state.record_peer_segment_latency(&node, 20, 4);
    let before = state.observed_latency_ms_per_layer(&node).unwrap();
    state.merge_peer_segment_latency(&node, 9999.0, 0.0);
    let after = state.observed_latency_ms_per_layer(&node).unwrap();
    assert!(
        (after - before).abs() < 1e-6,
        "zero-trust merge must be a no-op (before={before}, after={after})"
    );
}

#[test]
fn merge_peer_segment_latency_below_seed_threshold_skips_insert() {
    // A low-trust sender (weight < 0.3) cannot seed a fresh entry, even
    // though it would shift an existing one. Defends against a low-trust
    // peer painting us an out-of-band picture of a stranger.
    let state = make_shared_state();
    let stranger = NodeId([12u8; 32]);

    state.merge_peer_segment_latency(&stranger, 500.0, 0.1);
    assert!(state.observed_latency_ms_per_layer(&stranger).is_none());

    state.merge_peer_segment_latency(&stranger, 500.0, 0.29);
    assert!(state.observed_latency_ms_per_layer(&stranger).is_none());

    // Exactly at the threshold seeds.
    state.merge_peer_segment_latency(&stranger, 500.0, 0.3);
    assert_eq!(
        state.observed_latency_ms_per_layer(&stranger).unwrap(),
        500.0
    );
}

#[test]
fn merge_peer_segment_latency_trust_weighted_ema() {
    // Weight-scaled α: effective_α = 0.3 * weight. Direct samples move
    // the EMA more than foreign samples from less-trusted peers.
    let state = make_shared_state();
    let node = NodeId([13u8; 32]);

    // Start with a direct sample of 10 ms/layer (20 ms over 2 layers).
    state.record_peer_segment_latency(&node, 20, 2);
    let v0 = state.observed_latency_ms_per_layer(&node).unwrap();
    assert!((v0 - 10.0).abs() < 1e-5);

    // Foreign sample of 100 ms/layer at weight 1.0 →
    // effective_α = 0.3, EMA = 0.3*100 + 0.7*10 = 37.0.
    state.merge_peer_segment_latency(&node, 100.0, 1.0);
    let v1 = state.observed_latency_ms_per_layer(&node).unwrap();
    assert!(
        (v1 - 37.0).abs() < 1e-4,
        "weight=1.0 merge, expected 37.0, got {v1}"
    );

    // Reset and try with weight 0.5 →
    // effective_α = 0.15, EMA = 0.15*100 + 0.85*10 = 23.5.
    let state2 = make_shared_state();
    state2.record_peer_segment_latency(&node, 20, 2);
    state2.merge_peer_segment_latency(&node, 100.0, 0.5);
    let v2 = state2.observed_latency_ms_per_layer(&node).unwrap();
    assert!(
        (v2 - 23.5).abs() < 1e-4,
        "weight=0.5 merge, expected 23.5, got {v2}"
    );

    // Verify ordering: higher weight moves the EMA farther from baseline.
    assert!(
        v1 > v2,
        "weight=1.0 ({v1}) should move farther than 0.5 ({v2})"
    );
}

#[test]
fn merge_peer_segment_latency_preserves_direct_observations() {
    // Scenario: we've been sampling a peer directly for a while. A foreign
    // gossip arrives with the same EMA value at trust 1.0. The merge
    // shouldn't move our EMA (sample == old → blend is a no-op).
    let state = make_shared_state();
    let node = NodeId([15u8; 32]);
    // Three direct samples at identical per-layer rate converge exactly.
    for _ in 0..3 {
        state.record_peer_segment_latency(&node, 50, 5);
    }
    let before = state.observed_latency_ms_per_layer(&node).unwrap();
    assert!((before - 10.0).abs() < 1e-5);

    // Foreign reports our own observed value back to us at full trust.
    state.merge_peer_segment_latency(&node, before, 1.0);
    let after = state.observed_latency_ms_per_layer(&node).unwrap();
    assert!(
        (after - before).abs() < 1e-5,
        "matching foreign sample must not move the EMA (before={before}, after={after})"
    );
}

#[test]
fn merge_peer_segment_latency_ignores_bad_samples() {
    // Guards: non-finite samples, non-positive samples.
    let state = make_shared_state();
    let node = NodeId([14u8; 32]);

    state.merge_peer_segment_latency(&node, f32::NAN, 1.0);
    state.merge_peer_segment_latency(&node, f32::INFINITY, 1.0);
    state.merge_peer_segment_latency(&node, 0.0, 1.0);
    state.merge_peer_segment_latency(&node, -5.0, 1.0);
    assert!(state.observed_latency_ms_per_layer(&node).is_none());
}
