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

/// Model id for peer-speed tests. The speed map is keyed by node; the model
/// only marks the (peer, model) pair warm, which these tests don't assert on.
/// Mark a (peer, model) pair as already having the model resident.
///
/// The first segment to a pair is COLD — it paid to load the model — and no
/// longer feeds the ranking figure, so tests about the EMA arithmetic have to
/// say the peer is warm. Done directly rather than with a warm-up sample, which
/// would perturb the very numbers these tests assert.
fn mark_model_warm(state: &std::sync::Arc<crate::daemon::SharedState>, node: &NodeId) {
    state.metrics.peer_model_warm_at.insert(
        (node.clone(), speed_test_model()),
        std::time::Instant::now(),
    );
}

fn speed_test_model() -> ModelId {
    ModelId("peer-speed-test-model".into())
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
            ack_srtt_ms: None,
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
            ack_srtt_ms: None,
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

/// Routing prices a peer by what forwarding to it has actually cost when it
/// has that figure, and by the health ping only until then. The ping is
/// taken idle; the acknowledgement latency is taken on real work and sees a
/// loaded peer's queueing (gotcha #386). Two holders of the same shard: the
/// one whose ping looks worse but whose forwards come back faster wins —
/// and without the measured figure the ping decides, as before.
#[test]
fn a_measured_ack_latency_outranks_the_health_ping_when_choosing_a_holder() {
    fn holder(latency_ms: u32, ack_srtt_ms: Option<u32>, id: u8) -> (NodeId, PeerInfo) {
        let node = NodeId([id; 32]);
        (
            node.clone(),
            PeerInfo {
                node_id: node,
                addresses: vec![],
                capability: None,
                last_seen: chrono::Utc::now(),
                latency_ms: Some(latency_ms),
                trust_score: 0.8,
                peer_id_bytes: None,
                ack_srtt_ms,
                active_request_count: 0,
                first_seen: 0,
                verified_transaction_count: 0,
                is_lan_peer: false,
            },
        )
    }
    fn chosen(slow_ping_ack: Option<u32>) -> NodeId {
        let state = make_shared_state();
        let local_id = state.identity.node_id().clone();
        let shard = ShardId {
            model_id: ModelId("test-model".into()),
            index: 0,
        };
        let manifest = make_manifest(
            "test-model",
            32,
            vec![ShardInfo {
                index: 0,
                layer_range: (0, 32),
                size_bytes: 2_000_000_000,
                hash: [0u8; 32],
                tensors: vec![],
            }],
        );
        state.model_registry.register_manifest(manifest);
        // Slow ping (400 ms) but, when given one, a fast measured ACK (20 ms).
        let (slow_ping, slow_info) = holder(400, slow_ping_ack, 2);
        // Fast ping (50 ms), never forwarded to, so no measured figure.
        let (fast_ping, fast_info) = holder(50, None, 3);
        for (node, info) in [
            (slow_ping.clone(), slow_info),
            (fast_ping.clone(), fast_info),
        ] {
            state
                .model_registry
                .record_shard_holder(shard.clone(), node.clone());
            state.peer_registry.insert(node.clone(), info);
            state.connected_node_ids.insert(node);
        }
        let scheduler = PipelineScheduler::new(state);
        let assignment = scheduler
            .assemble_pipeline(&ModelId("test-model".into()), &local_id)
            .unwrap();
        assert_eq!(assignment.segments.len(), 1);
        assignment.segments[0].node_id.clone()
    }
    assert_eq!(
        chosen(Some(20)),
        NodeId([2u8; 32]),
        "the measured 20 ms must beat a 50 ms ping"
    );
    // Control: with no measured figure the ping decides, and the 50 ms peer wins.
    assert_eq!(chosen(None), NodeId([3u8; 32]));
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

/// Minimal candidate for the greedy tests — everything neutral so the field
/// under test is the only thing that varies.
fn simple_candidate(byte: u8, ranges: Vec<(u32, u32)>) -> NodeCandidate {
    NodeCandidate {
        node_id: NodeId([byte; 32]),
        shard_id: ShardId {
            model_id: ModelId("test".into()),
            index: 0,
        },
        available_ranges: ranges,
        reach: super::ReachTier::DirectMeasured,
        latency_ms: 10,
        load: 0.0,
        trust_score: 1.0,
        can_be_first: true,
        can_be_last: true,
        region_score: 1.0,
        est_tokens_per_sec: 0.0,
        observed_latency_ms_per_layer: None,
        observed_delegated_ms_per_layer: None,
        expected_attempts: 1.0,
        is_pool_member: false,
        gpu_vram_available_mb: None,
        max_hostable_layers: None,
        observed_prefill_ms_per_layer_byte: None,
        has_gpu: false,
    }
}

/// **The greedy fallback used to hand a node every layer it HELD, which is a
/// different question from how many it can fit in memory at once.** Measured on
/// the live swarm 2026-08-31: a 6 GB card was assigned all 48 layers of an
/// 8,571 MB model as one segment and failed at segment 0, about one request in
/// four. `parallax_assign` has always honoured the cap; the fallback beneath it
/// never did — and the fallback runs precisely when the model is awkward enough
/// for parallax to give up, so the guard was missing exactly where it mattered.
#[test]
fn greedy_never_hands_a_node_more_layers_than_it_can_hold() {
    let state = make_shared_state();
    let scheduler = PipelineScheduler::new(state);

    // One node declares every layer of a 48-layer model but can hold 12.
    let mut big = simple_candidate(1, vec![(0, 48)]);
    big.max_hostable_layers = Some(12);
    // A second node can take the rest, so a correct assignment exists.
    let mut rest = simple_candidate(2, vec![(0, 48)]);
    rest.max_hostable_layers = Some(48);

    let segments = scheduler
        .greedy_assign(48, &[big, rest], false)
        .expect("a valid assignment exists");

    let first = &segments[0];
    assert!(
        first.layer_range.1 - first.layer_range.0 <= 12,
        "the capped node was handed {} layers, cap is 12",
        first.layer_range.1 - first.layer_range.0
    );
    assert_eq!(
        segments.last().unwrap().layer_range.1,
        48,
        "the whole model must still be covered"
    );
}

/// **The cap is a preference, not a gate.** If honouring every node's memory
/// bound leaves layers uncovered, assigning anyway beats refusing the request —
/// which is the same judgement `parallax_assign` makes when it logs "no route
/// fits the peers' advertised memory". A swarm exists precisely for models its
/// members cannot comfortably hold; declining to try would make the capacity
/// check worse than not having one.
#[test]
fn a_model_that_fits_nobody_is_still_assigned_rather_than_refused() {
    let state = make_shared_state();
    let scheduler = PipelineScheduler::new(state);
    // One holder, 48 layers, and it admits to holding only 8 at a time.
    let mut only = simple_candidate(1, vec![(0, 48)]);
    only.max_hostable_layers = Some(8);

    let segments = scheduler
        .greedy_assign(48, &[only], false)
        .expect("must fall back to an unbounded route rather than refuse");
    assert_eq!(
        segments.last().unwrap().layer_range.1,
        48,
        "the whole model must be covered by the fallback"
    );
}

/// `None` means UNKNOWN and must never exclude — an unreadable capability is
/// not evidence that a node is small, which is the distinction
/// [`max_hostable_layers`] exists to preserve.
#[test]
fn an_unknown_capacity_still_takes_the_whole_range() {
    let state = make_shared_state();
    let scheduler = PipelineScheduler::new(state);
    let mut only = simple_candidate(1, vec![(0, 48)]);
    only.max_hostable_layers = None;
    let segments = scheduler.greedy_assign(48, &[only], false).unwrap();
    assert_eq!(segments.len(), 1, "unknown capacity must not fragment");
    assert_eq!(segments[0].layer_range, (0, 48));
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
            reach: super::ReachTier::Local,
            latency_ms: 0,
            load: 0.0,
            trust_score: 1.0,
            can_be_first: true,
            can_be_last: true,
            region_score: 1.0,
            est_tokens_per_sec: 0.0,
            observed_latency_ms_per_layer: None,
            observed_delegated_ms_per_layer: None,
            expected_attempts: 1.0,
            is_pool_member: false,
            gpu_vram_available_mb: None,
            max_hostable_layers: None,
            observed_prefill_ms_per_layer_byte: None,
            has_gpu: false,
        },
        NodeCandidate {
            node_id: NodeId([2u8; 32]),
            shard_id: ShardId {
                model_id: ModelId("test".into()),
                index: 1,
            },
            available_ranges: vec![(2, 10)],
            reach: super::ReachTier::DirectMeasured,
            latency_ms: 10,
            load: 0.0,
            trust_score: 0.8,
            can_be_first: false,
            can_be_last: false,
            region_score: 0.7,
            est_tokens_per_sec: 0.0,
            observed_latency_ms_per_layer: None,
            observed_delegated_ms_per_layer: None,
            expected_attempts: 1.0,
            is_pool_member: false,
            gpu_vram_available_mb: None,
            max_hostable_layers: None,
            observed_prefill_ms_per_layer_byte: None,
            has_gpu: false,
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
            ack_srtt_ms: None,
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
            ack_srtt_ms: None,
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
/// `detect_tp_groups` only forms a group for a segment assigned to US, so these
/// tests silently depend on the local node winning layers 0-16.
///
/// Asserted explicitly because when that premise broke on CI the failure read
/// `left: 0, right: 1` — true, and useless. It said nothing about the local node
/// having been outbid, which is what had actually happened.
fn assert_local_holds_the_first_segment(
    assignment: &crate::inference::scheduler::PipelineAssignment,
    local_id: &NodeId,
) {
    let first = assignment
        .segments
        .first()
        .expect("pipeline assembled no segments at all");
    assert_eq!(
        &first.node_id, local_id,
        "layers {:?} went to {} rather than the local node, so no tensor-parallel \
         group can form — the peer fixture is no longer slow enough to lose, or \
         local pricing has moved (gotcha #429)",
        first.layer_range, first.node_id
    );
}

/// A peer slow enough that **no machine can lose to it**, so local-versus-peer
/// selection in these tests is deterministic.
///
/// The local node is priced from `mem_bandwidth`, i.e. from whatever hardware
/// the test runs on, and the comparison is
/// `local_ms_per_layer = 1000 / (bw/4.4*0.75) / 32 = 187.5 / bw`. So a peer at
/// `t` tok/s costs `1000/t/32` ms per layer and the local node only loses below
/// `bw = 187.5 * t * 32 / 1000` GB/s.
///
/// **0.5 tok/s put that break-even at 3.0 GB/s and CI fell under it** — a shared
/// runner, under build load, on an unoptimised build that measures its own loop
/// rather than the memory (gotcha #427). It passed on both machines here and
/// failed on GitHub, which is the whole failure mode gotcha #429 describes: a
/// test that prices the local node is testing the machine it runs on, and
/// picking a *closer* threshold only moves where it breaks.
///
/// 0.01 tok/s puts the break-even at **0.06 GB/s**, and that is not merely a
/// distant threshold — it is unreachable. `measured_gbps` refuses to report
/// anything below **1.0 GB/s**, returning `None` for an implausible reading, and
/// `None` prices the local node at the `UNKNOWN_COMPUTE_MS` prior of 25 ms per
/// layer. So both arms are covered: a measurement that lands gives at worst
/// 187.5 ms per layer, and one that does not gives 25, against the peer's 3125.
/// The local node cannot lose, on any machine, in any build profile.
///
/// These tests are about tensor-parallel GROUPING; the pricing contest is
/// scenery, and it should be scenery that cannot fall over.
fn slow_peer_capability(node: &NodeId) -> crate::types::NodeCapability {
    crate::types::NodeCapability {
        node_id: node.clone(),
        cpu: None,
        gpu: None,
        ram_total_mb: 16_384,
        ram_available_mb: 16_384,
        ram_model_budget_mb: None,
        disk_available_mb: 100_000,
        bandwidth_mbps: 100.0,
        hosted_shards: vec![],
        max_contribution: crate::types::ContributionLevel::Moderate,
        uptime_seconds: 3600,
        version: "0.1.0".into(),
        region: None,
        est_tokens_per_sec_7b: 0.01,
        os: None,
        observed_latencies: vec![],
        relay_capable: false,
        protocol_version: 0,
        features: 0,
        relay_reservations: vec![],
        anchor_mode: false,
    }
}

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
                // Pinned rather than left `None`, so which node wins layers
                // 0-16 does not depend on the host's memory bandwidth.
                //
                // The local node is now priced from `mem_bandwidth`, which is a
                // real measurement of whatever machine the test runs on — and
                // an unoptimised build measures its own loop rather than the
                // memory (gotcha #427), so it reads ~5 GB/s here against ~30 in
                // release. A peer left at `None` falls back to
                // `UNKNOWN_COMPUTE_MS`, and against that prior the local node
                // won in a release build and lost in a debug one. These tests
                // are about TENSOR-PARALLEL GROUPING, not about who wins a
                // pricing contest, so the peer is given a speed slow enough
                // that the local node takes the segment on any machine.
                capability: Some(slow_peer_capability(&node)),
                last_seen: chrono::Utc::now(),
                latency_ms: Some(latency),
                trust_score: 0.9,
                peer_id_bytes: None,
                ack_srtt_ms: None,
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
            ack_srtt_ms: None,
            active_request_count: 0,
            first_seen: 0,
            verified_transaction_count: 0,
            is_lan_peer: true,
        },
    );
    state.connected_node_ids.insert(node_b);

    // Pinned, not `new`: this test is about tensor-parallel GROUPING, and
    // `node_b` above carries `capability: None` — an UNPRICED peer. Since
    // gotcha #479 an unpriced peer competes on the `UNKNOWN_COMPUTE_MS` prior
    // instead of being vetoed outright, and that prior beats a local node
    // below ~1.25 tok/s. A debug build measures its own loop rather than the
    // memory (gotcha #427) and reads ~0.85 tok/s here against ~5 in release,
    // so without the pin this test would assert a routing verdict that
    // depends on the build profile.
    let scheduler = PipelineScheduler::with_local_processor_speed(state, LOCAL_PROCESSOR_TPS);
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

    assert_local_holds_the_first_segment(&assignment, &local_id);
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

    assert_local_holds_the_first_segment(&assignment, &local_id);
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
            ack_srtt_ms: None,
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
            ack_srtt_ms: None,
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
            ack_srtt_ms: None,
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
            ack_srtt_ms: None,
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
    mark_model_warm(&state, &node);
    // First sample: 20 ms over 4 layers → 5 ms/layer. No prior → EMA = 5.
    state.record_peer_segment_latency(
        &node,
        &speed_test_model(),
        crate::daemon::state::WorkKind::Decode,
        20,
        4,
        0,
    );
    let v1 = state.observed_latency_ms_per_layer(&node).unwrap();
    assert!((v1 - 5.0).abs() < 1e-5, "first sample EMA = {v1}");
    // Second sample: 40 ms over 4 layers → 10 ms/layer. EMA = 0.3*10 + 0.7*5 = 6.5.
    state.record_peer_segment_latency(
        &node,
        &speed_test_model(),
        crate::daemon::state::WorkKind::Decode,
        40,
        4,
        0,
    );
    let v2 = state.observed_latency_ms_per_layer(&node).unwrap();
    assert!((v2 - 6.5).abs() < 1e-5, "second sample EMA = {v2}");
    // Width-normalised: a 2-layer segment at 20 ms → 10 ms/layer.
    state.record_peer_segment_latency(
        &node,
        &speed_test_model(),
        crate::daemon::state::WorkKind::Decode,
        20,
        2,
        0,
    );
    let v3 = state.observed_latency_ms_per_layer(&node).unwrap();
    // EMA = 0.3*10 + 0.7*6.5 = 7.55.
    assert!((v3 - 7.55).abs() < 1e-5, "third sample EMA = {v3}");
    // Zero-layer guard: no panic, no update.
    state.record_peer_segment_latency(
        &node,
        &speed_test_model(),
        crate::daemon::state::WorkKind::Decode,
        100,
        0,
        0,
    );
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
    mark_model_warm(&state, &node);

    // No entry yet + weight 0 → no insert.
    state.merge_peer_segment_latency(&node, 100.0, 0.0);
    assert!(state.observed_latency_ms_per_layer(&node).is_none());

    // Seed a direct observation, then try to poison it with weight 0.
    state.record_peer_segment_latency(
        &node,
        &speed_test_model(),
        crate::daemon::state::WorkKind::Decode,
        20,
        4,
        0,
    );
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

    // Asserted on whether an ENTRY was seeded, not through
    // `observed_latency_ms_per_layer`: ranking deliberately no longer surfaces a
    // figure this node never measured, so a purely gossiped value reads as None
    // there however well-trusted the reporter. The seeding rule under test is
    // unchanged.
    state.merge_peer_segment_latency(&stranger, 500.0, 0.1);
    assert!(!state.metrics.peer_speed.contains_key(&stranger));

    state.merge_peer_segment_latency(&stranger, 500.0, 0.29);
    assert!(!state.metrics.peer_speed.contains_key(&stranger));

    // Exactly at the threshold seeds.
    state.merge_peer_segment_latency(&stranger, 500.0, 0.3);
    assert!(
        state.metrics.peer_speed.contains_key(&stranger),
        "at the threshold the entry must be created"
    );
}

#[test]
fn merge_peer_segment_latency_trust_weighted_ema() {
    // Weight-scaled α: effective_α = 0.3 * weight. Direct samples move
    // the EMA more than foreign samples from less-trusted peers.
    let state = make_shared_state();
    let node = NodeId([13u8; 32]);
    mark_model_warm(&state, &node);

    // Start with a direct sample of 10 ms/layer (20 ms over 2 layers).
    state.record_peer_segment_latency(
        &node,
        &speed_test_model(),
        crate::daemon::state::WorkKind::Decode,
        20,
        2,
        0,
    );
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
    mark_model_warm(&state2, &node);
    state2.record_peer_segment_latency(
        &node,
        &speed_test_model(),
        crate::daemon::state::WorkKind::Decode,
        20,
        2,
        0,
    );
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
        state.record_peer_segment_latency(
            &node,
            &speed_test_model(),
            crate::daemon::state::WorkKind::Decode,
            50,
            5,
            0,
        );
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

// ---------------------------------------------------------------------------
// Reachability tier + cost-based segment selection.
//
// Both pin behaviour that was found wrong on 2026-08-01 against a live swarm.
// ---------------------------------------------------------------------------

fn cost_cand(
    byte: u8,
    ranges: Vec<(u32, u32)>,
    reach: super::ReachTier,
    latency_ms: u32,
    load: f32,
    ms_per_layer: Option<f32>,
) -> NodeCandidate {
    NodeCandidate {
        node_id: NodeId([byte; 32]),
        shard_id: ShardId {
            model_id: ModelId("m".into()),
            index: 0,
        },
        available_ranges: ranges,
        reach,
        latency_ms,
        load,
        trust_score: 1.0,
        can_be_first: true,
        can_be_last: true,
        region_score: 1.0,
        est_tokens_per_sec: 0.0,
        observed_latency_ms_per_layer: ms_per_layer,
        observed_delegated_ms_per_layer: None,
        expected_attempts: 1.0,
        is_pool_member: false,
        gpu_vram_available_mb: None,
        max_hostable_layers: None,
        observed_prefill_ms_per_layer_byte: None,
        has_gpu: false,
    }
}

/// The documented guarantee — "a directly-connected holder always outranks a
/// relayed one" — used to be an additive 150ms penalty, which a 570ms direct
/// peer defeated. As an ordering key it holds for any latency at all.
#[test]
fn a_direct_holder_outranks_a_relayed_one_at_any_latency() {
    assert!(super::ReachTier::DirectMeasured < super::ReachTier::RelayedMeasured);
    assert!(super::ReachTier::DirectUnmeasured < super::ReachTier::RelayedMeasured);
    assert!(super::ReachTier::Local < super::ReachTier::DirectMeasured);

    // The live shape: a relay-only peer we never timed vs a measured direct
    // peer at 570 ms. Under the old additive penalty the relayed peer scored
    // 100 + 150 = 250 and won.
    let mut v = [
        cost_cand(
            1,
            vec![(0, 8)],
            super::ReachTier::RelayedUnmeasured,
            450,
            0.0,
            None,
        ),
        cost_cand(
            2,
            vec![(0, 8)],
            super::ReachTier::DirectMeasured,
            570,
            0.0,
            None,
        ),
    ];
    v.sort_by(|a, b| {
        a.reach
            .cmp(&b.reach)
            .then_with(|| a.latency_ms.cmp(&b.latency_ms))
    });
    assert_eq!(
        v[0].node_id,
        NodeId([2u8; 32]),
        "the measured direct peer must rank first despite being slower"
    );
}

/// An unmeasured peer must not be treated as better than a measured mediocre
/// one. The old `unwrap_or(100)` made "we know nothing" score better than most
/// real peers.
#[test]
fn an_unmeasured_peer_ranks_behind_a_measured_one_in_its_tier() {
    assert!(super::ReachTier::DirectMeasured < super::ReachTier::DirectUnmeasured);
    assert!(super::ReachTier::RelayedMeasured < super::ReachTier::RelayedUnmeasured);
    // The old default was a flat 100ms, which flattered an unknown peer over
    // most measured ones. Compare against a mid-range real peer rather than
    // the literal, so this reads as the property it is defending.
    let typical_measured_peer_ms: u32 = 100;
    assert!(
        super::UNMEASURED_PEER_LATENCY_MS > typical_measured_peer_ms,
        "an unknown peer must not be assumed faster than a typical real one"
    );
}

/// The reported defect: `load` is a whole-request integer compared BEFORE
/// latency, so one in-flight request on a 4ms LAN peer diverted the segment to
/// a peer 100x further away.
#[test]
fn one_busy_request_does_not_divert_a_segment_to_a_far_peer() {
    let lan_busy = cost_cand(
        1,
        vec![(0, 8)],
        super::ReachTier::DirectMeasured,
        4,
        1.0, // one request in flight
        Some(20.0),
    );
    let far_idle = cost_cand(
        2,
        vec![(0, 8)],
        super::ReachTier::DirectMeasured,
        411,
        0.0,
        Some(20.0),
    );

    let lan = PipelineScheduler::estimated_cost_per_layer(&lan_busy, (0, 8), 0);
    let far = PipelineScheduler::estimated_cost_per_layer(&far_idle, (0, 8), 0);
    assert!(
        lan < far,
        "a busy 4ms peer should still beat an idle 411ms one (lan={lan}, far={far})"
    );
}

/// Load must still matter — it is a real penalty, just not one that outranks
/// everything. A heavily loaded peer loses to an equally-distant idle one.
#[test]
fn load_still_penalises_a_peer_when_distance_is_equal() {
    let busy = cost_cand(
        1,
        vec![(0, 8)],
        super::ReachTier::DirectMeasured,
        10,
        4.0,
        Some(20.0),
    );
    let idle = cost_cand(
        2,
        vec![(0, 8)],
        super::ReachTier::DirectMeasured,
        10,
        0.0,
        Some(20.0),
    );
    assert!(
        PipelineScheduler::estimated_cost_per_layer(&idle, (0, 8), 0)
            < PipelineScheduler::estimated_cost_per_layer(&busy, (0, 8), 0),
        "at equal distance the idle peer must win"
    );
}

/// A wide segment amortises its round trip; a narrow one cannot. This is what
/// lets coverage and latency be priced against each other instead of ranked.
#[test]
fn wider_coverage_amortises_a_distant_peers_round_trip() {
    let far_wide = cost_cand(
        1,
        vec![(0, 32)],
        super::ReachTier::DirectMeasured,
        400,
        0.0,
        Some(5.0),
    );
    let far_narrow = cost_cand(
        2,
        vec![(0, 2)],
        super::ReachTier::DirectMeasured,
        400,
        0.0,
        Some(5.0),
    );
    let wide = PipelineScheduler::estimated_cost_per_layer(&far_wide, (0, 32), 0);
    let narrow = PipelineScheduler::estimated_cost_per_layer(&far_narrow, (0, 2), 0);
    assert!(
        wide < narrow,
        "32 layers should amortise the RTT better than 2 (wide={wide}, narrow={narrow})"
    );
}

/// A genuinely faster peer wins when everything else is equal — the measured
/// per-layer cost has to actually count for something.
#[test]
fn a_faster_peer_wins_at_equal_distance_and_load() {
    let fast = cost_cand(
        1,
        vec![(0, 8)],
        super::ReachTier::DirectMeasured,
        10,
        0.0,
        Some(2.0),
    );
    let slow = cost_cand(
        2,
        vec![(0, 8)],
        super::ReachTier::DirectMeasured,
        10,
        0.0,
        Some(200.0),
    );
    assert!(
        PipelineScheduler::estimated_cost_per_layer(&fast, (0, 8), 0)
            < PipelineScheduler::estimated_cost_per_layer(&slow, (0, 8), 0)
    );
}

// ---------------------------------------------------------------------------
// Whole-model delegation: handing a model to a peer instead of falling back to
// the local CPU. See `delegation_target` for why this is a two-way choice
// rather than a price fed into the routing search.
// ---------------------------------------------------------------------------

/// A peer that could plausibly take a delegated model: nearby, measured,
/// trusted, holds everything, and advertises a roomy GPU.
fn willing_peer(byte: u8, layers: u32) -> NodeCandidate {
    let mut c = cost_cand(
        byte,
        vec![(0, layers)],
        super::ReachTier::DirectMeasured,
        5,
        0.0,
        None,
    );
    c.gpu_vram_available_mb = Some(24_000);
    c
}

const LAYERS: u32 = 28;
/// A model needing 4 GB, against peers advertising 24 GB free.
const MODEL_MB: u64 = 4_000;

fn local_id() -> NodeId {
    NodeId([0xAA; 32])
}

fn local_full_coverage() -> NodeCandidate {
    let mut c = cost_cand(
        0xAA,
        vec![(0, LAYERS)],
        super::ReachTier::Local,
        0,
        0.0,
        None,
    );
    c.node_id = local_id();
    c
}

#[test]
fn a_full_gpu_hands_the_model_to_a_nearby_peer() {
    let cands = vec![local_full_coverage(), willing_peer(0xBB, LAYERS)];
    let picked = super::delegation_target(
        &cands,
        &super::DelegationInput {
            local_node_id: &local_id(),
            num_layers: LAYERS,
            layers_to_assign: LAYERS,
            local_serves_on_cpu: true,
            model_vram_mb: MODEL_MB,
            local_cpu_tokens_per_sec: 0.0,
            prompt_tokens: None,
        },
    );
    assert_eq!(
        picked.map(|c| c.node_id.clone()),
        Some(NodeId([0xBB; 32])),
        "a model that does not fit our GPU should go to a peer whose GPU it fits"
    );
}

/// The condition that decides everything. A node that is not CPU-bound for
/// lack of VRAM — its GPU fits the model, or it has no usable GPU, or the user
/// asked for the CPU, or we simply could not tell — keeps the request. See
/// `ModelProcessPool::is_cpu_bound_for_lack_of_vram`.
#[test]
fn a_healthy_local_gpu_keeps_the_request_here() {
    let cands = vec![local_full_coverage(), willing_peer(0xBB, LAYERS)];
    assert!(
        super::delegation_target(
            &cands,
            &super::DelegationInput {
                local_node_id: &local_id(),
                num_layers: LAYERS,
                layers_to_assign: LAYERS,
                local_serves_on_cpu: false,
                model_vram_mb: MODEL_MB,
                local_cpu_tokens_per_sec: 0.0,
                prompt_tokens: None
            }
        )
        .is_none(),
        "a node that is not CPU-bound for lack of VRAM must not delegate"
    );
}

/// **The failure that caused the previous attempt to be reverted.** A machine
/// holding the model sent the whole request to another country while a peer
/// 5 ms away was available: five minutes, then failure. Distance is now a hard
/// bound, not a term in a score that something else can outweigh.
#[test]
fn a_distant_peer_is_never_handed_the_model() {
    let mut far = willing_peer(0xBB, LAYERS);
    far.latency_ms = super::DELEGATE_MAX_LATENCY_MS + 1;
    let cands = vec![local_full_coverage(), far];
    assert!(
        super::delegation_target(
            &cands,
            &super::DelegationInput {
                local_node_id: &local_id(),
                num_layers: LAYERS,
                layers_to_assign: LAYERS,
                local_serves_on_cpu: true,
                model_vram_mb: MODEL_MB,
                local_cpu_tokens_per_sec: 0.0,
                prompt_tokens: None
            }
        )
        .is_none(),
        "a peer beyond the latency bound must never be delegated to"
    );
}

/// The bound must admit the fleet this actually runs on. Until 2026-08-31 it
/// was 200 ms, chosen to mean "LAN or metro" — and on a swarm whose only GPU
/// peer sits at ~600 ms that made the whole feature unreachable, which is the
/// same way the original 50 ms value failed, one order of magnitude out.
///
/// Measured on that peer: a whole-model generation (the delegation shape) ran
/// at 21-25 tok/s where this node's own processor fallback does 9-10, so
/// delegating across ~600 ms was ~2.3x better than keeping the request. The
/// asserted latency here is that real peer's.
#[test]
fn a_peer_a_continent_away_is_still_worth_delegating_to() {
    let mut wan = willing_peer(0xBB, LAYERS);
    wan.latency_ms = 600;
    let cands = vec![local_full_coverage(), wan];
    assert_eq!(
        super::delegation_target(
            &cands,
            &super::DelegationInput {
                local_node_id: &local_id(),
                num_layers: LAYERS,
                layers_to_assign: LAYERS,
                local_serves_on_cpu: true,
                model_vram_mb: MODEL_MB,
                local_cpu_tokens_per_sec: 0.0,
                prompt_tokens: None
            }
        )
        .map(|c| c.node_id.clone()),
        Some(NodeId([0xBB; 32])),
        "a 600 ms GPU peer measured 2.3x faster than the local processor fallback \
         and must not be excluded by the distance bound"
    );
}

/// **The reported failure (2026-09-05).** A peer that looks fast on the decode
/// metric and is catastrophic on prefill must not be handed a long prompt.
///
/// `est_tokens_per_sec` is a memory-bandwidth estimate of how fast a machine
/// WRITES tokens. Prefill is compute-bound and its hardware spread is ~55x
/// against decode's ~6x, so on a long prompt the old gate compared machines on
/// the wrong axis. Live: an Apple M4 advertising 14.82 tok/s against a local
/// 6.46 cleared the processor branch twice and took 5-6 minutes to the first
/// token, while the routing search had priced it at ~234 minutes of prefill and
/// avoided it.
#[test]
fn a_peer_that_is_slow_at_reading_the_prompt_is_not_handed_a_long_one() {
    let mut slow_prefill = willing_peer(0xBB, LAYERS);
    slow_prefill.gpu_vram_available_mb = None; // processor-only, like the M4
    slow_prefill.est_tokens_per_sec = 14.82; // flattering decode figure
                                             // Measured: this peer is dreadful at reading a prompt.
    slow_prefill.observed_prefill_ms_per_layer_byte = Some(1.0);

    let mut local = local_full_coverage();
    local.est_tokens_per_sec = 6.46;
    local.observed_prefill_ms_per_layer_byte = Some(0.000_01);

    let cands = vec![local, slow_prefill];
    assert!(
        super::delegation_target(
            &cands,
            &super::DelegationInput {
                local_node_id: &local_id(),
                num_layers: LAYERS,
                layers_to_assign: LAYERS,
                local_serves_on_cpu: true,
                model_vram_mb: MODEL_MB,
                local_cpu_tokens_per_sec: 6.46,
                prompt_tokens: Some(12_000)
            }
        )
        .is_none(),
        "a peer priced slower than us for this prompt must not be delegated to, \
         however good its decode figure looks"
    );
}

/// The control. The same peer, the same flattering decode figure — but a SHORT
/// prompt, where prefill is not what dominates. It must still be delegated to,
/// or the gate has simply been turned off.
#[test]
fn a_short_prompt_still_goes_to_the_faster_processor() {
    let mut peer = willing_peer(0xBB, LAYERS);
    peer.gpu_vram_available_mb = None;
    peer.est_tokens_per_sec = 14.82;
    peer.observed_prefill_ms_per_layer_byte = Some(0.000_01);

    let mut local = local_full_coverage();
    local.est_tokens_per_sec = 6.46;
    local.observed_prefill_ms_per_layer_byte = Some(0.000_01);

    let cands = vec![local, peer];
    assert_eq!(
        super::delegation_target(
            &cands,
            &super::DelegationInput {
                local_node_id: &local_id(),
                num_layers: LAYERS,
                layers_to_assign: LAYERS,
                local_serves_on_cpu: true,
                model_vram_mb: MODEL_MB,
                local_cpu_tokens_per_sec: 6.46,
                prompt_tokens: Some(16)
            }
        )
        .map(|c| c.node_id.clone()),
        Some(NodeId([0xBB; 32])),
        "a genuinely faster machine must still get a short request"
    );
}

/// A peer nobody has measured is not evidence of anything. Refusing on missing
/// information would strand this node on its processor beside a machine that
/// may well be faster — the failure `delegation_target` exists to fix.
#[test]
fn an_unmeasured_peer_is_still_tried() {
    let mut peer = willing_peer(0xBB, LAYERS);
    peer.observed_prefill_ms_per_layer_byte = None;
    let cands = vec![local_full_coverage(), peer];
    assert_eq!(
        super::delegation_target(
            &cands,
            &super::DelegationInput {
                local_node_id: &local_id(),
                num_layers: LAYERS,
                layers_to_assign: LAYERS,
                local_serves_on_cpu: true,
                model_vram_mb: MODEL_MB,
                local_cpu_tokens_per_sec: 0.0,
                prompt_tokens: Some(12_000)
            }
        )
        .map(|c| c.node_id.clone()),
        Some(NodeId([0xBB; 32])),
        "unknown must not exclude — the first request is what produces the \
         measurement that would stop the second"
    );
}

/// Privacy's cost is REPORTED and never acted on — but the figure has to be
/// right, or the notice is noise. Keeping the two end layers on a machine that
/// is slow at reading a prompt must price as a real addition.
#[test]
fn keeping_the_ends_local_prices_as_a_real_cost_on_a_long_prompt() {
    let mut peer = willing_peer(0xBB, LAYERS);
    peer.observed_prefill_ms_per_layer_byte = Some(0.000_001); // a fast card

    let mut local = local_full_coverage();
    // A processor that is dreadful at reading a prompt — the reported shape.
    local.observed_prefill_ms_per_layer_byte = Some(0.01);

    let cands = vec![local, peer.clone()];
    let extra = super::privacy_cost_ms(&peer, &cands, &local_id(), LAYERS, Some(10_000))
        .expect("both sides are priceable");
    assert!(
        extra > 0.0,
        "holding the first and last layers on a slow processor must cost \
         something against handing the whole model to the card, not nothing"
    );
}

/// The control: when the local machine reads a prompt as fast as the peer,
/// privacy is nearly free and must not produce a notice.
#[test]
fn privacy_is_not_reported_as_expensive_when_it_is_cheap() {
    let mut peer = willing_peer(0xBB, LAYERS);
    peer.observed_prefill_ms_per_layer_byte = Some(0.000_001);

    let mut local = local_full_coverage();
    local.observed_prefill_ms_per_layer_byte = Some(0.000_001);

    let cands = vec![local, peer.clone()];
    let extra = super::privacy_cost_ms(&peer, &cands, &local_id(), LAYERS, Some(10_000))
        .expect("priceable");
    assert!(
        extra < super::PRIVACY_COST_REPORT_MS,
        "an equally fast local machine must not trigger a notice — got {extra}"
    );
}

/// A short prompt cannot be priced into a privacy warning, and a model too
/// small to have a middle has no boomerang to price at all.
#[test]
fn there_is_nothing_to_report_without_a_prompt_or_a_middle() {
    let peer = willing_peer(0xBB, LAYERS);
    let cands = vec![local_full_coverage(), peer.clone()];
    assert!(super::privacy_cost_ms(&peer, &cands, &local_id(), LAYERS, None).is_none());
    assert!(super::privacy_cost_ms(&peer, &cands, &local_id(), LAYERS, Some(0)).is_none());
    assert!(super::privacy_cost_ms(&peer, &cands, &local_id(), 2, Some(10_000)).is_none());
}

/// Widening the bound must not change WHICH peer wins when several qualify.
/// `candidates` arrives sorted nearest-first, so a distant peer can only ever
/// be a fallback — that ordering is what makes raising the ceiling safe, and
/// it is the property the 2026-08-03 revert (`cbbed678`) existed to protect.
#[test]
fn a_nearer_peer_still_wins_over_a_distant_one() {
    // NOT 0xAA — that is `local_id()`, and the local node is never its own
    // delegate, which is what the first cut of this test tripped over.
    let mut near = willing_peer(0xCC, LAYERS);
    near.latency_ms = 5;
    let mut far = willing_peer(0xBB, LAYERS);
    far.latency_ms = 900;
    let cands = vec![local_full_coverage(), near, far];
    assert_eq!(
        super::delegation_target(
            &cands,
            &super::DelegationInput {
                local_node_id: &local_id(),
                num_layers: LAYERS,
                layers_to_assign: LAYERS,
                local_serves_on_cpu: true,
                model_vram_mb: MODEL_MB,
                local_cpu_tokens_per_sec: 0.0,
                prompt_tokens: None
            }
        )
        .map(|c| c.node_id.clone()),
        Some(NodeId([0xCC; 32])),
        "the nearest qualifying peer must still win; a wider bound only adds fallbacks"
    );
}

/// A relayed peer is reachable but not suitable: relaying a whole generation is
/// not what that path is sized for, and an unmeasured latency is not a bound.
#[test]
fn only_a_directly_measured_peer_qualifies() {
    for tier in [
        super::ReachTier::DirectUnmeasured,
        super::ReachTier::RelayedMeasured,
        super::ReachTier::RelayedUnmeasured,
    ] {
        let mut p = willing_peer(0xBB, LAYERS);
        p.reach = tier;
        let cands = vec![local_full_coverage(), p];
        assert!(
            super::delegation_target(
                &cands,
                &super::DelegationInput {
                    local_node_id: &local_id(),
                    num_layers: LAYERS,
                    layers_to_assign: LAYERS,
                    local_serves_on_cpu: true,
                    model_vram_mb: MODEL_MB,
                    local_cpu_tokens_per_sec: 0.0,
                    prompt_tokens: None
                }
            )
            .is_none(),
            "{tier:?} must not be delegated to"
        );
    }
}

/// A node with no card at all runs the model on its processor, and a peer
/// that would run it several times faster is worth handing it to — with the
/// same wide margin the speed branch has always demanded, so a nose ahead is
/// not enough (gotcha #442: a processor-only node holding every shard ran
/// the model itself with GPU peers idle beside it).
#[test]
fn a_node_with_no_card_hands_the_model_to_a_much_faster_processor_peer() {
    let mut faster = willing_peer(0xBB, LAYERS);
    faster.gpu_vram_available_mb = None; // a processor peer, no card advertised
    faster.est_tokens_per_sec = 12.0;
    let cands = vec![local_full_coverage(), faster];
    assert_eq!(
        super::delegation_target(
            &cands,
            &super::DelegationInput {
                local_node_id: &local_id(),
                num_layers: LAYERS,
                layers_to_assign: LAYERS,
                local_serves_on_cpu: true,
                model_vram_mb: MODEL_MB,
                local_cpu_tokens_per_sec: 4.0,
                prompt_tokens: None
            }
        )
        .map(|c| c.node_id.clone()),
        Some(NodeId([0xBB; 32])),
        "three times our processor speed is worth the hand-off"
    );
    let mut barely = willing_peer(0xBB, LAYERS);
    barely.gpu_vram_available_mb = None;
    barely.est_tokens_per_sec = 6.0;
    let cands = vec![local_full_coverage(), barely];
    assert!(
        super::delegation_target(
            &cands,
            &super::DelegationInput {
                local_node_id: &local_id(),
                num_layers: LAYERS,
                layers_to_assign: LAYERS,
                local_serves_on_cpu: true,
                model_vram_mb: MODEL_MB,
                local_cpu_tokens_per_sec: 4.0,
                prompt_tokens: None
            }
        )
        .is_none(),
        "1.5x is inside the margin a self-reported figure gets"
    );
    // And a node that would NOT run it on its processor never delegates.
    let mut faster = willing_peer(0xBB, LAYERS);
    faster.est_tokens_per_sec = 12.0;
    let cands = vec![local_full_coverage(), faster];
    assert!(super::delegation_target(
        &cands,
        &super::DelegationInput {
            local_node_id: &local_id(),
            num_layers: LAYERS,
            layers_to_assign: LAYERS,
            local_serves_on_cpu: false,
            model_vram_mb: MODEL_MB,
            local_cpu_tokens_per_sec: 4.0,
            prompt_tokens: None
        }
    )
    .is_none());
}

/// Delegation is whole-model or nothing. A peer holding part of the model is a
/// candidate for the routing search, not for this — a split pays a round trip
/// per token and measured slower every time it was tried.
#[test]
fn a_peer_holding_only_some_layers_is_not_a_delegate() {
    let cands = vec![
        local_full_coverage(),
        willing_peer(0xBB, LAYERS - 1), // one layer short
    ];
    assert!(super::delegation_target(
        &cands,
        &super::DelegationInput {
            local_node_id: &local_id(),
            num_layers: LAYERS,
            layers_to_assign: LAYERS,
            local_serves_on_cpu: true,
            model_vram_mb: MODEL_MB,
            local_cpu_tokens_per_sec: 0.0,
            prompt_tokens: None
        }
    )
    .is_none());
}

/// The peer's free VRAM is self-reported, so it is used only as a yes/no gate —
/// and with a margin, because it is a moment out of date and our size estimate
/// is for OUR placement. A peer with no GPU, or a tight one, keeps the request
/// local where the outcome is merely slow.
#[test]
fn a_peer_without_room_to_spare_is_not_a_delegate() {
    for free in [None, Some(0), Some(MODEL_MB), Some(MODEL_MB + 1)] {
        let mut p = willing_peer(0xBB, LAYERS);
        p.gpu_vram_available_mb = free;
        let cands = vec![local_full_coverage(), p];
        assert!(
            super::delegation_target(
                &cands,
                &super::DelegationInput {
                    local_node_id: &local_id(),
                    num_layers: LAYERS,
                    layers_to_assign: LAYERS,
                    local_serves_on_cpu: true,
                    model_vram_mb: MODEL_MB,
                    local_cpu_tokens_per_sec: 0.0,
                    prompt_tokens: None
                }
            )
            .is_none(),
            "free={free:?} is not enough room for a {MODEL_MB} MB model"
        );
    }
    // Comfortably above the margin, so it qualifies — otherwise the assertions
    // above would pass for the wrong reason.
    let mut ok = willing_peer(0xBB, LAYERS);
    ok.gpu_vram_available_mb = Some((MODEL_MB as f64 * super::DELEGATE_VRAM_MARGIN) as u64 + 1);
    let cands = vec![local_full_coverage(), ok];
    assert!(super::delegation_target(
        &cands,
        &super::DelegationInput {
            local_node_id: &local_id(),
            num_layers: LAYERS,
            layers_to_assign: LAYERS,
            local_serves_on_cpu: true,
            model_vram_mb: MODEL_MB,
            local_cpu_tokens_per_sec: 0.0,
            prompt_tokens: None
        }
    )
    .is_some());
}

/// The boomerang gives its peer every layer but two, and until 2026-09-04 the
/// only memory check on that path was `boomerang_assignment`'s `covers()` —
/// which asks whether the peer HOLDS those layers, never whether it can run
/// them. Reported live on v0.3.153: a peer whose own bound read 2-15 layers was
/// handed 34 of a 36-layer model twice in a row, timing out once and answering
/// `CUDA_ERROR_OUT_OF_MEMORY` the second time (gotcha #454).
#[test]
fn a_peer_without_room_for_the_boomerangs_middle_is_not_given_it() {
    let middle = super::delegated_layer_span(LAYERS, true);
    let mut cramped = willing_peer(0xBB, LAYERS);
    cramped.max_hostable_layers = Some(middle - 1);
    let cands = vec![local_full_coverage(), cramped];
    assert!(
        super::delegation_target(
            &cands,
            &super::DelegationInput {
                local_node_id: &local_id(),
                num_layers: LAYERS,
                layers_to_assign: middle,
                local_serves_on_cpu: true,
                model_vram_mb: MODEL_MB,
                local_cpu_tokens_per_sec: 0.0,
                prompt_tokens: None
            }
        )
        .is_none(),
        "a peer one layer short of the middle must not be handed the middle"
    );

    // The control: room for exactly the middle qualifies, so the assertion
    // above cannot be passing because something else disqualified this peer.
    let mut exact = willing_peer(0xBB, LAYERS);
    exact.max_hostable_layers = Some(middle);
    let cands = vec![local_full_coverage(), exact];
    assert!(
        super::delegation_target(
            &cands,
            &super::DelegationInput {
                local_node_id: &local_id(),
                num_layers: LAYERS,
                layers_to_assign: middle,
                local_serves_on_cpu: true,
                model_vram_mb: MODEL_MB,
                local_cpu_tokens_per_sec: 0.0,
                prompt_tokens: None
            }
        )
        .is_some(),
        "room for exactly the layers it is given is enough"
    );

    // And the boomerang is genuinely smaller than the whole model, or the two
    // shapes would be checked against the same number and the span helper
    // would be pointless.
    assert!(middle < super::delegated_layer_span(LAYERS, false));
    assert_eq!(super::delegated_layer_span(LAYERS, false), LAYERS);
}

/// `delegated_layer_span` must equal what `boomerang_assignment` actually hands
/// over. The two being written out separately is how the boomerang came to be
/// admitted by a check that had never been told its span.
#[test]
fn the_span_the_peer_is_checked_against_is_the_span_it_is_given() {
    let local = local_full_coverage();
    let peer = willing_peer(0xBB, LAYERS);
    let segments = super::boomerang_assignment(&local, &peer, LAYERS).expect("boomerang builds");
    let middle = segments
        .iter()
        .find(|s| s.node_id == peer.node_id)
        .expect("the peer has the middle segment");
    assert_eq!(
        middle.layer_range.1 - middle.layer_range.0,
        super::delegated_layer_span(LAYERS, true),
        "the capacity gate and the assignment must agree on the layer count"
    );
    // A model too short to split three ways is not a boomerang at all, and the
    // span helper must say so rather than underflowing.
    assert!(super::boomerang_assignment(&local, &peer, 2).is_none());
    assert_eq!(super::delegated_layer_span(2, true), 2);
}

/// Free VRAM is a figure about the MODEL; `max_hostable_layers` is a figure
/// about this REQUEST, because it charges the prompt's KV cache per layer. A
/// card with room for the weights and none for an 18,000-token conversation
/// passes the first and must fail the second.
///
/// Reported live on v0.3.153 (gotcha #455): the acceptance check is priced at
/// `ADMISSION_KV_CONTEXT` — a fixed 4096 tokens — so the same peer accepted a
/// 29-token request (0.98 s), an 8841-token one (238 s) and an ~18,000-token
/// one that returned nothing at all for its full 600 s deadline.
#[test]
fn a_prompt_too_long_for_the_peers_card_is_not_delegated_to_it() {
    let mut roomy_for_weights = willing_peer(0xBB, LAYERS);
    // Far more free VRAM than the model needs — the old check's only input.
    roomy_for_weights.gpu_vram_available_mb = Some(MODEL_MB * 6);
    // But this prompt's KV leaves room for a fraction of the layers.
    roomy_for_weights.max_hostable_layers = Some(LAYERS / 2);
    let cands = vec![local_full_coverage(), roomy_for_weights];
    assert!(
        super::delegation_target(
            &cands,
            &super::DelegationInput {
                local_node_id: &local_id(),
                num_layers: LAYERS,
                layers_to_assign: LAYERS,
                local_serves_on_cpu: true,
                model_vram_mb: MODEL_MB,
                local_cpu_tokens_per_sec: 0.0,
                prompt_tokens: None
            }
        )
        .is_none(),
        "a card that cannot hold this prompt's KV cache must not be given the model"
    );

    // The control: the same peer, same free VRAM, a prompt that fits.
    let mut fits = willing_peer(0xBB, LAYERS);
    fits.gpu_vram_available_mb = Some(MODEL_MB * 6);
    fits.max_hostable_layers = Some(LAYERS);
    let cands = vec![local_full_coverage(), fits];
    assert!(super::delegation_target(
        &cands,
        &super::DelegationInput {
            local_node_id: &local_id(),
            num_layers: LAYERS,
            layers_to_assign: LAYERS,
            local_serves_on_cpu: true,
            model_vram_mb: MODEL_MB,
            local_cpu_tokens_per_sec: 0.0,
            prompt_tokens: None
        }
    )
    .is_some());
}

/// The processor-speed branch had no memory check of any kind either — it is
/// the branch a card-less peer takes, and a card-less peer's layers land in its
/// RAM, which is just as finite.
#[test]
fn the_processor_speed_branch_is_bounded_by_memory_too() {
    let mut fast_but_full = willing_peer(0xBB, LAYERS);
    fast_but_full.gpu_vram_available_mb = None;
    fast_but_full.est_tokens_per_sec = 12.0;
    fast_but_full.max_hostable_layers = Some(LAYERS - 1);
    let cands = vec![local_full_coverage(), fast_but_full];
    assert!(
        super::delegation_target(
            &cands,
            &super::DelegationInput {
                local_node_id: &local_id(),
                num_layers: LAYERS,
                layers_to_assign: LAYERS,
                local_serves_on_cpu: true,
                model_vram_mb: MODEL_MB,
                local_cpu_tokens_per_sec: 4.0,
                prompt_tokens: None
            }
        )
        .is_none(),
        "being three times faster does not create memory it does not have"
    );
}

/// Unknown capacity must never exclude — a peer that gossips no capability, or
/// gossips the zero every node before v0.3.103 sent, tells us nothing, and
/// reading that as "no room" would empty the candidate set during any rollout.
/// `max_hostable_layers` owns that contract; the delegation gate inherits it.
#[test]
fn a_peer_that_has_told_us_nothing_about_its_memory_is_still_eligible() {
    let mut silent = willing_peer(0xBB, LAYERS);
    silent.max_hostable_layers = None;
    let cands = vec![local_full_coverage(), silent];
    assert!(
        super::delegation_target(
            &cands,
            &super::DelegationInput {
                local_node_id: &local_id(),
                num_layers: LAYERS,
                layers_to_assign: LAYERS,
                local_serves_on_cpu: true,
                model_vram_mb: MODEL_MB,
                local_cpu_tokens_per_sec: 0.0,
                prompt_tokens: None
            }
        )
        .is_some(),
        "unknown is not the same fact as full"
    );
}

/// Delegation sends the plaintext prompt, so an unknown peer is not eligible.
#[test]
fn an_untrusted_peer_is_not_shown_the_prompt() {
    let mut p = willing_peer(0xBB, LAYERS);
    p.trust_score = super::DELEGATE_MIN_TRUST - 0.01;
    let cands = vec![local_full_coverage(), p];
    assert!(super::delegation_target(
        &cands,
        &super::DelegationInput {
            local_node_id: &local_id(),
            num_layers: LAYERS,
            layers_to_assign: LAYERS,
            local_serves_on_cpu: true,
            model_vram_mb: MODEL_MB,
            local_cpu_tokens_per_sec: 0.0,
            prompt_tokens: None
        }
    )
    .is_none());
}

/// Without a size for the model there is nothing to check a peer's room
/// against, and guessing is how the previous attempt went wrong.
#[test]
fn an_unknown_model_size_keeps_the_request_here() {
    let cands = vec![local_full_coverage(), willing_peer(0xBB, LAYERS)];
    assert!(super::delegation_target(
        &cands,
        &super::DelegationInput {
            local_node_id: &local_id(),
            num_layers: LAYERS,
            layers_to_assign: LAYERS,
            local_serves_on_cpu: true,
            model_vram_mb: 0,
            local_cpu_tokens_per_sec: 0.0,
            prompt_tokens: None
        }
    )
    .is_none());
}

/// With no peer able to help, the local node keeps the request and answers
/// slowly. Answering slowly beats not answering — which is what excluding the
/// local node outright would produce.
#[test]
fn with_no_willing_peer_the_request_stays_local() {
    let cands = vec![local_full_coverage()];
    assert!(super::delegation_target(
        &cands,
        &super::DelegationInput {
            local_node_id: &local_id(),
            num_layers: LAYERS,
            layers_to_assign: LAYERS,
            local_serves_on_cpu: true,
            model_vram_mb: MODEL_MB,
            local_cpu_tokens_per_sec: 0.0,
            prompt_tokens: None
        }
    )
    .is_none());
}

/// Never to ourselves, however the candidate list is ordered.
#[test]
fn the_local_node_is_never_its_own_delegate() {
    let mut me = local_full_coverage();
    me.reach = super::ReachTier::DirectMeasured;
    me.gpu_vram_available_mb = Some(80_000);
    let cands = vec![me];
    assert!(super::delegation_target(
        &cands,
        &super::DelegationInput {
            local_node_id: &local_id(),
            num_layers: LAYERS,
            layers_to_assign: LAYERS,
            local_serves_on_cpu: true,
            model_vram_mb: MODEL_MB,
            local_cpu_tokens_per_sec: 0.0,
            prompt_tokens: None
        }
    )
    .is_none());
}

/// Prompt privacy must NOT disqualify a peer. `delegation_target` answers "is
/// there a peer worth involving"; the caller turns that into a whole-model
/// hand-off or a boomerang. Refusing the peer here would strand a privacy-on
/// node on its CPU for no privacy gain — the boomerang keeps the guarantee in
/// full, because the peer only ever sees encrypted intermediate activations.
#[test]
fn privacy_is_the_callers_decision_not_a_peer_filter() {
    let cands = vec![local_full_coverage(), willing_peer(0xBB, LAYERS)];
    assert!(
        super::delegation_target(
            &cands,
            &super::DelegationInput {
                local_node_id: &local_id(),
                num_layers: LAYERS,
                layers_to_assign: LAYERS,
                local_serves_on_cpu: true,
                model_vram_mb: MODEL_MB,
                local_cpu_tokens_per_sec: 0.0,
                prompt_tokens: None
            }
        )
        .is_some(),
        "the peer is eligible regardless of privacy; the caller picks the shape"
    );
}

/// The boomerang keeps both ends here and gives the peer everything between.
/// Every layer left local is one running on the CPU we are trying to get off,
/// so the split is deliberately lopsided.
#[test]
fn the_boomerang_keeps_both_ends_local_and_gives_the_peer_the_middle() {
    let local = local_full_coverage();
    let peer = willing_peer(0xBB, LAYERS);
    let segs = super::boomerang_assignment(&local, &peer, LAYERS).expect("should build");
    assert_eq!(segs.len(), 3);
    assert_eq!(segs[0].node_id, local_id());
    assert_eq!(segs[0].layer_range, (0, 1), "embedding stays here");
    assert_eq!(segs[1].node_id, NodeId([0xBB; 32]));
    assert_eq!(segs[1].layer_range, (1, LAYERS - 1), "peer runs the middle");
    assert_eq!(segs[2].node_id, local_id());
    assert_eq!(
        segs[2].layer_range,
        (LAYERS - 1, LAYERS),
        "sampling stays here"
    );
    // Contiguous and complete: a gap would strand layers, an overlap would run
    // them twice.
    assert_eq!(segs[0].layer_range.1, segs[1].layer_range.0);
    assert_eq!(segs[1].layer_range.1, segs[2].layer_range.0);
    assert_eq!(segs[2].layer_range.1, LAYERS);
}

/// Prompt privacy is the WHOLE point of this shape, so a local node that cannot
/// own both ends must not get one — better to run slowly than to leak.
#[test]
fn a_local_node_missing_an_end_gets_no_boomerang() {
    let peer = willing_peer(0xBB, LAYERS);
    for (first, last) in [(false, true), (true, false)] {
        let mut local = local_full_coverage();
        local.can_be_first = first;
        local.can_be_last = last;
        assert!(
            super::boomerang_assignment(&local, &peer, LAYERS).is_none(),
            "can_be_first={first} can_be_last={last} must not boomerang"
        );
    }
}

/// A peer that does not cover the middle cannot be given it.
#[test]
fn a_peer_missing_the_middle_gets_no_boomerang() {
    let local = local_full_coverage();
    let mut peer = willing_peer(0xBB, LAYERS);
    peer.available_ranges = vec![(0, 2)]; // holds the start, not the middle
    assert!(super::boomerang_assignment(&local, &peer, LAYERS).is_none());
}

/// Too short to split three ways — one layer each end leaves nothing between.
#[test]
fn a_model_too_short_to_split_gets_no_boomerang() {
    let peer = willing_peer(0xBB, 2);
    let mut local = local_full_coverage();
    local.available_ranges = vec![(0, 2)];
    assert!(super::boomerang_assignment(&local, &peer, 2).is_none());
}

/// A peer with no graphics card can still take the work — but only if it is
/// measurably faster at the thing that limits generation. Until 2026-08-18
/// there was no honest way to check this: every processor-only node advertised
/// the same hardcoded 1.70 tokens/s, so the comparison would have been a
/// constant against itself.
#[test]
fn a_clearly_faster_processor_peer_can_take_the_work() {
    let mut peer = willing_peer(0xBB, LAYERS);
    peer.gpu_vram_available_mb = None; // no card at all
    peer.est_tokens_per_sec = 4.0;
    let cands = vec![local_full_coverage(), peer];
    // Ours is 1.0; theirs is 4x that.
    assert!(
        super::delegation_target(
            &cands,
            &super::DelegationInput {
                local_node_id: &local_id(),
                num_layers: LAYERS,
                layers_to_assign: LAYERS,
                local_serves_on_cpu: true,
                model_vram_mb: MODEL_MB,
                local_cpu_tokens_per_sec: 1.0,
                prompt_tokens: None
            }
        )
        .is_some(),
        "a peer 4x faster should take it"
    );
}

/// A nose ahead is not enough. Being wrong about a processor peer must leave
/// the request no worse off than staying here, so the margin is wide.
#[test]
fn a_marginally_faster_processor_peer_does_not() {
    let mut peer = willing_peer(0xBB, LAYERS);
    peer.gpu_vram_available_mb = None;
    peer.est_tokens_per_sec = 1.0 * super::DELEGATE_MIN_CPU_SPEEDUP - 0.01;
    let cands = vec![local_full_coverage(), peer];
    assert!(super::delegation_target(
        &cands,
        &super::DelegationInput {
            local_node_id: &local_id(),
            num_layers: LAYERS,
            layers_to_assign: LAYERS,
            local_serves_on_cpu: true,
            model_vram_mb: MODEL_MB,
            local_cpu_tokens_per_sec: 1.0,
            prompt_tokens: None
        }
    )
    .is_none());
}

/// With our own speed unknown there is nothing to compare against, so a peer
/// without a card is refused rather than guessed at.
#[test]
fn an_unknown_local_speed_refuses_a_processor_peer() {
    let mut peer = willing_peer(0xBB, LAYERS);
    peer.gpu_vram_available_mb = None;
    peer.est_tokens_per_sec = 999.0;
    let cands = vec![local_full_coverage(), peer];
    assert!(super::delegation_target(
        &cands,
        &super::DelegationInput {
            local_node_id: &local_id(),
            num_layers: LAYERS,
            layers_to_assign: LAYERS,
            local_serves_on_cpu: true,
            model_vram_mb: MODEL_MB,
            local_cpu_tokens_per_sec: 0.0,
            prompt_tokens: None
        }
    )
    .is_none());
}

/// A card with room still wins without any speed comparison — it is a clear
/// improvement over our processor fallback whatever the advertised rates say.
#[test]
fn a_card_with_room_needs_no_speed_comparison() {
    let mut peer = willing_peer(0xBB, LAYERS);
    peer.est_tokens_per_sec = 0.0; // says nothing about its speed
    let cands = vec![local_full_coverage(), peer];
    assert!(super::delegation_target(
        &cands,
        &super::DelegationInput {
            local_node_id: &local_id(),
            num_layers: LAYERS,
            layers_to_assign: LAYERS,
            local_serves_on_cpu: true,
            model_vram_mb: MODEL_MB,
            local_cpu_tokens_per_sec: 99.0,
            prompt_tokens: None
        }
    )
    .is_some());
}

/// The ACK fast-fail abandons a silent peer early, and is justified ONLY by the
/// failover it enables. It asked whether the REQUEST had any standby — but
/// standbys are per segment, so a plan with one standby was fast-failing the
/// four segments that had none, where giving up can only turn a slow success
/// into a 503.
#[test]
fn a_peer_whose_segment_has_no_backup_is_not_abandoned_early() {
    use crate::types::{ModelId, ShardId};
    let seg = |byte: u8, r: (u32, u32)| PipelineSegment {
        node_id: NodeId([byte; 32]),
        shard_id: ShardId {
            model_id: ModelId("m".into()),
            index: 0,
        },
        layer_range: r,
    };
    let segments = vec![seg(10, (0, 16)), seg(11, (16, 32))];
    // One standby, and it covers only the FIRST segment.
    let standbys = vec![seg(20, (0, 16))];

    assert!(
        super::peer_segment_has_standby(&segments, &standbys, &NodeId([10u8; 32])),
        "segment 0 has a backup, so abandoning its holder can buy a failover"
    );
    assert!(
        !super::peer_segment_has_standby(&segments, &standbys, &NodeId([11u8; 32])),
        "segment 1 has none — abandoning its holder can only turn a slow \
         success into a 503, which is the whole reason this gate exists"
    );
}

/// A peer this plan cannot place — a chain hop, a tensor-parallel member —
/// keeps the behaviour it had before the standby gate existed, which is to
/// fast-fail. Same convention as a missing pipeline entry counting as "yes".
#[test]
fn a_peer_the_plan_cannot_place_keeps_the_old_behaviour() {
    use crate::types::{ModelId, ShardId};
    let seg = |byte: u8, r: (u32, u32)| PipelineSegment {
        node_id: NodeId([byte; 32]),
        shard_id: ShardId {
            model_id: ModelId("m".into()),
            index: 0,
        },
        layer_range: r,
    };
    let segments = vec![seg(10, (0, 16))];
    assert!(
        super::peer_segment_has_standby(&segments, &[], &NodeId([99u8; 32])),
        "a stranger to this plan is not something we can reason about"
    );
    // And a holder we CAN place, with no standby anywhere, is still refused.
    assert!(!super::peer_segment_has_standby(
        &segments,
        &[],
        &NodeId([10u8; 32])
    ));
}

/// Gotcha #462, the routing half. A standby is chosen per segment and the local
/// node is sorted first for every one of them, so one machine became the
/// standby for all four remote segments of a 48-layer model it held 12 of.
/// Three failed over to it in turn and its worker was killed.
///
/// The plan said `standbys=4`; it had the capacity to be one.
#[test]
fn one_node_is_not_made_standby_for_more_layers_than_it_can_run() {
    use crate::inference::scheduler::{NodeCandidate, ReachTier};
    use crate::types::{ModelId, ShardId};

    let small = NodeId([1u8; 32]);

    // Holds the whole 48-layer model, but can only run 20 layers at once.
    let mk = |byte: u8, cap: Option<u32>| NodeCandidate {
        node_id: NodeId([byte; 32]),
        shard_id: ShardId {
            model_id: ModelId("m".into()),
            index: 0,
        },
        available_ranges: vec![(0, 48)],
        reach: ReachTier::DirectMeasured,
        latency_ms: 5,
        load: 0.0,
        trust_score: 1.0,
        can_be_first: true,
        can_be_last: true,
        region_score: 1.0,
        est_tokens_per_sec: 5.0,
        observed_latency_ms_per_layer: None,
        observed_delegated_ms_per_layer: None,
        expected_attempts: 1.0,
        is_pool_member: false,
        gpu_vram_available_mb: None,
        max_hostable_layers: cap,
        observed_prefill_ms_per_layer_byte: None,
        has_gpu: false,
    };

    // Four remote primaries of 12 layers each; `small` is the only other
    // holder, so it is the only standby candidate for all four.
    let seg = |byte: u8, r: (u32, u32)| PipelineSegment {
        node_id: NodeId([byte; 32]),
        shard_id: ShardId {
            model_id: ModelId("m".into()),
            index: 0,
        },
        layer_range: r,
    };
    let segments = vec![
        seg(10, (0, 12)),
        seg(11, (12, 24)),
        seg(12, (24, 36)),
        seg(13, (36, 48)),
    ];
    let candidates = vec![
        mk(10, Some(12)),
        mk(11, Some(12)),
        mk(12, Some(12)),
        mk(13, Some(12)),
        mk(1, Some(20)), // `small`
    ];

    let scheduler = PipelineScheduler::new(make_shared_state());
    let standbys = scheduler.find_standbys(&segments, &candidates, Some(100), 48);

    let taken = standbys.iter().filter(|s| s.node_id == small).count();
    assert_eq!(
        taken, 1,
        "a node that can run 20 layers may stand in for one 12-layer segment, \
         not four — before this it was named standby for every segment and its \
         worker was killed when three of them came home"
    );
    // And the plan is honest about what that leaves uncovered.
    assert_eq!(
        super::segments_without_standby(&segments, &standbys).len(),
        3,
        "the three it cannot cover must be reported, not counted as covered"
    );
}

/// The other side: the check must not be a blanket refusal. Raise the same
/// node's ceiling and it stands in for all four again.
///
/// This is deliberately NOT called a control — it fails without the fix too,
/// for a different reason (with no capacity check the tie-break hands the role
/// to whichever peer sorts first). The real control is
/// `a_standby_is_chosen_by_cost_not_by_ping`, whose candidates all carry
/// `max_hostable_layers: None`: the fix is inert there, and it must keep
/// passing unchanged.
///
/// Worth noting what excludes the peers here: each is primary for 12 layers
/// and can host 12, so its own segment uses its whole ceiling. That is correct
/// — it genuinely could not absorb another segment — and it is why this fix
/// reduces standby coverage on a tight swarm rather than only re-ordering it.
#[test]
fn a_node_with_room_still_stands_in_for_every_segment() {
    use crate::inference::scheduler::{NodeCandidate, ReachTier};
    use crate::types::{ModelId, ShardId};

    let big = NodeId([1u8; 32]);
    let mk = |byte: u8, cap: Option<u32>| NodeCandidate {
        node_id: NodeId([byte; 32]),
        shard_id: ShardId {
            model_id: ModelId("m".into()),
            index: 0,
        },
        available_ranges: vec![(0, 48)],
        reach: ReachTier::DirectMeasured,
        latency_ms: 5,
        load: 0.0,
        trust_score: 1.0,
        can_be_first: true,
        can_be_last: true,
        region_score: 1.0,
        est_tokens_per_sec: 5.0,
        observed_latency_ms_per_layer: None,
        observed_delegated_ms_per_layer: None,
        expected_attempts: 1.0,
        is_pool_member: false,
        gpu_vram_available_mb: None,
        max_hostable_layers: cap,
        observed_prefill_ms_per_layer_byte: None,
        has_gpu: false,
    };
    let seg = |byte: u8, r: (u32, u32)| PipelineSegment {
        node_id: NodeId([byte; 32]),
        shard_id: ShardId {
            model_id: ModelId("m".into()),
            index: 0,
        },
        layer_range: r,
    };
    let segments = vec![
        seg(10, (0, 12)),
        seg(11, (12, 24)),
        seg(12, (24, 36)),
        seg(13, (36, 48)),
    ];
    let candidates = vec![
        mk(10, Some(12)),
        mk(11, Some(12)),
        mk(12, Some(12)),
        mk(13, Some(12)),
        mk(1, Some(48)),
    ];

    let scheduler = PipelineScheduler::new(make_shared_state());
    let standbys = scheduler.find_standbys(&segments, &candidates, Some(100), 48);
    assert_eq!(
        standbys.iter().filter(|s| s.node_id == big).count(),
        4,
        "48 layers of room covers four 12-layer segments"
    );
}

/// Primary duty is part of the tally: a node running a segment must be able to
/// run that segment AND anything it stands in for, because a failover is when
/// it does both at once.
#[test]
fn being_primary_uses_up_the_room_a_standby_would_need() {
    assert!(
        super::standby_has_room(Some(20), 0, 12),
        "control: with nothing committed, 12 layers fit inside 20"
    );
    assert!(
        !super::standby_has_room(Some(20), 12, 12),
        "already holding 12 of its 20, it cannot also stand in for 12 more"
    );
    // Unknown never excludes — the contract `max_hostable_layers` sets, and
    // what keeps a mixed-version swarm working during a rollout.
    assert!(super::standby_has_room(None, 9_000, 9_000));
}

/// A standby serves the SAME request the primary was chosen for, so it has to be
/// priced by the same model. It used to be ranked on `latency_ms` alone — a
/// health-ping round trip, which says nothing about how fast a machine computes
/// — so a long prompt correctly steered to a fast node would fail over to
/// whichever node happened to answer a ping quickest.
#[test]
fn a_standby_is_chosen_by_cost_not_by_ping() {
    use crate::inference::scheduler::{NodeCandidate, ReachTier};
    use crate::types::{ModelId, ShardId};

    let local = NodeId([9u8; 32]);
    let near_slow = NodeId([1u8; 32]);
    let far_fast = NodeId([2u8; 32]);

    let mk = |byte: u8, latency_ms: u32, tps: f32, gpu: bool| NodeCandidate {
        node_id: NodeId([byte; 32]),
        shard_id: ShardId {
            model_id: ModelId("m".into()),
            index: 0,
        },
        available_ranges: vec![(0, 32)],
        reach: ReachTier::DirectMeasured,
        latency_ms,
        load: 0.0,
        trust_score: 1.0,
        can_be_first: true,
        can_be_last: true,
        region_score: 1.0,
        est_tokens_per_sec: tps,
        observed_latency_ms_per_layer: None,
        observed_delegated_ms_per_layer: None,
        expected_attempts: 1.0,
        is_pool_member: false,
        gpu_vram_available_mb: None,
        max_hostable_layers: None,
        observed_prefill_ms_per_layer_byte: None,
        has_gpu: gpu,
    };

    // The primary, plus two possible standbys: one very close and very slow,
    // one further away and much faster.
    let primary = mk(3, 10, 5.0, false);
    let candidates = vec![
        primary.clone(),
        mk(1, 2, 0.4, false),   // near_slow  — wins on ping by a mile
        mk(2, 120, 20.0, true), // far_fast   — the machine that can do the work
    ];
    let segments = vec![PipelineSegment {
        node_id: primary.node_id.clone(),
        shard_id: primary.shard_id.clone(),
        layer_range: (0, 32),
    }];

    let scheduler = PipelineScheduler::new(make_shared_state());
    let standbys = scheduler.find_standbys(&segments, &candidates, Some(6000), 32);
    assert_eq!(standbys.len(), 1, "expected one standby");
    assert_eq!(
        standbys[0].node_id, far_fast,
        "a 6000-token prompt must fail over to the machine that can read it, not \
         the one with the shortest ping"
    );
    let _ = near_slow;
    let _ = local;
}

#[test]
fn a_plan_reports_which_segments_have_no_standby_not_just_how_many_it_found() {
    let node = |b: u8| NodeId([b; 32]);
    let seg = |n: u8, range: (u32, u32)| PipelineSegment {
        node_id: node(n),
        shard_id: ShardId {
            model_id: ModelId("m".into()),
            index: 0,
        },
        layer_range: range,
    };
    // The shape the field report carried: local ends, one peer holding the
    // middle nobody else has, and a standby that covers only the tail.
    let segments = vec![seg(1, (0, 2)), seg(2, (2, 30)), seg(1, (30, 32))];
    let standbys = vec![seg(2, (30, 32))];

    assert_eq!(
        super::segments_without_standby(&segments, &standbys),
        vec![0, 1],
        "a standby for the tail covers neither of the other two segments"
    );
    // …and the plain count that used to be the only thing logged says "1",
    // which is what made the failure look like a contradiction.
    assert_eq!(standbys.len(), 1);

    // Control: a standby whose range spans the middle segment does cover it.
    let wide = vec![seg(3, (0, 32))];
    assert!(super::segments_without_standby(&segments, &wide).is_empty());
    assert!(super::standby_covers(&seg(3, (0, 32)), (2, 30)));
    assert!(!super::standby_covers(&seg(2, (30, 32)), (2, 30)));
}

/// A peer already serving this model must never be capped by its free memory.
///
/// `vram_available_mb` is what the card has spare RIGHT NOW, so a resident
/// model's own weights are missing from it — the one node that certainly can
/// hold the model advertises the least room for it. Capping on that figure
/// routes around the best-placed machine in the swarm, which is the opposite of
/// what the bound is for.
/// A capability announcement carrying a card with `free_mb` free.
fn capability_with_gpu(free_mb: Option<u64>) -> crate::types::NodeCapability {
    crate::types::NodeCapability {
        node_id: NodeId([0u8; 32]),
        gpu: free_mb.map(|free| crate::types::GpuInfo {
            name: "test card".into(),
            vram_total_mb: 8192,
            vram_available_mb: free,
            compute_capability: None,
            memory_bandwidth_gbps: 0.0,
        }),
        cpu: None,
        ram_total_mb: 0,
        ram_available_mb: 0,
        ram_model_budget_mb: None,
        disk_available_mb: 0,
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
    }
}

#[test]
fn a_peer_already_serving_the_model_is_not_capped_by_its_free_memory() {
    // A 3 GB model over 32 layers, and a card reporting 200 MB free because it
    // is busy running that very model.
    let bytes_per_layer = 3_000u64 * 1_048_576 / 32;
    let cap = capability_with_gpu(Some(200));

    let cold = super::max_hostable_layers(Some(&cap), bytes_per_layer, false, 0, 0);
    assert!(
        cold.is_some_and(|k| k < 32),
        "a COLD peer with 200 MB free cannot take a 3 GB model: {cold:?}"
    );

    let warm = super::max_hostable_layers(Some(&cap), bytes_per_layer, true, 0, 0);
    assert_eq!(
        warm, None,
        "a peer already serving this model has already paid for it — the free \
         figure excludes the weights it is running"
    );
}

/// The prompt's KV cache is part of what a peer must hold for THIS request, and
/// the bound used to price weights only (gotcha #447): a warm 6 GB card 500 ms
/// away was handed 24 layers of an 8,111-token prompt, ~2.4 GB of KV it did not
/// have, and its worker died in attention. Gemma-2 geometry: 4 KV heads × 256,
/// f32 plus the f16 mirror = 12 KB per position per layer, 295 KB over 24 layers.
#[test]
fn a_long_prompt_shrinks_the_layers_a_peer_may_take() {
    let bytes_per_layer = 1_600u64 * 1_048_576 / 26; // a 1.6 GB Q4 over 26 layers
    let cap = capability_with_gpu(Some(3_000));
    let per_position_per_layer = 2 * 4 * 256 * 6; // K+V, 4 heads × 256, f32 + f16 mirror
    let short = super::max_hostable_layers(
        Some(&cap),
        bytes_per_layer,
        false,
        per_position_per_layer * 19,
        0,
    );
    let long = super::max_hostable_layers(
        Some(&cap),
        bytes_per_layer,
        false,
        per_position_per_layer * 8_111,
        0,
    );
    assert!(short.is_some() && long.is_some());
    assert!(
        long.unwrap() < short.unwrap(),
        "an 8k prompt must leave room for fewer layers than a 19-token one: {long:?} vs {short:?}"
    );
    // 3000 MB / 1.1 margin ≈ 2727 MB usable; each layer costs ~63 MB of weights
    // plus ~99.6 MB of KV for 8,111 positions → 16 layers, not 26.
    assert!(
        long.unwrap() < 26,
        "the whole model no longer 'fits': {long:?}"
    );
}

/// A WARM peer has paid for its weights but not for this prompt's cache: its
/// free memory bounds the layers by the KV term alone. Before, warm meant
/// uncapped, which is exactly the #447 card.
#[test]
fn a_warm_peer_is_still_bounded_by_the_prompts_kv() {
    let bytes_per_layer = 1_600u64 * 1_048_576 / 26;
    let cap = capability_with_gpu(Some(1_000)); // 1 GB free beside the resident weights
    let per_position_per_layer = 2 * 4 * 256 * 6;
    // Unknown prompt: warm stays uncapped, as it always was.
    assert_eq!(
        super::max_hostable_layers(Some(&cap), bytes_per_layer, true, 0, 0),
        None
    );
    // 8,111 positions × 12 KB ≈ 99.6 MB per layer against ~909 MB usable → 9 layers.
    let capped = super::max_hostable_layers(
        Some(&cap),
        bytes_per_layer,
        true,
        per_position_per_layer * 8_111,
        0,
    );
    assert!(
        capped.is_some_and(|k| k < 24),
        "a warm card with 1 GB free cannot take 24 layers of an 8k prompt: {capped:?}"
    );
}

/// When this node keeps a request a peer was priced far cheaper for, the
/// decision has to name that peer.
///
/// The candidate list is logged with a cost per node, so three reports in one
/// day reduced to "a peer priced 55x cheaper was right there and nothing says
/// why it was not used" — twice with the reporter reasonably inferring a
/// penalty mechanism that does not exist. The reason was always logged; what it
/// was a reason ABOUT was not.
#[test]
fn the_cheapest_peer_that_was_passed_over_can_be_named() {
    let local = local_full_coverage();
    let mut near = willing_peer(0xBB, LAYERS);
    near.latency_ms = 5;
    near.est_tokens_per_sec = 40.0;
    let mut far = willing_peer(0xCC, LAYERS);
    far.latency_ms = 400;
    far.est_tokens_per_sec = 3.0;
    // Holds only half the model, so it is not a whole-model alternative at all.
    let partial = willing_peer(0xDD, LAYERS / 2);

    let cands = vec![local, far, near, partial];
    let (peer, cost) =
        super::cheapest_whole_model_peer(&cands, &local_id(), LAYERS, Some(4_000)).expect("a peer");
    assert_eq!(
        peer.node_id,
        NodeId([0xBB; 32]),
        "the nearest, fastest whole-model holder is the one a reader will ask about"
    );
    assert!(cost > 0.0);

    // The local node is never its own alternative, and a node holding only part
    // of the model is not one either.
    let alone = vec![local_full_coverage(), willing_peer(0xDD, LAYERS / 2)];
    assert!(super::cheapest_whole_model_peer(&alone, &local_id(), LAYERS, Some(4_000)).is_none());
}

/// Two requests scheduled inside one 30-second gossip window must not both be
/// told the peer has all its memory free.
///
/// Live on v0.3.153/.154 (gotcha #457): two requests 3 ms apart, both accepted
/// whole onto one peer against the identical `peer_free_vram_mb=Some(4598)`.
/// Sixteen seconds later one died with a driver-level out-of-memory inside
/// `mlp` while the other went on decoding — and the loser, resent alone
/// afterwards, completed cleanly in 62 s.
#[test]
fn memory_already_booked_on_a_peer_is_not_offered_twice() {
    let cap = capability_with_gpu(Some(4_600));
    // 100 MB per layer, no prompt term — the arithmetic is not what is under
    // test here, the deduction is.
    let bytes_per_layer = 100 * 1_048_576;

    let free = super::max_hostable_layers(Some(&cap), bytes_per_layer, false, 0, 0)
        .expect("an advertised card yields a bound");
    let booked = super::max_hostable_layers(Some(&cap), bytes_per_layer, false, 0, 4_000)
        .expect("still a bound, just a smaller one");
    assert!(
        booked < free,
        "memory this node has already committed must not be offered again: \
         {free} layers free, {booked} after booking 4000 MB"
    );

    // Over-committed reads as no room, never as all of it — the subtraction
    // saturates rather than wrapping.
    assert_eq!(
        super::max_hostable_layers(Some(&cap), bytes_per_layer, false, 0, 99_999),
        Some(0)
    );

    // And a peer that has told us NOTHING is still unknown, not full: our own
    // commitments cannot manufacture information the peer never sent.
    let silent = capability_with_gpu(None);
    assert_eq!(
        super::max_hostable_layers(Some(&silent), bytes_per_layer, false, 0, 4_000),
        None
    );
}

/// The coordinator prices a position the way the worker charges it, mirror
/// included on a card and excluded on a processor.
#[test]
fn a_prompt_position_is_priced_like_the_worker_charges_it() {
    let meta = crate::inference::split::GgufTensorMeta {
        tensors: Default::default(),
        tensor_data_offset: 0,
        model_name: None,
        head_count: 8,
        head_count_kv: 4,
        block_count: 26,
        embedding_length: 2304,
        head_dim: 256,
        rope_dim: 256,
        rope_freq_base: 10_000.0,
        rms_norm_eps: 1e-6,
        expert_count: 0,
        architecture: "gemma2".into(),
    };
    // GQA on a card: f32 + f16 mirror → 6 bytes per element.
    assert_eq!(
        super::kv_bytes_per_position_per_layer(&meta, true),
        2 * 4 * 256 * 6
    );
    // On a processor there is no mirror.
    assert_eq!(
        super::kv_bytes_per_position_per_layer(&meta, false),
        2 * 4 * 256 * 4
    );
    // An MHA model keeps no mirror even on a card.
    let mha = crate::inference::split::GgufTensorMeta {
        head_count: 8,
        head_count_kv: 8,
        ..meta
    };
    assert_eq!(
        super::kv_bytes_per_position_per_layer(&mha, true),
        2 * 8 * 256 * 4
    );
}

/// Unknown must still mean unknown: a peer that gossips nothing, or gossips the
/// zero every node before v0.3.103 sent, is routed to exactly as before.
#[test]
fn an_unreadable_memory_figure_never_caps_a_peer() {
    assert_eq!(super::max_hostable_layers(None, 1024, false, 0, 0), None);

    let zeroed = capability_with_gpu(Some(0));
    assert_eq!(
        super::max_hostable_layers(Some(&zeroed), 1024, false, 0, 0),
        None,
        "zero free VRAM is what a pre-v0.3.103 node always advertised — it is \
         no information, not 'no room' (gotcha #330)"
    );
}

// ---------------------------------------------------------------------------
// A node holding every layer that would run the model on its PROCESSOR lets
// the priced search compete with its fast path (gotcha #444). In this build
// `ModelProcessPool::serves_on_cpu` is always true — there is no card — so
// every assembly below is on the processor side of that question.
// ---------------------------------------------------------------------------

/// What the local processor is said to manage on a 7B in these tests: the
/// tester's node (5.5 tok/s on its 14B, more on a 7B). Pinned, because the
/// real figure is measured on the machine running the test — a loaded CI
/// runner measured itself slow enough to send the short-prompt control to
/// cards 900 ms away (2026-09-03).
const LOCAL_PROCESSOR_TPS: f32 = 8.0;

/// A holder with a card advertising room, at `latency_ms`, stating a speed.
fn gpu_holder_info(node: &NodeId, latency_ms: u32, tokens_per_sec: f32) -> PeerInfo {
    let mut cap = capability_with_gpu(Some(24_000));
    cap.node_id = node.clone();
    cap.est_tokens_per_sec_7b = tokens_per_sec;
    PeerInfo {
        node_id: node.clone(),
        addresses: vec![],
        capability: Some(cap),
        last_seen: chrono::Utc::now(),
        latency_ms: Some(latency_ms),
        trust_score: 0.9,
        peer_id_bytes: None,
        ack_srtt_ms: None,
        active_request_count: 0,
        first_seen: 0,
        verified_transaction_count: 0,
        is_lan_peer: false,
    }
}

/// The local node holds both halves of a two-shard model; peers B and C hold
/// one half each, on cards, `peer_latency_ms` away. No single peer holds the
/// whole model, so the whole-model hand-off cannot fire — only a pipeline can.
fn processor_holder_beside_two_gpu_halves(
    peer_latency_ms: u32,
    peer_tokens_per_sec: f32,
) -> (Arc<SharedState>, NodeId, NodeId, NodeId) {
    let state = make_shared_state();
    let local = state.identity.node_id().clone();
    let b = NodeId([0xB1; 32]);
    let c = NodeId([0xC1; 32]);
    let model = "split-14b";
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
        .register_manifest(make_manifest(model, 32, shards));
    let sid = |i: u32| ShardId {
        model_id: ModelId(model.into()),
        index: i,
    };
    state
        .model_registry
        .record_shard_holder(sid(0), local.clone());
    state
        .model_registry
        .record_shard_holder(sid(1), local.clone());
    state.model_registry.record_shard_holder(sid(0), b.clone());
    state.model_registry.record_shard_holder(sid(1), c.clone());
    for n in [&b, &c] {
        state.peer_registry.insert(
            n.clone(),
            gpu_holder_info(n, peer_latency_ms, peer_tokens_per_sec),
        );
        state.connected_node_ids.insert(n.clone());
    }
    (state, local, b, c)
}

/// The tester's case (gotcha #442, second half): a processor-only node that
/// holds every shard of a model no single card holds, two GPU peers on the LAN
/// each holding half, and an agent-sized prompt. The day before it held the
/// last shard, a three-segment pipeline across those cards answered in seconds;
/// holding it turned every request into minutes on the processor, because full
/// local coverage ended the search. The priced search must win here.
///
/// Holding both ends turns prompt privacy on automatically, so the winning
/// shape is the boomerang: embedding here, the two cards for the middle,
/// sampling here. The peers see encrypted hidden states and never the prompt.
#[test]
fn a_processor_bound_holder_hands_a_long_prompt_to_a_pipeline_of_faster_cards() {
    let (state, local, b, c) = processor_holder_beside_two_gpu_halves(5, 20.0);
    assert!(
        state.encrypted_pipeline_for(&ModelId("split-14b".into())),
        "the fixture holds both ends, so privacy must be auto-on"
    );
    let scheduler = PipelineScheduler::with_local_processor_speed(state, LOCAL_PROCESSOR_TPS);
    let assignment = scheduler
        .assemble_pipeline_for(
            &ModelId("split-14b".into()),
            &local,
            uuid::Uuid::new_v4(),
            Some(14_000),
        )
        .unwrap();
    let shape: Vec<(NodeId, (u32, u32))> = assignment
        .segments
        .iter()
        .map(|s| (s.node_id.clone(), s.layer_range))
        .collect();
    assert_eq!(
        shape,
        vec![
            (local.clone(), (0, 1)),
            (b, (1, 16)),
            (c, (16, 31)),
            (local, (31, 32)),
        ],
        "embedding and sampling stay home, the cards take the middle; got {:?}",
        assignment.segments
    );
}

/// The same node with prompt privacy switched off hands the two cards a half
/// each and keeps nothing.
#[test]
fn with_privacy_off_the_processor_bound_holder_hands_the_whole_model_to_the_cards() {
    let (state, local, b, c) = processor_holder_beside_two_gpu_halves(5, 20.0);
    state
        .encrypted_pipeline_models
        .insert(ModelId("split-14b".into()), false);
    assert!(!state.encrypted_pipeline_for(&ModelId("split-14b".into())));
    let scheduler = PipelineScheduler::with_local_processor_speed(state, LOCAL_PROCESSOR_TPS);
    let assignment = scheduler
        .assemble_pipeline_for(
            &ModelId("split-14b".into()),
            &local,
            uuid::Uuid::new_v4(),
            Some(14_000),
        )
        .unwrap();
    let nodes: Vec<NodeId> = assignment
        .segments
        .iter()
        .map(|s| s.node_id.clone())
        .collect();
    assert_eq!(nodes, vec![b, c], "{:?}", assignment.segments);
}

/// The control, and the guard against the reverted `cbbed678`: distant cards
/// and a short prompt. Every remote hop is charged per token, so at 900 ms a
/// pipeline costs far more than any processor's decode — the request stays
/// here, in the fast path's shape with nobody else involved. The same fixture
/// with the cards 5 ms away goes to them, so it is the distance that decides,
/// not the absence of a route.
#[test]
fn a_short_prompt_stays_on_the_processor_when_the_cards_are_far_away() {
    let assemble = |peer_latency_ms: u32| {
        let (state, local, _b, _c) = processor_holder_beside_two_gpu_halves(peer_latency_ms, 20.0);
        let scheduler = PipelineScheduler::with_local_processor_speed(state, LOCAL_PROCESSOR_TPS);
        let assignment = scheduler
            .assemble_pipeline_for(
                &ModelId("split-14b".into()),
                &local,
                uuid::Uuid::new_v4(),
                None,
            )
            .unwrap();
        (local, assignment)
    };
    let (local, far) = assemble(900);
    assert_eq!(far.segments.len(), 1, "{:?}", far.segments);
    assert_eq!(far.segments[0].node_id, local);
    assert!(
        far.standbys.is_empty() && far.tp_groups.is_empty(),
        "the fast path's shape: nobody else involved"
    );
    let (local, near) = assemble(5);
    assert!(
        near.segments.iter().any(|s| s.node_id != local),
        "with the same cards on the LAN the route exists and is taken: {:?}",
        near.segments
    );
}

/// The price decides, not a distance constant: a processor slow enough makes
/// even cards 900 ms away the better route for a short prompt. This is the
/// arm a loaded CI runner produced by accident when the local speed was still
/// measured rather than pinned — correct routing for the input it was given.
#[test]
fn a_slow_enough_processor_hands_even_a_short_prompt_to_distant_cards() {
    let (state, local, _b, _c) = processor_holder_beside_two_gpu_halves(900, 20.0);
    let scheduler = PipelineScheduler::with_local_processor_speed(state, 0.2);
    let assignment = scheduler
        .assemble_pipeline_for(
            &ModelId("split-14b".into()),
            &local,
            uuid::Uuid::new_v4(),
            None,
        )
        .unwrap();
    assert!(
        assignment.segments.iter().any(|s| s.node_id != local),
        "at 0.2 tok/s the processor loses to distant cards: {:?}",
        assignment.segments
    );
}

/// Prices under which the comparison added for report #017 stands aside, so a
/// test about one of the OTHER grounds is not silently decided by it.
fn chain_is_cheaper() -> super::RoutePrices {
    super::RoutePrices {
        local_ms: 1000.0,
        chain_ms: 100.0,
        local_route_is_available: true,
    }
}

/// The gate is named for a comparison it did not make. It approved a chain
/// priced at 3.5x the local processor on a live node — 22117 ms against
/// 6313 ms — and logged "a pipeline across peers' cards is priced faster"
/// over its own numbers (report #017).
///
/// The DP does minimise, so this is not merely a redundant second opinion:
/// its capacity-respecting pass DROPS the "stay fully local" vertex when this
/// node cannot hold every layer, so what it returns is the cheapest FEASIBLE
/// chain, which is not the same claim.
#[test]
fn a_pipeline_priced_dearer_than_the_processor_does_not_take_the_request() {
    let mut local = local_full_coverage();
    local.est_tokens_per_sec = 20.0;
    let peer = willing_peer(0xBB, LAYERS);
    let remote_chain = vec![PipelineSegment {
        node_id: peer.node_id.clone(),
        shard_id: peer.shard_id.clone(),
        layer_range: (0, LAYERS),
    }];
    let dearer = super::RoutePrices {
        local_ms: 6313.48,
        chain_ms: 22117.55,
        local_route_is_available: true,
    };
    assert!(
        super::pipeline_may_replace_processor_route(
            &remote_chain,
            &[local.clone(), peer.clone()],
            &local_id(),
            dearer
        )
        .is_err(),
        "the observed live figures: a chain at 3.5x the processor is not faster"
    );

    // The control, on the identical fixture: the same chain priced cheaper is
    // still taken, so this refuses on the PRICE and not on the shape.
    assert!(
        super::pipeline_may_replace_processor_route(
            &remote_chain,
            &[local, peer],
            &local_id(),
            chain_is_cheaper()
        )
        .is_ok(),
        "a genuinely cheaper chain must still displace the processor"
    );
}

/// ...and the case that makes the comparison safe to add. A node whose loader
/// will refuse the whole model has no local route to keep, so the chain wins
/// at any price — it is not competing with the processor, it is the only way
/// the request is answered at all. Without this, adding the comparison would
/// have sent report #018's request home to a 503.
#[test]
fn a_node_that_cannot_hold_every_layer_gives_the_chain_the_request_at_any_price() {
    let mut local = local_full_coverage();
    local.est_tokens_per_sec = 20.0;
    let peer = willing_peer(0xBB, LAYERS);
    let remote_chain = vec![PipelineSegment {
        node_id: peer.node_id.clone(),
        shard_id: peer.shard_id.clone(),
        layer_range: (0, LAYERS),
    }];
    let dearer_but_the_only_route = super::RoutePrices {
        local_ms: 6313.48,
        chain_ms: 22117.55,
        local_route_is_available: false,
    };
    assert!(
        super::pipeline_may_replace_processor_route(
            &remote_chain,
            &[local, peer],
            &local_id(),
            dearer_but_the_only_route
        )
        .is_ok(),
        "there is no local route to give up, so the price of one is not a reason to stay"
    );
}

/// Unknown never excludes, on the local node as on every peer: an unreadable
/// footprint or an unset budget must leave the plan exactly as it was.
///
/// The pool here holds no worker, so every answer comes from the bound —
/// which is the arm this test is about. The resident-worker arm is
/// `a_model_this_node_is_already_running_is_not_judged_too_big_for_it`.
#[test]
fn an_unreadable_local_capacity_still_lets_this_node_run_the_whole_model() {
    let pool = crate::inference::process_pool::ModelProcessPool::new(std::path::PathBuf::from(
        "/tmp/swarmllm-local-capacity-test",
    ));
    let model = ModelId("split-14b".into());
    let mut local = local_full_coverage();
    assert!(
        local.max_hostable_layers.is_none(),
        "fixture must start unbounded, or this asserts nothing"
    );
    assert!(super::local_can_hold_every_layer(
        &pool, &model, &local, LAYERS
    ));
    local.max_hostable_layers = Some(LAYERS);
    assert!(super::local_can_hold_every_layer(
        &pool, &model, &local, LAYERS
    ));
    local.max_hostable_layers = Some(LAYERS - 1);
    assert!(
        !super::local_can_hold_every_layer(&pool, &model, &local, LAYERS),
        "one layer short is short"
    );
}

/// The two remaining grounds for keeping the request here, after gotcha #479
/// removed the third (a blanket veto on any chain containing an unmeasured
/// peer). The all-local arm is unchanged; the other is now about the BASELINE
/// being known rather than the peer.
#[test]
fn the_processor_route_is_kept_for_an_all_local_chain_or_an_unpriced_baseline() {
    let mut local = local_full_coverage();
    let peer = willing_peer(0xBB, LAYERS);
    assert_eq!(
        peer.est_tokens_per_sec, 0.0,
        "the fixture must start unpriced"
    );
    let remote_chain = vec![PipelineSegment {
        node_id: peer.node_id.clone(),
        shard_id: peer.shard_id.clone(),
        layer_range: (0, LAYERS),
    }];

    // Our own speed unknown: nothing to compare, so stay.
    assert!(
        super::pipeline_may_replace_processor_route(
            &remote_chain,
            &[local.clone(), peer.clone()],
            &local_id(),
            chain_is_cheaper()
        )
        .is_err(),
        "with no price for running it here there is nothing to give up"
    );

    // Our own speed known: the search's verdict stands, even though the peer
    // is priced at the prior — that prior is what makes it safe.
    local.est_tokens_per_sec = 0.4;
    assert!(
        super::pipeline_may_replace_processor_route(
            &remote_chain,
            &[local.clone(), peer.clone()],
            &local_id(),
            chain_is_cheaper()
        )
        .is_ok(),
        "an unmeasured peer is priced pessimistically, not excluded"
    );

    // An all-local chain is the fast path by another name, whatever is known.
    let local_chain = vec![PipelineSegment {
        node_id: local.node_id.clone(),
        shard_id: local.shard_id.clone(),
        layer_range: (0, LAYERS),
    }];
    assert!(
        super::pipeline_may_replace_processor_route(
            &local_chain,
            &[local, peer],
            &local_id(),
            chain_is_cheaper()
        )
        .is_err(),
        "an all-local chain is the fast path by another name"
    );
}

/// The local candidate is priced by the device the REQUEST would use, not the
/// device the node owns: a card this model does not fit contributes neither
/// its speed nor its prefill prior.
#[test]
fn the_local_candidate_is_priced_by_the_device_the_request_would_use() {
    let identity = Identity::generate();
    let temp = tempfile::tempdir().unwrap();
    let db = Database::open(temp.path()).unwrap();
    let executor = Arc::new(Mutex::new(ModelExecutor::new()));
    let card = crate::inference::executor::GpuInfo {
        name: "NVIDIA GeForce RTX 3070".into(),
        vram_total_mb: 8192,
        vram_free_mb: 7000,
        backend: "cuda".into(),
    };
    let (state, _, _) = SharedState::new(Config::default(), identity, db, executor, Some(card));
    let local = state.identity.node_id().clone();
    let mid = ModelId("too-big-for-the-card".into());
    state.model_registry.register_manifest(make_manifest(
        &mid.0,
        32,
        vec![ShardInfo {
            index: 0,
            layer_range: (0, 32),
            size_bytes: 4_000_000_000,
            hash: [0u8; 32],
            tensors: vec![],
        }],
    ));
    state.model_registry.record_shard_holder(
        ShardId {
            model_id: mid.clone(),
            index: 0,
        },
        local.clone(),
    );
    let manifest = state.model_registry.get_manifest(&mid).unwrap();
    let scheduler = PipelineScheduler::new(state);
    let pick = |cands: Vec<NodeCandidate>| cands.into_iter().find(|c| c.node_id == local).unwrap();
    let on_card =
        pick(scheduler.gather_candidates(&manifest, &local, uuid::Uuid::new_v4(), None, &|| false));
    let on_processor =
        pick(scheduler.gather_candidates(&manifest, &local, uuid::Uuid::new_v4(), None, &|| true));
    assert!(
        on_card.has_gpu,
        "a request the card runs gets the card's prefill prior"
    );
    assert!(
        !on_processor.has_gpu,
        "a request the processor runs must not get the card's prefill prior"
    );
    assert!(on_card.est_tokens_per_sec > 0.0);
    assert!(
        on_processor.est_tokens_per_sec < on_card.est_tokens_per_sec,
        "the processor figure ({}) must not be the card's ({})",
        on_processor.est_tokens_per_sec,
        on_card.est_tokens_per_sec
    );
}

/// The live reproduction of gotcha #478, measured on the release pair
/// 2026-09-06: a processor-only node holding llama-3.2-1b whole, and an
/// otherwise perfectly good card 496 ms away.
///
/// Under prompt privacy the peer is given the MIDDLE, and a middle is entered
/// once per token — `vertex_cost` exempts only a remote range covering the
/// whole model from per-token network, and says so. So the shape actually
/// assigned pays `2 x 496 ms` per token, against a local processor decoding at
/// 4.28 tok/s (~234 ms/token). Handing it over is four times slower, and the
/// measured request took 9.1 s to return a single token.
///
/// The gate priced the peer at `(0, num_layers)` — the delegated shape, with
/// no per-token network at all — while assigning `(1, n-1)`. Same class as
/// gotcha #434: what was priced was not what was run.
fn boomerang_pair() -> (Vec<NodeCandidate>, u32) {
    const N: u32 = 16;
    let mut local = cost_cand(0xAA, vec![(0, N)], super::ReachTier::Local, 0, 0.0, None);
    local.node_id = local_id();
    local.est_tokens_per_sec = 4.279; // measured on the i5-10500T
    let mut peer = cost_cand(
        0xBB,
        vec![(0, N)],
        super::ReachTier::DirectMeasured,
        496,
        0.0,
        None,
    );
    peer.est_tokens_per_sec = 15.206; // the Mac mini, as it advertised itself
    peer.gpu_vram_available_mb = Some(24_000);
    (vec![local, peer], N)
}

#[test]
fn a_boomerangs_middle_is_priced_as_the_middle_it_will_be_given() {
    let (cands, n) = boomerang_pair();
    assert!(
        super::delegation_target(
            &cands,
            &super::DelegationInput {
                local_node_id: &local_id(),
                num_layers: n,
                // What prompt privacy actually hands over: the middle.
                layers_to_assign: super::delegated_layer_span(n, true),
                local_serves_on_cpu: true,
                model_vram_mb: 1_200,
                local_cpu_tokens_per_sec: 4.279,
                prompt_tokens: Some(16),
            }
        )
        .is_none(),
        "a middle segment half a second away is entered once per token; handing \
         it over must not be priced as though the peer ran the whole model"
    );
}

/// The control, and the thing that must not regress: the SAME peer, offered
/// the whole model, is a genuine improvement and is still taken. Without this
/// the fix above could be "never delegate", which is the failure
/// `delegation_target` exists to prevent.
#[test]
fn the_same_peer_still_gets_the_whole_model_when_that_is_the_shape() {
    let (cands, n) = boomerang_pair();
    assert_eq!(
        super::delegation_target(
            &cands,
            &super::DelegationInput {
                local_node_id: &local_id(),
                num_layers: n,
                layers_to_assign: super::delegated_layer_span(n, false),
                local_serves_on_cpu: true,
                model_vram_mb: 1_200,
                local_cpu_tokens_per_sec: 4.279,
                prompt_tokens: Some(16),
            }
        )
        .map(|c| c.node_id.clone()),
        Some(NodeId([0xBB; 32])),
        "delegating the whole model round-trips once for the request, not once \
         per token — that peer is still worth handing it to"
    );
}

/// A pipeline segment for the route-veto tests.
fn chain_seg(node_id: NodeId, layer_range: (u32, u32)) -> PipelineSegment {
    PipelineSegment {
        node_id,
        shard_id: crate::types::ShardId {
            model_id: crate::types::ModelId("m".into()),
            index: 0,
        },
        layer_range,
    }
}

/// Gotcha #479. A chain the search priced cheaper is no longer refused merely
/// for containing a peer we have never measured — `UNKNOWN_COMPUTE_MS` already
/// prices that peer pessimistically, and refusing the outcome afterwards meant
/// the prior could never do the job it was raised from 0 to do.
#[test]
fn a_chain_with_an_unmeasured_peer_is_allowed_once_our_own_speed_is_known() {
    let mut local = local_full_coverage();
    local.est_tokens_per_sec = 0.4; // a processor that would take many minutes
    let unmeasured = cost_cand(
        0xBB,
        vec![(0, LAYERS)],
        super::ReachTier::DirectMeasured,
        20,
        0.0,
        None,
    );
    assert!(
        !super::priced_from_a_measurement(&unmeasured),
        "fixture must actually be unpriced, or this test asserts nothing"
    );
    let chain = vec![
        chain_seg(local_id(), (0, 1)),
        chain_seg(NodeId([0xBB; 32]), (1, LAYERS - 1)),
        chain_seg(local_id(), (LAYERS - 1, LAYERS)),
    ];
    assert!(
        super::pipeline_may_replace_processor_route(
            &chain,
            &[local, unmeasured],
            &local_id(),
            chain_is_cheaper()
        )
        .is_ok(),
        "the search already priced this cheaper than a 0.4 tok/s processor; \
         the prior is the conservatism, not a veto on top of it"
    );
}

/// The baseline has to be known, because giving it up is the whole question.
#[test]
fn a_chain_is_kept_home_when_our_own_speed_is_not_yet_measured() {
    let local = local_full_coverage(); // est_tokens_per_sec stays 0.0
    assert!(
        !super::priced_from_a_measurement(&local),
        "fixture must be unpriced locally"
    );
    let mut peer = cost_cand(
        0xBB,
        vec![(0, LAYERS)],
        super::ReachTier::DirectMeasured,
        20,
        0.0,
        None,
    );
    peer.est_tokens_per_sec = 30.0;
    let chain = vec![chain_seg(NodeId([0xBB; 32]), (0, LAYERS))];
    assert!(
        super::pipeline_may_replace_processor_route(
            &chain,
            &[local, peer],
            &local_id(),
            chain_is_cheaper()
        )
        .is_err(),
        "with no price for running it here there is nothing to compare against"
    );
}

/// The control that makes removing the veto safe: the prior alone keeps an
/// unmeasured peer from beating an ORDINARY processor, so the search only ever
/// reaches for one when staying here would be dire. If this ever fails, the
/// veto was load-bearing after all and the doc on
/// `pipeline_may_replace_processor_route` is wrong.
#[test]
fn the_unknown_prior_alone_outprices_an_unmeasured_peer_against_a_normal_processor() {
    let range = (0, LAYERS);
    let unmeasured = cost_cand(
        0xBB,
        vec![range],
        super::ReachTier::DirectMeasured,
        20,
        0.0,
        None,
    );
    let mut ordinary_local = local_full_coverage();
    ordinary_local.est_tokens_per_sec = 5.3; // this development machine, measured

    let peer_ms =
        super::parallax::vertex_cost(&unmeasured, range, &local_id(), LAYERS, Some(64)).total();
    let local_ms =
        super::parallax::vertex_cost(&ordinary_local, range, &local_id(), LAYERS, Some(64)).total();
    assert!(
        peer_ms > local_ms,
        "an unmeasured peer must not outprice an ordinary processor: \
         peer={peer_ms} local={local_ms}"
    );

    // ... and the crossover is where the doc says it is: a processor this slow
    // takes minutes, which is the case the search is now allowed to act on.
    let mut dire_local = local_full_coverage();
    dire_local.est_tokens_per_sec = 0.4;
    let dire_ms =
        super::parallax::vertex_cost(&dire_local, range, &local_id(), LAYERS, Some(64)).total();
    assert!(
        peer_ms < dire_ms,
        "against a 0.4 tok/s processor the unmeasured peer must finally win: \
         peer={peer_ms} dire={dire_ms}"
    );
}

/// Gotcha #485. The greedy fallback can return to a candidate it has already
/// used, and applied the memory cap fresh each time — so a node capped at N
/// could be handed several segments summing to more than N. That is the shape
/// the #452 field report showed (one node given 0..35 AND 47..48 of a 48-layer
/// model), and the invariant the DP has guarded since it was written.
///
/// Note what the sibling test above could not see: it asserts on
/// `segments[0]`, so it checks the per-SEGMENT cap and passes happily while
/// the per-NODE total is exceeded.
#[test]
fn the_greedy_fallback_does_not_hand_one_node_more_than_it_can_hold() {
    let state = make_shared_state();
    let scheduler = PipelineScheduler::new(state);

    // Node 1 declares every layer but can hold 12. Node 2 covers only the
    // middle, so a correct plan uses node 1, then node 2, and would otherwise
    // come back to node 1 for the tail — the return visit is the whole point.
    let mut wide = simple_candidate(1, vec![(0, 48)]);
    wide.max_hostable_layers = Some(12);
    let mut middle = simple_candidate(2, vec![(12, 30)]);
    middle.max_hostable_layers = Some(18);
    // A third node with room for the tail, so a plan that respects every
    // bound genuinely exists and the test is not asserting a refusal.
    let mut tail = simple_candidate(3, vec![(24, 48)]);
    tail.max_hostable_layers = Some(24);

    let segments = scheduler
        .greedy_assign(48, &[wide, middle, tail], false)
        .expect("a plan respecting every bound exists");

    let mut per_node: std::collections::HashMap<NodeId, u32> = std::collections::HashMap::new();
    for seg in &segments {
        *per_node.entry(seg.node_id.clone()).or_insert(0) += seg.layer_range.1 - seg.layer_range.0;
    }
    let wide_total = per_node.get(&NodeId([1u8; 32])).copied().unwrap_or(0);
    assert!(
        wide_total <= 12,
        "node 1 is capped at 12 and was handed {wide_total} layers across \
         {} segments: {segments:?}",
        segments.len()
    );
    // And the plan is still a plan: every layer covered, contiguously.
    assert_eq!(segments.first().map(|s| s.layer_range.0), Some(0));
    assert_eq!(segments.last().map(|s| s.layer_range.1), Some(48));
}

/// When no assignment fits every bound, the constrained pass must FAIL so the
/// caller re-runs without the bound — never hand the layer over anyway, which
/// would fragment the overage into one-layer segments and still exceed it.
#[test]
fn an_impossible_bound_falls_through_to_the_relaxed_pass() {
    let state = make_shared_state();
    let scheduler = PipelineScheduler::new(state);

    // The only holder of the whole model can hold a third of it, and nobody
    // else covers anything.
    let mut only = simple_candidate(1, vec![(0, 48)]);
    only.max_hostable_layers = Some(16);

    let constrained = scheduler.greedy_assign_inner(48, &[only.clone()], false, true);
    assert!(
        constrained.is_err(),
        "no plan fits, so the constrained pass must say so: {constrained:?}"
    );

    // …and the public entry point recovers, because a served request beats a
    // refused one and the holder's own admission is the backstop.
    let segments = scheduler
        .greedy_assign(48, &[only], false)
        .expect("the relaxed pass routes it");
    assert_eq!(segments.last().map(|s| s.layer_range.1), Some(48));
}
