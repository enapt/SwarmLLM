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
    let picked = super::delegation_target(&cands, &local_id(), LAYERS, true, MODEL_MB, 0.0);
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
        super::delegation_target(&cands, &local_id(), LAYERS, false, MODEL_MB, 0.0).is_none(),
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
        super::delegation_target(&cands, &local_id(), LAYERS, true, MODEL_MB, 0.0).is_none(),
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
        super::delegation_target(&cands, &local_id(), LAYERS, true, MODEL_MB, 0.0)
            .map(|c| c.node_id.clone()),
        Some(NodeId([0xBB; 32])),
        "a 600 ms GPU peer measured 2.3x faster than the local processor fallback \
         and must not be excluded by the distance bound"
    );
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
        super::delegation_target(&cands, &local_id(), LAYERS, true, MODEL_MB, 0.0)
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
            super::delegation_target(&cands, &local_id(), LAYERS, true, MODEL_MB, 0.0).is_none(),
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
        super::delegation_target(&cands, &local_id(), LAYERS, true, MODEL_MB, 4.0)
            .map(|c| c.node_id.clone()),
        Some(NodeId([0xBB; 32])),
        "three times our processor speed is worth the hand-off"
    );
    let mut barely = willing_peer(0xBB, LAYERS);
    barely.gpu_vram_available_mb = None;
    barely.est_tokens_per_sec = 6.0;
    let cands = vec![local_full_coverage(), barely];
    assert!(
        super::delegation_target(&cands, &local_id(), LAYERS, true, MODEL_MB, 4.0).is_none(),
        "1.5x is inside the margin a self-reported figure gets"
    );
    // And a node that would NOT run it on its processor never delegates.
    let mut faster = willing_peer(0xBB, LAYERS);
    faster.est_tokens_per_sec = 12.0;
    let cands = vec![local_full_coverage(), faster];
    assert!(super::delegation_target(&cands, &local_id(), LAYERS, false, MODEL_MB, 4.0).is_none());
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
    assert!(super::delegation_target(&cands, &local_id(), LAYERS, true, MODEL_MB, 0.0).is_none());
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
            super::delegation_target(&cands, &local_id(), LAYERS, true, MODEL_MB, 0.0).is_none(),
            "free={free:?} is not enough room for a {MODEL_MB} MB model"
        );
    }
    // Comfortably above the margin, so it qualifies — otherwise the assertions
    // above would pass for the wrong reason.
    let mut ok = willing_peer(0xBB, LAYERS);
    ok.gpu_vram_available_mb = Some((MODEL_MB as f64 * super::DELEGATE_VRAM_MARGIN) as u64 + 1);
    let cands = vec![local_full_coverage(), ok];
    assert!(super::delegation_target(&cands, &local_id(), LAYERS, true, MODEL_MB, 0.0).is_some());
}

/// Delegation sends the plaintext prompt, so an unknown peer is not eligible.
#[test]
fn an_untrusted_peer_is_not_shown_the_prompt() {
    let mut p = willing_peer(0xBB, LAYERS);
    p.trust_score = super::DELEGATE_MIN_TRUST - 0.01;
    let cands = vec![local_full_coverage(), p];
    assert!(super::delegation_target(&cands, &local_id(), LAYERS, true, MODEL_MB, 0.0).is_none());
}

/// Without a size for the model there is nothing to check a peer's room
/// against, and guessing is how the previous attempt went wrong.
#[test]
fn an_unknown_model_size_keeps_the_request_here() {
    let cands = vec![local_full_coverage(), willing_peer(0xBB, LAYERS)];
    assert!(super::delegation_target(&cands, &local_id(), LAYERS, true, 0, 0.0).is_none());
}

/// With no peer able to help, the local node keeps the request and answers
/// slowly. Answering slowly beats not answering — which is what excluding the
/// local node outright would produce.
#[test]
fn with_no_willing_peer_the_request_stays_local() {
    let cands = vec![local_full_coverage()];
    assert!(super::delegation_target(&cands, &local_id(), LAYERS, true, MODEL_MB, 0.0).is_none());
}

/// Never to ourselves, however the candidate list is ordered.
#[test]
fn the_local_node_is_never_its_own_delegate() {
    let mut me = local_full_coverage();
    me.reach = super::ReachTier::DirectMeasured;
    me.gpu_vram_available_mb = Some(80_000);
    let cands = vec![me];
    assert!(super::delegation_target(&cands, &local_id(), LAYERS, true, MODEL_MB, 0.0).is_none());
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
        super::delegation_target(&cands, &local_id(), LAYERS, true, MODEL_MB, 0.0).is_some(),
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
        super::delegation_target(&cands, &local_id(), LAYERS, true, MODEL_MB, 1.0).is_some(),
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
    assert!(super::delegation_target(&cands, &local_id(), LAYERS, true, MODEL_MB, 1.0).is_none());
}

/// With our own speed unknown there is nothing to compare against, so a peer
/// without a card is refused rather than guessed at.
#[test]
fn an_unknown_local_speed_refuses_a_processor_peer() {
    let mut peer = willing_peer(0xBB, LAYERS);
    peer.gpu_vram_available_mb = None;
    peer.est_tokens_per_sec = 999.0;
    let cands = vec![local_full_coverage(), peer];
    assert!(super::delegation_target(&cands, &local_id(), LAYERS, true, MODEL_MB, 0.0).is_none());
}

/// A card with room still wins without any speed comparison — it is a clear
/// improvement over our processor fallback whatever the advertised rates say.
#[test]
fn a_card_with_room_needs_no_speed_comparison() {
    let mut peer = willing_peer(0xBB, LAYERS);
    peer.est_tokens_per_sec = 0.0; // says nothing about its speed
    let cands = vec![local_full_coverage(), peer];
    assert!(super::delegation_target(&cands, &local_id(), LAYERS, true, MODEL_MB, 99.0).is_some());
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

    let cold = super::max_hostable_layers(Some(&cap), bytes_per_layer, false, 0);
    assert!(
        cold.is_some_and(|k| k < 32),
        "a COLD peer with 200 MB free cannot take a 3 GB model: {cold:?}"
    );

    let warm = super::max_hostable_layers(Some(&cap), bytes_per_layer, true, 0);
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
    );
    let long = super::max_hostable_layers(
        Some(&cap),
        bytes_per_layer,
        false,
        per_position_per_layer * 8_111,
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
        super::max_hostable_layers(Some(&cap), bytes_per_layer, true, 0),
        None
    );
    // 8,111 positions × 12 KB ≈ 99.6 MB per layer against ~909 MB usable → 9 layers.
    let capped = super::max_hostable_layers(
        Some(&cap),
        bytes_per_layer,
        true,
        per_position_per_layer * 8_111,
    );
    assert!(
        capped.is_some_and(|k| k < 24),
        "a warm card with 1 GB free cannot take 24 layers of an 8k prompt: {capped:?}"
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
    assert_eq!(super::max_hostable_layers(None, 1024, false, 0), None);

    let zeroed = capability_with_gpu(Some(0));
    assert_eq!(
        super::max_hostable_layers(Some(&zeroed), 1024, false, 0),
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

/// A peer priced at the shared unknown prior is not evidence of a faster
/// route: a route this node can price is never given up for one it cannot.
/// Advertised speed counts as knowing; an all-local chain means the search
/// agrees with the fast path.
#[test]
fn an_unpriced_peer_cannot_displace_the_processor_route() {
    let local = local_full_coverage();
    let mut peer = willing_peer(0xBB, LAYERS);
    assert_eq!(
        peer.est_tokens_per_sec, 0.0,
        "the fixture must start unpriced"
    );
    let remote_chain = vec![PipelineSegment {
        node_id: peer.node_id.clone(),
        shard_id: peer.shard_id.clone(),
        layer_range: (0, LAYERS),
    }];
    let cands = vec![local.clone(), peer.clone()];
    assert!(
        super::pipeline_may_replace_processor_route(&remote_chain, &cands, &local_id()).is_err(),
        "an unpriced peer must not take the request"
    );
    peer.est_tokens_per_sec = 20.0;
    let cands = vec![local.clone(), peer];
    assert!(
        super::pipeline_may_replace_processor_route(&remote_chain, &cands, &local_id()).is_ok(),
        "an advertised speed is a price"
    );
    let local_chain = vec![PipelineSegment {
        node_id: local.node_id.clone(),
        shard_id: local.shard_id.clone(),
        layer_range: (0, LAYERS),
    }];
    assert!(
        super::pipeline_may_replace_processor_route(&local_chain, &cands, &local_id()).is_err(),
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
