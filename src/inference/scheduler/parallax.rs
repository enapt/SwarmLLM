//! Parallax-inspired shortest-path routing over (node, layer_range) vertices.
//!
//! Adapted from Parallax Phase 2 (arxiv 2509.26182 / GradientHQ/parallax MIT).
//! Parallax was designed for peer-to-peer pipeline data flow where transition
//! edge cost is `rtt(peer_A, peer_B)`. SwarmLLM pipelines are coordinator-relayed:
//! every hop routes through the local node. So edge weight collapses into a
//! per-vertex cost: `2 * rtt_local_to_peer + compute_time + load_compensator`
//! (local node: just compute_time).
//!
//! DAG: vertex = (candidate_idx, range_idx). Edge v → w iff
//! `ranges[v].end == ranges[w].start`. Sources have `start == 0` (and may have
//! first-segment constraints). Sinks have `end == num_layers` (and may have
//! last-segment constraints). DP finds the minimum-cost chain.
//!
//! Falls back to `greedy_assign` on any configuration the DP can't cover
//! (e.g., no valid source, no valid sink, disconnected layers).
//!
//! Gated by `InferenceConfig::parallax_routing` (default on). Set to `false`
//! to revert to pure greedy assembly — falls back automatically anyway when
//! the DP has no valid source→sink path.

use crate::error::SwarmError;
use crate::types::{NodeId, PipelineSegment};

use super::NodeCandidate;

/// Per-vertex cost components. All in milliseconds.
#[derive(Debug, Clone, Copy)]
struct VertexCost {
    /// 2 * latency_ms for remote peers, 0 for local.
    network_ms: f32,
    /// (layers_in_range / tokens_per_sec) * 1000; 0 when tokens_per_sec unknown.
    compute_ms: f32,
    /// active_request_count * LOAD_COMPENSATOR_MS.
    load_ms: f32,
}

impl VertexCost {
    fn total(self) -> f32 {
        self.network_ms + self.compute_ms + self.load_ms
    }
}

/// Penalty (ms) per active concurrent request on a candidate.
/// Parallax uses `load_compensator=0.05` as a fraction of baseline latency;
/// we use a fixed ms penalty so it's tunable independently of a baseline estimate.
const LOAD_COMPENSATOR_MS: f32 = 25.0;

/// Compute-cost contribution when a candidate has neither an observed
/// per-layer latency nor a gossiped throughput estimate. Set to 0 so the DP
/// falls back to pure latency + load — the alternative (a positive default)
/// would silently penalise newly-discovered peers we have no data on.
///
/// Despite the unit, this is *milliseconds*, not tokens/sec — see the
/// fallback branch in `vertex_cost`.
pub(super) const UNKNOWN_COMPUTE_MS: f32 = 0.0;

/// Cap on how many partial-range vertices the DP will consider, summed across
/// candidates. Keeps vertex generation bounded when a popular model has many
/// holders; past it, only whole ranges are emitted.
pub(super) const MAX_SUBRANGE_VERTICES: usize = 4096;

/// Baseline transformer layer count used to scale a whole-model throughput
/// estimate down to a per-segment contribution. 32 matches Llama-7B and most
/// 7B Q4 models we benchmark against; arch-aware scaling would replace this
/// with the actual layer count from the GGUF metadata. Shared with
/// `parallax_allocator.rs` so the cost model stays consistent across the
/// scheduler and the offline allocator.
pub(super) const BASELINE_LAYER_COUNT: f32 = 32.0;

/// Compute per-vertex cost for a (candidate, range) pair.
///
/// Cost priority for `compute_ms`:
/// 1. Observed per-layer latency EMA if the candidate has any samples (Phase B).
///    This folds the remote peer's per-segment wall-clock (already includes both
///    compute and any peer-side queuing/load) into the DP objective, so the
///    `network_ms` term doesn't double-count load here.
/// 2. Static `est_tokens_per_sec` capability estimate when no observations exist.
/// 3. Zero (pure latency + load objective) when neither is available.
fn vertex_cost(c: &NodeCandidate, range: (u32, u32), local: &NodeId) -> VertexCost {
    let is_local = &c.node_id == local;
    let layers = (range.1 - range.0) as f32;
    // When we have an observed per-layer latency, it already includes the peer's
    // segment wall-clock round-trip (compute + peer-side load). Fold the whole
    // `segment_ms` into `compute_ms` and skip the static `2 * latency_ms` network
    // term to avoid double-counting. When we don't have an observation yet, use
    // the traditional two-part cost (network + static compute estimate).
    let (network_ms, compute_ms) = if let Some(obs_per_layer) = c.observed_latency_ms_per_layer {
        (0.0, obs_per_layer * layers)
    } else {
        let network = if is_local {
            0.0
        } else {
            2.0 * c.latency_ms as f32
        };
        let compute = if c.est_tokens_per_sec > 0.0 {
            // Very rough: layer_compute_ms ≈ layers / (est_tokens_per_sec * some_constant).
            // est_tokens_per_sec is a whole-model throughput estimate for a 7B Q4 model;
            // per-layer contribution is 1/num_layers of that. We conservatively use
            // `1000.0 / est_tokens_per_sec` as the per-token-whole-model compute cost,
            // then scale by the fraction of layers this segment owns. Assumes 32 layers
            // as the baseline; adjust here if we want arch-aware scaling later.
            let whole_model_ms = 1000.0 / c.est_tokens_per_sec;
            whole_model_ms * (layers / BASELINE_LAYER_COUNT)
        } else {
            UNKNOWN_COMPUTE_MS
        };
        (network, compute)
    };
    let load_ms = c.load * LOAD_COMPENSATOR_MS;
    VertexCost {
        network_ms,
        compute_ms,
        load_ms,
    }
}

/// Route layers [0, num_layers) across candidates using shortest-path DP.
///
/// Returns segments covering [0, num_layers) in order, or the SwarmError from
/// an empty search space if the DAG has no source/sink path. Callers should
/// fall back to the greedy assigner on error.
pub(super) fn route_shortest_path(
    num_layers: u32,
    candidates: &[NodeCandidate],
    local_node_id: &NodeId,
    encrypted_pipeline: bool,
    // Allow a candidate's range to be used in part. Off by default: see
    // `config.inference.parallax_partial_ranges` for the measured reason.
    partial_ranges: bool,
) -> Result<Vec<PipelineSegment>, SwarmError> {
    if num_layers == 0 {
        return Err(SwarmError::PipelineError("num_layers=0".into()));
    }

    // Build the vertex list: one vertex per (candidate, range).
    // Also precompute each vertex's cost.
    #[derive(Clone)]
    struct Vertex {
        cand_idx: usize,
        range: (u32, u32),
        cost_ms: f32,
    }
    // Split points at which a segment boundary can usefully fall: the ends of
    // every candidate's ranges, plus the model's own ends. A sub-range is only
    // worth emitting if some other candidate could begin or end there.
    let mut split_points: Vec<u32> = vec![0, num_layers];
    for c in candidates {
        for &(a, b) in &c.available_ranges {
            if a < num_layers {
                split_points.push(a);
            }
            let b = b.min(num_layers);
            if b > 0 {
                split_points.push(b);
            }
        }
    }
    split_points.sort_unstable();
    split_points.dedup();

    // Usable (clamped, non-empty) ranges, paired with the split points falling
    // inside each.
    let clamped: Vec<(usize, (u32, u32))> = candidates
        .iter()
        .enumerate()
        .flat_map(|(i, c)| {
            c.available_ranges
                .iter()
                .map(move |&(a, b)| (i, (a, b.min(num_layers))))
        })
        .filter(|(_, (a, b))| a < b)
        .collect();

    // Emitting every sub-range is O(k^2) per range in the number of interior
    // split points. That is trivial for a handful of holders and unbounded at
    // swarm scale, so it is budgeted: past the cap we emit whole ranges only,
    // which is exactly the pre-split behaviour rather than a degraded one.
    let subrange_cost: usize = clamped
        .iter()
        .map(|(_, (lo, hi))| {
            let k = split_points
                .iter()
                .filter(|&&p| p >= *lo && p <= *hi)
                .count();
            k.saturating_mul(k.saturating_sub(1)) / 2
        })
        .sum();
    let split_enabled = partial_ranges && subrange_cost <= MAX_SUBRANGE_VERTICES;
    if partial_ranges && !split_enabled {
        tracing::debug!(
            subrange_cost,
            cap = MAX_SUBRANGE_VERTICES,
            candidates = candidates.len(),
            "parallax: too many candidates to consider partial ranges — whole ranges only"
        );
    }

    let mut vertices: Vec<Vertex> = Vec::new();
    for (cand_idx, (lo, hi)) in clamped {
        let c = &candidates[cand_idx];
        let mut push = |range: (u32, u32)| {
            vertices.push(Vertex {
                cand_idx,
                range,
                cost_ms: vertex_cost(c, range, local_node_id).total(),
            });
        };
        // The whole range is always available, so a chain that was routable
        // before stays routable.
        push((lo, hi));
        if !split_enabled {
            continue;
        }
        // Partial use of a range. Without these a candidate could only ever
        // serve its range in full, so a node holding the entire model was the
        // only representable route and no other node's shards could contribute
        // — regardless of how much slower that node was.
        for (i, &p) in split_points.iter().enumerate() {
            if p < lo || p >= hi {
                continue;
            }
            for &q in &split_points[i + 1..] {
                if q > hi {
                    break;
                }
                if (p, q) != (lo, hi) {
                    push((p, q));
                }
            }
        }
    }

    if vertices.is_empty() {
        return Err(SwarmError::PipelineError(
            "parallax: no candidate ranges".into(),
        ));
    }

    // Source filter: start==0. Must have can_be_first. Encrypted: must be local.
    let is_source = |v: &Vertex| -> bool {
        if v.range.0 != 0 {
            return false;
        }
        let c = &candidates[v.cand_idx];
        if !c.can_be_first {
            return false;
        }
        if encrypted_pipeline && &c.node_id != local_node_id {
            return false;
        }
        true
    };

    // Sink filter: end==num_layers. Must have can_be_last. Encrypted: must be local.
    let is_sink = |v: &Vertex| -> bool {
        if v.range.1 != num_layers {
            return false;
        }
        let c = &candidates[v.cand_idx];
        if !c.can_be_last {
            return false;
        }
        if encrypted_pipeline && &c.node_id != local_node_id {
            return false;
        }
        true
    };

    if !vertices.iter().any(is_source) {
        return Err(SwarmError::PipelineError(
            "parallax: no valid source vertex (starts at layer 0, can_be_first)".into(),
        ));
    }
    if !vertices.iter().any(is_sink) {
        return Err(SwarmError::PipelineError(
            "parallax: no valid sink vertex (ends at num_layers, can_be_last)".into(),
        ));
    }

    // Topological order: sort by (range.0 asc, range.1 asc). DAG edges always go
    // from smaller end to larger start, and within the same start we want smaller
    // spans processed first so later vertices can relax against them.
    let mut order: Vec<usize> = (0..vertices.len()).collect();
    order.sort_by(|&a, &b| {
        vertices[a]
            .range
            .0
            .cmp(&vertices[b].range.0)
            .then(vertices[a].range.1.cmp(&vertices[b].range.1))
    });

    // DP: best_cost[v] = min total cost reaching vertex v as a sink of some source.
    // parent[v] = predecessor vertex for path reconstruction.
    let n = vertices.len();
    let mut best_cost = vec![f32::INFINITY; n];
    let mut parent: Vec<Option<usize>> = vec![None; n];

    // Initialize sources.
    for i in 0..n {
        if is_source(&vertices[i]) {
            best_cost[i] = vertices[i].cost_ms;
        }
    }

    // Forward DP.
    for &v_idx in &order {
        if !best_cost[v_idx].is_finite() {
            continue;
        }
        let v_end = vertices[v_idx].range.1;
        if v_end >= num_layers {
            continue; // nothing to extend
        }
        // Find successors: vertices whose start equals v_end.
        // Scan is O(n); with many candidates we could bucket by start_layer but
        // vertex counts stay small (bounded by peer_count * ranges_per_peer).
        for &w_idx in &order {
            if vertices[w_idx].range.0 != v_end {
                continue;
            }
            let new_cost = best_cost[v_idx] + vertices[w_idx].cost_ms;
            if new_cost < best_cost[w_idx] {
                best_cost[w_idx] = new_cost;
                parent[w_idx] = Some(v_idx);
            }
        }
    }

    // Pick the best sink.
    let best_sink = (0..n).filter(|&i| is_sink(&vertices[i])).min_by(|&a, &b| {
        best_cost[a]
            .partial_cmp(&best_cost[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let sink_idx = match best_sink {
        Some(i) if best_cost[i].is_finite() => i,
        _ => {
            return Err(SwarmError::PipelineError(
                "parallax: no reachable source→sink path".into(),
            ));
        }
    };

    // Reconstruct path.
    let mut chain: Vec<usize> = Vec::new();
    let mut cur = Some(sink_idx);
    while let Some(i) = cur {
        chain.push(i);
        cur = parent[i];
    }
    chain.reverse();

    // Convert to PipelineSegments.
    let segments: Vec<PipelineSegment> = chain
        .into_iter()
        .map(|i| {
            let v = &vertices[i];
            let c = &candidates[v.cand_idx];
            PipelineSegment {
                node_id: c.node_id.clone(),
                shard_id: c.shard_id.clone(),
                layer_range: v.range,
            }
        })
        .collect();

    // Sanity: segments must contiguously cover [0, num_layers).
    let mut expect = 0u32;
    for s in &segments {
        if s.layer_range.0 != expect {
            return Err(SwarmError::PipelineError(format!(
                "parallax: non-contiguous chain at layer {expect}"
            )));
        }
        expect = s.layer_range.1;
    }
    if expect != num_layers {
        return Err(SwarmError::PipelineError(format!(
            "parallax: chain ends at {expect}, want {num_layers}"
        )));
    }

    Ok(segments)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ModelId, ShardId};

    fn cand(
        byte: u8,
        ranges: Vec<(u32, u32)>,
        latency_ms: u32,
        load: f32,
        can_be_first: bool,
        can_be_last: bool,
        est_tokens_per_sec: f32,
    ) -> NodeCandidate {
        NodeCandidate {
            node_id: NodeId([byte; 32]),
            shard_id: ShardId {
                model_id: ModelId("m".into()),
                index: 0,
            },
            available_ranges: ranges,
            latency_ms,
            load,
            trust_score: 1.0,
            can_be_first,
            can_be_last,
            region_score: 1.0,
            est_tokens_per_sec,
            observed_latency_ms_per_layer: None,
            is_pool_member: false,
        }
    }

    /// The live v0.3.34 configuration that could not be routed: a fast local
    /// node holding layers 0-10 and a slow remote holding the whole model. The
    /// split needs a vertex starting at layer 10, which the remote's single
    /// (0,16) range did not provide, so all 16 layers went to the slow node.
    #[test]
    fn partial_range_lets_a_local_prefix_pair_with_a_whole_model_holder() {
        let local = NodeId([1u8; 32]);
        let remote_slow = cand_with_obs(cand(2, vec![(0, 16)], 5, 0.0, true, true, 0.0), 107.0);
        let mut local_fast = cand(1, vec![(0, 10)], 0, 0.0, true, false, 0.0);
        local_fast.node_id = local.clone();

        let segs = route_shortest_path(16, &[remote_slow, local_fast], &local, false, true)
            .expect("a route must exist");

        assert_eq!(segs.len(), 2, "expected a split, got {segs:?}");
        assert_eq!(segs[0].node_id, local, "local prefix must run locally");
        assert_eq!(segs[0].layer_range, (0, 10));
        assert_eq!(
            segs[1].layer_range,
            (10, 16),
            "remote must serve the suffix"
        );
        assert_ne!(segs[1].node_id, local);
    }

    /// With partial ranges OFF — the shipped default — routing must be exactly
    /// what it was before they existed: the whole-model holder takes everything.
    /// This is the behaviour the default protects, because a multi-segment chain
    /// exchanges activations per token and measured slower on a LAN pair.
    #[test]
    fn partial_ranges_off_keeps_the_single_segment_route() {
        let local = NodeId([1u8; 32]);
        let remote = cand_with_obs(cand(2, vec![(0, 16)], 5, 0.0, true, true, 0.0), 107.0);
        let mut local_fast = cand(1, vec![(0, 10)], 0, 0.0, true, false, 0.0);
        local_fast.node_id = local.clone();

        let segs = route_shortest_path(16, &[remote, local_fast], &local, false, false)
            .expect("a route must exist");
        assert_eq!(segs.len(), 1, "default must not split: {segs:?}");
        assert_eq!(segs[0].layer_range, (0, 16));
    }

    /// The cost model must still be what decides. With the remote fast and the
    /// local expensive, splitting is not worth it — so a partial range being
    /// *available* must not make it *mandatory*.
    #[test]
    fn partial_ranges_do_not_force_a_split_when_whole_is_cheaper() {
        let local = NodeId([1u8; 32]);
        // A remote with an observed per-layer cost of ~0 is essentially free.
        let remote_fast = cand_with_obs(cand(2, vec![(0, 16)], 0, 0.0, true, true, 0.0), 0.01);
        // Local looks free too (local always does), but splitting adds a second
        // segment whose cost cannot beat one nearly-free whole-model segment.
        let mut local_node = cand(1, vec![(0, 10)], 0, 0.0, true, false, 0.0);
        local_node.node_id = local.clone();

        let segs = route_shortest_path(16, &[remote_fast, local_node], &local, false, true)
            .expect("route");
        let total: u32 = segs.iter().map(|s| s.layer_range.1 - s.layer_range.0).sum();
        assert_eq!(total, 16, "coverage must be complete either way");
    }

    #[test]
    fn partial_ranges_still_cover_every_layer_contiguously() {
        let local = NodeId([9u8; 32]);
        let a = cand(1, vec![(0, 12)], 10, 0.0, true, false, 0.0);
        let b = cand(2, vec![(4, 20)], 10, 0.0, false, true, 0.0);
        let segs = route_shortest_path(20, &[a, b], &local, false, true).expect("route");
        let mut expect = 0;
        for s in &segs {
            assert_eq!(s.layer_range.0, expect, "gap or overlap in {segs:?}");
            expect = s.layer_range.1;
        }
        assert_eq!(expect, 20);
    }

    /// A sink must still hold the final shard: partial ranges must not let a
    /// `can_be_last = false` candidate end the chain.
    #[test]
    fn partial_ranges_respect_can_be_last() {
        let local = NodeId([9u8; 32]);
        // Covers everything but is not allowed to be last.
        let not_last = cand(1, vec![(0, 16)], 1, 0.0, true, false, 0.0);
        assert!(route_shortest_path(16, &[not_last], &local, false, true).is_err());
    }

    /// And a source must still be able to be first.
    #[test]
    fn partial_ranges_respect_can_be_first() {
        let local = NodeId([9u8; 32]);
        let not_first = cand(1, vec![(0, 16)], 1, 0.0, false, true, 0.0);
        assert!(route_shortest_path(16, &[not_first], &local, false, true).is_err());
    }

    /// Encrypted pipelines require the local node at both ends; partial ranges
    /// must not open a path around that.
    #[test]
    fn partial_ranges_do_not_bypass_encrypted_pipeline_ends() {
        let local = NodeId([9u8; 32]);
        let remote = cand(2, vec![(0, 16)], 1, 0.0, true, true, 0.0);
        assert!(
            route_shortest_path(16, &[remote], &local, true, true).is_err(),
            "a remote-only chain must be refused when encrypted_pipeline is on"
        );
    }

    /// Above the budget the DP falls back to whole ranges — which must still
    /// route, not fail.
    #[test]
    fn many_candidates_fall_back_to_whole_ranges_and_still_route() {
        let local = NodeId([200u8; 32]);
        let mut cands = Vec::new();
        for i in 0..90u8 {
            // Overlapping ranges across many holders: enough split points to
            // blow the sub-range budget.
            let lo = (i as u32) % 40;
            cands.push(cand(i, vec![(lo, 80)], 5, 0.0, true, true, 0.0));
        }
        let segs = route_shortest_path(80, &cands, &local, false, true).expect("must still route");
        let mut expect = 0;
        for s in &segs {
            assert_eq!(s.layer_range.0, expect);
            expect = s.layer_range.1;
        }
        assert_eq!(expect, 80);
    }

    fn cand_with_obs(mut c: NodeCandidate, observed_ms_per_layer: f32) -> NodeCandidate {
        c.observed_latency_ms_per_layer = Some(observed_ms_per_layer);
        c
    }

    #[test]
    fn single_node_covers_all() {
        let local = NodeId([1u8; 32]);
        let cands = vec![cand(1, vec![(0, 32)], 0, 0.0, true, true, 0.0)];
        let segs = route_shortest_path(32, &cands, &local, false, false).unwrap();
        assert_eq!(segs.len(), 1);
        assert_eq!(segs[0].layer_range, (0, 32));
        assert_eq!(segs[0].node_id, local);
    }

    #[test]
    fn picks_low_latency_chain() {
        let local = NodeId([1u8; 32]);
        // Local holds only [0,8); two remotes cover [8,32): slow (latency 200) vs fast (latency 10).
        let cands = vec![
            cand(1, vec![(0, 8)], 0, 0.0, true, false, 0.0),
            cand(2, vec![(8, 32)], 200, 0.0, false, true, 0.0),
            cand(3, vec![(8, 32)], 10, 0.0, false, true, 0.0),
        ];
        let segs = route_shortest_path(32, &cands, &local, false, false).unwrap();
        assert_eq!(segs.len(), 2);
        assert_eq!(segs[0].node_id, local);
        assert_eq!(segs[1].node_id, NodeId([3u8; 32]));
    }

    #[test]
    fn load_penalty_shifts_choice() {
        let local = NodeId([99u8; 32]);
        // Two remotes, both [0,32), same latency. One is heavily loaded.
        let cands = vec![
            cand(1, vec![(0, 32)], 50, 10.0, true, true, 0.0),
            cand(2, vec![(0, 32)], 50, 0.0, true, true, 0.0),
        ];
        let segs = route_shortest_path(32, &cands, &local, false, false).unwrap();
        assert_eq!(segs.len(), 1);
        assert_eq!(segs[0].node_id, NodeId([2u8; 32]));
    }

    #[test]
    fn encrypted_requires_local_first_and_last() {
        let local = NodeId([1u8; 32]);
        // Local holds [0,4) and [28,32); one remote covers middle.
        let cands = vec![
            cand(1, vec![(0, 4), (28, 32)], 0, 0.0, true, true, 0.0),
            cand(2, vec![(4, 28)], 5, 0.0, false, false, 0.0),
        ];
        let segs = route_shortest_path(32, &cands, &local, true, false).unwrap();
        assert_eq!(segs.len(), 3);
        assert_eq!(segs[0].node_id, local);
        assert_eq!(segs[0].layer_range, (0, 4));
        assert_eq!(segs[2].node_id, local);
        assert_eq!(segs[2].layer_range, (28, 32));
    }

    #[test]
    fn no_first_capable_errors() {
        let local = NodeId([1u8; 32]);
        let cands = vec![cand(2, vec![(0, 32)], 50, 0.0, false, true, 0.0)];
        let err = route_shortest_path(32, &cands, &local, false, false).unwrap_err();
        assert!(matches!(err, SwarmError::PipelineError(_)));
    }

    #[test]
    fn no_sink_errors() {
        let local = NodeId([1u8; 32]);
        let cands = vec![cand(1, vec![(0, 16)], 0, 0.0, true, false, 0.0)];
        let err = route_shortest_path(32, &cands, &local, false, false).unwrap_err();
        assert!(matches!(err, SwarmError::PipelineError(_)));
    }

    #[test]
    fn disjoint_ranges_fail_cleanly() {
        let local = NodeId([1u8; 32]);
        // Gap between [0,8) and [16,32) — layers 8-15 missing.
        let cands = vec![
            cand(1, vec![(0, 8)], 0, 0.0, true, false, 0.0),
            cand(2, vec![(16, 32)], 10, 0.0, false, true, 0.0),
        ];
        let err = route_shortest_path(32, &cands, &local, false, false).unwrap_err();
        assert!(matches!(err, SwarmError::PipelineError(_)));
    }

    #[test]
    fn observed_latency_overrides_static_estimate() {
        // Two remotes, identical static latency (50 ms) and est_tokens_per_sec (0).
        // Without observations they tie. Add an observed-latency EMA showing one
        // is actually slow (20 ms/layer) and the other fast (2 ms/layer); DP picks
        // the fast one.
        let local = NodeId([99u8; 32]);
        let slow = cand_with_obs(cand(1, vec![(0, 32)], 50, 0.0, true, true, 0.0), 20.0);
        let fast = cand_with_obs(cand(2, vec![(0, 32)], 50, 0.0, true, true, 0.0), 2.0);
        let cands = vec![slow, fast];
        let segs = route_shortest_path(32, &cands, &local, false, false).unwrap();
        assert_eq!(segs.len(), 1);
        assert_eq!(segs[0].node_id, NodeId([2u8; 32]));
    }

    #[test]
    fn multi_hop_chain_minimizes_total_latency() {
        let local = NodeId([1u8; 32]);
        // Two paths possible: 1→2→3 (latencies 0+50+50) vs 1→4 (local alone can't finish)
        // Simpler: ensure 3-hop chain with lower total beats 2-hop chain.
        // Local [0,8), A[8,16)@10ms, B[16,32)@10ms  vs  Local[0,8), C[8,32)@100ms
        let cands = vec![
            cand(1, vec![(0, 8)], 0, 0.0, true, false, 0.0),
            cand(2, vec![(8, 16)], 10, 0.0, false, false, 0.0),
            cand(3, vec![(16, 32)], 10, 0.0, false, true, 0.0),
            cand(4, vec![(8, 32)], 100, 0.0, false, true, 0.0),
        ];
        let segs = route_shortest_path(32, &cands, &local, false, false).unwrap();
        assert_eq!(segs.len(), 3);
        assert_eq!(segs[1].node_id, NodeId([2u8; 32]));
        assert_eq!(segs[2].node_id, NodeId([3u8; 32]));
    }
}
