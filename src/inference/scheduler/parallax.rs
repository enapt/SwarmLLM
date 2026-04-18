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
//! Default off behind `InferenceConfig::parallax_routing`.

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

/// Fallback tokens/sec when a candidate hasn't gossiped an estimate.
/// Below this threshold (e.g. est_tokens_per_sec == 0) compute_ms contribution is 0,
/// making the DP fall back to pure latency + load as the cost.
const DEFAULT_TOKENS_PER_SEC: f32 = 0.0;

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
            whole_model_ms * (layers / 32.0)
        } else {
            DEFAULT_TOKENS_PER_SEC
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
    let mut vertices: Vec<Vertex> = Vec::new();
    for (cand_idx, c) in candidates.iter().enumerate() {
        for &range in &c.available_ranges {
            if range.0 >= range.1 {
                continue;
            }
            if range.1 > num_layers {
                // A range that exceeds the model is still usable up to num_layers.
                let capped = (range.0, num_layers);
                if capped.0 >= capped.1 {
                    continue;
                }
                let cost = vertex_cost(c, capped, local_node_id).total();
                vertices.push(Vertex {
                    cand_idx,
                    range: capped,
                    cost_ms: cost,
                });
                continue;
            }
            let cost = vertex_cost(c, range, local_node_id).total();
            vertices.push(Vertex {
                cand_idx,
                range,
                cost_ms: cost,
            });
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

    fn cand_with_obs(mut c: NodeCandidate, observed_ms_per_layer: f32) -> NodeCandidate {
        c.observed_latency_ms_per_layer = Some(observed_ms_per_layer);
        c
    }

    #[test]
    fn single_node_covers_all() {
        let local = NodeId([1u8; 32]);
        let cands = vec![cand(1, vec![(0, 32)], 0, 0.0, true, true, 0.0)];
        let segs = route_shortest_path(32, &cands, &local, false).unwrap();
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
        let segs = route_shortest_path(32, &cands, &local, false).unwrap();
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
        let segs = route_shortest_path(32, &cands, &local, false).unwrap();
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
        let segs = route_shortest_path(32, &cands, &local, true).unwrap();
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
        let err = route_shortest_path(32, &cands, &local, false).unwrap_err();
        assert!(matches!(err, SwarmError::PipelineError(_)));
    }

    #[test]
    fn no_sink_errors() {
        let local = NodeId([1u8; 32]);
        let cands = vec![cand(1, vec![(0, 16)], 0, 0.0, true, false, 0.0)];
        let err = route_shortest_path(32, &cands, &local, false).unwrap_err();
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
        let err = route_shortest_path(32, &cands, &local, false).unwrap_err();
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
        let segs = route_shortest_path(32, &cands, &local, false).unwrap();
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
        let segs = route_shortest_path(32, &cands, &local, false).unwrap();
        assert_eq!(segs.len(), 3);
        assert_eq!(segs[1].node_id, NodeId([2u8; 32]));
        assert_eq!(segs[2].node_id, NodeId([3u8; 32]));
    }
}
