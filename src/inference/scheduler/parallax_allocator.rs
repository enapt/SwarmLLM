//! Parallax offline layer allocator (Item 16 Phase C).
//!
//! Ports the two-phase idea from Parallax's
//! `DynamicProgrammingLayerAllocator`: given a cluster snapshot (peers with
//! layer capacity + compute throughput + local RTT), propose how shards
//! *should* be distributed to form one or more balanced pipelines that cover
//! all of the model's layers.
//!
//! v1 uses a simpler greedy multi-pipeline packer rather than Parallax's full
//! DP — the DP's state space (`(i, open_residuals, finished_pipes)`) expands
//! fast when peers are heterogeneous, and a greedy with a `Z(k) = k² / s*(k)`
//! objective captures most of the win on our typical cluster sizes (tens of
//! peers, not hundreds).
//!
//! Output is a **recommendation** only in v1 — we log it and expose it via
//! the scheduler, but `AutoShardManager` / `ShardRebalancer` don't auto-act
//! on it yet. That wiring is Phase C.2.

use std::collections::HashMap;

use crate::types::NodeId;

/// A peer's capacity and performance profile for allocation planning.
#[derive(Debug, Clone)]
pub struct PeerCapacity {
    /// Peer identity.
    pub node_id: NodeId,
    /// How many transformer layers this peer can host in VRAM for the target
    /// model. Caller computes as `vram_bytes / avg_bytes_per_layer`. 0 means
    /// the peer can't host any layers (still counted in the pool but skipped).
    pub layer_capacity: u32,
    /// Compute throughput proxy. Higher = faster. Matches the
    /// `est_tokens_per_sec_7b` signal we gossip in `NodeCapability`. 0 means
    /// unknown (allocator treats as average).
    pub tokens_per_sec: f32,
    /// Round-trip latency from the local coordinator to this peer in ms.
    /// Local node: 0. Unknown: a conservative default (caller's choice).
    pub latency_ms: u32,
}

/// One contiguous layer range assigned to one peer in a recommended pipeline.
#[derive(Debug, Clone, PartialEq)]
pub struct AllocationSegment {
    pub node_id: NodeId,
    pub layer_range: (u32, u32),
}

/// A complete recommended pipeline covering `[0, num_layers)`.
#[derive(Debug, Clone)]
pub struct PipelineAllocation {
    pub segments: Vec<AllocationSegment>,
    /// Estimated end-to-end pipeline latency (ms) under this allocation.
    /// Sum of per-segment `(2 * latency + compute_ms) + load_penalty` — same
    /// cost model family the Phase A routing DP uses so Phase C and Phase A
    /// see the world the same way.
    pub est_pipeline_latency_ms: f32,
}

/// Full allocator recommendation: one or more pipelines + a throughput score.
#[derive(Debug, Clone)]
pub struct AllocationPlan {
    /// Recommended pipelines. `len()` = degree of parallelism (k).
    pub pipelines: Vec<PipelineAllocation>,
    /// Parallax's `Z(k) = k² / s*(k)` throughput objective, where k = number
    /// of parallel pipelines and `s*(k)` = average stages per pipeline. Higher
    /// is better — reflects the tradeoff between parallelism (k) and latency
    /// (stages/pipeline).
    pub throughput_score: f32,
}

/// Recommend a layer allocation across peers. Tries every feasible pipeline
/// count `k` in `[1..=max_pipelines]` and picks the one that maximises
/// `Z(k) = k² / avg_stages`.
///
/// Returns `None` if no feasible pipeline can be formed (total cluster
/// capacity < num_layers, or there's no way to cover `[0, num_layers)`).
pub fn recommend_allocation(
    peers: &[PeerCapacity],
    num_layers: u32,
    max_pipelines: u32,
) -> Option<AllocationPlan> {
    if num_layers == 0 || peers.is_empty() {
        return None;
    }
    let total_capacity: u64 = peers.iter().map(|p| p.layer_capacity as u64).sum();
    if total_capacity < num_layers as u64 {
        return None;
    }
    // Cap `k` by the theoretical feasibility: total_capacity / num_layers is
    // the highest `k` where a full-coverage allocation across k pipelines is
    // possible. Don't cap by peer count — a single peer with large capacity
    // can legitimately host multiple parallel pipelines (logical constructs;
    // compute contention is priced into tokens_per_sec).
    let feasible_k = (total_capacity / num_layers.max(1) as u64).min(u32::MAX as u64) as u32;
    let max_k = max_pipelines.min(feasible_k).max(1);

    let mut best: Option<AllocationPlan> = None;
    for k in 1..=max_k {
        if let Some(plan) = pack_k_pipelines(peers, num_layers, k) {
            let replace = match &best {
                Some(cur) => plan.throughput_score > cur.throughput_score,
                None => true,
            };
            if replace {
                best = Some(plan);
            }
        }
    }
    best
}

/// Try to pack `k` pipelines over the peer set. Maintains a remaining-capacity
/// budget per peer; for each pipeline, greedily walks peers in priority order
/// (fastest-first, low-latency tie-break) and takes as many layers as fit.
/// Requires `total_capacity >= k * num_layers` to succeed.
fn pack_k_pipelines(peers: &[PeerCapacity], num_layers: u32, k: u32) -> Option<AllocationPlan> {
    if k == 0 {
        return None;
    }
    // Sort peers: fastest first (high tokens_per_sec), tie-break on lowest latency.
    let mut ordered: Vec<&PeerCapacity> = peers.iter().filter(|p| p.layer_capacity > 0).collect();
    ordered.sort_by(|a, b| {
        b.tokens_per_sec
            .partial_cmp(&a.tokens_per_sec)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.latency_ms.cmp(&b.latency_ms))
    });
    if ordered.is_empty() {
        return None;
    }

    let total_capacity: u64 = ordered.iter().map(|p| p.layer_capacity as u64).sum();
    if total_capacity < (k as u64) * (num_layers as u64) {
        return None;
    }

    let mut remaining: HashMap<NodeId, u32> = HashMap::new();
    for p in &ordered {
        remaining.insert(p.node_id.clone(), p.layer_capacity);
    }

    // Greedily build k pipelines in peer-sorted order.
    let mut pipelines: Vec<PipelineAllocation> = Vec::with_capacity(k as usize);
    for _pipe_idx in 0..k {
        let mut assigned: Vec<AllocationSegment> = Vec::new();
        let mut est_latency: f32 = 0.0;
        let mut layer_cursor: u32 = 0;
        for p in &ordered {
            if layer_cursor >= num_layers {
                break;
            }
            let avail = remaining.get(&p.node_id).copied().unwrap_or(0);
            if avail == 0 {
                continue;
            }
            let remaining_model_layers = num_layers - layer_cursor;
            let take = avail.min(remaining_model_layers);
            if take == 0 {
                continue;
            }
            let range = (layer_cursor, layer_cursor + take);
            // Cost: 2 * rtt + compute_ms. Local node (latency 0) has no RTT term.
            let net_ms = 2.0 * p.latency_ms as f32;
            let compute_ms = if p.tokens_per_sec > 0.0 {
                (1000.0 / p.tokens_per_sec) * (take as f32 / 32.0)
            } else {
                0.0
            };
            est_latency += net_ms + compute_ms;
            assigned.push(AllocationSegment {
                node_id: p.node_id.clone(),
                layer_range: range,
            });
            if let Some(slot) = remaining.get_mut(&p.node_id) {
                *slot -= take;
            }
            layer_cursor += take;
        }
        if layer_cursor < num_layers {
            return None;
        }
        pipelines.push(PipelineAllocation {
            segments: assigned,
            est_pipeline_latency_ms: est_latency,
        });
    }

    // Parallax's Z(k) = k^2 / s*(k)  — maximise parallelism relative to
    // average stages per pipeline.
    let avg_stages: f32 = pipelines
        .iter()
        .map(|p| p.segments.len() as f32)
        .sum::<f32>()
        / pipelines.len() as f32;
    let throughput_score = if avg_stages > 0.0 {
        (k as f32).powi(2) / avg_stages
    } else {
        0.0
    };

    Some(AllocationPlan {
        pipelines,
        throughput_score,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn peer(byte: u8, layer_capacity: u32, tps: f32, latency_ms: u32) -> PeerCapacity {
        PeerCapacity {
            node_id: NodeId([byte; 32]),
            layer_capacity,
            tokens_per_sec: tps,
            latency_ms,
        }
    }

    #[test]
    fn single_node_full_capacity() {
        let peers = vec![peer(1, 32, 50.0, 0)];
        let plan = recommend_allocation(&peers, 32, 4).unwrap();
        // One pipeline, one segment covering everything.
        assert_eq!(plan.pipelines.len(), 1);
        assert_eq!(plan.pipelines[0].segments.len(), 1);
        assert_eq!(plan.pipelines[0].segments[0].layer_range, (0, 32));
    }

    #[test]
    fn balanced_cluster_prefers_more_pipelines() {
        // Three identical peers, each with enough capacity for the full model.
        // Allocator should prefer k=3 (three parallel pipelines) because
        // Z(k) = k^2 / 1 grows with k.
        let peers = vec![
            peer(1, 32, 50.0, 10),
            peer(2, 32, 50.0, 10),
            peer(3, 32, 50.0, 10),
        ];
        let plan = recommend_allocation(&peers, 32, 4).unwrap();
        assert_eq!(plan.pipelines.len(), 3);
        for pipe in &plan.pipelines {
            assert_eq!(pipe.segments.len(), 1);
            assert_eq!(pipe.segments[0].layer_range, (0, 32));
        }
    }

    #[test]
    fn heterogeneous_peers_fill_greedy_by_throughput() {
        // Fast peer with half the layers + slow peer with half. k=1 pipeline.
        let peers = vec![peer(1, 16, 100.0, 10), peer(2, 16, 10.0, 200)];
        let plan = recommend_allocation(&peers, 32, 1).unwrap();
        assert_eq!(plan.pipelines.len(), 1);
        let p = &plan.pipelines[0];
        assert_eq!(p.segments.len(), 2);
        // Fast peer was sorted first, takes the first half.
        assert_eq!(p.segments[0].node_id, NodeId([1u8; 32]));
        assert_eq!(p.segments[0].layer_range, (0, 16));
        assert_eq!(p.segments[1].node_id, NodeId([2u8; 32]));
        assert_eq!(p.segments[1].layer_range, (16, 32));
    }

    #[test]
    fn infeasible_when_total_capacity_too_small() {
        // 10 layers of capacity total but model has 32 layers → no plan.
        let peers = vec![peer(1, 5, 50.0, 0), peer(2, 5, 50.0, 0)];
        let plan = recommend_allocation(&peers, 32, 4);
        assert!(plan.is_none());
    }

    #[test]
    fn zero_capacity_peers_skipped() {
        // One fast zero-capacity peer + one useful peer.
        let peers = vec![peer(1, 0, 1000.0, 0), peer(2, 32, 50.0, 10)];
        let plan = recommend_allocation(&peers, 32, 4).unwrap();
        assert_eq!(plan.pipelines.len(), 1);
        assert_eq!(plan.pipelines[0].segments.len(), 1);
        assert_eq!(plan.pipelines[0].segments[0].node_id, NodeId([2u8; 32]));
    }

    #[test]
    fn water_filling_splits_big_peer_across_pipelines() {
        // One peer has 64 layers of capacity (enough for 2 pipelines of 32);
        // allocator should split its capacity into two pipelines.
        let peers = vec![peer(1, 64, 50.0, 10), peer(2, 32, 50.0, 10)];
        let plan = recommend_allocation(&peers, 32, 3).unwrap();
        assert_eq!(plan.pipelines.len(), 3);
        // Each pipeline gets one segment from the big peer (32-layer capacity)
        // or one segment from the smaller peer.
        let total_layers: u32 = plan
            .pipelines
            .iter()
            .map(|p| {
                p.segments
                    .iter()
                    .map(|s| s.layer_range.1 - s.layer_range.0)
                    .sum::<u32>()
            })
            .sum();
        assert_eq!(total_layers, 32 * 3);
    }

    #[test]
    fn throughput_score_monotonic_in_k() {
        // Z(k) = k^2 / avg_stages — for a balanced cluster, avg_stages stays
        // constant as k grows (each pipeline still has 1 stage), so Z(k)
        // grows quadratically.
        let peers = vec![
            peer(1, 32, 50.0, 10),
            peer(2, 32, 50.0, 10),
            peer(3, 32, 50.0, 10),
            peer(4, 32, 50.0, 10),
        ];
        let plan1 = pack_k_pipelines(&peers, 32, 1).unwrap();
        let plan4 = pack_k_pipelines(&peers, 32, 4).unwrap();
        assert!(
            plan4.throughput_score > plan1.throughput_score,
            "k=4 ({}) should beat k=1 ({})",
            plan4.throughput_score,
            plan1.throughput_score
        );
    }
}
