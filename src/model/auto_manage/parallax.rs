//! Auto-manage Phase C.2 integration with the Parallax offline layer
//! allocator.
//!
//! Each auto-manage evaluation cycle runs `PipelineScheduler::allocate_offline`
//! over every known model and feeds the result into a per-shard stability
//! counter. When the allocator consistently recommends that the local node
//! hold a given shard for `PARALLAX_STABILITY_THRESHOLD` consecutive ticks,
//! the shard's acquire score gets a soft bias; when the allocator
//! consistently wants it off us, the prune score gets a bias in the other
//! direction.
//!
//! Score bias only — all existing hard constraints (trust gates, configured
//! ranges, locked shards, pool pins, encrypted-pipeline protection, hash-ring
//! dedup) still apply ahead of the bias.

use std::collections::HashSet;

use crate::types::ShardId;

use super::manager::AutoShardManager;

/// Magnitude threshold for acting on the stability counter. Must be held in
/// agreement for this many consecutive ticks before score bias kicks in.
pub(super) const PARALLAX_STABILITY_THRESHOLD: i32 = 3;

/// Hard cap on counter magnitude. A long-stable recommendation can't require
/// many ticks of opposing signal to flip back to neutral.
const PARALLAX_STABILITY_MAX: i32 = 10;

/// Multiplicative acquire-score bonus applied when a shard's stability
/// counter is ≥ threshold. 1.5× matches the regional `source_bonus` order —
/// noticeable but not overwhelming.
pub(super) const PARALLAX_ACQUIRE_BONUS: f64 = 1.5;

/// Additive prune-score bonus applied when a shard's stability counter is
/// ≤ `-threshold`. Same magnitude as the VRAM-pressure prune bonus family.
pub(super) const PARALLAX_PRUNE_BONUS: f64 = 0.5;

impl AutoShardManager {
    /// Recompute Parallax stability counters for every known shard.
    ///
    /// Positive increment means the Parallax allocator placed *some part of*
    /// the shard's layer range on the local node in its best recommended
    /// plan for that model; negative decrement means it didn't (the
    /// allocator sees a better place for those layers).
    ///
    /// No-op when `parallax_auto_rebalance=false`. Per-model: if
    /// `allocate_offline` returns `None` (infeasible cluster), that model's
    /// shards are skipped entirely rather than falling through to a
    /// misleading "not recommended" signal.
    pub(super) fn update_parallax_stability(&self) {
        if !self.shared_state.cfg().auto_manage.parallax_auto_rebalance {
            return;
        }
        let local_node_id = self.shared_state.identity.node_id().clone();
        let scheduler =
            crate::inference::scheduler::PipelineScheduler::new(self.shared_state.clone());
        // One pipeline per peer is the natural ceiling — Z(k) won't recommend
        // more pipelines than there are distinct peers in a balanced cluster.
        let max_pipelines = (self.shared_state.peer_registry.len() as u32 + 1).clamp(1, 16);

        for manifest in self.shared_state.model_registry.models() {
            let plan = match scheduler.allocate_offline(&manifest.id, max_pipelines) {
                Some(p) => p,
                None => continue,
            };

            // Collect every layer range the allocator wants on the local node
            // across any recommended pipeline. A shard is "recommended" iff
            // any of its layers falls inside at least one of these ranges.
            let mut local_ranges: Vec<(u32, u32)> = Vec::new();
            for pipe in &plan.pipelines {
                for seg in &pipe.segments {
                    if seg.node_id == local_node_id {
                        local_ranges.push(seg.layer_range);
                    }
                }
            }

            let recommended: HashSet<u32> = manifest
                .shards
                .iter()
                .filter(|s| shard_overlaps_any(s.layer_range, &local_ranges))
                .map(|s| s.index)
                .collect();

            for shard in &manifest.shards {
                let sid = ShardId {
                    model_id: manifest.id.clone(),
                    index: shard.index,
                };
                let delta = if recommended.contains(&shard.index) {
                    1
                } else {
                    -1
                };
                let mut entry = self
                    .shared_state
                    .models
                    .parallax_stability
                    .entry(sid)
                    .or_insert(0);
                *entry = (*entry + delta).clamp(-PARALLAX_STABILITY_MAX, PARALLAX_STABILITY_MAX);
            }
        }
    }

    /// True when the Parallax allocator has consistently recommended this
    /// node hold the given shard for at least `PARALLAX_STABILITY_THRESHOLD`
    /// ticks. `gather_candidates` uses this to multiply the acquire score.
    pub(super) fn parallax_should_boost_acquire(&self, shard_id: &ShardId) -> bool {
        if !self.shared_state.cfg().auto_manage.parallax_auto_rebalance {
            return false;
        }
        self.shared_state
            .models
            .parallax_stability
            .get(shard_id)
            .map(|r| *r.value() >= PARALLAX_STABILITY_THRESHOLD)
            .unwrap_or(false)
    }

    /// True when the Parallax allocator has consistently recommended *against*
    /// this node holding the given shard. Prune path uses this to add a
    /// prune-score bonus.
    pub(super) fn parallax_should_boost_prune(&self, shard_id: &ShardId) -> bool {
        if !self.shared_state.cfg().auto_manage.parallax_auto_rebalance {
            return false;
        }
        self.shared_state
            .models
            .parallax_stability
            .get(shard_id)
            .map(|r| *r.value() <= -PARALLAX_STABILITY_THRESHOLD)
            .unwrap_or(false)
    }
}

/// Returns true when `shard_range` overlaps any of the candidate ranges.
/// Half-open intervals `[start, end)`.
fn shard_overlaps_any(shard_range: (u32, u32), candidates: &[(u32, u32)]) -> bool {
    candidates
        .iter()
        .any(|c| shard_range.0 < c.1 && shard_range.1 > c.0)
}

#[cfg(test)]
mod tests {
    use super::super::test_support::{
        make_test_manager, make_test_manager_with_config, register_manifest_with_shards,
    };
    use super::*;
    use crate::config::Config;
    use crate::types::{ModelId, NodeId};

    #[test]
    fn overlap_detects_partial_and_full() {
        assert!(shard_overlaps_any((0, 8), &[(0, 16)])); // full inside
        assert!(shard_overlaps_any((8, 16), &[(0, 16)])); // full inside edge
        assert!(shard_overlaps_any((14, 24), &[(0, 16)])); // partial overlap
        assert!(shard_overlaps_any((0, 32), &[(10, 20)])); // range inside shard
    }

    #[test]
    fn overlap_rejects_disjoint() {
        assert!(!shard_overlaps_any((16, 32), &[(0, 16)])); // touches boundary
        assert!(!shard_overlaps_any((20, 30), &[(0, 16)]));
        assert!(!shard_overlaps_any((0, 8), &[])); // no candidates
    }

    #[test]
    fn overlap_any_of_multiple_candidates() {
        assert!(shard_overlaps_any((10, 12), &[(0, 5), (8, 14), (20, 24)]));
        assert!(!shard_overlaps_any((30, 40), &[(0, 5), (8, 14), (20, 24)]));
    }

    #[test]
    fn stability_increments_when_local_is_the_only_peer() {
        // With only the local node in the cluster and enough local capacity,
        // Parallax assigns every layer to local → every shard is recommended
        // → all counters increment on each tick.
        let (state, manager) = make_test_manager();
        let local = state.identity.node_id().clone();
        let mid = register_manifest_with_shards(
            &state,
            "test-model",
            32,
            &[(0, 8), (8, 16), (16, 24), (24, 32)],
        );
        // Register local as holder of all shards so allocate_offline treats it
        // as having enough layer capacity.
        for i in 0..4 {
            state.model_registry.record_shard_holder(
                crate::types::ShardId {
                    model_id: mid.clone(),
                    index: i,
                },
                local.clone(),
            );
        }

        manager.update_parallax_stability();
        manager.update_parallax_stability();
        manager.update_parallax_stability();

        for i in 0..4 {
            let sid = crate::types::ShardId {
                model_id: mid.clone(),
                index: i,
            };
            let v = *state.models.parallax_stability.get(&sid).unwrap().value();
            assert_eq!(v, 3, "shard {i} stability should be +3 after 3 ticks");
        }
        assert!(
            manager.parallax_should_boost_acquire(&crate::types::ShardId {
                model_id: mid.clone(),
                index: 0
            })
        );
        assert!(
            !manager.parallax_should_boost_prune(&crate::types::ShardId {
                model_id: mid,
                index: 0
            })
        );
    }

    #[test]
    fn stability_clamps_at_bounds() {
        let (state, manager) = make_test_manager();
        let local = state.identity.node_id().clone();
        let mid = register_manifest_with_shards(&state, "test-model", 16, &[(0, 16)]);
        state.model_registry.record_shard_holder(
            crate::types::ShardId {
                model_id: mid.clone(),
                index: 0,
            },
            local.clone(),
        );

        // Tick 20 times → should clamp at +10, not grow unbounded.
        for _ in 0..20 {
            manager.update_parallax_stability();
        }
        let sid = crate::types::ShardId {
            model_id: mid,
            index: 0,
        };
        let v = *state.models.parallax_stability.get(&sid).unwrap().value();
        assert_eq!(v, 10, "counter should clamp at PARALLAX_STABILITY_MAX");
    }

    #[test]
    fn feature_flag_off_is_noop() {
        let (state, manager) = make_test_manager();
        // Toggle flag off. Config is read via shared_state.config (Arc), so
        // we need to rebuild state with a different config.
        let config = Config {
            auto_manage: crate::config::AutoManageConfig {
                parallax_auto_rebalance: false,
                ..crate::config::AutoManageConfig::default()
            },
            ..state.config.clone()
        };
        let (state2, manager2) = make_test_manager_with_config(config);

        let local = state2.identity.node_id().clone();
        let mid = register_manifest_with_shards(&state2, "test-model", 16, &[(0, 16)]);
        state2.model_registry.record_shard_holder(
            crate::types::ShardId {
                model_id: mid.clone(),
                index: 0,
            },
            local,
        );

        manager2.update_parallax_stability();
        let sid = crate::types::ShardId {
            model_id: mid.clone(),
            index: 0,
        };
        assert!(
            state2.models.parallax_stability.get(&sid).is_none(),
            "flag=false → no counter updates"
        );
        assert!(!manager2.parallax_should_boost_acquire(&sid));
        assert!(!manager2.parallax_should_boost_prune(&sid));
        // Quiet the unused-manager warning in the outer fixture.
        let _ = manager;
    }

    #[test]
    fn stability_follows_allocator_when_faster_remote_preempts_local() {
        // 2-peer cluster where a remote peer with a high tokens/sec capability
        // and ample RAM wins Parallax's fastest-first ordering and covers all
        // layers. Every shard should land in the "not recommended to local"
        // camp — counters decrement, prune boost fires once stable.
        let (state, manager) = make_test_manager();
        let local = state.identity.node_id().clone();
        let mid = register_manifest_with_shards(&state, "test-model", 32, &[(0, 16), (16, 32)]);

        // Local holds shard 0 → local_layer_capacity = 16.
        state.model_registry.record_shard_holder(
            crate::types::ShardId {
                model_id: mid.clone(),
                index: 0,
            },
            local.clone(),
        );

        // Remote: huge RAM → capacity >> num_layers, fast tps → wins sort.
        let remote = NodeId([42u8; 32]);
        state.peer_registry.insert(
            remote.clone(),
            crate::types::PeerInfo {
                node_id: remote.clone(),
                addresses: vec![],
                capability: Some(crate::types::NodeCapability {
                    cpu: None,
                    node_id: remote.clone(),
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
                    est_tokens_per_sec_7b: 50.0,
                    os: None,
                    observed_latencies: vec![],
                    relay_capable: false,
                    protocol_version: 0,
                    features: 0,
                    relay_reservations: vec![],
                    anchor_mode: false,
                }),
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
        // Scheduler liveness oracle (R142.9): allocate_offline now filters by
        // connected_node_ids — without this, the remote peer would be treated
        // as disconnected and the allocator would refuse to plan against it.
        state.connected_node_ids.insert(remote.clone());

        // Sanity: the allocator does in fact pick a plan with no local
        // segments. Stability counters track that recommendation.
        let scheduler = crate::inference::scheduler::PipelineScheduler::new(state.clone());
        let plan = scheduler.allocate_offline(&mid, 3).expect("has plan");
        let local_segments: usize = plan
            .pipelines
            .iter()
            .flat_map(|p| &p.segments)
            .filter(|s| s.node_id == local)
            .count();
        assert_eq!(
            local_segments, 0,
            "allocator should place nothing on local when remote dominates"
        );

        for _ in 0..PARALLAX_STABILITY_THRESHOLD {
            manager.update_parallax_stability();
        }
        let s0 = crate::types::ShardId {
            model_id: mid.clone(),
            index: 0,
        };
        let s1 = crate::types::ShardId {
            model_id: mid,
            index: 1,
        };
        let v0 = *state.models.parallax_stability.get(&s0).unwrap().value();
        let v1 = *state.models.parallax_stability.get(&s1).unwrap().value();
        assert!(v0 <= -PARALLAX_STABILITY_THRESHOLD, "shard 0 counter={v0}");
        assert!(v1 <= -PARALLAX_STABILITY_THRESHOLD, "shard 1 counter={v1}");

        assert!(manager.parallax_should_boost_prune(&s0));
        assert!(manager.parallax_should_boost_prune(&s1));
        assert!(!manager.parallax_should_boost_acquire(&s0));
    }

    #[test]
    fn unknown_shard_is_neutral() {
        let (_state, manager) = make_test_manager();
        let sid = crate::types::ShardId {
            model_id: ModelId("no-such".into()),
            index: 0,
        };
        // Never seen this shard — neither boost fires.
        assert!(!manager.parallax_should_boost_acquire(&sid));
        assert!(!manager.parallax_should_boost_prune(&sid));
    }
}
