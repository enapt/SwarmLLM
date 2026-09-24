//! Per-(model, segment, holder) forward latency, as a moving average.
//!
//! Recorded on every successful segment forward
//! (`SharedState::record_segment_latency`, from `pipeline/distributed.rs`) and
//! read by the peer performance table (`SharedState::peer_performance_rows`),
//! which collapses it per holder to answer "which computer is dragging the
//! pipeline".
//!
//! **This was the measuring half of hedged verify dispatch, which was removed
//! on 2026-09-24 (FUTURE_WORK #94).** A hedge duplicated a decode-time verify
//! forward to a second holder under a new request id — a machine holding no
//! cache for that conversation, so its answer could never be right. Making it
//! right means replaying the whole history to that machine first, a prompt pass
//! per hedge, which costs more than the wait it exists to save. Hedging works
//! for requests any replica can answer on its own (Dean & Barroso, *The Tail at
//! Scale*); a stateful decode step is not one, which is why no inference
//! server hedges decode steps. This node fails over with a replay instead
//! (`distributed::assemble_replay`).

use dashmap::DashMap;

use crate::types::{unix_now_ms, NodeId};

/// One (model, segment, holder) triple. Different models and segments can have
/// very different latency on the same holder, so the average is kept per
/// triple and only collapsed per holder when it is read.
#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct SegmentKey {
    pub model_id: crate::types::ModelId,
    pub segment_idx: u8,
    pub holder: NodeId,
}

/// Moving-average latency for one triple. Bounded memory at any request rate.
#[derive(Clone, Copy, Debug, Default)]
pub struct LatencyStats {
    /// EWMA of observed forward latency in milliseconds.
    pub ewma_ms: f32,
    /// Samples seen so far — what the performance table reports as confidence.
    pub samples: u32,
    /// Wall-clock ms of the last observation. Drives `evict_stale`: a peer that
    /// has left the swarm stops receiving observations, and without eviction
    /// every departed peer would leave one entry per (model × segment) behind.
    pub last_observed_at_ms: u64,
}

impl LatencyStats {
    /// Weight on the newest sample: converges in about ten.
    const ALPHA: f32 = 0.2;

    pub fn observe(&mut self, latency_ms: f32) {
        if self.samples == 0 {
            self.ewma_ms = latency_ms;
        } else {
            self.ewma_ms += Self::ALPHA * (latency_ms - self.ewma_ms);
        }
        self.samples = self.samples.saturating_add(1);
        self.last_observed_at_ms = unix_now_ms();
    }
}

/// Shared across the pipeline executor; concurrent via `DashMap` shards.
#[derive(Default)]
pub struct SegmentLatencyTracker {
    stats: DashMap<SegmentKey, LatencyStats>,
}

impl SegmentLatencyTracker {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a successful forward.
    pub fn observe(&self, key: SegmentKey, latency_ms: f32) {
        self.stats.entry(key).or_default().observe(latency_ms);
    }

    /// The stats for `key`, or `None` if nothing has been observed.
    pub fn get(&self, key: &SegmentKey) -> Option<LatencyStats> {
        self.stats.get(key).map(|e| *e.value())
    }

    /// Drop entries whose last observation is older than `max_age_ms`, and
    /// return how many went. Called on the health-monitor tick to bound memory.
    pub fn evict_stale(&self, now_ms: u64, max_age_ms: u64) -> usize {
        let before = self.stats.len();
        self.stats.retain(|_, s| {
            // A default entry with no observation yet is kept; the next
            // `observe` stamps it.
            s.last_observed_at_ms == 0 || now_ms.saturating_sub(s.last_observed_at_ms) < max_age_ms
        });
        before - self.stats.len()
    }

    /// Sample-weighted latency per holder, collapsed across (model, segment):
    /// `(holder, ewma_ms, samples)`, omitting holders with no samples — a
    /// zero-sample average is a default, not a measurement.
    pub fn latency_by_holder(&self) -> Vec<(NodeId, f32, u32)> {
        let mut acc: std::collections::HashMap<NodeId, (f64, u64)> =
            std::collections::HashMap::new();
        for e in self.stats.iter() {
            let s = e.value();
            if s.samples == 0 {
                continue;
            }
            let slot = acc.entry(e.key().holder.clone()).or_insert((0.0, 0));
            slot.0 += s.ewma_ms as f64 * s.samples as f64;
            slot.1 += s.samples as u64;
        }
        acc.into_iter()
            .map(|(holder, (weighted, n))| (holder, (weighted / n as f64) as f32, n as u32))
            .collect()
    }

    /// Snapshot for `/api/admin/stats` and the WebSocket stats tick.
    pub fn metrics(&self) -> SegmentLatencyMetrics {
        SegmentLatencyMetrics {
            tracked_keys: self.stats.len(),
        }
    }
}

#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
pub struct SegmentLatencyMetrics {
    pub tracked_keys: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(seed: u8) -> SegmentKey {
        SegmentKey {
            model_id: crate::types::ModelId(format!("m{seed}")),
            segment_idx: 0,
            holder: crate::types::NodeId([seed; 32]),
        }
    }

    #[test]
    fn ewma_converges_to_repeated_sample() {
        let mut s = LatencyStats::default();
        for _ in 0..30 {
            s.observe(100.0);
        }
        assert!((s.ewma_ms - 100.0).abs() < 0.5);
        assert_eq!(s.samples, 30);
    }

    #[test]
    fn latency_by_holder_weights_by_samples_across_models() {
        let t = SegmentLatencyTracker::new();
        let holder = crate::types::NodeId([7; 32]);
        for (model, ms, n) in [("a", 100.0, 3), ("b", 200.0, 1)] {
            for _ in 0..n {
                t.observe(
                    SegmentKey {
                        model_id: crate::types::ModelId(model.into()),
                        segment_idx: 0,
                        holder: holder.clone(),
                    },
                    ms,
                );
            }
        }
        let rows = t.latency_by_holder();
        assert_eq!(rows.len(), 1);
        let (h, ewma, samples) = &rows[0];
        assert_eq!(h, &holder);
        assert_eq!(*samples, 4);
        assert!(
            (ewma - 125.0).abs() < 0.01,
            "(3×100 + 1×200) / 4, got {ewma}"
        );
    }

    #[test]
    fn evict_stale_drops_aged_entries() {
        let t = SegmentLatencyTracker::new();
        t.observe(key(1), 100.0);
        t.observe(key(2), 100.0);
        // Backdate key(1) to 2 h ago — past the 1 h horizon HealthMonitor uses.
        let now = unix_now_ms();
        if let Some(mut e) = t.stats.get_mut(&key(1)) {
            e.last_observed_at_ms = now.saturating_sub(2 * 3_600_000);
        }
        assert_eq!(t.evict_stale(now, 3_600_000), 1);
        assert!(t.get(&key(1)).is_none());
        assert!(t.get(&key(2)).is_some());
    }

    #[test]
    fn evict_stale_preserves_fresh_observations() {
        let t = SegmentLatencyTracker::new();
        t.observe(key(1), 100.0);
        assert_eq!(t.evict_stale(unix_now_ms(), 3_600_000), 0);
        assert!(t.get(&key(1)).is_some());
    }

    #[test]
    fn observe_stamps_last_observed_at_ms() {
        let t = SegmentLatencyTracker::new();
        let before = unix_now_ms();
        t.observe(key(1), 100.0);
        let after = unix_now_ms();
        let stats = t.get(&key(1)).unwrap();
        assert!(stats.last_observed_at_ms >= before);
        assert!(stats.last_observed_at_ms <= after.saturating_add(1));
    }
}
