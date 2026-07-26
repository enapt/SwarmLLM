//! SWARM-SPEC Layer 2: adaptive pipeline hedging.
//!
//! Tracks per-segment-holder latency EWMA. When an in-flight forward
//! exceeds `1.5 × p99_estimate` for that holder, dispatch a duplicate
//! forward to the second-best holder; whichever Response arrives first
//! wins; the loser is cancelled via the existing
//! `SwarmMessage::CancelInference` cross-wire cancel (R126).
//!
//! # Why P2P-native
//!
//! Data-centre frameworks don't hedge because NVLink RTT is sub-ms.
//! P2P RTT distributions are long-tail (NAT traversal, residential
//! bandwidth, congested upstreams). Hedging cuts the p95-p99 by
//! 30-50% at a bounded cost of ~5% wasted bandwidth on cancelled
//! losers.
//!
//! # Scope
//!
//! This module owns the DECISION (latency tracking + should-hedge
//! query + rate-limit budget). The duplicate-dispatch + cancel
//! integration lives in `pipeline/distributed.rs` once the scaffolding
//! is wired. Splitting the module this way keeps the tested unit
//! small and the integration point reviewable.

use dashmap::DashMap;

use crate::types::{unix_now_ms, NodeId};

/// Composite key for per-segment-holder latency tracking. Different
/// models / segments can have very different latency profiles on the
/// same physical holder (large vs small model, prefill-heavy vs
/// decode-heavy), so we key on all three.
#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct HedgeKey {
    pub model_id: crate::types::ModelId,
    pub segment_idx: u8,
    pub holder: NodeId,
}

/// Rolling latency stats for one (model, segment, holder) triple.
/// EWMA-based — bounded memory regardless of request rate.
#[derive(Clone, Copy, Debug)]
pub struct HedgeStats {
    /// EWMA of observed forward latency in milliseconds.
    pub ewma_ms: f32,
    /// EWMA of observed latency variance — proxy for tail-heaviness.
    /// `ewma_ms + 3·sqrt(ewma_var)` is our p99-ish estimate.
    pub ewma_var: f32,
    /// Number of samples seen so far. Used to gate hedging until we
    /// have enough samples to estimate variance meaningfully.
    pub samples: u32,
    /// Wall-clock ms of the last `observe` call. Drives stale-entry
    /// eviction in `HedgeTracker::evict_stale` — entries for peers
    /// that have left the swarm stop receiving observations and grow
    /// stale, accumulating one `(model × segment × holder)` triple
    /// per departed peer.
    pub last_observed_at_ms: u64,
}

impl Default for HedgeStats {
    fn default() -> Self {
        Self {
            ewma_ms: 0.0,
            ewma_var: 0.0,
            samples: 0,
            last_observed_at_ms: 0,
        }
    }
}

impl HedgeStats {
    /// EWMA decay factor. 0.2 means ~20% weight on the new sample,
    /// 80% on the prior EWMA — converges in ~10 samples.
    const ALPHA: f32 = 0.2;

    /// Update with a fresh latency sample.
    pub fn observe(&mut self, latency_ms: f32) {
        if self.samples == 0 {
            self.ewma_ms = latency_ms;
            self.ewma_var = 0.0;
        } else {
            let delta = latency_ms - self.ewma_ms;
            self.ewma_ms += Self::ALPHA * delta;
            // Variance EWMA via squared deviation
            let sq = delta * delta;
            self.ewma_var = (1.0 - Self::ALPHA) * self.ewma_var + Self::ALPHA * sq;
        }
        self.samples = self.samples.saturating_add(1);
        self.last_observed_at_ms = unix_now_ms();
    }

    /// Rough p99 estimate: mean + 3σ. Conservative on heavy-tail
    /// distributions but cheap (no histogram).
    pub fn p99_estimate_ms(&self) -> f32 {
        self.ewma_ms + 3.0 * self.ewma_var.sqrt()
    }
}

/// Per-holder hedge tracker. Shared across the pipeline executor;
/// concurrent reads/writes via DashMap shards.
#[derive(Default)]
pub struct HedgeTracker {
    stats: DashMap<HedgeKey, HedgeStats>,
    /// Hedge counter: count of (decisions made, hedges actually fired,
    /// hedges where the duplicate won). Used to derive the live
    /// hedge rate against the budget.
    decisions: std::sync::atomic::AtomicU64,
    hedges_fired: std::sync::atomic::AtomicU64,
    hedges_won: std::sync::atomic::AtomicU64,
    /// Rolling start time — hedge rate is "hedges_fired / decisions
    /// since reset". Reset every `RESET_INTERVAL_SECS`.
    window_start_ms: std::sync::atomic::AtomicU64,
}

/// Reset window for the hedge-rate counter. Each window measures the
/// hedge rate independently; long-running daemons don't accumulate
/// rate-budget across days. 600s = 10 minutes — long enough for the
/// EWMA to stabilise, short enough to react to a network shift.
pub const HEDGE_RATE_WINDOW_SECS: u64 = 600;

impl HedgeTracker {
    pub fn new() -> Self {
        Self {
            stats: DashMap::new(),
            decisions: 0.into(),
            hedges_fired: 0.into(),
            hedges_won: 0.into(),
            window_start_ms: unix_now_ms().into(),
        }
    }

    /// Record a successful forward observation.
    pub fn observe(&self, key: HedgeKey, latency_ms: f32) {
        let mut entry = self.stats.entry(key).or_default();
        entry.observe(latency_ms);
    }

    /// Snapshot the current stats for `key`. `None` if no samples yet.
    pub fn get(&self, key: &HedgeKey) -> Option<HedgeStats> {
        self.stats.get(key).map(|e| *e.value())
    }

    /// Should we hedge a forward that's been in flight for `elapsed_ms`
    /// against the stats for `key`? Returns true when:
    /// - hedging is enabled
    /// - we have enough samples for a meaningful p99 estimate
    ///   (`min_samples`)
    /// - elapsed exceeds `factor × p99_estimate`
    /// - we're under the rolling rate budget
    pub fn should_hedge(&self, key: &HedgeKey, elapsed_ms: f32, cfg: HedgeConfig) -> bool {
        self.maybe_reset_window();
        if !cfg.enabled {
            return false;
        }
        let stats = match self.get(key) {
            Some(s) => s,
            None => return false,
        };
        if stats.samples < cfg.min_samples {
            return false;
        }
        let threshold = stats.p99_estimate_ms() * cfg.after_factor;
        if elapsed_ms < threshold {
            return false;
        }
        // Rate budget check. Acquire ordering pairs with the
        // Release stores in maybe_reset_window so a thread that
        // observes the new window_start_ms also observes the zeroed
        // counters on weakly-ordered architectures (ARM, RISC-V).
        let decisions = self.decisions.load(std::sync::atomic::Ordering::Acquire);
        let fired = self.hedges_fired.load(std::sync::atomic::Ordering::Acquire);
        if decisions > 0 {
            let current_rate = fired as f32 / decisions as f32;
            if current_rate >= cfg.max_rate {
                return false;
            }
        }
        true
    }

    /// Record a decision point — whether or not we actually hedged.
    /// Drives the rate budget.
    pub fn record_decision(&self, hedged: bool, won: bool) {
        self.decisions
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if hedged {
            self.hedges_fired
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if won {
                self.hedges_won
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
        }
    }

    fn maybe_reset_window(&self) {
        let now = unix_now_ms();
        let start = self
            .window_start_ms
            .load(std::sync::atomic::Ordering::Relaxed);
        // Reset on elapsed-window OR on backward clock jump (NTP
        // correction, container/VM clock drift). Without the
        // `start > now` arm a backwards jump leaves `now - start = 0`
        // permanently below the threshold, freezing the rate budget
        // indefinitely and blocking all future hedges once the cap
        // was hit.
        let backward_jump = start > now;
        let elapsed_ok = now.saturating_sub(start) >= HEDGE_RATE_WINDOW_SECS * 1000;
        if elapsed_ok || backward_jump {
            // Best-effort CAS. If two threads race we'll just reset once.
            if self
                .window_start_ms
                .compare_exchange(
                    start,
                    now,
                    std::sync::atomic::Ordering::AcqRel,
                    std::sync::atomic::Ordering::Relaxed,
                )
                .is_ok()
            {
                self.decisions
                    .store(0, std::sync::atomic::Ordering::Release);
                self.hedges_fired
                    .store(0, std::sync::atomic::Ordering::Release);
                self.hedges_won
                    .store(0, std::sync::atomic::Ordering::Release);
            }
        }
    }

    /// Drop per-(model, segment, holder) entries whose last observation
    /// is older than `max_age_ms`. Called periodically by the daemon
    /// to bound memory — peers that have left the swarm stop receiving
    /// observations and would otherwise accumulate one entry per
    /// (model × segment) they ever touched. Returns the number of
    /// entries evicted.
    pub fn evict_stale(&self, now_ms: u64, max_age_ms: u64) -> usize {
        let before = self.stats.len();
        self.stats.retain(|_, s| {
            if s.last_observed_at_ms == 0 {
                // Default entry with no observations — keep; the next
                // call to `observe` will stamp it.
                return true;
            }
            now_ms.saturating_sub(s.last_observed_at_ms) < max_age_ms
        });
        before - self.stats.len()
    }

    /// Sample-weighted latency per holder, collapsed across (model, segment).
    ///
    /// The tracker keys on the triple because that is what hedging needs to
    /// decide. Operators ask a coarser question — "which peer is dragging the
    /// pipeline" — so aggregate here rather than making every caller
    /// re-implement the weighting. Returns `(holder, ewma_ms, samples)` with
    /// entries carrying no samples omitted, since a zero-sample EWMA is a
    /// default value rather than a measurement.
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

    /// Current snapshot of hedge metrics — exposed via /api/admin/stats
    /// for operator visibility.
    pub fn metrics(&self) -> HedgeMetrics {
        HedgeMetrics {
            tracked_keys: self.stats.len(),
            decisions: self.decisions.load(std::sync::atomic::Ordering::Relaxed),
            hedges_fired: self.hedges_fired.load(std::sync::atomic::Ordering::Relaxed),
            hedges_won: self.hedges_won.load(std::sync::atomic::Ordering::Relaxed),
            window_start_ms: self
                .window_start_ms
                .load(std::sync::atomic::Ordering::Relaxed),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct HedgeConfig {
    pub enabled: bool,
    /// Hedge when elapsed > `after_factor × p99_estimate`. Default 1.5.
    /// Lower values hedge more aggressively (higher cost, more tail
    /// latency cut). Higher values are more conservative.
    pub after_factor: f32,
    /// Maximum fraction of decisions that fire a hedge. Default 0.05.
    /// Prevents runaway duplicate traffic when the network is in a
    /// degraded state (every request would exceed threshold).
    pub max_rate: f32,
    /// Minimum samples before we trust the EWMA enough to hedge
    /// against it. Default 20 — at α=0.2 the variance EWMA reaches
    /// ~90% of its true value by sample 20; below that the variance
    /// estimate severely undershoots and p99_estimate_ms() collapses
    /// to roughly the mean, causing hedge to fire on any +50%
    /// latency spike during the warm-up window (violating the
    /// max_rate budget guarantee). Was 5 in the initial scaffolding;
    /// bumped to 20 after the L2 review found warm-up over-firing.
    pub min_samples: u32,
}

impl Default for HedgeConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            after_factor: 1.5,
            max_rate: 0.05,
            min_samples: 20,
        }
    }
}

#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
pub struct HedgeMetrics {
    pub tracked_keys: usize,
    pub decisions: u64,
    pub hedges_fired: u64,
    pub hedges_won: u64,
    pub window_start_ms: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(seed: u8) -> HedgeKey {
        HedgeKey {
            model_id: crate::types::ModelId(format!("m{seed}")),
            segment_idx: 0,
            holder: crate::types::NodeId([seed; 32]),
        }
    }

    #[test]
    fn ewma_converges_to_repeated_sample() {
        let mut s = HedgeStats::default();
        for _ in 0..30 {
            s.observe(100.0);
        }
        assert!((s.ewma_ms - 100.0).abs() < 0.5);
        // Variance should also converge to ~0 for a constant signal.
        assert!(s.ewma_var < 0.5);
        // p99 with zero variance ≈ mean
        assert!((s.p99_estimate_ms() - 100.0).abs() < 1.0);
    }

    #[test]
    fn ewma_widens_p99_under_jitter() {
        let mut s = HedgeStats::default();
        // Alternate 80 and 120 → mean 100, std ~20
        for i in 0..40 {
            s.observe(if i % 2 == 0 { 80.0 } else { 120.0 });
        }
        assert!(
            s.ewma_var > 100.0,
            "var should reflect jitter: {}",
            s.ewma_var
        );
        let p99 = s.p99_estimate_ms();
        assert!(p99 > 100.0 + 30.0, "p99 should exceed mean+30: {}", p99);
    }

    #[test]
    fn should_hedge_disabled_returns_false() {
        let t = HedgeTracker::new();
        let k = key(1);
        for _ in 0..10 {
            t.observe(k.clone(), 100.0);
        }
        let cfg = HedgeConfig {
            enabled: false,
            ..HedgeConfig::default()
        };
        assert!(!t.should_hedge(&k, 1000.0, cfg));
    }

    #[test]
    fn should_hedge_below_min_samples_returns_false() {
        let t = HedgeTracker::new();
        let k = key(2);
        // Only 2 samples — below default min_samples=20.
        t.observe(k.clone(), 100.0);
        t.observe(k.clone(), 100.0);
        let cfg = HedgeConfig {
            enabled: true,
            ..HedgeConfig::default()
        };
        assert!(!t.should_hedge(&k, 1000.0, cfg));
    }

    #[test]
    fn should_hedge_fires_when_elapsed_exceeds_factor_times_p99() {
        let t = HedgeTracker::new();
        let k = key(3);
        // Default min_samples is 20 (bumped from 5 after L2 review).
        // Feed enough observations to clear the gate.
        for _ in 0..25 {
            t.observe(k.clone(), 100.0);
        }
        let cfg = HedgeConfig {
            enabled: true,
            after_factor: 1.5,
            ..HedgeConfig::default()
        };
        // p99 ~ 100, threshold = 150. Elapsed 200 should fire.
        assert!(t.should_hedge(&k, 200.0, cfg));
        // Elapsed 120 should NOT fire.
        assert!(!t.should_hedge(&k, 120.0, cfg));
    }

    #[test]
    fn should_hedge_respects_max_rate_budget() {
        let t = HedgeTracker::new();
        let k = key(4);
        for _ in 0..25 {
            t.observe(k.clone(), 100.0);
        }
        let cfg = HedgeConfig {
            enabled: true,
            after_factor: 1.0,
            max_rate: 0.05,
            ..HedgeConfig::default()
        };
        // Simulate having already fired 10 hedges in 100 decisions
        // (10% rate, above the 5% budget).
        for _ in 0..100 {
            t.record_decision(false, false);
        }
        for _ in 0..10 {
            t.record_decision(true, false);
        }
        // Even with a clear timeout (elapsed >> p99), budget refuses.
        assert!(!t.should_hedge(&k, 1000.0, cfg));
    }

    #[test]
    fn metrics_count_decisions_correctly() {
        let t = HedgeTracker::new();
        t.record_decision(false, false);
        t.record_decision(true, false);
        t.record_decision(true, true);
        let m = t.metrics();
        assert_eq!(m.decisions, 3);
        assert_eq!(m.hedges_fired, 2);
        assert_eq!(m.hedges_won, 1);
    }

    #[test]
    fn evict_stale_drops_aged_entries() {
        let t = HedgeTracker::new();
        // Observe two keys, then synthetically backdate the first's
        // `last_observed_at_ms` so it looks aged.
        t.observe(key(1), 100.0);
        t.observe(key(2), 100.0);
        // Backdate key(1) to 2h ago — past the 1h eviction horizon used
        // in HealthMonitor.
        let now = unix_now_ms();
        let two_hours_ago = now.saturating_sub(2 * 3_600_000);
        if let Some(mut e) = t.stats.get_mut(&key(1)) {
            e.last_observed_at_ms = two_hours_ago;
        }
        let evicted = t.evict_stale(now, 3_600_000);
        assert_eq!(evicted, 1);
        assert!(t.get(&key(1)).is_none());
        assert!(t.get(&key(2)).is_some());
    }

    #[test]
    fn evict_stale_preserves_fresh_observations() {
        let t = HedgeTracker::new();
        t.observe(key(1), 100.0);
        let now = unix_now_ms();
        // Eviction with a long max_age — fresh entry must remain.
        let evicted = t.evict_stale(now, 3_600_000);
        assert_eq!(evicted, 0);
        assert!(t.get(&key(1)).is_some());
    }

    #[test]
    fn observe_stamps_last_observed_at_ms() {
        let t = HedgeTracker::new();
        let before = unix_now_ms();
        t.observe(key(1), 100.0);
        let after = unix_now_ms();
        let stats = t.get(&key(1)).unwrap();
        assert!(stats.last_observed_at_ms >= before);
        assert!(stats.last_observed_at_ms <= after.saturating_add(1));
    }
}
