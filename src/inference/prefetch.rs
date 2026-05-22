//! SWARM-SPEC Layer 3: conversation-level predictive prefetch.
//!
//! Uses peer idle time between user turns (typically 10-60s) to
//! pre-compute work that would otherwise block the next request's
//! time-to-first-token:
//!
//! - **Activation seeding**: from the END-state of the assistant's
//!   response, run K layers forward with M candidate first-user-tokens
//!   ("Yes", "Continue", "Show me", learned per-session). Cache
//!   activations. On user's next message, skip K layers of prefill
//!   if the first token matches a candidate.
//! - **Prefix-cache gossip warming**: re-emit `PrefixCacheAnnounce` to
//!   peers that are predicted to host shards we'll need. Already wired
//!   via existing gossip; this layer just nudges the cadence.
//! - **Pipeline placement prediction**: pre-compute the best
//!   `PipelineAssignment` for the predicted next request. When the
//!   request fires, skip the scheduling decision (~10-50 ms saved on
//!   TTFT).
//!
//! # Scope of this commit
//!
//! Ships the **session-end recorder** + **idle-time predictor** +
//! **candidate-token learner** as standalone, testable units. The
//! actual prefetch dispatch (running activations forward, calling
//! scheduler, gossiping warming) is a series of follow-on commits;
//! each can plug into this orchestrator.
//!
//! Why this design: novel features are best built BOTTOM-UP. The
//! predictor + learner unit can be exercised in isolation (no
//! cluster needed), validated against synthetic traces, and the
//! dispatch wiring becomes a focused integration job.

use std::sync::Arc;

use dashmap::DashMap;

/// Per-session conversation history used to predict the next
/// user-turn's first-token candidates.
#[derive(Clone, Debug, Default)]
pub struct SessionPrefetchHistory {
    /// Histogram of first-tokens observed at the start of each
    /// user message in this session. The top-K most frequent are
    /// the activation-seeding candidates.
    pub first_token_counts: std::collections::HashMap<u32, u32>,
    /// Timestamp (unix-ms) of the last assistant response sent.
    /// Used to compute idle time → prefetch budget.
    pub last_response_at_ms: u64,
    /// Total user turns observed so far. Drives confidence — low
    /// turn count means we don't trust the histogram yet.
    pub user_turns: u32,
}

impl SessionPrefetchHistory {
    /// Top-K first-token candidates by observed frequency. Returns
    /// at most `k` tokens; empty Vec if no turns observed yet.
    pub fn top_k_candidates(&self, k: usize) -> Vec<u32> {
        if self.first_token_counts.is_empty() {
            return Vec::new();
        }
        let mut entries: Vec<(u32, u32)> = self
            .first_token_counts
            .iter()
            .map(|(t, c)| (*t, *c))
            .collect();
        entries.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        entries.into_iter().take(k).map(|(t, _)| t).collect()
    }

    /// Record a user-turn first-token observation.
    pub fn observe_user_turn(&mut self, first_token: u32) {
        *self.first_token_counts.entry(first_token).or_insert(0) += 1;
        self.user_turns = self.user_turns.saturating_add(1);
    }

    /// Record a completed assistant response. Updates the idle-time
    /// anchor so subsequent prefetch-budget queries reflect the new
    /// "we are now idle" state.
    pub fn record_response_completion(&mut self, now_ms: u64) {
        self.last_response_at_ms = now_ms;
    }

    /// Estimated idle time in ms since the last assistant response.
    /// 0 if no response has been sent yet.
    pub fn idle_ms(&self, now_ms: u64) -> u64 {
        if self.last_response_at_ms == 0 {
            return 0;
        }
        now_ms.saturating_sub(self.last_response_at_ms)
    }
}

/// Global prefetch orchestrator. Holds per-session history + decides
/// when to fire prefetches based on idle time and the rate budget.
pub struct PrefetchOrchestrator {
    histories: DashMap<String, SessionPrefetchHistory>,
    /// Number of prefetch operations dispatched in the current
    /// window. Drives the rate limit so we don't spend more than
    /// `max_prefetch_rate` of total compute on prefetches.
    dispatched: std::sync::atomic::AtomicU64,
    /// Number of prefetches that turned out to be useful (their
    /// cache was hit by the next real request).
    useful: std::sync::atomic::AtomicU64,
    /// Window start for rate accounting. Resets every
    /// `PREFETCH_WINDOW_SECS`.
    window_start_ms: std::sync::atomic::AtomicU64,
}

/// Rolling window for prefetch-rate accounting. 10 minutes — long
/// enough to amortise startup transients, short enough to react to
/// a workload shift.
pub const PREFETCH_WINDOW_SECS: u64 = 600;

impl Default for PrefetchOrchestrator {
    fn default() -> Self {
        Self::new()
    }
}

impl PrefetchOrchestrator {
    pub fn new() -> Self {
        Self {
            histories: DashMap::new(),
            dispatched: 0.into(),
            useful: 0.into(),
            window_start_ms: now_ms().into(),
        }
    }

    /// Observe a user-turn first-token for `session_id`.
    pub fn observe_user_turn(&self, session_id: &str, first_token: u32) {
        let mut entry = self.histories.entry(session_id.to_string()).or_default();
        entry.observe_user_turn(first_token);
    }

    /// Mark that the assistant just finished responding to a turn.
    /// Drives idle-time accounting and is the natural trigger for
    /// "should we prefetch now?" downstream.
    pub fn record_response_completion(&self, session_id: &str, now_ms: u64) {
        let mut entry = self.histories.entry(session_id.to_string()).or_default();
        entry.record_response_completion(now_ms);
    }

    /// Snapshot of the session's history (clone). None if no history
    /// recorded.
    pub fn get_history(&self, session_id: &str) -> Option<SessionPrefetchHistory> {
        self.histories.get(session_id).map(|e| e.value().clone())
    }

    /// Should we fire a prefetch for `session_id` right now?
    ///
    /// Returns the candidate first-tokens to prefetch for (up to
    /// `cfg.max_candidates`) when:
    /// - prefetch enabled in cfg
    /// - this session has at least `cfg.min_turns_for_prediction` turns
    ///   of history (otherwise prediction is noise)
    /// - we've been idle ≥ `cfg.min_idle_ms` (avoid prefetching while
    ///   the user is still receiving the response)
    /// - we're under the rate budget
    ///
    /// Returns empty Vec when the cascade decides not to prefetch.
    pub fn should_prefetch(&self, session_id: &str, now_ms: u64, cfg: PrefetchConfig) -> Vec<u32> {
        self.maybe_reset_window();
        if !cfg.enabled {
            return Vec::new();
        }
        let history = match self.get_history(session_id) {
            Some(h) => h,
            None => return Vec::new(),
        };
        if history.user_turns < cfg.min_turns_for_prediction {
            return Vec::new();
        }
        let idle = history.idle_ms(now_ms);
        if idle < cfg.min_idle_ms {
            return Vec::new();
        }
        // Rate budget — total compute spent on prefetch shouldn't
        // exceed cfg.max_rate of total decisions. Acquire pairs with
        // the Release stores in maybe_reset_window so a thread that
        // observes the new window_start_ms also observes the zeroed
        // counters on weakly-ordered architectures.
        let dispatched = self.dispatched.load(std::sync::atomic::Ordering::Acquire);
        let useful = self.useful.load(std::sync::atomic::Ordering::Acquire);
        if dispatched > 0 {
            // If our hit-rate is below the floor AND we've already
            // dispatched a meaningful number, throttle.
            let hit_rate = useful as f32 / dispatched as f32;
            if dispatched > cfg.min_dispatches_for_throttle && hit_rate < cfg.min_useful_rate {
                return Vec::new();
            }
        }
        history.top_k_candidates(cfg.max_candidates)
    }

    /// Record that a prefetch was dispatched (whether it ends up
    /// useful or not). Drives the rate budget.
    pub fn record_dispatch(&self) {
        self.dispatched
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    /// Record that a previously-dispatched prefetch was actually
    /// used by a subsequent request — the cached activations were
    /// hit. Drives the hit-rate calculation that throttles useless
    /// prefetch storms.
    pub fn record_useful(&self) {
        self.useful
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    /// Metrics snapshot for the admin dashboard.
    pub fn metrics(&self) -> PrefetchMetrics {
        PrefetchMetrics {
            tracked_sessions: self.histories.len(),
            dispatched: self.dispatched.load(std::sync::atomic::Ordering::Relaxed),
            useful: self.useful.load(std::sync::atomic::Ordering::Relaxed),
            window_start_ms: self
                .window_start_ms
                .load(std::sync::atomic::Ordering::Relaxed),
        }
    }

    /// Drop session history for sessions that haven't seen activity
    /// in `max_idle_ms`. Called periodically by the daemon to bound
    /// memory. Returns the number of sessions evicted.
    pub fn evict_idle(&self, now_ms: u64, max_idle_ms: u64) -> usize {
        let before = self.histories.len();
        self.histories.retain(|_, h| {
            if h.last_response_at_ms == 0 {
                return true; // active prefetch-pending session
            }
            now_ms.saturating_sub(h.last_response_at_ms) < max_idle_ms
        });
        before - self.histories.len()
    }

    fn maybe_reset_window(&self) {
        let now = now_ms();
        let start = self
            .window_start_ms
            .load(std::sync::atomic::Ordering::Relaxed);
        // Reset on elapsed-window OR on backward clock jump (NTP
        // correction, container/VM clock drift). Without the
        // `start > now` arm a backwards jump leaves `now - start = 0`
        // permanently below the threshold, freezing the rate budget
        // indefinitely and blocking all future prefetches once the
        // throttle was triggered.
        let backward_jump = start > now;
        let elapsed_ok = now.saturating_sub(start) >= PREFETCH_WINDOW_SECS * 1000;
        if (elapsed_ok || backward_jump)
            && self
                .window_start_ms
                .compare_exchange(
                    start,
                    now,
                    std::sync::atomic::Ordering::AcqRel,
                    std::sync::atomic::Ordering::Relaxed,
                )
                .is_ok()
        {
            self.dispatched
                .store(0, std::sync::atomic::Ordering::Release);
            self.useful.store(0, std::sync::atomic::Ordering::Release);
        }
    }
}

/// Configuration for the prefetch orchestrator. All fields are
/// per-instance (no global state) so tests and prod can run with
/// different settings.
#[derive(Clone, Copy, Debug)]
pub struct PrefetchConfig {
    pub enabled: bool,
    /// Minimum idle ms before we consider prefetching. Avoids
    /// prefetching while the user is still receiving the response.
    /// Default 2000 ms — typical typing delay.
    pub min_idle_ms: u64,
    /// Minimum user turns of history before predictions are trusted.
    /// Below this we have no signal. Default 2.
    pub min_turns_for_prediction: u32,
    /// Max prefetch candidates emitted per cycle. Each candidate
    /// costs O(K layers) of work; default 3 balances cost vs hit
    /// rate.
    pub max_candidates: usize,
    /// Minimum useful-prefetch fraction we expect; below this and
    /// after `min_dispatches_for_throttle`, throttle prefetching.
    pub min_useful_rate: f32,
    /// Don't throttle until we have this many dispatched samples to
    /// estimate the hit rate from.
    pub min_dispatches_for_throttle: u64,
}

impl Default for PrefetchConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            min_idle_ms: 2000,
            min_turns_for_prediction: 2,
            max_candidates: 3,
            min_useful_rate: 0.15,
            min_dispatches_for_throttle: 50,
        }
    }
}

#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
pub struct PrefetchMetrics {
    pub tracked_sessions: usize,
    pub dispatched: u64,
    pub useful: u64,
    pub window_start_ms: u64,
}

fn now_ms() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Public Arc handle — daemon state holds one, all paths share it.
pub type PrefetchHandle = Arc<PrefetchOrchestrator>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn top_k_orders_by_count() {
        let mut h = SessionPrefetchHistory::default();
        h.observe_user_turn(5);
        h.observe_user_turn(5);
        h.observe_user_turn(7);
        h.observe_user_turn(7);
        h.observe_user_turn(7);
        h.observe_user_turn(9);
        // Counts: 7→3, 5→2, 9→1
        assert_eq!(h.top_k_candidates(2), vec![7, 5]);
        assert_eq!(h.top_k_candidates(10), vec![7, 5, 9]);
        assert!(h.top_k_candidates(0).is_empty());
    }

    #[test]
    fn empty_history_returns_no_candidates() {
        let h = SessionPrefetchHistory::default();
        assert!(h.top_k_candidates(5).is_empty());
    }

    #[test]
    fn idle_ms_after_response_completion() {
        let mut h = SessionPrefetchHistory::default();
        h.record_response_completion(1_000);
        assert_eq!(h.idle_ms(5_000), 4_000);
        assert_eq!(h.idle_ms(500), 0); // saturating
    }

    #[test]
    fn idle_ms_zero_before_first_response() {
        let h = SessionPrefetchHistory::default();
        assert_eq!(h.idle_ms(10_000), 0);
    }

    #[test]
    fn should_prefetch_returns_empty_when_disabled() {
        let o = PrefetchOrchestrator::new();
        o.observe_user_turn("s1", 5);
        o.observe_user_turn("s1", 5);
        o.observe_user_turn("s1", 5);
        o.record_response_completion("s1", 100);
        let cfg = PrefetchConfig {
            enabled: false,
            ..PrefetchConfig::default()
        };
        assert!(o.should_prefetch("s1", 10_000, cfg).is_empty());
    }

    #[test]
    fn should_prefetch_requires_min_turns() {
        let o = PrefetchOrchestrator::new();
        o.observe_user_turn("s1", 5);
        o.record_response_completion("s1", 100);
        let cfg = PrefetchConfig {
            enabled: true,
            min_turns_for_prediction: 2,
            min_idle_ms: 0,
            ..PrefetchConfig::default()
        };
        // Only 1 user turn — below min.
        assert!(o.should_prefetch("s1", 10_000, cfg).is_empty());
    }

    #[test]
    fn should_prefetch_requires_idle() {
        let o = PrefetchOrchestrator::new();
        o.observe_user_turn("s1", 5);
        o.observe_user_turn("s1", 5);
        o.record_response_completion("s1", 1_000);
        let cfg = PrefetchConfig {
            enabled: true,
            min_idle_ms: 2_000,
            min_turns_for_prediction: 2,
            ..PrefetchConfig::default()
        };
        // 500 ms idle — below threshold.
        assert!(o.should_prefetch("s1", 1_500, cfg).is_empty());
        // 3000 ms idle — fires.
        assert!(!o.should_prefetch("s1", 4_000, cfg).is_empty());
    }

    #[test]
    fn should_prefetch_returns_top_candidates() {
        let o = PrefetchOrchestrator::new();
        for _ in 0..3 {
            o.observe_user_turn("s1", 10);
        }
        for _ in 0..2 {
            o.observe_user_turn("s1", 20);
        }
        o.observe_user_turn("s1", 30);
        o.record_response_completion("s1", 0);
        let cfg = PrefetchConfig {
            enabled: true,
            min_idle_ms: 0,
            min_turns_for_prediction: 1,
            max_candidates: 2,
            ..PrefetchConfig::default()
        };
        let candidates = o.should_prefetch("s1", 10_000, cfg);
        assert_eq!(candidates, vec![10, 20]);
    }

    #[test]
    fn throttles_when_hit_rate_falls_below_threshold() {
        let o = PrefetchOrchestrator::new();
        for _ in 0..5 {
            o.observe_user_turn("s1", 10);
        }
        o.record_response_completion("s1", 0);
        let cfg = PrefetchConfig {
            enabled: true,
            min_idle_ms: 0,
            min_turns_for_prediction: 1,
            max_candidates: 3,
            min_useful_rate: 0.20,
            min_dispatches_for_throttle: 10,
        };
        // 100 dispatches with 0 useful → 0% hit rate, below 20% floor.
        for _ in 0..100 {
            o.record_dispatch();
        }
        assert!(o.should_prefetch("s1", 10_000, cfg).is_empty());

        // Now flip: 100 dispatches, 30 useful → 30%, above floor.
        for _ in 0..30 {
            o.record_useful();
        }
        assert!(!o.should_prefetch("s1", 10_000, cfg).is_empty());
    }

    #[test]
    fn evict_idle_drops_old_sessions() {
        let o = PrefetchOrchestrator::new();
        o.record_response_completion("s_recent", 9_000);
        o.record_response_completion("s_old", 1_000);
        o.observe_user_turn("s_active", 7); // never completed → preserved
                                            // Now = 10_000. max_idle = 2_000. Recent is 1000 idle → keep.
                                            // Old is 9000 idle → drop. Active has no completion → keep.
        let evicted = o.evict_idle(10_000, 2_000);
        assert_eq!(evicted, 1);
        assert!(o.get_history("s_old").is_none());
        assert!(o.get_history("s_recent").is_some());
        assert!(o.get_history("s_active").is_some());
    }

    #[test]
    fn metrics_reflect_dispatch_state() {
        let o = PrefetchOrchestrator::new();
        o.record_dispatch();
        o.record_dispatch();
        o.record_dispatch();
        o.record_useful();
        let m = o.metrics();
        assert_eq!(m.dispatched, 3);
        assert_eq!(m.useful, 1);
    }
}
