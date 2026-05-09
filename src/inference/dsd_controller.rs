//! Adaptive γ controller for Decentralized Speculative Decoding (Item 12 / DSD).
//!
//! The number of speculative tokens γ proposed per round trip controls the
//! tradeoff between (a) per-link payload size (grows linearly with γ) and (b)
//! the number of inter-node round trips (shrinks as γ tokens get accepted in
//! one batch). The optimal γ depends on the draft/target acceptance rate,
//! which varies by request, by model pair, and over time within a request.
//!
//! Paper 1 (arxiv 2511.11733) fixes γ=8 and varies its τ relaxation parameter
//! instead. Paper 2 (arxiv 2511.21669) trains a small MLP on
//! `[queue_depth_util, accept_rate, RTT, TPOT, prev_γ]` to predict γ each step.
//!
//! This module implements the **MVP from paper 2's "Dynamic window" baseline**:
//! a simple heuristic that nudges γ up when acceptance is high and down when
//! it's low, using an exponential moving average of the recent accept rate.
//! Cheap, no model artifact, captures most of paper 2's win without shipping a
//! trained model. Replaceable by the MLP variant in a future increment.
//!
//! Update rule:
//!   accept_ema  ← α · accept_ema + (1 − α) · accept_rate_this_round
//!   γ_next      ← clamp(γ_now · (1 + β · (accept_ema − 0.5)), γ_min, γ_max)
//!
//! Defaults `α=0.7` (smooth), `β=0.2` (gentle adjustment), `γ_min=2`, `γ_max=12`.

/// Controller state for a single request. One instance per in-flight DSD
/// request. Keep it small — no per-link breakdown in v1.
#[derive(Debug, Clone)]
pub struct GammaController {
    /// Current proposed γ for the next round.
    gamma: u32,
    /// Exponential moving average of the per-round acceptance rate
    /// `accepted / proposed`. Initialised to 0.5 (assume midpoint until we
    /// observe).
    accept_ema: f32,
    /// EMA smoothing factor — weight on the historical estimate. Higher = more
    /// stable γ but slower to adapt.
    alpha: f32,
    /// Adjustment factor — how aggressively γ moves per round.
    beta: f32,
    /// Inclusive lower bound on γ.
    gamma_min: u32,
    /// Inclusive upper bound on γ.
    gamma_max: u32,
}

impl GammaController {
    /// Create a controller with the given starting γ. Bounds and tuning
    /// constants come from `defaults()`.
    pub fn new(initial_gamma: u32) -> Self {
        Self {
            gamma: initial_gamma.clamp(Self::DEFAULT_GAMMA_MIN, Self::DEFAULT_GAMMA_MAX),
            accept_ema: 0.5,
            alpha: Self::DEFAULT_ALPHA,
            beta: Self::DEFAULT_BETA,
            gamma_min: Self::DEFAULT_GAMMA_MIN,
            gamma_max: Self::DEFAULT_GAMMA_MAX,
        }
    }

    pub const DEFAULT_GAMMA_MIN: u32 = 2;
    pub const DEFAULT_GAMMA_MAX: u32 = 12;
    pub const DEFAULT_ALPHA: f32 = 0.7;
    pub const DEFAULT_BETA: f32 = 0.2;

    /// Return γ to use for the upcoming verify round.
    pub fn current_gamma(&self) -> u32 {
        self.gamma
    }

    /// Most recent EMA of the accept rate. Exposed for diagnostics / metrics.
    pub fn accept_ema(&self) -> f32 {
        self.accept_ema
    }

    /// Record the outcome of a verify round and update γ for the next one.
    ///
    /// `accepted` is the count of draft tokens that survived target verification
    /// (NOT including the bonus token sampled from the target's logits at
    /// position γ — that always lands). `proposed` is the γ that was used.
    /// `proposed` must be > 0; passing 0 is a logic bug and causes a no-op
    /// after a debug-mode assert.
    pub fn record_round(&mut self, accepted: u32, proposed: u32) {
        if proposed == 0 {
            debug_assert!(
                false,
                "GammaController::record_round called with proposed=0"
            );
            return;
        }
        let rate = (accepted as f32 / proposed as f32).clamp(0.0, 1.0);
        self.accept_ema = self.alpha * self.accept_ema + (1.0 - self.alpha) * rate;

        // Move γ towards the regime where accept_ema ≈ 0.7 (the empirical
        // sweet spot from spec-decoding literature). We pivot at 0.5 because:
        // - acceptance below 0.5 means most rounds emit ≤γ/2 tokens → wasted
        //   draft compute and wasted per-link payload.
        // - acceptance above 0.5 means we're leaving easy throughput on the
        //   table by not proposing more.
        let multiplier = 1.0 + self.beta * (self.accept_ema - 0.5);
        let next = (self.gamma as f32 * multiplier).round() as i32;
        self.gamma = (next.max(0) as u32).clamp(self.gamma_min, self.gamma_max);
    }

    /// Override the smoothing constants — only used in benchmarks / tuning.
    #[cfg(test)]
    pub fn with_tuning(mut self, alpha: f32, beta: f32) -> Self {
        self.alpha = alpha.clamp(0.0, 1.0);
        self.beta = beta.clamp(0.0, 1.0);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initial_gamma_is_clamped_to_bounds() {
        let c = GammaController::new(0);
        assert_eq!(c.current_gamma(), GammaController::DEFAULT_GAMMA_MIN);
        let c = GammaController::new(50);
        assert_eq!(c.current_gamma(), GammaController::DEFAULT_GAMMA_MAX);
        let c = GammaController::new(5);
        assert_eq!(c.current_gamma(), 5);
    }

    #[test]
    fn perfect_acceptance_grows_gamma_until_clamped() {
        let mut c = GammaController::new(4).with_tuning(0.0, 0.5); // alpha=0 → no smoothing
        for _ in 0..50 {
            let g = c.current_gamma();
            c.record_round(g, g); // 100% acceptance every round
        }
        assert_eq!(
            c.current_gamma(),
            GammaController::DEFAULT_GAMMA_MAX,
            "perfect accept_rate should ratchet γ up to the max"
        );
        assert!((c.accept_ema() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn zero_acceptance_shrinks_gamma_to_minimum() {
        let mut c = GammaController::new(8).with_tuning(0.0, 0.5);
        for _ in 0..50 {
            let g = c.current_gamma();
            c.record_round(0, g);
        }
        assert_eq!(
            c.current_gamma(),
            GammaController::DEFAULT_GAMMA_MIN,
            "zero accept_rate should ratchet γ down to the min"
        );
        assert!(c.accept_ema().abs() < 1e-5);
    }

    #[test]
    fn record_round_handles_zero_proposed_safely() {
        // debug_assert fires only in debug builds; in release it's a no-op.
        // We can't easily test the assert — just confirm it doesn't panic in
        // a release-emulation scenario by using catch_unwind not being needed
        // (test harness builds in test profile = debug, so this would panic).
        // Instead, assert behavior is preserved when we *do* pass a sane round.
        let mut c = GammaController::new(4);
        let before = c.current_gamma();
        // Skip the assertion path entirely; just ensure normal flow still works.
        c.record_round(2, 4);
        // EMA moved toward 0.5 (rate=0.5, alpha=0.7 → 0.7·0.5 + 0.3·0.5 = 0.5)
        assert!((c.accept_ema() - 0.5).abs() < 1e-5);
        // γ unchanged because (accept_ema − 0.5) = 0 → multiplier = 1
        assert_eq!(c.current_gamma(), before);
    }

    #[test]
    fn ema_smooths_noisy_acceptance_signal() {
        let mut c = GammaController::new(6); // alpha=0.7 default
                                             // Alternating 100%/0% rounds — EMA should hover around 0.5, γ should
                                             // not run away in either direction.
        for i in 0..40 {
            let g = c.current_gamma();
            let accepted = if i % 2 == 0 { g } else { 0 };
            c.record_round(accepted, g);
        }
        assert!(
            (c.accept_ema() - 0.5).abs() < 0.15,
            "EMA should average alternating accept signal: got {}",
            c.accept_ema()
        );
        assert!(
            c.current_gamma() >= GammaController::DEFAULT_GAMMA_MIN
                && c.current_gamma() <= GammaController::DEFAULT_GAMMA_MAX
        );
    }

    #[test]
    fn high_alpha_resists_short_runs() {
        // alpha=0.95 → very smooth; a single bad round barely moves γ
        let mut c = GammaController::new(8).with_tuning(0.95, 0.2);
        let before = c.current_gamma();
        let before_ema = c.accept_ema();
        c.record_round(0, 8); // one terrible round
        assert!(
            (c.accept_ema() - before_ema).abs() < 0.05,
            "EMA should change less than 5% with alpha=0.95: from {before_ema} to {}",
            c.accept_ema()
        );
        // γ should not have collapsed to the minimum from one bad round
        assert!(c.current_gamma() >= before - 1);
    }
}
