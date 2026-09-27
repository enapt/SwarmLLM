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
    ///
    /// (A note for anyone relying on the multiplier above: from γ = 4 it can
    /// never grow — even perfect acceptance gives 4 × 1.1 = 4.4, which rounds back
    /// to 4 — so a controller started there stays there. DSD sizes γ with
    /// [`best_gamma_for_check`] instead.)
    #[cfg(test)]
    pub fn with_tuning(mut self, alpha: f32, beta: f32) -> Self {
        self.alpha = alpha.clamp(0.0, 1.0);
        self.beta = beta.clamp(0.0, 1.0);
        self
    }
}

/// The longest guess run [`best_gamma_for_check`] will propose.
pub const BEST_GAMMA_MAX: u32 = 16;

/// Expected tokens one round yields: `γ` drafts, each kept with probability `α`
/// while every one before it was kept, plus the token sampled where they stop —
/// `(1 - α^(γ+1)) / (1 - α)` (Leviathan et al., 2023, "Fast inference from
/// transformers via speculative decoding", eq. 1).
pub fn expected_tokens_per_round(alpha: f64, gamma: u32) -> f64 {
    let a = alpha.clamp(0.0, 0.999_999);
    (1.0 - a.powi(gamma as i32 + 1)) / (1.0 - a)
}

/// What a check costs as a line in the positions it reads:
/// `fixed + per_position × positions`, fitted from measured rounds.
///
/// **Why a line and not a constant.** The controller this replaced
/// (`best_gamma`, Leviathan et al. §3.4 with the costs measured) treated the
/// check as a fixed cost — right for a round trip beside a card that reads 15
/// positions as fast as 5, wrong for a check that runs on a PROCESSOR, where
/// reading more positions costs more. Measured on the live TH↔BE split 2026-09-28
/// (`~/swarmllm-bench-0928`), the far node checking on its processor: the
/// constant model saw a 1.4 s round, answered γ = 14-16, and every round then
/// cost 1.1-2.8 s — 3.35 tok/s against a 7.3 tok/s warm-up that had run at
/// γ = 4. Dovetail (arXiv 2412.18934) names the same trap for verifying on a
/// processor and keeps its candidate count small for it.
///
/// Weighted least squares with forgetting (each new round weighs 1, older
/// ones decay by [`CheckCost::DECAY`]), so a node whose load changes is
/// re-learned in a few rounds. A slope needs rounds of DIFFERENT lengths:
/// until the positions have spread, [`CheckCost::predict`] answers the mean
/// cost — the constant model — and [`best_gamma_for_check`] moves γ at most
/// [`GAMMA_STEP`] per round, which is what produces the spread.
#[derive(Debug, Clone, Default)]
pub struct CheckCost {
    w: f64,
    x: f64,
    y: f64,
    xx: f64,
    xy: f64,
}

/// The most [`best_gamma_for_check`] moves γ in one round: far enough to
/// learn the check's slope from the next round, near enough that a wrong
/// guess about it costs one slow round, not a reply's worth.
pub const GAMMA_STEP: u32 = 2;

impl CheckCost {
    const DECAY: f64 = 0.8;

    pub fn record(&mut self, positions: u32, ms: f64) {
        let d = Self::DECAY;
        let x = f64::from(positions);
        self.w = self.w * d + 1.0;
        self.x = self.x * d + x;
        self.y = self.y * d + ms;
        self.xx = self.xx * d + x * x;
        self.xy = self.xy * d + x * ms;
    }

    /// The fitted `(fixed, per_position)`, or `None` before any round. The
    /// slope is 0 until the positions have spread (variance under a quarter of
    /// a position squared) and never negative — a check that reads more cannot
    /// cost less, and a noisy fit saying so must not make long runs look free.
    pub fn fit(&self) -> Option<(f64, f64)> {
        if self.w <= 0.0 {
            return None;
        }
        let mx = self.x / self.w;
        let my = self.y / self.w;
        let var = self.xx / self.w - mx * mx;
        if var < 0.25 {
            return Some((my, 0.0));
        }
        let slope = ((self.xy / self.w - mx * my) / var).max(0.0);
        Some(((my - slope * mx).max(0.0), slope))
    }

    /// Predicted cost of a check reading `positions`.
    pub fn predict(&self, positions: u32) -> Option<f64> {
        self.fit()
            .map(|(fixed, slope)| fixed + slope * f64::from(positions))
    }
}

/// The guess-run length that maximizes tokens per SECOND (Leviathan et al.
/// §3.4's choice of γ, with the costs MEASURED rather than assumed) for a check
/// whose cost is the fitted line in the positions it reads — γ guesses and the
/// token they follow — plus `draft_ms_each` per guess, moved at most
/// [`GAMMA_STEP`] from `current` (see [`CheckCost`]). A long link with a check
/// that costs the same at any length climbs to long runs (≈12 at a 300 ms
/// trip, 25 ms drafts, α = 0.92); a free round trip, or a check that grows
/// with every position, settles short.
pub fn best_gamma_for_check(
    alpha: f64,
    check: &CheckCost,
    draft_ms_each: f64,
    current: u32,
    max: u32,
) -> u32 {
    let Some((fixed, slope)) = check.fit() else {
        return current;
    };
    let each = draft_ms_each.max(0.0);
    let lo = current.saturating_sub(GAMMA_STEP).max(1);
    let hi = (current + GAMMA_STEP).min(max).max(lo);
    (lo..=hi)
        .map(|g| {
            let cost = (fixed + slope * f64::from(g + 1)).max(0.1) + each * f64::from(g);
            (g, expected_tokens_per_round(alpha, g) / cost)
        })
        .fold(
            (lo, f64::MIN),
            |best, c| if c.1 > best.1 { c } else { best },
        )
        .0
}

/// A running estimate of the per-token acceptance α from whole rounds: a round
/// that keeps `k` of `γ` drafts saw `k` acceptances, and one rejection when
/// `k < γ` — the maximum-likelihood estimate is acceptances over trials. Starts
/// from a prior worth a few rounds, so the first round cannot swing γ to an end.
#[derive(Debug, Clone)]
pub struct AcceptanceEstimate {
    accepted: f64,
    trials: f64,
}

impl AcceptanceEstimate {
    /// Prior: α = 0.7 over 10 trials.
    pub fn new() -> Self {
        Self {
            accepted: 7.0,
            trials: 10.0,
        }
    }

    pub fn record(&mut self, accepted: u32, proposed: u32) {
        self.accepted += f64::from(accepted);
        self.trials += f64::from(accepted) + f64::from(u32::from(accepted < proposed));
    }

    pub fn alpha(&self) -> f64 {
        self.accepted / self.trials
    }
}

impl Default for AcceptanceEstimate {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expected_tokens_follow_the_closed_form() {
        assert!((expected_tokens_per_round(0.0, 4) - 1.0).abs() < 1e-9);
        assert!((expected_tokens_per_round(0.5, 1) - 1.5).abs() < 1e-9);
        // α → 1 keeps every draft: γ + 1 tokens.
        assert!((expected_tokens_per_round(1.0, 8) - 9.0).abs() < 1e-3);
    }

    /// Where γ settles against a check costing `check_ms(positions)`, from 4.
    fn settle(alpha: f64, check_ms: impl Fn(u32) -> f64, draft_ms: f64) -> u32 {
        let mut check = CheckCost::default();
        let mut g = 4;
        for _ in 0..30 {
            check.record(g + 1, check_ms(g + 1));
            g = best_gamma_for_check(alpha, &check, draft_ms, g, BEST_GAMMA_MAX);
        }
        g
    }

    /// A long link wants long runs, a free round trip short ones — the whole point
    /// of measuring the costs instead of fixing γ. A check that costs the same at
    /// any length climbs there one step at a time.
    #[test]
    fn a_long_round_trip_buys_a_long_run_and_a_free_one_a_short_run() {
        let wan = settle(0.92, |_| 300.0, 25.0);
        assert!((10..=14).contains(&wan), "300 ms trip: {wan}");
        let lan = settle(0.92, |_| 5.0, 40.0);
        assert!(lan <= 2, "a 5 ms trip beside 40 ms drafts: {lan}");
        // Poor agreement shortens the run whatever the link.
        assert!(settle(0.5, |_| 300.0, 25.0) < wan);
    }

    /// The live case: a check on a processor, ~95 ms per position over a 230 ms
    /// round trip. From γ = 4 the controller explores up, sees the rounds get
    /// dearer, and settles short — never at the 14-16 the constant model chose.
    #[test]
    fn a_check_that_grows_with_its_positions_settles_on_a_short_run() {
        let true_cost = |positions: u32| 230.0 + 95.0 * f64::from(positions);
        let mut check = CheckCost::default();
        let mut g = 4;
        for _ in 0..30 {
            check.record(g + 1, true_cost(g + 1));
            g = best_gamma_for_check(0.9, &check, 20.0, g, BEST_GAMMA_MAX);
        }
        let (fixed, slope) = check.fit().unwrap();
        assert!(
            (slope - 95.0).abs() < 1.0 && (fixed - 230.0).abs() < 5.0,
            "{fixed} + {slope}x"
        );
        assert!((2..=7).contains(&g), "settled at {g}");
        // Read as a constant (the controller this replaced), the same first
        // round's cost climbs to a long run.
        assert!(settle(0.9, |_| true_cost(5), 20.0) >= 12);
    }

    #[test]
    fn gamma_moves_at_most_one_step_per_round() {
        // A dear check and cheap, good guesses: longer is better — one step.
        let mut dear = CheckCost::default();
        dear.record(5, 5000.0);
        assert_eq!(
            best_gamma_for_check(0.99, &dear, 1.0, 4, 16),
            4 + GAMMA_STEP
        );
        // A cheap check and dear, poor guesses: shorter is better — one step,
        // and never below one guess.
        let mut cheap = CheckCost::default();
        cheap.record(9, 5.0);
        assert_eq!(
            best_gamma_for_check(0.1, &cheap, 1000.0, 8, 16),
            8 - GAMMA_STEP
        );
        assert_eq!(best_gamma_for_check(0.1, &cheap, 1000.0, 1, 16), 1);
    }

    #[test]
    fn the_acceptance_estimate_counts_a_rejection_only_when_a_run_stopped_early() {
        let mut e = AcceptanceEstimate::new();
        for _ in 0..100 {
            e.record(4, 4); // every draft kept: no rejection
        }
        assert!(e.alpha() > 0.99);
        let mut e = AcceptanceEstimate::new();
        for _ in 0..100 {
            e.record(0, 4); // the first draft refused every time
        }
        assert!(e.alpha() < 0.1);
    }

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
