//! How many tokens to guess per round of speculation across computers
//! (Decentralized Speculative Decoding, `pipeline::dsd`; arXiv 2511.11733 /
//! 2511.21669).
//!
//! γ trades the per-round cost of guessing and checking against the round
//! trips it saves, and the best γ depends on how often the guesses are kept
//! and on what a round actually costs — both measured here, per request:
//! [`AcceptanceEstimate`] for how often, [`CheckCost`] for what a check of `n`
//! positions costs, [`best_gamma_for_check`] for the choice (Leviathan et al.
//! 2023 §3.4). Two controllers came before it and are gone: a multiplicative
//! "dynamic window" that rounding pinned at γ = 4, and a constant-cost choice
//! that drove a processor-bound check to γ = 14-16 (2026-09-28).

/// The longest guess run [`best_gamma_for_check`] will propose — the same cap
/// every speculating path reads through `InferenceConfig::guesses_per_check`.
pub const BEST_GAMMA_MAX: u32 = crate::config::MAX_GUESSES_PER_CHECK;

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
    // A starting γ from configuration may lie outside [1, max]; the answer
    // never does — past `max` a verify can exceed what a peer accepts on the
    // wire (`protocol::MAX_DRAFT_TOKENS`).
    let max = max.max(1);
    let current = current.clamp(1, max);
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

    /// A configured starting γ past the cap is brought inside it at once, with
    /// or without a measurement to go on.
    #[test]
    fn a_starting_gamma_past_the_cap_is_clamped() {
        assert_eq!(
            best_gamma_for_check(0.9, &CheckCost::default(), 10.0, 40, 16),
            16
        );
        let mut check = CheckCost::default();
        check.record(41, 300.0);
        assert!(best_gamma_for_check(0.99, &check, 1.0, 40, 16) <= 16);
        assert_eq!(
            best_gamma_for_check(0.9, &CheckCost::default(), 10.0, 0, 16),
            1
        );
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
}
