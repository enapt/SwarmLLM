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
///
/// **One stalled check is held back, not fitted** (FUTURE_WORK #244). On the
/// 12h split rig (2026-10-12) one streamed check of 3 positions took 4.5 s
/// among ~40 ms ones — the forward waited ~4.4 s before it was even sent — and
/// the fit became 0 ms fixed + 468 ms a position: guessing never pays on a line
/// through the origin, so γ went to 0 for the rest of the reply and, remembered,
/// for the next request. A point at a position count the fit has seen little
/// of has the leverage to do that, so capping its value is not enough. A check
/// over [`Self::OUTLIER_FACTOR`] times what it should cost waits for the next
/// round: a normal one drops it as a stall, a second slow one records both — a
/// lasting change is learned one round late. The drafter's cost had the same
/// fault and went to a median ([`RecentMedian`]); robust filters bound one
/// sample's influence the same way (Ruckdeschel's clipped RLS, arXiv 1204.3358),
/// and servers that time their checks take medians over repeats (D-cut, arXiv
/// 2607.14647). `SWARMLLM_DSD_CHECK_OUTLIERS=keep` fits every check as it came.
#[derive(Debug, Clone, Default)]
pub struct CheckCost {
    w: f64,
    x: f64,
    y: f64,
    xx: f64,
    xy: f64,
    /// Rounds fitted — nothing is held back before [`Self::HOLD_AFTER`].
    rounds: u32,
    /// A check held back as a possible stall: `(positions, ms)`.
    held: Option<(u32, f64)>,
    /// Checks ever held back — see [`Self::held_back`].
    held_back: u32,
}

/// The most [`best_gamma_for_check`] moves γ in one round: far enough to
/// learn the check's slope from the next round, near enough that a wrong
/// guess about it costs one slow round, not a reply's worth.
pub const GAMMA_STEP: u32 = 2;

impl CheckCost {
    const DECAY: f64 = 0.8;
    /// How many times its expected cost a check must take to be held back.
    pub const OUTLIER_FACTOR: f64 = 3.0;
    /// Rounds the fit needs before it can call a check an outlier.
    const HOLD_AFTER: u32 = 3;

    pub fn record(&mut self, positions: u32, ms: f64) {
        self.record_with(positions, ms, hold_outlying_checks());
    }

    fn record_with(&mut self, positions: u32, ms: f64, hold: bool) {
        if hold && self.rounds >= Self::HOLD_AFTER && self.is_outlier(positions, ms) {
            match self.held.take() {
                None => {
                    self.held = Some((positions, ms));
                    self.held_back = self.held_back.saturating_add(1);
                    return;
                }
                // Two slow rounds running: the cost changed. Fit both.
                Some((p, held_ms)) => self.fit_round(p, held_ms),
            }
        } else {
            self.held = None;
        }
        self.fit_round(positions, ms);
    }

    /// Slower than [`Self::OUTLIER_FACTOR`] times its expected cost — the
    /// line's prediction, or the mean cost scaled by positions, whichever is
    /// higher: a check the line has not seen this long yet may cost
    /// proportionally more without being a stall (a processor check of 25 ms +
    /// 95 ms a position costs 3.4× more at 4 positions than at 1).
    fn is_outlier(&self, positions: u32, ms: f64) -> bool {
        let Some(line) = self.predict(positions) else {
            return false;
        };
        let proportional = (self.y / self.w) * f64::from(positions) / (self.x / self.w).max(1.0);
        ms > Self::OUTLIER_FACTOR * line.max(proportional)
    }

    fn fit_round(&mut self, positions: u32, ms: f64) {
        let d = Self::DECAY;
        let x = f64::from(positions);
        self.w = self.w * d + 1.0;
        self.x = self.x * d + x;
        self.y = self.y * d + ms;
        self.xx = self.xx * d + x * x;
        self.xy = self.xy * d + x * ms;
        self.rounds = self.rounds.saturating_add(1);
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

    /// Checks held back as possible stalls since this fit began — it is
    /// carried from request to request ([`Learned::check`]), so a request
    /// reports the difference across itself.
    pub fn held_back(&self) -> u32 {
        self.held_back
    }
}

/// `SWARMLLM_DSD_CHECK_OUTLIERS=keep` fits every check as it came, the rule
/// before [`CheckCost::OUTLIER_FACTOR`], for A/B inside one binary.
fn hold_outlying_checks() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        !matches!(
            std::env::var("SWARMLLM_DSD_CHECK_OUTLIERS").ok().as_deref(),
            Some("keep")
        )
    })
}

/// The guess-run length that maximizes tokens per SECOND (Leviathan et al.
/// §3.4's choice of γ, with the costs MEASURED rather than assumed) for a check
/// whose cost is the fitted line in the positions it reads — γ guesses and the
/// token they follow — plus `draft_ms_each` per guess, moved at most
/// [`GAMMA_STEP`] from `current` (see [`CheckCost`]). A long link with a check
/// that costs the same at any length climbs to long runs (≈12 at a 300 ms
/// trip, 25 ms drafts, α = 0.92); a free round trip, or a check that grows
/// with every position, settles short.
///
/// **Zero is an answer**: a round that guesses nothing — the token already
/// sampled checked alone, a plain decode step through the same path — where
/// guessing costs more than the round trips it saves. SmartSpec (arXiv
/// 2406.14066, the vLLM team) chooses each request's speculation length from
/// zero up by estimated goodput for the same reason: speculation that does not
/// pay makes a reply SLOWER. Measured 2026-09-29 on a two-node split with ~0 ms
/// between the machines and the drafter on the processor (150-185 ms a guess):
/// 6.8 tok/s where plain decoding ran 38-42 — this function could not go below
/// one guess. Zero-guess rounds still feed [`CheckCost`], so a round trip that
/// grows brings the guesses back.
pub fn best_gamma_for_check(
    alpha: f64,
    check: &CheckCost,
    draft_ms_each: f64,
    current: u32,
    max: u32,
) -> u32 {
    // A starting γ from configuration may lie outside [0, max]; the answer
    // never does — past `max` a verify can exceed what a peer accepts on the
    // wire (`protocol::MAX_DRAFT_TOKENS`).
    let max = max.max(1);
    let current = current.min(max);
    let Some((fixed, slope)) = check.fit() else {
        return current;
    };
    let lo = current.saturating_sub(GAMMA_STEP);
    let hi = (current + GAMMA_STEP).min(max).max(lo);
    best_in(lo..=hi, alpha, fixed, slope, draft_ms_each)
}

/// [`best_gamma_for_check`] over the WHOLE range `0..=max` — for a request that
/// measured its costs without walking γ: a streamed one (`pipeline::dsd_stream`)
/// guesses a fixed chunk and never moves γ itself, so what it remembers
/// (`remember`) must be the rounds' verdict on those costs, or a split where
/// guessing does not pay would stream guesses for ever — zero is what makes the
/// next request on the same machines step aside (`steps_aside`; before zero
/// existed, guessing ran 6× slower than plain on a near split, FUTURE_WORK #140).
/// `fallback` before any check was timed.
pub fn best_gamma_overall(
    alpha: f64,
    check: &CheckCost,
    draft_ms_each: f64,
    fallback: u32,
    max: u32,
) -> u32 {
    match check.fit() {
        Some((fixed, slope)) => best_in(0..=max.max(1), alpha, fixed, slope, draft_ms_each),
        None => fallback.min(max.max(1)),
    }
}

/// The γ in `range` with the most tokens per millisecond: a round costs the
/// fitted check over γ + 1 positions plus `draft_ms_each` per guess. Ties go
/// to the smaller γ.
fn best_in(
    range: std::ops::RangeInclusive<u32>,
    alpha: f64,
    fixed: f64,
    slope: f64,
    draft_ms_each: f64,
) -> u32 {
    let each = draft_ms_each.max(0.0);
    let lo = *range.start();
    range
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

/// What one request's speculation learned — how often the guesses were kept,
/// what a check cost, what a guess cost, and the γ it settled on — handed to
/// the next request on the same model and machines.
///
/// Each request used to learn from nothing: its first round guessed at the
/// configured length and γ moved [`GAMMA_STEP`] a round from there, so where
/// guessing does not pay every request spent its first few rounds (a second,
/// with a drafter on the processor) proving it again.
/// The median of a cost's last [`RecentMedian::WINDOW`] samples — what the
/// guess-count choice reads for the drafter's time per guess.
///
/// It was an average (0.7 old + 0.3 new), and ONE slow call moved it: on the
/// split rig (2026-09-30) a few 68-76 ms drafting calls among 12 ms ones
/// took it to 41-85 ms a guess, `best_gamma_for_check` then chose γ = 0, a
/// round that guesses nothing records no new cost, so the figure could not
/// recover for the rest of the request — and, remembered, for ten minutes.
/// A median ignores a few outliers either way.
#[derive(Debug, Clone, Default)]
pub struct RecentMedian {
    samples: std::collections::VecDeque<f64>,
}

impl RecentMedian {
    pub const WINDOW: usize = 8;

    /// Start from a remembered figure, if there is one.
    pub fn seeded(from: Option<f64>) -> Self {
        let mut m = Self::default();
        if let Some(v) = from {
            m.record(v);
        }
        m
    }

    pub fn record(&mut self, sample: f64) {
        if !sample.is_finite() {
            return;
        }
        if self.samples.len() == Self::WINDOW {
            self.samples.pop_front();
        }
        self.samples.push_back(sample);
    }

    /// Samples held, at most [`Self::WINDOW`].
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// The median (the lower middle of an even count), or `None` before any sample.
    pub fn median(&self) -> Option<f64> {
        if self.samples.is_empty() {
            return None;
        }
        let mut sorted: Vec<f64> = self.samples.iter().copied().collect();
        sorted.sort_by(f64::total_cmp);
        Some(sorted[(sorted.len() - 1) / 2])
    }
}

#[derive(Debug, Clone)]
pub struct Learned {
    pub acceptance: AcceptanceEstimate,
    pub check: CheckCost,
    pub draft_ms_each: Option<f64>,
    pub gamma: u32,
    /// When it was learned — see [`steps_aside`].
    pub at: std::time::Instant,
}

/// How long a "guessing does not pay here" verdict stands before speculation
/// is tried again on the same model and machines — links and loads change.
pub const REPROBE_AFTER: std::time::Duration = std::time::Duration::from_secs(600);

/// Should speculation stay out of this request altogether? Yes when the last
/// request on the same model and machines settled on γ = 0 within
/// [`REPROBE_AFTER`]: entering anyway loads the drafter and has it read the
/// prompt before the first token — +0.3 s of time to first token on every
/// request of a near split (2026-09-29), and the drafter's memory held for
/// nothing — to run plain rounds the ordinary loop runs as well.
pub fn steps_aside(learned: &Learned, now: std::time::Instant) -> bool {
    learned.gamma == 0 && now.duration_since(learned.at) < REPROBE_AFTER
}

/// The most distinct (model, machines) pairs remembered; past it the memory
/// starts again rather than growing with every plan a node ever made.
const LEARNED_CAPACITY: usize = 64;

fn learned_store() -> &'static std::sync::Mutex<std::collections::HashMap<String, Learned>> {
    static STORE: std::sync::OnceLock<
        std::sync::Mutex<std::collections::HashMap<String, Learned>>,
    > = std::sync::OnceLock::new();
    STORE.get_or_init(Default::default)
}

/// What the last request on `key` learned, if any. `key` names the model and
/// the machines of the plan, in order — a check's cost is theirs.
pub fn recall(key: &str) -> Option<Learned> {
    learned_store()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .get(key)
        .cloned()
}

/// Keep what this request learned for the next one on `key`.
pub fn remember(key: String, learned: Learned) {
    let mut store = learned_store().lock().unwrap_or_else(|e| e.into_inner());
    if store.len() >= LEARNED_CAPACITY && !store.contains_key(&key) {
        store.clear();
    }
    store.insert(key, learned);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Where guessing costs more than the round trips it saves, the controller
    /// stops guessing — the split of 2026-09-29: ~25 ms checks, a drafter on
    /// the processor at ~170 ms a guess, α ≈ 0.68.
    #[test]
    fn guessing_that_does_not_pay_goes_to_zero_and_comes_back_when_it_does() {
        let fast = settle(0.68, |p| 25.0 + 0.5 * f64::from(p), 170.0);
        assert_eq!(fast, 0, "a near link and a slow drafter: plain rounds");
        // From zero, a long round trip brings the guesses back (a plain round
        // still feeds the check's cost).
        let mut check = CheckCost::default();
        let mut g = 0;
        for _ in 0..10 {
            check.record(g + 1, 500.0);
            g = best_gamma_for_check(0.7, &check, 30.0, g, BEST_GAMMA_MAX);
        }
        assert!(g >= 2, "a 500 ms check with 30 ms drafts: {g}");
    }

    /// A streamed request walks no γ; what it remembers comes from the whole
    /// range at once. From the stream's default of 4 a near split must reach
    /// zero in ONE step — `best_gamma_for_check` moves at most `GAMMA_STEP` —
    /// and a long link must keep guessing.
    #[test]
    fn a_streamed_request_remembers_the_verdict_over_the_whole_range() {
        // Near: the 2026-09-29 split (25 ms checks, a processor drafter at 170 ms).
        let mut near = CheckCost::default();
        for p in [1, 2, 3, 3, 2, 1, 3] {
            near.record(p, 25.0 + 0.5 * f64::from(p));
        }
        assert_eq!(best_gamma_overall(0.68, &near, 170.0, 4, BEST_GAMMA_MAX), 0);
        assert!(
            best_gamma_for_check(0.68, &near, 170.0, 4, BEST_GAMMA_MAX) > 0,
            "the step-wise search cannot reach zero from 4 in one call — why this exists"
        );
        // Far: TH↔IT measured 2026-10-01 (checks ~330 ms fixed + ~22 ms a
        // position, 8 ms guesses on the card, α ≈ 0.68).
        let mut far = CheckCost::default();
        for p in [4, 4, 3, 4, 2, 4] {
            far.record(p, 330.0 + 22.0 * f64::from(p));
        }
        assert!(best_gamma_overall(0.68, &far, 8.0, 4, BEST_GAMMA_MAX) >= 2);
        // Nothing timed: the stream's own starting point, clamped.
        assert_eq!(
            best_gamma_overall(0.7, &CheckCost::default(), 8.0, 4, 16),
            4
        );
        assert_eq!(
            best_gamma_overall(0.7, &CheckCost::default(), 8.0, 40, 16),
            16
        );
    }

    #[test]
    fn what_a_request_learned_is_there_for_the_next_one() {
        let key = format!("m|test-{}", std::process::id());
        assert!(recall(&key).is_none());
        let mut check = CheckCost::default();
        check.record(2, 25.0);
        remember(
            key.clone(),
            Learned {
                acceptance: AcceptanceEstimate::new(),
                check,
                draft_ms_each: Some(170.0),
                gamma: 0,
                at: std::time::Instant::now(),
            },
        );
        let got = recall(&key).expect("remembered");
        assert_eq!(got.gamma, 0);
        assert_eq!(got.draft_ms_each, Some(170.0));
        assert!(got.check.fit().is_some());
    }

    #[test]
    fn a_zero_verdict_keeps_speculation_out_until_it_is_worth_asking_again() {
        let t0 = std::time::Instant::now();
        let learned = |gamma| Learned {
            acceptance: AcceptanceEstimate::new(),
            check: CheckCost::default(),
            draft_ms_each: Some(170.0),
            gamma,
            at: t0,
        };
        assert!(steps_aside(
            &learned(0),
            t0 + std::time::Duration::from_secs(5)
        ));
        assert!(
            !steps_aside(&learned(2), t0),
            "guessing that paid keeps its place"
        );
        assert!(
            !steps_aside(&learned(0), t0 + REPROBE_AFTER),
            "after a while the link may have changed: probe again"
        );
    }

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
        // Zero is a γ of its own now (a plain round), kept until a
        // measurement says otherwise.
        assert_eq!(
            best_gamma_for_check(0.9, &CheckCost::default(), 10.0, 0, 16),
            0
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
        // down to no guesses at all (a plain round) where none pays.
        let mut cheap = CheckCost::default();
        cheap.record(9, 5.0);
        assert_eq!(
            best_gamma_for_check(0.1, &cheap, 1000.0, 8, 16),
            8 - GAMMA_STEP
        );
        assert_eq!(best_gamma_for_check(0.1, &cheap, 1000.0, 1, 16), 0);
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

    /// Two slow calls among fast ones do not make guessing look dear — the
    /// average they replaced went from 12 to 41 ms a guess on exactly that.
    #[test]
    fn a_few_slow_drafting_calls_do_not_move_the_draft_cost() {
        let mut cost = RecentMedian::seeded(Some(12.0));
        for x in [11.5, 70.0, 12.5, 76.0, 12.0] {
            cost.record(x);
        }
        let m = cost.median().unwrap();
        assert!((11.0..=13.0).contains(&m), "median {m}");
        // It does follow a lasting change.
        for _ in 0..RecentMedian::WINDOW {
            cost.record(30.0);
        }
        assert_eq!(cost.median(), Some(30.0));
        assert_eq!(RecentMedian::default().median(), None);
    }

    /// The streamed check of the 12h rig (2026-10-12): 1-2 rows a check at
    /// ~35 ms + 10 ms a row, then ONE of 3 rows that waited 4.5 s to be sent.
    /// Fitted, it made the line 0 + ~2 s a position and the stream's verdict
    /// zero guesses — remembered for the next request too.
    #[test]
    fn one_stalled_check_does_not_end_the_guessing() {
        let rows = [1, 2, 1, 1, 2, 1, 2, 1, 1, 2, 1, 1];
        let run = |hold: bool| {
            let mut check = CheckCost::default();
            for p in rows {
                check.record_with(p, 35.0 + 10.0 * f64::from(p), hold);
            }
            check.record_with(3, 4524.0, hold);
            check
        };
        let kept = run(false);
        assert_eq!(
            best_gamma_overall(0.622, &kept, 7.2, 3, BEST_GAMMA_MAX),
            0,
            "fitting the stall is what the rig saw"
        );
        assert_eq!(kept.held_back(), 0);
        let held = run(true);
        assert_eq!(held.held_back(), 1);
        let (fixed, slope) = held.fit().unwrap();
        assert!(fixed > 30.0 && slope < 20.0, "{fixed} + {slope}x");
        assert!(best_gamma_overall(0.622, &held, 7.2, 3, BEST_GAMMA_MAX) > 0);
        // The next normal round drops it.
        let mut after = held.clone();
        after.record_with(2, 55.0, true);
        assert!(after.held.is_none());
        assert!(best_gamma_overall(0.622, &after, 7.2, 3, BEST_GAMMA_MAX) > 0);
    }

    /// A cost that really changed is learned one round late, not never.
    #[test]
    fn a_lasting_slowdown_is_learned_one_round_late() {
        let mut check = CheckCost::default();
        for p in [4, 3, 4, 2, 4, 3, 4, 4] {
            check.record_with(p, 35.0 + 10.0 * f64::from(p), true);
        }
        let slow = |p: u32| 10.0 * (35.0 + 10.0 * f64::from(p));
        check.record_with(4, slow(4), true);
        assert!(check.held.is_some(), "one slow round could be a stall");
        let before = check.predict(4).unwrap();
        check.record_with(3, slow(3), true);
        assert!(check.held.is_none());
        assert!(
            check.predict(4).unwrap() > 4.0 * before,
            "both slow rounds fitted: {before} -> {}",
            check.predict(4).unwrap()
        );
    }

    /// A long check on a processor costs several times a short one — never a
    /// stall. 25 ms + 95 ms a position: 3.4× more at 4 positions than at 1,
    /// met first by a line that has only seen 1-position checks (no slope yet),
    /// then alternating.
    #[test]
    fn a_check_that_costs_more_for_more_positions_is_never_held() {
        let mut check = CheckCost::default();
        for i in 0..20 {
            let p = if i < 6 || i % 2 == 0 { 1 } else { 4 };
            check.record_with(p, 25.0 + 95.0 * f64::from(p), true);
            assert!(check.held.is_none(), "round {i} held");
        }
        let (fixed, slope) = check.fit().unwrap();
        assert!(
            (fixed - 25.0).abs() < 1.0 && (slope - 95.0).abs() < 1.0,
            "{fixed} + {slope}x"
        );
    }
}
