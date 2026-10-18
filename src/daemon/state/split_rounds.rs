//! How many tokens each round of a split this node drives has been yielding,
//! per model — what the planner divides a split's per-token costs by
//! (FUTURE_WORK #3 (d)).
//!
//! Routing charges a split's per-token costs (every segment's compute, every
//! hop) once per pass of the reply. A split driven from here rarely makes one
//! pass per token: the guess-check paths (`pipeline::dsd`, its continuous
//! stream, the n-gram loop) send several guessed tokens through the segments
//! in ONE round and keep those the model agrees with. On this node's own rigs
//! the splits with a drafter ran in 0.39 of their price (n=25, 2026-10-15) —
//! the gain the price never saw.
//!
//! **What is counted is what happened**, whichever path ran: the rounds a
//! reply's decode sent through the first segment (`RequestTrace::
//! note_decode_round`, at the three places a round leaves) and the tokens it
//! produced, recorded once per finished attempt by `PipelineExecutor::execute`.
//! The prompt pass is counted as a round too, because it yields the reply's
//! first token and the price counts that token among its passes — so a plain
//! loop reads exactly 1 token per round, never a little more. vLLM reports the
//! same quantity as its "mean acceptance length" (tokens per verification step,
//! the bonus token included); SmartSpec (Liu et al. 2024) and SGLang's adaptive
//! speculation steer by a running average of it.
//!
//! **A ratio of sums, not a mean of ratios**: the price is an expected time, a
//! round costs the same wherever it falls, so the expected rounds of a reply
//! are its tokens over the tokens a round yields across ALL recent tokens — a
//! two-token reply's ratio must not weigh as much as a 300-token one's.
//!
//! What this does not see, each named so it can be checked:
//! - a round that checks several guesses costs more than a plain step where a
//!   segment computes on a processor (SmartSpec's linear batch-time model) —
//!   the γ controller (`dsd_controller::best_gamma_for_check`) holds such a
//!   check short, and stops guessing where it does not pay;
//! - the continuous stream sends its next chunk before the last is answered,
//!   so its rounds overlap and the gain is larger than counted;
//! - the drafter's own time per guess, small beside a hop.

use dashmap::DashMap;
use std::collections::VecDeque;

/// Splits remembered per model: as many as `reply_lengths` remembers replies,
/// for the same reason.
pub const RECENT_SPLITS: usize = 32;

/// Splits a model must have finished before its figure is trusted over one
/// token a round. Below it one reply would decide the next route.
pub const SPLITS_BEFORE_TRUSTED: usize = 4;

/// At most this many models are tracked.
const MAX_MODELS: usize = 256;

#[derive(Default)]
pub struct SplitRounds {
    /// (tokens, rounds) of each recent split, oldest first.
    by_model: DashMap<String, VecDeque<(u32, u32)>>,
}

impl SplitRounds {
    /// A split of `model` this node drove produced `tokens` in `rounds`, the
    /// prompt pass included. Either at zero tells nothing and is not recorded.
    pub fn record(&self, model: &str, tokens: u32, rounds: u32) {
        if tokens == 0 || rounds == 0 {
            return;
        }
        if !self.by_model.contains_key(model) && self.by_model.len() >= MAX_MODELS {
            return;
        }
        let mut recent = self.by_model.entry(model.to_string()).or_default();
        if recent.len() >= RECENT_SPLITS {
            recent.pop_front();
        }
        recent.push_back((tokens, rounds));
    }

    /// The tokens a round of `model`'s recent splits yielded, or `None` until
    /// [`SPLITS_BEFORE_TRUSTED`] have finished. Never below one: a round that
    /// yielded nothing (a stream's chunk sent on guesses that were then
    /// refused) overlapped another, and must not price a split above the one
    /// round a token it was priced at before.
    pub fn tokens_per_round(&self, model: &str) -> Option<f32> {
        let recent = self.by_model.get(model)?;
        if recent.len() < SPLITS_BEFORE_TRUSTED {
            return None;
        }
        let (tokens, rounds) = recent
            .iter()
            .fold((0u64, 0u64), |(t, r), &(tokens, rounds)| {
                (t + u64::from(tokens), r + u64::from(rounds))
            });
        Some((tokens as f64 / rounds.max(1) as f64).max(1.0) as f32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The figure is the recent splits' tokens over their rounds — a long
    /// reply weighs as its tokens do — once enough have finished.
    #[test]
    fn a_models_splits_yield_their_tokens_over_their_rounds() {
        let rounds = SplitRounds::default();
        for _ in 0..SPLITS_BEFORE_TRUSTED - 1 {
            rounds.record("m", 300, 100);
        }
        assert_eq!(rounds.tokens_per_round("m"), None, "too few to trust");
        // A two-token reply in two rounds (the prompt pass and one step).
        rounds.record("m", 2, 2);
        let each = rounds.tokens_per_round("m").expect("trusted");
        assert!((each - 902.0 / 302.0).abs() < 1e-4, "{each}");
        assert_eq!(rounds.tokens_per_round("other"), None);
        // Nothing to learn from an empty reply or an uncounted one.
        rounds.record("m", 0, 5);
        rounds.record("m", 5, 0);
        assert!((rounds.tokens_per_round("m").unwrap() - each).abs() < 1e-6);
    }

    /// A plain loop reads exactly one token a round, and a stream whose
    /// chunks were refused never reads below it.
    #[test]
    fn a_round_is_never_priced_at_less_than_a_token() {
        let rounds = SplitRounds::default();
        for _ in 0..SPLITS_BEFORE_TRUSTED {
            rounds.record("plain", 40, 40);
            rounds.record("wasted", 10, 14);
        }
        assert_eq!(rounds.tokens_per_round("plain"), Some(1.0));
        assert_eq!(rounds.tokens_per_round("wasted"), Some(1.0));
    }

    #[test]
    fn only_the_most_recent_splits_count() {
        let rounds = SplitRounds::default();
        for _ in 0..RECENT_SPLITS {
            rounds.record("m", 400, 100);
        }
        for _ in 0..RECENT_SPLITS {
            rounds.record("m", 60, 60);
        }
        assert_eq!(rounds.tokens_per_round("m"), Some(1.0));
    }
}
