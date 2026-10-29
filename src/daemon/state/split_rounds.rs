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
//! note_decode_round`, at the places a round leaves) and the tokens it
//! produced, recorded once per finished attempt by `PipelineExecutor::execute`.
//! A path whose forwards OVERLAP — the flowing tree (`pipeline::dsd_flow`) and
//! the continuous stream (`pipeline::dsd_stream`) — counts, when it ends, the
//! round trips it waited ([`round_trips_waited`]), not one a forward.
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
//! - the figure is per MODEL, every shape and split of it together: a flow
//!   whose far card is saturated (a lopsided split) yields fewer tokens a
//!   round than one at an even split, and the price of either reads both;
//! - the drafter's own time per guess, small beside a hop.
//!
//! Kept across restarts (`recent_by_model`): a restart priced every split at
//! one round a token until four had finished.

use super::recent_by_model::RecentByModel;
use crate::storage::db::Database;

/// What a stored round means. 1: the prompt pass is a round, and a flow or a
/// stream counts the round trips it waited (`round_trips_waited`, v0.3.261).
/// Bump it where a round comes to be counted another way, so a figure
/// counted the old way is never priced.
const ROUNDS_COUNTED_AS: u32 = 1;

/// The rounds a reply whose forwards overlap waited: its decode time over the
/// least time any of its forwards took to be answered, at most the forwards it
/// sent — `0` until one has been answered (none then waited a round).
///
/// The price charges a split ROUND by round, a round being one crossing of
/// every hop and one pass of each segment (`NodeCandidate::rounds_entered`). A
/// path that waits for each answer before sending again spends one of those a
/// round and counts each as it leaves. One that keeps forwards in flight does
/// not: the 14B's flow over Thailand ↔ Australia sent 252-322 forwards for a
/// 160-token prose reply (2026-10-29), counted one a forward they read 0.5
/// tokens a round — floored to 1, so a split that ran at 7-9 tokens a second
/// was priced at about 2, and, the figure being per model, every other split
/// of the model with it (one such reply among four tree-round replies took
/// their 4 tokens a round to 1.7). Its decode over its least answer is the
/// rounds it waited in the price's own unit: 37-50 for those replies (3.2-4.3
/// tokens a round, 7.7-10.3 on code), and 90-98 (1.6-1.8) where the far card
/// was saturated (36 of 48 layers there, answers queued up to 2.2 s), so a
/// queue is charged, not hidden. The least answer, not a mean: a queued one is
/// the queue's time, which the count must then show as rounds. At most one a
/// forward, so no reply is priced slower than that count priced it.
pub fn round_trips_waited(decode_ms: f64, least_answer_ms: Option<f64>, forwards: u32) -> u32 {
    let Some(least) = least_answer_ms.filter(|ms| ms.is_finite() && *ms > 0.0) else {
        return 0;
    };
    if forwards == 0 || !decode_ms.is_finite() {
        return 0;
    }
    let waited = (decode_ms.max(0.0) / least).ceil();
    // `as` saturates a float past `u32::MAX`; the clamp then holds it.
    (waited as u32).clamp(1, forwards)
}

pub struct SplitRounds {
    /// (tokens, rounds) of each recent split, oldest first.
    recent: RecentByModel<(u32, u32)>,
}

impl Default for SplitRounds {
    fn default() -> Self {
        Self {
            recent: RecentByModel::new("split_rounds", ROUNDS_COUNTED_AS),
        }
    }
}

impl SplitRounds {
    /// A split of `model` this node drove produced `tokens` in `rounds`, the
    /// prompt pass included. Either at zero tells nothing and is not recorded.
    pub fn record(&self, model: &str, tokens: u32, rounds: u32) {
        if usable(&(tokens, rounds)) {
            self.recent.push(model, (tokens, rounds));
        }
    }

    /// The tokens a round of `model`'s recent splits yielded, or `None` until
    /// [`BEFORE_TRUSTED`](super::recent_by_model::BEFORE_TRUSTED) have finished. Never below one: a round that
    /// yielded nothing (a stream's chunk sent on guesses that were then
    /// refused) overlapped another, and must not price a split above the one
    /// round a token it was priced at before.
    pub fn tokens_per_round(&self, model: &str) -> Option<f32> {
        self.recent.with_trusted(model, |recent| {
            let (tokens, rounds) = recent
                .iter()
                .fold((0u64, 0u64), |(t, r), &(tokens, rounds)| {
                    (t + u64::from(tokens), r + u64::from(rounds))
                });
            (tokens as f64 / rounds.max(1) as f64).max(1.0) as f32
        })
    }

    /// The splits an earlier run stored (`recent_by_model`).
    pub(super) fn restore(&self, db: &Database) {
        self.recent.restore(db, usable);
    }

    /// Store the recent splits if one finished since the last write.
    pub(super) fn persist(&self, db: &Database) {
        self.recent.persist(db);
    }
}

/// A split with either count at zero tells nothing.
fn usable(&(tokens, rounds): &(u32, u32)) -> bool {
    tokens > 0 && rounds > 0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::daemon::state::recent_by_model::{BEFORE_TRUSTED, RECENT};

    /// The figure is the recent splits' tokens over their rounds — a long
    /// reply weighs as its tokens do — once enough have finished.
    #[test]
    fn a_models_splits_yield_their_tokens_over_their_rounds() {
        let rounds = SplitRounds::default();
        for _ in 0..BEFORE_TRUSTED - 1 {
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
        for _ in 0..BEFORE_TRUSTED {
            rounds.record("plain", 40, 40);
            rounds.record("wasted", 10, 14);
        }
        assert_eq!(rounds.tokens_per_round("plain"), Some(1.0));
        assert_eq!(rounds.tokens_per_round("wasted"), Some(1.0));
    }

    /// A reply whose forwards overlapped is counted the round trips it waited:
    /// the 14B's flows on the real TH ↔ AU link (2026-10-29), at an even split
    /// and with the far card saturated, against their own lines.
    #[test]
    fn overlapping_forwards_are_counted_the_round_trips_they_waited() {
        // flowed 19.8 s, least answer 445 ms, 292 forwards, 160 tokens.
        let even = round_trips_waited(19_800.0, Some(445.0), 292);
        assert_eq!(even, 45);
        assert!((160.0 / even as f64 - 3.56).abs() < 0.01);
        // Saturated (12 of 48 here): 53.7 s, least 548 ms, 409 forwards.
        let saturated = round_trips_waited(53_700.0, Some(548.0), 409);
        assert_eq!(saturated, 98);
        assert!(160.0 / (saturated as f64) < 160.0 / (even as f64) / 2.0);
        // Never more than a round a forward: one sent at a time, each waited on
        // behind this node's own slow work, is one round each.
        assert_eq!(round_trips_waited(10_000.0, Some(100.0), 12), 12);
        // Nothing answered, nothing sent, nothing measurable: no round.
        assert_eq!(round_trips_waited(5_000.0, None, 10), 0);
        assert_eq!(round_trips_waited(5_000.0, Some(400.0), 0), 0);
        assert_eq!(round_trips_waited(5_000.0, Some(0.0), 10), 0);
        assert_eq!(round_trips_waited(f64::NAN, Some(400.0), 10), 0);
        // A decode shorter than a round trip still waited one.
        assert_eq!(round_trips_waited(10.0, Some(400.0), 3), 1);
    }

    #[test]
    fn only_the_most_recent_splits_count() {
        let rounds = SplitRounds::default();
        for _ in 0..RECENT {
            rounds.record("m", 400, 100);
        }
        for _ in 0..RECENT {
            rounds.record("m", 60, 60);
        }
        assert_eq!(rounds.tokens_per_round("m"), Some(1.0));
    }
}
