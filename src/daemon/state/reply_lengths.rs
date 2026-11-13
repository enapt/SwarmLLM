//! How long each model's recent replies ran — what a request that states no
//! shorter bound is priced for (FUTURE_WORK #3 (b)).
//!
//! Routing charges a reply's per-token costs (every segment's compute, every
//! hop a token crosses) once per pass, and a request rarely says how many
//! passes it will take: `max_tokens` is a ceiling, 2,048 by default, for a
//! reply of fifty. The planner assumed 64 (`parallax::ASSUMED_FORWARD_PASSES`)
//! and the request's own `max_tokens` lowers that; a longer reply was
//! under-priced by its length ÷ 64 — 2.49x and 2.62x at 160 tokens on two
//! peers' hand-offs (2026-10-18). So the expectation is the one thing a node
//! can know about a reply before it is written: how long this model's replies
//! have been. A scheduler that has no length predictor does the same — the
//! historical mean per workload is the baseline the output-length predictors
//! in the literature (S³, Jin et al. 2023; "Response Length Perception",
//! Zheng et al. 2023) are measured against.
//!
//! **The MEAN, not the median**: the price is an expected time, and a pass
//! costs the same wherever in the reply it falls, so the expected cost of a
//! reply is the cost of its expected length. A median would price a model
//! answering thirty short replies and two of 2,000 tokens as if the long ones
//! never happened.
//!
//! A reply cut off by its `max_tokens` counts at the length it was allowed —
//! a lower bound on what the model would have written. Dropping it instead
//! would keep only the replies short enough to finish, which is lower still.
//!
//! Kept across restarts (`recent_by_model`): a restart priced every reply at
//! the prior until four had finished.
//!
//! **Before four have finished, the prior is ONE reply beside them** (2026-11-13):
//! the planner's assumed reply ([`PRIOR_REPLY_TOKENS`]) counted as one more
//! finished reply, in the same mean, and dropped at
//! [`BEFORE_TRUSTED`](super::recent_by_model::BEFORE_TRUSTED). All or nothing,
//! a fresh node priced its first four replies at 64 tokens whatever it had
//! seen: three 160-token hand-offs of a 14B over Thailand ↔ Australia were
//! each priced 2.9-4.2 s and took 11.9-13.4 (`real_link.sh`, arms `flowall`) —
//! 2.5x of the 3-4x from the length alone. One pseudo-observation is how this
//! crate already starts a figure it has not measured (`split_rounds::
//! YieldPrior`, `AcceptanceEstimate`'s ten trials); LEO (Stillger et al.,
//! VLDB 2001) corrects a cost model's INPUT from each execution rather than
//! waiting to trust a total.

use super::recent_by_model::RecentByModel;
use super::split_rounds::PRIOR_REPLY_TOKENS;
use crate::storage::db::Database;

/// What a stored length means: 1, a finished reply's completion tokens.
const LENGTHS_COUNTED_AS: u32 = 1;

pub struct ReplyLengths {
    recent: RecentByModel<u32>,
}

impl Default for ReplyLengths {
    fn default() -> Self {
        Self {
            recent: RecentByModel::new("reply_lengths", LENGTHS_COUNTED_AS),
        }
    }
}

impl ReplyLengths {
    /// A finished reply of `completion_tokens` tokens for `model`. A reply of
    /// none tells nothing about length (a refusal, an empty answer) and is
    /// not recorded.
    pub fn record(&self, model: &str, completion_tokens: u32) {
        if completion_tokens > 0 {
            self.recent.push(model, completion_tokens);
        }
    }

    /// The mean length of `model`'s recent replies — with the planner's
    /// assumed reply ([`PRIOR_REPLY_TOKENS`]) counted as one more until it has
    /// written [`BEFORE_TRUSTED`](super::recent_by_model::BEFORE_TRUSTED) —
    /// or `None` where it has written none, so the caller's prior stands.
    pub fn expected(&self, model: &str) -> Option<u32> {
        let sum = |recent: &std::collections::VecDeque<u32>| {
            (
                recent.iter().map(|&n| u64::from(n)).sum::<u64>() as f64,
                recent.len() as f64,
            )
        };
        let (total, replies) = match self.recent.with_trusted(model, sum) {
            Some(trusted) => trusted,
            None => {
                let (total, replies) = self.recent.with_any(model, sum)?;
                (total + PRIOR_REPLY_TOKENS, replies + 1.0)
            }
        };
        Some((total / replies).round().max(1.0) as u32)
    }

    /// The lengths an earlier run stored (`recent_by_model`).
    pub(super) fn restore(&self, db: &Database) {
        self.recent.restore(db, |&n| n > 0);
    }

    /// Store the recent lengths if a reply finished since the last write.
    pub(super) fn persist(&self, db: &Database) {
        self.recent.persist(db);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::daemon::state::recent_by_model::RECENT;

    #[test]
    fn the_expectation_is_the_mean_of_recent_replies_once_there_are_enough() {
        let lengths = ReplyLengths::default();
        for n in [10, 20, 30] {
            lengths.record("m", n);
        }
        assert_eq!(
            lengths.expected("m"),
            Some(31),
            "three replies beside the prior's one: (60 + 64) / 4"
        );
        lengths.record("m", 340);
        assert_eq!(lengths.expected("m"), Some(100), "trusted: the prior gone");
        assert_eq!(lengths.expected("other"), None);
        // An empty reply says nothing about length.
        lengths.record("m", 0);
        assert_eq!(lengths.expected("m"), Some(100));
    }

    /// **The first replies move the price at once** (2026-11-13): the 14B's
    /// three 160-token hand-offs on a fresh node were each priced at the
    /// assumed 64. The prior now counts as one reply beside them.
    #[test]
    fn a_first_reply_moves_the_price_beside_the_prior() {
        let lengths = ReplyLengths::default();
        assert_eq!(
            lengths.expected("m"),
            None,
            "nothing written: the caller's prior"
        );
        lengths.record("m", 160);
        assert_eq!(lengths.expected("m"), Some(112), "(64 + 160) / 2");
        lengths.record("m", 160);
        assert_eq!(lengths.expected("m"), Some(128), "(64 + 320) / 3");
        lengths.record("m", 160);
        assert_eq!(lengths.expected("m"), Some(136));
        lengths.record("m", 160);
        assert_eq!(lengths.expected("m"), Some(160), "four: the replies alone");
    }

    #[test]
    fn only_the_most_recent_replies_count() {
        let lengths = ReplyLengths::default();
        for _ in 0..RECENT {
            lengths.record("m", 500);
        }
        for _ in 0..RECENT {
            lengths.record("m", 20);
        }
        assert_eq!(lengths.expected("m"), Some(20));
    }
}
