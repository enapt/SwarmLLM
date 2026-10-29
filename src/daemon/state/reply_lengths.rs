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

use super::recent_by_model::RecentByModel;
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

    /// The mean length of `model`'s recent replies, or `None` until it has
    /// written [`BEFORE_TRUSTED`](super::recent_by_model::BEFORE_TRUSTED) — the caller's prior stands until
    /// then.
    pub fn expected(&self, model: &str) -> Option<u32> {
        self.recent.with_trusted(model, |recent| {
            let sum: u64 = recent.iter().map(|&n| u64::from(n)).sum();
            (sum as f64 / recent.len() as f64).round().max(1.0) as u32
        })
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
            None,
            "three replies decided the price"
        );
        lengths.record("m", 340);
        assert_eq!(lengths.expected("m"), Some(100));
        assert_eq!(lengths.expected("other"), None);
        // An empty reply says nothing about length.
        lengths.record("m", 0);
        assert_eq!(lengths.expected("m"), Some(100));
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
