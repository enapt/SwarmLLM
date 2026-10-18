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

use dashmap::DashMap;
use std::collections::VecDeque;

/// Replies remembered per model: enough that one outlier moves the mean by a
/// thirty-second of itself, few enough that a change in how the model is used
/// shows within a session.
pub const RECENT_REPLIES: usize = 32;

/// Replies a model must have written before its mean is trusted over the
/// prior. Below it one long or short reply would decide the next route.
pub const REPLIES_BEFORE_TRUSTED: usize = 4;

/// At most this many models are tracked — a model id reaches here only from
/// a request that completed, but the map is bounded anyway.
const MAX_MODELS: usize = 256;

#[derive(Default)]
pub struct ReplyLengths {
    by_model: DashMap<String, VecDeque<u32>>,
}

impl ReplyLengths {
    /// A finished reply of `completion_tokens` tokens for `model`. A reply of
    /// none tells nothing about length (a refusal, an empty answer) and is
    /// not recorded.
    pub fn record(&self, model: &str, completion_tokens: u32) {
        if completion_tokens == 0 {
            return;
        }
        if !self.by_model.contains_key(model) && self.by_model.len() >= MAX_MODELS {
            return;
        }
        let mut recent = self.by_model.entry(model.to_string()).or_default();
        if recent.len() >= RECENT_REPLIES {
            recent.pop_front();
        }
        recent.push_back(completion_tokens);
    }

    /// The mean length of `model`'s recent replies, or `None` until it has
    /// written [`REPLIES_BEFORE_TRUSTED`] — the caller's prior stands until
    /// then.
    pub fn expected(&self, model: &str) -> Option<u32> {
        let recent = self.by_model.get(model)?;
        if recent.len() < REPLIES_BEFORE_TRUSTED {
            return None;
        }
        let sum: u64 = recent.iter().map(|&n| u64::from(n)).sum();
        Some((sum as f64 / recent.len() as f64).round().max(1.0) as u32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        for _ in 0..RECENT_REPLIES {
            lengths.record("m", 500);
        }
        for _ in 0..RECENT_REPLIES {
            lengths.record("m", 20);
        }
        assert_eq!(lengths.expected("m"), Some(20));
    }
}
