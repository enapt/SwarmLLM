//! A head that declined to lead a model's split is not priced, or asked, as its
//! leader again for a while.
//!
//! Field case (2026-10-29, v0.3.261, Llama-3.3-70B): the planner priced a
//! chain led by the Australian RTX 3060 node (it holds the whole model and
//! advertised room for all 80 layers), handed it over, and the head declined
//! within a second — its own memory check refused a 36.8 GB load against a
//! 32.6 GB live budget. The requester then drove that same chain itself
//! (`try_delegated_split` falls back to the plan it has), a plan it chose ONLY
//! because it priced it as led: three far machines, every round crossing each
//! of them, 0.95-1.42 tok/s where the plan through this node had run at
//! 1.97-2.29. And nothing remembered the decline, so every request asked
//! again, waited the second, and drove the led-priced plan again.
//!
//! The decline is a FACT about the head, not a failure of the request: the
//! planner may price a chain as led only by a head that would lead it
//! (`parallax::chain_is_led_by_its_head`, FUTURE_WORK #3 (a)), and this is the
//! one evidence it has that a head would not. So for a while after a decline,
//! `SharedState::peer_may_lead` answers no for that head and model — read by
//! both the hand-off (`remote_generate::delegation_eligible`) and the planner's
//! led pricing (`RingFacts::leads_splits`), so a chain is never priced as led
//! by a head the hand-off then passes over, nor handed to one it priced
//! driven.
//!
//! The wait doubles while the head keeps declining and is capped, as Envoy's
//! ejection does (`peer_outliers`): a head whose memory has come back is asked
//! again soon, and one that has not costs one request per wait. A split the
//! head leads — a token delivered, or a reply finished — forgets it.

use std::time::{Duration, Instant};

use dashmap::DashMap;

use crate::types::{ModelId, NodeId};

/// The wait after the first decline; each further one doubles it.
pub const FIRST_WAIT: Duration = Duration::from_secs(5 * 60);
/// The longest wait.
pub const LONGEST_WAIT: Duration = Duration::from_secs(60 * 60);

#[derive(Clone, Copy, Debug)]
struct Declined {
    declines: u32,
    until: Instant,
}

/// Lives on `state.metrics` beside `peer_outliers`. Written ONLY by
/// `SharedState::note_head_declined_to_lead` / `note_head_led`, read ONLY by
/// `SharedState::peer_may_lead`.
#[derive(Default)]
pub struct DeclinedLeads {
    map: DashMap<(NodeId, ModelId), Declined>,
}

/// How long a head is passed over after its `n`th decline in a row (1-based).
pub fn wait_after(n: u32) -> Duration {
    let doublings = n.saturating_sub(1).min(16);
    FIRST_WAIT
        .saturating_mul(1u32 << doublings)
        .min(LONGEST_WAIT)
}

impl DeclinedLeads {
    /// `head` declined to lead `model`'s split. Returns how long it is now
    /// passed over for.
    pub fn note_declined(&self, head: &NodeId, model: &ModelId, now: Instant) -> Duration {
        let mut entry = self
            .map
            .entry((head.clone(), model.clone()))
            .or_insert(Declined {
                declines: 0,
                until: now,
            });
        let d = entry.value_mut();
        d.declines = d.declines.saturating_add(1);
        let wait = wait_after(d.declines);
        d.until = now + wait;
        wait
    }

    /// `head` led `model`'s split: forget its declines.
    pub fn note_led(&self, head: &NodeId, model: &ModelId) {
        let key = (head.clone(), model.clone());
        // A read first: this runs on every led split, and there is usually
        // nothing to remove.
        if self.map.contains_key(&key) {
            self.map.remove(&key);
        }
    }

    /// Is `head` passed over as `model`'s leader at `now`?
    pub fn declining(&self, head: &NodeId, model: &ModelId, now: Instant) -> bool {
        self.map
            .get(&(head.clone(), model.clone()))
            .is_some_and(|d| now < d.until)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn head() -> NodeId {
        NodeId([0xaa; 32])
    }
    fn model() -> ModelId {
        ModelId("llama-3.3-70b-instruct-q4-k-m".into())
    }

    /// The field case's shape: the head declines every time. Each decline
    /// passes it over for longer, up to the cap, and the wait ends.
    #[test]
    fn a_head_declining_every_time_is_passed_over_for_longer_each_time() {
        let d = DeclinedLeads::default();
        let t = Instant::now();
        assert!(!d.declining(&head(), &model(), t));
        assert_eq!(d.note_declined(&head(), &model(), t), FIRST_WAIT);
        assert!(d.declining(&head(), &model(), t));
        assert!(d.declining(&head(), &model(), t + FIRST_WAIT / 2));
        assert!(
            !d.declining(&head(), &model(), t + FIRST_WAIT),
            "the wait ends: a head whose memory came back is asked again"
        );
        let t2 = t + FIRST_WAIT;
        assert_eq!(d.note_declined(&head(), &model(), t2), FIRST_WAIT * 2);
        assert!(d.declining(&head(), &model(), t2 + FIRST_WAIT));
        let mut t3 = t2;
        for _ in 0..10 {
            t3 += LONGEST_WAIT;
            assert!(d.note_declined(&head(), &model(), t3) <= LONGEST_WAIT);
        }
        assert_eq!(wait_after(u32::MAX), LONGEST_WAIT);
    }

    /// A split the head leads forgets every decline: the next one starts from
    /// the first wait.
    #[test]
    fn a_split_the_head_leads_forgets_its_declines() {
        let d = DeclinedLeads::default();
        let t = Instant::now();
        d.note_declined(&head(), &model(), t);
        d.note_declined(&head(), &model(), t);
        d.note_led(&head(), &model());
        assert!(!d.declining(&head(), &model(), t));
        assert_eq!(d.note_declined(&head(), &model(), t), FIRST_WAIT);
    }

    /// Per (head, model): the same machine may lead a smaller model, and
    /// another head is not passed over for this one's decline.
    #[test]
    fn a_decline_is_the_heads_for_that_model_only() {
        let d = DeclinedLeads::default();
        let t = Instant::now();
        d.note_declined(&head(), &model(), t);
        assert!(!d.declining(&head(), &ModelId("qwen2.5-14b".into()), t));
        assert!(!d.declining(&NodeId([0xbb; 32]), &model(), t));
    }
}
