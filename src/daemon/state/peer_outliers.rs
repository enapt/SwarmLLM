//! A peer that keeps failing one model is left out of that model's plans for a
//! while — outlier ejection, as Envoy does it for upstream hosts.
//!
//! Field report (2026-10-01, v0.3.218): peer `bf7b3263` held layers 7..18 of
//! qwen3-30b-a3b on a 6 GB laptop card. It answered every prompt pass and then
//! went silent at the first decode step — 5 of 6 pipelines that included it
//! died, 3 of 3 without it were fine. Within one request the existing rules
//! held: the silent peer was barred from that request's retry
//! (`blacklist_holder_for_request`). But nothing carried the lesson to the NEXT
//! request, and the only cross-request signal — the delivery-reliability
//! multiplier — priced it at 2.04×, which a fast peer survives: the scheduler
//! kept picking it, and every pick cost a 30 s deadline.
//!
//! Envoy's outlier detection is the reference (`consecutive_gateway_failure`,
//! `base_ejection_time` multiplied by the number of times the host has been
//! ejected, and a panic threshold below which ejected hosts are used anyway):
//!
//! - **Consecutive** failures, reset by a request the peer saw through — so a
//!   peer that fails now and then is priced (the multiplier), and only one that
//!   fails EVERY time is ejected.
//! - **Per (peer, model)**: the failure above is a 6 GB card and a 30B model; the
//!   same machine may serve a small one perfectly.
//! - **The ejection grows** with each repeat and is capped, so a peer that has
//!   recovered is tried again soon, and one that has not costs little.
//! - **Never the only way to a part** (the panic threshold): the scheduler admits
//!   an ejected peer for a part nobody else can serve (`gather_candidates`), so an
//!   ejection can make a plan slower, never make a model unroutable.
//!
//! What counts as a failure is what the delivery figure already calls one — a
//! segment that timed out or was abandoned on our side, a whole-model reply that
//! went silent — via `SharedState::record_peer_delivery`, the one place those
//! are recorded. Refusals (out of memory, too long) are not failures here: they
//! arrive fast, carry a reason, and are re-planned within the request.

use std::time::{Duration, Instant};

use dashmap::DashMap;

use crate::types::{ModelId, NodeId};

/// Failures in a row, with no request completed in between, before a peer is
/// ejected from a model. Two, not Envoy's default five: each of ours costs a
/// deadline of tens of seconds to a person waiting, where Envoy's cost
/// milliseconds to a load balancer.
pub const FAILURES_TO_EJECT: u32 = 2;
/// The first ejection; each further one doubles it.
pub const BASE_EJECTION: Duration = Duration::from_secs(120);
/// The longest an ejection lasts.
pub const MAX_EJECTION: Duration = Duration::from_secs(30 * 60);

#[derive(Clone, Copy, Debug, Default)]
struct Outlier {
    consecutive: u32,
    ejections: u32,
    ejected_until: Option<Instant>,
}

/// Lives on `state.metrics` beside the other per-peer measurements.
#[derive(Default)]
pub struct PeerOutliers {
    map: DashMap<(NodeId, ModelId), Outlier>,
}

/// How long the `n`th ejection (1-based) lasts.
pub fn ejection_length(n: u32) -> Duration {
    let doublings = n.saturating_sub(1).min(16);
    BASE_EJECTION
        .saturating_mul(1u32 << doublings)
        .min(MAX_EJECTION)
}

impl PeerOutliers {
    /// `peer` failed `model`. Returns how long it is now ejected for, when this
    /// failure ejected it.
    pub fn note_failure(&self, peer: &NodeId, model: &ModelId, now: Instant) -> Option<Duration> {
        let mut entry = self.map.entry((peer.clone(), model.clone())).or_default();
        let o = entry.value_mut();
        // A failure while already ejected — the panic threshold admitted it —
        // extends nothing: the ejection is already counting down.
        if o.ejected_until.is_some_and(|until| now < until) {
            return None;
        }
        o.consecutive = o.consecutive.saturating_add(1);
        if o.consecutive < FAILURES_TO_EJECT {
            return None;
        }
        o.ejections = o.ejections.saturating_add(1);
        o.consecutive = 0;
        let length = ejection_length(o.ejections);
        o.ejected_until = Some(now + length);
        Some(length)
    }

    /// `peer` saw a request for `model` through. Forgets everything about it:
    /// a recovered peer starts from a clean slate, as in Envoy, where a host
    /// that stays healthy has its ejection multiplier wound back.
    pub fn note_success(&self, peer: &NodeId, model: &ModelId) {
        // A read first: this runs once per completed request per peer, and in
        // the steady state there is nothing to remove.
        if self.map.contains_key(&(peer.clone(), model.clone())) {
            self.map.remove(&(peer.clone(), model.clone()));
        }
    }

    /// Is `peer` ejected from `model` at `now`?
    pub fn is_ejected(&self, peer: &NodeId, model: &ModelId, now: Instant) -> bool {
        self.map
            .get(&(peer.clone(), model.clone()))
            .is_some_and(|o| o.ejected_until.is_some_and(|until| now < until))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn peer() -> NodeId {
        NodeId([0xbf; 32])
    }
    fn model() -> ModelId {
        ModelId("qwen3-30b-a3b-instruct-2507-q4-k-m".into())
    }

    /// The field report's shape: every request through the peer fails. The
    /// second in a row ejects it; without this, every later request picked it
    /// again and paid the 30 s deadline.
    #[test]
    fn a_peer_failing_every_request_is_ejected_on_the_second() {
        let o = PeerOutliers::default();
        let t = Instant::now();
        assert_eq!(o.note_failure(&peer(), &model(), t), None);
        assert!(!o.is_ejected(&peer(), &model(), t));
        assert_eq!(o.note_failure(&peer(), &model(), t), Some(BASE_EJECTION));
        assert!(o.is_ejected(&peer(), &model(), t));
        assert!(!o.is_ejected(&peer(), &model(), t + BASE_EJECTION));
    }

    /// A peer that fails now and then, with requests completed in between, is
    /// never ejected — that is the reliability multiplier's job, not this one.
    #[test]
    fn a_completed_request_between_failures_resets_the_count() {
        let o = PeerOutliers::default();
        let t = Instant::now();
        for _ in 0..5 {
            assert_eq!(o.note_failure(&peer(), &model(), t), None);
            o.note_success(&peer(), &model());
        }
        assert!(!o.is_ejected(&peer(), &model(), t));
    }

    /// The same machine may serve another model perfectly.
    #[test]
    fn an_ejection_is_for_one_model() {
        let o = PeerOutliers::default();
        let t = Instant::now();
        o.note_failure(&peer(), &model(), t);
        o.note_failure(&peer(), &model(), t);
        assert!(o.is_ejected(&peer(), &model(), t));
        assert!(!o.is_ejected(&peer(), &ModelId("qwen2.5-0.5b".into()), t));
        assert!(!o.is_ejected(&NodeId([1; 32]), &model(), t));
    }

    /// Each repeat doubles the ejection, up to the cap; a failure during an
    /// ejection (the peer was the only way to a part) does not extend it.
    #[test]
    fn ejections_grow_and_are_capped() {
        assert_eq!(ejection_length(1), BASE_EJECTION);
        assert_eq!(ejection_length(2), BASE_EJECTION * 2);
        assert_eq!(ejection_length(3), BASE_EJECTION * 4);
        assert_eq!(ejection_length(40), MAX_EJECTION);

        let o = PeerOutliers::default();
        let mut t = Instant::now();
        o.note_failure(&peer(), &model(), t);
        assert_eq!(o.note_failure(&peer(), &model(), t), Some(BASE_EJECTION));
        assert_eq!(
            o.note_failure(&peer(), &model(), t),
            None,
            "already ejected"
        );
        t += BASE_EJECTION;
        o.note_failure(&peer(), &model(), t);
        assert_eq!(
            o.note_failure(&peer(), &model(), t),
            Some(BASE_EJECTION * 2)
        );
    }
}
