//! What a chain hop owes its coordinator for each forward it handed on.
//!
//! A node that forwards a request id it RECEIVED is a hop of a chained run
//! (`verify_may_chain`, direct peer chaining): the coordinator waits on the
//! run as a whole and the hop holds no wait of its own. So a failure that
//! reaches the hop about its onward forward reaches nobody unless the hop
//! passes it on — and the coordinator then waits out the chain's deadline for
//! a failure known in milliseconds. Two such failures:
//!
//! - **This node could not deliver the onward forward** (no route, a seal
//!   that failed, no receipt acknowledgement, `OutboundFailure`): reported by
//!   `NetworkManager::fail_tensor_forward`, the one place a forward fails.
//! - **The next hop refused it** after acknowledging it — a forward whose seal
//!   did not open is answered on the connection it came in on
//!   (`pipeline_stream::refusal_for_undecryptable`), which is THIS node, not
//!   the coordinator. The next hop cannot answer the coordinator itself: the
//!   frame's `requester_node_id` is cleartext the seal did not authenticate,
//!   so a forger could make an honest hop refuse another node's request
//!   (FUTURE_WORK #305). This node recorded the coordinator when it handed
//!   the forward on, from a forward whose seal DID open.
//!
//! Both go through [`HopForwards::take_onward_failure`]: only an error, only
//! from the node this hop handed the request to, only about the step it
//! handed on (or naming none), and only once. What goes to the coordinator is
//! this node's own report — the same reason, answering the recorded step,
//! with no refusal type (the refusal's "send it again" is the next hop's
//! claim about the next hop's link, which the coordinator cannot act on: a
//! chained run resends nothing) — attributed to this node, a chain member the
//! coordinator's wait already accepts. Measured with `split_rig.sh
//! refusechain` (TinyLlama over B → C, C refusing one chained step): v0.3.244
//! dropped the refusal at B and the request took 45.8 s, its coordinator
//! timing the step out; with this, 1.5 s. Being step-named, the report may be
//! resent once after an `OutboundFailure` (`resend_lost_result`); the
//! coordinator's wait takes one copy and logs the other as moved on.
//!
//! A record is kept past the next hop's receipt acknowledgement (a refusal
//! comes after it) for as long as a forward may run, and then forgotten; a
//! newer forward for the same request replaces it. At most
//! [`MAX_HOP_FORWARDS`] are held: a peer choosing request ids cannot grow
//! this without bound.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use crate::types::{LayerResult, NetworkFinishReason, NodeId, ResultStep};

/// How long a hop remembers a forward it handed on: as long as one may run
/// (`MAX_TENSOR_FORWARD_SECS`, request-response's own timeout). Past it the
/// coordinator's wait is over whatever the hop says.
pub(super) const HOP_FORWARD_KEEP: Duration = Duration::from_secs(super::MAX_TENSOR_FORWARD_SECS);

/// The most forwards a hop remembers at once — one per chained run this node
/// is a hop of. Past it the oldest record goes; it only means that run's
/// coordinator learns of a failure at its own deadline, as before #305.
pub(super) const MAX_HOP_FORWARDS: usize = 1024;

/// One forward this node handed on as a hop.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct HopForward {
    /// The node waiting on the run — named by a forward whose seal opened.
    pub coordinator: NodeId,
    /// The node this hop handed the forward to.
    pub onward: NodeId,
    /// The forward handed on: what a failure must answer for the
    /// coordinator's wait to find it.
    pub step: ResultStep,
    pub sent_at: Instant,
}

/// The forwards this node handed on as a hop, by request id.
#[derive(Debug, Default)]
pub(super) struct HopForwards {
    by_request: HashMap<uuid::Uuid, HopForward>,
}

impl HopForwards {
    /// Remember a forward handed on. A newer forward for the same request
    /// replaces the older one's record: a chained run sends its next step only
    /// once the last was answered.
    pub(super) fn record(&mut self, request_id: uuid::Uuid, hop: HopForward) {
        if !self.by_request.contains_key(&request_id) && self.by_request.len() >= MAX_HOP_FORWARDS {
            let oldest = self
                .by_request
                .iter()
                .min_by_key(|(_, h)| h.sent_at)
                .map(|(id, _)| *id);
            if let Some(id) = oldest {
                self.by_request.remove(&id);
            }
        }
        self.by_request.insert(request_id, hop);
    }

    /// If `result`, received from `sender`, is a failure of a forward this
    /// node handed on, the record and the report owed to its coordinator.
    /// Taken: a forward's failure is passed on once.
    pub(super) fn take_onward_failure(
        &mut self,
        sender: Option<&NodeId>,
        result: &LayerResult,
    ) -> Option<(HopForward, LayerResult)> {
        let Some(NetworkFinishReason::Error(reason)) = &result.finish_reason else {
            return None;
        };
        let hop = self.by_request.get(&result.request_id)?;
        if sender != Some(&hop.onward) {
            return None;
        }
        // A failure naming another step is about a forward this record no
        // longer describes (an older step, or one never handed on from here).
        if result.answers_step.is_some_and(|s| s != hop.step) {
            return None;
        }
        let hop = self.by_request.remove(&result.request_id)?;
        let report = LayerResult::error(result.request_id, reason.clone()).answering_step(hop.step);
        Some((hop, report))
    }

    /// Forget every record older than [`HOP_FORWARD_KEEP`]; how many went.
    pub(super) fn expire(&mut self, now: Instant) -> usize {
        let before = self.by_request.len();
        self.by_request
            .retain(|_, h| now.saturating_duration_since(h.sent_at) < HOP_FORWARD_KEEP);
        before - self.by_request.len()
    }

    pub(super) fn len(&self) -> usize {
        self.by_request.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node(b: u8) -> NodeId {
        NodeId([b; 32])
    }

    fn step(index_pos: u32) -> ResultStep {
        ResultStep {
            index_pos,
            layer_range: (14, 28),
            stream_seq: None,
        }
    }

    fn hop(at: Instant) -> HopForward {
        HopForward {
            coordinator: node(1),
            onward: node(3),
            step: step(40),
            sent_at: at,
        }
    }

    /// The refusal the next hop sends back on the connection it came in on
    /// becomes this node's report to the coordinator: same reason, the step
    /// handed on, no refusal type — once.
    #[test]
    fn the_next_hops_refusal_is_passed_to_the_coordinator_once() {
        let mut hops = HopForwards::default();
        let rid = uuid::Uuid::new_v4();
        hops.record(rid, hop(Instant::now()));
        let refusal = LayerResult::error(rid, "Could not decrypt forward")
            .answering_step(step(40))
            .with_refusal(crate::types::ForwardRefusal::Undecryptable);

        let (to, report) = hops
            .take_onward_failure(Some(&node(3)), &refusal)
            .expect("passed on");
        assert_eq!(to.coordinator, node(1));
        assert_eq!(report.request_id, rid);
        assert_eq!(report.answers_step, Some(step(40)));
        assert_eq!(
            report.refusal, None,
            "the next hop's link is not the coordinator's to re-key"
        );
        assert!(matches!(
            &report.finish_reason,
            Some(NetworkFinishReason::Error(r)) if r == "Could not decrypt forward"
        ));
        assert!(
            hops.take_onward_failure(Some(&node(3)), &refusal).is_none(),
            "once"
        );
    }

    /// Only the node the forward went to, only a failure, only the step that
    /// went: anything else leaves the record where it is.
    #[test]
    fn nothing_else_is_passed_on() {
        let mut hops = HopForwards::default();
        let rid = uuid::Uuid::new_v4();
        hops.record(rid, hop(Instant::now()));
        let error = LayerResult::error(rid, "Peer departed");

        assert!(
            hops.take_onward_failure(Some(&node(9)), &error).is_none(),
            "a stranger"
        );
        assert!(
            hops.take_onward_failure(None, &error).is_none(),
            "unauthenticated"
        );
        let other_step = error.clone().answering_step(step(41));
        assert!(
            hops.take_onward_failure(Some(&node(3)), &other_step)
                .is_none(),
            "another step"
        );
        let mut success = LayerResult::error(rid, "x");
        success.finish_reason = None;
        assert!(
            hops.take_onward_failure(Some(&node(3)), &success).is_none(),
            "not a failure"
        );
        let other_request = LayerResult::error(uuid::Uuid::new_v4(), "x");
        assert!(hops
            .take_onward_failure(Some(&node(3)), &other_request)
            .is_none());

        assert_eq!(hops.len(), 1, "the record waits for its own failure");
        // A failure naming no step — an older next hop, or this node's own
        // delivery failure built before the step was known — still counts.
        assert!(hops.take_onward_failure(Some(&node(3)), &error).is_some());
    }

    /// A record outlives the receipt acknowledgement for as long as a forward
    /// may run, then goes; the map never holds more than its ceiling.
    #[test]
    fn records_expire_and_are_capped() {
        let mut hops = HopForwards::default();
        let t0 = Instant::now();
        let old = uuid::Uuid::new_v4();
        hops.record(old, hop(t0));
        assert_eq!(
            hops.expire(t0 + HOP_FORWARD_KEEP.saturating_sub(Duration::from_secs(1))),
            0
        );
        assert_eq!(hops.expire(t0 + HOP_FORWARD_KEEP), 1);

        let first = uuid::Uuid::new_v4();
        hops.record(first, hop(t0));
        for i in 1..MAX_HOP_FORWARDS + 10 {
            hops.record(
                uuid::Uuid::new_v4(),
                hop(t0 + Duration::from_millis(i as u64)),
            );
        }
        assert_eq!(hops.len(), MAX_HOP_FORWARDS);
        let error = LayerResult::error(first, "x");
        assert!(
            hops.take_onward_failure(Some(&node(3)), &error).is_none(),
            "the oldest went first"
        );
    }
}
