//! A forward's round trip, split at its receipt (`docs/plans/large_model_splits.md`
//! build order item 1).
//!
//! Every tensor forward is acknowledged on receipt, before any work
//! (`requests.rs`), and its result comes back as its own request. So, timed
//! from the same send:
//! - the ACK's time is the network both ways plus the far side taking the
//!   message in, and
//! - the result's time AFTER the ACK is what the far side did with it —
//!   dispatch, its worker, the compute, sending the answer — because the two
//!   replies cross the same path back and their one-way delays cancel.
//!
//! QUIC makes the same cut for its RTT samples, from the other side: the
//! receiver reports how long it held an ACK (`ack_delay`) and the sender
//! subtracts it (RFC 9002 § 5.3). Here the ACK is never held, so the receiver's
//! time shows up in the result instead.
//!
//! On the real Thailand↔Australia link a check cost 472-520 ms against a
//! 395 ms minimum round trip and ~20 ms of compute; which side the rest is on
//! decides what to remove. The split is logged at `debug`, per forward. It is
//! only measured, never acted on.
//!
//! In a chained run the ACK is the FIRST hop's and the result the tail's, so
//! the far side's share covers every hop after the first.

use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

use crate::types::ResultStep;

/// How many acknowledged forwards wait for their results. A result that never
/// comes (a failure answered elsewhere, a stream restarted past it) must not
/// pin memory; a decode step's result arrives within one round trip, so a few
/// hundred covers every forward a node has in flight.
const MOST_KEPT: usize = 256;

/// A forward that was acknowledged and not yet answered.
struct Receipt {
    sent_at: Instant,
    ack: Duration,
    acked_by: libp2p::PeerId,
    bytes: usize,
}

/// One forward's time, cut at its receipt.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct Split {
    /// Send to acknowledgement.
    pub(super) ack_ms: f64,
    /// Send to result.
    pub(super) answer_ms: f64,
    pub(super) acked_by: libp2p::PeerId,
    pub(super) bytes: usize,
}

impl Split {
    /// What the far side spent between taking the forward in and answering.
    pub(super) fn after_receipt_ms(&self) -> f64 {
        (self.answer_ms - self.ack_ms).max(0.0)
    }
}

type Key = (uuid::Uuid, ResultStep);

/// Acknowledged forwards by (request, step), oldest dropped past [`MOST_KEPT`].
#[derive(Default)]
pub(super) struct ForwardReceipts {
    by_step: HashMap<Key, Receipt>,
    order: VecDeque<Key>,
}

impl ForwardReceipts {
    /// The forward `step` of `request_id`, sent at `sent_at`, was acknowledged
    /// by `peer` after `ack`.
    pub(super) fn note_ack(
        &mut self,
        request_id: uuid::Uuid,
        step: ResultStep,
        sent_at: Instant,
        ack: Duration,
        peer: libp2p::PeerId,
        bytes: usize,
    ) {
        let key = (request_id, step);
        let receipt = Receipt {
            sent_at,
            ack,
            acked_by: peer,
            bytes,
        };
        if self.by_step.insert(key, receipt).is_none() {
            self.order.push_back(key);
        }
        while self.order.len() > MOST_KEPT {
            if let Some(old) = self.order.pop_front() {
                self.by_step.remove(&old);
            }
        }
    }

    /// The split of the forward a result answers, if its ACK was seen — once.
    pub(super) fn take(&mut self, request_id: uuid::Uuid, step: ResultStep) -> Option<Split> {
        let receipt = self.by_step.remove(&(request_id, step))?;
        Some(Split {
            ack_ms: receipt.ack.as_secs_f64() * 1000.0,
            answer_ms: receipt.sent_at.elapsed().as_secs_f64() * 1000.0,
            acked_by: receipt.acked_by,
            bytes: receipt.bytes,
        })
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.by_step.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn step(index_pos: u32) -> ResultStep {
        ResultStep {
            index_pos,
            layer_range: (14, 28),
            stream_seq: None,
        }
    }

    #[test]
    fn a_result_is_split_at_its_acknowledgement_once() {
        let mut receipts = ForwardReceipts::default();
        let id = uuid::Uuid::new_v4();
        let peer = libp2p::PeerId::random();
        let sent_at = Instant::now()
            .checked_sub(Duration::from_millis(500))
            .expect("the clock is past half a second");
        receipts.note_ack(id, step(7), sent_at, Duration::from_millis(400), peer, 8192);

        let split = receipts
            .take(id, step(7))
            .expect("acknowledged, then answered");
        assert_eq!(split.ack_ms, 400.0);
        assert!(split.answer_ms >= 500.0);
        assert!(split.after_receipt_ms() >= 100.0);
        assert_eq!((split.acked_by, split.bytes), (peer, 8192));
        assert!(receipts.take(id, step(7)).is_none(), "once");
    }

    #[test]
    fn a_result_for_another_step_or_request_finds_nothing() {
        let mut receipts = ForwardReceipts::default();
        let id = uuid::Uuid::new_v4();
        let peer = libp2p::PeerId::random();
        receipts.note_ack(id, step(7), Instant::now(), Duration::ZERO, peer, 1);
        assert!(receipts.take(id, step(8)).is_none());
        assert!(receipts.take(uuid::Uuid::new_v4(), step(7)).is_none());
        assert!(receipts.take(id, step(7)).is_some());
    }

    #[test]
    fn forwards_never_answered_do_not_pile_up() {
        let mut receipts = ForwardReceipts::default();
        let peer = libp2p::PeerId::random();
        let first = uuid::Uuid::new_v4();
        receipts.note_ack(first, step(0), Instant::now(), Duration::ZERO, peer, 1);
        for i in 0..(MOST_KEPT as u32 + 10) {
            receipts.note_ack(
                uuid::Uuid::new_v4(),
                step(i),
                Instant::now(),
                Duration::ZERO,
                peer,
                1,
            );
        }
        assert_eq!(receipts.len(), MOST_KEPT);
        assert_eq!(receipts.order.len(), MOST_KEPT);
        assert!(
            receipts.take(first, step(0)).is_none(),
            "the oldest went first"
        );
    }
}
