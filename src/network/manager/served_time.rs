//! When each forward arrived here, so its answer can say how long this node
//! held it (#388: `LayerResult::served_us`, `features::SERVED_TIME`).
//!
//! The coordinator's round trip is the link both ways plus this node's time.
//! Told this node's time, it can tell a queue on the LINK — which sending
//! fewer rows shortens — from this computer being slow on its own, which it
//! does not. Swift splits a round trip the same way (Kumar et al., SIGCOMM
//! 2020 § 3.2: the receiver echoes its own time in the ACK, the sender takes
//! it off to leave the fabric's); QUIC's `ack_delay` is the same idea for an
//! ACK the receiver held (RFC 9002 § 5.3). The receipt ACK alone cannot split
//! it: quinn sends streams of one priority in turn (`send_fairness`), so a
//! small receipt passes answer bytes queued on this node's uplink.
//!
//! The clock starts when the network task decodes the forward and stops when
//! it hands the answer to the transport. Dispatch, the wait for the stream's
//! turn, the worker and the command queue back are this node's; the
//! transport's send buffer and the wire are the link's — a queue on this
//! node's uplink IS a queue on the link, which is what the guard is for.
//! A network-loop stall before the decode is counted as the link's (the
//! stall tripwire logs one past `NETWORK_LOOP_STALL_WARN`).
//!
//! **Kept per sending peer** (security review of #388): every forward is
//! remembered under the peer libp2p authenticated it from (never anything an
//! address says, gotcha #888), each peer in a bounded share of its own, so a
//! peer sending forwards it never means to see answered pushes out only its
//! own. An answer reads the share of the peer it goes to: a run's coordinator
//! is the peer that sent its forward; a chained hop's answer goes to a
//! coordinator that did not, and says nothing — a chain reads whole delays.
//!
//! **Rounded to the millisecond**: the coordinator's guard reads milliseconds,
//! and the microseconds would show it this computer's other work to the
//! microsecond.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use crate::types::ResultStep;

/// Forwards remembered per generation, per peer. Two generations are kept, so
/// an arrival is remembered for between one and two generations of that
/// peer's later arrivals: a decode answer leaves within one round trip, and a
/// flow keeps at most 12 forwards out a request. A forward never answered here
/// — a chained hop hands its output on — costs one entry until its generation
/// is dropped.
const GENERATION: usize = 64;

/// Peers remembered at once. Past it, the peer whose latest forward is the
/// oldest is forgotten — a peer gone, never one sending now. Above the default
/// connection caps (150-300, `NetworkConfig::effective_max_connections`), so
/// connected coordinators do not push each other out.
const MOST_PEERS: usize = 512;

type Key = (uuid::Uuid, ResultStep);

/// One peer's forwards not yet answered — at most two generations.
#[derive(Default)]
struct Share {
    current: HashMap<Key, Instant>,
    previous: HashMap<Key, Instant>,
    latest: Option<Instant>,
}

impl Share {
    fn arrived(&mut self, key: Key, at: Instant) {
        if self.current.len() >= GENERATION {
            self.previous = std::mem::take(&mut self.current);
        }
        self.current.insert(key, at);
        self.latest = Some(self.latest.map_or(at, |l| l.max(at)));
    }

    fn take(&mut self, key: &Key) -> Option<Instant> {
        self.current
            .remove(key)
            .or_else(|| self.previous.remove(key))
    }

    fn is_empty(&self) -> bool {
        self.current.is_empty() && self.previous.is_empty()
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.current.len() + self.previous.len()
    }
}

/// Arrival times of the forwards not yet answered, per sending peer.
#[derive(Default)]
pub(super) struct ServedSince {
    peers: HashMap<libp2p::PeerId, Share>,
}

impl ServedSince {
    /// A forward for `step` of `request_id` arrived from `peer` at `at`.
    pub(super) fn arrived(
        &mut self,
        peer: libp2p::PeerId,
        request_id: uuid::Uuid,
        step: ResultStep,
        at: Instant,
    ) {
        if !self.peers.contains_key(&peer) && self.peers.len() >= MOST_PEERS {
            let stalest = self
                .peers
                .iter()
                .min_by_key(|(_, share)| share.latest)
                .map(|(p, _)| *p);
            if let Some(stalest) = stalest {
                self.peers.remove(&stalest);
            }
        }
        self.peers
            .entry(peer)
            .or_default()
            .arrived((request_id, step), at);
    }

    /// The answer to `step` leaves now for `peer`: how long this node held its
    /// forward, in microseconds rounded to the millisecond — `None` for a
    /// forward that peer did not send, or one not remembered. Forgets it
    /// either way.
    pub(super) fn answered(
        &mut self,
        peer: &libp2p::PeerId,
        request_id: uuid::Uuid,
        step: ResultStep,
        now: Instant,
    ) -> Option<u32> {
        let share = self.peers.get_mut(peer)?;
        let at = share.take(&(request_id, step));
        if share.is_empty() {
            self.peers.remove(peer);
        }
        Some(micros_to_the_millisecond(
            now.saturating_duration_since(at?),
        ))
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.peers.values().map(Share::len).sum()
    }
}

/// A duration on the wire: microseconds, rounded to the nearest millisecond;
/// an hour and more saturates.
pub(super) fn micros_to_the_millisecond(held: Duration) -> u32 {
    let ms = (held.as_micros() + 500) / 1000;
    u32::try_from(ms * 1000).unwrap_or(u32::MAX)
}

/// What an answer tells the computer it goes to about how this node served
/// the forward — the ONE rule for both ways an answer leaves: as its own
/// request (`NetworkManager::handle_send_tensor_result`) and as a frame on a
/// persistent pipeline stream (`pipeline_stream::handle_inbound_stream`). The
/// stream's frame once left with the model process's report unrounded and to
/// a coordinator that never advertised it (security review 2026-11-13).
///
/// Each field goes only where the recipient SENT the forward — a chained
/// tail's report would be read as the run's head's — and READS it (`reads`,
/// asked of the recipient's advertised features), to the millisecond;
/// otherwise it is cleared, so nothing the pool filled leaves by accident.
/// `held` is this node's time for the recipient's own forward
/// ([`ServedSince::answered`]); a stream does not keep one.
pub(in crate::network) fn tell_the_recipient(
    result: &mut crate::types::LayerResult,
    sent_the_forward: bool,
    held: Option<u32>,
    reads: impl Fn(u64) -> bool,
) {
    use crate::types::features::{HOW_SERVED, SERVED_TIME};
    let to_the_ms = |us: u32| micros_to_the_millisecond(Duration::from_micros(u64::from(us)));
    result.how_served = result
        .how_served
        .filter(|_| sent_the_forward && reads(HOW_SERVED))
        .map(|how| crate::types::HowServed {
            load_us: to_the_ms(how.load_us),
            worker_us: to_the_ms(how.worker_us),
            ..how
        });
    result.served_us = held
        .filter(|_| sent_the_forward && reads(SERVED_TIME))
        .map(to_the_ms);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn step(seq: u32) -> ResultStep {
        ResultStep {
            index_pos: 40,
            layer_range: (0, 24),
            stream_seq: Some(seq),
        }
    }

    #[test]
    fn an_answer_reports_the_time_since_its_forward_arrived_once() {
        let mut served = ServedSince::default();
        let peer = libp2p::PeerId::random();
        let rid = uuid::Uuid::new_v4();
        let t0 = Instant::now();
        served.arrived(peer, rid, step(1), t0);
        served.arrived(peer, rid, step(2), t0 + Duration::from_millis(5));
        assert_eq!(
            served.answered(&peer, rid, step(2), t0 + Duration::from_millis(47)),
            Some(42_000)
        );
        // Answered once: a second answer to the same step has no time.
        assert_eq!(
            served.answered(&peer, rid, step(2), t0 + Duration::from_millis(60)),
            None
        );
        // Another request's step of the same shape is not this one.
        assert_eq!(
            served.answered(
                &peer,
                uuid::Uuid::new_v4(),
                step(1),
                t0 + Duration::from_millis(60)
            ),
            None
        );
        // Nor is an answer going to a peer that did not send it (a chain's
        // coordinator).
        assert_eq!(
            served.answered(
                &libp2p::PeerId::random(),
                rid,
                step(1),
                t0 + Duration::from_millis(60)
            ),
            None
        );
        assert_eq!(
            served.answered(&peer, rid, step(1), t0 + Duration::from_millis(60)),
            Some(60_000)
        );
        assert_eq!(served.len(), 0);
        assert!(
            served.peers.is_empty(),
            "a peer with nothing out is forgotten"
        );
    }

    /// Forwards never answered here cannot grow a peer's share past two
    /// generations, and a forward still out is remembered through one turnover.
    #[test]
    fn unanswered_forwards_are_forgotten_two_generations_on() {
        let mut served = ServedSince::default();
        let peer = libp2p::PeerId::random();
        let rid = uuid::Uuid::new_v4();
        let t0 = Instant::now();
        served.arrived(peer, rid, step(0), t0);
        for seq in 1..=(3 * GENERATION as u32) {
            served.arrived(peer, rid, step(seq), t0);
            assert!(served.len() <= 2 * GENERATION);
            if seq == GENERATION as u32 {
                // One turnover: the first is in the previous generation.
                assert!(served.peers[&peer].previous.contains_key(&(rid, step(0))));
            }
        }
        assert_eq!(served.answered(&peer, rid, step(0), t0), None);
        let newest = 3 * GENERATION as u32;
        assert_eq!(served.answered(&peer, rid, step(newest), t0), Some(0));
    }

    /// The security review's case: a peer sending forwards it never sees
    /// answered pushes out only its own — another coordinator's forward out at
    /// the same time keeps its time.
    #[test]
    fn a_peer_flooding_forwards_pushes_out_only_its_own() {
        let mut served = ServedSince::default();
        let honest = libp2p::PeerId::random();
        let flooder = libp2p::PeerId::random();
        let rid = uuid::Uuid::new_v4();
        let t0 = Instant::now();
        served.arrived(honest, rid, step(7), t0);
        for seq in 0..(10 * GENERATION as u32) {
            served.arrived(flooder, uuid::Uuid::new_v4(), step(seq), t0);
        }
        assert!(served.len() <= 2 * GENERATION + 1);
        assert_eq!(
            served.answered(&honest, rid, step(7), t0 + Duration::from_millis(30)),
            Some(30_000)
        );
    }

    /// Past `MOST_PEERS`, the peer gone longest is forgotten, never one that
    /// sent a forward since.
    #[test]
    fn past_the_peer_bound_the_stalest_peer_is_forgotten() {
        let mut served = ServedSince::default();
        let rid = uuid::Uuid::new_v4();
        let t0 = Instant::now();
        let stale = libp2p::PeerId::random();
        served.arrived(stale, rid, step(0), t0);
        for i in 1..MOST_PEERS as u64 {
            served.arrived(
                libp2p::PeerId::random(),
                rid,
                step(0),
                t0 + Duration::from_millis(i),
            );
        }
        let busy = libp2p::PeerId::random();
        // At the bound: the newcomer takes the stalest peer's place.
        served.arrived(busy, rid, step(0), t0 + Duration::from_secs(10));
        assert_eq!(served.peers.len(), MOST_PEERS);
        assert!(!served.peers.contains_key(&stale));
        assert!(served.peers.contains_key(&busy));
    }

    fn served(load_us: u32, worker_us: u32) -> crate::types::LayerResult {
        let mut result: crate::types::LayerResult = serde_json::from_value(serde_json::json!({
            "request_id": uuid::Uuid::nil(),
            "token_ids": [],
            "finish_reason": null,
        }))
        .expect("a bare result");
        result.how_served = Some(crate::types::HowServed {
            layers: crate::types::LayersFrom::Copy,
            load_us,
            worker_us,
        });
        result.served_us = Some(77);
        result
    }

    #[test]
    fn an_answer_tells_only_the_sender_that_reads_it_and_to_the_millisecond() {
        use crate::types::features::{HOW_SERVED, SERVED_TIME};
        let every = |_: u64| true;
        // The forward's sender, reading both: both go, to the millisecond —
        // `served_us` from this node's own held time, never the pool's.
        let mut told = served(2_400, 31_499);
        tell_the_recipient(&mut told, true, Some(12_345), every);
        let how = told.how_served.expect("the sender reads it");
        assert_eq!((how.load_us, how.worker_us), (2_000, 31_000));
        assert_eq!(how.layers, crate::types::LayersFrom::Copy);
        assert_eq!(told.served_us, Some(12_000));
        // A recipient that did not send the forward is told neither.
        let mut chained = served(2_400, 31_499);
        tell_the_recipient(&mut chained, false, Some(12_000), every);
        assert_eq!((chained.how_served, chained.served_us), (None, None));
        // Each field only to a recipient that advertises it.
        let mut older = served(2_400, 31_499);
        tell_the_recipient(&mut older, true, Some(12_000), |f| f == SERVED_TIME);
        assert_eq!(older.how_served, None);
        assert_eq!(older.served_us, Some(12_000));
        let mut reads_how = served(2_400, 31_499);
        tell_the_recipient(&mut reads_how, true, None, |f| f == HOW_SERVED);
        assert!(reads_how.how_served.is_some());
        // No held time (a stream keeps none): the pool's value never leaves.
        assert_eq!(reads_how.served_us, None);
    }

    /// Both ways an answer leaves go through the one rule — the stream's
    /// frame skipped it before 2026-11-13.
    #[test]
    fn both_ways_an_answer_leaves_ask_the_one_rule() {
        for (path, src) in [
            ("network/manager/tensors.rs", include_str!("tensors.rs")),
            (
                "network/pipeline_stream.rs",
                include_str!("../pipeline_stream.rs"),
            ),
        ] {
            assert!(
                src.contains("served_time::tell_the_recipient("),
                "{path} sends a LayerResult without `tell_the_recipient`"
            );
        }
    }

    #[test]
    fn a_held_time_is_rounded_to_the_millisecond_and_an_hour_saturates() {
        assert_eq!(micros_to_the_millisecond(Duration::from_micros(7)), 0);
        assert_eq!(
            micros_to_the_millisecond(Duration::from_micros(41_499)),
            41_000
        );
        assert_eq!(
            micros_to_the_millisecond(Duration::from_micros(41_500)),
            42_000
        );
        assert_eq!(
            micros_to_the_millisecond(Duration::from_secs(5_000)),
            u32::MAX
        );
    }
}
