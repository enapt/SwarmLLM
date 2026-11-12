//! How long this node held each request's forwards as a hop of a chained split
//! another node coordinates (#427), reported to that coordinator once its reply
//! is over (`SwarmMessage::HopTimings`, `features::HOP_TIMINGS`).
//!
//! A coordinator times a chain only end to end: a middle hop hands its output
//! to the next hop, and the tail answers a coordinator that did not send it the
//! forward, so no hop's answer carries its own time (`LayerResult::served_us`
//! goes only to the peer that sent the forward, `served_time`). Without it the
//! coordinator prices every hop of a chain from its stated speed, or from
//! prompt passes that carry none of the per-visit cost (#427: 545 ms predicted
//! a pass against ~600 measured).
//!
//! **Each hop reports straight to the coordinator, once a reply** — as
//! OpenTelemetry's services export their own finished spans to the collector,
//! batched, never through the next service. A list each hop appended to on the
//! forward (in-band network telemetry's metadata stack) was rejected: every hop
//! re-seals the forward, so a later hop could rewrite an earlier one's entry.
//!
//! **Held** runs from the forward's arrival here (`ServedSince`, keyed by the
//! peer that sent it, stamped before its seal is opened) to the moment this
//! node starts handing it on or answering — the inbound open, the dispatch,
//! the wait for the stream's turn, the worker and the queue back to the
//! network task: what the coordinator pays this hop a visit beyond the link.
//! The onward seal runs after (off the loop, ~50-200 µs) and is not in it. One-position decode steps only, as the worker's own
//! published speed is (`OwnDecodeSpeed`): a prompt pass or a check of several
//! positions is other work.
//!
//! **Bounded**: [`MOST_REQUESTS`] requests, the first [`MOST_STEPS`] steps of
//! each, a request forgotten [`FORGET_AFTER`] after its last step — and each
//! peer that sends this node forwards its own share, [`MOST_REQUESTS_A_PEER`],
//! as `ServedSince` keeps one: a peer minting request ids pushes out only its
//! own records, never another coordinator's (security review 2026-11-12;
//! Bitcoin Core evicts from the group holding the most the same way). A
//! report goes only to the coordinator the request's FIRST step named, and is
//! taken once: a forward naming another coordinator, width or sending peer for
//! the same request id spoils the record rather than redirect it.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use crate::types::{HopTimings, NodeId};

/// Requests remembered at once; past it the one whose last step is oldest is
/// forgotten. A chain hop serves a few replies at a time.
const MOST_REQUESTS: usize = 256;

/// Requests one sending peer's forwards may hold at once; past it its own
/// stalest goes.
const MOST_REQUESTS_A_PEER: usize = 64;

/// Steps kept per request — a median of the first this many.
const MOST_STEPS: usize = 512;

/// A request whose coordinator never asked is forgotten this long after its
/// last step (a coordinator that went away, or one that never cancels).
const FORGET_AFTER: Duration = Duration::from_secs(30 * 60);

struct Held {
    /// The peer whose forwards these are (the authenticated sender, never an
    /// address): whose share the record counts in.
    from: libp2p::PeerId,
    coordinator: NodeId,
    layers: u32,
    held_us: Vec<u32>,
    steps: u32,
    /// Two coordinators, layer counts or sending peers for one request: its
    /// steps are not one hop's work for one coordinator, so nothing is told.
    spoiled: bool,
    last: Instant,
}

/// What this node held each chained request's decode steps for.
#[derive(Default)]
pub(super) struct HopHeld {
    requests: HashMap<uuid::Uuid, Held>,
}

impl HopHeld {
    /// A decode step of `request_id` for `coordinator`, `layers` wide, sent
    /// here by `from`, was held here `held_us`.
    pub(super) fn note(
        &mut self,
        request_id: uuid::Uuid,
        from: &libp2p::PeerId,
        coordinator: &NodeId,
        layers: u32,
        held_us: u32,
        now: Instant,
    ) {
        if !self.requests.contains_key(&request_id) {
            self.make_room_for(from);
        }
        let held = self.requests.entry(request_id).or_insert_with(|| Held {
            from: *from,
            coordinator: coordinator.clone(),
            layers,
            held_us: Vec::new(),
            steps: 0,
            spoiled: false,
            last: now,
        });
        held.last = now;
        if held.coordinator != *coordinator || held.layers != layers || held.from != *from {
            held.spoiled = true;
            held.held_us = Vec::new();
            return;
        }
        if held.spoiled {
            return;
        }
        held.steps = held.steps.saturating_add(1);
        if held.held_us.len() < MOST_STEPS {
            held.held_us.push(held_us);
        }
    }

    /// Room for one more of `from`'s requests: past its share, its own stalest
    /// goes; with every place taken, the stalest of the peer holding the most.
    fn make_room_for(&mut self, from: &libp2p::PeerId) {
        let mut held_by: HashMap<libp2p::PeerId, usize> = HashMap::new();
        for held in self.requests.values() {
            *held_by.entry(held.from).or_default() += 1;
        }
        let whose = if held_by.get(from).copied().unwrap_or(0) >= MOST_REQUESTS_A_PEER {
            *from
        } else if self.requests.len() >= MOST_REQUESTS {
            match held_by.into_iter().max_by_key(|(_, n)| *n) {
                Some((most, _)) => most,
                None => return,
            }
        } else {
            return;
        };
        let stalest = self
            .requests
            .iter()
            .filter(|(_, held)| held.from == whose)
            .min_by_key(|(_, held)| held.last)
            .map(|(id, _)| *id);
        if let Some(stalest) = stalest {
            self.requests.remove(&stalest);
        }
    }

    /// `by` asked (its `CancelInference` for `request_id`): this node's report,
    /// where `by` is the coordinator the request's steps were held for. Taken
    /// once. Another node asking leaves the record for the coordinator.
    pub(super) fn report_for(&mut self, request_id: uuid::Uuid, by: &NodeId) -> Option<HopTimings> {
        if self.requests.get(&request_id)?.coordinator != *by {
            return None;
        }
        let mut held = self.requests.remove(&request_id)?;
        if held.spoiled || held.held_us.is_empty() {
            return None;
        }
        held.held_us.sort_unstable();
        Some(HopTimings {
            request_id,
            layers: held.layers,
            steps: held.steps,
            median_held_us: held.held_us[held.held_us.len() / 2],
        })
    }

    /// Forget requests whose last step is older than [`FORGET_AFTER`];
    /// returns how many.
    pub(super) fn expire(&mut self, now: Instant) -> usize {
        let before = self.requests.len();
        self.requests
            .retain(|_, held| now.saturating_duration_since(held.last) < FORGET_AFTER);
        before - self.requests.len()
    }
}

impl super::NetworkManager {
    /// The forward `held` describes is leaving this node now — handed on, or
    /// its tail's answer to the coordinator (`NetworkCommand::NoteHeldAsHop`,
    /// sent just before that send). Its arrival is taken either way (nobody
    /// else will answer that peer for it); a decode step's held time is kept
    /// for the coordinator, never this node's own requests.
    pub(super) fn note_held_as_hop(&mut self, held: crate::types::HeldAsHop, now: Instant) {
        let Ok(predecessor) = libp2p::PeerId::from_bytes(&held.predecessor_peer_bytes) else {
            return;
        };
        let held_us = self
            .served_since
            .answered(&predecessor, held.request_id, held.step, now);
        if !held.decode_step || held.coordinator == *self.shared_state.identity.node_id() {
            return;
        }
        if let Some(us) = held_us {
            self.hop_held.note(
                held.request_id,
                &predecessor,
                &held.coordinator,
                held.layers,
                us,
                now,
            );
        }
    }

    /// `by` cancelled `request_id` — its reply is over. Where `by` is the
    /// coordinator this node held the request's steps for, and it takes them,
    /// tell it (`SwarmMessage::HopTimings`): over this node's own session,
    /// gated at the sender.
    pub(super) fn report_hop_timings(&mut self, request_id: uuid::Uuid, by: &NodeId) {
        let Some(report) = self.hop_held.report_for(request_id, by) else {
            return;
        };
        if !self
            .shared_state
            .peer_advertises_feature(by, crate::types::features::HOP_TIMINGS)
        {
            return;
        }
        let Some(target) = self.shared_state.resolve_connected_peer_id_bytes(by) else {
            return;
        };
        tracing::debug!(
            %request_id,
            coordinator = %by,
            layers = report.layers,
            steps = report.steps,
            median_held_us = report.median_held_us,
            "DIAG: told the coordinator how long this hop held its chain's steps"
        );
        self.handle_send_rr_message(
            target,
            crate::types::SwarmMessage::HopTimings(report),
            "HopTimings",
            None,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node(b: u8) -> NodeId {
        NodeId([b; 32])
    }

    fn peer() -> libp2p::PeerId {
        libp2p::PeerId::random()
    }

    /// A hop tells the coordinator its steps' median held time, once, and only
    /// the coordinator its steps named.
    #[test]
    fn the_coordinator_is_told_the_median_once() {
        let mut held = HopHeld::default();
        let rid = uuid::Uuid::new_v4();
        let t0 = Instant::now();
        let from = peer();
        for us in [40_000, 9_000, 12_000, 11_000, 10_000] {
            held.note(rid, &from, &node(1), 16, us, t0);
        }
        assert_eq!(held.report_for(rid, &node(2)), None, "not its coordinator");
        assert_eq!(
            held.report_for(rid, &node(1)),
            Some(HopTimings {
                request_id: rid,
                layers: 16,
                steps: 5,
                median_held_us: 11_000,
            }),
            "the median, so one slow step (40 ms) does not move it"
        );
        assert_eq!(held.report_for(rid, &node(1)), None, "taken once");
    }

    /// A forward naming another coordinator, width or sending peer for the
    /// same request spoils the record: it never redirects the report.
    #[test]
    fn a_second_coordinator_or_width_spoils_the_record() {
        let mut held = HopHeld::default();
        let t0 = Instant::now();
        let from = peer();
        let rid = uuid::Uuid::new_v4();
        held.note(rid, &from, &node(1), 16, 10_000, t0);
        held.note(rid, &from, &node(9), 16, 10_000, t0);
        assert_eq!(held.report_for(rid, &node(9)), None);
        assert_eq!(held.report_for(rid, &node(1)), None, "spoiled for both");
        let other = uuid::Uuid::new_v4();
        held.note(other, &from, &node(1), 16, 10_000, t0);
        held.note(other, &from, &node(1), 8, 10_000, t0);
        assert_eq!(held.report_for(other, &node(1)), None);
        let third = uuid::Uuid::new_v4();
        held.note(third, &from, &node(1), 16, 10_000, t0);
        held.note(third, &peer(), &node(1), 16, 10_000, t0);
        assert_eq!(held.report_for(third, &node(1)), None, "another sender");
    }

    /// Bounded in requests and in steps, and a request nobody asked about is
    /// forgotten.
    #[test]
    fn the_record_is_bounded_and_forgets() {
        let mut held = HopHeld::default();
        let t0 = Instant::now();
        let senders: Vec<_> = (0..8).map(|_| peer()).collect();
        let first = uuid::Uuid::new_v4();
        held.note(first, &senders[0], &node(1), 4, 1_000, t0);
        for i in 0..MOST_REQUESTS {
            held.note(
                uuid::Uuid::new_v4(),
                &senders[i % senders.len()],
                &node(1),
                4,
                1_000,
                t0 + Duration::from_millis(1 + i as u64),
            );
        }
        assert_eq!(held.requests.len(), MOST_REQUESTS);
        assert!(!held.requests.contains_key(&first), "the stalest went");
        let long = uuid::Uuid::new_v4();
        let late = t0 + Duration::from_secs(1);
        for _ in 0..MOST_STEPS + 10 {
            held.note(long, &senders[1], &node(1), 4, 1_000, late);
        }
        assert_eq!(held.requests[&long].held_us.len(), MOST_STEPS);
        assert_eq!(held.requests[&long].steps as usize, MOST_STEPS + 10);
        assert_eq!(held.expire(late + FORGET_AFTER), MOST_REQUESTS);
        assert!(held.requests.is_empty());
    }

    /// A peer minting request ids — each one a forward this node ran — pushes
    /// out only its own records, never another coordinator's live chain
    /// (security review 2026-11-12, as `ServedSince` keeps a share per sender).
    #[test]
    fn a_flooding_peer_pushes_out_only_its_own() {
        let mut held = HopHeld::default();
        let t0 = Instant::now();
        let honest = peer();
        let live = uuid::Uuid::new_v4();
        held.note(live, &honest, &node(1), 16, 10_000, t0);
        let flood = peer();
        for i in 0..MOST_REQUESTS * 2 {
            let at = t0 + Duration::from_millis(1 + i as u64);
            held.note(uuid::Uuid::new_v4(), &flood, &node(7), 16, 1_000, at);
        }
        let flooded = held.requests.values().filter(|h| h.from == flood).count();
        assert_eq!(flooded, MOST_REQUESTS_A_PEER);
        assert!(
            held.requests.contains_key(&live),
            "past its share, its own went"
        );
        // Many flooders fill every place: the peer holding the most gives way,
        // never the one with a single live chain.
        let floods: Vec<_> = (0..MOST_REQUESTS / MOST_REQUESTS_A_PEER + 2)
            .map(|_| peer())
            .collect();
        for (n, f) in floods.iter().enumerate() {
            for i in 0..MOST_REQUESTS_A_PEER {
                let at = t0 + Duration::from_secs(1 + n as u64) + Duration::from_millis(i as u64);
                held.note(uuid::Uuid::new_v4(), f, &node(7), 16, 1_000, at);
            }
        }
        assert_eq!(held.requests.len(), MOST_REQUESTS);
        assert!(
            held.requests.contains_key(&live),
            "the stalest record, kept"
        );
    }
}
