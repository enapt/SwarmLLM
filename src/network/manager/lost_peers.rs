//! A lost peer is looked up in the DHT, so a new address of its is found (#465).
//!
//! A node that loses a peer redials the addresses it holds (2-5 s, then +5, +15,
//! +45, +120, +300 s) and its discovery tick dials the same ones from the peer
//! cache. Nothing asked where the peer is NOW: identify does not push a new
//! address, and peer exchange names a few peers an answer. A computer that
//! cannot be dialled itself — the Australian peer on 2026-11-16 — then never
//! came back to a peer whose public address or port had changed, and when a
//! rig answered its dial under another identity it forgot the address it had
//! (#155's rule) and never came back at all.
//!
//! **Prior art**: go-libp2p's routed host looks a peer up in the DHT
//! (`FindPeer`) when it has no address to dial. Here a lost peer that was
//! directly dialable is looked up — `kademlia.get_closest_peers(peer)`, whose
//! walk dials the peer at whatever address its neighbours hold (libp2p-kad 0.48
//! lists in the answer only the peers the walk REACHED) — first a minute after
//! the loss, then at a gap that doubles to an hour, for a day. A peer nobody
//! can dial is not looked up: no answer could reach it from here.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use libp2p::PeerId;

/// The first lookup waits this long after the loss: the redial goes first.
pub(super) const FIRST_LOOKUP_AFTER: Duration = Duration::from_secs(60);
/// The gap between lookups of one peer doubles to this.
pub(super) const MOST_BETWEEN_LOOKUPS: Duration = Duration::from_secs(3600);
/// A peer not back after this long is forgotten here.
pub(super) const LOOK_UP_FOR: Duration = Duration::from_secs(24 * 3600);
/// Lookups started in one liveness tick (30 s), at most.
pub(super) const LOOKUPS_A_TICK: usize = 4;
/// Lost peers remembered at once; past it the longest lost goes.
pub(super) const MOST_LOST_PEERS: usize = 256;

struct Lost {
    since: Instant,
    next: Instant,
    gap: Duration,
}

/// Peers remembered as directly dialable at once; past it the longest unseen goes.
pub(super) const MOST_DIALABLE_PEERS: usize = 1024;

/// Peers this node lost and is looking for, and the peers it may look for.
#[derive(Default)]
pub(super) struct LostPeers {
    lost: HashMap<PeerId, Lost>,
    /// Peers that named a direct internet address at an Identify this run
    /// (`worth_looking_up`), and when last. Kept apart from the registry: a
    /// computer stopping reconnects for a moment on its way out, and that last
    /// Identify may name only a private address — judged from it, the loss
    /// that followed was never looked up (probe, 2026-11-16).
    dialable: HashMap<PeerId, Instant>,
}

impl LostPeers {
    /// `peer` named a direct internet address at an Identify.
    pub(super) fn note_dialable(&mut self, peer: PeerId, now: Instant) {
        if !self.dialable.contains_key(&peer) && self.dialable.len() >= MOST_DIALABLE_PEERS {
            if let Some(oldest) = self
                .dialable
                .iter()
                .min_by_key(|(_, at)| **at)
                .map(|(p, _)| *p)
            {
                self.dialable.remove(&oldest);
            }
        }
        self.dialable.insert(peer, now);
    }

    /// `peer`'s last connection closed. A peer that ever named a direct
    /// internet address this run is looked for: its first lookup is due
    /// [`FIRST_LOOKUP_AFTER`] from now; a peer already lost keeps its schedule.
    /// Any other is not — no answer could reach it from here.
    pub(super) fn note_lost(&mut self, peer: PeerId, now: Instant) {
        if self.lost.contains_key(&peer) || !self.dialable.contains_key(&peer) {
            return;
        }
        if self.lost.len() >= MOST_LOST_PEERS {
            if let Some(oldest) = self
                .lost
                .iter()
                .min_by_key(|(_, l)| l.since)
                .map(|(p, _)| *p)
            {
                self.lost.remove(&oldest);
            }
        }
        self.lost.insert(
            peer,
            Lost {
                since: now,
                next: now + FIRST_LOOKUP_AFTER,
                gap: FIRST_LOOKUP_AFTER,
            },
        );
    }

    /// `peer` is connected again.
    pub(super) fn note_found(&mut self, peer: &PeerId) {
        self.lost.remove(peer);
    }

    /// The lost peers to look up now — at most [`LOOKUPS_A_TICK`], the longest
    /// overdue first; each chosen is scheduled again at twice its gap (at most
    /// [`MOST_BETWEEN_LOOKUPS`]). A peer lost longer than [`LOOK_UP_FOR`], or
    /// connected (`connected`), is forgotten.
    ///
    /// A peer that cannot be looked up this tick (`waits`: cut off for its
    /// release, or a lookup of it still running) is passed over: it takes none
    /// of the tick's lookups and keeps its turn. Chosen and then skipped by the
    /// caller, it lost a turn, its gap doubled with no lookup made, and a peer
    /// that was due waited a tick (review of #465, 2026-11-17). Prior art:
    /// gRPC's connection backoff (`doc/connection-backoff.md`) multiplies the
    /// backoff once per failed ATTEMPT, never for a turn that made none.
    pub(super) fn due(
        &mut self,
        now: Instant,
        connected: impl Fn(&PeerId) -> bool,
        waits: impl Fn(&PeerId) -> bool,
    ) -> Vec<PeerId> {
        self.lost
            .retain(|p, l| now.saturating_duration_since(l.since) < LOOK_UP_FOR && !connected(p));
        let mut due: Vec<(Instant, PeerId)> = self
            .lost
            .iter()
            .filter(|(p, l)| l.next <= now && !waits(p))
            .map(|(p, l)| (l.next, *p))
            .collect();
        due.sort();
        due.truncate(LOOKUPS_A_TICK);
        for (_, peer) in &due {
            if let Some(l) = self.lost.get_mut(peer) {
                l.gap = (l.gap * 2).min(MOST_BETWEEN_LOOKUPS);
                l.next = now + l.gap;
            }
        }
        due.into_iter().map(|(_, p)| p).collect()
    }

    pub(super) fn len(&self) -> usize {
        self.lost.len()
    }
}

/// Whether a lost peer is worth looking up: it had an address on the open
/// internet that is not a relay circuit. A peer nobody can dial is found only
/// when it dials us.
pub(super) fn worth_looking_up(advertised: &[String]) -> bool {
    advertised.iter().any(|a| {
        a.parse::<libp2p::Multiaddr>().is_ok_and(|m| {
            crate::network::relay::addr_is_direct_transport(&m)
                && crate::pool::invite::multiaddr_is_internet_reachable(&m)
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_lost_peer_is_looked_up_after_a_minute_then_less_and_less_often() {
        let mut lost = LostPeers::default();
        let p = PeerId::random();
        let t0 = Instant::now();
        lost.note_dialable(p, t0);
        lost.note_lost(p, t0);
        assert!(
            lost.due(t0, |_| false, |_| false).is_empty(),
            "the redial goes first"
        );
        assert!(lost
            .due(t0 + Duration::from_secs(59), |_| false, |_| false)
            .is_empty());
        let mut at = t0 + FIRST_LOOKUP_AFTER;
        let mut gaps = Vec::new();
        let mut last = at;
        for _ in 0..12 {
            // Step a tick at a time until it is due.
            while lost.due(at, |_| false, |_| false).is_empty() {
                at += Duration::from_secs(30);
            }
            gaps.push(at.saturating_duration_since(last).as_secs());
            last = at;
            at += Duration::from_secs(30);
        }
        // 0 (the first), then 120, 240, 480, … capped at an hour.
        assert_eq!(gaps[0], 0);
        assert!(
            gaps.windows(2).all(|w| w[1] >= w[0]),
            "gaps never shrink: {gaps:?}"
        );
        assert_eq!(
            *gaps.last().unwrap(),
            MOST_BETWEEN_LOOKUPS.as_secs(),
            "{gaps:?}"
        );
        // Losing it again does not start the schedule over.
        lost.note_lost(p, at);
        assert!(lost
            .due(at + FIRST_LOOKUP_AFTER, |_| false, |_| false)
            .is_empty());
    }

    #[test]
    fn a_found_or_long_lost_peer_is_not_looked_up() {
        let mut lost = LostPeers::default();
        let (a, b, c) = (PeerId::random(), PeerId::random(), PeerId::random());
        let t0 = Instant::now();
        for p in [a, b, c] {
            lost.note_dialable(p, t0);
            lost.note_lost(p, t0);
        }
        lost.note_found(&a);
        let due = lost.due(t0 + FIRST_LOOKUP_AFTER, |p| *p == b, |_| false);
        assert_eq!(due, vec![c], "found and connected peers are not looked up");
        assert_eq!(lost.len(), 1);
        assert!(
            lost.due(t0 + LOOK_UP_FOR, |_| false, |_| false).is_empty(),
            "a day lost: forgotten"
        );
        assert_eq!(lost.len(), 0);
    }

    #[test]
    fn a_tick_starts_a_few_lookups_and_the_list_is_bounded() {
        let mut lost = LostPeers::default();
        let t0 = Instant::now();
        for i in 0..(MOST_LOST_PEERS + 10) {
            let p = PeerId::random();
            lost.note_dialable(p, t0);
            lost.note_lost(p, t0 + Duration::from_millis(i as u64));
        }
        assert_eq!(lost.len(), MOST_LOST_PEERS);
        let at = t0 + FIRST_LOOKUP_AFTER + Duration::from_secs(1);
        assert_eq!(lost.due(at, |_| false, |_| false).len(), LOOKUPS_A_TICK);
    }

    #[test]
    fn a_peer_passed_over_keeps_its_turn_and_takes_no_lookup() {
        // Review of #465 (2026-11-17): a peer cut off for its release, or one
        // whose lookup was still running, was chosen, scheduled again at twice
        // its gap and then skipped — its turn lost, and one of the tick's
        // lookups taken from a peer that was due.
        let mut lost = LostPeers::default();
        let t0 = Instant::now();
        let waiting = PeerId::random();
        lost.note_dialable(waiting, t0);
        lost.note_lost(waiting, t0); // the longest overdue
        for i in 0..LOOKUPS_A_TICK {
            let p = PeerId::random();
            lost.note_dialable(p, t0);
            lost.note_lost(p, t0 + Duration::from_millis(1 + i as u64));
        }
        let at = t0 + FIRST_LOOKUP_AFTER + Duration::from_secs(1);
        let due = lost.due(at, |_| false, |p| *p == waiting);
        assert!(!due.contains(&waiting));
        assert_eq!(
            due.len(),
            LOOKUPS_A_TICK,
            "the passed-over peer took no lookup"
        );
        // Its wait over, it is looked up at the next tick, not a doubled gap later.
        let next_tick = at + Duration::from_secs(30);
        assert_eq!(lost.due(next_tick, |_| false, |_| false), vec![waiting]);
    }

    #[test]
    fn a_peer_that_flapped_on_its_way_out_is_still_looked_up() {
        // Seen 2026-11-16: the stopping computer reconnected for a moment, named
        // only a private address, and closed — the loss judged from that last
        // Identify was never looked up.
        let mut lost = LostPeers::default();
        let (p, never) = (PeerId::random(), PeerId::random());
        let t0 = Instant::now();
        lost.note_dialable(p, t0); // the Identify that named its public address
        lost.note_lost(p, t0);
        lost.note_found(&p); // the moment's reconnect …
        lost.note_lost(p, t0 + Duration::from_secs(4)); // … and the close after it
        lost.note_lost(never, t0); // never named a direct internet address
        let due = lost.due(
            t0 + Duration::from_secs(4) + FIRST_LOOKUP_AFTER,
            |_| false,
            |_| false,
        );
        assert_eq!(due, vec![p]);
    }

    #[test]
    fn only_a_peer_with_a_direct_internet_address_is_looked_up() {
        let p = PeerId::random();
        assert!(worth_looking_up(&[format!(
            "/ip4/171.97.115.249/tcp/8810/p2p/{p}"
        )]));
        assert!(worth_looking_up(&[
            "/ip4/192.168.1.5/tcp/8810".into(),
            "/ip4/8.8.4.4/udp/8800/quic-v1".into()
        ]));
        // The Australian peer's view of itself: Docker, Tailscale, LAN, a relay circuit.
        assert!(!worth_looking_up(&[
            "/ip4/172.18.0.1/tcp/8810".into(),
            "/ip4/100.70.76.52/tcp/8810".into(),
            "/ip4/192.168.88.254/udp/8800/quic-v1".into(),
            format!("/ip4/212.132.104.177/tcp/8810/p2p/{p}/p2p-circuit"),
        ]));
        assert!(!worth_looking_up(&[]));
    }
}
