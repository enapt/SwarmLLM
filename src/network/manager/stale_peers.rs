//! What this node has stopped believing about other computers (#413).
//!
//! **(A) A (peer, address) pair another computer answered at** — a dial to the
//! peer reached somebody else there (`DialError::WrongPeerId`). #155 forgets the
//! pair in Kademlia's table and the peer registry, but the peer cache MERGES what
//! it stored before into every save (`peer_cache::merge_for_save`), so the pair
//! came back from the disk at every start, was seeded into Kademlia again
//! (`dial_bootstrap_peers`) and handed to every node that asked. [`ProvenWrong`]
//! keeps the pairs this run proved, so the cache drops them on read and on save.
//! Exactly the pair, never the peer — whoever answers at an address can name any
//! peer there (#155's rule).
//!
//! **(B) A peer every dial has failed for, an hour after it was last reached.**
//! libp2p-kad 0.48 forgets a failing address only after a transport error and
//! never a peer's LAST one (`address_failed`), and evicts nobody unless a bucket
//! is full. So an identity that is gone for good — every test node that ever
//! ran on this machine with a fresh key — stays in the routing table of every
//! node it once met, and every lookup that reaches one is answered with it: a
//! fresh node with an empty cache dialled 33-48 such identities 67-74 times in
//! its first 90 s (2026-11-06), and the LAN box 29 of them 418 times in the
//! 25 s after a restart, 21 learned from other nodes' answers. [`LastReached`]
//! says when a peer has been out of reach for [`UNREACHABLE_FOR`]; the dial
//! failure that finds it so takes it out of the table, and the cache stops
//! seeding it back (a dial of it still goes; a connection puts it back).
//!
//! Prior art: Kademlia (Maymounkov & Mazières, 2002) treats a contact that fails
//! to respond as stale and replaces it; go-libp2p-kad-dht's routing-table refresh
//! probes its peers and evicts the ones that fail (go-libp2p-kad-dht #283's
//! "missing" → "dead"). Ours evicts on the failures lookups already make, with an
//! hour's grace so a computer restarting or changing networks is not dropped.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use libp2p::{Multiaddr, PeerId};

/// Pairs remembered at most. Each is one identity at one address; a node meets a
/// few dozen dead identities, not thousands.
pub(super) const PROVEN_WRONG_PAIRS: usize = 512;

/// How long a peer must have been out of reach — since its last connection
/// closed, or since this node started — before a failed dial takes it out of
/// the routing table. Kademlia's own refresh is every 10 min
/// (`behaviour::KAD_REFRESH_SECS`), so a restart or a network change is never
/// mistaken for a departure.
pub(super) const UNREACHABLE_FOR: Duration = Duration::from_secs(60 * 60);

/// Peers whose last connection time is kept at most. Past it the oldest go,
/// which reads the same as never having been reached this run.
const LAST_REACHED_PEERS: usize = 4096;

/// An address as compared here: without its `/p2p/<peer>` part.
fn bare(address: &Multiaddr) -> Multiaddr {
    let mut bare = address.clone();
    if matches!(
        bare.iter().last(),
        Some(libp2p::multiaddr::Protocol::P2p(_))
    ) {
        bare.pop();
    }
    bare
}

/// (A): the (peer, address) pairs this run proved wrong.
#[derive(Default)]
pub(super) struct ProvenWrong {
    pairs: HashMap<(PeerId, Multiaddr), Instant>,
}

impl ProvenWrong {
    /// A dial to `peer` at `address` answered as another computer.
    pub(super) fn note(&mut self, peer: PeerId, address: &Multiaddr, now: Instant) {
        if self.pairs.len() >= PROVEN_WRONG_PAIRS
            && !self.pairs.contains_key(&(peer, bare(address)))
        {
            if let Some(oldest) = self
                .pairs
                .iter()
                .min_by_key(|(_, at)| **at)
                .map(|(pair, _)| pair.clone())
            {
                self.pairs.remove(&oldest);
            }
        }
        self.pairs.insert((peer, bare(address)), now);
    }

    /// Whether `cached` — a peer-cache entry, `<address>/p2p/<peer>` — names a
    /// pair proven wrong. An entry naming no peer names no pair.
    pub(super) fn names(&self, cached: &str) -> bool {
        crate::network::peer_cache::peer_and_address(cached)
            .is_some_and(|pair| self.pairs.contains_key(&pair))
    }

    /// The pairs, for the cache writer, which runs off this loop.
    pub(super) fn pairs(&self) -> Vec<(PeerId, Multiaddr)> {
        self.pairs.keys().cloned().collect()
    }

    /// `peer` connected: what was proven about it may no longer hold — it may
    /// be back at the address — so its pairs go (go-libp2p clears a peer's dial
    /// backoff on a connection the same way). Review of #413: kept, a pair
    /// would have dropped the returned peer's live address from every save.
    pub(super) fn reached(&mut self, peer: &PeerId) {
        self.pairs.retain(|(held, _), _| held != peer);
    }
}

/// (B): when each peer was last reached.
pub(super) struct LastReached {
    /// What a peer with no time of its own counts from: the node's start, or
    /// the newest time dropped to keep the table bounded — never earlier than
    /// any peer's real last reach, so a dropped time can only delay an eviction
    /// (review of #413: counting a dropped peer from the start evicted peers
    /// reached minutes before).
    floor: Instant,
    by_peer: HashMap<PeerId, Instant>,
}

impl LastReached {
    pub(super) fn new(started: Instant) -> Self {
        Self {
            floor: started,
            by_peer: HashMap::new(),
        }
    }

    /// `peer` is connected now, or its last connection just closed.
    pub(super) fn reached(&mut self, peer: PeerId, now: Instant) {
        if self.by_peer.len() >= LAST_REACHED_PEERS && !self.by_peer.contains_key(&peer) {
            self.prune(now);
            if self.by_peer.len() >= LAST_REACHED_PEERS {
                // Every time is fresh and the table is full: the older half goes,
                // and every peer without a time now counts from the newest of them.
                let mut times: Vec<Instant> = self.by_peer.values().copied().collect();
                times.sort_unstable();
                let cut = times[times.len() / 2];
                self.by_peer.retain(|_, at| *at > cut);
                self.floor = self.floor.max(cut);
            }
        }
        self.by_peer.insert(peer, now);
    }

    /// Whether `peer` has been out of reach for [`UNREACHABLE_FOR`]. The caller
    /// knows it is not connected now.
    pub(super) fn out_of_reach_long(&self, peer: &PeerId, now: Instant) -> bool {
        let since = self.by_peer.get(peer).copied().unwrap_or(self.floor);
        now.saturating_duration_since(since) >= UNREACHABLE_FOR
    }

    /// Drop the times that no longer decide anything: one older than
    /// [`UNREACHABLE_FOR`] reads the same as none (every kept time is at or
    /// after the floor).
    pub(super) fn prune(&mut self, now: Instant) {
        self.by_peer
            .retain(|_, at| now.saturating_duration_since(*at) < UNREACHABLE_FOR);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn addr(s: &str) -> Multiaddr {
        s.parse().unwrap()
    }

    /// A pair proven wrong is recognised in the cache's own form, with or
    /// without the `/p2p/` part the swarm adds — and only that pair: the same
    /// peer elsewhere, and another peer at the same address, are not named.
    #[test]
    fn a_pair_proven_wrong_names_exactly_that_cache_entry() {
        let mut wrong = ProvenWrong::default();
        let stale = PeerId::random();
        let lan = addr("/ip4/192.168.1.60/udp/8800/quic-v1");
        wrong.note(stale, &lan.clone().with_p2p(stale).unwrap(), Instant::now());
        assert!(wrong.names(&format!("{lan}/p2p/{stale}")));
        assert!(!wrong.names(&format!("/ip4/198.51.100.9/tcp/8810/p2p/{stale}")));
        assert!(!wrong.names(&format!("{lan}/p2p/{}", PeerId::random())));
        assert!(!wrong.names(&lan.to_string()), "an entry naming no peer");
    }

    #[test]
    fn the_pairs_are_bounded_oldest_first() {
        let mut wrong = ProvenWrong::default();
        let start = Instant::now();
        let first = PeerId::random();
        let lan = addr("/ip4/192.168.1.60/tcp/8810");
        wrong.note(first, &lan, start);
        for n in 1..=PROVEN_WRONG_PAIRS {
            wrong.note(
                PeerId::random(),
                &lan,
                start + Duration::from_secs(n as u64),
            );
        }
        assert_eq!(wrong.pairs().len(), PROVEN_WRONG_PAIRS);
        assert!(
            !wrong.names(&format!("{lan}/p2p/{first}")),
            "the oldest went"
        );
    }

    /// Out of reach counts from the last connection, or from the start for a
    /// peer never reached this run — so a node that just started drops nobody.
    #[test]
    fn out_of_reach_counts_from_the_last_connection_or_the_start() {
        let start = Instant::now();
        let mut reached = LastReached::new(start);
        let never = PeerId::random();
        let recent = PeerId::random();
        assert!(!reached.out_of_reach_long(&never, start + UNREACHABLE_FOR / 2));
        assert!(reached.out_of_reach_long(&never, start + UNREACHABLE_FOR));
        reached.reached(recent, start + UNREACHABLE_FOR);
        assert!(!reached.out_of_reach_long(&recent, start + UNREACHABLE_FOR * 3 / 2));
        assert!(reached.out_of_reach_long(&recent, start + UNREACHABLE_FOR * 2));
        reached.prune(start + UNREACHABLE_FOR * 2);
        assert!(
            reached.out_of_reach_long(&recent, start + UNREACHABLE_FOR * 2),
            "a pruned time reads the same"
        );
        assert!(reached.by_peer.is_empty());
    }

    /// Bounded — and a peer whose fresh time was dropped to stay bounded is not
    /// read as out of reach before its hour is up: it counts from the cut.
    #[test]
    fn the_times_are_bounded_without_ageing_a_dropped_peer() {
        let started = Instant::now();
        let start = started + UNREACHABLE_FOR * 2; // a node up for two hours
        let mut reached = LastReached::new(started);
        let first = PeerId::random();
        reached.reached(first, start);
        for n in 1..=LAST_REACHED_PEERS {
            reached.reached(PeerId::random(), start + Duration::from_millis(n as u64));
        }
        assert!(reached.by_peer.len() <= LAST_REACHED_PEERS);
        assert!(
            !reached.by_peer.contains_key(&first),
            "the oldest half went"
        );
        assert!(!reached.out_of_reach_long(&first, start + UNREACHABLE_FOR / 2));
    }

    /// A connection clears what was proven about its peer — it may be back at
    /// the address — and nobody else's.
    #[test]
    fn a_connection_clears_its_peers_pairs_only() {
        let mut wrong = ProvenWrong::default();
        let back = PeerId::random();
        let other = PeerId::random();
        let lan = addr("/ip4/192.168.1.60/udp/8800/quic-v1");
        wrong.note(back, &lan, Instant::now());
        wrong.note(other, &lan, Instant::now());
        wrong.reached(&back);
        assert!(!wrong.names(&format!("{lan}/p2p/{back}")));
        assert!(wrong.names(&format!("{lan}/p2p/{other}")));
    }
}
