//! Messages a peer sends before this node has handled its Identify.
//!
//! A connection's key is authenticated by the transport handshake, but this
//! node turns a connection into a PEER — its NodeId mapped (`peer_to_node`),
//! its registry entry, its place in `connected_node_ids` — only when its
//! Identify is handled (`identify::handle_identify_received`). Identify is its
//! own substream, with no order against request-response, so a peer that
//! handled OUR Identify first can have its catch-up — its capability and its
//! parts list (#120, #359) — land before its Identify is handled here.
//! Dispatched then, they carried no sender and the dispatcher dropped them
//! ("Dropping unauthenticated NodeCapabilityUpdate / ShardAnnounce"): in a
//! real-link rig a peer's pair was dropped 345 ms before its Identify
//! (2026-10-29), leaving the new node without that peer's parts until its next
//! gossip repeat (5-40 min, `QuietRepeat`). On the live node after a restart, a
//! v0.3.268 computer stayed connected for good: the capability that names its
//! release was one of them, and nothing else it sends names it (2026-11-06).
//!
//! So the catch-up kinds are HELD — bounded per peer, per network and in bytes,
//! and briefly — and handed on in
//! the order they came once Identify has made the connection a peer — the order
//! every handler assumes. A peer refused at Identify (foreign, a forged key,
//! cut off) or gone has its messages dropped, as they all were before. Any
//! other kind, or past the bounds, is dispatched as before (no sender).
//!
//! Prior art: Bitcoin Core treats a message before `VERACK` as a violation —
//! sound only because its handshake and its messages share ONE ordered stream.
//! go-libp2p starts protocols before Identify finishes to save a round trip
//! (go-libp2p #2551), so the order is the application's to restore. Holding
//! what came early is TCP's out-of-order queue in miniature: bounded, released
//! in order, dropped with the connection.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use libp2p::PeerId;

use crate::network::helpers::NetGroup;
use crate::types::SwarmMessage;

/// Messages held for one peer at most. Its catch-up is a capability and a
/// parts list, the list in pieces past `MAX_SHARDS_PER_ANNOUNCE` parts.
pub(super) const HOLD_PER_PEER: usize = 8;
/// Bytes held for one peer at most, counted as the messages' JSON. A real
/// catch-up is tens of KB (a 512-part list ~50 KB); a list's entry COUNT is
/// bounded at the door but not its strings, so one message can be the whole
/// 4 MB frame (security review, 2026-11-06).
pub(super) const HOLD_BYTES_PER_PEER: usize = 1024 * 1024;
/// Bytes held in all at most.
pub(super) const HOLD_BYTES_IN_ALL: usize = 8 * 1024 * 1024;
/// Peers held for at most.
pub(super) const HOLD_PEERS: usize = 32;
/// Peers held for from one network at most — keys are free, networks are not
/// (`NetGroup`, the share every multipliable bound here is counted by, #299):
/// 32 fresh keys from one host would otherwise fill the table and turn the
/// hold off for everyone else.
pub(super) const HOLD_PEERS_PER_NETWORK: usize = 4;
/// How long a peer's first held message waits for its Identify. A handled
/// Identify follows within a second or two even on a busy start (345 ms in the
/// rig log above, 800 ms on the live node's restart).
pub(super) const HOLD_FOR: Duration = Duration::from_secs(10);

struct Held {
    first_at: Instant,
    network: Option<NetGroup>,
    bytes: usize,
    messages: Vec<SwarmMessage>,
}

/// What each peer sent before its Identify was handled. Written and read only
/// by the network manager: `dispatch_authenticated` holds, the Identify handler
/// releases (or forgets, for a peer it refuses), the last close forgets, the
/// liveness tick expires.
#[derive(Default)]
pub(super) struct HeldUntilIdentified {
    by_peer: HashMap<PeerId, Held>,
    bytes: usize,
}

impl HeldUntilIdentified {
    /// Hold `msg`, sent by `peer` before its Identify was handled — `network`
    /// being where its connection was counted from (`PeerRemote::group`) — or
    /// hand it back (`Some`) when it is not a catch-up or a bound is reached;
    /// `None` is held.
    #[must_use = "a message handed back was not held and must still be dispatched"]
    pub(super) fn hold(
        &mut self,
        peer: PeerId,
        network: Option<NetGroup>,
        msg: SwarmMessage,
        now: Instant,
    ) -> Option<SwarmMessage> {
        if !is_a_catch_up(&msg) {
            return Some(msg);
        }
        let size = json_size(&msg);
        if !self.by_peer.contains_key(&peer) {
            if !self.room_for_a_peer_from(network) {
                self.expire(now);
                if !self.room_for_a_peer_from(network) {
                    return Some(msg);
                }
            }
            self.by_peer.insert(
                peer,
                Held {
                    first_at: now,
                    network,
                    bytes: 0,
                    messages: Vec::new(),
                },
            );
        }
        let in_all = self.bytes;
        let held = self.by_peer.get_mut(&peer).expect("inserted above");
        if held.messages.len() >= HOLD_PER_PEER
            || held.bytes.saturating_add(size) > HOLD_BYTES_PER_PEER
            || in_all.saturating_add(size) > HOLD_BYTES_IN_ALL
        {
            if held.messages.is_empty() {
                self.by_peer.remove(&peer);
            }
            return Some(msg);
        }
        held.bytes += size;
        held.messages.push(msg);
        self.bytes += size;
        None
    }

    fn room_for_a_peer_from(&self, network: Option<NetGroup>) -> bool {
        self.by_peer.len() < HOLD_PEERS
            && self
                .by_peer
                .values()
                .filter(|held| held.network == network)
                .count()
                < HOLD_PEERS_PER_NETWORK
    }

    /// Everything held for `peer`, in the order it came, and forget it.
    pub(super) fn release(&mut self, peer: &PeerId) -> Vec<SwarmMessage> {
        match self.by_peer.remove(peer) {
            Some(held) => {
                self.bytes -= held.bytes;
                held.messages
            }
            None => Vec::new(),
        }
    }

    /// Drop what is held for `peer`; how many messages went.
    pub(super) fn forget(&mut self, peer: &PeerId) -> usize {
        self.release(peer).len()
    }

    /// Drop every peer's messages that have waited past [`HOLD_FOR`]; how many
    /// messages went.
    pub(super) fn expire(&mut self, now: Instant) -> usize {
        let mut dropped = 0;
        let mut bytes = 0;
        self.by_peer.retain(|_, held| {
            let keep = now.saturating_duration_since(held.first_at) < HOLD_FOR;
            if !keep {
                dropped += held.messages.len();
                bytes += held.bytes;
            }
            keep
        });
        self.bytes -= bytes;
        dropped
    }

    #[cfg(test)]
    fn peers_held(&self) -> usize {
        self.by_peer.len()
    }
}

/// The kinds a peer sends as it identifies this node — the catch-up in
/// `identify::handle_identify_received` — within the bounds the dispatcher
/// takes them at, so nothing larger than it would accept is held.
fn is_a_catch_up(msg: &SwarmMessage) -> bool {
    match msg {
        SwarmMessage::NodeCapabilityUpdate(cap) => crate::daemon::dispatch::capability_fits(cap),
        SwarmMessage::ShardAnnounce(announce) => announce.fits_one_message(),
        // A name the dispatcher would take is at most 32 characters (#417).
        SwarmMessage::NicknameGossip(gossip) => {
            crate::identity::nickname::validate_nickname(&gossip.record.nickname).is_ok()
        }
        _ => false,
    }
}

/// The message's size as JSON, counted without building it.
fn json_size(msg: &SwarmMessage) -> usize {
    struct Count(usize);
    impl std::io::Write for Count {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            self.0 += bytes.len();
            Ok(bytes.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    let mut count = Count(0);
    match serde_json::to_writer(&mut count, msg) {
        Ok(()) => count.0,
        // Unmeasurable is never small.
        Err(_) => usize::MAX / 2,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Built through the one constructor (gotcha #406); the timestamp carries a
    /// tag so a test can tell the messages apart.
    fn announce_of(shards: Vec<crate::types::ShardId>, tag: u64) -> SwarmMessage {
        let mut announce = crate::model::manifest::shard_announce(
            &crate::model::registry::ModelRegistry::new(),
            crate::types::NodeId([1u8; 32]),
            shards,
            vec![],
        );
        announce.timestamp = chrono::DateTime::from_timestamp(tag as i64, 0).unwrap();
        SwarmMessage::ShardAnnounce(announce)
    }

    fn announce(tag: u64) -> SwarmMessage {
        announce_of(vec![], tag)
    }

    fn tag_of(msg: &SwarmMessage) -> u64 {
        match msg {
            SwarmMessage::ShardAnnounce(a) => a.timestamp.timestamp() as u64,
            _ => panic!("not an announce"),
        }
    }

    /// **What came early is handed on in the order it came**, once, and only
    /// for the peer it came from.
    #[test]
    fn held_messages_are_released_in_order_once() {
        let mut held = HeldUntilIdentified::default();
        let (peer, other) = (PeerId::random(), PeerId::random());
        let now = Instant::now();
        for tag in 1..=3 {
            assert!(held.hold(peer, None, announce(tag), now).is_none());
        }
        assert!(held.hold(other, None, announce(9), now).is_none());
        let out: Vec<u64> = held.release(&peer).iter().map(tag_of).collect();
        assert_eq!(out, vec![1, 2, 3]);
        assert!(held.release(&peer).is_empty(), "released once");
        assert_eq!(held.peers_held(), 1, "the other peer's still waits");
    }

    /// **Only the catch-up is held**: any other kind goes on as before.
    #[test]
    fn only_a_catch_up_is_held() {
        let mut held = HeldUntilIdentified::default();
        let ping = SwarmMessage::HealthPing {
            nonce: 1,
            timestamp: 2,
            node_id: None,
            active_request_count: 0,
        };
        assert!(held
            .hold(PeerId::random(), None, ping, Instant::now())
            .is_some());
        assert_eq!(held.peers_held(), 0);
    }

    /// **Nothing larger than the dispatcher takes is held**: a parts list past
    /// the per-message bound is handed back.
    #[test]
    fn a_catch_up_past_the_dispatchers_bounds_is_not_held() {
        let mut held = HeldUntilIdentified::default();
        let shards = (0..=crate::types::MAX_SHARDS_PER_ANNOUNCE as u32)
            .map(|index| crate::types::ShardId {
                model_id: crate::types::ModelId("m".into()),
                index,
            })
            .collect();
        let too_long = announce_of(shards, 0);
        assert!(held
            .hold(PeerId::random(), None, too_long, Instant::now())
            .is_some());
    }

    /// #417: a peer's name, sent with its catch-up, waits for its Identify like
    /// the rest of it — and a name the dispatcher would refuse does not.
    #[test]
    fn a_peers_name_is_held_and_one_the_dispatcher_refuses_is_not() {
        let mut held = HeldUntilIdentified::default();
        let name = |nickname: &str| {
            SwarmMessage::NicknameGossip(crate::types::NicknameGossip {
                record: crate::types::NicknameRecord {
                    node_id: crate::types::NodeId([1u8; 32]),
                    nickname: nickname.into(),
                    timestamp: chrono::Utc::now(),
                    signature: vec![0u8; 64],
                },
            })
        };
        let peer = PeerId::random();
        assert!(held
            .hold(peer, None, name("kitchen-pc"), Instant::now())
            .is_none());
        assert_eq!(held.release(&peer).len(), 1);
        assert!(held
            .hold(
                PeerId::random(),
                None,
                name(&"x".repeat(33)),
                Instant::now()
            )
            .is_some());
    }

    /// **The hold is bounded per peer and in peers**, and a full table makes
    /// room only from peers that waited past `HOLD_FOR`.
    #[test]
    fn the_hold_is_bounded_and_expires() {
        let mut held = HeldUntilIdentified::default();
        let start = Instant::now();
        let peer = PeerId::random();
        for tag in 0..HOLD_PER_PEER as u64 {
            assert!(held.hold(peer, network(0), announce(tag), start).is_none());
        }
        assert!(
            held.hold(peer, network(0), announce(99), start).is_some(),
            "per peer"
        );
        for n in 1..HOLD_PEERS as u8 {
            assert!(held
                .hold(PeerId::random(), network(n), announce(0), start)
                .is_none());
        }
        assert!(
            held.hold(PeerId::random(), network(200), announce(0), start)
                .is_some(),
            "in peers"
        );
        let later = start + HOLD_FOR;
        assert!(
            held.hold(PeerId::random(), network(200), announce(0), later)
                .is_none(),
            "room made from the peers that waited too long"
        );
        assert_eq!(held.peers_held(), 1);
        assert_eq!(held.expire(later + HOLD_FOR), 1);
        assert_eq!(held.forget(&peer), 0, "expired already");
        assert_eq!(held.bytes, 0, "every byte given back");
    }

    /// A public IPv4 /16 of its own.
    fn network(n: u8) -> Option<NetGroup> {
        Some(NetGroup::Direct(
            crate::network::helpers::Network::Ipv4Slash16([10, n]),
        ))
    }

    /// **Keys are free, networks are not**: one network holds at most
    /// `HOLD_PEERS_PER_NETWORK` peers' messages, so fresh keys from one host
    /// cannot fill the table for everyone else (security review, 2026-11-06).
    #[test]
    fn one_network_holds_only_its_share_of_peers() {
        let mut held = HeldUntilIdentified::default();
        let now = Instant::now();
        for _ in 0..HOLD_PEERS_PER_NETWORK {
            assert!(held
                .hold(PeerId::random(), network(1), announce(0), now)
                .is_none());
        }
        assert!(
            held.hold(PeerId::random(), network(1), announce(0), now)
                .is_some(),
            "its network's share is full"
        );
        assert!(
            held.hold(PeerId::random(), network(2), announce(0), now)
                .is_none(),
            "another network still has room"
        );
    }

    /// **The hold is bounded in BYTES, not only in entries**: a list's
    /// entry count is checked at the door but not its strings, so one message
    /// can be the whole 4 MB frame (security review, 2026-11-06).
    #[test]
    fn the_hold_is_bounded_in_bytes() {
        let mut held = HeldUntilIdentified::default();
        let now = Instant::now();
        let heavy = |len: usize| {
            announce_of(
                vec![crate::types::ShardId {
                    model_id: crate::types::ModelId("x".repeat(len)),
                    index: 0,
                }],
                0,
            )
        };
        let peer = PeerId::random();
        assert!(
            held.hold(peer, network(1), heavy(HOLD_BYTES_PER_PEER), now)
                .is_some(),
            "one message past a peer's bytes"
        );
        assert_eq!(
            held.peers_held(),
            0,
            "a refused first message holds no place"
        );
        // Many peers, each within its own bytes, stop at the total.
        let mut taken = 0;
        for n in 0..HOLD_PEERS as u8 {
            if held
                .hold(
                    PeerId::random(),
                    network(n),
                    heavy(HOLD_BYTES_PER_PEER / 2),
                    now,
                )
                .is_none()
            {
                taken += 1;
            }
        }
        assert!(held.bytes <= HOLD_BYTES_IN_ALL);
        assert!(
            taken < HOLD_PEERS,
            "the total bound refused some: took {taken}"
        );
    }
}
