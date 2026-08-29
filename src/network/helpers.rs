//! Pure helpers for the network manager — address filtering, event naming,
//! IP extraction. Kept separate from the `NetworkManager` impl so they can be
//! unit-tested in isolation and do not bloat the main event loop file.

use libp2p::swarm::SwarmEvent;
use libp2p::Multiaddr;

use super::behaviour::SwarmBehaviourEvent;

/// Append `/p2p/<peer_id>` to `addr` unless it already terminates in a peer-id
/// component. A relay-circuit listener ends in `/p2p-circuit`, so it correctly
/// gets the id appended (`.../p2p-circuit/p2p/<peer>` — the canonical relayed
/// dial form). A bare `/ip4/.../tcp/port` address gets the id appended. An
/// address that already names a peer is returned verbatim so we never
/// double-append.
pub(crate) fn ensure_p2p_suffix(addr: Multiaddr, peer_id: libp2p::PeerId) -> Multiaddr {
    if matches!(
        addr.iter().last(),
        Some(libp2p::multiaddr::Protocol::P2p(_))
    ) {
        addr
    } else {
        addr.with(libp2p::multiaddr::Protocol::P2p(peer_id))
    }
}

/// Which peer does this address lead to?
///
/// The LAST `/p2p/` component names the target. A relay-circuit address is
/// `…/p2p/<relay>/p2p-circuit/p2p/<target>`, so reading the FIRST one yields
/// the RELAY — and a dial built from that asks libp2p to reach the relay at an
/// address that resolves to somebody else, while every "do we already know
/// this peer?" check upstream asks about the wrong node.
pub(crate) fn target_peer_from_address(addr: &Multiaddr) -> Option<libp2p::PeerId> {
    addr.iter()
        .filter_map(|proto| match proto {
            libp2p::multiaddr::Protocol::P2p(pid) => Some(pid),
            _ => None,
        })
        .last()
}

/// Terminate a peer's advertised address with `/p2p/<peer_id>` so whoever
/// receives it can tell **whose** address it is.
///
/// An address is the only thing a PEX response carries, so it has to answer
/// that question on its own: the receiver needs it to skip a peer it is already
/// connected to, and to apply the not-a-SwarmLLM-node gate. Returns the address
/// unchanged when the peer id is unknown or the string does not parse — the
/// receiver then falls back to resolving it from its own registry.
pub(crate) fn label_address_with_peer(addr: &str, peer_id: Option<libp2p::PeerId>) -> String {
    let Some(peer_id) = peer_id else {
        return addr.to_string();
    };
    match addr.parse::<Multiaddr>() {
        Ok(parsed) => ensure_p2p_suffix(parsed, peer_id).to_string(),
        Err(_) => addr.to_string(),
    }
}

/// Check if a multiaddr string contains a private/loopback/link-local/CGN IP.
/// Used for PEX filtering to prevent leaking internal topology.
pub(crate) fn is_non_public_addr(addr_str: &str) -> bool {
    if let Ok(addr) = addr_str.parse::<Multiaddr>() {
        addr.iter().any(|proto| match proto {
            libp2p::multiaddr::Protocol::Ip4(ip) => {
                ip.is_private()
                    || ip.is_loopback()
                    || ip.is_link_local()
                    || ip.is_unspecified()
                    // RFC 6598 CGN / Tailscale 100.64.0.0/10
                    || (ip.octets()[0] == 100 && (64..128).contains(&ip.octets()[1]))
                    // link-local metadata
                    || ip == std::net::Ipv4Addr::new(169, 254, 169, 254)
            }
            libp2p::multiaddr::Protocol::Ip6(ip) => {
                ip.is_loopback()
                    || (ip.segments()[0] & 0xffc0) == 0xfe80 // link-local
                    || (ip.segments()[0] & 0xfe00) == 0xfc00 // unique local (fd/fc)
            }
            _ => false,
        })
    } else {
        true // unparseable addresses are not public
    }
}

/// Get a human-readable name for a SwarmEvent (for debug logging).
pub(crate) fn swarm_event_name(event: &SwarmEvent<SwarmBehaviourEvent>) -> &'static str {
    match event {
        SwarmEvent::Behaviour(b) => match b {
            SwarmBehaviourEvent::Gossipsub(_) => "Gossipsub",
            SwarmBehaviourEvent::RequestResponse(_) => "RequestResponse",
            SwarmBehaviourEvent::Kademlia(_) => "Kademlia",
            SwarmBehaviourEvent::Identify(_) => "Identify",
            SwarmBehaviourEvent::AutonatClient(_) => "AutoNATClient",
            SwarmBehaviourEvent::AutonatServer(_) => "AutoNATServer",
            SwarmBehaviourEvent::Dcutr(_) => "DCUtR",
            SwarmBehaviourEvent::Upnp(_) => "UPnP",
            SwarmBehaviourEvent::RelayClient(_) => "RelayClient",
            SwarmBehaviourEvent::RelayServer(_) => "RelayServer",
            SwarmBehaviourEvent::ConnectionLimits(_) => "ConnectionLimits",
            SwarmBehaviourEvent::BlockedPeers(_) => "BlockedPeers",
            SwarmBehaviourEvent::Mdns(_) => "mDNS",
            SwarmBehaviourEvent::PipelineStream(_) => "PipelineStream",
        },
        SwarmEvent::ConnectionEstablished { .. } => "ConnectionEstablished",
        SwarmEvent::ConnectionClosed { .. } => "ConnectionClosed",
        SwarmEvent::IncomingConnection { .. } => "IncomingConnection",
        SwarmEvent::IncomingConnectionError { .. } => "IncomingConnectionError",
        SwarmEvent::OutgoingConnectionError { .. } => "OutgoingConnectionError",
        SwarmEvent::NewListenAddr { .. } => "NewListenAddr",
        SwarmEvent::ExpiredListenAddr { .. } => "ExpiredListenAddr",
        SwarmEvent::ListenerClosed { .. } => "ListenerClosed",
        SwarmEvent::ListenerError { .. } => "ListenerError",
        SwarmEvent::Dialing { .. } => "Dialing",
        SwarmEvent::NewExternalAddrCandidate { .. } => "NewExternalAddrCandidate",
        SwarmEvent::ExternalAddrConfirmed { .. } => "ExternalAddrConfirmed",
        SwarmEvent::ExternalAddrExpired { .. } => "ExternalAddrExpired",
        _ => "Unknown",
    }
}

/// Check whether a 4-byte IPv4 address is non-public (loopback, RFC 1918,
/// link-local, CGN/Tailscale, unspecified). Single source of truth for the
/// "this peer's IP shouldn't count toward subnet anti-gaming or PEX leakage"
/// rule. Mirrors the parse-from-string `is_non_public_addr` for callers that
/// already have raw bytes (Identify handler, anti-gaming subnet tracker).
pub(crate) fn is_non_public_ipv4_bytes(b: &[u8; 4]) -> bool {
    let ip = std::net::Ipv4Addr::new(b[0], b[1], b[2], b[3]);
    ip.is_private()
        || ip.is_loopback()
        || ip.is_link_local()
        || ip.is_unspecified()
        // RFC 6598 CGN / Tailscale 100.64.0.0/10
        || (b[0] == 100 && (64..128).contains(&b[1]))
        // link-local metadata
        || ip == std::net::Ipv4Addr::new(169, 254, 169, 254)
}

/// Extract IPv4 bytes from a multiaddr, if present.
pub(crate) fn extract_ipv4_bytes(addr: &Multiaddr) -> Option<[u8; 4]> {
    for proto in addr.iter() {
        if let libp2p::multiaddr::Protocol::Ip4(ip) = proto {
            return Some(ip.octets());
        }
    }
    None
}

#[cfg(test)]
mod pex_address_tests {
    use super::{label_address_with_peer, target_peer_from_address};
    use libp2p::Multiaddr;

    fn addr(s: &str) -> Multiaddr {
        s.parse().expect("test multiaddr")
    }

    /// A PEX response is nothing but addresses, so an address that names nobody
    /// leaves the receiver unable to tell whether it already holds a connection
    /// to that peer. It then dials unconditionally, every round.
    #[test]
    fn an_advertised_address_is_labelled_with_its_owner() {
        let pid = libp2p::PeerId::random();
        let out = label_address_with_peer("/ip4/203.0.113.5/tcp/8810", Some(pid));
        assert_eq!(out, format!("/ip4/203.0.113.5/tcp/8810/p2p/{pid}"));
        assert_eq!(
            target_peer_from_address(&addr(&out)),
            Some(pid),
            "the receiver must read back exactly the peer the sender labelled"
        );
    }

    /// Never double-append, and never claim ownership we cannot support: an
    /// unknown peer id leaves the address exactly as it was, so the receiver
    /// falls back to resolving it from its own registry.
    #[test]
    fn an_address_is_left_alone_when_it_is_already_labelled_or_the_owner_is_unknown() {
        let pid = libp2p::PeerId::random();
        let labelled = format!("/ip4/203.0.113.5/tcp/8810/p2p/{pid}");
        assert_eq!(label_address_with_peer(&labelled, Some(pid)), labelled);
        assert_eq!(labelled.matches("/p2p/").count(), 1);

        assert_eq!(
            label_address_with_peer("/ip4/203.0.113.5/tcp/8810", None),
            "/ip4/203.0.113.5/tcp/8810"
        );
        assert_eq!(
            label_address_with_peer("not-a-multiaddr", Some(pid)),
            "not-a-multiaddr"
        );
    }

    /// A relay circuit names TWO peers and only the last one is the
    /// destination. Reading the first yields the relay, and a dial built from
    /// that asks libp2p to reach the relay at an address belonging to someone
    /// else — while the already-connected check upstream asks about a node
    /// nobody was talking about.
    #[test]
    fn a_relay_circuit_resolves_to_the_target_not_the_relay() {
        let relay = libp2p::PeerId::random();
        let target = libp2p::PeerId::random();
        let circuit = addr(&format!(
            "/dns4/relay.example/udp/8800/quic-v1/p2p/{relay}/p2p-circuit/p2p/{target}"
        ));

        assert_eq!(target_peer_from_address(&circuit), Some(target));
        assert_ne!(
            target_peer_from_address(&circuit),
            Some(relay),
            "the relay is the path, not the peer"
        );
    }

    /// An address with no peer component at all is the pre-upgrade case and
    /// must report honestly that it names nobody, rather than guessing.
    #[test]
    fn a_bare_address_names_nobody() {
        assert_eq!(
            target_peer_from_address(&addr("/ip4/203.0.113.5/tcp/8810")),
            None
        );
    }
}
