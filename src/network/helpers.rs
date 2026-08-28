//! Pure helpers for the network manager — address filtering, event naming,
//! IP extraction. Kept separate from the `NetworkManager` impl so they can be
//! unit-tested in isolation and do not bloat the main event loop file.

use libp2p::swarm::SwarmEvent;
use libp2p::Multiaddr;

use super::behaviour::SwarmBehaviourEvent;

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
