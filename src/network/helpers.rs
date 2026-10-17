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
            libp2p::multiaddr::Protocol::Ip4(ip) => is_non_public_ipv4_bytes(&ip.octets()),
            libp2p::multiaddr::Protocol::Ip6(ip) => is_non_public_ipv6(&ip),
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

/// The IPv6 half of [`is_non_public_ipv4_bytes`]: loopback, link-local
/// (`fe80::/10`) and unique-local (`fc00::/7`).
pub(crate) fn is_non_public_ipv6(ip: &std::net::Ipv6Addr) -> bool {
    ip.is_loopback()
        || (ip.segments()[0] & 0xffc0) == 0xfe80 // link-local
        || (ip.segments()[0] & 0xfe00) == 0xfc00 // unique local (fd/fc)
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

/// Which network an address reaches this node from — see [`NetGroup`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Network {
    /// A public IPv4 address's /16.
    Ipv4Slash16([u8; 2]),
    /// A public IPv6 address's /32.
    Ipv6Slash32([u8; 4]),
    /// A private, loopback, link-local or CGN address: the host itself.
    Host(std::net::IpAddr),
    /// A host name this node's own configuration dials (a bootstrap address),
    /// by the first 8 bytes of a BLAKE3 hash of it.
    Configured([u8; 8]),
    /// No IP and no name of ours: ONE network for every such address.
    Unaddressed,
}

impl Network {
    /// The network of a DIRECT connection's address: its first IP, or its
    /// first host name if that is one of `names`.
    pub(crate) fn of_address(addr: &Multiaddr, names: &ConfiguredNames) -> Self {
        use libp2p::multiaddr::Protocol;
        for proto in addr.iter() {
            match &proto {
                Protocol::Ip4(ip) => return Self::of_ipv4(*ip),
                Protocol::Ip6(ip) => return Self::of_ipv6(*ip),
                Protocol::P2pCircuit => break,
                other => {
                    if let Some(name) = dns_name(other) {
                        return names.network_of(name);
                    }
                }
            }
        }
        Self::Unaddressed
    }

    fn of_ipv4(ip: std::net::Ipv4Addr) -> Self {
        let o = ip.octets();
        if is_non_public_ipv4_bytes(&o) {
            Self::Host(ip.into())
        } else {
            Self::Ipv4Slash16([o[0], o[1]])
        }
    }

    fn of_ipv6(ip: std::net::Ipv6Addr) -> Self {
        match ip.to_ipv4_mapped() {
            Some(v4) => Self::of_ipv4(v4),
            None if is_non_public_ipv6(&ip) => Self::Host(ip.into()),
            None => {
                let o = ip.octets();
                Self::Ipv6Slash32([o[0], o[1], o[2], o[3]])
            }
        }
    }

    /// A tag, then the prefix.
    fn key_bytes(&self) -> [u8; 17] {
        let mut out = [0u8; 17];
        match self {
            Self::Ipv4Slash16(p) => {
                out[0] = 1;
                out[1..3].copy_from_slice(p);
            }
            Self::Ipv6Slash32(p) => {
                out[0] = 2;
                out[1..5].copy_from_slice(p);
            }
            Self::Host(std::net::IpAddr::V4(ip)) => {
                out[0] = 3;
                out[1..5].copy_from_slice(&ip.octets());
            }
            Self::Host(std::net::IpAddr::V6(ip)) => {
                out[0] = 4;
                out[1..17].copy_from_slice(&ip.octets());
            }
            Self::Configured(h) => {
                out[0] = 5;
                out[1..9].copy_from_slice(h);
            }
            Self::Unaddressed => {}
        }
        out
    }
}

/// The host names this node's own configuration dials — its bootstrap
/// addresses — which are the only names that count as a network of their own.
/// libp2p records a dial by the address it was given, unresolved, so the
/// default anchor (`/dns4/swarmllm.duckdns.org/…`) has no IP to count by.
#[derive(Default)]
pub(crate) struct ConfiguredNames(Vec<String>);

impl ConfiguredNames {
    pub(crate) fn of_bootstrap(addresses: &[String]) -> Self {
        let mut names = Vec::new();
        for addr in addresses.iter().filter_map(|a| a.parse::<Multiaddr>().ok()) {
            for proto in addr.iter() {
                if let Some(name) = dns_name(&proto) {
                    let name = name.to_ascii_lowercase();
                    if !names.contains(&name) {
                        names.push(name);
                    }
                }
            }
        }
        Self(names)
    }

    fn network_of(&self, name: &str) -> Network {
        let name = name.to_ascii_lowercase();
        if self.0.contains(&name) {
            let hash = blake3::hash(name.as_bytes());
            Network::Configured(hash.as_bytes()[..8].try_into().expect("8 bytes"))
        } else {
            Network::Unaddressed
        }
    }
}

/// The relay a circuit address runs through: the peer named just before
/// `/p2p-circuit`.
fn circuit_relay(addr: &Multiaddr) -> Option<libp2p::PeerId> {
    use libp2p::multiaddr::Protocol;
    let mut relay = None;
    for proto in addr.iter() {
        match proto {
            Protocol::P2p(id) => relay = Some(id),
            Protocol::P2pCircuit => return relay,
            _ => {}
        }
    }
    None
}

fn dns_name<'a>(proto: &'a libp2p::multiaddr::Protocol<'a>) -> Option<&'a str> {
    use libp2p::multiaddr::Protocol;
    match proto {
        Protocol::Dns(n) | Protocol::Dns4(n) | Protocol::Dns6(n) | Protocol::Dnsaddr(n) => Some(n),
        _ => None,
    }
}

/// Which network a connection reaches this node from — the unit a resource
/// shared out per sender is counted by (FUTURE_WORK #299: the relayed-body
/// replay memory). A key pair costs nothing to mint and an address in another
/// network does, so a share per PeerId is a share per wish.
///
/// Bitcoin Core's rule (`NetGroupManager::GetGroup`): a public IPv4 address
/// counts as its /16, a public IPv6 address as its /32. Departures:
/// - a private, loopback, link-local or CGN address counts as its HOST, where
///   Bitcoin pools every unroutable address into one group: a peer can show
///   such a source only from inside that network, and pooling would put every
///   machine on a LAN behind one share. Anyone inside it can still add hosts
///   (a tailnet, a LAN) — an accepted risk, the network being the owner's;
/// - a host name counts as a network only when this node's own configuration
///   names it ([`ConfiguredNames`]); any other name is the peer's to choose,
///   so it is [`Network::Unaddressed`], one network for all of them;
/// - a relay circuit is [`NetGroup::Circuit`] of its relay's network — never
///   the relay's own share, which an attacker reaching this node through that
///   relay's circuits could otherwise fill (security review of #299: the
///   anchor relays for every NAT'd node) — and that network is the one the
///   connection TO the relay was counted in, never an address written in the
///   circuit's (see [`Self::of_connection`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum NetGroup {
    /// A direct connection from this network.
    Direct(Network),
    /// A circuit through a relay whose own connection is in this network.
    Circuit(Network),
}

impl NetGroup {
    /// Where a peer with no connection on record is counted.
    pub(crate) const UNADDRESSED: Self = Self::Direct(Network::Unaddressed);

    /// The network of one connection, from the swarm's own record of it.
    ///
    /// A circuit's is `Circuit` of `relay_network(relay)`, the network this
    /// node's own connection to the relay was counted in — never an IP or a
    /// name written in the circuit's address. libp2p-relay 0.21 opens a circuit
    /// over an existing connection to the relay and never dials the address it
    /// was handed (`priv_client.rs`, `DialReq` and the reservation alike), so
    /// that part says whatever its advertiser wrote (second security-review
    /// pass on #299: one relay plus a PEX reply of fake prefixes minted a group
    /// per prefix). The relay's peer id IS authenticated — the circuit runs
    /// over the connection to it. An inbound circuit is found by the address it
    /// arrived on, since libp2p-relay's listener reports the remote as a bare
    /// `/p2p/<source>` (gotcha #179).
    pub(crate) fn of_connection(
        endpoint: &libp2p::core::ConnectedPoint,
        names: &ConfiguredNames,
        relay_network: impl Fn(&libp2p::PeerId) -> Option<Network>,
    ) -> Self {
        use crate::network::relay::is_relay_circuit_addr;
        let (circuit, remote) = match endpoint {
            libp2p::core::ConnectedPoint::Dialer { address, .. } => {
                (is_relay_circuit_addr(address).then_some(address), address)
            }
            libp2p::core::ConnectedPoint::Listener {
                local_addr,
                send_back_addr,
            } => (
                is_relay_circuit_addr(local_addr).then_some(local_addr),
                send_back_addr,
            ),
        };
        match circuit {
            Some(addr) => Self::Circuit(
                circuit_relay(addr)
                    .and_then(|relay| relay_network(&relay))
                    .unwrap_or(Network::Unaddressed),
            ),
            None => Self::Direct(Network::of_address(remote, names)),
        }
    }

    /// The bytes that name this group: direct or circuit, then the network.
    pub(crate) fn key_bytes(&self) -> [u8; 18] {
        let (tag, network) = match self {
            Self::Direct(n) => (0, n),
            Self::Circuit(n) => (1, n),
        };
        let mut out = [0u8; 18];
        out[0] = tag;
        out[1..].copy_from_slice(&network.key_bytes());
        out
    }
}

#[cfg(test)]
mod netgroup_tests {
    use super::{ConfiguredNames, NetGroup, Network};
    use libp2p::core::{ConnectedPoint, Endpoint};
    use libp2p::{Multiaddr, PeerId};

    fn net(s: &str) -> Network {
        Network::of_address(
            &s.parse::<Multiaddr>().expect("test multiaddr"),
            &ConfiguredNames::default(),
        )
    }

    fn dialed(address: &str) -> ConnectedPoint {
        ConnectedPoint::Dialer {
            address: address.parse().unwrap(),
            role_override: Endpoint::Dialer,
            port_use: libp2p::core::transport::PortUse::Reuse,
        }
    }

    fn default_names() -> ConfiguredNames {
        ConfiguredNames::of_bootstrap(&crate::config::network::default_bootstrap_peers())
    }

    /// FUTURE_WORK #299: what it costs to be a second group. A public address
    /// is its /16 (IPv4) or /32 (IPv6), so another port, another host in the
    /// same block or an IPv4-mapped spelling is the same group; a private
    /// address is its own host; and nothing a peer names freely — a DNS name,
    /// a circuit relay with no connection here — buys a group of its own.
    #[test]
    fn a_group_is_a_network_not_a_name() {
        let a = net("/ip4/203.0.113.5/tcp/8810");
        assert_eq!(a, Network::Ipv4Slash16([203, 0]));
        assert_eq!(a, net("/ip4/203.0.200.9/udp/8800/quic-v1"));
        assert_eq!(a, net("/ip6/::ffff:203.0.7.7/tcp/1"));
        assert_ne!(a, net("/ip4/203.1.113.5/tcp/8810"));

        assert_eq!(
            net("/ip6/2001:db8:1:2::5/tcp/1"),
            net("/ip6/2001:db8:ffff::9/tcp/2")
        );
        assert_ne!(net("/ip6/2001:db8::1/tcp/1"), net("/ip6/2001:db9::1/tcp/1"));

        for (one, other) in [
            ("/ip4/192.168.1.10/tcp/1", "/ip4/192.168.1.11/tcp/1"),
            ("/ip4/100.64.0.1/tcp/1", "/ip4/100.64.0.2/tcp/1"),
            ("/ip6/fd00::1/tcp/1", "/ip6/fd00::2/tcp/1"),
        ] {
            assert!(matches!(net(one), Network::Host(_)), "{one}");
            assert_ne!(net(one), net(other), "two LAN machines are two groups");
        }

        assert_eq!(net("/dns4/a.example/tcp/1"), Network::Unaddressed);
        assert_eq!(net("/dns4/b.example/tcp/1"), Network::Unaddressed);
        let unknown_relay = format!(
            "/p2p/{}/p2p-circuit/p2p/{}",
            PeerId::random(),
            PeerId::random()
        );
        assert_eq!(
            NetGroup::of_connection(&dialed(&unknown_relay), &default_names(), |_| None),
            NetGroup::Circuit(Network::Unaddressed)
        );

        let groups = [
            NetGroup::Direct(a),
            NetGroup::Direct(net("/ip6/2001:db8::1/tcp/1")),
            NetGroup::Direct(net("/ip4/192.168.1.10/tcp/1")),
            NetGroup::Direct(net("/ip6/fd00::1/tcp/1")),
            NetGroup::UNADDRESSED,
            NetGroup::Circuit(Network::Unaddressed),
            NetGroup::Circuit(a),
            NetGroup::Direct(Network::Configured([7; 8])),
        ];
        for (i, x) in groups.iter().enumerate() {
            for y in &groups[i + 1..] {
                assert_ne!(x.key_bytes(), y.key_bytes(), "{x:?} vs {y:?}");
            }
        }
    }

    /// Security review of #299: libp2p records a dial by the name it was
    /// given, so the default anchor has no IP. Its name — ours, from the
    /// configuration — is a network of its own; any other name is the peer's
    /// choice and counts with every unaddressed one, never with the anchor.
    #[test]
    fn only_a_name_this_node_configured_is_a_network_of_its_own() {
        let names = default_names();
        let of = |s: &str| Network::of_address(&s.parse().unwrap(), &names);
        let anchor = of("/dns4/swarmllm.duckdns.org/tcp/8810");
        assert!(matches!(anchor, Network::Configured(_)));
        assert_eq!(anchor, of("/dns4/SwarmLLM.duckdns.org/udp/8800/quic-v1"));
        assert_eq!(of("/dns4/elsewhere.example/tcp/8810"), Network::Unaddressed);
        assert_eq!(of("/dns4/a.example/tcp/1"), of("/dns6/b.example/tcp/1"));
    }

    /// A circuit is counted by the network of this node's own connection to
    /// its relay — whichever side the swarm saw it from (an inbound one's
    /// remote is a bare `/p2p/<source>`, so its relay is read from the address
    /// it arrived on) — never as the relay's own direct share, and never by the
    /// IP or name its address carries, which nobody dialed (second
    /// security-review pass on #299).
    #[test]
    fn a_circuit_is_counted_by_its_relays_own_connection() {
        let names = default_names();
        let relay = PeerId::random();
        let relays_network = Network::Ipv4Slash16([198, 51]);
        let relay_network = |p: &PeerId| (*p == relay).then_some(relays_network);
        let through_relay = NetGroup::Circuit(relays_network);

        for written in [
            "/ip4/198.51.100.7",
            "/ip4/1.0.0.1",
            "/dns4/swarmllm.duckdns.org",
        ] {
            let circuit = format!("{written}/tcp/8810/p2p/{relay}/p2p-circuit");
            assert_eq!(
                NetGroup::of_connection(
                    &dialed(&format!("{circuit}/p2p/{}", PeerId::random())),
                    &names,
                    relay_network
                ),
                through_relay,
                "{written}: a dialed circuit names no network of its own"
            );
            for _ in 0..2 {
                let inbound = ConnectedPoint::Listener {
                    local_addr: circuit.parse().unwrap(),
                    send_back_addr: format!("/p2p/{}", PeerId::random()).parse().unwrap(),
                };
                assert_eq!(
                    NetGroup::of_connection(&inbound, &names, relay_network),
                    through_relay,
                    "{written}: one group per relay, never one per source"
                );
            }
        }
        assert_ne!(through_relay, NetGroup::Direct(relays_network));

        let direct = ConnectedPoint::Listener {
            local_addr: "/ip4/0.0.0.0/tcp/8810".parse().unwrap(),
            send_back_addr: "/ip4/203.0.113.5/tcp/40000".parse().unwrap(),
        };
        assert_eq!(
            NetGroup::of_connection(&direct, &names, relay_network),
            NetGroup::Direct(Network::Ipv4Slash16([203, 0]))
        );
    }
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
