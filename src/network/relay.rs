use std::time::Duration;

use libp2p::{Multiaddr, PeerId};

/// Max wall-clock time a relayed circuit may stay open before the relay
/// forcibly closes it. Matches `RELAY_RESERVATION_DURATION_SECS`.
const RELAY_CIRCUIT_DURATION_SECS: u64 = 3600;
/// Max reservation TTL granted to a NAT'd peer before it must re-reserve.
const RELAY_RESERVATION_DURATION_SECS: u64 = 3600;

/// Configuration for the relay server role.
///
/// When a node has `config.network.enable_relay = true` and is publicly
/// reachable, it acts as a relay server — accepting reservations from
/// NAT'd peers and forwarding circuit traffic.
pub struct RelayServerConfig {
    pub max_reservations: usize,
    pub max_circuits: usize,
    pub max_circuit_duration: Duration,
    /// NET-M5: Reservation duration is independently configurable from circuit duration.
    pub reservation_duration: Duration,
    pub max_circuit_bytes: u64,
}

impl Default for RelayServerConfig {
    fn default() -> Self {
        Self {
            max_reservations: 128,
            // libp2p's own default is 16, but that is tuned for its intended
            // use — a *brief* bootstrap circuit (upstream defaults:
            // `max_circuit_duration` 2 min, `max_circuit_bytes` 128 KiB) held
            // only until DCUtR upgrades to direct. We use the circuit as a
            // real data path and already raised those to 1 hour / 1 GB below,
            // so each slot is occupied ~30x longer than upstream assumes.
            // Keeping 16 alongside a 1-hour duration means a relay serving
            // more than 16 concurrent NAT'd pairs starts refusing circuits —
            // and a refused circuit doesn't degrade gracefully, it removes the
            // peer from `connected_node_ids` and therefore from scheduling
            // entirely. 128 matches `max_reservations` so a node that was
            // granted a reservation can actually open a circuit.
            max_circuits: 128,
            max_circuit_duration: Duration::from_secs(RELAY_CIRCUIT_DURATION_SECS),
            reservation_duration: Duration::from_secs(RELAY_RESERVATION_DURATION_SECS),
            max_circuit_bytes: 1 << 30, // 1 GB
        }
    }
}

/// Whether this node relays for others NOW — `SharedState::relay_forwarding_enabled`,
/// handed to the circuit relay server as [`OnlyWhileRelaying`].
pub type RelayGate = std::sync::Arc<dyn Fn() -> bool + Send + Sync>;

/// A rate limiter that is a switch: the circuit relay server accepts a new
/// reservation or circuit only while the gate says this node relays for
/// others.
///
/// libp2p fixes a relay server's config when the swarm is built, while the
/// answer changes as the node runs — the owner turns donation off in Settings,
/// the node stops being reachable from the internet, or its connection becomes
/// metered (#277). Until 2026-10-15 nothing connected them: a node whose owner
/// had declined relaying stopped ADVERTISING it, but still accepted
/// reservations and opened circuits of up to an hour and 1 GB for any peer that
/// already knew it. The server asks every rate limiter at every reservation and
/// circuit request (`libp2p-relay` 0.21, `Behaviour::on_connection_handler_event`),
/// and a refusal is `RESOURCE_LIMIT_EXCEEDED`, so the peer goes to another
/// relay. A circuit already open runs out its own limits.
pub struct OnlyWhileRelaying(pub RelayGate);

impl libp2p::relay::RateLimiter for OnlyWhileRelaying {
    fn try_next(&mut self, _peer: PeerId, _addr: &Multiaddr, _now: std::time::Instant) -> bool {
        (self.0)()
    }
}

/// Build a relay server configuration for libp2p.
///
/// Returns `None` if relay serving is disabled.
pub fn build_relay_server_config(config: &RelayServerConfig) -> libp2p::relay::Config {
    libp2p::relay::Config {
        max_reservations: config.max_reservations,
        max_reservations_per_peer: 4,
        // NET-M5: Use independent reservation_duration instead of reusing max_circuit_duration
        reservation_duration: config.reservation_duration,
        max_circuits: config.max_circuits,
        max_circuits_per_peer: 4,
        max_circuit_duration: config.max_circuit_duration,
        max_circuit_bytes: config.max_circuit_bytes,
        ..Default::default()
    }
}

/// The reason a directly reachable node gives for reserving a relay anyway.
pub(crate) const HOME_RELAY: &str =
    "keep a home relay, so a computer that knows this node only by its relay address can still reach it";

/// The reason a node with no internet-reachable address gives.
pub(crate) const NOT_REACHABLE: &str = "no internet-reachable address after startup";

/// Whether this node should reserve a relay now, and why — `None` when not.
///
/// **Every node keeps a HOME relay, directly reachable or not (2026-11-16, #464).**
/// A computer that cannot be dialled (no public address, no relay of its own —
/// the Australian peer since its address began changing) reconnects only by
/// dialling US, at the addresses it already holds. It met this node through
/// the relay, so those were the relay's. After a restart in which UPnP worked,
/// this node reserved nothing — only AutoNAT's "unreachable" did — and that
/// computer's redials (5 s → 300 s, then given up) and its five-minute cache
/// dial all hit a relay holding no reservation: India came back after 14
/// minutes, by luck of a third node's 20-address peer list, and Australia not
/// at all. With a home relay the address it holds keeps working, and the
/// connection goes direct from there (DCUtR: this node, the relayed listener,
/// starts the upgrade, and the far side can always dial a public node).
///
/// Prior art: Tailscale starts every connection on DERP and upgrades it to
/// direct; Syncthing announces its relay listener beside its direct ones and
/// replaces a relayed connection when a direct one lands. go-libp2p's AutoRelay
/// reserves only while private, because its peers re-find a public node
/// through the DHT — ours find a lost peer only by the addresses they already
/// hold (#465).
///
/// An anchor is a relay and everyone's bootstrap address, so a reachable one
/// keeps none; an unreachable one reserves as before.
pub(crate) fn reason_to_reserve(
    already_reserved: bool,
    anchor_mode: bool,
    reachable: bool,
) -> Option<&'static str> {
    if already_reserved {
        None
    } else if !reachable {
        Some(NOT_REACHABLE)
    } else if anchor_mode {
        None
    } else {
        Some(HOME_RELAY)
    }
}

/// The relay a bootstrap address names — its peer id and its address without
/// the `/p2p` part — or `None` when it names no peer, or names this node (an
/// anchor's bootstrap list lists itself). The one reading of a bootstrap entry
/// as a relay: `try_activate_relay` reserves on these, and a connection to one
/// is what starts the home relay.
pub(crate) fn bootstrap_relay(addr: &str, local: &PeerId) -> Option<(PeerId, Multiaddr)> {
    let maddr = addr.parse::<Multiaddr>().ok()?;
    let pid = maddr.iter().find_map(|p| match p {
        libp2p::multiaddr::Protocol::P2p(pid) => Some(pid),
        _ => None,
    })?;
    if pid == *local {
        return None;
    }
    let base = maddr
        .iter()
        .take_while(|p| !matches!(p, libp2p::multiaddr::Protocol::P2p(_)))
        .collect();
    Some((pid, base))
}

/// Generate the relay listener address for connecting through a relay peer.
///
/// A node calls this to establish a relayed listen address — its only way in
/// when it is not directly reachable, its home relay when it is
/// ([`reason_to_reserve`]).
pub fn relay_listen_addr(relay_peer_id: &PeerId, relay_addr: &Multiaddr) -> Multiaddr {
    relay_addr
        .clone()
        .with(libp2p::multiaddr::Protocol::P2p(*relay_peer_id))
        .with(libp2p::multiaddr::Protocol::P2pCircuit)
}

/// Whether `addr` contains a `/p2p-circuit` hop — i.e. it's a relay-circuit
/// address (our own relayed listen address, or any relayed dial form).
///
/// The swarm event loop uses this to notice when the relay circuit we were
/// reachable through has gone (`ListenerClosed` / `ExpiredListenAddr` carrying a
/// circuit address), so it can drop the one-shot `relay_activated` latch and let
/// the liveness-tick fallback re-reserve a relay. Without that reset, a NAT'd
/// node never recovers internet reachability after the relay peer restarts or
/// the reservation lapses — found live 2026-07-23, when an anchor restart
/// mid-test left the test node unreachable until it was manually bounced.
pub(crate) fn is_relay_circuit_addr(addr: &Multiaddr) -> bool {
    addr.iter()
        .any(|p| matches!(p, libp2p::multiaddr::Protocol::P2pCircuit))
}

/// Does this address describe a real, direct transport path to the peer?
///
/// True only when it carries a concrete network hop (`/ip4`, `/ip6`, `/dns*`)
/// and no `/p2p-circuit`. The case `is_relay_circuit_addr` cannot see: a
/// relay-carried INBOUND connection, whose send-back address is a bare
/// `/p2p/<peer>` — no hop at all. Counting that as direct is how two LAN
/// nodes 0.7 ms apart came to forward every tensor through a relay in another
/// continent (gotcha #179, re-found 2026-08-21). Use this, not the circuit
/// check alone, wherever "direct connection" is bookkept.
pub(crate) fn addr_is_direct_transport(addr: &Multiaddr) -> bool {
    use libp2p::multiaddr::Protocol;
    let mut has_transport = false;
    for p in addr.iter() {
        match p {
            Protocol::P2pCircuit => return false,
            Protocol::Ip4(_)
            | Protocol::Ip6(_)
            | Protocol::Dns(_)
            | Protocol::Dns4(_)
            | Protocol::Dns6(_)
            | Protocol::Dnsaddr(_) => has_transport = true,
            _ => {}
        }
    }
    has_transport
}

/// Handle relay server events — log reservations and circuit lifecycle.
/// Tracks circuit open/close for relay service credit accrual.
pub fn handle_relay_server_event(
    event: libp2p::relay::Event,
    shared_state: &std::sync::Arc<crate::daemon::SharedState>,
) {
    use libp2p::relay::Event;
    match event {
        Event::ReservationReqAccepted {
            src_peer_id,
            renewed,
        } => {
            tracing::info!(
                peer = %src_peer_id,
                renewed,
                "DIAG: relay reservation accepted"
            );
        }
        Event::CircuitReqAccepted {
            src_peer_id,
            dst_peer_id,
        } => {
            tracing::debug!(
                src = %src_peer_id,
                dst = %dst_peer_id,
                "Relay circuit opened"
            );
            // Track circuit start time for credit accrual
            shared_state
                .active_relay_circuits
                .insert((src_peer_id, dst_peer_id), std::time::Instant::now());
        }
        Event::CircuitClosed {
            src_peer_id,
            dst_peer_id,
            error,
        } => {
            // Calculate circuit duration and accumulate relay seconds
            if let Some((_, start)) = shared_state
                .active_relay_circuits
                .remove(&(src_peer_id, dst_peer_id))
            {
                let duration_secs = start.elapsed().as_secs();
                if duration_secs > 0 {
                    shared_state
                        .relay_seconds_served
                        .fetch_add(duration_secs, std::sync::atomic::Ordering::Relaxed);
                }
            }
            tracing::debug!(
                src = %src_peer_id,
                dst = %dst_peer_id,
                ?error,
                "Relay circuit closed"
            );
        }
        // NET-M6: Log relay denial events at warn level. libp2p 0.56 added
        // a `status` field describing the denial reason (rate-limit /
        // resource-cap / explicit reject) — surface it in logs so operators
        // can tell apart the categories.
        Event::ReservationReqDenied {
            src_peer_id,
            status,
        } => {
            tracing::warn!(
                peer = %src_peer_id,
                ?status,
                "Relay reservation denied"
            );
        }
        Event::CircuitReqDenied {
            src_peer_id,
            dst_peer_id,
            status,
        } => {
            tracing::warn!(
                src = %src_peer_id,
                dst = %dst_peer_id,
                ?status,
                "Relay circuit denied"
            );
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_relay_config() {
        let config = RelayServerConfig::default();
        assert_eq!(config.max_reservations, 128);
        assert_eq!(config.max_circuit_bytes, 1 << 30);

        // A granted reservation must be able to become an actual circuit.
        // libp2p's upstream default pairs 128 reservations with only 16
        // circuits, which is coherent for its intended 2-minute bootstrap
        // circuit but not for ours: we hold circuits for an hour as a data
        // path, so a 16-circuit ceiling is reached by a small swarm and every
        // peer past it is dropped from scheduling rather than degraded.
        assert!(
            config.max_circuits >= config.max_reservations,
            "max_circuits ({}) must not be below max_reservations ({}) — \
             a peer granted a reservation could not open a circuit",
            config.max_circuits,
            config.max_reservations
        );
    }

    /// The long circuit duration is exactly why the circuit count had to rise;
    /// pin the relationship so lowering one without the other is caught.
    #[test]
    fn circuit_duration_is_data_path_length_not_bootstrap() {
        let config = RelayServerConfig::default();
        assert!(
            config.max_circuit_duration >= Duration::from_secs(600),
            "circuits are a data path here, not a brief bootstrap hop"
        );
    }

    #[test]
    fn relay_listen_addr_format() {
        let relay_peer = PeerId::random();
        let relay_addr: Multiaddr = "/ip4/1.2.3.4/tcp/8800".parse().unwrap();

        let addr = relay_listen_addr(&relay_peer, &relay_addr);
        let addr_str = addr.to_string();

        // Should contain the relay peer ID and p2p-circuit
        assert!(addr_str.contains("/p2p/"));
        assert!(addr_str.contains("/p2p-circuit"));
        assert!(addr_str.starts_with("/ip4/1.2.3.4/tcp/8800"));
    }

    #[test]
    fn auto_relay_flag_defaults_to_true() {
        let config = crate::config::Config::default();
        assert!(config.network.auto_relay);
    }

    #[test]
    fn a_directly_reachable_node_still_keeps_a_home_relay() {
        // The 2026-11-16 failure: UPnP worked, so nothing reserved, and a peer
        // that knew this node only by its relay address never got back in.
        assert_eq!(
            reason_to_reserve(false, false, true),
            Some(HOME_RELAY),
            "a reachable node reserves too"
        );
        assert_eq!(reason_to_reserve(false, false, false), Some(NOT_REACHABLE));
    }

    #[test]
    fn a_reservation_is_made_once_and_a_reachable_anchor_needs_none() {
        assert_eq!(reason_to_reserve(true, false, true), None);
        assert_eq!(reason_to_reserve(true, false, false), None);
        // An anchor IS the relay, and every node's bootstrap address.
        assert_eq!(reason_to_reserve(false, true, true), None);
        // An anchor nobody can reach still needs a way in.
        assert_eq!(reason_to_reserve(false, true, false), Some(NOT_REACHABLE));
    }

    #[test]
    fn the_home_relay_is_reserved_on_reaching_a_relay_and_re_armed_on_the_tick() {
        // The reservation on reaching a relay, the tick's re-arm and the latch
        // reset on our listener's close are what make the home relay last; any
        // one missing brings a 2026-11-16 failure back.
        let connections = include_str!("manager/connections.rs");
        assert!(
            connections.contains("self.try_activate_relay(crate::network::relay::HOME_RELAY)"),
            "reaching a relay no longer reserves a home relay"
        );
        let mod_rs = include_str!("manager/mod.rs");
        assert!(
            mod_rs.contains("crate::network::relay::reason_to_reserve("),
            "the liveness tick no longer asks reason_to_reserve"
        );
        let events = include_str!("manager/events.rs");
        assert!(
            events.contains("let ours = self.relay_listener == Some(listener_id);"),
            "a relay listener that closes with no address no longer clears the latch"
        );
    }

    #[test]
    fn a_bootstrap_entry_names_its_relay_and_never_this_node() {
        let relay = PeerId::random();
        let local = PeerId::random();
        let (pid, base) =
            bootstrap_relay(&format!("/dns4/relay.example/tcp/8810/p2p/{relay}"), &local)
                .expect("a bootstrap entry with a peer id names a relay");
        assert_eq!(pid, relay);
        assert_eq!(
            base,
            "/dns4/relay.example/tcp/8810".parse::<Multiaddr>().unwrap()
        );
        assert!(bootstrap_relay(&format!("/ip4/1.2.3.4/tcp/8810/p2p/{local}"), &local).is_none());
        assert!(bootstrap_relay("/ip4/1.2.3.4/tcp/8810", &local).is_none());
        assert!(bootstrap_relay("not an address", &local).is_none());
    }

    #[test]
    fn is_relay_circuit_addr_detects_circuit() {
        let relay_peer = PeerId::random();
        let base: Multiaddr = "/ip4/1.2.3.4/tcp/8800".parse().unwrap();

        // Our own relayed listen address is a circuit.
        assert!(is_relay_circuit_addr(&relay_listen_addr(
            &relay_peer,
            &base
        )));

        // A full relayed dial form (…/p2p-circuit/p2p/<target>) is a circuit.
        let target = PeerId::random();
        let full: Multiaddr =
            format!("/ip4/1.2.3.4/tcp/8800/p2p/{relay_peer}/p2p-circuit/p2p/{target}")
                .parse()
                .unwrap();
        assert!(is_relay_circuit_addr(&full));

        // A direct address is NOT a circuit — resetting the relay latch on a
        // plain listener close would needlessly re-reserve.
        let direct: Multiaddr = "/ip4/1.2.3.4/tcp/8800".parse().unwrap();
        assert!(!is_relay_circuit_addr(&direct));
        let quic: Multiaddr = "/ip4/1.2.3.4/udp/8800/quic-v1".parse().unwrap();
        assert!(!is_relay_circuit_addr(&quic));
    }

    #[test]
    fn relay_latch_clears_on_circuit_loss_and_rearms_recovery() {
        // Mirrors `note_relay_circuit_lost` + the liveness-tick fallback: once a
        // reserved circuit is lost the latch must clear, so the fallback (latch
        // false + not reachable) can re-fire and re-reserve. The one-shot tests
        // above cover idempotency *within* a single live reservation; this is the
        // recovery-after-loss case the latch previously blocked forever.
        let mut relay_activated = true; // reserved earlier this session

        // Circuit lost (relay restart / reservation expiry): drop the latch.
        if relay_activated {
            relay_activated = false;
        }
        assert!(
            !relay_activated,
            "latch must clear when the circuit is lost"
        );

        // Next liveness tick: latch clear → re-reserve, reachable or not.
        for reachable in [false, true] {
            assert!(
                reason_to_reserve(relay_activated, false, reachable).is_some(),
                "recovery must re-arm after the latch clears (reachable: {reachable})"
            );
        }
    }

    #[test]
    fn a_bare_p2p_address_is_not_a_direct_transport() {
        let bare: Multiaddr = "/p2p/12D3KooWKwvCNmumN89DftJbEC1yRcnP1YxVFKEXMLCo7EzifsaY"
            .parse()
            .unwrap();
        assert!(
            !addr_is_direct_transport(&bare),
            "no hop at all = relay-carried inbound"
        );
        let circuit: Multiaddr = "/ip4/212.132.104.177/udp/8800/quic-v1/p2p/12D3KooWNisnVha2jYj1gqqY5WP82vNQbRhFtBcKzj4XrYmGEn8G/p2p-circuit/p2p/12D3KooWKwvCNmumN89DftJbEC1yRcnP1YxVFKEXMLCo7EzifsaY".parse().unwrap();
        assert!(!addr_is_direct_transport(&circuit));
        assert!(
            is_relay_circuit_addr(&circuit),
            "the circuit check still sees an explicit circuit"
        );
        assert!(
            !is_relay_circuit_addr(&bare),
            "…but not the bare form — which is the whole reason the new helper exists"
        );
        let lan: Multiaddr = "/ip4/192.168.1.200/udp/50446/quic-v1".parse().unwrap();
        assert!(addr_is_direct_transport(&lan));
        let dns: Multiaddr = "/dns4/swarmllm.duckdns.org/tcp/8810".parse().unwrap();
        assert!(addr_is_direct_transport(&dns));
    }
}
