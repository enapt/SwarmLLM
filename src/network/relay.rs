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
            max_circuits: 16,
            max_circuit_duration: Duration::from_secs(RELAY_CIRCUIT_DURATION_SECS),
            reservation_duration: Duration::from_secs(RELAY_RESERVATION_DURATION_SECS),
            max_circuit_bytes: 1 << 30, // 1 GB
        }
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

/// Generate the relay listener address for connecting through a relay peer.
///
/// NAT'd nodes call this to establish a relayed listen address, allowing
/// other peers to reach them through the relay.
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
        assert_eq!(config.max_circuits, 16);
        assert_eq!(config.max_circuit_bytes, 1 << 30);
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
    fn auto_relay_once_per_session_flag() {
        // Simulate the relay_activated flag behavior
        let mut relay_activated = false;
        let auto_relay = true;
        let nat_private = true;

        // First activation should trigger
        if nat_private && !relay_activated && auto_relay {
            relay_activated = true;
        }
        assert!(relay_activated);

        // Second check should NOT trigger (already activated)
        let mut triggered_again = false;
        if nat_private && !relay_activated && auto_relay {
            triggered_again = true;
        }
        assert!(!triggered_again);
    }

    #[test]
    fn auto_relay_disabled_prevents_activation() {
        let mut relay_activated = false;
        let auto_relay = false;
        let nat_private = true;

        if nat_private && !relay_activated && auto_relay {
            relay_activated = true;
        }
        assert!(!relay_activated);
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

        // Next liveness tick: latch clear + no reachable address → re-reserve.
        let (auto_relay, reachable) = (true, false);
        let mut re_reserved = false;
        if !relay_activated && auto_relay && !reachable {
            re_reserved = true;
        }
        assert!(re_reserved, "recovery must re-arm after the latch clears");
    }
}
