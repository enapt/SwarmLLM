use std::time::Duration;

use libp2p::{Multiaddr, PeerId};

/// Configuration for the relay server role.
///
/// When a node has `config.network.enable_relay = true` and is publicly
/// reachable, it acts as a relay server — accepting reservations from
/// NAT'd peers and forwarding circuit traffic.
pub struct RelayServerConfig {
    pub max_reservations: usize,
    pub max_circuits: usize,
    pub max_circuit_duration: Duration,
    pub max_circuit_bytes: u64,
}

impl Default for RelayServerConfig {
    fn default() -> Self {
        Self {
            max_reservations: 128,
            max_circuits: 16,
            max_circuit_duration: Duration::from_secs(3600),
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
        reservation_duration: config.max_circuit_duration,
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

/// Handle relay server events — log reservations and circuit lifecycle.
pub fn handle_relay_server_event(event: libp2p::relay::Event) {
    use libp2p::relay::Event;
    match event {
        Event::ReservationReqAccepted {
            src_peer_id,
            renewed,
        } => {
            tracing::info!(
                peer = %src_peer_id,
                renewed,
                "Relay reservation accepted"
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
        }
        Event::CircuitClosed {
            src_peer_id,
            dst_peer_id,
            error,
        } => {
            tracing::debug!(
                src = %src_peer_id,
                dst = %dst_peer_id,
                ?error,
                "Relay circuit closed"
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
}
