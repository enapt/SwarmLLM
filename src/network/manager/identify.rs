//! Identify protocol handler — peer NodeId derivation, encryption session
//! establishment, peer_registry/peer_to_node insertion, LAN auto-detection,
//! anti-gaming subnet tracking, and bounded eviction of distant peers.
//!
//! Called from `events.rs::handle_swarm_event` on `IdentifyEvent::Received`.
//! Must remain synchronous (no `.await`) — the swarm event loop cannot tolerate
//! suspending here. All cross-task state access uses try_lock / DashMap /
//! atomics for that reason.

use crate::network::helpers::{extract_ipv4_bytes, is_non_public_ipv4_bytes};
use crate::types::PeerInfo;

use super::NetworkManager;

impl NetworkManager {
    /// Handle Identify protocol — peer identified, establish encryption, register in peer_registry.
    ///
    /// Must remain `fn` (not `async fn`): this is called from inside the
    /// swarm event loop and any `.await` on a lock / I/O would stall the
    /// entire NetworkManager. All state access below uses `try_lock` /
    /// DashMap / atomics for exactly this reason.
    pub(super) fn handle_identify_received(
        &mut self,
        peer_id: libp2p::PeerId,
        info: libp2p::identify::Info,
        connection_id: libp2p::swarm::ConnectionId,
    ) {
        tracing::debug!(
            %peer_id,
            protocol_version = %info.protocol_version,
            listen_addrs = ?info.listen_addrs,
            "Identified peer"
        );
        // Add ONLY the connected address to Kademlia — not all listen_addrs.
        // Adding all addresses causes Kademlia to route DHT queries through
        // addresses we haven't connected on, triggering redundant dials that
        // create multiple connections per peer. request_response round-robins
        // across connections, and degraded connections silently drop messages.
        if let Some(connected_addr) = self.connection_addrs.get(&connection_id) {
            self.swarm
                .behaviour_mut()
                .kademlia
                .add_address(&peer_id, connected_addr.clone());
            tracing::debug!(
                %peer_id,
                addr = %connected_addr,
                "Added connected address to Kademlia (skipped {} other listen_addrs)",
                info.listen_addrs.len().saturating_sub(1)
            );
        } else {
            // No tracked connection address. This is the NORMAL case for an
            // INBOUND connection: we deliberately do not record the remote
            // address of a connection we didn't dial, because it is the peer's
            // ephemeral source port rather than anything it listens on
            // (see `handle_connection_established`).
            //
            // So fall back to what the peer ADVERTISES, filtered for addresses
            // worth dialling — `first()` alone could hand Kademlia the peer's
            // loopback or a private address from a network we aren't on, which
            // then fails every dial.
            let dialable = info.listen_addrs.iter().find(|a| {
                super::events::addr_is_remotely_reachable(a)
                    && !crate::network::relay::is_relay_circuit_addr(a)
            });
            match dialable {
                Some(addr) => {
                    self.swarm
                        .behaviour_mut()
                        .kademlia
                        .add_address(&peer_id, addr.clone());
                    tracing::debug!(
                        %peer_id,
                        addr = %addr,
                        "Inbound connection — used the peer's advertised address for Kademlia"
                    );
                }
                None => {
                    // Nothing dialable advertised: a fully NAT'd peer reachable
                    // only via a relay. Adding nothing is correct — a bogus entry
                    // would make every future dial to it fail with a refusal.
                    tracing::debug!(
                        %peer_id,
                        advertised = info.listen_addrs.len(),
                        "Inbound connection with no directly dialable advertised \
                         address — leaving Kademlia untouched (relay-only peer)"
                    );
                }
            }
        }
        // Verify announced key matches the authenticated PeerId from Noise handshake
        // to prevent NodeId spoofing via forged Identify messages.
        let announced_peer_id = info.public_key.to_peer_id();
        if announced_peer_id != peer_id {
            tracing::warn!(
                %peer_id,
                announced = %announced_peer_id,
                "Peer announced mismatched public key in Identify — ignoring"
            );
            return;
        }

        // Derive NodeId from the peer's Ed25519 public key (32 bytes)
        // per spec: NodeId(verifying_key.to_bytes())
        let node_id = if let Ok(ed_key) = info.public_key.clone().try_into_ed25519() {
            crate::types::NodeId(ed_key.to_bytes())
        } else {
            // Fallback for non-Ed25519 keys: hash the peer_id
            let hash = blake3::hash(&peer_id.to_bytes());
            crate::types::NodeId(*hash.as_bytes())
        };

        // Establish encryption session from the peer's Ed25519 public key
        if let Ok(ed_key) = info.public_key.clone().try_into_ed25519() {
            if let Some(x25519_pub) =
                crate::crypto::session::ed25519_pubkey_to_x25519(&ed_key.to_bytes())
            {
                tracing::info!(
                    %peer_id,
                    node_id = %node_id,
                    session_type = "static",
                    "DIAG: key exchange initiated"
                );
                self.shared_state
                    .session_manager
                    .establish_session(&node_id, x25519_pub);
                tracing::info!(
                    %peer_id,
                    node_id = %node_id,
                    session_type = "static",
                    session_count = self.shared_state.session_manager.session_count(),
                    "DIAG: encryption session established"
                );
            }
        }

        let now_ts = crate::types::unix_now_secs();
        // Preserve first_seen from existing entry or use current time
        let first_seen = self
            .shared_state
            .peer_registry
            .get(&node_id)
            .map(|p| p.first_seen)
            .unwrap_or(now_ts);
        // Preserve trust, capability, and verified count from existing entry
        let existing = self.shared_state.peer_registry.get(&node_id);
        let trust_score = existing.as_ref().map(|p| p.trust_score).unwrap_or(0.5);
        let capability = existing.as_ref().and_then(|p| p.capability.clone());
        let vtc = existing
            .as_ref()
            .map(|p| p.verified_transaction_count)
            .unwrap_or(0);
        let was_lan = existing.as_ref().map(|p| p.is_lan_peer).unwrap_or(false);
        drop(existing);
        // A peer is on our LAN only if the ACTUAL connection to it runs over a
        // private/loopback/link-local address, or it observes US on such an
        // address (same private network). We deliberately do NOT infer LAN from
        // the peer's advertised `listen_addrs`: a public node bound to 0.0.0.0
        // advertises `127.0.0.1` (and often a private cloud-interface IP) too,
        // which used to mislabel every remote peer — e.g. a public relay anchor
        // reached over its public IP — as "LAN".
        let addr_is_lan = multiaddr_is_local(&info.observed_addr)
            || self
                .peer_remote_addrs
                .get(&peer_id)
                .map(multiaddr_is_local)
                .unwrap_or(false);
        let is_lan = was_lan || addr_is_lan;
        let peer_info = PeerInfo {
            node_id: node_id.clone(),
            addresses: info
                .listen_addrs
                .iter()
                .take(8)
                .map(|a| a.to_string())
                .collect(),
            capability,
            last_seen: chrono::Utc::now(),
            latency_ms: None,
            trust_score,
            peer_id_bytes: Some(peer_id.to_bytes()),
            active_request_count: 0,
            first_seen,
            verified_transaction_count: vtc,
            is_lan_peer: is_lan,
        };
        // Insert peer_registry BEFORE peer_to_node to prevent TOCTOU race
        // where dispatch can resolve NodeId from peer_to_node but peer_registry
        // check fails because insert hasn't happened yet.
        self.shared_state
            .peer_registry
            .insert(node_id.clone(), peer_info);
        // If identify just newly marked this peer as LAN (based on its advertised
        // addresses), bump the LAN peer counter and emit a discovery event. This
        // covers peers that were reached via loopback probe or bootstrap where
        // mDNS never fired but the addresses are clearly local.
        if !was_lan && addr_is_lan {
            let count = self
                .shared_state
                .lan_peer_count
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
                + 1;
            let msg = format!(
                "Found {} peer{} on your local network",
                count,
                if count == 1 { "" } else { "s" }
            );
            tracing::info!(%peer_id, lan_peers = count, "LAN peer detected from listen_addrs");
            self.shared_state.emit_activity(
                crate::daemon::state::ActivityEvent::new("network", "lan_peer_discovered", msg)
                    .with_detail_num(count as i64)
                    .with_toast("success", 8000),
            );
        }
        // Restore persisted trust score from DB (survives restarts)
        let persisted_trust = self.shared_state.credits.trust_manager.get_trust(&node_id);
        if (persisted_trust - 0.5_f32).abs() > f32::EPSILON {
            if let Some(mut peer) = self.shared_state.peer_registry.get_mut(&node_id) {
                peer.trust_score = persisted_trust;
            }
        }
        self.shared_state
            .signal_dashboard(crate::daemon::state::DashboardSignal::PeersChanged);

        // Emit activity event for peer connection
        {
            let label = crate::identity::nickname::short_display_name(
                &node_id,
                &self.shared_state.nickname_registry,
            );
            let gpu_name = self.shared_state.peer_registry.get(&node_id).and_then(|p| {
                p.capability
                    .as_ref()
                    .and_then(|c| c.gpu.as_ref().map(|g| g.name.clone()))
            });
            let detail = if is_lan { "LAN" } else { "WAN" };
            self.shared_state.emit_activity(
                crate::daemon::state::ActivityEvent::new(
                    "network",
                    "peer_connected",
                    format!(
                        "Peer connected: {}{}",
                        label,
                        gpu_name
                            .as_ref()
                            .map(|g| format!(" ({})", g))
                            .unwrap_or_default()
                    ),
                )
                .with_node(format!("{}", node_id))
                .with_detail_str(detail.to_string()),
            );
        }

        // S3: Cap peer_registry to prevent unbounded growth at 10K+ nodes.
        // Evict highest-latency non-LAN non-pipeline peer when over limit.
        const MAX_PEER_REGISTRY: usize = 200;
        if self.shared_state.peer_registry.len() > MAX_PEER_REGISTRY {
            // Find the worst peer to evict: highest latency, not LAN, not in active pipeline
            let active_pipeline_nodes: std::collections::HashSet<_> = {
                let segments: Vec<_> = self
                    .shared_state
                    .active_pipelines
                    .iter()
                    .flat_map(|e| {
                        e.value()
                            .segments
                            .iter()
                            .map(|s| s.node_id.clone())
                            .collect::<Vec<_>>()
                    })
                    .collect();
                segments.into_iter().collect()
            };
            let evict_candidate = self
                .shared_state
                .peer_registry
                .iter()
                .filter(|e| {
                    !e.is_lan_peer
                        && !active_pipeline_nodes.contains(e.key())
                        && *e.key() != node_id
                })
                // Prefer evicting peers with known high latency over unmeasured peers.
                // Unmeasured peers (None) get 0 so they survive until measured.
                .max_by_key(|e| e.latency_ms.unwrap_or(0))
                .map(|e| e.key().clone());
            if let Some(evict_id) = evict_candidate {
                self.shared_state.peer_registry.remove(&evict_id);
                // Also remove from peer_to_node and disconnect
                let evict_peer = self
                    .peer_to_node
                    .iter()
                    .find(|e| *e.value() == evict_id)
                    .map(|e| *e.key());
                if let Some(pid) = evict_peer {
                    self.peer_to_node.remove(&pid);
                    let _ = self.swarm.disconnect_peer_id(pid);
                }
                tracing::debug!(
                    evicted = %evict_id,
                    registry_size = self.shared_state.peer_registry.len(),
                    "Evicted distant peer to stay under registry cap"
                );
            }
        }

        // NET-C4: Populate reverse PeerId → NodeId lookup (capped)
        const MAX_PEER_TO_NODE: usize = 10_000;
        if self.peer_to_node.len() < MAX_PEER_TO_NODE || self.peer_to_node.contains_key(&peer_id) {
            self.peer_to_node.insert(peer_id, node_id.clone());
        }
        // Ground-truth connection set — consumed by HealthMonitor to skip eviction.
        // Populated here (not at ConnectionEstablished) because we only learn
        // the NodeId after Identify. Identify also re-pushes periodically;
        // DashSet insert is idempotent so repeat inserts are harmless.
        self.shared_state.connected_node_ids.insert(node_id.clone());
        // R110: refresh swarm-capacity snapshot so the dashboard banner
        // reflects the new contributor without waiting for the next ~1.5s
        // stats-cache tick.
        crate::daemon::state::refresh_swarm_capacity(&self.shared_state);
        // Persistent NodeId → PeerId mapping (survives disconnects, same cap)
        if self.shared_state.peer_id_map.len() < MAX_PEER_TO_NODE
            || self.shared_state.peer_id_map.contains_key(&node_id)
        {
            self.shared_state
                .peer_id_map
                .insert(node_id.clone(), peer_id.to_bytes());
        }

        // Layer 6: Track subnet for anti-gaming — extract IPv4 from listen addrs.
        // Skip non-public IPs (loopback, RFC 1918, link-local, CGN/Tailscale,
        // unspecified) so subnet clustering can't false-positive on internal
        // addresses. Uses the shared helper rather than re-implementing the
        // RFC checks inline (the inline version was missing 169.254.x.x and
        // the CGN range, leaking those into the anti-gaming tracker).
        for addr in &info.listen_addrs {
            if let Some(ip_bytes) = extract_ipv4_bytes(addr) {
                if is_non_public_ipv4_bytes(&ip_bytes) {
                    continue;
                }
                // Use try_lock() to avoid blocking the event loop.
                // If contended, skip — next Identify event will catch it.
                if let Ok(mut anti_gaming) = self.shared_state.credits.anti_gaming.try_lock() {
                    anti_gaming.register_subnet(&node_id, ip_bytes);
                }
                break; // One IP per peer is enough
            }
        }
    }
}

/// Does this multiaddr denote a private / loopback / link-local address — one
/// that implies same-LAN reachability? Applied by the identify handler to the
/// ACTUAL connection address and to the peer's observed view of us, NOT to the
/// peer's advertised listen_addrs (those include 127.0.0.1 for any 0.0.0.0-bound
/// node, which would mislabel remote peers as LAN).
fn multiaddr_is_local(a: &libp2p::Multiaddr) -> bool {
    a.iter().any(|proto| match proto {
        libp2p::multiaddr::Protocol::Ip4(ip) => {
            ip.is_loopback() || ip.is_private() || ip.is_link_local()
        }
        libp2p::multiaddr::Protocol::Ip6(ip) => {
            ip.is_loopback() || (ip.segments()[0] & 0xfe00) == 0xfc00
        }
        _ => false,
    })
}

#[cfg(test)]
mod tests {
    use super::multiaddr_is_local;

    fn a(s: &str) -> libp2p::Multiaddr {
        s.parse().unwrap()
    }

    #[test]
    fn public_addr_is_not_local() {
        // The bug this fixes: a public relay anchor reached over its public IP
        // must NOT be classified as LAN.
        assert!(!multiaddr_is_local(&a("/ip4/212.132.104.177/tcp/8810")));
        assert!(!multiaddr_is_local(&a("/ip4/8.8.8.8/udp/8800/quic-v1")));
        assert!(!multiaddr_is_local(&a("/ip6/2001:db8::1/tcp/8810")));
    }

    #[test]
    fn private_and_loopback_are_local() {
        assert!(multiaddr_is_local(&a("/ip4/192.168.1.5/tcp/8810")));
        assert!(multiaddr_is_local(&a("/ip4/10.0.0.7/tcp/8810")));
        assert!(multiaddr_is_local(&a("/ip4/127.0.0.1/tcp/8810")));
        assert!(multiaddr_is_local(&a("/ip6/::1/tcp/8810")));
        assert!(multiaddr_is_local(&a("/ip6/fc00::1/tcp/8810")));
    }
}
