//! Per-connection lifecycle handlers.
//!
//! `handle_connection_established` tracks the new ConnectionId/remote address,
//! sends the initial PEX request on first connection, and bumps the peer count.
//! `handle_connection_closed` cleans up `peer_registry`, encryption sessions,
//! pending shard downloads, and active-pipeline reconciliation. Both fire from
//! the swarm event loop in `events.rs`.

use libp2p::request_response::OutboundRequestId;

use crate::network::protocol::SwarmRequest;
use crate::types::SwarmMessage;

use super::{
    NetworkManager, MAX_CONNECTION_ADDRS, MAX_PEER_REMOTE_ADDRS, MAX_PENDING_REDIAL,
    MAX_PING_ENTRIES, PING_SENT_TIMES_CUTOFF_SECS, REDIAL_JITTER_MIN_MS, REDIAL_JITTER_RANGE_MS,
};

impl NetworkManager {
    /// Handle new peer connection — track address, send PEX request.
    pub(super) fn handle_connection_established(
        &mut self,
        peer_id: libp2p::PeerId,
        connection_id: libp2p::swarm::ConnectionId,
        num_established: std::num::NonZeroU32,
        endpoint: &libp2p::core::ConnectedPoint,
    ) {
        let remote_addr = endpoint.get_remote_address();
        let is_loopback = remote_addr.iter().any(|proto| {
            matches!(proto, libp2p::multiaddr::Protocol::Ip4(ip) if ip.is_loopback())
                || matches!(proto, libp2p::multiaddr::Protocol::Ip6(ip) if ip.is_loopback())
        });
        tracing::info!(
            %peer_id, %connection_id, count = num_established,
            remote_addr = %remote_addr,
            is_loopback,
            is_dialer = endpoint.is_dialer(),
            total_established = self.swarm.network_info().connection_counters().num_established(),
            total_peers = self.swarm.connected_peers().count(),
            pending_tensor_forwards = self.pending_tensor_outbound.len(),
            "DIAG: connection established"
        );
        // Track which address each connection uses — the Identify handler
        // uses this to add only the connected address to Kademlia.
        // SEC: Cap connection_addrs to prevent unbounded memory growth.
        if self.connection_addrs.len() >= MAX_CONNECTION_ADDRS {
            // Evict oldest half — stale ConnectionIds from missed close events.
            let mut ids: Vec<_> = self.connection_addrs.keys().cloned().collect();
            ids.sort();
            for id in ids.iter().take(MAX_CONNECTION_ADDRS / 2) {
                self.connection_addrs.remove(id);
            }
        }
        self.connection_addrs
            .insert(connection_id, remote_addr.clone());
        // Cap peer_remote_addrs at MAX_PEER_REMOTE_ADDRS — disconnected peers'
        // entries are removed in handle_connection_closed, but a cap defends
        // against missed close events leaking entries indefinitely. Drop a
        // random half via take() since peer_id has no natural ordering.
        if self.peer_remote_addrs.len() >= MAX_PEER_REMOTE_ADDRS {
            let to_drop: Vec<_> = self
                .peer_remote_addrs
                .keys()
                .take(MAX_PEER_REMOTE_ADDRS / 2)
                .copied()
                .collect();
            for k in to_drop {
                self.peer_remote_addrs.remove(&k);
            }
        }
        self.peer_remote_addrs.insert(peer_id, remote_addr.clone());
        self.update_peer_count();

        // Layer 5: Peer Exchange — send PEX request on first connection only
        if num_established.get() == 1 && self.shared_state.config.network.peer_exchange {
            // SEC: Cap ping_sent_times to prevent unbounded growth from connection storms.
            // Prune stale entries before inserting.
            if self.ping_sent_times.len() >= MAX_PING_ENTRIES {
                let cutoff = std::time::Instant::now()
                    - std::time::Duration::from_secs(PING_SENT_TIMES_CUTOFF_SECS);
                self.ping_sent_times
                    .retain(|_, (_, sent_at)| *sent_at > cutoff);
            }
            let req = SwarmRequest::Message(Box::new(SwarmMessage::PeerExchangeRequest));
            let outbound_id = self
                .swarm
                .behaviour_mut()
                .request_response
                .send_request(&peer_id, req);
            // Track send time for RTT measurement
            self.ping_sent_times
                .insert(outbound_id, (peer_id, std::time::Instant::now()));
            tracing::debug!(%peer_id, "Sent PEX request");
        }
    }

    /// Handle peer disconnection — cleanup registry, sessions, downloads.
    pub(super) fn handle_connection_closed(
        &mut self,
        peer_id: libp2p::PeerId,
        connection_id: libp2p::swarm::ConnectionId,
        cause: Option<libp2p::swarm::ConnectionError>,
        num_established: u32,
    ) {
        let closed_addr = self.connection_addrs.remove(&connection_id);
        if num_established == 0 {
            self.peer_remote_addrs.remove(&peer_id);
        }
        // Check if any in-flight tensor forwards are affected
        let affected_tensors: Vec<_> = self
            .pending_tensor_outbound
            .values()
            .map(|(u, _, _, _, _)| u.to_string())
            .collect();
        // Classify the close reason so operators can triage dropouts at a glance
        // without having to parse nested ConnectionError Debug output.
        let reason = match &cause {
            None => "clean_close",
            Some(libp2p::swarm::ConnectionError::KeepAliveTimeout) => "idle_timeout",
            Some(libp2p::swarm::ConnectionError::IO(_)) => "io_error",
        };
        // Emit at info! for clean and idle-timeout closes — those are normal
        // network churn and would otherwise drown warn-level alerting (in a
        // 20-peer node, ~20 idle-timeout closes fire per IDLE_CONNECTION_TIMEOUT
        // cycle). Reserve warn! for io_error which signals a real transport
        // problem worth a triage glance.
        let warn_level = matches!(reason, "io_error");
        if warn_level {
            tracing::warn!(
                %peer_id, %connection_id,
                reason,
                ?cause,
                ?closed_addr,
                remaining = num_established,
                pending_tensor_forwards = self.pending_tensor_outbound.len(),
                affected_request_ids = ?affected_tensors.iter().take(5).collect::<Vec<_>>(),
                total_peers = self.swarm.connected_peers().count(),
                "DIAG: connection closed"
            );
        } else {
            tracing::info!(
                %peer_id, %connection_id,
                reason,
                ?cause,
                ?closed_addr,
                remaining = num_established,
                pending_tensor_forwards = self.pending_tensor_outbound.len(),
                affected_request_ids = ?affected_tensors.iter().take(5).collect::<Vec<_>>(),
                total_peers = self.swarm.connected_peers().count(),
                "DIAG: connection closed"
            );
        }

        // Skip cleanup if other connections to this peer remain
        if num_established > 0 {
            tracing::debug!(%peer_id, remaining = num_established, "Other connections remain, skipping cleanup");
        } else if self.swarm.is_connected(&peer_id) {
            // Swarm still considers peer connected (race: another
            // connection was just established) — skip cleanup.
            tracing::debug!(%peer_id, "Peer still connected per swarm, skipping cleanup");
            self.update_peer_count();
        } else {
            self.update_peer_count();

            // NET-I1: Drain pending shard requests for this peer and kick each
            // into the failover path. Without this, libp2p's subsequent
            // OutboundFailure events fire AFTER we've already dropped the
            // pending_shard_requests entries, so the retry handler no-ops and
            // the user just sees a dead download.
            let drained_ids: Vec<OutboundRequestId> = self
                .pending_shard_requests
                .iter()
                .filter(|(_, (pid, _))| *pid == peer_id)
                .map(|(rid, _)| *rid)
                .collect();
            for rid in &drained_ids {
                if let Some((_, shard_id)) = self.pending_shard_requests.remove(rid) {
                    tracing::debug!(
                        %peer_id,
                        model = %shard_id.model_id,
                        index = shard_id.index,
                        "Disconnected peer had pending shard request — routing to failover"
                    );
                    self.retry_shard_or_fallback(shard_id, peer_id, "peer disconnected");
                }
            }

            // NET-I3: Clean up peer_shard_downloads for disconnected peer.
            // Entries for peers that disconnect mid-download would otherwise
            // be orphaned permanently, accumulating stale data.
            let node_id_for_cleanup = self.peer_to_node.get(&peer_id).map(|r| r.clone());
            if let Some(ref nid) = node_id_for_cleanup {
                // Remove from libp2p-connected ground-truth set so HealthMonitor
                // can now evict the peer_registry entry if it goes stale.
                self.shared_state.connected_node_ids.remove(nid);
                // R110: refresh swarm-capacity so the banner contributor count
                // and serveable-models list reflect the lost peer immediately
                // (the WS stats-cache otherwise lags by ~1.5s, leaving the
                // peer-list panel and the banner inconsistent under churn).
                crate::daemon::state::refresh_swarm_capacity(&self.shared_state);
                self.shared_state
                    .models
                    .peer_shard_downloads
                    .retain(|_shard_id, peers| {
                        peers.retain(|(n, _)| n != nid);
                        !peers.is_empty()
                    });

                // NET-I4: Clean up stale peer_credit_balances entry.
                // Prevents unbounded growth and stale entries skewing priority tier percentiles.
                self.shared_state.credits.peer_credit_balances.remove(nid);
            }

            // NET-I2: Remove peer from registry, but skip if in active pipelines.
            // Clone the NodeId and drop the DashMap Ref BEFORE calling remove(),
            // otherwise get() holds a read lock and remove() needs a write lock
            // on the same shard → synchronous deadlock that freezes the event loop.
            let node_id_opt = node_id_for_cleanup;
            if let Some(node_id) = node_id_opt {
                let in_active_pipeline = self.shared_state.active_pipelines.iter().any(|entry| {
                    entry
                        .value()
                        .segments
                        .iter()
                        .any(|seg| seg.node_id == node_id)
                });
                // Clear encryption session on full disconnect to
                // prevent epoch desync after reconnection.
                // Only remove if no new connection has been established
                // (prevents race where reconnect arrives before close is processed).
                // Keep the session alive if the peer is in an active pipeline —
                // reconnection will refresh it, and removing it mid-pipeline
                // causes "seal() failed" on pending TP forwards.
                if !self.swarm.is_connected(&peer_id) && !in_active_pipeline {
                    self.shared_state.session_manager.remove_session(&node_id);
                }

                if !in_active_pipeline {
                    // Capture info before removing
                    let label = crate::identity::nickname::short_display_name(
                        &node_id,
                        &self.shared_state.nickname_registry,
                    );

                    // Remove peer_to_node BEFORE peer_registry to prevent
                    // dispatch from resolving NodeId for a peer that's being removed
                    self.peer_to_node.remove(&peer_id);
                    self.shared_state.peer_registry.remove(&node_id);
                    // Also evict from shard holder registry. Without this, the
                    // pipeline scheduler keeps offering the dead peer as a
                    // candidate for ~90s (until the health-monitor stale-peer
                    // sweep runs), causing remote-generate to time out at the
                    // first-token timeout (120s) instead of routing to a live
                    // holder immediately.
                    self.shared_state
                        .model_registry
                        .remove_peer_from_all_shards(&node_id);
                    self.shared_state
                        .signal_dashboard(crate::daemon::state::DashboardSignal::PeersChanged);

                    self.shared_state.emit_activity(
                        crate::daemon::state::ActivityEvent::new(
                            "network",
                            "peer_disconnected",
                            format!("Peer disconnected: {}", label),
                        )
                        .with_node(format!("{}", node_id)),
                    );
                    tracing::debug!(%peer_id, "Removed disconnected peer from registry");
                } else {
                    tracing::info!(%peer_id, "Keeping peer in registry (active pipeline) — scheduling reconnect");
                    // Active pipeline needs this peer — reconnect immediately.
                    if let Some(addr) = closed_addr.clone() {
                        self.try_enqueue_redial(peer_id, addr, 500);
                    }
                }
            } else {
                // Peer was never registered (connection died before Identify).
                // This typically happens during mDNS simultaneous-dial race.
                // Schedule a re-dial with random jitter to break symmetry.
                if let Some(addr) = closed_addr {
                    use std::hash::{Hash, Hasher};
                    let mut hasher = std::collections::hash_map::DefaultHasher::new();
                    peer_id.hash(&mut hasher);
                    let jitter_ms =
                        REDIAL_JITTER_MIN_MS + (hasher.finish() % REDIAL_JITTER_RANGE_MS);
                    tracing::info!(
                        %peer_id, %addr, jitter_ms,
                        "Scheduling re-dial after connection race"
                    );
                    self.try_enqueue_redial(peer_id, addr, jitter_ms);
                }
            }
        } // end else (num_established == 0)
    }

    /// Enqueue a re-dial unless this peer already has one queued or the
    /// queue is at `MAX_PENDING_REDIAL`. Shared by both the active-pipeline
    /// and the unregistered-peer reconnect paths so the dedup+cap
    /// invariant lives in one place (R97).
    fn try_enqueue_redial(
        &mut self,
        peer_id: libp2p::PeerId,
        addr: libp2p::Multiaddr,
        delay_ms: u64,
    ) {
        let already_queued = self
            .pending_redial
            .iter()
            .any(|(pid, _, _)| *pid == peer_id);
        if !already_queued && self.pending_redial.len() < MAX_PENDING_REDIAL {
            let scheduled = std::time::Instant::now() + std::time::Duration::from_millis(delay_ms);
            self.pending_redial.push((peer_id, addr, scheduled));
        }
    }
}
