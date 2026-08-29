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
    NetworkManager, MAX_CONNECTION_ADDRS, MAX_CONSECUTIVE_RR_FAILURES, MAX_PEER_REMOTE_ADDRS,
    MAX_PENDING_REDIAL, MAX_PING_ENTRIES, MAX_REDIAL_ATTEMPTS, MAX_REDIAL_TRACKED_PEERS,
    PING_SENT_TIMES_CUTOFF_SECS, REDIAL_BACKOFF_MS, REDIAL_JITTER_MIN_MS, REDIAL_JITTER_RANGE_MS,
    RR_FAILURES_AFTER_SILENCE, RR_SILENCE_BEFORE_SHORT_RUN_COUNTS,
};

/// Addresses to attempt when re-dialling a peer whose last connection just
/// closed, most-specific first.
///
/// `closed_addr` is the address of the connection that died, and is `None`
/// whenever that connection was INBOUND — `handle_connection_established`
/// records addresses only for connections we dialled, because an inbound
/// connection's remote address is the peer's ephemeral source port (gotcha
/// #165). It is therefore never sufficient on its own, and `advertised_dialable`
/// (the peer's own identify-advertised listen addresses, already filtered by
/// [`crate::network::peer_cache::filter_dialable`]) carries the rest.
///
/// A relay-circuit `closed_addr` is dropped: re-dialling the circuit we just
/// lost rebuilds a relayed path to a peer we may well reach directly, and the
/// advertised set is the better answer. An empty result is legitimate — the
/// caller dials by peer id and lets the behaviours supply addresses.
fn redial_addresses(
    closed_addr: Option<&libp2p::Multiaddr>,
    advertised_dialable: &[String],
) -> Vec<libp2p::Multiaddr> {
    let mut out: Vec<libp2p::Multiaddr> = Vec::new();
    if let Some(addr) = closed_addr {
        if !crate::network::relay::is_relay_circuit_addr(addr) {
            out.push(addr.clone());
        }
    }
    for s in advertised_dialable {
        if let Ok(a) = s.parse::<libp2p::Multiaddr>() {
            if !out.contains(&a) {
                out.push(a);
            }
        }
    }
    out
}

/// Should a peer with this failure run, silent for this long, be disconnected?
///
/// Two ways in, because the two signals fail differently. A long run alone
/// (`MAX_CONSECUTIVE_RR_FAILURES`) catches a peer that answers occasionally but
/// mostly does not. A shorter run plus sustained silence
/// (`RR_FAILURES_AFTER_SILENCE` after `RR_SILENCE_BEFORE_SHORT_RUN_COUNTS`)
/// catches a dead return path in about two minutes instead of ten.
///
/// The short path cannot be reached by a merely busy peer: any successful
/// response resets both the run and the clock, so reaching it requires
/// answering NOTHING for the whole window. Neither threshold may go near the
/// anchor's measured worst healthy run of 5.
fn should_close_unresponsive(failures: u32, silent_for: std::time::Duration) -> bool {
    if failures >= MAX_CONSECUTIVE_RR_FAILURES {
        return true;
    }
    failures >= RR_FAILURES_AFTER_SILENCE && silent_for >= RR_SILENCE_BEFORE_SHORT_RUN_COUNTS
}

/// Backoff before the next re-dial, or `None` once the peer should be treated
/// as departed.
///
/// Indexing the schedule through this function is what makes "we have run out
/// of attempts" and "how long until the next one" the same decision, so a
/// schedule change cannot leave a stale bound behind or index past the end.
fn redial_backoff_ms(attempts_so_far: u32) -> Option<u64> {
    REDIAL_BACKOFF_MS.get(attempts_so_far as usize).copied()
}

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
        // Reaching the peer is what the retry schedule was counting down to, so
        // the budget resets here rather than expiring — a peer that flaps gets a
        // fresh set of attempts each time it comes back.
        self.redial_attempts.remove(&peer_id);
        // A fresh connection starts with a clean slate — the count is about one
        // connection's silence, not the peer's history.
        self.rr_failures.remove(&peer_id);
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
        // Only record the address for a connection WE dialled. For an outbound
        // connection the remote address is the peer's real listen address, so it
        // is dialable. For an INBOUND connection it is the peer's ephemeral
        // source port — recording it makes Kademlia hand out an address nothing
        // is listening on, and every later dial to that peer gets
        // "Connection refused".
        //
        // Confirmed live 2026-07-25: an external tester could not dial us back
        // even though our gossip reached them. Their node had learned four
        // ephemeral TCP ports for us (36986/37384/39802/39846) from OUR outbound
        // connections to THEM, and was dialling those instead of a real address.
        // Shared code, so both ends did it to each other; only the NAT'd side
        // shows the symptom, because the reachable side can always be dialled on
        // its real port anyway.
        if matches!(endpoint, libp2p::core::ConnectedPoint::Dialer { .. }) {
            self.connection_addrs
                .insert(connection_id, remote_addr.clone());
        } else if !is_loopback {
            // A remote peer dialled US and it worked — the only direct proof
            // that inbound reaches this node. Outbound succeeds from behind
            // almost any firewall, so nothing else in a healthy-looking node
            // distinguishes "reachable" from "silently dropping inbound".
            // Loopback is excluded: the dashboard's own browser would satisfy
            // it while proving nothing about the LAN.
            self.shared_state.record_inbound_connection_observed();
        }
        // NETWORKING_PLAN Phase 1 — record DIRECT connections so the relay
        // send path can prefer the app-level relay over a peer reachable only
        // via a flaky relay circuit. "Direct" means a real transport hop: a
        // relay-carried INBOUND connection has a bare `/p2p/<peer>` here and
        // must not be booked as direct (gotcha #179). Bounded alongside
        // connection_addrs (evict arbitrarily if a missed close leaked entries).
        if crate::network::relay::addr_is_direct_transport(remote_addr) {
            if self.peer_direct_conns.len() >= MAX_CONNECTION_ADDRS {
                let stale: Vec<_> = self
                    .peer_direct_conns
                    .keys()
                    .take(MAX_CONNECTION_ADDRS / 2)
                    .copied()
                    .collect();
                for p in stale {
                    self.peer_direct_conns.remove(&p);
                }
            }
            self.peer_direct_conns
                .entry(peer_id)
                .or_default()
                .insert(connection_id);
        }
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
        // Drop the failure count with the connection it described. Also the
        // only place the map shrinks, so cap it against a churn storm.
        self.rr_failures.remove(&peer_id);
        if self.rr_failures.len() > MAX_REDIAL_TRACKED_PEERS {
            self.rr_failures.clear();
        }
        // NETWORKING_PLAN Phase 1 — drop this connection from the peer's direct
        // set (no-op if it was a relay circuit, which was never inserted).
        if let Some(set) = self.peer_direct_conns.get_mut(&peer_id) {
            set.remove(&connection_id);
            if set.is_empty() {
                self.peer_direct_conns.remove(&peer_id);
            }
        }
        // `num_established == 0` for THIS close event does not mean the peer is
        // gone — a peer with two transports (TCP + QUIC) or a fast reconnect can
        // have a fresh connection the swarm already counts. The rest of this
        // function guards the identical race with `!is_connected` (see the branch
        // below); the abort path must too, or a benign TCP blip kills a live
        // remote-generate that is still returnable over the surviving connection.
        if num_established == 0 && !self.swarm.is_connected(&peer_id) {
            self.peer_remote_addrs.remove(&peer_id);
            // Abort any inbound remote-generations we were running for this
            // coordinator. With their last connection gone there is no route to
            // stream tokens back — a NAT'd coordinator can't be re-dialed (the
            // reverse token path relies on the existing connection), so
            // continuing only burns compute and worker time on output nobody
            // can receive (external report 2026-07-23). Aborting the task drops
            // the generate future, which cancels the worker via its
            // ResponseGuard (R147). Explicit CancelInference still handles the
            // connected-but-cancelled case; this covers the silent disconnect.
            let peer_bytes = peer_id.to_bytes();
            let orphaned: Vec<uuid::Uuid> = self
                .shared_state
                .inbound_generate_aborts
                .iter()
                .filter(|e| e.value().1 == peer_bytes)
                .map(|e| *e.key())
                .collect();
            for rid in orphaned {
                if let Some((_, (abort, _))) =
                    self.shared_state.inbound_generate_aborts.remove(&rid)
                {
                    abort.abort();
                    tracing::info!(
                        request_id = %rid,
                        %peer_id,
                        "Coordinator disconnected — aborting inbound remote-generate (no route to return tokens)"
                    );
                }
            }
            // Same reasoning for segment forwards: with the coordinator gone
            // there is nowhere to send the computed activations, so finishing
            // one only burns the worker on output nobody can receive — and
            // holds up every other request queued behind it.
            let orphaned_forwards: Vec<uuid::Uuid> = self
                .shared_state
                .inbound_forward_aborts
                .iter()
                .filter(|e| e.value().1 == peer_bytes)
                .map(|e| *e.key())
                .collect();
            for rid in orphaned_forwards {
                if let Some((_, (abort, _))) = self.shared_state.inbound_forward_aborts.remove(&rid)
                {
                    abort.abort();
                    tracing::info!(
                        request_id = %rid,
                        %peer_id,
                        "Coordinator disconnected — abandoning inbound segment forward (no route to return it)"
                    );
                }
            }
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
                if self.shared_state.connected_node_ids.remove(nid).is_some() {
                    // Pairs with "Peer connected" in identify.rs. Between them,
                    // every change to the peer count an operator sees has a
                    // line, which it did not before.
                    tracing::info!(
                        %peer_id,
                        node = %nid,
                        peers = self.shared_state.connected_node_ids.len(),
                        "Peer disconnected"
                    );
                }
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
                // Addresses to re-dial this peer on. MUST be read before the
                // `peer_registry` removal below, which is the only place that
                // still knows where this peer listens.
                //
                // `closed_addr` alone is not enough. It is populated only for
                // connections WE dialled (see `handle_connection_established` —
                // an inbound connection's remote address is the peer's
                // ephemeral source port, and recording it poisoned Kademlia,
                // gotcha #165). So for a peer that dialled US, `closed_addr` is
                // always `None`, and gating the re-dial on it meant no re-dial
                // was ever scheduled for that peer. That silently disabled the
                // 2026-07-27 fix for exactly the case it was written for:
                // observed 2026-08-02 with two LAN nodes 2 ms apart, mutually
                // invisible for over two hours after one inbound connection
                // dropped, both still connected to the same anchor, zero dial
                // attempts between them.
                //
                // A relay-circuit address is skipped as a *hint* — re-dialling
                // the circuit we just lost recreates a relayed path to a peer
                // we may be able to reach directly. The peer's own advertised
                // listen addresses are the better answer, and libp2p dials the
                // whole set concurrently.
                let advertised: Vec<String> = self
                    .shared_state
                    .peer_registry
                    .get(&node_id)
                    .map(|entry| entry.addresses.clone())
                    .unwrap_or_default();
                let local_addrs = self.shared_state.listen_multiaddrs.load();
                let dialable = crate::network::peer_cache::filter_dialable(
                    &advertised,
                    self.swarm.local_peer_id(),
                    &local_addrs,
                );
                let redial_addrs = redial_addresses(closed_addr.as_ref(), &dialable);
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

                    // Schedule one re-dial. Dropping the peer from the registry
                    // above is correct — a disconnected peer must not stay
                    // schedulable — but on its own it also meant we simply
                    // forgot a peer that is still running, and nothing brought
                    // it back: re-discovery relies on the peer ANNOUNCING
                    // itself, which only happens when it restarts. Two healthy
                    // LAN nodes were observed staying mutually invisible for
                    // 17+ minutes after transient `io_err` churn, both still
                    // connected to the same anchor, with zero dial attempts
                    // between them; restarting either one reconnected the pair
                    // in 6 seconds, which is what pinned it here rather than on
                    // addresses or discovery.
                    //
                    // Jittered like the unregistered-peer path below so both
                    // ends re-dialling at once don't recreate the
                    // simultaneous-dial race. `try_enqueue_redial` dedups and
                    // caps, so a peer that has genuinely left costs one attempt.
                    use std::hash::{Hash, Hasher};
                    let mut hasher = std::collections::hash_map::DefaultHasher::new();
                    peer_id.hash(&mut hasher);
                    let jitter_ms =
                        REDIAL_JITTER_MIN_MS + (hasher.finish() % REDIAL_JITTER_RANGE_MS);
                    tracing::info!(
                        %peer_id,
                        addr_count = redial_addrs.len(),
                        jitter_ms,
                        "Scheduling re-dial for disconnected peer"
                    );
                    self.try_enqueue_redial(peer_id, redial_addrs, jitter_ms);
                } else {
                    tracing::info!(%peer_id, "Keeping peer in registry (active pipeline) — scheduling reconnect");
                    // Active pipeline needs this peer — reconnect immediately.
                    self.try_enqueue_redial(peer_id, redial_addrs, 500);
                }
            } else if self.foreign_peers.contains(&peer_id) {
                // Identify succeeded and told us this peer does not speak
                // SwarmLLM, so it has no registry entry BY DECISION rather than
                // because Identify never ran. Without this arm the branch below
                // reads that absence as "disconnected too early" and re-dials it
                // forever — declining to register a foreign peer would otherwise
                // trade a wrong peer-list entry for an endless dial loop.
                tracing::trace!(%peer_id, "Not re-dialling a peer that does not speak SwarmLLM");
            } else {
                // Peer was never registered (connection died before Identify).
                // This typically happens during mDNS simultaneous-dial race.
                // Schedule a re-dial with random jitter to break symmetry.
                // No registry entry exists yet, so `closed_addr` is all we have;
                // when it is absent the dial falls back to whatever addresses
                // the behaviours know for this peer.
                let redial_addrs: Vec<libp2p::Multiaddr> = closed_addr.into_iter().collect();
                use std::hash::{Hash, Hasher};
                let mut hasher = std::collections::hash_map::DefaultHasher::new();
                peer_id.hash(&mut hasher);
                let jitter_ms = REDIAL_JITTER_MIN_MS + (hasher.finish() % REDIAL_JITTER_RANGE_MS);
                tracing::info!(
                    %peer_id,
                    addr_count = redial_addrs.len(),
                    jitter_ms,
                    "Scheduling re-dial after connection race"
                );
                self.try_enqueue_redial(peer_id, redial_addrs, jitter_ms);
            }
        } // end else (num_established == 0)
    }

    /// **Every outbound dial goes through here.** Refuses a peer Identify has
    /// already shown does not speak SwarmLLM, and names the site that asked so
    /// a log can say which path dialled.
    ///
    /// A choke point rather than a check per site, because the check-per-site
    /// version is what shipped and it did not hold: two of the six sites
    /// consulted `foreign_peers` and the node still opened 17 connections to
    /// three `openhydra` nodes in seven minutes — each one disconnected 43 ms
    /// later by the Identify gate, then dialled again. Finding *which* of the
    /// remaining four was responsible took a `-vv` capture and got no further
    /// than "not the ones with the guard"; making the wrong call
    /// unrepresentable is cheaper than the investigation, and
    /// `every_dial_goes_through_the_foreign_peer_gate` in
    /// `tests/repo_consistency.rs` keeps it that way.
    ///
    /// A dial with no peer id in it — an invite code, a bootstrap address —
    /// cannot be checked and is passed through. That is correct rather than a
    /// gap: those are addresses a user or operator named explicitly, and the
    /// Identify gate still refuses whatever answers.
    pub(super) fn dial_checked(
        &mut self,
        opts: impl Into<libp2p::swarm::dial_opts::DialOpts>,
        site: &'static str,
    ) -> Result<(), libp2p::swarm::DialError> {
        let opts = opts.into();
        if let Some(peer_id) = opts.get_peer_id() {
            // Never dial ourselves. A third party can hand our own address back
            // to us — PEX relays whatever its registry holds, and a relay
            // circuit terminating at us names us in its last `/p2p/` hop — and
            // the sender-side self-filter cannot help, because the sender is
            // someone else. libp2p refuses it, so this was never harmful, just
            // permanently wasteful: measured on a test node, 36 dials to its
            // own peer id in nine minutes, every one failing.
            //
            // `peer_cache::filter_storable` already drops self-routing
            // addresses for the cache; this is the same rule for every other
            // source, applied where all of them meet.
            if &peer_id == self.swarm.local_peer_id() {
                tracing::debug!(site, "Not dialling ourselves");
                return Ok(());
            }
            if self.foreign_peers.contains(&peer_id) {
                // DEBUG, not TRACE: this is the line that says the gate is
                // working, and looking for it at `-vv` and finding nothing is
                // how a run that proved nothing was briefly read as a run that
                // proved the gate useless.
                tracing::debug!(
                    %peer_id, site,
                    "Not dialling a peer that does not speak SwarmLLM"
                );
                return Ok(());
            }
        }
        // DEBUG, not TRACE, and for the same reason the refusal above is: the
        // only way to answer "was that connection our doing?" is to have said
        // so at a level a running node actually emits. Twice now an
        // investigation has turned on that question (#404, #405) and had to
        // rebuild to answer it.
        tracing::debug!(peer_id = ?opts.get_peer_id(), site, "Dialing");
        self.swarm.dial(opts)
    }

    /// Dial bootstrap / cached peers — **one dial per peer**, through the
    /// foreign-peer gate.
    ///
    /// Replaces `discovery::bootstrap_peers`, which dialled every address
    /// separately with a bare `swarm.dial(addr)`: no `PeerCondition`, so
    /// libp2p's per-peer dedup could not see it, and no foreign-peer gate,
    /// because it never reached [`Self::dial_checked`]. That is how a peer
    /// cached at two addresses collected three connections (gotcha #405), and
    /// why the #404 audit missed this site — the repo test looked for
    /// `self.swarm.dial(` and this was a free function taking `&mut Swarm`.
    ///
    /// `DisconnectedAndNotDialing` is libp2p's own default and the point of the
    /// change: it refuses a second dial while one is still in flight, which
    /// `Disconnected` does not.
    ///
    /// Handing one attempt every address is not slower than dialling each
    /// separately — libp2p races them concurrently inside the attempt and keeps
    /// only the first to connect.
    pub(super) fn dial_bootstrap_peers(&mut self, addrs: &[String]) -> usize {
        let mut dialed = 0;
        for entry in crate::network::discovery::plan_bootstrap_dials(addrs) {
            let opts = match entry.peer {
                Some(peer_id) => {
                    if self.swarm.is_connected(&peer_id) {
                        continue;
                    }
                    for addr in &entry.addrs {
                        self.swarm
                            .behaviour_mut()
                            .kademlia
                            .add_address(&peer_id, addr.clone());
                    }
                    libp2p::swarm::dial_opts::DialOpts::peer_id(peer_id)
                        .addresses(entry.addrs.clone())
                        .condition(
                            libp2p::swarm::dial_opts::PeerCondition::DisconnectedAndNotDialing,
                        )
                        .build()
                }
                // Names nobody, so there is nothing to dedup against and
                // nothing for the gate to check. Genuine first contact.
                None => match entry.addrs.first() {
                    Some(addr) => addr.clone().into(),
                    None => continue,
                },
            };
            match self.dial_checked(opts, "bootstrap") {
                Ok(()) => {
                    dialed += 1;
                    tracing::debug!(
                        peer_id = ?entry.peer,
                        addr_count = entry.addrs.len(),
                        "Dialing bootstrap peer"
                    );
                }
                Err(e) => {
                    tracing::debug!(peer_id = ?entry.peer, error = %e, "DIAG: bootstrap dial failed");
                }
            }
        }
        dialed
    }

    /// Enqueue a re-dial unless this peer already has one queued or the
    /// queue is at `MAX_PENDING_REDIAL`. Shared by both the active-pipeline
    /// and the unregistered-peer reconnect paths so the dedup+cap
    /// invariant lives in one place (R97).
    ///
    /// `addrs` may be empty: the dial is then made by peer id alone and the
    /// behaviours supply the addresses. An empty list is NOT a reason to skip
    /// the re-dial — that gate is what left two LAN peers mutually invisible
    /// (see the note in `handle_connection_closed`).
    fn try_enqueue_redial(
        &mut self,
        peer_id: libp2p::PeerId,
        addrs: Vec<libp2p::Multiaddr>,
        delay_ms: u64,
    ) {
        let already_queued = self
            .pending_redial
            .iter()
            .any(|(pid, _, _)| *pid == peer_id);
        if !already_queued && self.pending_redial.len() < MAX_PENDING_REDIAL {
            let scheduled = std::time::Instant::now() + std::time::Duration::from_millis(delay_ms);
            // Remember the addresses so a dial FAILURE can retry with the same
            // set; the entry is also what marks this peer as one we have been
            // connected to, and so may retry at all.
            if self.redial_attempts.len() >= MAX_REDIAL_TRACKED_PEERS {
                self.redial_attempts.clear();
            }
            self.redial_attempts
                .entry(peer_id)
                .or_insert_with(|| (addrs.clone(), 0));
            self.pending_redial.push((peer_id, addrs, scheduled));
        }
    }

    /// Count a request/response failure against `peer`, and close the
    /// connection once the peer has failed `MAX_CONSECUTIVE_RR_FAILURES` in a
    /// row without answering anything.
    ///
    /// A live connection is not evidence a peer is usable. libp2p holds a
    /// TCP+yamux connection open indefinitely while every request to it times
    /// out, and `connected_node_ids` — which the scheduler treats as the
    /// liveness oracle — is derived from exactly that. Closing the connection
    /// is what removes the peer from it, and `handle_connection_closed` then
    /// schedules the bounded-backoff re-dial, so a peer that recovers comes
    /// back on its own.
    ///
    /// A peer serving an active pipeline is never closed on this path. A long
    /// forward legitimately keeps a node busy for minutes, and tearing down the
    /// connection mid-pipeline would fail a request that was about to succeed —
    /// the same exemption `check_peer_health` already makes for staleness.
    pub(super) fn note_rr_failure(&mut self, peer: libp2p::PeerId) {
        let now = std::time::Instant::now();
        let entry = self.rr_failures.entry(peer).or_insert((0, now));
        entry.0 += 1;
        let failures = entry.0;
        let silent_for = now.saturating_duration_since(entry.1);
        if !should_close_unresponsive(failures, silent_for) {
            return;
        }
        if let Some(node_id) = self.peer_to_node.get(&peer).map(|r| r.clone()) {
            let in_active_pipeline = self.shared_state.active_pipelines.iter().any(|entry| {
                entry
                    .value()
                    .segments
                    .iter()
                    .any(|seg| seg.node_id == node_id)
            });
            if in_active_pipeline {
                tracing::debug!(
                    %peer,
                    failures,
                    "Peer is unresponsive but is serving an active pipeline — not closing"
                );
                return;
            }
        }
        self.rr_failures.remove(&peer);
        tracing::warn!(
            %peer,
            failures,
            "Peer has not answered {} requests in a row — closing the connection so it \
             stops being offered work; a re-dial is scheduled",
            failures
        );
        // Triggers ConnectionClosed → registry eviction + jittered re-dial.
        let _ = self.swarm.disconnect_peer_id(peer);
    }

    /// Re-enqueue a re-dial after a dial failure, on a bounded backoff.
    ///
    /// Only peers with a `redial_attempts` entry are retried — that entry is
    /// created by `try_enqueue_redial`, so it means "we were connected to this
    /// peer and lost it", not "some dial somewhere failed". Bootstrap, PEX and
    /// DHT dial targets are therefore untouched, which is what keeps this from
    /// becoming a dial storm.
    pub(super) fn schedule_redial_retry(&mut self, peer_id: libp2p::PeerId) {
        let Some((addrs, attempts)) = self.redial_attempts.get_mut(&peer_id) else {
            return;
        };
        let Some(delay_ms) = redial_backoff_ms(*attempts) else {
            tracing::debug!(
                %peer_id,
                attempts = *attempts,
                "Giving up re-dialling peer — treating it as departed"
            );
            self.redial_attempts.remove(&peer_id);
            return;
        };
        *attempts += 1;
        let attempt_no = *attempts;
        let addrs = addrs.clone();
        tracing::info!(
            %peer_id,
            attempt = attempt_no,
            max_attempts = MAX_REDIAL_ATTEMPTS,
            delay_ms,
            addr_count = addrs.len(),
            "Dial failed — retrying re-dial after backoff"
        );
        // Bypass `try_enqueue_redial`: it would re-create the attempts entry we
        // are counting down, and the dedup check has already been satisfied by
        // this peer's entry being drained before the dial was made.
        if self.pending_redial.len() < MAX_PENDING_REDIAL
            && !self
                .pending_redial
                .iter()
                .any(|(pid, _, _)| *pid == peer_id)
        {
            let scheduled = std::time::Instant::now() + std::time::Duration::from_millis(delay_ms);
            self.pending_redial.push((peer_id, addrs, scheduled));
        }
    }
}

#[cfg(test)]
mod redial_address_tests {
    use super::{redial_addresses, redial_backoff_ms, MAX_REDIAL_ATTEMPTS};
    use libp2p::Multiaddr;

    use super::{
        should_close_unresponsive, MAX_CONSECUTIVE_RR_FAILURES, RR_FAILURES_AFTER_SILENCE,
        RR_SILENCE_BEFORE_SHORT_RUN_COUNTS,
    };
    use std::time::Duration;

    /// The anchor — a healthy, critical relay — was measured reaching 5
    /// consecutive failures in normal operation. Neither route to closing may
    /// fire anywhere near that, whatever the elapsed time.
    #[test]
    fn a_healthy_peers_worst_measured_run_never_closes() {
        for secs in [0u64, 90, 600, 3600] {
            assert!(
                !should_close_unresponsive(5, Duration::from_secs(secs)),
                "5 failures must never close, even after {secs}s — that is the relay's worst run"
            );
        }
    }

    /// A burst of failures with no elapsed silence must not close: a busy peer
    /// fails in bursts while still answering in between, and any answer resets
    /// both signals.
    #[test]
    fn a_burst_without_silence_does_not_close() {
        assert!(!should_close_unresponsive(
            RR_FAILURES_AFTER_SILENCE,
            Duration::from_secs(1)
        ));
    }

    /// The point of the change: a dead return path is caught in about two
    /// minutes rather than ten.
    #[test]
    fn sustained_silence_closes_on_the_shorter_run() {
        assert!(should_close_unresponsive(
            RR_FAILURES_AFTER_SILENCE,
            RR_SILENCE_BEFORE_SHORT_RUN_COUNTS
        ));
        assert!(
            !should_close_unresponsive(
                RR_FAILURES_AFTER_SILENCE - 1,
                RR_SILENCE_BEFORE_SHORT_RUN_COUNTS
            ),
            "the short route still needs its own failure count"
        );
    }

    /// The long run remains a route on its own, for a peer that answers
    /// occasionally and so keeps resetting the clock without ever recovering.
    #[test]
    fn a_long_run_closes_regardless_of_elapsed_time() {
        assert!(should_close_unresponsive(
            MAX_CONSECUTIVE_RR_FAILURES,
            Duration::from_secs(0)
        ));
    }

    #[test]
    fn the_retry_schedule_is_bounded_and_increasing() {
        let mut prev = 0;
        for attempt in 0..MAX_REDIAL_ATTEMPTS {
            let d = redial_backoff_ms(attempt).expect("attempt within budget has a delay");
            assert!(d > prev, "delays must grow: {d} after {prev}");
            prev = d;
        }
        assert_eq!(
            redial_backoff_ms(MAX_REDIAL_ATTEMPTS),
            None,
            "the budget must run out — an unbounded retry is a dial storm"
        );
    }

    #[test]
    fn the_retry_window_outlasts_a_peer_reboot() {
        // The point of retrying at all: the first re-dial lands 2-5s after the
        // drop, when a rebooting peer is still down. If the whole schedule
        // expired before it came back, the retry would buy nothing.
        let total: u64 = (0..MAX_REDIAL_ATTEMPTS).filter_map(redial_backoff_ms).sum();
        assert!(
            total >= 300_000,
            "retry window {total}ms is too short to cover a restart"
        );
    }

    const LAN: &str = "/ip4/192.168.1.60/udp/8800/quic-v1";
    const LAN_TCP: &str = "/ip4/192.168.1.60/tcp/8810";
    const CIRCUIT: &str = "/ip4/212.132.104.177/tcp/8810/p2p/12D3KooWNisnVha2jYj1gqqY5WP82vNQbRhFtBcKzj4XrYmGEn8G/p2p-circuit";

    #[test]
    fn an_inbound_close_still_yields_addresses_to_dial() {
        // The regression, in one assertion. `closed_addr` is None for every
        // connection the PEER dialled, and the re-dial used to be gated on it —
        // so a peer that reached us inbound was never re-dialled at all. Two LAN
        // nodes 2 ms apart stayed mutually invisible for over two hours.
        let advertised = vec![LAN.to_string(), LAN_TCP.to_string()];
        let out = redial_addresses(None, &advertised);

        assert_eq!(out.len(), 2, "advertised addresses must be used: {out:?}");
        assert_eq!(out[0], LAN.parse::<Multiaddr>().unwrap());
    }

    #[test]
    fn the_closed_address_is_tried_first() {
        let closed: Multiaddr = LAN_TCP.parse().unwrap();
        let out = redial_addresses(Some(&closed), &[LAN.to_string()]);

        assert_eq!(out[0], closed, "the address that just worked leads");
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn a_relay_circuit_is_not_re_dialled_as_a_hint() {
        // Re-dialling the circuit we just lost rebuilds a relayed path to a peer
        // we may be able to reach directly — which is how the observed pair ended
        // up talking through an anchor in another country while 2 ms apart.
        let closed: Multiaddr = CIRCUIT.parse().unwrap();
        let out = redial_addresses(Some(&closed), &[LAN.to_string()]);

        assert_eq!(out, vec![LAN.parse::<Multiaddr>().unwrap()]);
    }

    #[test]
    fn no_hints_and_no_advertised_addresses_is_allowed() {
        // Empty is a legitimate answer: the caller dials by peer id and the
        // behaviours supply addresses. It must NOT be turned back into a
        // "skip the re-dial" signal.
        assert!(redial_addresses(None, &[]).is_empty());
    }

    #[test]
    fn duplicates_are_collapsed() {
        let closed: Multiaddr = LAN.parse().unwrap();
        let out = redial_addresses(Some(&closed), &[LAN.to_string(), LAN_TCP.to_string()]);

        assert_eq!(out.len(), 2, "the shared address is listed once: {out:?}");
    }
}
