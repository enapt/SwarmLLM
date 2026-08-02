//! Top-level swarm event dispatch.
//!
//! `handle_swarm_event` is the single match-on-`SwarmEvent` function that all
//! incoming libp2p activity flows through. It branches into the per-protocol
//! handlers in sibling modules — `requests.rs` for request_response,
//! `identify.rs` for Identify, `connections.rs` for ConnectionEstablished and
//! ConnectionClosed, `dht.rs` for DHT provider results — and handles the rest
//! inline: gossipsub Message (signed-seal verify + dispatcher push), gossipsub
//! Subscribed (replay buffered topic publishes), AutoNAT v2 client results
//! (reachable → ExternalAddrConfirmed; unreachable → `try_activate_relay`),
//! AutoNAT v2 server dial-back probes, UPnP mapping events, mDNS
//! Discovered/Expired (LAN peer count + peer_registry sync), Kademlia DHT
//! record verify, ExternalAddrConfirmed (Kademlia → Server mode), plus the
//! various connection-error edges.

use libp2p::gossipsub::{self, IdentTopic};
use libp2p::request_response;
use libp2p::swarm::SwarmEvent;
use libp2p::Multiaddr;

use crate::network::behaviour::SwarmBehaviourEvent;
use crate::network::helpers::swarm_event_name;
use crate::network::protocol::{self, SwarmRequest, SwarmResponse};
use crate::types::SwarmMessage;

use super::NetworkManager;

/// Minimum spacing between relay-activation attempts until one succeeds. Bounds
/// how often repeated AutoNAT-unreachable results (or the startup fallback tick)
/// can issue `listen_on` for a relay circuit.
const RELAY_RETRY_MIN_SECS: u64 = 20;

impl NetworkManager {
    pub(super) async fn handle_swarm_event(&mut self, event: SwarmEvent<SwarmBehaviourEvent>) {
        tracing::debug!(event_type = %swarm_event_name(&event), "DIAG: processing swarm event");
        match event {
            // ── GossipSub messages ──
            SwarmEvent::Behaviour(SwarmBehaviourEvent::Gossipsub(gossipsub::Event::Message {
                propagation_source,
                message,
                ..
            })) => {
                // SEC: All gossip MUST be signed + sealed. No unsigned fallback.
                let decoded = self
                    .shared_state
                    .gossip_sealer
                    .open_signed(&message.data)
                    .map_err(|e| {
                        tracing::warn!(
                            source = ?message.source,
                            error = %e,
                            "Rejecting unsigned/invalid gossip message"
                        );
                        e
                    })
                    .and_then(|(sender_pub, plaintext)| {
                        let msg = protocol::decode_message(&plaintext)?;
                        Ok((crate::types::NodeId(sender_pub), msg))
                    });

                match decoded {
                    Ok((sender_node_id, msg)) => {
                        // NET-M10: Reject gossip messages with timestamps older than 5 minutes
                        // or more than 30s in the future. Routed through the centralised
                        // one-sided helper (gotcha #44) so the .abs()-style replay-window
                        // doubling can't sneak back in.
                        let now_secs = chrono::Utc::now().timestamp() as u64;
                        const SKEW_TOLERANCE_SECS: u64 = 30;
                        const MAX_AGE_SECS: u64 = 300;
                        let fresh = |ts: u64, kind: &'static str| -> bool {
                            crate::daemon::dispatch::timestamp_fresh_one_sided(
                                ts,
                                now_secs,
                                MAX_AGE_SECS,
                                SKEW_TOLERANCE_SECS,
                                kind,
                            )
                        };
                        let too_old = match &msg {
                            SwarmMessage::HealthPing { timestamp, .. }
                            | SwarmMessage::HealthPong { timestamp, .. } => {
                                !fresh(*timestamp, "gossip_health")
                            }
                            SwarmMessage::ShardAnnounce(ann) => {
                                !fresh(ann.timestamp.timestamp() as u64, "gossip_shard_announce")
                            }
                            SwarmMessage::CreditGossip(gossip) => {
                                !fresh(gossip.timestamp.timestamp() as u64, "gossip_credit")
                            }
                            _ => false,
                        };
                        if too_old {
                            tracing::debug!(
                                source = %propagation_source,
                                "Dropping stale gossip message (>5 min old)"
                            );
                        } else {
                            tracing::debug!(
                                source = %propagation_source,
                                sender = %sender_node_id,
                                "Received signed GossipSub message"
                            );
                            let authed = crate::types::AuthenticatedMessage {
                                sender: Some(sender_node_id),
                                message: msg,
                            };
                            if let Err(e) = self.outbound_tx.try_send(authed) {
                                self.shared_state
                                    .metrics
                                    .channel_metrics
                                    .network_out
                                    .record_dropped();
                                tracing::warn!(error = %e, "Dispatcher backpressured, dropping gossipsub message");
                            } else {
                                self.shared_state
                                    .metrics
                                    .channel_metrics
                                    .network_out
                                    .record_sent();
                            }
                        }
                    }
                    Err(e) => {
                        tracing::debug!(error = %e, "Failed to decode/verify gossipsub message");
                    }
                }
            }

            // ── JSON request/response (control messages, shard transfers) ──
            SwarmEvent::Behaviour(SwarmBehaviourEvent::RequestResponse(
                request_response::Event::Message { peer, message, .. },
            )) => match message {
                request_response::Message::Request {
                    request, channel, ..
                } => {
                    let kind = match &request {
                        SwarmRequest::Message(_) => "message",
                        SwarmRequest::ShardTransfer(_) => "shard",
                        SwarmRequest::TensorPayload(_) => "tensor",
                        SwarmRequest::PrefixKvFetch(_) => "prefix_kv_fetch",
                        SwarmRequest::RelayedTensor(_) => "relayed_tensor",
                    };
                    tracing::info!(%peer, kind, "Received request");
                    self.handle_request(peer, request, channel).await;
                }
                request_response::Message::Response {
                    request_id,
                    response,
                } => {
                    let kind = match &response {
                        SwarmResponse::Message(_) => "message",
                        SwarmResponse::ShardData(_) => "shard",
                        SwarmResponse::Ack => "ack",
                        SwarmResponse::TensorPayload(_) => "tensor",
                        SwarmResponse::PrefixKvData(_) => "prefix_kv_data",
                    };
                    let was_tensor = self.pending_tensor_outbound.contains_key(&request_id);
                    tracing::info!(
                        %peer,
                        kind,
                        ?request_id,
                        was_tensor_forward = was_tensor,
                        pending_tensor_out = self.pending_tensor_outbound.len(),
                        "DIAG: received response"
                    );
                    // Clean up tensor outbound tracking (response received = not a failure)
                    self.pending_tensor_outbound.remove(&request_id);
                    self.pending_tensor_result_outbound.remove(&request_id);
                    // A Response proves the peer received the request — the send was
                    // NOT silently dropped, so the RR_ACK_TIMEOUT (10s) sweep must no
                    // longer apply. Without this, a remote-generate whose first token
                    // legitimately takes longer than 10s (a cold model load, a slow
                    // CPU peer, a large prompt) has its streaming channel closed at
                    // 10s and is reported as "peer never acknowledged" — even though
                    // the peer DID acknowledge and is working. From here the proper
                    // FIRST_TOKEN_TIMEOUT (120s) governs. streaming_token_txs is left
                    // intact so tokens keep flowing.
                    self.pending_rr_observability.remove(&request_id);
                    self.handle_response(peer, request_id, response).await;
                }
            },

            // ── Request/response failures ──
            SwarmEvent::Behaviour(SwarmBehaviourEvent::RequestResponse(
                request_response::Event::OutboundFailure {
                    peer,
                    request_id,
                    error,
                    ..
                },
            )) => {
                tracing::warn!(
                    %peer,
                    ?request_id,
                    %error,
                    is_connected = self.swarm.is_connected(&peer),
                    pending_tensor_out = self.pending_tensor_outbound.len(),
                    pending_channels = self.pending_tensor_channels.len(),
                    "DIAG: OutboundFailure"
                );
                // Check if this was a pending tensor forward — notify the pipeline
                if let Some((inference_uuid, sent_at, _target, _, _)) =
                    self.pending_tensor_outbound.remove(&request_id)
                {
                    let age_ms = sent_at.elapsed().as_millis();
                    tracing::error!(
                        %peer,
                        inference_request_id = %inference_uuid,
                        %error,
                        age_ms,
                        "Tensor forward OutboundFailure — notifying pipeline"
                    );
                    // Send an error LayerResult so the pipeline can failover immediately
                    self.fail_tensor_forward(
                        inference_uuid,
                        &peer,
                        format!("OutboundFailure: {error}"),
                    );
                }
                // Log result-send fallback failures with UUID context.
                // We can't notify the upstream requester from here — their pipeline
                // has its own timeout via their pending_tensor_outbound watchdog.
                if let Some((result_uuid, _)) =
                    self.pending_tensor_result_outbound.remove(&request_id)
                {
                    // Mirrors the rr-message branch below: this is the best-effort
                    // result-fallback path, and the upstream pipeline's own
                    // pending_tensor_outbound watchdog handles the user-visible
                    // failure. warn! is enough; error! would page on every retry.
                    tracing::warn!(
                        %peer,
                        inference_request_id = %result_uuid,
                        %error,
                        "Tensor result fallback OutboundFailure — upstream will timeout"
                    );
                }
                if let Some((label, _, delivery_uuid)) =
                    self.pending_rr_observability.remove(&request_id)
                {
                    tracing::warn!(
                        %peer,
                        label,
                        %error,
                        "rr-message OutboundFailure — upstream will handle via its own timeout"
                    );
                    // Close the streaming caller's channel immediately so it
                    // sees the failure now, not after FIRST_TOKEN_TIMEOUT.
                    if let Some(uuid) = delivery_uuid {
                        self.shared_state.streaming_token_txs.remove(&uuid);
                    }
                }
                // Item 8 Phase 2: unblock a pending prefix-KV fetch on failure.
                if let Some(uuid) = self.pending_prefix_kv_outbound.remove(&request_id) {
                    if let Some((_, tx)) = self.shared_state.pending_prefix_kv_fetches.remove(&uuid)
                    {
                        let _ = tx.send(None);
                    }
                }
                // Check if this was a pending shard download request
                if let Some((_peer_id, shard_id)) = self.pending_shard_requests.remove(&request_id)
                {
                    let progress = self
                        .shard_download_progress
                        .get(&shard_id)
                        .copied()
                        .unwrap_or(0);
                    // Self-healing path — retry_shard_or_fallback tries up to
                    // MAX_P2P_RETRIES other peers and only then falls back to HF.
                    // The exhausted-all-peers case surfaces its own error-level
                    // event in shard_transfer.rs, so warn! here keeps normal
                    // single-peer hiccups out of the operator triage queue.
                    tracing::warn!(
                        %peer,
                        model = %shard_id.model_id,
                        shard_index = shard_id.index,
                        %error,
                        bytes_downloaded = progress,
                        "DIAG: shard download OutboundFailure — attempting peer failover"
                    );
                    self.retry_shard_or_fallback(shard_id, peer, &format!("{error}"));
                }
            }
            SwarmEvent::Behaviour(SwarmBehaviourEvent::RequestResponse(
                request_response::Event::InboundFailure {
                    peer,
                    request_id,
                    error,
                    ..
                },
            )) => {
                // Note: pending_tensor_channels is keyed by Uuid (from the parsed
                // message), not InboundRequestId — we can't directly remove the entry
                // here. The stale timeout cleanup (every 30s) handles orphaned channels.
                tracing::warn!(
                    %peer,
                    ?request_id,
                    %error,
                    pending_channels = self.pending_tensor_channels.len(),
                    "DIAG: InboundFailure — response send may have failed, stale cleanup will reclaim"
                );
            }
            SwarmEvent::Behaviour(SwarmBehaviourEvent::RequestResponse(
                request_response::Event::ResponseSent {
                    peer, request_id, ..
                },
            )) => {
                tracing::info!(%peer, ?request_id, "DIAG: ResponseSent event — response written to wire");
            }

            // ── GossipSub peer subscribed — flush matching buffered messages (NET-I4) ──
            //
            // Only the just-subscribed topic is eligible for replay — a
            // Subscribed{peer_id, topic=X} event tells us the mesh now has
            // at least one peer on topic X, but says nothing about topic Y.
            // Before this filter, ANY Subscribed event iterated the whole
            // buffer and called publish() on every entry; publish() would
            // still return Err for topics with no subscribers (gossipsub
            // routes correctly, so this wasn't an info leak), but the
            // entry got re-buffered, wasting a full O(buffer) pass per
            // Subscribed event on a multi-topic mesh.
            SwarmEvent::Behaviour(SwarmBehaviourEvent::Gossipsub(
                gossipsub::Event::Subscribed { peer_id, topic },
            )) => {
                tracing::debug!(%peer_id, %topic, "Peer subscribed to topic");
                let subscribed_topic_str = topic.to_string();
                let has_match = self
                    .buffered_gossip
                    .iter()
                    .any(|(t, _)| t == &subscribed_topic_str);
                if has_match {
                    let mut remaining = Vec::with_capacity(self.buffered_gossip.len());
                    let mut replayed = 0;
                    for (topic_str, data) in std::mem::take(&mut self.buffered_gossip) {
                        if topic_str != subscribed_topic_str {
                            remaining.push((topic_str, data));
                            continue;
                        }
                        let gossip_topic = IdentTopic::new(&topic_str);
                        match self
                            .swarm
                            .behaviour_mut()
                            .gossipsub
                            .publish(gossip_topic, data.clone())
                        {
                            Ok(_) => replayed += 1,
                            Err(_) => remaining.push((topic_str, data)),
                        }
                    }
                    self.buffered_gossip = remaining;
                    if replayed > 0 {
                        tracing::info!(
                            topic = %subscribed_topic_str,
                            count = replayed,
                            "Replayed buffered GossipSub messages for newly-subscribed topic"
                        );
                    }
                }
            }

            // ── Relay server events ──
            SwarmEvent::Behaviour(SwarmBehaviourEvent::RelayServer(event)) => {
                crate::network::relay::handle_relay_server_event(event, &self.shared_state);
            }

            // ── DCUtR — hole-punch outcomes ──
            //
            // These were previously swallowed by the catch-all arm, which is
            // how a *structurally disabled* DCUtR (the per-peer connection cap
            // denied the direct connection a hole punch needs) went unnoticed
            // through several releases: NAT traversal is the load-bearing
            // mechanism of this project and it emitted nothing either way.
            // Success/failure is logged at INFO so a support log shows whether a
            // node ever escapes the relay, and counted so
            // `GET /api/admin/diagnostics` can report it without log scraping.
            SwarmEvent::Behaviour(SwarmBehaviourEvent::Dcutr(event)) => {
                use std::sync::atomic::Ordering;
                match &event.result {
                    Ok(connection_id) => {
                        self.shared_state
                            .hole_punch_successes
                            .fetch_add(1, Ordering::Relaxed);
                        tracing::info!(
                            peer = %event.remote_peer_id,
                            ?connection_id,
                            "DIAG: hole punch succeeded — upgraded to a direct connection"
                        );
                    }
                    Err(e) => {
                        self.shared_state
                            .hole_punch_failures
                            .fetch_add(1, Ordering::Relaxed);
                        // Expected against symmetric NAT / CGNAT, where hole
                        // punching cannot work at all — the relay carries the
                        // traffic instead. Not an error condition on its own.
                        tracing::info!(
                            peer = %event.remote_peer_id,
                            error = %e,
                            "DIAG: hole punch failed — staying on the relay path"
                        );
                    }
                }
            }

            // ── AutoNAT v2 client — reachability test results for OUR addresses ──
            SwarmEvent::Behaviour(SwarmBehaviourEvent::AutonatClient(event)) => {
                let tested_addr = event.tested_addr;
                let server = event.server;
                match event.result {
                    Ok(()) => {
                        // Address is reachable from the internet. The v2 client has
                        // already emitted ToSwarm::ExternalAddrConfirmed for it
                        // (caught by the ExternalAddrConfirmed arm below → Kademlia
                        // Server mode + refresh_listen_multiaddrs).
                        tracing::info!(%tested_addr, %server, "AutoNAT: address confirmed reachable (public)");
                        if let Ok(mut stats) = self.shared_state.metrics.node_stats.try_write() {
                            stats.nat_status = Some("Public".to_string());
                        }
                    }
                    Err(e) => {
                        // Address is NOT reachable — we're behind NAT/CGNAT for it.
                        // Reserve a relay so peers can still reach us. (v2 fixes
                        // v1's false-"Public" that silently skipped this.)
                        tracing::info!(%tested_addr, %server, error = %e, "AutoNAT: address not reachable (private) — activating relay");
                        if let Ok(mut stats) = self.shared_state.metrics.node_stats.try_write() {
                            stats.nat_status = Some("Private (relay)".to_string());
                        }
                        self.try_activate_relay("AutoNAT reported our address unreachable");
                    }
                }
            }

            // ── AutoNAT v2 server — we answered another node's dial-back probe ──
            SwarmEvent::Behaviour(SwarmBehaviourEvent::AutonatServer(event)) => {
                tracing::debug!(?event, "AutoNAT server: served a dial-back probe");
            }

            // ── UPnP gateway port-mapping ──
            SwarmEvent::Behaviour(SwarmBehaviourEvent::Upnp(event)) => {
                use libp2p::upnp::Event as UpnpEvent;
                match event {
                    UpnpEvent::NewExternalAddr(addr) => {
                        // The UPnP behaviour has already confirmed this address
                        // with the swarm (ExternalAddrConfirmed), which our
                        // handler above catches. Refresh explicitly too so the
                        // invite-code snapshot picks up the public address even
                        // if the confirm event ordering differs, and let the
                        // user know their node is now internet-reachable.
                        tracing::info!(%addr, "UPnP mapped a public address — node is now reachable from the internet");
                        if let Ok(mut stats) = self.shared_state.metrics.node_stats.try_write() {
                            stats.nat_status = Some("Public (UPnP-mapped)".to_string());
                        }
                        self.refresh_listen_multiaddrs();
                        self.shared_state.emit_activity(
                            crate::daemon::state::ActivityEvent::new(
                                "network",
                                "upnp_mapped",
                                format!("Your router opened a public address — this node is now reachable across the internet ({addr})"),
                            )
                            .with_detail_str(addr.to_string())
                            .with_toast("success", 6000),
                        );
                    }
                    UpnpEvent::ExpiredExternalAddr(addr) => {
                        tracing::warn!(%addr, "UPnP external address mapping expired");
                        self.refresh_listen_multiaddrs();
                    }
                    UpnpEvent::GatewayNotFound => {
                        tracing::info!(
                            "UPnP: no IGD gateway found — router has UPnP disabled or none is present. \
                             Internet peers will need a relay or a manually port-forwarded address."
                        );
                    }
                    UpnpEvent::NonRoutableGateway => {
                        // The gateway exists but is itself behind another NAT —
                        // the classic carrier-grade NAT (CGNAT) signature. Port
                        // mapping cannot make this node publicly reachable.
                        tracing::warn!(
                            "UPnP: gateway is not routable (behind carrier-grade NAT). This node cannot be \
                             reached directly from the internet — it will need a public relay to receive inbound connections."
                        );
                        if let Ok(mut stats) = self.shared_state.metrics.node_stats.try_write() {
                            stats.nat_status = Some("Private (CGNAT — relay required)".to_string());
                        }
                    }
                }
            }

            // ── Identify ──
            SwarmEvent::Behaviour(SwarmBehaviourEvent::Identify(
                libp2p::identify::Event::Received {
                    peer_id,
                    info,
                    connection_id,
                },
            )) => {
                self.handle_identify_received(peer_id, info, connection_id);
            }

            // ── Kademlia ──
            SwarmEvent::Behaviour(SwarmBehaviourEvent::Kademlia(
                libp2p::kad::Event::OutboundQueryProgressed { id, result, .. },
            )) => {
                use libp2p::kad::QueryResult;
                match result {
                    QueryResult::GetRecord(Ok(libp2p::kad::GetRecordOk::FoundRecord(
                        peer_record,
                    ))) => {
                        // Verify Ed25519 signature on DHT records before trusting
                        match crate::network::discovery::verify_dht_value(&peer_record.record.value)
                        {
                            Ok((pubkey, payload)) => {
                                tracing::debug!(
                                    key = ?peer_record.record.key,
                                    signer = %hex::encode(&pubkey[..8]),
                                    payload_len = payload.len(),
                                    "DHT record verified"
                                );
                                // Process verified payload: deserialize NodeCapability
                                // and update peer registry with the advertised capabilities.
                                let key_bytes = peer_record.record.key.as_ref();
                                let key_str = String::from_utf8_lossy(key_bytes);
                                if key_str.starts_with("/swarm/node/") {
                                    if let Ok(cap) = serde_json::from_slice::<
                                        crate::types::NodeCapability,
                                    >(payload)
                                    {
                                        let node_id = crate::types::NodeId(pubkey);
                                        if let Some(mut entry) =
                                            self.shared_state.peer_registry.get_mut(&node_id)
                                        {
                                            entry.capability = Some(cap);
                                        }
                                    }
                                }
                            }
                            Err(_) => {
                                tracing::warn!(
                                    key = ?peer_record.record.key,
                                    "DHT record failed signature verification — ignoring"
                                );
                            }
                        }
                    }
                    // S5: DHT provider query results — merge discovered holders
                    // into the bounded shard_holders cache.
                    QueryResult::GetProviders(Ok(
                        libp2p::kad::GetProvidersOk::FoundProviders { providers, .. },
                    )) => {
                        self.handle_dht_providers_found(id, &providers);
                    }
                    QueryResult::GetProviders(Ok(
                        libp2p::kad::GetProvidersOk::FinishedWithNoAdditionalRecord { .. },
                    )) => {
                        // Query finished — clean up tracking
                        self.pending_provider_queries.remove(&id);
                    }
                    QueryResult::GetProviders(Err(ref e)) => {
                        tracing::debug!(error = ?e, "DHT provider query failed — cleaning up");
                        self.pending_provider_queries.remove(&id);
                    }
                    _ => {
                        tracing::debug!(?result, "Kademlia query progressed");
                    }
                }
            }

            // ── mDNS ──
            SwarmEvent::Behaviour(SwarmBehaviourEvent::Mdns(libp2p::mdns::Event::Discovered(
                peers,
            ))) => {
                // Group the batch by peer so a peer advertising several
                // interfaces — the norm on WSL2/Docker: loopback + LAN +
                // NAT-gateway (`10.255.255.254`) + link-local (`169.254/16`) —
                // gets ONE dial carrying all of its addresses, not a separate
                // concurrent dial per address. libp2p tries the addresses within
                // a single dial and keeps exactly one connection, aborting the
                // rest. The previous per-address loop fired N concurrent dials
                // to the same peer; under `max_established_per_peer = 1` the
                // extras are denied inconsistently, leaving a stale connection
                // entry that silently breaks request_response routing — a lost
                // tensor forward, and the residual of the NET-I5 churn bug.
                //
                // Do NOT add mDNS addresses to Kademlia (its periodic refresh
                // would re-dial them all every 30s, recreating the churn);
                // identify handles address exchange once connected.
                let mut by_peer: std::collections::HashMap<libp2p::PeerId, Vec<libp2p::Multiaddr>> =
                    std::collections::HashMap::new();
                for (peer_id, addr) in peers {
                    by_peer.entry(peer_id).or_default().push(addr);
                }
                // Deterministic dialer: only the node with the smaller PeerId
                // initiates the mDNS auto-dial; the larger-PeerId node waits to
                // be dialed. Both sides run this rule, so exactly ONE of the pair
                // dials — eliminating the bidirectional simultaneous-dial race
                // where A→B and B→A both connect, `max_established_per_peer = 1`
                // denies one on each side inconsistently, and the survivor can be
                // a half-open connection that silently breaks request_response
                // routing (a lost tensor forward; the core of the NET-I5 churn).
                // Only affects LAN mDNS auto-discovery — bootstrap/DHT/relay dials
                // are unchanged. The larger-PeerId side is still reached: the
                // smaller side dials it, and the redial-with-jitter tick recovers
                // if that dial is lost.
                let local_bytes = self.swarm.local_peer_id().to_bytes();
                for (peer_id, addrs) in by_peer {
                    let we_dial = local_bytes < peer_id.to_bytes();
                    if !self.swarm.is_connected(&peer_id) && we_dial {
                        tracing::info!(
                            %peer_id, addr_count = addrs.len(),
                            "LAN peer discovered automatically — no configuration needed"
                        );
                        // Use Disconnected (not DisconnectedAndNotDialing) so mDNS
                        // can override a failing bootstrap dial attempt. Without this,
                        // a peer that restarts with a new identity can't reconnect
                        // because the stale bootstrap dial blocks mDNS.
                        let opts = libp2p::swarm::dial_opts::DialOpts::peer_id(peer_id)
                            .condition(libp2p::swarm::dial_opts::PeerCondition::Disconnected)
                            .addresses(addrs)
                            .build();
                        if let Err(e) = self.swarm.dial(opts) {
                            tracing::debug!(%peer_id, error = %e, "mDNS: dial skipped");
                        }
                    } else {
                        tracing::debug!(%peer_id, we_dial, "mDNS: not dialing (already connected or peer is designated dialer)");
                    }
                    // Mark as LAN peer if we can derive their NodeId
                    if let Some(node_id) = self.peer_to_node.get(&peer_id) {
                        if let Some(mut peer) = self.shared_state.peer_registry.get_mut(&*node_id) {
                            if !peer.is_lan_peer {
                                peer.is_lan_peer = true;
                                drop(peer);
                                // Increment LAN peer count and notify via unified activity event
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
                                tracing::info!(lan_peers = count, message = %msg, "LAN peer discovery update");
                                self.shared_state.emit_activity(
                                    crate::daemon::state::ActivityEvent::new(
                                        "network",
                                        "lan_peer_discovered",
                                        msg,
                                    )
                                    .with_detail_num(count as i64)
                                    .with_toast("success", 8000),
                                );
                            }
                        }
                    }
                }
            }

            SwarmEvent::Behaviour(SwarmBehaviourEvent::Mdns(libp2p::mdns::Event::Expired(
                peers,
            ))) => {
                for (peer_id, _addr) in peers {
                    tracing::debug!(%peer_id, "mDNS: peer expired");
                    // Decrement LAN peer count if this was a tracked LAN peer
                    if let Some(node_id) = self.peer_to_node.get(&peer_id) {
                        if let Some(mut peer) = self.shared_state.peer_registry.get_mut(&*node_id) {
                            if peer.is_lan_peer {
                                peer.is_lan_peer = false;
                                drop(peer);
                                let _ = self.shared_state.lan_peer_count.fetch_update(
                                    std::sync::atomic::Ordering::Relaxed,
                                    std::sync::atomic::Ordering::Relaxed,
                                    |v| v.checked_sub(1),
                                );
                            }
                        }
                    }
                }
            }

            SwarmEvent::ConnectionEstablished {
                peer_id,
                connection_id,
                num_established,
                endpoint,
                ..
            } => {
                self.handle_connection_established(
                    peer_id,
                    connection_id,
                    num_established,
                    &endpoint,
                );
            }

            SwarmEvent::ConnectionClosed {
                peer_id,
                connection_id,
                cause,
                num_established,
                ..
            } => {
                self.handle_connection_closed(peer_id, connection_id, cause, num_established);
            }

            SwarmEvent::NewListenAddr { address, .. } => {
                tracing::info!(%address, "New listen address");
                self.refresh_listen_multiaddrs();
            }

            SwarmEvent::ExpiredListenAddr { address, .. } => {
                tracing::info!(%address, "Listen address expired");
                if crate::network::relay::is_relay_circuit_addr(&address) {
                    self.note_relay_circuit_lost("reservation expired");
                }
                self.refresh_listen_multiaddrs();
            }

            SwarmEvent::ListenerClosed { addresses, .. } => {
                tracing::debug!(?addresses, "Listener closed");
                if addresses
                    .iter()
                    .any(crate::network::relay::is_relay_circuit_addr)
                {
                    self.note_relay_circuit_lost("listener closed");
                }
                self.refresh_listen_multiaddrs();
            }

            // NET-I7: Switch Kademlia to Server mode when external address is confirmed
            SwarmEvent::ExternalAddrConfirmed { address } => {
                tracing::info!(%address, "External address confirmed — switching Kademlia to Server mode");
                self.swarm
                    .behaviour_mut()
                    .kademlia
                    .set_mode(Some(libp2p::kad::Mode::Server));
                self.refresh_listen_multiaddrs();
            }

            SwarmEvent::OutgoingConnectionError { peer_id, error, .. } => {
                tracing::debug!(
                    ?peer_id, %error,
                    "Outgoing connection failed"
                );
                // A dial that fails raises THIS event, not `ConnectionClosed`,
                // so the re-dial scheduled when the peer dropped is not
                // re-enqueued by anything. One attempt is not enough: it lands
                // 2-5s after the drop, which is exactly when a rebooting peer is
                // still down. Retry on a bounded backoff.
                if let Some(peer_id) = peer_id {
                    self.schedule_redial_retry(peer_id);
                }
            }

            SwarmEvent::IncomingConnectionError { error, .. } => {
                tracing::debug!(
                    %error,
                    "Incoming connection failed"
                );
            }

            other => {
                tracing::trace!(?other, "Unhandled swarm event");
            }
        }
    }

    /// Drop the `relay_activated` latch after the relay circuit we were reachable
    /// through is lost (relay peer restarted, connection dropped, or reservation
    /// expired). This re-arms the recovery paths: the liveness-tick fallback
    /// (`mod.rs`) re-checks reachability every tick and, seeing the latch clear +
    /// no internet-reachable address, calls `try_activate_relay` again — which
    /// re-reserves once a relay peer is reachable (bootstrap re-dial handles the
    /// reconnect). Without this reset the one-shot latch stays set forever and a
    /// NAT'd node never regains internet reachability until a manual restart
    /// (found live 2026-07-23: an anchor restart mid-test stranded the test node).
    /// `last_relay_attempt` is deliberately left intact so a flapping relay is
    /// still rate-limited by `RELAY_RETRY_MIN_SECS`.
    pub(super) fn note_relay_circuit_lost(&mut self, reason: &str) {
        if !self.relay_activated {
            return; // we had no relay reservation to lose
        }
        self.relay_activated = false;
        tracing::info!(
            reason,
            "Relay circuit lost — will re-reserve on the next liveness tick"
        );
    }

    /// Reserve a relay circuit on a bootstrap peer so a node that isn't directly
    /// reachable (NAT/CGNAT) can still receive inbound connections.
    ///
    /// Called from two places: the AutoNAT v2 client's "address not reachable"
    /// result (primary), and a startup fallback timer in the run loop (in case
    /// AutoNAT never gets a conclusive answer — e.g. no AutoNAT servers reachable).
    /// Idempotent: latches `relay_activated` once a relay listen succeeds, and
    /// rate-limits attempts to at most once per `RELAY_RETRY_MIN_SECS` until then,
    /// so repeated triggers don't spam `listen_on`. No-op when `auto_relay` is off
    /// or there are no bootstrap peers to relay through.
    pub(super) fn try_activate_relay(&mut self, reason: &str) {
        if self.relay_activated || !self.shared_state.config.network.auto_relay {
            return;
        }
        let now = std::time::Instant::now();
        if let Some(last) = self.last_relay_attempt {
            if now.duration_since(last).as_secs() < RELAY_RETRY_MIN_SECS {
                return;
            }
        }
        let bootstrap_addrs = self.shared_state.config.network.bootstrap_peers.clone();
        if bootstrap_addrs.is_empty() {
            return; // nothing publicly-reachable to relay through
        }
        self.last_relay_attempt = Some(now);
        tracing::info!(
            reason,
            "Activating relay listener (node not directly reachable)"
        );
        let mut relayed = false;
        for addr_str in &bootstrap_addrs {
            let Ok(maddr) = addr_str.parse::<Multiaddr>() else {
                continue;
            };
            let Some(relay_pid) = maddr.iter().find_map(|p| match p {
                libp2p::multiaddr::Protocol::P2p(pid) => Some(pid),
                _ => None,
            }) else {
                continue;
            };
            // Relay-listen address without the trailing /p2p component.
            let base: Multiaddr = maddr
                .iter()
                .take_while(|p| !matches!(p, libp2p::multiaddr::Protocol::P2p(_)))
                .collect();
            let relay_addr = crate::network::relay::relay_listen_addr(&relay_pid, &base);
            match self.swarm.listen_on(relay_addr.clone()) {
                Ok(_) => {
                    tracing::info!(relay_peer = %relay_pid, %relay_addr, "Relay listen activated");
                    relayed = true;
                    break; // one relay is sufficient
                }
                Err(e) => {
                    tracing::debug!(relay_peer = %relay_pid, error = %e, "Failed to listen via relay peer");
                }
            }
        }
        if relayed {
            self.relay_activated = true;
        } else {
            tracing::warn!(
                "No relay peers accepted yet — will retry; node may be unreachable meanwhile"
            );
        }
    }

    /// Rebuild `state.listen_multiaddrs` from the swarm's current listeners
    /// **unioned with its confirmed external addresses**.
    ///
    /// The listeners are the bound sockets — on a NAT'd node those are private
    /// LAN IPs (`192.168.x`, `10.x`), useless to a remote dialer. The confirmed
    /// external addresses (UPnP-mapped, AutoNAT-confirmed, relay-circuit, or
    /// manually declared via `network.external_address`) are the ones an
    /// internet peer can actually reach. Without the union the invite code
    /// silently ships a LAN-only address that works on the LAN and dies over
    /// the internet — the exact failure a fresh node hits.
    ///
    /// Each entry is appended with `/p2p/<local_peer_id>` (when it doesn't
    /// already terminate in a peer id) so a remote dialer can both connect and
    /// verify the target identity. We deliberately keep LAN + Tailscale CGN
    /// (100.64.0.0/10) addresses too — those are reachable by a node on the
    /// same overlay. Only loopback / unspecified / link-local / metadata
    /// addresses are dropped: nothing a remote dialer could productively use.
    pub(super) fn refresh_listen_multiaddrs(&self) {
        let local_peer_id = *self.swarm.local_peer_id();
        let candidates = self
            .swarm
            .listeners()
            .cloned()
            .chain(self.swarm.external_addresses().cloned());
        let addrs = build_reachable_multiaddr_list(candidates, local_peer_id);

        // NETWORKING_PLAN Phase 3 — re-evaluate whether we can donate relay
        // capacity. Derived from the swarm's CONFIRMED external addresses only
        // (UPnP-mapped / AutoNAT-confirmed / manually declared), never from a
        // bound listener: binding a socket says nothing about whether anyone
        // outside can reach it.
        let publicly_reachable = self
            .swarm
            .external_addresses()
            .any(multiaddr_is_public_relay_candidate);
        let was = self
            .shared_state
            .publicly_reachable
            .swap(publicly_reachable, std::sync::atomic::Ordering::Relaxed);
        if was != publicly_reachable {
            tracing::info!(
                publicly_reachable,
                relay_forwarding = self.shared_state.relay_forwarding_enabled(),
                "NETWORKING_PLAN: public reachability changed — relay-donation status updated"
            );
        }

        self.shared_state
            .listen_multiaddrs
            .store(std::sync::Arc::new(addrs));
    }
}

/// Whether an address makes this node a viable **relay for others**.
///
/// Stricter than `addr_is_remotely_reachable` (which keeps LAN + CGNAT so peers
/// on the same overlay can connect) and stricter than
/// `pool::invite::any_internet_reachable` (which counts `/p2p-circuit`, correct
/// for an invite code but wrong here). To forward for others a node needs an
/// address strangers can dial directly:
///
/// - a **global** IPv4/IPv6 address, or a DNS name — yes;
/// - `/p2p-circuit` — no: reachable only *through* someone else's relay means
///   this node is itself NAT'd and cannot forward;
/// - RFC1918 / CGNAT / ULA / loopback / link-local — no.
fn multiaddr_is_public_relay_candidate(addr: &libp2p::Multiaddr) -> bool {
    use libp2p::multiaddr::Protocol;
    let mut public = false;
    for proto in addr.iter() {
        match proto {
            Protocol::P2pCircuit => return false,
            Protocol::Ip4(ip) => {
                if !ip.is_private()
                    && !ip.is_loopback()
                    && !ip.is_link_local()
                    && !ip.is_broadcast()
                    && !ip.is_documentation()
                    && !ip.is_unspecified()
                    // CGNAT 100.64.0.0/10 — a carrier-NAT address is not
                    // dialable from outside the carrier's network.
                    && !(ip.octets()[0] == 100 && (64..128).contains(&ip.octets()[1]))
                {
                    public = true;
                }
            }
            Protocol::Ip6(ip) => {
                // Unique-local (fc00::/7) and link-local (fe80::/10) are not
                // globally dialable.
                let seg = ip.segments()[0];
                if !ip.is_loopback()
                    && !ip.is_unspecified()
                    && (seg & 0xfe00) != 0xfc00
                    && (seg & 0xffc0) != 0xfe80
                {
                    public = true;
                }
            }
            Protocol::Dns(_) | Protocol::Dns4(_) | Protocol::Dns6(_) | Protocol::Dnsaddr(_) => {
                public = true;
            }
            _ => {}
        }
    }
    public
}

/// Build the deduped, `/p2p`-suffixed, remotely-reachable multiaddr string list
/// from an iterator of candidate addresses (listeners ∪ external addresses).
/// Extracted from `refresh_listen_multiaddrs` so the filter/suffix/dedup logic
/// is unit-testable without a live swarm.
fn build_reachable_multiaddr_list(
    candidates: impl Iterator<Item = Multiaddr>,
    local_peer_id: libp2p::PeerId,
) -> Vec<String> {
    let mut addrs: Vec<String> = candidates
        .filter(addr_is_remotely_reachable)
        .map(|addr| ensure_p2p_suffix(addr, local_peer_id).to_string())
        .collect();
    addrs.sort();
    addrs.dedup();
    addrs
}

/// Append `/p2p/<local_peer_id>` to `addr` unless it already terminates in a
/// peer-id component. A relay-circuit listener ends in `/p2p-circuit`, so it
/// correctly gets our id appended (`.../p2p-circuit/p2p/<us>` — the canonical
/// relayed dial form). A bare `/ip4/.../tcp/port` external address gets our id
/// appended. An address a user already suffixed is returned verbatim so we
/// never double-append.
fn ensure_p2p_suffix(addr: Multiaddr, local_peer_id: libp2p::PeerId) -> Multiaddr {
    if matches!(
        addr.iter().last(),
        Some(libp2p::multiaddr::Protocol::P2p(_))
    ) {
        addr
    } else {
        addr.with(libp2p::multiaddr::Protocol::P2p(local_peer_id))
    }
}

/// Decide whether an address is something a remote peer could plausibly dial.
/// Excludes loopback, unspecified, IPv4 link-local, and the AWS/GCP IMDS
/// address; keeps everything else (LAN, CGN/Tailscale, public).
///
/// Shared with `network::peer_cache::filter_dialable` — the same question is
/// asked of our own listen addresses before we advertise them and of a peer's
/// advertised addresses before we cache and re-dial them. Keep it one
/// predicate; a cache that admits addresses the advertiser would have
/// suppressed just relocates the problem.
pub(crate) fn addr_is_remotely_reachable(addr: &Multiaddr) -> bool {
    for proto in addr.iter() {
        match proto {
            libp2p::multiaddr::Protocol::Ip4(ip)
                if ip.is_loopback()
                    || ip.is_unspecified()
                    || ip.is_link_local()
                    || ip == std::net::Ipv4Addr::new(169, 254, 169, 254) =>
            {
                return false;
            }
            libp2p::multiaddr::Protocol::Ip6(ip)
                if ip.is_loopback()
                    || ip.is_unspecified()
                    // IPv6 link-local (fe80::/10)
                    || (ip.segments()[0] & 0xffc0) == 0xfe80 =>
            {
                return false;
            }
            _ => {}
        }
    }
    true
}

#[cfg(test)]
mod listen_filter_tests {
    use super::*;

    fn addr(s: &str) -> Multiaddr {
        s.parse().unwrap()
    }

    #[test]
    fn drops_loopback_and_unspecified() {
        assert!(!addr_is_remotely_reachable(&addr(
            "/ip4/127.0.0.1/tcp/8810"
        )));
        assert!(!addr_is_remotely_reachable(&addr("/ip4/0.0.0.0/tcp/8810")));
        assert!(!addr_is_remotely_reachable(&addr("/ip6/::1/tcp/8810")));
        assert!(!addr_is_remotely_reachable(&addr("/ip6/::/tcp/8810")));
    }

    #[test]
    fn drops_link_local_and_metadata() {
        assert!(!addr_is_remotely_reachable(&addr(
            "/ip4/169.254.1.5/tcp/8810"
        )));
        assert!(!addr_is_remotely_reachable(&addr(
            "/ip4/169.254.169.254/tcp/8810"
        )));
        assert!(!addr_is_remotely_reachable(&addr("/ip6/fe80::1/tcp/8810")));
    }

    #[test]
    fn keeps_lan_and_tailscale_and_public() {
        // RFC 1918 LAN
        assert!(addr_is_remotely_reachable(&addr(
            "/ip4/192.168.1.5/tcp/8810"
        )));
        assert!(addr_is_remotely_reachable(&addr("/ip4/10.0.0.5/tcp/8810")));
        // Tailscale CGN
        assert!(addr_is_remotely_reachable(&addr(
            "/ip4/100.64.10.5/tcp/8810"
        )));
        // Public
        assert!(addr_is_remotely_reachable(&addr(
            "/ip4/203.0.113.5/udp/8800/quic-v1"
        )));
        // IPv6 ULA + global
        assert!(addr_is_remotely_reachable(&addr("/ip6/fc00::1/tcp/8810")));
        assert!(addr_is_remotely_reachable(&addr(
            "/ip6/2001:db8::1/tcp/8810"
        )));
    }

    #[test]
    fn union_includes_external_address_and_drops_loopback() {
        let pid = libp2p::PeerId::random();
        let candidates = vec![
            addr("/ip4/127.0.0.1/tcp/8810"),   // loopback listener → dropped
            addr("/ip4/192.168.1.5/tcp/8810"), // LAN listener → kept
            addr("/ip4/203.0.113.5/tcp/8810"), // confirmed external → kept
        ];
        let list = build_reachable_multiaddr_list(candidates.into_iter(), pid);
        assert_eq!(list.len(), 2);
        assert!(list.iter().all(|s| s.contains(&pid.to_string())));
        assert!(list.iter().any(|s| s.contains("192.168.1.5")));
        assert!(list.iter().any(|s| s.contains("203.0.113.5")));
        assert!(!list.iter().any(|s| s.contains("127.0.0.1")));
    }

    #[test]
    fn ensure_p2p_suffix_appends_when_absent_and_preserves_when_present() {
        let pid = libp2p::PeerId::random();
        let suffixed = ensure_p2p_suffix(addr("/ip4/203.0.113.5/tcp/8810"), pid);
        assert!(suffixed.to_string().ends_with(&format!("/p2p/{pid}")));

        // Already carries a peer id → returned verbatim, no double-append.
        let already = addr(&format!("/ip4/203.0.113.5/tcp/8810/p2p/{pid}"));
        let out = ensure_p2p_suffix(already.clone(), pid);
        assert_eq!(out, already);
        assert_eq!(out.to_string().matches("/p2p/").count(), 1);
    }

    /// A node may only donate itself as a relay when strangers can dial it
    /// directly. The permissive direction is the dangerous one: a NAT'd node
    /// that advertised `relay_capable` would attract forwards it then drops.
    #[test]
    fn only_genuinely_public_addresses_qualify_as_relay_candidates() {
        for a in [
            // Real globally-routable addresses. Deliberately NOT the RFC 5737
            // documentation ranges (192.0.2/198.51.100/203.0.113): those are
            // excluded by `is_documentation()` and would make this vacuous.
            "/ip4/93.184.216.34/tcp/8810",
            "/ip4/8.8.8.8/tcp/8810",
            "/ip6/2606:4700:4700::1111/tcp/8810",
            "/dns4/anchor.example.org/tcp/8810",
            "/dnsaddr/bootstrap.example.org",
        ] {
            let parsed: libp2p::Multiaddr = a.parse().unwrap();
            assert!(
                super::multiaddr_is_public_relay_candidate(&parsed),
                "{a} should qualify as a relay candidate"
            );
        }

        for a in [
            // RFC1918 LAN + Docker bridge.
            "/ip4/192.168.1.10/tcp/8810",
            "/ip4/10.0.0.5/tcp/8810",
            "/ip4/172.17.0.1/tcp/8810",
            // CGNAT / Tailscale — reachable on the overlay, not from outside.
            "/ip4/100.64.12.9/tcp/8810",
            "/ip4/100.127.255.1/tcp/8810",
            // Loopback + link-local.
            "/ip4/127.0.0.1/tcp/8810",
            "/ip4/169.254.1.1/tcp/8810",
            // IPv6 ULA + link-local.
            "/ip6/fd00::1/tcp/8810",
            "/ip6/fe80::1/tcp/8810",
            // RFC 5737 documentation ranges are not globally routable.
            "/ip4/203.0.113.7/tcp/8810",
            "/ip4/192.0.2.1/tcp/8810",
        ] {
            let parsed: libp2p::Multiaddr = a.parse().unwrap();
            assert!(
                !super::multiaddr_is_public_relay_candidate(&parsed),
                "{a} must NOT qualify as a relay candidate"
            );
        }
    }

    /// Reachability *through* someone else's relay means this node is itself
    /// NAT'd — it cannot forward for others regardless of what the address
    /// looks like before the circuit hop.
    #[test]
    fn circuit_addresses_never_qualify_as_relay_candidates() {
        let relay = libp2p::PeerId::random();
        for a in [
            format!("/ip4/93.184.216.34/tcp/8810/p2p/{relay}/p2p-circuit"),
            format!("/dns4/anchor.example.org/tcp/8810/p2p/{relay}/p2p-circuit"),
        ] {
            let parsed: libp2p::Multiaddr = a.parse().unwrap();
            assert!(
                !super::multiaddr_is_public_relay_candidate(&parsed),
                "a /p2p-circuit address must never make us a relay: {a}"
            );
        }
    }

    #[test]
    fn relay_circuit_address_gets_our_suffix() {
        let pid = libp2p::PeerId::random();
        let relay = libp2p::PeerId::random();
        // A relay-reservation listener ends in /p2p-circuit; our id appends.
        let circuit = addr(&format!(
            "/ip4/203.0.113.9/tcp/8810/p2p/{relay}/p2p-circuit"
        ));
        let s = ensure_p2p_suffix(circuit, pid).to_string();
        assert!(s.ends_with(&format!("/p2p-circuit/p2p/{pid}")), "got {s}");
    }

    #[test]
    fn union_dedups_listener_and_external_overlap() {
        let pid = libp2p::PeerId::random();
        let candidates = vec![
            addr("/ip4/203.0.113.5/tcp/8810"),
            addr("/ip4/203.0.113.5/tcp/8810"), // same addr present in both sets
        ];
        let list = build_reachable_multiaddr_list(candidates.into_iter(), pid);
        assert_eq!(list.len(), 1);
    }
}
