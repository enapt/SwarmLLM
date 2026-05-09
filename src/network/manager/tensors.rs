//! Tensor + streaming-token send paths and inbound tensor payload decoding.
//!
//! Outbound:
//! - `handle_send_tensor` — encrypts (or fails) a `LayerForward` for a peer
//!   and dispatches it as a `SwarmRequest::TensorPayload`.
//! - `handle_send_tensor_result` — prefers writing a `LayerResult` back on
//!   the original forward's stored `ResponseChannel`; falls back to a fresh
//!   request when the channel is gone.
//! - `send_tensor_result_as_request` — fallback path for the above.
//! - `handle_send_streaming_token` / `handle_send_rr_message` — small JSON
//!   point-to-point helpers via request_response.
//!
//! Inbound:
//! - `handle_tensor_payload` decodes the WIRE_TAG_TENSOR-tagged payload
//!   (forward, result, encrypted) and dispatches into the daemon's
//!   AuthenticatedMessage stream.
//!
//! Plus `prepare_shard_read` (sync helper used by `requests.rs::handle_request`
//! before spawning blocking I/O) and `resolve_peer_id` (peer-bytes decode).

use crate::network::protocol::{self, SwarmRequest, SwarmResponse};
use crate::types::SwarmMessage;

use super::NetworkManager;

impl NetworkManager {
    /// Send a tensor forward to a specific peer via the unified binary tensor protocol.
    /// Uses WIRE_TAG_TENSOR (0x01) framing. Encrypts activations when an encryption
    /// session exists, falls back to plaintext.
    pub(super) fn handle_send_tensor(
        &mut self,
        target_peer_bytes: Vec<u8>,
        forward: crate::types::LayerForward,
    ) {
        let Some(peer_id) = Self::resolve_peer_id(&target_peer_bytes, "tensor send") else {
            return;
        };

        // Try to find the peer's NodeId for encryption
        let peer_node_id = self.peer_to_node_id(&peer_id);
        let use_encryption =
            self.shared_state.config.network.enable_encryption && peer_node_id.is_some();

        let payload = if use_encryption {
            let node_id = match peer_node_id {
                Some(n) => n,
                None => {
                    tracing::warn!(%peer_id, "Encryption enabled but no NodeId for peer");
                    return;
                }
            };
            let aad = protocol::build_layer_forward_aad(&forward);

            tracing::debug!(
                request_id = %forward.request_id,
                %peer_id,
                node_id = %node_id,
                aad_len = aad.len(),
                activation_len = forward.activations.len(),
                has_session = self.shared_state.session_manager.has_session(&node_id),
                session_count = self.shared_state.session_manager.session_count(),
                "DIAG: encrypting tensor forward"
            );

            match self
                .shared_state
                .session_manager
                .seal(&node_id, &forward.activations, &aad)
            {
                Ok(sealed) => match protocol::encode_layer_forward_encrypted(&forward, sealed) {
                    Ok(p) => p,
                    Err(e) => {
                        tracing::warn!(error = %e, "Failed to encode encrypted tensor");
                        return;
                    }
                },
                Err(e) => {
                    // SEC: Never fall back to plaintext — fail the forward instead.
                    // Plaintext fallback would silently strip encryption, allowing
                    // eavesdroppers to read intermediate tensor activations.
                    tracing::warn!(
                        error = %e,
                        request_id = %forward.request_id,
                        %peer_id,
                        node_id = %node_id,
                        aad_len = aad.len(),
                        has_session = self.shared_state.session_manager.has_session(&node_id),
                        "DIAG: seal() failed — dropping forward (no plaintext fallback)"
                    );
                    // Notify the pipeline immediately so it fails fast
                    // instead of waiting for the AllReduce timeout.
                    let request_id = forward.request_id;
                    if let Some((_, channel)) = self.pending_tensor_channels.remove(&request_id) {
                        let _ = self
                            .swarm
                            .behaviour_mut()
                            .request_response
                            .send_response(channel, SwarmResponse::Ack);
                    }
                    let error_result = crate::types::LayerResult::error(
                        request_id,
                        "Encryption session lost — reconnecting",
                    );
                    if let Some((_, tx)) =
                        self.shared_state.pending_layer_results.remove(&request_id)
                    {
                        let _ = tx.send(error_result);
                    }
                    return;
                }
            }
        } else {
            match protocol::encode_layer_forward(&forward) {
                Ok(p) => p,
                Err(e) => {
                    tracing::warn!(error = %e, "Failed to encode tensor forward");
                    return;
                }
            }
        };

        let payload_len = payload.len();
        let is_connected = self.swarm.is_connected(&peer_id);
        // DIAG: Check if request_response behaviour thinks this peer is connected
        let rr_is_connected = self
            .swarm
            .behaviour()
            .request_response
            .is_connected(&peer_id);
        // Count total established connections (all peers) for diagnostics
        let total_conn_count = self
            .swarm
            .network_info()
            .connection_counters()
            .num_established();
        tracing::info!(
            %peer_id,
            request_id = %forward.request_id,
            rr_is_connected,
            swarm_is_connected = is_connected,
            "DIAG: PRE-send_request state"
        );
        // CRITICAL: If the swarm says the peer is not connected, fail immediately.
        // The rr behaviour may have a stale connection entry (rr_connected=true)
        // after a disconnect, causing send_request() to target a dead ConnectionId
        // whose NotifyHandler is silently dropped by the swarm pool.
        if !is_connected {
            tracing::warn!(
                %peer_id,
                request_id = %forward.request_id,
                rr_is_connected,
                "Peer not connected — failing tensor forward immediately"
            );
            self.fail_tensor_forward(forward.request_id, &peer_id, "Peer not connected".into());
            return;
        }
        let req = SwarmRequest::TensorPayload(payload);
        let outbound_id = self
            .swarm
            .behaviour_mut()
            .request_response
            .send_request(&peer_id, req);
        // Track OutboundRequestId → (UUID, time, peer, layers, activation_size)
        // so we can notify pipeline on OutboundFailure and compute adaptive stale timeouts.
        let num_layers = forward.layer_range.1.saturating_sub(forward.layer_range.0);
        let activation_bytes = forward.activations.len();
        self.pending_tensor_outbound.insert(
            outbound_id,
            (
                forward.request_id,
                std::time::Instant::now(),
                peer_id,
                num_layers,
                activation_bytes,
            ),
        );
        // DIAG: check is_pending_outbound immediately — confirms the request was registered
        let is_rr_pending = self
            .swarm
            .behaviour()
            .request_response
            .is_pending_outbound(&peer_id, &outbound_id);
        // DIAG: enumerate connection IDs for this peer to detect stale conn_id issues.
        // Allocated and stringified eagerly by the tracing! macro regardless of subscriber
        // level, so gate the expensive `connection_addrs` enumeration on whether DEBUG is
        // actually enabled. At the default `info` filter the per-token call rate would
        // otherwise burn ~50–100 KB of throwaway heap per LayerForward.
        let peer_established_count = self
            .swarm
            .connected_peers()
            .filter(|p| **p == peer_id)
            .count();
        if tracing::enabled!(tracing::Level::DEBUG) {
            let all_conn_ids: Vec<_> = self
                .connection_addrs
                .iter()
                .map(|(cid, addr)| format!("{cid:?}→{addr}"))
                .collect();
            tracing::debug!(
                %peer_id,
                request_id = %forward.request_id,
                seq = forward.sequence_num,
                encrypted = use_encryption,
                payload_len,
                is_connected,
                total_connections = total_conn_count,
                peer_established_count,
                is_rr_pending,
                pending_tensor_count = self.pending_tensor_outbound.len(),
                ?outbound_id,
                tracked_connections = ?all_conn_ids,
                "DIAG: sent tensor forward via send_request (verbose)"
            );
        } else {
            tracing::info!(
                %peer_id,
                request_id = %forward.request_id,
                seq = forward.sequence_num,
                encrypted = use_encryption,
                payload_len,
                is_connected,
                total_connections = total_conn_count,
                peer_established_count,
                is_rr_pending,
                pending_tensor_count = self.pending_tensor_outbound.len(),
                ?outbound_id,
                "DIAG: sent tensor forward via send_request"
            );
        }
    }

    /// Send a tensor result back to the requesting peer.
    ///
    /// Prefers sending the result as a **response** on the original forward's
    /// ResponseChannel (stored in `pending_tensor_channels`). This keeps the
    /// entire forward→result exchange on a single QUIC substream, halving
    /// substream usage and preventing the stall that occurred when results
    /// were sent as separate requests.
    ///
    /// Falls back to a new request if no stored channel is found (e.g. timeout
    /// or the forward arrived via gossip).
    pub(super) fn handle_send_tensor_result(
        &mut self,
        target_peer_bytes: Vec<u8>,
        result: crate::types::LayerResult,
    ) {
        // Persistent pipeline stream: if this request came in on a stream, the
        // handler task registered a oneshot in `pending_stream_result_routes`.
        // Delivering there writes the result frame back on the same stream —
        // no request_response traffic needed.
        if let Some((_, tx)) = self
            .shared_state
            .pending_stream_result_routes
            .remove(&result.request_id)
        {
            let request_id = result.request_id;
            let tokens = result.token_ids.len();
            if tx.send(result).is_err() {
                tracing::debug!(
                    %request_id,
                    "pipeline stream result route dropped — handler task exited"
                );
            } else {
                tracing::debug!(
                    %request_id,
                    tokens,
                    "DIAG: delivered result via pipeline stream route"
                );
            }
            return;
        }

        let Some(peer_id) = Self::resolve_peer_id(&target_peer_bytes, "tensor result") else {
            return;
        };

        let is_connected = self.swarm.is_connected(&peer_id);
        if !is_connected {
            tracing::error!(
                %peer_id,
                request_id = %result.request_id,
                "DIAG: cannot send tensor result — peer NOT connected"
            );
        }

        // Try to send as response on the original forward's channel
        if let Some((_inserted, channel)) = self.pending_tensor_channels.remove(&result.request_id)
        {
            match protocol::encode_layer_result(&result) {
                Ok(payload) => {
                    let payload_len = payload.len();
                    let resp = SwarmResponse::TensorPayload(payload);
                    if self
                        .swarm
                        .behaviour_mut()
                        .request_response
                        .send_response(channel, resp)
                        .is_err()
                    {
                        tracing::warn!(
                            %peer_id,
                            request_id = %result.request_id,
                            "ResponseChannel closed, falling back to new request"
                        );
                        // Channel was closed (timeout/disconnect) — fall back to new request
                        self.send_tensor_result_as_request(&peer_id, &result, is_connected);
                    } else {
                        tracing::info!(
                            %peer_id,
                            request_id = %result.request_id,
                            payload_len,
                            is_connected,
                            tokens = result.token_ids.len(),
                            activations_bytes = result.activations.len(),
                            finish = ?result.finish_reason,
                            pending_channels = self.pending_tensor_channels.len(),
                            "DIAG: sent tensor result as response (same substream)"
                        );
                    }
                }
                Err(e) => {
                    tracing::warn!(error = %e, request_id = %result.request_id, "Failed to encode tensor result");
                }
            }
        } else {
            // No stored ResponseChannel — peer may have reconnected or sent a result for a different substream
            tracing::debug!(
                %peer_id,
                request_id = %result.request_id,
                "No stored ResponseChannel, sending result as new request"
            );
            self.send_tensor_result_as_request(&peer_id, &result, is_connected);
        }
    }

    /// Timeout recovery: send tensor result as a new outbound request when the
    /// original response channel was closed (peer disconnect or timeout).
    pub(super) fn send_tensor_result_as_request(
        &mut self,
        peer_id: &libp2p::PeerId,
        result: &crate::types::LayerResult,
        is_connected: bool,
    ) {
        if !is_connected {
            tracing::warn!(
                %peer_id,
                request_id = %result.request_id,
                "Dropping tensor result fallback — peer not connected"
            );
            return;
        }
        match protocol::encode_layer_result(result) {
            Ok(payload) => {
                let payload_len = payload.len();
                let req = SwarmRequest::TensorPayload(payload);
                let outbound_id = self
                    .swarm
                    .behaviour_mut()
                    .request_response
                    .send_request(peer_id, req);
                // Track for observability so OutboundFailure can log the result UUID.
                self.pending_tensor_result_outbound
                    .insert(outbound_id, (result.request_id, std::time::Instant::now()));
                tracing::info!(
                    %peer_id,
                    request_id = %result.request_id,
                    payload_len,
                    is_connected,
                    tokens = result.token_ids.len(),
                    activations_bytes = result.activations.len(),
                    finish = ?result.finish_reason,
                    ?outbound_id,
                    "DIAG: sent tensor result as new request (fallback)"
                );
            }
            Err(e) => {
                tracing::warn!(error = %e, request_id = %result.request_id, "Failed to encode tensor result (fallback)");
            }
        }
    }

    /// Process an inbound binary tensor payload (from either request or response).
    /// Returns `Some(request_id)` if this was a LayerForward (so the caller can
    /// save the ResponseChannel for sending the result back as a response).
    pub(super) fn handle_tensor_payload(
        &mut self,
        peer: libp2p::PeerId,
        payload: &[u8],
    ) -> Option<uuid::Uuid> {
        let tag = payload.first().copied().unwrap_or(0);
        tracing::debug!(%peer, tag, payload_len = payload.len(), "handle_tensor_payload");
        match tag {
            protocol::TENSOR_TAG_FORWARD => {
                tracing::info!(%peer, payload_len = payload.len(), "Received tensor LayerForward");
                match protocol::decode_layer_forward(payload) {
                    Ok(mut forward) => {
                        let request_id = forward.request_id;
                        let is_tp = forward.tp_meta.is_some();
                        forward.sender_peer_bytes = Some(peer.to_bytes());
                        let msg = SwarmMessage::LayerForward(forward);
                        if let Err(e) = self.dispatch_authenticated(Some(&peer), msg) {
                            self.shared_state
                                .metrics
                                .channel_metrics
                                .network_out
                                .record_dropped();
                            tracing::warn!(error = %e, "Outbound channel full, dropping tensor forward");
                            return None;
                        } else {
                            self.shared_state
                                .metrics
                                .channel_metrics
                                .network_out
                                .record_sent();
                        }
                        // TP forwards send their response via separate TpAllReduceRequest,
                        // NOT through the original request_response channel. Return None
                        // so the caller ACKs immediately instead of storing the channel.
                        if is_tp {
                            return None;
                        }
                        Some(request_id)
                    }
                    Err(e) => {
                        tracing::warn!(error = %e, "Failed to decode tensor forward");
                        None
                    }
                }
            }
            protocol::TENSOR_TAG_RESULT => {
                tracing::info!(%peer, payload_len = payload.len(), "Received tensor LayerResult");
                match protocol::decode_layer_result(payload) {
                    Ok(result) => {
                        tracing::debug!(
                            %peer,
                            request_id = %result.request_id,
                            tokens = result.token_ids.len(),
                            activations_bytes = result.activations.len(),
                            "Decoded tensor LayerResult, dispatching"
                        );
                        if let Err(e) = self
                            .dispatch_authenticated(Some(&peer), SwarmMessage::LayerResult(result))
                        {
                            self.shared_state
                                .metrics
                                .channel_metrics
                                .network_out
                                .record_dropped();
                            tracing::warn!(error = %e, "Outbound channel full, dropping tensor result");
                        } else {
                            self.shared_state
                                .metrics
                                .channel_metrics
                                .network_out
                                .record_sent();
                        }
                    }
                    Err(e) => {
                        tracing::warn!(error = %e, "Failed to decode tensor result");
                    }
                }
                None
            }
            protocol::TENSOR_TAG_ENCRYPTED => {
                tracing::info!(
                    %peer,
                    payload_len = payload.len(),
                    "DIAG: Received encrypted tensor"
                );
                match protocol::decode_layer_forward_encrypted(payload) {
                    Ok((mut forward, sealed, aad)) => {
                        let sender_node_id = self.peer_to_node_id(&peer);
                        tracing::debug!(
                            %peer,
                            request_id = %forward.request_id,
                            sender_node_id = ?sender_node_id.as_ref().map(|n| format!("{}", n)),
                            aad_len = aad.len(),
                            sealed_len = sealed.len(),
                            has_session = sender_node_id.as_ref().is_some_and(|n| self.shared_state.session_manager.has_session(n)),
                            "DIAG: decrypting tensor"
                        );
                        if let Some(node_id) = sender_node_id {
                            match self
                                .shared_state
                                .session_manager
                                .open(&node_id, &sealed, &aad)
                            {
                                Ok(plaintext) => {
                                    let request_id = forward.request_id;
                                    let is_tp = forward.tp_meta.is_some();
                                    forward.activations = plaintext;
                                    forward.sender_peer_bytes = Some(peer.to_bytes());
                                    if let Err(e) = self.dispatch_authenticated(
                                        Some(&peer),
                                        SwarmMessage::LayerForward(forward),
                                    ) {
                                        self.shared_state
                                            .metrics
                                            .channel_metrics
                                            .network_out
                                            .record_dropped();
                                        tracing::warn!(error = %e, "Outbound channel full, dropping decrypted tensor");
                                        return None;
                                    } else {
                                        self.shared_state
                                            .metrics
                                            .channel_metrics
                                            .network_out
                                            .record_sent();
                                    }
                                    // TP forwards respond via TpAllReduceRequest, not this channel
                                    if is_tp {
                                        return None;
                                    }
                                    return Some(request_id);
                                }
                                Err(e) => {
                                    tracing::error!(
                                        error = %e,
                                        %peer,
                                        request_id = %forward.request_id,
                                        node_id = %node_id,
                                        aad_len = aad.len(),
                                        sealed_len = sealed.len(),
                                        model_id = %forward.model_id,
                                        layer_range = ?forward.layer_range,
                                        seq = forward.sequence_num,
                                        "DIAG: decrypt FAILED — possible AAD mismatch, key mismatch, or corruption"
                                    );
                                }
                            }
                        } else {
                            tracing::warn!(
                                %peer,
                                "Encrypted tensor from unknown peer — dropping"
                            );
                        }
                    }
                    Err(e) => {
                        tracing::warn!(error = %e, "Failed to decode encrypted tensor");
                    }
                }
                None
            }
            _ => {
                tracing::warn!(%peer, tag, "Unknown tensor message tag");
                None
            }
        }
    }

    /// Check if a shard is available locally and return the path + request params.
    /// Returns None if the shard is not available. This is sync to avoid holding
    /// `&self` across async boundaries (NetworkManager is not Send).
    pub(super) fn prepare_shard_read(
        &self,
        req: &crate::types::ShardRequest,
    ) -> Option<(std::path::PathBuf, u64, u64, crate::types::ModelId, u32)> {
        let model_id = &req.shard_id.model_id;
        let shard_index = req.shard_id.index;

        let shard_path = self.shard_store.shard_path(model_id, shard_index);
        if shard_path.exists() {
            return Some((
                shard_path,
                req.chunk_offset,
                req.chunk_size,
                model_id.clone(),
                shard_index,
            ));
        }

        tracing::debug!(
            model = %model_id,
            shard = shard_index,
            "Shard not available locally"
        );
        None
    }

    /// Send a StreamingToken to a specific peer via the JSON request_response protocol.
    pub(super) fn handle_send_streaming_token(
        &mut self,
        target_peer_bytes: Vec<u8>,
        token: crate::types::StreamingToken,
    ) {
        let Some(peer_id) = Self::resolve_peer_id(&target_peer_bytes, "streaming token") else {
            return;
        };

        if !self.swarm.is_connected(&peer_id) {
            tracing::warn!(
                %peer_id,
                "Dropping streaming token — peer not connected"
            );
            return;
        }
        let msg = SwarmMessage::StreamingToken(token);
        let req = SwarmRequest::Message(Box::new(msg));
        let req_id = self
            .swarm
            .behaviour_mut()
            .request_response
            .send_request(&peer_id, req);
        self.pending_rr_observability.insert(
            req_id,
            (
                "streaming_token".to_string(),
                std::time::Instant::now(),
                None,
            ),
        );
    }

    /// Send an arbitrary SwarmMessage to a specific peer via request_response.
    /// Used for AllReduce and other point-to-point messages.
    ///
    /// `delivery_request_id` is the optional Uuid identifying a streaming
    /// request whose `streaming_token_txs` channel should be closed if no ACK
    /// arrives within `RR_ACK_TIMEOUT`. Used by the remote-generate fast path
    /// to fail fast on rare libp2p rr silent-drops; pass `None` for fire-and-
    /// forget messages.
    pub(super) fn handle_send_rr_message(
        &mut self,
        target_peer_bytes: Vec<u8>,
        msg: SwarmMessage,
        label: &str,
        delivery_request_id: Option<uuid::Uuid>,
    ) {
        let Some(peer_id) = Self::resolve_peer_id(&target_peer_bytes, label) else {
            // Notify the streaming caller immediately — otherwise it sits at
            // FIRST_TOKEN_TIMEOUT (120s) waiting for a peer we can't even
            // reach.
            if let Some(uuid) = delivery_request_id {
                self.shared_state.streaming_token_txs.remove(&uuid);
            }
            return;
        };

        if !self.swarm.is_connected(&peer_id) {
            tracing::warn!(
                %peer_id,
                label,
                "Dropping rr message — peer not connected"
            );
            if let Some(uuid) = delivery_request_id {
                self.shared_state.streaming_token_txs.remove(&uuid);
            }
            return;
        }
        let req = SwarmRequest::Message(Box::new(msg));
        let req_id = self
            .swarm
            .behaviour_mut()
            .request_response
            .send_request(&peer_id, req);
        self.pending_rr_observability.insert(
            req_id,
            (
                label.to_string(),
                std::time::Instant::now(),
                delivery_request_id,
            ),
        );
    }
}
