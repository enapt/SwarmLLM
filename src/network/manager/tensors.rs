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
use crate::types::{NetworkCommand, SwarmMessage};

use super::NetworkManager;

impl NetworkManager {
    /// Send a tensor forward to a specific peer via the unified binary tensor protocol.
    /// Uses WIRE_TAG_TENSOR (0x01) framing. Encrypts activations when an encryption
    /// session exists, falls back to plaintext.
    ///
    /// Encryption path (default config — `enable_encryption=true`) offloads the
    /// CPU-bound ChaCha20-Poly1305 sealing + protocol encoding to a
    /// `tokio::spawn` task so the NetworkManager event loop stays responsive
    /// during high-volume distributed inference. The spawn posts back a
    /// `NetworkCommand::SendEncodedTensor` once the wire-ready payload is in
    /// hand; the critical task then performs only `send_request` +
    /// `pending_tensor_outbound` bookkeeping. ~50–200µs saved per token on the
    /// event loop for 1MB activations; multiplied across concurrent decode
    /// traffic this is the difference between a smooth event loop and
    /// observable jitter on libp2p ping / gossip / connection events.
    ///
    /// Plaintext path stays inline — `encode_layer_forward` is memcpy-cheap
    /// and the spawn dispatch overhead would exceed the saving.
    pub(super) fn handle_send_tensor(
        &mut self,
        target_peer_bytes: Vec<u8>,
        forward: crate::types::LayerForward,
    ) {
        let Some(peer_id) = Self::resolve_peer_id(&target_peer_bytes, "tensor send") else {
            return;
        };

        // NETWORKING_PLAN tensor relay — when the target is unreachable OR
        // reachable only via a flaky relay *circuit*, route the forward through
        // a relay (ephemeral-sealed for the target) instead of the circuit RR.
        // Encode the plaintext forward here; `try_relay_tensor` seals it. Only
        // fall through to the direct send when there is no relay path.
        let connected = self.swarm.is_connected(&peer_id);
        if !connected || !self.has_direct_connection(&peer_id) {
            match protocol::encode_layer_forward(&forward) {
                Ok(encoded) => {
                    if self.try_relay_tensor(
                        &target_peer_bytes,
                        &encoded,
                        /*is_result*/ false,
                        forward.request_id,
                    ) {
                        return;
                    }
                }
                Err(e) => {
                    tracing::warn!(error = %e, "Failed to encode forward for relay — will try direct");
                }
            }
            if !connected {
                tracing::warn!(
                    %peer_id,
                    request_id = %forward.request_id,
                    "Peer not connected and no relay path — failing tensor forward"
                );
                self.fail_tensor_forward(forward.request_id, &peer_id, "Peer not connected".into());
                return;
            }
            // Circuit-only but no relay path — fall through to a best-effort
            // direct send over the circuit.
        }

        // Try to find the peer's NodeId for encryption
        let peer_node_id = self.peer_to_node_id(&peer_id);
        let use_encryption =
            self.shared_state.config.network.enable_encryption && peer_node_id.is_some();

        if use_encryption {
            // Encryption path — offload ChaCha20 sealing + encode to a spawn
            // task so the event loop is not blocked. The shared
            // `encode_forward_for_wire` helper (network/pipeline_stream.rs)
            // performs the same encrypt+encode as the inline pre-R139 path;
            // wire bytes are byte-identical with the persistent-stream path.
            let shared_state = self.shared_state.clone();
            let internal_cmd_tx = self.internal_cmd_tx.clone();
            let request_id = forward.request_id;
            let num_layers = forward.layer_range.1.saturating_sub(forward.layer_range.0);
            let activation_bytes = forward.activations.len();
            let target_peer_bytes_for_cmd = target_peer_bytes.clone();
            tokio::spawn(async move {
                let payload = match crate::network::pipeline_stream::encode_forward_for_wire(
                    &forward,
                    &peer_id,
                    &shared_state,
                ) {
                    Ok(p) => p,
                    Err(e) => {
                        // SEC: Never fall back to plaintext — fail the forward
                        // instead. Plaintext fallback would silently strip
                        // encryption, allowing eavesdroppers to read
                        // intermediate tensor activations.
                        tracing::warn!(
                            error = %e,
                            %request_id,
                            %peer_id,
                            activation_len = activation_bytes,
                            "DIAG: tensor encrypt+encode failed — dropping forward"
                        );
                        fail_pending_forward(
                            &shared_state,
                            request_id,
                            &peer_id,
                            format!("Encryption failed: {e}"),
                        );
                        return;
                    }
                };
                let cmd = NetworkCommand::SendEncodedTensor {
                    target_peer_bytes: target_peer_bytes_for_cmd,
                    payload,
                    request_id,
                    num_layers,
                    activation_bytes,
                };
                if let Err(e) = internal_cmd_tx.send(cmd).await {
                    tracing::warn!(
                        error = %e,
                        %request_id,
                        "internal_cmd_tx send failed — dropping encoded tensor"
                    );
                    fail_pending_forward(
                        &shared_state,
                        request_id,
                        &peer_id,
                        "Internal command queue closed",
                    );
                }
            });
            return;
        }

        // Plaintext path — stays inline (encode_layer_forward is memcpy-cheap).
        let payload = match protocol::encode_layer_forward(&forward) {
            Ok(p) => p,
            Err(e) => {
                tracing::warn!(error = %e, "Failed to encode tensor forward");
                return;
            }
        };

        let num_layers = forward.layer_range.1.saturating_sub(forward.layer_range.0);
        let activation_bytes = forward.activations.len();
        let request_id = forward.request_id;
        let sequence_num = forward.sequence_num;
        // Plaintext forwards go straight through — no spawn, no extra channel
        // hop. `dispatch_tensor_payload` is the shared write path for both
        // the inline-plaintext route here and the post-encrypt
        // `handle_send_encoded_tensor` continuation.
        self.dispatch_tensor_payload(
            peer_id,
            payload,
            request_id,
            Some(sequence_num),
            num_layers,
            activation_bytes,
            /*encrypted*/ false,
        );
    }

    /// Continuation of `handle_send_tensor`'s encryption branch. The spawned
    /// encrypt task posts a `NetworkCommand::SendEncodedTensor` back through
    /// `internal_cmd_tx` once the wire-ready payload is built; the
    /// NetworkManager event loop picks it up here and performs only the
    /// synchronous `send_request` + bookkeeping that requires `&mut self`.
    pub(super) fn handle_send_encoded_tensor(
        &mut self,
        target_peer_bytes: Vec<u8>,
        payload: Vec<u8>,
        request_id: uuid::Uuid,
        num_layers: u32,
        activation_bytes: usize,
    ) {
        let Some(peer_id) = Self::resolve_peer_id(&target_peer_bytes, "encoded tensor send") else {
            return;
        };
        // Re-check connectivity — the peer may have disconnected during the
        // brief encrypt window. The original `handle_send_tensor` already
        // fast-failed at entry, so this catches only mid-encrypt disconnects.
        if !self.swarm.is_connected(&peer_id) {
            tracing::warn!(
                %peer_id,
                %request_id,
                "Peer disconnected during encrypt — failing tensor forward"
            );
            self.fail_tensor_forward(
                request_id,
                &peer_id,
                "Peer disconnected during encrypt".into(),
            );
            return;
        }
        self.dispatch_tensor_payload(
            peer_id,
            payload,
            request_id,
            None,
            num_layers,
            activation_bytes,
            /*encrypted*/ true,
        );
    }

    /// Shared write path used by `handle_send_tensor` (plaintext) and
    /// `handle_send_encoded_tensor` (post-encrypt). Performs `send_request`
    /// and records the pending-outbound entry. R108 verbose DIAG logging is
    /// gated behind `tracing::Level::DEBUG` so default `info` builds pay
    /// nothing.
    #[allow(clippy::too_many_arguments)]
    fn dispatch_tensor_payload(
        &mut self,
        peer_id: libp2p::PeerId,
        payload: Vec<u8>,
        request_id: uuid::Uuid,
        sequence_num: Option<u32>,
        num_layers: u32,
        activation_bytes: usize,
        encrypted: bool,
    ) {
        let payload_len = payload.len();
        if tracing::enabled!(tracing::Level::DEBUG) {
            let rr_is_connected = self
                .swarm
                .behaviour()
                .request_response
                .is_connected(&peer_id);
            tracing::debug!(
                %peer_id,
                %request_id,
                rr_is_connected,
                "DIAG: PRE-send_request state"
            );
        }
        let req = SwarmRequest::TensorPayload(payload);
        let outbound_id = self
            .swarm
            .behaviour_mut()
            .request_response
            .send_request(&peer_id, req);
        self.pending_tensor_outbound.insert(
            outbound_id,
            (
                request_id,
                std::time::Instant::now(),
                peer_id,
                num_layers,
                activation_bytes,
            ),
        );
        if tracing::enabled!(tracing::Level::DEBUG) {
            let is_rr_pending = self
                .swarm
                .behaviour()
                .request_response
                .is_pending_outbound(&peer_id, &outbound_id);
            let peer_established_count = self
                .swarm
                .connected_peers()
                .filter(|p| **p == peer_id)
                .count();
            let total_conn_count = self
                .swarm
                .network_info()
                .connection_counters()
                .num_established();
            let all_conn_ids: Vec<_> = self
                .connection_addrs
                .iter()
                .map(|(cid, addr)| format!("{cid:?}→{addr}"))
                .collect();
            tracing::debug!(
                %peer_id,
                %request_id,
                seq = ?sequence_num,
                encrypted,
                payload_len,
                total_connections = total_conn_count,
                peer_established_count,
                is_rr_pending,
                pending_tensor_count = self.pending_tensor_outbound.len(),
                ?outbound_id,
                tracked_connections = ?all_conn_ids,
                "DIAG: sent tensor forward via send_request (verbose)"
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
        // NETWORKING_PLAN tensor relay — the return path. When the coordinator
        // (origin) is unreachable OR reachable only via a circuit — the common
        // case for a forward we received THROUGH a relay — route the result back
        // via a relay instead of dropping it. Uses the reverse route we learned
        // when the forward arrived.
        if !is_connected || !self.has_direct_connection(peer_id) {
            match protocol::encode_layer_result(result) {
                Ok(encoded) => {
                    if self.try_relay_tensor(
                        &peer_id.to_bytes(),
                        &encoded,
                        /*is_result*/ true,
                        result.request_id,
                    ) {
                        return;
                    }
                }
                Err(e) => {
                    tracing::warn!(error = %e, request_id = %result.request_id, "Failed to encode result for relay");
                }
            }
            if !is_connected {
                tracing::warn!(
                    %peer_id,
                    request_id = %result.request_id,
                    "Dropping tensor result fallback — peer not connected and no relay path"
                );
                return;
            }
            // Circuit-only but no relay path — fall through to best-effort direct.
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
                // R108: per-token; downgrade from info to debug to match the
                // `handle_tensor_payload` entry log at line 381.
                tracing::debug!(%peer, payload_len = payload.len(), "Received tensor LayerForward");
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
                tracing::debug!(%peer, payload_len = payload.len(), "Received tensor LayerResult");
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
                tracing::debug!(
                    %peer,
                    payload_len = payload.len(),
                    "DIAG: Received encrypted tensor"
                );
                // Decode the envelope on the critical task (cheap parse — just
                // header + trailer split, no crypto). ChaCha20 sealing of the
                // activation blob runs on a tokio::spawn so the event loop
                // is not blocked for ~50–200µs per inbound forward.
                let (forward, sealed, aad) = match protocol::decode_layer_forward_encrypted(payload)
                {
                    Ok(parts) => parts,
                    Err(e) => {
                        tracing::warn!(error = %e, "Failed to decode encrypted tensor");
                        return None;
                    }
                };
                let sender_node_id = self.peer_to_node_id(&peer);
                let request_id = forward.request_id;
                let is_tp = forward.tp_meta.is_some();
                tracing::debug!(
                    %peer,
                    %request_id,
                    sender_node_id = ?sender_node_id.as_ref().map(|n| format!("{}", n)),
                    aad_len = aad.len(),
                    sealed_len = sealed.len(),
                    has_session = sender_node_id.as_ref().is_some_and(|n| self.shared_state.session_manager.has_session(n)),
                    "DIAG: decrypting tensor"
                );
                let Some(node_id) = sender_node_id else {
                    tracing::warn!(%peer, "Encrypted tensor from unknown peer — dropping");
                    return None;
                };
                // Spawn decrypt + dispatch. The caller stores the
                // ResponseChannel keyed by the returned request_id; a decrypt
                // failure answers on it with `LayerResult::error` so the
                // coordinator can fail over at once.
                let shared_state = self.shared_state.clone();
                let outbound_tx = self.outbound_tx.clone();
                let self_command_tx = self.self_command_tx.clone();
                let peer_bytes = peer.to_bytes();
                tokio::spawn(async move {
                    let mut forward = forward;
                    let plaintext = match shared_state.session_manager.open(&node_id, &sealed, &aad)
                    {
                        Ok(p) => p,
                        Err(e) => {
                            tracing::error!(
                                error = %e,
                                %request_id,
                                %node_id,
                                aad_len = aad.len(),
                                sealed_len = sealed.len(),
                                model_id = %forward.model_id,
                                layer_range = ?forward.layer_range,
                                seq = forward.sequence_num,
                                "DIAG: decrypt FAILED — possible AAD mismatch, key mismatch, or corruption"
                            );
                            // Answer, rather than leaving the coordinator to
                            // discover this by timing out. Dropping the forward
                            // silently cost it the WHOLE segment budget — 30s on
                            // a decode hop, up to several minutes on a prefill —
                            // and it then had no error to attribute, so it could
                            // not fail over promptly either. Observed live
                            // 2026-08-02: a 10-minute key rotation re-keyed this
                            // session between the prefill and the first decode
                            // token, the peer still held the old key, and a
                            // request that was otherwise healthy died 30s later.
                            //
                            // The reason is deliberately generic: we cannot tell
                            // a rotation race from a tampered ciphertext, and
                            // saying which would tell an attacker whether their
                            // forgery had the right key.
                            let _ = self_command_tx
                                .send(crate::types::NetworkCommand::SendTensorResult {
                                    target_peer_bytes: peer_bytes.clone(),
                                    result: crate::types::LayerResult::error(
                                        request_id,
                                        "Could not decrypt forward".to_string(),
                                    ),
                                })
                                .await;
                            return;
                        }
                    };
                    forward.activations = plaintext;
                    forward.sender_peer_bytes = Some(peer_bytes.clone());
                    // R139 Tier 4K — if this is a chunked transfer, route
                    // through the assembly state instead of dispatching to
                    // the worker. The worker only sees the reassembled
                    // single LayerForward once the final chunk lands.
                    let dispatch_forward = if forward.chunk_meta.is_some() {
                        match shared_state.try_assemble_chunked_forward(forward, peer_bytes) {
                            Ok(Some(complete)) => complete,
                            Ok(None) => {
                                // Accumulating — wait for more chunks. Don't
                                // dispatch and don't error.
                                return;
                            }
                            Err(e) => {
                                tracing::warn!(
                                    error = %e,
                                    %request_id,
                                    "Chunk assembly rejected — dropping forward"
                                );
                                return;
                            }
                        }
                    } else {
                        forward
                    };
                    let msg = crate::types::AuthenticatedMessage {
                        sender: Some(node_id),
                        message: SwarmMessage::LayerForward(dispatch_forward),
                    };
                    if let Err(e) = outbound_tx.try_send(msg) {
                        shared_state
                            .metrics
                            .channel_metrics
                            .network_out
                            .record_dropped();
                        tracing::warn!(
                            error = %e,
                            %request_id,
                            "Outbound channel full, dropping decrypted tensor"
                        );
                    } else {
                        shared_state
                            .metrics
                            .channel_metrics
                            .network_out
                            .record_sent();
                    }
                });
                // TP forwards respond via TpAllReduceRequest, not the original
                // RR channel — caller should ACK rather than store the channel.
                if is_tp {
                    None
                } else {
                    Some(request_id)
                }
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

        let msg = SwarmMessage::StreamingToken(token);
        let connected = self.swarm.is_connected(&peer_id);
        // NETWORKING_PLAN Phase 1 — prefer the app-level relay when the
        // originator is unreachable OR reachable only via a flaky relay circuit
        // (a NAT'd coordinator we can't round-trip request_response to).
        if !connected || !self.has_direct_connection(&peer_id) {
            if self.try_relay_send(&target_peer_bytes, &msg) {
                return;
            }
            if !connected {
                tracing::warn!(
                    %peer_id,
                    "Dropping streaming token — peer not connected and no relay path"
                );
                return;
            }
            // Circuit-only but no relay path — fall through to a best-effort
            // send over the circuit.
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

        let connected = self.swarm.is_connected(&peer_id);
        // NETWORKING_PLAN Phase 1 — prefer the app-level relay when the target is
        // unreachable OR reachable only via a flaky relay circuit. Only the
        // remote-generate inference fast path is relay-eligible; anything else
        // (TP AllReduce) returns false and falls through.
        if !connected || !self.has_direct_connection(&peer_id) {
            if self.try_relay_send(&target_peer_bytes, &msg) {
                return;
            }
            if !connected {
                tracing::warn!(
                    %peer_id,
                    label,
                    "Dropping rr message — peer not connected and no relay path"
                );
                if let Some(uuid) = delivery_request_id {
                    self.shared_state.streaming_token_txs.remove(&uuid);
                }
                return;
            }
            // Circuit-only but no relay path — fall through to a best-effort
            // send over the circuit.
        }
        let req = SwarmRequest::Message(Box::new(msg));
        let req_id = self
            .swarm
            .behaviour_mut()
            .request_response
            .send_request(&peer_id, req);
        // Size the ACK deadline for THIS peer. A caller blocked on the reply is
        // failed fast when the send is silently dropped, and the whole value of
        // that depends on not also failing a peer that is merely far away.
        let deadline = self.ack_deadline_secs(&peer_id);
        self.pending_rr_observability.insert(
            req_id,
            (
                label.to_string(),
                std::time::Instant::now(),
                delivery_request_id.map(|u| (u, deadline)),
            ),
        );
    }

    /// How long to wait for a peer's ACK before treating the send as dropped.
    ///
    /// `RR_ACK_TIMEOUT_SECS` is the floor, so nothing changes on a LAN. Beyond
    /// that it scales with the round trip we have actually measured to this
    /// peer, because the ACK IS a round trip: a fixed 10s deadline gives a peer
    /// 6 seconds away almost no margin, and declared one such peer dead on a
    /// request it was answering correctly. Capped so an implausible latency
    /// reading cannot switch the sweep off.
    fn ack_deadline_secs(&self, peer_id: &libp2p::PeerId) -> u64 {
        let rtt_ms = crate::network::transport::peer_id_to_node_id(peer_id)
            .and_then(|nid| {
                self.shared_state
                    .peer_registry
                    .get(&nid)
                    .and_then(|p| p.latency_ms)
            })
            .unwrap_or(0);
        ack_deadline_from_rtt(rtt_ms)
    }
}

/// Turn a measured round trip into an ACK deadline.
///
/// Split out from the lookup so it can be tested without a swarm. An unknown
/// latency (`0`) yields the floor, which is what a freshly-connected peer gets
/// before its first ping lands — generous is the safe direction, since being
/// wrong means failing a request the peer was answering correctly.
fn ack_deadline_from_rtt(rtt_ms: u32) -> u64 {
    let scaled = (rtt_ms.saturating_mul(super::RR_ACK_RTT_HEADROOM) as u64).div_ceil(1000);
    scaled.clamp(super::RR_ACK_TIMEOUT_SECS, super::RR_ACK_TIMEOUT_MAX_SECS)
}

/// Resolve the `pending_layer_results` waiter for `request_id` with a
/// `LayerResult::error(reason)`. Used from within `tokio::spawn` closures that
/// don't have `&mut self` access (where the
/// `NetworkManager::fail_tensor_forward` method can't be called).
///
/// `target_peer` is the peer whose forward failed. The waiter is resolved only
/// if it is still expecting that peer — a request that has since failed over is
/// waiting on a standby, and must not be failed by the abandoned forward.
/// Silently returns when nothing matches.
pub(crate) fn fail_pending_forward(
    shared_state: &std::sync::Arc<crate::daemon::SharedState>,
    request_id: uuid::Uuid,
    target_peer: &libp2p::PeerId,
    reason: impl Into<String>,
) {
    let result = crate::types::LayerResult::error(request_id, reason.into());
    let sender = crate::network::transport::peer_id_to_node_id(target_peer);
    shared_state.resolve_pending_layer_result(sender.as_ref(), result);
}

#[cfg(test)]
mod ack_deadline_tests {
    use super::ack_deadline_from_rtt;
    use crate::network::manager::{RR_ACK_TIMEOUT_MAX_SECS, RR_ACK_TIMEOUT_SECS};

    /// A LAN peer must be unaffected — the fast-fail this deadline drives is
    /// worth having, and nothing about it was wrong at 1ms.
    #[test]
    fn a_nearby_peer_keeps_the_floor() {
        assert_eq!(ack_deadline_from_rtt(1), RR_ACK_TIMEOUT_SECS);
        assert_eq!(ack_deadline_from_rtt(12), RR_ACK_TIMEOUT_SECS);
        assert_eq!(ack_deadline_from_rtt(600), RR_ACK_TIMEOUT_SECS);
    }

    /// The case that produced a false "peer never acknowledged": a peer ~6s
    /// away had under twice its round trip to answer in, against a deadline
    /// chosen when ACKs were sub-millisecond.
    #[test]
    fn a_distant_peer_gets_room_for_its_round_trip() {
        let d = ack_deadline_from_rtt(6278);
        assert!(
            d > RR_ACK_TIMEOUT_SECS,
            "a 6.3s round trip must not be judged against a 10s deadline"
        );
        assert!(d >= 18, "expected roughly 3x the round trip, got {d}");
    }

    /// An implausible reading must not switch the sweep off — the sweep is what
    /// turns a silent drop into a fast failure instead of a two-minute hang.
    #[test]
    fn an_absurd_latency_is_capped() {
        assert_eq!(ack_deadline_from_rtt(u32::MAX), RR_ACK_TIMEOUT_MAX_SECS);
        assert_eq!(ack_deadline_from_rtt(600_000), RR_ACK_TIMEOUT_MAX_SECS);
    }

    /// Never measured yet — take the floor rather than zero, which would fail
    /// every send to a peer we have not pinged.
    #[test]
    fn an_unknown_latency_takes_the_floor() {
        assert_eq!(ack_deadline_from_rtt(0), RR_ACK_TIMEOUT_SECS);
    }
}
