//! Persistent bidirectional stream per pipeline session.
//!
//! The distributed inference hot path (`forward_through_segments`) historically
//! used libp2p `request_response` — which opens a fresh substream per token and
//! routes each forward through the behaviour's handler poll loop. On loopback,
//! that framing + correlation overhead dominated (~100 ms of the ~148 ms
//! per-token decode latency).
//!
//! This module exposes a `/swarmllm/pipeline/1.0.0` protocol built on
//! `libp2p-stream`. One long-lived stream per `(peer, request_id)` carries all
//! forward + result frames for the lifetime of a pipeline session. Frames use
//! the same `[len:4 LE][payload]` framing as the existing wire codec so the
//! already-encoded tensor payloads pass through unchanged (including ChaCha
//! sealing and the TENSOR_TAG_* byte at payload[0]).
//!
//! Coordinator side:
//!   - `PipelineStreamClient::send_forward` opens or reuses a stream to the
//!     target peer, queues the encoded forward for the writer task, and returns
//!     immediately. The decoded `LayerResult` arrives via the existing
//!     `SharedState.pending_layer_results` oneshot (same as the RR path).
//!
//! Remote side:
//!   - `spawn_accept_loop` accepts inbound streams and spawns a per-stream
//!     handler task. Each handler reads forwards sequentially, dispatches them
//!     via the normal dispatch path, awaits the result via
//!     `SharedState.pending_stream_result_routes`, and writes the result frame
//!     back on the same stream.
//!
//! Failover: on any stream I/O error, the client handle is evicted from the map
//! and the existing `NetworkCommand::SendTensor` path remains available as a
//! drop-in fallback. Callers can retry through the RR path unchanged.
//!
//! Feature-flagged behind `config.inference.persistent_pipeline_stream`.

use std::sync::Arc;

use dashmap::DashMap;
use futures::{AsyncReadExt, AsyncWriteExt, StreamExt};
use libp2p::{PeerId, StreamProtocol};
use libp2p_stream::{Control, IncomingStreams};
use tokio::sync::{mpsc, oneshot, Mutex};
use tokio::task::JoinHandle;
use uuid::Uuid;

use crate::daemon::SharedState;
use crate::error::SwarmError;
use crate::network::protocol::{self, TENSOR_TAG_ENCRYPTED, TENSOR_TAG_FORWARD, TENSOR_TAG_RESULT};
use crate::types::{LayerForward, LayerResult, SwarmMessage};

/// Stream-protocol identifier for per-pipeline bidirectional streams.
pub const PROTOCOL_PIPELINE: &str = "/swarmllm/pipeline/1.0.0";

/// Maximum frame payload size (16 MiB) — matches `MAX_JSON_MSG_SIZE` in the
/// existing codec plus headroom for larger activation payloads.
const MAX_FRAME_LEN: usize = 16 * 1024 * 1024;

/// Per-stream outbound queue capacity. One in-flight forward + headroom for
/// speculative batching in Item 2 of the speedup plan.
const OUTBOUND_QUEUE: usize = 8;

/// Handle to an open outbound pipeline stream. Dropping this handle aborts the
/// reader/writer tasks and closes the stream.
struct OutboundStreamHandle {
    tx: mpsc::Sender<Vec<u8>>,
    _reader: JoinHandle<()>,
    _writer: JoinHandle<()>,
}

impl Drop for OutboundStreamHandle {
    fn drop(&mut self) {
        self._reader.abort();
        self._writer.abort();
    }
}

/// Coordinator-side handle for opening outbound pipeline streams.
pub struct PipelineStreamClient {
    control: Mutex<Control>,
    streams: DashMap<Uuid, Arc<OutboundStreamHandle>>,
}

impl PipelineStreamClient {
    pub fn new(control: Control) -> Self {
        Self {
            control: Mutex::new(control),
            streams: DashMap::new(),
        }
    }

    /// Send an already-encoded forward frame to `peer_id` under `request_id`.
    /// Opens a new stream if none exists for this request. Returns immediately
    /// after queueing — the result comes back asynchronously via
    /// `SharedState.pending_layer_results` dispatched by the reader task.
    pub async fn send_forward(
        &self,
        request_id: Uuid,
        peer_id: PeerId,
        payload: Vec<u8>,
        shared_state: Arc<SharedState>,
    ) -> Result<(), SwarmError> {
        // Fast path: stream already open.
        if let Some(entry) = self.streams.get(&request_id) {
            let handle = entry.value().clone();
            drop(entry);
            return handle
                .tx
                .send(payload)
                .await
                .map_err(|_| SwarmError::Network("pipeline stream writer closed".into()));
        }

        // Slow path: open a new stream. Guarded by a single control so only one
        // open is in flight at a time (per the libp2p-stream backpressure note).
        let stream = {
            let mut control = self.control.lock().await;
            control
                .open_stream(peer_id, StreamProtocol::new(PROTOCOL_PIPELINE))
                .await
                .map_err(|e| SwarmError::Network(format!("open_stream failed: {e}")))?
        };

        let (read_half, write_half) = stream.split();
        let (tx, rx) = mpsc::channel::<Vec<u8>>(OUTBOUND_QUEUE);

        let writer_request_id = request_id;
        let writer = tokio::spawn(writer_task(write_half, rx, writer_request_id));
        let reader_state = shared_state.clone();
        let reader_request_id = request_id;
        let reader = tokio::spawn(async move {
            reader_task_outbound(read_half, reader_request_id, peer_id, reader_state).await
        });

        let handle = Arc::new(OutboundStreamHandle {
            tx: tx.clone(),
            _reader: reader,
            _writer: writer,
        });
        self.streams.insert(request_id, handle);

        // RAII guard: if the caller's future is cancelled between the
        // `streams.insert` above and the `tx.send().await` below — e.g.,
        // the parent dropped this future via tokio::select! — the entry
        // would otherwise sit in the DashMap with both background tasks
        // running until pipeline completion. Disarmed only on successful
        // first send; the explicit `close()` call at pipeline completion
        // remains the steady-state cleanup path.
        struct InsertGuard<'a> {
            streams: &'a DashMap<Uuid, Arc<OutboundStreamHandle>>,
            request_id: Uuid,
            armed: bool,
        }
        impl Drop for InsertGuard<'_> {
            fn drop(&mut self) {
                if self.armed {
                    self.streams.remove(&self.request_id);
                }
            }
        }
        let mut guard = InsertGuard {
            streams: &self.streams,
            request_id,
            armed: true,
        };

        let result = tx
            .send(payload)
            .await
            .map_err(|_| SwarmError::Network("pipeline stream writer closed".into()));
        if result.is_ok() {
            guard.armed = false;
        }
        result
    }

    /// Drop the stream for `request_id`. Writer/reader tasks terminate on drop.
    pub fn close(&self, request_id: Uuid) {
        self.streams.remove(&request_id);
    }
}

/// Coordinator-side writer task: pulls encoded payloads from the outbound
/// queue and writes them to the stream as length-prefixed frames.
async fn writer_task<W>(mut write: W, mut rx: mpsc::Receiver<Vec<u8>>, request_id: Uuid)
where
    W: futures::AsyncWrite + Unpin,
{
    while let Some(payload) = rx.recv().await {
        if let Err(e) = write_frame(&mut write, &payload).await {
            tracing::warn!(
                %request_id,
                error = %e,
                "pipeline stream writer terminated"
            );
            break;
        }
    }
    // Best-effort close — ignore errors, the peer may have already closed.
    let _ = write.close().await;
}

/// Coordinator-side reader task: reads result frames off the stream, decodes
/// them, and dispatches via the existing `pending_layer_results` oneshot map.
async fn reader_task_outbound<R>(
    mut read: R,
    request_id: Uuid,
    peer_id: PeerId,
    shared_state: Arc<SharedState>,
) where
    R: futures::AsyncRead + Unpin,
{
    // Results on this stream can only come from the peer it was opened to, so
    // every resolve below is attributed to that peer — a waiter that has failed
    // over to a standby is not satisfied by this stream's traffic.
    let sender = shared_state.peer_to_node_id_from_registry(&peer_id);
    loop {
        let frame = match read_frame(&mut read).await {
            Ok(f) => f,
            Err(e) => {
                tracing::debug!(
                    %request_id,
                    error = %e,
                    "pipeline stream reader terminated"
                );
                // Evict stale result channel so the pipeline fails fast instead
                // of waiting for the adaptive stale-tensor cleanup.
                shared_state.resolve_pending_layer_result(
                    sender.as_ref(),
                    LayerResult::error(request_id, format!("pipeline stream closed: {e}")),
                );
                return;
            }
        };

        let tag = frame.first().copied().unwrap_or(0);
        if tag != TENSOR_TAG_RESULT {
            tracing::warn!(
                %request_id,
                tag,
                "pipeline stream reader: unexpected frame tag (expected RESULT)"
            );
            continue;
        }

        match protocol::decode_layer_result(&frame) {
            Ok(result) => {
                let rid = result.request_id;
                if !shared_state.resolve_pending_layer_result(sender.as_ref(), result) {
                    // Common after R136 L2 hedging — when a hedge race
                    // resolves, the loser's response arrives here with no
                    // pending oneshot (the guard drop already evicted the
                    // map entry). Trace-level so it doesn't pollute -v / -vv
                    // logs with what is normal operation.
                    tracing::trace!(
                        %rid,
                        "pipeline stream reader: no pending oneshot (late result or hedge loser)"
                    );
                }
            }
            Err(e) => {
                tracing::warn!(
                    %request_id,
                    error = %e,
                    "pipeline stream reader: decode_layer_result failed"
                );
            }
        }
    }
}

/// Spawn the remote-side accept loop. Each inbound stream gets its own
/// handler task. Returns immediately; the accept task runs in the background
/// until the `IncomingStreams` handle is dropped or shutdown is signaled.
///
/// R107: the body is wrapped in `FutureExt::catch_unwind` so that a panic
/// inside `incoming.next()` (e.g. a libp2p_stream invariant violation) is
/// logged at `error!` instead of silently terminating the entire inbound
/// pipeline path for the rest of the daemon's lifetime. The handle the
/// caller drops cannot signal a panic upward; without this wrapper a
/// healthy-looking daemon would silently refuse new inbound pipeline
/// streams.
pub fn spawn_accept_loop(
    mut incoming: IncomingStreams,
    shared_state: Arc<SharedState>,
    outbound_tx: mpsc::Sender<crate::types::AuthenticatedMessage>,
    mut shutdown_rx: tokio::sync::watch::Receiver<bool>,
) -> JoinHandle<()> {
    use futures::FutureExt;
    tokio::spawn(async move {
        tracing::info!(
            protocol = PROTOCOL_PIPELINE,
            "pipeline stream accept loop started"
        );
        let result = std::panic::AssertUnwindSafe(async {
            loop {
                tokio::select! {
                    biased;
                    _ = shutdown_rx.changed() => {
                        tracing::debug!("pipeline accept loop: shutdown observed");
                        break;
                    }
                    next = incoming.next() => match next {
                        Some((peer_id, stream)) => {
                            let state = shared_state.clone();
                            let out_tx = outbound_tx.clone();
                            let stream_shutdown = shutdown_rx.clone();
                            tokio::spawn(async move {
                                handle_inbound_stream(peer_id, stream, state, out_tx, stream_shutdown).await;
                            });
                        }
                        None => break,
                    },
                }
            }
        })
        .catch_unwind()
        .await;
        if result.is_err() {
            tracing::error!(
                protocol = PROTOCOL_PIPELINE,
                "pipeline stream accept loop panicked — inbound pipeline acceptance has stopped"
            );
        }
        tracing::info!("pipeline stream accept loop terminated");
    })
}

/// Remote-side per-stream handler. Reads forward frames sequentially,
/// dispatches each via the existing authenticated dispatch path, awaits the
/// result through `pending_stream_result_routes`, and writes the result frame
/// back on the same stream.
async fn handle_inbound_stream(
    peer_id: PeerId,
    stream: libp2p::Stream,
    shared_state: Arc<SharedState>,
    outbound_tx: mpsc::Sender<crate::types::AuthenticatedMessage>,
    mut shutdown_rx: tokio::sync::watch::Receiver<bool>,
) {
    let (mut read, mut write) = stream.split();

    // SEC: refuse streams from peers that haven't completed Identify yet.
    // The libp2p connection is established as soon as Noise+Yamux handshake
    // finishes — *before* `peer_registry` is populated by the Identify
    // event. Accepting frames at that point lets an unauthenticated peer
    // (a) leak `pending_stream_result_routes` entries (each frame inserts a
    // new oneshot keyed by attacker-chosen UUID), and (b) hold a Tokio task
    // forever per UUID waiting on `rx.await` because the dispatch path
    // drops messages with `sender = None`. Wait for the registry entry
    // before serving anything; if it never arrives, the stream times out
    // on the connection-idle path naturally.
    if shared_state
        .peer_to_node_id_from_registry(&peer_id)
        .is_none()
    {
        tracing::debug!(%peer_id, "pipeline stream from unregistered peer — closing");
        let _ = write.close().await;
        return;
    }
    tracing::info!(%peer_id, "pipeline stream handler started");

    loop {
        let frame = tokio::select! {
            biased;
            _ = shutdown_rx.changed() => {
                tracing::debug!(%peer_id, "pipeline stream handler: shutdown observed");
                let _ = write.close().await;
                return;
            }
            res = read_frame(&mut read) => match res {
                Ok(f) => f,
                Err(e) => {
                    tracing::debug!(%peer_id, error = %e, "pipeline stream handler terminated");
                    let _ = write.close().await;
                    return;
                }
            },
        };

        // Decode + decrypt (mirrors the handle_tensor_payload logic for
        // TENSOR_TAG_FORWARD / TENSOR_TAG_ENCRYPTED).
        let tag = frame.first().copied().unwrap_or(0);
        let (forward, request_id) =
            match decode_inbound_forward(&frame, tag, &peer_id, &shared_state) {
                Some(pair) => pair,
                None => continue,
            };

        // R139 Tier 4K — if this is a chunked transfer, route the chunk
        // through the assembly state. Only the final-chunk completion
        // dispatches to the worker; intermediate chunks accumulate.
        let forward = if forward.chunk_meta.is_some() {
            let peer_bytes = peer_id.to_bytes();
            match shared_state.try_assemble_chunked_forward(forward, peer_bytes) {
                Ok(Some(complete)) => complete,
                Ok(None) => continue,
                Err(e) => {
                    tracing::warn!(
                        error = %e,
                        %request_id,
                        "Chunk assembly rejected on stream — dropping forward"
                    );
                    continue;
                }
            }
        } else {
            forward
        };

        // Register the result route BEFORE dispatching so we never race the
        // dispatcher's SendTensorResult against our own insertion.
        //
        // SEC: an RAII guard removes the entry on ALL exit paths — including
        // when the future is dropped mid-`outbound_tx.send().await` (task
        // abort during shutdown, supervisor `abort_all`). Without this guard
        // the explicit `remove` calls only ran on `is_err` / `Err(_)` arms;
        // a cancellation between insert and send completion leaked the
        // oneshot Sender into the map indefinitely.
        let (tx, rx) = oneshot::channel();
        shared_state
            .pending_stream_result_routes
            .insert(request_id, tx);
        struct StreamRouteGuard<'a> {
            state: &'a Arc<SharedState>,
            request_id: uuid::Uuid,
            armed: bool,
        }
        impl Drop for StreamRouteGuard<'_> {
            fn drop(&mut self) {
                if self.armed {
                    self.state
                        .pending_stream_result_routes
                        .remove(&self.request_id);
                }
            }
        }
        let mut route_guard = StreamRouteGuard {
            state: &shared_state,
            request_id,
            armed: true,
        };

        // Stamp the authenticated sender and dispatch via the normal path.
        let auth = crate::types::AuthenticatedMessage {
            sender: shared_state.peer_to_node_id_from_registry(&peer_id),
            message: SwarmMessage::LayerForward(forward),
        };
        if outbound_tx.send(auth).await.is_err() {
            // Guard drop will remove the entry.
            tracing::warn!(%peer_id, %request_id, "dispatch channel closed");
            let _ = write.close().await;
            return;
        }

        // Await the result produced by the dispatcher (delivered by
        // `try_deliver_stream_result` from the network manager).
        let result = match rx.await {
            Ok(r) => {
                // Successful delivery — disarm guard, dispatcher already
                // consumed the entry.
                route_guard.armed = false;
                r
            }
            Err(_) => {
                tracing::warn!(%peer_id, %request_id, "result oneshot dropped");
                // Guard drop will clean up.
                continue;
            }
        };

        // Encode result and write frame back.
        let encoded = match protocol::encode_layer_result(&result) {
            Ok(bytes) => bytes,
            Err(e) => {
                tracing::warn!(error = %e, %request_id, "encode_layer_result failed");
                continue;
            }
        };
        if let Err(e) = write_frame(&mut write, &encoded).await {
            tracing::debug!(%peer_id, %request_id, error = %e, "pipeline stream write failed");
            let _ = write.close().await;
            return;
        }
    }
}

/// Decode an inbound forward frame (plaintext TENSOR_TAG_FORWARD or sealed
/// TENSOR_TAG_ENCRYPTED). Returns `None` and logs on decode errors.
fn decode_inbound_forward(
    frame: &[u8],
    tag: u8,
    peer_id: &PeerId,
    shared_state: &SharedState,
) -> Option<(LayerForward, Uuid)> {
    match tag {
        TENSOR_TAG_FORWARD => match protocol::decode_layer_forward(frame) {
            Ok(mut forward) => {
                let rid = forward.request_id;
                forward.sender_peer_bytes = Some(peer_id.to_bytes());
                Some((forward, rid))
            }
            Err(e) => {
                tracing::warn!(%peer_id, error = %e, "decode_layer_forward failed");
                None
            }
        },
        TENSOR_TAG_ENCRYPTED => {
            let (mut forward, sealed, aad) = match protocol::decode_layer_forward_encrypted(frame) {
                Ok(v) => v,
                Err(e) => {
                    tracing::warn!(%peer_id, error = %e, "decode_layer_forward_encrypted failed");
                    return None;
                }
            };
            let node_id = shared_state.peer_to_node_id_from_registry(peer_id)?;
            match shared_state.session_manager.open(&node_id, &sealed, &aad) {
                Ok(plaintext) => {
                    let rid = forward.request_id;
                    forward.activations = plaintext;
                    forward.sender_peer_bytes = Some(peer_id.to_bytes());
                    Some((forward, rid))
                }
                Err(e) => {
                    tracing::warn!(%peer_id, %node_id, error = %e, "session open() failed on pipeline stream");
                    None
                }
            }
        }
        other => {
            tracing::warn!(%peer_id, tag = other, "unexpected forward frame tag");
            None
        }
    }
}

/// R139 Tier 4K — split a `LayerForward` carrying a large activation tensor
/// into K chunks of at most `chunk_size_bytes` for STREAM-style chunked
/// transport. The activation byte buffer is split at byte-offset boundaries;
/// each emitted `LayerForward` carries a `ChunkMeta { chunk_idx, total_chunks }`
/// that the AAD helper binds into the Poly1305 tag (single source of truth via
/// `build_layer_forward_aad`). Cleartext metadata (request_id, layer_range,
/// model_id, etc.) is duplicated across all K frames — receivers assemble by
/// `request_id`.
///
/// Returns the input verbatim wrapped in a single-element vec when the
/// activation is at or below `chunk_size_bytes` (single-chunk implicit
/// fallback — caller can ship as today's monolithic frame without the
/// chunk-meta trailer).
///
/// Chunk-size policy and rationale: 256 KiB default (matches age STREAM
/// construction + TokenWeave MLSys 2026 K=2-4 sweet spot); 64 KiB floor;
/// SwarmLLM's `inference.streaming_chunk_size_bytes` /
/// `streaming_min_activation_bytes` knobs override these. See
/// `docs/FUTURE_WORK.md § Tier 4K`.
pub fn chunk_layer_forward(forward: &LayerForward, chunk_size_bytes: usize) -> Vec<LayerForward> {
    let total = forward.activations.len();
    let chunk_size = chunk_size_bytes.max(1);
    if total <= chunk_size {
        return vec![forward.clone()];
    }
    let total_chunks = total.div_ceil(chunk_size);
    // u32 wire-format cap: chunk_idx and total_chunks are encoded as u32 LE.
    let total_chunks = total_chunks.min(u32::MAX as usize) as u32;
    let mut out = Vec::with_capacity(total_chunks as usize);
    for idx in 0..total_chunks {
        let start = (idx as usize) * chunk_size;
        let end = ((idx as usize + 1) * chunk_size).min(total);
        let mut chunk = forward.clone();
        chunk.activations = forward.activations[start..end].to_vec();
        chunk.chunk_meta = Some(crate::types::ChunkMeta {
            chunk_idx: idx,
            total_chunks,
        });
        out.push(chunk);
    }
    out
}

/// Encode a `LayerForward` into wire bytes suitable for a pipeline-stream
/// frame. Applies ChaCha sealing when encryption is enabled and a session
/// exists for the target peer. Mirrors the logic in
/// `NetworkManager::handle_send_tensor` so the RR and stream paths produce
/// byte-identical frames.
pub fn encode_forward_for_wire(
    forward: &LayerForward,
    peer_id: &PeerId,
    shared_state: &SharedState,
) -> Result<Vec<u8>, SwarmError> {
    let peer_node_id = shared_state.peer_to_node_id_from_registry(peer_id);
    let use_encryption = shared_state.config.network.enable_encryption && peer_node_id.is_some();

    if use_encryption {
        let node_id = peer_node_id.ok_or_else(|| {
            SwarmError::Network("encryption requested but no NodeId for peer".into())
        })?;
        // Single source of truth — `protocol::build_layer_forward_aad` is
        // shared with the RR encrypt path (`network/manager/tensors.rs`)
        // and the decrypt path (`decode_layer_forward_encrypted`). Inline
        // construction here would drift the moment a new authenticated
        // field is added to LayerForward.
        let aad = protocol::build_layer_forward_aad(forward);
        let sealed = shared_state
            .session_manager
            .seal(&node_id, &forward.activations, &aad)
            .map_err(|e| SwarmError::Network(format!("seal failed: {e}")))?;
        protocol::encode_layer_forward_encrypted(forward, sealed)
            .map_err(|e| SwarmError::Network(format!("encrypted encode: {e}")))
    } else {
        protocol::encode_layer_forward(forward)
            .map_err(|e| SwarmError::Network(format!("forward encode: {e}")))
    }
}

/// Write a length-prefixed frame: `[len:4 LE][payload]`.
async fn write_frame<W: futures::AsyncWrite + Unpin>(
    w: &mut W,
    payload: &[u8],
) -> std::io::Result<()> {
    if payload.len() > MAX_FRAME_LEN {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("frame too large: {}", payload.len()),
        ));
    }
    let len = (payload.len() as u32).to_le_bytes();
    w.write_all(&len).await?;
    w.write_all(payload).await?;
    w.flush().await
}

/// Read one length-prefixed frame. Returns the payload bytes (first byte is
/// the TENSOR_TAG_* type tag handled by the existing codec).
async fn read_frame<R: futures::AsyncRead + Unpin>(r: &mut R) -> std::io::Result<Vec<u8>> {
    let mut len_buf = [0u8; 4];
    r.read_exact(&mut len_buf).await?;
    let len = u32::from_le_bytes(len_buf) as usize;
    if len == 0 || len > MAX_FRAME_LEN {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("invalid frame length: {len}"),
        ));
    }
    let mut payload = vec![0u8; len];
    r.read_exact(&mut payload).await?;
    Ok(payload)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn frame_roundtrip() {
        let payload = b"hello pipeline world".to_vec();
        let mut buf: Vec<u8> = Vec::new();
        {
            let mut cursor = futures::io::Cursor::new(&mut buf);
            write_frame(&mut cursor, &payload).await.unwrap();
        }
        let mut cursor = futures::io::Cursor::new(&buf[..]);
        let out = read_frame(&mut cursor).await.unwrap();
        assert_eq!(out, payload);
    }

    #[tokio::test]
    async fn frame_rejects_oversized_length() {
        let mut buf: Vec<u8> = Vec::new();
        buf.extend_from_slice(&(MAX_FRAME_LEN as u32 + 1).to_le_bytes());
        let mut cursor = futures::io::Cursor::new(&buf[..]);
        let err = read_frame(&mut cursor).await.expect_err("should reject");
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    }

    // --- R139 A-rev: chunk_layer_forward splitter ------------------------

    fn base_layer_forward(activations: Vec<u8>) -> LayerForward {
        LayerForward {
            request_id: uuid::Uuid::from_u128(0xABCD_1234_5678_9ABC_DEF0_1122_3344_5566),
            sequence_num: 1,
            index_pos: 0,
            activations,
            format: crate::types::TensorFormat::FP32,
            model_id: crate::types::ModelId("test-model".into()),
            layer_range: (0, 8),
            tp_meta: None,
            vision_embeddings: None,
            chain: Vec::new(),
            sender_peer_bytes: None,
            requester_node_id: None,
            pre_embedded: false,
            generated_ids: Vec::new(),
            adapter_id: None,
            draft_tokens: Vec::new(),
            spec_logits_requested: false,
            truncate_kv_to: None,
            chunk_meta: None,
        }
    }

    #[test]
    fn chunk_layer_forward_passthrough_when_under_threshold() {
        let forward = base_layer_forward(vec![0u8; 1024]);
        let chunks = chunk_layer_forward(&forward, 2048);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].activations, forward.activations);
        assert!(chunks[0].chunk_meta.is_none());
    }

    #[test]
    fn chunk_layer_forward_passthrough_at_exact_size() {
        let forward = base_layer_forward(vec![0u8; 2048]);
        let chunks = chunk_layer_forward(&forward, 2048);
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].chunk_meta.is_none());
    }

    #[test]
    fn chunk_layer_forward_splits_into_k_equal_chunks() {
        let mut activations = Vec::with_capacity(8 * 1024);
        for i in 0..(8 * 1024) {
            activations.push((i & 0xFF) as u8);
        }
        let forward = base_layer_forward(activations.clone());
        let chunks = chunk_layer_forward(&forward, 2048);
        assert_eq!(chunks.len(), 4);
        for (idx, c) in chunks.iter().enumerate() {
            let cm = c.chunk_meta.expect("chunk should have chunk_meta");
            assert_eq!(cm.chunk_idx as usize, idx);
            assert_eq!(cm.total_chunks, 4);
            assert_eq!(c.activations.len(), 2048);
            assert_eq!(c.request_id, forward.request_id);
        }
        // Reassembling must equal the original.
        let reassembled: Vec<u8> = chunks
            .iter()
            .flat_map(|c| c.activations.iter().copied())
            .collect();
        assert_eq!(reassembled, activations);
    }

    #[test]
    fn chunk_layer_forward_handles_uneven_final_chunk() {
        let activations = vec![0xCDu8; 5000];
        let forward = base_layer_forward(activations.clone());
        let chunks = chunk_layer_forward(&forward, 2048);
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].activations.len(), 2048);
        assert_eq!(chunks[1].activations.len(), 2048);
        assert_eq!(chunks[2].activations.len(), 904);
        assert!(chunks[2].chunk_meta.unwrap().is_final());
    }
}
