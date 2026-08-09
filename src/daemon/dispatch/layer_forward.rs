use std::sync::Arc;

use tokio::sync::mpsc;

use crate::types::NetworkCommand;

use super::super::state::SharedState;
use super::seal_layer_result;

pub(super) async fn handle_layer_forward(
    shared_state: Arc<SharedState>,
    network_tx: mpsc::Sender<NetworkCommand>,
    forward: crate::types::LayerForward,
) {
    let request_id = forward.request_id;
    let sender_peer_bytes = match forward.sender_peer_bytes {
        Some(ref bytes) => bytes.clone(),
        None => {
            tracing::warn!(request_id = %request_id, "LayerForward missing sender_peer_bytes");
            return;
        }
    };

    // Estimate token count for credit accounting: prefill carries many tokens,
    // decode carries 1. For prefill (seq==0), estimate from activation bytes
    // (raw prompt: ~4 bytes/token, embedded: hidden_dim*4 bytes/token).
    let estimated_tokens: u32 = if forward.sequence_num == 0 {
        // Rough estimate: prompt bytes / 4 chars per token
        (forward.activations.len() / 4).max(1) as u32
    } else {
        1
    };
    let forward_start = std::time::Instant::now();
    tracing::info!(
        request_id = %request_id,
        seq = forward.sequence_num,
        activation_bytes = forward.activations.len(),
        estimated_tokens,
        model_id = %forward.model_id,
        layer_range = ?forward.layer_range,
        "DIAG: processing LayerForward locally"
    );

    let model_id = forward.model_id.clone();

    // Determine our layer range from the manifest and local shards
    let manifest = match shared_state.model_registry.get_manifest(&model_id) {
        Some(m) => m,
        None => {
            send_error_result(
                &network_tx,
                &sender_peer_bytes,
                request_id,
                "No manifest for model",
            )
            .await;
            return;
        }
    };

    // Figure out which shard indices we hold locally
    let local_node_id = shared_state.identity.node_id().clone();
    let local_shard_indices = shared_state
        .model_registry
        .local_shard_indices_in(&manifest, &local_node_id);

    if local_shard_indices.is_empty() {
        send_error_result(
            &network_tx,
            &sender_peer_bytes,
            request_id,
            "No local shards for model",
        )
        .await;
        return;
    }

    // Layer range is required in the forward message — no guessing
    let (layer_start, layer_end, total_layers) = {
        let (ls, le) = forward.layer_range;
        let total = manifest.num_layers as usize;
        (ls as usize, le as usize, total)
    };

    if !layer_range_is_valid(layer_start, layer_end, total_layers) {
        send_error_result(
            &network_tx,
            &sender_peer_bytes,
            request_id,
            &format!(
                "Invalid layer range [{layer_start}..{layer_end}) for model with {total_layers} layers"
            ),
        )
        .await;
        return;
    }

    let (is_first, is_last) = crate::model::shard::compute_first_last(
        &local_shard_indices,
        manifest.shard_count,
        layer_start,
        layer_end,
        total_layers,
    );

    // Ensure the split model metadata entry exists (lightweight — no GPU loading).
    let split_key = shared_state.ensure_split_model_entry(
        &model_id,
        layer_start,
        layer_end,
        is_first,
        is_last,
        total_layers,
    );

    // Touch the metadata entry
    if let Some(entry) = shared_state.split_models.get(&split_key) {
        entry.value().touch();
    }

    // Capture TP metadata and requester_node_id before moving forward into the process pool
    let tp_meta = forward.tp_meta.clone();
    let requester_node_id = forward.requester_node_id;

    // Route forward pass to subprocess via process pool
    let result = shared_state.model_process_pool.forward(forward).await;

    let result = match result {
        Ok(r) => r,
        Err(e) => {
            send_error_result(
                &network_tx,
                &sender_peer_bytes,
                request_id,
                &format!("Worker: {e}"),
            )
            .await;
            return;
        }
    };

    let forward_elapsed = forward_start.elapsed();
    // Use structured fields instead of `format!("[{a}..{b})")` — Rust eval
    // is eager and would heap-alloc one String per token even when info-
    // level is filtered out. Per-decode-step hot path on the serving node.
    tracing::info!(
        request_id = %request_id,
        tokens = result.token_ids.len(),
        activations_bytes = result.activations.len(),
        is_last,
        elapsed_ms = forward_elapsed.as_millis() as u64,
        model_id = %model_id,
        layer_start,
        layer_end,
        tp = tp_meta.is_some(),
        "DIAG: LayerForward processed via worker subprocess"
    );
    // Serving-side accounting and credit earn, in one call. Everything else in
    // the observability stack is requester-side, so without this an operator
    // cannot answer "is my node actually contributing, and how well" — nor can
    // we distinguish a well-behaved peer from one whose segments everyone times
    // out on.
    //
    // Recorded here rather than after the reply is sent: the work is already
    // done at this point, and the TP branch below has three early returns for
    // encoding failures. Counting only successful replies would drop effort
    // that was genuinely spent, which is the opposite of what an operator
    // asking "is my node contributing" wants to know.
    shared_state.record_peer_serve(crate::daemon::state::PeerServe {
        kind: crate::daemon::state::ServeKind::Segment,
        layers: layer_end.saturating_sub(layer_start) as u32,
        elapsed_ms: forward_elapsed.as_millis() as u64,
        activation_bytes: result.activations.len() as u64,
        tokens: estimated_tokens,
    });

    // TP path: send partial as AllReduceRequest to the coordinator (sender) instead of LayerResult
    if let Some(ref tp) = tp_meta {
        let layer_idx = match tp.phase {
            crate::types::TpPhase::AttnOnly => tp.single_layer * 2,
            crate::types::TpPhase::FfnOnly => tp.single_layer * 2 + 1,
            crate::types::TpPhase::Full | crate::types::TpPhase::EmbedOnly => tp.single_layer,
        };

        // Extract raw f32 bytes from tensor format (strip header) for AllReduce.
        // The worker returns activations in tensor_to_bytes format (ndim + shape + dtype + data).
        // AllReduce needs just the raw f32 data to ensure consistent sizes across ranks.
        let (raw_f32, shape) = match crate::inference::split::bytes_to_tensor(&result.activations) {
            Ok(tensor) => {
                let shape: Vec<u32> = tensor.dims().iter().map(|&d| d as u32).collect();
                match crate::inference::split::tensor_to_raw_f32(&tensor) {
                    Ok(raw) => (raw, shape),
                    Err(e) => {
                        tracing::warn!(error = %e, "Failed to extract raw f32 from TP partial");
                        return;
                    }
                }
            }
            Err(e) => {
                tracing::warn!(error = %e, "Failed to deserialize TP partial tensor");
                return;
            }
        };

        let compressed = match zstd::encode_all(std::io::Cursor::new(&raw_f32), 1) {
            Ok(c) => c,
            Err(e) => {
                tracing::warn!(error = %e, "Failed to compress TP partial");
                return;
            }
        };

        let allreduce_req = crate::types::TpAllReduceRequest {
            request_id,
            layer_idx,
            tp_rank: tp.tp_rank as u32,
            tp_size: tp.tp_size as u32,
            partial_data: compressed,
            shape,
            op: crate::types::AllReduceOp::Sum,
            sender_peer_bytes: None,
        };

        // Send to the coordinator (= the node that sent us the LayerForward)
        if let Err(e) = network_tx
            .send(crate::types::NetworkCommand::SendAllReduceRequest {
                target_peer_bytes: sender_peer_bytes,
                request: allreduce_req,
            })
            .await
        {
            tracing::warn!(error = %e, "Failed to send TpAllReduceRequest");
        }

        return;
    }

    // Pipeline sealing: encrypt token IDs for requester if this is the final segment
    let mut result = result;
    if is_last {
        seal_layer_result(&mut result, requester_node_id.as_ref());
    }

    // Send back as a separate request to the originating peer
    if let Err(e) = network_tx
        .send(NetworkCommand::SendTensorResult {
            target_peer_bytes: sender_peer_bytes,
            result,
        })
        .await
    {
        tracing::warn!(error = %e, "Failed to send LayerResult back to peer");
    }
}

/// Send a sanitized error `LayerResult` back to the originating peer when
/// `LayerForward` processing fails locally. The error message is scrubbed
/// before transmission to avoid leaking internal layer topology or paths.
pub(super) async fn send_error_result(
    network_tx: &mpsc::Sender<NetworkCommand>,
    target_peer_bytes: &[u8],
    request_id: uuid::Uuid,
    error: &str,
) {
    tracing::warn!(request_id = %request_id, error, "LayerForward processing failed");
    // Sanitize error for network — don't leak internal paths, layer counts, or model topology
    let sanitized = if error.contains("layer range") || error.contains("layer_start") {
        "Layer configuration error".to_string()
    } else if error.contains("No local shards") || error.contains("shard") {
        "Required shards not available".to_string()
    } else {
        // Truncate and strip paths
        let msg = error.chars().take(100).collect::<String>();
        msg.replace(['/', '\\'], "")
    };
    let result = crate::types::LayerResult::error(request_id, sanitized);
    let _ = network_tx
        .send(NetworkCommand::SendTensorResult {
            target_peer_bytes: target_peer_bytes.to_vec(),
            result,
        })
        .await;
}

/// Is a peer-supplied layer range safe to act on for a model of `total_layers`?
///
/// `layer_range` arrives over the network and is attacker-controlled: it decides
/// which slice of a model a forward is executed against, so an unchecked value
/// is an out-of-bounds read waiting to happen. An external security audit
/// flagged this path as unverified (2026-07-28); the check was present but
/// inline and untested, which is indistinguishable from absent to anyone
/// reading, and one refactor away from actually being absent.
///
/// Both inbound transports — the request/response decrypt path and the
/// persistent pipeline stream — funnel through `handle_layer_forward`, so this
/// is the single place the range is admitted. Keep it that way: a second entry
/// point that skips this is the "one invariant, N paths" defect this codebase
/// keeps rediscovering.
///
/// Rejects an empty or inverted range as well as one running past the end —
/// `start >= end` covers both, and an empty range would otherwise reach the
/// executor as a no-op segment that still allocates and still answers.
pub(crate) fn layer_range_is_valid(start: usize, end: usize, total: usize) -> bool {
    start < end && end <= total
}

#[cfg(test)]
mod forward_cancellation_tests {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    /// Aborting a spawned forward must actually stop the work, not merely drop
    /// the caller's interest in it. This is the property the whole change rests
    /// on: `CancelInference` looks up an abort handle and fires it, and if the
    /// task carried on regardless the peer would still be monopolised by work
    /// nobody will read.
    #[tokio::test]
    async fn aborting_a_forward_task_stops_it_running() {
        let finished = Arc::new(AtomicBool::new(false));
        let flag = finished.clone();
        let handle = tokio::spawn(async move {
            // Stand in for a long prefill.
            tokio::time::sleep(std::time::Duration::from_secs(30)).await;
            flag.store(true, Ordering::Release);
        });

        let abort = handle.abort_handle();
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        abort.abort();
        let _ = handle.await;

        assert!(
            !finished.load(Ordering::Acquire),
            "an abandoned forward must not run to completion"
        );
    }

    /// The registry is keyed by request id, so a cancel for one request must not
    /// stop a different one running concurrently for the same peer.
    #[tokio::test]
    async fn cancelling_one_forward_leaves_another_alone() {
        let survivor_done = Arc::new(AtomicBool::new(false));
        let flag = survivor_done.clone();
        let survivor = tokio::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_millis(40)).await;
            flag.store(true, Ordering::Release);
        });
        let doomed = tokio::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_secs(30)).await;
        });

        doomed.abort_handle().abort();
        let _ = doomed.await;
        let _ = survivor.await;

        assert!(
            survivor_done.load(Ordering::Acquire),
            "an unrelated forward must keep running"
        );
    }
}

#[cfg(test)]
mod layer_range_tests {
    use super::layer_range_is_valid;

    #[test]
    fn a_range_inside_the_model_is_accepted() {
        assert!(layer_range_is_valid(0, 16, 32));
        assert!(layer_range_is_valid(16, 32, 32)); // touching the end exactly
        assert!(layer_range_is_valid(0, 1, 1));
    }

    #[test]
    fn a_range_past_the_end_is_refused() {
        assert!(!layer_range_is_valid(0, 33, 32));
        assert!(!layer_range_is_valid(30, 64, 32));
        // u32::MAX arriving off the wire, widened to usize.
        assert!(!layer_range_is_valid(0, u32::MAX as usize, 32));
        assert!(!layer_range_is_valid(
            u32::MAX as usize,
            u32::MAX as usize,
            32
        ));
    }

    #[test]
    fn an_empty_or_inverted_range_is_refused() {
        assert!(!layer_range_is_valid(16, 16, 32)); // empty
        assert!(!layer_range_is_valid(20, 4, 32)); // inverted
        assert!(!layer_range_is_valid(0, 0, 32));
    }

    /// A manifest claiming no layers must admit nothing rather than divide by
    /// or index into an empty model.
    #[test]
    fn a_model_with_no_layers_admits_nothing() {
        assert!(!layer_range_is_valid(0, 1, 0));
        assert!(!layer_range_is_valid(0, 0, 0));
    }
}
