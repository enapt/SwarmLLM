use std::sync::Arc;

use tokio::sync::mpsc;

use crate::types::NetworkCommand;

use super::super::state::SharedState;
use super::{seal_layer_result, track_forward_participation};

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
    let mut local_shard_indices: Vec<u32> = Vec::new();
    for shard_info in &manifest.shards {
        let shard_id = crate::types::ShardId {
            model_id: model_id.clone(),
            index: shard_info.index,
        };
        let holders = shared_state.model_registry.shard_holders(&shard_id);
        if holders.contains(&local_node_id) {
            local_shard_indices.push(shard_info.index);
        }
    }

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

    if layer_start >= layer_end || layer_end > total_layers {
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
    tracing::info!(
        request_id = %request_id,
        tokens = result.token_ids.len(),
        activations_bytes = result.activations.len(),
        is_last,
        elapsed_ms = forward_elapsed.as_millis() as u64,
        model_id = %model_id,
        layers = format!("[{layer_start}..{layer_end})"),
        tp = tp_meta.is_some(),
        "DIAG: LayerForward processed via worker subprocess"
    );

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

        track_forward_participation(&shared_state, estimated_tokens);
        return;
    }

    track_forward_participation(&shared_state, estimated_tokens);

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

/// Handle a VisionEncodeRequest: encode the image using local mmproj and respond.
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
    let result = crate::types::LayerResult {
        request_id,
        token_ids: vec![],
        finish_reason: Some(crate::types::NetworkFinishReason::Error(sanitized)),
        activations: vec![],
        sealed_token_ids: None,
        spec_logits: Vec::new(),
    };
    let _ = network_tx
        .send(NetworkCommand::SendTensorResult {
            target_peer_bytes: target_peer_bytes.to_vec(),
            result,
        })
        .await;
}
