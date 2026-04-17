//! Tensor-parallel segment execution — 2-phase AllReduce coordination.
//!
//! Carved out of pipeline.rs to keep the segment + per-token hot path
//! (forward_through_segments, process_local_segment, wait_for_result)
//! separate from the AllReduce protocol. The function is a private method
//! on PipelineExecutor and is only called from forward_through_segments.

use crate::error::SwarmError;
use crate::inference::split;
use crate::types::{NetworkCommand, PipelineSegment};

use super::PipelineExecutor;

impl PipelineExecutor {
    /// Execute a tensor-parallel segment.
    ///
    /// The local node's TP computation is done inline; remote nodes receive
    /// LayerForward messages with TensorParallelMeta.
    #[allow(clippy::too_many_arguments)]
    pub(super) async fn execute_tp_segment(
        &self,
        request_id: uuid::Uuid,
        sequence_num: u32,
        index_pos: usize,
        activation_bytes: &[u8],
        segment: &PipelineSegment,
        tp_group: &crate::types::TensorParallelGroup,
        _is_last: bool,
    ) -> Result<Vec<u8>, SwarmError> {
        let model_id = &segment.shard_id.model_id;
        let (layer_start, layer_end) = (
            segment.layer_range.0 as usize,
            segment.layer_range.1 as usize,
        );
        let local_node_id = self.shared_state.identity.node_id().clone();
        let tp_size = tp_group.tp_size();

        // Find our rank in the TP group
        let local_tp_rank = tp_group.rank_of(&local_node_id);

        tracing::info!(
            request_id = %request_id,
            tp_size,
            local_rank = ?local_tp_rank,
            layers = ?(layer_start..layer_end),
            "Starting tensor-parallel segment execution"
        );

        let split_key = self.ensure_split_model_entry(model_id, layer_start, layer_end)?;

        // Touch the metadata entry
        if let Some(entry) = self.shared_state.split_models.get(&split_key) {
            entry.value().touch();
        }

        // TP execution: layer-by-layer with AllReduce coordination in main process.
        // Each layer's partial computation is routed to the worker subprocess via
        // LayerForward with tp_meta set. Remote TP peers receive the same LayerForward
        // over the network and send their partials as TpAllReduceRequests to us (rank 0).
        if let Some(tp_rank) = local_tp_rank {
            // We are in the TP group — per-layer partial computation via subprocess + AllReduce

            // Resolve PeerId bytes for each remote TP peer (needed to send LayerForward)
            let remote_tp_peers: Vec<(usize, Vec<u8>)> = tp_group
                .nodes
                .iter()
                .enumerate()
                .filter(|(rank, _)| *rank != tp_rank)
                .filter_map(|(rank, node_id)| {
                    self.shared_state
                        .peer_id_map
                        .get(node_id)
                        .map(|r| (rank, r.value().clone()))
                        .or_else(|| {
                            self.shared_state
                                .peer_registry
                                .get(node_id)
                                .and_then(|p| p.peer_id_bytes.clone().map(|b| (rank, b)))
                        })
                })
                .collect();

            if remote_tp_peers.len() != tp_size - 1 {
                return Err(SwarmError::Network(format!(
                    "Cannot resolve PeerId for all TP peers (got {}, need {})",
                    remote_tp_peers.len(),
                    tp_size - 1
                )));
            }

            // Pre-embed the prompt so we have tensor-format hidden states for residual management.
            // The EmbedOnly phase tokenizes + embeds without running any layers.
            let embedded = {
                let embed_fwd = crate::types::LayerForward {
                    request_id,
                    sequence_num,
                    index_pos: index_pos as u32,
                    activations: activation_bytes.to_vec(),
                    format: crate::types::TensorFormat::FP16,
                    model_id: model_id.clone(),
                    layer_range: (layer_start as u32, layer_end as u32),
                    tp_meta: Some(crate::types::TensorParallelMeta {
                        tp_rank: tp_rank as u8,
                        tp_size: tp_size as u8,
                        single_layer: 0,
                        phase: crate::types::TpPhase::EmbedOnly,
                    }),
                    vision_embeddings: None,
                    sender_peer_bytes: None,
                    requester_node_id: None,
                    pre_embedded: false,
                    adapter_id: None,
                    draft_tokens: Vec::new(),
                    spec_logits_requested: false,
                    truncate_kv_to: None,
                };
                self.shared_state
                    .model_process_pool
                    .forward(embed_fwd)
                    .await?
            };
            let mut current_activations_bytes = embedded.activations;

            for abs_layer in layer_start..layer_end {
                // 2-phase TP protocol: AttnOnly → AllReduce → residual add → FfnOnly → AllReduce → residual add
                // Residuals are managed here (not in the worker) since AllReduce must complete before add.

                // --- Send AttnOnly LayerForward to remote TP peers FIRST (so they start in parallel) ---
                for (rank, peer_bytes) in &remote_tp_peers {
                    let remote_forward = crate::types::LayerForward {
                        request_id,
                        sequence_num,
                        index_pos: index_pos as u32,
                        activations: current_activations_bytes.clone(),
                        format: crate::types::TensorFormat::FP16,
                        model_id: model_id.clone(),
                        layer_range: (layer_start as u32, layer_end as u32),
                        tp_meta: Some(crate::types::TensorParallelMeta {
                            tp_rank: *rank as u8,
                            tp_size: tp_size as u8,
                            single_layer: abs_layer as u32,
                            phase: crate::types::TpPhase::AttnOnly,
                        }),
                        vision_embeddings: None,
                        sender_peer_bytes: None,
                        requester_node_id: None,
                        pre_embedded: true,
                        adapter_id: None,
                        draft_tokens: Vec::new(),
                        spec_logits_requested: false,
                        truncate_kv_to: None,
                    };
                    let _ = self
                        .network_tx
                        .send(NetworkCommand::SendTensor {
                            target_peer_bytes: peer_bytes.clone(),
                            forward: remote_forward,
                        })
                        .await;
                }

                // Phase 1: AttnOnly — norm → head-sliced attention → partial output (local)
                let attn_forward = crate::types::LayerForward {
                    request_id,
                    sequence_num,
                    index_pos: index_pos as u32,
                    activations: current_activations_bytes.clone(),
                    format: crate::types::TensorFormat::FP16,
                    model_id: model_id.clone(),
                    layer_range: (layer_start as u32, layer_end as u32),
                    tp_meta: Some(crate::types::TensorParallelMeta {
                        tp_rank: tp_rank as u8,
                        tp_size: tp_size as u8,
                        single_layer: abs_layer as u32,
                        phase: crate::types::TpPhase::AttnOnly,
                    }),
                    vision_embeddings: None,
                    sender_peer_bytes: None,
                    requester_node_id: None,
                    pre_embedded: true,
                    adapter_id: None,
                    draft_tokens: Vec::new(),
                    spec_logits_requested: false,
                    truncate_kv_to: None,
                };
                let attn_partial = self
                    .shared_state
                    .model_process_pool
                    .forward(attn_forward)
                    .await?;

                // AllReduce attention partials → full post-attention output
                // Extract raw f32 bytes (strip tensor header) for AllReduce
                let attn_tensor = split::bytes_to_tensor(&attn_partial.activations)
                    .map_err(|e| SwarmError::Internal(format!("Deserialize attn partial: {e}")))?;
                let attn_shape: Vec<u32> = attn_tensor.dims().iter().map(|&d| d as u32).collect();
                let attn_raw = split::tensor_to_raw_f32(&attn_tensor)
                    .map_err(|e| SwarmError::Internal(format!("Extract attn f32: {e}")))?;
                let attn_compressed = zstd::encode_all(std::io::Cursor::new(&attn_raw), 1)
                    .map_err(|e| SwarmError::Internal(format!("Compress attn partial: {e}")))?;

                // Use layer*2 as AllReduce step ID for attn phase, layer*2+1 for FFN phase
                let attn_resp = crate::inference::allreduce::allreduce_sum(
                    &self.shared_state,
                    &self.network_tx,
                    &self.shared_state.allreduce_registry,
                    request_id,
                    abs_layer as u32 * 2,
                    tp_group,
                    tp_rank,
                    attn_compressed,
                    attn_shape,
                )
                .await?;

                // Decompress raw f32 AllReduce result, reconstruct tensor, add residual
                let post_attn_raw = zstd::decode_all(std::io::Cursor::new(&attn_resp.reduced_data))
                    .map_err(|e| SwarmError::Internal(format!("Decompress attn AR: {e}")))?;
                let attn_reduced_bytes =
                    split::raw_f32_to_tensor_bytes(&post_attn_raw, &attn_resp.shape);
                // Residual add: post_attn = AllReduce(attn_partials) + layer_input
                let post_attn_bytes =
                    split::tensor_bytes_add(&attn_reduced_bytes, &current_activations_bytes)
                        .map_err(|e| SwarmError::Internal(format!("Attn residual add: {e}")))?;

                // --- Send FfnOnly LayerForward to remote TP peers ---
                for (rank, peer_bytes) in &remote_tp_peers {
                    let remote_forward = crate::types::LayerForward {
                        request_id,
                        sequence_num,
                        index_pos: index_pos as u32,
                        activations: post_attn_bytes.clone(),
                        format: crate::types::TensorFormat::FP16,
                        model_id: model_id.clone(),
                        layer_range: (layer_start as u32, layer_end as u32),
                        tp_meta: Some(crate::types::TensorParallelMeta {
                            tp_rank: *rank as u8,
                            tp_size: tp_size as u8,
                            single_layer: abs_layer as u32,
                            phase: crate::types::TpPhase::FfnOnly,
                        }),
                        vision_embeddings: None,
                        sender_peer_bytes: None,
                        requester_node_id: None,
                        pre_embedded: true,
                        adapter_id: None,
                        draft_tokens: Vec::new(),
                        spec_logits_requested: false,
                        truncate_kv_to: None,
                    };
                    let _ = self
                        .network_tx
                        .send(NetworkCommand::SendTensor {
                            target_peer_bytes: peer_bytes.clone(),
                            forward: remote_forward,
                        })
                        .await;
                }

                // Phase 2: FfnOnly — ffn_norm → column-sliced FFN → partial output (local)
                let ffn_forward = crate::types::LayerForward {
                    request_id,
                    sequence_num,
                    index_pos: index_pos as u32,
                    activations: post_attn_bytes.clone(),
                    format: crate::types::TensorFormat::FP16,
                    model_id: model_id.clone(),
                    layer_range: (layer_start as u32, layer_end as u32),
                    tp_meta: Some(crate::types::TensorParallelMeta {
                        tp_rank: tp_rank as u8,
                        tp_size: tp_size as u8,
                        single_layer: abs_layer as u32,
                        phase: crate::types::TpPhase::FfnOnly,
                    }),
                    vision_embeddings: None,
                    sender_peer_bytes: None,
                    requester_node_id: None,
                    pre_embedded: true,
                    adapter_id: None,
                    draft_tokens: Vec::new(),
                    spec_logits_requested: false,
                    truncate_kv_to: None,
                };
                let ffn_partial = self
                    .shared_state
                    .model_process_pool
                    .forward(ffn_forward)
                    .await?;

                // AllReduce FFN partials → full post-FFN output
                // Extract raw f32 bytes (strip tensor header) for AllReduce
                let ffn_tensor = split::bytes_to_tensor(&ffn_partial.activations)
                    .map_err(|e| SwarmError::Internal(format!("Deserialize ffn partial: {e}")))?;
                let ffn_shape: Vec<u32> = ffn_tensor.dims().iter().map(|&d| d as u32).collect();
                let ffn_raw = split::tensor_to_raw_f32(&ffn_tensor)
                    .map_err(|e| SwarmError::Internal(format!("Extract ffn f32: {e}")))?;
                let ffn_compressed = zstd::encode_all(std::io::Cursor::new(&ffn_raw), 1)
                    .map_err(|e| SwarmError::Internal(format!("Compress ffn partial: {e}")))?;

                let ffn_resp = crate::inference::allreduce::allreduce_sum(
                    &self.shared_state,
                    &self.network_tx,
                    &self.shared_state.allreduce_registry,
                    request_id,
                    abs_layer as u32 * 2 + 1,
                    tp_group,
                    tp_rank,
                    ffn_compressed,
                    ffn_shape,
                )
                .await?;

                // Decompress raw f32 AllReduce result, reconstruct tensor, add residual
                let ffn_raw = zstd::decode_all(std::io::Cursor::new(&ffn_resp.reduced_data))
                    .map_err(|e| SwarmError::Internal(format!("Decompress ffn AR: {e}")))?;
                let ffn_reduced_bytes = split::raw_f32_to_tensor_bytes(&ffn_raw, &ffn_resp.shape);
                // Residual add: next_layer_input = AllReduce(ffn_partials) + post_attn
                current_activations_bytes =
                    split::tensor_bytes_add(&ffn_reduced_bytes, &post_attn_bytes)
                        .map_err(|e| SwarmError::Internal(format!("FFN residual add: {e}")))?;
            }

            if _is_last {
                // Last segment: the final activations need token sampling
                let final_tensor = split::bytes_to_tensor(&current_activations_bytes)
                    .map_err(|e| SwarmError::Internal(format!("Deserialize TP output: {e}")))?;
                let token_id =
                    split::sample_token_with_params(&final_tensor, &self.request.sampling_params)?;
                let mut result = vec![0x01];
                result.extend_from_slice(&(token_id as i64).to_le_bytes());
                Ok(result)
            } else {
                // Intermediate segment: prefix raw activations with 0x00 tag
                let mut result = vec![0x00];
                result.extend(current_activations_bytes);
                Ok(result)
            }
        } else {
            // Not in TP group — run full forward via subprocess (non-TP path)
            let layer_forward = crate::types::LayerForward {
                request_id,
                sequence_num,
                index_pos: index_pos as u32,
                activations: activation_bytes.to_vec(),
                format: crate::types::TensorFormat::FP16,
                model_id: model_id.clone(),
                layer_range: (layer_start as u32, layer_end as u32),
                tp_meta: None,
                vision_embeddings: None,
                sender_peer_bytes: None,
                requester_node_id: None,
                pre_embedded: false,
                adapter_id: None,
                draft_tokens: Vec::new(),
                spec_logits_requested: false,
                truncate_kv_to: None,
            };
            let layer_result = self
                .shared_state
                .model_process_pool
                .forward(layer_forward)
                .await?;

            if _is_last {
                let final_tensor = split::bytes_to_tensor(&layer_result.activations)
                    .map_err(|e| SwarmError::Internal(format!("Deserialize output: {e}")))?;
                let token_id =
                    split::sample_token_with_params(&final_tensor, &self.request.sampling_params)?;
                let mut result = vec![0x01];
                result.extend_from_slice(&(token_id as i64).to_le_bytes());
                Ok(result)
            } else {
                let mut result = vec![0x00];
                result.extend(layer_result.activations);
                Ok(result)
            }
        }
    }
}
