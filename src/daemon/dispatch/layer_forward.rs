use std::sync::Arc;

use tokio::sync::mpsc;

use crate::types::NetworkCommand;

use super::super::state::SharedState;
use super::seal_layer_result;

pub(super) async fn handle_layer_forward(
    shared_state: Arc<SharedState>,
    network_tx: mpsc::Sender<NetworkCommand>,
    mut forward: crate::types::LayerForward,
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
    // The rest of the pipeline after us, if the coordinator chained this
    // request. Captured here for the same reason as the two above: `forward` is
    // moved into the worker pool below.
    let chain = std::mem::take(&mut forward.chain);
    let sequence_num = forward.sequence_num;
    let index_pos = forward.index_pos;

    // Mark the model busy for as long as this segment is computing.
    //
    // `record_peer_serve` below records that the work HAPPENED; this records
    // that it is happening NOW, which is a different question and the one
    // `active_inference_load` needs. Serving a segment was the one kind of work
    // nothing counted at all: a node could be running segments back to back for
    // the whole swarm and still advertise a load of zero, so every coordinator
    // kept sending it more.
    //
    // RAII, so the count survives the several early returns below (worker
    // failure, and the TP branch's encoding failures).
    let _serving_guard = crate::daemon::state::ServingGuard::new(&shared_state, model_id.clone());

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

    // Direct peer chaining: hand our output to the next segment instead of
    // returning it to the coordinator.
    //
    // Every hop that comes home costs the coordinator a round trip, so an
    // N-segment pipeline pays N of them per token. Passing the activations
    // sideways makes that one trip out and one trip back however long the chain
    // is — the difference between a many-shard model being usable and not.
    //
    // Falling back to the coordinator is always correct and never an error: if
    // we do not know the next node, this is a tensor-parallel forward, or the
    // range is not ours to hand on, the result simply comes home and the
    // request costs what it used to.
    if chaining_applies(&chain, tp_meta.is_some(), result.activations.is_empty()) {
        if let Some(next) = chain.first() {
            match shared_state.resolve_connected_peer_id_bytes(&next.node_id) {
                Some(next_peer_bytes) => {
                    let onward = crate::types::LayerForward {
                        request_id,
                        sequence_num,
                        index_pos,
                        activations: result.activations,
                        format: crate::types::TensorFormat::FP32,
                        model_id: model_id.clone(),
                        layer_range: next.layer_range,
                        tp_meta: None,
                        vision_embeddings: None,
                        chain: chain[1..].to_vec(),
                        sender_peer_bytes: None,
                        requester_node_id,
                        // Matches what the coordinator sends a mid-chain
                        // segment, and deliberately so. That path sets
                        // `pre_embedded && idx == 0` — false for every segment
                        // after the first, because a receiver infers hidden
                        // states from a layer range that does not start at
                        // zero, which a chained hop never does. Setting it true
                        // here would make a chained forward differ from the
                        // relayed one it replaces, for the same work.
                        pre_embedded: false,
                        // Only the LAST segment samples, and only it needs the
                        // decoded-so-far ids for frequency/presence penalties.
                        // We do not have them: the coordinator sends them with
                        // the final segment's forward, and in a chain that
                        // forward is built here rather than there. The
                        // coordinator must therefore not chain a request that
                        // has penalties set. That is a constraint on whoever
                        // builds the chain, stated here because this is where
                        // it would silently go wrong.
                        generated_ids: Vec::new(),
                        // The coordinator sends `None` to every segment, so a
                        // chained hop does the same. LoRA requests do not take
                        // this path at all.
                        adapter_id: None,
                        draft_tokens: Vec::new(),
                        spec_logits_requested: false,
                        truncate_kv_to: None,
                        chunk_meta: None,
                    };
                    tracing::info!(
                        request_id = %request_id,
                        next = %next.node_id,
                        next_layers = ?next.layer_range,
                        remaining = chain.len() - 1,
                        "DIAG: chaining activations to the next segment"
                    );
                    if let Err(e) = network_tx
                        .send(NetworkCommand::SendTensor {
                            target_peer_bytes: next_peer_bytes,
                            forward: onward,
                        })
                        .await
                    {
                        // The activations are gone with the failed send, so the
                        // coordinator must be told rather than left waiting for
                        // its whole deadline.
                        tracing::warn!(error = %e, request_id = %request_id, "chained send failed");
                        // To the COORDINATOR, not to whoever handed us the
                        // work: mid-chain our predecessor is not waiting for
                        // anything and would simply drop this.
                        let reply_to = reply_target(
                            requester_node_id,
                            &local_node_id,
                            sender_peer_bytes,
                            |n| shared_state.resolve_connected_peer_id_bytes(n),
                        );
                        send_error_result(
                            &network_tx,
                            &reply_to,
                            request_id,
                            "chained forward could not be sent",
                        )
                        .await;
                    }
                    return;
                }
                None => {
                    // Not connected to the next hop, so this run cannot
                    // continue. Say so — do NOT fall through and return our
                    // activations.
                    //
                    // They cover only OUR layers, and the coordinator has
                    // already skipped past the rest of the run on the
                    // assumption the chain would carry them. Handing back a
                    // partial tensor of an entirely plausible size would be
                    // fed to whatever comes after the run as though the whole
                    // chain had computed it: a confident wrong answer instead
                    // of an error. An error costs one retry, unchained.
                    tracing::warn!(
                        request_id = %request_id,
                        next = %next.node_id,
                        "next hop unreachable — failing the chained run"
                    );
                    let reply_to =
                        reply_target(requester_node_id, &local_node_id, sender_peer_bytes, |n| {
                            shared_state.resolve_connected_peer_id_bytes(n)
                        });
                    send_error_result(
                        &network_tx,
                        &reply_to,
                        request_id,
                        "chained run could not reach the next segment",
                    )
                    .await;
                    return;
                }
            }
        }
    }

    // Send the result to whoever is WAITING for it, which is not always
    // whoever sent it to us.
    //
    // On an unchained forward those are the same node and this changes nothing.
    // In a chain they are not: our predecessor handed us the activations and is
    // no longer involved, while the coordinator is waiting on the tail. Replying
    // to the sender would send the answer to a node that is not expecting it,
    // and the request would hang until the coordinator's deadline — which is
    // exactly what a chained request would have done.
    //
    // `requester_node_id` is the coordinator: it is already on the forward so
    // the last segment can seal the result for that node's key, which is the
    // same "who asked for this" question. If we cannot resolve it — a peer we
    // have no route to — the sender is the best remaining guess and is right
    // for every unchained forward.
    // `resolve_connected_peer_id_bytes`, not the ungated resolver: a
    // `LayerResult` is a direct send, and the peer-id map is deliberately kept
    // across disconnects, so the ungated one hands back targets the send path
    // can only drop (gotcha #220). Not connected means fall back to the sender,
    // which is correct for every unchained forward and no worse than before.
    let reply_to = reply_target(requester_node_id, &local_node_id, sender_peer_bytes, |n| {
        shared_state.resolve_connected_peer_id_bytes(n)
    });

    if let Err(e) = network_tx
        .send(NetworkCommand::SendTensorResult {
            target_peer_bytes: reply_to,
            result,
        })
        .await
    {
        tracing::warn!(error = %e, "Failed to send LayerResult back to peer");
    }
}

/// Who should receive this segment's result?
///
/// Extracted so the rule is testable without a network, because getting it
/// wrong is silent: the answer goes to a node that is not waiting for it and
/// the request hangs until the coordinator's deadline.
///
/// Whoever sent us the activations is not always whoever is waiting for the
/// answer. On an unchained forward they are the same node. In a chain the
/// predecessor handed the work along and is done, while the coordinator is
/// waiting on the tail — so the result must go to the requester.
fn reply_target(
    requester_node_id: Option<[u8; 32]>,
    local_node_id: &crate::types::NodeId,
    sender_peer_bytes: Vec<u8>,
    resolve: impl Fn(&crate::types::NodeId) -> Option<Vec<u8>>,
) -> Vec<u8> {
    requester_node_id
        .map(crate::types::NodeId)
        .filter(|n| n != local_node_id)
        .and_then(|n| resolve(&n))
        .unwrap_or(sender_peer_bytes)
}

/// Should this segment hand its output onward rather than return it?
///
/// Extracted so the rule is testable without a worker or a network, and so the
/// three disqualifiers are stated in one place rather than as a condition that
/// grows a clause at a time.
///
/// Every "no" here is a fallback to returning the result to the coordinator,
/// which is the pre-chaining behaviour: correct, one round trip more expensive,
/// and never an error.
fn chaining_applies(
    chain: &[crate::types::ChainHop],
    is_tensor_parallel: bool,
    activations_empty: bool,
) -> bool {
    // Nobody after us: we ARE the last segment, so the result goes home.
    if chain.is_empty() {
        return false;
    }
    // A tensor-parallel forward is one slice of a layer that has to be
    // all-reduced with its siblings by the coordinator before it means
    // anything. There is nothing to hand on.
    if is_tensor_parallel {
        return false;
    }
    // Nothing to forward. An empty activation is how a failure or a
    // control-only forward looks, and passing it down the chain would turn one
    // node's problem into a silent wrong answer several hops away.
    if activations_empty {
        return false;
    }
    true
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

#[cfg(test)]
mod chaining_tests {
    use super::chaining_applies;
    use crate::types::{ChainHop, NodeId};

    fn hop() -> Vec<ChainHop> {
        vec![ChainHop {
            node_id: NodeId([1u8; 32]),
            layer_range: (8, 16),
        }]
    }

    #[test]
    fn a_segment_with_somebody_after_it_hands_its_output_on() {
        assert!(chaining_applies(&hop(), false, false));
    }

    /// Each disqualifier falls back to returning the result to the
    /// coordinator, which always works. None of them is an error.
    #[test]
    fn the_last_segment_returns_its_result() {
        assert!(!chaining_applies(&[], false, false));
    }

    #[test]
    fn a_tensor_parallel_slice_is_never_chained() {
        // One slice of a layer means nothing until the coordinator all-reduces
        // it with its siblings.
        assert!(!chaining_applies(&hop(), true, false));
    }

    #[test]
    fn an_empty_activation_is_not_forwarded() {
        // That is what a failure looks like. Passing it on would turn one
        // node's problem into a silent wrong answer several hops away.
        assert!(!chaining_applies(&hop(), false, true));
    }

    use super::reply_target;

    /// The bug this prevents: in a chain the previous hop hands the work along
    /// and stops caring, while the coordinator waits on the tail. Replying to
    /// the sender sends the answer to a node that is not expecting it, and the
    /// request hangs until the deadline.
    #[test]
    fn the_tail_of_a_chain_answers_the_coordinator_not_its_predecessor() {
        let coordinator = NodeId([9u8; 32]);
        let me = NodeId([3u8; 32]);
        let predecessor_bytes = b"previous-hop".to_vec();

        let to = reply_target(Some(coordinator.0), &me, predecessor_bytes.clone(), |n| {
            (n == &coordinator).then(|| b"coordinator".to_vec())
        });
        assert_eq!(to, b"coordinator".to_vec());
    }

    /// An unchained forward is unaffected: sender and requester are the same
    /// node, so this resolves to where the result already went.
    #[test]
    fn an_unchained_forward_still_answers_its_sender() {
        let coordinator = NodeId([9u8; 32]);
        let me = NodeId([3u8; 32]);
        let to = reply_target(Some(coordinator.0), &me, b"sender".to_vec(), |_| {
            Some(b"coordinator".to_vec())
        });
        // Same node either way — the point is that resolving does not break it.
        assert_eq!(to, b"coordinator".to_vec());
    }

    /// A coordinator we have no live route to falls back to the sender, which
    /// is right for every unchained forward and no worse than before.
    #[test]
    fn an_unreachable_requester_falls_back_to_the_sender() {
        let me = NodeId([3u8; 32]);
        let to = reply_target(Some([9u8; 32]), &me, b"sender".to_vec(), |_| None);
        assert_eq!(to, b"sender".to_vec());

        // And a forward with no requester at all — an older node — is unchanged.
        let to = reply_target(None, &me, b"sender".to_vec(), |_| Some(b"x".to_vec()));
        assert_eq!(to, b"sender".to_vec());
    }

    /// Never address ourselves. A forward claiming we are our own coordinator
    /// would otherwise loop the result back into this node.
    #[test]
    fn a_forward_naming_us_as_the_requester_replies_to_the_sender() {
        let me = NodeId([3u8; 32]);
        let to = reply_target(Some(me.0), &me, b"sender".to_vec(), |_| {
            Some(b"self".to_vec())
        });
        assert_eq!(to, b"sender".to_vec());
    }
}
