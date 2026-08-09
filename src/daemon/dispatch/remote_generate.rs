//! Remote-generate fast path handler.
//!
//! When a peer receives `SwarmMessage::RemoteGenerateRequest`, it owns the
//! entire decode loop for that inference — no per-token round trip back to
//! the coordinator. Tokens are streamed back to the coordinator as
//! `StreamingToken` messages as they are produced by the worker subprocess.
//!
//! Eligibility for this fast path is decided by the coordinator (single-
//! segment pipeline, no TP, no vision, no LoRA, no pipeline sealing). The
//! handler here only has to invoke the worker and shuttle tokens.

use std::sync::Arc;

use tokio::sync::mpsc;

use crate::types::{
    GenerateUsage, ModelId, NetworkCommand, NetworkFinishReason, RemoteGenerateRequest,
    StreamingToken,
};

use super::super::state::SharedState;

/// Handle an inbound `RemoteGenerateRequest`. Returns immediately — the
/// decode loop runs in a spawned task so the dispatch loop doesn't block.
pub(super) async fn handle_remote_generate_request(
    shared_state: Arc<SharedState>,
    network_tx: mpsc::Sender<NetworkCommand>,
    mut req: RemoteGenerateRequest,
) {
    let sender_bytes: Vec<u8> = match req.sender_peer_bytes.as_ref() {
        Some(b) => b.clone(),
        None => {
            tracing::warn!(
                request_id = %req.request_id,
                "RemoteGenerateRequest missing sender_peer_bytes — dropping"
            );
            return;
        }
    };

    let request_id = req.request_id;
    let model_id = req.model_id.clone();
    let layer_range = req.layer_range;
    // Serving-side accounting starts here. This path — not `layer_forward` — is
    // how a single-segment request is served, i.e. the common case, so without
    // it a node doing all its work through the fast path reports contributing
    // nothing at all.
    let serve_start = std::time::Instant::now();

    // SEC: clamp peer-supplied sampling params before they reach the worker.
    // The local API path runs `build_sampling_params` which clamps everything;
    // RemoteGenerateRequest arrives over the wire from peers and bypasses
    // that. Without this clamp a malicious peer can pin a worker for hours
    // with `max_tokens = u32::MAX`, or NaN-poison `temperature/top_p`.
    {
        let s = &mut req.sampling;
        s.temperature = if s.temperature.is_finite() {
            s.temperature.clamp(0.0, 2.0)
        } else {
            1.0
        };
        s.top_p = if s.top_p.is_finite() {
            s.top_p.clamp(f32::EPSILON, 1.0)
        } else {
            1.0
        };
        s.frequency_penalty = if s.frequency_penalty.is_finite() {
            s.frequency_penalty.clamp(-2.0, 2.0)
        } else {
            0.0
        };
        s.presence_penalty = if s.presence_penalty.is_finite() {
            s.presence_penalty.clamp(-2.0, 2.0)
        } else {
            0.0
        };
        s.max_tokens = s.max_tokens.clamp(1, crate::api::DEFAULT_MAX_TOKENS);
        s.top_logprobs = s.top_logprobs.min(20);
        if s.stop.len() > crate::api::MAX_STOP_SEQUENCES {
            s.stop.truncate(crate::api::MAX_STOP_SEQUENCES);
        }
    }

    tracing::info!(
        %request_id,
        model = %model_id,
        layer_range = ?layer_range,
        prompt_len = req.prompt.len(),
        max_tokens = req.sampling.max_tokens,
        "handling RemoteGenerateRequest — running full decode locally"
    );

    // Verify the model is hosted here. The coordinator is responsible for
    // picking a valid holder, but double-check to avoid spawning a worker
    // for a model we don't have shards for.
    if !can_serve_layer_range(&shared_state, &model_id, layer_range) {
        tracing::warn!(
            %request_id,
            model = %model_id,
            "RemoteGenerateRequest for a layer range this node cannot serve — sending error"
        );
        if let Err(e) = network_tx
            .send(NetworkCommand::SendStreamingToken {
                target_peer_bytes: sender_bytes,
                token: StreamingToken {
                    request_id,
                    token_id: 0,
                    finish_reason: Some(NetworkFinishReason::Error(
                        "model not hosted on target".into(),
                    )),
                    text: String::new(),
                    usage: None,
                    matched_stop_sequence: None,
                    logprob: None,
                },
            })
            .await
        {
            tracing::error!(
                %request_id,
                error = %e,
                "DIAG: could not queue the rejection back to the coordinator —                  it will see a silent timeout instead of a reason"
            );
        }
        return;
    }

    // Channel from the worker (via ModelProcessPool::generate) → the network
    // forwarding task below. Must be bounded to apply back-pressure if the
    // network can't keep up.
    let (token_tx, mut token_rx) = crate::inference::router::StreamingTokenTx::channel(64);

    // Spawn the generate call. It holds the model worker's socket lock for
    // the entire decode, which is fine — other requests for the same model
    // queue behind it (same behaviour as the local-API Generate path).
    let pool = shared_state.model_process_pool.clone();
    let layer_range_u32 = (layer_range.0, layer_range.1);
    // Mark the model as in-use-for-a-peer for the whole generate. Without this
    // the idle-VRAM unload sees only `active_pipelines` — which covers requests
    // this node ORIGINATED — and will happily kill the worker mid-answer on a
    // node that does nothing but serve (caught by soak, 2026-07-28).
    let serving_guard = crate::daemon::state::ServingGuard::new(&shared_state, model_id.clone());
    let gen_fut = tokio::spawn(async move {
        let _serving_guard = serving_guard;
        pool.generate(
            &model_id,
            layer_range_u32,
            req.prompt,
            req.sampling,
            request_id,
            req.session_id,
            Some(token_tx),
        )
        .await
    });
    // Register the abort handle so an inbound `SwarmMessage::CancelInference`
    // can stop this decode before it streams more wasted tokens back to the
    // originator. The map entry is removed below once the decode completes
    // naturally (or after abort fires).
    shared_state
        .inbound_generate_aborts
        .insert(request_id, (gen_fut.abort_handle(), sender_bytes.clone()));

    // Forward each token back to the coordinator as a `StreamingToken`.
    // Skip the "done" event emitted by `ModelProcessPool::generate` at the
    // end — the `gen_fut.await` path below emits ONE authoritative done
    // token with full usage info. Emitting both caused double-done on the
    // coordinator, which stopped at the first (usage-less) one.
    let forward_net_tx = network_tx.clone();
    let forward_sender = sender_bytes.clone();
    // Sequence number for the reply stream. Every token is an independent
    // request_response send — one substream each — so the network gives NO
    // ordering guarantee between them, and the coordinator used to stop at
    // whichever token carrying a finish_reason arrived first, discarding
    // anything still in flight. On a LAN the race is too narrow to see; at
    // 6s RTT it truncated most replies (observed 2026-08-09 against a peer
    // in another country: the same 2-token answer arrived as "", "ch" and
    // "Cherry" on successive attempts).
    //
    // `token_id` has been on the wire since the type was introduced and was
    // hardcoded to 0 at every send site. Filling it in is additive: a
    // coordinator too old to look at it is unaffected, and a new coordinator
    // treats an all-zero stream as unsequenced and behaves exactly as before.
    let stream_seq = std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0));
    let forward_seq = stream_seq.clone();
    let forward_task = tokio::spawn(async move {
        while let Some(evt) = token_rx.recv().await {
            // Skip events that carry a `finish_reason` — those are the
            // generate loop's end-of-stream marker, not real tokens. The
            // final-done path (gen_result) emits the authoritative done
            // with usage info.
            if evt.finish_reason.is_some() {
                continue;
            }
            if evt.text.is_empty() {
                continue;
            }
            let token = StreamingToken {
                request_id,
                token_id: forward_seq.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
                finish_reason: None,
                text: evt.text,
                usage: None,
                matched_stop_sequence: None,
                logprob: None,
            };
            if forward_net_tx
                .send(NetworkCommand::SendStreamingToken {
                    target_peer_bytes: forward_sender.clone(),
                    token,
                })
                .await
                .is_err()
            {
                tracing::debug!(%request_id, "network channel closed — stopping forward task");
                break;
            }
        }
    });

    // Wait for the decode to complete, then emit a final "done" token with
    // usage info. If the generate call errored, emit an Error finish token.
    let gen_result = gen_fut.await;
    let _ = forward_task.await;
    // Drop the abort handle now that the decode has exited. A late
    // CancelInference for this request_id becomes a no-op (the entry is gone).
    shared_state.inbound_generate_aborts.remove(&request_id);
    // Count the work regardless of outcome: time and layers were spent either
    // way, and an operator asking "is my node contributing" wants the truth
    // about effort, not only about successes. Bill only for tokens we can
    // evidence, though — a failed decode falls back to the 1-token floor rather
    // than guessing from the requested `max_tokens`, which the peer chose.
    let served_tokens = match gen_result {
        Ok(Ok(ref out)) => out.prompt_tokens.saturating_add(out.completion_tokens),
        _ => 1,
    };
    shared_state.record_peer_serve(crate::daemon::state::PeerServe {
        kind: crate::daemon::state::ServeKind::WholeRequest,
        layers: layer_range.1.saturating_sub(layer_range.0),
        elapsed_ms: serve_start.elapsed().as_millis() as u64,
        // The fast path streams tokens rather than returning activations, so
        // there are no activation bytes to attribute.
        activation_bytes: 0,
        tokens: served_tokens,
    });

    // The done token carries how many content tokens were sent, so the
    // coordinator can tell "the stream ended" from "the end overtook tokens
    // still in flight". `forward_task` has been awaited above, so every send
    // has been queued and the counter is final.
    let streamed_count = stream_seq.load(std::sync::atomic::Ordering::Relaxed);

    let final_token = match gen_result {
        Ok(Ok(out)) => StreamingToken {
            request_id,
            token_id: streamed_count,
            finish_reason: Some(match out.finish_reason.as_str() {
                "stop" => NetworkFinishReason::Stop,
                "length" => NetworkFinishReason::MaxTokens,
                other => NetworkFinishReason::Error(other.to_string()),
            }),
            text: String::new(),
            usage: Some(GenerateUsage {
                prompt_tokens: out.prompt_tokens,
                completion_tokens: out.completion_tokens,
            }),
            matched_stop_sequence: out.matched_stop_sequence,
            logprob: None,
        },
        Ok(Err(e)) => {
            tracing::warn!(%request_id, error = %e, "remote-generate worker error");
            StreamingToken {
                request_id,
                token_id: streamed_count,
                finish_reason: Some(NetworkFinishReason::Error(e.to_string())),
                text: String::new(),
                usage: None,
                matched_stop_sequence: None,
                logprob: None,
            }
        }
        Err(e) => {
            tracing::warn!(%request_id, error = %e, "remote-generate task join failed");
            StreamingToken {
                request_id,
                token_id: streamed_count,
                finish_reason: Some(NetworkFinishReason::Error(format!("task join: {e}"))),
                text: String::new(),
                usage: None,
                matched_stop_sequence: None,
                logprob: None,
            }
        }
    };
    // Do NOT discard this result. Everything above is careful to send a reason
    // back for every failure mode, but that is worthless if the reply itself is
    // dropped: the coordinator then sees "peer never acknowledged" and the
    // serving node's log shows a completed request. That asymmetry is what made
    // an external tester's serving-side failures undiagnosable across several
    // rounds — from the outside it was indistinguishable from a routing bug on
    // the requester.
    let was_error = matches!(
        final_token.finish_reason,
        Some(NetworkFinishReason::Error(_))
    );
    if let Err(e) = network_tx
        .send(NetworkCommand::SendStreamingToken {
            target_peer_bytes: sender_bytes,
            token: final_token,
        })
        .await
    {
        tracing::error!(
            %request_id,
            error = %e,
            was_error_reply = was_error,
            "DIAG: could not queue the final token back to the coordinator —              it will time out with no reason. This node served the request; the              reply is what was lost"
        );
    }
}

/// Can this node run `layer_range` of `model_id` on its own?
///
/// **Knowing about a model is not the same as being able to run it.** This
/// previously answered "is there a manifest on disk", which is true for a node
/// holding a SINGLE shard — every holder needs the manifest. A peer's
/// whole-model `RemoteGenerateRequest` was then accepted by a node holding only
/// part of the model, and the worker was asked for a full decode over a shard
/// window that does not start at layer 0. Nothing embeds in that case (the
/// embedding table is only loaded for a first segment), so raw token ids reach
/// the first attention block and it fails with a shape mismatch —
/// `attn_norm: shape mismatch in rms-norm [1, 128] [3072]`, where 128 is the
/// prefill chunk size and 3072 the hidden size. Reported externally 2026-07-27.
///
/// Refusing here is the right answer: the caller already handles the rejection
/// by sending an error back, and will pick another holder.
fn can_serve_layer_range(
    shared_state: &SharedState,
    model_id: &ModelId,
    layer_range: (u32, u32),
) -> bool {
    let Some(manifest) = shared_state.model_registry.get_manifest(model_id) else {
        // No manifest at all — we genuinely do not know this model.
        return false;
    };

    // An empty or inverted range is not something we can satisfy.
    if layer_range.0 >= layer_range.1 {
        return false;
    }

    let local_node_id = shared_state.identity.node_id().clone();
    let shard_store = shared_state.shard_store();
    let mut local_shard_indices: Vec<u32> = manifest
        .shards
        .iter()
        .filter(|s| {
            let sid = crate::types::ShardId {
                model_id: model_id.clone(),
                index: s.index,
            };
            if !shared_state
                .model_registry
                .shard_holders(&sid)
                .contains(&local_node_id)
            {
                return false;
            }
            let path = shard_store.shard_path(model_id, s.index);
            path.exists() && crate::model::auto_manage::shard_size_ok(&path, s.size_bytes)
        })
        .map(|s| s.index)
        .collect();
    local_shard_indices.sort_unstable();
    if local_shard_indices.is_empty() {
        return false;
    }

    // Whole model present — any range within it is servable.
    if local_shard_indices.len() == manifest.shard_count as usize {
        return layer_range.1 as usize <= manifest.num_layers as usize;
    }

    // Otherwise the requested span must sit entirely inside one contiguous
    // range we actually hold.
    let covered = crate::inference::split::available_layer_ranges_from_manifest(
        &manifest,
        &local_shard_indices,
    );
    range_is_covered(&covered, (layer_range.0 as usize, layer_range.1 as usize))
}

/// Is `want` contained in ONE of the contiguous ranges in `covered`?
///
/// Split out from [`can_serve_layer_range`] so the decision can be tested
/// without standing up a node. Containment must be within a single range —
/// holding layers 0-4 and 8-12 does not mean we can serve 0-12, because the
/// middle is missing and a decode cannot skip it.
fn range_is_covered(covered: &[(usize, usize)], want: (usize, usize)) -> bool {
    if want.0 >= want.1 {
        return false;
    }
    covered
        .iter()
        .any(|&(start, end)| start <= want.0 && want.1 <= end)
}

#[cfg(test)]
mod serve_range_tests {
    use super::range_is_covered;

    /// The reported case: this node holds only the tail, a peer asks it to run
    /// the whole model. Accepting fed raw token ids into layer 21 and failed
    /// with `attn_norm: shape mismatch in rms-norm [1, 128] [3072]`.
    #[test]
    fn a_tail_only_node_refuses_a_whole_model_request() {
        let covered = [(21usize, 28usize)];
        assert!(!range_is_covered(&covered, (0, 28)), "must refuse");
        // But it can still serve the tail segment it actually holds.
        assert!(range_is_covered(&covered, (21, 28)));
        assert!(range_is_covered(&covered, (22, 27)), "a sub-span is fine");
    }

    #[test]
    fn a_whole_model_holder_serves_anything_inside_it() {
        let covered = [(0usize, 28usize)];
        assert!(range_is_covered(&covered, (0, 28)));
        assert!(range_is_covered(&covered, (12, 21)));
    }

    /// Containment must be within ONE range. Holding both ends but not the
    /// middle cannot serve a span crossing the gap — a decode cannot skip
    /// layers. This is the prompt-privacy shard layout, so it matters.
    #[test]
    fn a_gap_between_two_held_ranges_is_not_coverage() {
        let covered = [(0usize, 3usize), (21, 28)];
        assert!(
            !range_is_covered(&covered, (0, 28)),
            "must not span the gap"
        );
        assert!(!range_is_covered(&covered, (2, 22)));
        assert!(range_is_covered(&covered, (0, 3)));
        assert!(range_is_covered(&covered, (21, 28)));
    }

    #[test]
    fn holding_nothing_serves_nothing() {
        assert!(!range_is_covered(&[], (0, 28)));
    }

    #[test]
    fn an_empty_or_inverted_span_is_refused() {
        let covered = [(0usize, 28usize)];
        assert!(!range_is_covered(&covered, (5, 5)));
        assert!(!range_is_covered(&covered, (9, 4)));
    }
}
