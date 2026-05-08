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
    if !has_model_locally(&shared_state, &model_id) {
        tracing::warn!(
            %request_id,
            model = %model_id,
            "RemoteGenerateRequest for model not hosted here — sending error"
        );
        let _ = network_tx
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
                },
            })
            .await;
        return;
    }

    // Channel from the worker (via ModelProcessPool::generate) → the network
    // forwarding task below. Must be bounded to apply back-pressure if the
    // network can't keep up.
    let (token_tx, mut token_rx) =
        mpsc::channel::<crate::inference::router::StreamingTokenEvent>(64);

    // Spawn the generate call. It holds the model worker's socket lock for
    // the entire decode, which is fine — other requests for the same model
    // queue behind it (same behaviour as the local-API Generate path).
    let pool = shared_state.model_process_pool.clone();
    let layer_range_u32 = (layer_range.0, layer_range.1);
    let gen_fut = tokio::spawn(async move {
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

    // Forward each token back to the coordinator as a `StreamingToken`.
    // Skip the "done" event emitted by `ModelProcessPool::generate` at the
    // end — the `gen_fut.await` path below emits ONE authoritative done
    // token with full usage info. Emitting both caused double-done on the
    // coordinator, which stopped at the first (usage-less) one.
    let forward_net_tx = network_tx.clone();
    let forward_sender = sender_bytes.clone();
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
                token_id: 0,
                finish_reason: None,
                text: evt.text,
                usage: None,
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
    let final_token = match gen_result {
        Ok(Ok(out)) => StreamingToken {
            request_id,
            token_id: 0,
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
        },
        Ok(Err(e)) => {
            tracing::warn!(%request_id, error = %e, "remote-generate worker error");
            StreamingToken {
                request_id,
                token_id: 0,
                finish_reason: Some(NetworkFinishReason::Error(e.to_string())),
                text: String::new(),
                usage: None,
            }
        }
        Err(e) => {
            tracing::warn!(%request_id, error = %e, "remote-generate task join failed");
            StreamingToken {
                request_id,
                token_id: 0,
                finish_reason: Some(NetworkFinishReason::Error(format!("task join: {e}"))),
                text: String::new(),
                usage: None,
            }
        }
    };
    let _ = network_tx
        .send(NetworkCommand::SendStreamingToken {
            target_peer_bytes: sender_bytes,
            token: final_token,
        })
        .await;
}

fn has_model_locally(shared_state: &SharedState, model_id: &ModelId) -> bool {
    // Check for local split-model entries (the on-demand loaded path).
    if shared_state.split_model_index.contains_key(model_id) {
        return true;
    }
    // Check for shards on disk — the worker will load them on first use.
    let model_dir = shared_state.model_dir(&model_id.0);
    let manifest_path = model_dir.join("manifest.json");
    manifest_path.exists()
}
