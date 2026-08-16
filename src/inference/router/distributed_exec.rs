//! Distributed inference execution paths. Handles `execute_request` (the
//! per-request router entry), `execute_distributed_batch` (batched flavour),
//! and `finalize_request` (shared stats/credit accounting for both local and
//! distributed completions).

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use tokio::sync::mpsc;

use crate::daemon::SharedState;
use crate::error::SwarmError;
use crate::inference::chat_template;
use crate::inference::pipeline::PipelineExecutor;
use crate::inference::scheduler::PipelineScheduler;
use crate::types::{InferenceRequest, NetworkCommand, PipelineAssignment};

use super::spot_check::spot_check_distributed_result;
use super::types::{InferenceOutput, QueuedRequest, StreamingTokenTx};

const MODEL_LOAD_WAIT_SECS: u64 = 60;

/// Finalize a completed request: update stats and apply credit charges.
/// When `escrow_id` is `Some`, the escrow already deducted credits — skip the
/// direct charge to avoid double-billing the local API consumer.
pub(super) async fn finalize_request(
    shared_state: &SharedState,
    request: &InferenceRequest,
    output: &Result<InferenceOutput, SwarmError>,
    escrow_id: Option<uuid::Uuid>,
) {
    if let Err(ref e) = output {
        tracing::error!(
            request_id = %request.id,
            model = %request.model_id,
            error = %e,
            "Inference request failed"
        );
        // Emit failure activity
        shared_state.emit_activity(
            crate::daemon::state::ActivityEvent::new(
                "inference",
                "inference_failed",
                format!("Inference failed: {}", e),
            )
            .with_model(request.model_id.0.clone())
            .with_detail_str(format!("{}", e))
            .with_toast("warning", 5000),
        );

        // Refund the escrow if one was created. Without this, a failed
        // request leaves credits locked until the 5-minute cleanup tick
        // (which only fires on `expires_at` expiry — itself defaulted to
        // a generous window). Credit enforcement isn't gating real users
        // yet, but the bookkeeping has to be correct for when it is.
        if let Some(eid) = escrow_id {
            match shared_state
                .credits
                .escrow_manager
                .refund_escrow(eid, &shared_state.credits.credit_balance)
                .await
            {
                Ok(amount) => {
                    tracing::info!(
                        request_id = %request.id,
                        escrow_id = %eid,
                        amount,
                        "DIAG: refunded escrow on inference failure"
                    );
                }
                Err(re) => {
                    // Don't propagate — the user already got their failure.
                    // Log so the cleanup sweep can still mop up.
                    tracing::warn!(
                        request_id = %request.id,
                        escrow_id = %eid,
                        error = %re,
                        "Escrow refund failed; cleanup tick will retry"
                    );
                }
            }
        }
    }

    // Local API requests use NodeId([0; 32]) as requester sentinel
    let is_local_api_request = request.requester == crate::types::NodeId([0u8; 32]);

    if let Ok(ref result) = output {
        // `requests_served` is deliberately NOT bumped here. This is the
        // router's own completion hook and it fires for local API requests too,
        // so counting it made a user's own chat register as "requests your
        // computer handled for others". Serving is recorded at
        // `SharedState::record_peer_serve`. `inference_requests_total` below is
        // the all-requests Prometheus counter and correctly includes local ones.
        shared_state
            .metrics
            .inference_requests_total
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        // Emit inference completion activity
        {
            let display = shared_state.model_registry.display_name(&request.model_id);
            let total_tokens = result.prompt_tokens + result.completion_tokens;
            shared_state.emit_activity(
                crate::daemon::state::ActivityEvent::new(
                    "inference",
                    "inference_completed",
                    format!(
                        "Completed on {} — {} prompt + {} generated = {} total tokens ({})",
                        display,
                        result.prompt_tokens,
                        result.completion_tokens,
                        total_tokens,
                        result.finish_reason,
                    ),
                )
                .with_model(request.model_id.0.clone())
                .with_detail_num(total_tokens as i64)
                .with_detail_str(result.finish_reason.clone()),
            );
        }

        // Credit operations:
        // - Per-layer earn credits are handled in process_local_segment
        //   and handle_layer_forward (each node earns for layers it processed)
        // - Here we debit the local API consumer for requesting inference
        // - Skip if escrow was used — escrow already deducted the estimated cost
        // - Pool members (slaves): charge goes to the MASTER's balance via credit forward.
        //   The slave's dashboard is fully usable; usage is billed to the pool owner.
        // Snapshot both fields from a single pool_state read so the later
        // pool_tx-held branch doesn't need a nested pool_state.read(). Tokio's
        // RwLock is write-preferring; nesting two reads on different locks lets
        // a queued write on pool_state stall the inference completion path.
        let (is_pool_member, pool_id_opt) = {
            let ps = shared_state.credits.pool_state.read().await;
            let me = shared_state.identity.node_id();
            let pid = ps.as_ref().map(|s| s.pool_id.clone());
            let member = pid.as_ref().map(|p| p != me).unwrap_or(false);
            (member, pid)
        };
        if is_local_api_request && escrow_id.is_none() {
            let total_tokens = result.prompt_tokens + result.completion_tokens;
            let spent = crate::credit::ledger::RATE_INFERENCE_CONSUME * total_tokens as i64;

            if is_pool_member {
                // Slave device: forward the spend to the master's balance.
                // Use the same credit forward mechanism as earning, but negative.
                // Clone the sender out of the RwLock guard before awaiting so a
                // concurrent PoolManager start/stop doesn't stall behind this send.
                let pool_tx = shared_state.credits.pool_tx.read().await.clone();
                if let Some(tx) = pool_tx {
                    let my_id = shared_state.identity.node_id();
                    if let Some(pid) = pool_id_opt.clone() {
                        let forward = crate::pool::crypto::create_credit_forward(
                            &shared_state.identity,
                            &pid,
                            my_id,
                            &pid,
                            -spent, // negative = spend deduction
                        );
                        let _ = tx
                            .send(crate::pool::types::PoolCommand::ProcessCreditForward { forward })
                            .await;
                        tracing::info!(
                            spent,
                            request_id = %request.id,
                            "DIAG: forwarded inference spend to pool owner"
                        );
                    }
                }
            } else {
                // Normal node or pool owner: charge locally
                if let Err(e) = crate::credit::ledger::apply_credit_direct_noted(
                    &shared_state.credits.credit_balance,
                    &shared_state.db,
                    -spent,
                    crate::credit::ledger::CreditDelta::Spending,
                    "inference_charge",
                )
                .await
                {
                    tracing::warn!(error = %e, "Failed to persist credit spend");
                }
                tracing::info!(
                    spent,
                    total_tokens,
                    request_id = %request.id,
                    "DIAG: spent credits for consuming inference"
                );
            }
        }
    }

    // Clean up per-request KV-cache entries now that the request is done.
    // EXCEPT when the request has a session_id — those entries persist for
    // multi-turn reuse. They'll be cleaned up by the TTL-based expiry instead.
    if request.session_id.is_none() {
        let req_id_str = request.id.to_string();
        shared_state.kv_cache_store.cleanup_request_id(&req_id_str);
    } else {
        tracing::debug!(
            request_id = %request.id,
            session_id = ?request.session_id,
            "Preserving KV-cache for multi-turn session"
        );
    }
}

/// Execute a batch of distributed inference requests concurrently.
///
/// Each request gets its own pipeline. They share the active_count
/// and are finalized independently.
pub(super) async fn execute_distributed_batch(
    shared_state: Arc<SharedState>,
    network_tx: mpsc::Sender<NetworkCommand>,
    scheduler: PipelineScheduler,
    batch: Vec<QueuedRequest>,
    active_count: Arc<AtomicUsize>,
    queue_notify: Arc<tokio::sync::Notify>,
) {
    // Pair each JoinHandle with the request_id of the spawned task so the
    // panic-recovery arm below can clean up `active_pipelines` too — the
    // normal-completion path inside the spawn removes the entry, but a
    // panic bypasses that. Without the pair, a panicked request leaves a
    // permanent `active_pipelines` entry that blocks shard pruning and
    // inflates the scheduler's local-load metric.
    let mut handles: Vec<(uuid::Uuid, tokio::task::JoinHandle<bool>)> =
        Vec::with_capacity(batch.len());

    for queued in batch {
        let shared_state = shared_state.clone();
        let network_tx = network_tx.clone();
        let scheduler = scheduler.clone();
        let active_count = active_count.clone();
        let queue_notify = queue_notify.clone();

        let request_id = queued.request.id;
        handles.push((
            request_id,
            tokio::spawn(async move {
                let request = queued.request;
                let result_tx = queued.result_tx;
                let token_tx = queued.token_tx;
                let trace = queued.trace;
                trace.mark_dequeued();

                let output = execute_request(
                    shared_state.clone(),
                    network_tx,
                    scheduler,
                    request.clone(),
                    token_tx,
                    None, // No pipeline affinity for batched requests
                    trace.clone(),
                )
                .await;
                match &output {
                    Ok(r) => trace.mark_finished(
                        crate::inference::trace::Outcome::Ok,
                        r.prompt_tokens,
                        r.completion_tokens,
                    ),
                    Err(e) => trace.mark_finished(
                        crate::inference::trace::Outcome::Error(
                            crate::inference::trace::error_kind(e).to_string(),
                        ),
                        0,
                        0,
                    ),
                }
                shared_state.publish_request_trace(&trace);

                finalize_request(&shared_state, &request, &output, None).await;
                shared_state.release_request_state(&request.id);
                // Decrement active_count and wake drain_queue so the next queued
                // request can dispatch (without notify, the queue stalls until a
                // new Submit arrives).
                active_count.fetch_sub(1, Ordering::Relaxed);
                queue_notify.notify_one();
                if result_tx.send(output).is_err() {
                    tracing::warn!(
                        request_id = %request.id,
                        "DIAG: distributed batch result_tx receiver dropped"
                    );
                }
                true // task completed normally, already decremented
            }),
        ));
    }

    // Wait for all requests in the batch to complete.
    // Only decrement if the task panicked BEFORE it could decrement itself.
    for (request_id, handle) in handles {
        match handle.await {
            Ok(_) => {} // task already decremented
            Err(_) => {
                // Panic: clean up both counters AND the active_pipelines
                // entry the panicked task would have removed on its
                // normal-exit path. Without this, the entry stays
                // forever and blocks shard pruning (gotcha #85).
                shared_state.release_request_state(&request_id);
                active_count.fetch_sub(1, Ordering::Relaxed);
                queue_notify.notify_one();
            }
        }
    }
}

/// Execute a single inference request — either locally or via distributed pipeline.
/// How long to wait for the DHT provider results we just asked for before
/// giving up on assembling a pipeline.
///
/// The query fired a few lines earlier is fire-and-forget, so on the FIRST
/// request for a model — including every request after a restart, since holder
/// claims are rebuilt from gossip — the holder cache is still empty and assembly
/// fails with "No node available". The user sees a hard error and a manual retry
/// seconds later succeeds. Reproduced independently twice on 2026-07-26.
///
/// Kept short: a model that genuinely has no holder should still fail quickly.
const DHT_ASSEMBLY_GRACE: std::time::Duration = std::time::Duration::from_millis(1500);
const DHT_ASSEMBLY_POLL: std::time::Duration = std::time::Duration::from_millis(250);

/// Assemble a pipeline, giving in-flight DHT provider results a brief chance to
/// land if the first attempt finds no holder.
///
/// Only retries the "nothing known to serve this" case. Any other scheduling
/// error is returned immediately — waiting would add latency without changing
/// the outcome.
async fn assemble_awaiting_dht(
    scheduler: &PipelineScheduler,
    model_id: &crate::types::ModelId,
    local_node_id: &crate::types::NodeId,
    request_id: uuid::Uuid,
) -> Result<PipelineAssignment, SwarmError> {
    let first = scheduler.assemble_pipeline_for(model_id, local_node_id, request_id);
    let Err(err) = first else {
        return first;
    };
    if !assembly_failed_for_lack_of_holders(&err) {
        return Err(err);
    }

    let deadline = std::time::Instant::now() + DHT_ASSEMBLY_GRACE;
    while std::time::Instant::now() < deadline {
        tokio::time::sleep(DHT_ASSEMBLY_POLL).await;
        if let Ok(assignment) = scheduler.assemble_pipeline_for(model_id, local_node_id, request_id)
        {
            tracing::info!(
                %request_id,
                model = %model_id,
                "Pipeline assembled after waiting for DHT provider results"
            );
            return Ok(assignment);
        }
    }
    // Report the ORIGINAL error: it names the layer that had no holder, which is
    // more useful than "we also failed 1.5s later".
    Err(err)
}

/// Did assembly fail because no node is known to hold what we need, as opposed
/// to a real scheduling constraint?
///
/// Only this case benefits from waiting — the holder cache may simply not have
/// been populated yet.
fn assembly_failed_for_lack_of_holders(err: &SwarmError) -> bool {
    // Matched on the TYPE, because matching the prose broke silently. The
    // scheduler's wording changed to "No reachable node holds the part of …" on
    // 2026-08-10 and neither substring below matches it, so the DHT wait — the
    // whole point of which is "the holder cache may simply not be populated
    // yet" — stopped firing for the commonest missing-holder failure and nobody
    // saw a thing. The remaining substrings cover the `PipelineError` forms that
    // are still stringly-typed.
    if matches!(err, SwarmError::ModelIncompleteInSwarm { .. }) {
        return true;
    }
    let msg = err.to_string();
    msg.contains("No node available") || msg.contains("No shard holders")
}

pub(super) async fn execute_request(
    shared_state: Arc<SharedState>,
    network_tx: mpsc::Sender<NetworkCommand>,
    scheduler: PipelineScheduler,
    request: InferenceRequest,
    token_tx: Option<StreamingTokenTx>,
    preferred_pipeline: Option<PipelineAssignment>,
    // `trace` is required, not optional: a path that quietly passes nothing
    // produces a trace with no route, which is indistinguishable from a local
    // request in every surface that renders it.
    trace: Arc<crate::inference::trace::RequestTrace>,
) -> Result<InferenceOutput, SwarmError> {
    let model_id = &request.model_id;

    // Update model trust on inference request — promotes to DemandVerified
    // after threshold, enabling auto-manage to propagate this model.
    {
        let mut trust = shared_state
            .models
            .model_trust
            .entry(model_id.clone())
            .or_insert_with(crate::types::ModelTrustInfo::new_discovered);
        trust.record_request();
        // Persist on promotion only (not every request)
        if trust.total_requests == 3 {
            let _ = shared_state
                .db
                .put_json("model_trust", &model_id.0, trust.value());
        }
    }

    // Check if we can handle this entirely locally.
    // Use the atomic flag to avoid locking the executor mutex just to check readiness.
    // Skip the llama.cpp path when a LoRA adapter is requested — LoRA is only
    // supported on the split model (candle) path via forward_with_lora().
    let local_node_id = shared_state.identity.node_id().clone();
    let is_split_mode = shared_state.config.inference.shard_range.is_some();
    let has_lora = request.lora_adapter.is_some();
    if shared_state
        .model_loaded
        .load(std::sync::atomic::Ordering::Acquire)
        && !is_split_mode
        && !has_lora
    {
        // Local-only inference path (single node has the model loaded)
        let mut executor = shared_state.executor.lock().await;
        tracing::info!(
            request_id = %request.id,
            model = %model_id,
            "Executing inference locally"
        );

        let prompt = {
            let info = shared_state.loaded_model_info.read().await;
            match info.as_ref() {
                Some(i) => chat_template::build_prompt(
                    &request.messages,
                    i.chat_template.as_deref(),
                    &i.bos_token,
                    &i.eos_token,
                    Some(i.name.as_str()),
                ),
                // See local_exec: the model id alone is enough for the family
                // fallback, so don't collapse to ChatML.
                None => chat_template::build_prompt(
                    &request.messages,
                    None,
                    "",
                    "",
                    Some(request.model_id.0.as_str()),
                ),
            }
        };

        // Use streaming generation if token_tx is present
        if let Some(ref tx) = token_tx {
            let tx = tx.clone();
            let mut accumulated = String::new();
            let gen_result = executor.generate_stream(
                &prompt,
                &request.sampling_params,
                |token: &str| -> bool {
                    accumulated.push_str(token);
                    let event = super::types::StreamingTokenEvent {
                        text: token.to_string(),
                        finish_reason: None,
                        matched_stop_sequence: None,
                    };
                    tx.try_send(event).is_ok()
                },
            )?;
            // Send final done event
            let done_event = super::types::StreamingTokenEvent {
                text: String::new(),
                finish_reason: Some(gen_result.finish_reason.as_str().to_string()),
                matched_stop_sequence: gen_result.matched_stop_sequence.clone(),
            };
            if tx.try_send(done_event).is_err() {
                tracing::warn!(
                    request_id = %request.id,
                    "DIAG: streaming done_event send failed — receiver dropped"
                );
            }
            return Ok(InferenceOutput::from_gen_result(
                request.id,
                request.session_id.clone(),
                accumulated,
                gen_result.finish_reason.as_str().to_string(),
                &gen_result,
            ));
        }

        let (content, gen_result) = executor.generate(&prompt, &request.sampling_params)?;

        return Ok(InferenceOutput::from_gen_result(
            request.id,
            request.session_id.clone(),
            content,
            gen_result.finish_reason.as_str().to_string(),
            &gen_result,
        ));
    }

    // ── On-demand shard loading ────────────────────────────────────────
    // If this model has shards on disk but they aren't loaded in split_models,
    // load them now (with LRU eviction if needed) instead of failing.
    {
        let already_loaded = shared_state
            .split_models
            .iter()
            .any(|e| e.key().0 == *model_id);
        if !already_loaded {
            let model_dir = shared_state.model_dir(&model_id.0);
            let has_shards_on_disk = model_dir.exists()
                && (model_dir.join("shard_000.bin").exists()
                    || model_dir.join("model.gguf").exists());

            if has_shards_on_disk {
                tracing::info!(
                    request_id = %request.id,
                    model = %model_id,
                    "On-demand loading: model has shards on disk but not loaded"
                );

                // check_and_load_model has internal TOCTOU guard via loading_models.
                // If another task is already loading, it returns immediately.
                // We then wait on the notify for up to 60s.
                let maybe_notify = shared_state
                    .models
                    .loading_models
                    .get(model_id)
                    .map(|r| r.value().clone());

                if let Some(notify) = maybe_notify {
                    // Another task is loading — wait for it
                    let _ = tokio::time::timeout(
                        std::time::Duration::from_secs(MODEL_LOAD_WAIT_SECS),
                        notify.notified(),
                    )
                    .await;
                } else {
                    // No one loading — trigger load (guard inside check_and_load_model)
                    let vram_budget = crate::model::auto_manage::compute_vram_budget(&shared_state);
                    crate::model::auto_manage::check_and_load_model(
                        &shared_state,
                        model_id,
                        vram_budget,
                    )
                    .await;
                }
            }
        }
    }

    // Distributed inference path: assemble pipeline across nodes
    // Check if the requested model exists in the registry before attempting pipeline assembly.
    // This gives a clearer error than "No model loaded" when the model name is wrong.
    {
        let has_manifest = shared_state.model_registry.get_manifest(model_id).is_some();
        let has_split = shared_state
            .split_models
            .iter()
            .any(|e| e.key().0 == *model_id);
        if !has_manifest && !has_split {
            return Err(shared_state.model_registry.model_not_found_error(model_id));
        }
    }

    // S5: Fire-and-forget DHT provider query to pre-warm the shard holder cache.
    // Results arrive asynchronously and are merged into model_registry by NetworkManager.
    // First request for a model may miss the cache, but subsequent ones benefit.
    let _ = shared_state.dht_query_tx.try_send(model_id.clone());

    let schedule_start = std::time::Instant::now();
    tracing::info!(
        request_id = %request.id,
        model = %model_id,
        "Assembling distributed pipeline"
    );

    // Pipeline affinity: reuse previous pipeline if all nodes are still connected.
    // peer_registry is intentionally preserved across mid-pipeline disconnects
    // (gotcha #86); use connected_node_ids as the actual liveness oracle, same
    // gate as scheduler::gather_candidates.
    let assignment = if let Some(prev) = preferred_pipeline {
        let all_connected = prev.segments.iter().all(|seg| {
            seg.node_id == local_node_id || shared_state.connected_node_ids.contains(&seg.node_id)
        });
        if all_connected && !prev.segments.is_empty() {
            tracing::info!(
                request_id = %request.id,
                segments = prev.segments.len(),
                "Reusing previous pipeline (KV cache affinity)"
            );
            PipelineAssignment {
                request_id: request.id,
                ..prev
            }
        } else {
            assemble_awaiting_dht(&scheduler, model_id, &local_node_id, request.id).await?
        }
    } else {
        assemble_awaiting_dht(&scheduler, model_id, &local_node_id, request.id).await?
    };
    let schedule_ms = schedule_start.elapsed().as_millis() as u64;

    tracing::info!(
        request_id = %request.id,
        segments = assignment.segments.len(),
        standbys = assignment.standbys.len(),
        schedule_ms,
        "DIAG: pipeline assembled"
    );
    for (i, seg) in assignment.segments.iter().enumerate() {
        tracing::info!(
            request_id = %request.id,
            segment = i,
            node = %seg.node_id,
            layer_start = seg.layer_range.0,
            layer_end = seg.layer_range.1,
            "Pipeline segment"
        );
    }

    // Record the route while the assignment is in hand. Region comes from the
    // peer's voluntarily-declared `NodeCapability.region`.
    //
    // Transport is decided by whether we hold a DIRECT connection, not by
    // whether the peer is relay-*capable*: `peer_reachable_via_relay` is an
    // eligibility check that is true for any relay-capable peer, so using it
    // here labelled directly-connected LAN peers as `relayed` (observed live
    // 2026-07-26 — a forward that demonstrably went straight out over LAN QUIC
    // was reported as a relay hop). `connected_node_ids` is the authoritative
    // direct-connection oracle and is what the scheduler itself filters on; a
    // holder that got a segment without being in it is only reachable through
    // the relay tier.
    {
        let segs = crate::inference::trace::segments_from_assignment(
            assignment.segments.iter(),
            &local_node_id,
            |seg| {
                (
                    seg.node_id.clone(),
                    seg.layer_range.0,
                    seg.layer_range.1,
                    shared_state
                        .model_registry
                        .shards_spanned_by_segment(seg)
                        .into_iter()
                        .map(|s| s.index)
                        .collect(),
                    !shared_state.connected_node_ids.contains(&seg.node_id),
                )
            },
            |node| {
                shared_state
                    .peer_registry
                    .get(node)
                    .and_then(|p| p.capability.as_ref().and_then(|c| c.region.clone()))
            },
        );
        trace.mark_assembled(
            crate::inference::trace::classify_route(&segs),
            segs,
            schedule_ms,
        );
    }

    // Store assignment in shared state for monitoring. `active_traces` is
    // inserted here and removed at every site that removes `active_pipelines`,
    // so the two share one lifetime and one cleanup path.
    let assignment_ref = assignment.clone();
    shared_state
        .active_pipelines
        .insert(request.id, assignment.clone());
    shared_state.active_traces.insert(request.id, trace.clone());

    // Execute the distributed pipeline
    let execute_start = std::time::Instant::now();
    let network_tx_for_error = network_tx.clone();
    let mut pipeline = PipelineExecutor::new(
        shared_state.clone(),
        network_tx,
        request.clone(),
        assignment,
    );

    let result = pipeline.execute(token_tx).await;
    let execute_ms = execute_start.elapsed().as_millis() as u64;
    match &result {
        Ok(output) => {
            tracing::info!(
                request_id = %request.id,
                schedule_ms,
                execute_ms,
                total_ms = schedule_ms + execute_ms,
                prompt_tokens = output.prompt_tokens,
                completion_tokens = output.completion_tokens,
                finish_reason = %output.finish_reason,
                "DIAG: execute_request completed successfully"
            );

            // Update trust for all remote peers that participated in the pipeline
            for seg in &assignment_ref.segments {
                if seg.node_id != local_node_id {
                    shared_state.credits.trust_manager.update_trust(
                        &shared_state.peer_registry,
                        &seg.node_id,
                        crate::credit::trust::TrustEvent::InferenceSuccess,
                    );
                }
            }

            // Spot-check: probabilistically verify remote peer output
            spot_check_distributed_result(
                &shared_state,
                &request,
                &assignment_ref,
                &local_node_id,
                output,
            )
            .await;
        }
        Err(ref e) => {
            tracing::error!(
                request_id = %request.id,
                schedule_ms,
                execute_ms,
                error = %e,
                "DIAG: execute_request failed"
            );

            // Apply credit penalty for distributed inference failure — but
            // only when a remote peer's serving actually let us down. See
            // `failure_is_penalty_worthy`.
            let penalty = shared_state.config.pool.credit_rates.penalty_serve_failure;
            let had_remote_segment = assignment_ref
                .segments
                .iter()
                .any(|seg| seg.node_id != local_node_id);
            if !failure_is_penalty_worthy(e, had_remote_segment) {
                tracing::info!(
                    request_id = %request.id,
                    error = %e,
                    had_remote_segment,
                    "Skipping credit penalty — failure is not attributable to a peer"
                );
                crate::inference::pipeline::broadcast_pipeline_error(
                    &network_tx_for_error,
                    request.id,
                    &e.to_string(),
                )
                .await;
                return result;
            }

            // Pool slaves: forward the negative delta to the master so the
            // pool owner sees the penalty, not the slave's local balance.
            // Without this branch the slave's local balance went negative on
            // failures even though it doesn't own the credits, which then
            // gated the slave's own future inference via MIN_BALANCE_FOR_INFERENCE.
            let pool_id_opt: Option<crate::types::NodeId> = {
                let ps = shared_state.credits.pool_state.read().await;
                let me = shared_state.identity.node_id();
                ps.as_ref().and_then(|s| {
                    if s.pool_id != *me {
                        Some(s.pool_id.clone())
                    } else {
                        None
                    }
                })
            };
            if let Some(pid) = pool_id_opt {
                let pool_tx = shared_state.credits.pool_tx.read().await.clone();
                if let Some(tx) = pool_tx {
                    let my_id = shared_state.identity.node_id();
                    let forward = crate::pool::crypto::create_credit_forward(
                        &shared_state.identity,
                        &pid,
                        my_id,
                        &pid,
                        -penalty,
                    );
                    if let Err(e) = tx
                        .send(crate::pool::types::PoolCommand::ProcessCreditForward { forward })
                        .await
                    {
                        tracing::warn!(error = %e, "Failed to forward failure penalty to pool master — falling back to local apply");
                        let _ = crate::credit::ledger::apply_credit_direct_noted(
                            &shared_state.credits.credit_balance,
                            &shared_state.db,
                            -penalty,
                            crate::credit::ledger::CreditDelta::Spending,
                            "inference_charge_retry",
                        )
                        .await;
                    }
                }
            } else if let Err(pe) = crate::credit::ledger::apply_credit_direct_noted(
                &shared_state.credits.credit_balance,
                &shared_state.db,
                -penalty,
                crate::credit::ledger::CreditDelta::Spending,
                "inference_charge",
            )
            .await
            {
                tracing::warn!(error = %pe, "Failed to apply failure penalty");
            } else {
                tracing::info!(
                    penalty,
                    request_id = %request.id,
                    "Applied credit penalty for distributed inference failure"
                );
            }

            // Broadcast pipeline error so peers can update shard availability
            crate::inference::pipeline::broadcast_pipeline_error(
                &network_tx_for_error,
                request.id,
                &e.to_string(),
            )
            .await;
        }
    }
    result
}

/// Should a failed distributed request cost the serving side credits?
///
/// Every failure used to apply the flat `penalty_serve_failure`, including
/// failures that were entirely our own doing. A user debugging the three bugs
/// in the 2026-07-21 report drove their own node from 0 to -470 credits
/// without a single peer ever misbehaving — the penalties came from an
/// unnecessary AllReduce group, a local GPU OOM, and a config setting that
/// wasn't being honoured.
///
/// The rule: penalise only when a remote peer was actually asked to serve and
/// the failure is consistent with that peer letting us down. Anything the
/// local node caused, or that failed before a peer was ever engaged, is free.
///
/// Being wrong in the "penalise" direction is the expensive mistake — it
/// pushes an honest, reachable, correctly-configured node toward Bronze tier
/// and eventually below `MIN_BALANCE_FOR_INFERENCE`, degrading a node for
/// bugs it did not cause. Being wrong the other way just means a genuinely
/// bad peer keeps its credits for one more request; trust scoring and
/// spot-checks still catch it.
fn failure_is_penalty_worthy(err: &SwarmError, had_remote_segment: bool) -> bool {
    // Nobody else was involved — there is no peer to blame.
    if !had_remote_segment {
        return false;
    }
    match err {
        // Local-only failures. `ServiceUnavailable` is by definition "THIS
        // server can't serve" (worker died, subprocess spawn failed, GPU OOM);
        // `Internal` is our own bug; the rest are scheduling/config problems
        // that a peer has no part in.
        SwarmError::ServiceUnavailable(_)
        | SwarmError::NotImplemented(_)
        | SwarmError::LocalOnly(_)
        | SwarmError::Internal(_)
        | SwarmError::Validation(_)
        | SwarmError::Config(_)
        | SwarmError::NotFound(_)
        | SwarmError::Unauthorized(_)
        | SwarmError::NoModelLoaded
        | SwarmError::ModelNotAvailable(_)
        | SwarmError::InsufficientCapacity(_)
        | SwarmError::PrivateModeUnavailable { .. }
        | SwarmError::PromptPrivacyUnavailable { .. }
        | SwarmError::ModelIncompleteInSwarm { .. }
        | SwarmError::InsufficientCredits { .. }
        | SwarmError::InsufficientDisk { .. }
        | SwarmError::Database(_)
        // Failover exhaustion is a summary naming no culprit — the segment
        // failure that triggered it carries its own attribution.
        | SwarmError::SegmentFailoverExhausted(_)
        | SwarmError::PipelineError(_) => false,

        // Everything else reaches the wire: network faults, timeouts waiting
        // on a peer, decryption failures on a peer's payload, shard integrity
        // mismatches, bad signatures, inference errors raised on a remote
        // segment. These are the failures the penalty exists for.
        _ => true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ModelId;

    /// The DHT wait must key on the failure's TYPE, not its wording.
    ///
    /// The scheduler's message became "No reachable node holds the part of …"
    /// on 2026-08-10 and matched neither substring this predicate looked for, so
    /// the wait-and-retry — whose entire purpose is "the holder cache may not be
    /// populated yet" — stopped firing for the commonest missing-holder failure,
    /// silently. Reworded prose must never be able to switch routing off again.
    #[test]
    fn the_dht_wait_survives_a_reworded_message() {
        assert!(assembly_failed_for_lack_of_holders(
            &SwarmError::ModelIncompleteInSwarm {
                model_id: "m".into(),
                layer: 0,
            }
        ));
    }

    /// Only the "nothing known to serve this" failure benefits from waiting for
    /// DHT results. Waiting on anything else just adds latency to a verdict that
    /// will not change.
    #[test]
    fn only_missing_holder_failures_wait_for_dht() {
        assert!(assembly_failed_for_lack_of_holders(
            &SwarmError::PipelineError("No node available for layer 10".into())
        ));
        assert!(assembly_failed_for_lack_of_holders(
            &SwarmError::PipelineError("No shard holders for model".into())
        ));

        for e in [
            // The typed variant this failure now wears (2026-08-16) must keep
            // NOT waiting: the holders are known, one just failed — DHT
            // results cannot change the verdict.
            SwarmError::SegmentFailoverExhausted(
                "Segment 1 failed with no standby available".into(),
            ),
            SwarmError::PipelineError(
                "Timed out waiting for segment result (30s, 6 layers)".into(),
            ),
            SwarmError::ModelNotAvailable(ModelId("m".to_string())),
            SwarmError::ServiceUnavailable("worker spawn failed".into()),
            SwarmError::InsufficientCapacity(ModelId("m".to_string())),
        ] {
            assert!(
                !assembly_failed_for_lack_of_holders(&e),
                "must not wait on: {e}"
            );
        }
    }

    #[test]
    fn local_only_pipeline_is_never_penalised() {
        // No remote segment means no peer to blame, whatever the error.
        assert!(!failure_is_penalty_worthy(
            &SwarmError::Network("connection reset".into()),
            false
        ));
        assert!(!failure_is_penalty_worthy(
            &SwarmError::InferenceTimeout(120),
            false
        ));
    }

    #[test]
    fn locally_attributable_failures_are_not_penalised() {
        // The three bug-report cases, in order.
        assert!(
            !failure_is_penalty_worthy(
                &SwarmError::Internal("AllReduce timeout after 10s for layer 0".into()),
                true
            ),
            "an unnecessary TP group is our scheduling mistake, not a peer's fault"
        );
        assert!(
            !failure_is_penalty_worthy(
                &SwarmError::ServiceUnavailable("worker fatal error: out of memory".into()),
                true
            ),
            "a local GPU OOM says nothing about the peer"
        );
        assert!(!failure_is_penalty_worthy(
            &SwarmError::ModelNotAvailable(ModelId("m".into())),
            true
        ));
        assert!(!failure_is_penalty_worthy(
            &SwarmError::PipelineError("no node available for layer 3".into()),
            true
        ));
        assert!(
            !failure_is_penalty_worthy(
                &SwarmError::SegmentFailoverExhausted(
                    "Segment 1 failed with no standby available".into()
                ),
                true
            ),
            "failover exhaustion names no culprit — attribution belongs to the \
             segment failure that triggered it"
        );
    }

    /// AllReduce splits by attribution, not by subsystem: a peer that puts
    /// non-finite floats on the wire is chargeable, a ring that merely stalls
    /// is not — the slow member could be us, and R146 is the record of what
    /// charging an ambiguous failure costs an honest node.
    #[test]
    fn allreduce_failures_split_by_attribution() {
        // Peer sent poisoned data — `Inference`, chargeable.
        assert!(failure_is_penalty_worthy(
            &SwarmError::Inference(
                "Ring AllReduce step 2: received non-finite values from peer".into()
            ),
            true
        ));
        // Ring stalled — `Internal`, charges nobody.
        assert!(!failure_is_penalty_worthy(
            &SwarmError::Internal("Ring AllReduce timeout at step 2 for layer 0".into()),
            true
        ));
        // Our own partial was corrupt before it left this node — never a peer's
        // fault regardless of who else is in the ring.
        assert!(!failure_is_penalty_worthy(
            &SwarmError::Internal(
                "Ring AllReduce: local partial contains NaN/Inf — IPC corruption or hardware fault"
                    .into()
            ),
            true
        ));
    }

    #[test]
    fn genuine_remote_serve_failures_are_penalised() {
        assert!(failure_is_penalty_worthy(
            &SwarmError::Network("peer dropped mid-forward".into()),
            true
        ));
        assert!(failure_is_penalty_worthy(
            &SwarmError::InferenceTimeout(120),
            true
        ));
        assert!(failure_is_penalty_worthy(
            &SwarmError::ShardIntegrity {
                expected: "aaa".into(),
                actual: "bbb".into()
            },
            true
        ));
        assert!(failure_is_penalty_worthy(
            &SwarmError::DecryptionFailed,
            true
        ));
        assert!(failure_is_penalty_worthy(
            &SwarmError::InvalidSignature,
            true
        ));
        // A peer that took the request and went silent is the canonical
        // "timeout waiting on a peer". It escaped the penalty for as long as
        // it was filed under `PipelineError` (exempt as a local scheduling
        // problem).
        assert!(failure_is_penalty_worthy(
            &SwarmError::PeerUnresponsive("peer never acknowledged".into()),
            true
        ));
    }
}
