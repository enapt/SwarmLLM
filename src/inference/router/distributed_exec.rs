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
        // AtomicU64 increment — try_write() silently dropped under contention.
        shared_state
            .metrics
            .requests_served_atomic
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        // Update Prometheus metrics
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
                if let Some(ref tx) = *shared_state.credits.pool_tx.read().await {
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
                if let Err(e) = crate::credit::ledger::apply_credit_direct(
                    &shared_state.credits.credit_balance,
                    &shared_state.db,
                    -spent,
                    crate::credit::ledger::CreditDelta::Spending,
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
    let mut handles = Vec::with_capacity(batch.len());

    for queued in batch {
        let shared_state = shared_state.clone();
        let network_tx = network_tx.clone();
        let scheduler = scheduler.clone();
        let active_count = active_count.clone();
        let queue_notify = queue_notify.clone();

        handles.push(tokio::spawn(async move {
            let request = queued.request;
            let result_tx = queued.result_tx;
            let token_tx = queued.token_tx;

            let output = execute_request(
                shared_state.clone(),
                network_tx,
                scheduler,
                request.clone(),
                token_tx,
                None, // No pipeline affinity for batched requests
            )
            .await;

            finalize_request(&shared_state, &request, &output, None).await;
            shared_state.active_pipelines.remove(&request.id);
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
        }));
    }

    // Wait for all requests in the batch to complete.
    // Only decrement if the task panicked BEFORE it could decrement itself.
    for handle in handles {
        match handle.await {
            Ok(_) => {} // task already decremented
            Err(_) => {
                active_count.fetch_sub(1, Ordering::Relaxed);
                queue_notify.notify_one();
            }
        }
    }
}

/// Execute a single inference request — either locally or via distributed pipeline.
pub(super) async fn execute_request(
    shared_state: Arc<SharedState>,
    network_tx: mpsc::Sender<NetworkCommand>,
    scheduler: PipelineScheduler,
    request: InferenceRequest,
    token_tx: Option<StreamingTokenTx>,
    preferred_pipeline: Option<PipelineAssignment>,
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
                ),
                None => chat_template::chatml_fallback(&request.messages),
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
                    };
                    tx.try_send(event).is_ok()
                },
            )?;
            // Send final done event
            let done_event = super::types::StreamingTokenEvent {
                text: String::new(),
                finish_reason: Some(gen_result.finish_reason.as_str().to_string()),
            };
            if tx.try_send(done_event).is_err() {
                tracing::warn!(
                    request_id = %request.id,
                    "DIAG: streaming done_event send failed — receiver dropped"
                );
            }
            return Ok(InferenceOutput {
                request_id: request.id,
                content: accumulated,
                prompt_tokens: gen_result.prompt_tokens,
                completion_tokens: gen_result.completion_tokens,
                finish_reason: gen_result.finish_reason.as_str().to_string(),
                session_id: request.session_id.clone(),
                token_logprobs: vec![],
            });
        }

        let (content, gen_result) = executor.generate(&prompt, &request.sampling_params)?;

        return Ok(InferenceOutput {
            request_id: request.id,
            content,
            prompt_tokens: gen_result.prompt_tokens,
            completion_tokens: gen_result.completion_tokens,
            finish_reason: gen_result.finish_reason.as_str().to_string(),
            session_id: request.session_id.clone(),
            token_logprobs: vec![],
        });
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

    // Pipeline affinity: reuse previous pipeline if all nodes are still connected
    let assignment = if let Some(prev) = preferred_pipeline {
        let all_connected = prev.segments.iter().all(|seg| {
            seg.node_id == local_node_id || shared_state.peer_registry.contains_key(&seg.node_id)
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
            scheduler.assemble_pipeline_for(model_id, &local_node_id, request.id)?
        }
    } else {
        scheduler.assemble_pipeline_for(model_id, &local_node_id, request.id)?
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

    // Store assignment in shared state for monitoring
    let assignment_ref = assignment.clone();
    shared_state
        .active_pipelines
        .insert(request.id, assignment.clone());

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

            // Apply credit penalty for distributed inference failure.
            // Pool slaves: forward the negative delta to the master so the
            // pool owner sees the penalty, not the slave's local balance.
            // Without this branch the slave's local balance went negative on
            // failures even though it doesn't own the credits, which then
            // gated the slave's own future inference via MIN_BALANCE_FOR_INFERENCE.
            let penalty = shared_state.config.pool.credit_rates.penalty_serve_failure;
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
                if let Some(ref tx) = *shared_state.credits.pool_tx.read().await {
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
                        let _ = crate::credit::ledger::apply_credit_direct(
                            &shared_state.credits.credit_balance,
                            &shared_state.db,
                            -penalty,
                            crate::credit::ledger::CreditDelta::Spending,
                        )
                        .await;
                    }
                }
            } else if let Err(pe) = crate::credit::ledger::apply_credit_direct(
                &shared_state.credits.credit_balance,
                &shared_state.db,
                -penalty,
                crate::credit::ledger::CreditDelta::Spending,
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
