//! Local (in-process) batched inference path. Used when the entire model is
//! loaded in the daemon's executor and split mode is disabled.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use crate::daemon::SharedState;
use crate::error::SwarmError;
use crate::inference::chat_template;

use super::distributed_exec::finalize_request;
use super::types::{InferenceOutput, QueuedRequest, StreamingTokenEvent};

/// RAII guard that decrements active_count for any unprocessed batch items on drop.
/// Ensures active_count is always decremented even if batch processing panics mid-loop.
pub(super) struct BatchCleanup {
    pub(super) active_count: Arc<AtomicUsize>,
    pub(super) queue_notify: Arc<tokio::sync::Notify>,
    pub(super) remaining: usize,
}

impl BatchCleanup {
    fn complete_one(&mut self) {
        if self.remaining > 0 {
            self.active_count.fetch_sub(1, Ordering::Relaxed);
            self.remaining -= 1;
            // Wake drain_queue so the next queued request can dispatch.
            self.queue_notify.notify_one();
        }
    }
}

impl Drop for BatchCleanup {
    fn drop(&mut self) {
        if self.remaining > 0 {
            self.active_count
                .fetch_sub(self.remaining, Ordering::Relaxed);
            self.queue_notify.notify_one();
        }
    }
}

/// Execute a batch of requests locally, sharing the model lock.
///
/// Acquires the executor mutex once and processes all requests sequentially.
/// Each request gets its own generation call and independent output.
pub(super) async fn execute_local_batch(
    shared_state: Arc<SharedState>,
    batch: Vec<QueuedRequest>,
    active_count: Arc<AtomicUsize>,
    queue_notify: Arc<tokio::sync::Notify>,
) {
    let mut executor = shared_state.executor.lock().await;
    let batch_size = batch.len();
    let mut cleanup = BatchCleanup {
        active_count: active_count.clone(),
        queue_notify: queue_notify.clone(),
        remaining: batch_size,
    };

    tracing::info!(batch_size, "Executing local inference batch");

    for queued in batch {
        let request = queued.request;
        let result_tx = queued.result_tx;
        let token_tx = queued.token_tx;

        // Honor external cancel between batch items. Without this, a
        // cancelled request still runs full generation under the executor
        // mutex — blocking every later request in the batch (and any new
        // request, since the mutex is held end-to-end). Same cancel
        // contract as execute_distributed line 174. Falls through to the
        // standard finalize path with an empty-content "stop" output.
        let output = if request.is_cancelled() {
            tracing::info!(
                request_id = %request.id,
                "DIAG: local batch item cancelled externally before generation"
            );
            // If the request is streaming, emit a final stop event so the
            // SSE client closes cleanly.
            if let Some(ref tx) = token_tx {
                let _ = tx.try_send(StreamingTokenEvent {
                    text: String::new(),
                    finish_reason: Some("stop".to_string()),
                    matched_stop_sequence: None,
                });
            }
            Ok(InferenceOutput {
                request_id: request.id,
                content: String::new(),
                prompt_tokens: 0,
                completion_tokens: 0,
                finish_reason: "stop".to_string(),
                session_id: request.session_id.clone(),
                token_logprobs: vec![],
                matched_stop_sequence: None,
            })
        } else if executor.is_loaded() {
            // Hold the loaded_model_info read lock once and derive both the
            // chat-templated prompt and the stop-string list from a single
            // guard. Avoids re-acquiring the lock per batch item.
            let (prompt, local_stop_strings) = {
                let info = shared_state.loaded_model_info.read().await;
                let prompt = match info.as_ref() {
                    Some(i) => chat_template::build_prompt(
                        &request.messages,
                        i.chat_template.as_deref(),
                        &i.bos_token,
                        &i.eos_token,
                        Some(i.name.as_str()),
                    ),
                    // No loaded-model metadata, but we still know WHICH model
                    // was asked for — enough for the family fallback to pick a
                    // format. Going straight to ChatML here would prompt a
                    // Llama-3 or Mistral model in a foreign format, which is the
                    // failure this release is about.
                    None => chat_template::build_prompt(
                        &request.messages,
                        None,
                        "",
                        "",
                        Some(request.model_id.0.as_str()),
                    ),
                };
                let stops = chat_template::extract_stop_strings(
                    info.as_ref().and_then(|i| i.chat_template.as_deref()),
                );
                (prompt, stops)
            };

            tracing::info!(
                request_id = %request.id,
                model = %request.model_id,
                "Executing inference locally (batched)"
            );

            // Use streaming generation if the request has a token channel
            if let Some(ref tx) = token_tx {
                let tx = tx.clone();
                let session_id = request.session_id.clone();
                let mut accumulated = String::new();
                let stop_strings = local_stop_strings.clone();
                let mut hit_stop = false;
                match executor.generate_stream(
                    &prompt,
                    &request.sampling_params,
                    |token: &str| -> bool {
                        accumulated.push_str(token);
                        // Check for chat template stop strings
                        if let Some(stop) = stop_strings
                            .iter()
                            .find(|s| accumulated.contains(s.as_str()))
                        {
                            // Truncate accumulated text at the stop string
                            if let Some(pos) = accumulated.find(stop.as_str()) {
                                accumulated.truncate(pos);
                            }
                            hit_stop = true;
                            return false; // Signal to stop generation
                        }
                        let event = StreamingTokenEvent {
                            text: token.to_string(),
                            finish_reason: None,
                            matched_stop_sequence: None,
                        };
                        tx.try_send(event).is_ok()
                    },
                ) {
                    Ok(gen_result) => {
                        let finish = if hit_stop {
                            "stop".to_string()
                        } else {
                            gen_result.finish_reason.as_str().to_string()
                        };
                        // Send final done event
                        let done_event = StreamingTokenEvent {
                            text: String::new(),
                            finish_reason: Some(finish.clone()),
                            matched_stop_sequence: gen_result.matched_stop_sequence.clone(),
                        };
                        let _ = tx.try_send(done_event);
                        Ok(InferenceOutput::from_gen_result(
                            request.id,
                            session_id,
                            accumulated,
                            finish,
                            &gen_result,
                        ))
                    }
                    Err(e) => Err(e),
                }
            } else {
                match executor.generate(&prompt, &request.sampling_params) {
                    Ok((mut content, gen_result)) => {
                        // Second pass with the TEMPLATE-derived stops, which the
                        // executor never sees (it only knows the caller's own
                        // `params.stop`). Same finaliser, so the ordering rule
                        // cannot drift from the other reply paths.
                        let mut finish = gen_result.finish_reason.as_str().to_string();
                        if crate::inference::finalize_reply_text(&mut content, &local_stop_strings)
                            .is_some()
                        {
                            finish = "stop".to_string();
                        }
                        Ok(InferenceOutput::from_gen_result(
                            request.id,
                            request.session_id.clone(),
                            content,
                            finish,
                            &gen_result,
                        ))
                    }
                    Err(e) => Err(e),
                }
            }
        } else {
            Err(SwarmError::NoModelLoaded)
        };

        finalize_request(&shared_state, &request, &output, None).await;
        shared_state.active_pipelines.remove(&request.id);
        cleanup.complete_one();
        if result_tx.send(output).is_err() {
            tracing::warn!(
                request_id = %request.id,
                "DIAG: batch result_tx receiver dropped"
            );
        }
    }

    tracing::debug!(batch_size, "Local batch complete");
}
