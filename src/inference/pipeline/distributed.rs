//! Distributed (across-node) pipeline execution: the per-token generation
//! loop, per-segment forward sequencing, and standby failover.

use crate::error::SwarmError;
use crate::inference::chat_template;
use crate::inference::router::{InferenceOutput, StreamingTokenEvent, StreamingTokenTx};
use crate::types::{LayerForward, LayerResult, NetworkCommand, NetworkFinishReason, TensorFormat};

use super::prompt::{template_from_header, CachedDecoder};
use super::{PipelineExecutor, MAX_PENDING_LAYER_RESULTS};

impl PipelineExecutor {
    /// Execute across multiple network nodes.
    ///
    /// In this phase, we implement the protocol for forwarding activations:
    /// 1. Build initial activation tensor from the prompt
    /// 2. Send LayerForward to each segment in sequence
    /// 3. Wait for the result from the last segment
    /// 4. Collect tokens until finish condition
    ///
    /// If `token_tx` is provided, each decoded token is sent on the channel
    /// as it arrives, enabling true SSE streaming for distributed inference.
    pub(super) async fn execute_distributed(
        &mut self,
        token_tx: Option<StreamingTokenTx>,
    ) -> Result<InferenceOutput, SwarmError> {
        let request_id = self.request.id;
        let max_tokens = self.request.sampling_params.max_tokens;

        if max_tokens == 0 {
            return Ok(InferenceOutput {
                request_id,
                content: String::new(),
                prompt_tokens: 0,
                completion_tokens: 0,
                finish_reason: "length".to_string(),
                session_id: self.request.session_id.clone(),
                token_logprobs: vec![],
                matched_stop_sequence: None,
                trace: None,
            });
        }

        // Item 12 Phase 4: DSD multi-segment greedy speculative. Falls through
        // when fewer than 2 segments (Item 2 covers single-segment) or any
        // other precondition fails (TP groups, non-greedy, no draft, etc.).
        if let Some(out) = self.try_dsd_distributed(token_tx.clone()).await? {
            return Ok(out);
        }

        // Item 2 Phase 3: greedy single-segment distributed speculative
        // path. Requires draft model loaded.
        if let Some(out) = self.try_speculative_distributed(token_tx.clone()).await? {
            return Ok(out);
        }

        // SWARM-SPEC Layer 1 (R136): n-gram-only spec path, no draft
        // model required. Runs BEFORE remote_generate fast path because
        // n-gram hit-rate on code/RAG (99% / 96% from synthetic bench)
        // accepts multiple tokens per round, which beats remote_generate's
        // one-token-per-RTT throughput when the workload is
        // input-grounded. Falls through (Ok(None)) when ngram is disabled, a
        // draft model is configured, the assignment is empty or entirely
        // local, the request is otherwise disqualified, no tokenizer is
        // loaded, or — since the measurement below — the loop has not been
        // accepting enough tokens per round to pay for the logits it returns.
        //
        // This comment used to say it also fell through when "segments aren't
        // 1". It never did, and the difference is expensive: on a multi-segment
        // pipeline this path takes over from the standard loop, which means no
        // chaining (every hop round-trips the coordinator) and a full-vocabulary
        // f32 return per round. The payoff gate is what bounds that now; the
        // wire itself is still the wrong shape for a miss round, and that is
        // written up in `docs/FUTURE_WORK.md`.
        if let Some(out) = self.try_ngram_only_distributed(token_tx.clone()).await? {
            return Ok(out);
        }

        // Remote-generate fast path for single-segment distributed: bypass
        // the per-token coordinator/remote round trip entirely. Remote
        // worker runs the full decode loop and streams tokens back. Falls
        // through on non-eligibility (multi-segment, TP, vision, LoRA,
        // encrypted pipeline).
        if let Some(out) = self.try_remote_generate_fastpath(token_tx.clone()).await? {
            return Ok(out);
        }

        // Read GGUF header ONCE and cache for both prompt building and stop strings
        let header_data: Option<(Option<String>, String, String)> = {
            let model_id = &self.request.model_id;
            let header_path = crate::model::shard::model_dir(
                &self.shared_state.config.node.data_dir,
                &model_id.0,
            )
            .join(crate::model::shard::HEADER_FILENAME);
            template_from_header(&header_path)
        };

        // Build the initial prompt representation
        let prompt = self.build_prompt_with_header(header_data.as_ref()).await;
        let prompt_bytes = prompt.as_bytes().to_vec();

        let mut generated_tokens: Vec<u32> = Vec::new();
        let mut finish_reason = String::new();
        // Outer-scope flag tracking whether a stop-string fired during the
        // decode loop. Drives the post-loop KV-truncate to remote segments
        // for session-keyed requests (gotcha #4 — stop tokens otherwise
        // contaminate the next session turn's KV).
        let mut hit_stop_string_outer = false;
        // Captures the actual user-provided stop string that matched, so the
        // final `InferenceOutput.matched_stop_sequence` mirrors the
        // local-worker contract that Anthropic clients depend on.
        let mut matched_stop_seq: Option<String> = None;

        // Cumulative position for RoPE / KV-cache
        let mut index_pos: usize = 0;
        // Will be set after the first forward pass (once the split model is loaded with tokenizer)
        let mut prompt_token_count: Option<usize> = None;

        // Cached EOS tokens and decoder — extracted once after prefill under a single
        // model lock acquisition. Avoids per-token mutex + DashMap scan.
        let mut cached_eos: Option<std::collections::HashSet<u32>> = None;
        let mut cached_decoder: Option<CachedDecoder> = None;
        let is_streaming = token_tx.is_some();
        // For streaming: accumulate decoded text to avoid redundant final decode
        let mut streamed_text = if is_streaming {
            Some(String::new())
        } else {
            None
        };
        // Text-based stop sequences from the cached GGUF header (read once above)
        let stop_strings = if let Some((ref tmpl, _, _)) = header_data {
            chat_template::extract_stop_strings(tmpl.as_deref())
        } else {
            let info = self.shared_state.loaded_model_info.read().await;
            let tmpl = info.as_ref().and_then(|i| i.chat_template.as_deref());
            chat_template::extract_stop_strings(tmpl)
        };
        // Accumulate decoded text for stop-string matching (both streaming and non-streaming)
        let mut accumulated_text = String::new();

        // T14: Pre-compute vision embeddings before the token generation loop.
        // This decouples vision encoding from the text pipeline — any node with
        // mmproj can encode, and the embeddings travel with LayerForward.
        // Collect images once to avoid scanning messages twice.
        let has_images =
            !crate::inference::vision::collect_images(&self.request.messages).is_empty();
        let mut precomputed_vision: Option<Vec<u8>> = if has_images {
            match self.precompute_vision_embeddings().await {
                Ok(Some(bytes)) => Some(bytes),
                Ok(None) => None,
                Err(e) => {
                    tracing::warn!(
                        request_id = %request_id,
                        error = %e,
                        "Vision pre-computation failed, proceeding without images"
                    );
                    None
                }
            }
        } else {
            None
        };

        // Local embedding privacy: check if we should embed locally before sending
        // activations to the first pipeline segment. This prevents remote nodes from
        // seeing raw token IDs — they only receive hidden-state activation tensors.
        // Auto-enabled when encrypted_pipeline is active (it requires both ends local).
        let model_id = &self.assignment.segments[0].shard_id.model_id;
        let encrypted_for_model = self
            .shared_state
            .encrypted_pipeline_models
            .get(model_id)
            .map(|r| *r.value())
            .unwrap_or(self.shared_state.config.inference.encrypted_pipeline);
        let use_local_embedding =
            self.shared_state.config.inference.local_embedding_privacy || encrypted_for_model;
        let local_embedder = if use_local_embedding {
            self.shared_state
                .local_embedders
                .get(model_id)
                .map(|e| e.value().clone())
        } else {
            None
        };

        // Hoist the EOS fallback set out of the decode loop. The fallback is
        // only consulted on the very first forward (seq_num==0) before
        // `cached_eos` is populated; afterward `cached_eos` is always Some,
        // so allocating a fresh HashSet per token was pure waste.
        //
        // EMPTY, deliberately. It used to hold Llama-2's `</s>` (id 2), which is
        // an ordinary token in every later family — `#` in Qwen2.5 — so any
        // model whose EOS could not be resolved had its reply cut at the first
        // `#`. An unknown end-of-turn lets the reply run long, which is visible;
        // a wrong one truncates in silence.
        let default_eos: std::collections::HashSet<u32> = std::collections::HashSet::new();

        // Token generation loop
        let mut prompt_bytes_opt = Some(prompt_bytes);
        for seq_num in 0..max_tokens {
            // Cancellation observation. Tripped externally by /v1/responses/{id}/cancel
            // (and any other cancel handler that flips the request's cancel flag).
            // We check at the top of the per-token loop so the longest a cancel
            // can sit unobserved is one forward_through_segments.
            if self.request.is_cancelled() {
                tracing::info!(
                    request_id = %request_id,
                    seq_num,
                    "DIAG: inference cancelled externally"
                );
                finish_reason = "stop".to_string();
                break;
            }
            let (activations, pre_embedded) = if let Some(ref embedder) = local_embedder {
                // Local embedding privacy: embed locally, never send raw tokens
                if seq_num == 0 {
                    let prompt =
                        std::str::from_utf8(prompt_bytes_opt.as_ref().unwrap()).unwrap_or("");
                    let (bytes, token_count) = embedder.embed_prompt(prompt)?;
                    // Set prompt_token_count from local tokenization
                    if prompt_token_count.is_none() {
                        prompt_token_count = Some(token_count);
                        index_pos = token_count;
                    }
                    prompt_bytes_opt.take();
                    (bytes, true)
                } else {
                    let last_token = generated_tokens.last().copied().unwrap_or(0);
                    let bytes = embedder.embed_token(last_token)?;
                    (bytes, true)
                }
            } else if seq_num == 0 {
                (
                    prompt_bytes_opt
                        .take()
                        .expect("seq_num==0 implies prompt_bytes set"),
                    false,
                )
            } else {
                // For subsequent tokens, encode the last generated token ID as i64 LE bytes
                // so the first segment can embed it directly.
                let last_token = generated_tokens.last().copied().unwrap_or(0) as i64;
                (last_token.to_le_bytes().to_vec(), false)
            };

            tracing::debug!(
                request_id = %request_id,
                seq_num,
                index_pos,
                activation_bytes = activations.len(),
                generated_so_far = generated_tokens.len(),
                "DIAG: starting forward_through_segments"
            );

            // Forward through each segment. Time only when DEBUG is enabled —
            // the DIAG log below is at debug! level (matches the rest of the
            // DIAG instrumentation in this file), so info-level operation
            // doesn't pay for the per-token Instant::now syscall.
            let fwd_start = if tracing::enabled!(tracing::Level::DEBUG) {
                Some(std::time::Instant::now())
            } else {
                None
            };
            // Attach pre-computed vision on first forward only (take ownership to avoid clone)
            let vision_for_forward = if seq_num == 0 {
                precomputed_vision.take()
            } else {
                None
            };
            match self
                .forward_through_segments(
                    request_id,
                    seq_num,
                    index_pos,
                    activations,
                    vision_for_forward,
                    pre_embedded,
                    &generated_tokens,
                )
                .await
            {
                Ok(result) => {
                    tracing::debug!(
                        request_id = %request_id,
                        seq_num,
                        fwd_ms = fwd_start.map(|s| s.elapsed().as_millis() as u64).unwrap_or(0),
                        tokens = result.token_ids.len(),
                        activations_bytes = result.activations.len(),
                        finish = ?result.finish_reason,
                        logprobs = result.token_logprobs.len(),
                        "DIAG: forward_through_segments returned OK"
                    );
                    // Accumulate per-token logprobs from the final segment.
                    // Empty when the request didn't ask for logprobs, or when
                    // the worker hasn't been extended to compute them on the
                    // per-segment Forward IPC path. The output is drained in
                    // `InferenceOutput.token_logprobs` below.
                    if !result.token_logprobs.is_empty() {
                        if let Ok(mut g) = self.collected_logprobs.lock() {
                            g.extend(result.token_logprobs.iter().cloned());
                        }
                    }
                    // Honor matched_stop_sequence from the remote worker if it
                    // ran its own detection (rare today — most stop-string
                    // matching happens at the coordinator). Coordinator-side
                    // capture below takes precedence on a conflict.
                    if matched_stop_seq.is_none() {
                        if let Some(ref ms) = result.matched_stop_sequence {
                            matched_stop_seq = Some(ms.clone());
                        }
                    }
                    // After the first forward pass, extract everything we need from the model
                    // in a SINGLE lock acquisition: prompt token count, EOS tokens, and
                    // cached decoder for lock-free per-token decoding.
                    if seq_num == 0 {
                        let (ptc, eos, decoder) = self.extract_model_cache(&prompt).await;
                        // For VLM: the <image> token (1 tok) was replaced by N vision
                        // tokens per image. The vision module produces
                        // (image_size/patch_size)^2 + 1 tokens per image. Look up the
                        // actual count from the cached vision module if available.
                        let has_images =
                            crate::inference::vision::has_images(&self.request.messages);
                        let vision_expand = if has_images {
                            let model_id = &self.assignment.segments[0].shard_id.model_id;
                            self.shared_state
                                .vision_modules
                                .get(model_id)
                                .map(|vm| {
                                    let num_patches = vm.value().num_image_tokens();
                                    let num_images: usize =
                                        self.request.messages.iter().map(|m| m.images.len()).sum();
                                    // Each <image> token (1) is replaced by num_patches tokens
                                    num_patches * num_images - num_images
                                })
                                .unwrap_or(0)
                        } else {
                            0
                        };
                        index_pos = ptc + vision_expand;
                        prompt_token_count = Some(ptc + vision_expand);
                        cached_eos = Some(eos.into_iter().collect());
                        cached_decoder = Some(decoder);
                    } else {
                        index_pos += 1;
                    }

                    generated_tokens.extend(&result.token_ids);

                    // Decode and stream each non-EOS token, checking for stop strings.
                    let eos = cached_eos.as_ref().unwrap_or(&default_eos);
                    let decoder = cached_decoder.as_ref();
                    let mut hit_stop_string = false;
                    for &tid in &result.token_ids {
                        if !eos.contains(&tid) {
                            let text = match decoder {
                                Some(d) => d.decode_tokens(&[tid]),
                                None => format!("[{tid}]"),
                            };
                            accumulated_text.push_str(&text);

                            // Check if accumulated text contains a stop string
                            if let Some(stop) = crate::inference::sampling::find_stop_sequence(
                                &accumulated_text,
                                &stop_strings,
                            ) {
                                matched_stop_seq = Some(stop.to_string());
                                // Trim everything from the stop string onwards
                                if let Some(pos) = accumulated_text.find(stop) {
                                    accumulated_text.truncate(pos);
                                    if let Some(ref mut st) = streamed_text {
                                        // Remove the stop string from streamed text too
                                        // Use find (not rfind) to match the first occurrence,
                                        // consistent with accumulated_text truncation above.
                                        if let Some(spos) = st.find(stop) {
                                            st.truncate(spos);
                                        }
                                    }
                                }
                                hit_stop_string = true;
                                break;
                            }

                            if let Some(ref tx) = token_tx {
                                if let Some(ref mut st) = streamed_text {
                                    st.push_str(&text);
                                }
                                if tx
                                    .send(StreamingTokenEvent {
                                        text,
                                        finish_reason: None,
                                        matched_stop_sequence: None,
                                    })
                                    .await
                                    .is_err()
                                {
                                    // Client disconnected — stop generating tokens
                                    tracing::info!(
                                        request_id = %request_id,
                                        seq_num,
                                        "Streaming client disconnected — stopping generation"
                                    );
                                    finish_reason = "stop".to_string();
                                    break;
                                }
                            }
                        }
                    }

                    // Client disconnect already set finish_reason — break outer loop
                    if !finish_reason.is_empty() {
                        break;
                    }

                    if hit_stop_string {
                        hit_stop_string_outer = true;
                        finish_reason = "stop".to_string();
                        if let Some(ref tx) = token_tx {
                            let _ = tx
                                .send(StreamingTokenEvent {
                                    text: String::new(),
                                    finish_reason: Some("stop".to_string()),
                                    matched_stop_sequence: matched_stop_seq.clone(),
                                })
                                .await;
                        }
                        break;
                    }

                    // Check for EOS tokens in the result — the worker may return EOS
                    // as a token ID without setting finish_reason explicitly.
                    if result.token_ids.iter().any(|t| eos.contains(t)) {
                        finish_reason = "stop".to_string();
                        if let Some(ref tx) = token_tx {
                            let _ = tx
                                .send(StreamingTokenEvent {
                                    text: String::new(),
                                    finish_reason: Some("stop".to_string()),
                                    matched_stop_sequence: None,
                                })
                                .await;
                        }
                        break;
                    }

                    if let Some(reason) = result.finish_reason {
                        finish_reason = match reason {
                            NetworkFinishReason::Stop => "stop".to_string(),
                            NetworkFinishReason::MaxTokens => "length".to_string(),
                            NetworkFinishReason::Error(e) => {
                                // Same recovery as the remote-generate sibling:
                                // the class does not survive the wire, and
                                // without it the caller is told this server
                                // broke and the peer is charged for it.
                                return Err(crate::error::reclassify_flattened_error(&e)
                                    .unwrap_or(SwarmError::Inference(e)));
                            }
                        };
                        // Send finish event on streaming channel
                        if let Some(ref tx) = token_tx {
                            let _ = tx
                                .send(StreamingTokenEvent {
                                    text: String::new(),
                                    finish_reason: Some(finish_reason.clone()),
                                    matched_stop_sequence: None,
                                })
                                .await;
                        }
                        break;
                    }
                }
                Err(e) => {
                    // Note: failover for remote-segment timeouts/errors is
                    // attempted INSIDE forward_through_segments
                    // (see failover_segment). Reaching this arm means either
                    // a local-segment failure (which has no automatic
                    // failover; that's a deferred enhancement) or that
                    // failover itself returned an error.
                    crate::log_failure!(
                        &e,
                        request_id = %request_id,
                        error = %e,
                        seq_num,
                        "Pipeline failed and failover (if eligible) was unsuccessful"
                    );
                    return Err(e);
                }
            }
        }

        // If we ran out of tokens without a stop signal
        if generated_tokens.len() as u32 >= max_tokens && finish_reason.is_empty() {
            finish_reason = "length".to_string();
            if let Some(ref tx) = token_tx {
                let _ = tx
                    .send(StreamingTokenEvent {
                        text: String::new(),
                        finish_reason: Some("length".to_string()),
                        matched_stop_sequence: None,
                    })
                    .await;
            }
        }

        // Stop-sequence KV cleanup for session-keyed requests. When a stop
        // string fires mid-decode, the remote KV cache holds tokens up to
        // (and including) the stop tokens — feeding that state into the next
        // session turn would prepend the stop string to the new context.
        // Truncate every remote segment's KV back to `prompt_token_count` so
        // the next turn re-prefills (fast via prefix-cache) without the
        // contaminated suffix. Only matters when session_id is set;
        // request-scoped KV is cleaned up by the per-request TTL anyway.
        let needs_kv_reset = hit_stop_string_outer
            && self.request.session_id.is_some()
            && !self.assignment.segments.is_empty();
        if needs_kv_reset {
            if let Some(ptc) = prompt_token_count {
                self.send_kv_truncate_to_segments(request_id, ptc as u32)
                    .await;
            }
        }

        // Tear down the persistent pipeline stream (if one was opened). Drops
        // the client handle which aborts the per-stream reader/writer tasks.
        if let Some(client) = self.shared_state.pipeline_stream_client.get() {
            client.close(request_id);
        }

        // Strip EOS tokens before decoding (loaded from GGUF metadata)
        let eos_tokens = cached_eos.unwrap_or_default();
        let clean_tokens: Vec<u32> = generated_tokens
            .iter()
            .copied()
            .filter(|t| !eos_tokens.contains(t))
            .collect();

        // For streaming: use already-decoded text. For non-streaming: use accumulated_text
        // (which has stop strings already trimmed), falling back to full decode.
        let mut generated_text = if let Some(text) = streamed_text {
            text
        } else if !accumulated_text.is_empty() {
            accumulated_text
        } else {
            match cached_decoder.as_ref() {
                Some(d) => d.decode_tokens(&clean_tokens),
                None => self.decode_tokens(&clean_tokens).await,
            }
        };

        // Reply text is finalised in exactly one place — see
        // `finalize_reply_text`. This path previously trimmed before scrubbing
        // and skipped the leading-newline cleanup, so it could still return an
        // answer-less reply after the scrub was supposedly everywhere.
        crate::inference::finalize_reply_text(&mut generated_text, &stop_strings);

        // Batch credit write — one DB persist for the entire request instead of per-token.
        // Formula: rate * tokens (no layer multiplier — balanced with consume side).
        // Deliberately no credit earn here. `PipelineExecutor` is built at one
        // production site — the router's coordinator path — so a local segment
        // in this assignment is always work this node is doing for ITSELF, and
        // paying for it credited the node for its own chat. Observed
        // 2026-08-09: a purely local request logged `segment_served_earning
        // +20` alongside the escrow charges for the same request.
        //
        // That contradicts what the product tells users — "earn credits by
        // hosting model shards and serving inference for others", "inference
        // across your own devices is free" — and it inflated `lifetime_earned`
        // with credits no peer ever paid, which is precisely the unexplainable
        // movement the transaction log was added to eliminate. Serving is
        // earned at `SharedState::record_peer_serve`, reached only from the two
        // inbound paths.

        crate::inference::report_short_reply(
            &request_id,
            clean_tokens.len() as u32,
            self.request.sampling_params.max_tokens,
            matched_stop_seq.as_deref(),
        );
        Ok(InferenceOutput {
            request_id,
            content: generated_text,
            prompt_tokens: prompt_token_count.unwrap_or_else(|| prompt.chars().count() / 4) as u32,
            completion_tokens: clean_tokens.len() as u32,
            finish_reason: if finish_reason.is_empty() {
                "stop".to_string()
            } else {
                finish_reason
            },
            session_id: self.request.session_id.clone(),
            token_logprobs: self
                .collected_logprobs
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .drain(..)
                .collect(),
            // Captured at the coordinator above when `find_stop_sequence`
            // fired on the accumulated decoded text; honest source of the
            // user-provided string that triggered termination.
            matched_stop_sequence: matched_stop_seq,
            trace: None,
        })
    }

    /// Send a truncation-only `LayerForward` to every remote segment in the
    /// pipeline so each peer can shrink its KV cache back to `truncate_to`
    /// positions. Used after a stop-string fires on a session-keyed request,
    /// so the next turn doesn't see the contaminating stop tokens. Errors
    /// are logged but not propagated — the request itself has already
    /// completed; failed-truncate just means the next session turn re-
    /// prefills from scratch (which is correct behaviour, just slower).
    async fn send_kv_truncate_to_segments(&self, request_id: uuid::Uuid, truncate_to: u32) {
        for segment in &self.assignment.segments {
            // Skip the local segment — its KV is owned by the worker process,
            // and the per-request TTL plus session-scoped lookup keys handle
            // it correctly. Only remote segments need an explicit signal.
            if segment.node_id == *self.shared_state.identity.node_id() {
                continue;
            }
            let target_peer_bytes = match self.shared_state.resolve_peer_id_bytes(&segment.node_id)
            {
                Some(p) => p,
                None => continue,
            };
            let forward = super::build_kv_truncate_forward(
                request_id,
                segment,
                truncate_to,
                self.shared_state.identity.node_id().0,
            );
            if let Err(e) = self
                .network_tx
                .send(crate::types::NetworkCommand::SendTensor {
                    target_peer_bytes,
                    forward,
                })
                .await
            {
                tracing::debug!(
                    request_id = %request_id,
                    node = %segment.node_id,
                    error = %e,
                    "DIAG: stop-sequence KV-truncate send failed; next session turn re-prefills"
                );
            } else {
                tracing::debug!(
                    request_id = %request_id,
                    node = %segment.node_id,
                    truncate_to,
                    "DIAG: sent stop-sequence KV-truncate to segment"
                );
            }
        }
    }

    /// Forward activation data through all pipeline segments in order.
    ///
    /// If tensor-parallel groups are available for a segment's layer range,
    /// the executor uses layer-by-layer AllReduce across the TP group instead
    /// of sending the full layer range to a single node.
    #[allow(clippy::too_many_arguments)]
    /// Run one forward through the pipeline, surfacing a peer's stated failure.
    ///
    /// The whole body lives in `forward_through_segments_inner`; this wrapper
    /// exists so the error check cannot be skipped. The inner function has six
    /// `Ok` return sites (local last segment, remote last segment, and four
    /// failover paths), and checking each one is the "one invariant, N paths"
    /// mistake this codebase keeps paying for — the verify hops
    /// (`forward_verify_through_segments`, `send_verify_batch`) had the check
    /// and the prefill hops did not, in the same files.
    ///
    /// See [`super::peer_error_from_result`] for what was measured.
    pub(super) async fn forward_through_segments(
        &mut self,
        request_id: uuid::Uuid,
        sequence_num: u32,
        index_pos: usize,
        initial_activations: Vec<u8>,
        precomputed_vision: Option<Vec<u8>>,
        pre_embedded: bool,
        generated_ids: &[u32],
    ) -> Result<LayerResult, SwarmError> {
        let result = self
            .forward_through_segments_inner(
                request_id,
                sequence_num,
                index_pos,
                initial_activations,
                precomputed_vision,
                pre_embedded,
                generated_ids,
            )
            .await?;
        if let Some(err) = super::peer_error_from_result(&result) {
            return Err(err);
        }
        Ok(result)
    }

    // The argument list is the one `forward_through_segments` has always had —
    // this is that function's body, extracted verbatim so the wrapper above can
    // own the error check. Clippy skips the `pub(super)` wrapper under
    // `avoid-breaking-exported-api` and flags only this private half.
    #[allow(clippy::too_many_arguments)]
    async fn forward_through_segments_inner(
        &mut self,
        request_id: uuid::Uuid,
        sequence_num: u32,
        index_pos: usize,
        initial_activations: Vec<u8>,
        precomputed_vision: Option<Vec<u8>>,
        pre_embedded: bool,
        generated_ids: &[u32],
    ) -> Result<LayerResult, SwarmError> {
        let mut activations = initial_activations;
        let num_segments = self.assignment.segments.len();
        let pipeline_start = std::time::Instant::now();

        // How far a chained run has already carried us. When a run of remote
        // segments is handed over in one message, its tail reports back here
        // and the segments in between must not be sent to again.
        let mut chained_through = 0usize;
        // Segments a failed chained run touched, and the position to rewind
        // their KV to before they are re-run: `(first, last, index_pos)`. Every
        // hop of a chain appends this step's positions to its cache whether or
        // not the answer made it home, so the unchained re-run must tell each
        // of them to truncate first or the same positions land twice.
        let mut rewind: Option<(usize, usize, u32)> = None;
        // `while` rather than `for`: a chained failure re-runs the SAME index
        // unchained (no increment), everything else advances at the bottom.
        let mut idx = 0usize;
        while idx < num_segments {
            if idx < chained_through {
                idx += 1;
                continue;
            }
            let is_last = idx == num_segments - 1;
            let segment = &self.assignment.segments[idx];

            // Check if this segment has a tensor-parallel group
            let tp_group = self
                .assignment
                .tp_groups
                .iter()
                .find(|g| {
                    g.layer_range.0 <= segment.layer_range.0
                        && g.layer_range.1 >= segment.layer_range.1
                })
                .cloned();

            // Tensor-parallel execution: layer-by-layer with AllReduce.
            // A `None` outcome means either "no TP group for this segment" or
            // "the TP group failed and we degraded to plain local compute" —
            // both fall through to the standard path below.
            let tp_outcome = match tp_group {
                Some(ref group) => {
                    match self
                        .execute_tp_segment(
                            request_id,
                            sequence_num,
                            index_pos,
                            &activations,
                            segment,
                            group,
                            is_last,
                        )
                        .await
                    {
                        Ok(result) => Some(result),
                        // Graceful degradation: a TP peer that stalls or drops
                        // must not kill a request this node can serve alone.
                        // We hold the segment's full layer range (TP groups are
                        // only formed around a local segment), so reset the
                        // partial KV this request wrote during the failed
                        // AllReduce rounds and recompute the segment locally.
                        Err(e) if segment.node_id == *self.shared_state.identity.node_id() => {
                            tracing::warn!(
                                request_id = %request_id,
                                segment = idx,
                                layers = ?(segment.layer_range.0..segment.layer_range.1),
                                error = %e,
                                "Tensor-parallel segment failed — falling back to local compute"
                            );
                            self.reset_kv_after_tp_failure(request_id, segment, index_pos)
                                .await;
                            None
                        }
                        Err(e) => return Err(e),
                    }
                }
                None => None,
            };

            if let Some(tp_result) = tp_outcome {
                // Parse the tagged result: 0x01 prefix = sampled token, 0x00 = raw activations
                if !tp_result.is_empty() && tp_result[0] == 0x01 {
                    // Last segment returned a sampled token ID
                    let token_id = if tp_result.len() >= 9 {
                        let raw = i64::from_le_bytes(tp_result[1..9].try_into().unwrap());
                        if raw >= 0 && raw <= u32::MAX as i64 {
                            raw as u32
                        } else {
                            tracing::warn!(
                                raw_token = raw,
                                "Out-of-range token ID from peer — clamping to 0"
                            );
                            0u32
                        }
                    } else {
                        0u32
                    };
                    // Check EOS
                    let eos_tokens = self
                        .shared_state
                        .split_models
                        .get(&(
                            segment.shard_id.model_id.clone(),
                            segment.layer_range.0 as usize,
                            segment.layer_range.1 as usize,
                        ))
                        .map(|e| e.value().eos_tokens.clone())
                        .unwrap_or_default();
                    let finish = if eos_tokens.contains(&token_id) {
                        Some(NetworkFinishReason::Stop)
                    } else {
                        None
                    };
                    return Ok(LayerResult {
                        request_id,
                        token_ids: vec![token_id],
                        finish_reason: finish,
                        activations: vec![],
                        sealed_token_ids: None,
                        spec_logits: Vec::new(),
                        matched_stop_sequence: None,
                        token_logprobs: Vec::new(),
                    });
                } else {
                    // Intermediate segment: strip the 0x00 tag and continue
                    activations = if !tp_result.is_empty() {
                        tp_result[1..].to_vec()
                    } else {
                        tp_result
                    };
                }
                idx += 1;
                continue;
            }

            // Standard pipeline execution (no TP)
            let segment_start = std::time::Instant::now();
            // If this is the local node, process locally — move the activation
            // buffer in instead of cloning. We replace `activations` with
            // `result.activations` immediately after, so the previous buffer
            // is dead by then anyway.
            if segment.node_id == *self.shared_state.identity.node_id() {
                let prev_activations = std::mem::take(&mut activations);
                let result = self
                    .process_local_segment(
                        segment,
                        sequence_num,
                        index_pos,
                        prev_activations,
                        if idx == 0 {
                            precomputed_vision.as_deref()
                        } else {
                            None
                        },
                        pre_embedded && idx == 0,
                        generated_ids,
                    )
                    .await?;
                let segment_ms = segment_start.elapsed().as_millis() as u64;
                tracing::debug!(
                    request_id = %request_id,
                    segment = idx,
                    segment_ms,
                    activation_bytes = result.activations.len(),
                    "DIAG: local segment complete"
                );
                self.shared_state.record_segment_timing(
                    request_id,
                    idx as u16,
                    segment_ms as u32,
                    result.activations.len() as u32,
                );
                // Measure ourselves too. Without this the scheduler had no idea
                // what our own hardware costs, so the local node was free by
                // construction and could never lose a comparison against a peer
                // — even a peer that was genuinely faster.
                self.shared_state.record_peer_segment_latency(
                    &segment.node_id,
                    &segment.shard_id.model_id,
                    super::work_kind_for(sequence_num),
                    segment_ms,
                    segment.layer_range.1 - segment.layer_range.0,
                    result.activations.len(),
                );
                if is_last {
                    tracing::info!(
                        request_id = %request_id,
                        num_segments,
                        pipeline_ms = pipeline_start.elapsed().as_millis() as u64,
                        "DIAG: forward_through_segments completed (last segment local)"
                    );
                    return Ok(result);
                }
                // Use hidden-state activations for the next segment
                activations = result.activations;
            } else {
                // Only clone activations when sending over the network
                // T17: Attach vision embeddings on first forward (seq_num==0, first segment)
                // Direct peer chaining: how many segments after this one can
                // take the activations straight from their predecessor?
                //
                // Empty unless the operator enabled it, and empty for anything
                // the planner refuses — a local segment, a peer without the
                // feature, a gap in the layer ranges. Empty means every line
                // below behaves exactly as it did before chaining existed.
                // `generated_ids` is NOT the right question, though it reads
                // like it: it accumulates the completion so far, so it is empty
                // only before the prompt pass and non-empty for every decode
                // step after. Gating on it disabled chaining for the whole
                // per-token phase — which is where the round trips are, and the
                // only reason this exists. What matters is whether the sampler
                // will NEED those ids, which is the condition
                // `apply_repetition_penalties` itself uses.
                let needs_generated_ids = self.request.sampling_params.frequency_penalty != 0.0
                    || self.request.sampling_params.presence_penalty != 0.0;
                let chain: Vec<crate::types::ChainHop> =
                    if self.shared_state.cfg().inference.pipeline_chaining
                        && !needs_generated_ids
                        && !self.chaining_disabled
                    {
                        let st = &self.shared_state;
                        super::plan_chain(
                            &self.assignment.segments,
                            idx,
                            st.identity.node_id(),
                            |n| st.peer_supports_pipeline_chain(n),
                            self.shared_state.cfg().inference.max_chain_hops as usize,
                        )
                    } else {
                        // `generated_ids` are needed by the segment that samples,
                        // and they travel with the coordinator's own forward. In a
                        // chain that forward is built on a serving node, which does
                        // not have them — so a request carrying penalties is not
                        // chained rather than silently losing them.
                        Vec::new()
                    };
                let awaiting_node = chain
                    .last()
                    .map(|h| h.node_id.clone())
                    .unwrap_or_else(|| segment.node_id.clone());
                // "Is this the final segment" has to be asked of the node that
                // ANSWERS, and a chained run answers from its tail. Asking it of
                // the head means a run that ends at the last segment is not
                // recognised as finishing the pipeline, and the coordinator
                // walks off the end of the loop with the reply in its hand.
                let run_is_last = idx + chain.len() == num_segments - 1;

                let vision_for_wire = if idx == 0 && sequence_num == 0 {
                    precomputed_vision.clone()
                } else {
                    None
                };
                let forward = LayerForward {
                    request_id,
                    sequence_num,
                    index_pos: index_pos as u32,
                    activations: activations.clone(),
                    format: TensorFormat::FP32,
                    model_id: segment.shard_id.model_id.clone(),
                    layer_range: segment.layer_range,
                    vision_embeddings: vision_for_wire,
                    chain: chain.clone(),
                    sender_peer_bytes: None,
                    tp_meta: None,
                    // Pipeline sealing: attach our node ID so the final segment
                    // can seal the result tokens for our X25519 key.
                    // Named ONLY on a chained send: it is the reply-to the tail
                    // answers, and it rides the wire as the 0x07 trailer — which
                    // no released node expects on an ordinary one-hop forward.
                    // Unchained, the receiver answers the sender, which IS us.
                    requester_node_id: if chain.is_empty() {
                        None
                    } else {
                        Some(self.shared_state.identity.node_id().0)
                    },
                    // Local embedding privacy: only the first segment of the first
                    // forward needs this flag (subsequent segments receive hidden states anyway).
                    pre_embedded: pre_embedded && idx == 0,
                    // Only the LAST segment samples — others just propagate
                    // hidden state. Sending generated_ids to intermediate
                    // segments is wasted bytes. Send empty for non-last
                    // segments and when the caller passed an empty slice
                    // (penalties == 0 fast path; no wire bloat).
                    generated_ids: if is_last && !generated_ids.is_empty() {
                        generated_ids.to_vec()
                    } else {
                        Vec::new()
                    },
                    adapter_id: None,
                    draft_tokens: Vec::new(),
                    spec_logits_requested: false,
                    // Rewind a segment a failed chained run already ran at this
                    // position (see `rewind`); `None` for every ordinary forward.
                    truncate_kv_to: rewind
                        .filter(|(first, last, _)| idx >= *first && idx <= *last)
                        .map(|(_, _, pos)| pos),
                    chunk_meta: None,
                    sampling: None,
                };

                let target_peer_bytes = self
                    .shared_state
                    .resolve_peer_id_bytes(&segment.node_id)
                    .ok_or_else(|| {
                    SwarmError::Network(format!("No peer_id_bytes for node {}", segment.node_id))
                })?;

                // Register the result channel BEFORE sending so we never miss
                // a fast response.
                if self.shared_state.pending_layer_results.len() >= MAX_PENDING_LAYER_RESULTS {
                    return Err(SwarmError::ServiceUnavailable(
                        "Pipeline overloaded — too many pending layer results".into(),
                    ));
                }
                let (tx, rx) = tokio::sync::oneshot::channel();
                self.shared_state.pending_layer_results.insert(
                    request_id,
                    crate::daemon::state::PendingLayerResult {
                        tx,
                        // Pin to whichever node will actually answer. Without a
                        // chain that is this segment's node; with one it is the
                        // tail, because the hops in between hand the
                        // activations along and never report here. If this
                        // forward times out and we fail over, the abandoned
                        // forward's late error is attributed to THAT node and
                        // must not resolve the standby's waiter.
                        awaiting: Some(awaiting_node.clone()),
                        // Any hop of the run may report a failure it cannot
                        // recover from; see `PendingLayerResult::chain_members`.
                        //
                        // Includes the HEAD, which `awaiting` does not cover
                        // once a chain is planned — that pins the tail. A head
                        // that cannot reach its successor has exactly the same
                        // problem as a hop part-way along, and leaving it out
                        // would have kept the hang this is here to prevent for
                        // the most common chain of all: a run of two.
                        chain_members: std::iter::once(segment.node_id.clone())
                            .chain(chain.iter().map(|h| h.node_id.clone()))
                            .collect(),
                    },
                );
                // NOTE: the dsd.rs / speculative.rs PendingLayerResultGuard
                // pattern (gotcha #45) is NOT applied here because
                // `failover_segment(&mut self, ...)` mid-loop needs `&mut self`
                // while a guard would hold a `&` borrow on
                // `self.shared_state.pending_layer_results` for the full
                // iteration. Every error/failover branch in the loop body has
                // an explicit `pending_layer_results.remove(&request_id)`
                // immediately above the `return Err`/failover call.

                // Per-token call in the decode loop. tracing::info! eagerly
                // formats `%request_id` (UUID Display) and `%segment.node_id`
                // (hex) on every call regardless of subscriber level. Drop
                // to debug! to match the surrounding DIAG: gating; ~4 String
                // allocations per token per remote segment saved.
                tracing::debug!(
                    request_id = %request_id,
                    seq = sequence_num,
                    segment = idx,
                    node = %segment.node_id,
                    activation_bytes = activations.len(),
                    "Sending LayerForward to remote segment"
                );

                // R139 Tier 4K — daemon-side STREAM-chunked send (gated by
                // `inference.streaming_chunked_send`). Splits large
                // activations into K chunks at the wire boundary and ships
                // them sequentially over the SAME persistent stream — QUIC
                // preserves order within a stream so no reorder/loss
                // handling is needed. Chunked send is only wired on the
                // stream path; RR fallback ships the un-chunked forward
                // because the RR ResponseChannel pattern is 1:1 (a future
                // commit can plumb chunked-over-RR with explicit Acks).
                let streaming_cfg = &self.shared_state.config.inference;
                let chunked_eligible = streaming_cfg.streaming_chunked_send
                    && streaming_cfg.persistent_pipeline_stream
                    && (forward.activations.len() as u32)
                        > streaming_cfg.streaming_min_activation_bytes;
                let chunk_size = streaming_cfg.streaming_chunk_size_bytes.max(1) as usize;

                // Persistent pipeline stream path: if enabled AND the client
                // handle is installed, encode + seal locally and ship on the
                // stream. Falls back to NetworkCommand::SendTensor on any
                // setup failure (stream open error, encoding error, etc.).
                let used_stream = if streaming_cfg.persistent_pipeline_stream {
                    if let Some(client) = self.shared_state.pipeline_stream_client.get() {
                        match libp2p::PeerId::from_bytes(&target_peer_bytes) {
                            Ok(peer_id) => {
                                // Build the per-frame slice WITHOUT cloning the
                                // activation buffer on the non-chunked path.
                                // Chunked path owns its frames (already split copies);
                                // non-chunked path borrows the original `&forward`.
                                let chunks: Vec<crate::types::LayerForward>;
                                let frames: &[crate::types::LayerForward] = if chunked_eligible {
                                    chunks = crate::network::pipeline_stream::chunk_layer_forward(
                                        &forward, chunk_size,
                                    );
                                    &chunks
                                } else {
                                    std::slice::from_ref(&forward)
                                };
                                let mut all_ok = true;
                                for chunk in frames {
                                    match crate::network::pipeline_stream::encode_forward_for_wire(
                                        chunk,
                                        &peer_id,
                                        &self.shared_state,
                                    ) {
                                        Ok(payload) => match client
                                            .send_forward(
                                                request_id,
                                                peer_id,
                                                payload,
                                                self.shared_state.clone(),
                                            )
                                            .await
                                        {
                                            Ok(()) => {}
                                            Err(e) => {
                                                tracing::warn!(
                                                    %request_id,
                                                    error = %e,
                                                    "pipeline stream send failed — falling back to RR"
                                                );
                                                client.close(request_id);
                                                all_ok = false;
                                                break;
                                            }
                                        },
                                        Err(e) => {
                                            tracing::warn!(
                                                %request_id,
                                                error = %e,
                                                "pipeline stream encode failed — falling back to RR"
                                            );
                                            all_ok = false;
                                            break;
                                        }
                                    }
                                }
                                all_ok
                            }
                            Err(e) => {
                                tracing::warn!(
                                    %request_id,
                                    error = %e,
                                    "pipeline stream PeerId parse failed — falling back to RR"
                                );
                                false
                            }
                        }
                    } else {
                        false
                    }
                } else {
                    false
                };

                if !used_stream
                    && self
                        .network_tx
                        .send(NetworkCommand::SendTensor {
                            target_peer_bytes: target_peer_bytes.clone(),
                            forward,
                        })
                        .await
                        .is_err()
                {
                    self.shared_state.pending_layer_results.remove(&request_id);
                    return Err(SwarmError::Network(
                        "Failed to send LayerForward".to_string(),
                    ));
                }

                // The deadline covers everything we handed over in one
                // message: a chained run reports back only from its tail, so
                // budgeting for one segment would time out a healthy chain.
                let num_layers = chain
                    .last()
                    .map(|h| h.layer_range.1)
                    .unwrap_or(segment.layer_range.1)
                    - segment.layer_range.0;
                let budget = super::local::SegmentBudget::for_forward(
                    &self.shared_state,
                    &segment.node_id,
                    &segment.shard_id.model_id,
                    super::work_kind_for(sequence_num),
                    num_layers,
                    activations.len(),
                    // Segment 0 of a non-pre-embedded pipeline is handed the
                    // prompt itself; every later hop carries hidden states.
                    if idx == 0 && !pre_embedded {
                        super::local::ActivationUnits::PromptBytes
                    } else {
                        super::local::ActivationUnits::HiddenStates
                    },
                );
                let result = Self::wait_for_result(
                    rx,
                    request_id,
                    idx,
                    &segment.node_id,
                    num_layers,
                    activations.len(),
                    budget,
                )
                .await;

                // A chained run tells us that something went wrong but not
                // WHICH hop it was, so replacing this segment's holder would be
                // a guess — and a wrong guess re-sends the whole run to the
                // same nodes that just failed. Re-run this segment UNCHAINED
                // instead: the plain path names its culprit and fails over per
                // segment, and the input activations are untouched because a
                // chained run consumes nothing until its answer is accepted.
                //
                // Until 2026-08-21 this returned `PeerUnresponsive` with a log
                // line promising a retry, on the theory that the router would
                // re-plan. It did not: the router's transient-failure check
                // matches other wording, and a re-plan would have chained again
                // anyway. Every chained failure was a hard 503 after the full
                // deadline (observed on two machines).
                //
                // The hops that DID run appended this step's positions to their
                // KV, so the re-run carries `truncate_kv_to` for every segment
                // the chain covered. `chaining_disabled` is per request: one bad
                // hand-off says nothing about anyone else's peers.
                if !chain.is_empty() {
                    let failed = match &result {
                        Err(e) => Some(e.to_string()),
                        Ok(r) => match &r.finish_reason {
                            Some(NetworkFinishReason::Error(m)) => Some(m.clone()),
                            _ => None,
                        },
                    };
                    if let Some(reason) = failed {
                        self.shared_state.pending_layer_results.remove(&request_id);
                        if !self.chaining_disabled {
                            self.chaining_disabled = true;
                            rewind = Some((idx, idx + chain.len(), index_pos as u32));
                            tracing::warn!(
                                request_id = %request_id,
                                segment = idx,
                                hops = chain.len(),
                                head = %segment.node_id,
                                tail = %awaiting_node,
                                error = %reason,
                                "chained run failed — re-running this segment unchained for the rest of the request"
                            );
                            continue;
                        }
                        // Unreachable in practice — no chain is planned once
                        // disabled — kept so a future planner change cannot
                        // loop here.
                        return Err(SwarmError::PeerUnresponsive(format!(
                            "chained pipeline of {} hops failed: {reason}",
                            chain.len() + 1
                        )));
                    }
                }

                match result {
                    Ok(result) => {
                        // Check if the remote node returned an error — if so, failover
                        if let Some(NetworkFinishReason::Error(ref err_msg)) = result.finish_reason
                        {
                            // A refusal that describes the REQUEST is reproduced
                            // by every holder, so there is nothing to fail over
                            // TO. Return it as the caller's own error instead of
                            // spending a standby per peer and then reporting the
                            // model as under-replicated.
                            if let Some(err) = super::every_holder_would_refuse(err_msg) {
                                tracing::info!(
                                    request_id = %request_id,
                                    segment = idx,
                                    node = %segment.node_id,
                                    error = %err_msg,
                                    "Remote segment refused the request itself — not failing over"
                                );
                                self.shared_state.pending_layer_results.remove(&request_id);
                                return Err(err);
                            }
                            tracing::warn!(
                                request_id = %request_id,
                                segment = idx,
                                node = %segment.node_id,
                                error = %err_msg,
                                "Remote segment returned error, attempting failover"
                            );
                            // If the holder said it doesn't have the shard, its
                            // gossiped claim is stale — drop it now so failover
                            // and every later request skip it, rather than
                            // re-picking it until the next ShardAnnounce lands.
                            if super::remote_error_means_missing_shard(err_msg) {
                                self.shared_state.retract_shard_holder_claims_for_range(
                                    &segment.shard_id.model_id,
                                    &segment.node_id,
                                    segment.layer_range,
                                    "remote reported the shard data as missing",
                                );
                                // Make the retraction stick for the retry: the DHT still
                                // advertises this holder, so the next assembly would
                                // otherwise re-learn the claim and pick it again.
                                self.shared_state
                                    .blacklist_holder_for_request(request_id, &segment.node_id);
                            }
                            // Remove stale pending entry before failover inserts a new one
                            self.shared_state.pending_layer_results.remove(&request_id);
                            let failover_result = self
                                .failover_segment(
                                    idx,
                                    request_id,
                                    sequence_num,
                                    index_pos,
                                    &activations,
                                    is_last,
                                )
                                .await?;
                            if run_is_last {
                                return Ok(failover_result);
                            }
                            activations = failover_result.activations;
                        } else {
                            let seg_elapsed_ms = segment_start.elapsed().as_millis() as u64;
                            // A chained run answered for every segment it
                            // covered, so the loop must not send to them again.
                            //
                            // Set only for a reply that is actually going to be
                            // used. The activation-shape check below can still
                            // reject this result and fail over — and that
                            // failover replaces only THIS segment's holder,
                            // producing activations for this segment's layers
                            // alone. Committing the skip before that point
                            // would let the loop resume past hops that were
                            // never recomputed, feeding a partial tensor
                            // forward as though the whole chain had run: a
                            // wrong answer rather than an error.
                            let chain_covered = idx + 1 + chain.len();
                            // The measurement covers the whole run — one send,
                            // one reply, however many nodes were between. Charge
                            // it to the head, over the layers actually run,
                            // rather than pretending we timed one segment.
                            let seg_layers = chain
                                .last()
                                .map(|h| h.layer_range.1)
                                .unwrap_or(segment.layer_range.1)
                                - segment.layer_range.0;
                            self.shared_state.record_peer_segment_latency(
                                &segment.node_id,
                                &segment.shard_id.model_id,
                                super::work_kind_for(sequence_num),
                                seg_elapsed_ms,
                                seg_layers,
                                result.activations.len(),
                            );
                            // SWARM-SPEC Layer 2: also record against the
                            // hedge tracker. Keyed on (model, segment_idx,
                            // holder) so different models / segments on the
                            // same physical peer get distinct EWMAs.
                            self.shared_state.record_hedge_observation(
                                &self.request.model_id,
                                idx as u8,
                                &segment.node_id,
                                seg_elapsed_ms as f32,
                            );
                            tracing::debug!(
                                request_id = %request_id,
                                segment = idx,
                                segment_ms = seg_elapsed_ms,
                                activation_bytes = result.activations.len(),
                                "DIAG: remote segment complete"
                            );
                            self.shared_state.record_segment_timing(
                                request_id,
                                idx as u16,
                                seg_elapsed_ms as u32,
                                result.activations.len() as u32,
                            );
                            if run_is_last {
                                tracing::info!(
                                    request_id = %request_id,
                                    num_segments,
                                    pipeline_ms = pipeline_start.elapsed().as_millis() as u64,
                                    "DIAG: forward_through_segments completed (last segment remote)"
                                );
                                // Pipeline sealing: unseal token IDs if the final node sealed them
                                let result = self.unseal_result(result);
                                return Ok(result);
                            }
                            // SEC: Validate intermediate-segment activation shape.
                            // Transformer layers preserve [seq, hidden] shape, so the
                            // byte length must match the input we forwarded. A malicious
                            // peer returning a wrong-shaped tensor would crash the next
                            // segment's worker (gotcha #20) — fail fast and let
                            // failover handle the segment instead.
                            //
                            // BUG-FIX (R105): the shape-preservation invariant only holds
                            // for INTERMEDIATE segments (idx > 0). The first segment
                            // performs token-embedding (8 bytes/token i64 → hidden_dim*4
                            // bytes/token f32 hidden state), so input ≠ output by design.
                            // Without this guard, every decode token whose first
                            // segment is remote tripped a spurious failover — wasting
                            // latency, falsely penalising the first-segment peer's trust
                            // score, and risking a hard fail when no standby is
                            // available. Skip the check for idx == 0 unless the input
                            // is already pre-embedded (in which case the shape DOES
                            // preserve and the check is meaningful).
                            let is_embedding_expansion = idx == 0 && !pre_embedded;
                            if !is_embedding_expansion
                                && result.activations.len() != activations.len()
                            {
                                tracing::warn!(
                                    request_id = %request_id,
                                    segment = idx,
                                    node = %segment.node_id,
                                    expected = activations.len(),
                                    got = result.activations.len(),
                                    "Remote segment returned wrong activation shape — failing over"
                                );
                                self.shared_state.pending_layer_results.remove(&request_id);
                                let failover_result = self
                                    .failover_segment(
                                        idx,
                                        request_id,
                                        sequence_num,
                                        index_pos,
                                        &activations,
                                        is_last,
                                    )
                                    .await?;
                                // The standby covered THIS segment only, so
                                // the rest of the run still has to be done. Do
                                // not commit the skip, and ask the ordinary
                                // per-segment question about finishing rather
                                // than the run's.
                                if is_last {
                                    return Ok(failover_result);
                                }
                                activations = failover_result.activations;
                            } else {
                                // The chain's answer is accepted, so the
                                // segments it covered are genuinely done.
                                chained_through = chain_covered;
                                activations = result.activations;
                            }
                        }
                    }
                    Err(e) => {
                        // Timeout or channel drop — remove stale entry and failover
                        self.shared_state.pending_layer_results.remove(&request_id);
                        tracing::warn!(
                            request_id = %request_id,
                            segment = idx,
                            node = %segment.node_id,
                            error = %e,
                            seq = sequence_num,
                            segment_ms = segment_start.elapsed().as_millis() as u64,
                            "Remote segment timed out, attempting failover"
                        );
                        let failover_result = self
                            .failover_segment(
                                idx,
                                request_id,
                                sequence_num,
                                index_pos,
                                &activations,
                                is_last,
                            )
                            .await?;
                        if is_last {
                            return Ok(failover_result);
                        }
                        activations = failover_result.activations;
                    }
                }
            }
            idx += 1;
        }

        Err(SwarmError::PipelineError(
            "Pipeline completed without producing a result".to_string(),
        ))
    }

    /// Tell a node we are abandoning to stop working on this segment.
    ///
    /// Without this it never finds out. It computes the whole forward to
    /// completion and every other request that arrives meanwhile queues
    /// behind work whose result nobody will read. Measured on two machines:
    /// a ~2000-token prefill left a CPU node saturated for several minutes
    /// after the coordinator had already given up, and an unrelated short
    /// request sent during that window failed for no reason of its own —
    /// then succeeded in 42s once the node went idle.
    ///
    /// Sent BEFORE the standby search, and regardless of its outcome,
    /// because the case that hurt had NO standby: the request was already
    /// lost, and the only thing still worth doing was freeing the peer.
    ///
    /// Best-effort by design. `CancelInference` is relay-eligible, so a
    /// NAT'd peer is reachable, but a peer that never receives it is no
    /// worse off than before. A peer that has already finished treats it as
    /// a no-op ("no in-flight decode for request").
    ///
    /// NOTE: today only the remote-generate path registers an abort handle,
    /// so a peer serving a *segment* will log that no-op rather than
    /// actually stopping. Sending it is still the correct half to ship
    /// first — it costs one small message, it is what the peer-side change
    /// will need in place, and it already stops us treating a written-off
    /// node as idle. See `docs/FUTURE_WORK.md` for the peer-side half.
    async fn cancel_segment_on(
        &self,
        node_id: &crate::types::NodeId,
        request_id: uuid::Uuid,
        segment_idx: usize,
    ) {
        let Some(target_peer_bytes) = self.shared_state.resolve_peer_id_bytes(node_id) else {
            return;
        };
        let _ = self
            .network_tx
            .send(NetworkCommand::SendDirectMessage {
                target_peer_bytes,
                message: crate::types::SwarmMessage::CancelInference(
                    swarmllm_types::CancelInference { request_id },
                ),
                delivery_request_id: None,
            })
            .await;
        // info!, not debug!. The receiving side logs this at debug when it
        // finds nothing to abort (the normal case today, and normal for
        // hedge losers), so at default verbosity there is otherwise NO
        // record anywhere that the cancel was sent — which made the send
        // unverifiable in exactly the situation an operator cares about.
        tracing::info!(
            request_id = %request_id,
            abandoned_node = %node_id,
            segment = segment_idx,
            "DIAG: asked the abandoned node to stop working on this segment"
        );
    }

    /// Hand a failed segment to a standby — and, if that standby fails too,
    /// to the next one, until one answers or none is left.
    ///
    /// **A standby's error is a failure of that standby, not the segment's
    /// output.** Until 2026-09-02 this returned whatever the first standby
    /// sent back, and the segment loop took it as the segment's result. A
    /// standby that REFUSES the segment — out of memory, a shard it does not
    /// hold — answers with an error `LayerResult` whose activations are
    /// empty, and those empty bytes were forwarded to the next segment, whose
    /// worker failed them as `Internal error: Tensor bytes too short`: an
    /// internal error, blamed on a segment that was fine. Measured on the
    /// live swarm 2026-09-01 (gotcha #435): segment 0's standby answered
    /// "needs about 10362 MB of memory" in 1.1 s, segment 1 was then sent 0
    /// bytes, and the request failed as "Segment 1 failed with no standby
    /// available". An external tester reported the same `Tensor bytes too
    /// short` on the same model the same day, once, gone on retry — which is
    /// what a failover that happens to land on a refusing standby looks like
    /// from outside.
    ///
    /// Three things this keeps. Every node the segment has already been
    /// tried on — the failed holder and each standby that failed in turn — is
    /// excluded from the search, so two failing standbys cannot hand the
    /// segment back and forth until the request's deadline. A refusal that
    /// describes the REQUEST (`every_holder_would_refuse`) is returned as the
    /// caller's own error at once, exactly as the primary path does, because
    /// every standby would reproduce it. And a standby that says it does not
    /// hold the shard loses its claim, as a primary holder would.
    async fn failover_segment(
        &mut self,
        failed_idx: usize,
        request_id: uuid::Uuid,
        sequence_num: u32,
        index_pos: usize,
        activations: &[u8],
        _is_last: bool,
    ) -> Result<LayerResult, SwarmError> {
        let failed_segment = self.assignment.segments[failed_idx].clone();
        // Everyone this segment has been tried on for this request.
        let mut tried: Vec<crate::types::NodeId> = vec![failed_segment.node_id.clone()];
        // The node abandoned on the previous round — the failed holder first,
        // then each standby that failed in turn.
        let mut abandoned = failed_segment.node_id.clone();
        let mut last_failure: Option<String> = None;

        loop {
            self.cancel_segment_on(&abandoned, request_id, failed_idx)
                .await;

            // Find a standby covering this segment's layer range that has not
            // already failed it.
            let standby = self
                .assignment
                .standbys
                .iter()
                .find(|s| {
                    s.layer_range.0 <= failed_segment.layer_range.0
                        && s.layer_range.1 >= failed_segment.layer_range.1
                        && !tried.contains(&s.node_id)
                })
                .cloned();

            let Some(backup) = standby else {
                tracing::error!(
                    request_id = %request_id,
                    failed_segment = failed_idx,
                    failed_node = %failed_segment.node_id,
                    failed_layer_range = ?failed_segment.layer_range,
                    tried = ?tried.iter().map(|n| n.to_string()).collect::<Vec<_>>(),
                    last_failure = ?last_failure,
                    total_standbys = self.assignment.standbys.len(),
                    standby_nodes = ?self.assignment.standbys.iter().map(|s| format!("{}[{:?}]", s.node_id, s.layer_range)).collect::<Vec<_>>(),
                    "DIAG: NO standby available for failed segment — pipeline will fail"
                );
                // `SegmentFailoverExhausted`, not `PipelineError`: 503, so
                // the caller learns nothing is wrong with their request or
                // this node — there was simply nobody free to take the
                // segment over. See the variant's doc for why neither
                // `ModelIncompleteInSwarm` nor `ServiceUnavailable` fits.
                //
                // The last standby's own words are carried along so the caller
                // is told WHY the last machine that was tried could not serve
                // it, rather than only that it could not.
                return Err(SwarmError::SegmentFailoverExhausted(exhausted_message(
                    failed_idx,
                    last_failure.as_deref(),
                )));
            };

            tracing::warn!(
                request_id = %request_id,
                failed_node = %abandoned,
                backup_node = %backup.node_id,
                failed_layer_range = ?failed_segment.layer_range,
                backup_layer_range = ?backup.layer_range,
                segment = failed_idx,
                attempt = tried.len(),
                total_segments = self.assignment.segments.len(),
                total_standbys = self.assignment.standbys.len(),
                "DIAG: failing over to standby node"
            );

            // Register a response channel BEFORE sending the request.
            // SEC: a RAII guard removes the entry on every error path
            // (including a panic between insert and wait). Without it a
            // failed `wait_for_result` would leak one slot per double
            // timeout — at MAX_PENDING_LAYER_RESULTS the pipeline starts
            // rejecting all new requests with ServiceUnavailable.
            let (tx, rx) = tokio::sync::oneshot::channel();
            self.shared_state.pending_layer_results.insert(
                request_id,
                crate::daemon::state::PendingLayerResult {
                    tx,
                    // Pin to the standby. The forward we just gave up on is
                    // still outstanding to the failed node; when it is
                    // reaped, its synthetic error carries this same
                    // `request_id` and would otherwise resolve THIS waiter,
                    // discarding the standby's real result.
                    awaiting: Some(backup.node_id.clone()),
                    chain_members: Vec::new(),
                },
            );
            let mut pending_guard = super::PendingLayerResultGuard::new(
                &self.shared_state.pending_layer_results,
                request_id,
            );

            // Send to backup node via directed tensor protocol
            let forward = LayerForward {
                request_id,
                sequence_num,
                index_pos: index_pos as u32,
                activations: activations.to_vec(),
                format: TensorFormat::FP32,
                model_id: backup.shard_id.model_id.clone(),
                layer_range: backup.layer_range,
                tp_meta: None,
                vision_embeddings: None,
                chain: Vec::new(),
                sender_peer_bytes: None,
                // Unchained failover: the standby answers its sender, which is
                // us. Naming ourselves would put a 0x07 trailer on a frame an
                // older standby does not expect.
                requester_node_id: None,
                pre_embedded: false,
                generated_ids: Vec::new(),
                adapter_id: None,
                draft_tokens: Vec::new(),
                spec_logits_requested: false,
                truncate_kv_to: None,
                chunk_meta: None,
                sampling: None,
            };

            let target_peer_bytes = match self.shared_state.resolve_peer_id_bytes(&backup.node_id) {
                Some(b) => b,
                None => {
                    return Err(SwarmError::Network(format!(
                        "No peer_id_bytes for backup node {}",
                        backup.node_id
                    )));
                }
            };
            if self
                .network_tx
                .send(NetworkCommand::SendTensor {
                    target_peer_bytes,
                    forward,
                })
                .await
                .is_err()
            {
                return Err(SwarmError::Network(
                    "Failed to send to standby node".to_string(),
                ));
            }

            // Wait for standby response via the oneshot channel
            let num_layers = failed_segment.layer_range.1 - failed_segment.layer_range.0;
            let budget = super::local::SegmentBudget::for_forward(
                &self.shared_state,
                &backup.node_id,
                &backup.shard_id.model_id,
                super::work_kind_for(sequence_num),
                num_layers,
                activations.len(),
                if failed_idx == 0 {
                    super::local::ActivationUnits::PromptBytes
                } else {
                    super::local::ActivationUnits::HiddenStates
                },
            );
            let result = Self::wait_for_result(
                rx,
                request_id,
                failed_idx,
                &backup.node_id,
                num_layers,
                activations.len(),
                budget,
            )
            .await;

            let result = match result {
                Ok(result) => {
                    // dispatcher already removed the entry on deliver
                    pending_guard.disarm();
                    result
                }
                Err(e) => {
                    // The guard removes the waiter as it goes out of scope.
                    tracing::warn!(
                        request_id = %request_id,
                        segment = failed_idx,
                        standby = %backup.node_id,
                        error = %e,
                        "Standby did not answer — trying the next standby"
                    );
                    last_failure = Some(e.to_string());
                    tried.push(backup.node_id.clone());
                    abandoned = backup.node_id;
                    continue;
                }
            };

            if let Some(NetworkFinishReason::Error(ref err_msg)) = result.finish_reason {
                if let Some(err) = super::every_holder_would_refuse(err_msg) {
                    tracing::info!(
                        request_id = %request_id,
                        segment = failed_idx,
                        standby = %backup.node_id,
                        error = %err_msg,
                        "Standby refused the request itself — not trying another"
                    );
                    return Err(err);
                }
                if super::remote_error_means_missing_shard(err_msg) {
                    self.shared_state.retract_shard_holder_claims_for_range(
                        &backup.shard_id.model_id,
                        &backup.node_id,
                        backup.layer_range,
                        "standby reported the shard data as missing",
                    );
                    self.shared_state
                        .blacklist_holder_for_request(request_id, &backup.node_id);
                }
                tracing::warn!(
                    request_id = %request_id,
                    segment = failed_idx,
                    standby = %backup.node_id,
                    error = %err_msg,
                    "Standby returned an error — trying the next standby"
                );
                last_failure = Some(err_msg.clone());
                tried.push(backup.node_id.clone());
                abandoned = backup.node_id;
                continue;
            }

            // Update the assignment so subsequent tokens use the standby
            // directly, avoiding repeated failover + 30s timeout per token.
            self.assignment.segments[failed_idx].node_id = backup.node_id;
            self.assignment.segments[failed_idx].layer_range = backup.layer_range;

            return Ok(result);
        }
    }
}

/// How many characters of a standby's own failure message the exhaustion
/// error carries. A worker's refusal names sizes and context lengths and can
/// run to several hundred characters; the caller needs the reason, not the
/// arithmetic.
const EXHAUSTED_REASON_MAX_CHARS: usize = 200;

/// The message a request fails with when every standby for `segment` has been
/// tried, carrying the last standby's stated reason where there is one.
pub(super) fn exhausted_message(segment: usize, last_failure: Option<&str>) -> String {
    let base = format!("Segment {segment} failed with no standby available");
    match last_failure.map(str::trim).filter(|s| !s.is_empty()) {
        None => base,
        Some(reason) => {
            let shown: String = reason.chars().take(EXHAUSTED_REASON_MAX_CHARS).collect();
            let ellipsis = if shown.len() < reason.len() {
                "…"
            } else {
                ""
            };
            format!("{base} (last standby: {shown}{ellipsis})")
        }
    }
}
