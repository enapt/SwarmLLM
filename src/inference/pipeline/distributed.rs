//! Distributed (across-node) pipeline execution: the per-token generation
//! loop, per-segment forward sequencing, and standby failover.

use crate::error::SwarmError;
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
        // Read here rather than beside the decode loop below: the five
        // alternative paths tried first each run a complete decode, and
        // `keeping_the_partial` needs the same answer they do.
        let is_streaming = token_tx.is_some();

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
        let outcome = self.try_dsd_distributed(token_tx.clone()).await;
        if let Some(out) = self.keeping_the_partial(outcome, token_tx.as_ref()).await? {
            return Ok(out);
        }

        // Item 2 Phase 3: greedy single-segment distributed speculative
        // path. Requires draft model loaded.
        let outcome = self.try_speculative_distributed(token_tx.clone()).await;
        if let Some(out) = self.keeping_the_partial(outcome, token_tx.as_ref()).await? {
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
        let outcome = self.try_ngram_only_distributed(token_tx.clone()).await;
        if let Some(out) = self.keeping_the_partial(outcome, token_tx.as_ref()).await? {
            return Ok(out);
        }

        // The plan named this node for all of it. Run it as the local
        // generation it is, rather than sending ourselves a LayerForward per
        // token — which is the ONLY path that consults the prefix cache, so
        // without this a node that stands its API fast path aside to let the
        // scheduler consider the swarm re-prefills every prompt for ever
        // (report #018).
        let outcome = self.try_local_generate_fastpath(token_tx.clone()).await;
        if let Some(out) = self.keeping_the_partial(outcome, token_tx.as_ref()).await? {
            return Ok(out);
        }

        // Remote-generate fast path for single-segment distributed: bypass
        // the per-token coordinator/remote round trip entirely. Remote
        // worker runs the full decode loop and streams tokens back. Falls
        // through on non-eligibility (multi-segment, TP, vision, LoRA,
        // encrypted pipeline).
        let outcome = self.try_remote_generate_fastpath(token_tx.clone()).await;
        if let Some(out) = self.keeping_the_partial(outcome, token_tx.as_ref()).await? {
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
        // Set when the decode was ended by a failure rather than by the model.
        // The reply built below is still handed to `note_salvaged_reply`, but
        // this function still returns the `Err` — see `may_salvage`.
        let mut interrupted_by: Option<SwarmError> = None;
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
        // For streaming: accumulate decoded text to avoid redundant final decode
        let mut streamed_text = if is_streaming {
            Some(String::new())
        } else {
            None
        };
        // Every stop sequence that ends this reply — the caller's own as well
        // as the template's. This used to derive the template half here and
        // never read `sampling_params.stop` at all, so a caller's `stop` was
        // ignored on every distributed request; `reply_stops` is the one answer
        // and the prompt build above has already warmed it, so this costs no
        // second header parse.
        let stop_strings = self.reply_stops().await.to_vec();
        // Accumulate decoded text for stop-string matching (both streaming and non-streaming)
        let mut accumulated_text = String::new();
        // Bytes of a character the tokens so far have not finished — see
        // `CachedDecoder::decode_tokens_streaming`. Both the streamed text AND
        // the final reply are built from per-token pieces here, so decoding
        // each token alone put U+FFFD in both for every split character.
        let mut utf8_carry: Vec<u8> = Vec::new();

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
        // `encrypted_pipeline_for` is the single answer. Deriving it here from
        // the per-model map plus the global flag implemented two thirds of the
        // precedence rule and missed the third — `encrypted_pipeline_auto`, on
        // by default, which switches privacy on wherever this node holds both
        // ends of the model. This path is defence in depth for exactly that
        // case, and it was defending less than the scheduler does.
        let encrypted_for_model = self.shared_state.encrypted_pipeline_for(model_id);
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

        // The node holding segment 0 has to be told that what it is receiving
        // is already embedded, or it takes `String::from_utf8_lossy` to a float
        // tensor and tokenises the bytes as a prompt — silent nonsense, which
        // is what every build before 2026-09-21 did, because the flag only ever
        // travelled inside the tensor-parallel trailer.
        //
        // **Refuse rather than fall back.** Sending raw token ids instead would
        // work perfectly and quietly break the one promise this setting makes.
        // A local segment 0 needs nothing: the forward never reaches a wire.
        if local_embedder.is_some() {
            let first = &self.assignment.segments[0];
            let first_is_remote = first.node_id != *self.shared_state.identity.node_id();
            if first_is_remote
                && !self.shared_state.peer_advertises_feature(
                    &first.node_id,
                    swarmllm_types::node::features::FORWARD_PRE_EMBEDDED,
                )
            {
                tracing::warn!(
                    request_id = %request_id,
                    peer = %first.node_id,
                    model = %model_id,
                    "Refusing prompt privacy: the node holding the first segment \
                     cannot be told the prompt is already embedded"
                );
                return Err(SwarmError::PromptPrivacyUnavailable {
                    model_id: model_id.0.clone(),
                });
            }
        }

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
                                Some(d) => d.decode_tokens_streaming(&[tid], &mut utf8_carry),
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
                        match reason {
                            NetworkFinishReason::Stop => finish_reason = "stop".to_string(),
                            NetworkFinishReason::MaxTokens => finish_reason = "length".to_string(),
                            NetworkFinishReason::Error(e) => {
                                // Same recovery as the remote-generate sibling:
                                // the class does not survive the wire, and
                                // without it the caller is told this server
                                // broke and the peer is charged for it.
                                let err = crate::error::reclassify_flattened_error(&e)
                                    .unwrap_or(SwarmError::Inference(e));
                                // Reaching the window is a finish, not a failure.
                                let Some(err) =
                                    length_finish_or_error(err, !generated_tokens.is_empty())
                                else {
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
                                    break;
                                };
                                if may_salvage(is_streaming, &generated_tokens) {
                                    interrupted_by = Some(err);
                                    finish_reason =
                                        crate::inference::FINISH_REASON_INTERRUPTED.to_string();
                                    break;
                                }
                                return Err(err);
                            }
                        }
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
                    // Reaching the window is a finish, not a failure — checked
                    // before the log line below, which would otherwise record a
                    // completed reply as a pipeline failure.
                    let Some(e) = length_finish_or_error(e, !generated_tokens.is_empty()) else {
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
                        break;
                    };
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
                    if may_salvage(is_streaming, &generated_tokens) {
                        interrupted_by = Some(e);
                        finish_reason = crate::inference::FINISH_REASON_INTERRUPTED.to_string();
                        break;
                    }
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

        // A reply that ended part-way through a character (cut by `max_tokens`)
        // renders that fragment as a whole-reply decode would: U+FFFD. After a
        // stop sequence the text was truncated before it, so there is nothing
        // to add.
        let tail = crate::inference::tokenizer::flush_utf8_carry(&mut utf8_carry);
        if !tail.is_empty() && !hit_stop_string_outer {
            accumulated_text.push_str(&tail);
            if let Some(ref mut st) = streamed_text {
                st.push_str(&tail);
            }
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
        let output = InferenceOutput {
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
        };

        // A decode ended by a failure still returns that failure. Everything
        // that reasons about failures — the log line, the peer penalty, the
        // trust update, the error broadcast in `execute_request` — therefore
        // sees exactly what it saw before. All that is added is a copy of the
        // work already done, which the router hands to the caller only once the
        // attempt and its retry are definitively over.
        if let Some(err) = interrupted_by {
            tracing::info!(
                request_id = %request_id,
                completion_tokens = output.completion_tokens,
                error = %err,
                "DIAG: keeping the partial reply of a request that failed part-way"
            );
            self.shared_state.note_salvaged_reply(request_id, output);
            return Err(err);
        }

        Ok(output)
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
        // The reply's history rides with a forward only when the sampler at the
        // far end will READ it (`sampling::sampler_reads_history`). Decided once,
        // here, because every send below — the ordinary forward, all three
        // failover arms, the local segment — takes it from this binding. The
        // caller always passes the whole completion so far, so without this the
        // default request (no penalties) shipped every token id generated so
        // far on every decode step: the `0x08` trailer plus the worker's JSON,
        // ~8 KB a forward by token 2000, bigger than the hidden state itself.
        let generated_ids: &[u32] =
            if crate::inference::sampling::sampler_reads_history(&self.request.sampling_params) {
                generated_ids
            } else {
                &[]
            };
        let mut activations = initial_activations;
        // Read LIVE, never cached across the loop.
        //
        // A failover can now replace one segment with SEVERAL — when no single
        // stand-in holds the failed range but a few cover it between them
        // (`docs/FUTURE_WORK.md` #17) — and it splices them into the assignment
        // so the replacement survives into every later decode step. A count
        // taken before the loop would then be stale for the rest of THIS
        // forward, in the two places it decides something:
        //
        //   * the loop bound, so the spliced-in tail would never run;
        //   * `is_last`, which decides WHICH SEGMENT SAMPLES — off by one, the
        //     wrong segment samples and the reply is quietly not the model's.
        //
        // Re-reading costs a `Vec::len` per segment per token and removes the
        // whole class. Later tokens were always consistent, because this
        // function re-enters per forward; only the forward that failed over
        // could see the stale value.
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
        while idx < self.assignment.segments.len() {
            if idx < chained_through {
                // A chained run carried this segment, so its input never passed
                // through here and the history we hold for it is no longer the
                // whole story. Say so rather than letting a later replay be
                // assembled from a hole — see `retained_activations`.
                self.shared_state
                    .retained_activations
                    .mark_unrestorable(request_id, self.assignment.segments[idx].layer_range);
                idx += 1;
                continue;
            }
            let is_last = idx == self.assignment.segments.len() - 1;
            let segment = &self.assignment.segments[idx];

            // What we are about to send this segment, kept so a stand-in can be
            // replayed it and take the segment over mid-reply. Retained ONLY
            // where a standby actually covers the range: a segment nothing can
            // take over gains nothing from being restorable, and retaining it
            // would spend the budget that protects the segments that can.
            let has_standby = self
                .assignment
                .standbys
                .iter()
                .any(|s| crate::inference::scheduler::standby_covers(s, segment.layer_range));
            // Segment 0's input is TOKEN IDS unless the caller pre-embedded
            // them, and `[1, seq]` ids are indistinguishable from a flat
            // `[seq, hidden]` state by shape alone — so its span cannot be read
            // and its history cannot be proved contiguous. Excluded rather than
            // guessed. (The wrong guess would self-correct into "unrestorable"
            // at the next step, but relying on that is relying on an accident.)
            let input_is_hidden_state = idx > 0 || pre_embedded;
            self.shared_state.retained_activations.record(
                request_id,
                segment.layer_range,
                index_pos as u32,
                &activations,
                has_standby && input_is_hidden_state,
            );

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
                    // A tensor-parallel segment is driven across a group rather
                    // than by the single forward we just recorded, so a replay
                    // of that record would not rebuild what the group holds.
                    self.shared_state
                        .retained_activations
                        .mark_unrestorable(request_id, segment.layer_range);
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
                        locally_constructed: false,
                        refusal: None,
                        answers_index_pos: None,
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
                        num_segments = self.assignment.segments.len(),
                        pipeline_ms = pipeline_start.elapsed().as_millis() as u64,
                        "DIAG: forward_through_segments completed (last segment local)"
                    );
                    return Ok(result);
                }
                // Use hidden-state activations for the next segment
                activations = result.activations;
            } else {
                // A prompt this peer has already told us it would refuse is
                // not sent to it: uploading the hidden states and waiting for
                // its cold load buys nothing but the refusal. Decided on what
                // the peer ADVERTISES (`peer_served_context`) against the
                // positions the input carries, so a peer that says nothing, or
                // a first segment handed prompt text, is sent to as before.
                if let Some((positions, limit)) = self.advertised_context_refusal(
                    segment,
                    sequence_num,
                    index_pos,
                    idx,
                    pre_embedded,
                    &activations,
                ) {
                    tracing::info!(
                        request_id = %request_id,
                        segment = idx,
                        node = %segment.node_id,
                        tokens = positions,
                        peer_limit = limit,
                        "DIAG: not sending the prompt to a peer that serves a shorter \
                         conversation than this one — trying a standby"
                    );
                    self.shared_state
                        .blacklist_holder_for_request(request_id, &segment.node_id);
                    let refusal = crate::error::longer_than_served(positions, limit).to_string();
                    let failover_result = self
                        .failover_segment(
                            idx,
                            request_id,
                            FailoverInput {
                                sequence_num,
                                index_pos,
                                activations: &activations,
                                pre_embedded,
                                generated_ids,
                                is_last,
                                precomputed_vision: precomputed_vision.as_deref(),
                                original_failure: &refusal,
                            },
                        )
                        .await?;
                    match failover_result {
                        Takeover::Finished(result) => return Ok(result),
                        Takeover::Continue(next) => activations = next,
                    }
                    idx += 1;
                    continue;
                }
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
                let needs_generated_ids = crate::inference::sampling::sampler_reads_history(
                    &self.request.sampling_params,
                );
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
                let run_is_last = idx + chain.len() == self.assignment.segments.len() - 1;

                let vision_for_wire = if idx == 0 && sequence_num == 0 {
                    precomputed_vision.clone()
                } else {
                    None
                };
                // A closure so the SAME forward can be built again if the peer
                // refuses it unopened (`ResendOnRefusal`), from what this loop
                // already holds; called once on every other path.
                let rebuild_forward = || LayerForward {
                    request_id,
                    sequence_num,
                    index_pos: index_pos as u32,
                    activations: activations.clone(),
                    format: TensorFormat::FP32,
                    model_id: segment.shard_id.model_id.clone(),
                    layer_range: segment.layer_range,
                    vision_embeddings: vision_for_wire.clone(),
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
                    // segments, and it is already empty when the sampler will
                    // not read it (gated at the top of this function).
                    //
                    // ⚠ **And only to a peer that can read them.** These ride in
                    // the `0x08` trailer, which a node predating it does not
                    // parse — and it rebuilds the seal's AAD from the trailers
                    // it DID parse, so sending one blind makes every encrypted
                    // forward to that peer fail to open. An older peer keeps the
                    // behaviour it has always had: no penalties applied, which
                    // is what this field failing to reach the wire at all meant
                    // for everyone until 2026-09-21.
                    generated_ids: if is_last
                        && !generated_ids.is_empty()
                        && self.shared_state.peer_advertises_feature(
                            &segment.node_id,
                            swarmllm_types::node::features::FORWARD_GENERATED_IDS,
                        ) {
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
                    // The caller's temperature, top-p, top-k and penalties, for the
                    // segment that SAMPLES — or for a chain's head, which hands
                    // them down to the tail that does (`plan_chain` only
                    // chains through peers that can). ⚠ Only to a peer that
                    // reads the `0x0A` trailer: an older one rebuilds the
                    // seal's AAD from the trailers it parsed. Without it a
                    // remote last segment sampled at 0.7 whatever was asked
                    // (FUTURE_WORK #106).
                    sampling: if run_is_last
                        && self.shared_state.peer_advertises_feature(
                            &segment.node_id,
                            swarmllm_types::node::features::FORWARD_SAMPLING,
                        ) {
                        Some(self.request.sampling_params.clone())
                    } else {
                        None
                    },
                };
                let forward = rebuild_forward();

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
                        // Every hop of a chain forwards this same position, so
                        // the tail's answer names it too.
                        expects_index_pos: Some(forward.index_pos),
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
                    &self.shared_state,
                    rx,
                    request_id,
                    idx,
                    &segment.node_id,
                    num_layers,
                    activations.len(),
                    budget,
                    self.request.cancel.as_ref(),
                    if chain.is_empty() {
                        super::local::ResendOnRefusal::SameForward {
                            network_tx: &self.network_tx,
                            target_peer_bytes: &target_peer_bytes,
                            rebuild: &rebuild_forward,
                        }
                    } else {
                        // Any hop of a chain may be the one that refused, and
                        // the hops before it have already run this step. The
                        // chained branch below re-runs the segment unchained,
                        // with a KV rewind, and THAT send may resend.
                        super::local::ResendOnRefusal::Never(
                            "chained run — re-run unchained instead",
                        )
                    },
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
                            let declared = self
                                .shared_state
                                .model_declared_context(&segment.shard_id.model_id);
                            if let Some(err) = super::every_holder_would_refuse(err_msg, declared) {
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
                            // A prompt longer than THIS peer serves: its own
                            // limit, not the model's, so a standby may serve
                            // more. Barred for the request either way — it
                            // will refuse this length again on any re-plan.
                            if let Some(refusal) = crate::error::served_context_refusal(err_msg) {
                                tracing::info!(
                                    request_id = %request_id,
                                    segment = idx,
                                    node = %segment.node_id,
                                    tokens = refusal.tokens,
                                    peer_limit = refusal.limit,
                                    model_limit = ?declared,
                                    "DIAG: remote segment serves a shorter conversation than this \
                                     one — trying a standby that may serve more"
                                );
                                self.shared_state
                                    .blacklist_holder_for_request(request_id, &segment.node_id);
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
                                    FailoverInput {
                                        sequence_num,
                                        index_pos,
                                        activations: &activations,
                                        pre_embedded,
                                        generated_ids,
                                        is_last,
                                        precomputed_vision: precomputed_vision.as_deref(),
                                        original_failure: err_msg,
                                    },
                                )
                                .await?;
                            // Whether this finished the pipeline is the
                            // failover's answer, not `run_is_last`'s: a
                            // composite takeover of the last segment leaves
                            // parts behind this one still to run.
                            match failover_result {
                                Takeover::Finished(result) => return Ok(result),
                                Takeover::Continue(next) => activations = next,
                            }
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
                            // Per (model, segment, holder), for the peer
                            // performance table.
                            self.shared_state.record_segment_latency(
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
                                    num_segments = self.assignment.segments.len(),
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
                            // By declared SHAPE, never byte length: a peer on
                            // the other `activation_compression` setting answers
                            // the same tensor as f32 where we sent Q8_0 (#98).
                            if !is_embedding_expansion
                                && !crate::inference::tensor_util::activation_shape_matches(
                                    &activations,
                                    &result.activations,
                                )
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
                                        FailoverInput {
                                            sequence_num,
                                            index_pos,
                                            activations: &activations,
                                            pre_embedded,
                                            generated_ids,
                                            is_last,
                                            precomputed_vision: precomputed_vision.as_deref(),
                                            original_failure:
                                                "remote segment returned the wrong activation shape",
                                        },
                                    )
                                    .await?;
                                // The standby covered THIS segment only, so
                                // the rest of the run still has to be done. Do
                                // not commit the skip, and let the failover say
                                // whether it finished — see `Takeover`.
                                match failover_result {
                                    Takeover::Finished(result) => return Ok(result),
                                    Takeover::Continue(next) => activations = next,
                                }
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
                        // Not a failure of the peer: the client left. Tell the
                        // peer to stop and end the request; failing over would
                        // send the same prompt to another machine for nobody.
                        if crate::inference::cancel::is_request_abandoned(&e) {
                            tracing::info!(
                                request_id = %request_id,
                                segment = idx,
                                node = %segment.node_id,
                                seq = sequence_num,
                                segment_ms = segment_start.elapsed().as_millis() as u64,
                                "DIAG: request cancelled while a remote segment was computing — \
                                 telling the peer to stop, not failing over"
                            );
                            self.cancel_segment_on(&segment.node_id, request_id, idx)
                                .await;
                            return Err(e);
                        }
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
                                FailoverInput {
                                    sequence_num,
                                    index_pos,
                                    activations: &activations,
                                    pre_embedded,
                                    generated_ids,
                                    is_last,
                                    precomputed_vision: precomputed_vision.as_deref(),
                                    original_failure: &e.to_string(),
                                },
                            )
                            .await?;
                        match failover_result {
                            Takeover::Finished(result) => return Ok(result),
                            Takeover::Continue(next) => activations = next,
                        }
                    }
                }
            }
            idx += 1;
        }

        // OURS: the decode loop ran to its end without producing a result and
        // without raising a reason for it. Nothing external can put us here, so
        // it is `Internal` rather than `PipelineError` — whose hint would have
        // sent the reader after a missing model part (`docs/FUTURE_WORK.md` #86).
        Err(SwarmError::Internal(
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
        // finds nothing to abort (the normal case today), so at default
        // verbosity there is otherwise NO
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
    /// Record who serves the failed segment from now on, so later tokens go
    /// straight there instead of failing over again every step.
    ///
    /// With one stand-in this rewrites the segment in place, exactly as it
    /// always did. With several — a cover assembled from nodes that hold a
    /// piece each — the one segment becomes N, spliced in at the same index and
    /// in layer order.
    ///
    /// **Applied only after the first part has answered**, so a cover that
    /// cannot be reached leaves the assignment untouched and the caller is free
    /// to try the next one. Splicing first and unwinding on failure would leave
    /// a half-installed chain if the unwind were ever missed.
    ///
    /// The caller's loop re-reads `segments.len()` every iteration, so it walks
    /// into the spliced parts and recomputes `is_last` against the new length.
    /// That is load-bearing: `is_last` decides which segment samples. But the
    /// loop only gets there if the caller does not RETURN first — which is why
    /// `failover_segment` says whether it finished (`Takeover`) rather than
    /// leaving that to an `is_last` computed before this splice (gotcha #706).
    fn install_takeover(
        assignment: &mut crate::types::PipelineAssignment,
        failed_idx: usize,
        cover: &[crate::types::PipelineSegment],
        request_id: uuid::Uuid,
    ) {
        if cover.is_empty() {
            return;
        }
        if cover.len() > 1 {
            tracing::info!(
                request_id = %request_id,
                segment = failed_idx,
                parts = cover.len(),
                nodes = ?cover.iter().map(|p| format!("{}[{}-{}]", p.node_id, p.layer_range.0, p.layer_range.1)).collect::<Vec<_>>(),
                "DIAG: segment taken over by several nodes covering it between them"
            );
        }
        assignment
            .segments
            .splice(failed_idx..=failed_idx, cover.iter().cloned());
    }

    /// How far into the conversation a forward's input reaches — `index_pos`
    /// plus the positions it carries — when that can be read off the input.
    /// Hidden states declare their shape; a first segment handed prompt TEXT
    /// does not, and answers `None` rather than a guess.
    fn conversation_positions(
        input: &[u8],
        index_pos: usize,
        input_is_hidden_state: bool,
    ) -> Option<usize> {
        if !input_is_hidden_state {
            return None;
        }
        crate::inference::tensor_util::activation_positions(input)
            .map(|p| index_pos.saturating_add(p as usize))
    }

    /// `(positions, peer_limit)` when `segment`'s node has ADVERTISED that it
    /// serves a shorter conversation than this prompt pass carries, i.e. it
    /// would refuse it. `None` whenever either number is unknown — a peer on an
    /// older build, or prompt text — which sends the forward as before.
    fn advertised_context_refusal(
        &self,
        segment: &crate::types::PipelineSegment,
        sequence_num: u32,
        index_pos: usize,
        idx: usize,
        pre_embedded: bool,
        activations: &[u8],
    ) -> Option<(usize, usize)> {
        if sequence_num != 0 {
            return None;
        }
        let positions =
            Self::conversation_positions(activations, index_pos, idx > 0 || pre_embedded)?;
        let limit = self
            .shared_state
            .peer_served_context(&segment.node_id, &segment.shard_id.model_id)?;
        (positions > limit).then_some((positions, limit))
    }

    async fn failover_segment(
        &mut self,
        failed_idx: usize,
        request_id: uuid::Uuid,
        input: FailoverInput<'_>,
    ) -> Result<Takeover, SwarmError> {
        let FailoverInput {
            sequence_num,
            index_pos,
            activations,
            pre_embedded,
            generated_ids,
            is_last,
            original_failure,
            precomputed_vision,
        } = input;
        let failed_segment = self.assignment.segments[failed_idx].clone();
        // Everyone this segment has been tried on for this request.
        let mut tried: Vec<crate::types::NodeId> = vec![failed_segment.node_id.clone()];
        // The node abandoned on the previous round — the failed holder first,
        // then each standby that failed in turn.
        let mut abandoned = failed_segment.node_id.clone();
        let mut last_failure: Option<String> = Some(original_failure.to_string());

        // A stand-in cannot continue a reply that is already under way.
        //
        // The forward below carries the CURRENT step and nothing else, and the
        // KV cache is keyed by `(layer range, request id)` — so a machine that
        // has not served this segment for this request holds nothing, and no
        // path rebuilds it. `split::executor` then takes `kv_offset` from the
        // cache rather than from `index_pos`, and no check disagrees, so the
        // replacement answers from the current token alone and the reply
        // carries on regardless.
        //
        // Measured (`examples/failover_kv_probe.rs`, llama-3.2-3b): replacing 4
        // of 28 layers takes the probability of the token the healthy machine
        // would have chosen from 0.997 to 0.119; half the model takes it to
        // 0.005. Both API surfaces sample by default, so that diverges at once.
        // Nothing errors and nothing warns — the reply just stops being the
        // model's.
        //
        // **There is no safe early window**: after ONE decode step it is worse
        // (0.0000), because what is missing is the PROMPT, which the prompt
        // pass wrote and the stand-in never saw. So the test is the work kind,
        // not how far in we are.
        //
        // Ending here is not a lost reply. `should_retry_after` retries this
        // variant when a remote segment was involved, and a retry re-runs from
        // the prompt on a fresh route — a correct whole answer, which beats a
        // long one that is quietly wrong. Where the reply has already been
        // streamed the retry is suppressed and the reader keeps what they were
        // sent; where it has not, `note_salvaged_reply` hands back everything
        // generated before the failure, marked unfinished.
        //
        // See `docs/FUTURE_WORK.md` § "A failover after the prompt pass
        // silently loses the failed segment's KV context".
        // The history to replay onto whatever takes this segment over, when
        // there is a provably complete one. On the prompt pass there is nothing
        // to replay — the forward already carries every position — so this is
        // asked only where the reply is already under way.
        //
        // `restorable_history` answers `None` unless it holds positions
        // `0..index_pos` CONTIGUOUSLY, so a partial history refuses here rather
        // than rebuilding a cache that is plausible and wrong. That is the same
        // failure this whole path exists to prevent, and it would be invisible
        // in exactly the same way.
        let replay_history = if failover_can_restore_state(sequence_num) {
            None
        } else {
            self.shared_state.retained_activations.restorable_history(
                request_id,
                failed_segment.layer_range,
                index_pos as u32,
            )
        };
        if let Some(ref history) = replay_history {
            tracing::info!(
                request_id = %request_id,
                segment = failed_idx,
                sequence_num,
                index_pos,
                replay_steps = history.len(),
                replay_bytes = history.iter().map(|s| s.len()).sum::<usize>(),
                "DIAG: failing over mid-reply by replaying this segment's retained inputs"
            );
        }
        if !failover_can_restore_state(sequence_num) && replay_history.is_none() {
            self.cancel_segment_on(&abandoned, request_id, failed_idx)
                .await;
            tracing::warn!(
                request_id = %request_id,
                failed_segment = failed_idx,
                failed_node = %failed_segment.node_id,
                failed_layer_range = ?failed_segment.layer_range,
                sequence_num,
                index_pos,
                last_failure = ?last_failure,
                standbys_covering_this_segment = self
                    .assignment
                    .standbys
                    .iter()
                    .filter(|s| crate::inference::scheduler::standby_covers(
                        s,
                        failed_segment.layer_range
                    ))
                    .count(),
                "DIAG: not failing over mid-reply — a stand-in holds none of this \
                 segment's conversation state and would answer from the current \
                 token alone"
            );
            // Bar the machine that just dropped, for this request only. The
            // retry this error invites re-plans, and without this it re-learns
            // the same holder and reproduces the same failure — the pairing
            // `is_transient_remote_failure`'s doc calls "what makes the retry
            // actually work". `tried` holds only the failed node here, since
            // this returns before any standby is attempted.
            for node in std::iter::once(&failed_segment.node_id).chain(tried.iter()) {
                self.shared_state
                    .blacklist_holder_for_request(request_id, node);
            }
            return Err(SwarmError::SegmentFailoverExhausted(cannot_resume_message(
                failed_idx,
                sequence_num,
                last_failure.as_deref(),
            )));
        }

        // What the stand-in is actually sent, and from which position. With a
        // replay it is every position of the segment's history plus this step,
        // starting at 0 — which is what a fresh cache makes it, so no receiver
        // needs to know anything new. Without one it is the current step alone,
        // exactly as before.
        //
        // Assembled once rather than per standby attempt: the history does not
        // change between attempts, and decoding and concatenating it is the one
        // genuinely costly part of this path.
        let (replay_payload, replay_index_pos) = match replay_history {
            Some(ref history) => match assemble_replay(history, activations) {
                Ok(bytes) => (Some(bytes), 0u32),
                Err(e) => {
                    // The history was there and could not be assembled. Treat
                    // it as no history at all rather than sending a partial
                    // one: this is the branch where being wrong is silent.
                    tracing::warn!(
                        request_id = %request_id,
                        segment = failed_idx,
                        error = %e,
                        "Could not assemble the retained history for replay —                          ending the reply rather than moving it without one"
                    );
                    self.cancel_segment_on(&abandoned, request_id, failed_idx)
                        .await;
                    for node in std::iter::once(&failed_segment.node_id).chain(tried.iter()) {
                        self.shared_state
                            .blacklist_holder_for_request(request_id, node);
                    }
                    return Err(SwarmError::SegmentFailoverExhausted(cannot_resume_message(
                        failed_idx,
                        sequence_num,
                        last_failure.as_deref(),
                    )));
                }
            },
            None => (None, index_pos as u32),
        };
        let send_activations: &[u8] = replay_payload.as_deref().unwrap_or(activations);

        // The widest refusal for LENGTH met while placing this segment — the
        // original failure's, a stand-in's, or one a stand-in advertised — so
        // that running out of stand-ins ends on the caller's own 400 naming
        // the limit, never on "too few machines hold this model", the advice
        // `every_holder_would_refuse` exists to keep away from a length.
        let mut context_refusal = crate::error::served_context_refusal(original_failure);
        // Stand-ins that have ADVERTISED a shorter context than this input are
        // not asked, as the primary is not sent to (`advertised_context_refusal`).
        let positions = Self::conversation_positions(
            send_activations,
            replay_index_pos as usize,
            failed_idx > 0 || pre_embedded,
        );
        if let Some(positions) = positions {
            for standby in &self.assignment.standbys {
                if tried.contains(&standby.node_id) {
                    continue;
                }
                let Some(limit) = self
                    .shared_state
                    .peer_served_context(&standby.node_id, &standby.shard_id.model_id)
                else {
                    continue;
                };
                if positions > limit {
                    tracing::info!(
                        request_id = %request_id,
                        segment = failed_idx,
                        standby = %standby.node_id,
                        tokens = positions,
                        standby_limit = limit,
                        "DIAG: skipping a standby that serves a shorter conversation than this one"
                    );
                    tried.push(standby.node_id.clone());
                    context_refusal = widest_context_refusal(
                        context_refusal,
                        crate::error::ServedContextRefusal {
                            tokens: positions,
                            limit,
                        },
                    );
                }
            }
        }
        let declared = self
            .shared_state
            .model_declared_context(&failed_segment.shard_id.model_id);

        loop {
            self.cancel_segment_on(&abandoned, request_id, failed_idx)
                .await;

            // What can take this segment's layer range over — one stand-in that
            // holds all of it, or several that cover it between them.
            //
            // **A composite is only offered where no replay is needed**, i.e.
            // the prompt pass. Mid-reply a stand-in must be given the segment's
            // retained input history, and that history exists only for the
            // range the COORDINATOR sent to. The second part of a composite is
            // fed by the first part's OUTPUT, which never passed through here
            // and was never retained — so there is nothing to replay onto it,
            // and a cache rebuilt without it is plausible and wrong, which is
            // the exact failure this path refuses elsewhere. Not a limitation to
            // be lifted casually: it needs a retention scheme that does not
            // exist (`docs/FUTURE_WORK.md` #17).
            let cover = if failover_can_restore_state(sequence_num) {
                crate::inference::scheduler::standby_cover_for(
                    &self.assignment.standbys,
                    failed_segment.layer_range,
                    &tried,
                )
            } else {
                self.assignment
                    .standbys
                    .iter()
                    .find(|s| {
                        crate::inference::scheduler::standby_covers(s, failed_segment.layer_range)
                            && !tried.contains(&s.node_id)
                    })
                    .map(|s| vec![s])
            };
            let cover: Option<Vec<crate::types::PipelineSegment>> =
                cover.map(|parts| parts.into_iter().cloned().collect());

            // The first part is what this call forwards to; any parts after it
            // are spliced in behind and run by the loop that called us, which
            // recomputes `is_last` from the live segment count.
            let composite = cover.as_ref().is_some_and(|c| c.len() > 1);
            // Only the FINAL part of a cover ends the pipeline, so a part with
            // others behind it must not be told it samples — `generated_ids`
            // rides on that flag, and the wrong segment sampling is a reply that
            // is quietly not the model's.
            let is_last = is_last && !composite;
            let backup = cover.as_ref().map(|c| c[0].clone());

            let Some(backup) = backup else {
                tracing::error!(
                    request_id = %request_id,
                    failed_segment = failed_idx,
                    failed_node = %failed_segment.node_id,
                    failed_layer_range = ?failed_segment.layer_range,
                    tried = ?tried.iter().map(|n| n.to_string()).collect::<Vec<_>>(),
                    last_failure = ?last_failure,
                    // Both counts, because only the second one is about THIS
                    // segment: standbys are chosen per segment, so a plan can
                    // carry one and still have none for the range that failed.
                    // Reporting only the total made this line read as a
                    // contradiction of the plan's own `standbys=N` (gotcha #451).
                    total_standbys = self.assignment.standbys.len(),
                    standbys_covering_this_segment = self
                        .assignment
                        .standbys
                        .iter()
                        .filter(|s| crate::inference::scheduler::standby_covers(
                            s,
                            failed_segment.layer_range
                        ))
                        .count(),
                    standby_nodes = ?self.assignment.standbys.iter().map(|s| format!("{}[{:?}]", s.node_id, s.layer_range)).collect::<Vec<_>>(),
                    "DIAG: NO standby available for failed segment — pipeline will fail"
                );
                // Bar every machine that just failed this segment from the
                // router's retry. The retry re-runs the whole scheduler, and
                // without this it re-learns the same holders and produces the
                // identical plan — observed live, a peer that answered
                // `CUDA_ERROR_OUT_OF_MEMORY` was handed the same 34 layers
                // again on the very next attempt (gotcha #454). This is the
                // same pairing the missing-shard path already makes, and for
                // the same reason: a retry only helps if the routing input has
                // changed by the time it runs.
                //
                // Scoped to this request id, so nothing here is held against
                // the peer for anyone else's traffic — it may be perfectly
                // healthy and merely full.
                for node in std::iter::once(&failed_segment.node_id).chain(tried.iter()) {
                    self.shared_state
                        .blacklist_holder_for_request(request_id, node);
                }
                // Nobody left could take it, and at least one machine said the
                // conversation is longer than it serves: that is the caller's
                // answer, as a 400 naming the limit. "Too few machines" would
                // send them to fetch a model that is not what is missing.
                if let Some(refusal) = context_refusal {
                    return Err(longer_than_the_swarm_serves(
                        refusal.tokens.max(positions.unwrap_or(0)),
                        refusal.limit,
                        failed_segment.layer_range,
                    ));
                }
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

            // The standby the scheduler most often picks is THIS node, and it
            // was the one case that could not work.
            //
            // `find_standbys` sorts the local node FIRST, deliberately — a node
            // holding every shard is the most reliable fallback there is, and
            // its own comment says so. But this function only ever knew how to
            // DIAL a standby, and the local node has no `peer_id_bytes`: it is
            // the node running this code, not something to reach. So the most
            // preferred standby was a guaranteed second failure, and the whole
            // request died with `No peer_id_bytes for backup node` one line
            // after the scheduler correctly chose the machine that could have
            // answered (gotcha #458).
            //
            // The main loop has run local segments in-process since the
            // beginning; failover simply never learned to. It goes before the
            // waiter registration below because there is nothing to wait for.
            if backup.node_id == *self.shared_state.identity.node_id() {
                match self
                    .process_local_segment(
                        &backup,
                        sequence_num,
                        replay_index_pos as usize,
                        send_activations.to_vec(),
                        // Same conditions the main loop applies: the image and
                        // the pre-embedded flag belong to segment 0 alone, the
                        // generated ids to the segment that samples.
                        if failed_idx == 0 {
                            precomputed_vision
                        } else {
                            None
                        },
                        pre_embedded && failed_idx == 0,
                        if is_last { generated_ids } else { &[] },
                    )
                    .await
                {
                    Ok(result) => {
                        Self::install_takeover(
                            &mut self.assignment,
                            failed_idx,
                            cover.as_deref().unwrap_or_default(),
                            request_id,
                        );
                        return Ok(Takeover::of(result, is_last));
                    }
                    Err(e) => {
                        // Our own failure is a failure of this standby like any
                        // other — try the next one rather than ending the
                        // request (gotcha #435's rule, applied to ourselves).
                        tracing::warn!(
                            request_id = %request_id,
                            segment = failed_idx,
                            error = %e,
                            "Local standby could not run the segment — trying the next standby"
                        );
                        if let Some(r) = crate::error::served_context_refusal(&e.to_string()) {
                            context_refusal = widest_context_refusal(context_refusal, r);
                        }
                        last_failure = Some(e.to_string());
                        tried.push(backup.node_id.clone());
                        abandoned = backup.node_id;
                        continue;
                    }
                }
            }

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
                    // The position the standby is SENT (0 for a replay), not
                    // the step the pipeline is on — that is what it answers.
                    expects_index_pos: Some(replay_index_pos),
                },
            );
            let mut pending_guard = super::PendingLayerResultGuard::new(
                &self.shared_state.pending_layer_results,
                request_id,
            );

            // Send to backup node via directed tensor protocol. Rebuildable
            // for the same reason as the main loop's (`ResendOnRefusal`): a
            // standby that could not open it never ran it either.
            // The history rides in the `0x08` trailer, which a standby predating
            // it cannot parse — and it rebuilds the seal's AAD from the trailers
            // it did parse, so one sent blind makes every encrypted forward to
            // it fail to open. Gated on the STANDBY's features: the ordinary
            // send checks the planned peer's, and this is a different peer.
            let standby_reads_history = is_last
                && !generated_ids.is_empty()
                && self.shared_state.peer_advertises_feature(
                    &backup.node_id,
                    swarmllm_types::node::features::FORWARD_GENERATED_IDS,
                );
            let rebuild_forward = || LayerForward {
                request_id,
                sequence_num,
                // 0 when replaying: the stand-in holds no cache for this
                // request, so a forward carrying every position IS a prompt
                // pass to it, and `split::executor` takes `kv_offset` from that
                // empty cache rather than from this field.
                index_pos: replay_index_pos,
                activations: send_activations.to_vec(),
                format: TensorFormat::FP32,
                model_id: backup.shard_id.model_id.clone(),
                layer_range: backup.layer_range,
                tp_meta: None,
                vision_embeddings: if failed_idx == 0 && sequence_num == 0 {
                    precomputed_vision.map(|v| v.to_vec())
                } else {
                    None
                },
                chain: Vec::new(),
                sender_peer_bytes: None,
                // Unchained failover: the standby answers its sender, which is
                // us. Naming ourselves would put a 0x07 trailer on a frame an
                // older standby does not expect.
                requester_node_id: None,
                pre_embedded: pre_embedded && failed_idx == 0,
                generated_ids: if standby_reads_history {
                    generated_ids.to_vec()
                } else {
                    Vec::new()
                },
                adapter_id: None,
                draft_tokens: Vec::new(),
                spec_logits_requested: false,
                truncate_kv_to: None,
                chunk_meta: None,
                // Same rule as the planned send, asked of the STAND-IN — a
                // different peer with its own features (gotcha #703).
                sampling: if is_last
                    && self.shared_state.peer_advertises_feature(
                        &backup.node_id,
                        swarmllm_types::node::features::FORWARD_SAMPLING,
                    ) {
                    Some(self.request.sampling_params.clone())
                } else {
                    None
                },
            };
            let forward = rebuild_forward();

            let Some(target_peer_bytes) = self.shared_state.resolve_peer_id_bytes(&backup.node_id)
            else {
                // A standby we cannot address is a failure of that standby, not
                // of the request: try the next one. Returning here ended the
                // whole failover on the first unreachable entry, however many
                // good standbys stood behind it.
                tracing::warn!(
                    request_id = %request_id,
                    segment = failed_idx,
                    standby = %backup.node_id,
                    "Standby has no reachable address — trying the next standby"
                );
                last_failure = Some(format!("standby {} is not reachable", backup.node_id));
                tried.push(backup.node_id.clone());
                abandoned = backup.node_id;
                continue;
            };
            if self
                .network_tx
                .send(NetworkCommand::SendTensor {
                    target_peer_bytes: target_peer_bytes.clone(),
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
                &self.shared_state,
                rx,
                request_id,
                failed_idx,
                &backup.node_id,
                num_layers,
                activations.len(),
                budget,
                self.request.cancel.as_ref(),
                super::local::ResendOnRefusal::SameForward {
                    network_tx: &self.network_tx,
                    target_peer_bytes: &target_peer_bytes,
                    rebuild: &rebuild_forward,
                },
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
                if let Some(err) = super::every_holder_would_refuse(err_msg, declared) {
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
                if let Some(r) = crate::error::served_context_refusal(err_msg) {
                    context_refusal = widest_context_refusal(context_refusal, r);
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
            Self::install_takeover(
                &mut self.assignment,
                failed_idx,
                cover.as_deref().unwrap_or_default(),
                request_id,
            );

            // `is_last` here is the one shadowed above: the failed segment was
            // the last AND nothing was spliced in behind this part.
            return Ok(Takeover::of(result, is_last));
        }
    }
}

/// How many characters of the failing machine's own message the exhaustion
/// error carries. A worker's refusal names sizes and context lengths and can
/// run to several hundred characters; the caller needs the reason, not the
/// arithmetic.
const EXHAUSTED_REASON_MAX_CHARS: usize = 200;

/// Two refusals for length folded into one that covers both: the longest
/// conversation refused, and the most any refusing machine serves — the figure
/// the caller would have to get under.
fn widest_context_refusal(
    seen: Option<crate::error::ServedContextRefusal>,
    new: crate::error::ServedContextRefusal,
) -> Option<crate::error::ServedContextRefusal> {
    Some(match seen {
        Some(s) => crate::error::ServedContextRefusal {
            tokens: s.tokens.max(new.tokens),
            limit: s.limit.max(new.limit),
        },
        None => new,
    })
}

/// The caller's answer when every machine that could run `layer_range` serves
/// a shorter conversation than this one.
///
/// A 400, like the local refusal it replaces here — the conversation is too
/// long for what the swarm can run right now, and an agent client reads a 400
/// about length as "compact and retry". But NOT that refusal's advice: a peer's
/// message tells its reader to raise `max_seq_len_override`, which here is
/// another computer's setting (field report, 2026-09-25). This names whose
/// limit it is and the two things the caller can actually do.
fn longer_than_the_swarm_serves(
    tokens: usize,
    limit: usize,
    layer_range: (u32, u32),
) -> SwarmError {
    SwarmError::Validation(longer_than_the_swarm_serves_text(
        tokens,
        limit,
        layer_range,
    ))
}

/// The words of [`longer_than_the_swarm_serves`], shared with the whole-model
/// path (`remote_generate`), whose refusal says the same thing about the same
/// kind of limit and must not say it differently.
pub(super) fn longer_than_the_swarm_serves_text(
    tokens: usize,
    limit: usize,
    layer_range: (u32, u32),
) -> String {
    format!(
        "This conversation is {tokens} tokens, but the computers that could run {span} of \
         this model right now serve at most {limit}. That limit is set on those computers, \
         so changing max_seq_len_override here does not raise it. Shorten the conversation, \
         or download that part of the model so this computer runs it under its own limit.",
        span = crate::error::describe_missing_layers(layer_range.0, layer_range.1),
    )
}

/// The message a request fails with when every standby for `segment` has been
/// tried, carrying the last stated reason.
///
/// "Last failure", not "last standby": with no standby at all — which is every
/// single-peer delegation, by design — the reason carried is the ORIGINAL
/// segment failure, and calling that a standby's words would be a lie about
/// which machine said it.
/// What a failover hands back, and whether it finishes the pipeline.
///
/// Decided in `failover_segment`, where the takeover is installed, because only
/// there is it known whether the failed segment went to ONE stand-in or was
/// spliced into SEVERAL. The callers used to ask their own `is_last` — computed
/// before the splice — so when the failed segment was the last one, a composite
/// takeover returned its FIRST part's hidden states as the pipeline's answer:
/// the parts spliced in behind it never ran the prompt, and the next token
/// failed on a node holding no conversation. Found by the first live run of
/// `docs/FUTURE_WORK.md` #17 (`examples/split_rig.sh failover`, 2026-09-25);
/// the unit tests and the segment-count guard all passed.
enum Takeover {
    /// The stand-in ran the model's last layers and sampled: the answer.
    Finished(LayerResult),
    /// Hidden states for the next segment — after a composite splice, the next
    /// PART of the cover, which the caller's loop runs next.
    Continue(Vec<u8>),
}

impl Takeover {
    fn of(result: LayerResult, finishes_pipeline: bool) -> Self {
        if finishes_pipeline {
            Takeover::Finished(result)
        } else {
            Takeover::Continue(result.activations)
        }
    }
}

/// Everything a failover needs to reproduce the forward the failed segment was
/// given.
///
/// It is a struct because the list kept being trimmed. `failover_segment` was
/// written as a simplified copy of the main send and drifted from it: the
/// forward it built hardcoded `pre_embedded: false`, `generated_ids: []` and
/// `vision_embeddings: None`, so failing over segment 0 under local-embedding
/// privacy handed a standby hidden states labelled as token ids, failing over
/// the last segment dropped the repetition-penalty context, and failing over a
/// vision request dropped the image. None of those crash — they quietly change
/// the answer, which is why none of them was reported.
///
/// Adding a field to the wire forward means adding it here and deciding what a
/// failover should send, rather than defaulting it to nothing by omission.
struct FailoverInput<'a> {
    sequence_num: u32,
    index_pos: usize,
    activations: &'a [u8],
    /// True when `activations` are already embedded hidden states rather than
    /// token ids. Only ever true for segment 0, under local-embedding privacy.
    pre_embedded: bool,
    /// Tokens generated so far, for the repetition penalties the LAST segment
    /// applies when it samples. Empty everywhere else, and empty when the
    /// caller has no penalties configured — `forward_through_segments_inner`
    /// empties it at its top, and this is taken from there.
    generated_ids: &'a [u32],
    /// Whether the failed segment is the one that samples.
    is_last: bool,
    /// Image embeddings, for the first segment of the first forward of a
    /// vision request.
    precomputed_vision: Option<&'a [u8]>,
    /// Why the segment failed in the first place, in the words the failing node
    /// or the transport used. Seeds `last_failure`, so a request with NO
    /// standby — which is every single-peer delegation, by design — still
    /// reports the actual cause instead of only "no standby available".
    ///
    /// Without it, a mid-stream `OutboundFailure: connection lost` reached the
    /// caller as a bare `Segment 1 failed with no standby available`, and an
    /// operator had to correlate two log lines to learn what had happened. It
    /// also cost the retry: `is_transient_remote_failure` matches
    /// "OutboundFailure" in the message text, and the message no longer carried
    /// it.
    original_failure: &'a str,
}

/// Can a stand-in reproduce what the machine it replaces would have computed?
///
/// Only on the prompt pass. There the stand-in is handed the whole prompt and
/// builds its own KV cache, which is the case standbys were designed for and the
/// only one every existing failover test exercises. After it, the cache the
/// failed machine had accumulated is gone and nothing rebuilds it.
///
/// The test is the WORK KIND rather than the elapsed reply, because the missing
/// state is the prompt itself — failing over one token in is measurably worse
/// than twenty-four tokens in, not better (gotcha #508).
/// Concatenate a segment's retained inputs, plus the step being taken over,
/// into ONE forward covering positions `0..=current`.
///
/// This is the whole of the replay protocol: a stand-in holds no cache for this
/// request, so a forward at `index_pos` 0 carrying every position is an
/// ordinary prompt pass to it, and the existing path serves it unchanged. No
/// new message type, no capability bit, nothing an older peer would refuse.
///
/// The last position's output is what the caller wanted from the takeover step,
/// so the result is used exactly as an unreplayed forward's would be.
///
/// Measured (`examples/failover_kv_probe.rs`, llama-3.2-3b): a stand-in given
/// this reaches P = 0.9965 of the intact machine's own next token against its
/// 0.9966, where the same stand-in without it reaches 0.119. The residual is
/// accumulation order — one wide prefill sums differently from a run of
/// single-position decodes — and shows as cosine 0.9997-0.9999 rather than the
/// control's exact 1.000000.
fn assemble_replay(history: &[Vec<u8>], current: &[u8]) -> Result<Vec<u8>, SwarmError> {
    use crate::inference::tensor_util::{bytes_to_tensor, tensor_to_bytes};
    let mut parts: Vec<candle_core::Tensor> = Vec::with_capacity(history.len() + 1);
    for step in history.iter().chain(std::iter::once(&current.to_vec())) {
        parts.push(bytes_to_tensor(step)?);
    }
    // The sequence axis is the second-to-last, matching `activation_positions`.
    let dim = parts
        .first()
        .map(|t| t.dims().len().saturating_sub(2))
        .ok_or_else(|| SwarmError::Internal("replay history is empty".into()))?;
    let joined = candle_core::Tensor::cat(&parts, dim).map_err(SwarmError::internal)?;
    tensor_to_bytes(&joined)
}

pub(super) fn failover_can_restore_state(sequence_num: u32) -> bool {
    matches!(
        super::work_kind_for(sequence_num),
        crate::daemon::state::WorkKind::Prefill
    )
}

/// Why a reply already under way was ended rather than moved to another machine.
///
/// Deliberately NOT `exhausted_message`'s wording: standbys may well have been
/// available here, and saying "none available" would send an operator looking
/// for capacity they already have. Same variant, because the next step is the
/// same one — retry, or hold the model locally so the reply never depends on a
/// remote segment.
pub(super) fn cannot_resume_message(
    segment: usize,
    sequence_num: u32,
    last_failure: Option<&str>,
) -> String {
    let base = format!(
        "Segment {segment} lost its machine {sequence_num} tokens into the reply, and a \
         stand-in cannot continue it — the replacement holds none of the conversation \
         state the failed machine had built"
    );
    match last_failure.map(str::trim).filter(|s| !s.is_empty()) {
        None => base,
        Some(reason) => {
            let shown: String = reason.chars().take(EXHAUSTED_REASON_MAX_CHARS).collect();
            let ellipsis = if shown.len() < reason.len() {
                "…"
            } else {
                ""
            };
            format!("{base} (cause: {shown}{ellipsis})")
        }
    }
}

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
            format!("{base} (last failure: {shown}{ellipsis})")
        }
    }
}

impl PipelineExecutor {
    /// The one place a failure inside an ALTERNATIVE generation path keeps what
    /// that path had already generated.
    ///
    /// `execute_distributed` tries five paths before reaching its own decode
    /// loop, and each of them runs a complete decode of its own. Every one was
    /// called with `?`, so a failure part-way through a reply propagated past
    /// the loop's `may_salvage` arms entirely and the caller got a bare error
    /// with nothing in it — report #028's "total loss" again, on the paths that
    /// grew up beside that fix rather than through it. `grep -c may_salvage`
    /// was 0 in all three speculative files. Observed: ~220 tokens generated
    /// over 40 s, request failed, nothing salvaged (FUTURE_WORK #88).
    ///
    /// **It reads rather than records.** The shared emit helpers fill
    /// `self.partial_reply` as they turn tokens into text, so no path has to
    /// remember anything and a sixth path added to that list inherits this. A
    /// helper nobody is obliged to call will eventually not be called, which is
    /// what this entry was.
    ///
    /// The failure is still returned unchanged: the log line, the peer penalty,
    /// the trust update and the error broadcast all see exactly what they saw
    /// before, and `router::salvaged_reply_if_lost` hands the partial over only
    /// once the attempt AND its retry are definitively over. A complete answer
    /// from a second route beats a truncated one from the first.
    async fn keeping_the_partial(
        &self,
        outcome: Result<Option<InferenceOutput>, SwarmError>,
        token_tx: Option<&StreamingTokenTx>,
    ) -> Result<Option<InferenceOutput>, SwarmError> {
        let Err(err) = outcome else {
            return outcome;
        };
        let is_streaming = token_tx.is_some();
        let request_id = self.request.id;
        let partial = self.partial_reply.taken();

        // A reply that ran into the model's context window FINISHED — and this
        // is the only place all five alternative coordinators can be told so at
        // once. The standard loop below has its own two arms; these five do
        // not, and the DEFAULT distributed path is one of them
        // (`try_ngram_only_distributed` — a node holding nothing takes it for
        // every request), which is why the first version of this fix, written
        // only in the standard loop, left a streamed chat ending on
        // "Validation error: this conversation is 257 tokens" for a 61-token
        // prompt. Same shape as FUTURE_WORK #88, in the same five paths.
        let produced_any = partial.as_ref().is_some_and(|(_, ids, _)| !ids.is_empty());
        let err = match length_finish_or_error(err, produced_any) {
            Some(err) => err,
            None => {
                let (mut content, ids, prompt_tokens) =
                    partial.expect("produced_any is false without a partial");
                crate::inference::finalize_reply_text(&mut content, self.reply_stops().await);
                tracing::info!(
                    request_id = %request_id,
                    completion_tokens = ids.len(),
                    "DIAG: the reply reached the model's context window — finishing for length"
                );
                // A streamed reply MUST get its terminal event: `api::openai::
                // streaming` reads "no finish event arrived" as "this path never
                // streamed" and re-emits the whole content as one delta, so
                // returning Ok without this hands the reader the reply twice
                // (gotcha #414).
                if let Some(tx) = token_tx {
                    let _ = tx
                        .send(StreamingTokenEvent {
                            text: String::new(),
                            finish_reason: Some("length".to_string()),
                            matched_stop_sequence: None,
                        })
                        .await;
                }
                return Ok(Some(InferenceOutput {
                    request_id,
                    content,
                    prompt_tokens,
                    completion_tokens: ids.len() as u32,
                    finish_reason: "length".to_string(),
                    session_id: self.request.session_id.clone(),
                    token_logprobs: vec![],
                    matched_stop_sequence: None,
                    trace: None,
                }));
            }
        };

        let Some((mut content, ids, prompt_tokens)) = partial else {
            return Err(err);
        };
        if !may_salvage(is_streaming, &ids) {
            return Err(err);
        }
        // Finalised like any other reply — a partial is still a reply, and the
        // text accumulated here has been through no scrub at all (gotcha #643).
        crate::inference::finalize_reply_text(&mut content, self.reply_stops().await);
        tracing::info!(
            request_id = %request_id,
            completion_tokens = ids.len(),
            error = %err,
            "DIAG: keeping the partial reply of a request that failed part-way"
        );
        self.shared_state.note_salvaged_reply(
            request_id,
            InferenceOutput {
                request_id,
                content,
                prompt_tokens,
                completion_tokens: ids.len() as u32,
                finish_reason: crate::inference::FINISH_REASON_INTERRUPTED.to_string(),
                session_id: self.request.session_id.clone(),
                token_logprobs: vec![],
                matched_stop_sequence: None,
                trace: None,
            },
        );
        Err(err)
    }
}

/// May a reply that ends in a failure still be handed to the caller?
///
/// Three conditions, and each one is load-bearing.
///
/// **Something was actually generated.** An empty salvage is not a salvage: it
/// would replace an error carrying a class, a hint and a peer with a `200`
/// carrying nothing, which is gotcha #433's lie pointing the other way. With no
/// tokens the error is the better answer and is returned unchanged.
///
/// **The request is not being streamed.** A streamed reply has already been
/// delivered token by token, so the client keeps the text whatever happens next
/// and the honest terminal event is the error it already gets. It also must not
/// become an `Ok`: `api::openai::streaming` treats "no finish event arrived" as
/// "this path never streamed" and re-emits the whole content as one delta, so a
/// salvage there would hand the reader the reply twice (gotcha #414).
/// Non-streaming is exactly the case report #028 names as a total loss.
///
/// **It never pre-empts the retry.** This only decides whether to RECORD the
/// partial reply; `router::salvaged_reply_if_lost` takes it after the retry has
/// run and also failed. A complete answer from a second route beats a truncated
/// one from the first, so salvage is the last resort and never the first.
pub(crate) fn may_salvage(is_streaming: bool, generated: &[u32]) -> bool {
    !is_streaming && !generated.is_empty()
}

/// A reply that ran into the model's context window has FINISHED, not failed.
///
/// Returns `None` when the decode loop should end the reply with
/// `finish_reason: "length"`, and `Some(err)` with the error to report
/// otherwise.
///
/// **One place, because the decode loop has two failure arms** — a peer's
/// `NetworkFinishReason::Error` and a local `Err` — and the arm that is easy to
/// forget is the one whose peer is on the far side of two boundaries that keep
/// no types. Both reach this; a third would inherit it.
///
/// **With nothing produced it stays an error, and wears the old wording.** The
/// window can only be reached at the first decode step if the conversation
/// already filled it, and "this conversation is longer than this model is set
/// to serve" is then exactly true and exactly actionable — it is what the
/// caller saw before this variant existed, so nothing regresses for the case
/// the 400 was always right about. Naming `max_seq_len_override` matters more
/// than the count: an agentic client's prompt is its tool schema, and
/// "send less" is not something its user can do.
pub(crate) fn length_finish_or_error(err: SwarmError, produced_any: bool) -> Option<SwarmError> {
    let SwarmError::ContextWindowReached { used, window } = err else {
        return Some(err);
    };
    if produced_any {
        return None;
    }
    Some(SwarmError::Validation(format!(
        "This conversation is {used} tokens, longer than the {window} this model \
         is currently set to serve. Raise it in Settings → Advanced → \
         max_seq_len_override (the model itself supports more), or send a \
         shorter prompt or a smaller max_tokens."
    )))
}

#[cfg(test)]
mod salvage_tests {
    use super::PipelineExecutor;
    use crate::error::SwarmError;
    use crate::types::{
        ChatMessage, InferenceRequest, ModelId, NetworkCommand, PipelineAssignment,
        PipelineSegment, PriorityTier, Role, SamplingParams, ShardId,
    };

    fn verbatim_decoder(vocab: &[&str]) -> super::super::prompt::CachedDecoder {
        let mut byte_decoder = std::collections::HashMap::new();
        for b in 0u8..=127 {
            byte_decoder.insert(b as char, b);
        }
        super::super::prompt::CachedDecoder {
            vocab: vocab.iter().map(|s| (*s).to_string()).collect(),
            byte_decoder,
            is_sentencepiece: false,
            has_tokenizer: true,
        }
    }

    fn executor(state: std::sync::Arc<crate::daemon::SharedState>) -> PipelineExecutor {
        let request = InferenceRequest {
            id: uuid::Uuid::new_v4(),
            model_id: ModelId("m".into()),
            messages: vec![ChatMessage {
                role: Role::User,
                content: "hello".into(),
                images: vec![],
            }],
            sampling_params: SamplingParams::default(),
            stream: false,
            requester: state.identity.node_id().clone(),
            priority: PriorityTier::Silver,
            created_at: chrono::Utc::now(),
            session_id: None,
            lora_adapter: None,
            tools: None,
            cancel: None,
            route_override: None,
        };
        let (tx, _rx) = tokio::sync::mpsc::channel::<NetworkCommand>(8);
        let assignment = PipelineAssignment {
            request_id: request.id,
            segments: vec![PipelineSegment {
                node_id: state.identity.node_id().clone(),
                shard_id: ShardId {
                    model_id: ModelId("m".into()),
                    index: 0,
                },
                layer_range: (0, 1),
            }],
            standbys: vec![],
            tp_groups: vec![],
            supports_speculative: false,
        };
        PipelineExecutor::new(state, tx, request, assignment)
    }

    /// Emit three tokens the way a speculative coordinator does on a
    /// NON-streamed request — the case `emit_streaming_batch` used to leave
    /// immediately, having recorded nothing.
    async fn generate_something(exec: &PipelineExecutor) {
        let decoder = verbatim_decoder(&["Half ", "an ", "answer"]);
        let eos = std::collections::HashSet::new();
        let mut finish = String::new();
        super::super::emit_first_streaming_token(&exec.partial_reply, &None, &decoder, 0, &eos)
            .await;
        super::super::emit_streaming_batch(
            &exec.partial_reply,
            &None,
            &decoder,
            &[1, 2],
            &eos,
            &mut finish,
        )
        .await;
    }

    /// **FUTURE_WORK #88.** The salvage built for report #028 lived only in
    /// `execute_distributed`'s own decode loop, while the five alternative
    /// paths tried before it were each called with `?` — so ~220 generated
    /// tokens were discarded and the caller got a bare error. The partial is
    /// now accumulated by the shared emit helpers and read at one choke point.
    #[tokio::test]
    async fn a_path_that_fails_part_way_still_hands_back_what_it_generated() {
        let state = crate::inference::pipeline::tests::make_test_state();
        let exec = executor(state.clone());
        let id = exec.request.id;
        generate_something(&exec).await;

        let out = exec
            .keeping_the_partial(
                Err(SwarmError::PeerUnresponsive(
                    "the tail peer went silent".into(),
                )),
                None,
            )
            .await;

        // The failure is still the failure: everything that reasons about one
        // must see exactly what it saw before.
        assert!(matches!(out, Err(SwarmError::PeerUnresponsive(_))));

        let salvaged = state
            .take_salvaged_reply(id)
            .expect("the tokens it generated must be kept");
        assert_eq!(salvaged.content, "Half an answer");
        assert_eq!(salvaged.completion_tokens, 3);
        assert_eq!(
            salvaged.finish_reason,
            crate::inference::FINISH_REASON_INTERRUPTED,
            "a decode a failure ended did not stop naturally"
        );
    }

    /// The default distributed path reaches the window and FINISHES.
    ///
    /// All five alternative coordinators funnel through here, and none of them
    /// asks this for itself — the first version of this fix lived in the
    /// standard decode loop only, and a streamed chat on
    /// `try_ngram_only_distributed` (what a node holding nothing takes for
    /// every request) still ended on "Validation error: this conversation is
    /// 257 tokens" for a 61-token prompt. Measured on a two-node rig.
    #[tokio::test]
    async fn a_reply_that_reached_the_window_comes_back_as_a_finished_reply() {
        let state = crate::inference::pipeline::tests::make_test_state();
        let exec = executor(state.clone());
        let id = exec.request.id;
        generate_something(&exec).await;

        let out = exec
            .keeping_the_partial(
                Err(SwarmError::ContextWindowReached {
                    used: 257,
                    window: 256,
                }),
                None,
            )
            .await
            .expect("reaching the window is not a failure")
            .expect("the reply must come back");

        assert_eq!(out.finish_reason, "length");
        assert_eq!(out.content, "Half an answer");
        assert_eq!(out.completion_tokens, 3);
        assert!(
            state.take_salvaged_reply(id).is_none(),
            "this is a completed reply, not a salvage — recording it as one \
             would let the router hand it over as a last resort instead of the \
             answer it is"
        );
    }

    /// And a STREAMED one gets its terminal event, or the SSE encoder reads the
    /// missing finish as "this path never streamed" and re-emits the whole
    /// reply as one delta (gotcha #414).
    #[tokio::test]
    async fn a_streamed_reply_that_reached_the_window_is_told_it_finished() {
        let state = crate::inference::pipeline::tests::make_test_state();
        let exec = executor(state.clone());
        generate_something(&exec).await;
        let (tx, mut rx) = super::StreamingTokenTx::channel(8);

        let out = exec
            .keeping_the_partial(
                Err(SwarmError::ContextWindowReached {
                    used: 257,
                    window: 256,
                }),
                Some(&tx),
            )
            .await
            .expect("reaching the window is not a failure")
            .expect("the reply must come back");
        assert_eq!(out.finish_reason, "length");

        let event = rx.try_recv().expect("a terminal event must have been sent");
        assert_eq!(event.finish_reason.as_deref(), Some("length"));
        assert!(
            event.text.is_empty(),
            "the terminal event carries no text — the deltas already did"
        );
    }

    /// Null control: nothing generated is still a failure, and it wears the
    /// message the caller used to get. The window can only be hit at the first
    /// decode step if the conversation already filled it, and there "this
    /// conversation is too long" is exactly true.
    #[tokio::test]
    async fn reaching_the_window_with_nothing_generated_is_still_an_error() {
        let state = crate::inference::pipeline::tests::make_test_state();
        let exec = executor(state.clone());
        let out = exec
            .keeping_the_partial(
                Err(SwarmError::ContextWindowReached {
                    used: 257,
                    window: 256,
                }),
                None,
            )
            .await;
        let Err(SwarmError::Validation(ref text)) = out else {
            panic!("expected a validation error, got {out:?}");
        };
        assert!(text.contains("max_seq_len_override"));
    }

    /// Null control 1: a STREAMED request keeps its error untouched. The
    /// client already has the text, and an `Ok` here makes the SSE encoder
    /// re-emit the whole reply as one delta (gotcha #414).
    #[tokio::test]
    async fn a_streamed_request_salvages_nothing() {
        let state = crate::inference::pipeline::tests::make_test_state();
        let exec = executor(state.clone());
        let id = exec.request.id;
        generate_something(&exec).await;
        let (tx, _rx) = super::StreamingTokenTx::channel(8);
        let out = exec
            .keeping_the_partial(Err(SwarmError::Inference("boom".into())), Some(&tx))
            .await;
        assert!(out.is_err());
        assert!(
            state.take_salvaged_reply(id).is_none(),
            "a streamed reply must not be salvaged"
        );
    }

    /// Null control 2: a path that generated nothing keeps its error. An empty
    /// salvage is a failure wearing a 200, and it would replace an error
    /// carrying the class, the hint and the peer attribution.
    #[tokio::test]
    async fn a_failure_with_nothing_generated_salvages_nothing() {
        let state = crate::inference::pipeline::tests::make_test_state();
        let exec = executor(state.clone());
        let id = exec.request.id;
        let out = exec
            .keeping_the_partial(Err(SwarmError::Inference("boom".into())), None)
            .await;
        assert!(out.is_err());
        assert!(state.take_salvaged_reply(id).is_none());
    }

    /// And a SUCCESS is returned untouched, with nothing recorded — the
    /// choke point wraps every alternative path, so it sees far more
    /// successes than failures.
    #[tokio::test]
    async fn a_successful_path_is_passed_straight_through() {
        let state = crate::inference::pipeline::tests::make_test_state();
        let exec = executor(state.clone());
        let id = exec.request.id;
        generate_something(&exec).await;
        let out = exec.keeping_the_partial(Ok(None), None).await;
        assert!(matches!(out, Ok(None)));
        assert!(state.take_salvaged_reply(id).is_none());
    }
}

#[cfg(test)]
mod served_context_tests {
    use super::{longer_than_the_swarm_serves, widest_context_refusal, PipelineExecutor};
    use crate::error::ServedContextRefusal;

    /// The answer when nobody can serve the length is the caller's 400, and it
    /// does NOT repeat the peer's advice to raise a setting on THIS computer —
    /// the thing the field report (2026-09-25) said could not help.
    #[test]
    fn the_swarms_limit_is_named_as_someone_elses_setting() {
        let err = longer_than_the_swarm_serves(8560, 8192, (1, 13));
        let (status, msg, _) = crate::error::classify_error(&err);
        assert_eq!(status, axum::http::StatusCode::BAD_REQUEST);
        assert!(msg.contains("8560") && msg.contains("8192"), "{msg}");
        assert!(msg.contains("layers 1-12"), "names the part: {msg}");
        assert!(
            !msg.contains("Raise it in Settings"),
            "must not tell the caller to change their own limit: {msg}"
        );
        // And it is NOT read back as one node's refusal — it is the final word,
        // and a coordinator further up must not fail over on it.
        assert_eq!(crate::error::served_context_refusal(&err.to_string()), None);
    }

    /// Folding refusals keeps the longest conversation and the most any
    /// machine serves — the figure the caller would have to get under.
    #[test]
    fn refusals_fold_to_the_widest() {
        let a = ServedContextRefusal {
            tokens: 8320,
            limit: 8192,
        };
        let b = ServedContextRefusal {
            tokens: 8560,
            limit: 4096,
        };
        assert_eq!(
            widest_context_refusal(Some(a), b),
            Some(ServedContextRefusal {
                tokens: 8560,
                limit: 8192
            })
        );
        assert_eq!(widest_context_refusal(None, b), Some(b));
    }

    /// Positions are read only off hidden states, and counted from the start
    /// of the conversation; prompt text answers `None`, never a guess.
    #[test]
    fn positions_are_read_off_hidden_states_only() {
        // [1, 7, 4] f32 hidden states, as `tensor_to_bytes` writes them.
        let t = candle_core::Tensor::zeros(
            (1, 7, 4),
            candle_core::DType::F32,
            &candle_core::Device::Cpu,
        )
        .unwrap();
        let bytes = crate::inference::tensor_util::tensor_to_bytes(&t).unwrap();
        assert_eq!(
            PipelineExecutor::conversation_positions(&bytes, 0, true),
            Some(7)
        );
        assert_eq!(
            PipelineExecutor::conversation_positions(&bytes, 100, true),
            Some(107)
        );
        assert_eq!(
            PipelineExecutor::conversation_positions(b"hello world", 0, false),
            None
        );
    }
}

#[cfg(test)]
mod context_window_finish_tests {
    use super::length_finish_or_error;
    use crate::error::SwarmError;

    /// The defect this exists for: 40 seconds of work, a complete reply, and a
    /// 400 blaming the caller's prompt for a length the model chose
    /// (`docs/FUTURE_WORK.md` #85). A reply that reached the window has
    /// finished.
    #[test]
    fn a_reply_that_reached_the_window_has_finished_not_failed() {
        let err = SwarmError::ContextWindowReached {
            used: 260,
            window: 256,
        };
        assert!(
            length_finish_or_error(err, true).is_none(),
            "with tokens produced this must end the reply, not fail it"
        );
    }

    /// With nothing produced the conversation really was too long to start, so
    /// the old message is still the right one — and still a 400.
    #[test]
    fn nothing_produced_keeps_the_message_the_caller_used_to_get() {
        let err = SwarmError::ContextWindowReached {
            used: 257,
            window: 256,
        };
        let reported = length_finish_or_error(err, false).expect("must still be an error");
        let SwarmError::Validation(ref text) = reported else {
            panic!("expected a validation error, got {reported:?}");
        };
        assert!(text.contains("257"), "names the conversation length");
        assert!(text.contains("256"), "names the window");
        assert!(
            text.contains("max_seq_len_override"),
            "names the setting that fixes it — the one action an agentic \
             client's operator can actually take"
        );
        let (status, _, kind) = crate::error::classify_error(&reported);
        assert_eq!(status, axum::http::StatusCode::BAD_REQUEST);
        assert_eq!(kind, "invalid_request_error");
    }

    /// Every other failure passes through untouched — this must not swallow a
    /// peer going silent or a segment failing over.
    #[test]
    fn any_other_failure_is_returned_unchanged() {
        for err in [
            SwarmError::PeerUnresponsive("gone".into()),
            SwarmError::Inference("something else".into()),
            SwarmError::Validation("a prompt too long at prefill".into()),
        ] {
            let name = format!("{err:?}");
            assert!(
                length_finish_or_error(err, true).is_some(),
                "{name} must not be treated as a finished reply"
            );
        }
    }

    /// The class has to survive the worker IPC hop AND the network hop, both of
    /// which deliver a bare string. This is the round-trip that makes the fix
    /// work at all on a peer-served segment — without it the coordinator sees
    /// `Inference(..)`, reports 500, and the reply is lost exactly as before.
    #[test]
    fn the_variant_survives_a_boundary_that_keeps_no_types() {
        let original = SwarmError::ContextWindowReached {
            used: 4097,
            window: 4096,
        };
        let flattened = original.to_string();
        let recovered = crate::error::reclassify_flattened_error(&flattened)
            .expect("the Display form must reclassify");
        assert!(
            matches!(
                recovered,
                SwarmError::ContextWindowReached {
                    used: 4097,
                    window: 4096
                }
            ),
            "got {recovered:?} from {flattened:?}"
        );
        // And a lookalike that is not this error must not be mistaken for it.
        assert!(
            crate::error::reclassify_flattened_error("Context window reached: soon").is_none(),
            "an unparseable tail is not this error"
        );
    }
}

#[cfg(test)]
mod composite_takeover_tests {
    use super::PipelineExecutor;
    use crate::types::{ModelId, NodeId, PipelineAssignment, PipelineSegment, ShardId};

    fn seg(n: u8, range: (u32, u32)) -> PipelineSegment {
        PipelineSegment {
            node_id: NodeId([n; 32]),
            shard_id: ShardId {
                model_id: ModelId("m".into()),
                index: 0,
            },
            layer_range: range,
        }
    }

    fn assignment(segments: Vec<PipelineSegment>) -> PipelineAssignment {
        PipelineAssignment {
            request_id: uuid::Uuid::nil(),
            segments,
            standbys: vec![],
            tp_groups: vec![],
            supports_speculative: false,
        }
    }

    /// One stand-in rewrites the segment in place; several replace it with one
    /// segment each, in layer order and at the same index.
    #[test]
    fn a_takeover_installs_one_segment_or_the_whole_cover_in_its_place() {
        // The single-stand-in case must be byte-for-byte what it always was.
        let mut a = assignment(vec![seg(1, (0, 8)), seg(2, (8, 32))]);
        PipelineExecutor::install_takeover(&mut a, 1, &[seg(7, (8, 32))], uuid::Uuid::nil());
        assert_eq!(a.segments.len(), 2, "one stand-in replaces one segment");
        assert_eq!(a.segments[1].node_id, NodeId([7; 32]));

        // The composite case: one segment becomes two, spliced at index 1 so
        // the segment BEFORE it and the segment AFTER it both keep their place.
        let mut a = assignment(vec![seg(1, (0, 8)), seg(2, (8, 24)), seg(3, (24, 32))]);
        PipelineExecutor::install_takeover(
            &mut a,
            1,
            &[seg(7, (8, 16)), seg(8, (16, 24))],
            uuid::Uuid::nil(),
        );
        assert_eq!(
            a.segments
                .iter()
                .map(|s| (s.node_id.clone(), s.layer_range))
                .collect::<Vec<_>>(),
            vec![
                (NodeId([1; 32]), (0, 8)),
                (NodeId([7; 32]), (8, 16)),
                (NodeId([8; 32]), (16, 24)),
                (NodeId([3; 32]), (24, 32)),
            ],
            "the cover goes in where the failed segment was, in layer order, \
             and the tail segment is still behind it"
        );

        // An empty cover is a no-op rather than a segment silently vanishing.
        let mut a = assignment(vec![seg(1, (0, 8)), seg(2, (8, 32))]);
        PipelineExecutor::install_takeover(&mut a, 1, &[], uuid::Uuid::nil());
        assert_eq!(a.segments.len(), 2);
        assert_eq!(a.segments[1].node_id, NodeId([2; 32]));
    }

    /// **The property the whole splice rests on: after it, `is_last` still
    /// names the segment that ends the pipeline.**
    ///
    /// `forward_through_segments` used to cache `segments.len()` before its
    /// loop. Nothing spliced, so it was never stale — but the moment one
    /// segment can become several, a cached count is wrong for the rest of that
    /// forward in the two places it decides something: the loop bound, so the
    /// spliced-in tail never runs, and `is_last`, which decides WHICH SEGMENT
    /// SAMPLES. An off-by-one there produces a reply that is quietly not the
    /// model's, and nothing errors.
    ///
    /// This asserts the arithmetic both ways round, because the stale reading
    /// is the one that looks right.
    #[test]
    fn is_last_still_names_the_final_segment_after_a_cover_is_spliced_in() {
        let mut a = assignment(vec![seg(1, (0, 8)), seg(2, (8, 24)), seg(3, (24, 32))]);
        let stale = a.segments.len();

        PipelineExecutor::install_takeover(
            &mut a,
            1,
            &[seg(7, (8, 16)), seg(8, (16, 24))],
            uuid::Uuid::nil(),
        );

        let live = a.segments.len();
        assert_eq!((stale, live), (3, 4), "the splice added one segment");

        // Live: the last index is the tail segment, which is the one that ends
        // the model and therefore the one that samples.
        let last_live = live - 1;
        assert_eq!(a.segments[last_live].layer_range, (24, 32));

        // Stale: the count taken before the splice names the segment BEFORE it.
        let last_stale = stale - 1;
        assert_ne!(
            a.segments[last_stale].layer_range,
            (24, 32),
            "this is the defect the live read exists to prevent — the cached \
             count names a middle segment, which would then sample and end the \
             pipeline early"
        );
        assert_eq!(a.segments[last_stale].layer_range, (16, 24));
    }
}
