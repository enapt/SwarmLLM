use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::IntoResponse;
use axum::Json;
use futures::stream::Stream;
use std::convert::Infallible;
use tokio_stream::StreamExt;

use crate::api::server::AppState;
use crate::api::sse::{send_role_preamble, StreamEvent};
use crate::api::SSE_KEEPALIVE_INTERVAL_SECS;
use crate::error::ApiError;
use crate::inference::chat_template;
use crate::inference::router::{RouterCommand, StreamingTokenEvent};
use crate::types::{ChatMessage, InferenceRequest, ModelId, SamplingParams};

use super::resolver::get_split_model_meta;
use super::types::{
    ChatChoice, ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse,
    ChatMessageResponse, ChoiceLogProbs, ChunkChoice, Delta, TokenLogProb, TopLogProb, Usage,
};

/// Build a non-streaming `ChatCompletionResponse` from inference output.
/// Shared by `router_inference` (router/distributed path with optional
/// token logprobs + multi-turn session id) and
/// `split_non_stream_response` (direct split-model fast path with neither).
/// Either field can be empty/`None` and the helper folds it into the
/// right shape.
//
// Args are all primitives that map 1:1 to fields on the response shape;
// grouping them into a struct would just rename them at every call site.
#[allow(clippy::too_many_arguments)]
fn build_chat_completion_response(
    request_id: String,
    created: i64,
    model: String,
    content: String,
    finish_reason: String,
    prompt_tokens: u32,
    completion_tokens: u32,
    session_id: Option<String>,
    token_logprobs: &[crate::inference::router::TokenLogProbEntry],
) -> ChatCompletionResponse {
    let logprobs = if token_logprobs.is_empty() {
        None
    } else {
        Some(ChoiceLogProbs {
            content: token_logprobs
                .iter()
                .map(|entry| TokenLogProb {
                    token: entry.token.clone(),
                    logprob: entry.logprob,
                    bytes: None,
                    top_logprobs: entry
                        .top_logprobs
                        .iter()
                        .map(|(t, lp)| TopLogProb {
                            token: t.clone(),
                            logprob: *lp,
                            bytes: None,
                        })
                        .collect(),
                })
                .collect(),
        })
    };
    ChatCompletionResponse {
        id: request_id,
        object: "chat.completion",
        created,
        model,
        choices: vec![ChatChoice {
            index: 0,
            message: ChatMessageResponse {
                role: "assistant".into(),
                content: Some(content),
                tool_calls: None,
            },
            finish_reason,
            logprobs,
        }],
        usage: Usage::from_counts(prompt_tokens, completion_tokens),
        session_id,
    }
}

/// Spawn a split-model generation task and return the token receiver.
///
/// Resolves model metadata, builds the prompt, and spawns the worker subprocess.
/// Returns None if the model is not loaded as a split model.
pub fn spawn_split_stream(
    state: &AppState,
    model_id: &crate::types::ModelId,
    messages: &[crate::types::ChatMessage],
    params: crate::types::SamplingParams,
    request_id: &str,
) -> Option<tokio::sync::mpsc::Receiver<crate::inference::router::StreamingTokenEvent>> {
    let meta = get_split_model_meta(state, model_id)?;
    let prompt = crate::inference::chat_template::build_prompt(
        messages,
        meta.chat_template.as_deref(),
        &meta.bos_token,
        &meta.eos_token_str,
    );
    let rid = uuid::Uuid::parse_str(request_id).unwrap_or_else(|_| uuid::Uuid::new_v4());
    let (token_tx, token_rx) =
        tokio::sync::mpsc::channel::<crate::inference::router::StreamingTokenEvent>(64);
    let pool = state.shared_state.model_process_pool.clone();
    let model_id = model_id.clone();
    let layer_range = meta.layer_range;
    // Watch handle for consumer liveness. `Sender::closed()` resolves when the
    // RECEIVER is dropped, which is what happens when the SSE bridge task exits
    // — and that task exits as soon as a send to the disconnected client fails.
    let disconnect_watch = token_tx.clone();
    tokio::spawn(async move {
        let generate = pool.generate(
            &model_id,
            layer_range,
            prompt,
            params,
            rid,
            None,
            Some(token_tx),
        );
        tokio::select! {
            result = generate => {
                if let Err(e) = result {
                    // Streaming channel closes when this scope ends. Without the
                    // log the SSE stream silently truncates with no operator-
                    // visible diagnostic — workers crashing mid-stream looked
                    // like a clean client disconnect.
                    tracing::warn!(
                        error = %e,
                        model = %model_id,
                        request_id = %rid,
                        "DIAG: local streaming generate failed",
                    );
                }
            }
            // Nobody is reading any more. Selecting this branch DROPS the
            // `generate` future, which drops its armed `ResponseGuard`, which
            // tells the worker to stop — the whole point of the cancellation
            // path. Before this existed the task was spawned detached and its
            // JoinHandle discarded, so a client that hung up mid-stream left
            // the worker generating its full token budget into a channel with
            // no receiver. Measured: 754% CPU still burning 30s after the
            // client was killed.
            _ = disconnect_watch.closed() => {
                tracing::info!(
                    model = %model_id,
                    request_id = %rid,
                    "Client disconnected mid-stream — abandoning generation",
                );
            }
        }
    });
    Some(token_rx)
}

/// Submit a streaming inference request to the router.
///
/// Creates the InferenceRequest and sends StreamSubmit. Returns receivers for
/// the final result and streaming tokens. Used by both openai and anthropic handlers.
pub async fn submit_stream_to_router(
    router_tx: &tokio::sync::mpsc::Sender<RouterCommand>,
    model_id: ModelId,
    messages: Vec<ChatMessage>,
    sampling_params: SamplingParams,
    session_id: Option<String>,
    lora_adapter: Option<String>,
    cancel: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
) -> Result<
    (
        tokio::sync::oneshot::Receiver<
            Result<crate::inference::router::InferenceOutput, crate::error::SwarmError>,
        >,
        tokio::sync::mpsc::Receiver<StreamingTokenEvent>,
    ),
    ApiError,
> {
    let (result_tx, result_rx) = tokio::sync::oneshot::channel();
    let (token_tx, token_rx) = tokio::sync::mpsc::channel::<StreamingTokenEvent>(64);

    let mut inference_req = InferenceRequest::local(
        model_id,
        messages,
        sampling_params,
        true,
        session_id,
        lora_adapter,
    );
    inference_req.cancel = cancel;

    router_tx
        .send(RouterCommand::StreamSubmit {
            request: inference_req,
            result_tx,
            token_tx,
        })
        .await
        .map_err(|_| {
            ApiError(crate::error::SwarmError::ServiceUnavailable(
                "Router unavailable".into(),
            ))
        })?;

    Ok((result_rx, token_rx))
}

/// Run non-streaming inference on a local split model.
///
/// Shared core for both OpenAI and Anthropic split_non_stream handlers.
/// Resolves model metadata, builds the prompt, and runs generation via subprocess.
pub async fn run_split_generate(
    state: &AppState,
    model_id: &crate::types::ModelId,
    messages: &[ChatMessage],
    params: SamplingParams,
    request_id: &str,
) -> Result<crate::inference::router::InferenceOutput, ApiError> {
    let meta = get_split_model_meta(state, model_id)
        .ok_or(ApiError(crate::error::SwarmError::NoModelLoaded))?;

    let prompt = chat_template::build_prompt(
        messages,
        meta.chat_template.as_deref(),
        &meta.bos_token,
        &meta.eos_token_str,
    );

    tracing::debug!(
        request_id = %request_id,
        prompt_len = prompt.len(),
        "DIAG: non-stream built prompt (subprocess)"
    );

    let rid = uuid::Uuid::parse_str(request_id).unwrap_or_else(|_| uuid::Uuid::new_v4());
    state
        .shared_state
        .model_process_pool
        .generate(model_id, meta.layer_range, prompt, params, rid, None, None)
        .await
        .map_err(ApiError)
}

/// Dispatch to `router_inference` or `router_inference_stream` based on `req.stream`.
pub(super) async fn dispatch_inference(
    router_tx: tokio::sync::mpsc::Sender<RouterCommand>,
    req: &ChatCompletionRequest,
    messages: Vec<ChatMessage>,
    request_id: String,
    created: i64,
    cancel: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
) -> Result<axum::response::Response, ApiError> {
    if req.stream {
        router_inference_stream(router_tx, req, messages, request_id, created, cancel).await
    } else {
        router_inference(router_tx, req, messages, request_id, created, cancel).await
    }
}

/// Route inference through the InferenceRouter (non-streaming).
pub(super) async fn router_inference(
    router_tx: tokio::sync::mpsc::Sender<RouterCommand>,
    req: &ChatCompletionRequest,
    messages: Vec<ChatMessage>,
    request_id: String,
    created: i64,
    cancel: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
) -> Result<axum::response::Response, ApiError> {
    let mut inference_req = InferenceRequest::local(
        ModelId(req.model.clone()),
        messages,
        req.to_sampling_params(),
        false,
        req.session_id.clone(),
        req.lora_adapter.clone(),
    );
    inference_req.cancel = cancel;

    let output = crate::api::submit_to_router(&router_tx, inference_req).await?;

    let response = build_chat_completion_response(
        request_id,
        created,
        req.model.clone(),
        output.content,
        output.finish_reason,
        output.prompt_tokens,
        output.completion_tokens,
        output.session_id,
        &output.token_logprobs,
    );

    Ok(Json(response).into_response())
}

/// Route streaming inference through the InferenceRouter.
///
/// Submits the request via `StreamSubmit` so the pipeline executor sends decoded
/// tokens incrementally. Each token is forwarded as an SSE chunk, providing true
/// token-by-token streaming for distributed inference.
async fn router_inference_stream(
    router_tx: tokio::sync::mpsc::Sender<RouterCommand>,
    req: &ChatCompletionRequest,
    messages: Vec<ChatMessage>,
    request_id: String,
    created: i64,
    cancel: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
) -> Result<axum::response::Response, ApiError> {
    let (result_rx, mut token_rx) = submit_stream_to_router(
        &router_tx,
        ModelId(req.model.clone()),
        messages,
        req.to_sampling_params(),
        req.session_id.clone(),
        req.lora_adapter.clone(),
        cancel,
    )
    .await?;

    let model_name = req.model.clone();
    let rid = request_id.clone();
    let stream_session_id = req.session_id.clone();
    // OpenAI 2024+ stream_options.include_usage: opt-in via the extras
    // HashMap (the request type doesn't model `stream_options` explicitly,
    // so it round-trips through the catch-all). When set, emit a final
    // chunk with `choices: []` and the usage object filled.
    let include_usage = req
        .extras
        .get("stream_options")
        .and_then(|v| v.get("include_usage"))
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    // Bridge the streaming token channel into SSE events
    let (sse_tx, sse_rx) = tokio::sync::mpsc::channel::<StreamEvent>(64);

    tokio::spawn(async move {
        let stream_start = std::time::Instant::now();
        let mut token_count: u64 = 0;
        // Track authoritative token counts for the optional terminal usage chunk.
        let mut prompt_tokens_final: Option<u32> = None;
        let mut completion_tokens_final: Option<u32> = None;

        // Send initial role delta
        if !send_role_preamble(&sse_tx).await {
            tracing::warn!(
                "DIAG: SSE role delta send failed — client disconnected before stream started"
            );
            return;
        }

        // Read tokens from the pipeline as they arrive
        let mut got_finish = false;
        let mut client_disconnected = false;
        while let Some(event) = token_rx.recv().await {
            if let Some(ref reason) = event.finish_reason {
                got_finish = true;
                if !event.text.is_empty() {
                    token_count += 1;
                    if sse_tx
                        .send(StreamEvent::Delta {
                            content: Some(event.text),
                            role: None,
                            finish_reason: None,
                        })
                        .await
                        .is_err()
                    {
                        tracing::warn!(
                            token_count,
                            "DIAG: SSE final text delta send failed — client disconnected"
                        );
                    }
                }
                if sse_tx
                    .send(StreamEvent::Delta {
                        content: None,
                        role: None,
                        finish_reason: Some(reason.clone()),
                    })
                    .await
                    .is_err()
                {
                    tracing::debug!(token_count, finish_reason = %reason, "DIAG: SSE finish delta send failed — client disconnected");
                }
                break;
            }
            if !event.text.is_empty() {
                token_count += 1;
                if sse_tx
                    .send(StreamEvent::Delta {
                        content: Some(event.text),
                        role: None,
                        finish_reason: None,
                    })
                    .await
                    .is_err()
                {
                    tracing::warn!(
                        token_count,
                        elapsed_ms = stream_start.elapsed().as_millis() as u64,
                        "DIAG: SSE client disconnected mid-stream — cancelling pipeline"
                    );
                    client_disconnected = true;
                    break;
                }
            }
        }

        // Client disconnected: drop token_rx to signal pipeline to stop generating,
        // and skip the result_rx fallback to avoid blocking on a now-useless pipeline.
        if client_disconnected {
            drop(token_rx);
            tracing::info!(
                token_count,
                elapsed_ms = stream_start.elapsed().as_millis() as u64,
                "SSE pipeline cancelled due to client disconnect"
            );
            // Send Done in case sse_rx is still draining
            let _ = sse_tx.send(StreamEvent::Done).await;
            return;
        }

        // Fallback: if pipeline finished without sending streaming events
        // (e.g., local-only path or error), read the final result.
        if !got_finish {
            tracing::debug!(
                token_count,
                "DIAG: SSE stream no finish event from pipeline, reading result_rx fallback"
            );
            match result_rx.await {
                Ok(Ok(output)) => {
                    prompt_tokens_final = Some(output.prompt_tokens);
                    completion_tokens_final = Some(output.completion_tokens);
                    if !output.content.is_empty()
                        && sse_tx
                            .send(StreamEvent::Delta {
                                content: Some(output.content),
                                role: None,
                                finish_reason: None,
                            })
                            .await
                            .is_err()
                    {
                        tracing::warn!(
                            "DIAG: SSE fallback content send failed — client disconnected"
                        );
                    }
                    if sse_tx
                        .send(StreamEvent::Delta {
                            content: None,
                            role: None,
                            finish_reason: Some(output.finish_reason.clone()),
                        })
                        .await
                        .is_err()
                    {
                        tracing::debug!(finish_reason = %output.finish_reason, "DIAG: SSE fallback finish send failed");
                    }
                }
                Ok(Err(e)) => {
                    tracing::debug!(error = %e, "DIAG: SSE fallback got pipeline error");
                    if sse_tx
                        .send(StreamEvent::Error {
                            message: format!("{e}"),
                        })
                        .await
                        .is_err()
                    {
                        tracing::debug!("DIAG: SSE error event send failed — client disconnected");
                    }
                }
                Err(_) => {
                    tracing::debug!("DIAG: SSE result_rx channel dropped — pipeline task died without sending result");
                    if sse_tx
                        .send(StreamEvent::Delta {
                            content: None,
                            role: None,
                            finish_reason: Some("stop".into()),
                        })
                        .await
                        .is_err()
                    {
                        tracing::debug!("DIAG: SSE channel-drop finish send also failed");
                    }
                }
            }
        } else if include_usage {
            // include_usage path with token-stream finish: the streamed
            // events already conveyed the deltas, but only `result_rx`
            // carries authoritative token counts. Best-effort await with
            // a short timeout so an upstream stall doesn't keep the
            // client SSE open longer than necessary.
            match tokio::time::timeout(std::time::Duration::from_secs(5), result_rx).await {
                Ok(Ok(Ok(output))) => {
                    prompt_tokens_final = Some(output.prompt_tokens);
                    completion_tokens_final = Some(output.completion_tokens);
                }
                _ => {
                    tracing::debug!(
                        "DIAG: include_usage requested but result_rx timed out or errored — falling back to streamed token count"
                    );
                }
            }
        }

        if include_usage {
            let prompt_tokens = prompt_tokens_final.unwrap_or(0);
            let completion_tokens = completion_tokens_final.unwrap_or(token_count as u32);
            if sse_tx
                .send(StreamEvent::Usage {
                    prompt_tokens,
                    completion_tokens,
                })
                .await
                .is_err()
            {
                tracing::debug!("DIAG: include_usage chunk send failed — client disconnected");
            }
        }
        if sse_tx.send(StreamEvent::Done).await.is_err() {
            tracing::debug!("DIAG: SSE Done send failed — client already disconnected");
        }
        let elapsed = stream_start.elapsed();
        tracing::info!(
            elapsed_ms = elapsed.as_millis() as u64,
            token_count,
            "DIAG: SSE distributed stream completed"
        );
    });

    Ok(stream_events_to_sse(sse_rx, rid, created, model_name, stream_session_id).into_response())
}

/// Direct split-model generation (non-streaming).
///
/// Bypasses the distributed pipeline executor and generates directly from
/// the locally loaded SplitModel. Much faster for single-node inference
/// because it avoids per-token pipeline coordination overhead.
/// Builds the prompt from the model's own chat template to avoid template mismatch.
pub(super) async fn split_non_stream_response(
    state: AppState,
    request_id: String,
    created: i64,
    model_name: String,
    messages: Vec<ChatMessage>,
    params: SamplingParams,
    model_id: crate::types::ModelId,
) -> Result<axum::response::Response, ApiError> {
    let output =
        run_split_generate(&state, &model_id, &messages, params.clone(), &request_id).await?;

    let response = build_chat_completion_response(
        request_id,
        created,
        model_name,
        output.content,
        output.finish_reason.clone(),
        output.prompt_tokens,
        output.completion_tokens,
        None,
        &[],
    );

    Ok(Json(response).into_response())
}

/// Direct split-model streaming generation.
///
/// Same fast path as split_non_stream_response but streams tokens via SSE.
/// Builds the prompt from the model's own chat template to avoid template mismatch.
pub(super) async fn split_stream_response(
    state: AppState,
    request_id: String,
    created: i64,
    model_name: String,
    messages: Vec<ChatMessage>,
    params: SamplingParams,
    model_id: crate::types::ModelId,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let (tx, rx) = tokio::sync::mpsc::channel::<StreamEvent>(64);
    let request_id_inner = request_id.clone();

    tokio::spawn(async move {
        let request_id = request_id_inner;
        // Send initial role delta
        let stream_start = std::time::Instant::now();
        let mut token_count: u64 = 0;

        if !send_role_preamble(&tx).await {
            tracing::debug!("DIAG: split stream role delta send failed — client disconnected");
            return;
        }

        // Spawn split-model generation and get token receiver
        let mut token_rx =
            match spawn_split_stream(&state, &model_id, &messages, params, &request_id) {
                Some(rx) => rx,
                None => {
                    tracing::debug!(model_id = %model_id, "DIAG: split stream model not found");
                    let _ = tx.send(StreamEvent::Done).await;
                    return;
                }
            };

        // Forward streaming tokens from the worker to SSE events.
        // Default to "stop"; the worker will override with "length" if the
        // generation hit max_tokens. "length" as a default would tell clients
        // the response was truncated even on a clean exit.
        let mut finish = "stop".to_string();
        while let Some(event) = token_rx.recv().await {
            if let Some(fr) = &event.finish_reason {
                finish = fr.clone();
                break;
            }
            token_count += 1;
            if tx
                .send(StreamEvent::Delta {
                    content: Some(event.text),
                    role: None,
                    finish_reason: None,
                })
                .await
                .is_err()
            {
                tracing::warn!(
                    token_count,
                    elapsed_ms = stream_start.elapsed().as_millis() as u64,
                    "DIAG: split stream client disconnected mid-decode"
                );
                return;
            }
        }

        tracing::info!(
            model_id = %model_id,
            token_count,
            finish = %finish,
            "DIAG: split stream decode loop complete (subprocess)"
        );

        // Send finish
        if tx
            .send(StreamEvent::Delta {
                content: None,
                role: None,
                finish_reason: Some(finish),
            })
            .await
            .is_err()
        {
            tracing::debug!(token_count, "DIAG: split stream finish delta send failed");
        }
        if tx.send(StreamEvent::Done).await.is_err() {
            tracing::debug!("DIAG: split stream Done send failed — client disconnected");
        }
        let elapsed = stream_start.elapsed();
        tracing::info!(
            elapsed_ms = elapsed.as_millis() as u64,
            token_count,
            "DIAG: split stream completed"
        );
    });

    stream_events_to_sse(rx, request_id, created, model_name, None)
}

pub(super) async fn stream_response(
    state: AppState,
    request_id: String,
    created: i64,
    model_name: String,
    prompt: String,
    params: SamplingParams,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let (tx, rx) = tokio::sync::mpsc::channel::<StreamEvent>(64);

    // Spawn generation in background
    tokio::spawn(async move {
        let stream_start = std::time::Instant::now();
        let mut token_count: u64 = 0;

        // Send initial role delta
        if !send_role_preamble(&tx).await {
            tracing::debug!("DIAG: local stream role delta send failed — client disconnected");
            return;
        }

        let mut executor = state.executor.lock().await;
        let result = executor.generate_stream(&prompt, &params, |token| {
            token_count += 1;
            let send_result = tx.try_send(StreamEvent::Delta {
                content: Some(token.to_string()),
                role: None,
                finish_reason: None,
            });
            if send_result.is_err() {
                tracing::warn!(
                    token_count,
                    "DIAG: local stream token send failed — channel full or client disconnected"
                );
            }
            send_result.is_ok()
        });

        // Send finish reason. OpenAI spec restricts finish_reason to
        // stop|length|tool_calls|content_filter|function_call. Map an
        // execution error to "stop" — the error is already logged here and
        // the caller has separate paths to surface it (HTTP 500 before the
        // SSE opens, or an `error` SSE event for in-stream failures).
        let finish = match result {
            Ok(r) => r.finish_reason.as_str().to_string(),
            Err(ref e) => {
                tracing::error!(error = %e, "DIAG: local stream generate_stream error");
                "stop".to_string()
            }
        };
        if tx
            .send(StreamEvent::Delta {
                content: None,
                role: None,
                finish_reason: Some(finish.clone()),
            })
            .await
            .is_err()
        {
            tracing::debug!(token_count, finish_reason = %finish, "DIAG: local stream finish send failed");
        }
        if tx.send(StreamEvent::Done).await.is_err() {
            tracing::debug!("DIAG: local stream Done send failed — client disconnected");
        }
        let elapsed = stream_start.elapsed();
        tracing::info!(
            elapsed_ms = elapsed.as_millis() as u64,
            token_count,
            "DIAG: local stream completed"
        );
    });

    stream_events_to_sse(rx, request_id, created, model_name, None)
}

/// Convert a StreamEvent receiver into an SSE stream of OpenAI-format chat completion chunks.
fn stream_events_to_sse(
    rx: tokio::sync::mpsc::Receiver<StreamEvent>,
    request_id: String,
    created: i64,
    model_name: String,
    session_id: Option<String>,
) -> Sse<impl futures::Stream<Item = Result<Event, Infallible>>> {
    let mut json_buf = Vec::with_capacity(512);
    let stream = tokio_stream::wrappers::ReceiverStream::new(rx).map(move |event| match event {
        StreamEvent::Delta {
            content,
            role,
            finish_reason,
        } => {
            let sid = if finish_reason.is_some() {
                session_id.clone()
            } else {
                None
            };
            let chunk = ChatCompletionChunk {
                id: request_id.clone(),
                object: "chat.completion.chunk",
                created,
                model: model_name.clone(),
                choices: vec![ChunkChoice {
                    index: 0,
                    delta: Delta {
                        role,
                        content,
                        tool_calls: None,
                    },
                    finish_reason,
                    logprobs: None,
                }],
                session_id: sid,
                usage: None,
            };
            json_buf.clear();
            let json = if serde_json::to_writer(&mut json_buf, &chunk).is_ok() {
                // R108: per-token. Move the buffer into String without
                // copying; serde_json::to_writer guarantees valid UTF-8 so
                // `from_utf8` cannot fail. Re-prime json_buf for the next
                // event with the same starting capacity.
                let taken = std::mem::take(&mut json_buf);
                json_buf = Vec::with_capacity(512);
                String::from_utf8(taken).unwrap_or_default()
            } else {
                String::new()
            };
            Ok::<_, Infallible>(Event::default().data(json))
        }
        StreamEvent::Error { message } => {
            let error_json = serde_json::json!({
                "error": {
                    "message": message,
                    "type": "server_error"
                }
            });
            Ok(Event::default().data(serde_json::to_string(&error_json).unwrap_or_default()))
        }
        StreamEvent::Usage {
            prompt_tokens,
            completion_tokens,
        } => {
            // OpenAI 2024+ stream_options.include_usage: emit one extra
            // chunk with empty `choices: []` and the usage object filled,
            // immediately before `[DONE]`.
            let chunk = ChatCompletionChunk {
                id: request_id.clone(),
                object: "chat.completion.chunk",
                created,
                model: model_name.clone(),
                choices: Vec::new(),
                session_id: None,
                usage: Some(crate::api::openai::types::Usage::from_counts(
                    prompt_tokens,
                    completion_tokens,
                )),
            };
            let json = serde_json::to_string(&chunk).unwrap_or_default();
            Ok(Event::default().data(json))
        }
        StreamEvent::Done => Ok(Event::default().data("[DONE]")),
    });
    Sse::new(stream).keep_alive(
        KeepAlive::new().interval(std::time::Duration::from_secs(SSE_KEEPALIVE_INTERVAL_SECS)),
    )
}
