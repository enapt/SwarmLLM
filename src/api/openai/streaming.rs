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
    tools_requested: bool,
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
    // A local model can only express a tool call as text, so recover it here.
    // Gated on the request actually carrying tools: otherwise a model asked to
    // "reply in JSON" could have a legitimate answer reinterpreted as a call.
    // Truncated output deliberately stays text (see `api::tool_parse`).
    let (content, tool_calls, finish_reason) = if tools_requested {
        match crate::api::tool_parse::parse_tool_calls(&content) {
            Some(parsed) => {
                let calls: Vec<crate::api::openai::types::ToolCall> = parsed
                    .into_iter()
                    .map(|c| crate::api::openai::types::ToolCall {
                        id: c.id,
                        tool_type: "function".into(),
                        function: crate::api::openai::types::FunctionCall {
                            name: c.name,
                            arguments: c.arguments,
                        },
                    })
                    .collect();
                // OpenAI sends content: null alongside tool_calls, and
                // finish_reason must be "tool_calls" or clients never dispatch
                // them (the reported response said "length").
                (None, Some(calls), "tool_calls".to_string())
            }
            None => (Some(content), None, finish_reason),
        }
    } else {
        (Some(content), None, finish_reason)
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
                content,
                tool_calls,
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
/// Why a local split-model stream produced nothing, in the terms a client needs
/// to hear it: the message AND the error type.
///
/// The type is recorded here, at the one place the typed error still exists,
/// rather than left for the SSE layer to infer. It used to be dropped
/// (`e.to_string()`), so both stream encoders had to pick a type blind and both
/// picked the same wrong one — reporting an over-long prompt, a policy refusal,
/// and a genuine crash identically as this server failing. Do not re-derive a
/// type by matching on `message`: that is the substring-matching-user-prose trap
/// (gotcha #295), and the wording is what changes.
#[derive(Clone, Debug)]
pub struct StreamFailure {
    pub message: String,
    pub error_type: &'static str,
}

impl StreamFailure {
    /// Classify at the point of failure, where the typed error is still in hand.
    pub fn from_error(err: &crate::error::SwarmError) -> Self {
        let (_status, _message, error_type) = crate::error::classify_error(err);
        // `err.to_string()`, not the classified message: that one is
        // deliberately redacted to "An internal error occurred" for the
        // catch-all, and this slot feeds an operator-facing log as well as the
        // client frame.
        StreamFailure {
            message: err.to_string(),
            error_type,
        }
    }
}

/// Records why a local split-model stream produced nothing, so the SSE layer can
/// tell the client instead of closing the stream silently.
pub type SplitStreamFailure = std::sync::Arc<std::sync::Mutex<Option<StreamFailure>>>;

/// Token counts from a completed local split-model generation.
///
/// The counts come back on the `generate` future, not on the token stream, so
/// they have to be carried out of the spawned task separately — the same shape
/// as [`SplitStreamFailure`].
pub type SplitStreamUsage = std::sync::Arc<std::sync::Mutex<Option<(u32, u32)>>>;

/// A local split-model stream: the token receiver, the failure slot to consult
/// if the stream ends without producing tokens, and the final token counts.
pub type SplitStream = (
    tokio::sync::mpsc::Receiver<crate::inference::router::StreamingTokenEvent>,
    SplitStreamFailure,
    SplitStreamUsage,
);

/// Keeps an `active_traces` entry alive for the duration of a request and
/// removes it however the request ends.
///
/// The router path gets this from `ActivePipelineGuard`; the local split fast
/// path has no pipeline, so it needs its own. RAII rather than an explicit
/// removal because the streaming task has two exits — generation finishing and
/// the client disconnecting — and a leaked entry would keep a stale request
/// showing as "in progress" on every surface forever.
pub(crate) struct TraceGuard {
    state: std::sync::Arc<crate::daemon::SharedState>,
    request_id: uuid::Uuid,
}

impl TraceGuard {
    pub(crate) fn register(
        state: &std::sync::Arc<crate::daemon::SharedState>,
        request_id: uuid::Uuid,
        trace: std::sync::Arc<crate::inference::trace::RequestTrace>,
    ) -> Self {
        state.active_traces.insert(request_id, trace);
        Self {
            state: state.clone(),
            request_id,
        }
    }
}

impl Drop for TraceGuard {
    fn drop(&mut self) {
        self.state.active_traces.remove(&self.request_id);
    }
}

pub fn spawn_split_stream(
    state: &AppState,
    model_id: &crate::types::ModelId,
    messages: &[crate::types::ChatMessage],
    params: crate::types::SamplingParams,
    request_id: &str,
    // Required rather than defaulted: a path that silently passes nothing
    // produces a request the metrics never see, which is the bug this
    // parameter was added to fix. `None` is legitimate only where no trace
    // exists at all.
    trace: Option<std::sync::Arc<crate::inference::trace::RequestTrace>>,
) -> Option<SplitStream> {
    let meta = get_split_model_meta(&state.shared_state, model_id)?;
    let prompt = crate::inference::chat_template::build_prompt(
        messages,
        meta.chat_template.as_deref(),
        &meta.bos_token,
        &meta.eos_token_str,
        Some(model_id.0.as_str()),
    );

    // Add the stop strings implied by the chat template to whatever the caller
    // asked for (external report 2026-07-25: a model returned raw
    // `<|im_end|>` / `<|im_start|>assistant` markers as visible content).
    //
    // The template tells us how a turn ends, which is not always the same as
    // the tokenizer's declared EOS id — a GGUF carrying a ChatML template but a
    // Llama-style `eos_token_id` will emit `<|im_end|>`, which is not an EOS
    // token, so generation runs on and the markers reach the user verbatim.
    //
    // `local_exec.rs` and `distributed.rs` both already did this; this fast path
    // (used by BOTH the OpenAI and Anthropic local-complete routes, and so by
    // the dashboard compare page) was the one place that didn't, which is why
    // the leak only showed up on some models via some endpoints.
    let params =
        crate::inference::chat_template::with_template_stops(params, meta.chat_template.as_deref());
    let rid = crate::api::request_uuid(request_id);
    let (token_tx, token_rx) = crate::inference::router::StreamingTokenTx::channel(64);
    // Attach the caller's trace so TTFT is stamped on the first token that
    // carries text. `StreamingTokenTx` exists precisely so no emit site has to
    // remember this — but the sender still has to be TOLD which trace it
    // belongs to, and this path never did, so every locally-streamed request
    // reported no first-token time.
    let token_tx = match trace.as_ref() {
        Some(t) => token_tx.with_trace(t.clone()),
        None => token_tx,
    };
    // Records why generation stopped when it produced nothing. Without this the
    // channel simply closes and the client sees an empty-but-successful stream,
    // so the dashboard had to *guess* a reason ("the model might still be
    // loading") — which is what an out-of-VRAM model switch looked like in the
    // external report of 2026-07-25.
    let failure: SplitStreamFailure = std::sync::Arc::new(std::sync::Mutex::new(None));
    let failure_sink = failure.clone();
    // `stream_options.include_usage` asks for real token counts on a streamed
    // response. They arrive on the generate future rather than the token
    // stream, so capture them here (external report 2026-07-25: streamed
    // responses reported {0,0,0} while the non-streaming equivalent did not).
    let usage: SplitStreamUsage = std::sync::Arc::new(std::sync::Mutex::new(None));
    let usage_sink = usage.clone();
    let pool = state.shared_state.model_process_pool.clone();
    let model_id = model_id.clone();
    let layer_range = meta.layer_range;
    // Watch handle for consumer liveness. `Sender::closed()` resolves when the
    // RECEIVER is dropped, which is what happens when the SSE bridge task exits
    // — and that task exits as soon as a send to the disconnected client fails.
    let disconnect_watch = token_tx.clone();
    // Register an in-flight trace so worker progress has somewhere to land.
    //
    // This fast path bypasses the router, and the router was the ONLY place
    // inserting into `active_traces` — so a local split request produced no
    // trace at all, and a progress report keyed to it was silently dropped by
    // the forwarder. That is precisely the case progress exists for: a long
    // prompt on a CPU-only node, answered locally, prefilling for minutes.
    //
    // Guarded by `TraceGuard` rather than removed at the end of the happy path,
    // because this task can also exit through the disconnect branch below.
    // Reuse the caller's trace when it supplied one, rather than making a
    // second object for the same request id. Both would register under `rid`,
    // so the later one silently displaced the earlier and only one of them
    // could ever be published — leaving progress reports landing on an object
    // nothing would go on to record.
    let trace_guard = TraceGuard::register(
        &state.shared_state,
        rid,
        trace.clone().unwrap_or_else(|| {
            std::sync::Arc::new(crate::inference::trace::RequestTrace::new(
                rid,
                model_id.0.clone(),
                "chat",
            ))
        }),
    );
    tokio::spawn(async move {
        let _trace_guard = trace_guard;
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
                if let Ok(ref out) = result {
                    if let Ok(mut slot) = usage_sink.lock() {
                        *slot = Some((out.prompt_tokens, out.completion_tokens));
                    }
                }
                if let Err(e) = result {
                    // Streaming channel closes when this scope ends. Without the
                    // log the SSE stream silently truncates with no operator-
                    // visible diagnostic — workers crashing mid-stream looked
                    // like a clean client disconnect.
                    crate::log_failure!(
                        &e,
                        error = %e,
                        model = %model_id,
                        request_id = %rid,
                        "DIAG: local streaming generate failed",
                    );
                    // Also hand the reason to the SSE layer so the client is
                    // told what happened instead of receiving silence.
                    if let Ok(mut slot) = failure_sink.lock() {
                        *slot = Some(StreamFailure::from_error(&e));
                    }
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
    Some((token_rx, failure, usage))
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
        // The id the router will trace this request under. `InferenceRequest`
        // mints its own, unrelated to the public `swarm-<hex>` request id, so
        // the caller cannot derive it — returning it is the only way a caller
        // can look the request up in `active_traces`.
        uuid::Uuid,
    ),
    ApiError,
> {
    let (result_tx, result_rx) = tokio::sync::oneshot::channel();
    let (token_tx, token_rx) = crate::inference::router::StreamingTokenTx::channel(64);

    let mut inference_req = InferenceRequest::local(
        model_id,
        messages,
        sampling_params,
        true,
        session_id,
        lora_adapter,
    );
    inference_req.cancel = cancel;
    let traced_id = inference_req.id;

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

    Ok((result_rx, token_rx, traced_id))
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
    let meta = get_split_model_meta(&state.shared_state, model_id)
        .ok_or(ApiError(crate::error::SwarmError::NoModelLoaded))?;

    let prompt = chat_template::build_prompt(
        messages,
        meta.chat_template.as_deref(),
        &meta.bos_token,
        &meta.eos_token_str,
        Some(model_id.0.as_str()),
    );

    tracing::debug!(
        request_id = %request_id,
        prompt_len = prompt.len(),
        "DIAG: non-stream built prompt (subprocess)"
    );

    let params =
        crate::inference::chat_template::with_template_stops(params, meta.chat_template.as_deref());

    let rid = crate::api::request_uuid(request_id);

    // This path deliberately bypasses the router, so it must build its own
    // trace — it is the local-complete fast path the dashboard chat uses, i.e.
    // the most-seen surface, and leaving it untraced would show no route
    // exactly where a user is most likely to look. Route is genuinely Local:
    // one node, whole model, no wire.
    let trace = std::sync::Arc::new(crate::inference::trace::RequestTrace::new(
        rid,
        model_id.0.clone(),
        "chat",
    ));
    // Visible as in-flight for as long as it runs, so worker progress lands on
    // it. Without this the trace existed only to be published at the END, which
    // is no help at all to somebody asking why a request is taking minutes.
    let _trace_guard = TraceGuard::register(&state.shared_state, rid, trace.clone());
    trace.mark_dequeued();
    trace.mark_assembled(
        crate::inference::trace::Route::Local,
        crate::inference::trace::local_segment(
            state.shared_state.identity.node_id(),
            meta.layer_range,
        ),
        // No scheduling happens on the local-complete fast path — it bypasses
        // the scheduler entirely, so reporting 0 is accurate, not a placeholder.
        0,
    );

    let mut output = state
        .shared_state
        .model_process_pool
        .generate(model_id, meta.layer_range, prompt, params, rid, None, None)
        .await
        .map_err(ApiError)?;

    trace.mark_finished(
        crate::inference::trace::Outcome::Ok,
        output.prompt_tokens,
        output.completion_tokens,
    );
    state.shared_state.publish_request_trace(&trace);
    output.trace = Some(trace.snapshot());
    Ok(output)
}

/// Dispatch to `router_inference` or `router_inference_stream` based on `req.stream`.
pub(super) async fn dispatch_inference(
    state: &AppState,
    router_tx: tokio::sync::mpsc::Sender<RouterCommand>,
    req: &ChatCompletionRequest,
    messages: Vec<ChatMessage>,
    request_id: String,
    created: i64,
    cancel: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
) -> Result<axum::response::Response, ApiError> {
    if req.stream {
        router_inference_stream(state, router_tx, req, messages, request_id, created, cancel).await
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
    let trace = output.trace.clone();

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
        req.tools.as_ref().is_some_and(|t| !t.is_empty()),
    );

    Ok(crate::api::attach_route_headers(
        Json(response).into_response(),
        trace.as_ref(),
        false,
    ))
}

/// Route streaming inference through the InferenceRouter.
///
/// Submits the request via `StreamSubmit` so the pipeline executor sends decoded
/// tokens incrementally. Each token is forwarded as an SSE chunk, providing true
/// token-by-token streaming for distributed inference.
/// Emit recovered tool calls as one OpenAI streaming `tool_calls` delta.
///
/// Returns true if any call was emitted, so the caller can set
/// `finish_reason: "tool_calls"`. Shared by the router path and the local
/// split-model fast path: a wire format written twice diverges, which is
/// precisely the class of bug this release fixed in stop-string handling.
async fn emit_openai_tool_calls(
    tx: &tokio::sync::mpsc::Sender<StreamEvent>,
    buffered: &str,
) -> bool {
    let Some(parsed) = crate::api::tool_parse::parse_tool_calls(buffered) else {
        return false;
    };
    let calls: Vec<crate::api::openai::StreamToolCall> = parsed
        .into_iter()
        .enumerate()
        .map(|(i, c)| crate::api::openai::StreamToolCall {
            index: i as u32,
            id: Some(c.id),
            tool_type: Some("function".into()),
            function: Some(crate::api::openai::StreamFunctionCall {
                name: Some(c.name),
                arguments: Some(c.arguments),
            }),
        })
        .collect();
    if calls.is_empty() {
        return false;
    }
    let _ = crate::api::sse_send_live(tx, StreamEvent::ToolCalls { calls }).await;
    true
}

async fn router_inference_stream(
    state: &AppState,
    router_tx: tokio::sync::mpsc::Sender<RouterCommand>,
    req: &ChatCompletionRequest,
    messages: Vec<ChatMessage>,
    request_id: String,
    created: i64,
    cancel: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
) -> Result<axum::response::Response, ApiError> {
    let (result_rx, mut token_rx, traced_id) = submit_stream_to_router(
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

    // Captured before the spawn: `req` is borrowed and cannot cross into the
    // task. A local model expresses tool calls as text, so this decides whether
    // the token stream is forwarded live or withheld for inspection.
    let tools_requested = req.tools.as_ref().is_some_and(|t| !t.is_empty());

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
        // Withheld text when tools are in play — see `emit_openai_tool_calls`.
        let mut buffered = String::new();
        let mut client_disconnected = false;
        loop {
            let event = tokio::select! {
                biased;
                // Client dropped the connection: the SSE body's receiver is gone.
                // Catch it the instant it happens rather than only when the next
                // token's send fails — for a slow (mostly-CPU) generation that gap
                // can be tens of seconds of wasted worker compute (external report,
                // v0.3.15: ~27s late). The `client_disconnected` block below drops
                // token_rx to signal the pipeline to stop.
                _ = sse_tx.closed() => {
                    tracing::warn!(
                        token_count,
                        elapsed_ms = stream_start.elapsed().as_millis() as u64,
                        "DIAG: SSE client disconnected (connection closed) — cancelling pipeline"
                    );
                    client_disconnected = true;
                    break;
                }
                maybe = token_rx.recv() => match maybe {
                    Some(e) => e,
                    None => break,
                },
            };
            if let Some(ref reason) = event.finish_reason {
                got_finish = true;
                if !event.text.is_empty() {
                    token_count += 1;
                    if tools_requested {
                        buffered.push_str(&event.text);
                    } else if sse_tx
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
                // Flush what tool inspection withheld before the finish delta,
                // so finish_reason can reflect a tool call.
                let mut reason = reason.clone();
                if tools_requested && !buffered.is_empty() {
                    if emit_openai_tool_calls(&sse_tx, &buffered).await {
                        reason = "tool_calls".to_string();
                    } else {
                        let _ = crate::api::sse_send_live(
                            &sse_tx,
                            StreamEvent::Delta {
                                content: Some(buffered.clone()),
                                role: None,
                                finish_reason: None,
                            },
                        )
                        .await;
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
                if tools_requested {
                    buffered.push_str(&event.text);
                    continue;
                }
                // Closed OR stalled (non-reading) consumer → cancel the pipeline.
                if !crate::api::sse_send_live(
                    &sse_tx,
                    StreamEvent::Delta {
                        content: Some(event.text),
                        role: None,
                        finish_reason: None,
                    },
                )
                .await
                {
                    tracing::warn!(
                        token_count,
                        elapsed_ms = stream_start.elapsed().as_millis() as u64,
                        "DIAG: SSE consumer gone mid-stream (closed or not reading) — cancelling pipeline"
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
                            error_type: crate::error::classify_error(&e).2,
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

    let progress_handle = Some((state.shared_state.clone(), traced_id));
    Ok(stream_events_to_sse(
        sse_rx,
        rid,
        created,
        model_name,
        stream_session_id,
        progress_handle,
    )
    .into_response())
}

/// Direct split-model generation (non-streaming).
///
/// Bypasses the distributed pipeline executor and generates directly from
/// the locally loaded SplitModel. Much faster for single-node inference
/// because it avoids per-token pipeline coordination overhead.
/// Builds the prompt from the model's own chat template to avoid template mismatch.
#[allow(clippy::too_many_arguments)]
pub(super) async fn split_non_stream_response(
    state: AppState,
    request_id: String,
    created: i64,
    model_name: String,
    messages: Vec<ChatMessage>,
    params: SamplingParams,
    model_id: crate::types::ModelId,
    tools_requested: bool,
) -> Result<axum::response::Response, ApiError> {
    let output =
        run_split_generate(&state, &model_id, &messages, params.clone(), &request_id).await?;
    let trace = output.trace.clone();

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
        tools_requested,
    );

    Ok(crate::api::attach_route_headers(
        Json(response).into_response(),
        trace.as_ref(),
        false,
    ))
}

/// Direct split-model streaming generation.
///
/// Same fast path as split_non_stream_response but streams tokens via SSE.
/// Builds the prompt from the model's own chat template to avoid template mismatch.
#[allow(clippy::too_many_arguments)]
pub(super) async fn split_stream_response(
    state: AppState,
    request_id: String,
    created: i64,
    model_name: String,
    messages: Vec<ChatMessage>,
    params: SamplingParams,
    model_id: crate::types::ModelId,
    tools_requested: bool,
    include_usage: bool,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    // Captured before `state` is moved into the generation task. This is the
    // LOCAL-model streaming path — the one most likely to sit in a long prefill
    // on a modest machine — so omitting progress here would leave it missing
    // from exactly the case it exists for.
    let progress_handle = Some((
        state.shared_state.clone(),
        crate::api::request_uuid(&request_id),
    ));
    let (tx, rx) = tokio::sync::mpsc::channel::<StreamEvent>(64);
    let request_id_inner = request_id.clone();

    // Like `split_non_stream_response`, this path bypasses the router and so
    // must build its own trace. It had none at all, so a locally-streamed
    // request was invisible: `swarmllm_inference_requests_total` did not move,
    // no route was recorded, and the OTel first-token and per-output-token
    // histograms stayed empty however much the node served. Verified live
    // 2026-08-05 — 11 before a streaming request, 11 after. Streaming is the
    // path every interactive client uses, so on a single-node install the
    // shipped Grafana dashboards could read near-zero while the machine worked
    // continuously.
    let stream_rid = crate::api::request_uuid(&request_id);
    let stream_trace = std::sync::Arc::new(crate::inference::trace::RequestTrace::new(
        stream_rid,
        model_id.0.clone(),
        "chat",
    ));

    tokio::spawn(async move {
        let request_id = request_id_inner;
        // Visible as in-flight while it runs, so worker progress lands on it.
        let _trace_guard =
            TraceGuard::register(&state.shared_state, stream_rid, stream_trace.clone());
        stream_trace.mark_dequeued();
        // Send initial role delta
        let stream_start = std::time::Instant::now();
        let mut token_count: u64 = 0;

        if !send_role_preamble(&tx).await {
            tracing::debug!("DIAG: split stream role delta send failed — client disconnected");
            return;
        }

        // Spawn split-model generation and get token receiver
        let (mut token_rx, failure, usage_slot) = match spawn_split_stream(
            &state,
            &model_id,
            &messages,
            params,
            &request_id,
            Some(stream_trace.clone()),
        ) {
            Some(pair) => pair,
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
        // With tools in play we cannot forward text as it arrives: whether the
        // output is a tool call is only knowable once it is complete, and having
        // streamed the raw JSON we could not retract it. So buffer, then emit
        // either tool_calls or the text at the end. Matches OpenAI, which does
        // not stream partial text for a tool call either.
        let mut buffered = String::new();
        loop {
            let event = tokio::select! {
                biased;
                // Local-complete fast path: cancel the instant the client drops,
                // not only on the next token's failed send (v0.3.16 fixed the
                // router path; this sibling was missed — a zero-token generation
                // would otherwise run to its natural end, ~55s, with nobody
                // listening).
                _ = tx.closed() => {
                    tracing::warn!(
                        token_count,
                        elapsed_ms = stream_start.elapsed().as_millis() as u64,
                        "DIAG: split stream client disconnected (connection closed) — cancelling decode"
                    );
                    return;
                }
                maybe = token_rx.recv() => match maybe {
                    Some(e) => e,
                    None => break,
                },
            };
            if let Some(fr) = &event.finish_reason {
                finish = fr.clone();
                break;
            }
            token_count += 1;
            if tools_requested {
                // Hold it back until we know whether this is a tool call.
                buffered.push_str(&event.text);
                continue;
            }
            // Stop the instant the consumer closes OR stops reading (a stalled
            // send past SSE_CONSUMER_STALL_TIMEOUT). Returning drops token_rx →
            // cancels the worker, bounding runaway compute for a client that
            // walked away without closing the connection (Finding 2).
            if !crate::api::sse_send_live(
                &tx,
                StreamEvent::Delta {
                    content: Some(event.text),
                    role: None,
                    finish_reason: None,
                },
            )
            .await
            {
                tracing::warn!(
                    token_count,
                    elapsed_ms = stream_start.elapsed().as_millis() as u64,
                    "DIAG: split stream consumer gone (closed or not reading) — cancelling decode"
                );
                return;
            }
        }

        // Flush what we withheld for tool inspection: either structured calls,
        // or the text unchanged if it turned out to be an ordinary answer (a
        // model given tools is free to just reply).
        if tools_requested && !buffered.is_empty() {
            if emit_openai_tool_calls(&tx, &buffered).await {
                finish = "tool_calls".to_string();
            } else {
                let _ = crate::api::sse_send_live(
                    &tx,
                    StreamEvent::Delta {
                        content: Some(buffered.clone()),
                        role: None,
                        finish_reason: None,
                    },
                )
                .await;
            }
        }

        // Zero tokens plus a recorded failure means generation never started —
        // tell the client why rather than closing a successful-looking empty
        // stream and leaving it to guess (external report 2026-07-25).
        if token_count == 0 {
            let reason = failure.lock().ok().and_then(|s| s.clone());
            if let Some(reason) = reason {
                // A proper SSE error frame, not text smuggled into content:
                // clients can distinguish a failure from a model that chose to
                // reply with that string.
                let _ = crate::api::sse_send_live(
                    &tx,
                    StreamEvent::Error {
                        message: reason.message,
                        error_type: reason.error_type,
                    },
                )
                .await;
                finish = "error".to_string();
            }
        }

        // Terminal usage chunk, matching what the router path already emits when
        // `stream_options.include_usage` is set. Without this the local
        // fast path reported {0,0,0} on an otherwise identical request.
        if include_usage {
            let counts = usage_slot.lock().ok().and_then(|s| *s);
            if let Some((prompt_tokens, completion_tokens)) = counts {
                let _ = crate::api::sse_send_live(
                    &tx,
                    StreamEvent::Usage {
                        prompt_tokens,
                        completion_tokens,
                    },
                )
                .await;
            }
        }

        tracing::info!(
            model_id = %model_id,
            token_count,
            finish = %finish,
            "DIAG: split stream decode loop complete (subprocess)"
        );

        // Publish before the terminal SSE frames: a client that has already
        // walked away must not cost us the record of work we actually did.
        let (p_tok, c_tok) = usage_slot
            .lock()
            .ok()
            .and_then(|s| *s)
            .unwrap_or((0, token_count as u32));
        stream_trace.mark_finished(
            if finish == "error" {
                crate::inference::trace::Outcome::Error("SplitStreamError".into())
            } else {
                crate::inference::trace::Outcome::Ok
            },
            p_tok,
            c_tok,
        );
        state.shared_state.publish_request_trace(&stream_trace);

        // Send finish (guarded so a stalled consumer can't hang this task after
        // the worker has already finished).
        if !crate::api::sse_send_live(
            &tx,
            StreamEvent::Delta {
                content: None,
                role: None,
                finish_reason: Some(finish),
            },
        )
        .await
        {
            tracing::debug!(token_count, "DIAG: split stream finish delta send failed");
        }
        if !crate::api::sse_send_live(&tx, StreamEvent::Done).await {
            tracing::debug!("DIAG: split stream Done send failed — client disconnected");
        }
        let elapsed = stream_start.elapsed();
        tracing::info!(
            elapsed_ms = elapsed.as_millis() as u64,
            token_count,
            "DIAG: split stream completed"
        );
    });

    stream_events_to_sse(rx, request_id, created, model_name, None, progress_handle)
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
    // Captured before `state` moves into the generation task below.
    let progress_handle = Some((
        state.shared_state.clone(),
        crate::api::request_uuid(&request_id),
    ));

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
                crate::log_failure!(e, error = %e, "DIAG: local stream generate_stream error");
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

    stream_events_to_sse(rx, request_id, created, model_name, None, progress_handle)
}

/// Convert a StreamEvent receiver into an SSE stream of OpenAI-format chat completion chunks.
///
/// `progress` is an optional (state, request uuid) pair used to interleave SSE
/// **comment** frames describing what a not-yet-streaming request is doing.
/// Reading a long prompt is ~99% of a long request, so without this the client
/// sees an idle socket for minutes and cannot tell work from a hang.
///
/// Comments are the right frame for it: `:`-prefixed lines are ignored by every
/// conforming SSE client, so this cannot corrupt an OpenAI-compatible response,
/// while a human watching `curl` sees the phase and ETA. It also subsumes the
/// keep-alive, which sent an empty comment on the same schedule for liveness
/// alone.
///
/// This sits inside the shared encoder rather than in the three callers on
/// purpose — a streaming path added later inherits it without its author
/// having to know (see `.claude/rules/architecture.md` § "One invariant, N
/// paths").
fn stream_events_to_sse(
    rx: tokio::sync::mpsc::Receiver<StreamEvent>,
    request_id: String,
    created: i64,
    model_name: String,
    session_id: Option<String>,
    progress: Option<(std::sync::Arc<crate::daemon::SharedState>, uuid::Uuid)>,
) -> Sse<impl futures::Stream<Item = Result<Event, Infallible>>> {
    let mut json_buf = Vec::with_capacity(512);
    // Set when the terminal frame is emitted, so the progress ticker below
    // knows to stop. `merge` ends only when BOTH sides end, so without this the
    // response would stay open forever after `[DONE]`.
    let finished = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let finished_for_map = finished.clone();
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
        StreamEvent::ToolCalls { calls } => {
            // One chunk carrying the whole tool-call set, with content null —
            // the shape a client expects alongside finish_reason "tool_calls".
            let chunk = ChatCompletionChunk {
                id: request_id.clone(),
                object: "chat.completion.chunk",
                created,
                model: model_name.clone(),
                choices: vec![ChunkChoice {
                    index: 0,
                    delta: Delta {
                        role: None,
                        content: None,
                        tool_calls: Some(calls),
                    },
                    finish_reason: None,
                    logprobs: None,
                }],
                session_id: None,
                usage: None,
            };
            Ok::<_, Infallible>(
                Event::default().data(serde_json::to_string(&chunk).unwrap_or_default()),
            )
        }
        StreamEvent::Error {
            message,
            error_type,
        } => {
            let error_json = serde_json::json!({
                "error": {
                    "message": message,
                    "type": error_type
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
        StreamEvent::Done => {
            finished_for_map.store(true, std::sync::atomic::Ordering::Relaxed);
            Ok(Event::default().data("[DONE]"))
        }
    });

    // Interleave progress comments with the token stream. A merged ticker
    // rather than a timeout on the receiver, because it has to keep firing
    // while the receiver is blocked — which is precisely the situation being
    // reported on.
    //
    // **The ticker MUST terminate.** `merge` ends only when both sides end, so
    // an unbounded ticker would hold the SSE response open forever after
    // `[DONE]`. It keys off a flag set by the mapper above rather than off the
    // trace's lifetime: not every streaming path registers a trace, and a
    // termination condition that depends on one would hang exactly the paths
    // that have none.
    let ticker = futures::stream::unfold((progress, finished), |(p, finished)| async move {
        tokio::time::sleep(std::time::Duration::from_secs(SSE_KEEPALIVE_INTERVAL_SECS)).await;
        if finished.load(std::sync::atomic::Ordering::Relaxed) {
            return None;
        }
        let text = p
            .as_ref()
            .and_then(|(state, rid)| state.active_traces.get(rid).and_then(|t| t.progress()))
            .map(|s| crate::api::sse::format_progress_comment(&s))
            // No snapshot yet, or already streaming: an empty comment is a
            // valid keep-alive, which is what this subsumes.
            .unwrap_or_default();
        Some((Ok(Event::default().comment(text)), (p, finished)))
    });

    Sse::new(StreamExt::merge(stream, ticker)).keep_alive(
        KeepAlive::new().interval(std::time::Duration::from_secs(SSE_KEEPALIVE_INTERVAL_SECS)),
    )
}
