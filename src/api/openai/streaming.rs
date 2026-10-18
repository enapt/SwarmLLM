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
use crate::inference::router::RouterCommand;
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
    // Which computers served it — REQUIRED, so a new caller cannot build a
    // reply that leaves its route out of the body (#251 (3)).
    trace: Option<&crate::inference::trace::TraceSnapshot>,
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
    let (content, reasoning_content, tool_calls, finish_reason) = if tools_requested {
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
                // Content BEFORE the call is kept, not discarded. OpenAI's
                // wire format allows `content` and `tool_calls` in one message
                // and vLLM's reference parser does exactly this
                // (`content = model_output[:start]`, null only when empty) —
                // and it is what lets the streaming path emit that prose as it
                // is produced instead of holding the whole reply back. Dropping
                // it was silently throwing away text the model had produced.
                //
                // `finish_reason` must be "tool_calls" or clients never
                // dispatch them (the reported response said "length").
                let leading = crate::api::tool_parse::leading_content(&content).map(str::to_string);
                (leading, None, Some(calls), "tool_calls".to_string())
            }
            None => {
                let (content, reasoning) =
                    crate::api::tool_parse::split_unfinished_reasoning(content);
                (Some(content), reasoning, None, finish_reason)
            }
        }
    } else {
        // A reply that ended inside its scratchpad: the stream sent it as
        // `reasoning_content` and no content, so this must too — the same reply
        // may not answer differently for `stream` (see `ReasoningOut::Separate`).
        let (content, reasoning) = crate::api::tool_parse::split_unfinished_reasoning(content);
        (Some(content), reasoning, None, finish_reason)
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
                reasoning_content,
                tool_calls,
            },
            finish_reason,
            logprobs,
        }],
        usage: Usage::from_counts(prompt_tokens, completion_tokens),
        session_id,
        swarmllm: crate::api::SwarmllmReply::for_trace(trace),
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
    crate::inference::router::StreamingTokenRx,
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

/// Records that THIS computer serves the request alone, over the layers it
/// holds: the route of both local fast paths, which bypass the router and so
/// must say it themselves.
///
/// Called before generating, because a streamed reply reads its route when it
/// FINISHES (#251 (3)) and the reader may take the last token before anything
/// after the generate call runs. v0.3.243 shipped with only the unstreamed path
/// recording it: every streamed reply this node served alone said
/// `"segments": 0, "nodes": []` on OpenAI's finish chunk, Anthropic's
/// `message_delta` and Responses' `response.completed` (found live after the
/// release; FUTURE_WORK #306).
///
/// No scheduling happens on these paths, so 0 ms of it is the measurement, not
/// a placeholder.
fn mark_served_here(
    state: &crate::daemon::SharedState,
    trace: &crate::inference::trace::RequestTrace,
    layer_range: (u32, u32),
) {
    trace.mark_assembled(
        crate::inference::trace::Route::Local,
        crate::inference::trace::local_segment(state.identity.node_id(), layer_range),
        0,
    );
}

#[allow(clippy::too_many_arguments)]
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
    tools: Option<&[serde_json::Value]>,
    // The caller's instruction to keep the request on this computer (#251
    // (2b)) — the worker then saves its prompt's cache for reuse here only.
    keep: crate::types::KeepAsked,
) -> Option<SplitStream> {
    let meta = get_split_model_meta(&state.shared_state, model_id)?;
    let prompt = crate::inference::chat_template::build_prompt(
        messages,
        meta.chat_template.as_deref(),
        &meta.bos_token,
        &meta.eos_token_str,
        Some(model_id.0.as_str()),
        tools,
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
    let served = trace.unwrap_or_else(|| {
        std::sync::Arc::new(crate::inference::trace::RequestTrace::new(
            rid,
            model_id.0.clone(),
            "chat",
        ))
    });
    // Here, inside the one function both streamed surfaces call, so neither
    // caller can forget it.
    mark_served_here(&state.shared_state, &served, layer_range);
    let trace_guard = TraceGuard::register(&state.shared_state, rid, served);
    tokio::spawn(async move {
        let _trace_guard = trace_guard;
        let generate = pool.generate(
            &model_id,
            layer_range,
            prompt,
            params,
            rid,
            // Asked of this node's own API: the owner's request.
            crate::inference::process_pool::Requester::Owner,
            keep,
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
#[allow(clippy::too_many_arguments)]
pub async fn submit_stream_to_router(
    router_tx: &tokio::sync::mpsc::Sender<RouterCommand>,
    model_id: ModelId,
    messages: Vec<ChatMessage>,
    sampling_params: SamplingParams,
    session_id: Option<String>,
    lora_adapter: Option<String>,
    cancel: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
    tools: Option<Vec<serde_json::Value>>,
    // A caller's `swarm_route` block, already validated by the handler. Taken
    // as a parameter rather than re-derived here because this helper is shared
    // with the Anthropic surface, which has no such field and passes `None`.
    route_override: Option<crate::inference::route_override::RoutePlanOverride>,
    // The caller's instruction to keep the request on this computer (#251
    // (2b)) — required, so neither surface can forget to pass it on.
    keep: crate::types::KeepAsked,
) -> Result<
    (
        tokio::sync::oneshot::Receiver<
            Result<crate::inference::router::InferenceOutput, crate::error::SwarmError>,
        >,
        crate::inference::router::StreamingTokenRx,
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
        tools,
        keep,
    );
    inference_req.cancel = cancel;
    inference_req.route_override = route_override;
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
    tools: Option<&[serde_json::Value]>,
    // The caller's instruction to keep the request on this computer (#251
    // (2b)) — the worker then saves its prompt's cache for reuse here only.
    keep: crate::types::KeepAsked,
) -> Result<crate::inference::router::InferenceOutput, ApiError> {
    let meta = get_split_model_meta(&state.shared_state, model_id)
        .ok_or(ApiError(crate::error::SwarmError::NoModelLoaded))?;

    let prompt = chat_template::build_prompt(
        messages,
        meta.chat_template.as_deref(),
        &meta.bos_token,
        &meta.eos_token_str,
        Some(model_id.0.as_str()),
        tools,
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
    mark_served_here(&state.shared_state, &trace, meta.layer_range);

    let mut output = state
        .shared_state
        .model_process_pool
        .generate(
            model_id,
            meta.layer_range,
            prompt,
            params,
            rid,
            // Asked of this node's own API: the owner's request.
            crate::inference::process_pool::Requester::Owner,
            keep,
            None,
            None,
        )
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
        req.tool_definitions(),
        req.keep_asked(),
    );
    inference_req.cancel = cancel;
    inference_req.route_override = req.route_plan_override();

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
        trace.as_ref(),
    );

    Ok(crate::api::attach_route_headers(
        Json(response).into_response(),
        trace.as_ref(),
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
/// Flush what tool inspection withheld: the tool calls if the text turned out
/// to contain any, otherwise the text unchanged (a model given tools is free to
/// just reply).
///
/// It owns BOTH outcomes so no caller can flush one and forget the other, and
/// so that content preceding a call is emitted rather than discarded — the
/// non-streaming path does the same, and vLLM's reference parser does the same.
/// Feed one generated token through the reply buffer and send what it
/// released — the scratchpad first, then content, since a token that closes
/// the block can release both. The ONE per-token send on this surface, so the
/// router path and the local split path cannot stream reasoning differently.
/// `false` once the client is gone (closed, or not reading).
async fn push_and_send(
    tx: &tokio::sync::mpsc::Sender<StreamEvent>,
    buffered: &mut crate::api::tool_parse::StreamingToolText,
    token: &str,
) -> bool {
    let content = buffered.push(token);
    if let Some(text) = buffered.take_reasoning() {
        if !crate::api::sse_send_live(tx, StreamEvent::Reasoning { text }).await {
            return false;
        }
    }
    match content {
        Some(safe) => {
            crate::api::sse_send_live(
                tx,
                StreamEvent::Delta {
                    content: Some(safe),
                    role: None,
                },
            )
            .await
        }
        None => true,
    }
}

async fn emit_openai_tool_calls(
    tx: &tokio::sync::mpsc::Sender<StreamEvent>,
    buffered: &mut crate::api::tool_parse::StreamingToolText,
) -> bool {
    // A reply that never asked for tools is not searched for one: the buffer
    // still holds text back (a reasoning preamble it is part-way through), so
    // this must run — but parsing prose for a call could only ever produce a
    // false `finish_reason: "tool_calls"` on an ordinary answer.
    let parsed = if buffered.detects_tools() {
        crate::api::tool_parse::parse_tool_calls(buffered.reply_text())
    } else {
        None
    };
    // The scratchpad goes out before anything else the flush releases: the
    // tail `take_reasoning` held back, or the whole of a block the reply never
    // closed. Whether a call ends it decides where an unclosed block stops.
    if let Some(text) = buffered.finish_reasoning(parsed.is_some()) {
        let _ = crate::api::sse_send_live(tx, StreamEvent::Reasoning { text }).await;
    }
    if !buffered.detects_tools() {
        if let Some(text) = buffered.pending_all() {
            let _ = crate::api::sse_send_live(
                tx,
                StreamEvent::Delta {
                    content: Some(text),
                    role: None,
                },
            )
            .await;
        }
        return false;
    }
    // Whatever is left over: the prose before a call, or the whole reply when
    // there is none.
    let leftover = match parsed {
        Some(_) => buffered.pending_content(),
        None => buffered.pending_all(),
    };
    if let Some(text) = leftover {
        let _ = crate::api::sse_send_live(
            tx,
            StreamEvent::Delta {
                content: Some(text),
                role: None,
            },
        )
        .await;
    }
    let Some(parsed) = parsed else {
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
    // Every streamed request gets a cancel flag, the caller's or one of its
    // own, because the flag is what a long wait inside the pipeline watches
    // (`inference::cancel`). Dropping `token_rx` on disconnect, which this path
    // has always done, reaches the pipeline only when it next has a token to
    // send — never during the prompt pass, which on a processor can outlast
    // any client (gotcha #445).
    let cancel =
        cancel.unwrap_or_else(|| std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)));
    let (result_rx, mut token_rx, traced_id) = submit_stream_to_router(
        &router_tx,
        ModelId(req.model.clone()),
        messages,
        req.to_sampling_params(),
        req.session_id.clone(),
        req.lora_adapter.clone(),
        Some(cancel.clone()),
        req.tool_definitions(),
        req.route_plan_override(),
        req.keep_asked(),
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
        // Text held back for tool inspection — but only the part that could
        // still BE a tool call. See `tool_parse::content_prefix_len`.
        let mut buffered = crate::api::tool_parse::StreamingToolText::new(
            tools_requested,
            crate::api::tool_parse::ReasoningOut::Separate,
        );
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
                    cancel.store(true, std::sync::atomic::Ordering::Release);
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
                    // Through the buffer whether or not tools were asked
                    // for: it is also the only thing that removes a reasoning
                    // model's `<think>` preamble from a streamed reply
                    // (report #032).
                    if !push_and_send(&sse_tx, &mut buffered, &event.text).await {
                        tracing::warn!(
                            token_count,
                            "DIAG: SSE final text delta send failed — client disconnected"
                        );
                    }
                }
                // Flush what tool inspection withheld before the finish delta,
                // so finish_reason can reflect a tool call.
                let mut reason = reason.clone();
                // Unconditional: the buffer withholds text whether or not
                // tools were asked for, and the flush helper declines to look
                // for a call when they were not.
                if !buffered.is_empty() && emit_openai_tool_calls(&sse_tx, &mut buffered).await {
                    reason = "tool_calls".to_string();
                }
                // The route of the trace the router attached to the token
                // sender, read now that the reply is over (#251 (3)).
                if sse_tx
                    .send(StreamEvent::Finish {
                        reason: reason.clone(),
                        route: token_rx.route(),
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
                // Through the buffer whether or not tools were asked for —
                // see the finish branch above (report #032).
                // Closed OR stalled (non-reading) consumer → cancel the pipeline.
                if !push_and_send(&sse_tx, &mut buffered, &event.text).await {
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
                            })
                            .await
                            .is_err()
                    {
                        tracing::warn!(
                            "DIAG: SSE fallback content send failed — client disconnected"
                        );
                    }
                    if sse_tx
                        .send(StreamEvent::Finish {
                            reason: output.finish_reason.clone(),
                            route: output.trace.as_ref().map(|t| t.route_summary()),
                        })
                        .await
                        .is_err()
                    {
                        tracing::debug!(finish_reason = %output.finish_reason, "DIAG: SSE fallback finish send failed");
                    }
                }
                Ok(Err(e)) => {
                    tracing::debug!(error = %e, "DIAG: SSE fallback got pipeline error");
                    if sse_tx.send(StreamEvent::from_error(&e)).await.is_err() {
                        tracing::debug!("DIAG: SSE error event send failed — client disconnected");
                    }
                }
                Err(_) => {
                    // The pipeline task died without reporting. That is a fault
                    // of this node, and it is reported as one — not as the
                    // model choosing to stop, which is what this arm used to
                    // send (gotcha #433).
                    let died = crate::error::SwarmError::Internal(
                        "the inference pipeline stopped without reporting a result".into(),
                    );
                    tracing::warn!(
                        "DIAG: SSE result_rx channel dropped — pipeline task died without sending result"
                    );
                    if sse_tx.send(StreamEvent::from_error(&died)).await.is_err() {
                        tracing::debug!("DIAG: SSE channel-drop error send also failed");
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
    // The tool DEFINITIONS, not a flag derived from them: the prompt builder
    // needs them to decide whether this model's own template renders tools or
    // whether it must be told about them in prose. `tools_requested` is derived
    // below rather than passed beside them, so the two cannot disagree.
    tools: Option<Vec<serde_json::Value>>,
    keep: crate::types::KeepAsked,
) -> Result<axum::response::Response, ApiError> {
    let tools_requested = tools.as_ref().is_some_and(|t| !t.is_empty());
    let output = run_split_generate(
        &state,
        &model_id,
        &messages,
        params.clone(),
        &request_id,
        tools.as_deref(),
        keep,
    )
    .await?;
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
        trace.as_ref(),
    );

    Ok(crate::api::attach_route_headers(
        Json(response).into_response(),
        trace.as_ref(),
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
    // The tool DEFINITIONS, not a flag derived from them: the prompt builder
    // needs them to decide whether this model's own template renders tools or
    // whether it must be told about them in prose. `tools_requested` is derived
    // below rather than passed beside them, so the two cannot disagree.
    tools: Option<Vec<serde_json::Value>>,
    include_usage: bool,
    keep: crate::types::KeepAsked,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let tools_requested = tools.as_ref().is_some_and(|t| !t.is_empty());
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
            tools.as_deref(),
            keep,
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
        let mut buffered = crate::api::tool_parse::StreamingToolText::new(
            tools_requested,
            crate::api::tool_parse::ReasoningOut::Separate,
        );
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
                // Not `break` yet. The producer writes the token counts to the
                // usage slot only after its `generate` returns — which is AFTER
                // it sent this event — and the channel closes when that task
                // ends. Draining to the close is what guarantees the counts
                // are there when the trace and the usage chunk read them;
                // read a moment earlier, `prompt_tokens` was 0 on every
                // streamed request without `include_usage` (FUTURE_WORK,
                // 2026-08-30). Bounded: a producer that does not end promptly
                // must not hold the client's finish frame.
                let _ = tokio::time::timeout(std::time::Duration::from_secs(2), async {
                    while token_rx.recv().await.is_some() {}
                })
                .await;
                break;
            }
            token_count += 1;
            // With tools in play the buffer holds back only what could still
            // BE a tool call; without them, only a reasoning preamble. Either
            // way every token goes through it — this gate used to skip the
            // buffer entirely, so an ordinary chat message streamed the model's
            // whole `<think>` scratchpad to the user (report #032).
            // Stop the instant the consumer closes OR stops reading (a stalled
            // send past SSE_CONSUMER_STALL_TIMEOUT). Returning drops token_rx →
            // cancels the worker, bounding runaway compute for a client that
            // walked away without closing the connection (Finding 2).
            if !push_and_send(&tx, &mut buffered, &event.text).await {
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
        // Unconditional — see the sibling flush in the router path.
        if !buffered.is_empty() && emit_openai_tool_calls(&tx, &mut buffered).await {
            finish = "tool_calls".to_string();
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
                    // No hint: this failure arrived already flattened to
                    // message+type (a peer's, or one recorded earlier), so
                    // there is no `SwarmError` to ask for advice about. Better
                    // an honest absence than a guess from prose.
                    StreamEvent::Error {
                        message: reason.message,
                        error_type: reason.error_type,
                        hint: None,
                        hint_key: None,
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
            StreamEvent::Finish {
                reason: finish,
                route: Some(stream_trace.snapshot().route_summary()),
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
        let result = crate::inference::executor::without_starving_the_runtime(|| {
            executor.generate_stream(&prompt, &params, |token| {
                token_count += 1;
                // Waits for room rather than ending the reply on a full buffer —
                // the log line here used to say "channel full or client
                // disconnected" and stop for either.
                let delivered = crate::api::sse_send_live_blocking(
                    &tx,
                    StreamEvent::Delta {
                        content: Some(token.to_string()),
                        role: None,
                    },
                );
                if !delivered {
                    tracing::warn!(
                    token_count,
                    "DIAG: local stream token send failed — consumer gone or stalled past the limit"
                );
                }
                delivered
            })
        });

        // A failure is reported as one. This arm used to map every execution
        // error to `finish_reason: "stop"`, with a comment asserting that the
        // caller surfaced the error some other way — nothing did, and a
        // streamed request for a model no node held answered `200`, an empty
        // delta and `stop` where its non-streaming sibling answered 503 with a
        // hint (gotcha #433). `finish_reason` carries only what the OpenAI spec
        // defines, so an error is an `error` frame and no finish at all.
        match result {
            Ok(r) => {
                let finish = r.finish_reason.as_str().to_string();
                // The in-process executor, with no router: no trace, so no
                // route — its non-streamed sibling refuses rather than answers.
                if tx
                    .send(StreamEvent::Finish {
                        reason: finish.clone(),
                        route: None,
                    })
                    .await
                    .is_err()
                {
                    tracing::debug!(token_count, finish_reason = %finish, "DIAG: local stream finish send failed");
                }
            }
            Err(ref e) => {
                crate::log_failure!(e, error = %e, "DIAG: local stream generate_stream error");
                if tx.send(StreamEvent::from_error(e)).await.is_err() {
                    tracing::debug!(token_count, "DIAG: local stream error send failed");
                }
            }
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
    // `watch` rather than an `AtomicBool`: the ticker has to WAIT on this, not
    // merely read it — see `api::sse::progress_ticker`.
    let (finished_tx, finished_rx) = tokio::sync::watch::channel(false);
    // Every event below is a `data:` event; the ticker sends OpenAI's opening
    // chunk only after a stretch with none — see `api::sse::IdleKeepAlive`.
    let activity = crate::api::sse::DataActivity::new();
    let idle = crate::api::sse::IdleKeepAlive {
        activity: activity.clone(),
        event: {
            let (id, model) = (request_id.clone(), model_name.clone());
            Box::new(move || {
                let chunk = ChatCompletionChunk {
                    id: id.clone(),
                    object: "chat.completion.chunk",
                    created,
                    model: model.clone(),
                    choices: vec![ChunkChoice {
                        index: 0,
                        delta: Delta {
                            role: Some("assistant".into()),
                            content: Some(String::new()),
                            reasoning_content: None,
                            tool_calls: None,
                        },
                        finish_reason: None,
                        logprobs: None,
                    }],
                    session_id: None,
                    usage: None,
                    swarmllm: None,
                };
                Event::default().data(serde_json::to_string(&chunk).unwrap_or_default())
            })
        },
    };
    let stream =
        futures::StreamExt::inspect(tokio_stream::wrappers::ReceiverStream::new(rx), move |_| {
            activity.note()
        })
        .map(move |event| match event {
            StreamEvent::Delta { content, role } => {
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
                            reasoning_content: None,
                            tool_calls: None,
                        },
                        finish_reason: None,
                        logprobs: None,
                    }],
                    session_id: None,
                    usage: None,
                    swarmllm: None,
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
            // The finish chunk: the reply's last word on itself — the session
            // to send back, and which computers served it (#251 (3)).
            StreamEvent::Finish { reason, route } => {
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
                            reasoning_content: None,
                            tool_calls: None,
                        },
                        finish_reason: Some(reason),
                        logprobs: None,
                    }],
                    session_id: session_id.clone(),
                    usage: None,
                    swarmllm: crate::api::SwarmllmReply::for_route(route),
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
                            reasoning_content: None,
                            tool_calls: Some(calls),
                        },
                        finish_reason: None,
                        logprobs: None,
                    }],
                    session_id: None,
                    usage: None,
                    swarmllm: None,
                };
                Ok::<_, Infallible>(
                    Event::default().data(serde_json::to_string(&chunk).unwrap_or_default()),
                )
            }
            StreamEvent::Reasoning { text } => {
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
                            reasoning_content: Some(text),
                            tool_calls: None,
                        },
                        finish_reason: None,
                        logprobs: None,
                    }],
                    session_id: None,
                    usage: None,
                    swarmllm: None,
                };
                Ok::<_, Infallible>(
                    Event::default().data(serde_json::to_string(&chunk).unwrap_or_default()),
                )
            }
            StreamEvent::Error {
                message,
                error_type,
                hint,
                hint_key,
            } => {
                let mut error_obj = serde_json::json!({
                    "message": message,
                    "type": error_type
                });
                // Same two additive fields, spelled the same way, as the
                // non-streaming envelope in `error.rs` — `hint` is English prose
                // for any client, `hint_key` is what the dashboard looks up to say
                // it in the reader's own language. Omitted entirely when there is
                // no advice, rather than sent as null.
                if let Some(h) = hint {
                    error_obj["hint"] = serde_json::Value::String(h.to_string());
                }
                if let Some(k) = hint_key {
                    error_obj["hint_key"] = serde_json::Value::String(k.to_string());
                }
                let error_json = serde_json::json!({ "error": error_obj });
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
                    swarmllm: None,
                };
                let json = serde_json::to_string(&chunk).unwrap_or_default();
                Ok(Event::default().data(json))
            }
            StreamEvent::Done => {
                let _ = finished_tx.send(true);
                Ok(Event::default().data("[DONE]"))
            }
        });

    // Interleave progress comments with the token stream. A merged ticker
    // rather than a timeout on the receiver, because it has to keep firing
    // while the receiver is blocked — which is precisely the situation being
    // reported on.
    //
    // **The ticker MUST terminate, and terminate PROMPTLY.** `merge` ends only
    // when both sides end, so an unbounded ticker holds the SSE response open
    // for ever — and one that only checks after a full interval holds it open
    // for the rest of that interval, every time. Both live in
    // `api::sse::progress_ticker`, shared with the Anthropic encoder, which had
    // a byte-identical copy of this and the same defect.
    let ticker = crate::api::sse::progress_ticker(
        progress,
        finished_rx,
        std::time::Duration::from_secs(SSE_KEEPALIVE_INTERVAL_SECS),
        Some(idle),
    );

    Sse::new(StreamExt::merge(stream, ticker)).keep_alive(
        KeepAlive::new().interval(std::time::Duration::from_secs(SSE_KEEPALIVE_INTERVAL_SECS)),
    )
}

#[cfg(test)]
mod route_in_body_tests {
    use super::*;
    use crate::api::route_test_support::{fake_router, sse_payloads, test_app, two_computer_trace};

    fn chat_request() -> ChatCompletionRequest {
        serde_json::from_value(serde_json::json!({
            "model": "m",
            "stream": true,
            "messages": [{"role": "user", "content": "hi"}],
        }))
        .unwrap()
    }

    /// **A streamed chat reply names the computers that served it** (#251
    /// (3)). It has no `x-swarm-*` headers — they flush before a route exists
    /// — so the finish chunk carries `swarmllm.route`, read from the trace the
    /// ROUTER attached to the token sender: the API code never sees that trace
    /// any other way. Driven through the real stream task and encoder.
    #[tokio::test]
    async fn a_streamed_chat_reply_names_its_route_on_the_finish_chunk() {
        let app = test_app();
        let response = router_inference_stream(
            &app.state,
            fake_router("hello"),
            &chat_request(),
            Vec::new(),
            "swarm-test".into(),
            0,
            None,
        )
        .await
        .unwrap_or_else(|e| panic!("{}", e.0));
        let chunks: Vec<serde_json::Value> = sse_payloads(response)
            .await
            .into_iter()
            .filter(|p| p != "[DONE]")
            .map(|p| serde_json::from_str(&p).unwrap())
            .collect();
        let finish: Vec<_> = chunks
            .iter()
            .filter(|c| c["choices"][0]["finish_reason"].is_string())
            .collect();
        assert_eq!(finish.len(), 1, "one finish chunk: {chunks:?}");
        let route = &finish[0]["swarmllm"]["route"];
        assert_eq!(route["route"], "distributed", "{chunks:?}");
        assert_eq!(route["segments"], 2);
        assert_eq!(route["peers"], 1);
        assert_eq!(route["nodes"].as_array().map(Vec::len), Some(2));
        assert_eq!(
            chunks
                .iter()
                .filter(|c| c.get("swarmllm").is_some())
                .count(),
            1,
            "only the finish chunk carries it: {chunks:?}"
        );
    }

    /// **A reply streamed by a model this computer holds whole names this
    /// computer** — the local fast path, which bypasses the router and so
    /// records its own route. v0.3.243 left it unrecorded: the finish chunk
    /// (and Anthropic's `message_delta`, same function) said `"segments": 0,
    /// "nodes": []`. The test above drives a FAKE ROUTER, so it never reached
    /// this path.
    ///
    /// Read straight after `spawn_split_stream` returns: on this single-thread
    /// runtime its generation task has not started (the test never yields), so
    /// the route cannot have come from anything after the generate call, and
    /// no worker is spawned.
    #[tokio::test]
    async fn a_reply_streamed_by_a_model_held_here_names_this_computer() {
        let app = test_app();
        let model = ModelId("m".into());
        app.state.shared_state.split_models.insert(
            (model.clone(), 0, 22),
            crate::inference::split::SplitModelEntry {
                last_used: std::sync::atomic::AtomicU64::new(0),
                estimated_vram_mb: 0,
                is_complete: true,
                eos_tokens: vec![],
                eos_token_str: String::new(),
                bos_token: String::new(),
                cached_chat_template: None,
                vocab: None,
                layer_start: 0,
                layer_end: 22,
            },
        );
        let rid = crate::api::request_uuid("swarm-local");
        let trace =
            std::sync::Arc::new(crate::inference::trace::RequestTrace::new(rid, "m", "chat"));
        let messages: Vec<ChatMessage> =
            vec![
                serde_json::from_value(serde_json::json!({"role": "user", "content": "hi"}))
                    .unwrap(),
            ];
        let spawned = spawn_split_stream(
            &app.state,
            &model,
            &messages,
            SamplingParams::default(),
            "swarm-local",
            Some(trace.clone()),
            None,
            crate::types::KeepAsked::Unasked,
        );
        assert!(spawned.is_some(), "the model is held whole here");
        let route = trace.snapshot().route_summary();
        let me = trace_short_id(app.state.shared_state.identity.node_id());
        assert_eq!(route.route, "local");
        assert_eq!(route.segments, 1, "{route:?}");
        assert_eq!(route.peers, 0, "{route:?}");
        assert_eq!(route.nodes, vec![me], "{route:?}");
    }

    /// The short id a route names a computer by.
    fn trace_short_id(node: &crate::types::NodeId) -> String {
        crate::inference::trace::local_segment(node, (0, 1))[0]
            .short_node()
            .to_string()
    }

    /// **A streamed Responses reply names the route of the chat reply under
    /// it** — in the response object of `response.completed`, which is also
    /// what is stored and read back. Its stream is built on the chat stream, so
    /// this drives the two together.
    #[tokio::test]
    async fn a_streamed_response_names_the_route_of_the_chat_reply_under_it() {
        let app = test_app();
        let state = app.state.clone();
        let chat_future = async move {
            router_inference_stream(
                &state,
                fake_router("hello"),
                &chat_request(),
                Vec::new(),
                "swarm-test".into(),
                0,
                None,
            )
            .await
        };
        let req: crate::api::openai::responses::types::ResponsesRequest =
            serde_json::from_value(serde_json::json!({"model": "m", "input": "hi"})).unwrap();
        let events: Vec<_> = futures::StreamExt::collect(
            crate::api::openai::responses::stream::run_streaming_buffered(
                chat_future,
                req,
                "resp_x".into(),
                "msg_x".into(),
                0,
                None,
            ),
        )
        .await;
        let completed = events
            .iter()
            .find(|e| e.event_name == "response.completed")
            .expect("the response completes");
        let route = &completed.data["response"]["swarmllm"]["route"];
        assert_eq!(route["route"], "distributed", "{:?}", completed.data);
        assert_eq!(route["peers"], 1);
    }

    /// A chat completion that is not streamed carries the same object in its
    /// body as well as the headers, so one parser reads both forms; a reply
    /// with no swarm route leaves it out rather than guessing.
    #[test]
    fn an_unstreamed_chat_reply_carries_the_route_in_its_body_too() {
        let snapshot = two_computer_trace(uuid::Uuid::nil()).snapshot();
        let build = |trace| {
            serde_json::to_value(build_chat_completion_response(
                "id".into(),
                0,
                "m".into(),
                "hi".into(),
                "stop".into(),
                1,
                1,
                None,
                &[],
                false,
                trace,
            ))
            .unwrap()
        };
        let served = build(Some(&snapshot));
        assert_eq!(served["swarmllm"]["route"]["segments"], 2);
        assert_eq!(
            served["swarmllm"]["route"],
            serde_json::to_value(snapshot.route_summary()).unwrap()
        );
        assert!(build(None).get("swarmllm").is_none());
    }
}
