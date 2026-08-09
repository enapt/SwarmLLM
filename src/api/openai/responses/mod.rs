//! OpenAI `/v1/responses` endpoint — request/response types, translation
//! to/from Chat Completions, and HTTP handler.
//!
//! Capabilities (all shipped, see `docs/plans/archive/responses_api{,_v2}.md` for
//! the per-milestone history):
//! - Local inference via translation to `/v1/chat/completions`, including
//!   function tools and `tool_choice`.
//! - Cloud-proxy verbatim path for OpenAI-routed models, plus an
//!   Anthropic-translated bridge for Claude models (delegates to
//!   `anthropic_bridge`).
//! - SSE streaming (`stream.rs`), with V1 first-token latency fix.
//! - Multimodal input parts (`input_image`, `input_file` UTF-8 only).
//! - redb persistence (`store=true`, the OpenAI default), with
//!   `previous_response_id` chaining and `GET /v1/responses/:id/input_items`
//!   pagination.
//! - `background=true` (M9) — non-streaming returns 202 + Location;
//!   `background=true && stream=true` (V8) returns SSE backed by a
//!   resumable replay buffer at `GET /v1/responses/:id?stream=true`.
//!
//! Built-in tools (`web_search`, `file_search`, `computer_use_preview`,
//! `code_interpreter`, `image_generation`, `mcp`, `custom`) are rejected
//! with 400 on the local path and forwarded verbatim on the cloud path —
//! they require backing infra SwarmLLM does not run.

pub mod anthropic_bridge;
pub mod background;
pub mod store;
pub mod stream;
pub mod translate;
pub mod types;

pub use types::*;

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use axum::body::to_bytes;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use dashmap::DashMap;

use crate::api::server::{AppState, JsonBody};
use crate::error::{ApiError, SwarmError};

/// Caps applied at the `/v1/responses` ingress so caller-controlled strings
/// don't reach cloud-proxy serializers (where they'd burn upstream quota or
/// pollute log lines), redb keys, or the translation layer with megabyte
/// payloads. Match the `validate_chat_request` shape so the local path keeps
/// the same overall budget after Responses → Chat translation.
const MAX_PREVIOUS_RESPONSE_ID_LEN: usize = 64;
const MAX_RESPONSES_INSTRUCTIONS_BYTES: usize = 2 * 1024 * 1024;
const MAX_RESPONSES_USER_BYTES: usize = 256;
const MAX_RESPONSES_MODEL_LEN: usize = 256;
const MAX_RESPONSES_SHORT_FIELD_LEN: usize = 64;
const MAX_RESPONSES_METADATA_BYTES: usize = 64 * 1024;
/// Cap on `#[serde(flatten)] extras` — the catch-all for unknown top-level
/// fields. Without this, a request with thousands of unknown keys (or one
/// huge value) is materialised into a `HashMap<String, Value>` *before* any
/// named-field validation runs and is then forwarded verbatim on the
/// cloud-proxy path. Keep both the count and per-value size bounded.
const MAX_RESPONSES_EXTRAS_COUNT: usize = 32;
const MAX_RESPONSES_EXTRA_VALUE_BYTES: usize = 4 * 1024;
/// Cap on the count of items in `ResponsesRequest.input`. Each item is a
/// fully-deserialised structure with its own `extras` map; without this,
/// `validate_responses_ingress` would have to walk an unbounded list to
/// enforce per-item caps, and a malicious client could ship millions of
/// near-empty items past the wire-byte limit.
const MAX_RESPONSES_INPUT_ITEMS: usize = 1024;
/// Cap on caller-supplied query strings on `GET /v1/responses/:id/input_items`
/// (`after`, `before`, `order`, `include`). Cursors are short by construction
/// (`item_N`); any megabyte-class value is hostile rather than a real cursor.
const MAX_INPUT_ITEMS_QUERY_LEN: usize = 64;
/// Default page size for `GET /v1/responses/:id/input_items`. Matches OpenAI.
const INPUT_ITEMS_DEFAULT_PAGE_SIZE: u32 = 20;
/// Hard cap on the `limit` query param for input-items pagination.
const INPUT_ITEMS_MAX_PAGE_SIZE: u32 = 100;

/// Validate a `id` path parameter on a `/v1/responses/{id}` route. Mirrors
/// the `previous_response_id` cap so caller-supplied identifiers never
/// exceed the size we're willing to look up, log, or reflect into 404
/// bodies and DashMap keys (`BACKGROUND_CANCEL`, `BACKGROUND_STATE`).
pub(crate) fn validate_response_id(id: &str) -> Result<(), ApiError> {
    if id.is_empty()
        || id.len() > MAX_PREVIOUS_RESPONSE_ID_LEN
        || !id
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
    {
        return Err(ApiError(SwarmError::Validation(
            "response id must be 1..=64 ASCII alphanumeric characters \
             (with `_` / `-`)"
                .into(),
        )));
    }
    Ok(())
}

/// Validate caller-supplied identifiers and bounded strings BEFORE the
/// cloud-proxy / Anthropic-bridge / local-inference branches. Each branch
/// otherwise has its own validation surface (or none) — running this once at
/// the top closes the gap.
fn validate_responses_ingress(req: &ResponsesRequest) -> Result<(), ApiError> {
    // Bound the `#[serde(flatten)] extras` catch-all FIRST — any other
    // named-field validation comes after the request has already been
    // deserialised, but extras are also memory pressure inside that
    // deserialisation. The Axum DefaultBodyLimit caps the wire bytes; this
    // caps the post-deserialisation cardinality and per-value size that
    // gets forwarded on the cloud-proxy path.
    // Cap the tool list, as Chat Completions and the Anthropic surface already
    // do. This was the one API surface with no bound at all: OpenAI enforces 128
    // tools per request, and each definition is injected into the prompt and
    // billed as input tokens, so an unbounded list is both a spec divergence and
    // a way to inflate a request without limit. Flagged by an external tester who
    // sent 33 tools expecting a rejection — 33 is under the cap so it correctly
    // passed, but nothing would have stopped 10,000.
    if let Some(ref tools) = req.tools {
        use super::responses::types::{ToolDef, TypedToolDef};
        crate::api::validate_tools(
            tools,
            |t| match t {
                ToolDef::Typed(TypedToolDef::Function { name, .. }) => Some(name.as_str()),
                ToolDef::Raw(v) => v.get("name").and_then(|x| x.as_str()),
            },
            |t| match t {
                ToolDef::Typed(TypedToolDef::Function { description, .. }) => {
                    description.as_deref()
                }
                ToolDef::Raw(v) => v.get("description").and_then(|x| x.as_str()),
            },
            |t| match t {
                ToolDef::Typed(TypedToolDef::Function { parameters, .. }) => {
                    parameters.as_ref().map(|p| p.to_string().len())
                }
                ToolDef::Raw(v) => v.get("parameters").map(|p| p.to_string().len()),
            },
        )?;
    }

    if req.extras.len() > MAX_RESPONSES_EXTRAS_COUNT {
        return Err(ApiError(SwarmError::Validation(format!(
            "too many unknown request fields ({} present, max {MAX_RESPONSES_EXTRAS_COUNT})",
            req.extras.len()
        ))));
    }
    for (k, v) in &req.extras {
        let value_len = v.to_string().len();
        if value_len > MAX_RESPONSES_EXTRA_VALUE_BYTES {
            return Err(ApiError(SwarmError::Validation(format!(
                "unknown field `{k}` value too large ({value_len} bytes, max {MAX_RESPONSES_EXTRA_VALUE_BYTES})"
            ))));
        }
    }
    if req.model.is_empty() || req.model.len() > MAX_RESPONSES_MODEL_LEN {
        return Err(ApiError(SwarmError::Validation(format!(
            "model must be 1..={MAX_RESPONSES_MODEL_LEN} bytes"
        ))));
    }
    if let Some(prev_id) = req.previous_response_id.as_deref() {
        validate_response_id(prev_id)?;
    }
    if let Some(instructions) = req.instructions.as_deref() {
        if instructions.len() > MAX_RESPONSES_INSTRUCTIONS_BYTES {
            return Err(ApiError(SwarmError::Validation(format!(
                "instructions too large ({} bytes, max {MAX_RESPONSES_INSTRUCTIONS_BYTES})",
                instructions.len()
            ))));
        }
    }
    if let Some(user) = req.user.as_deref() {
        if user.len() > MAX_RESPONSES_USER_BYTES {
            return Err(ApiError(SwarmError::Validation(format!(
                "user identifier too long (max {MAX_RESPONSES_USER_BYTES} bytes)"
            ))));
        }
    }
    for (name, value) in [
        ("truncation", req.truncation.as_deref()),
        ("service_tier", req.service_tier.as_deref()),
    ] {
        if let Some(s) = value {
            if s.len() > MAX_RESPONSES_SHORT_FIELD_LEN {
                return Err(ApiError(SwarmError::Validation(format!(
                    "{name} too long (max {MAX_RESPONSES_SHORT_FIELD_LEN} chars)"
                ))));
            }
        }
    }
    if let Some(metadata) = req.metadata.as_ref() {
        let total: usize = metadata
            .iter()
            .map(|(k, v)| k.len() + v.to_string().len())
            .sum();
        if total > MAX_RESPONSES_METADATA_BYTES {
            return Err(ApiError(SwarmError::Validation(format!(
                "metadata too large ({total} bytes, max {MAX_RESPONSES_METADATA_BYTES})"
            ))));
        }
    }
    // Bound `input` item count and per-message extras so a request with
    // thousands of message items (each carrying its own `#[serde(flatten)]`
    // extras map) can't bypass the top-level extras cap.
    if let crate::api::openai::responses::types::ResponsesInput::Items(items) = &req.input {
        if items.len() > MAX_RESPONSES_INPUT_ITEMS {
            return Err(ApiError(SwarmError::Validation(format!(
                "input has too many items ({} present, max {MAX_RESPONSES_INPUT_ITEMS})",
                items.len()
            ))));
        }
        for item in items {
            if let crate::api::openai::responses::types::InputItem::Typed(
                crate::api::openai::responses::types::TypedInputItem::Message(msg),
            ) = item
            {
                if msg.extras.len() > MAX_RESPONSES_EXTRAS_COUNT {
                    return Err(ApiError(SwarmError::Validation(format!(
                        "input message has too many unknown fields ({} present, max {MAX_RESPONSES_EXTRAS_COUNT})",
                        msg.extras.len()
                    ))));
                }
                for (k, v) in &msg.extras {
                    let value_len = v.to_string().len();
                    if value_len > MAX_RESPONSES_EXTRA_VALUE_BYTES {
                        return Err(ApiError(SwarmError::Validation(format!(
                            "input message field `{k}` value too large ({value_len} bytes, max {MAX_RESPONSES_EXTRA_VALUE_BYTES})"
                        ))));
                    }
                }
            }
        }
    }
    Ok(())
}

/// Tool `type` strings that map to OpenAI-hosted infrastructure SwarmLLM
/// does not run.
pub(crate) const BUILTIN_TOOL_TYPES: &[&str] = &[
    "web_search",
    "file_search",
    "computer_use_preview",
    "code_interpreter",
    "image_generation",
    "mcp",
    "custom",
];

/// Cap on the body size we'll buffer when forwarding an upstream HTTP
/// response (Chat Completions translation, Anthropic bridge, error envelope
/// stringification) into memory. 16 MiB is a generous bound — local
/// inference responses are normally well under 1 MiB; the cap exists to
/// prevent an unbounded internal allocation if something goes sideways.
/// Shared across the responses module so a future tuning is single-sourced.
pub(super) const MAX_UPSTREAM_BODY_BYTES: usize = 16 * 1024 * 1024;

/// Buffer a Chat Completions response body, parse it as JSON, and run
/// the Responses-shaped translation. Used by both `create_response`
/// (sync) and `run_background_inference` (background) which previously
/// hand-rolled three separate `match`/error sites each. Returns a
/// human-readable error string so each caller can wrap it into the
/// error type their control flow uses (`ApiError`, `ResponseError`,
/// etc.).
async fn buffer_and_translate_chat_response(
    body: axum::body::Body,
    req: &ResponsesRequest,
    response_id: &str,
    created_at: i64,
) -> Result<ResponsesResponse, String> {
    let bytes = to_bytes(body, MAX_UPSTREAM_BODY_BYTES)
        .await
        .map_err(|e| format!("buffer error: {e}"))?;
    let chat_value: serde_json::Value =
        serde_json::from_slice(&bytes).map_err(|e| format!("parse chat JSON: {e}"))?;
    translate::chat_response_to_responses(&chat_value, req, response_id, created_at)
        .map_err(|e| e.to_string())
}

/// In-flight map of background response ids to their cancel flags.
///
/// The cancel handler flips the flag; the background worker checks it at
/// completion time — if set, the worker discards its inference result
/// instead of overwriting the stored `cancelled` record. Real-time token-
/// level interruption is out of scope: the chat handler owns its own
/// cancellation surface (per-request, not per-response).
///
/// Keys are removed when the background worker finishes (whether it wrote
/// a completed result or was cancelled). A parallel `BACKGROUND_CANCEL_AGES`
/// map is used by `prune_stale_background_state` to evict entries whose
/// owning task was cancelled externally (e.g. process shutdown mid-flight)
/// before its cleanup path could run.
static BACKGROUND_CANCEL: std::sync::LazyLock<DashMap<String, Arc<AtomicBool>>> =
    std::sync::LazyLock::new(DashMap::new);

/// Insert times for `BACKGROUND_CANCEL` entries. Maintained by
/// `register_background_cancel` / `unregister_background_cancel`. The
/// hourly responses sweep evicts entries older than
/// `BACKGROUND_CANCEL_MAX_AGE_SECS` from both maps so a runaway leak
/// stays bounded under task-cancel-without-cleanup conditions.
pub(crate) static BACKGROUND_CANCEL_AGES: std::sync::LazyLock<DashMap<String, std::time::Instant>> =
    std::sync::LazyLock::new(DashMap::new);

/// Generous upper bound on background-inference duration. Anything older
/// is almost certainly a leak (the longest legitimate background run is
/// a few minutes; this leaves an order of magnitude headroom).
pub(crate) const BACKGROUND_CANCEL_MAX_AGE_SECS: u64 = 7200;

/// Insert into `BACKGROUND_CANCEL` and record the timestamp atomically so
/// the sweep has a consistent view. Replaces the bare
/// `BACKGROUND_CANCEL.insert` calls everywhere.
pub(crate) fn register_background_cancel(response_id: &str, flag: Arc<AtomicBool>) {
    BACKGROUND_CANCEL.insert(response_id.to_string(), flag);
    BACKGROUND_CANCEL_AGES.insert(response_id.to_string(), std::time::Instant::now());
}

/// Remove from both `BACKGROUND_CANCEL` and `BACKGROUND_CANCEL_AGES`.
/// Replaces the bare `BACKGROUND_CANCEL.remove` calls everywhere.
pub(crate) fn unregister_background_cancel(response_id: &str) {
    BACKGROUND_CANCEL.remove(response_id);
    BACKGROUND_CANCEL_AGES.remove(response_id);
}

/// Drop `BACKGROUND_CANCEL` + `BACKGROUND_CANCEL_AGES` +
/// `background::BACKGROUND_STATE` entries older than
/// `BACKGROUND_CANCEL_MAX_AGE_SECS`. Returns the count pruned. Intended
/// to be called by the hourly responses sweep.
pub(crate) fn prune_stale_background_state() -> usize {
    let now = std::time::Instant::now();
    let stale: Vec<String> = BACKGROUND_CANCEL_AGES
        .iter()
        .filter(|e| now.duration_since(*e.value()).as_secs() > BACKGROUND_CANCEL_MAX_AGE_SECS)
        .map(|e| e.key().clone())
        .collect();
    let count = stale.len();
    for id in stale {
        BACKGROUND_CANCEL.remove(&id);
        BACKGROUND_CANCEL_AGES.remove(&id);
        background::BACKGROUND_STATE.remove(&id);
    }
    count
}

/// Default `max_output_tokens` when the caller didn't specify one.
/// Single source of truth for the four response-skeleton sites — keep
/// in sync if upstream OpenAI changes their default.
pub(super) const DEFAULT_MAX_OUTPUT_TOKENS: u32 = 2048;
/// Default sampling temperature when the caller didn't specify one. Mirrors
/// `DEFAULT_MAX_OUTPUT_TOKENS` shape — single source of truth so the
/// response skeleton (which records what was used) and the translation
/// layer (which actually applies the value) can't drift.
pub(super) const DEFAULT_TEMPERATURE: f32 = 0.7;
/// Default top-p when the caller didn't specify one.
pub(super) const DEFAULT_TOP_P: f32 = 0.9;

/// Generate a fresh `resp_<32-hex>` response id. Single source of truth
/// for the prefix convention so a future change (e.g. namespace bump)
/// can't leave one path emitting the old prefix.
pub(super) fn new_response_id() -> String {
    format!("resp_{}", uuid::Uuid::new_v4().simple())
}

/// Generate a fresh `msg_<32-hex>` message id (used as `OutputItem.id`).
pub(super) fn new_message_id() -> String {
    format!("msg_{}", uuid::Uuid::new_v4().simple())
}

/// Build a `ResponsesResponse` skeleton from a request — used by every
/// path that needs to emit a response object before inference produces
/// any content. Status is parameterized: `InProgress` for the lifecycle
/// events at stream open, `Queued` for the M9/V8 placeholder seeded into
/// redb, etc. Callers mutate the result post-construction when they need
/// status-specific overrides (e.g. `background = Some(true)` for V8).
///
/// All `Option` fields are cloned from `req` so the response carries the
/// caller's exact configuration knobs back to them, which is what the
/// OpenAI Responses API contract requires.
pub(super) fn build_response_skeleton(
    req: &ResponsesRequest,
    response_id: &str,
    created_at: i64,
    status: ResponseStatus,
) -> ResponsesResponse {
    ResponsesResponse {
        id: response_id.into(),
        object: "response".into(),
        created_at,
        status,
        model: req.model.clone(),
        output: Vec::new(),
        output_text: None,
        usage: ResponsesUsage::default(),
        error: None,
        incomplete_details: None,
        previous_response_id: req.previous_response_id.clone(),
        instructions: req.instructions.clone(),
        tools: req.tools.clone(),
        tool_choice: req.tool_choice.clone(),
        parallel_tool_calls: req.parallel_tool_calls,
        temperature: Some(req.temperature.unwrap_or(DEFAULT_TEMPERATURE)),
        top_p: Some(req.top_p.unwrap_or(DEFAULT_TOP_P)),
        max_output_tokens: Some(req.max_output_tokens.unwrap_or(DEFAULT_MAX_OUTPUT_TOKENS)),
        truncation: req.truncation.clone(),
        metadata: req.metadata.clone(),
        user: req.user.clone(),
        reasoning: req.reasoning.clone(),
        text: req.text.clone(),
        modalities: req.modalities.clone(),
        service_tier: req.service_tier.clone(),
        background: req.background,
        extras: HashMap::new(),
    }
}

/// Walk a tools array and return the first built-in tool type encountered.
pub(crate) fn first_builtin_tool(tools: &[ToolDef]) -> Option<&'static str> {
    for t in tools {
        let kind = t.type_str()?;
        for &builtin in BUILTIN_TOOL_TYPES {
            if kind == builtin {
                return Some(builtin);
            }
        }
    }
    None
}

/// `POST /v1/responses`.
///
/// Routing order:
/// 1. Cloud proxy: if the model resolves to an OpenAI-compatible provider,
///    proxy the request body verbatim to the upstream `/responses`
///    endpoint. Built-in tools, streaming, background, reasoning effort,
///    text.verbosity, include[], previous_response_id, and any future
///    field all round-trip via `#[serde(flatten)] extras` and the upstream
///    handles them. Anthropic / subprocess providers return a clear 400
///    pointing the caller at `/v1/messages`.
/// 2. Local inference: built-in-tool gate, M6/M8/M9 stubs, then translate
///    to Chat Completions and run the local model.
pub async fn create_response(
    State(state): State<AppState>,
    headers: axum::http::HeaderMap,
    JsonBody(req): JsonBody<ResponsesRequest>,
) -> Result<Response, ApiError> {
    // Validate caller-supplied identifiers BEFORE any routing decision so the
    // cloud-proxy paths don't forward attacker-sized strings to upstream
    // providers (where they'd land in our log lines or burn quota).
    validate_responses_ingress(&req)?;
    // Counted here, alongside the chat and messages endpoints. The Responses
    // API is a third way to send a message to a model and was the only one not
    // counted, so a caller using it saw the dashboard's request total stay at
    // zero.
    crate::api::increment_requests_made(&state.shared_state);

    // ---- 1a. Cloud proxy passthrough (M5 / V3). ----
    // Serialize the request struct back to JSON so flatten-extras and any
    // unmodeled OpenAI knobs reach the upstream verbatim.
    let body_value = serde_json::to_value(&req).map_err(|e| {
        ApiError(SwarmError::Internal(format!(
            "Failed to serialize Responses request for cloud proxy (model={}): {e}",
            req.model
        )))
    })?;
    let stream = req.stream.unwrap_or(false);
    if let Some(response) =
        crate::api::providers::try_proxy_openai_responses(&state, &body_value, stream).await?
    {
        return Ok(response);
    }

    // ---- 1b. Anthropic-translated passthrough (V3). ----
    // When the model resolves to an Anthropic provider (or the
    // claude-subscription subprocess), translate the Responses request
    // to an Anthropic Messages request, forward, and translate back.
    // `try_proxy_anthropic_responses` returns Ok(None) when the model
    // isn't Anthropic, letting us fall through to local inference.
    if let Some(response) =
        anthropic_bridge::try_proxy_anthropic_responses(&state, &headers, &req).await?
    {
        return Ok(response);
    }

    // ---- 2. Local inference path. ----
    // Built-in tool gate. Built-ins require backing infra we don't run;
    // `function` tools translate through to Chat (M4).
    if let Some(tools) = req.tools.as_deref() {
        if let Some(builtin) = first_builtin_tool(tools) {
            return Err(ApiError(SwarmError::Validation(format!(
                "Built-in tool `{builtin}` is not supported by /v1/responses on \
                 this server. Only `function` tools are accepted; OpenAI-hosted \
                 tools (web_search, file_search, computer_use_preview, \
                 code_interpreter, image_generation, mcp, custom) require backing \
                 infrastructure SwarmLLM does not run."
            ))));
        }
    }

    // M8: load the prior record when the caller is chaining. Both the
    // streaming and non-streaming paths below pass it through to
    // translate::request_to_chat so the prior turn's messages prepend
    // to the current input. (Cloud proxy already returned above with
    // the field forwarded verbatim.)
    let prior = match req.previous_response_id.as_ref() {
        // Shape was validated at ingress (validate_responses_ingress); here we
        // only need the not-found path.
        Some(prev_id) => match store::load(&state.db, prev_id).map_err(ApiError)? {
            Some(record) => Some(record),
            None => {
                return Err(ApiError(SwarmError::Validation(format!(
                    "previous_response_id `{prev_id}` not found or expired. \
                     Either pass the prior turn's messages inline via `input` \
                     or re-run the original call with store=true (default)."
                ))));
            }
        },
        None => None,
    };

    // V8 (responses_api_v2): background=true && stream=true returns 202
    // Accepted + Location header pointing at the GET resume endpoint.
    // The server runs the inference internally and buffers SSE events
    // for subsequent GET calls with `?stream=true&starting_after={seq}`.
    if req.background.unwrap_or(false) && stream {
        return background::start_background_stream(state, headers, req, prior).await;
    }

    // Streaming (M6) — local-inference SSE. Cloud proxy already streamed
    // above if it matched.
    if stream {
        return stream::run_streaming(state, headers, req, prior).await;
    }

    // Background mode (M9, non-stream).
    if req.background.unwrap_or(false) {
        return start_background(state, headers, req, prior).await;
    }

    // Translate to a Chat Completions request and call the existing
    // handler. Translation failures bubble up as 400 via Validation.
    let chat_req = translate::request_to_chat(&req, prior.as_ref())?;

    let chat_response = crate::api::openai::chat_completions(
        State(state.clone()),
        headers.clone(),
        JsonBody(chat_req),
    )
    .await?;

    // 7. If the chat handler returned an error response, pass it through
    //    verbatim — error JSON has the same shape both APIs use.
    if !chat_response.status().is_success() {
        return Ok(chat_response);
    }

    // 8. Parse the chat response body and translate to a Responses shape.
    let (parts, body) = chat_response.into_parts();
    let response_id = new_response_id();
    let created_at = chrono::Utc::now().timestamp();
    let resp = buffer_and_translate_chat_response(body, &req, &response_id, created_at)
        .await
        .map_err(|msg| {
            ApiError(SwarmError::Internal(format!(
                "Chat→Responses translation failed (model={}): {msg}",
                req.model
            )))
        })?;

    // M7: persist the completed response when store=true (the OpenAI default).
    if req.store.unwrap_or(true) {
        let record = store::ResponsesRecord::new(
            req.clone(),
            resp.clone(),
            created_at,
            store::DEFAULT_TTL_SECS,
        );
        if let Err(e) = store::store(&state.db, &record) {
            // Failure to persist should not kill the response — log and
            // return the generated answer. Caller can retry a GET if they
            // care (they'll get 404 and know to pass the full turn inline).
            tracing::warn!(error = %e, id = %resp.id, "responses store failed");
        }
    }

    let mut out = (StatusCode::OK, Json(resp)).into_response();
    // Preserve any non-content headers the chat handler set (rate-limit
    // headers, custom auth echoes, etc.). Keep our own status.
    for (name, value) in parts.headers.iter() {
        if name == axum::http::header::CONTENT_TYPE || name == axum::http::header::CONTENT_LENGTH {
            continue;
        }
        out.headers_mut().insert(name.clone(), value.clone());
    }
    Ok(out)
}

/// Spawn a background inference task and return a queued placeholder
/// response immediately. The task runs the same translate → chat path as
/// the synchronous handler; the cancel flag is checked right before the
/// completed record is written so a cancel request mid-inference leaves
/// `status="cancelled"` as the final stored state.
async fn start_background(
    state: AppState,
    headers: axum::http::HeaderMap,
    req: ResponsesRequest,
    prior: Option<store::ResponsesRecord>,
) -> Result<Response, ApiError> {
    let response_id = new_response_id();
    let created_at = chrono::Utc::now().timestamp();

    let cancel_flag = Arc::new(AtomicBool::new(false));
    register_background_cancel(&response_id, cancel_flag.clone());

    // Seed redb with a queued placeholder so a GET before inference runs
    // returns meaningful state (id, model, queued).
    let mut queued =
        build_response_skeleton(&req, &response_id, created_at, ResponseStatus::Queued);
    queued.background = Some(true);
    let record = store::ResponsesRecord::new(
        req.clone(),
        queued.clone(),
        created_at,
        store::DEFAULT_TTL_SECS,
    );
    if let Err(e) = store::store(&state.db, &record) {
        unregister_background_cancel(&response_id);
        return Err(ApiError(e));
    }

    let state_bg = state.clone();
    let headers_bg = headers.clone();
    let req_bg = req.clone();
    let prior_bg = prior;
    let response_id_bg = response_id.clone();
    let flag_bg = cancel_flag.clone();
    // Pre-clone for the panic-cleanup branch so the inference future can
    // consume the originals without lifetime tangling.
    let state_panic = state.clone();
    let id_panic = response_id.clone();
    tokio::spawn(async move {
        // Wrap the inference in catch_unwind so a panic inside any of
        // the chained calls (translate, chat_completions, buffer/parse,
        // chat→responses translate) doesn't leak the BACKGROUND_CANCEL
        // entry AND doesn't strand the redb record at status=in_progress
        // forever. Without this guard, a polling client would never see
        // a terminal state — the V8 streaming path has the same guard.
        use futures::FutureExt;
        let outcome = std::panic::AssertUnwindSafe(run_background_inference(
            state_bg,
            headers_bg,
            req_bg,
            prior_bg,
            response_id_bg,
            created_at,
            flag_bg,
        ))
        .catch_unwind()
        .await;
        if outcome.is_err() {
            tracing::error!(
                response_id = %id_panic,
                "M9 background task panicked — writing failed terminal state"
            );
            // Best-effort: stamp a terminal `failed` record so polling
            // clients see closure instead of permanent in_progress.
            if let Ok(Some(mut rec)) = store::load(&state_panic.db, &id_panic) {
                rec.response.status = ResponseStatus::Failed;
                rec.response.error = Some(ResponseError::new(
                    "internal_error",
                    "background task panicked",
                ));
                if let Err(e) = store::store(&state_panic.db, &rec) {
                    tracing::error!(
                        response_id = %id_panic,
                        error = %e,
                        "Failed to persist panic-terminal record"
                    );
                }
            }
            unregister_background_cancel(&id_panic);
        }
    });

    // 202 Accepted — matches the V8 streaming path and the OpenAI
    // Responses spec for queued background work.
    Ok((StatusCode::ACCEPTED, Json(queued)).into_response())
}

/// The tokio task body for a background response. Runs translate →
/// chat_completions → translate-back, then writes the final record
/// (unless the cancel flag was flipped in the meantime). Errors are
/// captured into `status="failed"` with an `error` object so a polling
/// caller gets a stable shape.
/// Run inference for an M9 (non-stream) background `/v1/responses` request.
///
/// **Invariant — `BACKGROUND_STATE` is intentionally NOT populated here.**
/// Only the V8 streaming path (`background::register_background_stream`)
/// inserts into `BACKGROUND_STATE`; the M9 path uses `BACKGROUND_CANCEL`
/// alone for cancel signalling and persists progress through redb. If you
/// add `BACKGROUND_STATE` registration to this M9 path, you MUST also
/// call `background::deregister_background_stream` on every exit (success,
/// failure, cancel) — otherwise `list_responses` will accumulate stale
/// `live=true` entries forever.
async fn run_background_inference(
    state: AppState,
    headers: axum::http::HeaderMap,
    req: ResponsesRequest,
    prior: Option<store::ResponsesRecord>,
    response_id: String,
    created_at: i64,
    cancel_flag: Arc<AtomicBool>,
) {
    // Mark in_progress before running so pollers see the transition.
    if let Ok(Some(mut rec)) = store::load(&state.db, &response_id) {
        rec.response.status = ResponseStatus::InProgress;
        if let Err(e) = store::store(&state.db, &rec) {
            // Was discarded silently; pollers then saw stale `queued`
            // forever on a DB-write failure with no operator visibility.
            // Mirrors the warn already in place on the finalize-write path.
            tracing::warn!(
                error = %e,
                id = %response_id,
                "background in_progress store failed",
            );
        }
    }

    let finalize = |status: ResponseStatus,
                    output: Vec<OutputItem>,
                    output_text: Option<String>,
                    usage: ResponsesUsage,
                    error: Option<ResponseError>| {
        // Cancel-wins policy: if the flag flipped during inference, keep
        // the cancelled record that POST .../cancel wrote; drop our
        // finalized result on the floor.
        if cancel_flag.load(Ordering::SeqCst) {
            unregister_background_cancel(&response_id);
            return;
        }
        if let Ok(Some(mut rec)) = store::load(&state.db, &response_id) {
            rec.response.status = status;
            rec.response.output = output;
            rec.response.output_text = output_text;
            rec.response.usage = usage;
            rec.response.error = error;
            if let Err(e) = store::store(&state.db, &rec) {
                tracing::warn!(error = %e, id = %response_id, "background finalize store failed");
            }
        }
        unregister_background_cancel(&response_id);
    };

    let chat_req = match translate::request_to_chat(&req, prior.as_ref()) {
        Ok(c) => c,
        Err(e) => {
            finalize(
                ResponseStatus::Failed,
                Vec::new(),
                None,
                ResponsesUsage::default(),
                Some(ResponseError::new("invalid_request_error", e.to_string())),
            );
            return;
        }
    };

    let chat_response = match crate::api::openai::chat_completions(
        State(state.clone()),
        headers,
        JsonBody(chat_req),
    )
    .await
    {
        Ok(r) => r,
        Err(e) => {
            finalize(
                ResponseStatus::Failed,
                Vec::new(),
                None,
                ResponsesUsage::default(),
                Some(ResponseError::new("internal_error", e.0.to_string())),
            );
            return;
        }
    };

    if !chat_response.status().is_success() {
        let bytes = match to_bytes(chat_response.into_body(), MAX_UPSTREAM_BODY_BYTES).await {
            Ok(b) => b,
            Err(e) => {
                finalize(
                    ResponseStatus::Failed,
                    Vec::new(),
                    None,
                    ResponsesUsage::default(),
                    Some(ResponseError::new(
                        "internal_error",
                        format!("buffer error: {e}"),
                    )),
                );
                return;
            }
        };
        let msg = String::from_utf8_lossy(&bytes).to_string();
        finalize(
            ResponseStatus::Failed,
            Vec::new(),
            None,
            ResponsesUsage::default(),
            Some(ResponseError::new("upstream_error", msg)),
        );
        return;
    }

    let resp = match buffer_and_translate_chat_response(
        chat_response.into_body(),
        &req,
        &response_id,
        created_at,
    )
    .await
    {
        Ok(r) => r,
        Err(msg) => {
            finalize(
                ResponseStatus::Failed,
                Vec::new(),
                None,
                ResponsesUsage::default(),
                Some(ResponseError::new("internal_error", msg)),
            );
            return;
        }
    };

    finalize(resp.status, resp.output, resp.output_text, resp.usage, None);
}

/// `GET /v1/responses/:id` — retrieve a stored response.
pub async fn get_response(
    State(state): State<AppState>,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> Result<Response, ApiError> {
    validate_response_id(&id)?;
    match store::load(&state.db, &id).map_err(ApiError)? {
        Some(record) => Ok((StatusCode::OK, Json(record.response)).into_response()),
        // Unknown id is a resource lookup miss, not a request-shape problem,
        // so emit 404 (matches OpenAI's behavior on the same endpoint).
        None => Err(ApiError(SwarmError::NotFound(format!(
            "Response `{id}` not found or expired. Retention is 30 days; \
             pass store=false to opt out of persistence."
        )))),
    }
}

/// `POST /v1/responses/:id/cancel` — flip the cancel flag and mark the
/// stored record as `cancelled`. Idempotent: a second call is a no-op.
/// Returns the updated response. 404 if the id is unknown.
pub async fn cancel_response(
    State(state): State<AppState>,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> Result<Response, ApiError> {
    validate_response_id(&id)?;
    // Signal the in-flight task (if any) to drop its result.
    if let Some(entry) = BACKGROUND_CANCEL.get(&id) {
        entry.store(true, Ordering::SeqCst);
    }
    // V8: also wake any resume-stream listeners so the cancelled state
    // becomes visible to a connected GET caller without waiting for the
    // next event push.
    if let Some(bg) = background::lookup_background_state(&id) {
        bg.notify.notify_waiters();
    }

    match store::load(&state.db, &id).map_err(ApiError)? {
        Some(mut record) => {
            record.response.status = ResponseStatus::Cancelled;
            store::store(&state.db, &record).map_err(ApiError)?;
            Ok((StatusCode::OK, Json(record.response)).into_response())
        }
        None => Err(ApiError(SwarmError::NotFound(format!(
            "Response `{id}` not found or expired. Cancel only applies to \
             background responses with store=true (the default)."
        )))),
    }
}

/// Query parameters for `GET /v1/responses/:id/input_items`.
#[derive(Debug, serde::Deserialize)]
pub struct ListInputItemsParams {
    /// Cursor: return items after the one with this id.
    #[serde(default)]
    pub after: Option<String>,
    /// Page size. Defaults to 20 (matches OpenAI's default). Capped at 100.
    #[serde(default)]
    pub limit: Option<u32>,
    /// `"asc"` (default) returns items in the order they appeared in the
    /// original request; `"desc"` reverses.
    #[serde(default)]
    pub order: Option<String>,
    /// Forward-compat: OpenAI SDKs pass `before` for reverse-cursor
    /// pagination. We accept and ignore it for now (single-direction
    /// cursor is sufficient for the shapes our callers use).
    #[serde(default)]
    pub before: Option<String>,
    /// Forward-compat for `include[reasoning.encrypted_content]` etc.
    /// Currently unused on the local path (we don't generate reasoning
    /// input items; cloud-proxied responses are served from the verbatim
    /// stored body).
    #[serde(default)]
    pub include: Option<String>,
}

/// `GET /v1/responses/:id/input_items` — paginated list of the input
/// items that were sent as the request body. V4 (responses_api_v2):
/// small bookkeeping endpoint hit by OpenAI SDKs in retried-tool-call
/// flows.
///
/// Synthetic ids: input items don't carry stable ids on the wire, so we
/// emit `item_{n}` where `n` is the zero-based position in the original
/// request. A `Text` input produces a single synthetic `message` item.
/// Cursor (`after`) matches by the synthetic id.
pub async fn list_input_items(
    State(state): State<AppState>,
    axum::extract::Path(id): axum::extract::Path<String>,
    axum::extract::Query(params): axum::extract::Query<ListInputItemsParams>,
) -> Result<Response, ApiError> {
    validate_response_id(&id)?;
    for (name, value) in [
        ("after", params.after.as_deref()),
        ("before", params.before.as_deref()),
        ("order", params.order.as_deref()),
        ("include", params.include.as_deref()),
    ] {
        if let Some(s) = value {
            if s.len() > MAX_INPUT_ITEMS_QUERY_LEN {
                return Err(ApiError(SwarmError::Validation(format!(
                    "{name} parameter too long ({} bytes, max {MAX_INPUT_ITEMS_QUERY_LEN})",
                    s.len()
                ))));
            }
        }
    }
    let record = store::load(&state.db, &id)
        .map_err(ApiError)?
        .ok_or_else(|| {
            // Unknown id → 404; the request shape is fine.
            ApiError(SwarmError::NotFound(format!(
                "Response `{id}` not found or expired. Retention is 30 days; \
                 pass store=false to opt out of persistence."
            )))
        })?;

    let body = build_input_items_page(&record.request.input, &params);
    Ok((StatusCode::OK, Json(body)).into_response())
}

/// Pure pagination helper: turn the stored `ResponsesInput` into the
/// OpenAI list-object JSON shape. Separated from the handler so cursor
/// + limit + order logic can be unit-tested without the full AppState.
pub(crate) fn build_input_items_page(
    input: &ResponsesInput,
    params: &ListInputItemsParams,
) -> serde_json::Value {
    let mut items_with_ids: Vec<(String, serde_json::Value)> = match input {
        ResponsesInput::Text(s) => {
            let item_id = "item_0".to_string();
            let v = serde_json::json!({
                "id": item_id,
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": s}],
            });
            vec![(item_id, v)]
        }
        ResponsesInput::Items(items) => items
            .iter()
            .enumerate()
            .map(|(i, item)| {
                let item_id = format!("item_{i}");
                let mut v = serde_json::to_value(item).unwrap_or(serde_json::Value::Null);
                if let Some(obj) = v.as_object_mut() {
                    obj.insert("id".into(), serde_json::Value::String(item_id.clone()));
                }
                (item_id, v)
            })
            .collect(),
    };

    // `order=desc` reverses the stable list before cursoring so `after`
    // semantics still mean "the next page of results" regardless of
    // direction.
    if matches!(params.order.as_deref(), Some("desc")) {
        items_with_ids.reverse();
    }

    let start = match params.after.as_deref() {
        Some(cursor) => items_with_ids
            .iter()
            .position(|(i, _)| i == cursor)
            .map(|i| i + 1)
            .unwrap_or(items_with_ids.len()),
        None => 0,
    };

    let limit = params
        .limit
        .unwrap_or(INPUT_ITEMS_DEFAULT_PAGE_SIZE)
        .clamp(1, INPUT_ITEMS_MAX_PAGE_SIZE) as usize;
    let total = items_with_ids.len();
    let end = start.saturating_add(limit).min(total);
    let page: Vec<serde_json::Value> = items_with_ids[start..end]
        .iter()
        .map(|(_, v)| v.clone())
        .collect();

    let first_id = page
        .first()
        .and_then(|v| v.get("id").and_then(|x| x.as_str()))
        .map(String::from);
    let last_id = page
        .last()
        .and_then(|v| v.get("id").and_then(|x| x.as_str()))
        .map(String::from);
    let has_more = end < total;

    serde_json::json!({
        "object": "list",
        "data": page,
        "first_id": first_id,
        "last_id": last_id,
        "has_more": has_more,
    })
}

/// `DELETE /v1/responses/:id` — remove a stored response.
pub async fn delete_response(
    State(state): State<AppState>,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> Result<Response, ApiError> {
    validate_response_id(&id)?;
    // Check existence so we can return a meaningful success body.
    let existed = store::load(&state.db, &id).map_err(ApiError)?.is_some();
    store::delete(&state.db, &id).map_err(ApiError)?;
    let body = serde_json::json!({
        "id": id,
        "object": "response.deleted",
        "deleted": existed,
    });
    Ok((StatusCode::OK, Json(body)).into_response())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn tools_from(json_value: serde_json::Value) -> Vec<ToolDef> {
        serde_json::from_value(json_value).unwrap()
    }

    #[test]
    fn function_only_returns_none() {
        let tools = tools_from(json!([
            {"type": "function", "name": "f", "parameters": {"type": "object"}},
        ]));
        assert_eq!(first_builtin_tool(&tools), None);
    }

    #[test]
    fn detects_each_builtin_tool() {
        for &kind in BUILTIN_TOOL_TYPES {
            let tools = tools_from(json!([{"type": kind}]));
            assert_eq!(
                first_builtin_tool(&tools),
                Some(kind),
                "expected {kind} to be flagged"
            );
        }
    }

    #[test]
    fn detects_builtin_when_mixed_with_function() {
        let tools = tools_from(json!([
            {"type": "function", "name": "f", "parameters": {"type": "object"}},
            {"type": "web_search"},
            {"type": "function", "name": "g", "parameters": {"type": "object"}},
        ]));
        assert_eq!(first_builtin_tool(&tools), Some("web_search"));
    }

    // ------------------------------------------------------------------
    // V4: input_items pagination
    // ------------------------------------------------------------------

    fn empty_params() -> ListInputItemsParams {
        ListInputItemsParams {
            after: None,
            limit: None,
            order: None,
            before: None,
            include: None,
        }
    }

    #[test]
    fn input_items_text_input_produces_single_synthetic_message() {
        let input = ResponsesInput::Text("hi there".into());
        let page = build_input_items_page(&input, &empty_params());
        assert_eq!(page["object"], "list");
        assert_eq!(page["has_more"], false);
        let data = page["data"].as_array().unwrap();
        assert_eq!(data.len(), 1);
        assert_eq!(data[0]["id"], "item_0");
        assert_eq!(data[0]["type"], "message");
        assert_eq!(data[0]["role"], "user");
        assert_eq!(data[0]["content"][0]["type"], "input_text");
        assert_eq!(data[0]["content"][0]["text"], "hi there");
        assert_eq!(page["first_id"], "item_0");
        assert_eq!(page["last_id"], "item_0");
    }

    #[test]
    fn input_items_array_input_paginates_by_limit() {
        let input: ResponsesInput = serde_json::from_value(json!([
            {"type": "message", "role": "user", "content": "one"},
            {"type": "message", "role": "user", "content": "two"},
            {"type": "message", "role": "user", "content": "three"},
            {"type": "message", "role": "user", "content": "four"},
        ]))
        .unwrap();

        let mut params = empty_params();
        params.limit = Some(2);
        let page = build_input_items_page(&input, &params);
        assert_eq!(page["data"].as_array().unwrap().len(), 2);
        assert_eq!(page["first_id"], "item_0");
        assert_eq!(page["last_id"], "item_1");
        assert_eq!(page["has_more"], true);
    }

    #[test]
    fn input_items_after_cursor_returns_next_page() {
        let input: ResponsesInput = serde_json::from_value(json!([
            {"type": "message", "role": "user", "content": "one"},
            {"type": "message", "role": "user", "content": "two"},
            {"type": "message", "role": "user", "content": "three"},
            {"type": "message", "role": "user", "content": "four"},
        ]))
        .unwrap();

        let mut params = empty_params();
        params.limit = Some(2);
        params.after = Some("item_1".into());
        let page = build_input_items_page(&input, &params);
        let data = page["data"].as_array().unwrap();
        assert_eq!(data.len(), 2);
        assert_eq!(data[0]["id"], "item_2");
        assert_eq!(data[1]["id"], "item_3");
        assert_eq!(page["has_more"], false);
    }

    #[test]
    fn input_items_after_cursor_at_end_returns_empty_page() {
        let input: ResponsesInput = serde_json::from_value(json!([
            {"type": "message", "role": "user", "content": "only"},
        ]))
        .unwrap();
        let mut params = empty_params();
        params.after = Some("item_0".into());
        let page = build_input_items_page(&input, &params);
        assert!(page["data"].as_array().unwrap().is_empty());
        assert_eq!(page["has_more"], false);
        assert_eq!(page["first_id"], serde_json::Value::Null);
        assert_eq!(page["last_id"], serde_json::Value::Null);
    }

    #[test]
    fn input_items_order_desc_reverses_iteration() {
        let input: ResponsesInput = serde_json::from_value(json!([
            {"type": "message", "role": "user", "content": "a"},
            {"type": "message", "role": "user", "content": "b"},
            {"type": "message", "role": "user", "content": "c"},
        ]))
        .unwrap();
        let mut params = empty_params();
        params.order = Some("desc".into());
        let page = build_input_items_page(&input, &params);
        let data = page["data"].as_array().unwrap();
        // All three present because default limit=20 > 3.
        assert_eq!(data.len(), 3);
        // Items returned in reverse order (last first).
        assert_eq!(data[0]["id"], "item_2");
        assert_eq!(data[1]["id"], "item_1");
        assert_eq!(data[2]["id"], "item_0");
    }

    #[test]
    fn input_items_limit_capped_at_100() {
        let input = ResponsesInput::Text("hi".into());
        let mut params = empty_params();
        params.limit = Some(10_000);
        let page = build_input_items_page(&input, &params);
        // 1 item so only 1 returned, but the clamp path shouldn't panic.
        assert_eq!(page["data"].as_array().unwrap().len(), 1);
    }

    #[test]
    fn input_items_function_call_items_preserve_fields_and_add_id() {
        let input: ResponsesInput = serde_json::from_value(json!([
            {"type": "message", "role": "user", "content": "call X"},
            {"type": "function_call", "call_id": "c1", "name": "lookup", "arguments": "{\"q\":\"x\"}"},
            {"type": "function_call_output", "call_id": "c1", "output": "{\"result\":42}"},
        ]))
        .unwrap();
        let page = build_input_items_page(&input, &empty_params());
        let data = page["data"].as_array().unwrap();
        assert_eq!(data.len(), 3);
        assert_eq!(data[0]["id"], "item_0");
        assert_eq!(data[1]["id"], "item_1");
        assert_eq!(data[1]["type"], "function_call");
        assert_eq!(data[1]["name"], "lookup");
        assert_eq!(data[2]["id"], "item_2");
        assert_eq!(data[2]["type"], "function_call_output");
        assert_eq!(data[2]["call_id"], "c1");
    }

    #[test]
    fn unknown_tool_type_does_not_trigger_rejection() {
        // Future / unmodeled tool types round-trip via Raw and should be
        // forwarded (cloud proxy in M5) rather than 400ing here.
        let tools = tools_from(json!([{"type": "future_tool_type_2027"}]));
        assert_eq!(first_builtin_tool(&tools), None);
    }
}
