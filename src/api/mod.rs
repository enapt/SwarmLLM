/// Scrub API keys from an error body and truncate to 512 chars (char-boundary safe).
pub(crate) fn scrub_truncate_error(body: &str) -> String {
    let scrubbed = crate::crypto::scrub_api_keys(body);
    if scrubbed.len() > 512 {
        let mut idx = 512;
        while !scrubbed.is_char_boundary(idx) {
            idx -= 1;
        }
        format!("{}…[truncated]", &scrubbed[..idx])
    } else {
        scrubbed
    }
}

/// Strip the `Bearer` scheme from an `Authorization` header value.
///
/// **The scheme is case-INSENSITIVE** (RFC 7235 §2.1: "the scheme is
/// matched case-insensitively"), and several HTTP clients emit it lowercase.
/// A plain `strip_prefix("Bearer ")` rejected `bearer <key>` with a bare 401,
/// which is indistinguishable from a wrong key — so the user's next move is to
/// go hunting for a credential problem that does not exist. Reported by a
/// tester 2026-07-29; the header *name* was already case-insensitive (axum
/// normalizes it), which made the failure look arbitrary.
///
/// This is the single place the scheme is parsed. Do not re-derive it with
/// `strip_prefix` at a call site: six sites had their own copy, and one of
/// them accepting a form the others reject is worse than all of them being
/// strict.
pub(crate) fn strip_bearer_scheme(value: &str) -> Option<&str> {
    const SCHEME: &str = "bearer";
    let (scheme, rest) = value.split_at_checked(SCHEME.len())?;
    if !scheme.eq_ignore_ascii_case(SCHEME) {
        return None;
    }
    // RFC 7235 requires whitespace between the scheme and the token, so a
    // header like `Bearerxyz` must NOT be accepted as the token `xyz`.
    Some(rest.strip_prefix(' ')?.trim_start_matches(' '))
}

/// Extract a Bearer token from `Authorization: Bearer <tok>` or `x-api-key` header.
/// Returns an empty string if neither header is present.
pub(crate) fn extract_bearer_token(headers: &axum::http::HeaderMap) -> &str {
    headers
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(strip_bearer_scheme)
        .or_else(|| headers.get("x-api-key").and_then(|v| v.to_str().ok()))
        .unwrap_or("")
}

/// Strip `provider:` prefix from a model name, returning the bare model name.
pub(crate) fn strip_provider_prefix(model: &str) -> &str {
    model.split_once(':').map_or(model, |(_, name)| name)
}

// Shared validation limits for API request parameters.
// Used by both openai.rs and anthropic.rs handlers.
pub(crate) const MAX_TOOLS: usize = 128;
pub(crate) const MAX_TOOL_NAME_LEN: usize = 256;
// Claude Code's built-in tools carry long safety/protocol text — its Bash tool
// alone is ~6 KB — so a 4 KB cap rejected a stock `claude` session on its very
// first request (external report 2026-07-24), blocking the "Claude Code backend"
// use case entirely. Anthropic's own API bounds tool descriptions by the context
// window, not a small per-tool cap; 32 KB comfortably fits real agent toolsets
// (with headroom) while still bounding abuse, and stays under the 64 KB schema cap.
pub(crate) const MAX_TOOL_DESCRIPTION_LEN: usize = 32768;
pub(crate) const MAX_STOP_SEQUENCES: usize = 16;
pub(crate) const MAX_TOOL_SCHEMA_BYTES: usize = 65536;
pub(crate) const DEFAULT_TOP_K: u32 = 40;
pub(crate) const DEFAULT_MAX_TOKENS: u32 = 32768;
pub(crate) const SSE_KEEPALIVE_INTERVAL_SECS: u64 = 15;

/// A generation streaming to a client must stop if the consumer stops reading
/// for this long. `tx.closed()` catches a client that CLOSES the connection;
/// this catches one that holds the connection open but stops reading (crash,
/// client-side timeout-and-retry, closed laptop): the SSE buffer fills, the
/// next `send` blocks on backpressure, and a stall past this bound means the
/// consumer is gone. Returning then drops the token receiver, which cancels the
/// worker — bounding runaway compute on a shared/public node to this window
/// instead of the whole token budget (external report 2026-07-24, Finding 2).
///
/// Generous enough never to truncate a live client: `send` unblocks the instant
/// the client accepts a single token, so even a ~0.1 tok/s consumer clears the
/// buffer far inside this window; only a fully-stopped reader trips it.
pub(crate) const SSE_CONSUMER_STALL_TIMEOUT: std::time::Duration =
    std::time::Duration::from_secs(60);

/// Send an event to the SSE bridge channel, treating BOTH a dropped receiver
/// (client closed the connection) AND a prolonged backpressure stall (client
/// stopped reading but held the connection open) as "consumer gone". Returns
/// `false` when the caller should stop generating. See
/// [`SSE_CONSUMER_STALL_TIMEOUT`].
pub(crate) async fn sse_send_live<T>(tx: &tokio::sync::mpsc::Sender<T>, ev: T) -> bool {
    matches!(
        tokio::time::timeout(SSE_CONSUMER_STALL_TIMEOUT, tx.send(ev)).await,
        Ok(Ok(()))
    )
}

/// Build SamplingParams with standard clamping applied across all API handlers.
/// All fields are pre-clamped to safe ranges:
/// - temperature: [0.0, 2.0]
/// - top_p: (EPSILON, 1.0]
/// - max_tokens: [1, DEFAULT_MAX_TOKENS]
/// - frequency/presence penalties: [-2.0, 2.0]
/// - top_logprobs: [0, 20]
///
/// Single source of truth — if bounds change, they change here, not at each caller.
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_sampling_params(
    temperature: f32,
    top_p: f32,
    top_k: u32,
    max_tokens: u32,
    stop: Vec<String>,
    frequency_penalty: f32,
    presence_penalty: f32,
    logprobs: bool,
    top_logprobs: u32,
) -> crate::types::SamplingParams {
    crate::types::SamplingParams {
        temperature: temperature.clamp(0.0, 2.0),
        top_p: top_p.clamp(f32::EPSILON, 1.0),
        top_k,
        max_tokens: max_tokens.clamp(1, DEFAULT_MAX_TOKENS),
        stop,
        frequency_penalty: frequency_penalty.clamp(-2.0, 2.0),
        presence_penalty: presence_penalty.clamp(-2.0, 2.0),
        logprobs,
        top_logprobs: top_logprobs.min(20),
    }
}

/// Attach routing + timing headers to a response.
///
/// The single place any response path adds them, so all four (OpenAI,
/// Anthropic, Responses, MCP) stay identical by construction rather than by
/// four authors remembering. A `None` trace leaves the response untouched —
/// cloud-proxied and pre-dispatch-rejected requests have no swarm route, and an
/// absent header is honest where `x-swarm-route: local` would not be.
///
/// Pass `streaming: true` for SSE: headers flush before the body, so TTFT and
/// decode are not yet known and are omitted rather than reported as zero.
pub(crate) fn attach_route_headers(
    mut response: axum::response::Response,
    trace: Option<&crate::inference::trace::TraceSnapshot>,
    streaming: bool,
) -> axum::response::Response {
    let Some(snap) = trace else {
        return response;
    };
    let headers = response.headers_mut();
    for (name, value) in crate::inference::trace::response_headers(snap, streaming) {
        // Values are node-id hex, ISO region codes, integers and Server-Timing
        // tokens, so a parse failure means we built something malformed — skip
        // rather than fail the user's request over a diagnostic header.
        match axum::http::HeaderValue::from_str(&value) {
            Ok(v) => {
                headers.insert(name, v);
            }
            Err(e) => tracing::debug!(name, value, error = %e, "skipping malformed route header"),
        }
    }
    response
}

/// Submit a non-streaming inference request to the router and await the result.
pub(crate) async fn submit_to_router(
    router_tx: &tokio::sync::mpsc::Sender<crate::inference::router::RouterCommand>,
    inference_req: crate::types::InferenceRequest,
) -> Result<crate::inference::router::InferenceOutput, crate::error::ApiError> {
    let (result_tx, result_rx) = tokio::sync::oneshot::channel();
    router_tx
        .send(crate::inference::router::RouterCommand::Submit {
            request: inference_req,
            result_tx,
        })
        .await
        .map_err(|_| {
            crate::error::ApiError(crate::error::SwarmError::ServiceUnavailable(
                "Router unavailable".into(),
            ))
        })?;
    result_rx
        .await
        .map_err(|_| {
            crate::error::ApiError(crate::error::SwarmError::ServiceUnavailable(
                "Router dropped the request".into(),
            ))
        })?
        .map_err(crate::error::ApiError)
}

/// Resolve chat template, BOS token, and EOS token for a model.
///
/// Fallback chain:
/// 1. `loaded_model_info` (in-memory, fastest — available when model is loaded locally)
/// 2. GGUF header file on disk (for distributed-only nodes that have the probe)
/// 3. HuggingFace metadata probe (downloads header if available)
/// 4. Empty defaults (template=None, bos="", eos="")
pub(crate) async fn resolve_chat_template(
    state: &crate::api::server::AppState,
    model_name: &str,
) -> (Option<String>, String, String) {
    // 1. In-memory loaded model info
    {
        let info = state.shared_state.loaded_model_info.read().await;
        if let Some(i) = info.as_ref() {
            return (
                i.chat_template.clone(),
                i.bos_token.clone(),
                i.eos_token.clone(),
            );
        }
    }

    // 2. GGUF header on disk
    let header_path = state
        .shared_state
        .model_dir(model_name)
        .join(crate::model::shard::HEADER_FILENAME);
    if header_path.exists() {
        if let Some((t, b, e)) = crate::inference::pipeline::template_from_header(&header_path) {
            return (t, b, e);
        }
    }

    // 3. HuggingFace metadata probe
    let mid = crate::types::ModelId(model_name.to_string());
    if let Some(hf_src) = state.shared_state.models.hf_sources.get(&mid) {
        let model_dir = state.shared_state.model_dir(model_name);
        let shard_size = state.shared_state.config.model.shard_size_bytes();
        if let Ok(info) = crate::model::huggingface::probe_gguf_file(
            &hf_src.repo_id,
            &hf_src.filename,
            shard_size,
        )
        .await
        {
            if let Ok(hp) = crate::model::huggingface::download_gguf_header(
                &hf_src.repo_id,
                &hf_src.filename,
                &model_dir,
                info.header_size,
            )
            .await
            {
                if let Some((t, b, e)) = crate::inference::pipeline::template_from_header(&hp) {
                    return (t, b, e);
                }
            }
        }
    }

    // 4. Empty defaults
    (None, String::new(), String::new())
}

pub mod admin;
pub mod admin_hf;
pub mod admin_models;
pub mod admin_providers;
pub mod anthropic;
#[cfg(feature = "claude-subscription")]
pub mod claude_session;
#[cfg(feature = "claude-subscription")]
pub mod claude_sub;
pub mod dashboard_trust;
pub mod identity;
pub mod mcp;
pub mod metrics;
pub mod middleware;
pub mod openai;
pub mod pool;
pub mod providers;
pub mod server;
pub mod sse;
pub mod tailscale;

/// Derive the internal request UUID from a public API request id.
///
/// **Deterministic on purpose.** API request ids look like `swarm-<hex>`, which
/// is not a UUID, so `Uuid::parse_str` fails on every real request. Two call
/// sites each doing `parse_str(...).unwrap_or_else(|_| Uuid::new_v4())` will
/// therefore mint two DIFFERENT random ids for the same request and silently
/// fail to find each other — which is exactly how the streaming progress
/// lookup came to return nothing while the admin API showed the request fine
/// (found by running it, 2026-07-28).
///
/// Hashing the id means any number of call sites agree without threading a
/// value between them. A genuine UUID is still parsed as itself so ids that
/// already round-trip keep doing so.
pub fn request_uuid(request_id: &str) -> uuid::Uuid {
    if let Ok(u) = uuid::Uuid::parse_str(request_id) {
        return u;
    }
    let digest = blake3::hash(request_id.as_bytes());
    let mut bytes = [0u8; 16];
    bytes.copy_from_slice(&digest.as_bytes()[..16]);
    uuid::Uuid::from_bytes(bytes)
}

#[cfg(test)]
mod request_uuid_tests {
    use super::request_uuid;

    #[test]
    fn the_same_api_id_always_yields_the_same_uuid() {
        let a = request_uuid("swarm-b553d8a867eb4fe580b5da821327c029");
        let b = request_uuid("swarm-b553d8a867eb4fe580b5da821327c029");
        assert_eq!(
            a, b,
            "derivation must not be random — call sites must agree"
        );
    }

    #[test]
    fn different_api_ids_yield_different_uuids() {
        assert_ne!(request_uuid("swarm-aaa"), request_uuid("swarm-bbb"));
    }

    #[test]
    fn a_real_uuid_is_preserved() {
        let u = uuid::Uuid::new_v4();
        assert_eq!(request_uuid(&u.to_string()), u);
    }
}
pub mod tool_parse;
pub mod websocket;

/// Increment the requests_made counter (best-effort, non-blocking).
pub(crate) fn increment_requests_made(state: &crate::daemon::state::SharedState) {
    if let Ok(mut stats) = state.metrics.node_stats.try_write() {
        stats.requests_made += 1;
    }
}

/// Validate common request parameters shared between OpenAI and Anthropic handlers.
/// Checks model name length, message count, temperature/top_p/top_logprobs ranges.
pub(crate) fn validate_common_params(
    model_len: usize,
    message_count: usize,
    temperature: f64,
) -> Result<(), crate::error::ApiError> {
    if model_len > 256 {
        return Err(crate::error::ApiError(
            crate::error::SwarmError::Validation("model name too long (max 256 chars)".into()),
        ));
    }
    if message_count == 0 {
        return Err(crate::error::ApiError(
            crate::error::SwarmError::Validation("messages array must not be empty".into()),
        ));
    }
    if message_count > 4096 {
        return Err(crate::error::ApiError(
            crate::error::SwarmError::Validation("Too many messages (max 4096)".into()),
        ));
    }
    if !(0.0..=2.0).contains(&temperature) {
        return Err(crate::error::ApiError(
            crate::error::SwarmError::Validation(format!(
                "temperature must be between 0 and 2, got {temperature}"
            )),
        ));
    }
    Ok(())
}

/// Validate optional sampling parameters that need not be present (top_p,
/// top_logprobs, presence_penalty, frequency_penalty, seed). Reject silly
/// values rather than silently clamping inside `build_sampling_params` —
/// matches the OpenAI spec contract for clients (a 400 makes the client fix
/// its bug; a clamp hides it).
pub(crate) fn validate_optional_sampling(
    top_p: Option<f64>,
    top_logprobs: Option<u32>,
    presence_penalty: Option<f64>,
    frequency_penalty: Option<f64>,
) -> Result<(), crate::error::ApiError> {
    if let Some(tp) = top_p {
        if !(0.0..=1.0).contains(&tp) {
            return Err(crate::error::ApiError(
                crate::error::SwarmError::Validation(format!(
                    "top_p must be between 0 and 1, got {tp}"
                )),
            ));
        }
    }
    if let Some(tl) = top_logprobs {
        if tl > 20 {
            return Err(crate::error::ApiError(
                crate::error::SwarmError::Validation(format!(
                    "top_logprobs must be <= 20, got {tl}"
                )),
            ));
        }
    }
    if let Some(p) = presence_penalty {
        if !(-2.0..=2.0).contains(&p) {
            return Err(crate::error::ApiError(
                crate::error::SwarmError::Validation(format!(
                    "presence_penalty must be between -2 and 2, got {p}"
                )),
            ));
        }
    }
    if let Some(f) = frequency_penalty {
        if !(-2.0..=2.0).contains(&f) {
            return Err(crate::error::ApiError(
                crate::error::SwarmError::Validation(format!(
                    "frequency_penalty must be between -2 and 2, got {f}"
                )),
            ));
        }
    }
    Ok(())
}

/// Validate stop sequences (shared between OpenAI and Anthropic handlers).
pub(crate) fn validate_stop_sequences(stops: &[String]) -> Result<(), crate::error::ApiError> {
    if stops.len() > MAX_STOP_SEQUENCES {
        return Err(crate::error::ApiError(
            crate::error::SwarmError::Validation(format!(
                "Too many stop sequences (max {MAX_STOP_SEQUENCES})"
            )),
        ));
    }
    if stops.iter().any(|s| s.is_empty() || s.len() > 256) {
        return Err(crate::error::ApiError(
            crate::error::SwarmError::Validation("Stop sequences must be 1–256 chars each".into()),
        ));
    }
    Ok(())
}

/// Validate total prompt content size against per-message and total limits.
pub(crate) fn validate_content_size(
    content_sizes: impl Iterator<Item = usize>,
) -> Result<(), crate::error::ApiError> {
    const MAX_MESSAGE_CONTENT_BYTES: usize = 2 * 1024 * 1024;
    const MAX_TOTAL_PROMPT_BYTES: usize = 4 * 1024 * 1024;
    let mut total: usize = 0;
    for size in content_sizes {
        if size > MAX_MESSAGE_CONTENT_BYTES {
            return Err(crate::error::ApiError(
                crate::error::SwarmError::Validation(
                    "Message content too large (max 2MB per message)".into(),
                ),
            ));
        }
        total = total.saturating_add(size);
    }
    if total > MAX_TOTAL_PROMPT_BYTES {
        return Err(crate::error::ApiError(
            crate::error::SwarmError::Validation(format!(
                "Total prompt content too large ({total} bytes, max {MAX_TOTAL_PROMPT_BYTES}). Reduce your messages."
            )),
        ));
    }
    Ok(())
}

/// Validate tools array — count and per-tool field sizes.
/// `name_fn` and `desc_fn` and `schema_fn` extract fields from each tool.
pub(crate) fn validate_tools<T>(
    tools: &[T],
    name_fn: impl Fn(&T) -> Option<&str>,
    desc_fn: impl Fn(&T) -> Option<&str>,
    schema_size_fn: impl Fn(&T) -> Option<usize>,
) -> Result<(), crate::error::ApiError> {
    if tools.len() > MAX_TOOLS {
        return Err(crate::error::ApiError(
            crate::error::SwarmError::Validation(format!("Too many tools (max {MAX_TOOLS})")),
        ));
    }
    for tool in tools {
        if let Some(name) = name_fn(tool) {
            if name.len() > MAX_TOOL_NAME_LEN {
                return Err(crate::error::ApiError(
                    crate::error::SwarmError::Validation(format!(
                        "Tool name too long: {} chars (max {MAX_TOOL_NAME_LEN})",
                        name.len()
                    )),
                ));
            }
        }
        if let Some(desc) = desc_fn(tool) {
            if desc.len() > MAX_TOOL_DESCRIPTION_LEN {
                return Err(crate::error::ApiError(
                    crate::error::SwarmError::Validation(format!(
                        "Tool description too long: {} chars (max {MAX_TOOL_DESCRIPTION_LEN})",
                        desc.len()
                    )),
                ));
            }
        }
        if let Some(size) = schema_size_fn(tool) {
            if size > MAX_TOOL_SCHEMA_BYTES {
                return Err(crate::error::ApiError(
                    crate::error::SwarmError::Validation(
                        "Tool parameters/schema too large (max 64KB)".into(),
                    ),
                ));
            }
        }
    }
    Ok(())
}

/// How many of a model's shards this node holds, and how many are reachable
/// network-wide right now.
///
/// The single answer to "can this node serve the model itself". `/v1/models`
/// used to label a model `local` when it matched `loaded_model_info` — the
/// most-recently-loaded singleton — which had nothing to do with shard
/// possession. A tester found the flag effectively inverted: four models held
/// completely (4/4, 3/3, 2/2) were reported `network`, while the only partially
/// held model (2/8) was reported `local`, because it happened to be the last one
/// touched. A client choosing models by `owned_by` to avoid network round trips
/// picked exactly wrong. `/api/admin/shard-storage` was already correct because
/// it counted shards; this is that computation, shared.
///
/// The global count uses `any_holder_reachable`, matching the scheduler's
/// liveness oracle, so a departed peer's stale announce cannot make a model look
/// servable when it is not.
pub(crate) fn count_shard_availability(
    m: &crate::types::ModelManifest,
    shared: &crate::daemon::SharedState,
) -> (usize, usize) {
    let local_node_id = shared.identity.node_id().clone();
    let mut local_count = 0usize;
    let mut global_count = 0usize;
    for idx in 0..m.shard_count {
        let sid = crate::types::ShardId {
            model_id: m.id.clone(),
            index: idx,
        };
        let holders = shared.model_registry.shard_holders(&sid);
        if holders.contains(&local_node_id) {
            local_count += 1;
        }
        if shared.any_holder_reachable(&holders) {
            global_count += 1;
        }
    }
    (local_count, global_count)
}

/// Build a credit balance summary JSON object.
///
/// `lifetime_refunded` and `net_spent` are reported alongside the raw counters
/// because `lifetime_spent` is GROSS reservations and stays monotonic across
/// refunds. Publishing only earned/spent/balance made a healthy node look
/// broken — reported 2026-07-29 as a ~905k "arithmetic anomaly" that was
/// simply 97% of that node's reservations being refunded after failed
/// requests. See `CreditBalance::lifetime_refunded`.
pub(crate) fn credit_summary_json(credit: &swarmllm_types::CreditBalance) -> serde_json::Value {
    serde_json::json!({
        "balance": credit.balance,
        "lifetime_earned": credit.lifetime_earned,
        "lifetime_spent": credit.lifetime_spent,
        "lifetime_refunded": credit.lifetime_refunded,
        "net_spent": credit.net_spent(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // Minimal tool shape for exercising validate_tools' field extractors.
    struct Tool {
        name: &'static str,
        desc: String,
        schema_bytes: usize,
    }

    fn check(tools: &[Tool]) -> Result<(), crate::error::ApiError> {
        validate_tools(
            tools,
            |t| Some(t.name),
            |t| Some(t.desc.as_str()),
            |t| Some(t.schema_bytes),
        )
    }

    #[test]
    fn accepts_claude_code_scale_tool_description() {
        // Claude Code's built-in Bash tool description is ~6 KB (external report
        // 2026-07-24). A stock `claude` session must connect on its first request,
        // so the cap has to clear that comfortably.
        let tool = Tool {
            name: "Bash",
            desc: "x".repeat(6_174),
            schema_bytes: 512,
        };
        assert!(check(&[tool]).is_ok());
    }

    #[test]
    fn still_rejects_abusive_tool_description() {
        // The cap is an abuse guard, not gone — a description past 32 KB is refused.
        let tool = Tool {
            name: "Bash",
            desc: "x".repeat(MAX_TOOL_DESCRIPTION_LEN + 1),
            schema_bytes: 512,
        };
        assert!(check(&[tool]).is_err());
    }

    #[test]
    fn tool_description_cap_covers_claude_code_toolset_headroom() {
        // Compile-time guard against a future re-tightening below Claude Code's
        // real needs (its Bash tool description is ~6 KB); this build fails if
        // the cap is ever dropped under 2× that.
        const { assert!(MAX_TOOL_DESCRIPTION_LEN >= 6_174 * 2) };
    }

    #[tokio::test]
    async fn sse_send_live_true_when_consumer_reads() {
        let (tx, mut rx) = tokio::sync::mpsc::channel::<u8>(4);
        assert!(sse_send_live(&tx, 7u8).await);
        assert_eq!(rx.recv().await, Some(7));
    }

    #[tokio::test]
    async fn sse_send_live_false_when_consumer_closed() {
        // A dropped receiver (client closed the connection) → "consumer gone".
        let (tx, rx) = tokio::sync::mpsc::channel::<u8>(4);
        drop(rx);
        assert!(!sse_send_live(&tx, 7u8).await);
    }
}
