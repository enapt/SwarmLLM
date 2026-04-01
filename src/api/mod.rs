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

/// Strip `provider:` prefix from a model name, returning the bare model name.
pub(crate) fn strip_provider_prefix(model: &str) -> &str {
    model.split_once(':').map_or(model, |(_, name)| name)
}

// Shared validation limits for API request parameters.
// Used by both openai.rs and anthropic.rs handlers.
pub(crate) const MAX_TOOLS: usize = 128;
pub(crate) const MAX_TOOL_NAME_LEN: usize = 256;
pub(crate) const MAX_TOOL_DESCRIPTION_LEN: usize = 4096;
pub(crate) const MAX_STOP_SEQUENCES: usize = 16;
pub(crate) const MAX_TOOL_SCHEMA_BYTES: usize = 65536;

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

pub mod admin;
pub mod admin_hf;
pub mod admin_models;
pub mod admin_providers;
pub mod anthropic;
pub mod identity;
pub mod mcp;
pub mod metrics;
pub mod middleware;
pub mod openai;
pub mod pool;
pub mod providers;
pub mod server;
pub mod websocket;

/// Increment the requests_made counter (best-effort, non-blocking).
pub(crate) fn increment_requests_made(state: &crate::daemon::state::SharedState) {
    if let Ok(mut stats) = state.metrics.node_stats.try_write() {
        stats.requests_made += 1;
    }
}

/// Validate common request parameters shared between OpenAI and Anthropic handlers.
/// Checks model name length, message count, and temperature range.
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

/// Build a credit balance summary JSON object.
pub(crate) fn credit_summary_json(credit: &swarmllm_types::CreditBalance) -> serde_json::Value {
    serde_json::json!({
        "balance": credit.balance,
        "lifetime_earned": credit.lifetime_earned,
        "lifetime_spent": credit.lifetime_spent,
    })
}
