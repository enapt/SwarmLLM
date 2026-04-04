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
pub(crate) const DEFAULT_TOP_K: u32 = 40;
pub(crate) const DEFAULT_MAX_TOKENS: u32 = 32768;
pub(crate) const SSE_KEEPALIVE_INTERVAL_SECS: u64 = 15;

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
    let header_path =
        crate::model::shard::model_dir(&state.shared_state.config.node.data_dir, model_name)
            .join(crate::model::shard::HEADER_FILENAME);
    if header_path.exists() {
        if let Some((t, b, e)) = crate::inference::pipeline::template_from_header(&header_path) {
            return (t, b, e);
        }
    }

    // 3. HuggingFace metadata probe
    let mid = crate::types::ModelId(model_name.to_string());
    if let Some(hf_src) = state.shared_state.models.hf_sources.get(&mid) {
        let model_dir =
            crate::model::shard::model_dir(&state.shared_state.config.node.data_dir, model_name);
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
