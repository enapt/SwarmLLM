use axum::extract::State;
use axum::response::IntoResponse;
use axum::Json;

use crate::api::server::AppState;
use crate::config::{ProviderEntry, ProvidersConfig};
use crate::error::ApiError;

/// Build a passthrough response from a proxied reqwest response.
///
/// Streams SSE for streaming requests, returns JSON body for non-streaming.
pub async fn build_passthrough_response(
    resp: reqwest::Response,
    stream: bool,
) -> Result<axum::response::Response, ApiError> {
    if stream {
        let byte_stream = resp.bytes_stream();
        let body = axum::body::Body::from_stream(byte_stream);
        build_sse_response(body)
    } else {
        // Surface a body-read failure as a 502 ProviderError instead of
        // unwrap_or_default()'ing into an empty 200. The caller (chat
        // completions, anthropic proxy) hands the response straight to the
        // user; an empty 200 looks like the provider returned an empty
        // object, hiding the real failure (transport drop mid-body).
        let body = resp.text().await.map_err(|e| {
            ApiError(crate::error::SwarmError::ProviderError {
                status: 502,
                body: format!("Failed to read provider response body: {e}"),
            })
        })?;
        axum::response::Response::builder()
            .header("content-type", "application/json")
            .body(axum::body::Body::from(body))
            .map_err(|e| {
                ApiError(crate::error::SwarmError::Internal(format!(
                    "Failed to build response: {e}"
                )))
            })
    }
}

/// Build an SSE response from a body stream.
/// Shared helper used by claude_sub, claude_session, and passthrough responses.
pub fn build_sse_response(body: axum::body::Body) -> Result<axum::response::Response, ApiError> {
    axum::response::Response::builder()
        .header("content-type", "text/event-stream")
        .header("cache-control", "no-cache")
        .body(body)
        .map_err(|e| {
            ApiError(crate::error::SwarmError::Internal(format!(
                "Failed to build SSE response: {e}"
            )))
        })
}

/// Extract a friendly error message from a provider error response body.
///
/// Scrubs API keys, attempts to parse JSON and extract a human-readable message
/// field (trying keys in the given priority order), then truncates.
/// Returns the provider error as an `ApiError` with the original HTTP status.
///
/// Pass an empty `key_priority` (`&[]`) for a pass-through caller that doesn't
/// know the provider's JSON error shape — the scrubbed/truncated body is
/// returned as-is. Either way the helper emits a single structured
/// `tracing::warn!` so call sites don't need to log separately.
pub(crate) fn extract_provider_error(
    raw_body: &str,
    status: reqwest::StatusCode,
    provider_label: &str,
    key_priority: &[&[&str]],
) -> ApiError {
    let scrubbed = crate::crypto::scrub_api_keys(raw_body);
    tracing::warn!(status = %status, body = %scrubbed, provider = %provider_label, "Provider returned error");
    let friendly = serde_json::from_str::<serde_json::Value>(&scrubbed)
        .ok()
        .and_then(|v| {
            for keys in key_priority {
                let mut node = Some(&v);
                for &k in *keys {
                    node = node.and_then(|n| n.get(k));
                }
                if let Some(msg) = node.and_then(|n| n.as_str()) {
                    return Some(msg.to_string());
                }
            }
            None
        })
        .unwrap_or(scrubbed);
    let friendly = super::scrub_truncate_error(&friendly);
    ApiError(crate::error::SwarmError::ProviderError {
        status: status.as_u16(),
        body: friendly,
    })
}

/// JSON key priority for OpenAI-compatible provider error responses.
pub(crate) const OPENAI_ERROR_KEYS: &[&[&str]] =
    &[&["detail"], &["error", "message"], &["message"]];
/// JSON key priority for Anthropic provider error responses.
pub(crate) const ANTHROPIC_ERROR_KEYS: &[&[&str]] =
    &[&["error", "message"], &["message"], &["detail"]];

/// Known provider base URLs (OpenAI-compatible).
pub fn provider_base_url(name: &str) -> Option<&'static str> {
    match name {
        "anthropic" => Some("https://api.anthropic.com"),
        "openai" => Some("https://api.openai.com/v1"),
        "deepseek" => Some("https://api.deepseek.com/v1"),
        "mistral" => Some("https://api.mistral.ai/v1"),
        "groq" => Some("https://api.groq.com/openai/v1"),
        "nvidia_nim" => Some("https://integrate.api.nvidia.com/v1"),
        "cerebras" => Some("https://api.cerebras.ai/v1"),
        "sambanova" => Some("https://api.sambanova.ai/v1"),
        "fireworks" => Some("https://api.fireworks.ai/inference/v1"),
        "together" => Some("https://api.together.xyz/v1"),
        "deepinfra" => Some("https://api.deepinfra.com/v1/openai"),
        "moonshot" => Some("https://api.moonshot.ai/v1"),
        _ => None,
    }
}

/// Resolved provider info for routing.
pub struct ProviderInfo {
    pub name: String,
    pub base_url: String,
    pub api_key: String,
    pub is_anthropic: bool,
    /// True for subprocess-based providers (e.g. claude-subscription).
    pub is_subprocess: bool,
}

/// Resolve a model name to a cloud provider.
///
/// Supports two routing modes:
/// 1. Model prefix: `claude-*` → Anthropic, `gpt-*` → OpenAI, `deepseek-*` → DeepSeek, etc.
/// 2. Explicit syntax: `provider:model` (e.g. `openai:gpt-4o`)
pub fn resolve_provider(model: &str, config: &ProvidersConfig) -> Option<ProviderInfo> {
    let result = resolve_provider_inner(model, config);
    tracing::debug!(
        model_id = model,
        resolved_provider = ?result.as_ref().map(|p| &p.name),
        "DIAG: provider resolution"
    );
    result
}

fn resolve_provider_inner(model: &str, config: &ProvidersConfig) -> Option<ProviderInfo> {
    // Explicit provider:model syntax
    if let Some((provider_name, _model_name)) = model.split_once(':') {
        return resolve_by_name(provider_name, config);
    }

    // Model prefix routing
    let lower = model.to_lowercase();

    // Claude subscription: route through local CLI subprocess (higher priority than API key)
    #[cfg(feature = "claude-subscription")]
    if lower.starts_with("claude-") || lower.starts_with("claude3") {
        if let Some(ref sub_config) = config.claude_subscription {
            if sub_config.enabled {
                return Some(ProviderInfo {
                    name: "claude_subscription".into(),
                    base_url: String::new(),
                    api_key: String::new(),
                    is_anthropic: true,
                    is_subprocess: true,
                });
            }
        }
    }

    if lower.starts_with("claude-") || lower.starts_with("claude3") {
        return resolve_by_name("anthropic", config);
    }
    if lower.starts_with("gpt-")
        || lower.starts_with("o1-")
        || lower.starts_with("o3-")
        || lower.starts_with("o4-")
        || lower == "o1"
        || lower == "o3"
        || lower == "o4"
        || lower == "o3-mini"
        || lower == "o4-mini"
    {
        return resolve_by_name("openai", config);
    }
    if lower.starts_with("deepseek") {
        return resolve_by_name("deepseek", config);
    }
    if lower.starts_with("mistral")
        || lower.starts_with("magistral")
        || lower.starts_with("ministral")
        || lower.starts_with("codestral")
        || lower.starts_with("pixtral")
    {
        return resolve_by_name("mistral", config);
    }
    // NVIDIA NIM models use org/model format (meta/, nvidia/, google/, microsoft/, etc.)
    if lower.starts_with("nvidia/") || lower.starts_with("nim/") {
        return resolve_by_name("nvidia_nim", config);
    }
    // Nemotron models are NVIDIA-specific
    if lower.contains("nemotron") && config.nvidia_nim.is_some() {
        return resolve_by_name("nvidia_nim", config);
    }
    // NIM uses org/model format (e.g. meta/llama-3.1-8b-instruct). If NIM is configured
    // and the model has an org/ prefix that didn't match another provider, route to NIM.
    if config.nvidia_nim.is_some() && lower.contains('/') && !lower.starts_with("accounts/")
    // fireworks uses accounts/ prefix
    {
        return resolve_by_name("nvidia_nim", config);
    }
    if lower.starts_with("llama-") && config.groq.is_some() {
        // Groq is popular for fast Llama inference
        return resolve_by_name("groq", config);
    }
    if lower.starts_with("gemma") && config.groq.is_some() {
        return resolve_by_name("groq", config);
    }
    if lower.starts_with("moonshot-") || lower.starts_with("kimi") || lower.starts_with("k2") {
        return resolve_by_name("moonshot", config);
    }
    // Fireworks uses accounts/ prefix
    if lower.starts_with("accounts/fireworks") {
        return resolve_by_name("fireworks", config);
    }

    // Custom providers only match via explicit `provider:model` syntax (handled above)

    None
}

/// Helper: build ProviderInfo from a config entry and known provider name.
/// Returns `None` if the provider name has no known base URL.
fn make_provider(name: &str, entry: &ProviderEntry, is_anthropic: bool) -> Option<ProviderInfo> {
    Some(ProviderInfo {
        name: name.into(),
        base_url: provider_base_url(name)?.into(),
        api_key: entry.api_key.clone(),
        is_anthropic,
        is_subprocess: false,
    })
}

pub fn resolve_by_name(name: &str, config: &ProvidersConfig) -> Option<ProviderInfo> {
    match name {
        "anthropic" => config
            .anthropic
            .as_ref()
            .and_then(|e| make_provider("anthropic", e, true)),
        "openai" => config
            .openai
            .as_ref()
            .and_then(|e| make_provider("openai", e, false)),
        "deepseek" => config
            .deepseek
            .as_ref()
            .and_then(|e| make_provider("deepseek", e, false)),
        "mistral" => config
            .mistral
            .as_ref()
            .and_then(|e| make_provider("mistral", e, false)),
        "groq" => config
            .groq
            .as_ref()
            .and_then(|e| make_provider("groq", e, false)),
        "nvidia_nim" | "nvidia" | "nim" => config
            .nvidia_nim
            .as_ref()
            .and_then(|e| make_provider("nvidia_nim", e, false)),
        "cerebras" => config
            .cerebras
            .as_ref()
            .and_then(|e| make_provider("cerebras", e, false)),
        "sambanova" => config
            .sambanova
            .as_ref()
            .and_then(|e| make_provider("sambanova", e, false)),
        "fireworks" => config
            .fireworks
            .as_ref()
            .and_then(|e| make_provider("fireworks", e, false)),
        "together" => config
            .together
            .as_ref()
            .and_then(|e| make_provider("together", e, false)),
        "deepinfra" => config
            .deepinfra
            .as_ref()
            .and_then(|e| make_provider("deepinfra", e, false)),
        "moonshot" | "kimi" => config
            .moonshot
            .as_ref()
            .and_then(|e| make_provider("moonshot", e, false)),
        _ => {
            // Check custom providers
            config
                .custom
                .iter()
                .find(|c| c.name == name)
                .map(|c| ProviderInfo {
                    name: c.name.clone(),
                    base_url: c.base_url.clone(),
                    api_key: c.api_key.clone(),
                    is_anthropic: false,
                    is_subprocess: false,
                })
        }
    }
}

/// Total HTTP timeout for proxied provider requests. Tuned for reasoning
/// models (DeepSeek R1, etc.) that can take 60-120s before the first token.
const PROVIDER_PROXY_TIMEOUT_SECS: u64 = 300;
/// TCP connect timeout for proxied provider requests.
const PROVIDER_PROXY_CONNECT_SECS: u64 = 30;

/// DNS resolver that filters out private/internal IP addresses *at request
/// time*. Closes the TOCTOU gap in `validate_provider_url`: that helper
/// only runs once at config-update or pre-request, but the actual TCP
/// connection happens later and re-resolves the hostname. A malicious
/// authoritative DNS server can return a public IP for the validation
/// query and a private IP (e.g. cloud metadata `169.254.169.254`) for
/// the request query. Injecting this resolver into the shared client
/// makes the filter run on the same lookup that drives the connection,
/// so the request and the check can't disagree.
struct PrivateIpBlockingResolver;

impl reqwest::dns::Resolve for PrivateIpBlockingResolver {
    fn resolve(&self, name: reqwest::dns::Name) -> reqwest::dns::Resolving {
        let host = name.as_str().to_string();
        Box::pin(async move {
            // GaiResolver (reqwest's default) uses `(host, 0).to_socket_addrs()`
            // and lets the consumer fill in the port. We do the same.
            let resolved: Result<Vec<std::net::SocketAddr>, std::io::Error> =
                tokio::task::spawn_blocking(move || {
                    std::net::ToSocketAddrs::to_socket_addrs(&(host.as_str(), 0u16))
                        .map(|iter| iter.collect())
                })
                .await
                .unwrap_or_else(|join_err| {
                    Err(std::io::Error::other(format!(
                        "DNS resolver task panicked: {join_err}"
                    )))
                });

            let addrs =
                resolved.map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send + Sync>)?;

            let safe: Vec<std::net::SocketAddr> = addrs
                .into_iter()
                .filter(|sa| !is_private_ip(sa.ip()))
                .collect();

            if safe.is_empty() {
                return Err(Box::new(std::io::Error::new(
                    std::io::ErrorKind::PermissionDenied,
                    "all resolved addresses are private/internal",
                ))
                    as Box<dyn std::error::Error + Send + Sync>);
            }

            let iter: Box<dyn Iterator<Item = std::net::SocketAddr> + Send> =
                Box::new(safe.into_iter());
            Ok(iter as reqwest::dns::Addrs)
        })
    }
}

/// Lazily-initialized shared reqwest client for provider proxying.
static PROVIDER_CLIENT: std::sync::LazyLock<reqwest::Client> = std::sync::LazyLock::new(|| {
    crate::http::build_client(|b| {
        b.timeout(std::time::Duration::from_secs(PROVIDER_PROXY_TIMEOUT_SECS))
            .connect_timeout(std::time::Duration::from_secs(PROVIDER_PROXY_CONNECT_SECS))
            .dns_resolver(std::sync::Arc::new(PrivateIpBlockingResolver))
    })
});

pub(crate) fn get_provider_client() -> &'static reqwest::Client {
    &PROVIDER_CLIENT
}

/// Proxy an OpenAI-compatible chat completion request to a cloud provider.
/// Called from openai.rs as a fallback when no local model matches.
pub async fn try_proxy_openai(
    state: &AppState,
    body: &serde_json::Value,
    stream: bool,
) -> Result<Option<axum::response::Response>, ApiError> {
    let model = body
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or_default();

    let config = state.shared_state.metrics.providers_config.read().await;
    let provider = match resolve_provider(model, &config) {
        Some(p) if !p.is_anthropic => p,
        _ => {
            // Fallback: check provider_model_map (populated by list_provider_models)
            if let Some(entry) = state.shared_state.metrics.provider_model_map.get(model) {
                let provider_name = entry.value().clone();
                match resolve_by_name(&provider_name, &config) {
                    Some(p) if !p.is_anthropic => {
                        drop(config);
                        tracing::info!(
                            provider = %p.name,
                            model = %model,
                            "Proxying OpenAI-compatible request to cloud provider (model map fallback)"
                        );
                        let response =
                            proxy_openai_compatible(&p.base_url, &p.api_key, body, stream).await?;
                        return Ok(Some(response));
                    }
                    _ => return Ok(None),
                }
            }
            return Ok(None);
        }
    };
    drop(config);

    tracing::info!(
        provider = %provider.name,
        model = %model,
        "Proxying OpenAI-compatible request to cloud provider"
    );

    // `provider:model` selects the provider HERE; the provider itself has never
    // heard of the prefix, so it must not travel upstream. DeepSeek rejects
    // `deepseek:deepseek-v4-flash` with "the supported API model names are
    // deepseek-v4-pro or deepseek-v4-flash" (external report 2026-07-26).
    // The Anthropic surface already stripped it via `strip_provider_prefix`;
    // this one forwarded the body verbatim.
    let body = strip_prefix_in_body(body);

    let response =
        proxy_openai_compatible(&provider.base_url, &provider.api_key, &body, stream).await?;
    Ok(Some(response))
}

/// Return `body` with any `provider:` prefix removed from its `model` field.
///
/// Clones only when a prefix is actually present, so the common path is a cheap
/// reference-preserving passthrough.
fn strip_prefix_in_body(body: &serde_json::Value) -> std::borrow::Cow<'_, serde_json::Value> {
    let Some(model) = body.get("model").and_then(|m| m.as_str()) else {
        return std::borrow::Cow::Borrowed(body);
    };
    let bare = crate::api::strip_provider_prefix(model);
    if bare == model {
        return std::borrow::Cow::Borrowed(body);
    }
    let mut owned = body.clone();
    if let Some(obj) = owned.as_object_mut() {
        obj.insert("model".into(), serde_json::Value::String(bare.to_string()));
    }
    std::borrow::Cow::Owned(owned)
}

/// Validate that a provider base_url uses an allowed scheme and does not target
/// private/internal IP ranges (SSRF prevention for custom providers).
pub(crate) async fn validate_provider_url(base_url: &str) -> Result<(), crate::error::SwarmError> {
    if !base_url.starts_with("https://") && !base_url.starts_with("http://") {
        return Err(crate::error::SwarmError::Validation(
            "Provider base_url must use http or https scheme".into(),
        ));
    }
    // Extract host portion: strip scheme, then take up to the next '/' or ':'
    let after_scheme = if let Some(rest) = base_url.strip_prefix("https://") {
        rest
    } else if let Some(rest) = base_url.strip_prefix("http://") {
        rest
    } else {
        return Ok(());
    };
    let host = after_scheme
        .split('/')
        .next()
        .unwrap_or("")
        .split(':')
        .next()
        .unwrap_or("");
    // Strip IPv6 brackets if present
    let host = host.trim_start_matches('[').trim_end_matches(']');
    if let Ok(ip) = host.parse::<std::net::IpAddr>() {
        if is_private_ip(ip) {
            return Err(crate::error::SwarmError::Validation(
                "Provider base_url must not point to private/internal IP ranges".into(),
            ));
        }
    }
    // Resolve DNS hostnames and check resolved IP against private ranges
    // to prevent DNS-based SSRF (e.g., attacker-controlled DNS resolving to 169.254.169.254).
    // DNS resolution is blocking — run in spawn_blocking to avoid stalling the Tokio executor.
    if host.parse::<std::net::IpAddr>().is_err() && !host.is_empty() {
        let host_owned = host.to_string();
        let resolved = tokio::task::spawn_blocking(move || {
            std::net::ToSocketAddrs::to_socket_addrs(&(&*host_owned, 80u16))
                .ok()
                .and_then(|addrs| addrs.into_iter().find(|a| is_private_ip(a.ip())))
        })
        .await
        .unwrap_or(None);
        if let Some(addr) = resolved {
            return Err(crate::error::SwarmError::Validation(format!(
                "Provider base_url hostname '{}' resolves to private IP {}",
                host,
                addr.ip()
            )));
        }
    }

    // Block known cloud metadata endpoints to mitigate DNS rebinding attacks
    let blocked_hosts = [
        // GCP
        "metadata.google.internal",
        "metadata.gcp.internal",
        "instance-data",
        // Cloud metadata IP (AWS, Azure, Oracle, etc.)
        "169.254.169.254",
        // Azure IMDS
        "metadata.azure.com",
        // AWS IMDSv1/v2
        "instance-data.ec2.internal",
        // DigitalOcean
        "metadata.digitalocean.com",
        // Alibaba Cloud
        "100.100.100.200",
    ];
    let host_lower = host.to_lowercase();
    for blocked in &blocked_hosts {
        if host_lower.contains(blocked) {
            return Err(crate::error::SwarmError::Validation(
                "Provider base_url points to blocked internal hostname".into(),
            ));
        }
    }
    Ok(())
}

/// Check if an IP address is in a private or link-local range.
fn is_private_ip(ip: std::net::IpAddr) -> bool {
    match ip {
        std::net::IpAddr::V4(v4) => v4.is_private() || v4.is_link_local() || v4.is_loopback(),
        std::net::IpAddr::V6(v6) => {
            // Check IPv4-mapped IPv6 addresses (::ffff:x.x.x.x) — these can bypass
            // naive V6-only checks to reach private V4 addresses.
            if let Some(v4) = v6.to_ipv4_mapped() {
                return v4.is_private() || v4.is_loopback() || v4.is_link_local();
            }
            // fe80::/10 link-local, fc00::/7 unique local, ::1 loopback
            let segments = v6.segments();
            v6.is_loopback() || (segments[0] & 0xffc0) == 0xfe80 || (segments[0] & 0xfe00) == 0xfc00
        }
    }
}

/// Try to proxy an OpenAI Responses request (`POST /v1/responses`) to a
/// cloud provider. Mirrors `try_proxy_openai` but targets the `/responses`
/// path on the upstream provider.
///
/// Returns:
/// - `Ok(Some(response))` when an OpenAI-compatible provider matched and
///   the request was proxied (provider's response, including non-2xx).
/// - `Err(_)` when an Anthropic / subprocess provider matched (those don't
///   speak the OpenAI Responses API — caller should surface the message
///   instead of falling through), or when the upstream request errored.
/// - `Ok(None)` when no cloud provider matches the model — caller should
///   continue with local inference.
pub async fn try_proxy_openai_responses(
    state: &AppState,
    body: &serde_json::Value,
    stream: bool,
) -> Result<Option<axum::response::Response>, ApiError> {
    let model = body
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or_default();

    let config = state.shared_state.metrics.providers_config.read().await;
    let provider = match resolve_provider(model, &config).or_else(|| {
        // Same model_map fallback try_proxy_openai uses.
        state
            .shared_state
            .metrics
            .provider_model_map
            .get(model)
            .and_then(|e| resolve_by_name(&e.value().clone(), &config))
    }) {
        Some(p) => p,
        None => return Ok(None),
    };
    drop(config);

    if provider.is_subprocess || provider.is_anthropic {
        // V3 (responses_api_v2): Anthropic providers used to 400 here
        // ("use /v1/messages"). The caller now tries the Anthropic
        // Responses bridge next — we signal "not mine" instead of
        // erroring so the fallthrough path can translate.
        return Ok(None);
    }

    tracing::info!(
        provider = %provider.name,
        model = %model,
        "Proxying /v1/responses request to cloud provider"
    );

    let response =
        proxy_openai_responses(&provider.base_url, &provider.api_key, body, stream).await?;
    Ok(Some(response))
}

/// Low-level POST against an OpenAI-compatible provider endpoint. Handles
/// URL validation, Bearer auth, content-type, three-branch reqwest error
/// classification, non-2xx body extraction, and SSE/JSON passthrough.
/// `endpoint` is the trailing path segment appended to `base_url` (e.g.
/// `"/chat/completions"`, `"/responses"`).
async fn post_openai_compat(
    base_url: &str,
    endpoint: &str,
    api_key: &str,
    body: &serde_json::Value,
    stream: bool,
) -> Result<axum::response::Response, ApiError> {
    // SEC: Validate base_url to prevent SSRF via custom provider configuration.
    validate_provider_url(base_url).await.map_err(ApiError)?;

    let client = get_provider_client();
    let url = format!("{}{}", base_url, endpoint);

    let resp = client
        .post(&url)
        .header("Authorization", format!("Bearer {}", api_key))
        .header("Content-Type", "application/json")
        .json(body)
        .send()
        .await
        .map_err(|e| {
            tracing::warn!(error = %e, url = %url, "Provider proxy request failed");
            let msg = if e.is_timeout() {
                "Provider request timed out. The model may be slow to respond — try again or use a different model.".to_string()
            } else if e.is_connect() {
                "Could not connect to provider API. Check your internet connection.".to_string()
            } else {
                format!("Provider request failed: {e}")
            };
            ApiError(crate::error::SwarmError::ProviderError {
                status: 504,
                body: msg,
            })
        })?;

    if !resp.status().is_success() {
        let status = resp.status();
        let raw_body = resp.text().await.unwrap_or_default();
        return Err(extract_provider_error(
            &raw_body,
            status,
            "Provider",
            OPENAI_ERROR_KEYS,
        ));
    }

    let response = build_passthrough_response(resp, stream).await?;
    Ok(response.into_response())
}

/// Low-level proxy: POST `body` verbatim to `{base_url}/responses`.
/// `body` should already include the original caller's `extras` so unknown
/// fields (`reasoning.effort`, `service_tier`, `text.verbosity`, `include`,
/// `previous_response_id`, ...) round-trip without translation.
pub async fn proxy_openai_responses(
    base_url: &str,
    api_key: &str,
    body: &serde_json::Value,
    stream: bool,
) -> Result<axum::response::Response, ApiError> {
    post_openai_compat(base_url, "/responses", api_key, body, stream).await
}

/// Generic OpenAI-compatible proxy: rewrite base URL + auth header, forward as-is.
pub async fn proxy_openai_compatible(
    base_url: &str,
    api_key: &str,
    body: &serde_json::Value,
    stream: bool,
) -> Result<axum::response::Response, ApiError> {
    post_openai_compat(base_url, "/chat/completions", api_key, body, stream).await
}

/// Proxy a request to the Anthropic Messages API.
///
/// `beta_header` is the caller's `anthropic-beta` value (forwarded verbatim when
/// present) — unlocks features like advanced-tool-use, context-1m, token-
/// efficient-tools, code-execution on the upstream API.
/// `version_header` overrides the default `anthropic-version` when the caller
/// supplied one; otherwise we pin `2023-06-01` (the release the rest of our
/// type surface was built against).
pub async fn proxy_to_anthropic(
    api_key: &str,
    body: &serde_json::Value,
    stream: bool,
    beta_header: Option<&str>,
    version_header: Option<&str>,
) -> Result<axum::response::Response, ApiError> {
    let client = get_provider_client();
    let url = "https://api.anthropic.com/v1/messages";

    let mut req = client
        .post(url)
        .header("x-api-key", api_key)
        .header("anthropic-version", version_header.unwrap_or("2023-06-01"))
        .header("Content-Type", "application/json");
    if let Some(beta) = beta_header {
        req = req.header("anthropic-beta", beta);
    }
    let resp = req.json(body).send().await.map_err(|e| {
        tracing::warn!(error = %e, "Anthropic proxy request failed");
        ApiError(crate::error::SwarmError::Network(format!(
            "Anthropic proxy failed: {e}"
        )))
    })?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        return Err(extract_provider_error(
            &body,
            status,
            "Anthropic",
            ANTHROPIC_ERROR_KEYS,
        ));
    }

    {
        let response = build_passthrough_response(resp, stream).await?;
        Ok(response.into_response())
    }
}

/// GET /v1/providers — List configured providers (public, no keys exposed).
pub async fn list_providers(State(state): State<AppState>) -> Json<serde_json::Value> {
    let config = state.shared_state.metrics.providers_config.read().await;

    let mut providers = vec![
        serde_json::json!({
            "name": "anthropic",
            "configured": config.anthropic.is_some(),
        }),
        serde_json::json!({
            "name": "openai",
            "configured": config.openai.is_some(),
        }),
        serde_json::json!({
            "name": "deepseek",
            "configured": config.deepseek.is_some(),
        }),
        serde_json::json!({
            "name": "mistral",
            "configured": config.mistral.is_some(),
        }),
        serde_json::json!({
            "name": "groq",
            "configured": config.groq.is_some(),
        }),
        serde_json::json!({
            "name": "nvidia_nim",
            "configured": config.nvidia_nim.is_some(),
        }),
        serde_json::json!({
            "name": "cerebras",
            "configured": config.cerebras.is_some(),
        }),
        serde_json::json!({
            "name": "sambanova",
            "configured": config.sambanova.is_some(),
        }),
        serde_json::json!({
            "name": "fireworks",
            "configured": config.fireworks.is_some(),
        }),
        serde_json::json!({
            "name": "together",
            "configured": config.together.is_some(),
        }),
        serde_json::json!({
            "name": "deepinfra",
            "configured": config.deepinfra.is_some(),
        }),
        serde_json::json!({
            "name": "moonshot",
            "configured": config.moonshot.is_some(),
        }),
    ];

    for custom in &config.custom {
        providers.push(serde_json::json!({
            "name": custom.name,
            "configured": true,
            "custom": true,
        }));
    }

    Json(serde_json::json!({ "providers": providers }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{CustomProvider, ProviderEntry, ProvidersConfig};

    #[tokio::test]
    async fn private_ip_blocking_resolver_filters_loopback() {
        use reqwest::dns::Resolve;
        let resolver = PrivateIpBlockingResolver;
        // `localhost` resolves to 127.0.0.1 (and possibly ::1) — both private.
        let name: reqwest::dns::Name = "localhost".parse().unwrap();
        let result = resolver.resolve(name).await;
        assert!(
            result.is_err(),
            "resolver should reject hostnames whose every resolved IP is private/loopback"
        );
    }

    #[tokio::test]
    async fn private_ip_blocking_resolver_passes_public_dns() {
        // example.com is a stable public hostname. Its resolution always
        // returns at least one public IP (93.184.215.14 / 2606:2800:21f:cb07:6820:80da:af6b:8b2c
        // at time of writing). If DNS is unreachable in CI we tolerate the
        // failure mode (returns Err) but the resolver itself must not be
        // the thing rejecting public IPs.
        use reqwest::dns::Resolve;
        let resolver = PrivateIpBlockingResolver;
        let name: reqwest::dns::Name = "example.com".parse().unwrap();
        let result = resolver.resolve(name).await;
        if let Ok(addrs) = result {
            let v: Vec<_> = addrs.collect();
            assert!(
                !v.is_empty(),
                "public DNS should yield at least one address"
            );
            for sa in &v {
                assert!(
                    !is_private_ip(sa.ip()),
                    "resolver leaked private IP {} for example.com",
                    sa.ip()
                );
            }
        }
        // If the lookup itself fails (offline CI), we don't fail the test —
        // that's a network condition, not a resolver bug.
    }

    #[test]
    fn resolve_claude_to_anthropic() {
        let config = ProvidersConfig {
            anthropic: Some(ProviderEntry {
                api_key: "sk-ant-test".into(),
                default_model: None,
            }),
            ..Default::default()
        };
        let p = resolve_provider("claude-opus-4-8", &config).unwrap();
        assert_eq!(p.name, "anthropic");
        assert!(p.is_anthropic);
    }

    #[test]
    fn resolve_gpt_to_openai() {
        let config = ProvidersConfig {
            openai: Some(ProviderEntry {
                api_key: "sk-test".into(),
                default_model: None,
            }),
            ..Default::default()
        };
        let p = resolve_provider("gpt-4o", &config).unwrap();
        assert_eq!(p.name, "openai");
        assert!(!p.is_anthropic);
    }

    #[test]
    fn resolve_deepseek_prefix() {
        let config = ProvidersConfig {
            deepseek: Some(ProviderEntry {
                api_key: "sk-ds".into(),
                default_model: None,
            }),
            ..Default::default()
        };
        let p = resolve_provider("deepseek-chat", &config).unwrap();
        assert_eq!(p.name, "deepseek");
        assert_eq!(p.base_url, "https://api.deepseek.com/v1");
    }

    #[test]
    fn resolve_explicit_provider_syntax() {
        let config = ProvidersConfig {
            groq: Some(ProviderEntry {
                api_key: "gsk-test".into(),
                default_model: None,
            }),
            ..Default::default()
        };
        let p = resolve_provider("groq:llama-3.1-70b", &config).unwrap();
        assert_eq!(p.name, "groq");
        assert_eq!(p.base_url, "https://api.groq.com/openai/v1");
    }

    #[test]
    fn resolve_custom_provider() {
        let config = ProvidersConfig {
            custom: vec![CustomProvider {
                name: "mycloud".into(),
                base_url: "https://api.mycloud.example/v1".into(),
                api_key: "tok-test".into(),
                default_model: None,
            }],
            ..Default::default()
        };
        let p = resolve_provider("mycloud:meta-llama/Llama-3-70b", &config).unwrap();
        assert_eq!(p.name, "mycloud");
        assert_eq!(p.base_url, "https://api.mycloud.example/v1");
    }

    #[test]
    fn resolve_nvidia_nim_prefix() {
        let config = ProvidersConfig {
            nvidia_nim: Some(ProviderEntry {
                api_key: "nvapi-test".into(),
                default_model: None,
            }),
            ..Default::default()
        };
        let p = resolve_provider("nvidia/llama-3.1-nemotron-70b-instruct", &config).unwrap();
        assert_eq!(p.name, "nvidia_nim");
        assert_eq!(p.base_url, "https://integrate.api.nvidia.com/v1");
    }

    #[test]
    fn resolve_nvidia_nim_explicit() {
        let config = ProvidersConfig {
            nvidia_nim: Some(ProviderEntry {
                api_key: "nvapi-test".into(),
                default_model: None,
            }),
            ..Default::default()
        };
        let p = resolve_provider("nim:meta/llama-3.1-8b-instruct", &config).unwrap();
        assert_eq!(p.name, "nvidia_nim");
    }

    #[test]
    fn resolve_cerebras_explicit() {
        let config = ProvidersConfig {
            cerebras: Some(ProviderEntry {
                api_key: "csk-test".into(),
                default_model: None,
            }),
            ..Default::default()
        };
        let p = resolve_provider("cerebras:llama3.1-8b", &config).unwrap();
        assert_eq!(p.name, "cerebras");
        assert_eq!(p.base_url, "https://api.cerebras.ai/v1");
    }

    #[test]
    fn resolve_together_explicit() {
        let config = ProvidersConfig {
            together: Some(ProviderEntry {
                api_key: "tok-test".into(),
                default_model: None,
            }),
            ..Default::default()
        };
        let p = resolve_provider("together:meta-llama/Llama-3-70b", &config).unwrap();
        assert_eq!(p.name, "together");
        assert_eq!(p.base_url, "https://api.together.xyz/v1");
    }

    #[test]
    fn resolve_nemotron_to_nvidia() {
        let config = ProvidersConfig {
            nvidia_nim: Some(ProviderEntry {
                api_key: "nvapi-test".into(),
                default_model: None,
            }),
            ..Default::default()
        };
        let p = resolve_provider("nemotron-4-340b-instruct", &config).unwrap();
        assert_eq!(p.name, "nvidia_nim");
    }

    #[test]
    fn resolve_unconfigured_returns_none() {
        let config = ProvidersConfig::default();
        assert!(resolve_provider("gpt-4o", &config).is_none());
        assert!(resolve_provider("claude-opus-4-8", &config).is_none());
    }

    #[test]
    fn provider_base_urls() {
        assert_eq!(
            provider_base_url("openai"),
            Some("https://api.openai.com/v1")
        );
        assert_eq!(
            provider_base_url("deepseek"),
            Some("https://api.deepseek.com/v1")
        );
        assert_eq!(
            provider_base_url("mistral"),
            Some("https://api.mistral.ai/v1")
        );
        assert_eq!(
            provider_base_url("groq"),
            Some("https://api.groq.com/openai/v1")
        );
        assert_eq!(
            provider_base_url("nvidia_nim"),
            Some("https://integrate.api.nvidia.com/v1")
        );
        assert_eq!(
            provider_base_url("cerebras"),
            Some("https://api.cerebras.ai/v1")
        );
        assert_eq!(
            provider_base_url("sambanova"),
            Some("https://api.sambanova.ai/v1")
        );
        assert_eq!(
            provider_base_url("fireworks"),
            Some("https://api.fireworks.ai/inference/v1")
        );
        assert_eq!(
            provider_base_url("together"),
            Some("https://api.together.xyz/v1")
        );
        assert_eq!(
            provider_base_url("deepinfra"),
            Some("https://api.deepinfra.com/v1/openai")
        );
        assert_eq!(
            provider_base_url("moonshot"),
            Some("https://api.moonshot.ai/v1")
        );
        assert_eq!(provider_base_url("unknown"), None);
    }

    #[test]
    fn resolve_kimi_k2_to_moonshot() {
        let config = ProvidersConfig {
            moonshot: Some(ProviderEntry {
                api_key: "sk-kimi".into(),
                default_model: None,
            }),
            ..Default::default()
        };
        // kimi prefix (current flagship)
        let p = resolve_provider("kimi-k3", &config).unwrap();
        assert_eq!(p.name, "moonshot");
        assert_eq!(p.base_url, "https://api.moonshot.ai/v1");

        // k2 prefix (bare shorthand)
        let p2 = resolve_provider("k2-base", &config).unwrap();
        assert_eq!(p2.name, "moonshot");

        // moonshot- prefix (legacy IDs still route)
        let p3 = resolve_provider("moonshot-v1-8k", &config).unwrap();
        assert_eq!(p3.name, "moonshot");
    }
}
