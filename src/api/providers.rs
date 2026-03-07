use axum::extract::State;
use axum::response::IntoResponse;
use axum::Json;

use crate::api::server::AppState;
use crate::config::ProvidersConfig;
use crate::error::ApiError;

/// Known provider base URLs (OpenAI-compatible).
pub fn provider_base_url(name: &str) -> Option<&'static str> {
    match name {
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
        _ => None,
    }
}

/// Resolved provider info for routing.
pub struct ProviderInfo {
    pub name: String,
    pub base_url: String,
    pub api_key: String,
    pub is_anthropic: bool,
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
    if lower.starts_with("claude-") || lower.starts_with("claude3") {
        return resolve_by_name("anthropic", config);
    }
    if lower.starts_with("gpt-") || lower.starts_with("o1-") || lower.starts_with("o3-") {
        return resolve_by_name("openai", config);
    }
    if lower.starts_with("deepseek") {
        return resolve_by_name("deepseek", config);
    }
    if lower.starts_with("mistral")
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
    // Fireworks uses accounts/ prefix
    if lower.starts_with("accounts/fireworks") {
        return resolve_by_name("fireworks", config);
    }

    // Custom providers only match via explicit `provider:model` syntax (handled above)

    None
}

pub fn resolve_by_name(name: &str, config: &ProvidersConfig) -> Option<ProviderInfo> {
    match name {
        "anthropic" => config.anthropic.as_ref().map(|e| ProviderInfo {
            name: "anthropic".into(),
            base_url: "https://api.anthropic.com".into(),
            api_key: e.api_key.clone(),
            is_anthropic: true,
        }),
        "openai" => config.openai.as_ref().map(|e| ProviderInfo {
            name: "openai".into(),
            base_url: provider_base_url("openai").unwrap().into(),
            api_key: e.api_key.clone(),
            is_anthropic: false,
        }),
        "deepseek" => config.deepseek.as_ref().map(|e| ProviderInfo {
            name: "deepseek".into(),
            base_url: provider_base_url("deepseek").unwrap().into(),
            api_key: e.api_key.clone(),
            is_anthropic: false,
        }),
        "mistral" => config.mistral.as_ref().map(|e| ProviderInfo {
            name: "mistral".into(),
            base_url: provider_base_url("mistral").unwrap().into(),
            api_key: e.api_key.clone(),
            is_anthropic: false,
        }),
        "groq" => config.groq.as_ref().map(|e| ProviderInfo {
            name: "groq".into(),
            base_url: provider_base_url("groq").unwrap().into(),
            api_key: e.api_key.clone(),
            is_anthropic: false,
        }),
        "nvidia_nim" | "nvidia" | "nim" => config.nvidia_nim.as_ref().map(|e| ProviderInfo {
            name: "nvidia_nim".into(),
            base_url: provider_base_url("nvidia_nim").unwrap().into(),
            api_key: e.api_key.clone(),
            is_anthropic: false,
        }),
        "cerebras" => config.cerebras.as_ref().map(|e| ProviderInfo {
            name: "cerebras".into(),
            base_url: provider_base_url("cerebras").unwrap().into(),
            api_key: e.api_key.clone(),
            is_anthropic: false,
        }),
        "sambanova" => config.sambanova.as_ref().map(|e| ProviderInfo {
            name: "sambanova".into(),
            base_url: provider_base_url("sambanova").unwrap().into(),
            api_key: e.api_key.clone(),
            is_anthropic: false,
        }),
        "fireworks" => config.fireworks.as_ref().map(|e| ProviderInfo {
            name: "fireworks".into(),
            base_url: provider_base_url("fireworks").unwrap().into(),
            api_key: e.api_key.clone(),
            is_anthropic: false,
        }),
        "together" => config.together.as_ref().map(|e| ProviderInfo {
            name: "together".into(),
            base_url: provider_base_url("together").unwrap().into(),
            api_key: e.api_key.clone(),
            is_anthropic: false,
        }),
        "deepinfra" => config.deepinfra.as_ref().map(|e| ProviderInfo {
            name: "deepinfra".into(),
            base_url: provider_base_url("deepinfra").unwrap().into(),
            api_key: e.api_key.clone(),
            is_anthropic: false,
        }),
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
                })
        }
    }
}

/// Lazily-initialized shared reqwest client for provider proxying.
/// Uses a long timeout (5 min) because reasoning models (DeepSeek R1, etc.)
/// can take 60-120s before the first token arrives.
static PROVIDER_CLIENT: std::sync::LazyLock<reqwest::Client> = std::sync::LazyLock::new(|| {
    reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(300))
        .connect_timeout(std::time::Duration::from_secs(30))
        .build()
        .unwrap_or_else(|_| reqwest::Client::new())
});

fn get_provider_client() -> &'static reqwest::Client {
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

    let config = state.shared_state.providers_config.read().await;
    let provider = match resolve_provider(model, &config) {
        Some(p) if !p.is_anthropic => p,
        _ => {
            // Fallback: check provider_model_map (populated by list_provider_models)
            if let Some(entry) = state.shared_state.provider_model_map.get(model) {
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

    let response =
        proxy_openai_compatible(&provider.base_url, &provider.api_key, body, stream).await?;
    Ok(Some(response))
}

/// Generic OpenAI-compatible proxy: rewrite base URL + auth header, forward as-is.
pub async fn proxy_openai_compatible(
    base_url: &str,
    api_key: &str,
    body: &serde_json::Value,
    stream: bool,
) -> Result<axum::response::Response, ApiError> {
    let client = get_provider_client();
    let url = format!("{}/chat/completions", base_url);

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
        let scrubbed_body = crate::crypto::scrub_api_keys(&raw_body);
        tracing::warn!(status = %status, body = %scrubbed_body, "Provider returned error");
        // Try to extract a human-readable message from the provider's JSON error
        let friendly = serde_json::from_str::<serde_json::Value>(&raw_body)
            .ok()
            .and_then(|v| {
                v.get("detail")
                    .or_else(|| v.get("error").and_then(|e| e.get("message")))
                    .or_else(|| v.get("message"))
                    .and_then(|m| m.as_str().map(|s| s.to_string()))
            })
            .unwrap_or(raw_body);
        return Err(ApiError(crate::error::SwarmError::ProviderError {
            status: status.as_u16(),
            body: friendly,
        }));
    }

    if stream {
        let byte_stream = resp.bytes_stream();
        let body = axum::body::Body::from_stream(byte_stream);
        let response = axum::response::Response::builder()
            .header("content-type", "text/event-stream")
            .header("cache-control", "no-cache")
            .body(body)
            .map_err(|e| {
                ApiError(crate::error::SwarmError::Internal(format!(
                    "Failed to build response: {e}"
                )))
            })?;
        Ok(response.into_response())
    } else {
        let body = resp.text().await.unwrap_or_default();
        let response = axum::response::Response::builder()
            .header("content-type", "application/json")
            .body(axum::body::Body::from(body))
            .map_err(|e| {
                ApiError(crate::error::SwarmError::Internal(format!(
                    "Failed to build response: {e}"
                )))
            })?;
        Ok(response.into_response())
    }
}

/// Proxy a request to the Anthropic Messages API.
pub async fn proxy_to_anthropic(
    api_key: &str,
    body: &serde_json::Value,
    stream: bool,
) -> Result<axum::response::Response, ApiError> {
    let client = get_provider_client();
    let url = "https://api.anthropic.com/v1/messages";

    let resp = client
        .post(url)
        .header("x-api-key", api_key)
        .header("anthropic-version", "2023-06-01")
        .header("Content-Type", "application/json")
        .json(body)
        .send()
        .await
        .map_err(|e| {
            tracing::warn!(error = %e, "Anthropic proxy request failed");
            ApiError(crate::error::SwarmError::Internal(format!(
                "Anthropic proxy failed: {e}"
            )))
        })?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        let scrubbed_body = crate::crypto::scrub_api_keys(&body);
        tracing::warn!(status = %status, body = %scrubbed_body, "Anthropic returned error");
        return Err(ApiError(crate::error::SwarmError::Internal(format!(
            "Anthropic returned error status {status}: {scrubbed_body}"
        ))));
    }

    if stream {
        let byte_stream = resp.bytes_stream();
        let body = axum::body::Body::from_stream(byte_stream);
        let response = axum::response::Response::builder()
            .header("content-type", "text/event-stream")
            .header("cache-control", "no-cache")
            .body(body)
            .map_err(|e| {
                ApiError(crate::error::SwarmError::Internal(format!(
                    "Failed to build response: {e}"
                )))
            })?;
        Ok(response.into_response())
    } else {
        let body = resp.text().await.unwrap_or_default();
        let response = axum::response::Response::builder()
            .header("content-type", "application/json")
            .body(axum::body::Body::from(body))
            .map_err(|e| {
                ApiError(crate::error::SwarmError::Internal(format!(
                    "Failed to build response: {e}"
                )))
            })?;
        Ok(response.into_response())
    }
}

/// GET /v1/providers — List configured providers (public, no keys exposed).
pub async fn list_providers(State(state): State<AppState>) -> Json<serde_json::Value> {
    let config = state.shared_state.providers_config.read().await;

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

    #[test]
    fn resolve_claude_to_anthropic() {
        let config = ProvidersConfig {
            anthropic: Some(ProviderEntry {
                api_key: "sk-ant-test".into(),
                default_model: None,
            }),
            ..Default::default()
        };
        let p = resolve_provider("claude-opus-4-6", &config).unwrap();
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
        assert!(resolve_provider("claude-opus-4-6", &config).is_none());
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
        assert_eq!(provider_base_url("unknown"), None);
    }
}
