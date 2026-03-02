use axum::extract::State;
use axum::response::IntoResponse;
use axum::Json;

use crate::api::server::AppState;
use crate::config::ProvidersConfig;
use crate::error::ApiError;

/// Known provider base URLs (OpenAI-compatible).
fn provider_base_url(name: &str) -> Option<&'static str> {
    match name {
        "openai" => Some("https://api.openai.com/v1"),
        "deepseek" => Some("https://api.deepseek.com/v1"),
        "mistral" => Some("https://api.mistral.ai/v1"),
        "groq" => Some("https://api.groq.com/openai/v1"),
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
    if lower.starts_with("llama-") && config.groq.is_some() {
        // Groq is popular for fast Llama inference
        return resolve_by_name("groq", config);
    }
    if lower.starts_with("gemma") && config.groq.is_some() {
        return resolve_by_name("groq", config);
    }

    // Custom providers only match via explicit `provider:model` syntax (handled above)

    None
}

fn resolve_by_name(name: &str, config: &ProvidersConfig) -> Option<ProviderInfo> {
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
static PROVIDER_CLIENT: std::sync::OnceLock<reqwest::Client> = std::sync::OnceLock::new();

fn get_provider_client() -> &'static reqwest::Client {
    PROVIDER_CLIENT.get_or_init(|| {
        reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(120))
            .build()
            .unwrap_or_else(|_| reqwest::Client::new())
    })
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
        _ => return Ok(None),
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
            ApiError(crate::error::SwarmError::Internal(format!(
                "Provider proxy failed: {e}"
            )))
        })?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        tracing::warn!(status = %status, body = %body, "Provider returned error");
        return Err(ApiError(crate::error::SwarmError::Internal(format!(
            "Provider returned error status {status}: {body}"
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
        tracing::warn!(status = %status, body = %body, "Anthropic returned error");
        return Err(ApiError(crate::error::SwarmError::Internal(format!(
            "Anthropic returned error status {status}: {body}"
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
                name: "together".into(),
                base_url: "https://api.together.xyz/v1".into(),
                api_key: "tok-test".into(),
                default_model: None,
            }],
            ..Default::default()
        };
        let p = resolve_provider("together:meta-llama/Llama-3-70b", &config).unwrap();
        assert_eq!(p.name, "together");
        assert_eq!(p.base_url, "https://api.together.xyz/v1");
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
        assert_eq!(provider_base_url("unknown"), None);
    }
}
