use axum::extract::State;
use axum::Json;
use serde::Deserialize;

use crate::api::server::AppState;
use crate::error::ApiError;

/// Timeout for fetching model lists from provider APIs.
const PROVIDER_LIST_TIMEOUT_SECS: u64 = 5;
/// Total timeout for model availability probes (includes inference).
const PROVIDER_PROBE_TIMEOUT_SECS: u64 = 10;
/// Connect timeout for model availability probes.
const PROVIDER_PROBE_CONNECT_SECS: u64 = 5;
/// Total timeout for lightweight model health checks.
const PROVIDER_HEALTH_TIMEOUT_SECS: u64 = 5;
/// Connect timeout for lightweight model health checks.
const PROVIDER_HEALTH_CONNECT_SECS: u64 = 3;

// ── Cloud Provider Management ──

/// GET /api/admin/providers — List configured provider status (no keys exposed).
pub async fn get_providers(State(state): State<AppState>) -> Json<serde_json::Value> {
    let config = state.shared_state.metrics.providers_config.read().await;

    let entries: &[(&str, &Option<crate::config::ProviderEntry>)] = &[
        ("anthropic", &config.anthropic),
        ("openai", &config.openai),
        ("deepseek", &config.deepseek),
        ("mistral", &config.mistral),
        ("groq", &config.groq),
        ("nvidia_nim", &config.nvidia_nim),
        ("cerebras", &config.cerebras),
        ("sambanova", &config.sambanova),
        ("fireworks", &config.fireworks),
        ("together", &config.together),
        ("deepinfra", &config.deepinfra),
        ("moonshot", &config.moonshot),
    ];

    let providers: Vec<_> = entries
        .iter()
        .map(|(name, entry)| {
            let source = if entry.is_some() && config.env_sourced.contains(*name) {
                "env"
            } else if entry.is_some() {
                "config"
            } else {
                "none"
            };
            serde_json::json!({
                "name": name,
                "configured": entry.is_some(),
                "source": source,
            })
        })
        .collect();

    let key_source = match config.key_source {
        crate::config::ProviderKeySource::Auto => "auto",
        crate::config::ProviderKeySource::Env => "env",
        crate::config::ProviderKeySource::Dashboard => "dashboard",
    };

    #[allow(unused_mut)]
    let mut result = serde_json::json!({ "providers": providers, "key_source": key_source });

    // Claude subscription status (feature-gated)
    #[cfg(feature = "claude-subscription")]
    {
        if let Some(ref sub_config) = config.claude_subscription {
            result["claude_subscription"] = serde_json::json!({
                "enabled": sub_config.enabled,
                "binary": sub_config.binary(),
            });
        }
    }

    Json(result)
}

#[derive(Debug, Deserialize)]
pub struct ProvidersUpdate {
    #[serde(default)]
    pub anthropic_key: Option<String>,
    #[serde(default)]
    pub openai_key: Option<String>,
    #[serde(default)]
    pub deepseek_key: Option<String>,
    #[serde(default)]
    pub mistral_key: Option<String>,
    #[serde(default)]
    pub groq_key: Option<String>,
    #[serde(default)]
    pub nvidia_nim_key: Option<String>,
    #[serde(default)]
    pub cerebras_key: Option<String>,
    #[serde(default)]
    pub sambanova_key: Option<String>,
    #[serde(default)]
    pub fireworks_key: Option<String>,
    #[serde(default)]
    pub together_key: Option<String>,
    #[serde(default)]
    pub deepinfra_key: Option<String>,
    #[serde(default)]
    pub moonshot_key: Option<String>,
    /// Key source mode: "auto", "env", or "dashboard".
    #[serde(default)]
    pub key_source: Option<String>,
    /// Claude subscription: enable/disable (feature-gated).
    #[serde(default)]
    pub claude_subscription_enabled: Option<bool>,
}

/// PUT /api/admin/providers — Update provider API keys. Empty string = remove key.
pub async fn update_providers(
    State(state): State<AppState>,
    Json(body): Json<ProvidersUpdate>,
) -> Result<Json<serde_json::Value>, ApiError> {
    // Validate all keys before applying any changes
    let all_keys: &[(&str, &Option<String>)] = &[
        ("anthropic", &body.anthropic_key),
        ("openai", &body.openai_key),
        ("deepseek", &body.deepseek_key),
        ("mistral", &body.mistral_key),
        ("groq", &body.groq_key),
        ("nvidia_nim", &body.nvidia_nim_key),
        ("cerebras", &body.cerebras_key),
        ("sambanova", &body.sambanova_key),
        ("fireworks", &body.fireworks_key),
        ("together", &body.together_key),
        ("deepinfra", &body.deepinfra_key),
        ("moonshot", &body.moonshot_key),
    ];
    for (name, key) in all_keys {
        if let Some(k) = key {
            if let Err(e) = crate::crypto::validate_api_key(k) {
                return Err(ApiError(crate::error::SwarmError::Validation(format!(
                    "Invalid API key for {name}: {e}"
                ))));
            }
        }
    }

    // Build updated config on a clone — only hold the read lock briefly to snapshot,
    // then release before encryption + DB write to avoid blocking inference readers.
    let mut new_config = state
        .shared_state
        .metrics
        .providers_config
        .read()
        .await
        .clone();

    fn update_entry(entry: &mut Option<crate::config::ProviderEntry>, key: Option<String>) {
        if let Some(k) = key {
            if k.is_empty() {
                *entry = None;
            } else {
                *entry = Some(crate::config::ProviderEntry {
                    api_key: k,
                    default_model: entry.as_ref().and_then(|e| e.default_model.clone()),
                });
            }
        }
    }

    update_entry(&mut new_config.anthropic, body.anthropic_key);
    update_entry(&mut new_config.openai, body.openai_key);
    update_entry(&mut new_config.deepseek, body.deepseek_key);
    update_entry(&mut new_config.mistral, body.mistral_key);
    update_entry(&mut new_config.groq, body.groq_key);
    update_entry(&mut new_config.nvidia_nim, body.nvidia_nim_key);
    update_entry(&mut new_config.cerebras, body.cerebras_key);
    update_entry(&mut new_config.sambanova, body.sambanova_key);
    update_entry(&mut new_config.fireworks, body.fireworks_key);
    update_entry(&mut new_config.together, body.together_key);
    update_entry(&mut new_config.deepinfra, body.deepinfra_key);
    update_entry(&mut new_config.moonshot, body.moonshot_key);

    // Update claude subscription toggle (feature-gated)
    #[cfg(feature = "claude-subscription")]
    if let Some(enabled) = body.claude_subscription_enabled {
        let sub = new_config
            .claude_subscription
            .get_or_insert_with(Default::default);
        sub.enabled = enabled;
        tracing::info!(
            enabled,
            "Claude subscription provider toggled via admin API"
        );
        let msg = if enabled {
            "Claude Code subscription provider enabled"
        } else {
            "Claude Code subscription provider disabled"
        };
        state.shared_state.emit_activity(
            crate::daemon::state::ActivityEvent::new(
                "provider",
                "claude_subscription",
                msg.to_string(),
            )
            .with_toast(if enabled { "info" } else { "warning" }, 3000),
        );
    }

    // Update key source mode if provided
    if let Some(ref ks) = body.key_source {
        new_config.key_source = match ks.as_str() {
            "auto" => crate::config::ProviderKeySource::Auto,
            "env" => crate::config::ProviderKeySource::Env,
            "dashboard" => crate::config::ProviderKeySource::Dashboard,
            _ => {
                return Err(ApiError(crate::error::SwarmError::Validation(format!(
                    "invalid key_source '{ks}': must be 'auto', 'env', or 'dashboard'"
                ))));
            }
        };
        // Re-apply env vars with the new mode
        new_config.fill_from_env();
    }

    // Encrypt and persist BEFORE committing to in-memory state.
    // No write lock held during encryption + DB write — avoids blocking inference readers.
    let signing_key_bytes = state.shared_state.identity.signing_key_bytes();
    match crate::crypto::encrypt_config(&new_config, &signing_key_bytes) {
        Ok(encrypted) => {
            let _ = state
                .shared_state
                .db
                .put_json("providers", "config", &encrypted);
        }
        Err(e) => {
            tracing::error!(error = %e, "Failed to encrypt provider keys for storage — keys NOT saved");
            return Err(ApiError(crate::error::SwarmError::Internal(
                "Failed to encrypt provider configuration".into(),
            )));
        }
    }

    // Build response before moving new_config into the write guard
    let response = serde_json::json!({
        "status": "ok",
        "anthropic": new_config.anthropic.is_some(),
        "openai": new_config.openai.is_some(),
        "deepseek": new_config.deepseek.is_some(),
        "mistral": new_config.mistral.is_some(),
        "groq": new_config.groq.is_some(),
        "nvidia_nim": new_config.nvidia_nim.is_some(),
        "cerebras": new_config.cerebras.is_some(),
        "sambanova": new_config.sambanova.is_some(),
        "fireworks": new_config.fireworks.is_some(),
        "together": new_config.together.is_some(),
        "deepinfra": new_config.deepinfra.is_some(),
        "moonshot": new_config.moonshot.is_some(),
    });

    // Persist succeeded — briefly acquire write lock to commit to in-memory state
    *state.shared_state.metrics.providers_config.write().await = new_config;

    tracing::info!(target: "swarmllm::api::admin_providers", "Cloud provider configuration updated");

    // Invalidate provider models cache so next fetch picks up new keys
    {
        let mut cache = state
            .shared_state
            .metrics
            .provider_models_cache
            .write()
            .await;
        cache.0.clear();
    }

    // Notify WebSocket clients so model list and mode indicator refresh immediately
    let _ = state
        .shared_state
        .events
        .dashboard_tx
        .send(crate::daemon::state::DashboardSignal::ModelsChanged);

    Ok(Json(response))
}

/// GET /api/admin/provider-models — Fetch available models from configured providers.
///
/// Returns cached results instantly if available (< 60s old), and refreshes
/// in the background. On first call (empty cache), blocks until fetch completes.
/// This prevents slow/flaky provider APIs from making the dashboard feel broken.
pub async fn list_provider_models(State(state): State<AppState>) -> Json<serde_json::Value> {
    const CACHE_TTL: std::time::Duration = std::time::Duration::from_secs(60);

    // Check cache — return immediately if fresh
    {
        let cache = state
            .shared_state
            .metrics
            .provider_models_cache
            .read()
            .await;
        let (ref cached_models, ref ts) = *cache;
        if !cached_models.is_empty() {
            if ts.elapsed() < CACHE_TTL {
                return Json(serde_json::json!({ "models": cached_models }));
            }
            // Stale but non-empty: return stale data and refresh in background
            let stale = cached_models.clone();
            let bg_state = state.clone();
            tokio::spawn(async move {
                let models = fetch_provider_models_inner(&bg_state).await;
                if !models.is_empty() {
                    let mut cache = bg_state
                        .shared_state
                        .metrics
                        .provider_models_cache
                        .write()
                        .await;
                    *cache = (models, std::time::Instant::now());
                }
            });
            return Json(serde_json::json!({ "models": stale }));
        }
    }

    // Empty cache (first call) — block and fetch
    let models = fetch_provider_models_inner(&state).await;
    {
        let mut cache = state
            .shared_state
            .metrics
            .provider_models_cache
            .write()
            .await;
        *cache = (models.clone(), std::time::Instant::now());
    }
    Json(serde_json::json!({ "models": models }))
}

/// Inner function that actually fetches models from all configured providers.
async fn fetch_provider_models_inner(state: &AppState) -> Vec<serde_json::Value> {
    let config = state.shared_state.metrics.providers_config.read().await;
    let mut models = Vec::new();

    // Collect (provider_name, base_url, api_key, needs_prefix) for all configured providers
    let mut fetch_tasks: Vec<(&str, String, String, bool)> = Vec::new();

    // Collect configured OpenAI-compatible providers for parallel /models fetch.
    // needs_prefix: if true, model IDs are prefixed with "provider:" for routing.
    let candidates: &[(&str, Option<&crate::config::ProviderEntry>, bool)] = &[
        ("openai", config.openai.as_ref(), false),
        ("deepseek", config.deepseek.as_ref(), false),
        ("mistral", config.mistral.as_ref(), false),
        ("groq", config.groq.as_ref(), false),
        ("nvidia_nim", config.nvidia_nim.as_ref(), false),
        ("cerebras", config.cerebras.as_ref(), true),
        ("sambanova", config.sambanova.as_ref(), true),
        ("fireworks", config.fireworks.as_ref(), false),
        ("together", config.together.as_ref(), true),
        ("deepinfra", config.deepinfra.as_ref(), true),
        ("moonshot", config.moonshot.as_ref(), true),
    ];

    for &(name, ref entry, needs_prefix) in candidates {
        if let Some(e) = entry {
            if let Some(base) = crate::api::providers::provider_base_url(name) {
                fetch_tasks.push((name, base.to_string(), e.api_key.clone(), needs_prefix));
            }
        }
    }

    // Anthropic has no /models endpoint — use static list
    if config.anthropic.is_some() {
        for (id, name) in [
            ("claude-opus-4-6", "Claude Opus 4.6"),
            ("claude-sonnet-4-6", "Claude Sonnet 4.6"),
            ("claude-haiku-4-5-20251001", "Claude Haiku 4.5"),
        ] {
            models.push(serde_json::json!({
                "id": id, "name": name, "provider": "anthropic"
            }));
        }
    }

    // Moonshot/Kimi static fallback — common models that may not appear in /models
    if config.moonshot.is_some() {
        for (id, name) in [
            ("moonshot:kimi-k2-0527", "Kimi K2 (Kimi 2.5)"),
            ("moonshot:moonshot-v1-8k", "Moonshot v1 8K"),
            ("moonshot:moonshot-v1-32k", "Moonshot v1 32K"),
            ("moonshot:moonshot-v1-128k", "Moonshot v1 128K"),
        ] {
            // Only add if not already present from /models fetch
            if !models
                .iter()
                .any(|m| m.get("id").and_then(|v| v.as_str()) == Some(id))
            {
                models.push(serde_json::json!({
                    "id": id, "name": name, "provider": "moonshot"
                }));
            }
        }
    }

    // Collect custom provider models before dropping config
    for custom in &config.custom {
        if let Some(ref model) = custom.default_model {
            models.push(serde_json::json!({
                "id": format!("{}:{}", custom.name, model),
                "name": model,
                "provider": custom.name,
            }));
        }
    }

    // Claude subscription: add Claude models when enabled (uses CLI, not API key)
    #[cfg(feature = "claude-subscription")]
    if let Some(ref sub_config) = config.claude_subscription {
        if sub_config.enabled {
            let provider_label = "claude_subscription";
            for (id, name, ctx) in [
                ("claude-opus-4-6", "Claude Opus 4.6", "1M"),
                ("claude-sonnet-4-6", "Claude Sonnet 4.6", "200K"),
                ("claude-haiku-4-5-20251001", "Claude Haiku 4.5", "200K"),
            ] {
                // Don't duplicate if already present from Anthropic API key
                if !models
                    .iter()
                    .any(|m| m.get("id").and_then(|v| v.as_str()) == Some(id))
                {
                    models.push(serde_json::json!({
                        "id": id,
                        "name": name,
                        "provider": provider_label,
                        "meta": {
                            "context_length": ctx,
                            "source": "subscription",
                        },
                    }));
                }
            }
        }
    }

    drop(config);

    // Fetch models from all OpenAI-compatible providers in parallel
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(PROVIDER_LIST_TIMEOUT_SECS))
        .build()
        .unwrap_or_else(|_| reqwest::Client::new());

    let fetches =
        fetch_tasks
            .into_iter()
            .map(|(provider_name, base_url, api_key, needs_prefix)| {
                let client = client.clone();
                async move {
                    let url = format!("{}/models", base_url);
                    let resp = client
                        .get(&url)
                        .header("Authorization", format!("Bearer {}", api_key))
                        .send()
                        .await;

                    match resp {
                        Ok(r) if r.status().is_success() => {
                            if let Ok(body) = r.json::<serde_json::Value>().await {
                                // OpenAI /models returns { "data": [ { "id": "...", ... }, ... ] }
                                if let Some(data) = body.get("data").and_then(|d| d.as_array()) {
                                    let mut result = Vec::new();
                                    for m in data {
                                        if let Some(id) = m.get("id").and_then(|v| v.as_str()) {
                                            let display = id.rsplit('/').next().unwrap_or(id);
                                            let routed_id = if needs_prefix {
                                                format!("{}:{}", provider_name, id)
                                            } else {
                                                id.to_string()
                                            };
                                            // Pass through extra metadata from the provider.
                                            // Skip standard fields we already handle (id, object, owned_by).
                                            let mut meta = serde_json::Map::new();
                                            if let Some(obj) = m.as_object() {
                                                for (k, v) in obj {
                                                    match k.as_str() {
                                                        "id" | "object" => {}
                                                        _ => {
                                                            meta.insert(k.clone(), v.clone());
                                                        }
                                                    }
                                                }
                                            }
                                            let mut entry = serde_json::json!({
                                                "id": routed_id,
                                                "name": display,
                                                "provider": provider_name,
                                            });
                                            if !meta.is_empty() {
                                                entry["meta"] = serde_json::Value::Object(meta);
                                            }
                                            result.push(entry);
                                        }
                                    }
                                    return (provider_name, result);
                                }
                            }
                            (provider_name, Vec::new())
                        }
                        _ => {
                            tracing::debug!(
                                provider = provider_name,
                                "Failed to fetch /models, no fallback"
                            );
                            (provider_name, Vec::new())
                        }
                    }
                }
            });

    let results = futures::future::join_all(fetches).await;
    // Clear stale entries before repopulating — prevents misdirecting requests
    // to models removed from provider catalogs between refresh cycles.
    state.shared_state.metrics.provider_model_map.clear();
    for (provider, provider_models) in &results {
        for m in provider_models {
            if let Some(id) = m.get("id").and_then(|v| v.as_str()) {
                state
                    .shared_state
                    .metrics
                    .provider_model_map
                    .insert(id.to_string(), provider.to_string());
            }
        }
    }
    for (_provider, provider_models) in results {
        models.extend(provider_models);
    }

    models
}

/// GET /api/admin/provider-health — Lightweight health probe for configured providers.
///
/// Sends a tiny chat completion request (max_tokens=1) to one model per provider
/// to measure latency and confirm availability. Returns per-provider status.
pub async fn provider_health(State(state): State<AppState>) -> Json<serde_json::Value> {
    let config = state.shared_state.metrics.providers_config.read().await;

    // Build (provider_name, base_url, api_key, test_model) tuples
    let mut probes: Vec<(&str, String, String, String)> = Vec::new();

    let candidates: &[(&str, Option<&crate::config::ProviderEntry>, &str)] = &[
        ("openai", config.openai.as_ref(), "gpt-4o-mini"),
        (
            "anthropic",
            config.anthropic.as_ref(),
            "claude-haiku-4-5-20251001",
        ),
        ("deepseek", config.deepseek.as_ref(), "deepseek-chat"),
        ("mistral", config.mistral.as_ref(), "mistral-small-latest"),
        ("groq", config.groq.as_ref(), "llama-3.1-8b-instant"),
        (
            "nvidia_nim",
            config.nvidia_nim.as_ref(),
            "meta/llama-3.1-8b-instruct",
        ),
        ("cerebras", config.cerebras.as_ref(), "llama-3.3-70b"),
        (
            "sambanova",
            config.sambanova.as_ref(),
            "Meta-Llama-3.1-8B-Instruct",
        ),
        (
            "fireworks",
            config.fireworks.as_ref(),
            "accounts/fireworks/models/llama-v3p1-8b-instruct",
        ),
        (
            "together",
            config.together.as_ref(),
            "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        ),
        (
            "deepinfra",
            config.deepinfra.as_ref(),
            "meta-llama/Meta-Llama-3.1-8B-Instruct",
        ),
        ("moonshot", config.moonshot.as_ref(), "moonshot-v1-8k"),
    ];

    for &(name, ref entry, test_model) in candidates {
        if let Some(e) = entry {
            if let Some(base) = crate::api::providers::provider_base_url(name) {
                probes.push((
                    name,
                    base.to_string(),
                    e.api_key.clone(),
                    test_model.to_string(),
                ));
            }
        }
    }

    drop(config);

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(PROVIDER_PROBE_TIMEOUT_SECS))
        .connect_timeout(std::time::Duration::from_secs(PROVIDER_PROBE_CONNECT_SECS))
        .build()
        .unwrap_or_else(|_| reqwest::Client::new());

    let probes_futures = probes
        .into_iter()
        .map(|(name, base_url, api_key, test_model)| {
            let client = client.clone();
            async move {
                let url = if name == "anthropic" {
                    format!("{}/messages", base_url)
                } else {
                    format!("{}/chat/completions", base_url)
                };

                let body = if name == "anthropic" {
                    serde_json::json!({
                        "model": test_model,
                        "max_tokens": 1,
                        "messages": [{"role": "user", "content": "hi"}]
                    })
                } else {
                    serde_json::json!({
                        "model": test_model,
                        "max_tokens": 1,
                        "messages": [{"role": "user", "content": "hi"}],
                        "stream": false
                    })
                };

                let start = std::time::Instant::now();
                let result = if name == "anthropic" {
                    client
                        .post(&url)
                        .header("x-api-key", &api_key)
                        .header("anthropic-version", "2023-06-01")
                        .header("content-type", "application/json")
                        .json(&body)
                        .send()
                        .await
                } else {
                    client
                        .post(&url)
                        .header("authorization", format!("Bearer {}", api_key))
                        .header("content-type", "application/json")
                        .json(&body)
                        .send()
                        .await
                };
                let latency_ms = start.elapsed().as_millis() as u64;

                match result {
                    Ok(resp) => {
                        let status_code = resp.status().as_u16();
                        let (status, detail) = if resp.status().is_success() {
                            ("up".to_string(), String::new())
                        } else if status_code == 401 || status_code == 403 {
                            ("auth_error".to_string(), "Invalid API key".to_string())
                        } else if status_code == 429 {
                            ("rate_limited".to_string(), "Rate limited".to_string())
                        } else if status_code == 503 || status_code == 502 {
                            ("overloaded".to_string(), "Service overloaded".to_string())
                        } else {
                            let body_text = resp.text().await.unwrap_or_default();
                            // SEC: Scrub potential API keys from upstream error responses
                            let scrubbed = crate::crypto::scrub_api_keys(&body_text);
                            let short = if scrubbed.len() > 100 {
                                scrubbed[..100].to_string()
                            } else {
                                scrubbed
                            };
                            (format!("error_{}", status_code), short)
                        };
                        serde_json::json!({
                            "provider": name,
                            "status": status,
                            "latency_ms": latency_ms,
                            "detail": detail,
                        })
                    }
                    Err(e) => {
                        let status = if e.is_timeout() {
                            "timeout"
                        } else if e.is_connect() {
                            "unreachable"
                        } else {
                            "error"
                        };
                        // Don't leak internal error details — use only the categorized status
                        let detail = match status {
                            "timeout" => "Connection timed out".to_string(),
                            "unreachable" => "Could not connect to provider".to_string(),
                            _ => "Provider health check failed".to_string(),
                        };
                        serde_json::json!({
                            "provider": name,
                            "status": status,
                            "latency_ms": latency_ms,
                            "detail": detail,
                        })
                    }
                }
            }
        });

    let results = futures::future::join_all(probes_futures).await;
    Json(serde_json::json!({ "providers": results }))
}

/// POST /api/admin/provider-model-status — Probe availability of specific cloud models.
///
/// Accepts `{ "models": ["model-id-1", "model-id-2", ...] }`.
/// Sends a tiny max_tokens=1 request to each model with a 5s timeout.
/// Returns per-model status (up/timeout/error) and latency.
/// Capped at 20 models per request to prevent abuse.
#[derive(Deserialize)]
pub struct ModelStatusRequest {
    models: Vec<String>,
}

pub async fn provider_model_status(
    State(state): State<AppState>,
    Json(body): Json<ModelStatusRequest>,
) -> Json<serde_json::Value> {
    let config = state.shared_state.metrics.providers_config.read().await;
    let models: Vec<String> = body.models.into_iter().take(20).collect();

    // Resolve provider for each model
    let mut probes: Vec<(String, String, String)> = Vec::new(); // (model_id, base_url, api_key)
    for model_id in &models {
        if let Some(p) = crate::api::providers::resolve_provider(model_id, &config) {
            if !p.is_anthropic {
                probes.push((model_id.clone(), p.base_url.clone(), p.api_key.clone()));
            }
        } else if let Some(entry) = state
            .shared_state
            .metrics
            .provider_model_map
            .get(model_id.as_str())
        {
            let pname = entry.value().clone();
            if let Some(p) = crate::api::providers::resolve_by_name(&pname, &config) {
                if !p.is_anthropic {
                    probes.push((model_id.clone(), p.base_url.clone(), p.api_key.clone()));
                }
            }
        }
    }
    drop(config);

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(PROVIDER_HEALTH_TIMEOUT_SECS))
        .connect_timeout(std::time::Duration::from_secs(PROVIDER_HEALTH_CONNECT_SECS))
        .build()
        .unwrap_or_else(|_| reqwest::Client::new());

    let futures = probes.into_iter().map(|(model_id, base_url, api_key)| {
        let client = client.clone();
        async move {
            // SEC: Validate provider URL to prevent SSRF via custom providers
            if let Err(e) = super::providers::validate_provider_url(&base_url).await {
                return serde_json::json!({
                    "model": model_id,
                    "status": "error",
                    "error": format!("SSRF blocked: {e}"),
                });
            }
            let url = format!("{}/chat/completions", base_url);
            let body = serde_json::json!({
                "model": model_id,
                "max_tokens": 1,
                "messages": [{"role": "user", "content": "hi"}],
                "stream": false
            });
            let start = std::time::Instant::now();
            let result = client
                .post(&url)
                .header("authorization", format!("Bearer {}", api_key))
                .header("content-type", "application/json")
                .json(&body)
                .send()
                .await;
            let latency_ms = start.elapsed().as_millis() as u64;

            match result {
                Ok(resp) => {
                    let code = resp.status().as_u16();
                    let status = if resp.status().is_success() {
                        "up"
                    } else if code == 429 {
                        "rate_limited"
                    } else if code == 404 {
                        "not_found"
                    } else if code == 503 || code == 502 {
                        "unavailable"
                    } else {
                        "error"
                    };
                    serde_json::json!({
                        "model": model_id,
                        "status": status,
                        "latency_ms": latency_ms,
                    })
                }
                Err(e) => {
                    let status = if e.is_timeout() { "timeout" } else { "error" };
                    serde_json::json!({
                        "model": model_id,
                        "status": status,
                        "latency_ms": latency_ms,
                    })
                }
            }
        }
    });

    let results = futures::future::join_all(futures).await;
    Json(serde_json::json!({ "models": results }))
}

// ========================================================================
// Update / Version Endpoints
// ========================================================================

/// GET /api/admin/version — Current and latest version info.
pub async fn version_info(State(state): State<AppState>) -> Json<serde_json::Value> {
    let update_state = state.shared_state.events.update_state.read().await;
    let current_version = env!("CARGO_PKG_VERSION");

    let (latest_version, update_available, changelog) =
        if let Some(ref info) = update_state.update_available {
            (
                Some(info.latest_version.clone()),
                true,
                Some(info.changelog.clone()),
            )
        } else {
            (None, false, None)
        };

    let channel = match state.shared_state.config.updates.auto_update {
        crate::config::AutoUpdateMode::Disabled => "disabled",
        crate::config::AutoUpdateMode::Stable => "stable",
        crate::config::AutoUpdateMode::All => "all",
    };

    Json(serde_json::json!({
        "current_version": current_version,
        "latest_version": latest_version,
        "update_available": update_available,
        "channel": channel,
        "last_checked": update_state.last_checked,
        "last_error": update_state.last_error,
        "changelog": changelog,
    }))
}

/// POST /api/admin/update/check — Trigger an immediate update check.
pub async fn check_update(
    State(state): State<AppState>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let config = state.shared_state.config.updates.clone();
    let update_state = state.shared_state.events.update_state.clone();
    let dash_tx = state.shared_state.events.dashboard_tx.clone();

    let checker = crate::update::UpdateChecker::new(
        config,
        "enapt/SwarmLLM".to_string(),
        update_state.clone(),
        dash_tx,
    );

    match checker.check_for_update().await {
        Ok(Some(info)) => {
            // Auto-download
            let mut info = info;
            if let Ok(tmp_path) = checker.download_update(&info).await {
                info.downloaded = true;
                let _ = tmp_path; // path is known from binary location
            }
            let mut us = update_state.write().await;
            us.update_available = Some(info.clone());
            us.last_checked = Some(chrono::Utc::now().to_rfc3339());
            us.last_error = None;
            // Notify WebSocket
            let _ = state.shared_state.events.dashboard_tx.send(
                crate::daemon::state::DashboardSignal::UpdateAvailable(info.clone()),
            );
            Ok(Json(serde_json::json!({
                "status": "update_available",
                "info": info,
            })))
        }
        Ok(None) => {
            let mut us = update_state.write().await;
            us.last_checked = Some(chrono::Utc::now().to_rfc3339());
            us.last_error = None;
            Ok(Json(serde_json::json!({
                "status": "up_to_date",
                "current_version": env!("CARGO_PKG_VERSION"),
            })))
        }
        Err(e) => {
            let mut us = update_state.write().await;
            us.last_checked = Some(chrono::Utc::now().to_rfc3339());
            us.last_error = Some(e.to_string());
            Err(ApiError(e))
        }
    }
}

/// POST /api/admin/update/apply — Apply a downloaded update (restart required).
pub async fn apply_update(
    State(state): State<AppState>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let update_state = state.shared_state.events.update_state.read().await;
    let info = match &update_state.update_available {
        Some(info) if info.downloaded => info.clone(),
        Some(_) => {
            return Err(ApiError(crate::error::SwarmError::Validation(
                "Update not yet downloaded — call POST /api/admin/update/check first".into(),
            )));
        }
        None => {
            return Err(ApiError(crate::error::SwarmError::Validation(
                "No update available".into(),
            )));
        }
    };
    drop(update_state);

    let config = state.shared_state.config.updates.clone();
    let (tx, _) = tokio::sync::broadcast::channel(1);
    let checker = crate::update::UpdateChecker::new(
        config,
        "enapt/SwarmLLM".to_string(),
        state.shared_state.events.update_state.clone(),
        tx,
    );

    let binary_path = std::env::current_exe().map_err(|e| {
        ApiError(crate::error::SwarmError::Internal(format!(
            "Cannot determine binary path: {e}"
        )))
    })?;
    let tmp_path = binary_path.with_extension("update.tmp");

    if !tmp_path.exists() {
        return Err(ApiError(crate::error::SwarmError::Validation(
            "Downloaded update file not found — re-run update check first".to_string(),
        )));
    }

    checker.apply_update(&tmp_path).map_err(ApiError)?;

    Ok(Json(serde_json::json!({
        "status": "applied",
        "version": info.latest_version,
        "message": "Update applied. Restart the daemon to use the new version.",
    })))
}
