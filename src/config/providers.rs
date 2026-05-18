//! Cloud provider configuration.
//!
//! Defines the provider-config types (`ProvidersConfig`, `ProviderEntry`,
//! `CustomProvider`, `ProviderKeySource`) and their security-sensitive
//! `Debug`/`Drop` impls — `Debug` redacts the api_key, and `Drop` zeroizes
//! it via the `zeroize` crate. Also hosts `load_dotenv`, the `.env` loader
//! used at daemon startup to seed `OPENAI_API_KEY` etc. into the process
//! environment with a hard blocklist of dangerous names (LD_*, PATH, GIT_,
//! etc.) that could otherwise be exploited if the data dir's `.env` file is
//! writable by an attacker.

use serde::{Deserialize, Serialize};

/// Controls how provider API keys are sourced.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProviderKeySource {
    /// Dashboard/database keys take priority; env vars fill gaps (default).
    #[default]
    Auto,
    /// Environment variables / .env file always override dashboard keys.
    Env,
    /// Only use dashboard-entered keys; ignore environment variables entirely.
    Dashboard,
}

/// Cloud provider configuration for multi-provider API gateway.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ProvidersConfig {
    /// Controls key source priority: auto (db > env), env (env > db), dashboard (db only).
    #[serde(default)]
    pub key_source: ProviderKeySource,
    #[serde(default)]
    pub anthropic: Option<ProviderEntry>,
    #[serde(default)]
    pub openai: Option<ProviderEntry>,
    #[serde(default)]
    pub deepseek: Option<ProviderEntry>,
    #[serde(default)]
    pub mistral: Option<ProviderEntry>,
    #[serde(default)]
    pub groq: Option<ProviderEntry>,
    #[serde(default)]
    pub nvidia_nim: Option<ProviderEntry>,
    #[serde(default)]
    pub cerebras: Option<ProviderEntry>,
    #[serde(default)]
    pub sambanova: Option<ProviderEntry>,
    #[serde(default)]
    pub fireworks: Option<ProviderEntry>,
    #[serde(default)]
    pub together: Option<ProviderEntry>,
    #[serde(default)]
    pub deepinfra: Option<ProviderEntry>,
    #[serde(default)]
    pub moonshot: Option<ProviderEntry>,
    #[serde(default)]
    pub custom: Vec<CustomProvider>,

    /// Claude subscription: proxy through locally-authenticated `claude` CLI subprocess.
    #[cfg(feature = "claude-subscription")]
    #[serde(default)]
    pub claude_subscription: Option<crate::api::claude_sub::ClaudeSubscriptionConfig>,

    /// Tracks which providers were loaded from environment variables / .env file.
    /// Not persisted — runtime only. Used by the UI to show source attribution.
    #[serde(skip)]
    pub env_sourced: std::collections::HashSet<String>,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct ProviderEntry {
    pub api_key: String,
    #[serde(default)]
    pub default_model: Option<String>,
}

impl std::fmt::Debug for ProviderEntry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProviderEntry")
            .field("api_key", &"[REDACTED]")
            .field("default_model", &self.default_model)
            .finish()
    }
}

impl Drop for ProviderEntry {
    fn drop(&mut self) {
        zeroize::Zeroize::zeroize(&mut self.api_key);
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct CustomProvider {
    pub name: String,
    pub base_url: String,
    pub api_key: String,
    #[serde(default)]
    pub default_model: Option<String>,
}

impl std::fmt::Debug for CustomProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CustomProvider")
            .field("name", &self.name)
            .field("base_url", &self.base_url)
            .field("api_key", &"[REDACTED]")
            .field("default_model", &self.default_model)
            .finish()
    }
}

impl Drop for CustomProvider {
    fn drop(&mut self) {
        zeroize::Zeroize::zeroize(&mut self.api_key);
    }
}

impl ProvidersConfig {
    /// Standard env var name → provider name mappings.
    const ENV_MAPPINGS: &[(&str, &str)] = &[
        ("OPENAI_API_KEY", "openai"),
        ("ANTHROPIC_API_KEY", "anthropic"),
        ("DEEPSEEK_API_KEY", "deepseek"),
        ("MISTRAL_API_KEY", "mistral"),
        ("GROQ_API_KEY", "groq"),
        ("NVIDIA_NIM_API_KEY", "nvidia_nim"),
        ("CEREBRAS_API_KEY", "cerebras"),
        ("SAMBANOVA_API_KEY", "sambanova"),
        ("FIREWORKS_API_KEY", "fireworks"),
        ("TOGETHER_API_KEY", "together"),
        ("DEEPINFRA_API_KEY", "deepinfra"),
        ("MOONSHOT_API_KEY", "moonshot"),
    ];

    /// Apply environment variable keys according to `key_source` mode.
    /// - `Auto`: env fills gaps (only where no key is set)
    /// - `Env`: env always overwrites existing keys
    /// - `Dashboard`: env vars ignored entirely
    pub fn fill_from_env(&mut self) {
        if self.key_source == ProviderKeySource::Dashboard {
            return;
        }
        let force = self.key_source == ProviderKeySource::Env;

        for (env_var, name) in Self::ENV_MAPPINGS {
            if let Ok(key) = std::env::var(env_var) {
                let key = key.trim().to_string();
                if key.is_empty() {
                    continue;
                }
                let field = self.field_mut(name);
                if force || field.is_none() {
                    if field.is_some() && force {
                        tracing::info!(env_var, "Environment key overriding dashboard key");
                    } else {
                        tracing::info!(env_var, "Loaded provider key from environment");
                    }
                    *field = Some(ProviderEntry {
                        api_key: key,
                        default_model: field.as_ref().and_then(|e| e.default_model.clone()),
                    });
                    self.env_sourced.insert(name.to_string());
                }
            }
        }

        // Auto-detect Claude CLI and initialize subscription config if not already set.
        // This ensures the dashboard shows the Claude Code card on first startup.
        #[cfg(feature = "claude-subscription")]
        if self.claude_subscription.is_none()
            && std::process::Command::new("claude")
                .arg("--version")
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .status()
                .is_ok()
        {
            {
                tracing::info!("Auto-detected Claude CLI — enabling subscription provider");
                self.claude_subscription = Some(crate::api::claude_sub::ClaudeSubscriptionConfig {
                    enabled: true,
                    ..Default::default()
                });
            }
        }
    }

    /// Check which env vars are available (for setup detection).
    pub fn detect_env_keys() -> Vec<(&'static str, &'static str)> {
        Self::ENV_MAPPINGS
            .iter()
            .filter(|(env_var, _)| {
                std::env::var(env_var)
                    .map(|v| !v.trim().is_empty())
                    .unwrap_or(false)
            })
            .copied()
            .collect()
    }

    fn field_mut(&mut self, name: &str) -> &mut Option<ProviderEntry> {
        match name {
            "openai" => &mut self.openai,
            "anthropic" => &mut self.anthropic,
            "deepseek" => &mut self.deepseek,
            "mistral" => &mut self.mistral,
            "groq" => &mut self.groq,
            "nvidia_nim" => &mut self.nvidia_nim,
            "cerebras" => &mut self.cerebras,
            "sambanova" => &mut self.sambanova,
            "fireworks" => &mut self.fireworks,
            "together" => &mut self.together,
            "deepinfra" => &mut self.deepinfra,
            "moonshot" => &mut self.moonshot,
            other => unreachable!("unknown provider: {other} — update field_mut() match arms"),
        }
    }

    /// R137 (closes R72 deferral): the 12-provider name list was repeated
    /// 4×+ across `api/admin_providers.rs`. This is the canonical iteration
    /// over keyed cloud providers — call sites destructure to whichever
    /// data they need (name only, name+entry, name+is_some, etc).
    /// `custom` and `claude_subscription` are deliberately NOT included
    /// here — they have different shapes and shouldn't share this path.
    pub fn keyed_entries(&self) -> [(&'static str, &Option<ProviderEntry>); 12] {
        [
            ("anthropic", &self.anthropic),
            ("openai", &self.openai),
            ("deepseek", &self.deepseek),
            ("mistral", &self.mistral),
            ("groq", &self.groq),
            ("nvidia_nim", &self.nvidia_nim),
            ("cerebras", &self.cerebras),
            ("sambanova", &self.sambanova),
            ("fireworks", &self.fireworks),
            ("together", &self.together),
            ("deepinfra", &self.deepinfra),
            ("moonshot", &self.moonshot),
        ]
    }

    /// R137: stable list of keyed-provider names. Matches `keyed_entries`
    /// order. Useful for response-shape building where you need just the
    /// name + a derived value (e.g. `is_some` bool map).
    pub const PROVIDER_NAMES: &'static [&'static str] = &[
        "anthropic",
        "openai",
        "deepseek",
        "mistral",
        "groq",
        "nvidia_nim",
        "cerebras",
        "sambanova",
        "fireworks",
        "together",
        "deepinfra",
        "moonshot",
    ];
}

/// Load a `.env` file into the process environment.
/// Searches in order: `<data_dir>/.env`, `./.env`.
/// Lines are `KEY=VALUE` or `KEY="VALUE"` format. Blank lines and `#` comments are skipped.
/// Does NOT override existing environment variables.
pub fn load_dotenv(data_dir: &std::path::Path) {
    let candidates = [data_dir.join(".env"), std::path::PathBuf::from(".env")];
    for path in &candidates {
        if path.is_file() {
            if let Ok(content) = std::fs::read_to_string(path) {
                let mut count = 0;
                for line in content.lines() {
                    let line = line.trim();
                    if line.is_empty() || line.starts_with('#') {
                        continue;
                    }
                    if let Some((key, val)) = line.split_once('=') {
                        let key = key.trim();
                        let val = val.trim();
                        // Strip surrounding quotes
                        let val = val
                            .strip_prefix('"')
                            .and_then(|v| v.strip_suffix('"'))
                            .or_else(|| val.strip_prefix('\'').and_then(|v| v.strip_suffix('\'')))
                            .unwrap_or(val);
                        // SEC: Only allow known-safe env vars from .env file.
                        // Dangerous vars like LD_PRELOAD, PATH, LD_LIBRARY_PATH could
                        // be used for code execution if .env file is writable.
                        const BLOCKED_ENV_PREFIXES: &[&str] = &[
                            "LD_",
                            "DYLD_",
                            "PATH",
                            "HOME",
                            "USER",
                            "SHELL",
                            "PYTHONPATH",
                            "RUBYLIB",
                            "PERL5LIB",
                            "NODE_PATH",
                            "CARGO",
                            "RUSTFLAGS",
                            "CC",
                            "CXX",
                            "CFLAGS",
                            "LDFLAGS",
                            "http_proxy",
                            "https_proxy",
                            "HTTP_PROXY",
                            "HTTPS_PROXY",
                            "no_proxy",
                            "NO_PROXY",
                            "ALL_PROXY",
                            // SEC: Additional dangerous env vars that can execute code
                            "BASH_ENV",
                            "ENV",
                            "IFS",
                            "CDPATH",
                            "PROMPT_COMMAND",
                            "BASH_FUNC",
                            "GCONV_PATH",
                            "GETCONF_DIR",
                            "HOSTALIASES",
                            "RESOLV_HOST_CONF",
                            "RES_OPTIONS",
                            "TMPDIR",
                            "TMP",
                            "TEMP",
                            "EDITOR",
                            "VISUAL",
                            "GIT_",
                            "SSL_CERT",
                            "OPENSSL_",
                            "CURL_",
                            "MALLOC_",
                            "LOCALDOMAIN",
                        ];
                        let key_upper = key.to_uppercase();
                        let is_blocked = BLOCKED_ENV_PREFIXES.iter().any(|prefix| {
                            let p_upper = prefix.to_uppercase();
                            // For prefixes that already end in '_' (e.g. "LD_", "DYLD_"),
                            // check starts_with directly. For exact names, also check
                            // with trailing underscore to block sub-variants.
                            key_upper == p_upper
                                || key_upper.starts_with(&p_upper)
                                || key_upper
                                    .starts_with(&format!("{}_", p_upper.trim_end_matches('_')))
                        });
                        if is_blocked {
                            tracing::warn!(key = key, "Blocked dangerous env var from .env file");
                            continue;
                        }
                        if !key.is_empty() && std::env::var(key).is_err() {
                            std::env::set_var(key, val);
                            count += 1;
                        }
                    }
                }
                tracing::info!(
                    path = %path.display(),
                    vars_loaded = count,
                    "Loaded .env file"
                );
                return; // Use first .env found
            }
        }
    }
}
