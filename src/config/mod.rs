use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

use crate::error::SwarmError;
// Re-export ContributionMode from swarmllm-types crate
pub use crate::types::ContributionMode;

/// Hot-reloadable operational parameters that can be changed without restart.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OperationalParams {
    pub max_concurrent_requests: u32,
    pub auto_manage_interval_minutes: u32,
    pub max_batch_size: u32,
    pub max_peers: u32,
    pub session_timeout_secs: u64,
    pub contribution: ContributionMode,
    pub contribution_auto: bool,
    pub max_gpu_vram_mb: u64,
}

impl OperationalParams {
    /// Extract operational params from a full Config.
    pub fn from_config(config: &Config) -> Self {
        Self {
            max_concurrent_requests: config.inference.max_concurrent_requests,
            auto_manage_interval_minutes: config.auto_manage.interval_minutes,
            max_batch_size: config.inference.max_batch_size,
            max_peers: config.network.max_peers,
            session_timeout_secs: config.inference.session_timeout_seconds,
            contribution: config.node.contribution.clone(),
            contribution_auto: config.node.contribution_auto,
            max_gpu_vram_mb: config.resources.max_gpu_vram_mb,
        }
    }
}

/// Reload only operational (hot-reloadable) parameters from the config file.
/// Returns the new params or an error if the file cannot be read/parsed.
pub fn reload_operational_params(config_path: &Path) -> Result<OperationalParams, SwarmError> {
    if !config_path.exists() {
        return Err(SwarmError::Config(format!(
            "Config file not found: {}",
            config_path.display()
        )));
    }
    let contents = std::fs::read_to_string(config_path).map_err(SwarmError::Io)?;
    let config: Config = toml::from_str(&contents).map_err(|e| {
        SwarmError::Config(format!("Failed to parse {}: {e}", config_path.display()))
    })?;
    Ok(OperationalParams::from_config(&config))
}

/// Top-level configuration for the SwarmLLM daemon.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Config {
    #[serde(default)]
    pub node: NodeConfig,
    #[serde(default)]
    pub resources: ResourceConfig,
    #[serde(default)]
    pub network: NetworkConfig,
    #[serde(default)]
    pub inference: InferenceConfig,
    #[serde(default)]
    pub logging: LoggingConfig,
    #[serde(default)]
    pub ui: UiConfig,
    #[serde(default)]
    pub updates: UpdateConfig,
    #[serde(default)]
    pub pool: PoolConfig,
    #[serde(default)]
    pub api: ApiConfig,
    #[serde(default)]
    pub auto_manage: AutoManageConfig,
    #[serde(default)]
    pub model: ModelConfig,
    #[serde(default)]
    pub identity: IdentityConfig,
    #[serde(default)]
    pub providers: ProvidersConfig,
}

/// Provider config types (ProvidersConfig, ProviderEntry, CustomProvider,
/// ProviderKeySource) and the .env loader. Their Debug/Drop impls handle
/// API-key redaction and zeroization.
mod providers;
pub use providers::*;

/// Credit-economy + device-pool config: CreditRateConfig, PoolConfig.
mod credit;
pub use credit::*;

/// Network/transport config: NetworkConfig and is_wsl2 helper.
mod network;
pub use network::*;

/// Operational config: LoggingConfig, UiConfig, UpdateConfig +
/// AutoUpdateMode, ApiConfig.
mod ops;
pub use ops::*;

/// Node + resource + identity config: NodeConfig, ResourceConfig,
/// ResourceSchedule, IdentityConfig + resolve_data_dir helper.
mod node;
pub use node::*;

/// Inference / auto-shard-management / model-storage config:
/// InferenceConfig, AutoManageConfig + ModelAutoManagePolicy,
/// ModelConfig + shard-size constants.
mod inference;
pub use inference::*;

// ---- Defaults ----

/// Parse the `SWARMLLM_NETWORK_BOOTSTRAP_PEERS` env var value into a
/// list of multiaddrs. Splits on commas and any whitespace (newline,
/// tab, space) so users can paste either a comma list or a multi-line
/// `.env` value; empty entries are dropped.
pub(crate) fn parse_bootstrap_peers_env(value: &str) -> Vec<String> {
    value
        .split(|c: char| c == ',' || c.is_whitespace())
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(String::from)
        .collect()
}

/// Read the HuggingFace API token from standard env vars (`HF_TOKEN` preferred,
/// `HUGGING_FACE_HUB_TOKEN` fallback). Returns `None` if unset or empty.
/// Lives here so all external-service credentials flow through `config`.
pub fn hf_api_token() -> Option<String> {
    std::env::var("HF_TOKEN")
        .or_else(|_| std::env::var("HUGGING_FACE_HUB_TOKEN"))
        .ok()
        .filter(|t| !t.is_empty())
}

pub(super) fn default_true() -> bool {
    true
}

// ---- Impl defaults ----

impl Default for Config {
    fn default() -> Self {
        toml::from_str("").expect("empty TOML should parse to defaults")
    }
}

impl Config {
    /// Load config with priority: CLI overrides > env vars > config file > defaults.
    pub fn load_or_create(
        config_path: Option<&Path>,
        cli_port: Option<u16>,
        cli_data_dir: Option<&Path>,
        cli_model_path: Option<&Path>,
        cli_gpu_layers: Option<u32>,
        cli_bootstrap: Vec<String>,
    ) -> Result<Self, SwarmError> {
        tracing::debug!(
            config_path = ?config_path,
            cli_port = ?cli_port,
            cli_data_dir = ?cli_data_dir,
            "DIAG: config load_or_create starting"
        );

        // 1. Start with defaults
        let mut config = Self::default();

        // 1b. Apply data_dir overrides BEFORE config file lookup, so the config
        // is loaded from the correct data directory when SWARMLLM_NODE_DATA_DIR
        // or --data-dir is set. Without this, config.toml is always loaded from
        // the default data dir (~/.local/share/swarmllm/), ignoring per-node
        // overrides and applying stale settings (e.g. enable_autonat=true).
        if let Some(dir) = cli_data_dir {
            config.node.data_dir = dir.to_path_buf();
        } else if let Ok(val) = std::env::var("SWARMLLM_NODE_DATA_DIR") {
            config.node.data_dir = PathBuf::from(val);
        }

        // 2. Load from config file if it exists
        let path = config_path
            .map(PathBuf::from)
            .unwrap_or_else(|| config.node.data_dir.join("config.toml"));

        let mut config_text = String::new();
        if path.exists() {
            let contents = std::fs::read_to_string(&path).map_err(SwarmError::Io)?;
            config_text = contents.clone();
            config = toml::from_str(&contents).map_err(|e| {
                SwarmError::Config(format!("Failed to parse {}: {e}", path.display()))
            })?;
            // Re-apply data_dir overrides since toml::from_str replaces the entire config
            if let Some(dir) = cli_data_dir {
                config.node.data_dir = dir.to_path_buf();
            } else if let Ok(val) = std::env::var("SWARMLLM_NODE_DATA_DIR") {
                config.node.data_dir = PathBuf::from(val);
            }
            tracing::info!(path = %path.display(), "Loaded config");
        }

        // 3. Apply environment variable overrides (SWARMLLM_ prefix)
        if let Ok(val) = std::env::var("SWARMLLM_NODE_LISTEN_PORT") {
            match val.parse() {
                Ok(port) => config.node.listen_port = port,
                Err(e) => tracing::warn!(
                    var = "SWARMLLM_NODE_LISTEN_PORT",
                    value = %val,
                    error = %e,
                    "Failed to parse env var, ignoring"
                ),
            }
        }
        if let Ok(val) = std::env::var("SWARMLLM_NODE_DATA_DIR") {
            config.node.data_dir = PathBuf::from(val);
        }
        if let Ok(val) = std::env::var("SWARMLLM_LOGGING_LEVEL") {
            match val.as_str() {
                "trace" | "debug" | "info" | "warn" | "error" => config.logging.level = val,
                _ => tracing::warn!(
                    value = %val,
                    "Ignoring invalid SWARMLLM_LOGGING_LEVEL (expected: trace/debug/info/warn/error)"
                ),
            }
        }
        if let Ok(val) = std::env::var("SWARMLLM_INFERENCE_MODEL_PATH") {
            config.inference.model_path = Some(PathBuf::from(val));
        }
        if let Ok(val) = std::env::var("SWARMLLM_INFERENCE_GPU_LAYERS") {
            match val.parse() {
                Ok(n) => config.inference.gpu_layers = n,
                Err(e) => tracing::warn!(
                    var = "SWARMLLM_INFERENCE_GPU_LAYERS",
                    value = %val,
                    error = %e,
                    "Failed to parse env var, ignoring"
                ),
            }
        }
        // Bearer token for the HTTP API. Advertised in `.env.example` for
        // Docker deployments where users need a deterministic key (vs. the
        // auto-generated 32-byte one) without baking it into config.toml.
        // Empty string is treated as "unset" so commenting `SWARMLLM_API_KEY=`
        // doesn't blank the persisted key.
        if let Ok(val) = std::env::var("SWARMLLM_API_KEY") {
            if !val.is_empty() {
                config.api.api_key = Some(val);
            }
        }
        // Bootstrap peers as a comma-, space-, or newline-separated list of
        // multiaddrs. Mirrors `--bootstrap` on the CLI for headless
        // deployments. Empty entries are filtered.
        if let Ok(val) = std::env::var("SWARMLLM_NETWORK_BOOTSTRAP_PEERS") {
            let peers = parse_bootstrap_peers_env(&val);
            if !peers.is_empty() {
                config.network.bootstrap_peers = peers;
            }
        }

        // 4. Apply CLI overrides (highest priority)
        if let Some(port) = cli_port {
            config.node.listen_port = port;
        }
        if let Some(dir) = cli_data_dir {
            config.node.data_dir = dir.to_path_buf();
        }
        if let Some(path) = cli_model_path {
            config.inference.model_path = Some(path.to_path_buf());
        }
        if let Some(n) = cli_gpu_layers {
            config.inference.gpu_layers = n;
        }
        if !cli_bootstrap.is_empty() {
            config.network.bootstrap_peers = cli_bootstrap;
        }

        // 5. Auto-detect WSL2 and apply safe network defaults.
        // Only overrides values the user didn't explicitly set in config.toml.
        if network::is_wsl2() {
            // Parse TOML into a Value to check which keys were explicitly set.
            // Raw string search (e.g., config_text.contains("enable_quic")) would
            // match commented-out keys or keys in string values — false positives.
            let explicit_network_keys: std::collections::HashSet<String> =
                toml::from_str::<toml::Value>(&config_text)
                    .ok()
                    .and_then(|v| v.get("network").and_then(|n| n.as_table().cloned()))
                    .map(|t| t.keys().cloned().collect())
                    .unwrap_or_default();
            let has = |key: &str| explicit_network_keys.contains(key);
            let net = &mut config.network;
            let mut adapted = Vec::new();
            if !has("enable_quic") {
                net.enable_quic = false;
                adapted.push("enable_quic=false");
            }
            if !has("enable_autonat") {
                net.enable_autonat = false;
                adapted.push("enable_autonat=false");
            }
            if !has("enable_dcutr") {
                net.enable_dcutr = false;
                adapted.push("enable_dcutr=false");
            }
            if !has("enable_mdns") {
                net.enable_mdns = false;
                adapted.push("enable_mdns=false");
            }
            if !has("listen_address") {
                net.listen_address = "127.0.0.1".to_string();
                adapted.push("listen_address=127.0.0.1");
            }
            if !adapted.is_empty() {
                tracing::info!(
                    settings = adapted.join(", "),
                    "WSL2 detected: auto-applied safe network defaults (set explicitly in config.toml to override)"
                );
            }
        }

        // Validate
        config.validate()?;

        tracing::debug!(
            port = config.node.listen_port,
            data_dir = %config.node.data_dir.display(),
            "DIAG: config load_or_create complete"
        );

        Ok(config)
    }

    fn validate(&self) -> Result<(), SwarmError> {
        // Reject port 0 (DAE-M13)
        if self.node.listen_port == 0 {
            return Err(SwarmError::Config("listen_port must not be 0".to_string()));
        }

        // Warn on privileged ports (not fatal — user may have permissions)
        if self.node.listen_port < 1024 {
            tracing::warn!(
                port = self.node.listen_port,
                "Using privileged port — may require elevated permissions"
            );
        }

        // Validate inference config (DAE-I4)
        if self.inference.max_concurrent_requests < 1 {
            return Err(SwarmError::Config(
                "max_concurrent_requests must be >= 1".to_string(),
            ));
        }
        if self.inference.session_timeout_seconds == 0 {
            return Err(SwarmError::Config(
                "session_timeout_seconds must be > 0".to_string(),
            ));
        }
        if self.inference.max_batch_size == 0 {
            return Err(SwarmError::Config(
                "max_batch_size must be >= 1".to_string(),
            ));
        }
        if self.inference.speculative_gamma == 0 {
            return Err(SwarmError::Config(
                "speculative_gamma must be > 0".to_string(),
            ));
        }
        if !(0.0..1.0).contains(&self.inference.swift_skip_ratio) {
            return Err(SwarmError::Config(format!(
                "swift_skip_ratio must be in [0.0, 1.0) (got {}); a value >= 1.0 would cause empty layer ranges in the SWIFT draft pass",
                self.inference.swift_skip_ratio
            )));
        }
        if !(0.0..=1.0).contains(&self.inference.cross_node_prefix_trust_min) {
            return Err(SwarmError::Config(format!(
                "cross_node_prefix_trust_min must be in [0.0, 1.0] (got {}); values outside this range either trust everyone or no one",
                self.inference.cross_node_prefix_trust_min
            )));
        }
        if self.network.relay_max_circuits == 0 {
            return Err(SwarmError::Config(
                "relay_max_circuits must be > 0".to_string(),
            ));
        }
        if self.resources.schedule.reduced_hours_start > 23 {
            return Err(SwarmError::Config(
                "reduced_hours_start must be 0-23".to_string(),
            ));
        }
        if self.resources.schedule.reduced_hours_end > 23 {
            return Err(SwarmError::Config(
                "reduced_hours_end must be 0-23".to_string(),
            ));
        }

        // Validate model path exists if specified
        if let Some(ref path) = self.inference.model_path {
            if !path.exists() {
                return Err(SwarmError::Config(format!(
                    "Model file not found: {}",
                    path.display()
                )));
            }
        }

        // Validate model/shard config
        self.model.validate()?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_is_valid() {
        let config = Config::default();
        assert_eq!(config.node.listen_port, 8800);
        assert_eq!(config.resources.max_disk_mb, 50_000);
        assert_eq!(config.inference.session_timeout_seconds, 600);
    }

    #[test]
    fn parse_toml_with_overrides() {
        let toml_str = r#"
[node]
listen_port = 9000

[inference]
max_concurrent_requests = 5
"#;
        let config: Config = toml::from_str(toml_str).unwrap();
        assert_eq!(config.node.listen_port, 9000);
        assert_eq!(config.inference.max_concurrent_requests, 5);
        // Defaults still apply for unset fields
        assert_eq!(config.resources.max_disk_mb, 50_000);
    }

    #[test]
    fn load_from_nonexistent_path_uses_defaults() {
        let config = Config::load_or_create(
            Some(Path::new("/nonexistent/config.toml")),
            None,
            None,
            None,
            None,
            vec![],
        )
        .unwrap();
        assert_eq!(config.node.listen_port, 8800);
    }

    #[test]
    fn cli_overrides_take_priority() {
        let config = Config::load_or_create(
            Some(Path::new("/nonexistent")),
            Some(9999),
            None,
            None,
            None,
            vec![],
        )
        .unwrap();
        assert_eq!(config.node.listen_port, 9999);
    }

    #[test]
    fn config_parses_with_missing_sections() {
        // A minimal config (only [node]) must deserialize and rely on
        // serde defaults for every other section. This is the back-
        // compat invariant: old config.toml files written before new
        // sections existed must still load.
        let toml_str = r#"
[node]
listen_port = 8800
"#;
        let config: Config = toml::from_str(toml_str).unwrap();
        assert_eq!(config.node.listen_port, 8800);
        // Sections that didn't exist in the file should fall back to
        // their Default impls, not panic.
        assert_eq!(config.resources.max_disk_mb, 50_000);
        assert_eq!(config.inference.max_concurrent_requests, 10);
    }

    #[test]
    fn config_ignores_unknown_top_level_field() {
        // `serde(default)` + non-`deny_unknown_fields` means a future
        // version of swarmllm that adds new sections is forward-compat
        // when read by an older binary. Verify behavior.
        let toml_str = r#"
[node]
listen_port = 8800

[future_section_we_dont_know_about]
some_setting = "value"
"#;
        // toml::from_str must not error on the unknown section. If a
        // future change adds #[serde(deny_unknown_fields)] to Config,
        // this test will fire and the author must update the contract
        // (or document the breakage).
        let config: Config = toml::from_str(toml_str).unwrap();
        assert_eq!(config.node.listen_port, 8800);
    }

    #[test]
    fn config_old_field_names_still_parse() {
        // Sanity: a minimal file with only [api] api_key set parses
        // and the old field name is preserved through round-trip.
        // Catches accidental rename/removal of public config fields.
        let toml_str = r#"
[api]
api_key = "test-key-abc"
"#;
        let config: Config = toml::from_str(toml_str).unwrap();
        assert_eq!(config.api.api_key.as_deref(), Some("test-key-abc"));
    }

    #[test]
    fn bootstrap_peers_env_parses_commas_and_whitespace() {
        // Comma-separated.
        let parsed = parse_bootstrap_peers_env(
            "/ip4/1.1.1.1/udp/8800/quic-v1,/ip4/2.2.2.2/udp/8800/quic-v1",
        );
        assert_eq!(
            parsed,
            vec![
                "/ip4/1.1.1.1/udp/8800/quic-v1".to_string(),
                "/ip4/2.2.2.2/udp/8800/quic-v1".to_string(),
            ]
        );

        // Whitespace and newlines (multi-line .env value).
        let parsed = parse_bootstrap_peers_env(
            "/ip4/1.1.1.1/udp/8800/quic-v1\n/ip4/2.2.2.2/udp/8800/quic-v1",
        );
        assert_eq!(parsed.len(), 2);

        // Mixed separators with empty entries.
        let parsed = parse_bootstrap_peers_env(", ,/ip4/1.1.1.1/udp/8800/quic-v1, , ,");
        assert_eq!(parsed, vec!["/ip4/1.1.1.1/udp/8800/quic-v1".to_string()]);

        // Empty string yields no peers.
        assert!(parse_bootstrap_peers_env("").is_empty());
        assert!(parse_bootstrap_peers_env("   \n  ").is_empty());
    }

    #[test]
    fn default_shard_size_mb() {
        let config = Config::default();
        assert_eq!(config.model.shard_size_mb, 512);
        assert_eq!(config.model.shard_size_bytes(), 512 * 1024 * 1024);
    }

    #[test]
    fn parse_custom_shard_size() {
        let toml_str = r#"
[model]
shard_size_mb = 256
"#;
        let config: Config = toml::from_str(toml_str).unwrap();
        assert_eq!(config.model.shard_size_mb, 256);
        assert_eq!(config.model.shard_size_bytes(), 256 * 1024 * 1024);
    }

    #[test]
    fn shard_size_validation_rejects_too_small() {
        let model_config = ModelConfig { shard_size_mb: 32 };
        assert!(model_config.validate().is_err());
    }

    #[test]
    fn shard_size_validation_rejects_too_large() {
        let model_config = ModelConfig {
            shard_size_mb: 4096,
        };
        assert!(model_config.validate().is_err());
    }

    #[test]
    fn shard_size_validation_accepts_valid() {
        let model_config = ModelConfig { shard_size_mb: 512 };
        assert!(model_config.validate().is_ok());
        let model_config = ModelConfig { shard_size_mb: 64 };
        assert!(model_config.validate().is_ok());
        let model_config = ModelConfig {
            shard_size_mb: 2048,
        };
        assert!(model_config.validate().is_ok());
    }

    #[test]
    fn shard_size_non_power_of_two_warns_but_passes() {
        let model_config = ModelConfig { shard_size_mb: 300 };
        // Should succeed (only a warning, not an error)
        assert!(model_config.validate().is_ok());
    }

    #[test]
    fn identity_region_defaults_to_none() {
        let config = Config::default();
        assert!(config.identity.region.is_none());
    }

    #[test]
    fn parse_identity_region() {
        let toml_str = r#"
[identity]
region = "US"
"#;
        let config: Config = toml::from_str(toml_str).unwrap();
        assert_eq!(config.identity.region.as_deref(), Some("US"));
    }

    #[test]
    fn auto_relay_defaults_to_true() {
        let config = Config::default();
        assert!(config.network.auto_relay);
    }

    #[test]
    fn parse_auto_relay_disabled() {
        let toml_str = r#"
[network]
auto_relay = false
"#;
        let config: Config = toml::from_str(toml_str).unwrap();
        assert!(!config.network.auto_relay);
    }

    #[test]
    fn max_concurrent_downloads_defaults_to_3() {
        let config = Config::default();
        assert_eq!(config.auto_manage.max_concurrent_downloads, 3);
    }

    #[test]
    fn operational_params_from_config() {
        let config = Config::default();
        let params = OperationalParams::from_config(&config);
        assert_eq!(params.max_concurrent_requests, 10);
        assert_eq!(params.auto_manage_interval_minutes, 5);
        assert_eq!(params.max_batch_size, 1);
        assert_eq!(params.max_peers, 200);
        assert_eq!(params.session_timeout_secs, 600);
        assert!(params.contribution_auto);
        assert_eq!(params.max_gpu_vram_mb, 0);
    }

    #[test]
    fn operational_params_equality() {
        let config = Config::default();
        let p1 = OperationalParams::from_config(&config);
        let p2 = OperationalParams::from_config(&config);
        assert_eq!(p1, p2);
    }

    #[test]
    fn reload_operational_params_from_file() {
        let dir = tempfile::tempdir().unwrap();
        let config_path = dir.path().join("config.toml");
        let toml_str = r#"
[inference]
max_concurrent_requests = 20
max_batch_size = 4
session_timeout_seconds = 300

[auto_manage]
interval_minutes = 10

[network]
max_peers = 100
"#;
        std::fs::write(&config_path, toml_str).unwrap();
        let params = reload_operational_params(&config_path).unwrap();
        assert_eq!(params.max_concurrent_requests, 20);
        assert_eq!(params.max_batch_size, 4);
        assert_eq!(params.session_timeout_secs, 300);
        assert_eq!(params.auto_manage_interval_minutes, 10);
        assert_eq!(params.max_peers, 100);
    }

    #[test]
    fn reload_operational_params_missing_file() {
        let result = reload_operational_params(Path::new("/nonexistent/config.toml"));
        assert!(result.is_err());
    }

    #[test]
    fn reload_operational_params_partial_config() {
        let dir = tempfile::tempdir().unwrap();
        let config_path = dir.path().join("config.toml");
        // Only override one field — rest should use defaults
        let toml_str = r#"
[inference]
max_concurrent_requests = 42
"#;
        std::fs::write(&config_path, toml_str).unwrap();
        let params = reload_operational_params(&config_path).unwrap();
        assert_eq!(params.max_concurrent_requests, 42);
        // Defaults for others
        assert_eq!(params.max_batch_size, 1);
        assert_eq!(params.max_peers, 200);
    }

    #[test]
    fn vram_budget_explicit_cap() {
        let rc = ResourceConfig {
            max_gpu_vram_mb: 4000,
            ..Default::default()
        };
        assert_eq!(rc.inference_vram_budget_mb(8000), Some(4000));
    }

    #[test]
    fn vram_budget_auto_80_percent() {
        let rc = ResourceConfig::default(); // max_gpu_vram_mb = 0
        assert_eq!(rc.inference_vram_budget_mb(8000), Some(6400));
    }

    #[test]
    fn vram_budget_no_gpu() {
        let rc = ResourceConfig::default();
        assert_eq!(rc.inference_vram_budget_mb(0), None);
    }
}
