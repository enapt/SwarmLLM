use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

use crate::error::SwarmError;

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
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NodeConfig {
    #[serde(default = "default_data_dir")]
    pub data_dir: PathBuf,
    #[serde(default = "default_port")]
    pub listen_port: u16,
    #[serde(default)]
    pub contribution: ContributionMode,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ContributionMode {
    Minimal,
    #[default]
    Moderate,
    Maximum,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ResourceConfig {
    #[serde(default)]
    pub max_gpu_vram_mb: u64,
    #[serde(default)]
    pub max_ram_mb: u64,
    #[serde(default = "default_max_disk")]
    pub max_disk_mb: u64,
    #[serde(default)]
    pub max_bandwidth_mbps: u64,
    #[serde(default)]
    pub schedule: ResourceSchedule,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ResourceSchedule {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "default_reduced_hours_start")]
    pub reduced_hours_start: u32,
    #[serde(default = "default_reduced_hours_end")]
    pub reduced_hours_end: u32,
    #[serde(default = "default_reduced_contribution")]
    pub reduced_contribution: String,
}

impl Default for ResourceSchedule {
    fn default() -> Self {
        Self {
            enabled: false,
            reduced_hours_start: default_reduced_hours_start(),
            reduced_hours_end: default_reduced_hours_end(),
            reduced_contribution: default_reduced_contribution(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NetworkConfig {
    #[serde(default)]
    pub bootstrap_peers: Vec<String>,
    #[serde(default = "default_true")]
    pub peer_exchange: bool,
    #[serde(default = "default_true")]
    pub enable_relay: bool,
    #[serde(default = "default_true")]
    pub enable_relay_client: bool,
    #[serde(default = "default_max_peers")]
    pub max_peers: u32,
    /// Maximum duration for a single relay circuit in seconds.
    #[serde(default = "default_relay_circuit_duration")]
    pub relay_max_circuit_duration_secs: u64,
    /// Maximum number of relay circuits this node will serve simultaneously.
    #[serde(default = "default_relay_max_circuits")]
    pub relay_max_circuits: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InferenceConfig {
    #[serde(default)]
    pub default_model: String,
    #[serde(default = "default_session_timeout")]
    pub session_timeout_seconds: u64,
    #[serde(default = "default_max_concurrent")]
    pub max_concurrent_requests: u32,
    #[serde(default)]
    pub model_path: Option<PathBuf>,
    #[serde(default = "default_gpu_layers")]
    pub gpu_layers: u32,
    /// KV-cache session TTL in seconds (default 600 = 10 minutes).
    #[serde(default = "default_kv_cache_ttl")]
    pub kv_cache_ttl_secs: Option<u64>,
    /// Enable speculative decoding when a valid draft-target model pair is available.
    #[serde(default)]
    pub speculative_decoding: bool,
    /// Number of draft tokens to propose per verification step (default: 4).
    #[serde(default = "default_speculative_gamma")]
    pub speculative_gamma: u32,
    /// Optional shard range for split inference (e.g. "0-4").
    /// When set, the node only claims these shard indices instead of all shards.
    #[serde(skip)]
    pub shard_range: Option<(u32, u32)>,
    /// Maximum number of requests to batch together for inference.
    /// Default 1 means no batching (sequential, backward-compatible).
    #[serde(default = "default_max_batch_size")]
    pub max_batch_size: u32,
    /// How long (ms) to wait for additional requests before dispatching a partial batch.
    #[serde(default = "default_batch_timeout_ms")]
    pub batch_timeout_ms: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LoggingConfig {
    #[serde(default = "default_log_level")]
    pub level: String,
    #[serde(default = "default_log_format")]
    pub format: String,
    #[serde(default)]
    pub file: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct UiConfig {
    #[serde(default = "default_true")]
    pub open_browser_on_start: bool,
    #[serde(default = "default_theme")]
    pub theme: String,
}

impl Default for UiConfig {
    fn default() -> Self {
        Self {
            open_browser_on_start: true,
            theme: default_theme(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct UpdateConfig {
    #[serde(default = "default_auto_update")]
    pub auto_update: AutoUpdateMode,
    #[serde(default = "default_check_interval_hours")]
    pub check_interval_hours: u32,
    #[serde(default = "default_true")]
    pub auto_restart: bool,
    #[serde(default = "default_keep_versions")]
    pub keep_versions: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum AutoUpdateMode {
    Disabled,
    Stable,
    All,
}

impl Default for UpdateConfig {
    fn default() -> Self {
        Self {
            auto_update: AutoUpdateMode::Stable,
            check_interval_hours: default_check_interval_hours(),
            auto_restart: true,
            keep_versions: default_keep_versions(),
        }
    }
}

fn default_auto_update() -> AutoUpdateMode {
    AutoUpdateMode::Stable
}

fn default_check_interval_hours() -> u32 {
    6
}

fn default_keep_versions() -> u32 {
    3
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PoolConfig {
    #[serde(default = "default_max_pool_size")]
    pub max_pool_size: u32,
    #[serde(default = "default_invitation_ttl_hours")]
    pub invitation_ttl_hours: u32,
    #[serde(default = "default_pool_rate_limit")]
    pub rate_limit_per_hour: u32,
    #[serde(default = "default_pool_gossip_interval")]
    pub gossip_interval_secs: u64,
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            max_pool_size: default_max_pool_size(),
            invitation_ttl_hours: default_invitation_ttl_hours(),
            rate_limit_per_hour: default_pool_rate_limit(),
            gossip_interval_secs: default_pool_gossip_interval(),
        }
    }
}

fn default_max_pool_size() -> u32 {
    10
}

fn default_invitation_ttl_hours() -> u32 {
    24
}

fn default_pool_rate_limit() -> u32 {
    3
}

fn default_pool_gossip_interval() -> u64 {
    600
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ApiConfig {
    /// Bearer token for API authentication. If empty, one is auto-generated on first run.
    #[serde(default)]
    pub api_key: Option<String>,
}

/// Configuration for automatic shard management.
///
/// When enabled, the node periodically evaluates network shard coverage
/// and downloads rarest shards for popular models — filling gaps to
/// improve overall network availability.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AutoManageConfig {
    /// Master toggle for auto shard management.
    #[serde(default)]
    pub enabled: bool,
    /// Maximum disk space (MB) the auto-manager may use for shard storage.
    /// Defaults to the global `max_disk_mb` if 0.
    #[serde(default)]
    pub max_storage_mb: u64,
    /// How often (in minutes) the auto-manager evaluates and downloads.
    #[serde(default = "default_auto_manage_interval")]
    pub interval_minutes: u32,
    /// Maximum number of shards to hold at once (0 = unlimited within disk budget).
    #[serde(default)]
    pub max_shards: u32,
}

impl Default for AutoManageConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            max_storage_mb: 0,
            interval_minutes: default_auto_manage_interval(),
            max_shards: 0,
        }
    }
}

fn default_auto_manage_interval() -> u32 {
    60
}

/// Configuration for model storage and sharding.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Size of each shard in megabytes when splitting a model for distribution.
    /// Must be between 64 and 2048 (inclusive). Default: 512.
    /// Changing this only affects newly created shards — existing shards keep their original size.
    #[serde(default = "default_shard_size_mb")]
    pub shard_size_mb: u64,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            shard_size_mb: default_shard_size_mb(),
        }
    }
}

fn default_shard_size_mb() -> u64 {
    512
}

/// Minimum allowed shard size in MB.
pub const SHARD_SIZE_MIN_MB: u64 = 64;
/// Maximum allowed shard size in MB.
pub const SHARD_SIZE_MAX_MB: u64 = 2048;

impl ModelConfig {
    /// Return the configured shard size in bytes.
    pub fn shard_size_bytes(&self) -> u64 {
        self.shard_size_mb * 1024 * 1024
    }

    /// Validate and clamp shard_size_mb to allowed range.
    pub fn validate(&self) -> Result<(), SwarmError> {
        if self.shard_size_mb < SHARD_SIZE_MIN_MB || self.shard_size_mb > SHARD_SIZE_MAX_MB {
            return Err(SwarmError::Config(format!(
                "shard_size_mb must be between {} and {} (got {})",
                SHARD_SIZE_MIN_MB, SHARD_SIZE_MAX_MB, self.shard_size_mb
            )));
        }
        if !self.shard_size_mb.is_power_of_two() {
            tracing::warn!(
                shard_size_mb = self.shard_size_mb,
                "shard_size_mb is not a power of 2 — this may cause suboptimal alignment"
            );
        }
        Ok(())
    }
}

/// Identity configuration (voluntary self-reported metadata).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct IdentityConfig {
    /// Optional ISO 3166-1 alpha-2 country code (e.g. "US", "DE", "JP").
    /// Voluntarily self-reported; used for the network map visualization.
    #[serde(default)]
    pub region: Option<String>,
}

fn default_theme() -> String {
    "dark".into()
}

// ---- Defaults ----

fn default_data_dir() -> PathBuf {
    dirs::data_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("swarmllm")
}

fn default_port() -> u16 {
    8800
}

fn default_max_disk() -> u64 {
    50_000
}

fn default_true() -> bool {
    true
}

fn default_max_peers() -> u32 {
    200
}

fn default_session_timeout() -> u64 {
    600
}

fn default_max_concurrent() -> u32 {
    10
}

fn default_gpu_layers() -> u32 {
    0
}

fn default_kv_cache_ttl() -> Option<u64> {
    Some(600)
}

fn default_speculative_gamma() -> u32 {
    4
}

fn default_max_batch_size() -> u32 {
    1
}

fn default_batch_timeout_ms() -> u64 {
    50
}

fn default_relay_circuit_duration() -> u64 {
    3600
}

fn default_relay_max_circuits() -> usize {
    16
}

fn default_reduced_hours_start() -> u32 {
    22
}

fn default_reduced_hours_end() -> u32 {
    8
}

fn default_reduced_contribution() -> String {
    "minimal".into()
}

fn default_log_level() -> String {
    "info".into()
}

fn default_log_format() -> String {
    "pretty".into()
}

// ---- Impl defaults ----

impl Default for Config {
    fn default() -> Self {
        toml::from_str("").expect("empty TOML should parse to defaults")
    }
}

impl Default for NodeConfig {
    fn default() -> Self {
        Self {
            data_dir: default_data_dir(),
            listen_port: default_port(),
            contribution: ContributionMode::default(),
        }
    }
}

impl Default for ResourceConfig {
    fn default() -> Self {
        Self {
            max_gpu_vram_mb: 0,
            max_ram_mb: 0,
            max_disk_mb: default_max_disk(),
            max_bandwidth_mbps: 0,
            schedule: ResourceSchedule::default(),
        }
    }
}

impl Default for NetworkConfig {
    fn default() -> Self {
        Self {
            bootstrap_peers: vec![],
            peer_exchange: true,
            enable_relay: true,
            enable_relay_client: true,
            max_peers: default_max_peers(),
            relay_max_circuit_duration_secs: default_relay_circuit_duration(),
            relay_max_circuits: default_relay_max_circuits(),
        }
    }
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            default_model: String::new(),
            session_timeout_seconds: default_session_timeout(),
            max_concurrent_requests: default_max_concurrent(),
            model_path: None,
            gpu_layers: default_gpu_layers(),
            kv_cache_ttl_secs: default_kv_cache_ttl(),
            speculative_decoding: false,
            speculative_gamma: default_speculative_gamma(),
            shard_range: None,
            max_batch_size: default_max_batch_size(),
            batch_timeout_ms: default_batch_timeout_ms(),
        }
    }
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: default_log_level(),
            format: default_log_format(),
            file: None,
        }
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
        // 1. Start with defaults
        let mut config = Self::default();

        // 2. Load from config file if it exists
        let path = config_path
            .map(PathBuf::from)
            .unwrap_or_else(|| config.node.data_dir.join("config.toml"));

        if path.exists() {
            let contents = std::fs::read_to_string(&path).map_err(SwarmError::Io)?;
            config = toml::from_str(&contents).map_err(|e| {
                SwarmError::Config(format!("Failed to parse {}: {e}", path.display()))
            })?;
            tracing::info!(path = %path.display(), "Loaded config");
        }

        // 3. Apply environment variable overrides (SWARMLLM_ prefix)
        if let Ok(val) = std::env::var("SWARMLLM_NODE_LISTEN_PORT") {
            if let Ok(port) = val.parse() {
                config.node.listen_port = port;
            }
        }
        if let Ok(val) = std::env::var("SWARMLLM_NODE_DATA_DIR") {
            config.node.data_dir = PathBuf::from(val);
        }
        if let Ok(val) = std::env::var("SWARMLLM_LOGGING_LEVEL") {
            config.logging.level = val;
        }
        if let Ok(val) = std::env::var("SWARMLLM_INFERENCE_MODEL_PATH") {
            config.inference.model_path = Some(PathBuf::from(val));
        }
        if let Ok(val) = std::env::var("SWARMLLM_INFERENCE_GPU_LAYERS") {
            if let Ok(n) = val.parse() {
                config.inference.gpu_layers = n;
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

        // Validate
        config.validate()?;

        Ok(config)
    }

    fn validate(&self) -> Result<(), SwarmError> {
        // Warn on privileged ports (not fatal — user may have permissions)
        if self.node.listen_port > 0 && self.node.listen_port < 1024 {
            tracing::warn!(
                port = self.node.listen_port,
                "Using privileged port — may require elevated permissions"
            );
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
        let model_config = ModelConfig { shard_size_mb: 4096 };
        assert!(model_config.validate().is_err());
    }

    #[test]
    fn shard_size_validation_accepts_valid() {
        let model_config = ModelConfig { shard_size_mb: 512 };
        assert!(model_config.validate().is_ok());
        let model_config = ModelConfig { shard_size_mb: 64 };
        assert!(model_config.validate().is_ok());
        let model_config = ModelConfig { shard_size_mb: 2048 };
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
}
