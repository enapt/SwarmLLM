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
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NetworkConfig {
    #[serde(default)]
    pub bootstrap_peers: Vec<String>,
    #[serde(default = "default_true")]
    pub enable_relay: bool,
    #[serde(default = "default_true")]
    pub enable_relay_client: bool,
    #[serde(default = "default_max_peers")]
    pub max_peers: u32,
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
        }
    }
}

impl Default for NetworkConfig {
    fn default() -> Self {
        Self {
            bootstrap_peers: vec![],
            enable_relay: true,
            enable_relay_client: true,
            max_peers: default_max_peers(),
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

        Ok(config)
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
        )
        .unwrap();
        assert_eq!(config.node.listen_port, 9999);
    }
}
