use serde::{Deserialize, Serialize};
use std::collections::HashMap;
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

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NodeConfig {
    #[serde(default = "default_data_dir")]
    pub data_dir: PathBuf,
    #[serde(default = "default_port")]
    pub listen_port: u16,
    #[serde(default)]
    pub contribution: ContributionMode,
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
    /// Pruning aggressiveness during reduced hours: "normal", "aggressive", "conservative".
    #[serde(default = "default_prune_aggressiveness")]
    pub prune_aggressiveness: String,
}

impl Default for ResourceSchedule {
    fn default() -> Self {
        Self {
            enabled: false,
            reduced_hours_start: default_reduced_hours_start(),
            reduced_hours_end: default_reduced_hours_end(),
            reduced_contribution: default_reduced_contribution(),
            prune_aggressiveness: default_prune_aggressiveness(),
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
    /// Automatically activate relay listener when NAT is detected as Private.
    #[serde(default = "default_true")]
    pub auto_relay: bool,
    /// Enable mDNS for automatic LAN peer discovery (default: true).
    #[serde(default = "default_true")]
    pub enable_mdns: bool,
    /// Gossip network ID for grouping nodes. All nodes sharing the same ID
    /// can decode each other's sealed gossip. Defaults to "swarmllm-mainnet-v1".
    /// Set to a custom value (e.g. "my-private-net") for private networks.
    #[serde(default)]
    pub gossip_network_id: Option<String>,
    /// Enable AutoNAT for NAT detection (default: true).
    /// Disable on WSL2 to prevent protocol negotiation noise that starves outbound substreams.
    #[serde(default = "default_true")]
    pub enable_autonat: bool,
    /// Enable DCUtR for hole punching (default: true).
    /// Disable on WSL2 to prevent protocol negotiation noise that starves outbound substreams.
    #[serde(default = "default_true")]
    pub enable_dcutr: bool,
    /// Enable E2E encryption for tensor forwards and control messages (default: true).
    #[serde(default = "default_true")]
    pub enable_encryption: bool,
    /// Enable zstd compression for tensor payloads sent over the network.
    /// Only payloads larger than `tensor_compress_threshold` bytes are compressed.
    #[serde(default = "default_true")]
    pub tensor_compression: bool,
    /// Zstd compression level (1-22, default 1 for speed).
    #[serde(default = "default_tensor_compress_level")]
    pub tensor_compress_level: i32,
    /// Minimum payload size in bytes before compression is applied (default 1024).
    #[serde(default = "default_tensor_compress_threshold")]
    pub tensor_compress_threshold: usize,
    /// IP address to bind P2P listeners on (default: "0.0.0.0" = all interfaces).
    /// Set to "127.0.0.1" on WSL2 to prevent connections via unreliable NAT adapters.
    #[serde(default = "default_listen_address")]
    pub listen_address: String,
    /// Enable QUIC transport (default: true).
    /// Disable on WSL2 to prevent QUIC connection races with TCP (QUIC handshake is faster
    /// than TCP+Noise+Yamux, causing max_established_per_peer=1 to kill the TCP connection).
    #[serde(default = "default_true")]
    pub enable_quic: bool,
}

fn default_listen_address() -> String {
    "0.0.0.0".to_string()
}

/// Detect WSL2 by checking /proc/version for "microsoft" or "WSL".
fn is_wsl2() -> bool {
    std::fs::read_to_string("/proc/version")
        .map(|v| {
            let lower = v.to_lowercase();
            lower.contains("microsoft") || lower.contains("wsl")
        })
        .unwrap_or(false)
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
    /// Path to a smaller draft model for speculative decoding.
    /// Must be a GGUF file. The draft model should be much smaller than the
    /// main model (ideally <1/10th parameters) and share the same vocabulary.
    #[serde(default)]
    pub draft_model_path: Option<PathBuf>,
    /// GPU layers to offload for the draft model (default: same as main model).
    #[serde(default)]
    pub draft_gpu_layers: Option<u32>,
    /// Optional shard range for split inference (e.g. "0-4").
    /// When set, the node only claims these shard indices instead of all shards.
    #[serde(default)]
    pub shard_range: Option<(u32, u32)>,
    /// Maximum number of requests to batch together for inference.
    /// Default 1 means no batching (sequential, backward-compatible).
    #[serde(default = "default_max_batch_size")]
    pub max_batch_size: u32,
    /// How long (ms) to wait for additional requests before dispatching a partial batch.
    #[serde(default = "default_batch_timeout_ms")]
    pub batch_timeout_ms: u64,
    /// Maximum GPU memory (MB) for cached split models. When exceeded, the
    /// least-recently-used models are evicted. Default: None (unlimited).
    #[serde(default)]
    pub max_split_model_memory_mb: Option<u64>,
    /// Maximum number of prefix cache entries for cross-request KV state sharing.
    /// When true, KV-cache multi-turn sessions do NOT persist the `cached_prompt`
    /// field to the database — prompts stay in-memory only and are lost on restart.
    /// This prevents user prompts from being written to disk. Default: false.
    #[serde(default)]
    pub privacy_mode: bool,
    /// When true, the requesting node performs token→embedding locally before sending
    /// activations to the first pipeline segment. Remote nodes never see raw token IDs,
    /// only hidden-state activation tensors (which are harder to invert).
    /// Requires the embedding table to be available locally (auto-extracted from shard_000).
    /// Default: false.
    #[serde(default)]
    pub local_embedding_privacy: bool,
    /// When true, forces the requesting node to hold the final shard and perform
    /// token sampling locally. Combined with local_embedding_privacy (auto-enabled),
    /// this ensures no remote node ever sees plaintext — only intermediate activations.
    /// The pipeline "boomerangs" through remote nodes and returns to the requester.
    /// Requires the requester to hold both shard 0 (embedding) and the final shard (output head).
    /// Only useful for models with 3+ shards (2-shard = fully local, no distribution).
    /// Default: false.
    #[serde(default)]
    pub encrypted_pipeline: bool,
    /// Maximum peer RTT (ms) to consider for tensor parallelism AllReduce.
    /// Peers with measured latency above this threshold are excluded from TP groups.
    /// Default: 10ms (LAN-only).
    #[serde(default = "default_tp_max_latency_ms")]
    pub tp_max_latency_ms: u32,
}

fn default_tp_max_latency_ms() -> u32 {
    10
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
        }
    }
}

fn default_auto_update() -> AutoUpdateMode {
    AutoUpdateMode::Stable
}

fn default_check_interval_hours() -> u32 {
    6
}

/// Configurable credit earn/spend rates per pool or globally.
/// All values are in credits per unit of work.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CreditRateConfig {
    /// Credits earned per layer per token for serving inference.
    #[serde(default = "default_rate_inference_serve")]
    pub inference_serve: i64,
    /// Credits spent per layer per token for consuming inference.
    #[serde(default = "default_rate_inference_consume")]
    pub inference_consume: i64,
    /// Credits earned per GB per hour for hosting shards.
    #[serde(default = "default_rate_shard_hosting")]
    pub shard_hosting: i64,
    /// Credits earned per GB transferred for seeding shards.
    #[serde(default = "default_rate_shard_seeding")]
    pub shard_seeding: i64,
    /// Credits earned per connection hour for relay service.
    #[serde(default = "default_rate_relay_service")]
    pub relay_service: i64,
    /// Credits deducted as penalty for serve failures.
    #[serde(default = "default_rate_penalty")]
    pub penalty_serve_failure: i64,
}

impl Default for CreditRateConfig {
    fn default() -> Self {
        Self {
            inference_serve: default_rate_inference_serve(),
            inference_consume: default_rate_inference_consume(),
            shard_hosting: default_rate_shard_hosting(),
            shard_seeding: default_rate_shard_seeding(),
            relay_service: default_rate_relay_service(),
            penalty_serve_failure: default_rate_penalty(),
        }
    }
}

fn default_rate_inference_serve() -> i64 {
    10
}
fn default_rate_inference_consume() -> i64 {
    10
}
fn default_rate_shard_hosting() -> i64 {
    1
}
fn default_rate_shard_seeding() -> i64 {
    5
}
fn default_rate_relay_service() -> i64 {
    2
}
fn default_rate_penalty() -> i64 {
    50
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
    /// Global credit rate overrides. Pools can further override these per-pool.
    #[serde(default)]
    pub credit_rates: CreditRateConfig,
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            max_pool_size: default_max_pool_size(),
            invitation_ttl_hours: default_invitation_ttl_hours(),
            rate_limit_per_hour: default_pool_rate_limit(),
            gossip_interval_secs: default_pool_gossip_interval(),
            credit_rates: CreditRateConfig::default(),
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
    10
}

fn default_pool_gossip_interval() -> u64 {
    600
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ApiConfig {
    /// Bearer token for API authentication. If empty, one is auto-generated on first run.
    #[serde(default)]
    pub api_key: Option<String>,
    /// Expose hidden state tensors at `/v1/internal/hidden-states` for research.
    /// Disabled by default — enable only for trusted research environments.
    #[serde(default)]
    pub expose_hidden_states: bool,
    /// Rate limit (requests per minute) for `/v1/` and `/api/chat` endpoints.
    /// Default: 60.
    #[serde(default)]
    pub rate_limit_rpm: Option<u64>,
    /// Rate limit (requests per minute) for `/api/admin/` endpoints.
    /// Default: 200.
    #[serde(default)]
    pub rate_limit_admin_rpm: Option<u64>,
}

/// Configuration for automatic shard management.
///
/// When enabled, the node periodically evaluates network shard coverage
/// and downloads rarest shards for popular models — filling gaps to
/// improve overall network availability.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AutoManageConfig {
    /// Master toggle for auto shard management.
    #[serde(default = "default_true")]
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
    /// Override interval in seconds (for testing). Takes precedence over `interval_minutes`.
    #[serde(default)]
    pub interval_seconds: Option<u64>,
    /// Maximum number of concurrent shard downloads (default 3).
    #[serde(default = "default_max_concurrent_downloads")]
    pub max_concurrent_downloads: usize,
    /// Default cap on auto-managed shards per model (0 = unlimited).
    /// Prevents auto-manage from downloading ALL shards of a single model.
    #[serde(default)]
    pub default_model_shard_cap: u32,
    /// Per-model auto-manage overrides keyed by model ID.
    #[serde(default)]
    pub model_policies: HashMap<String, ModelAutoManagePolicy>,
    /// Enable automatic shard pruning (removal of over-replicated shards).
    #[serde(default = "default_true")]
    pub prune_enabled: bool,
    /// Minimum number of replicas to maintain per shard across the network.
    #[serde(default = "default_min_replicas")]
    pub min_replicas: u32,
    /// Cooldown in seconds between prune actions on the same model.
    #[serde(default = "default_prune_cooldown_secs")]
    pub prune_cooldown_secs: u64,
    /// Block pruning if remaining holders have avg load above this threshold.
    #[serde(default = "default_max_holder_load_for_prune")]
    pub max_holder_load_for_prune: u32,
}

/// Per-model auto-manage policy controlling whether a model participates
/// in automatic shard downloads and how many shards to acquire.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelAutoManagePolicy {
    /// Whether auto-manage may download shards for this model.
    #[serde(default = "default_true")]
    pub enabled: bool,
    /// Maximum shards auto-manage will acquire for this model (0 = unlimited / use global default).
    #[serde(default)]
    pub max_shards: u32,
    /// Whether auto-manage may prune (delete) over-replicated shards for this model.
    #[serde(default = "default_true")]
    pub prune_enabled: bool,
}

impl Default for AutoManageConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_storage_mb: 0,
            interval_minutes: default_auto_manage_interval(),
            max_shards: 0,
            interval_seconds: None,
            max_concurrent_downloads: default_max_concurrent_downloads(),
            default_model_shard_cap: 0,
            model_policies: HashMap::new(),
            prune_enabled: true,
            min_replicas: default_min_replicas(),
            prune_cooldown_secs: default_prune_cooldown_secs(),
            max_holder_load_for_prune: default_max_holder_load_for_prune(),
        }
    }
}

fn default_max_concurrent_downloads() -> usize {
    3
}

fn default_min_replicas() -> u32 {
    2
}

fn default_prune_cooldown_secs() -> u64 {
    300
}

fn default_max_holder_load_for_prune() -> u32 {
    3
}

fn default_auto_manage_interval() -> u32 {
    5
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
        .unwrap_or_else(|| {
            // Fallback to a well-known path instead of "." (current directory)
            #[cfg(unix)]
            {
                PathBuf::from("/var/lib/swarmllm")
            }
            #[cfg(not(unix))]
            {
                PathBuf::from(".")
            }
        })
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

fn default_tensor_compress_level() -> i32 {
    1
}

fn default_tensor_compress_threshold() -> usize {
    1024
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

fn default_prune_aggressiveness() -> String {
    "normal".into()
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

impl ResourceConfig {
    /// Compute the effective VRAM budget for inference model loading.
    ///
    /// - If `max_gpu_vram_mb > 0`: use it as a hard cap.
    /// - Else if GPU detected (`gpu_vram_total_mb > 0`): use 80% of total.
    /// - Else: `None` (CPU-only node, no budget = unlimited).
    pub fn inference_vram_budget_mb(&self, gpu_vram_total_mb: u64) -> Option<u64> {
        if self.max_gpu_vram_mb > 0 {
            Some(self.max_gpu_vram_mb)
        } else if gpu_vram_total_mb > 0 {
            Some((gpu_vram_total_mb as f64 * 0.8) as u64)
        } else {
            None
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
            auto_relay: true,
            enable_mdns: true,
            enable_autonat: true,
            enable_dcutr: true,
            enable_encryption: true,
            gossip_network_id: None,
            tensor_compression: true,
            tensor_compress_level: default_tensor_compress_level(),
            tensor_compress_threshold: default_tensor_compress_threshold(),
            listen_address: default_listen_address(),
            enable_quic: true,
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
            draft_model_path: None,
            draft_gpu_layers: None,
            shard_range: None,
            max_batch_size: default_max_batch_size(),
            batch_timeout_ms: default_batch_timeout_ms(),
            max_split_model_memory_mb: None,
            privacy_mode: false,
            local_embedding_privacy: false,
            encrypted_pipeline: false,
            tp_max_latency_ms: default_tp_max_latency_ms(),
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
        if is_wsl2() {
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
