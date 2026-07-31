//! Node identity / resource / scheduling config.
//!
//! Hosts the node-side config types — `NodeConfig` (data_dir, listen_port,
//! contribution mode), `ResourceConfig` (vram/ram/disk/bandwidth caps +
//! `ResourceSchedule`), `ResourceSchedule` (off-hours throttling),
//! and `IdentityConfig` (voluntary self-reported region).
//!
//! Also exposes:
//! - `resolve_data_dir`: lightweight data-dir resolver used by CLI
//!   subcommands that don't load the full Config.
//! - `inference_vram_budget_mb` (on `ResourceConfig`): computes the
//!   effective VRAM budget honoring `max_gpu_vram_mb` cap or 80% of total.

use super::ContributionMode;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NodeConfig {
    #[serde(default = "default_data_dir")]
    pub data_dir: PathBuf,
    #[serde(default = "default_port")]
    pub listen_port: u16,
    #[serde(default)]
    pub contribution: ContributionMode,
    /// When true, auto-manage scales contribution up AND down within the
    /// caps in `[resources]`. When false, `contribution` is pinned at the
    /// user-set level (today's behaviour). Default: true — opt-in users
    /// who want a fixed level must explicitly set this to false. The auto
    /// path is the recommended one because at swarm scale a node's
    /// shards are over-replicated globally and holding them wastes VRAM.
    #[serde(default = "default_contribution_auto")]
    pub contribution_auto: bool,
    /// Anchor mode: run as a pure bootstrap / relay / AutoNAT node. Skips all
    /// inference — no models load, no HuggingFace polling, no shard
    /// acquisition, no auto-manage — and binds the dashboard/API to loopback
    /// only. The node still participates fully in the P2P network (relay
    /// server, AutoNAT prober, DCUtR, DHT, gossip), so it helps the swarm
    /// bootstrap without exposing any inference surface to the internet. Set
    /// via `--anchor`, `[node] anchor_mode = true`, or
    /// `SWARMLLM_NODE_ANCHOR_MODE=true`. Default: false.
    #[serde(default)]
    pub anchor_mode: bool,
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

/// Identity configuration (voluntary self-reported metadata).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct IdentityConfig {
    /// Optional ISO 3166-1 alpha-2 country code (e.g. "US", "DE", "JP").
    /// Voluntarily self-reported; used for the network map visualization.
    #[serde(default)]
    pub region: Option<String>,
}

/// Resolve the effective data dir using the same precedence as the full config
/// loader: CLI override > `SWARMLLM_NODE_DATA_DIR` env var > [`default_data_dir`].
/// Used by lightweight subcommands (Status, Chat, etc.) that don't load the
/// full config but still need a data-dir path.
pub fn resolve_data_dir(cli_data_dir: Option<&std::path::Path>) -> PathBuf {
    if let Some(dir) = cli_data_dir {
        return dir.to_path_buf();
    }
    if let Ok(val) = std::env::var("SWARMLLM_NODE_DATA_DIR") {
        return PathBuf::from(val);
    }
    default_data_dir()
}

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

fn default_contribution_auto() -> bool {
    true
}

impl Default for NodeConfig {
    fn default() -> Self {
        Self {
            data_dir: default_data_dir(),
            listen_port: default_port(),
            contribution: ContributionMode::default(),
            contribution_auto: default_contribution_auto(),
            anchor_mode: false,
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

    /// Compute the effective system-RAM budget for inference model loading.
    ///
    /// - If `max_ram_mb > 0`: use it as a hard cap.
    /// - Else if the GPU will actually run the models: 50% of total RAM.
    /// - Else (**CPU-only**, by hardware *or* by `gpu_layers = 0`): 80%.
    /// - Else: `None` (could not read the machine; do not invent a limit).
    ///
    /// `has_gpu` must mean "the GPU will run the models", not "a GPU exists" —
    /// a node with `inference.gpu_layers = 0` runs everything on the CPU
    /// whatever its hardware, and needs the CPU-only fraction. The caller
    /// derives it through `daemon::shard_loader::force_cpu_for`.
    ///
    /// **Why the split.** The documented default was a flat 50%, which is right
    /// where a GPU does the inference — system RAM is then support work, and
    /// leaving half the machine to the OS and everything else is generous. On a
    /// CPU-only node it is the wrong shape entirely: serving models *is* what
    /// the machine is for, and half of it is a hard capability cut. An 8 GB
    /// CPU-only node would get 4096 MB and start refusing
    /// `llama-3.2-3b-instruct-q4-k-m`, which estimates ~4575 MB and which such
    /// nodes serve today. Small CPU-only containers are a primary deployment
    /// target here, so a default that silently drops a common model from them
    /// is worse than the swapping it was written to prevent.
    ///
    /// The 50% figure itself is not new policy — `config/default.toml` has
    /// shipped `max_ram_mb = 0  # 0 = auto (50% of system RAM)` for as long as
    /// the field existed, while nothing in the codebase read it. Implementing an
    /// inert default is exactly when its value first has to be justified rather
    /// than inherited.
    ///
    /// Sibling of [`Self::inference_vram_budget_mb`].
    pub fn inference_ram_budget_mb(&self, system_ram_total_mb: u64, has_gpu: bool) -> Option<u64> {
        if self.max_ram_mb > 0 {
            return Some(self.max_ram_mb);
        }
        if system_ram_total_mb == 0 {
            return None;
        }
        if has_gpu {
            Some(system_ram_total_mb / 2)
        } else {
            Some(system_ram_total_mb / 5 * 4)
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
