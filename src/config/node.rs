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
    /// Cap on CPU threads used for inference. `0` (default) resolves from
    /// `node.contribution` — see [`ResourceConfig::inference_cpu_threads`].
    #[serde(default)]
    pub max_cpu_threads: u32,
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

/// Fraction of total VRAM inference may claim, by contribution setting.
///
/// This used to be a flat 0.8 whatever the user had chosen — so a node set to
/// contribute **minimally** still had 80% of its graphics card claimed. That is
/// the wrong default for who actually runs this: home machines, gaming PCs,
/// people who want to help without handing over the box. `Minimal` is also the
/// DEFAULT mode, so out of the box the software said "contribute minimally" and
/// then took 6.5 GB of an 8 GB card.
///
/// Observed 2026-08-04 on a development machine set to `minimal`: 7990 of
/// 8192 MiB in use.
///
/// The numbers are deliberately not aggressive. Inference still needs room to
/// be useful, and this is a ceiling on what may be RESIDENT, not a promise to
/// use it — the idle-unload path returns memory once the machine is quiet, so
/// the ceiling only binds while work is actually being done. Anyone who wants a
/// different figure sets `max_gpu_vram_mb` and that wins outright.
fn vram_fraction_for(contribution: ContributionMode) -> f64 {
    match contribution {
        // Default. Leave the majority of the card to whatever else the person
        // is doing with their computer.
        ContributionMode::Minimal => 0.5,
        ContributionMode::Moderate => 0.65,
        // An explicit offer of the machine.
        ContributionMode::Maximum => 0.8,
    }
}

impl ResourceConfig {
    /// Compute the effective VRAM budget for inference model loading.
    ///
    /// - If `max_gpu_vram_mb > 0`: use it as a hard cap.
    /// - Else if GPU detected (`gpu_vram_total_mb > 0`): use 80% of total.
    /// - Else: `None` (CPU-only node, no budget = unlimited).
    pub fn inference_vram_budget_mb(
        &self,
        gpu_vram_total_mb: u64,
        contribution: ContributionMode,
    ) -> Option<u64> {
        if self.max_gpu_vram_mb > 0 {
            // An explicit ceiling is the user's own decision and always wins.
            Some(self.max_gpu_vram_mb)
        } else if gpu_vram_total_mb > 0 {
            Some((gpu_vram_total_mb as f64 * vram_fraction_for(contribution)) as u64)
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
    /// Upload cap for serving shards to peers, in Mbps. `0` means unlimited.
    ///
    /// Uploading model files to other peers is **pure contribution** — none of
    /// it is the owner's own work — so an unset cap must not mean "use the whole
    /// connection". It did: the default is `0`, and `0` was taken as unlimited
    /// at every contribution level, so a stock install would saturate a home
    /// uplink seeding shards. Saturating someone's internet is the classic
    /// complaint against peer-to-peer software and it is exactly the kind of
    /// swamping a contribution setting exists to prevent.
    ///
    /// Unlike memory, there is no total to take a fraction of — the node cannot
    /// know the link speed — so an unset cap resolves to a conservative absolute
    /// figure instead, and only `Maximum` (an explicit offer of the machine)
    /// keeps unlimited. An explicit `max_bandwidth_mbps` always wins, in either
    /// direction.
    ///
    /// Note this throttles shard SERVING only. Tensor forwards during inference
    /// are latency-critical and bounded by the concurrency limits instead.
    pub fn shard_upload_mbps(&self, contribution: ContributionMode) -> u64 {
        if self.max_bandwidth_mbps > 0 {
            return self.max_bandwidth_mbps;
        }
        match contribution {
            // ~1.25 MB/s. Slow for a 500 MB shard, but this is background
            // seeding on a connection someone else is trying to use.
            ContributionMode::Minimal => 10,
            ContributionMode::Moderate => 50,
            // Offered the machine: take them at their word.
            ContributionMode::Maximum => 0,
        }
    }

    /// CPU threads the inference worker may use.
    ///
    /// **`total_cores` must be PHYSICAL cores, not logical.** Both halves of
    /// this function's job depend on that distinction.
    ///
    /// CPU inference had no thread limit at all: candle parallelises through
    /// rayon, whose default pool is every *logical* core, and nothing narrowed
    /// it. Measured on a 6-core node set to `Minimal` (2026-08-04), a single
    /// request held **529-534% of 600%** with ~10% of the machine idle — the
    /// whole box, at the lowest contribution setting. Of every resource this
    /// software spends, CPU starvation is the one the person sitting in front
    /// of the machine feels first: it is what makes a desktop stutter.
    ///
    /// **More threads is not more throughput.** Swept on a Ryzen 7 5800H
    /// (8 physical / 16 logical), phi-3.5 Q4_K_M, 201 tokens, model warm:
    ///
    /// | threads | tok/s |
    /// |---------|-------|
    /// | 4       | 2.26  |
    /// | 6       | 2.36  |
    /// | 8       | 2.18  |
    /// | 12      | 1.75  |
    /// | 16      | 1.49  |
    ///
    /// Throughput is flat to about the physical core count and then falls off a
    /// cliff — 16 threads is **37% slower** than 6. Quantised inference is bound
    /// by memory bandwidth rather than arithmetic, so two threads on one
    /// physical core contend for the same cache and load ports instead of
    /// adding anything. llama.cpp defaults to physical cores for this reason.
    ///
    /// The first version of this scaled a fraction of *logical* cores, which
    /// made `Maximum` (16 threads here) the **slowest** setting available: a
    /// user generously offering their whole machine got 37% less throughput
    /// than the default. Taking the fraction of physical cores instead means
    /// every level sits on the plateau, and offering more never costs
    /// performance — it only shortens how long the machine is busy.
    ///
    /// Below the plateau the reduction is real but sub-proportional, and worth
    /// it regardless: a node whose owner uninstalls it because their machine
    /// stutters contributes nothing. GPU nodes are unaffected, since offloaded
    /// layers do not run on these threads.
    ///
    /// Never returns 0, which rayon would read as "pick the default" — i.e.
    /// every logical core, the exact behaviour being fixed.
    pub fn inference_cpu_threads(
        &self,
        physical_cores: usize,
        logical_cores: usize,
        contribution: ContributionMode,
    ) -> usize {
        let physical = physical_cores.max(1);
        let logical = logical_cores.max(physical);
        if self.max_cpu_threads > 0 {
            // An explicit setting wins outright, in either direction — the same
            // rule the memory and bandwidth ceilings follow.
            //
            // Clamped to LOGICAL cores, not physical: asking for more threads
            // than the OS has is meaningless, but asking for more than the
            // physical count is a legitimate (if usually slower here) choice
            // that belongs to whoever set it. Quietly overriding a deliberate
            // number is worse than honouring a suboptimal one.
            return (self.max_cpu_threads as usize).clamp(1, logical);
        }
        let fraction = match contribution {
            // Default. Leave half the machine to whatever else its owner is
            // doing — the same split `vram_fraction_for` uses.
            ContributionMode::Minimal => 0.5,
            ContributionMode::Moderate => 0.75,
            ContributionMode::Maximum => 1.0,
        };
        (((physical as f64) * fraction).round() as usize).clamp(1, physical)
    }

    /// Physical and logical core counts, in that order.
    ///
    /// The two differ on any SMT machine and the difference is load-bearing —
    /// see [`Self::inference_cpu_threads`]. `available_parallelism` reports
    /// LOGICAL cores, so it is the wrong input for a thread budget on its own.
    pub fn detect_cpu_topology() -> (usize, usize) {
        let logical = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        // Falls back to the logical count on platforms where it cannot tell,
        // which is the pre-existing behaviour rather than a new guess.
        let physical = num_cpus::get_physical().max(1).min(logical.max(1));
        (physical.max(1), logical.max(1))
    }

    pub fn inference_ram_budget_mb(
        &self,
        system_ram_total_mb: u64,
        has_gpu: bool,
        contribution: ContributionMode,
    ) -> Option<u64> {
        if self.max_ram_mb > 0 {
            return Some(self.max_ram_mb);
        }
        if system_ram_total_mb == 0 {
            return None;
        }
        // The situational base: system RAM is support work where a GPU runs the
        // models, and is the whole job where it does not.
        let base = if has_gpu {
            system_ram_total_mb as f64 * 0.5
        } else {
            system_ram_total_mb as f64 * 0.8
        };
        // Then scaled by what the owner agreed to give, exactly as VRAM is.
        // A contribution level that governs one kind of memory and not the other
        // is not a limit, it is a suggestion — and RAM exhaustion is worse than
        // VRAM exhaustion, because the failure mode is swapping, which degrades
        // the entire machine rather than just this daemon.
        //
        // Expressed relative to Maximum so the documented CPU-only 80% and
        // GPU-node 50% still hold for a node that has explicitly offered itself.
        let scale = vram_fraction_for(contribution) / vram_fraction_for(ContributionMode::Maximum);
        Some((base * scale) as u64)
    }
}

impl Default for ResourceConfig {
    fn default() -> Self {
        Self {
            max_gpu_vram_mb: 0,
            max_ram_mb: 0,
            max_disk_mb: default_max_disk(),
            max_bandwidth_mbps: 0,
            max_cpu_threads: 0,
            schedule: ResourceSchedule::default(),
        }
    }
}
