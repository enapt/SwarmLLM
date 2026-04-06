//! Auto-manages shard downloads and pruning to improve network health.
//!
//! Periodically evaluates shard rarity, model popularity, VRAM fitness,
//! and resource pressure to download under-replicated shards and prune
//! over-replicated ones.

mod download;
mod prune;
mod scoring;
pub mod vram;

pub mod manager;
pub mod scan;

pub use manager::AutoShardManager;
#[cfg(test)]
pub(crate) use prune::pressure_adjusted_target;
pub use scan::{check_and_load_model, rescan_local_shards};
pub use vram::{compute_vram_budget, estimate_model_vram_mb, global_pool_vram_mb, local_vram_mb};
