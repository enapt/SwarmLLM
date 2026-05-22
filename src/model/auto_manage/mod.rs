//! Auto-manages shard downloads and pruning to improve network health.
//!
//! Periodically evaluates shard rarity, model popularity, VRAM fitness,
//! and resource pressure to download under-replicated shards and prune
//! over-replicated ones.

mod download;
mod parallax;
mod prune;
pub mod quant;
mod scoring;
pub mod vram;
pub mod wishlist;

pub mod manager;
pub mod scan;

pub use manager::AutoShardManager;
#[cfg(test)]
pub(crate) use prune::pressure_adjusted_target;
pub use scan::{check_and_load_model, rescan_local_shards, spawn_check_and_load};
pub use vram::{compute_vram_budget, estimate_model_vram_mb, global_pool_vram_mb, local_vram_mb};
pub use wishlist::{compute_wishlist, refresh_wishlist, Wishlist, WishlistEntry, WishlistStatus};

/// Returns true when a shard file on disk looks fully downloaded.
///
/// The check succeeds when `expected_size > 0` AND the file's metadata length
/// is within 10% of the expected size (tolerates small compression/tail
/// differences). A zero expected size means the manifest has no length info,
/// in which case we refuse to validate — otherwise an empty file would pass.
///
/// The 10% tolerance is acceptable when there's also a non-zero BLAKE3
/// hash to verify against. For zero-hash placeholder manifests, use
/// `shard_size_exact` instead — without a hash to corroborate, accepting
/// "close enough" lets a wrong-but-similar-size file register as a valid
/// holder.
pub(crate) fn shard_size_ok(path: &std::path::Path, expected_size: u64) -> bool {
    expected_size > 0
        && std::fs::metadata(path)
            .map(|m| {
                let actual = m.len();
                actual >= expected_size * 9 / 10 && actual <= expected_size * 11 / 10
            })
            .unwrap_or(false)
}

/// Stricter sibling of `shard_size_ok` for the zero-hash placeholder path.
/// Requires an exact byte-level size match. Used when the manifest has no
/// hash to verify against so size is the only signal we have.
pub(crate) fn shard_size_exact(path: &std::path::Path, expected_size: u64) -> bool {
    expected_size > 0
        && std::fs::metadata(path)
            .map(|m| m.len() == expected_size)
            .unwrap_or(false)
}

/// Compute the auto-manage byte budget after applying the
/// `ContributionMode` scaling. Source of truth for both the
/// auto-manage download scheduler (`scoring::remaining_budget_bytes`)
/// and the admin storage-breakdown API (`api::admin::storage_breakdown`)
/// — both used to mirror this body with a brittle comment pointing at
/// the other. If the scaling factors change here, both sides pick it up.
///
/// `auto_max_storage_mb=0` falls back to half of `resources.max_disk_mb`.
pub(crate) fn compute_budget_max_bytes(
    auto_max_storage_mb: u64,
    resources_max_disk_mb: u64,
    contribution: &swarmllm_types::ContributionMode,
) -> u64 {
    let raw_max_bytes = if auto_max_storage_mb > 0 {
        auto_max_storage_mb
            .saturating_mul(1024)
            .saturating_mul(1024)
    } else {
        resources_max_disk_mb
            .saturating_mul(1024)
            .saturating_mul(1024)
            / 2
    };
    match contribution {
        swarmllm_types::ContributionMode::Minimal => raw_max_bytes / 4,
        swarmllm_types::ContributionMode::Moderate => raw_max_bytes,
        swarmllm_types::ContributionMode::Maximum => raw_max_bytes.saturating_mul(3) / 2,
    }
}
