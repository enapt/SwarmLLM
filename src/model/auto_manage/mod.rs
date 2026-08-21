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
#[cfg(test)]
mod test_support;
pub mod vram;
pub mod wishlist;

pub mod manager;
pub mod scan;

pub use manager::AutoShardManager;
#[cfg(test)]
pub(crate) use prune::pressure_adjusted_target;
pub use scan::{check_and_load_model, rescan_local_shards, spawn_check_and_load, RescanOutcome};
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
/// Fraction of genuinely-free disk the shard budget may claim.
///
/// The rest is left for the OS, logs, the database and an in-flight download —
/// filling a disk to the last byte breaks far more than shard acquisition.
const FREE_DISK_HEADROOM_PCT: u64 = 80;

pub fn compute_budget_max_bytes(
    auto_max_storage_mb: u64,
    resources_max_disk_mb: u64,
    contribution: &swarmllm_types::ContributionMode,
    free_disk_bytes: Option<u64>,
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
    let by_config = match contribution {
        swarmllm_types::ContributionMode::Minimal => raw_max_bytes / 4,
        swarmllm_types::ContributionMode::Moderate => raw_max_bytes,
        swarmllm_types::ContributionMode::Maximum => raw_max_bytes.saturating_mul(3) / 2,
    };

    // Clamp to disk that actually exists.
    //
    // `max_disk_mb` was taken at face value: 50000 (50 GB) was accepted on a
    // 20 GB filesystem with ~15 GB free, and with the shard caps at their
    // unlimited defaults the node would keep accepting shards until `ENOSPC`
    // rather than pruning. Reported 2026-07-30. A configured ceiling is a
    // ceiling, not a promise that the space exists.
    //
    // `None` means the free space could not be read; do not invent a limit from
    // a failed syscall.
    let Some(free) = free_disk_bytes else {
        return by_config;
    };
    let by_disk = free / 100 * FREE_DISK_HEADROOM_PCT;
    if by_disk < by_config {
        tracing::warn!(
            configured_mb = by_config / (1024 * 1024),
            free_mb = free / (1024 * 1024),
            allowed_mb = by_disk / (1024 * 1024),
            "Storage budget is larger than the free space on disk — limiting it to \
             {}% of what is actually free so the disk cannot be filled",
            FREE_DISK_HEADROOM_PCT
        );
    }
    by_config.min(by_disk)
}

/// Free bytes on the filesystem holding `path`, or `None` if it cannot be read.
pub fn free_disk_bytes_for(path: &std::path::Path) -> Option<u64> {
    let mut disks = sysinfo::Disks::new_with_refreshed_list();
    disks.refresh(true);
    // Longest matching mount point wins, so a nested mount is preferred over `/`.
    disks
        .list()
        .iter()
        .filter(|d| path.starts_with(d.mount_point()))
        .max_by_key(|d| d.mount_point().as_os_str().len())
        .map(|d| d.available_space())
}

#[cfg(test)]
mod tests {
    use super::*;
    use swarmllm_types::ContributionMode;

    fn mib(n: u64) -> u64 {
        n.saturating_mul(1024).saturating_mul(1024)
    }

    /// The reported case: a 50 GB ceiling configured on a filesystem with ~15 GB
    /// free. A ceiling is not a promise the space exists, and with the shard
    /// caps at their unlimited defaults the node filled the disk instead of
    /// pruning.
    #[test]
    fn budget_is_clamped_to_free_disk() {
        let free = mib(15_000);
        let got = compute_budget_max_bytes(50_000, 0, &ContributionMode::Moderate, Some(free));
        assert!(
            got < mib(50_000),
            "must not exceed the configured ceiling's intent"
        );
        assert_eq!(got, free / 100 * FREE_DISK_HEADROOM_PCT);
        assert!(got < free, "must leave headroom, not fill the disk");
    }

    /// Plenty of free space must leave the configured budget untouched — this
    /// clamp is a safety net, not a second policy.
    #[test]
    fn ample_free_disk_does_not_reduce_the_budget() {
        let got =
            compute_budget_max_bytes(1024, 0, &ContributionMode::Moderate, Some(mib(500_000)));
        assert_eq!(got, mib(1024));
    }

    /// An unreadable free-space figure must not invent a limit.
    #[test]
    fn unknown_free_disk_leaves_the_budget_alone() {
        let with = compute_budget_max_bytes(1024, 0, &ContributionMode::Moderate, None);
        assert_eq!(with, mib(1024));
    }

    #[test]
    fn budget_uses_explicit_auto_storage_when_positive() {
        // auto_max_storage_mb > 0 takes precedence over resources_max_disk_mb.
        let got = compute_budget_max_bytes(1024, 999_999, &ContributionMode::Moderate, None);
        assert_eq!(got, mib(1024));
    }

    #[test]
    fn budget_falls_back_to_half_of_resources_when_auto_unset() {
        // auto_max_storage_mb == 0 falls back to resources/2.
        let got = compute_budget_max_bytes(0, 2048, &ContributionMode::Moderate, None);
        assert_eq!(got, mib(2048) / 2);
    }

    #[test]
    fn budget_minimal_scales_to_quarter() {
        let got = compute_budget_max_bytes(1024, 0, &ContributionMode::Minimal, None);
        assert_eq!(got, mib(1024) / 4);
    }

    #[test]
    fn budget_maximum_scales_to_3_over_2() {
        let got = compute_budget_max_bytes(1024, 0, &ContributionMode::Maximum, None);
        assert_eq!(got, mib(1024).saturating_mul(3) / 2);
    }

    #[test]
    fn budget_handles_zero_inputs() {
        // Neither configured → zero budget.
        assert_eq!(
            compute_budget_max_bytes(0, 0, &ContributionMode::Moderate, None),
            0
        );
        assert_eq!(
            compute_budget_max_bytes(0, 0, &ContributionMode::Maximum, None),
            0
        );
    }
}
