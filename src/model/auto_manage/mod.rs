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

/// Fraction of genuinely-free disk the shard budget may claim.
///
/// The rest is left for the OS, logs, the database and an in-flight download —
/// filling a disk to the last byte breaks far more than shard acquisition.
const FREE_DISK_HEADROOM_PCT: u64 = 80;

/// The share of `max_disk_mb` auto-manage may fill when `max_storage_mb` is
/// left at 0, by contribution level.
///
/// **These are the numbers the setup wizard shows** — "≤ 25%", "≤ 50%",
/// "≤ 75%+" — and its disk preview multiplies the disk by exactly these. The
/// daemon used to take HALF of `max_disk_mb` and then quarter it again for
/// Minimal, so the product promised 25% and delivered 12.5%, and a node at
/// the DEFAULT level (Minimal) with a default 50 GB limit had a 6.25 GB
/// budget: one 7B model and it was full for good (gotcha #448).
pub fn contribution_disk_share_pct(contribution: &swarmllm_types::ContributionMode) -> u64 {
    match contribution {
        swarmllm_types::ContributionMode::Minimal => 25,
        swarmllm_types::ContributionMode::Moderate => 50,
        swarmllm_types::ContributionMode::Maximum => 75,
    }
}

/// Which rule produced a [`StorageBudget`], so a refusal can say WHY the
/// figure is what it is. A node that says "no remaining storage budget" while
/// the config reads 50 GB and the disk holds 18 GB sends a careful reader
/// looking for a phantom reservation; naming the limit is what stops that.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StorageLimit {
    /// `[auto_manage] max_storage_mb` was set and is honoured as written.
    Explicit { max_storage_mb: u64 },
    /// `max_storage_mb` is 0: a share of `max_disk_mb` by contribution level.
    ContributionShare {
        contribution: swarmllm_types::ContributionMode,
        pct: u64,
        max_disk_mb: u64,
    },
    /// The configured figure exceeded `max_disk_mb`, the ceiling on everything.
    MaxDisk { max_disk_mb: u64 },
    /// The filesystem has less room than the configuration asks for.
    FreeDisk { free_mb: u64 },
    /// Neither `max_storage_mb` nor `max_disk_mb` is set — nothing may be held.
    NothingConfigured,
}

impl std::fmt::Display for StorageLimit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Explicit { max_storage_mb } => {
                write!(f, "auto_manage.max_storage_mb = {max_storage_mb}")
            }
            Self::ContributionShare {
                contribution,
                pct,
                max_disk_mb,
            } => write!(
                f,
                "{pct}% of max_disk_mb = {max_disk_mb} at {} contribution",
                contribution_name(contribution)
            ),
            Self::MaxDisk { max_disk_mb } => write!(f, "resources.max_disk_mb = {max_disk_mb}"),
            Self::FreeDisk { free_mb } => write!(
                f,
                "{FREE_DISK_HEADROOM_PCT}% of the {free_mb} MB free on disk"
            ),
            Self::NothingConfigured => write!(f, "max_disk_mb is 0 — no storage configured"),
        }
    }
}

fn contribution_name(contribution: &swarmllm_types::ContributionMode) -> &'static str {
    match contribution {
        swarmllm_types::ContributionMode::Minimal => "minimal",
        swarmllm_types::ContributionMode::Moderate => "moderate",
        swarmllm_types::ContributionMode::Maximum => "maximum",
    }
}

/// How many bytes of shards this node may hold in total, and which rule
/// decided it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StorageBudget {
    pub bytes: u64,
    pub limited_by: StorageLimit,
}

impl StorageBudget {
    /// Bytes still available for downloads given what is already held.
    pub fn remaining(&self, held_bytes: u64) -> u64 {
        self.bytes.saturating_sub(held_bytes)
    }
}

/// **The one answer to "how much shard storage may this node hold?"**
///
/// Consulted by the download scheduler (`scoring::remaining_budget`), by the
/// prune pass's disk pressure (`prune::compute_resource_pressure`), by the
/// settings storage bar (`api::admin::storage_breakdown`), by the pool page
/// (`api::pool`) and by the diagnostics report. Until 2026-09-03 the first
/// two disagreed: the download side quartered the figure for Minimal and the
/// prune side did not, so a node holding 18 GB against 50 GB configured was
/// OVER budget for downloading (12.5 GB) and at 36% for pruning — refused
/// every download, pruned nothing, for ever (gotcha #448). Two accountants
/// for one disk wedge exactly where they disagree.
///
/// The rule, in order:
/// 1. An explicit `max_storage_mb` is honoured as written. Scaling a number
///    the user typed by a level they may not connect to it is how the
///    tester above ended up with a quarter of what they set; the VRAM
///    budget already follows this precedent ("`max_gpu_vram_mb` wins
///    outright").
/// 2. Otherwise `max_disk_mb` × [`contribution_disk_share_pct`].
/// 3. Never more than `max_disk_mb`, the ceiling on everything (Maximum used
///    to grant 150% of an explicit figure, above the disk limit).
/// 4. Never more than what is HELD plus 80% of what is FREE. `max_disk_mb`
///    was taken at face value: 50 GB was accepted on a 20 GB filesystem
///    with ~15 GB free and the node kept accepting shards until `ENOSPC`
///    rather than pruning (reported 2026-07-30). The held term is what
///    makes the clamp invariant under our own holdings — free space
///    already excludes them, so clamping the TOTAL to a fraction of free
///    and then subtracting held again under-counted the room by exactly
///    what was held. `None` for free space means it could not be read;
///    do not invent a limit from a failed syscall.
pub fn storage_budget(
    auto_max_storage_mb: u64,
    max_disk_mb: u64,
    contribution: &swarmllm_types::ContributionMode,
    free_disk_bytes: Option<u64>,
    held_bytes: u64,
) -> StorageBudget {
    let mib = |mb: u64| mb.saturating_mul(1024).saturating_mul(1024);
    let (mut bytes, mut limited_by) = if auto_max_storage_mb > 0 {
        (
            mib(auto_max_storage_mb),
            StorageLimit::Explicit {
                max_storage_mb: auto_max_storage_mb,
            },
        )
    } else if max_disk_mb > 0 {
        let pct = contribution_disk_share_pct(contribution);
        (
            mib(max_disk_mb) / 100 * pct,
            StorageLimit::ContributionShare {
                contribution: contribution.clone(),
                pct,
                max_disk_mb,
            },
        )
    } else {
        (0, StorageLimit::NothingConfigured)
    };
    if max_disk_mb > 0 && bytes > mib(max_disk_mb) {
        bytes = mib(max_disk_mb);
        limited_by = StorageLimit::MaxDisk { max_disk_mb };
    }
    if let Some(free) = free_disk_bytes {
        let by_disk = held_bytes.saturating_add(free / 100 * FREE_DISK_HEADROOM_PCT);
        if by_disk < bytes {
            bytes = by_disk;
            limited_by = StorageLimit::FreeDisk {
                free_mb: free / (1024 * 1024),
            };
        }
    }
    StorageBudget { bytes, limited_by }
}

/// Bytes and count of the shards `node_id` holds, priced by the manifest.
///
/// Reads the registry's reverse index (`shards_for_node`), so it counts ONLY
/// shards the node has actually registered as held — a manifest this node
/// knows but holds no part of contributes nothing, and a quarantined file
/// (`.quarantine`, `.mismatched`) is not a held shard. The tester who
/// reported #448 hypothesised a "phantom reservation" for such manifests;
/// there is none, and this is where that can be checked.
///
/// **Every surface that reports "used" goes through here** — the download
/// scheduler, prune pressure, the settings bar, the pool page and the
/// diagnostics report — so they cannot disagree about what is held.
pub fn held_shard_bytes(
    state: &crate::daemon::SharedState,
    node_id: &crate::types::NodeId,
) -> (u64, u32) {
    let local_shards = state.model_registry.shards_for_node(node_id);
    let count = local_shards.len() as u32;
    let bytes = local_shards
        .iter()
        .filter_map(|sid| {
            let manifest = state.model_registry.get_manifest(&sid.model_id)?;
            manifest
                .shards
                .iter()
                .find(|s| s.index == sid.index)
                .map(|si| si.size_bytes)
        })
        .sum();
    (bytes, count)
}

/// The storage budget for THIS node, right now: live config, live
/// contribution level, the disk it is actually on, and what it holds.
pub fn storage_budget_now(state: &crate::daemon::SharedState) -> (StorageBudget, u64, u32) {
    let (held_bytes, held_shards) = held_shard_bytes(state, state.identity.node_id());
    let live = state.cfg();
    let budget = storage_budget(
        live.auto_manage.max_storage_mb,
        live.resources.max_disk_mb,
        &state.contribution(),
        free_disk_bytes_for(&state.config.node.data_dir),
        held_bytes,
    );
    (budget, held_bytes, held_shards)
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

    /// The reported case (gotcha #448): `max_storage_mb = 50000`, contribution
    /// Minimal, 18 GB held, 158 GB free. The old rule quartered the explicit
    /// figure to 12.5 GB and refused every download; a number the user typed
    /// is honoured as typed.
    #[test]
    fn an_explicit_limit_is_honoured_whatever_the_contribution_level() {
        for level in [
            ContributionMode::Minimal,
            ContributionMode::Moderate,
            ContributionMode::Maximum,
        ] {
            let b = storage_budget(50_000, 50_000, &level, Some(mib(158_000)), mib(18_000));
            assert_eq!(b.bytes, mib(50_000), "{level:?}");
            assert_eq!(
                b.limited_by,
                StorageLimit::Explicit {
                    max_storage_mb: 50_000
                }
            );
            assert_eq!(b.remaining(mib(18_000)), mib(32_000));
        }
    }

    /// With `max_storage_mb` unset the budget is the share of `max_disk_mb`
    /// the setup wizard promises: 25 / 50 / 75%. The old rule gave Minimal
    /// 12.5% (half, then a quarter of that).
    #[test]
    fn the_default_budget_is_the_share_the_wizard_promises() {
        let cases = [
            (ContributionMode::Minimal, 25),
            (ContributionMode::Moderate, 50),
            (ContributionMode::Maximum, 75),
        ];
        for (level, pct) in cases {
            let b = storage_budget(0, 50_000, &level, None, 0);
            assert_eq!(b.bytes, mib(50_000) / 100 * pct, "{level:?}");
            assert_eq!(
                b.limited_by,
                StorageLimit::ContributionShare {
                    contribution: level,
                    pct,
                    max_disk_mb: 50_000
                }
            );
        }
        // The control the fix is measured against: the default install
        // (Minimal, 50 GB) used to have 6.25 GB; it now has 12.5 GB.
        assert_eq!(
            storage_budget(0, 50_000, &ContributionMode::Minimal, None, 0).bytes,
            mib(12_500)
        );
    }

    /// `max_disk_mb` is the ceiling on everything. Maximum used to grant 150%
    /// of an explicit figure and could exceed the disk limit the same panel
    /// sets.
    #[test]
    fn the_budget_never_exceeds_the_disk_limit() {
        let b = storage_budget(80_000, 50_000, &ContributionMode::Maximum, None, 0);
        assert_eq!(b.bytes, mib(50_000));
        assert_eq!(
            b.limited_by,
            StorageLimit::MaxDisk {
                max_disk_mb: 50_000
            }
        );
    }

    /// A 50 GB ceiling configured on a filesystem with ~15 GB free. A ceiling
    /// is not a promise the space exists; with the shard caps unlimited the
    /// node filled the disk instead of pruning (reported 2026-07-30).
    #[test]
    fn budget_is_clamped_to_free_disk() {
        let free = mib(15_000);
        let b = storage_budget(50_000, 0, &ContributionMode::Moderate, Some(free), 0);
        assert!(b.bytes < mib(50_000));
        assert_eq!(b.bytes, free / 100 * FREE_DISK_HEADROOM_PCT);
        assert!(b.bytes < free, "must leave headroom, not fill the disk");
        assert_eq!(b.limited_by, StorageLimit::FreeDisk { free_mb: 15_000 });
    }

    /// Free space already excludes what this node holds, so the clamp is on
    /// held + 80% of free. Clamping the TOTAL to 80% of free and subtracting
    /// held again under-counted the room by exactly what was held: 18 GB
    /// held with 30 GB free used to leave 6 GB of room, not 24.
    #[test]
    fn the_free_disk_clamp_counts_what_is_already_held() {
        let held = mib(18_000);
        let free = mib(30_000);
        let b = storage_budget(200_000, 0, &ContributionMode::Moderate, Some(free), held);
        assert_eq!(b.bytes, held + free / 100 * FREE_DISK_HEADROOM_PCT);
        assert_eq!(b.remaining(held), free / 100 * FREE_DISK_HEADROOM_PCT);
    }

    /// Plenty of free space must leave the configured budget untouched — this
    /// clamp is a safety net, not a second policy.
    #[test]
    fn ample_free_disk_does_not_reduce_the_budget() {
        let b = storage_budget(1024, 0, &ContributionMode::Moderate, Some(mib(500_000)), 0);
        assert_eq!(b.bytes, mib(1024));
        assert_eq!(
            b.limited_by,
            StorageLimit::Explicit {
                max_storage_mb: 1024
            }
        );
    }

    /// An unreadable free-space figure must not invent a limit.
    #[test]
    fn unknown_free_disk_leaves_the_budget_alone() {
        let b = storage_budget(1024, 0, &ContributionMode::Moderate, None, 0);
        assert_eq!(b.bytes, mib(1024));
    }

    #[test]
    fn budget_handles_zero_inputs() {
        // Neither configured → zero budget, and the limit says so.
        let b = storage_budget(0, 0, &ContributionMode::Maximum, None, 0);
        assert_eq!(b.bytes, 0);
        assert_eq!(b.limited_by, StorageLimit::NothingConfigured);
    }

    /// The limit names itself in words a log reader can act on.
    #[test]
    fn the_limit_describes_itself() {
        let share = storage_budget(0, 50_000, &ContributionMode::Minimal, None, 0);
        assert_eq!(
            share.limited_by.to_string(),
            "25% of max_disk_mb = 50000 at minimal contribution"
        );
        let explicit = storage_budget(50_000, 50_000, &ContributionMode::Minimal, None, 0);
        assert_eq!(
            explicit.limited_by.to_string(),
            "auto_manage.max_storage_mb = 50000"
        );
    }
}
