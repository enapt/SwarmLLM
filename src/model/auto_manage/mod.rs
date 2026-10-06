//! Auto-manages shard downloads and pruning to improve network health.
//!
//! Periodically evaluates shard rarity, model popularity, VRAM fitness,
//! and resource pressure to download under-replicated shards and prune
//! over-replicated ones.

pub mod canonical;
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

/// Share of the FILESYSTEM this node's files never take: it stays free for the
/// OS, logs, the database, an in-flight download and whatever else the owner
/// runs on that disk.
///
/// The figure is Kubernetes' — the kubelet treats a node filesystem with under
/// 10% available as a hard eviction threshold (`evictionHard:
/// nodefs.available<10%`, the KubeletConfiguration default): below it the
/// machine is in trouble whatever is using the space.
///
/// The rule this replaced, "held + 80% of what is free", had no floor. The
/// room it offered shrank with every download but stayed open while a part fit
/// in 80% of what was left, so a node whose limit was the disk itself filled it
/// to within 1.25 parts of full — 99.5% of a 30 GB container with 550 MB parts
/// — into whatever its owner had put there to stop exactly that, and its disk
/// pressure sat at
/// the top of the range, where prune sheds and the download pass refilled
/// (2026-10-06, gotcha #795).
const FREE_DISK_RESERVE_PCT: u64 = 10;

/// The filesystem holding the data directory, as it stands now.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DiskSpace {
    /// What may still be written (`available`, i.e. after any root reserve).
    pub free_bytes: u64,
    /// The filesystem's size.
    pub total_bytes: u64,
}

impl DiskSpace {
    /// What this node's files leave free on this filesystem, always.
    pub fn reserve_bytes(&self) -> u64 {
        self.total_bytes / 100 * FREE_DISK_RESERVE_PCT
    }
}

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
    FreeDisk { free_mb: u64, reserve_mb: u64 },
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
            Self::FreeDisk {
                free_mb,
                reserve_mb,
            } => write!(
                f,
                "the disk: {free_mb} MB free, and {reserve_mb} MB \
                 ({FREE_DISK_RESERVE_PCT}% of it) is always left free"
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
/// 4. Never more than what is HELD plus what is FREE beyond
///    [`FREE_DISK_RESERVE_PCT`] of the filesystem. `max_disk_mb` was taken at
///    face value: 50 GB was accepted on a 20 GB filesystem with ~15 GB free
///    and the node kept accepting shards until `ENOSPC` rather than pruning
///    (reported 2026-07-30). The held term is what makes the clamp invariant
///    under our own holdings — free space already excludes them, so a part
///    downloaded or deleted moves bytes between the two terms and leaves the
///    budget where it was. The reserve is a FLOOR on free space, not a share
///    of it: "80% of what is free" never closed while a part still fit, and
///    filled a container to 99.5% (gotcha #795). `None` means the disk could
///    not be read; do not invent a limit from a failed syscall.
pub fn storage_budget(
    auto_max_storage_mb: u64,
    max_disk_mb: u64,
    contribution: &swarmllm_types::ContributionMode,
    disk: Option<DiskSpace>,
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
    if let Some(disk) = disk {
        let by_disk =
            held_bytes.saturating_add(disk.free_bytes.saturating_sub(disk.reserve_bytes()));
        if by_disk < bytes {
            bytes = by_disk;
            limited_by = StorageLimit::FreeDisk {
                free_mb: disk.free_bytes / (1024 * 1024),
                reserve_mb: disk.reserve_bytes() / (1024 * 1024),
            };
        }
    }
    StorageBudget { bytes, limited_by }
}

/// Everything the storage budget is computed from, read ONCE.
///
/// Both auto-manage passes judge a part by the disk pressure on this node, and
/// the download pass has to ask what that pressure WOULD be once a part has
/// landed — the question that keeps it from fetching what prune sheds next
/// (`prune::AutoShardManager::would_shed_copy`, gotcha #795). The budget itself
/// cannot answer it: the free-disk rule makes it depend on what is held, so
/// "after this download" means re-deriving it from its inputs, which is what
/// [`StorageReading::with_added`] does.
#[derive(Clone, Debug)]
pub struct StorageReading {
    max_storage_mb: u64,
    max_disk_mb: u64,
    contribution: swarmllm_types::ContributionMode,
    disk: Option<DiskSpace>,
    /// Bytes under the models directory (`held_disk_bytes`).
    pub held_bytes: u64,
}

impl StorageReading {
    /// This node's storage as it stands now: live config, live contribution
    /// level, the disk it is on, and what it holds.
    pub fn now(state: &crate::daemon::SharedState) -> Self {
        let live = state.cfg();
        Self {
            max_storage_mb: live.auto_manage.max_storage_mb,
            max_disk_mb: live.resources.max_disk_mb,
            contribution: state.contribution(),
            disk: disk_space_for(&state.config.node.data_dir),
            held_bytes: held_disk_bytes(&state.config.node.data_dir),
        }
    }

    pub fn budget(&self) -> StorageBudget {
        storage_budget(
            self.max_storage_mb,
            self.max_disk_mb,
            &self.contribution,
            self.disk,
            self.held_bytes,
        )
    }

    /// Disk pressure, 0.0-1.0: what is held over the budget. 0 when there is
    /// no budget to measure against — nothing configured is not "full".
    pub fn pressure(&self) -> f64 {
        let budget = self.budget();
        if budget.bytes == 0 {
            return 0.0;
        }
        (self.held_bytes as f64 / budget.bytes as f64).min(1.0)
    }

    /// The same node with `bytes` more on its disk — held, and no longer free.
    pub fn with_added(&self, bytes: u64) -> Self {
        let mut next = self.clone();
        next.held_bytes = next.held_bytes.saturating_add(bytes);
        if let Some(disk) = next.disk.as_mut() {
            disk.free_bytes = disk.free_bytes.saturating_sub(bytes);
        }
        next
    }
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

/// Bytes this node actually occupies under its models directory.
///
/// `held_shard_bytes` prices the MANIFEST, so it can only see `shard_NNN.bin`.
/// A model directory holds more than that, and all of it is real disk:
/// `gguf_header.bin` (one per model), `tied_output_weight.bin` (**279 MB for a
/// 1.3 GB model**), `mmproj.gguf` for a vision model (595 MB for LLaVA-7B),
/// quarantined files, and `.tmp` left behind by a download that died.
///
/// Measured on the live node 2026-09-14: **33062 MB on disk against 31423 MB
/// counted** — a 1637 MB gap, ~5%, of which `tied_output_weight.bin` was 1566 MB.
/// So a user who set a disk limit got a node that quietly used 5% more than it
/// believed, and a cancelled download left ~287 MB attached to a model counting
/// as zero shards and therefore zero bytes, which nothing swept.
///
/// A walk of stat calls over a few dozen files, so it is not cached: the
/// consumers are a timer pass and user-triggered handlers, and a cached
/// storage figure that lags a prune is its own defect.
///
/// **This is only safe to charge because the bytes are RECLAIMABLE.** Prune
/// deletes `shard_NNN.bin` and nothing else, so charging for files it cannot
/// remove would leave a node near its limit shedding shards forever chasing a
/// floor it can never reach — the shape of the sole-replica prune report
/// (`docs/FUTURE_WORK.md` item 49). `shard::cleanup_orphaned_model_files`,
/// called from the prune cycle, is the other half and must stay.
pub fn held_disk_bytes(data_dir: &std::path::Path) -> u64 {
    let models_dir = data_dir.join("models");
    let Ok(entries) = std::fs::read_dir(&models_dir) else {
        return 0;
    };
    let mut total = 0u64;
    for model in entries.flatten() {
        let Ok(files) = std::fs::read_dir(model.path()) else {
            continue;
        };
        for file in files.flatten() {
            if let Ok(meta) = file.metadata() {
                if meta.is_file() {
                    total = total.saturating_add(meta.len());
                }
            }
        }
    }
    total
}

/// The storage budget for THIS node, right now: live config, live
/// contribution level, the disk it is actually on, and what it holds.
///
/// The byte figure is what is ON DISK (`held_disk_bytes`); the count is the
/// registry's, because a shard count answers a different question — how many
/// pieces prune may consider — and pricing it off the directory would count a
/// header as a shard.
pub fn storage_budget_now(state: &crate::daemon::SharedState) -> (StorageBudget, u64, u32) {
    let (_manifest_bytes, held_shards) = held_shard_bytes(state, state.identity.node_id());
    let reading = StorageReading::now(state);
    (reading.budget(), reading.held_bytes, held_shards)
}

/// The filesystem holding `path` — free and total bytes — or `None` if it
/// cannot be read.
pub fn disk_space_for(path: &std::path::Path) -> Option<DiskSpace> {
    let mut disks = sysinfo::Disks::new_with_refreshed_list();
    disks.refresh(true);
    // Longest matching mount point wins, so a nested mount is preferred over `/`.
    disks
        .list()
        .iter()
        .filter(|d| path.starts_with(d.mount_point()))
        .max_by_key(|d| d.mount_point().as_os_str().len())
        .map(|d| DiskSpace {
            free_bytes: d.available_space(),
            total_bytes: d.total_space(),
        })
}

#[cfg(test)]
mod held_disk_bytes_tests {
    use super::held_disk_bytes;

    /// **The under-count this exists for.** Pricing the manifest sees only
    /// `shard_NNN.bin`, so the live node measured 33062 MB on disk against
    /// 31423 MB counted — and 1566 MB of that 1637 MB gap was one file kind,
    /// `tied_output_weight.bin`. A user who set a disk limit got a node using
    /// about 5% more than it believed.
    #[test]
    fn the_figure_includes_what_is_not_a_shard() {
        let data_dir = tempfile::tempdir().unwrap();
        let model = data_dir.path().join("models").join("tinyllama-1.1b");
        std::fs::create_dir_all(&model).unwrap();
        std::fs::write(model.join("shard_000.bin"), vec![0u8; 1000]).unwrap();
        std::fs::write(model.join("gguf_header.bin"), vec![0u8; 200]).unwrap();
        std::fs::write(model.join("tied_output_weight.bin"), vec![0u8; 500]).unwrap();
        std::fs::write(model.join("manifest.json"), vec![0u8; 30]).unwrap();
        // The sharp edge: bytes a failed download left behind.
        std::fs::write(model.join("shard_001.bin.tmp"), vec![0u8; 70]).unwrap();

        assert_eq!(
            held_disk_bytes(data_dir.path()),
            1800,
            "every file under the models directory is real disk"
        );
    }

    /// Several models add up, and a directory that cannot be read is 0 rather
    /// than a panic — this runs on a timer against a live filesystem.
    #[test]
    fn models_sum_and_a_missing_directory_is_zero() {
        let data_dir = tempfile::tempdir().unwrap();
        for (name, size) in [("a", 100usize), ("b", 250)] {
            let model = data_dir.path().join("models").join(name);
            std::fs::create_dir_all(&model).unwrap();
            std::fs::write(model.join("shard_000.bin"), vec![0u8; size]).unwrap();
        }
        assert_eq!(held_disk_bytes(data_dir.path()), 350);

        let empty = tempfile::tempdir().unwrap();
        assert_eq!(held_disk_bytes(empty.path()), 0, "no models directory yet");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use swarmllm_types::ContributionMode;

    fn mib(n: u64) -> u64 {
        n.saturating_mul(1024).saturating_mul(1024)
    }

    fn disk(free_mb: u64, total_mb: u64) -> Option<DiskSpace> {
        Some(DiskSpace {
            free_bytes: mib(free_mb),
            total_bytes: mib(total_mb),
        })
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
            let b = storage_budget(50_000, 50_000, &level, disk(158_000, 500_000), mib(18_000));
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

    /// A 50 GB ceiling configured on a 20 GB filesystem with ~15 GB free. A
    /// ceiling is not a promise the space exists; with the shard caps unlimited
    /// the node filled the disk instead of pruning (reported 2026-07-30).
    #[test]
    fn budget_is_clamped_to_free_disk() {
        let b = storage_budget(
            50_000,
            0,
            &ContributionMode::Moderate,
            disk(15_000, 20_000),
            0,
        );
        assert!(b.bytes < mib(50_000));
        assert_eq!(
            b.bytes,
            mib(15_000) - mib(20_000) / 100 * FREE_DISK_RESERVE_PCT
        );
        assert!(
            b.bytes < mib(15_000),
            "must leave headroom, not fill the disk"
        );
        assert_eq!(
            b.limited_by,
            StorageLimit::FreeDisk {
                free_mb: 15_000,
                reserve_mb: 2_000
            }
        );
    }

    /// Free space already excludes what this node holds, so the clamp is on
    /// held + free beyond the reserve. Clamping the TOTAL and subtracting held
    /// again under-counted the room by exactly what was held: 18 GB held with
    /// 30 GB free used to leave 6 GB of room, not 24.
    #[test]
    fn the_free_disk_clamp_counts_what_is_already_held() {
        let held = mib(18_000);
        let b = storage_budget(
            200_000,
            0,
            &ContributionMode::Moderate,
            disk(30_000, 60_000),
            held,
        );
        assert_eq!(b.bytes, held + mib(30_000) - mib(6_000));
        assert_eq!(b.remaining(held), mib(24_000));
    }

    /// **The defect the reserve replaced** (gotcha #795). A tester's 30 GB
    /// container whose limit was the disk itself: download 550 MB parts for as
    /// long as the budget offers room. "Held + 80% of free" kept offering room
    /// while a part fit in 80% of what was left: 50 parts, 148 MB free, the
    /// disk 99.5% full, into the owner's own fill safeguard. Now: 44 parts and
    /// 88.8% — a tenth of the disk stays free however long the node downloads.
    #[test]
    fn downloading_until_the_budget_says_stop_leaves_a_tenth_of_the_disk_free() {
        const PART: u64 = 550 * 1024 * 1024;
        let total = mib(30_720);
        // 3 GB of operating system and other files; everything else ours to take.
        let mut reading = StorageReading {
            max_storage_mb: 500_000,
            max_disk_mb: 500_000,
            contribution: ContributionMode::Maximum,
            disk: Some(DiskSpace {
                free_bytes: total - mib(3_072),
                total_bytes: total,
            }),
            held_bytes: 0,
        };
        let mut parts = 0;
        while reading.budget().remaining(reading.held_bytes) >= PART {
            reading = reading.with_added(PART);
            parts += 1;
        }
        let free = reading.disk.unwrap().free_bytes;
        assert!(
            free >= total / 10,
            "{parts} parts left {} MB free on a {} MB disk — under the tenth it must keep",
            free / mib(1),
            total / mib(1)
        );
        assert!(
            free < total / 10 + PART,
            "and it fills up to the reserve, not short of it ({} MB free)",
            free / mib(1)
        );
    }

    /// A part this node downloads or deletes moves bytes between "held" and
    /// "free", so a disk-limited budget must not move with it. If it did, the
    /// pressure prune reads would shift under each part it acts on.
    #[test]
    fn a_disk_limited_budget_does_not_move_with_what_this_node_holds() {
        let reading = StorageReading {
            max_storage_mb: 500_000,
            max_disk_mb: 500_000,
            contribution: ContributionMode::Moderate,
            disk: disk(12_000, 30_000),
            held_bytes: mib(15_000),
        };
        let before = reading.budget();
        assert!(matches!(before.limited_by, StorageLimit::FreeDisk { .. }));
        let after = reading.with_added(mib(550));
        assert_eq!(after.budget().bytes, before.bytes);
        assert!(after.pressure() > reading.pressure());
    }

    /// Plenty of free space must leave the configured budget untouched — this
    /// clamp is a safety net, not a second policy.
    #[test]
    fn ample_free_disk_does_not_reduce_the_budget() {
        let b = storage_budget(
            1024,
            0,
            &ContributionMode::Moderate,
            disk(500_000, 1_000_000),
            0,
        );
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
