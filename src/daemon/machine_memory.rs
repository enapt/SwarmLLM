//! How much memory THIS process may use — the one reading of it.
//!
//! The host's figures (`sysinfo`, i.e. `/proc/meminfo` on Linux), capped by the
//! memory limit of the cgroup this process runs in when one is set: a
//! container's `docker run --memory`, a Kubernetes limit, a `systemd-run -p
//! MemoryMax=` scope. Without the cap, a node in a 4 GB container planned and
//! admitted against the whole host's memory, and the kernel's OOM killer, not
//! the node, decided what did not fit. The JVM has read its container's limit
//! since JDK 10 for the same reason; .NET and Node.js do too.
//!
//! Five places read machine memory — the planner's RAM budget, the worker's
//! admission, the health monitor's advertisement, the stats API and the pool's
//! device stats — and they read it here, so none of them can disagree with the
//! others about how much there is (guard
//! `machine_memory_is_read_in_one_place`).
//!
//! **Available under a limit is the limit less what the kernel could not
//! reclaim** — `memory.current` less the file cache on BOTH LRU lists
//! (`active_file`, `inactive_file`) and reclaimable slab. That is LXCFS's
//! `MemAvailable` for a cgroup v2 container, and the same thing the host's
//! `MemAvailable` counts: the kernel drops clean cache, active or not, before it
//! OOM-kills anything under the limit. `memory.current` alone counts every page
//! of a model file the node has read, so it would look full and refuse work
//! (`sysinfo::System::cgroup_limits` stops there, which is why it is not used).
//!
//! ⚠ Not the kubelet's working set (`memory.current` less `inactive_file`
//! only), which this used first: that is an EVICTION metric, and it counts
//! active cache as used. A file read twice — a shard hashed, then loaded — is
//! on the active list, so after a few models the release gate's 13 GB scope
//! read 11.6 GB "used" with no model loaded, and every node refused a 2.8 GB
//! segment that .232, reading the host's figure, had served (gate .233 step 12e,
//! 2026-10-10).
//!
//! The cgroup is found the way the JVM finds it: this process's own path from
//! `/proc/self/cgroup`, and every ancestor up to the mount's root, so a limit
//! on an enclosing slice counts as well as one on the scope itself. Inside a
//! container with its own cgroup namespace the path is `/` and the mount's
//! root IS the container's cgroup.

/// Memory this process may use, in bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MachineMemory {
    pub total_bytes: u64,
    pub available_bytes: u64,
}

/// A cgroup memory limit and the room left under it, in bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CgroupLimit {
    limit_bytes: u64,
    headroom_bytes: u64,
}

impl MachineMemory {
    /// The host's figures under a cgroup's limit: the tighter of each.
    fn capped_by(self, limit: CgroupLimit) -> Self {
        Self {
            total_bytes: self.total_bytes.min(limit.limit_bytes),
            available_bytes: self.available_bytes.min(limit.headroom_bytes),
        }
    }
}

/// The memory this process may use right now, or `None` if the machine's
/// figures could not be read. Cheap (a few small reads), never cached here: a
/// caller on a hot path caches it itself.
pub fn machine_memory() -> Option<MachineMemory> {
    let mut sys = sysinfo::System::new();
    sys.refresh_memory();
    let host = MachineMemory {
        total_bytes: sys.total_memory(),
        available_bytes: sys.available_memory(),
    };
    if host.total_bytes == 0 {
        return None;
    }
    let read = |path: &str| std::fs::read_to_string(path).ok();
    Some(match cgroup_limit(&read, host.total_bytes) {
        Some(limit) => host.capped_by(limit),
        None => host,
    })
}

/// [`machine_memory`] in whole MB: `(total, available)`, `(0, 0)` when
/// unreadable — the shape the budget code has always taken.
pub fn machine_memory_mb() -> (u64, u64) {
    machine_memory().map_or((0, 0), |m| {
        (
            m.total_bytes / (1024 * 1024),
            m.available_bytes / (1024 * 1024),
        )
    })
}

/// The tightest memory limit on this process's cgroup or any ancestor, with
/// the room left under it. `None` when no limit below `host_total` is set —
/// which is the host's own root cgroup, where cgroup v2 has no `memory.max`
/// and cgroup v1 reports a limit of ~2^63.
fn cgroup_limit(read: &dyn Fn(&str) -> Option<String>, host_total: u64) -> Option<CgroupLimit> {
    let membership = read("/proc/self/cgroup")?;
    // cgroup v2: one line, `0::/path`.
    if let Some(path) = membership
        .lines()
        .find_map(|l| l.strip_prefix("0::"))
        .map(str::trim)
    {
        let found = tightest("/sys/fs/cgroup", path, host_total, |dir| {
            let limit = parse_bytes(read(&format!("{dir}/memory.max"))?.trim())?;
            let current = parse_bytes(read(&format!("{dir}/memory.current"))?.trim())?;
            let stat = read(&format!("{dir}/memory.stat"))?;
            let reclaimable = ["active_file", "inactive_file", "slab_reclaimable"]
                .iter()
                .filter_map(|key| stat_value(&stat, key))
                .sum::<u64>();
            Some((limit, current.saturating_sub(reclaimable)))
        });
        if found.is_some() {
            return found;
        }
    }
    // cgroup v1: `N:memory:/path` (the controller may share a line with others).
    let path = membership.lines().find_map(|l| {
        let mut parts = l.splitn(3, ':');
        let (_, controllers, path) = (parts.next()?, parts.next()?, parts.next()?);
        controllers
            .split(',')
            .any(|c| c == "memory")
            .then(|| path.trim())
    })?;
    tightest("/sys/fs/cgroup/memory", path, host_total, |dir| {
        let limit = parse_bytes(read(&format!("{dir}/memory.limit_in_bytes"))?.trim())?;
        let usage = parse_bytes(read(&format!("{dir}/memory.usage_in_bytes"))?.trim())?;
        let stat = read(&format!("{dir}/memory.stat"))?;
        let reclaimable = ["total_active_file", "total_inactive_file"]
            .iter()
            .filter_map(|key| stat_value(&stat, key))
            .sum::<u64>();
        Some((limit, usage.saturating_sub(reclaimable)))
    })
}

/// Walk `path` from the cgroup itself up to the mount's root, asking `at` for
/// `(limit, memory the kernel could not reclaim)` in each directory that
/// exists, and keep the limit leaving the LEAST room — an enclosing slice's
/// limit binds a scope inside it.
fn tightest(
    mount: &str,
    path: &str,
    host_total: u64,
    at: impl Fn(&str) -> Option<(u64, u64)>,
) -> Option<CgroupLimit> {
    let mut best: Option<CgroupLimit> = None;
    let mut rel = path.trim_end_matches('/').to_string();
    loop {
        let dir = if rel.is_empty() {
            mount.to_string()
        } else {
            format!("{mount}{rel}")
        };
        if let Some((limit, in_use)) = at(&dir) {
            if limit < host_total {
                let here = CgroupLimit {
                    limit_bytes: limit,
                    headroom_bytes: limit.saturating_sub(in_use),
                };
                best = Some(match best {
                    Some(b) => CgroupLimit {
                        limit_bytes: b.limit_bytes.min(here.limit_bytes),
                        headroom_bytes: b.headroom_bytes.min(here.headroom_bytes),
                    },
                    None => here,
                });
            }
        }
        if rel.is_empty() {
            return best;
        }
        rel.truncate(rel.rfind('/').unwrap_or(0));
    }
}

/// A byte count from a cgroup file; `max` (no limit) and anything unreadable
/// is `None`.
fn parse_bytes(s: &str) -> Option<u64> {
    s.parse().ok()
}

/// The value of `key` in a `memory.stat` file (`key value` per line).
fn stat_value(stat: &str, key: &str) -> Option<u64> {
    stat.lines().find_map(|l| {
        let mut it = l.split_whitespace();
        (it.next()? == key).then(|| it.next()?.parse().ok())?
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    const GIB: u64 = 1024 * 1024 * 1024;
    const HOST: u64 = 16 * GIB;

    fn fs(files: &[(&str, String)]) -> impl Fn(&str) -> Option<String> {
        let map: HashMap<String, String> = files
            .iter()
            .map(|(k, v)| (k.to_string(), v.clone()))
            .collect();
        move |p: &str| map.get(p).cloned()
    }

    /// The real reading, for a check by hand on Linux: under a scope with a
    /// limit (`systemd-run --user --scope -p MemoryMax=3G cargo dev-test --lib
    /// this_process -- --ignored --nocapture`) the total is the limit.
    #[test]
    #[ignore]
    fn this_process_reads_its_own_cgroup() {
        let m = machine_memory().expect("readable");
        println!(
            "machine_memory: total {} MB, available {} MB; /proc/self/cgroup: {:?}",
            m.total_bytes >> 20,
            m.available_bytes >> 20,
            std::fs::read_to_string("/proc/self/cgroup")
                .unwrap_or_default()
                .trim()
        );
    }

    fn stat(active: u64, inactive: u64) -> String {
        format!(
            "anon 123\nfile 456\nactive_file {active}\ninactive_file {inactive}\nslab_reclaimable 0\n"
        )
    }

    #[test]
    fn a_container_limit_caps_the_host_and_page_cache_is_room() {
        // Docker on cgroup v2: a private namespace, so the path is `/` and the
        // limit sits at the mount's root. 4 GiB limit, 1.5 GiB charged of
        // which 1 GiB is cache, half of it on the active list: 3.5 GiB of
        // room, not 2.5 (and not 3.0, the kubelet's working set).
        let read = fs(&[
            ("/proc/self/cgroup", "0::/\n".into()),
            ("/sys/fs/cgroup/memory.max", format!("{}\n", 4 * GIB)),
            (
                "/sys/fs/cgroup/memory.current",
                format!("{}\n", 3 * GIB / 2),
            ),
            ("/sys/fs/cgroup/memory.stat", stat(GIB / 2, GIB / 2)),
        ]);
        let limit = cgroup_limit(&read, HOST).expect("a limit is set");
        assert_eq!(limit.limit_bytes, 4 * GIB);
        assert_eq!(limit.headroom_bytes, 4 * GIB - GIB / 2);
        let host = MachineMemory {
            total_bytes: HOST,
            available_bytes: 12 * GIB,
        };
        let capped = host.capped_by(limit);
        assert_eq!(capped.total_bytes, 4 * GIB);
        assert_eq!(capped.available_bytes, 4 * GIB - GIB / 2);
    }

    /// The release gate's shape (step 12e, .233): nodes that had read and
    /// loaded several models' files left most of the scope's charge as cache
    /// on the ACTIVE list. The kubelet's working set counted it as used and
    /// every node refused a segment it had room for.
    #[test]
    fn active_file_cache_is_room_too() {
        let read = fs(&[
            ("/proc/self/cgroup", "0::/gate.scope\n".into()),
            (
                "/sys/fs/cgroup/gate.scope/memory.max",
                format!("{}", 13 * GIB),
            ),
            (
                "/sys/fs/cgroup/gate.scope/memory.current",
                format!("{}", 12 * GIB),
            ),
            (
                "/sys/fs/cgroup/gate.scope/memory.stat",
                format!(
                    "anon {}\nfile {}\nactive_file {}\ninactive_file {}\nslab_reclaimable {}\n",
                    4 * GIB,
                    15 * GIB / 2,
                    7 * GIB,
                    GIB / 2,
                    GIB / 2
                ),
            ),
        ]);
        let limit = cgroup_limit(&read, HOST).expect("the scope's limit");
        // 12 GiB charged, 8 of it reclaimable: 4 GiB in use, 9 GiB of room
        // (the working set would have left 1.5).
        assert_eq!(limit.headroom_bytes, 9 * GIB);
    }

    #[test]
    fn the_host_wins_where_it_has_less_room_than_the_limit() {
        let limit = CgroupLimit {
            limit_bytes: 8 * GIB,
            headroom_bytes: 7 * GIB,
        };
        let host = MachineMemory {
            total_bytes: HOST,
            available_bytes: 2 * GIB,
        };
        assert_eq!(host.capped_by(limit).available_bytes, 2 * GIB);
        assert_eq!(host.capped_by(limit).total_bytes, 8 * GIB);
    }

    #[test]
    fn a_limit_on_an_enclosing_slice_binds_the_scope_inside_it() {
        // A host scope with no limit of its own inside a 6 GiB slice — the
        // release gate's safety scope holding the rig's nodes has this shape.
        let read = fs(&[
            (
                "/proc/self/cgroup",
                "0::/user.slice/gate.slice/node.scope\n".into(),
            ),
            (
                "/sys/fs/cgroup/user.slice/gate.slice/node.scope/memory.max",
                "max\n".into(),
            ),
            (
                "/sys/fs/cgroup/user.slice/gate.slice/node.scope/memory.current",
                format!("{GIB}"),
            ),
            (
                "/sys/fs/cgroup/user.slice/gate.slice/node.scope/memory.stat",
                stat(0, 0),
            ),
            (
                "/sys/fs/cgroup/user.slice/gate.slice/memory.max",
                format!("{}", 6 * GIB),
            ),
            (
                "/sys/fs/cgroup/user.slice/gate.slice/memory.current",
                format!("{}", 5 * GIB),
            ),
            (
                "/sys/fs/cgroup/user.slice/gate.slice/memory.stat",
                stat(0, GIB),
            ),
            ("/sys/fs/cgroup/user.slice/memory.max", "max\n".into()),
            (
                "/sys/fs/cgroup/user.slice/memory.current",
                format!("{}", 9 * GIB),
            ),
            ("/sys/fs/cgroup/user.slice/memory.stat", stat(0, 0)),
        ]);
        let limit = cgroup_limit(&read, HOST).expect("the slice's limit");
        assert_eq!(limit.limit_bytes, 6 * GIB);
        // 6 GiB less a 4 GiB working set (5 charged, 1 reclaimable).
        assert_eq!(limit.headroom_bytes, 2 * GIB);
    }

    #[test]
    fn of_two_limits_the_one_leaving_less_room_decides() {
        let read = fs(&[
            ("/proc/self/cgroup", "0::/outer/inner\n".into()),
            (
                "/sys/fs/cgroup/outer/inner/memory.max",
                format!("{}", 3 * GIB),
            ),
            (
                "/sys/fs/cgroup/outer/inner/memory.current",
                format!("{GIB}"),
            ),
            ("/sys/fs/cgroup/outer/inner/memory.stat", stat(0, 0)),
            ("/sys/fs/cgroup/outer/memory.max", format!("{}", 8 * GIB)),
            (
                "/sys/fs/cgroup/outer/memory.current",
                format!("{}", 7 * GIB),
            ),
            ("/sys/fs/cgroup/outer/memory.stat", stat(0, 0)),
        ]);
        let limit = cgroup_limit(&read, HOST).unwrap();
        assert_eq!(limit.limit_bytes, 3 * GIB);
        // inner leaves 2 GiB, outer 1 GiB: the outer is what is really left.
        assert_eq!(limit.headroom_bytes, GIB);
    }

    #[test]
    fn the_hosts_own_root_cgroup_sets_no_limit() {
        // cgroup v2's root has no memory.max at all; an unlimited scope says
        // `max`. Neither caps anything.
        let read = fs(&[
            ("/proc/self/cgroup", "0::/init.scope\n".into()),
            ("/sys/fs/cgroup/init.scope/memory.max", "max\n".into()),
            ("/sys/fs/cgroup/init.scope/memory.current", format!("{GIB}")),
            ("/sys/fs/cgroup/init.scope/memory.stat", stat(0, 0)),
        ]);
        assert_eq!(cgroup_limit(&read, HOST), None);
        // No /proc/self/cgroup (not Linux): no limit.
        assert_eq!(cgroup_limit(&fs(&[]), HOST), None);
    }

    #[test]
    fn cgroup_v1_reads_the_memory_controller_and_ignores_the_unlimited_value() {
        let v1 = |limit: u64| {
            fs(&[
                (
                    "/proc/self/cgroup",
                    "12:pids:/docker/abc\n4:cpu,memory:/docker/abc\n1:name=systemd:/docker/abc\n"
                        .into(),
                ),
                // The container sees its own cgroup at the controller's root.
                (
                    "/sys/fs/cgroup/memory/memory.limit_in_bytes",
                    format!("{limit}\n"),
                ),
                (
                    "/sys/fs/cgroup/memory/memory.usage_in_bytes",
                    format!("{GIB}\n"),
                ),
                (
                    "/sys/fs/cgroup/memory/memory.stat",
                    format!(
                        "rss 1\ntotal_active_file {}\ntotal_inactive_file {}\n",
                        GIB / 4,
                        GIB / 4
                    ),
                ),
            ])
        };
        let limit = cgroup_limit(&v1(2 * GIB), HOST).expect("a v1 limit");
        assert_eq!(limit.limit_bytes, 2 * GIB);
        assert_eq!(limit.headroom_bytes, 2 * GIB - GIB / 2);
        // v1's "unlimited" is ~2^63, far above the host: no limit.
        assert_eq!(cgroup_limit(&v1(9_223_372_036_854_771_712), HOST), None);
    }
}
