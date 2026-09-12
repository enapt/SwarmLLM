//! How much memory one of our processes is actually holding.
//!
//! The dashboard's RAM bar is the daemon plus its model workers, and a worker
//! is where the model weights and the conversation caches live — so this figure
//! is the one a user checks when their machine feels full.
//!
//! It read a single number from `sysinfo`, and on macOS that number disagreed
//! with the machine. A tester on a 16 GB Mac mini reported the bar sitting at a
//! few hundred MB while the system was visibly using 13-15 GB, with the
//! `swarmllm` worker itself shown at ~13 GB in Activity Monitor — the tooltip
//! said `worker: 13 MB` for a process holding a 14B model (report #017). The
//! system-wide figure the same endpoint reads was correct throughout, so the
//! daemon could see the memory; only the per-process half collapsed.
//!
//! macOS keeps two accountings of a process's memory and they are not
//! interchangeable. `sysinfo` reports `pti_resident_size` from
//! `proc_pidinfo(PROC_PIDTASKINFO)`. Activity Monitor reports the memory
//! *footprint* — `ri_phys_footprint` from `proc_pid_rusage` — which is what
//! Apple's own guidance points at, and the two can differ by orders of
//! magnitude depending on how the pages were obtained: the footprint ledger
//! excludes clean file-backed pages, and resident-size accounting can miss
//! memory the footprint ledger charges.
//!
//! Rather than pick the accounting that happens to be right on one machine,
//! this reports the LARGEST of the readings available. Under-reporting is the
//! failure that was actually observed and the one that matters — a bar reading
//! near zero on a machine that is nearly full tells the user their node is idle
//! when it is the thing filling their memory. Over-reporting is bounded by the
//! machine's own total, which the bar is drawn against.
//!
//! Every platform except macOS has one reading, so this is `sysinfo`'s figure
//! there and the module is a straight pass-through.

/// Bytes `pid` is holding, by the most inclusive accounting this platform has.
///
/// `sysinfo_bytes` is what `sysinfo::Process::memory()` reported for the same
/// pid — passed in rather than read here, because the caller already holds a
/// refreshed `System` and refreshing a second one costs a scan per call.
pub fn resident_bytes(pid: u32, sysinfo_bytes: u64) -> u64 {
    sysinfo_bytes.max(platform_bytes(pid))
}

#[cfg(target_os = "macos")]
fn platform_bytes(pid: u32) -> u64 {
    // SAFETY: `proc_pid_rusage` fills the buffer it is given and reports
    // failure in its return value. `rusage_info_v2` is the flavour asked for
    // (`RUSAGE_INFO_V2`), so the kernel writes exactly this layout; the buffer
    // is fully initialised before any field is read, and nothing borrows from
    // it. A pid we may not inspect, or one that has exited, returns non-zero
    // and the zeroed buffer is discarded.
    unsafe {
        let mut info: libc::rusage_info_v2 = std::mem::zeroed();
        // `rusage_info_t` is itself `*mut c_void` (C's `typedef void
        // *rusage_info_t`), so the parameter is `void **` and the buffer is
        // cast to `*mut rusage_info_t` — one cast, not two. Casting through to
        // `*mut c_void` does not type-check, and this file only compiles on
        // macOS, so nothing on a Linux box would have said so.
        let rc = libc::proc_pid_rusage(
            pid as libc::c_int,
            libc::RUSAGE_INFO_V2,
            &mut info as *mut libc::rusage_info_v2 as *mut libc::rusage_info_t,
        );
        if rc != 0 {
            return 0;
        }
        // Both, because which one carries the model's weights depends on how
        // they were mapped, and this module's whole point is not to have to
        // decide that per machine.
        info.ri_phys_footprint.max(info.ri_resident_size)
    }
}

#[cfg(not(target_os = "macos"))]
fn platform_bytes(_pid: u32) -> u64 {
    // One accounting, and `sysinfo` already read it.
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The reported failure, as arithmetic: whatever the platform reading says,
    /// the answer is never SMALLER than what `sysinfo` reported. A worker
    /// holding 13 GB must not be published as 13 MB.
    #[test]
    fn the_figure_is_never_lower_than_what_sysinfo_reported() {
        let sysinfo_says = 13 * 1024 * 1024;
        assert!(resident_bytes(std::process::id(), sysinfo_says) >= sysinfo_says);
    }

    /// And a platform that reads nothing back leaves the figure exactly as it
    /// was, so this can only ever raise it.
    #[test]
    fn a_platform_with_nothing_to_add_changes_nothing() {
        // A pid that will not be running: positive (so it is a pid at all
        // rather than a wildcard) and past any plausible allocation, so the
        // platform read fails and contributes 0.
        let absent = i32::MAX as u32;
        assert_eq!(resident_bytes(absent, 4096), 4096);
        assert_eq!(resident_bytes(absent, 0), 0);
    }

    /// On macOS the real reading must actually arrive — a permanently-failing
    /// call would leave this silently equal to the figure it exists to correct,
    /// which is the shape of a guard that cannot fire.
    #[cfg(target_os = "macos")]
    #[test]
    fn this_process_reports_a_footprint_on_macos() {
        assert!(
            platform_bytes(std::process::id()) > 0,
            "proc_pid_rusage must answer for our own process, or the correction \
             never happens and the bar keeps under-reporting"
        );
    }
}
