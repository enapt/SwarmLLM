//! Where a slow network-loop arm's time went (#314).
//!
//! `DIAG: network event loop stalled` names the arm that took the time; this
//! says what the arm's THREAD was doing meanwhile. An arm with no `.await` in
//! it (`liveness_interval` only writes a map in memory) cannot be held up by
//! another task's commit, the stall the database wrapper fixed: that wait lands
//! before the arm starts. A tester's node logged 289 ms in such arms all the
//! while a large copy filled its disk (field report 2026-10-20), so the
//! thread itself was held — a lock, the disk, a page fault, or the processor.
//!
//! On Linux the thread's own counters (`getrusage(RUSAGE_THREAD)`, `man 2
//! getrusage`) tell those apart: time on the processor against time off it,
//! the waits (each block on a lock or the disk is one), the times another
//! thread took the processor, and the page faults that had to read the disk —
//! code or memory a copy flooding the page cache pushed out. One syscall at
//! each arm's start, a second only when the arm is slow.

use std::time::{Duration, Instant};

/// When an arm began, and its thread's counters then.
#[derive(Clone, Copy)]
pub(super) struct ArmStart {
    at: Instant,
    thread: Option<(std::thread::ThreadId, Counters)>,
}

impl ArmStart {
    pub(super) fn now() -> Self {
        Self {
            at: Instant::now(),
            thread: Counters::of_this_thread().map(|c| (std::thread::current().id(), c)),
        }
    }

    pub(super) fn elapsed(&self) -> Duration {
        self.at.elapsed()
    }

    /// What this thread spent since the arm began — `None` where the arm ended
    /// on another thread (an `.await` moved the task: the counters then belong
    /// to two threads) or the platform keeps none.
    pub(super) fn thread_spent(&self) -> Option<ThreadSpent> {
        let (thread, then) = self.thread?;
        if std::thread::current().id() != thread {
            return None;
        }
        let now = Counters::of_this_thread()?;
        let wall_us = u64::try_from(self.at.elapsed().as_micros()).unwrap_or(u64::MAX);
        let on_cpu_us = now.cpu_us.saturating_sub(then.cpu_us);
        Some(ThreadSpent {
            on_cpu_ms: on_cpu_us / 1000,
            off_cpu_ms: wall_us.saturating_sub(on_cpu_us) / 1000,
            disk_faults: now.disk_faults.saturating_sub(then.disk_faults),
            waits: now.waits.saturating_sub(then.waits),
            preempted: now.preempted.saturating_sub(then.preempted),
        })
    }
}

/// A slow arm's thread, from its start to now.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct ThreadSpent {
    pub(super) on_cpu_ms: u64,
    /// Wall time not on the processor: blocked, or waiting for a processor.
    pub(super) off_cpu_ms: u64,
    /// Page faults that read the disk (`ru_majflt`).
    pub(super) disk_faults: u64,
    /// Times the thread gave up the processor to wait (`ru_nvcsw`).
    pub(super) waits: u64,
    /// Times another thread took the processor from it (`ru_nivcsw`).
    pub(super) preempted: u64,
}

impl ThreadSpent {
    /// The reading in a few words, for the log line: which of the causes the
    /// counters point at. A judgement, so the counters are logged beside it.
    pub(super) fn likely(&self) -> &'static str {
        if self.off_cpu_ms <= self.on_cpu_ms {
            "work on the processor"
        } else if self.disk_faults > 0 {
            "page faults that read the disk"
        } else if self.waits > 0 {
            "blocked (a lock, the disk or another wait)"
        } else if self.preempted > 0 {
            "waiting for a processor"
        } else {
            "off the processor, cause not counted"
        }
    }
}

#[derive(Clone, Copy)]
struct Counters {
    cpu_us: u64,
    disk_faults: u64,
    waits: u64,
    preempted: u64,
}

impl Counters {
    #[cfg(target_os = "linux")]
    fn of_this_thread() -> Option<Self> {
        // SAFETY: `usage` is a valid, writable `rusage`; RUSAGE_THREAD reads
        // the calling thread's counters and writes nothing else.
        let mut usage: libc::rusage = unsafe { std::mem::zeroed() };
        if unsafe { libc::getrusage(libc::RUSAGE_THREAD, &mut usage) } != 0 {
            return None;
        }
        let micros = |t: libc::timeval| {
            (t.tv_sec.max(0) as u64)
                .saturating_mul(1_000_000)
                .saturating_add(t.tv_usec.max(0) as u64)
        };
        let count = |n: libc::c_long| n.max(0) as u64;
        Some(Self {
            cpu_us: micros(usage.ru_utime).saturating_add(micros(usage.ru_stime)),
            disk_faults: count(usage.ru_majflt),
            waits: count(usage.ru_nvcsw),
            preempted: count(usage.ru_nivcsw),
        })
    }

    #[cfg(not(target_os = "linux"))]
    fn of_this_thread() -> Option<Self> {
        None
    }
}

#[cfg(all(test, target_os = "linux"))]
mod tests {
    use super::*;

    /// A thread that sleeps through an arm spent it off the processor, in a
    /// wait; one that computes spent it on the processor. The two readings the
    /// stall line turns on.
    #[test]
    fn a_blocked_arm_and_a_busy_arm_read_differently() {
        let start = ArmStart::now();
        std::thread::sleep(Duration::from_millis(150));
        let blocked = start.thread_spent().expect("Linux keeps thread counters");
        assert!(blocked.off_cpu_ms >= 140, "{blocked:?}");
        assert!(blocked.waits >= 1, "{blocked:?}");
        assert_eq!(
            blocked.likely(),
            "blocked (a lock, the disk or another wait)"
        );

        // Spin until the THREAD has spent 60 ms on the processor, whatever else
        // the machine runs — the wall clock would make this flaky under load.
        let start = ArmStart::now();
        let cpu_at_start = Counters::of_this_thread().unwrap().cpu_us;
        let give_up = Instant::now() + Duration::from_secs(10);
        let mut x = 0u64;
        while Counters::of_this_thread().unwrap().cpu_us < cpu_at_start + 60_000
            && Instant::now() < give_up
        {
            x = std::hint::black_box(x.wrapping_mul(6364136223846793005).wrapping_add(1));
        }
        let busy = start.thread_spent().unwrap();
        assert!(busy.on_cpu_ms >= 55, "{busy:?}");
        assert!(
            busy.on_cpu_ms > busy.off_cpu_ms || busy.preempted > 0,
            "{busy:?}"
        );
    }

    /// Counters from two threads are not one arm's: an arm that ended on
    /// another thread reports nothing rather than a difference of strangers.
    #[test]
    fn an_arm_that_ended_on_another_thread_reports_nothing() {
        let start = ArmStart::now();
        let spent = std::thread::spawn(move || start.thread_spent())
            .join()
            .unwrap();
        assert_eq!(spent, None);
    }
}
