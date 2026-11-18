//! Reading a held part in the background without taking the disk from the
//! person at the computer (FUTURE_WORK #477).
//!
//! The startup check used to hash every held part flat out, one after
//! another: a field report (2026-11-18) had 443 parts saturate a mechanical
//! disk for 10-15 minutes after every start, the computer close to unusable
//! meanwhile. [`super::parts_checked`] removes most of those reads; what is
//! still read (a part never checked here, one that changed, one due its
//! periodic re-check) goes through [`hash_file_paced`]:
//!
//! - **a ceiling** — at most [`BACKGROUND_READ_BYTES_PER_SEC`];
//! - **a busy disk slows it further** — a read much slower than the quickest
//!   recent one means something else is using the disk, and the pace drops to
//!   an eighth for [`BUSY_FOR`] after the last such read (only reads at a
//!   spinning disk's speed are compared, [`QUICKER_THAN_A_SPINNING_DISK`]);
//! - **large reads** ([`CHUNK_BYTES`]), so a mechanical disk's head comes back
//!   to this file as few times as possible.
//!
//! Prior art: Linux md's resync runs between `speed_limit_min` and
//! `speed_limit_max` and drops to the minimum while the array sees other I/O
//! (`is_mddev_idle`); ZFS and Btrfs scrubs yield to normal I/O; Windows and
//! macOS give background work a low I/O priority (`THREAD_MODE_BACKGROUND_BEGIN`,
//! `IOPOL_THROTTLE`). The OS priorities are not used here: they are three
//! platform paths, two of which this project cannot test, and Linux honours
//! `ioprio` only under some I/O schedulers. md itself judges "idle" by counting
//! the OTHER I/O the member disks completed (`is_mddev_idle`, reworked 2025),
//! counters macOS and Windows keep elsewhere. Judging the disk by our OWN read
//! times works on every platform and is checked by the tests below. A/B:
//! `SWARMLLM_PART_CHECK_PACE=0` reads flat out, as before.

use std::collections::VecDeque;
use std::io::Read;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::error::SwarmError;

/// The most a background check reads a second: 32 MiB. A first pass over the
/// field report's 443 parts (~220 GB) takes about two hours at it, every part
/// served meanwhile — the check never gated serving (registration at start
/// checks sizes only). A mechanical disk reads 100-200 MB/s, so the check holds
/// a fifth of it at most, less once anything else reads.
pub const BACKGROUND_READ_BYTES_PER_SEC: u64 = 32 * 1024 * 1024;

/// While the disk is busy with other work the check reads this fraction of
/// [`BACKGROUND_READ_BYTES_PER_SEC`] (4 MiB/s) — md's `speed_limit_min` idea.
const BUSY_DISK_DIVISOR: u64 = 8;

/// One read. A mechanical disk pays a seek every time its head comes back to
/// this file, so fewer, larger reads cost the person at the computer less.
pub const CHUNK_BYTES: usize = 4 * 1024 * 1024;

/// A whole-chunk read quicker than this came from the page cache or a
/// solid-state disk, and never enters the reads the quickest is taken over.
/// The fastest spinning disk needs ~15 ms for 4 MiB (~270 MB/s on its outer
/// tracks); the page cache answers in under a millisecond and an NVMe disk in
/// ~1.5 ms, with or without eight other readers (measured 2026-11-18). Judged
/// against a cached part, a mechanical disk's own speed — ~25 ms a read on a
/// desktop disk's outer tracks to ~85 ms on a laptop disk's inner ones — reads
/// as contention whatever floor the cached time is given; so a read that quick
/// is never compared with, and the slowdown serves the disks it is for.
const QUICKER_THAN_A_SPINNING_DISK: Duration = Duration::from_millis(10);

/// A whole-chunk read this many times slower than the quickest of the recent
/// ones is a disk somebody else is using.
const BUSY_WHEN_SLOWER_BY: u32 = 3;

/// How many recent whole-chunk reads the quickest is taken over (256 MiB).
const RECENT_READS: usize = 64;

/// How long the slower pace lasts after the last slow read.
pub const BUSY_FOR: Duration = Duration::from_secs(5);

/// The longest single sleep, so a stop is noticed within it.
const SLEEP_PIECE: Duration = Duration::from_millis(250);

/// How a background check reads: its ceiling, what it has seen of the disk,
/// and the flag that stops it mid-file at shutdown. One per checking task;
/// the disk's record carries over from one part to the next.
pub struct ReadPace {
    /// 0 = flat out (`SWARMLLM_PART_CHECK_PACE=0`).
    bytes_per_sec: u64,
    stop: Arc<AtomicBool>,
    disk: parking_lot::Mutex<DiskWatch>,
    /// Bytes read through this pace, partial reads included.
    read: AtomicU64,
}

impl ReadPace {
    /// The background check's pace, or flat out under
    /// `SWARMLLM_PART_CHECK_PACE=0` (read once per call — a task builds one).
    pub fn background(stop: Arc<AtomicBool>) -> Self {
        let flat_out = std::env::var("SWARMLLM_PART_CHECK_PACE").is_ok_and(|v| v.trim() == "0");
        Self::with_rate(
            if flat_out {
                0
            } else {
                BACKGROUND_READ_BYTES_PER_SEC
            },
            stop,
        )
    }

    /// A pace with its own ceiling (0 = none) — for tests and probes.
    pub fn with_rate(bytes_per_sec: u64, stop: Arc<AtomicBool>) -> Self {
        Self {
            bytes_per_sec,
            stop,
            disk: parking_lot::Mutex::new(DiskWatch::default()),
            read: AtomicU64::new(0),
        }
    }

    fn stopped(&self) -> bool {
        self.stop.load(Ordering::Acquire)
    }

    /// How many whole-chunk reads found the disk busy since this pace was
    /// built (a pass reports the difference).
    pub fn busy_reads(&self) -> u64 {
        self.disk.lock().busy_reads
    }

    /// How many bytes were read through this pace since it was built — what
    /// was READ, a stopped or failed read's share included, never a size
    /// assumed from a manifest.
    pub fn bytes_read(&self) -> u64 {
        self.read.load(Ordering::Relaxed)
    }

    /// Sleep `d` in pieces, giving up at a stop.
    fn sleep(&self, d: Duration) -> Result<(), SwarmError> {
        let until = Instant::now() + d;
        loop {
            if self.stopped() {
                return Err(stopped());
            }
            let left = until.saturating_duration_since(Instant::now());
            if left.is_zero() {
                return Ok(());
            }
            std::thread::sleep(left.min(SLEEP_PIECE));
        }
    }
}

/// What the check has seen of the disk: its recent whole-chunk read times and
/// until when it counts as busy.
#[derive(Default)]
struct DiskWatch {
    recent: VecDeque<Duration>,
    busy_until: Option<Instant>,
    busy_reads: u64,
}

impl DiskWatch {
    /// Note one read of `whole` chunk (a file's short last read says nothing
    /// comparable) that took `took`, at `now`; whether the disk counts as busy.
    fn note(&mut self, whole: bool, took: Duration, now: Instant) -> bool {
        if whole {
            if let Some(quickest) = self.recent.iter().min().copied() {
                if took > quickest * BUSY_WHEN_SLOWER_BY {
                    self.busy_until = Some(now + BUSY_FOR);
                    self.busy_reads += 1;
                }
            }
            if took >= QUICKER_THAN_A_SPINNING_DISK {
                if self.recent.len() == RECENT_READS {
                    self.recent.pop_front();
                }
                self.recent.push_back(took);
            }
        }
        self.busy_until.is_some_and(|until| now < until)
    }
}

/// The error a stopped read returns. Never judged: the task that set the stop
/// has already left (a stop is the node shutting down).
fn stopped() -> SwarmError {
    SwarmError::Io(std::io::Error::new(
        std::io::ErrorKind::Interrupted,
        "the background part check stopped",
    ))
}

/// Read until `buf` is full or the file ends; how many bytes.
fn read_up_to(file: &mut std::fs::File, buf: &mut [u8]) -> std::io::Result<usize> {
    let mut filled = 0;
    while filled < buf.len() {
        match file.read(&mut buf[filled..]) {
            Ok(0) => break,
            Ok(n) => filled += n,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
    Ok(filled)
}

/// The BLAKE3 hash of the file at `path` — the same answer as
/// [`super::shard::hash_file_blake3`] — read at `pace`.
pub fn hash_file_paced(path: &Path, pace: &ReadPace) -> Result<[u8; 32], SwarmError> {
    let mut file = std::fs::File::open(path).map_err(SwarmError::Io)?;
    let mut hasher = blake3::Hasher::new();
    let mut buf = vec![0u8; CHUNK_BYTES];
    loop {
        if pace.stopped() {
            return Err(stopped());
        }
        let began = Instant::now();
        let n = read_up_to(&mut file, &mut buf).map_err(SwarmError::Io)?;
        if n == 0 {
            break;
        }
        let took = began.elapsed();
        pace.read.fetch_add(n as u64, Ordering::Relaxed);
        hasher.update(&buf[..n]);
        if pace.bytes_per_sec == 0 {
            continue;
        }
        let busy = pace
            .disk
            .lock()
            .note(n == CHUNK_BYTES, took, Instant::now());
        let rate = if busy {
            pace.bytes_per_sec / BUSY_DISK_DIVISOR
        } else {
            pace.bytes_per_sec
        }
        .max(1);
        // This read's share of the second, less what reading and hashing it
        // already took.
        let share = Duration::from_secs_f64(n as f64 / rate as f64);
        pace.sleep(share.saturating_sub(began.elapsed()))?;
    }
    Ok(*hasher.finalize().as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn file_of(len: usize) -> (tempfile::TempDir, std::path::PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("part.bin");
        let bytes: Vec<u8> = (0..len).map(|i| (i % 251) as u8).collect();
        std::fs::write(&path, bytes).unwrap();
        (dir, path)
    }

    #[test]
    fn a_paced_read_hashes_to_what_a_plain_read_does() {
        // Two whole chunks and a short last one.
        let (_dir, path) = file_of(2 * CHUNK_BYTES + 12_345);
        let pace = ReadPace::with_rate(0, Arc::new(AtomicBool::new(false)));
        assert_eq!(
            hash_file_paced(&path, &pace).unwrap(),
            super::super::shard::hash_file_blake3(&path).unwrap()
        );
        // What the pass reports as read is what was read.
        assert_eq!(pace.bytes_read(), (2 * CHUNK_BYTES + 12_345) as u64);
        // And an empty file.
        let (_dir, empty) = file_of(0);
        assert_eq!(
            hash_file_paced(&empty, &pace).unwrap(),
            super::super::shard::hash_file_blake3(&empty).unwrap()
        );
    }

    #[test]
    fn a_paced_read_takes_its_share_of_the_second() {
        // 12 MiB at 32 MiB/s: three reads of 125 ms each, at least 375 ms.
        let (_dir, path) = file_of(3 * CHUNK_BYTES);
        let pace = ReadPace::with_rate(32 * 1024 * 1024, Arc::new(AtomicBool::new(false)));
        let began = Instant::now();
        hash_file_paced(&path, &pace).unwrap();
        assert!(
            began.elapsed() >= Duration::from_millis(360),
            "12 MiB at 32 MiB/s read in {:?}",
            began.elapsed()
        );
        // Flat out the same file is read with no sleep at all.
        let flat = ReadPace::with_rate(0, Arc::new(AtomicBool::new(false)));
        let began = Instant::now();
        hash_file_paced(&path, &flat).unwrap();
        assert!(began.elapsed() < Duration::from_millis(360));
    }

    #[test]
    fn a_stop_ends_the_read_mid_file() {
        let (_dir, path) = file_of(3 * CHUNK_BYTES);
        let stop = Arc::new(AtomicBool::new(false));
        // 4 MiB a minute: without the stop this read would take minutes.
        let pace = ReadPace::with_rate(CHUNK_BYTES as u64 / 60, stop.clone());
        let stopper = {
            let stop = stop.clone();
            std::thread::spawn(move || {
                std::thread::sleep(Duration::from_millis(100));
                stop.store(true, Ordering::Release);
            })
        };
        let began = Instant::now();
        let err = hash_file_paced(&path, &pace).unwrap_err();
        stopper.join().unwrap();
        assert!(
            began.elapsed() < Duration::from_secs(2),
            "{:?}",
            began.elapsed()
        );
        assert!(
            matches!(&err, SwarmError::Io(e) if e.kind() == std::io::ErrorKind::Interrupted),
            "{err:?}"
        );
    }

    #[test]
    fn a_read_much_slower_than_the_recent_quickest_marks_the_disk_busy() {
        let t0 = Instant::now();
        let mut disk = DiskWatch::default();
        // Nothing to compare the first read with.
        assert!(!disk.note(true, Duration::from_millis(500), t0));
        for _ in 0..10 {
            assert!(!disk.note(true, Duration::from_millis(25), t0));
        }
        // Not three times the quickest (25 ms): the disk's own wobble.
        assert!(!disk.note(true, Duration::from_millis(70), t0));
        // Slower than three times the quickest.
        assert!(disk.note(true, Duration::from_millis(90), t0));
        assert_eq!(disk.busy_reads, 1);
        // Busy for BUSY_FOR after it, whatever the reads in between say.
        assert!(disk.note(true, Duration::from_millis(25), t0 + Duration::from_secs(1)));
        assert!(!disk.note(true, Duration::from_millis(25), t0 + BUSY_FOR));
    }

    #[test]
    fn reads_quicker_than_a_spinning_disk_never_set_the_speed_to_beat() {
        let t0 = Instant::now();
        let mut disk = DiskWatch::default();
        // A part still in the page cache, then one on a solid-state disk.
        for us in [300, 1_500, 8_000] {
            assert!(!disk.note(true, Duration::from_micros(us), t0), "{us} us");
        }
        assert!(disk.recent.is_empty());
        // The next part comes off a laptop's mechanical disk at 60-95 ms a
        // read, inner tracks and a seek included: its normal speed.
        for ms in [84, 60, 95, 70] {
            assert!(!disk.note(true, Duration::from_millis(ms), t0), "{ms} ms");
        }
        assert_eq!(disk.busy_reads, 0);
        // Three times ITS quickest is a disk somebody else is using.
        assert!(disk.note(true, Duration::from_millis(181), t0));
        assert_eq!(disk.busy_reads, 1);
    }

    #[test]
    fn a_files_short_last_read_is_never_compared() {
        let t0 = Instant::now();
        let mut disk = DiskWatch::default();
        disk.note(true, Duration::from_millis(25), t0);
        assert!(!disk.note(false, Duration::from_secs(1), t0));
        assert_eq!(disk.busy_reads, 0);
        // Nor does it enter the window: a short read's time says nothing of
        // what a whole one takes.
        assert_eq!(disk.recent.len(), 1);
    }
}
