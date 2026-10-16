//! The graphics card's memory pool: a worker keeps what it frees while it is
//! serving, and hands it back once it has been idle for [`IDLE_TRIM`]
//! (`docs/FUTURE_WORK.md` #146, deep dive).
//!
//! **Why.** cudarc allocates every candle buffer with `cuMemAllocAsync` from
//! the device's current memory pool and frees into it. A pool's release
//! threshold defaults to ZERO — "all unused memory in the pool is released
//! back to the OS during every synchronization operation" (NVIDIA, "Using the
//! CUDA Stream-Ordered Memory Allocator", part 1) — and nothing in cudarc,
//! candle or this repo raised it. This worker synchronizes at every admission
//! (#121's budget reading) and whenever a token's logits come back to the
//! host, so each admission's buffers and each prefix-cache snapshot went back
//! to the driver and were fetched again fresh. On WSL2 a fresh allocation is a
//! trip through the host's GPU channel, and on the dev machine that trip grew
//! ~1000× over two days of Windows uptime (the same first snapshot copy:
//! 0.007 s at 8 h, 8.8 s at 49 h) until a worker admitting several chats
//! stalled 2-60 s per admission and the PC hung (gotchas #754, #755).
//!
//! **What others do.** PyTorch's `cudaMallocAsync` backend sets the threshold
//! to `UINT64_MAX`; RAPIDS RMM holds its pool for the process's life; ggml
//! plans its buffers once. NVIDIA (part 2): "Exclusive to a single process:
//! use the maximum release threshold", but "Shared among cooperating
//! processes: … set each process pool to an appropriate value to avoid any
//! one process monopolizing all device memory." A node's workers DO share the
//! card — one process per model, beside the desktop — and the driver reclaims
//! a pool's unused memory only "to enable an unrelated memory allocation
//! request in the SAME process to succeed" (part 1), never for another one.
//!
//! **So, two halves:**
//! - While a worker serves, its pool keeps everything
//!   ([`keep_freed_memory`], threshold max), and what the pool holds unused is
//!   counted as free FOR THIS PROCESS ([`reusable_bytes`], read by the KV
//!   budget) — otherwise #121's refusal would come straight back, since
//!   `cuMemGetInfo` reports kept memory as used.
//! - Once the worker has had nothing to do for [`IDLE_TRIM`], it hands the
//!   unused part back ([`trim`]), so another worker, or a model the daemon
//!   wants to load, sees the room. The daemon reads the card through
//!   `nvidia-smi`, which cannot tell a worker's kept memory from another
//!   program's.
//!
//! `SWARMLLM_CUDA_POOL_KEEP=0` leaves the pool at the driver's default and
//! never trims — the A/B inside one binary.
//!
//! **And a measurement of the slowdown itself** ([`probe_once`]): the first
//! time a worker picks the card it times `PROBE_ALLOCATIONS` fresh
//! allocations straight from the driver, the shape of the step that slowed.
//! Every worker logs the figure; past [`SLOW_FRESH_ALLOCATIONS`] the daemon
//! tells the owner, once, that restarting Windows restores it
//! ([`SlowCardNotice`]). Nothing here changes what the node does — the pool
//! above and `card_pace` are the defences; this is the diagnosis the owner can
//! act on, which neither of them gives. microsoft/WSL#41701 (open, 2026-10-01)
//! reports the same shape with no fix from Microsoft or NVIDIA; the one remedy
//! reported to work besides a reboot is `pnputil /restart-device` on the card,
//! too risky to suggest for a card that drives the display. The PC here
//! crashed twice at 52-54 h of Windows uptime (gotchas #754, #762).

use candle_core::Device;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

/// How long a worker with nothing to do keeps its freed card memory.
///
/// A conversation's next turn, an agent's next call and a batch of chats all
/// arrive within seconds of the last; a person reading a long reply may take
/// a minute or two, and pays one fresh allocation — what every admission paid
/// before this module. Well inside the daemon's own idle unload
/// (`inference.idle_unload_secs`, 5 minutes by default), which retires the
/// whole worker and frees everything anyway.
pub(crate) const IDLE_TRIM: Duration = Duration::from_secs(60);

/// A pool's two figures, in bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PoolUsage {
    /// Card memory the pool holds, in use or not (`RESERVED_MEM_CURRENT`).
    pub(crate) reserved: u64,
    /// Of that, what live buffers occupy (`USED_MEM_CURRENT`).
    pub(crate) used: u64,
}

impl PoolUsage {
    /// What this process can allocate from the pool without asking the driver.
    pub(crate) fn reusable(self) -> u64 {
        self.reserved.saturating_sub(self.used)
    }
}

/// Whether the pool keeps what it frees (on unless `SWARMLLM_CUDA_POOL_KEEP=0`).
pub(crate) fn keep_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED
        .get_or_init(|| keep_enabled_for(std::env::var("SWARMLLM_CUDA_POOL_KEEP").ok().as_deref()))
}

/// The switch's reading of the variable, apart from its `OnceLock` so a test
/// can ask it twice.
fn keep_enabled_for(v: Option<&str>) -> bool {
    !matches!(v, Some("0") | Some("off") | Some("false"))
}

/// Raise the release threshold of `device`'s memory pool to the maximum, once
/// per process. A no-op on the processor, on a card without memory pools
/// (cudarc then allocates synchronously and there is no pool to configure),
/// and under `SWARMLLM_CUDA_POOL_KEEP=0`.
///
/// Called by the split loader's device choice, the one place a worker picks
/// the card, so no model reaches the card without it.
pub(crate) fn keep_freed_memory(device: &Device) {
    #[cfg(feature = "candle-cuda")]
    if let Device::Cuda(dev) = device {
        static DONE: std::sync::OnceLock<()> = std::sync::OnceLock::new();
        DONE.get_or_init(|| {
            if !keep_enabled() {
                tracing::info!(
                    "DIAG: card memory pool left at the driver default — freed memory goes \
                     back at every synchronize (SWARMLLM_CUDA_POOL_KEEP=0)"
                );
                return;
            }
            match cuda::set_release_threshold(dev, u64::MAX) {
                Ok(true) => tracing::info!(
                    "DIAG: card memory pool keeps freed memory while serving \
                     (release threshold max; handed back after {} s idle)",
                    IDLE_TRIM.as_secs()
                ),
                Ok(false) => tracing::info!(
                    "DIAG: card has no memory pool — allocations are synchronous, nothing to keep"
                ),
                Err(e) => tracing::warn!(
                    error = %e,
                    "card memory pool: could not raise the release threshold — freed \
                     memory goes back at every synchronize, as before"
                ),
            }
        });
    }
    let _ = device;
}

/// The pool's figures for `device`, or `None` on the processor, on a card
/// without memory pools, or when the driver will not say.
pub(crate) fn usage(device: &Device) -> Option<PoolUsage> {
    #[cfg(feature = "candle-cuda")]
    if let Device::Cuda(dev) = device {
        return cuda::usage(dev).ok().flatten();
    }
    let _ = device;
    None
}

/// The most `device`'s pool has had in use since the last call (since the
/// process started, the first time), the count then started again — what one
/// forward needed at its peak, when called after each (`prof::dump_and_reset`).
/// `None` on the processor or a card without a pool.
pub(crate) fn take_peak_use(device: &Device) -> Option<u64> {
    #[cfg(feature = "candle-cuda")]
    if let Device::Cuda(dev) = device {
        return cuda::take_peak_use(dev).ok().flatten();
    }
    let _ = device;
    None
}

/// [`take_peak_use`] without the reset: the peak since the last reset, for a
/// reader that must not take it from the profile's per-forward report
/// (`SplitModel`'s prompt measurement, #138 (1e)).
pub(crate) fn peak_use(device: &Device) -> Option<u64> {
    #[cfg(feature = "candle-cuda")]
    if let Device::Cuda(dev) = device {
        return cuda::peak_use(dev).ok().flatten();
    }
    let _ = device;
    None
}

/// Bytes this process may allocate on `device` from memory its pool already
/// holds — added to `cuMemGetInfo`'s free figure wherever this process asks
/// "how much room do I have". Zero when unknown: conservative, never larger
/// than the truth.
pub(crate) fn reusable_bytes(device: &Device) -> u64 {
    usage(device).map_or(0, PoolUsage::reusable)
}

/// Make `device`'s pool hold one free block of `bytes` — one growth at most,
/// none when it already holds one — so the forward that follows allocates from
/// memory the pool has. Called before a prompt forward that sends RAM-held
/// weights to the card (`SplitModel::begin_prompt_measure`, FUTURE_WORK #292).
///
/// **Why.** Under WSL2, once a process has sent pageable copies to the card,
/// growing the pool stalls the calling thread for 0.2-9 s at a time, the card
/// idle: a probe whose growths took 0.5-7 ms stalled 0.16-2.2 s at 11 of 30
/// once it had copied 1.2 GB (`growprobe.py`, Windows up 87 h), and a Gemma 4
/// 26B-A4B prompt with its experts in RAM read its first 512-token chunk in
/// 23.6-24.8 s, and in 8.6-8.7 s primed (gotcha #883).
/// A pool that holds the forward's memory first grows nowhere in it (the same
/// probe primed with 1.2 GB: every allocation inside it 0.0 ms, after the
/// copies too). NVIDIA's sample of the stream-ordered allocator raises the
/// release threshold so that "the application is no longer allocating/freeing
/// memory from the OS" at steady state (`streamOrderedAllocation.cu`); RAPIDS
/// RMM's pool takes its initial size up front. The pool keeps the block
/// ([`keep_freed_memory`]) and counts it as this process's room
/// ([`reusable_bytes`]), so no budget reading moves. Returns how long it took,
/// `None` when nothing was done: the processor, no pool, a capture under way,
/// `SWARMLLM_PROMPT_POOL_PRIME=0` (the A/B), or the driver refused.
pub(crate) fn prime(device: &Device, bytes: u64) -> Option<Duration> {
    static ON: OnceLock<bool> = OnceLock::new();
    let on = *ON.get_or_init(|| std::env::var("SWARMLLM_PROMPT_POOL_PRIME").as_deref() != Ok("0"));
    if !on || bytes == 0 {
        return None;
    }
    #[cfg(feature = "candle-cuda")]
    if let Device::Cuda(dev) = device {
        let started = Instant::now();
        return match cuda::prime(dev, bytes as usize) {
            Ok(true) => Some(started.elapsed()),
            Ok(false) => None,
            Err(e) => {
                tracing::debug!(error = %e, bytes, "card pool: could not be primed");
                None
            }
        };
    }
    let _ = device;
    None
}

/// Hand what the pool holds unused back to the driver, after synchronizing so
/// buffers freed on the stream count as unused. Returns the figures before
/// and after, or `None` where there is nothing to trim or it could not be read.
pub(crate) fn trim(device: &Device) -> Option<(PoolUsage, PoolUsage)> {
    #[cfg(feature = "candle-cuda")]
    if let Device::Cuda(dev) = device {
        return cuda::trim(dev).ok().flatten();
    }
    let _ = device;
    None
}

/// Fresh allocations [`probe_once`] times, each [`PROBE_BYTES_EACH`]: 64 MB
/// in 16 calls. The step that slowed (#146's deep dive) was a few hundred MB
/// in ~64 calls, and whether the cost grows per call or per byte is not
/// known — this has some of each, and costs ~ms on a healthy card.
#[cfg(any(feature = "candle-cuda", test))]
pub(crate) const PROBE_ALLOCATIONS: usize = 16;
/// See [`PROBE_ALLOCATIONS`].
#[cfg(any(feature = "candle-cuda", test))]
pub(crate) const PROBE_BYTES_EACH: usize = 4 << 20;

/// A probe at least this slow is worth telling the owner about. ~100× what a
/// healthy card takes; by #146's series (a few hundred MB: 0.007 s at 8 h of
/// Windows uptime, 2.3 s at 24 h, 8.8 s at 49 h) the probe crosses it at
/// roughly a day of uptime, before the 52-54 h where this PC crashed.
pub(crate) const SLOW_FRESH_ALLOCATIONS: Duration = Duration::from_millis(500);

/// [`SLOW_FRESH_ALLOCATIONS`], unless `SWARMLLM_CARD_PROBE_SLOW_MS` says
/// otherwise — for a check that must watch the notice fire end to end on a
/// healthy card (`0` makes every probe slow). Read in both processes: the
/// worker's log line and the daemon's notice agree on what "slow" means.
fn slow_threshold() -> Duration {
    static T: OnceLock<Duration> = OnceLock::new();
    *T.get_or_init(|| {
        std::env::var("SWARMLLM_CARD_PROBE_SLOW_MS")
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok())
            .map_or(SLOW_FRESH_ALLOCATIONS, Duration::from_millis)
    })
}

/// Whether this process reaches its card through WSL2's paravirtual GPU
/// (`/dev/dxg`) — a node under WSL2, or in a Docker Desktop container on it.
/// The one place a fresh card allocation is KNOWN to slow with the host's
/// uptime (microsoft/WSL#41701, gotchas #754/#755), so the one place a
/// prefix-cache snapshot is moved off the card to keep the pool's high-water
/// mark down (`prefix_cache::SnapshotHome::for_device`, #257). A native Linux
/// driver grows its pool fast at any uptime, and a native Windows build's
/// WDDM has not been measured. Read once.
pub(crate) fn card_is_reached_through_wsl() -> bool {
    static WSL: OnceLock<bool> = OnceLock::new();
    *WSL.get_or_init(|| cfg!(target_os = "linux") && std::path::Path::new("/dev/dxg").exists())
}

/// How often the owner is reminded while the probes stay slow: the slowdown
/// lasts until Windows restarts, and a notice per worker start would nag.
pub(crate) const SLOW_CARD_NOTICE_EVERY: Duration = Duration::from_secs(12 * 3600);

/// This worker's probe: `None` inside while it has not run, or could not.
static PROBE: OnceLock<Option<Duration>> = OnceLock::new();
static PROBE_REPORTED: AtomicBool = AtomicBool::new(false);

/// Whether the probe runs (on unless `SWARMLLM_CARD_PROBE=0`).
#[cfg(feature = "candle-cuda")]
fn probe_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| keep_enabled_for(std::env::var("SWARMLLM_CARD_PROBE").ok().as_deref()))
}

/// Time `PROBE_ALLOCATIONS` fresh allocations on `device`, once per worker,
/// and log the figure — every worker's start adds a point to the uptime
/// curve in `node.log`. A no-op on the processor. Called by the split
/// loader's device choice beside [`keep_freed_memory`], before the model's
/// weights take the memory.
pub(crate) fn probe_once(device: &Device) {
    #[cfg(feature = "candle-cuda")]
    if let Device::Cuda(dev) = device {
        PROBE.get_or_init(|| {
            if !probe_enabled() {
                return None;
            }
            match cuda::probe(dev) {
                Ok(took) => {
                    let slow = took >= slow_threshold();
                    tracing::info!(
                        took_ms = took.as_millis() as u64,
                        allocations = PROBE_ALLOCATIONS,
                        mb_each = PROBE_BYTES_EACH >> 20,
                        slow,
                        "DIAG: card allocation probe — fresh memory from the driver{}",
                        if slow {
                            " is SLOW (on WSL2 this grows with Windows uptime; restarting \
                             Windows restores it — microsoft/WSL#41701)"
                        } else {
                            ""
                        }
                    );
                    Some(took)
                }
                Err(e) => {
                    // Not enough free memory for the probe is an answer too:
                    // nothing to measure, and nothing the owner should hear.
                    tracing::debug!(error = %e, "card allocation probe did not run");
                    None
                }
            }
        });
    }
    let _ = device;
}

/// The probe's figure, handed out ONCE — the worker sends it to the daemon
/// (`WorkerMsg::CardAllocationProbe`) after the message that loaded a model.
pub(crate) fn take_probe_report() -> Option<Duration> {
    let took = (*PROBE.get()?)?;
    (!PROBE_REPORTED.swap(true, Ordering::Relaxed)).then_some(took)
}

/// The daemon's side: the slowest probe its workers reported since the owner
/// was last told, and when that was. Held by `ModelProcessPool`, written by
/// each worker's reader, read by the health monitor's tick.
#[derive(Debug, Default)]
pub(crate) struct SlowCardNotice {
    /// Milliseconds; 0 = nothing slow waiting to be told.
    pending_ms: AtomicU32,
    last_told: Mutex<Option<Instant>>,
}

impl SlowCardNotice {
    /// A worker's probe arrived.
    pub(crate) fn record(&self, took: Duration) {
        if took >= slow_threshold() {
            let ms = u32::try_from(took.as_millis()).unwrap_or(u32::MAX).max(1);
            self.pending_ms.fetch_max(ms, Ordering::Relaxed);
        }
    }

    /// The slowest probe to tell the owner about now, if any: at most once per
    /// [`SLOW_CARD_NOTICE_EVERY`]. A slow probe inside that window is dropped
    /// rather than kept — the owner has just been told the same thing.
    pub(crate) fn take(&self, now: Instant) -> Option<Duration> {
        let ms = self.pending_ms.swap(0, Ordering::Relaxed);
        if ms == 0 {
            return None;
        }
        let mut last = self.last_told.lock().unwrap_or_else(|e| e.into_inner());
        if last.is_some_and(|t| now.duration_since(t) < SLOW_CARD_NOTICE_EVERY) {
            return None;
        }
        *last = Some(now);
        Some(Duration::from_millis(u64::from(ms)))
    }
}

#[cfg(feature = "candle-cuda")]
mod cuda {
    use super::PoolUsage;
    use candle_core::cuda_backend::cudarc::driver::{result, sys, DriverError};

    /// [`super::PROBE_ALLOCATIONS`] allocations straight from the driver
    /// (`cuMemAlloc`, never the pool, which may already hold memory), timed
    /// with their frees. One untimed allocation first, so a context's
    /// first-allocation setup is not counted. Whatever was taken is freed on
    /// every path.
    pub(super) fn probe(dev: &candle_core::CudaDevice) -> Result<std::time::Duration, DriverError> {
        let stream = dev.cuda_stream();
        stream.context().bind_to_thread()?;
        // SAFETY: a bound context; each pointer is freed exactly once below.
        unsafe {
            let warm = result::malloc_sync(super::PROBE_BYTES_EACH)?;
            result::free_sync(warm)?;
            let started = std::time::Instant::now();
            let mut taken = Vec::with_capacity(super::PROBE_ALLOCATIONS);
            let mut failed = None;
            for _ in 0..super::PROBE_ALLOCATIONS {
                match result::malloc_sync(super::PROBE_BYTES_EACH) {
                    Ok(ptr) => taken.push(ptr),
                    Err(e) => {
                        failed = Some(e);
                        break;
                    }
                }
            }
            for ptr in taken {
                let _ = result::free_sync(ptr);
            }
            match failed {
                Some(e) => Err(e),
                None => Ok(started.elapsed()),
            }
        }
    }

    /// The device's CURRENT pool — the one `cuMemAllocAsync` draws from — or
    /// `None` when cudarc allocates without one, or while the device's stream
    /// is being captured: a pool call from the capturing thread invalidates
    /// the capture. The profile's per-forward report reads the pool at the end
    /// of a forward, which a captured decode step reaches inside its last
    /// group — so with `SWARMLLM_PROFILE=1` every captured step was refused
    /// and ran again uncaptured, two reports a position (gotcha #841).
    fn current_pool(
        dev: &candle_core::CudaDevice,
    ) -> Result<Option<sys::CUmemoryPool>, DriverError> {
        let stream = dev.cuda_stream();
        let ctx = stream.context();
        if !ctx.has_async_alloc() {
            return Ok(None);
        }
        if !matches!(
            stream.capture_status(),
            Ok(sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE)
        ) {
            return Ok(None);
        }
        ctx.bind_to_thread()?;
        // SAFETY: `cu_device` comes from a live context cudarc created.
        unsafe { result::device::get_mem_pool(ctx.cu_device()) }.map(Some)
    }

    fn attribute(
        pool: sys::CUmemoryPool,
        attr: sys::CUmemPool_attribute,
    ) -> Result<u64, DriverError> {
        let mut value: u64 = 0;
        // SAFETY: both attributes read here are `cuuint64_t`.
        unsafe {
            result::mem_pool::get_attribute(
                pool,
                attr,
                (&mut value as *mut u64).cast::<core::ffi::c_void>(),
            )?;
        }
        Ok(value)
    }

    fn read(pool: sys::CUmemoryPool) -> Result<PoolUsage, DriverError> {
        Ok(PoolUsage {
            reserved: attribute(
                pool,
                sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT,
            )?,
            used: attribute(
                pool,
                sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_USED_MEM_CURRENT,
            )?,
        })
    }

    /// The pool's high watermark of memory in use (`USED_MEM_HIGH`), left as
    /// it is.
    pub(super) fn peak_use(dev: &candle_core::CudaDevice) -> Result<Option<u64>, DriverError> {
        let Some(pool) = current_pool(dev)? else {
            return Ok(None);
        };
        attribute(
            pool,
            sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_USED_MEM_HIGH,
        )
        .map(Some)
    }

    /// The pool's high watermark of memory in use (`USED_MEM_HIGH`), then
    /// reset — the driver only lets it be set to zero — so the next call
    /// reads the peak since this one.
    pub(super) fn take_peak_use(dev: &candle_core::CudaDevice) -> Result<Option<u64>, DriverError> {
        let Some(pool) = current_pool(dev)? else {
            return Ok(None);
        };
        let peak = attribute(
            pool,
            sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_USED_MEM_HIGH,
        )?;
        let mut zero: u64 = 0;
        // SAFETY: the high watermark is a `cuuint64_t`.
        unsafe {
            result::mem_pool::set_attribute(
                pool,
                sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_USED_MEM_HIGH,
                (&mut zero as *mut u64).cast::<core::ffi::c_void>(),
            )?;
        }
        Ok(Some(peak))
    }

    /// One block of `bytes` from the device's pool, freed at once on its
    /// stream, then synchronized: the pool keeps it. `Ok(false)` without a
    /// pool, or while the stream is captured ([`current_pool`]).
    pub(super) fn prime(dev: &candle_core::CudaDevice, bytes: usize) -> Result<bool, DriverError> {
        if current_pool(dev)?.is_none() {
            return Ok(false);
        }
        let stream = dev.cuda_stream();
        // SAFETY: a bound context (`current_pool`); the block is never read or
        // written, and freed exactly once, on the stream it came from.
        unsafe {
            let block = result::malloc_async(stream.cu_stream(), bytes)?;
            result::free_async(block, stream.cu_stream())?;
        }
        stream.synchronize()?;
        Ok(true)
    }

    /// `Ok(false)` when the card has no pool to configure.
    pub(super) fn set_release_threshold(
        dev: &candle_core::CudaDevice,
        bytes: u64,
    ) -> Result<bool, DriverError> {
        let Some(pool) = current_pool(dev)? else {
            return Ok(false);
        };
        let mut value = bytes;
        // SAFETY: the release threshold is a `cuuint64_t`.
        unsafe {
            result::mem_pool::set_attribute(
                pool,
                sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
                (&mut value as *mut u64).cast::<core::ffi::c_void>(),
            )?;
        }
        Ok(true)
    }

    pub(super) fn usage(dev: &candle_core::CudaDevice) -> Result<Option<PoolUsage>, DriverError> {
        match current_pool(dev)? {
            Some(pool) => read(pool).map(Some),
            None => Ok(None),
        }
    }

    pub(super) fn trim(
        dev: &candle_core::CudaDevice,
    ) -> Result<Option<(PoolUsage, PoolUsage)>, DriverError> {
        let Some(pool) = current_pool(dev)? else {
            return Ok(None);
        };
        dev.cuda_stream().synchronize()?;
        let before = read(pool)?;
        // The ring a split's prompt copies land in (#138 (1d)) is this pool's
        // memory in use while it exists; given back first, the trim below
        // releases it with the rest. Failing leaves it where it was.
        if let Err(e) = dev.trim_copies() {
            tracing::debug!(error = %e, "card copy ring: could not give it back");
        }
        // SAFETY: a valid pool; only memory no live buffer holds is released.
        unsafe { result::mem_pool::trim_to(pool, 0)? };
        Ok(Some((before, read(pool)?)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_pool_keeps_freed_memory_unless_switched_off() {
        assert!(keep_enabled_for(None));
        assert!(keep_enabled_for(Some("1")));
        assert!(!keep_enabled_for(Some("0")));
        assert!(!keep_enabled_for(Some("off")));
        assert!(!keep_enabled_for(Some("false")));
    }

    #[test]
    fn only_memory_no_buffer_holds_is_reusable() {
        let u = PoolUsage {
            reserved: 3 << 30,
            used: 2 << 30,
        };
        assert_eq!(u.reusable(), 1 << 30);
        // The two attributes are read one after the other; a buffer allocated
        // between them must not wrap the figure into an enormous one.
        let racing = PoolUsage {
            reserved: 1 << 30,
            used: 2 << 30,
        };
        assert_eq!(racing.reusable(), 0);
    }

    #[test]
    fn the_processor_has_no_pool_and_nothing_kept() {
        let cpu = Device::Cpu;
        keep_freed_memory(&cpu);
        assert_eq!(usage(&cpu), None);
        assert_eq!(reusable_bytes(&cpu), 0);
        assert_eq!(trim(&cpu), None);
    }

    #[test]
    fn a_healthy_probe_tells_the_owner_nothing() {
        let notice = SlowCardNotice::default();
        notice.record(Duration::from_millis(3));
        notice.record(SLOW_FRESH_ALLOCATIONS.saturating_sub(Duration::from_millis(1)));
        assert_eq!(notice.take(Instant::now()), None);
    }

    #[test]
    fn a_slow_probe_is_told_once_per_window_and_the_slowest_is_what_is_told() {
        let notice = SlowCardNotice::default();
        let t0 = Instant::now();
        notice.record(Duration::from_millis(900));
        notice.record(Duration::from_millis(2_400));
        notice.record(Duration::from_millis(700));
        assert_eq!(notice.take(t0), Some(Duration::from_millis(2_400)));
        // Told: nothing more until a new slow probe arrives...
        assert_eq!(notice.take(t0 + Duration::from_secs(1)), None);
        // ...and one inside the window is dropped, not saved for later.
        notice.record(Duration::from_secs(3));
        assert_eq!(notice.take(t0 + Duration::from_secs(60)), None);
        assert_eq!(notice.take(t0 + SLOW_CARD_NOTICE_EVERY), None);
        // Past the window, a slow probe is told again.
        notice.record(Duration::from_secs(4));
        assert_eq!(
            notice.take(t0 + SLOW_CARD_NOTICE_EVERY),
            Some(Duration::from_secs(4))
        );
    }

    #[test]
    fn the_slow_threshold_sits_far_above_a_healthy_card_and_below_the_crash_hours() {
        // Healthy: #146's first copy of a few hundred MB took 0.007 s at 8 h of
        // uptime; the probe is a fraction of that work. 24 h: 2.3 s for the copy.
        assert!(SLOW_FRESH_ALLOCATIONS >= Duration::from_millis(100));
        assert!(SLOW_FRESH_ALLOCATIONS <= Duration::from_secs(2));
        assert_eq!(PROBE_ALLOCATIONS * PROBE_BYTES_EACH, 64 << 20);
    }

    #[test]
    fn the_processor_is_never_probed() {
        // Compared with what was there before: under the CUDA feature another
        // test in this process may have probed the card already.
        let before = PROBE.get().copied();
        probe_once(&Device::Cpu);
        assert_eq!(PROBE.get().copied(), before);
    }

    #[test]
    fn an_idle_worker_hands_memory_back_well_inside_the_idle_unload() {
        // The daemon retires an idle worker after `idle_unload_secs` (300 by
        // default); trimming later than that would never happen at all.
        assert!(IDLE_TRIM < Duration::from_secs(300));
        assert!(IDLE_TRIM >= Duration::from_secs(10));
    }
}
