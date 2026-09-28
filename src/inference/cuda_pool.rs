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

use candle_core::Device;
use std::time::Duration;

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

/// Bytes this process may allocate on `device` from memory its pool already
/// holds — added to `cuMemGetInfo`'s free figure wherever this process asks
/// "how much room do I have". Zero when unknown: conservative, never larger
/// than the truth.
pub(crate) fn reusable_bytes(device: &Device) -> u64 {
    usage(device).map_or(0, PoolUsage::reusable)
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

#[cfg(feature = "candle-cuda")]
mod cuda {
    use super::PoolUsage;
    use candle_core::cuda_backend::cudarc::driver::{result, sys, DriverError};

    /// The device's CURRENT pool — the one `cuMemAllocAsync` draws from — or
    /// `None` when cudarc allocates without one.
    fn current_pool(
        dev: &candle_core::CudaDevice,
    ) -> Result<Option<sys::CUmemoryPool>, DriverError> {
        let stream = dev.cuda_stream();
        let ctx = stream.context();
        if !ctx.has_async_alloc() {
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
    fn an_idle_worker_hands_memory_back_well_inside_the_idle_unload() {
        // The daemon retires an idle worker after `idle_unload_secs` (300 by
        // default); trimming later than that would never happen at all.
        assert!(IDLE_TRIM < Duration::from_secs(300));
        assert!(IDLE_TRIM >= Duration::from_secs(10));
    }
}
