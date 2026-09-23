use crate::backend::{BackendDevice, BackendStorage};
use crate::{CpuStorage, CpuStorageRef, DType, Layout, Result, Shape};
pub use candle_kernels as kernels;
pub use cudarc;
use cudarc::driver::CudaFunction;
use float8::F8E4M3;
use half::{bf16, f16};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};

use super::{CudaError, CudaStorage, CudaStorageSlice, WrapErr};

/// Unique identifier for cuda devices.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DeviceId(usize);

impl DeviceId {
    fn new() -> Self {
        // https://users.rust-lang.org/t/idiomatic-rust-way-to-generate-unique-id/33805
        use std::sync::atomic;
        static COUNTER: atomic::AtomicUsize = atomic::AtomicUsize::new(1);
        Self(COUNTER.fetch_add(1, atomic::Ordering::Relaxed))
    }
}

/// SwarmLLM patch: per-kernel-name launch counts, for answering which kernels
/// a decoded token actually spends its submissions on.
///
/// A token was measured at ~625 `cuLaunchKernel` calls on a 22-layer model —
/// 28 per layer, against roughly ten logical operations — and nsys on WSL2
/// gives no GPU-side kernel table, so the composition has to come from the
/// code. Counting in [`CudaDevice::get_or_load_func`] works because that is
/// the only path a candle kernel launch takes.
///
/// ⚠ **Does NOT see cuBLAS's own kernels** (`cudaLaunchKernel_v7000`, ~45 per
/// token), which are launched inside cuBLAS rather than through candle. Read
/// the total from `examples/decode_submissions.sh` and this for the breakdown.
///
/// Off unless `SWARMLLM_COUNT_KERNELS=1`, and then it takes a mutex per launch
/// — fine for a diagnostic run, not for a benchmark. **Never measure
/// throughput with this on.**
fn kernel_counts() -> &'static std::sync::Mutex<HashMap<String, u64>> {
    static C: std::sync::OnceLock<std::sync::Mutex<HashMap<String, u64>>> =
        std::sync::OnceLock::new();
    C.get_or_init(|| std::sync::Mutex::new(HashMap::new()))
}

/// SwarmLLM patch: the single answer to "is kernel counting on".
///
/// Public because the consumer that PRINTS the counts is in another crate and
/// must gate on the same answer — `inference::split::executor` only reaches its
/// reporting block when something asks for it, and before this was exported
/// that something could only be `SWARMLLM_PROFILE=1`. Setting the counting flag
/// and seeing nothing is the exact trap the comment on `forward_start` already
/// warns about for the profiler; two readers of one env var would be the other
/// way to get it wrong.
pub fn counting_kernels() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("SWARMLLM_COUNT_KERNELS").as_deref() == Ok("1"))
}

#[inline]
fn count_kernel_launch(fn_name: &str) {
    if !counting_kernels() {
        return;
    }
    if let Ok(mut m) = kernel_counts().lock() {
        *m.entry(fn_name.to_string()).or_insert(0) += 1;
    }
}

/// Drain the per-kernel launch counts, highest first. Empty unless
/// `SWARMLLM_COUNT_KERNELS=1`.
pub fn take_kernel_launch_counts() -> Vec<(String, u64)> {
    let mut v: Vec<(String, u64)> = match kernel_counts().lock() {
        Ok(mut m) => m.drain().collect(),
        Err(_) => return Vec::new(),
    };
    v.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    v
}

/// SwarmLLM patch: host→device copies, counted by the SOURCE LINE that asked
/// for them, under the same `SWARMLLM_COUNT_KERNELS=1` switch.
///
/// A decoded token was measured at ~30 `cuMemcpyHtoDAsync_v2` calls, FIXED per
/// token rather than per layer (30.3 on a 22-layer model, 31.8 on a 28-layer
/// one), and nsys on WSL2 cannot say whose they are: CPU sampling is
/// unavailable there, so `--cudabacktrace` has nothing to unwind with, and the
/// release binary is stripped. Each one is also an allocation and a free, so a
/// copy is three submissions on a path that is bound by submission COUNT.
///
/// Every host→device copy in candle goes through [`CudaDevice::clone_htod`] or
/// [`CudaDevice::memcpy_htod`] — nothing in this tree calls the cudarc stream
/// directly — so counting there is complete. Both are `#[track_caller]`, and so
/// is the chain above them that moves host data onto a device
/// (`Tensor::{new, from_vec, from_slice, to_device}` → `Device::storage*` →
/// `storage_from_*`), so the location recorded is the line in OUR code that
/// built the tensor, not a line inside candle. A copy candle makes for its own
/// reasons (a reduction's dims and strides, a strided op's layout) reports the
/// candle line of that op, which names the op.
fn htod_counts() -> &'static std::sync::Mutex<HashMap<String, u64>> {
    static C: std::sync::OnceLock<std::sync::Mutex<HashMap<String, u64>>> =
        std::sync::OnceLock::new();
    C.get_or_init(|| std::sync::Mutex::new(HashMap::new()))
}

#[inline]
fn count_htod_copy(at: &'static std::panic::Location<'static>) {
    if !counting_kernels() {
        return;
    }
    if let Ok(mut m) = htod_counts().lock() {
        *m.entry(format!("{}:{}", at.file(), at.line())).or_insert(0) += 1;
    }
}

/// Drain the host→device copy counts by source line, highest first. Empty
/// unless `SWARMLLM_COUNT_KERNELS=1`.
pub fn take_htod_copy_counts() -> Vec<(String, u64)> {
    let mut v: Vec<(String, u64)> = match htod_counts().lock() {
        Ok(mut m) => m.drain().collect(),
        Err(_) => return Vec::new(),
    };
    v.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    v
}

/// SwarmLLM patch: `SWARMLLM_CUDA_EVENT_TRACKING=1` keeps cudarc's
/// per-allocation read/write events on the default-stream device.
///
/// Off by default because that device only ever uses the default stream, so the
/// events synchronise nothing — see the argument in `BackendDevice::new`. Read
/// once and cached; it sits on the device-construction path, but the answer
/// must not change between two devices in one process.
fn cuda_event_tracking_requested() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("SWARMLLM_CUDA_EVENT_TRACKING").as_deref() == Ok("1"))
}

/// SwarmLLM patch: `SWARMLLM_CUDA_OWN_STREAM=1` gives every `CudaDevice` its
/// own explicitly created stream instead of the legacy null stream.
///
/// ⛔ **OFF by default, and the default was flipped back on 2026-09-22 after it
/// shipped broken.** CUDA refuses to capture a graph on the legacy stream, so
/// moving off it is the precondition for collapsing a token's ~513 launches and
/// ~1,300 alloc/free calls into one submission — but turning it on by default
/// made **every model emit garbage** in a full `--features cuda` build:
/// `给给给…` on tinyllama, `<|reserved_special_token_247|>…` on llama-3.2-3b.
/// The same binary with this off answers correctly, which is what isolated it.
///
/// ⚠ **It was verified clean under `--features candle-cuda`, and that is
/// exactly why it got through.** That feature set has no flash-attn and no
/// llama backend, so the attention path a release build actually uses was never
/// compiled, let alone run — gotcha #677 says in as many words that such a
/// binary is "NOT a drop-in for the release node". **A change whose blast
/// radius is every CUDA kernel in the process cannot be cleared by the cheap
/// gate.**
///
/// The cause is not yet established. Leading hypothesis: several `CudaDevice`s
/// are built (the daemon's capability probe and the shard loader) which used to
/// share the ONE legacy stream and now each get their own, so work that was
/// ordered by construction is now unordered with event tracking off.
/// **Do not re-enable without reproducing under `--features cuda` on a real
/// generation** — every unit test and the whole `candle-cuda` A/B passed while
/// this was broken.
///
/// Read once and cached; it sits on the device-construction path, and the
/// answer must not change between two devices in one process.
fn own_cuda_stream_requested() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("SWARMLLM_CUDA_OWN_STREAM").as_deref() == Ok("1"))
}

/// SwarmLLM patch: `SWARMLLM_ZERO_QMATMUL_BUFFERS=1` puts the zero-fill back
/// on the buffers [`CudaDevice::alloc_fully_overwritten`] hands out.
///
/// Read once and cached: this sits inside the per-op allocation path, which a
/// decoded token walks ~700 times, so a `getenv` per call would be its own
/// measurable cost.
fn zero_fully_overwritten_buffers() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("SWARMLLM_ZERO_QMATMUL_BUFFERS").as_deref() == Ok("1")
    })
}

struct CudaRng(cudarc::curand::CudaRng);
unsafe impl Send for CudaRng {}

pub struct ModuleStore {
    mdls: [Option<Arc<cudarc::driver::CudaModule>>; kernels::ALL_IDS.len()],
}

#[derive(Clone)]
pub struct CudaDevice {
    id: DeviceId,
    context: Arc<cudarc::driver::CudaContext>,
    modules: Arc<std::sync::RwLock<ModuleStore>>,
    custom_modules: Arc<std::sync::RwLock<HashMap<String, Arc<cudarc::driver::CudaModule>>>>,
    stream: Arc<cudarc::driver::CudaStream>,
    pub(crate) blas: Arc<cudarc::cublas::CudaBlas>,
    curand: Arc<Mutex<CudaRng>>,
    seed_value: Arc<RwLock<u64>>,
}

impl std::fmt::Debug for CudaDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "CudaDevice({:?})", self.id)
    }
}

impl CudaDevice {
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn alloc<T: cudarc::driver::DeviceRepr>(
        &self,
        len: usize,
    ) -> Result<cudarc::driver::CudaSlice<T>> {
        self.stream.alloc::<T>(len).w()
    }

    pub fn alloc_zeros<T: cudarc::driver::DeviceRepr + cudarc::driver::ValidAsZeroBits>(
        &self,
        len: usize,
    ) -> Result<cudarc::driver::CudaSlice<T>> {
        self.stream.alloc_zeros::<T>(len).w()
    }

    /// SwarmLLM patch: allocate a buffer whose every element the NEXT kernel
    /// writes, skipping the zero-fill `alloc_zeros` would submit.
    ///
    /// `alloc_zeros` is `alloc` **plus** a `cuMemsetD8Async`, and that memset
    /// is a full GPU submission — on a virtualised driver it costs about as
    /// much as a kernel launch (~10-12 us measured on WSL2) to zero a buffer
    /// that is overwritten in its entirety microseconds later. Measured on a
    /// 22-layer model: **323 memsets per decoded token**, 14.7 per layer, i.e.
    /// two per quantized matmul, for 3.8 ms of a 23 ms token. Decode on this
    /// box is bound by how many submissions a token costs, not by bandwidth,
    /// so a submission removed is time removed.
    ///
    /// ⚠ **Every caller must have READ the kernel that fills the buffer** and
    /// confirmed it ASSIGNS (never accumulates into) every element it owns.
    /// A kernel that leaves gaps, or one changed later to `+=`, turns this
    /// into garbage in a reply rather than a visible failure.
    ///
    /// `SWARMLLM_ZERO_QMATMUL_BUFFERS=1` restores the zero-fill, so the change
    /// can be A/B'd inside ONE binary — which is how its effect was
    /// attributed, per `.claude/rules/diagnosis.md` § 4.
    ///
    /// → `docs/invariants/inference.md` § "A decode token is bound by GPU
    /// submission count, not bandwidth"
    pub fn alloc_fully_overwritten<
        T: cudarc::driver::DeviceRepr + cudarc::driver::ValidAsZeroBits,
    >(
        &self,
        len: usize,
    ) -> Result<cudarc::driver::CudaSlice<T>> {
        if zero_fully_overwritten_buffers() {
            self.alloc_zeros::<T>(len)
        } else {
            unsafe { self.alloc::<T>(len) }
        }
    }

    // SwarmLLM patch: `#[track_caller]` + the count — see `count_htod_copy`.
    #[track_caller]
    pub fn memcpy_htod<
        T: cudarc::driver::DeviceRepr,
        Src: cudarc::driver::HostSlice<T> + ?Sized,
        Dst: cudarc::driver::DevicePtrMut<T>,
    >(
        &self,
        src: &Src,
        dst: &mut Dst,
    ) -> Result<()> {
        count_htod_copy(std::panic::Location::caller());
        self.stream.memcpy_htod(src, dst).w()
    }

    pub fn clone_dtoh<T: cudarc::driver::DeviceRepr, Src: cudarc::driver::DevicePtr<T>>(
        &self,
        src: &Src,
    ) -> Result<Vec<T>> {
        self.stream.clone_dtoh(src).w()
    }

    pub fn memcpy_dtod<
        T,
        Src: cudarc::driver::DevicePtr<T>,
        Dst: cudarc::driver::DevicePtrMut<T>,
    >(
        &self,
        src: &Src,
        dst: &mut Dst,
    ) -> Result<()> {
        self.stream.memcpy_dtod(src, dst).w()
    }

    pub fn memcpy_dtoh<
        T: cudarc::driver::DeviceRepr,
        Src: cudarc::driver::DevicePtr<T>,
        Dst: cudarc::driver::HostSlice<T>,
    >(
        &self,
        src: &Src,
        dst: &mut Dst,
    ) -> Result<()> {
        self.stream.memcpy_dtoh(src, dst).w()
    }

    // SwarmLLM patch: `#[track_caller]` + the count — see `count_htod_copy`.
    #[track_caller]
    pub fn clone_htod<T: cudarc::driver::DeviceRepr, Src: cudarc::driver::HostSlice<T> + ?Sized>(
        &self,
        src: &Src,
    ) -> Result<cudarc::driver::CudaSlice<T>> {
        count_htod_copy(std::panic::Location::caller());
        self.stream.clone_htod(src).w()
    }
}

pub struct CudaFunc {
    func: CudaFunction,
    stream: Arc<cudarc::driver::CudaStream>,
}

impl std::ops::Deref for CudaFunc {
    type Target = CudaFunction;

    fn deref(&self) -> &Self::Target {
        &self.func
    }
}

impl CudaFunc {
    pub fn into_cuda_function(self) -> CudaFunction {
        self.func
    }
}

#[macro_export]
macro_rules! builder_arg {
    ($b:ident, $($arg:expr),*) => {
        $(
            let __arg = $arg;
            $b.arg(&__arg);
        )*
    };
}

impl CudaFunc {
    pub fn builder(&self) -> cudarc::driver::LaunchArgs<'_> {
        self.stream.launch_builder(&self.func)
    }
}

impl CudaDevice {
    pub fn cuda_stream(&self) -> Arc<cudarc::driver::CudaStream> {
        self.stream.clone()
    }

    /// When turned on, all cuda tensors **created after calling this function** will
    /// not track uses via cuda events.
    ///
    /// # Safety
    ///
    /// It is up to the user to ensure proper synchronization between multiple streams:
    /// - Ensure that no tensor is freed before a use on another stream is finished.
    /// - Ensure that a tensor is not used on another stream before allocation on the
    ///   allocating stream finishes.
    /// - Ensure that a tensor is not written two concurrently by multiple streams.
    pub unsafe fn disable_event_tracking(&self) {
        self.context.disable_event_tracking()
    }

    pub fn is_event_tracking(&self) -> bool {
        self.context.is_event_tracking()
    }

    #[cfg(all(feature = "ug", not(target_arch = "wasm32")))]
    pub fn compile(
        &self,
        func_name: &'static str,
        kernel: candle_ug::lang::ssa::Kernel,
    ) -> Result<CudaFunc> {
        let mut buf = vec![];
        candle_ug::cuda::code_gen::gen(&mut buf, func_name, &kernel)?;
        let cuda_code = String::from_utf8(buf)?;
        let opts = cudarc::nvrtc::CompileOptions {
            use_fast_math: Some(true),
            ..Default::default()
        };
        let ptx = cudarc::nvrtc::safe::compile_ptx_with_opts(cuda_code, opts).w()?;
        let module = self.context.load_module(ptx).w()?;
        let func = module.load_function(func_name).w()?;
        Ok(CudaFunc {
            func,
            stream: self.stream.clone(),
        })
    }

    pub fn id(&self) -> DeviceId {
        self.id
    }

    pub fn get_or_load_custom_func(
        &self,
        fn_name: &str,
        module_name: &str,
        ptx: &str,
    ) -> Result<CudaFunc> {
        // SwarmLLM patch: count here TOO, not only in `get_or_load_func`.
        //
        // This is the path SwarmLLM's own fused kernels take
        // (`kernels/fused_decode.cu`, loaded as PTX). Counting only candle's
        // built-in kernels would make every fusion look better than it is: a
        // fused kernel replacing two candle ones would show as -2 launches per
        // layer when the truth is -1, and the instrument would be wrong in
        // exactly the direction that flatters the change it exists to judge.
        count_kernel_launch(fn_name);
        let ms = self.custom_modules.read().unwrap();
        if let Some(mdl) = ms.get(module_name).as_ref() {
            let func = mdl.load_function(fn_name).w()?;
            return Ok(CudaFunc {
                func,
                stream: self.stream.clone(),
            });
        }
        drop(ms);
        let mut ms = self.custom_modules.write().unwrap();
        let cuda_module = self.context.load_module(ptx.into()).w()?;
        ms.insert(module_name.to_string(), cuda_module.clone());
        let func = cuda_module.load_function(fn_name).w()?;
        Ok(CudaFunc {
            func,
            stream: self.stream.clone(),
        })
    }

    pub fn get_or_load_func(&self, fn_name: &str, mdl: &kernels::Module) -> Result<CudaFunc> {
        // SwarmLLM patch: the path every BUILT-IN candle kernel launch takes
        // (SwarmLLM's own PTX kernels go through `get_or_load_custom_func`,
        // which counts as well), so counting here answers
        // "which kernels does a token actually launch" — the question that
        // decides which fusion is worth doing. Off unless
        // `SWARMLLM_COUNT_KERNELS=1`; see `count_kernel_launch`.
        count_kernel_launch(fn_name);
        let ms = self.modules.read().unwrap();
        if let Some(mdl) = ms.mdls[mdl.index()].as_ref() {
            let func = mdl.load_function(fn_name).w()?;
            return Ok(CudaFunc {
                func,
                stream: self.stream.clone(),
            });
        }
        drop(ms);
        let mut ms = self.modules.write().unwrap();
        let cuda_module = self.context.load_module(mdl.ptx().into()).w()?;
        ms.mdls[mdl.index()] = Some(cuda_module.clone());
        let func = cuda_module.load_function(fn_name).w()?;
        Ok(CudaFunc {
            func,
            stream: self.stream.clone(),
        })
    }

    pub fn cublas_handle(&self) -> Arc<cudarc::cublas::CudaBlas> {
        self.blas.clone()
    }
}

impl CudaDevice {
    pub fn new_with_stream(ordinal: usize) -> Result<Self> {
        let context = cudarc::driver::CudaContext::new(ordinal).w()?;
        let stream = context.new_stream().w()?;
        let blas = cudarc::cublas::CudaBlas::new(stream.clone()).w()?;
        let curand = cudarc::curand::CudaRng::new(299792458, stream.clone()).w()?;
        let module_store = ModuleStore {
            mdls: [const { None }; kernels::ALL_IDS.len()],
        };
        Ok(Self {
            id: DeviceId::new(),
            context,
            stream,
            blas: Arc::new(blas),
            curand: Arc::new(Mutex::new(CudaRng(curand))),
            modules: Arc::new(std::sync::RwLock::new(module_store)),
            custom_modules: Arc::new(std::sync::RwLock::new(HashMap::new())),
            seed_value: Arc::new(RwLock::new(299792458)),
        })
    }
}

impl BackendDevice for CudaDevice {
    type Storage = CudaStorage;

    fn new(ordinal: usize) -> Result<Self> {
        let context = cudarc::driver::CudaContext::new(ordinal).w()?;
        // SwarmLLM patch: this constructor runs everything on the DEFAULT
        // stream, so cudarc's per-allocation events cannot be doing anything.
        //
        // cudarc creates a read event AND a write event for every `CudaSlice`
        // while `is_event_tracking()` is on (the default), and destroys both on
        // drop. Measured: **2,625 event API calls per decoded token** — four per
        // allocation, ~700 allocations a token — for ~1.5 ms of a 23 ms token.
        //
        // They exist to synchronise a buffer used across MULTIPLE streams.
        //
        // ⚠ **The invariant is ONE STREAM PER DEVICE, and buffers never
        // crossing devices.** It said "there is only ever one stream, process
        // wide" until 2026-09-22, which was true while every device took
        // `default_stream()` — cudarc hands back `cu_stream: null_mut()` for
        // that, the same legacy stream for all of them. Moving off it (below)
        // gives each device its OWN stream, so the old sentence would now be
        // false while the conclusion still holds. What it actually rests on:
        //   * This constructor is every production path into CUDA
        //     (`Device::new_cuda` / `cuda_if_available`), and it takes exactly
        //     one stream per device, whichever kind.
        //   * Same-stream ordering needs no events: `cuMemFreeAsync` on the
        //     allocating stream is ordered after work queued before it, which
        //     is the whole point of stream-ordered allocation.
        //   * Several devices ARE built — the daemon's capability probe and the
        //     shard loader — and candle gives them different `DeviceId`s and
        //     refuses to mix tensors across devices, so a buffer cannot reach
        //     another device's stream. **This bullet used to be a hypothetical
        //     about a constructor nobody called; since the migration it is the
        //     load-bearing one.**
        //   * `is_in_multi_stream_mode()` is FALSE by default, because
        //     `new_stream()` is what sets it and the default is back to
        //     `default_stream()`. cudarc's
        //     `is_managing_stream_synchronization()` —
        //     `is_in_multi_stream_mode() && is_event_tracking()` — is therefore
        //     false twice over, as it was before the migration attempt.
        //
        // ⚠ **Under `SWARMLLM_CUDA_OWN_STREAM=1` that changes**: multi-stream
        // mode becomes true, so `SWARMLLM_CUDA_EVENT_TRACKING=1` then stops
        // being a pure revert and also hands cudarc back stream-sync
        // management. It can only ADD synchronisation, so it stays a valid A/B
        // — but say which arm you are in when quoting it.
        //
        // ⚠⚠ **ANYONE GIVING ONE DEVICE A SECOND STREAM MUST RE-ENABLE THIS.**
        // That device's buffers would have nothing tracking their use, i.e. used
        // across streams unsynchronised, and the failure is a silently wrong
        // reply rather than an error. `docs/plans/local_decode_submissions.md`
        // § "Do NOT copy their concurrent streams" records why llama.cpp's
        // Q/K/V stream parallelisation is not wanted here regardless: our
        // bottleneck is the CPU issuing work, not the GPU idling between
        // dependent launches.
        //
        // → `docs/invariants/inference.md` § "A decode token is bound by GPU
        //   submission COUNT, not bandwidth"
        if !cuda_event_tracking_requested() {
            // SAFETY: one stream on this device, per the argument above. The
            // contract is that the caller orders cross-stream use; this device
            // has no second stream to order against.
            unsafe { context.disable_event_tracking() };
        }
        // SwarmLLM patch: an explicitly created (non-blocking) stream, NOT the
        // legacy null stream.
        //
        // **CUDA refuses to capture a graph on the legacy stream**, and a graph
        // is the one change that would collapse a decoded token's ~513 kernel
        // launches and ~1,300 allocation/free calls into a single submission.
        // Measured rather than assumed: `examples/cuda_graph_probe.cu` arm A
        // gets `cudaError 900` from `cudaStreamBeginCapture(cudaStreamLegacy)`
        // on this driver, and that arm is written so a SUCCESS would report the
        // claim wrong.
        //
        // Nothing else has to move with it, which is why this is one line:
        // `CudaBlas::new` and `CudaRng::new` are handed this stream (cublas via
        // `cublasSetStream_v2`), `candle-flash-attn` takes `dev.cuda_stream()`,
        // every launch goes through `self.stream.launch_builder`, and
        // `synchronize()` syncs `self.stream`. `default_stream()` had exactly
        // one use in this backend and this was it.
        //
        // ⚠ The stream is NON-BLOCKING, so it does NOT implicitly synchronise
        // with the legacy stream. Safe only because nothing in this process
        // uses the legacy stream any more — **a partial migration is worse than
        // either end state.** (A `llama`-feature build runs llama.cpp on its own
        // context and shares no buffers with candle.)
        //
        // ⛔ **Opt-in only** (`SWARMLLM_CUDA_OWN_STREAM=1`). Shipping it ON in
        // v0.3.199-alpha made every model emit garbage in a `--features cuda`
        // build while every test and the whole `candle-cuda` A/B stayed green —
        // see the note on `own_cuda_stream_requested`. The legacy stream is the
        // default until that is understood, which also means graph capture
        // stays unreachable by default, by design.
        let stream = if own_cuda_stream_requested() {
            context.new_stream().w()?
        } else {
            context.default_stream()
        };
        let blas = cudarc::cublas::CudaBlas::new(stream.clone()).w()?;
        let curand = cudarc::curand::CudaRng::new(299792458, stream.clone()).w()?;
        let module_store = ModuleStore {
            mdls: [const { None }; kernels::ALL_IDS.len()],
        };
        Ok(Self {
            id: DeviceId::new(),
            context,
            stream,
            blas: Arc::new(blas),
            curand: Arc::new(Mutex::new(CudaRng(curand))),
            modules: Arc::new(std::sync::RwLock::new(module_store)),
            custom_modules: Arc::new(std::sync::RwLock::new(HashMap::new())),
            seed_value: Arc::new(RwLock::new(299792458)),
        })
    }

    fn set_seed(&self, seed: u64) -> Result<()> {
        // We do not call set_seed but instead create a new curand object. This ensures that the
        // state will be identical and the same random numbers will be generated.
        let mut curand = self.curand.lock().unwrap();
        curand.0 = cudarc::curand::CudaRng::new(seed, self.stream.clone()).w()?;
        *self.seed_value.write().unwrap() = seed;
        Ok(())
    }

    fn get_current_seed(&self) -> Result<u64> {
        Ok(*self.seed_value.read().unwrap())
    }

    fn location(&self) -> crate::DeviceLocation {
        crate::DeviceLocation::Cuda {
            gpu_id: self.context.ordinal(),
        }
    }

    fn same_device(&self, rhs: &Self) -> bool {
        self.id == rhs.id
    }

    fn zeros_impl(&self, shape: &Shape, dtype: DType) -> Result<CudaStorage> {
        let elem_count = shape.elem_count();
        let slice = match dtype {
            DType::U8 => {
                let data = self.alloc_zeros::<u8>(elem_count)?;
                CudaStorageSlice::U8(data)
            }
            DType::U32 => {
                let data = self.alloc_zeros::<u32>(elem_count)?;
                CudaStorageSlice::U32(data)
            }
            DType::I16 => {
                let data = self.alloc_zeros::<i16>(elem_count)?;
                CudaStorageSlice::I16(data)
            }
            DType::I32 => {
                let data = self.alloc_zeros::<i32>(elem_count)?;
                CudaStorageSlice::I32(data)
            }
            DType::I64 => {
                let data = self.alloc_zeros::<i64>(elem_count)?;
                CudaStorageSlice::I64(data)
            }
            DType::BF16 => {
                let data = self.alloc_zeros::<bf16>(elem_count)?;
                CudaStorageSlice::BF16(data)
            }
            DType::F16 => {
                let data = self.alloc_zeros::<f16>(elem_count)?;
                CudaStorageSlice::F16(data)
            }
            DType::F32 => {
                let data = self.alloc_zeros::<f32>(elem_count)?;
                CudaStorageSlice::F32(data)
            }
            DType::F64 => {
                let data = self.alloc_zeros::<f64>(elem_count)?;
                CudaStorageSlice::F64(data)
            }
            DType::F8E4M3 => {
                let data = self.alloc_zeros::<F8E4M3>(elem_count)?;
                CudaStorageSlice::F8E4M3(data)
            }
            DType::F6E2M3 | DType::F6E3M2 | DType::F4 | DType::F8E8M0 => {
                return Err(
                    CudaError::InternalError("Dummy types not supported in CUDA backend").into(),
                )
            }
        };
        Ok(CudaStorage {
            slice,
            device: self.clone(),
        })
    }

    fn rand_uniform(&self, shape: &Shape, dtype: DType, lo: f64, up: f64) -> Result<CudaStorage> {
        let elem_count = shape.elem_count();
        let curand = self.curand.lock().unwrap();
        let slice = match dtype {
            // TODO: Add support for F16 and BF16 though this is likely to require some upstream
            // cudarc changes.
            DType::U8
            | DType::U32
            | DType::I16
            | DType::I32
            | DType::I64
            | DType::F16
            | DType::BF16 => Err(CudaError::UnsupportedDtype {
                dtype,
                op: "rand_uniform",
            })
            .w()?,
            DType::F32 => {
                let mut data = unsafe { self.alloc::<f32>(elem_count)? };
                curand.0.fill_with_uniform(&mut data).w()?;
                CudaStorageSlice::F32(data)
            }
            DType::F64 => {
                let mut data = unsafe { self.alloc::<f64>(elem_count)? };
                curand.0.fill_with_uniform(&mut data).w()?;
                CudaStorageSlice::F64(data)
            }
            DType::F8E4M3 | DType::F6E2M3 | DType::F6E3M2 | DType::F4 | DType::F8E8M0 => {
                Err(CudaError::UnsupportedDtype {
                    dtype,
                    op: "rand_uniform",
                })
                .w()?
            }
        };
        let slice = if lo == 0. && up == 1.0 {
            slice
        } else {
            use super::utils::Map1;
            let layout = Layout::contiguous(shape);
            super::Affine(up - lo, lo).map(&slice, self, &layout)?
        };
        Ok(CudaStorage {
            slice,
            device: self.clone(),
        })
    }

    fn rand_normal(&self, shape: &Shape, dtype: DType, mean: f64, std: f64) -> Result<CudaStorage> {
        // TODO: Add support for F16 and BF16 though this is likely to require some upstream
        // cudarc changes.
        let elem_count = shape.elem_count();
        let curand = self.curand.lock().unwrap();
        // curand can only generate an odd number of values.
        // https://github.com/huggingface/candle/issues/734
        let elem_count_round = if elem_count % 2 == 1 {
            elem_count + 1
        } else {
            elem_count
        };
        let slice = match dtype {
            DType::U8
            | DType::U32
            | DType::I16
            | DType::I32
            | DType::I64
            | DType::F16
            | DType::BF16 => Err(CudaError::UnsupportedDtype {
                dtype,
                op: "rand_normal",
            })
            .w()?,
            DType::F32 => {
                let mut data = unsafe { self.alloc::<f32>(elem_count_round)? };
                curand
                    .0
                    .fill_with_normal(&mut data, mean as f32, std as f32)
                    .w()?;
                CudaStorageSlice::F32(data)
            }
            DType::F64 => {
                let mut data = unsafe { self.alloc::<f64>(elem_count_round)? };
                curand.0.fill_with_normal(&mut data, mean, std).w()?;
                CudaStorageSlice::F64(data)
            }
            DType::F8E4M3 | DType::F6E2M3 | DType::F6E3M2 | DType::F4 | DType::F8E8M0 => {
                Err(CudaError::UnsupportedDtype {
                    dtype,
                    op: "rand_normal",
                })
                .w()?
            }
        };
        Ok(CudaStorage {
            slice,
            device: self.clone(),
        })
    }

    unsafe fn alloc_uninit(&self, shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        let elem_count = shape.elem_count();
        let slice = match dtype {
            DType::U8 => {
                let data = self.alloc::<u8>(elem_count)?;
                CudaStorageSlice::U8(data)
            }
            DType::U32 => {
                let data = self.alloc::<u32>(elem_count)?;
                CudaStorageSlice::U32(data)
            }
            DType::I16 => {
                let data = self.alloc::<i16>(elem_count)?;
                CudaStorageSlice::I16(data)
            }
            DType::I32 => {
                let data = self.alloc::<i32>(elem_count)?;
                CudaStorageSlice::I32(data)
            }
            DType::I64 => {
                let data = self.alloc::<i64>(elem_count)?;
                CudaStorageSlice::I64(data)
            }
            DType::BF16 => {
                let data = self.alloc::<bf16>(elem_count)?;
                CudaStorageSlice::BF16(data)
            }
            DType::F16 => {
                let data = self.alloc::<f16>(elem_count)?;
                CudaStorageSlice::F16(data)
            }
            DType::F32 => {
                let data = self.alloc::<f32>(elem_count)?;
                CudaStorageSlice::F32(data)
            }
            DType::F64 => {
                let data = self.alloc::<f64>(elem_count)?;
                CudaStorageSlice::F64(data)
            }
            DType::F8E4M3 => {
                let data = self.alloc::<F8E4M3>(elem_count)?;
                CudaStorageSlice::F8E4M3(data)
            }
            DType::F6E2M3 | DType::F6E3M2 | DType::F4 | DType::F8E8M0 => {
                return Err(
                    CudaError::InternalError("Dummy types not supported in CUDA backend").into(),
                )
            }
        };
        Ok(CudaStorage {
            slice,
            device: self.clone(),
        })
    }

    // SwarmLLM patch: `#[track_caller]` on this and the two below carries the
    // tensor-creating caller's line down to `count_htod_copy`.
    #[track_caller]
    fn storage_from_slice<T: crate::WithDType>(&self, s: &[T]) -> Result<Self::Storage> {
        let slice = match T::cpu_storage_ref(s) {
            CpuStorageRef::U8(storage) => {
                let data = self.clone_htod(storage)?;
                CudaStorageSlice::U8(data)
            }
            CpuStorageRef::U32(storage) => {
                let data = self.clone_htod(storage)?;
                CudaStorageSlice::U32(data)
            }
            CpuStorageRef::I16(storage) => {
                let data = self.clone_htod(storage)?;
                CudaStorageSlice::I16(data)
            }
            CpuStorageRef::I32(storage) => {
                let data = self.clone_htod(storage)?;
                CudaStorageSlice::I32(data)
            }
            CpuStorageRef::I64(storage) => {
                let data = self.clone_htod(storage)?;
                CudaStorageSlice::I64(data)
            }
            CpuStorageRef::BF16(storage) => {
                let data = self.clone_htod(storage)?;
                CudaStorageSlice::BF16(data)
            }
            CpuStorageRef::F16(storage) => {
                let data = self.clone_htod(storage)?;
                CudaStorageSlice::F16(data)
            }
            CpuStorageRef::F32(storage) => {
                let data = self.clone_htod(storage)?;
                CudaStorageSlice::F32(data)
            }
            CpuStorageRef::F64(storage) => {
                let data = self.clone_htod(storage)?;
                CudaStorageSlice::F64(data)
            }
            CpuStorageRef::F8E4M3(storage) => {
                let data = self.clone_htod(storage)?;
                CudaStorageSlice::F8E4M3(data)
            }
            CpuStorageRef::F4(_)
            | CpuStorageRef::F6E2M3(_)
            | CpuStorageRef::F6E3M2(_)
            | CpuStorageRef::F8E8M0(_) => {
                return Err(CudaError::UnsupportedDtype {
                    dtype: T::DTYPE,
                    op: "storage_from_slice",
                }
                .into());
            }
        };
        Ok(CudaStorage {
            slice,
            device: self.clone(),
        })
    }

    #[track_caller]
    fn storage_from_cpu_storage(&self, storage: &CpuStorage) -> Result<CudaStorage> {
        let slice = match storage {
            CpuStorage::U8(storage) => {
                let data = self.clone_htod(storage)?;
                CudaStorageSlice::U8(data)
            }
            CpuStorage::U32(storage) => {
                let data = self.clone_htod(storage)?;
                CudaStorageSlice::U32(data)
            }
            CpuStorage::I16(storage) => {
                let data = self.clone_htod(storage)?;
                CudaStorageSlice::I16(data)
            }
            CpuStorage::I32(storage) => {
                let data = self.clone_htod(storage)?;
                CudaStorageSlice::I32(data)
            }
            CpuStorage::I64(storage) => {
                let data = self.clone_htod(storage)?;
                CudaStorageSlice::I64(data)
            }
            CpuStorage::BF16(storage) => {
                let data = self.clone_htod(storage)?;
                CudaStorageSlice::BF16(data)
            }
            CpuStorage::F16(storage) => {
                let data = self.clone_htod(storage)?;
                CudaStorageSlice::F16(data)
            }
            CpuStorage::F32(storage) => {
                let data = self.clone_htod(storage)?;
                CudaStorageSlice::F32(data)
            }
            CpuStorage::F64(storage) => {
                let data = self.clone_htod(storage)?;
                CudaStorageSlice::F64(data)
            }
            CpuStorage::F8E4M3(storage) => {
                let data = self.clone_htod(storage)?;
                CudaStorageSlice::F8E4M3(data)
            }
            CpuStorage::F4(_)
            | CpuStorage::F6E2M3(_)
            | CpuStorage::F6E3M2(_)
            | CpuStorage::F8E8M0(_) => {
                return Err(CudaError::UnsupportedDtype {
                    dtype: storage.dtype(),
                    op: "storage_from_cpu_storage",
                }
                .into());
            }
        };
        Ok(CudaStorage {
            slice,
            device: self.clone(),
        })
    }

    #[track_caller]
    fn storage_from_cpu_storage_owned(&self, storage: CpuStorage) -> Result<CudaStorage> {
        let slice = match storage {
            CpuStorage::U8(storage) => {
                let data = self.clone_htod(&storage)?;
                CudaStorageSlice::U8(data)
            }
            CpuStorage::U32(storage) => {
                let data = self.clone_htod(&storage)?;
                CudaStorageSlice::U32(data)
            }
            CpuStorage::I16(storage) => {
                let data = self.clone_htod(&storage)?;
                CudaStorageSlice::I16(data)
            }
            CpuStorage::I32(storage) => {
                let data = self.clone_htod(&storage)?;
                CudaStorageSlice::I32(data)
            }
            CpuStorage::I64(storage) => {
                let data = self.clone_htod(&storage)?;
                CudaStorageSlice::I64(data)
            }
            CpuStorage::BF16(storage) => {
                let data = self.clone_htod(&storage)?;
                CudaStorageSlice::BF16(data)
            }
            CpuStorage::F16(storage) => {
                let data = self.clone_htod(&storage)?;
                CudaStorageSlice::F16(data)
            }
            CpuStorage::F32(storage) => {
                let data = self.clone_htod(&storage)?;
                CudaStorageSlice::F32(data)
            }
            CpuStorage::F64(storage) => {
                let data = self.clone_htod(&storage)?;
                CudaStorageSlice::F64(data)
            }
            CpuStorage::F8E4M3(storage) => {
                let data = self.clone_htod(&storage)?;
                CudaStorageSlice::F8E4M3(data)
            }
            CpuStorage::F4(_)
            | CpuStorage::F6E2M3(_)
            | CpuStorage::F6E3M2(_)
            | CpuStorage::F8E8M0(_) => {
                return Err(CudaError::UnsupportedDtype {
                    dtype: storage.dtype(),
                    op: "storage_from_cpu_storage_owned",
                }
                .into());
            }
        };
        Ok(CudaStorage {
            slice,
            device: self.clone(),
        })
    }

    fn synchronize(&self) -> Result<()> {
        self.stream.synchronize().map_err(crate::Error::wrap)?;
        Ok(())
    }
}
