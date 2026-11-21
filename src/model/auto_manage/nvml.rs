//! The graphics card read through ONE NVML session in this process
//! (2026-11-21, FUTURE_WORK #490).
//!
//! Every `nvidia-smi` the node started under WSL2 left ~15 KB of Windows'
//! non-paged kernel memory behind for good — 5,000 launches grew the pool by
//! 75 MB, the same readings through one session by 0.2 MB — and the node
//! started ~9,500 a day: free card memory for the budget and the capability,
//! the heat watch every 30 s. So the library `nvidia-smi` itself reads through
//! (`libnvidia-ml.so.1`, `nvml.dll`) is loaded once, initialised once and kept,
//! on ONE thread that owns it.
//!
//! Prior art: Ollama loads that library at runtime and looks up only the
//! functions it calls (`discover/native_probe_linux.go`; on Windows System32,
//! then `NVIDIA Corporation\NVSMI`), PyTorch opens it with a raw `CDLL`; both
//! initialise per probe or per call. One kept session rests on our own
//! measurement. The `nvml-wrapper` crate was weighed and left: its memory
//! reading calls only `_v2`, with no fallback (FUTURE_WORK #490).
//!
//! It keeps what `vram::BoundedCommand` gave `nvidia-smi`:
//! - **A reading that does not answer within [`READING_TIMEOUT`] is unknown**
//!   (`None`), and while a call has outlived that bound every reading is
//!   unknown at once. A driver resetting the card holds a call for as long as
//!   the reset takes (14 minutes on 2026-10-04, #220); NVML has no timeout and
//!   a call stuck in the driver cannot be cancelled, so the session is never
//!   shut down under one, and `still_not_answering` (CUDA builds) counts from its start.
//! - **Its wait never holds a runtime worker** (#244):
//!   [`crate::inference::executor::without_starving_the_runtime`].
//! - **A missing library is no card**: every reading `None`, as a missing
//!   `nvidia-smi` was. A failed open is tried again after [`REOPEN_AFTER`].
//!
//! A field is read only where its call answered `NVML_SUCCESS`: a call that
//! answers NOT_SUPPORTED leaves its buffer as the last call wrote it
//! (`ACOUSTIC_CURR` under WSL2, 2026-11-21). The daemon still holds no CUDA
//! context: NVML opens none.

use std::ffi::{c_char, c_int, c_uint, c_ulonglong, c_void};
use std::sync::mpsc;
use std::sync::{Arc, Mutex, OnceLock, PoisonError};
use std::time::{Duration, Instant};

/// How long a caller waits for one reading of the card.
///
/// `nvidia-smi` answered in ~90 ms here; an NVML call in well under one. It
/// does not answer at all while the graphics driver is resetting the card (a
/// TDR), and on 2026-10-04 one reset took 14 minutes: a node's re-plan sat in
/// GPU admission (`compute_vram_budget`) for all of it, and every other caller
/// — the capability broadcast, the dashboard — would have queued behind the
/// driver the same way (FUTURE_WORK #220). A bound turns "the driver is busy"
/// into the `None` every caller already treats as unknown.
pub(crate) const READING_TIMEOUT: Duration = Duration::from_secs(10);

/// How long after a failed open the library is tried again: a driver
/// installed or restarted while the node runs is found, without a load
/// attempt at every reading on a computer with no card.
const REOPEN_AFTER: Duration = Duration::from_secs(30);

// From nvml.h.
const NVML_SUCCESS: c_int = 0;
const NVML_ERROR_UNINITIALIZED: c_int = 1;
const NVML_ERROR_DRIVER_NOT_LOADED: c_int = 9;
const NVML_ERROR_FUNCTION_NOT_FOUND: c_int = 13;
const NVML_ERROR_GPU_IS_LOST: c_int = 15;
const NVML_TEMPERATURE_GPU: c_uint = 0;
/// `NVML_TEMPERATURE_THRESHOLD_ACOUSTIC_CURR` (5) and `…_GPS_CURR` (7): the
/// temperature the driver holds the card at. `nvidia-smi -q`'s "GPU Target
/// Temperature Specification" is GPS_CURR on the RTX 3070 Laptop (87 °C both,
/// 2026-11-21); ACOUSTIC_CURR answers NOT_SUPPORTED there under WSL2.
/// `nvidia-smi`'s reading took the lowest "Target Temperature" it printed.
const TARGET_THRESHOLDS: [c_uint; 2] = [5, 7];
/// `nvmlClocksEventReasonSwThermalSlowdown` / `…HwThermalSlowdown`.
const SW_THERMAL_SLOWDOWN: c_ulonglong = 0x20;
const HW_THERMAL_SLOWDOWN: c_ulonglong = 0x40;
/// `NVML_DEVICE_NAME_V2_BUFFER_SIZE`.
const NAME_BUFFER: usize = 96;

/// `nvmlMemory_t`: `used` counts the driver's reserve too. A field only the
/// library writes or reads, never this module, begins with `_`.
#[repr(C)]
#[derive(Default)]
struct MemoryV1 {
    total: c_ulonglong,
    _free: c_ulonglong,
    used: c_ulonglong,
}

/// `nvmlMemory_v2_t`: `used` is what `nvidia-smi`'s `memory.used` prints (877
/// MB against v1's 1,050 on the RTX 3070 Laptop, 2026-11-21 — v1 counts the
/// driver's 174 MB reserve).
#[repr(C)]
#[derive(Default)]
struct MemoryV2 {
    _version: c_uint,
    total: c_ulonglong,
    _reserved: c_ulonglong,
    _free: c_ulonglong,
    used: c_ulonglong,
}

/// `nvmlMemory_v2`: the struct's size, with version 2 in the top byte.
const MEMORY_V2_VERSION: c_uint = std::mem::size_of::<MemoryV2>() as c_uint | (2 << 24);

type DeviceHandle = *mut c_void;
type PlainFn = unsafe extern "C" fn() -> c_int;
type CountFn = unsafe extern "C" fn(*mut c_uint) -> c_int;
type HandleFn = unsafe extern "C" fn(c_uint, *mut DeviceHandle) -> c_int;
type NameFn = unsafe extern "C" fn(DeviceHandle, *mut c_char, c_uint) -> c_int;
type MemoryV1Fn = unsafe extern "C" fn(DeviceHandle, *mut MemoryV1) -> c_int;
type MemoryV2Fn = unsafe extern "C" fn(DeviceHandle, *mut MemoryV2) -> c_int;
#[cfg(feature = "candle-cuda")]
type ComputeCapFn = unsafe extern "C" fn(DeviceHandle, *mut c_int, *mut c_int) -> c_int;
/// `nvmlDeviceGetTemperature` and `nvmlDeviceGetTemperatureThreshold` alike.
type TemperatureFn = unsafe extern "C" fn(DeviceHandle, c_uint, *mut c_uint) -> c_int;
type ReasonsFn = unsafe extern "C" fn(DeviceHandle, *mut c_ulonglong) -> c_int;

/// A card's memory, in MB as `nvidia-smi` printed it: used rounded UP
/// (839,172,096 B, 800.3 MiB, printed 801 on 2026-11-21), so free = total −
/// used is never more than it was.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CardMemory {
    pub(crate) total_mb: u64,
    pub(crate) used_mb: u64,
}

/// One card's temperature and the driver's cooling flags.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct CardTemperature {
    pub(crate) temp_c: f32,
    /// "SW Thermal Slowdown": the driver cuts clocks to hold its target.
    pub(crate) driver_cooling: bool,
    /// "HW Thermal Slowdown": the hardware's own protection.
    pub(crate) hardware_protecting: bool,
}

/// The first card's name, and its memory where the driver says.
pub(crate) fn first_card_name_and_memory() -> Option<(String, Option<CardMemory>)> {
    ask(|nvml| {
        let card = nvml.device(0)?;
        let name = nvml.name(card)?;
        Ok((name, nvml.memory(card).ok()))
    })
}

/// The first card's memory.
pub(crate) fn first_card_memory() -> Option<CardMemory> {
    ask(|nvml| nvml.memory(nvml.device(0)?))
}

/// The first card's CUDA compute capability, `(major, minor)`.
#[cfg(feature = "candle-cuda")]
pub(crate) fn first_card_compute_capability() -> Option<(u32, u32)> {
    ask(|nvml| nvml.compute_capability(nvml.device(0)?))
}

/// Every card's temperature and cooling flags; a card whose temperature does
/// not answer is left out, and one whose flags do not reads them as off.
pub(crate) fn card_temperatures() -> Option<Vec<CardTemperature>> {
    ask(|nvml| {
        let mut cards = Vec::new();
        for index in 0..nvml.count()? {
            let Some(card) = skip_unless_lost(nvml.device(index))? else {
                continue;
            };
            let Some(temp) = skip_unless_lost(nvml.temperature(card))? else {
                continue;
            };
            cards.extend(card_temperature(temp, nvml.reasons(card)));
        }
        Ok(cards)
    })
}

/// The temperature the driver holds the cards at — the lowest any card names,
/// `Some(None)` where none does — or `None` while the driver does not answer.
pub(crate) fn target_temperature_c() -> Option<Option<f32>> {
    ask(|nvml| {
        let mut answers = Vec::new();
        for index in 0..nvml.count()? {
            let Some(card) = skip_unless_lost(nvml.device(index))? else {
                continue;
            };
            answers.extend(TARGET_THRESHOLDS.map(|which| nvml.threshold(card, which)));
        }
        Ok(lowest_target(answers))
    })
}

/// Has a reading of the card outlived [`READING_TIMEOUT`] and been waiting
/// for at least `for_at_least` — has the graphics driver not answered for
/// that long? Never waits, and never starts the session to ask.
#[cfg(feature = "candle-cuda")]
pub(crate) fn still_not_answering(for_at_least: Duration) -> bool {
    READER
        .get()
        .and_then(Option::as_ref)
        .is_some_and(|reader| reader.still_not_answering(for_at_least))
}

static READER: OnceLock<Option<Reader<Session>>> = OnceLock::new();

/// One reading through the session, on its thread.
fn ask<T: Send + 'static>(
    read: impl FnOnce(&Nvml) -> Result<T, c_int> + Send + 'static,
) -> Option<T> {
    READER
        .get_or_init(|| {
            if library_paths().is_empty() {
                return None;
            }
            Reader::start("card-reader", READING_TIMEOUT, Session::default)
        })
        .as_ref()?
        .ask(move |session: &mut Session| session.read(read))
}

/// `Ok(None)` for an answer that skips one card, the error for one that says
/// the session must open again.
fn skip_unless_lost<T>(answer: Result<T, c_int>) -> Result<Option<T>, c_int> {
    match answer {
        Ok(value) => Ok(Some(value)),
        Err(code) if reopens(code) => Err(code),
        Err(_) => Ok(None),
    }
}

/// The card or the driver went away under the session: open it again.
fn reopens(code: c_int) -> bool {
    matches!(
        code,
        NVML_ERROR_UNINITIALIZED | NVML_ERROR_DRIVER_NOT_LOADED | NVML_ERROR_GPU_IS_LOST
    )
}

/// The lowest target any card answered, within 40-120 °C as
/// `nvidia-smi`'s was read.
fn lowest_target(answers: impl IntoIterator<Item = Result<c_uint, c_int>>) -> Option<f32> {
    answers
        .into_iter()
        .filter_map(Result::ok)
        .map(|t| t as f32)
        .filter(|t| (40.0..120.0).contains(t))
        .reduce(f32::min)
}

/// A card's reading, unless its temperature is out of any card's range.
fn card_temperature(
    temp_c: c_uint,
    reasons: Result<c_ulonglong, c_int>,
) -> Option<CardTemperature> {
    let temp_c = temp_c as f32;
    let reasons = reasons.unwrap_or(0);
    (0.0..150.0).contains(&temp_c).then_some(CardTemperature {
        temp_c,
        driver_cooling: reasons & SW_THERMAL_SLOWDOWN != 0,
        hardware_protecting: reasons & HW_THERMAL_SLOWDOWN != 0,
    })
}

/// Where the library may be, in the order tried.
fn library_paths() -> Vec<std::path::PathBuf> {
    #[cfg(windows)]
    {
        // Full paths only, so no folder searched before System32 can stand in
        // for the driver's own library.
        let mut paths = Vec::new();
        let root = std::env::var_os("SystemRoot").unwrap_or_else(|| "C:\\Windows".into());
        paths.push(
            std::path::Path::new(&root)
                .join("System32")
                .join("nvml.dll"),
        );
        for programs in ["ProgramW6432", "ProgramFiles"] {
            if let Some(dir) = std::env::var_os(programs) {
                paths.push(
                    std::path::Path::new(&dir)
                        .join("NVIDIA Corporation")
                        .join("NVSMI")
                        .join("nvml.dll"),
                );
            }
        }
        paths
    }
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        // The loader's own search first (`ld.so.conf` names WSL's folder too),
        // then WSL's folder directly.
        vec![
            "libnvidia-ml.so.1".into(),
            "/usr/lib/wsl/lib/libnvidia-ml.so.1".into(),
        ]
    }
    #[cfg(not(any(windows, all(unix, not(target_os = "macos")))))]
    {
        Vec::new()
    }
}

/// NVIDIA's management library, loaded and initialised: the functions this
/// module calls, copied out of `library`, which holds them for as long as it
/// lives — `None` only for a test's stand-in functions.
struct Nvml {
    shutdown: PlainFn,
    count: CountFn,
    handle: HandleFn,
    name: Option<NameFn>,
    memory_v2: Option<MemoryV2Fn>,
    memory_v1: Option<MemoryV1Fn>,
    #[cfg(feature = "candle-cuda")]
    compute_cap: Option<ComputeCapFn>,
    temperature: Option<TemperatureFn>,
    threshold: Option<TemperatureFn>,
    reasons: Option<ReasonsFn>,
    library: Option<libloading::Library>,
}

impl Nvml {
    fn open() -> Option<Self> {
        Self::open_from(library_paths())
    }

    fn open_from(paths: Vec<std::path::PathBuf>) -> Option<Self> {
        paths.into_iter().find_map(|path| {
            // SAFETY: NVIDIA's management library — the one `nvidia-smi` reads
            // through — loaded the way Ollama and PyTorch load it.
            let library = unsafe { libloading::Library::new(&path) }.ok()?;
            let nvml = Self::initialise(library)?;
            tracing::debug!(
                library = %path.display(),
                "DIAG: the graphics card is read through NVML"
            );
            Some(nvml)
        })
    }

    /// The library's functions, and `nvmlInit_v2` called once; `None` where
    /// it lacks one this module cannot do without, or the driver refuses.
    fn initialise(library: libloading::Library) -> Option<Self> {
        fn function<T: Copy>(library: &libloading::Library, name: &[u8]) -> Option<T> {
            // SAFETY: every `T` asked for below is the C signature nvml.h
            // gives `name`.
            unsafe { library.get::<T>(name) }.ok().map(|f| *f)
        }
        let init: PlainFn = function(&library, b"nvmlInit_v2\0")?;
        let shutdown: PlainFn = function(&library, b"nvmlShutdown\0")?;
        let count: CountFn = function(&library, b"nvmlDeviceGetCount_v2\0")?;
        let handle: HandleFn = function(&library, b"nvmlDeviceGetHandleByIndex_v2\0")?;
        let name = function(&library, b"nvmlDeviceGetName\0");
        let memory_v2 = function(&library, b"nvmlDeviceGetMemoryInfo_v2\0");
        let memory_v1 = function(&library, b"nvmlDeviceGetMemoryInfo\0");
        #[cfg(feature = "candle-cuda")]
        let compute_cap = function(&library, b"nvmlDeviceGetCudaComputeCapability\0");
        let temperature = function(&library, b"nvmlDeviceGetTemperature\0");
        let threshold = function(&library, b"nvmlDeviceGetTemperatureThreshold\0");
        // `…ThrottleReasons` is the older name; current drivers take both.
        let reasons = function(&library, b"nvmlDeviceGetCurrentClocksEventReasons\0")
            .or_else(|| function(&library, b"nvmlDeviceGetCurrentClocksThrottleReasons\0"));
        // SAFETY: takes nothing; NVML counts its initialisations, and `Drop`
        // ends this one.
        if unsafe { init() } != NVML_SUCCESS {
            return None;
        }
        Some(Self {
            shutdown,
            count,
            handle,
            name,
            memory_v2,
            memory_v1,
            #[cfg(feature = "candle-cuda")]
            compute_cap,
            temperature,
            threshold,
            reasons,
            library: Some(library),
        })
    }

    fn count(&self) -> Result<c_uint, c_int> {
        let mut count: c_uint = 0;
        // SAFETY: the call writes `count`.
        answered(unsafe { (self.count)(&mut count) }).map(|()| count)
    }

    fn device(&self, index: c_uint) -> Result<DeviceHandle, c_int> {
        let mut card: DeviceHandle = std::ptr::null_mut();
        // SAFETY: the call writes `card`.
        answered(unsafe { (self.handle)(index, &mut card) }).map(|()| card)
    }

    fn name(&self, card: DeviceHandle) -> Result<String, c_int> {
        let name = self.name.ok_or(NVML_ERROR_FUNCTION_NOT_FOUND)?;
        let mut buffer: [c_char; NAME_BUFFER] = [0; NAME_BUFFER];
        // SAFETY: the call writes at most `NAME_BUFFER` bytes, NUL included.
        answered(unsafe { name(card, buffer.as_mut_ptr(), NAME_BUFFER as c_uint) })?;
        let bytes: Vec<u8> = buffer.iter().map(|c| c.to_ne_bytes()[0]).collect();
        let name = std::ffi::CStr::from_bytes_until_nul(&bytes)
            .map_err(|_| NVML_ERROR_FUNCTION_NOT_FOUND)?
            .to_string_lossy()
            .trim()
            .to_string();
        Ok(name)
    }

    /// `_v2`'s figures, which `nvidia-smi` prints; v1's where this driver has
    /// no `_v2` or it refuses.
    fn memory(&self, card: DeviceHandle) -> Result<CardMemory, c_int> {
        const MB: c_ulonglong = 1 << 20;
        if let Some(memory_v2) = self.memory_v2 {
            let mut memory = MemoryV2 {
                _version: MEMORY_V2_VERSION,
                ..MemoryV2::default()
            };
            // SAFETY: the call writes `memory`, whose version it checks.
            match answered(unsafe { memory_v2(card, &mut memory) }) {
                Ok(()) => {
                    return Ok(CardMemory {
                        total_mb: memory.total / MB,
                        used_mb: memory.used.div_ceil(MB),
                    })
                }
                Err(code) if reopens(code) => return Err(code),
                Err(_) => {}
            }
        }
        let memory_v1 = self.memory_v1.ok_or(NVML_ERROR_FUNCTION_NOT_FOUND)?;
        let mut memory = MemoryV1::default();
        // SAFETY: the call writes `memory`.
        answered(unsafe { memory_v1(card, &mut memory) })?;
        Ok(CardMemory {
            total_mb: memory.total / MB,
            used_mb: memory.used.div_ceil(MB),
        })
    }

    #[cfg(feature = "candle-cuda")]
    fn compute_capability(&self, card: DeviceHandle) -> Result<(u32, u32), c_int> {
        let compute_cap = self.compute_cap.ok_or(NVML_ERROR_FUNCTION_NOT_FOUND)?;
        let (mut major, mut minor): (c_int, c_int) = (0, 0);
        // SAFETY: the call writes both.
        answered(unsafe { compute_cap(card, &mut major, &mut minor) })?;
        Ok((
            u32::try_from(major).map_err(|_| NVML_ERROR_FUNCTION_NOT_FOUND)?,
            u32::try_from(minor).map_err(|_| NVML_ERROR_FUNCTION_NOT_FOUND)?,
        ))
    }

    fn temperature(&self, card: DeviceHandle) -> Result<c_uint, c_int> {
        let temperature = self.temperature.ok_or(NVML_ERROR_FUNCTION_NOT_FOUND)?;
        let mut temp: c_uint = 0;
        // SAFETY: the call writes `temp`.
        answered(unsafe { temperature(card, NVML_TEMPERATURE_GPU, &mut temp) }).map(|()| temp)
    }

    fn threshold(&self, card: DeviceHandle, which: c_uint) -> Result<c_uint, c_int> {
        let threshold = self.threshold.ok_or(NVML_ERROR_FUNCTION_NOT_FOUND)?;
        let mut temp: c_uint = 0;
        // SAFETY: the call writes `temp`.
        answered(unsafe { threshold(card, which, &mut temp) }).map(|()| temp)
    }

    fn reasons(&self, card: DeviceHandle) -> Result<c_ulonglong, c_int> {
        let reasons = self.reasons.ok_or(NVML_ERROR_FUNCTION_NOT_FOUND)?;
        let mut bits: c_ulonglong = 0;
        // SAFETY: the call writes `bits`.
        answered(unsafe { reasons(card, &mut bits) }).map(|()| bits)
    }
}

impl Drop for Nvml {
    fn drop(&mut self) {
        // SAFETY: ends the one initialisation `initialise` made; the session
        // is dropped only between calls, never under one still in the driver.
        unsafe { (self.shutdown)() };
        drop(self.library.take());
    }
}

/// `Ok` where NVML answered `NVML_SUCCESS`: only then is what it wrote read.
fn answered(code: c_int) -> Result<(), c_int> {
    if code == NVML_SUCCESS {
        Ok(())
    } else {
        Err(code)
    }
}

/// The library on the card-reader thread: opened at the first reading, and
/// again after a failure once [`REOPEN_AFTER`] has passed since the last try.
#[derive(Default)]
struct Session {
    nvml: Option<Nvml>,
    tried_at: Option<Instant>,
}

impl Session {
    fn read<T>(&mut self, read: impl FnOnce(&Nvml) -> Result<T, c_int>) -> Option<T> {
        if self.nvml.is_none() && self.tried_at.is_none_or(|at| at.elapsed() >= REOPEN_AFTER) {
            self.tried_at = Some(Instant::now());
            self.nvml = Nvml::open();
        }
        let answer = read(self.nvml.as_ref()?);
        match answer {
            Ok(value) => Some(value),
            Err(code) => {
                if reopens(code) {
                    self.nvml = None;
                }
                None
            }
        }
    }
}

type Job<C> = Box<dyn FnOnce(&mut C) + Send>;

/// One thread that owns a context (`C`: the NVML session) and runs readings
/// on it one at a time, with `vram::BoundedCommand`'s rules for the wait.
struct Reader<C> {
    jobs: mpsc::Sender<Job<C>>,
    running: Arc<Mutex<Running>>,
    timeout: Duration,
}

/// The reading the thread is running, if any.
#[derive(Default)]
struct Running {
    since: Option<Instant>,
    /// A caller gave up on it: every reading is unknown at once until it ends.
    outlived_its_bound: bool,
}

impl<C: 'static> Reader<C> {
    /// `open` makes the context on the thread itself, so it never crosses one.
    fn start(
        name: &str,
        timeout: Duration,
        open: impl FnOnce() -> C + Send + 'static,
    ) -> Option<Self> {
        let (jobs, received) = mpsc::channel::<Job<C>>();
        let running: Arc<Mutex<Running>> = Arc::default();
        let on_thread = running.clone();
        std::thread::Builder::new()
            .name(name.to_string())
            .spawn(move || {
                let mut context = open();
                while let Ok(job) = received.recv() {
                    let started = Instant::now();
                    lock(&on_thread).since = Some(started);
                    job(&mut context);
                    let mut running = lock(&on_thread);
                    if std::mem::take(&mut running.outlived_its_bound) {
                        tracing::info!(
                            stalled_secs = started.elapsed().as_secs(),
                            "the graphics card answers again after a stall"
                        );
                    }
                    running.since = None;
                }
            })
            .ok()?;
        Some(Self {
            jobs,
            running,
            timeout,
        })
    }

    fn ask<T: Send + 'static>(
        &self,
        read: impl FnOnce(&mut C) -> Option<T> + Send + 'static,
    ) -> Option<T> {
        if lock(&self.running).outlived_its_bound {
            return None;
        }
        let (answer, answered) = mpsc::sync_channel(1);
        self.jobs
            .send(Box::new(move |context: &mut C| {
                let _ = answer.send(read(context));
            }))
            .ok()?;
        let waited = crate::inference::executor::without_starving_the_runtime(|| {
            answered.recv_timeout(self.timeout)
        });
        match waited {
            Ok(value) => value,
            Err(mpsc::RecvTimeoutError::Timeout) => {
                let mut running = lock(&self.running);
                if running.since.is_some() && !running.outlived_its_bound {
                    running.outlived_its_bound = true;
                    tracing::warn!(
                        timeout_secs = self.timeout.as_secs(),
                        "a reading of the graphics card did not answer — the driver may be \
                         resetting the card; its readings count as unknown until it answers again"
                    );
                }
                None
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => None,
        }
    }

    #[cfg(any(test, feature = "candle-cuda"))]
    fn still_not_answering(&self, for_at_least: Duration) -> bool {
        let running = lock(&self.running);
        running.outlived_its_bound && running.since.is_some_and(|at| at.elapsed() >= for_at_least)
    }
}

fn lock(running: &Mutex<Running>) -> std::sync::MutexGuard<'_, Running> {
    running.lock().unwrap_or_else(PoisonError::into_inner)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ms(n: u64) -> Duration {
        Duration::from_millis(n)
    }

    /// A reading that outlives its bound is unknown; while it is still stuck
    /// in the driver every reading is unknown AT ONCE (no caller queues behind
    /// it), the driver counts as not answering from its start, and the first
    /// reading after it ends answers again — `BoundedCommand`'s rules (#220).
    #[test]
    fn a_reading_that_outlives_its_bound_makes_the_next_unknown_at_once() {
        let reader = Reader::start("test-reader", ms(50), || ()).unwrap();
        assert_eq!(reader.ask(|_| Some(7)), Some(7));
        assert!(!reader.still_not_answering(Duration::ZERO));

        let asked = Instant::now();
        assert_eq!(
            reader.ask(|_| {
                std::thread::sleep(ms(400));
                Some(1)
            }),
            None
        );
        assert!(
            asked.elapsed() < ms(300),
            "the caller waited its bound only"
        );
        let asked = Instant::now();
        assert_eq!(reader.ask(|_| Some(2)), None, "stuck: unknown");
        assert!(asked.elapsed() < ms(20), "at once, not after another bound");
        assert!(reader.still_not_answering(Duration::ZERO));
        // One slow answer is not a reset (gotcha #802): it counts only once
        // stuck for the minute placement waits (`DRIVER_NOT_ANSWERING_AFTER`).
        let minute = Duration::from_secs(60);
        assert!(!reader.still_not_answering(minute));
        let since = lock(&reader.running).since;
        lock(&reader.running).since =
            since.and_then(|at| at.checked_sub(minute + Duration::from_secs(1)));
        assert!(reader.still_not_answering(minute));
        lock(&reader.running).since = since;

        std::thread::sleep(ms(450));
        assert!(!reader.still_not_answering(Duration::ZERO), "it answered");
        assert_eq!(reader.ask(|_| Some(3)), Some(3));
    }

    /// #244: a slow reading must not park the task queued behind it on the
    /// same runtime worker. One worker, so the ticker keeps ticking only if
    /// the wait hands the worker over.
    #[test]
    fn a_slow_card_reading_does_not_hold_the_runtime_worker() {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(1)
            .enable_time()
            .build()
            .unwrap();
        let ticks: Arc<Mutex<Vec<Instant>>> = Arc::default();
        let reader = Arc::new(Reader::start("test-reader", Duration::from_secs(5), || ()).unwrap());
        let largest_gap = rt.block_on(async {
            let recorded = ticks.clone();
            let ticker = tokio::spawn(async move {
                loop {
                    recorded.lock().unwrap().push(Instant::now());
                    tokio::time::sleep(ms(10)).await;
                }
            });
            tokio::time::sleep(ms(50)).await;
            let asked = Instant::now();
            let answer = tokio::spawn(async move {
                reader.ask(|_| {
                    std::thread::sleep(ms(500));
                    Some(())
                })
            })
            .await
            .unwrap();
            let done = Instant::now();
            ticker.abort();
            assert_eq!(answer, Some(()));
            let during: Vec<Instant> = ticks
                .lock()
                .unwrap()
                .iter()
                .copied()
                .filter(|t| *t >= asked && *t <= done)
                .collect();
            std::iter::once(asked)
                .chain(during)
                .chain(std::iter::once(done))
                .collect::<Vec<_>>()
                .windows(2)
                .map(|w| w[1] - w[0])
                .max()
                .unwrap()
        });
        assert!(
            largest_gap < ms(250),
            "the ticker waited {largest_gap:?} behind a 0.5 s reading"
        );
    }

    #[test]
    fn no_library_reads_as_no_card() {
        assert!(Nvml::open_from(vec!["/nonexistent/libnvidia-ml.so.1".into()]).is_none());
        let mut session = Session {
            tried_at: Some(Instant::now()),
            ..Session::default()
        };
        assert_eq!(
            session.read(|_| Ok(1)),
            None,
            "not opened again before its time"
        );
    }

    // A stand-in driver: one card, whose acoustic target answers NOT_SUPPORTED
    // after writing into the buffer anyway — WSL2's answer on 2026-11-21.
    unsafe extern "C" fn ok() -> c_int {
        NVML_SUCCESS
    }
    unsafe extern "C" fn one_card(count: *mut c_uint) -> c_int {
        unsafe { *count = 1 };
        NVML_SUCCESS
    }
    unsafe extern "C" fn card(_: c_uint, card: *mut DeviceHandle) -> c_int {
        unsafe { *card = std::ptr::dangling_mut::<c_void>() };
        NVML_SUCCESS
    }
    unsafe extern "C" fn thresholds(_: DeviceHandle, which: c_uint, temp: *mut c_uint) -> c_int {
        match which {
            5 => {
                unsafe { *temp = 45 };
                3
            }
            7 => {
                unsafe { *temp = 87 };
                NVML_SUCCESS
            }
            _ => 3,
        }
    }
    unsafe extern "C" fn temperature(_: DeviceHandle, _: c_uint, temp: *mut c_uint) -> c_int {
        unsafe { *temp = 63 };
        NVML_SUCCESS
    }
    unsafe extern "C" fn cooling(_: DeviceHandle, bits: *mut c_ulonglong) -> c_int {
        unsafe { *bits = SW_THERMAL_SLOWDOWN | 0x1 };
        NVML_SUCCESS
    }
    unsafe extern "C" fn memory_v2_refused(_: DeviceHandle, _: *mut MemoryV2) -> c_int {
        NVML_ERROR_FUNCTION_NOT_FOUND
    }
    unsafe extern "C" fn memory_v1(_: DeviceHandle, memory: *mut MemoryV1) -> c_int {
        unsafe {
            *memory = MemoryV1 {
                total: 8192 << 20,
                _free: 7141 << 20,
                // This card's reading at 2026-11-21, `nvidia-smi`'s 801.
                used: 839_172_096,
            }
        };
        NVML_SUCCESS
    }

    fn stand_in() -> Nvml {
        Nvml {
            shutdown: ok,
            count: one_card,
            handle: card,
            name: None,
            memory_v2: Some(memory_v2_refused),
            memory_v1: Some(memory_v1),
            #[cfg(feature = "candle-cuda")]
            compute_cap: None,
            temperature: Some(temperature),
            threshold: Some(thresholds),
            reasons: Some(cooling),
            library: None,
        }
    }

    /// A field is read only where its call answered: the acoustic target's
    /// NOT_SUPPORTED left 45 in its buffer, and the target is the 87 the
    /// driver did answer — not the lower stale figure.
    #[test]
    fn a_field_is_read_only_where_its_call_answered() {
        let nvml = stand_in();
        let card = nvml.device(0).unwrap();
        let answers = TARGET_THRESHOLDS.map(|which| nvml.threshold(card, which));
        assert_eq!(answers, [Err(3), Ok(87)]);
        assert_eq!(lowest_target(answers), Some(87.0));
        assert_eq!(
            card_temperature(nvml.temperature(card).unwrap(), nvml.reasons(card)),
            Some(CardTemperature {
                temp_c: 63.0,
                driver_cooling: true,
                hardware_protecting: false,
            })
        );
        assert_eq!(nvml.name(card), Err(NVML_ERROR_FUNCTION_NOT_FOUND));
        assert_eq!(
            nvml.memory(card),
            Ok(CardMemory {
                total_mb: 8192,
                used_mb: 801,
            }),
            "v1 where `_v2` is refused, used rounded up as `nvidia-smi` prints it"
        );
    }

    #[test]
    fn a_target_is_the_lowest_a_card_names_within_any_cards_range() {
        assert_eq!(lowest_target([Ok(87), Ok(83)]), Some(83.0));
        assert_eq!(lowest_target([Ok(0), Ok(255), Err(3)]), None);
        assert_eq!(lowest_target([]), None);
        assert_eq!(card_temperature(200, Ok(0)), None);
        assert_eq!(
            card_temperature(70, Err(3)),
            Some(CardTemperature {
                temp_c: 70.0,
                driver_cooling: false,
                hardware_protecting: false,
            }),
            "flags that do not answer read as off, as `nvidia-smi`'s N/A did"
        );
        assert!(reopens(NVML_ERROR_GPU_IS_LOST));
        assert!(!reopens(3));
    }

    /// The struct NVML checks its version against: 40 bytes, version 2.
    #[test]
    fn the_memory_v2_version_is_nvmls() {
        assert_eq!(MEMORY_V2_VERSION, 0x0200_0028);
    }
}
