//! Model process pool — manages one subprocess per loaded ModelId.
//!
//! When a model is unloaded, its worker process is killed and the OS/CUDA
//! driver reclaims all GPU memory immediately — no restart required.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use dashmap::DashMap;
use tokio::process::Child;
use tokio::sync::{mpsc, Mutex};
use uuid::Uuid;

use crate::error::SwarmError;
use crate::inference::router::StreamingTokenEvent;
use crate::inference::worker_ipc::*;
use crate::types::{ModelId, PrefixBlockEntry, SamplingParams};

/// Cross-platform IPC stream halves between daemon and model worker.
/// Unix: AF_UNIX socket with 0o600 filesystem perms (current user only).
/// Windows: named pipe with default DACL (current logon session only).
pub(crate) type IpcReader = interprocess::local_socket::tokio::RecvHalf;
pub(crate) type IpcWriter = interprocess::local_socket::tokio::SendHalf;

/// Item 8 Phase 1: each prefix-cache insert in a worker emits one of these
/// over the pool's `prefix_manifest_tx`. The daemon-side forwarder drains
/// the channel, broadcasts a `SwarmMessage::PrefixCacheAnnounce`, and folds
/// the blocks into the local cross-node index (so a single-node loopback
/// test sees the wire path end-to-end).
#[derive(Clone, Debug)]
pub struct PrefixManifestEvent {
    pub model_id: ModelId,
    pub blocks: Vec<PrefixBlockEntry>,
}

/// Progress on a request that has not produced its first token yet. Workers
/// emit one per prefill chunk; the daemon-side forwarder stamps it onto the
/// request's `RequestTrace`, which is what every surface reads from.
#[derive(Clone, Debug)]
pub struct ProgressEvent {
    pub request_id: Uuid,
    pub phase: crate::inference::worker_ipc::ProgressPhase,
    pub done: u32,
    pub total: u32,
}

/// Item 8 Phase 2b: worker-initiated cross-node prefix-KV probe. The
/// daemon's probe handler drains these off the channel, runs
/// `SharedState::try_fetch_cross_node_prefix`, and sends the (possibly
/// empty) result back via `ModelProcessPool::send_prefix_fetch_result`.
#[derive(Clone, Debug)]
pub struct PrefixProbeEvent {
    pub model_id: ModelId,
    pub request_id: Uuid,
    pub blocks: Vec<PrefixBlockEntry>,
}

const WORKER_CONNECT_TIMEOUT_SECS: u64 = 30;

/// Exponential backoff for repeat worker-spawn failures: 1s, 2s, 4s, 8s,
/// 16s, 32s, capped at 60s. The arriving request gets `ModelNotAvailable`
/// during the cooldown window, so a permanently-broken model can't drown
/// the inference path in 30-second connect timeouts.
fn spawn_failure_cooldown(consecutive_failures: u32) -> std::time::Duration {
    let secs = match consecutive_failures {
        0 | 1 => 1,
        2 => 2,
        3 => 4,
        4 => 8,
        5 => 16,
        6 => 32,
        _ => 60,
    };
    std::time::Duration::from_secs(secs)
}
/// Default KV-cache TTL in seconds (10 minutes). Overridden by config at startup.
pub const DEFAULT_KV_CACHE_TTL_SECS: u64 = 600;

/// Per-request buffered channel capacity for multiplexed worker responses.
/// Long decode streams emit one WorkerMsg::Token per generated token; 256 gives
/// plenty of headroom for a caller that's slow to consume without stalling the
/// reader actor.
const RESPONSE_CHANNEL_CAPACITY: usize = 256;

/// Response channel entry: a bounded mpsc sender carrying `(WorkerMsg, payload_bytes)`.
type ResponseTx = mpsc::Sender<(WorkerMsg, Vec<u8>)>;

/// Shared map from `request_id` to the caller's response channel, tagged with
/// a unique per-attempt token. The reader actor looks up each inbound message's
/// `request_id` here to route the reply.
///
/// The token exists because a `request_id` is NOT unique across concurrent
/// attempts: a router retry (and a hedge race) re-sends the *same* id while the
/// original is still in flight. Keyed by id alone, the retry's insert silently
/// dropped the original's sender and the original's cleanup then removed the
/// retry's — killing both. See `WorkerHandle::register_response`.
type ResponseMap = Arc<DashMap<Uuid, (u64, ResponseTx)>>;

/// Source of per-attempt tokens for `ResponseMap`. Process-wide; only equality
/// matters, never ordering.
/// One GPU-resident model considered for reclaim, as the planner sees it.
#[derive(Debug, Clone)]
struct VramReclaimCandidate {
    model: ModelId,
    /// What admission has charged for it — the figure the budget is spent in.
    charge_mb: u64,
    idle_secs: u64,
    /// A request is in flight against it right now.
    busy: bool,
}

/// Choose which resident models to unload so `needed_mb` fits in `budget`.
///
/// Pure so the policy can be tested without spawning a worker: the wiring is
/// verified by running the node, this decides what the wiring does.
///
/// Returns an EMPTY plan when the request cannot be satisfied even by
/// reclaiming everything eligible. That is the load-bearing case: unloading
/// models and still not fitting costs the user a cold start and buys nothing,
/// because the model was going to the CPU either way.
fn plan_vram_reclaim(
    budget_mb: u64,
    committed_mb: u64,
    needed_mb: u64,
    candidates: Vec<VramReclaimCandidate>,
) -> Vec<(ModelId, u64)> {
    if committed_mb.saturating_add(needed_mb) <= budget_mb {
        return Vec::new();
    }
    let mut eligible: Vec<VramReclaimCandidate> = candidates
        .into_iter()
        .filter(|c| !c.busy && c.charge_mb > 0 && c.idle_secs >= vram_make_room_min_idle_secs())
        .collect();
    // Most idle first: `spawned_at` cannot tell a worker answering steadily for
    // an hour from one loaded an hour ago and never used since.
    eligible.sort_by_key(|a| std::cmp::Reverse(a.idle_secs));

    let mut plan = Vec::new();
    let mut freed = 0u64;
    for c in eligible {
        if committed_mb.saturating_sub(freed).saturating_add(needed_mb) <= budget_mb {
            break;
        }
        freed = freed.saturating_add(c.charge_mb);
        plan.push((c.model, c.charge_mb));
    }
    if committed_mb.saturating_sub(freed).saturating_add(needed_mb) > budget_mb {
        return Vec::new();
    }
    plan
}

/// How long a GPU-resident model must have gone unused before another model
/// may take the card from it.
///
/// **Protects a model under active use from being displaced. It is not thrash
/// prevention, and it was measured that thrash prevention is not worth buying.**
///
/// It was 60 s, invented on the reasoning that two models alternating faster
/// than they load would evict each other on every request. Measured on
/// 2026-08-28 — two models alternating in multi-turn conversation on one card,
/// same binary, floor 60 s against floor 0 s: **299 s against 82 s, a 3.65x
/// loss.** An external tester's one-shot measurement on different hardware had
/// already found the same direction (8.1 s on the processor against 2.5 s to
/// swap and run). Both arms are in `docs/FUTURE_WORK.md`.
///
/// Two things that measurement showed, both against my own prior reasoning:
///
/// - **The swap is the cheap side.** A reload plus re-prefill on the card cost
///   7-14 s per turn; the same model on the processor cost 15-20 s per turn
///   *with* a warm prefix cache, and 230 s for its first. I had argued the
///   opposite — that discarding the prefix cache on eviction would eat the gain
///   — and the multi-turn case I said would show it is the case that refutes it.
/// - **Idle time cannot express "is swapping worth it".** Under alternation the
///   incumbent has always just been used, so an idle-time floor is inert or
///   total and never in between: at 60 s no swap ever happened and one model
///   held the card indefinitely while the other ran on the processor for the
///   whole conversation.
///
/// So this is now only what an idle-time predicate *can* honestly do: refuse to
/// take the card from a model still in active use, on top of the in-flight
/// check that `plan_vram_reclaim` already applies. Sized to a few seconds of
/// continuing traffic rather than to the cost of a load — a model being asked
/// something every few seconds keeps the card; one that has gone quiet for the
/// length of a turn gives it up.
///
/// Costing the swap properly — load time against what the processor would cost
/// for this model — is the real answer and needs per-model figures for both;
/// `docs/FUTURE_WORK.md` carries it with the measurements that justify it.
const VRAM_MAKE_ROOM_MIN_IDLE_SECS_DEFAULT: u64 = 5;

/// `SWARMLLM_VRAM_SWAP_MIN_IDLE_SECS` — pin the floor, for A/B measurement.
///
/// Same discipline as `SWARMLLM_DECODE_THREADS` and `SWARMLLM_DECODE_ATTN`:
/// both arms must be the same binary or the comparison measures the build.
/// `=0` makes every swap eligible, `=60` restores the old constant. Kept out of
/// `config/default.toml` deliberately — it exists to measure the policy, not to
/// hand users a dial in place of one.
fn vram_make_room_min_idle_secs() -> u64 {
    static CACHED: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| {
        match std::env::var("SWARMLLM_VRAM_SWAP_MIN_IDLE_SECS")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
        {
            Some(secs) => {
                tracing::info!(
                    secs,
                    default = VRAM_MAKE_ROOM_MIN_IDLE_SECS_DEFAULT,
                    "SWARMLLM_VRAM_SWAP_MIN_IDLE_SECS override in force"
                );
                secs
            }
            None => VRAM_MAKE_ROOM_MIN_IDLE_SECS_DEFAULT,
        }
    })
}

/// Everything the "should this worker go back to the card?" decision looks at.
///
/// Pure, for the same reason [`plan_vram_reclaim`] is: the policy can then be
/// tested without spawning a worker or owning a GPU, and the method that
/// gathers these values does nothing but gather them.
#[derive(Debug, Clone, Copy)]
struct PromotionInputs {
    /// This node put the running worker on the processor (`--gpu-layers 0`).
    cpu_placed: bool,
    /// A reason this model would go to the processor again whatever the memory
    /// situation: the user's own `gpu_layers = 0`, or a card below this build's
    /// kernel floor. **A VRAM pin is deliberately NOT one of them** — that is
    /// the condition this whole decision exists to reconsider.
    permanently_cpu_bound: bool,
    /// A request is in flight against the worker right now.
    busy: bool,
    idle_secs: u64,
    /// What the card would have to give it, as admission priced it.
    gpu_estimate_mb: u64,
    budget_mb: u64,
    committed_mb: u64,
    /// Megabytes the pool would be willing to reclaim from OTHER models right
    /// now, from a dry run of `plan_vram_reclaim` — so this asks whether the
    /// model fits in the room that can be made, not only in the room lying
    /// about. The plan carries its own guards (idle floor, in-flight, and
    /// all-or-nothing), so a non-zero figure is memory the pool has already
    /// agreed it may take.
    reclaimable_mb: u64,
}

/// Should this processor-resident worker be retired so the model reloads onto
/// the graphics card?
///
/// **The alternative to acting is a model stuck on the processor for as long as
/// it stays resident.** A GPU OOM (or a momentarily full card) pins a model to
/// the CPU; freeing memory lifts the pin; and nothing then re-examined the
/// worker that pin had produced. `get_or_spawn`'s fast path returns any
/// resident worker regardless of device, and the CPU worker survives until
/// `idle_unload_secs` (15 minutes) of *no requests at all* — which a user
/// actively working with the model never reaches. So lifting the pin bought
/// nothing for the model it was lifted for. Reported by an external tester
/// 2026-08-27: llama-3.2-3b took the card, gemma-2-2b-it spawned on the
/// processor 35 s later, the idle-unload then freed llama and logged `GPU
/// memory freed — clearing CPU pins`, and re-requesting gemma against an idle
/// card (653 of 6141 MB in use) reused the processor worker.
///
/// Four guards, three of them carried over from [`plan_vram_reclaim`] because
/// this destroys something that is working and they are what makes that safe:
///
/// - **Only a worker this node demoted.** On a machine with no card,
///   `cpu_placed` is false and there is nowhere to promote to.
/// - **The reason must actually be gone.** Of the three causes in
///   [`ModelProcessPool::cpu_reason`], only the OOM pin ever clears — so this
///   fires on the event that lifted it rather than polling for one.
/// - **Never a busy worker, and not one used inside
///   [`vram_make_room_min_idle_secs`].** Unloading kills the subprocess, and a
///   request that arrives between the check and the kill dies with it. The idle
///   floor makes that window empty rather than merely unlikely. A model under
///   continuous load therefore waits for a gap in the traffic — the pin stays
///   lifted, so the promotion happens at the first one.
/// - **Cost the move before making it.** Retiring a worker and then failing
///   admission costs a cold start and buys nothing, so the model must fit the
///   budget as it stands *now*. An unreadable estimate (0) or an unset budget
///   is not evidence, and leaves the model where it is.
fn should_return_to_gpu(i: &PromotionInputs) -> bool {
    if !i.cpu_placed || i.permanently_cpu_bound || i.busy {
        return false;
    }
    if i.idle_secs < vram_make_room_min_idle_secs() {
        return false;
    }
    if i.budget_mb == 0 || i.gpu_estimate_mb == 0 {
        return false;
    }
    let free_after_reclaim = i
        .budget_mb
        .saturating_sub(i.committed_mb.saturating_sub(i.reclaimable_mb));
    i.gpu_estimate_mb <= free_after_reclaim
}

static RESPONSE_ATTEMPT_TOKEN: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

/// A handle to a running model worker subprocess.
///
/// The socket is split into a shared writer (one-at-a-time under `Mutex`) and
/// a reader owned by a dedicated actor task. The reader dispatches each
/// incoming `WorkerMsg` to the per-request channel keyed by `request_id`,
/// allowing N concurrent `forward()` / `generate()` callers to interleave
/// their requests through the same worker without serializing on a single
/// full-request mutex.
struct WorkerHandle {
    /// The worker subprocess.
    ///
    /// `Option` so `Drop` can move it into a reaper task. Killing a process and
    /// the OS reclaiming its device memory are separate events, and the budget
    /// must not be handed back between them — see `exited`.
    child: Option<Child>,
    /// Set once the subprocess has actually been reaped, not merely signalled.
    ///
    /// `Child::start_kill` sends the signal and returns; the kernel tears the
    /// process down and frees its CUDA allocations afterwards. Releasing the
    /// VRAM charge on the signal made `vram_committed_mb` report zero while the
    /// card was still full, so an admission landing in that window passed
    /// against memory that did not exist and the new worker died with
    /// `CUDA_ERROR_OUT_OF_MEMORY`. Reported as an out-of-memory crash that
    /// "landed exactly when auto-manage was mid-eviction of a different model".
    exited: Arc<AtomicBool>,
    /// Write half of the IPC socket. Brief lock held only for the duration of
    /// one outbound framed message (header + optional binary payload).
    writer: Mutex<IpcWriter>,
    /// Per-request response channels. The reader actor inserts `(msg, payload)`
    /// tuples keyed by `request_id`; callers register a channel before sending
    /// their request and drain it until they get a terminal message.
    responses: ResponseMap,
    /// Set to true when the reader actor observes a socket error. Subsequent
    /// callers short-circuit with an error + trigger worker eviction.
    dead: Arc<AtomicBool>,
    /// Socket path used to connect, kept for drop-time cleanup.
    ///
    /// Unix only: the filesystem entry has to be unlinked when the worker goes
    /// away, whereas a Windows named pipe is reclaimed by the kernel once every
    /// handle closes. Carrying the field on Windows would be storing a value
    /// nothing reads, which is what the Windows lint build reports.
    #[cfg(unix)]
    socket_name: String,
    /// Handle to the reader actor task. Aborted on drop so the task doesn't
    /// outlive its worker; also unblocks any pending `recv_worker` in tests.
    reader_handle: tokio::task::JoinHandle<()>,
    /// When this worker was spawned.
    ///
    /// Idle-unload needs a floor on how long a model *could* have been idle. A
    /// model with no recorded request history is not "idle forever" — it cannot
    /// have been idle longer than it has existed. Without this, a model loaded
    /// seconds ago was evicted as though it had sat unused for the whole
    /// configured window, killing the request that had just loaded it.
    spawned_at: std::time::Instant,
    /// Seconds since this worker last had a request registered against it.
    ///
    /// `spawned_at` is only an upper bound on idleness — it cannot tell a model
    /// answering steadily for an hour from one loaded an hour ago and never
    /// used since, and those deserve opposite treatment when something has to
    /// give up the card. Stamped in `register_response`, which is the ONE place
    /// every execution path (local, distributed, peer-served) passes through,
    /// so no caller can forget it.
    ///
    /// Stored as seconds since `spawned_at` rather than a wall-clock stamp, so
    /// a clock step cannot make a busy worker look idle for hours.
    last_used: AtomicU64,
    /// Why this node put this worker on the processor — `None` means it was
    /// spawned for the graphics card.
    ///
    /// A property of the WORKER, not of the model: it records where the process
    /// actually went and why, which is the only thing still true minutes later.
    /// [`ModelProcessPool::cpu_reason`] answers for a spawn happening *now*, and
    /// the pin it consults is deliberately cleared while the worker it produced
    /// is still running — so asking it about a resident worker gives the wrong
    /// answer precisely when the answer matters, both for the promotion decision
    /// and for the dashboard's "why is this not on my GPU?".
    ///
    /// Note it is `None` on a node with no graphics card at all: nothing was
    /// demoted, there is simply nowhere else to run.
    placed_on_cpu_because: Option<CpuReason>,
    /// What the card would have to give this model, as admission priced it at
    /// spawn — 0 when the geometry could not be read.
    ///
    /// Carried here so a later request can ask "would it fit now?" without
    /// going back to disk: `estimate_gpu_footprint_mb` re-reads
    /// `gguf_header.bin` and scans the model directory, which is fine once per
    /// spawn and not fine once per request.
    gpu_estimate_mb: u64,
    /// A hybrid split — `(layers on the card, layers in total)` — when admission
    /// chose one; `None` for a worker wholly on either device.
    ///
    /// Recorded because nothing else says it. The split was decided here, sent
    /// to the worker as `--gpu-layers`, logged once at spawn, and then existed
    /// nowhere a person could read: the models page showed `fits_on_gpu:
    /// true` and no placement note for a model running 13 of its 28 layers on
    /// the card, so a user could not tell why it was slower than expected.
    gpu_layers_on_card: Option<(usize, usize)>,
    /// Is this worker's memory charged against the system-RAM budget?
    ///
    /// Decided ONCE at spawn by `charges_ram` and recorded, because a later
    /// segment charged to the same worker must go to the same accountant. Its
    /// three inputs — going to the processor, no card detected, a build with no
    /// CUDA — are stable for a running worker, but re-deriving them would be
    /// the same "prediction versus fact" mistake `placed_on_cpu_because` exists
    /// to avoid.
    charged_against_ram: bool,
}

/// Longest path a Unix domain socket may have, INCLUDING its NUL terminator.
///
/// `sun_path` in `sockaddr_un` is a fixed-size array, not a pointer: 104 bytes
/// on macOS and the BSDs, 108 on Linux. A path one byte over does not truncate
/// — `bind` refuses outright, and the failure surfaces from `interprocess` as
/// "local socket name length exceeds capacity of sun_path of sockaddr_un".
///
/// **This is not a theoretical limit.** macOS gives every user a private
/// per-boot temporary directory (`/var/folders/xx/…/T/`, measured at 49
/// characters on a Mac mini M4), and the old name — `swarmllm-worker-` plus a
/// 36-character UUID plus `.sock`, 57 characters — took that to 106. So EVERY
/// worker spawn failed on macOS, which means every request failed: with prompt
/// privacy on, the first and last layers are always local, so a node that
/// cannot start a worker cannot answer at all, whatever the swarm holds
/// (reported 2026-09-03, two models, first attempt each time). `TMPDIR=/tmp`
/// was the tester's workaround. Linux never saw it: `/tmp` is 4 characters.
#[cfg(unix)]
const SUN_PATH_MAX: usize = if cfg!(any(
    target_vendor = "apple",
    target_os = "freebsd",
    target_os = "openbsd",
    target_os = "netbsd",
    target_os = "dragonfly"
)) {
    104
} else {
    108
};

/// The socket filename, kept SHORT on purpose.
///
/// 12 hex characters of randomness rather than a 36-character UUID: the name
/// only has to be unique among this machine's live workers, and every
/// character spent here is a character the containing directory cannot use.
/// 48 bits collides with probability ~1e-10 at a thousand concurrent workers,
/// and a collision is a clean bind failure and a retry, not corruption.
#[cfg(unix)]
fn worker_socket_filename() -> String {
    let u = Uuid::new_v4();
    format!("swarmllm-{}.sock", &u.simple().to_string()[..12])
}

/// The first candidate directory in which `filename` fits inside `limit`.
///
/// Pure so the arithmetic can be tested without a filesystem, on any platform:
/// the macOS failure was a length calculation, and length calculations are
/// exactly what a Linux-only test suite cannot see.
///
/// `limit` counts the NUL terminator, so the path itself must be strictly
/// shorter than it.
#[cfg(unix)]
fn first_dir_that_fits(
    candidates: &[std::path::PathBuf],
    filename: &str,
    limit: usize,
) -> Option<std::path::PathBuf> {
    candidates
        .iter()
        .map(|d| d.join(filename))
        .find(|p| p.as_os_str().len() < limit)
}

/// A short, private per-user directory to fall back to when `$TMPDIR` is too
/// long for a socket path.
///
/// **Only consulted when `$TMPDIR` does not fit**, which on Linux is never —
/// so an ordinary node neither creates this nor touches `/tmp` at all.
///
/// `/tmp` is world-writable, so another local user can pre-create
/// `/tmp/swarmllm-<uid>` and would otherwise own the place our sockets live.
/// Two rules follow, and both matter:
///
/// - **Create it with its mode, never chmod it afterwards.** `mkdir(2)` with
///   0700 is atomic and fails outright if the name is taken, so there is no
///   moment when the directory exists more permissively than intended — and,
///   more to the point, we never change the permissions of something we did
///   not just create. `create_dir_all` + `set_permissions` would have
///   followed a planted symlink and chmod'd its target.
/// - **An existing directory is VERIFIED, not repaired.** `symlink_metadata`,
///   not `metadata`, because a symlink pointing elsewhere is precisely the
///   attack and following it is how you fail to notice. Anything that is not
///   a real directory, not ours, or group/world accessible is refused; the
///   caller still has `$TMPDIR` and the error names both.
#[cfg(unix)]
fn short_private_socket_dir() -> Option<std::path::PathBuf> {
    use std::os::unix::fs::{DirBuilderExt, MetadataExt, PermissionsExt};
    let uid = unsafe { libc::getuid() };
    let dir = std::path::PathBuf::from(format!("/tmp/swarmllm-{uid}"));
    match std::fs::DirBuilder::new().mode(0o700).create(&dir) {
        // Ours, made just now, with the mode already applied.
        Ok(()) => return Some(dir),
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {}
        Err(_) => return None,
    }
    let md = std::fs::symlink_metadata(&dir).ok()?;
    (md.is_dir() && md.uid() == uid && md.permissions().mode() & 0o077 == 0).then_some(dir)
}

/// Where this worker's IPC socket goes: the first directory whose resulting
/// path fits [`SUN_PATH_MAX`].
///
/// `$TMPDIR` first, because on macOS it is per-user and private and on Linux it
/// is short; the `/tmp/swarmllm-<uid>` fallback exists only for the case that
/// made this function necessary. Returns the path AND the directories that
/// were tried, so a failure can name them — the reported error said only that
/// the limit was exceeded, and the tester could not get the offending path out
/// of the logs even at debug level.
#[cfg(unix)]
fn worker_socket_path() -> Result<String, SwarmError> {
    let filename = worker_socket_filename();
    let mut candidates = vec![std::env::temp_dir()];
    // The fallback is built ONLY if the temp directory does not fit. On Linux
    // it never does not, so an ordinary node never creates it — a directory
    // nobody needs is still a directory somebody has to explain.
    if first_dir_that_fits(&candidates, &filename, SUN_PATH_MAX).is_none() {
        if let Some(short) = short_private_socket_dir() {
            candidates.push(short);
        }
    }
    match first_dir_that_fits(&candidates, &filename, SUN_PATH_MAX) {
        Some(p) => p
            .to_str()
            .map(str::to_string)
            .ok_or_else(|| SwarmError::Internal("socket path not UTF-8".into())),
        None => Err(SwarmError::ServiceUnavailable(format!(
            "no directory short enough for a worker socket (the limit is {SUN_PATH_MAX} \
             characters on this platform, including the file name): tried {}. Set TMPDIR to \
             a shorter path, for example TMPDIR=/tmp.",
            candidates
                .iter()
                .map(|d| format!("{} ({} chars)", d.display(), d.as_os_str().len()))
                .collect::<Vec<_>>()
                .join(", ")
        ))),
    }
}

/// Pull the `request_id` field out of any `WorkerMsg` variant that carries one.
/// `Ready` / `Bye` have no request_id and are dropped by the reader actor
/// (they're only relevant during spawn, which handshakes synchronously).
/// `BatchResult` carries N request_ids and is handled specially by the reader
/// actor — this helper returns `None` for it. `PrefixManifestUpdate` also
/// carries no request_id; the reader actor routes it through the dedicated
/// `prefix_manifest_tx` channel instead.
fn worker_msg_request_id(msg: &WorkerMsg) -> Option<Uuid> {
    match msg {
        WorkerMsg::LayerResult(r) => Some(r.request_id),
        WorkerMsg::Token { request_id, .. }
        | WorkerMsg::GenerateDone { request_id, .. }
        | WorkerMsg::Error { request_id, .. } => Some(*request_id),
        // `PrefixSnapshotResponse` is correlated by `request_id` via the
        // normal response-routing channel — `fetch_local_snapshot`
        // registers a receiver up-front and waits for the reply.
        WorkerMsg::PrefixSnapshotResponse { request_id, .. } => Some(*request_id),
        // `Progress` carries a request_id but is explicitly NOT a response:
        // routing it to the per-request channel would hand the waiting caller a
        // progress note where it expects a result. Side-band, like the prefix
        // messages above.
        WorkerMsg::BatchResult { .. }
        | WorkerMsg::PrefixManifestUpdate { .. }
        | WorkerMsg::PrefixFetchProbe { .. }
        | WorkerMsg::Progress { .. }
        | WorkerMsg::Ready
        | WorkerMsg::Bye => None,
    }
}

/// Reader actor: owns the read half of the worker socket, dispatches each
/// inbound message to the right per-request channel. Exits when the socket
/// errors out (worker died, IPC corrupted); sets `dead` and drops all
/// in-flight response senders to wake waiting callers with `None`.
async fn reader_actor(
    mut reader: IpcReader,
    responses: ResponseMap,
    dead: Arc<AtomicBool>,
    model_id: ModelId,
    prefix_manifest_tx: Option<mpsc::Sender<PrefixManifestEvent>>,
    prefix_probe_tx: Option<mpsc::Sender<PrefixProbeEvent>>,
    progress_tx: Option<mpsc::Sender<ProgressEvent>>,
) {
    loop {
        match recv_worker(&mut reader).await {
            Ok((msg, payload)) => {
                // BatchResult carries N inner results + a concatenated payload.
                // Split it so each caller sees a synthesized LayerResult on
                // their individual response channel — callers using the batch
                // API register one channel per request_id up front.
                if let WorkerMsg::BatchResult {
                    results,
                    activation_lens,
                } = msg
                {
                    dispatch_batch_result(&responses, results, activation_lens, payload).await;
                    continue;
                }
                // Cross-node prefix-cache announcement (Item 8 Phase 1).
                // No request_id; route through the daemon-installed
                // forwarder channel. Drop silently when no forwarder is
                // installed (unit tests / pre-startup).
                if let WorkerMsg::PrefixManifestUpdate {
                    model_id: announce_model,
                    blocks,
                } = msg
                {
                    if let Some(tx) = prefix_manifest_tx.as_ref() {
                        // try_send: never block the IPC reader. Daemon being
                        // slow or absent must not stall worker responses.
                        if let Err(tokio::sync::mpsc::error::TrySendError::Full(_)) =
                            tx.try_send(PrefixManifestEvent {
                                model_id: announce_model,
                                blocks,
                            })
                        {
                            tracing::debug!("prefix manifest channel full — dropping announce");
                        }
                    }
                    continue;
                }
                // Prefill/load progress. Best-effort by construction: a full
                // channel means the display lags, which is strictly better than
                // stalling the worker's result path to deliver a status note.
                if let WorkerMsg::Progress {
                    request_id,
                    phase,
                    done,
                    total,
                } = msg
                {
                    if let Some(tx) = progress_tx.as_ref() {
                        let _ = tx.try_send(ProgressEvent {
                            request_id,
                            phase,
                            done,
                            total,
                        });
                    }
                    continue;
                }
                // Worker-initiated cross-node probe (Item 8 Phase 2b).
                if let WorkerMsg::PrefixFetchProbe {
                    request_id,
                    model_id: probe_model,
                    blocks,
                } = msg
                {
                    if let Some(tx) = prefix_probe_tx.as_ref() {
                        let probe_model_for_log = probe_model.clone();
                        if let Err(tokio::sync::mpsc::error::TrySendError::Full(_)) =
                            tx.try_send(PrefixProbeEvent {
                                model_id: probe_model,
                                request_id,
                                blocks,
                            })
                        {
                            // Dropped probes look indistinguishable from cross-node
                            // misses on the worker side — log so operators can
                            // distinguish channel-saturation from actual misses
                            // when chasing prefix-cache hit-rate regressions.
                            tracing::debug!(
                                model = %probe_model_for_log,
                                %request_id,
                                "prefix probe channel full — dropping probe"
                            );
                        }
                    }
                    continue;
                }
                if let Some(rid) = worker_msg_request_id(&msg) {
                    // `get` returns a Ref, which holds a shard lock. Clone the
                    // Sender and drop the Ref *before* awaiting `send` so we
                    // don't hold a DashMap shard across an await point (that
                    // would risk deadlock on a concurrent insert/remove).
                    if let Some(tx) = responses.get(&rid).map(|r| r.value().1.clone()) {
                        // Send best-effort; if the caller has already hung up
                        // we just drop the message.
                        let _ = tx.send((msg, payload)).await;
                    } else {
                        tracing::debug!(
                            request_id = %rid,
                            "Worker response for unknown request_id (caller dropped?)"
                        );
                    }
                }
            }
            Err(e) => {
                tracing::warn!(model = %model_id, error = %e, "Worker reader exiting — evicting");
                // R107: Release ordering pairs with Acquire loads at every
                // dead-check call site. On weakly-ordered CPUs (ARM), a
                // Relaxed load could legally observe `dead == false` after
                // this store and the subsequent `responses.clear()`,
                // letting a concurrent caller insert into `responses` after
                // the clear — that channel would never receive anything,
                // and the caller's `recv()` loop has no timeout, so it
                // would hang forever. Release/Acquire prevents this race.
                dead.store(true, Ordering::Release);
                // Clear all pending response channels; dropping the Senders
                // makes each caller's `recv()` return `None`, which we map
                // to a "worker died" error.
                responses.clear();
                return;
            }
        }
    }
}

/// Split an incoming IpcLayerResult's binary payload back into the two
/// mutually exclusive fields it can carry (activations XOR spec_logits).
/// The worker guarantees only one of `has_activations` / `has_spec_logits`
/// is true; if the invariant is violated we preserve activations and drop
/// spec_logits with a warning (safer than panicking on the hot path).
fn reconstruct_layer_payload(
    has_activations: bool,
    has_spec_logits: bool,
    spec_logits_dims: Option<(u32, u32)>,
    payload: Vec<u8>,
) -> Result<(Vec<u8>, Vec<Vec<f32>>), SwarmError> {
    if has_activations && has_spec_logits {
        tracing::warn!(
            "IpcLayerResult has both has_activations and has_spec_logits — treating as activations"
        );
        return Ok((payload, Vec::new()));
    }
    if has_spec_logits {
        let dims = spec_logits_dims
            .ok_or_else(|| SwarmError::Internal("spec_logits flagged but dims missing".into()))?;
        let logits = crate::inference::worker_ipc::decode_spec_logits(&payload, dims)
            .map_err(SwarmError::Internal)?;
        Ok((Vec::new(), logits))
    } else if has_activations {
        Ok((payload, Vec::new()))
    } else {
        Ok((Vec::new(), Vec::new()))
    }
}

/// Fan out a BatchResult to N per-request channels. Splits the concatenated
/// payload into per-slot byte slices by `activation_lens`, wraps each inner
/// `IpcLayerResult` in a `WorkerMsg::LayerResult`, and delivers to the caller
/// registered under each `request_id`.
async fn dispatch_batch_result(
    responses: &ResponseMap,
    results: Vec<IpcLayerResult>,
    activation_lens: Vec<u32>,
    payload: Vec<u8>,
) {
    if results.len() != activation_lens.len() {
        tracing::warn!(
            results = results.len(),
            lens = activation_lens.len(),
            "BatchResult len mismatch — dropping batch"
        );
        return;
    }
    let mut cursor = 0usize;
    for (r, len) in results.into_iter().zip(activation_lens) {
        let len = len as usize;
        // SEC: only consume payload bytes for slots that actually carry an
        // activation tensor. A malformed worker response with len > 0 on a
        // has_activations=false slot would otherwise silently shift the
        // cursor past valid bytes, corrupting EVERY subsequent slot's
        // payload without any error returned.
        let (slot_payload, advance) = if r.has_activations {
            let end = cursor.saturating_add(len);
            if end <= payload.len() {
                (payload[cursor..end].to_vec(), len)
            } else {
                tracing::warn!(
                    request_id = %r.request_id,
                    cursor, len, payload_len = payload.len(),
                    "BatchResult activation length exceeds payload"
                );
                (Vec::new(), 0)
            }
        } else {
            if len != 0 {
                tracing::warn!(
                    request_id = %r.request_id,
                    len,
                    "BatchResult non-activation slot has len > 0 — ignoring (potential malformed worker response)"
                );
            }
            (Vec::new(), 0)
        };
        cursor = cursor.saturating_add(advance);
        let rid = r.request_id;
        let tx_opt = responses.get(&rid).map(|e| e.value().1.clone());
        if let Some(tx) = tx_opt {
            let _ = tx.send((WorkerMsg::LayerResult(r), slot_payload)).await;
        } else {
            tracing::debug!(
                request_id = %rid,
                "Batch slot reply for unknown request_id (caller dropped?)"
            );
        }
    }
}

/// How long an eviction waits for the subprocess to actually be reaped before
/// handing its memory budget back.
///
/// Bounds the wait on what varies — process teardown, which is fast (the
/// kernel is freeing already-allocated pages, not doing work proportional to
/// the model) but not instant. Long enough that the normal case always
/// completes inside it; short enough that a wedged process cannot stall an
/// eviction triggered from the request path.
const WORKER_EXIT_WAIT: std::time::Duration = std::time::Duration::from_secs(5);

/// How long a retiring worker is given to finish the requests already in flight
/// before it is told to shut down.
///
/// Retirement is triggered by models that are *idle* (both displacement paths
/// impose an idle floor), so in practice there is nothing to wait for and the
/// drain returns immediately. This bounds the exception, not the rule.
///
/// **Finite deliberately.** A request that never completes must not hold a
/// model's memory for the life of the daemon — that would refuse every later
/// load on the device, which is a worse failure than the one being closed here.
/// Same trade, and same reasoning, as `WORKER_EXIT_WAIT` below it.
const WORKER_DRAIN_WAIT: std::time::Duration = std::time::Duration::from_secs(10);

/// Poll interval while waiting for a worker to be reaped.
const WORKER_EXIT_POLL: std::time::Duration = std::time::Duration::from_millis(20);

impl Drop for WorkerHandle {
    fn drop(&mut self) {
        // Stop the reader actor so it doesn't outlive the socket.
        self.reader_handle.abort();
        // Kill the child process if still running, then reap it in the
        // background and publish the fact. `start_kill` only signals; whoever
        // is waiting to reuse this worker's memory needs to know when the
        // process is actually gone, which is when the OS has taken its device
        // allocations back.
        if let Some(mut child) = self.child.take() {
            let _ = child.start_kill();
            let exited = self.exited.clone();
            // No runtime in some unit tests; there the flag simply stays false
            // and the waiter falls through on its timeout, which is the same
            // conservative outcome as before this existed.
            if let Ok(rt) = tokio::runtime::Handle::try_current() {
                rt.spawn(async move {
                    let _ = child.wait().await;
                    exited.store(true, Ordering::Release);
                });
            } else {
                self.exited.store(true, Ordering::Release);
            }
        }
        // Clean up the socket file (Unix only — Windows named pipes are
        // reclaimed by the kernel when all handles close).
        #[cfg(unix)]
        let _ = std::fs::remove_file(&self.socket_name);
    }
}

/// RAII guard: unregister a request's response channel when the caller drops
/// it (whether the request finished, errored, or the caller was cancelled).
/// Without this, a cancelled request leaks its entry in `responses` forever.
struct ResponseGuard {
    responses: ResponseMap,
    request_id: Uuid,
    /// Identifies THIS attempt's entry. `Drop` removes the map entry only when
    /// it still carries this token, so a superseded attempt's cleanup cannot
    /// tear down the attempt that displaced it.
    token: u64,
    /// Worker to notify if this guard drops without being disarmed. `None`
    /// suppresses the cancel entirely (used where the request is known to have
    /// already reached the worker's terminal state).
    worker: Option<Arc<WorkerHandle>>,
}

/// Claim `request_id` in `map` for a fresh attempt. Returns the new attempt's
/// token plus the token it displaced, if another attempt was already
/// registered under the same id.
fn claim_response_slot(map: &ResponseMap, request_id: Uuid, tx: ResponseTx) -> (u64, Option<u64>) {
    let token = RESPONSE_ATTEMPT_TOKEN.fetch_add(1, Ordering::Relaxed);
    let displaced = map.insert(request_id, (token, tx)).map(|(prev, _)| prev);
    (token, displaced)
}

/// True when `request_id` is registered to an attempt other than `token` —
/// i.e. a later attempt has taken the slot over.
fn response_slot_superseded(map: &ResponseMap, request_id: Uuid, token: u64) -> bool {
    map.get(&request_id).is_some_and(|e| e.value().0 != token)
}

impl WorkerHandle {
    /// Register `tx` as the response channel for `request_id` and return the
    /// guard that owns the entry. This is the ONLY supported way to populate
    /// `responses`: inserting directly cannot produce a correctly-tokened
    /// guard, so a superseded attempt would tear down its successor.
    ///
    /// `cancel_on_drop` sends the worker a cancel if the guard drops while
    /// still armed; pass `false` where there is no compute worth abandoning.
    ///
    /// Returns the guard plus the token displaced from the map, if any. A
    /// `Some` displaced token means another attempt is in flight under the
    /// same `request_id` — the caller has superseded it.
    fn register_response(
        self: &Arc<Self>,
        request_id: Uuid,
        tx: ResponseTx,
        cancel_on_drop: bool,
    ) -> (ResponseGuard, Option<u64>) {
        self.last_used
            .store(self.spawned_at.elapsed().as_secs(), Ordering::Relaxed);
        let (token, displaced) = claim_response_slot(&self.responses, request_id, tx);
        if displaced.is_some() {
            tracing::warn!(
                %request_id,
                "Second in-flight generate for this request_id — superseding \
                 the earlier attempt (router retry or hedge race)"
            );
        }
        (
            ResponseGuard {
                responses: self.responses.clone(),
                request_id,
                token,
                worker: cancel_on_drop.then(|| self.clone()),
            },
            displaced,
        )
    }

    /// Seconds since this worker last had a request registered against it.
    ///
    /// A worker that has never been used reports its full residency, which is
    /// the honest answer: it has been idle for as long as it has existed.
    fn idle_secs(&self) -> u64 {
        self.spawned_at
            .elapsed()
            .as_secs()
            .saturating_sub(self.last_used.load(Ordering::Relaxed))
    }

    /// True when `request_id`'s entry has been taken over by a later attempt.
    /// Distinguishes "a retry displaced me" from "the worker really died",
    /// which present identically as a closed response channel.
    fn response_superseded(&self, request_id: Uuid, token: u64) -> bool {
        response_slot_superseded(&self.responses, request_id, token)
    }
}

impl ResponseGuard {
    /// Mark the request as completed so dropping this guard does NOT cancel it.
    /// Every path that returns a terminal result must call this — otherwise a
    /// normal completion sends a spurious cancel for a request the worker has
    /// already finished. (Harmless, since the worker treats unmatched cancels
    /// as no-ops and sweeps them, but it's pure noise on the IPC socket.)
    fn disarm(&mut self) {
        self.worker = None;
    }
}

impl Drop for ResponseGuard {
    fn drop(&mut self) {
        // Identity-checked: if a retry with the same request_id has already
        // replaced our entry, that entry belongs to the retry and removing it
        // would fail a request that is still perfectly healthy.
        self.responses
            .remove_if(&self.request_id, |_, (tok, _)| *tok == self.token);

        // Still armed → the caller went away before a terminal reply arrived:
        // client disconnected, `tokio::select!` timeout fired, or a hedge race
        // resolved and this was the loser. Tell the worker to stop; otherwise
        // it computes to completion and the reader actor silently discards the
        // result. For a `Generate` that is the whole remaining token budget of
        // GPU time spent on output nobody will read.
        let Some(worker) = self.worker.take() else {
            return;
        };
        // Drop is sync, so the IPC write is handed to a task. `try_current`
        // rather than `Handle::current` because a guard can be dropped outside
        // a runtime in tests, where losing a best-effort cancel is fine.
        let Ok(handle) = tokio::runtime::Handle::try_current() else {
            return;
        };
        let request_id = self.request_id;
        handle.spawn(async move {
            if worker.dead.load(Ordering::Acquire) {
                return;
            }
            let mut writer = worker.writer.lock().await;
            let _ = send_daemon(&mut *writer, &DaemonMsg::CancelRequest { request_id }, &[]).await;
        });
    }
}

/// Auto-coalescing batch scheduler loop. One per `ModelProcessPool`. Collects
/// `Forward` requests into time-windowed batches grouped by `model_id`, then
/// dispatches each group via `pool.forward_batch(...)`. Responses are fanned
/// out to the per-request `oneshot::Sender`s.
///
/// Worker-side CPU fallback (`run_fused_batch_forward` errors out on CPU, which
/// `handle_batch_forward` catches and runs sequentially) means this is safe on
/// every device — GPU workers run the fused `SplitModel::forward_batch` path,
/// CPU workers run the sequential path with zero regression.
async fn batch_scheduler_loop(
    pool: Arc<ModelProcessPool>,
    mut rx: mpsc::Receiver<BatchSchedulerMsg>,
    mut shutdown_rx: tokio::sync::watch::Receiver<bool>,
) {
    use std::collections::HashMap;
    use std::time::{Duration, Instant};
    loop {
        let first = tokio::select! {
            biased;
            _ = shutdown_rx.changed() => {
                tracing::debug!("batch_scheduler_loop: shutdown observed");
                return;
            }
            msg = rx.recv() => match msg {
                Some(m) => m,
                None => return, // sender side dropped
            },
        };
        let collection_ms = pool
            .batch_collection_ms
            .load(std::sync::atomic::Ordering::Relaxed);
        let max_batch = pool
            .max_concurrent_decode_batch
            .load(std::sync::atomic::Ordering::Relaxed) as usize;
        let max_batch = max_batch.max(1);

        let mut pending: Vec<BatchSchedulerMsg> = vec![first];
        if collection_ms > 0 {
            let deadline = Instant::now() + Duration::from_millis(collection_ms);
            while pending.len() < max_batch {
                let now = Instant::now();
                if now >= deadline {
                    break;
                }
                match tokio::time::timeout(deadline - now, rx.recv()).await {
                    Ok(Some(msg)) => pending.push(msg),
                    Ok(None) => {
                        // Sender dropped — process what we have and exit.
                        dispatch_scheduler_pending(&pool, pending).await;
                        return;
                    }
                    Err(_) => break, // deadline elapsed
                }
            }
        }

        // Group by model_id — forward_batch requires a single model.
        let mut by_model: HashMap<ModelId, Vec<BatchSchedulerMsg>> = HashMap::new();
        for msg in pending {
            let BatchSchedulerMsg::Forward { ref fwd, .. } = msg;
            by_model.entry(fwd.model_id.clone()).or_default().push(msg);
        }
        for (_, msgs) in by_model {
            dispatch_scheduler_group(&pool, msgs).await;
        }
    }
}

async fn dispatch_scheduler_pending(pool: &Arc<ModelProcessPool>, pending: Vec<BatchSchedulerMsg>) {
    use std::collections::HashMap;
    let mut by_model: HashMap<ModelId, Vec<BatchSchedulerMsg>> = HashMap::new();
    for msg in pending {
        let BatchSchedulerMsg::Forward { ref fwd, .. } = msg;
        by_model.entry(fwd.model_id.clone()).or_default().push(msg);
    }
    for (_, msgs) in by_model {
        dispatch_scheduler_group(pool, msgs).await;
    }
}

/// Dispatch a same-model group: 1 request → `pool.forward_direct`, ≥2 →
/// `pool.forward_batch`. Fan out to each `resp_tx`.
async fn dispatch_scheduler_group(pool: &Arc<ModelProcessPool>, msgs: Vec<BatchSchedulerMsg>) {
    if msgs.is_empty() {
        return;
    }
    if msgs.len() == 1 {
        let BatchSchedulerMsg::Forward { fwd, resp_tx } =
            msgs.into_iter().next().expect("len == 1 checked above");
        let result = pool.forward_direct(fwd, None).await;
        let _ = resp_tx.send(result);
        return;
    }
    let (forwards, resp_txs): (Vec<_>, Vec<_>) = msgs
        .into_iter()
        .map(|BatchSchedulerMsg::Forward { fwd, resp_tx }| (fwd, resp_tx))
        .unzip();
    let n = forwards.len();
    match pool.forward_batch(forwards).await {
        Ok(results) => {
            if results.len() != n {
                for tx in resp_txs {
                    let _ = tx.send(Err(SwarmError::Internal(format!(
                        "batch result count mismatch: expected {n}, got {}",
                        results.len()
                    ))));
                }
                return;
            }
            // Fan out in sender order (forward_batch preserves input order).
            for (tx, result) in resp_txs.into_iter().zip(results) {
                let _ = tx.send(Ok(result));
            }
        }
        Err(e) => {
            // SwarmError isn't Clone, so we can't propagate `e` to each
            // tx. The vast-majority batch failure cause is worker death
            // (subprocess closed IPC) — ServiceUnavailable is correct
            // for that. Was previously Internal which mapped to 500 and
            // misled operators about the failure class.
            let err_str = e.to_string();
            for tx in resp_txs {
                let _ = tx.send(Err(SwarmError::ServiceUnavailable(format!(
                    "batch failed: {err_str}"
                ))));
            }
        }
    }
}

/// Manages one worker subprocess per loaded ModelId.
///
/// When a model is unloaded, its worker process is killed and the OS/CUDA
/// driver reclaims all GPU memory immediately — no restart required.
///
/// ## Concurrency model
///
/// Each WorkerHandle has a `Mutex<write_half>` and a reader-actor task that
/// multiplexes inbound `WorkerMsg`s by `request_id` to per-request channels.
/// `forward()` and `generate()` only hold the write mutex long enough to send
/// one framed IPC message; waiting for the response happens off-lock on the
/// per-request channel. **Multiple concurrent `forward()` / `generate()` calls
/// against the same model no longer block each other**, as long as the worker
/// itself can make progress on them.
///
/// Compute-side serialization still applies: the worker subprocess handles one
/// forward call at a time internally (until Item 7 BatchGenerate lands proper
/// slot batching). So two concurrent requests share the worker in a "fair"
/// interleaved fashion — each request's message arrives at the worker in the
/// order it was sent, and responses flow back in whatever order the worker
/// emits them.
/// Why a model is running on the CPU.
///
/// Kept as three distinct values rather than a bool because they call for three
/// different actions from the user: change a setting, free some memory, or
/// accept that this machine's GPU is too old. Downstream they were all just
/// `--gpu-layers 0`, which is how a correct setting and an override came to look
/// identical — see [`ModelProcessPool::cpu_reason`].
/// Does a model being spawned count against the system-RAM budget (and get a
/// KV budget handed to its worker)? Yes when it is being sent to the CPU, when
/// no GPU was detected on this node, or when this build cannot drive one.
/// Pure, so the truth table is pinned by `ram_is_charged_wherever_the_model_
/// can_only_land_in_ram`.
pub(crate) fn charges_ram(going_to_cpu: bool, gpu_detected: bool, build_has_cuda: bool) -> bool {
    going_to_cpu || !gpu_detected || !build_has_cuda
}

/// Add `mb` to a model's recorded charge.
///
/// Segments of one model share a worker, so a second segment ADDS to what the
/// first reserved rather than replacing it. Replacing was correct only while a
/// spawn charged the WHOLE model whatever it was about to load; now that it
/// prices the segment, an overwrite would silently forget the first one.
fn add_reserved(map: &dashmap::DashMap<ModelId, u64>, model_id: &ModelId, mb: u64) {
    *map.entry(model_id.clone()).or_insert(0) += mb;
}

/// The layer count, first-segment flag and weight bytes a footprint estimate
/// should describe for `segment` of a `block_count`-layer model whose shards
/// total `shard_bytes`.
///
/// `None` asks about the whole model. Weights are charged in proportion to the
/// layers mapped — the same approximation `auto_manage::estimate_segment_vram_mb`
/// and the scheduler's `bytes_per_layer` make, so the planner and the loader
/// price a segment identically.
pub(crate) fn segment_shape(
    block_count: u64,
    shard_bytes: u64,
    segment: Option<(u32, u32)>,
) -> (u64, bool, u64) {
    let Some((start, end)) = segment else {
        return (block_count, true, shard_bytes);
    };
    let layers = u64::from(end.saturating_sub(start)).clamp(1, block_count.max(1));
    let weights = if block_count == 0 || layers >= block_count {
        shard_bytes
    } else {
        shard_bytes / block_count * layers
    };
    (layers, start == 0, weights)
}

/// `(fixed_mb, per_layer_mb)` from the cost of a one-layer and a two-layer
/// segment. The estimate is affine in the layer count — weights and KV scale
/// with it, the process overhead does not — so two points determine it, and
/// taking them from the estimator itself means nothing here restates its
/// arithmetic.
pub(crate) fn cost_curve_from(one_layer_mb: u64, two_layer_mb: u64) -> (u64, u64) {
    let per_layer = two_layer_mb.saturating_sub(one_layer_mb);
    (one_layer_mb.saturating_sub(per_layer), per_layer)
}

/// How many layers fit in `free_mb` once the fixed terms are paid.
///
/// `None` when there is no per-layer cost to divide by — unknowable, never
/// "no room" (see `max_hostable_layers` for why that distinction is load
/// bearing).
pub(crate) fn layers_that_fit(free_mb: u64, fixed_mb: u64, per_layer_mb: u64) -> Option<u32> {
    if per_layer_mb == 0 {
        return None;
    }
    Some((free_mb.saturating_sub(fixed_mb) / per_layer_mb) as u32)
}

/// One resident model-worker subprocess as the status surfaces report it.
///
/// Read through [`ModelProcessPool::worker_summaries`]; every field is a fact
/// the pool already keeps, so the report cannot disagree with the decisions
/// the pool makes from the same numbers.
#[derive(Debug, Clone, serde::Serialize)]
pub struct WorkerSummary {
    pub model: String,
    /// The subprocess id, when the child is still ours to name.
    pub pid: Option<u32>,
    /// `"graphics card"` or `"processor"`.
    pub device: &'static str,
    /// Why it runs on the processor, when it does (`CpuReason::as_str`).
    pub cpu_reason: Option<&'static str>,
    /// Requests this worker is computing right now.
    pub in_flight: usize,
    /// Seconds since a request was last registered against it.
    pub idle_secs: u64,
    /// Seconds since it was spawned.
    pub age_secs: u64,
    /// The reader actor saw its socket close: the process is gone or going.
    pub dead: bool,
    /// What admission priced the model at, in MB; 0 when unpriced.
    pub gpu_estimate_mb: u64,
    /// For a hybrid split, how many of `layers_total` run on the card; `None`
    /// for a worker wholly on either device.
    pub gpu_layers_on_card: Option<u32>,
    /// Layers in the model this worker runs, known only for a hybrid split.
    pub layers_total: Option<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CpuReason {
    /// `inference.gpu_layers = 0` — the user asked for CPU.
    Configured,
    /// This build's CUDA kernels need a newer card than the one present.
    GpuTooOld,
    /// The model's estimated footprint exceeds the free VRAM budget. Unlike the
    /// other two this clears itself once memory frees up.
    NotEnoughVram,
}

impl CpuReason {
    /// Stable machine-readable tag. Used in logs and in the admin API, so keep
    /// these values stable — something will grep for them.
    pub fn as_str(self) -> &'static str {
        match self {
            CpuReason::Configured => "configured_cpu_only",
            CpuReason::GpuTooOld => "gpu_too_old_for_this_build",
            CpuReason::NotEnoughVram => "not_enough_vram",
        }
    }
}

pub struct ModelProcessPool {
    workers: DashMap<ModelId, Arc<WorkerHandle>>,
    /// Serializes worker spawning to prevent TOCTOU races where two concurrent
    /// callers both miss the DashMap lookup and each spawn a subprocess.
    spawn_lock: Mutex<()>,
    /// Crash-loop backoff: per-model `(last_failure_at, consecutive_failures)`.
    /// A spawn that fails (or crashes immediately) bumps the counter; further
    /// `get_or_spawn` calls within the cooldown window short-circuit with
    /// `ModelNotAvailable` instead of repeatedly burning the 30 s
    /// `WORKER_CONNECT_TIMEOUT_SECS` on every arriving request. First
    /// successful spawn clears the entry.
    spawn_failures: DashMap<ModelId, (std::time::Instant, u32)>,
    data_dir: PathBuf,
    /// Active shard windows: which shards each model worker should load.
    /// If absent, the worker loads all on-disk shards (default behavior).
    active_shard_windows: DashMap<ModelId, Vec<u32>>,
    /// Device placement passed to future-spawned workers, mirroring
    /// `InferenceConfig::gpu_layers`: `-1` auto, `0` CPU only, `>0` GPU.
    gpu_layers: std::sync::atomic::AtomicI32,
    // (see `CpuReason` for why the three CPU causes are kept distinct)
    /// Models forced onto the CPU because a worker died of a GPU OOM while
    /// serving them. Without this, the respawned worker makes the identical
    /// allocation and dies the same way, and the user sees an unbroken run of
    /// 500s with no path out.
    ///
    /// **The pin is NOT for the life of the daemon**, and this doc used to say
    /// it was. `unload_model` clears every pin whenever unloading actually
    /// freed graphics memory, because the condition that caused the OOM has
    /// then changed — see the `freed_gpu_memory` branch there, and
    /// `worker_should_return_to_gpu` for the resident-worker half. Reasoning
    /// about this field as permanent is how a model that could move back to
    /// the card gets left on the processor (gotcha #401).
    cpu_pinned_models: dashmap::DashSet<ModelId>,
    /// GPU memory budget in MB, 0 = unset (no admission control).
    ///
    /// Mirror of the effective `resources` budget, set at startup like the other
    /// knobs here. The pool cannot reach `SharedState`, and this has to be
    /// readable from inside `spawn_lock` where the admission decision happens.
    vram_budget_mb: std::sync::atomic::AtomicU64,
    /// CPU threads each worker's rayon pool may use. 0 until set at startup,
    /// which the spawn path reads as "not configured" and leaves alone.
    cpu_threads: std::sync::atomic::AtomicUsize,
    /// GPU memory charged to each live worker, in MB.
    ///
    /// Admission needs to know what is already committed, and it cannot ask the
    /// device: `nvidia-smi` reports the whole machine (so it counts other
    /// programs, and on WSL per-process figures are unavailable), and a worker
    /// that has been admitted but has not finished loading has allocated
    /// nothing yet while still owing its footprint. Charging at admission and
    /// crediting on unload is the only view that is correct at the moment the
    /// decision is made.
    vram_reserved_mb: dashmap::DashMap<ModelId, u64>,
    /// System RAM budget in MB, 0 = unset (no admission control).
    ///
    /// The CPU-side sibling of `vram_budget_mb`, and it matters most exactly
    /// where that one gives up: refusing a model the GPU cannot hold loads it
    /// on the CPU instead, so the busier the GPU admission control is, the more
    /// weight lands in system RAM. With no ceiling there, the fallback that
    /// keeps a node answering is also what drives a small machine into swap.
    ram_budget_mb: std::sync::atomic::AtomicU64,
    /// How `ram_budget_mb` was arrived at, in one sentence, for the refusal
    /// message — the number alone surprised a user whose `max_ram_mb` was
    /// larger (external report, 2026-08-21).
    ram_budget_note: std::sync::Mutex<String>,
    /// System RAM charged to each live CPU worker, in MB. Same
    /// charge-at-admission / credit-on-unload discipline as
    /// `vram_reserved_mb`, and for the same reason: a worker owes its
    /// footprint from the moment it is admitted, long before it has loaded
    /// anything to measure.
    ram_reserved_mb: dashmap::DashMap<ModelId, u64>,
    /// Layer ranges of each model that have been priced and charged.
    ///
    /// One worker serves a model and its own `models` map is keyed by layer
    /// range, so it can come to hold several segments at once — a privacy
    /// boomerang gives the local worker both ends. Charging the whole model at
    /// spawn made that safe by over-pricing; charging the segment makes it
    /// exact, and this is what keeps the second segment from arriving free.
    charged_segments: dashmap::DashMap<ModelId, Vec<(u32, u32)>>,
    /// Live RAM budget source, installed once by the daemon
    /// (`set_ram_budget_provider`): returns the cap from the CURRENT config and
    /// the anti-swap headroom from memory free NOW. Without one (tests), the
    /// stored `ram_budget_mb` cap is used alone.
    #[allow(clippy::type_complexity)]
    ram_budget_provider: std::sync::OnceLock<
        Box<dyn Fn() -> Option<crate::model::auto_manage::vram::RamBudget> + Send + Sync>,
    >,
    /// Whether this node detected a GPU at startup (`SharedState::gpu_info`).
    /// Defaults to `true` so nothing changes for a pool nobody told; the daemon
    /// sets it once. See `charges_ram`.
    gpu_detected: std::sync::atomic::AtomicBool,
    /// KV budget handed to each CPU worker at spawn (`--kv-budget-bytes`):
    /// the model's typical-context KV charge plus the RAM budget still
    /// uncommitted at admission. Set by `record_cpu_kv_budget`, cleared with
    /// the reservation.
    cpu_kv_budget_bytes: dashmap::DashMap<ModelId, u64>,
    /// Activity event sender for dashboard notifications.
    activity_tx:
        std::sync::OnceLock<tokio::sync::broadcast::Sender<crate::daemon::state::ActivityEvent>>,
    /// KV-cache session TTL passed to worker subprocesses (from config).
    kv_cache_ttl_secs: std::sync::atomic::AtomicU64,
    /// Prefix-cache config snapshot applied to future-spawned workers.
    /// Reading/writing is Relaxed — workers are spawned rarely enough that
    /// we don't care about cross-thread immediacy.
    prefix_cache_enabled: std::sync::atomic::AtomicBool,
    prefix_cache_max_entries: std::sync::atomic::AtomicU32,
    prefix_cache_max_prompt_tokens: std::sync::atomic::AtomicU32,
    prefix_cache_max_mb: std::sync::atomic::AtomicU32,
    prefix_cache_block_tokens: std::sync::atomic::AtomicU32,
    prefix_cache_min_tokens: std::sync::atomic::AtomicU32,
    /// SWIFT (arxiv 2410.06916) self-speculative decoding settings applied
    /// to future-spawned workers.
    swift_self_speculative: std::sync::atomic::AtomicBool,
    swift_calibration_tokens: std::sync::atomic::AtomicU32,
    swift_gamma: std::sync::atomic::AtomicU32,
    /// Stored as parts-per-thousand to fit into AtomicU32 (e.g. 0.45 → 450).
    swift_skip_ratio_milli: std::sync::atomic::AtomicU32,
    /// Force `standard_attention` everywhere (baseline + speculative paths).
    /// Required for SWIFT correctness; optional for benchmarking baselines.
    force_standard_attn: std::sync::atomic::AtomicBool,
    /// 0 means use the GGUF context_length verbatim. >0 caps it for KV-cache
    /// pre-allocation, so 128K-context models fit on small VRAM.
    max_seq_len_override: std::sync::atomic::AtomicU32,
    /// Quantize intermediate-segment hidden state activations to Q8_0 before
    /// returning them to the daemon (which forwards to the next pipeline peer).
    /// Receivers auto-dispatch on the dtype tag. See Item 13 in
    /// `docs/plans/archive/distributed_inference_speedup.md`.
    activation_compression: std::sync::atomic::AtomicBool,
    /// Continuous batching: when on, `forward()` routes through an
    /// auto-coalescing scheduler that collects concurrent arrivals into a
    /// time-windowed batch and dispatches via `forward_batch` (one
    /// `DaemonMsg::BatchForward` IPC message, one fused `SplitModel::forward_batch`
    /// call on GPU, falls through to sequential on CPU). Off → `forward()`
    /// bypasses the scheduler and sends a single `DaemonMsg::Forward`.
    continuous_batching: std::sync::atomic::AtomicBool,
    /// Time-window in milliseconds for the scheduler to wait for additional
    /// arrivals after the first request lands in an empty batch. Matches
    /// `InferenceConfig::batch_collection_ms`.
    batch_collection_ms: std::sync::atomic::AtomicU64,
    /// Maximum batch size. Matches `InferenceConfig::max_concurrent_decode_batch`.
    max_concurrent_decode_batch: std::sync::atomic::AtomicU32,
    /// Item 7 Phase 2: Sarathi prefill chunk size (in prompt tokens). Passed
    /// into spawned workers as `--prefill-chunk-tokens`. Matches
    /// `InferenceConfig::prefill_chunk_tokens`.
    prefill_chunk_tokens: std::sync::atomic::AtomicU32,
    /// Runtime mirror of `InferenceConfig::prefill_target_ms`.
    prefill_target_ms: std::sync::atomic::AtomicU64,
    /// Longest n-gram the local draft-free speculator will try to match.
    /// Matches `InferenceConfig::ngram_max_size`.
    ngram_max_size: std::sync::atomic::AtomicU32,
    /// How many tokens a local speculative round drafts. **Zero means the
    /// speculator is off**, which is how `InferenceConfig::ngram_lookup_enabled`
    /// reaches the worker — one value carries both the switch and the width, so
    /// a worker cannot be spawned with them disagreeing.
    ngram_pred_tokens: std::sync::atomic::AtomicU32,
    /// Item 7 Phase 4: fuse concurrent same-shape Prefilling slots into one
    /// `forward_batch` call. Passed into spawned workers as
    /// `--batched-prefill-forward`. Matches
    /// `InferenceConfig::batched_prefill_forward`. When false, Phase A always
    /// runs singleton forwards (useful for A/B benchmarks).
    batched_prefill_forward: std::sync::atomic::AtomicBool,
    /// Global batch scheduler. Created once by `start_batch_scheduler` from
    /// daemon setup (where `Arc<Self>` is available). When unset, `forward()`
    /// bypasses batching entirely regardless of the `continuous_batching` flag.
    batch_scheduler: std::sync::OnceLock<mpsc::Sender<BatchSchedulerMsg>>,
    /// Item 8 Phase 1: cross-node prefix-cache announcement sink. Set by
    /// `SharedState::new` after the daemon spawns its forwarder task; reader
    /// actors call `try_send` so a slow daemon never backpressures the worker
    /// IPC reader. When unset (e.g. unit tests constructing a bare pool),
    /// inbound `PrefixManifestUpdate` messages are dropped silently.
    prefix_manifest_tx: std::sync::OnceLock<mpsc::Sender<PrefixManifestEvent>>,
    progress_tx: std::sync::OnceLock<mpsc::Sender<ProgressEvent>>,
    /// Item 8 Phase 2b: worker-initiated fetch probes land here. Daemon
    /// drains and responds via `send_prefix_fetch_result`. Unset → drop.
    prefix_probe_tx: std::sync::OnceLock<mpsc::Sender<PrefixProbeEvent>>,
}

/// Command into the batch scheduler task.
enum BatchSchedulerMsg {
    Forward {
        fwd: crate::types::LayerForward,
        resp_tx: tokio::sync::oneshot::Sender<Result<crate::types::LayerResult, SwarmError>>,
    },
}

/// Can this `LayerForward` share a `BatchForward` message with others? Matches
/// the worker-side `batch_eligible` check: same layer_range + decode-only + no
/// vision/LoRA/spec/TP/pre-embedded/truncate. If any of these trip, we send a
/// single `DaemonMsg::Forward` instead.
fn forward_is_schedulable(f: &crate::types::LayerForward) -> bool {
    if f.sequence_num == 0 || f.index_pos == 0 {
        return false;
    }
    if f.tp_meta.is_some() {
        return false;
    }
    if f.vision_embeddings.is_some() {
        return false;
    }
    if f.adapter_id.is_some() {
        return false;
    }
    if !f.draft_tokens.is_empty() {
        return false;
    }
    if f.spec_logits_requested {
        return false;
    }
    if f.pre_embedded {
        return false;
    }
    if f.truncate_kv_to.is_some() {
        return false;
    }
    true
}

impl ModelProcessPool {
    pub fn new(data_dir: PathBuf) -> Self {
        Self {
            workers: DashMap::new(),
            spawn_lock: Mutex::new(()),
            spawn_failures: DashMap::new(),
            data_dir,
            active_shard_windows: DashMap::new(),
            gpu_layers: std::sync::atomic::AtomicI32::new(-1),
            cpu_pinned_models: dashmap::DashSet::new(),
            vram_budget_mb: std::sync::atomic::AtomicU64::new(0),
            cpu_threads: std::sync::atomic::AtomicUsize::new(0),
            vram_reserved_mb: dashmap::DashMap::new(),
            ram_budget_mb: std::sync::atomic::AtomicU64::new(0),
            ram_budget_provider: std::sync::OnceLock::new(),
            gpu_detected: std::sync::atomic::AtomicBool::new(true),
            cpu_kv_budget_bytes: dashmap::DashMap::new(),
            ram_budget_note: std::sync::Mutex::new(String::new()),
            ram_reserved_mb: dashmap::DashMap::new(),
            charged_segments: dashmap::DashMap::new(),
            activity_tx: std::sync::OnceLock::new(),
            kv_cache_ttl_secs: std::sync::atomic::AtomicU64::new(DEFAULT_KV_CACHE_TTL_SECS),
            prefix_cache_enabled: std::sync::atomic::AtomicBool::new(true),
            prefix_cache_max_entries: std::sync::atomic::AtomicU32::new(16),
            prefix_cache_max_prompt_tokens: std::sync::atomic::AtomicU32::new(8192),
            prefix_cache_max_mb: std::sync::atomic::AtomicU32::new(2048),
            prefix_cache_block_tokens: std::sync::atomic::AtomicU32::new(64),
            prefix_cache_min_tokens: std::sync::atomic::AtomicU32::new(32),
            swift_self_speculative: std::sync::atomic::AtomicBool::new(false),
            swift_calibration_tokens: std::sync::atomic::AtomicU32::new(32),
            swift_gamma: std::sync::atomic::AtomicU32::new(4),
            swift_skip_ratio_milli: std::sync::atomic::AtomicU32::new(450),
            force_standard_attn: std::sync::atomic::AtomicBool::new(false),
            max_seq_len_override: std::sync::atomic::AtomicU32::new(0),
            activation_compression: std::sync::atomic::AtomicBool::new(false),
            continuous_batching: std::sync::atomic::AtomicBool::new(false),
            batch_collection_ms: std::sync::atomic::AtomicU64::new(5),
            max_concurrent_decode_batch: std::sync::atomic::AtomicU32::new(8),
            prefill_chunk_tokens: std::sync::atomic::AtomicU32::new(128),
            ngram_max_size: std::sync::atomic::AtomicU32::new(0),
            ngram_pred_tokens: std::sync::atomic::AtomicU32::new(0),
            prefill_target_ms: std::sync::atomic::AtomicU64::new(200),
            batched_prefill_forward: std::sync::atomic::AtomicBool::new(true),
            batch_scheduler: std::sync::OnceLock::new(),
            prefix_manifest_tx: std::sync::OnceLock::new(),
            progress_tx: std::sync::OnceLock::new(),
            prefix_probe_tx: std::sync::OnceLock::new(),
        }
    }

    /// Install the prefix-manifest sink. The daemon spawns a forwarder task
    /// that owns the receiver and turns each event into a gossip broadcast +
    /// local-index update. Idempotent — a second call is a no-op.
    pub fn set_progress_tx(&self, tx: mpsc::Sender<ProgressEvent>) {
        let _ = self.progress_tx.set(tx);
    }

    pub fn set_prefix_manifest_tx(&self, tx: mpsc::Sender<PrefixManifestEvent>) {
        let _ = self.prefix_manifest_tx.set(tx);
    }

    /// Install the prefix-probe sink. Daemon owns the receiver and answers
    /// each probe via `send_prefix_fetch_result`.
    pub fn set_prefix_probe_tx(&self, tx: mpsc::Sender<PrefixProbeEvent>) {
        let _ = self.prefix_probe_tx.set(tx);
    }

    /// Item 8 Phase 2b: deliver a cross-node-fetch result back to the
    /// worker that emitted the probe. Worker correlates by `request_id`
    /// via its `pending_fetches` map. Sends `None` when the daemon
    /// couldn't resolve a hit (no index match, peer miss, BLAKE3 fail,
    /// timeout) — caller falls through to normal prefill.
    pub async fn send_prefix_fetch_result(
        &self,
        model_id: &ModelId,
        request_id: Uuid,
        matched_tokens: u32,
        payload: Option<Vec<u8>>,
    ) -> Result<(), SwarmError> {
        let handle = self.get_existing(model_id).ok_or_else(|| {
            SwarmError::Internal(format!(
                "send_prefix_fetch_result: no worker for model {model_id}"
            ))
        })?;
        if handle.dead.load(Ordering::Acquire) {
            return Err(SwarmError::ServiceUnavailable("worker dead".into()));
        }
        let mut writer = handle.writer.lock().await;
        let present = payload.is_some();
        let payload_slice: &[u8] = payload.as_deref().unwrap_or(&[]);
        send_daemon(
            &mut *writer,
            &DaemonMsg::PrefixFetchResult {
                request_id,
                matched_tokens,
                present,
            },
            payload_slice,
        )
        .await
        .map_err(|e| SwarmError::Internal(format!("prefix fetch result send: {e}")))
    }

    /// Item 8 Phase 2b: serving side. The daemon received an inbound
    /// `SwarmRequest::PrefixKvFetch` — ask the local worker to extract a
    /// matching snapshot from its `PrefixCache`. Returns `Some(bytes)` on
    /// hit, `None` on miss / worker-unreachable (caller replies
    /// `PrefixKvData { payload: None }`).
    pub async fn fetch_local_snapshot(
        &self,
        model_id: &ModelId,
        block_hash: [u8; 32],
    ) -> Option<Vec<u8>> {
        let handle = self.get_existing(model_id)?;
        if handle.dead.load(Ordering::Acquire) {
            return None;
        }
        let request_id = Uuid::new_v4();
        let (tx, mut rx) = mpsc::channel::<(WorkerMsg, Vec<u8>)>(2);
        // Snapshot export is a cheap metadata read, not a decode loop — there
        // is no meaningful compute to abandon.
        let (_guard, _) = handle.register_response(request_id, tx, false);
        {
            let mut writer = handle.writer.lock().await;
            let msg = DaemonMsg::ExportPrefixSnapshot {
                request_id,
                model_id: model_id.clone(),
                block_hash,
            };
            if let Err(e) = send_daemon(&mut *writer, &msg, &[]).await {
                tracing::debug!(error = %e, "fetch_local_snapshot: send failed");
                return None;
            }
        }
        // Wait for the worker's reply. Bounded so a stalled worker can't
        // block the manager — serving-side misses are fine, and the network
        // peer will get a `None` reply. Sized for 7B-class model snapshots:
        // 28 MB (TinyLlama) serializes in <200 ms on CPU; 73 MB (Qwen-7B
        // GQA @ 640 tokens) measured at ~500 ms. Kept under the daemon's
        // network-probe window (~2.5 s) so a stuck worker still lets the
        // network peer see a clean miss.
        match tokio::time::timeout(std::time::Duration::from_millis(2000), rx.recv()).await {
            Ok(Some((WorkerMsg::PrefixSnapshotResponse { present, .. }, payload))) => {
                if present {
                    Some(payload)
                } else {
                    None
                }
            }
            Ok(Some(_)) => None,
            Ok(None) => None,
            Err(_) => {
                tracing::debug!("fetch_local_snapshot: timed out");
                None
            }
        }
    }

    fn get_existing(&self, model_id: &ModelId) -> Option<Arc<WorkerHandle>> {
        self.workers.get(model_id).map(|r| r.clone())
    }

    /// Start the global auto-coalescing batch scheduler task. Must be called
    /// from within a Tokio runtime; no-op if no runtime is available (sync
    /// tests constructing `SharedState` directly). Safe to call more than
    /// once (second call is a no-op via `OnceLock::set`).
    pub fn start_batch_scheduler(
        self: &Arc<Self>,
        shutdown_rx: tokio::sync::watch::Receiver<bool>,
    ) {
        if tokio::runtime::Handle::try_current().is_err() {
            tracing::debug!("start_batch_scheduler called outside a tokio runtime — skipping");
            return;
        }
        let pool = self.clone();
        let (tx, rx) = mpsc::channel::<BatchSchedulerMsg>(1024);
        if self.batch_scheduler.set(tx).is_err() {
            return;
        }
        tokio::spawn(async move {
            batch_scheduler_loop(pool, rx, shutdown_rx).await;
        });
    }

    /// Toggle auto-coalescing batch scheduler. When on, concurrent `forward()`
    /// calls for the same model are collected into one `BatchForward` IPC
    /// message. CPU workers fall through to sequential automatically; GPU
    /// workers run the fused `SplitModel::forward_batch` path. See Item 3
    /// Phase 2b in `docs/plans/archive/distributed_inference_speedup.md`.
    pub fn set_continuous_batching(&self, enabled: bool) {
        self.continuous_batching
            .store(enabled, std::sync::atomic::Ordering::Relaxed);
    }

    /// Set continuous-batching scheduler parameters. Takes effect on the next
    /// scheduler task creation (existing per-model scheduler loops keep the
    /// window they were spawned with).
    pub fn set_batch_params(&self, collection_ms: u64, max_batch: u32) {
        self.batch_collection_ms
            .store(collection_ms, std::sync::atomic::Ordering::Relaxed);
        self.max_concurrent_decode_batch
            .store(max_batch.max(1), std::sync::atomic::Ordering::Relaxed);
    }

    /// Sarathi prefill chunk size for future-spawned workers. Existing
    /// workers retain whatever chunk size they were spawned with.
    /// Set the local speculator's shape. `pred_tokens == 0` disables it.
    ///
    /// Takes effect for workers spawned afterwards, like every other spawn-time
    /// option here — a running worker holds its model and cache and is not
    /// recycled to change a setting (see `settings.contribution_restart_note`).
    pub fn set_ngram_params(&self, max_size: u32, pred_tokens: u32) {
        self.ngram_max_size
            .store(max_size, std::sync::atomic::Ordering::Relaxed);
        self.ngram_pred_tokens
            .store(pred_tokens, std::sync::atomic::Ordering::Relaxed);
    }

    pub fn set_prefill_chunk_tokens(&self, chunk_tokens: u32) {
        self.prefill_chunk_tokens
            .store(chunk_tokens.max(1), std::sync::atomic::Ordering::Relaxed);
    }

    /// Per-tick prefill wall-time budget for future-spawned workers. Existing
    /// workers retain whatever budget they were spawned with.
    pub fn set_prefill_target_ms(&self, target_ms: u64) {
        self.prefill_target_ms
            .store(target_ms.max(1), std::sync::atomic::Ordering::Relaxed);
    }

    /// Item 7 Phase 4: toggle prefill-chunk fusion inside the worker's
    /// `step_decode_pool` Phase A. Takes effect on the next spawned worker.
    pub fn set_batched_prefill_forward(&self, enabled: bool) {
        self.batched_prefill_forward
            .store(enabled, std::sync::atomic::Ordering::Relaxed);
    }

    /// Toggle Q8_0 quantization of intermediate-segment hidden state activations
    /// for future-spawned workers. Existing workers retain whatever flag they
    /// were spawned with — restart the worker to apply changes.
    pub fn set_activation_compression(&self, enabled: bool) {
        self.activation_compression
            .store(enabled, std::sync::atomic::Ordering::Relaxed);
    }

    /// Set the device placement for future-spawned workers.
    /// `-1` = auto (GPU when available), `0` = CPU only, `>0` = GPU.
    pub fn set_gpu_layers(&self, gpu_layers: i32) {
        self.gpu_layers
            .store(gpu_layers, std::sync::atomic::Ordering::Relaxed);
    }

    /// The `--gpu-layers` value a worker for `model_id` should be spawned with.
    /// A model pinned to the CPU by a previous OOM overrides the config.
    /// How many layers of this model would fit on the card, when the whole
    /// thing does not.
    ///
    /// `None` means do not split — either the geometry could not be read, or
    /// nothing meaningful fits, in which case the processor takes the model as
    /// it always did. An unreadable header is "do not judge", the same
    /// treatment [`Self::admit_to_gpu`] gives the same gap.
    ///
    /// **On by default since the numbers exist.** Measured on an RTX 3070
    /// against `qwen2.5-coder-7b`, every figure taken the same way on the same
    /// machine with the same prompt: 5.0 tok/s on the processor alone, 7.07
    /// with 14 of 28 layers on the card, 12.25 with 20 — **2.4x**, and monotone
    /// in the layer count, which is the part that says the mechanism is really
    /// doing the work rather than producing a nicer number by accident.
    ///
    /// The automatic choice was then exercised end to end: against a 3000 MB
    /// budget and a 5232 MB model it chose 13 layers unprompted and the worker
    /// settled at **2613 MB**, inside the budget with room to spare. That is
    /// the check that matters, because an estimate disagreeing with reality is
    /// this codebase's recurring placement bug (gotcha #388).
    ///
    /// `SWARMLLM_HYBRID_OFFLOAD=0` turns it off — an A/B switch inside one
    /// binary, the same discipline as `SWARMLLM_DECODE_ATTN` and
    /// `SWARMLLM_FORCE_STANDARD_ATTN`, so a regression can be attributed
    /// without building two binaries.
    ///
    /// It cannot make a request slower than the alternative it replaces: the
    /// alternative is the whole model on the processor, and this moves layers
    /// off it. The cost is one hidden-state copy per forward, which is a few
    /// KB while decoding.
    /// `(layers on the card, layers in total)` for a model that does not fit
    /// whole but partly does; `None` when it fits, when nothing fits, or when
    /// hybrid placement is switched off.
    fn partial_gpu_layers(
        &self,
        model_id: &ModelId,
        segment: Option<(u32, u32)>,
        estimated_mb: u64,
    ) -> Option<(usize, usize)> {
        if std::env::var("SWARMLLM_HYBRID_OFFLOAD").as_deref() == Ok("0") {
            return None;
        }
        let budget = self
            .vram_budget_mb
            .load(std::sync::atomic::Ordering::Relaxed);
        let available_mb = budget.saturating_sub(self.vram_committed_mb());
        // Nothing to split if the model would have fitted anyway.
        if available_mb >= estimated_mb {
            return None;
        }
        let inputs = self.footprint_inputs(model_id, segment)?;
        let layers = inputs.segment_layers as usize;
        if layers == 0 {
            return None;
        }
        // KV geometry per layer, in the units `kv_bytes_per_token` takes.
        let kv_elems = (inputs.head_count_kv * inputs.head_dim) as usize;
        let n = crate::inference::split::hybrid::plan_gpu_layers(
            available_mb.saturating_mul(1024 * 1024),
            inputs.quantized_weight_bytes,
            layers,
            kv_elems,
            kv_elems,
            false,
            inputs.effective_context,
        );
        (n > 0).then_some((n, layers))
    }

    fn effective_gpu_layers(&self, model_id: &ModelId) -> i32 {
        match self.cpu_reason(model_id) {
            Some(_) => 0,
            None => self.gpu_layers.load(std::sync::atomic::Ordering::Relaxed),
        }
    }

    /// Why this model is going to the CPU, when it is — `None` means the GPU.
    ///
    /// **The three causes are indistinguishable downstream and mean completely
    /// different things.** All of them spawn `swarmllm model-worker
    /// --gpu-layers 0` and log `requested: gpu_layers = 0`: one is the user's
    /// own setting, the other two are this node overriding it. A tester on
    /// 2026-08-10 concluded from exactly those two signals that
    /// `inference.gpu_layers` was being ignored on a node that was honouring it
    /// and then refusing the model for VRAM — and checking the worker's real
    /// command line, which is the correct way to rule out a mere logging bug,
    /// could not separate them either. The information existed; nothing carried
    /// it to where the question gets asked.
    fn cpu_reason(&self, model_id: &ModelId) -> Option<CpuReason> {
        // Checked first: a pin overrides whatever the config says, and is the
        // only one of the three that clears itself when memory frees up.
        //
        // **A pin means the card FAILED for this model** (`classify_worker_
        // error`), not merely that it was full at some moment. An admission
        // refusal deliberately takes no pin: it is arithmetic against the
        // budget as it stands, and re-taking it next spawn costs one header
        // read — whereas a pin is cleared only when a GPU-holding worker
        // unloads, so it outlives its own cause. See `get_or_spawn`.
        if self.cpu_pinned_models.contains(model_id) {
            return Some(CpuReason::NotEnoughVram);
        }
        // A GPU older than this build's kernel floor cannot run a single
        // forward, so sending a worker there produces a fatal CUDA error, a
        // killed worker and a spawn-backoff loop — for every model, forever.
        // Placing it on the CPU up front costs speed and keeps the node
        // answering. `local_gpu_is_supported` is cached, so this stays a plain
        // atomic load after the first call.
        #[cfg(feature = "candle-cuda")]
        if !crate::daemon::gpu_support::local_gpu_is_supported() {
            return Some(CpuReason::GpuTooOld);
        }
        if self.gpu_layers.load(std::sync::atomic::Ordering::Relaxed) == 0 {
            return Some(CpuReason::Configured);
        }
        None
    }

    /// Why this model is on the processor, for the admin API and the dashboard —
    /// so "why is this model not on my GPU" is answerable without reading the log
    /// at the moment the decision was taken.
    ///
    /// **A running worker is a fact; [`ModelProcessPool::cpu_reason`] is a
    /// prediction about the next spawn.** They differ whenever something has
    /// changed since the worker started — an OOM pin lifted by freed memory, or
    /// `gpu_layers` edited in Settings — and the question being asked here is
    /// about the model that is running. Answering it from the prediction meant
    /// the dashboard dropped its explanation from a model still sitting on the
    /// processor waiting to move back, and would have explained a model happily
    /// running on the card as "you configured CPU-only".
    ///
    /// With no worker resident there is nothing to contradict the prediction,
    /// and it is exactly right: it is what the next spawn will do.
    pub fn cpu_placement_reason(&self, model_id: &ModelId) -> Option<&'static str> {
        if let Some(handle) = self.workers.get(model_id) {
            return handle.placed_on_cpu_because.map(CpuReason::as_str);
        }
        self.cpu_reason(model_id).map(CpuReason::as_str)
    }

    /// Is this model currently forced onto the CPU after a GPU OOM?
    pub fn is_cpu_pinned(&self, model_id: &ModelId) -> bool {
        self.cpu_pinned_models.contains(model_id)
    }

    /// Does this model occupy graphics memory — or, for one not yet loaded,
    /// will it?
    ///
    /// The single answer to both halves of "may a graphics-memory budget be
    /// enforced against this model": whether loading it consumes any, and
    /// whether unloading it releases any. Resident answers from the worker's
    /// own recorded placement; not resident predicts from
    /// [`ModelProcessPool::cpu_reason`], which is what the next spawn will do.
    ///
    /// **Why it exists.** `SharedState::ensure_split_model_entry` runs an LRU
    /// eviction against the graphics budget every time a split-model entry is
    /// created, and knew nothing about placement — so creating an entry for a
    /// segment bound for the PROCESSOR evicted and killed a model that was
    /// running happily on the card, freeing memory for something that would
    /// never touch it. Measured here 2026-08-27: an 8B holding 4685 MB was
    /// unloaded 35 s after it loaded, on the load of a 1-layer `force_cpu=true`
    /// segment, and the tester who reported the sibling defect saw exactly the
    /// same shape (a 3B evicted 35 s in, `freed_by` naming itself).
    ///
    /// It reads through `charges_ram` so the three ways a model can only ever
    /// land in system memory — sent to the processor, no card detected, a build
    /// without CUDA — give one answer rather than three.
    pub fn model_uses_gpu_memory(&self, model_id: &ModelId) -> bool {
        let going_to_cpu = match self.workers.get(model_id) {
            Some(handle) => handle.placed_on_cpu_because.is_some(),
            None => self.cpu_reason(model_id).is_some(),
        };
        !charges_ram(
            going_to_cpu,
            self.gpu_detected.load(std::sync::atomic::Ordering::Relaxed),
            cfg!(feature = "candle-cuda"),
        )
    }

    /// The configured value, before any override — so a log line can show what
    /// the user asked for next to what actually happened.
    pub fn configured_gpu_layers(&self) -> i32 {
        self.gpu_layers.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Estimate this model's GPU footprint from its own GGUF header and shards.
    ///
    /// Reads `gguf_header.bin` from the model directory; a spawn happens once per
    /// model, so the read is not on any hot path. Returns 0 when the header or
    /// geometry cannot be read, which `admit_to_gpu` treats as "do not judge" —
    /// refusing the GPU because a file was unreadable would be a worse failure
    /// than the one being prevented.
    fn estimate_gpu_footprint_mb(&self, model_id: &ModelId, segment: Option<(u32, u32)>) -> u64 {
        use crate::model::auto_manage::vram::estimate_worker_vram_mb;
        self.footprint_inputs(model_id, segment)
            .map(|i| estimate_worker_vram_mb(&i))
            .unwrap_or(0)
    }

    /// Estimate this model's system-RAM footprint from the same geometry.
    ///
    /// Returns 0 on an unreadable header, which `admit_to_cpu` treats as "do
    /// not judge" — refusing to load because a file could not be read would be
    /// a worse failure than the one being prevented, and matches how the GPU
    /// side handles the same gap.
    fn estimate_cpu_footprint_mb(&self, model_id: &ModelId, segment: Option<(u32, u32)>) -> u64 {
        use crate::model::auto_manage::vram::estimate_worker_ram_mb;
        self.footprint_inputs(model_id, segment)
            .map(|i| estimate_worker_ram_mb(&i))
            .unwrap_or(0)
    }

    /// The itemised CPU estimate and where its context came from — what the
    /// refusal message is built from.
    fn cpu_footprint_detail(
        &self,
        model_id: &ModelId,
        segment: Option<(u32, u32)>,
    ) -> Option<(
        crate::model::auto_manage::vram::ResidentFootprint,
        u64,
        crate::model::auto_manage::vram::ContextSource,
    )> {
        use crate::model::auto_manage::vram::{cpu_footprint, ContextSource};
        let inputs = self.footprint_inputs(model_id, segment)?;
        let source = if self
            .max_seq_len_override
            .load(std::sync::atomic::Ordering::Relaxed)
            > 0
        {
            ContextSource::Override
        } else {
            ContextSource::DeclaredOrDefault
        };
        Some((cpu_footprint(&inputs), inputs.effective_context, source))
    }

    /// Read a model's real geometry from its GGUF header and on-disk shards.
    ///
    /// Shared by both footprint estimators so the GPU and CPU figures can never
    /// disagree about the model's shape — only about the per-process overhead
    /// that is genuinely device-specific. Returns `None` when the header or
    /// geometry cannot be read.
    /// `segment` is the layer range a spawn is about to load; `None` asks
    /// about the WHOLE model, which is the right question for "could this node
    /// host it at all" (the dashboard, `would_fit_on_gpu`) and the wrong one
    /// for a spawn.
    fn footprint_inputs(
        &self,
        model_id: &ModelId,
        segment: Option<(u32, u32)>,
    ) -> Option<crate::model::auto_manage::vram::VramFootprintInputs> {
        use crate::model::auto_manage::vram::VramFootprintInputs;
        let model_dir = crate::model::shard::model_dir(&self.data_dir, &model_id.0);
        let header = model_dir.join(crate::model::shard::HEADER_FILENAME);
        // ONE parse of the header, not two. This used to read the same file a
        // second time through `GgufTokenizerMeta::from_gguf_file` — which
        // materialises the entire vocabulary and merge list as owned `String`s
        // — purely to reach `vocab.len()` as a fallback below. The token count
        // is already in `ct`, and counting the array borrows it.
        let ct = crate::inference::split::read_gguf_header(&header).ok()?;
        let tensor_meta = crate::inference::split::GgufTensorMeta::from_content(&ct).ok()?;
        let arch = crate::inference::split::gguf_arch_str(&ct);
        let md_u32 = |suffix: &str| -> Option<u64> {
            ct.metadata
                .get(&format!("{arch}.{suffix}"))
                .and_then(|v| v.to_u32().ok())
                .map(u64::from)
        };
        let declared_ctx = md_u32("context_length").unwrap_or(4096);
        // The KV cache is sized to the EFFECTIVE context, which is the override
        // when set and otherwise the shipped default cap — not the GGUF value.
        // The daemon holds the override in its own atomic (the worker's global
        // is set at spawn), so apply the shared rule against that value rather
        // than re-deriving the arithmetic here.
        let override_ctx = self
            .max_seq_len_override
            .load(std::sync::atomic::Ordering::Relaxed) as usize;
        let effective_ctx =
            crate::inference::split::effective_context_length_with(declared_ctx as usize, {
                if override_ctx > 0 {
                    Some(override_ctx)
                } else {
                    None
                }
            }) as u64;
        // Counting the token array reproduces what `GgufTokenizerMeta` reported
        // here — it collects the same entries, discarding any that are not
        // strings — without allocating one of them.
        let vocab = md_u32("vocab_size").unwrap_or_else(|| {
            ct.metadata
                .get("tokenizer.ggml.tokens")
                .and_then(|v| v.to_vec().ok())
                .map(|arr| arr.iter().filter(|v| v.to_string().is_ok()).count() as u64)
                .unwrap_or(0)
        });

        // Only the shards actually on disk will be mapped.
        let shard_bytes: u64 = std::fs::read_dir(&model_dir)
            .map(|rd| {
                rd.filter_map(Result::ok)
                    .filter(|e| {
                        e.file_name()
                            .to_str()
                            .is_some_and(|n| n.starts_with("shard_") && n.ends_with(".bin"))
                    })
                    .filter_map(|e| e.metadata().ok().map(|m| m.len()))
                    .sum()
            })
            .unwrap_or(0);

        // Is this an UNQUANTIZED checkpoint? Read it off the largest tensor,
        // which on any architecture is one of the big linear weights: in a
        // quantized GGUF that is a Q* block type, and in an unquantized one it
        // is F16/BF16/F32. Norm and bias tensors are f32 even in a quantized
        // file, so "does any f32 tensor exist" would answer the wrong question.
        let unquantized_bytes_per_element = ct
            .tensor_infos
            .values()
            .max_by_key(|t| t.shape.elem_count())
            .and_then(|t| match t.ggml_dtype {
                candle_core::quantized::GgmlDType::F16
                | candle_core::quantized::GgmlDType::BF16 => Some(2),
                candle_core::quantized::GgmlDType::F32 => Some(4),
                _ => None,
            });

        // Shape-and-dtype half of "can the loader read embedding rows on
        // demand"; each estimator applies the device half itself.
        let embedding_gatherable = ct
            .tensor_infos
            .get("token_embd.weight")
            .is_some_and(|info| {
                crate::inference::split::table_supports_row_gather(
                    info.ggml_dtype,
                    info.shape.dims(),
                )
            });

        // What this worker will ACTUALLY map. `VramFootprintInputs` has always
        // documented `segment_layers` as "Layers in THIS segment, not the whole
        // model" and `quantized_weight_bytes` as "the shard bytes this worker
        // will map" — and its only caller passed the whole model for both,
        // whatever it was about to load. On a node holding every shard, which
        // is the node most likely to be given a fraction of a big model, that
        // priced a 36-of-48-layer segment as all 48, and a privacy boomerang's
        // two end layers as the entire model. Reported from the field: a 16 GB
        // Mac mini refused every part of a 14B it was holding, retried, and
        // produced the identical plan (gotcha #452).
        // Proportional weights slightly under-charge a first segment (shard 0
        // also carries the embedding table) and the estimator adds that table
        // back whenever `is_first`.
        let (segment_layers, is_first, segment_weight_bytes) =
            segment_shape(tensor_meta.block_count as u64, shard_bytes, segment);

        Some(VramFootprintInputs {
            quantized_weight_bytes: segment_weight_bytes,
            unquantized_bytes_per_element,
            embedding_gatherable,
            vocab_size: vocab,
            embedding_length: tensor_meta.embedding_length as u64,
            segment_layers,
            head_count_kv: tensor_meta.head_count_kv as u64,
            head_dim: tensor_meta.head_dim as u64,
            rope_dim: tensor_meta.rope_dim as u64,
            effective_context: effective_ctx,
            is_first,
        })
    }

    /// Set the GPU memory budget used for admission. 0 disables the check.
    pub fn set_vram_budget_mb(&self, budget_mb: u64) {
        self.vram_budget_mb
            .store(budget_mb, std::sync::atomic::Ordering::Relaxed);
    }

    /// Set the CPU thread count handed to each worker's rayon pool.
    ///
    /// Resolved from `resources.max_cpu_threads` / `node.contribution` at
    /// startup, like the other mirrors here — the pool cannot reach
    /// `SharedState`, and this has to be readable from inside the spawn path.
    pub fn set_cpu_threads(&self, threads: usize) {
        self.cpu_threads
            .store(threads.max(1), std::sync::atomic::Ordering::Relaxed);
    }

    /// GPU memory already committed to live workers, in MB.
    pub(crate) fn vram_committed_mb(&self) -> u64 {
        self.vram_reserved_mb.iter().map(|e| *e.value()).sum()
    }

    /// Does a model of this size fit, given the budget and what is already
    /// committed?
    ///
    /// The one arithmetic rule, shared by [`ModelProcessPool::would_fit_on_gpu`]
    /// and [`ModelProcessPool::gpu_estimate_and_fit`]. They differ only in how
    /// hard they work to avoid pricing the model at all; they must never differ
    /// in the answer, and a second copy of this comparison is how they would.
    /// `None` means unknowable — no budget set, or no local geometry to read —
    /// and never "no".
    fn fits_in_budget(&self, estimated_mb: u64, budget_mb: u64) -> Option<bool> {
        if budget_mb == 0 || estimated_mb == 0 {
            return None;
        }
        Some(self.vram_committed_mb().saturating_add(estimated_mb) <= budget_mb)
    }

    /// Both halves of "how big is this model, and does it fit" from ONE reading
    /// of its geometry.
    ///
    /// For a caller that wants both — the admin model listing does, for every
    /// model, on every request — asking [`ModelProcessPool::estimated_gpu_mb`]
    /// and [`ModelProcessPool::would_fit_on_gpu`] separately reads
    /// `gguf_header.bin` and scans the model directory TWICE for one answer
    /// each. See `inference::split::read_gguf_header` for what that costs.
    ///
    /// The fit verdict is the same rule `would_fit_on_gpu` applies, in the same
    /// order, and its comment there is the explanation. The one difference is
    /// deliberate: a model already resident ON the card still gets priced here,
    /// because the caller asked for the estimate too — so this does exactly the
    /// work a caller wanting both would have done anyway, never more.
    ///
    /// Single-answer callers keep the single-answer methods; nothing is made
    /// slower to make this faster.
    pub fn gpu_estimate_and_fit(&self, model_id: &ModelId) -> (Option<u64>, Option<bool>) {
        let resident_on_gpu = self
            .workers
            .get(model_id)
            .map(|h| h.placed_on_cpu_because.is_none());
        let budget = self
            .vram_budget_mb
            .load(std::sync::atomic::Ordering::Relaxed);
        let estimated = self.estimate_gpu_footprint_mb(model_id, None);
        let fits = if resident_on_gpu == Some(true) {
            Some(true)
        } else {
            self.fits_in_budget(estimated, budget)
        };
        ((estimated != 0).then_some(estimated), fits)
    }

    /// Would this model fit on the GPU right now, without charging anything?
    ///
    /// The read-only half of [`Self::admit_to_gpu`], for callers that need to
    /// *decide* rather than *load* — the scheduler asking whether this node can
    /// serve a request well before it commits to serving it alone.
    ///
    /// Returns:
    /// - `Some(true)`  — it fits, or a worker for it is already live and paid for
    /// - `Some(false)` — it would be refused and fall back to the CPU
    /// - `None`        — no budget configured, or the geometry could not be read,
    ///   which must NOT be read as "no". Refusing to route on an unreadable file
    ///   would be a worse failure than the one being avoided, and matches how
    ///   `admit_to_gpu` treats the same gap.
    ///
    /// Deliberately shares `estimate_gpu_footprint_mb` and `vram_committed_mb`
    /// with the admission gate, so the scheduler's view and the loader's view
    /// cannot drift apart and disagree about whether a request could have run.
    pub fn would_fit_on_gpu(&self, model_id: &ModelId) -> Option<bool> {
        // Already resident ON THE GPU means already charged: running it costs
        // no new memory, whatever the budget currently says.
        //
        // **The residency check alone is not enough**, and answering it that
        // way made this function contradict itself. A worker that was refused
        // the GPU still lives in `workers` — it is running on the CPU — and
        // holds no VRAM at all, so "already charged" is false for it. Observed
        // live 2026-08-18: a node with a 400 MB budget reported `fits_on_gpu:
        // true` for a 3138 MB model in the same breath as
        // `cpu_placement_reason: not_enough_vram`, and a request that should
        // have gone to a peer with room ran on that node's CPU instead.
        //
        // A CPU-resident worker therefore falls through to the estimate below,
        // which is the right question for it: the VRAM is genuinely free.
        //
        // The test is the WORKER's own placement, not `cpu_reason`. That asks
        // where a spawn happening now would go, and its pin is cleared while
        // the processor worker it produced is still running — so through that
        // proxy a model waiting to move back to the card would report "already
        // charged" while holding no graphics memory at all, which is the exact
        // contradiction described above.
        let resident_on_gpu = self
            .workers
            .get(model_id)
            .map(|h| h.placed_on_cpu_because.is_none());
        if resident_on_gpu == Some(true) {
            return Some(true);
        }
        let budget = self
            .vram_budget_mb
            .load(std::sync::atomic::Ordering::Relaxed);
        if budget == 0 {
            return None;
        }
        // The early returns above are about avoiding WORK — pricing a model
        // means reading its header off disk. The verdict itself is shared, so
        // this and `gpu_estimate_and_fit` cannot answer differently.
        self.fits_in_budget(self.estimate_gpu_footprint_mb(model_id, None), budget)
    }

    /// This model's real GPU footprint in MB, or `None` when its geometry
    /// cannot be read (no local shards, unreadable header).
    ///
    /// The figure the LOADER decides with. Exposed because the admin API used to
    /// report `estimate_model_vram_mb` — `file_size * 1.15`, whose own doc says
    /// it is 56% low on phi-3.5-mini-q4 and "useless as an admission decision" —
    /// next to `cpu_placement_reason: not_enough_vram`, so the dashboard showed
    /// a model comfortably fitting a card the daemon had just refused it on.
    pub fn estimated_gpu_mb(&self, model_id: &ModelId) -> Option<u64> {
        match self.estimate_gpu_footprint_mb(model_id, None) {
            0 => None,
            mb => Some(mb),
        }
    }

    /// Is this model going to run on our CPU *because it does not fit our GPU*,
    /// on a node whose GPU otherwise works?
    ///
    /// The trigger for handing a whole model to a peer instead
    /// (`scheduler::delegation_target`), and deliberately narrower than "is
    /// this on the CPU". Two of the three ways a model lands on the CPU are NOT
    /// degradations and must not cause work to be sent away:
    ///
    /// - `inference.gpu_layers = 0` is the user saying they want the CPU. A
    ///   node cannot honour that by quietly using someone else's GPU.
    /// - A GPU below this build's kernel floor means EVERY model runs on the
    ///   CPU here. That is how the node works, not a fault of this model, and
    ///   treating it as degradation would turn a serving node into a proxy for
    ///   all of its traffic — a much larger change than this, and one the owner
    ///   has not asked for.
    ///
    /// What is left is the case reported on 2026-08-17: a working GPU that this
    /// particular model does not fit in.
    /// Would a request for `model_id` run on this node's PROCESSOR? True for
    /// a node with no usable card at all, one told to use its processor, a
    /// build without CUDA, and a card the model does not fit.
    ///
    /// The precondition for whole-model delegation (`scheduler::delegation_
    /// target`) since 2026-09-02. It used to be `is_cpu_bound_for_lack_of_vram`
    /// alone — "only a node with a working card the model does not fit is
    /// degraded" — which read a node with NO card as working normally. A
    /// tester's processor-only node that had just acquired every shard of a
    /// model then ran it locally at processor speed with two GPU nodes idle on
    /// the same pool, every request sitting for minutes (gotcha #442). Whether
    /// the node is "degraded" is not the question; where the request would
    /// run is, and a peer that would run it several times faster is worth
    /// asking either way. The peer-side gates (whole coverage, direct
    /// reachability, trust, room or a wide speed margin) are unchanged.
    pub fn serves_on_cpu(&self, model_id: &ModelId) -> bool {
        if !cfg!(feature = "candle-cuda") {
            return true;
        }
        if !self.gpu_detected.load(std::sync::atomic::Ordering::Relaxed) {
            return true;
        }
        if self.gpu_layers.load(std::sync::atomic::Ordering::Relaxed) == 0 {
            return true;
        }
        #[cfg(feature = "candle-cuda")]
        if !crate::daemon::gpu_support::local_gpu_is_supported() {
            return true;
        }
        self.is_cpu_bound_for_lack_of_vram(model_id)
    }

    pub fn is_cpu_bound_for_lack_of_vram(&self, model_id: &ModelId) -> bool {
        if self.gpu_layers.load(std::sync::atomic::Ordering::Relaxed) == 0 {
            return false;
        }
        #[cfg(feature = "candle-cuda")]
        if !crate::daemon::gpu_support::local_gpu_is_supported() {
            return false;
        }
        matches!(self.would_fit_on_gpu(model_id), Some(false))
    }

    /// The configured GPU memory budget in MB, or `None` when unset.
    ///
    /// Admission compares against THIS, not against the card's total VRAM — so
    /// anything reporting "will it fit" has to use the same number or it
    /// contradicts the daemon.
    pub fn vram_budget_mb(&self) -> Option<u64> {
        match self
            .vram_budget_mb
            .load(std::sync::atomic::Ordering::Relaxed)
        {
            0 => None,
            mb => Some(mb),
        }
    }

    /// Decide whether `model_id` may be loaded onto the GPU, and charge it if so.
    ///
    /// Called inside `spawn_lock`, which already serializes spawns, so the
    /// read-decide-charge sequence cannot interleave with another admission.
    /// That matters because a worker is admitted well before it allocates: the
    /// process starts here, and the model is loaded lazily on its first message.
    /// Two concurrent spawns that both consulted the *device* would each see
    /// plenty free and both proceed.
    ///
    /// Returning `false` means "spawn on the CPU instead" — deliberately slow
    /// rather than a `CUDA_ERROR_OUT_OF_MEMORY` that kills the worker and (until
    /// the pin became recoverable) cost the model its GPU for the whole run.
    fn admit_to_gpu(&self, model_id: &ModelId, estimated_mb: u64) -> bool {
        let budget = self
            .vram_budget_mb
            .load(std::sync::atomic::Ordering::Relaxed);
        if budget == 0 || estimated_mb == 0 {
            // No budget configured, or nothing to weigh: preserve the previous
            // behaviour rather than inventing a limit.
            add_reserved(&self.vram_reserved_mb, model_id, estimated_mb);
            return true;
        }
        let committed = self.vram_committed_mb();
        if committed.saturating_add(estimated_mb) <= budget {
            // Record what we believed BEFORE the worker allocates anything.
            //
            // Admission only ever logged when it refused, so a model that was
            // admitted and then died with CUDA_ERROR_OUT_OF_MEMORY left no trace
            // of what this gate had expected it to cost. That is precisely the
            // case that needs explaining, and without this line it cannot be
            // told apart from the outside: an estimate that was too low, a
            // budget already spent by a model still being evicted, and a
            // genuinely unbounded allocation all look identical in the logs.
            // Compare `estimated_mb` here against the worker's own
            // `vram_after_load_mb` — a large gap is an under-estimate, a close
            // match with an OOM anyway points at the eviction timing instead.
            tracing::info!(
                model = %model_id,
                estimated_mb,
                committed_mb = committed,
                budget_mb = budget,
                headroom_mb = budget.saturating_sub(committed.saturating_add(estimated_mb)),
                "DIAG: admitting model to GPU"
            );
            add_reserved(&self.vram_reserved_mb, model_id, estimated_mb);
            return true;
        }
        // Deliberately DEBUG, and deliberately silent about what happens next.
        // This used to be a WARN asserting "loading it on the CPU instead" — and
        // once `free_vram_for_admission` was added that became a statement the
        // very next line could falsify. Verified on this machine 2026-08-25: the
        // line was logged, an idle model was reclaimed 0.3 ms later, and the
        // model loaded on the GPU. An operator reading the log would have
        // concluded the opposite of what happened. A refusal here is one step in
        // a decision, not the decision; the outcome is announced by the caller,
        // where it is known.
        tracing::debug!(
            model = %model_id,
            estimated_mb,
            committed_mb = committed,
            budget_mb = budget,
            "DIAG: GPU admission refused — not enough budget at this moment"
        );
        false
    }

    /// Wait for a killed worker to actually be reaped, up to `limit`.
    ///
    /// Returns `true` if the process is confirmed gone. `false` means the wait
    /// expired — the caller must decide, and today it frees the budget anyway
    /// rather than strand the device forever on one stuck process.
    async fn await_worker_exit(exited: &Arc<AtomicBool>, limit: std::time::Duration) -> bool {
        let deadline = std::time::Instant::now() + limit;
        loop {
            if exited.load(Ordering::Acquire) {
                return true;
            }
            if std::time::Instant::now() >= deadline {
                return false;
            }
            tokio::time::sleep(WORKER_EXIT_POLL).await;
        }
    }

    /// Wait for a retiring worker's in-flight requests to finish.
    ///
    /// Returns the number still outstanding when it gave up — `0` means the
    /// worker drained. Polls rather than signals because `responses` is a
    /// `DashMap` shared with the reader actor, and the common case exits on the
    /// first check without sleeping at all.
    ///
    /// Takes the map rather than the handle so it is reachable from a test: a
    /// `WorkerHandle` owns a child process and a socket, and the thing being
    /// verified here is only the waiting.
    async fn await_responses_drained(responses: &ResponseMap, limit: std::time::Duration) -> usize {
        let deadline = std::time::Instant::now() + limit;
        loop {
            let outstanding = responses.len();
            if outstanding == 0 {
                return 0;
            }
            if std::time::Instant::now() >= deadline {
                return outstanding;
            }
            tokio::time::sleep(WORKER_EXIT_POLL).await;
        }
    }

    /// Reclaim graphics memory from models nothing is using, so `model_id` can
    /// have the card instead of being demoted to the CPU.
    ///
    /// Called when [`Self::admit_to_gpu`] refuses, BEFORE the refusal reaches
    /// the caller — the same shape as [`Self::free_ram_for_admission`], which
    /// has done exactly this for the RAM budget since v0.3.111. The GPU side
    /// never had it, so a model that happened to load first kept the card for
    /// as long as it stayed resident and everything asked for afterwards ran on
    /// the processor.
    ///
    /// Observed on this development machine 2026-08-25: an 8B loaded at 14:17
    /// and untouched since held a 6033 MB reservation against a 6722 MB budget;
    /// a 3B requested fifteen minutes later needed 3138 MB, was refused the 689
    /// MB that remained, and answered at 7 tok/s on the CPU while the card sat
    /// idle. The background idle-unload could not help: it runs on a timer, and
    /// its regional-demand clause deliberately keeps a model the swarm wants
    /// warm for up to an hour. Neither can act on the request arriving *now*.
    ///
    /// Three properties this must keep.
    ///
    /// **Plan before destroying.** Unloading models and *still* not fitting
    /// costs the user a reload for nothing, so the whole plan is costed first
    /// and abandoned whole if it cannot succeed. The RAM sibling can unload
    /// opportunistically because its alternative is failing the request
    /// outright; here the alternative is a slower answer, so a wasted eviction
    /// is a real regression rather than a lesser evil.
    ///
    /// **Take the card only from a model that has stopped being used.** A
    /// worker with a request in flight is never a candidate, and neither is one
    /// used within `vram_make_room_min_idle_secs()`. Without that, two models
    /// alternating faster than they load would evict each other on every
    /// request and each answer would pay a cold start. With it, that case
    /// simply keeps today's behaviour — one of them runs on the CPU — while a
    /// genuinely stale occupant is displaced.
    ///
    /// **Least-recently-used first**, by `idle_secs` rather than by residency:
    /// a worker answering steadily for an hour and one loaded an hour ago and
    /// never used since are indistinguishable by `spawned_at`, and deserve
    /// opposite treatment.
    ///
    /// Returns the megabytes reclaimed.
    async fn free_vram_for_admission(&self, exclude: &ModelId, needed_mb: u64) -> u64 {
        let budget = self
            .vram_budget_mb
            .load(std::sync::atomic::Ordering::Relaxed);
        if budget == 0 || needed_mb == 0 {
            return 0;
        }
        let committed = self.vram_committed_mb();
        if committed.saturating_add(needed_mb) <= budget {
            return 0;
        }

        // Candidates: workers holding a VRAM charge that are neither the model
        // being loaded, nor busy, nor recently used.
        let candidates: Vec<VramReclaimCandidate> = self
            .vram_reserved_mb
            .iter()
            .filter(|e| e.key() != exclude && *e.value() > 0)
            .filter_map(|e| {
                let worker = self.workers.get(e.key())?;
                Some(VramReclaimCandidate {
                    model: e.key().clone(),
                    charge_mb: *e.value(),
                    idle_secs: worker.idle_secs(),
                    busy: !worker.responses.is_empty(),
                })
            })
            .collect();

        let plan = plan_vram_reclaim(budget, committed, needed_mb, candidates);
        if plan.is_empty() {
            tracing::debug!(
                model = %exclude,
                needed_mb,
                committed_mb = committed,
                budget_mb = budget,
                "No idle model could be reclaimed to fit this one — leaving the GPU as it is"
            );
            return 0;
        }

        let mut reclaimed = 0u64;
        for (victim, charge) in plan {
            tracing::info!(
                model = %victim,
                reclaimed_mb = charge,
                for_model = %exclude,
                "Freeing graphics memory from an idle model so the requested one can use the GPU"
            );
            self.unload_model(&victim).await;
            reclaimed = reclaimed.saturating_add(charge);
        }
        reclaimed
    }

    /// Gather the inputs for [`should_return_to_gpu`] and ask it.
    ///
    /// Cheap enough for the request path — which is where it has to run, since
    /// the whole point is to act when the model is next wanted rather than on a
    /// timer. `cpu_placed` is a plain field, so a worker on the card (and every
    /// worker on a node without one) is dismissed by a bool read; the atomics
    /// and the small `vram_reserved_mb` sum are only reached for a model this
    /// node has actually demoted. Nothing here touches the disk.
    fn worker_should_return_to_gpu(&self, model_id: &ModelId, handle: &WorkerHandle) -> bool {
        if handle.placed_on_cpu_because.is_none() {
            return false;
        }
        let budget_mb = self
            .vram_budget_mb
            .load(std::sync::atomic::Ordering::Relaxed);
        should_return_to_gpu(&PromotionInputs {
            cpu_placed: true,
            permanently_cpu_bound: !self.gpu_is_usable(),
            busy: !handle.responses.is_empty(),
            idle_secs: handle.idle_secs(),
            gpu_estimate_mb: handle.gpu_estimate_mb,
            budget_mb,
            committed_mb: self.vram_committed_mb(),
            reclaimable_mb: self.reclaimable_vram_mb(model_id, handle.gpu_estimate_mb, budget_mb),
        })
    }

    /// Can this NODE put models on a graphics card at all, memory aside?
    ///
    /// False for the two of [`ModelProcessPool::cpu_reason`]'s three causes that
    /// never clear — the user's own `gpu_layers = 0`, and a card below this
    /// build's kernel floor — plus the cases where there is no card or no CUDA
    /// in this build. Deliberately takes no model: all of those are properties
    /// of the machine.
    ///
    /// **A VRAM pin is deliberately not consulted**: it is the recoverable one,
    /// and treating it as disqualifying is what made the promotion unable to
    /// reconsider the very condition it exists for.
    fn gpu_is_usable(&self) -> bool {
        if self.gpu_layers.load(std::sync::atomic::Ordering::Relaxed) == 0 {
            return false;
        }
        #[cfg(feature = "candle-cuda")]
        if !crate::daemon::gpu_support::local_gpu_is_supported() {
            return false;
        }
        // Same three inputs as everywhere else: sent to the processor by
        // configuration, no card, or a build that cannot drive one.
        !charges_ram(
            false,
            self.gpu_detected.load(std::sync::atomic::Ordering::Relaxed),
            cfg!(feature = "candle-cuda"),
        )
    }

    /// A DRY RUN of [`ModelProcessPool::free_vram_for_admission`]: how much the
    /// pool would be willing to take from other models right now.
    ///
    /// Same planner, same guards — idle floor, never a busy worker, and
    /// all-or-nothing — so the answer is memory the pool has already agreed it
    /// may reclaim, not a hope. Nothing is unloaded here; the real reclaim
    /// happens on the admission that follows, from the same plan.
    ///
    /// **Why the promotion needs it.** A pin is lifted only when a GPU-holding
    /// worker unloads, and the idle-unload timer is minutes away — so a model
    /// on the processor could sit there while its card-mate had been idle long
    /// enough to be reclaimed on demand. Asking only "does it fit in the room
    /// lying about" made the promotion wait for a timer instead of for the
    /// condition.
    fn reclaimable_vram_mb(&self, exclude: &ModelId, needed_mb: u64, budget_mb: u64) -> u64 {
        if budget_mb == 0 || needed_mb == 0 {
            return 0;
        }
        let committed = self.vram_committed_mb();
        if committed.saturating_add(needed_mb) <= budget_mb {
            return 0;
        }
        let candidates: Vec<VramReclaimCandidate> = self
            .vram_reserved_mb
            .iter()
            .filter(|e| e.key() != exclude && *e.value() > 0)
            .filter_map(|e| {
                let worker = self.workers.get(e.key())?;
                Some(VramReclaimCandidate {
                    model: e.key().clone(),
                    charge_mb: *e.value(),
                    idle_secs: worker.idle_secs(),
                    busy: !worker.responses.is_empty(),
                })
            })
            .collect();
        plan_vram_reclaim(budget_mb, committed, needed_mb, candidates)
            .iter()
            .map(|(_model, mb)| *mb)
            .sum()
    }

    /// Release a worker's charge. Must pair with every `admit_to_gpu`.
    fn release_vram_charge(&self, model_id: &ModelId) {
        self.vram_reserved_mb.remove(model_id);
        self.charged_segments.remove(model_id);
    }

    /// Set the system RAM budget used for CPU admission. 0 disables the check.
    pub fn set_ram_budget_mb(&self, budget_mb: u64) {
        self.ram_budget_mb
            .store(budget_mb, std::sync::atomic::Ordering::Relaxed);
    }

    /// Install the live budget source. Installed once; later calls are ignored.
    #[allow(clippy::type_complexity)]
    pub fn set_ram_budget_provider(
        &self,
        provider: Box<dyn Fn() -> Option<crate::model::auto_manage::vram::RamBudget> + Send + Sync>,
    ) {
        let _ = self.ram_budget_provider.set(provider);
    }

    /// The RAM budget to judge an admission against, RIGHT NOW. `None` = do
    /// not judge (no cap derivable, and none stored).
    fn ram_budget_now(&self) -> Option<crate::model::auto_manage::vram::RamBudget> {
        if let Some(p) = self.ram_budget_provider.get() {
            return p();
        }
        match self
            .ram_budget_mb
            .load(std::sync::atomic::Ordering::Relaxed)
        {
            0 => None,
            cap => Some(crate::model::auto_manage::vram::RamBudget::cap_only(cap)),
        }
    }

    /// See `ram_budget_note`.
    /// Tell the pool whether a GPU was detected on this node at all. A node
    /// without one never takes the VRAM-refusal branch (there is no budget to
    /// refuse against), so without this its models landed in system RAM with
    /// nothing charged and no KV budget handed to the worker — every CPU-only
    /// machine, which is exactly where swapping hurts most (2026-08-21).
    pub fn set_gpu_detected(&self, detected: bool) {
        self.gpu_detected
            .store(detected, std::sync::atomic::Ordering::Relaxed);
    }

    pub fn set_ram_budget_note(&self, note: String) {
        if let Ok(mut n) = self.ram_budget_note.lock() {
            *n = note;
        }
    }

    fn ram_budget_note(&self) -> String {
        self.ram_budget_note
            .lock()
            .map(|n| n.clone())
            .unwrap_or_default()
    }

    /// System RAM already committed to live CPU workers, in MB.
    fn ram_committed_mb(&self) -> u64 {
        self.ram_reserved_mb.iter().map(|e| *e.value()).sum()
    }

    /// After a CPU admission: how much KV this model's worker may hold — its
    /// typical-context charge (already inside the reservation) plus whatever of
    /// the RAM budget nothing else has claimed. Handed to the worker at spawn
    /// so its runtime guard refuses a conversation that would outgrow the
    /// room, with a 503 that re-routes, instead of swapping the machine.
    /// No budget configured → no guard (the pre-2026-08-21 behaviour).
    fn record_cpu_kv_budget(&self, model_id: &ModelId, segment: Option<(u32, u32)>) {
        let Some(budget) = self.ram_budget_now() else {
            self.cpu_kv_budget_bytes.remove(model_id);
            return;
        };
        let Some(inputs) = self.footprint_inputs(model_id, segment) else {
            self.cpu_kv_budget_bytes.remove(model_id);
            return;
        };
        let typical_kv = crate::model::auto_manage::vram::cpu_footprint(&inputs).kv_bytes;
        let this_mb = self.ram_reserved_mb.get(model_id).map(|v| *v).unwrap_or(0);
        let headroom_mb = budget.headroom_after(self.ram_committed_mb(), this_mb);
        let kv_budget = typical_kv.saturating_add(headroom_mb.saturating_mul(1024 * 1024));
        tracing::info!(
            model = %model_id,
            kv_budget_mb = kv_budget / (1024 * 1024),
            typical_kv_mb = typical_kv / (1024 * 1024),
            headroom_mb,
            "CPU worker KV budget: typical-context charge plus uncommitted RAM budget"
        );
        self.cpu_kv_budget_bytes.insert(model_id.clone(), kv_budget);
    }

    /// Decide whether `model_id` may be loaded into system RAM, and charge it
    /// if so. Called inside `spawn_lock` for the same read-decide-charge
    /// atomicity as [`Self::admit_to_gpu`].
    ///
    /// Unlike the GPU case there is **no further fallback**: the CPU already is
    /// the fallback. So returning `false` fails the spawn rather than demoting
    /// it, and the caller surfaces `ServiceUnavailable`. That is deliberate —
    /// the alternative is swapping, which does not merely slow this model down
    /// but degrades every other request on the machine, and does so without
    /// anything in the API response explaining why.
    fn admit_to_cpu(&self, model_id: &ModelId, estimated_mb: u64) -> bool {
        let Some(budget) = self.ram_budget_now() else {
            // No budget derivable: preserve the previous behaviour rather than
            // inventing a limit.
            add_reserved(&self.ram_reserved_mb, model_id, estimated_mb);
            return true;
        };
        if estimated_mb == 0 {
            // The model's geometry could not be read: nothing to weigh.
            add_reserved(&self.ram_reserved_mb, model_id, estimated_mb);
            return true;
        }
        let committed = self.ram_committed_mb();
        if budget.allows(committed, estimated_mb) {
            add_reserved(&self.ram_reserved_mb, model_id, estimated_mb);
            return true;
        }
        tracing::warn!(
            model = %model_id,
            estimated_mb,
            committed_mb = committed,
            cap_mb = budget.cap_mb,
            available_mb = budget.available_mb,
            live_headroom_mb = budget.live_headroom_mb,
            "Not enough system memory for this model right now — refusing to load it. \
             Loading it anyway would swap, which slows down every other request \
             on this machine, not just this one"
        );
        false
    }

    /// Make room in the RAM budget by unloading models nothing is using.
    ///
    /// Called when [`Self::admit_to_cpu`] refuses, BEFORE the refusal reaches
    /// the user. Without this, a node whose budget is already spent refuses the
    /// next model outright even when the resident one is doing nothing —
    /// reported from a live node: 6210 MB resident, 5986 MB wanted, 8000 MB
    /// budget, refused. The graphics-memory path has had pressure-based
    /// eviction for exactly this; system memory had none, so the request failed
    /// and only the next background cleanup pass fixed it.
    ///
    /// Only workers with **no in-flight requests** are candidates, read from
    /// the pool's own response channels, so this can never take memory from a
    /// request that is mid-generation. Longest-resident goes first.
    ///
    /// Deliberately does not consult the pin / reference-model lists that guard
    /// the background idle-unload. Those exist to keep a wanted model *warm*,
    /// which is a preference; this runs only when the alternative is failing
    /// the request outright, and a worker respawns on its next use. Returns the
    /// megabytes reclaimed.
    async fn free_ram_for_admission(&self, exclude: &ModelId, needed_mb: u64) -> u64 {
        if self.ram_budget_now().is_none() {
            return 0;
        }
        let mut freed = 0u64;
        // Re-read each round: unloading a model frees real memory, which the
        // live headroom sees.
        while let Some(budget) = self.ram_budget_now() {
            let committed = self.ram_committed_mb();
            if budget.allows(committed, needed_mb) {
                break;
            }
            // Longest-resident worker that is not serving anything.
            let victim = self
                .workers
                .iter()
                .filter(|e| e.key() != exclude && e.value().responses.is_empty())
                .min_by_key(|e| e.value().spawned_at)
                .map(|e| e.key().clone());
            let Some(victim) = victim else {
                break; // everything left is busy — refuse rather than interrupt it
            };
            let reclaimed = self.ram_reserved_mb.get(&victim).map(|v| *v).unwrap_or(0);
            tracing::info!(
                model = %victim,
                reclaimed_mb = reclaimed,
                for_model = %exclude,
                "Unloading an unused model to make room in the memory budget"
            );
            self.unload_model(&victim).await;
            freed = freed.saturating_add(reclaimed);
            if reclaimed == 0 {
                break; // nothing accounted to it; avoid spinning
            }
        }
        freed
    }

    /// Has this model's worker already been charged for `segment`?
    fn segment_is_charged(&self, model_id: &ModelId, segment: (u32, u32)) -> bool {
        self.charged_segments
            .get(model_id)
            .is_some_and(|v| v.contains(&segment))
    }

    fn record_charged_segment(&self, model_id: &ModelId, segment: (u32, u32)) {
        let mut e = self.charged_segments.entry(model_id.clone()).or_default();
        if !e.contains(&segment) {
            e.push(segment);
        }
    }

    /// Weigh and charge a layer range a live worker has not held before.
    ///
    /// Only the LAYERS are charged, never the fixed terms again: the process
    /// overhead and — unless this is the model's first segment — the embedding
    /// table are already inside the first segment's reservation, and charging
    /// them twice would refuse a second segment that fits perfectly well.
    /// `segment_cost_curve` takes both figures from the same estimator the
    /// admission gate uses, so nothing here restates its arithmetic.
    ///
    /// Refusing is a `ServiceUnavailable`, i.e. a 503 the coordinator fails
    /// over from — the same answer the worker's own KV guard gives, and the
    /// right one: this node cannot take that range, another might.
    ///
    /// **What this charge does NOT bound**, and what does. One request may pile
    /// several segments onto one worker: a node is its own preferred standby
    /// (`find_standbys` sorts it first), so each remote segment that fails over
    /// lands here again. Reported from a 16 GB processor-only Mac mini on
    /// v0.3.155: a 5-segment plan gave the local node 12 of a 48-layer 14B,
    /// two remote segments then failed over to it in turn, it was charged +9
    /// and +8 layers, and at 29 of 48 the worker process died — losing a
    /// request that had already streamed 238 tokens over ~10 minutes.
    ///
    /// Neither top-up was wrongly admitted. The running total IS tracked —
    /// `add_reserved` accumulates and `budget.allows` weighs
    /// `committed + estimate` against the cap — and on a 16 GB machine the cap
    /// (13107 MB) was never reached. What ran out was the machine, not the
    /// budget, because the budget's other term is a LOAD-TIME prediction: the
    /// worker is granted "whatever of the RAM budget nothing else has claimed"
    /// as room for its KV cache to grow into (`record_cpu_kv_budget`), and
    /// nothing revisited that grant as the same worker took on 17 more layers
    /// of weights.
    ///
    /// So the bound lives with the worker, not here:
    /// `SplitModel::kv_budget_now` reconciles the grant against free system
    /// memory at every decision that takes memory, exactly as the graphics
    /// side has done since gotcha #440. A count cap would have been the wrong
    /// bound anyway — the number of segments one worker may hold is not the
    /// quantity that runs out.
    async fn charge_additional_segment(
        &self,
        model_id: &ModelId,
        segment: (u32, u32),
        handle: &Arc<WorkerHandle>,
    ) -> Result<(), SwarmError> {
        // Serialised against spawns for the same read-decide-charge atomicity
        // the admission gates already rely on.
        let _guard = self.spawn_lock.lock().await;
        if self.segment_is_charged(model_id, segment) {
            return Ok(());
        }
        let on_gpu = handle.placed_on_cpu_because.is_none();
        let layers = u64::from(segment.1.saturating_sub(segment.0)).max(1);
        let Some((_fixed_mb, per_layer_mb)) = self.segment_cost_curve(model_id, on_gpu) else {
            // Unreadable geometry: nothing to weigh, and refusing on a file we
            // could not read would be a worse failure than the one prevented —
            // the same treatment `admit_to_gpu` and `admit_to_cpu` give it.
            self.record_charged_segment(model_id, segment);
            return Ok(());
        };
        let delta_mb = per_layer_mb.saturating_mul(layers);
        let admitted = if on_gpu {
            self.admit_to_gpu(model_id, delta_mb)
        } else if handle.charged_against_ram {
            self.admit_to_cpu(model_id, delta_mb)
        } else {
            true
        };
        if !admitted {
            return Err(SwarmError::ServiceUnavailable(format!(
                "{} layers {}..{} of {} need about {} MB more than this node has left \
                 (its worker is already holding {} MB) — another holder will have to \
                 take that part",
                layers,
                segment.0,
                segment.1,
                model_id.0,
                delta_mb,
                self.ram_reserved_mb.get(model_id).map(|v| *v).unwrap_or(0),
            )));
        }
        self.record_charged_segment(model_id, segment);
        tracing::info!(
            model = %model_id,
            layers = format!("[{}..{})", segment.0, segment.1),
            delta_mb,
            on_gpu,
            "Charging an additional segment to a live worker"
        );
        Ok(())
    }

    /// `(fixed_mb, per_layer_mb)` for this model on the given device.
    ///
    /// Derived by pricing two segment sizes through the SAME estimator the
    /// admission gate uses and taking the difference, so the incremental charge
    /// above and the scheduler's local capacity bound cannot drift from what
    /// the loader will actually be weighed against. Priced as a MIDDLE segment
    /// (`is_first: false`), so `fixed_mb` is the process overhead alone.
    pub(crate) fn segment_cost_curve(
        &self,
        model_id: &ModelId,
        on_gpu: bool,
    ) -> Option<(u64, u64)> {
        use crate::model::auto_manage::vram::{estimate_worker_ram_mb, estimate_worker_vram_mb};
        let base = self.footprint_inputs(model_id, None)?;
        if base.segment_layers == 0 {
            return None;
        }
        let at = |layers: u64| {
            let mut i = base;
            i.segment_layers = layers;
            i.is_first = false;
            i.quantized_weight_bytes = base.quantized_weight_bytes / base.segment_layers * layers;
            if on_gpu {
                estimate_worker_vram_mb(&i)
            } else {
                estimate_worker_ram_mb(&i)
            }
        };
        // Two points on a line that is affine in the layer count: the weights
        // and the KV cache both scale with it, everything else does not.
        Some(cost_curve_from(at(1), at(2)))
    }

    /// The most layers of `model_id` this node could admit right now, on the
    /// device a request for it would actually use.
    ///
    /// **The scheduler's answer to the question the loader will be asked**, and
    /// deliberately built from the same estimator and the same budgets — so a
    /// plan this node makes is a plan it can load.
    ///
    /// Before this existed, the local candidate was the ONE candidate the
    /// pipeline search priced as memory-unconstrained: every peer carried a
    /// `max_hostable_layers` from its advertised free memory, and the local
    /// node carried `None`, on the reasoning that our own admission check is
    /// the authority. That reasoning is right about WHO decides and wrong about
    /// WHEN: admission runs at load time, after the plan is committed and too
    /// late to reshape it. Reported from the field — a 16 GB processor-only Mac
    /// mini holding every shard of a 48-layer 14B was assigned 36 of its
    /// layers, refused them at load, retried, and produced the identical plan
    /// (gotcha #452). This node has the BEST information about its own memory,
    /// not the worst; it should be the most accurately bounded candidate.
    ///
    /// `None` means unknowable — no budget, or geometry that could not be read
    /// — and never "no room", the same reading `max_hostable_layers` gives an
    /// unreadable peer.
    pub fn max_local_hostable_layers(&self, model_id: &ModelId, on_gpu: bool) -> Option<u32> {
        let (fixed_mb, per_layer_mb) = self.segment_cost_curve(model_id, on_gpu)?;
        // What is free after everything already charged, on whichever budget
        // this model would be weighed against.
        let free_mb = if on_gpu {
            let budget = self
                .vram_budget_mb
                .load(std::sync::atomic::Ordering::Relaxed);
            if budget == 0 {
                return None;
            }
            budget.saturating_sub(self.vram_committed_mb())
        } else {
            let budget = self.ram_budget_now()?;
            budget.headroom_after(self.ram_committed_mb(), 0)
        };
        // The fixed terms are paid once, whatever the segment's length.
        layers_that_fit(free_mb, fixed_mb, per_layer_mb)
    }

    /// Retire a worker whose process is gone, releasing the memory it no
    /// longer holds. Returns whether this call was the one that retired it.
    ///
    /// **The single answer to "this worker died".** Every site that discovers
    /// `dead == true` goes through it, and so does the periodic reap below —
    /// so a worker that nobody asks about again is still cleaned up.
    ///
    /// What it replaced: three call sites did `workers.remove(&model_id)` and
    /// nothing else, and only the graceful `unload_model` released the charge.
    /// So a worker that exited any other way — an internal crash, an OS
    /// OOM-kill, or a user closing the process in a system monitor to free
    /// memory — left its whole reservation charged for the daemon's lifetime.
    /// The charge is summed across every model by `ram_committed_mb` /
    /// `vram_committed_mb`, so the phantom blocked not just that model but
    /// every later load on the node, quoting memory that was in fact free.
    /// Reported from a 16 GB Mac mini: 6 consecutive requests refused with a
    /// byte-for-byte identical "11487 MB is already in use" over a minute,
    /// right after the tester had killed the worker to free RAM. Only a daemon
    /// restart cleared it.
    ///
    /// Two things a change here must keep. It takes `spawn_lock`, because a
    /// spawn charges and inserts under that lock and releasing between the two
    /// would free the NEW worker's charge. And it removes CONDITIONALLY
    /// (`remove_if` on `dead`), so a live worker that replaced this one in the
    /// map is never retired by a late caller holding the corpse — releasing is
    /// reached only when the remove actually happened.
    async fn retire_dead_worker(&self, model_id: &ModelId) -> bool {
        let _guard = self.spawn_lock.lock().await;
        self.retire_dead_worker_locked(model_id)
    }

    /// Take a worker out of the pool and release what it held. **The one way a
    /// worker leaves `workers`**, apart from `unload_model`, which must drain
    /// before killing and so does its own removal before ending in the same
    /// `after_worker_gone`.
    ///
    /// `pred` decides whether the entry currently under that key is the one
    /// being evicted, so a caller holding a dead handle cannot remove the live
    /// worker that has already replaced it.
    ///
    /// **Why it exists.** Gotcha #461 fixed three sites that removed a dead
    /// worker without releasing its memory charge — and there were nine. The
    /// other six are the paths a worker dies on in practice: a failed IPC send,
    /// a closed reader channel on forward / batch-forward / generate, and
    /// `classify_worker_error`'s fatal arm, which is the CUDA-OOM path and the
    /// one the reporting node hits most. None of them released anything, and
    /// the health-tick reap cannot save them: it scans `workers`, and these
    /// have already removed the entry, so the charge leaked permanently
    /// (gotcha #467).
    fn evict_worker_where(
        &self,
        model_id: &ModelId,
        pred: impl Fn(&Arc<WorkerHandle>) -> bool,
        how: &str,
    ) -> bool {
        let Some((_, handle)) = self.workers.remove_if(model_id, |_, cur| pred(cur)) else {
            return false;
        };
        let freed_gpu_memory = handle.placed_on_cpu_because.is_none();
        self.after_worker_gone(model_id, freed_gpu_memory, how);
        true
    }

    /// Evict the worker the caller was talking to, and only that one.
    fn evict_this_worker(&self, model_id: &ModelId, handle: &Arc<WorkerHandle>, how: &str) {
        self.evict_worker_where(model_id, |cur| Arc::ptr_eq(cur, handle), how);
    }

    /// The body of [`Self::retire_dead_worker`], with `spawn_lock` already
    /// held. Split out so the background reap can use `try_lock`.
    fn retire_dead_worker_locked(&self, model_id: &ModelId) -> bool {
        self.evict_worker_where(
            model_id,
            |h| h.dead.load(Ordering::Acquire),
            "exited unexpectedly",
        )
    }

    /// Retire every worker whose process has exited. Wired to the health tick.
    ///
    /// The call-site path above only fires when someone asks for that same
    /// model again. A node whose 14B worker died and is then asked for a
    /// different model would otherwise keep the dead one's reservation for
    /// ever — the charge is a single shared budget, so it refuses the OTHER
    /// model. Returns how many were retired.
    pub async fn reap_dead_workers(&self) -> usize {
        let dead: Vec<ModelId> = self
            .workers
            .iter()
            .filter(|e| e.value().dead.load(Ordering::Acquire))
            .map(|e| e.key().clone())
            .collect();
        if dead.is_empty() {
            // The common case takes no lock at all.
            return 0;
        }
        // `try_lock`, never `lock`: a spawn holds `spawn_lock` for the whole
        // model load, which is minutes for a large model on a processor, and
        // this runs on the health monitor's task alongside peer pings and
        // gossip. A corpse can wait 30 s for the next tick — the same trade
        // the anti-gaming cleanup on that tick already makes.
        let Ok(_guard) = self.spawn_lock.try_lock() else {
            tracing::debug!(
                pending = dead.len(),
                "A spawn is in progress — leaving dead workers for the next tick"
            );
            return 0;
        };
        dead.iter()
            .filter(|m| self.retire_dead_worker_locked(m))
            .count()
    }

    /// Release a worker's RAM charge. Must pair with every `admit_to_cpu`.
    fn release_ram_charge(&self, model_id: &ModelId) {
        self.ram_reserved_mb.remove(model_id);
        self.cpu_kv_budget_bytes.remove(model_id);
        self.charged_segments.remove(model_id);
    }

    /// Models currently forced onto the CPU after a GPU OOM.
    ///
    /// Reported by `GET /api/admin/stats` because `inference_backend` describes
    /// the BUILD, not what any model is actually running on — it kept saying
    /// "CUDA" while every model had been pinned to the CPU, so an operator
    /// losing ~10x throughput had nothing to see. Reported 2026-07-30.
    pub fn cpu_pinned_model_ids(&self) -> Vec<String> {
        let mut v: Vec<String> = self
            .cpu_pinned_models
            .iter()
            .map(|m| m.key().0.clone())
            .collect();
        v.sort();
        v
    }

    /// Clear a CPU pin so the next worker spawn may use the GPU again.
    /// Exposed for the admin "retry on GPU" path — VRAM pressure is usually
    /// transient (another model unloaded, another process exited).
    pub fn clear_cpu_pin(&self, model_id: &ModelId) -> bool {
        self.cpu_pinned_models.remove(model_id).is_some()
    }

    /// Force every attention call through `standard_attention` on
    /// future-spawned workers (auto-enabled while SWIFT is on).
    pub fn set_force_standard_attn(&self, force: bool) {
        self.force_standard_attn
            .store(force, std::sync::atomic::Ordering::Relaxed);
    }

    /// Cap the GGUF context_length when constructing the KV cache. Pass `None`
    /// to use the GGUF value verbatim.
    pub fn set_max_seq_len_override(&self, override_val: Option<u32>) {
        self.max_seq_len_override.store(
            override_val.unwrap_or(0),
            std::sync::atomic::Ordering::Relaxed,
        );
    }

    /// Apply SWIFT self-speculative decoding settings to future-spawned workers.
    pub fn set_swift_config(
        &self,
        enabled: bool,
        calibration_tokens: u32,
        gamma: u32,
        skip_ratio: f32,
    ) {
        use std::sync::atomic::Ordering;
        self.swift_self_speculative
            .store(enabled, Ordering::Relaxed);
        self.swift_calibration_tokens
            .store(calibration_tokens, Ordering::Relaxed);
        self.swift_gamma.store(gamma.max(1), Ordering::Relaxed);
        let milli = (skip_ratio.clamp(0.0, 0.95) * 1000.0).round() as u32;
        self.swift_skip_ratio_milli.store(milli, Ordering::Relaxed);
    }

    /// Set the KV-cache TTL for worker subprocesses (called once after config load).
    pub fn set_kv_cache_ttl(&self, ttl_secs: u64) {
        self.kv_cache_ttl_secs
            .store(ttl_secs, std::sync::atomic::Ordering::Relaxed);
    }

    /// Apply the prefix-cache section of inference config to future-spawned workers.
    pub fn set_prefix_cache_config(
        &self,
        enabled: bool,
        max_entries: u32,
        max_prompt_tokens: u32,
        block_tokens: u32,
        min_tokens: u32,
        max_mb: u32,
    ) {
        use std::sync::atomic::Ordering;
        self.prefix_cache_enabled.store(enabled, Ordering::Relaxed);
        self.prefix_cache_max_entries
            .store(max_entries, Ordering::Relaxed);
        self.prefix_cache_max_mb.store(max_mb, Ordering::Relaxed);
        self.prefix_cache_max_prompt_tokens
            .store(max_prompt_tokens, Ordering::Relaxed);
        self.prefix_cache_block_tokens
            .store(block_tokens, Ordering::Relaxed);
        self.prefix_cache_min_tokens
            .store(min_tokens, Ordering::Relaxed);
    }

    /// Set the activity event sender (called once after SharedState is created).
    pub fn set_activity_tx(
        &self,
        tx: tokio::sync::broadcast::Sender<crate::daemon::state::ActivityEvent>,
    ) {
        let _ = self.activity_tx.set(tx);
    }

    /// Get or spawn a worker for this model.
    /// `segment` is the layer range this request needs. One worker serves a
    /// model, and its `models` map is keyed by layer range — so a worker can
    /// come to hold SEVERAL segments of one model, and each is charged as it is
    /// first asked for. See [`Self::charge_additional_segment`].
    async fn get_or_spawn(
        &self,
        model_id: &ModelId,
        segment: (u32, u32),
    ) -> Result<Arc<WorkerHandle>, SwarmError> {
        // Fast path: worker already exists — and, unless this node demoted it
        // to the processor and the card has since made room, that is the answer.
        let existing = self.workers.get(model_id).map(|h| h.clone());
        if let Some(handle) = existing {
            if !self.worker_should_return_to_gpu(model_id, &handle) {
                if self.segment_is_charged(model_id, segment) {
                    return Ok(handle);
                }
                // A range this worker has not been asked for before: it is
                // about to load more weights, so weigh them first.
                self.charge_additional_segment(model_id, segment, &handle)
                    .await?;
                return Ok(handle);
            }
        }

        // Slow path: serialize spawns to prevent duplicate workers
        let _guard = self.spawn_lock.lock().await;
        // Re-check after acquiring lock (another task may have spawned it), and
        // re-take the promotion decision under the lock: nothing else can charge
        // the budget while we hold it, so the "it fits" the retirement is
        // justified by is still true at the admission below.
        let retire_for_gpu = match self.workers.get(model_id) {
            Some(handle) => {
                if !self.worker_should_return_to_gpu(model_id, &handle) {
                    return Ok(handle.clone());
                }
                true
            }
            None => false,
        };
        // Crash-loop backoff: refuse to re-spawn while a recent failure is
        // still inside its cooldown window. Without this a permanently-broken
        // model (corrupt shards, GPU OOM on load) burns one
        // WORKER_CONNECT_TIMEOUT_SECS = 30s spawn attempt per arriving
        // request and saturates the whole inference path.
        // Check INSIDE the spawn_lock so a concurrent failure that records
        // itself between our entry and our spawn cannot be bypassed.
        if let Some(entry) = self.spawn_failures.get(model_id) {
            let (at, count) = *entry;
            let cooldown = spawn_failure_cooldown(count);
            if at.elapsed() < cooldown {
                let remaining = cooldown.saturating_sub(at.elapsed());
                return Err(SwarmError::ServiceUnavailable(format!(
                    "Worker spawn for {} failing repeatedly — backing off for {:?} (attempt #{})",
                    model_id.0, remaining, count
                )));
            }
        }
        // Deliberately AFTER the crash-loop check: retiring a worker we would
        // then refuse to respawn would leave the model with nothing at all.
        if retire_for_gpu {
            tracing::info!(
                model = %model_id,
                committed_mb = self.vram_committed_mb(),
                budget_mb = self
                    .vram_budget_mb
                    .load(std::sync::atomic::Ordering::Relaxed),
                "Graphics memory has freed up — retiring this model's processor worker \
                 so it can be admitted to the GPU again"
            );
            // Drops the worker and its RAM charge; the spawn below re-runs
            // admission from scratch. Deliberately says nothing here about
            // where the model ends up: admission has the last word and prices
            // the model again from its own geometry, so announcing the outcome
            // at this point is a claim the next line could falsify — the exact
            // mistake `admit_to_gpu`'s own refusal log was corrected for. The
            // user-facing event is emitted below, once it is a fact.
            self.unload_model(model_id).await;
        }
        // Admission control. Inside `spawn_lock`, so read-decide-charge cannot
        // interleave with another spawn — which matters because a worker is
        // admitted long before it allocates (the model loads lazily on its first
        // message), so two spawns that both asked the device would each see
        // plenty free and both proceed. Refusing here means the model loads on
        // the CPU: slower, but not a dead worker and a lost GPU.
        // Why this spawn is going to the processor, if it is — decided ONCE
        // here and handed to `spawn_worker`, which used to re-derive it from
        // `cpu_reason`. Re-deriving is what made the decision depend on mutable
        // state read at a different moment: with admission no longer pinning,
        // `cpu_reason` is `None` for a model this very function is about to
        // place on the processor, and a worker spawned on that answer would go
        // to the card and die there.
        let mut placed_on_cpu_because = self.cpu_reason(model_id);
        let mut going_to_cpu = placed_on_cpu_because.is_some();
        // Set when the model does not fit the card whole but part of it does:
        // `(layers on the card, layers in total)`.
        let mut hybrid_layers: Option<(usize, usize)> = None;
        // What the card would have to give this model. Read from disk ONCE per
        // spawn: the admission gate weighs it, the CPU-fallback log reports it,
        // and the worker carries it so a later request can ask whether the
        // model would fit now without re-reading the geometry.
        let estimated = self.estimate_gpu_footprint_mb(model_id, Some(segment));
        if !going_to_cpu {
            // Demoting to the CPU is the last resort, not the first answer:
            // reclaim the card from models nothing is using, then ask again.
            // Ask ONCE unless the answer was no — `admit_to_gpu` charges the
            // reservation when it succeeds, so a second call would weigh the
            // model against its own charge and refuse one already let in.
            let mut admitted = self.admit_to_gpu(model_id, estimated);
            if !admitted {
                let freed = self.free_vram_for_admission(model_id, estimated).await;
                if freed > 0 {
                    tracing::info!(
                        model = %model_id,
                        freed_mb = freed,
                        "Reclaimed graphics memory from idle models; retrying admission"
                    );
                    admitted = self.admit_to_gpu(model_id, estimated);
                }
            }
            // Before giving the card up entirely, ask whether PART of the
            // model fits. This is the whole point of hybrid placement: a model
            // 20% too large used to lose the card completely, a ~24x cliff on
            // this swarm, with graphics memory sitting free and unused beside
            // it (three reports, most recently 5151 MB free while the model
            // ran on the processor).
            if !admitted {
                if let Some((n, total)) =
                    self.partial_gpu_layers(model_id, Some(segment), estimated)
                {
                    tracing::info!(
                        model = %model_id,
                        gpu_layers = n,
                        total_layers = total,
                        estimated_mb = estimated,
                        "Model does not fit the card whole — placing its first {n} of {total} \
                         layers there and the rest on the processor"
                    );
                    hybrid_layers = Some((n, total));
                    // It IS going to the card, just not all of it, so this is
                    // not a CPU fallback and must not be reported as one.
                    admitted = true;
                }
            }
            if !admitted {
                // Now it IS true: the budget could not be made to fit, even
                // after reclaiming every idle model that could be spared.
                tracing::warn!(
                    model = %model_id,
                    estimated_mb = estimated,
                    committed_mb = self.vram_committed_mb(),
                    budget_mb = self
                        .vram_budget_mb
                        .load(std::sync::atomic::Ordering::Relaxed),
                    "Not enough GPU memory budget for this model — loading it on the CPU \
                     instead (slower, but it answers). It will use the GPU again once \
                     memory frees up"
                );
                going_to_cpu = true;
                placed_on_cpu_because = Some(CpuReason::NotEnoughVram);
                // **Deliberately NOT a pin.** A refusal here is arithmetic
                // against the budget as it stands right now, and re-taking it
                // on the next spawn costs one header read — whereas a pin
                // outlives the condition that caused it. It is cleared only
                // when a GPU-holding worker unloads, so a model demoted while
                // the card was busy stayed on the processor for as long as the
                // occupant was kept resident, without admission ever being
                // asked again. Measured here 2026-08-28: 50 minutes on the
                // processor with the occupant idle throughout and reclaimable
                // the whole time, because `effective_gpu_layers` saw the pin
                // and never reached admission at all.
                //
                // `cpu_pinned_models` now means only what its name says: the
                // graphics card has FAILED for this model (`classify_worker_
                // error`), which is the case where retrying really does cost a
                // load.
                if let Some(tx) = self.activity_tx.get() {
                    let _ = tx.send(
                        crate::daemon::state::ActivityEvent::new(
                            "inference",
                            "model_cpu_fallback",
                            format!(
                                "{} needs more graphics memory than is free — running it on the CPU for now",
                                model_id.0
                            ),
                        )
                        .with_model(model_id.0.clone())
                        .with_toast("warning", 8000),
                    );
                }
            }
        }
        // Say WHICH of the three reasons put this model on the CPU, every time
        // a worker spawns — not only at the moment a VRAM pin is taken. The
        // worker's command line and the loader's own log line are identical for
        // all three, so a node inspected later cannot otherwise be asked
        // "is this my setting, or did you override it?". Reported 2026-08-10 by
        // a tester who reasonably concluded the setting was being ignored.
        if going_to_cpu {
            let reason = placed_on_cpu_because
                .map(CpuReason::as_str)
                .unwrap_or("not_enough_vram");
            tracing::info!(
                model = %model_id,
                reason,
                configured_gpu_layers = self.configured_gpu_layers(),
                estimated_vram_mb = estimated,
                vram_budget_mb = self.vram_budget_mb.load(std::sync::atomic::Ordering::Relaxed),
                "Model will run on the CPU"
            );
        }
        // Anything landing in system RAM — a CPU-only node, or the fallback
        // just taken above — is charged against the RAM budget. There is no
        // further device to demote to, so this refuses rather than degrades:
        // swapping would slow every other request on the machine, not just
        // this model, and nothing in the response would say so.
        //
        // `going_to_cpu` alone missed every CPU-only node: with no GPU there is
        // no VRAM budget, `admit_to_gpu` admits everything, and the model lands
        // in RAM uncharged. `charges_ram` adds "no GPU detected" and "this build
        // has no CUDA"; placement is deliberately NOT changed by it — the worker
        // falls back to the CPU on its own, so a working card whose probe failed
        // is never sent to the CPU by this (unreadable is unknown, not absent).
        let charge_ram = charges_ram(
            going_to_cpu,
            self.gpu_detected.load(std::sync::atomic::Ordering::Relaxed),
            cfg!(feature = "candle-cuda"),
        );
        if charge_ram && !going_to_cpu {
            tracing::debug!(
                model = %model_id,
                "No usable GPU on this node — charging the model against the RAM budget"
            );
        }
        if charge_ram {
            let estimated = self.estimate_cpu_footprint_mb(model_id, Some(segment));
            // Refusing is the last resort: first reclaim memory from models
            // nothing is using, then ask again. Only then does the user see an
            // error.
            //
            // Ask ONCE unless the answer was no. `admit_to_cpu` charges the
            // reservation when it succeeds, so calling it a second time after a
            // success weighs the model against its own charge and refuses a
            // model that had already been let in.
            let mut admitted = self.admit_to_cpu(model_id, estimated);
            if !admitted {
                let freed = self.free_ram_for_admission(model_id, estimated).await;
                if freed > 0 {
                    tracing::info!(
                        model = %model_id,
                        freed_mb = freed,
                        "Reclaimed memory from unused models; retrying admission"
                    );
                }
                admitted = self.admit_to_cpu(model_id, estimated);
            }
            if !admitted {
                self.release_vram_charge(model_id);
                if let Some(tx) = self.activity_tx.get() {
                    let _ = tx.send(
                        crate::daemon::state::ActivityEvent::new(
                            "inference",
                            "model_ram_refused",
                            format!(
                                "{} needs more memory than this node is allowed to use — not loading it",
                                model_id.0
                            ),
                        )
                        .with_model(model_id.0.clone())
                        .with_toast("warning", 8000),
                    );
                }
                let in_use = self.ram_committed_mb();
                // The figure the model was judged against RIGHT NOW, and why —
                // the cap, or the live anti-swap headroom.
                let (budget, note) = self
                    .ram_budget_now()
                    .map(|b| b.limiting_figure(in_use, estimated))
                    .unwrap_or_else(|| {
                        (
                            self.ram_budget_mb
                                .load(std::sync::atomic::Ordering::Relaxed),
                            self.ram_budget_note(),
                        )
                    });
                let message = match self.cpu_footprint_detail(model_id, Some(segment)) {
                    Some((footprint, effective_context, source)) => {
                        crate::model::auto_manage::vram::describe_cpu_refusal(
                            &model_id.0,
                            &footprint,
                            effective_context,
                            source,
                            budget,
                            &note,
                            in_use,
                        )
                    }
                    None => format!(
                        "{} needs about {} MB of memory but this node's budget allows {} MB \
                         and {} MB is already in use. Raise `resources.max_ram_mb`, or free \
                         memory by unloading another model.",
                        model_id.0, estimated, budget, in_use,
                    ),
                };
                return Err(SwarmError::ServiceUnavailable(message));
            }
            self.record_cpu_kv_budget(model_id, Some(segment));
        }

        match self
            .spawn_worker(
                model_id,
                placed_on_cpu_because,
                charge_ram,
                estimated,
                hybrid_layers,
            )
            .await
        {
            Ok(handle) => {
                // Reset the failure counter on first success.
                self.spawn_failures.remove(model_id);
                let handle = Arc::new(handle);
                self.workers.insert(model_id.clone(), handle.clone());
                // The range the admission above weighed is now this worker's
                // first charged segment. Without recording it, every later
                // forward for the same range would look like a new one and be
                // charged again.
                self.record_charged_segment(model_id, segment);
                // Only now is it a fact. The user was told when this model was
                // demoted to the processor, so they are told when it comes
                // back — and if admission refused it a second time they were
                // told that instead, above, rather than this.
                if retire_for_gpu && !going_to_cpu {
                    if let Some(tx) = self.activity_tx.get() {
                        let _ = tx.send(
                            crate::daemon::state::ActivityEvent::new(
                                "inference",
                                "model_gpu_restored",
                                format!(
                                    "{} is back on the GPU — it will run faster again",
                                    model_id.0
                                ),
                            )
                            .with_model(model_id.0.clone())
                            .with_toast("info", 5000),
                        );
                    }
                }
                Ok(handle)
            }
            Err(e) => {
                // The worker never started, so it owes nothing. Leaving the
                // charge would shrink the budget permanently on every failure.
                // Both devices: a CPU-bound spawn was charged RAM above.
                self.release_vram_charge(model_id);
                self.release_ram_charge(model_id);
                let count = self
                    .spawn_failures
                    .entry(model_id.clone())
                    .and_modify(|v| *v = (std::time::Instant::now(), v.1.saturating_add(1)))
                    .or_insert((std::time::Instant::now(), 1))
                    .1;
                let cooldown = spawn_failure_cooldown(count);
                // Severity from the classification, not from the call site:
                // most of `spawn_worker`'s arms are `ServiceUnavailable`
                // (socket bind, spawn, accept, init timeout), which is a 503
                // and a WARN. Logging every one at ERROR reports ordinary
                // transient spawn contention as this node being broken.
                crate::log_failure!(
                    &e,
                    model = %model_id,
                    consecutive_failures = count,
                    cooldown_secs = cooldown.as_secs(),
                    error = %e,
                    "Worker spawn failed — backing off"
                );
                if let Some(tx) = self.activity_tx.get() {
                    let _ = tx.send(
                        crate::daemon::state::ActivityEvent::new(
                            "model",
                            "worker_spawn_failed",
                            format!(
                                "Worker spawn failed for {} (attempt #{}, cooldown {}s)",
                                model_id.0,
                                count,
                                cooldown.as_secs()
                            ),
                        )
                        .with_model(model_id.0.clone())
                        .with_toast("error", 6000),
                    );
                }
                Err(e)
            }
        }
    }

    /// `placed_on_cpu_because` is the placement `get_or_spawn` DECIDED, not one
    /// re-derived here: it is the only caller, it has just weighed admission,
    /// and its answer is the one the worker must be spawned with.
    ///
    /// `gpu_estimate_mb` is what admission priced this model at — carried onto
    /// the handle so a later request can re-ask whether it fits without going
    /// back to disk for the geometry. 0 means it could not be read.
    async fn spawn_worker(
        &self,
        model_id: &ModelId,
        placed_on_cpu_because: Option<CpuReason>,
        charged_against_ram: bool,
        gpu_estimate_mb: u64,
        // A partial split decided by admission: `(this many of the worker's
        // layers go on the card, layers in total)`, the rest on the processor.
        // `None` is the all-or-nothing placement `placed_on_cpu_because`
        // describes.
        gpu_layers_override: Option<(usize, usize)>,
    ) -> Result<WorkerHandle, SwarmError> {
        use interprocess::local_socket::{tokio::prelude::*, ListenerOptions};

        // Cross-platform socket naming:
        //  * Unix: filesystem path under `$TMPDIR/swarmllm-worker-<uuid>.sock`.
        //    `chmod 0o600` below restricts connect() to the current user.
        //  * Windows: namespace name `swarmllm-worker-<uuid>` (becomes
        //    `\\.\pipe\swarmllm-worker-<uuid>`). The default DACL on a named
        //    pipe grants access only to the current logon session — the
        //    equivalent of 0o600 for cross-user isolation.
        // Unix: the shortest path that fits `sun_path` — see
        // [`worker_socket_path`] for why that is not a detail.
        #[cfg(unix)]
        let socket_name: String = worker_socket_path()?;
        #[cfg(windows)]
        let socket_name: String = format!("swarmllm-worker-{}", uuid::Uuid::new_v4());

        // RAII guard: remove the Unix socket file if spawn errors out partway.
        // Defused on success; WorkerHandle's Drop then owns the cleanup.
        // Windows named pipes are kernel-reclaimed — no guard needed.
        #[cfg(unix)]
        let socket_guard = {
            struct SocketCleanup(String);
            impl Drop for SocketCleanup {
                fn drop(&mut self) {
                    let _ = std::fs::remove_file(&self.0);
                }
            }
            SocketCleanup(socket_name.clone())
        };

        // Build the interprocess `Name` — filesystem path on Unix, namespace
        // name on Windows. The Name borrows from socket_name, which outlives
        // create_tokio() below.
        #[cfg(unix)]
        let ipc_name = socket_name
            .as_str()
            .to_fs_name::<interprocess::local_socket::GenericFilePath>()
            .map_err(|e| SwarmError::Internal(format!("ipc name: {e}")))?;
        #[cfg(windows)]
        let ipc_name = socket_name
            .as_str()
            .to_ns_name::<interprocess::local_socket::GenericNamespaced>()
            .map_err(|e| SwarmError::Internal(format!("ipc name: {e}")))?;

        // SEC: On Unix, set umask(0o177) BEFORE bind so the socket file is
        // created with mode 0o600 atomically. The previous approach
        // (post-bind set_permissions) leaves a window where a local attacker
        // racing inotify on /tmp could connect first and impersonate the
        // worker, receiving plaintext prompts.
        //
        // umask is process-global; we save and restore it. Other threads
        // creating files during this brief window would see 0o600 perms
        // applied — that's strictly more restrictive, not less, so safe.
        // libc::umask is async-signal-safe and not blocking.
        #[cfg(unix)]
        let prev_umask = unsafe { libc::umask(0o177) };

        // Start listening before spawning so the worker can connect immediately.
        let listener = ListenerOptions::new()
            .name(ipc_name)
            .create_tokio()
            .map_err(|e| {
                #[cfg(unix)]
                unsafe {
                    libc::umask(prev_umask);
                }
                // Name the path. The reported failure ("length exceeds
                // capacity of sun_path") said what was wrong and not what
                // it was wrong about, and the path could not be recovered
                // from the logs at any verbosity.
                SwarmError::ServiceUnavailable(format!(
                    "socket bind at {socket_name} ({} chars): {e}",
                    socket_name.len()
                ))
            })?;

        #[cfg(unix)]
        unsafe {
            libc::umask(prev_umask);
        }

        // Defense-in-depth chmod — covers any platform/filesystem where the
        // umask path didn't kick in (some FUSE mounts, weird umasks).
        // On Windows the default named-pipe DACL already scopes to the
        // current logon session — equivalent isolation, no extra call.
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let _ = std::fs::set_permissions(&socket_name, std::fs::Permissions::from_mode(0o600));
        }

        // Spawn the worker subprocess (same binary, model-worker subcommand)
        let exe = crate::current_exe_path()
            .map_err(|e| SwarmError::ServiceUnavailable(format!("current_exe: {e}")))?;
        let socket_str = socket_name.as_str();
        let data_dir_str = self
            .data_dir
            .to_str()
            .ok_or_else(|| SwarmError::Internal("data dir path is not valid UTF-8".into()))?;
        let mut args = vec![
            "model-worker".to_string(),
            "--socket".to_string(),
            socket_str.to_string(),
            "--data-dir".to_string(),
            data_dir_str.to_string(),
        ];

        // Give the worker the verbosity the daemon was started with. Inference
        // happens in here, so a daemon run with `-v` that leaves this process at
        // INFO is blind exactly where it matters most.
        let verbosity = crate::DAEMON_VERBOSITY.load(Ordering::Relaxed);
        for _ in 0..verbosity {
            args.push("-v".to_string());
        }

        // Pass KV-cache TTL from config
        let ttl = self
            .kv_cache_ttl_secs
            .load(std::sync::atomic::Ordering::Relaxed);
        args.push("--kv-cache-ttl".to_string());
        args.push(ttl.to_string());

        // Pass prefix-cache config from the active inference settings.
        {
            use std::sync::atomic::Ordering;
            args.push("--prefix-cache-enabled".to_string());
            args.push(
                self.prefix_cache_enabled
                    .load(Ordering::Relaxed)
                    .to_string(),
            );
            args.push("--prefix-cache-max-entries".to_string());
            args.push(
                self.prefix_cache_max_entries
                    .load(Ordering::Relaxed)
                    .to_string(),
            );
            args.push("--prefix-cache-max-prompt-tokens".to_string());
            args.push(
                self.prefix_cache_max_prompt_tokens
                    .load(Ordering::Relaxed)
                    .to_string(),
            );
            args.push("--prefix-cache-max-mb".to_string());
            args.push(self.prefix_cache_max_mb.load(Ordering::Relaxed).to_string());
            args.push("--prefix-cache-block-tokens".to_string());
            args.push(
                self.prefix_cache_block_tokens
                    .load(Ordering::Relaxed)
                    .to_string(),
            );
            args.push("--prefix-cache-min-tokens".to_string());
            args.push(
                self.prefix_cache_min_tokens
                    .load(Ordering::Relaxed)
                    .to_string(),
            );
            args.push("--swift-self-speculative".to_string());
            args.push(
                self.swift_self_speculative
                    .load(Ordering::Relaxed)
                    .to_string(),
            );
            args.push("--swift-calibration-tokens".to_string());
            args.push(
                self.swift_calibration_tokens
                    .load(Ordering::Relaxed)
                    .to_string(),
            );
            args.push("--swift-gamma".to_string());
            args.push(self.swift_gamma.load(Ordering::Relaxed).to_string());
            args.push("--swift-skip-ratio".to_string());
            let ratio = self.swift_skip_ratio_milli.load(Ordering::Relaxed) as f32 / 1000.0;
            args.push(format!("{ratio}"));
            args.push("--force-standard-attn".to_string());
            args.push(self.force_standard_attn.load(Ordering::Relaxed).to_string());
            let cap = self.max_seq_len_override.load(Ordering::Relaxed);
            if cap > 0 {
                args.push("--max-seq-len-override".to_string());
                args.push(cap.to_string());
            }
            args.push("--activation-compression".to_string());
            args.push(
                self.activation_compression
                    .load(Ordering::Relaxed)
                    .to_string(),
            );
            // BatchGenerate (Item 7): enabled iff the same `continuous_batching`
            // flag that gates daemon-side `forward()` coalescing is on.
            args.push("--batch-generate".to_string());
            args.push(self.continuous_batching.load(Ordering::Relaxed).to_string());
            args.push("--batch-generate-max-slots".to_string());
            args.push(
                self.max_concurrent_decode_batch
                    .load(Ordering::Relaxed)
                    .to_string(),
            );
            args.push("--prefill-chunk-tokens".to_string());
            args.push(
                self.prefill_chunk_tokens
                    .load(Ordering::Relaxed)
                    .to_string(),
            );
            args.push("--ngram-max-size".to_string());
            args.push(self.ngram_max_size.load(Ordering::Relaxed).to_string());
            args.push("--ngram-pred-tokens".to_string());
            args.push(self.ngram_pred_tokens.load(Ordering::Relaxed).to_string());
            args.push("--prefill-target-ms".to_string());
            args.push(self.prefill_target_ms.load(Ordering::Relaxed).to_string());
            args.push("--batched-prefill-forward".to_string());
            args.push(
                self.batched_prefill_forward
                    .load(Ordering::Relaxed)
                    .to_string(),
            );
        }

        // Device placement. Before R146 this never reached the worker at all:
        // the split loader called `Device::cuda_if_available(0)` unconditionally
        // and `inference.gpu_layers` was read only by the legacy llama.cpp
        // executor, so a CUDA build ignored the setting entirely.
        args.push("--gpu-layers".to_string());
        // The placement the caller decided — the ONE statement of where this
        // worker is going, used for both the spawn argument and the handle.
        args.push(
            match (placed_on_cpu_because, gpu_layers_override) {
                (Some(_), _) => 0,
                // A partial split: the worker places this many of its layers on
                // the card and the rest on the processor. Reaches the loader
                // through `GPU_LAYER_LIMIT`, the same route `--kv-budget-bytes`
                // takes — the daemon decides, the worker obeys.
                (None, Some((n, _))) => n as i32,
                (None, None) => self.gpu_layers.load(std::sync::atomic::Ordering::Relaxed),
            }
            .to_string(),
        );
        if let Some(b) = self.cpu_kv_budget_bytes.get(model_id) {
            args.push("--kv-budget-bytes".to_string());
            args.push(b.value().to_string());
        }

        // If a shard window is set for this model, pass it to the worker
        if let Some(window) = self.active_shard_windows.get(model_id) {
            let window_str = window
                .iter()
                .map(|i| i.to_string())
                .collect::<Vec<_>>()
                .join(",");
            args.push("--shard-window".to_string());
            args.push(window_str);
        }

        // Bound the worker's CPU parallelism to what the owner agreed to give.
        //
        // candle parallelises CPU tensor ops through rayon, whose global pool
        // defaults to every logical core, and nothing narrowed it — so a single
        // request took the whole machine regardless of contribution level
        // (measured: 529-534% of 600% on a 6-core node set to Minimal). The
        // worker is a separate process, so setting this in its environment is
        // enough to size its rayon pool and cannot affect the daemon's own
        // runtime.
        //
        // Deliberately does NOT override an operator-set RAYON_NUM_THREADS:
        // someone who exported it has made exactly this decision already.
        let cpu_threads = self.cpu_threads.load(std::sync::atomic::Ordering::Relaxed);
        let mut command = tokio::process::Command::new(&exe);
        command.args(&args).kill_on_drop(true);
        if cpu_threads > 0 && std::env::var_os("RAYON_NUM_THREADS").is_none() {
            command.env("RAYON_NUM_THREADS", cpu_threads.to_string());
        }
        let child = command
            .spawn()
            .map_err(|e| SwarmError::ServiceUnavailable(format!("spawn worker: {e}")))?;

        // Wait for worker to connect
        let conn = tokio::time::timeout(
            std::time::Duration::from_secs(WORKER_CONNECT_TIMEOUT_SECS),
            listener.accept(),
        )
        .await
        .map_err(|_| SwarmError::ServiceUnavailable("worker connect timeout".into()))?
        .map_err(|e| SwarmError::ServiceUnavailable(format!("accept: {e}")))?;

        let (mut read_half, write_half) = conn.split();

        // Read Ready message
        let (ready_msg, _) = recv_worker(&mut read_half)
            .await
            .map_err(|e| SwarmError::ServiceUnavailable(format!("read ready: {e}")))?;
        match ready_msg {
            WorkerMsg::Ready => {}
            other => {
                return Err(SwarmError::ServiceUnavailable(format!(
                    "expected Ready, got {other:?}"
                )))
            }
        }

        tracing::info!(model_id = %model_id.0, "DIAG: model worker subprocess started");

        // Build a descriptive message including shard window if available
        let shard_info = self.active_shard_windows.get(model_id).map(|w| {
            let indices: Vec<_> = w.iter().map(|i| (i + 1).to_string()).collect();
            if indices.len() == 1 {
                format!("shard {}", indices[0])
            } else {
                format!("shards {}", indices.join(", "))
            }
        });
        match shard_info {
            Some(shards) => tracing::info!(
                model_id = %model_id,
                shards = %shards,
                "spawning worker"
            ),
            None => tracing::info!(model_id = %model_id, "spawning worker"),
        };

        // Success — defuse the cleanup guard on Unix; WorkerHandle now owns
        // the socket file and its Drop will unlink on process exit.
        #[cfg(unix)]
        std::mem::forget(socket_guard);
        let responses: ResponseMap = Arc::new(DashMap::new());
        let dead = Arc::new(AtomicBool::new(false));
        let reader_handle = tokio::spawn(reader_actor(
            read_half,
            responses.clone(),
            dead.clone(),
            model_id.clone(),
            self.prefix_manifest_tx.get().cloned(),
            self.prefix_probe_tx.get().cloned(),
            self.progress_tx.get().cloned(),
        ));
        Ok(WorkerHandle {
            child: Some(child),
            exited: Arc::new(AtomicBool::new(false)),
            writer: Mutex::new(write_half),
            responses,
            dead,
            #[cfg(unix)]
            socket_name,
            reader_handle,
            spawned_at: std::time::Instant::now(),
            last_used: AtomicU64::new(0),
            placed_on_cpu_because,
            charged_against_ram,
            gpu_estimate_mb,
            // Only a split is worth recording; a worker wholly on the card
            // says so through `placed_on_cpu_because: None` alone.
            gpu_layers_on_card: gpu_layers_override.filter(|_| placed_on_cpu_because.is_none()),
        })
    }

    /// Send a LayerForward to the worker, get a LayerResult back.
    /// Send a single forward. When `continuous_batching` is on and the request
    /// is schedulable (decode-only, no vision/LoRA/spec/TP/pre-embedded), routes
    /// through the auto-coalescing scheduler which collects concurrent
    /// arrivals within `batch_collection_ms` and dispatches via
    /// `forward_batch`. Otherwise goes direct.
    pub async fn forward(
        &self,
        forward: crate::types::LayerForward,
    ) -> Result<crate::types::LayerResult, SwarmError> {
        self.forward_for_request(forward, None).await
    }

    /// [`Self::forward`] for a request THIS node coordinates: the wait for the
    /// worker's answer watches the request's cancel flag and ends — telling the
    /// worker to stop — the moment it flips (`inference::cancel`).
    ///
    /// Only the wait is watched. The send is never interrupted, because a
    /// `Forward` frame written half-way would corrupt the worker's stream for
    /// every request after it; and a forward the batch scheduler takes (a
    /// decode step, one token's work) is not watched either — the per-token
    /// loop above it already reads the flag between steps.
    ///
    /// `None` is a forward with no request of ours behind it — a segment served
    /// for a remote coordinator, whose cancel arrives over the network as
    /// `CancelInference` and reaches the worker through `cancel_request`.
    pub async fn forward_for_request(
        &self,
        forward: crate::types::LayerForward,
        cancel: Option<Arc<std::sync::atomic::AtomicBool>>,
    ) -> Result<crate::types::LayerResult, SwarmError> {
        if self
            .continuous_batching
            .load(std::sync::atomic::Ordering::Relaxed)
            && forward_is_schedulable(&forward)
        {
            if let Some(tx) = self.batch_scheduler.get() {
                let (resp_tx, resp_rx) = tokio::sync::oneshot::channel();
                if tx
                    .send(BatchSchedulerMsg::Forward {
                        fwd: forward,
                        resp_tx,
                    })
                    .await
                    .is_ok()
                {
                    return resp_rx.await.unwrap_or_else(|_| {
                        Err(SwarmError::Internal(
                            "batch scheduler dropped response".into(),
                        ))
                    });
                }
                // If the scheduler channel is closed, fall through to direct.
                return Err(SwarmError::Internal(
                    "batch scheduler channel closed".into(),
                ));
            }
        }
        self.forward_direct(forward, cancel).await
    }

    async fn forward_direct(
        &self,
        forward: crate::types::LayerForward,
        cancel: Option<Arc<std::sync::atomic::AtomicBool>>,
    ) -> Result<crate::types::LayerResult, SwarmError> {
        // `None` means the caller had no request context — a segment served
        // for a REMOTE coordinator, whose parameters are not on the wire.
        // Defaulting there preserves the previous behaviour; see the field's
        // documentation on `LayerForward`.
        let forward_sampling = forward.sampling.clone().unwrap_or_default();
        let model_id = forward.model_id.clone();
        // Loading a model is the long wait nothing was watching. Every cancel
        // checkpoint sat downstream of it, so a client that gave up while its
        // segment was still loading cancelled nothing at all — the load ran on
        // and the request then went on to claim a KV cache and run a prefill
        // for nobody (gotcha #459).
        //
        // Bracketed, not wrapped: `unless_cancelled` stops a wait by dropping
        // the future, and dropping a load half-done abandons a spawning
        // subprocess — and the model may be exactly what the next request
        // wants. So a client that has already gone starts no load, the load
        // itself always finishes, and a client that leaves DURING it gets no
        // forward sent on its behalf, which is where the memory would have
        // gone.
        crate::inference::cancel::bail_if_cancelled(cancel.as_ref())?;
        let handle = self.get_or_spawn(&model_id, forward.layer_range).await?;
        crate::inference::cancel::bail_if_cancelled(cancel.as_ref())?;

        // Destructure to avoid cloning activations (can be large tensor data)
        let crate::types::LayerForward {
            request_id,
            sequence_num,
            index_pos,
            activations,
            format,
            model_id: fwd_model_id,
            layer_range,
            tp_meta,
            vision_embeddings,
            chain: _,
            sender_peer_bytes: _,
            requester_node_id,
            pre_embedded,
            generated_ids,
            adapter_id,
            draft_tokens,
            spec_logits_requested,
            truncate_kv_to,
            chunk_meta: _,
            sampling: _,
        } = forward;

        // Split vision embeddings out of the JSON header into the binary
        // payload prefix — serde_json encodes `Vec<u8>` as a JSON array of
        // integers (~5× bloat) and can push the header past `MAX_HEADER`.
        // Layout: `[vision_bytes][activation_bytes]` with
        // `vision_embeddings_len` recording the prefix length.
        let (vision_prefix, vision_len) = match vision_embeddings {
            Some(bytes) => {
                let len = u32::try_from(bytes.len()).map_err(|_| {
                    SwarmError::Internal("vision embeddings > u32::MAX bytes".into())
                })?;
                (bytes, len)
            }
            None => (Vec::new(), 0u32),
        };

        let ipc_fwd = IpcForward {
            request_id,
            sequence_num,
            index_pos,
            format,
            model_id: fwd_model_id,
            layer_range,
            tp_meta,
            vision_embeddings_len: vision_len,
            requester_node_id,
            pre_embedded,
            sampling: forward_sampling,
            adapter_id,
            draft_tokens,
            generated_ids,
            spec_logits_requested,
            truncate_kv_to,
        };

        if handle.dead.load(Ordering::Acquire) {
            self.retire_dead_worker(&model_id).await;
            return Err(SwarmError::ServiceUnavailable("worker is dead".into()));
        }

        // Register a response channel BEFORE sending so the reader actor can
        // route any early error/reply. Unregistered on drop via ResponseGuard.
        let (resp_tx, mut resp_rx) = mpsc::channel(RESPONSE_CHANNEL_CAPACITY);
        let (mut guard, _) = handle.register_response(request_id, resp_tx, true);

        let payload_buf: Vec<u8> = if vision_len == 0 {
            activations
        } else {
            let mut buf = Vec::with_capacity(vision_prefix.len() + activations.len());
            buf.extend_from_slice(&vision_prefix);
            buf.extend_from_slice(&activations);
            buf
        };

        {
            let mut writer = handle.writer.lock().await;
            if let Err(e) =
                send_daemon(&mut *writer, &DaemonMsg::Forward(ipc_fwd), &payload_buf).await
            {
                drop(writer);
                self.evict_this_worker(&model_id, &handle, "IPC send failed");
                tracing::warn!(model = %model_id, error = %e, "Worker send failed — evicting dead worker");
                // The worker never received the request; nothing to cancel.
                guard.disarm();
                return Err(SwarmError::ServiceUnavailable(format!("send Forward: {e}")));
            }
        }

        // The wait — and ONLY the wait — ends early when the request is
        // cancelled. Returning from `unless_cancelled` with the request
        // abandoned drops this block with `guard` still armed, and
        // `ResponseGuard::drop` is what tells the worker: it sends
        // `CancelRequest`, the worker skips the forward if it has not started
        // and stops between layers if it has (gotcha #445 — a local segment of
        // an agent-sized prompt on a processor ran for 81 minutes after the
        // client had gone, because nothing here was watching).
        let wait = async {
            loop {
                match resp_rx.recv().await {
                    Some((msg, payload)) => match msg {
                        WorkerMsg::LayerResult(r) if r.request_id == request_id => {
                            let (activations, spec_logits) = reconstruct_layer_payload(
                                r.has_activations,
                                r.has_spec_logits,
                                r.spec_logits_dims,
                                payload,
                            )?;
                            guard.disarm();
                            return Ok(crate::types::LayerResult {
                                request_id: r.request_id,
                                token_ids: r.token_ids,
                                finish_reason: r.finish_reason,
                                activations,
                                sealed_token_ids: if r.sealed { r.sealed_payload } else { None },
                                spec_logits,
                                matched_stop_sequence: r.matched_stop_sequence,
                                token_logprobs: r.logprobs.unwrap_or_default(),
                            });
                        }
                        WorkerMsg::Error {
                            request_id: rid,
                            message,
                            fatal,
                        } if rid == request_id => {
                            // Worker already reached a terminal state for this id.
                            guard.disarm();
                            return Err(self.classify_worker_error(&model_id, message, fatal));
                        }
                        _ => continue,
                    },
                    None => {
                        // Reader actor closed the channel — worker died while we were waiting.
                        // Subprocess lifecycle failure → ServiceUnavailable (per
                        // .claude/rules/completeness.md); Internal is for code bugs.
                        self.evict_this_worker(&model_id, &handle, "closed its connection");
                        guard.disarm();
                        return Err(SwarmError::ServiceUnavailable(
                            "worker closed connection before reply".into(),
                        ));
                    }
                }
            }
        };
        crate::inference::cancel::unless_cancelled(wait, cancel.as_ref()).await
    }

    /// Batched forward: send N forwards in a single `DaemonMsg::BatchForward`
    /// message. Worker runs them through the fused `SplitModel::forward_batch`
    /// when they're compatible (same model/layer_range, decode-only, no vision
    /// /LoRA/spec/TP) for a real matmul-fusion speedup, otherwise falls back
    /// to the sequential path internally. Either way, callers get one
    /// `LayerResult` per input in the same order.
    ///
    /// All inputs must target the same `model_id`. Returns an error if the
    /// batch is empty or targets mixed models.
    pub async fn forward_batch(
        &self,
        forwards: Vec<crate::types::LayerForward>,
    ) -> Result<Vec<crate::types::LayerResult>, SwarmError> {
        if forwards.is_empty() {
            return Err(SwarmError::Validation("forward_batch: empty input".into()));
        }
        let model_id = forwards[0].model_id.clone();
        if forwards.iter().any(|f| f.model_id != model_id) {
            return Err(SwarmError::Validation(
                "forward_batch: all inputs must share model_id".into(),
            ));
        }
        // Every forward in a batch shares a model AND a layer range — see
        // `batch_eligible`, which is what put them in one batch.
        let handle = self
            .get_or_spawn(&model_id, forwards[0].layer_range)
            .await?;
        if handle.dead.load(Ordering::Acquire) {
            self.retire_dead_worker(&model_id).await;
            return Err(SwarmError::ServiceUnavailable("worker is dead".into()));
        }

        // Register one response channel per request_id BEFORE sending.
        type SlotRx = mpsc::Receiver<(WorkerMsg, Vec<u8>)>;
        let n = forwards.len();
        let mut receivers: Vec<(Uuid, SlotRx)> = Vec::with_capacity(n);
        let mut guards: Vec<ResponseGuard> = Vec::with_capacity(n);
        for f in &forwards {
            let (tx, rx) = mpsc::channel(RESPONSE_CHANNEL_CAPACITY);
            // A BatchForward is one fused matmul over N requests; the worker
            // cannot skip a single member, so cancelling one is meaningless.
            // Matches `cancelled_request_id`'s exclusion.
            let (g, _) = handle.register_response(f.request_id, tx, false);
            guards.push(g);
            receivers.push((f.request_id, rx));
        }

        // Build IPC requests + concatenated activation payload.
        let mut ipc_requests: Vec<IpcForward> = Vec::with_capacity(n);
        let mut activation_lens: Vec<u32> = Vec::with_capacity(n);
        let mut concat_payload: Vec<u8> = Vec::new();
        for f in forwards {
            let crate::types::LayerForward {
                request_id,
                sequence_num,
                index_pos,
                activations,
                format,
                model_id,
                layer_range,
                tp_meta,
                // `forward_is_schedulable` / `batch_eligible` reject vision
                // forwards before they reach this path, so we assert here
                // rather than forward anything — any non-None value would
                // imply a scheduler bug.
                vision_embeddings: _,
                chain: _,
                sender_peer_bytes: _,
                requester_node_id,
                pre_embedded,
                generated_ids,
                adapter_id,
                draft_tokens,
                spec_logits_requested,
                truncate_kv_to,
                chunk_meta: _,
                // Per-item: a batch can carry forwards from different requests,
                // so the parameters travel with each one rather than being
                // taken from the batch head.
                sampling,
            } = f;
            let forward_sampling = sampling.unwrap_or_default();
            activation_lens.push(activations.len() as u32);
            concat_payload.extend_from_slice(&activations);
            ipc_requests.push(IpcForward {
                request_id,
                sequence_num,
                index_pos,
                format,
                model_id,
                layer_range,
                tp_meta,
                vision_embeddings_len: 0,
                requester_node_id,
                pre_embedded,
                sampling: forward_sampling,
                adapter_id,
                draft_tokens,
                generated_ids,
                spec_logits_requested,
                truncate_kv_to,
            });
        }

        {
            let mut writer = handle.writer.lock().await;
            if let Err(e) = send_daemon(
                &mut *writer,
                &DaemonMsg::BatchForward {
                    requests: ipc_requests,
                    activation_lens,
                },
                &concat_payload,
            )
            .await
            {
                drop(writer);
                self.evict_this_worker(&model_id, &handle, "IPC send failed");
                tracing::warn!(model = %model_id, error = %e, "Worker send failed — evicting dead worker");
                return Err(SwarmError::ServiceUnavailable(format!(
                    "send BatchForward: {e}"
                )));
            }
        }

        // Collect results in the original request order. Each receiver fires
        // exactly once (LayerResult or Error) before being dropped.
        let mut results: Vec<crate::types::LayerResult> = Vec::with_capacity(n);
        for (rid, mut rx) in receivers {
            loop {
                match rx.recv().await {
                    Some((WorkerMsg::LayerResult(r), payload)) if r.request_id == rid => {
                        let (activations, spec_logits) = reconstruct_layer_payload(
                            r.has_activations,
                            r.has_spec_logits,
                            r.spec_logits_dims,
                            payload,
                        )?;
                        results.push(crate::types::LayerResult {
                            request_id: r.request_id,
                            token_ids: r.token_ids,
                            finish_reason: r.finish_reason,
                            activations,
                            sealed_token_ids: if r.sealed { r.sealed_payload } else { None },
                            spec_logits,
                            matched_stop_sequence: r.matched_stop_sequence,
                            token_logprobs: r.logprobs.unwrap_or_default(),
                        });
                        break;
                    }
                    Some((
                        WorkerMsg::Error {
                            request_id: err_rid,
                            message,
                            fatal,
                        },
                        _,
                    )) if err_rid == rid => {
                        return Err(self.classify_worker_error(&model_id, message, fatal));
                    }
                    Some(_) => continue,
                    None => {
                        // Subprocess lifecycle failure → ServiceUnavailable
                        // (mirrors the single-forward arm above).
                        self.evict_this_worker(&model_id, &handle, "closed its connection");
                        return Err(SwarmError::ServiceUnavailable(
                            "worker closed connection during batch forward".into(),
                        ));
                    }
                }
            }
        }
        drop(guards);
        Ok(results)
    }

    /// Run full generation in the worker, streaming tokens back.
    #[allow(clippy::too_many_arguments)]
    pub async fn generate(
        &self,
        model_id: &ModelId,
        layer_range: (u32, u32),
        prompt: String,
        sampling: SamplingParams,
        request_id: uuid::Uuid,
        session_id: Option<String>,
        token_tx: Option<crate::inference::router::StreamingTokenTx>,
    ) -> Result<crate::inference::router::InferenceOutput, SwarmError> {
        // Track whether any token has been streamed to the caller, so a retry
        // can never duplicate output the client already saw.
        let emitted = std::sync::atomic::AtomicBool::new(false);
        let was_pinned = self.is_cpu_pinned(model_id);
        let first = self
            .generate_attempt(
                model_id,
                layer_range,
                prompt.clone(),
                sampling.clone(),
                request_id,
                session_id.clone(),
                token_tx.clone(),
                &emitted,
            )
            .await;
        match first {
            Ok(out) => Ok(out),
            Err(e) => {
                // A GPU OOM on the first (cold) request kills the worker and pins
                // the model to CPU. The request that TRIGGERED the OOM used to eat
                // the failure (0 tokens) while the very next request succeeded on
                // CPU. Retry that same request once — now it loads on CPU. The OOM
                // happens at load/prefill, so nothing was streamed yet (guarded by
                // `emitted`), making the retry duplicate-free (field report,
                // 2026-07-23, RTX 4050 6 GB).
                let freshly_pinned = !was_pinned && self.is_cpu_pinned(model_id);
                if freshly_pinned
                    && !emitted.load(std::sync::atomic::Ordering::Relaxed)
                    && matches!(e, SwarmError::ServiceUnavailable(_))
                {
                    tracing::warn!(
                        model = %model_id,
                        "Retrying request on CPU after a GPU out-of-memory auto-pinned the model"
                    );
                    let emitted_retry = std::sync::atomic::AtomicBool::new(false);
                    self.generate_attempt(
                        model_id,
                        layer_range,
                        prompt,
                        sampling,
                        request_id,
                        session_id,
                        token_tx,
                        &emitted_retry,
                    )
                    .await
                } else {
                    Err(e)
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    async fn generate_attempt(
        &self,
        model_id: &ModelId,
        layer_range: (u32, u32),
        prompt: String,
        sampling: SamplingParams,
        request_id: uuid::Uuid,
        session_id: Option<String>,
        token_tx: Option<crate::inference::router::StreamingTokenTx>,
        emitted: &std::sync::atomic::AtomicBool,
    ) -> Result<crate::inference::router::InferenceOutput, SwarmError> {
        let handle = self.get_or_spawn(model_id, layer_range).await?;

        // Kept for the post-generation stop-marker trim below; `sampling` is
        // moved into the IPC message.
        let stop_sequences = sampling.stop.clone();
        // Same reason as `stop_sequences`: `sampling` is moved below, and the
        // short-reply report needs to tell "asked for one token" from "stopped
        // after one".
        let requested_max_tokens = sampling.max_tokens;

        let gen = IpcGenerate {
            request_id,
            model_id: model_id.clone(),
            layer_range,
            prompt,
            sampling,
            session_id,
        };

        if handle.dead.load(Ordering::Acquire) {
            self.retire_dead_worker(model_id).await;
            return Err(SwarmError::ServiceUnavailable("worker is dead".into()));
        }

        let (resp_tx, mut resp_rx) = mpsc::channel(RESPONSE_CHANNEL_CAPACITY);
        let (mut guard, _) = handle.register_response(request_id, resp_tx, true);
        let attempt_token = guard.token;

        {
            let mut writer = handle.writer.lock().await;
            if let Err(e) = send_daemon(&mut *writer, &DaemonMsg::Generate(gen), &[]).await {
                drop(writer);
                self.evict_this_worker(model_id, &handle, "IPC send failed");
                tracing::warn!(model = %model_id, error = %e, "Worker send failed — evicting dead worker");
                // The worker never received the request; nothing to cancel.
                guard.disarm();
                return Err(SwarmError::ServiceUnavailable(format!(
                    "send Generate: {e}"
                )));
            }
        }

        let mut content = String::new();
        #[allow(unused_assignments)]
        let mut prompt_tokens = 0u32;
        #[allow(unused_assignments)]
        let mut completion_tokens = 0u32;
        #[allow(unused_assignments)]
        let mut finish_reason = String::new();
        let mut token_logprobs: Vec<crate::inference::router::TokenLogProbEntry> = Vec::new();
        #[allow(unused_assignments)]
        let mut matched_stop_sequence: Option<String> = None;

        loop {
            let (msg, _) = match resp_rx.recv().await {
                Some(v) => v,
                None => {
                    guard.disarm();
                    // A closed channel has two very different causes. If a
                    // later attempt has taken over this request_id, the worker
                    // is fine and is still computing for that attempt —
                    // evicting it here would destroy a healthy worker (forcing
                    // a full model reload) and abort the attempt that replaced
                    // us.
                    if handle.response_superseded(request_id, attempt_token) {
                        return Err(SwarmError::ServiceUnavailable(
                            "superseded by a retry of the same request".into(),
                        ));
                    }
                    // Subprocess lifecycle failure → ServiceUnavailable.
                    // Was Internal; operators saw 500s here and misattributed
                    // them to code bugs rather than worker crashes.
                    self.evict_this_worker(model_id, &handle, "closed its connection");
                    return Err(SwarmError::ServiceUnavailable(
                        "worker closed connection mid-generate".into(),
                    ));
                }
            };
            match msg {
                WorkerMsg::Token {
                    request_id: rid,
                    text,
                    is_eos,
                    logprob,
                    token_id: _,
                } if rid == request_id => {
                    if !is_eos {
                        content.push_str(&text);
                        if let Some(lp) = logprob {
                            token_logprobs.push(crate::inference::router::TokenLogProbEntry {
                                token: text.clone(),
                                logprob: lp,
                                top_logprobs: Vec::new(),
                            });
                        }
                        if let Some(ref tx) = token_tx {
                            // Mark output as delivered so the OOM CPU-retry in the
                            // `generate` wrapper can't re-run and duplicate tokens.
                            emitted.store(true, std::sync::atomic::Ordering::Relaxed);
                            let _ = tx
                                .send(StreamingTokenEvent {
                                    text: text.clone(),
                                    finish_reason: None,
                                    matched_stop_sequence: None,
                                })
                                .await;
                        }
                    }
                }
                WorkerMsg::GenerateDone {
                    request_id: rid,
                    prompt_tokens: pt,
                    completion_tokens: ct,
                    finish_reason: fr,
                    matched_stop_sequence: ms,
                } if rid == request_id => {
                    prompt_tokens = pt as u32;
                    completion_tokens = ct as u32;
                    finish_reason = fr;
                    matched_stop_sequence = ms;
                    // GenerateDone is terminal — the worker is finished.
                    guard.disarm();
                    break;
                }
                WorkerMsg::Error {
                    request_id: rid,
                    message,
                    fatal,
                } if rid == request_id => {
                    guard.disarm();
                    return Err(self.classify_worker_error(model_id, message, fatal));
                }
                _ => continue,
            }
        }

        if let Some(ref tx) = token_tx {
            let _ = tx
                .send(StreamingTokenEvent {
                    text: String::new(),
                    finish_reason: Some(finish_reason.clone()),
                    matched_stop_sequence: matched_stop_sequence.clone(),
                })
                .await;
        }

        // Strip a trailing PARTIAL stop marker before returning.
        //
        // The worker checks stop strings against its accumulated text and
        // withholds the token that completes a match — but a marker like
        // `<|im_end|>` is decoded across several tokens, and the earlier pieces
        // (`<|im_end`, `|`) match nothing yet, so they were already sent. The
        // consumer therefore ends up holding a partial marker.
        //
        // `local_exec.rs` and `distributed.rs` already do this; the split path
        // — everything routed through a worker subprocess, which is every
        // locally-held model — did not, which is why a Llama-3.2 GGUF emitting
        // ChatML markers returned a bare `<|im_end|` as its entire reply across
        // three releases that each claimed to fix it (external report
        // 2026-07-25).
        //
        // Also trims a COMPLETE marker: `find_stop_sequence` uses `contains`,
        // so a match can sit anywhere in the accumulated text, and the token
        // that completed it is withheld rather than the marker being removed.
        crate::inference::finalize_reply_text(&mut content, &stop_sequences);
        crate::inference::report_short_reply(
            &request_id,
            completion_tokens,
            requested_max_tokens,
            matched_stop_sequence.as_deref(),
        );

        Ok(crate::inference::router::InferenceOutput {
            request_id,
            content,
            prompt_tokens,
            completion_tokens,
            finish_reason,
            session_id: None,
            token_logprobs,
            matched_stop_sequence,
            trace: None,
        })
    }

    /// Turn a `WorkerMsg::Error` into a `SwarmError`, evicting the worker first
    /// when the failure was fatal to its device state.
    ///
    /// Before this existed, a worker that hit a CUDA OOM mid-forward reported
    /// the error and stayed resident holding its full VRAM allocation — the
    /// only two code paths that ever killed a worker were explicit unloads.
    /// Each retry then had *less* free VRAM than the last, so a single OOM
    /// reliably cascaded into permanent failure for that model until someone
    /// killed the process by hand.
    ///
    /// Eviction drops the last `Arc<WorkerHandle>` once the caller's clone goes
    /// out of scope, and `WorkerHandle::Drop` kills the child, which is what
    /// actually returns the VRAM to the OS.
    fn classify_worker_error(
        &self,
        model_id: &ModelId,
        message: String,
        fatal_flag: bool,
    ) -> SwarmError {
        // Trust the worker's own verdict, but re-derive it from the message as
        // well: a worker binary older than the `fatal` field always sends
        // `false`, and a stranded worker is much worse than a needless respawn.
        if fatal_flag || crate::inference::worker_ipc::worker_error_is_fatal(&message) {
            // No handle in scope here, so this evicts whatever is under the key
            // — the behaviour it has always had. The release is the new part.
            self.evict_worker_where(model_id, |_| true, "fatal error");
            tracing::warn!(
                model = %model_id,
                error = %message,
                "Fatal worker error — killing worker to reclaim its device memory"
            );

            // Some GPU failures repeat verbatim on the respawn — same model,
            // same device, same outcome — so retrying on the GPU just burns
            // another load. Pin the model to the CPU: slower, but it answers.
            let was_on_gpu = self.effective_gpu_layers(model_id) != 0;
            let permanent = crate::inference::worker_ipc::permanent_gpu_failure(&message);
            if let (true, Some(kind)) = (was_on_gpu, permanent) {
                use crate::inference::worker_ipc::PermanentGpuFailure;
                self.cpu_pinned_models.insert(model_id.clone());
                let reason = match kind {
                    PermanentGpuFailure::OutOfMemory => "GPU out of memory",
                    PermanentGpuFailure::ArchitectureTooOld => {
                        "GPU is older than this build's kernels"
                    }
                    PermanentGpuFailure::NoKvCacheRoom => {
                        "no GPU memory left for the conversation cache"
                    }
                };
                tracing::warn!(
                    model = %model_id,
                    reason,
                    "Pinning this model to CPU until graphics memory frees up"
                );
                if let Some(tx) = self.activity_tx.get() {
                    // Distinct `kind` per cause, NOT one kind with two
                    // messages: the frontend translates by kind
                    // (`I18n.t('activity.' + kind)`) and only falls back to
                    // this English text when the key is missing. Reusing
                    // `model_cpu_fallback` would have told everyone outside
                    // English that they had run out of GPU memory — sending
                    // them to free VRAM that was never the problem.
                    let (event_kind, text) = match kind {
                        // `NoKvCacheRoom` deliberately SHARES this kind rather
                        // than getting its own. The rule above is that a
                        // distinct cause needs a distinct kind because the
                        // frontend translates by kind — but the reason it
                        // exists is that the arch case sends people to free
                        // VRAM that was never the problem. Here VRAM IS the
                        // problem and freeing it IS the remedy, so this text is
                        // already correct and already translated; a second key
                        // saying the same thing in 21 locales would be
                        // duplication, not precision.
                        PermanentGpuFailure::OutOfMemory | PermanentGpuFailure::NoKvCacheRoom => (
                            "model_cpu_fallback",
                            format!(
                                "{} ran out of GPU memory — switched to CPU (slower, but working)",
                                model_id.0
                            ),
                        ),
                        // Phrased for someone who will not know what a compute
                        // capability is, and who needs to know the node is
                        // still working rather than what CUDA reported.
                        PermanentGpuFailure::ArchitectureTooOld => (
                            "model_cpu_fallback_gpu_too_old",
                            format!(
                                "This graphics card is too old for this version's GPU \
                                 support — {} switched to CPU (slower, but working)",
                                model_id.0
                            ),
                        ),
                    };
                    let _ = tx.send(
                        crate::daemon::state::ActivityEvent::new("inference", event_kind, text)
                            .with_model(model_id.0.clone())
                            .with_toast("warning", 8000),
                    );
                }
            }

            // Subprocess lifecycle failure → ServiceUnavailable, not Internal:
            // this server can't serve right now, but it isn't a code bug.
            return SwarmError::ServiceUnavailable(format!("worker fatal error: {message}"));
        }

        // A worker-side VALIDATION failure is the caller's input, not our bug.
        //
        // The worker raises a real `SwarmError::Validation` for things like a
        // prompt longer than the model's context window, but crossing the IPC
        // boundary flattens it to a string — so everything arrived here as
        // `Inference`, which maps to HTTP 500 `server_error`. A prompt that is
        // too long is the one thing the user CAN fix, and telling them the
        // server broke is both wrong and actively harmful: any client with
        // retry-on-5xx will re-send a request that cannot ever succeed.
        //
        // Re-deriving the class from the message is the same approach
        // `worker_error_is_fatal` already takes for the same reason (the typed
        // information does not survive the hop). Observed against a 2176-token
        // prompt on a 2048-context model, 2026-07-29.
        // One recovery, shared with the network boundary, which has the same
        // problem for the same reason (`crate::error::reclassify_flattened_error`).
        if let Some(recovered) = crate::error::reclassify_flattened_error(&message) {
            return recovered;
        }
        SwarmError::Inference(message)
    }

    /// Tell every live worker to abandon `request_id`.
    ///
    /// Used for cancellations that arrive over the network
    /// (`SwarmMessage::CancelInference`), where the sender knows only the
    /// request id — the coordinator's id space is global, ours is per-worker.
    /// Workers are few (one per loaded model) and a cancel for an unknown id
    /// is a no-op on the worker side, so the fan-out is cheap and safe.
    ///
    /// Locally-originated cancels don't need this: `ResponseGuard` knows its
    /// own worker and messages it directly on drop.
    pub async fn cancel_request(&self, request_id: Uuid) {
        // Collect handles before awaiting — never hold a DashMap ref across an
        // await point.
        let workers: Vec<Arc<WorkerHandle>> = self
            .workers
            .iter()
            .filter(|e| !e.value().dead.load(Ordering::Acquire))
            .map(|e| e.value().clone())
            .collect();
        if workers.is_empty() {
            return;
        }
        for worker in workers {
            let mut writer = worker.writer.lock().await;
            let _ = send_daemon(&mut *writer, &DaemonMsg::CancelRequest { request_id }, &[]).await;
        }
    }

    /// Unload all segments for a model (kills the worker subprocess).
    pub async fn unload_model(&self, model_id: &ModelId) {
        if let Some((_, handle)) = self.workers.remove(model_id) {
            // Was this worker holding GPU memory? Only then does killing it
            // change the pressure that caused any outstanding CPU pins.
            //
            // Read from the WORKER, not from `effective_gpu_layers`. That asks
            // where a spawn happening *now* would go, and its pin is cleared
            // while the processor worker it produced is still running — so a
            // model on its way back to the card answered "GPU" and lifted every
            // other model's pin on the strength of memory it never held.
            let freed_gpu_memory = handle.placed_on_cpu_because.is_none();

            // Retire by DRAINING, not by killing under a live request.
            //
            // The `remove` above already closed new admission: every forward
            // path re-acquires the handle from `workers`, so a request arriving
            // now spawns a fresh worker rather than joining this one. What is
            // left is a request that took the handle just before the remove and
            // is still in flight — and `DaemonMsg::Shutdown` below stops the
            // worker immediately, so sending it under that request kills the
            // reply. Dropping the handle does not, since the map holds an `Arc`
            // and the in-flight caller holds another; the explicit shutdown is
            // the whole race.
            //
            // Both displacement paths mitigated this with an idle floor plus a
            // not-busy check, which makes the window small without closing it
            // (`docs/FUTURE_WORK.md`). This closes it, in the one place every
            // retirement funnels through.
            let stranded =
                Self::await_responses_drained(&handle.responses, WORKER_DRAIN_WAIT).await;
            if stranded > 0 {
                tracing::warn!(
                    model_id = %model_id,
                    stranded_requests = stranded,
                    waited_ms = WORKER_DRAIN_WAIT.as_millis(),
                    "Retiring a worker with requests still in flight — they will fail; \
                     waiting longer would hold this model's memory indefinitely"
                );
            }

            // Try graceful shutdown first
            if let Ok(mut writer) = handle.writer.try_lock() {
                let _ = send_daemon(&mut *writer, &DaemonMsg::Shutdown, &[]).await;
            }
            // Dropping the handle signals the child; it does NOT wait for the
            // OS to reclaim its device memory. Hold the budget until the
            // process is genuinely gone, or the next admission decides against
            // memory that is still occupied and its worker dies with
            // CUDA_ERROR_OUT_OF_MEMORY — the exact failure admission exists to
            // prevent.
            let exited = handle.exited.clone();
            drop(handle);
            let freed_cleanly = Self::await_worker_exit(&exited, WORKER_EXIT_WAIT).await;
            if !freed_cleanly {
                // Release anyway. Holding a charge forever for a process that
                // will not die would refuse every later load on this device,
                // which is a worse failure than the race being closed here.
                tracing::warn!(
                    model_id = %model_id,
                    waited_ms = WORKER_EXIT_WAIT.as_millis(),
                    "Worker did not exit in time — freeing its memory budget anyway; \
                     a load starting now could still find the device occupied"
                );
            }
            self.after_worker_gone(model_id, freed_gpu_memory, "stopped");
        }
    }

    /// Everything that follows a worker's process no longer existing, however
    /// it stopped: release both memory budgets, and — if it was holding
    /// graphics memory — lift the CPU pins that its occupancy was the reason
    /// for.
    ///
    /// **One place, because the two halves are one event and were implemented
    /// separately.** `unload_model` did both; `retire_dead_worker` (added the
    /// same day, for gotcha #461) did only the first, so a GPU worker that
    /// CRASHED or was OOM-killed freed its card and left every other model
    /// pinned to the processor — the ~10x throughput loss gotcha #401 exists to
    /// prevent, reachable by a route that fix never covered. A crash frees the
    /// card exactly as a graceful unload does; the pin's own clearing condition
    /// ("GPU memory freed") does not care which happened.
    fn after_worker_gone(&self, model_id: &ModelId, freed_gpu_memory: bool, how: &str) {
        self.release_vram_charge(model_id);
        // The process is gone, so it holds neither device's memory. A CPU
        // worker never had a VRAM charge and vice versa; releasing both is
        // correct and keeps the two budgets from drifting on churn.
        self.release_ram_charge(model_id);
        tracing::info!(
            model_id = %model_id,
            device = if freed_gpu_memory { "gpu" } else { "cpu" },
            how,
            "Model worker stopped and its memory budget released"
        );

        // A GPU OOM pins its model to the CPU, and that pin used to last for
        // the life of the process — a ~10x throughput loss that no API
        // response mentioned, triggered by nothing more than the daemon's
        // own background model churn. The pin's reasoning ("the OOM will
        // repeat verbatim on respawn") only held while VRAM pressure was
        // static, which it was, because eviction never actually freed
        // anything. Now that it does, releasing memory is exactly the event
        // that makes a retry worth it — the same condition
        // `clear_cpu_pin`'s own documentation names.
        //
        // Worst case a pinned model retries the GPU, OOMs again and re-pins,
        // costing one model load. That is a far better trade than staying on
        // the CPU for ever.
        if freed_gpu_memory && !self.cpu_pinned_models.is_empty() {
            let lifted: Vec<String> = self
                .cpu_pinned_models
                .iter()
                .map(|m| m.key().0.clone())
                .collect();
            self.cpu_pinned_models.clear();
            tracing::info!(
                freed_by = %model_id,
                lifted = ?lifted,
                "GPU memory freed — clearing CPU pins so these models may use the GPU again"
            );
        }
    }

    /// Unload a model and clear its shard window so next spawn uses defaults.
    pub async fn unload_and_clear_window(&self, model_id: &ModelId) {
        self.unload_model(model_id).await;
        self.clear_shard_window(model_id);
    }

    /// Check if a worker is running for a model.
    pub fn is_loaded(&self, model_id: &ModelId) -> bool {
        self.workers.contains_key(model_id)
    }

    /// List all currently loaded model IDs.
    pub fn loaded_model_ids(&self) -> Vec<ModelId> {
        self.workers.iter().map(|e| e.key().clone()).collect()
    }

    /// Every worker this node has resident, as `/v1/status` and `swarmllm
    /// status` show it — sorted by model so two readings line up.
    ///
    /// **Why**: a tester found a worker at 400% CPU for 81 minutes on a request
    /// whose client had long gone, and wrote that "nothing in `swarmllm status`
    /// or the API surfaced these or offered a way to reap them" — `kill -9` was
    /// the only tool (gotcha #445). The pool is the one owner of these
    /// processes, so it is the one place that can say what they are doing:
    /// `in_flight` is the pool's own response map, which every execution path
    /// passes through (see `models_with_inflight_requests`). Retiring one is
    /// `POST /api/admin/models/{id}/unload`, which drains and then stops it.
    pub fn worker_summaries(&self) -> Vec<WorkerSummary> {
        let mut out: Vec<WorkerSummary> = self
            .workers
            .iter()
            .map(|e| {
                let h = e.value();
                let cpu_reason = h.placed_on_cpu_because.map(|r| r.as_str());
                WorkerSummary {
                    model: e.key().0.clone(),
                    pid: h.child.as_ref().and_then(|c| c.id()),
                    device: if cpu_reason.is_some() {
                        "processor"
                    } else {
                        "graphics card"
                    },
                    cpu_reason,
                    in_flight: h.responses.len(),
                    idle_secs: h.idle_secs(),
                    age_secs: h.spawned_at.elapsed().as_secs(),
                    dead: h.dead.load(Ordering::Acquire),
                    gpu_estimate_mb: h.gpu_estimate_mb,
                    gpu_layers_on_card: h.gpu_layers_on_card.map(|(n, _)| n as u32),
                    layers_total: h.gpu_layers_on_card.map(|(_, t)| t as u32),
                }
            })
            .collect();
        out.sort_by(|a, b| a.model.cmp(&b.model));
        out
    }

    /// The hybrid split a resident worker runs with — `(layers on the card,
    /// layers in total)` — or `None` when the model has no worker or its
    /// worker is wholly on one device. The models page reads this so a model
    /// running 13 of 28 layers on the card says so, instead of showing
    /// `fits_on_gpu: true` and nothing else.
    pub fn hybrid_gpu_layers(&self, model_id: &ModelId) -> Option<(usize, usize)> {
        self.workers
            .get(model_id)
            .and_then(|h| h.gpu_layers_on_card)
    }

    /// Models with at least one request in flight *right now*.
    ///
    /// Read from the pool's own per-request response channels, which is the one
    /// place EVERY execution path passes through — local, distributed, and
    /// peer-served alike. The callers' own bookkeeping does not have that
    /// property: `active_pipelines` is populated only by the distributed path
    /// (`distributed_exec` inserts; `local_exec` only removes) and
    /// `serving_models` only by peer-served work, so a node answering its own
    /// client locally appeared in NEITHER. That is how a worker was killed
    /// mid-generation seven seconds after loading (reported 2026-07-31).
    ///
    /// Anything asking "is this model busy?" should use this rather than
    /// re-deriving it from a caller-side map that covers one path.
    pub fn models_with_inflight_requests(&self) -> Vec<ModelId> {
        self.workers
            .iter()
            .filter(|e| !e.value().responses.is_empty())
            .map(|e| e.key().clone())
            .collect()
    }

    /// How long each loaded model's worker has existed. Upper bound on how long
    /// it can possibly have been idle.
    pub fn model_residency_secs(&self) -> Vec<(ModelId, u64)> {
        self.workers
            .iter()
            .map(|e| (e.key().clone(), e.value().spawned_at.elapsed().as_secs()))
            .collect()
    }

    /// Seconds since each resident worker last had a request registered
    /// against it — `WorkerHandle::last_used`, stamped in `register_response`,
    /// which every execution path passes through (local, distributed,
    /// peer-served). This is the fact the idle unload must read: the
    /// `model_trust.last_request_at` it used to rely on has NO writer in the
    /// current code, so for a model in constant local use the only evidence
    /// of use was the worker being busy at the instant of the check.
    pub fn model_idle_secs(&self) -> Vec<(ModelId, u64)> {
        self.workers
            .iter()
            .map(|e| (e.key().clone(), e.value().idle_secs()))
            .collect()
    }

    /// Restart a model's worker with a new shard window.
    /// Kills the current worker → OS/CUDA frees VRAM → next inference request
    /// triggers `get_or_spawn` which reads the new window.
    pub async fn restart_with_window(&self, model_id: &ModelId, window: Vec<u32>) {
        tracing::info!(
            model_id = %model_id,
            window = ?window,
            "Restarting worker with narrower shard window"
        );
        self.active_shard_windows.insert(model_id.clone(), window);
        // Kill the existing worker — next request will re-spawn with new window
        self.unload_model(model_id).await;
    }

    /// Clear a shard window (revert to loading all on-disk shards).
    fn clear_shard_window(&self, model_id: &ModelId) {
        self.active_shard_windows.remove(model_id);
    }

    /// Get the current shard window for a model, if any.
    pub fn get_shard_window(&self, model_id: &ModelId) -> Option<Vec<u32>> {
        self.active_shard_windows.get(model_id).map(|v| v.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The reported macOS failure, as arithmetic (2026-09-03, Mac mini M4).
    ///
    /// macOS hands every user a private per-boot temp directory; the tester
    /// measured theirs at 49 characters. The old socket name was
    /// `swarmllm-worker-` + a 36-character UUID + `.sock` = 57, so the path
    /// came to 49 + 1 + 57 = 107 against a 104-byte `sun_path` — every worker
    /// spawn failed, and with it every request. The new name is short enough
    /// that the same directory fits with room to spare.
    ///
    /// Written against the CONSTANT rather than the host so it fails on Linux
    /// too: this is exactly the class of bug a Linux-only suite cannot see.
    #[cfg(unix)]
    #[test]
    fn a_mac_style_temp_dir_fits_a_worker_socket() {
        // Trailing slash included: that is the form macOS hands out
        // (`confstr(_CS_DARWIN_USER_TEMP_DIR)`), and it is the character that
        // takes the reported directory to the measured 49.
        let tmpdir = std::path::PathBuf::from("/var/folders/8k/9mzq0l7d5xv3r1p2t6y4w8x00000gn/T/");
        assert_eq!(tmpdir.as_os_str().len(), 49, "the reported TMPDIR length");

        // Control: the name this replaced did NOT fit.
        let old_name = format!("swarmllm-worker-{}.sock", uuid::Uuid::new_v4());
        assert_eq!(old_name.len(), 57);
        assert!(
            first_dir_that_fits(std::slice::from_ref(&tmpdir), &old_name, 104).is_none(),
            "the old name must be what broke, or this test proves nothing"
        );

        // The fix: the same directory now fits, on the tightest platform.
        let name = worker_socket_filename();
        let chosen = first_dir_that_fits(&[tmpdir], &name, 104)
            .expect("a 49-character temp dir must fit a short socket name");
        assert!(chosen.as_os_str().len() < 104);
    }

    /// A directory too long for the limit is skipped for the next candidate —
    /// which is what the `/tmp/swarmllm-<uid>` fallback is for.
    #[cfg(unix)]
    #[test]
    fn an_over_long_directory_falls_through_to_a_shorter_one() {
        let long = std::path::PathBuf::from(format!("/var/folders/{}", "x".repeat(90)));
        let short = std::path::PathBuf::from("/tmp/swarmllm-501");
        let name = worker_socket_filename();
        assert_eq!(
            first_dir_that_fits(&[long.clone(), short.clone()], &name, SUN_PATH_MAX),
            Some(short.join(&name))
        );
        // And when nothing fits, nothing is invented.
        assert_eq!(first_dir_that_fits(&[long], &name, SUN_PATH_MAX), None);
    }

    /// The name stays short and stays unique — the two properties in tension.
    #[cfg(unix)]
    #[test]
    fn the_socket_filename_is_short_and_unique() {
        let a = worker_socket_filename();
        let b = worker_socket_filename();
        assert_ne!(a, b);
        assert!(a.len() <= 32, "socket filename grew: {a}");
        assert!(a.starts_with("swarmllm-") && a.ends_with(".sock"));
    }

    /// An ordinary node must not create the `/tmp` fallback it never uses.
    /// The directory is only built when the temp directory does not fit, so on
    /// Linux — where `$TMPDIR` is `/tmp` — building a path leaves nothing
    /// behind. A directory nobody needs is still a directory somebody has to
    /// explain, and this one lives in a world-writable place.
    #[cfg(all(unix, not(target_vendor = "apple")))]
    #[test]
    fn a_node_whose_temp_dir_fits_never_creates_the_fallback() {
        let uid = unsafe { libc::getuid() };
        let fallback = std::path::PathBuf::from(format!("/tmp/swarmllm-{uid}"));
        let existed = fallback.exists();
        let name = worker_socket_filename();
        assert!(
            first_dir_that_fits(
                std::slice::from_ref(&std::env::temp_dir()),
                &name,
                SUN_PATH_MAX
            )
            .is_some(),
            "control: this machine's temp dir must fit, or the test proves nothing"
        );
        let _ = worker_socket_path().expect("a path must be buildable");
        assert_eq!(
            fallback.exists(),
            existed,
            "building a socket path must not create {} when the temp dir already fits",
            fallback.display()
        );
    }

    /// On THIS machine, whatever its temp directory, a worker socket path can
    /// be built and is inside the limit. The end-to-end form of the above.
    #[cfg(unix)]
    #[test]
    fn this_machine_can_build_a_worker_socket_path() {
        let p = worker_socket_path().expect("a socket path must be buildable here");
        assert!(
            p.len() < SUN_PATH_MAX,
            "{p} is {} chars, limit {SUN_PATH_MAX}",
            p.len()
        );
    }

    /// A worker with nothing in flight retires immediately — the common case,
    /// since both displacement paths only retire idle models. If this ever
    /// sleeps, every unload pays for a race that is not happening.
    #[tokio::test]
    async fn an_idle_worker_drains_at_once() {
        let map: ResponseMap = Arc::new(DashMap::new());
        let t0 = std::time::Instant::now();
        let stranded =
            ModelProcessPool::await_responses_drained(&map, std::time::Duration::from_secs(30))
                .await;
        assert_eq!(stranded, 0);
        assert!(
            t0.elapsed() < std::time::Duration::from_millis(200),
            "an empty map must not wait; took {:?}",
            t0.elapsed()
        );
    }

    /// The point of the change: a request still in flight is WAITED for, rather
    /// than having `DaemonMsg::Shutdown` sent under it.
    #[tokio::test]
    async fn a_request_in_flight_is_waited_for() {
        let map: ResponseMap = Arc::new(DashMap::new());
        let id = Uuid::new_v4();
        map.insert(id, (1, dummy_tx()));

        let finisher = map.clone();
        tokio::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_millis(150)).await;
            finisher.remove(&id);
        });

        let t0 = std::time::Instant::now();
        let stranded =
            ModelProcessPool::await_responses_drained(&map, std::time::Duration::from_secs(30))
                .await;
        assert_eq!(stranded, 0, "it should have drained, not timed out");
        assert!(
            t0.elapsed() >= std::time::Duration::from_millis(100),
            "it must actually have waited for the in-flight request"
        );
    }

    /// ...but the wait is BOUNDED. A request that never completes must not hold
    /// this model's memory for the life of the daemon: that would refuse every
    /// later load on the device, which is worse than the race being closed.
    #[tokio::test]
    async fn a_stuck_request_does_not_block_retirement_for_ever() {
        let map: ResponseMap = Arc::new(DashMap::new());
        map.insert(Uuid::new_v4(), (1, dummy_tx()));
        map.insert(Uuid::new_v4(), (2, dummy_tx()));

        let stranded =
            ModelProcessPool::await_responses_drained(&map, std::time::Duration::from_millis(120))
                .await;
        assert_eq!(
            stranded, 2,
            "it must give up and report what it stranded, so the log can say so"
        );
    }

    /// Builds a guard the way `register_response` does, for map-level tests.
    fn guard_for(map: &ResponseMap, request_id: Uuid, token: u64) -> ResponseGuard {
        ResponseGuard {
            responses: map.clone(),
            request_id,
            token,
            worker: None,
        }
    }

    fn dummy_tx() -> ResponseTx {
        mpsc::channel(1).0
    }

    #[test]
    fn second_attempt_on_same_request_id_displaces_the_first() {
        let map: ResponseMap = Arc::new(DashMap::new());
        let rid = Uuid::new_v4();
        let (t1, d1) = claim_response_slot(&map, rid, dummy_tx());
        assert_eq!(d1, None, "first claim displaces nothing");
        let (t2, d2) = claim_response_slot(&map, rid, dummy_tx());
        assert_eq!(d2, Some(t1), "retry reports the attempt it superseded");
        assert_ne!(t1, t2, "attempts get distinct tokens");
    }

    #[test]
    fn superseded_attempt_is_distinguishable_from_a_dead_worker() {
        let map: ResponseMap = Arc::new(DashMap::new());
        let rid = Uuid::new_v4();
        let (t1, _) = claim_response_slot(&map, rid, dummy_tx());
        // Nobody has taken over yet: a closed channel here really is a dead worker.
        assert!(!response_slot_superseded(&map, rid, t1));
        let (t2, _) = claim_response_slot(&map, rid, dummy_tx());
        // Now attempt 1 is superseded, attempt 2 is not.
        assert!(response_slot_superseded(&map, rid, t1));
        assert!(!response_slot_superseded(&map, rid, t2));
    }

    /// The live failure: a retry displaced the original, then the original's
    /// cleanup removed the *retry's* channel, so both attempts died and a
    /// healthy worker was evicted.
    #[test]
    fn superseded_attempt_cleanup_does_not_evict_its_successor() {
        let map: ResponseMap = Arc::new(DashMap::new());
        let rid = Uuid::new_v4();
        let (t1, _) = claim_response_slot(&map, rid, dummy_tx());
        let g1 = guard_for(&map, rid, t1);
        let (t2, _) = claim_response_slot(&map, rid, dummy_tx());

        drop(g1); // attempt 1 gives up after being displaced

        let held = map.get(&rid).map(|e| e.value().0);
        assert_eq!(held, Some(t2), "the retry's channel must survive");
    }

    #[test]
    fn last_attempt_cleanup_clears_the_slot() {
        let map: ResponseMap = Arc::new(DashMap::new());
        let rid = Uuid::new_v4();
        let (t1, _) = claim_response_slot(&map, rid, dummy_tx());
        let (t2, _) = claim_response_slot(&map, rid, dummy_tx());
        drop(guard_for(&map, rid, t1));
        drop(guard_for(&map, rid, t2));
        assert!(map.get(&rid).is_none(), "no entry may be leaked");
    }

    #[test]
    fn unrelated_request_ids_are_untouched() {
        let map: ResponseMap = Arc::new(DashMap::new());
        let (a, b) = (Uuid::new_v4(), Uuid::new_v4());
        let (ta, _) = claim_response_slot(&map, a, dummy_tx());
        let (tb, _) = claim_response_slot(&map, b, dummy_tx());
        drop(guard_for(&map, a, ta));
        assert!(map.get(&a).is_none());
        assert_eq!(map.get(&b).map(|e| e.value().0), Some(tb));
    }

    fn test_pool() -> ModelProcessPool {
        ModelProcessPool::new(std::path::PathBuf::from("/tmp/swarmllm-test-pool"))
    }

    /// Build a real `WorkerHandle` over a loopback socket pair, so the
    /// retirement path can be exercised without spawning a subprocess.
    /// `child: None` — `Drop` skips the kill and marks `exited` directly.
    async fn fake_worker_handle(dead_now: bool) -> Arc<WorkerHandle> {
        fake_worker_handle_on(dead_now, Some(CpuReason::Configured)).await
    }

    /// `placed_on_cpu_because: None` means the worker was holding the card.
    async fn fake_worker_handle_on(
        dead_now: bool,
        placed_on_cpu_because: Option<CpuReason>,
    ) -> Arc<WorkerHandle> {
        use interprocess::local_socket::{tokio::prelude::*, ListenerOptions};
        let name = format!(
            "/tmp/swarmllm-retire-test-{}.sock",
            uuid::Uuid::new_v4().simple()
        );
        let _ = std::fs::remove_file(&name);
        let sockname = name
            .clone()
            .to_fs_name::<interprocess::local_socket::GenericFilePath>()
            .expect("fs name");
        let listener = ListenerOptions::new()
            .name(sockname.clone())
            .create_tokio()
            .expect("listen");
        let connect = tokio::spawn(async move {
            interprocess::local_socket::tokio::Stream::connect(sockname).await
        });
        let server = listener.accept().await.expect("accept");
        let client = connect.await.expect("join").expect("connect");
        // Keep the client end alive for the handle's lifetime.
        std::mem::forget(client);
        let (read_half, write_half) = server.split();
        let responses: ResponseMap = Arc::new(DashMap::new());
        let dead = Arc::new(AtomicBool::new(dead_now));
        let reader_handle = tokio::spawn(async move {
            let _ = read_half;
            std::future::pending::<()>().await;
        });
        Arc::new(WorkerHandle {
            child: None,
            exited: Arc::new(AtomicBool::new(true)),
            writer: Mutex::new(write_half),
            responses,
            dead,
            #[cfg(unix)]
            socket_name: name,
            reader_handle,
            spawned_at: std::time::Instant::now(),
            last_used: AtomicU64::new(0),
            placed_on_cpu_because,
            charged_against_ram: true,
            gpu_estimate_mb: 0,
            gpu_layers_on_card: None,
        })
    }

    /// Report #008: a worker that exits any way other than a graceful unload
    /// used to keep its whole reservation charged for the daemon's lifetime.
    /// The charge is one shared budget, so the phantom refused every later
    /// load on the node — quoting memory that was in fact free.
    #[tokio::test]
    async fn a_worker_that_died_gives_its_memory_budget_back() {
        let p = test_pool();
        p.set_ram_budget_mb(8000);
        let m = ModelId("dead-one".into());
        assert!(p.admit_to_cpu(&m, 5000), "the model is admitted");
        assert_eq!(p.ram_committed_mb(), 5000);

        p.workers.insert(m.clone(), fake_worker_handle(true).await);
        assert!(p.retire_dead_worker(&m).await, "a dead worker is retired");

        assert_eq!(
            p.ram_committed_mb(),
            0,
            "the dead worker's charge must be released — before this fix the \
             three `dead` call sites removed the map entry and nothing else, \
             so the budget stayed spent until the daemon restarted"
        );
        assert!(p.workers.get(&m).is_none());
        // And the budget is genuinely usable again.
        assert!(p.admit_to_cpu(&ModelId("next".into()), 5000));
    }

    /// The control: a LIVE worker is never retired, and its charge stands.
    /// Without it the test above would pass on code that released
    /// unconditionally, which would free memory a running worker still holds.
    #[tokio::test]
    async fn a_live_worker_keeps_its_memory_budget() {
        let p = test_pool();
        p.set_ram_budget_mb(8000);
        let m = ModelId("live-one".into());
        assert!(p.admit_to_cpu(&m, 5000));
        p.workers.insert(m.clone(), fake_worker_handle(false).await);

        assert!(
            !p.retire_dead_worker(&m).await,
            "a live worker is not retired"
        );
        assert_eq!(p.ram_committed_mb(), 5000);
        assert!(p.workers.get(&m).is_some());
        assert_eq!(p.reap_dead_workers().await, 0);
    }

    /// The reap exists because the call-site check only fires when someone
    /// asks for THAT model again. A node whose big model died and is then
    /// asked for a different one would otherwise keep the corpse's charge for
    /// ever — and it is the other model that gets refused.
    /// Gotcha #467: the fatal-error arm — the CUDA out-of-memory path, and the
    /// one the reporting node hits — removed the worker and released nothing.
    ///
    /// The health-tick reap cannot cover this: it scans `workers`, and this has
    /// already taken the entry out. So the charge leaked for the daemon's life.
    #[tokio::test]
    async fn a_worker_killed_by_a_fatal_error_gives_its_memory_back() {
        let p = test_pool();
        p.set_ram_budget_mb(8000);
        let m = ModelId("oomed".into());
        assert!(p.admit_to_cpu(&m, 6000));
        p.workers.insert(m.clone(), fake_worker_handle(false).await);

        let err = p.classify_worker_error(&m, "CUDA_ERROR_OUT_OF_MEMORY".into(), true);
        assert!(matches!(err, SwarmError::ServiceUnavailable(_)));

        assert!(p.workers.get(&m).is_none(), "the worker is evicted");
        assert_eq!(
            p.ram_committed_mb(),
            0,
            "and its charge is released — the reap cannot do it afterwards, \
             because the entry is already gone from `workers`"
        );
        assert!(p.admit_to_cpu(&ModelId("next".into()), 6000));
    }

    /// A caller holding a handle that has already been replaced must not evict
    /// the live worker that replaced it. `evict_this_worker` compares identity,
    /// not just the key.
    #[tokio::test]
    async fn evicting_a_stale_handle_leaves_the_live_worker_alone() {
        let p = test_pool();
        p.set_ram_budget_mb(8000);
        let m = ModelId("replaced".into());
        let stale = fake_worker_handle(true).await;
        let live = fake_worker_handle(false).await;
        p.workers.insert(m.clone(), live.clone());
        assert!(p.admit_to_cpu(&m, 3000));

        p.evict_this_worker(&m, &stale, "IPC send failed");

        assert!(
            p.workers.get(&m).is_some(),
            "the live worker under that key is not the one the caller was using"
        );
        assert_eq!(p.ram_committed_mb(), 3000, "and its charge stands");
    }

    /// A worker that CRASHED frees the card exactly as a graceful unload does,
    /// so the CPU pins its occupancy caused must lift either way.
    ///
    /// `unload_model` had always done this; `retire_dead_worker` — added hours
    /// earlier the same day for gotcha #461 — released the memory and stopped
    /// there, so a GPU worker killed by the OS left every other model on the
    /// processor at ~10x the cost, which is the loss gotcha #401 exists to
    /// prevent. One invariant, two paths, and the new path had half of it.
    #[tokio::test]
    async fn a_crashed_gpu_worker_lets_pinned_models_try_the_card_again() {
        let p = test_pool();
        let pinned = ModelId("was-pushed-to-the-cpu".into());
        p.cpu_pinned_models.insert(pinned.clone());

        let on_card = ModelId("held-the-card".into());
        p.workers
            .insert(on_card.clone(), fake_worker_handle_on(true, None).await);
        assert!(p.retire_dead_worker(&on_card).await);

        assert!(
            p.cpu_pinned_models.is_empty(),
            "the card is free again, so the pin its occupancy caused must lift"
        );
    }

    /// The control: a PROCESSOR worker dying frees no graphics memory, so it
    /// must not lift anything. Without this the test above would pass on code
    /// that cleared pins unconditionally, which would send a model back to a
    /// card that is still full.
    #[tokio::test]
    async fn a_crashed_cpu_worker_lifts_no_pins() {
        let p = test_pool();
        let pinned = ModelId("still-pushed-to-the-cpu".into());
        p.cpu_pinned_models.insert(pinned.clone());

        let on_cpu = ModelId("was-on-the-cpu".into());
        p.workers
            .insert(on_cpu.clone(), fake_worker_handle(true).await);
        assert!(p.retire_dead_worker(&on_cpu).await);

        assert!(
            p.cpu_pinned_models.contains(&pinned),
            "no graphics memory was freed, so nothing may be sent back to the card"
        );
    }

    #[tokio::test]
    async fn the_reap_frees_a_dead_worker_nobody_asks_about_again() {
        let p = test_pool();
        p.set_ram_budget_mb(8000);
        let big = ModelId("big-and-dead".into());
        assert!(p.admit_to_cpu(&big, 7000));
        p.workers
            .insert(big.clone(), fake_worker_handle(true).await);

        // A DIFFERENT model is refused while the corpse holds the budget.
        assert!(
            !p.admit_to_cpu(&ModelId("small".into()), 2000),
            "the dead worker's charge blocks an unrelated model"
        );

        assert_eq!(p.reap_dead_workers().await, 1);
        assert_eq!(p.ram_committed_mb(), 0);
        assert!(p.admit_to_cpu(&ModelId("small".into()), 2000));
    }

    #[test]
    fn ram_is_charged_wherever_the_model_can_only_land_in_ram() {
        // Sent to the CPU explicitly or by VRAM refusal: charged.
        assert!(charges_ram(true, true, true));
        // GPU present and usable, model going to it: not charged.
        assert!(!charges_ram(false, true, true));
        // No GPU detected: charged — this is every CPU-only node, which used
        // to skip RAM admission entirely (2026-08-21).
        assert!(charges_ram(false, false, true));
        // A build without CUDA cannot put the model anywhere else.
        assert!(charges_ram(false, true, false));
    }

    #[test]
    fn effective_gpu_layers_defaults_to_auto() {
        let pool = test_pool();
        assert_eq!(pool.effective_gpu_layers(&ModelId("m".into())), -1);
    }

    #[test]
    fn effective_gpu_layers_follows_config() {
        let pool = test_pool();
        pool.set_gpu_layers(0);
        assert_eq!(pool.effective_gpu_layers(&ModelId("m".into())), 0);
        pool.set_gpu_layers(20);
        assert_eq!(pool.effective_gpu_layers(&ModelId("m".into())), 20);
    }

    #[test]
    fn cpu_pin_overrides_config_per_model() {
        // After a GPU OOM the pinned model must respawn on CPU even though the
        // config says otherwise — otherwise the respawned worker makes the
        // identical allocation and dies the same way, forever.
        let pool = test_pool();
        pool.set_gpu_layers(-1);
        let pinned = ModelId("oom-model".into());
        let other = ModelId("healthy-model".into());
        pool.cpu_pinned_models.insert(pinned.clone());

        assert_eq!(pool.effective_gpu_layers(&pinned), 0);
        assert_eq!(pool.effective_gpu_layers(&other), -1);
        assert!(pool.is_cpu_pinned(&pinned));
        assert!(!pool.is_cpu_pinned(&other));

        assert!(pool.clear_cpu_pin(&pinned));
        assert_eq!(pool.effective_gpu_layers(&pinned), -1);
        assert!(!pool.clear_cpu_pin(&pinned), "second clear is a no-op");
    }

    /// A pin lasted the life of the process because `clear_cpu_pin` had no
    /// caller — a ~10x throughput loss triggered by the daemon's own background
    /// churn, with nothing in the API to show it. Freeing GPU memory is the
    /// event that makes a retry worthwhile.
    #[tokio::test]
    async fn unloading_a_gpu_worker_lifts_cpu_pins() {
        let pool = test_pool();
        pool.set_gpu_layers(-1);
        let pinned = ModelId("oom-model".into());
        pool.cpu_pinned_models.insert(pinned.clone());
        assert!(pool.is_cpu_pinned(&pinned));

        // No worker registered for this id, but it is not CPU-pinned itself, so
        // it counts as a GPU tenant whose removal frees memory.
        pool.unload_model(&ModelId("some-other-gpu-model".into()))
            .await;

        assert!(
            pool.is_cpu_pinned(&pinned),
            "no worker existed, so nothing was actually freed and the pin stands"
        );
    }

    /// Unloading a model that was ITSELF pinned to the CPU frees no GPU memory,
    /// so it must not trigger a retry storm for everything else.
    #[tokio::test]
    async fn unloading_a_cpu_pinned_worker_does_not_lift_other_pins() {
        let pool = test_pool();
        pool.set_gpu_layers(-1);
        let a = ModelId("cpu-bound".into());
        let b = ModelId("also-pinned".into());
        pool.cpu_pinned_models.insert(a.clone());
        pool.cpu_pinned_models.insert(b.clone());

        pool.unload_model(&a).await;
        assert!(
            pool.is_cpu_pinned(&b),
            "unloading a CPU worker frees no VRAM, so other pins must stand"
        );
    }

    /// `resources.max_ram_mb` shipped documented as "0 = auto (50% of system
    /// RAM)" while nothing read it, so a node had no memory ceiling at all.
    /// Admission now behaves like its GPU sibling.
    /// The reporter's machine: cap 18000 MB, 14773 MB free at the time, a
    /// 13149 MB model. The cap allows it; the live headroom (70% of free) does
    /// not — and the same pool admits it once memory frees up, because the
    /// provider is asked every time rather than once at startup.
    #[test]
    fn a_live_headroom_refuses_what_the_cap_would_allow_and_relents_when_memory_frees() {
        use crate::model::auto_manage::vram::RamBudget;
        let free = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(14773));
        let p = test_pool();
        let f = free.clone();
        p.set_ram_budget_provider(Box::new(move || {
            Some(RamBudget::from_machine(
                18000,
                18000,
                32000,
                f.load(std::sync::atomic::Ordering::Relaxed),
            ))
        }));
        let m = ModelId("llama-8b".into());
        assert!(
            !p.admit_to_cpu(&m, 13149),
            "10341 MB of headroom cannot take 13149 MB"
        );
        free.store(26000, std::sync::atomic::Ordering::Relaxed);
        assert!(p.admit_to_cpu(&m, 13149), "18200 MB of headroom can");
        p.release_ram_charge(&m);
    }

    #[test]
    fn ram_admission_refuses_once_the_budget_is_committed() {
        let p = test_pool();
        p.set_ram_budget_mb(6000);
        assert!(p.admit_to_cpu(&ModelId("first".into()), 4000));
        assert!(
            !p.admit_to_cpu(&ModelId("second".into()), 2500),
            "4000 + 2500 exceeds 6000 — must refuse rather than swap"
        );
        // Something that does fit in the remainder is still admitted.
        assert!(p.admit_to_cpu(&ModelId("small".into()), 1500));
    }

    /// A worker loads the layer range it is asked for, not the whole model, and
    /// the estimate must say so — this is the field defect (gotcha #452) at the
    /// arithmetic level.
    #[test]
    fn a_segment_is_priced_as_a_segment_not_as_the_whole_model() {
        const SHARDS: u64 = 8_566 * 1024 * 1024;
        // The Mac mini's plan: 36 of a 48-layer 14B, starting at layer 0.
        let (layers, is_first, weights) = super::segment_shape(48, SHARDS, Some((0, 36)));
        assert_eq!(layers, 36);
        assert!(is_first, "layer 0 carries the embedding table");
        assert_eq!(weights, SHARDS / 48 * 36);

        // A privacy boomerang's far end: one layer, no embedding table. This is
        // the case the old whole-model charge got worst — a node was priced for
        // an entire model to hold two of its layers.
        let (layers, is_first, weights) = super::segment_shape(48, SHARDS, Some((47, 48)));
        assert_eq!(layers, 1);
        assert!(!is_first);
        assert_eq!(weights, SHARDS / 48);

        // Asking about the whole model is still the whole model — the question
        // `would_fit_on_gpu` and the dashboard ask.
        assert_eq!(
            super::segment_shape(48, SHARDS, None),
            (48, true, SHARDS),
            "None must keep meaning the whole model"
        );
        // A range covering everything is the whole model, exactly.
        assert_eq!(
            super::segment_shape(48, SHARDS, Some((0, 48))),
            (48, true, SHARDS)
        );
        // Degenerate inputs must not divide by zero or price nothing.
        assert_eq!(
            super::segment_shape(0, SHARDS, Some((0, 0))),
            (1, true, SHARDS)
        );
    }

    /// The curve the planner and the incremental charge both read is taken from
    /// the estimator, not restated.
    #[test]
    fn the_segment_cost_curve_recovers_the_fixed_and_per_layer_terms() {
        // cost(n) = 300 + 210n
        assert_eq!(super::cost_curve_from(510, 720), (300, 210));
        // A flat estimate has no per-layer term, which is unknowable rather
        // than unlimited.
        assert_eq!(super::cost_curve_from(400, 400), (400, 0));
        assert_eq!(super::layers_that_fit(7910, 300, 0), None);
        // 7910 MB of headroom, 300 fixed, 210 a layer → 36.
        assert_eq!(super::layers_that_fit(7910, 300, 210), Some(36));
        // Less headroom than the fixed cost is zero layers, not an underflow.
        assert_eq!(super::layers_that_fit(100, 300, 210), Some(0));
    }

    /// Two segments of one model share a worker, so the second must be charged
    /// on top of the first rather than replacing it.
    #[test]
    fn a_second_segment_adds_to_the_charge_it_does_not_replace_it() {
        let p = test_pool();
        p.set_ram_budget_mb(6000);
        let m = ModelId("boomerang".into());
        assert!(p.admit_to_cpu(&m, 4000), "the first segment fits");
        assert!(p.admit_to_cpu(&m, 1500), "and so does the second");
        assert_eq!(
            p.ram_committed_mb(),
            5500,
            "both segments are charged; an overwrite would report 1500"
        );
        assert!(
            !p.admit_to_cpu(&ModelId("other".into()), 1000),
            "5500 + 1000 exceeds 6000 — the second segment must still be visible"
        );
        p.release_ram_charge(&m);
        assert_eq!(p.ram_committed_mb(), 0, "releasing frees every segment");
        assert!(p.admit_to_cpu(&ModelId("other".into()), 1000));
    }

    /// A range is charged the first time it is asked for and not again — the
    /// spawn path records its own, or every later forward would re-charge it.
    #[test]
    fn a_layer_range_is_charged_once_and_forgotten_on_release() {
        let p = test_pool();
        let m = ModelId("m".into());
        assert!(!p.segment_is_charged(&m, (0, 8)));
        p.record_charged_segment(&m, (0, 8));
        assert!(p.segment_is_charged(&m, (0, 8)));
        assert!(
            !p.segment_is_charged(&m, (8, 16)),
            "a different range is a different charge"
        );
        p.record_charged_segment(&m, (0, 8));
        p.record_charged_segment(&m, (8, 16));
        assert_eq!(
            p.charged_segments.get(&m).map(|v| v.len()),
            Some(2),
            "recording the same range twice must not double it"
        );
        p.release_ram_charge(&m);
        assert!(
            !p.segment_is_charged(&m, (0, 8)),
            "a retired worker's segments go with its charge"
        );
    }

    /// Unknowable is not "no room": a model this node holds no header for must
    /// leave the scheduler's local bound unset, exactly as an unreadable peer
    /// capability does.
    #[test]
    fn an_unreadable_model_leaves_the_local_capacity_bound_unset() {
        let p = test_pool();
        p.set_ram_budget_mb(6000);
        assert_eq!(
            p.max_local_hostable_layers(&ModelId("no-such-model".into()), false),
            None
        );
    }

    /// Releasing must credit the budget back, or churn shrinks it permanently.
    #[test]
    fn releasing_a_ram_charge_frees_the_budget_again() {
        let p = test_pool();
        p.set_ram_budget_mb(6000);
        let a = ModelId("a".into());
        assert!(p.admit_to_cpu(&a, 5000));
        assert!(!p.admit_to_cpu(&ModelId("b".into()), 2000));
        p.release_ram_charge(&a);
        assert!(
            p.admit_to_cpu(&ModelId("b".into()), 2000),
            "the freed charge must be usable again"
        );
    }

    /// No budget configured, or an unreadable header (estimate 0), must not
    /// start refusing loads — that would be a worse failure than the one being
    /// prevented, and mirrors how `admit_to_gpu` treats the same gap.
    #[test]
    fn ram_admission_does_not_judge_what_it_cannot_measure() {
        let p = test_pool();
        // Budget unset.
        assert!(p.admit_to_cpu(&ModelId("nobudget".into()), 99_999));

        let p2 = test_pool();
        p2.set_ram_budget_mb(1);
        assert!(
            p2.admit_to_cpu(&ModelId("mystery".into()), 0),
            "an unreadable model must not be refused on a guess"
        );
    }

    #[test]
    fn cpu_pinned_ids_are_reported_for_the_api() {
        let pool = test_pool();
        pool.cpu_pinned_models.insert(ModelId("zeta".into()));
        pool.cpu_pinned_models.insert(ModelId("alpha".into()));
        assert_eq!(pool.cpu_pinned_model_ids(), vec!["alpha", "zeta"]);
        pool.clear_cpu_pin(&ModelId("alpha".into()));
        assert_eq!(pool.cpu_pinned_model_ids(), vec!["zeta"]);
    }

    /// The full classification, not just the string helper: a GPU with no room
    /// left must come back as `ServiceUnavailable` so the router can hand the
    /// request to a peer. As `Inference` it was HTTP 500, which a coordinator
    /// reads as "this node is broken" rather than "ask someone else" — costing
    /// an answer another peer could have given.
    ///
    /// Not marked fatal: the worker is fine, the GPU is merely full, so it must
    /// keep its worker rather than being evicted and reloaded.
    #[test]
    fn a_full_gpu_is_classified_as_service_unavailable_not_an_inference_fault() {
        let pool = test_pool();
        let model = ModelId("m".into());
        let err = pool.classify_worker_error(
            &model,
            "prefill forward: Service unavailable: Not enough free GPU memory to continue \
             this conversation (168 MB of KV cache already in use, budget 323 MB)."
                .into(),
            false,
        );
        assert!(
            matches!(err, SwarmError::ServiceUnavailable(_)),
            "a full GPU must be 503 so the router re-routes, got {err:?}"
        );
    }

    #[test]
    fn fatal_error_evicts_worker_and_returns_service_unavailable() {
        let pool = test_pool();
        let model = ModelId("m".into());
        let err = pool.classify_worker_error(
            &model,
            "Forward: Cuda(DriverError(CUDA_ERROR_OUT_OF_MEMORY, \"out of memory\"))".into(),
            true,
        );
        assert!(matches!(err, SwarmError::ServiceUnavailable(_)));
        // OOM on a GPU-eligible model pins it to CPU for the next spawn.
        assert!(pool.is_cpu_pinned(&model));
    }

    #[test]
    fn non_fatal_error_stays_inference_and_leaves_model_gpu_eligible() {
        let pool = test_pool();
        let model = ModelId("m".into());
        let err = pool.classify_worker_error(&model, "Tokenize: bad prompt".into(), false);
        assert!(matches!(err, SwarmError::Inference(_)));
        assert!(!pool.is_cpu_pinned(&model));
    }

    #[test]
    fn fatal_non_oom_error_does_not_pin_to_cpu() {
        // An illegal memory access is fatal to the worker but is not evidence
        // that the GPU is too small — retrying on the GPU is right here.
        let pool = test_pool();
        let model = ModelId("m".into());
        let err = pool.classify_worker_error(
            &model,
            "cuda error: an illegal memory access was encountered".into(),
            true,
        );
        assert!(matches!(err, SwarmError::ServiceUnavailable(_)));
        assert!(!pool.is_cpu_pinned(&model));
    }
}

#[cfg(test)]
mod validation_detail_tests {
    use crate::error::{reclassify_flattened_error, SwarmError};

    /// These live here because this is the boundary they were written against,
    /// but the recovery is now shared with the network boundary — see
    /// `crate::error::reclassify_flattened_error`.
    fn model_unavailable_detail(m: &str) -> Option<String> {
        match reclassify_flattened_error(m) {
            Some(SwarmError::ModelNotAvailable(id)) => Some(id.0),
            _ => None,
        }
    }
    fn service_unavailable_detail(m: &str) -> Option<String> {
        match reclassify_flattened_error(m) {
            Some(SwarmError::ServiceUnavailable(d)) => Some(d),
            _ => None,
        }
    }
    fn validation_detail(m: &str) -> Option<String> {
        match reclassify_flattened_error(m) {
            Some(SwarmError::Validation(d)) => Some(d),
            _ => None,
        }
    }

    /// A model whose files are gone must reach the caller as 404, not 500.
    /// Reported with a real trace: the message plainly said "Model not
    /// available" and the client still got `server_error`.
    #[test]
    fn a_missing_model_is_recovered_from_a_flattened_message() {
        let msg = "Inference error: Model not available: Manifest not found: \
                   /home/u/.local/share/swarmllm/models/qwen2.5-0.5b/manifest.json";
        let got = model_unavailable_detail(msg).expect("should be recognised");
        assert!(got.starts_with("Manifest not found:"));
    }

    /// Doubly-wrapped messages happen (worker → pipeline → router). Take the
    /// innermost marker so the id survives rather than a wrapper fragment.
    #[test]
    fn the_innermost_marker_wins() {
        let msg = "Worker: Model not available: outer: Model not available: real-model-id";
        assert_eq!(
            model_unavailable_detail(msg).as_deref(),
            Some("real-model-id")
        );
    }

    /// A GPU that cannot fit another conversation must reach the caller as 503,
    /// not 500 — a coordinator re-routes on 503 and gives up on 500, so the
    /// wrong status costs an answer another peer could have given.
    ///
    /// Observed with two processes sharing one GPU: `Inference error: prefill
    /// forward: Service unavailable: Not enough free GPU memory to continue
    /// this conversation (168 MB of KV cache already in use, budget 323 MB)` —
    /// served as HTTP 500 with the words "Service unavailable" inside it.
    #[test]
    fn a_gpu_out_of_room_is_recovered_as_service_unavailable() {
        let msg = "Inference error: prefill forward: Service unavailable: Not enough free \
                   GPU memory to continue this conversation (168 MB of KV cache already in \
                   use, budget 323 MB).";
        let got = service_unavailable_detail(msg).expect("should be recognised");
        assert!(
            got.starts_with("Not enough free GPU memory"),
            "must keep the actionable part, got {got:?}"
        );
    }

    /// The innermost marker wins, as with its two siblings — a doubly-wrapped
    /// message must still yield the original reason rather than a wrapper.
    #[test]
    fn the_innermost_service_unavailable_marker_wins() {
        let msg = "outer: Service unavailable: mid: Service unavailable: the real reason";
        assert_eq!(
            service_unavailable_detail(msg).as_deref(),
            Some("the real reason")
        );
    }

    /// An ordinary failure must NOT be reclassified as unavailable — that would
    /// turn a genuine bug into a "try again later" and hide it.
    #[test]
    fn an_ordinary_failure_is_not_reclassified_as_unavailable() {
        assert!(service_unavailable_detail("Inference error: tensor shape mismatch").is_none());
        assert!(service_unavailable_detail("Service unavailable: ").is_none());
    }

    /// An ordinary inference failure must NOT be reclassified as a missing
    /// model — that would turn a real server fault into a 404 and hide it.
    #[test]
    fn an_unrelated_failure_is_left_alone() {
        assert!(model_unavailable_detail("CUDA out of memory").is_none());
        assert!(model_unavailable_detail("Model not available: ").is_none());
    }

    /// The case this exists for: an over-long prompt must reach the user as
    /// their problem, with the sentence that says how to fix it intact.
    #[test]
    fn a_context_overflow_is_recognised_and_unwrapped() {
        let msg = "prefill forward: Validation error: Sequence length (2176) exceeds \
                   model context window (2048). Reduce your prompt or max_tokens.";
        let got = validation_detail(msg).expect("should be recognised as validation");
        assert!(got.starts_with("Sequence length (2176) exceeds"));
        assert!(got.ends_with("Reduce your prompt or max_tokens."));
        assert!(
            !got.contains("prefill forward"),
            "call-site plumbing must be dropped"
        );
    }

    /// Genuine faults must keep mapping to a server error — this must not
    /// become a way to relabel our own bugs as the caller's fault.
    #[test]
    fn other_worker_errors_are_not_treated_as_validation() {
        assert!(validation_detail("Internal error: blk.0.attn_q: missing region").is_none());
        assert!(validation_detail("CUDA_ERROR_OUT_OF_MEMORY").is_none());
        assert!(validation_detail("worker closed connection mid-generate").is_none());
    }

    /// A marker with nothing after it carries no information for the user, so
    /// it stays an inference error rather than becoming an empty 400.
    #[test]
    fn an_empty_detail_is_not_a_validation_error() {
        assert!(validation_detail("prefill forward: Validation error: ").is_none());
    }

    /// Nested prefixes: take the LAST marker so the innermost detail wins.
    #[test]
    fn the_innermost_detail_wins() {
        let got = validation_detail("a: Validation error: b: Validation error: real detail");
        assert_eq!(got.as_deref(), Some("real detail"));
    }
}

#[cfg(test)]
mod admission_tests {
    use super::*;

    fn pool() -> ModelProcessPool {
        ModelProcessPool::new(std::path::PathBuf::from("/tmp/swarmllm-admission-test"))
    }

    /// The combined accessor and the single-answer methods must agree.
    ///
    /// `gpu_estimate_and_fit` exists ONLY to avoid reading a model's header
    /// twice for two answers that come from one reading. If it also had its own
    /// opinion it would be worse than the duplication it removes — the admin
    /// listing would report a different verdict from the one admission
    /// actually applies, which is the contradiction gotcha #329 was about.
    #[test]
    fn the_combined_accessor_agrees_with_the_single_answer_methods() {
        let p = pool();
        let m = ModelId("no-such-model".into());
        // No geometry to read: both must say "unknowable", not "no".
        assert_eq!(
            p.gpu_estimate_and_fit(&m),
            (p.estimated_gpu_mb(&m), p.would_fit_on_gpu(&m))
        );
        assert_eq!(p.gpu_estimate_and_fit(&m), (None, None));
        // A budget alone does not make an unreadable model judgeable.
        p.set_vram_budget_mb(6000);
        assert_eq!(
            p.gpu_estimate_and_fit(&m),
            (p.estimated_gpu_mb(&m), p.would_fit_on_gpu(&m))
        );
    }

    /// The arithmetic both of them share, over the cases neither can reach
    /// without a model on disk.
    #[test]
    fn an_unknown_size_or_budget_is_never_reported_as_not_fitting() {
        let p = pool();
        assert_eq!(p.fits_in_budget(0, 6000), None, "no geometry is unknowable");
        assert_eq!(p.fits_in_budget(4000, 0), None, "no budget is unknowable");
        assert_eq!(p.fits_in_budget(4000, 6000), Some(true));
        assert_eq!(p.fits_in_budget(6001, 6000), Some(false));
        assert_eq!(
            p.fits_in_budget(6000, 6000),
            Some(true),
            "exactly full fits"
        );
        // Charged models are counted against the budget, not ignored.
        assert!(p.admit_to_gpu(&ModelId("resident".into()), 4000));
        assert_eq!(p.fits_in_budget(2500, 6000), Some(false));
        assert_eq!(p.fits_in_budget(2000, 6000), Some(true));
    }

    /// With no budget configured, behaviour must be exactly as before — this
    /// gate must not start refusing GPU loads on nodes that never asked for a
    /// limit.
    #[test]
    fn no_budget_admits_everything() {
        let p = pool();
        assert!(p.admit_to_gpu(&ModelId("a".into()), 99_999));
        assert!(p.admit_to_gpu(&ModelId("b".into()), 99_999));
    }

    /// A model that fits is admitted and charged; the next one is judged against
    /// what is already committed, not against an empty device.
    #[test]
    fn a_second_model_is_judged_against_the_first() {
        let p = pool();
        p.set_vram_budget_mb(6000);
        assert!(p.admit_to_gpu(&ModelId("first".into()), 4000));
        assert_eq!(p.vram_committed_mb(), 4000);
        // 4000 + 2500 > 6000 → refused, and NOT charged.
        assert!(!p.admit_to_gpu(&ModelId("second".into()), 2500));
        assert_eq!(p.vram_committed_mb(), 4000, "a refusal must not charge");
        // Something that does fit still gets in.
        assert!(p.admit_to_gpu(&ModelId("small".into()), 1500));
        assert_eq!(p.vram_committed_mb(), 5500);
    }

    /// The reported failure in miniature: two models that each fit alone but not
    /// together. Before admission control both proceeded and the second worker
    /// died with a real CUDA OOM.
    #[test]
    fn two_models_that_do_not_fit_together_are_not_both_admitted() {
        let p = pool();
        p.set_vram_budget_mb(6000);
        let a = ModelId("phi-3.5".into());
        let b = ModelId("llama-3b".into());
        assert!(p.admit_to_gpu(&a, 5900), "fits on its own");
        assert!(
            !p.admit_to_gpu(&b, 3000),
            "must be refused rather than left to OOM"
        );
    }

    fn cand(model: &str, charge_mb: u64, idle_secs: u64) -> VramReclaimCandidate {
        VramReclaimCandidate {
            model: ModelId(model.into()),
            charge_mb,
            idle_secs,
            busy: false,
        }
    }

    /// The reported case, in miniature. An 8B loaded fifteen minutes ago holds
    /// 6033 MB of a 6722 MB budget; a 3B needing 3138 MB arrives. Before this,
    /// admission simply refused and the 3B answered at 7 tok/s on the CPU while
    /// the card sat idle.
    #[test]
    fn an_idle_model_gives_up_the_card_to_the_one_being_asked_for() {
        let plan = plan_vram_reclaim(6722, 6033, 3138, vec![cand("llama-8b", 6033, 900)]);
        assert_eq!(
            plan,
            vec![(ModelId("llama-8b".into()), 6033)],
            "the idle occupant must be reclaimed rather than the newcomer demoted"
        );
    }

    /// Reclaiming must stop as soon as the model fits: taking more than needed
    /// costs cold starts nobody asked for.
    #[test]
    fn only_as_many_models_are_freed_as_the_new_one_needs() {
        let plan = plan_vram_reclaim(
            8000,
            7500,
            1000,
            vec![cand("stale", 600, 3600), cand("older", 900, 1800)],
        );
        assert_eq!(plan.len(), 1, "one is enough: 7500-600+1000 <= 8000");
        assert_eq!(plan[0].0, ModelId("stale".into()), "most idle goes first");
    }

    /// The property that makes this safe to run on every refused admission:
    /// when reclaiming everything eligible would STILL not fit, nothing is
    /// unloaded. The model is going to the CPU either way, so a cold start
    /// bought nothing.
    #[test]
    fn nothing_is_unloaded_when_it_would_still_not_fit() {
        let plan = plan_vram_reclaim(6000, 5000, 5500, vec![cand("small", 1000, 3600)]);
        assert!(
            plan.is_empty(),
            "must not pay a reload for a model that cannot fit anyway"
        );
    }

    /// A worker with a request in flight is never taken, however idle it looks
    /// by the clock — that is killing an answer mid-generation.
    #[test]
    fn a_busy_worker_is_never_reclaimed() {
        let mut busy = cand("serving", 6033, 9999);
        busy.busy = true;
        assert!(plan_vram_reclaim(6722, 6033, 3138, vec![busy]).is_empty());
    }

    /// Two models alternating faster than they load must NOT evict each other
    /// on every request: below the idle floor the previous behaviour (the
    /// newcomer runs on the CPU) is kept, so this can only improve placement.
    #[test]
    fn a_recently_used_model_keeps_the_card() {
        let plan = plan_vram_reclaim(
            6722,
            6033,
            3138,
            vec![cand(
                "just-answered",
                6033,
                VRAM_MAKE_ROOM_MIN_IDLE_SECS_DEFAULT - 1,
            )],
        );
        assert!(plan.is_empty(), "displacing it would thrash both models");
    }

    /// Nothing to do when it already fits — this must not unload models to make
    /// room that is already there.
    #[test]
    fn a_model_that_already_fits_reclaims_nothing() {
        assert!(plan_vram_reclaim(8000, 1000, 2000, vec![cand("idle", 1000, 3600)]).is_empty());
    }

    /// The reported case, at the moment the user asks for the model again:
    /// gemma-2-2b-it is resident on the processor, llama has since been idle-
    /// unloaded so the pin is gone, and the card is nearly empty (653 of 6141
    /// MB). Before this, `get_or_spawn`'s fast path handed back the processor
    /// worker and the model stayed there for as long as it lived.
    fn promotable() -> PromotionInputs {
        PromotionInputs {
            cpu_placed: true,
            permanently_cpu_bound: false,
            busy: false,
            idle_secs: 300,
            gpu_estimate_mb: 2200,
            budget_mb: 6141,
            committed_mb: 653,
            reclaimable_mb: 0,
        }
    }

    #[test]
    fn a_demoted_model_returns_to_the_card_once_it_fits_again() {
        assert!(
            should_return_to_gpu(&promotable()),
            "the pin was lifted and the memory is there — retire the processor worker"
        );
    }

    /// The gate that keeps this off every other node: a worker already on the
    /// card, and every worker on a machine that has none, is not a candidate.
    /// Nothing was demoted, so there is nowhere to promote to.
    #[test]
    fn a_worker_this_node_did_not_demote_is_left_alone() {
        let mut i = promotable();
        i.cpu_placed = false;
        assert!(!should_return_to_gpu(&i));
    }

    /// The two reasons that never clear — `gpu_layers = 0` and a card below
    /// the kernel floor — still block: respawning would land on the processor
    /// again and cost a reload for nothing. (A VRAM pin does NOT block; that is
    /// the condition this decision exists to reconsider.)
    #[test]
    fn a_model_permanently_bound_to_the_processor_is_not_retired() {
        let mut i = promotable();
        i.permanently_cpu_bound = true;
        assert!(!should_return_to_gpu(&i));
    }

    /// Retiring a worker kills the subprocess. A request in flight dies with
    /// it, so a busy worker is never touched however long ago it was started.
    #[test]
    fn a_busy_worker_is_never_retired() {
        let mut i = promotable();
        i.busy = true;
        assert!(!should_return_to_gpu(&i));
    }

    /// The idle floor is what makes the busy check a guarantee rather than a
    /// hope: with no request in the last minute there is no window between the
    /// decision and the kill for one to arrive in. A model under continuous
    /// load waits for a gap — the pin stays lifted, so it gets one.
    #[test]
    fn a_model_answering_right_now_waits_for_a_gap() {
        let mut i = promotable();
        i.idle_secs = VRAM_MAKE_ROOM_MIN_IDLE_SECS_DEFAULT - 1;
        assert!(!should_return_to_gpu(&i));
    }

    /// Cost the move before making it. Retiring a working worker and then
    /// having admission refuse it costs a cold start and leaves the model
    /// exactly where it was.
    #[test]
    fn a_model_that_still_would_not_fit_stays_on_the_processor() {
        let mut i = promotable();
        i.committed_mb = 4500; // 4500 + 2200 > 6141
        assert!(!should_return_to_gpu(&i));
    }

    /// **An admission refusal must not become a standing verdict.** It used to
    /// pin the model to the processor, and the pin is cleared only when a
    /// GPU-holding worker unloads — so a model demoted while the card was busy
    /// stayed there for as long as the occupant was kept resident, with
    /// admission never asked again. Measured 2026-08-28: 50 minutes on the
    /// processor, the occupant idle throughout and reclaimable the whole time.
    ///
    /// A worker spawn is where this shows: with no pin, `cpu_reason` is `None`
    /// and the ordinary admission path — including its reclaim — runs afresh.
    #[test]
    fn a_refused_admission_leaves_the_model_free_to_try_again() {
        let pool = pool();
        pool.set_gpu_layers(-1);
        pool.set_vram_budget_mb(6185);
        let occupant = ModelId("has-the-card".into());
        let newcomer = ModelId("wants-the-card".into());

        assert!(pool.admit_to_gpu(&occupant, 6033), "the card starts empty");
        assert!(
            !pool.admit_to_gpu(&newcomer, 3138),
            "and is then too full for the newcomer"
        );

        assert!(
            !pool.is_cpu_pinned(&newcomer),
            "a refusal is a fact about this moment, not a verdict about the model"
        );
        assert_eq!(
            pool.cpu_reason(&newcomer),
            None,
            "so the next spawn reaches admission — and its reclaim — instead of \
             short-circuiting to the processor"
        );
    }

    /// **The card is full, but of a model the pool would give up.** A VRAM pin
    /// is only lifted when a GPU-holding worker unloads, and the idle-unload
    /// timer is minutes away — so asking "does it fit in the room lying about"
    /// left a model on the processor while its card-mate had already been idle
    /// long enough to be reclaimed on demand. The dry run answers with room the
    /// pool has agreed it may make, under the same guards the real reclaim uses.
    #[test]
    fn room_the_pool_would_make_counts_as_room() {
        let mut i = promotable();
        i.committed_mb = 6033; // an 8B has the card
        i.gpu_estimate_mb = 3138;
        assert!(
            !should_return_to_gpu(&i),
            "with nothing reclaimable it must stay where it is"
        );
        i.reclaimable_mb = 6033; // ... and that 8B is idle past the floor
        assert!(
            should_return_to_gpu(&i),
            "the pool would free the card, so the move is worth making"
        );
    }

    /// A partial reclaim that still would not fit is not a reason to move.
    #[test]
    fn a_reclaim_that_would_not_be_enough_changes_nothing() {
        let mut i = promotable();
        i.committed_mb = 6033;
        i.gpu_estimate_mb = 3138;
        i.reclaimable_mb = 1000; // 6033 - 1000 + 3138 > 6141
        assert!(!should_return_to_gpu(&i));
    }

    /// Unreadable geometry and an unset budget are not evidence. `admit_to_gpu`
    /// treats the same gap as "do not judge" and lets a spawn through, but the
    /// question here is whether to DESTROY something that works, and the answer
    /// to that on no information is no.
    #[test]
    fn an_unknown_footprint_or_budget_leaves_the_model_where_it_is() {
        let mut no_estimate = promotable();
        no_estimate.gpu_estimate_mb = 0;
        assert!(!should_return_to_gpu(&no_estimate));

        let mut no_budget = promotable();
        no_budget.budget_mb = 0;
        assert!(!should_return_to_gpu(&no_budget));
    }

    /// Releasing a charge frees the budget for the next admission — otherwise
    /// unloading a model would not actually make room.
    #[test]
    fn releasing_a_charge_frees_the_budget() {
        let p = pool();
        p.set_vram_budget_mb(6000);
        let a = ModelId("a".into());
        assert!(p.admit_to_gpu(&a, 5000));
        assert!(!p.admit_to_gpu(&ModelId("b".into()), 2000));
        p.release_vram_charge(&a);
        assert_eq!(p.vram_committed_mb(), 0);
        assert!(p.admit_to_gpu(&ModelId("b".into()), 2000));
    }

    /// The budget must not come back before the memory does.
    ///
    /// `Child::start_kill` only signals; the OS frees the process's device
    /// allocations afterwards. Releasing the charge on the signal made the
    /// budget read as free while the card was still full, so the next admission
    /// passed against memory that did not exist and its worker died with
    /// CUDA_ERROR_OUT_OF_MEMORY.
    #[tokio::test]
    async fn waiting_for_exit_returns_as_soon_as_the_process_is_reaped() {
        let exited = Arc::new(AtomicBool::new(false));
        let flag = exited.clone();
        tokio::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_millis(40)).await;
            flag.store(true, Ordering::Release);
        });

        let started = std::time::Instant::now();
        let confirmed =
            ModelProcessPool::await_worker_exit(&exited, std::time::Duration::from_secs(5)).await;

        assert!(confirmed, "must report the process confirmed gone");
        assert!(
            started.elapsed() < std::time::Duration::from_secs(1),
            "must return when the process exits, not sit out the whole limit"
        );
    }

    /// A process that never dies must not strand the device forever. The wait
    /// expires and says so, and the caller frees the budget anyway — a worse
    /// outcome than the race, but better than refusing every later load.
    #[tokio::test]
    async fn waiting_for_exit_gives_up_rather_than_stranding_the_device() {
        let never = Arc::new(AtomicBool::new(false));
        let confirmed =
            ModelProcessPool::await_worker_exit(&never, std::time::Duration::from_millis(60)).await;
        assert!(!confirmed, "an expired wait must be reported, not hidden");
    }

    /// An already-reaped worker costs nothing to wait for.
    #[tokio::test]
    async fn waiting_for_an_already_exited_worker_is_immediate() {
        let done = Arc::new(AtomicBool::new(true));
        let started = std::time::Instant::now();
        assert!(
            ModelProcessPool::await_worker_exit(&done, std::time::Duration::from_secs(5)).await
        );
        assert!(started.elapsed() < std::time::Duration::from_millis(100));
    }

    /// An unreadable header yields a 0 estimate, which must NOT be read as "it
    /// fits in nothing" nor as a refusal — failing to read a file is a worse
    /// reason to lose the GPU than the OOM this prevents.
    #[test]
    fn an_unknown_footprint_does_not_refuse() {
        let p = pool();
        p.set_vram_budget_mb(1);
        assert!(p.admit_to_gpu(&ModelId("mystery".into()), 0));
    }

    /// A missing model directory must yield 0 rather than panicking.
    #[test]
    fn estimating_a_missing_model_is_zero() {
        let p = pool();
        assert_eq!(
            p.estimate_gpu_footprint_mb(&ModelId("nope".into()), None),
            0
        );
    }
    /// Where a request would RUN is the delegation precondition, not whether
    /// a card we have is too small (gotcha #442). No card, or a node told to
    /// use its processor, both answer "on the processor".
    #[test]
    fn a_node_with_no_card_serves_on_its_processor() {
        let pool =
            ModelProcessPool::new(std::path::PathBuf::from("/tmp/swarmllm-serves-on-cpu-test"));
        let model = ModelId("m".into());
        pool.set_gpu_detected(false);
        assert!(pool.serves_on_cpu(&model), "no card detected");
        pool.set_gpu_detected(true);
        pool.gpu_layers
            .store(0, std::sync::atomic::Ordering::Relaxed);
        assert!(pool.serves_on_cpu(&model), "told to use the processor");
        pool.gpu_layers
            .store(-1, std::sync::atomic::Ordering::Relaxed);
        // A build without CUDA runs everything on the processor; with CUDA, a
        // card that is present and not known to be too small is not degraded.
        assert_eq!(pool.serves_on_cpu(&model), !cfg!(feature = "candle-cuda"));
    }
}
