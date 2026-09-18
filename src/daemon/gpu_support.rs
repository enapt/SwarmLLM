//! Can this build's CUDA kernels actually run on the GPU that is present?
//!
//! **Why this exists.** `Device::cuda_if_available(0)` answers "is there a
//! CUDA device", which is not the same question. Creating a context succeeds on
//! any NVIDIA card with a driver; the architecture mismatch only surfaces later,
//! when the driver tries to load a module whose PTX targets an arch newer than
//! the device. So a card below this build's floor would start the daemon
//! cleanly, log `GPU detected`, advertise itself to the swarm as a GPU node, and
//! then fail *every* inference request with
//! `DriverError(CUDA_ERROR_NO_BINARY_FOR_GPU)` — a message that tells a
//! non-technical owner nothing about what to do.
//!
//! That became reachable when flash-attention-2 was re-enabled in the `cuda`
//! feature: every `candle-flash-attn` kernel source is `_sm80` and uses Ampere's
//! async-copy instructions, so the CUDA builds compile at compute capability 8.0
//! and pre-Ampere cards (GTX 16-series, RTX 20-series) are outside it.
//!
//! The floor is a property of the BUILD, not of the machine, so it is a
//! compile-time constant here and must be kept equal to `CUDA_COMPUTE_CAP` in
//! `.github/workflows/release.yml`. `compute_cap_matches_release_workflow` in
//! `tests/repo_consistency.rs` fails the build if they drift.

/// Minimum CUDA compute capability this build's kernels are compiled for.
///
/// Must equal `CUDA_COMPUTE_CAP` in `.github/workflows/release.yml`, expressed
/// as (major, minor) — the workflow writes it as the two digits concatenated,
/// so `80` here is `(8, 0)`.
pub const MIN_COMPUTE_CAP: (u32, u32) = (8, 0);

/// Is a card with this compute capability able to run our kernels?
///
/// Forward compatibility is real: PTX compiled for `compute_80` is JIT-compiled
/// by the driver onto any newer architecture, so Ada, Hopper and Blackwell all
/// pass. Only *older* cards fail, and they fail hard rather than degrading.
pub fn compute_cap_supported(cap: (u32, u32)) -> bool {
    cap >= MIN_COMPUTE_CAP
}

/// Parse the `major.minor` string `nvidia-smi --query-gpu=compute_cap` prints.
///
/// Returns `None` for anything unexpected, which every caller must treat as
/// "unknown", never as "unsupported" — refusing the GPU because a subprocess
/// printed something surprising would break working cards, which is a worse
/// failure than the one this module prevents.
pub fn parse_compute_cap(s: &str) -> Option<(u32, u32)> {
    let (major, minor) = s.trim().split_once('.')?;
    Some((major.trim().parse().ok()?, minor.trim().parse().ok()?))
}

/// The message shown to someone whose card this build has left behind.
///
/// Names the card, the requirement, and what actually happens next — the node
/// keeps working on the CPU. Written for someone who does not know what a
/// compute capability is, which is why it translates the number into a card
/// generation they can check against the box.
///
/// It deliberately does NOT point at "a build for older cards": no pre-Ampere
/// CUDA asset is published (see docs/FUTURE_WORK.md). Telling someone to go and
/// find a download that does not exist is worse than telling them nothing, and
/// this text is the only thing most people will ever read about it.
pub fn unsupported_gpu_message(gpu_name: &str, cap: (u32, u32)) -> String {
    format!(
        "{} is too old for GPU acceleration in this version (your card is NVIDIA compute \
         capability {}.{}; {}.{} or newer is needed, which means an RTX 30-series or newer). \
         Running on the processor instead — everything still works, just slower. \
         Nothing to change; this message is for information only.",
        gpu_name, cap.0, cap.1, MIN_COMPUTE_CAP.0, MIN_COMPUTE_CAP.1
    )
}

/// Has the graphics stack stopped working *since this node started*?
///
/// **This is the half of the GPU question that is not a property of the card.**
/// Everything above asks "is this card good enough", which genuinely cannot
/// change while the process runs. This asks "does the graphics stack still
/// work at all", which changes whenever the owner updates a driver — and on
/// the machines this project targets, that happens often and without warning.
///
/// The NVIDIA driver is two halves that must agree: a kernel module loaded into
/// the running kernel, and userspace libraries on disk that talk to it. An
/// update replaces the libraries immediately but cannot replace a module that
/// is loaded and in use, so the old module keeps serving processes that already
/// mapped the old libraries while every NEW process gets the new ones and
/// fails. (Under WSL the same shape arrives via the Windows host driver, which
/// is what `/usr/lib/wsl/lib` is projected from.)
///
/// That asymmetry is exactly why this must be a latch rather than a probe of
/// our own process: **the daemon keeps working while every worker it spawns
/// dies**, because the daemon mapped the libraries at startup and the worker is
/// a fresh `exec` of the same binary. Nothing the daemon can ask about itself
/// reveals the fault; only a worker failing to start does.
///
/// Latched rather than re-probed because it is not recoverable in-process: the
/// libraries are resolved at `exec`, so this node cannot get its GPU back
/// without a restart, and a flapping answer would move models between devices
/// for no reason. [`clear_gpu_runtime_failure`] exists for the one case that
/// disproves it — a worker that later starts successfully.
static GPU_RUNTIME_FAILED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Has a worker failed to start in a way that says the graphics stack is gone?
pub fn gpu_runtime_has_failed() -> bool {
    GPU_RUNTIME_FAILED.load(std::sync::atomic::Ordering::Relaxed)
}

/// Record that the graphics stack is no longer usable.
///
/// Returns `true` only the FIRST time, so the caller can tell the owner once
/// instead of once per refused request — this fires on a path that repeats for
/// every model and every arriving request.
pub fn note_gpu_runtime_failure() -> bool {
    !GPU_RUNTIME_FAILED.swap(true, std::sync::atomic::Ordering::Relaxed)
}

/// Undo the latch, for the one observation that disproves it: a worker that
/// started. Keeps a single unlucky spawn failure from stranding the card for
/// the lifetime of the process.
pub fn clear_gpu_runtime_failure() -> bool {
    GPU_RUNTIME_FAILED.swap(false, std::sync::atomic::Ordering::Relaxed)
}

/// What to tell the owner when the graphics stack has gone out from under us.
///
/// Written for someone who does not know what a driver or a library is: it
/// names the cause in their terms, and gives the one action that fixes it.
///
/// **It deliberately does not promise that the processor takes over.** The
/// obvious reassurance — "running on the processor meanwhile" — is FALSE here,
/// and measurably so: this binary is linked against `libcuda.so.1`, so when
/// that library goes bad *every* new process fails in the loader, and a
/// CPU-only worker fails exactly as a GPU one does (both exit 127, verified
/// 2026-09-18). Telling someone their node is still serving when it is not is
/// the same class of mistake as pointing them at a download that does not
/// exist — see [`unsupported_gpu_message`], which is guarded against precisely
/// that.
///
/// It says *restart SwarmLLM* rather than *reboot* because restarting the
/// process is sufficient: the fault is that this process holds libraries that
/// no longer match the driver, and a new process picks up matching ones.
pub fn gpu_became_unavailable_message() -> String {
    "SwarmLLM cannot use your graphics card any more, so it cannot load models on this computer. \
     This almost always means the graphics driver was updated — or a game updated it — while \
     SwarmLLM was running. Restarting SwarmLLM fixes it."
        .to_string()
}

/// Is the local GPU usable by this build's kernels?
///
/// Returns `true` when there is no GPU at all, or when the capability could not
/// be read — this answers "should we STOP using the GPU", and the only `false`
/// is a card we positively know is too old, or a graphics stack we have watched
/// fail.
///
/// **Two questions, two lifetimes.** The compute-capability half is cached
/// forever because it is a property of the card and of this binary. The
/// [`gpu_runtime_has_failed`] half is read every call, because a driver update
/// changes it underneath a running node — the original version of this function
/// cached the whole answer on the reasoning that it "cannot change while the
/// process runs", which was true of the only half it then asked about.
#[cfg(feature = "candle-cuda")]
pub fn local_gpu_is_supported() -> bool {
    use std::sync::OnceLock;
    if gpu_runtime_has_failed() {
        return false;
    }
    static SUPPORTED: OnceLock<bool> = OnceLock::new();
    *SUPPORTED.get_or_init(
        || match crate::model::auto_manage::vram::detect_gpu_compute_cap() {
            Some(cap) if !compute_cap_supported(cap) => {
                let (name, _) = crate::model::auto_manage::vram::detect_gpu_nvidia_smi();
                let name = name.unwrap_or_else(|| "This NVIDIA GPU".to_string());
                tracing::warn!(
                    gpu = %name,
                    compute_cap = format!("{}.{}", cap.0, cap.1),
                    required = format!("{}.{}", MIN_COMPUTE_CAP.0, MIN_COMPUTE_CAP.1),
                    "{}",
                    unsupported_gpu_message(&name, cap)
                );
                false
            }
            _ => true,
        },
    )
}

/// How long to give the `--version` probe below. Generous: it is only ever run
/// on a path that has already failed, and the answer decides what the owner is
/// told, so a timeout that fires early would mislabel a slow machine.
const EXEC_PROBE_TIMEOUT_SECS: u64 = 10;

/// Can this binary still start a new process at all?
///
/// **Why ask the binary instead of inspecting the error.** When the graphics
/// libraries are pulled out from under a running node, the worker dies in the
/// dynamic loader, before `main`, and what it prints is neither ours nor
/// predictable: the failure observed on 2026-09-18 was
/// `Inconsistency detected by ld.so: dl-setup_hash.c: 36: _dl_setup_hash:
/// Assertion ... failed!` — which names no library, carries no error code, and
/// matches none of the words anyone would search for. Classifying that text
/// would be guesswork. Re-running the binary is not: the executable either
/// starts or it does not, and on a build linked against `libcuda.so.1` an
/// executable that has stopped starting IS the graphics stack failing.
///
/// `--version` is the cheapest thing this binary does. On a healthy node it
/// answers in milliseconds and never touches the GPU.
///
/// Returns `Err` with whatever the failed attempt printed, so the cause finally
/// reaches the log attributed to the thing it broke, instead of as an
/// unattributed line that happens to sit above a timeout.
pub async fn executable_still_starts(exe: &std::path::Path) -> Result<(), String> {
    let run = tokio::time::timeout(
        std::time::Duration::from_secs(EXEC_PROBE_TIMEOUT_SECS),
        tokio::process::Command::new(exe)
            .arg("--version")
            .stdin(std::process::Stdio::null())
            .output(),
    )
    .await;
    match run {
        Ok(Ok(out)) if out.status.success() && !out.stdout.is_empty() => Ok(()),
        // Exit 127 with an empty stdout is the signature: the loader refused
        // before `main` ran. Report whichever stream carries the reason.
        Ok(Ok(out)) => {
            let text = String::from_utf8_lossy(&out.stderr);
            let text = text.trim();
            Err(if text.is_empty() {
                format!("`--version` exited {} and printed nothing", out.status)
            } else {
                text.to_string()
            })
        }
        Ok(Err(e)) => Err(format!("could not run this program again: {e}")),
        Err(_) => Err(format!(
            "`--version` did not answer within {EXEC_PROBE_TIMEOUT_SECS}s"
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ampere_and_newer_are_supported() {
        assert!(compute_cap_supported((8, 0)), "A100 / the floor itself");
        assert!(compute_cap_supported((8, 6)), "RTX 3070 — the dev machine");
        assert!(compute_cap_supported((8, 9)), "RTX 4090 (Ada)");
        assert!(compute_cap_supported((9, 0)), "H100 (Hopper)");
        assert!(compute_cap_supported((12, 0)), "RTX 50-series (Blackwell)");
    }

    #[test]
    fn pre_ampere_is_not_supported() {
        assert!(
            !compute_cap_supported((7, 5)),
            "RTX 2080 / GTX 1660 (Turing)"
        );
        assert!(!compute_cap_supported((7, 0)), "V100 (Volta)");
        assert!(!compute_cap_supported((6, 1)), "GTX 1080 (Pascal)");
        assert!(!compute_cap_supported((5, 2)), "GTX 970 (Maxwell)");
    }

    #[test]
    fn minor_version_is_compared_within_a_major() {
        // The comparison is on the tuple, so this must not degrade to a
        // major-only check: 7.5 is below 8.0 despite being the highest 7.x.
        assert!(!compute_cap_supported((7, 9)));
        assert!(compute_cap_supported((8, 1)));
    }

    #[test]
    fn parses_what_nvidia_smi_prints() {
        assert_eq!(parse_compute_cap("8.6"), Some((8, 6)));
        assert_eq!(parse_compute_cap("7.5"), Some((7, 5)));
        // --format=csv,noheader leaves the trailing newline on.
        assert_eq!(parse_compute_cap("8.9\n"), Some((8, 9)));
        assert_eq!(parse_compute_cap(" 12.0 "), Some((12, 0)));
    }

    /// The latch is process-global, so the tests that move it must not run
    /// beside each other — two of them interleaving would each see the other's
    /// write and both could pass while the logic was wrong.
    static LATCH_TESTS: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn the_runtime_latch_reports_only_the_first_failure() {
        let _guard = LATCH_TESTS.lock().unwrap_or_else(|e| e.into_inner());
        clear_gpu_runtime_failure();

        assert!(!gpu_runtime_has_failed(), "starts clean");
        assert!(
            note_gpu_runtime_failure(),
            "the first failure is the one worth telling the owner about"
        );
        assert!(gpu_runtime_has_failed());
        // This path repeats for every model and every arriving request, so a
        // second `true` here is a toast storm on a node that is already unwell.
        assert!(
            !note_gpu_runtime_failure(),
            "a repeat failure must not notify again"
        );
        assert!(
            !note_gpu_runtime_failure(),
            "and must keep not notifying, however many arrive"
        );

        clear_gpu_runtime_failure();
    }

    #[test]
    fn a_worker_that_starts_clears_the_latch() {
        let _guard = LATCH_TESTS.lock().unwrap_or_else(|e| e.into_inner());
        clear_gpu_runtime_failure();

        note_gpu_runtime_failure();
        assert!(gpu_runtime_has_failed());
        // Marking the device unhealthy is the easy half. The half that gets
        // forgotten is coming back — NVIDIA's own k8s device plugin is
        // repeatedly bug-reported for exactly this (#1014, gpu-operator #1065).
        assert!(
            clear_gpu_runtime_failure(),
            "clearing a set latch reports that it was set"
        );
        assert!(!gpu_runtime_has_failed(), "the card is usable again");
        assert!(
            !clear_gpu_runtime_failure(),
            "clearing an unset latch is a no-op, so an ordinary spawn is quiet"
        );
    }

    #[test]
    fn the_unavailable_message_says_what_broke_and_what_to_do() {
        let msg = gpu_became_unavailable_message();
        assert!(
            msg.contains("graphics card"),
            "must name the thing that broke, in the owner's words: {msg}"
        );
        assert!(
            msg.to_lowercase().contains("driver"),
            "must name the cause, or a restart looks arbitrary: {msg}"
        );
        assert!(
            msg.contains("Restart") || msg.contains("restart"),
            "must give the one action that fixes it: {msg}"
        );
    }

    #[test]
    fn the_unavailable_message_does_not_promise_the_processor_takes_over() {
        // The obvious reassurance is FALSE here and was nearly shipped. This
        // binary links `libcuda.so.1`, so a bad library stops EVERY new process
        // in the loader: a CPU-only worker exits 127 exactly as a GPU one does
        // (verified 2026-09-18). `unsupported_gpu_message` may promise the
        // processor because there the card is merely old and workers start
        // fine; here the same sentence would tell someone their node is still
        // serving while it serves nothing.
        let msg = gpu_became_unavailable_message().to_lowercase();
        for false_promise in ["processor", "cpu", "still works", "just slower"] {
            assert!(
                !msg.contains(false_promise),
                "must not claim the processor takes over ({false_promise:?}): {msg}"
            );
        }
    }

    #[tokio::test]
    async fn a_binary_that_runs_is_reported_as_startable() {
        // `echo --version` exits 0 and prints something, which is all the
        // probe asks: it is testing that a new process can be created at all.
        let ok = executable_still_starts(std::path::Path::new("/bin/echo")).await;
        assert!(ok.is_ok(), "a runnable binary must probe clean: {ok:?}");
    }

    #[tokio::test]
    async fn a_binary_that_cannot_be_run_is_reported_with_its_reason() {
        let missing = std::path::Path::new("/nonexistent/swarmllm-not-here");
        let err = executable_still_starts(missing)
            .await
            .expect_err("a path that does not exist cannot start");
        // The reason has to survive to the log: the whole point is that the
        // 2026-09-18 outage had its cause on disk and unattributed.
        assert!(
            !err.is_empty(),
            "the failure must carry something a reader can act on"
        );
    }

    #[test]
    fn unparseable_output_is_unknown_not_unsupported() {
        // Every one of these must reach the caller as None so it treats the
        // capability as unknown and leaves the GPU alone. A card that works
        // must never be sent to the CPU because nvidia-smi was odd.
        for junk in [
            "",
            "N/A",
            "8",
            "eight.six",
            "Failed to initialize NVML: Driver/library version mismatch",
        ] {
            assert_eq!(parse_compute_cap(junk), None, "input {junk:?}");
        }
    }

    #[test]
    fn the_message_names_the_card_and_says_what_happens_next() {
        let msg = unsupported_gpu_message("NVIDIA GeForce RTX 2060", (7, 5));
        assert!(msg.contains("RTX 2060"), "must name the card: {msg}");
        assert!(msg.contains("7.5"), "must state what it has: {msg}");
        assert!(msg.contains("8.0"), "must state what is needed: {msg}");
        assert!(
            msg.contains("processor"),
            "must say the node keeps working, not just that it failed: {msg}"
        );
        assert!(
            msg.contains("RTX 30-series"),
            "must translate the number into something checkable against the box: {msg}"
        );
        // No pre-Ampere CUDA asset is published, so this must not send anyone
        // looking for one. Cheap to assert, and the kind of promise that rots
        // silently once someone edits the copy.
        for absent in ["releases page", "download", "older cards"] {
            assert!(
                !msg.contains(absent),
                "must not promise a build that does not exist ({absent:?}): {msg}"
            );
        }
    }
}
