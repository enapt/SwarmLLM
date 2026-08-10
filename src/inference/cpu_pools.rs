//! Separate CPU thread pools for the two phases of inference.
//!
//! # Why
//!
//! Reading a prompt and writing a reply want opposite thread counts, and one
//! pool cannot serve both. Measured on an 8-physical-core Ryzen 7 5800H,
//! llama-3.2-3b Q4_K_M, 896-token prompt (`examples/prefill_bench.rs`):
//!
//! | threads | prompt processing tok/s | decode tok/s |
//! |---|---|---|
//! | 2  | 12.98 | 3.92 |
//! | 3  | 18.53 | 4.54 |
//! | **4**  | 23.64 | **5.26** |
//! | 6  | 30.78 | 5.11 |
//! | 8  | 35.59 | 4.56 |
//! | 12 | 41.78 | 3.54 |
//! | **14** | **43.25** | 2.94 |
//! | 16 | 43.09 | 2.64 |
//!
//! Prompt processing scales to **1.83x** past decode's optimum; decode peaks at
//! 4 and then falls off a cliff, **2.0x worse at 14**. So a node given more of
//! its owner's machine got faster at reading prompts and *slower at replying* —
//! a setting that asks people to contribute more compute made the thing they
//! notice most get worse. That is the defect this fixes.
//!
//! # Why decode falls off, and what the cap may therefore assume
//!
//! Decode is memory-bandwidth-bound, not compute-bound: one generated token
//! streams the whole weight set through the CPU to do one row of arithmetic,
//! measured at 22.1 GB/s against 31-33 GB/s achievable on the box above (69% of
//! roofline — see `docs/FUTURE_WORK.md`). Threads beyond the number needed to
//! saturate memory bandwidth add contention without adding bandwidth.
//!
//! **The count that saturates bandwidth is a property of the machine, and a
//! fraction of the core count does not predict it.** The first version of this
//! module capped decode at `max(4, physical/2)`, which is right for the Ryzen
//! above and WRONG for a second machine measured on 2026-08-08 — an Intel
//! i5-10500T (6 physical / 12 logical, DDR4-2666, 35 W), where decode climbs
//! monotonically to all six cores:
//!
//! | threads | prompt processing tok/s | decode tok/s |
//! |---|---|---|
//! | 2 | 12.43 | 4.37 |
//! | 3 | 16.82 | 5.36 |
//! | 4 | 20.70 | 5.76 |
//! | 5 | 22.62 | 6.24 |
//! | **6** | **28.50** | **7.10** |
//!
//! Capping that machine at 4 would have made its replies **23% slower**. The
//! mechanism explains both: a Zen 3 core pulls ~10-12 GB/s so three or four
//! saturate the Ryzen's ~32 GB/s, while a 35 W Comet Lake core at 2.3 GHz pulls
//! far less and six do not saturate its ~41 GB/s. Peak threads = bandwidth
//! divided by per-core draw, which core count alone cannot tell you.
//!
//! So this now caps at what BOTH machines support and the mechanism predicts:
//! **decode never uses more threads than there are physical cores.** SMT
//! siblings share a core's load/store ports, so they add contention to a
//! bandwidth-bound loop without adding a path to memory. Neither machine is
//! harmed by that rule, and it recovers the large loss when someone sets
//! `max_cpu_threads` to their logical count — a natural thing to do.
//!
//! What it deliberately does NOT do is guess the sub-physical optimum. On the
//! Ryzen that leaves 4 threads' worth of decode speed on the table (5.26 vs
//! 4.56 tok/s at 8). Recovering it needs calibration on the machine itself, not
//! a better constant; see `docs/FUTURE_WORK.md`.
//!
//! Prefill is left on the global pool untouched, so a node on the default
//! `contribution = "minimal"` takes exactly the code path it did before, with
//! no pool and no `install`.

use std::sync::OnceLock;

/// How many threads decode should use, given what the owner offered and what
/// the machine has.
///
/// `offered` is the global rayon pool size — the contribution ceiling the
/// daemon already applied. `physical` is physical cores, NOT logical: that is
/// the whole rule. See the module docs for the two machines this is drawn
/// from, and for why a fraction of the core count was wrong.
///
/// Pure so the policy is testable without building pools or owning a CPU.
pub(crate) fn decode_threads(offered: usize, physical: usize) -> usize {
    // A cap, never a request: `offered` is what the owner agreed to give.
    offered.max(1).min(physical.max(1))
}

/// Explicit override, read once. `0` disables the cap (decode uses the global
/// pool, i.e. the behaviour before this module existed).
fn env_decode_threads() -> Option<usize> {
    static OVERRIDE: OnceLock<Option<usize>> = OnceLock::new();
    *OVERRIDE.get_or_init(|| {
        std::env::var("SWARMLLM_DECODE_THREADS")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
    })
}

/// The decode pool, or `None` when decode should just use the global pool.
///
/// `None` is the common case and matters: at the default contribution the
/// ceiling already equals decode's optimum, so there is no second pool, no
/// extra threads, and no `install` on the hot path.
fn decode_pool() -> Option<&'static rayon::ThreadPool> {
    static POOL: OnceLock<Option<rayon::ThreadPool>> = OnceLock::new();
    POOL.get_or_init(|| {
        let offered = rayon::current_num_threads();
        let want = match env_decode_threads() {
            Some(0) => return None,
            Some(n) => n.min(offered),
            None => decode_threads(offered, num_cpus::get_physical()),
        };
        if want >= offered {
            return None;
        }
        match rayon::ThreadPoolBuilder::new()
            .num_threads(want)
            .thread_name(|i| format!("swarm-decode-{i}"))
            .build()
        {
            Ok(p) => {
                tracing::info!(
                    decode_threads = want,
                    offered,
                    physical = num_cpus::get_physical(),
                    "Decode runs on a narrower thread pool than prompt processing — decode is \
                     bandwidth-bound and slows down with more threads"
                );
                Some(p)
            }
            Err(e) => {
                // Not fatal: falling back to the global pool is exactly the
                // previous behaviour, so a pool that cannot be built costs
                // performance and nothing else.
                tracing::warn!(error = %e, "Could not build the decode thread pool — using the global one");
                None
            }
        }
    })
    .as_ref()
}

/// Run one forward pass on the pool that suits its phase.
///
/// `seq_len` is the number of query positions: `1` is decode (one new token
/// against the cache), anything more is prompt processing. That is the same
/// predicate the attention dispatch uses to pick a kernel, and for the same
/// underlying reason — the two phases are different shapes of work.
///
/// Prefill returns `f()` directly, so it keeps the global pool and pays
/// nothing.
pub(crate) fn in_phase_pool<R: Send>(seq_len: usize, f: impl FnOnce() -> R + Send) -> R {
    if seq_len != 1 {
        return f();
    }
    match decode_pool() {
        Some(pool) => pool.install(f),
        None => f(),
    }
}

/// Is the machine currently too hot? Observed by [`super::thermal`] and reported
/// to the user; it does NOT narrow the pools.
///
/// **A thermal throttle was built here on 2026-08-10 and removed the same day
/// because it measurably did nothing.** Routing both phases through a
/// half-width pool while hot left CPU usage unchanged — 744% peak against 741%,
/// wall 118 s against 115 s, on llama-3.2-3b Q4_K_M with a ~700-token prompt at
/// `contribution = "maximum"`. The pool was genuinely built and genuinely
/// installed (its `swarm-cool-*` threads are visible in `/proc/<pid>/task`), yet
/// the work kept running ~8 threads wide, so `install` is not confining
/// candle's `par_chunks_mut` on this path and the reason is not yet known. See
/// `docs/FUTURE_WORK.md` § "Thermal throttling had no measurable effect".
///
/// Kept because the *observation* is worth having on its own: the user is told
/// the machine is hot, which is what nothing did before.
static MACHINE_IS_HOT: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub(crate) fn machine_is_hot() -> bool {
    MACHINE_IS_HOT.load(std::sync::atomic::Ordering::Relaxed)
}

/// Record the hot/not-hot state. Returns whether it changed, so the caller can
/// log a transition rather than a level.
pub fn set_machine_is_hot(hot: bool) -> bool {
    MACHINE_IS_HOT.swap(hot, std::sync::atomic::Ordering::Relaxed) != hot
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The rule, stated once: never more threads than physical cores, and
    /// never more than the owner offered.
    #[test]
    fn decode_never_uses_smt_siblings() {
        // Ryzen 5800H, 8 physical / 16 logical.
        assert_eq!(
            decode_threads(16, 8),
            8,
            "logical count offered -> physical"
        );
        assert_eq!(decode_threads(14, 8), 8);
        assert_eq!(decode_threads(8, 8), 8, "already at physical -> unchanged");
        // Intel i5-10500T, 6 physical / 12 logical.
        assert_eq!(decode_threads(12, 6), 6);
        assert_eq!(decode_threads(6, 6), 6);
    }

    /// **The regression this rule exists to avoid.** An earlier version capped
    /// at `max(4, physical/2)`, which measured correctly on the Ryzen and would
    /// have made the 6-core Intel 23% slower at generating (5.76 tok/s at four
    /// threads against 7.10 at six). A fraction of the core count does not
    /// predict where memory bandwidth saturates.
    #[test]
    fn a_six_core_machine_keeps_all_six_for_decode() {
        assert_eq!(decode_threads(6, 6), 6);
        assert_ne!(decode_threads(6, 6), 4, "the measured regression");
    }

    /// Every contribution level is at or below physical cores, so none of them
    /// changes behaviour — this only bites when someone sets `max_cpu_threads`
    /// above the physical count.
    #[test]
    fn no_contribution_level_is_affected() {
        for physical in [2usize, 4, 6, 8, 16, 64] {
            for fraction in [0.5f64, 0.75, 1.0] {
                let offered = ((physical as f64 * fraction).round() as usize).max(1);
                assert_eq!(
                    decode_threads(offered, physical),
                    offered,
                    "physical={physical} offered={offered} must be untouched"
                );
            }
        }
    }

    /// It is a cap on what the owner offered, never a request for more. A node
    /// told to contribute two threads must not quietly run more.
    #[test]
    fn never_exceeds_what_was_offered() {
        for offered in 1..=64 {
            for physical in [1usize, 2, 4, 6, 8, 16, 64] {
                assert!(
                    decode_threads(offered, physical) <= offered,
                    "offered={offered} physical={physical}"
                );
            }
        }
    }

    /// Zero and one must not produce a 0-thread pool — rayon reads 0 as "every
    /// core", the behaviour being fixed.
    #[test]
    fn degenerate_inputs_stay_sane() {
        assert!(decode_threads(0, 0) >= 1);
        assert_eq!(decode_threads(1, 1), 1);
        assert!(decode_threads(8, 0) >= 1);
    }
}
