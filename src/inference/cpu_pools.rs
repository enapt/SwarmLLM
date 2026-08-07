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
//! # Why decode falls off, and what that implies for the cap
//!
//! Decode is memory-bandwidth-bound, not compute-bound: one generated token
//! streams the whole weight set through the CPU to do one row of arithmetic,
//! measured at 22.1 GB/s against 31-33 GB/s achievable on this box (69% of
//! roofline — see `docs/FUTURE_WORK.md`). Threads beyond the number needed to
//! saturate memory bandwidth add contention and cache pressure without adding
//! bandwidth, so they cost rather than pay.
//!
//! **The count that saturates bandwidth is a property of the machine, and this
//! was measured on exactly one.** [`decode_threads`] therefore only ever
//! reduces below what the owner offered — never above — and never below
//! [`DECODE_THREAD_FLOOR`], so it cannot hurt a small machine. On a very wide
//! server it is likely still too generous, which leaves that machine no worse
//! off than before. `SWARMLLM_DECODE_THREADS` overrides it outright.
//!
//! Prefill is left on the global pool untouched, so the common case — a node on
//! the default `contribution = "minimal"`, whose ceiling already equals decode's
//! optimum — takes exactly the code path it did before, with no pool and no
//! `install`.

use std::sync::OnceLock;

/// Decode never gets fewer threads than this, whatever the arithmetic says.
///
/// The measured optimum here is 4, and a machine with 4 or fewer physical cores
/// is measurably worse below that (3.92 tok/s at 2 threads against 5.26 at 4).
/// A rule derived from one 8-core box must not make small machines slower.
pub(crate) const DECODE_THREAD_FLOOR: usize = 4;

/// How many threads decode should use, given what the owner offered and what
/// the machine has.
///
/// `offered` is the global rayon pool size — the contribution ceiling the
/// daemon already applied. `physical` is physical cores, not logical: the
/// bandwidth argument counts memory-attached cores, and SMT siblings share a
/// port rather than adding one.
///
/// Pure so the policy is testable without building pools or owning a CPU.
pub(crate) fn decode_threads(offered: usize, physical: usize) -> usize {
    let offered = offered.max(1);
    // Half the physical cores, floored so small machines are untouched, and
    // never more than was offered — this is a cap, not a request.
    offered.min(DECODE_THREAD_FLOOR.max(physical.max(1) / 2))
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

#[cfg(test)]
mod tests {
    use super::*;

    /// The measured case: 8 physical cores, all offered. Decode is capped at
    /// its optimum instead of taking everything.
    #[test]
    fn caps_decode_on_the_machine_this_was_measured_on() {
        assert_eq!(decode_threads(8, 8), 4);
        assert_eq!(decode_threads(16, 8), 4, "logical cores offered, still 4");
    }

    /// The default contribution already lands on decode's optimum, so nothing
    /// changes and no second pool is built.
    #[test]
    fn default_contribution_is_unchanged() {
        // `contribution = "minimal"` gives half the physical cores.
        assert_eq!(decode_threads(4, 8), 4);
    }

    /// A rule derived from one 8-core box must never make a small machine
    /// slower — 2 threads measured 3.92 tok/s against 5.26 at 4.
    #[test]
    fn never_reduces_below_the_floor_on_small_machines() {
        for physical in 1..=8 {
            let offered = physical;
            let got = decode_threads(offered, physical);
            assert!(
                got >= offered.min(DECODE_THREAD_FLOOR),
                "physical={physical} offered={offered} got={got}"
            );
        }
        assert_eq!(decode_threads(2, 2), 2, "a 2-core box keeps both");
        assert_eq!(decode_threads(4, 4), 4, "a 4-core box keeps all four");
    }

    /// It is a cap on what the owner offered, never a request for more. A node
    /// set to contribute two threads must not quietly run four.
    #[test]
    fn never_exceeds_what_was_offered() {
        for offered in 1..=64 {
            for physical in [1usize, 2, 4, 8, 16, 64] {
                assert!(
                    decode_threads(offered, physical) <= offered,
                    "offered={offered} physical={physical}"
                );
            }
        }
    }

    /// A wide machine gets a real reduction rather than the whole box.
    #[test]
    fn a_wide_machine_is_capped() {
        assert_eq!(decode_threads(64, 64), 32);
        assert_eq!(decode_threads(32, 16), 8);
    }

    /// Zero and one are not special-cased away into a panic or a 0-thread pool
    /// (rayon reads 0 as "every core", the behaviour being fixed).
    #[test]
    fn degenerate_inputs_stay_sane() {
        assert!(decode_threads(0, 0) >= 1);
        assert_eq!(decode_threads(1, 1), 1);
    }
}
