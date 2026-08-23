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
//! # The sub-physical optimum is measured, not guessed (2026-08-22)
//!
//! The cap above is a ceiling, and on a machine whose cores each pull a lot of
//! bandwidth the real optimum sits well below it. That was known and deferred
//! when the loss was 15% (5.26 vs 4.56 tok/s at 8 on the Ryzen). **The v0.3.112
//! CPU kernels changed the premise**: with the arithmetic per byte cut, decode
//! became far more bandwidth-bound and the same comparison is now much wider.
//! Re-measured 2026-08-22, llama-3.2-3b Q4_K_M, 256-token prompt, min of 2:
//!
//! | decode threads | tok/s |
//! |---|---|
//! | 1 | 6.8 |
//! | 2 | 12.7 |
//! | **4** | **18.9** |
//! | 6 | 15.0 |
//! | 8 (= the cap here, so the global pool) | 8.3 |
//!
//! A node set to `contribution = "maximum"` therefore replied **much slower**
//! than the same node on the default `minimal`, because minimal's ceiling of 4
//! happens to land on the optimum. Giving the swarm more of your machine made
//! your own replies worse — the same defect this module was written to fix,
//! reopened by making the kernels faster.
//!
//! No constant fixes it, and the second machine proves it: on the i5-10500T
//! llama-3.2-1b Q8_0 still climbs monotonically to all six cores (4.9 → 13.95
//! tok/s) while tinyllama Q4_K_M on that same box peaks around three. The
//! optimum depends on the machine AND the model, so it is now **calibrated at
//! run time**: the first decode steps of a worker's life are timed round-robin
//! across candidate widths and the best is kept. A worker process serves one
//! model, so a process-global calibration is per-model by construction.
//!
//! The measurement is free — those are real tokens the user asked for, not a
//! synthetic probe — and self-correcting: it only moves off the widest
//! candidate when the margin exceeds `CALIBRATION_MARGIN_PCT`, so noise leaves
//! the previous behaviour in place.
//!
//! Measured A/B inside one binary (`SWARMLLM_DECODE_CALIBRATE=0` is the off
//! arm), 3 interleaved reps per arm, on BOTH machines — which is the part that
//! matters, because the first version of this passed on one machine and
//! regressed the other.
//!
//! Ryzen 5800H, llama-3.2-3b Q4_K_M, 896-token prompt, `offered = 8`
//! (`contribution = "maximum"`), the case this exists for:
//!
//! | calibration off | on | it chose | its own timings |
//! |---|---|---|---|
//! | 7.68-7.94 tok/s | **14.17-14.52** | 4 of 8, every run | 8:126-133 6:74-82 4:59-66 2:83-92 ms |
//!
//! **1.85-1.89x**, prompt processing unchanged (both arms use everything
//! offered). i5-10500T, `offered = 6`: it keeps **6 of 6 on every run** for both
//! tinyllama Q4_K_M and llama-3.2-1b Q8_0 — the right answer there, and it must
//! not be talked out of it by noise.
//!
//! # Deciding, and why the obvious rules are wrong
//!
//! Comparisons use the MEDIAN of `SAMPLES_PER_CANDIDATE`, and a width is only
//! taken when its WORST timing still beats the offered width's typical one.
//! Both of those were paid for:
//!
//! - **Min-of-N is wrong here**, though it is this project's rule for
//!   benchmarks. There the environment is controlled and every error adds time,
//!   so the fastest run is the least contaminated. Here each sample is a
//!   different token, at a different cache length, on a machine doing whatever
//!   else its owner asked — so the minimum is the LUCKIEST sample. On the i5 it
//!   picked 4 of 6 on one run and 6 of 6 on the next for the same model, and
//!   cost 6-8%.
//! - **A percentage margin alone is wrong too.** On the i5 the gap between
//!   widths (~16 ms) is the size of the spread of ONE width re-measured (6
//!   threads: 48, 52, 36 ms across three runs), so any threshold small enough to
//!   catch a real win also catches noise. On the Ryzen the gap is 69 ms against
//!   a ~3 ms spread. The question is "is the gain bigger than this machine's own
//!   noise", and 15% only happened to be a proxy for that on one box.
//!
//! # What it costs, and why a benchmark overstates it
//!
//! The measurement is paid in the user's own first tokens, ONCE per worker
//! process. A bench pays that inside a single short run; a real worker pays it
//! once and then serves for hours. Measured on the i5 (tinyllama, same binary,
//! calibration on vs off):
//!
//! | decode tokens | on | off |
//! |---|---|---|
//! | 48 | 25.38 tok/s | 27.87 (**-8.9%**) |
//! | 256 | 26.18 | 25.83 (**level**) |
//!
//! So the visible cost is a short-run artefact. It is minimised anyway:
//! `ELIMINATION_RATIO` drops a hopeless width after one look, and
//! `EARLY_SETTLE_SAMPLES` stops the whole search once the offered width is
//! holding its own — the common case, since most machines have no sub-physical
//! optimum worth taking.
//!
//! Prefill is left on the global pool untouched, so a node on the default
//! `contribution = "minimal"` takes exactly the code path it did before, with
//! no pool and no `install`.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

/// Timed decode steps per candidate width before a choice is made. Three is
/// enough to take a min that survives one scheduler hiccup, and costs only the
/// first ~dozen tokens of a worker's life.
const SAMPLES_PER_CANDIDATE: usize = 5;

/// How much faster a narrower pool must be before decode moves off the widest
/// candidate. Below this the two are indistinguishable on a busy machine and
/// the previous behaviour (use what the owner offered) stands.
///
/// **Sized for the flat case, not the sharp one.** Where narrowing genuinely
/// helps it helps enormously — the Ryzen measures 60 ms against 129 ms, which
/// clears any threshold — so nothing is lost by being strict. Where the curve
/// is flat, as on the i5 (23.7 vs 22.5 tok/s across four widths, ~7% end to
/// end), a small margin just lets sampling noise pick a winner, and the choice
/// came out differently on two consecutive runs of the same model. 3% did
/// exactly that.
const CALIBRATION_MARGIN_PCT: u64 = 15;

/// After one full cycle, a candidate this much slower than the best so far is
/// dropped instead of being sampled again.
///
/// Calibration is paid for out of the user's own first tokens, so its cost is
/// worst on a short reply. On the Ryzen the narrowest candidate measured 156 ms
/// against 70 ms — hopeless after one look, and re-timing it twice more was
/// most of the price. 3/2 is deliberately loose: it only ever discards a
/// candidate that lost badly, never one that is merely behind.
const ELIMINATION_RATIO: (u64, u64) = (3, 2);

/// Samples per candidate after which, if nothing is beating the width the owner
/// offered, calibration stops and keeps it.
///
/// Most machines have no sub-physical optimum worth taking — the i5 measured
/// here wants all six cores for both its models — and on those every further
/// sample is pure cost to the user, paid in their own first tokens. Settling
/// early takes that from ~15 tokens to ~6. The sharp case is unaffected: where
/// a narrower width really wins it is already ahead by 2x after two samples, so
/// this never fires and the full comparison runs.
const EARLY_SETTLE_SAMPLES: usize = 2;

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

/// Candidate decode widths for this machine, widest first.
///
/// Anchored on [`decode_threads`] — the cap is still a cap, calibration only
/// chooses beneath it — and thinned out to quarters so at most four pools are
/// ever built. Both measured optima are members: 4 of the Ryzen's 8, and all 6
/// of the i5's 6 (the widest candidate is always present).
pub(crate) fn decode_candidates(offered: usize, physical: usize) -> Vec<usize> {
    let cap = decode_threads(offered, physical);
    let mut v: Vec<usize> = [cap, cap * 3 / 4, cap / 2, cap / 4]
        .into_iter()
        .map(|n| n.max(1))
        .collect();
    v.sort_unstable_by(|a, b| b.cmp(a));
    v.dedup();
    v
}

/// Which candidate to time next, and the bookkeeping to decide between them.
///
/// The cursor alternates direction every cycle. That matters: the KV cache
/// grows with every token, so decode gets steadily slower on its own, and a
/// fixed round-robin order would hand the first candidate every short-KV slot
/// and the last one every long-KV slot — measuring position, not width.
struct Calibration {
    candidates: Vec<usize>,
    cursor: AtomicUsize,
    /// Decided width, or 0 while still measuring.
    chosen: AtomicUsize,
    state: Mutex<CalibrationState>,
}

struct CalibrationState {
    /// Every timing seen per candidate. Compared by MEDIAN, deliberately not by
    /// minimum.
    ///
    /// Min-of-N is this project's rule for benchmarks, where the environment is
    /// controlled and every error source only adds time — there the fastest run
    /// is the least contaminated. It is the wrong statistic HERE: each sample is
    /// a different token, at a different cache length, on a machine doing
    /// whatever else its owner asked of it. The minimum is then the luckiest
    /// sample rather than the truest one, and on a machine whose widths are
    /// close together the luckiest sample decides the winner. Measured on the
    /// i5 (2026-08-22): the same model chose 4 of 6 on one run and 6 of 6 on the
    /// next, from single samples that disagreed by 30%.
    samples_ns: Vec<Vec<u64>>,
    counts: Vec<usize>,
    /// Candidates that lost their first cycle badly enough not to be re-timed.
    eliminated: Vec<bool>,
}

/// Median of a set of timings, or `None` when there are none.
///
/// See `CalibrationState::samples_ns` for why this is a median and not a
/// minimum.
fn median_ns(v: &[u64]) -> Option<u64> {
    if v.is_empty() {
        return None;
    }
    let mut sorted = v.to_vec();
    sorted.sort_unstable();
    Some(sorted[sorted.len() / 2])
}

impl Calibration {
    fn new(candidates: Vec<usize>) -> Self {
        let n = candidates.len();
        Self {
            candidates,
            cursor: AtomicUsize::new(0),
            chosen: AtomicUsize::new(0),
            state: Mutex::new(CalibrationState {
                samples_ns: vec![Vec::new(); n],
                counts: vec![0; n],
                eliminated: vec![false; n],
            }),
        }
    }

    /// Report the decision once, the same way from either exit.
    fn announce(&self, st: &CalibrationState, final_idx: usize, medians: &[Option<u64>]) {
        let _ = st;
        let table: Vec<String> = self
            .candidates
            .iter()
            .zip(medians.iter())
            .map(|(t, ns)| format!("{t}:{}ms", ns.map(|v| v / 1_000_000).unwrap_or(0)))
            .collect();
        tracing::info!(
            decode_threads = self.candidates[final_idx],
            offered = self.candidates[0],
            measured = %table.join(" "),
            "Decode thread width calibrated on this machine from real tokens"
        );
        // The benches run without a tracing subscriber, and a calibration you
        // cannot see is a calibration you cannot check.
        if std::env::var("SWARMLLM_DECODE_CALIBRATE_VERBOSE").is_ok() {
            eprintln!(
                "CALIB chose {} of {} — {}",
                self.candidates[final_idx],
                self.candidates[0],
                table.join(" ")
            );
        }
    }

    /// Index of the candidate to time on this call, skipping any already
    /// eliminated. Falls back to the raw rotation if the state is unreadable.
    fn next_index(&self) -> usize {
        let n = self.candidates.len();
        let raw = |step: usize| {
            let (cycle, pos) = (step / n, step % n);
            if cycle % 2 == 0 {
                pos
            } else {
                n - 1 - pos
            }
        };
        let step = self.cursor.fetch_add(1, Ordering::Relaxed);
        let Ok(st) = self.state.lock() else {
            return raw(step);
        };
        if st.eliminated.iter().all(|e| !e) {
            return raw(step);
        }
        // Walk forward to the next live candidate so the rotation still
        // alternates over whichever ones remain.
        for extra in 0..n {
            let idx = raw(step + extra);
            if !st.eliminated[idx] {
                return idx;
            }
        }
        raw(step)
    }

    /// Record one timing and, once every candidate has been seen enough, settle.
    fn record(&self, idx: usize, ns: u64) {
        let mut st = match self.state.lock() {
            Ok(g) => g,
            // A poisoned lock means another thread panicked mid-forward. Losing
            // calibration is not worth propagating that: keep the widest.
            Err(_) => return,
        };
        st.counts[idx] += 1;
        st.samples_ns[idx].push(ns);
        // Once every candidate has been seen once, drop the hopeless ones so
        // the user does not pay to re-time them. One sample is enough to spot
        // hopeless — it is not enough to pick a winner, which is why only the
        // elimination step reads a single measurement.
        if st.counts.iter().all(|c| *c >= 1) {
            if let Some(best) = st
                .samples_ns
                .iter()
                .filter_map(|v| v.iter().min())
                .min()
                .copied()
            {
                for i in 0..st.samples_ns.len() {
                    if let Some(v) = st.samples_ns[i].iter().min().copied() {
                        if v * ELIMINATION_RATIO.1 > best * ELIMINATION_RATIO.0 {
                            st.eliminated[i] = true;
                        }
                    }
                }
            }
        }
        // Early settle: if the offered width is holding its own after a couple
        // of looks, there is nothing here worth the user's tokens to find.
        let live_seen = |n: usize| {
            st.counts
                .iter()
                .zip(st.eliminated.iter())
                .all(|(c, dead)| *dead || *c >= n)
        };
        if live_seen(EARLY_SETTLE_SAMPLES) {
            let medians: Vec<Option<u64>> = st.samples_ns.iter().map(|v| median_ns(v)).collect();
            let best_live = medians
                .iter()
                .enumerate()
                .filter(|(i, _)| !st.eliminated[*i])
                .filter_map(|(_, ns)| *ns)
                .min();
            if let (Some(w), Some(b)) = (medians[0], best_live) {
                if b * 100 > w * (100 - CALIBRATION_MARGIN_PCT) {
                    self.chosen.store(self.candidates[0], Ordering::Relaxed);
                    self.announce(&st, 0, &medians);
                    return;
                }
            }
        }
        let unfinished = st
            .counts
            .iter()
            .zip(st.eliminated.iter())
            .any(|(c, dead)| !*dead && *c < SAMPLES_PER_CANDIDATE);
        if unfinished {
            return;
        }
        let medians: Vec<Option<u64>> = st.samples_ns.iter().map(|v| median_ns(v)).collect();
        let widest_ns = medians[0];
        let best = medians
            .iter()
            .enumerate()
            .filter(|(i, _)| !st.eliminated[*i])
            .filter_map(|(i, ns)| ns.map(|v| (i, v)))
            .min_by_key(|(_, v)| *v);
        let (pick_idx, pick_ns) = match (best, widest_ns) {
            (Some(b), Some(_)) => b,
            _ => (0, 0),
        };
        // Only move off the widest when the gain is real. Two conditions, and
        // the second is the one that separates the two machines this was
        // measured on:
        //
        //   1. a percentage margin, so a hair's difference is ignored; and
        //   2. the candidate's WORST timing must still beat the widest's
        //      typical one.
        //
        // A percentage alone cannot tell the cases apart. On the flat i5 the
        // gap between widths (~16 ms) is the same size as the spread of one
        // width measured repeatedly (6 threads came out at 48, 52 and 36 ms
        // across three runs), so any threshold small enough to catch a real
        // win also catches noise — and it did, moving to 3 threads on one run
        // in six. On the Ryzen the gap is 69 ms against a spread of ~2 ms.
        // "Bigger than the noise" is the question; "bigger than 15%" only
        // happened to be a proxy for it on one machine.
        let pick_worst = st.samples_ns[pick_idx].iter().max().copied();
        let keep_widest = match (widest_ns, pick_worst) {
            (Some(w), Some(worst)) => {
                pick_ns * 100 > w * (100 - CALIBRATION_MARGIN_PCT) || worst >= w
            }
            _ => true,
        };
        let final_idx = if keep_widest { 0 } else { pick_idx };
        self.chosen
            .store(self.candidates[final_idx], Ordering::Relaxed);
        self.announce(&st, final_idx, &medians);
    }
}

/// One rayon pool per candidate width, built on first use and shared after.
///
/// `None` means "use the global pool", which is what the widest candidate
/// always resolves to — so the common path still builds nothing.
fn pool_for(threads: usize) -> Option<Arc<rayon::ThreadPool>> {
    static POOLS: OnceLock<Mutex<HashMap<usize, Option<Arc<rayon::ThreadPool>>>>> = OnceLock::new();
    if threads >= rayon::current_num_threads() {
        return None;
    }
    let map = POOLS.get_or_init(|| Mutex::new(HashMap::new()));
    let mut guard = map.lock().ok()?;
    guard
        .entry(threads)
        .or_insert_with(|| {
            match rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .thread_name(move |i| format!("swarm-decode{threads}-{i}"))
                .build()
            {
                Ok(p) => Some(Arc::new(p)),
                Err(e) => {
                    // Falling back to the global pool is the previous
                    // behaviour, so this costs speed and nothing else.
                    tracing::warn!(error = %e, threads, "Could not build a decode thread pool — using the global one");
                    None
                }
            }
        })
        .clone()
}

/// The calibrator, or `None` when there is nothing to choose between.
fn calibration() -> Option<&'static Calibration> {
    static CALIB: OnceLock<Option<Calibration>> = OnceLock::new();
    CALIB
        .get_or_init(|| {
            // An explicit width (or an explicit 0) is a decision already made.
            if env_decode_threads().is_some() || !env_calibration_enabled() {
                return None;
            }
            let candidates =
                decode_candidates(rayon::current_num_threads(), num_cpus::get_physical());
            if candidates.len() < 2 {
                return None;
            }
            Some(Calibration::new(candidates))
        })
        .as_ref()
}

/// `SWARMLLM_DECODE_CALIBRATE=0` pins the previous behaviour, for A/B inside
/// one binary — the same discipline as `SWARMLLM_FORCE_STANDARD_ATTN`.
fn env_calibration_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        !matches!(
            std::env::var("SWARMLLM_DECODE_CALIBRATE").ok().as_deref(),
            Some("0")
        )
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
    if seq_len > DECODE_SHAPED_MAX_TOKENS {
        return f();
    }
    let Some(calib) = calibration() else {
        // Pinned width, or nothing to choose between: the original path.
        return match decode_pool() {
            Some(pool) => pool.install(f),
            None => f(),
        };
    };
    let settled = calib.chosen.load(Ordering::Relaxed);
    if settled != 0 {
        return match pool_for(settled) {
            Some(pool) => pool.install(f),
            None => f(),
        };
    }
    if seq_len != 1 {
        // Decode-SHAPED but not a decode token. It gets the narrow pool, and it
        // must NOT be timed: the calibration compares candidate widths by the
        // cost of one token, and a sample from a different query length is not
        // that measurement. Mixing shapes into the rotation is the same class of
        // error as taking the fastest sample instead of a typical one — it makes
        // the comparison decide something other than what it is asked (#367).
        return match decode_pool() {
            Some(pool) => pool.install(f),
            None => f(),
        };
    }
    // Still measuring: time this token on the next candidate in the rotation.
    let idx = calib.next_index();
    let threads = calib.candidates[idx];
    let started = Instant::now();
    let out = match pool_for(threads) {
        Some(pool) => pool.install(f),
        None => f(),
    };
    calib.record(idx, started.elapsed().as_nanos() as u64);
    out
}

/// Longest query block still treated as decode-shaped, i.e. narrow-pool work.
///
/// The phase predicate used to be `seq_len == 1`, which reads as the obvious
/// definition of decode and is the wrong question to ask here. What the pool
/// choice actually turns on is whether the matmuls are bandwidth-bound, and
/// they stay bandwidth-bound well past one row: a speculative verify of 4
/// tokens re-reads the same weights a single token does, it just brings three
/// more activation rows along.
///
/// Measured on this machine (Ryzen 7 5800H, 8 physical / 16 logical) with
/// `examples/qmatmul_bench`, ms for one Q4_K projection (k=3072, n=8192):
///
/// ```text
///   threads    m=1     m=2     m=4     m=8    m=16    m=32   m=128
///         8   0.157   0.273   0.349   0.844   1.644   2.943  10.844
///        16   0.339   0.424   0.629   1.564   3.096   4.207  10.941
/// ```
///
/// The narrow pool wins at every width up to 32 — by 1.4x-2.2x — and the two
/// draw level at 128. So the boundary is not sharp and does not need to be:
/// anywhere in 32..128 costs nothing either way, while classifying a 2-token
/// forward as prompt processing costs ~1.5x. 32 sits at the last width with a
/// measured margin.
///
/// Prompt chunks (`inference.prefill_chunk_tokens`, 128 by default) stay on the
/// global pool, which is what the prefill measurements behind `cpu_pools` were
/// taken on.
const DECODE_SHAPED_MAX_TOKENS: usize = 32;

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

    /// The cap is still the ceiling, and both machines' measured optima are
    /// candidates: 4 of the Ryzen's 8, and all 6 of the i5's 6.
    #[test]
    fn candidates_span_the_measured_optima_and_never_exceed_the_cap() {
        let ryzen = decode_candidates(8, 8);
        assert_eq!(
            ryzen[0], 8,
            "the widest candidate is what the owner offered"
        );
        assert!(
            ryzen.contains(&4),
            "the Ryzen's measured optimum: {ryzen:?}"
        );
        let i5 = decode_candidates(6, 6);
        assert_eq!(
            i5[0], 6,
            "the i5 wants all six and must be able to keep them"
        );
        assert!(i5.contains(&3), "tinyllama peaked near three there: {i5:?}");
        for c in decode_candidates(16, 8) {
            assert!(c <= 8, "calibration must not reach past physical cores");
        }
        assert!(
            decode_candidates(1, 1).len() < 2,
            "one core leaves nothing to choose between, so no calibration runs"
        );
    }

    /// The KV cache grows every token, so decode slows down on its own. A fixed
    /// rotation would give the first candidate every short-KV slot — measuring
    /// position rather than width. Direction alternates to cancel that.
    #[test]
    fn the_rotation_alternates_so_a_growing_cache_cannot_pick_the_winner() {
        let c = Calibration::new(vec![8, 6, 4, 2]);
        let seen: Vec<usize> = (0..8).map(|_| c.next_index()).collect();
        assert_eq!(seen, vec![0, 1, 2, 3, 3, 2, 1, 0]);
        for i in 0..4 {
            let slots: Vec<usize> = seen
                .iter()
                .enumerate()
                .filter(|(_, v)| **v == i)
                .map(|(pos, _)| pos % 4)
                .collect();
            assert_ne!(
                slots[0], slots[1],
                "candidate {i} saw the same slot twice — the bias is not cancelled"
            );
        }
    }

    /// The i5 case, replayed: four widths within a few percent of each other,
    /// and one lucky sample on the narrower one. Deciding on the minimum picked
    /// that sample's width — differently on two consecutive runs of the same
    /// model — and cost real speed. The median ignores it.
    #[test]
    fn one_lucky_sample_cannot_decide_a_flat_machine() {
        let c = Calibration::new(vec![6, 4]);
        // 6 is genuinely a touch faster; 4 gets one outlier far below its norm.
        c.record(0, 44_000_000);
        c.record(1, 33_000_000); // the lucky one
        for _ in 1..SAMPLES_PER_CANDIDATE {
            c.record(0, 44_000_000);
            c.record(1, 47_000_000);
        }
        assert_eq!(
            c.chosen.load(Ordering::Relaxed),
            6,
            "a single lucky sample decided the width — this is the i5 regression"
        );
    }

    /// On a machine with nothing to find, calibration must stop early rather
    /// than spend the user's tokens confirming it. The i5's own figures: every
    /// width within a few ms of the six it was offered.
    #[test]
    fn a_machine_with_nothing_to_find_stops_looking() {
        let c = Calibration::new(vec![6, 4, 3, 1]);
        for _ in 0..EARLY_SETTLE_SAMPLES {
            c.record(0, 45_000_000);
            c.record(1, 42_000_000);
            c.record(2, 47_000_000);
            c.record(3, 83_000_000);
        }
        assert_eq!(
            c.chosen.load(Ordering::Relaxed),
            6,
            "settled on what was offered"
        );
        let st = c.state.lock().unwrap();
        assert!(
            st.counts.iter().all(|c| *c <= EARLY_SETTLE_SAMPLES),
            "kept sampling after there was nothing to find: {:?}",
            st.counts
        );
    }

    /// ...but a machine that DOES have something to find is not settled early:
    /// the Ryzen's 4 is already 2x ahead after two samples, so the comparison
    /// runs to completion and takes it.
    #[test]
    fn a_real_optimum_is_not_settled_away_early() {
        let c = Calibration::new(vec![8, 4]);
        for _ in 0..EARLY_SETTLE_SAMPLES {
            c.record(0, 129_000_000);
            c.record(1, 60_000_000);
        }
        assert_eq!(
            c.chosen.load(Ordering::Relaxed),
            0,
            "settled before confirming a 2x difference"
        );
        for _ in EARLY_SETTLE_SAMPLES..SAMPLES_PER_CANDIDATE {
            c.record(0, 129_000_000);
            c.record(1, 60_000_000);
        }
        assert_eq!(c.chosen.load(Ordering::Relaxed), 4);
    }

    /// The real i5 numbers, where a percentage margin is not enough: 3 threads
    /// posted the best median (36 ms) against 6 threads' 52 ms — a 31% gap that
    /// clears any sane threshold — but 3's own timings ranged up past what 6
    /// typically does, which is what "this machine is flat and noisy" looks
    /// like. It must keep the width its owner offered.
    #[test]
    fn a_gap_no_bigger_than_the_noise_does_not_move_the_width() {
        let c = Calibration::new(vec![6, 3]);
        for v in [52_000_000, 48_000_000, 52_000_000, 36_000_000, 52_000_000] {
            c.record(0, v);
        }
        // Best median, but its worst run is slower than 6's typical run.
        for v in [36_000_000, 30_000_000, 36_000_000, 60_000_000, 36_000_000] {
            c.record(1, v);
        }
        assert_eq!(
            c.chosen.load(Ordering::Relaxed),
            6,
            "moved on a gap the size of the machine's own noise — the i5 regression"
        );
    }

    /// And the sharp case still wins: where narrowing genuinely helps it helps
    /// by far more than any margin, so being strict costs nothing.
    #[test]
    fn a_machine_with_a_real_optimum_still_finds_it() {
        let c = Calibration::new(vec![8, 4]);
        // The Ryzen's actual figures, including their real run-to-run spread:
        // 4 reproduced at 59/60/62 ms against 8's 125-129.
        for v in [
            129_000_000,
            125_000_000,
            126_000_000,
            128_000_000,
            127_000_000,
        ] {
            c.record(0, v);
        }
        for v in [60_000_000, 59_000_000, 62_000_000, 59_000_000, 61_000_000] {
            c.record(1, v);
        }
        assert_eq!(
            c.chosen.load(Ordering::Relaxed),
            4,
            "a 69ms gap against a 3ms spread is exactly what this should catch"
        );
    }

    /// Calibration is paid for out of the user's own first tokens, so a
    /// candidate that lost its first cycle badly is not timed again.
    #[test]
    fn a_hopeless_candidate_is_dropped_after_one_look() {
        let c = Calibration::new(vec![8, 6, 4, 2]);
        // One cycle: 2 threads is more than 1.5x off the best and is hopeless.
        c.record(0, 90_000_000);
        c.record(1, 76_000_000);
        c.record(2, 60_000_000);
        c.record(3, 160_000_000);
        {
            let st = c.state.lock().unwrap();
            assert!(
                st.eliminated[3],
                "160ms against 60ms should not be re-timed"
            );
            assert!(!st.eliminated[2], "the leader must survive");
            assert!(
                !st.eliminated[1],
                "76ms is behind but not hopeless — keep it"
            );
            // 90 against 60 is EXACTLY the ratio, and the rule discards only
            // what is strictly past it. Pinned deliberately: the boundary is
            // where a loose threshold would start throwing away real
            // candidates, which is the failure mode that matters here.
            assert!(
                !st.eliminated[0],
                "a candidate exactly at the ratio must survive"
            );
        }
        // The rotation must now only offer the survivors.
        for _ in 0..8 {
            let idx = c.next_index();
            let st = c.state.lock().unwrap();
            assert!(
                !st.eliminated[idx],
                "the rotation offered an eliminated candidate"
            );
        }
        // Finishing the survivors is enough to settle, without the dead ones.
        for _ in 0..SAMPLES_PER_CANDIDATE {
            c.record(0, 90_000_000);
            c.record(1, 76_000_000);
            c.record(2, 60_000_000);
        }
        assert_eq!(
            c.chosen.load(Ordering::Relaxed),
            4,
            "settled on the fastest survivor without ever re-timing the hopeless one"
        );
        assert_eq!(
            c.state.lock().unwrap().counts[3],
            1,
            "the eliminated candidate was timed exactly once"
        );
    }

    /// A clear winner is taken; a photo finish leaves the owner's own setting
    /// alone rather than chasing noise.
    #[test]
    fn calibration_takes_a_real_gain_and_ignores_a_marginal_one() {
        let c = Calibration::new(vec![8, 4]);
        for _ in 0..SAMPLES_PER_CANDIDATE {
            c.record(0, 100_000_000);
            c.record(1, 50_000_000);
        }
        assert_eq!(
            c.chosen.load(Ordering::Relaxed),
            4,
            "twice as fast at half the width and it was not taken"
        );

        let tie = Calibration::new(vec![8, 4]);
        for _ in 0..SAMPLES_PER_CANDIDATE {
            tie.record(0, 100_000_000);
            tie.record(1, 91_000_000);
        }
        assert_eq!(
            tie.chosen.load(Ordering::Relaxed),
            8,
            "9% is inside the spread of a flat machine — keep what was offered"
        );
    }

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
