//! Per-stage wall-clock accounting for one transformer block.
//!
//! Exists because a 3-9x faster quantized matmul moved prompt processing only
//! 1.15-1.24x, which puts the matmul at roughly a quarter of the time and left
//! the rest unaccounted for. There is no `perf` on the CPU test box, so the
//! breakdown has to come from the code.
//!
//! Accumulation is unconditional — `Instant::now` is ~25 ns against stages that
//! run for milliseconds, so the cost is far below the noise floor of the thing
//! being measured. Only the *dump* is gated, on `SWARMLLM_PROFILE=1`, and it
//! prints once per forward pass and resets.
//!
//! Stages are wall-clock and NON-overlapping, so they sum to roughly the block
//! time. A stage that internally uses rayon (every matmul does) still measures
//! correctly: the caller blocks for the whole parallel region.
use std::sync::atomic::{AtomicU64, Ordering};

macro_rules! stages {
    ($($variant:ident => $label:literal),+ $(,)?) => {
        #[derive(Copy, Clone)]
        pub(crate) enum Stage { $($variant),+ }
        pub(crate) const LABELS: &[&str] = &[$($label),+];
    };
}

stages! {
    QkvProj    => "qkv projections      (quantized matmul)",
    AttnShape  => "rope / transpose / q-k norm",
    KvCache    => "kv cache append",
    AttnCore   => "attention scores + softmax + AV",
    AttnOut    => "output projection    (quantized matmul)",
    FfnUpGate  => "ffn up + gate        (quantized matmul)",
    FfnAct     => "activation * gate    (elementwise)",
    FfnDown    => "ffn down             (quantized matmul)",
    FfnProbe   => "ffn width probe      (per-call, m=1)",
    Norms      => "rms norms",
    Residual   => "residual adds",
}

pub(crate) const N: usize = LABELS.len();

#[allow(clippy::declare_interior_mutable_const)]
const ZERO: AtomicU64 = AtomicU64::new(0);
static ACC: [AtomicU64; N] = [ZERO; N];

/// Adds an elapsed span to a stage. Prefer [`timed`] over calling this directly.
#[inline]
pub(crate) fn add(stage: Stage, nanos: u64) {
    ACC[stage as usize].fetch_add(nanos, Ordering::Relaxed);
}

/// Times an expression into a stage and returns its value.
macro_rules! timed {
    ($stage:expr, $e:expr) => {{
        let __t = std::time::Instant::now();
        let __r = $e;
        $crate::inference::prof::add($stage, __t.elapsed().as_nanos() as u64);
        __r
    }};
}
pub(crate) use timed;

pub(crate) fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("SWARMLLM_PROFILE").as_deref() == Ok("1"))
}

/// Prints the breakdown and zeroes the counters. `total_ms` is the measured
/// wall time of the whole forward, so the report can show what the stages do
/// NOT account for — the gap is as informative as the stages themselves.
pub(crate) fn dump_and_reset(tag: &str, total_ms: f64) {
    let vals: Vec<u64> = ACC.iter().map(|a| a.swap(0, Ordering::Relaxed)).collect();
    let summed: u64 = vals.iter().sum();
    let total_ns = total_ms * 1e6;
    eprintln!("PROF {tag} — total {total_ms:.0} ms");
    let mut order: Vec<usize> = (0..N).collect();
    order.sort_by_key(|&i| std::cmp::Reverse(vals[i]));
    for i in order {
        if vals[i] == 0 {
            continue;
        }
        eprintln!(
            "  {:>7.1} ms  {:>5.1}%  {}",
            vals[i] as f64 / 1e6,
            vals[i] as f64 / total_ns * 100.0,
            LABELS[i]
        );
    }
    let unattributed = total_ns - summed as f64;
    eprintln!(
        "  {:>7.1} ms  {:>5.1}%  unattributed (allocation, copies, dispatch)",
        unattributed / 1e6,
        unattributed / total_ns * 100.0
    );
}
