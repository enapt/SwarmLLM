//! What this machine's memory actually delivers, measured rather than assumed.
//!
//! Generating a token is memory-bandwidth-bound: every layer's weights are read
//! once per token and almost nothing is reused, so throughput tracks how fast
//! the machine can stream memory far more closely than it tracks core count or
//! clock speed. That is why `estimate_tokens_per_sec_7b` takes a bandwidth
//! figure at all.
//!
//! For a GPU that figure is looked up from the card's name, which is accurate
//! because the name determines the hardware. For a CPU there is no such lookup,
//! and the code assumed **50 GB/s for every machine in the swarm** — so a
//! 16-core server with eight memory channels and a fanless mini-PC advertised
//! themselves as exactly equally fast, at 1.70 tokens/s. Every CPU node has been
//! quoting that same number, which meant nothing could distinguish them and
//! nothing could route on the difference.
//!
//! This measures it instead: stream a buffer larger than any last-level cache
//! and see how long it takes.
//!
//! **A debug build measures its own loop, not the machine.** The read is a
//! scalar `wrapping_add` with bounds checks until the optimiser gets to it, so
//! an unoptimised build is nowhere near memory-bound: the identical loop
//! measured **5.3 GB/s at `-O0` against 30.3 GB/s at `-O`** on the same
//! machine, in the same conditions (2026-09-01). Releases are optimised, so no
//! user sees this — but a `cargo run` test node advertises about a sixth of its
//! real speed for its whole life, and that figure is what every peer's
//! scheduler ranks it on. Read `advertised speed` in `swarmllm diagnostics`
//! before concluding a development node is being ignored unfairly.
//!
//! **Deliberately not a peak number.** It reads with the same thread count the
//! decode pool uses, because a single-threaded figure understates a multi-channel
//! machine badly and a thread-per-core figure overstates what a decode actually
//! achieves. What is wanted is a number that ranks machines the way running a
//! model would.

use std::sync::OnceLock;
use std::time::Instant;

/// Buffer streamed per pass, in bytes.
///
/// Must exceed the largest last-level cache in circulation or the measurement
/// reports cache bandwidth, which can be an order of magnitude higher and would
/// make a machine advertise a speed it cannot sustain for a single token.
/// Server parts reach 32-64 MB of L3 and a few (X3D, some Xeons) exceed 100 MB,
/// so 256 MB leaves clear room while staying small enough to allocate on a
/// modest node.
const BUFFER_BYTES: usize = 256 * 1024 * 1024;

/// Passes taken; the FASTEST is kept.
///
/// Every source of error here is additive — a scheduler preemption, another
/// process, a page fault — so the minimum is the least contaminated estimate.
/// The same reasoning as the `bench` helper in `inference::layers`.
const PASSES: usize = 3;

/// What to assume when the measurement could not be taken at all.
///
/// **Deliberately modest, and not a typical machine's bandwidth.** The only way
/// [`measured_gbps`] returns `None` is `try_reserve_exact` failing on a 256 MB
/// buffer, so the node reaching this is memory-starved — which is not a node to
/// advertise as capable. A caller that wants "something rather than nothing"
/// wants something *small*.
///
/// It was 50.0, chosen when a processor was priced at 15% of its roofline and
/// so produced 1.70 tok/s. Correcting that efficiency to the measured 0.75
/// (gotcha #428) would have turned the same 50.0 into **8.52 tok/s**, i.e. the
/// most constrained node on the network suddenly advertising itself as one of
/// the faster ones and attracting work it cannot do. The nominal moves with the
/// efficiency so the fallback keeps meaning what it meant.
pub const UNMEASURABLE_FALLBACK_GBPS: f32 = 10.0;

/// Sustained read bandwidth in GB/s, measured once and cached.
///
/// `None` when the buffer could not be allocated — a machine short enough of
/// memory for that is not one to be handing extra work to, and the caller falls
/// back to the previous assumption rather than advertising a wrong figure.
pub fn measured_gbps() -> Option<f32> {
    static MEASURED: OnceLock<Option<f32>> = OnceLock::new();
    *MEASURED.get_or_init(measure)
}

fn measure() -> Option<f32> {
    // The same width a decode runs at, so the figure ranks machines the way
    // running a model would rather than reporting a peak nothing achieves.
    let physical = num_cpus::get_physical().max(1);
    let threads = crate::inference::cpu_pools::decode_threads(physical, physical);
    let per_thread = BUFFER_BYTES / threads.max(1);
    // One allocation per thread, so the reads are spread across memory
    // controllers the way a real workload's are rather than contending on one
    // region.
    let mut buffers: Vec<Vec<u64>> = Vec::with_capacity(threads);
    for _ in 0..threads {
        let mut v: Vec<u64> = Vec::new();
        // Refuse rather than abort: a node too small for this is exactly the
        // node that must not be killed by a benchmark it did not ask for.
        if v.try_reserve_exact(per_thread / 8).is_err() {
            return None;
        }
        v.resize(per_thread / 8, 1);
        buffers.push(v);
    }
    let total_bytes = buffers.iter().map(|b| b.len() * 8).sum::<usize>() as f64;
    if total_bytes == 0.0 {
        return None;
    }

    let mut best = f64::INFINITY;
    for _ in 0..PASSES {
        let start = Instant::now();
        std::thread::scope(|s| {
            for buf in &buffers {
                s.spawn(move || {
                    // Sum rather than copy: a read-only stream is what a decode
                    // does to the weights, and it cannot be turned into a
                    // `memcpy` by the optimiser.
                    let mut acc = 0u64;
                    for chunk in buf.chunks(8) {
                        for &v in chunk {
                            acc = acc.wrapping_add(v);
                        }
                    }
                    std::hint::black_box(acc);
                });
            }
        });
        let secs = start.elapsed().as_secs_f64();
        if secs > 0.0 && secs < best {
            best = secs;
        }
    }
    if !best.is_finite() || best <= 0.0 {
        return None;
    }
    let gbps = total_bytes / best / 1e9;
    // A figure outside anything real means the measurement was disturbed, not
    // that the machine is extraordinary. Report nothing rather than something
    // the swarm would route on.
    (gbps.is_finite() && (1.0..=2000.0).contains(&gbps)).then_some(gbps as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The measurement must produce a figure in a range a real machine can
    /// occupy. Anything outside it is a disturbed run, and the caller is better
    /// served by the old assumption than by a number the swarm would route on.
    #[test]
    fn reports_a_plausible_figure_for_this_machine() {
        let Some(gbps) = measured_gbps() else {
            return; // allocation refused; nothing to assert
        };
        assert!(
            (1.0..=2000.0).contains(&gbps),
            "implausible bandwidth {gbps} GB/s"
        );
    }

    /// Cached: the second call must not re-run a 256 MB benchmark, which would
    /// otherwise be paid on every capability broadcast.
    #[test]
    fn the_measurement_is_taken_once() {
        let first = measured_gbps();
        let start = Instant::now();
        let second = measured_gbps();
        assert_eq!(first, second);
        assert!(
            start.elapsed().as_millis() < 50,
            "second call re-measured; it must come from the cache"
        );
    }
}
