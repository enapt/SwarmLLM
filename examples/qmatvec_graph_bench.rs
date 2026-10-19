//! Do our decode matrix-vector products lose to llama.cpp's at a SMALL model's
//! shapes? (FUTURE_WORK #309: TinyLlama decodes at 0.80 of llama.cpp on the
//! card, every 3B-and-larger family at 0.84-0.97, and the step is on the card,
//! not the host.)
//!
//! Each product — the activation quantized to q8_1, then the quantized
//! matrix-vector kernel, as `QMatMul::forward` runs them — is repeated `REPS`
//! times inside ONE captured CUDA graph, the graph is replayed, and the best
//! replay's wall time is divided by `REPS`. That is how llama.cpp's
//! `test-backend-ops perf -o MUL_MAT` measures (one op node repeated in one
//! graph, the same weights every time, wall time over runs), so the two "us a
//! product" figures compare like for like. Launch cost is excluded on both
//! sides; a matrix that fits the card's L2 (4 MB on a 3070 Laptop) is read from
//! it on both sides too.
//!
//! The second table cycles 22 distinct matrices per shape (TinyLlama's 22
//! layers), so nothing is reused from L2 — closer to a real step.
//!
//! A microbenchmark explains where a step's time goes; it never decides
//! between two kernels — that is measured by generating tokens (gotcha #266:
//! timed in isolation, an attention kernel mispredicted the choice three times).
//!
//! ```bash
//! CARGO_TARGET_DIR=~/swarm-target-cuda CMAKE_CUDA_ARCHITECTURES=86-real \
//!   cargo build --release --features cuda --example qmatvec_graph_bench -j 2
//! ~/swarm-target-cuda/release/examples/qmatvec_graph_bench
//! ```
//! Needs ~1 GB of card memory; run it with the node stopped.

use candle_core::cuda_backend::cudarc::driver::{result, sys};
use candle_core::quantized::{GgmlDType, QMatMul, QTensor};
use candle_core::{Device, Module, Tensor};

/// Products recorded into one graph.
const REPS: usize = 256;
/// Timed replays of each graph; the best is kept.
const REPLAYS: usize = 7;

/// One instantiated graph, destroyed on drop.
struct Exec(sys::CUgraphExec);

impl Drop for Exec {
    fn drop(&mut self) {
        // SAFETY: made by `cuGraphInstantiateWithFlags`, destroyed only here.
        let _ = unsafe { result::graph::exec_destroy(self.0) };
    }
}

/// Record `body` into a graph on `dev`'s stream and instantiate it.
fn capture(
    dev: &candle_core::CudaDevice,
    body: impl FnOnce() -> anyhow::Result<()>,
) -> anyhow::Result<Exec> {
    let stream = dev.cuda_stream();
    stream.context().bind_to_thread()?;
    let copies = candle_core::cuda::htod_copies_so_far();
    // SAFETY: a live stream cudarc created for this device.
    unsafe {
        result::stream::begin_capture(
            stream.cu_stream(),
            sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
        )?
    };
    let ran = body();
    // SAFETY: the stream this thread began capturing.
    let graph = unsafe { result::stream::end_capture(stream.cu_stream()) }?;
    ran?;
    anyhow::ensure!(!graph.is_null(), "the capture was invalidated");
    anyhow::ensure!(
        candle_core::cuda::htod_copies_so_far() == copies,
        "a host-to-device copy inside the capture would replay from a stale address"
    );
    let mut exec: sys::CUgraphExec = std::ptr::null_mut();
    // SAFETY: `graph` is a complete captured graph; flags 0.
    unsafe { sys::cuGraphInstantiateWithFlags(&mut exec, graph, 0) }.result()?;
    // SAFETY: instantiated above; the exec does not need the graph.
    unsafe { result::graph::destroy(graph) }?;
    Ok(Exec(exec))
}

/// Best wall time of one replay of `exec`, in seconds.
fn best_replay(dev: &Device, cuda: &candle_core::CudaDevice, exec: &Exec) -> anyhow::Result<f64> {
    let stream = cuda.cuda_stream();
    // SAFETY: a live exec, launched on the stream it was captured from.
    unsafe { result::graph::launch(exec.0, stream.cu_stream()) }?;
    dev.synchronize()?;
    let mut best = f64::INFINITY;
    for _ in 0..REPLAYS {
        let t = std::time::Instant::now();
        // SAFETY: as above.
        unsafe { result::graph::launch(exec.0, stream.cu_stream()) }?;
        dev.synchronize()?;
        best = best.min(t.elapsed().as_secs_f64());
    }
    Ok(best)
}

fn qmm(rows: usize, cols: usize, dtype: GgmlDType, dev: &Device) -> anyhow::Result<QMatMul> {
    let w = Tensor::randn(0f32, 0.02, (rows, cols), &Device::Cpu)?;
    Ok(QMatMul::from_qtensor(QTensor::quantize_onto(
        &w, dtype, dev,
    )?)?)
}

fn main() -> anyhow::Result<()> {
    let dev = Device::new_cuda(0)?;
    let Device::Cuda(cuda) = &dev else {
        anyhow::bail!("not a CUDA device")
    };
    // TinyLlama 1.1B Q4_K_M: hidden 2048, 4 KV heads of 64, FFN 5632, vocab
    // 32000; llama.cpp's MUL_MAT case is (type, m = rows, n = 1, k = cols).
    let (hidden, ffn, kv, vocab, layers) = (2048usize, 5632usize, 256usize, 32000usize, 22usize);
    let shapes = [
        ("q/o   2048x2048 Q4_K", hidden, hidden, GgmlDType::Q4K),
        ("k      256x2048 Q4_K", kv, hidden, GgmlDType::Q4K),
        ("v      256x2048 Q6_K", kv, hidden, GgmlDType::Q6K),
        ("gate/up 5632x2048 Q4_K", ffn, hidden, GgmlDType::Q4K),
        ("down   2048x5632 Q6_K", hidden, ffn, GgmlDType::Q6K),
        ("down   2048x5632 Q4_K", hidden, ffn, GgmlDType::Q4K),
        ("head  32000x2048 Q6_K", vocab, hidden, GgmlDType::Q6K),
    ];
    println!("one matrix, {REPS} products in one graph, best of {REPLAYS} replays:");
    for &(name, rows, cols, dt) in &shapes {
        let m = qmm(rows, cols, dt, &dev)?;
        let x = Tensor::randn(0f32, 1.0, (1, cols), &dev)?;
        for _ in 0..3 {
            m.forward(&x)?;
        }
        dev.synchronize()?;
        // Each product's output and q8_1 scratch are allocated and freed
        // inside the graph, as in a captured decode step.
        let exec = capture(cuda, || {
            for _ in 0..REPS {
                drop(m.forward(&x)?);
            }
            Ok(())
        })?;
        let per = best_replay(&dev, cuda, &exec)? / REPS as f64;
        let bytes = rows * cols / dt.block_size() * dt.type_size();
        println!(
            "  {name:24} {:7.2} us   {:6.1} GB/s   ({:.2} MB)",
            per * 1e6,
            bytes as f64 / per / 1e9,
            bytes as f64 / 1e6
        );
    }

    println!(
        "{layers} distinct matrices cycled (nothing reused from L2), {REPS} products in one graph:"
    );
    for &(name, rows, cols, dt) in &shapes[..6] {
        let ms: Vec<QMatMul> = (0..layers)
            .map(|_| qmm(rows, cols, dt, &dev))
            .collect::<anyhow::Result<_>>()?;
        let x = Tensor::randn(0f32, 1.0, (1, cols), &dev)?;
        for m in &ms {
            m.forward(&x)?;
        }
        dev.synchronize()?;
        let exec = capture(cuda, || {
            for i in 0..REPS {
                drop(ms[i % layers].forward(&x)?);
            }
            Ok(())
        })?;
        let per = best_replay(&dev, cuda, &exec)? / REPS as f64;
        let bytes = rows * cols / dt.block_size() * dt.type_size();
        println!(
            "  {name:24} {:7.2} us   {:6.1} GB/s",
            per * 1e6,
            bytes as f64 / per / 1e9
        );
    }
    Ok(())
}
