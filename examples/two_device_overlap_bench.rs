//! FUTURE_WORK #462's first check: do two candle devices on ONE card, driven from two threads of
//! one process, run their work at the same time — unlike two processes, which this card takes in
//! turns (0.89 of the work one at a time, 2026-11-15)?
//!
//! A = a flowing split's near half on this computer: 26 layers of Qwen2.5-14B Q4_K_M's weight
//! products at 24 rows (the 8 x 3 tree). B = the guessing model's call: Qwen2.5-0.5B Q8_0's
//! products at 8 rows — 24 layers and the output head — three levels. Only the weight products
//! (the bulk of both steps' card time; attention, norms and the host walk between levels are not
//! here). Each side runs eager (the processor launching every product) and as ONE CUDA graph;
//! alone, together on two devices (two streams, the design candidate), and — the control —
//! together on ONE device (one stream), where only the processor's launching can hide.
//!
//! Prior art: PEARL (arXiv 2408.11850) and PipeInfer run drafting beside verification, each on
//! its own GPU; on one GPU the means is streams of one context (CUDA MPS for several processes).
//! The trap a candle version can hit and a raw-CUDA probe cannot: the stream-ordered allocator's
//! default pool may make one stream wait on another when it reuses memory the other freed
//! (`cudaMemPoolReuseAllowInternalDependencies`, on by default) — candle allocates every
//! product's output, so the eager arm would show it as lost overlap.
//!
//! ```bash
//! CARGO_TARGET_DIR=~/swarm-target-cuda CMAKE_CUDA_ARCHITECTURES=86-real \
//!   cargo build --release --features cuda --example two_device_overlap_bench -j 2
//! ```
//! Needs ~1.5 GB of card memory; run it with the node stopped.

use std::sync::{Arc, Barrier, Mutex};
use std::time::Instant;

use candle_core::quantized::{GgmlDType, QMatMul, QTensor};
use candle_core::{Device, Module, Tensor};
use cudarc::driver::{result, sys, CudaStream};

fn qmm(rows: usize, cols: usize, dtype: GgmlDType, dev: &Device) -> anyhow::Result<QMatMul> {
    let w = Tensor::randn(0f32, 0.02, (rows, cols), &Device::Cpu)?;
    Ok(QMatMul::from_qtensor(QTensor::quantize_onto(
        &w, dtype, dev,
    )?)?)
}

/// One side's products in the order one call runs them: (matrix, input).
struct Work {
    dev: Device,
    mats: Vec<QMatMul>,
    inputs: Vec<Tensor>,
    order: Vec<(usize, usize)>,
}

impl Work {
    fn run(&self) -> anyhow::Result<()> {
        for &(m, x) in &self.order {
            drop(self.mats[m].forward(&self.inputs[x])?);
        }
        Ok(())
    }
    fn call(&self) -> anyhow::Result<()> {
        self.run()?;
        self.dev.synchronize()?;
        Ok(())
    }
}

/// Qwen2.5-14B Q4_K_M's 26 near layers at 24 rows. The large matrices exceed the card's 4 MB L2,
/// so one copy each stands for every layer; K and V (2.9 and 4.3 MB) get their own per layer.
fn near(dev: &Device) -> anyhow::Result<Work> {
    use GgmlDType::{Q4K, Q6K};
    let (h, f, kv, rows, layers) = (5120, 13824, 1024, 24, 26);
    let mut mats = vec![
        qmm(h, h, Q4K, dev)?, // q
        qmm(h, h, Q4K, dev)?, // o
        qmm(f, h, Q4K, dev)?, // gate
        qmm(f, h, Q4K, dev)?, // up
        qmm(h, f, Q6K, dev)?, // down
    ];
    let mut order = Vec::new();
    for _ in 0..layers {
        let k = mats.len();
        mats.push(qmm(kv, h, Q4K, dev)?);
        mats.push(qmm(kv, h, Q6K, dev)?);
        order.extend([(0, 0), (k, 0), (k + 1, 0), (1, 0), (2, 0), (3, 0), (4, 1)]);
    }
    let inputs = vec![
        Tensor::randn(0f32, 1.0, (rows, h), dev)?,
        Tensor::randn(0f32, 1.0, (rows, f), dev)?,
    ];
    Ok(Work {
        dev: dev.clone(),
        mats,
        inputs,
        order,
    })
}

/// Qwen2.5-0.5B Q8_0 at 8 rows: 24 layers (every matrix its own — they fit L2) and the head,
/// three levels a call.
fn drafter(dev: &Device) -> anyhow::Result<Work> {
    let q8 = GgmlDType::Q8_0;
    let (h, f, kv, vocab, rows, layers, levels) = (896, 4864, 128, 151936, 8, 24, 3);
    let mut mats = vec![qmm(vocab, h, q8, dev)?];
    let mut one_level = Vec::new();
    for _ in 0..layers {
        let b = mats.len();
        for (r, c) in [(h, h), (kv, h), (kv, h), (h, h), (f, h), (f, h), (h, f)] {
            mats.push(qmm(r, c, q8, dev)?);
        }
        one_level.extend([
            (b, 0),
            (b + 1, 0),
            (b + 2, 0),
            (b + 3, 0),
            (b + 4, 0),
            (b + 5, 0),
        ]);
        one_level.push((b + 6, 1));
    }
    one_level.push((0, 0));
    let order = (0..levels)
        .flat_map(|_| one_level.iter().copied())
        .collect();
    let inputs = vec![
        Tensor::randn(0f32, 1.0, (rows, h), dev)?,
        Tensor::randn(0f32, 1.0, (rows, f), dev)?,
    ];
    Ok(Work {
        dev: dev.clone(),
        mats,
        inputs,
        order,
    })
}

/// One side captured as a single CUDA graph on its device's own stream (its outputs are freed
/// inside the capture, so the graph replays without anything left allocated).
struct Graph {
    exec: sys::CUgraphExec,
    stream: Arc<CudaStream>,
    dev: Device,
}

// SAFETY: a graph exec may be launched from any thread; this probe launches each one from ONE
// thread at a time (the side that owns it in an arm) and destroys it after every launch has ended.
unsafe impl Send for Graph {}
unsafe impl Sync for Graph {}

impl Graph {
    fn capture(work: &Work) -> anyhow::Result<Self> {
        let stream = work.dev.as_cuda_device()?.cuda_stream();
        stream.context().bind_to_thread()?;
        // SAFETY: a live stream cudarc created for this device; this thread captures it.
        unsafe {
            result::stream::begin_capture(
                stream.cu_stream(),
                sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
            )?
        };
        let ran = work.run();
        // SAFETY: the stream this thread began capturing.
        let graph = unsafe { result::stream::end_capture(stream.cu_stream()) };
        ran?;
        let graph = graph?;
        anyhow::ensure!(!graph.is_null(), "the capture was invalidated");
        // SAFETY: a graph the capture just returned.
        let exec = unsafe { result::graph::instantiate(graph, sys::CUgraphInstantiate_flags(0))? };
        unsafe { result::graph::destroy(graph)? };
        Ok(Self {
            exec,
            stream,
            dev: work.dev.clone(),
        })
    }
    fn launch(&self) -> anyhow::Result<()> {
        self.stream.context().bind_to_thread()?;
        // SAFETY: an instantiated graph and the stream it was captured on.
        unsafe { result::graph::launch(self.exec, self.stream.cu_stream())? };
        Ok(())
    }
    /// Launch into another stream of the same context and wait for it: a graph launch takes the
    /// priority of the stream it is launched into.
    fn call_on(&self, s: &Prio) -> anyhow::Result<()> {
        self.stream.context().bind_to_thread()?;
        // SAFETY: an instantiated graph; a live stream of its context.
        unsafe {
            result::graph::launch(self.exec, s.0)?;
            result::stream::synchronize(s.0)?;
        }
        Ok(())
    }
    fn call(&self) -> anyhow::Result<()> {
        self.launch()?;
        self.dev.synchronize()?;
        Ok(())
    }
}

/// A stream of the card's HIGHEST priority, in the context both devices share: as blocks of the
/// near layers' kernels finish, the waiting blocks of this stream's kernels are scheduled first
/// (a running block is never pre-empted). Does that fill the near kernels' tails more often?
struct Prio(sys::CUstream);

// SAFETY: a stream handle is usable from any thread with its context current (`call_on` binds).
unsafe impl Send for Prio {}
unsafe impl Sync for Prio {}

fn top_priority_stream(dev: &Device) -> anyhow::Result<(Prio, i32, i32)> {
    dev.as_cuda_device()?
        .cuda_stream()
        .context()
        .bind_to_thread()?;
    let (mut least, mut greatest) = (0, 0);
    let mut s = std::ptr::null_mut();
    // SAFETY: out-pointers to locals; the context is current on this thread.
    unsafe {
        sys::cuCtxGetStreamPriorityRange(&mut least, &mut greatest).result()?;
        sys::cuStreamCreateWithPriority(&mut s, 1 /* CU_STREAM_NON_BLOCKING */, greatest)
            .result()?;
    }
    Ok((Prio(s), least, greatest))
}

fn ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1e3
}

/// `a` on this thread and `b` on another, both released together; the wall time until both end.
fn on_two_threads(
    a: &(dyn Fn() -> anyhow::Result<()> + Sync),
    b: &(dyn Fn() -> anyhow::Result<()> + Sync),
    reps: usize,
) -> anyhow::Result<Vec<f64>> {
    let (start, done) = (Barrier::new(2), Barrier::new(2));
    let b_ends: Mutex<Vec<Instant>> = Mutex::new(Vec::new());
    std::thread::scope(|s| {
        let other = s.spawn(|| {
            for _ in 0..reps {
                start.wait();
                b().expect("side B");
                b_ends.lock().unwrap().push(Instant::now());
                done.wait();
            }
        });
        let mut walls = Vec::new();
        for r in 0..reps {
            start.wait();
            let t0 = Instant::now();
            a().expect("side A");
            let a_end = Instant::now();
            done.wait();
            let b_end = b_ends.lock().unwrap()[r];
            walls.push(a_end.max(b_end).duration_since(t0).as_secs_f64() * 1e3);
        }
        other.join().expect("thread B");
        Ok(walls)
    })
}

fn median(v: &[f64]) -> f64 {
    let mut v = v.to_vec();
    v.sort_by(|a, b| a.total_cmp(b));
    v[v.len() / 2]
}

fn least(v: &[f64]) -> f64 {
    v.iter().copied().fold(f64::INFINITY, f64::min)
}

fn main() -> anyhow::Result<()> {
    let dev_a = Device::new_cuda(0)?;
    let dev_b = Device::new_cuda(0)?;
    eprintln!("building the weights (a minute) …");
    let a = near(&dev_a)?;
    let b = drafter(&dev_b)?;
    let b_same = drafter(&dev_a)?; // the control: B on A's device, so on A's one stream
    for _ in 0..3 {
        a.call()?;
        b.call()?;
        b_same.call()?;
    }
    // In the product a drafter's level (8 rows) is captured and the near half's 24-row forward
    // is not (`cuda_graph::MAX_POSITIONS`), so "eager A beside graph B" is the product's pairing.
    let (gb, gb_same) = (Graph::capture(&b)?, Graph::capture(&b_same)?);
    let ga = match Graph::capture(&a) {
        Ok(g) => Some(g),
        Err(e) => {
            println!("graph A not captured ({e:#}); its arms are skipped");
            let _ = dev_a.as_cuda_device()?.cuda_stream().context().check_err();
            None
        }
    };
    let (prio, least_p, greatest_p) = top_priority_stream(&dev_b)?;
    println!("stream priorities: least {least_p}, greatest {greatest_p} (B's arms below at {greatest_p})");
    for _ in 0..3 {
        gb.call_on(&prio)?;
        gb.call()?;
        gb_same.call()?;
        if let Some(g) = &ga {
            g.call()?;
        }
    }

    let (rounds, reps) = (6, 8);
    let names = [
        "eager A alone",
        "eager B alone (device 2)",
        "eager B alone (device 1)",
        "eager A + eager B, two devices",
        "eager A + eager B, ONE device",
        "graph B alone (device 2)",
        "graph B alone (device 1)",
        "eager A + graph B, two devices",
        "eager A + graph B, ONE device",
        "graph A alone",
        "graph A + graph B, two devices",
        "graph A + graph B, ONE device",
        "graph B alone, top-priority stream",
        "eager A + graph B, top priority",
        "graph A + graph B, top priority",
    ];
    let mut t: Vec<Vec<f64>> = vec![Vec::new(); names.len()];
    let alone = |f: &dyn Fn() -> anyhow::Result<()>, out: &mut Vec<f64>| -> anyhow::Result<()> {
        for _ in 0..reps {
            let t0 = Instant::now();
            f()?;
            out.push(ms(t0));
        }
        Ok(())
    };
    let graphs_together = |x: &Graph, y: &Graph, out: &mut Vec<f64>| -> anyhow::Result<()> {
        for _ in 0..reps {
            let t0 = Instant::now();
            x.launch()?;
            y.launch()?;
            x.dev.synchronize()?;
            y.dev.synchronize()?;
            out.push(ms(t0));
        }
        Ok(())
    };
    for round in 0..rounds {
        eprintln!("round {} of {rounds}", round + 1);
        alone(&|| a.call(), &mut t[0])?;
        alone(&|| b.call(), &mut t[1])?;
        alone(&|| b_same.call(), &mut t[2])?;
        t[3].extend(on_two_threads(&|| a.call(), &|| b.call(), reps)?);
        t[4].extend(on_two_threads(&|| a.call(), &|| b_same.call(), reps)?);
        alone(&|| gb.call(), &mut t[5])?;
        alone(&|| gb_same.call(), &mut t[6])?;
        t[7].extend(on_two_threads(&|| a.call(), &|| gb.call(), reps)?);
        t[8].extend(on_two_threads(&|| a.call(), &|| gb_same.call(), reps)?);
        alone(&|| gb.call_on(&prio), &mut t[12])?;
        t[13].extend(on_two_threads(&|| a.call(), &|| gb.call_on(&prio), reps)?);
        if let Some(ga) = &ga {
            alone(&|| ga.call(), &mut t[9])?;
            graphs_together(ga, &gb, &mut t[10])?;
            graphs_together(ga, &gb_same, &mut t[11])?;
            t[14].extend(on_two_threads(&|| ga.call(), &|| gb.call_on(&prio), reps)?);
        }
    }
    println!("{rounds} rounds x {reps} repeats; median (least) ms");
    for (n, v) in names.iter().zip(&t) {
        if !v.is_empty() {
            println!("  {n:34} {:7.2} ({:6.2})", median(v), least(v));
        }
    }
    for (label, a_i, b_i, ab_i) in [
        ("eager + eager, two devices", 0, 1, 3),
        ("eager + eager, one device", 0, 2, 4),
        ("eager + graph, two devices", 0, 5, 7),
        ("eager + graph, one device", 0, 6, 8),
        ("graph + graph, two devices", 9, 5, 10),
        ("graph + graph, one device", 9, 6, 11),
        ("eager + graph, B top priority", 0, 12, 13),
        ("graph + graph, B top priority", 9, 12, 14),
    ] {
        if t[ab_i].is_empty() {
            continue;
        }
        let (a_ms, b_ms, ab_ms) = (median(&t[a_i]), median(&t[b_i]), median(&t[ab_i]));
        println!(
            "{label:28} in series {:6.2}, together {:6.2}: {:4.0}% of B hidden under A",
            a_ms + b_ms,
            ab_ms,
            100.0 * (a_ms + b_ms - ab_ms) / b_ms
        );
    }
    // SAFETY: instantiated above, no launch outstanding (every call synchronized).
    unsafe {
        result::stream::destroy(prio.0)?;
        for g in [Some(&gb), Some(&gb_same), ga.as_ref()]
            .into_iter()
            .flatten()
        {
            result::graph::exec_destroy(g.exec)?;
        }
    }
    Ok(())
}
