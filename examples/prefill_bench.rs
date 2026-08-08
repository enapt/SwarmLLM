//! Times prompt processing and decode on a REAL model loaded from its shard
//! directory, with no daemon, no network and no scheduler in the way.
//!
//! Attention changes are easy to measure wrongly: a microbenchmark of one op
//! says nothing about a forward pass that is dominated by allocation and
//! memory traffic, and a full daemon run buries the signal under chunking
//! policy, batching and the API. This sits in between — the same
//! `SplitModel::forward` production uses, driven directly.
//!
//! ```bash
//! SWARM_BENCH_MODEL=~/.local/share/swarmllm/models/llama-3.2-3b-instruct-q4-k-m \
//!   RAYON_NUM_THREADS=4 \
//!   cargo run --release --no-default-features --features dev --example prefill_bench
//! ```
//!
//! Set `SWARMLLM_PROFILE=1` for the per-stage breakdown, `SWARM_BENCH_PROMPT`
//! for the prompt length in tokens (default 896) and `SWARM_BENCH_DECODE` for
//! how many tokens to generate afterwards (default 32).
//!
//! **Min of N, not mean** — see the note on `bench` in
//! `src/inference/layers/mod.rs`. Every source of error here is additive, so
//! the minimum is the least contaminated estimate.

use std::path::{Path, PathBuf};
use std::time::Instant;

use candle_core::{Device, Tensor};
use swarmllm::inference::split::{KvCacheStore, SplitModel};
use swarmllm_types::{ModelManifest, ShardTensorEntry};

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// Load every layer of a model from its shard directory, exactly as a node
/// holding the whole model does.
fn load(model_dir: &Path) -> anyhow::Result<SplitModel> {
    swarmllm::inference::split::ensure_gguf_header(model_dir)
        .map_err(|e| anyhow::anyhow!("gguf header: {e}"))?;
    let manifest: ModelManifest =
        serde_json::from_slice(&std::fs::read(model_dir.join("manifest.json"))?)?;

    let mut shard_files: Vec<(u32, PathBuf)> = Vec::new();
    for shard in &manifest.shards {
        let path = model_dir.join(format!("shard_{:03}.bin", shard.index));
        if path.exists() {
            shard_files.push((shard.index, path));
        }
    }
    shard_files.sort_by_key(|(i, _)| *i);
    anyhow::ensure!(
        shard_files.len() == manifest.shards.len(),
        "need every shard to run the whole model: have {} of {}",
        shard_files.len(),
        manifest.shards.len()
    );

    let tensor_entries: Vec<Vec<ShardTensorEntry>> = shard_files
        .iter()
        .map(|(idx, _)| {
            manifest
                .shards
                .iter()
                .find(|s| s.index == *idx)
                .map(|s| s.tensors.clone())
                .unwrap_or_default()
        })
        .collect();

    // `SWARM_BENCH_DEVICE=cuda` picks the auto-detecting loader, which is what
    // a GPU node uses; anything else pins the CPU one. Explicit rather than
    // auto-detected: a benchmark that silently changes device between runs is
    // worse than one that refuses.
    let want_gpu = std::env::var("SWARM_BENCH_DEVICE").as_deref() == Ok("cuda");
    let load = if want_gpu {
        SplitModel::load_from_shards
    } else {
        SplitModel::load_from_shards_cpu
    };
    load(
        model_dir,
        shard_files,
        &tensor_entries,
        manifest.total_size_bytes,
        0,
        manifest.num_layers as usize,
        true,
        true,
    )
    .map_err(|e| anyhow::anyhow!("load: {e}"))
}

fn main() -> anyhow::Result<()> {
    let dir = std::env::var("SWARM_BENCH_MODEL").map_err(|_| {
        anyhow::anyhow!("set SWARM_BENCH_MODEL to a model directory holding every shard")
    })?;
    let model_dir = PathBuf::from(shellexpand(&dir));
    let prompt_tokens = env_usize("SWARM_BENCH_PROMPT", 896);
    let decode_tokens = env_usize("SWARM_BENCH_DECODE", 32);
    let reps = env_usize("SWARM_BENCH_REPS", 3);

    println!("loading {}", model_dir.display());
    let t = Instant::now();
    let mut model = load(&model_dir)?;
    // Report the device the model ACTUALLY loaded onto, never the one that was
    // asked for. A binary built without the CUDA features answers a `cuda`
    // request by silently loading on the CPU, and a benchmark that echoes the
    // request back reads as a GPU result while measuring a processor — which
    // is exactly what happened here once, producing numbers 50x off before the
    // mismatch with an earlier run gave it away.
    let on_gpu = model.device().is_cuda();
    println!(
        "device={} (requested {})",
        if on_gpu { "CUDA" } else { "CPU" },
        std::env::var("SWARM_BENCH_DEVICE").unwrap_or_else(|_| "cpu".into())
    );
    if std::env::var("SWARM_BENCH_DEVICE").as_deref() == Ok("cuda") && !on_gpu {
        anyhow::bail!(
            "asked for CUDA and got CPU — this binary has no GPU support compiled in \
             (build with `--features flash-attn`) or no device was available. Refusing to \
             report processor timings as GPU ones."
        );
    }
    println!(
        "loaded {} layers in {:.1}s, threads={}\n",
        model.total_layers,
        t.elapsed().as_secs_f64(),
        rayon::current_num_threads()
    );

    // Token ids are arbitrary — the arithmetic per position does not depend on
    // which token it is, only on how many there are. Keep them well inside any
    // vocabulary and away from 0 so no path treats them as padding.
    let ids: Vec<i64> = (0..prompt_tokens)
        .map(|i| (i % 20_000 + 100) as i64)
        .collect();
    let input = Tensor::from_vec(ids.clone(), &[1, prompt_tokens], &Device::Cpu)?;

    let device = model.device().clone();
    let mut best_prefill = f64::INFINITY;
    let mut best_decode = f64::INFINITY;
    for rep in 0..reps {
        // A fresh store per rep: reusing one would let the prefix cache serve
        // the second run and measure a lookup instead of a prefill.
        let store = KvCacheStore::new(std::time::Duration::from_secs(600));
        let req = format!("bench-{rep}");

        let t = Instant::now();
        let logits = model
            .forward(&input, 0, &store, &req)
            .map_err(|e| anyhow::anyhow!("prefill: {e}"))?;
        // CUDA work is enqueued, not executed, by the time `forward` returns.
        // Without this the timer measures how long it takes to SUBMIT the
        // forward pass — which on a first attempt reported 3977 tok/s of
        // prompt processing, about 23 TFLOPS on a laptop 3070, i.e. obvious
        // nonsense that would have gone straight into a README. CPU timings
        // never needed it because CPU ops are synchronous.
        sync(&device);
        let prefill = t.elapsed().as_secs_f64();
        best_prefill = best_prefill.min(prefill);
        drop(logits);

        let t = Instant::now();
        for pos in (prompt_tokens..).take(decode_tokens) {
            let step = Tensor::from_vec(vec![7i64], &[1, 1], &Device::Cpu)?;
            let logits = model
                .forward(&step, pos, &store, &req)
                .map_err(|e| anyhow::anyhow!("decode: {e}"))?;
            drop(logits);
        }
        sync(&device);
        let decode = t.elapsed().as_secs_f64() / decode_tokens as f64;
        best_decode = best_decode.min(decode);

        let occ = store.occupancy();
        println!(
            "  rep {rep}: prefill {:.2}s ({:.1} tok/s)   decode {:.1} ms/token",
            prefill,
            prompt_tokens as f64 / prefill,
            decode * 1000.0
        );
        // ONE request's KV cache. The interesting number is how far
        // `allocated` sits above `used`: candle reserves the whole context
        // window on the first append, so a short conversation costs the same
        // as a full-length one.
        println!(
            "          KV cache: {} entries, {:.0} MB allocated, {:.0} MB used ({:.0}% utilisation), {} positions",
            occ.entries,
            occ.allocated_bytes as f64 / 1e6,
            occ.used_bytes as f64 / 1e6,
            occ.utilisation() * 100.0,
            occ.tokens,
        );
    }

    println!(
        "\nBEST prompt processing  {:>8.2} tok/s   ({prompt_tokens} tokens in {:.2}s)",
        prompt_tokens as f64 / best_prefill,
        best_prefill
    );
    println!(
        "BEST decode             {:>8.2} tok/s   ({:.1} ms/token at ~{} KV)",
        1.0 / best_decode,
        best_decode * 1000.0,
        prompt_tokens + decode_tokens / 2
    );
    Ok(())
}

/// Block until the device has finished the work queued on it.
///
/// A no-op on CPU, where every op has already run. On CUDA it is the
/// difference between timing the work and timing the submission of the work.
fn sync(device: &Device) {
    if let Err(e) = device.synchronize() {
        eprintln!("WARNING: device synchronize failed ({e}) — timings are not trustworthy");
    }
}

/// Expand a leading `~` so the documented invocation works from a shell that
/// did not expand it (e.g. when the value came from a file or a CI variable).
fn shellexpand(p: &str) -> String {
    match p.strip_prefix("~/") {
        Some(rest) => match std::env::var("HOME") {
            Ok(home) => format!("{home}/{rest}"),
            Err(_) => p.to_string(),
        },
        None => p.to_string(),
    }
}
