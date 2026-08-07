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

    SplitModel::load_from_shards_cpu(
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
        let decode = t.elapsed().as_secs_f64() / decode_tokens as f64;
        best_decode = best_decode.min(decode);

        println!(
            "  rep {rep}: prefill {:.2}s ({:.1} tok/s)   decode {:.1} ms/token",
            prefill,
            prompt_tokens as f64 / prefill,
            decode * 1000.0
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
