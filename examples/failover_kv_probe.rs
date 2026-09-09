//! Does a stand-in taking over a pipeline segment mid-reply compute the same
//! thing the machine it replaced would have?
//!
//! Reproduces the claim in `docs/FUTURE_WORK.md` § "A mid-decode failover
//! silently loses the failed segment's KV context" with no daemon, no network
//! and no killing of anyone's node. Two real segments of a real model:
//!
//! * segment A, layers `[0, k)` — the part that KEEPS its cache, as it does in
//!   a real failover; it is a different machine and nothing happened to it
//! * segment B, layers `[k, n)` — the part that FAILS, whose replacement holds
//!   no cache for this request
//!
//! `KvCacheStore` is keyed by request id, and `failover_segment` re-sends the
//! current step's forward unchanged. So a stand-in is exactly "the same input,
//! at the same `index_pos`, against a request id with no cache" — which is what
//! this drives, after decoding far enough that there is real history to lose.
//!
//! ```bash
//! SWARM_KV_PROBE_MODEL=~/.local/share/swarmllm/models/llama-3.2-3b-instruct-q4-k-m \
//!   cargo run --release --no-default-features --features dev \
//!   --example failover_kv_probe
//! ```
//!
//! `SWARM_KV_PROBE_PROMPT` sets the prompt length in tokens (default 64),
//! `SWARM_KV_PROBE_DECODE` how many tokens to generate before the takeover
//! (default 24), and `SWARM_KV_PROBE_SPLIT` the fraction of layers that stay on
//! segment A (default 0.5).
//!
//! **What a null result looks like**, and it is a real possibility worth
//! stating before running: if the stand-in's next token matches, and the
//! logits track, then losing the cache does not change the answer materially
//! and the FUTURE_WORK entry is wrong. That would be the more useful outcome.

use std::path::{Path, PathBuf};

use candle_core::{Device, Tensor};
use swarmllm::inference::split::{KvCacheStore, SplitModel};
use swarmllm_types::{ModelManifest, ShardTensorEntry};

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn shellexpand(p: &str) -> String {
    if let Some(rest) = p.strip_prefix("~/") {
        if let Ok(home) = std::env::var("HOME") {
            return format!("{home}/{rest}");
        }
    }
    p.to_string()
}

/// Load one contiguous layer range as its own segment, the way a node holding
/// part of a model does.
fn load_segment(
    model_dir: &Path,
    layer_start: usize,
    layer_end: usize,
    is_first: bool,
    is_last: bool,
) -> anyhow::Result<SplitModel> {
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
        "need every shard of {}: have {} of {}",
        model_dir.display(),
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
        layer_start,
        layer_end,
        is_first,
        is_last,
    )
    .map_err(|e| anyhow::anyhow!("load {layer_start}..{layer_end}: {e}"))
}

/// Greedy pick, plus the top of the distribution so a near-miss can be told
/// from a rout.
fn argmax_and_top(logits: &Tensor) -> anyhow::Result<(u32, Vec<(u32, f32)>)> {
    let flat = logits.flatten_all()?;
    let v: Vec<f32> = flat.to_vec1()?;
    let vocab = v.len();
    // The last position's row when several are returned.
    let row = &v[vocab.saturating_sub(vocab)..];
    let mut idx: Vec<u32> = (0..row.len() as u32).collect();
    idx.sort_by(|a, b| {
        row[*b as usize]
            .partial_cmp(&row[*a as usize])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let top: Vec<(u32, f32)> = idx.iter().take(5).map(|i| (*i, row[*i as usize])).collect();
    Ok((idx[0], top))
}

/// Softmax probability the given row assigns to `token`, and the top-1 margin
/// over the runner-up.
///
/// The headline metric, because raw-logit cosine is not one: logit vectors
/// carry a large common component (the head's frequency prior), so two rows can
/// agree on the answer and still score near zero. What decides the reply is the
/// DISTRIBUTION — how much mass sits on the token the healthy machine would
/// have chosen, and how confidently. Both API surfaces sample by default (0.7
/// on OpenAI, 1.0 on Anthropic), so a collapsed margin diverges even where the
/// greedy pick happens to survive.
fn prob_and_margin(logits: &Tensor, token: u32) -> anyhow::Result<(f32, f32)> {
    let v: Vec<f32> = logits.flatten_all()?.to_vec1()?;
    let max = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = v.iter().map(|x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    let p = exps[token as usize] / sum;
    let mut sorted = v.clone();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    Ok((p, sorted[0] - sorted[1]))
}

/// Cosine similarity, so "different" can be quantified rather than asserted.
fn cosine(a: &Tensor, b: &Tensor) -> anyhow::Result<f32> {
    let x: Vec<f32> = a.flatten_all()?.to_vec1()?;
    let y: Vec<f32> = b.flatten_all()?.to_vec1()?;
    anyhow::ensure!(
        x.len() == y.len(),
        "shape mismatch {} vs {}",
        x.len(),
        y.len()
    );
    let dot: f32 = x.iter().zip(&y).map(|(p, q)| p * q).sum();
    let nx: f32 = x.iter().map(|p| p * p).sum::<f32>().sqrt();
    let ny: f32 = y.iter().map(|q| q * q).sum::<f32>().sqrt();
    Ok(if nx == 0.0 || ny == 0.0 {
        0.0
    } else {
        dot / (nx * ny)
    })
}

fn main() -> anyhow::Result<()> {
    let dir = std::env::var("SWARM_KV_PROBE_MODEL").map_err(|_| {
        anyhow::anyhow!("set SWARM_KV_PROBE_MODEL to a model dir holding every shard")
    })?;
    let model_dir = PathBuf::from(shellexpand(&dir));
    let prompt_tokens = env_usize("SWARM_KV_PROBE_PROMPT", 64);
    let decode_tokens = env_usize("SWARM_KV_PROBE_DECODE", 24);

    let manifest: ModelManifest =
        serde_json::from_slice(&std::fs::read(model_dir.join("manifest.json"))?)?;
    let num_layers = manifest.num_layers as usize;
    let split_frac: f64 = std::env::var("SWARM_KV_PROBE_SPLIT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.5);
    let k = ((num_layers as f64 * split_frac) as usize).clamp(1, num_layers - 1);

    println!("model {}", model_dir.display());
    println!("segment A = layers [0,{k}) — keeps its cache (a different machine)");
    println!("segment B = layers [{k},{num_layers}) — fails; its stand-in holds no cache\n");

    let mut a = load_segment(&model_dir, 0, k, true, false)?;
    let mut b = load_segment(&model_dir, k, num_layers, false, true)?;

    let store = KvCacheStore::new(std::time::Duration::from_secs(600));
    let req = "probe-request";
    // The control. A SECOND request driven through the identical tokens, so it
    // ends the decode holding an equivalent cache. At the takeover step it is
    // handed the same hidden state as the stand-in — the only difference being
    // that it HAS the history. If this arm disagreed too, the probe would be
    // measuring request-id sensitivity rather than the missing cache, and its
    // verdict would be worthless. Always run, never optional: a comparison that
    // cannot report "no difference" is not a measurement.
    let twin = "probe-request-twin";

    // Arbitrary ids, well inside any vocabulary and away from 0.
    let ids: Vec<i64> = (0..prompt_tokens)
        .map(|i| (i % 20_000 + 100) as i64)
        .collect();
    let prompt = Tensor::from_vec(ids, &[1, prompt_tokens], &Device::Cpu)?;

    // Prompt pass through both segments.
    let h = a
        .forward(&prompt, 0, &store, req)
        .map_err(|e| anyhow::anyhow!("A prefill: {e}"))?;
    let logits = b
        .forward_pre_embedded(&h, 0, &store, req)
        .map_err(|e| anyhow::anyhow!("B prefill: {e}"))?;
    let (mut next, _) = argmax_and_top(&logits)?;

    // The control's own prompt pass, through the same tokens.
    let h_twin = a
        .forward(&prompt, 0, &store, twin)
        .map_err(|e| anyhow::anyhow!("A prefill (twin): {e}"))?;
    b.forward_pre_embedded(&h_twin, 0, &store, twin)
        .map_err(|e| anyhow::anyhow!("B prefill (twin): {e}"))?;

    // Decode, so there is real history for the stand-in to be missing.
    let mut pos = prompt_tokens;
    for _ in 0..decode_tokens {
        let t = Tensor::from_vec(vec![next as i64], &[1, 1], &Device::Cpu)?;
        let h = a
            .forward(&t, pos, &store, req)
            .map_err(|e| anyhow::anyhow!("A decode: {e}"))?;
        let logits = b
            .forward_pre_embedded(&h, pos, &store, req)
            .map_err(|e| anyhow::anyhow!("B decode: {e}"))?;
        // The control follows the same tokens, one step behind in the same
        // loop, so it arrives at the takeover holding an equivalent history.
        let h_twin = a
            .forward(&t, pos, &store, twin)
            .map_err(|e| anyhow::anyhow!("A decode (twin): {e}"))?;
        b.forward_pre_embedded(&h_twin, pos, &store, twin)
            .map_err(|e| anyhow::anyhow!("B decode (twin): {e}"))?;

        next = argmax_and_top(&logits)?.0;
        pos += 1;
    }
    println!("decoded {decode_tokens} tokens; now at position {pos}\n");

    // The takeover. Segment A is untouched — it is a different machine and
    // nothing happened to it — so both arms get the SAME input hidden state.
    let t = Tensor::from_vec(vec![next as i64], &[1, 1], &Device::Cpu)?;
    let h = a
        .forward(&t, pos, &store, req)
        .map_err(|e| anyhow::anyhow!("A takeover step: {e}"))?;

    // What the machine that failed WOULD have produced: its own cache, intact.
    let healthy = b
        .forward_pre_embedded(&h, pos, &store, req)
        .map_err(|e| anyhow::anyhow!("B healthy: {e}"))?;

    // The control: a different request id that DOES hold the history.
    let control = b
        .forward_pre_embedded(&h, pos, &store, twin)
        .map_err(|e| anyhow::anyhow!("B control: {e}"))?;

    // What the stand-in produces: same input, same index_pos, no cache. This is
    // precisely what `failover_segment` sends.
    let standby = b
        .forward_pre_embedded(&h, pos, &store, "stand-in-has-no-cache")
        .map_err(|e| anyhow::anyhow!("B stand-in: {e}"))?;

    let (tok_healthy, top_healthy) = argmax_and_top(&healthy)?;
    let (tok_control, _) = argmax_and_top(&control)?;
    let (tok_standby, top_standby) = argmax_and_top(&standby)?;
    let cos_control = cosine(&healthy, &control)?;
    let cos = cosine(&healthy, &standby)?;

    let (p_healthy, margin_healthy) = prob_and_margin(&healthy, tok_healthy)?;
    let (p_control, _) = prob_and_margin(&control, tok_healthy)?;
    let (p_standby, margin_standby) = prob_and_margin(&standby, tok_healthy)?;

    println!("next token, machine intact      : {tok_healthy}  top5 {top_healthy:?}");
    println!("next token, control (has history): {tok_control}  cosine {cos_control:.6}");
    println!("next token, stand-in (no cache)  : {tok_standby}  top5 {top_standby:?}");
    println!("\ncosine(logits) vs intact — control {cos_control:.6}, stand-in {cos:.6}");
    println!("P(intact machine's token) — intact {p_healthy:.4}, control {p_control:.4}, stand-in {p_standby:.4}");
    println!("top-1 margin — intact {margin_healthy:.3}, stand-in {margin_standby:.3}");

    let control_agrees = tok_control == tok_healthy && cos_control > 0.999;
    if !control_agrees {
        println!(
            "\nPROBE INVALID — the control disagrees with the intact machine, so this \
             is measuring something other than the missing cache. Do not read the \
             stand-in result."
        );
        return Ok(());
    }

    if tok_healthy == tok_standby && cos > 0.999 {
        println!(
            "\nNULL RESULT — the stand-in agrees too. Losing the cache did not change \
             the answer here; the FUTURE_WORK entry overstates it."
        );
    } else {
        println!(
            "\nCONFIRMED — the control, holding the history, reproduces the intact \
             machine exactly; the stand-in, differing ONLY in that it has no cache, \
             computes a different continuation. Nothing errored, nothing warned."
        );
    }
    Ok(())
}
