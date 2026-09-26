//! Our logits for a fixed token sequence, for comparison against llama.cpp's.
//!
//! An architecture is only as right as its agreement with an independent
//! implementation, and replies alone are a weak witness: a random-weight test
//! model (the only kind small enough to keep for every family) has near-flat
//! distributions, so a correct implementation and a wrong one both produce
//! noise. The logit VALUES are not noise — llama.cpp computes the same numbers
//! or it does not. `examples/compare_logits_reference.py` does the comparing.
//!
//! Runs the model on the processor, as two segments when asked, so the hidden
//! state crosses a boundary exactly as it does between two nodes:
//!
//! * a prompt pass over the first `N - DECODE` tokens, logits at every position
//!   (`forward_verify_all_positions`);
//! * then `DECODE` single-token steps against the cache that pass wrote — the
//!   path every generated token takes, and the one a prefill-only check misses.
//!
//! ```bash
//! LOGITS_PROBE_GGUF=~/swarmllm-ref/qwen3moe/qwen3moe-test-q8_0.gguf \
//! LOGITS_PROBE_OUT=/tmp/q3moe LOGITS_PROBE_SPLIT=1 \
//!   cargo run --release --no-default-features --features dev \
//!   --example logits_reference_probe
//! python3 examples/compare_logits_reference.py <gguf> /tmp/q3moe
//! ```
//!
//! `LOGITS_PROBE_IDS` (a JSON array of token ids) instead of the synthetic sequence;
//! `LOGITS_PROBE_N` (default 24) tokens, `LOGITS_PROBE_DECODE` (default 4) of
//! them as decode steps, `LOGITS_PROBE_SPLIT` the first layer of the second
//! segment (default: one segment). Writes `<OUT>.f32` (`[N, vocab]`, little
//! endian) and `<OUT>.json` (the token ids and the shape).

use std::path::PathBuf;

use candle_core::{Device, Tensor};
use swarmllm::inference::split::{GgufTensorMeta, KvCacheStore, SplitModel};

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn expand(p: &str) -> PathBuf {
    match (p.strip_prefix("~/"), std::env::var("HOME")) {
        (Some(rest), Ok(home)) => PathBuf::from(home).join(rest),
        _ => PathBuf::from(p),
    }
}

fn main() -> anyhow::Result<()> {
    let gguf = expand(
        &std::env::var("LOGITS_PROBE_GGUF")
            .map_err(|_| anyhow::anyhow!("set LOGITS_PROBE_GGUF to a whole .gguf file"))?,
    );
    let out = expand(&std::env::var("LOGITS_PROBE_OUT").unwrap_or_else(|_| "logits_probe".into()));
    // `LOGITS_PROBE_IDS=<file>`: a JSON array of token ids to use instead of the
    // synthetic sequence — natural text, so a disagreement cannot be blamed on
    // the input being nonsense (#124).
    let given_ids: Option<Vec<i64>> = match std::env::var("LOGITS_PROBE_IDS") {
        Ok(path) => Some(serde_json::from_slice(&std::fs::read(expand(&path))?)?),
        Err(_) => None,
    };
    let n = given_ids
        .as_ref()
        .map(|v| v.len())
        .unwrap_or_else(|| env_usize("LOGITS_PROBE_N", 24));
    let decode = env_usize("LOGITS_PROBE_DECODE", 4).min(n.saturating_sub(1));
    let layers = GgufTensorMeta::from_gguf_file(&gguf)?.block_count;
    let split = std::env::var("LOGITS_PROBE_SPLIT")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&k| k > 0 && k < layers);

    // Arbitrary ids well inside any vocabulary and away from the specials,
    // unless a real sequence was given.
    let ids: Vec<i64> =
        given_ids.unwrap_or_else(|| (0..n).map(|i| ((i * 997) % 5000 + 100) as i64).collect());
    let prefill = n - decode;

    let mut segments: Vec<SplitModel> = match split {
        Some(k) => vec![
            SplitModel::load_from_gguf(&gguf, 0, k, true, false, true)?,
            SplitModel::load_from_gguf(&gguf, k, layers, false, true, true)?,
        ],
        None => vec![SplitModel::load_from_gguf(
            &gguf, 0, layers, true, true, true,
        )?],
    };
    println!(
        "{}: {layers} layers as {} segment(s), {n} tokens ({prefill} prefill + {decode} decode)",
        gguf.display(),
        segments.len()
    );
    let store = KvCacheStore::new(std::time::Duration::from_secs(600));
    let req = "logits-probe";

    // Run `input` (token ids) through every segment; the LAST one answers with
    // logits at every position when `all_positions`, else at the last one.
    let mut run = |input: Tensor, pos: usize, all_positions: bool| -> anyhow::Result<Tensor> {
        let last = segments.len() - 1;
        let mut h = input;
        for (i, seg) in segments.iter_mut().enumerate() {
            h = match (i == 0, i == last, all_positions) {
                (true, true, true) => seg.forward_verify_all_positions(&h, pos, &store, req)?,
                (false, true, true) => {
                    seg.forward_verify_all_positions_pre_embedded(&h, pos, &store, req)?
                }
                (true, _, _) => seg.forward(&h, pos, &store, req)?,
                (false, _, _) => seg.forward_pre_embedded(&h, pos, &store, req)?,
            };
        }
        Ok(h)
    };

    let mut rows: Vec<Vec<f32>> = Vec::with_capacity(n);
    let prompt = Tensor::from_vec(ids[..prefill].to_vec(), &[1, prefill], &Device::Cpu)?;
    let all = run(prompt, 0, true)?.squeeze(0)?; // [prefill, vocab]
    for p in 0..prefill {
        rows.push(all.get(p)?.to_dtype(candle_core::DType::F32)?.to_vec1()?);
    }
    for (step, &id) in ids[prefill..].iter().enumerate() {
        let pos = prefill + step;
        let t = Tensor::from_vec(vec![id], &[1, 1], &Device::Cpu)?;
        let logits = run(t, pos, false)?.flatten_all()?;
        rows.push(logits.to_dtype(candle_core::DType::F32)?.to_vec1()?);
    }

    let vocab = rows[0].len();
    let mut bytes = Vec::with_capacity(n * vocab * 4);
    for row in &rows {
        anyhow::ensure!(row.len() == vocab, "position logits differ in length");
        for v in row {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
    }
    std::fs::write(out.with_extension("f32"), bytes)?;
    std::fs::write(
        out.with_extension("json"),
        serde_json::to_vec(&serde_json::json!({
            "gguf": gguf, "tokens": ids, "positions": n, "vocab": vocab,
            "prefill": prefill, "decode": decode, "split": split,
        }))?,
    )?;
    println!(
        "wrote {} ({n} x {vocab})",
        out.with_extension("f32").display()
    );
    Ok(())
}
