//! KV-refresh probe for split speculation (`docs/plans/split_speculation.md`
//! Phase 4, "Untested idea — KV refresh").
//!
//! A near machine drafting with a low-bit SHADOW of the far machine's layers
//! makes small errors on every past token, and each new draft attends over all
//! of them. The far machine computes the EXACT keys and values of every
//! confirmed token anyway. If it sent them back, the shadow would attend over an
//! exact history and approximate only the token being drafted.
//!
//! This measures that before anyone builds it: the target writes a greedy reply,
//! then two copies of the shadow read the same reply token by token — one on its
//! own cache, one whose far layers' cache is overwritten, before every token,
//! with an independent copy of the target's — and each is scored by how often
//! its next-token argmax equals the target's.
//!
//! Run: `SWARMLLM_KV_REFRESH_TARGET=<q4.gguf> SWARMLLM_KV_REFRESH_SHADOW=<shadow.gguf>
//! [SWARMLLM_KV_REFRESH_SPLIT=14] [SWARMLLM_KV_REFRESH_TOKENS=100]
//! cargo test --release kv_refresh_probe -- --ignored --nocapture`

use super::super::kv_cache::KvCacheStore;
use super::super::model::SplitModel;
use candle_core::{Device, Tensor};

fn ids(tokens: &[u32]) -> Tensor {
    let v: Vec<i64> = tokens.iter().map(|&t| t as i64).collect();
    Tensor::from_vec(v, (1, tokens.len()), &Device::Cpu).expect("ids tensor")
}

fn argmax(logits: &Tensor) -> u32 {
    let v: Vec<f32> = logits
        .flatten_all()
        .and_then(|t| t.to_dtype(candle_core::DType::F32))
        .and_then(|t| t.to_vec1())
        .expect("logits");
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i as u32)
        .expect("non-empty logits")
}

/// Replace `dst`'s cache for layers `from..` with independent copies of `src`'s.
fn refresh_far_layers(
    src: &KvCacheStore,
    src_key: &str,
    dst: &KvCacheStore,
    dst_key: &str,
    from: usize,
    num_layers: usize,
) {
    let copies: Vec<_> = {
        let entry = src.get_or_create_keyed(src_key, num_layers);
        entry.layers[from..]
            .iter()
            .map(|l| l.as_ref().map(|kv| kv.deep_copy().expect("deep copy")))
            .collect()
    };
    let mut entry = dst.get_or_create_keyed(dst_key, num_layers);
    for (i, kv) in copies.into_iter().enumerate() {
        entry.layers[from + i] = kv;
    }
}

#[test]
#[ignore = "needs SWARMLLM_KV_REFRESH_TARGET and SWARMLLM_KV_REFRESH_SHADOW; prints agreement"]
fn kv_refresh_probe() {
    let (Ok(tpath), Ok(spath)) = (
        std::env::var("SWARMLLM_KV_REFRESH_TARGET"),
        std::env::var("SWARMLLM_KV_REFRESH_SHADOW"),
    ) else {
        eprintln!("set SWARMLLM_KV_REFRESH_TARGET and SWARMLLM_KV_REFRESH_SHADOW");
        return;
    };
    let env_usize = |k: &str, d: usize| {
        std::env::var(k)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(d)
    };
    let split = env_usize("SWARMLLM_KV_REFRESH_SPLIT", 14);
    let n_gen = env_usize("SWARMLLM_KV_REFRESH_TOKENS", 100);

    let load = |p: &str| {
        SplitModel::load_from_gguf(std::path::Path::new(p), 0, 999, true, true, true)
            .expect("load gguf")
    };
    let mut target = load(&tpath);
    // ONE shadow for both arms: the KV cache lives in the store, not the model,
    // so two stores give two independent histories over the same weights (and
    // three 7B models at once is past the build slice's memory).
    let mut shadow = load(&spath);
    let num_layers = 256; // entries are sized up front; unused tail slots stay None
    let chat = |u: &str| {
        format!("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{u}<|im_end|>\n<|im_start|>assistant\n")
    };
    let prompts = [
        "Explain how a city's water supply works, from the reservoir to the tap. Use several paragraphs.",
        "Write a Python function that parses a CSV file of transactions (date, description, amount) and returns the total spent per month, with a short docstring and error handling.",
        "What are the main differences between TCP and UDP? Give concrete examples of when to use each.",
        "Summarize the causes and consequences of the 2008 financial crisis for a high-school student.",
    ];
    let eos: Vec<u32> = target.eos_tokens().to_vec();

    let (mut n, mut hit_own, mut hit_refreshed) = (0usize, 0usize, 0usize);
    for (pi, prompt) in prompts.iter().enumerate() {
        let (kt, ko, kr) = (format!("t{pi}"), format!("o{pi}"), format!("r{pi}"));
        let (st, so, sr) = (
            KvCacheStore::new(std::time::Duration::from_secs(3600)),
            KvCacheStore::new(std::time::Duration::from_secs(3600)),
            KvCacheStore::new(std::time::Duration::from_secs(3600)),
        );
        let key = |m: &SplitModel, id: &str| KvCacheStore::cache_key(m.kv_model_key(), id);
        let prompt_ids = target.encode_ids(&chat(prompt));

        // The prompt pass is exact on both machines, so the refreshed copy starts
        // from the target's cache for the far layers.
        let mut next = argmax(&target.forward(&ids(&prompt_ids), 0, &st, &kt).expect("t"));
        shadow.forward(&ids(&prompt_ids), 0, &so, &ko).expect("o");
        shadow.forward(&ids(&prompt_ids), 0, &sr, &kr).expect("r");
        let (tk, rk) = (key(&target, &kt), key(&shadow, &kr));
        refresh_far_layers(&st, &tk, &sr, &rk, split, num_layers);

        for pos in (prompt_ids.len()..).take(n_gen) {
            if eos.contains(&next) {
                break;
            }
            let x = [next];
            // Before the target consumes x: its far-layer cache holds exactly the
            // confirmed history the far machine would send back.
            refresh_far_layers(&st, &tk, &sr, &rk, split, num_layers);
            let t_next = argmax(&target.forward(&ids(&x), pos, &st, &kt).expect("t"));
            let o_next = argmax(&shadow.forward(&ids(&x), pos, &so, &ko).expect("o"));
            let r_next = argmax(&shadow.forward(&ids(&x), pos, &sr, &kr).expect("r"));
            n += 1;
            hit_own += usize::from(o_next == t_next);
            hit_refreshed += usize::from(r_next == t_next);
            next = t_next;
        }
        eprintln!(
            "prompt {pi}: positions so far {n}, own {:.3}, refreshed {:.3}",
            hit_own as f64 / n.max(1) as f64,
            hit_refreshed as f64 / n.max(1) as f64
        );
    }
    eprintln!(
        "KV REFRESH PROBE split={split} positions={n}: shadow on its own cache {:.4}, \
         with the target's far-layer cache {:.4}",
        hit_own as f64 / n.max(1) as f64,
        hit_refreshed as f64 / n.max(1) as f64
    );
}
