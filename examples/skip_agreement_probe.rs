//! How well does a model guess its own next tokens with a block of its layers
//! skipped — the guesser a card/processor split could run on its idle card,
//! skipping the layers the processor holds (deep dive 2026-10-28).
//!
//! The full model writes a greedy reply. At every position, with the full
//! model's cache for everything before it, each candidate skip set guesses
//! `GAMMA` tokens in a chain (its own cache for its own guesses, as a drafter
//! would), and the probe counts how many of them the full model's reply
//! agrees with, in a row. The cache is cut back after every chain, so each
//! position is guessed from exactly the state a check would start from — a
//! broken drafting loop cannot make a skip set look worse than it is
//! (SWIFT's "accept rate effectively zero", FUTURE_WORK_ARCHIVE 2026-08-06).
//!
//! Candidates, each skipping `SKIP` layers (default 44% of them, a 14B's
//! processor share on an 8 GB card):
//! * `tail` — the last `SKIP` layers: the card-first placement every split
//!   uses today, guessing from the card's layers and the output head;
//! * `mid@s` — the block `[s, s + SKIP)`, every `STEP`-th start: a placement
//!   that keeps the model's first AND last layers on the card.
//!
//! ```bash
//! PROBE_GGUF=~/swarmllm-ref/qwen2.5-coder-7b-instruct-q4-k-m.gguf PROBE_DEVICE=cuda \
//!   cargo run --release --no-default-features --features dev --example skip_agreement_probe
//! ```
//! `PROBE_MODEL=<model directory>` loads a model a node holds as shards instead;
//! `PROBE_GPU_LAYERS=n` keeps only n layers on the card (a model too big for it);
//! `PROBE_PROMPTS=<file>` holds rendered prompts separated by a line `=====`
//! (default: a code request and a prose one in ChatML); `PROBE_N` reply tokens
//! (default 96); `PROBE_GAMMA` (4); `PROBE_SKIP` layers; `PROBE_STEP` (2).
//! Each line also reads how often the model's token was among the first
//! guess's best two / four choices (`top2` / `top4`), and with
//! `PROBE_TREE2=1` what the better of two chains — from the best and the
//! second-best first guess, a tree's first level — keeps a round (`tree2`).
//!
//! `PROBE_BRIDGE=1` instead fixes ONE block — `[PROBE_START, + SKIP)`, the
//! start defaulting to a tenth of the depth as `hybrid::processor_run` places
//! a whole model's processor run — and compares what stands in for it while
//! the card guesses (`split::SkipBridge`), each fitted from the real forwards
//! before the position guessed: `none` (the hidden state passes through, as
//! the card guesses today), `mean` (the block's mean effect so far), `prev`
//! (its effect at the latest real position) and `recall@λ` (the mean plus a
//! linear kernel ridge over the last `PROBE_WINDOW` positions, 256, by the
//! cosine of their hidden states where the block starts), and `linear@λ` (a
//! ridge map over EVERY real position so far, the prompt's included, refitted
//! every `PROBE_REFIT` tokens), and `online` (the library's
//! `split::RunStandIn`, which the worker's card guesses read: `recall@0.3`'s
//! arithmetic kept current a row at a time, over the same `PROBE_WINDOW` — it
//! should agree with `recall@0.3` to rounding; `PROBE_STANDS=a,b` runs only
//! those named, and `online` alone is the cheap way to try a long window,
//! since `recall` inverts its window whole at every position). It also prints how
//! far each stand-in's estimate of the block's effect lands from the effect
//! the next real forward measures (`err`, relative; `none` is 1).
//!
//! Any stand named `<stand>~ρ` or `<stand>~ρxk` in `PROBE_STANDS` runs again
//! with its next round's guesses shifted by ρ times the mean of its last k
//! residuals (what the real forwards measured less the stand's own estimate,
//! k = 1 by default) — whether a stand-in's miss at one position carries to
//! the next. A conversation starts with none.
//!
//! Two more ask whether a stand-in can outlive its conversation — what a node
//! would carry from the replies it wrote before (FUTURE_WORK #366 (1)):
//! `prior@λ` is a ridge map fitted ONLY on the earlier prompts' real
//! positions (prompt and reply, in `PROBE_PROMPTS` order; the first prompt has
//! none, so it guesses as `none`), and `pooled@λ` the same map with this
//! conversation's positions so far added, refitted every `PROBE_REFIT`
//! tokens. Both keep sufficient statistics (`ΦᵀΦ`, `ΦᵀΔ`, the sums), never
//! the rows, so the carried state is two hidden × hidden matrices however
//! long the history. `prior+recall@λ` is the form a worker could run: the
//! prior's map, fitted once a conversation (off the reply's path), plus
//! `recall@λ`'s window fitted to what the prior leaves of each effect.
//! `prior<N>+recall@λ` is the same in the arithmetic a worker would keep on
//! its card: the prior as kernel ridge over the last `N` positions of the
//! earlier conversations (features and weights, `N × hidden` each — the
//! dual of the map, equal to it while `N` holds every earlier position),
//! the window fitted to its residuals, both one `Recall` bridge. `carried`
//! is the library's own (`split::StandInPrior` beside `split::RunStandIn`, as
//! a worker runs them): the prior refitted once a conversation is out, and
//! with a prompt longer than the window for its own reply.

use std::path::PathBuf;

use candle_core::{DType, Tensor};
use swarmllm::inference::split::{
    GgufTensorMeta, KvCacheStore, RunStandIn, SkipBridge, SplitModel, StandInPrior, STAND_IN_CARRY,
    STAND_IN_CARRY_POSITIONS,
};
use swarmllm_types::{ModelManifest, ShardTensorEntry};

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

fn load_from_model_dir(dir: &std::path::Path, force_cpu: bool) -> anyhow::Result<SplitModel> {
    swarmllm::inference::split::ensure_gguf_header(dir)
        .map_err(|e| anyhow::anyhow!("gguf header: {e}"))?;
    let manifest: ModelManifest =
        serde_json::from_slice(&std::fs::read(dir.join("manifest.json"))?)?;
    let mut shards: Vec<(u32, PathBuf, Vec<ShardTensorEntry>)> = Vec::new();
    for shard in &manifest.shards {
        let path = dir.join(format!("shard_{:03}.bin", shard.index));
        anyhow::ensure!(path.exists(), "missing {}", path.display());
        shards.push((shard.index, path, shard.tensors.clone()));
    }
    shards.sort_by_key(|(i, _, _)| *i);
    let entries: Vec<Vec<ShardTensorEntry>> = shards.iter().map(|s| s.2.clone()).collect();
    let files: Vec<(u32, PathBuf)> = shards.into_iter().map(|(i, p, _)| (i, p)).collect();
    let layers = manifest.num_layers as usize;
    let load = if force_cpu {
        SplitModel::load_from_shards_cpu
    } else {
        SplitModel::load_from_shards
    };
    load(
        dir,
        files,
        &entries,
        manifest.total_size_bytes,
        0,
        layers,
        true,
        true,
    )
    .map_err(|e| anyhow::anyhow!("load: {e}"))
}

/// The first of equal maxima, as `sampling::argmax` picks.
fn argmax(logits: &candle_core::Tensor) -> anyhow::Result<u32> {
    let v: Vec<f32> = logits
        .to_dtype(candle_core::DType::F32)?
        .flatten_all()?
        .to_vec1()?;
    let mut best = 0usize;
    for (i, &x) in v.iter().enumerate() {
        if x > v[best] {
            best = i;
        }
    }
    Ok(best as u32)
}

/// The `k` best ids, best first; equal scores go to the smaller id, as
/// `argmax` breaks them.
fn ranked(logits: &candle_core::Tensor, k: usize) -> anyhow::Result<Vec<u32>> {
    let v: Vec<f32> = logits
        .to_dtype(candle_core::DType::F32)?
        .flatten_all()?
        .to_vec1()?;
    let mut best: Vec<usize> = Vec::with_capacity(k + 1);
    for (i, &x) in v.iter().enumerate() {
        let at = best.iter().position(|&b| x > v[b]).unwrap_or(best.len());
        if at < k {
            best.insert(at, i);
            best.truncate(k);
        }
    }
    Ok(best.into_iter().map(|i| i as u32).collect())
}

const DEFAULT_PROMPTS: &str = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nWrite a Python function that reads a CSV file of orders with columns id, customer, amount and date, and returns the total amount per customer per month. Use type hints, a docstring, and handle malformed rows by skipping them with a warning.<|im_end|>\n<|im_start|>assistant\n
=====
<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nExplain to someone with no science background how a vaccine teaches the immune system to recognise a virus, in three short paragraphs.<|im_end|>\n<|im_start|>assistant\n";

/// What stands in for the skipped block while a candidate guesses.
#[derive(Clone, Copy)]
enum Stand {
    None,
    Mean,
    Prev,
    Recall(f64),
    /// A linear map fitted over EVERY real position so far — the prompt's
    /// included — by ridge regression (`ridge_by_cg`), refitted every
    /// `PROBE_REFIT` tokens (16): what a node could fit from the hidden states
    /// each check hands back, with no copy of the skipped layers.
    Linear(f64),
    /// `split::RunStandIn`: `Recall(0.3)` kept current a row at a time.
    Online,
    /// `split::RunStandIn` over `split::StandInPrior`, as a worker runs them.
    Carried,
    /// `Carried` with the library's own carry (`RunStandIn::with_carry` at
    /// `STAND_IN_CARRY` × `STAND_IN_CARRY_POSITIONS`): must read as the
    /// probe's `carried~0.5x4`, its arithmetic done here.
    CarriedLib,
    /// A ridge map fitted only on the EARLIER prompts' real positions
    /// (`Carried`), once per prompt.
    Prior(f64),
    /// `Prior`'s statistics plus this conversation's positions so far,
    /// refitted every `PROBE_REFIT` tokens.
    Pooled(f64),
    /// `Prior`'s map plus `Recall`'s window fitted to its residuals.
    PriorRecall(f64),
    /// The same with the prior as kernel ridge over the last `usize`
    /// positions of the earlier conversations (module docs) — and, where
    /// the `bool` says, this conversation's prompt after them, the prior
    /// fitted once when the prompt has been read.
    PriorRows(f64, usize, bool),
}

struct Candidate {
    name: String,
    mask: Vec<bool>,
    stand: Stand,
    /// Summed relative error of its estimate of the block's effect, and how
    /// many it made (bridge mode).
    err: f64,
    errs: usize,
    /// Estimates that were not finite, left out of `err`.
    nonfinite: usize,
    /// First guesses the full model agreed with.
    first: usize,
    /// Guesses accepted in a row, summed over positions.
    run: usize,
    /// Positions guessed.
    tried: usize,
    /// Positions whose first guess had the model's token among its best two
    /// / four choices — what a tree's first level could keep.
    top2: usize,
    top4: usize,
    /// Tokens kept by the better of two chains, from the first guess's best
    /// and second-best choices (`PROBE_TREE2=1`), summed over positions.
    tree2: usize,
    /// `<stand>~ρxk`: ρ times the mean of the last `k` residuals — the
    /// effect a real forward measured less this stand's estimate of it —
    /// added to every guess of the next round (0: off).
    carry: f64,
    carry_k: usize,
    resids: std::collections::VecDeque<Tensor>,
}

fn main() -> anyhow::Result<()> {
    let model_dir = std::env::var("PROBE_MODEL").ok().map(|p| expand(&p));
    let force_cpu = std::env::var("PROBE_DEVICE").as_deref() != Ok("cuda");
    if let Some(n) = std::env::var("PROBE_GPU_LAYERS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
    {
        swarmllm::inference::split::GPU_LAYER_LIMIT.store(n, std::sync::atomic::Ordering::Relaxed);
    }
    let (mut model, layers) = match &model_dir {
        Some(dir) => {
            let m = load_from_model_dir(dir, force_cpu)?;
            let manifest: ModelManifest =
                serde_json::from_slice(&std::fs::read(dir.join("manifest.json"))?)?;
            (m, manifest.num_layers as usize)
        }
        None => {
            let gguf = expand(
                &std::env::var("PROBE_GGUF")
                    .map_err(|_| anyhow::anyhow!("set PROBE_GGUF or PROBE_MODEL"))?,
            );
            let layers = GgufTensorMeta::from_gguf_file(&gguf)?.block_count;
            (
                SplitModel::load_from_gguf(&gguf, 0, layers, true, true, force_cpu)?,
                layers,
            )
        }
    };
    let n = env_usize("PROBE_N", 96);
    let gamma = env_usize("PROBE_GAMMA", 4).max(1);
    let tree2 = std::env::var("PROBE_TREE2").is_ok_and(|v| v == "1");
    let skip = env_usize("PROBE_SKIP", (layers as f64 * 0.44).round() as usize).min(layers - 2);
    let step = env_usize("PROBE_STEP", 2).max(1);
    let prompts_text = match std::env::var("PROBE_PROMPTS") {
        Ok(p) => std::fs::read_to_string(expand(&p))?,
        Err(_) => DEFAULT_PROMPTS.replace("\\n", "\n"),
    };
    let prompts: Vec<String> = prompts_text
        .split("\n=====\n")
        .map(|s| s.trim_start_matches('\n').to_string())
        .filter(|s| !s.trim().is_empty())
        .collect();

    let mut candidates: Vec<Candidate> = Vec::new();
    let mk = |name: String, range: std::ops::Range<usize>| {
        let mut mask = vec![false; layers];
        for i in range {
            mask[i] = true;
        }
        Candidate {
            name,
            mask,
            stand: Stand::None,
            err: 0.0,
            errs: 0,
            nonfinite: 0,
            first: 0,
            run: 0,
            tried: 0,
            top2: 0,
            top4: 0,
            tree2: 0,
            carry: 0.0,
            carry_k: 1,
            resids: std::collections::VecDeque::new(),
        }
    };
    let bridge_mode = std::env::var("PROBE_BRIDGE").is_ok_and(|v| v == "1");
    let block_start = env_usize("PROBE_START", ((layers as f64) * 0.1).round() as usize)
        .clamp(1, layers - skip - 1);
    let window = env_usize("PROBE_WINDOW", 256).max(1);
    if bridge_mode {
        let stands: Vec<(String, Stand)> = vec![
            ("none".to_string(), Stand::None),
            ("mean".to_string(), Stand::Mean),
            ("prev".to_string(), Stand::Prev),
            ("recall@0.3".to_string(), Stand::Recall(0.3)),
            ("recall@3".to_string(), Stand::Recall(3.0)),
            ("linear@0.3".to_string(), Stand::Linear(0.3)),
            ("linear@3".to_string(), Stand::Linear(3.0)),
            ("online".to_string(), Stand::Online),
            ("carried".to_string(), Stand::Carried),
            ("carried-lib".to_string(), Stand::CarriedLib),
            ("prior@0.3".to_string(), Stand::Prior(0.3)),
            ("pooled@0.3".to_string(), Stand::Pooled(0.3)),
            ("prior+recall@0.3".to_string(), Stand::PriorRecall(0.3)),
            (
                "prior512+recall@0.3".to_string(),
                Stand::PriorRows(0.3, 512, false),
            ),
            (
                "prior1024+recall@0.3".to_string(),
                Stand::PriorRows(0.3, 1024, false),
            ),
            (
                "prior2048+recall@0.3".to_string(),
                Stand::PriorRows(0.3, 2048, false),
            ),
            (
                "prior1024p+recall@0.3".to_string(),
                Stand::PriorRows(0.3, 1024, true),
            ),
            (
                "prior2048p+recall@0.3".to_string(),
                Stand::PriorRows(0.3, 2048, true),
            ),
        ];
        let only = std::env::var("PROBE_STANDS").unwrap_or_default();
        for (name, stand) in &stands {
            if !only.is_empty() && !only.split(',').any(|s| s == name) {
                continue;
            }
            let mut c = mk(name.clone(), block_start..block_start + skip);
            c.stand = *stand;
            candidates.push(c);
        }
        // `<stand>~ρ` / `<stand>~ρxk`: that stand, its next round's guesses
        // shifted by ρ times the mean of its last k residuals.
        for entry in only.split(',').filter(|s| s.contains('~')) {
            let (base, carry) = entry.split_once('~').expect("has a ~");
            let (rho, k) = match carry.split_once('x') {
                Some((r, k)) => (r.parse::<f64>()?, k.parse::<usize>()?.max(1)),
                None => (carry.parse::<f64>()?, 1),
            };
            let Some((_, stand)) = stands.iter().find(|(n, _)| n == base) else {
                anyhow::bail!("no stand {base} for {entry}");
            };
            let mut c = mk(entry.to_string(), block_start..block_start + skip);
            c.stand = *stand;
            c.carry = rho;
            c.carry_k = k;
            candidates.push(c);
        }
    } else if skip > 0 {
        // `PROBE_SKIP=0` only writes the replies, one token a forward — what an
        // expert trace (`SWARMLLM_TRACE_EXPERTS`) is read off.
        candidates.push(mk("tail".into(), layers - skip..layers));
        let mut s = 1;
        while s + skip < layers {
            candidates.push(mk(format!("mid@{s}"), s..s + skip));
            s += step;
        }
    }
    println!(
        "{layers} layers, skipping {skip} ({:.0}%), {} candidates, gamma {gamma}, {n} tokens a prompt, {} prompt(s)",
        100.0 * skip as f64 / layers as f64,
        candidates.len(),
        prompts.len()
    );

    let key = model.kv_model_key().to_string();
    let mut carried: Option<Carried> = None;
    // The library's prior across conversations (`Stand::Carried`).
    let mut model_prior: Option<StandInPrior> = None;
    // Every earlier conversation's `(φ, Δ)` rows, oldest first.
    let mut earlier: Vec<(Tensor, Tensor)> = Vec::new();
    for (pi, prompt) in prompts.iter().enumerate() {
        let store = KvCacheStore::new(std::time::Duration::from_secs(3600));
        let req = format!("skip-probe-{pi}");
        let ids = model.encode_ids(prompt);
        store.set_reserved_positions(&req, ids.len() + n + gamma + 1);
        let t0 = std::time::Instant::now();
        let capture: std::collections::HashSet<usize> = [block_start - 1, block_start + skip - 1]
            .into_iter()
            .collect();
        let mut history = History {
            online: bridge_mode
                .then(|| {
                    let mut skips = vec![false; layers];
                    skips[block_start..block_start + skip].fill(true);
                    RunStandIn::for_run(&skips, model.device()).map(|s| s.with_window(window))
                })
                .flatten()
                .map(std::cell::RefCell::new),
            carried: bridge_mode
                .then(|| {
                    let mut skips = vec![false; layers];
                    skips[block_start..block_start + skip].fill(true);
                    RunStandIn::for_run(&skips, model.device())
                        .map(|s| s.with_window(window).collecting())
                })
                .flatten()
                .map(std::cell::RefCell::new),
            carried_lib: bridge_mode
                .then(|| {
                    let mut skips = vec![false; layers];
                    skips[block_start..block_start + skip].fill(true);
                    RunStandIn::for_run(&skips, model.device()).map(|s| {
                        s.with_window(window)
                            .with_carry(STAND_IN_CARRY, STAND_IN_CARRY_POSITIONS)
                    })
                })
                .flatten()
                .map(std::cell::RefCell::new),
            ..History::default()
        };
        let logits = if bridge_mode {
            let (logits, caught) = model.forward_capturing(
                &model.tensor_from_ids(&ids)?,
                0,
                &store,
                &req,
                &capture,
            )?;
            history.push(&caught, block_start, skip)?;
            history.start_carried(&caught, &mut model_prior)?;
            logits
        } else {
            model.forward(&model.tensor_from_ids(&ids)?, 0, &store, &req)?
        };
        for cand in candidates.iter_mut() {
            cand.resids.clear();
        }
        let mut tok = argmax(&logits)?;
        // reply[i] is the token at position ids.len() + i.
        let mut reply: Vec<u32> = vec![tok];
        // Each candidate's chain guessed at each reply index.
        let mut chains: Vec<Vec<Vec<u32>>> = vec![Vec::new(); candidates.len()];
        // Each candidate's four best first guesses, and (`PROBE_TREE2`) its
        // chain from the second best, at each reply index.
        let mut firsts: Vec<Vec<Vec<u32>>> = vec![Vec::new(); candidates.len()];
        let mut seconds: Vec<Vec<Vec<u32>>> = vec![Vec::new(); candidates.len()];
        for i in 0..n {
            let p = ids.len() + i;
            let run_end = block_start + skip;
            let fitted: Vec<Option<SkipBridge>> = candidates
                .iter()
                .map(|c| history.bridge(c.stand, run_end, window, carried.as_ref(), &earlier))
                .collect::<anyhow::Result<_>>()?;
            let bridges: Vec<Option<SkipBridge>> = fitted
                .iter()
                .zip(&candidates)
                .map(|(b, cand)| carried_forward(b.clone(), cand))
                .collect::<anyhow::Result<_>>()?;
            for (c, cand) in candidates.iter().enumerate() {
                // The guess's logits after feeding `t` at position `at`.
                let mut guess_at = |t: u32, at: usize| -> anyhow::Result<Tensor> {
                    let input = model.token_tensor(t)?;
                    Ok(match &bridges[c] {
                        Some(b) => model
                            .forward_with_skip_bridge(&input, at, &store, &req, &cand.mask, b)?,
                        None => {
                            model.forward_with_skip_mask(&input, at, &store, &req, &cand.mask)?
                        }
                    })
                };
                // Guess g follows the token at position p + g.
                let first = ranked(&guess_at(tok, p)?, 4)?;
                let mut chain = vec![first[0]];
                while chain.len() < gamma {
                    chain.push(argmax(&guess_at(chain[chain.len() - 1], p + chain.len())?)?);
                }
                if tree2 {
                    // The second branch shares the first level's forward.
                    store.truncate_request_to(&key, &req, p + 1)?;
                    let mut second = vec![first[1]];
                    while second.len() < gamma {
                        second.push(argmax(&guess_at(
                            second[second.len() - 1],
                            p + second.len(),
                        )?)?);
                    }
                    seconds[c].push(second);
                }
                store.truncate_request_to(&key, &req, p)?;
                chains[c].push(chain);
                firsts[c].push(first);
            }
            let l = if bridge_mode {
                let (l, caught) = model.forward_capturing(
                    &model.token_tensor(tok)?,
                    p,
                    &store,
                    &req,
                    &capture,
                )?;
                // How far each stand-in's estimate lands from the effect this
                // real forward measured, before it joins the history.
                let (h_in, delta) = History::rows(&caught, block_start, skip)?;
                for (c, cand) in candidates.iter_mut().enumerate() {
                    if cand.carry != 0.0 {
                        // The fitted stand's own miss here, before any carry.
                        if let Some(b) = &fitted[c] {
                            let r = (&delta - History::estimate(b, &h_in)?)?;
                            let r = r.reshape(r.elem_count())?;
                            if r.sum_all()?.to_scalar::<f32>()?.is_finite() {
                                cand.resids.push_back(r);
                                while cand.resids.len() > cand.carry_k {
                                    cand.resids.pop_front();
                                }
                            }
                        }
                    }
                    let est = match &bridges[c] {
                        Some(b) => History::estimate(b, &h_in)?,
                        None => h_in.zeros_like()?,
                    };
                    let num = (est - &delta)?.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
                    let den = delta.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt().max(1e-6);
                    let e = f64::from(num / den);
                    if e.is_finite() {
                        cand.err += e;
                        cand.errs += 1;
                    } else {
                        cand.nonfinite += 1;
                    }
                }
                history.push(&caught, block_start, skip)?;
                l
            } else {
                model.forward(&model.token_tensor(tok)?, p, &store, &req)?
            };
            tok = argmax(&l)?;
            reply.push(tok);
        }
        if let (Some(phi), Some(delta)) = (&history.phi, &history.delta) {
            let this = Carried::of(phi, delta)?;
            match carried.as_mut() {
                Some(c) => c.merge(&this)?,
                None => carried = Some(this),
            }
            earlier.push((phi.clone(), delta.clone()));
        }
        if let (Some(si), Some(prior)) = (history.carried.take(), model_prior.as_mut()) {
            prior.absorb(&mut si.into_inner(), model.device())?;
        }
        let mut this_prompt = String::new();
        for (c, cand) in candidates.iter_mut().enumerate() {
            let (first_before, run_before, tried_before) = (cand.first, cand.run, cand.tried);
            let (top2_before, top4_before, tree2_before) = (cand.top2, cand.top4, cand.tree2);
            let nonfinite = std::mem::take(&mut cand.nonfinite);
            for (i, chain) in chains[c].iter().enumerate() {
                // The chain guessed at reply index i follows reply[i]: its
                // guesses are reply[i+1..].
                let truth = &reply[i + 1..];
                let ok = chain.iter().zip(truth).take_while(|(a, b)| a == b).count();
                cand.tried += 1;
                cand.first += usize::from(ok > 0);
                cand.run += ok;
                let rank = firsts[c][i].iter().position(|&t| Some(&t) == truth.first());
                cand.top2 += usize::from(rank.is_some_and(|r| r < 2));
                cand.top4 += usize::from(rank.is_some());
                if let Some(second) = seconds[c].get(i) {
                    let ok2 = second.iter().zip(truth).take_while(|(a, b)| a == b).count();
                    cand.tree2 += ok.max(ok2);
                }
            }
            let tried = (cand.tried - tried_before).max(1) as f64;
            this_prompt.push_str(&format!(
                "  {:<10} {:>7.1}% {:>10.2}   top2 {:.1}% top4 {:.1}%{}{}\n",
                cand.name,
                100.0 * (cand.first - first_before) as f64 / tried,
                (cand.run - run_before) as f64 / tried,
                100.0 * (cand.top2 - top2_before) as f64 / tried,
                100.0 * (cand.top4 - top4_before) as f64 / tried,
                if tree2 {
                    format!(" tree2 {:.2}", (cand.tree2 - tree2_before) as f64 / tried)
                } else {
                    String::new()
                },
                if bridge_mode {
                    format!(
                        "   err {:.3}{}",
                        cand.err / cand.errs.max(1) as f64,
                        if nonfinite > 0 {
                            format!(" not-finite {nonfinite}")
                        } else {
                            String::new()
                        }
                    )
                } else {
                    String::new()
                }
            ));
        }
        if history.not_finite > 0 {
            this_prompt.push_str(&format!(
                "  captured positions not finite: {}\n",
                history.not_finite
            ));
        }
        println!(
            "prompt {pi}: {} prompt tokens, {n} reply tokens, {:.1} s",
            ids.len(),
            t0.elapsed().as_secs_f64()
        );
        print!("{this_prompt}");
    }
    if bridge_mode {
        println!(
            "block {block_start}..{} skipped, stand-ins:",
            block_start + skip
        );
    }
    println!("{:<10} {:>8} {:>10}", "skip", "first%", "run/round");
    for cand in &candidates {
        let tried = cand.tried.max(1) as f64;
        println!(
            "{:<10} {:>7.1}% {:>10.2}   top2 {:.1}% top4 {:.1}%{}{}",
            cand.name,
            100.0 * cand.first as f64 / tried,
            cand.run as f64 / tried,
            100.0 * cand.top2 as f64 / tried,
            100.0 * cand.top4 as f64 / tried,
            if tree2 {
                format!(" tree2 {:.2}", cand.tree2 as f64 / tried)
            } else {
                String::new()
            },
            if bridge_mode {
                format!("   err {:.3}", cand.err / cand.errs.max(1) as f64)
            } else {
                String::new()
            }
        );
    }
    Ok(())
}

/// The block's start and effect at every real position so far (bridge mode),
/// `[n, hidden]` f32 on the card: `phi` the hidden state where the block
/// starts, normalised; `delta` what the block added to it.
#[derive(Default)]
struct History {
    phi: Option<Tensor>,
    delta: Option<Tensor>,
    /// `Linear` fits by λ's bits: the history length fitted at, the mean
    /// effect, the map `[hidden, hidden]`.
    fits: std::cell::RefCell<std::collections::HashMap<u64, (usize, Tensor, Tensor)>>,
    eye: std::cell::RefCell<Option<Tensor>>,
    /// The library's stand-in for the same block, fed the same forwards.
    online: Option<std::cell::RefCell<RunStandIn>>,
    /// The same over the model's prior (`Stand::Carried`); fed the prompt by
    /// `start_carried`, every position after it by `push`.
    carried: Option<std::cell::RefCell<RunStandIn>>,
    /// `Stand::CarriedLib`'s: `carried`'s twin with the library's carry, fed
    /// the same prompt and positions; never taken into the prior.
    carried_lib: Option<std::cell::RefCell<RunStandIn>>,
    carried_started: bool,
    /// Captured positions whose `φ` or `Δ` was not finite.
    not_finite: usize,
    /// `Prior` fits by λ's bits (once a prompt) and `Pooled` fits by λ's
    /// bits (the history length fitted at): the mean effect, the map.
    prior_fits: std::cell::RefCell<std::collections::HashMap<u64, (Tensor, Tensor)>>,
    pooled_fits: std::cell::RefCell<std::collections::HashMap<u64, (usize, Tensor, Tensor)>>,
    /// `PriorRows` fits by (λ's bits, rows, prompt): the mean effect, the features
    /// `H` and the weights `A`.
    #[allow(clippy::type_complexity)]
    row_fits:
        std::cell::RefCell<std::collections::HashMap<(u64, usize, bool), (Tensor, Tensor, Tensor)>>,
}

/// What a node would carry from the conversations it served before this one
/// (bridge mode): the block's sufficient statistics over every real position
/// of the earlier prompts — `ΦᵀΦ`, `ΦᵀΔ`, `Σφ`, `ΣΔ` and the count.
struct Carried {
    g: Tensor,
    c: Tensor,
    sum_phi: Tensor,
    sum_delta: Tensor,
    n: usize,
}

impl Carried {
    fn of(phi: &Tensor, delta: &Tensor) -> anyhow::Result<Self> {
        let pt = phi.t()?.contiguous()?;
        Ok(Self {
            g: pt.matmul(&phi.contiguous()?)?,
            c: pt.matmul(&delta.contiguous()?)?,
            sum_phi: phi.sum(0)?,
            sum_delta: delta.sum(0)?,
            n: phi.dim(0)?,
        })
    }

    fn merge(&mut self, more: &Carried) -> anyhow::Result<()> {
        self.g = (&self.g + &more.g)?;
        self.c = (&self.c + &more.c)?;
        self.sum_phi = (&self.sum_phi + &more.sum_phi)?;
        self.sum_delta = (&self.sum_delta + &more.sum_delta)?;
        self.n += more.n;
        Ok(())
    }

    /// The ridge map of these statistics, Δ centred on its mean as `Linear`
    /// centres it: `(G + λI)·W = C − Σφ ⊗ mean(Δ)`. Returns `(mean, W)`.
    fn fit(&self, lambda: f64) -> anyhow::Result<(Tensor, Tensor)> {
        let base = (&self.sum_delta / self.n.max(1) as f64)?;
        let d = base.dim(0)?;
        let outer = self
            .sum_phi
            .reshape((d, 1))?
            .matmul(&base.reshape((1, d))?)?;
        let b = (&self.c - outer)?;
        Ok((base, ridge_by_cg_gram(&self.g, &b, lambda)?))
    }
}

impl History {
    /// `(φ(h_in), Δ)` rows off a capturing forward, `[positions, hidden]`.
    fn rows(
        caught: &std::collections::HashMap<usize, Tensor>,
        start: usize,
        skip: usize,
    ) -> anyhow::Result<(Tensor, Tensor)> {
        let h_in = &caught[&(start - 1)];
        let dev = h_in.device().clone();
        let h_in = h_in.squeeze(0)?.to_dtype(DType::F32)?;
        let h_out = caught[&(start + skip - 1)]
            .to_device(&dev)?
            .squeeze(0)?
            .to_dtype(DType::F32)?;
        let delta = (&h_out - &h_in)?;
        let norm = h_in.sqr()?.sum_keepdim(1)?.sqrt()?.clamp(1e-6, f64::MAX)?;
        Ok((h_in.broadcast_div(&norm)?, delta))
    }

    fn push(
        &mut self,
        caught: &std::collections::HashMap<usize, Tensor>,
        start: usize,
        skip: usize,
    ) -> anyhow::Result<()> {
        let (phi, delta) = Self::rows(caught, start, skip)?;
        let finite = |t: &Tensor| -> anyhow::Result<bool> {
            Ok(t.sum_all()?.to_scalar::<f32>()?.is_finite())
        };
        if !(finite(&phi)? && finite(&delta)?) {
            self.not_finite += phi.dim(0)?;
        }
        if let Some(online) = &self.online {
            online.borrow_mut().learn(caught, phi.dim(0)?)?;
        }
        if let (true, Some(carried)) = (self.carried_started, &self.carried) {
            carried.borrow_mut().learn(caught, phi.dim(0)?)?;
        }
        if let (true, Some(lib)) = (self.carried_started, &self.carried_lib) {
            lib.borrow_mut().learn(caught, phi.dim(0)?)?;
        }
        let join = |old: Option<Tensor>, new: Tensor| -> anyhow::Result<Tensor> {
            Ok(match old {
                Some(o) => Tensor::cat(&[&o, &new], 0)?,
                None => new,
            })
        };
        self.phi = Some(join(self.phi.take(), phi)?);
        self.delta = Some(join(self.delta.take(), delta)?);
        Ok(())
    }

    /// The prompt into `Stand::Carried`'s stand-in as a worker feeds it
    /// (`model_worker::learn_the_prompt`): measured against the model's prior,
    /// refitted with the prompt first where it is longer than the window.
    fn start_carried(
        &mut self,
        caught: &std::collections::HashMap<usize, Tensor>,
        prior: &mut Option<StandInPrior>,
    ) -> anyhow::Result<()> {
        let Some(carried) = &self.carried else {
            return Ok(());
        };
        let mut si = carried.borrow_mut();
        let rows = caught.values().next().map_or(0, |t| t.dim(1).unwrap_or(0));
        let (phi, delta) = si.rows_of(caught, rows)?;
        let prior = prior.get_or_insert_with(|| StandInPrior::for_stand_in(&si));
        let window = si.window();
        let fit = prior.fit_for_prompt(&phi, &delta, window)?;
        if let Some(lib) = &self.carried_lib {
            let mut lib = lib.borrow_mut();
            lib.set_prior(fit.clone())?;
            lib.learn_rows(phi.clone(), delta.clone())?;
        }
        si.set_prior(fit)?;
        si.learn_rows(phi, delta)?;
        self.carried_started = true;
        Ok(())
    }

    /// The bridge a stand-in makes of the history; `None` for `none`.
    fn bridge(
        &self,
        stand: Stand,
        run_end: usize,
        window: usize,
        carried: Option<&Carried>,
        earlier: &[(Tensor, Tensor)],
    ) -> anyhow::Result<Option<SkipBridge>> {
        let (Some(phi), Some(delta)) = (&self.phi, &self.delta) else {
            return Ok(None);
        };
        let n = delta.dim(0)?;
        let mean = || -> anyhow::Result<Tensor> { Ok(delta.mean(0)?) };
        Ok(match stand {
            Stand::None => None,
            Stand::Mean => Some(SkipBridge::Shift {
                run_end,
                shift: mean()?,
            }),
            Stand::Prev => Some(SkipBridge::Shift {
                run_end,
                shift: delta.get(n - 1)?,
            }),
            Stand::Recall(lambda) => {
                let from = n.saturating_sub(window);
                let m = n - from;
                let f = phi.narrow(0, from, m)?.contiguous()?;
                let base = mean()?;
                let r = delta.narrow(0, from, m)?.broadcast_sub(&base)?;
                let k: Vec<Vec<f32>> = f.matmul(&f.t()?.contiguous()?)?.to_vec2()?;
                let inv = invert_with_ridge(&k, lambda);
                let inv = Tensor::from_vec(inv, (m, m), f.device())?;
                Some(SkipBridge::Recall {
                    run_end,
                    prior: None,
                    base,
                    features: f,
                    weights: inv.matmul(&r.contiguous()?)?,
                })
            }
            Stand::Carried => match &self.carried {
                Some(carried) => {
                    let mut carried = carried.borrow_mut();
                    carried.refresh()?;
                    carried.bridge().cloned()
                }
                None => None,
            },
            Stand::CarriedLib => match &self.carried_lib {
                Some(lib) => {
                    let mut lib = lib.borrow_mut();
                    lib.refresh()?;
                    lib.bridge().cloned()
                }
                None => None,
            },
            Stand::Online => match &self.online {
                Some(online) => {
                    let mut online = online.borrow_mut();
                    online.refresh()?;
                    online.bridge().cloned()
                }
                None => None,
            },
            Stand::Prior(lambda) => {
                let Some(carried) = carried else {
                    return Ok(None);
                };
                let (base, w) = self.prior_fit(carried, lambda)?;
                Some(self.as_map(run_end, &base, &w)?)
            }
            Stand::PriorRecall(lambda) => {
                let Some(carried) = carried else {
                    return self.bridge(Stand::Recall(lambda), run_end, window, None, earlier);
                };
                let (pbase, w) = self.prior_fit(carried, lambda)?;
                let from = n.saturating_sub(window);
                let m = n - from;
                let f = phi.narrow(0, from, m)?.contiguous()?;
                // What the prior leaves of each effect in the window.
                let left = (delta.narrow(0, from, m)? - f.matmul(&w)?.broadcast_add(&pbase)?)?;
                let rbase = left.mean(0)?;
                let r = left.broadcast_sub(&rbase)?;
                let k: Vec<Vec<f32>> = f.matmul(&f.t()?.contiguous()?)?.to_vec2()?;
                let inv = invert_with_ridge(&k, lambda);
                let alpha = Tensor::from_vec(inv, (m, m), f.device())?.matmul(&r.contiguous()?)?;
                let mut eye = self.eye.borrow_mut();
                if eye.is_none() {
                    *eye = Some(Tensor::eye(w.dim(0)?, DType::F32, w.device())?);
                }
                // `φ·[I; F]ᵀ·[W; α] = φ·W + k(φ, F)·α`: one `Recall` bridge.
                Some(SkipBridge::Recall {
                    run_end,
                    prior: None,
                    base: (pbase + rbase)?,
                    features: Tensor::cat(&[eye.as_ref().expect("set above"), &f], 0)?,
                    weights: Tensor::cat(&[&w, &alpha], 0)?,
                })
            }
            Stand::PriorRows(lambda, most, with_prompt) => {
                if earlier.is_empty() && !with_prompt {
                    return self.bridge(Stand::Recall(lambda), run_end, window, None, earlier);
                }
                // The first call of a prompt comes after its pass: the
                // history is then the prompt's positions alone.
                let this = with_prompt.then(|| (phi.clone(), delta.clone()));
                let (pbase, h, a) = self.row_fit(earlier, this, lambda, most)?;
                let from = n.saturating_sub(window);
                let m = n - from;
                let f = phi.narrow(0, from, m)?.contiguous()?;
                let prior = f.matmul(&h.t()?)?.matmul(&a)?.broadcast_add(&pbase)?;
                let left = (delta.narrow(0, from, m)? - prior)?;
                let rbase = left.mean(0)?;
                let r = left.broadcast_sub(&rbase)?;
                let k: Vec<Vec<f32>> = f.matmul(&f.t()?.contiguous()?)?.to_vec2()?;
                let inv = invert_with_ridge(&k, lambda);
                let alpha = Tensor::from_vec(inv, (m, m), f.device())?.matmul(&r.contiguous()?)?;
                // `φ·[H; F]ᵀ·[A; α]`: the prior's rows and the window's, one bridge.
                Some(SkipBridge::Recall {
                    run_end,
                    prior: None,
                    base: (pbase + rbase)?,
                    features: Tensor::cat(&[&h, &f], 0)?,
                    weights: Tensor::cat(&[&a, &alpha], 0)?,
                })
            }
            Stand::Pooled(lambda) => {
                let refit = env_usize("PROBE_REFIT", 16).max(1);
                let mut fits = self.pooled_fits.borrow_mut();
                let key = lambda.to_bits();
                if fits.get(&key).is_none_or(|(at, ..)| n >= at + refit) {
                    let mut all = Carried::of(phi, delta)?;
                    if let Some(carried) = carried {
                        all.merge(carried)?;
                    }
                    let (base, w) = all.fit(lambda)?;
                    fits.insert(key, (n, base, w));
                }
                let (_, base, w) = &fits[&key];
                Some(self.as_map(run_end, base, w)?)
            }
            Stand::Linear(lambda) => {
                let refit = env_usize("PROBE_REFIT", 16).max(1);
                let mut fits = self.fits.borrow_mut();
                let key = lambda.to_bits();
                if fits.get(&key).is_none_or(|(at, ..)| n >= at + refit) {
                    let base = mean()?;
                    let r = delta.broadcast_sub(&base)?;
                    let w = ridge_by_cg(phi, &r, lambda)?;
                    fits.insert(key, (n, base, w));
                }
                let (_, base, w) = &fits[&key];
                let mut eye = self.eye.borrow_mut();
                if eye.is_none() {
                    *eye = Some(Tensor::eye(w.dim(0)?, DType::F32, w.device())?);
                }
                // `φ·Iᵀ·W + base`: the map as `Recall`'s arithmetic.
                Some(SkipBridge::Recall {
                    run_end,
                    prior: None,
                    base: base.clone(),
                    features: eye.clone().expect("set above"),
                    weights: w.clone(),
                })
            }
        })
    }

    /// `Prior`'s fit for this prompt: once, by λ's bits.
    fn prior_fit(&self, carried: &Carried, lambda: f64) -> anyhow::Result<(Tensor, Tensor)> {
        let mut fits = self.prior_fits.borrow_mut();
        let key = lambda.to_bits();
        if let std::collections::hash_map::Entry::Vacant(slot) = fits.entry(key) {
            slot.insert(carried.fit(lambda)?);
        }
        Ok(fits[&key].clone())
    }

    /// `PriorRows`'s fit for this prompt, once: kernel ridge over the last
    /// `most` earlier positions, `A = (H·Hᵀ + λI)⁻¹·(Δ − mean)`. Returns
    /// `(mean, H, A)`.
    fn row_fit(
        &self,
        earlier: &[(Tensor, Tensor)],
        this: Option<(Tensor, Tensor)>,
        lambda: f64,
        most: usize,
    ) -> anyhow::Result<(Tensor, Tensor, Tensor)> {
        let mut fits = self.row_fits.borrow_mut();
        let key = (lambda.to_bits(), most, this.is_some());
        if let std::collections::hash_map::Entry::Vacant(slot) = fits.entry(key) {
            let rows: Vec<&(Tensor, Tensor)> = earlier.iter().chain(this.as_ref()).collect();
            let phis: Vec<&Tensor> = rows.iter().map(|(p, _)| p).collect();
            let deltas: Vec<&Tensor> = rows.iter().map(|(_, d)| d).collect();
            let (h, d) = (Tensor::cat(&phis, 0)?, Tensor::cat(&deltas, 0)?);
            let all = h.dim(0)?;
            let from = all.saturating_sub(most);
            let (h, d) = (
                h.narrow(0, from, all - from)?.contiguous()?,
                d.narrow(0, from, all - from)?.contiguous()?,
            );
            let base = d.mean(0)?;
            let r = d.broadcast_sub(&base)?;
            let k: Vec<Vec<f32>> = h.matmul(&h.t()?.contiguous()?)?.to_vec2()?;
            let inv = invert_with_ridge(&k, lambda);
            let m = all - from;
            let a = Tensor::from_vec(inv, (m, m), h.device())?.matmul(&r.contiguous()?)?;
            slot.insert((base, h, a));
        }
        Ok(fits[&key].clone())
    }

    /// A hidden × hidden map `W` and mean effect as a bridge: `φ·Iᵀ·W + base`,
    /// `Recall`'s arithmetic.
    fn as_map(&self, run_end: usize, base: &Tensor, w: &Tensor) -> anyhow::Result<SkipBridge> {
        let mut eye = self.eye.borrow_mut();
        if eye.is_none() {
            *eye = Some(Tensor::eye(w.dim(0)?, DType::F32, w.device())?);
        }
        Ok(SkipBridge::Recall {
            run_end,
            prior: None,
            base: base.clone(),
            features: eye.clone().expect("set above"),
            weights: w.clone(),
        })
    }

    /// What `bridge` estimates the block adds at positions whose normalised
    /// start is `phi` — the arithmetic `SkipBridge` applies, on `phi`'s rows.
    fn estimate(bridge: &SkipBridge, phi: &Tensor) -> anyhow::Result<Tensor> {
        Ok(match bridge {
            SkipBridge::Shift { shift, .. } => phi.zeros_like()?.broadcast_add(shift)?,
            SkipBridge::Recall {
                base,
                features,
                weights,
                prior,
                ..
            } => {
                let mut e = phi.matmul(&features.t()?.contiguous()?)?.matmul(weights)?;
                if let Some((h, a)) = prior {
                    e = (e + phi.matmul(&h.t()?.contiguous()?)?.matmul(a)?)?;
                }
                e.broadcast_add(base)?
            }
        })
    }
}

/// `bridge` with `cand.carry` times the mean of the candidate's last
/// residuals added to what it adds at every position (`<stand>~ρxk`).
fn carried_forward(
    bridge: Option<SkipBridge>,
    cand: &Candidate,
) -> anyhow::Result<Option<SkipBridge>> {
    if cand.carry == 0.0 || cand.resids.is_empty() {
        return Ok(bridge);
    }
    let rows: Vec<Tensor> = cand.resids.iter().cloned().collect();
    let shift = (Tensor::stack(&rows, 0)?.mean(0)? * cand.carry)?;
    Ok(match bridge {
        None => None,
        Some(SkipBridge::Shift { run_end, shift: s }) => Some(SkipBridge::Shift {
            run_end,
            shift: (s + &shift)?,
        }),
        Some(SkipBridge::Recall {
            run_end,
            base,
            features,
            weights,
            prior,
        }) => Some(SkipBridge::Recall {
            run_end,
            base: (base + &shift)?,
            features,
            weights,
            prior,
        }),
    })
}

/// The ridge map `W` (`[hidden, hidden]`) minimising `‖Φ·W − R‖² + λ‖W‖²`
/// over every history row: `(ΦᵀΦ + λI)·W = ΦᵀR`, solved by conjugate
/// gradients column by column on the rows' device (`PROBE_CG_ITERS`, 60;
/// stopping early regularises further). Primal, so its cost is the hidden
/// width's, whatever the history's length.
fn ridge_by_cg(phi: &Tensor, r: &Tensor, lambda: f64) -> anyhow::Result<Tensor> {
    let pt = phi.t()?.contiguous()?;
    ridge_by_cg_gram(
        &pt.matmul(&phi.contiguous()?)?,
        &pt.matmul(&r.contiguous()?)?,
        lambda,
    )
}

/// `ridge_by_cg` from its statistics: `(G + λI)·W = B`.
fn ridge_by_cg_gram(g: &Tensor, b: &Tensor, lambda: f64) -> anyhow::Result<Tensor> {
    let apply = |p: &Tensor| -> candle_core::Result<Tensor> { g.matmul(p)? + (p * lambda)? };
    let mut w = b.zeros_like()?;
    let mut res = b.clone();
    let mut p = res.clone();
    let mut rs = res.sqr()?.sum_keepdim(0)?;
    for _ in 0..env_usize("PROBE_CG_ITERS", 60) {
        let ap = apply(&p)?;
        let alpha = rs.broadcast_div(&p.mul(&ap)?.sum_keepdim(0)?.clamp(1e-20, f64::MAX)?)?;
        w = (w + p.broadcast_mul(&alpha)?)?;
        res = (res - ap.broadcast_mul(&alpha)?)?;
        let rs_new = res.sqr()?.sum_keepdim(0)?;
        let beta = rs_new.broadcast_div(&rs.clamp(1e-30, f64::MAX)?)?;
        p = (&res + p.broadcast_mul(&beta)?)?;
        rs = rs_new;
    }
    Ok(w)
}

/// `(K + λI)⁻¹` by Gauss-Jordan in f64, row-major f32 out. `K` is a Gram
/// matrix of unit vectors, so with λ > 0 it is well conditioned.
fn invert_with_ridge(k: &[Vec<f32>], lambda: f64) -> Vec<f32> {
    let m = k.len();
    let mut a: Vec<Vec<f64>> = (0..m)
        .map(|i| {
            let mut row: Vec<f64> = k[i].iter().map(|&v| f64::from(v)).collect();
            row[i] += lambda;
            row.extend((0..m).map(|j| if i == j { 1.0 } else { 0.0 }));
            row
        })
        .collect();
    for col in 0..m {
        let pivot = (col..m)
            .max_by(|&x, &y| a[x][col].abs().total_cmp(&a[y][col].abs()))
            .unwrap_or(col);
        a.swap(col, pivot);
        let d = a[col][col];
        for v in a[col].iter_mut() {
            *v /= d;
        }
        let pivot_row = a[col].clone();
        for (r, row) in a.iter_mut().enumerate() {
            let f = row[col];
            if r != col && f != 0.0 {
                for (v, p) in row.iter_mut().zip(&pivot_row) {
                    *v -= f * p;
                }
            }
        }
    }
    a.iter()
        .flat_map(|row| row[m..].iter().map(|&v| v as f32))
        .collect()
}
