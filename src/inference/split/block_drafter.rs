//! A block drafter (DFlash, #428): a small model trained beside its target
//! that guesses a whole block of tokens in ONE forward.
//!
//! A chain drafter (a small model, the file's own guess head) makes one guess
//! a forward, each reading its own output, so its guesses are kept less often
//! the deeper they go and its time grows with their number. A block drafter
//! reads the target's own hidden states — several of its layers' outputs at
//! every committed position — and fills a block of mask tokens after the
//! last token at once: one forward, every guess conditioned on what the
//! target computed, and the check is the ordinary chain check
//! (`model_worker::ngram_spec_round`). Its guesses are kept 5.4-5.8 a check
//! at a block of 8 on Qwen3.5-9B (z-lab's card, greedy, maths and code).
//!
//! Prior art: DFlash (arXiv 2602.06036, z-lab), its reference
//! `dflash/model.py` (github.com/z-lab/dflash, read at `95c8aeca`,
//! 2026-08-18) and llama.cpp's `draft-dflash` (`src/models/dflash.cpp`,
//! `common/speculative.cpp`, `conversion/qwen.py` `DFlashModel`, read at
//! `~/llama.cpp-ref` `4b1a27f`). The shape, as both run it:
//!
//! - context rows: the target's outputs of layers `target_layer_ids`
//!   (`hidden_states[id + 1]`; llama.cpp stores `id + 1` as
//!   `dflash.target_layers` and reads the INPUT of that layer), concatenated
//!   in that order, through `fc` and `hidden_norm` (`enc.output_norm`);
//! - a block `[last token, mask × (B − 1)]` at the next positions, embedded
//!   with the TARGET's token table;
//! - each drafter layer: queries from the block alone; keys and values from
//!   the context rows AND the block (the context rows are NOT normed by the
//!   layer's input norm), `k_norm` on the keys, NEOX RoPE; the context rows'
//!   keys and values stay in the drafter's cache, the block's are cut away
//!   after the draft (z-lab's `_crop_to(start)`);
//! - the output norm, then the TARGET's output head; row `i ≥ 1` guesses the
//!   token AT position `start + i`.
//!
//! What a layer sees is the one place the two references part:
//! llama.cpp attends both ways everywhere unless the file writes
//! `dflash.attention.causal`; z-lab's reference, since its DFlash 2 commit
//! (`c2394722`, 2026-08-17), makes a `sliding_attention` layer one-way when
//! the config names no `is_causal` — and Qwen3.5-9B-DFlash's names none.
//! [`attention_rule`] reads the file first, then takes z-lab's: on ONE fixed
//! reply of Qwen3.5-9B (81 positions each, 2026-11-08) its drafter kept 6.47
//! of 15 a block on code and 1.44 on a story, against 5.98 and 1.30 both
//! ways (`a_block_drafters_acceptance_over_a_fixed_reply`).
//!
//! Ours differs in one place: the context rows of a long prompt are its last
//! [`BLOCK_PROMPT_ROWS`] only (the references take every row), so a long
//! agent prompt costs one bounded catch-up after the first token.

use std::path::Path;

use candle_core::quantized::gguf_file;
use candle_core::{DType, Device, Tensor};
use candle_nn::Module;

use super::kv_cache::{KvStorage, LayerKv};
use super::model::SplitModel;
use super::QMatMul;
use crate::error::SwarmError;
use crate::inference::residual_norm::RmsNorm;

/// The drafter's file, beside the model's parts in its directory.
pub(crate) const BLOCK_DRAFTER_FILE: &str = "block_drafter.gguf";

/// The most prompt positions the drafter takes in before its first block:
/// the prompt's last ones (the head's [`super::MTP_PROMPT_ROWS`] reasoning;
/// eight layers' rows are kept for each, so half as many).
pub(crate) const BLOCK_PROMPT_ROWS: usize = 2_048;

/// The most positions one request's drafter cache holds. Past it the drafter
/// stops for that reply. Its whole cache at this length is charged when the
/// drafter is attached ([`SplitModel::attach_block_drafter`]).
pub(crate) const BLOCK_MOST_POSITIONS: usize = 8_192;

/// Context rows taken in per forward when the drafter catches up.
pub(crate) const BLOCK_CATCH_UP_CHUNK: usize = 512;

/// What a drafter layer's block rows see of the block (the context rows are
/// seen whole by every row, inside a sliding layer's window).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum AttentionRule {
    /// Every layer both ways — llama.cpp's default.
    BothWays,
    /// Sliding layers one way, full layers both ways — z-lab's reference
    /// since 2026-08-17 for a config naming no `is_causal`, and ours.
    SlidingOneWay,
    /// Every layer one way — a file writing `dflash.attention.causal = true`.
    OneWay,
}

/// The rule a drafter's layers attend by: the file's `dflash.attention.causal`
/// where it writes one, else z-lab's (sliding layers one way) —
/// `SWARMLLM_BLOCK_DRAFTER_CAUSAL=none` attends both ways as llama.cpp does,
/// for A/B inside one binary.
pub(crate) fn attention_rule(file_says: Option<bool>) -> AttentionRule {
    match file_says {
        Some(true) => AttentionRule::OneWay,
        Some(false) => AttentionRule::BothWays,
        None => match std::env::var("SWARMLLM_BLOCK_DRAFTER_CAUSAL").as_deref() {
            Ok("none") => AttentionRule::BothWays,
            _ => AttentionRule::SlidingOneWay,
        },
    }
}

/// `SWARMLLM_BLOCK_DRAFTER=0`: no block drafter is attached, for A/B inside
/// one binary (a Qwen 3.5 file's own guess head guesses instead).
pub(crate) fn block_drafter_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("SWARMLLM_BLOCK_DRAFTER").as_deref() != Ok("0"))
}

/// A drafter its card/processor split's card has no room for goes to system
/// memory ([`drafter_home`], #451) — ON since measured (2026-11-14, Qwen3.5-9B
/// at 25 of 32 layers on an RTX 3070, 12 runs a prompt: code 30.4 against 20.9
/// tok/s at the median, prose and a story no slower). `SWARMLLM_BLOCK_DRAFTER_IN_RAM=0`
/// leaves it off, for A/B inside one binary.
fn block_drafter_in_system_memory_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("SWARMLLM_BLOCK_DRAFTER_IN_RAM").as_deref() != Ok("0"))
}

/// `SWARMLLM_BLOCK_DRAFTER=processor`: attach the drafter beside a whole
/// model on the processor even where its own guess head guesses
/// ([`SplitModel::head_guesses_better_here`]), for A/B inside one binary.
pub(crate) fn block_drafter_on_the_processor_anyway() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("SWARMLLM_BLOCK_DRAFTER").as_deref() == Ok("processor"))
}

struct DrafterLayer {
    attn_norm: RmsNorm,
    wq: QMatMul,
    wk: QMatMul,
    wv: QMatMul,
    wo: QMatMul,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    ffn_norm: RmsNorm,
    gate: QMatMul,
    up: QMatMul,
    down: QMatMul,
    /// Reads only the last `window` positions (`sliding_attention`).
    windowed: bool,
    /// A block row sees no later block row.
    one_way: bool,
}

/// A loaded block drafter, attached to the whole model it guesses for.
pub(crate) struct BlockDrafter {
    fc: QMatMul,
    hidden_norm: RmsNorm,
    layers: Vec<DrafterLayer>,
    norm: RmsNorm,
    /// The target layers (absolute) whose outputs make a context row, in the
    /// order `fc` reads them.
    target_layers: Vec<usize>,
    block_size: usize,
    mask_token: u32,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    cos: Tensor,
    sin: Tensor,
    window: Option<usize>,
    rule: AttentionRule,
    device: Device,
    /// Weights and a full cache, charged when attached.
    pub(crate) bytes: u64,
}

/// One request's drafter cache: each layer's keys and values of the context
/// rows taken in. Its positions count from the first row taken in — RoPE is
/// relative and the mask reads only distances, so a drafter starting
/// mid-prompt attends as one started at 0 would over the same rows.
#[derive(Default)]
pub(crate) struct BlockCache {
    kv: Vec<Option<LayerKv>>,
    /// Context rows held — the next block's first position.
    len: usize,
}

/// The additive mask of one layer's block rows, `rows` queries from position
/// `start` over the keys `0..start + rows` — `None` where every key is seen.
/// A row at `p` sees a key at `j` where (one-way) `j ≤ p`, and (windowed)
/// `|p − j| < window`: z-lab's `_attention_mask` with `query_position =
/// key_len − q_len + i` — a key AHEAD of the row (a both-ways block) is held
/// to the window too, which binds only where the block is longer than it.
pub(crate) fn block_mask(
    rows: usize,
    start: usize,
    one_way: bool,
    window: Option<usize>,
    device: &Device,
) -> candle_core::Result<Option<Tensor>> {
    let keys = start + rows;
    let window_binds = window.is_some_and(|w| start + rows > w);
    if !one_way && !window_binds {
        return Ok(None);
    }
    let mask: Vec<f32> = (0..rows)
        .flat_map(|i| {
            let p = start + i;
            (0..keys).map(move |j| {
                let seen = (!one_way || j <= p) && window.is_none_or(|w| p.abs_diff(j) < w);
                if seen {
                    0.0
                } else {
                    f32::NEG_INFINITY
                }
            })
        })
        .collect();
    Tensor::from_slice(&mask, (rows, keys), device).map(Some)
}

fn meta_u32(v: &gguf_file::Value) -> Option<u32> {
    v.to_u32()
        .ok()
        .or_else(|| v.to_i32().ok().and_then(|x| u32::try_from(x).ok()))
        .or_else(|| v.to_u64().ok().and_then(|x| u32::try_from(x).ok()))
}

/// The model's layers a drafter reads, off its file: `dflash.target_layers`
/// stores each as its id + 1 (llama.cpp reads that layer's INPUT, the output
/// of the id). `None` where the key is missing or names no layer.
fn target_layers_in(ct: &gguf_file::Content) -> Option<Vec<usize>> {
    ct.metadata
        .get("dflash.target_layers")?
        .to_vec()
        .ok()?
        .iter()
        .map(|v| meta_u32(v).and_then(|id| (id as usize).checked_sub(1)))
        .collect()
}

/// What a split's coordinator needs of the drafter beside its first segment
/// before it asks the segments for rows (#428 stage 3), read off the file's
/// header ([`plan_in`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct BlockPlan {
    /// The model's layers whose outputs the drafter reads.
    pub target_layers: Vec<usize>,
    /// Its block: the next token and as many guesses less one.
    pub block_size: usize,
}

/// [`BlockPlan`] of the drafter in `dir` ([`BLOCK_DRAFTER_FILE`]); `None`
/// where there is none, it is switched off ([`block_drafter_on`]) or its
/// header does not name its layers. Whether the worker attached it is the
/// worker's answer, at the first draft.
pub(crate) fn plan_in(dir: &Path) -> Option<BlockPlan> {
    if !block_drafter_on() {
        return None;
    }
    let file = std::fs::File::open(dir.join(BLOCK_DRAFTER_FILE)).ok()?;
    let ct = gguf_file::Content::read(&mut std::io::BufReader::new(file)).ok()?;
    let target_layers = target_layers_in(&ct).filter(|l| !l.is_empty())?;
    let block_size = ct
        .metadata
        .get("dflash.block_size")
        .and_then(meta_u32)
        .unwrap_or(16) as usize;
    Some(BlockPlan {
        target_layers,
        block_size,
    })
}

impl BlockDrafter {
    /// Load `path` for `target`, refusing a file it cannot run as written:
    /// not a `dflash` file, a part this loader does not build (DFlash 2's
    /// convolutions and selector, a draft vocabulary, scaled embeddings or
    /// logits, M-RoPE sections that are not plain), or one that does not fit
    /// the target (its width, its layers, its token table). It runs on the
    /// target's own device — the card where the model's first layers are, as
    /// llama.cpp loads a DFlash drafter on the target's devices.
    pub(crate) fn load(
        path: &Path,
        target: &SplitModel,
        device: &Device,
    ) -> Result<Self, SwarmError> {
        let fail = |what: String| {
            SwarmError::Internal(format!("block drafter {}: {what}", path.display()))
        };
        // The block is embedded by the model's token table and read by its
        // head: a whole model, or a first segment beside the head's part
        // (`SplitModel::drafter_head`), whose other layers' rows its peers
        // send (#428 stage 3).
        if target.layer_start != 0
            || target.tok_embeddings.is_none()
            || (target.output.is_none() && target.drafter_head.is_none())
        {
            return Err(fail(
                "the model's token table and output head are not both here".into(),
            ));
        }
        let file = std::fs::File::open(path).map_err(|e| fail(e.to_string()))?;
        let mut file = std::io::BufReader::new(file);
        let ct = gguf_file::Content::read(&mut file).map_err(|e| fail(e.to_string()))?;
        let md = |key: &str| ct.metadata.get(key);
        let arch = md("general.architecture")
            .and_then(|v| v.to_string().ok())
            .cloned()
            .unwrap_or_default();
        if arch != "dflash" {
            return Err(fail(format!("architecture {arch:?}, not \"dflash\"")));
        }
        let num = |key: &str| md(&format!("dflash.{key}")).and_then(meta_u32);
        let shape = DrafterShape::read(&ct).map_err(fail)?;
        let DrafterShape {
            hidden,
            n_layers,
            n_head,
            n_kv_head,
            head_dim,
        } = shape;
        let block_size = num("block_size").unwrap_or(16) as usize;
        let eps = md("dflash.attention.layer_norm_rms_epsilon")
            .and_then(|v| v.to_f32().ok())
            .unwrap_or(1e-6) as f64;
        let theta = md("dflash.rope.freq_base")
            .and_then(|v| v.to_f32().ok())
            .unwrap_or(10_000.0);
        let mask_token = md("tokenizer.ggml.mask_token_id")
            .and_then(meta_u32)
            .ok_or_else(|| fail("no tokenizer.ggml.mask_token_id".into()))?;
        let target_layers = target_layers_in(&ct)
            .ok_or_else(|| fail("no dflash.target_layers, or one that is not a layer".into()))?;
        // What this loader does not build is refused by name, never skipped.
        for (key, plain) in [
            ("dflash.embedding_scale", 1.0f32),
            ("dflash.logit_scale", 1.0),
            ("dflash.final_logit_softcapping", 0.0),
        ] {
            if let Some(v) = md(key).and_then(|v| v.to_f32().ok()) {
                if v != plain {
                    return Err(fail(format!("{key} = {v} is not built")));
                }
            }
        }
        if let Some(sections) = md("dflash.rope.dimension_sections").and_then(|v| v.to_vec().ok()) {
            let s: Vec<u32> = sections.iter().filter_map(meta_u32).collect();
            if s.first().copied() != Some((head_dim / 2) as u32)
                || s.iter().skip(1).any(|&x| x != 0)
            {
                return Err(fail(format!("rope sections {s:?} are not plain")));
            }
        }
        if let Some(name) = ct.tensor_infos.keys().find(|name| {
            [
                "conv_",
                "selector",
                "d2t",
                "markov",
                "conf_proj",
                "hc_",
                "attn_sinks",
                "_exps",
                "attn_q_a",
                "attn_kv",
            ]
            .iter()
            .any(|part| name.contains(part))
        }) {
            return Err(fail(format!("tensor {name} is not built")));
        }
        if hidden != target.hidden_dim {
            return Err(fail(format!(
                "width {hidden}, the model's is {}",
                target.hidden_dim
            )));
        }
        if target_layers.is_empty() || target_layers.iter().any(|&l| l >= target.total_layers) {
            return Err(fail(format!(
                "target layers {target_layers:?} of a {}-layer model",
                target.total_layers
            )));
        }
        if target
            .embedding_rows()
            .is_none_or(|rows| mask_token as usize >= rows)
        {
            return Err(fail(format!(
                "mask token {mask_token} is past the token table"
            )));
        }
        if block_size < 2 || n_head == 0 || n_kv_head == 0 || !n_head.is_multiple_of(n_kv_head) {
            return Err(fail(format!(
                "block {block_size}, heads {n_head}/{n_kv_head}"
            )));
        }
        let window = md("dflash.attention.sliding_window")
            .and_then(meta_u32)
            .map(|w| w as usize)
            .filter(|&w| w > 0);
        let pattern: Vec<bool> = md("dflash.attention.sliding_window_pattern")
            .and_then(|v| v.to_vec().ok())
            .map(|v| v.iter().map(|b| b.to_bool().unwrap_or(false)).collect())
            .unwrap_or_default();
        let rule = attention_rule(md("dflash.attention.causal").and_then(|v| v.to_bool().ok()));

        // The model's device, or system memory beside a card that has no
        // room for it ([`drafter_home`]).
        let device = device.clone();
        // What it is charged: the one figure the rule before the load weighed.
        let bytes = charged(&ct, &shape);
        let mut tensor = |name: String| {
            if !ct.tensor_infos.contains_key(&name) {
                return Err(fail(format!("no {name}")));
            }
            ct.tensor(&mut file, &name, &device)
                .map_err(|e| fail(format!("{name}: {e}")))
        };
        let mm = |t: candle_core::quantized::QTensor| {
            QMatMul::from_qtensor(t).map_err(SwarmError::internal)
        };
        let norm = |t: candle_core::quantized::QTensor| {
            RmsNorm::from_qtensor(t, eps).map_err(SwarmError::internal)
        };
        let fc_weight = tensor("fc.weight".into())?;
        let fc_in = fc_weight.shape().dims().get(1).copied().unwrap_or(0);
        if fc_in != target_layers.len() * hidden {
            return Err(fail(format!(
                "fc reads {fc_in} values, {} layers of {hidden} are {}",
                target_layers.len(),
                target_layers.len() * hidden
            )));
        }
        let fc = mm(fc_weight)?;
        let hidden_norm = norm(tensor("enc.output_norm.weight".into())?)?;
        let out_norm = norm(tensor("output_norm.weight".into())?)?;
        let mut layers = Vec::with_capacity(n_layers);
        for l in 0..n_layers {
            let p = |what: &str| format!("blk.{l}.{what}.weight");
            let windowed = window.is_some() && pattern.get(l).copied().unwrap_or(false);
            layers.push(DrafterLayer {
                attn_norm: norm(tensor(p("attn_norm"))?)?,
                wq: mm(tensor(p("attn_q"))?)?,
                wk: mm(tensor(p("attn_k"))?)?,
                wv: mm(tensor(p("attn_v"))?)?,
                wo: mm(tensor(p("attn_output"))?)?,
                q_norm: norm(tensor(p("attn_q_norm"))?)?,
                k_norm: norm(tensor(p("attn_k_norm"))?)?,
                ffn_norm: norm(tensor(p("ffn_norm"))?)?,
                gate: mm(tensor(p("ffn_gate"))?)?,
                up: mm(tensor(p("ffn_up"))?)?,
                down: mm(tensor(p("ffn_down"))?)?,
                windowed,
                one_way: match rule {
                    AttentionRule::BothWays => false,
                    AttentionRule::SlidingOneWay => windowed,
                    AttentionRule::OneWay => true,
                },
            });
        }
        let (cos, sin) = super::rope::precompute_freqs_cis(
            head_dim,
            theta,
            BLOCK_MOST_POSITIONS + block_size,
            &device,
        )
        .map_err(|e| fail(format!("rope tables: {e}")))?;
        Ok(Self {
            fc,
            hidden_norm,
            layers,
            norm: out_norm,
            target_layers,
            block_size,
            mask_token,
            n_head,
            n_kv_head,
            head_dim,
            cos,
            sin,
            window,
            rule,
            device,
            bytes,
        })
    }

    /// The target layers (absolute) a forward keeps for the drafter.
    pub(crate) fn target_layers(&self) -> &[usize] {
        &self.target_layers
    }

    /// The block the drafter was trained on: the most guesses a draft makes
    /// is one less.
    pub(crate) fn block_size(&self) -> usize {
        self.block_size
    }

    /// Context rows (`[1, n, hidden]`) from the target's kept outputs — one
    /// `[1, n, hidden]` per [`Self::target_layers`] entry, in that order.
    pub(crate) fn context_rows(&self, kept: &[Tensor]) -> Result<Tensor, SwarmError> {
        let err = |what: &'static str| {
            move |e: candle_core::Error| SwarmError::Internal(format!("block drafter {what}: {e}"))
        };
        let on: Vec<Tensor> = kept
            .iter()
            .map(|t| {
                t.to_device(&self.device)
                    .and_then(|t| t.to_dtype(DType::F32))
            })
            .collect::<Result<_, _>>()
            .map_err(err("context to its device"))?;
        let joined = Tensor::cat(&on, 2).map_err(err("context rows"))?;
        let x = self.fc.forward(&joined).map_err(err("fc"))?;
        self.hidden_norm.forward(&x).map_err(err("hidden norm"))
    }

    /// One layer over the block `x` (`[1, rows, hidden]`), its context rows
    /// `ctx` taken in at the cache's next positions `base..`.
    #[allow(clippy::too_many_arguments)]
    fn layer_forward(
        &self,
        layer: &DrafterLayer,
        x: &Tensor,
        ctx: Option<&Tensor>,
        base: usize,
        kv: &mut Option<LayerKv>,
        kv_reserve: usize,
    ) -> candle_core::Result<Tensor> {
        let (_, rows, _) = x.dims3()?;
        let (h, kh, d) = (self.n_head, self.n_kv_head, self.head_dim);
        let normed = layer.attn_norm.forward(x)?;
        // The context rows feed keys and values unnormed by this layer.
        let kv_in = match ctx {
            Some(c) => Tensor::cat(&[c, &normed], 1)?,
            None => normed.clone(),
        };
        let (n_in, k, v) = self.keys_and_values(layer, &kv_in, base)?;
        let start = base + n_in - rows;
        let q = layer.wq.forward(&normed)?.reshape((1, rows, h, d))?;
        let q = layer.q_norm.forward(&q)?.transpose(1, 2)?.contiguous()?;
        let q = self.rope(&q, start, rows)?;
        let (k_all, v_all) = crate::inference::layers::append_to_cache(
            kv,
            &k,
            &v,
            base,
            KvStorage::F32,
            BLOCK_MOST_POSITIONS,
            kv_reserve,
            None,
        )?;
        let mask = block_mask(
            rows,
            start,
            layer.one_way,
            self.window.filter(|_| layer.windowed),
            &self.device,
        )?;
        let y = crate::inference::layers::matmul_attention(
            &q,
            &k_all,
            &v_all,
            mask.as_ref(),
            d,
            h,
            kh,
            None,
            None,
        )?;
        let y = y.transpose(1, 2)?.reshape((1, rows, h * d))?;
        let x = (x + layer.wo.forward(&y)?)?;
        let normed = layer.ffn_norm.forward(&x)?;
        let ff = layer.down.forward(&crate::inference::fast_math::silu_mul(
            &layer.gate.forward(&normed)?,
            &layer.up.forward(&normed)?,
        )?)?;
        x + ff
    }

    /// Take in `context` alone (`[1, n, hidden]`) — a long catch-up's earlier
    /// pieces: each layer's keys and values of those rows, no block. A
    /// context row's keys depend on nothing else, so they are the ones a
    /// draft would have made of them.
    pub(crate) fn take_in(
        &self,
        cache: &mut BlockCache,
        context: &Tensor,
        kv_reserve: usize,
        for_the_owner: bool,
    ) -> Result<(), SwarmError> {
        let err = |what: &'static str| {
            move |e: candle_core::Error| SwarmError::Internal(format!("block drafter {what}: {e}"))
        };
        let n = context.dim(1).map_err(err("context"))?;
        if cache.len + n > BLOCK_MOST_POSITIONS {
            return Err(SwarmError::Internal(format!(
                "block drafter: its cache would pass {BLOCK_MOST_POSITIONS} positions"
            )));
        }
        if cache.kv.len() != self.layers.len() {
            cache.kv.resize_with(self.layers.len(), || None);
        }
        let ctx = context.to_device(&self.device).map_err(err("context"))?;
        let base = cache.len;
        crate::inference::cpu_pools::in_phase_pool(n, 1, for_the_owner, false, || {
            for (layer, kv) in self.layers.iter().zip(cache.kv.iter_mut()) {
                let (_, k, v) = self.keys_and_values(layer, &ctx, base)?;
                crate::inference::layers::append_to_cache(
                    kv,
                    &k,
                    &v,
                    base,
                    KvStorage::F32,
                    BLOCK_MOST_POSITIONS,
                    kv_reserve,
                    None,
                )?;
            }
            Ok::<_, candle_core::Error>(())
        })
        .map_err(err("take in"))?;
        cache.len += n;
        Ok(())
    }

    /// `rows` (`[1, n, hidden]`, at positions `base..`) as one layer's keys
    /// and values, heads first and turned: `(n, k, v)`.
    fn keys_and_values(
        &self,
        layer: &DrafterLayer,
        rows: &Tensor,
        base: usize,
    ) -> candle_core::Result<(usize, Tensor, Tensor)> {
        let (kh, d) = (self.n_kv_head, self.head_dim);
        let n = rows.dim(1)?;
        let k = layer.wk.forward(rows)?.reshape((1, n, kh, d))?;
        let k = layer.k_norm.forward(&k)?.transpose(1, 2)?.contiguous()?;
        let v = layer
            .wv
            .forward(rows)?
            .reshape((1, n, kh, d))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = self.rope(&k, base, n)?;
        Ok((n, k, v))
    }

    /// NEOX RoPE over every channel of `t` (`[1, heads, n, d]`) at
    /// positions `at..at + n`.
    fn rope(&self, t: &Tensor, at: usize, n: usize) -> candle_core::Result<Tensor> {
        crate::inference::layers::rope_over_heads(
            t,
            &self.cos.narrow(0, at, n)?,
            &self.sin.narrow(0, at, n)?,
            self.head_dim,
            true,
        )
    }

    /// Take in `context` (`[1, n, hidden]`, from [`Self::context_rows`]) at
    /// the cache's next positions, then guess `guesses` tokens after `anchor`
    /// — the last token, not yet read by the target — in one forward of the
    /// block `[anchor, mask × guesses]`. The block's own keys are cut from
    /// the cache after; the context rows stay. The target's table embeds the
    /// block and its head reads the rows OUTSIDE the drafter's forward, so
    /// no closure here holds the target (a card build's `SplitModel` is not
    /// `Sync`, gotcha #912).
    ///
    /// One paced step (`inference::pace`, #440 step 4): a draft is a forward of
    /// its own, beside the target's.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn draft(
        &self,
        target: &SplitModel,
        cache: &mut BlockCache,
        context: Option<&Tensor>,
        anchor: u32,
        guesses: usize,
        kv_reserve: usize,
        for_the_owner: bool,
        over: super::GuessOver<'_>,
    ) -> Result<Vec<u32>, SwarmError> {
        let step = crate::inference::pace::Step::begin();
        let out = self.draft_unpaced(
            target,
            cache,
            context,
            anchor,
            guesses,
            kv_reserve,
            for_the_owner,
            over,
        );
        step.end(
            || {
                let _ = self.device.synchronize();
            },
            || false,
        );
        out
    }

    #[allow(clippy::too_many_arguments)]
    fn draft_unpaced(
        &self,
        target: &SplitModel,
        cache: &mut BlockCache,
        context: Option<&Tensor>,
        anchor: u32,
        guesses: usize,
        kv_reserve: usize,
        for_the_owner: bool,
        over: super::GuessOver<'_>,
    ) -> Result<Vec<u32>, SwarmError> {
        let err = |what: &'static str| {
            move |e: candle_core::Error| SwarmError::Internal(format!("block drafter {what}: {e}"))
        };
        let guesses = guesses.min(self.block_size - 1);
        if guesses == 0 {
            return Ok(Vec::new());
        }
        let n_ctx = context
            .map_or(Ok(0), |c| c.dim(1))
            .map_err(err("context"))?;
        let rows = guesses + 1;
        let start = cache.len + n_ctx;
        if start + rows > BLOCK_MOST_POSITIONS {
            return Err(SwarmError::Internal(format!(
                "block drafter: its cache would pass {BLOCK_MOST_POSITIONS} positions"
            )));
        }
        if cache.kv.len() != self.layers.len() {
            cache.kv.resize_with(self.layers.len(), || None);
        }
        let ids: Vec<i64> = std::iter::once(i64::from(anchor))
            .chain(std::iter::repeat_n(i64::from(self.mask_token), guesses))
            .collect();
        let embedded = target
            .embed_ids(&ids)?
            .to_device(&self.device)
            .and_then(|t| t.to_dtype(DType::F32))
            .map_err(err("block embedding"))?;
        let ctx = context
            .map(|c| c.to_device(&self.device))
            .transpose()
            .map_err(err("context"))?;
        let base = cache.len;
        let out = crate::inference::cpu_pools::in_phase_pool(
            rows + n_ctx,
            1,
            for_the_owner,
            false,
            || {
                let mut x = embedded;
                for (layer, kv) in self.layers.iter().zip(cache.kv.iter_mut()) {
                    x = self
                        .layer_forward(layer, &x, ctx.as_ref(), base, kv, kv_reserve)
                        .map_err(err("layer"))?;
                }
                self.norm
                    .forward(&x)
                    .and_then(|x| x.narrow(1, 1, guesses))
                    .map_err(err("output norm"))
            },
        );
        // The block's own keys go, the context rows stay — whatever the
        // forward did.
        for kv in cache.kv.iter_mut().flatten() {
            kv.truncate(start);
        }
        cache.len = start;
        let out = out?;
        // On the processor the guesses read the hot rows and the
        // conversation's own words (#425 (b), FR-Spec), where the model's
        // head keeps them: there the head read for every row was most of a
        // draft (247 ms at 15 guesses on a 9B). The check reads every word,
        // so the reply is the same; only a guess past them is lost.
        if matches!(over, super::GuessOver::HotAnd(_)) && target.mtp_hot_rows().is_some() {
            return target.mtp_guesses(&out, over);
        }
        // A drafter in system memory beside a card's head hands it the
        // block's rows (guesses x hidden, a few hundred KB) — llama.cpp's
        // copy between backends.
        let out = if out.device().same_device(target.device()) {
            out
        } else {
            out.to_device(target.device())
                .map_err(err("block to the output head"))?
        };
        let logits = match (&target.output, &target.drafter_head) {
            (None, Some(head)) => head.forward(&out).map_err(err("output head"))?,
            _ => target.logits_from(&out)?,
        };
        // On a card the guesses are picked there, so only their ids cross
        // to the host (llama.cpp samples a draft on the device too): the
        // ONE greedy pick, the first of equal maxima. A score that is not
        // finite stops the drafter for the reply.
        #[cfg(feature = "candle-cuda")]
        if logits.device().is_cuda() {
            let rows = logits.squeeze(0).map_err(err("logits"))?;
            return crate::inference::tree_rank::greedy_picks(&rows)
                .map_err(err("guesses picked on the card"))?
                .ok_or_else(|| {
                    SwarmError::Internal("block drafter: a guess's scores are not finite".into())
                });
        }
        let rows: Vec<Vec<f32>> = logits
            .to_dtype(DType::F32)
            .and_then(|l| l.squeeze(0))
            .and_then(|l| l.to_vec2())
            .map_err(err("logits"))?;
        Ok(rows
            .iter()
            .map(|row| crate::inference::sampling::argmax(row))
            .collect())
    }
}

/// A drafter's widths, read off its header: the ONE reading, by
/// [`BlockDrafter::load`] and by what it is charged ([`charged`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct DrafterShape {
    hidden: usize,
    n_layers: usize,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
}

impl DrafterShape {
    /// The widths, or the key a drafter must carry and this header does not.
    fn read(ct: &gguf_file::Content) -> Result<Self, String> {
        let num = |key: &str| ct.metadata.get(&format!("dflash.{key}")).and_then(meta_u32);
        let need = |key: &str| {
            num(key)
                .map(|v| v as usize)
                .ok_or_else(|| format!("no dflash.{key}"))
        };
        let hidden = need("embedding_length")?;
        let n_head = need("attention.head_count")?;
        Ok(Self {
            hidden,
            n_layers: need("block_count")?,
            n_head,
            n_kv_head: need("attention.head_count_kv")?,
            head_dim: num("attention.key_length").map_or(hidden / n_head.max(1), |d| d as usize),
        })
    }

    /// Its cache at its longest ([`BLOCK_MOST_POSITIONS`]): keys and values,
    /// f32, every layer.
    fn cache_at_longest(&self) -> u64 {
        (self.n_layers as u64)
            .saturating_mul(2 * 4 * BLOCK_MOST_POSITIONS as u64)
            .saturating_mul(self.n_kv_head as u64)
            .saturating_mul(self.head_dim as u64)
    }
}

/// What a drafter is charged when attached: every tensor in its file and its
/// cache at its longest. The ONE figure — weighed off the header before any
/// weight is loaded ([`charged_in`]) and kept as [`BlockDrafter::bytes`] after
/// — so the device chosen before the load is the one the rule after it keeps.
/// Weighed apart (the weights alone before, the cache added after), a drafter
/// whose weights fitted its card and whose cache did not was loaded there,
/// refused, and never offered system memory: Qwen3.5-9B's 1,309 MB of weights
/// and 384 MB of cache beside 1,715 MB a card had left (#451, 2026-11-14).
/// llama.cpp's `common/fit.cpp` likewise sizes a load's weights AND its
/// context before it places anything.
fn charged(ct: &gguf_file::Content, shape: &DrafterShape) -> u64 {
    ct.tensor_infos
        .values()
        .map(super::tensor_byte_size)
        .fold(0u64, u64::saturating_add)
        .saturating_add(shape.cache_at_longest())
}

/// [`charged`] for the drafter file at `path`, read off its header alone.
/// `None` where the header cannot be read or lacks a width (the load then
/// says why).
fn charged_in(path: &Path) -> Option<u64> {
    let file = std::fs::File::open(path).ok()?;
    let ct = gguf_file::Content::read(&mut std::io::BufReader::new(file)).ok()?;
    let shape = DrafterShape::read(&ct).ok()?;
    Some(charged(&ct, &shape))
}

/// May a drafter taking `takes` bytes come off a conversation `budget`?
/// Only while one conversation of the model's whole context
/// (`one_conversation`) still fits in what is left; an unknown budget refuses
/// nothing ([`SplitModel::attach_block_drafter`]).
fn fits_beside_one_conversation(budget: Option<u64>, takes: u64, one_conversation: u64) -> bool {
    budget.is_none_or(|b| takes.saturating_add(one_conversation) <= b)
}

/// Where a block drafter's weights go.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DrafterHome {
    /// The model's own device.
    OwnDevice,
    /// System memory, beside a card/processor split whose card has no room.
    SystemMemory,
    /// Neither.
    NoRoom,
}

/// Where a drafter that `takes` bytes goes (2026-11-14): the model's device
/// while one conversation (`device.1`) still fits beside it in that budget
/// (`device.0`) — else, where `system` is offered (a whole model split
/// between a card and the processor, [`block_drafter_in_system_memory_on`]),
/// system memory, under the same rule against the processor-held layers'
/// budget and conversation, which must be KNOWN: a new placement in the
/// machine's memory is never unchecked. llama.cpp places a draft model on
/// devices of its own (`--spec-draft-device`, `--spec-draft-ngl auto`); a 27B
/// on a 12 GB card had no room for its 1.85 GB drafter, and its machine had.
fn drafter_home(
    takes: u64,
    device: (Option<u64>, u64),
    system: Option<(Option<u64>, u64)>,
) -> DrafterHome {
    if fits_beside_one_conversation(device.0, takes, device.1) {
        return DrafterHome::OwnDevice;
    }
    match system {
        Some((Some(budget), one)) if takes.saturating_add(one) <= budget => {
            DrafterHome::SystemMemory
        }
        _ => DrafterHome::NoRoom,
    }
}

impl SplitModel {
    /// Attach the block drafter in `dir` ([`BLOCK_DRAFTER_FILE`]), where
    /// there is one and the switch is on. What it takes — its weights, its
    /// cache at its longest and, beside a split's first segment, the output
    /// head it borrows — comes off this model's conversation budget for its
    /// device, and it is attached only while ONE conversation of the model's
    /// whole context still fits in what is left: vLLM's rule, which sizes
    /// the cache from what every model's weights leave and refuses only where
    /// one `max_model_len` sequence would not fit
    /// (`_check_enough_kv_cache_memory`). The guess head's quarter refused
    /// this 1.7 GB drafter beside a 9B's first segment on a 16 GB machine
    /// (6.1 GB budget) that had room for 137K positions beside it. Not
    /// attached, the borrowed head goes too. A file that cannot be used is
    /// logged and left; the model serves without it.
    /// A whole model on the processor with its own guess head: there the head
    /// writes faster than a block drafter beside it — a block's draft costs
    /// 25-105 ms and its 16-position check ~4 steps, where the head guesses at
    /// ~6 ms a word (Qwen3.5-4B Q4_K_M, 8 threads: drafter on 16.8-19.0 /
    /// 16.8-17.8 / 11.1-11.7 / 14.5-17.2 tok/s on a palindrome, a BST class, a
    /// story and a hash map against 16.0-18.0 / 17.0-19.2 / 11.9-12.2 /
    /// 15.6-19.4 without it; `~/rig428/cpu2`, 2026-11-08 night). A split's first
    /// segment, a card, or a model without a guess head keep the drafter.
    fn head_guesses_better_here(&self) -> bool {
        self.mtp.is_some()
            && self.output.is_some()
            && self.layer_start == 0
            && self.layer_end == self.total_layers
            && self.device().is_cpu()
    }

    pub(crate) fn attach_block_drafter(&mut self, dir: &Path) {
        self.block_drafter_state = self.try_attach_block_drafter(dir);
    }

    /// Whether a block drafter sits beside these layers, and if not, why —
    /// the word a hand-off's last frame carries (`HowGuessed::block_drafter`).
    pub(crate) fn block_drafter_state(&self) -> swarmllm_types::BlockDrafterState {
        self.block_drafter_state
    }

    /// [`Self::attach_block_drafter`]'s work, answering what came of it.
    fn try_attach_block_drafter(&mut self, dir: &Path) -> swarmllm_types::BlockDrafterState {
        use swarmllm_types::BlockDrafterState;
        // A guess head beside a split's first segment keeps the output head
        // it reads, charged before any drafter, which then shares it.
        let head_charged = self.keep_the_split_guess_head();
        let path = dir.join(BLOCK_DRAFTER_FILE);
        let head_here = self.output.is_some() || self.drafter_head.is_some();
        let refused = if !block_drafter_on() {
            Some(BlockDrafterState::Off)
        } else if self.layer_start != 0 || !head_here {
            Some(BlockDrafterState::NotThisSegment)
        } else if !path.is_file() {
            Some(BlockDrafterState::NotHeld)
        } else {
            None
        };
        if let Some(refused) = refused {
            self.drop_drafter_head();
            return refused;
        }
        if self.head_guesses_better_here() && !block_drafter_on_the_processor_anyway() {
            tracing::info!(
                "block drafter not attached: on the processor this model's own guess head writes faster"
            );
            self.drop_drafter_head();
            return BlockDrafterState::HeadWritesFaster;
        }
        // What it is charged is known off its header before any weight is
        // loaded ([`charged`]: its weights AND its cache, the figure the rule
        // below weighs, which only hot rows can raise — and only on the
        // processor, never beside a card): a device it does not fit beside one
        // conversation is never handed it, and a card that cannot hold it
        // hands it to system memory BEFORE the load, where that is offered.
        // Loaded first, Qwen3.6-27B's 1.85 GB drafter beside the 27 of 64
        // layers a 12 GB card held ran the card out of memory on every load
        // (field report v0.3.280, 2026-11-11). Prior art: llama.cpp's
        // `common/fit.cpp` sizes what a load takes before it allocates.
        let mut home = DrafterHome::OwnDevice;
        if let Some(drafter_charged) = charged_in(&path) {
            let takes = drafter_charged.saturating_add(if head_charged {
                0
            } else {
                self.drafter_head_bytes
            });
            let one_conversation = self.kv_growth.bytes_for(self.max_seq_len);
            home = drafter_home(
                takes,
                (self.kv_budget_bytes, one_conversation),
                self.drafter_may_sit_in_system_memory(),
            );
            if home == DrafterHome::NoRoom {
                tracing::info!(
                    mb = takes >> 20,
                    budget_mb = ?self.kv_budget_bytes.map(|b| b >> 20),
                    one_conversation_mb = one_conversation >> 20,
                    system_memory = ?self.drafter_may_sit_in_system_memory()
                        .map(|(b, one)| (b.map(|b| b >> 20), one >> 20)),
                    "block drafter not attached: it does not fit beside one conversation \
                     of this model on its device"
                );
                self.drop_drafter_head();
                return BlockDrafterState::NoRoom;
            }
        }
        let device = match home {
            DrafterHome::SystemMemory => Device::Cpu,
            _ => self.device.clone(),
        };
        let drafter = match BlockDrafter::load(&path, self, &device) {
            Ok(d) => d,
            Err(e) => {
                tracing::warn!(error = %e, "block drafter not attached");
                self.drop_drafter_head();
                return BlockDrafterState::FailedToLoad;
            }
        };
        // The head's first rows, where the drafter reads it on the processor
        // and no guess head keeps them (`SplitModel::drafter_hot`).
        let hot = if self.mtp_hot_rows().is_some() {
            None
        } else {
            let head = self.output.as_ref().or(self.drafter_head.as_ref());
            match head.and_then(|h| super::mtp::hot_rows_for(Some(h), false).map(|r| (h, r))) {
                Some((h, (rows, bytes))) => {
                    match super::mtp::head_rows(h, super::mtp::HeadRows::First(rows)) {
                        Ok(rows) => Some((rows, bytes)),
                        Err(e) => {
                            tracing::warn!(error = %e, "block drafter: reads every word");
                            None
                        }
                    }
                }
                None => None,
            }
        };
        let hot_bytes = hot.as_ref().map_or(0, |(_, bytes)| *bytes);
        let head_bytes = if head_charged {
            0
        } else {
            self.drafter_head_bytes
        };
        let takes = drafter
            .bytes
            .saturating_add(head_bytes)
            .saturating_add(hot_bytes);
        // The drafter and the head it borrows sit on the model's own device
        // (`BlockDrafter::load`): its budget pays — or, for a drafter in
        // system memory beside a card/processor split, the split's
        // processor-side one (`host_kv_budget_bytes`), which must be known.
        let in_system_memory = home == DrafterHome::SystemMemory;
        let (budget, growth) = if in_system_memory {
            (&mut self.host_kv_budget_bytes, &self.host_kv_growth)
        } else {
            (&mut self.kv_budget_bytes, &self.kv_growth)
        };
        let one_conversation = growth.bytes_for(self.max_seq_len);
        let fits = (!in_system_memory || budget.is_some())
            && fits_beside_one_conversation(*budget, takes, one_conversation);
        if fits {
            if let Some(b) = budget.as_mut() {
                *b -= takes;
            }
        }
        tracing::info!(
            mb = takes >> 20,
            head_mb = self.drafter_head_bytes >> 20,
            hot_rows_mb = hot_bytes >> 20,
            budget_left_mb = ?budget.map(|b| b >> 20),
            one_conversation_mb = one_conversation >> 20,
            layers = drafter.layers.len(),
            block = drafter.block_size,
            target_layers = ?drafter.target_layers,
            rule = ?drafter.rule,
            in_system_memory,
            attached = fits,
            "block drafter (#428)"
        );
        if fits {
            self.block_drafter = Some(drafter);
            self.drafter_hot = hot.map(|(rows, _)| rows);
            if in_system_memory {
                BlockDrafterState::AttachedInSystemMemory
            } else {
                BlockDrafterState::Attached
            }
        } else {
            self.drop_drafter_head();
            BlockDrafterState::NoRoom
        }
    }

    /// Whether a drafter the card has no room for may sit in system memory,
    /// and if so the processor-held layers' budget and one conversation's
    /// cache there ([`drafter_home`]): a WHOLE model (its own head here, none
    /// borrowed) split between a card and the processor, with the switch on.
    fn drafter_may_sit_in_system_memory(&self) -> Option<(Option<u64>, u64)> {
        (block_drafter_in_system_memory_on()
            && self.device.is_cuda()
            && self.output.is_some()
            && self.drafter_head.is_none()
            && self.host_kv_growth.per_position() > 0)
            .then(|| {
                (
                    self.host_kv_budget_bytes,
                    self.host_kv_growth.bytes_for(self.max_seq_len),
                )
            })
    }

    /// The output head a split's first segment read for its drafter, given
    /// back where no drafter is attached — unless a guess head beside the
    /// segment reads it ([`Self::keep_the_split_guess_head`]).
    fn drop_drafter_head(&mut self) {
        self.drafter_hot = None;
        if self.split_head_reads_layer().is_some() {
            return;
        }
        self.drafter_head = None;
        self.drafter_head_bytes = 0;
    }

    /// A guess head beside a split's first segment (#425 (a)) keeps the output
    /// head it reads (`drafter_head`) and, on the processor, that head's first
    /// rows (`MtpHead::hot`, #425 (b)), charged to the model's own budget while
    /// one conversation of its whole context still fits beside them — the block
    /// drafter's rule (`fits_beside_one_conversation`). Without the head, or
    /// past that, the guess head goes. `true` where the borrowed head stays,
    /// charged.
    fn keep_the_split_guess_head(&mut self) -> bool {
        if self.split_head_reads_layer().is_none() {
            return false;
        }
        let Some(head) = self.drafter_head.as_ref() else {
            tracing::info!(
                "guess head beside the split's first segment not kept: the output head could not be read"
            );
            self.mtp = None;
            return false;
        };
        let on_card = self.mtp.as_ref().is_some_and(|m| m.device.is_cuda());
        let hot = match super::mtp::hot_rows_for(Some(head), on_card) {
            Some((rows, bytes)) => {
                match super::mtp::head_rows(head, super::mtp::HeadRows::First(rows)) {
                    Ok(rows) => Some((rows, bytes)),
                    Err(e) => {
                        tracing::warn!(error = %e, "guess head beside the first segment: reads every word");
                        None
                    }
                }
            }
            None => None,
        };
        let hot_bytes = hot.as_ref().map_or(0, |(_, bytes)| *bytes);
        let takes = self.drafter_head_bytes.saturating_add(hot_bytes);
        let one_conversation = self.kv_growth.bytes_for(self.max_seq_len);
        let fits = fits_beside_one_conversation(self.kv_budget_bytes, takes, one_conversation);
        if fits {
            if let Some(b) = self.kv_budget_bytes.as_mut() {
                *b -= takes;
            }
        }
        tracing::info!(
            head_mb = self.drafter_head_bytes >> 20,
            hot_rows_mb = hot_bytes >> 20,
            on_card,
            reads_layer = ?self.split_head_reads_layer(),
            budget_left_mb = ?self.kv_budget_bytes.map(|b| b >> 20),
            one_conversation_mb = one_conversation >> 20,
            kept = fits,
            "guess head beside the split's first segment (#425 (a))"
        );
        if !fits {
            self.mtp = None;
            return false;
        }
        if let (Some(m), Some((rows, _))) = (self.mtp.as_mut(), hot) {
            m.hot = Some(rows);
        }
        true
    }

    pub(crate) fn block_drafter(&self) -> Option<&BlockDrafter> {
        self.block_drafter.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A drafter its card has no room for goes to system memory only where
    /// that is offered and its budget is KNOWN and holds it beside one
    /// conversation of the processor-held layers; the card is asked first.
    /// The shape: a 27B on a 12 GB card, the 1.85 GB drafter.
    #[test]
    fn a_drafter_the_card_cannot_hold_goes_to_system_memory_only_where_it_fits() {
        const MB: u64 = 1 << 20;
        let takes = 1_900 * MB;
        let card_full = (Some(1_000 * MB), 600 * MB);
        let card_room = (Some(4_000 * MB), 600 * MB);
        let machine = Some((Some(9_000 * MB), 3_000 * MB));
        assert_eq!(
            drafter_home(takes, card_room, machine),
            DrafterHome::OwnDevice
        );
        assert_eq!(
            drafter_home(takes, card_full, machine),
            DrafterHome::SystemMemory
        );
        // Not offered (a model on one device, the switch off): as before.
        assert_eq!(drafter_home(takes, card_full, None), DrafterHome::NoRoom);
        // Offered, but the machine's budget is unknown: never unchecked.
        assert_eq!(
            drafter_home(takes, card_full, Some((None, 3_000 * MB))),
            DrafterHome::NoRoom
        );
        // Offered, and one conversation would not fit beside it there.
        assert_eq!(
            drafter_home(takes, card_full, Some((Some(4_000 * MB), 3_000 * MB))),
            DrafterHome::NoRoom
        );
    }

    /// What a drafter is charged before its load is what it is charged after:
    /// its tensors AND its cache at its longest, read off the header. Weighed
    /// as its weights alone, Qwen3.5-9B's drafter (1,309 MB, 6 layers of 8 KV
    /// heads × 128) fitted a card with 1,715 MB left beside a 192 MB
    /// conversation, was loaded there, refused once its 384 MB cache joined,
    /// and never offered system memory (#451's probe, 2026-11-14).
    #[test]
    fn a_drafter_is_charged_its_cache_before_it_is_placed() {
        use candle_core::quantized::{GgmlDType, QTensor};
        use gguf_file::Value;
        const MB: u64 = 1 << 20;
        let md: Vec<(&str, Value)> = vec![
            ("general.architecture", Value::String("dflash".into())),
            ("dflash.embedding_length", Value::U32(4096)),
            ("dflash.block_count", Value::U32(6)),
            ("dflash.attention.head_count", Value::U32(32)),
            ("dflash.attention.head_count_kv", Value::U32(8)),
            ("dflash.attention.key_length", Value::U32(128)),
        ];
        let refs: Vec<(&str, &Value)> = md.iter().map(|(k, v)| (*k, v)).collect();
        let weight = QTensor::quantize(
            &Tensor::zeros((4, 64), candle_core::DType::F32, &Device::Cpu).unwrap(),
            GgmlDType::F32,
        )
        .unwrap();
        let mut buf = std::io::Cursor::new(Vec::new());
        gguf_file::write(&mut buf, &refs, &[("fc.weight", &weight)]).unwrap();
        buf.set_position(0);
        let ct = gguf_file::Content::read(&mut buf).unwrap();
        let shape = DrafterShape::read(&ct).unwrap();
        assert_eq!(shape.cache_at_longest(), 384 * MB);
        assert_eq!(charged(&ct, &shape), 4 * 64 * 4 + 384 * MB);

        // The probe's figures: the card is refused it, system memory takes it.
        let weights = 1_309 * MB;
        let card = (Some(1_715 * MB), 192 * MB);
        let machine = Some((Some(9_000 * MB), 1_000 * MB));
        let takes = weights + shape.cache_at_longest();
        assert_eq!(
            drafter_home(takes, card, machine),
            DrafterHome::SystemMemory
        );
        // Its weights alone chose the card the check after the load refused.
        assert_eq!(drafter_home(weights, card, machine), DrafterHome::OwnDevice);
        assert!(!fits_beside_one_conversation(card.0, takes, card.1));

        // A header without a width charges nothing it cannot read.
        let bare = [("general.architecture", Value::String("dflash".into()))];
        let refs: Vec<(&str, &Value)> = bare.iter().map(|(k, v)| (*k, v)).collect();
        let mut buf = std::io::Cursor::new(Vec::new());
        gguf_file::write(&mut buf, &refs, &[]).unwrap();
        buf.set_position(0);
        let ct = gguf_file::Content::read(&mut buf).unwrap();
        assert_eq!(
            DrafterShape::read(&ct),
            Err("no dflash.embedding_length".to_string())
        );
    }

    /// The drafter comes off the conversation budget while one conversation
    /// of the model's context still fits beside it — the case that refused
    /// it under the guess head's quarter: Qwen3.5-9B's first segment on a
    /// 16 GB machine, 6,069 MB of budget, the drafter and its borrowed head
    /// 2,527 MB, 8,192 positions at 32 KB.
    #[test]
    fn a_drafter_takes_room_while_one_whole_conversation_still_fits() {
        const MB: u64 = 1 << 20;
        let one = 8_192 * 32 * 1024;
        assert!(fits_beside_one_conversation(
            Some(6_069 * MB),
            2_527 * MB,
            one
        ));
        assert!(fits_beside_one_conversation(
            Some(2_527 * MB + one),
            2_527 * MB,
            one
        ));
        assert!(!fits_beside_one_conversation(
            Some(2_527 * MB + one - 1),
            2_527 * MB,
            one
        ));
        assert!(!fits_beside_one_conversation(
            Some(2_000 * MB),
            2_527 * MB,
            0
        ));
        assert!(fits_beside_one_conversation(None, u64::MAX, u64::MAX));
    }

    /// z-lab's `_attention_mask`, written out: query `i` of `q_len` at
    /// `key_len − q_len + i`, key `j`.
    fn reference_sees(
        i: usize,
        j: usize,
        q_len: usize,
        key_len: usize,
        causal: bool,
        w: Option<usize>,
    ) -> bool {
        let qp = (key_len - q_len + i) as i64;
        let kp = j as i64;
        let mut seen = true;
        if causal {
            seen &= kp <= qp;
        }
        if let Some(w) = w.map(|w| w as i64) {
            seen &= qp - kp < w;
            if !causal {
                seen &= kp - qp < w;
            }
        }
        seen
    }

    /// The block's mask is the reference's at every shape it is asked:
    /// one way and both, a window that binds and one that does not.
    #[test]
    fn a_blocks_mask_is_the_references() -> candle_core::Result<()> {
        let dev = Device::Cpu;
        for &(rows, start) in &[(1usize, 0usize), (4, 0), (8, 3), (16, 40), (16, 5)] {
            for &one_way in &[false, true] {
                for &window in &[None, Some(8usize), Some(4096)] {
                    let keys = start + rows;
                    let mask = block_mask(rows, start, one_way, window, &dev)?;
                    let got: Vec<Vec<f32>> = match &mask {
                        Some(m) => m.to_vec2()?,
                        None => vec![vec![0.0; keys]; rows],
                    };
                    for (i, row) in got.iter().enumerate() {
                        for (j, &v) in row.iter().enumerate() {
                            let want = reference_sees(i, j, rows, keys, one_way, window);
                            assert_eq!(
                                v == 0.0,
                                want,
                                "rows {rows} start {start} one_way {one_way} window {window:?} at ({i}, {j})"
                            );
                        }
                    }
                    if !one_way && window.is_none_or(|w| keys <= w) {
                        assert!(mask.is_none(), "a mask that hides nothing is not built");
                    }
                }
            }
        }
        Ok(())
    }
}
