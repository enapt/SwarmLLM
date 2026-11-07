//! A file's next-token-prediction block: the guess head a model was trained
//! with (#425).
//!
//! Qwen 3.5, 3.6 and 3.8 files (`qwen35`, `qwen35moe`) carry one extra decoder
//! block past their last layer (`nextn_predict_layers = 1`): a full-attention
//! layer that reads the NEXT token's embedding beside the trunk's final
//! hidden state and predicts the token after it. The model was trained with
//! it, so its guesses agree with the model far more often than a separate
//! small drafter's — and Qwen 3.5's recurrent layers refuse a drafter
//! (`SplitModel::carries_recurrent_state`), which left these models with the
//! n-gram alone.
//!
//! Prior art: llama.cpp `models/qwen35.cpp` (`graph_mtp`) and
//! `common/speculative.cpp` (`common_speculative_impl_draft_mtp`), read at
//! `~/llama.cpp-ref` (2026-09-25). Position `p` of the head pairs token `x_p`
//! with the trunk's hidden state at `p - 1` (after `output_norm`); its cache
//! is filled over every position the trunk writes, and a draft feeds each
//! step's own normed output back as the next step's hidden state. vLLM's
//! Qwen3-Next MTP proposer is the same shape. Ours differs in one place: the
//! head takes in only ACCEPTED positions (llama.cpp runs it over the whole
//! check, then drops the rejected rows), so no row is computed to be thrown
//! away, and the first draft step rides on the same forward.

use std::borrow::Cow;

use candle_core::quantized::{QStorage, QTensor};
use candle_core::{DType, Device, IndexOp, Tensor, D};
use candle_nn::Module;

use super::kv_cache::LayerKv;
use super::model::SplitModel;
use super::{FfnVariant, QMatMul, Qwen35AttnWeights};
use crate::error::SwarmError;
use crate::inference::residual_norm::{Residual, RmsNorm};

/// The most positions one request's head keeps in its cache. Past it the
/// head stops guessing for that reply (the n-gram goes on): its cache is not
/// charged by admission, and this keeps it to ~64 MB on a card at the widest
/// file here (Qwen3.6-27B: 4 KV heads × 256 × K and V in f16).
pub(crate) const MTP_MOST_POSITIONS: usize = 16_384;

/// The most prompt positions the head reads before the first guess: the
/// prompt's last ones. A guess depends mostly on the recent context, and this
/// bounds the first catch-up — one layer over these rows, after the first
/// token is out — on a long agent prompt.
pub(crate) const MTP_PROMPT_ROWS: usize = 4_096;

/// Rows the head takes in per forward when it catches up.
pub(crate) const MTP_CATCH_UP_CHUNK: usize = 512;

/// Rows of the output head a guess on the processor reads by default
/// (#425 (b)): the tokenizer's first 32,768 words, FR-Spec's best size.
///
/// Prior art: FR-Spec (arXiv 2502.14856) — a draft's output head over the
/// most frequent 32K words cut its cost ~75% with no retraining (that head is
/// most of a draft's time on a large vocabulary); SGLang ships it as
/// `--speculative-token-map`, a frequency file made per model from a corpus;
/// EAGLE-3 trains its draft over a 32K vocabulary with a draft-to-target map.
/// Ours differs in three places. No frequency file exists for Qwen 3.5's
/// tokenizer, so the hot rows are its FIRST ids: a byte-pair tokenizer's
/// merges are made most frequent first, and this one's ids are its merges in
/// order (`256 + k` is merge `k`, read off the 4B's file 2026-11-07). On
/// samples tokenized as the file does, 32K covered 89-94% of English, code
/// and JSON and 16-17% of Thai and Chinese — so a reply's guesses also read
/// the conversation's own words past them (a copied name is the head's
/// surest guess), and a conversation in words mostly past them reads the
/// whole vocabulary (`DraftWords` in `model_worker`). And only where the
/// output head is on the processor: there one guess took 20-21 ms, most of
/// it this 248,320-row head; on a card a whole guess took 3 ms.
pub(crate) const DRAFT_HOT_ROWS: usize = 32_768;

/// The most of a conversation's own words past the hot rows a reply's
/// guesses read beside them; with more, its guesses read every word.
pub(crate) const DRAFT_MOST_OWN_WORDS: usize = 8_192;

/// `SWARMLLM_MTP_VOCAB`: `full` — a guess reads the whole output head, as
/// before (#425 (b)); a number — that many hot rows; else [`DRAFT_HOT_ROWS`].
fn draft_hot_rows() -> Option<usize> {
    static ROWS: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    *ROWS.get_or_init(|| match std::env::var("SWARMLLM_MTP_VOCAB").as_deref() {
        Ok("full") => None,
        Ok(n) => Some(n.parse().ok().filter(|&n| n > 0).unwrap_or(DRAFT_HOT_ROWS)),
        Err(_) => Some(DRAFT_HOT_ROWS),
    })
}

/// The head, loaded only where the segment ends the model.
pub(crate) struct MtpHead {
    pub(super) eh_proj: QMatMul,
    pub(super) enorm: RmsNorm,
    pub(super) hnorm: RmsNorm,
    /// `nextn.shared_head_norm`, else the model's own `output_norm`.
    pub(super) head_norm: RmsNorm,
    pub(super) attn: Qwen35AttnWeights,
    pub(super) ffn: FfnVariant,
    pub(super) attention_norm: RmsNorm,
    pub(super) post_attention_norm: RmsNorm,
    /// The output head's first rows, which a guess reads where the head is
    /// on the processor (#425 (b), [`DRAFT_HOT_ROWS`]); `None` reads them all.
    pub(super) hot: Option<QMatMul>,
    /// Where the block runs: the trunk's last layer's device (its experts, for
    /// a mixture-of-experts block on a card, in the machine's memory).
    pub(super) device: Device,
}

/// One request's cache of the head's attention layer. Its positions count
/// from the first row it took in — RoPE is relative, so a head that starts
/// mid-prompt attends exactly as one that started at 0 would over the same
/// rows.
#[derive(Default)]
pub(crate) struct MtpCache {
    kv: Option<LayerKv>,
    len: usize,
}

impl MtpCache {
    /// Rows held — the head's next position.
    pub(crate) fn len(&self) -> usize {
        self.len
    }

    /// Keep the first `len` rows: a draft's own rows go once it is checked.
    pub(crate) fn truncate(&mut self, len: usize) {
        if len < self.len {
            if let Some(kv) = self.kv.as_mut() {
                kv.truncate(len);
            }
            self.len = len;
        }
    }
}

/// `SWARMLLM_MTP=0`: no next-token-prediction block is loaded or used, for
/// A/B inside one binary.
pub(crate) fn mtp_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("SWARMLLM_MTP").as_deref() != Ok("0"))
}

/// The index of the file's next-token-prediction block where this load reads
/// it, else `None`: the switch is on, the segment is the WHOLE model (the
/// worker's own replies are its only reader so far, and they need the token
/// table), the family is Qwen 3.5's (`qwen35`, `qwen35moe` — llama.cpp's
/// `graph_mtp` is written for these two), the file declares exactly one block
/// (`nextn_layers`), and that block is the shape the head runs: a
/// full-attention layer with `eh_proj`, `enorm` and `hnorm`, and neither a
/// token table nor an output head of its own (llama.cpp reads those where
/// present; no file here carries them).
pub(crate) fn block_to_load(
    ct: &candle_core::quantized::gguf_file::Content,
    arch: &crate::inference::model_arch::ModelArch,
    nextn_layers: u32,
    whole: bool,
    layer_end: usize,
    block_count: usize,
) -> Option<usize> {
    use crate::inference::model_arch::ModelArch;
    if !mtp_on()
        || !whole
        || layer_end != block_count
        || nextn_layers != 1
        || !matches!(arch, ModelArch::Qwen35 | ModelArch::Qwen35Moe)
    {
        return None;
    }
    let has = |s: &str| {
        ct.tensor_infos
            .contains_key(&format!("blk.{block_count}.{s}"))
    };
    let shaped = [
        "nextn.eh_proj.weight",
        "nextn.enorm.weight",
        "nextn.hnorm.weight",
        "attn_norm.weight",
        "post_attention_norm.weight",
        "attn_q.weight",
        "attn_k.weight",
        "attn_v.weight",
        "attn_output.weight",
        "attn_q_norm.weight",
        "attn_k_norm.weight",
    ]
    .iter()
    .all(|s| has(s));
    if !shaped || has("nextn.embed_tokens.weight") || has("nextn.shared_head_head.weight") {
        tracing::info!(
            block = block_count,
            "next-token-prediction block not in the shape the head runs; not loaded"
        );
        return None;
    }
    Some(block_count)
}

/// What block `block`'s weights take: `(on its device, in the machine's
/// memory beside a card)` — every `blk.{block}` tensor, its routed experts in
/// the second where a card-held block keeps them there
/// (`LayerPlacement::for_the_next_token_block`).
pub(crate) fn block_bytes(
    ct: &candle_core::quantized::gguf_file::Content,
    block: usize,
    on_card: bool,
) -> (u64, u64) {
    ct.tensor_infos
        .iter()
        .filter(|(name, _)| {
            matches!(super::segment_tensor(name), super::SegmentTensor::Layer(i) if i == block)
        })
        .fold((0, 0), |(here, beside), (name, info)| {
            let bytes = super::tensor_byte_size(info);
            if on_card && name.contains("_exps.") {
                (here, beside + bytes)
            } else {
                (here + bytes, beside)
            }
        })
}

/// Rows of a quantized `[rows, cols]` head in whole blocks, and one row's
/// bytes.
fn rows_and_row_bytes(q: &QTensor) -> Option<(usize, usize)> {
    let dims = q.shape().dims();
    let [rows, cols] = dims[..] else {
        return None;
    };
    let dtype = q.dtype();
    (cols % dtype.block_size() == 0).then(|| (rows, cols / dtype.block_size() * dtype.type_size()))
}

/// The hot rows a guess reads first, `(rows, bytes)`, where they apply: the
/// switch is on, the output head is quantized and on the processor, so is
/// the block (`block_on_card` false), and the head has more rows than that.
pub(crate) fn hot_rows_for(output: Option<&QMatMul>, block_on_card: bool) -> Option<(usize, u64)> {
    let rows = draft_hot_rows()?;
    let q = output?.quantized_weight()?;
    if block_on_card || !q.device().is_cpu() {
        return None;
    }
    let (vocab, row_bytes) = rows_and_row_bytes(q)?;
    (rows < vocab).then(|| (rows, (rows * row_bytes) as u64))
}

/// Rows of a quantized output head.
pub(crate) enum HeadRows<'a> {
    /// The first `n`.
    First(usize),
    /// These, in this order.
    Of(&'a [u32]),
}

/// A copy of `rows` of `output` (quantized, `[vocab, hidden]`) as a head of
/// its own, on the device `output` is on. Each row is the same bytes, so it
/// scores its word exactly as the whole head does.
pub(crate) fn head_rows(output: &QMatMul, rows: HeadRows<'_>) -> Result<QMatMul, SwarmError> {
    let fail = |what: String| SwarmError::Internal(format!("draft words: {what}"));
    let q = output
        .quantized_weight()
        .ok_or_else(|| fail("the output head is not quantized".into()))?;
    let (vocab, row_bytes) =
        rows_and_row_bytes(q).ok_or_else(|| fail(format!("an output head of {:?}", q.shape())))?;
    let cols = q.shape().dims()[1];
    let data = q.data().map_err(|e| fail(e.to_string()))?;
    let (n, bytes): (usize, Cow<'_, [u8]>) = match rows {
        HeadRows::First(n) => (
            n,
            Cow::Borrowed(
                data.get(..n * row_bytes)
                    .ok_or_else(|| fail(format!("{n} rows of {vocab}")))?,
            ),
        ),
        HeadRows::Of(ids) => {
            let mut v = Vec::with_capacity(ids.len() * row_bytes);
            for &id in ids {
                let at = id as usize * row_bytes;
                v.extend_from_slice(
                    data.get(at..at + row_bytes)
                        .ok_or_else(|| fail(format!("word {id} of {vocab}")))?,
                );
            }
            (ids.len(), Cow::Owned(v))
        }
    };
    if n == 0 {
        return Err(fail("no rows".into()));
    }
    QStorage::from_data(bytes, &q.device(), q.dtype())
        .and_then(|storage| QTensor::new(storage, (n, cols)))
        .and_then(QMatMul::from_qtensor)
        .map_err(|e| fail(e.to_string()))
}

/// What a guess picks from ([`SplitModel::mtp_guess`]).
#[derive(Clone, Copy)]
pub(crate) enum GuessOver<'a> {
    /// The whole vocabulary.
    Everything,
    /// The hot rows, and a reply's own words past them: `(their rows, their
    /// ids)`.
    HotAnd(Option<(&'a QMatMul, &'a [u32])>),
}

impl SplitModel {
    /// The layer a forward keeps for the head: the trunk's last (absolute).
    pub(crate) fn mtp_capture_layer(&self) -> Option<usize> {
        self.mtp
            .as_ref()
            .and_then(|_| self.layer_end.checked_sub(1))
    }

    /// The head's hidden-state input from rows a forward kept after the
    /// trunk's last layer: normed as the output head reads them
    /// (llama.cpp's `t_h_nextn`).
    pub(crate) fn mtp_hidden_from(&self, kept: &Tensor) -> Result<Tensor, SwarmError> {
        let norm = self
            .norm
            .as_ref()
            .ok_or_else(|| SwarmError::Internal("next-token head: no output norm".into()))?;
        // The norm lives on the model's main device (`output_on_primary_device`).
        kept.to_device(&self.device)
            .and_then(|k| norm.forward(&k))
            .map_err(|e| SwarmError::Internal(format!("next-token head: hidden norm: {e}")))
    }

    /// The hot rows a guess reads first (`MtpHead::hot`), where it has them.
    pub(crate) fn mtp_hot_rows(&self) -> Option<usize> {
        self.mtp.as_ref()?.hot.as_ref()?.out_features().ok()
    }

    /// The output head's rows for `ids` — a conversation's own words past
    /// the hot rows — for a guess to read beside them.
    pub(crate) fn mtp_rows_of(&self, ids: &[u32]) -> Result<QMatMul, SwarmError> {
        let output = self
            .output
            .as_ref()
            .ok_or_else(|| SwarmError::Internal("draft words: no output head".into()))?;
        head_rows(output, HeadRows::Of(ids))
    }

    /// The head's guess from its last normed row (`[1, hidden]`, from
    /// [`Self::mtp_forward`]): the best word of the whole vocabulary, or of
    /// the hot rows and a reply's own words — scored as the whole head scores
    /// them, the file's suppressed words included (#282). Both are read where
    /// the whole head was before the hot rows, so an A/B of the two differs
    /// only in the rows read. No closure here may hold `self`: on a card build
    /// a `SplitModel` is not `Sync` (gotcha #912).
    pub(crate) fn mtp_guess(&self, row: &Tensor, over: GuessOver<'_>) -> Result<u32, SwarmError> {
        let err = |what: &'static str| {
            move |e: candle_core::Error| {
                SwarmError::Internal(format!("next-token head {what}: {e}"))
            }
        };
        let flat = |t: Tensor| {
            t.flatten_all()
                .and_then(|t| t.to_dtype(DType::F32))
                .and_then(|t| t.to_vec1::<f32>())
        };
        let row = row
            .to_device(&self.device)
            .map_err(err("to the output head"))?;
        let GuessOver::HotAnd(words) = over else {
            let logits = self.logits_from(&row)?;
            return Ok(crate::inference::sampling::argmax(
                &flat(logits).map_err(err("logits"))?,
            ));
        };
        let hot = self
            .mtp
            .as_ref()
            .and_then(|h| h.hot.as_ref())
            .ok_or_else(|| SwarmError::Internal("next-token head: no hot rows".into()))?;
        let mut logits = hot.forward(&row).map_err(err("hot rows"))?;
        let n = logits.dim(D::Minus1).map_err(err("hot rows"))?;
        if let Some(s) = &self.suppressed_logits {
            logits = s
                .narrow(0, 0, n)
                .and_then(|s| s.to_dtype(logits.dtype()))
                .and_then(|s| logits.broadcast_add(&s))
                .map_err(err("suppressed words"))?;
        }
        let mut scores = flat(logits).map_err(err("hot rows"))?;
        if let Some((rows, ids)) = words {
            let mut own = rows.forward(&row).map_err(err("own words"))?;
            if let Some(s) = &self.suppressed_logits {
                own = Tensor::new(ids, s.device())
                    .and_then(|i| s.index_select(&i, 0))
                    .and_then(|s| s.to_dtype(own.dtype()))
                    .and_then(|s| own.broadcast_add(&s))
                    .map_err(err("suppressed words"))?;
            }
            scores.extend(flat(own).map_err(err("own words"))?);
        }
        let best = crate::inference::sampling::argmax(&scores) as usize;
        match words {
            Some((_, ids)) if best >= n => ids.get(best - n).copied().ok_or_else(|| {
                SwarmError::Internal("next-token head: a word past its rows".into())
            }),
            _ => Ok(best as u32),
        }
    }

    /// One forward of the head over `tokens`, each beside the hidden state of
    /// the position before it (`hidden`, `[1, n, hidden]`), at the cache's
    /// next positions. Returns the head's own normed rows — the hidden state
    /// a next draft step reads — and the last of them (`[1, hidden]`), which
    /// [`Self::mtp_guess`] reads.
    pub(crate) fn mtp_forward(
        &self,
        cache: &mut MtpCache,
        tokens: &[u32],
        hidden: &Tensor,
        kv_reserve: usize,
        for_the_owner: bool,
    ) -> Result<(Tensor, Tensor), SwarmError> {
        let head = self
            .mtp
            .as_ref()
            .ok_or_else(|| SwarmError::Internal("no next-token head".into()))?;
        let n = tokens.len();
        if n == 0 || hidden.dim(1).ok() != Some(n) {
            return Err(SwarmError::Internal(format!(
                "next-token head: {n} tokens beside {:?} hidden rows",
                hidden.dims()
            )));
        }
        let err = |what: &str| {
            let what = what.to_string();
            move |e: candle_core::Error| {
                SwarmError::Internal(format!("next-token head {what}: {e}"))
            }
        };
        let ids: Vec<i64> = tokens.iter().map(|&t| i64::from(t)).collect();
        let embedded = self
            .embed_ids(&ids)?
            .to_device(&head.device)
            .map_err(err("embedding"))?;
        let hidden = hidden.to_device(&head.device).map_err(err("hidden"))?;
        let mask = if n > 1 {
            Some(
                self.causal_mask(n, cache.len + n)
                    .and_then(|m| m.to_device(&head.device))
                    .map_err(err("mask"))?,
            )
        } else {
            None
        };
        let max_seq_len = self.max_seq_len;
        let on_processor = usize::from(!head.device.is_cuda());
        let (out, logits_row) = crate::inference::cpu_pools::in_phase_pool(
            n,
            on_processor,
            for_the_owner,
            false,
            || {
                // llama.cpp: concat(enorm(embedding), hnorm(hidden)) → eh_proj.
                let joined = Tensor::cat(
                    &[
                        head.enorm.forward(&embedded).map_err(err("enorm"))?,
                        head.hnorm.forward(&hidden).map_err(err("hnorm"))?,
                    ],
                    2,
                )
                .map_err(err("concat"))?;
                let x = head.eh_proj.forward(&joined).map_err(err("eh_proj"))?;
                let (residual, x) = Residual::Ready(x)
                    .add_norm(&head.attention_norm)
                    .map_err(err("attn_norm"))?;
                let attn = head
                    .attn
                    .forward_attn(
                        &x,
                        mask.as_ref(),
                        cache.len,
                        &mut cache.kv,
                        max_seq_len,
                        kv_reserve,
                    )
                    .map_err(err("attention"))?;
                let (residual, normed) = Residual::pending(attn, residual)
                    .add_norm(&head.post_attention_norm)
                    .map_err(err("post_attention_norm"))?;
                let ffn_out = head
                    .ffn
                    .forward(&normed, &residual, None)
                    .map_err(err("ffn"))?;
                let out = Residual::pending(ffn_out, residual)
                    .into_tensor()
                    .and_then(|t| head.head_norm.forward(&t))
                    .map_err(err("head_norm"))?;
                let last = out.i((.., n - 1, ..)).map_err(err("last row"))?;
                Ok::<_, SwarmError>((out, last))
            },
        )?;
        cache.len += n;
        Ok((out, logits_row))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::quantized::GgmlDType;

    /// The hot rows and a reply's own words are copies of the whole head's
    /// rows, so each scores its word to the bit as the whole head does — a
    /// guess read off them can differ only where the best word is not among
    /// them.
    #[test]
    fn a_heads_rows_score_each_word_exactly_as_the_whole_head_does() -> candle_core::Result<()> {
        let dev = Device::Cpu;
        let w = Tensor::randn(0f32, 1.0, (10, 64), &dev)?;
        let whole = QMatMul::from_qtensor(QTensor::quantize(&w, GgmlDType::Q8_0)?)?;
        let x = Tensor::randn(0f32, 1.0, (1, 64), &dev)?;
        let all: Vec<f32> = whole.forward(&x)?.flatten_all()?.to_vec1()?;
        let first: Vec<f32> = head_rows(&whole, HeadRows::First(4))
            .expect("first rows")
            .forward(&x)?
            .flatten_all()?
            .to_vec1()?;
        assert_eq!(first, all[..4].to_vec());
        let own: Vec<f32> = head_rows(&whole, HeadRows::Of(&[7, 2]))
            .expect("own words")
            .forward(&x)?
            .flatten_all()?
            .to_vec1()?;
        assert_eq!(own, vec![all[7], all[2]]);
        assert!(
            head_rows(&whole, HeadRows::Of(&[10])).is_err(),
            "a word past the head"
        );
        assert!(
            head_rows(&whole, HeadRows::First(11)).is_err(),
            "more rows than the head"
        );
        Ok(())
    }
}
