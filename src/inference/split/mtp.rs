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

use candle_core::{Device, IndexOp, Tensor};
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

    /// One forward of the head over `tokens`, each beside the hidden state of
    /// the position before it (`hidden`, `[1, n, hidden]`), at the cache's
    /// next positions. Returns the head's own normed rows — the hidden state
    /// a next draft step reads — and the last row's logits (`[1, vocab]`).
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
        let logits = self.logits_from(
            &logits_row
                .to_device(&self.device)
                .map_err(err("to the head"))?,
        )?;
        Ok((out, logits))
    }
}
