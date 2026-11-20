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
/// it this 248,320-row head; on a card a whole guess took 3 ms. Measured on a
/// card twice and not kept: a whole model (#425 (b), 2026-11-07: the story
/// lost), and beside a split's first segment (#425 (a), 2026-11-09), where an
/// idle card clocks down between rounds and a guess cost 4.2-19.3 ms against
/// 2.9 back to back — the hot rows cut it, and no reply got faster
/// (`~/rig428/fieldh1`, two arms a side; the code replies kept fewer guesses).
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

/// The vocabulary's alphabet past its first `hot` ids: its one-character
/// entries that are printable ASCII or general punctuation (U+2010-U+2027,
/// U+2030-U+205E: quotes, dashes, the ellipsis). A guess on the processor
/// reads them beside the hot rows from a reply's start (`DraftWords` in
/// `model_worker`, #428 stage 4). The hot rows are the FIRST ids because a
/// byte-level vocabulary keeps its alphabet there and makes its merges most
/// frequent first; a SentencePiece vocabulary may keep it at the END —
/// Gemma 4's 262,144 hold every printable ASCII character at
/// 236,744-236,995, and those were 15 of the 19 words past the hot rows in
/// a code reply and 15 of the 32 in a story (`,` alone eight times). FR-Spec
/// (ACL 2025) ranks a draft vocabulary by corpus frequency; this file states
/// none (Gemma's scores are all -1000), and the alphabet is the part of it
/// every text needs. Empty where the alphabet sits inside the hot rows
/// (Qwen, Llama 3: the first 256 ids).
pub(crate) fn alphabet_past(vocabulary: &[String], hot: usize) -> Vec<u32> {
    let wanted = |c: char| {
        c.is_ascii_graphic()
            || ('\u{2010}'..='\u{2027}').contains(&c)
            || ('\u{2030}'..='\u{205E}').contains(&c)
    };
    vocabulary
        .iter()
        .enumerate()
        .skip(hot)
        .filter(|(_, piece)| {
            let mut chars = piece.chars();
            matches!((chars.next(), chars.next()), (Some(c), None) if wanted(c))
        })
        .filter_map(|(id, _)| u32::try_from(id).ok())
        .collect()
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
    /// The trunk's output norm, where the head sits beside a split's first
    /// segment ([`HeadBeside::FirstSegment`]), which loads none of its own:
    /// the trunk's last-layer rows are normed by it before the head reads
    /// them ([`SplitModel::mtp_hidden_from`]). `None` beside the whole model.
    pub(super) trunk_norm: Option<RmsNorm>,
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

/// `SWARMLLM_SPLIT_GUESS_HEAD=0`: no guess head is loaded beside a split's
/// first segment (#425 (a)), for A/B inside one binary.
pub(crate) fn split_guess_head_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("SWARMLLM_SPLIT_GUESS_HEAD").as_deref() != Ok("0"))
}

/// `SWARMLLM_SPLIT_GUESS_HEAD_IN_RAM=0`: a guess head beside a split's first
/// segment whose card has no room for it is not loaded, as before #485 —
/// for A/B inside one binary.
pub(crate) fn split_guess_head_in_ram_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("SWARMLLM_SPLIT_GUESS_HEAD_IN_RAM").as_deref() != Ok("0"))
}

/// Where a guess head beside a split's first segment loads (#485).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SplitHeadHome {
    /// On the card, beside the segment's last layer.
    Card,
    /// In the machine's memory, the output head it reads with it.
    SystemMemory,
    /// Nowhere.
    NoRoom,
}

/// Where a guess head beside a split's first segment, whose last layer is on
/// a card, loads (#485): on the card where its block takes at most a quarter
/// of the card's conversation budget (`card.0`) AND one conversation of the
/// model's whole context (`card.1`) still fits beside the block and the output
/// head it reads (`block_drafter::fits_beside_one_conversation`'s rule) —
/// both checks it met before, one at load and one at attach; else in the
/// machine's memory where that is KNOWN (`machine`, live) and the block, the
/// head and the head's hot rows take at most a quarter of it (the quarter the
/// block was always held to); else not at all. A planner fills a first
/// segment's card with layers, so the card almost never had the room: every
/// 27B chain from this node's 8 GB card that ended on a peer ran unguessed
/// (2026-11-20: a 227 MB block against a 127 MB budget). llama.cpp places a
/// draft model on devices of its own the same way (`--spec-draft-device`,
/// `--spec-draft-ngl`); the block drafter's `drafter_home` (#451) is the same
/// rule for it.
pub(crate) fn split_head_home(
    block: u64,
    head: u64,
    hot: u64,
    card: (Option<u64>, u64),
    machine: Option<u64>,
) -> SplitHeadHome {
    let on_card = card.0.is_none_or(|budget| {
        block.saturating_mul(4) <= budget
            && block.saturating_add(head).saturating_add(card.1) <= budget
    });
    if on_card {
        return SplitHeadHome::Card;
    }
    let takes = block.saturating_add(head).saturating_add(hot);
    if split_guess_head_in_ram_on() && machine.is_some_and(|room| takes.saturating_mul(4) <= room) {
        SplitHeadHome::SystemMemory
    } else {
        SplitHeadHome::NoRoom
    }
}

/// Does a split's next-token block load, `here` bytes beside its device and
/// `beside` of its routed experts in the machine's memory? Sent to the
/// machine's memory ([`split_head_home`]) it was weighed there with its head;
/// with no room anywhere it does not load. On the card — its home, or no home
/// weighed — the card takes a quarter of its budget at most, and experts kept
/// beside it need the machine's conversation budget to come off: with none
/// they would sit there uncharged. A card home weighs only the card, so it
/// still asks that of its experts (#485's review: a MoE block's 855 MB of
/// experts would have come off a smaller machine budget unchecked).
pub(crate) fn guess_block_fits(
    home: Option<SplitHeadHome>,
    on_card: bool,
    (here, beside): (u64, u64),
    card_budget: Option<u64>,
    host_budget: Option<u64>,
) -> bool {
    let quarter =
        |bytes: u64, budget: Option<u64>| budget.is_none_or(|b| bytes.saturating_mul(4) <= b);
    match home {
        Some(SplitHeadHome::SystemMemory) => true,
        Some(SplitHeadHome::NoRoom) => false,
        Some(SplitHeadHome::Card) | None if on_card => {
            quarter(here, card_budget)
                && (beside == 0 || host_budget.is_some_and(|h| beside.saturating_mul(4) <= h))
        }
        Some(SplitHeadHome::Card) | None if host_budget.is_some() => quarter(here, host_budget),
        Some(SplitHeadHome::Card) | None => quarter(here, card_budget),
    }
}

/// What the hot rows of an output head `info` names would take, read on the
/// processor ([`hot_rows_for`]'s rows, sized off the tensor table before the
/// head is read): 0 where a guess reads the whole head.
pub(crate) fn hot_rows_bytes_of(info: &candle_core::quantized::gguf_file::TensorInfo) -> u64 {
    let Some(rows) = draft_hot_rows() else {
        return 0;
    };
    let dims = info.shape.dims();
    let [vocab, cols] = dims[..] else {
        return 0;
    };
    let dtype = info.ggml_dtype;
    if rows >= vocab || cols % dtype.block_size() != 0 {
        return 0;
    }
    (rows * (cols / dtype.block_size() * dtype.type_size())) as u64
}

/// Where a load may read the file's next-token-prediction block.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum HeadBeside {
    /// The segment is the whole model: its own replies read the head (#425).
    WholeModel,
    /// A split's first segment, from layer 0, with the parts holding the head,
    /// the output norm and the output head on disk here (#425 (a)): the
    /// split's guesses read it, beside the token table they need, fed the
    /// trunk's last layer by the segment that runs it — where a split's block
    /// drafter sits (#428 stage 3).
    FirstSegment,
    /// Neither.
    Nowhere,
}

/// The index of the file's next-token-prediction block where this load reads
/// it, else `None`: the switch is on, the segment is the WHOLE model or a
/// split's first segment beside the head's parts (`beside`), the family is
/// Qwen 3.5's (`qwen35`, `qwen35moe` — llama.cpp's `graph_mtp` is written for
/// these two), the file declares exactly one block (`nextn_layers`), and that
/// block is the shape the head runs: a full-attention layer with `eh_proj`,
/// `enorm` and `hnorm`, and neither a token table nor an output head of its
/// own (llama.cpp reads those where present; no file here carries them).
pub(crate) fn block_to_load(
    ct: &candle_core::quantized::gguf_file::Content,
    arch: &crate::inference::model_arch::ModelArch,
    nextn_layers: u32,
    beside: HeadBeside,
    layer_end: usize,
    block_count: usize,
) -> Option<usize> {
    use crate::inference::model_arch::ModelArch;
    let placed = match beside {
        HeadBeside::WholeModel => layer_end == block_count,
        HeadBeside::FirstSegment => split_guess_head_on() && layer_end < block_count,
        HeadBeside::Nowhere => false,
    };
    if !mtp_on() || !placed || !matches!(arch, ModelArch::Qwen35 | ModelArch::Qwen35Moe) {
        return None;
    }
    // Said, as llama.cpp's load prints `n_nextn_layer`: unsloth's
    // Qwen3.5-9B Q4_K_M leaves the block out where bartowski's keeps it, and
    // a tester's run could not tell whether the head had been used (field
    // report 2026-11-08, FUTURE_WORK #431).
    if nextn_layers != 1 {
        tracing::info!(
            nextn_layers,
            "this file carries no guess head of its own (its upload left the next-token-prediction \
             block out); replies guess from the conversation's own text only"
        );
        return None;
    }
    if !head_in_shape(block_count, |name| ct.tensor_infos.contains_key(name)) {
        tracing::info!(
            block = block_count,
            "next-token-prediction block not in the shape the head runs; not loaded"
        );
        return None;
    }
    Some(block_count)
}

/// Does a file whose trunk is `block_count` blocks carry, as block
/// `block_count`, a next-token-prediction block in the shape the head runs?
/// Read off its tensor names (`has`), so a header's tensor table answers as
/// the loaded file does — [`block_to_load`] and the drafter's fetch
/// (`model::huggingface::block_drafter`) both ask it.
pub(crate) fn head_in_shape(block_count: usize, has: impl Fn(&str) -> bool) -> bool {
    let at = |s: &str| has(&format!("blk.{block_count}.{s}"));
    [
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
    .all(|s| at(s))
        && !at("nextn.embed_tokens.weight")
        && !at("nextn.shared_head_head.weight")
}

/// What block `block`'s weights take: `(on its device, in the machine's
/// memory beside a card)` — every `blk.{block}` tensor, its routed experts in
/// the second where a card-held block keeps them there (`experts_beside`, as
/// `LayerPlacement::for_the_next_token_block` placed them).
pub(crate) fn block_bytes(
    ct: &candle_core::quantized::gguf_file::Content,
    block: usize,
    experts_beside: bool,
) -> (u64, u64) {
    ct.tensor_infos
        .iter()
        .filter(|(name, _)| {
            matches!(super::segment_tensor(name), super::SegmentTensor::Layer(i) if i == block)
        })
        .fold((0, 0), |(here, beside), (name, info)| {
            let bytes = super::tensor_byte_size(info);
            if experts_beside && name.contains("_exps.") {
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

/// A guess's scores, one row a guess, and the words past the hot rows
/// (`SplitModel::mtp_score_table`).
type ScoreTable<'w> = (Vec<Vec<f32>>, Option<(usize, &'w [u32])>);

/// The word at index `i` of a guess's score table
/// (`SplitModel::mtp_score_table`): itself, or past the hot rows one of the
/// reply's own words.
fn word_at(i: usize, past: Option<(usize, &[u32])>) -> Result<u32, SwarmError> {
    match past {
        Some((n, ids)) if i >= n => ids
            .get(i - n)
            .copied()
            .ok_or_else(|| SwarmError::Internal("next-token head: a word past its rows".into())),
        _ => Ok(i as u32),
    }
}

impl SplitModel {
    /// The layer a forward keeps for the head: the trunk's last (absolute) —
    /// only where this model ends the trunk; a head beside a split's first
    /// segment reads that layer's rows off the split instead
    /// ([`Self::split_head_reads_layer`]).
    pub(crate) fn mtp_capture_layer(&self) -> Option<usize> {
        self.mtp
            .as_ref()
            .filter(|_| self.output.is_some())
            .and_then(|_| self.layer_end.checked_sub(1))
    }

    /// The trunk layer whose rows a guess head beside this split's first
    /// segment reads (#425 (a), [`HeadBeside::FirstSegment`]): the model's
    /// last, which another segment runs. `None` where no head sits here so.
    pub(crate) fn split_head_reads_layer(&self) -> Option<usize> {
        self.mtp
            .as_ref()
            .filter(|h| h.trunk_norm.is_some() && self.output.is_none())
            .and_then(|_| self.total_layers.checked_sub(1))
    }

    /// A guess head beside this split's first segment whose block runs in the
    /// machine's memory beside a model on a card (#485) — sent there for want
    /// of room on the card ([`split_head_home`]), or placed beside a last
    /// layer the processor holds. Its output head is read there too where no
    /// block drafter needs it on the card (`shards`), and its guesses are
    /// scored where that head is ([`Self::guess_scoring_device`]). Read onto
    /// the card instead, the 27B's 994 MB head found no room beside 20 layers
    /// on this node's 8 GB card and the head guessed nothing (2026-11-20).
    pub(crate) fn guess_head_in_system_memory(&self) -> bool {
        self.split_head_reads_layer().is_some()
            && self.mtp.as_ref().is_some_and(|h| !h.device.is_cuda())
            && self.device.is_cuda()
    }

    /// Is the output head a guess head beside a split's first segment reads
    /// (`drafter_head`) in the machine's memory, beside a model on a card?
    pub(crate) fn split_head_reads_from_system_memory(&self) -> bool {
        self.device.is_cuda()
            && self
                .drafter_head
                .as_ref()
                .and_then(QMatMul::quantized_weight)
                .is_some_and(|head| !head.device().is_cuda())
    }

    /// Where a guess's rows are scored: on the device of the head that scores
    /// them — the model's own head lives on its main device; a borrowed one
    /// (`drafter_head`) where it was read, the machine's memory beside a guess
    /// head sent there ([`Self::guess_head_in_system_memory`]).
    fn guess_scoring_device(&self) -> Device {
        match (
            &self.output,
            self.drafter_head
                .as_ref()
                .and_then(QMatMul::quantized_weight),
        ) {
            (None, Some(head)) => head.device().clone(),
            _ => self.device.clone(),
        }
    }

    /// The head's hidden-state input from rows a forward kept after the
    /// trunk's last layer: normed as the output head reads them
    /// (llama.cpp's `t_h_nextn`) — by the model's own output norm, else, beside
    /// a split's first segment, the head's copy of it.
    pub(crate) fn mtp_hidden_from(&self, kept: &Tensor) -> Result<Tensor, SwarmError> {
        let err = |e: candle_core::Error| {
            SwarmError::Internal(format!("next-token head: hidden norm: {e}"))
        };
        // The model's norm lives on its main device (`output_on_primary_device`);
        // the head's copy on the head's.
        if let Some(norm) = self.norm.as_ref() {
            return kept
                .to_device(&self.device)
                .and_then(|k| norm.forward(&k))
                .map_err(err);
        }
        let (norm, device) = self
            .mtp
            .as_ref()
            .and_then(|h| h.trunk_norm.as_ref().map(|n| (n, &h.device)))
            .ok_or_else(|| SwarmError::Internal("next-token head: no output norm".into()))?;
        kept.to_device(device)
            .and_then(|k| norm.forward(&k))
            .map_err(err)
    }

    /// The output head's logits for a guess's rows: the model's own head
    /// ([`Self::logits_from`]), else the one a split's first segment borrowed
    /// (`drafter_head`), with the file's suppressed words as `logits_from`
    /// adds them (#282). A block drafter's guesses read it the same way.
    fn guess_logits(&self, rows: &Tensor) -> Result<Tensor, SwarmError> {
        let (None, Some(head)) = (&self.output, &self.drafter_head) else {
            return self.logits_from(rows);
        };
        let err = |e: candle_core::Error| SwarmError::Internal(format!("borrowed head: {e}"));
        let logits = head.forward(rows).map_err(err)?;
        match &self.suppressed_logits {
            // The row lives on the model's main device; a borrowed head may
            // score in the machine's memory (#485).
            Some(s) => s
                .to_dtype(logits.dtype())
                .and_then(|row| row.to_device(logits.device()))
                .and_then(|row| logits.broadcast_add(&row))
                .map_err(err),
            None => Ok(logits),
        }
    }

    /// The hot rows a guess reads first — the guess head's (`MtpHead::hot`),
    /// else a block drafter's (`SplitModel::drafter_hot`) — where there are.
    fn hot_head(&self) -> Option<&QMatMul> {
        self.mtp
            .as_ref()
            .and_then(|h| h.hot.as_ref())
            .or(self.drafter_hot.as_ref())
    }

    /// How many words a block drafter's OWN hot rows hold
    /// (`SplitModel::drafter_hot`) — where no guess head keeps them, its
    /// guesses keep their own words ([`Self::mtp_rows_of`]).
    pub(crate) fn block_hot_rows(&self) -> Option<usize> {
        self.drafter_hot.as_ref()?.out_features().ok()
    }

    /// How many words the hot rows hold ([`Self::hot_head`]).
    pub(crate) fn mtp_hot_rows(&self) -> Option<usize> {
        self.hot_head()?.out_features().ok()
    }

    /// [`alphabet_past`] the first `hot` ids of this model's vocabulary —
    /// none where it has no vocabulary.
    pub(crate) fn alphabet_past(&self, hot: usize) -> Vec<u32> {
        self.vocabulary
            .as_deref()
            .map_or_else(Vec::new, |v| alphabet_past(v, hot))
    }

    /// The output head's rows for `ids` — a conversation's own words past
    /// the hot rows — for a guess to read beside them.
    pub(crate) fn mtp_rows_of(&self, ids: &[u32]) -> Result<QMatMul, SwarmError> {
        let output = self
            .output
            .as_ref()
            .or(self.drafter_head.as_ref())
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
        self.mtp_guesses(row, over)?
            .first()
            .copied()
            .ok_or_else(|| SwarmError::Internal("next-token head: no row to read".into()))
    }

    /// The best word for each of `rows` (normed, `[.., hidden]`, one guess
    /// a row), over the words `over` names — [`Self::mtp_guess`] for one row,
    /// and a block drafter's guesses on the processor (#428): its rows read
    /// the hot rows in ONE product, as the head's guess reads them.
    pub(crate) fn mtp_guesses(
        &self,
        rows: &Tensor,
        over: GuessOver<'_>,
    ) -> Result<Vec<u32>, SwarmError> {
        let (scores, past) = self.mtp_score_table(rows, over)?;
        scores
            .iter()
            .map(|row| word_at(crate::inference::sampling::argmax(row) as usize, past))
            .collect()
    }

    /// Each of `rows`' `k` likeliest words over what `over` names, likeliest
    /// first, as (word, log-probability among them) — [`Self::mtp_guesses`]'
    /// scores ranked, so a row's first is its guess: what a block drafter's
    /// tree is grown from on the processor (#437). A word the hot rows and a
    /// reply's own words both name is offered once, at its better place.
    /// `None` where a row has nothing finite.
    pub(crate) fn mtp_ranked(
        &self,
        rows: &Tensor,
        over: GuessOver<'_>,
        k: usize,
    ) -> Result<Option<super::draft::RankedRows>, SwarmError> {
        let (scores, past) = self.mtp_score_table(rows, over)?;
        // Ranked past `k` where the reply's own words could repeat a hot one.
        let wider = past.map_or(k, |(_, ids)| k + ids.len().min(k));
        let Some(ranked) = super::draft::likeliest_of_each(&scores, wider) else {
            return Ok(None);
        };
        ranked
            .into_iter()
            .map(|row| {
                let mut seen = std::collections::HashSet::with_capacity(row.len());
                let mut words = Vec::with_capacity(k);
                for (i, logp) in row {
                    let w = word_at(i as usize, past)?;
                    if seen.insert(w) {
                        words.push((w, logp));
                    }
                    if words.len() == k {
                        break;
                    }
                }
                Ok(words)
            })
            .collect::<Result<_, _>>()
            .map(Some)
    }

    /// The scores [`Self::mtp_guesses`] and [`Self::mtp_ranked`] read, one
    /// row a guess: the whole head's, or the hot rows' and then a reply's own
    /// words' — the file's suppressed words added to both (#282) — with the
    /// words past the hot rows (`(n, ids)`: index `n + j` is `ids[j]`).
    fn mtp_score_table<'w>(
        &self,
        rows: &Tensor,
        over: GuessOver<'w>,
    ) -> Result<ScoreTable<'w>, SwarmError> {
        let err = |what: &'static str| {
            move |e: candle_core::Error| {
                SwarmError::Internal(format!("next-token head {what}: {e}"))
            }
        };
        let hidden = rows.dim(D::Minus1).map_err(err("rows"))?;
        let rows = rows
            .to_device(&self.guess_scoring_device())
            .and_then(|r| r.reshape(((), hidden)))
            .map_err(err("to the output head"))?;
        let table = |t: Tensor| t.to_dtype(DType::F32).and_then(|t| t.to_vec2::<f32>());
        let GuessOver::HotAnd(words) = over else {
            let logits = self.guess_logits(&rows)?;
            return Ok((table(logits).map_err(err("logits"))?, None));
        };
        let hot = self
            .hot_head()
            .ok_or_else(|| SwarmError::Internal("next-token head: no hot rows".into()))?;
        let mut logits = hot.forward(&rows).map_err(err("hot rows"))?;
        let n = logits.dim(D::Minus1).map_err(err("hot rows"))?;
        if let Some(s) = &self.suppressed_logits {
            logits = s
                .narrow(0, 0, n)
                .and_then(|s| s.to_dtype(logits.dtype()))
                .and_then(|s| s.to_device(logits.device()))
                .and_then(|s| logits.broadcast_add(&s))
                .map_err(err("suppressed words"))?;
        }
        let mut scores = table(logits).map_err(err("hot rows"))?;
        if let Some((own_rows, ids)) = words {
            let mut own = own_rows.forward(&rows).map_err(err("own words"))?;
            if let Some(s) = &self.suppressed_logits {
                own = Tensor::new(ids, s.device())
                    .and_then(|i| s.index_select(&i, 0))
                    .and_then(|s| s.to_dtype(own.dtype()))
                    .and_then(|s| s.to_device(own.device()))
                    .and_then(|s| own.broadcast_add(&s))
                    .map_err(err("suppressed words"))?;
            }
            for (row, own) in scores.iter_mut().zip(table(own).map_err(err("own words"))?) {
                row.extend(own);
            }
        }
        Ok((scores, words.map(|(_, ids)| (n, ids))))
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

    /// The alphabet past the hot rows is the vocabulary's one-character
    /// printable ASCII and punctuation entries there, and nothing inside them
    /// (#428 stage 4: Gemma 4 keeps `,` at 236,764).
    #[test]
    fn the_alphabet_past_the_hot_rows_is_its_one_character_text() {
        let vocab: Vec<String> = [
            "a", ".", "▁the", ",", "’", "ab", "\u{200b}", "漢", "▁", "\n", "…", "",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        assert_eq!(alphabet_past(&vocab, 2), vec![3, 4, 10]);
        assert_eq!(alphabet_past(&vocab, 0), vec![0, 1, 3, 4, 10]);
        assert!(alphabet_past(&vocab, vocab.len()).is_empty());
    }

    /// A guess head beside a split's first segment stays on the card wherever
    /// it fitted there before, and goes to the machine's memory only where the
    /// card has no room for the block or for the output head beside one
    /// conversation — with the Qwen3.8-27B's figures from this node's 8 GB card
    /// (2026-11-20: a 227 MB block, its 994 MB head and 131 MB of hot rows
    /// against a 127 MB card budget), which loaded no head at all before #485.
    #[test]
    fn a_card_home_still_asks_the_machines_budget_for_the_blocks_experts() {
        const MB: u64 = 1 << 20;
        let card = Some(8_000 * MB);
        // Qwen3.6-35B-A3B's block on a big card, its 855 MB of experts beside
        // it, a machine budget of 1 GB: the experts do not fit there.
        let (here, beside) = (60 * MB, 855 * MB);
        assert!(!guess_block_fits(
            Some(SplitHeadHome::Card),
            true,
            (here, beside),
            card,
            Some(1_024 * MB)
        ));
        assert!(guess_block_fits(
            Some(SplitHeadHome::Card),
            true,
            (here, beside),
            card,
            Some(4_000 * MB)
        ));
        assert!(
            !guess_block_fits(Some(SplitHeadHome::Card), true, (here, beside), card, None),
            "experts beside a card with no machine budget would sit uncharged"
        );
        assert!(guess_block_fits(
            Some(SplitHeadHome::SystemMemory),
            false,
            (here, 0),
            Some(MB),
            None
        ));
        assert!(!guess_block_fits(
            Some(SplitHeadHome::NoRoom),
            true,
            (here, 0),
            card,
            None
        ));
    }

    #[test]
    fn a_split_guess_head_goes_to_the_machines_memory_only_where_its_card_has_no_room() {
        const MB: u64 = 1 << 20;
        let (block, head, hot, one) = (227 * MB, 994 * MB, 131 * MB, 96 * MB);
        let machine = Some(40_000 * MB);
        let home = |card: Option<u64>, machine: Option<u64>| {
            split_head_home(block, head, hot, (card, one), machine)
        };
        // Room on the card, as before: the card, whatever the machine has.
        assert_eq!(home(Some(8_000 * MB), machine), SplitHeadHome::Card);
        assert_eq!(home(Some(8_000 * MB), None), SplitHeadHome::Card);
        assert_eq!(home(None, None), SplitHeadHome::Card);
        // The 27B's first segment: no room on the card.
        assert_eq!(home(Some(127 * MB), machine), SplitHeadHome::SystemMemory);
        // The block's quarter fits, the head beside one conversation does not.
        assert_eq!(home(Some(1_000 * MB), machine), SplitHeadHome::SystemMemory);
        // The machine is held to the same quarter, with the hot rows counted.
        let takes = block + head + hot;
        assert_eq!(
            home(Some(127 * MB), Some(4 * takes)),
            SplitHeadHome::SystemMemory
        );
        assert_eq!(
            home(Some(127 * MB), Some(4 * takes - 1)),
            SplitHeadHome::NoRoom
        );
        // A machine whose room is not known is never handed it.
        assert_eq!(home(Some(127 * MB), None), SplitHeadHome::NoRoom);
    }

    /// The hot rows weighed off the tensor table before a head is read are
    /// what the head read on the processor keeps ([`hot_rows_for`]).
    #[test]
    fn a_heads_hot_rows_are_sized_off_its_tensor_table_as_the_read_head_keeps_them(
    ) -> candle_core::Result<()> {
        let rows = draft_hot_rows().expect("hot rows on by default");
        let vocab = rows + 100;
        let w = Tensor::zeros((vocab, 64), DType::F32, &Device::Cpu)?;
        let q = QTensor::quantize(&w, GgmlDType::Q8_0)?;
        let info = candle_core::quantized::gguf_file::TensorInfo {
            ggml_dtype: GgmlDType::Q8_0,
            shape: q.shape().clone(),
            offset: 0,
        };
        let head = QMatMul::from_qtensor(q)?;
        let kept = hot_rows_for(Some(&head), false).expect("hot rows of a wide head");
        assert_eq!(hot_rows_bytes_of(&info), kept.1);
        assert_eq!(kept.1, (rows * 2 * 34) as u64);
        // A head no wider than the hot rows is read whole: nothing extra.
        let narrow = candle_core::quantized::gguf_file::TensorInfo {
            ggml_dtype: GgmlDType::Q8_0,
            shape: candle_core::Shape::from((rows, 64)),
            offset: 0,
        };
        assert_eq!(hot_rows_bytes_of(&narrow), 0);
        Ok(())
    }
}
