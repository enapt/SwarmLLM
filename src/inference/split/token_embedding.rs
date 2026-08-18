//! The token-embedding table, and how a row of it is read.
//!
//! `token_embd.weight` arrives quantized in every GGUF we serve. Until
//! 2026-08-18 the loader dequantized the whole table to [`EMBEDDING_DTYPE`] and
//! held that dense copy for the worker's life, because `Embedding::new` takes a
//! dense tensor. On a large-vocabulary model that is the single largest
//! allocation a worker makes — and an embedding lookup reads at most `seq_len`
//! of its rows, one per decode step.
//!
//! Measured on the headers of the models this repo tests against:
//!
//! | model | table today | quantized | saved |
//! |---|---|---|---|
//! | llama-3.2-3b-q4-k-m (tied) | 751 MB f16 + 308 MB Q6_K | 308 MB | **751 MB** |
//! | llama-3.2-1b-q8-0 (tied) | 501 MB f16 + 266 MB Q8_0 | 266 MB | **501 MB** |
//! | meta-llama-3.1-8b-q4-k-m | 1002 MB f16 | 282 MB Q4_K | **720 MB** |
//! | phi-3.5-mini-q4-k-m | 188 MB f16 | 53 MB Q4_K | **135 MB** |
//!
//! The tied rows save more than the difference because a weight-tied model
//! loaded `token_embd.weight` **twice** — once dequantized for the lookup, once
//! quantized for its LM head. [`super::loader`] now loads it once and shares the
//! `Arc<QTensor>` with `QMatMul::from_arc`, so those rows collapse to one copy.
//!
//! This is what llama.cpp does (`ggml_get_rows` on a quantized source, and the
//! `k_get_rows_kq` CUDA kernel added for it).
//!
//! **The result is bit-identical to dequantizing the whole table.** Block
//! dequantization is per-block, a row is a whole number of blocks (enforced by
//! [`rows_are_block_aligned`]), and candle's CPU `dequantize_f16` is
//! `dequantize()?.to_dtype(F16)?` — an elementwise cast. So gathering rows then
//! dequantizing and casting produces exactly the bytes the old path produced.
//! `quantized_rows_match_dequantizing_the_whole_table` asserts that rather than
//! a tolerance, which is what makes this safe to enable by default.
//!
//! **The trap: the gather must stay where the table lives.** `QTensor::data()`
//! is a zero-copy borrow on CPU but a full device-to-host copy on CUDA, so the
//! CPU implementation reused on a GPU would move the whole table across PCIe on
//! every decoded token. llama.cpp hit precisely this — with the lookup "kicked
//! out of the graph" a Qwen3-1.7B decoded at 6.18 ms/token against 1.72 once it
//! was done on-device.
//!
//! So each device has its own gather, both behind `QTensor::gather_rows` (a
//! vendored candle patch). CUDA runs `index_select` over a `[vocab, row_bytes]`
//! byte view of the quantized buffer — no new kernel, because `is_u32_u8` is
//! already instantiated in `candle-kernels` and the buffer's padding is only
//! ever trailing, so rows are contiguous. Metal has no implementation and keeps
//! the dense table.
//!
//! **The acceptance test for the CUDA half is decode rate, not memory.** The
//! failure mode above frees exactly as much memory while being far slower, so a
//! memory reading alone cannot tell the two apart.

use std::sync::Arc;

use candle_core::quantized::QTensor;
use candle_core::{Device, Tensor};
use candle_nn::{Embedding, Module};

use super::loader::EMBEDDING_DTYPE;

/// How a segment reads its token-embedding table.
///
/// Both variants return [`EMBEDDING_DTYPE`], so every call site is identical
/// and the choice is invisible above this type.
pub(crate) enum TokenEmbedding {
    /// The whole table, dequantized at load. Used for a GGUF that ships an
    /// unquantized `token_embd.weight` (there is nothing to save), for a row
    /// length that is not block-aligned, and on Metal.
    Dense(Embedding),
    /// The table left quantized, with rows dequantized as they are looked up.
    Quantized(QuantizedRows),
}

impl TokenEmbedding {
    /// Build the row-gathering form, or `None` when this table cannot use it.
    ///
    /// Returning `None` is never an error — the caller dequantizes as before,
    /// which is always correct and merely larger.
    pub(crate) fn try_quantized(table: &Arc<QTensor>, device: &Device) -> Option<Self> {
        let dims = table.shape().dims();
        let dtype = table.dtype();
        if !rows_on_demand_eligible(device, dtype, dims) {
            return None;
        }
        let hidden = match *dims {
            [_vocab, hidden] => hidden,
            _ => return None,
        };
        Some(Self::Quantized(QuantizedRows {
            table: Arc::clone(table),
            hidden,
        }))
    }

    /// Look up `ids`, returning `ids.shape() ++ [hidden]` at [`EMBEDDING_DTYPE`].
    pub(crate) fn forward(&self, ids: &Tensor) -> candle_core::Result<Tensor> {
        match self {
            Self::Dense(e) => e.forward(ids),
            Self::Quantized(q) => q.forward(ids),
        }
    }

    /// Bytes this table holds resident, for the log line that reports placement.
    pub(crate) fn resident_bytes(&self) -> usize {
        match self {
            Self::Dense(e) => e.embeddings().elem_count() * EMBEDDING_DTYPE.size_in_bytes(),
            Self::Quantized(q) => q.table.storage_size_in_bytes(),
        }
    }
}

/// Will this table's rows be read on demand rather than dequantized whole?
///
/// **The single answer to that question**, because two places have to agree on
/// it: the loader, which allocates, and the footprint estimators, which decide
/// whether a model is admitted at all. A disagreement here is invisible until a
/// node either refuses a model that would have fitted or is admitted and then
/// runs out of memory — the same trap `EMBEDDING_DTYPE` already carries a test
/// for. Never re-derive it from a device check at a call site.
pub(crate) fn rows_on_demand_eligible(
    device: &Device,
    dtype: candle_core::quantized::GgmlDType,
    dims: &[usize],
) -> bool {
    // Metal has no gather implementation, so it keeps the dense table. CPU and
    // CUDA each have one that stays on their own device.
    let device_can_gather = matches!(device, Device::Cpu | Device::Cuda(_));
    device_can_gather && table_supports_row_gather(dtype, dims)
}

/// `SWARMLLM_DENSE_EMBEDDING=1` restores the dequantize-the-whole-table
/// behaviour inside this binary.
///
/// The same discipline as `SWARMLLM_FORCE_STANDARD_ATTN` and
/// `SWARMLLM_DECODE_THREADS=0`: a memory or speed claim about this change has
/// to be measurable as an A/B on ONE binary, or the comparison is confounded by
/// build profile and whatever else moved between two of them. It is also the
/// escape hatch if a quantization type ever dequantizes differently row-wise
/// than in bulk.
fn dense_embedding_forced() -> bool {
    static FORCED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FORCED.get_or_init(|| {
        std::env::var("SWARMLLM_DENSE_EMBEDDING")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
    })
}

/// The device-independent half of [`rows_on_demand_eligible`]: can this table's
/// rows be addressed at all?
///
/// Split out because the footprint estimators are built once and consulted for
/// both a CPU and a CUDA worker, so they need the shape-and-dtype question
/// separately from the device question.
pub(crate) fn table_supports_row_gather(
    dtype: candle_core::quantized::GgmlDType,
    dims: &[usize],
) -> bool {
    // Checked HERE, in the innermost predicate, so both the loader and the
    // footprint estimators inherit it. Checking it one level up instead left
    // the estimator believing a gather would happen while the loader went
    // dense — the model would then be admitted against a figure ~750 MB below
    // what it actually takes, which is the precise disagreement this pair of
    // functions exists to prevent.
    if dense_embedding_forced() {
        return false;
    }
    let hidden = match *dims {
        [_vocab, hidden] => hidden,
        _ => return false,
    };
    let block = dtype.block_size();
    // An unquantized table has nothing to save, and its "blocks" are single
    // elements, so the gather would be a plain copy of the whole thing.
    if block <= 1 {
        return false;
    }
    rows_are_block_aligned(hidden, block)
}

/// Is a row of `hidden` elements a whole number of quantization blocks?
///
/// K-quants block 256 elements and ggml itself requires `ne0 % QK_K == 0`, so
/// this holds for every K-quantized table in practice — but a row that straddles
/// a block cannot be sliced out of the buffer at all, and answering that with a
/// silent wrong read rather than a fallback is the one way this optimisation
/// could corrupt an embedding. Checked rather than assumed.
pub(crate) fn rows_are_block_aligned(hidden: usize, block_size: usize) -> bool {
    block_size > 0 && hidden.is_multiple_of(block_size)
}

/// A quantized `token_embd.weight`, read a row at a time.
///
/// Row addressing (byte offsets, bounds checks, block arithmetic) lives in
/// `QTensor::gather_rows` rather than here, so the CPU and CUDA implementations
/// cannot disagree about where a row is.
pub(crate) struct QuantizedRows {
    table: Arc<QTensor>,
    hidden: usize,
}

impl QuantizedRows {
    fn forward(&self, ids: &Tensor) -> candle_core::Result<Tensor> {
        let id_shape = ids.dims().to_vec();
        // GGUF token ids reach us as u32 or i64 depending on the caller, and the
        // gather requires u32 on the table's own device — so normalise both
        // rather than requiring the caller to.
        let flat = ids.flatten_all()?;
        let flat = match flat.dtype() {
            candle_core::DType::U32 => flat,
            candle_core::DType::I64 => flat.to_dtype(candle_core::DType::U32)?,
            other => candle_core::bail!("embedding ids must be u32 or i64, got {other:?}"),
        };
        let flat = flat.to_device(&self.table.device())?;

        // One call for both devices. On CPU it slices rows out of a zero-copy
        // borrow; on CUDA it runs `index_select` over a byte view of the
        // quantized buffer, on the device — see the vendored
        // `QTensor::gather_rows`. Doing the CUDA case the CPU way would copy the
        // whole table host-ward on every lookup, which is the trap in the module
        // docs above.
        let rows = self.table.gather_rows(&flat)?;

        // `dequantize -> to_dtype(F16)` is exactly what `dequantize_f16` does on
        // CPU, and CUDA has a native f16 kernel; going through `dequantize_f16`
        // takes the better of the two per device without changing the result.
        let dense = rows.dequantize_f16(&self.table.device())?;
        debug_assert_eq!(dense.dtype(), EMBEDDING_DTYPE);

        let mut out_shape = id_shape;
        out_shape.push(self.hidden);
        dense.reshape(out_shape)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::quantized::GgmlDType;

    /// Build a small quantized table with distinguishable rows.
    fn table(vocab: usize, hidden: usize, dtype: GgmlDType) -> Arc<QTensor> {
        let data: Vec<f32> = (0..vocab * hidden)
            .map(|i| (i % 97) as f32 * 0.01 - 0.5)
            .collect();
        let dense = Tensor::from_vec(data, (vocab, hidden), &Device::Cpu).unwrap();
        Arc::new(QTensor::quantize(&dense, dtype).unwrap())
    }

    /// The property the whole change rests on: reading rows out of the
    /// quantized table gives byte-for-byte what dequantizing the entire table
    /// and then selecting those rows gives. Exact equality, not a tolerance —
    /// if this ever needs a tolerance the premise is wrong and the optimisation
    /// is unsafe.
    #[test]
    fn quantized_rows_match_dequantizing_the_whole_table() {
        for dtype in [GgmlDType::Q4K, GgmlDType::Q6K, GgmlDType::Q8_0] {
            let t = table(64, 256, dtype);
            let emb = TokenEmbedding::try_quantized(&t, &Device::Cpu)
                .unwrap_or_else(|| panic!("{dtype:?} should take the quantized path"));

            let ids = Tensor::from_vec(vec![7u32, 0, 63, 7, 31], (5,), &Device::Cpu).unwrap();
            let got = emb.forward(&ids).unwrap();

            // The path this replaces: whole table -> f16, then select.
            let whole = t.dequantize_f16(&Device::Cpu).unwrap();
            let want = whole.index_select(&ids, 0).unwrap();

            assert_eq!(got.dims(), want.dims(), "{dtype:?} shape");
            assert_eq!(
                got.flatten_all().unwrap().to_vec1::<half::f16>().unwrap(),
                want.flatten_all().unwrap().to_vec1::<half::f16>().unwrap(),
                "{dtype:?} rows must be bit-identical to the dequantized table"
            );
        }
    }

    /// A repeated id must produce the same row twice — the gather indexes by
    /// id, so an off-by-one in the byte range would still look plausible on a
    /// strictly increasing id list.
    #[test]
    fn a_repeated_id_yields_the_same_row() {
        let t = table(32, 256, GgmlDType::Q4K);
        let emb = TokenEmbedding::try_quantized(&t, &Device::Cpu).unwrap();
        let ids = Tensor::from_vec(vec![5u32, 5], (2,), &Device::Cpu).unwrap();
        let got = emb.forward(&ids).unwrap();
        let rows = got.to_vec2::<half::f16>().unwrap();
        assert_eq!(rows[0], rows[1]);
    }

    /// Shape in, shape out: a `[batch, seq]` id tensor must come back as
    /// `[batch, seq, hidden]`, matching `Embedding::forward`.
    #[test]
    fn batched_ids_keep_their_shape() {
        let t = table(32, 256, GgmlDType::Q4K);
        let emb = TokenEmbedding::try_quantized(&t, &Device::Cpu).unwrap();
        let ids = Tensor::from_vec(vec![1u32, 2, 3, 4, 5, 6], (2, 3), &Device::Cpu).unwrap();
        let got = emb.forward(&ids).unwrap();
        assert_eq!(got.dims(), &[2, 3, 256]);

        let whole = t.dequantize_f16(&Device::Cpu).unwrap();
        let want = whole
            .index_select(&ids.flatten_all().unwrap(), 0)
            .unwrap()
            .reshape((2, 3, 256))
            .unwrap();
        assert_eq!(
            got.flatten_all().unwrap().to_vec1::<half::f16>().unwrap(),
            want.flatten_all().unwrap().to_vec1::<half::f16>().unwrap()
        );
    }

    /// i64 ids are what the tokenizer path produces; they must work too.
    #[test]
    fn i64_ids_are_accepted() {
        let t = table(32, 256, GgmlDType::Q4K);
        let emb = TokenEmbedding::try_quantized(&t, &Device::Cpu).unwrap();
        let u = Tensor::from_vec(vec![9u32, 2], (2,), &Device::Cpu).unwrap();
        let i = Tensor::from_vec(vec![9i64, 2], (2,), &Device::Cpu).unwrap();
        assert_eq!(
            emb.forward(&u).unwrap().to_vec2::<half::f16>().unwrap(),
            emb.forward(&i).unwrap().to_vec2::<half::f16>().unwrap()
        );
    }

    /// An unquantized table has nothing to gain and must keep the dense path,
    /// so a node serving an F16 GGUF is completely unaffected by this change.
    #[test]
    fn an_unquantized_table_declines_the_quantized_path() {
        let data: Vec<f32> = (0..32 * 256).map(|i| i as f32).collect();
        let dense = Tensor::from_vec(data, (32, 256), &Device::Cpu).unwrap();
        let t = Arc::new(QTensor::quantize(&dense, GgmlDType::F16).unwrap());
        assert!(TokenEmbedding::try_quantized(&t, &Device::Cpu).is_none());
    }

    /// The gather slices whole rows out of one buffer, so a row that is not a
    /// whole number of blocks cannot be addressed. Refuse rather than read
    /// across a block boundary.
    #[test]
    fn a_row_that_straddles_a_block_is_refused() {
        assert!(rows_are_block_aligned(4096, 256));
        assert!(rows_are_block_aligned(3584, 256));
        assert!(!rows_are_block_aligned(896, 256));
        assert!(!rows_are_block_aligned(0, 0));
    }

    /// The CUDA gather must meet the SAME standard as the CPU one: the rows it
    /// returns are exactly what dequantizing the whole table and selecting
    /// those rows returns.
    ///
    /// Compared per device, because `dequantize_f16` uses a native f16 kernel on
    /// CUDA and a dequantize-then-cast on CPU. The property that matters is that
    /// each device reproduces ITS OWN previous behaviour — that is what makes
    /// this change invisible to a running node.
    ///
    /// Ignored without a GPU; run with `--ignored` on a CUDA machine.
    #[cfg(feature = "candle-cuda")]
    #[test]
    fn cuda_gathered_rows_match_dequantizing_the_whole_table() {
        let device = match Device::new_cuda(0) {
            Ok(d) => d,
            Err(_) => return, // no CUDA device present; nothing to assert
        };
        for dtype in [GgmlDType::Q4K, GgmlDType::Q6K, GgmlDType::Q8_0] {
            let vocab = 512;
            let hidden = 256;
            let data: Vec<f32> = (0..vocab * hidden)
                .map(|i| (i % 97) as f32 * 0.01 - 0.5)
                .collect();
            let dense = Tensor::from_vec(data, (vocab, hidden), &device).unwrap();
            let t = Arc::new(QTensor::quantize(&dense, dtype).unwrap());

            let emb = TokenEmbedding::try_quantized(&t, &device)
                .unwrap_or_else(|| panic!("{dtype:?} should take the quantized path on CUDA"));

            // Deliberately unsorted, with a repeat and both endpoints — a
            // gather that mixed up row offsets would still look plausible on a
            // strictly increasing list.
            let ids = Tensor::from_vec(vec![7u32, 0, 511, 7, 300, 1], (6,), &device).unwrap();
            let got = emb.forward(&ids).unwrap();

            let whole = t.dequantize_f16(&device).unwrap();
            let want = whole.index_select(&ids, 0).unwrap();

            assert_eq!(got.dims(), want.dims(), "{dtype:?} shape");
            assert_eq!(
                got.flatten_all().unwrap().to_vec1::<half::f16>().unwrap(),
                want.flatten_all().unwrap().to_vec1::<half::f16>().unwrap(),
                "{dtype:?} CUDA rows must be bit-identical to the dequantized table"
            );
        }
    }

    /// An id past the end of the vocabulary must be an error, not a read of
    /// whatever bytes follow the table.
    #[test]
    fn an_out_of_range_id_is_refused() {
        let t = table(16, 256, GgmlDType::Q4K);
        let emb = TokenEmbedding::try_quantized(&t, &Device::Cpu).unwrap();
        let ids = Tensor::from_vec(vec![16u32], (1,), &Device::Cpu).unwrap();
        assert!(emb.forward(&ids).is_err());
    }
}
