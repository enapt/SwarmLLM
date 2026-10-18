//! Gemma 4 E2B/E4B's per-layer inputs (FUTURE_WORK #115 (3c)).
//!
//! Every layer of these sizes adds, after its feed-forward, an embedding of
//! its own (`layers::PerLayerEmbedding`) gated by an INPUT built once per
//! forward from the prompt's tokens — llama.cpp `models/gemma4.cpp`
//! (`build_inp_per_layer`, `project_per_layer_inputs`), HF
//! `Gemma4TextModel.get_per_layer_inputs` / `project_per_layer_inputs`:
//!
//! ```text
//! rows   = per_layer_token_embd[token] · √n_embd_per_layer      [layers, width]
//! proj   = rms_norm(per_layer_model_proj · x₀ / √n_embd) · proj_norm   (per layer)
//! inputs = (proj + rows) / √2
//! ```
//!
//! `x₀` is the token embedding already scaled by `√n_embd`. Each layer reads
//! its own `width`-wide slice, so the rows of one layer are independent of
//! every other's — the norm is per slice.
//!
//! **The table stays in the machine's memory** (`gguf_meta::host_resident`):
//! `[layers × width, vocab]` is over half the E2B's file and a forward reads
//! one row of it a token, as llama.cpp keeps it (`TENSOR_READ_LAZY`, its input
//! layer on the processor). Rows of a quantized table are gathered and
//! dequantized straight to f32, as `ggml_get_rows` does — not through the f16
//! the token table's rows take (`token_embedding`), a rounding this input has
//! no reason to carry.
//!
//! Only the embedding step sees the tokens, so a model with these inputs runs
//! only where one computer holds it all (`GgufTensorMeta::runs_only_whole`).

use std::sync::Arc;

use candle_core::quantized::QTensor;
use candle_core::{DType, Device, Module, Tensor};

use super::token_embedding::{table_supports_row_gather, TokenEmbedding};
use super::QMatMul;
use crate::inference::residual_norm::RmsNorm;

/// What builds a forward's per-layer inputs — the first segment's, beside the
/// token embedding.
pub(crate) struct PerLayerInputs {
    table: PerLayerTable,
    /// `per_layer_model_proj`: `n_embd → layers × width`, on the model's device.
    model_proj: QMatMul,
    /// `per_layer_proj_norm`: one `width`-wide weight every layer's slice shares.
    proj_norm: RmsNorm,
    /// Every layer of the model — the table's rows hold one slice per layer.
    layers: usize,
    /// `{arch}.embedding_length_per_layer_input`.
    width: usize,
}

/// `per_layer_token_embd`, on the processor.
enum PerLayerTable {
    /// Rows gathered out of the quantized table and dequantized to f32.
    Quantized(Arc<QTensor>),
    /// An unquantized upload's table (F16/BF16), held at f16 like a dense
    /// token table — a gather would copy the whole of it.
    Dense(TokenEmbedding),
}

impl PerLayerInputs {
    /// `table` on the processor; the projection and norm on the model's device.
    pub(crate) fn new(
        table: QTensor,
        model_proj: QMatMul,
        proj_norm: RmsNorm,
        layers: usize,
        width: usize,
    ) -> candle_core::Result<Self> {
        let dims = table.shape().dims().to_vec();
        if dims != [dims[0], layers * width] {
            candle_core::bail!(
                "per_layer_token_embd is {dims:?}, not [vocab, {layers} layers × {width}]"
            );
        }
        let table = if table_supports_row_gather(table.dtype(), &dims) {
            PerLayerTable::Quantized(Arc::new(table))
        } else {
            let dense = table.dequantize_f16(&Device::Cpu)?;
            PerLayerTable::Dense(TokenEmbedding::Dense(candle_nn::Embedding::new(
                dense,
                layers * width,
            )))
        };
        Ok(Self {
            table,
            model_proj,
            proj_norm,
            layers,
            width,
        })
    }

    /// The forward's inputs for every layer, `[b, seq, layers, width]` at f32 on
    /// `x0`'s device: `ids` the prompt's tokens (`[b, seq]`), `x0` their
    /// embedding as the first layer reads it (`[b, seq, n_embd]`, scaled).
    pub(crate) fn forward(&self, ids: &Tensor, x0: &Tensor) -> candle_core::Result<Tensor> {
        let rows = self.rows_on(ids, x0.device())?;
        self.forward_from_rows(&rows, x0)
    }

    /// Each token's row of the table, scaled by `√width`, on `device` —
    /// `[tokens, layers × width]`. The table is in the machine's memory, so this
    /// is the forward's one host→device copy: a decode step recorded as a CUDA
    /// graph makes it BEFORE its capture (`executor::RecordedStep`), where a
    /// copy inside would be replayed from a stale host address (#295).
    pub(crate) fn rows_on(&self, ids: &Tensor, device: &Device) -> candle_core::Result<Tensor> {
        (self.rows(ids)? * (self.width as f64).sqrt())?.to_device(device)
    }

    /// [`Self::forward`] from rows [`Self::rows_on`] already put on `x0`'s
    /// device — every operation on that device.
    pub(crate) fn forward_from_rows(
        &self,
        rows: &Tensor,
        x0: &Tensor,
    ) -> candle_core::Result<Tensor> {
        let (b, seq, n_embd) = x0.dims3()?;
        let rows = rows.reshape((b, seq, self.layers, self.width))?;
        let proj = (self.model_proj.forward(x0)? * (1.0 / (n_embd as f64).sqrt()))?;
        let proj = self
            .proj_norm
            .forward(&proj.reshape((b, seq, self.layers, self.width))?)?;
        (proj + rows)? * std::f64::consts::FRAC_1_SQRT_2
    }

    /// Each token's row of the table, `[tokens, layers × width]` at f32 on the
    /// processor.
    fn rows(&self, ids: &Tensor) -> candle_core::Result<Tensor> {
        let flat = ids.flatten_all()?.to_device(&Device::Cpu)?;
        let flat = match flat.dtype() {
            DType::U32 => flat,
            DType::I64 => flat.to_dtype(DType::U32)?,
            other => candle_core::bail!("per-layer ids must be u32 or i64, got {other:?}"),
        };
        match &self.table {
            PerLayerTable::Quantized(table) => table
                .gather_rows(&flat)?
                .dequantize(&Device::Cpu)?
                .reshape((flat.elem_count(), self.layers * self.width)),
            PerLayerTable::Dense(table) => table.forward(&flat)?.to_dtype(DType::F32),
        }
    }

    /// Bytes the table holds in the machine's memory, for the load's log line.
    pub(crate) fn table_bytes(&self) -> usize {
        match &self.table {
            PerLayerTable::Quantized(t) => t.storage_size_in_bytes(),
            PerLayerTable::Dense(t) => t.resident_bytes(),
        }
    }
}

/// Layer `layer`'s slice of a forward's `inputs` (`[b, seq, width]`), a view;
/// [`beside`] brings it to the device the layer ran on.
pub(crate) fn layer_input(inputs: &Tensor, layer: usize) -> candle_core::Result<Tensor> {
    inputs.narrow(2, layer, 1)?.squeeze(2)
}

/// `input` on `sum`'s device — itself where it already is. A view is copied
/// out first: `to_device` on a view moves its WHOLE storage, every layer's
/// slice of it.
pub(crate) fn beside(input: &Tensor, sum: &Tensor) -> candle_core::Result<Tensor> {
    if input.device().same_device(sum.device()) {
        return Ok(input.clone());
    }
    input.force_contiguous()?.to_device(sum.device())
}
