//! Weight-loading paths for `SplitModel` — GGUF file loading, shard loading,
//! and the per-architecture tensor decode logic. Split out of `model.rs` so
//! the struct definition and the hot-path execution code stay readable.

// ── Split model: loads only a range of layers from a GGUF ──

mod shards;

use std::path::Path;

use crate::inference::residual_norm::RmsNorm;
use candle_core::quantized::gguf_file;
use candle_core::quantized::QTensor;
use candle_core::Device;
use candle_nn::Embedding;

use crate::error::SwarmError;

use super::kv_budget::KvGrowth;
use super::model::SplitModel;
use super::rope::{
    load_longrope_factors, precompute_freqs_cis, precompute_freqs_cis_longrope,
    precompute_freqs_cis_scaled,
};
use super::token_embedding::TokenEmbedding;
use super::{
    Activation, DeepSeekMeta, DeltaNetWeights, FfnVariant, LayerVariant, LayerWeights, MlaWeights,
    Mlp, ModelArch, MoeFfn, MoeGatingFunc, MoeRoutingConfig, QMatMul, Qwen35AttnWeights,
    DEFAULT_MAX_SEQ_LEN,
};

/// Per-call options for [`SplitModel::load_model_from_content`]. Bundles the
/// layer range + first/last role flags + the optional `parallel_data` mmap
/// slice so the loader signature doesn't need `#[allow(too_many_arguments)]`.
#[derive(Debug, Clone, Copy)]
pub(super) struct SplitLoadOptions<'a> {
    pub layer_start: usize,
    pub layer_end: usize,
    pub is_first: bool,
    pub is_last: bool,
    /// When `Some`, enables parallel dense-layer loading via per-thread cursors
    /// (mmap path). When `None`, loads sequentially (ShardReader path —
    /// `ShardReader` can't be shared across threads).
    pub parallel_data: Option<&'a [u8]>,
    /// How many of this segment's layers to place on the graphics card, from
    /// the start of the range. `None` means the old all-or-nothing behaviour:
    /// whatever `device` is, every layer goes there.
    ///
    /// The count is decided by `ModelProcessPool`, never here — graphics memory
    /// has one owner (`.claude/rules/arch-worker-memory.md`), and a loader that sized
    /// its own placement would be the second accountant that #401/#402 were
    /// about. This obeys a number; it does not choose one.
    pub gpu_layers: Option<usize>,
    /// How many of the card-held layers, from the start, keep their routed
    /// experts in the machine's memory (`split::cpu_expert_layers`, #138 (3)).
    /// Obeyed the same way.
    pub cpu_expert_layers: Option<usize>,
}

/// Element type the token-embedding table is dequantized to.
///
/// Also the type an embedding LOOKUP returns, whether or not a dense table
/// exists — `token_embedding::TokenEmbedding` returns this from both of its
/// variants precisely so no call site has to know which it got.
///
/// **The single source of truth for that dtype.** Three places have to agree:
/// this loader (which allocates it), [`segment_weight_bytes`] (which subtracts
/// it from the KV-cache budget), and
/// `model::auto_manage::vram::EMBEDDING_TABLE_BYTES_PER_ELEMENT` (which decides
/// whether the model is admitted to the GPU at all). A disagreement is
/// invisible until a node either refuses a model that would have fitted or
/// OOMs on one it thought would. `embedding_dtype_matches_the_vram_estimate`
/// pins the last of those three.
///
/// f16 rather than f32: `Embedding::new` needs a dense tensor, but the values
/// come from a quantized tensor whose own block scales are f16, so the wider
/// type stores nothing extra. On a large-vocabulary model the difference is
/// decisive — Gemma 2 2B's 256,000-token table costs 2250 MB at f32 against
/// 1629 MB for the entire rest of the model, which by itself pushed a 6 GB
/// card to a CPU fallback (reported 2026-08-01).
pub const EMBEDDING_DTYPE: candle_core::DType = candle_core::DType::F16;

/// Sum the on-device byte cost of the tensors this segment will load.
///
/// Only the segment's own layers count — a node holding layers 8..16 never
/// allocates the others. `token_embd.weight` is charged at its **dequantized**
/// size ([`EMBEDDING_DTYPE`]) when the loader dequantizes it (`Embedding::new`
/// takes a dense tensor), which on a 128K-vocab model is several times its
/// quantized size and is exactly the term a file-size-based estimate misses —
/// and at its quantized size when `rows_on_demand` says the loader will read
/// rows out of it instead. `rows_on_demand` MUST be the same answer
/// `token_embedding::rows_on_demand_eligible` gives the loader, or this
/// function prices an allocation that never happens.
///
/// Used only to decide how much VRAM is left for the KV cache, so an
/// over-estimate is the safe direction: it shortens context rather than
/// letting the allocation fail.
fn segment_weight_bytes(
    ct: &gguf_file::Content,
    layer_start: usize,
    layer_end: usize,
    is_first: bool,
    is_last: bool,
    rows_on_demand: bool,
) -> u64 {
    let tensor_bytes = |info: &gguf_file::TensorInfo| -> u64 {
        let elems = info.shape.elem_count() as u64;
        let block = info.ggml_dtype.block_size().max(1) as u64;
        elems * info.ggml_dtype.type_size() as u64 / block
    };

    let mut total = 0u64;
    for (name, info) in ct.tensor_infos.iter() {
        let counted = if let Some(rest) = name.strip_prefix("blk.") {
            match rest
                .split_once('.')
                .and_then(|(n, _)| n.parse::<usize>().ok())
            {
                Some(idx) => (layer_start..layer_end).contains(&idx),
                None => false,
            }
        } else if name == "token_embd.weight" {
            // Charged below at its dequantized size instead.
            false
        } else if name == "output.weight" || name == "output_norm.weight" {
            is_last
        } else {
            // Small odds and ends (rope factors, per-model extras).
            true
        };
        if counted {
            total = total.saturating_add(tensor_bytes(info));
        }
    }

    // `token_embd.weight`, whose residency depends on which path the loader
    // takes and on whether the model ties its LM head to it. Mirrors the
    // allocation in `load_split_model` exactly; see the truth table there.
    if let Some(info) = ct.tensor_infos.get("token_embd.weight") {
        let tied = !ct.tensor_infos.contains_key("output.weight");
        if is_first {
            total = total.saturating_add(if rows_on_demand {
                // Held quantized, rows dequantized as looked up.
                tensor_bytes(info)
            } else {
                info.shape.elem_count() as u64 * EMBEDDING_DTYPE.size_in_bytes() as u64
            });
        }
        // A weight-tied model reuses token_embd as its LM head. When this
        // segment holds the embedding in quantized form the head SHARES that
        // tensor and costs nothing extra; otherwise it is a second, quantized
        // copy alongside the dequantized one.
        if is_last && tied && !(is_first && rows_on_demand) {
            total = total.saturating_add(tensor_bytes(info));
        }
    }
    total
}

/// One of a GGUF's stacked expert tensors (`ffn_{gate,up,down}_exps`), cut into
/// one QUANTIZED matrix per expert and placed on `device`.
///
/// The stack is `[n_experts, rows, cols]` (candle's order — ggml's `ne`
/// reversed) and stored expert-major, so expert `e` is the contiguous
/// `rows × cols` block at byte `e × per_expert`, made of whole quantization
/// blocks because every row is. It is read on the processor, sliced, and only
/// the slices go to `device`; nothing is dequantized (`layers::ExpertFfn`).
fn split_expert_stack<R: std::io::Read + std::io::Seek>(
    ct: &gguf_file::Content,
    reader: &mut R,
    name: &str,
    device: &Device,
) -> Result<Vec<QTensor>, SwarmError> {
    let [whole] = split_expert_stack_rows(ct, reader, name, device)?;
    Ok(whole)
}

/// [`split_expert_stack`] cut into `P` row bands per expert, each expert's
/// `P` matrices of `rows / P` rows: a FUSED stack (`ffn_gate_up_exps`, Gemma
/// 4) holds each expert's gate rows and then its up rows — llama.cpp's
/// `build_moe_ffn` views the product at 0 and at `n_ff`. A band is whole rows,
/// so whole quantization blocks.
fn split_expert_stack_rows<R: std::io::Read + std::io::Seek, const P: usize>(
    ct: &gguf_file::Content,
    reader: &mut R,
    name: &str,
    device: &Device,
) -> Result<[Vec<QTensor>; P], SwarmError> {
    let stack = ExpertStack::read(ct, reader, name, P)?;
    let data = stack.data(name)?;
    let band = stack.per_expert / P;
    let rows = stack.rows / P;
    let mut bands: [Vec<QTensor>; P] = std::array::from_fn(|_| Vec::with_capacity(stack.n));
    for e in 0..stack.n {
        for (p, out) in bands.iter_mut().enumerate() {
            let at = e * stack.per_expert + p * band;
            out.push(
                candle_core::quantized::ggml_file::qtensor_from_ggml(
                    stack.dtype,
                    &data[at..at + band],
                    vec![rows, stack.cols],
                    device,
                )
                .map_err(|err| SwarmError::Internal(format!("{name} expert {e}: {err}")))?,
            );
        }
    }
    Ok(bands)
}

/// A stacked expert tensor read on the processor, with its geometry checked.
struct ExpertStack {
    tensor: QTensor,
    dtype: candle_core::quantized::GgmlDType,
    n: usize,
    rows: usize,
    cols: usize,
    per_expert: usize,
}

impl ExpertStack {
    /// `name` as `[n_experts, rows, cols]` whose expert blocks hold `bands`
    /// equal row bands each.
    fn read<R: std::io::Read + std::io::Seek>(
        ct: &gguf_file::Content,
        reader: &mut R,
        name: &str,
        bands: usize,
    ) -> Result<Self, SwarmError> {
        let tensor = ct
            .tensor(reader, name, &Device::Cpu)
            .map_err(|e| SwarmError::Internal(format!("{name}: {e}")))?;
        let (n, rows, cols) = match *tensor.shape().dims() {
            [n, rows, cols] => (n, rows, cols),
            ref dims => {
                return Err(SwarmError::Internal(format!(
                    "{name}: an expert stack has 3 dimensions, this one {dims:?}"
                )))
            }
        };
        let dtype = tensor.dtype();
        let per_expert = rows * cols / dtype.block_size() * dtype.type_size();
        let stack = Self {
            tensor,
            dtype,
            n,
            rows,
            cols,
            per_expert,
        };
        let len = stack.data(name)?.len();
        if n == 0 || per_expert.checked_mul(n) != Some(len) || rows % bands != 0 {
            return Err(SwarmError::Internal(format!(
                "{name}: {len} bytes is not {n} experts of {rows}x{cols} {dtype:?} in {bands} bands"
            )));
        }
        Ok(stack)
    }

    /// The stack's bytes — borrowed, the tensor being on the processor.
    fn data(&self, name: &str) -> Result<std::borrow::Cow<'_, [u8]>, SwarmError> {
        self.tensor
            .data()
            .map_err(|e| SwarmError::Internal(format!("{name}: {e}")))
    }
}

/// A fused `ffn_gate_up_exps` stack as the two stacks the card kernel reads,
/// `[n_experts, n_ff, hidden]` each: every expert's gate rows into one, its
/// up rows into the other ([`split_expert_stack_rows`]).
fn unfuse_expert_stack<R: std::io::Read + std::io::Seek>(
    ct: &gguf_file::Content,
    reader: &mut R,
    name: &str,
    device: &Device,
) -> Result<[QTensor; 2], SwarmError> {
    let stack = ExpertStack::read(ct, reader, name, 2)?;
    let data = stack.data(name)?;
    let band = stack.per_expert / 2;
    let halves = [0, 1].map(|half| {
        let mut bytes = Vec::with_capacity(stack.n * band);
        for e in 0..stack.n {
            let at = e * stack.per_expert + half * band;
            bytes.extend_from_slice(&data[at..at + band]);
        }
        candle_core::quantized::ggml_file::qtensor_from_ggml(
            stack.dtype,
            &bytes,
            vec![stack.n, stack.rows / 2, stack.cols],
            device,
        )
        .map_err(|err| SwarmError::Internal(format!("{name} half {half}: {err}")))
    });
    let [gate, up] = halves;
    Ok([gate?, up?])
}

/// A layer's mixture-of-experts feed-forward, loaded the one way every MoE
/// family here shares: the router dequantized (`[n_experts, hidden]`, small),
/// the routed experts kept QUANTIZED — as the file's three stacks on a card the
/// grouped kernel reads (`load_stacked_experts`), as one matrix per expert
/// everywhere else — and the shared expert, with its sigmoid gate where the
/// model has one, when present.
///
/// One function because the three per-family copies it replaces each
/// dequantized every expert to f32, which made every real-sized MoE model
/// unloadable (`layers::ExpertFfn`); a family added later inherits the right
/// behaviour by calling it.
///
/// Takes the PLACEMENT, not a device: layer `layer_idx` loads on
/// `placement.device_for` and its routed experts on
/// `placement.experts_device_for` — the machine's memory for a card-held layer
/// whose experts stay in RAM (#138 (3)) — so no per-layer loop can put them
/// anywhere else.
///
/// Gemma 4's experts come as a FUSED gate/up stack (`ffn_gate_up_exps`) with a
/// scale per expert on the down projection (`ffn_down_exps.scale`); both are
/// read here for any family whose file carries them. Its dense branch beside
/// the experts is the caller's ([`load_gemma4_beside`]).
#[allow(clippy::too_many_arguments)]
fn load_moe_ffn<R: std::io::Read + std::io::Seek>(
    ct: &gguf_file::Content,
    reader: &mut R,
    prefix: &str,
    placement: &super::hybrid::LayerPlacement,
    layer_idx: usize,
    n_experts_used: usize,
    routing: MoeRoutingConfig,
    activation: Activation,
) -> Result<MoeFfn, SwarmError> {
    let device = &placement.device_for(layer_idx);
    let experts_device = &placement.experts_device_for(layer_idx);
    let name = |t: &str| format!("{prefix}.{t}.weight");
    // llama.cpp also reads a scale on a separate gate or up stack (and on the
    // fused one, as `up_exps_s`); no file this build has met carries one, and
    // answering without it would be silently wrong.
    for t in [
        "ffn_gate_exps.scale",
        "ffn_up_exps.scale",
        "ffn_gate_up_exps.scale",
    ] {
        if ct.tensor_infos.contains_key(&format!("{prefix}.{t}")) {
            return Err(SwarmError::Internal(format!(
                "{prefix}.{t}: an expert scale on the gate or up projection is not supported"
            )));
        }
    }
    let router = ct
        .tensor(reader, &name("ffn_gate_inp"), device)
        .and_then(|t| t.dequantize(device))
        .map_err(|e| SwarmError::Internal(format!("{prefix}.ffn_gate_inp: {e}")))?;
    let n_router = router.dim(0).map_err(SwarmError::internal)?;
    let expert_scales = read_f32s(ct, reader, &format!("{prefix}.ffn_down_exps.scale"))?
        .map(|host| {
            if host.len() != n_router {
                return Err(SwarmError::Internal(format!(
                    "{prefix}.ffn_down_exps.scale: {} scales for {n_router} experts",
                    host.len()
                )));
            }
            if routing.weight_before_ffn {
                return Err(SwarmError::Internal(format!(
                    "{prefix}.ffn_down_exps.scale: an output scale on experts weighted at their input"
                )));
            }
            super::ExpertScales::new(host, device).map_err(SwarmError::internal)
        })
        .transpose()?;
    let (experts, stacked) = if let Some(stacked) =
        load_stacked_experts(ct, reader, prefix, experts_device, n_router, routing)?
    {
        (Vec::new(), Some(stacked))
    } else {
        (
            load_experts_one_by_one(ct, reader, prefix, experts_device, n_router)?,
            None,
        )
    };

    // Present or absent as a whole in every family; a present tensor that
    // fails to read is an error, not an absence.
    let mut optional = |t: &str| -> Result<Option<QTensor>, SwarmError> {
        let n = name(t);
        if !ct.tensor_infos.contains_key(&n) {
            return Ok(None);
        }
        ct.tensor(reader, &n, device)
            .map(Some)
            .map_err(|e| SwarmError::Internal(format!("{n}: {e}")))
    };
    let shared = |t: Option<QTensor>| {
        t.map(QMatMul::from_qtensor)
            .transpose()
            .map_err(SwarmError::internal)
    };
    let shared_gate = shared(optional("ffn_gate_shexp")?)?;
    let shared_up = shared(optional("ffn_up_shexp")?)?;
    let shared_down = shared(optional("ffn_down_shexp")?)?;
    let shared_gate_inp = optional("ffn_gate_inp_shexp")?
        .map(|t| t.dequantize(device))
        .transpose()
        .map_err(SwarmError::internal)?;

    Ok(MoeFfn {
        gate: router,
        experts,
        stacked,
        prompt_card: None,
        shared_gate,
        shared_down,
        shared_up,
        shared_gate_inp,
        n_experts_used,
        routing,
        activation,
        expert_scales,
        beside: None,
    })
}

/// What [`load_standard_ffn`] needs of the model, beside the layer.
#[derive(Clone, Copy)]
struct StandardFfn {
    experts_used: usize,
    routing: MoeRoutingConfig,
    activation: Activation,
    gemma4: bool,
    rms_norm_eps: f64,
}

/// A standard-layout layer's feed-forward: routed experts where the layer
/// holds them (Qwen2/Qwen3-MoE; Gemma 4 with its dense branch beside them,
/// [`load_gemma4_beside`]), a dense one otherwise ([`load_dense_mlp`]). ONE
/// function for the parallel and the sequential per-layer loops, so neither
/// can load a layer the other would not (gotcha #714).
fn load_standard_ffn<R: std::io::Read + std::io::Seek>(
    ct: &gguf_file::Content,
    reader: &mut R,
    prefix: &str,
    placement: &super::hybrid::LayerPlacement,
    layer_idx: usize,
    how: StandardFfn,
) -> Result<FfnVariant, SwarmError> {
    let device = placement.device_for(layer_idx);
    if !holds_routed_experts(ct, prefix) {
        return load_dense_mlp(ct, reader, prefix, &device, how.activation).map(FfnVariant::Dense);
    }
    let mut moe = load_moe_ffn(
        ct,
        reader,
        prefix,
        placement,
        layer_idx,
        how.experts_used,
        how.routing,
        how.activation,
    )?;
    if how.gemma4 {
        let mlp = load_dense_mlp(ct, reader, prefix, &device, how.activation)?;
        moe.beside = Some(Box::new(load_gemma4_beside(
            ct,
            reader,
            prefix,
            &device,
            mlp,
            how.rms_norm_eps,
        )?));
    }
    Ok(FfnVariant::MoE(moe))
}

/// Does layer `prefix` hold routed experts — `ffn_gate_exps`, or Gemma 4's
/// fused `ffn_gate_up_exps`?
fn holds_routed_experts(ct: &gguf_file::Content, prefix: &str) -> bool {
    ct.tensor_infos
        .contains_key(&format!("{prefix}.ffn_gate_exps.weight"))
        || experts_fused(ct, prefix)
}

/// A dense gated feed-forward: separate `ffn_gate` and `ffn_up`, or — Phi-3 —
/// one `ffn_up` holding gate then up, sliced without re-quantizing.
fn load_dense_mlp<R: std::io::Read + std::io::Seek>(
    ct: &gguf_file::Content,
    reader: &mut R,
    prefix: &str,
    device: &Device,
    activation: Activation,
) -> Result<Mlp, SwarmError> {
    let mut read = |t: &str| {
        ct.tensor(reader, &format!("{prefix}.{t}.weight"), device)
            .map_err(|e| SwarmError::Internal(format!("Failed to load {prefix}.{t}: {e}")))
    };
    let ffn_down = QMatMul::from_qtensor(read("ffn_down")?).map_err(SwarmError::internal)?;
    let has_ffn_gate = ct
        .tensor_infos
        .contains_key(&format!("{prefix}.ffn_gate.weight"));
    let (ffn_gate, ffn_up) = if has_ffn_gate {
        let gate = read("ffn_gate")?;
        let up = read("ffn_up")?;
        (
            QMatMul::from_qtensor(gate).map_err(SwarmError::internal)?,
            QMatMul::from_qtensor(up).map_err(SwarmError::internal)?,
        )
    } else {
        let fused_qt = read("ffn_up")?;
        let half = fused_qt.shape().dims()[0] / 2;
        let fused = QMatMul::make_fused(fused_qt).map_err(SwarmError::internal)?;
        (
            QMatMul::from_fused_slice(fused.clone(), 0, half),
            QMatMul::from_fused_slice(fused, half, half),
        )
    };
    Ok(Mlp {
        ffn_gate: Some(ffn_gate),
        ffn_down,
        ffn_up,
        activation,
    })
}

/// Gemma 4's MoE layer's dense branch and its four norms ([`super::DenseBeside`],
/// llama.cpp `gemma4.cpp`): `post_ffw_norm_1`, `pre_ffw_norm_2`,
/// `post_ffw_norm_2`, and the router's norm, whose weight is
/// `ffn_gate_inp.scale / √n_embd` (`ggml_rms_norm`, then `ggml_scale` by
/// `1/√n_embd`, then `ggml_mul` by the scale).
fn load_gemma4_beside<R: std::io::Read + std::io::Seek>(
    ct: &gguf_file::Content,
    reader: &mut R,
    prefix: &str,
    device: &Device,
    mlp: Mlp,
    eps: f64,
) -> Result<super::DenseBeside, SwarmError> {
    let mut norm = |t: &str| {
        let n = format!("{prefix}.{t}.weight");
        ct.tensor(reader, &n, device)
            .and_then(|q| RmsNorm::from_qtensor(q, eps))
            .map_err(|e| SwarmError::Internal(format!("{n}: {e}")))
    };
    let mlp_post_norm = norm("post_ffw_norm_1")?;
    let experts_norm = norm("pre_ffw_norm_2")?;
    let experts_post_norm = norm("post_ffw_norm_2")?;
    let scale_name = format!("{prefix}.ffn_gate_inp.scale");
    let scale = read_f32s(ct, reader, &scale_name)?
        .ok_or_else(|| SwarmError::Internal(format!("{scale_name}: not in the file")))?;
    let n_embd = scale.len();
    let weight = candle_core::Tensor::from_vec(scale, (n_embd,), device)
        .and_then(|w| w * (1.0 / (n_embd as f64).sqrt()))
        .map_err(|e| SwarmError::Internal(format!("{scale_name}: {e}")))?;
    Ok(super::DenseBeside {
        mlp,
        mlp_post_norm,
        experts_norm,
        experts_post_norm,
        router_norm: RmsNorm::new(weight, eps),
    })
}

/// Does layer `prefix` hold its experts' gate and up projections as ONE
/// stack (`ffn_gate_up_exps`, Gemma 4)?
fn experts_fused(ct: &gguf_file::Content, prefix: &str) -> bool {
    ct.tensor_infos
        .contains_key(&format!("{prefix}.ffn_gate_up_exps.weight"))
}

/// A small F32 tensor's numbers (a scale vector), or `None` when the file
/// carries no `name`.
fn read_f32s<R: std::io::Read + std::io::Seek>(
    ct: &gguf_file::Content,
    reader: &mut R,
    name: &str,
) -> Result<Option<Vec<f32>>, SwarmError> {
    if !ct.tensor_infos.contains_key(name) {
        return Ok(None);
    }
    ct.tensor(reader, name, &Device::Cpu)
        .and_then(|t| t.dequantize(&Device::Cpu))
        .and_then(|t| t.flatten_all()?.to_vec1::<f32>())
        .map(Some)
        .map_err(|e| SwarmError::Internal(format!("{name}: {e}")))
}

/// A layer's three expert stacks loaded whole onto a card, where
/// `layers::StackedExperts::fit` says the grouped kernel takes them; `None`
/// otherwise (one matrix per expert, `load_experts_one_by_one`). Read from the
/// table before any byte is, so a layer is never loaded twice. A fused
/// gate/up stack is cut into the two the kernel reads ([`unfuse_expert_stack`]).
fn load_stacked_experts<R: std::io::Read + std::io::Seek>(
    ct: &gguf_file::Content,
    reader: &mut R,
    prefix: &str,
    device: &Device,
    n_router: usize,
    routing: MoeRoutingConfig,
) -> Result<Option<super::StackedExperts>, SwarmError> {
    let fused = experts_fused(ct, prefix);
    let names = if fused {
        ["ffn_gate_up_exps", "ffn_gate_up_exps", "ffn_down_exps"]
    } else {
        ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"]
    }
    .map(|t| format!("{prefix}.{t}.weight"));
    let infos = names
        .iter()
        .map(|n| {
            ct.tensor_infos
                .get(n)
                .ok_or_else(|| SwarmError::Internal(format!("{n}: not in the file")))
        })
        .collect::<Result<Vec<_>, _>>()?;
    // candle's order: [n_experts, rows, cols]; each product's input is `cols`.
    let dims: Vec<&[usize]> = infos.iter().map(|i| i.shape.dims()).collect();
    if dims.iter().any(|d| d.len() != 3 || d[0] != n_router) {
        return Ok(None);
    }
    let dtypes = [
        infos[0].ggml_dtype,
        infos[1].ggml_dtype,
        infos[2].ggml_dtype,
    ];
    if !super::StackedExperts::fit(device.is_cuda(), dtypes, [dims[0][2], dims[2][2]], routing) {
        return Ok(None);
    }
    let read = |reader: &mut R, n: &str| {
        ct.tensor(reader, n, device)
            .map(std::sync::Arc::new)
            .map_err(|e| SwarmError::Internal(format!("{n}: {e}")))
    };
    let (gate, up) = if fused {
        let [gate, up] = unfuse_expert_stack(ct, reader, &names[0], device)?;
        (std::sync::Arc::new(gate), std::sync::Arc::new(up))
    } else {
        (read(reader, &names[0])?, read(reader, &names[1])?)
    };
    Ok(Some(super::StackedExperts {
        gate,
        up,
        down: read(reader, &names[2])?,
    }))
}

/// A layer's routed experts as one QUANTIZED matrix each
/// (`split_expert_stack`), in router order.
fn load_experts_one_by_one<R: std::io::Read + std::io::Seek>(
    ct: &gguf_file::Content,
    reader: &mut R,
    prefix: &str,
    device: &Device,
    n_router: usize,
) -> Result<Vec<super::ExpertFfn>, SwarmError> {
    let name = |t: &str| format!("{prefix}.{t}.weight");
    let (gate, up) = if experts_fused(ct, prefix) {
        let [gate, up] = split_expert_stack_rows(ct, reader, &name("ffn_gate_up_exps"), device)?;
        (gate, up)
    } else {
        (
            split_expert_stack(ct, reader, &name("ffn_gate_exps"), device)?,
            split_expert_stack(ct, reader, &name("ffn_up_exps"), device)?,
        )
    };
    let down = split_expert_stack(ct, reader, &name("ffn_down_exps"), device)?;
    if gate.len() != n_router || up.len() != n_router || down.len() != n_router {
        return Err(SwarmError::Internal(format!(
            "{prefix}: the router chooses among {n_router} experts, the stacks hold {}/{}/{}",
            gate.len(),
            up.len(),
            down.len()
        )));
    }
    gate.into_iter()
        .zip(up)
        .zip(down)
        .map(|((g, u), d)| {
            Ok(super::ExpertFfn {
                gate: QMatMul::from_qtensor(g)?,
                up: QMatMul::from_qtensor(u)?,
                down: QMatMul::from_qtensor(d)?,
            })
        })
        .collect::<Result<Vec<_>, candle_core::Error>>()
        .map_err(SwarmError::internal)
}

/// Whether a family's router renormalizes its top-k weights when the GGUF does
/// not say (`{arch}.expert_weights_norm` absent). llama.cpp hardcodes this in
/// each family's graph rather than writing a key, so the answer is per family:
/// `qwen2moe.cpp` passes `norm_w = false` and `qwen3moe.cpp` `true` (read
/// 2026-09-25). Everything else keeps the long-standing default of `true`.
fn moe_renormalizes_by_default(arch: &str) -> bool {
    arch != "qwen2moe"
}

/// A layer's output scale — Gemma 4's `layer_output_scale`, one number the
/// layer's whole output is multiplied by (llama.cpp's `out_scale`) — or `None`
/// for a file that carries none.
fn layer_output_scale<R: std::io::Read + std::io::Seek>(
    ct: &gguf_file::Content,
    reader: &mut R,
    prefix: &str,
) -> Result<Option<f64>, SwarmError> {
    let name = format!("{prefix}.layer_output_scale.weight");
    if !ct.tensor_infos.contains_key(&name) {
        return Ok(None);
    }
    let value = ct
        .tensor(reader, &name, &Device::Cpu)
        .and_then(|t| t.dequantize(&Device::Cpu))
        .and_then(|t| t.flatten_all()?.to_vec1::<f32>())
        .map_err(|e| SwarmError::Internal(format!("{name}: {e}")))?;
    match value.as_slice() {
        [scale] => Ok(Some(f64::from(*scale))),
        _ => Err(SwarmError::Internal(format!(
            "{name} holds {} numbers, expected one",
            value.len()
        ))),
    }
}

/// Whether layer `prefix` uses its K projection's output as V: a Gemma 4
/// global layer, which carries no `attn_v` (llama.cpp's
/// `use_alternative_attention`, "if v_proj is not present, use Kcur as Vcur").
/// Every other family's file carries one, and its absence stays an error.
fn value_is_key(ct: &gguf_file::Content, gemma4: bool, prefix: &str) -> bool {
    gemma4
        && !ct
            .tensor_infos
            .contains_key(&format!("{prefix}.attn_v.weight"))
}

/// The token ids a model file says are never to be sampled —
/// `tokenizer.ggml.suppress_tokens`, which llama.cpp merges into every
/// request's logit bias (`common/sampling.cpp`, "model suppress tokens"; Gemma
/// 4's 12B lists `<audio|>` and `<image|>`). Ids past the head's `vocab` are
/// dropped: they name no logit. Empty without the key.
fn suppressed_token_ids(
    metadata: &std::collections::HashMap<String, gguf_file::Value>,
    vocab: usize,
) -> Vec<usize> {
    let Some(Ok(listed)) = metadata
        .get("tokenizer.ggml.suppress_tokens")
        .map(gguf_file::Value::to_vec)
    else {
        return Vec::new();
    };
    listed
        .iter()
        .filter_map(|v| match v {
            gguf_file::Value::I32(id) => usize::try_from(*id).ok(),
            gguf_file::Value::U32(id) => Some(*id as usize),
            _ => None,
        })
        .filter(|&id| id < vocab)
        .collect()
}

/// The row the output head adds to every row of logits so the file's
/// suppressed tokens are never chosen (`SplitModel::suppressed_logits`, #282):
/// [`SUPPRESSED_LOGIT`](super::executor::SUPPRESSED_LOGIT) at each listed id, 0
/// elsewhere, built once on the head's device — so a captured decode step adds
/// it without a copy from the host. `None` when the file lists none.
fn suppressed_logits_row(
    metadata: &std::collections::HashMap<String, gguf_file::Value>,
    vocab: usize,
    device: &Device,
) -> Result<Option<candle_core::Tensor>, SwarmError> {
    let ids = suppressed_token_ids(metadata, vocab);
    if ids.is_empty() {
        return Ok(None);
    }
    let mut row = vec![0f32; vocab];
    for &id in &ids {
        row[id] = super::executor::SUPPRESSED_LOGIT;
    }
    tracing::debug!(tokens = ?ids, "The model file's suppressed tokens are never sampled");
    candle_core::Tensor::from_vec(row, vocab, device)
        .map(Some)
        .map_err(|e| SwarmError::Internal(format!("suppressed_tokens row: {e}")))
}

/// The device a split model loads onto: the card when there is one and the
/// worker was not told to use the processor (`force_cpu`, from
/// `inference.gpu_layers = 0`), the processor otherwise.
///
/// The ONE place a worker picks the card, so it is also where the card's
/// memory pool is told to keep what it frees (`inference::cuda_pool`,
/// `docs/FUTURE_WORK.md` #146) — a model that reached the card any other way
/// would run with a pool that hands memory back to the driver at every
/// synchronize. Guard: `every_split_model_reaches_the_card_through_load_device`.
///
/// And where the worker's one allocation probe runs (`cuda_pool::probe_once`),
/// before the model's weights take the memory — so every worker that uses the
/// card measures the host's allocation path once (#762).
pub(super) fn load_device(force_cpu: bool) -> Device {
    if force_cpu {
        return Device::Cpu;
    }
    let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
    crate::inference::cuda_pool::keep_freed_memory(&device);
    crate::inference::cuda_pool::probe_once(&device);
    device
}

impl SplitModel {
    /// Load a partial model from a GGUF file, only loading the specified layer range.
    ///
    /// - `layer_start..layer_end`: the transformer block range this node owns
    /// - `is_first`: if true, also loads the embedding table
    /// - `is_last`: if true, also loads the final norm and LM head
    /// - `force_cpu`: mirrors `inference.gpu_layers == 0` — skip CUDA entirely
    pub fn load_from_gguf(
        gguf_path: &Path,
        layer_start: usize,
        layer_end: usize,
        is_first: bool,
        is_last: bool,
        force_cpu: bool,
    ) -> Result<Self, SwarmError> {
        let file = std::fs::File::open(gguf_path).map_err(SwarmError::Io)?;
        // SAFETY: `memmap2::Mmap::map` is unsafe because the OS guarantees the
        // mapping is valid only as long as the underlying file content does
        // not change concurrently. We hold the `File` for the whole call, the
        // GGUF file is read-only and atomically published by HF/our shard
        // download path (no in-place truncation/rewriting), and we drop both
        // the file handle and the mmap before this function returns.
        //
        // The non-obvious safety question is whether returned tensors alias
        // the mmap memory after we drop it. They do not: `gguf_file::Content::
        // tensor` reads each block via `ct.tensor(file, name, device)`, which
        // calls `read_to_end` on the cursor and constructs `QTensor` /
        // `Tensor` from a heap-allocated `Vec<u8>` owned by the candle layer
        // (see candle's `quantized::ggml_file::qtensor_from_ggml`). The
        // `parallel_data: Some(mmap.as_ref())` slice path used inside
        // `load_model_from_content` makes per-thread `Cursor`s over the mmap
        // for *parallel reads* of the same blocks, but the read still
        // copies into a `Vec` before constructing the tensor. No tensor
        // returned from this loader retains a reference to the mmap region.
        let mmap = unsafe { memmap2::Mmap::map(&file) }
            .map_err(|e| SwarmError::Internal(format!("Failed to mmap GGUF: {e}")))?;
        let mut file = std::io::Cursor::new(mmap.as_ref());
        let ct = gguf_file::Content::read(&mut file).map_err(|e| {
            SwarmError::Internal(format!(
                "Failed to read GGUF: {}",
                super::explain_gguf_parse_error(&e)
            ))
        })?;

        let device = load_device(force_cpu);
        if device.is_cuda() {
            tracing::info!(layers = %(layer_start..=layer_end).count(), layer_start, layer_end, "Split model using CUDA GPU");
        } else if force_cpu {
            tracing::info!(layers = %(layer_start..=layer_end).count(), layer_start, layer_end, "Split model using CPU (gpu_layers = 0)");
        } else {
            tracing::info!(layers = %(layer_start..=layer_end).count(), layer_start, layer_end, "Split model using CPU (no CUDA available)");
        }

        // This range's own placement where the daemon sent one
        // (`split::placement_of`, #265), else the spawn's.
        let placement = crate::inference::split::placement_of(layer_start, layer_end);
        Self::load_model_from_content(
            ct,
            &mut file,
            device,
            SplitLoadOptions {
                layer_start,
                layer_end,
                is_first,
                is_last,
                parallel_data: Some(mmap.as_ref()),
                gpu_layers: placement.gpu_layers,
                cpu_expert_layers: placement.cpu_expert_layers,
            },
        )
    }

    /// Shared model loading body for both GGUF and shard paths.
    ///
    /// Parses GGUF metadata, loads tensors by architecture, extracts tokenizer/
    /// vocabulary/chat template, and constructs the SplitModel.
    fn load_model_from_content<R: std::io::Read + std::io::Seek>(
        ct: gguf_file::Content,
        mut file: &mut R,
        device: Device,
        opts: SplitLoadOptions<'_>,
    ) -> Result<Self, SwarmError> {
        let SplitLoadOptions {
            layer_start,
            layer_end,
            is_first,
            is_last,
            parallel_data,
            gpu_layers,
            cpu_expert_layers,
        } = opts;
        // Detect architecture prefix from GGUF metadata
        let arch_str = super::gguf_arch_str(&ct);
        let model_arch = ModelArch::from_gguf_arch(&arch_str);

        tracing::info!(arch = %model_arch, "Detected model architecture");

        // Asked before the hyperparameters are read: a family this build has
        // no loader for may lack the keys `from_content` requires.
        if !model_arch.is_supported() {
            return Err(super::ModelRefusal::Architecture(arch_str).into_error());
        }

        // Extract the shared hyperparameters via the unified GgufTensorMeta
        // path so gguf_meta.rs remains the single source of truth for field
        // parsing. Architecture-specific extras (context_length, Gemma
        // softcaps) stay inline since they're not part of the base metadata.
        let meta = super::GgufTensorMeta::from_content(&ct)?;
        // Every load of a model's layers passes here — a request served on
        // this node and a segment served for a peer alike — so a model the
        // router let through (no header there to ask) is refused before it
        // repeats one word for half a minute (field report 2026-10-15).
        if let Some(refusal) = meta.refusal() {
            return Err(refusal.into_error());
        }
        let head_count = meta.head_count;
        // The widest layer's — what every model-wide estimate charges each
        // layer. A layer is built with its own (`attention_shapes`).
        let head_count_kv = meta.head_count_kv;
        let attention_shapes = super::gguf_meta::attention_shapes(&ct)?;
        let block_count = meta.block_count;
        let embedding_length = meta.embedding_length;
        let head_dim = meta.head_dim;
        let rope_dim = meta.rope_dim;
        let rms_norm_eps = meta.rms_norm_eps;
        let rope_freq_base = meta.rope_freq_base;
        let arch = &arch_str; // keep a reference for the extras below

        // Arch-specific metadata lookups (context length, Gemma softcaps,
        // DeepSeek MoE/MLA params) that aren't part of the shared base.
        let md_get = |suffix: &str| {
            let key = format!("{arch}.{suffix}");
            ct.metadata
                .get(&key)
                .ok_or_else(|| SwarmError::Internal(format!("Missing GGUF metadata: {key}")))
        };

        // R132: per-arch MoE router policy. GGUF metadata
        // `{arch}.expert_gating_func` (uint: 1=softmax / 2=sigmoid;
        // matches llama.cpp's enum) + `{arch}.expert_weights_norm`
        // (bool). Missing keys fall back to (Softmax, renormalize) —
        // matches Mixtral / Qwen3-MoE with `norm_topk_prob=true` /
        // Llama 4 default. Differentiates DeepSeek-V2 strict (no
        // renorm) and DeepSeek-V3 (sigmoid gate).
        let moe_routing = if matches!(model_arch, ModelArch::Llama4) {
            // llama.cpp's Llama 4 graph reads NO routing key: it passes sigmoid
            // gating and `norm_w = false`, and applies each weight to the
            // expert's INPUT (`models/llama4.cpp` + `build_moe_ffn`'s
            // `weight_before_ffn`; the same in master and in 0.3.16, read
            // 2026-09-25). Our generic softmax + renormalize gave every token
            // of the top-1 Scout a weight of 1.0 on the output.
            MoeRoutingConfig {
                gating_func: MoeGatingFunc::Sigmoid,
                renormalize_weights: false,
                weight_before_ffn: true,
            }
        } else if matches!(model_arch, ModelArch::Gemma4) {
            // Gemma 4's graph reads no routing key either: softmax over every
            // expert, the top k renormalized (`norm_w = true`), weights on the
            // output (`models/gemma4.cpp`, master read 2026-10-16) — what
            // softmax over the k picks computes (`topk_host`).
            MoeRoutingConfig {
                gating_func: MoeGatingFunc::Softmax,
                renormalize_weights: true,
                weight_before_ffn: false,
            }
        } else {
            MoeRoutingConfig {
                gating_func: md_get("expert_gating_func")
                    .and_then(|v| v.to_u32().map_err(SwarmError::internal))
                    .ok()
                    .map(|n| {
                        if n == 2 {
                            MoeGatingFunc::Sigmoid
                        } else {
                            MoeGatingFunc::Softmax
                        }
                    })
                    .unwrap_or(MoeGatingFunc::Softmax),
                renormalize_weights: md_get("expert_weights_norm")
                    .and_then(|v| v.to_bool().map_err(SwarmError::internal))
                    .unwrap_or_else(|_| moe_renormalizes_by_default(&arch_str)),
                weight_before_ffn: false,
            }
        };
        tracing::debug!(
            gating_func = ?moe_routing.gating_func,
            renormalize = moe_routing.renormalize_weights,
            "Loaded MoE routing config from GGUF"
        );

        let mut context_length = md_get("context_length")
            .and_then(|v| v.to_u32().map_err(SwarmError::internal))
            .unwrap_or(DEFAULT_MAX_SEQ_LEN as u32) as usize;
        // One rule, in `split::effective_context_length`; the branches below only
        // decide what to SAY about it. Computing it here as well is how the
        // loader, the VRAM estimator and `/v1/models` would come to disagree
        // about the same number.
        let effective = super::effective_context_length(context_length);
        if let Some(cap) = super::max_seq_len_override() {
            if cap < context_length {
                tracing::info!(
                    gguf_context_length = context_length,
                    override_value = cap,
                    "Clamping context_length to MAX_SEQ_LEN_OVERRIDE — KV-cache and RoPE table sized to override"
                );
                context_length = effective;
            }
        } else {
            // No explicit override. Two independent reductions apply, in order.
            //
            // 1. A default target, the way llama.cpp's `-c` defaults to 4096
            //    rather than to whatever the model advertises. This bounds how
            //    far a conversation may grow, and with it how large the KV
            //    cache can get: a 128K declaration lets one conversation grow
            //    to a 128K cache.
            //
            //    Until 2026-08-07 it was worse than that — the cache allocated
            //    `max_seq_len` on its first append, so a 128K declaration was a
            //    128K *allocation* for a twenty-token chat. `new_kv_cache` now
            //    grows on demand, so this is a ceiling rather than a
            //    reservation. It still matters: growth is only bounded because
            //    this number is.
            //
            //    Measured on an RTX 3070 with llama-3.2-1b-instruct-q8-0:
            //    serving the full declared context cost 7138 MiB and serving
            //    4096 cost 3110 MiB, at an identical 46-53 tok/s.
            if context_length > DEFAULT_MAX_SEQ_LEN {
                tracing::info!(
                    gguf_context_length = context_length,
                    default_context = DEFAULT_MAX_SEQ_LEN,
                    "Using the default {DEFAULT_MAX_SEQ_LEN}-token context — this model declares \
                     {context_length}, and letting one conversation grow that far would claim \
                     several GB of KV cache for no gain at typical conversation lengths. Raise \
                     it with inference.max_seq_len_override",
                );
                context_length = effective;
            }
        }
        debug_assert_eq!(
            context_length, effective,
            "the loader must serve exactly the context the estimator charges for"
        );

        // 2. Record how much GPU memory this segment's KV cache may occupy in
        //    total. This USED to shrink `context_length` so that one
        //    conversation at its full length would fit, because the cache
        //    reserved its whole ceiling on the first append. It no longer
        //    does (`layers::new_kv_cache` grows in quanta), so clamping here
        //    would shorten every user's context to guard a case most never
        //    reach — and it never bounded concurrency anyway, since it sized
        //    for ONE conversation.
        //
        //    What replaces it is a budget checked at runtime, before a forward
        //    claims another quantum, against what the WHOLE store holds. See
        //    `kv_budget::quantum_exceeds_headroom`. The failure mode improves
        //    too: instead of a context silently cut at load and kept for the
        //    daemon's life, a request is refused with 503 at the moment memory
        //    is actually short — which in a swarm routes it to a peer.
        //
        //    **Recorded whatever the context setting** (FUTURE_WORK #194). It
        //    used to be skipped when `inference.max_seq_len_override` was set
        //    — "an explicit setting wins" — a rule inherited from the
        //    load-time shrink above it replaced (2907cb7e), where it meant
        //    "never shorten a context the operator named". The runtime budget
        //    shortens nothing; it refuses one request that does not fit NOW.
        //    Skipped, a card holding a model for an agent (the override is
        //    how the README tells agent users to make room for their prompt)
        //    admitted every prompt and let the cache spill into host memory —
        //    the 1-3 tok/s crawl #440 removed for everyone else — and the
        //    load-time fallback it claimed as a backstop runs before the
        //    cache grows, so it never saw the overflow.
        let mut kv_budget_bytes: Option<u64> = None;
        let mut kv_growth = KvGrowth::default();
        // How many of this segment's layers `device` holds — every one, unless
        // the pool placed part of the segment on the processor (hybrid). The
        // SAME answer the placement below acts on (`hybrid::layers_on_device`):
        // the budget must charge the card for what it holds, not for the whole
        // segment, or a 36-of-40 placement is told it has 15 MB for
        // conversations (#104).
        let hybrid_ok = super::hybrid::arch_supports_hybrid(&model_arch);
        let segment_layer_count = layer_end.min(block_count).saturating_sub(layer_start);
        let device_layers = super::hybrid::layers_on_device(
            gpu_layers,
            hybrid_ok,
            device.is_cuda(),
            segment_layer_count,
        );
        // How many of those card-held layers, from the start, keep their
        // routed experts in the machine's memory (#138 (3)) — on a card, for
        // an architecture whose split has been checked, never more than the
        // card holds. The SAME count the placement below acts on and the card
        // budget takes those experts' bytes off by.
        let experts_in_ram_layers = match cpu_expert_layers {
            Some(n) if hybrid_ok && device.is_cuda() => n.min(device_layers),
            _ => 0,
        };
        let experts_in_ram_bytes = crate::inference::split::routed_expert_bytes(
            &ct,
            layer_start,
            layer_start + experts_in_ram_layers,
        );
        // What a conversation's cache costs as it grows over layers `from..to`
        // kept on a card or on the processor (`kv_budget::KvGrowth`), read off
        // the header by the ONE layout every estimate before a load reads too
        // (`kv_budget::KvLayout`): each layer by its own shape, a
        // sliding-window layer charged its window (FUTURE_WORK #281).
        let kv_layout = super::kv_budget::KvLayout::read(&ct, &meta, context_length)?;
        let kv_growth_over =
            |from: usize, to: usize, on_card: bool| kv_layout.growth(from, to, on_card);
        // The positions a budget holds, as the logs below state it: the whole
        // context where the cost stops growing inside the budget.
        let affordable = |growth: &KvGrowth, budget: u64| match growth.positions_within(budget) {
            usize::MAX => context_length,
            positions => positions,
        };
        // The cache ON `device`, needed by both devices' guards below — a
        // card's guard must not count the cache of layers it does not hold.
        let growth = kv_growth_over(layer_start, layer_start + device_layers, device.is_cuda());
        // A card/processor split's OTHER memory (#262): the processor-held
        // layers keep their caches in the machine's, which the card's budget
        // neither covers nor may be charged with. The daemon's figure when it
        // handed one (`--kv-budget-bytes`), else what the machine has
        // available less the margin — the same bound a processor worker is
        // held to as the machine fills (`budget_reconciled_with_device`, every
        // claim). Ollama weighs a split's processor share against the
        // machine's memory the same way (`llm/server.go`).
        let mut host_kv_budget_bytes: Option<u64> = None;
        let mut host_kv_growth = KvGrowth::default();
        if device.is_cuda() && device_layers < segment_layer_count {
            host_kv_growth = kv_growth_over(
                layer_start + device_layers,
                layer_start + segment_layer_count,
                false,
            );
            host_kv_budget_bytes = super::cpu_kv_budget_bytes().or_else(|| {
                super::kv_budget::host_free_and_total_bytes().map(|(available, total)| {
                    available.saturating_sub(super::kv_budget::device_free_margin_bytes(total))
                })
            });
            if let Some(budget) = host_kv_budget_bytes {
                tracing::info!(
                    processor_layers = segment_layer_count - device_layers,
                    budget_mb = budget / (1024 * 1024),
                    kv_bytes_per_position = host_kv_growth.per_position(),
                    affordable_tokens = affordable(&host_kv_growth, budget),
                    "Card/processor split: the processor-held layers' conversation cache is \
                     charged to this machine's memory, the card-held layers' to the card"
                );
            }
        }
        if device.is_cuda() {
            let rows_on_demand = ct
                .tensor_infos
                .get("token_embd.weight")
                .is_some_and(|info| {
                    super::token_embedding::rows_on_demand_eligible(
                        &device,
                        info.ggml_dtype,
                        info.shape.dims(),
                    )
                });
            // The weights THE CARD holds: its layers, less the experts kept in
            // the machine's memory, plus the embedding and head this segment
            // carries (both load on `device` in a hybrid split too).
            let weight_bytes = segment_weight_bytes(
                &ct,
                layer_start,
                layer_start + device_layers,
                is_first,
                is_last,
                rows_on_demand,
            )
            .saturating_sub(experts_in_ram_bytes);

            // A missing nvidia-smi means "unknown", never "zero" — a budget
            // of zero would refuse every request, which is far worse than the
            // no-budget behaviour that preceded this.
            if let Some(free_mb) = crate::model::auto_manage::vram::query_gpu_vram_free_mb() {
                // nvidia-smi counts what this worker's own pool keeps as used;
                // it is this process's to allocate (a range loaded into a
                // worker that has already served, `inference::cuda_pool`).
                let free_bytes = free_mb
                    .saturating_mul(1024 * 1024)
                    .saturating_add(crate::inference::cuda_pool::reusable_bytes(&device));
                let budget = super::kv_budget::kv_headroom_bytes(weight_bytes, free_bytes);
                kv_budget_bytes = Some(budget);
                kv_growth = growth;
                // How far the budget stretches, stated in the units a user
                // thinks in. Logged at INFO when the whole declared context
                // fits and WARN when it does not, because the second case is
                // one where a long conversation can be refused mid-way.
                let affordable = affordable(&growth, budget);
                if affordable >= context_length {
                    tracing::info!(
                        context = context_length,
                        affordable_tokens = affordable,
                        free_vram_mb = free_mb,
                        "GPU KV-cache budget comfortably covers this model's context"
                    );
                } else if affordable == 0 {
                    // NOT the same case as the one below, and it must not say
                    // "shorter conversations are unaffected" — at zero, NOTHING
                    // is unaffected. Every request to this worker is refused
                    // whatever its length.
                    //
                    // The comment above guards nvidia-smi being UNREADABLE
                    // ("unknown, never zero"). This is the other way in: it read
                    // fine and reported so little free memory that the budget
                    // came out at zero, which the `Some(budget)` check then
                    // enforces literally. Measured on this node 2026-08-25
                    // 23:22:48 — six refusals followed, including a 42-token
                    // conversation, plus a 4-way concurrent batch that failed
                    // outright and a streamed reply that came back empty.
                    //
                    // Admission charges KV at `ADMISSION_KV_CONTEXT` and so
                    // should have refused this placement; free memory evidently
                    // fell between the admission decision and the load. The
                    // model still comes up here and then serves nothing, which
                    // is worse than running slowly on the processor — see
                    // `docs/FUTURE_WORK.md`.
                    tracing::warn!(
                        context = context_length,
                        free_vram_mb = free_mb,
                        weight_mb = weight_bytes / (1024 * 1024),
                        kv_bytes_per_position = growth.per_position(),
                        "There is no free graphics memory left for this model's conversation \
                         cache, so it cannot serve from the card — moving it to the processor, \
                         which is slower but answers. Close other GPU programs, or unload \
                         another model, and it will use the card again",
                    );
                    // REFUSE the placement rather than completing it.
                    //
                    // Coming up here means coming up broken: the worker is
                    // alive, advertised and accepting requests, and 503s every
                    // one of them — measured 2026-08-25, six refusals in eleven
                    // minutes including a 42-token conversation, a 4-way batch
                    // that failed outright, and a streamed reply that returned
                    // empty. Serving slowly on the processor beats serving
                    // nothing at all, and that is the choice being made here.
                    //
                    // The error carries `NO_KV_CACHE_ROOM_MARKER`, which
                    // `worker_ipc::permanent_gpu_failure` matches to pin the
                    // model to the CPU; the pool lifts that pin as soon as any
                    // GPU memory is freed, so a transient shortage costs one
                    // model load rather than the rest of the run.
                    //
                    // This is deliberately NOT a load-time device switch. The
                    // daemon charged this model against the VRAM budget at
                    // admission, so quietly landing it in RAM instead would
                    // leave both budgets describing something that is not
                    // there — see `ModelProcessPool::charges_ram`. Placement
                    // changes go through the pool.
                    return Err(SwarmError::ServiceUnavailable(format!(
                        "{} — this model needs about {} MB for a {}-token conversation cache, \
                         and the card has none left after its {} MB of weights; running it on \
                         the processor instead, which is slower but answers",
                        crate::inference::worker_ipc::NO_KV_CACHE_ROOM_MARKER,
                        growth.bytes_for(context_length) / (1024 * 1024),
                        context_length,
                        weight_bytes / (1024 * 1024),
                    )));
                } else {
                    tracing::warn!(
                        context = context_length,
                        affordable_tokens = affordable,
                        free_vram_mb = free_mb,
                        weight_mb = weight_bytes / (1024 * 1024),
                        kv_bytes_per_position = growth.per_position(),
                        "Free GPU memory covers about {affordable} tokens of conversation, \
                         short of this model's {context_length}-token context. Shorter \
                         conversations are unaffected; a long one may be refused once memory \
                         runs out. Close other GPU programs or use a smaller model to raise \
                         this",
                    );
                }
            }
        } else if !device.is_cuda() {
            // A CPU worker's budget is handed to it by the daemon at spawn
            // (`--kv-budget-bytes`, computed at admission from the node's RAM
            // budget). With it in place admission charges a typical context
            // instead of the whole ceiling, and a conversation that outgrows
            // the room is refused here — 503, re-routed — instead of swapping
            // the machine or, before 2026-08-21, never loading at all.
            if let Some(budget) = super::cpu_kv_budget_bytes() {
                kv_budget_bytes = Some(budget);
                kv_growth = growth;
                let affordable = affordable(&growth, budget);
                tracing::info!(
                    context = context_length,
                    affordable_tokens = affordable,
                    budget_mb = budget / (1024 * 1024),
                    kv_bytes_per_position = growth.per_position(),
                    "CPU KV-cache budget: conversations up to about {affordable} tokens fit \
                     in this node's memory budget; longer ones are refused and re-routed"
                );
            }
        }

        // Gemma 2 attention logit soft-capping (from GGUF metadata)
        let attn_logit_softcap = ct
            .metadata
            .get(&format!("{arch}.attn_logit_softcapping"))
            .and_then(|v| v.to_f32().ok())
            .filter(|&v| v > 0.0);

        // Sliding-window attention (Gemma 2), per layer — `SlidingWindow` is
        // the one reading.
        let sliding_window = super::SlidingWindow::read(&ct);
        if let Some(window) = sliding_window.as_ref() {
            tracing::debug!(
                window = window.size,
                windowed_layers = (layer_start..layer_end)
                    .filter(|&il| window.for_layer(il).is_some())
                    .count(),
                "Sliding-window attention"
            );
        }
        // A query scale other than 1/sqrt(head_dim) (Gemma 2's 27B, Gemma 4) —
        // the one reading is `split::query_scale`, per layer since a layer's
        // head width is its own.
        let shape_of = |layer_idx: usize| attention_shapes[layer_idx];
        let query_scale_of =
            |layer_idx: usize| super::query_scale(&ct, shape_of(layer_idx).head_dim);
        if let Some(scale) = query_scale_of(layer_start) {
            tracing::debug!(scale, "Queries scaled for the family's attention scale");
        }
        // Gemma 4: V is normalised per head with no weight before it is cached
        // (`ggml_rms_norm(Vcur)`), and a global layer with no V projection uses
        // its K projection's output as V (`use_alternative_attention`).
        let gemma4 = matches!(model_arch, ModelArch::Gemma4);
        let v_rms_norm = gemma4.then_some(rms_norm_eps);

        // Gemma 2 final logit soft-capping (from GGUF metadata)
        let final_logit_softcap = ct
            .metadata
            .get(&format!("{arch}.final_logit_softcapping"))
            .and_then(|v| v.to_f32().ok())
            .filter(|&v| v > 0.0);

        let use_rope_contiguous = model_arch.use_rope_contiguous();
        let activation = model_arch.default_activation();

        // Long RoPE (SuRoPE) for Phi-3.5 and similar extended-context models
        let longrope = load_longrope_factors(&ct, &mut file, arch, context_length);
        // Per-pair divisors and the attention factor, when the file carries any.
        let rope_factors = if let Some((factors, attn_factor)) = longrope {
            tracing::info!(
                factors_len = factors.len(),
                attn_factor,
                "Using Long RoPE (SuRoPE)"
            );
            Some((factors, attn_factor))
        } else if let Some(factors) = match super::rope::load_rope_freqs(&ct, &mut file, rope_dim) {
            Ok(f) => f,
            // The factors exist but this node can read neither shard 0 nor the
            // sidecar. Serve as every release before this one did — plain RoPE —
            // and say so, rather than refuse layers a request may need: the
            // drift is gradual (98 -> 93 of 100 top-1 by 1,000 positions on
            // llama-3.2-3b), a refusal is total.
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    "This model's RoPE frequency factors are not readable here (no shard 0, no \
                     rope_freqs.bin) — serving with plain RoPE, which drifts from the model on \
                     long conversations. Any shard fetched from HuggingFace brings the sidecar."
                );
                None
            }
        } {
            // Llama 3.1/3.2's scaling, as llama.cpp applies it: each pair's
            // angle divided by its factor, no attention scaling (#124).
            tracing::info!(
                factors_len = factors.len(),
                "Using the model's RoPE frequency factors (rope_freqs)"
            );
            Some((factors, 1.0))
        } else {
            None
        };
        // How each layer turns — `split::rope::LayerRope`, the one reading:
        // a scaling on every position (linear or YaRN), and Gemma 3's windowed
        // layers at a base and scaling of their own.
        let layer_rope = super::rope::LayerRope::read(&ct, rope_freq_base);
        let rope_table =
            |dim: usize, base: f32, scaling: &super::rope::RopeScaling, file_factors: bool| {
                let rope_factors = if file_factors { &rope_factors } else { &None };
                match (rope_factors, scaling) {
                    // Plain or linearly scaled RoPE keeps its own builder: at a
                    // scale of 1, the tables every release before built, bit for bit.
                    (None, super::rope::RopeScaling::Linear(scale)) => {
                        precompute_freqs_cis_scaled(dim, base, *scale, context_length, &device)
                    }
                    // A pair's angle times its multiplier is the same angle as its
                    // divisor over the multiplier; the lifts on cos/sin multiply.
                    // With no scaling, every multiplier and lift is 1 and the
                    // file's divisors are untouched.
                    _ => {
                        let (multipliers, lift) = scaling.per_pair(base, dim);
                        let (divisors, attn_factor) = match rope_factors {
                            Some((factors, attn_factor)) => (factors.clone(), *attn_factor),
                            None => (vec![1.0; dim / 2], 1.0),
                        };
                        let factors: Vec<f32> = divisors
                            .iter()
                            .zip(&multipliers)
                            .map(|(divisor, multiplier)| divisor / multiplier)
                            .collect();
                        precompute_freqs_cis_longrope(
                            dim,
                            base,
                            context_length,
                            &factors,
                            attn_factor * lift,
                            &device,
                        )
                    }
                }
                .map_err(SwarmError::internal)
            };
        let (cos, sin) = rope_table(rope_dim, rope_freq_base, &layer_rope.scaling, true)?;
        let windowed_rope = match layer_rope.windowed {
            Some(w) => {
                tracing::info!(
                    windowed_base = w.base,
                    windowed_scaling = ?w.scaling,
                    windowed_dim = ?w.dim,
                    windowed_file_factors = w.file_factors,
                    base = rope_freq_base,
                    scaling = ?layer_rope.scaling,
                    "Sliding-window layers rotate at their own base"
                );
                Some((
                    w.window,
                    rope_table(
                        w.dim.unwrap_or(rope_dim),
                        w.base,
                        &w.scaling,
                        w.file_factors,
                    )?,
                ))
            }
            None => None,
        };
        if layer_rope.scaling != super::rope::RopeScaling::NONE {
            tracing::info!(scaling = ?layer_rope.scaling, "RoPE scaling");
        }
        // Where each layer of this segment goes.
        //
        // `gpu_layers` is a decision made by `ModelProcessPool` and passed in;
        // nothing here inspects free VRAM. When it is `None`, or the device is
        // not a card, every layer goes on `device` exactly as before — so a
        // model that fits, and every processor-only node, takes an unchanged
        // path.
        if (gpu_layers.is_some() || cpu_expert_layers.is_some()) && device.is_cuda() && !hybrid_ok {
            tracing::warn!(
                arch = %model_arch,
                "Hybrid placement was requested but is not verified for this architecture — \
                 loading the whole segment on the graphics card instead. See \
                 hybrid::arch_supports_hybrid"
            );
        }
        let placement = match device_layers < segment_layer_count {
            true => {
                let n = device_layers;
                tracing::info!(
                    gpu_layers = n,
                    segment_layers = segment_layer_count,
                    "Hybrid placement: first {n} layers on the graphics card, rest on the processor"
                );
                super::hybrid::LayerPlacement::split(
                    layer_start,
                    device.clone(),
                    n,
                    cos.clone(),
                    sin.clone(),
                )
                .map_err(SwarmError::internal)?
            }
            false => super::hybrid::LayerPlacement::uniform(
                layer_start,
                device.clone(),
                cos.clone(),
                sin.clone(),
            )
            .map_err(SwarmError::internal)?,
        }
        .with_experts_on_the_processor(experts_in_ram_layers);
        let placement = match windowed_rope {
            Some((window, (cos, sin))) => placement
                .with_windowed_rope(window, cos, sin)
                .map_err(SwarmError::internal)?,
            None => placement,
        };
        if experts_in_ram_layers > 0 {
            tracing::info!(
                layers = experts_in_ram_layers,
                experts_mb = experts_in_ram_bytes >> 20,
                "Experts in the machine's memory: the routed experts of the first \
                 {experts_in_ram_layers} card-held layers stay in RAM, everything else of \
                 theirs on the graphics card"
            );
        }

        // Helper: create RmsNorm from GGUF weight tensor.
        // Note: GGUF norm weights for Gemma models already include the +1 offset
        // (added by convert_hf_to_gguf.py's modify_tensors), so we use them as-is.
        let make_norm = |qtensor: QTensor, eps: f64| -> Result<RmsNorm, SwarmError> {
            RmsNorm::from_qtensor(qtensor, eps).map_err(SwarmError::internal)
        };

        // Load embedding table only for first segment.
        //
        // Kept as an `Arc` so a weight-tied model can hand the SAME tensor to
        // its LM head below instead of loading `token_embd.weight` a second
        // time. That second load is why the tied models save more than the
        // dequantized-vs-quantized difference alone.
        let tok_embd_q: Option<std::sync::Arc<QTensor>> = if is_first {
            Some(std::sync::Arc::new(
                ct.tensor(&mut file, "token_embd.weight", &device)
                    .map_err(|e| SwarmError::Internal(format!("Failed to load embeddings: {e}")))?,
            ))
        } else {
            None
        };

        let tok_embeddings = match &tok_embd_q {
            None => None,
            Some(q) => match TokenEmbedding::try_quantized(q, &device) {
                // Rows read out of the quantized table as they are looked up —
                // bit-identical to the dense path and several times smaller.
                Some(e) => Some(e),
                // Not eligible (a device with no gather — Metal; the processor
                // and a card both have one —, an unquantized table, or a row
                // that is not block-aligned: `rows_on_demand_eligible`):
                // dequantize the whole table, as before.
                //
                // Resident for the worker's whole life at EMBEDDING_DTYPE — on
                // a large-vocabulary model this is the biggest single
                // allocation the worker makes, so the narrower type is what
                // decides whether a modest GPU can serve the model at all.
                //
                // On CUDA candle has a specialised f16 dequantize kernel for
                // the QUANTIZED block types, so those avoid a transient f32
                // copy of the full table. An UNQUANTIZED GGUF (F16/BF16/F32 —
                // what HuggingFace offers as the "F16" variant) has no such
                // kernel and is converted via the host instead; see the
                // `has_dequantize_kernel` patch in
                // `vendor/candle/candle-core/src/quantized/cuda.rs`. Until that
                // patch this line was the single reason a CUDA node could not
                // serve an unquantized model at all: it loaded, then failed
                // every request.
                None => {
                    let dense = q.dequantize_f16(&device).map_err(SwarmError::internal)?;
                    debug_assert_eq!(dense.dtype(), EMBEDDING_DTYPE);
                    Some(TokenEmbedding::Dense(Embedding::new(
                        dense,
                        embedding_length,
                    )))
                }
            },
        };

        // The embedding table is the largest single allocation on a
        // large-vocabulary model, and which path it took is the difference
        // between fitting on a modest card and not. Say so, rather than
        // leaving it to be inferred from total memory.
        if let Some(emb) = &tok_embeddings {
            tracing::info!(
                rows_read_on_demand = matches!(emb, TokenEmbedding::Quantized(_)),
                resident_mb = emb.resident_bytes() / (1024 * 1024),
                "Loaded token embedding table"
            );
        }

        // Load output norm and LM head only for last segment
        let norm = if is_last {
            let norm_tensor = ct
                .tensor(&mut file, "output_norm.weight", &device)
                .map_err(|e| SwarmError::Internal(format!("Failed to load output_norm: {e}")))?;
            Some(make_norm(norm_tensor, rms_norm_eps)?)
        } else {
            None
        };

        let output = if is_last {
            match ct.tensor(&mut file, "output.weight", &device) {
                // An untied model ships its own head.
                Ok(t) => Some(QMatMul::from_qtensor(t).map_err(SwarmError::internal)?),
                // Weight-tied: the head IS `token_embd.weight`. When this
                // segment is also the first one we already hold that tensor, so
                // share it rather than loading a second copy — on
                // llama-3.2-3b that copy is 308 MB, and it was resident
                // alongside a 751 MB dequantized one for every single-node run.
                Err(_) => {
                    let tied = match &tok_embd_q {
                        Some(q) => std::sync::Arc::clone(q),
                        None => std::sync::Arc::new(
                            ct.tensor(&mut file, "token_embd.weight", &device)
                                .map_err(|e| {
                                    SwarmError::Internal(format!("Failed to load output head: {e}"))
                                })?,
                        ),
                    };
                    Some(QMatMul::from_arc(tied).map_err(SwarmError::internal)?)
                }
            }
        } else {
            None
        };
        let suppressed_logits = match &output {
            Some(head) => {
                let vocab = head.out_features().map_err(SwarmError::internal)?;
                suppressed_logits_row(&ct.metadata, vocab, &device)?
            }
            None => None,
        };

        // Load only the specified layer range (capped at actual block count)
        let layer_end = layer_end.min(block_count);
        let mut layers: Vec<LayerVariant> = Vec::with_capacity(layer_end - layer_start);

        if matches!(model_arch, ModelArch::DeepSeek2) {
            // ── DeepSeek-V2/V3: MLA + MoE loading ──
            let ds_meta = DeepSeekMeta {
                n_experts: md_get("expert_count")
                    .and_then(|v| v.to_u32().map_err(SwarmError::internal))
                    .unwrap_or(0) as usize,
                n_experts_used: md_get("expert_used_count")
                    .and_then(|v| v.to_u32().map_err(SwarmError::internal))
                    .unwrap_or(6) as usize,
                n_shared_experts: md_get("expert_shared_count")
                    .and_then(|v| v.to_u32().map_err(SwarmError::internal))
                    .unwrap_or(0) as usize,
                kv_lora_rank: md_get("attention.kv_lora_rank")
                    .and_then(|v| v.to_u32().map_err(SwarmError::internal))
                    .unwrap_or(512) as usize,
                q_lora_rank: md_get("attention.q_lora_rank")
                    .and_then(|v| v.to_u32().map_err(SwarmError::internal))
                    .unwrap_or(0) as usize,
                key_length: md_get("attention.key_length")
                    .and_then(|v| v.to_u32().map_err(SwarmError::internal))
                    .unwrap_or(head_dim as u32) as usize,
                value_length: md_get("attention.value_length")
                    .and_then(|v| v.to_u32().map_err(SwarmError::internal))
                    .unwrap_or(head_dim as u32) as usize,
                rope_dim,
            };

            // DeepSeek RoPE may use a different dimension for MLA layers
            let mla_rope_dim = ds_meta.rope_dim;
            let (mla_cos, mla_sin) =
                precompute_freqs_cis(mla_rope_dim, rope_freq_base, context_length, &device)
                    .map_err(SwarmError::internal)?;

            tracing::info!(
                n_experts = ds_meta.n_experts,
                n_experts_used = ds_meta.n_experts_used,
                n_shared_experts = ds_meta.n_shared_experts,
                kv_lora_rank = ds_meta.kv_lora_rank,
                q_lora_rank = ds_meta.q_lora_rank,
                key_length = ds_meta.key_length,
                value_length = ds_meta.value_length,
                "Loading DeepSeek-V2/V3 MoE+MLA model"
            );

            for layer_idx in layer_start..layer_end {
                // Per-layer placement, applied by shadowing so that every
                // tensor load below picks up the right device with no
                // per-call-site edit — see `hybrid::LayerPlacement`.
                let device = placement.device_for(layer_idx);
                let (cos, sin) = placement.rope_for(layer_idx);
                let prefix = format!("blk.{layer_idx}");

                // Per-layer detection: MLA vs dense attention
                let has_mla = ct
                    .tensor_infos
                    .contains_key(&format!("{prefix}.attn_q_a.weight"));
                // Per-layer detection: MoE vs dense FFN
                let has_moe = ct
                    .tensor_infos
                    .contains_key(&format!("{prefix}.ffn_gate_exps.weight"));

                let attn_norm = ct
                    .tensor(&mut file, &format!("{prefix}.attn_norm.weight"), &device)
                    .map_err(|e| {
                        SwarmError::Internal(format!("Failed to load {prefix}.attn_norm: {e}"))
                    })?;
                let ffn_norm_t = ct
                    .tensor(&mut file, &format!("{prefix}.ffn_norm.weight"), &device)
                    .map_err(|e| {
                        SwarmError::Internal(format!("Failed to load {prefix}.ffn_norm: {e}"))
                    })?;

                if has_mla {
                    // MLA attention weights
                    let q_a = ct
                        .tensor(&mut file, &format!("{prefix}.attn_q_a.weight"), &device)
                        .map_err(|e| SwarmError::Internal(format!("{prefix}.attn_q_a: {e}")))?;
                    let q_a_norm_t = ct
                        .tensor(
                            &mut file,
                            &format!("{prefix}.attn_q_a_norm.weight"),
                            &device,
                        )
                        .map_err(|e| {
                            SwarmError::Internal(format!("{prefix}.attn_q_a_norm: {e}"))
                        })?;
                    let q_b = ct
                        .tensor(&mut file, &format!("{prefix}.attn_q_b.weight"), &device)
                        .map_err(|e| SwarmError::Internal(format!("{prefix}.attn_q_b: {e}")))?;
                    let kv_a = ct
                        .tensor(&mut file, &format!("{prefix}.attn_kv_a.weight"), &device)
                        .map_err(|e| SwarmError::Internal(format!("{prefix}.attn_kv_a: {e}")))?;
                    let kv_a_norm_t = ct
                        .tensor(
                            &mut file,
                            &format!("{prefix}.attn_kv_a_norm.weight"),
                            &device,
                        )
                        .map_err(|e| {
                            SwarmError::Internal(format!("{prefix}.attn_kv_a_norm: {e}"))
                        })?;
                    let kv_b = ct
                        .tensor(&mut file, &format!("{prefix}.attn_kv_b.weight"), &device)
                        .map_err(|e| SwarmError::Internal(format!("{prefix}.attn_kv_b: {e}")))?;
                    let wo = ct
                        .tensor(&mut file, &format!("{prefix}.attn_output.weight"), &device)
                        .map_err(|e| SwarmError::Internal(format!("{prefix}.attn_output: {e}")))?;

                    let attention = MlaWeights {
                        q_a: QMatMul::from_qtensor(q_a).map_err(SwarmError::internal)?,
                        q_a_norm: RmsNorm::from_qtensor(q_a_norm_t, rms_norm_eps)
                            .map_err(SwarmError::internal)?,
                        q_b: QMatMul::from_qtensor(q_b).map_err(SwarmError::internal)?,
                        kv_a: QMatMul::from_qtensor(kv_a).map_err(SwarmError::internal)?,
                        kv_a_norm: RmsNorm::from_qtensor(kv_a_norm_t, rms_norm_eps)
                            .map_err(SwarmError::internal)?,
                        kv_b: QMatMul::from_qtensor(kv_b).map_err(SwarmError::internal)?,
                        output: QMatMul::from_qtensor(wo).map_err(SwarmError::internal)?,
                        n_head: head_count,
                        key_length: ds_meta.key_length,
                        value_length: ds_meta.value_length,
                        kv_lora_rank: ds_meta.kv_lora_rank,
                        rope_dim: mla_rope_dim,
                        cos: mla_cos.clone(),
                        sin: mla_sin.clone(),
                    };

                    // FFN: MoE or dense
                    let ffn = if has_moe {
                        FfnVariant::MoE(load_moe_ffn(
                            &ct,
                            &mut file,
                            &prefix,
                            &placement,
                            layer_idx,
                            ds_meta.n_experts_used,
                            moe_routing,
                            Activation::SiLU,
                        )?)
                    } else {
                        // Dense FFN for early DeepSeek layers
                        let ffn_gate = ct
                            .tensor(&mut file, &format!("{prefix}.ffn_gate.weight"), &device)
                            .map_err(|e| SwarmError::Internal(format!("{prefix}.ffn_gate: {e}")))?;
                        let ffn_down = ct
                            .tensor(&mut file, &format!("{prefix}.ffn_down.weight"), &device)
                            .map_err(|e| SwarmError::Internal(format!("{prefix}.ffn_down: {e}")))?;
                        let ffn_up = ct
                            .tensor(&mut file, &format!("{prefix}.ffn_up.weight"), &device)
                            .map_err(|e| SwarmError::Internal(format!("{prefix}.ffn_up: {e}")))?;
                        FfnVariant::Dense(Mlp {
                            ffn_gate: Some(
                                QMatMul::from_qtensor(ffn_gate).map_err(SwarmError::internal)?,
                            ),
                            ffn_down: QMatMul::from_qtensor(ffn_down)
                                .map_err(SwarmError::internal)?,
                            ffn_up: QMatMul::from_qtensor(ffn_up).map_err(SwarmError::internal)?,
                            activation: Activation::SiLU,
                        })
                    };

                    layers.push(LayerVariant::DeepSeek {
                        attention,
                        ffn,
                        attention_norm: make_norm(attn_norm, rms_norm_eps)?,
                        ffn_norm: make_norm(ffn_norm_t, rms_norm_eps)?,
                    });
                } else {
                    // Dense attention layer (first few DeepSeek layers use standard attention)
                    let attention_wq = ct
                        .tensor(&mut file, &format!("{prefix}.attn_q.weight"), &device)
                        .map_err(|e| SwarmError::Internal(format!("{prefix}.attn_q: {e}")))?;
                    let attention_wk = ct
                        .tensor(&mut file, &format!("{prefix}.attn_k.weight"), &device)
                        .map_err(|e| SwarmError::Internal(format!("{prefix}.attn_k: {e}")))?;
                    let attention_wv = ct
                        .tensor(&mut file, &format!("{prefix}.attn_v.weight"), &device)
                        .map_err(|e| SwarmError::Internal(format!("{prefix}.attn_v: {e}")))?;
                    let attention_wo = ct
                        .tensor(&mut file, &format!("{prefix}.attn_output.weight"), &device)
                        .map_err(|e| SwarmError::Internal(format!("{prefix}.attn_output: {e}")))?;
                    let attention_bq = ct
                        .tensor(&mut file, &format!("{prefix}.attn_q.bias"), &device)
                        .ok()
                        .map(|t| t.dequantize(&device))
                        .transpose()
                        .map_err(|e| SwarmError::Internal(format!("attn_q.bias: {e}")))?;
                    let attention_bk = ct
                        .tensor(&mut file, &format!("{prefix}.attn_k.bias"), &device)
                        .ok()
                        .map(|t| t.dequantize(&device))
                        .transpose()
                        .map_err(|e| SwarmError::Internal(format!("attn_k.bias: {e}")))?;
                    let attention_bv = ct
                        .tensor(&mut file, &format!("{prefix}.attn_v.bias"), &device)
                        .ok()
                        .map(|t| t.dequantize(&device))
                        .transpose()
                        .map_err(|e| SwarmError::Internal(format!("attn_v.bias: {e}")))?;

                    // Dense FFN for early DeepSeek layers
                    let ffn_gate = ct
                        .tensor(&mut file, &format!("{prefix}.ffn_gate.weight"), &device)
                        .map_err(|e| SwarmError::Internal(format!("{prefix}.ffn_gate: {e}")))?;
                    let ffn_down = ct
                        .tensor(&mut file, &format!("{prefix}.ffn_down.weight"), &device)
                        .map_err(|e| SwarmError::Internal(format!("{prefix}.ffn_down: {e}")))?;
                    let ffn_up = ct
                        .tensor(&mut file, &format!("{prefix}.ffn_up.weight"), &device)
                        .map_err(|e| SwarmError::Internal(format!("{prefix}.ffn_up: {e}")))?;

                    // Dense DeepSeek layers use standard attention — wrap as LayerVariant::Dense
                    layers.push(LayerVariant::Dense(LayerWeights {
                        attention_wq: QMatMul::from_qtensor(attention_wq)
                            .map_err(SwarmError::internal)?,
                        attention_wk: QMatMul::from_qtensor(attention_wk)
                            .map_err(SwarmError::internal)?,
                        attention_wv: QMatMul::from_qtensor(attention_wv)
                            .map_err(SwarmError::internal)?,
                        attention_wo: QMatMul::from_qtensor(attention_wo)
                            .map_err(SwarmError::internal)?,
                        attention_bq,
                        attention_bk,
                        attention_bv,
                        attention_norm: make_norm(attn_norm, rms_norm_eps)?,
                        attn_q_norm: None,
                        attn_k_norm: None,
                        ffn: FfnVariant::Dense(Mlp {
                            ffn_gate: Some(
                                QMatMul::from_qtensor(ffn_gate).map_err(SwarmError::internal)?,
                            ),
                            ffn_down: QMatMul::from_qtensor(ffn_down)
                                .map_err(SwarmError::internal)?,
                            ffn_up: QMatMul::from_qtensor(ffn_up).map_err(SwarmError::internal)?,
                            activation: Activation::SiLU,
                        }),
                        ffn_norm: make_norm(ffn_norm_t, rms_norm_eps)?,
                        post_attention_norm: None,
                        post_ffw_norm: None,
                        n_head: head_count,
                        n_kv_head: head_count_kv,
                        head_dim,
                        cos: cos.clone(),
                        sin: sin.clone(),
                        use_rope_contiguous,
                        attn_logit_softcap,
                        rope_dim,
                        skip_rope: false,
                        qk_rms_norm_after_rope: None,
                        sliding_window: None,
                        query_scale: None,
                        v_rms_norm: None,
                        output_scale: None,
                    }));
                }
            }
        } else if matches!(model_arch, ModelArch::Llama4) {
            // ── Llama 4 Scout/Maverick: iRoPE + MoE loading ──
            // iRoPE pattern: every 4th layer (index % 4 == 3) is NoPE (no positional encoding)
            // MoE: router selects top-k experts from stacked expert tensors

            // Read Llama 4 MoE metadata
            let n_experts = ct
                .metadata
                .get(&format!("{arch}.expert_count"))
                .and_then(|v| v.to_u32().ok())
                .unwrap_or(0) as usize;
            let n_experts_used = ct
                .metadata
                .get(&format!("{arch}.expert_used_count"))
                .and_then(|v| v.to_u32().ok())
                .unwrap_or(1) as usize;

            for layer_idx in layer_start..layer_end {
                // Per-layer placement, applied by shadowing so that every
                // tensor load below picks up the right device with no
                // per-call-site edit — see `hybrid::LayerPlacement`.
                let device = placement.device_for(layer_idx);
                let (cos, sin) = placement.rope_for(layer_idx);
                let prefix = format!("blk.{layer_idx}");
                let is_nope = layer_idx % 4 == 3; // NoPE every 4th layer

                let attention_wq = ct
                    .tensor(&mut file, &format!("{prefix}.attn_q.weight"), &device)
                    .map_err(|e| SwarmError::Internal(format!("{prefix}.attn_q: {e}")))?;
                let attention_wk = ct
                    .tensor(&mut file, &format!("{prefix}.attn_k.weight"), &device)
                    .map_err(|e| SwarmError::Internal(format!("{prefix}.attn_k: {e}")))?;
                let attention_wv = ct
                    .tensor(&mut file, &format!("{prefix}.attn_v.weight"), &device)
                    .map_err(|e| SwarmError::Internal(format!("{prefix}.attn_v: {e}")))?;
                let attention_wo = ct
                    .tensor(&mut file, &format!("{prefix}.attn_output.weight"), &device)
                    .map_err(|e| SwarmError::Internal(format!("{prefix}.attn_output: {e}")))?;
                let attn_norm = ct
                    .tensor(&mut file, &format!("{prefix}.attn_norm.weight"), &device)
                    .map_err(|e| SwarmError::Internal(format!("{prefix}.attn_norm: {e}")))?;
                let ffn_norm = ct
                    .tensor(&mut file, &format!("{prefix}.ffn_norm.weight"), &device)
                    .map_err(|e| SwarmError::Internal(format!("{prefix}.ffn_norm: {e}")))?;

                // QKV biases (optional)
                let attention_bq = ct
                    .tensor(&mut file, &format!("{prefix}.attn_q.bias"), &device)
                    .ok()
                    .map(|t| t.dequantize(&device))
                    .transpose()
                    .map_err(|e| SwarmError::Internal(format!("attn_q.bias: {e}")))?;
                let attention_bk = ct
                    .tensor(&mut file, &format!("{prefix}.attn_k.bias"), &device)
                    .ok()
                    .map(|t| t.dequantize(&device))
                    .transpose()
                    .map_err(|e| SwarmError::Internal(format!("attn_k.bias: {e}")))?;
                let attention_bv = ct
                    .tensor(&mut file, &format!("{prefix}.attn_v.bias"), &device)
                    .ok()
                    .map(|t| t.dequantize(&device))
                    .transpose()
                    .map_err(|e| SwarmError::Internal(format!("attn_v.bias: {e}")))?;

                // Check if this layer has MoE
                let has_moe = ct
                    .tensor_infos
                    .contains_key(&format!("{prefix}.ffn_gate_exps.weight"));

                let ffn = if has_moe && n_experts > 0 {
                    FfnVariant::MoE(load_moe_ffn(
                        &ct,
                        &mut file,
                        &prefix,
                        &placement,
                        layer_idx,
                        n_experts_used,
                        moe_routing,
                        Activation::SiLU,
                    )?)
                } else {
                    // Dense FFN — gate is optional (absent in Starcoder2's 2-layer MLP)
                    let ffn_gate_t = ct
                        .tensor(&mut file, &format!("{prefix}.ffn_gate.weight"), &device)
                        .ok();
                    let ffn_down_t = ct
                        .tensor(&mut file, &format!("{prefix}.ffn_down.weight"), &device)
                        .map_err(|e| SwarmError::Internal(format!("{prefix}.ffn_down: {e}")))?;
                    let ffn_up_t = ct
                        .tensor(&mut file, &format!("{prefix}.ffn_up.weight"), &device)
                        .map_err(|e| SwarmError::Internal(format!("{prefix}.ffn_up: {e}")))?;
                    FfnVariant::Dense(Mlp {
                        ffn_gate: ffn_gate_t
                            .map(QMatMul::from_qtensor)
                            .transpose()
                            .map_err(SwarmError::internal)?,
                        ffn_down: QMatMul::from_qtensor(ffn_down_t)
                            .map_err(SwarmError::internal)?,
                        ffn_up: QMatMul::from_qtensor(ffn_up_t).map_err(SwarmError::internal)?,
                        activation,
                    })
                };

                // Both MoE and dense FFN layers use standard attention via LayerVariant::Dense.
                // The FfnVariant enum handles MoE vs dense FFN dispatch in the forward pass.
                layers.push(LayerVariant::Dense(LayerWeights {
                    attention_wq: QMatMul::from_qtensor(attention_wq)
                        .map_err(SwarmError::internal)?,
                    attention_wk: QMatMul::from_qtensor(attention_wk)
                        .map_err(SwarmError::internal)?,
                    attention_wv: QMatMul::from_qtensor(attention_wv)
                        .map_err(SwarmError::internal)?,
                    attention_wo: QMatMul::from_qtensor(attention_wo)
                        .map_err(SwarmError::internal)?,
                    attention_bq,
                    attention_bk,
                    attention_bv,
                    attention_norm: make_norm(attn_norm, rms_norm_eps)?,
                    attn_q_norm: None,
                    attn_k_norm: None,
                    ffn,
                    ffn_norm: make_norm(ffn_norm, rms_norm_eps)?,
                    post_attention_norm: None,
                    post_ffw_norm: None,
                    n_head: head_count,
                    n_kv_head: head_count_kv,
                    head_dim,
                    cos: cos.clone(),
                    sin: sin.clone(),
                    use_rope_contiguous,
                    attn_logit_softcap,
                    rope_dim,
                    skip_rope: is_nope,
                    // llama.cpp `use_kq_norm`: on for every Llama 4 but the
                    // 128-expert Maverick, and only where RoPE runs.
                    qk_rms_norm_after_rope: (!is_nope && n_experts != 128).then_some(rms_norm_eps),
                    sliding_window: None,
                    query_scale: None,
                    v_rms_norm: None,
                    output_scale: None,
                }));
            }
        } else if model_arch.is_hybrid_ssm() {
            // ── Qwen 3.5: Gated DeltaNet layers + a full-attention layer every
            // `full_attention_interval`-th — written against llama.cpp's
            // `models/qwen35.cpp` and a real file's tensor list
            // (`docs/plans/qwen35_support.md`, FUTURE_WORK #117). The names
            // below are the GGUF's; llama.cpp's C++ members differ
            // (`attn_post_norm` ← `post_attention_norm`, `ssm_dt` ←
            // `ssm_dt.bias`), and reading the members as names is how the
            // first version of this loader could load no real file.
            let md_usize = |suffix: &str| -> Result<usize, SwarmError> {
                md_get(suffix)
                    .and_then(|v| v.to_u32().map_err(SwarmError::internal))
                    .map(|v| v as usize)
            };
            let conv_kernel = md_usize("ssm.conv_kernel")?;
            let state_size = md_usize("ssm.state_size")?;
            let n_k_heads = md_usize("ssm.group_count")?;
            let n_v_heads = md_usize("ssm.time_step_rank")?;
            let is_moe = matches!(model_arch, ModelArch::Qwen35Moe);
            tracing::info!(
                conv_kernel,
                state_size,
                n_k_heads,
                n_v_heads,
                rope_dim,
                is_moe,
                "Loading Qwen 3.5 (Gated DeltaNet + attention)"
            );
            let q35_tensor = |file: &mut R, name: String, device: &Device| {
                ct.tensor(file, &name, device)
                    .map_err(|e| SwarmError::Internal(format!("{name}: {e}")))
            };
            let q35_dense = |file: &mut R, name: String, device: &Device| {
                q35_tensor(file, name, device)?
                    .dequantize(device)
                    .map_err(SwarmError::internal)
            };
            let q35_mm = |file: &mut R, name: String, device: &Device| {
                QMatMul::from_qtensor(q35_tensor(file, name, device)?).map_err(SwarmError::internal)
            };

            for layer_idx in layer_start..layer_end {
                // Per-layer placement, applied by shadowing — see
                // `hybrid::LayerPlacement`. Every tensor below loads through
                // these names; that is what puts Qwen 3.5 on the hybrid
                // allowlist (`hybrid::arch_supports_hybrid`, FUTURE_WORK #228).
                let device = placement.device_for(layer_idx);
                let (cos, sin) = placement.rope_for(layer_idx);
                let prefix = format!("blk.{layer_idx}");
                let is_deltanet = super::layer_is_recurrent(&ct, layer_idx);

                let attn_norm =
                    q35_tensor(&mut *file, format!("{prefix}.attn_norm.weight"), &device)?;
                let post_attn_norm = q35_tensor(
                    &mut *file,
                    format!("{prefix}.post_attention_norm.weight"),
                    &device,
                )?;
                let ffn = if is_moe
                    && ct
                        .tensor_infos
                        .contains_key(&format!("{prefix}.ffn_gate_exps.weight"))
                {
                    let n_experts_used = md_usize("expert_used_count").unwrap_or(2);
                    FfnVariant::MoE(load_moe_ffn(
                        &ct,
                        &mut file,
                        &prefix,
                        &placement,
                        layer_idx,
                        n_experts_used,
                        moe_routing,
                        Activation::SiLU,
                    )?)
                } else {
                    FfnVariant::Dense(Mlp {
                        ffn_gate: Some(q35_mm(
                            &mut *file,
                            format!("{prefix}.ffn_gate.weight"),
                            &device,
                        )?),
                        ffn_down: q35_mm(&mut *file, format!("{prefix}.ffn_down.weight"), &device)?,
                        ffn_up: q35_mm(&mut *file, format!("{prefix}.ffn_up.weight"), &device)?,
                        activation,
                    })
                };

                if is_deltanet {
                    let conv1d =
                        q35_dense(&mut *file, format!("{prefix}.ssm_conv1d.weight"), &device)?;
                    layers.push(LayerVariant::Qwen35Ssm {
                        weights: DeltaNetWeights {
                            wqkv: q35_mm(&mut *file, format!("{prefix}.attn_qkv.weight"), &device)?,
                            wz: q35_mm(&mut *file, format!("{prefix}.attn_gate.weight"), &device)?,
                            w_beta: q35_mm(
                                &mut *file,
                                format!("{prefix}.ssm_beta.weight"),
                                &device,
                            )?,
                            w_alpha: q35_mm(
                                &mut *file,
                                format!("{prefix}.ssm_alpha.weight"),
                                &device,
                            )?,
                            dt_bias: q35_dense(
                                &mut *file,
                                format!("{prefix}.ssm_dt.bias"),
                                &device,
                            )?,
                            a: q35_dense(&mut *file, format!("{prefix}.ssm_a"), &device)?,
                            conv1d,
                            ssm_norm: RmsNorm::from_qtensor(
                                q35_tensor(
                                    &mut *file,
                                    format!("{prefix}.ssm_norm.weight"),
                                    &device,
                                )?,
                                rms_norm_eps,
                            )
                            .map_err(SwarmError::internal)?,
                            ssm_out: q35_mm(
                                &mut *file,
                                format!("{prefix}.ssm_out.weight"),
                                &device,
                            )?,
                            n_k_heads,
                            k_head_dim: state_size,
                            n_v_heads,
                            v_head_dim: state_size,
                            conv_kernel,
                            eps: rms_norm_eps,
                        },
                        ffn,
                        attention_norm: make_norm(attn_norm, rms_norm_eps)?,
                        post_attention_norm: make_norm(post_attn_norm, rms_norm_eps)?,
                    });
                } else {
                    let norm = |file: &mut R, name: String| {
                        RmsNorm::from_qtensor(q35_tensor(file, name, &device)?, rms_norm_eps)
                            .map_err(SwarmError::internal)
                    };
                    layers.push(LayerVariant::Qwen35Attn {
                        weights: Qwen35AttnWeights {
                            wq_gate: q35_mm(
                                &mut *file,
                                format!("{prefix}.attn_q.weight"),
                                &device,
                            )?,
                            wk: q35_mm(&mut *file, format!("{prefix}.attn_k.weight"), &device)?,
                            wv: q35_mm(&mut *file, format!("{prefix}.attn_v.weight"), &device)?,
                            wo: q35_mm(
                                &mut *file,
                                format!("{prefix}.attn_output.weight"),
                                &device,
                            )?,
                            q_norm: norm(&mut *file, format!("{prefix}.attn_q_norm.weight"))?,
                            k_norm: norm(&mut *file, format!("{prefix}.attn_k_norm.weight"))?,
                            n_head: head_count,
                            n_kv_head: head_count_kv,
                            head_dim,
                            cos: cos.clone(),
                            sin: sin.clone(),
                            rope_dim,
                        },
                        ffn,
                        attention_norm: make_norm(attn_norm, rms_norm_eps)?,
                        post_attention_norm: make_norm(post_attn_norm, rms_norm_eps)?,
                    });
                }
            }
        } else {
            // ── Standard dense architecture loading (Llama, Qwen2, Gemma, GLM-4, etc.) ──
            //
            // Qwen2-MoE and Qwen3-MoE (`qwen2moe`, `qwen3moe`) are this layout
            // with the feed-forward swapped for routed experts: a layer holding
            // `ffn_gate_exps` loads one through `load_moe_ffn`, every other
            // layer is dense exactly as before. Until 2026-09-25 this branch
            // required `ffn_down` of every layer, so `qwen2moe` — listed as
            // supported — failed to load, and `qwen3moe` was refused outright.
            // Gemma 4's 26B-A4B holds routed experts in every layer with a
            // dense feed-forward beside them (`load_standard_ffn`).
            let experts_used = ct
                .metadata
                .get(&format!("{arch}.expert_used_count"))
                .and_then(|v| v.to_u32().ok())
                .unwrap_or(0) as usize;
            let standard_ffn = StandardFfn {
                experts_used,
                routing: moe_routing,
                activation,
                gemma4,
                rms_norm_eps,
            };
            if let Some(mmap_ref) = parallel_data {
                // Parallel layer loading: each thread gets its own Cursor into mmap'd data.
                // ~N× speedup for N layers on NVMe/SSD.
                let ct_ref = &ct;
                let layer_results: Vec<Result<LayerVariant, SwarmError>> =
                    std::thread::scope(|s| {
                        let handles: Vec<_> = (layer_start..layer_end)
                            .map(|layer_idx| {
                                // Each layer on the device its placement names,
                                // with that device's RoPE tables — the SHADOWING
                                // the other per-layer loops do (see the sequential
                                // loop below). This loop took the segment's
                                // `device` for every layer, so a hybrid split
                                // loaded WHOLE on the card on this (mmap) path
                                // while the model recorded (`placement.devices`)
                                // and the KV budget charged
                                // (`hybrid::layers_on_device`) only the first n
                                // there (review of #104, 2026-09-25).
                                let layer_device = placement.device_for(layer_idx);
                                let (cos, sin) = placement.rope_for(layer_idx);
                                let window =
                                    sliding_window.as_ref().and_then(|w| w.for_layer(layer_idx));
                                let shape = shape_of(layer_idx);
                                let query_scale = query_scale_of(layer_idx);
                                // Borrowed into the thread: a mixture-of-experts
                                // layer asks it where its experts go.
                                let placement = &placement;
                                s.spawn(move || -> Result<LayerVariant, SwarmError> {
                                    let device_ref = &layer_device;
                                    let mut cursor = std::io::Cursor::new(mmap_ref);
                                    let prefix = format!("blk.{layer_idx}");

                                    // Try separate Q/K/V first; fall back to fused attn_qkv
                                    // (Phi-3 uses fused attn_qkv.weight instead of separate attn_q/k/v)
                                    let has_fused_qkv = ct_ref
                                        .tensor_infos
                                        .contains_key(&format!("{prefix}.attn_qkv.weight"));
                                    let (qkv_q, qkv_k, qkv_v) = if has_fused_qkv {
                                        // Fused QKV: keep original quantized tensor, split output in forward
                                        let fused_qt = ct_ref
                                            .tensor(
                                                &mut cursor,
                                                &format!("{prefix}.attn_qkv.weight"),
                                                device_ref,
                                            )
                                            .map_err(|e| {
                                                SwarmError::Internal(format!(
                                                    "{prefix}.attn_qkv: {e}"
                                                ))
                                            })?;
                                        let q_dim = head_count * shape.head_dim;
                                        let k_dim = shape.kv_width();
                                        let fused = QMatMul::make_fused(fused_qt)
                                            .map_err(SwarmError::internal)?;
                                        (
                                            QMatMul::from_fused_slice(fused.clone(), 0, q_dim),
                                            QMatMul::from_fused_slice(fused.clone(), q_dim, k_dim),
                                            QMatMul::from_fused_slice(fused, q_dim + k_dim, k_dim),
                                        )
                                    } else {
                                        let wq = ct_ref
                                            .tensor(
                                                &mut cursor,
                                                &format!("{prefix}.attn_q.weight"),
                                                device_ref,
                                            )
                                            .map_err(|e| {
                                                SwarmError::Internal(format!(
                                                    "Failed to load {prefix}.attn_q: {e}"
                                                ))
                                            })?;
                                        let wk = ct_ref
                                            .tensor(
                                                &mut cursor,
                                                &format!("{prefix}.attn_k.weight"),
                                                device_ref,
                                            )
                                            .map_err(|e| {
                                                SwarmError::Internal(format!(
                                                    "Failed to load {prefix}.attn_k: {e}"
                                                ))
                                            })?;
                                        let wk = QMatMul::from_qtensor(wk)
                                            .map_err(SwarmError::internal)?;
                                        let wv = if value_is_key(ct_ref, gemma4, &prefix) {
                                            wk.clone()
                                        } else {
                                            let wv = ct_ref
                                                .tensor(
                                                    &mut cursor,
                                                    &format!("{prefix}.attn_v.weight"),
                                                    device_ref,
                                                )
                                                .map_err(|e| {
                                                    SwarmError::Internal(format!(
                                                        "Failed to load {prefix}.attn_v: {e}"
                                                    ))
                                                })?;
                                            QMatMul::from_qtensor(wv)
                                                .map_err(SwarmError::internal)?
                                        };
                                        (
                                            QMatMul::from_qtensor(wq)
                                                .map_err(SwarmError::internal)?,
                                            wk,
                                            wv,
                                        )
                                    };
                                    let attention_wo = ct_ref
                                        .tensor(
                                            &mut cursor,
                                            &format!("{prefix}.attn_output.weight"),
                                            device_ref,
                                        )
                                        .map_err(|e| {
                                            SwarmError::Internal(format!(
                                                "Failed to load {prefix}.attn_output: {e}"
                                            ))
                                        })?;

                                    let attention_bq = ct_ref
                                        .tensor(
                                            &mut cursor,
                                            &format!("{prefix}.attn_q.bias"),
                                            device_ref,
                                        )
                                        .ok()
                                        .map(|t| t.dequantize(device_ref))
                                        .transpose()
                                        .map_err(|e| {
                                            SwarmError::Internal(format!(
                                                "attn_q.bias dequant: {e}"
                                            ))
                                        })?;
                                    let attention_bk = ct_ref
                                        .tensor(
                                            &mut cursor,
                                            &format!("{prefix}.attn_k.bias"),
                                            device_ref,
                                        )
                                        .ok()
                                        .map(|t| t.dequantize(device_ref))
                                        .transpose()
                                        .map_err(|e| {
                                            SwarmError::Internal(format!(
                                                "attn_k.bias dequant: {e}"
                                            ))
                                        })?;
                                    let attention_bv = ct_ref
                                        .tensor(
                                            &mut cursor,
                                            &format!("{prefix}.attn_v.bias"),
                                            device_ref,
                                        )
                                        .ok()
                                        .map(|t| t.dequantize(device_ref))
                                        .transpose()
                                        .map_err(|e| {
                                            SwarmError::Internal(format!(
                                                "attn_v.bias dequant: {e}"
                                            ))
                                        })?;

                                    let ffn = load_standard_ffn(
                                        ct_ref,
                                        &mut cursor,
                                        &prefix,
                                        placement,
                                        layer_idx,
                                        standard_ffn,
                                    )?;
                                    let attn_norm = ct_ref
                                        .tensor(
                                            &mut cursor,
                                            &format!("{prefix}.attn_norm.weight"),
                                            device_ref,
                                        )
                                        .map_err(|e| {
                                            SwarmError::Internal(format!(
                                                "Failed to load {prefix}.attn_norm: {e}"
                                            ))
                                        })?;
                                    let ffn_norm = ct_ref
                                        .tensor(
                                            &mut cursor,
                                            &format!("{prefix}.ffn_norm.weight"),
                                            device_ref,
                                        )
                                        .map_err(|e| {
                                            SwarmError::Internal(format!(
                                                "Failed to load {prefix}.ffn_norm: {e}"
                                            ))
                                        })?;

                                    // Qwen3 QK normalization (optional)
                                    let attn_q_norm = ct_ref
                                        .tensor(
                                            &mut cursor,
                                            &format!("{prefix}.attn_q_norm.weight"),
                                            device_ref,
                                        )
                                        .ok()
                                        .map(|t| RmsNorm::from_qtensor(t, rms_norm_eps))
                                        .transpose()
                                        .map_err(|e| {
                                            SwarmError::Internal(format!("attn_q_norm: {e}"))
                                        })?;
                                    let attn_k_norm = ct_ref
                                        .tensor(
                                            &mut cursor,
                                            &format!("{prefix}.attn_k_norm.weight"),
                                            device_ref,
                                        )
                                        .ok()
                                        .map(|t| RmsNorm::from_qtensor(t, rms_norm_eps))
                                        .transpose()
                                        .map_err(|e| {
                                            SwarmError::Internal(format!("attn_k_norm: {e}"))
                                        })?;

                                    // Gemma 2 post-norms (optional)
                                    let post_attention_norm = ct_ref
                                        .tensor(
                                            &mut cursor,
                                            &format!("{prefix}.post_attention_norm.weight"),
                                            device_ref,
                                        )
                                        .ok()
                                        .map(|t| make_norm(t, rms_norm_eps))
                                        .transpose()?;
                                    let post_ffw_norm = ct_ref
                                        .tensor(
                                            &mut cursor,
                                            &format!("{prefix}.post_ffw_norm.weight"),
                                            device_ref,
                                        )
                                        .ok()
                                        .map(|t| make_norm(t, rms_norm_eps))
                                        .transpose()?;

                                    let output_scale =
                                        layer_output_scale(ct_ref, &mut cursor, &prefix)?;
                                    Ok(LayerVariant::Dense(LayerWeights {
                                        attention_wq: qkv_q,
                                        attention_wk: qkv_k,
                                        attention_wv: qkv_v,
                                        attention_wo: QMatMul::from_qtensor(attention_wo)
                                            .map_err(SwarmError::internal)?,
                                        attention_bq,
                                        attention_bk,
                                        attention_bv,
                                        attention_norm: make_norm(attn_norm, rms_norm_eps)?,
                                        attn_q_norm,
                                        attn_k_norm,
                                        ffn,
                                        ffn_norm: make_norm(ffn_norm, rms_norm_eps)?,
                                        post_attention_norm,
                                        post_ffw_norm,
                                        n_head: head_count,
                                        n_kv_head: shape.n_kv_head,
                                        head_dim: shape.head_dim,
                                        cos: cos.clone(),
                                        sin: sin.clone(),
                                        use_rope_contiguous,
                                        attn_logit_softcap,
                                        rope_dim: shape.rope_dim,
                                        skip_rope: false,
                                        qk_rms_norm_after_rope: None,
                                        sliding_window: window,
                                        query_scale,
                                        v_rms_norm,
                                        output_scale,
                                    }))
                                })
                            })
                            .collect();
                        handles
                            .into_iter()
                            .map(|h| {
                                h.join().unwrap_or_else(|_| {
                                    Err(SwarmError::Internal("layer load thread panicked".into()))
                                })
                            })
                            .collect()
                    });
                for result in layer_results {
                    layers.push(result?);
                }
            } else {
                // Sequential fallback for ShardReader (gaps between tensors prevent read_to_end).
                // Same per-layer logic as the parallel path, using ct.tensor(file, ...) directly.
                for layer_idx in layer_start..layer_end {
                    // Per-layer placement, applied by shadowing so that every
                    // tensor load below picks up the right device with no
                    // per-call-site edit — see `hybrid::LayerPlacement`.
                    let device = placement.device_for(layer_idx);
                    let (cos, sin) = placement.rope_for(layer_idx);
                    let shape = shape_of(layer_idx);
                    let prefix = format!("blk.{layer_idx}");
                    let has_fused_qkv = ct
                        .tensor_infos
                        .contains_key(&format!("{prefix}.attn_qkv.weight"));
                    let (qkv_q, qkv_k, qkv_v) = if has_fused_qkv {
                        let fused_qt = ct
                            .tensor(file, &format!("{prefix}.attn_qkv.weight"), &device)
                            .map_err(|e| SwarmError::Internal(format!("{prefix}.attn_qkv: {e}")))?;
                        let q_dim = head_count * shape.head_dim;
                        let k_dim = shape.kv_width();
                        let fused = QMatMul::make_fused(fused_qt).map_err(SwarmError::internal)?;
                        (
                            QMatMul::from_fused_slice(fused.clone(), 0, q_dim),
                            QMatMul::from_fused_slice(fused.clone(), q_dim, k_dim),
                            QMatMul::from_fused_slice(fused, q_dim + k_dim, k_dim),
                        )
                    } else {
                        let wq = ct
                            .tensor(file, &format!("{prefix}.attn_q.weight"), &device)
                            .map_err(|e| SwarmError::Internal(format!("{prefix}.attn_q: {e}")))?;
                        let wk = ct
                            .tensor(file, &format!("{prefix}.attn_k.weight"), &device)
                            .map_err(|e| SwarmError::Internal(format!("{prefix}.attn_k: {e}")))?;
                        let wk = QMatMul::from_qtensor(wk).map_err(SwarmError::internal)?;
                        let wv = if value_is_key(&ct, gemma4, &prefix) {
                            wk.clone()
                        } else {
                            let wv = ct
                                .tensor(file, &format!("{prefix}.attn_v.weight"), &device)
                                .map_err(|e| {
                                    SwarmError::Internal(format!("{prefix}.attn_v: {e}"))
                                })?;
                            QMatMul::from_qtensor(wv).map_err(SwarmError::internal)?
                        };
                        (
                            QMatMul::from_qtensor(wq).map_err(SwarmError::internal)?,
                            wk,
                            wv,
                        )
                    };
                    let wo = ct
                        .tensor(file, &format!("{prefix}.attn_output.weight"), &device)
                        .map_err(|e| SwarmError::Internal(format!("{prefix}.attn_output: {e}")))?;
                    let attn_norm = ct
                        .tensor(file, &format!("{prefix}.attn_norm.weight"), &device)
                        .map_err(|e| SwarmError::Internal(format!("{prefix}.attn_norm: {e}")))?;
                    let ffn_norm_t = ct
                        .tensor(file, &format!("{prefix}.ffn_norm.weight"), &device)
                        .map_err(|e| SwarmError::Internal(format!("{prefix}.ffn_norm: {e}")))?;

                    let bq = ct
                        .tensor(file, &format!("{prefix}.attn_q.bias"), &device)
                        .ok()
                        .map(|t| t.dequantize(&device))
                        .transpose()
                        .map_err(|e| SwarmError::Internal(format!("bq: {e}")))?;
                    let bk = ct
                        .tensor(file, &format!("{prefix}.attn_k.bias"), &device)
                        .ok()
                        .map(|t| t.dequantize(&device))
                        .transpose()
                        .map_err(|e| SwarmError::Internal(format!("bk: {e}")))?;
                    let bv = ct
                        .tensor(file, &format!("{prefix}.attn_v.bias"), &device)
                        .ok()
                        .map(|t| t.dequantize(&device))
                        .transpose()
                        .map_err(|e| SwarmError::Internal(format!("bv: {e}")))?;

                    let post_attn_norm = ct
                        .tensor(
                            file,
                            &format!("{prefix}.post_attention_norm.weight"),
                            &device,
                        )
                        .ok();
                    let post_ffw_norm = ct
                        .tensor(file, &format!("{prefix}.post_ffw_norm.weight"), &device)
                        .ok();

                    // FFN: routed experts (Qwen2/Qwen3-MoE, Gemma 4), else dense.
                    let ffn =
                        load_standard_ffn(&ct, file, &prefix, &placement, layer_idx, standard_ffn)?;

                    let attn_q_norm = ct
                        .tensor(file, &format!("{prefix}.attn_q_norm.weight"), &device)
                        .ok();
                    let attn_k_norm = ct
                        .tensor(file, &format!("{prefix}.attn_k_norm.weight"), &device)
                        .ok();
                    let output_scale = layer_output_scale(&ct, file, &prefix)?;

                    layers.push(LayerVariant::Dense(LayerWeights {
                        attention_wq: qkv_q,
                        attention_wk: qkv_k,
                        attention_wv: qkv_v,
                        attention_wo: QMatMul::from_qtensor(wo).map_err(SwarmError::internal)?,
                        attention_bq: bq,
                        attention_bk: bk,
                        attention_bv: bv,
                        attention_norm: make_norm(attn_norm, rms_norm_eps)?,
                        attn_q_norm: attn_q_norm
                            .map(|t| make_norm(t, rms_norm_eps))
                            .transpose()?,
                        attn_k_norm: attn_k_norm
                            .map(|t| make_norm(t, rms_norm_eps))
                            .transpose()?,
                        ffn,
                        ffn_norm: make_norm(ffn_norm_t, rms_norm_eps)?,
                        post_attention_norm: post_attn_norm
                            .map(|t| make_norm(t, rms_norm_eps))
                            .transpose()?,
                        post_ffw_norm: post_ffw_norm
                            .map(|t| make_norm(t, rms_norm_eps))
                            .transpose()?,
                        n_head: head_count,
                        n_kv_head: shape.n_kv_head,
                        head_dim: shape.head_dim,
                        cos: cos.clone(),
                        sin: sin.clone(),
                        use_rope_contiguous,
                        attn_logit_softcap,
                        rope_dim: shape.rope_dim,
                        skip_rope: false,
                        qk_rms_norm_after_rope: None,
                        sliding_window: sliding_window
                            .as_ref()
                            .and_then(|w| w.for_layer(layer_idx)),
                        query_scale: query_scale_of(layer_idx),
                        v_rms_norm,
                        output_scale,
                    }));
                }
            }
        }

        // Extract tokenizer metadata via centralized GgufTokenizerMeta
        let tok_meta = super::gguf_meta::GgufTokenizerMeta::from_content(&ct);
        let vocabulary = if tok_meta.vocab.is_empty() {
            None
        } else {
            tracing::info!(vocab_size = tok_meta.vocab.len(), "Loaded GGUF vocabulary");
            Some(tok_meta.vocab.clone())
        };

        let tokenizer = tok_meta.build_tokenizer();
        if tokenizer.is_some() {
            tracing::info!(
                tokenizer_model = %tok_meta.tokenizer_model,
                pre_type = %tok_meta.pre_tokenizer,
                "Built tokenizer from GGUF metadata"
            );
        }

        // EOS tokens with architecture-specific fallbacks
        let eos_tokens = tok_meta.eos_tokens_with_arch_fallback(arch);
        tracing::info!(eos_tokens = ?eos_tokens, "Loaded EOS tokens from GGUF");

        // Chat template from GGUF metadata
        let chat_template = tok_meta.chat_template.filter(|s| !s.is_empty());

        // Resolve BOS/EOS token strings from their IDs + vocabulary
        let bos_id = ct
            .metadata
            .get("tokenizer.ggml.bos_token_id")
            .and_then(|v| v.to_u32().ok());
        let eos_str_id = ct
            .metadata
            .get("tokenizer.ggml.eos_token_id")
            .and_then(|v| v.to_u32().ok());
        let vocab_ref = vocabulary.as_deref().unwrap_or(&[]);
        let bos_token = bos_id
            .and_then(|id| vocab_ref.get(id as usize).cloned())
            .unwrap_or_default();
        let eos_token = eos_str_id
            .and_then(|id| vocab_ref.get(id as usize).cloned())
            .unwrap_or_default();

        if let Some(ref tmpl) = chat_template {
            tracing::info!(
                bos = %bos_token,
                eos = %eos_token,
                template_len = tmpl.len(),
                template_preview = &tmpl[..tmpl.len().min(200)],
                "Loaded chat template from GGUF header"
            );
        }

        // A card's prompt pass accumulates quantized products in f16 (llama.cpp's
        // default), except where an architecture's numbers need f32 — marked
        // here, once every layer is built, whatever branch built it (#147).
        // `SWARMLLM_F32_MARKS=0` leaves them unmarked, for the A/B.
        if model_arch.accumulates_outputs_in_f32()
            && std::env::var("SWARMLLM_F32_MARKS").as_deref() != Ok("0")
        {
            let (marked, missed) =
                crate::inference::layers::accumulate_output_projections_in_f32(&mut layers);
            tracing::info!(
                arch = %model_arch,
                marked,
                missed,
                "projections that keep an f32 accumulator on a card (llama.cpp's rule for this architecture)"
            );
        }

        // A processor-held layer multiplies a prompt on the card its model's
        // other layers use, each weight copied over for the call, as
        // llama.cpp's op offload does — its projections (#138 (1b)) and a
        // mixture-of-experts layer's chosen experts (#138 (1)). Given here,
        // once every layer is built, whatever branch built it. The A/Bs:
        // `SWARMLLM_PROMPT_CARD_PROJECTIONS=0`, `SWARMLLM_MOE_PROMPT_CARD=0`.
        // What each prompt token costs those layers on the card sizes their
        // chunks (#138 (1e), `SplitModel::prompt_reader`); 0 when none copies.
        let (mut prompt_bytes_per_token, mut prompt_stack_bytes) = (0, 0);
        if let Some(card) = placement.card_beside_the_processor(layers.len()) {
            let given = crate::inference::layers::let_processor_layers_read_prompts_on(
                &mut layers,
                card,
                std::env::var("SWARMLLM_PROMPT_CARD_PROJECTIONS").as_deref() != Ok("0"),
                std::env::var("SWARMLLM_MOE_PROMPT_CARD").as_deref() != Ok("0"),
            );
            prompt_bytes_per_token = crate::inference::layers::prompt_bytes_per_token(&mut layers);
            prompt_stack_bytes = crate::inference::layers::prompt_stack_bytes(&layers);
            tracing::info!(
                projection_layers = given.projection_layers,
                expert_layers = given.expert_layers,
                min_tokens = crate::inference::layers::PROMPT_ON_CARD_MIN_TOKENS,
                activation_kb_per_token = prompt_bytes_per_token >> 10,
                expert_stacks_mb = prompt_stack_bytes >> 20,
                "DIAG: processor-held layers read a prompt on the card"
            );
        }

        let has_biases = layers
            .first()
            .is_some_and(|l| matches!(l, LayerVariant::Dense(lw) if lw.attention_bq.is_some()));
        tracing::info!(
            arch = %model_arch,
            layers = format!("[{layer_start}..{layer_end})"),
            total = block_count,
            is_first,
            is_last,
            has_qkv_biases = has_biases,
            rope = if use_rope_contiguous { "contiguous" } else { "interleaved" },
            activation = ?activation,
            context_length,
            "Loaded split model segment"
        );

        Ok(Self {
            tok_embeddings,
            // Where each layer actually went. Empty when the segment is on one
            // device, which is what the forward loop checks before it does any
            // per-layer device work at all — an unsplit model pays nothing.
            layer_devices: if placement.is_hybrid(layers.len()) {
                placement.devices(layers.len())
            } else {
                Vec::new()
            },
            layers,
            norm,
            output,
            masks: None,
            kv_budget_bytes,
            kv_growth,
            host_kv_budget_bytes,
            host_kv_growth,
            layer_start,
            layer_end,
            total_layers: block_count,
            hidden_dim: embedding_length,
            arch: model_arch,
            device: {
                // Stage timers block on this device when
                // SWARMLLM_PROFILE_SYNC=1, so a GPU profile attributes time to
                // the op that spent it rather than to whichever later op
                // happened to block. No-op otherwise.
                crate::inference::prof::set_sync_device(&device);
                device
            },
            vocabulary,
            tokenizer,
            eos_tokens,
            chat_template,
            bos_token,
            eos_token,
            max_seq_len: context_length,
            kv_model_key: format!("{layer_start}-{layer_end}-{block_count}"),
            final_logit_softcap,
            suppressed_logits,
            batch_calls: 0,
            batch_fellback: 0,
            batch_stats_reported_at: None,
            decode_graph: Default::default(),
            prompt_room: super::model::PromptRoom {
                bytes_per_token: prompt_bytes_per_token,
                stack_bytes: prompt_stack_bytes,
                last: None,
            },
        })
    }
}

#[cfg(test)]
mod embedding_dtype_tests {
    use super::EMBEDDING_DTYPE;
    use crate::model::auto_manage::vram::EMBEDDING_TABLE_BYTES_PER_ELEMENT;

    /// The loader allocates the embedding table; the VRAM estimator decides
    /// whether that allocation is allowed to happen. They hold the same fact in
    /// two places, and nothing at runtime would notice them disagreeing — the
    /// node would simply refuse models that fit, or admit ones that OOM.
    ///
    /// Changing `EMBEDDING_DTYPE` without updating the estimator is exactly the
    /// mistake this catches.
    #[test]
    fn embedding_dtype_matches_the_vram_estimate() {
        assert_eq!(
            EMBEDDING_DTYPE.size_in_bytes() as u64,
            EMBEDDING_TABLE_BYTES_PER_ELEMENT,
            "the loader dequantizes token_embd to {:?} ({} bytes/element) but the VRAM \
             estimator charges {} bytes/element",
            EMBEDDING_DTYPE,
            EMBEDDING_DTYPE.size_in_bytes(),
            EMBEDDING_TABLE_BYTES_PER_ELEMENT
        );
    }
}

#[cfg(test)]
mod expert_stack_tests {
    use super::{load_moe_ffn, split_expert_stack, Activation, MoeRoutingConfig};
    use candle_core::quantized::{gguf_file, GgmlDType, QTensor};
    use candle_core::{Device, Tensor};

    /// A GGUF holding one layer's router and three `[n, rows, cols]` Q8_0
    /// expert stacks, written and parsed in memory.
    fn moe_gguf(
        n: usize,
        hidden: usize,
        inter: usize,
    ) -> (gguf_file::Content, std::io::Cursor<Vec<u8>>) {
        let dev = Device::Cpu;
        let q = |shape: (usize, usize, usize), seed: f32| {
            let v: Vec<f32> = (0..shape.0 * shape.1 * shape.2)
                .map(|i| ((i as f32 * seed).sin()) * 0.5)
                .collect();
            QTensor::quantize(&Tensor::from_vec(v, shape, &dev).unwrap(), GgmlDType::Q8_0).unwrap()
        };
        let router = QTensor::quantize(
            &Tensor::from_vec(
                (0..n * hidden)
                    .map(|i| (i as f32 * 0.37).cos())
                    .collect::<Vec<f32>>(),
                (n, hidden),
                &dev,
            )
            .unwrap(),
            GgmlDType::F32,
        )
        .unwrap();
        let gate = q((n, inter, hidden), 0.013);
        let up = q((n, inter, hidden), 0.029);
        let down = q((n, hidden, inter), 0.041);
        written(&[
            ("blk.0.ffn_gate_inp.weight", &router),
            ("blk.0.ffn_gate_exps.weight", &gate),
            ("blk.0.ffn_up_exps.weight", &up),
            ("blk.0.ffn_down_exps.weight", &down),
        ])
    }

    /// `tensors` as a GGUF, written and parsed in memory.
    fn written(tensors: &[(&str, &QTensor)]) -> (gguf_file::Content, std::io::Cursor<Vec<u8>>) {
        let mut buf = std::io::Cursor::new(Vec::new());
        gguf_file::write(&mut buf, &[], tensors).unwrap();
        buf.set_position(0);
        let ct = gguf_file::Content::read(&mut buf).unwrap();
        (ct, buf)
    }

    /// Gemma 4's FUSED gate/up stack (`ffn_gate_up_exps`) holds each expert's
    /// gate rows and then its up rows (llama.cpp's `build_moe_ffn` views the
    /// product at 0 and at `n_ff`). Cut into one matrix per expert, and into
    /// the two stacks the card kernel reads, each half holds exactly what a
    /// file with separate stacks holds; the down projection's per-expert scale
    /// is read beside them.
    #[test]
    fn a_fused_gate_up_stack_is_cut_into_the_gate_and_up_a_separate_file_holds() {
        let (n, hidden, inter) = (4, 64, 96);
        let dev = Device::Cpu;
        let values = |shape: (usize, usize, usize), seed: f32| {
            let v: Vec<f32> = (0..shape.0 * shape.1 * shape.2)
                .map(|i| ((i as f32 * seed).sin()) * 0.5)
                .collect();
            Tensor::from_vec(v, shape, &dev).unwrap()
        };
        let q8 = |t: &Tensor| QTensor::quantize(t, GgmlDType::Q8_0).unwrap();
        let f32s = |t: &Tensor| QTensor::quantize(t, GgmlDType::F32).unwrap();
        let (gate, up) = (
            values((n, inter, hidden), 0.013),
            values((n, inter, hidden), 0.029),
        );
        let fused = q8(&Tensor::cat(&[&gate, &up], 1).unwrap());
        let down = q8(&values((n, hidden, inter), 0.041));
        let router = f32s(&values((1, n, hidden), 0.37).squeeze(0).unwrap());
        let scales = vec![0.5f32, 1.25, 2.0, 0.75];
        let scale = f32s(&Tensor::new(scales.as_slice(), &dev).unwrap());
        let (ct, mut reader) = written(&[
            ("blk.0.ffn_gate_inp.weight", &router),
            ("blk.0.ffn_gate_up_exps.weight", &fused),
            ("blk.0.ffn_down_exps.weight", &down),
            ("blk.0.ffn_down_exps.scale", &scale),
        ]);
        let dq = |q: &QTensor| q.dequantize(&dev).unwrap();
        let same = |a: &Tensor, b: &Tensor| {
            (a - b)
                .unwrap()
                .abs()
                .unwrap()
                .max_all()
                .unwrap()
                .to_scalar::<f32>()
                .unwrap()
                == 0.0
        };
        let (want_gate, want_up) = (dq(&q8(&gate)), dq(&q8(&up)));
        let moe = load_moe_ffn(
            &ct,
            &mut reader,
            "blk.0",
            &placed_on(dev.clone(), 0),
            0,
            2,
            MoeRoutingConfig::default(),
            Activation::Gelu,
        )
        .unwrap();
        assert_eq!(moe.experts.len(), n);
        for (e, expert) in moe.experts.iter().enumerate() {
            let held =
                |w: &crate::inference::layers::QMatMul| dq(w.quantized_weight().unwrap().as_ref());
            assert!(
                same(&held(&expert.gate), &want_gate.get(e).unwrap()),
                "expert {e}'s gate"
            );
            assert!(
                same(&held(&expert.up), &want_up.get(e).unwrap()),
                "expert {e}'s up"
            );
        }
        assert_eq!(moe.activation, Activation::Gelu);
        assert_eq!(moe.expert_scales.map(|s| s.host), Some(scales));
        let [g, u] =
            super::unfuse_expert_stack(&ct, &mut reader, "blk.0.ffn_gate_up_exps.weight", &dev)
                .unwrap();
        assert_eq!(g.shape().dims(), &[n, inter, hidden]);
        assert_eq!(g.dtype(), GgmlDType::Q8_0);
        assert!(same(&dq(&g), &want_gate), "the stacked gate");
        assert!(same(&dq(&u), &want_up), "the stacked up");
    }

    /// Layer 0 wholly on `device`, its experts in the machine's memory when
    /// `experts_in_ram` is 1.
    fn placed_on(
        device: Device,
        experts_in_ram: usize,
    ) -> crate::inference::split::hybrid::LayerPlacement {
        let rope = Tensor::zeros((1, 1), candle_core::DType::F32, &device).unwrap();
        crate::inference::split::hybrid::LayerPlacement::uniform(0, device, rope.clone(), rope)
            .unwrap()
            .with_experts_on_the_processor(experts_in_ram)
    }

    /// #138 (3): a card-held layer told to keep its experts in RAM loads its
    /// router on the card and every expert on the processor, one quantized
    /// matrix each (never stacked — the card kernel cannot read RAM), in the
    /// form a prompt's experts can be copied to the card. Fed an activation on
    /// the card, it picks the experts the processor picks, answers a decode
    /// step as the layer held wholly on the processor does, and a prompt as
    /// that layer does when offered the card (#138 (1), the same kernel) —
    /// within float rounding: the card multiplies the router itself. The
    /// card's f16 prompt kernel is not the processor's answer (0.149 off a
    /// largest output of 5.56 here, both paths alike); it was judged against
    /// llama.cpp on a real model, not here.
    #[cfg(feature = "candle-cuda")]
    #[test]
    fn a_card_layer_with_its_experts_in_ram_answers_as_a_processor_layer_does() {
        use super::MoeFfn;
        // The card as a worker opens it (`load_device`, the one place).
        let card = super::load_device(false);
        if !card.is_cuda() {
            return;
        }
        let (n, hidden, inter) = (8, 64, 32);
        let (ct, mut reader) = moe_gguf(n, hidden, inter);
        let load = |placement, reader: &mut std::io::Cursor<Vec<u8>>| {
            load_moe_ffn(
                &ct,
                reader,
                "blk.0",
                &placement,
                0,
                2,
                MoeRoutingConfig::default(),
                Activation::SiLU,
            )
            .unwrap()
        };
        let processor = load(placed_on(Device::Cpu, 0), &mut reader);
        let mut moe = load(placed_on(card.clone(), 1), &mut reader);
        assert!(moe.gate.device().is_cuda(), "the router left the card");
        assert!(moe.stacked.is_none(), "experts in RAM were stacked");
        assert_eq!(moe.experts.len(), n);
        assert!(
            moe.experts_on_the_processor(),
            "the experts went to the card"
        );
        // What the loader's pass gives such a layer
        // (`let_processor_layers_read_prompts_on`): the card, for a prompt.
        assert!(moe.experts_could_stack_on_a_card());
        moe.prompt_card = Some(card.clone());
        let offered = MoeFfn {
            prompt_card: Some(card.clone()),
            ..processor.clone()
        };
        let largest_gap = |a: &Tensor, b: &Tensor| {
            (a.to_device(&Device::Cpu).unwrap() - b.to_device(&Device::Cpu).unwrap())
                .unwrap()
                .abs()
                .unwrap()
                .max_all()
                .unwrap()
                .to_scalar::<f32>()
                .unwrap()
        };
        for tokens in [1, 40] {
            let x = Tensor::from_vec(
                (0..tokens * hidden)
                    .map(|i| (i as f32 * 0.11).sin())
                    .collect::<Vec<f32>>(),
                (1, tokens, hidden),
                &Device::Cpu,
            )
            .unwrap();
            let flat = x.reshape((tokens, hidden)).unwrap();
            let picks = |routes: Vec<Vec<(usize, f32)>>| -> Vec<Vec<usize>> {
                routes
                    .into_iter()
                    .map(|r| r.into_iter().map(|p| p.0).collect())
                    .collect()
            };
            assert_eq!(
                picks(moe.route(&flat.to_device(&card).unwrap()).unwrap()),
                picks(processor.route(&flat).unwrap()),
                "{tokens} tokens: the card picked other experts"
            );
            let got = moe.forward(&x.to_device(&card).unwrap()).unwrap();
            assert!(
                got.device().is_cuda(),
                "the answer did not come back to the card"
            );
            // A decode step runs the processor's own experts; a prompt runs
            // the card's kernel, as a processor layer offered the card does.
            let want = if tokens < crate::inference::layers::PROMPT_ON_CARD_MIN_TOKENS {
                processor.forward(&x).unwrap()
            } else {
                offered.forward(&x).unwrap()
            };
            let gap = largest_gap(&got, &want);
            assert!(
                gap < 1e-4,
                "{tokens} tokens: {gap} off the processor layer's answer"
            );
        }
    }

    /// #292: the card's grouped kernels read the 32-block types a row too
    /// narrow for a K-quant is stored in — llama.cpp's quantizer puts a Q2_K
    /// file's 704-wide down experts in Q4_0, a Q4_K one's in Q5_0, a Q5_K one's
    /// in Q5_1 — so such a layer stays STACKED on the card. A decode step's
    /// one-row kernel is the arithmetic of candle's own card product (llama.cpp's
    /// MMVQ: the same `vec_dot`, the activation in q8_1), so the layer answers as
    /// it does holding one matrix per expert on the card, within float
    /// rounding (~1e-7 of its largest output). A prompt's tensor-core kernel
    /// multiplies in f16, and lands nearer the exact answer (the same experts
    /// in f32) than the processor's product, which rounds the activation to 8
    /// bits: 2.8-7.7e-4 of the largest output against 6.4-9.4e-3. A misread
    /// block (a nibble, a fifth bit, a scale) is off by the answer's own size —
    /// a Q5_0 fifth bit read from the wrong lane was 0.55 off, Q4_0 read as
    /// Q4_1 NaN. The weights are hashed, not smooth: smooth ones cancel in the
    /// sum, and every type was then ~5% off the processor, Q4_K's and Q2_K's
    /// (read since #138) as much as the new ones.
    #[cfg(feature = "candle-cuda")]
    #[test]
    fn a_card_layer_keeps_32_block_experts_stacked_and_reads_them_right() {
        use super::MoeFfn;
        use crate::inference::layers::{ExpertFfn, QMatMul, MOE_CARD_ONE_ROW_MAX_TOKENS};
        use GgmlDType as G;
        let card = super::load_device(false);
        if !card.is_cuda() {
            return;
        }
        // Gate and up 256 wide (a K-quant fits), down 96 (only a 32-block
        // type does), as a 26B-A4B's 2816 and 704.
        let (n, hidden, inter) = (8, 256, 96);
        // splitmix64 into [-0.5, 0.5).
        let hashed = |count: usize, salt: u64| -> Vec<f32> {
            (0..count as u64)
                .map(|i| {
                    let mut z = (i ^ (salt << 32)).wrapping_mul(0x9E37_79B9_7F4A_7C15);
                    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                    ((z ^ (z >> 31)) >> 40) as f32 / (1u64 << 24) as f32 - 0.5
                })
                .collect()
        };
        let file = |dtypes: [G; 3]| {
            let q = |shape: (usize, usize, usize), salt: u64, dtype: G| {
                let v = hashed(shape.0 * shape.1 * shape.2, salt);
                QTensor::quantize(&Tensor::from_vec(v, shape, &Device::Cpu).unwrap(), dtype)
                    .unwrap()
            };
            let router = Tensor::from_vec(hashed(n * hidden, 1), (n, hidden), &Device::Cpu);
            let router = QTensor::quantize(&router.unwrap(), G::F32).unwrap();
            written(&[
                ("blk.0.ffn_gate_inp.weight", &router),
                (
                    "blk.0.ffn_gate_exps.weight",
                    &q((n, inter, hidden), 2, dtypes[0]),
                ),
                (
                    "blk.0.ffn_up_exps.weight",
                    &q((n, inter, hidden), 3, dtypes[1]),
                ),
                (
                    "blk.0.ffn_down_exps.weight",
                    &q((n, hidden, inter), 4, dtypes[2]),
                ),
            ])
        };
        let load = |dtypes: [G; 3], device: &Device| -> MoeFfn {
            let (ct, mut reader) = file(dtypes);
            load_moe_ffn(
                &ct,
                &mut reader,
                "blk.0",
                &placed_on(device.clone(), 0),
                0,
                2,
                MoeRoutingConfig::default(),
                Activation::Gelu,
            )
            .unwrap()
        };
        // The same layer holding one quantized matrix per expert on the card.
        let per_expert = |dtypes: [G; 3], stacked: &MoeFfn| -> MoeFfn {
            let (ct, mut reader) = file(dtypes);
            let mut stack = |name: &str| {
                split_expert_stack(&ct, &mut reader, &format!("blk.0.{name}.weight"), &card)
                    .unwrap()
                    .into_iter()
                    .map(|q| QMatMul::from_qtensor(q).unwrap())
            };
            let (gate, up, down) = (
                stack("ffn_gate_exps"),
                stack("ffn_up_exps").collect::<Vec<_>>(),
                stack("ffn_down_exps").collect::<Vec<_>>(),
            );
            MoeFfn {
                experts: gate
                    .zip(up)
                    .zip(down)
                    .map(|((gate, up), down)| ExpertFfn { gate, up, down })
                    .collect(),
                stacked: None,
                ..stacked.clone()
            }
        };
        // The largest gap from `want`, over want's largest value.
        let relative_gap = |got: &Tensor, want: &Tensor| {
            let largest = |t: Tensor| {
                t.abs()
                    .unwrap()
                    .max_all()
                    .unwrap()
                    .to_scalar::<f32>()
                    .unwrap()
            };
            let want = want.to_device(&Device::Cpu).unwrap();
            let gap = largest((got.to_device(&Device::Cpu).unwrap() - &want).unwrap());
            gap / largest(want)
        };
        // (tokens, gap, ceiling): 1 and 5 tokens against one matrix per expert
        // on the card; 40 against the exact answer, no further off than the
        // processor's own product is.
        let gaps = |dtypes: [G; 3]| -> Vec<(usize, f32, f32)> {
            let processor = load(dtypes, &Device::Cpu);
            let on_card = load(dtypes, &card);
            assert!(
                on_card.stacked.is_some(),
                "{dtypes:?}: not stacked on the card"
            );
            let unstacked = per_expert(dtypes, &on_card);
            // The same layer with every expert dequantized to f32.
            let exact = MoeFfn {
                experts: processor
                    .experts
                    .iter()
                    .map(|e| {
                        let f = |w: &QMatMul| {
                            let d = w
                                .quantized_weight()
                                .unwrap()
                                .dequantize(&Device::Cpu)
                                .unwrap();
                            QMatMul::from_qtensor(QTensor::quantize(&d, G::F32).unwrap()).unwrap()
                        };
                        ExpertFfn {
                            gate: f(&e.gate),
                            up: f(&e.up),
                            down: f(&e.down),
                        }
                    })
                    .collect(),
                ..processor.clone()
            };
            [1usize, 5, 40]
                .into_iter()
                .map(|tokens| {
                    let x = Tensor::from_vec(
                        hashed(tokens * hidden, 5),
                        (1, tokens, hidden),
                        &Device::Cpu,
                    )
                    .unwrap();
                    let flat = x.reshape((tokens, hidden)).unwrap();
                    let picks = |routes: Vec<Vec<(usize, f32)>>| -> Vec<Vec<usize>> {
                        routes
                            .into_iter()
                            .map(|r| r.into_iter().map(|p| p.0).collect())
                            .collect()
                    };
                    assert_eq!(
                        picks(on_card.route(&flat.to_device(&card).unwrap()).unwrap()),
                        picks(processor.route(&flat).unwrap()),
                        "{dtypes:?}, {tokens} tokens: the card picked other experts"
                    );
                    let on = x.to_device(&card).unwrap();
                    let got = on_card.forward(&on).unwrap();
                    if tokens > MOE_CARD_ONE_ROW_MAX_TOKENS {
                        let truth = exact.forward(&x).unwrap();
                        let processor_gap = relative_gap(&processor.forward(&x).unwrap(), &truth);
                        (tokens, relative_gap(&got, &truth), processor_gap)
                    } else {
                        let want = unstacked.forward(&on).unwrap();
                        (tokens, relative_gap(&got, &want), 1e-4)
                    }
                })
                .collect()
        };
        let mut failures = Vec::new();
        for dtypes in [
            [G::Q8_0, G::Q8_0, G::Q4_0],
            [G::Q8_0, G::Q8_0, G::Q4_1],
            [G::Q8_0, G::Q8_0, G::Q5_0],
            [G::Q8_0, G::Q8_0, G::Q5_1],
            [G::Q4_1, G::Q4_0, G::Q5_1],
            [G::Q2K, G::Q2K, G::Q4_0],
            [G::Q4K, G::Q4K, G::Q5_0],
            [G::Q5K, G::Q5K, G::Q5_1],
        ] {
            for (tokens, gap, ceiling) in gaps(dtypes) {
                eprintln!("{dtypes:?} {tokens} tokens: {gap:.2e} off (ceiling {ceiling:.2e})");
                // A misread scale can be NaN, which no ceiling compares above.
                if gap.is_nan() || gap >= ceiling {
                    failures.push(format!(
                        "{dtypes:?}, {tokens} tokens: {gap} (ceiling {ceiling})"
                    ));
                }
            }
        }
        assert!(failures.is_empty(), "off the reference: {failures:#?}");
    }

    /// Each expert is its own slice of the stack, still QUANTIZED, and holds
    /// exactly that slice's values. The dtype check is the point: dequantizing
    /// the stacks made a 30B-A3B model need ~116 GB where its file is 11 GB.
    #[test]
    fn an_expert_stack_is_cut_into_quantized_experts_holding_their_own_slice() {
        let (n, hidden, inter) = (4, 64, 96);
        let (ct, mut reader) = moe_gguf(n, hidden, inter);
        let experts =
            split_expert_stack(&ct, &mut reader, "blk.0.ffn_gate_exps.weight", &Device::Cpu)
                .unwrap();
        assert_eq!(experts.len(), n);
        let whole = ct
            .tensor(&mut reader, "blk.0.ffn_gate_exps.weight", &Device::Cpu)
            .unwrap()
            .dequantize(&Device::Cpu)
            .unwrap();
        for (e, q) in experts.iter().enumerate() {
            assert_eq!(q.dtype(), GgmlDType::Q8_0, "expert {e} was dequantized");
            assert_eq!(q.shape().dims(), &[inter, hidden]);
            let got = q.dequantize(&Device::Cpu).unwrap();
            let want = whole.get(e).unwrap();
            let diff = (got - want)
                .unwrap()
                .abs()
                .unwrap()
                .max_all()
                .unwrap()
                .to_scalar::<f32>()
                .unwrap();
            assert_eq!(diff, 0.0, "expert {e} holds someone else's rows");
        }
    }

    /// The whole feed-forward: one expert per router row, every one quantized,
    /// and no shared expert invented where the file has none.
    #[test]
    fn a_moe_layer_loads_with_every_expert_quantized() {
        let (n, hidden, inter) = (3, 64, 32);
        let (ct, mut reader) = moe_gguf(n, hidden, inter);
        let moe = load_moe_ffn(
            &ct,
            &mut reader,
            "blk.0",
            &placed_on(Device::Cpu, 0),
            0,
            2,
            MoeRoutingConfig::default(),
            Activation::SiLU,
        )
        .unwrap();
        assert_eq!(moe.experts.len(), n);
        for e in &moe.experts {
            for m in [&e.gate, &e.up, &e.down] {
                assert!(
                    matches!(
                        &m.inner,
                        crate::inference::layers::QMatMulInner::Standard(
                            candle_core::quantized::QMatMul::QTensor(_)
                        )
                    ),
                    "an expert matrix was dequantized"
                );
            }
        }
        assert!(moe.shared_gate.is_none() && moe.shared_gate_inp.is_none());
        let x = Tensor::ones((1, 5, hidden), candle_core::DType::F32, &Device::Cpu).unwrap();
        assert_eq!(moe.forward(&x).unwrap().dims(), &[1, 5, hidden]);
    }
}

#[cfg(test)]
mod suppressed_token_tests {
    use super::{suppressed_logits_row, suppressed_token_ids};
    use candle_core::quantized::gguf_file::Value;
    use candle_core::Device;
    use std::collections::HashMap;

    /// The ids are read as llama.cpp writes them (an `i32` array; Gemma 4 12B:
    /// 258883 `<audio|>`, 258882 `<image|>`), and an id naming no logit — past
    /// the head, or negative — is dropped rather than indexing out of the row.
    #[test]
    fn a_files_suppressed_tokens_become_one_row_of_the_heads_width() {
        let key = "tokenizer.ggml.suppress_tokens".to_string();
        let mut md = HashMap::new();
        assert!(suppressed_token_ids(&md, 10).is_empty());
        md.insert(
            key,
            Value::Array(vec![
                Value::I32(7),
                Value::I32(2),
                Value::I32(-1),
                Value::I32(10),
                Value::U32(4),
            ]),
        );
        assert_eq!(suppressed_token_ids(&md, 10), vec![7, 2, 4]);

        let row = suppressed_logits_row(&md, 10, &Device::Cpu)
            .unwrap()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let low = super::super::executor::SUPPRESSED_LOGIT;
        assert_eq!(row, [0., 0., low, 0., low, 0., 0., low, 0., 0.]);
        assert!(suppressed_logits_row(&HashMap::new(), 10, &Device::Cpu)
            .unwrap()
            .is_none());
    }
}
