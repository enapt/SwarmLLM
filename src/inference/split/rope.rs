use candle_core::quantized::gguf_file;
use candle_core::{DType, Device, Result as CandleResult, Tensor};

pub(super) fn precompute_freqs_cis(
    head_dim: usize,
    freq_base: f32,
    max_seq_len: usize,
    device: &Device,
) -> CandleResult<(Tensor, Tensor)> {
    let theta: Vec<_> = (0..head_dim)
        .step_by(2)
        .map(|i| 1f32 / freq_base.powf(i as f32 / head_dim as f32))
        .collect();
    let theta = Tensor::new(theta.as_slice(), device)?;
    let idx_theta = Tensor::arange(0, max_seq_len as u32, device)?
        .to_dtype(DType::F32)?
        .reshape((max_seq_len, 1))?
        .matmul(&theta.reshape((1, theta.elem_count()))?)?;
    Ok((idx_theta.cos()?, idx_theta.sin()?))
}

/// Precompute RoPE frequencies for Long RoPE (SuRoPE) models like Phi-3.5.
/// Per-dimension frequency scaling factors from `rope_factors_long/short.weight` in GGUF.
pub(super) fn precompute_freqs_cis_longrope(
    head_dim: usize,
    freq_base: f32,
    max_seq_len: usize,
    rope_factors: &[f32],
    attn_factor: f32,
    device: &Device,
) -> CandleResult<(Tensor, Tensor)> {
    let half_dim = head_dim / 2;
    if rope_factors.len() != half_dim {
        return Err(candle_core::Error::Msg(format!(
            "LongRoPE factors length {} != expected half_dim {}",
            rope_factors.len(),
            half_dim
        )));
    }
    let theta: Vec<_> = (0..half_dim)
        .map(|i| 1f32 / (rope_factors[i] * freq_base.powf(2.0 * i as f32 / head_dim as f32)))
        .collect();
    let theta = Tensor::new(theta.as_slice(), device)?;
    let idx_theta = Tensor::arange(0, max_seq_len as u32, device)?
        .to_dtype(DType::F32)?
        .reshape((max_seq_len, 1))?
        .matmul(&theta.reshape((1, theta.elem_count()))?)?;
    let cos = (idx_theta.cos()? * attn_factor as f64)?;
    let sin = (idx_theta.sin()? * attn_factor as f64)?;
    Ok((cos, sin))
}

/// Load Long RoPE (SuRoPE) frequency scaling factors from GGUF tensors.
pub(super) fn load_longrope_factors<R: std::io::Read + std::io::Seek>(
    ct: &gguf_file::Content,
    reader: &mut R,
    arch: &str,
    context_length: usize,
) -> Option<(Vec<f32>, f32)> {
    let has_long = ct.tensor_infos.contains_key("rope_factors_long.weight");
    let has_short = ct.tensor_infos.contains_key("rope_factors_short.weight");
    if !has_long || !has_short {
        return None;
    }
    let original_ctx = ct
        .metadata
        .get(&format!("{arch}.rope.scaling.original_context_length"))
        .and_then(|v| v.to_u32().ok())
        .unwrap_or(4096) as usize;
    let tensor_name = if context_length > original_ctx {
        "rope_factors_long.weight"
    } else {
        "rope_factors_short.weight"
    };
    let cpu = &Device::Cpu;
    let factors_qt = ct.tensor(reader, tensor_name, cpu).ok()?;
    let factors_t = factors_qt.dequantize(cpu).ok()?;
    let factors: Vec<f32> = factors_t.flatten_all().ok()?.to_vec1().ok()?;
    let scale = context_length as f64 / original_ctx as f64;
    let attn_factor = if scale <= 1.0 {
        1.0f32
    } else {
        ct.metadata
            .get(&format!("{arch}.rope.scaling.attn_factor"))
            .and_then(|v| v.to_f32().ok())
            .unwrap_or_else(|| (1.0 + scale.ln() / (original_ctx as f64).ln()).sqrt() as f32)
    };
    tracing::info!(
        original_ctx,
        context_length,
        tensor = tensor_name,
        attn_factor,
        factors_len = factors.len(),
        "Loaded Long RoPE factors"
    );
    Some((factors, attn_factor))
}
