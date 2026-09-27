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

/// The per-dimension frequency divisors a GGUF carries as `rope_freqs.weight` —
/// how llama.cpp's converter bakes Llama 3.1/3.2's "llama3" RoPE scaling into the
/// file (llama.cpp PR #8676: `1 / ((1 - smooth) / factor + smooth)` per pair,
/// 1.0 for fast dimensions up to the scaling factor — 32 on Llama 3.2 — for slow
/// ones). llama.cpp divides each pair's angle by its factor (`ggml_rope_ext`'s
/// `freq_factors`), so the slow dimensions turn up to 32x slower than plain RoPE,
/// which is what the model was trained with (#124).
///
/// `Ok(None)`: the file has no such tensor (most models). `Err`: it has one this
/// reader cannot read — a node holding only later layers, since the tensor sits
/// in the prefix shard — and the caller must look elsewhere rather than silently
/// use plain RoPE.
pub(super) fn load_rope_freqs<R: std::io::Read + std::io::Seek>(
    ct: &gguf_file::Content,
    reader: &mut R,
    rope_dim: usize,
) -> Result<Option<Vec<f32>>, String> {
    if !ct.tensor_infos.contains_key(ROPE_FREQS_TENSOR) {
        return Ok(None);
    }
    let cpu = &Device::Cpu;
    let factors: Vec<f32> = ct
        .tensor(reader, ROPE_FREQS_TENSOR, cpu)
        .and_then(|t| t.dequantize(cpu))
        .and_then(|t| t.flatten_all())
        .and_then(|t| t.to_vec1())
        .map_err(|e| format!("{ROPE_FREQS_TENSOR}: {e}"))?;
    check_rope_freqs(&factors, rope_dim)?;
    Ok(Some(factors))
}

use super::gguf_meta::ROPE_FREQS_TENSOR;

/// A factor list the rotation can use: one per rotated pair, every one finite
/// and positive (it is a divisor).
pub(crate) fn check_rope_freqs(factors: &[f32], rope_dim: usize) -> Result<(), String> {
    if factors.len() != rope_dim / 2 {
        return Err(format!(
            "{ROPE_FREQS_TENSOR} has {} factors, expected {} (half the rotated dimensions)",
            factors.len(),
            rope_dim / 2
        ));
    }
    if factors.iter().any(|f| !f.is_finite() || *f <= 0.0) {
        return Err(format!(
            "{ROPE_FREQS_TENSOR} holds a factor that is not a positive number"
        ));
    }
    Ok(())
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Factors of 1.0 divide nothing: the scaled rotation must equal plain RoPE
    /// exactly, so a model whose factors are all 1 is unchanged by #124's fix.
    #[test]
    fn unit_factors_are_plain_rope() {
        let dev = Device::Cpu;
        let (c0, s0) = precompute_freqs_cis(64, 500_000.0, 300, &dev).unwrap();
        let (c1, s1) =
            precompute_freqs_cis_longrope(64, 500_000.0, 300, &[1.0; 32], 1.0, &dev).unwrap();
        let diff = |a: &Tensor, b: &Tensor| {
            (a - b)
                .unwrap()
                .abs()
                .unwrap()
                .flatten_all()
                .unwrap()
                .max(0)
                .unwrap()
                .to_scalar::<f32>()
                .unwrap()
        };
        assert_eq!(diff(&c0, &c1), 0.0);
        assert_eq!(diff(&s0, &s1), 0.0);
    }

    /// A factor of 32 turns its pair 32x slower — llama.cpp's
    /// `theta_base / freq_factor` — and leaves the other pairs alone.
    #[test]
    fn a_factor_divides_its_pairs_angle() {
        let dev = Device::Cpu;
        let mut factors = vec![1.0f32; 4];
        factors[3] = 32.0;
        let (c, s) = precompute_freqs_cis_longrope(8, 10_000.0, 100, &factors, 1.0, &dev).unwrap();
        let (cp, sp) = precompute_freqs_cis(8, 10_000.0, 100, &dev).unwrap();
        let at = |t: &Tensor, p: usize, i: usize| {
            t.get(p)
                .unwrap()
                .get(i)
                .unwrap()
                .to_scalar::<f32>()
                .unwrap()
        };
        let pos = 99;
        let theta_plain = 1.0f32 / 10_000f32.powf(6.0 / 8.0);
        let want = (pos as f32 * theta_plain / 32.0).cos();
        assert!(
            (at(&c, pos, 3) - want).abs() < 1e-5,
            "{} vs {want}",
            at(&c, pos, 3)
        );
        assert_eq!(
            at(&c, pos, 0),
            at(&cp, pos, 0),
            "an unscaled pair is untouched"
        );
        assert_eq!(at(&s, pos, 1), at(&sp, pos, 1));
    }

    #[test]
    fn factors_of_the_wrong_length_or_sign_are_refused() {
        assert!(check_rope_freqs(&[1.0; 32], 64).is_ok());
        assert!(check_rope_freqs(&[1.0; 31], 64).is_err());
        let mut bad = vec![1.0f32; 32];
        bad[5] = 0.0;
        assert!(check_rope_freqs(&bad, 64).is_err());
        bad[5] = f32::NAN;
        assert!(check_rope_freqs(&bad, 64).is_err());
    }
}
