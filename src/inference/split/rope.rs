use candle_core::quantized::gguf_file;
use candle_core::{DType, Device, Result as CandleResult, Tensor};

pub(super) fn precompute_freqs_cis(
    head_dim: usize,
    freq_base: f32,
    max_seq_len: usize,
    device: &Device,
) -> CandleResult<(Tensor, Tensor)> {
    precompute_freqs_cis_scaled(head_dim, freq_base, 1.0, max_seq_len, device)
}

/// [`precompute_freqs_cis`] with every position multiplied by `freq_scale`
/// ([`LayerRope::scale`]) — llama.cpp's `freq_scale` with no YaRN ramp. A scale
/// of 1 is plain RoPE, bit for bit.
pub(super) fn precompute_freqs_cis_scaled(
    head_dim: usize,
    freq_base: f32,
    freq_scale: f32,
    max_seq_len: usize,
    device: &Device,
) -> CandleResult<(Tensor, Tensor)> {
    let theta: Vec<_> = (0..head_dim)
        .step_by(2)
        .map(|i| freq_scale / freq_base.powf(i as f32 / head_dim as f32))
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
use super::SlidingWindow;

/// How each layer of a model rotates, read as llama.cpp reads it — the ONE
/// reading the loader builds its RoPE tables from (FUTURE_WORK #115 (1)).
///
/// - **Linear scaling** (`llama-model.cpp`, the shared hparams):
///   `{arch}.rope.scaling.type` defaults to `linear`, and every position is
///   multiplied by `1 / rope.scaling.factor` (or the older key
///   `rope.scale_linear`). Gemma 3 4B-27B declare a factor of 8. Read without
///   it, their global layers turned 8x too fast at every position past the
///   first. YaRN scales by the factor too but ramps it per dimension
///   (`rope_yarn`); this build does not implement the ramp, so a file that
///   declares YaRN is served unscaled and says so at load, rather than
///   half-applied (FUTURE_WORK #280).
/// - **Windowed layers' own rotation** (`llama_model::get_rope_freq_base` /
///   `get_rope_freq_scale`, chosen by `is_swa(il)`): Gemma 3's local layers
///   turn at base 10,000 (`rope.freq_base_swa` when the file names one) and
///   are never scaled, while its global layers use the file's base — 1,000,000
///   on every size — and the factor (`llama_model_gemma3::load_arch_hparams`).
///   Gemma 2's follow the global ones unless the file names a base
///   (`llama_model_gemma2`). Which layers are windowed is read from the file
///   whatever `SWARMLLM_SLIDING_WINDOW` says, so that A/B arm changes the
///   window alone.
///
/// A/B arms: `SWARMLLM_ROPE_SCALING=0` reads no scaling,
/// `SWARMLLM_LAYER_ROPE=0` gives windowed layers the global tables.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct LayerRope {
    /// What every non-windowed layer multiplies its positions by.
    pub(crate) scale: f32,
    /// The windowed layers' rotation, where it differs from the rest.
    pub(crate) windowed: Option<WindowedRope>,
}

/// [`LayerRope::windowed`]: the layers, and the base and scale they turn at.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct WindowedRope {
    pub(crate) window: SlidingWindow,
    pub(crate) base: f32,
    pub(crate) scale: f32,
}

impl LayerRope {
    /// `global_base` is the file's `rope.freq_base`, as the loader already
    /// read it.
    pub(crate) fn read(ct: &gguf_file::Content, global_base: f32) -> Self {
        let arch = super::gguf_meta::gguf_arch_str(ct);
        let number = |key: &str| {
            ct.metadata
                .get(&format!("{arch}.{key}"))
                .and_then(|v| v.to_f32().ok())
        };
        let scale = if std::env::var("SWARMLLM_ROPE_SCALING").as_deref() == Ok("0") {
            1.0
        } else {
            let kind = ct
                .metadata
                .get(&format!("{arch}.rope.scaling.type"))
                .and_then(|v| v.to_string().ok())
                .map_or("linear", |s| s.as_str());
            let factor = number("rope.scaling.factor")
                .or_else(|| number("rope.scale_linear"))
                .filter(|f| f.is_finite() && *f > 0.0);
            match (kind, factor) {
                (_, None) | ("none", _) => 1.0,
                ("linear", Some(factor)) => 1.0 / factor,
                (kind, Some(factor)) => {
                    tracing::warn!(
                        scaling = kind,
                        factor,
                        "This model declares a RoPE scaling this build does not implement — \
                         serving it unscaled, which drifts from the model on long conversations"
                    );
                    1.0
                }
            }
        };
        let windowed = (std::env::var("SWARMLLM_LAYER_ROPE").as_deref() != Ok("0"))
            .then(|| SlidingWindow::from_file(ct))
            .flatten()
            .and_then(|window| {
                let named = number("rope.freq_base_swa");
                let (base, windowed_scale) = match arch.as_str() {
                    "gemma3" => (named.unwrap_or(10_000.0), 1.0),
                    _ => (named.unwrap_or(global_base), scale),
                };
                (base != global_base || windowed_scale != scale).then_some(WindowedRope {
                    window,
                    base,
                    scale: windowed_scale,
                })
            });
        Self { scale, windowed }
    }
}

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

    use crate::inference::split::gguf_meta::sliding_window_tests::header;
    use gguf_file::Value;

    /// A scale of 1 is plain RoPE bit for bit — every model that declares no
    /// scaling is unchanged — and a scale of 1/8 turns position 8p to where
    /// plain RoPE turns p.
    #[test]
    fn a_scale_multiplies_every_position() {
        let dev = Device::Cpu;
        let (c0, s0) = precompute_freqs_cis(16, 1_000_000.0, 64, &dev).unwrap();
        let (c1, s1) = precompute_freqs_cis_scaled(16, 1_000_000.0, 1.0, 64, &dev).unwrap();
        assert_eq!(c0.to_vec2::<f32>().unwrap(), c1.to_vec2::<f32>().unwrap());
        assert_eq!(s0.to_vec2::<f32>().unwrap(), s1.to_vec2::<f32>().unwrap());

        let (c8, s8) = precompute_freqs_cis_scaled(16, 1_000_000.0, 0.125, 64, &dev).unwrap();
        let (plain_c, plain_s) = (c0.to_vec2::<f32>().unwrap(), s0.to_vec2::<f32>().unwrap());
        let (scaled_c, scaled_s) = (c8.to_vec2::<f32>().unwrap(), s8.to_vec2::<f32>().unwrap());
        for p in 0..8 {
            for i in 0..8 {
                assert!(
                    (scaled_c[8 * p][i] - plain_c[p][i]).abs() < 1e-5,
                    "cos {p} {i}"
                );
                assert!(
                    (scaled_s[8 * p][i] - plain_s[p][i]).abs() < 1e-5,
                    "sin {p} {i}"
                );
            }
        }
    }

    fn gemma3(extra: Vec<(&str, Value)>) -> gguf_file::Content {
        let mut keys = vec![("gemma3.rope.freq_base", Value::F32(1_000_000.0))];
        keys.extend(extra);
        header("gemma3", keys)
    }

    /// The real 4B's keys (`sliding_window` 1024, linear, factor 8): its
    /// global layers turn at 1/8 speed, its windowed ones at base 10,000 and
    /// never scaled — `llama_model::get_rope_freq_base/scale` by `is_swa(il)`.
    #[test]
    fn gemma3_global_layers_scale_and_windowed_layers_turn_at_their_own_base() {
        let r = LayerRope::read(
            &gemma3(vec![
                ("gemma3.attention.sliding_window", Value::U32(1024)),
                ("gemma3.rope.scaling.type", Value::String("linear".into())),
                ("gemma3.rope.scaling.factor", Value::F32(8.0)),
            ]),
            1_000_000.0,
        );
        assert_eq!(r.scale, 0.125);
        let w = r.windowed.expect("the windowed layers turn differently");
        assert_eq!((w.base, w.scale), (10_000.0, 1.0));
        assert!(w.window.for_layer(4).is_some() && w.window.for_layer(5).is_none());

        // The 1B: no factor, and still a base of its own on windowed layers.
        let r = LayerRope::read(
            &gemma3(vec![("gemma3.attention.sliding_window", Value::U32(512))]),
            1_000_000.0,
        );
        assert_eq!(r.scale, 1.0);
        assert_eq!(r.windowed.map(|w| w.base), Some(10_000.0));

        // A base the file names wins.
        let r = LayerRope::read(
            &gemma3(vec![
                ("gemma3.attention.sliding_window", Value::U32(512)),
                ("gemma3.rope.freq_base_swa", Value::F32(20_000.0)),
            ]),
            1_000_000.0,
        );
        assert_eq!(r.windowed.map(|w| w.base), Some(20_000.0));

        // No window named: no windowed layers, every layer the global rotation.
        let r = LayerRope::read(
            &gemma3(vec![("gemma3.rope.scaling.factor", Value::F32(8.0))]),
            1_000_000.0,
        );
        assert_eq!((r.scale, r.windowed), (0.125, None));
    }

    /// Gemma 2's windowed layers rotate as the rest (`llama_model_gemma2`
    /// copies the global base and scale) unless the file names a base, so the
    /// model it has always run builds no second table.
    #[test]
    fn gemma2_windowed_layers_rotate_as_the_rest_unless_the_file_names_a_base() {
        let r = LayerRope::read(&header("gemma2", vec![]), 10_000.0);
        assert_eq!((r.scale, r.windowed), (1.0, None));
        let r = LayerRope::read(
            &header(
                "gemma2",
                vec![("gemma2.rope.freq_base_swa", Value::F32(5_000.0))],
            ),
            10_000.0,
        );
        assert_eq!(r.windowed.map(|w| (w.base, w.scale)), Some((5_000.0, 1.0)));
    }

    /// llama.cpp's reading for any family: no type means linear, the older
    /// `rope.scale_linear` key counts, `none` scales nothing — and YaRN, whose
    /// per-dimension ramp is not implemented, is served unscaled rather than
    /// half-applied (FUTURE_WORK #280).
    #[test]
    fn only_linear_scaling_is_applied() {
        let scale =
            |keys: Vec<(&str, Value)>| LayerRope::read(&header("llama", keys), 10_000.0).scale;
        assert_eq!(scale(vec![]), 1.0);
        assert_eq!(
            scale(vec![("llama.rope.scaling.factor", Value::F32(4.0))]),
            0.25
        );
        assert_eq!(
            scale(vec![("llama.rope.scale_linear", Value::F32(2.0))]),
            0.5
        );
        let typed = |kind: &str| {
            scale(vec![
                ("llama.rope.scaling.type", Value::String(kind.into())),
                ("llama.rope.scaling.factor", Value::F32(4.0)),
            ])
        };
        assert_eq!(typed("linear"), 0.25);
        assert_eq!(typed("none"), 1.0);
        assert_eq!(typed("yarn"), 1.0);
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
