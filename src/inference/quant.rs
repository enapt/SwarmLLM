//! Q8_0 group-wise activation quantization for inter-node hidden state transfer.
//!
//! Layout matches llama.cpp Q8_0 precedent:
//!   For every group of [`GROUP_SIZE`] (=32) f32 values:
//!     [f16 scale (2 bytes)][i8 q[0..32] (32 bytes)] = 34 bytes
//!
//! `scale = max(|x|) / 127`, `q[i] = round(x[i] / scale)` clamped to i8.
//! Dequant: `x[i] = q[i] * scale`.
//!
//! Compression ratio: 32 * 4 = 128 bytes f32  →  34 bytes Q8_0  =  ~3.76× reduction.
//!
//! When `num_elements % 32 != 0`, the trailing partial group is still emitted
//! as a full 34-byte block; only the first `num_elements % 32` lanes carry
//! valid values. Trailing lanes are zero-quantized and ignored on decode.

use half::f16;

pub const GROUP_SIZE: usize = 32;
pub const BLOCK_BYTES: usize = 2 + GROUP_SIZE; // f16 scale + 32 i8 values

/// Quantize a slice of f32 values to Q8_0 wire bytes.
///
/// Returns `ceil(values.len() / 32) * 34` bytes.
pub fn quantize_q8_0(values: &[f32]) -> Vec<u8> {
    let n = values.len();
    let num_blocks = n.div_ceil(GROUP_SIZE);
    let mut out = Vec::with_capacity(num_blocks * BLOCK_BYTES);
    for blk in 0..num_blocks {
        let start = blk * GROUP_SIZE;
        let end = (start + GROUP_SIZE).min(n);
        let chunk = &values[start..end];

        // Per-block absolute max (skip non-finite values defensively)
        let mut amax = 0.0f32;
        for &v in chunk {
            if v.is_finite() {
                let a = v.abs();
                if a > amax {
                    amax = a;
                }
            }
        }
        let scale = if amax == 0.0 { 0.0 } else { amax / 127.0 };
        let inv_scale = if scale == 0.0 { 0.0 } else { 1.0 / scale };

        // Encode scale as f16
        out.extend_from_slice(&f16::from_f32(scale).to_le_bytes());

        // Quantize lanes; pad partial trailing group with zeros
        for lane in 0..GROUP_SIZE {
            let q = if lane < chunk.len() {
                let v = chunk[lane];
                let qf = (v * inv_scale).round();
                qf.clamp(-127.0, 127.0) as i8
            } else {
                0i8
            };
            out.push(q as u8);
        }
    }
    out
}

/// Dequantize Q8_0 bytes back to a `Vec<f32>` of length `num_elements`.
///
/// `bytes.len()` must equal `ceil(num_elements / 32) * 34`.
pub fn dequantize_q8_0(bytes: &[u8], num_elements: usize) -> Result<Vec<f32>, String> {
    let num_blocks = num_elements.div_ceil(GROUP_SIZE);
    let expected = num_blocks * BLOCK_BYTES;
    if bytes.len() != expected {
        return Err(format!(
            "Q8_0 byte length mismatch: got {}, expected {} ({} elements)",
            bytes.len(),
            expected,
            num_elements
        ));
    }
    let mut out = Vec::with_capacity(num_elements);
    let mut written = 0usize;
    for blk in 0..num_blocks {
        let off = blk * BLOCK_BYTES;
        let scale = f16::from_le_bytes([bytes[off], bytes[off + 1]]).to_f32();
        let qs = &bytes[off + 2..off + BLOCK_BYTES];
        let take = (num_elements - written).min(GROUP_SIZE);
        for &qb in &qs[..take] {
            let q = qb as i8;
            out.push(q as f32 * scale);
        }
        written += take;
    }
    Ok(out)
}

/// Number of bytes needed to encode `n` f32 values as Q8_0.
pub fn q8_0_byte_len(n: usize) -> usize {
    n.div_ceil(GROUP_SIZE) * BLOCK_BYTES
}

#[cfg(test)]
mod tests {
    use super::*;

    fn max_abs_err(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max)
    }

    #[test]
    fn exact_block_size_roundtrip() {
        let vals: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) * 0.1).collect();
        let bytes = quantize_q8_0(&vals);
        assert_eq!(bytes.len(), 34);
        let recovered = dequantize_q8_0(&bytes, 32).unwrap();
        assert_eq!(recovered.len(), 32);
        // Per-block scale = 1.6/127 ≈ 0.0126 → max quant error ≈ 0.0063
        let err = max_abs_err(&vals, &recovered);
        assert!(err < 0.01, "max err {err} too high");
    }

    #[test]
    fn partial_trailing_block() {
        // 50 elements → 2 blocks (one full, one with 18 valid lanes)
        let vals: Vec<f32> = (0..50).map(|i| (i as f32) * 0.05 - 1.0).collect();
        let bytes = quantize_q8_0(&vals);
        assert_eq!(bytes.len(), 2 * 34);
        let recovered = dequantize_q8_0(&bytes, 50).unwrap();
        assert_eq!(recovered.len(), 50);
        let err = max_abs_err(&vals, &recovered);
        assert!(err < 0.02, "max err {err} too high");
    }

    #[test]
    fn all_zeros_roundtrip() {
        let vals = vec![0.0f32; 64];
        let bytes = quantize_q8_0(&vals);
        let recovered = dequantize_q8_0(&bytes, 64).unwrap();
        assert!(recovered.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn outlier_block_isolated() {
        // Outlier in one block must not degrade the next block's precision.
        let mut vals = vec![0.01f32; 32]; // small values
        vals.extend(std::iter::repeat_n(100.0f32, 32)); // outlier-only block
        vals.extend(std::iter::repeat_n(0.01f32, 32)); // small values
        let bytes = quantize_q8_0(&vals);
        let recovered = dequantize_q8_0(&bytes, 96).unwrap();
        // First and third blocks should round-trip nearly exactly because
        // their local scale is tiny.
        let err_block_0 = max_abs_err(&vals[..32], &recovered[..32]);
        let err_block_2 = max_abs_err(&vals[64..], &recovered[64..]);
        assert!(
            err_block_0 < 1e-4,
            "block 0 contaminated by outliers: err={err_block_0}"
        );
        assert!(
            err_block_2 < 1e-4,
            "block 2 contaminated by outliers: err={err_block_2}"
        );
    }

    #[test]
    fn wrong_byte_length_rejected() {
        let bad = vec![0u8; 33];
        assert!(dequantize_q8_0(&bad, 32).is_err());
    }

    #[test]
    fn typical_hidden_state_quality() {
        // Simulate a typical post-LayerNorm hidden state slice (zero-mean unit-var).
        let n = 4096; // hidden_dim
        let vals: Vec<f32> = (0..n)
            .map(|i| {
                let phase = i as f32 * 0.07;
                phase.sin() + 0.3 * (phase * 3.1).cos()
            })
            .collect();
        let bytes = quantize_q8_0(&vals);
        assert_eq!(bytes.len(), q8_0_byte_len(n));
        let recovered = dequantize_q8_0(&bytes, n).unwrap();

        // Compression ratio
        let ratio = (n * 4) as f32 / bytes.len() as f32;
        assert!(
            ratio > 3.5,
            "Q8_0 should give >3.5× compression, got {ratio}"
        );

        // RMS error must be far below the activation magnitude (~1.0)
        let rms: f32 = (vals
            .iter()
            .zip(recovered.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            / n as f32)
            .sqrt();
        assert!(rms < 0.005, "RMS error {rms} above tolerance");
    }
}
