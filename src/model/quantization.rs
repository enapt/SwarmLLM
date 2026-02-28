use crate::types::{ModelArchitecture, ModelManifest, Quantization};

// ---- GGUF Quantization Types ----

/// GGUF quantization types as numeric codes (matches the GGUF specification).
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum GgufQuantType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q2_K = 10,
    Q3_K_S = 11,
    Q3_K_M = 12,
    Q3_K_L = 13,
    Q4_K_S = 14,
    Q4_K_M = 15,
    Q5_K_S = 16,
    Q5_K_M = 17,
    Q6_K = 18,
}

impl GgufQuantType {
    /// Number of bytes per quantized block.
    pub fn bytes_per_block(&self) -> usize {
        match self {
            Self::F32 => 4 * 32, // 32 f32 values
            Self::F16 => 2 * 32, // 32 f16 values
            Self::Q4_0 => 18,    // 2 bytes scale + 16 bytes quants (32 values)
            Self::Q4_1 => 20,    // 2+2 bytes scale/min + 16 bytes quants
            Self::Q5_0 => 22,    // 2 bytes scale + 4 bytes high bits + 16 bytes quants
            Self::Q5_1 => 24,
            Self::Q8_0 => 34, // 2 bytes scale + 32 bytes quants
            Self::Q2_K => 84, // 256 values per block
            Self::Q3_K_S => 110,
            Self::Q3_K_M => 110,
            Self::Q3_K_L => 110,
            Self::Q4_K_S => 144, // 256 values per block
            Self::Q4_K_M => 144, // 256 values per block
            Self::Q5_K_S => 176,
            Self::Q5_K_M => 176,
            Self::Q6_K => 210, // 256 values per block
        }
    }

    /// Number of elements (values) per quantized block.
    pub fn block_size_elements(&self) -> usize {
        match self {
            Self::F32
            | Self::F16
            | Self::Q4_0
            | Self::Q4_1
            | Self::Q5_0
            | Self::Q5_1
            | Self::Q8_0 => 32,
            Self::Q2_K
            | Self::Q3_K_S
            | Self::Q3_K_M
            | Self::Q3_K_L
            | Self::Q4_K_S
            | Self::Q4_K_M
            | Self::Q5_K_S
            | Self::Q5_K_M
            | Self::Q6_K => 256,
        }
    }
}

/// Convert the high-level `Quantization` enum to the corresponding GGUF type code.
pub fn quantization_to_gguf(q: &Quantization) -> GgufQuantType {
    match q {
        Quantization::Q4KM => GgufQuantType::Q4_K_M,
        Quantization::Q5KM => GgufQuantType::Q5_K_M,
        Quantization::Q6K => GgufQuantType::Q6_K,
        Quantization::Q8_0 => GgufQuantType::Q8_0,
        Quantization::FP16 => GgufQuantType::F16,
    }
}

// ---- Dequantization ----

/// Dequantize a Q8_0 block (34 bytes) into 32 f32 values.
///
/// Q8_0 format: 1 f16 scale + 32 int8 quantized values.
pub fn dequantize_q8_0_block(block: &[u8; 34]) -> [f32; 32] {
    let scale = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
    let mut output = [0.0f32; 32];
    for i in 0..32 {
        output[i] = (block[2 + i] as i8) as f32 * scale;
    }
    output
}

/// Dequantize a Q4_K_M block (144 bytes) into 256 f32 values.
///
/// Q4_K_M uses super-blocks of 256 values with sub-block scales and minimums.
/// Format per super-block: 12 bytes scales/mins metadata + 4 bytes d/dmin + 128 bytes quants.
pub fn dequantize_q4km_block(block: &[u8; 144]) -> [f32; 256] {
    // Q4_K_M block layout:
    //   d:  f16 (2 bytes) — super-block scale
    //   dmin: f16 (2 bytes) — super-block minimum scale
    //   scales: [u8; 12] — packed sub-block scales and minimums
    //   qs: [u8; 128] — packed 4-bit quantized values (256 values, 2 per byte)
    let d = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
    let dmin = half::f16::from_le_bytes([block[2], block[3]]).to_f32();

    // Extract sub-block scales and minimums from the 12-byte packed representation
    let scales_raw = &block[4..16];
    let qs = &block[16..144];

    let mut sc = [0u8; 8];
    let mut m = [0u8; 8];

    // Unpack the 6-bit scales and minimums from 12 bytes
    for i in 0..4 {
        sc[i] = scales_raw[i] & 0x3F;
        m[i] = scales_raw[i + 4] & 0x3F;
    }
    for i in 4..8 {
        let idx = i - 4;
        sc[i] = ((scales_raw[idx] >> 6) & 0x03) | ((scales_raw[idx + 8] & 0x0F) << 2);
        m[i] = ((scales_raw[idx + 4] >> 6) & 0x03) | ((scales_raw[idx + 8] >> 4) << 2);
    }

    let mut output = [0.0f32; 256];

    for j in 0..8 {
        let scale = d * sc[j] as f32;
        let min = dmin * m[j] as f32;
        let base_idx = j * 32;
        let qs_offset = j * 16;

        for l in 0..16 {
            let byte = qs[qs_offset + l];
            let lo = (byte & 0x0F) as f32;
            let hi = ((byte >> 4) & 0x0F) as f32;
            output[base_idx + l] = lo * scale - min;
            output[base_idx + 16 + l] = hi * scale - min;
        }
    }

    output
}

// ---- FP16 Conversion Utilities ----

/// Convert a slice of f32 values to FP16 bytes (little-endian).
pub fn f32_to_f16_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|v| half::f16::from_f32(*v).to_le_bytes())
        .collect()
}

/// Convert FP16 bytes (little-endian) back to f32 values.
pub fn f16_bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|chunk| half::f16::from_le_bytes([chunk[0], chunk[1]]).to_f32())
        .collect()
}

// ---- MoE Sharding Strategy ----

/// Expert-aware shard boundaries for Mixture of Experts models.
///
/// For models like Mixtral 8x7B, each transformer layer has multiple expert
/// FFN blocks. Optimal sharding keeps all experts for a given layer together
/// on the same node to avoid cross-node expert routing.
pub struct MoeShardStrategy {
    pub num_experts: u32,
    pub experts_per_token: u32,
    pub total_layers: u32,
}

impl MoeShardStrategy {
    pub fn new(num_experts: u32, experts_per_token: u32, total_layers: u32) -> Self {
        Self {
            num_experts,
            experts_per_token,
            total_layers,
        }
    }

    /// Calculate expert-aligned shard boundaries.
    ///
    /// Returns layer ranges `(start, end)` where each range keeps all experts
    /// for the included layers together. Layer ranges are inclusive start,
    /// exclusive end.
    pub fn compute_shard_boundaries(&self, num_shards: u32) -> Vec<(u32, u32)> {
        if num_shards == 0 || self.total_layers == 0 {
            return vec![];
        }
        let num_shards = num_shards.min(self.total_layers);
        let layers_per_shard = self.total_layers / num_shards;
        let remainder = self.total_layers % num_shards;
        let mut boundaries = Vec::with_capacity(num_shards as usize);
        let mut start = 0u32;

        for i in 0..num_shards {
            let extra = if i < remainder { 1 } else { 0 };
            let end = start + layers_per_shard + extra;
            boundaries.push((start, end));
            start = end;
        }

        boundaries
    }

    /// Rank experts by activation frequency (descending).
    ///
    /// Given per-expert hit counts from the router, returns expert indices
    /// sorted from most to least frequently activated.
    pub fn rank_experts_by_frequency(&self, routing_counts: &[u64]) -> Vec<u32> {
        let mut indexed: Vec<(u32, u64)> = routing_counts
            .iter()
            .enumerate()
            .map(|(i, &c)| (i as u32, c))
            .collect();
        indexed.sort_by(|a, b| b.1.cmp(&a.1));
        indexed.into_iter().map(|(idx, _)| idx).collect()
    }

    /// Check if a layer range is expert-aligned (contains complete expert sets).
    ///
    /// For Mixtral-style models, each transformer block contains exactly one
    /// set of experts, so any integer layer range is aligned.
    pub fn is_expert_aligned(&self, layer_start: u32, layer_end: u32) -> bool {
        layer_end > layer_start
    }
}

/// Compute minimum VRAM (MB) needed to host one layer's expert FFN block
/// for an MoE model.
///
/// Separates attention (~30% of layer) from FFN (~70% of layer) for more
/// accurate estimates. In MoE models, attention is shared but FFN is split
/// across experts.
pub fn moe_min_vram_mb(manifest: &ModelManifest) -> u64 {
    let (num_experts, _experts_per_token) = match &manifest.architecture {
        ModelArchitecture::Mixtral {
            num_experts,
            experts_per_token,
        } => (*num_experts, *experts_per_token),
        ModelArchitecture::DeepSeek {
            num_experts,
            experts_per_token,
        } => (*num_experts, *experts_per_token),
        _ => return 0,
    };

    if manifest.num_layers == 0 || num_experts == 0 {
        return 0;
    }

    // Per-layer size from total model size
    let per_layer_bytes = manifest.total_size_bytes / manifest.num_layers as u64;

    // In transformer layers, attention is ~30% and FFN is ~70% of parameters.
    // For MoE: attention weights are shared, FFN weights are split across experts.
    let attention_bytes = per_layer_bytes * 30 / 100; // shared attention block
    let ffn_total_bytes = per_layer_bytes * 70 / 100; // total FFN across all experts
    let per_expert_ffn_bytes = ffn_total_bytes / num_experts as u64;

    // Minimum VRAM = attention (always needed) + one expert FFN block
    (attention_bytes + per_expert_ffn_bytes) / (1024 * 1024)
}

/// Estimate VRAM (MB) needed for a model, factoring in context length for KV cache.
///
/// KV cache size per layer = 2 * hidden_dim * context_len * sizeof(fp16)
/// For a rough estimate without exact hidden_dim, we derive it from model params:
///   hidden_dim ≈ sqrt(params_billions * 1e9 / (12 * num_layers))
/// Default context length is 4096 if not specified.
pub fn estimate_vram_with_context_mb(
    total_size_bytes: u64,
    num_layers: u32,
    num_params_billions: f64,
    context_length: Option<u32>,
) -> u64 {
    let weights_mb = total_size_bytes as f64 / (1024.0 * 1024.0);
    let ctx_len = context_length.unwrap_or(4096) as f64;

    if num_layers == 0 || num_params_billions <= 0.0 {
        // Fallback: flat 15% overhead estimate
        return (weights_mb * 1.15) as u64;
    }

    // Rough hidden_dim estimate from parameter count
    // params ≈ 12 * num_layers * hidden_dim^2 (standard transformer scaling)
    let hidden_dim = ((num_params_billions * 1e9) / (12.0 * num_layers as f64)).sqrt();

    // KV cache: 2 (K+V) * num_layers * hidden_dim * context_len * 2 (fp16 bytes)
    let kv_cache_bytes = 2.0 * num_layers as f64 * hidden_dim * ctx_len * 2.0;
    let kv_cache_mb = kv_cache_bytes / (1024.0 * 1024.0);

    // Total = weights + KV cache + ~5% overhead for activations/scratch
    (weights_mb + kv_cache_mb + weights_mb * 0.05) as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gguf_q8_0_block_size() {
        assert_eq!(GgufQuantType::Q8_0.bytes_per_block(), 34);
        assert_eq!(GgufQuantType::Q8_0.block_size_elements(), 32);
    }

    #[test]
    fn gguf_q4km_block_size() {
        assert_eq!(GgufQuantType::Q4_K_M.bytes_per_block(), 144);
        assert_eq!(GgufQuantType::Q4_K_M.block_size_elements(), 256);
    }

    #[test]
    fn quantization_to_gguf_mapping() {
        assert_eq!(
            quantization_to_gguf(&Quantization::Q4KM),
            GgufQuantType::Q4_K_M
        );
        assert_eq!(
            quantization_to_gguf(&Quantization::Q8_0),
            GgufQuantType::Q8_0
        );
        assert_eq!(
            quantization_to_gguf(&Quantization::FP16),
            GgufQuantType::F16
        );
    }

    #[test]
    fn dequantize_q8_0_basic() {
        // Block: scale = 1.0 (f16), then 32 int8 values = [1, 2, 3, ..., 32]
        let mut block = [0u8; 34];
        let scale_bytes = half::f16::from_f32(1.0).to_le_bytes();
        block[0] = scale_bytes[0];
        block[1] = scale_bytes[1];
        for i in 0..32 {
            block[2 + i] = (i as i8 + 1) as u8;
        }

        let output = dequantize_q8_0_block(&block);
        assert!((output[0] - 1.0).abs() < 0.01);
        assert!((output[1] - 2.0).abs() < 0.01);
        assert!((output[31] - 32.0).abs() < 0.01);
    }

    #[test]
    fn dequantize_q8_0_with_scale() {
        // Scale = 0.5, values = [10, 20]
        let mut block = [0u8; 34];
        let scale_bytes = half::f16::from_f32(0.5).to_le_bytes();
        block[0] = scale_bytes[0];
        block[1] = scale_bytes[1];
        block[2] = 10u8;
        block[3] = 20u8;

        let output = dequantize_q8_0_block(&block);
        assert!((output[0] - 5.0).abs() < 0.01);
        assert!((output[1] - 10.0).abs() < 0.01);
    }

    #[test]
    fn f32_f16_roundtrip() {
        let values = vec![1.0f32, -2.5, 0.0, 3.125, 100.0];
        let bytes = f32_to_f16_bytes(&values);
        assert_eq!(bytes.len(), values.len() * 2);

        let recovered = f16_bytes_to_f32(&bytes);
        assert_eq!(recovered.len(), values.len());

        // FP16 has limited precision, check within tolerance
        for (orig, rec) in values.iter().zip(recovered.iter()) {
            assert!(
                (orig - rec).abs() < 0.1,
                "f16 round-trip too lossy: {orig} -> {rec}"
            );
        }
    }

    #[test]
    fn f16_empty_input() {
        assert!(f32_to_f16_bytes(&[]).is_empty());
        assert!(f16_bytes_to_f32(&[]).is_empty());
    }

    #[test]
    fn moe_shard_boundaries_even_split() {
        let strategy = MoeShardStrategy::new(8, 2, 32);
        let boundaries = strategy.compute_shard_boundaries(4);
        assert_eq!(boundaries, vec![(0, 8), (8, 16), (16, 24), (24, 32)]);
    }

    #[test]
    fn moe_shard_boundaries_uneven() {
        let strategy = MoeShardStrategy::new(8, 2, 33);
        let boundaries = strategy.compute_shard_boundaries(4);
        // 33 / 4 = 8 remainder 1, so first shard gets an extra layer
        assert_eq!(boundaries, vec![(0, 9), (9, 17), (17, 25), (25, 33)]);
    }

    #[test]
    fn moe_shard_boundaries_more_shards_than_layers() {
        let strategy = MoeShardStrategy::new(8, 2, 4);
        let boundaries = strategy.compute_shard_boundaries(10);
        // Capped to 4 shards
        assert_eq!(boundaries.len(), 4);
    }

    #[test]
    fn rank_experts_by_frequency_ordering() {
        let strategy = MoeShardStrategy::new(4, 2, 32);
        let counts = vec![10, 100, 50, 5];
        let ranked = strategy.rank_experts_by_frequency(&counts);
        assert_eq!(ranked, vec![1, 2, 0, 3]);
    }

    #[test]
    fn expert_alignment_valid() {
        let strategy = MoeShardStrategy::new(8, 2, 32);
        assert!(strategy.is_expert_aligned(0, 8));
        assert!(strategy.is_expert_aligned(4, 12));
        assert!(!strategy.is_expert_aligned(5, 5)); // empty range
    }
}
