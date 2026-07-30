use crate::daemon::SharedState;
use crate::types::ModelArchitecture;

/// Estimate VRAM required to run a model based on size and quantization.
///
/// Rule of thumb for quantized GGUF models:
/// - The model weights need ~model_size_bytes of VRAM (already quantized)
/// - KV cache overhead adds ~10-20% on top depending on context length
/// - We use 1.15x multiplier as a conservative estimate
pub fn estimate_model_vram_mb(total_size_bytes: u64) -> u64 {
    // Quantized model weights are already compressed; VRAM ~= file size + ~15% overhead
    (total_size_bytes as f64 * 1.15 / (1024.0 * 1024.0)) as u64
}

/// The pieces a worker's GPU footprint is actually made of.
///
/// [`estimate_model_vram_mb`] is `file_size * 1.15`, a flat 15% for "KV cache
/// overhead" with no term for vocabulary, context length, layer count or
/// KV-head geometry. Measured against real loads on an RTX 3070 it is
/// **56%-117% low**, which is fine for the prune scoring it was written for and
/// useless as an admission decision: a gate fed those numbers admits models
/// that cannot fit and the worker dies with `CUDA_ERROR_OUT_OF_MEMORY` anyway.
///
/// The dominant missing term is the **embedding table, dequantized to f32**.
/// The loader does `tok_embd.dequantize(&device)` because `Embedding::new` takes
/// a dense tensor, so a 128k-vocabulary model carries
/// `128256 * 2048 * 4 = 1002 MB` that no file-size multiple can see — larger
/// than the entire quantized checkpoint for a 1B model. The KV cache is the
/// second, and is the one `inference.max_seq_len_override` and the 4096 default
/// bound (see `inference::split::kv_budget`).
#[derive(Debug, Clone, Copy)]
pub struct VramFootprintInputs {
    /// Sum of the shard bytes this worker will map. Quantized, on-disk size.
    pub quantized_weight_bytes: u64,
    /// Rows in the embedding table — `{arch}.vocab_size`, or the token count.
    pub vocab_size: u64,
    /// `{arch}.embedding_length` (hidden dim).
    pub embedding_length: u64,
    /// Layers in THIS segment, not the whole model.
    pub segment_layers: u64,
    /// `{arch}.attention.head_count_kv`.
    pub head_count_kv: u64,
    /// `{arch}.attention.key_length`, or `embedding_length / head_count`.
    pub head_dim: u64,
    /// `{arch}.rope.dimension_count`.
    pub rope_dim: u64,
    /// The context length the KV cache will actually be sized to — i.e. AFTER
    /// the `DEFAULT_MAX_SEQ_LEN` cap and any override, not the raw GGUF value.
    pub effective_context: u64,
    /// Does this segment hold the embedding table (layer 0)?
    pub is_first: bool,
}

/// Bytes a CUDA worker process costs before any model weights: driver context,
/// cuBLAS handles and workspace.
///
/// Provisional. Two clean measurements implied 170 MB and 651 MB, which no
/// constant fits, so this is NOT yet calibrated — WSL reports `[N/A]` for
/// per-process GPU memory, so the deltas had to come from whole-device sampling
/// with another daemon resident. The worker now reports its own measured
/// footprint (`vram_measured_mb`); calibrate from that and prefer the measured
/// value over this estimate wherever one exists.
pub const CUDA_PROCESS_OVERHEAD_BYTES: u64 = 320 * 1024 * 1024;

/// Estimate a worker's GPU footprint in MB from the model's real geometry.
///
/// Deliberately errs HIGH: for an admission decision, over-estimating costs a
/// model that would have fitted, while under-estimating costs a hard OOM and —
/// until this release — a permanent fall back to the CPU.
pub fn estimate_worker_vram_mb(i: &VramFootprintInputs) -> u64 {
    const F32: u64 = 4;
    let mut bytes = i.quantized_weight_bytes;

    // Embedding table, dequantized to f32 by the loader. First segment only.
    if i.is_first {
        bytes = bytes.saturating_add(
            i.vocab_size
                .saturating_mul(i.embedding_length)
                .saturating_mul(F32),
        );
    }

    // KV cache: candle allocates the whole [B, H, ctx, D] buffer per layer on
    // the first append, for K and V, as f32.
    bytes = bytes.saturating_add(
        i.segment_layers
            .saturating_mul(2)
            .saturating_mul(i.head_count_kv)
            .saturating_mul(i.head_dim)
            .saturating_mul(i.effective_context)
            .saturating_mul(F32),
    );

    // RoPE cos/sin tables, sized to the same context.
    bytes = bytes.saturating_add(
        i.effective_context
            .saturating_mul(i.rope_dim.max(2) / 2)
            .saturating_mul(F32)
            .saturating_mul(2),
    );

    bytes = bytes.saturating_add(CUDA_PROCESS_OVERHEAD_BYTES);
    bytes / (1024 * 1024)
}

/// MoE-aware VRAM estimation. For Mixture-of-Experts models, only a fraction
/// of expert weights are active at any time, so the actual VRAM requirement
/// is much lower than the total file size.
///
/// Formula:
///   active_fraction = 0.40 (attention/embeddings, always loaded)
///                   + 0.60 * (experts_per_token / num_experts)
///   effective_vram = total_size * active_fraction * 1.15 (KV overhead)
pub fn estimate_model_vram_mb_arch(total_size_bytes: u64, arch: &ModelArchitecture) -> u64 {
    let (num_experts, experts_per_token) = match arch {
        ModelArchitecture::Mixtral {
            num_experts,
            experts_per_token,
        }
        | ModelArchitecture::DeepSeek {
            num_experts,
            experts_per_token,
        }
        | ModelArchitecture::Llama4 {
            num_experts,
            experts_per_token,
        }
        | ModelArchitecture::Qwen35Moe {
            num_experts,
            experts_per_token,
        } => (*num_experts, *experts_per_token),
        // Dense architectures — all weights active
        _ => return estimate_model_vram_mb(total_size_bytes),
    };

    if num_experts == 0 || experts_per_token >= num_experts {
        return estimate_model_vram_mb(total_size_bytes);
    }

    let active_fraction = 0.40 + 0.60 * (experts_per_token as f64 / num_experts as f64);
    (total_size_bytes as f64 * active_fraction * 1.15 / (1024.0 * 1024.0)) as u64
}

/// Lookup table for GPU memory bandwidth in GB/s.
/// Used for bandwidth-based speed estimation (tokens/s ≈ bandwidth / model_size * efficiency).
pub fn gpu_memory_bandwidth_gbps(name: &str) -> f32 {
    let name_upper = name.to_uppercase();
    // Match most specific first
    match () {
        // NVIDIA data center
        _ if name_upper.contains("H100") => 3352.0,
        _ if name_upper.contains("H200") => 4800.0,
        _ if name_upper.contains("A100") && name_upper.contains("80") => 2039.0,
        _ if name_upper.contains("A100") => 1555.0,
        _ if name_upper.contains("A6000") => 768.0,
        _ if name_upper.contains("A5000") => 768.0,
        _ if name_upper.contains("A4000") => 448.0,
        _ if name_upper.contains("V100") => 900.0,
        _ if name_upper.contains("L40S") => 864.0,
        _ if name_upper.contains("L40") => 864.0,
        _ if name_upper.contains("L4") => 300.0,
        _ if name_upper.contains("T4") => 300.0,
        // NVIDIA consumer - RTX 40 series
        _ if name_upper.contains("4090") => 1008.0,
        _ if name_upper.contains("4080") && name_upper.contains("SUPER") => 736.0,
        _ if name_upper.contains("4080") => 717.0,
        _ if name_upper.contains("4070") && name_upper.contains("SUPER") => 504.0,
        _ if name_upper.contains("4070") => 504.0,
        _ if name_upper.contains("4060") && name_upper.contains("TI") => 288.0,
        _ if name_upper.contains("4060") => 272.0,
        // NVIDIA consumer - RTX 30 series
        _ if name_upper.contains("3090") && name_upper.contains("TI") => 1008.0,
        _ if name_upper.contains("3090") => 936.0,
        _ if name_upper.contains("3080") && name_upper.contains("TI") => 912.0,
        _ if name_upper.contains("3080") => 760.0,
        _ if name_upper.contains("3070") && name_upper.contains("TI") => 608.0,
        _ if name_upper.contains("3070") => 448.0,
        _ if name_upper.contains("3060") && name_upper.contains("TI") => 448.0,
        _ if name_upper.contains("3060") => 360.0,
        // NVIDIA consumer - RTX 20 series
        _ if name_upper.contains("2080") && name_upper.contains("TI") => 616.0,
        _ if name_upper.contains("2080") => 448.0,
        _ if name_upper.contains("2070") => 448.0,
        _ if name_upper.contains("2060") => 336.0,
        // Apple Silicon
        _ if name_upper.contains("M4 MAX") => 546.0,
        _ if name_upper.contains("M4 PRO") => 273.0,
        _ if name_upper.contains("M4") => 120.0,
        _ if name_upper.contains("M3 MAX") => 400.0,
        _ if name_upper.contains("M3 PRO") => 150.0,
        _ if name_upper.contains("M3") => 100.0,
        _ if name_upper.contains("M2 MAX") => 400.0,
        _ if name_upper.contains("M2 PRO") => 200.0,
        _ if name_upper.contains("M2") => 100.0,
        _ if name_upper.contains("M1 MAX") => 400.0,
        _ if name_upper.contains("M1 PRO") => 200.0,
        _ if name_upper.contains("M1") => 68.0,
        // AMD
        _ if name_upper.contains("7900 XTX") => 960.0,
        _ if name_upper.contains("7900 XT") => 800.0,
        _ if name_upper.contains("MI300") => 5300.0,
        _ if name_upper.contains("MI250") => 3277.0,
        // Default: conservative estimate for unknown GPU
        _ => 300.0,
    }
}

/// Estimate tokens/s for a 7B model based on GPU memory bandwidth.
///
/// Formula: tokens/s = memory_bandwidth_GB_s / model_size_GB * efficiency
/// - GPU efficiency: 0.30 (accounts for compute overhead, KV cache, etc.)
/// - CPU efficiency: 0.15
///
/// For a 7B Q4_K_M model (~4.4GB), RTX 3070 (448 GB/s):
///   448 / 4.4 * 0.30 ≈ 30.5 tokens/s
pub fn estimate_tokens_per_sec_7b(bandwidth_gbps: f32, is_gpu: bool) -> f32 {
    const MODEL_SIZE_7B_Q4: f32 = 4.4; // ~4.4 GB for 7B Q4_K_M
    let efficiency = if is_gpu { 0.30 } else { 0.15 };
    bandwidth_gbps / MODEL_SIZE_7B_Q4 * efficiency
}

/// Compute the optimal shard window for a model given VRAM budget.
///
/// Returns `(start, end)` inclusive shard indices that fit within the budget.
/// Always prefers shard 0 (embeddings) and the last shard (output head) for
/// "boomerang" local inference. Fills remaining budget with contiguous middle shards.
///
/// Returns None if even 2 shards don't fit.
pub fn compute_optimal_shard_window(
    shard_count: u32,
    shard_vram_each_mb: u64,
    vram_budget_mb: u64,
) -> Option<Vec<u32>> {
    if shard_count == 0 || shard_vram_each_mb == 0 {
        return None;
    }

    let shards_that_fit = (vram_budget_mb / shard_vram_each_mb) as u32;
    if shards_that_fit == 0 {
        return None;
    }

    // If all shards fit, load everything
    if shards_that_fit >= shard_count {
        return Some((0..shard_count).collect());
    }

    let last_shard = shard_count - 1;
    let mut selected = Vec::new();

    if shard_count == 1 {
        return Some(vec![0]);
    }

    // Always include shard 0 (embeddings) and last shard (output head)
    selected.push(0);
    if shards_that_fit >= 2 {
        selected.push(last_shard);
    }

    // Fill remaining budget with contiguous middle shards starting from shard 1
    for i in 1..last_shard {
        if selected.len() as u32 >= shards_that_fit {
            break;
        }
        selected.push(i);
    }

    selected.sort();
    Some(selected)
}

/// Compute the total VRAM available across the entire network (all peers + local node).
pub fn global_pool_vram_mb(shared: &SharedState) -> u64 {
    let mut total = 0u64;

    // Local GPU -- use gpu_info if available, fallback to nvidia-smi
    total += local_vram_mb(shared);

    // Private mode: only count allowed nodes' VRAM
    let allowed_set = crate::pool::scope::allowed_node_set(shared);

    for peer in shared.peer_registry.iter() {
        // In private mode, skip peers outside the allowed set
        if let Some(ref allowed) = allowed_set {
            if !allowed.contains(peer.key()) {
                continue;
            }
        }
        if let Some(ref cap) = peer.capability {
            if let Some(ref gpu) = cap.gpu {
                total += gpu.vram_total_mb;
            }
        }
    }

    total
}

/// Get local VRAM in MB, with nvidia-smi fallback when gpu_info is None.
pub fn local_vram_mb(shared: &SharedState) -> u64 {
    if let Some(ref gpu) = shared.gpu_info {
        return gpu.vram_total_mb;
    }
    // Fallback: detect via nvidia-smi
    detect_vram_nvidia_smi().unwrap_or(0)
}

/// Fallback GPU VRAM detection via nvidia-smi.
fn detect_vram_nvidia_smi() -> Option<u64> {
    detect_gpu_nvidia_smi().1
}

/// Detect GPU name and total VRAM via nvidia-smi.
pub(crate) fn detect_gpu_nvidia_smi() -> (Option<String>, Option<u64>) {
    let output = std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,memory.total",
            "--format=csv,noheader,nounits",
        ])
        .output();

    match output {
        Ok(out) if out.status.success() => {
            let text = String::from_utf8_lossy(&out.stdout);
            let line = text.trim();
            if let Some((name, vram_str)) = line.split_once(',') {
                let name = name.trim().to_string();
                let vram_mb = vram_str.trim().parse::<u64>().ok();
                (Some(name), vram_mb)
            } else {
                (None, None)
            }
        }
        _ => (None, None),
    }
}

/// Query live GPU VRAM usage in MB via nvidia-smi.
///
/// Called on each auto-manage tick (~5 min) for accurate VRAM pressure.
/// Returns None if nvidia-smi is unavailable or fails.
pub(crate) fn query_gpu_vram_used() -> Option<u64> {
    let output = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=memory.used", "--format=csv,noheader,nounits"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&output.stdout);
    text.trim().parse::<u64>().ok()
}

/// Query live free GPU VRAM in MB via a single `nvidia-smi` call.
///
/// `memory.total - memory.used`, so it accounts for everything already
/// resident — other processes, and any model this daemon already loaded.
/// One combined query rather than [`detect_gpu_nvidia_smi`] +
/// [`query_gpu_vram_used`] because the two would be sampled at different
/// instants, and the subtraction of two racing samples can go negative.
///
/// Called once per model load by the split loader to size the KV cache
/// (`inference::split::kv_budget`). Returns None when nvidia-smi is
/// unavailable, which the caller must treat as "unknown", never as "zero".
pub(crate) fn query_gpu_vram_free_mb() -> Option<u64> {
    let output = std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=memory.total,memory.used",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&output.stdout);
    let line = text.trim().lines().next()?;
    let (total, used) = line.split_once(',')?;
    let total = total.trim().parse::<u64>().ok()?;
    let used = used.trim().parse::<u64>().ok()?;
    Some(total.saturating_sub(used))
}

/// Estimate VRAM for a segment (layer range) by scaling the full-model estimate
/// by the fraction of layers covered.
pub(super) fn estimate_segment_vram_mb(
    manifest: &crate::types::ModelManifest,
    layer_start: usize,
    layer_end: usize,
) -> u64 {
    let total_layers = manifest.num_layers as usize;
    if total_layers == 0 {
        return estimate_model_vram_mb(manifest.total_size_bytes);
    }
    let fraction = (layer_end - layer_start) as f64 / total_layers as f64;
    let full_vram = estimate_model_vram_mb(manifest.total_size_bytes);
    (full_vram as f64 * fraction).ceil() as u64
}

/// Compute the VRAM budget from SharedState for passing to `check_and_load_model`.
pub fn compute_vram_budget(shared: &crate::daemon::SharedState) -> Option<u64> {
    let gpu_total = shared
        .gpu_info
        .as_ref()
        .map(|g| g.vram_total_mb)
        .unwrap_or(0);
    shared.config.resources.inference_vram_budget_mb(gpu_total)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn moe_vram_lower_than_dense() {
        // Mixtral 8x7B ≈ 47GB on disk
        let total_bytes = 47u64 * 1024 * 1024 * 1024;
        let dense_vram = estimate_model_vram_mb(total_bytes);
        let moe_vram = estimate_model_vram_mb_arch(
            total_bytes,
            &ModelArchitecture::Mixtral {
                num_experts: 8,
                experts_per_token: 2,
            },
        );
        // MoE should be significantly less than dense
        assert!(
            moe_vram < dense_vram,
            "MoE={moe_vram} should be < dense={dense_vram}"
        );
        // Mixtral 8x7B with 2/8 experts: active_fraction = 0.40 + 0.60*0.25 = 0.55
        // So MoE VRAM should be ~55% of dense
        // active_fraction = 0.40 + 0.60*0.25 = 0.55, so ~55% of dense ≈ 30GB
        assert!(
            moe_vram < 31 * 1024,
            "MoE 8x7B should fit in <31GB, got {moe_vram}MB"
        );
    }

    #[test]
    fn dense_arch_unchanged() {
        let total_bytes = 4u64 * 1024 * 1024 * 1024;
        let dense = estimate_model_vram_mb(total_bytes);
        let llama = estimate_model_vram_mb_arch(total_bytes, &ModelArchitecture::Llama);
        assert_eq!(dense, llama);
    }

    #[test]
    fn gpu_bandwidth_known_gpus() {
        assert_eq!(gpu_memory_bandwidth_gbps("NVIDIA RTX 4090"), 1008.0);
        assert_eq!(gpu_memory_bandwidth_gbps("NVIDIA GeForce RTX 3070"), 448.0);
        assert_eq!(gpu_memory_bandwidth_gbps("NVIDIA A100 80GB"), 2039.0);
        assert_eq!(gpu_memory_bandwidth_gbps("Apple M3 Max"), 400.0);
    }

    #[test]
    fn gpu_bandwidth_unknown_returns_default() {
        assert_eq!(gpu_memory_bandwidth_gbps("Unknown GPU XYZ"), 300.0);
    }

    #[test]
    fn speed_estimation_sanity() {
        // RTX 3070: 448 GB/s, 7B Q4 ≈ 4.4GB
        let tps = estimate_tokens_per_sec_7b(448.0, true);
        assert!(tps > 20.0 && tps < 50.0, "Expected ~30 t/s, got {tps}");
    }

    #[test]
    fn shard_window_all_fit() {
        let result = compute_optimal_shard_window(4, 500, 3000);
        assert_eq!(result, Some(vec![0, 1, 2, 3]));
    }

    #[test]
    fn shard_window_partial_prefers_first_last() {
        // 8 shards, 500MB each, budget for 3
        let result = compute_optimal_shard_window(8, 500, 1500).unwrap();
        assert!(result.contains(&0), "Must include shard 0");
        assert!(result.contains(&7), "Must include last shard");
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn shard_window_too_small() {
        let result = compute_optimal_shard_window(8, 500, 100);
        assert_eq!(result, None);
    }

    #[test]
    fn shard_window_single_shard() {
        let result = compute_optimal_shard_window(1, 500, 1000);
        assert_eq!(result, Some(vec![0]));
    }
}

#[cfg(test)]
mod footprint_tests {
    use super::*;

    /// tinyllama-1.1b-q4 as it actually sits on disk: 636 MB of shards, 32k
    /// vocabulary, 22 layers, 4 KV heads, head_dim 64, 2048 context.
    fn tinyllama() -> VramFootprintInputs {
        VramFootprintInputs {
            quantized_weight_bytes: 667_078_656,
            vocab_size: 32_000,
            embedding_length: 2048,
            segment_layers: 22,
            head_count_kv: 4,
            head_dim: 64,
            rope_dim: 64,
            effective_context: 2048,
            is_first: true,
        }
    }

    /// llama-3.2-1b-q8: only 821 MB of weights but a 128k vocabulary, so the
    /// f32 embedding table alone is larger than the checkpoint.
    fn llama32_1b() -> VramFootprintInputs {
        VramFootprintInputs {
            quantized_weight_bytes: 860_807_296,
            vocab_size: 128_256,
            embedding_length: 2048,
            segment_layers: 16,
            head_count_kv: 8,
            head_dim: 64,
            rope_dim: 64,
            effective_context: 4096,
            is_first: true,
        }
    }

    /// The whole point: the old estimator is wildly low on a large-vocabulary
    /// model, because a file-size multiple cannot see a dequantized embedding
    /// table. Measured on an RTX 3070: 1145 MiB and 2731 MiB respectively.
    #[test]
    fn beats_the_file_size_multiple_on_a_large_vocabulary_model() {
        let old = estimate_model_vram_mb(llama32_1b().quantized_weight_bytes);
        let new = estimate_worker_vram_mb(&llama32_1b());
        const MEASURED: u64 = 2731;
        let old_err = (MEASURED as i64 - old as i64).abs();
        let new_err = (MEASURED as i64 - new as i64).abs();
        assert!(
            new_err < old_err,
            "new estimate {new} must be closer to {MEASURED} than old {old}"
        );
        // The old one is not merely worse, it is under by more than half.
        assert!(old < MEASURED / 2, "old estimate {old} vs {MEASURED}");
    }

    /// The embedding term dominates on a large vocabulary and must only be
    /// charged to the segment that actually holds it.
    #[test]
    fn the_embedding_table_is_charged_to_the_first_segment_only() {
        let first = estimate_worker_vram_mb(&llama32_1b());
        let mut middle = llama32_1b();
        middle.is_first = false;
        let mid = estimate_worker_vram_mb(&middle);
        // 128256 * 2048 * 4 = 1002 MB
        assert!(
            (first - mid) > 900 && (first - mid) < 1100,
            "first {first} minus middle {mid} should be ~1002 MB"
        );
    }

    /// Context length drives the KV term, which is what the 4096 default caps.
    /// A 128k context must estimate far higher than the capped one.
    #[test]
    fn context_length_moves_the_estimate() {
        let capped = estimate_worker_vram_mb(&llama32_1b());
        let mut uncapped = llama32_1b();
        uncapped.effective_context = 131_072;
        let full = estimate_worker_vram_mb(&uncapped);
        assert!(
            full > capped + 7_000,
            "131k context should add many GB over 4096: {capped} -> {full}"
        );
    }

    /// A short-context, small-vocabulary model should not be wildly
    /// over-estimated either — over-estimating costs admissions.
    #[test]
    fn a_small_model_is_not_grossly_over_estimated() {
        const MEASURED: u64 = 1145;
        let est = estimate_worker_vram_mb(&tinyllama());
        assert!(
            est >= MEASURED,
            "must not under-estimate ({est} < {MEASURED})"
        );
        assert!(
            est < MEASURED * 2,
            "must not be more than 2x over ({est} vs {MEASURED})"
        );
    }

    /// Degenerate geometry must not panic or overflow.
    #[test]
    fn degenerate_inputs_are_safe() {
        let z = VramFootprintInputs {
            quantized_weight_bytes: 0,
            vocab_size: 0,
            embedding_length: 0,
            segment_layers: 0,
            head_count_kv: 0,
            head_dim: 0,
            rope_dim: 0,
            effective_context: 0,
            is_first: true,
        };
        // Just the process overhead.
        assert_eq!(
            estimate_worker_vram_mb(&z),
            CUDA_PROCESS_OVERHEAD_BYTES / (1024 * 1024)
        );
        let huge = VramFootprintInputs {
            quantized_weight_bytes: u64::MAX,
            vocab_size: u64::MAX,
            embedding_length: u64::MAX,
            segment_layers: u64::MAX,
            head_count_kv: u64::MAX,
            head_dim: u64::MAX,
            rope_dim: u64::MAX,
            effective_context: u64::MAX,
            is_first: true,
        };
        let _ = estimate_worker_vram_mb(&huge); // must not panic
    }
}
