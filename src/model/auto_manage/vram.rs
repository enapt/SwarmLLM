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

    // All known peers
    for peer in shared.peer_registry.iter() {
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
