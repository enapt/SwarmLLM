use crate::daemon::SharedState;

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
    let output = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=memory.total", "--format=csv,noheader,nounits"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&output.stdout);
    text.trim().parse::<u64>().ok()
}

/// Query live GPU VRAM usage in MB via nvidia-smi.
///
/// Called on each auto-manage tick (~5 min) for accurate VRAM pressure.
/// Returns None if nvidia-smi is unavailable or fails.
pub(super) fn query_gpu_vram_used() -> Option<u64> {
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
