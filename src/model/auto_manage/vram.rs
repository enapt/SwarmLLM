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
/// KV-head geometry. Measured against real steady-state loads on an RTX 3070 it
/// is **56% low on phi-3.5-mini-q4**, which is fine for the prune scoring it was
/// written for and
/// useless as an admission decision: a gate fed those numbers admits models
/// that cannot fit and the worker dies with `CUDA_ERROR_OUT_OF_MEMORY` anyway.
///
/// The dominant missing term is the **dequantized embedding table**. The loader
/// must hand `Embedding::new` a dense tensor, so a 128k-vocabulary model carries
/// `128256 * 2048 * 2 = 501 MB` that no file-size multiple can see — a large
/// fraction of the entire quantized checkpoint for a 1B model, and on Gemma 2's
/// 256k vocabulary larger than the checkpoint outright. It is resident at
/// `inference::split::loader::EMBEDDING_DTYPE`; the two MUST agree, and
/// `EMBEDDING_TABLE_BYTES_PER_ELEMENT` below is the copy of that fact used here.
/// The KV cache is the second, and is the one `inference.max_seq_len_override`
/// and the 4096 default bound (see `inference::split::kv_budget`).
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

/// Bytes a CUDA worker process costs beyond its tensors: driver context, cuBLAS
/// handles, activation and prefill-chunk buffers.
///
/// **Calibrated against measured steady state, 2026-07-30.** phi-3.5-mini-q4 on
/// an RTX 3070: weights 2282 + f32 embedding 376 + KV 3072 + RoPE 1.5 = 5731 MB
/// of tensors, against 6037 MiB measured once a request had completed — so ~306
/// MB of everything else. 320 MB reproduces the total to **+0.2%**, where the
/// file-size estimator is 56.5% low on the same model.
///
/// It is a fixed constant for a term that is partly fixed (driver context, ~120
/// MB) and partly scaled by hidden size and prefill chunk. On a small model that
/// over-charges by ~190 MB, which is the safe direction for admission.
///
/// **Do not calibrate against the worker's `vram_after_load_mb`.** That is
/// sampled when loading finishes, and candle allocates the KV cache lazily on
/// the *first append* — i.e. during the first forward, after that sample. Load
/// time and steady state differed by 3265 MB on phi-3.5; comparing the estimate
/// to the load figure makes it look ~2x high when it is not.
pub const CUDA_PROCESS_OVERHEAD_BYTES: u64 = 320 * 1024 * 1024;

/// Baseline resident set of a worker subprocess that is NOT using CUDA: the
/// binary, its runtime, the tokenizer and the IPC buffers. The CPU counterpart
/// of [`CUDA_PROCESS_OVERHEAD_BYTES`], and much smaller because there is no
/// device context to establish.
///
/// Errs high for the same reason the VRAM estimate does, but the failure it
/// guards against is worse: under-estimating here means swapping, which
/// degrades every request on the machine rather than just this model's.
pub const CPU_PROCESS_OVERHEAD_BYTES: u64 = 256 * 1024 * 1024;

/// A CPU worker establishes no device context, so it must never be charged
/// more than a CUDA one. Enforced at build time: inverting these would silently
/// make the CPU budget the stricter of the two, refusing loads that fit.
const _: () = assert!(CPU_PROCESS_OVERHEAD_BYTES < CUDA_PROCESS_OVERHEAD_BYTES);

/// Bytes per element of the resident token-embedding table.
///
/// **This must match what the loader actually allocates.** The loader
/// dequantizes `token_embd.weight` via
/// `SplitModel::EMBEDDING_DTYPE` — the two are checked against each
/// other by `embedding_dtype_matches_the_vram_estimate` in
/// `inference::split::loader`, because a disagreement here is invisible until
/// a node either refuses a model that fits or OOMs on one that does not.
///
/// f16 rather than f32 because the values come from a quantized tensor whose
/// own block scales are f16 — the wider type stores no additional information,
/// and on a large-vocabulary model it was costing more memory than the entire
/// rest of the model. Reported 2026-08-01: a 6 GB card refused Gemma 2 2B
/// (1629 MB of weights) at an estimated 5447 MB, of which 2250 MB was this
/// table at f32.
pub const EMBEDDING_TABLE_BYTES_PER_ELEMENT: u64 = 2;

/// The part of a worker's footprint that does not depend on where it runs:
/// weights, the dequantized embedding table, the KV cache and the RoPE tables.
/// A model's shape costs the same in system RAM as it does in VRAM; only the
/// per-process overhead differs, which is why the two public estimators below
/// are this plus a different constant.
fn estimate_model_resident_bytes(i: &VramFootprintInputs) -> u64 {
    const F32: u64 = 4;
    let mut bytes = i.quantized_weight_bytes;

    // Embedding table, dequantized by the loader at
    // `EMBEDDING_TABLE_BYTES_PER_ELEMENT`. First segment only.
    //
    // On a modern large-vocabulary model this is the single largest term —
    // larger than the quantized weights themselves. Gemma 2 2B carries a
    // 256,000-token vocabulary at hidden size 2304: 1125 MB at f16, against
    // 1629 MB for the entire rest of the model. Llama 3.1's 128,256-token
    // vocabulary at hidden size 4096 is the same order.
    if i.is_first {
        bytes = bytes.saturating_add(
            i.vocab_size
                .saturating_mul(i.embedding_length)
                .saturating_mul(EMBEDDING_TABLE_BYTES_PER_ELEMENT),
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

    bytes
}

/// Estimate a worker's GPU footprint in MB from the model's real geometry.
///
/// Deliberately errs HIGH: for an admission decision, over-estimating costs a
/// model that would have fitted, while under-estimating costs a hard OOM and —
/// until this release — a permanent fall back to the CPU.
pub fn estimate_worker_vram_mb(i: &VramFootprintInputs) -> u64 {
    estimate_model_resident_bytes(i).saturating_add(CUDA_PROCESS_OVERHEAD_BYTES) / (1024 * 1024)
}

/// Estimate a worker's system-RAM footprint in MB from the same geometry.
///
/// Identical to [`estimate_worker_vram_mb`] but for the per-process overhead —
/// a CPU worker establishes no device context. Used for CPU admission, which
/// exists because the GPU path's own fallback is "load it in system RAM
/// instead": the more often that fires, the more weight lands here.
pub fn estimate_worker_ram_mb(i: &VramFootprintInputs) -> u64 {
    estimate_model_resident_bytes(i).saturating_add(CPU_PROCESS_OVERHEAD_BYTES) / (1024 * 1024)
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

/// Percentage of *currently free* system RAM a configured budget may claim.
///
/// Mirrors `FREE_DISK_HEADROOM_PCT`: a configured ceiling is a ceiling, not a
/// promise the memory exists. Lower than the disk figure because the operating
/// system and every other process on the box need headroom to keep running,
/// and the failure mode when they do not get it is swapping — which degrades
/// the whole machine rather than just this daemon.
pub const FREE_RAM_HEADROOM_PCT: u64 = 70;

/// Effective system-RAM budget for CPU model loading, in MB.
///
/// Takes the configured budget (or the documented 50%-of-total default) and
/// clamps it to what is genuinely free right now. `None` means "do not judge":
/// either no budget could be derived, or the machine could not be read — and a
/// limit must never be invented from a failed measurement.
pub fn compute_ram_budget(shared: &crate::daemon::SharedState) -> Option<u64> {
    let mut sys = sysinfo::System::new();
    sys.refresh_memory();
    let total_mb = sys.total_memory() / (1024 * 1024);
    let available_mb = sys.available_memory() / (1024 * 1024);
    // "Has a GPU" here must mean "the GPU will actually run the models", not
    // merely that one is installed. A node with `inference.gpu_layers = 0` runs
    // everything on the CPU regardless of its hardware, so it is in exactly the
    // CPU-only situation the higher fraction exists for. Routed through the
    // canonical placement mapping so this cannot drift from where models are
    // really loaded.
    let has_gpu = shared.gpu_info.is_some()
        && !crate::daemon::shard_loader::force_cpu_for(shared.config.inference.gpu_layers);

    let configured = shared.config.resources.max_ram_mb;
    let by_config = shared
        .config
        .resources
        .inference_ram_budget_mb(total_mb, has_gpu)?;

    // Clamp ONLY an explicitly configured number. The automatic value is
    // already a fraction of this machine's own total memory, so clamping it
    // again against *free* memory discounts it twice — and would make the
    // ceiling depend on how much page cache happened to be warm at startup,
    // which is neither stable nor something a user can reason about. A
    // configured value is the only one that can exceed the machine.
    if configured == 0 || available_mb == 0 {
        return Some(by_config);
    }
    let by_machine = (available_mb / 100 * FREE_RAM_HEADROOM_PCT).max(total_mb / 4);
    if by_machine < by_config {
        tracing::warn!(
            configured_mb = by_config,
            total_mb,
            available_mb,
            allowed_mb = by_machine,
            "Configured memory budget is larger than this machine can spare — limiting it \
             so that loading a model cannot push the system into swap"
        );
    }
    Some(by_config.min(by_machine))
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

    /// phi-3.5-mini-q4: 32 layers, full MHA (32 KV heads), head_dim 96, 32k
    /// vocabulary, 131072 declared context capped to 4096.
    fn phi35() -> VramFootprintInputs {
        VramFootprintInputs {
            quantized_weight_bytes: 2_392_493_952,
            vocab_size: 32_064,
            embedding_length: 3072,
            segment_layers: 32,
            head_count_kv: 32,
            head_dim: 96,
            rope_dim: 96,
            effective_context: 4096,
            is_first: true,
        }
    }

    /// The calibration case, measured on an RTX 3070 once a request had
    /// completed (so the KV cache was actually allocated): **6037 MiB**.
    ///
    /// This is the test that justifies `CUDA_PROCESS_OVERHEAD_BYTES`. The
    /// file-size estimator is 56% low on the same model, which is what made it
    /// useless for admission.
    #[test]
    fn matches_measured_steady_state_on_phi35() {
        // 6037 MiB was measured while the embedding table was resident at f32.
        // The table is now f16, so the REAL steady state falls by exactly the
        // same amount the estimate does — the calibration still holds, but it
        // has to be compared against the adjusted figure or it would be
        // validating the estimate against a measurement of different code.
        const MEASURED_WITH_F32_EMBEDDING: u64 = 6037;
        let embedding_elems = phi35().vocab_size * phi35().embedding_length;
        let saved_mb = embedding_elems * (4 - EMBEDDING_TABLE_BYTES_PER_ELEMENT) / (1024 * 1024);
        let measured = MEASURED_WITH_F32_EMBEDDING - saved_mb;

        let new = estimate_worker_vram_mb(&phi35());
        let err_pct = 100.0 * (new as f64 - measured as f64) / measured as f64;
        assert!(
            err_pct.abs() < 5.0,
            "estimate {new} vs adjusted measurement {measured} is {err_pct:+.1}% — outside 5%"
        );
        const MEASURED: u64 = 6037;
        let old = estimate_model_vram_mb(phi35().quantized_weight_bytes);
        assert!(
            old < MEASURED * 2 / 3,
            "the file-size estimator should be far low: {old} vs {MEASURED}"
        );
    }

    /// A large vocabulary is where the file-size multiple fails worst, because
    /// the dequantized embedding table is invisible to it — on this model that
    /// table is larger than the whole checkpoint.
    #[test]
    fn beats_the_file_size_multiple_on_a_large_vocabulary_model() {
        let old = estimate_model_vram_mb(llama32_1b().quantized_weight_bytes);
        let new = estimate_worker_vram_mb(&llama32_1b());
        assert!(new > old, "new {new} must exceed old {old}");
        // The f16 embedding alone (128256 * 2048 * 2 = 501 MB) is still a large
        // fraction of the 821 MB checkpoint, and entirely invisible to a
        // file-size multiple.
        assert!(
            new - old > 500,
            "the embedding term alone should dominate: {old} -> {new}"
        );
    }

    /// The embedding term dominates on a large vocabulary and must only be
    /// charged to the segment that actually holds it.
    #[test]
    fn the_embedding_table_is_charged_to_the_first_segment_only() {
        let first = estimate_worker_vram_mb(&llama32_1b());
        let mut middle = llama32_1b();
        middle.is_first = false;
        let mid = estimate_worker_vram_mb(&middle);
        // 128256 * 2048 * 2 = 501 MB
        assert!(
            (first - mid) > 450 && (first - mid) < 550,
            "first {first} minus middle {mid} should be ~501 MB"
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

    /// gemma-2-2b-it-q4_k_m: 1629 MB of shards but a **256,000**-token
    /// vocabulary at hidden size 2304 — the case reported on 2026-08-01.
    fn gemma2_2b(effective_context: u64) -> VramFootprintInputs {
        VramFootprintInputs {
            quantized_weight_bytes: 1629 * 1024 * 1024,
            vocab_size: 256_000,
            embedding_length: 2304,
            segment_layers: 26,
            head_count_kv: 4,
            head_dim: 256,
            rope_dim: 256,
            effective_context,
            is_first: true,
        }
    }

    /// The reported failure: a 6 GB card (4600 MB budget) refused Gemma 2 2B
    /// with `estimated_mb=5447` and `committed_mb=0` — nothing else was using
    /// the GPU at all.
    ///
    /// The estimate was not wrong; the allocation genuinely was that large,
    /// and 2250 MB of it was one f32 embedding table — more than the 1629 MB
    /// the rest of the model weighs. At f16 the same model fits, which is the
    /// whole point: a modest laptop GPU can serve it again.
    #[test]
    fn gemma2_2b_fits_a_6gb_card_now_that_the_embedding_is_f16() {
        const BUDGET_MB: u64 = 4600;
        for ctx in [2048, 6144] {
            let est = estimate_worker_vram_mb(&gemma2_2b(ctx));
            assert!(
                est <= BUDGET_MB,
                "gemma-2-2b at ctx {ctx} estimates {est} MB, over the {BUDGET_MB} MB budget \
                 that made it fall back to the CPU"
            );
        }
    }

    /// Guards the arithmetic itself against a silent regression: the f32
    /// table was 2250 MB, which alone exceeded half the card.
    #[test]
    fn the_gemma2_embedding_table_is_no_longer_the_largest_single_term() {
        let i = gemma2_2b(6144);
        let embedding_mb =
            i.vocab_size * i.embedding_length * EMBEDDING_TABLE_BYTES_PER_ELEMENT / (1024 * 1024);
        assert_eq!(embedding_mb, 1125, "256000 x 2304 x 2 bytes");
        assert!(
            embedding_mb < i.quantized_weight_bytes / (1024 * 1024),
            "the embedding table must no longer outweigh the model's own weights"
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
        let _ = estimate_worker_ram_mb(&huge); // the CPU sibling likewise
    }

    /// The two estimators must agree about the model and differ only by the
    /// per-process overhead, which is the one genuinely device-specific term.
    /// If they ever disagree about shape, one of the two budgets is wrong.
    #[test]
    fn ram_and_vram_estimates_differ_only_by_process_overhead() {
        for i in [tinyllama(), llama32_1b(), phi35()] {
            let vram = estimate_worker_vram_mb(&i);
            let ram = estimate_worker_ram_mb(&i);
            let delta_mb =
                (CUDA_PROCESS_OVERHEAD_BYTES - CPU_PROCESS_OVERHEAD_BYTES) / (1024 * 1024);
            assert_eq!(
                vram - ram,
                delta_mb,
                "the two must differ only by the process overhead"
            );
        }
    }

    /// A CPU worker establishes no device context, so its baseline is lower —
    /// but still counted, because the process itself is not free.
    #[test]
    fn the_cpu_estimate_still_counts_a_process_baseline() {
        let est = estimate_worker_ram_mb(&tinyllama());
        let weights_only = tinyllama().quantized_weight_bytes / (1024 * 1024);
        assert!(
            est > weights_only,
            "estimate {est} MB must exceed the raw weights {weights_only} MB"
        );
    }
}
