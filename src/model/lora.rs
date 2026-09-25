//! LoRA (Low-Rank Adaptation) adapter support.
//!
//! Enables per-request adapter selection for fine-tuned model behavior without
//! modifying base weights. Adapters are loaded from safetensors files containing
//! paired A/B matrices. The LoRA operation is:
//!
//!   output = base_weight @ x + (B @ A @ x) * scale
//!
//! where `scale = alpha / rank`.
//!
//! Adapters are registered via the admin API and cached in memory. The forward pass
//! applies the requested adapter by adding the low-rank delta to each matching layer.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use candle_core::{DType, Device, Tensor};
use serde::{Deserialize, Serialize};

use crate::error::SwarmError;

/// A loaded LoRA adapter with per-layer A/B matrices.
#[derive(Debug)]
pub struct LoraAdapter {
    pub metadata: AdapterMetadata,
    /// Per-layer LoRA weights, keyed by the base weight name they modify.
    /// e.g., "blk.0.attn_q" → LoraLayerWeights { a, b }
    pub weights: HashMap<String, LoraLayerWeights>,
}

/// Metadata for a registered adapter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdapterMetadata {
    /// Unique adapter identifier (user-provided or auto-generated).
    pub id: String,
    /// Human-readable name.
    pub name: String,
    /// The base model this adapter was trained for.
    pub base_model: String,
    /// LoRA rank (dimension of the low-rank matrices).
    pub rank: usize,
    /// LoRA alpha (scaling factor). Scale = alpha / rank.
    pub alpha: f32,
    /// Path to the safetensors file on disk.
    pub path: PathBuf,
    /// Number of layer weight pairs loaded.
    pub num_layers: usize,
    /// File size in bytes.
    pub size_bytes: u64,
    /// BLAKE3 hash (hex) of the safetensors file at registration time.
    /// Verified on subsequent loads to detect tampering / file swap.
    #[serde(default)]
    pub blake3: String,
    /// One past the highest layer index the adapter touches.
    #[serde(default)]
    pub layer_count: usize,
    /// `(in_dim, out_dim)` per projection kind (`attn_q`, `ffn_down`, …), read
    /// off the A/B matrices — what [`check_fits`] compares against a model's
    /// geometry, so an adapter made for another model is refused up front
    /// rather than failing inside a matmul.
    #[serde(default)]
    pub projections: std::collections::BTreeMap<String, (usize, usize)>,
}

/// Is `id` usable as a single, plain file name under the adapter directory?
///
/// The one answer for the admin API (registration), the chat API (a request
/// naming an adapter) and the worker (loading it): `..`, separators, NUL and
/// hidden names are refused everywhere, so no id can reach outside
/// `<data_dir>/adapters/` (gotcha #94).
pub fn is_safe_adapter_id(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= 128
        && !id.contains('/')
        && !id.contains('\\')
        && !id.contains('\0')
        && id != "."
        && id != ".."
        && !id.starts_with('.')
        && Path::new(id).components().count() == 1
}

/// Where a registration is recorded: `<adapter_dir>/<id>.adapter.json`.
///
/// This small file is what makes an adapter usable: it survives a restart
/// (the registry re-reads every one at startup) and it is how a model WORKER —
/// a separate process with no registry — finds the file, rank and alpha an id
/// stands for (`load_registered_adapter`). Until 2026-09-25 registration lived
/// only in the daemon's memory and the worker looked for a directory layout
/// nothing created, so no request ever had its adapter applied
/// (`docs/FUTURE_WORK.md` #110).
pub fn registration_path(adapter_dir: &Path, id: &str) -> PathBuf {
    adapter_dir.join(format!("{id}.adapter.json"))
}

/// What [`check_fits`] needs to know about a model — all of it in the GGUF
/// header, so the router can answer before any worker loads anything.
#[derive(Debug, Clone, Copy)]
pub struct AdapterTarget<'a> {
    /// GGUF `general.architecture`.
    pub architecture: &'a str,
    pub block_count: usize,
    pub embedding_length: usize,
    /// `head_count × head_dim`.
    pub q_width: usize,
    /// `head_count_kv × head_dim`.
    pub kv_width: usize,
    /// Mixture-of-experts feed-forward layers.
    pub moe: bool,
}

/// Can this adapter be applied, all of it, to this model? `Err` names why not,
/// for a caller to show.
///
/// Refused, rather than half-applied: a model whose layers the adapter code
/// does not reach (DeepSeek-2's MLA, Qwen 3.5's hybrid layers), and
/// feed-forward changes on a mixture-of-experts model — the split executor
/// applies an adapter to `LayerVariant::Dense` layers and `FfnVariant::Dense`
/// feed-forwards only, and anything else answering from the base model is the
/// silent failure #110 was. Then the geometry a GGUF header can answer: every
/// layer index below `block_count`, every projection's embedding-side width,
/// and the attention projections' head-side widths. The FFN width is not in
/// the header, so `ffn_gate`/`ffn_up` outputs and `ffn_down`'s input are not
/// checked — an adapter that passes the rest was made for this architecture
/// and size.
pub fn check_fits(meta: &AdapterMetadata, model: &AdapterTarget<'_>) -> Result<(), String> {
    use crate::inference::model_arch::ModelArch;
    if matches!(
        ModelArch::from_gguf_arch(model.architecture),
        ModelArch::DeepSeek2 | ModelArch::Qwen35 | ModelArch::Qwen35Moe | ModelArch::Unknown(_)
    ) {
        return Err(format!(
            "adapters are not supported for {} models yet",
            model.architecture
        ));
    }
    if model.moe && meta.projections.keys().any(|p| p.starts_with("ffn_")) {
        return Err(
            "it changes the feed-forward layers, and this model's are mixture-of-experts, \
             which adapters are not applied to"
                .into(),
        );
    }
    if meta.layer_count > model.block_count {
        return Err(format!(
            "it changes {} layers and this model has {} — it was made for a different model",
            meta.layer_count, model.block_count
        ));
    }
    for (proj, &(input, output)) in &meta.projections {
        let (want_in, want_out) = match proj.as_str() {
            "attn_q" => (Some(model.embedding_length), Some(model.q_width)),
            "attn_k" | "attn_v" => (Some(model.embedding_length), Some(model.kv_width)),
            "attn_output" => (Some(model.q_width), Some(model.embedding_length)),
            "ffn_gate" | "ffn_up" => (Some(model.embedding_length), None),
            "ffn_down" => (None, Some(model.embedding_length)),
            _ => (None, None),
        };
        if want_in.is_some_and(|w| w != input) || want_out.is_some_and(|w| w != output) {
            return Err(format!(
                "its {proj} is {input}→{output} and this model's is {}→{} — it was made for a \
                 different model",
                want_in.map_or("?".to_string(), |w| w.to_string()),
                want_out.map_or("?".to_string(), |w| w.to_string()),
            ));
        }
    }
    Ok(())
}

/// The order a model keeps its query and key ROWS in, against the Hugging Face
/// checkpoint an adapter was trained on — what an adapter's `B` matrices for
/// `attn_q` / `attn_k` must be put into before their output meets the model's.
///
/// llama.cpp's converter reorders Llama and Mistral q/k rows for its
/// interleaved RoPE (`LlamaModel.permute` in `convert_hf_to_gguf.py`), and
/// `convert_lora_to_gguf.py` runs the same `modify_tensors` over an adapter,
/// its `LoraTorchTensor` routing that reshape-and-swap onto `B`. So a PEFT
/// adapter's `B` rows are reordered identically, or each head's delta lands on
/// the wrong rotary dimensions. GLM-4, Llama-4 and DeepSeek-2 rotate
/// interleaved too, but their checkpoints are already laid out that way
/// (Llama-4's converter sets `undo_permute = False`), so theirs is
/// `AsTrained`, like every contiguous-RoPE family.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QkRowOrder {
    AsTrained,
    LlamaPermuted { n_head: usize, n_head_kv: usize },
}

impl QkRowOrder {
    pub fn for_arch(
        arch: &crate::inference::model_arch::ModelArch,
        n_head: usize,
        n_head_kv: usize,
    ) -> Self {
        use crate::inference::model_arch::ModelArch;
        match arch {
            ModelArch::Llama | ModelArch::Mistral => Self::LlamaPermuted { n_head, n_head_kv },
            _ => Self::AsTrained,
        }
    }

    /// Put an adapter's q/k `B` rows into this order.
    fn apply(self, adapter: &mut LoraAdapter) -> Result<(), SwarmError> {
        let Self::LlamaPermuted { n_head, n_head_kv } = self else {
            return Ok(());
        };
        for (key, w) in adapter.weights.iter_mut() {
            let heads = if key.ends_with(".attn_q") {
                n_head
            } else if key.ends_with(".attn_k") {
                n_head_kv
            } else {
                continue;
            };
            w.b = permute_rows_like_llama_cpp(&w.b, heads)?;
        }
        Ok(())
    }
}

/// `LlamaModel.permute` over a `B` matrix's rows: `[out, rank]` viewed as
/// `[heads, 2, out / heads / 2, rank]`, the two middle axes swapped.
fn permute_rows_like_llama_cpp(b: &Tensor, heads: usize) -> Result<Tensor, SwarmError> {
    let (out, rank) = b
        .dims2()
        .map_err(|e| SwarmError::Internal(format!("LoRA B shape: {e}")))?;
    if heads == 0 || out % (heads * 2) != 0 {
        return Err(SwarmError::Validation(format!(
            "an adapter projection of {out} rows does not split into {heads} heads"
        )));
    }
    b.reshape((heads, 2, out / heads / 2, rank))
        .and_then(|t| t.transpose(1, 2))
        .and_then(|t| t.contiguous())
        .and_then(|t| t.reshape((out, rank)))
        .map_err(|e| SwarmError::Internal(format!("LoRA row order: {e}")))
}

/// [`load_registered_adapter`], cached for the life of the calling process —
/// a model worker applies the adapter on every forward of a reply, and reading
/// and parsing the file per token would cost more than the model.
///
/// Keyed by id, the registration's BLAKE3, the device and the row order:
/// re-registering a DIFFERENT file under the same id changes the key, so a
/// stale copy is never served. The registration record is re-read on each call — a few hundred
/// bytes — which is what makes that check possible. Bounded: past a handful of
/// adapters the cache is simply cleared, since a worker serves one model.
pub fn cached_registered_adapter(
    adapter_dir: &Path,
    id: &str,
    device: &Device,
    qk_rows: QkRowOrder,
) -> Result<std::sync::Arc<LoraAdapter>, SwarmError> {
    use std::sync::{Arc, Mutex, OnceLock};
    static CACHE: OnceLock<Mutex<HashMap<String, Arc<LoraAdapter>>>> = OnceLock::new();
    const MAX_CACHED: usize = 8;
    if !is_safe_adapter_id(id) {
        return Err(SwarmError::Validation(
            "adapter id must be a single safe file name".into(),
        ));
    }
    let record_hash = std::fs::read_to_string(registration_path(adapter_dir, id))
        .ok()
        .and_then(|t| serde_json::from_str::<AdapterMetadata>(&t).ok())
        .map(|m| m.blake3)
        .unwrap_or_default();
    let key = format!("{id}\u{0}{record_hash}\u{0}{device:?}\u{0}{qk_rows:?}");
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(hit) = cache.lock().ok().and_then(|c| c.get(&key).cloned()) {
        return Ok(hit);
    }
    let adapter = Arc::new(load_registered_adapter(adapter_dir, id, device, qk_rows)?);
    if let Ok(mut c) = cache.lock() {
        if c.len() >= MAX_CACHED {
            c.clear();
        }
        c.insert(key, adapter.clone());
    }
    Ok(adapter)
}

/// Load the adapter registered as `id` — for a model WORKER, which has no
/// registry: the id's `<id>.adapter.json` names the file, rank and alpha.
///
/// Refuses an unsafe id, a file outside `adapter_dir`, and a file whose BLAKE3
/// no longer matches the registration (a swapped file is refused, never used).
/// Weights are converted to F32 on `device`, and the q/k rows put in the
/// model's order (`qk_rows` — required, because a Llama adapter applied in
/// the checkpoint's order runs without error and answers wrongly).
pub fn load_registered_adapter(
    adapter_dir: &Path,
    id: &str,
    device: &Device,
    qk_rows: QkRowOrder,
) -> Result<LoraAdapter, SwarmError> {
    if !is_safe_adapter_id(id) {
        return Err(SwarmError::Validation(
            "adapter id must be a single safe file name".into(),
        ));
    }
    let record = std::fs::read_to_string(registration_path(adapter_dir, id)).map_err(|_| {
        SwarmError::Validation(format!(
            "LoRA adapter '{id}' is not registered on this computer"
        ))
    })?;
    let meta: AdapterMetadata = serde_json::from_str(&record).map_err(|e| {
        SwarmError::Validation(format!("adapter '{id}' registration unreadable: {e}"))
    })?;
    let root = adapter_dir
        .canonicalize()
        .unwrap_or_else(|_| adapter_dir.to_path_buf());
    let file = meta.path.canonicalize().map_err(|_| {
        SwarmError::Validation(format!(
            "adapter '{id}' file is missing: {}",
            meta.path.display()
        ))
    })?;
    if !file.starts_with(&root) {
        return Err(SwarmError::Validation(format!(
            "adapter '{id}' file is outside the adapter directory"
        )));
    }
    let mut adapter = load_adapter(
        &file,
        &meta.id,
        &meta.name,
        &meta.base_model,
        meta.rank,
        meta.alpha,
        device,
    )?;
    if !adapter.metadata.blake3.eq_ignore_ascii_case(&meta.blake3) {
        return Err(SwarmError::Validation(format!(
            "adapter '{id}' file changed since it was registered — register it again"
        )));
    }
    qk_rows.apply(&mut adapter)?;
    Ok(adapter)
}

/// LoRA weight pair for a single base weight (e.g., one attention projection).
#[derive(Debug)]
pub struct LoraLayerWeights {
    /// Down-projection: A ∈ R^{rank × in_dim}
    pub a: Tensor,
    /// Up-projection: B ∈ R^{out_dim × rank}
    pub b: Tensor,
}

/// Load a LoRA adapter from a safetensors file.
///
/// Safetensors LoRA files contain tensors named like:
///   `base_model.model.model.layers.{N}.self_attn.q_proj.lora_A.weight`
///   `base_model.model.model.layers.{N}.self_attn.q_proj.lora_B.weight`
///
/// We parse these into a normalized key format:
///   `blk.{N}.attn_q` (matching GGUF naming conventions)
fn load_adapter(
    path: &Path,
    adapter_id: &str,
    name: &str,
    base_model: &str,
    rank: usize,
    alpha: f32,
    device: &Device,
) -> Result<LoraAdapter, SwarmError> {
    let file_size = std::fs::metadata(path)
        .map_err(|e| SwarmError::Internal(format!("Cannot read adapter file: {e}")))?
        .len();

    // Reject adapters larger than 2 GB to prevent OOM from crafted files
    const MAX_ADAPTER_SIZE: u64 = 2 * 1024 * 1024 * 1024;
    if file_size > MAX_ADAPTER_SIZE {
        return Err(SwarmError::Validation(format!(
            "Adapter file too large: {} bytes (max {} bytes)",
            file_size, MAX_ADAPTER_SIZE
        )));
    }

    let file_data = std::fs::read(path)
        .map_err(|e| SwarmError::Internal(format!("Failed to read adapter file: {e}")))?;

    // The hash goes into the metadata; the registration record keeps it, and
    // `load_registered_adapter` refuses a file that no longer matches. (A
    // `<file>.blake3` pin written on first load used to do this job, and since
    // nothing removed it, an updated adapter could never be registered again.)
    let computed_hex = blake3::hash(&file_data).to_hex().to_string();

    let tensors = safetensors::SafeTensors::deserialize(&file_data).map_err(|e| {
        SwarmError::Validation(format!("not a LoRA adapter (safetensors) file: {e}"))
    })?;
    // A fused projection's name CONTAINS an unfused one's (`qkv_proj` holds
    // `v_proj`, `gate_up_proj` holds `up_proj`), so `normalize_lora_key` would
    // file it under the wrong projection. The model splits its fused weights
    // at load and there is no split of the adapter to match — refuse.
    if let Some((fused, _)) = tensors
        .tensors()
        .into_iter()
        .find(|(n, _)| n.contains("qkv_proj") || n.contains("gate_up_proj"))
    {
        return Err(SwarmError::Validation(format!(
            "adapters on fused projections are not supported yet ({fused}) — Phi-3 and GLM-4 \
             adapters train these"
        )));
    }

    // Group tensors by their base weight, pairing A and B matrices
    let mut a_tensors: HashMap<String, Tensor> = HashMap::new();
    let mut b_tensors: HashMap<String, Tensor> = HashMap::new();

    for (tensor_name, _) in tensors.tensors() {
        let normalized = normalize_lora_key(&tensor_name);
        if let Some(base_key) = normalized {
            let tensor_data = tensors.tensor(&tensor_name).map_err(|e| {
                SwarmError::Internal(format!("Failed to read tensor {tensor_name}: {e}"))
            })?;

            // F32 on the model's device: `apply_lora` multiplies these by the
            // layer's activations, which are F32 wherever the layer runs, and a
            // BF16 or CPU-resident matrix does not multiply with them.
            let t = safetensor_to_candle(tensor_data, &Device::Cpu)?
                .to_dtype(DType::F32)
                .and_then(|t| t.to_device(device))
                .map_err(|e| SwarmError::Internal(format!("LoRA tensor {tensor_name}: {e}")))?;

            if tensor_name.contains("lora_A") || tensor_name.contains("lora_a") {
                a_tensors.insert(base_key, t);
            } else if tensor_name.contains("lora_B") || tensor_name.contains("lora_b") {
                b_tensors.insert(base_key, t);
            }
        }
    }

    // Pair up A and B matrices, validating each A's leading dim matches the
    // declared rank. If the registration says r=16 but the tensor shape says
    // r=8, every layer using it will silently produce wrong output because
    // `apply_lora` computes `scale = alpha / rank` (16) instead of `alpha / 8`.
    // Reject the adapter rather than load a misconfigured one.
    let mut weights = HashMap::new();
    for (key, a) in &a_tensors {
        if let Some(b) = b_tensors.get(key) {
            let actual_rank = a.dims().first().copied().unwrap_or(0);
            if actual_rank != rank {
                return Err(SwarmError::Validation(format!(
                    "LoRA rank mismatch for {key}: registered as rank {rank} but the file's matrices are rank {actual_rank} — register it with rank {actual_rank}"
                )));
            }
            weights.insert(
                key.clone(),
                LoraLayerWeights {
                    a: a.clone(),
                    b: b.clone(),
                },
            );
        } else {
            tracing::warn!(key, "LoRA A matrix without matching B matrix, skipping");
        }
    }

    let num_layers = weights.len();
    if num_layers == 0 {
        // Registering it would succeed and then change nothing — the silent
        // shape #110 was.
        return Err(SwarmError::Validation(
            "the file holds no LoRA A/B matrix pairs for attention or feed-forward projections"
                .into(),
        ));
    }
    let mut layer_count = 0usize;
    let mut projections = std::collections::BTreeMap::new();
    for (key, w) in &weights {
        // `blk.{n}.{proj}` — the form `normalize_lora_key` produced.
        let mut parts = key.splitn(3, '.');
        let (_, n, proj) = (parts.next(), parts.next(), parts.next());
        if let Some(n) = n.and_then(|n| n.parse::<usize>().ok()) {
            layer_count = layer_count.max(n + 1);
        }
        if let (Some(proj), Some(&input), Some(&output)) =
            (proj, w.a.dims().get(1), w.b.dims().first())
        {
            projections.insert(proj.to_string(), (input, output));
        }
    }
    tracing::info!(adapter_id, num_layers, rank, alpha, "Loaded LoRA adapter");

    tracing::debug!(
        adapter_id,
        name,
        base_model,
        rank,
        num_layers,
        size_bytes = file_size,
        "DIAG: lora adapter loaded"
    );

    Ok(LoraAdapter {
        metadata: AdapterMetadata {
            id: adapter_id.to_string(),
            name: name.to_string(),
            base_model: base_model.to_string(),
            rank,
            alpha,
            path: path.to_path_buf(),
            num_layers,
            size_bytes: file_size,
            blake3: computed_hex,
            layer_count,
            projections,
        },
        weights,
    })
}

/// Apply a LoRA delta to a base weight's output.
///
/// Computes: `base_output + (B @ A @ x) * scale`
/// where `scale = alpha / rank`.
///
/// `x` is `(batch, seq, in_dim)`, A is `(rank, in_dim)`, B is `(out_dim, rank)`.
/// Candle matmul requires matching dimensions, so we broadcast 2D weights to 3D.
pub fn apply_lora(
    base_output: &Tensor,
    x: &Tensor,
    lora: &LoraLayerWeights,
    alpha: f32,
    rank: usize,
) -> Result<Tensor, SwarmError> {
    if rank == 0 {
        return Err(SwarmError::Validation("LoRA rank must be > 0".into()));
    }
    let scale = alpha / rank as f32;
    // Loaded as F32 on the model's device, so this is free on every layer of a
    // model on ONE device. A hybrid model's layers past the card boundary run on
    // the processor, and there the matrices are copied to meet `x`.
    let meet = |t: &Tensor| -> Result<Tensor, SwarmError> {
        let t = if t.device().same_device(x.device()) {
            t.clone()
        } else {
            t.to_device(x.device())
                .map_err(|e| SwarmError::Internal(format!("LoRA device: {e}")))?
        };
        if t.dtype() == x.dtype() {
            Ok(t)
        } else {
            t.to_dtype(x.dtype())
                .map_err(|e| SwarmError::Internal(format!("LoRA dtype: {e}")))
        }
    };
    let (a, b) = (meet(&lora.a)?, meet(&lora.b)?);

    // A^T: (in_dim, rank) — unsqueeze to (1, in_dim, rank) for batch matmul
    let a_t = a
        .t()
        .and_then(|t| t.unsqueeze(0))
        .map_err(|e| SwarmError::Internal(format!("LoRA A prep: {e}")))?;
    // x: (b, seq, in_dim) @ A^T: (1, in_dim, rank) → (b, seq, rank)
    let ax = x
        .matmul(&a_t)
        .map_err(|e| SwarmError::Internal(format!("LoRA A matmul: {e}")))?;

    // B^T: (rank, out_dim) — unsqueeze to (1, rank, out_dim) for batch matmul
    let b_t = b
        .t()
        .and_then(|t| t.unsqueeze(0))
        .map_err(|e| SwarmError::Internal(format!("LoRA B prep: {e}")))?;
    // ax: (b, seq, rank) @ B^T: (1, rank, out_dim) → (b, seq, out_dim)
    let bax = ax
        .matmul(&b_t)
        .map_err(|e| SwarmError::Internal(format!("LoRA B matmul: {e}")))?;

    // Scale and add to base output
    let scaled =
        (bax * scale as f64).map_err(|e| SwarmError::Internal(format!("LoRA scale: {e}")))?;

    (base_output + &scaled).map_err(|e| SwarmError::Internal(format!("LoRA residual add: {e}")))
}

/// Normalize a safetensors LoRA tensor name to a GGUF-style key.
///
/// Input: `base_model.model.model.layers.5.self_attn.q_proj.lora_A.weight`
/// Output: Some("blk.5.attn_q")
fn normalize_lora_key(name: &str) -> Option<String> {
    // Try to extract layer number and projection type
    let parts: Vec<&str> = name.split('.').collect();

    // Find "layers" followed by a number
    let layer_idx = parts.iter().position(|&p| p == "layers")?;
    let layer_num: usize = parts.get(layer_idx + 1)?.parse().ok()?;

    // Determine the projection type
    let proj = if name.contains("q_proj") || name.contains("attn_q") {
        "attn_q"
    } else if name.contains("k_proj") || name.contains("attn_k") {
        "attn_k"
    } else if name.contains("v_proj") || name.contains("attn_v") {
        "attn_v"
    } else if name.contains("o_proj") || name.contains("attn_output") {
        "attn_output"
    } else if name.contains("gate_proj") || name.contains("ffn_gate") {
        "ffn_gate"
    } else if name.contains("up_proj") || name.contains("ffn_up") {
        "ffn_up"
    } else if name.contains("down_proj") || name.contains("ffn_down") {
        "ffn_down"
    } else {
        return None;
    };

    Some(format!("blk.{layer_num}.{proj}"))
}

/// Convert a safetensors tensor view to a candle Tensor.
fn safetensor_to_candle(
    view: safetensors::tensor::TensorView<'_>,
    device: &Device,
) -> Result<Tensor, SwarmError> {
    let shape: Vec<usize> = view.shape().to_vec();
    let dtype = match view.dtype() {
        safetensors::Dtype::F32 => DType::F32,
        safetensors::Dtype::F16 => DType::F16,
        safetensors::Dtype::BF16 => DType::BF16,
        other => {
            return Err(SwarmError::Validation(format!(
                "Unsupported LoRA tensor dtype: {other:?}"
            )))
        }
    };

    let data = view.data();
    Tensor::from_raw_buffer(data, dtype, &shape, device)
        .map_err(|e| SwarmError::Internal(format!("Failed to create tensor: {e}")))
}

/// The adapters registered on THIS node, by id.
///
/// Holds METADATA only. The daemon never runs a model — its workers do, and
/// they load an adapter themselves from its registration file
/// (`load_registered_adapter`) — so keeping the tensors here cost memory and
/// applied nothing. Registration loads the file once to validate it, then
/// records `<id>.adapter.json` beside it; `new` re-reads every such record, so
/// a registration survives a restart.
pub struct AdapterRegistry {
    adapters: dashmap::DashMap<String, AdapterMetadata>,
    /// Directory where adapter files are stored.
    adapter_dir: PathBuf,
}

impl AdapterRegistry {
    pub fn new(data_dir: &Path) -> Self {
        let adapter_dir = data_dir.join("adapters");
        if !adapter_dir.exists() {
            let _ = std::fs::create_dir_all(&adapter_dir);
        }
        let adapters = dashmap::DashMap::new();
        for entry in std::fs::read_dir(&adapter_dir)
            .into_iter()
            .flatten()
            .flatten()
        {
            let name = entry.file_name().to_string_lossy().to_string();
            let Some(id) = name.strip_suffix(".adapter.json") else {
                continue;
            };
            let parsed = std::fs::read_to_string(entry.path())
                .ok()
                .and_then(|t| serde_json::from_str::<AdapterMetadata>(&t).ok());
            match parsed {
                Some(meta) if meta.id == id && is_safe_adapter_id(id) => {
                    adapters.insert(id.to_string(), meta);
                }
                _ => {
                    tracing::warn!(file = %name, "Ignoring an unreadable LoRA adapter registration")
                }
            }
        }
        Self {
            adapters,
            adapter_dir,
        }
    }

    /// Register a new adapter from a file path.
    #[allow(clippy::too_many_arguments)]
    pub fn register(
        &self,
        adapter_id: &str,
        name: &str,
        base_model: &str,
        rank: usize,
        alpha: f32,
        path: &Path,
        device: &Device,
    ) -> Result<AdapterMetadata, SwarmError> {
        if !is_safe_adapter_id(adapter_id) {
            return Err(SwarmError::Validation(
                "adapter id must be a single safe file name (letters, digits, - and _)".into(),
            ));
        }
        if self.adapters.contains_key(adapter_id) {
            return Err(SwarmError::Validation(format!(
                "Adapter '{adapter_id}' already registered"
            )));
        }

        // Loaded once to prove the file IS an adapter of the declared rank —
        // then dropped: the workers load their own copy.
        let metadata =
            load_adapter(path, adapter_id, name, base_model, rank, alpha, device)?.metadata;
        let record = serde_json::to_string_pretty(&metadata)
            .map_err(|e| SwarmError::Internal(format!("adapter registration: {e}")))?;
        std::fs::write(registration_path(&self.adapter_dir, adapter_id), record).map_err(|e| {
            SwarmError::Internal(format!("could not record adapter '{adapter_id}': {e}"))
        })?;
        self.adapters
            .insert(adapter_id.to_string(), metadata.clone());
        Ok(metadata)
    }

    /// A registered adapter's metadata, by ID.
    pub fn get(&self, adapter_id: &str) -> Option<AdapterMetadata> {
        self.adapters.get(adapter_id).map(|r| r.value().clone())
    }

    /// Unregister an adapter. Its file is left on disk; its registration
    /// record goes, so it does not come back at the next start.
    pub fn remove(&self, adapter_id: &str) -> bool {
        let removed = self.adapters.remove(adapter_id).is_some();
        if removed {
            let _ = std::fs::remove_file(registration_path(&self.adapter_dir, adapter_id));
        }
        removed
    }

    /// List all registered adapters.
    pub fn list(&self) -> Vec<AdapterMetadata> {
        self.adapters.iter().map(|r| r.value().clone()).collect()
    }

    /// Get the adapter storage directory.
    pub fn adapter_dir(&self) -> &Path {
        &self.adapter_dir
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_lora_key_standard() {
        assert_eq!(
            normalize_lora_key("base_model.model.model.layers.5.self_attn.q_proj.lora_A.weight"),
            Some("blk.5.attn_q".to_string())
        );
        assert_eq!(
            normalize_lora_key("base_model.model.model.layers.12.self_attn.v_proj.lora_B.weight"),
            Some("blk.12.attn_v".to_string())
        );
        assert_eq!(
            normalize_lora_key("base_model.model.model.layers.0.mlp.gate_proj.lora_A.weight"),
            Some("blk.0.ffn_gate".to_string())
        );
    }

    #[test]
    fn normalize_lora_key_unknown_projection() {
        assert_eq!(
            normalize_lora_key("base_model.model.layers.0.some_unknown.lora_A.weight"),
            None
        );
    }

    #[test]
    fn normalize_lora_key_no_layer() {
        assert_eq!(normalize_lora_key("model.embed_tokens.weight"), None);
    }

    #[test]
    fn apply_lora_basic() {
        let device = Device::Cpu;
        let in_dim = 8;
        let out_dim = 8;
        let rank = 2;
        let alpha = 4.0;

        // Base output: (1, 1, out_dim)
        let base_output = Tensor::zeros((1, 1, out_dim), DType::F32, &device).unwrap();
        // Input: (1, 1, in_dim)
        let x = Tensor::ones((1, 1, in_dim), DType::F32, &device).unwrap();
        // A: (rank, in_dim), B: (out_dim, rank)
        let a = Tensor::ones((rank, in_dim), DType::F32, &device).unwrap();
        let b = Tensor::ones((out_dim, rank), DType::F32, &device).unwrap();

        let lora = LoraLayerWeights { a, b };
        let result = apply_lora(&base_output, &x, &lora, alpha, rank).unwrap();

        // Expected: 0 + (B @ A @ x) * (alpha/rank) = ones * in_dim * rank * (4/2) = 8 * 2 * 2 = 32
        let vals = result.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for &v in &vals {
            assert!((v - 32.0).abs() < 1e-5, "Expected 32.0, got {v}");
        }
    }

    #[test]
    fn adapter_registry_crud() {
        let dir = tempfile::tempdir().unwrap();
        let registry = AdapterRegistry::new(dir.path());

        assert!(registry.list().is_empty());
        assert!(registry.get("nonexistent").is_none());
        assert!(!registry.remove("nonexistent"));
    }

    #[test]
    fn apply_lora_multi_seq() {
        // Verify LoRA works with seq_len > 1
        let device = Device::Cpu;
        let seq_len = 4;
        let in_dim = 16;
        let out_dim = 16;
        let rank = 4;
        let alpha = 8.0;

        let base_output = Tensor::zeros((1, seq_len, out_dim), DType::F32, &device).unwrap();
        let x = Tensor::ones((1, seq_len, in_dim), DType::F32, &device).unwrap();
        let a = Tensor::ones((rank, in_dim), DType::F32, &device).unwrap();
        let b = Tensor::ones((out_dim, rank), DType::F32, &device).unwrap();
        let lora = LoraLayerWeights { a, b };

        let result = apply_lora(&base_output, &x, &lora, alpha, rank).unwrap();
        assert_eq!(result.dims(), &[1, seq_len, out_dim]);

        // Expected: (B @ A @ x) * scale = ones * in_dim * rank * (8/4) = 16 * 4 * 2 = 128
        let vals = result.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(vals.len(), seq_len * out_dim);
        for &v in &vals {
            assert!((v - 128.0).abs() < 1e-3, "Expected 128.0, got {v}");
        }
    }

    #[test]
    fn apply_lora_random_weights_changes_output() {
        // Verify LoRA actually modifies the base output (non-trivial delta)
        let device = Device::Cpu;
        let in_dim = 8;
        let out_dim = 8;
        let rank = 2;
        let alpha = 4.0;

        let base_output = Tensor::ones((1, 1, out_dim), DType::F32, &device).unwrap();
        let x = Tensor::randn(0f32, 1.0, (1, 1, in_dim), &device).unwrap();
        let a = Tensor::randn(0f32, 1.0, (rank, in_dim), &device).unwrap();
        let b = Tensor::randn(0f32, 1.0, (out_dim, rank), &device).unwrap();
        let lora = LoraLayerWeights { a, b };

        let result = apply_lora(&base_output, &x, &lora, alpha, rank).unwrap();
        let base_vals = base_output.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let result_vals = result.flatten_all().unwrap().to_vec1::<f32>().unwrap();

        // Output should differ from base (LoRA adds a non-zero delta)
        let diff: f32 = base_vals
            .iter()
            .zip(result_vals.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(diff > 0.01, "LoRA should modify the output, diff={diff}");
    }

    /// Helper to write a minimal safetensors file with given tensor data.
    fn write_safetensors(path: &std::path::Path, tensors: Vec<(String, Vec<f32>, Vec<usize>)>) {
        let byte_data: Vec<(String, Vec<u8>, Vec<usize>)> = tensors
            .into_iter()
            .map(|(name, floats, shape)| {
                let bytes: Vec<u8> = floats.iter().flat_map(|f| f.to_le_bytes()).collect();
                (name, bytes, shape)
            })
            .collect();

        let views: Vec<(String, safetensors::tensor::TensorView<'_>)> = byte_data
            .iter()
            .map(|(name, data, shape)| {
                (
                    name.clone(),
                    safetensors::tensor::TensorView::new(
                        safetensors::Dtype::F32,
                        shape.to_vec(),
                        data,
                    )
                    .unwrap(),
                )
            })
            .collect();

        let serialized = safetensors::tensor::serialize(
            views.iter().map(|(n, v)| (n.as_str(), v.clone())),
            None,
        )
        .unwrap();
        std::fs::write(path, serialized).unwrap();
    }

    #[test]
    fn load_adapter_from_generated_safetensors() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test_adapter.safetensors");

        let rank = 4usize;
        let dim = 32usize;
        let mut tensors = Vec::new();

        for layer in 0..2 {
            for proj in &["q_proj", "v_proj"] {
                let a_key =
                    format!("base_model.model.model.layers.{layer}.self_attn.{proj}.lora_A.weight");
                tensors.push((a_key, vec![0.01f32; rank * dim], vec![rank, dim]));

                let b_key =
                    format!("base_model.model.model.layers.{layer}.self_attn.{proj}.lora_B.weight");
                tensors.push((b_key, vec![0.01f32; dim * rank], vec![dim, rank]));
            }
        }

        write_safetensors(&path, tensors);

        let adapter = load_adapter(
            &path,
            "test-id",
            "Test Adapter",
            "llama-test",
            rank,
            8.0,
            &Device::Cpu,
        )
        .unwrap();

        assert_eq!(adapter.metadata.id, "test-id");
        assert_eq!(adapter.metadata.name, "Test Adapter");
        assert_eq!(adapter.metadata.rank, rank);
        assert_eq!(adapter.metadata.alpha, 8.0);
        assert_eq!(adapter.weights.len(), 4); // 2 layers × 2 projections
        assert!(adapter.weights.contains_key("blk.0.attn_q"));
        assert!(adapter.weights.contains_key("blk.0.attn_v"));
        assert!(adapter.weights.contains_key("blk.1.attn_q"));
        assert!(adapter.weights.contains_key("blk.1.attn_v"));

        let w = adapter.weights.get("blk.0.attn_q").unwrap();
        assert_eq!(w.a.dims(), &[rank, dim]);
        assert_eq!(w.b.dims(), &[dim, rank]);
    }

    #[test]
    fn load_adapter_and_apply() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("e2e_adapter.safetensors");

        let rank = 2usize;
        let dim = 8usize;
        write_safetensors(
            &path,
            vec![
                (
                    "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight".into(),
                    vec![0.1; rank * dim],
                    vec![rank, dim],
                ),
                (
                    "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight".into(),
                    vec![0.1; dim * rank],
                    vec![dim, rank],
                ),
            ],
        );

        let adapter = load_adapter(&path, "e2e", "E2E", "test", rank, 4.0, &Device::Cpu).unwrap();
        assert_eq!(adapter.weights.len(), 1);

        let base_output = Tensor::zeros((1, 1, dim), DType::F32, &Device::Cpu).unwrap();
        let x = Tensor::ones((1, 1, dim), DType::F32, &Device::Cpu).unwrap();
        let lora_weights = adapter.weights.get("blk.0.attn_q").unwrap();
        let result = apply_lora(
            &base_output,
            &x,
            lora_weights,
            adapter.metadata.alpha,
            adapter.metadata.rank,
        )
        .unwrap();

        let vals = result.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let sum: f32 = vals.iter().sum();
        assert!(
            sum.abs() > 0.001,
            "LoRA output should be non-zero, got sum={sum}"
        );

        // Uniform input + uniform weights → all output values identical
        let first = vals[0];
        for &v in &vals[1..] {
            assert!((v - first).abs() < 1e-5, "Values should be uniform");
        }
    }

    #[test]
    fn normalize_lora_key_all_projections() {
        // Verify all 7 supported projection types
        let cases = vec![
            ("layers.0.self_attn.q_proj.lora_A", "blk.0.attn_q"),
            ("layers.1.self_attn.k_proj.lora_B", "blk.1.attn_k"),
            ("layers.2.self_attn.v_proj.lora_A", "blk.2.attn_v"),
            ("layers.3.self_attn.o_proj.lora_B", "blk.3.attn_output"),
            ("layers.4.mlp.gate_proj.lora_A", "blk.4.ffn_gate"),
            ("layers.5.mlp.up_proj.lora_B", "blk.5.ffn_up"),
            ("layers.6.mlp.down_proj.lora_A", "blk.6.ffn_down"),
        ];
        for (input, expected) in cases {
            assert_eq!(
                normalize_lora_key(input),
                Some(expected.to_string()),
                "Failed for input: {input}"
            );
        }
    }

    #[test]
    fn adapter_registry_with_real_file() {
        let dir = tempfile::tempdir().unwrap();
        let adapter_path = dir.path().join("adapters").join("reg_test.safetensors");
        std::fs::create_dir_all(adapter_path.parent().unwrap()).unwrap();

        let rank = 2usize;
        let dim = 4usize;
        write_safetensors(
            &adapter_path,
            vec![
                (
                    "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight".into(),
                    vec![0.0; rank * dim],
                    vec![rank, dim],
                ),
                (
                    "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight".into(),
                    vec![0.0; dim * rank],
                    vec![dim, rank],
                ),
            ],
        );

        let registry = AdapterRegistry::new(dir.path());
        let meta = registry
            .register(
                "reg-1",
                "RegTest",
                "llama",
                rank,
                4.0,
                &adapter_path,
                &Device::Cpu,
            )
            .unwrap();
        assert_eq!(meta.id, "reg-1");
        assert_eq!(meta.num_layers, 1);

        let loaded = registry.get("reg-1").unwrap();
        assert_eq!(loaded.name, "RegTest");
        assert_eq!(registry.list().len(), 1);

        // Duplicate registration fails
        let dup = registry.register(
            "reg-1",
            "Dup",
            "llama",
            rank,
            4.0,
            &adapter_path,
            &Device::Cpu,
        );
        assert!(dup.is_err());

        // Remove
        assert!(registry.remove("reg-1"));
        assert!(registry.get("reg-1").is_none());
        assert!(registry.list().is_empty());
    }

    /// A rank-2 q/v adapter for a `dim`-wide model, every weight `value`, in
    /// `<data_dir>/adapters/<file>` — the path the admin API confines to.
    fn write_qv_adapter(data_dir: &Path, file: &str, dim: usize, value: f32) -> PathBuf {
        let path = data_dir.join("adapters").join(file);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        let mut tensors = Vec::new();
        for proj in ["q_proj", "v_proj"] {
            let base = format!("base_model.model.model.layers.1.self_attn.{proj}");
            tensors.push((
                format!("{base}.lora_A.weight"),
                vec![value; 2 * dim],
                vec![2, dim],
            ));
            tensors.push((
                format!("{base}.lora_B.weight"),
                vec![value; dim * 2],
                vec![dim, 2],
            ));
        }
        write_safetensors(&path, tensors);
        path
    }

    /// #110: registration lived only in the daemon's memory, so a worker could
    /// not find an adapter and a restart forgot every one. The record on disk
    /// is what both read now.
    #[test]
    fn a_registration_survives_a_restart_and_removing_it_forgets_it() {
        let dir = tempfile::tempdir().unwrap();
        let file = write_qv_adapter(dir.path(), "persist.safetensors", 8, 0.1);
        let registry = AdapterRegistry::new(dir.path());
        let meta = registry
            .register("keep-me", "Keep", "tiny", 2, 4.0, &file, &Device::Cpu)
            .unwrap();
        assert_eq!(meta.layer_count, 2, "layer 1 touched → two layers deep");
        assert_eq!(meta.projections.get("attn_q"), Some(&(8, 8)));

        let restarted = AdapterRegistry::new(dir.path());
        let back = restarted
            .get("keep-me")
            .expect("the registration survives a restart");
        assert_eq!(back.blake3, meta.blake3);
        assert_eq!(back.alpha, 4.0);

        assert!(restarted.remove("keep-me"));
        assert!(!registration_path(restarted.adapter_dir(), "keep-me").exists());
        assert!(AdapterRegistry::new(dir.path()).get("keep-me").is_none());
        assert!(file.exists(), "unregistering leaves the file alone");
    }

    /// A worker has no registry: it loads by id from the record, and refuses
    /// every way that record could point it somewhere it should not go.
    #[test]
    fn a_worker_loads_only_what_was_registered() {
        let dir = tempfile::tempdir().unwrap();
        let adapters = dir.path().join("adapters");
        let file = write_qv_adapter(dir.path(), "w.safetensors", 8, 0.1);
        let registry = AdapterRegistry::new(dir.path());
        registry
            .register("w", "W", "tiny", 2, 4.0, &file, &Device::Cpu)
            .unwrap();

        let loaded =
            load_registered_adapter(&adapters, "w", &Device::Cpu, QkRowOrder::AsTrained).unwrap();
        assert_eq!(loaded.weights.len(), 2);
        assert_eq!(loaded.metadata.alpha, 4.0);

        let err = |r: Result<LoraAdapter, SwarmError>| r.unwrap_err().to_string();
        assert!(err(load_registered_adapter(
            &adapters,
            "nope",
            &Device::Cpu,
            QkRowOrder::AsTrained
        ))
        .contains("not registered"));
        assert!(err(load_registered_adapter(
            &adapters,
            "../w",
            &Device::Cpu,
            QkRowOrder::AsTrained
        ))
        .contains("safe file name"));

        // A record naming a file outside the directory is refused, whatever
        // its hash says.
        let outside = write_qv_adapter(&dir.path().join("elsewhere"), "o.safetensors", 8, 0.1);
        let mut forged = registry.get("w").unwrap();
        forged.id = "forged".into();
        forged.path = outside;
        std::fs::write(
            registration_path(&adapters, "forged"),
            serde_json::to_string(&forged).unwrap(),
        )
        .unwrap();
        assert!(err(load_registered_adapter(
            &adapters,
            "forged",
            &Device::Cpu,
            QkRowOrder::AsTrained
        ))
        .contains("outside the adapter directory"));

        // The file swapped after registration is refused, never used.
        write_qv_adapter(dir.path(), "w.safetensors", 8, 0.2);
        assert!(err(load_registered_adapter(
            &adapters,
            "w",
            &Device::Cpu,
            QkRowOrder::AsTrained
        ))
        .contains("changed since it was registered"));
    }

    /// A `<file>.blake3` pin written on first load, and never removed, made an
    /// updated adapter impossible to register again under any id.
    #[test]
    fn an_updated_adapter_file_can_be_registered_again() {
        let dir = tempfile::tempdir().unwrap();
        let file = write_qv_adapter(dir.path(), "u.safetensors", 8, 0.1);
        let registry = AdapterRegistry::new(dir.path());
        let first = registry
            .register("u", "U", "tiny", 2, 4.0, &file, &Device::Cpu)
            .unwrap();
        assert!(registry.remove("u"));
        write_qv_adapter(dir.path(), "u.safetensors", 8, 0.2);
        let second = registry
            .register("u", "U", "tiny", 2, 4.0, &file, &Device::Cpu)
            .expect("the retrained file registers");
        assert_ne!(first.blake3, second.blake3);
    }

    /// The worker's cache is keyed by the registration's hash, so the file
    /// re-registered under the same id is what the next reply gets.
    #[test]
    fn the_worker_cache_never_serves_a_replaced_adapter() {
        let dir = tempfile::tempdir().unwrap();
        let adapters = dir.path().join("adapters");
        let file = write_qv_adapter(dir.path(), "c.safetensors", 8, 0.1);
        let registry = AdapterRegistry::new(dir.path());
        // An id no other test uses: the cache is process-wide.
        let id = "cache-test-replaced";
        registry
            .register(id, "C", "tiny", 2, 4.0, &file, &Device::Cpu)
            .unwrap();
        let a_value = |a: &LoraAdapter| {
            a.weights["blk.1.attn_q"]
                .a
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()[0]
        };
        let first =
            cached_registered_adapter(&adapters, id, &Device::Cpu, QkRowOrder::AsTrained).unwrap();
        assert!((a_value(&first) - 0.1).abs() < 1e-6);
        let again =
            cached_registered_adapter(&adapters, id, &Device::Cpu, QkRowOrder::AsTrained).unwrap();
        assert!(
            std::sync::Arc::ptr_eq(&first, &again),
            "a hit is not re-read"
        );

        registry.remove(id);
        write_qv_adapter(dir.path(), "c.safetensors", 8, 0.3);
        registry
            .register(id, "C", "tiny", 2, 4.0, &file, &Device::Cpu)
            .unwrap();
        let fresh =
            cached_registered_adapter(&adapters, id, &Device::Cpu, QkRowOrder::AsTrained).unwrap();
        assert!(
            (a_value(&fresh) - 0.3).abs() < 1e-6,
            "served the replaced file"
        );
    }

    #[test]
    fn a_file_with_no_adapter_matrices_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("adapters").join("empty.safetensors");
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        write_safetensors(
            &path,
            vec![("model.embed_tokens.weight".into(), vec![0.0; 4], vec![2, 2])],
        );
        let registry = AdapterRegistry::new(dir.path());
        let err = registry
            .register("e", "E", "tiny", 2, 4.0, &path, &Device::Cpu)
            .unwrap_err();
        assert!(err.to_string().contains("no LoRA"), "{err}");
        assert!(registry.list().is_empty());
    }

    /// TinyLlama's geometry: 22 blocks, 2048 wide, 32 query heads and 4 KV
    /// heads of 64.
    #[test]
    fn an_adapter_is_refused_where_it_cannot_all_be_applied() {
        let meta = |layers: usize, proj: &[(&str, (usize, usize))]| AdapterMetadata {
            id: "x".into(),
            name: "x".into(),
            base_model: "x".into(),
            rank: 8,
            alpha: 16.0,
            path: PathBuf::new(),
            num_layers: proj.len(),
            size_bytes: 0,
            blake3: String::new(),
            layer_count: layers,
            projections: proj.iter().map(|(k, v)| (k.to_string(), *v)).collect(),
        };
        let tiny = AdapterTarget {
            architecture: "llama",
            block_count: 22,
            embedding_length: 2048,
            q_width: 2048,
            kv_width: 256,
            moe: false,
        };
        let fits = |m: &AdapterMetadata| check_fits(m, &tiny);

        let made_for_it = meta(22, &[("attn_q", (2048, 2048)), ("attn_v", (2048, 256))]);
        assert!(fits(&made_for_it).is_ok());
        let ffn = meta(22, &[("ffn_up", (2048, 5632)), ("ffn_down", (5632, 2048))]);
        assert!(fits(&ffn).is_ok(), "the FFN width is not in the header");

        // Qwen2.5-Coder-7B's: 3584 wide, 28 layers.
        let qwen = meta(28, &[("attn_q", (3584, 3584))]);
        assert!(fits(&qwen).unwrap_err().contains("28 layers"));
        let wide = meta(22, &[("attn_q", (3584, 3584))]);
        assert!(fits(&wide).unwrap_err().contains("attn_q"));
        // Right width, wrong head layout — a model of the same size with
        // full multi-head attention.
        let mha = meta(22, &[("attn_v", (2048, 2048))]);
        assert!(fits(&mha).unwrap_err().contains("attn_v"));
        let out = meta(22, &[("attn_output", (2048, 4096))]);
        assert!(fits(&out).is_err());

        // Layers the executor never hands an adapter: refused, not ignored.
        for arch in ["deepseek2", "qwen35", "qwen35moe", "mamba"] {
            let other = AdapterTarget {
                architecture: arch,
                ..tiny
            };
            assert!(
                check_fits(&made_for_it, &other)
                    .unwrap_err()
                    .contains("not supported"),
                "{arch}"
            );
        }
        let moe = AdapterTarget { moe: true, ..tiny };
        assert!(check_fits(&ffn, &moe)
            .unwrap_err()
            .contains("mixture-of-experts"));
        assert!(
            check_fits(&made_for_it, &moe).is_ok(),
            "attention changes still apply to a mixture-of-experts model"
        );
    }

    /// llama.cpp's `LlamaModel.permute`, worked by hand for 2 heads of 4 rows:
    /// `[heads, 2, rows/heads/2]` = [[[0,1],[2,3]],[[4,5],[6,7]]], swap the
    /// middle axes → [[[0,2],[1,3]],[[4,6],[5,7]]].
    #[test]
    fn llama_q_and_k_rows_are_reordered_as_llama_cpp_reorders_them() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("adapters").join("rows.safetensors");
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        // B's row r holds r in both rank columns, so the order is readable.
        let rows = |n: usize| {
            (0..n)
                .flat_map(|r| [r as f32, r as f32])
                .collect::<Vec<_>>()
        };
        let mut tensors = Vec::new();
        for (proj, out) in [("q_proj", 8), ("k_proj", 4), ("v_proj", 4)] {
            let base = format!("base_model.model.model.layers.0.self_attn.{proj}");
            tensors.push((
                format!("{base}.lora_A.weight"),
                vec![0.1; 2 * 8],
                vec![2, 8],
            ));
            tensors.push((format!("{base}.lora_B.weight"), rows(out), vec![out, 2]));
        }
        write_safetensors(&path, tensors);
        let registry = AdapterRegistry::new(dir.path());
        registry
            .register("rows", "R", "tiny", 2, 4.0, &path, &Device::Cpu)
            .unwrap();
        let adapters = dir.path().join("adapters");
        let order = |a: &LoraAdapter, key: &str| -> Vec<f32> {
            let b = a.weights[key].b.to_vec2::<f32>().unwrap();
            b.iter().map(|r| r[0]).collect()
        };

        // 2 query heads, 1 KV head (GQA): k is permuted over ITS head count.
        let llama = QkRowOrder::LlamaPermuted {
            n_head: 2,
            n_head_kv: 1,
        };
        let a = load_registered_adapter(&adapters, "rows", &Device::Cpu, llama).unwrap();
        assert_eq!(order(&a, "blk.0.attn_q"), [0., 2., 1., 3., 4., 6., 5., 7.]);
        assert_eq!(order(&a, "blk.0.attn_k"), [0., 2., 1., 3.]);
        assert_eq!(
            order(&a, "blk.0.attn_v"),
            [0., 1., 2., 3.],
            "v is never permuted"
        );

        let as_trained =
            load_registered_adapter(&adapters, "rows", &Device::Cpu, QkRowOrder::AsTrained)
                .unwrap();
        assert_eq!(
            order(&as_trained, "blk.0.attn_q"),
            [0., 1., 2., 3., 4., 5., 6., 7.]
        );

        use crate::inference::model_arch::ModelArch;
        for (arch, permuted) in [
            (ModelArch::Llama, true),
            (ModelArch::Mistral, true),
            (ModelArch::Qwen2, false),
            (ModelArch::Gemma2, false),
            (ModelArch::Glm4, false),
            (ModelArch::Llama4, false),
        ] {
            let got = QkRowOrder::for_arch(&arch, 32, 4);
            assert_eq!(got != QkRowOrder::AsTrained, permuted, "{arch}");
        }
    }

    /// `qkv_proj` contains `v_proj` and `gate_up_proj` contains `up_proj`: a
    /// Phi-3 adapter would be filed under the wrong projection.
    #[test]
    fn an_adapter_on_fused_projections_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("adapters").join("phi.safetensors");
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        let base = "base_model.model.model.layers.0.self_attn.qkv_proj";
        write_safetensors(
            &path,
            vec![
                (
                    format!("{base}.lora_A.weight"),
                    vec![0.1; 2 * 8],
                    vec![2, 8],
                ),
                (
                    format!("{base}.lora_B.weight"),
                    vec![0.1; 24 * 2],
                    vec![24, 2],
                ),
            ],
        );
        let err = AdapterRegistry::new(dir.path())
            .register("phi", "P", "phi3", 2, 4.0, &path, &Device::Cpu)
            .unwrap_err();
        assert!(err.to_string().contains("fused"), "{err}");
    }

    /// The adapter is F32; a layer whose activations are another type or on
    /// another device still gets the delta instead of a matmul error.
    #[test]
    fn apply_lora_meets_the_activations_type() {
        let device = Device::Cpu;
        let x = Tensor::ones((1, 1, 4), DType::F16, &device).unwrap();
        let base = Tensor::zeros((1, 1, 4), DType::F16, &device).unwrap();
        let lora = LoraLayerWeights {
            a: Tensor::ones((2, 4), DType::F32, &device).unwrap(),
            b: Tensor::ones((4, 2), DType::F32, &device).unwrap(),
        };
        let out = apply_lora(&base, &x, &lora, 2.0, 2).unwrap();
        assert_eq!(out.dtype(), DType::F16);
        let v = out.to_dtype(DType::F32).unwrap().flatten_all().unwrap();
        // ones · 4 wide · rank 2 · scale 1 = 8
        assert_eq!(v.to_vec1::<f32>().unwrap(), vec![8.0; 4]);
    }
}
