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
}

/// LoRA weight pair for a single base weight (e.g., one attention projection).
#[derive(Debug)]
pub struct LoraLayerWeights {
    /// Down-projection: A ∈ R^{rank × in_dim}
    pub a: Tensor,
    /// Up-projection: B ∈ R^{out_dim × rank}
    pub b: Tensor,
}

/// Load a LoRA adapter from a directory containing a safetensors file and
/// an `adapter_config.json` metadata file.
///
/// Used by the model_worker subprocess which doesn't have access to the
/// AdapterRegistry. The config JSON must have: name, base_model, rank, alpha.
pub fn load_adapter_from_dir(dir: &Path) -> Result<LoraAdapter, SwarmError> {
    let config_path = dir.join("adapter_config.json");
    let config_str = std::fs::read_to_string(&config_path)
        .map_err(|e| SwarmError::Validation(format!("Cannot read adapter_config.json: {e}")))?;

    #[derive(Deserialize)]
    struct AdapterConfig {
        #[serde(default)]
        adapter_id: Option<String>,
        #[serde(default)]
        name: Option<String>,
        #[serde(default)]
        base_model_name_or_path: Option<String>,
        #[serde(default, alias = "rank")]
        r: Option<usize>,
        #[serde(default)]
        lora_alpha: Option<f32>,
    }

    let config: AdapterConfig = serde_json::from_str(&config_str)
        .map_err(|e| SwarmError::Validation(format!("Invalid adapter_config.json: {e}")))?;

    let adapter_id = config
        .adapter_id
        .unwrap_or_else(|| dir.file_name().unwrap_or_default().to_string_lossy().into());
    let name = config.name.unwrap_or_else(|| adapter_id.clone());
    let base_model = config.base_model_name_or_path.unwrap_or_default();
    let rank = config.r.unwrap_or(16);
    let alpha = config.lora_alpha.unwrap_or(rank as f32);

    // Find the safetensors file in the directory
    let safetensors_path = std::fs::read_dir(dir)
        .map_err(|e| SwarmError::Internal(format!("Cannot read adapter dir: {e}")))?
        .filter_map(|entry| entry.ok())
        .find(|entry| {
            entry
                .path()
                .extension()
                .is_some_and(|ext| ext == "safetensors")
        })
        .map(|entry| entry.path())
        .ok_or_else(|| {
            SwarmError::Validation("No .safetensors file found in adapter directory".into())
        })?;

    let device = Device::Cpu;
    load_adapter(
        &safetensors_path,
        &adapter_id,
        &name,
        &base_model,
        rank,
        alpha,
        &device,
    )
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
    let tensors = safetensors::SafeTensors::deserialize(&file_data)
        .map_err(|e| SwarmError::Internal(format!("Failed to parse safetensors: {e}")))?;

    // Group tensors by their base weight, pairing A and B matrices
    let mut a_tensors: HashMap<String, Tensor> = HashMap::new();
    let mut b_tensors: HashMap<String, Tensor> = HashMap::new();

    for (tensor_name, _) in tensors.tensors() {
        let normalized = normalize_lora_key(&tensor_name);
        if let Some(base_key) = normalized {
            let tensor_data = tensors.tensor(&tensor_name).map_err(|e| {
                SwarmError::Internal(format!("Failed to read tensor {tensor_name}: {e}"))
            })?;

            let t = safetensor_to_candle(tensor_data, device)?;

            if tensor_name.contains("lora_A") || tensor_name.contains("lora_a") {
                a_tensors.insert(base_key, t);
            } else if tensor_name.contains("lora_B") || tensor_name.contains("lora_b") {
                b_tensors.insert(base_key, t);
            }
        }
    }

    // Pair up A and B matrices, validating each A's leading dim matches the
    // declared rank. If `adapter_config.json` advertises r=16 but the tensor
    // shape says r=8, every layer using it will silently produce wrong output
    // because `apply_lora` computes `scale = alpha / rank` (16) instead of
    // `alpha / 8`. Reject the adapter rather than load a misconfigured one.
    let mut weights = HashMap::new();
    for (key, a) in &a_tensors {
        if let Some(b) = b_tensors.get(key) {
            let actual_rank = a.dims().first().copied().unwrap_or(0);
            if actual_rank != rank {
                return Err(SwarmError::Validation(format!(
                    "LoRA A matrix rank mismatch for {key}: adapter_config.json declares r={rank} but tensor leading dim is {actual_rank}. Adapter would compute scale = alpha / {rank} instead of alpha / {actual_rank}, producing numerically incorrect output."
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

    // A^T: (in_dim, rank) — unsqueeze to (1, in_dim, rank) for batch matmul
    let a_t = lora
        .a
        .t()
        .and_then(|t| t.unsqueeze(0))
        .map_err(|e| SwarmError::Internal(format!("LoRA A prep: {e}")))?;
    // x: (b, seq, in_dim) @ A^T: (1, in_dim, rank) → (b, seq, rank)
    let ax = x
        .matmul(&a_t)
        .map_err(|e| SwarmError::Internal(format!("LoRA A matmul: {e}")))?;

    // B^T: (rank, out_dim) — unsqueeze to (1, rank, out_dim) for batch matmul
    let b_t = lora
        .b
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

/// Registry for loaded LoRA adapters.
///
/// Thread-safe storage for adapters, keyed by adapter ID.
/// Used by SharedState to provide per-request adapter selection.
pub struct AdapterRegistry {
    adapters: dashmap::DashMap<String, std::sync::Arc<LoraAdapter>>,
    /// Directory where adapter files are stored.
    adapter_dir: PathBuf,
}

impl AdapterRegistry {
    pub fn new(data_dir: &Path) -> Self {
        let adapter_dir = data_dir.join("adapters");
        if !adapter_dir.exists() {
            let _ = std::fs::create_dir_all(&adapter_dir);
        }
        Self {
            adapters: dashmap::DashMap::new(),
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
        if self.adapters.contains_key(adapter_id) {
            return Err(SwarmError::Validation(format!(
                "Adapter '{adapter_id}' already registered"
            )));
        }

        let adapter = load_adapter(path, adapter_id, name, base_model, rank, alpha, device)?;
        let metadata = adapter.metadata.clone();
        self.adapters
            .insert(adapter_id.to_string(), std::sync::Arc::new(adapter));
        Ok(metadata)
    }

    /// Get a loaded adapter by ID.
    pub fn get(&self, adapter_id: &str) -> Option<std::sync::Arc<LoraAdapter>> {
        self.adapters.get(adapter_id).map(|r| r.value().clone())
    }

    /// Remove an adapter.
    pub fn remove(&self, adapter_id: &str) -> bool {
        self.adapters.remove(adapter_id).is_some()
    }

    /// List all registered adapters.
    pub fn list(&self) -> Vec<AdapterMetadata> {
        self.adapters
            .iter()
            .map(|r| r.value().metadata.clone())
            .collect()
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
        assert_eq!(loaded.metadata.name, "RegTest");
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
}
