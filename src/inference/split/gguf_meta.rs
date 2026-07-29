use std::collections::HashMap;
use std::io::{Read as IoRead, Seek, SeekFrom};
use std::path::Path;

use candle_core::quantized::gguf_file;

use crate::error::SwarmError;
use crate::inference::tokenizer::SplitTokenizer;

/// Sidecar file carrying `token_embd.weight` for weight-tied models.
///
/// A node serving the LAST pipeline segment needs the output head, but for a
/// weight-tied model that tensor physically lives in shard 0 — which that node
/// often does not hold. This file carries the raw tensor bytes so the head can
/// be loaded without shard 0. Written by `extract_tied_output_weight` and
/// `download_tied_output_weight`; read back by `ShardReader`.
pub const TIED_OUTPUT_FILENAME: &str = "tied_output_weight.bin";

/// Extract the GGUF `general.architecture` string, defaulting to `"llama"` when absent.
pub fn gguf_arch_str(ct: &gguf_file::Content) -> String {
    ct.metadata
        .get("general.architecture")
        .and_then(|v| v.to_string().ok().cloned())
        .unwrap_or_else(|| "llama".to_string())
}

/// Metadata extracted from GGUF header, stored in manifest for all nodes.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct GgufTensorMeta {
    /// Tensor name → (offset from tensor_data_start, size in bytes, dtype tag).
    pub tensors: HashMap<String, TensorLocation>,
    /// Offset in the GGUF file where tensor data begins.
    pub tensor_data_offset: u64,
    /// Friendly model name from GGUF `general.name` metadata.
    pub model_name: Option<String>,
    /// Model hyperparameters extracted from GGUF metadata.
    pub head_count: usize,
    pub head_count_kv: usize,
    pub block_count: usize,
    pub embedding_length: usize,
    /// Per-head dimension. Prefers `<arch>.attention.key_length` from GGUF
    /// (Qwen3 uses 128 vs embed/heads=64); falls back to `embedding_length /
    /// head_count`. `serde(default)` so older manifests still deserialize.
    #[serde(default)]
    pub head_dim: usize,
    pub rope_dim: usize,
    pub rope_freq_base: f32,
    pub rms_norm_eps: f64,
    /// DeepSeek-V2/V3 expert count (0 for non-MoE models).
    #[serde(default)]
    pub expert_count: usize,
    /// Raw GGUF architecture string (e.g. "llama", "qwen2", "qwen35").
    #[serde(default)]
    pub architecture: String,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct TensorLocation {
    /// Byte offset relative to tensor_data_offset.
    pub offset: u64,
    /// Total size in bytes.
    pub size: u64,
}

impl GgufTensorMeta {
    /// Location of the tensor doubling as the output head on a weight-tied
    /// model, or `None` when the model ships a separate `output.weight`.
    ///
    /// Weight tying means reusing `token_embd.weight` as the LM head, so the
    /// GGUF carries no `output.weight` at all. This is the single definition of
    /// "is this model weight-tied" — the two sidecar writers
    /// (`extract_tied_output_weight`, `download_tied_output_weight`) and the
    /// reader (`ShardReader`) all consult it, so a producer can never disagree
    /// with the consumer about which tensor the sidecar holds.
    pub fn tied_output_location(&self) -> Option<&TensorLocation> {
        if self.tensors.contains_key("output.weight") {
            return None;
        }
        self.tensors.get("token_embd.weight")
    }

    /// Extract tensor metadata from a GGUF file header on disk.
    /// Only needs to read the header, not the full file.
    pub fn from_gguf_file(path: &Path) -> Result<Self, SwarmError> {
        let mut file = std::fs::File::open(path).map_err(SwarmError::Io)?;
        let ct = gguf_file::Content::read(&mut file)
            .map_err(|e| SwarmError::Internal(format!("Failed to read GGUF header: {e}")))?;
        Self::from_content(&ct)
    }

    /// Extract tensor metadata from an already-parsed GGUF `Content`.
    /// Supports multiple architecture prefixes (llama, qwen2, mistral, etc.).
    pub fn from_content(ct: &gguf_file::Content) -> Result<Self, SwarmError> {
        let model_name = ct
            .metadata
            .get("general.name")
            .and_then(|v| v.to_string().ok().cloned());

        let arch = gguf_arch_str(ct);

        let md_get = |suffix: &str| {
            let key = format!("{arch}.{suffix}");
            ct.metadata
                .get(&key)
                .ok_or_else(|| SwarmError::Internal(format!("Missing GGUF metadata: {key}")))
        };
        let md_u32 = |suffix: &str| -> Result<usize, SwarmError> {
            Ok(md_get(suffix)?
                .to_u32()
                .map_err(|e| SwarmError::Internal(format!("Bad metadata: {e}")))?
                as usize)
        };

        let head_count = md_u32("attention.head_count")?;
        if head_count == 0 {
            return Err(SwarmError::Inference(
                "GGUF metadata error: attention.head_count is zero".into(),
            ));
        }
        // Sanity caps on peer-supplied GGUF dimensions. Without these a crafted
        // GGUF file (downloaded from network or from HF) can drive the worker
        // into oversized KV cache / mask allocations and OOM the subprocess.
        // Limits chosen well above any current real architecture:
        // - block_count (= layer count): 256 is 2× DeepSeek's 128
        // - embedding_length: 65 536 is 4× any current 70B model's hidden dim
        // - head_count: 256 (Llama-3 70B has 64)
        const MAX_BLOCK_COUNT: usize = 256;
        const MAX_EMBEDDING_LENGTH: usize = 65_536;
        const MAX_HEAD_COUNT: usize = 256;
        if head_count > MAX_HEAD_COUNT {
            return Err(SwarmError::Inference(format!(
                "GGUF metadata error: attention.head_count={head_count} exceeds cap {MAX_HEAD_COUNT}"
            )));
        }
        let head_count_kv = md_u32("attention.head_count_kv")?;
        if head_count_kv > MAX_HEAD_COUNT {
            return Err(SwarmError::Inference(format!(
                "GGUF metadata error: attention.head_count_kv={head_count_kv} exceeds cap {MAX_HEAD_COUNT}"
            )));
        }
        let block_count = md_u32("block_count")?;
        if block_count > MAX_BLOCK_COUNT {
            return Err(SwarmError::Inference(format!(
                "GGUF metadata error: block_count={block_count} exceeds cap {MAX_BLOCK_COUNT}"
            )));
        }
        let embedding_length = md_u32("embedding_length")?;
        if embedding_length == 0 || embedding_length > MAX_EMBEDDING_LENGTH {
            return Err(SwarmError::Inference(format!(
                "GGUF metadata error: embedding_length={embedding_length} out of range (1..={MAX_EMBEDDING_LENGTH})"
            )));
        }
        // head_dim: prefer attention.key_length (Qwen3 uses 128 vs embed/heads=64)
        let head_dim = ct
            .metadata
            .get(&format!("{arch}.attention.key_length"))
            .and_then(|v| v.to_u32().ok())
            .map(|v| v as usize)
            .unwrap_or(embedding_length / head_count);
        // rope.dimension_count may not exist for all architectures — derive from head_dim
        let rope_dim = md_get("rope.dimension_count")
            .and_then(|v| v.to_u32().map_err(SwarmError::internal))
            .unwrap_or(head_dim as u32) as usize;
        let rms_norm_eps = md_get("attention.layer_norm_rms_epsilon")?
            .to_f32()
            .map_err(|e| SwarmError::Internal(format!("Bad metadata: {e}")))?
            as f64;
        let rope_freq_base = ct
            .metadata
            .get(&format!("{arch}.rope.freq_base"))
            .and_then(|v| v.to_f32().ok())
            .unwrap_or(10000f32);

        let mut tensors = HashMap::new();
        for (name, info) in &ct.tensor_infos {
            // Use checked arithmetic to prevent integer overflow on crafted GGUF headers.
            // Cap elem_count to 2^40 (~1 trillion) — no legitimate tensor exceeds this.
            let block_size = info.ggml_dtype.block_size();
            let elem_count = info.shape.elem_count();
            const MAX_ELEM_COUNT: usize = 1 << 40;
            let size = if block_size == 0 || elem_count > MAX_ELEM_COUNT {
                0u64
            } else {
                info.ggml_dtype
                    .type_size()
                    .checked_mul(elem_count)
                    .map(|v| (v / block_size) as u64)
                    .unwrap_or(0)
            };
            tensors.insert(
                name.clone(),
                TensorLocation {
                    offset: info.offset,
                    size,
                },
            );
        }

        // Read expert count for DeepSeek-V2/V3 models
        let expert_count = ct
            .metadata
            .get(&format!("{arch}.expert_count"))
            .and_then(|v| v.to_u32().ok())
            .unwrap_or(0) as usize;

        Ok(GgufTensorMeta {
            tensors,
            tensor_data_offset: ct.tensor_data_offset,
            model_name,
            head_count,
            head_count_kv,
            block_count,
            embedding_length,
            head_dim,
            rope_dim,
            rope_freq_base,
            rms_norm_eps,
            expert_count,
            architecture: arch,
        })
    }
}

/// Tokenizer metadata extracted from GGUF header — consolidates all vocab/BOS/EOS/template
/// extraction that was previously duplicated across 9 call sites.
#[derive(Clone, Debug, Default)]
pub struct GgufTokenizerMeta {
    pub vocab: Vec<String>,
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<u32>,
    /// All EOS token IDs (primary + extras from `eos_token_ids` array).
    pub eos_token_ids: Vec<u32>,
    pub chat_template: Option<String>,
    pub merges: Vec<String>,
    pub pre_tokenizer: String,
    pub tokenizer_model: String,
    pub scores: Vec<f32>,
    pub add_space_prefix: bool,
    pub add_bos_token: bool,
}

impl GgufTokenizerMeta {
    /// Extract tokenizer metadata from a GGUF header file on disk.
    pub fn from_gguf_file(path: &Path) -> Result<Self, SwarmError> {
        let mut file = std::fs::File::open(path).map_err(SwarmError::Io)?;
        let ct = gguf_file::Content::read(&mut file)
            .map_err(|e| SwarmError::Internal(format!("Failed to read GGUF header: {e}")))?;
        Ok(Self::from_content(&ct))
    }

    /// Extract tokenizer metadata from an already-parsed GGUF Content.
    pub fn from_content(ct: &gguf_file::Content) -> Self {
        let md = &ct.metadata;

        let vocab: Vec<String> = md
            .get("tokenizer.ggml.tokens")
            .and_then(|v| v.to_vec().ok())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.to_string().ok().cloned())
                    .collect()
            })
            .unwrap_or_default();

        let bos_token_id = md
            .get("tokenizer.ggml.bos_token_id")
            .and_then(|v| v.to_u32().ok());

        let eos_token_id = md
            .get("tokenizer.ggml.eos_token_id")
            .and_then(|v| v.to_u32().ok());

        // Collect all EOS IDs: primary + extras array
        let mut eos_ids = Vec::new();
        if let Some(id) = eos_token_id {
            eos_ids.push(id);
        }
        if let Some(extra) = md
            .get("tokenizer.ggml.eos_token_ids")
            .and_then(|v| v.to_vec().ok())
        {
            for v in extra {
                if let Ok(id) = v.to_u32() {
                    if !eos_ids.contains(&id) {
                        eos_ids.push(id);
                    }
                }
            }
        }

        let chat_template = md
            .get("tokenizer.chat_template")
            .and_then(|v| v.to_string().ok().cloned());

        let merges: Vec<String> = md
            .get("tokenizer.ggml.merges")
            .and_then(|v| v.to_vec().ok())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.to_string().ok().cloned())
                    .collect()
            })
            .unwrap_or_default();

        let pre_tokenizer = md
            .get("tokenizer.ggml.pre")
            .and_then(|v| v.to_string().ok().cloned())
            .unwrap_or_else(|| "gpt2".to_string());

        let tokenizer_model = md
            .get("tokenizer.ggml.model")
            .and_then(|v| v.to_string().ok().cloned())
            .unwrap_or_else(|| "gpt2".to_string());

        let scores: Vec<f32> = md
            .get("tokenizer.ggml.scores")
            .and_then(|v| v.to_vec().ok())
            .map(|arr| arr.iter().filter_map(|v| v.to_f32().ok()).collect())
            .unwrap_or_default();

        let add_space_prefix = md
            .get("tokenizer.ggml.add_space_prefix")
            .and_then(|v| v.to_bool().ok())
            .unwrap_or(true);

        // llama.cpp defaults this to TRUE for SentencePiece vocabs and false
        // only for BPE; this field is consumed solely by the SPM path below.
        // Defaulting to false meant every Llama-family GGUF that simply omits
        // the key — TinyLlama and Phi-3.5 among them — was prefilled with no
        // BOS at position 0, which is out-of-distribution for models trained
        // with one and produced degenerate replies.
        let add_bos_token = md
            .get("tokenizer.ggml.add_bos_token")
            .and_then(|v| v.to_bool().ok())
            .unwrap_or(true);

        Self {
            vocab,
            bos_token_id,
            eos_token_id,
            eos_token_ids: eos_ids,
            chat_template,
            merges,
            pre_tokenizer,
            tokenizer_model,
            scores,
            add_space_prefix,
            add_bos_token,
        }
    }

    /// Resolve BOS token ID to its string representation from the vocab.
    pub fn bos_string(&self) -> String {
        self.bos_token_id
            .and_then(|id| self.vocab.get(id as usize))
            .cloned()
            .unwrap_or_default()
    }

    /// Resolve primary EOS token ID to its string representation from the vocab.
    pub fn eos_string(&self) -> String {
        self.eos_token_id
            .and_then(|id| self.vocab.get(id as usize))
            .cloned()
            .unwrap_or_default()
    }

    /// Get EOS token IDs with architecture-specific fallbacks.
    pub fn eos_tokens_with_arch_fallback(&self, arch: &str) -> Vec<u32> {
        let mut ids = self.eos_token_ids.clone();
        if ids.is_empty() {
            ids.push(2); // common default
        }
        // Qwen2 uses additional EOS tokens
        if arch.starts_with("qwen") {
            for &extra in &[151643u32, 151645] {
                if !ids.contains(&extra) {
                    ids.push(extra);
                }
            }
        }
        // Gemma uses token 107 (<end_of_turn>) as EOS
        if (arch == "gemma" || arch == "gemma2") && !ids.contains(&107) {
            ids.push(107);
        }
        ids
    }

    /// Build a `SplitTokenizer` from extracted metadata, or `None` if vocab is empty.
    pub fn build_tokenizer(&self) -> Option<SplitTokenizer> {
        if self.vocab.is_empty() {
            return None;
        }
        if !self.merges.is_empty() {
            Some(SplitTokenizer::from_bpe(
                &self.vocab,
                &self.merges,
                &self.pre_tokenizer,
                &self.tokenizer_model,
                self.add_bos_token,
                self.bos_token_id,
            ))
        } else if self.tokenizer_model == "llama" && !self.scores.is_empty() {
            Some(SplitTokenizer::from_sentencepiece(
                &self.vocab,
                &self.scores,
                self.add_space_prefix,
                self.add_bos_token,
                self.bos_token_id,
            ))
        } else {
            None
        }
    }
}

// ── GGUF Header Extraction ──

/// Save the raw GGUF header (metadata + tensor info table) to a file.
/// The header is everything from byte 0 up to (but not including) `tensor_data_offset`.
/// This allows nodes without shard_000 to reconstruct the GGUF parsing context.
///
/// The source can be a full GGUF file, OR shard_000.bin (which is the first
/// 512MB of the GGUF and always contains the complete header, since headers
/// are typically only a few MB).
pub fn save_gguf_header(gguf_or_shard0_path: &Path, output_path: &Path) -> Result<(), SwarmError> {
    let mut file = std::fs::File::open(gguf_or_shard0_path).map_err(SwarmError::Io)?;
    let ct = gguf_file::Content::read(&mut file)
        .map_err(|e| SwarmError::Internal(format!("Failed to read GGUF header: {e}")))?;

    let header_size = ct.tensor_data_offset as usize;
    // SEC: Cap header allocation to prevent OOM from malicious GGUF files
    const MAX_GGUF_HEADER_SIZE: usize = 64 * 1024 * 1024; // 64 MB
    if header_size > MAX_GGUF_HEADER_SIZE {
        return Err(SwarmError::Internal(format!(
            "GGUF header too large: {} bytes (max {})",
            header_size, MAX_GGUF_HEADER_SIZE
        )));
    }
    let mut header_buf = vec![0u8; header_size];
    file.seek(SeekFrom::Start(0)).map_err(SwarmError::Io)?;
    file.read_exact(&mut header_buf).map_err(SwarmError::Io)?;

    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent).map_err(SwarmError::Io)?;
    }
    // SEC: Atomic write to prevent corruption on kill/crash
    let tmp_path = output_path.with_extension("bin.tmp");
    std::fs::write(&tmp_path, &header_buf).map_err(SwarmError::Io)?;
    std::fs::rename(&tmp_path, output_path).map_err(SwarmError::Io)?;

    tracing::info!(
        header_bytes = header_size,
        path = %output_path.display(),
        "Saved GGUF header for shard-only operation"
    );
    Ok(())
}

/// Try to extract the GGUF header from shard_000.bin if it exists in the model directory.
/// This enables shard-only operation without needing the full GGUF or a `source_path`.
pub fn ensure_gguf_header(model_dir: &Path) -> Result<(), SwarmError> {
    let header_path = model_dir.join(crate::model::shard::HEADER_FILENAME);
    if header_path.exists() {
        return Ok(());
    }

    // shard_000.bin contains the GGUF header (first ~6MB of the file)
    let shard0_path = model_dir.join("shard_000.bin");
    if shard0_path.exists() {
        tracing::info!(
            model_dir = %model_dir.display(),
            "Extracting GGUF header from shard_000.bin"
        );
        return save_gguf_header(&shard0_path, &header_path);
    }

    // Try source_path as a fallback (with path containment check)
    let source_path_file = model_dir.join("source_path");
    if source_path_file.exists() {
        if let Ok(path_str) = std::fs::read_to_string(&source_path_file) {
            let gguf_path = std::path::PathBuf::from(path_str.trim());
            // SEC: Canonicalize both paths to prevent traversal bypass.
            // If either fails, skip — don't fall back to raw uncanonicalized paths.
            let canonical = match gguf_path.canonicalize() {
                Ok(c) => c,
                Err(_) => {
                    tracing::warn!(path = %gguf_path.display(), "source_path canonicalize failed — skipping");
                    return Err(SwarmError::Internal("source_path not resolvable".into()));
                }
            };
            // Allow source_path to be anywhere (it's typically the original GGUF
            // outside the model dir). Just verify the path exists and is a file.
            if canonical.exists() && canonical.is_file() {
                tracing::info!(
                    gguf = %canonical.display(),
                    "Extracting GGUF header from source path"
                );
                return save_gguf_header(&canonical, &header_path);
            }
        }
    }

    Err(SwarmError::Internal(format!(
        "Cannot create gguf_header.bin: no shard_000.bin or source GGUF found in {}",
        model_dir.display()
    )))
}
