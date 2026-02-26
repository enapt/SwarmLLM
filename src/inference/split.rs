//! Split inference engine using candle for layer-range execution.
//!
//! This module enables true distributed inference where each node processes
//! only the transformer layers it holds, forwarding hidden-state activations
//! between nodes. Uses candle for direct tensor computation with quantized
//! GGUF weights.

use std::collections::HashMap;
use std::path::Path;

use candle_core::quantized::gguf_file;
use candle_core::quantized::QTensor;
use candle_core::{DType, Device, IndexOp, Result as CandleResult, Tensor};
use candle_nn::{Embedding, Module};
use candle_transformers::quantized_nn::RmsNorm;

use crate::error::SwarmError;

const DEFAULT_MAX_SEQ_LEN: usize = 4096;

// ── BPE Tokenizer from GGUF merges ──

/// GPT-2/Qwen2 BPE tokenizer built from GGUF metadata.
/// Implements proper pre-tokenization, byte-level encoding, and BPE merging.
pub struct BpeTokenizer {
    /// token string → token ID
    token_to_id: HashMap<String, u32>,
    /// Merge pair (left, right) → merge rank (lower = higher priority)
    merge_ranks: HashMap<(String, String), usize>,
    /// Byte → GPT-2 unicode character mapping
    byte_encoder: [char; 256],
    /// GPT-2 unicode char → byte reverse mapping
    byte_decoder: HashMap<char, u8>,
    /// Pre-tokenization regex pattern
    pre_tok_re: fancy_regex::Regex,
    /// Special tokens sorted by length descending (for matching)
    special_tokens: Vec<(String, u32)>,
}

impl BpeTokenizer {
    /// Build a BPE tokenizer from GGUF vocabulary tokens, merge rules,
    /// and pre-tokenizer type.
    fn from_gguf(tokens: &[String], merges_raw: &[String], pre_type: &str) -> Self {
        let mut token_to_id = HashMap::with_capacity(tokens.len());
        for (i, tok) in tokens.iter().enumerate() {
            token_to_id.insert(tok.clone(), i as u32);
        }

        // Build merge rank lookup: (left, right) → rank
        let mut merge_ranks = HashMap::with_capacity(merges_raw.len());
        for (rank, line) in merges_raw.iter().enumerate() {
            let parts: Vec<&str> = line.splitn(2, ' ').collect();
            if parts.len() == 2 {
                merge_ranks.insert((parts[0].to_string(), parts[1].to_string()), rank);
            }
        }

        // Build GPT-2 byte encoder
        let (byte_encoder, byte_decoder) = build_gpt2_byte_encoder();

        // Pre-tokenization regex based on model type
        let pattern = match pre_type {
            "qwen2" => {
                // Qwen2 pre-tokenization pattern (from HuggingFace tokenizers)
                r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
            }
            "gpt-2" | "gpt2" => {
                r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
            }
            _ => {
                // Default fallback: split on whitespace boundaries
                r"[^\s]+|\s+"
            }
        };
        let pre_tok_re = fancy_regex::Regex::new(pattern)
            .unwrap_or_else(|_| fancy_regex::Regex::new(r"[^\s]+|\s+").unwrap());

        // Collect special tokens (e.g., <|im_start|>, <|im_end|>)
        let mut special_tokens: Vec<(String, u32)> = token_to_id
            .iter()
            .filter(|(t, _)| t.starts_with("<|") && t.ends_with("|>"))
            .map(|(t, &id)| (t.clone(), id))
            .collect();
        // Sort by length descending for longest-match-first
        special_tokens.sort_by(|a, b| b.0.len().cmp(&a.0.len()));

        Self {
            token_to_id,
            merge_ranks,
            byte_encoder,
            byte_decoder,
            pre_tok_re,
            special_tokens,
        }
    }

    /// Encode a string into token IDs.
    pub fn encode(&self, text: &str) -> Vec<i64> {
        if text.is_empty() {
            return vec![];
        }

        // 1. Split on special tokens first
        let segments = self.split_special_tokens(text);
        let mut all_ids = Vec::new();

        for (segment, is_special) in &segments {
            if *is_special {
                if let Some(&id) = self.token_to_id.get(segment.as_str()) {
                    all_ids.push(id as i64);
                }
            } else {
                // 2. Pre-tokenize regular text
                let pre_tokens = self.pre_tokenize(segment);
                // 3. BPE encode each pre-token
                for pre_tok in &pre_tokens {
                    all_ids.extend(self.bpe_encode_word(pre_tok));
                }
            }
        }

        all_ids
    }

    /// Split text at special token boundaries.
    fn split_special_tokens(&self, text: &str) -> Vec<(String, bool)> {
        let mut result = Vec::new();
        let mut remaining = text;

        while !remaining.is_empty() {
            // Check if remaining starts with any special token
            let mut found = false;
            for (special, _) in &self.special_tokens {
                if remaining.starts_with(special.as_str()) {
                    result.push((special.clone(), true));
                    remaining = &remaining[special.len()..];
                    found = true;
                    break;
                }
            }
            if !found {
                // Find next special token occurrence
                let next_pos = self
                    .special_tokens
                    .iter()
                    .filter_map(|(s, _)| remaining.find(s.as_str()))
                    .min();
                match next_pos {
                    Some(pos) => {
                        result.push((remaining[..pos].to_string(), false));
                        remaining = &remaining[pos..];
                    }
                    None => {
                        result.push((remaining.to_string(), false));
                        remaining = "";
                    }
                }
            }
        }
        result
    }

    /// Pre-tokenize text using the model's regex pattern.
    fn pre_tokenize(&self, text: &str) -> Vec<String> {
        let mut pieces = Vec::new();
        let mut search_start = 0;
        while search_start < text.len() {
            match self.pre_tok_re.find_from_pos(text, search_start) {
                Ok(Some(m)) => {
                    pieces.push(m.as_str().to_string());
                    search_start = m.end();
                }
                _ => break,
            }
        }
        pieces
    }

    /// BPE encode a single pre-token word.
    /// Converts bytes → GPT-2 unicode chars, then applies BPE merges.
    fn bpe_encode_word(&self, word: &str) -> Vec<i64> {
        // Convert each byte to its GPT-2 unicode character
        let chars: Vec<String> = word
            .bytes()
            .map(|b| self.byte_encoder[b as usize].to_string())
            .collect();

        if chars.is_empty() {
            return vec![];
        }

        // Single char: direct lookup
        if chars.len() == 1 {
            return vec![self
                .token_to_id
                .get(&chars[0])
                .copied()
                .unwrap_or(0) as i64];
        }

        // Apply BPE merges using the standard algorithm:
        // Repeatedly find the highest-priority (lowest rank) merge pair and apply it.
        let mut symbols = chars;
        loop {
            // Find the pair with the lowest merge rank
            let mut best_rank = usize::MAX;
            let mut best_idx = usize::MAX;
            for i in 0..symbols.len() - 1 {
                let pair = (symbols[i].clone(), symbols[i + 1].clone());
                if let Some(&rank) = self.merge_ranks.get(&pair) {
                    if rank < best_rank {
                        best_rank = rank;
                        best_idx = i;
                    }
                }
            }

            if best_idx == usize::MAX {
                break; // No more merges applicable
            }

            // Apply the merge: combine symbols[best_idx] and symbols[best_idx+1]
            let merged = format!("{}{}", symbols[best_idx], symbols[best_idx + 1]);
            symbols[best_idx] = merged;
            symbols.remove(best_idx + 1);

            if symbols.len() == 1 {
                break;
            }
        }

        // Convert BPE tokens to IDs
        symbols
            .iter()
            .map(|t| self.token_to_id.get(t).copied().unwrap_or(0) as i64)
            .collect()
    }

    /// Decode a BPE token string back to UTF-8 bytes.
    /// Reverses the GPT-2 unicode byte encoding.
    pub fn decode_token(&self, token_str: &str) -> Vec<u8> {
        token_str
            .chars()
            .map(|ch| self.byte_decoder.get(&ch).copied().unwrap_or(b'?'))
            .collect()
    }
}

/// Build the GPT-2 byte encoder mapping.
/// Maps each byte (0-255) to a unicode character such that:
/// - Printable bytes map to themselves (as unicode chars)
/// - Non-printable bytes map to U+0100, U+0101, etc.
fn build_gpt2_byte_encoder() -> ([char; 256], HashMap<char, u8>) {
    let mut encoder = ['\0'; 256];
    let mut decoder = HashMap::new();
    let mut offset = 0u32;

    for b in 0u16..=255 {
        let is_printable = (33..=126).contains(&b)
            || (161..=172).contains(&b)
            || (174..=255).contains(&b);
        if is_printable {
            let ch = char::from_u32(b as u32).unwrap();
            encoder[b as usize] = ch;
            decoder.insert(ch, b as u8);
        } else {
            let ch = char::from_u32(256 + offset).unwrap();
            encoder[b as usize] = ch;
            decoder.insert(ch, b as u8);
            offset += 1;
        }
    }

    (encoder, decoder)
}

// ── Quantized MatMul wrapper ──

#[derive(Debug, Clone)]
struct QMatMul {
    inner: candle_core::quantized::QMatMul,
}

impl QMatMul {
    fn from_qtensor(qtensor: QTensor) -> CandleResult<Self> {
        let inner = candle_core::quantized::QMatMul::from_qtensor(qtensor)?;
        Ok(Self { inner })
    }

    fn forward(&self, xs: &Tensor) -> CandleResult<Tensor> {
        self.inner.forward(xs)
    }
}

// ── MLP / FFN ──

#[derive(Debug, Clone)]
struct Mlp {
    ffn_gate: QMatMul,
    ffn_down: QMatMul,
    ffn_up: QMatMul,
}

impl Mlp {
    fn forward(&self, xs: &Tensor) -> CandleResult<Tensor> {
        let gate = self.ffn_gate.forward(xs)?;
        let up = self.ffn_up.forward(xs)?;
        self.ffn_down.forward(&(candle_nn::ops::silu(&gate)? * up)?)
    }
}

// ── Per-layer weights ──

#[derive(Debug, Clone)]
struct LayerWeights {
    attention_wq: QMatMul,
    attention_wk: QMatMul,
    attention_wv: QMatMul,
    attention_wo: QMatMul,
    /// Qwen2 has QKV biases; for architectures without biases these are None.
    attention_bq: Option<Tensor>,
    attention_bk: Option<Tensor>,
    attention_bv: Option<Tensor>,
    attention_norm: RmsNorm,
    mlp: Mlp,
    ffn_norm: RmsNorm,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    cos: Tensor,
    sin: Tensor,
    neg_inf: Tensor,
    kv_cache: Option<(Tensor, Tensor)>,
    /// If true, use contiguous RoPE (rope); if false, use interleaved (rope_i).
    use_rope_contiguous: bool,
}

fn masked_fill(on_false: &Tensor, mask: &Tensor, on_true: &Tensor) -> CandleResult<Tensor> {
    let shape = mask.shape();
    mask.where_cond(&on_true.broadcast_as(shape.dims())?, on_false)
}

impl LayerWeights {
    fn apply_rotary_emb(&self, x: &Tensor, index_pos: usize) -> CandleResult<Tensor> {
        let (_b_sz, _n_head, seq_len, _n_embd) = x.dims4()?;
        let cos = self.cos.narrow(0, index_pos, seq_len)?;
        let sin = self.sin.narrow(0, index_pos, seq_len)?;
        if self.use_rope_contiguous {
            candle_nn::rotary_emb::rope(&x.contiguous()?, &cos, &sin)
        } else {
            candle_nn::rotary_emb::rope_i(&x.contiguous()?, &cos, &sin)
        }
    }

    fn forward_attn(
        &mut self,
        x: &Tensor,
        mask: Option<&Tensor>,
        index_pos: usize,
    ) -> CandleResult<Tensor> {
        let (b_sz, seq_len, n_embd) = x.dims3()?;
        let mut q = self.attention_wq.forward(x)?;
        let mut k = self.attention_wk.forward(x)?;
        let mut v = self.attention_wv.forward(x)?;

        // Apply QKV biases if present (Qwen2 has biases)
        if let Some(ref bq) = self.attention_bq {
            q = q.broadcast_add(bq)?;
        }
        if let Some(ref bk) = self.attention_bk {
            k = k.broadcast_add(bk)?;
        }
        if let Some(ref bv) = self.attention_bv {
            v = v.broadcast_add(bv)?;
        }

        let q = q
            .reshape((b_sz, seq_len, self.n_head, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = k
            .reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = v
            .reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        let q = self.apply_rotary_emb(&q, index_pos)?;
        let k = self.apply_rotary_emb(&k, index_pos)?;

        // KV-cache concatenation
        let (k, v) = match &self.kv_cache {
            None => (k, v),
            Some((k_cache, v_cache)) => {
                if index_pos == 0 {
                    (k, v)
                } else {
                    let k = Tensor::cat(&[k_cache, &k], 2)?;
                    let v = Tensor::cat(&[v_cache, &v], 2)?;
                    (k, v)
                }
            }
        };
        self.kv_cache = Some((k.clone(), v.clone()));

        // GQA: repeat K/V heads to match Q head count
        let k = candle_transformers::utils::repeat_kv(k, self.n_head / self.n_kv_head)?;
        let v = candle_transformers::utils::repeat_kv(v, self.n_head / self.n_kv_head)?;

        let att = (q.matmul(&k.t()?)? / (self.head_dim as f64).sqrt())?;
        let att = match mask {
            None => att,
            Some(mask) => {
                let mask = mask.broadcast_as(att.shape())?;
                masked_fill(&att, &mask, &self.neg_inf)?
            }
        };
        let att = candle_nn::ops::softmax_last_dim(&att)?;
        let y = att.matmul(&v.contiguous()?)?;

        let y = y.transpose(1, 2)?.reshape(&[b_sz, seq_len, n_embd])?;
        self.attention_wo.forward(&y)
    }
}

// ── Split model: loads only a range of layers from a GGUF ──

/// A partial Llama model that loads and runs only a specific range of layers.
/// Used for split inference where each node holds different layers.
pub struct SplitModel {
    /// Token embedding table (only loaded by the first segment).
    tok_embeddings: Option<Embedding>,
    /// Transformer layers for this segment's range.
    layers: Vec<LayerWeights>,
    /// Final RMSNorm (only loaded by the last segment).
    norm: Option<RmsNorm>,
    /// LM head / output projection (only loaded by the last segment).
    output: Option<QMatMul>,
    /// Causal attention masks cache.
    masks: HashMap<usize, Tensor>,
    /// Layer range this model covers: [start, end) out of total_layers.
    pub layer_start: usize,
    pub layer_end: usize,
    pub total_layers: usize,
    /// Hidden dimension (embedding_length).
    pub hidden_dim: usize,
    /// Device (CPU or CUDA).
    device: Device,
    /// Vocabulary from GGUF (token ID → string), for decoding generated tokens.
    vocabulary: Option<Vec<String>>,
    /// BPE tokenizer built from GGUF merges table.
    tokenizer: Option<BpeTokenizer>,
}

/// Metadata extracted from GGUF header, stored in manifest for all nodes.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct GgufTensorMeta {
    /// Tensor name → (offset from tensor_data_start, size in bytes, dtype tag).
    pub tensors: HashMap<String, TensorLocation>,
    /// Offset in the GGUF file where tensor data begins.
    pub tensor_data_offset: u64,
    /// Model hyperparameters extracted from GGUF metadata.
    pub head_count: usize,
    pub head_count_kv: usize,
    pub block_count: usize,
    pub embedding_length: usize,
    pub rope_dim: usize,
    pub rope_freq_base: f32,
    pub rms_norm_eps: f64,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct TensorLocation {
    /// Byte offset relative to tensor_data_offset.
    pub offset: u64,
    /// Total size in bytes.
    pub size: u64,
}

impl GgufTensorMeta {
    /// Extract tensor metadata from a GGUF file header.
    /// Only needs to read the header, not the full file.
    /// Supports multiple architecture prefixes (llama, qwen2, mistral, etc.)
    pub fn from_gguf_file(path: &Path) -> Result<Self, SwarmError> {
        let mut file = std::fs::File::open(path).map_err(SwarmError::Io)?;
        let ct = gguf_file::Content::read(&mut file)
            .map_err(|e| SwarmError::Internal(format!("Failed to read GGUF header: {e}")))?;

        // Detect architecture prefix from general.architecture metadata
        let arch = ct
            .metadata
            .get("general.architecture")
            .and_then(|v| v.to_string().ok().cloned())
            .unwrap_or_else(|| "llama".to_string());

        let md_get = |suffix: &str| {
            let key = format!("{arch}.{suffix}");
            ct.metadata
                .get(&key)
                .ok_or_else(|| SwarmError::Internal(format!("Missing GGUF metadata: {key}")))
        };

        let head_count = md_get("attention.head_count")?
            .to_u32()
            .map_err(|e| SwarmError::Internal(format!("Bad metadata: {e}")))?
            as usize;
        let head_count_kv = md_get("attention.head_count_kv")?
            .to_u32()
            .map_err(|e| SwarmError::Internal(format!("Bad metadata: {e}")))?
            as usize;
        let block_count = md_get("block_count")?
            .to_u32()
            .map_err(|e| SwarmError::Internal(format!("Bad metadata: {e}")))?
            as usize;
        let embedding_length = md_get("embedding_length")?
            .to_u32()
            .map_err(|e| SwarmError::Internal(format!("Bad metadata: {e}")))?
            as usize;
        // rope.dimension_count may not exist for all architectures — derive from head_dim
        let rope_dim = md_get("rope.dimension_count")
            .and_then(|v| v.to_u32().map_err(|e| SwarmError::Internal(e.to_string())))
            .unwrap_or((embedding_length / head_count) as u32) as usize;
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
            let size = info.ggml_dtype.type_size() * info.shape.elem_count()
                / info.ggml_dtype.block_size();
            tensors.insert(
                name.clone(),
                TensorLocation {
                    offset: info.offset,
                    size: size as u64,
                },
            );
        }

        Ok(GgufTensorMeta {
            tensors,
            tensor_data_offset: ct.tensor_data_offset,
            head_count,
            head_count_kv,
            block_count,
            embedding_length,
            rope_dim,
            rope_freq_base,
            rms_norm_eps,
        })
    }
}

fn precompute_freqs_cis(
    head_dim: usize,
    freq_base: f32,
    max_seq_len: usize,
    device: &Device,
) -> CandleResult<(Tensor, Tensor)> {
    let theta: Vec<_> = (0..head_dim)
        .step_by(2)
        .map(|i| 1f32 / freq_base.powf(i as f32 / head_dim as f32))
        .collect();
    let theta = Tensor::new(theta.as_slice(), device)?;
    let idx_theta = Tensor::arange(0, max_seq_len as u32, device)?
        .to_dtype(DType::F32)?
        .reshape((max_seq_len, 1))?
        .matmul(&theta.reshape((1, theta.elem_count()))?)?;
    Ok((idx_theta.cos()?, idx_theta.sin()?))
}

impl SplitModel {
    /// Load a partial model from a GGUF file, only loading the specified layer range.
    ///
    /// - `layer_start..layer_end`: the transformer block range this node owns
    /// - `is_first`: if true, also loads the embedding table
    /// - `is_last`: if true, also loads the final norm and LM head
    pub fn load_from_gguf(
        gguf_path: &Path,
        layer_start: usize,
        layer_end: usize,
        is_first: bool,
        is_last: bool,
    ) -> Result<Self, SwarmError> {
        let mut file = std::fs::File::open(gguf_path).map_err(SwarmError::Io)?;
        let ct = gguf_file::Content::read(&mut file)
            .map_err(|e| SwarmError::Internal(format!("Failed to read GGUF: {e}")))?;

        let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
        if device.is_cuda() {
            tracing::info!("Split model using CUDA GPU");
        } else {
            tracing::info!("Split model using CPU (no CUDA available)");
        }

        // Detect architecture prefix from GGUF metadata
        let arch = ct
            .metadata
            .get("general.architecture")
            .and_then(|v| v.to_string().ok().cloned())
            .unwrap_or_else(|| "llama".to_string());

        let md_get = |suffix: &str| {
            let key = format!("{arch}.{suffix}");
            ct.metadata
                .get(&key)
                .ok_or_else(|| SwarmError::Internal(format!("Missing GGUF metadata: {key}")))
        };

        let head_count = md_get("attention.head_count")?
            .to_u32()
            .map_err(|e| SwarmError::Internal(e.to_string()))? as usize;
        let head_count_kv = md_get("attention.head_count_kv")?
            .to_u32()
            .map_err(|e| SwarmError::Internal(e.to_string()))? as usize;
        let block_count = md_get("block_count")?
            .to_u32()
            .map_err(|e| SwarmError::Internal(e.to_string()))? as usize;
        let embedding_length = md_get("embedding_length")?
            .to_u32()
            .map_err(|e| SwarmError::Internal(e.to_string()))?
            as usize;
        let rope_dim = md_get("rope.dimension_count")
            .and_then(|v| v.to_u32().map_err(|e| SwarmError::Internal(e.to_string())))
            .unwrap_or((embedding_length / head_count) as u32) as usize;
        let rms_norm_eps = md_get("attention.layer_norm_rms_epsilon")?
            .to_f32()
            .map_err(|e| SwarmError::Internal(e.to_string()))? as f64;
        let rope_freq_base = ct
            .metadata
            .get(&format!("{arch}.rope.freq_base"))
            .and_then(|v| v.to_f32().ok())
            .unwrap_or(10000f32);
        let context_length = md_get("context_length")
            .and_then(|v| v.to_u32().map_err(|e| SwarmError::Internal(e.to_string())))
            .unwrap_or(DEFAULT_MAX_SEQ_LEN as u32) as usize;

        // Determine RoPE variant: Qwen2 uses contiguous (split), Llama uses interleaved
        let use_rope_contiguous = matches!(arch.as_str(), "qwen2" | "qwen3");

        let head_dim = embedding_length / head_count;
        let (cos, sin) = precompute_freqs_cis(rope_dim, rope_freq_base, context_length, &device)
            .map_err(|e| SwarmError::Internal(e.to_string()))?;
        let neg_inf = Tensor::new(f32::NEG_INFINITY, &device)
            .map_err(|e| SwarmError::Internal(e.to_string()))?;

        // Load embedding table only for first segment
        let tok_embeddings = if is_first {
            let tok_embd = ct
                .tensor(&mut file, "token_embd.weight", &device)
                .map_err(|e| SwarmError::Internal(format!("Failed to load embeddings: {e}")))?;
            let tok_embd = tok_embd
                .dequantize(&device)
                .map_err(|e| SwarmError::Internal(e.to_string()))?;
            Some(Embedding::new(tok_embd, embedding_length))
        } else {
            None
        };

        // Load output norm and LM head only for last segment
        let norm = if is_last {
            let norm_tensor = ct
                .tensor(&mut file, "output_norm.weight", &device)
                .map_err(|e| SwarmError::Internal(format!("Failed to load output_norm: {e}")))?;
            Some(
                RmsNorm::from_qtensor(norm_tensor, rms_norm_eps)
                    .map_err(|e| SwarmError::Internal(e.to_string()))?,
            )
        } else {
            None
        };

        let output = if is_last {
            let output_tensor = ct
                .tensor(&mut file, "output.weight", &device)
                .or_else(|_| ct.tensor(&mut file, "token_embd.weight", &device))
                .map_err(|e| SwarmError::Internal(format!("Failed to load output head: {e}")))?;
            Some(
                QMatMul::from_qtensor(output_tensor)
                    .map_err(|e| SwarmError::Internal(e.to_string()))?,
            )
        } else {
            None
        };

        // Load only the specified layer range (capped at actual block count)
        let layer_end = layer_end.min(block_count);
        let mut layers = Vec::with_capacity(layer_end - layer_start);
        for layer_idx in layer_start..layer_end {
            let prefix = format!("blk.{layer_idx}");

            let attention_wq = ct
                .tensor(&mut file, &format!("{prefix}.attn_q.weight"), &device)
                .map_err(|e| {
                    SwarmError::Internal(format!("Failed to load {prefix}.attn_q: {e}"))
                })?;
            let attention_wk = ct
                .tensor(&mut file, &format!("{prefix}.attn_k.weight"), &device)
                .map_err(|e| {
                    SwarmError::Internal(format!("Failed to load {prefix}.attn_k: {e}"))
                })?;
            let attention_wv = ct
                .tensor(&mut file, &format!("{prefix}.attn_v.weight"), &device)
                .map_err(|e| {
                    SwarmError::Internal(format!("Failed to load {prefix}.attn_v: {e}"))
                })?;
            let attention_wo = ct
                .tensor(&mut file, &format!("{prefix}.attn_output.weight"), &device)
                .map_err(|e| {
                    SwarmError::Internal(format!("Failed to load {prefix}.attn_output: {e}"))
                })?;

            // Load QKV biases (present in Qwen2, absent in Llama)
            let attention_bq = ct
                .tensor(&mut file, &format!("{prefix}.attn_q.bias"), &device)
                .ok()
                .map(|t| t.dequantize(&device))
                .transpose()
                .map_err(|e| SwarmError::Internal(format!("attn_q.bias dequant: {e}")))?;
            let attention_bk = ct
                .tensor(&mut file, &format!("{prefix}.attn_k.bias"), &device)
                .ok()
                .map(|t| t.dequantize(&device))
                .transpose()
                .map_err(|e| SwarmError::Internal(format!("attn_k.bias dequant: {e}")))?;
            let attention_bv = ct
                .tensor(&mut file, &format!("{prefix}.attn_v.bias"), &device)
                .ok()
                .map(|t| t.dequantize(&device))
                .transpose()
                .map_err(|e| SwarmError::Internal(format!("attn_v.bias dequant: {e}")))?;

            let ffn_gate = ct
                .tensor(&mut file, &format!("{prefix}.ffn_gate.weight"), &device)
                .map_err(|e| {
                    SwarmError::Internal(format!("Failed to load {prefix}.ffn_gate: {e}"))
                })?;
            let ffn_down = ct
                .tensor(&mut file, &format!("{prefix}.ffn_down.weight"), &device)
                .map_err(|e| {
                    SwarmError::Internal(format!("Failed to load {prefix}.ffn_down: {e}"))
                })?;
            let ffn_up = ct
                .tensor(&mut file, &format!("{prefix}.ffn_up.weight"), &device)
                .map_err(|e| {
                    SwarmError::Internal(format!("Failed to load {prefix}.ffn_up: {e}"))
                })?;
            let attn_norm = ct
                .tensor(&mut file, &format!("{prefix}.attn_norm.weight"), &device)
                .map_err(|e| {
                    SwarmError::Internal(format!("Failed to load {prefix}.attn_norm: {e}"))
                })?;
            let ffn_norm = ct
                .tensor(&mut file, &format!("{prefix}.ffn_norm.weight"), &device)
                .map_err(|e| {
                    SwarmError::Internal(format!("Failed to load {prefix}.ffn_norm: {e}"))
                })?;

            layers.push(LayerWeights {
                attention_wq: QMatMul::from_qtensor(attention_wq)
                    .map_err(|e| SwarmError::Internal(e.to_string()))?,
                attention_wk: QMatMul::from_qtensor(attention_wk)
                    .map_err(|e| SwarmError::Internal(e.to_string()))?,
                attention_wv: QMatMul::from_qtensor(attention_wv)
                    .map_err(|e| SwarmError::Internal(e.to_string()))?,
                attention_wo: QMatMul::from_qtensor(attention_wo)
                    .map_err(|e| SwarmError::Internal(e.to_string()))?,
                attention_bq,
                attention_bk,
                attention_bv,
                attention_norm: RmsNorm::from_qtensor(attn_norm, rms_norm_eps)
                    .map_err(|e| SwarmError::Internal(e.to_string()))?,
                mlp: Mlp {
                    ffn_gate: QMatMul::from_qtensor(ffn_gate)
                        .map_err(|e| SwarmError::Internal(e.to_string()))?,
                    ffn_down: QMatMul::from_qtensor(ffn_down)
                        .map_err(|e| SwarmError::Internal(e.to_string()))?,
                    ffn_up: QMatMul::from_qtensor(ffn_up)
                        .map_err(|e| SwarmError::Internal(e.to_string()))?,
                },
                ffn_norm: RmsNorm::from_qtensor(ffn_norm, rms_norm_eps)
                    .map_err(|e| SwarmError::Internal(e.to_string()))?,
                n_head: head_count,
                n_kv_head: head_count_kv,
                head_dim,
                cos: cos.clone(),
                sin: sin.clone(),
                neg_inf: neg_inf.clone(),
                kv_cache: None,
                use_rope_contiguous,
            });
        }

        // Load vocabulary from GGUF metadata for token decoding
        let vocabulary = ct
            .metadata
            .get("tokenizer.ggml.tokens")
            .and_then(|v| v.to_vec().ok())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.to_string().ok().cloned())
                    .collect::<Vec<String>>()
            });
        if let Some(ref v) = vocabulary {
            tracing::info!(vocab_size = v.len(), "Loaded GGUF vocabulary");
        }

        // Load BPE merges, pre-tokenizer type, and build tokenizer
        let tokenizer = if let Some(ref vocab) = vocabulary {
            let merges_raw = ct
                .metadata
                .get("tokenizer.ggml.merges")
                .and_then(|v| v.to_vec().ok())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.to_string().ok().cloned())
                        .collect::<Vec<String>>()
                })
                .unwrap_or_default();
            let pre_type = ct
                .metadata
                .get("tokenizer.ggml.pre")
                .and_then(|v| v.to_string().ok().cloned())
                .unwrap_or_else(|| "gpt2".to_string());
            if !merges_raw.is_empty() {
                tracing::info!(
                    merges = merges_raw.len(),
                    pre_type = %pre_type,
                    "Loaded BPE tokenizer from GGUF"
                );
                Some(BpeTokenizer::from_gguf(vocab, &merges_raw, &pre_type))
            } else {
                None
            }
        } else {
            None
        };

        let has_biases = layers.first().map_or(false, |l| l.attention_bq.is_some());
        tracing::info!(
            arch = %arch,
            layers = format!("[{layer_start}..{layer_end})"),
            total = block_count,
            is_first,
            is_last,
            has_qkv_biases = has_biases,
            rope = if use_rope_contiguous { "contiguous" } else { "interleaved" },
            context_length,
            "Loaded split model segment"
        );

        Ok(Self {
            tok_embeddings,
            layers,
            norm,
            output,
            masks: HashMap::new(),
            layer_start,
            layer_end,
            total_layers: block_count,
            hidden_dim: embedding_length,
            device,
            vocabulary,
            tokenizer,
        })
    }

    /// Build a causal mask for the given sequence length.
    fn mask(&mut self, t: usize) -> CandleResult<Tensor> {
        if let Some(mask) = self.masks.get(&t) {
            return Ok(mask.clone());
        }
        let mask: Vec<_> = (0..t)
            .flat_map(|i| (0..t).map(move |j| u8::from(j > i)))
            .collect();
        let mask = Tensor::from_slice(&mask, (t, t), &self.device)?;
        self.masks.insert(t, mask.clone());
        Ok(mask)
    }

    /// Run the forward pass for this segment's layer range.
    ///
    /// - For the first segment: `input` is token IDs (i64 tensor, shape [1, seq_len]).
    ///   We apply the embedding lookup and return hidden states.
    /// - For intermediate segments: `input` is hidden state activations (f32, [1, seq, hidden_dim]).
    /// - For the last segment: returns logits (f32, [vocab_size]) for the last token position.
    /// - For intermediate segments: returns hidden states (f32, [1, seq, hidden_dim]).
    pub fn forward(&mut self, input: &Tensor, index_pos: usize) -> Result<Tensor, SwarmError> {
        let is_first = self.layer_start == 0;
        let is_last = self.layer_end == self.total_layers;

        // Move input to model's device if needed (e.g. CPU → CUDA)
        let input = input
            .to_device(&self.device)
            .map_err(|e| SwarmError::Internal(format!("Device transfer failed: {e}")))?;

        // Determine the hidden state to start from
        let mut layer_in = if is_first {
            // First segment: input is token IDs → apply embedding
            self.tok_embeddings
                .as_ref()
                .ok_or_else(|| SwarmError::Internal("Missing embedding table".into()))?
                .forward(&input)
                .map_err(|e| SwarmError::Internal(format!("Embedding forward failed: {e}")))?
        } else {
            // Non-first segment: input is already hidden states
            input
        };

        // Get seq_len for mask
        let seq_len = layer_in
            .dim(1)
            .map_err(|e| SwarmError::Internal(e.to_string()))?;
        let mask = if seq_len == 1 {
            None
        } else {
            Some(
                self.mask(seq_len)
                    .map_err(|e| SwarmError::Internal(e.to_string()))?,
            )
        };

        // Run through our layers
        for layer in self.layers.iter_mut() {
            let x = layer_in;
            let residual = &x;
            let x = layer
                .attention_norm
                .forward(&x)
                .map_err(|e| SwarmError::Internal(format!("attn_norm: {e}")))?;
            let attn = layer
                .forward_attn(&x, mask.as_ref(), index_pos)
                .map_err(|e| SwarmError::Internal(format!("attn: {e}")))?;
            let x = (attn + residual).map_err(|e| SwarmError::Internal(e.to_string()))?;

            let residual = &x;
            let x = layer
                .ffn_norm
                .forward(&x)
                .map_err(|e| SwarmError::Internal(format!("ffn_norm: {e}")))?;
            let x = layer
                .mlp
                .forward(&x)
                .map_err(|e| SwarmError::Internal(format!("mlp: {e}")))?;
            layer_in = (x + residual).map_err(|e| SwarmError::Internal(e.to_string()))?;
        }

        if is_last {
            // Last segment: apply final norm, extract last token, project to logits
            let norm = self
                .norm
                .as_ref()
                .ok_or_else(|| SwarmError::Internal("Missing final norm".into()))?;
            let output = self
                .output
                .as_ref()
                .ok_or_else(|| SwarmError::Internal("Missing output head".into()))?;

            let x = norm
                .forward(&layer_in)
                .map_err(|e| SwarmError::Internal(format!("final_norm: {e}")))?;
            let x = x
                .i((.., seq_len - 1, ..))
                .map_err(|e| SwarmError::Internal(format!("last_token_select: {e}")))?;
            let logits = output
                .forward(&x)
                .map_err(|e| SwarmError::Internal(format!("output_proj: {e}")))?;
            Ok(logits)
        } else {
            // Intermediate segment: return hidden states for next segment
            Ok(layer_in)
        }
    }

    /// Clear KV-cache (for new generation session).
    pub fn clear_kv_cache(&mut self) {
        for layer in &mut self.layers {
            layer.kv_cache = None;
        }
    }

    /// Return a reference to the loaded vocabulary, if available.
    pub fn vocab(&self) -> Option<&Vec<String>> {
        self.vocabulary.as_ref()
    }

    /// Return a reference to the BPE tokenizer, if available.
    pub fn tokenizer(&self) -> Option<&BpeTokenizer> {
        self.tokenizer.as_ref()
    }
}

/// Serialize a candle Tensor to bytes for network transmission.
/// Format: [4B ndim][4B*ndim shape][4B dtype_tag][data bytes]
pub fn tensor_to_bytes(tensor: &Tensor) -> Result<Vec<u8>, SwarmError> {
    let tensor = tensor
        .to_dtype(DType::F32)
        .map_err(|e| SwarmError::Internal(e.to_string()))?;
    let shape = tensor.shape().dims();
    let data = tensor
        .flatten_all()
        .map_err(|e| SwarmError::Internal(e.to_string()))?
        .to_vec1::<f32>()
        .map_err(|e| SwarmError::Internal(e.to_string()))?;

    let mut bytes = Vec::new();
    // ndim
    bytes.extend_from_slice(&(shape.len() as u32).to_le_bytes());
    // shape
    for &dim in shape {
        bytes.extend_from_slice(&(dim as u32).to_le_bytes());
    }
    // dtype tag (0 = f32)
    bytes.extend_from_slice(&0u32.to_le_bytes());
    // raw f32 data
    for val in &data {
        bytes.extend_from_slice(&val.to_le_bytes());
    }
    Ok(bytes)
}

/// Deserialize bytes back to a candle Tensor.
pub fn bytes_to_tensor(bytes: &[u8]) -> Result<Tensor, SwarmError> {
    if bytes.len() < 4 {
        return Err(SwarmError::Internal("Tensor bytes too short".into()));
    }

    let mut pos = 0;
    let ndim = u32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap()) as usize;
    pos += 4;

    let mut shape = Vec::with_capacity(ndim);
    for _ in 0..ndim {
        let dim = u32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap()) as usize;
        shape.push(dim);
        pos += 4;
    }

    let _dtype_tag = u32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap());
    pos += 4;

    let num_elements: usize = shape.iter().product();
    let mut data = Vec::with_capacity(num_elements);
    for _ in 0..num_elements {
        if pos + 4 > bytes.len() {
            return Err(SwarmError::Internal("Tensor data truncated".into()));
        }
        let val = f32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap());
        data.push(val);
        pos += 4;
    }

    let tensor = Tensor::from_vec(data, shape.as_slice(), &Device::Cpu)
        .map_err(|e| SwarmError::Internal(e.to_string()))?;
    Ok(tensor)
}

/// Sample the next token from logits using temperature and top-p.
pub fn sample_token(logits: &Tensor, temperature: f32, top_p: f32) -> Result<u32, SwarmError> {
    let logits = logits
        .squeeze(0)
        .map_err(|e| SwarmError::Internal(e.to_string()))?;
    let logits = logits
        .to_dtype(DType::F32)
        .map_err(|e| SwarmError::Internal(e.to_string()))?;
    let logits_vec = logits
        .to_vec1::<f32>()
        .map_err(|e| SwarmError::Internal(e.to_string()))?;

    if temperature <= 0.0 {
        // Greedy: argmax
        let (idx, _) = logits_vec
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .ok_or_else(|| SwarmError::Internal("Empty logits".into()))?;
        return Ok(idx as u32);
    }

    // Apply temperature
    let scaled: Vec<f32> = logits_vec.iter().map(|&x| x / temperature).collect();

    // Softmax
    let max_val = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = scaled.iter().map(|&x| (x - max_val).exp()).collect();
    let sum: f32 = exps.iter().sum();
    let probs: Vec<f32> = exps.iter().map(|&x| x / sum).collect();

    // Top-p (nucleus) sampling
    let mut sorted_indices: Vec<usize> = (0..probs.len()).collect();
    sorted_indices.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());

    let mut cumulative = 0.0;
    let mut cutoff_idx = sorted_indices.len();
    for (i, &idx) in sorted_indices.iter().enumerate() {
        cumulative += probs[idx];
        if cumulative >= top_p {
            cutoff_idx = i + 1;
            break;
        }
    }

    // Renormalize over the top-p subset
    let subset = &sorted_indices[..cutoff_idx];
    let subset_sum: f32 = subset.iter().map(|&i| probs[i]).sum();
    let renormed: Vec<f32> = subset.iter().map(|&i| probs[i] / subset_sum).collect();

    // Random sample
    let r: f32 = rand::random();
    let mut cumulative = 0.0;
    for (i, &p) in renormed.iter().enumerate() {
        cumulative += p;
        if r < cumulative {
            return Ok(subset[i] as u32);
        }
    }

    Ok(*subset.last().unwrap_or(&0) as u32)
}

/// Determine which layer range a node should process based on which shard files
/// it has locally. Maps byte-range shards to the GGUF tensor layout to figure out
/// which transformer blocks' weights are fully contained in the local shards.
pub fn compute_local_layer_range(
    meta: &GgufTensorMeta,
    shard_size: u64,
    local_shard_indices: &[u32],
) -> (usize, usize) {
    // Calculate which byte ranges are locally available
    let mut available_ranges: Vec<(u64, u64)> = local_shard_indices
        .iter()
        .map(|&idx| {
            let start = idx as u64 * shard_size;
            let end = start + shard_size;
            (start, end)
        })
        .collect();
    available_ranges.sort_by_key(|r| r.0);

    // Check which layers have ALL their tensors in available byte ranges
    let mut layer_start = meta.block_count;
    let mut layer_end = 0;

    for layer_idx in 0..meta.block_count {
        let prefix = format!("blk.{layer_idx}");
        let tensor_names = [
            format!("{prefix}.attn_q.weight"),
            format!("{prefix}.attn_k.weight"),
            format!("{prefix}.attn_v.weight"),
            format!("{prefix}.attn_output.weight"),
            format!("{prefix}.attn_norm.weight"),
            format!("{prefix}.ffn_norm.weight"),
            format!("{prefix}.ffn_gate.weight"),
            format!("{prefix}.ffn_down.weight"),
            format!("{prefix}.ffn_up.weight"),
        ];

        let all_available = tensor_names.iter().all(|name| {
            if let Some(loc) = meta.tensors.get(name) {
                let tensor_start = meta.tensor_data_offset + loc.offset;
                let tensor_end = tensor_start + loc.size;
                // Check if this tensor is fully within any available range
                available_ranges
                    .iter()
                    .any(|&(rs, re)| tensor_start >= rs && tensor_end <= re)
            } else {
                false
            }
        });

        if all_available {
            if layer_idx < layer_start {
                layer_start = layer_idx;
            }
            if layer_idx + 1 > layer_end {
                layer_end = layer_idx + 1;
            }
        }
    }

    if layer_start >= layer_end {
        // No complete layers available
        (0, 0)
    } else {
        (layer_start, layer_end)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tensor_roundtrip() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = Tensor::from_vec(data.clone(), &[2, 3], &Device::Cpu).unwrap();
        let bytes = tensor_to_bytes(&tensor).unwrap();
        let restored = bytes_to_tensor(&bytes).unwrap();
        let restored_data = restored.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(data, restored_data);
        assert_eq!(restored.shape().dims(), &[2, 3]);
    }

    #[test]
    fn sample_greedy() {
        let logits = Tensor::from_vec(vec![0.1f32, 0.2, 5.0, 0.3], &[1, 4], &Device::Cpu).unwrap();
        let token = sample_token(&logits, 0.0, 1.0).unwrap();
        assert_eq!(token, 2); // index of 5.0
    }

    #[test]
    fn layer_range_computation() {
        // Create a simple metadata with known tensor offsets
        let mut tensors = HashMap::new();
        let tensor_names_per_layer = [
            "attn_q.weight",
            "attn_k.weight",
            "attn_v.weight",
            "attn_output.weight",
            "attn_norm.weight",
            "ffn_norm.weight",
            "ffn_gate.weight",
            "ffn_down.weight",
            "ffn_up.weight",
        ];

        // Each layer's tensors take ~100 bytes, starting at offset 0
        for layer_idx in 0..4 {
            for (i, name) in tensor_names_per_layer.iter().enumerate() {
                let offset = (layer_idx * 900 + i * 100) as u64;
                tensors.insert(
                    format!("blk.{layer_idx}.{name}"),
                    TensorLocation { offset, size: 100 },
                );
            }
        }

        let meta = GgufTensorMeta {
            tensors,
            tensor_data_offset: 0,
            head_count: 8,
            head_count_kv: 8,
            block_count: 4,
            embedding_length: 512,
            rope_dim: 64,
            rope_freq_base: 10000.0,
            rms_norm_eps: 1e-6,
        };

        // Shards of size 1800 bytes: shard 0 covers bytes 0-1800 (layers 0-1)
        // shard 1 covers bytes 1800-3600 (layers 2-3)
        let range = compute_local_layer_range(&meta, 1800, &[0]);
        assert_eq!(range, (0, 2)); // layers 0 and 1

        let range = compute_local_layer_range(&meta, 1800, &[1]);
        assert_eq!(range, (2, 4)); // layers 2 and 3

        let range = compute_local_layer_range(&meta, 1800, &[0, 1]);
        assert_eq!(range, (0, 4)); // all layers
    }
}
