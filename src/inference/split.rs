//! Split inference engine using candle for layer-range execution.
//!
//! This module enables true distributed inference where each node processes
//! only the transformer layers it holds, forwarding hidden-state activations
//! between nodes. Uses candle for direct tensor computation with quantized
//! GGUF weights.

use std::collections::HashMap;
use std::io::{Read as IoRead, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use candle_core::quantized::gguf_file;
use candle_core::quantized::QTensor;
use candle_core::{DType, Device, IndexOp, Result as CandleResult, Tensor};
use candle_nn::kv_cache::KvCache;
use candle_nn::{Embedding, Module};
use candle_transformers::quantized_nn::RmsNorm;

use crate::error::SwarmError;
use crate::model::lora::LoraAdapter;

const DEFAULT_MAX_SEQ_LEN: usize = 4096;

// ── Per-request KV-cache store ──

/// Concurrent per-request KV-cache storage.
///
/// Instead of storing KV-cache inside `LayerWeights` (which couples cache lifetime
/// to the model and prevents concurrent requests), this stores caches externally,
/// keyed by `(model_key, request_id)`.
///
/// Each entry is a `Vec<Option<(Tensor, Tensor)>>` — one `(K, V)` pair per layer.
/// Entries are created lazily on first use and cleaned up when the request completes
/// or after a timeout.
pub struct KvCacheStore {
    /// Per-request KV-cache: (model_key, request_id) → per-layer (K, V) pairs.
    caches: dashmap::DashMap<(String, String), KvCacheEntry>,
    /// TTL for abandoned cache entries.
    ttl: std::time::Duration,
}

pub(crate) struct KvCacheEntry {
    /// Per-layer KV cache. Index corresponds to layer index within the model segment.
    /// Each `KvCache` pre-allocates a buffer and appends new K/V without `Tensor::cat`.
    pub(crate) layers: Vec<Option<KvCache>>,
    /// When this entry was last accessed.
    pub(crate) last_accessed: std::time::Instant,
}

impl KvCacheStore {
    /// Create a new KV-cache store with the given TTL for abandoned entries.
    pub fn new(ttl: std::time::Duration) -> Self {
        Self {
            caches: dashmap::DashMap::new(),
            ttl,
        }
    }

    /// Get or create the KV-cache entry for a request. Returns a mutable ref guard.
    pub(crate) fn get_or_create(
        &self,
        model_key: &str,
        request_id: &str,
        num_layers: usize,
    ) -> dashmap::mapref::one::RefMut<'_, (String, String), KvCacheEntry> {
        let key = (model_key.to_string(), request_id.to_string());
        self.caches.entry(key).or_insert_with(|| KvCacheEntry {
            layers: vec![None; num_layers],
            last_accessed: std::time::Instant::now(),
        })
    }

    /// Clear (remove) the KV-cache for a specific request.
    pub fn clear_request(&self, model_key: &str, request_id: &str) {
        let key = (model_key.to_string(), request_id.to_string());
        self.caches.remove(&key);
    }

    /// Clean up all expired cache entries. Returns the number of entries removed.
    pub fn cleanup_expired(&self) -> usize {
        let ttl = self.ttl;
        let before = self.caches.len();
        self.caches
            .retain(|_, entry| entry.last_accessed.elapsed() <= ttl);
        before - self.caches.len()
    }

    /// Remove all cache entries for a given request_id (across all models).
    pub fn cleanup_request_id(&self, request_id: &str) {
        self.caches
            .retain(|(_model_key, req_id), _| req_id != request_id);
    }

    /// Get the number of active cache entries.
    pub fn active_entries(&self) -> usize {
        self.caches.len()
    }
}

// ── Split model entry with LRU tracking ──

/// Key type for split_models DashMap: (model_id, layer_start, layer_end).
pub type SplitModelKey = (crate::types::ModelId, usize, usize);

/// Wrapper around a SplitModel that tracks last-used time for LRU eviction.
pub struct SplitModelEntry {
    pub model: std::sync::Arc<tokio::sync::Mutex<SplitModel>>,
    pub last_used: std::sync::atomic::AtomicU64,
    /// Estimated VRAM usage in MB for this model segment.
    pub estimated_vram_mb: u64,
    /// Optional batch forwarder for this model segment.
    pub batch_forwarder: Option<std::sync::Arc<BatchForwarder>>,
}

impl SplitModelEntry {
    /// Create a new entry wrapping a split model.
    pub fn new(model: SplitModel) -> Self {
        let estimated_vram_mb = model.estimate_vram_mb();
        Self {
            model: std::sync::Arc::new(tokio::sync::Mutex::new(model)),
            last_used: std::sync::atomic::AtomicU64::new(
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs(),
            ),
            estimated_vram_mb,
            batch_forwarder: None,
        }
    }

    /// Create a new entry with batching enabled.
    pub fn new_with_batching(
        model: SplitModel,
        kv_cache_store: std::sync::Arc<KvCacheStore>,
        max_batch_size: usize,
    ) -> Self {
        let estimated_vram_mb = model.estimate_vram_mb();
        let model_arc = std::sync::Arc::new(tokio::sync::Mutex::new(model));
        let batch_forwarder = if max_batch_size > 1 {
            Some(std::sync::Arc::new(BatchForwarder::new(
                model_arc.clone(),
                kv_cache_store,
                max_batch_size,
            )))
        } else {
            None
        };
        Self {
            model: model_arc,
            last_used: std::sync::atomic::AtomicU64::new(
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs(),
            ),
            estimated_vram_mb,
            batch_forwarder,
        }
    }

    /// Touch this entry to update its last-used time.
    pub fn touch(&self) {
        self.last_used.store(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            std::sync::atomic::Ordering::Relaxed,
        );
    }

    /// Get the last-used timestamp in seconds since epoch.
    pub fn last_used_secs(&self) -> u64 {
        self.last_used.load(std::sync::atomic::Ordering::Relaxed)
    }
}

/// A single item in a batched forward pass.
pub struct BatchItem<'a> {
    /// Input tensor for this request.
    pub input: &'a Tensor,
    /// Sequence position for RoPE and KV-cache.
    pub index_pos: usize,
    /// Request ID for per-request KV-cache isolation.
    pub request_id: &'a str,
}

/// Evict least-recently-used split models from the cache until total estimated
/// VRAM usage is under `budget_mb`. Models that have active requests (present
/// in `active_pipelines`) are never evicted.
///
/// Returns the number of models evicted.
pub fn evict_split_models_lru(
    split_models: &dashmap::DashMap<SplitModelKey, SplitModelEntry>,
    active_pipelines: &dashmap::DashMap<uuid::Uuid, crate::types::PipelineAssignment>,
    budget_mb: u64,
    needed_mb: u64,
) -> usize {
    let mut total_mb: u64 = split_models
        .iter()
        .map(|e| e.value().estimated_vram_mb)
        .sum();

    if total_mb + needed_mb <= budget_mb {
        return 0;
    }

    // Collect all active model_ids from active pipelines to protect them
    let active_model_ids: std::collections::HashSet<crate::types::ModelId> = {
        let mut ids = std::collections::HashSet::new();
        for entry in active_pipelines.iter() {
            for seg in &entry.value().segments {
                ids.insert(seg.shard_id.model_id.clone());
            }
        }
        ids
    };

    // Collect eviction candidates sorted by last_used (ascending = LRU first)
    let mut candidates: Vec<(SplitModelKey, u64, u64)> = split_models
        .iter()
        .filter(|e| !active_model_ids.contains(&e.key().0))
        .map(|e| {
            let key = e.key().clone();
            let last_used = e.value().last_used_secs();
            let vram = e.value().estimated_vram_mb;
            (key, last_used, vram)
        })
        .collect();

    candidates.sort_by_key(|(_key, last_used, _vram)| *last_used);

    let mut evicted = 0;
    for (key, _last_used, vram) in candidates {
        if total_mb + needed_mb <= budget_mb {
            break;
        }
        if split_models.remove(&key).is_some() {
            tracing::info!(
                model = %key.0,
                layers = format!("{}-{}", key.1, key.2),
                vram_mb = vram,
                "Evicted LRU split model to free VRAM"
            );
            total_mb = total_mb.saturating_sub(vram);
            evicted += 1;
        }
    }

    evicted
}

// ── Batch forwarder ──

/// A pending forward request waiting to be batched.
struct PendingForward {
    input: Tensor,
    index_pos: usize,
    request_id: String,
    result_tx: tokio::sync::oneshot::Sender<Result<Tensor, SwarmError>>,
}

/// Collects concurrent forward requests for the same model and processes them
/// as a single batched forward pass.  Each `BatchForwarder` is tied to one
/// `SplitModelEntry` (i.e. one loaded model segment).
///
/// Callers submit work via `submit()` which returns a oneshot receiver.
/// A background drain loop (or the first waiter) acquires the model lock,
/// collects all pending items, and calls `forward_batch`.
pub struct BatchForwarder {
    queue: tokio::sync::Mutex<Vec<PendingForward>>,
    notify: tokio::sync::Notify,
    model: std::sync::Arc<tokio::sync::Mutex<SplitModel>>,
    kv_cache_store: std::sync::Arc<KvCacheStore>,
    /// Maximum batch size (from config). 1 = no batching.
    max_batch_size: usize,
}

impl BatchForwarder {
    /// Create a new batch forwarder for a split model.
    pub fn new(
        model: std::sync::Arc<tokio::sync::Mutex<SplitModel>>,
        kv_cache_store: std::sync::Arc<KvCacheStore>,
        max_batch_size: usize,
    ) -> Self {
        Self {
            queue: tokio::sync::Mutex::new(Vec::new()),
            notify: tokio::sync::Notify::new(),
            model,
            kv_cache_store,
            max_batch_size: max_batch_size.max(1),
        }
    }

    /// Submit a forward request and wait for the result.
    ///
    /// The request will be batched with other concurrent requests for the same
    /// model.  Returns the output tensor for this request.
    pub async fn submit(
        &self,
        input: Tensor,
        index_pos: usize,
        request_id: String,
    ) -> Result<Tensor, SwarmError> {
        let (tx, rx) = tokio::sync::oneshot::channel();
        {
            let mut q = self.queue.lock().await;
            q.push(PendingForward {
                input,
                index_pos,
                request_id,
                result_tx: tx,
            });
        }
        self.notify.notify_one();

        // Try to become the batch processor.  If we can acquire the model lock,
        // drain the queue and process the batch.  If another task already holds
        // the lock (processing a previous batch), we just wait on our oneshot.
        if let Ok(mut model_guard) = self.model.try_lock() {
            self.drain_and_process(&mut model_guard).await;
        }

        rx.await
            .map_err(|_| SwarmError::Internal("Batch forwarder dropped".into()))?
    }

    /// Drain the pending queue and run a batched forward pass.
    async fn drain_and_process(&self, model: &mut SplitModel) {
        let pending: Vec<PendingForward> = {
            let mut q = self.queue.lock().await;
            if q.is_empty() {
                return;
            }
            let take = q.len().min(self.max_batch_size);
            q.drain(..take).collect()
        };

        if pending.is_empty() {
            return;
        }

        let items: Vec<BatchItem<'_>> = pending
            .iter()
            .map(|p| BatchItem {
                input: &p.input,
                index_pos: p.index_pos,
                request_id: &p.request_id,
            })
            .collect();

        let results = model.forward_batch(&items, &self.kv_cache_store);

        match results {
            Ok(outputs) => {
                for (pending_item, output) in pending.into_iter().zip(outputs) {
                    let _ = pending_item.result_tx.send(Ok(output));
                }
            }
            Err(e) => {
                let msg = e.to_string();
                for pending_item in pending {
                    let _ = pending_item
                        .result_tx
                        .send(Err(SwarmError::Internal(msg.clone())));
                }
            }
        }
    }
}

// ── BPE Tokenizer from GGUF merges ──

/// BPE tokenizer built from GGUF metadata.
/// Supports both GPT-2/Qwen2 byte-level BPE and SentencePiece BPE (LLaMA).
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
    /// Whether this is a SentencePiece tokenizer (uses ▁ for spaces, no byte encoding)
    is_sentencepiece: bool,
}

impl BpeTokenizer {
    /// Build a BPE tokenizer from GGUF vocabulary tokens, merge rules,
    /// pre-tokenizer type, and tokenizer model type.
    fn from_gguf(
        tokens: &[String],
        merges_raw: &[String],
        pre_type: &str,
        tokenizer_model: &str,
    ) -> Self {
        let is_sentencepiece = tokenizer_model == "llama";
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

        // Collect special tokens (e.g., <|im_start|>, <|im_end|>, <s>, </s>, <unk>)
        let mut special_tokens: Vec<(String, u32)> = token_to_id
            .iter()
            .filter(|(t, _)| {
                (t.starts_with("<|") && t.ends_with("|>"))
                    || *t == "<s>"
                    || *t == "</s>"
                    || *t == "<unk>"
                    || *t == "<pad>"
            })
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
            is_sentencepiece,
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
            } else if self.is_sentencepiece {
                // SentencePiece: replace spaces with ▁, then BPE encode
                // SentencePiece convention: leading space becomes ▁
                let normalized = format!("\u{2581}{}", segment.replace(' ', "\u{2581}"));
                all_ids.extend(self.bpe_encode_word(&normalized));
            } else {
                // GPT-2: pre-tokenize with regex, then BPE encode each piece
                let pre_tokens = self.pre_tokenize(segment);
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
    /// For GPT-2: converts bytes → GPT-2 unicode chars, then applies BPE merges.
    /// For SentencePiece: uses raw unicode chars directly with ▁ for leading spaces.
    fn bpe_encode_word(&self, word: &str) -> Vec<i64> {
        let chars: Vec<String> = if self.is_sentencepiece {
            // SentencePiece: each character is used as-is (▁ already inserted by pre_tokenize)
            word.chars().map(|c| c.to_string()).collect()
        } else {
            // GPT-2: convert each byte to its GPT-2 unicode character
            word.bytes()
                .map(|b| self.byte_encoder[b as usize].to_string())
                .collect()
        };

        if chars.is_empty() {
            return vec![];
        }

        // Single char: direct lookup
        if chars.len() == 1 {
            return vec![self.token_to_id.get(&chars[0]).copied().unwrap_or(0) as i64];
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
    /// For GPT-2: reverses the GPT-2 unicode byte encoding.
    /// For SentencePiece: converts ▁ back to space, handles <0xNN> byte tokens.
    pub fn decode_token(&self, token_str: &str) -> Vec<u8> {
        if self.is_sentencepiece {
            // Handle byte fallback tokens like <0x0A> (newline)
            if token_str.starts_with("<0x") && token_str.ends_with('>') && token_str.len() == 6 {
                if let Ok(byte) = u8::from_str_radix(&token_str[3..5], 16) {
                    return vec![byte];
                }
            }
            // Special tokens like <s>, </s>, <unk> → empty (don't emit)
            if token_str.starts_with('<') && token_str.ends_with('>') {
                return vec![];
            }
            // SentencePiece: ▁ (U+2581) → space, everything else is raw UTF-8
            token_str.replace('\u{2581}', " ").into_bytes()
        } else {
            // GPT-2: reverse byte encoding
            token_str
                .chars()
                .map(|ch| self.byte_decoder.get(&ch).copied().unwrap_or(b'?'))
                .collect()
        }
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
        let is_printable =
            (33..=126).contains(&b) || (161..=172).contains(&b) || (174..=255).contains(&b);
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

// ── Model architecture detection ──

/// Known model architectures from GGUF `general.architecture` metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelArch {
    /// Llama family (Llama 1/2/3, CodeLlama, Yi, Mistral 7B)
    Llama,
    /// Qwen2 / Qwen2.5 / Qwen3
    Qwen2,
    /// Google Gemma 1
    Gemma,
    /// Google Gemma 2 — different RmsNorm (+1), Gelu activation, attention logit soft-capping
    Gemma2,
    /// Microsoft Phi-3/3.5 — SuRoPE scaling, partial rotary embedding
    Phi3,
    /// Mistral (when explicitly tagged, most Mistral GGUFs use "llama" arch)
    Mistral,
    /// StarCoder2
    Starcoder2,
    /// DeepSeek-V2/V3 — MoE + MLA, NOT supported for split inference
    DeepSeek2,
    /// Architecture not recognized — falls back to Llama-like behavior
    Unknown(String),
}

impl ModelArch {
    /// Detect architecture from GGUF `general.architecture` metadata string.
    pub fn from_gguf_arch(arch: &str) -> Self {
        match arch {
            "llama" => ModelArch::Llama,
            "qwen2" | "qwen3" => ModelArch::Qwen2,
            "gemma" => ModelArch::Gemma,
            "gemma2" => ModelArch::Gemma2,
            "phi3" => ModelArch::Phi3,
            "mistral" => ModelArch::Mistral,
            "starcoder2" => ModelArch::Starcoder2,
            "deepseek2" => ModelArch::DeepSeek2,
            other => ModelArch::Unknown(other.to_string()),
        }
    }

    /// Whether this architecture uses contiguous RoPE (vs interleaved).
    pub fn use_rope_contiguous(&self) -> bool {
        matches!(self, ModelArch::Qwen2)
    }

    /// Default activation function for this architecture's MLP.
    fn default_activation(&self) -> Activation {
        match self {
            ModelArch::Gemma | ModelArch::Gemma2 => Activation::Gelu,
            _ => Activation::SiLU,
        }
    }

    /// Whether this architecture uses the Gemma-style RmsNorm (adds 1 to weights).
    pub fn use_gemma_norm(&self) -> bool {
        matches!(self, ModelArch::Gemma | ModelArch::Gemma2)
    }

    /// Whether this architecture is supported for split inference.
    pub fn is_supported(&self) -> bool {
        !matches!(self, ModelArch::DeepSeek2)
    }
}

impl std::fmt::Display for ModelArch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelArch::Llama => write!(f, "llama"),
            ModelArch::Qwen2 => write!(f, "qwen2"),
            ModelArch::Gemma => write!(f, "gemma"),
            ModelArch::Gemma2 => write!(f, "gemma2"),
            ModelArch::Phi3 => write!(f, "phi3"),
            ModelArch::Mistral => write!(f, "mistral"),
            ModelArch::Starcoder2 => write!(f, "starcoder2"),
            ModelArch::DeepSeek2 => write!(f, "deepseek2"),
            ModelArch::Unknown(s) => write!(f, "{s}"),
        }
    }
}

/// Activation function used in the MLP/FFN.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Activation {
    /// SiLU / Swish — used by Llama, Qwen2, Mistral, Phi-3
    SiLU,
    /// Gelu — used by Gemma, Gemma 2
    Gelu,
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
    activation: Activation,
}

impl Mlp {
    fn forward(&self, xs: &Tensor, lora: Option<(&LoraAdapter, usize)>) -> CandleResult<Tensor> {
        let mut gate = self.ffn_gate.forward(xs)?;
        let mut up = self.ffn_up.forward(xs)?;

        if let Some((adapter, abs_layer)) = lora {
            let key_gate = format!("blk.{abs_layer}.ffn_gate");
            if let Some(lw) = adapter.weights.get(&key_gate) {
                gate = crate::model::lora::apply_lora(
                    &gate,
                    xs,
                    lw,
                    adapter.metadata.alpha,
                    adapter.metadata.rank,
                )
                .map_err(|e| candle_core::Error::Msg(format!("LoRA ffn_gate: {e}")))?;
            }
            let key_up = format!("blk.{abs_layer}.ffn_up");
            if let Some(lw) = adapter.weights.get(&key_up) {
                up = crate::model::lora::apply_lora(
                    &up,
                    xs,
                    lw,
                    adapter.metadata.alpha,
                    adapter.metadata.rank,
                )
                .map_err(|e| candle_core::Error::Msg(format!("LoRA ffn_up: {e}")))?;
            }
        }

        let activated = match self.activation {
            Activation::SiLU => candle_nn::ops::silu(&gate)?,
            Activation::Gelu => gate.gelu()?,
        };
        let combined = (activated * up)?;

        let mut down = self.ffn_down.forward(&combined)?;
        if let Some((adapter, abs_layer)) = lora {
            let key_down = format!("blk.{abs_layer}.ffn_down");
            if let Some(lw) = adapter.weights.get(&key_down) {
                down = crate::model::lora::apply_lora(
                    &down,
                    &combined,
                    lw,
                    adapter.metadata.alpha,
                    adapter.metadata.rank,
                )
                .map_err(|e| candle_core::Error::Msg(format!("LoRA ffn_down: {e}")))?;
            }
        }
        Ok(down)
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
    /// If true, use contiguous RoPE (rope); if false, use interleaved (rope_i).
    use_rope_contiguous: bool,
    /// Gemma 2 attention logit soft-capping: `tanh(logits / cap) * cap` before softmax.
    attn_logit_softcap: Option<f32>,
}

fn masked_fill(on_false: &Tensor, mask: &Tensor, on_true: &Tensor) -> CandleResult<Tensor> {
    let shape = mask.shape();
    mask.where_cond(&on_true.broadcast_as(shape.dims())?, on_false)
}

/// Standard O(n^2) matmul attention with optional causal mask.
/// Input/output layout: BHSD `(b, n_head, seq, head_dim)`.
/// Supports optional Gemma 2 attention logit soft-capping.
#[allow(clippy::too_many_arguments)]
fn standard_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    head_dim: usize,
    n_head: usize,
    n_kv_head: usize,
    neg_inf: &Tensor,
    attn_logit_softcap: Option<f32>,
) -> CandleResult<Tensor> {
    let k = candle_transformers::utils::repeat_kv(k.clone(), n_head / n_kv_head)?;
    let v = candle_transformers::utils::repeat_kv(v.clone(), n_head / n_kv_head)?;

    let att = (q.matmul(&k.t()?)? / (head_dim as f64).sqrt())?;
    // Gemma 2 attention logit soft-capping: tanh(logits / cap) * cap
    let att = if let Some(cap) = attn_logit_softcap {
        let cap_f64 = cap as f64;
        ((att / cap_f64)?.tanh()? * cap_f64)?
    } else {
        att
    };
    let att = match mask {
        None => att,
        Some(mask) => {
            let mask = mask.broadcast_as(att.shape())?;
            masked_fill(&att, &mask, neg_inf)?
        }
    };
    let att = candle_nn::ops::softmax_last_dim(&att)?;
    att.matmul(&v.contiguous()?)
}

/// Unified attention dispatch: selects the best backend for the device.
///
/// - **CPU**: Uses `candle_nn::cpu_flash_attention::run_flash_attn_cpu::<f32>()` which
///   handles GQA natively (no `repeat_kv` needed) and uses O(1) memory via tiled online softmax.
/// - **GPU with `flash-attn` feature**: Uses `candle_flash_attn::flash_attn()` with F16 —
///   fused CUDA kernel, GQA native, O(1) memory. Requires SM80+ (RTX 3070 SM86 compatible).
/// - **GPU without `flash-attn` / fallback**: Standard matmul path with `repeat_kv`.
///
/// All paths produce output in BHSD layout `(b, n_head, seq, head_dim)`.
#[allow(clippy::too_many_arguments)]
fn run_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    neg_inf: &Tensor,
    attn_logit_softcap: Option<f32>,
) -> CandleResult<Tensor> {
    match q.device() {
        Device::Cpu => {
            // CPU flash attention: input BSHD, output BHSD
            // Transpose Q/K/V from BHSD (b,h,s,d) to BSHD (b,s,h,d)
            let q_bshd = q.transpose(1, 2)?.contiguous()?;
            let k_bshd = k.transpose(1, 2)?.contiguous()?;
            let v_bshd = v.transpose(1, 2)?.contiguous()?;

            let softmax_scale = 1.0 / (head_dim as f32).sqrt();
            let seq_len = q.dim(2)?;

            // Build float additive mask for prefill (decode has seq_len==1, no mask needed)
            let flash_mask = if seq_len > 1 {
                if let Some(u8_mask) = mask {
                    // Convert u8 causal mask (1=masked, 0=visible) to float (NEG_INF=masked, 0=visible)
                    let zeros = u8_mask.zeros_like()?.to_dtype(DType::F32)?;
                    let neg_inf_scalar = Tensor::new(f32::NEG_INFINITY, u8_mask.device())?;
                    let float_mask = u8_mask.where_cond(
                        &neg_inf_scalar.broadcast_as(u8_mask.shape().dims())?,
                        &zeros,
                    )?;
                    Some(float_mask)
                } else {
                    None
                }
            } else {
                None
            };

            // run_flash_attn_cpu handles GQA natively — no repeat_kv needed
            // Output shape: (b, n_head, seq, head_dim) — BHSD
            let out = candle_nn::cpu_flash_attention::run_flash_attn_cpu::<f32>(
                &q_bshd,
                &k_bshd,
                &v_bshd,
                flash_mask.as_ref(),
                softmax_scale,
                None, // no ALiBi
                attn_logit_softcap,
            )?;
            Ok(out)
        }
        #[cfg(feature = "flash-attn")]
        Device::Cuda(_) => {
            let q_len = q.dim(2)?;
            let k_len = k.dim(2)?;

            // Flash attention only supports simple causal masking. When the KV cache
            // has pre-populated prefix entries (k_len > q_len with q_len > 1), the
            // offset causal mask can't be expressed via flash_attn's boolean causal
            // flag. Fall back to standard matmul attention with the explicit mask.
            if k_len > q_len && q_len > 1 {
                return standard_attention(
                    q,
                    k,
                    v,
                    mask,
                    head_dim,
                    n_head,
                    n_kv_head,
                    neg_inf,
                    attn_logit_softcap,
                );
            }

            // GPU flash attention: input BSHD, output BSHD → transpose to BHSD
            let q_bshd = q.transpose(1, 2)?.contiguous()?;
            let k_bshd = k.transpose(1, 2)?.contiguous()?;
            let v_bshd = v.transpose(1, 2)?.contiguous()?;

            // Flash attention requires F16
            let q_f16 = q_bshd.to_dtype(DType::F16)?;
            let k_f16 = k_bshd.to_dtype(DType::F16)?;
            let v_f16 = v_bshd.to_dtype(DType::F16)?;

            let softmax_scale = 1.0 / (head_dim as f32).sqrt();

            // flash_attn handles GQA and causal masking natively
            let out =
                candle_flash_attn::flash_attn(&q_f16, &k_f16, &v_f16, softmax_scale, q_len > 1)?;

            // Output is BSHD — transpose to BHSD and cast back to F32
            out.to_dtype(DType::F32)?.transpose(1, 2)?.contiguous()
        }
        _ => {
            // Fallback: standard matmul attention with optional soft-capping
            standard_attention(
                q,
                k,
                v,
                mask,
                head_dim,
                n_head,
                n_kv_head,
                neg_inf,
                attn_logit_softcap,
            )
        }
    }
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
        &self,
        x: &Tensor,
        mask: Option<&Tensor>,
        index_pos: usize,
        kv_cache: &mut Option<KvCache>,
        max_seq_len: usize,
        lora: Option<(&LoraAdapter, usize)>,
    ) -> CandleResult<Tensor> {
        let (b_sz, seq_len, n_embd) = x.dims3()?;
        let mut q = self.attention_wq.forward(x)?;
        let mut k = self.attention_wk.forward(x)?;
        let mut v = self.attention_wv.forward(x)?;

        // Apply LoRA deltas to Q/K/V projections if adapter is active
        if let Some((adapter, abs_layer)) = lora {
            let key_q = format!("blk.{abs_layer}.attn_q");
            if let Some(lw) = adapter.weights.get(&key_q) {
                q = crate::model::lora::apply_lora(
                    &q,
                    x,
                    lw,
                    adapter.metadata.alpha,
                    adapter.metadata.rank,
                )
                .map_err(|e| candle_core::Error::Msg(format!("LoRA attn_q: {e}")))?;
            }
            let key_k = format!("blk.{abs_layer}.attn_k");
            if let Some(lw) = adapter.weights.get(&key_k) {
                k = crate::model::lora::apply_lora(
                    &k,
                    x,
                    lw,
                    adapter.metadata.alpha,
                    adapter.metadata.rank,
                )
                .map_err(|e| candle_core::Error::Msg(format!("LoRA attn_k: {e}")))?;
            }
            let key_v = format!("blk.{abs_layer}.attn_v");
            if let Some(lw) = adapter.weights.get(&key_v) {
                v = crate::model::lora::apply_lora(
                    &v,
                    x,
                    lw,
                    adapter.metadata.alpha,
                    adapter.metadata.rank,
                )
                .map_err(|e| candle_core::Error::Msg(format!("LoRA attn_v: {e}")))?;
            }
        }

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

        // KV-cache: use pre-allocated KvCache buffers (avoids Tensor::cat per step)
        let (k, v) = match kv_cache {
            None => {
                let mut cache = KvCache::new(2, max_seq_len);
                let kv = cache.append(&k, &v)?;
                *kv_cache = Some(cache);
                kv
            }
            Some(cache) => {
                if index_pos == 0 {
                    cache.reset();
                }
                cache.append(&k, &v)?
            }
        };

        // Unified attention dispatch: flash (CPU/GPU) or standard matmul fallback.
        // All backends return BHSD (b, n_head, seq, head_dim).
        let y = run_attention(
            &q,
            &k,
            &v,
            mask,
            self.n_head,
            self.n_kv_head,
            self.head_dim,
            &self.neg_inf,
            self.attn_logit_softcap,
        )?;

        let y = y.transpose(1, 2)?.reshape(&[b_sz, seq_len, n_embd])?;
        let mut wo_out = self.attention_wo.forward(&y)?;

        // Apply LoRA delta to O projection
        if let Some((adapter, abs_layer)) = lora {
            let key_o = format!("blk.{abs_layer}.attn_output");
            if let Some(lw) = adapter.weights.get(&key_o) {
                wo_out = crate::model::lora::apply_lora(
                    &wo_out,
                    &y,
                    lw,
                    adapter.metadata.alpha,
                    adapter.metadata.rank,
                )
                .map_err(|e| candle_core::Error::Msg(format!("LoRA attn_output: {e}")))?;
            }
        }
        Ok(wo_out)
    }
}

// ── Split model: loads only a range of layers from a GGUF ──

/// A partial transformer model that loads and runs only a specific range of layers.
/// Used for split inference where each node holds different layers.
/// Supports multiple architectures: Llama, Qwen2, Gemma 2, Phi-3, Mistral.
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
    /// Detected model architecture.
    pub arch: ModelArch,
    /// Device (CPU or CUDA).
    device: Device,
    /// Vocabulary from GGUF (token ID → string), for decoding generated tokens.
    vocabulary: Option<Vec<String>>,
    /// BPE tokenizer built from GGUF merges table.
    tokenizer: Option<BpeTokenizer>,
    /// EOS token IDs loaded from GGUF metadata.
    eos_tokens: Vec<u32>,
    /// Chat template from GGUF `tokenizer.chat_template` (Jinja2 format).
    chat_template: Option<String>,
    /// BOS token string from GGUF metadata.
    bos_token: String,
    /// EOS token string from GGUF metadata.
    eos_token: String,
    /// Maximum sequence length for KV cache pre-allocation.
    max_seq_len: usize,
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

        // Extract friendly model name from GGUF metadata
        let model_name = ct
            .metadata
            .get("general.name")
            .and_then(|v| v.to_string().ok().cloned());

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
            model_name,
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
    let mut header_buf = vec![0u8; header_size];
    file.seek(SeekFrom::Start(0)).map_err(SwarmError::Io)?;
    file.read_exact(&mut header_buf).map_err(SwarmError::Io)?;

    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent).map_err(SwarmError::Io)?;
    }
    std::fs::write(output_path, &header_buf).map_err(SwarmError::Io)?;

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
    let header_path = model_dir.join("gguf_header.bin");
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

    // Try source_path as a fallback
    let source_path_file = model_dir.join("source_path");
    if source_path_file.exists() {
        if let Ok(path_str) = std::fs::read_to_string(&source_path_file) {
            let gguf_path = Path::new(path_str.trim());
            if gguf_path.exists() {
                tracing::info!(
                    gguf = %gguf_path.display(),
                    "Extracting GGUF header from source path"
                );
                return save_gguf_header(gguf_path, &header_path);
            }
        }
    }

    Err(SwarmError::Internal(format!(
        "Cannot create gguf_header.bin: no shard_000.bin or source GGUF found in {}",
        model_dir.display()
    )))
}

// ── ShardReader: virtual GGUF file from header + shard files ──

/// One tensor's mapping from virtual GGUF position to a shard file.
struct TensorMapEntry {
    /// Absolute byte offset in the virtual GGUF file.
    gguf_offset: u64,
    /// Index into the `shards` vec.
    shard_idx: usize,
    /// Byte offset within the shard file where this tensor's data starts.
    shard_local_offset: u64,
    /// Size of this tensor's data in bytes.
    size: u64,
}

/// Metadata for one shard file.
struct ShardFile {
    _index: u32,
    path: PathBuf,
    file_len: u64,
}

/// A reader that presents a GGUF header + v2 layer-aligned shard files as a
/// single contiguous seekable file.  This allows candle's `Content::read()`
/// and `ct.tensor()` to work transparently over shard files.
///
/// V2 shards contain packed tensor data (not byte-range slices of the GGUF).
/// The `tensor_map` translates virtual GGUF offsets → (shard_idx, shard_local_offset)
/// via binary search.
pub struct ShardReader {
    /// Raw GGUF header bytes (metadata + tensor info table), padded to tensor_data_offset.
    header: Vec<u8>,
    /// Shard files in order by index.
    shards: Vec<ShardFile>,
    /// Sorted by `gguf_offset` for binary search.
    tensor_map: Vec<TensorMapEntry>,
    /// Total size of the virtual GGUF file (header + all tensor data).
    total_size: u64,
    /// Current seek position in the virtual file.
    position: u64,
    /// Currently open shard file handle (cached to avoid repeated opens).
    current_shard: Option<(usize, std::fs::File)>,
}

impl ShardReader {
    /// Create a ShardReader from a GGUF header and v2 shard files with tensor maps.
    ///
    /// `shard_files` must be ordered by shard index.  Each shard's tensor entries
    /// describe which virtual-GGUF-offset ranges map to which shard-local offsets.
    pub fn new(
        header_path: &Path,
        shard_files: Vec<(u32, PathBuf)>,
        tensor_entries: &[Vec<crate::types::ShardTensorEntry>],
        total_gguf_size: u64,
        tensor_data_offset: u64,
    ) -> Result<Self, SwarmError> {
        let header = std::fs::read(header_path).map_err(SwarmError::Io)?;
        let header = if (header.len() as u64) < tensor_data_offset {
            let mut padded = header;
            padded.resize(tensor_data_offset as usize, 0);
            padded
        } else {
            header
        };

        let mut shards = Vec::with_capacity(shard_files.len());
        let mut tensor_map = Vec::new();

        for (i, (idx, path)) in shard_files.iter().enumerate() {
            let file_len = std::fs::metadata(path).map_err(SwarmError::Io)?.len();
            shards.push(ShardFile {
                _index: *idx,
                path: path.clone(),
                file_len,
            });

            // Build tensor map entries from the corresponding tensor_entries
            if let Some(entries) = tensor_entries.get(i) {
                for te in entries {
                    tensor_map.push(TensorMapEntry {
                        gguf_offset: te.gguf_offset,
                        shard_idx: i,
                        shard_local_offset: te.shard_offset,
                        size: te.size,
                    });
                }
            }
        }

        // Sort by gguf_offset for binary search
        tensor_map.sort_by_key(|e| e.gguf_offset);

        Ok(Self {
            header,
            shards,
            tensor_map,
            total_size: total_gguf_size,
            position: 0,
            current_shard: None,
        })
    }

    /// Find which shard (if any) contains the given virtual file position,
    /// returning (shard_vec_index, offset_within_shard_file).
    fn find_shard(&self, pos: u64) -> Option<(usize, u64)> {
        // Binary search: find the last entry where gguf_offset <= pos
        let idx = match self
            .tensor_map
            .binary_search_by_key(&pos, |e| e.gguf_offset)
        {
            Ok(i) => i,
            Err(0) => return None,
            Err(i) => i - 1,
        };

        let entry = &self.tensor_map[idx];
        if pos < entry.gguf_offset + entry.size {
            let delta = pos - entry.gguf_offset;
            Some((entry.shard_idx, entry.shard_local_offset + delta))
        } else {
            None
        }
    }
}

impl IoRead for ShardReader {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        if self.position >= self.total_size {
            return Ok(0);
        }

        let header_len = self.header.len() as u64;

        // Reading from header region
        if self.position < header_len {
            let start = self.position as usize;
            let available = (header_len - self.position) as usize;
            let to_read = buf.len().min(available);
            buf[..to_read].copy_from_slice(&self.header[start..start + to_read]);
            self.position += to_read as u64;
            return Ok(to_read);
        }

        // Reading from shard region via tensor map
        if let Some((shard_idx, offset_in_shard)) = self.find_shard(self.position) {
            if tracing::enabled!(tracing::Level::TRACE) {
                tracing::trace!(
                    pos = self.position,
                    shard_idx,
                    offset_in_shard,
                    buf_len = buf.len(),
                    "ShardReader::read"
                );
            }
            // Open the shard file if not already open
            let need_open = match &self.current_shard {
                Some((idx, _)) => *idx != shard_idx,
                None => true,
            };
            if need_open {
                let file = std::fs::File::open(&self.shards[shard_idx].path)
                    .map_err(std::io::Error::other)?;
                self.current_shard = Some((shard_idx, file));
            }

            let shard_file_len = self.shards[shard_idx].file_len;
            let (_, ref mut file) = self.current_shard.as_mut().unwrap();
            file.seek(SeekFrom::Start(offset_in_shard))?;
            let available_in_shard = shard_file_len.saturating_sub(offset_in_shard) as usize;
            let to_read = buf.len().min(available_in_shard);
            if to_read == 0 {
                tracing::error!(
                    pos = self.position,
                    shard_idx,
                    offset_in_shard,
                    shard_file_len,
                    buf_len = buf.len(),
                    "ShardReader: 0 bytes available at offset in shard"
                );
                return Ok(0);
            }
            let n = file.read(&mut buf[..to_read])?;
            self.position += n as u64;
            Ok(n)
        } else {
            // Position is in a gap (missing tensor / missing shard)
            let map_info: Vec<String> = self
                .tensor_map
                .iter()
                .take(5)
                .map(|e| {
                    format!(
                        "shard[{}]@gguf[{}..{})",
                        e.shard_idx,
                        e.gguf_offset,
                        e.gguf_offset + e.size
                    )
                })
                .collect();
            tracing::error!(
                pos = self.position,
                total_size = self.total_size,
                header_len = self.header.len(),
                buf_len = buf.len(),
                tensor_map_sample = ?map_info,
                "ShardReader: position is in a missing shard region"
            );
            Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                format!(
                    "ShardReader: position {} is in a missing region (total_size={})",
                    self.position, self.total_size
                ),
            ))
        }
    }
}

impl Seek for ShardReader {
    fn seek(&mut self, pos: SeekFrom) -> std::io::Result<u64> {
        let new_pos = match pos {
            SeekFrom::Start(p) => p as i64,
            SeekFrom::End(p) => self.total_size as i64 + p,
            SeekFrom::Current(p) => self.position as i64 + p,
        };
        if new_pos < 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "Seek before start",
            ));
        }
        self.position = new_pos as u64;
        Ok(self.position)
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
        let arch_str = ct
            .metadata
            .get("general.architecture")
            .and_then(|v| v.to_string().ok().cloned())
            .unwrap_or_else(|| "llama".to_string());
        let model_arch = ModelArch::from_gguf_arch(&arch_str);

        if !model_arch.is_supported() {
            return Err(SwarmError::Internal(format!(
                "Architecture '{arch_str}' is not supported for split inference. \
                 DeepSeek-V2/V3 uses MoE+MLA which requires a fundamentally different forward path."
            )));
        }

        tracing::info!(arch = %model_arch, "Detected model architecture");

        let arch = &arch_str; // keep for metadata key lookups
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

        // Gemma 2 attention logit soft-capping (from GGUF metadata)
        let attn_logit_softcap = ct
            .metadata
            .get(&format!("{arch}.attn_logit_softcapping"))
            .and_then(|v| v.to_f32().ok())
            .filter(|&v| v > 0.0);

        let use_rope_contiguous = model_arch.use_rope_contiguous();
        let activation = model_arch.default_activation();

        let head_dim = embedding_length / head_count;
        let (cos, sin) = precompute_freqs_cis(rope_dim, rope_freq_base, context_length, &device)
            .map_err(|e| SwarmError::Internal(e.to_string()))?;
        let neg_inf = Tensor::new(f32::NEG_INFINITY, &device)
            .map_err(|e| SwarmError::Internal(e.to_string()))?;
        let use_gemma_norm = model_arch.use_gemma_norm();

        // Helper: create RmsNorm, applying Gemma's +1 weight offset if needed.
        // Gemma stores weights that expect: output = x * (1 + w) / rms(x)
        // We add 1 at load time so the standard forward pass works correctly.
        let make_norm = |qtensor: QTensor, eps: f64| -> Result<RmsNorm, SwarmError> {
            if use_gemma_norm {
                let w = qtensor
                    .dequantize(&device)
                    .map_err(|e| SwarmError::Internal(e.to_string()))?;
                let w_plus_one = (w + 1.0).map_err(|e| SwarmError::Internal(e.to_string()))?;
                let qt = QTensor::quantize(&w_plus_one, candle_core::quantized::GgmlDType::F32)
                    .map_err(|e| SwarmError::Internal(e.to_string()))?;
                RmsNorm::from_qtensor(qt, eps).map_err(|e| SwarmError::Internal(e.to_string()))
            } else {
                RmsNorm::from_qtensor(qtensor, eps).map_err(|e| SwarmError::Internal(e.to_string()))
            }
        };

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
            Some(make_norm(norm_tensor, rms_norm_eps)?)
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
                attention_norm: make_norm(attn_norm, rms_norm_eps)?,
                mlp: Mlp {
                    ffn_gate: QMatMul::from_qtensor(ffn_gate)
                        .map_err(|e| SwarmError::Internal(e.to_string()))?,
                    ffn_down: QMatMul::from_qtensor(ffn_down)
                        .map_err(|e| SwarmError::Internal(e.to_string()))?,
                    ffn_up: QMatMul::from_qtensor(ffn_up)
                        .map_err(|e| SwarmError::Internal(e.to_string()))?,
                    activation,
                },
                ffn_norm: make_norm(ffn_norm, rms_norm_eps)?,
                n_head: head_count,
                n_kv_head: head_count_kv,
                head_dim,
                cos: cos.clone(),
                sin: sin.clone(),
                neg_inf: neg_inf.clone(),
                use_rope_contiguous,
                attn_logit_softcap,
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
            let tokenizer_model = ct
                .metadata
                .get("tokenizer.ggml.model")
                .and_then(|v| v.to_string().ok().cloned())
                .unwrap_or_else(|| "gpt2".to_string());
            if !merges_raw.is_empty() {
                tracing::info!(
                    merges = merges_raw.len(),
                    pre_type = %pre_type,
                    tokenizer_model = %tokenizer_model,
                    "Loaded BPE tokenizer from GGUF"
                );
                Some(BpeTokenizer::from_gguf(
                    vocab,
                    &merges_raw,
                    &pre_type,
                    &tokenizer_model,
                ))
            } else {
                None
            }
        } else {
            None
        };

        // Extract EOS token IDs from GGUF metadata
        let mut eos_tokens = Vec::new();
        if let Some(eos_id) = ct
            .metadata
            .get("tokenizer.ggml.eos_token_id")
            .and_then(|v| v.to_u32().ok())
        {
            eos_tokens.push(eos_id);
        }
        // Some models define additional EOS tokens; add architecture-specific defaults
        match arch.as_str() {
            "qwen2" => {
                // Qwen2 uses 151643 (<|endoftext|>) and 151645 (<|im_end|>)
                for &id in &[151643u32, 151645] {
                    if !eos_tokens.contains(&id) {
                        eos_tokens.push(id);
                    }
                }
            }
            _ => {
                // Common fallback EOS token for LLaMA-family models
                if !eos_tokens.contains(&2) {
                    eos_tokens.push(2);
                }
            }
        }
        if eos_tokens.is_empty() {
            tracing::warn!("No EOS token found in GGUF metadata, using default [2]");
            eos_tokens.push(2);
        } else {
            tracing::info!(eos_tokens = ?eos_tokens, "Loaded EOS tokens from GGUF");
        }

        // Extract chat template from GGUF metadata
        let chat_template = ct
            .metadata
            .get("tokenizer.chat_template")
            .and_then(|v| v.to_string().ok().cloned())
            .filter(|s| !s.is_empty());

        // Resolve BOS/EOS token strings from their IDs + vocabulary
        let bos_id = ct
            .metadata
            .get("tokenizer.ggml.bos_token_id")
            .and_then(|v| v.to_u32().ok());
        let eos_str_id = ct
            .metadata
            .get("tokenizer.ggml.eos_token_id")
            .and_then(|v| v.to_u32().ok());
        let vocab_ref = vocabulary.as_deref().unwrap_or(&[]);
        let bos_token = bos_id
            .and_then(|id| vocab_ref.get(id as usize).cloned())
            .unwrap_or_default();
        let eos_token = eos_str_id
            .and_then(|id| vocab_ref.get(id as usize).cloned())
            .unwrap_or_default();

        if chat_template.is_some() {
            tracing::info!(
                bos = %bos_token,
                eos = %eos_token,
                "Loaded chat template from GGUF"
            );
        }

        let has_biases = layers.first().is_some_and(|l| l.attention_bq.is_some());
        tracing::info!(
            arch = %model_arch,
            layers = format!("[{layer_start}..{layer_end})"),
            total = block_count,
            is_first,
            is_last,
            has_qkv_biases = has_biases,
            rope = if use_rope_contiguous { "contiguous" } else { "interleaved" },
            activation = ?activation,
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
            arch: model_arch,
            device,
            vocabulary,
            tokenizer,
            eos_tokens,
            chat_template,
            bos_token,
            eos_token,
            max_seq_len: context_length,
        })
    }

    /// Load a partial model from local shard files + GGUF header.
    ///
    /// This is the shard-only alternative to `load_from_gguf`. Instead of needing
    /// the full GGUF file, it reads from:
    /// - `gguf_header.bin`: the raw GGUF header (metadata + tensor info table)
    /// - `shard_NNN.bin` files: layer-aligned shard files with packed tensor data
    ///
    /// The `ShardReader` uses the tensor entries to map virtual GGUF positions
    /// to shard-local offsets, so candle's GGUF parser works unchanged.
    #[allow(clippy::too_many_arguments)]
    pub fn load_from_shards(
        model_dir: &Path,
        shard_files: Vec<(u32, PathBuf)>,
        tensor_entries: &[Vec<crate::types::ShardTensorEntry>],
        total_gguf_size: u64,
        layer_start: usize,
        layer_end: usize,
        is_first: bool,
        is_last: bool,
    ) -> Result<Self, SwarmError> {
        let header_path = model_dir.join("gguf_header.bin");
        if !header_path.exists() {
            return Err(SwarmError::Internal(format!(
                "GGUF header not found at {}. The originating node must generate this file.",
                header_path.display()
            )));
        }

        // Read header to get tensor_data_offset
        let header_bytes = std::fs::read(&header_path).map_err(SwarmError::Io)?;
        let tensor_data_offset = {
            let mut cursor = std::io::Cursor::new(&header_bytes);
            let ct = gguf_file::Content::read(&mut cursor)
                .map_err(|e| SwarmError::Internal(format!("Failed to parse GGUF header: {e}")))?;
            ct.tensor_data_offset
        };

        tracing::info!(
            model_dir = %model_dir.display(),
            header_bytes = header_bytes.len(),
            tensor_data_offset,
            shards = shard_files.len(),
            layers = format!("[{layer_start}..{layer_end})"),
            "Loading split model from shard files"
        );

        let mut reader = ShardReader::new(
            &header_path,
            shard_files,
            tensor_entries,
            total_gguf_size,
            tensor_data_offset,
        )?;

        // Use the same GGUF parsing path as load_from_gguf, but reading from ShardReader
        let ct = gguf_file::Content::read(&mut reader).map_err(|e| {
            SwarmError::Internal(format!("Failed to read GGUF via ShardReader: {e}"))
        })?;

        // From here, the exact same logic as load_from_gguf
        let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
        if device.is_cuda() {
            tracing::info!("Split model using CUDA GPU");
        } else {
            tracing::info!("Split model using CPU (no CUDA available)");
        }

        let arch_str = ct
            .metadata
            .get("general.architecture")
            .and_then(|v| v.to_string().ok().cloned())
            .unwrap_or_else(|| "llama".to_string());
        let model_arch = ModelArch::from_gguf_arch(&arch_str);

        if !model_arch.is_supported() {
            return Err(SwarmError::Internal(format!(
                "Architecture '{arch_str}' is not supported for split inference. \
                 DeepSeek-V2/V3 uses MoE+MLA which requires a fundamentally different forward path."
            )));
        }

        tracing::info!(arch = %model_arch, "Detected model architecture");

        let arch = &arch_str;
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

        let attn_logit_softcap = ct
            .metadata
            .get(&format!("{arch}.attn_logit_softcapping"))
            .and_then(|v| v.to_f32().ok())
            .filter(|&v| v > 0.0);

        let use_rope_contiguous = model_arch.use_rope_contiguous();
        let activation = model_arch.default_activation();

        let head_dim = embedding_length / head_count;
        let (cos, sin) = precompute_freqs_cis(rope_dim, rope_freq_base, context_length, &device)
            .map_err(|e| SwarmError::Internal(e.to_string()))?;
        let neg_inf = Tensor::new(f32::NEG_INFINITY, &device)
            .map_err(|e| SwarmError::Internal(e.to_string()))?;
        let use_gemma_norm = model_arch.use_gemma_norm();

        let make_norm = |qtensor: QTensor, eps: f64| -> Result<RmsNorm, SwarmError> {
            if use_gemma_norm {
                let w = qtensor
                    .dequantize(&device)
                    .map_err(|e| SwarmError::Internal(e.to_string()))?;
                let w_plus_one = (w + 1.0).map_err(|e| SwarmError::Internal(e.to_string()))?;
                let qt = QTensor::quantize(&w_plus_one, candle_core::quantized::GgmlDType::F32)
                    .map_err(|e| SwarmError::Internal(e.to_string()))?;
                RmsNorm::from_qtensor(qt, eps).map_err(|e| SwarmError::Internal(e.to_string()))
            } else {
                RmsNorm::from_qtensor(qtensor, eps).map_err(|e| SwarmError::Internal(e.to_string()))
            }
        };

        let tok_embeddings = if is_first {
            let tok_embd = ct
                .tensor(&mut reader, "token_embd.weight", &device)
                .map_err(|e| SwarmError::Internal(format!("Failed to load embeddings: {e}")))?;
            let tok_embd = tok_embd
                .dequantize(&device)
                .map_err(|e| SwarmError::Internal(e.to_string()))?;
            Some(Embedding::new(tok_embd, embedding_length))
        } else {
            None
        };

        let norm = if is_last {
            let norm_tensor = ct
                .tensor(&mut reader, "output_norm.weight", &device)
                .map_err(|e| SwarmError::Internal(format!("Failed to load output_norm: {e}")))?;
            Some(make_norm(norm_tensor, rms_norm_eps)?)
        } else {
            None
        };

        let output = if is_last {
            let output_tensor = ct
                .tensor(&mut reader, "output.weight", &device)
                .or_else(|_| ct.tensor(&mut reader, "token_embd.weight", &device))
                .or_else(|_| {
                    // Weight-tied model fallback: load from tied_output_weight.bin
                    // This file contains the raw tensor data for token_embd.weight,
                    // downloaded separately so nodes without shard 0 can still
                    // project logits in distributed inference.
                    let tied_path = model_dir.join("tied_output_weight.bin");
                    if tied_path.exists() {
                        tracing::info!("Loading output head from tied_output_weight.bin");
                        // Get tensor info from GGUF metadata
                        let embd_info =
                            ct.tensor_infos.get("token_embd.weight").ok_or_else(|| {
                                candle_core::Error::Msg(
                                    "token_embd.weight not in GGUF tensor info".into(),
                                )
                            })?;
                        let raw_data = std::fs::read(&tied_path).map_err(|e| {
                            candle_core::Error::Msg(format!(
                                "Failed to read tied_output_weight.bin: {e}"
                            ))
                        })?;
                        candle_core::quantized::ggml_file::qtensor_from_ggml(
                            embd_info.ggml_dtype,
                            &raw_data,
                            embd_info.shape.dims().to_vec(),
                            &device,
                        )
                    } else {
                        Err(candle_core::Error::Msg(
                            "No output.weight, no token_embd.weight in shards, \
                             and no tied_output_weight.bin found"
                                .into(),
                        ))
                    }
                })
                .map_err(|e| SwarmError::Internal(format!("Failed to load output head: {e}")))?;
            Some(
                QMatMul::from_qtensor(output_tensor)
                    .map_err(|e| SwarmError::Internal(e.to_string()))?,
            )
        } else {
            None
        };

        let layer_end = layer_end.min(block_count);
        let mut layers = Vec::with_capacity(layer_end - layer_start);
        for layer_idx in layer_start..layer_end {
            let prefix = format!("blk.{layer_idx}");

            let attention_wq = ct
                .tensor(&mut reader, &format!("{prefix}.attn_q.weight"), &device)
                .map_err(|e| {
                    SwarmError::Internal(format!("Failed to load {prefix}.attn_q: {e}"))
                })?;
            let attention_wk = ct
                .tensor(&mut reader, &format!("{prefix}.attn_k.weight"), &device)
                .map_err(|e| {
                    SwarmError::Internal(format!("Failed to load {prefix}.attn_k: {e}"))
                })?;
            let attention_wv = ct
                .tensor(&mut reader, &format!("{prefix}.attn_v.weight"), &device)
                .map_err(|e| {
                    SwarmError::Internal(format!("Failed to load {prefix}.attn_v: {e}"))
                })?;
            let attention_wo = ct
                .tensor(
                    &mut reader,
                    &format!("{prefix}.attn_output.weight"),
                    &device,
                )
                .map_err(|e| {
                    SwarmError::Internal(format!("Failed to load {prefix}.attn_output: {e}"))
                })?;

            let attention_bq = ct
                .tensor(&mut reader, &format!("{prefix}.attn_q.bias"), &device)
                .ok()
                .map(|t| t.dequantize(&device))
                .transpose()
                .map_err(|e| SwarmError::Internal(format!("attn_q.bias dequant: {e}")))?;
            let attention_bk = ct
                .tensor(&mut reader, &format!("{prefix}.attn_k.bias"), &device)
                .ok()
                .map(|t| t.dequantize(&device))
                .transpose()
                .map_err(|e| SwarmError::Internal(format!("attn_k.bias dequant: {e}")))?;
            let attention_bv = ct
                .tensor(&mut reader, &format!("{prefix}.attn_v.bias"), &device)
                .ok()
                .map(|t| t.dequantize(&device))
                .transpose()
                .map_err(|e| SwarmError::Internal(format!("attn_v.bias dequant: {e}")))?;

            let ffn_gate = ct
                .tensor(&mut reader, &format!("{prefix}.ffn_gate.weight"), &device)
                .map_err(|e| {
                    SwarmError::Internal(format!("Failed to load {prefix}.ffn_gate: {e}"))
                })?;
            let ffn_down = ct
                .tensor(&mut reader, &format!("{prefix}.ffn_down.weight"), &device)
                .map_err(|e| {
                    SwarmError::Internal(format!("Failed to load {prefix}.ffn_down: {e}"))
                })?;
            let ffn_up = ct
                .tensor(&mut reader, &format!("{prefix}.ffn_up.weight"), &device)
                .map_err(|e| {
                    SwarmError::Internal(format!("Failed to load {prefix}.ffn_up: {e}"))
                })?;
            let attn_norm = ct
                .tensor(&mut reader, &format!("{prefix}.attn_norm.weight"), &device)
                .map_err(|e| {
                    SwarmError::Internal(format!("Failed to load {prefix}.attn_norm: {e}"))
                })?;
            let ffn_norm = ct
                .tensor(&mut reader, &format!("{prefix}.ffn_norm.weight"), &device)
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
                attention_norm: make_norm(attn_norm, rms_norm_eps)?,
                mlp: Mlp {
                    ffn_gate: QMatMul::from_qtensor(ffn_gate)
                        .map_err(|e| SwarmError::Internal(e.to_string()))?,
                    ffn_down: QMatMul::from_qtensor(ffn_down)
                        .map_err(|e| SwarmError::Internal(e.to_string()))?,
                    ffn_up: QMatMul::from_qtensor(ffn_up)
                        .map_err(|e| SwarmError::Internal(e.to_string()))?,
                    activation,
                },
                ffn_norm: make_norm(ffn_norm, rms_norm_eps)?,
                n_head: head_count,
                n_kv_head: head_count_kv,
                head_dim,
                cos: cos.clone(),
                sin: sin.clone(),
                neg_inf: neg_inf.clone(),
                use_rope_contiguous,
                attn_logit_softcap,
            });
        }

        let vocabulary = ct
            .metadata
            .get("tokenizer.ggml.tokens")
            .and_then(|v| v.to_vec().ok())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.to_string().ok().cloned())
                    .collect::<Vec<String>>()
            });

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
            let tokenizer_model = ct
                .metadata
                .get("tokenizer.ggml.model")
                .and_then(|v| v.to_string().ok().cloned())
                .unwrap_or_else(|| "gpt2".to_string());
            if !merges_raw.is_empty() {
                tracing::info!(
                    merges = merges_raw.len(),
                    pre_type = %pre_type,
                    tokenizer_model = %tokenizer_model,
                    "Loaded BPE tokenizer from GGUF header"
                );
                Some(BpeTokenizer::from_gguf(
                    vocab,
                    &merges_raw,
                    &pre_type,
                    &tokenizer_model,
                ))
            } else {
                None
            }
        } else {
            None
        };

        let mut eos_tokens = Vec::new();
        if let Some(eos_id) = ct
            .metadata
            .get("tokenizer.ggml.eos_token_id")
            .and_then(|v| v.to_u32().ok())
        {
            eos_tokens.push(eos_id);
        }
        match arch.as_str() {
            "qwen2" => {
                for &id in &[151643u32, 151645] {
                    if !eos_tokens.contains(&id) {
                        eos_tokens.push(id);
                    }
                }
            }
            _ => {
                if !eos_tokens.contains(&2) {
                    eos_tokens.push(2);
                }
            }
        }
        if eos_tokens.is_empty() {
            eos_tokens.push(2);
        }

        // Extract chat template from GGUF header metadata
        let chat_template = ct
            .metadata
            .get("tokenizer.chat_template")
            .and_then(|v| v.to_string().ok().cloned())
            .filter(|s| !s.is_empty());

        let bos_id = ct
            .metadata
            .get("tokenizer.ggml.bos_token_id")
            .and_then(|v| v.to_u32().ok());
        let eos_str_id = ct
            .metadata
            .get("tokenizer.ggml.eos_token_id")
            .and_then(|v| v.to_u32().ok());
        let vocab_ref = vocabulary.as_deref().unwrap_or(&[]);
        let bos_token = bos_id
            .and_then(|id| vocab_ref.get(id as usize).cloned())
            .unwrap_or_default();
        let eos_token = eos_str_id
            .and_then(|id| vocab_ref.get(id as usize).cloned())
            .unwrap_or_default();

        if chat_template.is_some() {
            tracing::info!(
                bos = %bos_token,
                eos = %eos_token,
                "Loaded chat template from GGUF header"
            );
        }

        let has_biases = layers.first().is_some_and(|l| l.attention_bq.is_some());
        tracing::info!(
            arch = %model_arch,
            layers = format!("[{layer_start}..{layer_end})"),
            total = block_count,
            is_first,
            is_last,
            has_qkv_biases = has_biases,
            rope = if use_rope_contiguous { "contiguous" } else { "interleaved" },
            activation = ?activation,
            context_length,
            "Loaded split model from shard files"
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
            arch: model_arch,
            device,
            vocabulary,
            tokenizer,
            eos_tokens,
            chat_template,
            bos_token,
            eos_token,
            max_seq_len: context_length,
        })
    }

    /// Build a causal mask for the given sequence length.
    /// Capped at 16 entries — evicts a random entry when full.
    fn mask(&mut self, t: usize) -> CandleResult<Tensor> {
        if let Some(mask) = self.masks.get(&t) {
            return Ok(mask.clone());
        }
        let mask: Vec<_> = (0..t)
            .flat_map(|i| (0..t).map(move |j| u8::from(j > i)))
            .collect();
        let mask = Tensor::from_slice(&mask, (t, t), &self.device)?;
        // Evict an entry if at capacity (simple LRU approximation)
        if self.masks.len() >= 16 {
            if let Some(&key) = self.masks.keys().next() {
                self.masks.remove(&key);
            }
        }
        self.masks.insert(t, mask.clone());
        Ok(mask)
    }

    /// Build a causal mask with KV offset for prefix-cached inference.
    ///
    /// When the KV cache has been pre-populated with prefix tokens,
    /// query tokens (suffix) attend to all prefix positions plus earlier suffix
    /// positions with proper causal ordering.
    ///
    /// `query_len`: number of new (suffix) tokens being processed.
    /// `kv_len`: total KV length (prefix_len + query_len).
    fn mask_with_offset(&self, query_len: usize, kv_len: usize) -> CandleResult<Tensor> {
        let offset = kv_len - query_len;
        let mask: Vec<_> = (0..query_len)
            .flat_map(|i| (0..kv_len).map(move |j| u8::from(j > offset + i)))
            .collect();
        Tensor::from_slice(&mask, (query_len, kv_len), &self.device)
    }

    /// Run the forward pass for this segment's layer range.
    ///
    /// - For the first segment: `input` is token IDs (i64 tensor, shape [1, seq_len]).
    ///   We apply the embedding lookup and return hidden states.
    /// - For intermediate segments: `input` is hidden state activations (f32, [1, seq, hidden_dim]).
    /// - For the last segment: returns logits (f32, [vocab_size]) for the last token position.
    /// - For intermediate segments: returns hidden states (f32, [1, seq, hidden_dim]).
    ///
    /// `kv_cache_store` and `request_id` provide per-request KV-cache isolation.
    /// The cache is stored externally in the `KvCacheStore`, keyed by request_id.
    pub fn forward(
        &mut self,
        input: &Tensor,
        index_pos: usize,
        kv_cache_store: &KvCacheStore,
        request_id: &str,
    ) -> Result<Tensor, SwarmError> {
        self.forward_with_lora(input, index_pos, kv_cache_store, request_id, None)
    }

    /// Forward pass with optional LoRA adapter applied per-layer.
    ///
    /// When `lora_adapter` is `Some`, the adapter's low-rank deltas are applied
    /// to attention (Q/K/V/O) and MLP (gate/up/down) projections at each layer.
    pub fn forward_with_lora(
        &mut self,
        input: &Tensor,
        index_pos: usize,
        kv_cache_store: &KvCacheStore,
        request_id: &str,
        lora_adapter: Option<&LoraAdapter>,
    ) -> Result<Tensor, SwarmError> {
        // Use component presence rather than layer indices for shard-aware is_first/is_last
        let is_first = self.tok_embeddings.is_some();
        let is_last = self.output.is_some();

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

        // Build a model_key for the KV-cache store
        let model_key = format!(
            "{}-{}-{}",
            self.layer_start, self.layer_end, self.total_layers
        );
        let num_layers = self.layers.len();

        // Get or create the per-request cache entry, extract the layer caches,
        // then drop the DashMap guard before running the (potentially slow) forward pass.
        let mut layer_kv_caches: Vec<Option<KvCache>> = {
            let mut entry = kv_cache_store.get_or_create(&model_key, request_id, num_layers);
            entry.last_accessed = std::time::Instant::now();
            entry.layers.clone()
        };

        // Detect pre-populated prefix cache entries (KV already present from prefix
        // cache restoration). If so, build an offset causal mask that allows the
        // suffix query tokens to attend to all prefix KV positions.
        let kv_offset = layer_kv_caches
            .first()
            .and_then(|c| c.as_ref())
            .map(|c| c.current_seq_len())
            .unwrap_or(0);

        let mask = if seq_len == 1 {
            None
        } else if kv_offset > 0 {
            // Prefix cache: suffix query attends to (offset + seq_len) key positions
            Some(
                self.mask_with_offset(seq_len, kv_offset + seq_len)
                    .map_err(|e| SwarmError::Internal(e.to_string()))?,
            )
        } else {
            Some(
                self.mask(seq_len)
                    .map_err(|e| SwarmError::Internal(e.to_string()))?,
            )
        };

        let max_seq_len = self.max_seq_len;

        // Run through our layers
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let abs_layer = self.layer_start + layer_idx;
            let lora_param = lora_adapter.map(|a| (a, abs_layer));

            let x = layer_in;
            let residual = &x;
            let x = layer
                .attention_norm
                .forward(&x)
                .map_err(|e| SwarmError::Internal(format!("attn_norm: {e}")))?;
            let attn = layer
                .forward_attn(
                    &x,
                    mask.as_ref(),
                    index_pos,
                    &mut layer_kv_caches[layer_idx],
                    max_seq_len,
                    lora_param,
                )
                .map_err(|e| SwarmError::Internal(format!("attn: {e}")))?;
            let x = (attn + residual).map_err(|e| SwarmError::Internal(e.to_string()))?;

            let residual = &x;
            let x = layer
                .ffn_norm
                .forward(&x)
                .map_err(|e| SwarmError::Internal(format!("ffn_norm: {e}")))?;
            let x = layer
                .mlp
                .forward(&x, lora_param)
                .map_err(|e| SwarmError::Internal(format!("mlp: {e}")))?;
            layer_in = (x + residual).map_err(|e| SwarmError::Internal(e.to_string()))?;
        }

        // Write the updated KV-caches back to the store
        {
            let mut entry = kv_cache_store.get_or_create(&model_key, request_id, num_layers);
            entry.layers = layer_kv_caches;
            entry.last_accessed = std::time::Instant::now();
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

    /// Forward pass for multimodal (vision + text) inference.
    ///
    /// If this is the first segment and `vision_embeddings` is provided, the
    /// vision embeddings are prepended to the text token embeddings before
    /// entering the transformer layers. Otherwise falls back to regular `forward`.
    ///
    /// `vision_embeddings` shape: (num_image_tokens, hidden_dim)
    pub fn forward_multimodal(
        &mut self,
        input: &Tensor,
        index_pos: usize,
        kv_cache_store: &KvCacheStore,
        request_id: &str,
        vision_embeddings: Option<&Tensor>,
    ) -> Result<Tensor, SwarmError> {
        let is_first = self.tok_embeddings.is_some();

        match (is_first, vision_embeddings) {
            (true, Some(vision_emb)) => {
                // First segment with vision: embed text tokens, merge with vision, then forward
                let input_dev = input
                    .to_device(&self.device)
                    .map_err(|e| SwarmError::Internal(format!("Device transfer: {e}")))?;

                let text_embeddings = self
                    .tok_embeddings
                    .as_ref()
                    .ok_or_else(|| SwarmError::Internal("Missing embedding table".into()))?
                    .forward(&input_dev)
                    .map_err(|e| SwarmError::Internal(format!("Embedding: {e}")))?;

                // Merge: prepend vision tokens before text tokens
                let merged = crate::inference::vision::merge_vision_text_embeddings(
                    &text_embeddings,
                    vision_emb,
                    &[],
                )?;

                // Now run through layers with the merged hidden states.
                // Temporarily remove tok_embeddings so forward() treats input as hidden states.
                let tok_emb = self.tok_embeddings.take();
                let result = self.forward(&merged, index_pos, kv_cache_store, request_id);
                self.tok_embeddings = tok_emb;
                result
            }
            _ => {
                // No vision embeddings or not the first segment — regular forward
                self.forward(input, index_pos, kv_cache_store, request_id)
            }
        }
    }

    /// Run a batched forward pass for multiple decode-step requests (seq_len=1 each).
    ///
    /// Stacks inputs along the batch dimension so that MLP/norm computations
    /// benefit from GPU parallelism.  Attention is still per-request because
    /// each request has its own `index_pos` and KV-cache.
    ///
    /// Returns one output tensor per request in the same order as `items`.
    /// Falls back to sequential `forward()` if any item has seq_len > 1.
    pub fn forward_batch(
        &mut self,
        items: &[BatchItem<'_>],
        kv_cache_store: &KvCacheStore,
    ) -> Result<Vec<Tensor>, SwarmError> {
        if items.is_empty() {
            return Ok(Vec::new());
        }

        // Fallback: if only 1 item or any item is a prefill (seq_len > 1), run sequentially
        if items.len() == 1 {
            let item = &items[0];
            let out = self.forward(item.input, item.index_pos, kv_cache_store, item.request_id)?;
            return Ok(vec![out]);
        }

        let is_first = self.tok_embeddings.is_some();
        let is_last = self.output.is_some();

        // Convert each input to its hidden state (apply embedding if first segment)
        let mut per_request: Vec<Tensor> = Vec::with_capacity(items.len());
        for item in items {
            let input = item
                .input
                .to_device(&self.device)
                .map_err(|e| SwarmError::Internal(format!("Device transfer: {e}")))?;
            let hidden = if is_first {
                self.tok_embeddings
                    .as_ref()
                    .ok_or_else(|| SwarmError::Internal("Missing embedding table".into()))?
                    .forward(&input)
                    .map_err(|e| SwarmError::Internal(format!("Embedding: {e}")))?
            } else {
                input
            };
            per_request.push(hidden);
        }

        // Check if all items have seq_len=1 (decode mode) — only then can we batch
        let all_decode = per_request.iter().all(|t| t.dim(1).unwrap_or(0) == 1);

        if !all_decode {
            // Mixed or prefill batch: fall back to sequential processing
            let mut results = Vec::with_capacity(items.len());
            for item in items {
                results.push(self.forward(
                    item.input,
                    item.index_pos,
                    kv_cache_store,
                    item.request_id,
                )?);
            }
            return Ok(results);
        }

        let batch_size = items.len();
        let model_key = format!(
            "{}-{}-{}",
            self.layer_start, self.layer_end, self.total_layers
        );
        let num_layers = self.layers.len();

        // Extract all per-request KV-caches up front (drop DashMap guards immediately)
        let mut all_kv_caches: Vec<Vec<Option<KvCache>>> = items
            .iter()
            .map(|item| {
                let mut entry =
                    kv_cache_store.get_or_create(&model_key, item.request_id, num_layers);
                entry.last_accessed = std::time::Instant::now();
                entry.layers.clone()
            })
            .collect();

        let max_seq_len = self.max_seq_len;

        // Stack all hidden states into a single batch tensor: [batch, 1, hidden_dim]
        let batch_refs: Vec<&Tensor> = per_request.iter().collect();
        let mut batched = Tensor::cat(&batch_refs, 0)
            .map_err(|e| SwarmError::Internal(format!("Batch stack: {e}")))?;
        // Shape is now [batch_size, 1, hidden_dim]

        // Process through layers
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let residual = batched.clone();

            // Batched attention_norm (position-independent)
            let normed = layer
                .attention_norm
                .forward(&batched)
                .map_err(|e| SwarmError::Internal(format!("attn_norm: {e}")))?;

            // Attention: must be per-request due to different index_pos and KV-caches
            let mut attn_outputs: Vec<Tensor> = Vec::with_capacity(batch_size);
            for (req_idx, item) in items.iter().enumerate() {
                // Extract this request's slice: [1, 1, hidden_dim]
                let x_i = normed
                    .narrow(0, req_idx, 1)
                    .map_err(|e| SwarmError::Internal(format!("narrow: {e}")))?;
                let attn_out = layer
                    .forward_attn(
                        &x_i,
                        None, // seq_len=1 → no mask needed
                        item.index_pos,
                        &mut all_kv_caches[req_idx][layer_idx],
                        max_seq_len,
                        None, // LoRA not supported in batch mode
                    )
                    .map_err(|e| SwarmError::Internal(format!("attn: {e}")))?;
                attn_outputs.push(attn_out);
            }

            // Re-stack attention outputs: [batch, 1, hidden_dim]
            let attn_refs: Vec<&Tensor> = attn_outputs.iter().collect();
            let attn_batched = Tensor::cat(&attn_refs, 0)
                .map_err(|e| SwarmError::Internal(format!("attn restack: {e}")))?;

            // Batched residual add
            let x = (&attn_batched + &residual).map_err(|e| SwarmError::Internal(e.to_string()))?;

            // Batched FFN norm + MLP (these are position-independent)
            let residual2 = x.clone();
            let x = layer
                .ffn_norm
                .forward(&x)
                .map_err(|e| SwarmError::Internal(format!("ffn_norm: {e}")))?;
            let x = layer
                .mlp
                .forward(&x, None) // LoRA not supported in batch mode
                .map_err(|e| SwarmError::Internal(format!("mlp: {e}")))?;
            batched = (&x + &residual2).map_err(|e| SwarmError::Internal(e.to_string()))?;
        }

        // Write updated KV-caches back
        for (req_idx, item) in items.iter().enumerate() {
            let mut entry = kv_cache_store.get_or_create(&model_key, item.request_id, num_layers);
            entry.layers = all_kv_caches[req_idx].clone();
            entry.last_accessed = std::time::Instant::now();
        }

        // Split batch back into per-request outputs
        let mut results = Vec::with_capacity(batch_size);
        for req_idx in 0..batch_size {
            let per_req = batched
                .narrow(0, req_idx, 1)
                .map_err(|e| SwarmError::Internal(format!("split: {e}")))?;

            if is_last {
                let norm = self
                    .norm
                    .as_ref()
                    .ok_or_else(|| SwarmError::Internal("Missing final norm".into()))?;
                let output = self
                    .output
                    .as_ref()
                    .ok_or_else(|| SwarmError::Internal("Missing output head".into()))?;
                let x = norm
                    .forward(&per_req)
                    .map_err(|e| SwarmError::Internal(format!("final_norm: {e}")))?;
                // seq_len=1, so i((.., 0, ..)) selects the single token
                let x = x
                    .i((.., 0, ..))
                    .map_err(|e| SwarmError::Internal(format!("last_token: {e}")))?;
                let logits = output
                    .forward(&x)
                    .map_err(|e| SwarmError::Internal(format!("output_proj: {e}")))?;
                results.push(logits);
            } else {
                results.push(per_req);
            }
        }

        Ok(results)
    }

    /// Return a reference to the loaded vocabulary, if available.
    pub fn vocab(&self) -> Option<&Vec<String>> {
        self.vocabulary.as_ref()
    }

    /// Return a reference to the BPE tokenizer, if available.
    pub fn tokenizer(&self) -> Option<&BpeTokenizer> {
        self.tokenizer.as_ref()
    }

    /// Return the number of transformer layers in this segment.
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Return the maximum sequence length supported by this model.
    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    /// Return the EOS token IDs loaded from GGUF metadata.
    pub fn eos_tokens(&self) -> &[u32] {
        &self.eos_tokens
    }

    /// Return the chat template from GGUF metadata, if available.
    pub fn chat_template(&self) -> Option<&str> {
        self.chat_template.as_deref()
    }

    /// Return the BOS token string from GGUF metadata.
    pub fn bos_token(&self) -> &str {
        &self.bos_token
    }

    /// Return the EOS token string from GGUF metadata.
    pub fn eos_token_str(&self) -> &str {
        &self.eos_token
    }

    /// Estimate GPU memory usage in MB for this model segment.
    ///
    /// Uses a rough heuristic: for quantized models (Q4/Q5/Q6), each parameter
    /// uses ~0.5-0.75 bytes. We estimate based on `num_layers * hidden_dim^2 * 4`
    /// (4 weight matrices per layer: Q, K, V, O + MLP) with a quantization factor.
    pub fn estimate_vram_mb(&self) -> u64 {
        let num_layers = self.layers.len() as u64;
        let hidden = self.hidden_dim as u64;

        // Each layer has roughly:
        //   4 attention matrices (Q, K, V, O): ~4 * hidden^2 params
        //   3 MLP matrices (gate, up, down): gate + up = 2 * hidden * 4*hidden, down = 4*hidden * hidden
        //   So MLP ≈ 12 * hidden^2 params
        //   Total per layer ≈ 16 * hidden^2 params
        //   Quantized at ~0.5 bytes per param (Q4)
        let params_per_layer = 16 * hidden * hidden;
        let bytes_per_param_quantized: f64 = 0.5;
        let layer_bytes = (params_per_layer as f64 * bytes_per_param_quantized) as u64;
        let mut total_bytes = num_layers * layer_bytes;

        // Add embedding table if loaded (~vocab_size * hidden * 2 bytes for f16)
        if self.tok_embeddings.is_some() {
            let vocab_size = self.vocabulary.as_ref().map(|v| v.len()).unwrap_or(32000) as u64;
            total_bytes += vocab_size * hidden * 2;
        }

        // Add output projection if loaded
        if self.output.is_some() {
            let vocab_size = self.vocabulary.as_ref().map(|v| v.len()).unwrap_or(32000) as u64;
            total_bytes += vocab_size * hidden * 2;
        }

        total_bytes / (1024 * 1024) // Convert to MB
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

    // Validate minimum header size: ndim(4) + dtype(4) = 8 bytes minimum
    let ndim = u32::from_le_bytes(
        bytes[pos..pos + 4]
            .try_into()
            .map_err(|_| SwarmError::Internal("Tensor bytes too short for ndim".into()))?,
    ) as usize;
    pos += 4;

    // Sanity-check ndim to avoid OOM on malicious input
    if ndim > 8 {
        return Err(SwarmError::Internal(format!(
            "Tensor ndim {} exceeds maximum 8",
            ndim
        )));
    }

    let mut shape = Vec::with_capacity(ndim);
    for _ in 0..ndim {
        if pos + 4 > bytes.len() {
            return Err(SwarmError::Internal(
                "Tensor bytes truncated in shape".into(),
            ));
        }
        let dim = u32::from_le_bytes(
            bytes[pos..pos + 4]
                .try_into()
                .map_err(|_| SwarmError::Internal("Tensor shape parse error".into()))?,
        ) as usize;
        shape.push(dim);
        pos += 4;
    }

    if pos + 4 > bytes.len() {
        return Err(SwarmError::Internal(
            "Tensor bytes truncated at dtype".into(),
        ));
    }
    let _dtype_tag = u32::from_le_bytes(
        bytes[pos..pos + 4]
            .try_into()
            .map_err(|_| SwarmError::Internal("Tensor dtype parse error".into()))?,
    );
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
    sorted_indices.sort_by(|&a, &b| {
        probs[b]
            .partial_cmp(&probs[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

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

// ── V2 Layer-Aligned Sharding ──

/// Describes one layer-aligned shard: which layers it contains and their tensors.
#[derive(Clone, Debug)]
pub struct LayerShardLayout {
    pub index: u32,
    pub layer_start: u32,
    /// Exclusive upper bound of layer range.
    pub layer_end: u32,
    /// Tensor entries: (name, absolute_gguf_offset, size), sorted by offset.
    pub tensors: Vec<(String, u64, u64)>,
    /// Total size of this shard in bytes (sum of tensor sizes).
    pub size_bytes: u64,
}

/// Group layers into `shard_count` shards of roughly equal byte size.
///
/// Non-layer tensors: `token_embd*` → shard 0, `output*`/`output_norm*` → last shard.
/// Each shard contains ONLY complete transformer layers — no layer spans two shards.
pub fn compute_layer_shard_layouts(
    meta: &GgufTensorMeta,
    shard_count: u32,
) -> Vec<LayerShardLayout> {
    if shard_count == 0 {
        return vec![];
    }

    // Classify tensors: per-layer vs prefix (token_embd) vs suffix (output)
    let mut layer_sizes: Vec<(u32, u64)> = Vec::new(); // (layer_idx, total_bytes)
    let mut layer_tensors: HashMap<u32, Vec<(String, u64, u64)>> = HashMap::new();
    let mut prefix_tensors: Vec<(String, u64, u64)> = Vec::new();
    let mut prefix_size: u64 = 0;
    let mut suffix_tensors: Vec<(String, u64, u64)> = Vec::new();
    let mut suffix_size: u64 = 0;

    // Per-layer byte totals
    let mut per_layer_bytes: HashMap<u32, u64> = HashMap::new();

    for (name, loc) in &meta.tensors {
        let abs_offset = meta.tensor_data_offset + loc.offset;
        if name.starts_with("blk.") {
            // Parse layer index: "blk.{N}.suffix"
            if let Some(idx_str) = name.strip_prefix("blk.").and_then(|s| s.split('.').next()) {
                if let Ok(layer_idx) = idx_str.parse::<u32>() {
                    *per_layer_bytes.entry(layer_idx).or_insert(0) += loc.size;
                    layer_tensors.entry(layer_idx).or_default().push((
                        name.clone(),
                        abs_offset,
                        loc.size,
                    ));
                }
            }
        } else if name.starts_with("token_embd") {
            prefix_tensors.push((name.clone(), abs_offset, loc.size));
            prefix_size += loc.size;
        } else if name.starts_with("output") {
            suffix_tensors.push((name.clone(), abs_offset, loc.size));
            suffix_size += loc.size;
        } else {
            // Other tensors (rope_freqs, etc.) go to prefix
            prefix_tensors.push((name.clone(), abs_offset, loc.size));
            prefix_size += loc.size;
        }
    }

    // Sorted layer indices
    let mut layer_indices: Vec<u32> = per_layer_bytes.keys().copied().collect();
    layer_indices.sort();

    // Build (layer_idx, bytes) sorted by layer index
    for &idx in &layer_indices {
        layer_sizes.push((idx, *per_layer_bytes.get(&idx).unwrap_or(&0)));
    }

    let total_layer_bytes: u64 = layer_sizes.iter().map(|(_, s)| s).sum();
    let total_bytes = total_layer_bytes + prefix_size + suffix_size;

    // Single shard: everything in one
    if shard_count == 1 {
        let mut all_tensors = prefix_tensors;
        for &idx in &layer_indices {
            if let Some(t) = layer_tensors.get(&idx) {
                all_tensors.extend(t.iter().cloned());
            }
        }
        all_tensors.extend(suffix_tensors);
        all_tensors.sort_by_key(|(_, off, _)| *off);

        let layer_start = layer_indices.first().copied().unwrap_or(0);
        let layer_end = layer_indices.last().map(|&l| l + 1).unwrap_or(0);

        return vec![LayerShardLayout {
            index: 0,
            layer_start,
            layer_end,
            tensors: all_tensors,
            size_bytes: total_bytes,
        }];
    }

    // Target bytes per shard (including prefix/suffix distributed to first/last)
    let target_per_shard = total_bytes / shard_count as u64;

    // Greedily assign layers to shards
    let mut layouts: Vec<LayerShardLayout> = Vec::new();
    let mut current_tensors: Vec<(String, u64, u64)> = Vec::new();
    let mut current_size: u64 = 0;
    let mut current_layer_start: Option<u32> = None;
    let mut current_layer_end: u32 = 0;

    // Add prefix tensors to current (will be shard 0)
    current_tensors.extend(prefix_tensors.iter().cloned());
    current_size += prefix_size;

    for (i, &(layer_idx, layer_bytes)) in layer_sizes.iter().enumerate() {
        if current_layer_start.is_none() {
            current_layer_start = Some(layer_idx);
        }
        current_layer_end = layer_idx + 1;

        if let Some(t) = layer_tensors.get(&layer_idx) {
            current_tensors.extend(t.iter().cloned());
        }
        current_size += layer_bytes;

        // Check if this is the last layer going to the last shard
        let is_last_layer = i == layer_sizes.len() - 1;
        let remaining_shards = shard_count as usize - layouts.len() - 1;
        let remaining_layers = layer_sizes.len() - i - 1;

        // Emit shard when we've reached target size, OR when we must emit to ensure
        // enough shards are created for remaining layers. Without the force-emit check,
        // models where layers are large relative to target_per_shard produce fewer
        // shards than requested (e.g., 28 layers / 8 shards → 7 shards).
        let should_emit = if is_last_layer || remaining_shards == 0 {
            // Last layer → handled after loop (final shard with suffix).
            // No remaining shards → keep accumulating for final shard.
            false
        } else if remaining_shards > remaining_layers {
            // Must emit now: more shards needed than layers remaining
            true
        } else {
            current_size >= target_per_shard
        };

        if should_emit {
            current_tensors.sort_by_key(|(_, off, _)| *off);
            layouts.push(LayerShardLayout {
                index: layouts.len() as u32,
                layer_start: current_layer_start.unwrap_or(0),
                layer_end: current_layer_end,
                tensors: std::mem::take(&mut current_tensors),
                size_bytes: current_size,
            });
            current_size = 0;
            current_layer_start = None;
        }
    }

    // Final shard: add suffix tensors
    current_tensors.extend(suffix_tensors.iter().cloned());
    current_size += suffix_size;
    current_tensors.sort_by_key(|(_, off, _)| *off);

    layouts.push(LayerShardLayout {
        index: layouts.len() as u32,
        layer_start: current_layer_start.unwrap_or(0),
        layer_end: current_layer_end,
        tensors: current_tensors,
        size_bytes: current_size,
    });

    layouts
}

/// Return all contiguous layer ranges from manifest ShardInfo entries.
///
/// Reads layer_range directly from each shard — v2 manifests have accurate
/// layer ranges computed from GGUF tensor metadata.
pub fn available_layer_ranges_from_manifest(
    manifest: &crate::types::ModelManifest,
    local_shard_indices: &[u32],
) -> Vec<(usize, usize)> {
    // Collect layer ranges from shards we hold
    let mut layer_bits = vec![false; manifest.num_layers as usize];
    for shard in &manifest.shards {
        if local_shard_indices.contains(&shard.index) {
            let start = shard.layer_range.0 as usize;
            let end = (shard.layer_range.1 as usize).min(layer_bits.len());
            for bit in layer_bits.iter_mut().take(end).skip(start) {
                *bit = true;
            }
        }
    }

    // Extract contiguous ranges from the bitmap
    let mut ranges = Vec::new();
    let mut run_start = 0;
    let mut in_run = false;
    for (i, &avail) in layer_bits.iter().enumerate() {
        if avail {
            if !in_run {
                run_start = i;
                in_run = true;
            }
        } else if in_run {
            ranges.push((run_start, i));
            in_run = false;
        }
    }
    if in_run {
        ranges.push((run_start, layer_bits.len()));
    }
    ranges
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

    // V1 byte-range layer tests (layer_range_computation, layer_range_cross_shard_tensor,
    // layer_range_alphabetical_gguf_order, available_layer_ranges_non_contiguous) removed.
    // Layer ranges are now computed from manifest ShardInfo.layer_range data.

    #[test]
    fn available_layer_ranges_from_manifest_basic() {
        use crate::types::{ModelId, ModelManifest, ShardInfo};

        let manifest = ModelManifest {
            schema_version: 2,
            id: ModelId("test".into()),
            name: "test".into(),
            architecture: crate::types::ModelArchitecture::Llama,
            num_layers: 12,
            num_params_billions: 0.0,
            quantization: crate::types::Quantization::Q4KM,
            total_size_bytes: 1000,
            shard_count: 3,
            shards: vec![
                ShardInfo {
                    index: 0,
                    layer_range: (0, 4),
                    size_bytes: 300,
                    hash: [0u8; 32],
                    tensors: vec![],
                },
                ShardInfo {
                    index: 1,
                    layer_range: (4, 8),
                    size_bytes: 300,
                    hash: [0u8; 32],
                    tensors: vec![],
                },
                ShardInfo {
                    index: 2,
                    layer_range: (8, 12),
                    size_bytes: 400,
                    hash: [0u8; 32],
                    tensors: vec![],
                },
            ],
            tokenizer_hash: [0u8; 32],
            manifest_hash: [0u8; 32],
            publisher: crate::types::NodeId([0u8; 32]),
            publish_date: chrono::Utc::now(),
            license: "MIT".into(),
        };

        // Single shard
        let ranges = available_layer_ranges_from_manifest(&manifest, &[0]);
        assert_eq!(ranges, vec![(0, 4)]);

        // Non-contiguous shards
        let ranges = available_layer_ranges_from_manifest(&manifest, &[0, 2]);
        assert_eq!(ranges, vec![(0, 4), (8, 12)]);

        // All shards → single range
        let ranges = available_layer_ranges_from_manifest(&manifest, &[0, 1, 2]);
        assert_eq!(ranges, vec![(0, 12)]);
    }

    // ── KvCacheStore tests ──

    #[test]
    fn kv_cache_store_isolates_requests() {
        let store = KvCacheStore::new(std::time::Duration::from_secs(600));

        // Two different request IDs should get independent caches
        let model_key = "test-model";
        let req_a = "request-a";
        let req_b = "request-b";
        let num_layers = 2;

        // Create caches for both requests using KvCache
        {
            let mut entry_a = store.get_or_create(model_key, req_a, num_layers);
            let mut cache = KvCache::new(2, 128);
            let k = Tensor::from_vec(vec![1.0f32, 2.0], &[1, 1, 1, 2], &Device::Cpu).unwrap();
            let v = Tensor::from_vec(vec![3.0f32, 4.0], &[1, 1, 1, 2], &Device::Cpu).unwrap();
            cache.append(&k, &v).unwrap();
            entry_a.layers[0] = Some(cache);
        }
        {
            let mut entry_b = store.get_or_create(model_key, req_b, num_layers);
            let mut cache = KvCache::new(2, 128);
            let k = Tensor::from_vec(vec![10.0f32, 20.0], &[1, 1, 1, 2], &Device::Cpu).unwrap();
            let v = Tensor::from_vec(vec![30.0f32, 40.0], &[1, 1, 1, 2], &Device::Cpu).unwrap();
            cache.append(&k, &v).unwrap();
            entry_b.layers[0] = Some(cache);
        }

        // Verify request A has its own cache values
        {
            let entry_a = store.get_or_create(model_key, req_a, num_layers);
            let cache = entry_a.layers[0].as_ref().unwrap();
            let k = cache.k().unwrap().unwrap();
            let k_data = k.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            assert_eq!(k_data, vec![1.0, 2.0]);
        }

        // Verify request B has its own separate cache values
        {
            let entry_b = store.get_or_create(model_key, req_b, num_layers);
            let cache = entry_b.layers[0].as_ref().unwrap();
            let k = cache.k().unwrap().unwrap();
            let k_data = k.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            assert_eq!(k_data, vec![10.0, 20.0]);
        }

        assert_eq!(store.active_entries(), 2);
    }

    #[test]
    fn kv_cache_store_clear_request() {
        let store = KvCacheStore::new(std::time::Duration::from_secs(600));

        let model_key = "test-model";
        let req_a = "request-a";
        let req_b = "request-b";

        // Create caches for two requests
        store.get_or_create(model_key, req_a, 4);
        store.get_or_create(model_key, req_b, 4);
        assert_eq!(store.active_entries(), 2);

        // Clear only request A
        store.clear_request(model_key, req_a);
        assert_eq!(store.active_entries(), 1);

        // Request B should still exist
        let entry_b = store.get_or_create(model_key, req_b, 4);
        assert_eq!(entry_b.layers.len(), 4);
    }

    #[test]
    fn kv_cache_store_cleanup_request_id() {
        let store = KvCacheStore::new(std::time::Duration::from_secs(600));

        // Create caches for the same request across multiple models
        store.get_or_create("model-a", "req-1", 2);
        store.get_or_create("model-b", "req-1", 2);
        store.get_or_create("model-a", "req-2", 2);
        assert_eq!(store.active_entries(), 3);

        // cleanup_request_id removes all entries for req-1
        store.cleanup_request_id("req-1");
        assert_eq!(store.active_entries(), 1);
    }

    #[test]
    fn kv_cache_store_cleanup_expired() {
        let store = KvCacheStore::new(std::time::Duration::from_millis(1));

        store.get_or_create("model", "req-1", 2);
        store.get_or_create("model", "req-2", 2);
        assert_eq!(store.active_entries(), 2);

        // Wait for TTL to expire
        std::thread::sleep(std::time::Duration::from_millis(10));

        let cleaned = store.cleanup_expired();
        assert_eq!(cleaned, 2);
        assert_eq!(store.active_entries(), 0);
    }

    #[test]
    fn kv_cache_store_fresh_entry_survives_cleanup() {
        let store = KvCacheStore::new(std::time::Duration::from_millis(50));

        // Create an entry that will expire
        store.get_or_create("model", "req-old", 2);
        std::thread::sleep(std::time::Duration::from_millis(60));

        // Create a fresh entry
        store.get_or_create("model", "req-new", 2);

        // Cleanup should only remove the old one
        let cleaned = store.cleanup_expired();
        assert_eq!(cleaned, 1);
        assert_eq!(store.active_entries(), 1);
    }

    // ── LRU eviction tests ──

    fn make_dummy_entry(vram_mb: u64) -> SplitModelEntry {
        // Construct a minimal valid SplitModel for eviction tests.
        // We never call forward() on it — only use it for LRU tracking.
        let dummy_model = SplitModel {
            tok_embeddings: None,
            layers: Vec::new(),
            norm: None,
            output: None,
            masks: HashMap::new(),
            layer_start: 0,
            layer_end: 0,
            total_layers: 0,
            hidden_dim: 0,
            arch: ModelArch::Llama,
            device: candle_core::Device::Cpu,
            vocabulary: None,
            tokenizer: None,
            eos_tokens: Vec::new(),
            chat_template: None,
            bos_token: String::new(),
            eos_token: String::new(),
            max_seq_len: DEFAULT_MAX_SEQ_LEN,
        };
        SplitModelEntry {
            model: std::sync::Arc::new(tokio::sync::Mutex::new(dummy_model)),
            last_used: std::sync::atomic::AtomicU64::new(0),
            estimated_vram_mb: vram_mb,
            batch_forwarder: None,
        }
    }

    #[test]
    fn lru_eviction_respects_budget() {
        use crate::types::*;

        let split_models: dashmap::DashMap<SplitModelKey, SplitModelEntry> =
            dashmap::DashMap::new();
        let active_pipelines: dashmap::DashMap<uuid::Uuid, PipelineAssignment> =
            dashmap::DashMap::new();

        // Add two models: one old, one newer
        let key_a = (ModelId("model-a".into()), 0, 10);
        let mut entry_a = make_dummy_entry(500);
        entry_a.last_used = std::sync::atomic::AtomicU64::new(100); // older
        split_models.insert(key_a.clone(), entry_a);

        let key_b = (ModelId("model-b".into()), 0, 10);
        let mut entry_b = make_dummy_entry(500);
        entry_b.last_used = std::sync::atomic::AtomicU64::new(200); // newer
        split_models.insert(key_b.clone(), entry_b);

        // Budget is 1200MB, we need 400MB more → total 1000 + 400 = 1400 > 1200
        // Must evict 1 model (oldest) to bring it under: 500 + 400 = 900 ≤ 1200
        let evicted = evict_split_models_lru(&split_models, &active_pipelines, 1200, 400);
        assert_eq!(evicted, 1);
        assert_eq!(split_models.len(), 1);
        // The older model (model-a, last_used=100) should have been evicted
        assert!(!split_models.contains_key(&key_a));
        assert!(split_models.contains_key(&key_b));
    }

    #[test]
    fn lru_eviction_no_eviction_under_budget() {
        use crate::types::*;

        let split_models: dashmap::DashMap<SplitModelKey, SplitModelEntry> =
            dashmap::DashMap::new();
        let active_pipelines: dashmap::DashMap<uuid::Uuid, PipelineAssignment> =
            dashmap::DashMap::new();

        let key = (ModelId("model".into()), 0, 10);
        let entry = make_dummy_entry(200);
        split_models.insert(key, entry);

        // Budget is 1000MB, need 100MB → no eviction needed
        let evicted = evict_split_models_lru(&split_models, &active_pipelines, 1000, 100);
        assert_eq!(evicted, 0);
        assert_eq!(split_models.len(), 1);
    }

    #[test]
    fn lru_eviction_protects_active_models() {
        use crate::types::*;

        let split_models: dashmap::DashMap<SplitModelKey, SplitModelEntry> =
            dashmap::DashMap::new();
        let active_pipelines: dashmap::DashMap<uuid::Uuid, PipelineAssignment> =
            dashmap::DashMap::new();

        // Add two models
        let key_a = (ModelId("active-model".into()), 0, 10);
        let mut entry_a = make_dummy_entry(500);
        entry_a.last_used = std::sync::atomic::AtomicU64::new(100); // oldest
        split_models.insert(key_a.clone(), entry_a);

        let key_b = (ModelId("idle-model".into()), 0, 10);
        let mut entry_b = make_dummy_entry(500);
        entry_b.last_used = std::sync::atomic::AtomicU64::new(200);
        split_models.insert(key_b.clone(), entry_b);

        // Mark model-a as having an active pipeline
        let pipeline = PipelineAssignment {
            request_id: uuid::Uuid::new_v4(),
            segments: vec![PipelineSegment {
                node_id: NodeId([1u8; 32]),
                shard_id: ShardId {
                    model_id: ModelId("active-model".into()),
                    index: 0,
                },
                layer_range: (0, 10),
            }],
            standbys: vec![],
        };
        active_pipelines.insert(uuid::Uuid::new_v4(), pipeline);

        // Budget is 800MB, need 400MB → should evict idle-model (not active one)
        let evicted = evict_split_models_lru(&split_models, &active_pipelines, 800, 400);
        assert_eq!(evicted, 1);
        assert!(split_models.contains_key(&key_a)); // Protected by active pipeline
        assert!(!split_models.contains_key(&key_b)); // Evicted
    }

    #[test]
    fn lru_eviction_multiple_models() {
        use crate::types::*;

        let split_models: dashmap::DashMap<SplitModelKey, SplitModelEntry> =
            dashmap::DashMap::new();
        let active_pipelines: dashmap::DashMap<uuid::Uuid, PipelineAssignment> =
            dashmap::DashMap::new();

        // Add 3 models of 400MB each (total 1200MB)
        for i in 0..3 {
            let key = (ModelId(format!("model-{i}")), 0, 10);
            let mut entry = make_dummy_entry(400);
            entry.last_used = std::sync::atomic::AtomicU64::new(i as u64 * 100);
            split_models.insert(key, entry);
        }

        // Budget 800MB, need 200MB → need to free 600MB → evict 2 oldest
        let evicted = evict_split_models_lru(&split_models, &active_pipelines, 800, 200);
        assert_eq!(evicted, 2);
        assert_eq!(split_models.len(), 1);
        // Only model-2 (last_used=200, newest) should remain
        assert!(split_models.contains_key(&(ModelId("model-2".into()), 0, 10)));
    }

    // ── Batch forward tests ──

    /// Create a minimal SplitModel with real layers for testing forward/forward_batch.
    fn make_test_split_model(num_layers: usize, hidden_dim: usize) -> SplitModel {
        // Build a minimal model with random weights for testing.
        // Uses CPU device and identity-like weight matrices.
        let device = candle_core::Device::Cpu;
        let head_dim = 64;
        let n_head = hidden_dim / head_dim;
        let n_kv_head = n_head; // no GQA in test model

        let make_qmatmul = |in_d: usize, out_d: usize| -> QMatMul {
            // Create a random weight tensor and quantize it
            let w = Tensor::randn(0f32, 0.02, (out_d, in_d), &device).unwrap();
            let qt = QTensor::quantize(&w, candle_core::quantized::GgmlDType::F32).unwrap();
            QMatMul::from_qtensor(qt).unwrap()
        };

        let max_seq_len = 128;
        let rope_dim = head_dim;
        let freq_base = 10000.0f32;
        let theta: Vec<f32> = (0..rope_dim / 2)
            .map(|i| 1.0 / freq_base.powf(i as f32 * 2.0 / rope_dim as f32))
            .collect();
        let idx: Vec<f32> = (0..max_seq_len).map(|i| i as f32).collect();
        let theta_t = Tensor::from_vec(theta.clone(), (1, rope_dim / 2), &device).unwrap();
        let idx_t = Tensor::from_vec(idx.clone(), (max_seq_len, 1), &device).unwrap();
        let freqs = idx_t.matmul(&theta_t).unwrap();
        let cos = freqs.cos().unwrap();
        let sin = freqs.sin().unwrap();
        let neg_inf = Tensor::new(f32::NEG_INFINITY, &device).unwrap();

        let mut layers = Vec::new();
        for _ in 0..num_layers {
            let norm_w = Tensor::ones((hidden_dim,), DType::F32, &device).unwrap();
            let make_rms_norm = |w: &Tensor| {
                let qt = QTensor::quantize(w, candle_core::quantized::GgmlDType::F32).unwrap();
                RmsNorm::from_qtensor(qt, 1e-6).unwrap()
            };
            layers.push(LayerWeights {
                attention_wq: make_qmatmul(hidden_dim, hidden_dim),
                attention_wk: make_qmatmul(hidden_dim, hidden_dim),
                attention_wv: make_qmatmul(hidden_dim, hidden_dim),
                attention_wo: make_qmatmul(hidden_dim, hidden_dim),
                attention_bq: None,
                attention_bk: None,
                attention_bv: None,
                attention_norm: make_rms_norm(&norm_w),
                mlp: Mlp {
                    ffn_gate: make_qmatmul(hidden_dim, hidden_dim * 4),
                    ffn_down: make_qmatmul(hidden_dim * 4, hidden_dim),
                    ffn_up: make_qmatmul(hidden_dim, hidden_dim * 4),
                    activation: Activation::SiLU,
                },
                ffn_norm: make_rms_norm(&norm_w),
                n_head,
                n_kv_head,
                head_dim,
                cos: cos.clone(),
                sin: sin.clone(),
                neg_inf: neg_inf.clone(),
                use_rope_contiguous: true,
                attn_logit_softcap: None,
            });
        }

        SplitModel {
            tok_embeddings: None,
            layers,
            norm: None,
            output: None,
            masks: HashMap::new(),
            layer_start: 0,
            layer_end: num_layers,
            total_layers: num_layers + 2, // Not last segment
            hidden_dim,
            arch: ModelArch::Llama,
            device,
            vocabulary: None,
            tokenizer: None,
            eos_tokens: vec![2],
            chat_template: None,
            bos_token: String::new(),
            eos_token: String::new(),
            max_seq_len,
        }
    }

    #[test]
    fn forward_batch_matches_sequential() {
        let hidden_dim = 128;
        let num_layers = 2;
        let mut model = make_test_split_model(num_layers, hidden_dim);
        let kv_store = KvCacheStore::new(std::time::Duration::from_secs(600));

        // Create two different input tensors (simulating decode step, seq_len=1)
        let input_a = Tensor::randn(0f32, 1.0, (1, 1, hidden_dim), &Device::Cpu).unwrap();
        let input_b = Tensor::randn(0f32, 1.0, (1, 1, hidden_dim), &Device::Cpu).unwrap();
        let index_pos = 5;

        // Run sequentially
        let out_a = model
            .forward(&input_a, index_pos, &kv_store, "seq-a")
            .unwrap();

        // Clear KV for a fresh comparison
        kv_store.clear_request(
            &format!(
                "{}-{}-{}",
                model.layer_start, model.layer_end, model.total_layers
            ),
            "seq-a",
        );

        let out_b = model
            .forward(&input_b, index_pos, &kv_store, "seq-b")
            .unwrap();
        kv_store.clear_request(
            &format!(
                "{}-{}-{}",
                model.layer_start, model.layer_end, model.total_layers
            ),
            "seq-b",
        );

        // Run batched
        let items = vec![
            BatchItem {
                input: &input_a,
                index_pos,
                request_id: "batch-a",
            },
            BatchItem {
                input: &input_b,
                index_pos,
                request_id: "batch-b",
            },
        ];
        let batch_out = model.forward_batch(&items, &kv_store).unwrap();

        // Compare shapes
        assert_eq!(out_a.shape(), batch_out[0].shape());
        assert_eq!(out_b.shape(), batch_out[1].shape());

        // Compare values (should be close — same model, same inputs, same index_pos)
        let diff_a = (&out_a - &batch_out[0]).unwrap().abs().unwrap();
        let diff_b = (&out_b - &batch_out[1]).unwrap().abs().unwrap();
        let flat_a = diff_a.flatten_all().unwrap();
        let flat_b = diff_b.flatten_all().unwrap();
        let max_diff_a: f32 = flat_a.max(0).unwrap().to_vec0().unwrap();
        let max_diff_b: f32 = flat_b.max(0).unwrap().to_vec0().unwrap();

        // Allow small numerical differences from batched vs sequential path
        assert!(
            max_diff_a < 1e-4,
            "Batch output A differs from sequential: max_diff={max_diff_a}"
        );
        assert!(
            max_diff_b < 1e-4,
            "Batch output B differs from sequential: max_diff={max_diff_b}"
        );
    }

    #[test]
    fn forward_batch_single_item_matches_forward() {
        let hidden_dim = 128;
        let mut model = make_test_split_model(1, hidden_dim);
        let kv_store = KvCacheStore::new(std::time::Duration::from_secs(600));

        let input = Tensor::randn(0f32, 1.0, (1, 1, hidden_dim), &Device::Cpu).unwrap();
        let index_pos = 0;

        // Single-item batch should use forward() path
        let items = vec![BatchItem {
            input: &input,
            index_pos,
            request_id: "single",
        }];
        let batch_out = model.forward_batch(&items, &kv_store).unwrap();
        assert_eq!(batch_out.len(), 1);

        // Shape should be [1, 1, hidden_dim] for intermediate segment
        assert_eq!(batch_out[0].dims(), &[1, 1, hidden_dim]);
    }

    #[test]
    fn forward_batch_empty_returns_empty() {
        let mut model = make_test_split_model(1, 128);
        let kv_store = KvCacheStore::new(std::time::Duration::from_secs(600));

        let items: Vec<BatchItem<'_>> = vec![];
        let out = model.forward_batch(&items, &kv_store).unwrap();
        assert!(out.is_empty());
    }

    #[tokio::test]
    async fn batch_forwarder_collects_concurrent_requests() {
        let hidden_dim = 128;
        let model = make_test_split_model(1, hidden_dim);
        let kv_store = std::sync::Arc::new(KvCacheStore::new(std::time::Duration::from_secs(600)));
        let model_arc = std::sync::Arc::new(tokio::sync::Mutex::new(model));
        let forwarder = std::sync::Arc::new(BatchForwarder::new(
            model_arc, kv_store, 4, // max batch size
        ));

        // Spawn 3 concurrent forward requests
        let mut handles = Vec::new();
        for i in 0..3 {
            let forwarder = forwarder.clone();
            let input = Tensor::randn(0f32, 1.0, (1, 1, hidden_dim), &Device::Cpu).unwrap();
            handles.push(tokio::spawn(async move {
                forwarder.submit(input, 0, format!("req-{i}")).await
            }));
        }

        // All should complete successfully
        for handle in handles {
            let result = handle.await.unwrap();
            assert!(result.is_ok(), "Batch forward failed: {:?}", result.err());
            let output = result.unwrap();
            assert_eq!(output.dims()[2], hidden_dim);
        }
    }

    #[test]
    fn flash_attn_cpu_vs_standard_attention() {
        // Compare CPU flash attention output vs standard matmul attention
        let device = Device::Cpu;
        let b = 1;
        let n_head = 4;
        let n_kv_head = 2; // GQA: 4 Q heads, 2 KV heads
        let seq_len = 8;
        let head_dim = 32;

        let q = Tensor::randn(0f32, 0.1, (b, n_head, seq_len, head_dim), &device).unwrap();
        let k = Tensor::randn(0f32, 0.1, (b, n_kv_head, seq_len, head_dim), &device).unwrap();
        let v = Tensor::randn(0f32, 0.1, (b, n_kv_head, seq_len, head_dim), &device).unwrap();
        let neg_inf = Tensor::new(f32::NEG_INFINITY, &device).unwrap();

        // Build causal mask (u8: 1=masked, 0=visible)
        let mask_data: Vec<u8> = (0..seq_len)
            .flat_map(|i| (0..seq_len).map(move |j| u8::from(j > i)))
            .collect();
        let mask = Tensor::from_slice(&mask_data, (seq_len, seq_len), &device).unwrap();

        // Standard path
        let out_std = standard_attention(
            &q,
            &k,
            &v,
            Some(&mask),
            head_dim,
            n_head,
            n_kv_head,
            &neg_inf,
            None,
        )
        .unwrap();

        // Flash path (run_attention dispatches to CPU flash on CPU device)
        let out_flash = run_attention(
            &q,
            &k,
            &v,
            Some(&mask),
            n_head,
            n_kv_head,
            head_dim,
            &neg_inf,
            None,
        )
        .unwrap();

        assert_eq!(out_std.shape(), out_flash.shape());

        let diff = (&out_std - &out_flash).unwrap().abs().unwrap();
        let max_diff: f32 = diff
            .flatten_all()
            .unwrap()
            .max(0)
            .unwrap()
            .to_vec0()
            .unwrap();
        assert!(
            max_diff < 1e-4,
            "CPU flash attention differs from standard: max_diff={max_diff}"
        );
    }

    #[test]
    fn flash_attn_cpu_decode_no_mask() {
        // Test decode step (seq_len=1) — no mask needed
        let device = Device::Cpu;
        let b = 1;
        let n_head = 4;
        let n_kv_head = 2;
        let head_dim = 32;
        let kv_len = 16;

        let q = Tensor::randn(0f32, 0.1, (b, n_head, 1, head_dim), &device).unwrap();
        let k = Tensor::randn(0f32, 0.1, (b, n_kv_head, kv_len, head_dim), &device).unwrap();
        let v = Tensor::randn(0f32, 0.1, (b, n_kv_head, kv_len, head_dim), &device).unwrap();
        let neg_inf = Tensor::new(f32::NEG_INFINITY, &device).unwrap();

        // Standard path (no mask for decode)
        let out_std = standard_attention(
            &q, &k, &v, None, head_dim, n_head, n_kv_head, &neg_inf, None,
        )
        .unwrap();

        // Flash path
        let out_flash = run_attention(
            &q, &k, &v, None, n_head, n_kv_head, head_dim, &neg_inf, None,
        )
        .unwrap();

        assert_eq!(out_std.shape(), out_flash.shape());

        let diff = (&out_std - &out_flash).unwrap().abs().unwrap();
        let max_diff: f32 = diff
            .flatten_all()
            .unwrap()
            .max(0)
            .unwrap()
            .to_vec0()
            .unwrap();
        assert!(
            max_diff < 1e-4,
            "CPU flash decode differs from standard: max_diff={max_diff}"
        );
    }

    // ── Model architecture detection tests ──

    #[test]
    fn model_arch_detection() {
        assert_eq!(ModelArch::from_gguf_arch("llama"), ModelArch::Llama);
        assert_eq!(ModelArch::from_gguf_arch("qwen2"), ModelArch::Qwen2);
        assert_eq!(ModelArch::from_gguf_arch("qwen3"), ModelArch::Qwen2);
        assert_eq!(ModelArch::from_gguf_arch("gemma"), ModelArch::Gemma);
        assert_eq!(ModelArch::from_gguf_arch("gemma2"), ModelArch::Gemma2);
        assert_eq!(ModelArch::from_gguf_arch("phi3"), ModelArch::Phi3);
        assert_eq!(ModelArch::from_gguf_arch("mistral"), ModelArch::Mistral);
        assert_eq!(ModelArch::from_gguf_arch("deepseek2"), ModelArch::DeepSeek2);
        assert!(matches!(
            ModelArch::from_gguf_arch("unknown_arch"),
            ModelArch::Unknown(_)
        ));
    }

    #[test]
    fn model_arch_properties() {
        // RoPE contiguous: only Qwen2 family
        assert!(ModelArch::Qwen2.use_rope_contiguous());
        assert!(!ModelArch::Llama.use_rope_contiguous());
        assert!(!ModelArch::Gemma2.use_rope_contiguous());
        assert!(!ModelArch::Phi3.use_rope_contiguous());
        assert!(!ModelArch::Mistral.use_rope_contiguous());

        // Activation: Gemma uses Gelu, others SiLU
        assert_eq!(ModelArch::Gemma.default_activation(), Activation::Gelu);
        assert_eq!(ModelArch::Gemma2.default_activation(), Activation::Gelu);
        assert_eq!(ModelArch::Llama.default_activation(), Activation::SiLU);
        assert_eq!(ModelArch::Qwen2.default_activation(), Activation::SiLU);
        assert_eq!(ModelArch::Phi3.default_activation(), Activation::SiLU);
        assert_eq!(ModelArch::Mistral.default_activation(), Activation::SiLU);

        // Gemma norm: only Gemma family
        assert!(ModelArch::Gemma.use_gemma_norm());
        assert!(ModelArch::Gemma2.use_gemma_norm());
        assert!(!ModelArch::Llama.use_gemma_norm());
        assert!(!ModelArch::Qwen2.use_gemma_norm());

        // Supported: DeepSeek2 is not supported
        assert!(ModelArch::Llama.is_supported());
        assert!(ModelArch::Qwen2.is_supported());
        assert!(ModelArch::Gemma2.is_supported());
        assert!(ModelArch::Phi3.is_supported());
        assert!(!ModelArch::DeepSeek2.is_supported());
    }

    // ── GQA verification tests ──

    /// Helper: create a SplitModel with explicit GQA configuration.
    fn make_gqa_test_model(
        num_layers: usize,
        hidden_dim: usize,
        n_head: usize,
        n_kv_head: usize,
        use_rope_contiguous: bool,
        activation: Activation,
        attn_logit_softcap: Option<f32>,
        arch: ModelArch,
    ) -> SplitModel {
        let device = candle_core::Device::Cpu;
        let head_dim = hidden_dim / n_head;

        let make_qmatmul = |in_d: usize, out_d: usize| -> QMatMul {
            let w = Tensor::randn(0f32, 0.02, (out_d, in_d), &device).unwrap();
            let qt = QTensor::quantize(&w, candle_core::quantized::GgmlDType::F32).unwrap();
            QMatMul::from_qtensor(qt).unwrap()
        };

        let max_seq_len = 128;
        let rope_dim = head_dim;
        let (cos, sin) = precompute_freqs_cis(rope_dim, 10000.0, max_seq_len, &device).unwrap();
        let neg_inf = Tensor::new(f32::NEG_INFINITY, &device).unwrap();

        let kv_dim = n_kv_head * head_dim;
        let mut layers = Vec::new();
        for _ in 0..num_layers {
            let norm_w = Tensor::ones((hidden_dim,), DType::F32, &device).unwrap();
            let make_rms_norm = |w: &Tensor| {
                let qt = QTensor::quantize(w, candle_core::quantized::GgmlDType::F32).unwrap();
                RmsNorm::from_qtensor(qt, 1e-6).unwrap()
            };
            layers.push(LayerWeights {
                attention_wq: make_qmatmul(hidden_dim, hidden_dim),
                attention_wk: make_qmatmul(hidden_dim, kv_dim),
                attention_wv: make_qmatmul(hidden_dim, kv_dim),
                attention_wo: make_qmatmul(hidden_dim, hidden_dim),
                attention_bq: None,
                attention_bk: None,
                attention_bv: None,
                attention_norm: make_rms_norm(&norm_w),
                mlp: Mlp {
                    ffn_gate: make_qmatmul(hidden_dim, hidden_dim * 4),
                    ffn_down: make_qmatmul(hidden_dim * 4, hidden_dim),
                    ffn_up: make_qmatmul(hidden_dim, hidden_dim * 4),
                    activation,
                },
                ffn_norm: make_rms_norm(&norm_w),
                n_head,
                n_kv_head,
                head_dim,
                cos: cos.clone(),
                sin: sin.clone(),
                neg_inf: neg_inf.clone(),
                use_rope_contiguous,
                attn_logit_softcap,
            });
        }

        SplitModel {
            tok_embeddings: None,
            layers,
            norm: None,
            output: None,
            masks: HashMap::new(),
            layer_start: 0,
            layer_end: num_layers,
            total_layers: num_layers + 2,
            hidden_dim,
            arch,
            device,
            vocabulary: None,
            tokenizer: None,
            eos_tokens: vec![2],
            chat_template: None,
            bos_token: String::new(),
            eos_token: String::new(),
            max_seq_len,
        }
    }

    /// Helper: assert two tensors are close within tolerance.
    fn assert_tensors_close(a: &Tensor, b: &Tensor, tol: f32, msg: &str) {
        assert_eq!(a.shape(), b.shape(), "{msg}: shape mismatch");
        let diff = (a - b).unwrap().abs().unwrap();
        let max_diff: f32 = diff
            .flatten_all()
            .unwrap()
            .max(0)
            .unwrap()
            .to_vec0()
            .unwrap();
        assert!(max_diff < tol, "{msg}: max_diff={max_diff} >= tol={tol}");
    }

    #[test]
    fn gqa_standard_attention_llama3_ratio() {
        // Llama 3 8B: GQA ratio=4 (scaled: n_head=8, n_kv_head=2)
        let device = Device::Cpu;
        let (b, n_head, n_kv_head, seq_len, head_dim) = (1, 8, 2, 12, 32);

        let q = Tensor::randn(0f32, 0.1, (b, n_head, seq_len, head_dim), &device).unwrap();
        let k = Tensor::randn(0f32, 0.1, (b, n_kv_head, seq_len, head_dim), &device).unwrap();
        let v = Tensor::randn(0f32, 0.1, (b, n_kv_head, seq_len, head_dim), &device).unwrap();
        let neg_inf = Tensor::new(f32::NEG_INFINITY, &device).unwrap();

        let mask_data: Vec<u8> = (0..seq_len)
            .flat_map(|i| (0..seq_len).map(move |j| u8::from(j > i)))
            .collect();
        let mask = Tensor::from_slice(&mask_data, (seq_len, seq_len), &device).unwrap();

        let out = standard_attention(
            &q,
            &k,
            &v,
            Some(&mask),
            head_dim,
            n_head,
            n_kv_head,
            &neg_inf,
            None,
        )
        .unwrap();
        assert_eq!(out.dims(), &[b, n_head, seq_len, head_dim]);
        let flat: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
        assert!(
            flat.iter().all(|v| v.is_finite()),
            "Output contains NaN/Inf"
        );
    }

    #[test]
    fn gqa_standard_attention_mqa_ratio() {
        // Multi-Query Attention: n_kv_head=1 (extreme GQA)
        let device = Device::Cpu;
        let (b, n_head, n_kv_head, seq_len, head_dim) = (1, 8, 1, 6, 32);

        let q = Tensor::randn(0f32, 0.1, (b, n_head, seq_len, head_dim), &device).unwrap();
        let k = Tensor::randn(0f32, 0.1, (b, n_kv_head, seq_len, head_dim), &device).unwrap();
        let v = Tensor::randn(0f32, 0.1, (b, n_kv_head, seq_len, head_dim), &device).unwrap();
        let neg_inf = Tensor::new(f32::NEG_INFINITY, &device).unwrap();

        let out = standard_attention(
            &q, &k, &v, None, head_dim, n_head, n_kv_head, &neg_inf, None,
        )
        .unwrap();
        assert_eq!(out.dims(), &[b, n_head, seq_len, head_dim]);
    }

    #[test]
    fn gqa_flash_vs_standard_llama3_prefill() {
        // CPU flash vs standard with GQA ratio=4, causal mask
        let device = Device::Cpu;
        let (b, n_head, n_kv_head, seq_len, head_dim) = (1, 8, 2, 10, 32);

        let q = Tensor::randn(0f32, 0.1, (b, n_head, seq_len, head_dim), &device).unwrap();
        let k = Tensor::randn(0f32, 0.1, (b, n_kv_head, seq_len, head_dim), &device).unwrap();
        let v = Tensor::randn(0f32, 0.1, (b, n_kv_head, seq_len, head_dim), &device).unwrap();
        let neg_inf = Tensor::new(f32::NEG_INFINITY, &device).unwrap();

        let mask_data: Vec<u8> = (0..seq_len)
            .flat_map(|i| (0..seq_len).map(move |j| u8::from(j > i)))
            .collect();
        let mask = Tensor::from_slice(&mask_data, (seq_len, seq_len), &device).unwrap();

        let out_std = standard_attention(
            &q,
            &k,
            &v,
            Some(&mask),
            head_dim,
            n_head,
            n_kv_head,
            &neg_inf,
            None,
        )
        .unwrap();
        let out_flash = run_attention(
            &q,
            &k,
            &v,
            Some(&mask),
            n_head,
            n_kv_head,
            head_dim,
            &neg_inf,
            None,
        )
        .unwrap();
        assert_tensors_close(&out_std, &out_flash, 1e-4, "GQA ratio=4 flash vs standard");
    }

    #[test]
    fn gqa_flash_vs_standard_llama3_decode() {
        // Decode step (seq_len=1 Q, longer KV) with GQA ratio=4
        let device = Device::Cpu;
        let (b, n_head, n_kv_head, head_dim, kv_len) = (1, 8, 2, 32, 20);

        let q = Tensor::randn(0f32, 0.1, (b, n_head, 1, head_dim), &device).unwrap();
        let k = Tensor::randn(0f32, 0.1, (b, n_kv_head, kv_len, head_dim), &device).unwrap();
        let v = Tensor::randn(0f32, 0.1, (b, n_kv_head, kv_len, head_dim), &device).unwrap();
        let neg_inf = Tensor::new(f32::NEG_INFINITY, &device).unwrap();

        let out_std = standard_attention(
            &q, &k, &v, None, head_dim, n_head, n_kv_head, &neg_inf, None,
        )
        .unwrap();
        let out_flash = run_attention(
            &q, &k, &v, None, n_head, n_kv_head, head_dim, &neg_inf, None,
        )
        .unwrap();
        assert_tensors_close(&out_std, &out_flash, 1e-4, "GQA decode flash vs standard");
    }

    #[test]
    fn gqa_forward_llama3_style() {
        // End-to-end forward with Llama 3-style GQA
        let (hidden_dim, n_head, n_kv_head) = (256, 8, 2);
        let mut model = make_gqa_test_model(
            2,
            hidden_dim,
            n_head,
            n_kv_head,
            false,
            Activation::SiLU,
            None,
            ModelArch::Llama,
        );
        let kv_store = KvCacheStore::new(std::time::Duration::from_secs(600));

        // Prefill
        let input = Tensor::randn(0f32, 1.0, (1, 6, hidden_dim), &Device::Cpu).unwrap();
        let out = model.forward(&input, 0, &kv_store, "llama3").unwrap();
        assert_eq!(out.dims(), &[1, 6, hidden_dim]);
        let flat: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
        assert!(flat.iter().all(|v| v.is_finite()));

        // Decode
        let decode = Tensor::randn(0f32, 1.0, (1, 1, hidden_dim), &Device::Cpu).unwrap();
        let out = model.forward(&decode, 6, &kv_store, "llama3").unwrap();
        assert_eq!(out.dims(), &[1, 1, hidden_dim]);
    }

    #[test]
    fn gqa_forward_mistral_style() {
        // Mistral 7B: same GQA as Llama 3
        let (hidden_dim, n_head, n_kv_head) = (256, 8, 2);
        let mut model = make_gqa_test_model(
            2,
            hidden_dim,
            n_head,
            n_kv_head,
            false,
            Activation::SiLU,
            None,
            ModelArch::Mistral,
        );
        let kv_store = KvCacheStore::new(std::time::Duration::from_secs(600));

        let input = Tensor::randn(0f32, 1.0, (1, 8, hidden_dim), &Device::Cpu).unwrap();
        let out = model.forward(&input, 0, &kv_store, "mistral").unwrap();
        assert_eq!(out.dims(), &[1, 8, hidden_dim]);
        let flat: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
        assert!(flat.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn gqa_forward_phi3_mha_style() {
        // Phi-3-mini: MHA (n_head == n_kv_head)
        let (hidden_dim, n_head, n_kv_head) = (192, 6, 6);
        let mut model = make_gqa_test_model(
            2,
            hidden_dim,
            n_head,
            n_kv_head,
            false,
            Activation::SiLU,
            None,
            ModelArch::Phi3,
        );
        let kv_store = KvCacheStore::new(std::time::Duration::from_secs(600));

        let input = Tensor::randn(0f32, 1.0, (1, 8, hidden_dim), &Device::Cpu).unwrap();
        let out = model.forward(&input, 0, &kv_store, "phi3").unwrap();
        assert_eq!(out.dims(), &[1, 8, hidden_dim]);
    }

    #[test]
    fn gemma2_gelu_activation_forward() {
        // Gemma 2 uses Gelu activation in MLP
        let (hidden_dim, n_head, n_kv_head) = (256, 8, 4);
        let mut model = make_gqa_test_model(
            1,
            hidden_dim,
            n_head,
            n_kv_head,
            false,
            Activation::Gelu,
            None,
            ModelArch::Gemma2,
        );
        let kv_store = KvCacheStore::new(std::time::Duration::from_secs(600));

        let input = Tensor::randn(0f32, 1.0, (1, 6, hidden_dim), &Device::Cpu).unwrap();
        let out = model.forward(&input, 0, &kv_store, "gemma2-gelu").unwrap();
        assert_eq!(out.dims(), &[1, 6, hidden_dim]);
        let flat: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
        assert!(
            flat.iter().all(|v| v.is_finite()),
            "Gemma2 Gelu produced NaN/Inf"
        );
    }

    #[test]
    fn gemma2_attn_logit_softcap() {
        // Test attention logit soft-capping (Gemma 2 feature)
        // Use stddev=1.0 so logits are large enough for softcap to visibly affect output
        let device = Device::Cpu;
        let (b, n_head, n_kv_head, seq_len, head_dim) = (1, 4, 2, 6, 32);

        let q = Tensor::randn(0f32, 1.0, (b, n_head, seq_len, head_dim), &device).unwrap();
        let k = Tensor::randn(0f32, 1.0, (b, n_kv_head, seq_len, head_dim), &device).unwrap();
        let v = Tensor::randn(0f32, 1.0, (b, n_kv_head, seq_len, head_dim), &device).unwrap();
        let neg_inf = Tensor::new(f32::NEG_INFINITY, &device).unwrap();

        // Without soft-capping
        let out_no_cap = standard_attention(
            &q, &k, &v, None, head_dim, n_head, n_kv_head, &neg_inf, None,
        )
        .unwrap();

        // With soft-capping (cap=50.0 like Gemma 2)
        let out_capped = standard_attention(
            &q,
            &k,
            &v,
            None,
            head_dim,
            n_head,
            n_kv_head,
            &neg_inf,
            Some(50.0),
        )
        .unwrap();

        // Both should produce valid output
        assert_eq!(out_no_cap.shape(), out_capped.shape());
        let flat: Vec<f32> = out_capped.flatten_all().unwrap().to_vec1().unwrap();
        assert!(
            flat.iter().all(|v| v.is_finite()),
            "Soft-capped attention NaN/Inf"
        );

        // Outputs should differ (soft-capping changes the attention weights)
        let diff = (&out_no_cap - &out_capped).unwrap().abs().unwrap();
        let max_diff: f32 = diff
            .flatten_all()
            .unwrap()
            .max(0)
            .unwrap()
            .to_vec0()
            .unwrap();
        assert!(max_diff > 0.0, "Soft-capping should change the output");
    }

    #[test]
    fn gemma2_full_forward_with_softcap() {
        // Gemma 2 end-to-end: Gelu + softcap + GQA
        let (hidden_dim, n_head, n_kv_head) = (256, 8, 4);
        let mut model = make_gqa_test_model(
            2,
            hidden_dim,
            n_head,
            n_kv_head,
            false,
            Activation::Gelu,
            Some(50.0),
            ModelArch::Gemma2,
        );
        let kv_store = KvCacheStore::new(std::time::Duration::from_secs(600));

        // Prefill
        let input = Tensor::randn(0f32, 1.0, (1, 8, hidden_dim), &Device::Cpu).unwrap();
        let out = model.forward(&input, 0, &kv_store, "gemma2-full").unwrap();
        assert_eq!(out.dims(), &[1, 8, hidden_dim]);
        let flat: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
        assert!(flat.iter().all(|v| v.is_finite()));

        // Decode
        let decode = Tensor::randn(0f32, 1.0, (1, 1, hidden_dim), &Device::Cpu).unwrap();
        let out = model.forward(&decode, 8, &kv_store, "gemma2-full").unwrap();
        assert_eq!(out.dims(), &[1, 1, hidden_dim]);
    }

    #[test]
    fn qwen2_forward_with_biases() {
        // Qwen2: GQA + contiguous RoPE + QKV biases
        let device = Device::Cpu;
        let (hidden_dim, n_head, n_kv_head) = (256, 8, 2);
        let head_dim = hidden_dim / n_head;
        let kv_dim = n_kv_head * head_dim;

        let make_qmatmul = |in_d: usize, out_d: usize| -> QMatMul {
            let w = Tensor::randn(0f32, 0.02, (out_d, in_d), &device).unwrap();
            let qt = QTensor::quantize(&w, candle_core::quantized::GgmlDType::F32).unwrap();
            QMatMul::from_qtensor(qt).unwrap()
        };

        let max_seq_len = 128;
        let (cos, sin) = precompute_freqs_cis(head_dim, 10000.0, max_seq_len, &device).unwrap();
        let neg_inf = Tensor::new(f32::NEG_INFINITY, &device).unwrap();
        let norm_w = Tensor::ones((hidden_dim,), DType::F32, &device).unwrap();
        let make_rms_norm = |w: &Tensor| {
            let qt = QTensor::quantize(w, candle_core::quantized::GgmlDType::F32).unwrap();
            RmsNorm::from_qtensor(qt, 1e-6).unwrap()
        };

        let layer = LayerWeights {
            attention_wq: make_qmatmul(hidden_dim, hidden_dim),
            attention_wk: make_qmatmul(hidden_dim, kv_dim),
            attention_wv: make_qmatmul(hidden_dim, kv_dim),
            attention_wo: make_qmatmul(hidden_dim, hidden_dim),
            attention_bq: Some(Tensor::randn(0f32, 0.01, (hidden_dim,), &device).unwrap()),
            attention_bk: Some(Tensor::randn(0f32, 0.01, (kv_dim,), &device).unwrap()),
            attention_bv: Some(Tensor::randn(0f32, 0.01, (kv_dim,), &device).unwrap()),
            attention_norm: make_rms_norm(&norm_w),
            mlp: Mlp {
                ffn_gate: make_qmatmul(hidden_dim, hidden_dim * 4),
                ffn_down: make_qmatmul(hidden_dim * 4, hidden_dim),
                ffn_up: make_qmatmul(hidden_dim, hidden_dim * 4),
                activation: Activation::SiLU,
            },
            ffn_norm: make_rms_norm(&norm_w),
            n_head,
            n_kv_head,
            head_dim,
            cos,
            sin,
            neg_inf,
            use_rope_contiguous: true,
            attn_logit_softcap: None,
        };

        let mut model = SplitModel {
            tok_embeddings: None,
            layers: vec![layer],
            norm: None,
            output: None,
            masks: HashMap::new(),
            layer_start: 0,
            layer_end: 1,
            total_layers: 3,
            hidden_dim,
            arch: ModelArch::Qwen2,
            device,
            vocabulary: None,
            tokenizer: None,
            eos_tokens: vec![2],
            chat_template: None,
            bos_token: String::new(),
            eos_token: String::new(),
            max_seq_len,
        };

        let kv_store = KvCacheStore::new(std::time::Duration::from_secs(600));
        let input = Tensor::randn(0f32, 1.0, (1, 6, hidden_dim), &Device::Cpu).unwrap();
        let out = model.forward(&input, 0, &kv_store, "qwen2").unwrap();
        assert_eq!(out.dims(), &[1, 6, hidden_dim]);
        let flat: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
        assert!(flat.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn gqa_kv_cache_dimensions() {
        // KV-cache stores with n_kv_head (not n_head)
        let (hidden_dim, n_head, n_kv_head) = (256, 8, 2);
        let head_dim = hidden_dim / n_head;
        let mut model = make_gqa_test_model(
            1,
            hidden_dim,
            n_head,
            n_kv_head,
            false,
            Activation::SiLU,
            None,
            ModelArch::Llama,
        );
        let kv_store = KvCacheStore::new(std::time::Duration::from_secs(600));

        let seq_len = 5;
        let input = Tensor::randn(0f32, 1.0, (1, seq_len, hidden_dim), &Device::Cpu).unwrap();
        model.forward(&input, 0, &kv_store, "cache-test").unwrap();

        let model_key = format!(
            "{}-{}-{}",
            model.layer_start, model.layer_end, model.total_layers
        );
        let entry = kv_store.get_or_create(&model_key, "cache-test", 1);
        let k = entry.layers[0].as_ref().unwrap().k().unwrap().unwrap();
        assert_eq!(
            k.dims(),
            &[1, n_kv_head, seq_len, head_dim],
            "KV cache should have n_kv_head={n_kv_head}, not n_head={n_head}"
        );
    }

    #[test]
    fn gqa_multiple_decode_steps() {
        // Multiple decode steps with GQA
        let (hidden_dim, n_head, n_kv_head) = (256, 8, 2);
        let mut model = make_gqa_test_model(
            2,
            hidden_dim,
            n_head,
            n_kv_head,
            false,
            Activation::SiLU,
            None,
            ModelArch::Llama,
        );
        let kv_store = KvCacheStore::new(std::time::Duration::from_secs(600));

        let input = Tensor::randn(0f32, 1.0, (1, 4, hidden_dim), &Device::Cpu).unwrap();
        model.forward(&input, 0, &kv_store, "multi-decode").unwrap();

        for step in 0..10 {
            let decode = Tensor::randn(0f32, 1.0, (1, 1, hidden_dim), &Device::Cpu).unwrap();
            let out = model
                .forward(&decode, 4 + step, &kv_store, "multi-decode")
                .unwrap();
            assert_eq!(out.dims(), &[1, 1, hidden_dim]);
            let flat: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
            assert!(flat.iter().all(|v| v.is_finite()), "Step {step} NaN/Inf");
        }
    }

    #[test]
    fn mlp_activation_silu_vs_gelu() {
        // Verify SiLU and Gelu produce different outputs
        let device = Device::Cpu;
        let dim = 64;
        let make_qmatmul = |in_d: usize, out_d: usize| -> QMatMul {
            let w = Tensor::randn(0f32, 0.02, (out_d, in_d), &device).unwrap();
            let qt = QTensor::quantize(&w, candle_core::quantized::GgmlDType::F32).unwrap();
            QMatMul::from_qtensor(qt).unwrap()
        };

        // Shared weights
        let gate = make_qmatmul(dim, dim * 4);
        let down = make_qmatmul(dim * 4, dim);
        let up = make_qmatmul(dim, dim * 4);

        let mlp_silu = Mlp {
            ffn_gate: gate.clone(),
            ffn_down: down.clone(),
            ffn_up: up.clone(),
            activation: Activation::SiLU,
        };
        let mlp_gelu = Mlp {
            ffn_gate: gate,
            ffn_down: down,
            ffn_up: up,
            activation: Activation::Gelu,
        };

        let input = Tensor::randn(0f32, 1.0, (1, 4, dim), &device).unwrap();
        let out_silu = mlp_silu.forward(&input, None).unwrap();
        let out_gelu = mlp_gelu.forward(&input, None).unwrap();

        assert_eq!(out_silu.shape(), out_gelu.shape());
        let diff = (&out_silu - &out_gelu).unwrap().abs().unwrap();
        let max_diff: f32 = diff
            .flatten_all()
            .unwrap()
            .max(0)
            .unwrap()
            .to_vec0()
            .unwrap();
        assert!(
            max_diff > 0.0,
            "SiLU and Gelu should produce different outputs"
        );
    }
}
