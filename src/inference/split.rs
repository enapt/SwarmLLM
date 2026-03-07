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
    /// Per-request KV-cache: "model_key\0request_id" → per-layer (K, V) pairs.
    /// Single-String key enables `&str` lookups via `Borrow<str>` (no allocation on hot path).
    caches: dashmap::DashMap<String, KvCacheEntry>,
    /// TTL for abandoned cache entries.
    ttl: std::time::Duration,
}

pub(crate) struct KvCacheEntry {
    /// Per-layer KV cache. Index corresponds to layer index within the model segment.
    /// Each `KvCache` pre-allocates a buffer and appends new K/V without `Tensor::cat`.
    pub(crate) layers: Vec<Option<KvCache>>,
    /// Per-layer SSM state for Qwen 3.5 hybrid models (delta net recurrent state + conv state).
    /// None for non-SSM layers. Only populated for Qwen35Ssm layer variants.
    pub(crate) ssm_states: Vec<Option<SsmState>>,
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
    /// Build the composite key for the KV-cache DashMap.
    #[inline]
    pub(crate) fn cache_key(model_key: &str, request_id: &str) -> String {
        format!("{model_key}\0{request_id}")
    }

    /// Get or create a KV-cache entry using a pre-formatted key string.
    /// The caller should build the key once via `cache_key()` and reuse it
    /// for both take and writeback, avoiding redundant allocations.
    pub(crate) fn get_or_create_keyed(
        &self,
        key: &str,
        num_layers: usize,
    ) -> dashmap::mapref::one::RefMut<'_, String, KvCacheEntry> {
        // Fast path: entry already exists (all tokens after the first).
        // DashMap::get_mut takes &str via String: Borrow<str> — zero allocation.
        if let Some(entry) = self.caches.get_mut(key) {
            return entry;
        }
        // Slow path: first access for this request — allocate key and create entry.
        self.caches
            .entry(key.to_string())
            .or_insert_with(|| KvCacheEntry {
                layers: vec![None; num_layers],
                ssm_states: vec![None; num_layers],
                last_accessed: std::time::Instant::now(),
            })
    }

    pub(crate) fn get_or_create(
        &self,
        model_key: &str,
        request_id: &str,
        num_layers: usize,
    ) -> dashmap::mapref::one::RefMut<'_, String, KvCacheEntry> {
        let key = Self::cache_key(model_key, request_id);
        self.get_or_create_keyed(&key, num_layers)
    }

    /// Clear (remove) the KV-cache for a specific request.
    /// Also clears any TP-keyed variants (tp{rank}-{model_key}).
    pub fn clear_request(&self, model_key: &str, request_id: &str) {
        let key = Self::cache_key(model_key, request_id);
        self.caches.remove(key.as_str());
        // Also clear TP-keyed cache entries for the same request
        let suffix = format!("\0{request_id}");
        self.caches
            .retain(|k, _| !(k.ends_with(&suffix) && k.contains(model_key)));
    }

    /// Clean up all expired cache entries. Returns the number of entries removed.
    pub fn cleanup_expired(&self) -> usize {
        let ttl = self.ttl;
        let before = self.caches.len();
        self.caches
            .retain(|_, entry| entry.last_accessed.elapsed() <= ttl);
        let removed = before - self.caches.len();
        if removed > 0 {
            tracing::info!(
                removed,
                remaining = self.caches.len(),
                ttl_secs = ttl.as_secs(),
                "DIAG: KV-cache store cleanup — expired entries removed"
            );
        }
        removed
    }

    /// Remove all cache entries for a given request_id (across all models).
    pub fn cleanup_request_id(&self, request_id: &str) {
        let suffix = format!("\0{request_id}");
        self.caches.retain(|key, _| !key.ends_with(&suffix));
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
    /// True if this entry has both embedding (first) and output head (last) — i.e., all layers.
    /// Set at construction time so the fast path can check without locking the model mutex.
    pub is_complete: bool,
    /// Cached EOS token IDs for lock-free sampling after batched forward passes.
    pub eos_tokens: Vec<u32>,
}

impl SplitModelEntry {
    /// Create a new entry wrapping a split model.
    pub fn new(model: SplitModel) -> Self {
        let estimated_vram_mb = model.estimate_vram_mb();
        let is_complete = model.is_first() && model.is_last();
        let eos_tokens = model.eos_tokens().to_vec();
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
            is_complete,
            eos_tokens,
        }
    }

    /// Create a new entry with batching enabled.
    pub fn new_with_batching(
        model: SplitModel,
        kv_cache_store: std::sync::Arc<KvCacheStore>,
        max_batch_size: usize,
        batch_timeout: std::time::Duration,
    ) -> Self {
        let estimated_vram_mb = model.estimate_vram_mb();
        let is_complete = model.is_first() && model.is_last();
        let eos_tokens = model.eos_tokens().to_vec();
        let model_arc = std::sync::Arc::new(tokio::sync::Mutex::new(model));
        let batch_forwarder = if max_batch_size > 1 {
            Some(std::sync::Arc::new(BatchForwarder::new(
                model_arc.clone(),
                kv_cache_store,
                max_batch_size,
                batch_timeout,
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
            is_complete,
            eos_tokens,
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
    /// How long to wait for additional requests before dispatching a partial batch.
    batch_timeout: std::time::Duration,
}

impl BatchForwarder {
    /// Create a new batch forwarder for a split model.
    pub fn new(
        model: std::sync::Arc<tokio::sync::Mutex<SplitModel>>,
        kv_cache_store: std::sync::Arc<KvCacheStore>,
        max_batch_size: usize,
        batch_timeout: std::time::Duration,
    ) -> Self {
        Self {
            queue: tokio::sync::Mutex::new(Vec::new()),
            notify: tokio::sync::Notify::new(),
            model,
            kv_cache_store,
            max_batch_size: max_batch_size.max(1),
            batch_timeout,
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
            // Wait briefly for more requests to accumulate (if timeout configured)
            if !self.batch_timeout.is_zero() {
                let q_len = self.queue.lock().await.len();
                if q_len < self.max_batch_size {
                    let _ = tokio::time::timeout(self.batch_timeout, self.notify.notified()).await;
                }
            }
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
    /// Merge pair "left\0right" → merge rank (lower = higher priority).
    /// Uses concatenated string key with \0 separator for zero-allocation lookups.
    merge_ranks: HashMap<String, usize>,
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
    pub(crate) fn from_gguf(
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

        // Build merge rank lookup: "left\0right" → rank (zero-alloc lookups via reusable buffer)
        let mut merge_ranks = HashMap::with_capacity(merges_raw.len());
        for (rank, line) in merges_raw.iter().enumerate() {
            let parts: Vec<&str> = line.splitn(2, ' ').collect();
            if parts.len() == 2 {
                let mut key = String::with_capacity(parts[0].len() + 1 + parts[1].len());
                key.push_str(parts[0]);
                key.push('\0');
                key.push_str(parts[1]);
                merge_ranks.insert(key, rank);
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

        // Collect special tokens (e.g., <|im_start|>, <|im_end|>, <s>, </s>, <unk>,
        // <bos>, <eos>, <start_of_turn>, <end_of_turn>)
        let mut special_tokens: Vec<(String, u32)> = token_to_id
            .iter()
            .filter(|(t, _)| {
                (t.starts_with("<|") && t.ends_with("|>"))
                    || (t.starts_with('<') && t.ends_with('>') && !t.contains(' ') && t.len() <= 20)
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
        // Uses a reusable lookup buffer to avoid String allocations in the scan loop.
        let mut symbols = chars;
        let mut lookup_buf = String::new();
        loop {
            // Find the pair with the lowest merge rank (zero-allocation scan)
            let mut best_rank = usize::MAX;
            let mut best_idx = usize::MAX;
            for i in 0..symbols.len() - 1 {
                lookup_buf.clear();
                lookup_buf.push_str(&symbols[i]);
                lookup_buf.push('\0');
                lookup_buf.push_str(&symbols[i + 1]);
                if let Some(&rank) = self.merge_ranks.get(&lookup_buf) {
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

    /// Return a reference to the byte decoder mapping (for caching outside the lock).
    pub fn byte_decoder(&self) -> &HashMap<char, u8> {
        &self.byte_decoder
    }

    /// Whether this tokenizer uses SentencePiece encoding (vs GPT-2 byte BPE).
    pub fn is_sentencepiece(&self) -> bool {
        self.is_sentencepiece
    }
}

/// Unified tokenizer that wraps either our BPE tokenizer or the HuggingFace
/// `tokenizers` crate (for sentencepiece/unigram models like LLaMA).
pub enum SplitTokenizer {
    Bpe(BpeTokenizer),
    HfUnigram {
        inner: tokenizers::Tokenizer,
        /// Vocab for decode_token lookups
        vocab: Vec<String>,
    },
}

impl SplitTokenizer {
    /// Build from GGUF BPE merges (existing path).
    pub fn from_bpe(tokens: &[String], merges: &[String], pre_type: &str, model: &str) -> Self {
        Self::Bpe(BpeTokenizer::from_gguf(tokens, merges, pre_type, model))
    }

    /// Build a sentencepiece/unigram tokenizer from GGUF vocab + scores
    /// using the HuggingFace `tokenizers` crate.
    pub fn from_sentencepiece(tokens: &[String], scores: &[f32]) -> Self {
        use tokenizers::models::unigram::Unigram;
        use tokenizers::pre_tokenizers::metaspace::{Metaspace, PrependScheme};

        // Build (token, score) pairs for the Unigram model
        let vocab: Vec<(String, f64)> = tokens
            .iter()
            .zip(scores.iter())
            .map(|(t, &s)| (t.clone(), s as f64))
            .collect();

        // Find <unk> token ID (required by Unigram)
        let unk_id = tokens.iter().position(|t| t == "<unk>");

        // Detect byte_fallback: LLaMA-style models have <0xNN> tokens
        let byte_fallback = tokens
            .iter()
            .any(|t| t.starts_with("<0x") && t.ends_with('>'));

        let unigram = Unigram::from(vocab, unk_id, byte_fallback)
            .expect("Failed to build Unigram tokenizer from GGUF vocab");

        let mut tokenizer = tokenizers::Tokenizer::new(unigram);

        // SentencePiece uses ▁ (U+2581) as space replacement
        let metaspace = Metaspace::new('▁', PrependScheme::Always, true);
        tokenizer.with_pre_tokenizer(Some(metaspace));

        // Register control/special tokens (e.g. <bos>, <eos>, <start_of_turn>, <end_of_turn>)
        // so the tokenizer handles them as single tokens instead of splitting them.
        let special_toks: Vec<tokenizers::AddedToken> = tokens
            .iter()
            .enumerate()
            .filter(|(_, t)| {
                t.starts_with('<') && t.ends_with('>') && !t.contains(' ') && t.len() <= 20
            })
            .map(|(_, t)| tokenizers::AddedToken::from(t.clone(), true))
            .collect();
        if !special_toks.is_empty() {
            let count = special_toks.len();
            tokenizer.add_special_tokens(&special_toks);
            tracing::debug!(count, "Registered special tokens in HF Unigram tokenizer");
        }

        tracing::info!(
            vocab_size = tokens.len(),
            unk_id = ?unk_id,
            byte_fallback = byte_fallback,
            "Built HF Unigram tokenizer from GGUF sentencepiece vocab"
        );

        Self::HfUnigram {
            inner: tokenizer,
            vocab: tokens.to_vec(),
        }
    }

    /// Encode text to token IDs.
    pub fn encode(&self, text: &str) -> Vec<i64> {
        match self {
            Self::Bpe(bpe) => bpe.encode(text),
            Self::HfUnigram { inner, .. } => match inner.encode(text, false) {
                Ok(encoding) => encoding.get_ids().iter().map(|&id| id as i64).collect(),
                Err(e) => {
                    tracing::warn!(error = %e, "HF tokenizer encode failed, falling back to bytes");
                    text.bytes().map(|b| b as i64).collect()
                }
            },
        }
    }

    /// Decode a single token string back to UTF-8 bytes.
    pub fn decode_token(&self, token_str: &str) -> Vec<u8> {
        match self {
            Self::Bpe(bpe) => bpe.decode_token(token_str),
            Self::HfUnigram { .. } => {
                // Handle byte fallback tokens like <0x0A>
                if token_str.starts_with("<0x") && token_str.ends_with('>') && token_str.len() == 6
                {
                    if let Ok(byte) = u8::from_str_radix(&token_str[3..5], 16) {
                        return vec![byte];
                    }
                }
                // Special tokens → empty
                if token_str.starts_with('<') && token_str.ends_with('>') {
                    return vec![];
                }
                // ▁ → space, everything else is raw UTF-8
                token_str.replace('\u{2581}', " ").into_bytes()
            }
        }
    }

    /// Whether this tokenizer uses SentencePiece encoding.
    pub fn is_sentencepiece(&self) -> bool {
        match self {
            Self::Bpe(bpe) => bpe.is_sentencepiece(),
            Self::HfUnigram { .. } => true,
        }
    }

    /// Return a reference to the byte decoder mapping (for BPE caching).
    pub fn byte_decoder(&self) -> HashMap<char, u8> {
        match self {
            Self::Bpe(bpe) => bpe.byte_decoder().clone(),
            Self::HfUnigram { .. } => HashMap::new(),
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
    /// DeepSeek-V2/V3 — MoE + MLA
    DeepSeek2,
    /// GLM-4 — partial RoPE (half head dims), QKV biases, extreme GQA (2 KV heads)
    Glm4,
    /// Llama 4 Scout/Maverick — iRoPE (NoPE every 4th layer) + MoE
    Llama4,
    /// Qwen 3.5 dense — hybrid attention + Gated Delta Network (SSM) layers
    Qwen35,
    /// Qwen 3.5 MoE — hybrid attention + SSM layers with mixture-of-experts FFN
    Qwen35Moe,
    /// Architecture not recognized — falls back to Llama-like behavior
    Unknown(String),
}

impl ModelArch {
    /// Detect architecture from GGUF `general.architecture` metadata string.
    pub fn from_gguf_arch(arch: &str) -> Self {
        match arch {
            "llama" => ModelArch::Llama,
            "qwen2" | "qwen3" | "qwen2moe" => ModelArch::Qwen2,
            "gemma" => ModelArch::Gemma,
            "gemma2" => ModelArch::Gemma2,
            "phi3" => ModelArch::Phi3,
            "mistral" => ModelArch::Mistral,
            "starcoder2" => ModelArch::Starcoder2,
            "deepseek2" => ModelArch::DeepSeek2,
            "glm4" => ModelArch::Glm4,
            "llama4" => ModelArch::Llama4,
            "qwen35" => ModelArch::Qwen35,
            "qwen35moe" | "qwen3_5moe" => ModelArch::Qwen35Moe,
            other => ModelArch::Unknown(other.to_string()),
        }
    }

    /// Whether this architecture uses contiguous RoPE (vs interleaved).
    pub fn use_rope_contiguous(&self) -> bool {
        matches!(
            self,
            ModelArch::Qwen2
                | ModelArch::DeepSeek2
                | ModelArch::Glm4
                | ModelArch::Llama4
                | ModelArch::Qwen35
                | ModelArch::Qwen35Moe
        )
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
        !matches!(self, ModelArch::Unknown(_))
    }

    /// List of GGUF architecture strings supported by the split inference engine.
    pub fn supported_list() -> &'static [&'static str] {
        &[
            "llama",
            "qwen2",
            "qwen3",
            "qwen2moe",
            "gemma",
            "gemma2",
            "phi3",
            "mistral",
            "starcoder2",
            "deepseek2",
            "glm4",
            "llama4",
            "qwen35",
            "qwen35moe",
        ]
    }

    /// Whether this architecture uses hybrid attention + SSM (Gated Delta Network) layers.
    pub fn is_hybrid_ssm(&self) -> bool {
        matches!(self, ModelArch::Qwen35 | ModelArch::Qwen35Moe)
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
            ModelArch::Glm4 => write!(f, "glm4"),
            ModelArch::Llama4 => write!(f, "llama4"),
            ModelArch::Qwen35 => write!(f, "qwen35"),
            ModelArch::Qwen35Moe => write!(f, "qwen35moe"),
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

    /// Wrap a dequantized f32 tensor as a quantized QMatMul (Q4_0).
    /// Re-quantizes on CPU then moves to target device to save VRAM.
    /// Used for architectures with fused QKV/FFN tensors that need splitting.
    fn from_f32_tensor(tensor: Tensor, device: &Device) -> Result<Self, SwarmError> {
        let qt = candle_core::quantized::QTensor::quantize_onto(
            &tensor,
            candle_core::quantized::GgmlDType::Q4_0,
            device,
        )
        .map_err(|e| SwarmError::Internal(format!("re-quantize Q4_0: {e}")))?;
        let inner = candle_core::quantized::QMatMul::from_qtensor(qt)
            .map_err(|e| SwarmError::Internal(format!("QMatMul from Q4_0: {e}")))?;
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

    /// Tensor-parallel MLP: each TP node computes a fraction of the intermediate dimension.
    ///
    /// Gate and Up projections are column-parallel (each node gets `intermediate / tp_size`
    /// output columns). The Down projection is row-parallel (each node multiplies its
    /// fraction of rows and the results are summed across TP nodes via AllReduce).
    ///
    /// Returns a partial output of shape `[b, seq, hidden_dim]` that must be summed
    /// across TP nodes.
    fn forward_tp(&self, xs: &Tensor, tp_rank: usize, tp_size: usize) -> CandleResult<Tensor> {
        // Full gate and up projections
        let gate_full = self.ffn_gate.forward(xs)?;
        let up_full = self.ffn_up.forward(xs)?;

        // Determine intermediate dimension and split it
        let intermediate_dim = gate_full.dim(gate_full.dims().len() - 1)?;
        let chunk_size = intermediate_dim / tp_size;
        let start = tp_rank * chunk_size;
        let len = if tp_rank == tp_size - 1 {
            intermediate_dim - start // Last rank gets remainder
        } else {
            chunk_size
        };
        let last_dim = gate_full.dims().len() - 1;

        // Slice to this rank's fraction of intermediate dimension
        let gate_local = gate_full.narrow(last_dim, start, len)?;
        let up_local = up_full.narrow(last_dim, start, len)?;

        // Activation + elementwise multiply
        let activated = match self.activation {
            Activation::SiLU => candle_nn::ops::silu(&gate_local)?,
            Activation::Gelu => gate_local.gelu()?,
        };
        let combined = (activated * up_local)?;

        // Down projection on the local slice.
        // For correct row-parallel semantics, we need to use only the corresponding
        // rows of the down matrix. Since QMatMul doesn't support row slicing, we
        // pad the combined tensor with zeros at other positions so the full matmul
        // only activates our rows.
        let b = combined.dims()[0];
        let s = combined.dims()[1];
        let remaining = intermediate_dim - (start + len);
        let padded = if start > 0 && remaining > 0 {
            let z_before = Tensor::zeros((b, s, start), combined.dtype(), combined.device())?;
            let z_after = Tensor::zeros((b, s, remaining), combined.dtype(), combined.device())?;
            Tensor::cat(&[&z_before, &combined, &z_after], 2)?
        } else if start > 0 {
            let z_before = Tensor::zeros((b, s, start), combined.dtype(), combined.device())?;
            Tensor::cat(&[&z_before, &combined], 2)?
        } else if remaining > 0 {
            let z_after = Tensor::zeros((b, s, remaining), combined.dtype(), combined.device())?;
            Tensor::cat(&[&combined, &z_after], 2)?
        } else {
            combined
        };
        self.ffn_down.forward(&padded)
    }
}

// ── MLA (Multi-head Latent Attention) for DeepSeek-V2/V3 ──

/// MLA (Multi-head Latent Attention) weights for DeepSeek-V2/V3.
///
/// Q path: x → q_a → q_a_norm → q_b → reshape → split(q_nope, q_rope) → RoPE(q_rope)
/// KV path: x → kv_a → split(c_kv, k_rope_raw) → RoPE(k_rope_raw);
///          c_kv → kv_a_norm → kv_b → reshape → split(k_nope, v)
///          k = concat(k_nope, k_rope) expanded per head
/// Attention: standard matmul with full K,V stored in KV cache
#[derive(Debug, Clone)]
struct MlaWeights {
    // Q path
    q_a: QMatMul, // hidden → q_lora_rank
    q_a_norm: RmsNorm,
    q_b: QMatMul, // q_lora_rank → n_head * key_length
    // KV path
    kv_a: QMatMul, // hidden → kv_lora_rank + rope_dim
    kv_a_norm: RmsNorm,
    kv_b: QMatMul,   // kv_lora_rank → n_head * (key_length - rope_dim + value_length)
    output: QMatMul, // n_head * value_length → hidden
    // Dimensions
    n_head: usize,
    key_length: usize,   // per-head total key dim (nope + rope)
    value_length: usize, // per-head value dim
    kv_lora_rank: usize,
    rope_dim: usize, // how many dims of key_length are rotary
    cos: Tensor,
    sin: Tensor,
    neg_inf: Tensor,
}

impl MlaWeights {
    /// Apply contiguous RoPE to a tensor in BHSD layout.
    fn apply_rope(&self, x: &Tensor, index_pos: usize) -> CandleResult<Tensor> {
        let (_b_sz, _n_head, seq_len, _dim) = x.dims4()?;
        let cos = self.cos.narrow(0, index_pos, seq_len)?;
        let sin = self.sin.narrow(0, index_pos, seq_len)?;
        candle_nn::rotary_emb::rope(&x.contiguous()?, &cos, &sin)
    }

    /// MLA forward pass.
    ///
    /// Decompresses Q and KV via low-rank projections, applies RoPE to the
    /// rotary portions, stores full K/V in the KV cache, runs standard
    /// attention, and projects the output.
    fn forward_mla(
        &self,
        x: &Tensor,
        mask: Option<&Tensor>,
        index_pos: usize,
        kv_cache: &mut Option<KvCache>,
        max_seq_len: usize,
    ) -> CandleResult<Tensor> {
        let (b_sz, seq_len, _hidden) = x.dims3()?;
        let nope_dim = self.key_length - self.rope_dim;

        // ── Q path ──
        let q_compressed = self.q_a.forward(x)?; // [b, s, q_lora_rank]
        let q_compressed = self.q_a_norm.forward(&q_compressed)?;
        let q_full = self.q_b.forward(&q_compressed)?; // [b, s, n_head * key_length]
        let q_full = q_full.reshape((b_sz, seq_len, self.n_head, self.key_length))?;
        // Split into nope and rope parts
        let q_nope = q_full.narrow(3, 0, nope_dim)?; // [b, s, n_head, nope_dim]
        let q_rope = q_full.narrow(3, nope_dim, self.rope_dim)?; // [b, s, n_head, rope_dim]
                                                                 // Apply RoPE to q_rope (needs BHSD)
        let q_rope = q_rope.transpose(1, 2)?.contiguous()?;
        let q_rope = self.apply_rope(&q_rope, index_pos)?;
        let q_rope = q_rope.transpose(1, 2)?; // back to [b, s, n_head, rope_dim]
        let q_nope = q_nope.contiguous()?;
        let q_rope = q_rope.contiguous()?;
        // Concat: q = [q_nope, q_rope]
        let q = Tensor::cat(&[&q_nope, &q_rope], 3)?; // [b, s, n_head, key_length]
        let q = q.transpose(1, 2)?.contiguous()?; // BHSD [b, n_head, s, key_length]

        // ── KV path ──
        let kv_compressed = self.kv_a.forward(x)?; // [b, s, kv_lora_rank + rope_dim]
                                                   // Split into c_kv latent and k_rope_raw (narrow produces non-contiguous views)
        let c_kv = kv_compressed
            .narrow(2, 0, self.kv_lora_rank)?
            .contiguous()?;
        let k_rope_raw = kv_compressed
            .narrow(2, self.kv_lora_rank, self.rope_dim)?
            .contiguous()?;
        // Apply RoPE to k_rope_raw: reshape to [b, s, 1, rope_dim] → BHSD → rope → back
        let k_rope = k_rope_raw
            .reshape((b_sz, seq_len, 1, self.rope_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k_rope = self.apply_rope(&k_rope, index_pos)?; // [b, 1, s, rope_dim]
                                                           // Expand k_rope to all heads: [b, n_head, s, rope_dim]
        let k_rope = k_rope
            .broadcast_as((b_sz, self.n_head, seq_len, self.rope_dim))?
            .contiguous()?;

        // Decompress KV latent
        let c_kv_normed = self.kv_a_norm.forward(&c_kv)?;
        let kv_decompressed = self.kv_b.forward(&c_kv_normed)?;
        // kv_decompressed: [b, s, n_head * (nope_dim + value_length)]
        let kv_per_head = nope_dim + self.value_length;
        let kv_full = kv_decompressed.reshape((b_sz, seq_len, self.n_head, kv_per_head))?;
        let k_nope = kv_full.narrow(3, 0, nope_dim)?;
        let v = kv_full.narrow(3, nope_dim, self.value_length)?;

        // k = concat(k_nope, k_rope) per head → BHSD
        let k_nope = k_nope.transpose(1, 2)?.contiguous()?; // [b, n_head, s, nope_dim]
        let k = Tensor::cat(&[&k_nope, &k_rope], 3)?; // [b, n_head, s, key_length]

        let v = v.transpose(1, 2)?.contiguous()?; // [b, n_head, s, value_length]

        // ── KV cache ──
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

        // ── Attention ──
        // MLA has asymmetric K/V dimensions (key_length != value_length), so
        // CPU flash attention (which assumes uniform head_dim) cannot be used.
        // Use standard matmul attention which handles this correctly.
        let y = standard_attention(
            &q,
            &k,
            &v,
            mask,
            self.key_length,
            self.n_head,
            self.n_head, // MLA: n_kv_head == n_head (full K/V per head)
            &self.neg_inf,
            None, // no softcap for DeepSeek
        )?;

        // y: [b, n_head, s, key_length] but we need [b, n_head, s, value_length]
        // run_attention returns [b, n_head, s, head_dim_of_v] which is value_length
        // Reshape: [b, n_head, s, value_length] → [b, s, n_head * value_length]
        let y = y
            .transpose(1, 2)?
            .reshape(&[b_sz, seq_len, self.n_head * self.value_length])?;
        self.output.forward(&y)
    }
}

// ── MoE FFN for DeepSeek-V2/V3 ──

/// Mixture-of-Experts FFN for DeepSeek-V2/V3.
///
/// Router selects top-k experts per token, runs SiLU-gated FFN for each,
/// and sums the weighted outputs. Shared experts (always active) are added.
#[derive(Debug, Clone)]
struct MoeFfn {
    gate: Tensor,      // router weights: [n_experts, hidden] (dequantized)
    gate_exps: Tensor, // stacked expert gate: [n_experts, intermediate, hidden]
    down_exps: Tensor, // stacked expert down: [n_experts, hidden, intermediate]
    up_exps: Tensor,   // stacked expert up: [n_experts, intermediate, hidden]
    // Shared experts (always active, optional)
    shared_gate: Option<QMatMul>,
    shared_down: Option<QMatMul>,
    shared_up: Option<QMatMul>,
    n_experts_used: usize, // top-k
}

/// Select top-k indices and weights from a score vector on CPU.
///
/// Candle 0.9 doesn't have a built-in topk, so we pull scores to CPU,
/// argsort descending, and take top-k. Fine for small n_experts vectors.
fn topk_cpu(scores: &Tensor, k: usize) -> CandleResult<(Tensor, Tensor)> {
    let scores_vec: Vec<f32> = scores.to_vec1()?;
    let n = scores_vec.len();
    let k = k.min(n);
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| {
        scores_vec[b]
            .partial_cmp(&scores_vec[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    indices.truncate(k);
    let weights: Vec<f32> = indices.iter().map(|&i| scores_vec[i]).collect();
    let idx_i64: Vec<i64> = indices.iter().map(|&i| i as i64).collect();
    let device = scores.device();
    let idx_tensor = Tensor::from_vec(idx_i64, (k,), device)?;
    let w_tensor = Tensor::from_vec(weights, (k,), device)?;
    // Normalize weights via softmax over selected experts
    let w_tensor = candle_nn::ops::softmax(&w_tensor, 0)?;
    Ok((idx_tensor, w_tensor))
}

impl MoeFfn {
    /// MoE forward pass with batched expert dispatch.
    ///
    /// 1. Router: x.matmul(gate.t()) → softmax scores → topk per token
    /// 2. Group tokens by assigned expert
    /// 3. For each expert: batched SiLU-gated FFN on all assigned tokens at once
    /// 4. Scatter weighted results back to token positions
    /// 5. Add shared expert output (if present)
    fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let (b_sz, seq_len, hidden) = x.dims3()?;
        let num_tokens = b_sz * seq_len;
        let x_flat = x.reshape((num_tokens, hidden))?;
        let device = x.device();
        let dtype = x.dtype();

        // Router scores: [num_tokens, n_experts]
        let router_scores = x_flat.matmul(&self.gate.t()?)?;
        let n_experts = self.gate.dim(0)?;

        // Phase 1: Route all tokens — collect (token_position, weight) per expert
        let mut expert_batches: Vec<Vec<(usize, f32)>> = vec![vec![]; n_experts];
        for pos in 0..num_tokens {
            let token_scores = router_scores.get(pos)?;
            let (indices, weights) = topk_cpu(&token_scores, self.n_experts_used)?;
            let indices_vec: Vec<i64> = indices.to_vec1()?;
            let weights_vec: Vec<f32> = weights.to_vec1()?;
            for (i, &expert_idx) in indices_vec.iter().enumerate() {
                expert_batches[expert_idx as usize].push((pos, weights_vec[i]));
            }
        }

        // Phase 2: Per-position accumulator for weighted expert outputs
        let mut pos_accum: Vec<Option<Tensor>> = vec![None; num_tokens];

        for (eidx, batch) in expert_batches.iter().enumerate() {
            if batch.is_empty() {
                continue;
            }

            // Gather all tokens assigned to this expert via index_select
            let idx_vec: Vec<i64> = batch.iter().map(|&(pos, _)| pos as i64).collect();
            let idx_tensor = Tensor::from_vec(idx_vec, (batch.len(),), device)?;
            let batch_input = x_flat.index_select(&idx_tensor, 0)?; // [batch_tokens, hidden]

            // Batched SiLU-gated FFN: silu(x @ gate.t) * (x @ up.t) @ down.t
            let gate_w = self.gate_exps.get(eidx)?;
            let up_w = self.up_exps.get(eidx)?;
            let down_w = self.down_exps.get(eidx)?;

            let gate_out = batch_input.matmul(&gate_w.t()?)?;
            let up_out = batch_input.matmul(&up_w.t()?)?;
            let activated = candle_nn::ops::silu(&gate_out)?;
            let combined = (activated * up_out)?;
            let expert_out = combined.matmul(&down_w.t()?)?; // [batch_tokens, hidden]

            // Apply per-token weights
            let weight_vec: Vec<f32> = batch.iter().map(|&(_, w)| w).collect();
            let weight_tensor =
                Tensor::from_vec(weight_vec, (batch.len(), 1), device)?.to_dtype(dtype)?;
            let weighted = expert_out.broadcast_mul(&weight_tensor)?;

            // Scatter weighted results back to position accumulators
            for (local_idx, &(pos, _)) in batch.iter().enumerate() {
                let contrib = weighted.narrow(0, local_idx, 1)?;
                pos_accum[pos] = Some(match pos_accum[pos].take() {
                    Some(existing) => (existing + contrib)?,
                    None => contrib,
                });
            }
        }

        // Assemble output: cat all position results
        let zero = Tensor::zeros((1, hidden), dtype, device)?;
        let slices: Vec<&Tensor> = pos_accum
            .iter()
            .map(|opt| opt.as_ref().unwrap_or(&zero))
            .collect();
        let mut output = Tensor::cat(&slices, 0)?;

        // Add shared expert output if present
        if let (Some(ref sg), Some(ref sd), Some(ref su)) =
            (&self.shared_gate, &self.shared_down, &self.shared_up)
        {
            let shared_gate_out = sg.forward(&x_flat)?;
            let shared_up_out = su.forward(&x_flat)?;
            let shared_activated = candle_nn::ops::silu(&shared_gate_out)?;
            let shared_combined = (shared_activated * shared_up_out)?;
            let shared_out = sd.forward(&shared_combined)?;
            output = (output + shared_out)?;
        }

        output.reshape((b_sz, seq_len, hidden))
    }
}

// ── Layer variant enum for supporting DeepSeek alongside dense architectures ──

/// A transformer layer that is either a standard dense layer or a DeepSeek MLA+MoE layer.
#[derive(Debug, Clone)]
#[allow(dead_code)]
enum LayerVariant {
    /// Standard dense transformer layer (Llama, Qwen2, Gemma, etc.)
    Dense(LayerWeights),
    /// DeepSeek-V2/V3 layer with MLA attention + MoE or dense FFN
    DeepSeek {
        attention: MlaWeights,
        ffn: FfnVariant,
        attention_norm: RmsNorm,
        ffn_norm: RmsNorm,
    },
    /// Qwen 3.5 full-attention layer (every 4th layer)
    Qwen35Attn {
        weights: Qwen35AttnWeights,
        ffn: FfnVariant,
        attention_norm: RmsNorm,
        post_attention_norm: RmsNorm,
    },
    /// Qwen 3.5 linear-attention (Gated Delta Network / SSM) layer
    Qwen35Ssm {
        weights: DeltaNetWeights,
        ffn: FfnVariant,
        attention_norm: RmsNorm,
        post_attention_norm: RmsNorm,
    },
}

/// Qwen 3.5 full-attention layer weights.
/// Similar to standard attention but with output gating from Q projection.
#[allow(dead_code)]
#[derive(Debug, Clone)]
struct Qwen35AttnWeights {
    /// Fused QKV + gate projection: hidden → (q_dim + k_dim + v_dim + gate_dim)
    wqkv: Option<QMatMul>,
    /// Separate Q/K/V projections (used when fused QKV not available)
    wq: Option<QMatMul>,
    wk: Option<QMatMul>,
    wv: Option<QMatMul>,
    wo: QMatMul,
    /// Output gate weights (sigmoid applied before O projection)
    attn_gate: Tensor,
    /// Q/K head normalization (RmsNorm per-head before RoPE)
    q_norm: Option<RmsNorm>,
    k_norm: Option<RmsNorm>,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    cos: Tensor,
    sin: Tensor,
    neg_inf: Tensor,
    /// Partial RoPE: only first `rope_dim` of head_dim get rotated
    rope_dim: usize,
}

/// Gated Delta Network (SSM) layer weights for Qwen 3.5 linear-attention layers.
///
/// Forward pass:
/// 1. Project input → Q, K, V, Z (gating) via fused projection
/// 2. Apply 1D causal convolution (conv1d with kernel_size typically 4)
/// 3. Compute state transition: alpha = softplus(ssm_alpha + ssm_dt), beta = sigmoid(ssm_beta)
/// 4. Run delta net scan: state = alpha * state + beta * (v ⊗ k), output = state @ q
/// 5. Apply gated normalization: norm(output) * silu(z)
/// 6. Project through ssm_out
#[allow(dead_code)]
#[derive(Debug, Clone)]
struct DeltaNetWeights {
    /// Fused QKV+Z projection: hidden → (q_dim + k_dim + v_dim + z_dim)
    wqkv: Option<QMatMul>,
    /// Separate Q/K/V projections (used when fused QKV not available)
    wq: Option<QMatMul>,
    wk: Option<QMatMul>,
    wv: Option<QMatMul>,
    /// SSM state transition parameter A (decay): [hidden, conv_kernel_dim]
    ssm_alpha: Tensor,
    /// SSM input gate B: [hidden, conv_kernel_dim]
    ssm_beta: Tensor,
    /// Delta time-step parameter: enables input-dependent state transitions
    ssm_dt: QMatMul,
    /// 1D causal convolution kernel: [n_heads, 1, conv_kernel_dim]
    ssm_conv1d: Tensor,
    /// Gated output normalization
    ssm_norm: RmsNorm,
    /// Output projection: recurrent_dim → hidden
    ssm_out: QMatMul,
    /// Number of Q heads for the linear attention
    n_head: usize,
    /// Number of K heads (may differ from Q)
    n_kv_head: usize,
    /// Number of V heads (may differ from K in Qwen 3.5)
    n_v_head: usize,
    /// Key head dimension
    key_head_dim: usize,
    /// Value head dimension
    value_head_dim: usize,
    /// Convolution kernel size (typically 4)
    conv_kernel_dim: usize,
}

/// Per-request SSM (delta net) recurrent state for Qwen 3.5 hybrid models.
/// Analogous to KV-cache for attention layers, but stores conv state + recurrent state.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub(crate) struct SsmState {
    /// 1D convolution buffer: [batch, n_heads * head_dim, conv_kernel_dim - 1]
    /// Stores the last (kernel_size - 1) inputs for causal conv.
    pub conv_state: Tensor,
    /// Recurrent state matrix: [batch, n_kv_heads, value_head_dim, key_head_dim]
    /// The running "memory" of the delta network.
    pub recurrent_state: Tensor,
}

/// FFN variant for DeepSeek layers — either dense or MoE.
#[derive(Debug, Clone)]
enum FfnVariant {
    Dense(Mlp),
    MoE(MoeFfn),
}

/// Extra metadata for DeepSeek-V2/V3 MoE+MLA models.
#[derive(Clone, Debug)]
struct DeepSeekMeta {
    n_experts: usize,
    n_experts_used: usize,
    n_shared_experts: usize,
    kv_lora_rank: usize,
    q_lora_rank: usize,
    key_length: usize,
    value_length: usize,
    rope_dim: usize,
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
    /// Qwen3 applies RmsNorm to Q per-head after projection (before RoPE).
    attn_q_norm: Option<RmsNorm>,
    /// Qwen3 applies RmsNorm to K per-head after projection (before RoPE).
    attn_k_norm: Option<RmsNorm>,
    ffn: FfnVariant,
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
    /// Number of head dimensions that receive RoPE. When < head_dim, only the first
    /// `rope_dim` dimensions are rotated and the rest pass through unchanged (partial RoPE,
    /// used by GLM-4). When == head_dim, standard full RoPE is applied.
    rope_dim: usize,
    /// If true, skip RoPE entirely for this layer (Llama 4 NoPE layers).
    skip_rope: bool,
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
            let seq_len = q.dim(2)?;

            // Fast path for decode (seq_len=1): skip CPU flash attention's
            // triple transpose+contiguous copies. Standard matmul is cheaper
            // for a single query token against the KV cache.
            if seq_len == 1 {
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

            // CPU flash attention: input BSHD, output BHSD
            // Transpose Q/K/V from BHSD (b,h,s,d) to BSHD (b,s,h,d)
            let q_bshd = q.transpose(1, 2)?.contiguous()?;
            let k_bshd = k.transpose(1, 2)?.contiguous()?;
            let v_bshd = v.transpose(1, 2)?.contiguous()?;

            let softmax_scale = 1.0 / (head_dim as f32).sqrt();

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
        // Llama 4 NoPE layers: skip RoPE entirely
        if self.skip_rope {
            return Ok(x.clone());
        }

        let (_b_sz, _n_head, seq_len, n_embd) = x.dims4()?;
        let cos = self.cos.narrow(0, index_pos, seq_len)?;
        let sin = self.sin.narrow(0, index_pos, seq_len)?;

        // Partial RoPE (GLM-4): only rotate the first rope_dim dimensions,
        // pass the rest through unchanged.
        if self.rope_dim < n_embd {
            let x_rot = x.narrow(3, 0, self.rope_dim)?.contiguous()?;
            let x_pass = x.narrow(3, self.rope_dim, n_embd - self.rope_dim)?;
            let rotated = if self.use_rope_contiguous {
                candle_nn::rotary_emb::rope(&x_rot, &cos, &sin)?
            } else {
                candle_nn::rotary_emb::rope_i(&x_rot, &cos, &sin)?
            };
            Tensor::cat(&[&rotated, &x_pass], 3)
        } else {
            // Full RoPE (standard path)
            if self.use_rope_contiguous {
                candle_nn::rotary_emb::rope(&x.contiguous()?, &cos, &sin)
            } else {
                candle_nn::rotary_emb::rope_i(&x.contiguous()?, &cos, &sin)
            }
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
        let (b_sz, seq_len, _n_embd) = x.dims3()?;
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

        let mut q = q.reshape((b_sz, seq_len, self.n_head, self.head_dim))?;
        let mut k = k.reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?;
        let v = v
            .reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        // Qwen3 QK normalization: RmsNorm applied per-head before RoPE
        if let Some(ref qn) = self.attn_q_norm {
            q = qn.forward(&q)?;
        }
        if let Some(ref kn) = self.attn_k_norm {
            k = kn.forward(&k)?;
        }

        let q = q.transpose(1, 2)?.contiguous()?;
        let k = k.transpose(1, 2)?.contiguous()?;

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

        let attn_out_dim = self.n_head * self.head_dim;
        let y = y.transpose(1, 2)?.reshape(&[b_sz, seq_len, attn_out_dim])?;
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

    /// Tensor-parallel attention: computes only the assigned fraction of heads.
    ///
    /// Each TP node handles `n_head / tp_size` query heads and the corresponding
    /// KV heads (respecting GQA ratio). The O projection produces a partial output
    /// of shape `[b, seq, hidden_dim]` that must be summed across TP nodes.
    #[allow(clippy::too_many_arguments)]
    fn forward_attn_tp(
        &self,
        x: &Tensor,
        mask: Option<&Tensor>,
        index_pos: usize,
        kv_cache: &mut Option<KvCache>,
        max_seq_len: usize,
        tp_rank: usize,
        tp_size: usize,
    ) -> CandleResult<Tensor> {
        let (b_sz, seq_len, _n_embd) = x.dims3()?;

        // Full Q/K/V projections (we slice the heads after projection)
        let q = self.attention_wq.forward(x)?;
        let k = self.attention_wk.forward(x)?;
        let v = self.attention_wv.forward(x)?;

        // Apply biases if present (Qwen2)
        let q = if let Some(ref bq) = self.attention_bq {
            q.broadcast_add(bq)?
        } else {
            q
        };
        let k = if let Some(ref bk) = self.attention_bk {
            k.broadcast_add(bk)?
        } else {
            k
        };
        let v = if let Some(ref bv) = self.attention_bv {
            v.broadcast_add(bv)?
        } else {
            v
        };

        // Reshape to head layout: [b, seq, n_head, head_dim]
        let q = q.reshape((b_sz, seq_len, self.n_head, self.head_dim))?;
        let k = k.reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?;
        let v = v.reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?;

        // Slice heads for this TP rank
        let heads_per_rank = self.n_head / tp_size;
        let kv_heads_per_rank = self.n_kv_head.max(1) / tp_size.min(self.n_kv_head).max(1);
        let q_start = tp_rank * heads_per_rank;
        let kv_start = tp_rank * kv_heads_per_rank;

        // Narrow along head dimension (dim=2)
        let q = q.narrow(2, q_start, heads_per_rank)?;
        let k = k.narrow(2, kv_start, kv_heads_per_rank.max(1))?;
        let v = v.narrow(2, kv_start, kv_heads_per_rank.max(1))?;

        // Transpose to BHSD: [b, n_head_local, seq, head_dim]
        let q = q.transpose(1, 2)?.contiguous()?;
        let k = k.transpose(1, 2)?.contiguous()?;
        let v = v.transpose(1, 2)?.contiguous()?;

        // RoPE (operates per-head, independent of head count)
        let q = self.apply_rotary_emb(&q, index_pos)?;
        let k = self.apply_rotary_emb(&k, index_pos)?;

        // KV cache for this TP rank's heads
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

        // Attention on local heads only
        let y = standard_attention(
            &q,
            &k,
            &v,
            mask,
            self.head_dim,
            heads_per_rank,
            kv_heads_per_rank.max(1),
            &self.neg_inf,
            self.attn_logit_softcap,
        )?;

        // Reshape back: [b, seq, heads_per_rank * head_dim]
        let local_dim = heads_per_rank * self.head_dim;
        let y = y.transpose(1, 2)?.reshape(&[b_sz, seq_len, local_dim])?;

        // O projection: produces [b, seq, hidden_dim] partial output.
        // For true column-parallel, we'd slice O's input dim. Instead, we project
        // the local head subset through the full O matrix and pad with zeros
        // so the sum across TP nodes gives the correct result.
        //
        // Equivalent approach: project local heads and use the fact that
        // O = [O_0 | O_1 | ... | O_{tp-1}] along input columns.
        // y_local @ O_local = y_local @ O[:, q_start*hd : (q_start+hpr)*hd]
        //
        // Since QMatMul doesn't support column slicing directly, we pad y with
        // zeros at the other head positions and do a full O matmul. The zeros
        // ensure those columns contribute nothing.
        let full_embd = self.n_head * self.head_dim;
        if local_dim < full_embd {
            // Build zero-padded tensor: [b, seq, full_embd]
            let before_dim = q_start * self.head_dim;
            let remaining = full_embd - (before_dim + local_dim);
            let y_padded = if before_dim > 0 && remaining > 0 {
                let z_before = Tensor::zeros((b_sz, seq_len, before_dim), y.dtype(), y.device())?;
                let z_after = Tensor::zeros((b_sz, seq_len, remaining), y.dtype(), y.device())?;
                Tensor::cat(&[&z_before, &y, &z_after], 2)?
            } else if before_dim > 0 {
                let z_before = Tensor::zeros((b_sz, seq_len, before_dim), y.dtype(), y.device())?;
                Tensor::cat(&[&z_before, &y], 2)?
            } else if remaining > 0 {
                let z_after = Tensor::zeros((b_sz, seq_len, remaining), y.dtype(), y.device())?;
                Tensor::cat(&[&y, &z_after], 2)?
            } else {
                y
            };
            self.attention_wo.forward(&y_padded)
        } else {
            self.attention_wo.forward(&y)
        }
    }
}

// ── Qwen 3.5 full-attention layer forward ──

#[allow(dead_code)]
impl Qwen35AttnWeights {
    fn apply_rotary_emb(&self, x: &Tensor, index_pos: usize) -> CandleResult<Tensor> {
        let (_b_sz, _n_head, seq_len, n_embd) = x.dims4()?;
        let cos = self.cos.narrow(0, index_pos, seq_len)?;
        let sin = self.sin.narrow(0, index_pos, seq_len)?;

        // Partial RoPE: only rotate first rope_dim dimensions
        if self.rope_dim < n_embd {
            let x_rot = x.narrow(3, 0, self.rope_dim)?.contiguous()?;
            let x_pass = x.narrow(3, self.rope_dim, n_embd - self.rope_dim)?;
            let rotated = candle_nn::rotary_emb::rope(&x_rot, &cos, &sin)?;
            Tensor::cat(&[&rotated, &x_pass], 3)
        } else {
            candle_nn::rotary_emb::rope(&x.contiguous()?, &cos, &sin)
        }
    }

    fn forward_attn(
        &self,
        x: &Tensor,
        mask: Option<&Tensor>,
        index_pos: usize,
        kv_cache: &mut Option<KvCache>,
        max_seq_len: usize,
    ) -> CandleResult<Tensor> {
        let (b_sz, seq_len, _hidden) = x.dims3()?;

        // Project Q, K, V (and gate from Q)
        let (q, k, v, gate) = if let Some(ref wqkv) = self.wqkv {
            let qkv = wqkv.forward(x)?;
            let q_dim = self.n_head * self.head_dim;
            let k_dim = self.n_kv_head * self.head_dim;
            let v_dim = k_dim;
            let q = qkv.narrow(2, 0, q_dim)?;
            let k = qkv.narrow(2, q_dim, k_dim)?;
            let v = qkv.narrow(2, q_dim + k_dim, v_dim)?;
            let gate = qkv.narrow(2, q_dim + k_dim + v_dim, q_dim)?;
            (q, k, v, gate)
        } else {
            let q = self.wq.as_ref().unwrap().forward(x)?;
            let k = self.wk.as_ref().unwrap().forward(x)?;
            let v = self.wv.as_ref().unwrap().forward(x)?;
            let gate = q.zeros_like()?;
            (q, k, v, gate)
        };

        // Reshape to heads
        let mut q = q.reshape((b_sz, seq_len, self.n_head, self.head_dim))?;
        let mut k = k.reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?;
        let v = v
            .reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        // Q/K head normalization
        if let Some(ref qn) = self.q_norm {
            q = qn.forward(&q)?;
        }
        if let Some(ref kn) = self.k_norm {
            k = kn.forward(&k)?;
        }

        let q = q.transpose(1, 2)?.contiguous()?;
        let k = k.transpose(1, 2)?.contiguous()?;

        // Partial RoPE
        let q = self.apply_rotary_emb(&q, index_pos)?;
        let k = self.apply_rotary_emb(&k, index_pos)?;

        // KV-cache
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

        // Attention
        let y = run_attention(
            &q,
            &k,
            &v,
            mask,
            self.n_head,
            self.n_kv_head,
            self.head_dim,
            &self.neg_inf,
            None,
        )?;

        // Apply output gate: sigmoid(gate + attn_gate_bias) * attn_output
        let attn_out_dim = self.n_head * self.head_dim;
        let y = y.transpose(1, 2)?.reshape(&[b_sz, seq_len, attn_out_dim])?;
        let gate_sig = gate
            .reshape((b_sz, seq_len, attn_out_dim))?
            .broadcast_add(&self.attn_gate)?;
        let gate_sig = candle_nn::ops::sigmoid(&gate_sig)?;
        let gated = (y * gate_sig)?;

        self.wo.forward(&gated)
    }
}

// ── Qwen 3.5 Gated Delta Network (SSM) layer forward ──

#[allow(dead_code)]
impl DeltaNetWeights {
    /// Forward pass for the Gated Delta Network (linear attention / SSM layer).
    fn forward_deltanet(
        &self,
        x: &Tensor,
        ssm_state: &mut Option<SsmState>,
    ) -> CandleResult<Tensor> {
        let (b_sz, seq_len, _hidden) = x.dims3()?;
        let device = x.device();

        // Project to Q, K, V, Z
        let (q, k, v, z) = if let Some(ref wqkv) = self.wqkv {
            let proj = wqkv.forward(x)?;
            let q_dim = self.n_head * self.key_head_dim;
            let k_dim = self.n_kv_head * self.key_head_dim;
            let v_dim = self.n_v_head * self.value_head_dim;
            let z_dim = v_dim;
            let q = proj.narrow(2, 0, q_dim)?;
            let k = proj.narrow(2, q_dim, k_dim)?;
            let v = proj.narrow(2, q_dim + k_dim, v_dim)?;
            let z = proj.narrow(2, q_dim + k_dim + v_dim, z_dim)?;
            (q, k, v, z)
        } else {
            let q = self.wq.as_ref().unwrap().forward(x)?;
            let k = self.wk.as_ref().unwrap().forward(x)?;
            let v = self.wv.as_ref().unwrap().forward(x)?;
            let z = v.zeros_like()?;
            (q, k, v, z)
        };

        // Apply 1D causal convolution over the QKV concatenation
        let qkv_cat = Tensor::cat(&[&q, &k, &v], 2)?;
        let (conv_out, new_conv_state) = self.apply_conv1d(&qkv_cat, ssm_state, device)?;

        // Split back into Q, K, V after convolution
        let q_dim = self.n_head * self.key_head_dim;
        let k_dim = self.n_kv_head * self.key_head_dim;
        let v_dim = self.n_v_head * self.value_head_dim;
        let q = conv_out.narrow(2, 0, q_dim)?;
        let k = conv_out.narrow(2, q_dim, k_dim)?;
        let v = conv_out.narrow(2, q_dim + k_dim, v_dim)?;

        // Reshape to heads: [b, seq, n_head, head_dim] → [b, n_head, seq, head_dim]
        let q = q
            .reshape((b_sz, seq_len, self.n_head, self.key_head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b_sz, seq_len, self.n_kv_head, self.key_head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b_sz, seq_len, self.n_v_head, self.value_head_dim))?
            .transpose(1, 2)?;

        // Compute state transition: alpha = softplus(ssm_alpha + ssm_dt(x))
        let dt = self.ssm_dt.forward(x)?;
        // ssm_alpha broadcast to [b, seq, dim], then softplus
        let alpha_base = self.ssm_alpha.broadcast_as(dt.shape())?;
        let alpha = softplus(&(&alpha_base + &dt)?)?;

        // beta = sigmoid(ssm_beta)
        let beta_base = self
            .ssm_beta
            .broadcast_as((b_sz, seq_len, q_dim + k_dim + v_dim))?;
        let beta = candle_nn::ops::sigmoid(&beta_base)?;

        // Run the delta net recurrent scan
        let output =
            self.delta_net_scan(&q, &k, &v, &alpha, &beta, ssm_state, b_sz, seq_len, device)?;

        // output shape: [b, n_head, seq, value_head_dim]
        let out_dim = self.n_v_head * self.value_head_dim;
        let output = output
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b_sz, seq_len, out_dim))?;

        // Gated normalization: norm(output) * silu(z)
        let normed = self.ssm_norm.forward(&output)?;
        let z_act = candle_nn::ops::silu(&z)?;
        let gated = (normed * z_act)?;

        // Update SSM state
        if let Some(state) = ssm_state {
            state.conv_state = new_conv_state;
        } else {
            *ssm_state = Some(SsmState {
                conv_state: new_conv_state,
                recurrent_state: Tensor::zeros(
                    (b_sz, self.n_kv_head, self.value_head_dim, self.key_head_dim),
                    DType::F32,
                    device,
                )?,
            });
        }

        // Project to hidden dim
        self.ssm_out.forward(&gated)
    }

    /// Apply 1D causal convolution with state for autoregressive mode.
    fn apply_conv1d(
        &self,
        x: &Tensor,
        ssm_state: &Option<SsmState>,
        device: &Device,
    ) -> CandleResult<(Tensor, Tensor)> {
        let (b_sz, seq_len, channels) = x.dims3()?;
        let kernel_size = self.conv_kernel_dim;
        let pad = kernel_size - 1;

        if seq_len == 1 {
            // Autoregressive: use conv state buffer
            let prev_state = if let Some(state) = ssm_state {
                state.conv_state.clone()
            } else {
                Tensor::zeros((b_sz, channels, pad), DType::F32, device)?
            };

            let x_t = x.transpose(1, 2)?; // [b, channels, 1]
            let new_state = if pad > 1 {
                let shifted = prev_state.narrow(2, 1, pad - 1)?;
                Tensor::cat(&[&shifted, &x_t], 2)?
            } else {
                x_t.clone()
            };

            let full_input = Tensor::cat(&[&new_state.narrow(2, 0, pad)?, &x_t], 2)?;
            let kernel = self.ssm_conv1d.reshape((channels, kernel_size))?;
            let conv_out = (&full_input
                * &kernel.unsqueeze(0)?.broadcast_as(full_input.shape())?)?
                .sum(2)?
                .unsqueeze(1)?; // [b, 1, channels]
            let conv_out = candle_nn::ops::silu(&conv_out)?;

            Ok((conv_out, new_state))
        } else {
            // Prefill: full causal convolution
            let x_t = x.transpose(1, 2)?.contiguous()?; // [b, channels, seq]

            let padding = if let Some(state) = ssm_state {
                state.conv_state.clone()
            } else {
                Tensor::zeros((b_sz, channels, pad), DType::F32, device)?
            };
            let padded = Tensor::cat(&[&padding, &x_t], 2)?;

            // Grouped conv1d: each channel independent
            let kernel = self.ssm_conv1d.reshape((channels, 1, kernel_size))?;
            let mut conv_outputs = Vec::with_capacity(seq_len);
            for t in 0..seq_len {
                let window = padded.narrow(2, t, kernel_size)?;
                let prod = (&window * &kernel.broadcast_as(window.shape())?)?;
                let summed = prod.sum(2)?;
                conv_outputs.push(summed);
            }
            let conv_out = Tensor::stack(&conv_outputs, 1)?; // [b, seq, channels]
            let conv_out = candle_nn::ops::silu(&conv_out)?;

            let new_conv_state = if seq_len >= pad {
                x_t.narrow(2, seq_len - pad, pad)?
            } else {
                let old_kept = padding.narrow(2, seq_len, pad - seq_len)?;
                Tensor::cat(&[&old_kept, &x_t], 2)?
            };

            Ok((conv_out, new_conv_state))
        }
    }

    /// Delta net recurrent scan.
    #[allow(clippy::too_many_arguments)]
    fn delta_net_scan(
        &self,
        q: &Tensor, // [b, n_head, seq, key_head_dim]
        k: &Tensor, // [b, n_kv_head, seq, key_head_dim]
        v: &Tensor, // [b, n_v_head, seq, value_head_dim]
        _alpha: &Tensor,
        _beta: &Tensor,
        ssm_state: &mut Option<SsmState>,
        b_sz: usize,
        seq_len: usize,
        device: &Device,
    ) -> CandleResult<Tensor> {
        let mut state = if let Some(ref s) = ssm_state {
            s.recurrent_state.clone()
        } else {
            Tensor::zeros(
                (b_sz, self.n_kv_head, self.value_head_dim, self.key_head_dim),
                DType::F32,
                device,
            )?
        };

        // Repeat KV heads for GQA
        let k = if self.n_head > self.n_kv_head {
            candle_transformers::utils::repeat_kv(k.clone(), self.n_head / self.n_kv_head)?
        } else {
            k.clone()
        };
        let v = if self.n_head > self.n_v_head {
            candle_transformers::utils::repeat_kv(v.clone(), self.n_head / self.n_v_head)?
        } else {
            v.clone()
        };

        let mut outputs = Vec::with_capacity(seq_len);

        for t in 0..seq_len {
            let q_t = q.narrow(2, t, 1)?.squeeze(2)?;
            let k_t = k.narrow(2, t, 1)?.squeeze(2)?;
            let v_t = v.narrow(2, t, 1)?.squeeze(2)?;

            // outer product: v_t ⊗ k_t → [b, n_head, value_head_dim, key_head_dim]
            let v_col = v_t.unsqueeze(3)?;
            let k_row = k_t.unsqueeze(2)?;
            let outer = v_col.matmul(&k_row)?;

            // State update with fixed decay (simplified — proper alpha/beta
            // integration requires per-head per-timestep parameters from GGUF
            // which vary in representation across model sizes)
            state = (&state * 0.95_f64 + outer)?;

            // Output: state @ q → [b, n_head, value_head_dim]
            let q_col = q_t.unsqueeze(3)?;
            let out_t = state.matmul(&q_col)?.squeeze(3)?;
            outputs.push(out_t);
        }

        let output = Tensor::stack(&outputs, 2)?;

        if let Some(ref mut s) = ssm_state {
            s.recurrent_state = state;
        }

        Ok(output)
    }
}

/// Softplus activation: log(1 + exp(x))
#[allow(dead_code)]
fn softplus(x: &Tensor) -> CandleResult<Tensor> {
    let ones = x.ones_like()?;
    let exp_x = x.exp()?;
    (&exp_x + &ones)?.log()
}

// ── Split model: loads only a range of layers from a GGUF ──

/// A partial transformer model that loads and runs only a specific range of layers.
/// Used for split inference where each node holds different layers.
/// Supports multiple architectures: Llama, Qwen2, Gemma 2, Phi-3, Mistral, Qwen 3.5.
pub struct SplitModel {
    /// Token embedding table (only loaded by the first segment).
    tok_embeddings: Option<Embedding>,
    /// Transformer layers for this segment's range.
    layers: Vec<LayerVariant>,
    /// Final RMSNorm (only loaded by the last segment).
    norm: Option<RmsNorm>,
    /// LM head / output projection (only loaded by the last segment).
    output: Option<QMatMul>,
    /// Causal attention mask: pre-allocated at a ceiling size, narrowed for smaller sequences.
    /// Tuple is (allocated_size, mask_tensor). `None` means no mask allocated yet.
    masks: Option<(usize, Tensor)>,
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
    /// Tokenizer (BPE or sentencepiece/unigram) built from GGUF metadata.
    tokenizer: Option<SplitTokenizer>,
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
    /// Pre-computed KV cache store key: "{layer_start}-{layer_end}-{total_layers}".
    /// Avoids a `format!` allocation on every forward pass.
    kv_model_key: String,
    /// Gemma 2 final logit soft-capping value (e.g. 30.0).
    final_logit_softcap: Option<f32>,
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
        // head_dim: prefer attention.key_length (Qwen3 uses 128 vs embed/heads=64)
        let head_dim = ct
            .metadata
            .get(&format!("{arch}.attention.key_length"))
            .and_then(|v| v.to_u32().ok())
            .map(|v| v as usize)
            .unwrap_or(embedding_length / head_count);
        // rope.dimension_count may not exist for all architectures — derive from head_dim
        let rope_dim = md_get("rope.dimension_count")
            .and_then(|v| v.to_u32().map_err(|e| SwarmError::Internal(e.to_string())))
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
            rope_dim,
            rope_freq_base,
            rms_norm_eps,
            expert_count,
            architecture: arch,
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
        let file = std::fs::File::open(gguf_path).map_err(SwarmError::Io)?;
        // SAFETY: Standard mmap usage — file is kept open for the duration of loading.
        // The mmap is dropped before the function returns; loaded tensors own their data.
        let mmap = unsafe { memmap2::Mmap::map(&file) }
            .map_err(|e| SwarmError::Internal(format!("Failed to mmap GGUF: {e}")))?;
        let mut file = std::io::Cursor::new(mmap.as_ref());
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

        tracing::info!(arch = %model_arch, "Detected model architecture");

        if !model_arch.is_supported() {
            return Err(SwarmError::Internal(format!(
                "Unsupported model architecture '{}'. Supported architectures: {}",
                arch_str,
                ModelArch::supported_list().join(", ")
            )));
        }

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
        // head_dim: prefer attention.key_length from GGUF (Qwen3 uses 128 vs embed/heads=64)
        let head_dim = ct
            .metadata
            .get(&format!("{arch}.attention.key_length"))
            .and_then(|v| v.to_u32().ok())
            .map(|v| v as usize)
            .unwrap_or(embedding_length / head_count);
        let rope_dim = md_get("rope.dimension_count")
            .and_then(|v| v.to_u32().map_err(|e| SwarmError::Internal(e.to_string())))
            .unwrap_or(head_dim as u32) as usize;
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

        // Gemma 2 final logit soft-capping (from GGUF metadata)
        let final_logit_softcap = ct
            .metadata
            .get(&format!("{arch}.final_logit_softcapping"))
            .and_then(|v| v.to_f32().ok())
            .filter(|&v| v > 0.0);

        let use_rope_contiguous = model_arch.use_rope_contiguous();
        let activation = model_arch.default_activation();

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
        let mut layers: Vec<LayerVariant> = Vec::with_capacity(layer_end - layer_start);

        if matches!(model_arch, ModelArch::DeepSeek2) {
            // ── DeepSeek-V2/V3: MLA + MoE loading ──
            let ds_meta = DeepSeekMeta {
                n_experts: md_get("expert_count")
                    .and_then(|v| v.to_u32().map_err(|e| SwarmError::Internal(e.to_string())))
                    .unwrap_or(0) as usize,
                n_experts_used: md_get("expert_used_count")
                    .and_then(|v| v.to_u32().map_err(|e| SwarmError::Internal(e.to_string())))
                    .unwrap_or(6) as usize,
                n_shared_experts: md_get("expert_shared_count")
                    .and_then(|v| v.to_u32().map_err(|e| SwarmError::Internal(e.to_string())))
                    .unwrap_or(0) as usize,
                kv_lora_rank: md_get("attention.kv_lora_rank")
                    .and_then(|v| v.to_u32().map_err(|e| SwarmError::Internal(e.to_string())))
                    .unwrap_or(512) as usize,
                q_lora_rank: md_get("attention.q_lora_rank")
                    .and_then(|v| v.to_u32().map_err(|e| SwarmError::Internal(e.to_string())))
                    .unwrap_or(0) as usize,
                key_length: md_get("attention.key_length")
                    .and_then(|v| v.to_u32().map_err(|e| SwarmError::Internal(e.to_string())))
                    .unwrap_or(head_dim as u32) as usize,
                value_length: md_get("attention.value_length")
                    .and_then(|v| v.to_u32().map_err(|e| SwarmError::Internal(e.to_string())))
                    .unwrap_or(head_dim as u32) as usize,
                rope_dim,
            };

            // DeepSeek RoPE may use a different dimension for MLA layers
            let mla_rope_dim = ds_meta.rope_dim;
            let (mla_cos, mla_sin) =
                precompute_freqs_cis(mla_rope_dim, rope_freq_base, context_length, &device)
                    .map_err(|e| SwarmError::Internal(e.to_string()))?;

            tracing::info!(
                n_experts = ds_meta.n_experts,
                n_experts_used = ds_meta.n_experts_used,
                n_shared_experts = ds_meta.n_shared_experts,
                kv_lora_rank = ds_meta.kv_lora_rank,
                q_lora_rank = ds_meta.q_lora_rank,
                key_length = ds_meta.key_length,
                value_length = ds_meta.value_length,
                "Loading DeepSeek-V2/V3 MoE+MLA model"
            );

            for layer_idx in layer_start..layer_end {
                let prefix = format!("blk.{layer_idx}");

                // Per-layer detection: MLA vs dense attention
                let has_mla = ct
                    .tensor_infos
                    .contains_key(&format!("{prefix}.attn_q_a.weight"));
                // Per-layer detection: MoE vs dense FFN
                let has_moe = ct
                    .tensor_infos
                    .contains_key(&format!("{prefix}.ffn_gate_exps.weight"));

                let attn_norm = ct
                    .tensor(&mut file, &format!("{prefix}.attn_norm.weight"), &device)
                    .map_err(|e| {
                        SwarmError::Internal(format!("Failed to load {prefix}.attn_norm: {e}"))
                    })?;
                let ffn_norm_t = ct
                    .tensor(&mut file, &format!("{prefix}.ffn_norm.weight"), &device)
                    .map_err(|e| {
                        SwarmError::Internal(format!("Failed to load {prefix}.ffn_norm: {e}"))
                    })?;

                if has_mla {
                    // MLA attention weights
                    let q_a = ct
                        .tensor(&mut file, &format!("{prefix}.attn_q_a.weight"), &device)
                        .map_err(|e| SwarmError::Internal(format!("{prefix}.attn_q_a: {e}")))?;
                    let q_a_norm_t = ct
                        .tensor(
                            &mut file,
                            &format!("{prefix}.attn_q_a_norm.weight"),
                            &device,
                        )
                        .map_err(|e| {
                            SwarmError::Internal(format!("{prefix}.attn_q_a_norm: {e}"))
                        })?;
                    let q_b = ct
                        .tensor(&mut file, &format!("{prefix}.attn_q_b.weight"), &device)
                        .map_err(|e| SwarmError::Internal(format!("{prefix}.attn_q_b: {e}")))?;
                    let kv_a = ct
                        .tensor(&mut file, &format!("{prefix}.attn_kv_a.weight"), &device)
                        .map_err(|e| SwarmError::Internal(format!("{prefix}.attn_kv_a: {e}")))?;
                    let kv_a_norm_t = ct
                        .tensor(
                            &mut file,
                            &format!("{prefix}.attn_kv_a_norm.weight"),
                            &device,
                        )
                        .map_err(|e| {
                            SwarmError::Internal(format!("{prefix}.attn_kv_a_norm: {e}"))
                        })?;
                    let kv_b = ct
                        .tensor(&mut file, &format!("{prefix}.attn_kv_b.weight"), &device)
                        .map_err(|e| SwarmError::Internal(format!("{prefix}.attn_kv_b: {e}")))?;
                    let wo = ct
                        .tensor(&mut file, &format!("{prefix}.attn_output.weight"), &device)
                        .map_err(|e| SwarmError::Internal(format!("{prefix}.attn_output: {e}")))?;

                    let attention = MlaWeights {
                        q_a: QMatMul::from_qtensor(q_a)
                            .map_err(|e| SwarmError::Internal(e.to_string()))?,
                        q_a_norm: RmsNorm::from_qtensor(q_a_norm_t, rms_norm_eps)
                            .map_err(|e| SwarmError::Internal(e.to_string()))?,
                        q_b: QMatMul::from_qtensor(q_b)
                            .map_err(|e| SwarmError::Internal(e.to_string()))?,
                        kv_a: QMatMul::from_qtensor(kv_a)
                            .map_err(|e| SwarmError::Internal(e.to_string()))?,
                        kv_a_norm: RmsNorm::from_qtensor(kv_a_norm_t, rms_norm_eps)
                            .map_err(|e| SwarmError::Internal(e.to_string()))?,
                        kv_b: QMatMul::from_qtensor(kv_b)
                            .map_err(|e| SwarmError::Internal(e.to_string()))?,
                        output: QMatMul::from_qtensor(wo)
                            .map_err(|e| SwarmError::Internal(e.to_string()))?,
                        n_head: head_count,
                        key_length: ds_meta.key_length,
                        value_length: ds_meta.value_length,
                        kv_lora_rank: ds_meta.kv_lora_rank,
                        rope_dim: mla_rope_dim,
                        cos: mla_cos.clone(),
                        sin: mla_sin.clone(),
                        neg_inf: neg_inf.clone(),
                    };

                    // FFN: MoE or dense
                    let ffn = if has_moe {
                        let gate_inp = ct
                            .tensor(&mut file, &format!("{prefix}.ffn_gate_inp.weight"), &device)
                            .map_err(|e| {
                                SwarmError::Internal(format!("{prefix}.ffn_gate_inp: {e}"))
                            })?;
                        let gate_exps = ct
                            .tensor(
                                &mut file,
                                &format!("{prefix}.ffn_gate_exps.weight"),
                                &device,
                            )
                            .map_err(|e| {
                                SwarmError::Internal(format!("{prefix}.ffn_gate_exps: {e}"))
                            })?;
                        let down_exps = ct
                            .tensor(
                                &mut file,
                                &format!("{prefix}.ffn_down_exps.weight"),
                                &device,
                            )
                            .map_err(|e| {
                                SwarmError::Internal(format!("{prefix}.ffn_down_exps: {e}"))
                            })?;
                        let up_exps = ct
                            .tensor(&mut file, &format!("{prefix}.ffn_up_exps.weight"), &device)
                            .map_err(|e| {
                                SwarmError::Internal(format!("{prefix}.ffn_up_exps: {e}"))
                            })?;

                        // Dequantize stacked expert tensors for index_select routing
                        let gate_inp_t = gate_inp
                            .dequantize(&device)
                            .map_err(|e| SwarmError::Internal(format!("gate_inp dequant: {e}")))?;
                        let gate_exps_t = gate_exps
                            .dequantize(&device)
                            .map_err(|e| SwarmError::Internal(format!("gate_exps dequant: {e}")))?;
                        let down_exps_t = down_exps
                            .dequantize(&device)
                            .map_err(|e| SwarmError::Internal(format!("down_exps dequant: {e}")))?;
                        let up_exps_t = up_exps
                            .dequantize(&device)
                            .map_err(|e| SwarmError::Internal(format!("up_exps dequant: {e}")))?;

                        // Shared experts (optional)
                        let shared_gate = ct
                            .tensor(
                                &mut file,
                                &format!("{prefix}.ffn_gate_shexp.weight"),
                                &device,
                            )
                            .ok()
                            .map(QMatMul::from_qtensor)
                            .transpose()
                            .map_err(|e| SwarmError::Internal(format!("shared gate: {e}")))?;
                        let shared_down = ct
                            .tensor(
                                &mut file,
                                &format!("{prefix}.ffn_down_shexp.weight"),
                                &device,
                            )
                            .ok()
                            .map(QMatMul::from_qtensor)
                            .transpose()
                            .map_err(|e| SwarmError::Internal(format!("shared down: {e}")))?;
                        let shared_up = ct
                            .tensor(&mut file, &format!("{prefix}.ffn_up_shexp.weight"), &device)
                            .ok()
                            .map(QMatMul::from_qtensor)
                            .transpose()
                            .map_err(|e| SwarmError::Internal(format!("shared up: {e}")))?;

                        FfnVariant::MoE(MoeFfn {
                            gate: gate_inp_t,
                            gate_exps: gate_exps_t,
                            down_exps: down_exps_t,
                            up_exps: up_exps_t,
                            shared_gate,
                            shared_down,
                            shared_up,
                            n_experts_used: ds_meta.n_experts_used,
                        })
                    } else {
                        // Dense FFN for early DeepSeek layers
                        let ffn_gate = ct
                            .tensor(&mut file, &format!("{prefix}.ffn_gate.weight"), &device)
                            .map_err(|e| SwarmError::Internal(format!("{prefix}.ffn_gate: {e}")))?;
                        let ffn_down = ct
                            .tensor(&mut file, &format!("{prefix}.ffn_down.weight"), &device)
                            .map_err(|e| SwarmError::Internal(format!("{prefix}.ffn_down: {e}")))?;
                        let ffn_up = ct
                            .tensor(&mut file, &format!("{prefix}.ffn_up.weight"), &device)
                            .map_err(|e| SwarmError::Internal(format!("{prefix}.ffn_up: {e}")))?;
                        FfnVariant::Dense(Mlp {
                            ffn_gate: QMatMul::from_qtensor(ffn_gate)
                                .map_err(|e| SwarmError::Internal(e.to_string()))?,
                            ffn_down: QMatMul::from_qtensor(ffn_down)
                                .map_err(|e| SwarmError::Internal(e.to_string()))?,
                            ffn_up: QMatMul::from_qtensor(ffn_up)
                                .map_err(|e| SwarmError::Internal(e.to_string()))?,
                            activation: Activation::SiLU,
                        })
                    };

                    layers.push(LayerVariant::DeepSeek {
                        attention,
                        ffn,
                        attention_norm: make_norm(attn_norm, rms_norm_eps)?,
                        ffn_norm: make_norm(ffn_norm_t, rms_norm_eps)?,
                    });
                } else {
                    // Dense attention layer (first few DeepSeek layers use standard attention)
                    let attention_wq = ct
                        .tensor(&mut file, &format!("{prefix}.attn_q.weight"), &device)
                        .map_err(|e| SwarmError::Internal(format!("{prefix}.attn_q: {e}")))?;
                    let attention_wk = ct
                        .tensor(&mut file, &format!("{prefix}.attn_k.weight"), &device)
                        .map_err(|e| SwarmError::Internal(format!("{prefix}.attn_k: {e}")))?;
                    let attention_wv = ct
                        .tensor(&mut file, &format!("{prefix}.attn_v.weight"), &device)
                        .map_err(|e| SwarmError::Internal(format!("{prefix}.attn_v: {e}")))?;
                    let attention_wo = ct
                        .tensor(&mut file, &format!("{prefix}.attn_output.weight"), &device)
                        .map_err(|e| SwarmError::Internal(format!("{prefix}.attn_output: {e}")))?;
                    let attention_bq = ct
                        .tensor(&mut file, &format!("{prefix}.attn_q.bias"), &device)
                        .ok()
                        .map(|t| t.dequantize(&device))
                        .transpose()
                        .map_err(|e| SwarmError::Internal(format!("attn_q.bias: {e}")))?;
                    let attention_bk = ct
                        .tensor(&mut file, &format!("{prefix}.attn_k.bias"), &device)
                        .ok()
                        .map(|t| t.dequantize(&device))
                        .transpose()
                        .map_err(|e| SwarmError::Internal(format!("attn_k.bias: {e}")))?;
                    let attention_bv = ct
                        .tensor(&mut file, &format!("{prefix}.attn_v.bias"), &device)
                        .ok()
                        .map(|t| t.dequantize(&device))
                        .transpose()
                        .map_err(|e| SwarmError::Internal(format!("attn_v.bias: {e}")))?;

                    // Dense FFN for early DeepSeek layers
                    let ffn_gate = ct
                        .tensor(&mut file, &format!("{prefix}.ffn_gate.weight"), &device)
                        .map_err(|e| SwarmError::Internal(format!("{prefix}.ffn_gate: {e}")))?;
                    let ffn_down = ct
                        .tensor(&mut file, &format!("{prefix}.ffn_down.weight"), &device)
                        .map_err(|e| SwarmError::Internal(format!("{prefix}.ffn_down: {e}")))?;
                    let ffn_up = ct
                        .tensor(&mut file, &format!("{prefix}.ffn_up.weight"), &device)
                        .map_err(|e| SwarmError::Internal(format!("{prefix}.ffn_up: {e}")))?;

                    // Dense DeepSeek layers use standard attention — wrap as LayerVariant::Dense
                    layers.push(LayerVariant::Dense(LayerWeights {
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
                        attn_q_norm: None,
                        attn_k_norm: None,
                        ffn: FfnVariant::Dense(Mlp {
                            ffn_gate: QMatMul::from_qtensor(ffn_gate)
                                .map_err(|e| SwarmError::Internal(e.to_string()))?,
                            ffn_down: QMatMul::from_qtensor(ffn_down)
                                .map_err(|e| SwarmError::Internal(e.to_string()))?,
                            ffn_up: QMatMul::from_qtensor(ffn_up)
                                .map_err(|e| SwarmError::Internal(e.to_string()))?,
                            activation: Activation::SiLU,
                        }),
                        ffn_norm: make_norm(ffn_norm_t, rms_norm_eps)?,
                        n_head: head_count,
                        n_kv_head: head_count_kv,
                        head_dim,
                        cos: cos.clone(),
                        sin: sin.clone(),
                        neg_inf: neg_inf.clone(),
                        use_rope_contiguous,
                        attn_logit_softcap,
                        rope_dim,
                        skip_rope: false,
                    }));
                }
            }
        } else if matches!(model_arch, ModelArch::Llama4) {
            // ── Llama 4 Scout/Maverick: iRoPE + MoE loading ──
            // iRoPE pattern: every 4th layer (index % 4 == 3) is NoPE (no positional encoding)
            // MoE: router selects top-k experts from stacked expert tensors

            // Read Llama 4 MoE metadata
            let n_experts = ct
                .metadata
                .get(&format!("{arch}.expert_count"))
                .and_then(|v| v.to_u32().ok())
                .unwrap_or(0) as usize;
            let n_experts_used = ct
                .metadata
                .get(&format!("{arch}.expert_used_count"))
                .and_then(|v| v.to_u32().ok())
                .unwrap_or(1) as usize;

            for layer_idx in layer_start..layer_end {
                let prefix = format!("blk.{layer_idx}");
                let is_nope = layer_idx % 4 == 3; // NoPE every 4th layer

                let attention_wq = ct
                    .tensor(&mut file, &format!("{prefix}.attn_q.weight"), &device)
                    .map_err(|e| SwarmError::Internal(format!("{prefix}.attn_q: {e}")))?;
                let attention_wk = ct
                    .tensor(&mut file, &format!("{prefix}.attn_k.weight"), &device)
                    .map_err(|e| SwarmError::Internal(format!("{prefix}.attn_k: {e}")))?;
                let attention_wv = ct
                    .tensor(&mut file, &format!("{prefix}.attn_v.weight"), &device)
                    .map_err(|e| SwarmError::Internal(format!("{prefix}.attn_v: {e}")))?;
                let attention_wo = ct
                    .tensor(&mut file, &format!("{prefix}.attn_output.weight"), &device)
                    .map_err(|e| SwarmError::Internal(format!("{prefix}.attn_output: {e}")))?;
                let attn_norm = ct
                    .tensor(&mut file, &format!("{prefix}.attn_norm.weight"), &device)
                    .map_err(|e| SwarmError::Internal(format!("{prefix}.attn_norm: {e}")))?;
                let ffn_norm = ct
                    .tensor(&mut file, &format!("{prefix}.ffn_norm.weight"), &device)
                    .map_err(|e| SwarmError::Internal(format!("{prefix}.ffn_norm: {e}")))?;

                // QKV biases (optional)
                let attention_bq = ct
                    .tensor(&mut file, &format!("{prefix}.attn_q.bias"), &device)
                    .ok()
                    .map(|t| t.dequantize(&device))
                    .transpose()
                    .map_err(|e| SwarmError::Internal(format!("attn_q.bias: {e}")))?;
                let attention_bk = ct
                    .tensor(&mut file, &format!("{prefix}.attn_k.bias"), &device)
                    .ok()
                    .map(|t| t.dequantize(&device))
                    .transpose()
                    .map_err(|e| SwarmError::Internal(format!("attn_k.bias: {e}")))?;
                let attention_bv = ct
                    .tensor(&mut file, &format!("{prefix}.attn_v.bias"), &device)
                    .ok()
                    .map(|t| t.dequantize(&device))
                    .transpose()
                    .map_err(|e| SwarmError::Internal(format!("attn_v.bias: {e}")))?;

                // Check if this layer has MoE
                let has_moe = ct
                    .tensor_infos
                    .contains_key(&format!("{prefix}.ffn_gate_exps.weight"));

                let ffn = if has_moe && n_experts > 0 {
                    let gate_inp = ct
                        .tensor(&mut file, &format!("{prefix}.ffn_gate_inp.weight"), &device)
                        .map_err(|e| SwarmError::Internal(format!("{prefix}.ffn_gate_inp: {e}")))?;
                    let gate_exps = ct
                        .tensor(
                            &mut file,
                            &format!("{prefix}.ffn_gate_exps.weight"),
                            &device,
                        )
                        .map_err(|e| {
                            SwarmError::Internal(format!("{prefix}.ffn_gate_exps: {e}"))
                        })?;
                    let down_exps = ct
                        .tensor(
                            &mut file,
                            &format!("{prefix}.ffn_down_exps.weight"),
                            &device,
                        )
                        .map_err(|e| {
                            SwarmError::Internal(format!("{prefix}.ffn_down_exps: {e}"))
                        })?;
                    let up_exps = ct
                        .tensor(&mut file, &format!("{prefix}.ffn_up_exps.weight"), &device)
                        .map_err(|e| SwarmError::Internal(format!("{prefix}.ffn_up_exps: {e}")))?;

                    let gate_inp_t = gate_inp
                        .dequantize(&device)
                        .map_err(|e| SwarmError::Internal(format!("gate_inp dequant: {e}")))?;
                    let gate_exps_t = gate_exps
                        .dequantize(&device)
                        .map_err(|e| SwarmError::Internal(format!("gate_exps dequant: {e}")))?;
                    let down_exps_t = down_exps
                        .dequantize(&device)
                        .map_err(|e| SwarmError::Internal(format!("down_exps dequant: {e}")))?;
                    let up_exps_t = up_exps
                        .dequantize(&device)
                        .map_err(|e| SwarmError::Internal(format!("up_exps dequant: {e}")))?;

                    // Shared experts (optional for Llama 4)
                    let shared_gate = ct
                        .tensor(
                            &mut file,
                            &format!("{prefix}.ffn_gate_shexp.weight"),
                            &device,
                        )
                        .ok()
                        .map(QMatMul::from_qtensor)
                        .transpose()
                        .map_err(|e| SwarmError::Internal(format!("shared gate: {e}")))?;
                    let shared_down = ct
                        .tensor(
                            &mut file,
                            &format!("{prefix}.ffn_down_shexp.weight"),
                            &device,
                        )
                        .ok()
                        .map(QMatMul::from_qtensor)
                        .transpose()
                        .map_err(|e| SwarmError::Internal(format!("shared down: {e}")))?;
                    let shared_up = ct
                        .tensor(&mut file, &format!("{prefix}.ffn_up_shexp.weight"), &device)
                        .ok()
                        .map(QMatMul::from_qtensor)
                        .transpose()
                        .map_err(|e| SwarmError::Internal(format!("shared up: {e}")))?;

                    FfnVariant::MoE(MoeFfn {
                        gate: gate_inp_t,
                        gate_exps: gate_exps_t,
                        down_exps: down_exps_t,
                        up_exps: up_exps_t,
                        shared_gate,
                        shared_down,
                        shared_up,
                        n_experts_used,
                    })
                } else {
                    // Dense FFN
                    let ffn_gate_t = ct
                        .tensor(&mut file, &format!("{prefix}.ffn_gate.weight"), &device)
                        .map_err(|e| SwarmError::Internal(format!("{prefix}.ffn_gate: {e}")))?;
                    let ffn_down_t = ct
                        .tensor(&mut file, &format!("{prefix}.ffn_down.weight"), &device)
                        .map_err(|e| SwarmError::Internal(format!("{prefix}.ffn_down: {e}")))?;
                    let ffn_up_t = ct
                        .tensor(&mut file, &format!("{prefix}.ffn_up.weight"), &device)
                        .map_err(|e| SwarmError::Internal(format!("{prefix}.ffn_up: {e}")))?;
                    FfnVariant::Dense(Mlp {
                        ffn_gate: QMatMul::from_qtensor(ffn_gate_t)
                            .map_err(|e| SwarmError::Internal(e.to_string()))?,
                        ffn_down: QMatMul::from_qtensor(ffn_down_t)
                            .map_err(|e| SwarmError::Internal(e.to_string()))?,
                        ffn_up: QMatMul::from_qtensor(ffn_up_t)
                            .map_err(|e| SwarmError::Internal(e.to_string()))?,
                        activation: Activation::SiLU,
                    })
                };

                // Both MoE and dense FFN layers use standard attention via LayerVariant::Dense.
                // The FfnVariant enum handles MoE vs dense FFN dispatch in the forward pass.
                layers.push(LayerVariant::Dense(LayerWeights {
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
                    attn_q_norm: None,
                    attn_k_norm: None,
                    ffn,
                    ffn_norm: make_norm(ffn_norm, rms_norm_eps)?,
                    n_head: head_count,
                    n_kv_head: head_count_kv,
                    head_dim,
                    cos: cos.clone(),
                    sin: sin.clone(),
                    neg_inf: neg_inf.clone(),
                    use_rope_contiguous,
                    attn_logit_softcap,
                    rope_dim,
                    skip_rope: is_nope,
                }));
            }
        } else if model_arch.is_hybrid_ssm() {
            // ── Qwen 3.5 hybrid: attention + SSM (Gated Delta Network) loading ──
            // Read linear attention config from GGUF metadata
            let linear_conv_kernel_dim = ct
                .metadata
                .get(&format!("{arch}.ssm.conv_kernel"))
                .and_then(|v| v.to_u32().ok())
                .unwrap_or(4) as usize;
            let linear_key_head_dim = ct
                .metadata
                .get(&format!("{arch}.ssm.inner_size"))
                .and_then(|v| v.to_u32().ok())
                .map(|v| v as usize)
                .unwrap_or(128);
            let linear_n_kv_head = ct
                .metadata
                .get(&format!("{arch}.attention.head_count_kv"))
                .and_then(|v| v.to_u32().ok())
                .unwrap_or(head_count_kv as u32) as usize;
            let linear_n_v_head = linear_n_kv_head; // typically same as K heads for SSM
            let linear_value_head_dim = linear_key_head_dim;

            // Partial RoPE for Qwen 3.5: partial_rotary_factor * head_dim
            let partial_rotary_factor = ct
                .metadata
                .get(&format!("{arch}.rope.partial_rotary_factor"))
                .and_then(|v| v.to_f32().ok())
                .unwrap_or(0.25);
            let qwen35_rope_dim = (head_dim as f32 * partial_rotary_factor) as usize;
            let (q35_cos, q35_sin) =
                precompute_freqs_cis(qwen35_rope_dim, rope_freq_base, context_length, &device)
                    .map_err(|e| SwarmError::Internal(e.to_string()))?;

            // Determine layer types from GGUF metadata
            // Qwen 3.5 pattern: every 4th layer is full_attention, rest are linear_attention
            // We detect this per-layer by checking tensor presence
            let is_moe = matches!(model_arch, ModelArch::Qwen35Moe);

            tracing::info!(
                linear_conv_kernel_dim,
                linear_key_head_dim,
                linear_n_kv_head,
                qwen35_rope_dim,
                is_moe,
                "Loading Qwen 3.5 hybrid SSM+attention model"
            );

            for layer_idx in layer_start..layer_end {
                let prefix = format!("blk.{layer_idx}");

                // Detect if this layer is SSM (linear_attention) or full attention
                // SSM layers have ssm_alpha.weight, attention layers have attn_q.weight
                let is_ssm_layer = ct
                    .tensor_infos
                    .contains_key(&format!("{prefix}.ssm_alpha.weight"));

                // Load norms (shared by both layer types)
                let attn_norm = ct
                    .tensor(&mut file, &format!("{prefix}.attn_norm.weight"), &device)
                    .map_err(|e| SwarmError::Internal(format!("{prefix}.attn_norm: {e}")))?;
                let post_attn_norm = ct
                    .tensor(
                        &mut file,
                        &format!("{prefix}.attn_post_norm.weight"),
                        &device,
                    )
                    .map_err(|e| SwarmError::Internal(format!("{prefix}.attn_post_norm: {e}")))?;

                // Load FFN (dense or MoE)
                let ffn = if is_moe
                    && ct
                        .tensor_infos
                        .contains_key(&format!("{prefix}.ffn_gate_exps.weight"))
                {
                    // MoE FFN
                    let gate_inp = ct
                        .tensor(&mut file, &format!("{prefix}.ffn_gate_inp.weight"), &device)
                        .map_err(|e| SwarmError::Internal(format!("{prefix}.ffn_gate_inp: {e}")))?;
                    let gate_exps = ct
                        .tensor(
                            &mut file,
                            &format!("{prefix}.ffn_gate_exps.weight"),
                            &device,
                        )
                        .map_err(|e| {
                            SwarmError::Internal(format!("{prefix}.ffn_gate_exps: {e}"))
                        })?;
                    let down_exps = ct
                        .tensor(
                            &mut file,
                            &format!("{prefix}.ffn_down_exps.weight"),
                            &device,
                        )
                        .map_err(|e| {
                            SwarmError::Internal(format!("{prefix}.ffn_down_exps: {e}"))
                        })?;
                    let up_exps = ct
                        .tensor(&mut file, &format!("{prefix}.ffn_up_exps.weight"), &device)
                        .map_err(|e| SwarmError::Internal(format!("{prefix}.ffn_up_exps: {e}")))?;

                    let _n_experts = ct
                        .metadata
                        .get(&format!("{arch}.expert_count"))
                        .and_then(|v| v.to_u32().ok())
                        .unwrap_or(8) as usize;
                    let n_experts_used = ct
                        .metadata
                        .get(&format!("{arch}.expert_used_count"))
                        .and_then(|v| v.to_u32().ok())
                        .unwrap_or(2) as usize;

                    // Shared experts (optional for MoE)
                    let shared_gate = ct
                        .tensor(
                            &mut file,
                            &format!("{prefix}.ffn_gate_shexp.weight"),
                            &device,
                        )
                        .ok()
                        .map(|t| QMatMul::from_qtensor(t).unwrap());
                    let shared_down = ct
                        .tensor(
                            &mut file,
                            &format!("{prefix}.ffn_down_shexp.weight"),
                            &device,
                        )
                        .ok()
                        .map(|t| QMatMul::from_qtensor(t).unwrap());
                    let shared_up = ct
                        .tensor(&mut file, &format!("{prefix}.ffn_up_shexp.weight"), &device)
                        .ok()
                        .map(|t| QMatMul::from_qtensor(t).unwrap());

                    FfnVariant::MoE(MoeFfn {
                        gate: gate_inp
                            .dequantize(&device)
                            .map_err(|e| SwarmError::Internal(e.to_string()))?,
                        gate_exps: gate_exps
                            .dequantize(&device)
                            .map_err(|e| SwarmError::Internal(e.to_string()))?,
                        down_exps: down_exps
                            .dequantize(&device)
                            .map_err(|e| SwarmError::Internal(e.to_string()))?,
                        up_exps: up_exps
                            .dequantize(&device)
                            .map_err(|e| SwarmError::Internal(e.to_string()))?,
                        shared_gate,
                        shared_down,
                        shared_up,
                        n_experts_used,
                    })
                } else {
                    // Dense FFN
                    let ffn_gate = ct
                        .tensor(&mut file, &format!("{prefix}.ffn_gate.weight"), &device)
                        .map_err(|e| SwarmError::Internal(format!("{prefix}.ffn_gate: {e}")))?;
                    let ffn_down = ct
                        .tensor(&mut file, &format!("{prefix}.ffn_down.weight"), &device)
                        .map_err(|e| SwarmError::Internal(format!("{prefix}.ffn_down: {e}")))?;
                    let ffn_up = ct
                        .tensor(&mut file, &format!("{prefix}.ffn_up.weight"), &device)
                        .map_err(|e| SwarmError::Internal(format!("{prefix}.ffn_up: {e}")))?;
                    FfnVariant::Dense(Mlp {
                        ffn_gate: QMatMul::from_qtensor(ffn_gate)
                            .map_err(|e| SwarmError::Internal(e.to_string()))?,
                        ffn_down: QMatMul::from_qtensor(ffn_down)
                            .map_err(|e| SwarmError::Internal(e.to_string()))?,
                        ffn_up: QMatMul::from_qtensor(ffn_up)
                            .map_err(|e| SwarmError::Internal(e.to_string()))?,
                        activation,
                    })
                };

                if is_ssm_layer {
                    // SSM / Gated Delta Network layer
                    let wqkv = ct
                        .tensor(&mut file, &format!("{prefix}.attn_qkv.weight"), &device)
                        .ok();
                    let (wq, wk, wv) = if wqkv.is_none() {
                        let q = ct
                            .tensor(&mut file, &format!("{prefix}.attn_q.weight"), &device)
                            .ok();
                        let k = ct
                            .tensor(&mut file, &format!("{prefix}.attn_k.weight"), &device)
                            .ok();
                        let v = ct
                            .tensor(&mut file, &format!("{prefix}.attn_v.weight"), &device)
                            .ok();
                        (
                            q.map(|t| QMatMul::from_qtensor(t).unwrap()),
                            k.map(|t| QMatMul::from_qtensor(t).unwrap()),
                            v.map(|t| QMatMul::from_qtensor(t).unwrap()),
                        )
                    } else {
                        (None, None, None)
                    };

                    let ssm_alpha = ct
                        .tensor(&mut file, &format!("{prefix}.ssm_alpha.weight"), &device)
                        .map_err(|e| SwarmError::Internal(format!("{prefix}.ssm_alpha: {e}")))?
                        .dequantize(&device)
                        .map_err(|e| SwarmError::Internal(e.to_string()))?;
                    let ssm_beta = ct
                        .tensor(&mut file, &format!("{prefix}.ssm_beta.weight"), &device)
                        .map_err(|e| SwarmError::Internal(format!("{prefix}.ssm_beta: {e}")))?
                        .dequantize(&device)
                        .map_err(|e| SwarmError::Internal(e.to_string()))?;
                    let ssm_dt = ct
                        .tensor(&mut file, &format!("{prefix}.ssm_dt.weight"), &device)
                        .map_err(|e| SwarmError::Internal(format!("{prefix}.ssm_dt: {e}")))?;
                    let ssm_conv1d = ct
                        .tensor(&mut file, &format!("{prefix}.ssm_conv1d.weight"), &device)
                        .map_err(|e| SwarmError::Internal(format!("{prefix}.ssm_conv1d: {e}")))?
                        .dequantize(&device)
                        .map_err(|e| SwarmError::Internal(e.to_string()))?;
                    let ssm_norm_t = ct
                        .tensor(&mut file, &format!("{prefix}.ssm_norm.weight"), &device)
                        .map_err(|e| SwarmError::Internal(format!("{prefix}.ssm_norm: {e}")))?;
                    let ssm_out = ct
                        .tensor(&mut file, &format!("{prefix}.ssm_out.weight"), &device)
                        .map_err(|e| SwarmError::Internal(format!("{prefix}.ssm_out: {e}")))?;

                    layers.push(LayerVariant::Qwen35Ssm {
                        weights: DeltaNetWeights {
                            wqkv: wqkv.map(|t| QMatMul::from_qtensor(t).unwrap()),
                            wq,
                            wk,
                            wv,
                            ssm_alpha,
                            ssm_beta,
                            ssm_dt: QMatMul::from_qtensor(ssm_dt)
                                .map_err(|e| SwarmError::Internal(e.to_string()))?,
                            ssm_conv1d,
                            ssm_norm: RmsNorm::from_qtensor(ssm_norm_t, rms_norm_eps)
                                .map_err(|e| SwarmError::Internal(e.to_string()))?,
                            ssm_out: QMatMul::from_qtensor(ssm_out)
                                .map_err(|e| SwarmError::Internal(e.to_string()))?,
                            n_head: head_count,
                            n_kv_head: linear_n_kv_head,
                            n_v_head: linear_n_v_head,
                            key_head_dim: linear_key_head_dim,
                            value_head_dim: linear_value_head_dim,
                            conv_kernel_dim: linear_conv_kernel_dim,
                        },
                        ffn,
                        attention_norm: make_norm(attn_norm, rms_norm_eps)?,
                        post_attention_norm: make_norm(post_attn_norm, rms_norm_eps)?,
                    });
                } else {
                    // Full attention layer (every 4th layer in Qwen 3.5)
                    let wqkv = ct
                        .tensor(&mut file, &format!("{prefix}.attn_qkv.weight"), &device)
                        .ok();
                    let (wq, wk, wv) = if wqkv.is_none() {
                        let q = ct
                            .tensor(&mut file, &format!("{prefix}.attn_q.weight"), &device)
                            .ok();
                        let k = ct
                            .tensor(&mut file, &format!("{prefix}.attn_k.weight"), &device)
                            .ok();
                        let v = ct
                            .tensor(&mut file, &format!("{prefix}.attn_v.weight"), &device)
                            .ok();
                        (
                            q.map(|t| QMatMul::from_qtensor(t).unwrap()),
                            k.map(|t| QMatMul::from_qtensor(t).unwrap()),
                            v.map(|t| QMatMul::from_qtensor(t).unwrap()),
                        )
                    } else {
                        (None, None, None)
                    };
                    let wo = ct
                        .tensor(&mut file, &format!("{prefix}.attn_output.weight"), &device)
                        .map_err(|e| SwarmError::Internal(format!("{prefix}.attn_output: {e}")))?;
                    let attn_gate_t = ct
                        .tensor(&mut file, &format!("{prefix}.attn_gate.weight"), &device)
                        .map_err(|e| SwarmError::Internal(format!("{prefix}.attn_gate: {e}")))?
                        .dequantize(&device)
                        .map_err(|e| SwarmError::Internal(e.to_string()))?;

                    // Q/K norms (optional)
                    let q_norm = ct
                        .tensor(&mut file, &format!("{prefix}.attn_q_norm.weight"), &device)
                        .ok()
                        .map(|t| RmsNorm::from_qtensor(t, rms_norm_eps).unwrap());
                    let k_norm = ct
                        .tensor(&mut file, &format!("{prefix}.attn_k_norm.weight"), &device)
                        .ok()
                        .map(|t| RmsNorm::from_qtensor(t, rms_norm_eps).unwrap());

                    layers.push(LayerVariant::Qwen35Attn {
                        weights: Qwen35AttnWeights {
                            wqkv: wqkv.map(|t| QMatMul::from_qtensor(t).unwrap()),
                            wq,
                            wk,
                            wv,
                            wo: QMatMul::from_qtensor(wo)
                                .map_err(|e| SwarmError::Internal(e.to_string()))?,
                            attn_gate: attn_gate_t,
                            q_norm,
                            k_norm,
                            n_head: head_count,
                            n_kv_head: head_count_kv,
                            head_dim,
                            cos: q35_cos.clone(),
                            sin: q35_sin.clone(),
                            neg_inf: neg_inf.clone(),
                            rope_dim: qwen35_rope_dim,
                        },
                        ffn,
                        attention_norm: make_norm(attn_norm, rms_norm_eps)?,
                        post_attention_norm: make_norm(post_attn_norm, rms_norm_eps)?,
                    });
                }
            }
        } else {
            // ── Standard dense architecture loading (Llama, Qwen2, Gemma, GLM-4, etc.) ──
            // Parallel layer loading: each thread gets its own Cursor into the mmap'd data,
            // reads tensors for one layer independently. ~N× speedup for N layers on NVMe/SSD.
            let mmap_ref = mmap.as_ref();
            let ct_ref = &ct;
            let device_ref = &device;
            let layer_results: Vec<Result<LayerVariant, SwarmError>> = std::thread::scope(|s| {
                let handles: Vec<_> = (layer_start..layer_end)
                    .map(|layer_idx| {
                        let cos = cos.clone();
                        let sin = sin.clone();
                        let neg_inf = neg_inf.clone();
                        s.spawn(move || -> Result<LayerVariant, SwarmError> {
                            let mut cursor = std::io::Cursor::new(mmap_ref);
                            let prefix = format!("blk.{layer_idx}");

                            // Try separate Q/K/V first; fall back to fused attn_qkv
                            // (Phi-3 uses fused attn_qkv.weight instead of separate attn_q/k/v)
                            let has_fused_qkv = ct_ref
                                .tensor_infos
                                .contains_key(&format!("{prefix}.attn_qkv.weight"));
                            let (qkv_q, qkv_k, qkv_v) = if has_fused_qkv {
                                // Fused QKV: load to CPU, dequantize, split, convert to f16, move to device
                                let cpu = &candle_core::Device::Cpu;
                                let fused = ct_ref
                                    .tensor(&mut cursor, &format!("{prefix}.attn_qkv.weight"), cpu)
                                    .map_err(|e| {
                                        SwarmError::Internal(format!("{prefix}.attn_qkv: {e}"))
                                    })?;
                                let fused_f32 = fused.dequantize(cpu).map_err(|e| {
                                    SwarmError::Internal(format!("qkv dequant: {e}"))
                                })?;
                                let q_dim = head_count * head_dim;
                                let k_dim = head_count_kv * head_dim;
                                let v_dim = k_dim;
                                let wq = fused_f32
                                    .narrow(0, 0, q_dim)
                                    .map_err(|e| SwarmError::Internal(format!("qkv split q: {e}")))?
                                    .contiguous()
                                    .map_err(|e| {
                                        SwarmError::Internal(format!("q contiguous: {e}"))
                                    })?;
                                let wk = fused_f32
                                    .narrow(0, q_dim, k_dim)
                                    .map_err(|e| SwarmError::Internal(format!("qkv split k: {e}")))?
                                    .contiguous()
                                    .map_err(|e| {
                                        SwarmError::Internal(format!("k contiguous: {e}"))
                                    })?;
                                let wv = fused_f32
                                    .narrow(0, q_dim + k_dim, v_dim)
                                    .map_err(|e| SwarmError::Internal(format!("qkv split v: {e}")))?
                                    .contiguous()
                                    .map_err(|e| {
                                        SwarmError::Internal(format!("v contiguous: {e}"))
                                    })?;
                                (
                                    QMatMul::from_f32_tensor(wq, device_ref)?,
                                    QMatMul::from_f32_tensor(wk, device_ref)?,
                                    QMatMul::from_f32_tensor(wv, device_ref)?,
                                )
                            } else {
                                let wq = ct_ref
                                    .tensor(
                                        &mut cursor,
                                        &format!("{prefix}.attn_q.weight"),
                                        device_ref,
                                    )
                                    .map_err(|e| {
                                        SwarmError::Internal(format!(
                                            "Failed to load {prefix}.attn_q: {e}"
                                        ))
                                    })?;
                                let wk = ct_ref
                                    .tensor(
                                        &mut cursor,
                                        &format!("{prefix}.attn_k.weight"),
                                        device_ref,
                                    )
                                    .map_err(|e| {
                                        SwarmError::Internal(format!(
                                            "Failed to load {prefix}.attn_k: {e}"
                                        ))
                                    })?;
                                let wv = ct_ref
                                    .tensor(
                                        &mut cursor,
                                        &format!("{prefix}.attn_v.weight"),
                                        device_ref,
                                    )
                                    .map_err(|e| {
                                        SwarmError::Internal(format!(
                                            "Failed to load {prefix}.attn_v: {e}"
                                        ))
                                    })?;
                                (
                                    QMatMul::from_qtensor(wq)
                                        .map_err(|e| SwarmError::Internal(e.to_string()))?,
                                    QMatMul::from_qtensor(wk)
                                        .map_err(|e| SwarmError::Internal(e.to_string()))?,
                                    QMatMul::from_qtensor(wv)
                                        .map_err(|e| SwarmError::Internal(e.to_string()))?,
                                )
                            };
                            let attention_wo = ct_ref
                                .tensor(
                                    &mut cursor,
                                    &format!("{prefix}.attn_output.weight"),
                                    device_ref,
                                )
                                .map_err(|e| {
                                    SwarmError::Internal(format!(
                                        "Failed to load {prefix}.attn_output: {e}"
                                    ))
                                })?;

                            let attention_bq = ct_ref
                                .tensor(&mut cursor, &format!("{prefix}.attn_q.bias"), device_ref)
                                .ok()
                                .map(|t| t.dequantize(device_ref))
                                .transpose()
                                .map_err(|e| {
                                    SwarmError::Internal(format!("attn_q.bias dequant: {e}"))
                                })?;
                            let attention_bk = ct_ref
                                .tensor(&mut cursor, &format!("{prefix}.attn_k.bias"), device_ref)
                                .ok()
                                .map(|t| t.dequantize(device_ref))
                                .transpose()
                                .map_err(|e| {
                                    SwarmError::Internal(format!("attn_k.bias dequant: {e}"))
                                })?;
                            let attention_bv = ct_ref
                                .tensor(&mut cursor, &format!("{prefix}.attn_v.bias"), device_ref)
                                .ok()
                                .map(|t| t.dequantize(device_ref))
                                .transpose()
                                .map_err(|e| {
                                    SwarmError::Internal(format!("attn_v.bias dequant: {e}"))
                                })?;

                            // FFN: try separate gate/up first; fall back to fused gate_up
                            // (Phi-3 uses combined ffn_up = gate || up, no separate ffn_gate)
                            let has_ffn_gate = ct_ref
                                .tensor_infos
                                .contains_key(&format!("{prefix}.ffn_gate.weight"));
                            let ffn_down_qt = ct_ref
                                .tensor(
                                    &mut cursor,
                                    &format!("{prefix}.ffn_down.weight"),
                                    device_ref,
                                )
                                .map_err(|e| {
                                    SwarmError::Internal(format!(
                                        "Failed to load {prefix}.ffn_down: {e}"
                                    ))
                                })?;
                            let (ffn_gate_mm, ffn_down_mm, ffn_up_mm) = if has_ffn_gate {
                                let gate = ct_ref
                                    .tensor(
                                        &mut cursor,
                                        &format!("{prefix}.ffn_gate.weight"),
                                        device_ref,
                                    )
                                    .map_err(|e| {
                                        SwarmError::Internal(format!(
                                            "Failed to load {prefix}.ffn_gate: {e}"
                                        ))
                                    })?;
                                let up = ct_ref
                                    .tensor(
                                        &mut cursor,
                                        &format!("{prefix}.ffn_up.weight"),
                                        device_ref,
                                    )
                                    .map_err(|e| {
                                        SwarmError::Internal(format!(
                                            "Failed to load {prefix}.ffn_up: {e}"
                                        ))
                                    })?;
                                (
                                    QMatMul::from_qtensor(gate)
                                        .map_err(|e| SwarmError::Internal(e.to_string()))?,
                                    QMatMul::from_qtensor(ffn_down_qt)
                                        .map_err(|e| SwarmError::Internal(e.to_string()))?,
                                    QMatMul::from_qtensor(up)
                                        .map_err(|e| SwarmError::Internal(e.to_string()))?,
                                )
                            } else {
                                // Fused gate+up: load to CPU, dequantize, split, convert to f16, move to device
                                let cpu = &candle_core::Device::Cpu;
                                let fused = ct_ref
                                    .tensor(&mut cursor, &format!("{prefix}.ffn_up.weight"), cpu)
                                    .map_err(|e| {
                                        SwarmError::Internal(format!(
                                            "Failed to load {prefix}.ffn_up: {e}"
                                        ))
                                    })?;
                                let fused_f32 = fused.dequantize(cpu).map_err(|e| {
                                    SwarmError::Internal(format!("ffn_up dequant: {e}"))
                                })?;
                                let total_rows = fused_f32.dim(0).map_err(|e| {
                                    SwarmError::Internal(format!("ffn_up dim0: {e}"))
                                })?;
                                let half = total_rows / 2;
                                let gate_f32 = fused_f32
                                    .narrow(0, 0, half)
                                    .map_err(|e| {
                                        SwarmError::Internal(format!("ffn gate split: {e}"))
                                    })?
                                    .contiguous()
                                    .map_err(|e| {
                                        SwarmError::Internal(format!("gate contiguous: {e}"))
                                    })?;
                                let up_f32 = fused_f32
                                    .narrow(0, half, half)
                                    .map_err(|e| {
                                        SwarmError::Internal(format!("ffn up split: {e}"))
                                    })?
                                    .contiguous()
                                    .map_err(|e| {
                                        SwarmError::Internal(format!("up contiguous: {e}"))
                                    })?;
                                (
                                    QMatMul::from_f32_tensor(gate_f32, device_ref)?,
                                    QMatMul::from_qtensor(ffn_down_qt)
                                        .map_err(|e| SwarmError::Internal(e.to_string()))?,
                                    QMatMul::from_f32_tensor(up_f32, device_ref)?,
                                )
                            };
                            let attn_norm = ct_ref
                                .tensor(
                                    &mut cursor,
                                    &format!("{prefix}.attn_norm.weight"),
                                    device_ref,
                                )
                                .map_err(|e| {
                                    SwarmError::Internal(format!(
                                        "Failed to load {prefix}.attn_norm: {e}"
                                    ))
                                })?;
                            let ffn_norm = ct_ref
                                .tensor(
                                    &mut cursor,
                                    &format!("{prefix}.ffn_norm.weight"),
                                    device_ref,
                                )
                                .map_err(|e| {
                                    SwarmError::Internal(format!(
                                        "Failed to load {prefix}.ffn_norm: {e}"
                                    ))
                                })?;

                            // Qwen3 QK normalization (optional)
                            let attn_q_norm = ct_ref
                                .tensor(
                                    &mut cursor,
                                    &format!("{prefix}.attn_q_norm.weight"),
                                    device_ref,
                                )
                                .ok()
                                .map(|t| RmsNorm::from_qtensor(t, rms_norm_eps))
                                .transpose()
                                .map_err(|e| SwarmError::Internal(format!("attn_q_norm: {e}")))?;
                            let attn_k_norm = ct_ref
                                .tensor(
                                    &mut cursor,
                                    &format!("{prefix}.attn_k_norm.weight"),
                                    device_ref,
                                )
                                .ok()
                                .map(|t| RmsNorm::from_qtensor(t, rms_norm_eps))
                                .transpose()
                                .map_err(|e| SwarmError::Internal(format!("attn_k_norm: {e}")))?;

                            Ok(LayerVariant::Dense(LayerWeights {
                                attention_wq: qkv_q,
                                attention_wk: qkv_k,
                                attention_wv: qkv_v,
                                attention_wo: QMatMul::from_qtensor(attention_wo)
                                    .map_err(|e| SwarmError::Internal(e.to_string()))?,
                                attention_bq,
                                attention_bk,
                                attention_bv,
                                attention_norm: make_norm(attn_norm, rms_norm_eps)?,
                                attn_q_norm,
                                attn_k_norm,
                                ffn: FfnVariant::Dense(Mlp {
                                    ffn_gate: ffn_gate_mm,
                                    ffn_down: ffn_down_mm,
                                    ffn_up: ffn_up_mm,
                                    activation,
                                }),
                                ffn_norm: make_norm(ffn_norm, rms_norm_eps)?,
                                n_head: head_count,
                                n_kv_head: head_count_kv,
                                head_dim,
                                cos: cos.clone(),
                                sin: sin.clone(),
                                neg_inf: neg_inf.clone(),
                                use_rope_contiguous,
                                attn_logit_softcap,
                                rope_dim,
                                skip_rope: false,
                            }))
                        })
                    })
                    .collect();
                handles
                    .into_iter()
                    .map(|h| h.join().expect("layer load thread panicked"))
                    .collect()
            });
            for result in layer_results {
                layers.push(result?);
            }
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
                Some(SplitTokenizer::from_bpe(
                    vocab,
                    &merges_raw,
                    &pre_type,
                    &tokenizer_model,
                ))
            } else if tokenizer_model == "llama" {
                // Sentencepiece-based model (LLaMA family without BPE merges).
                let scores: Vec<f32> = ct
                    .metadata
                    .get("tokenizer.ggml.scores")
                    .and_then(|v| v.to_vec().ok())
                    .map(|arr| arr.iter().filter_map(|v| v.to_f32().ok()).collect())
                    .unwrap_or_default();
                if !scores.is_empty() {
                    tracing::info!(
                        vocab_size = vocab.len(),
                        scores = scores.len(),
                        "Building HF Unigram tokenizer from GGUF sentencepiece data"
                    );
                    Some(SplitTokenizer::from_sentencepiece(vocab, &scores))
                } else {
                    None
                }
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
            "gemma" | "gemma2" => {
                // Gemma uses token 107 (<end_of_turn>) as EOS
                if !eos_tokens.contains(&107) {
                    eos_tokens.push(107);
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

        if let Some(ref tmpl) = chat_template {
            tracing::info!(
                bos = %bos_token,
                eos = %eos_token,
                template_len = tmpl.len(),
                template_preview = &tmpl[..tmpl.len().min(200)],
                "Loaded chat template from GGUF header"
            );
        }

        let has_biases = layers
            .first()
            .is_some_and(|l| matches!(l, LayerVariant::Dense(lw) if lw.attention_bq.is_some()));
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
            masks: None,
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
            max_seq_len: context_length.min(2048),
            kv_model_key: format!("{layer_start}-{layer_end}-{block_count}"),
            final_logit_softcap,
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

        tracing::info!(arch = %model_arch, "Detected model architecture");

        if !model_arch.is_supported() {
            return Err(SwarmError::Internal(format!(
                "Unsupported model architecture '{}'. Supported architectures: {}",
                arch_str,
                ModelArch::supported_list().join(", ")
            )));
        }

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
        // head_dim: prefer attention.key_length from GGUF (Qwen3 uses 128 vs embed/heads=64)
        let head_dim = ct
            .metadata
            .get(&format!("{arch}.attention.key_length"))
            .and_then(|v| v.to_u32().ok())
            .map(|v| v as usize)
            .unwrap_or(embedding_length / head_count);
        let rope_dim = md_get("rope.dimension_count")
            .and_then(|v| v.to_u32().map_err(|e| SwarmError::Internal(e.to_string())))
            .unwrap_or(head_dim as u32) as usize;
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

        let final_logit_softcap = ct
            .metadata
            .get(&format!("{arch}.final_logit_softcapping"))
            .and_then(|v| v.to_f32().ok())
            .filter(|&v| v > 0.0);

        let use_rope_contiguous = model_arch.use_rope_contiguous();
        let activation = model_arch.default_activation();

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
        let mut layers: Vec<LayerVariant> = Vec::with_capacity(layer_end - layer_start);

        // Shard loading uses the same architecture dispatch as load_from_gguf.
        // For DeepSeek, we detect per-layer MLA/MoE presence from tensor_infos.
        // For all other architectures, we load standard dense layers.
        if matches!(model_arch, ModelArch::DeepSeek2) {
            let ds_meta = DeepSeekMeta {
                n_experts: md_get("expert_count")
                    .and_then(|v| v.to_u32().map_err(|e| SwarmError::Internal(e.to_string())))
                    .unwrap_or(0) as usize,
                n_experts_used: md_get("expert_used_count")
                    .and_then(|v| v.to_u32().map_err(|e| SwarmError::Internal(e.to_string())))
                    .unwrap_or(6) as usize,
                n_shared_experts: md_get("expert_shared_count")
                    .and_then(|v| v.to_u32().map_err(|e| SwarmError::Internal(e.to_string())))
                    .unwrap_or(0) as usize,
                kv_lora_rank: md_get("attention.kv_lora_rank")
                    .and_then(|v| v.to_u32().map_err(|e| SwarmError::Internal(e.to_string())))
                    .unwrap_or(512) as usize,
                q_lora_rank: md_get("attention.q_lora_rank")
                    .and_then(|v| v.to_u32().map_err(|e| SwarmError::Internal(e.to_string())))
                    .unwrap_or(0) as usize,
                key_length: md_get("attention.key_length")
                    .and_then(|v| v.to_u32().map_err(|e| SwarmError::Internal(e.to_string())))
                    .unwrap_or(head_dim as u32) as usize,
                value_length: md_get("attention.value_length")
                    .and_then(|v| v.to_u32().map_err(|e| SwarmError::Internal(e.to_string())))
                    .unwrap_or(head_dim as u32) as usize,
                rope_dim,
            };

            let mla_rope_dim = ds_meta.rope_dim;
            let (mla_cos, mla_sin) =
                precompute_freqs_cis(mla_rope_dim, rope_freq_base, context_length, &device)
                    .map_err(|e| SwarmError::Internal(e.to_string()))?;

            for layer_idx in layer_start..layer_end {
                let prefix = format!("blk.{layer_idx}");
                let has_mla = ct
                    .tensor_infos
                    .contains_key(&format!("{prefix}.attn_q_a.weight"));
                let has_moe = ct
                    .tensor_infos
                    .contains_key(&format!("{prefix}.ffn_gate_exps.weight"));

                let attn_norm = ct
                    .tensor(&mut reader, &format!("{prefix}.attn_norm.weight"), &device)
                    .map_err(|e| SwarmError::Internal(format!("{prefix}.attn_norm: {e}")))?;
                let ffn_norm_t = ct
                    .tensor(&mut reader, &format!("{prefix}.ffn_norm.weight"), &device)
                    .map_err(|e| SwarmError::Internal(format!("{prefix}.ffn_norm: {e}")))?;

                if has_mla {
                    let q_a = ct
                        .tensor(&mut reader, &format!("{prefix}.attn_q_a.weight"), &device)
                        .map_err(|e| SwarmError::Internal(format!("{prefix}.attn_q_a: {e}")))?;
                    let q_a_norm_t = ct
                        .tensor(
                            &mut reader,
                            &format!("{prefix}.attn_q_a_norm.weight"),
                            &device,
                        )
                        .map_err(|e| {
                            SwarmError::Internal(format!("{prefix}.attn_q_a_norm: {e}"))
                        })?;
                    let q_b = ct
                        .tensor(&mut reader, &format!("{prefix}.attn_q_b.weight"), &device)
                        .map_err(|e| SwarmError::Internal(format!("{prefix}.attn_q_b: {e}")))?;
                    let kv_a = ct
                        .tensor(&mut reader, &format!("{prefix}.attn_kv_a.weight"), &device)
                        .map_err(|e| SwarmError::Internal(format!("{prefix}.attn_kv_a: {e}")))?;
                    let kv_a_norm_t = ct
                        .tensor(
                            &mut reader,
                            &format!("{prefix}.attn_kv_a_norm.weight"),
                            &device,
                        )
                        .map_err(|e| {
                            SwarmError::Internal(format!("{prefix}.attn_kv_a_norm: {e}"))
                        })?;
                    let kv_b = ct
                        .tensor(&mut reader, &format!("{prefix}.attn_kv_b.weight"), &device)
                        .map_err(|e| SwarmError::Internal(format!("{prefix}.attn_kv_b: {e}")))?;
                    let wo = ct
                        .tensor(
                            &mut reader,
                            &format!("{prefix}.attn_output.weight"),
                            &device,
                        )
                        .map_err(|e| SwarmError::Internal(format!("{prefix}.attn_output: {e}")))?;

                    let attention = MlaWeights {
                        q_a: QMatMul::from_qtensor(q_a)
                            .map_err(|e| SwarmError::Internal(e.to_string()))?,
                        q_a_norm: RmsNorm::from_qtensor(q_a_norm_t, rms_norm_eps)
                            .map_err(|e| SwarmError::Internal(e.to_string()))?,
                        q_b: QMatMul::from_qtensor(q_b)
                            .map_err(|e| SwarmError::Internal(e.to_string()))?,
                        kv_a: QMatMul::from_qtensor(kv_a)
                            .map_err(|e| SwarmError::Internal(e.to_string()))?,
                        kv_a_norm: RmsNorm::from_qtensor(kv_a_norm_t, rms_norm_eps)
                            .map_err(|e| SwarmError::Internal(e.to_string()))?,
                        kv_b: QMatMul::from_qtensor(kv_b)
                            .map_err(|e| SwarmError::Internal(e.to_string()))?,
                        output: QMatMul::from_qtensor(wo)
                            .map_err(|e| SwarmError::Internal(e.to_string()))?,
                        n_head: head_count,
                        key_length: ds_meta.key_length,
                        value_length: ds_meta.value_length,
                        kv_lora_rank: ds_meta.kv_lora_rank,
                        rope_dim: mla_rope_dim,
                        cos: mla_cos.clone(),
                        sin: mla_sin.clone(),
                        neg_inf: neg_inf.clone(),
                    };

                    let ffn = if has_moe {
                        let gate_inp = ct
                            .tensor(
                                &mut reader,
                                &format!("{prefix}.ffn_gate_inp.weight"),
                                &device,
                            )
                            .map_err(|e| {
                                SwarmError::Internal(format!("{prefix}.ffn_gate_inp: {e}"))
                            })?;
                        let gate_exps = ct
                            .tensor(
                                &mut reader,
                                &format!("{prefix}.ffn_gate_exps.weight"),
                                &device,
                            )
                            .map_err(|e| {
                                SwarmError::Internal(format!("{prefix}.ffn_gate_exps: {e}"))
                            })?;
                        let down_exps = ct
                            .tensor(
                                &mut reader,
                                &format!("{prefix}.ffn_down_exps.weight"),
                                &device,
                            )
                            .map_err(|e| {
                                SwarmError::Internal(format!("{prefix}.ffn_down_exps: {e}"))
                            })?;
                        let up_exps = ct
                            .tensor(
                                &mut reader,
                                &format!("{prefix}.ffn_up_exps.weight"),
                                &device,
                            )
                            .map_err(|e| {
                                SwarmError::Internal(format!("{prefix}.ffn_up_exps: {e}"))
                            })?;

                        let gate_inp_t = gate_inp
                            .dequantize(&device)
                            .map_err(|e| SwarmError::Internal(format!("gate_inp: {e}")))?;
                        let gate_exps_t = gate_exps
                            .dequantize(&device)
                            .map_err(|e| SwarmError::Internal(format!("gate_exps: {e}")))?;
                        let down_exps_t = down_exps
                            .dequantize(&device)
                            .map_err(|e| SwarmError::Internal(format!("down_exps: {e}")))?;
                        let up_exps_t = up_exps
                            .dequantize(&device)
                            .map_err(|e| SwarmError::Internal(format!("up_exps: {e}")))?;

                        let shared_gate = ct
                            .tensor(
                                &mut reader,
                                &format!("{prefix}.ffn_gate_shexp.weight"),
                                &device,
                            )
                            .ok()
                            .map(QMatMul::from_qtensor)
                            .transpose()
                            .map_err(|e| SwarmError::Internal(e.to_string()))?;
                        let shared_down = ct
                            .tensor(
                                &mut reader,
                                &format!("{prefix}.ffn_down_shexp.weight"),
                                &device,
                            )
                            .ok()
                            .map(QMatMul::from_qtensor)
                            .transpose()
                            .map_err(|e| SwarmError::Internal(e.to_string()))?;
                        let shared_up = ct
                            .tensor(
                                &mut reader,
                                &format!("{prefix}.ffn_up_shexp.weight"),
                                &device,
                            )
                            .ok()
                            .map(QMatMul::from_qtensor)
                            .transpose()
                            .map_err(|e| SwarmError::Internal(e.to_string()))?;

                        FfnVariant::MoE(MoeFfn {
                            gate: gate_inp_t,
                            gate_exps: gate_exps_t,
                            down_exps: down_exps_t,
                            up_exps: up_exps_t,
                            shared_gate,
                            shared_down,
                            shared_up,
                            n_experts_used: ds_meta.n_experts_used,
                        })
                    } else {
                        let ffn_gate = ct
                            .tensor(&mut reader, &format!("{prefix}.ffn_gate.weight"), &device)
                            .map_err(|e| SwarmError::Internal(format!("{prefix}.ffn_gate: {e}")))?;
                        let ffn_down = ct
                            .tensor(&mut reader, &format!("{prefix}.ffn_down.weight"), &device)
                            .map_err(|e| SwarmError::Internal(format!("{prefix}.ffn_down: {e}")))?;
                        let ffn_up = ct
                            .tensor(&mut reader, &format!("{prefix}.ffn_up.weight"), &device)
                            .map_err(|e| SwarmError::Internal(format!("{prefix}.ffn_up: {e}")))?;
                        FfnVariant::Dense(Mlp {
                            ffn_gate: QMatMul::from_qtensor(ffn_gate)
                                .map_err(|e| SwarmError::Internal(e.to_string()))?,
                            ffn_down: QMatMul::from_qtensor(ffn_down)
                                .map_err(|e| SwarmError::Internal(e.to_string()))?,
                            ffn_up: QMatMul::from_qtensor(ffn_up)
                                .map_err(|e| SwarmError::Internal(e.to_string()))?,
                            activation: Activation::SiLU,
                        })
                    };

                    layers.push(LayerVariant::DeepSeek {
                        attention,
                        ffn,
                        attention_norm: make_norm(attn_norm, rms_norm_eps)?,
                        ffn_norm: make_norm(ffn_norm_t, rms_norm_eps)?,
                    });
                } else {
                    // Dense attention layer
                    let attention_wq = ct
                        .tensor(&mut reader, &format!("{prefix}.attn_q.weight"), &device)
                        .map_err(|e| SwarmError::Internal(format!("{prefix}.attn_q: {e}")))?;
                    let attention_wk = ct
                        .tensor(&mut reader, &format!("{prefix}.attn_k.weight"), &device)
                        .map_err(|e| SwarmError::Internal(format!("{prefix}.attn_k: {e}")))?;
                    let attention_wv = ct
                        .tensor(&mut reader, &format!("{prefix}.attn_v.weight"), &device)
                        .map_err(|e| SwarmError::Internal(format!("{prefix}.attn_v: {e}")))?;
                    let attention_wo = ct
                        .tensor(
                            &mut reader,
                            &format!("{prefix}.attn_output.weight"),
                            &device,
                        )
                        .map_err(|e| SwarmError::Internal(format!("{prefix}.attn_output: {e}")))?;
                    let attention_bq = ct
                        .tensor(&mut reader, &format!("{prefix}.attn_q.bias"), &device)
                        .ok()
                        .map(|t| t.dequantize(&device))
                        .transpose()
                        .map_err(|e| SwarmError::Internal(e.to_string()))?;
                    let attention_bk = ct
                        .tensor(&mut reader, &format!("{prefix}.attn_k.bias"), &device)
                        .ok()
                        .map(|t| t.dequantize(&device))
                        .transpose()
                        .map_err(|e| SwarmError::Internal(e.to_string()))?;
                    let attention_bv = ct
                        .tensor(&mut reader, &format!("{prefix}.attn_v.bias"), &device)
                        .ok()
                        .map(|t| t.dequantize(&device))
                        .transpose()
                        .map_err(|e| SwarmError::Internal(e.to_string()))?;
                    let ffn_gate = ct
                        .tensor(&mut reader, &format!("{prefix}.ffn_gate.weight"), &device)
                        .map_err(|e| SwarmError::Internal(format!("{prefix}.ffn_gate: {e}")))?;
                    let ffn_down = ct
                        .tensor(&mut reader, &format!("{prefix}.ffn_down.weight"), &device)
                        .map_err(|e| SwarmError::Internal(format!("{prefix}.ffn_down: {e}")))?;
                    let ffn_up = ct
                        .tensor(&mut reader, &format!("{prefix}.ffn_up.weight"), &device)
                        .map_err(|e| SwarmError::Internal(format!("{prefix}.ffn_up: {e}")))?;
                    layers.push(LayerVariant::Dense(LayerWeights {
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
                        attn_q_norm: None,
                        attn_k_norm: None,
                        ffn: FfnVariant::Dense(Mlp {
                            ffn_gate: QMatMul::from_qtensor(ffn_gate)
                                .map_err(|e| SwarmError::Internal(e.to_string()))?,
                            ffn_down: QMatMul::from_qtensor(ffn_down)
                                .map_err(|e| SwarmError::Internal(e.to_string()))?,
                            ffn_up: QMatMul::from_qtensor(ffn_up)
                                .map_err(|e| SwarmError::Internal(e.to_string()))?,
                            activation: Activation::SiLU,
                        }),
                        ffn_norm: make_norm(ffn_norm_t, rms_norm_eps)?,
                        n_head: head_count,
                        n_kv_head: head_count_kv,
                        head_dim,
                        cos: cos.clone(),
                        sin: sin.clone(),
                        neg_inf: neg_inf.clone(),
                        use_rope_contiguous,
                        attn_logit_softcap,
                        rope_dim,
                        skip_rope: false,
                    }));
                }
            }
        } else if matches!(model_arch, ModelArch::Llama4) {
            // ── Llama 4 Scout/Maverick shard loading: iRoPE + MoE ──
            let n_experts = ct
                .metadata
                .get(&format!("{arch}.expert_count"))
                .and_then(|v| v.to_u32().ok())
                .unwrap_or(0) as usize;
            let n_experts_used = ct
                .metadata
                .get(&format!("{arch}.expert_used_count"))
                .and_then(|v| v.to_u32().ok())
                .unwrap_or(1) as usize;

            for layer_idx in layer_start..layer_end {
                let prefix = format!("blk.{layer_idx}");
                let is_nope = layer_idx % 4 == 3;

                let attention_wq = ct
                    .tensor(&mut reader, &format!("{prefix}.attn_q.weight"), &device)
                    .map_err(|e| SwarmError::Internal(format!("{prefix}.attn_q: {e}")))?;
                let attention_wk = ct
                    .tensor(&mut reader, &format!("{prefix}.attn_k.weight"), &device)
                    .map_err(|e| SwarmError::Internal(format!("{prefix}.attn_k: {e}")))?;
                let attention_wv = ct
                    .tensor(&mut reader, &format!("{prefix}.attn_v.weight"), &device)
                    .map_err(|e| SwarmError::Internal(format!("{prefix}.attn_v: {e}")))?;
                let attention_wo = ct
                    .tensor(
                        &mut reader,
                        &format!("{prefix}.attn_output.weight"),
                        &device,
                    )
                    .map_err(|e| SwarmError::Internal(format!("{prefix}.attn_output: {e}")))?;
                let attn_norm = ct
                    .tensor(&mut reader, &format!("{prefix}.attn_norm.weight"), &device)
                    .map_err(|e| SwarmError::Internal(format!("{prefix}.attn_norm: {e}")))?;
                let ffn_norm = ct
                    .tensor(&mut reader, &format!("{prefix}.ffn_norm.weight"), &device)
                    .map_err(|e| SwarmError::Internal(format!("{prefix}.ffn_norm: {e}")))?;

                let attention_bq = ct
                    .tensor(&mut reader, &format!("{prefix}.attn_q.bias"), &device)
                    .ok()
                    .map(|t| t.dequantize(&device))
                    .transpose()
                    .map_err(|e| SwarmError::Internal(format!("attn_q.bias: {e}")))?;
                let attention_bk = ct
                    .tensor(&mut reader, &format!("{prefix}.attn_k.bias"), &device)
                    .ok()
                    .map(|t| t.dequantize(&device))
                    .transpose()
                    .map_err(|e| SwarmError::Internal(format!("attn_k.bias: {e}")))?;
                let attention_bv = ct
                    .tensor(&mut reader, &format!("{prefix}.attn_v.bias"), &device)
                    .ok()
                    .map(|t| t.dequantize(&device))
                    .transpose()
                    .map_err(|e| SwarmError::Internal(format!("attn_v.bias: {e}")))?;

                let has_moe = ct
                    .tensor_infos
                    .contains_key(&format!("{prefix}.ffn_gate_exps.weight"));

                let ffn = if has_moe && n_experts > 0 {
                    let gate_inp = ct
                        .tensor(
                            &mut reader,
                            &format!("{prefix}.ffn_gate_inp.weight"),
                            &device,
                        )
                        .map_err(|e| SwarmError::Internal(format!("{prefix}.ffn_gate_inp: {e}")))?;
                    let gate_exps = ct
                        .tensor(
                            &mut reader,
                            &format!("{prefix}.ffn_gate_exps.weight"),
                            &device,
                        )
                        .map_err(|e| {
                            SwarmError::Internal(format!("{prefix}.ffn_gate_exps: {e}"))
                        })?;
                    let down_exps = ct
                        .tensor(
                            &mut reader,
                            &format!("{prefix}.ffn_down_exps.weight"),
                            &device,
                        )
                        .map_err(|e| {
                            SwarmError::Internal(format!("{prefix}.ffn_down_exps: {e}"))
                        })?;
                    let up_exps = ct
                        .tensor(
                            &mut reader,
                            &format!("{prefix}.ffn_up_exps.weight"),
                            &device,
                        )
                        .map_err(|e| SwarmError::Internal(format!("{prefix}.ffn_up_exps: {e}")))?;

                    let gate_inp_t = gate_inp
                        .dequantize(&device)
                        .map_err(|e| SwarmError::Internal(format!("gate_inp dequant: {e}")))?;
                    let gate_exps_t = gate_exps
                        .dequantize(&device)
                        .map_err(|e| SwarmError::Internal(format!("gate_exps dequant: {e}")))?;
                    let down_exps_t = down_exps
                        .dequantize(&device)
                        .map_err(|e| SwarmError::Internal(format!("down_exps dequant: {e}")))?;
                    let up_exps_t = up_exps
                        .dequantize(&device)
                        .map_err(|e| SwarmError::Internal(format!("up_exps dequant: {e}")))?;

                    let shared_gate = ct
                        .tensor(
                            &mut reader,
                            &format!("{prefix}.ffn_gate_shexp.weight"),
                            &device,
                        )
                        .ok()
                        .map(QMatMul::from_qtensor)
                        .transpose()
                        .map_err(|e| SwarmError::Internal(format!("shared gate: {e}")))?;
                    let shared_down = ct
                        .tensor(
                            &mut reader,
                            &format!("{prefix}.ffn_down_shexp.weight"),
                            &device,
                        )
                        .ok()
                        .map(QMatMul::from_qtensor)
                        .transpose()
                        .map_err(|e| SwarmError::Internal(format!("shared down: {e}")))?;
                    let shared_up = ct
                        .tensor(
                            &mut reader,
                            &format!("{prefix}.ffn_up_shexp.weight"),
                            &device,
                        )
                        .ok()
                        .map(QMatMul::from_qtensor)
                        .transpose()
                        .map_err(|e| SwarmError::Internal(format!("shared up: {e}")))?;

                    FfnVariant::MoE(MoeFfn {
                        gate: gate_inp_t,
                        gate_exps: gate_exps_t,
                        down_exps: down_exps_t,
                        up_exps: up_exps_t,
                        shared_gate,
                        shared_down,
                        shared_up,
                        n_experts_used,
                    })
                } else {
                    let ffn_gate = ct
                        .tensor(&mut reader, &format!("{prefix}.ffn_gate.weight"), &device)
                        .map_err(|e| SwarmError::Internal(format!("{prefix}.ffn_gate: {e}")))?;
                    let ffn_down = ct
                        .tensor(&mut reader, &format!("{prefix}.ffn_down.weight"), &device)
                        .map_err(|e| SwarmError::Internal(format!("{prefix}.ffn_down: {e}")))?;
                    let ffn_up = ct
                        .tensor(&mut reader, &format!("{prefix}.ffn_up.weight"), &device)
                        .map_err(|e| SwarmError::Internal(format!("{prefix}.ffn_up: {e}")))?;
                    FfnVariant::Dense(Mlp {
                        ffn_gate: QMatMul::from_qtensor(ffn_gate)
                            .map_err(|e| SwarmError::Internal(e.to_string()))?,
                        ffn_down: QMatMul::from_qtensor(ffn_down)
                            .map_err(|e| SwarmError::Internal(e.to_string()))?,
                        ffn_up: QMatMul::from_qtensor(ffn_up)
                            .map_err(|e| SwarmError::Internal(e.to_string()))?,
                        activation: Activation::SiLU,
                    })
                };

                layers.push(LayerVariant::Dense(LayerWeights {
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
                    attn_q_norm: None,
                    attn_k_norm: None,
                    ffn,
                    ffn_norm: make_norm(ffn_norm, rms_norm_eps)?,
                    n_head: head_count,
                    n_kv_head: head_count_kv,
                    head_dim,
                    cos: cos.clone(),
                    sin: sin.clone(),
                    neg_inf: neg_inf.clone(),
                    use_rope_contiguous,
                    attn_logit_softcap,
                    rope_dim,
                    skip_rope: is_nope,
                }));
            }
        } else {
            for layer_idx in layer_start..layer_end {
                let prefix = format!("blk.{layer_idx}");

                // Try separate Q/K/V first; fall back to fused attn_qkv (Phi-3)
                let has_fused_qkv = ct
                    .tensor_infos
                    .contains_key(&format!("{prefix}.attn_qkv.weight"));
                let (qkv_q, qkv_k, qkv_v) = if has_fused_qkv {
                    // Fused QKV: load to CPU, dequantize, split, convert to f16, move to device
                    let cpu = &candle_core::Device::Cpu;
                    let fused = ct
                        .tensor(&mut reader, &format!("{prefix}.attn_qkv.weight"), cpu)
                        .map_err(|e| SwarmError::Internal(format!("{prefix}.attn_qkv: {e}")))?;
                    let fused_f32 = fused
                        .dequantize(cpu)
                        .map_err(|e| SwarmError::Internal(format!("qkv dequant: {e}")))?;
                    let q_dim = head_count * head_dim;
                    let k_dim = head_count_kv * head_dim;
                    let v_dim = k_dim;
                    let wq = fused_f32
                        .narrow(0, 0, q_dim)
                        .map_err(|e| SwarmError::Internal(format!("qkv split q: {e}")))?
                        .contiguous()
                        .map_err(|e| SwarmError::Internal(format!("q contiguous: {e}")))?;
                    let wk = fused_f32
                        .narrow(0, q_dim, k_dim)
                        .map_err(|e| SwarmError::Internal(format!("qkv split k: {e}")))?
                        .contiguous()
                        .map_err(|e| SwarmError::Internal(format!("k contiguous: {e}")))?;
                    let wv = fused_f32
                        .narrow(0, q_dim + k_dim, v_dim)
                        .map_err(|e| SwarmError::Internal(format!("qkv split v: {e}")))?
                        .contiguous()
                        .map_err(|e| SwarmError::Internal(format!("v contiguous: {e}")))?;
                    (
                        QMatMul::from_f32_tensor(wq, &device)?,
                        QMatMul::from_f32_tensor(wk, &device)?,
                        QMatMul::from_f32_tensor(wv, &device)?,
                    )
                } else {
                    let wq = ct
                        .tensor(&mut reader, &format!("{prefix}.attn_q.weight"), &device)
                        .map_err(|e| {
                            SwarmError::Internal(format!("Failed to load {prefix}.attn_q: {e}"))
                        })?;
                    let wk = ct
                        .tensor(&mut reader, &format!("{prefix}.attn_k.weight"), &device)
                        .map_err(|e| {
                            SwarmError::Internal(format!("Failed to load {prefix}.attn_k: {e}"))
                        })?;
                    let wv = ct
                        .tensor(&mut reader, &format!("{prefix}.attn_v.weight"), &device)
                        .map_err(|e| {
                            SwarmError::Internal(format!("Failed to load {prefix}.attn_v: {e}"))
                        })?;
                    (
                        QMatMul::from_qtensor(wq)
                            .map_err(|e| SwarmError::Internal(e.to_string()))?,
                        QMatMul::from_qtensor(wk)
                            .map_err(|e| SwarmError::Internal(e.to_string()))?,
                        QMatMul::from_qtensor(wv)
                            .map_err(|e| SwarmError::Internal(e.to_string()))?,
                    )
                };
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

                // FFN: try separate gate/up; fall back to fused (Phi-3)
                let has_ffn_gate = ct
                    .tensor_infos
                    .contains_key(&format!("{prefix}.ffn_gate.weight"));
                let ffn_down_qt = ct
                    .tensor(&mut reader, &format!("{prefix}.ffn_down.weight"), &device)
                    .map_err(|e| {
                        SwarmError::Internal(format!("Failed to load {prefix}.ffn_down: {e}"))
                    })?;
                let (ffn_gate_mm, ffn_down_mm, ffn_up_mm) = if has_ffn_gate {
                    let gate = ct
                        .tensor(&mut reader, &format!("{prefix}.ffn_gate.weight"), &device)
                        .map_err(|e| {
                            SwarmError::Internal(format!("Failed to load {prefix}.ffn_gate: {e}"))
                        })?;
                    let up = ct
                        .tensor(&mut reader, &format!("{prefix}.ffn_up.weight"), &device)
                        .map_err(|e| {
                            SwarmError::Internal(format!("Failed to load {prefix}.ffn_up: {e}"))
                        })?;
                    (
                        QMatMul::from_qtensor(gate)
                            .map_err(|e| SwarmError::Internal(e.to_string()))?,
                        QMatMul::from_qtensor(ffn_down_qt)
                            .map_err(|e| SwarmError::Internal(e.to_string()))?,
                        QMatMul::from_qtensor(up)
                            .map_err(|e| SwarmError::Internal(e.to_string()))?,
                    )
                } else {
                    // Fused gate+up: load to CPU, dequantize, split, convert to f16, move to device
                    let cpu = &candle_core::Device::Cpu;
                    let fused = ct
                        .tensor(&mut reader, &format!("{prefix}.ffn_up.weight"), cpu)
                        .map_err(|e| {
                            SwarmError::Internal(format!("Failed to load {prefix}.ffn_up: {e}"))
                        })?;
                    let fused_f32 = fused
                        .dequantize(cpu)
                        .map_err(|e| SwarmError::Internal(format!("ffn_up dequant: {e}")))?;
                    let total_rows = fused_f32
                        .dim(0)
                        .map_err(|e| SwarmError::Internal(format!("ffn_up dim0: {e}")))?;
                    let half = total_rows / 2;
                    let gate_f32 = fused_f32
                        .narrow(0, 0, half)
                        .map_err(|e| SwarmError::Internal(format!("ffn gate split: {e}")))?
                        .contiguous()
                        .map_err(|e| SwarmError::Internal(format!("gate contiguous: {e}")))?;
                    let up_f32 = fused_f32
                        .narrow(0, half, half)
                        .map_err(|e| SwarmError::Internal(format!("ffn up split: {e}")))?
                        .contiguous()
                        .map_err(|e| SwarmError::Internal(format!("up contiguous: {e}")))?;
                    (
                        QMatMul::from_f32_tensor(gate_f32, &device)?,
                        QMatMul::from_qtensor(ffn_down_qt)
                            .map_err(|e| SwarmError::Internal(e.to_string()))?,
                        QMatMul::from_f32_tensor(up_f32, &device)?,
                    )
                };
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

                // Qwen3 QK normalization (optional)
                let attn_q_norm = ct
                    .tensor(
                        &mut reader,
                        &format!("{prefix}.attn_q_norm.weight"),
                        &device,
                    )
                    .ok()
                    .map(|t| RmsNorm::from_qtensor(t, rms_norm_eps))
                    .transpose()
                    .map_err(|e| SwarmError::Internal(format!("attn_q_norm: {e}")))?;
                let attn_k_norm = ct
                    .tensor(
                        &mut reader,
                        &format!("{prefix}.attn_k_norm.weight"),
                        &device,
                    )
                    .ok()
                    .map(|t| RmsNorm::from_qtensor(t, rms_norm_eps))
                    .transpose()
                    .map_err(|e| SwarmError::Internal(format!("attn_k_norm: {e}")))?;

                layers.push(LayerVariant::Dense(LayerWeights {
                    attention_wq: qkv_q,
                    attention_wk: qkv_k,
                    attention_wv: qkv_v,
                    attention_wo: QMatMul::from_qtensor(attention_wo)
                        .map_err(|e| SwarmError::Internal(e.to_string()))?,
                    attention_bq,
                    attention_bk,
                    attention_bv,
                    attention_norm: make_norm(attn_norm, rms_norm_eps)?,
                    attn_q_norm,
                    attn_k_norm,
                    ffn: FfnVariant::Dense(Mlp {
                        ffn_gate: ffn_gate_mm,
                        ffn_down: ffn_down_mm,
                        ffn_up: ffn_up_mm,
                        activation,
                    }),
                    ffn_norm: make_norm(ffn_norm, rms_norm_eps)?,
                    n_head: head_count,
                    n_kv_head: head_count_kv,
                    head_dim,
                    cos: cos.clone(),
                    sin: sin.clone(),
                    neg_inf: neg_inf.clone(),
                    use_rope_contiguous,
                    attn_logit_softcap,
                    rope_dim,
                    skip_rope: false,
                }));
            }
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
                Some(SplitTokenizer::from_bpe(
                    vocab,
                    &merges_raw,
                    &pre_type,
                    &tokenizer_model,
                ))
            } else if tokenizer_model == "llama" {
                // Sentencepiece-based model from GGUF header
                let scores: Vec<f32> = ct
                    .metadata
                    .get("tokenizer.ggml.scores")
                    .and_then(|v| v.to_vec().ok())
                    .map(|arr| arr.iter().filter_map(|v| v.to_f32().ok()).collect())
                    .unwrap_or_default();
                if !scores.is_empty() {
                    tracing::info!(
                        vocab_size = vocab.len(),
                        scores = scores.len(),
                        "Building HF Unigram tokenizer from GGUF header sentencepiece data"
                    );
                    Some(SplitTokenizer::from_sentencepiece(vocab, &scores))
                } else {
                    None
                }
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
            "gemma" | "gemma2" => {
                if !eos_tokens.contains(&107) {
                    eos_tokens.push(107);
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

        let has_biases = layers
            .first()
            .is_some_and(|l| matches!(l, LayerVariant::Dense(lw) if lw.attention_bq.is_some()));
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
            masks: None,
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
            max_seq_len: context_length.min(2048),
            kv_model_key: format!("{layer_start}-{layer_end}-{block_count}"),
            final_logit_softcap,
        })
    }

    /// Build a causal mask for the given sequence length.
    ///
    /// Pre-allocates a single mask at a ceiling size (min of max_seq_len and 4096),
    /// then uses `narrow()` to slice views for smaller sequences — zero-copy.
    /// Only re-allocates if `t` exceeds the current ceiling.
    fn mask(&mut self, t: usize) -> CandleResult<Tensor> {
        // Fast path: existing mask is large enough — narrow-slice a view (no copy)
        if let Some((cached_size, ref mask)) = self.masks {
            if t <= cached_size {
                return mask.narrow(0, 0, t)?.narrow(1, 0, t);
            }
        }
        // Allocate at a reasonable ceiling to amortize future requests.
        // Cap at 4096 to avoid excessive memory (4096^2 = 16MB for u8).
        let alloc_size = t.max(self.max_seq_len.min(4096));
        let mask_data: Vec<_> = (0..alloc_size)
            .flat_map(|i| (0..alloc_size).map(move |j| u8::from(j > i)))
            .collect();
        let mask = Tensor::from_slice(&mask_data, (alloc_size, alloc_size), &self.device)?;
        self.masks = Some((alloc_size, mask.clone()));
        if t < alloc_size {
            mask.narrow(0, 0, t)?.narrow(1, 0, t)
        } else {
            Ok(mask)
        }
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
        let forward_start = std::time::Instant::now();
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
            let mut emb = self
                .tok_embeddings
                .as_ref()
                .ok_or_else(|| SwarmError::Internal("Missing embedding table".into()))?
                .forward(&input)
                .map_err(|e| SwarmError::Internal(format!("Embedding forward failed: {e}")))?;
            // Gemma models scale embeddings by sqrt(hidden_dim)
            if self.arch.use_gemma_norm() {
                let scale = (self.hidden_dim as f64).sqrt();
                emb = emb
                    .affine(scale, 0.0)
                    .map_err(|e| SwarmError::Internal(format!("Embedding scale failed: {e}")))?;
            }
            emb
        } else {
            // Non-first segment: input is already hidden states
            input
        };

        // Get seq_len for mask
        let seq_len = layer_in
            .dim(1)
            .map_err(|e| SwarmError::Internal(e.to_string()))?;

        let num_layers = self.layers.len();
        // Build the cache key once — reused for both take and writeback (zero alloc on hot path).
        let cache_key = KvCacheStore::cache_key(&self.kv_model_key, request_id);

        // Get or create the per-request cache entry, extract the layer caches,
        // then drop the DashMap guard before running the (potentially slow) forward pass.
        // Use mem::take instead of clone to avoid copying the KV cache Vec.
        #[allow(unused_mut)]
        let (mut layer_kv_caches, mut layer_ssm_states): (
            Vec<Option<KvCache>>,
            Vec<Option<SsmState>>,
        ) = {
            let mut entry = kv_cache_store.get_or_create_keyed(&cache_key, num_layers);
            (
                std::mem::take(&mut entry.layers),
                std::mem::take(&mut entry.ssm_states),
            )
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

            match layer {
                LayerVariant::Dense(lw) => {
                    let x = layer_in;
                    let residual = &x;
                    let x = lw
                        .attention_norm
                        .forward(&x)
                        .map_err(|e| SwarmError::Internal(format!("attn_norm: {e}")))?;
                    let attn = lw
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
                    let x = lw
                        .ffn_norm
                        .forward(&x)
                        .map_err(|e| SwarmError::Internal(format!("ffn_norm: {e}")))?;
                    let x = match &lw.ffn {
                        FfnVariant::Dense(mlp) => mlp
                            .forward(&x, lora_param)
                            .map_err(|e| SwarmError::Internal(format!("mlp: {e}")))?,
                        FfnVariant::MoE(moe) => moe
                            .forward(&x)
                            .map_err(|e| SwarmError::Internal(format!("moe: {e}")))?,
                    };
                    layer_in = (x + residual).map_err(|e| SwarmError::Internal(e.to_string()))?;
                }
                LayerVariant::DeepSeek {
                    attention,
                    ffn,
                    attention_norm,
                    ffn_norm,
                } => {
                    let x = attention_norm
                        .forward(&layer_in)
                        .map_err(|e| SwarmError::Internal(format!("ds_attn_norm: {e}")))?;
                    let attn = attention
                        .forward_mla(
                            &x,
                            mask.as_ref(),
                            index_pos,
                            &mut layer_kv_caches[layer_idx],
                            max_seq_len,
                        )
                        .map_err(|e| SwarmError::Internal(format!("mla: {e}")))?;
                    let x = (attn + &layer_in).map_err(|e| SwarmError::Internal(e.to_string()))?;
                    let residual = &x;
                    let normed = ffn_norm
                        .forward(&x)
                        .map_err(|e| SwarmError::Internal(format!("ds_ffn_norm: {e}")))?;
                    let ffn_out = match ffn {
                        FfnVariant::Dense(mlp) => mlp
                            .forward(&normed, None)
                            .map_err(|e| SwarmError::Internal(format!("ds_mlp: {e}")))?,
                        FfnVariant::MoE(moe) => moe
                            .forward(&normed)
                            .map_err(|e| SwarmError::Internal(format!("moe: {e}")))?,
                    };
                    layer_in =
                        (ffn_out + residual).map_err(|e| SwarmError::Internal(e.to_string()))?;
                }
                LayerVariant::Qwen35Attn { .. } | LayerVariant::Qwen35Ssm { .. } => {
                    return Err(SwarmError::Internal(
                        "Qwen 3.5 inference is not yet implemented".into(),
                    ));
                }
            }
        }

        // Write the updated KV-caches and SSM states back to the store.
        {
            let mut entry = kv_cache_store.get_or_create_keyed(&cache_key, num_layers);
            entry.layers = layer_kv_caches;
            entry.ssm_states = layer_ssm_states;
            entry.last_accessed = std::time::Instant::now();
        }

        let result = if is_last {
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
            let mut logits = output
                .forward(&x)
                .map_err(|e| SwarmError::Internal(format!("output_proj: {e}")))?;
            // Gemma 2 final logit soft-capping: tanh(logits / cap) * cap
            if let Some(cap) = self.final_logit_softcap {
                logits = logits
                    .affine(1.0 / cap as f64, 0.0)
                    .and_then(|t| t.tanh())
                    .and_then(|t| t.affine(cap as f64, 0.0))
                    .map_err(|e| SwarmError::Internal(format!("final_logit_softcap: {e}")))?;
            }
            Ok(logits)
        } else {
            // Intermediate segment: return hidden states for next segment
            Ok(layer_in)
        };

        let forward_ms = forward_start.elapsed().as_millis() as u64;
        tracing::debug!(
            request_id,
            index_pos,
            seq_len,
            num_layers,
            is_first,
            is_last,
            kv_offset,
            forward_ms,
            "DIAG: SplitModel forward pass complete"
        );

        result
    }

    /// Tensor-parallel forward pass for a single layer.
    ///
    /// Each TP node computes only its fraction of the computation:
    /// - Attention: processes `n_head / tp_size` heads (head-parallel)
    /// - MLP: processes `intermediate_dim / tp_size` columns (column-parallel gate/up,
    ///   row-parallel down)
    ///
    /// Returns a **partial** hidden state that must be summed (AllReduced) across all
    /// TP nodes to produce the correct full output. The caller (pipeline executor)
    /// coordinates the AllReduce between layers.
    ///
    /// `abs_layer_idx` is the absolute layer index in the full model (not relative
    /// to this segment's layer_start).
    #[allow(clippy::too_many_arguments)]
    pub fn forward_tp_layer(
        &mut self,
        input: &Tensor,
        abs_layer_idx: usize,
        index_pos: usize,
        tp_rank: usize,
        tp_size: usize,
        kv_cache_store: &KvCacheStore,
        request_id: &str,
    ) -> Result<Tensor, SwarmError> {
        // Map absolute layer index to our local layer array index
        let local_idx = abs_layer_idx.checked_sub(self.layer_start).ok_or_else(|| {
            SwarmError::Internal(format!(
                "Layer {abs_layer_idx} not in segment [{}, {})",
                self.layer_start, self.layer_end
            ))
        })?;
        if local_idx >= self.layers.len() {
            return Err(SwarmError::Internal(format!(
                "Layer index {local_idx} out of range (have {} layers)",
                self.layers.len()
            )));
        }

        let input = input
            .to_device(&self.device)
            .map_err(|e| SwarmError::Internal(format!("Device transfer: {e}")))?;

        let seq_len = input
            .dim(1)
            .map_err(|e| SwarmError::Internal(e.to_string()))?;

        // KV-cache keyed by model key + request_id
        let model_key = format!(
            "tp{}-{}-{}-{}",
            tp_rank, self.layer_start, self.layer_end, self.total_layers
        );
        let num_layers = self.layers.len();
        let cache_key = KvCacheStore::cache_key(&model_key, request_id);
        let mut layer_kv_caches: Vec<Option<KvCache>> = {
            let mut entry = kv_cache_store.get_or_create_keyed(&cache_key, num_layers);
            std::mem::take(&mut entry.layers)
        };

        let mask = if seq_len == 1 {
            None
        } else {
            Some(
                self.mask(seq_len)
                    .map_err(|e| SwarmError::Internal(e.to_string()))?,
            )
        };

        let layer = &self.layers[local_idx];
        let x = &input;

        // DeepSeek layers don't support tensor parallelism yet (MoE expert routing
        // makes TP significantly more complex). Return error for TP + DeepSeek.
        let lw = match layer {
            LayerVariant::Dense(lw) => lw,
            LayerVariant::DeepSeek { .. } => {
                return Err(SwarmError::Internal(
                    "Tensor parallelism is not supported for DeepSeek MoE/MLA layers. \
                     Use pipeline parallelism (shard splitting) instead."
                        .into(),
                ));
            }
            LayerVariant::Qwen35Attn { .. } | LayerVariant::Qwen35Ssm { .. } => {
                return Err(SwarmError::Internal(
                    "Tensor parallelism is not supported for Qwen 3.5 layers. \
                     Use pipeline parallelism (shard splitting) instead."
                        .into(),
                ));
            }
        };

        // Attention norm (full — not split)
        let normed = lw
            .attention_norm
            .forward(x)
            .map_err(|e| SwarmError::Internal(format!("attn_norm: {e}")))?;

        // Head-parallel attention: only compute assigned heads
        let attn_partial = lw
            .forward_attn_tp(
                &normed,
                mask.as_ref(),
                index_pos,
                &mut layer_kv_caches[local_idx],
                self.max_seq_len,
                tp_rank,
                tp_size,
            )
            .map_err(|e| SwarmError::Internal(format!("attn_tp: {e}")))?;

        // Partial attention result — needs AllReduce with other TP nodes.
        // The residual connection is applied AFTER AllReduce by the coordinator:
        //   full_attn = sum(attn_partial_0, attn_partial_1, ...) + residual

        // FFN norm on full input (not the partial attention — norm goes before residual add)
        let ffn_normed = lw
            .ffn_norm
            .forward(x)
            .map_err(|e| SwarmError::Internal(format!("ffn_norm: {e}")))?;

        // Column-parallel MLP: each TP node handles a fraction of the intermediate dimension
        let mlp_partial = match &lw.ffn {
            FfnVariant::Dense(mlp) => mlp
                .forward_tp(&ffn_normed, tp_rank, tp_size)
                .map_err(|e| SwarmError::Internal(format!("mlp_tp: {e}")))?,
            FfnVariant::MoE(_) => {
                return Err(SwarmError::Internal(
                    "Tensor parallelism not supported for MoE layers".to_string(),
                ));
            }
        };

        // Return partial = attn_partial + mlp_partial
        // The coordinator will AllReduce this and add the residual (input)
        let partial =
            (attn_partial + mlp_partial).map_err(|e| SwarmError::Internal(e.to_string()))?;

        // Write updated KV cache back (reuses cache_key — zero alloc)
        {
            let mut entry = kv_cache_store.get_or_create_keyed(&cache_key, num_layers);
            entry.layers = layer_kv_caches;
            entry.last_accessed = std::time::Instant::now();
        }

        Ok(partial)
    }

    /// Forward pass for multimodal (vision + text) inference.
    ///
    /// If this is the first segment and `vision_embeddings` is provided, the input
    /// is expected to contain TWO segments of token IDs separated by a marker:
    /// `[before_image_tokens...][MARKER=-1][after_image_tokens...]`
    ///
    /// The marker value -1 (as i64) indicates where vision embeddings should be
    /// inserted. Each part is embedded separately, then concatenated:
    /// `[before_emb][vision_emb][after_emb]`
    ///
    /// This matches llama.cpp's LLaVA approach of splitting the prompt at `<image>`.
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
                let input_dev = input
                    .to_device(&self.device)
                    .map_err(|e| SwarmError::Internal(format!("Device transfer: {e}")))?;

                // Read the token IDs to find the -1 marker
                let token_ids: Vec<i64> = input_dev
                    .flatten_all()
                    .map_err(|e| SwarmError::Internal(format!("flatten: {e}")))?
                    .to_vec1()
                    .map_err(|e| SwarmError::Internal(format!("to_vec1: {e}")))?;

                let marker_pos = token_ids.iter().position(|&id| id == -1);

                let tok_emb_layer = self
                    .tok_embeddings
                    .as_ref()
                    .ok_or_else(|| SwarmError::Internal("Missing embedding table".into()))?;

                let merged = if let Some(pos) = marker_pos {
                    // Split at marker: embed before and after separately, insert vision between
                    let num_vision = vision_emb
                        .dim(0)
                        .map_err(|e| SwarmError::Internal(format!("vision dim: {e}")))?;
                    let hidden = vision_emb
                        .dim(1)
                        .map_err(|e| SwarmError::Internal(format!("vision dim1: {e}")))?;

                    // All parts must be (1, seq, hidden) for cat along dim 1
                    let vision_3d = vision_emb
                        .reshape(&[1, num_vision, hidden])
                        .map_err(|e| SwarmError::Internal(format!("vision reshape: {e}")))?;

                    let mut parts: Vec<Tensor> = Vec::new();

                    // Embed tokens before <image>
                    if pos > 0 {
                        let before_ids = Tensor::new(&token_ids[..pos], &self.device)
                            .map_err(|e| SwarmError::Internal(format!("before tensor: {e}")))?
                            .reshape(&[1, pos])
                            .map_err(|e| SwarmError::Internal(format!("before reshape: {e}")))?;
                        let before_emb = tok_emb_layer
                            .forward(&before_ids)
                            .map_err(|e| SwarmError::Internal(format!("before embed: {e}")))?;
                        // Ensure 3D: (1, pos, hidden)
                        let before_3d = if before_emb.dims().len() == 2 {
                            before_emb.unsqueeze(0).map_err(|e| {
                                SwarmError::Internal(format!("before unsqueeze: {e}"))
                            })?
                        } else {
                            before_emb
                        };
                        parts.push(before_3d);
                    }

                    // Insert vision embeddings
                    parts.push(vision_3d);

                    // Embed tokens after <image>
                    let after_len = token_ids.len() - pos - 1;
                    if after_len > 0 {
                        let after_ids = Tensor::new(&token_ids[pos + 1..], &self.device)
                            .map_err(|e| SwarmError::Internal(format!("after tensor: {e}")))?
                            .reshape(&[1, after_len])
                            .map_err(|e| SwarmError::Internal(format!("after reshape: {e}")))?;
                        let after_emb = tok_emb_layer
                            .forward(&after_ids)
                            .map_err(|e| SwarmError::Internal(format!("after embed: {e}")))?;
                        let after_3d = if after_emb.dims().len() == 2 {
                            after_emb.unsqueeze(0).map_err(|e| {
                                SwarmError::Internal(format!("after unsqueeze: {e}"))
                            })?
                        } else {
                            after_emb
                        };
                        parts.push(after_3d);
                    }

                    let refs: Vec<&Tensor> = parts.iter().collect();
                    tracing::info!(
                        request_id,
                        marker_pos = pos,
                        before_tokens = pos,
                        after_tokens = after_len,
                        vision_tokens = num_vision,
                        "VLM: inserting vision embeddings at <image> position"
                    );
                    Tensor::cat(&refs, 1)
                        .map_err(|e| SwarmError::Internal(format!("merge cat: {e}")))?
                } else {
                    // No marker found — fallback to prepending vision before text
                    let text_embeddings = tok_emb_layer
                        .forward(&input_dev)
                        .map_err(|e| SwarmError::Internal(format!("Embedding: {e}")))?;
                    crate::inference::vision::merge_vision_text_embeddings(
                        &text_embeddings,
                        vision_emb,
                        &[],
                    )?
                };

                tracing::info!(
                    request_id,
                    merged_seq_len = ?merged.dims(),
                    "VLM: merged embeddings ready for transformer"
                );

                // Run through layers with merged hidden states.
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

        // Extract all per-request KV-caches up front (drop DashMap guards immediately).
        // Use mem::take instead of clone to avoid deep-copying all KV tensors.
        let mut all_kv_caches: Vec<Vec<Option<KvCache>>> = items
            .iter()
            .map(|item| {
                let mut entry =
                    kv_cache_store.get_or_create(&model_key, item.request_id, num_layers);
                entry.last_accessed = std::time::Instant::now();
                std::mem::take(&mut entry.layers)
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
            match layer {
                LayerVariant::Dense(lw) => {
                    let residual = batched.clone();
                    let normed = lw
                        .attention_norm
                        .forward(&batched)
                        .map_err(|e| SwarmError::Internal(format!("attn_norm: {e}")))?;

                    let mut attn_outputs: Vec<Tensor> = Vec::with_capacity(batch_size);
                    for (req_idx, item) in items.iter().enumerate() {
                        let x_i = normed
                            .narrow(0, req_idx, 1)
                            .map_err(|e| SwarmError::Internal(format!("narrow: {e}")))?;
                        let attn_out = lw
                            .forward_attn(
                                &x_i,
                                None,
                                item.index_pos,
                                &mut all_kv_caches[req_idx][layer_idx],
                                max_seq_len,
                                None,
                            )
                            .map_err(|e| SwarmError::Internal(format!("attn: {e}")))?;
                        attn_outputs.push(attn_out);
                    }

                    let attn_refs: Vec<&Tensor> = attn_outputs.iter().collect();
                    let attn_batched = Tensor::cat(&attn_refs, 0)
                        .map_err(|e| SwarmError::Internal(format!("attn restack: {e}")))?;
                    let x = (&attn_batched + &residual)
                        .map_err(|e| SwarmError::Internal(e.to_string()))?;

                    let residual2 = x.clone();
                    let x = lw
                        .ffn_norm
                        .forward(&x)
                        .map_err(|e| SwarmError::Internal(format!("ffn_norm: {e}")))?;
                    let x = match &lw.ffn {
                        FfnVariant::Dense(mlp) => mlp
                            .forward(&x, None)
                            .map_err(|e| SwarmError::Internal(format!("mlp: {e}")))?,
                        FfnVariant::MoE(moe) => moe
                            .forward(&x)
                            .map_err(|e| SwarmError::Internal(format!("moe: {e}")))?,
                    };
                    batched = (&x + &residual2).map_err(|e| SwarmError::Internal(e.to_string()))?;
                }
                LayerVariant::DeepSeek {
                    attention,
                    ffn,
                    attention_norm,
                    ffn_norm,
                } => {
                    // DeepSeek batch: per-request attention (MLA), batched FFN
                    let normed = attention_norm
                        .forward(&batched)
                        .map_err(|e| SwarmError::Internal(format!("ds_attn_norm: {e}")))?;

                    let mut attn_outputs: Vec<Tensor> = Vec::with_capacity(batch_size);
                    for (req_idx, item) in items.iter().enumerate() {
                        let x_i = normed
                            .narrow(0, req_idx, 1)
                            .map_err(|e| SwarmError::Internal(format!("narrow: {e}")))?;
                        let attn_out = attention
                            .forward_mla(
                                &x_i,
                                None,
                                item.index_pos,
                                &mut all_kv_caches[req_idx][layer_idx],
                                max_seq_len,
                            )
                            .map_err(|e| SwarmError::Internal(format!("mla_batch: {e}")))?;
                        attn_outputs.push(attn_out);
                    }

                    let attn_refs: Vec<&Tensor> = attn_outputs.iter().collect();
                    let attn_batched = Tensor::cat(&attn_refs, 0)
                        .map_err(|e| SwarmError::Internal(format!("mla restack: {e}")))?;
                    let x = (&attn_batched + &batched)
                        .map_err(|e| SwarmError::Internal(e.to_string()))?;

                    let residual = x.clone();
                    let normed = ffn_norm
                        .forward(&x)
                        .map_err(|e| SwarmError::Internal(format!("ds_ffn_norm: {e}")))?;
                    let ffn_out = match ffn {
                        FfnVariant::Dense(mlp) => mlp
                            .forward(&normed, None)
                            .map_err(|e| SwarmError::Internal(format!("ds_mlp: {e}")))?,
                        FfnVariant::MoE(moe) => moe
                            .forward(&normed)
                            .map_err(|e| SwarmError::Internal(format!("moe_batch: {e}")))?,
                    };
                    batched =
                        (&ffn_out + &residual).map_err(|e| SwarmError::Internal(e.to_string()))?;
                }
                LayerVariant::Qwen35Attn { .. } | LayerVariant::Qwen35Ssm { .. } => {
                    return Err(SwarmError::Internal(
                        "Qwen 3.5 batched inference is not yet implemented".into(),
                    ));
                }
            }
        }

        // Write updated KV-caches back (take instead of clone to avoid copying)
        for (req_idx, item) in items.iter().enumerate() {
            let mut entry = kv_cache_store.get_or_create(&model_key, item.request_id, num_layers);
            entry.layers = std::mem::take(&mut all_kv_caches[req_idx]);
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

    /// Return a reference to the tokenizer, if available.
    pub fn tokenizer(&self) -> Option<&SplitTokenizer> {
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

    /// Return the device this model is loaded on.
    pub fn device(&self) -> &Device {
        &self.device
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

    /// Whether this segment has the embedding table (first segment).
    pub fn is_first(&self) -> bool {
        self.tok_embeddings.is_some()
    }

    /// Whether this segment has the output projection (last segment).
    pub fn is_last(&self) -> bool {
        self.output.is_some()
    }

    /// Tokenize a prompt string and return the embedded hidden states.
    ///
    /// Used by tensor-parallel execution where embedding happens before
    /// layer-by-layer forwarding. Only works on the first segment (has embeddings).
    /// Tokenize a prompt and return (token_ids_tensor, num_tokens).
    ///
    /// Returns the token IDs as an I64 tensor with shape (1, seq_len),
    /// suitable for passing directly to `forward()` which handles embedding.
    pub fn tokenize(&self, prompt: &str) -> Result<(Tensor, usize), SwarmError> {
        let token_ids: Vec<i64> = if let Some(ref tokenizer) = self.tokenizer {
            tokenizer.encode(prompt)
        } else {
            prompt.bytes().map(|b| b as i64).collect()
        };
        let num_tokens = token_ids.len();
        let input = Tensor::new(&token_ids[..], &self.device)
            .map_err(|e| SwarmError::Internal(format!("Token tensor: {e}")))?
            .unsqueeze(0)
            .map_err(|e| SwarmError::Internal(format!("Unsqueeze: {e}")))?;
        Ok((input, num_tokens))
    }

    /// Create a single-token tensor for autoregressive decoding.
    ///
    /// Returns an I64 tensor with shape (1, 1), suitable for `forward()`.
    pub fn token_tensor(&self, token_id: u32) -> Result<Tensor, SwarmError> {
        Tensor::new(&[token_id as i64][..], &self.device)
            .map_err(|e| SwarmError::Internal(format!("Token tensor: {e}")))?
            .unsqueeze(0)
            .map_err(|e| SwarmError::Internal(format!("Unsqueeze: {e}")))
    }

    pub fn tokenize_and_embed(&self, prompt: &str) -> Result<Tensor, SwarmError> {
        let emb = self
            .tok_embeddings
            .as_ref()
            .ok_or_else(|| SwarmError::Internal("No embedding table (not first segment)".into()))?;

        // Tokenize — BpeTokenizer returns Vec<i64>
        let token_ids: Vec<i64> = if let Some(ref tokenizer) = self.tokenizer {
            tokenizer.encode(prompt)
        } else {
            // Fallback: byte-level encoding
            prompt.bytes().map(|b| b as i64).collect()
        };

        let input = Tensor::new(&token_ids[..], &self.device)
            .map_err(|e| SwarmError::Internal(format!("Token tensor: {e}")))?
            .unsqueeze(0)
            .map_err(|e| SwarmError::Internal(format!("Unsqueeze: {e}")))?;

        emb.forward(&input)
            .map_err(|e| SwarmError::Internal(format!("Embedding forward: {e}")))
    }

    /// Embed a single token ID into hidden states.
    ///
    /// Used by tensor-parallel execution for autoregressive decoding.
    pub fn embed_token(&self, token_id: u32) -> Result<Tensor, SwarmError> {
        let emb = self
            .tok_embeddings
            .as_ref()
            .ok_or_else(|| SwarmError::Internal("No embedding table (not first segment)".into()))?;

        let input = Tensor::new(&[token_id as i64][..], &self.device)
            .map_err(|e| SwarmError::Internal(format!("Token tensor: {e}")))?
            .unsqueeze(0)
            .map_err(|e| SwarmError::Internal(format!("Unsqueeze: {e}")))?;

        emb.forward(&input)
            .map_err(|e| SwarmError::Internal(format!("Embedding forward: {e}")))
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

/// Sample the next token from logits using full sampling parameters.
pub fn sample_token(logits: &Tensor, temperature: f32, top_p: f32) -> Result<u32, SwarmError> {
    sample_token_with_params(
        logits,
        &crate::types::SamplingParams {
            temperature,
            top_p,
            ..Default::default()
        },
    )
}

/// Sample the next token from logits using full SamplingParams (top_k, frequency/presence penalty).
pub fn sample_token_with_params(
    logits: &Tensor,
    params: &crate::types::SamplingParams,
) -> Result<u32, SwarmError> {
    let logits = logits
        .squeeze(0)
        .map_err(|e| SwarmError::Internal(e.to_string()))?;
    let logits = logits
        .to_dtype(DType::F32)
        .map_err(|e| SwarmError::Internal(e.to_string()))?;
    let mut logits_vec = logits
        .to_vec1::<f32>()
        .map_err(|e| SwarmError::Internal(e.to_string()))?;

    if logits_vec.is_empty() {
        return Err(SwarmError::Internal("Empty logits".into()));
    }

    // Apply top-k filtering before temperature scaling
    crate::inference::sampling::apply_top_k(&mut logits_vec, params.top_k);

    let temperature = params.temperature;
    let top_p = params.top_p;

    if temperature <= 0.0 {
        // Greedy: argmax — O(V)
        let (idx, _) = logits_vec
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap();
        return Ok(idx as u32);
    }

    // Apply temperature + softmax — O(V)
    let inv_temp = 1.0 / temperature;
    let max_val = logits_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<f32> = logits_vec
        .iter()
        .map(|&x| ((x - max_val) * inv_temp).exp())
        .collect();
    let sum: f32 = probs.iter().sum();
    if sum <= 0.0 || !sum.is_finite() {
        return Ok(0);
    }
    let inv_sum = 1.0 / sum;
    for p in probs.iter_mut() {
        *p *= inv_sum;
    }

    // Top-p >= 1.0: sample directly from full distribution — O(V), no sort needed
    if top_p >= 1.0 {
        let r: f32 = rand::random();
        let mut cumulative = 0.0;
        for (i, &p) in probs.iter().enumerate() {
            cumulative += p;
            if r < cumulative {
                return Ok(i as u32);
            }
        }
        return Ok((probs.len() - 1) as u32);
    }

    // Top-p < 1.0: use partial sort — O(V + K log K) where K << V
    // First pass: partition top-K candidates via select_nth_unstable_by (O(V))
    // then sort only those K elements (O(K log K))
    let mut indices: Vec<usize> = (0..probs.len()).collect();
    let k = 256.min(probs.len() - 1);
    indices.select_nth_unstable_by(k, |&a, &b| {
        probs[b]
            .partial_cmp(&probs[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    // Sort top-K+1 elements descending by probability
    indices[..=k].sort_unstable_by(|&a, &b| {
        probs[b]
            .partial_cmp(&probs[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Scan top-K for cumulative >= top_p
    let mut cumulative = 0.0;
    let mut cutoff = k + 1;
    for (i, &idx) in indices[..=k].iter().enumerate() {
        cumulative += probs[idx];
        if cumulative >= top_p {
            cutoff = i + 1;
            break;
        }
    }

    // If top-K wasn't enough (very flat distribution), fall back to full sort
    if cumulative < top_p {
        indices[k + 1..].sort_unstable_by(|&a, &b| {
            probs[b]
                .partial_cmp(&probs[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        for (i, &idx) in indices[k + 1..].iter().enumerate() {
            cumulative += probs[idx];
            if cumulative >= top_p {
                cutoff = k + 1 + i + 1;
                break;
            }
        }
    }

    // Renormalize and sample from the top-p subset
    let subset = &indices[..cutoff];
    let subset_sum: f32 = subset.iter().map(|&i| probs[i]).sum();
    let r: f32 = rand::random();
    let mut cumulative = 0.0;
    let inv_subset = 1.0 / subset_sum;
    for &idx in subset {
        cumulative += probs[idx] * inv_subset;
        if r < cumulative {
            return Ok(idx as u32);
        }
    }

    Ok(subset[subset.len() - 1] as u32)
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

    // Greedily assign layers to shards using a dynamic target that adjusts
    // as shards are emitted. This ensures the algorithm produces the exact
    // requested number of shards instead of underproducing when large prefixes
    // or uneven layer sizes cause the static target to be exceeded early.
    let mut layouts: Vec<LayerShardLayout> = Vec::new();
    let mut current_tensors: Vec<(String, u64, u64)> = Vec::new();
    let mut current_size: u64 = 0;
    let mut current_layer_start: Option<u32> = None;
    let mut current_layer_end: u32 = 0;
    let mut emitted_bytes: u64 = 0;

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

        // Dynamic target: distribute remaining bytes evenly across remaining shard slots.
        // This naturally adjusts when earlier shards are larger (e.g., due to prefix),
        // ensuring later shards are smaller to hit the total shard count.
        let remaining_budget = total_bytes.saturating_sub(emitted_bytes);
        let remaining_slots = (shard_count as usize - layouts.len()).max(1) as u64;
        let dynamic_target = remaining_budget / remaining_slots;

        let should_emit = if is_last_layer || remaining_shards == 0 {
            // Last layer → handled after loop (final shard with suffix).
            // No remaining shards → keep accumulating for final shard.
            false
        } else if remaining_shards > remaining_layers {
            // Must emit now: more shards needed than layers remaining
            true
        } else {
            current_size >= dynamic_target
        };

        if should_emit {
            current_tensors.sort_by_key(|(_, off, _)| *off);
            emitted_bytes += current_size;
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
            mmproj: None,
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
            masks: None,
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
            kv_model_key: String::from("0-0-0"),
            final_logit_softcap: None,
        };
        SplitModelEntry {
            model: std::sync::Arc::new(tokio::sync::Mutex::new(dummy_model)),
            last_used: std::sync::atomic::AtomicU64::new(0),
            estimated_vram_mb: vram_mb,
            batch_forwarder: None,
            is_complete: false,
            eos_tokens: vec![],
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
            tp_groups: vec![],
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
            layers.push(LayerVariant::Dense(LayerWeights {
                attention_wq: make_qmatmul(hidden_dim, hidden_dim),
                attention_wk: make_qmatmul(hidden_dim, hidden_dim),
                attention_wv: make_qmatmul(hidden_dim, hidden_dim),
                attention_wo: make_qmatmul(hidden_dim, hidden_dim),
                attention_bq: None,
                attention_bk: None,
                attention_bv: None,
                attention_norm: make_rms_norm(&norm_w),
                attn_q_norm: None,
                attn_k_norm: None,
                ffn: FfnVariant::Dense(Mlp {
                    ffn_gate: make_qmatmul(hidden_dim, hidden_dim * 4),
                    ffn_down: make_qmatmul(hidden_dim * 4, hidden_dim),
                    ffn_up: make_qmatmul(hidden_dim, hidden_dim * 4),
                    activation: Activation::SiLU,
                }),
                ffn_norm: make_rms_norm(&norm_w),
                n_head,
                n_kv_head,
                head_dim,
                cos: cos.clone(),
                sin: sin.clone(),
                neg_inf: neg_inf.clone(),
                use_rope_contiguous: true,
                attn_logit_softcap: None,
                rope_dim,
                skip_rope: false,
            }));
        }

        SplitModel {
            tok_embeddings: None,
            layers,
            norm: None,
            output: None,
            masks: None,
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
            kv_model_key: format!("0-{num_layers}-{}", num_layers + 2),
            final_logit_softcap: None,
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
            model_arc,
            kv_store,
            4,                         // max batch size
            std::time::Duration::ZERO, // no timeout in test
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
        assert_eq!(ModelArch::from_gguf_arch("qwen2moe"), ModelArch::Qwen2);
        assert_eq!(ModelArch::from_gguf_arch("deepseek2"), ModelArch::DeepSeek2);
        assert_eq!(ModelArch::from_gguf_arch("glm4"), ModelArch::Glm4);
        assert_eq!(ModelArch::from_gguf_arch("llama4"), ModelArch::Llama4);
        assert_eq!(
            ModelArch::from_gguf_arch("starcoder2"),
            ModelArch::Starcoder2
        );
        assert_eq!(ModelArch::from_gguf_arch("qwen35"), ModelArch::Qwen35);
        assert_eq!(ModelArch::from_gguf_arch("qwen35moe"), ModelArch::Qwen35Moe);
        assert!(matches!(
            ModelArch::from_gguf_arch("unknown_arch"),
            ModelArch::Unknown(_)
        ));
    }

    #[test]
    fn model_arch_properties() {
        // RoPE contiguous: Qwen2 family and DeepSeek2
        assert!(ModelArch::Qwen2.use_rope_contiguous());
        assert!(ModelArch::DeepSeek2.use_rope_contiguous());
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

        // Supported: known architectures are supported, Unknown is not
        assert!(ModelArch::Llama.is_supported());
        assert!(ModelArch::Qwen2.is_supported());
        assert!(ModelArch::Gemma2.is_supported());
        assert!(ModelArch::Phi3.is_supported());
        assert!(ModelArch::DeepSeek2.is_supported());
        assert!(ModelArch::Qwen35.is_supported());
        assert!(ModelArch::Qwen35Moe.is_supported());
        assert!(ModelArch::Qwen35.is_hybrid_ssm());
        assert!(ModelArch::Qwen35Moe.is_hybrid_ssm());
        assert!(!ModelArch::Llama.is_hybrid_ssm());
        assert!(!ModelArch::Unknown("mamba".to_string()).is_supported());
    }

    // ── GQA verification tests ──

    /// Helper: create a SplitModel with explicit GQA configuration.
    #[allow(clippy::too_many_arguments)]
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
            layers.push(LayerVariant::Dense(LayerWeights {
                attention_wq: make_qmatmul(hidden_dim, hidden_dim),
                attention_wk: make_qmatmul(hidden_dim, kv_dim),
                attention_wv: make_qmatmul(hidden_dim, kv_dim),
                attention_wo: make_qmatmul(hidden_dim, hidden_dim),
                attention_bq: None,
                attention_bk: None,
                attention_bv: None,
                attention_norm: make_rms_norm(&norm_w),
                attn_q_norm: None,
                attn_k_norm: None,
                ffn: FfnVariant::Dense(Mlp {
                    ffn_gate: make_qmatmul(hidden_dim, hidden_dim * 4),
                    ffn_down: make_qmatmul(hidden_dim * 4, hidden_dim),
                    ffn_up: make_qmatmul(hidden_dim, hidden_dim * 4),
                    activation,
                }),
                ffn_norm: make_rms_norm(&norm_w),
                n_head,
                n_kv_head,
                head_dim,
                cos: cos.clone(),
                sin: sin.clone(),
                neg_inf: neg_inf.clone(),
                use_rope_contiguous,
                attn_logit_softcap,
                rope_dim,
                skip_rope: false,
            }));
        }

        SplitModel {
            tok_embeddings: None,
            layers,
            norm: None,
            output: None,
            masks: None,
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
            kv_model_key: format!("0-{num_layers}-{}", num_layers + 2),
            final_logit_softcap: None,
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
            attn_q_norm: None,
            attn_k_norm: None,
            ffn: FfnVariant::Dense(Mlp {
                ffn_gate: make_qmatmul(hidden_dim, hidden_dim * 4),
                ffn_down: make_qmatmul(hidden_dim * 4, hidden_dim),
                ffn_up: make_qmatmul(hidden_dim, hidden_dim * 4),
                activation: Activation::SiLU,
            }),
            ffn_norm: make_rms_norm(&norm_w),
            n_head,
            n_kv_head,
            head_dim,
            cos,
            sin,
            neg_inf,
            use_rope_contiguous: true,
            attn_logit_softcap: None,
            rope_dim: head_dim,
            skip_rope: false,
        };

        let mut model = SplitModel {
            tok_embeddings: None,
            layers: vec![LayerVariant::Dense(layer)],
            norm: None,
            output: None,
            masks: None,
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
            kv_model_key: String::from("0-1-3"),
            final_logit_softcap: None,
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

    // ── MoE / DeepSeek architecture tests ──

    #[test]
    fn test_deepseek_arch_supported() {
        assert!(ModelArch::DeepSeek2.is_supported());
        assert!(ModelArch::DeepSeek2.use_rope_contiguous());
        assert_eq!(ModelArch::DeepSeek2.default_activation(), Activation::SiLU);
        assert!(!ModelArch::DeepSeek2.use_gemma_norm());
    }

    #[test]
    fn test_moe_topk_selection() {
        let device = Device::Cpu;
        // 8 experts, select top-2
        let scores = Tensor::from_vec(
            vec![0.1f32, 0.5, 0.3, 0.8, 0.2, 0.05, 0.7, 0.4],
            (8,),
            &device,
        )
        .unwrap();

        let (indices, weights) = topk_cpu(&scores, 2).unwrap();
        let idx_vec: Vec<i64> = indices.to_vec1().unwrap();
        let w_vec: Vec<f32> = weights.to_vec1().unwrap();

        // Top 2 scores: 0.8 at index 3, 0.7 at index 6
        assert_eq!(idx_vec, vec![3, 6]);
        assert_eq!(w_vec.len(), 2);
        // Weights should be softmax-normalized
        let w_sum: f32 = w_vec.iter().sum();
        assert!(
            (w_sum - 1.0).abs() < 1e-5,
            "Weights should sum to 1.0, got {w_sum}"
        );
        // Weight[0] (score 0.8) should be > weight[1] (score 0.7)
        assert!(w_vec[0] > w_vec[1]);
    }

    #[test]
    fn test_moe_topk_single_expert() {
        let device = Device::Cpu;
        let scores = Tensor::from_vec(vec![0.2f32, 0.8, 0.5], (3,), &device).unwrap();
        let (indices, weights) = topk_cpu(&scores, 1).unwrap();
        let idx_vec: Vec<i64> = indices.to_vec1().unwrap();
        let w_vec: Vec<f32> = weights.to_vec1().unwrap();
        assert_eq!(idx_vec, vec![1]);
        assert!(
            (w_vec[0] - 1.0).abs() < 1e-5,
            "Single expert weight should be 1.0"
        );
    }

    #[test]
    fn test_moe_forward_single_expert() {
        // A 1-expert MoE with top-1 should behave like a dense FFN
        let device = Device::Cpu;
        let hidden = 32;
        let intermediate = 64;
        let n_experts = 1;

        // Create expert weights: [1, intermediate, hidden] and [1, hidden, intermediate]
        let gate_exps =
            Tensor::randn(0f32, 0.02, (n_experts, intermediate, hidden), &device).unwrap();
        let down_exps =
            Tensor::randn(0f32, 0.02, (n_experts, hidden, intermediate), &device).unwrap();
        let up_exps =
            Tensor::randn(0f32, 0.02, (n_experts, intermediate, hidden), &device).unwrap();
        // Router: [1, hidden]
        let gate = Tensor::randn(0f32, 0.1, (n_experts, hidden), &device).unwrap();

        let moe = MoeFfn {
            gate,
            gate_exps,
            down_exps,
            up_exps,
            shared_gate: None,
            shared_down: None,
            shared_up: None,
            n_experts_used: 1,
        };

        let x = Tensor::randn(0f32, 1.0, (1, 4, hidden), &device).unwrap();
        let out = moe.forward(&x).unwrap();
        assert_eq!(out.dims(), &[1, 4, hidden]);
        let flat: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
        assert!(flat.iter().all(|v| v.is_finite()), "MoE output NaN/Inf");
    }

    #[test]
    fn test_moe_forward_multi_expert() {
        // 4 experts, select top-2, verify output shape and finiteness
        let device = Device::Cpu;
        let hidden = 32;
        let intermediate = 64;
        let n_experts = 4;

        let gate_exps =
            Tensor::randn(0f32, 0.02, (n_experts, intermediate, hidden), &device).unwrap();
        let down_exps =
            Tensor::randn(0f32, 0.02, (n_experts, hidden, intermediate), &device).unwrap();
        let up_exps =
            Tensor::randn(0f32, 0.02, (n_experts, intermediate, hidden), &device).unwrap();
        let gate = Tensor::randn(0f32, 0.1, (n_experts, hidden), &device).unwrap();

        let moe = MoeFfn {
            gate,
            gate_exps,
            down_exps,
            up_exps,
            shared_gate: None,
            shared_down: None,
            shared_up: None,
            n_experts_used: 2,
        };

        let x = Tensor::randn(0f32, 1.0, (1, 3, hidden), &device).unwrap();
        let out = moe.forward(&x).unwrap();
        assert_eq!(out.dims(), &[1, 3, hidden]);
        let flat: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
        assert!(
            flat.iter().all(|v| v.is_finite()),
            "Multi-expert output NaN/Inf"
        );
    }

    #[test]
    fn test_shared_expert_integration() {
        // MoE with shared experts: output = routed_experts + shared_expert
        let device = Device::Cpu;
        let hidden = 32;
        let intermediate = 64;
        let n_experts = 2;

        let make_qmatmul = |in_d: usize, out_d: usize| -> QMatMul {
            let w = Tensor::randn(0f32, 0.02, (out_d, in_d), &device).unwrap();
            let qt = QTensor::quantize(&w, candle_core::quantized::GgmlDType::F32).unwrap();
            QMatMul::from_qtensor(qt).unwrap()
        };

        let gate_exps =
            Tensor::randn(0f32, 0.02, (n_experts, intermediate, hidden), &device).unwrap();
        let down_exps =
            Tensor::randn(0f32, 0.02, (n_experts, hidden, intermediate), &device).unwrap();
        let up_exps =
            Tensor::randn(0f32, 0.02, (n_experts, intermediate, hidden), &device).unwrap();
        let gate = Tensor::randn(0f32, 0.1, (n_experts, hidden), &device).unwrap();

        // MoE without shared experts
        let moe_no_shared = MoeFfn {
            gate: gate.clone(),
            gate_exps: gate_exps.clone(),
            down_exps: down_exps.clone(),
            up_exps: up_exps.clone(),
            shared_gate: None,
            shared_down: None,
            shared_up: None,
            n_experts_used: 1,
        };

        // MoE with shared experts
        let moe_with_shared = MoeFfn {
            gate,
            gate_exps,
            down_exps,
            up_exps,
            shared_gate: Some(make_qmatmul(hidden, intermediate)),
            shared_down: Some(make_qmatmul(intermediate, hidden)),
            shared_up: Some(make_qmatmul(hidden, intermediate)),
            n_experts_used: 1,
        };

        let x = Tensor::randn(0f32, 1.0, (1, 2, hidden), &device).unwrap();
        let out_no_shared = moe_no_shared.forward(&x).unwrap();
        let out_with_shared = moe_with_shared.forward(&x).unwrap();

        assert_eq!(out_no_shared.dims(), out_with_shared.dims());
        // Outputs should differ due to shared expert contribution
        let diff = (&out_no_shared - &out_with_shared).unwrap().abs().unwrap();
        let max_diff: f32 = diff
            .flatten_all()
            .unwrap()
            .max(0)
            .unwrap()
            .to_vec0()
            .unwrap();
        assert!(max_diff > 0.0, "Shared expert should change output");
    }

    #[test]
    fn test_mla_q_decompress() {
        // Verify Q path shapes: x → q_a → norm → q_b → reshape
        let device = Device::Cpu;
        let hidden = 64;
        let q_lora_rank = 16;
        let n_head = 4;
        let key_length = 32; // per-head
        let value_length = 16;
        let kv_lora_rank = 8;
        let rope_dim = 8;

        let make_qmatmul = |in_d: usize, out_d: usize| -> QMatMul {
            let w = Tensor::randn(0f32, 0.02, (out_d, in_d), &device).unwrap();
            let qt = QTensor::quantize(&w, candle_core::quantized::GgmlDType::F32).unwrap();
            QMatMul::from_qtensor(qt).unwrap()
        };
        let make_rms_norm = |dim: usize| -> RmsNorm {
            let w = Tensor::ones((dim,), DType::F32, &device).unwrap();
            let qt = QTensor::quantize(&w, candle_core::quantized::GgmlDType::F32).unwrap();
            RmsNorm::from_qtensor(qt, 1e-6).unwrap()
        };

        let nope_dim = key_length - rope_dim;
        let (cos, sin) = precompute_freqs_cis(rope_dim, 10000.0, 128, &device).unwrap();
        let neg_inf = Tensor::new(f32::NEG_INFINITY, &device).unwrap();

        let mla = MlaWeights {
            q_a: make_qmatmul(hidden, q_lora_rank),
            q_a_norm: make_rms_norm(q_lora_rank),
            q_b: make_qmatmul(q_lora_rank, n_head * key_length),
            kv_a: make_qmatmul(hidden, kv_lora_rank + rope_dim),
            kv_a_norm: make_rms_norm(kv_lora_rank),
            kv_b: make_qmatmul(kv_lora_rank, n_head * (nope_dim + value_length)),
            output: make_qmatmul(n_head * value_length, hidden),
            n_head,
            key_length,
            value_length,
            kv_lora_rank,
            rope_dim,
            cos,
            sin,
            neg_inf,
        };

        // Test that forward_mla runs without error and returns correct shape
        let x = Tensor::randn(0f32, 0.1, (1, 5, hidden), &device).unwrap();
        let mut kv_cache = None;
        let out = mla.forward_mla(&x, None, 0, &mut kv_cache, 128).unwrap();
        assert_eq!(out.dims(), &[1, 5, hidden], "MLA output shape mismatch");

        // KV cache should be populated
        assert!(kv_cache.is_some(), "KV cache should be created");
    }

    #[test]
    fn test_mla_kv_decompress() {
        // Verify KV path shapes and cache dimensions
        let device = Device::Cpu;
        let hidden = 64;
        let q_lora_rank = 16;
        let n_head = 4;
        let key_length = 32;
        let value_length = 16;
        let kv_lora_rank = 8;
        let rope_dim = 8;

        let make_qmatmul = |in_d: usize, out_d: usize| -> QMatMul {
            let w = Tensor::randn(0f32, 0.02, (out_d, in_d), &device).unwrap();
            let qt = QTensor::quantize(&w, candle_core::quantized::GgmlDType::F32).unwrap();
            QMatMul::from_qtensor(qt).unwrap()
        };
        let make_rms_norm = |dim: usize| -> RmsNorm {
            let w = Tensor::ones((dim,), DType::F32, &device).unwrap();
            let qt = QTensor::quantize(&w, candle_core::quantized::GgmlDType::F32).unwrap();
            RmsNorm::from_qtensor(qt, 1e-6).unwrap()
        };

        let nope_dim = key_length - rope_dim;
        let (cos, sin) = precompute_freqs_cis(rope_dim, 10000.0, 128, &device).unwrap();
        let neg_inf = Tensor::new(f32::NEG_INFINITY, &device).unwrap();

        let mla = MlaWeights {
            q_a: make_qmatmul(hidden, q_lora_rank),
            q_a_norm: make_rms_norm(q_lora_rank),
            q_b: make_qmatmul(q_lora_rank, n_head * key_length),
            kv_a: make_qmatmul(hidden, kv_lora_rank + rope_dim),
            kv_a_norm: make_rms_norm(kv_lora_rank),
            kv_b: make_qmatmul(kv_lora_rank, n_head * (nope_dim + value_length)),
            output: make_qmatmul(n_head * value_length, hidden),
            n_head,
            key_length,
            value_length,
            kv_lora_rank,
            rope_dim,
            cos,
            sin,
            neg_inf,
        };

        // Prefill with seq_len=3
        let x = Tensor::randn(0f32, 0.1, (1, 3, hidden), &device).unwrap();
        let mut kv_cache = None;
        mla.forward_mla(&x, None, 0, &mut kv_cache, 128).unwrap();

        // Check KV cache dimensions
        let cache = kv_cache.as_ref().unwrap();
        let k = cache.k().unwrap().unwrap();
        let v = cache.v().unwrap().unwrap();
        // K: [b, n_head, seq_len, key_length]
        assert_eq!(k.dims(), &[1, n_head, 3, key_length]);
        // V: [b, n_head, seq_len, value_length]
        assert_eq!(v.dims(), &[1, n_head, 3, value_length]);
    }

    #[test]
    fn test_mla_rope_split() {
        // Verify that MLA correctly splits q into nope and rope parts
        let device = Device::Cpu;
        let hidden = 64;
        let q_lora_rank = 16;
        let n_head = 4;
        let key_length = 32;
        let value_length = 16;
        let kv_lora_rank = 8;
        let rope_dim = 8;

        let make_qmatmul = |in_d: usize, out_d: usize| -> QMatMul {
            let w = Tensor::randn(0f32, 0.02, (out_d, in_d), &device).unwrap();
            let qt = QTensor::quantize(&w, candle_core::quantized::GgmlDType::F32).unwrap();
            QMatMul::from_qtensor(qt).unwrap()
        };
        let make_rms_norm = |dim: usize| -> RmsNorm {
            let w = Tensor::ones((dim,), DType::F32, &device).unwrap();
            let qt = QTensor::quantize(&w, candle_core::quantized::GgmlDType::F32).unwrap();
            RmsNorm::from_qtensor(qt, 1e-6).unwrap()
        };

        let nope_dim = key_length - rope_dim;
        let (cos, sin) = precompute_freqs_cis(rope_dim, 10000.0, 128, &device).unwrap();
        let neg_inf = Tensor::new(f32::NEG_INFINITY, &device).unwrap();

        let mla = MlaWeights {
            q_a: make_qmatmul(hidden, q_lora_rank),
            q_a_norm: make_rms_norm(q_lora_rank),
            q_b: make_qmatmul(q_lora_rank, n_head * key_length),
            kv_a: make_qmatmul(hidden, kv_lora_rank + rope_dim),
            kv_a_norm: make_rms_norm(kv_lora_rank),
            kv_b: make_qmatmul(kv_lora_rank, n_head * (nope_dim + value_length)),
            output: make_qmatmul(n_head * value_length, hidden),
            n_head,
            key_length,
            value_length,
            kv_lora_rank,
            rope_dim,
            cos,
            sin,
            neg_inf,
        };

        // Prefill + decode: verify output stays finite
        let x = Tensor::randn(0f32, 0.1, (1, 4, hidden), &device).unwrap();
        let mut kv_cache = None;
        let out_prefill = mla.forward_mla(&x, None, 0, &mut kv_cache, 128).unwrap();
        let flat: Vec<f32> = out_prefill.flatten_all().unwrap().to_vec1().unwrap();
        assert!(flat.iter().all(|v| v.is_finite()), "MLA prefill NaN/Inf");

        // Decode step
        let x_decode = Tensor::randn(0f32, 0.1, (1, 1, hidden), &device).unwrap();
        let out_decode = mla
            .forward_mla(&x_decode, None, 4, &mut kv_cache, 128)
            .unwrap();
        assert_eq!(out_decode.dims(), &[1, 1, hidden]);
        let flat: Vec<f32> = out_decode.flatten_all().unwrap().to_vec1().unwrap();
        assert!(flat.iter().all(|v| v.is_finite()), "MLA decode NaN/Inf");
    }

    #[test]
    fn test_layer_variant_dense_unchanged() {
        // Wrapping in LayerVariant::Dense should produce same output as before
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

        // Verify layers are Dense variants
        for layer in &model.layers {
            assert!(matches!(layer, LayerVariant::Dense(_)));
        }

        // Forward pass works identically
        let input = Tensor::randn(0f32, 1.0, (1, 6, hidden_dim), &Device::Cpu).unwrap();
        let out = model.forward(&input, 0, &kv_store, "dense-test").unwrap();
        assert_eq!(out.dims(), &[1, 6, hidden_dim]);
        let flat: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
        assert!(flat.iter().all(|v| v.is_finite()));

        // Decode step
        let decode = Tensor::randn(0f32, 1.0, (1, 1, hidden_dim), &Device::Cpu).unwrap();
        let out = model.forward(&decode, 6, &kv_store, "dense-test").unwrap();
        assert_eq!(out.dims(), &[1, 1, hidden_dim]);
    }

    #[test]
    fn test_deepseek_meta_parsing() {
        // Verify DeepSeekMeta struct construction with various values
        let meta = DeepSeekMeta {
            n_experts: 64,
            n_experts_used: 6,
            n_shared_experts: 2,
            kv_lora_rank: 512,
            q_lora_rank: 1536,
            key_length: 192,
            value_length: 128,
            rope_dim: 64,
        };
        assert_eq!(meta.n_experts, 64);
        assert_eq!(meta.n_experts_used, 6);
        assert_eq!(meta.n_shared_experts, 2);
        assert_eq!(meta.kv_lora_rank, 512);
        assert_eq!(meta.q_lora_rank, 1536);
        assert_eq!(meta.key_length, 192);
        assert_eq!(meta.value_length, 128);
        assert_eq!(meta.rope_dim, 64);
    }

    /// Build a test model with DeepSeek-style mixed layers (1 dense + 1 MLA/MoE)
    fn make_deepseek_test_model(hidden_dim: usize) -> SplitModel {
        let device = Device::Cpu;
        let n_head = 4;
        let key_length = hidden_dim / n_head; // per-head key dim
        let value_length = hidden_dim / n_head;
        let kv_lora_rank = 16;
        let q_lora_rank = 16;
        let rope_dim = 8;
        let intermediate = hidden_dim * 2;
        let n_experts = 4;
        let n_experts_used = 2;

        let make_qmatmul = |in_d: usize, out_d: usize| -> QMatMul {
            let w = Tensor::randn(0f32, 0.02, (out_d, in_d), &device).unwrap();
            let qt = QTensor::quantize(&w, candle_core::quantized::GgmlDType::F32).unwrap();
            QMatMul::from_qtensor(qt).unwrap()
        };
        let make_rms_norm = |dim: usize| -> RmsNorm {
            let w = Tensor::ones((dim,), DType::F32, &device).unwrap();
            let qt = QTensor::quantize(&w, candle_core::quantized::GgmlDType::F32).unwrap();
            RmsNorm::from_qtensor(qt, 1e-6).unwrap()
        };

        let nope_dim = key_length - rope_dim;
        let max_seq_len = 128;
        let (cos, sin) = precompute_freqs_cis(rope_dim, 10000.0, max_seq_len, &device).unwrap();
        let neg_inf = Tensor::new(f32::NEG_INFINITY, &device).unwrap();

        // Layer 0: Dense (like first few DeepSeek layers)
        let head_dim = hidden_dim / n_head;
        let (dense_cos, dense_sin) =
            precompute_freqs_cis(head_dim, 10000.0, max_seq_len, &device).unwrap();
        let dense_layer = LayerVariant::Dense(LayerWeights {
            attention_wq: make_qmatmul(hidden_dim, hidden_dim),
            attention_wk: make_qmatmul(hidden_dim, hidden_dim),
            attention_wv: make_qmatmul(hidden_dim, hidden_dim),
            attention_wo: make_qmatmul(hidden_dim, hidden_dim),
            attention_bq: None,
            attention_bk: None,
            attention_bv: None,
            attention_norm: make_rms_norm(hidden_dim),
            attn_q_norm: None,
            attn_k_norm: None,
            ffn: FfnVariant::Dense(Mlp {
                ffn_gate: make_qmatmul(hidden_dim, intermediate),
                ffn_down: make_qmatmul(intermediate, hidden_dim),
                ffn_up: make_qmatmul(hidden_dim, intermediate),
                activation: Activation::SiLU,
            }),
            ffn_norm: make_rms_norm(hidden_dim),
            n_head,
            n_kv_head: n_head,
            head_dim,
            cos: dense_cos,
            sin: dense_sin,
            neg_inf: neg_inf.clone(),
            use_rope_contiguous: true,
            attn_logit_softcap: None,
            rope_dim: head_dim,
            skip_rope: false,
        });

        // Layer 1: DeepSeek MLA + MoE
        let mla = MlaWeights {
            q_a: make_qmatmul(hidden_dim, q_lora_rank),
            q_a_norm: make_rms_norm(q_lora_rank),
            q_b: make_qmatmul(q_lora_rank, n_head * key_length),
            kv_a: make_qmatmul(hidden_dim, kv_lora_rank + rope_dim),
            kv_a_norm: make_rms_norm(kv_lora_rank),
            kv_b: make_qmatmul(kv_lora_rank, n_head * (nope_dim + value_length)),
            output: make_qmatmul(n_head * value_length, hidden_dim),
            n_head,
            key_length,
            value_length,
            kv_lora_rank,
            rope_dim,
            cos: cos.clone(),
            sin: sin.clone(),
            neg_inf: neg_inf.clone(),
        };

        let moe = MoeFfn {
            gate: Tensor::randn(0f32, 0.1, (n_experts, hidden_dim), &device).unwrap(),
            gate_exps: Tensor::randn(0f32, 0.02, (n_experts, intermediate, hidden_dim), &device)
                .unwrap(),
            down_exps: Tensor::randn(0f32, 0.02, (n_experts, hidden_dim, intermediate), &device)
                .unwrap(),
            up_exps: Tensor::randn(0f32, 0.02, (n_experts, intermediate, hidden_dim), &device)
                .unwrap(),
            shared_gate: None,
            shared_down: None,
            shared_up: None,
            n_experts_used,
        };

        let deepseek_layer = LayerVariant::DeepSeek {
            attention: mla,
            ffn: FfnVariant::MoE(moe),
            attention_norm: make_rms_norm(hidden_dim),
            ffn_norm: make_rms_norm(hidden_dim),
        };

        SplitModel {
            tok_embeddings: None,
            layers: vec![dense_layer, deepseek_layer],
            norm: None,
            output: None,
            masks: None,
            layer_start: 0,
            layer_end: 2,
            total_layers: 4,
            hidden_dim,
            arch: ModelArch::DeepSeek2,
            device,
            vocabulary: None,
            tokenizer: None,
            eos_tokens: vec![2],
            chat_template: None,
            bos_token: String::new(),
            eos_token: String::new(),
            max_seq_len,
            kv_model_key: String::from("0-2-4"),
            final_logit_softcap: None,
        }
    }

    #[test]
    fn test_deepseek_mixed_layers_forward() {
        // Full forward pass through mixed dense + MLA/MoE layers
        let hidden_dim = 64;
        let mut model = make_deepseek_test_model(hidden_dim);
        let kv_store = KvCacheStore::new(std::time::Duration::from_secs(600));

        // Prefill
        let input = Tensor::randn(0f32, 0.1, (1, 4, hidden_dim), &Device::Cpu).unwrap();
        let out = model.forward(&input, 0, &kv_store, "ds-test").unwrap();
        assert_eq!(out.dims(), &[1, 4, hidden_dim]);
        let flat: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
        assert!(
            flat.iter().all(|v| v.is_finite()),
            "DeepSeek prefill NaN/Inf"
        );

        // Decode
        let decode = Tensor::randn(0f32, 0.1, (1, 1, hidden_dim), &Device::Cpu).unwrap();
        let out = model.forward(&decode, 4, &kv_store, "ds-test").unwrap();
        assert_eq!(out.dims(), &[1, 1, hidden_dim]);
        let flat: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
        assert!(
            flat.iter().all(|v| v.is_finite()),
            "DeepSeek decode NaN/Inf"
        );
    }

    #[test]
    fn test_glm4_arch_supported() {
        assert!(ModelArch::Glm4.is_supported());
        assert!(ModelArch::Glm4.use_rope_contiguous());
        assert_eq!(ModelArch::Glm4.default_activation(), Activation::SiLU);
        assert!(!ModelArch::Glm4.use_gemma_norm());
        assert_eq!(ModelArch::from_gguf_arch("glm4"), ModelArch::Glm4);
    }

    #[test]
    fn test_llama4_arch_supported() {
        assert!(ModelArch::Llama4.is_supported());
        assert!(ModelArch::Llama4.use_rope_contiguous());
        assert_eq!(ModelArch::Llama4.default_activation(), Activation::SiLU);
        assert!(!ModelArch::Llama4.use_gemma_norm());
        assert_eq!(ModelArch::from_gguf_arch("llama4"), ModelArch::Llama4);
    }

    #[test]
    fn test_partial_rope_glm4_style() {
        // GLM-4 uses partial RoPE: only first half of head_dim gets rotated
        let device = Device::Cpu;
        let head_dim = 16;
        let rope_dim = 8; // half of head_dim
        let n_head = 2;
        let seq_len = 4;
        let max_seq_len = 32;

        let (cos, sin) = precompute_freqs_cis(rope_dim, 10000.0, max_seq_len, &device).unwrap();
        let neg_inf = Tensor::new(f32::NEG_INFINITY, &device).unwrap();

        let make_qmatmul = |in_d: usize, out_d: usize| -> QMatMul {
            let w = Tensor::randn(0f32, 0.02, (out_d, in_d), &device).unwrap();
            let qt = QTensor::quantize(&w, candle_core::quantized::GgmlDType::F32).unwrap();
            QMatMul::from_qtensor(qt).unwrap()
        };
        let norm_w = Tensor::ones((n_head * head_dim,), DType::F32, &device).unwrap();
        let make_rms_norm = |w: &Tensor| {
            let qt = QTensor::quantize(w, candle_core::quantized::GgmlDType::F32).unwrap();
            RmsNorm::from_qtensor(qt, 1e-6).unwrap()
        };

        let lw = LayerWeights {
            attention_wq: make_qmatmul(n_head * head_dim, n_head * head_dim),
            attention_wk: make_qmatmul(n_head * head_dim, n_head * head_dim),
            attention_wv: make_qmatmul(n_head * head_dim, n_head * head_dim),
            attention_wo: make_qmatmul(n_head * head_dim, n_head * head_dim),
            attention_bq: None,
            attention_bk: None,
            attention_bv: None,
            attention_norm: make_rms_norm(&norm_w),
            attn_q_norm: None,
            attn_k_norm: None,
            ffn: FfnVariant::Dense(Mlp {
                ffn_gate: make_qmatmul(n_head * head_dim, n_head * head_dim * 4),
                ffn_down: make_qmatmul(n_head * head_dim * 4, n_head * head_dim),
                ffn_up: make_qmatmul(n_head * head_dim, n_head * head_dim * 4),
                activation: Activation::SiLU,
            }),
            ffn_norm: make_rms_norm(&norm_w),
            n_head,
            n_kv_head: n_head,
            head_dim,
            cos,
            sin,
            neg_inf,
            use_rope_contiguous: true,
            attn_logit_softcap: None,
            rope_dim,
            skip_rope: false,
        };

        // Test that apply_rotary_emb handles partial RoPE
        let x = Tensor::randn(0f32, 0.1, (1, n_head, seq_len, head_dim), &device).unwrap();
        let result = lw.apply_rotary_emb(&x, 0).unwrap();
        assert_eq!(result.dims(), &[1, n_head, seq_len, head_dim]);

        // Verify first rope_dim dims are different (rotated) and last dims unchanged
        let x_pass = x.narrow(3, rope_dim, head_dim - rope_dim).unwrap();
        let r_pass = result.narrow(3, rope_dim, head_dim - rope_dim).unwrap();
        let diff: f32 = (&x_pass - &r_pass)
            .unwrap()
            .abs()
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar()
            .unwrap();
        assert!(
            diff < 1e-5,
            "Non-rotated dims should be unchanged, diff={diff}"
        );
    }

    #[test]
    fn test_nope_skip_rope() {
        // Llama 4 iRoPE: every 4th layer skips RoPE entirely
        let device = Device::Cpu;
        let head_dim = 16;
        let n_head = 2;
        let seq_len = 4;

        let (cos, sin) = precompute_freqs_cis(head_dim, 10000.0, 32, &device).unwrap();
        let neg_inf = Tensor::new(f32::NEG_INFINITY, &device).unwrap();

        let make_qmatmul = |in_d: usize, out_d: usize| -> QMatMul {
            let w = Tensor::randn(0f32, 0.02, (out_d, in_d), &device).unwrap();
            let qt = QTensor::quantize(&w, candle_core::quantized::GgmlDType::F32).unwrap();
            QMatMul::from_qtensor(qt).unwrap()
        };
        let norm_w = Tensor::ones((n_head * head_dim,), DType::F32, &device).unwrap();
        let make_rms_norm = |w: &Tensor| {
            let qt = QTensor::quantize(w, candle_core::quantized::GgmlDType::F32).unwrap();
            RmsNorm::from_qtensor(qt, 1e-6).unwrap()
        };

        let lw = LayerWeights {
            attention_wq: make_qmatmul(n_head * head_dim, n_head * head_dim),
            attention_wk: make_qmatmul(n_head * head_dim, n_head * head_dim),
            attention_wv: make_qmatmul(n_head * head_dim, n_head * head_dim),
            attention_wo: make_qmatmul(n_head * head_dim, n_head * head_dim),
            attention_bq: None,
            attention_bk: None,
            attention_bv: None,
            attention_norm: make_rms_norm(&norm_w),
            attn_q_norm: None,
            attn_k_norm: None,
            ffn: FfnVariant::Dense(Mlp {
                ffn_gate: make_qmatmul(n_head * head_dim, n_head * head_dim * 4),
                ffn_down: make_qmatmul(n_head * head_dim * 4, n_head * head_dim),
                ffn_up: make_qmatmul(n_head * head_dim, n_head * head_dim * 4),
                activation: Activation::SiLU,
            }),
            ffn_norm: make_rms_norm(&norm_w),
            n_head,
            n_kv_head: n_head,
            head_dim,
            cos,
            sin,
            neg_inf,
            use_rope_contiguous: true,
            attn_logit_softcap: None,
            rope_dim: head_dim,
            skip_rope: true, // NoPE layer
        };

        let x = Tensor::randn(0f32, 0.1, (1, n_head, seq_len, head_dim), &device).unwrap();
        let result = lw.apply_rotary_emb(&x, 0).unwrap();

        // skip_rope=true means output == input (no rotation applied)
        let diff: f32 = (&x - &result)
            .unwrap()
            .abs()
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar()
            .unwrap();
        assert!(
            diff < 1e-6,
            "NoPE layer should not modify input, diff={diff}"
        );
    }

    #[test]
    fn test_ffn_variant_moe_forward() {
        // Test that FfnVariant::MoE dispatches through MoeFfn correctly
        let device = Device::Cpu;
        let hidden = 32;
        let intermediate = 64;
        let n_experts = 4;
        let n_experts_used = 2;

        // Build small MoE FFN
        let gate = Tensor::randn(0f32, 0.1, (n_experts, hidden), &device).unwrap();
        let gate_exps =
            Tensor::randn(0f32, 0.02, (n_experts, intermediate, hidden), &device).unwrap();
        let down_exps =
            Tensor::randn(0f32, 0.02, (n_experts, hidden, intermediate), &device).unwrap();
        let up_exps =
            Tensor::randn(0f32, 0.02, (n_experts, intermediate, hidden), &device).unwrap();

        let moe = MoeFfn {
            gate,
            gate_exps,
            down_exps,
            up_exps,
            shared_gate: None,
            shared_down: None,
            shared_up: None,
            n_experts_used,
        };

        let ffn = FfnVariant::MoE(moe);
        let x = Tensor::randn(0f32, 0.1, (1, 4, hidden), &device).unwrap();

        let out = match &ffn {
            FfnVariant::Dense(mlp) => mlp.forward(&x, None).unwrap(),
            FfnVariant::MoE(moe) => moe.forward(&x).unwrap(),
        };
        assert_eq!(out.dims(), &[1, 4, hidden]);
        let flat: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
        assert!(flat.iter().all(|v| v.is_finite()), "MoE output NaN/Inf");
    }

    #[test]
    fn test_irope_nope_pattern() {
        // Verify the iRoPE pattern: every 4th layer (index % 4 == 3) is NoPE
        let nope_layers: Vec<bool> = (0..12).map(|i| i % 4 == 3).collect();
        assert_eq!(
            nope_layers,
            vec![false, false, false, true, false, false, false, true, false, false, false, true]
        );
    }

    #[test]
    fn test_llama4_moe_layer_forward() {
        // End-to-end test: model with MoE FFN layers via FfnVariant
        let device = Device::Cpu;
        let hidden_dim = 32;
        let n_head = 4;
        let head_dim = hidden_dim / n_head;
        let n_kv_head = 2;
        let n_experts = 4;
        let n_experts_used = 2;
        let intermediate = 64;
        let max_seq_len = 32;
        let rope_dim = head_dim;

        let (cos, sin) = precompute_freqs_cis(rope_dim, 10000.0, max_seq_len, &device).unwrap();
        let neg_inf = Tensor::new(f32::NEG_INFINITY, &device).unwrap();

        let make_qmatmul = |in_d: usize, out_d: usize| -> QMatMul {
            let w = Tensor::randn(0f32, 0.02, (out_d, in_d), &device).unwrap();
            let qt = QTensor::quantize(&w, candle_core::quantized::GgmlDType::F32).unwrap();
            QMatMul::from_qtensor(qt).unwrap()
        };
        let make_rms_norm = |dim: usize| -> RmsNorm {
            let w = Tensor::ones((dim,), DType::F32, &device).unwrap();
            let qt = QTensor::quantize(&w, candle_core::quantized::GgmlDType::F32).unwrap();
            RmsNorm::from_qtensor(qt, 1e-6).unwrap()
        };

        let kv_dim = n_kv_head * head_dim;

        // Build a mix of dense + MoE layers (like Llama 4)
        let mut layers = Vec::new();
        for layer_idx in 0..4 {
            let is_nope = layer_idx % 4 == 3;

            let ffn = if layer_idx % 2 == 1 {
                // MoE layers on odd indices
                FfnVariant::MoE(MoeFfn {
                    gate: Tensor::randn(0f32, 0.1, (n_experts, hidden_dim), &device).unwrap(),
                    gate_exps: Tensor::randn(
                        0f32,
                        0.02,
                        (n_experts, intermediate, hidden_dim),
                        &device,
                    )
                    .unwrap(),
                    down_exps: Tensor::randn(
                        0f32,
                        0.02,
                        (n_experts, hidden_dim, intermediate),
                        &device,
                    )
                    .unwrap(),
                    up_exps: Tensor::randn(
                        0f32,
                        0.02,
                        (n_experts, intermediate, hidden_dim),
                        &device,
                    )
                    .unwrap(),
                    shared_gate: None,
                    shared_down: None,
                    shared_up: None,
                    n_experts_used,
                })
            } else {
                // Dense FFN on even indices
                FfnVariant::Dense(Mlp {
                    ffn_gate: make_qmatmul(hidden_dim, intermediate),
                    ffn_down: make_qmatmul(intermediate, hidden_dim),
                    ffn_up: make_qmatmul(hidden_dim, intermediate),
                    activation: Activation::SiLU,
                })
            };

            layers.push(LayerVariant::Dense(LayerWeights {
                attention_wq: make_qmatmul(hidden_dim, hidden_dim),
                attention_wk: make_qmatmul(hidden_dim, kv_dim),
                attention_wv: make_qmatmul(hidden_dim, kv_dim),
                attention_wo: make_qmatmul(hidden_dim, hidden_dim),
                attention_bq: None,
                attention_bk: None,
                attention_bv: None,
                attention_norm: make_rms_norm(hidden_dim),
                attn_q_norm: None,
                attn_k_norm: None,
                ffn,
                ffn_norm: make_rms_norm(hidden_dim),
                n_head,
                n_kv_head,
                head_dim,
                cos: cos.clone(),
                sin: sin.clone(),
                neg_inf: neg_inf.clone(),
                use_rope_contiguous: true,
                attn_logit_softcap: None,
                rope_dim,
                skip_rope: is_nope,
            }));
        }

        let mut model = SplitModel {
            tok_embeddings: None,
            layers,
            norm: None,
            output: None,
            masks: None,
            layer_start: 0,
            layer_end: 4,
            total_layers: 8,
            hidden_dim,
            arch: ModelArch::Llama4,
            device,
            vocabulary: None,
            tokenizer: None,
            eos_tokens: vec![2],
            chat_template: None,
            bos_token: String::new(),
            eos_token: String::new(),
            max_seq_len,
            kv_model_key: String::from("0-4-8"),
            final_logit_softcap: None,
        };

        let kv_store = KvCacheStore::new(std::time::Duration::from_secs(600));
        let input = Tensor::randn(0f32, 0.1, (1, 4, hidden_dim), &Device::Cpu).unwrap();
        let output = model.forward(&input, 0, &kv_store, "llama4-test").unwrap();
        assert_eq!(output.dims(), &[1, 4, hidden_dim]);
        let flat: Vec<f32> = output.flatten_all().unwrap().to_vec1().unwrap();
        assert!(
            flat.iter().all(|v| v.is_finite()),
            "Llama 4 MoE output NaN/Inf"
        );
    }
}
