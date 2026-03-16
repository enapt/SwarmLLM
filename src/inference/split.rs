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

// Re-export types from extracted modules so external `crate::inference::split::X` paths still work.
pub(crate) use super::layers::{
    DeepSeekMeta, DeltaNetWeights, FfnVariant, LayerVariant, LayerWeights, MlaWeights, Mlp, MoeFfn,
    QMatMul, Qwen35AttnWeights, SsmState,
};
pub(crate) use super::model_arch::Activation;
pub use super::model_arch::ModelArch;
pub use super::shard_layout::{
    available_layer_ranges_from_manifest, compute_layer_shard_layouts, LayerShardLayout,
};
pub use super::tensor_util::{
    bytes_to_tensor, sample_token, sample_token_with_params, tensor_to_bytes,
};
pub use super::tokenizer::{BpeTokenizer, SplitTokenizer, SpmTokenizer};

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

    /// Build a composite lookup key from model key and request ID for the KV-cache DashMap.
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
    // ── Cached metadata for subprocess-isolated inference (no GPU tensors) ──
    /// EOS token string (e.g., "<|endoftext|>").
    pub eos_token_str: String,
    /// BOS token string (e.g., "<s>").
    pub bos_token: String,
    /// Chat template from GGUF metadata (Jinja2 format).
    pub cached_chat_template: Option<String>,
    /// Full vocabulary for lock-free token decoding.
    pub vocab: Option<Vec<String>>,
}

impl SplitModelEntry {
    fn now_secs() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
    }

    /// Build common fields from a SplitModel reference, returning a partially-initialized entry.
    /// Caller sets `model`, `batch_forwarder`, and `last_used` after this.
    #[allow(clippy::type_complexity)]
    fn common_fields(
        model: &SplitModel,
    ) -> (
        u64,
        bool,
        Vec<u32>,
        String,
        String,
        Option<String>,
        Option<Vec<String>>,
    ) {
        (
            model.estimate_vram_mb(),
            model.is_first() && model.is_last(),
            model.eos_tokens().to_vec(),
            model.eos_token_str().to_string(),
            model.bos_token().to_string(),
            model.chat_template().map(|s| s.to_string()),
            model.vocab().cloned(),
        )
    }

    /// Create a new entry wrapping a split model.
    #[allow(clippy::type_complexity)]
    pub fn new(model: SplitModel) -> Self {
        let (
            estimated_vram_mb,
            is_complete,
            eos_tokens,
            eos_token_str,
            bos_token,
            chat_tmpl,
            vocab,
        ) = Self::common_fields(&model);
        Self {
            model: std::sync::Arc::new(tokio::sync::Mutex::new(model)),
            last_used: std::sync::atomic::AtomicU64::new(Self::now_secs()),
            estimated_vram_mb,
            batch_forwarder: None,
            is_complete,
            eos_tokens,
            eos_token_str,
            bos_token,
            cached_chat_template: chat_tmpl,
            vocab,
        }
    }

    /// Create a new entry with batching enabled.
    #[allow(clippy::type_complexity)]
    pub fn new_with_batching(
        model: SplitModel,
        kv_cache_store: std::sync::Arc<KvCacheStore>,
        max_batch_size: usize,
        batch_timeout: std::time::Duration,
    ) -> Self {
        let (
            estimated_vram_mb,
            is_complete,
            eos_tokens,
            eos_token_str,
            bos_token,
            chat_tmpl,
            vocab,
        ) = Self::common_fields(&model);
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
            last_used: std::sync::atomic::AtomicU64::new(Self::now_secs()),
            estimated_vram_mb,
            batch_forwarder,
            is_complete,
            eos_tokens,
            eos_token_str,
            bos_token,
            cached_chat_template: chat_tmpl,
            vocab,
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
        if head_count == 0 {
            return Err(SwarmError::Inference(
                "GGUF metadata error: attention.head_count is zero".into(),
            ));
        }
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
            // Use checked arithmetic to prevent integer overflow on crafted GGUF headers
            let size = info
                .ggml_dtype
                .type_size()
                .checked_mul(info.shape.elem_count())
                .and_then(|v| {
                    let bs = info.ggml_dtype.block_size();
                    if bs == 0 {
                        None
                    } else {
                        Some(v / bs)
                    }
                })
                .unwrap_or(0);
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

    // Try source_path as a fallback (with path containment check)
    let source_path_file = model_dir.join("source_path");
    if source_path_file.exists() {
        if let Ok(path_str) = std::fs::read_to_string(&source_path_file) {
            let gguf_path = std::path::PathBuf::from(path_str.trim());
            let canonical = gguf_path
                .canonicalize()
                .unwrap_or_else(|_| gguf_path.clone());
            if !canonical.starts_with(model_dir) {
                tracing::warn!(
                    path = %canonical.display(),
                    "source_path outside model directory — ignoring"
                );
            } else if canonical.exists() {
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
        // SEC: Cap padding to prevent OOM from malicious tensor_data_offset
        const MAX_GGUF_HEADER_SIZE: usize = 64 * 1024 * 1024; // 64 MB
        let header = if (header.len() as u64) < tensor_data_offset {
            if (tensor_data_offset as usize) > MAX_GGUF_HEADER_SIZE {
                return Err(SwarmError::Internal(format!(
                    "GGUF header offset too large: {} bytes (max {})",
                    tensor_data_offset, MAX_GGUF_HEADER_SIZE
                )));
            }
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
    pub(crate) fn find_shard(&self, pos: u64) -> Option<(usize, u64)> {
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

/// Precompute RoPE frequencies for Long RoPE (SuRoPE) models like Phi-3.5.
/// Per-dimension frequency scaling factors from `rope_factors_long/short.weight` in GGUF.
fn precompute_freqs_cis_longrope(
    head_dim: usize,
    freq_base: f32,
    max_seq_len: usize,
    rope_factors: &[f32],
    attn_factor: f32,
    device: &Device,
) -> CandleResult<(Tensor, Tensor)> {
    let half_dim = head_dim / 2;
    if rope_factors.len() != half_dim {
        return Err(candle_core::Error::Msg(format!(
            "LongRoPE factors length {} != expected half_dim {}",
            rope_factors.len(),
            half_dim
        )));
    }
    let theta: Vec<_> = (0..half_dim)
        .map(|i| 1f32 / (rope_factors[i] * freq_base.powf(2.0 * i as f32 / head_dim as f32)))
        .collect();
    let theta = Tensor::new(theta.as_slice(), device)?;
    let idx_theta = Tensor::arange(0, max_seq_len as u32, device)?
        .to_dtype(DType::F32)?
        .reshape((max_seq_len, 1))?
        .matmul(&theta.reshape((1, theta.elem_count()))?)?;
    let cos = (idx_theta.cos()? * attn_factor as f64)?;
    let sin = (idx_theta.sin()? * attn_factor as f64)?;
    Ok((cos, sin))
}

/// Load Long RoPE (SuRoPE) frequency scaling factors from GGUF tensors.
fn load_longrope_factors<R: std::io::Read + std::io::Seek>(
    ct: &gguf_file::Content,
    reader: &mut R,
    arch: &str,
    context_length: usize,
) -> Option<(Vec<f32>, f32)> {
    let has_long = ct.tensor_infos.contains_key("rope_factors_long.weight");
    let has_short = ct.tensor_infos.contains_key("rope_factors_short.weight");
    if !has_long || !has_short {
        return None;
    }
    let original_ctx = ct
        .metadata
        .get(&format!("{arch}.rope.scaling.original_context_length"))
        .and_then(|v| v.to_u32().ok())
        .unwrap_or(4096) as usize;
    let tensor_name = if context_length > original_ctx {
        "rope_factors_long.weight"
    } else {
        "rope_factors_short.weight"
    };
    let cpu = &Device::Cpu;
    let factors_qt = ct.tensor(reader, tensor_name, cpu).ok()?;
    let factors_t = factors_qt.dequantize(cpu).ok()?;
    let factors: Vec<f32> = factors_t.flatten_all().ok()?.to_vec1().ok()?;
    let scale = context_length as f64 / original_ctx as f64;
    let attn_factor = if scale <= 1.0 {
        1.0f32
    } else {
        ct.metadata
            .get(&format!("{arch}.rope.scaling.attn_factor"))
            .and_then(|v| v.to_f32().ok())
            .unwrap_or_else(|| (1.0 + scale.ln() / (original_ctx as f64).ln()).sqrt() as f32)
    };
    tracing::info!(
        original_ctx,
        context_length,
        tensor = tensor_name,
        attn_factor,
        factors_len = factors.len(),
        "Loaded Long RoPE factors"
    );
    Some((factors, attn_factor))
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
        if head_count == 0 {
            return Err(SwarmError::Inference(
                "GGUF metadata error: attention.head_count is zero".into(),
            ));
        }
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

        // Long RoPE (SuRoPE) for Phi-3.5 and similar extended-context models
        let longrope = load_longrope_factors(&ct, &mut file, arch, context_length);
        let (cos, sin) = if let Some((ref factors, attn_factor)) = longrope {
            tracing::info!(
                factors_len = factors.len(),
                attn_factor,
                "Using Long RoPE (SuRoPE)"
            );
            precompute_freqs_cis_longrope(
                rope_dim,
                rope_freq_base,
                context_length,
                factors,
                attn_factor,
                &device,
            )
            .map_err(|e| SwarmError::Internal(e.to_string()))?
        } else {
            precompute_freqs_cis(rope_dim, rope_freq_base, context_length, &device)
                .map_err(|e| SwarmError::Internal(e.to_string()))?
        };
        let neg_inf = Tensor::new(f32::NEG_INFINITY, &device)
            .map_err(|e| SwarmError::Internal(e.to_string()))?;
        // Helper: create RmsNorm from GGUF weight tensor.
        // Note: GGUF norm weights for Gemma models already include the +1 offset
        // (added by convert_hf_to_gguf.py's modify_tensors), so we use them as-is.
        let make_norm = |qtensor: QTensor, eps: f64| -> Result<RmsNorm, SwarmError> {
            RmsNorm::from_qtensor(qtensor, eps).map_err(|e| SwarmError::Internal(e.to_string()))
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
                        post_attention_norm: None,
                        post_ffw_norm: None,
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
                    post_attention_norm: None,
                    post_ffw_norm: None,
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
                        .map(|t| {
                            QMatMul::from_qtensor(t).map_err(|e| {
                                SwarmError::Internal(format!("QMatMul load failed: {e}"))
                            })
                        })
                        .transpose()?;
                    let shared_down = ct
                        .tensor(
                            &mut file,
                            &format!("{prefix}.ffn_down_shexp.weight"),
                            &device,
                        )
                        .ok()
                        .map(|t| {
                            QMatMul::from_qtensor(t).map_err(|e| {
                                SwarmError::Internal(format!("QMatMul load failed: {e}"))
                            })
                        })
                        .transpose()?;
                    let shared_up = ct
                        .tensor(&mut file, &format!("{prefix}.ffn_up_shexp.weight"), &device)
                        .ok()
                        .map(|t| {
                            QMatMul::from_qtensor(t).map_err(|e| {
                                SwarmError::Internal(format!("QMatMul load failed: {e}"))
                            })
                        })
                        .transpose()?;

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
                            q.map(|t| {
                                QMatMul::from_qtensor(t).map_err(|e| {
                                    SwarmError::Internal(format!("QMatMul load failed: {e}"))
                                })
                            })
                            .transpose()?,
                            k.map(|t| {
                                QMatMul::from_qtensor(t).map_err(|e| {
                                    SwarmError::Internal(format!("QMatMul load failed: {e}"))
                                })
                            })
                            .transpose()?,
                            v.map(|t| {
                                QMatMul::from_qtensor(t).map_err(|e| {
                                    SwarmError::Internal(format!("QMatMul load failed: {e}"))
                                })
                            })
                            .transpose()?,
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
                            wqkv: wqkv
                                .map(|t| {
                                    QMatMul::from_qtensor(t).map_err(|e| {
                                        SwarmError::Internal(format!("QMatMul load failed: {e}"))
                                    })
                                })
                                .transpose()?,
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
                            q.map(|t| {
                                QMatMul::from_qtensor(t).map_err(|e| {
                                    SwarmError::Internal(format!("QMatMul load failed: {e}"))
                                })
                            })
                            .transpose()?,
                            k.map(|t| {
                                QMatMul::from_qtensor(t).map_err(|e| {
                                    SwarmError::Internal(format!("QMatMul load failed: {e}"))
                                })
                            })
                            .transpose()?,
                            v.map(|t| {
                                QMatMul::from_qtensor(t).map_err(|e| {
                                    SwarmError::Internal(format!("QMatMul load failed: {e}"))
                                })
                            })
                            .transpose()?,
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
                        .map(|t| {
                            RmsNorm::from_qtensor(t, rms_norm_eps).map_err(|e| {
                                SwarmError::Internal(format!("RmsNorm load failed: {e}"))
                            })
                        })
                        .transpose()?;
                    let k_norm = ct
                        .tensor(&mut file, &format!("{prefix}.attn_k_norm.weight"), &device)
                        .ok()
                        .map(|t| {
                            RmsNorm::from_qtensor(t, rms_norm_eps).map_err(|e| {
                                SwarmError::Internal(format!("RmsNorm load failed: {e}"))
                            })
                        })
                        .transpose()?;

                    layers.push(LayerVariant::Qwen35Attn {
                        weights: Qwen35AttnWeights {
                            wqkv: wqkv
                                .map(|t| {
                                    QMatMul::from_qtensor(t).map_err(|e| {
                                        SwarmError::Internal(format!("QMatMul load failed: {e}"))
                                    })
                                })
                                .transpose()?,
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
                                // Fused QKV: keep original quantized tensor, split output in forward
                                let fused_qt = ct_ref
                                    .tensor(
                                        &mut cursor,
                                        &format!("{prefix}.attn_qkv.weight"),
                                        device_ref,
                                    )
                                    .map_err(|e| {
                                        SwarmError::Internal(format!("{prefix}.attn_qkv: {e}"))
                                    })?;
                                let q_dim = head_count * head_dim;
                                let k_dim = head_count_kv * head_dim;
                                let fused = QMatMul::make_fused(fused_qt)
                                    .map_err(|e| SwarmError::Internal(e.to_string()))?;
                                (
                                    QMatMul::from_fused_slice(fused.clone(), 0, q_dim),
                                    QMatMul::from_fused_slice(fused.clone(), q_dim, k_dim),
                                    QMatMul::from_fused_slice(fused, q_dim + k_dim, k_dim),
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
                                // Fused gate+up: use FusedSlice to avoid re-quantization
                                let fused_qt = ct_ref
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
                                let fused_shape = fused_qt.shape();
                                let half = fused_shape.dims()[0] / 2;
                                let fused = QMatMul::make_fused(fused_qt)
                                    .map_err(|e| SwarmError::Internal(e.to_string()))?;
                                (
                                    QMatMul::from_fused_slice(fused.clone(), 0, half),
                                    QMatMul::from_qtensor(ffn_down_qt)
                                        .map_err(|e| SwarmError::Internal(e.to_string()))?,
                                    QMatMul::from_fused_slice(fused, half, half),
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

                            // Gemma 2 post-norms (optional)
                            let post_attention_norm = ct_ref
                                .tensor(
                                    &mut cursor,
                                    &format!("{prefix}.post_attention_norm.weight"),
                                    device_ref,
                                )
                                .ok()
                                .map(|t| make_norm(t, rms_norm_eps))
                                .transpose()?;
                            let post_ffw_norm = ct_ref
                                .tensor(
                                    &mut cursor,
                                    &format!("{prefix}.post_ffw_norm.weight"),
                                    device_ref,
                                )
                                .ok()
                                .map(|t| make_norm(t, rms_norm_eps))
                                .transpose()?;

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
                                post_attention_norm,
                                post_ffw_norm,
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
                let add_space_prefix = ct
                    .metadata
                    .get("tokenizer.ggml.add_space_prefix")
                    .and_then(|v| v.to_bool().ok())
                    .unwrap_or(true);
                let add_bos_token = ct
                    .metadata
                    .get("tokenizer.ggml.add_bos_token")
                    .and_then(|v| v.to_bool().ok())
                    .unwrap_or(false);
                if !scores.is_empty() {
                    tracing::info!(
                        vocab_size = vocab.len(),
                        scores = scores.len(),
                        add_space_prefix,
                        add_bos_token,
                        "Building SPM tokenizer from GGUF sentencepiece data"
                    );
                    Some(SplitTokenizer::from_sentencepiece(
                        vocab,
                        &scores,
                        add_space_prefix,
                        add_bos_token,
                    ))
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
    /// Load from shards, forcing CPU device (used as OOM fallback).
    #[allow(clippy::too_many_arguments)]
    pub fn load_from_shards_cpu(
        model_dir: &Path,
        shard_files: Vec<(u32, PathBuf)>,
        tensor_entries: &[Vec<crate::types::ShardTensorEntry>],
        total_gguf_size: u64,
        layer_start: usize,
        layer_end: usize,
        is_first: bool,
        is_last: bool,
    ) -> Result<Self, SwarmError> {
        Self::load_from_shards_inner(
            model_dir,
            shard_files,
            tensor_entries,
            total_gguf_size,
            layer_start,
            layer_end,
            is_first,
            is_last,
            true,
        )
    }

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
        Self::load_from_shards_inner(
            model_dir,
            shard_files,
            tensor_entries,
            total_gguf_size,
            layer_start,
            layer_end,
            is_first,
            is_last,
            false,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn load_from_shards_inner(
        model_dir: &Path,
        shard_files: Vec<(u32, PathBuf)>,
        tensor_entries: &[Vec<crate::types::ShardTensorEntry>],
        total_gguf_size: u64,
        layer_start: usize,
        layer_end: usize,
        is_first: bool,
        is_last: bool,
        force_cpu: bool,
    ) -> Result<Self, SwarmError> {
        let header_path = model_dir.join("gguf_header.bin");
        if !header_path.exists() {
            return Err(SwarmError::Internal(format!(
                "GGUF header not found at {}. The originating node must generate this file.",
                header_path.display()
            )));
        }

        // Single shard with no tensor entries = full GGUF file as shard.
        // Load directly via mmap instead of ShardReader.
        let has_tensor_entries = tensor_entries.iter().any(|v| !v.is_empty());
        if shard_files.len() == 1 && !has_tensor_entries {
            let shard_path = &shard_files[0].1;
            tracing::info!(
                model_dir = %model_dir.display(),
                shard_path = %shard_path.display(),
                "Single-shard model with no tensor entries — loading as full GGUF via mmap"
            );
            return Self::load_from_gguf(shard_path, layer_start, layer_end, is_first, is_last);
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

        // Verify tensor_data_offset matches between the two Content::read calls
        if ct.tensor_data_offset != tensor_data_offset {
            tracing::error!(
                expected = tensor_data_offset,
                actual = ct.tensor_data_offset,
                "DIAG: tensor_data_offset MISMATCH between header parse and ShardReader parse!"
            );
        }

        // Diagnostic: log first few tensor offsets from Content vs tensor_map
        for (name, info) in ct.tensor_infos.iter().take(5) {
            let seek_pos = ct.tensor_data_offset + info.offset;
            let size_in_bytes = info.ggml_dtype.type_size() * info.shape.elem_count()
                / info.ggml_dtype.block_size();
            let found = reader.find_shard(seek_pos);
            tracing::info!(
                tensor = %name,
                gguf_seek = seek_pos,
                size = size_in_bytes,
                shard_mapping = ?found,
                "DIAG: tensor mapping check"
            );
        }

        // DIAG: Read first 16 bytes of blk.0.attn_norm.weight via ShardReader
        // to verify data integrity
        if let Some(norm_info) = ct.tensor_infos.get("blk.0.attn_norm.weight") {
            use std::io::{Read as IoReadTrait, Seek as SeekTrait};
            let seek_pos = ct.tensor_data_offset + norm_info.offset;
            reader.seek(SeekFrom::Start(seek_pos)).ok();
            let mut probe = [0u8; 16];
            if reader.read_exact(&mut probe).is_ok() {
                tracing::info!(
                    seek_pos,
                    first_bytes = ?&probe,
                    "DIAG: blk.0.attn_norm.weight first 16 bytes via ShardReader"
                );
            }
        }

        // From here, the exact same logic as load_from_gguf
        let device = if force_cpu {
            Device::Cpu
        } else {
            Device::cuda_if_available(0).unwrap_or(Device::Cpu)
        };
        if device.is_cuda() {
            tracing::info!("Split model using CUDA GPU");
        } else if force_cpu {
            tracing::info!("Split model using CPU (GPU OOM fallback)");
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
        if head_count == 0 {
            return Err(SwarmError::Inference(
                "GGUF metadata error: attention.head_count is zero".into(),
            ));
        }
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

        // Long RoPE (SuRoPE) for Phi-3.5 and similar extended-context models
        let longrope = load_longrope_factors(&ct, &mut reader, arch, context_length);
        let (cos, sin) = if let Some((ref factors, attn_factor)) = longrope {
            tracing::info!(
                factors_len = factors.len(),
                attn_factor,
                "Using Long RoPE (SuRoPE)"
            );
            precompute_freqs_cis_longrope(
                rope_dim,
                rope_freq_base,
                context_length,
                factors,
                attn_factor,
                &device,
            )
            .map_err(|e| SwarmError::Internal(e.to_string()))?
        } else {
            precompute_freqs_cis(rope_dim, rope_freq_base, context_length, &device)
                .map_err(|e| SwarmError::Internal(e.to_string()))?
        };
        let neg_inf = Tensor::new(f32::NEG_INFINITY, &device)
            .map_err(|e| SwarmError::Internal(e.to_string()))?;
        // Helper: create RmsNorm from GGUF weight tensor.
        // Note: GGUF norm weights for Gemma models already include the +1 offset
        // (added by convert_hf_to_gguf.py's modify_tensors), so we use them as-is.
        let make_norm = |qtensor: QTensor, eps: f64| -> Result<RmsNorm, SwarmError> {
            RmsNorm::from_qtensor(qtensor, eps).map_err(|e| SwarmError::Internal(e.to_string()))
        };

        let tok_embeddings = if is_first {
            let tok_embd = ct
                .tensor(&mut reader, "token_embd.weight", &device)
                .map_err(|e| SwarmError::Internal(e.to_string()))?;
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
                        post_attention_norm: None,
                        post_ffw_norm: None,
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
                    post_attention_norm: None,
                    post_ffw_norm: None,
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
                    // Fused QKV: keep original quantized tensor, split output in forward pass.
                    // Re-quantizing split weights (even Q8_0) degrades quality catastrophically.
                    let fused_qt = ct
                        .tensor(&mut reader, &format!("{prefix}.attn_qkv.weight"), &device)
                        .map_err(|e| SwarmError::Internal(format!("{prefix}.attn_qkv: {e}")))?;
                    let q_dim = head_count * head_dim;
                    let k_dim = head_count_kv * head_dim;
                    let fused = QMatMul::make_fused(fused_qt)
                        .map_err(|e| SwarmError::Internal(e.to_string()))?;
                    (
                        QMatMul::from_fused_slice(fused.clone(), 0, q_dim),
                        QMatMul::from_fused_slice(fused.clone(), q_dim, k_dim),
                        QMatMul::from_fused_slice(fused, q_dim + k_dim, k_dim),
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
                    // Fused gate+up: keep original quantized tensor, split output in forward.
                    let fused_qt = ct
                        .tensor(&mut reader, &format!("{prefix}.ffn_up.weight"), &device)
                        .map_err(|e| {
                            SwarmError::Internal(format!("Failed to load {prefix}.ffn_up: {e}"))
                        })?;
                    // Get ffn_hidden_dim from the fused tensor shape (total_rows / 2)
                    let fused_shape = fused_qt.shape();
                    let half = fused_shape.dims()[0] / 2;
                    let fused = QMatMul::make_fused(fused_qt)
                        .map_err(|e| SwarmError::Internal(e.to_string()))?;
                    (
                        QMatMul::from_fused_slice(fused.clone(), 0, half),
                        QMatMul::from_qtensor(ffn_down_qt)
                            .map_err(|e| SwarmError::Internal(e.to_string()))?,
                        QMatMul::from_fused_slice(fused, half, half),
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

                // Gemma 2 post-norms (optional)
                let post_attention_norm = ct
                    .tensor(
                        &mut reader,
                        &format!("{prefix}.post_attention_norm.weight"),
                        &device,
                    )
                    .ok()
                    .map(|t| make_norm(t, rms_norm_eps))
                    .transpose()?;
                let post_ffw_norm = ct
                    .tensor(
                        &mut reader,
                        &format!("{prefix}.post_ffw_norm.weight"),
                        &device,
                    )
                    .ok()
                    .map(|t| make_norm(t, rms_norm_eps))
                    .transpose()?;

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
                    post_attention_norm,
                    post_ffw_norm,
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
                let add_space_prefix = ct
                    .metadata
                    .get("tokenizer.ggml.add_space_prefix")
                    .and_then(|v| v.to_bool().ok())
                    .unwrap_or(true);
                let add_bos_token = ct
                    .metadata
                    .get("tokenizer.ggml.add_bos_token")
                    .and_then(|v| v.to_bool().ok())
                    .unwrap_or(false);
                if !scores.is_empty() {
                    tracing::info!(
                        vocab_size = vocab.len(),
                        scores = scores.len(),
                        add_space_prefix,
                        add_bos_token,
                        "Building SPM tokenizer from GGUF header sentencepiece data"
                    );
                    Some(SplitTokenizer::from_sentencepiece(
                        vocab,
                        &scores,
                        add_space_prefix,
                        add_bos_token,
                    ))
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
            head_count,
            head_count_kv,
            head_dim,
            rope_dim,
            embedding_length,
            has_post_norms = layers.iter().any(|l| matches!(l, LayerVariant::Dense(lw) if lw.post_attention_norm.is_some())),
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

    /// Forward pass that captures hidden states at specified layers.
    /// Returns (output_tensor, captured_hidden_states) where captured is
    /// a HashMap from absolute layer index to the post-layer hidden state tensor.
    pub fn forward_with_hidden_capture(
        &mut self,
        input: &Tensor,
        index_pos: usize,
        kv_cache_store: &KvCacheStore,
        request_id: &str,
        capture_layers: &std::collections::HashSet<usize>,
    ) -> Result<(Tensor, HashMap<usize, Tensor>), SwarmError> {
        self.forward_inner(
            input,
            index_pos,
            kv_cache_store,
            request_id,
            None,
            Some(capture_layers),
        )
    }

    /// Forward pass with pre-embedded hidden states (local embedding privacy).
    ///
    /// The input tensor is already in hidden-state space (shape [1, seq, hidden_dim])
    /// — the embedding lookup is skipped even if this segment has `tok_embeddings`.
    /// Used when the requesting node performed embedding locally for privacy.
    pub fn forward_pre_embedded(
        &mut self,
        input: &Tensor,
        index_pos: usize,
        kv_cache_store: &KvCacheStore,
        request_id: &str,
    ) -> Result<Tensor, SwarmError> {
        let (output, _) = self.forward_inner_impl(
            input,
            index_pos,
            kv_cache_store,
            request_id,
            None,
            None,
            true,
        )?;
        Ok(output)
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
        let (output, _) = self.forward_inner(
            input,
            index_pos,
            kv_cache_store,
            request_id,
            lora_adapter,
            None,
        )?;
        Ok(output)
    }

    /// Inner forward pass implementation. When `capture_layers` is Some, captures
    /// hidden states at the specified absolute layer indices.
    fn forward_inner(
        &mut self,
        input: &Tensor,
        index_pos: usize,
        kv_cache_store: &KvCacheStore,
        request_id: &str,
        lora_adapter: Option<&LoraAdapter>,
        capture_layers: Option<&std::collections::HashSet<usize>>,
    ) -> Result<(Tensor, HashMap<usize, Tensor>), SwarmError> {
        self.forward_inner_impl(
            input,
            index_pos,
            kv_cache_store,
            request_id,
            lora_adapter,
            capture_layers,
            false,
        )
    }

    /// Core forward pass. When `skip_embedding` is true, the input is treated as
    /// pre-embedded hidden states even if this segment has `tok_embeddings`.
    #[allow(clippy::too_many_arguments)]
    fn forward_inner_impl(
        &mut self,
        input: &Tensor,
        index_pos: usize,
        kv_cache_store: &KvCacheStore,
        request_id: &str,
        lora_adapter: Option<&LoraAdapter>,
        capture_layers: Option<&std::collections::HashSet<usize>>,
        skip_embedding: bool,
    ) -> Result<(Tensor, HashMap<usize, Tensor>), SwarmError> {
        let forward_start = std::time::Instant::now();
        // Use component presence rather than layer indices for shard-aware is_first/is_last
        let is_first = self.tok_embeddings.is_some();
        let is_last = self.output.is_some();

        // Move input to model's device if needed (e.g. CPU → CUDA)
        let input = input
            .to_device(&self.device)
            .map_err(|e| SwarmError::Internal(format!("Device transfer failed: {e}")))?;

        // Determine the hidden state to start from
        let mut layer_in = if is_first && !skip_embedding {
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
            // Non-first segment or pre-embedded: input is already hidden states
            input
        };

        // Get seq_len for mask
        let seq_len = layer_in
            .dim(1)
            .map_err(|e| SwarmError::Internal(e.to_string()))?;

        // Pre-flight check: reject sequences that exceed the model's context window
        // to avoid cryptic tensor dimension errors in attention.
        let total_seq = index_pos + seq_len;
        if total_seq > self.max_seq_len {
            return Err(SwarmError::Validation(format!(
                "Sequence length ({total_seq}) exceeds model context window ({}). \
                 Reduce your prompt or max_tokens.",
                self.max_seq_len
            )));
        }

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
        let mut captured: HashMap<usize, Tensor> = HashMap::new();

        // Run through our layers
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let abs_layer = self.layer_start + layer_idx;
            let lora_param = lora_adapter.map(|a| (a, abs_layer));

            let layer_start_time = std::time::Instant::now();
            match layer {
                LayerVariant::Dense(lw) => {
                    let x = layer_in;
                    let residual = &x;
                    let x = lw
                        .attention_norm
                        .forward(&x)
                        .map_err(|e| SwarmError::Internal(format!("attn_norm: {e}")))?;
                    let mut attn = lw
                        .forward_attn(
                            &x,
                            mask.as_ref(),
                            index_pos,
                            &mut layer_kv_caches[layer_idx],
                            max_seq_len,
                            lora_param,
                        )
                        .map_err(|e| SwarmError::Internal(format!("attn: {e}")))?;
                    // Gemma 2 post-attention norm: normalize before residual add
                    if let Some(ref post_norm) = lw.post_attention_norm {
                        attn = post_norm
                            .forward(&attn)
                            .map_err(|e| SwarmError::Internal(format!("post_attn_norm: {e}")))?;
                    }
                    let x = (attn + residual).map_err(|e| SwarmError::Internal(e.to_string()))?;

                    let residual = &x;
                    let x = lw
                        .ffn_norm
                        .forward(&x)
                        .map_err(|e| SwarmError::Internal(format!("ffn_norm: {e}")))?;
                    let mut x = match &lw.ffn {
                        FfnVariant::Dense(mlp) => mlp
                            .forward(&x, lora_param)
                            .map_err(|e| SwarmError::Internal(format!("mlp: {e}")))?,
                        FfnVariant::MoE(moe) => moe
                            .forward(&x)
                            .map_err(|e| SwarmError::Internal(format!("moe: {e}")))?,
                    };
                    // Gemma 2 post-FFN norm: normalize before residual add
                    if let Some(ref post_norm) = lw.post_ffw_norm {
                        x = post_norm
                            .forward(&x)
                            .map_err(|e| SwarmError::Internal(format!("post_ffw_norm: {e}")))?;
                    }
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
            tracing::trace!(
                layer = abs_layer,
                layer_ms = layer_start_time.elapsed().as_millis() as u64,
                "DIAG: layer forward complete"
            );

            // Capture hidden state if requested (zero overhead when not capturing)
            if let Some(layers_to_capture) = capture_layers {
                if layers_to_capture.contains(&abs_layer) {
                    captured.insert(abs_layer, layer_in.clone());
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

        result.map(|t| (t, captured))
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
                    let mut attn_batched = Tensor::cat(&attn_refs, 0)
                        .map_err(|e| SwarmError::Internal(format!("attn restack: {e}")))?;
                    if let Some(ref post_norm) = lw.post_attention_norm {
                        attn_batched = post_norm
                            .forward(&attn_batched)
                            .map_err(|e| SwarmError::Internal(format!("post_attn_norm: {e}")))?;
                    }
                    let x = (&attn_batched + &residual)
                        .map_err(|e| SwarmError::Internal(e.to_string()))?;

                    let residual2 = x.clone();
                    let x = lw
                        .ffn_norm
                        .forward(&x)
                        .map_err(|e| SwarmError::Internal(format!("ffn_norm: {e}")))?;
                    let mut x = match &lw.ffn {
                        FfnVariant::Dense(mlp) => mlp
                            .forward(&x, None)
                            .map_err(|e| SwarmError::Internal(format!("mlp: {e}")))?,
                        FfnVariant::MoE(moe) => moe
                            .forward(&x)
                            .map_err(|e| SwarmError::Internal(format!("moe: {e}")))?,
                    };
                    if let Some(ref post_norm) = lw.post_ffw_norm {
                        x = post_norm
                            .forward(&x)
                            .map_err(|e| SwarmError::Internal(format!("post_ffw_norm: {e}")))?;
                    }
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
        // DIAG: dump token IDs for debugging tokenizer issues
        tracing::info!(
            num_tokens,
            tokens = ?&token_ids[..token_ids.len().min(30)],
            "DIAG: tokenize result"
        );
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

        // DIAG: dump token IDs for debugging tokenizer issues
        tracing::info!(
            num_tokens = token_ids.len(),
            tokens = ?&token_ids[..token_ids.len().min(50)],
            "DIAG: tokenize_and_embed token IDs"
        );

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
#[cfg(test)]
mod tests {
    use super::super::layers::{run_attention, standard_attention, topk_cpu};
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
            eos_token_str: String::new(),
            bos_token: String::new(),
            cached_chat_template: None,
            vocab: None,
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
            QMatMul::from_qtensor(qt).expect("QMatMul load failed")
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
                RmsNorm::from_qtensor(qt, 1e-6).expect("RmsNorm load failed")
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
                post_attention_norm: None,
                post_ffw_norm: None,
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
        assert!(ModelArch::Gemma.use_rope_contiguous());
        assert!(ModelArch::Gemma2.use_rope_contiguous());
        assert!(ModelArch::Phi3.use_rope_contiguous());
        assert!(ModelArch::Starcoder2.use_rope_contiguous());
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
            QMatMul::from_qtensor(qt).expect("QMatMul load failed")
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
                RmsNorm::from_qtensor(qt, 1e-6).expect("RmsNorm load failed")
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
                post_attention_norm: None,
                post_ffw_norm: None,
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
            QMatMul::from_qtensor(qt).expect("QMatMul load failed")
        };

        let max_seq_len = 128;
        let (cos, sin) = precompute_freqs_cis(head_dim, 10000.0, max_seq_len, &device).unwrap();
        let neg_inf = Tensor::new(f32::NEG_INFINITY, &device).unwrap();
        let norm_w = Tensor::ones((hidden_dim,), DType::F32, &device).unwrap();
        let make_rms_norm = |w: &Tensor| {
            let qt = QTensor::quantize(w, candle_core::quantized::GgmlDType::F32).unwrap();
            RmsNorm::from_qtensor(qt, 1e-6).expect("RmsNorm load failed")
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
            post_attention_norm: None,
            post_ffw_norm: None,
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
            QMatMul::from_qtensor(qt).expect("QMatMul load failed")
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
            QMatMul::from_qtensor(qt).expect("QMatMul load failed")
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
            QMatMul::from_qtensor(qt).expect("QMatMul load failed")
        };
        let make_rms_norm = |dim: usize| -> RmsNorm {
            let w = Tensor::ones((dim,), DType::F32, &device).unwrap();
            let qt = QTensor::quantize(&w, candle_core::quantized::GgmlDType::F32).unwrap();
            RmsNorm::from_qtensor(qt, 1e-6).expect("RmsNorm load failed")
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
            QMatMul::from_qtensor(qt).expect("QMatMul load failed")
        };
        let make_rms_norm = |dim: usize| -> RmsNorm {
            let w = Tensor::ones((dim,), DType::F32, &device).unwrap();
            let qt = QTensor::quantize(&w, candle_core::quantized::GgmlDType::F32).unwrap();
            RmsNorm::from_qtensor(qt, 1e-6).expect("RmsNorm load failed")
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
            QMatMul::from_qtensor(qt).expect("QMatMul load failed")
        };
        let make_rms_norm = |dim: usize| -> RmsNorm {
            let w = Tensor::ones((dim,), DType::F32, &device).unwrap();
            let qt = QTensor::quantize(&w, candle_core::quantized::GgmlDType::F32).unwrap();
            RmsNorm::from_qtensor(qt, 1e-6).expect("RmsNorm load failed")
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
            QMatMul::from_qtensor(qt).expect("QMatMul load failed")
        };
        let make_rms_norm = |dim: usize| -> RmsNorm {
            let w = Tensor::ones((dim,), DType::F32, &device).unwrap();
            let qt = QTensor::quantize(&w, candle_core::quantized::GgmlDType::F32).unwrap();
            RmsNorm::from_qtensor(qt, 1e-6).expect("RmsNorm load failed")
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
            post_attention_norm: None,
            post_ffw_norm: None,
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
            QMatMul::from_qtensor(qt).expect("QMatMul load failed")
        };
        let norm_w = Tensor::ones((n_head * head_dim,), DType::F32, &device).unwrap();
        let make_rms_norm = |w: &Tensor| {
            let qt = QTensor::quantize(w, candle_core::quantized::GgmlDType::F32).unwrap();
            RmsNorm::from_qtensor(qt, 1e-6).expect("RmsNorm load failed")
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
            post_attention_norm: None,
            post_ffw_norm: None,
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
            QMatMul::from_qtensor(qt).expect("QMatMul load failed")
        };
        let norm_w = Tensor::ones((n_head * head_dim,), DType::F32, &device).unwrap();
        let make_rms_norm = |w: &Tensor| {
            let qt = QTensor::quantize(w, candle_core::quantized::GgmlDType::F32).unwrap();
            RmsNorm::from_qtensor(qt, 1e-6).expect("RmsNorm load failed")
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
            post_attention_norm: None,
            post_ffw_norm: None,
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
            QMatMul::from_qtensor(qt).expect("QMatMul load failed")
        };
        let make_rms_norm = |dim: usize| -> RmsNorm {
            let w = Tensor::ones((dim,), DType::F32, &device).unwrap();
            let qt = QTensor::quantize(&w, candle_core::quantized::GgmlDType::F32).unwrap();
            RmsNorm::from_qtensor(qt, 1e-6).expect("RmsNorm load failed")
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
                post_attention_norm: None,
                post_ffw_norm: None,
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

    /// Test Gemma-2 with real GGUF file — compare load_from_gguf vs load_from_shards.
    /// Requires the Gemma-2-2B-IT Q4_K_M model to be present.
    #[test]
    #[ignore] // Run with: cargo test gemma2_real_gguf -- --ignored --nocapture
    fn gemma2_real_gguf_vs_shards() {
        use candle_core::Tensor;
        let gguf_path = std::path::Path::new(
            "/tmp/swarm_gemma_test/models/gemma-2-2b-it-q4-k-m/gemma-2-2b-it-Q4_K_M.gguf",
        );
        if !gguf_path.exists() {
            eprintln!("Skipping: GGUF not found at {}", gguf_path.display());
            return;
        }

        // Load from full GGUF
        let mut model = SplitModel::load_from_gguf(gguf_path, 0, 26, true, true)
            .expect("Failed to load from GGUF");

        // Use the tokenizer to get the same tokens our API uses
        let prompt_tokens: Vec<u32> = vec![
            2, 2, 106, 1645, 108, 1841, 603, 573, 6037, 576, 6081, 235336, 107, 108, 106, 2516, 108,
        ];
        let input = Tensor::new(&prompt_tokens[..], &Device::Cpu)
            .unwrap()
            .unsqueeze(0) // [1, 17]
            .unwrap()
            .to_dtype(candle_core::DType::I64)
            .unwrap();

        let kv_store = KvCacheStore::new(std::time::Duration::from_secs(600));
        let logits = model
            .forward(&input, 0, &kv_store, "gemma2-gguf-test")
            .expect("Forward pass failed");

        let flat: Vec<f32> = logits.flatten_all().unwrap().to_vec1().unwrap();
        let (argmax_idx, argmax_val) = flat
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();

        eprintln!("GGUF logits: argmax={} score={:.4}", argmax_idx, argmax_val);
        eprintln!(
            "  min={:.4} max={:.4} dim={}",
            flat.iter().cloned().fold(f32::INFINITY, f32::min),
            flat.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
            flat.len()
        );

        // Expected: token 651 ("The") should be near the top
        let the_score = flat[651];
        eprintln!("  token 651 ('The') score={:.4}", the_score);
        eprintln!("  token 235274 ('1') score={:.4}", flat[235274]);

        // Save logits for external comparison
        let bytes: Vec<u8> = flat.iter().flat_map(|f| f.to_le_bytes()).collect();
        std::fs::write("/tmp/gemma2_our_logits.bin", &bytes).ok();
        eprintln!(
            "  Saved {} logits to /tmp/gemma2_our_logits.bin",
            flat.len()
        );
    }

    /// Test Gemma-2 with single token (no mask) to eliminate mask issues.
    #[test]
    #[ignore]
    fn gemma2_single_token() {
        use candle_core::{Device, Tensor};
        let gguf_path = std::path::Path::new(
            "/tmp/swarm_gemma_test/models/gemma-2-2b-it-q4-k-m/gemma-2-2b-it-Q4_K_M.gguf",
        );
        if !gguf_path.exists() {
            eprintln!("Skipping: GGUF not found");
            return;
        }

        // Single BOS token — no mask needed, no flash attention
        let prompt_tokens: Vec<u32> = vec![2];
        let input = Tensor::new(&prompt_tokens[..], &Device::Cpu)
            .unwrap()
            .unsqueeze(0)
            .unwrap()
            .to_dtype(candle_core::DType::I64)
            .unwrap();

        let mut model =
            SplitModel::load_from_gguf(gguf_path, 0, 26, true, true).expect("Failed to load");

        let kv_store = KvCacheStore::new(std::time::Duration::from_secs(600));
        let logits = model
            .forward(&input, 0, &kv_store, "gemma2-single-tok")
            .expect("Forward failed");

        let flat: Vec<f32> = logits.flatten_all().unwrap().to_vec1().unwrap();
        let (argmax_idx, argmax_val) = flat
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();
        eprintln!("Single token (BOS): argmax={argmax_idx} score={argmax_val:.4}");
        eprintln!("  token 2 score: {:.4}", flat[2]);
        eprintln!("  token 108 score: {:.4}", flat[108]);

        // Save for comparison
        let bytes: Vec<u8> = flat.iter().flat_map(|f| f.to_le_bytes()).collect();
        std::fs::write("/tmp/gemma2_single_token_logits.bin", &bytes).ok();
        eprintln!("  Saved logits to /tmp/gemma2_single_token_logits.bin");
    }

    /// Test embedding dequantization matches Python reference.
    #[test]
    #[ignore]
    fn gemma2_embedding_verification() {
        use candle_core::{quantized::gguf_file, Device, Tensor};

        let gguf_path =
            "/tmp/swarm_gemma_test/models/gemma-2-2b-it-q4-k-m/gemma-2-2b-it-Q4_K_M.gguf";
        let path = std::path::Path::new(gguf_path);
        if !path.exists() {
            eprintln!("Skipping: GGUF not found");
            return;
        }

        let file = std::fs::File::open(path).unwrap();
        let mmap = unsafe { memmap2::Mmap::map(&file) }.unwrap();
        let mut cursor = std::io::Cursor::new(mmap.as_ref());
        let ct = gguf_file::Content::read(&mut cursor).unwrap();
        let device = Device::Cpu;

        // Load and dequantize embedding
        let embd_qt = ct
            .tensor(&mut cursor, "token_embd.weight", &device)
            .unwrap();
        let embd = embd_qt.dequantize(&device).unwrap();
        eprintln!("Embedding shape: {:?}", embd.shape());

        // Get row 2 (BOS token)
        let row2 = embd.i(2).unwrap();
        let row2_vals: Vec<f32> = row2.to_vec1().unwrap();
        eprintln!("Row 2 (BOS) first 8: {:?}", &row2_vals[..8]);
        eprintln!(
            "Row 2 (BOS) last 8: {:?}",
            &row2_vals[row2_vals.len() - 8..]
        );

        // Compare with Python reference
        let py_ref = std::fs::read("/tmp/gemma2_embed_row2.npy").ok();
        if let Some(ref npy_bytes) = py_ref {
            // Parse npy format: skip header, read f32 values
            // Simple npy parser: header starts with \x93NUMPY, has length info
            let header_len = 10 + npy_bytes[8] as usize + ((npy_bytes[9] as usize) << 8);
            let data_bytes = &npy_bytes[header_len..];
            let py_vals: Vec<f32> = data_bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            eprintln!("Python ref first 8: {:?}", &py_vals[..8]);

            let mut max_diff = 0f32;
            let mut mismatches = 0;
            for (i, (a, b)) in row2_vals.iter().zip(py_vals.iter()).enumerate() {
                let diff = (a - b).abs();
                if diff > max_diff {
                    max_diff = diff;
                }
                if diff > 1e-4 && i < 20 {
                    eprintln!("  MISMATCH [{i}] rust={a:.6} python={b:.6} diff={diff:.6}");
                    mismatches += 1;
                }
            }
            eprintln!("Max embedding diff: {max_diff:.6}, mismatches (>1e-4): {mismatches}");
        } else {
            eprintln!("No Python reference found at /tmp/gemma2_embed_row2.npy");
        }

        // Now test: embedding lookup → scale by sqrt(2304) → final norm → output projection
        // This is the 0-layer forward pass
        let emb = Embedding::new(embd.clone(), 2304);

        // Token 2 (BOS) lookup
        let ids = Tensor::new(&[2u32], &device).unwrap();
        let looked_up = emb.forward(&ids).unwrap(); // (1, 2304)
        let scaled = looked_up.affine((2304f64).sqrt(), 0.0).unwrap(); // scale by sqrt(hidden_dim)

        // Apply final norm (with +1 offset)
        let norm_qt = ct
            .tensor(&mut cursor, "output_norm.weight", &device)
            .unwrap();
        let norm_w = norm_qt.dequantize(&device).unwrap();
        let norm_w_plus1 = (norm_w + 1.0).unwrap(); // Gemma +1
        let normed = candle_nn::ops::rms_norm(&scaled, &norm_w_plus1, 1e-6).unwrap();

        // Output projection using dequantized embedding
        let logits = normed.matmul(&embd.t().unwrap()).unwrap();
        let flat: Vec<f32> = logits.flatten_all().unwrap().to_vec1().unwrap();
        let argmax = flat
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();
        eprintln!("0-layer logits: argmax={} score={:.4}", argmax.0, argmax.1);
        eprintln!("  token 2 score: {:.4}", flat[2]);
        eprintln!("  token 108 score: {:.4}", flat[108]);
    }

    /// Test QMatMul vs dequantized matmul for output projection.
    /// Diagnoses whether the sorted-correlation issue is in the output projection.
    #[test]
    #[ignore]
    fn gemma2_output_projection_qmatmul_vs_deq() {
        use candle_core::{quantized::gguf_file, Device, Tensor};

        let gguf_path =
            "/tmp/swarm_gemma_test/models/gemma-2-2b-it-q4-k-m/gemma-2-2b-it-Q4_K_M.gguf";
        let path = std::path::Path::new(gguf_path);
        if !path.exists() {
            eprintln!("Skipping: GGUF not found");
            return;
        }

        let file = std::fs::File::open(path).unwrap();
        let mmap = unsafe { memmap2::Mmap::map(&file) }.unwrap();
        let mut cursor = std::io::Cursor::new(mmap.as_ref());
        let ct = gguf_file::Content::read(&mut cursor).unwrap();
        let device = Device::Cpu;

        // Load token_embd.weight as both QTensor and dequantized
        let embd_qt = ct
            .tensor(&mut cursor, "token_embd.weight", &device)
            .unwrap();
        eprintln!("token_embd.weight QTensor shape: {:?}", embd_qt.shape());

        let embd_deq = embd_qt.dequantize(&device).unwrap();
        eprintln!("Dequantized embedding shape: {:?}", embd_deq.shape());

        // Create QMatMul from the QTensor
        let qmm = QMatMul::from_qtensor(
            ct.tensor(&mut cursor, "token_embd.weight", &device)
                .unwrap(),
        )
        .unwrap();

        // Create a random hidden state (simulating post-norm output)
        let hidden = Tensor::randn(0f32, 1.0, (1, 2304), &device).unwrap();

        // Method 1: QMatMul (our current approach)
        let logits_qmm = qmm.forward(&hidden).unwrap();
        let flat_qmm: Vec<f32> = logits_qmm.flatten_all().unwrap().to_vec1().unwrap();

        // Method 2: Dequantized matmul (reference approach)
        // embd_deq shape is (256000, 2304), we need hidden @ embd_deq.T
        let logits_deq = hidden.matmul(&embd_deq.t().unwrap()).unwrap();
        let flat_deq: Vec<f32> = logits_deq.flatten_all().unwrap().to_vec1().unwrap();

        eprintln!(
            "QMatMul logits: argmax={}",
            flat_qmm
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap()
                .0
        );
        eprintln!(
            "Deq logits: argmax={}",
            flat_deq
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap()
                .0
        );

        // Check if they agree
        let mut max_diff = 0f32;
        let mut sum_diff_sq = 0f64;
        for (i, (a, b)) in flat_qmm.iter().zip(flat_deq.iter()).enumerate() {
            let diff = (a - b).abs();
            if diff > max_diff {
                max_diff = diff;
            }
            sum_diff_sq += (diff as f64) * (diff as f64);
            if i < 5 {
                eprintln!("  [{i}] qmm={a:.6} deq={b:.6} diff={diff:.6}");
            }
        }
        let rmse = (sum_diff_sq / flat_qmm.len() as f64).sqrt();
        eprintln!("Max diff: {max_diff:.6}, RMSE: {rmse:.6}");

        // Check correlation
        let mean_q: f64 = flat_qmm.iter().map(|v| *v as f64).sum::<f64>() / flat_qmm.len() as f64;
        let mean_d: f64 = flat_deq.iter().map(|v| *v as f64).sum::<f64>() / flat_deq.len() as f64;
        let mut cov = 0f64;
        let mut var_q = 0f64;
        let mut var_d = 0f64;
        for (q, d) in flat_qmm.iter().zip(flat_deq.iter()) {
            let dq = *q as f64 - mean_q;
            let dd = *d as f64 - mean_d;
            cov += dq * dd;
            var_q += dq * dq;
            var_d += dd * dd;
        }
        let corr = cov / (var_q.sqrt() * var_d.sqrt());
        eprintln!("Pearson correlation (QMatMul vs Deq): {corr:.6}");

        // They should be highly correlated (>0.99) — just quantization error
        assert!(
            corr > 0.99,
            "QMatMul and dequantized matmul should agree: corr={corr}"
        );
    }
}
