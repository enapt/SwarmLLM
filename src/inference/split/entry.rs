// ── Split model entry with LRU tracking ──

use candle_core::Tensor;

use super::model::SplitModel;

/// Key type for split_models DashMap: (model_id, layer_start, layer_end).
pub type SplitModelKey = (crate::types::ModelId, usize, usize);

/// Metadata entry for a loaded model segment.
///
/// GPU memory lives in a worker subprocess managed by `ModelProcessPool`.
/// This struct holds only lightweight metadata for routing and token decoding.
pub struct SplitModelEntry {
    pub last_used: std::sync::atomic::AtomicU64,
    /// Estimated VRAM usage in MB for this model segment.
    pub estimated_vram_mb: u64,
    /// True if this entry has both embedding (first) and output head (last) — i.e., all layers.
    pub is_complete: bool,
    /// Cached EOS token IDs for lock-free sampling.
    pub eos_tokens: Vec<u32>,
    /// EOS token string (e.g., "<|endoftext|>").
    pub eos_token_str: String,
    /// BOS token string (e.g., "<s>").
    pub bos_token: String,
    /// Chat template from GGUF metadata (Jinja2 format).
    pub cached_chat_template: Option<String>,
    /// Full vocabulary for lock-free token decoding.
    pub vocab: Option<Vec<String>>,
    /// Layer range this entry covers.
    pub layer_start: usize,
    pub layer_end: usize,
}

impl SplitModelEntry {
    fn now_secs() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
    }

    /// Extract metadata from a `SplitModel` reference, then drop the model.
    /// GPU memory will live in the worker subprocess instead.
    pub fn new(model: SplitModel, layer_start: usize, layer_end: usize) -> Self {
        let estimated_vram_mb = model.estimate_vram_mb();
        let is_complete = model.is_first() && model.is_last();
        let eos_tokens = model.eos_tokens().to_vec();
        let eos_token_str = model.eos_token_str().to_string();
        let bos_token = model.bos_token().to_string();
        let cached_chat_template = model.chat_template().map(|s| s.to_string());
        let vocab = model.vocab().cloned();
        // model is dropped here — its memory will be in the subprocess
        drop(model);
        Self {
            last_used: std::sync::atomic::AtomicU64::new(Self::now_secs()),
            estimated_vram_mb,
            is_complete,
            eos_tokens,
            eos_token_str,
            bos_token,
            cached_chat_template,
            vocab,
            layer_start,
            layer_end,
        }
    }

    /// Build a metadata entry from a GGUF header file on disk, without loading model weights.
    /// Used when routing inference to worker subprocesses.
    pub fn from_header(
        header_path: &std::path::Path,
        layer_start: usize,
        layer_end: usize,
        is_first: bool,
        is_last: bool,
        vram_estimate_mb: u64,
    ) -> Self {
        use candle_core::quantized::gguf_file;

        let (vocab, bos_token, eos_token_str, eos_tokens, chat_template) =
            if let Ok(bytes) = std::fs::read(header_path) {
                let mut cursor = std::io::Cursor::new(&bytes);
                if let Ok(ct) = gguf_file::Content::read(&mut cursor) {
                    let vocab: Vec<String> = ct
                        .metadata
                        .get("tokenizer.ggml.tokens")
                        .and_then(|v| v.to_vec().ok())
                        .map(|arr| {
                            arr.iter()
                                .filter_map(|v| v.to_string().ok().cloned())
                                .collect()
                        })
                        .unwrap_or_default();
                    let eos_id = ct
                        .metadata
                        .get("tokenizer.ggml.eos_token_id")
                        .and_then(|v| v.to_u32().ok())
                        .unwrap_or(2);
                    let bos_id = ct
                        .metadata
                        .get("tokenizer.ggml.bos_token_id")
                        .and_then(|v| v.to_u32().ok())
                        .unwrap_or(1);
                    let eos_tokens_extra: Vec<u32> = ct
                        .metadata
                        .get("tokenizer.ggml.eos_token_ids")
                        .and_then(|v| v.to_vec().ok())
                        .map(|arr| arr.iter().filter_map(|v| v.to_u32().ok()).collect())
                        .unwrap_or_default();
                    let mut eos_tok = vec![eos_id];
                    for t in eos_tokens_extra {
                        if t != eos_id {
                            eos_tok.push(t);
                        }
                    }
                    let bos = vocab.get(bos_id as usize).cloned().unwrap_or_default();
                    let eos_str = vocab.get(eos_id as usize).cloned().unwrap_or_default();
                    let tmpl = ct
                        .metadata
                        .get("tokenizer.chat_template")
                        .and_then(|v| v.to_string().ok().cloned());
                    (vocab, bos, eos_str, eos_tok, tmpl)
                } else {
                    // Fallback: include common EOS tokens across architectures
                    // (2=LLaMA </s>, 107=Gemma, 32000=Mistral/Mixtral)
                    (
                        vec![],
                        String::new(),
                        String::new(),
                        vec![2, 107, 32000],
                        None,
                    )
                }
            } else {
                // Fallback: include common EOS tokens across architectures
                // (2=LLaMA </s>, 107=Gemma, 32000=Mistral/Mixtral)
                (
                    vec![],
                    String::new(),
                    String::new(),
                    vec![2, 107, 32000],
                    None,
                )
            };

        Self {
            last_used: std::sync::atomic::AtomicU64::new(Self::now_secs()),
            estimated_vram_mb: vram_estimate_mb,
            is_complete: is_first && is_last,
            eos_tokens,
            eos_token_str,
            bos_token,
            cached_chat_template: chat_template,
            vocab: if vocab.is_empty() { None } else { Some(vocab) },
            layer_start,
            layer_end,
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

/// A single item in a batched forward pass (used internally by SplitModel::forward_batch).
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

// BatchForwarder removed — batching is handled inside worker subprocesses now.
