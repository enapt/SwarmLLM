// ── Split model entry with LRU tracking ──

use candle_core::Tensor;

use super::model::SplitModel;

/// Fallback EOS token IDs covering major architectures when GGUF metadata is unavailable.
/// 2=LLaMA `</s>`, 107=Gemma `<end_of_turn>`, 32000=Mistral/Mixtral `</s>`.
const FALLBACK_EOS_TOKEN_IDS: &[u32] = &[2, 107, 32000];

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
        crate::types::unix_now_secs()
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
        let tok = super::gguf_meta::GgufTokenizerMeta::from_gguf_file(header_path).ok();

        let (vocab, bos_token, eos_token_str, eos_tokens, chat_template) = if let Some(t) = tok {
            let bos = t.bos_string();
            let eos_str = t.eos_string();
            let eos_ids = if t.eos_token_ids.is_empty() {
                FALLBACK_EOS_TOKEN_IDS.to_vec()
            } else {
                t.eos_token_ids
            };
            (t.vocab, bos, eos_str, eos_ids, t.chat_template)
        } else {
            (
                vec![],
                String::new(),
                String::new(),
                FALLBACK_EOS_TOKEN_IDS.to_vec(),
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
        self.last_used
            .store(Self::now_secs(), std::sync::atomic::Ordering::Relaxed);
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

/// Megabytes of the budgeted memory the resident split-model entries have
/// been REGISTERED against.
///
/// `holds_budgeted_memory` decides which of them count: `split_models` holds
/// segments on both devices, and counting a processor-resident one against the
/// graphics card over-states what the card is carrying.
///
/// **This is a registration figure, not a residency figure.** What is actually
/// on the card is `ModelProcessPool::vram_committed_mb`, charged at spawn — see
/// [`trim_split_model_cache`] for why the two must not be confused.
pub fn split_models_committed_mb(
    split_models: &dashmap::DashMap<SplitModelKey, SplitModelEntry>,
    holds_budgeted_memory: &dyn Fn(&crate::types::ModelId) -> bool,
) -> u64 {
    split_models
        .iter()
        .filter(|e| holds_budgeted_memory(&e.key().0))
        .map(|e| e.value().estimated_vram_mb)
        .sum()
}

/// Bound the split-model METADATA cache to `max_entries`, least-recently-used
/// first. Entries whose model has an active pipeline are kept.
///
/// Returns the keys removed so the caller can synchronise the secondary
/// `split_model_index` (without this, the index Vec accumulates stale
/// `(layer_start, layer_end)` tuples indefinitely — readers compensate with a
/// `split_models.contains_key` check per range, but every check pays for the
/// stale entries until daemon restart).
///
/// **This replaced a VRAM-budget eviction, and the difference is the whole
/// point.** A `SplitModelEntry` is metadata read out of `gguf_header.bin`; it
/// occupies no device memory, and the weights it describes belong to a worker
/// subprocess that `ModelProcessPool` admits, charges and reclaims. Enforcing a
/// graphics budget here made a SECOND accountant for one card — with a
/// different estimate (weights only, against the pool's weights + KV), a
/// different in-flight oracle (`active_pipelines`, which per gotcha #194 cannot
/// see peer-served work or the split fast path), and no idle floor at all. It
/// evicted a model that had answered a request three seconds earlier;
/// `ModelProcessPool::free_vram_for_admission` refuses to touch one used inside
/// the last minute. Ollama's scheduler is the same shape and keeps one owner:
/// a single centralised free-space tracker, victims chosen by refCount then
/// keep-alive then last-used.
///
/// **Do not restore the unload that used to hang off this.** It was added
/// (2026-07-21) because evicting an entry was *supposed* to free graphics
/// memory and did not, so the budget was enforced against a phantom. That
/// premise is gone: this no longer claims to free anything, and the pool frees
/// on its own schedule (`free_vram_for_admission` on demand,
/// `try_idle_vram_unload` on the timer, which runs outside the auto-manage
/// gate). Trimming an entry that is still wanted now costs a re-read of the
/// header — `ensure_split_model_entry` rebuilds it — rather than a killed
/// worker.
pub fn trim_split_model_cache(
    split_models: &dashmap::DashMap<SplitModelKey, SplitModelEntry>,
    active_pipelines: &dashmap::DashMap<uuid::Uuid, crate::types::PipelineAssignment>,
    max_entries: usize,
) -> Vec<SplitModelKey> {
    if split_models.len() <= max_entries {
        return Vec::new();
    }

    let active_model_ids: std::collections::HashSet<crate::types::ModelId> = {
        let mut ids = std::collections::HashSet::new();
        for entry in active_pipelines.iter() {
            for seg in &entry.value().segments {
                ids.insert(seg.shard_id.model_id.clone());
            }
        }
        ids
    };

    let mut candidates: Vec<(SplitModelKey, u64)> = split_models
        .iter()
        .filter(|e| !active_model_ids.contains(&e.key().0))
        .map(|e| (e.key().clone(), e.value().last_used_secs()))
        .collect();
    candidates.sort_by_key(|(_key, last_used)| *last_used);

    let mut removed = Vec::new();
    for (key, _last_used) in candidates {
        if split_models.len() <= max_entries {
            break;
        }
        if split_models.remove(&key).is_some() {
            removed.push(key);
        }
    }
    removed
}
