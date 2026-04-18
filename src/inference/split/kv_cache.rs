// ── Per-request KV-cache store ──

use candle_nn::kv_cache::KvCache;

use super::SsmState;

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

impl KvCacheEntry {
    /// Truncate every layer's KV cache to exactly `target_len` sequence
    /// positions. No-op for layers whose current length is already ≤ target.
    /// Used by speculative decoding after partial acceptance — the remote
    /// forward wrote γ new KV entries, but only k ≤ γ are accepted, so the
    /// coordinator asks us to discard the trailing γ-k stale slots before
    /// the next round.
    ///
    /// Implementation: candle's `KvCache` exposes only `reset()` (to zero)
    /// and `append()` (to grow). To preserve the first `target_len`
    /// positions we snapshot them via narrow+contiguous, reset the cache,
    /// and re-append the snapshot. Cost is O(target_len * hidden) per layer
    /// per truncation.
    pub(crate) fn truncate_to(
        &mut self,
        target_len: usize,
    ) -> Result<(), crate::error::SwarmError> {
        use crate::error::SwarmError;
        for cache_opt in self.layers.iter_mut() {
            let Some(kv) = cache_opt else { continue };
            if kv.current_seq_len() <= target_len {
                continue;
            }
            let dim = kv.k_cache().dim();
            // Snapshot the prefix we want to keep.
            let (k_snap, v_snap) = {
                let k = kv
                    .k()
                    .map_err(|e| SwarmError::Internal(format!("kv k snapshot: {e}")))?;
                let v = kv
                    .v()
                    .map_err(|e| SwarmError::Internal(format!("kv v snapshot: {e}")))?;
                let k = k.ok_or_else(|| SwarmError::Internal("kv cache: k empty".into()))?;
                let v = v.ok_or_else(|| SwarmError::Internal("kv cache: v empty".into()))?;
                // Narrow to [0..target_len] on the sequence dim, then force a copy
                // so the subsequent reset+append doesn't alias the same buffer.
                let k_trunc = k
                    .narrow(dim, 0, target_len)
                    .and_then(|t| t.contiguous())
                    .map_err(|e| SwarmError::Internal(format!("kv narrow/contiguous k: {e}")))?;
                let v_trunc = v
                    .narrow(dim, 0, target_len)
                    .and_then(|t| t.contiguous())
                    .map_err(|e| SwarmError::Internal(format!("kv narrow/contiguous v: {e}")))?;
                (k_trunc, v_trunc)
            };
            kv.reset();
            // Re-append the snapshot. append() writes at current_seq_len (=0
            // after reset) and advances by target_len — exactly what we want.
            kv.append(&k_snap, &v_snap)
                .map_err(|e| SwarmError::Internal(format!("kv truncate re-append: {e}")))?;
        }
        Ok(())
    }
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

    #[cfg(test)]
    pub(crate) fn get_or_create(
        &self,
        model_key: &str,
        request_id: &str,
        num_layers: usize,
    ) -> dashmap::mapref::one::RefMut<'_, String, KvCacheEntry> {
        let key = Self::cache_key(model_key, request_id);
        self.get_or_create_keyed(&key, num_layers)
    }

    /// Look up the entry for a pre-formatted key. Returns an immutable
    /// reference (DashMap `Ref`); drop it before taking any mutable lock on
    /// the same key.
    pub(crate) fn get_entry<'a>(
        &'a self,
        key: &str,
    ) -> Option<dashmap::mapref::one::Ref<'a, String, KvCacheEntry>> {
        self.caches.get(key)
    }

    /// Truncate a request's KV cache (all layers) to `target_len`. No-op if
    /// no entry is present. Used by the speculative-decoding partial-accept
    /// fixup on the segment holder.
    pub fn truncate_request_to(
        &self,
        model_key: &str,
        request_id: &str,
        target_len: usize,
    ) -> Result<(), crate::error::SwarmError> {
        let key = Self::cache_key(model_key, request_id);
        if let Some(mut entry) = self.caches.get_mut(key.as_str()) {
            entry.truncate_to(target_len)?;
        }
        Ok(())
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
    #[cfg(test)]
    pub fn active_entries(&self) -> usize {
        self.caches.len()
    }
}
