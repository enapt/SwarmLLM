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
    /// Each `KvCache` holds a buffer sized in `KV_CACHE_GROWTH_TOKENS` quanta
    /// and appends into it in place, concatenating only when a conversation
    /// outgrows the current quantum.
    pub(crate) layers: Vec<Option<KvCache>>,
    /// Per-layer SSM state for Qwen 3.5 hybrid models (delta net recurrent state + conv state).
    /// None for non-SSM layers. Only populated for Qwen35Ssm layer variants.
    pub(crate) ssm_states: Vec<Option<SsmState>>,
    /// When this entry was last accessed.
    pub(crate) last_accessed: std::time::Instant,
}

/// What a [`KvCacheStore`] is currently holding.
///
/// `allocated_bytes` is the figure that matters, and it is NOT `token_count`
/// times a per-token cost. candle's `Cache::append` allocates its whole
/// capacity on the FIRST append and grows in whole increments after that, so
/// the reservation moves in steps of `KV_CACHE_GROWTH_TOKENS` and a
/// twenty-token conversation costs the same as a five-hundred-token one.
/// Anything reasoning about KV memory from token counts alone is reasoning
/// about a quantity the allocator does not use — which is how a careful,
/// quantified estimate of this cache came to be wrong by 7x (gotcha #261).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct KvOccupancy {
    /// Number of live cache entries — one per (model, request).
    pub entries: usize,
    /// Bytes actually reserved by the K/V tensors, including unused headroom.
    pub allocated_bytes: u64,
    /// Bytes covering positions that have really been written.
    pub used_bytes: u64,
    /// Sequence positions written across every entry and layer.
    pub tokens: usize,
}

impl KvOccupancy {
    /// Fraction of the reservation that holds real tokens, 0.0 when nothing is
    /// allocated. A persistently low value means the growth quantum is too
    /// coarse for the conversations this node actually serves.
    pub fn utilisation(&self) -> f64 {
        if self.allocated_bytes == 0 {
            return 0.0;
        }
        self.used_bytes as f64 / self.allocated_bytes as f64
    }
}

impl KvCacheEntry {
    /// Bytes this entry reserves, and how many of them hold real tokens.
    ///
    /// Reads the ALLOCATED buffer (`Cache::all_data`), not the used window
    /// (`Cache::current_data`), because the allocated buffer is what the
    /// process is holding — see [`KvOccupancy`].
    pub(crate) fn occupancy(&self) -> (u64, u64, usize) {
        let mut allocated = 0u64;
        let mut used = 0u64;
        let mut tokens = 0usize;
        for kv in self.layers.iter().flatten() {
            for cache in [kv.k_cache(), kv.v_cache()] {
                let Some(data) = cache.all_data().as_ref() else {
                    continue;
                };
                let elem = data.dtype().size_in_bytes() as u64;
                let total = data.elem_count() as u64 * elem;
                allocated += total;
                let cap = cache.max_seq_len().max(1);
                let live = cache.current_seq_len();
                used += total / cap as u64 * live as u64;
            }
            tokens += kv.current_seq_len();
        }
        // SSM state is per-layer and fixed-size, so it does not scale with the
        // conversation, but it is real memory and belongs in the total.
        for ssm in self.ssm_states.iter().flatten() {
            for t in [&ssm.recurrent_state, &ssm.conv_state] {
                let bytes = t.elem_count() as u64 * t.dtype().size_in_bytes() as u64;
                allocated += bytes;
                used += bytes;
            }
        }
        (allocated, used, tokens)
    }

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

    /// What this store is holding right now.
    ///
    /// Walks every entry, so call it on a housekeeping tick rather than per
    /// token. It exists because KV memory was previously only observable as
    /// process RSS, which is confounded by the allocator: memory freed to
    /// `malloc` need not return to the OS, so a flat RSS reading cannot
    /// distinguish "nothing was evicted" from "everything was evicted and the
    /// arena was kept". Two predictions about this cache made from RSS alone
    /// were wrong (see `docs/FUTURE_WORK.md`).
    pub fn occupancy(&self) -> KvOccupancy {
        let mut out = KvOccupancy::default();
        for entry in self.caches.iter() {
            let (allocated, used, tokens) = entry.value().occupancy();
            out.entries += 1;
            out.allocated_bytes += allocated;
            out.used_bytes += used;
            out.tokens += tokens;
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::layers::{kv_cache_reservation, new_kv_cache, KV_CACHE_GROWTH_TOKENS};
    use candle_core::{DType, Device, Tensor};

    /// Fill one layer of a request's cache with `n` positions, the way a
    /// forward pass does.
    fn append(store: &KvCacheStore, n: usize, max_seq_len: usize) {
        let mut entry = store.get_or_create("m", "r", 1);
        let k = Tensor::zeros((1usize, 2, n, 4), DType::F32, &Device::Cpu).unwrap();
        let mut kv = new_kv_cache(max_seq_len);
        kv.append(&k, &k.clone()).unwrap();
        entry.layers[0] = Some(kv);
    }

    /// The whole point of the growth quantum: a short conversation must not
    /// reserve a long one's memory.
    ///
    /// Fails if `new_kv_cache` goes back to passing `max_seq_len` through —
    /// candle allocates that many positions on the first append, so a 4-token
    /// request would hold a 4096-token buffer.
    #[test]
    fn a_short_conversation_does_not_reserve_the_whole_context() {
        let store = KvCacheStore::new(std::time::Duration::from_secs(60));
        append(&store, 4, 4096);
        let occ = store.occupancy();
        assert_eq!(occ.entries, 1);
        assert_eq!(occ.tokens, 4);
        // 2 caches (K and V) x quantum positions x 2 heads x 4 dim x 4 bytes.
        let expected = 2 * KV_CACHE_GROWTH_TOKENS as u64 * 2 * 4 * 4;
        assert_eq!(
            occ.allocated_bytes, expected,
            "a 4-token request reserved {} bytes; the growth quantum is {} positions",
            occ.allocated_bytes, KV_CACHE_GROWTH_TOKENS
        );
    }

    /// Occupancy has to count the ALLOCATED buffer, not the used window —
    /// reporting the used window would show 100% utilisation forever and hide
    /// exactly the over-reservation this instrumentation exists to expose.
    #[test]
    fn occupancy_separates_reserved_bytes_from_used_bytes() {
        let store = KvCacheStore::new(std::time::Duration::from_secs(60));
        append(&store, 4, 4096);
        let occ = store.occupancy();
        assert!(
            occ.used_bytes < occ.allocated_bytes,
            "used {} vs allocated {} — occupancy is reading the wrong buffer",
            occ.used_bytes,
            occ.allocated_bytes
        );
        // 4 of `KV_CACHE_GROWTH_TOKENS` positions are live.
        let expect_ratio = 4.0 / KV_CACHE_GROWTH_TOKENS as f64;
        assert!(
            (occ.utilisation() - expect_ratio).abs() < 1e-6,
            "utilisation {} vs expected {expect_ratio}",
            occ.utilisation()
        );
    }

    /// A conversation past one quantum grows rather than failing or
    /// over-reserving, and the reservation stays proportional to its length.
    #[test]
    fn a_long_conversation_grows_past_one_quantum() {
        let store = KvCacheStore::new(std::time::Duration::from_secs(60));
        let n = KV_CACHE_GROWTH_TOKENS + 10;
        append(&store, n, 4096);
        let occ = store.occupancy();
        assert_eq!(occ.tokens, n);
        let per_pos = 2u64 * 2 * 4 * 4;
        assert_eq!(
            occ.allocated_bytes,
            2 * KV_CACHE_GROWTH_TOKENS as u64 * per_pos
        );
        assert!(occ.utilisation() > 0.5);
    }

    /// An empty store reports nothing rather than dividing by zero.
    #[test]
    fn an_empty_store_reports_zero_and_does_not_divide_by_zero() {
        let store = KvCacheStore::new(std::time::Duration::from_secs(60));
        let occ = store.occupancy();
        assert_eq!(occ, KvOccupancy::default());
        assert_eq!(occ.utilisation(), 0.0);
    }

    /// Hydrating a prefix snapshot must reserve from its token count, so a
    /// snapshot minted by a peer that recorded a whole-context `max_seq_len`
    /// cannot re-inflate the reservation.
    #[test]
    fn hydration_reservation_rounds_up_to_a_whole_quantum() {
        assert_eq!(kv_cache_reservation(0), KV_CACHE_GROWTH_TOKENS);
        assert_eq!(kv_cache_reservation(1), KV_CACHE_GROWTH_TOKENS);
        assert_eq!(
            kv_cache_reservation(KV_CACHE_GROWTH_TOKENS),
            KV_CACHE_GROWTH_TOKENS
        );
        assert_eq!(
            kv_cache_reservation(KV_CACHE_GROWTH_TOKENS + 1),
            2 * KV_CACHE_GROWTH_TOKENS
        );
    }
}
