// ── Per-request KV-cache store ──

use candle_core::{DType, Tensor};
use candle_nn::kv_cache::KvCache;

use super::SsmState;

/// One layer's KV cache: the f32 BHSD cache every path reads, plus an optional
/// f16 BSHD mirror kept for the CUDA flash-attention kernel.
///
/// **Why the mirror exists.** `run_attention`'s CUDA arm needs f16 in BSHD while
/// the cache is f32 in BHSD, so it used to transpose and convert the WHOLE
/// history on every token — O(history) work to add ONE position, and the
/// dominant cost of long-context decode. The mirror is appended one position at
/// a time instead, so the per-token cost stops growing with the conversation.
/// Measured on an RTX 3070, attention arm vs the mirrored ceiling:
/// 272 KV 1.6x, 528 KV 1.75x, 912 KV 1.86x.
///
/// **Why it is a mirror and not a replacement.** Rounding f32 to f16 is done at
/// WRITE time here instead of at read time, and since the f32 source is never
/// itself overwritten by a rounded value, the kernel receives bitwise the same
/// numbers it received before — the flash path is numerically unchanged, not
/// merely close. `standard_attention` still reads the f32 cache, so MHA decode,
/// prefill-with-prefix and forced-standard spec/SWIFT sessions keep full
/// precision. Dropping the f32 copy would change those, and published results on
/// f16 KV divergence (arXiv 2604.15409) say the accumulation is worst under
/// exactly our conditions — long context and GQA — so the f32 copy stays.
///
/// **Keeping the two in step is the whole risk**, because a mirror that has
/// drifted produces plausible wrong attention rather than an error. Everything
/// that mutates the cache therefore goes through `append` / `reset` here, which
/// are inherent methods and so take priority over the `Deref` to `KvCache` —
/// existing call sites cannot reach the inner cache's versions by accident.
/// `KvCacheStore::truncate_to` gets correct behaviour for free from that, since
/// it truncates by `reset()` + `append()` of the kept prefix.
pub(crate) struct LayerKv {
    /// f32 BHSD, sequence on dim 2. The source of truth.
    main: KvCache,
    /// f16 BSHD, sequence on dim 1. `None` on CPU and until the first append,
    /// which is where the device becomes known.
    shadow: Option<KvCache>,
    /// Reservation to build the mirror with, mirroring `main`'s growth quantum.
    growth: usize,
    /// False when `main`'s sequence axis is not dim 2, which the mirror's
    /// transpose assumes. Such a cache never gets a mirror.
    mirrorable: bool,
}

impl LayerKv {
    pub(crate) fn new(growth: usize) -> Self {
        Self::with_dim(2, growth)
    }

    /// Build with an explicit sequence dim, for hydrating a prefix-cache
    /// snapshot that recorded its own.
    ///
    /// A mirror is only maintained when the sequence axis is dim 2 (BHSD), since
    /// `to_bshd_f16`'s `transpose(1, 2)` is meaningless otherwise. Any other
    /// layout simply gets no mirror and the original conversion path.
    pub(crate) fn with_dim(dim: usize, growth: usize) -> Self {
        Self {
            main: KvCache::new(dim, growth),
            shadow: None,
            growth,
            mirrorable: dim == 2,
        }
    }

    /// Whether a mirror is worth maintaining for tensors on this device.
    ///
    /// CUDA only, and only when the flash kernel is actually compiled in —
    /// otherwise it is memory and conversion work for a consumer that does not
    /// exist.
    fn wants_shadow(_k: &Tensor) -> bool {
        #[cfg(feature = "flash-attn")]
        {
            _k.device().is_cuda() && !mirror_disabled()
        }
        #[cfg(not(feature = "flash-attn"))]
        {
            false
        }
    }

    /// BHSD f32 -> BSHD f16, for the new positions only.
    fn to_bshd_f16(t: &Tensor) -> candle_core::Result<Tensor> {
        t.transpose(1, 2)?.contiguous()?.to_dtype(DType::F16)
    }

    /// Append new positions, updating the mirror in the same call.
    ///
    /// Returns the full f32 BHSD (K, V) exactly as `KvCache::append` does, so
    /// callers are unchanged.
    pub(crate) fn append(
        &mut self,
        k: &Tensor,
        v: &Tensor,
    ) -> candle_core::Result<(Tensor, Tensor)> {
        let out = self.main.append(k, v)?;
        if self.shadow.is_none() && self.mirrorable && Self::wants_shadow(k) {
            self.shadow = Some(KvCache::new(1, self.growth));
        }
        if let Some(shadow) = self.shadow.as_mut() {
            // Convert only what is being added — this is the point of the mirror.
            let k_new = Self::to_bshd_f16(k)?;
            let v_new = Self::to_bshd_f16(v)?;
            if let Err(e) = shadow.append(&k_new, &v_new) {
                // Never serve from a mirror that failed to take an append: drop
                // it and let the flash path fall back to converting the f32
                // cache. Wrong attention is far worse than losing the speedup.
                tracing::warn!("KV mirror append failed, falling back to conversion: {e}");
                self.shadow = None;
            }
        }
        Ok(out)
    }

    /// Reset both representations together.
    pub(crate) fn reset(&mut self) {
        self.main.reset();
        if let Some(shadow) = self.shadow.as_mut() {
            shadow.reset();
        }
    }

    /// Opt this cache out of mirroring — for a model whose decode path reads the
    /// f32 cache anyway. See `model_wants_kv_mirror`.
    pub(crate) fn set_mirror_wanted(&mut self, wanted: bool) {
        if !wanted {
            self.mirrorable = false;
            self.shadow = None;
        }
    }

    /// Build the mirror unconditionally, so its bookkeeping can be tested on a
    /// machine with no GPU. Production creation still goes through the device
    /// check in `append`.
    #[cfg(test)]
    pub(crate) fn force_shadow_for_test(&mut self) {
        if self.mirrorable {
            self.shadow = Some(KvCache::new(1, self.growth));
        }
    }

    /// Positions currently held by the mirror, or `None` if there isn't one.
    #[cfg(test)]
    pub(crate) fn shadow_len_for_test(&self) -> Option<usize> {
        self.shadow.as_ref().map(|s| s.current_seq_len())
    }

    /// Push the mirror out of step with the f32 cache on purpose, to prove the
    /// divergence guard actually refuses rather than serving it.
    #[cfg(test)]
    pub(crate) fn desync_mirror_for_test(&mut self, k: &Tensor) {
        if let Some(shadow) = self.shadow.as_mut() {
            let extra = Self::to_bshd_f16(k).unwrap();
            shadow.append(&extra, &extra).unwrap();
        }
    }

    /// Every underlying cache holding memory — the f32 pair, plus the f16
    /// mirror's pair when it exists. For accounting only.
    pub(crate) fn all_caches(&self) -> Vec<&candle_nn::kv_cache::Cache> {
        let mut out = vec![self.main.k_cache(), self.main.v_cache()];
        if let Some(shadow) = self.shadow.as_ref() {
            out.push(shadow.k_cache());
            out.push(shadow.v_cache());
        }
        out
    }

    /// The mirrored (K, V) for the flash kernel: f16, BSHD, full length.
    ///
    /// `None` means no mirror — the caller converts the f32 cache as before.
    /// The length check is the guard against a silently drifted mirror: if it
    /// ever disagrees with the real cache, this declines rather than handing the
    /// kernel a history of the wrong length.
    pub(crate) fn flash_operands(&self) -> Option<(Tensor, Tensor)> {
        let shadow = self.shadow.as_ref()?;
        if shadow.current_seq_len() != self.main.current_seq_len() {
            // Deliberately a warning and a fallback, not an assertion. Drift is
            // a bug, but the safe response is to convert the f32 cache and
            // answer correctly — a debug assertion here would take a
            // development node down instead, and would make the fallback itself
            // untestable.
            tracing::warn!(
                mirror = shadow.current_seq_len(),
                cache = self.main.current_seq_len(),
                "KV mirror length diverged from the f32 cache — converting instead"
            );
            return None;
        }
        match (shadow.k(), shadow.v()) {
            (Ok(Some(k)), Ok(Some(v))) => Some((k, v)),
            _ => None,
        }
    }
}

/// `SWARMLLM_DISABLE_KV_MIRROR=1` turns the mirror off at runtime, so the two
/// arms can be compared inside ONE binary. Two separately-compiled binaries
/// differ in more than the change under test, which is how this project has
/// mismeasured before — the same reason `SWARMLLM_FORCE_STANDARD_ATTN` exists.
#[cfg(feature = "flash-attn")]
fn mirror_disabled() -> bool {
    use std::sync::OnceLock;
    static OFF: OnceLock<bool> = OnceLock::new();
    *OFF.get_or_init(|| {
        std::env::var("SWARMLLM_DISABLE_KV_MIRROR")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
    })
}

impl std::ops::Deref for LayerKv {
    type Target = KvCache;
    fn deref(&self) -> &KvCache {
        &self.main
    }
}

impl std::ops::DerefMut for LayerKv {
    fn deref_mut(&mut self) -> &mut KvCache {
        &mut self.main
    }
}

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
    pub(crate) layers: Vec<Option<LayerKv>>,
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
            // The f16 mirror is real allocated VRAM (half the f32 cache's bytes
            // for the same positions), so it is counted here. Leaving it out
            // would under-report KV memory by a third and quietly loosen the
            // head-room admission check that reads this.
            for cache in kv.all_caches() {
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
                layers: (0..num_layers).map(|_| None).collect(),
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
        let mut kv = new_kv_cache(max_seq_len, true);
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

    // ── LayerKv: the f16 BSHD mirror ──
    //
    // These run on CPU, where `wants_shadow` is false and no mirror is built,
    // so they drive the mirror explicitly via `force_shadow_for_test`. That is
    // deliberate: the invariants below are about bookkeeping (does the mirror
    // track the f32 cache through append / reset / truncate), and bookkeeping is
    // exactly what silently drifts. The GPU-only part — that flash consumes it —
    // is covered by the end-to-end benchmark.

    fn t(dev: &candle_core::Device, b: usize, h: usize, s: usize, d: usize) -> Tensor {
        Tensor::arange(0f32, (b * h * s * d) as f32, dev)
            .unwrap()
            .reshape((b, h, s, d))
            .unwrap()
    }

    #[test]
    fn mirror_tracks_the_f32_cache_across_appends() {
        let dev = candle_core::Device::Cpu;
        let mut kv = LayerKv::new(64);
        kv.force_shadow_for_test();
        for _ in 0..5 {
            let k = t(&dev, 1, 2, 3, 4);
            kv.append(&k, &k).unwrap();
        }
        assert_eq!(kv.current_seq_len(), 15);
        let (mk, mv) = kv.flash_operands().expect("mirror should be available");
        // BSHD: [b, s, h, d] — sequence on dim 1, against BHSD's dim 2.
        assert_eq!(mk.dims(), &[1, 15, 2, 4]);
        assert_eq!(mv.dims(), &[1, 15, 2, 4]);
        assert_eq!(mk.dtype(), DType::F16);
    }

    #[test]
    fn mirror_holds_the_same_numbers_as_the_f32_cache() {
        let dev = candle_core::Device::Cpu;
        let mut kv = LayerKv::new(64);
        kv.force_shadow_for_test();
        let k = t(&dev, 1, 2, 3, 4);
        kv.append(&k, &k).unwrap();

        // The mirror must equal round_f16(f32 cache) transposed — that identity
        // is what makes the flash path numerically unchanged rather than merely
        // close, so it is asserted rather than assumed.
        let main = kv.k().unwrap().unwrap();
        let expect = main
            .transpose(1, 2)
            .unwrap()
            .contiguous()
            .unwrap()
            .to_dtype(DType::F16)
            .unwrap();
        let (mk, _) = kv.flash_operands().unwrap();
        let diff = (mk.to_dtype(DType::F32).unwrap() - expect.to_dtype(DType::F32).unwrap())
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert_eq!(diff, 0.0, "mirror diverged from the f32 cache it mirrors");
    }

    #[test]
    fn reset_clears_both_representations() {
        let dev = candle_core::Device::Cpu;
        let mut kv = LayerKv::new(64);
        kv.force_shadow_for_test();
        let k = t(&dev, 1, 2, 3, 4);
        kv.append(&k, &k).unwrap();
        kv.reset();
        assert_eq!(kv.current_seq_len(), 0);
        // A mirror still holding the old positions would silently feed the
        // kernel a history that no longer exists.
        assert!(
            kv.flash_operands().is_none() || kv.shadow_len_for_test() == Some(0),
            "mirror survived a reset"
        );
        kv.append(&k, &k).unwrap();
        assert_eq!(kv.shadow_len_for_test(), Some(3));
    }

    #[test]
    fn truncation_keeps_the_mirror_in_step() {
        // Speculative decode rejects tokens by truncating the cache. This is the
        // path where a drifted mirror would produce plausible wrong attention
        // rather than an error, so it is asserted end to end through the store.
        let dev = candle_core::Device::Cpu;
        let store = KvCacheStore::new(std::time::Duration::from_secs(60));
        {
            let mut e = store.get_or_create("m", "r", 1);
            let mut slot = LayerKv::new(64);
            slot.force_shadow_for_test();
            let k = t(&dev, 1, 2, 8, 4);
            slot.append(&k, &k).unwrap();
            assert_eq!(slot.current_seq_len(), 8);
            e.layers[0] = Some(slot);
        }
        store.truncate_request_to("m", "r", 5).unwrap();
        let key = KvCacheStore::cache_key("m", "r");
        let e = store.get_entry(&key).unwrap();
        let slot = e.layers[0].as_ref().unwrap();
        assert_eq!(slot.current_seq_len(), 5);
        assert_eq!(
            slot.shadow_len_for_test(),
            Some(5),
            "mirror kept a different number of positions than the f32 cache"
        );
        assert!(slot.flash_operands().is_some());
    }

    #[test]
    fn a_drifted_mirror_is_refused_rather_than_served() {
        // The last line of defence: if the two ever disagree, the flash path
        // must fall back to converting the f32 cache, not hand the kernel a
        // history of the wrong length.
        let dev = candle_core::Device::Cpu;
        let mut kv = LayerKv::new(64);
        kv.force_shadow_for_test();
        let k = t(&dev, 1, 2, 3, 4);
        kv.append(&k, &k).unwrap();
        assert!(kv.flash_operands().is_some());
        kv.desync_mirror_for_test(&k);
        assert!(
            kv.flash_operands().is_none(),
            "a mirror of the wrong length was served to the kernel"
        );
    }

    #[test]
    fn a_non_bhsd_cache_never_builds_a_mirror() {
        // Prefix-cache snapshots carry their own sequence dim; the mirror's
        // transpose only means anything for dim 2.
        let mut kv = LayerKv::with_dim(1, 64);
        kv.force_shadow_for_test();
        assert!(kv.flash_operands().is_none());
    }
}
