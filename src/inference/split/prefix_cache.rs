//! Cross-request prefix KV-cache.
//!
//! Stores per-model snapshots of the full KV-cache state captured at the end
//! of prompt prefill. On a subsequent request, looks up the longest cached
//! token prefix that is a strict prefix of the new prompt's tokens; on hit,
//! materialises a fresh `KvCacheEntry` seeded with the cached tensors so
//! prefill skips those positions and only processes the suffix.
//!
//! Scope (v1):
//! - Per-worker-subprocess, per-model-key. One instance manages all models
//!   inside a worker.
//! - Flat set of entries per model, bounded by `max_entries`. No radix tree;
//!   insertion happens at block boundaries AND at the full prompt tail so
//!   cross-user "same system prompt" sharing still hits (first N block
//!   boundaries coincide) while staying simple.
//! - LRU eviction by `last_hit` timestamp.
//! - Tensors are cloned on restore (candle `append` copies into a fresh
//!   pre-allocated buffer). No reference-counted sharing across live
//!   requests.
//!
//! What this does NOT do:
//! - SSM / hybrid-model state (Qwen3.5-SSM). Caches containing SSM state
//!   are skipped; the model runs a full prefill.
//! - Cross-node sharing. The snapshot lives only in this worker process.
//! - True radix-tree de-duplication. Storage is `O(entries * prompt_len *
//!   hidden * layers)` in the worst case — configure `max_entries` with
//!   this in mind.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::time::Instant;

use crate::inference::split::kv_cache::LayerKv;
use candle_core::Tensor;

use crate::error::SwarmError;
use crate::types::PrefixBlockEntry;

use super::kv_cache::{KvCacheEntry, KvCacheStore};

/// Compute the chained BLAKE3 hash chain over a token sequence, in blocks of
/// `block_size` tokens. Returns one entry per block boundary that lies inside
/// `tokens` (i.e. positions `block_size, 2*block_size, ...`). The trailing
/// partial block (if any) is NOT hashed — only complete blocks count, so two
/// peers running the same `block_size` will agree on every hash value for
/// every shared prefix.
///
/// Each block's hash is `blake3(prev_hash || u32_le(tokens[i*B..(i+1)*B]))`,
/// with `prev_hash` empty for the first block. This gives prefix-property:
/// two prompts that share the first `k` blocks have identical `block_hash[0..k]`.
pub fn compute_block_hashes(tokens: &[u32], block_size: usize) -> Vec<PrefixBlockEntry> {
    if block_size == 0 {
        return Vec::new();
    }
    let block_count = tokens.len() / block_size;
    let mut out = Vec::with_capacity(block_count);
    let mut prev: Option<[u8; 32]> = None;
    for i in 0..block_count {
        let mut hasher = blake3::Hasher::new();
        if let Some(p) = prev {
            hasher.update(&p);
        }
        let start = i * block_size;
        for &tok in &tokens[start..start + block_size] {
            hasher.update(&tok.to_le_bytes());
        }
        let h: [u8; 32] = hasher.finalize().into();
        prev = Some(h);
        out.push(PrefixBlockEntry {
            block_hash: h,
            token_count: ((i + 1) * block_size) as u32,
        });
    }
    out
}

/// Magic bytes identifying a serialized `KvSnapshot` on the wire.
/// `SKVX` = "SwarmLLM KV eXchange". Helps early-reject garbled payloads.
pub const KV_SNAPSHOT_MAGIC: &[u8; 4] = b"SKVX";
/// Snapshot wire format version. Bump when the frame layout changes
/// incompatibly; older nodes reject unknown versions on receive.
pub const KV_SNAPSHOT_VERSION: u32 = 1;

/// BLAKE3-verify that the chained hash over `tokens[..expected_token_count]`
/// at `block_size` granularity produces `expected_hash` at its last block.
/// Returns `true` iff the final chained hash matches — Phase 2 fetchers use
/// this to reject peers that return KV data not matching the requested
/// block hash.
pub fn verify_token_hash_chain(
    tokens: &[u32],
    block_size: usize,
    expected_token_count: usize,
    expected_hash: &[u8; 32],
) -> bool {
    if block_size == 0 || expected_token_count == 0 || expected_token_count > tokens.len() {
        return false;
    }
    if !expected_token_count.is_multiple_of(block_size) {
        // `token_count` must land on a block boundary — the chain hash is
        // only defined at complete-block positions.
        return false;
    }
    let manifest = compute_block_hashes(&tokens[..expected_token_count], block_size);
    match manifest.last() {
        Some(last) => {
            last.token_count as usize == expected_token_count && last.block_hash == *expected_hash
        }
        None => false,
    }
}

/// Per-layer KV snapshot at a specific token position.
pub struct KvSnapshot {
    /// Number of tokens covered by this snapshot (== K/V seq dim).
    pub token_count: usize,
    /// One `(K, V)` per layer in the same order as `KvCacheEntry.layers`.
    /// `None` for layers that had no cache at capture time (shouldn't happen
    /// for a completed prefill, but we tolerate it to avoid losing the entire
    /// cache to one missing layer).
    pub layers: Vec<Option<(Tensor, Tensor)>>,
    /// KV cache's per-layer seq dim (candle `KvCache::dim()`). Used when
    /// reconstructing a fresh `KvCache`.
    pub dim: usize,
    /// Pre-allocated capacity of the source KvCache (`max_seq_len`). Reused
    /// when rebuilding so the restored cache has the same headroom as a
    /// freshly-created one.
    pub max_seq_len: usize,
}

struct Entry {
    tokens: Vec<u32>,
    snapshot: Arc<KvSnapshot>,
    /// Logical clock tick of the most recent hit/insert. Used for LRU
    /// eviction (`bucket.sort_by_key`). Atomic so cache hits stay on the
    /// read-lock path — bumping the timestamp no longer needs a write
    /// lock upgrade.
    last_hit: AtomicU64,
}

impl Entry {
    /// Compute the chained BLAKE3 manifest for this entry's tokens at the
    /// configured block size. Only complete blocks contribute; the entry's
    /// trailing partial block (if any, when `tokens.len() % block_size != 0`)
    /// is omitted so cross-peer block boundaries align deterministically.
    fn manifest(&self, block_size: usize) -> Vec<PrefixBlockEntry> {
        compute_block_hashes(&self.tokens, block_size)
    }
}

struct Inner {
    /// Per-model entries keyed by `SplitModel::kv_model_key()`.
    per_model: HashMap<String, Vec<Entry>>,
}

/// Flat, longest-prefix prefix KV-cache shared across requests on a worker.
pub struct PrefixCache {
    inner: RwLock<Inner>,
    /// Monotonic logical clock for LRU ordering. Bumped on every hit and
    /// insert; the value is stored on the touched entry's `last_hit`
    /// atomic. Lookups can therefore record a hit without upgrading from
    /// read to write lock.
    clock: AtomicU64,
    /// Maximum entries retained per model. Older entries (by `last_hit`)
    /// are evicted when the cap is exceeded.
    max_entries: usize,
    /// Minimum prefix length (tokens) below which lookups return miss and
    /// inserts are skipped. Avoids caching trivial prompts.
    min_tokens: usize,
    /// Prompts longer than this (in tokens) are not inserted — they'd blow
    /// memory. Lookups against long prompts still walk the cache.
    max_prompt_tokens: usize,
    /// Granularity for multi-point inserts. For a prompt of length L, inserts
    /// happen at positions `block_tokens, 2*block_tokens, ...` that are `≥
    /// min_tokens` and `< L`, plus one at L itself. 0 disables multi-point
    /// inserts (only the full prompt is stored).
    block_tokens: usize,
    enabled: bool,
}

impl PrefixCache {
    pub fn new(
        enabled: bool,
        max_entries: usize,
        block_tokens: usize,
        min_tokens: usize,
        max_prompt_tokens: usize,
    ) -> Self {
        Self {
            inner: RwLock::new(Inner {
                per_model: HashMap::new(),
            }),
            clock: AtomicU64::new(0),
            max_entries,
            min_tokens,
            max_prompt_tokens,
            block_tokens,
            enabled,
        }
    }

    /// Bump the logical clock and return the new tick. Used as the
    /// timestamp written to an entry's `last_hit`.
    fn next_tick(&self) -> u64 {
        // Relaxed is fine: we only need monotonicity; total ordering across
        // entries doesn't matter as long as each entry's last_hit reflects a
        // tick from after the operation that touched it.
        self.clock.fetch_add(1, Ordering::Relaxed) + 1
    }

    pub fn enabled(&self) -> bool {
        self.enabled
    }

    /// Block-size used to chunk prompt tokens into chained BLAKE3 hashes for
    /// cross-node prefix-cache sharing. Returns 0 when multi-point inserts
    /// are disabled — callers should treat 0 as "no manifest" and skip
    /// announcing.
    pub fn block_tokens(&self) -> usize {
        self.block_tokens
    }

    /// Find the longest cached token prefix of `input_tokens` for `model_key`.
    /// Returns `None` if no suitable prefix is cached or the cache is disabled.
    pub fn lookup(&self, model_key: &str, input_tokens: &[u32]) -> Option<Arc<KvSnapshot>> {
        if !self.enabled || input_tokens.len() < self.min_tokens {
            return None;
        }
        // Must keep at least one token to forward — otherwise the model has
        // nothing to compute logits for. Clamp the usable prefix length.
        let usable_max = input_tokens.len().saturating_sub(1);

        // Fast path: read lock only. Walk to find the longest prefix, clone
        // the winning snapshot Arc, and bump the entry's atomic last_hit
        // without ever upgrading to a write lock.
        let inner = self.inner.read().ok()?;
        let entries = inner.per_model.get(model_key)?;
        let mut best: Option<&Entry> = None;
        for e in entries.iter() {
            if e.tokens.len() < self.min_tokens || e.tokens.len() > usable_max {
                continue;
            }
            if input_tokens.starts_with(&e.tokens) {
                match best {
                    None => best = Some(e),
                    Some(cur) if e.tokens.len() > cur.tokens.len() => {
                        best = Some(e);
                    }
                    _ => {}
                }
            }
        }
        let winner = best?;
        winner.last_hit.store(self.next_tick(), Ordering::Relaxed);
        let snapshot = winner.snapshot.clone();
        drop(inner);
        tracing::info!(
            model_key,
            matched_tokens = snapshot.token_count,
            prompt_tokens = input_tokens.len(),
            "DIAG: prefix-cache HIT"
        );
        Some(snapshot)
    }

    /// Capture the current KV state of `kv_store` for `request_id` into
    /// snapshots at block boundaries and insert them into the cache. Called
    /// after prefill (or at end of decode) when the full prompt KV is
    /// populated.
    ///
    /// `prompt_tokens` is the token sequence that produced the KV state —
    /// its length must equal the current KV cache seq_len for this request.
    pub fn insert_from_kv(
        &self,
        model_key: &str,
        request_id: &str,
        kv_store: &KvCacheStore,
        prompt_tokens: &[u32],
    ) -> Vec<PrefixBlockEntry> {
        // Each bail-out below says why. They were silent, and that made a
        // cross-node prefix-KV measurement undiagnosable: the producing node
        // simply never announced any blocks, with nothing in the log to say
        // which condition stopped it.
        if !self.enabled
            || prompt_tokens.len() < self.min_tokens
            || prompt_tokens.len() > self.max_prompt_tokens
        {
            tracing::debug!(
                model_key,
                enabled = self.enabled,
                prompt_tokens = prompt_tokens.len(),
                min_tokens = self.min_tokens,
                max_prompt_tokens = self.max_prompt_tokens,
                "prefix-cache: not snapshotting — disabled or prompt outside size bounds"
            );
            return Vec::new();
        }
        let key = KvCacheStore::cache_key(model_key, request_id);
        let Some(entry_ref) = kv_store_get(kv_store, &key) else {
            tracing::debug!(
                model_key,
                request_id,
                "prefix-cache: no KV entry to snapshot"
            );
            return Vec::new();
        };
        let entry = entry_ref;

        // Skip SSM/hybrid models — snapshot support not yet implemented.
        if entry.ssm_states.iter().any(|s| s.is_some()) {
            tracing::debug!(
                model_key,
                "prefix-cache: SSM state present, skipping snapshot"
            );
            return Vec::new();
        }

        // Figure out the seq dim / max_seq_len from the first populated layer.
        let Some(first_kv) = entry.layers.iter().flatten().next() else {
            tracing::debug!(
                model_key,
                "prefix-cache: not snapshotting — KV entry has no populated layers"
            );
            return Vec::new();
        };
        let dim = first_kv.k_cache().dim();
        let max_seq_len = first_kv.k_cache().max_seq_len();
        let available = first_kv.current_seq_len();
        // Bound insertion points by what the KV actually holds.
        let available = available.min(prompt_tokens.len());
        if available < self.min_tokens {
            tracing::debug!(
                model_key,
                available,
                min_tokens = self.min_tokens,
                "prefix-cache: not snapshotting — fewer KV positions than the floor"
            );
            return Vec::new();
        }

        let insert_points = self.compute_insert_points(available);
        if insert_points.is_empty() {
            tracing::debug!(
                model_key,
                available,
                block_tokens = self.block_tokens,
                min_tokens = self.min_tokens,
                "prefix-cache: not snapshotting — no block boundary at or above the floor"
            );
            return Vec::new();
        }

        let insert_points_len = insert_points.len();
        let mut snapshots: Vec<(usize, Arc<KvSnapshot>)> = Vec::with_capacity(insert_points_len);
        for pos in insert_points {
            match snapshot_at(&entry.layers, pos, dim, max_seq_len) {
                Ok(snap) => snapshots.push((pos, Arc::new(snap))),
                Err(e) => {
                    // Stop at the first failure instead of trying the rest.
                    //
                    // `compute_insert_points` returns ASCENDING block
                    // boundaries and a snapshot copies `[0..pos]` of every
                    // layer's K and V, so each remaining point allocates
                    // strictly MORE than the one that just failed. Once one
                    // fails there is no prospect of a later one succeeding.
                    //
                    // Continuing was not free: on a 6 GB card a long prefill
                    // failed here every 64 tokens, so the request paid a futile
                    // allocation attempt and emitted a warning per block, on
                    // exactly the machine least able to spare either. Reported
                    // 2026-08-02.
                    //
                    // Snapshots already taken at SMALLER positions are kept —
                    // they succeeded and are still useful prefixes — so this
                    // degrades to "cache as much as fitted" rather than
                    // dropping the lot.
                    tracing::warn!(
                        model_key,
                        pos,
                        captured = snapshots.len(),
                        error = %e,
                        "prefix-cache: snapshot failed — keeping what fitted and \
                         skipping the larger insert points for this request"
                    );
                    break;
                }
            }
        }
        drop(entry);

        if snapshots.is_empty() {
            tracing::debug!(
                model_key,
                attempted = insert_points_len,
                "prefix-cache: not snapshotting — every snapshot attempt failed"
            );
            return Vec::new();
        }

        let Ok(mut inner) = self.inner.write() else {
            return Vec::new();
        };
        let bucket = inner.per_model.entry(model_key.to_string()).or_default();
        for (pos, snap) in snapshots {
            let tokens = prompt_tokens[..pos].to_vec();
            let tick = self.next_tick();
            // Skip if an entry for this exact prefix already exists; bump LRU.
            if let Some(existing) = bucket.iter_mut().find(|e| e.tokens == tokens) {
                existing.last_hit.store(tick, Ordering::Relaxed);
                existing.snapshot = snap;
                continue;
            }
            bucket.push(Entry {
                tokens,
                snapshot: snap,
                last_hit: AtomicU64::new(tick),
            });
        }

        // Evict LRU until within cap.
        if bucket.len() > self.max_entries {
            bucket.sort_by_key(|e| e.last_hit.load(Ordering::Relaxed));
            let drop_count = bucket.len() - self.max_entries;
            bucket.drain(..drop_count);
        }

        let entry_count = bucket.len();
        // Compute the full post-insert manifest for this model so the caller
        // can announce our current cache state (not just the delta). Cheap:
        // BLAKE3 on a few KB of token IDs.
        let manifest = enumerate_manifest_locked(bucket, self.block_tokens);

        tracing::info!(
            model_key,
            entries = entry_count,
            manifest_len = manifest.len(),
            "DIAG: prefix-cache inserted snapshot"
        );
        manifest
    }

    /// Item 8 Phase 2b: look up a cached snapshot whose chained-hash
    /// manifest contains `block_hash`, serialize it, and return the wire
    /// bytes. Returns `None` when no matching entry exists (eviction race
    /// or never cached for this hash). Used by the serving-side
    /// `ExportPrefixSnapshot` handler.
    ///
    /// We must return a snapshot whose `token_count` equals the block
    /// boundary the hash represents — so we build a fresh snapshot by
    /// narrowing the entry's live KV tensors to that position. The
    /// PrefixCache doesn't store per-block sub-snapshots (would bloat
    /// memory); it stores one snapshot per entry at the entry's full
    /// length, plus the ability to re-narrow to any block boundary at
    /// serve time via the entry's KvSnapshot layers (already
    /// complete-seq-len copies narrowed at insert time).
    pub fn export_snapshot_bytes(&self, model_key: &str, block_hash: &[u8; 32]) -> Option<Vec<u8>> {
        if self.block_tokens == 0 {
            return None;
        }
        let inner = self.inner.read().ok()?;
        let bucket = inner.per_model.get(model_key)?;
        let block_size = self.block_tokens;
        // Find the entry whose manifest contains `block_hash`, along with
        // the block index where it appears (== token boundary).
        for entry in bucket {
            let manifest = entry.manifest(block_size);
            if let Some((i, bm)) = manifest
                .iter()
                .enumerate()
                .find(|(_, bm)| bm.block_hash == *block_hash)
            {
                let target_tokens = bm.token_count as usize;
                // Narrow the stored snapshot to the target boundary if the
                // entry itself is longer — each entry holds a snapshot at
                // `entry.tokens.len()` tokens but we want `target_tokens`.
                let narrowed = narrow_snapshot(&entry.snapshot, target_tokens).ok()?;
                let bytes = serialize_snapshot_with_block_size(
                    &narrowed,
                    &entry.tokens[..target_tokens],
                    Some(block_size),
                )
                .ok()?;
                tracing::debug!(
                    model_key,
                    block_index = i,
                    target_tokens,
                    bytes_len = bytes.len(),
                    "DIAG: export_snapshot_bytes HIT"
                );
                return Some(bytes);
            }
        }
        None
    }

    /// Snapshot the current per-model BLAKE3 block-hash manifest. Used by
    /// tests to validate the cache's announce-payload shape; production
    /// re-announce flow uses `enumerate_manifest_locked` directly from
    /// `insert_from_kv`. Returns an empty vec when the model has no entries
    /// or `block_tokens == 0`.
    #[cfg(test)]
    pub fn enumerate_manifest(&self, model_key: &str) -> Vec<PrefixBlockEntry> {
        if self.block_tokens == 0 {
            return Vec::new();
        }
        let Ok(inner) = self.inner.read() else {
            return Vec::new();
        };
        let Some(bucket) = inner.per_model.get(model_key) else {
            return Vec::new();
        };
        enumerate_manifest_locked(bucket, self.block_tokens)
    }

    /// Item 8 Phase 2b: hydrate a per-request KV entry directly from
    /// serialized snapshot bytes returned by a remote peer. Returns the
    /// number of tokens seeded, or an error if deserialization failed.
    /// Does NOT re-BLAKE3-verify — that happens earlier in the daemon's
    /// `try_fetch_cross_node_prefix` helper.
    pub fn hydrate_request_from_bytes(
        &self,
        kv_store: &KvCacheStore,
        model_key: &str,
        request_id: &str,
        bytes: &[u8],
        device: &candle_core::Device,
    ) -> Result<usize, SwarmError> {
        let (snap, _tokens) = deserialize_snapshot(bytes, device)?;
        self.hydrate_request_from_snapshot(kv_store, model_key, request_id, &snap)
    }

    /// Seed a fresh `KvCacheEntry` for `request_id` from `snapshot`. Returns
    /// the number of tokens seeded. Creates the entry if it doesn't exist.
    pub fn hydrate_request_from_snapshot(
        &self,
        kv_store: &KvCacheStore,
        model_key: &str,
        request_id: &str,
        snapshot: &KvSnapshot,
    ) -> Result<usize, SwarmError> {
        let num_layers = snapshot.layers.len();
        let key = KvCacheStore::cache_key(model_key, request_id);
        let mut entry = kv_store.get_or_create_keyed(&key, num_layers);
        for (i, layer_kv) in snapshot.layers.iter().enumerate() {
            let Some((k_src, v_src)) = layer_kv else {
                continue;
            };
            let kv_slot = &mut entry.layers[i];
            // Reserve from the snapshot's TOKEN COUNT, not its recorded
            // `max_seq_len`. Snapshots cross the network between peers
            // (`export_snapshot_bytes`), so that field may have been written by
            // a build that reserved a model's whole context window up front —
            // honouring it would re-inflate the reservation this node just
            // stopped making. See `KV_CACHE_GROWTH_TOKENS`.
            let mut kv = LayerKv::with_dim(
                snapshot.dim,
                crate::inference::layers::kv_cache_reservation(snapshot.token_count),
            );
            kv.append(k_src, v_src).map_err(|e| {
                SwarmError::Internal(format!(
                    "prefix-cache hydrate layer {i}: append failed: {e}"
                ))
            })?;
            *kv_slot = Some(kv);
        }
        entry.last_accessed = Instant::now();
        Ok(snapshot.token_count)
    }

    fn compute_insert_points(&self, available: usize) -> Vec<usize> {
        let mut points = Vec::new();
        if self.block_tokens > 0 {
            let mut p = self.block_tokens;
            while p < available {
                if p >= self.min_tokens {
                    points.push(p);
                }
                p += self.block_tokens;
            }
        }
        if available >= self.min_tokens {
            points.push(available);
        }
        points.dedup();
        points
    }

    #[cfg(test)]
    pub fn entry_count(&self, model_key: &str) -> usize {
        self.inner
            .read()
            .ok()
            .and_then(|i| i.per_model.get(model_key).map(|v| v.len()))
            .unwrap_or(0)
    }
}

fn kv_store_get<'a>(
    kv_store: &'a KvCacheStore,
    key: &str,
) -> Option<dashmap::mapref::one::Ref<'a, String, KvCacheEntry>> {
    kv_store.get_entry(key)
}

/// Build the deduped union of every entry's chained block manifest in a
/// bucket. Two entries in the same bucket whose tokens share a common prefix
/// produce identical block hashes for the shared portion (chained-hash
/// prefix property), so dedup-by-`block_hash` is correct and avoids
/// re-broadcasting the same block under multiple entries.
fn enumerate_manifest_locked(bucket: &[Entry], block_size: usize) -> Vec<PrefixBlockEntry> {
    if block_size == 0 || bucket.is_empty() {
        return Vec::new();
    }
    let mut seen: std::collections::HashSet<[u8; 32]> = std::collections::HashSet::new();
    let mut out = Vec::new();
    for entry in bucket {
        for block in entry.manifest(block_size) {
            if seen.insert(block.block_hash) {
                out.push(block);
            }
        }
    }
    out
}

/// Narrow an existing `KvSnapshot` to `target_tokens` on the seq dim.
/// Returns a fresh snapshot with independent tensors (contiguous copies),
/// so the caller can safely serialize + ship it without aliasing the
/// source. `target_tokens` must be `<= snap.token_count`.
fn narrow_snapshot(snap: &KvSnapshot, target_tokens: usize) -> Result<KvSnapshot, SwarmError> {
    if target_tokens > snap.token_count {
        return Err(SwarmError::Internal(format!(
            "narrow_snapshot: target {} > current {}",
            target_tokens, snap.token_count
        )));
    }
    if target_tokens == snap.token_count {
        // Shallow clone — candle Tensors are ref-counted.
        let layers: Vec<Option<(Tensor, Tensor)>> = snap
            .layers
            .iter()
            .map(|kv| kv.as_ref().map(|(k, v)| (k.clone(), v.clone())))
            .collect();
        return Ok(KvSnapshot {
            token_count: target_tokens,
            layers,
            dim: snap.dim,
            max_seq_len: snap.max_seq_len,
        });
    }
    let mut out: Vec<Option<(Tensor, Tensor)>> = Vec::with_capacity(snap.layers.len());
    for kv_opt in &snap.layers {
        let Some((k, v)) = kv_opt else {
            out.push(None);
            continue;
        };
        let k_narrow = k
            .narrow(snap.dim, 0, target_tokens)
            .and_then(|t| t.contiguous())
            .map_err(|e| SwarmError::Internal(format!("narrow_snapshot k: {e}")))?;
        let v_narrow = v
            .narrow(snap.dim, 0, target_tokens)
            .and_then(|t| t.contiguous())
            .map_err(|e| SwarmError::Internal(format!("narrow_snapshot v: {e}")))?;
        out.push(Some((k_narrow, v_narrow)));
    }
    Ok(KvSnapshot {
        token_count: target_tokens,
        layers: out,
        dim: snap.dim,
        max_seq_len: snap.max_seq_len,
    })
}

fn snapshot_at(
    layers: &[Option<LayerKv>],
    pos: usize,
    dim: usize,
    max_seq_len: usize,
) -> Result<KvSnapshot, SwarmError> {
    let mut out: Vec<Option<(Tensor, Tensor)>> = Vec::with_capacity(layers.len());
    for kv_opt in layers.iter() {
        let Some(kv) = kv_opt else {
            out.push(None);
            continue;
        };
        let cur = kv.current_seq_len();
        if cur < pos {
            return Err(SwarmError::Internal(format!(
                "snapshot_at: layer seq_len {cur} < requested pos {pos}"
            )));
        }
        let k_src = kv
            .k()
            .map_err(|e| SwarmError::Internal(format!("snapshot_at: k() failed: {e}")))?
            .ok_or_else(|| SwarmError::Internal("snapshot_at: k() returned None".into()))?;
        let v_src = kv
            .v()
            .map_err(|e| SwarmError::Internal(format!("snapshot_at: v() failed: {e}")))?
            .ok_or_else(|| SwarmError::Internal("snapshot_at: v() returned None".into()))?;
        // Narrow to [0..pos] on seq dim and force contiguous so the snapshot
        // doesn't alias the live KvCache buffer.
        let k_snap = k_src
            .narrow(dim, 0, pos)
            .and_then(|t| t.contiguous())
            .map_err(|e| SwarmError::Internal(format!("snapshot k narrow: {e}")))?;
        let v_snap = v_src
            .narrow(dim, 0, pos)
            .and_then(|t| t.contiguous())
            .map_err(|e| SwarmError::Internal(format!("snapshot v narrow: {e}")))?;
        out.push(Some((k_snap, v_snap)));
    }
    Ok(KvSnapshot {
        token_count: pos,
        layers: out,
        dim,
        max_seq_len,
    })
}

// ---- Item 8 Phase 2: KvSnapshot wire serialization -------------------------
//
// Format (all lengths little-endian):
//
//   [0..4]    magic = KV_SNAPSHOT_MAGIC ("SKVX")
//   [4..8]    version = KV_SNAPSHOT_VERSION (u32)
//   [8..16]   header_len (u64)
//   [16..16+H] serde_json header (see `SnapshotHeader`)
//   [16+H..]  concatenated per-layer [K f32 LE | V f32 LE] bytes, in the
//             same order as `header.layers`. Only layers with `Some(meta)`
//             contribute bytes; `None` entries emit zero bytes.
//
// Every K/V tensor is cast to `DType::F32` on the sender side before
// encoding. The receiver casts back to whatever its local `KvCache`
// expects (usually f32 on CPU, f16 on CUDA). This costs 2× wire size for
// an f16 sender but is portable across GPU/CPU peer pairings — a common
// configuration on SwarmLLM's heterogeneous swarm. Compression is
// handled one level up by the request_response zstd wrapper.

#[derive(serde::Serialize, serde::Deserialize)]
struct SnapshotLayerMeta {
    k_shape: Vec<usize>,
    v_shape: Vec<usize>,
    /// Number of f32 values in K (same count for V). Used to offset the
    /// binary body on receive.
    k_count: usize,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct SnapshotHeader {
    token_count: usize,
    /// Full u32 token IDs covering the snapshot. Receiver uses these to
    /// BLAKE3-verify the returned block matches the hash they requested.
    tokens: Vec<u32>,
    dim: usize,
    max_seq_len: usize,
    /// Block size the sender used to chain-hash the prompt prefix. The
    /// receiver re-hashes `tokens[..token_count]` at this `block_size` to
    /// verify against the requested `block_hash`. Optional for backward
    /// compatibility with pre-Phase-3 snapshots — when absent, the
    /// verifier falls back to a list of common block sizes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    block_size: Option<usize>,
    /// Per-layer metadata; `None` marks layers the sender had no cache for
    /// (shouldn't happen for a completed prefill, tolerated for robustness).
    layers: Vec<Option<SnapshotLayerMeta>>,
}

/// Serialize a `KvSnapshot` (plus the token IDs it covers) to a wire frame.
///
/// `tokens.len()` must be `>= snap.token_count`; only `tokens[..token_count]`
/// is recorded in the header. Every tensor in `snap.layers` is cast to f32
/// before encoding so the frame is device-independent.
///
/// `block_size: Some(N)` records the sender's block size so the receiver can
/// BLAKE3-verify `tokens[..token_count]` against the requested block hash
/// deterministically — every production caller passes it. `None` is tolerated
/// for pre-Phase-3 compatibility, where the verifier falls back to guessing
/// common defaults; the round-trip tests still cover that path.
pub fn serialize_snapshot_with_block_size(
    snap: &KvSnapshot,
    tokens: &[u32],
    block_size: Option<usize>,
) -> Result<Vec<u8>, SwarmError> {
    if tokens.len() < snap.token_count {
        return Err(SwarmError::Internal(format!(
            "serialize_snapshot: tokens.len() {} < token_count {}",
            tokens.len(),
            snap.token_count
        )));
    }
    let mut layer_meta: Vec<Option<SnapshotLayerMeta>> = Vec::with_capacity(snap.layers.len());
    let mut body: Vec<u8> = Vec::new();
    for kv_opt in &snap.layers {
        let Some((k, v)) = kv_opt else {
            layer_meta.push(None);
            continue;
        };
        let k_f32 = k
            .to_dtype(candle_core::DType::F32)
            .and_then(|t| t.contiguous())
            .map_err(|e| SwarmError::Internal(format!("serialize_snapshot k cast: {e}")))?;
        let v_f32 = v
            .to_dtype(candle_core::DType::F32)
            .and_then(|t| t.contiguous())
            .map_err(|e| SwarmError::Internal(format!("serialize_snapshot v cast: {e}")))?;
        let k_shape: Vec<usize> = k_f32.shape().dims().to_vec();
        let v_shape: Vec<usize> = v_f32.shape().dims().to_vec();
        let k_vec: Vec<f32> = k_f32
            .flatten_all()
            .and_then(|t| t.to_vec1::<f32>())
            .map_err(|e| SwarmError::Internal(format!("serialize_snapshot k flatten: {e}")))?;
        let v_vec: Vec<f32> = v_f32
            .flatten_all()
            .and_then(|t| t.to_vec1::<f32>())
            .map_err(|e| SwarmError::Internal(format!("serialize_snapshot v flatten: {e}")))?;
        let k_count = k_vec.len();
        body.reserve(4 * (k_vec.len() + v_vec.len()));
        for f in &k_vec {
            body.extend_from_slice(&f.to_le_bytes());
        }
        for f in &v_vec {
            body.extend_from_slice(&f.to_le_bytes());
        }
        layer_meta.push(Some(SnapshotLayerMeta {
            k_shape,
            v_shape,
            k_count,
        }));
    }
    let header = SnapshotHeader {
        token_count: snap.token_count,
        tokens: tokens[..snap.token_count].to_vec(),
        dim: snap.dim,
        max_seq_len: snap.max_seq_len,
        block_size,
        layers: layer_meta,
    };
    let header_bytes =
        serde_json::to_vec(&header).map_err(|e| SwarmError::Internal(format!("header: {e}")))?;
    let header_len = header_bytes.len() as u64;
    let mut out = Vec::with_capacity(4 + 4 + 8 + header_bytes.len() + body.len());
    out.extend_from_slice(KV_SNAPSHOT_MAGIC);
    out.extend_from_slice(&KV_SNAPSHOT_VERSION.to_le_bytes());
    out.extend_from_slice(&header_len.to_le_bytes());
    out.extend_from_slice(&header_bytes);
    out.extend_from_slice(&body);
    Ok(out)
}

/// Item 8 Phase 3: sanity-check a just-deserialized KV snapshot for
/// numerical corruption. Returns `true` if every populated layer's K and
/// V tensors contain only finite values (no NaN, no ±Inf). A failure
/// here almost always indicates a malicious or broken peer — the caller
/// drops the snapshot and penalizes trust. Called BEFORE hydration so a
/// bad peer can never poison our KV cache.
pub fn snapshot_is_finite(snap: &KvSnapshot) -> bool {
    for kv_opt in &snap.layers {
        let Some((k, v)) = kv_opt else { continue };
        for tensor in [k, v] {
            let flat: Vec<f32> = match tensor
                .to_dtype(candle_core::DType::F32)
                .and_then(|t| t.flatten_all())
                .and_then(|t| t.to_vec1::<f32>())
            {
                Ok(v) => v,
                Err(_) => return false,
            };
            if flat.iter().any(|f| !f.is_finite()) {
                return false;
            }
        }
    }
    true
}

/// Deserialize a wire-format KV snapshot on `device`. Returns the
/// reconstructed `KvSnapshot` plus the token IDs the sender claimed the
/// snapshot covers (caller MUST re-hash these and match against the
/// requested block hash before trusting the data).
pub fn deserialize_snapshot(
    bytes: &[u8],
    device: &candle_core::Device,
) -> Result<(KvSnapshot, Vec<u32>), SwarmError> {
    deserialize_snapshot_full(bytes, device).map(|(s, t, _)| (s, t))
}

/// Phase 3 variant: also returns the sender's `block_size` from the
/// header (`None` for pre-Phase-3 senders that didn't record it).
pub fn deserialize_snapshot_full(
    bytes: &[u8],
    device: &candle_core::Device,
) -> Result<(KvSnapshot, Vec<u32>, Option<usize>), SwarmError> {
    // Header framing: magic(4) + version(4) + header_len(8) = 16 bytes.
    if bytes.len() < 16 {
        return Err(SwarmError::Internal("snapshot: frame too short".into()));
    }
    if &bytes[..4] != KV_SNAPSHOT_MAGIC {
        return Err(SwarmError::Internal("snapshot: magic mismatch".into()));
    }
    let version = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
    if version != KV_SNAPSHOT_VERSION {
        return Err(SwarmError::Internal(format!(
            "snapshot: unsupported version {version}"
        )));
    }
    let header_len = u64::from_le_bytes(bytes[8..16].try_into().unwrap()) as usize;
    if bytes.len() < 16 + header_len {
        return Err(SwarmError::Internal("snapshot: truncated header".into()));
    }
    let header: SnapshotHeader = serde_json::from_slice(&bytes[16..16 + header_len])
        .map_err(|e| SwarmError::Internal(format!("snapshot header decode: {e}")))?;
    let body = &bytes[16 + header_len..];
    let mut cursor = 0usize;
    let mut layers: Vec<Option<(Tensor, Tensor)>> = Vec::with_capacity(header.layers.len());
    for meta_opt in header.layers {
        let Some(meta) = meta_opt else {
            layers.push(None);
            continue;
        };
        let k_bytes = meta.k_count * 4;
        let v_count: usize = meta.v_shape.iter().product();
        let v_bytes = v_count * 4;
        if cursor + k_bytes + v_bytes > body.len() {
            return Err(SwarmError::Internal(format!(
                "snapshot: body too short at layer (have {}, need {})",
                body.len() - cursor,
                k_bytes + v_bytes
            )));
        }
        let k_vec: Vec<f32> = (0..meta.k_count)
            .map(|i| {
                let off = cursor + i * 4;
                f32::from_le_bytes([body[off], body[off + 1], body[off + 2], body[off + 3]])
            })
            .collect();
        cursor += k_bytes;
        let v_vec: Vec<f32> = (0..v_count)
            .map(|i| {
                let off = cursor + i * 4;
                f32::from_le_bytes([body[off], body[off + 1], body[off + 2], body[off + 3]])
            })
            .collect();
        cursor += v_bytes;
        let k_tensor = Tensor::from_vec(k_vec, meta.k_shape.as_slice(), device)
            .map_err(|e| SwarmError::Internal(format!("snapshot k rebuild: {e}")))?;
        let v_tensor = Tensor::from_vec(v_vec, meta.v_shape.as_slice(), device)
            .map_err(|e| SwarmError::Internal(format!("snapshot v rebuild: {e}")))?;
        layers.push(Some((k_tensor, v_tensor)));
    }
    let snap = KvSnapshot {
        token_count: header.token_count,
        layers,
        dim: header.dim,
        max_seq_len: header.max_seq_len,
    };
    Ok((snap, header.tokens, header.block_size))
}

#[cfg(test)]
mod tests {

    /// The insert points must ASCEND, because the snapshot loop now stops at
    /// the first failure instead of trying the rest.
    ///
    /// That break is only sound while every later point allocates more than the
    /// one that failed — a snapshot copies `[0..pos]` of every layer's K and V,
    /// so ascending positions mean monotonically growing allocations. If this
    /// ever returned unordered or descending points, breaking would start
    /// skipping points that could still have fitted, and the prefix cache would
    /// quietly cache less than it can.
    #[test]
    fn insert_points_ascend_so_stopping_at_the_first_failure_is_sound() {
        for (block, min_tokens, available) in [
            (64usize, 32usize, 512usize),
            (16, 16, 100),
            (128, 64, 129),
            (64, 32, 64),
        ] {
            let cache = PrefixCache::new(true, 8, block, min_tokens, 8192);
            let points = cache.compute_insert_points(available);
            for w in points.windows(2) {
                assert!(
                    w[1] > w[0],
                    "insert points must strictly ascend (block={block}, min={min_tokens}, \
                     available={available}), got {points:?}"
                );
            }
            assert!(
                points.iter().all(|p| *p <= available),
                "no insert point may exceed the available positions: {points:?}"
            );
        }
    }

    use super::*;
    use candle_core::{DType, Device};

    fn make_fake_kv(
        kv_store: &KvCacheStore,
        model_key: &str,
        request_id: &str,
        num_layers: usize,
        seq_len: usize,
    ) {
        // Build tiny KV tensors [1, 1, seq_len, 4] on CPU so snapshot math has
        // something concrete to narrow.
        let device = Device::Cpu;
        let mut entry = kv_store.get_or_create(model_key, request_id, num_layers);
        for slot in entry.layers.iter_mut() {
            let k = Tensor::zeros((1usize, 1, seq_len, 4), DType::F32, &device).unwrap();
            let v = Tensor::zeros((1usize, 1, seq_len, 4), DType::F32, &device).unwrap();
            // dim=2 is the sequence dim for this shape
            let mut kv = LayerKv::with_dim(2, 4096);
            kv.append(&k, &v).unwrap();
            *slot = Some(kv);
        }
    }

    #[test]
    fn lookup_miss_when_disabled() {
        let pc = PrefixCache::new(false, 8, 32, 8, 8192);
        assert!(pc.lookup("m", &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).is_none());
    }

    #[test]
    fn lookup_miss_below_min_tokens() {
        let pc = PrefixCache::new(true, 8, 32, 8, 8192);
        assert!(pc.lookup("m", &[1, 2, 3]).is_none());
    }

    #[test]
    fn insert_and_lookup_exact_prefix() {
        let pc = PrefixCache::new(true, 8, 0, 4, 8192);
        let kv_store = KvCacheStore::new(std::time::Duration::from_secs(600));
        make_fake_kv(&kv_store, "m", "req-a", 2, 10);

        let tokens: Vec<u32> = (1..=10).collect();
        pc.insert_from_kv("m", "req-a", &kv_store, &tokens);
        assert_eq!(pc.entry_count("m"), 1);

        // New request with same tokens + 5 more: should hit at 10.
        let new_tokens: Vec<u32> = (1..=15).collect();
        let snap = pc.lookup("m", &new_tokens).expect("hit");
        assert_eq!(snap.token_count, 10);
    }

    #[test]
    fn block_aligned_inserts_enable_partial_match() {
        // block=4, so after a 10-token prompt we insert at 4, 8, 10.
        let pc = PrefixCache::new(true, 16, 4, 4, 8192);
        let kv_store = KvCacheStore::new(std::time::Duration::from_secs(600));
        make_fake_kv(&kv_store, "m", "req-a", 2, 10);

        let tokens_a: Vec<u32> = (1..=10).collect();
        pc.insert_from_kv("m", "req-a", &kv_store, &tokens_a);
        assert_eq!(pc.entry_count("m"), 3);

        // A different prompt that shares first 6 tokens should hit at 4
        // (longest block boundary that is a prefix).
        let mut tokens_b: Vec<u32> = (1..=6).collect();
        tokens_b.extend_from_slice(&[99, 99, 99, 99]);
        let snap = pc.lookup("m", &tokens_b).expect("hit");
        assert_eq!(snap.token_count, 4);
    }

    #[test]
    fn miss_when_no_shared_prefix() {
        let pc = PrefixCache::new(true, 8, 0, 4, 8192);
        let kv_store = KvCacheStore::new(std::time::Duration::from_secs(600));
        make_fake_kv(&kv_store, "m", "req-a", 2, 10);
        let tokens_a: Vec<u32> = (1..=10).collect();
        pc.insert_from_kv("m", "req-a", &kv_store, &tokens_a);

        let tokens_b: Vec<u32> = vec![99, 98, 97, 96, 95, 94, 93, 92];
        assert!(pc.lookup("m", &tokens_b).is_none());
    }

    #[test]
    fn hydrate_copies_snapshot_into_kv_store() {
        let pc = PrefixCache::new(true, 8, 0, 4, 8192);
        let kv_store = KvCacheStore::new(std::time::Duration::from_secs(600));
        make_fake_kv(&kv_store, "m", "req-a", 2, 10);
        let tokens: Vec<u32> = (1..=10).collect();
        pc.insert_from_kv("m", "req-a", &kv_store, &tokens);
        // lookup requires at least one token left to forward, so query with a
        // longer prompt that still has `tokens` as its strict prefix.
        let lookup_tokens: Vec<u32> = (1..=12).collect();
        let snap = pc.lookup("m", &lookup_tokens).expect("hit");

        let seeded = pc
            .hydrate_request_from_snapshot(&kv_store, "m", "req-b", &snap)
            .unwrap();
        assert_eq!(seeded, 10);

        // req-b now has a KV entry with 10 positions per layer.
        let key = KvCacheStore::cache_key("m", "req-b");
        let entry = kv_store.get_entry(&key).unwrap();
        for l in entry.layers.iter() {
            let kv = l.as_ref().expect("layer populated");
            assert_eq!(kv.current_seq_len(), 10);
        }
    }

    #[test]
    fn block_hash_chain_matches_for_shared_prefix() {
        // Two prompts that share the first two 4-token blocks must have
        // identical block_hash[0..2]. The third block diverges so
        // block_hash[2] must differ.
        let a: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
        let b: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8, 99, 99, 99, 99];
        let ha = compute_block_hashes(&a, 4);
        let hb = compute_block_hashes(&b, 4);
        assert_eq!(ha.len(), 3);
        assert_eq!(hb.len(), 3);
        assert_eq!(ha[0].block_hash, hb[0].block_hash);
        assert_eq!(ha[1].block_hash, hb[1].block_hash);
        assert_ne!(ha[2].block_hash, hb[2].block_hash);
        assert_eq!(ha[0].token_count, 4);
        assert_eq!(ha[1].token_count, 8);
        assert_eq!(ha[2].token_count, 12);
    }

    #[test]
    fn block_hash_chain_handles_partial_trailing_block() {
        // Last 3 tokens are incomplete and must NOT be hashed (block_size=4).
        let toks: Vec<u32> = (0..11).collect();
        let h = compute_block_hashes(&toks, 4);
        assert_eq!(h.len(), 2);
        assert_eq!(h[1].token_count, 8);
    }

    #[test]
    fn block_hash_chain_zero_block_size_is_empty() {
        let h = compute_block_hashes(&[1, 2, 3, 4], 0);
        assert!(h.is_empty());
    }

    #[test]
    fn insert_from_kv_returns_block_manifest() {
        // block=4 → 12 tokens → 3 blocks
        let pc = PrefixCache::new(true, 8, 4, 4, 8192);
        let kv_store = KvCacheStore::new(std::time::Duration::from_secs(600));
        make_fake_kv(&kv_store, "m", "req-a", 2, 12);
        let tokens: Vec<u32> = (1..=12).collect();
        let manifest = pc.insert_from_kv("m", "req-a", &kv_store, &tokens);
        assert_eq!(manifest.len(), 3);
        // token_count grows by block_size (4) each entry.
        assert_eq!(manifest[0].token_count, 4);
        assert_eq!(manifest[1].token_count, 8);
        assert_eq!(manifest[2].token_count, 12);
    }

    #[test]
    fn enumerate_manifest_dedups_across_entries() {
        // Two entries whose tokens share a 4-token prefix. They each insert
        // their own snapshot, but the chained hash for the shared block is
        // identical → manifest must dedup.
        let pc = PrefixCache::new(true, 8, 4, 4, 8192);
        let kv_store = KvCacheStore::new(std::time::Duration::from_secs(600));
        make_fake_kv(&kv_store, "m", "req-a", 2, 8);
        make_fake_kv(&kv_store, "m", "req-b", 2, 8);
        let a: Vec<u32> = vec![1, 2, 3, 4, 50, 51, 52, 53];
        let b: Vec<u32> = vec![1, 2, 3, 4, 60, 61, 62, 63];
        let _ = pc.insert_from_kv("m", "req-a", &kv_store, &a);
        let _ = pc.insert_from_kv("m", "req-b", &kv_store, &b);
        let manifest = pc.enumerate_manifest("m");
        // Block 0 (shared) appears once, plus block 1 from each prompt = 3.
        assert_eq!(manifest.len(), 3);
        // Compare against direct chain hashes to confirm dedup logic.
        let ha = compute_block_hashes(&a, 4);
        let hb = compute_block_hashes(&b, 4);
        assert!(manifest.iter().any(|e| e.block_hash == ha[0].block_hash));
        assert!(manifest.iter().any(|e| e.block_hash == ha[1].block_hash));
        assert!(manifest.iter().any(|e| e.block_hash == hb[1].block_hash));
    }

    #[test]
    fn enumerate_manifest_zero_block_size_is_empty() {
        let pc = PrefixCache::new(true, 8, 0, 4, 8192);
        let kv_store = KvCacheStore::new(std::time::Duration::from_secs(600));
        make_fake_kv(&kv_store, "m", "req-a", 2, 8);
        let _ = pc.insert_from_kv("m", "req-a", &kv_store, &(1..=8).collect::<Vec<_>>());
        // block_tokens=0 → enumerate_manifest is empty even though the
        // entry exists (it was inserted via the always-store-tail rule).
        assert!(pc.enumerate_manifest("m").is_empty());
    }

    #[test]
    fn verify_token_hash_chain_matches_manifest() {
        let tokens: Vec<u32> = (1..=12).collect();
        let manifest = compute_block_hashes(&tokens, 4);
        // Each manifest entry should verify against its own final block.
        for entry in &manifest {
            assert!(verify_token_hash_chain(
                &tokens,
                4,
                entry.token_count as usize,
                &entry.block_hash,
            ));
        }
    }

    #[test]
    fn verify_token_hash_chain_rejects_wrong_tokens() {
        let tokens_a: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let manifest = compute_block_hashes(&tokens_a, 4);
        let last = manifest.last().unwrap();
        // Different tokens must NOT verify against the original hash.
        let tokens_b: Vec<u32> = vec![9, 9, 9, 9, 9, 9, 9, 9];
        assert!(!verify_token_hash_chain(
            &tokens_b,
            4,
            last.token_count as usize,
            &last.block_hash,
        ));
    }

    #[test]
    fn verify_token_hash_chain_rejects_non_boundary() {
        let tokens: Vec<u32> = (1..=8).collect();
        let h = compute_block_hashes(&tokens, 4);
        let last = h.last().unwrap();
        // Off-boundary count (e.g. 5 with block_size=4) must be rejected.
        assert!(!verify_token_hash_chain(&tokens, 4, 5, &last.block_hash));
    }

    #[test]
    fn snapshot_roundtrip_preserves_tensors() {
        use candle_core::{DType, Device};
        let device = Device::Cpu;
        // Construct a tiny snapshot by hand — 2 layers, seq=4, head_dim=3.
        let k0 = Tensor::from_vec(
            (0..24).map(|i| i as f32).collect::<Vec<_>>(),
            (1usize, 2, 4, 3),
            &device,
        )
        .unwrap();
        let v0 = Tensor::from_vec(
            (0..24).map(|i| (i * 2) as f32).collect::<Vec<_>>(),
            (1usize, 2, 4, 3),
            &device,
        )
        .unwrap();
        let k1 = Tensor::from_vec(
            (0..24).map(|i| (i + 100) as f32).collect::<Vec<_>>(),
            (1usize, 2, 4, 3),
            &device,
        )
        .unwrap();
        let v1 = Tensor::from_vec(
            (0..24).map(|i| (i + 200) as f32).collect::<Vec<_>>(),
            (1usize, 2, 4, 3),
            &device,
        )
        .unwrap();
        let snap = KvSnapshot {
            token_count: 4,
            layers: vec![Some((k0, v0)), Some((k1, v1))],
            dim: 2,
            max_seq_len: 4096,
        };
        let tokens: Vec<u32> = vec![7, 8, 9, 10];
        let bytes = serialize_snapshot_with_block_size(&snap, &tokens, None).expect("serialize");
        let (decoded, decoded_tokens) = deserialize_snapshot(&bytes, &device).expect("deserialize");
        assert_eq!(decoded.token_count, 4);
        assert_eq!(decoded.dim, 2);
        assert_eq!(decoded.max_seq_len, 4096);
        assert_eq!(decoded_tokens, tokens);
        assert_eq!(decoded.layers.len(), 2);
        for (orig, got) in snap.layers.iter().zip(decoded.layers.iter()) {
            let (ok, ov) = orig.as_ref().expect("orig some");
            let (dk, dv) = got.as_ref().expect("decoded some");
            assert_eq!(ok.shape().dims(), dk.shape().dims());
            assert_eq!(ov.shape().dims(), dv.shape().dims());
            let ok_vec: Vec<f32> = ok.flatten_all().unwrap().to_vec1().unwrap();
            let dk_vec: Vec<f32> = dk
                .to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1()
                .unwrap();
            assert_eq!(ok_vec, dk_vec);
            let ov_vec: Vec<f32> = ov.flatten_all().unwrap().to_vec1().unwrap();
            let dv_vec: Vec<f32> = dv
                .to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1()
                .unwrap();
            assert_eq!(ov_vec, dv_vec);
        }
    }

    #[test]
    fn snapshot_rejects_bad_magic() {
        use candle_core::Device;
        let bad = vec![0u8; 32];
        assert!(deserialize_snapshot(&bad, &Device::Cpu).is_err());
    }

    #[test]
    fn snapshot_rejects_bad_version() {
        use candle_core::Device;
        let mut buf = Vec::new();
        buf.extend_from_slice(KV_SNAPSHOT_MAGIC);
        buf.extend_from_slice(&999u32.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes());
        assert!(deserialize_snapshot(&buf, &Device::Cpu).is_err());
    }

    #[test]
    fn export_snapshot_bytes_roundtrips_to_hashed_block() {
        // block=4 so a 12-token prompt inserts hashes at 4, 8, 12.
        let pc = PrefixCache::new(true, 8, 4, 4, 8192);
        let kv_store = KvCacheStore::new(std::time::Duration::from_secs(600));
        make_fake_kv(&kv_store, "m", "req-a", 2, 12);
        let tokens: Vec<u32> = (1..=12).collect();
        let manifest = pc.insert_from_kv("m", "req-a", &kv_store, &tokens);
        assert_eq!(manifest.len(), 3);
        // Request the middle block. Serving side should produce bytes
        // whose header `tokens` equal tokens[..8].
        let bytes = pc
            .export_snapshot_bytes("m", &manifest[1].block_hash)
            .expect("hit");
        let (snap, decoded_tokens) =
            deserialize_snapshot(&bytes, &candle_core::Device::Cpu).unwrap();
        assert_eq!(snap.token_count, 8);
        assert_eq!(decoded_tokens, tokens[..8]);
        // Same bytes should BLAKE3-verify against the requested hash.
        assert!(verify_token_hash_chain(
            &decoded_tokens,
            4,
            snap.token_count,
            &manifest[1].block_hash
        ));
    }

    #[test]
    fn snapshot_is_finite_accepts_plain_tensors() {
        use candle_core::Device;
        let device = Device::Cpu;
        let k = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (1usize, 1, 2, 2), &device).unwrap();
        let v = Tensor::from_vec(vec![5.0f32, 6.0, 7.0, 8.0], (1usize, 1, 2, 2), &device).unwrap();
        let snap = KvSnapshot {
            token_count: 2,
            layers: vec![Some((k, v))],
            dim: 2,
            max_seq_len: 128,
        };
        assert!(snapshot_is_finite(&snap));
    }

    #[test]
    fn snapshot_is_finite_rejects_nan() {
        use candle_core::Device;
        let device = Device::Cpu;
        let k =
            Tensor::from_vec(vec![1.0f32, f32::NAN, 3.0, 4.0], (1usize, 1, 2, 2), &device).unwrap();
        let v = Tensor::from_vec(vec![5.0f32, 6.0, 7.0, 8.0], (1usize, 1, 2, 2), &device).unwrap();
        let snap = KvSnapshot {
            token_count: 2,
            layers: vec![Some((k, v))],
            dim: 2,
            max_seq_len: 128,
        };
        assert!(!snapshot_is_finite(&snap));
    }

    #[test]
    fn snapshot_is_finite_rejects_inf() {
        use candle_core::Device;
        let device = Device::Cpu;
        let k = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (1usize, 1, 2, 2), &device).unwrap();
        let v = Tensor::from_vec(
            vec![5.0f32, 6.0, f32::INFINITY, 8.0],
            (1usize, 1, 2, 2),
            &device,
        )
        .unwrap();
        let snap = KvSnapshot {
            token_count: 2,
            layers: vec![Some((k, v))],
            dim: 2,
            max_seq_len: 128,
        };
        assert!(!snapshot_is_finite(&snap));
    }

    #[test]
    fn snapshot_is_finite_ignores_none_layers() {
        let snap = KvSnapshot {
            token_count: 0,
            layers: vec![None, None],
            dim: 2,
            max_seq_len: 128,
        };
        assert!(snapshot_is_finite(&snap));
    }

    #[test]
    fn export_snapshot_bytes_miss_on_unknown_hash() {
        let pc = PrefixCache::new(true, 8, 4, 4, 8192);
        let kv_store = KvCacheStore::new(std::time::Duration::from_secs(600));
        make_fake_kv(&kv_store, "m", "req-a", 2, 8);
        let tokens: Vec<u32> = (1..=8).collect();
        let _ = pc.insert_from_kv("m", "req-a", &kv_store, &tokens);
        // Hash derived from a different prompt should miss.
        let other = compute_block_hashes(&[99, 99, 99, 99, 99, 99, 99, 99], 4);
        assert!(pc
            .export_snapshot_bytes("m", &other[0].block_hash)
            .is_none());
    }

    #[test]
    fn snapshot_preserves_none_layers() {
        use candle_core::Device;
        let device = Device::Cpu;
        let k = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (1usize, 1, 2, 2), &device).unwrap();
        let v = Tensor::from_vec(vec![5.0f32, 6.0, 7.0, 8.0], (1usize, 1, 2, 2), &device).unwrap();
        let snap = KvSnapshot {
            token_count: 2,
            layers: vec![None, Some((k, v)), None],
            dim: 2,
            max_seq_len: 128,
        };
        let tokens: Vec<u32> = vec![11, 12];
        let bytes = serialize_snapshot_with_block_size(&snap, &tokens, None).unwrap();
        let (decoded, _) = deserialize_snapshot(&bytes, &device).unwrap();
        assert_eq!(decoded.layers.len(), 3);
        assert!(decoded.layers[0].is_none());
        assert!(decoded.layers[1].is_some());
        assert!(decoded.layers[2].is_none());
    }

    #[test]
    fn lru_eviction_drops_oldest() {
        let pc = PrefixCache::new(true, 2, 0, 4, 8192);
        let kv_store = KvCacheStore::new(std::time::Duration::from_secs(600));

        for (i, req) in ["r1", "r2", "r3"].iter().enumerate() {
            make_fake_kv(&kv_store, "m", req, 2, 10);
            let tokens: Vec<u32> = (i as u32 * 10..i as u32 * 10 + 10).collect();
            pc.insert_from_kv("m", req, &kv_store, &tokens);
        }

        assert_eq!(pc.entry_count("m"), 2);
    }
}
