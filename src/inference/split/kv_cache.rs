// ── Per-request KV-cache store ──

use candle_core::{DType, Tensor};

use super::SsmState;

/// One sequence-major cache buffer — candle's `Cache`, re-implemented here so
/// that dropping positions off the END is a bookkeeping change.
///
/// **Why not candle's.** `candle_nn::kv_cache::Cache` keeps `current_seq_len`
/// private and exposes only `reset()` and `append()`, so the only way to keep
/// the first `n` positions was to copy them out, reset, and append the copy:
/// two full copies of the retained prefix per layer, for K and for V, with the
/// f16 mirror rebuilt on top — every time a speculative draft was rejected.
/// That is O(context) work on a path that runs once per generated token. At
/// 6.4k positions of llama-3.2-3b it is ~2.2 GB read and ~2.2 GB written per
/// miss and, worse, two 2.2 GB temporaries allocated on a card already holding
/// the cache — measured live as the decode rate collapsing from 33 to 3-5
/// tok/s on a prompt full of tool schemas, which draft on almost every token
/// and miss on almost every draft (gotcha #439).
///
/// The buffer beyond `current_seq_len` is never read: every reader goes
/// through `current_data`, and `append` writes at `current_seq_len`. So a
/// truncation only has to move the length; the stale positions are overwritten
/// by the next append before anything can see them.
///
/// Growth semantics are candle's with one addition: the first append allocates
/// `max_seq_len` positions — the RESERVATION the cache was built with — and
/// each time the cache outgrows its buffer it is extended by `grow_by` with a
/// concatenation. [`KvOccupancy`] and the KV budget reason about that quantum
/// and must keep seeing the same numbers.
///
/// **A prompt of known length is reserved, not grown into.** Growth is a
/// `Tensor::cat`, which copies everything so far, so reaching N positions by
/// growing costs O(N²/quantum) in copies and one fresh device allocation per
/// step — 2279 allocations and 97 GB copied for a 20837-token prompt on a 3B,
/// double that with the f16 mirror, none of it charged by admission, on a card
/// that is by definition at its fullest (FUTURE_WORK #32). vLLM sizes a
/// request's blocks to its whole prompt before prefill runs (Kwon et al. 2023,
/// §4.1); here the worker records the admitted length and the executor builds
/// every layer's cache to it, so the prompt pass is ONE allocation per layer.
///
/// **The buffer may hold another dtype than its sources.** A card keeps its
/// cache as f16 (`KvStorage::F16`) while the projections that produce K and V
/// answer in f32, so every append converts as it writes
/// (`split::kv_write::write_into`); `store_dtype` is what a fresh buffer is
/// allocated as, and `None` means whatever the first source is.
///
/// **A sliding-window layer's cache can drop its oldest positions** (`keep`,
/// FUTURE_WORK #281): when its buffer is full it moves its newest `keep`
/// positions to the front instead of growing, as llama.cpp's windowed cache
/// holds the window plus one batch (`llama_kv_cache_iswa`). `current_seq_len`
/// still counts from the conversation's start — every reader of "how far has
/// this request got" reads it unchanged — and `start` is the position the
/// buffer's first row holds.
#[derive(Debug, Clone)]
pub(crate) struct SeqCache {
    /// `None` until the first append, when the batch/head shape is known.
    all_data: Option<Tensor>,
    dim: usize,
    current_seq_len: usize,
    grow_by: usize,
    /// Positions the BUFFER holds.
    max_seq_len: usize,
    store_dtype: Option<DType>,
    /// The position the buffer's first row holds: 0 until the cache slides.
    start: usize,
    /// How many of its newest positions a full buffer keeps instead of
    /// growing — a windowed layer's ([`positions_kept_for_window`]). `None`
    /// keeps every position.
    keep: Option<usize>,
}

impl SeqCache {
    /// A cache whose FIRST allocation covers `initial` positions and which
    /// grows by `grow_by` after that. candle's `Cache::new(dim, n)` is the
    /// case where the two agree.
    pub(crate) fn with_capacity(dim: usize, initial: usize, grow_by: usize) -> Self {
        Self {
            all_data: None,
            dim,
            current_seq_len: 0,
            grow_by: grow_by.max(1),
            max_seq_len: initial.max(1),
            store_dtype: None,
            start: 0,
            keep: None,
        }
    }

    /// The position the buffer's first row holds — 0 unless the cache has
    /// slid ([`Self::keep`]).
    pub(crate) fn held_from(&self) -> usize {
        self.start
    }

    /// Positions the buffer holds now: `current_seq_len - held_from`.
    pub(crate) fn held(&self) -> usize {
        self.current_seq_len - self.start
    }

    /// The highest position this buffer reaches without sliding or growing —
    /// what a CUDA graph capture and the head-room guard compare a forward
    /// against. `max_seq_len` for a cache that never slid.
    pub(crate) fn reach(&self) -> usize {
        self.start + self.max_seq_len
    }

    /// Keep only the newest `keep` positions when the buffer fills, or every
    /// position (`None`).
    pub(crate) fn slide_beyond(&mut self, keep: Option<usize>) {
        self.keep = keep;
    }

    /// Start an EMPTY cache at `position`: its first append is that position.
    /// How a snapshot of a slid cache is restored. Ignored once anything is
    /// held.
    pub(crate) fn begin_at(&mut self, position: usize) {
        if self.current_seq_len == self.start {
            self.start = position;
            self.current_seq_len = position;
        }
    }

    /// Make the first allocation no larger than `positions`. Ignored once the
    /// buffer exists.
    fn cap_first_allocation(&mut self, positions: usize) {
        if self.all_data.is_none() {
            self.max_seq_len = self.max_seq_len.min(positions).max(1);
        }
    }

    /// The dtype the buffer holds — what it was allocated as, or what the next
    /// allocation will be; `None` before the first append of a cache that
    /// follows its source.
    pub(crate) fn dtype(&self) -> Option<DType> {
        self.all_data
            .as_ref()
            .map(Tensor::dtype)
            .or(self.store_dtype)
    }

    pub(crate) fn dim(&self) -> usize {
        self.dim
    }

    pub(crate) fn current_seq_len(&self) -> usize {
        self.current_seq_len
    }

    pub(crate) fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    pub(crate) fn all_data(&self) -> &Option<Tensor> {
        &self.all_data
    }

    /// The positions held, as a view over the buffer: from [`Self::held_from`]
    /// to the current length.
    pub(crate) fn current_data(&self) -> candle_core::Result<Option<Tensor>> {
        match self.all_data.as_ref() {
            None => Ok(None),
            Some(d) => Ok(Some(d.narrow(self.dim, 0, self.held())?)),
        }
    }

    pub(crate) fn reset(&mut self) {
        self.current_seq_len = 0;
        self.start = 0;
        self.all_data = None;
    }

    /// An independent copy — see [`KvPair::deep_copy`].
    #[cfg(test)]
    pub(crate) fn deep_copy(&self) -> candle_core::Result<Self> {
        Ok(Self {
            all_data: self.all_data.as_ref().map(|t| t.copy()).transpose()?,
            dim: self.dim,
            current_seq_len: self.current_seq_len,
            grow_by: self.grow_by,
            max_seq_len: self.max_seq_len,
            store_dtype: self.store_dtype,
            start: self.start,
            keep: self.keep,
        })
    }

    /// Keep the first `len` positions and forget the rest. O(1): no copy, no
    /// allocation, the buffer is untouched. A `len` at or past the current
    /// length is a no-op. Never below [`Self::held_from`]: positions a slid
    /// cache no longer holds cannot be handed back, and
    /// `KvCacheEntry::truncate_to` refuses before asking (`LayerKv::can_truncate_to`).
    pub(crate) fn truncate(&mut self, len: usize) {
        self.current_seq_len = self.current_seq_len.min(len).max(self.start);
    }

    pub(crate) fn append(&mut self, src: &Tensor) -> candle_core::Result<()> {
        let seq_len = src.dim(self.dim)?;
        let dtype = self.dtype().unwrap_or(src.dtype());
        if self.all_data.is_none() {
            let mut shape = src.dims().to_vec();
            shape[self.dim] = self.max_seq_len;
            self.all_data = Some(Tensor::zeros(shape, dtype, src.device())?);
        }
        let ad = self
            .all_data
            .as_mut()
            .expect("all_data was just initialised");
        // A windowed layer's full buffer moves its newest `keep` positions to
        // the front rather than growing. The copy is made first: one op
        // reading and writing the same buffer would hold its storage twice.
        // Amortised, a position is copied `keep / grow_by` times — ~2 for a
        // 1,024 window — against the window's worth of keys every query reads.
        if let Some(keep) = self.keep {
            let held = self.current_seq_len - self.start;
            if held + seq_len > self.max_seq_len && held > keep {
                let newest = ad.narrow(self.dim, held - keep, keep)?.force_contiguous()?;
                super::kv_write::write_into(ad, &newest, self.dim, 0)?;
                self.start = self.current_seq_len - keep;
                KV_SLIDES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
        }
        let needed = self.current_seq_len - self.start + seq_len;
        while needed > self.max_seq_len {
            let step = growth_step(self.keep, self.grow_by, self.max_seq_len, needed);
            let mut shape = src.dims().to_vec();
            shape[self.dim] = step;
            let next_ad = Tensor::zeros(shape, dtype, src.device())?;
            *ad = Tensor::cat(&[&*ad, &next_ad], self.dim)?;
            self.max_seq_len += step;
            KV_GROWTH_STEPS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        // `slice_set` refuses a non-contiguous source, and a cache is in no
        // position to decline one: the alternative to copying is failing every
        // request on that model. Free when the tensor is already contiguous,
        // which is the normal case. Partial RoPE handed a view here on every
        // Phi-4-mini request ever made — `layers::rope_over_heads` is where
        // that is prevented now, and this is why a new producer of K or V
        // cannot bring the whole class back.
        //
        // Converting as it writes when the buffer keeps another dtype — the
        // half cache, fed f32 — in one launch on a card (`kv_write`).
        let src = src.contiguous()?;
        super::kv_write::write_into(ad, &src, self.dim, self.current_seq_len - self.start)?;
        self.current_seq_len += seq_len;
        Ok(())
    }

    /// Positions an append of `seq_len` at `index_pos` allocates — what
    /// [`Self::append`] grows the buffer by, without growing it: 0 when the
    /// forward fits, or fits once the cache slides; the whole first buffer
    /// when nothing is allocated yet. `index_pos == 0` starts the cache over
    /// (`layers::append_to_cache` resets it), keeping its size.
    pub(crate) fn positions_added_by(&self, index_pos: usize, seq_len: usize) -> usize {
        let held = if index_pos == 0 {
            0
        } else {
            self.current_seq_len - self.start
        };
        self.positions_added_holding(held, seq_len)
    }

    /// [`Self::positions_added_by`] for a cache that holds `held` positions
    /// when the `seq_len` arrive — a mirror is caught up to its f32 cache
    /// first ([`LayerKv::append`]), whatever it held before.
    fn positions_added_holding(&self, held: usize, seq_len: usize) -> usize {
        let needed = match self.keep {
            Some(keep) if held + seq_len > self.max_seq_len && held > keep => keep + seq_len,
            _ => held + seq_len,
        };
        let mut buffer = self.max_seq_len;
        while needed > buffer {
            buffer += growth_step(self.keep, self.grow_by, buffer, needed);
        }
        let first = if self.all_data.is_none() {
            self.max_seq_len
        } else {
            0
        };
        first + buffer - self.max_seq_len
    }
}

/// How far a [`SeqCache`] buffer of `buffer` positions grows in one step
/// toward holding `needed`. A cache that keeps every position (`keep: None`)
/// grows by whole quanta (`grow_by`). A windowed cache grows no further than
/// its `keep` and one quantum — the size whose slide always leaves a whole
/// quantum free, or it settles a quantum wider than it ever reads — and past
/// that, for a forward wider than a quantum, to exactly what the forward
/// needs: never wider than a cache that keeps every position, and never past
/// [`most_positions_a_window_holds`], which every estimate of its cost
/// charges. One rule for the growth and for the guard's reading of it
/// (`SeqCache::positions_added_by`).
fn growth_step(keep: Option<usize>, grow_by: usize, buffer: usize, needed: usize) -> usize {
    match keep {
        Some(keep) => grow_by.min((keep + grow_by).max(needed) - buffer),
        None => grow_by,
    }
}

/// Growth steps taken by every [`SeqCache`] in this process — one per
/// `Tensor::cat`. A reserved prompt pass leaves it unchanged; that is the
/// number the executor's prompt-pass DIAG reports and the mechanism check
/// for FUTURE_WORK #32 reads.
pub(crate) static KV_GROWTH_STEPS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// Slides taken by every windowed [`SeqCache`] in this process — the
/// mechanism check for FUTURE_WORK #281: a long conversation on a windowed
/// model that leaves it at 0 kept every position.
pub(crate) static KV_SLIDES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// See [`KV_SLIDES`].
pub(crate) fn kv_slides() -> u64 {
    KV_SLIDES.load(std::sync::atomic::Ordering::Relaxed)
}

/// Positions past a layer's window its cache keeps behind the next write, for
/// a rewind: a rejected draft (`cuda_graph::MAX_POSITIONS`, SWIFT's drafts) or a
/// split's stored prompt, cut at a whole 64-token block
/// (`split_prompt_cache::BLOCK_TOKENS`). A rewind past it is refused, never
/// served from keys the window lost ([`LayerKv::can_truncate_to`]).
pub(crate) const WINDOW_REWIND_POSITIONS: usize = 64;

/// How many of its newest positions a full sliding-window layer's cache keeps:
/// the `window - 1` keys before the next query, and room to rewind
/// ([`WINDOW_REWIND_POSITIONS`]). llama.cpp keeps `n_swa + n_ubatch`
/// (`llama_kv_cache_iswa`); the batch here is the room the buffer has past
/// this (`layers::new_kv_cache`).
pub(crate) fn positions_kept_for_window(window: usize) -> usize {
    window.saturating_sub(1) + WINDOW_REWIND_POSITIONS
}

/// The most positions a sliding-window layer's cache is ever allocated:
/// [`positions_kept_for_window`] and the widest forward it is handed, at least
/// one growth `quantum` (`SeqCache::growth_step`). llama.cpp sizes its windowed
/// cache the same way, `n_swa + n_ubatch` (`llama_kv_cache_iswa`). What every
/// estimate of a windowed cache's cost charges it (`kv_budget::KvGrowth`).
pub(crate) fn most_positions_a_window_holds(
    window: usize,
    quantum: usize,
    widest_forward: usize,
) -> usize {
    positions_kept_for_window(window) + quantum.max(widest_forward).max(1)
}

/// The window a layer's cache keeps: the layer's own `window`, where a cache
/// stored as `storage` slides — never one kept with the f16 mirror
/// ([`LayerKv::slide_beyond`]) — else `None`. The ONE answer for the cache, its
/// restoring from a snapshot (`PrefixCache::hydrate_request_from_snapshot`) and
/// every estimate of what it costs (`kv_budget::KvGrowth`).
pub(crate) fn window_kept(window: Option<usize>, storage: KvStorage) -> Option<usize> {
    window.filter(|_| caches_slide(storage == KvStorage::F32Mirrored))
}

/// [`window_kept`] on a node whose windowed caches slide where `slides` says —
/// a PEER's, from what it advertises (`features::KV_WINDOW_SLIDES`); this
/// node's own is [`window_kept`]. A cache kept with the f16 mirror never
/// slides on any node.
pub(crate) fn window_kept_where(
    window: Option<usize>,
    storage: KvStorage,
    slides: bool,
) -> Option<usize> {
    window.filter(|_| slides && storage != KvStorage::F32Mirrored)
}

/// How one layer keeps its cache: its storage (`layers::kv_storage`) and the
/// window it keeps ([`window_kept`]) — what a forward builds it as
/// (`layers::append_to_cache`), and what a cache restored from a snapshot is
/// built as (`PrefixCache::hydrate_request_from_snapshot`), which the snapshot
/// cannot carry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CacheLayout {
    pub(crate) storage: KvStorage,
    pub(crate) window: Option<usize>,
}

/// Does a windowed cache slide — [`windowed_caches_slide`], and not one kept
/// with the f16 mirror, which is kept in step by absolute positions against the
/// whole history?
fn caches_slide(mirrored: bool) -> bool {
    windowed_caches_slide() && !mirrored
}

/// `SWARMLLM_KV_WINDOW_SLIDE=0` keeps every position of a windowed layer, as
/// before FUTURE_WORK #281 — the A/B inside one binary. Read once.
pub(crate) fn windowed_caches_slide() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    #[cfg(test)]
    if NO_SLIDE_IN_TEST.with(std::cell::Cell::get) {
        return false;
    }
    *ON.get_or_init(|| std::env::var("SWARMLLM_KV_WINDOW_SLIDE").as_deref() != Ok("0"))
}

#[cfg(test)]
thread_local! {
    /// A test's own `SWARMLLM_KV_WINDOW_SLIDE=0`, on its thread only.
    pub(crate) static NO_SLIDE_IN_TEST: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// See [`KV_GROWTH_STEPS`].
pub(crate) fn kv_growth_steps() -> u64 {
    KV_GROWTH_STEPS.load(std::sync::atomic::Ordering::Relaxed)
}

/// A K cache and a V cache that move together — candle's `KvCache` over
/// [`SeqCache`], same method names so the call sites did not change.
#[derive(Debug, Clone)]
pub(crate) struct KvPair {
    k: SeqCache,
    v: SeqCache,
}

impl KvPair {
    /// See [`SeqCache::with_capacity`].
    pub(crate) fn with_capacity(dim: usize, initial: usize, grow_by: usize) -> Self {
        Self {
            k: SeqCache::with_capacity(dim, initial, grow_by),
            v: SeqCache::with_capacity(dim, initial, grow_by),
        }
    }

    /// The same, holding `dtype` whatever its sources are (`None`: the first
    /// source's) — see [`SeqCache`].
    fn with_capacity_in(dim: usize, initial: usize, grow_by: usize, dtype: Option<DType>) -> Self {
        let mut pair = Self::with_capacity(dim, initial, grow_by);
        pair.k.store_dtype = dtype;
        pair.v.store_dtype = dtype;
        pair
    }

    /// An INDEPENDENT copy (probe/test use). `Clone` shares the tensors'
    /// storage and the next `append` writes into that buffer in place, so a
    /// cloned cache handed to a second model is silently rewritten by it.
    #[cfg(test)]
    pub(crate) fn deep_copy(&self) -> candle_core::Result<Self> {
        Ok(Self {
            k: self.k.deep_copy()?,
            v: self.v.deep_copy()?,
        })
    }

    pub(crate) fn k_cache(&self) -> &SeqCache {
        &self.k
    }

    pub(crate) fn v_cache(&self) -> &SeqCache {
        &self.v
    }

    pub(crate) fn k(&self) -> candle_core::Result<Option<Tensor>> {
        self.k.current_data()
    }

    pub(crate) fn v(&self) -> candle_core::Result<Option<Tensor>> {
        self.v.current_data()
    }

    /// Append new positions and return the full (K, V) written so far. An
    /// empty cache answers with zero-length tensors rather than `None`, as
    /// candle's does, so attention code never has to special-case it.
    pub(crate) fn append(
        &mut self,
        k: &Tensor,
        v: &Tensor,
    ) -> candle_core::Result<(Tensor, Tensor)> {
        self.k.append(k)?;
        self.v.append(v)?;
        let out_k = match self.k.current_data()? {
            Some(t) => t,
            None => {
                let mut shape = k.dims().to_vec();
                shape[self.k.dim] = 0;
                Tensor::zeros(shape, k.dtype(), k.device())?
            }
        };
        let out_v = match self.v.current_data()? {
            Some(t) => t,
            None => {
                let mut shape = v.dims().to_vec();
                shape[self.v.dim] = 0;
                Tensor::zeros(shape, v.dtype(), v.device())?
            }
        };
        Ok((out_k, out_v))
    }

    pub(crate) fn current_seq_len(&self) -> usize {
        self.k.current_seq_len()
    }

    pub(crate) fn reset(&mut self) {
        self.k.reset();
        self.v.reset();
    }

    /// Keep the first `len` positions of both. See [`SeqCache::truncate`].
    pub(crate) fn truncate(&mut self, len: usize) {
        self.k.truncate(len);
        self.v.truncate(len);
    }

    /// See [`SeqCache::slide_beyond`].
    fn slide_beyond(&mut self, keep: Option<usize>) {
        self.k.slide_beyond(keep);
        self.v.slide_beyond(keep);
    }

    /// See [`SeqCache::begin_at`].
    fn begin_at(&mut self, position: usize) {
        self.k.begin_at(position);
        self.v.begin_at(position);
    }

    /// See [`SeqCache::cap_first_allocation`].
    fn cap_first_allocation(&mut self, positions: usize) {
        self.k.cap_first_allocation(positions);
        self.v.cap_first_allocation(positions);
    }
}

/// How one layer keeps its KV cache — `layers::kv_storage` is the one place
/// that decides, for the cache as it is built and for every estimate of what it
/// will cost (FUTURE_WORK #194).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum KvStorage {
    /// f32 BHSD — the processor; MLA; a card where the half cache does not apply
    /// and the model is not grouped-query.
    F32,
    /// f32 BHSD plus the f16 BSHD flash mirror — a grouped-query model on a card
    /// where the half cache does not apply (the layout before it).
    F32Mirrored,
    /// f16 BHSD and nothing else — a card, read in place by the flash kernel
    /// and the card's decode kernel. What llama.cpp stores by default.
    F16,
}

impl KvStorage {
    /// Bytes ONE cached element costs, every copy counted: 4 for f32, 6 with the
    /// f16 mirror beside it, 2 for the half cache.
    pub(crate) fn bytes_per_element(self) -> u64 {
        match self {
            Self::F32 => 4,
            Self::F32Mirrored => 6,
            Self::F16 => 2,
        }
    }
}

/// One layer's KV cache: on a card, the f16 BHSD half cache every path reads
/// in place ([`KvStorage::F16`]); otherwise the f32 BHSD cache every path
/// reads, plus an optional f16 BSHD mirror kept for the CUDA flash-attention
/// kernel. Most of what follows is about the second.
///
/// **Why the mirror exists.** `run_attention`'s CUDA arm needs f16 in BSHD while
/// the cache is f32 in BHSD, so it used to transpose and convert the WHOLE
/// history on every token — O(history) work to add ONE position, and the
/// dominant cost of long-context decode. The mirror is appended one position at
/// a time instead, so the per-token cost stops growing with the conversation.
/// Measured on an RTX 3070, attention arm vs the mirrored ceiling:
/// 272 KV 1.6x, 528 KV 1.75x, 912 KV 1.86x.
///
/// **Why it was a mirror and not a replacement (2026-08-10), and why on a card
/// it is now the replacement (2026-10-09).** Rounding f32 to f16 is done at
/// WRITE time here instead of at read time, and since the f32 source is never
/// itself overwritten by a rounded value, the kernel receives bitwise the same
/// numbers it received before — the flash path is numerically unchanged, not
/// merely close. The f32 copy was kept for `standard_attention`, on a reading
/// of arXiv 2604.15409 as "f16 KV diverges". That paper compares cache-ON with
/// cache-OFF decoding, BOTH in f16 — a different accumulation order, not a
/// different storage width — and the reference every reply here is judged
/// against, llama.cpp, stores f16 by default. Meanwhile the f32 copy plus the
/// mirror cost 3x llama.cpp's memory per cached token, which is what filled an
/// 8 GB card with an agent's prompt (#194). So a card now stores only f16
/// where its decode kernel and flash read it in place, and keeps this layout
/// where they do not (`layers::kv_storage`).
///
/// **Keeping the two in step is the whole risk**, because a mirror that has
/// drifted produces plausible wrong attention rather than an error. Everything
/// that mutates the cache therefore goes through `append` / `reset` here, which
/// are inherent methods and so take priority over the `Deref` to `KvCache` —
/// existing call sites cannot reach the inner cache's versions by accident.
/// `truncate` is the third such method: it moves both lengths together, in
/// place, and is what `KvCacheStore::truncate_to` calls on a rejected draft.
pub(crate) struct LayerKv {
    /// BHSD, sequence on dim 2. The source of truth: f32, or f16 for the half
    /// cache ([`KvStorage::F16`]).
    main: KvPair,
    /// f16 BSHD, sequence on dim 1. `None` on CPU and until the first append,
    /// which is where the device becomes known.
    shadow: Option<KvPair>,
    /// Growth quantum, shared with the mirror.
    growth: usize,
    /// Positions the first allocation covers — the admitted prompt, or one
    /// quantum when nothing was reserved. Shared with the mirror too: it used
    /// to be born one quantum wide beside a reserved f32 cache and then grow
    /// by concatenation, doubling every copy the reservation had just removed.
    initial: usize,
    /// False when `main`'s sequence axis is not dim 2, which the mirror's
    /// transpose assumes. Such a cache never gets a mirror.
    mirrorable: bool,
    /// The layer's sliding window, when its cache keeps only that much
    /// ([`Self::slide_beyond`]); `None` for a layer that reads its whole
    /// history.
    window: Option<usize>,
}

impl LayerKv {
    /// An independent copy of this layer's cache, mirror included — see
    /// [`KvPair::deep_copy`] for why `Clone` is not one.
    #[cfg(test)]
    pub(crate) fn deep_copy(&self) -> candle_core::Result<Self> {
        Ok(Self {
            main: self.main.deep_copy()?,
            shadow: self.shadow.as_ref().map(KvPair::deep_copy).transpose()?,
            growth: self.growth,
            initial: self.initial,
            mirrorable: self.mirrorable,
            window: self.window,
        })
    }

    /// Let a sliding-window layer's cache keep only what its queries can
    /// still read ([`positions_kept_for_window`], FUTURE_WORK #281), or every
    /// position (`None`). Asked by every forward (`layers::append_to_cache`),
    /// as [`Self::conform_to`] is: a cache restored from a snapshot cannot know
    /// its layer's window.
    ///
    /// **A cache with an f16 mirror never slides**: the mirror is kept in step
    /// by absolute positions against the cache's whole history. That layout is
    /// a card where the half cache does not apply ([`KvStorage::F32Mirrored`]);
    /// the processor and the half cache slide.
    pub(crate) fn slide_beyond(&mut self, window: Option<usize>) {
        let window = window.filter(|_| caches_slide(self.mirrorable));
        self.window = window;
        let keep = window.map(positions_kept_for_window);
        self.main.slide_beyond(keep);
        // Not yet allocated, a sliding cache is born the window and one growth
        // quantum wide — never the whole admitted prompt, which is where a long
        // prompt's memory goes.
        if let Some(keep) = keep {
            let born = keep + self.growth;
            self.main.cap_first_allocation(born);
            self.initial = self.initial.min(born);
        }
    }

    /// The first position this cache still holds: 0 unless it slid.
    pub(crate) fn held_from(&self) -> usize {
        self.main.k_cache().held_from()
    }

    /// The shortest history this cache can be cut back to and still answer
    /// the next query: one window past [`Self::held_from`] (vLLM's rule for a
    /// sliding-window prefix hit, `SlidingWindowManager`), 0 for a cache that
    /// holds its whole history. A rewind or a snapshot shorter than this
    /// would read keys the window has lost.
    pub(crate) fn serves_from(&self) -> usize {
        match (self.held_from(), self.window) {
            (0, _) => 0,
            (from, Some(window)) => from + window - 1,
            // Slid without a window: only a snapshot restored onto a cache
            // built without one — its own start is all it can promise.
            (from, None) => from,
        }
    }

    /// Can this cache be cut back to `len` positions? Always for a cache that
    /// holds its whole history, or a `len` it has not passed.
    pub(crate) fn can_truncate_to(&self, len: usize) -> bool {
        len >= self.current_seq_len() || len >= self.serves_from()
    }

    /// Start an EMPTY cache at `position` — restoring a snapshot of a slid
    /// cache ([`SeqCache::begin_at`]).
    pub(crate) fn begin_at(&mut self, position: usize) {
        self.main.begin_at(position);
    }

    /// Build with the first allocation sized to `initial` positions and
    /// growth of `growth` after that — how a prompt of known length is held in
    /// one allocation per layer. See [`SeqCache`]. `dim` is the sequence axis;
    /// a mirror is only maintained when it is dim 2 (BHSD), since
    /// `to_bshd_f16`'s `transpose(1, 2)` is meaningless otherwise — any other
    /// layout simply gets no mirror and the original conversion path.
    pub(crate) fn with_capacity(dim: usize, initial: usize, growth: usize) -> Self {
        Self {
            main: KvPair::with_capacity(dim, initial, growth),
            shadow: None,
            growth,
            initial,
            mirrorable: dim == 2,
            window: None,
        }
    }

    /// The same, stored the way `storage` says from the first append
    /// (`layers::new_kv_cache`).
    pub(crate) fn with_storage(
        dim: usize,
        initial: usize,
        growth: usize,
        storage: KvStorage,
    ) -> Self {
        let (dtype, mirrorable) = match storage {
            KvStorage::F16 => (Some(DType::F16), false),
            KvStorage::F32 => (Some(DType::F32), false),
            KvStorage::F32Mirrored => (Some(DType::F32), dim == 2),
        };
        Self {
            main: KvPair::with_capacity_in(dim, initial, growth, dtype),
            shadow: None,
            growth,
            initial,
            mirrorable,
            window: None,
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

    /// Does this cache hold — or, before its first append, will it hold — the
    /// half cache?
    pub(crate) fn holds_half(&self) -> bool {
        self.main.k_cache().dtype() == Some(DType::F16)
    }

    /// Make this cache the kind `storage` says, converting what it already
    /// holds if it is not.
    ///
    /// Every forward asks it (`layers::append_to_cache`), because a cache can
    /// arrive built by something that could not know: prefix-cache hydration
    /// makes one from a snapshot, which carries `n_kv_head` in its shape but
    /// not `n_head`, nor the device's choice of storage. Free when the cache is
    /// already right — the common case, every step after the first. Converting
    /// a held history is O(history), once: a snapshot taken from an f32 cache
    /// on another build, read into a half one, or the reverse.
    ///
    /// **A mirror is only ever turned OFF here, deliberately.** Turning one ON
    /// mid-conversation would start it empty against a cache already holding N
    /// positions; a cache that should mirror is built that way from the start
    /// (`new_kv_cache`) or filled in one append (prefix-cache hydration, which
    /// mirrors the whole snapshot correctly because it appends it at once).
    pub(crate) fn conform_to(&mut self, storage: KvStorage) -> candle_core::Result<()> {
        match storage {
            KvStorage::F16 => {
                self.mirrorable = false;
                self.shadow = None;
                self.convert_main(DType::F16)
            }
            KvStorage::F32 => {
                self.mirrorable = false;
                self.shadow = None;
                self.convert_main(DType::F32)
            }
            KvStorage::F32Mirrored => self.convert_main(DType::F32),
        }
    }

    /// Hold `dtype` from now on, converting the positions already held.
    fn convert_main(&mut self, dtype: DType) -> candle_core::Result<()> {
        let dim = self.main.k_cache().dim();
        let capacity = self.main.k_cache().max_seq_len();
        if self.main.k_cache().all_data().is_none() {
            // Nothing held: only what the first allocation will be changes.
            self.main = KvPair::with_capacity_in(dim, capacity, self.growth, Some(dtype));
            return Ok(());
        }
        if self.main.k_cache().dtype() == Some(dtype) {
            self.main.k.store_dtype = Some(dtype);
            self.main.v.store_dtype = Some(dtype);
            return Ok(());
        }
        let (k, v) = (self.main.k()?, self.main.v()?);
        let mut converted = KvPair::with_capacity_in(dim, capacity, self.growth, Some(dtype));
        // A slid cache stays slid: what it holds starts where it did.
        converted.begin_at(self.held_from());
        converted.slide_beyond(self.window.map(positions_kept_for_window));
        if let (Some(k), Some(v)) = (k, v) {
            if k.dim(dim)? > 0 {
                converted.append(&k, &v)?;
            }
        }
        self.main = converted;
        Ok(())
    }

    /// Append new positions, updating the mirror in the same call.
    ///
    /// Returns the full f32 BHSD (K, V) exactly as `KvPair::append` does, so
    /// callers are unchanged.
    pub(crate) fn append(
        &mut self,
        k: &Tensor,
        v: &Tensor,
    ) -> candle_core::Result<(Tensor, Tensor)> {
        let before = self.main.current_seq_len();
        let out = self.main.append(k, v)?;
        // The half cache IS what flash reads; there is nothing to keep in step.
        if self.holds_half() {
            return Ok(out);
        }
        // A decode step — ONE position — leaves the mirror behind. Its reader
        // is attention over the f32 cache (`layers::cuda_decode_prefers_standard`
        // and the card's decode kernel), so keeping the mirror in step cost four
        // launches per layer per token (two casts, two copies: 15% of a 7B's
        // decode launches) for a copy nothing read until the next prompt chunk.
        // That chunk catches it up below, in one conversion.
        if self.main.current_seq_len() == before + 1 && mirror_is_lazy() {
            return Ok(out);
        }
        // Never for a cache that has dropped positions: the mirror is kept in
        // step by absolute positions against the whole history.
        if self.shadow.is_none()
            && self.mirrorable
            && self.held_from() == 0
            && Self::wants_shadow(k)
        {
            self.shadow = Some(KvPair::with_capacity(1, self.initial, self.growth));
        }
        if let Some(shadow) = self.shadow.as_mut() {
            let have = shadow.current_seq_len();
            let caught_up = (|| -> candle_core::Result<()> {
                // The positions the f32 cache took without it: the decode
                // steps since the last chunk, or a whole history when the
                // mirror starts mid-conversation.
                if have < before {
                    let k_lag = out.0.narrow(2, have, before - have)?;
                    let v_lag = out.1.narrow(2, have, before - have)?;
                    shadow.append(&Self::to_bshd_f16(&k_lag)?, &Self::to_bshd_f16(&v_lag)?)?;
                } else if have > before {
                    shadow.truncate(before);
                }
                // Then only what is being added — this is the point of the mirror.
                shadow
                    .append(&Self::to_bshd_f16(k)?, &Self::to_bshd_f16(v)?)
                    .map(|_| ())
            })();
            if let Err(e) = caught_up {
                // Never serve from a mirror that failed to take an append: drop
                // it and let the flash path fall back to converting the f32
                // cache. Wrong attention is far worse than losing the speedup.
                tracing::warn!("KV mirror append failed, falling back to conversion: {e}");
                self.shadow = None;
            }
        }
        Ok(out)
    }

    /// Bring a lagging mirror level with the f32 cache — the catch-up the next
    /// multi-position `append` would do first, done NOW, outside any capture.
    ///
    /// `SplitModel::decode_step_with_graph` calls it before RECORDING a
    /// multi-position step. Recorded, the catch-up's copies run only when the
    /// graph is launched, while the mirror's length moves at once; and a
    /// catch-up is exactly what gets a capture refused — its slice of the lag is
    /// a shape seen for the first time, whose layout candle uploads from the
    /// host. Refused, the step is rolled back to its own start (`index_pos`),
    /// which leaves the mirror claiming the lagging positions it never
    /// received, and flash attention then reads them: plausible text, drifting
    /// from the model's own (a split's streamed checks picked llama.cpp's 2nd
    /// choice at one sentence in 6 of 11 replies, never with graphs off —
    /// 2026-09-30). Caught up here, a capture only appends what it adds, and a
    /// rollback to `index_pos` is exact.
    pub(crate) fn catch_up_mirror(&mut self) {
        let Some(shadow) = self.shadow.as_mut() else {
            return;
        };
        let have = shadow.current_seq_len();
        let len = self.main.current_seq_len();
        if have >= len {
            return;
        }
        let main = &self.main;
        let caught_up = (|| -> candle_core::Result<()> {
            let (Some(k), Some(v)) = (main.k()?, main.v()?) else {
                return Ok(());
            };
            let k_lag = k.narrow(2, have, len - have)?;
            let v_lag = v.narrow(2, have, len - have)?;
            shadow
                .append(&Self::to_bshd_f16(&k_lag)?, &Self::to_bshd_f16(&v_lag)?)
                .map(|_| ())
        })();
        if let Err(e) = caught_up {
            // As in `append`: never serve from a mirror that failed to take
            // one; the flash path converts the f32 cache instead.
            tracing::warn!("KV mirror catch-up failed, falling back to conversion: {e}");
            self.shadow = None;
        }
    }

    /// Reset both representations together.
    pub(crate) fn reset(&mut self) {
        self.main.reset();
        if let Some(shadow) = self.shadow.as_mut() {
            shadow.reset();
        }
    }

    /// Keep the first `len` positions of both representations, in place.
    ///
    /// This is what a rejected speculative draft costs now: two length fields
    /// move. It used to be a snapshot, a reset and a re-append of the kept
    /// prefix — see [`SeqCache`] for what that cost and where it was measured.
    pub(crate) fn truncate(&mut self, len: usize) {
        self.main.truncate(len);
        if let Some(shadow) = self.shadow.as_mut() {
            shadow.truncate(len);
        }
    }

    /// Build the mirror unconditionally, so its bookkeeping can be tested on a
    /// machine with no GPU. Production creation still goes through the device
    /// check in `append`.
    #[cfg(test)]
    pub(crate) fn force_shadow_for_test(&mut self) {
        if self.mirrorable {
            self.shadow = Some(KvPair::with_capacity(1, self.initial, self.growth));
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
    pub(crate) fn all_caches(&self) -> Vec<&SeqCache> {
        let mut out = vec![self.main.k_cache(), self.main.v_cache()];
        if let Some(shadow) = self.shadow.as_ref() {
            out.push(shadow.k_cache());
            out.push(shadow.v_cache());
        }
        out
    }

    /// Bytes an append of `seq_len` at `index_pos` newly allocates, read off
    /// the buffers (`SeqCache::positions_added_by`) at the dtype they hold,
    /// and the memory it lands in; `None` before the first append. The f16
    /// mirror is priced as [`Self::append`] treats it, never by the layer's
    /// storage: a decode step leaves it behind, and it is built only where
    /// [`Self::wants_shadow`] says so.
    fn bytes_added_by(&self, index_pos: usize, seq_len: usize) -> Option<(u64, KvHome)> {
        let (k, v) = (self.main.k_cache(), self.main.v_cache());
        let (Some(k_data), Some(v_data)) = (k.all_data(), v.all_data()) else {
            return None;
        };
        let per_position = |cache: &SeqCache, data: &Tensor| {
            (data.elem_count() / cache.max_seq_len().max(1)) as u64
        };
        let elems = per_position(k, k_data) + per_position(v, v_data);
        let added = k.positions_added_by(index_pos, seq_len) as u64;
        let mirrored = self.mirror_positions_added_by(index_pos, seq_len, k_data) as u64;
        let bytes = added
            .saturating_mul(elems)
            .saturating_mul(k_data.dtype().size_in_bytes() as u64)
            .saturating_add(
                mirrored
                    .saturating_mul(elems)
                    .saturating_mul(DType::F16.size_in_bytes() as u64),
            );
        Some((bytes, KvHome::of(k_data.device())))
    }

    /// Positions the f16 mirror allocates for an append of `seq_len` at
    /// `index_pos` — step for step what [`Self::append`] does to it: nothing
    /// for a half cache or a lazy decode step; else the mirror, caught up to
    /// the f32 cache, takes the new positions — built first, `initial` wide,
    /// where a cache that never slid [`Self::wants_shadow`].
    fn mirror_positions_added_by(&self, index_pos: usize, seq_len: usize, k: &Tensor) -> usize {
        if self.holds_half() || (seq_len == 1 && mirror_is_lazy()) {
            return 0;
        }
        let held = if index_pos == 0 {
            0
        } else {
            self.main.current_seq_len()
        };
        match self.shadow.as_ref() {
            Some(shadow) => shadow.k_cache().positions_added_holding(held, seq_len),
            None if self.mirrorable && self.held_from() == 0 && Self::wants_shadow(k) => {
                SeqCache::with_capacity(1, self.initial, self.growth)
                    .positions_added_holding(held, seq_len)
            }
            None => 0,
        }
    }

    /// The mirrored (K, V) for the flash kernel: f16, BSHD, full length.
    ///
    /// `None` means no mirror — the caller converts the f32 cache as before.
    /// The length check is the guard against a silently drifted mirror: if it
    /// ever disagrees with the real cache, this declines rather than handing the
    /// kernel a history of the wrong length.
    pub(crate) fn flash_operands(&self) -> Option<(Tensor, Tensor)> {
        let shadow = self.shadow.as_ref()?;
        // BEHIND is by design: decode steps leave it there (`append`), and the
        // next multi-position append catches it up before anything reads it.
        // Every decode step asks, so this must stay quiet.
        if shadow.current_seq_len() < self.main.current_seq_len() && mirror_is_lazy() {
            return None;
        }
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

/// Whether a one-position append leaves the f16 mirror behind (the default).
///
/// `SWARMLLM_KV_MIRROR_EAGER=1` keeps it in step on every append, as before —
/// the A/B inside one binary. `SWARMLLM_GQA_DECODE_FLASH=1` does too: it routes
/// decode through the flash kernel, the one decode path that reads the mirror,
/// and a lagging mirror would send every token to the whole-cache conversion.
fn mirror_is_lazy() -> bool {
    static LAZY: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *LAZY.get_or_init(|| {
        std::env::var("SWARMLLM_KV_MIRROR_EAGER").as_deref() != Ok("1")
            && std::env::var("SWARMLLM_GQA_DECODE_FLASH").as_deref() != Ok("1")
    })
}

impl std::ops::Deref for LayerKv {
    type Target = KvPair;
    fn deref(&self) -> &KvPair {
        &self.main
    }
}

impl std::ops::DerefMut for LayerKv {
    fn deref_mut(&mut self) -> &mut KvPair {
        &mut self.main
    }
}

/// How long a conversation's cache may sit untouched before it counts as OVER
/// when another needs the room — the daemon's own figure
/// (`process_pool::CONVERSATION_GAP_SECS`), so the two never disagree about
/// whether a conversation is still going. See
/// [`KvCacheStore::release_finished_conversations`].
pub(crate) const FINISHED_CONVERSATION_AFTER: std::time::Duration =
    std::time::Duration::from_secs(crate::inference::process_pool::CONVERSATION_GAP_SECS);

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
    /// Device bytes held by something OTHER than live request caches that the
    /// KV budget must still see — the prefix cache's snapshots. Set by the
    /// worker whenever that cache changes (gotcha #440); read into
    /// `KvOccupancy::external_bytes` so the per-chunk head-room guard charges
    /// it too.
    external_reserved: std::sync::atomic::AtomicU64,
    /// How to give some of that external memory back: evicts at least the
    /// asked-for bytes of cached prompts if it can, updates
    /// `external_reserved`, and returns what it freed. Installed by the
    /// worker, which owns the prefix cache. `None` means nothing can be
    /// evicted from here, and the guard refuses as it always did.
    external_evictor: std::sync::Mutex<Option<ExternalEvictor>>,
    /// Answers "has the client of this request gone away?" from inside a
    /// forward pass, so a prompt pass can stop between layers instead of
    /// running to the end for nobody (gotcha #441). Installed by the worker
    /// over its cancelled-request set; `None` means never cancelled here.
    cancel_oracle: std::sync::Mutex<Option<CancelOracle>>,
    /// Prompts admitted whose cache has not been fully written yet:
    /// `request_id -> (bytes admission promised in each KvHome, when)`.
    ///
    /// **Why a prompt's admission has to be recorded, not just decided.**
    /// `ensure_room_for_prompt` reads live occupancy, decides, evicts and
    /// returns; the prefill allocates afterwards. On the batched path those
    /// are not even adjacent — `admit_slot` marks the slot `Prefilling` and
    /// the chunks are run on later scheduler ticks — so a second prompt
    /// admitted in between reads an occupancy that does not yet contain the
    /// first, and both pass against the same free memory. No concurrency is
    /// needed for this: the worker's message loop is sequential and the window
    /// is still open, because it spans a return to that loop.
    ///
    /// Same shape as `ModelProcessPool::vram_reserved_mb` does for whole
    /// models, one level down.
    admitted_claims: dashmap::DashMap<String, ([u64; 2], std::time::Instant)>,
    /// Positions a request's caches are built to hold from their FIRST
    /// allocation: `request_id -> (positions, when)`. Set by the worker the
    /// moment a prompt's length is known (`ensure_room_for_prompt`), read by
    /// the executor as it builds each layer's cache and by prefix-cache
    /// hydration. A prompt held this way is one allocation per layer; grown
    /// into, it was one `Tensor::cat` per quantum, each copying everything so
    /// far, and none of those transients charged (FUTURE_WORK #32). Absent —
    /// a cache built before the worker saw the prompt, or the request of a
    /// build that never recorded one — means "grow as before".
    reserved_positions: dashmap::DashMap<String, (usize, std::time::Instant)>,
    /// Requests this machine's OWNER made — `request_id -> when` — as opposed
    /// to work served for the swarm. Set by the worker when a `Generate`
    /// arrives marked so (`mark_owner_request`), read where a forward picks
    /// its thread pool (`serves_the_owner`): the owner's prompt is read on
    /// every physical core, the swarm's within the contribution level
    /// (`cpu_pools::in_phase_pool`). Released with the rest of a request's
    /// bookkeeping, and swept like it.
    owner_requests: dashmap::DashMap<String, std::time::Instant>,
    /// Layer ranges a request's prompt was admitted for UP FRONT, all at once
    /// (`DaemonMsg::AdmitPrompt`, FUTURE_WORK #263): `request_id -> (ranges,
    /// when)`. A forward's own admission stands aside for a range in here
    /// (`admitted_up_front`) — its cache was promised with the rest, and
    /// admitting it again would weigh the request against its own promise.
    /// Released with the rest of a request's bookkeeping, and swept like it.
    ranges_admitted: dashmap::DashMap<String, (Vec<(u32, u32)>, std::time::Instant)>,
}

/// `SWARMLLM_KV_RESERVE=0` records no reservation, so a prompt grows into its
/// cache exactly as it did before 2026-09-12 and the two behaviours can be
/// compared inside ONE binary — the same discipline as
/// `SWARMLLM_FORCE_STANDARD_ATTN`. Read once.
fn kv_reserve_enabled() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        !std::env::var("SWARMLLM_KV_RESERVE")
            .map(|v| v == "0" || v.eq_ignore_ascii_case("false"))
            .unwrap_or(false)
    })
}

/// Given a request id, is that request cancelled?
pub type CancelOracle = Box<dyn Fn(&str) -> bool + Send + Sync>;

/// The message a forward returns when the request behind it was cancelled
/// partway through. Matched by `forward_was_cancelled` on the worker, which
/// is the one place allowed to look at it.
pub(crate) const CANCELLED_MID_FORWARD: &str =
    "cancelled by the client before the forward pass finished";

/// Did this forward stop because its request was cancelled?
pub(crate) fn forward_was_cancelled(e: &crate::error::SwarmError) -> bool {
    matches!(e, crate::error::SwarmError::Inference(m) if m == CANCELLED_MID_FORWARD)
}

/// Frees at least the asked-for bytes of externally held device memory if it
/// can, keeps `external_reserved` current, and returns what it freed.
pub type ExternalEvictor = Box<dyn Fn(u64) -> u64 + Send + Sync>;

/// Why a growth claim was refused: what the device held when it was asked,
/// DECOMPOSED.
///
/// The total alone is not readable. It is store-wide — every live cache plus
/// every prefix-cache snapshot, across all requests — and a reader naturally
/// compares it against the request in front of them. That is how ~1 GB came to
/// be recorded as an unexplained "wander": a refusal reported 4898 MB against a
/// request holding a 2184 MB cache and a 1804 MB snapshot, and the ~910 MB
/// difference was other entries, not a leak. The `entries` count that would
/// have said so is logged by a different line, at `debug`, which nodes do not
/// run at (`docs/FUTURE_WORK.md` item 11).
///
/// So the refusal carries its own parts and the caller prints them together:
/// `in_use_bytes == live_bytes + external_bytes`, over `entries` live caches.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ClaimRefused {
    pub(crate) in_use_bytes: u64,
    /// Bytes reserved by LIVE caches, summed across every request in the store.
    pub(crate) live_bytes: u64,
    /// Bytes held by the prefix cache's snapshots on the same device.
    pub(crate) external_bytes: u64,
    /// Live cache entries the figure covers — one per (model, request). Greater
    /// than 1 means `in_use_bytes` is not this request's cache.
    pub(crate) entries: usize,
}

pub(crate) struct KvCacheEntry {
    /// Per-layer KV cache. Index corresponds to layer index within the model segment.
    /// Each `KvPair` holds a buffer born at the request's reservation (the
    /// admitted prompt plus its reply reserve, or one `KV_CACHE_GROWTH_TOKENS`
    /// quantum when nothing was reserved) and appends into it in place,
    /// concatenating only when a conversation outgrows it.
    pub(crate) layers: Vec<Option<LayerKv>>,
    /// Per-layer SSM state for Qwen 3.5 hybrid models (delta net recurrent state + conv state).
    /// None for non-SSM layers. Only populated for Qwen35Ssm layer variants.
    pub(crate) ssm_states: Vec<Option<SsmState>>,
    /// When this entry was last accessed.
    pub(crate) last_accessed: std::time::Instant,
}

/// Which memory a KV budget is drawn from (#262): a graphics card's, or the
/// machine's. A card/processor split keeps the caches of its card-held layers
/// on the card and the rest in host memory, and each is charged to its own
/// budget — the card's figure once covered both, so a split was refused at a
/// quarter of what its card held and told the room was "other conversations".
/// vLLM keeps one block pool per device for the same reason, and Ollama weighs
/// a split's processor share against the machine's memory, not the card's.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum KvHome {
    Card,
    Host,
}

impl KvHome {
    /// Where a cache allocated on `device` lives.
    pub(crate) fn of(device: &candle_core::Device) -> Self {
        if device.is_cuda() {
            Self::Card
        } else {
            Self::Host
        }
    }

    fn index(self) -> usize {
        match self {
            Self::Card => 0,
            Self::Host => 1,
        }
    }
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
    /// Bytes actually reserved by the K/V tensors, including unused headroom,
    /// on EVERY device — a figure to report, never to weigh against a budget:
    /// a budget is one device's, and reads [`Self::allocated_in`].
    pub allocated_bytes: u64,
    /// Of `allocated_bytes`, the bytes in the machine's memory.
    pub host_bytes: u64,
    /// Bytes covering positions that have really been written.
    pub used_bytes: u64,
    /// Sequence positions written across every entry and layer.
    pub tokens: usize,
    /// Bytes held on the same device by the prefix cache's snapshots — not
    /// live requests, but not free either. Charged by the head-room guard.
    pub external_bytes: u64,
}

impl KvOccupancy {
    /// The allocated bytes in `home` — what a budget for that memory charges.
    pub(crate) fn allocated_in(&self, home: KvHome) -> u64 {
        match home {
            KvHome::Card => self.allocated_bytes.saturating_sub(self.host_bytes),
            KvHome::Host => self.host_bytes,
        }
    }

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
    /// Does this entry hold any of its request's conversation — a written
    /// position in any layer's cache, or a hybrid model's SSM state?
    pub(crate) fn holds_state(&self) -> bool {
        self.layers
            .iter()
            .flatten()
            .any(|kv| kv.current_seq_len() > 0)
            || self.ssm_states.iter().any(Option::is_some)
    }

    /// Bytes this entry reserves, how many of them hold real tokens, and how
    /// many are in host memory (`entries` = 1, no `external_bytes`).
    ///
    /// Reads the ALLOCATED buffer (`Cache::all_data`), not the used window
    /// (`Cache::current_data`), because the allocated buffer is what the
    /// process is holding — see [`KvOccupancy`].
    pub(crate) fn occupancy(&self) -> KvOccupancy {
        let mut allocated = 0u64;
        let mut host = 0u64;
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
                if KvHome::of(data.device()) == KvHome::Host {
                    host += total;
                }
                let cap = cache.max_seq_len().max(1);
                let live = cache.held();
                used += total / cap as u64 * live as u64;
            }
            tokens += kv.k_cache().held();
        }
        // SSM state is per-layer and fixed-size, so it does not scale with the
        // conversation, but it is real memory and belongs in the total.
        for ssm in self.ssm_states.iter().flatten() {
            for t in ssm.tensors() {
                let bytes = t.elem_count() as u64 * t.dtype().size_in_bytes() as u64;
                allocated += bytes;
                used += bytes;
                if KvHome::of(t.device()) == KvHome::Host {
                    host += bytes;
                }
            }
        }
        KvOccupancy {
            entries: 1,
            allocated_bytes: allocated,
            host_bytes: host,
            used_bytes: used,
            tokens,
            external_bytes: 0,
        }
    }

    /// Bytes a forward of `seq_len` positions at `index_pos` newly allocates
    /// in `home`, read off every layer's buffer — a sliding-window layer's
    /// slide included, which no per-position rate can see (FUTURE_WORK #281).
    /// `None` while no layer has allocated: the first forward is priced by the
    /// model's estimate (`kv_budget::KvGrowth`).
    fn bytes_a_forward_adds(&self, index_pos: usize, seq_len: usize, home: KvHome) -> Option<u64> {
        let mut added = None;
        for kv in self.layers.iter().flatten() {
            if let Some((bytes, layer_home)) = kv.bytes_added_by(index_pos, seq_len) {
                let total = added.get_or_insert(0u64);
                if layer_home == home {
                    *total = total.saturating_add(bytes);
                }
            }
        }
        added
    }

    /// Truncate every layer's KV cache to exactly `target_len` sequence
    /// positions. No-op for layers whose current length is already ≤ target.
    /// Used by speculative decoding after partial acceptance — the forward
    /// wrote γ new KV entries, but only k ≤ γ are accepted, so the trailing
    /// γ-k stale slots are discarded before the next round.
    ///
    /// In place, via [`LayerKv::truncate`]: two length fields per layer move
    /// and nothing is copied or allocated. `SWARMLLM_KV_TRUNCATE=copy` restores
    /// the previous snapshot-reset-append implementation so the two can be
    /// compared inside ONE binary (the same discipline as
    /// `SWARMLLM_FORCE_STANDARD_ATTN`); it is not a fallback and has no other
    /// use.
    pub(crate) fn truncate_to(
        &mut self,
        target_len: usize,
    ) -> Result<(), crate::error::SwarmError> {
        self.truncate_to_with(truncate_mode(), target_len)
    }

    /// Every layer's mirror level with its f32 cache — see
    /// [`LayerKv::catch_up_mirror`].
    pub(crate) fn catch_up_mirrors(&mut self) {
        for kv in self.layers.iter_mut().flatten() {
            kv.catch_up_mirror();
        }
    }

    pub(crate) fn truncate_to_with(
        &mut self,
        mode: TruncateMode,
        target_len: usize,
    ) -> Result<(), crate::error::SwarmError> {
        // A recurrent state (Qwen 3.5's DeltaNet) cannot give tokens back —
        // past the one step whose starting state it keeps
        // (`SsmState::rewind_to`): what a refused CUDA graph capture and a
        // rejected guess (`model_worker::ngram_spec_round`) take back.
        // Shortening the attention caches anywhere else would leave the two
        // describing different histories — a silently wrong reply — so
        // refuse. Every path that truncates otherwise is gated away from such
        // models first (`speculation_can_roll_back`, SWIFT, no prefix snapshot
        // of SSM state); this makes a new one loud.
        if !self
            .ssm_states
            .iter()
            .flatten()
            .all(|s| s.can_rewind_to(target_len))
        {
            return Err(crate::error::SwarmError::Internal(
                "cannot take tokens back out of a model with recurrent state \
                 (FUTURE_WORK #117)"
                    .into(),
            ));
        }
        // A sliding-window layer keeps its window and a rewind's worth
        // (`WINDOW_REWIND_POSITIONS`); a rewind further back would answer from
        // keys it no longer holds. Every rewind in this codebase is shorter —
        // a refused capture, a rejected draft, a stored split prompt's last
        // block — so this makes a new, longer one loud (FUTURE_WORK #281).
        if let Some(kv) = self
            .layers
            .iter()
            .flatten()
            .find(|kv| !kv.can_truncate_to(target_len))
        {
            return Err(crate::error::SwarmError::Internal(format!(
                "cannot take this conversation back to position {target_len}: a \
                 sliding-window layer holds it only from position {} (FUTURE_WORK #281)",
                kv.held_from()
            )));
        }
        for s in self.ssm_states.iter_mut().flatten() {
            s.rewind_to(target_len);
        }
        match mode {
            TruncateMode::InPlace => {
                for kv in self.layers.iter_mut().flatten() {
                    kv.truncate(target_len);
                }
                Ok(())
            }
            TruncateMode::Copy => self.truncate_by_copy(target_len),
        }
    }

    /// The implementation this replaced, kept ONLY for A/B measurement.
    ///
    /// Snapshot the kept prefix (one copy), reset, re-append it (a second copy,
    /// plus the f16 mirror rebuilt by `LayerKv::append`). O(target_len ×
    /// hidden) per layer per truncation, and two prefix-sized temporaries on
    /// the cache's device.
    fn truncate_by_copy(&mut self, target_len: usize) -> Result<(), crate::error::SwarmError> {
        use crate::error::SwarmError;
        for cache_opt in self.layers.iter_mut() {
            let Some(kv) = cache_opt else { continue };
            if kv.current_seq_len() <= target_len {
                continue;
            }
            let dim = kv.k_cache().dim();
            let from = kv.held_from();
            let (k_snap, v_snap) = {
                let k = kv
                    .k()
                    .map_err(|e| SwarmError::Internal(format!("kv k snapshot: {e}")))?;
                let v = kv
                    .v()
                    .map_err(|e| SwarmError::Internal(format!("kv v snapshot: {e}")))?;
                let k = k.ok_or_else(|| SwarmError::Internal("kv cache: k empty".into()))?;
                let v = v.ok_or_else(|| SwarmError::Internal("kv cache: v empty".into()))?;
                let k_trunc = k
                    .narrow(dim, 0, target_len - from)
                    .and_then(|t| t.contiguous())
                    .map_err(|e| SwarmError::Internal(format!("kv narrow/contiguous k: {e}")))?;
                let v_trunc = v
                    .narrow(dim, 0, target_len - from)
                    .and_then(|t| t.contiguous())
                    .map_err(|e| SwarmError::Internal(format!("kv narrow/contiguous v: {e}")))?;
                (k_trunc, v_trunc)
            };
            kv.reset();
            kv.begin_at(from);
            kv.append(&k_snap, &v_snap)
                .map_err(|e| SwarmError::Internal(format!("kv truncate re-append: {e}")))?;
        }
        Ok(())
    }
}

/// How a KV cache is shortened on a rejected speculative draft.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TruncateMode {
    /// Move the length; the buffer is untouched. The production path.
    InPlace,
    /// Snapshot, reset and re-append the kept prefix — the implementation this
    /// replaced, reachable only via `SWARMLLM_KV_TRUNCATE=copy` for A/B.
    Copy,
}

/// `SWARMLLM_KV_TRUNCATE=copy` selects the old copying truncation, so the two
/// can be compared inside ONE binary. Read once.
pub(crate) fn truncate_mode() -> TruncateMode {
    use std::sync::OnceLock;
    static MODE: OnceLock<TruncateMode> = OnceLock::new();
    *MODE.get_or_init(|| match std::env::var("SWARMLLM_KV_TRUNCATE") {
        Ok(v) if v.eq_ignore_ascii_case("copy") => TruncateMode::Copy,
        _ => TruncateMode::InPlace,
    })
}

impl KvCacheStore {
    /// Create a new KV-cache store with the given TTL for abandoned entries.
    pub fn new(ttl: std::time::Duration) -> Self {
        Self {
            caches: dashmap::DashMap::new(),
            ttl,
            external_reserved: std::sync::atomic::AtomicU64::new(0),
            external_evictor: std::sync::Mutex::new(None),
            cancel_oracle: std::sync::Mutex::new(None),
            admitted_claims: dashmap::DashMap::new(),
            reserved_positions: dashmap::DashMap::new(),
            owner_requests: dashmap::DashMap::new(),
            ranges_admitted: dashmap::DashMap::new(),
        }
    }

    /// Record how many positions this request's caches should hold from
    /// their first allocation. Replaces an earlier figure — a retry re-admits
    /// rather than stacking. Zero, or `SWARMLLM_KV_RESERVE=0`, records
    /// nothing, and the caches then grow as they always did.
    ///
    /// Public so `examples/prefill_bench.rs` can hold a prompt the way a worker
    /// admits one — reserved whole, not grown into a quantum at a time.
    pub fn set_reserved_positions(&self, request_id: &str, positions: usize) {
        if positions == 0 || !kv_reserve_enabled() {
            self.reserved_positions.remove(request_id);
            return;
        }
        self.reserved_positions.insert(
            request_id.to_string(),
            (positions, std::time::Instant::now()),
        );
    }

    /// Record that `request_id` is this machine's owner's own request — its
    /// prompt is read on the owner's full width (`cpu_pools::in_phase_pool`).
    pub(crate) fn mark_owner_request(&self, request_id: &str) {
        self.owner_requests
            .insert(request_id.to_string(), std::time::Instant::now());
    }

    /// Is `request_id` the owner's own request? `false` for anything not
    /// marked — work for the swarm. Marked by the worker at a `Generate`
    /// carrying `for_the_owner`, and at a `Forward` carrying it (the owner's
    /// segment of a request this node coordinates).
    pub(crate) fn serves_the_owner(&self, request_id: &str) -> bool {
        self.owner_requests.contains_key(request_id)
    }

    /// Positions reserved for this request's caches; 0 when nothing was
    /// recorded, which every caller reads as "one quantum, grow from there".
    pub(crate) fn reserved_positions(&self, request_id: &str) -> usize {
        self.reserved_positions
            .get(request_id)
            .map(|r| r.0)
            .unwrap_or(0)
    }

    /// Whether a forward reaching `positions` would allocate NOTHING in the
    /// caches under `key`: every layer has a cache, and every buffer in it —
    /// K and V, and the f16 mirror's pair where one is kept — already holds
    /// `positions`. Stricter than [`Self::allocated_positions`], which reads
    /// one layer's K buffer: a mirror hydrated from a prefix snapshot can be
    /// sized differently from the cache beside it.
    ///
    /// The question a CUDA graph capture asks before a decode step
    /// (`inference::cuda_graph`): a buffer that grows inside a capture frees
    /// memory allocated outside it and keeps memory allocated inside it, and
    /// the graph can then neither be replayed nor trusted.
    ///
    /// A recurrent layer (Qwen 3.5's DeltaNet) keeps no KV cache: it holds when
    /// the buffers its next step writes into exist (`SsmState`, #228 (4)). A
    /// slot `keeps_no_cache` names stays empty by design and holds too — a
    /// Gemma 4 E2B/E4B layer attending an earlier layer's cache
    /// (`SplitModel::slot_keeps_no_cache`, #295); read as a cache yet to be
    /// built, it kept every step of those models from being captured.
    pub(crate) fn every_cache_holds(
        &self,
        key: &str,
        positions: usize,
        keeps_no_cache: impl Fn(usize) -> bool,
    ) -> bool {
        self.caches.get(key).is_some_and(|entry| {
            !entry.layers.is_empty()
                && entry
                    .layers
                    .iter()
                    .enumerate()
                    .all(|(i, layer)| match layer {
                        Some(kv) => kv
                            .all_caches()
                            .iter()
                            .all(|c| c.all_data().is_some() && c.reach() >= positions),
                        None if keeps_no_cache(i) => true,
                        None => entry
                            .ssm_states
                            .get(i)
                            .and_then(Option::as_ref)
                            .is_some_and(SsmState::holds_the_next_step),
                    })
        })
    }

    /// Positions the caches under `key` are ALLOCATED for right now — the
    /// buffer, not the positions written — or 0 before the first append. What
    /// the head-room guard compares a forward against: a forward that fits
    /// inside this claims nothing, whatever quantum boundary it crosses.
    ///
    /// A cache that was `reset()` reads 0 here (its buffer is dropped) and
    /// re-allocates its previous size on the next append; the guard then
    /// charges the forward's own need, which is at most that size and is
    /// memory the budget had already seen this request hold.
    ///
    /// The LEAST any layer reaches ([`SeqCache::reach`]): a sliding-window
    /// layer's buffer reaches further than its size once it slides, and the
    /// first layer of a Gemma is one — read alone, it would say a forward fits
    /// that the whole-history layers must grow for (FUTURE_WORK #281).
    pub(crate) fn allocated_positions(&self, key: &str) -> usize {
        self.caches
            .get(key)
            .and_then(|entry| {
                entry
                    .layers
                    .iter()
                    .flatten()
                    .filter(|kv| kv.k_cache().all_data().is_some())
                    .map(|kv| kv.k_cache().reach())
                    .min()
            })
            .unwrap_or(0)
    }

    /// Bytes a forward of `seq_len` positions at `index_pos` newly allocates
    /// in `home` for the caches under `key` — what the head-room guard claims
    /// once they exist. `None` before the first append.
    pub(crate) fn bytes_a_forward_adds(
        &self,
        key: &str,
        index_pos: usize,
        seq_len: usize,
        home: KvHome,
    ) -> Option<u64> {
        self.caches
            .get(key)
            .and_then(|entry| entry.bytes_a_forward_adds(index_pos, seq_len, home))
    }

    /// Bytes the caches under `key` hold in `home`.
    pub(crate) fn allocated_bytes_of(&self, key: &str, home: KvHome) -> u64 {
        self.caches
            .get(key)
            .map_or(0, |entry| entry.occupancy().allocated_in(home))
    }

    /// Record that a prompt has been admitted for `bytes` it has not yet
    /// allocated. Replaces any earlier claim for the same request — a retry
    /// re-admits rather than stacking.
    pub(crate) fn record_prompt_admission(&self, request_id: &str, home: KvHome, bytes: u64) {
        let mut claim = self.admitted_claims.get(request_id).map_or([0; 2], |c| c.0);
        claim[home.index()] = bytes;
        if claim == [0; 2] {
            self.admitted_claims.remove(request_id);
            return;
        }
        self.admitted_claims
            .insert(request_id.to_string(), (claim, std::time::Instant::now()));
    }

    /// Record that `request_id`'s prompt was admitted up front over `ranges`
    /// (FUTURE_WORK #263). Replaces an earlier record — a re-plan admits its
    /// own shape afresh.
    pub(crate) fn record_ranges_admitted(&self, request_id: &str, ranges: Vec<(u32, u32)>) {
        self.ranges_admitted
            .insert(request_id.to_string(), (ranges, std::time::Instant::now()));
    }

    /// Was `range` covered by `request_id`'s up-front admission? A range inside
    /// an admitted one counts: a forward's range is a plan segment, which a
    /// failover may only narrow.
    pub(crate) fn admitted_up_front(&self, request_id: &str, range: (u32, u32)) -> bool {
        self.ranges_admitted.get(request_id).is_some_and(|r| {
            r.0.iter()
                .any(|&(start, end)| start <= range.0 && range.1 <= end)
        })
    }

    /// Bytes promised to admitted prompts that their caches do not hold yet.
    ///
    /// **Drawn down by what each request has actually allocated**, so a claim
    /// shrinks to nothing as its prefill proceeds rather than double-charging
    /// the same memory. That is also what bounds the damage if a claim is ever
    /// left behind: a fully-prefilled request contributes zero whether or not
    /// anything removed its entry.
    pub(crate) fn outstanding_admission_bytes(&self, home: KvHome) -> u64 {
        if self.admitted_claims.is_empty() {
            // The overwhelming common case — no scan at all.
            return 0;
        }
        // One pass over the caches, summing per request id. The cache key is
        // "model_key\0request_id", and one request can hold several entries
        // (tensor-parallel ranks), so they are added together.
        let mut allocated_by_request: std::collections::HashMap<&str, u64> =
            std::collections::HashMap::new();
        let keys: Vec<(String, u64)> = self
            .caches
            .iter()
            .map(|e| (e.key().clone(), e.value().occupancy().allocated_in(home)))
            .collect();
        for (k, allocated) in &keys {
            if let Some(rid) = k.split('\0').nth(1) {
                *allocated_by_request.entry(rid).or_insert(0) += *allocated;
            }
        }
        self.admitted_claims
            .iter()
            .map(|c| {
                let written = allocated_by_request
                    .get(c.key().as_str())
                    .copied()
                    .unwrap_or(0);
                c.value().0[home.index()].saturating_sub(written)
            })
            .sum()
    }

    /// Install the cancellation oracle a forward pass consults between layers.
    pub fn set_cancel_oracle(&self, f: CancelOracle) {
        if let Ok(mut slot) = self.cancel_oracle.lock() {
            *slot = Some(f);
        }
    }

    /// Has the request been cancelled? Cheap: one lock and one map probe, so
    /// a forward can afford it once per layer.
    pub fn request_cancelled(&self, request_id: &str) -> bool {
        match self.cancel_oracle.lock() {
            Ok(slot) => slot.as_ref().is_some_and(|f| f(request_id)),
            Err(_) => false,
        }
    }

    /// Install the way to shrink the external reservation on demand — the
    /// worker hands in a closure over its prefix cache (gotcha #440).
    pub fn set_external_evictor(&self, f: ExternalEvictor) {
        if let Ok(mut slot) = self.external_evictor.lock() {
            *slot = Some(f);
        }
    }

    /// May a forward allocate `claim` more bytes of KV cache in `home`?
    ///
    /// Live caches plus the prefix cache's snapshots plus the claim, against
    /// the budget. When over, **cached prompts are evicted before the request
    /// is refused** — the same order as whole-prompt admission, and the
    /// half that was missing when the snapshot charge first shipped: a
    /// prompt admitted at token 0 was then refused at its next growth quantum
    /// because the guard could see the cached prompts and not shrink them,
    /// where the release before had served it (slowly). A refusal names what
    /// the device held so the message can say so.
    ///
    /// Before either, conversations that are OVER give their room up
    /// ([`Self::release_finished_conversations`]): they are worth less than a
    /// cached prompt, which may yet be reused, and far less than the reply
    /// `claimant` is in the middle of.
    pub(crate) fn claim_room(
        &self,
        home: KvHome,
        budget_bytes: u64,
        claim: u64,
        claimant: &str,
    ) -> Result<(), ClaimRefused> {
        use super::kv_budget::claim_exceeds_headroom;
        // The prefix cache's snapshots are charged where they are kept beside
        // the live cache — a processor worker's, so `Host` — and nowhere else.
        let in_use = |occ: &KvOccupancy| occ.allocated_in(home).saturating_add(occ.external_bytes);
        let occ = self.occupancy();
        if !claim_exceeds_headroom(budget_bytes, in_use(&occ), claim) {
            return Ok(());
        }
        if self.release_finished_conversations(FINISHED_CONVERSATION_AFTER, claimant) > 0 {
            let occ = self.occupancy();
            if !claim_exceeds_headroom(budget_bytes, in_use(&occ), claim) {
                return Ok(());
            }
        }
        let occ = self.occupancy();
        let excess = in_use(&occ).saturating_add(claim) - budget_bytes;
        if occ.external_bytes > 0 {
            let freed = match self.external_evictor.lock() {
                Ok(slot) => slot.as_ref().map(|f| f(excess)).unwrap_or(0),
                Err(_) => 0,
            };
            if freed > 0 {
                let occ = self.occupancy();
                if !claim_exceeds_headroom(budget_bytes, in_use(&occ), claim) {
                    return Ok(());
                }
                return Err(ClaimRefused {
                    in_use_bytes: in_use(&occ),
                    live_bytes: occ.allocated_in(home),
                    external_bytes: occ.external_bytes,
                    entries: occ.entries,
                });
            }
        }
        Err(ClaimRefused {
            in_use_bytes: in_use(&occ),
            live_bytes: occ.allocated_in(home),
            external_bytes: occ.external_bytes,
            entries: occ.entries,
        })
    }

    /// Record how many device bytes the prefix cache currently holds, so the
    /// head-room guard charges them. Call after every insert or release.
    pub fn set_external_reserved(&self, bytes: u64) {
        self.external_reserved
            .store(bytes, std::sync::atomic::Ordering::Relaxed);
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

    /// Does this store hold any of `request_id`'s conversation for this model
    /// segment?
    ///
    /// Asked before a forward that CONTINUES a conversation is allowed to run
    /// (`model_worker::forward_lacks_its_conversation`). The forward creates
    /// its entry lazily, so "no entry" and "an entry with nothing written" mean
    /// the same thing here: nothing for attention to read.
    pub(crate) fn request_holds_state(&self, model_key: &str, request_id: &str) -> bool {
        let key = Self::cache_key(model_key, request_id);
        self.caches
            .get(key.as_str())
            .is_some_and(|entry| entry.holds_state())
    }

    /// How many positions a request's cache holds: the fewest any layer
    /// holds, so a pass that stopped part-way never reads as whole. `0` when
    /// there is no entry.
    pub(crate) fn request_positions(&self, model_key: &str, request_id: &str) -> usize {
        let key = Self::cache_key(model_key, request_id);
        self.caches.get(key.as_str()).map_or(0, |entry| {
            entry
                .layers
                .iter()
                .flatten()
                .map(|kv| kv.current_seq_len())
                .min()
                .unwrap_or(0)
        })
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

    /// Bring every layer's f16 mirror for this request level with its f32
    /// cache, outside any capture — see [`LayerKv::catch_up_mirror`].
    pub(crate) fn catch_up_mirrors(&self, model_key: &str, request_id: &str) {
        let key = Self::cache_key(model_key, request_id);
        if let Some(mut entry) = self.caches.get_mut(key.as_str()) {
            entry.catch_up_mirrors();
        }
    }

    /// Clear (remove) the KV-cache for a specific request.
    /// Also clears any TP-keyed variants (tp{rank}-{model_key}).
    pub fn clear_request(&self, model_key: &str, request_id: &str) {
        // The request is over, so whatever its admission promised is no longer
        // owed. Every worker path that finishes or abandons a request already
        // comes through here.
        self.forget_request_bookkeeping(request_id);
        self.clear_request_caches(model_key, request_id);
    }

    /// [`Self::clear_request`]'s caches without its bookkeeping: the start of a
    /// prompt pass over a range whose cache was admitted up front with the
    /// request's other ranges (FUTURE_WORK #263). The pass starts its cache
    /// afresh; the promise, the reservation and the owner's mark were made for
    /// the whole request and stay until it ends.
    pub(crate) fn clear_request_caches(&self, model_key: &str, request_id: &str) {
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
        // A claim outlives its request only if something failed to clear it.
        // The draw-down already makes a stale one contribute nothing once its
        // prefill finished, so this is the belt to that braces: it bounds a
        // claim from a prompt that was admitted and then never prefilled at
        // all, which nothing else would ever remove.
        self.admitted_claims
            .retain(|_, (_, at)| at.elapsed() <= ttl);
        self.reserved_positions
            .retain(|_, (_, at)| at.elapsed() <= ttl);
        self.owner_requests.retain(|_, at| at.elapsed() <= ttl);
        self.ranges_admitted
            .retain(|_, (_, at)| at.elapsed() <= ttl);
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

    /// Release the per-request bookkeeping that dies with a request.
    ///
    /// The two maps are recorded together and must be released together. They
    /// are separated here so that a path which ends a request cannot release
    /// one and forget the other — which is exactly what `cleanup_request_id`
    /// did, and it is not a small leak. `outstanding_admission_bytes` draws a
    /// claim down by what its request has ALLOCATED, read off `caches`; that is
    /// what was relied on to bound a claim nobody removed. Removing the cache
    /// entry removes the draw-down with it, so a claim left behind by a
    /// cleanup springs back from zero to its FULL size and charges the budget
    /// for memory that same cleanup just freed, until the TTL sweep. A 5101-
    /// token prompt left 1848 MB owed on a node with 3042 MB for conversations,
    /// so the next such prompt was refused for ten minutes — and the refusal
    /// tells the user to close other programs, which cannot help.
    fn forget_request_bookkeeping(&self, request_id: &str) {
        self.admitted_claims.remove(request_id);
        self.reserved_positions.remove(request_id);
        self.owner_requests.remove(request_id);
        self.ranges_admitted.remove(request_id);
    }

    /// Drop the cache of every conversation nobody has touched for `idle` — a
    /// reply that is over — except `keep`'s, with the bookkeeping of each
    /// request left with no cache at all. Returns the bytes freed.
    ///
    /// **A node serving a SEGMENT of another computer's request is never told
    /// that the reply finished.** `ReleaseRequestKv` reaches only the
    /// coordinator's own workers, and a worker's store never ran its TTL sweep
    /// at all — so those caches stayed for as long as the worker lived, and a
    /// live reply on the same worker was refused for "other conversations on
    /// this node" that had ended (report #005, 2026-10-07: 330 MB of a 469 MB
    /// budget, 71 s into a reply that had already streamed, so nothing could
    /// re-plan it).
    ///
    /// Asked only when a live conversation needs the room — at prompt admission
    /// and at a growth claim — and only of caches silent for
    /// [`FINISHED_CONVERSATION_AFTER`], the daemon's own reading of a finished
    /// conversation. One that was in fact only waiting (a long prompt pass on a
    /// later machine of the chain) is refused at its next forward
    /// (`model_worker::forward_lacks_its_conversation`) before any of its reply
    /// has reached anyone, which the router re-plans; a reply refused mid-stream
    /// is lost. That is the trade, and it is vLLM's direction too: it preempts
    /// the request that can be recomputed, not the one being served.
    pub(crate) fn release_finished_conversations(
        &self,
        idle: std::time::Duration,
        keep: &str,
    ) -> u64 {
        let mut freed = 0u64;
        let mut finished: Vec<String> = Vec::new();
        self.caches.retain(|key, entry| {
            if !conversation_has_ended(key, entry, idle, keep) {
                return true;
            }
            freed = freed.saturating_add(entry.occupancy().allocated_bytes);
            finished.push(request_of_key(key).to_string());
            false
        });
        finished.sort_unstable();
        finished.dedup();
        for request in &finished {
            let suffix = format!("\0{request}");
            if !self.caches.iter().any(|e| e.key().ends_with(&suffix)) {
                self.forget_request_bookkeeping(request);
            }
        }
        if !finished.is_empty() {
            tracing::info!(
                conversations = finished.len(),
                freed_mb = freed / (1024 * 1024),
                idle_secs = idle.as_secs(),
                "DIAG: KV cache — released conversations that had ended, for one that needs the room"
            );
        }
        freed
    }

    /// What [`Self::release_finished_conversations`] would free in `home` now,
    /// freeing nothing: the room a peer's question is weighed with
    /// (`model_worker::Admitting::OnlyWeigh`, FUTURE_WORK #252 (b)), which must
    /// not end another conversation to answer it. Their caches only — a
    /// finished request's admission claim has been drawn down by what it
    /// wrote, so it adds next to nothing.
    pub(crate) fn finished_conversations_bytes_in(
        &self,
        home: KvHome,
        idle: std::time::Duration,
        keep: &str,
    ) -> u64 {
        self.caches
            .iter()
            .filter(|e| conversation_has_ended(e.key(), e.value(), idle, keep))
            .map(|e| e.value().occupancy().allocated_in(home))
            .fold(0u64, u64::saturating_add)
    }

    /// Remove all cache entries for a given request_id (across all models).
    pub fn cleanup_request_id(&self, request_id: &str) {
        self.forget_request_bookkeeping(request_id);
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
            let one = entry.value().occupancy();
            out.entries += 1;
            out.allocated_bytes += one.allocated_bytes;
            out.host_bytes += one.host_bytes;
            out.used_bytes += one.used_bytes;
            out.tokens += one.tokens;
        }
        out.external_bytes = self
            .external_reserved
            .load(std::sync::atomic::Ordering::Relaxed);
        out
    }
}

/// The request a store key belongs to (`cache_key`: `model\0request`).
fn request_of_key(key: &str) -> &str {
    key.split('\0').nth(1).unwrap_or_default()
}

/// Has the conversation under `key` ended: silent for `idle`, and not `keep`'s
/// own? The one reading behind [`KvCacheStore::release_finished_conversations`]
/// and the read-only [`KvCacheStore::finished_conversations_bytes_in`].
fn conversation_has_ended(
    key: &str,
    entry: &KvCacheEntry,
    idle: std::time::Duration,
    keep: &str,
) -> bool {
    request_of_key(key) != keep && entry.last_accessed.elapsed() >= idle
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A refusal's total is store-wide, and it says so.
    ///
    /// The figure covers every live cache plus the prefix cache's snapshots,
    /// across all requests — so a reader comparing it against the request in
    /// front of them sees a gap that looks like a leak. It is not: it is the
    /// other entries. This pins that the parts are carried and that they add
    /// up, so the refusal line can print them together (FUTURE_WORK item 11).
    #[test]
    fn a_refusal_decomposes_the_figure_it_reports() {
        let store = KvCacheStore::new(std::time::Duration::from_secs(60));
        // Two live entries, so the total is demonstrably not one request's.
        for req in ["req-a", "req-b"] {
            let mut entry = store.get_or_create("m", req, 1);
            let k = Tensor::zeros((1usize, 2, 8, 4), DType::F32, &Device::Cpu).unwrap();
            let mut kv = new_kv_cache(64, KvStorage::F32Mirrored, 0, None);
            kv.append(&k, &k.clone()).unwrap();
            entry.layers[0] = Some(kv);
        }
        store.set_external_reserved(1_000_000);

        // A budget of zero refuses anything, which is all this needs.
        let refused = store
            .claim_room(KvHome::Host, 0, 1024, "r")
            .expect_err("a zero budget must refuse");
        assert_eq!(
            refused.in_use_bytes,
            refused.live_bytes + refused.external_bytes,
            "the reported total must be exactly the parts it is made of"
        );
        assert_eq!(
            refused.external_bytes, 1_000_000,
            "the snapshot charge is carried separately"
        );
        assert!(
            refused.entries >= 2,
            "the entry count must reveal that the total spans more than one \
             request; got {}",
            refused.entries
        );
        assert!(
            refused.live_bytes > 0,
            "live caches must be counted, not folded away"
        );
    }

    use crate::inference::layers::{kv_cache_reservation, new_kv_cache, KV_CACHE_GROWTH_TOKENS};
    use candle_core::{DType, Device, Tensor};

    /// A cache built for a known prompt length holds it in ONE allocation and
    /// grows only past it; the same prompt appended to an unreserved cache
    /// grows into place a quantum at a time. The buffer size is the evidence
    /// (`KV_GROWTH_STEPS` is process-wide and tests run in parallel).
    #[test]
    fn a_reserved_cache_holds_its_prompt_without_growing() {
        let dev = Device::Cpu;
        let prompt = Tensor::zeros((1usize, 2, 2000, 4), DType::F32, &dev).unwrap();
        let more = Tensor::zeros((1usize, 2, 100, 4), DType::F32, &dev).unwrap();

        let mut reserved = new_kv_cache(4096, KvStorage::F32, 2048, None);
        reserved.append(&prompt, &prompt).unwrap();
        assert_eq!(
            reserved.k_cache().max_seq_len(),
            2048,
            "the prompt must fit the first allocation exactly as reserved"
        );
        reserved.append(&more, &more).unwrap();
        assert_eq!(
            reserved.k_cache().max_seq_len(),
            2048 + KV_CACHE_GROWTH_TOKENS,
            "past the reservation a cache grows by the ordinary quantum"
        );

        // The control: no reservation, the same prompt, grown into place.
        let mut plain = new_kv_cache(4096, KvStorage::F32, 0, None);
        plain.append(&prompt, &prompt).unwrap();
        assert_eq!(
            plain.k_cache().max_seq_len(),
            4 * KV_CACHE_GROWTH_TOKENS,
            "512 -> 1024 -> 1536 -> 2048: three concatenations to hold 2000 positions"
        );
    }

    /// The reservation is clamped to what the model can ever use, and a zero
    /// reservation is the ordinary one-quantum start.
    #[test]
    fn a_reservation_is_clamped_to_the_context_window() {
        assert_eq!(
            new_kv_cache(4096, KvStorage::F32, 10_000, None)
                .k_cache()
                .max_seq_len(),
            4096
        );
        assert_eq!(
            new_kv_cache(4096, KvStorage::F32, 0, None)
                .k_cache()
                .max_seq_len(),
            KV_CACHE_GROWTH_TOKENS
        );
        assert_eq!(
            new_kv_cache(4096, KvStorage::F32, 100, None)
                .k_cache()
                .max_seq_len(),
            KV_CACHE_GROWTH_TOKENS
        );
        // A model whose whole window is under one quantum reserves the window.
        assert_eq!(
            new_kv_cache(64, KvStorage::F32, 0, None)
                .k_cache()
                .max_seq_len(),
            64
        );
    }

    /// Whose request this is lives exactly as long as the request, like the
    /// reservation: marked by the worker, read where a forward picks its
    /// threads, gone with `clear_request` and with `cleanup_request_id`.
    #[test]
    fn an_owner_request_is_marked_per_request_and_released_with_it() {
        let store = KvCacheStore::new(std::time::Duration::from_secs(60));
        assert!(!store.serves_the_owner("r1"), "unmarked is the swarm's");
        store.mark_owner_request("r1");
        assert!(store.serves_the_owner("r1"));
        assert!(!store.serves_the_owner("r2"), "another request is its own");
        store.clear_request("m", "r1");
        assert!(!store.serves_the_owner("r1"), "released with the request");
        store.mark_owner_request("r3");
        store.cleanup_request_id("r3");
        assert!(!store.serves_the_owner("r3"), "and by the other cleanup");
    }

    /// A peer's question is weighed with the room the release WOULD free
    /// (#252 (b)), and asking frees nothing: the figure is exactly what the
    /// release then frees in that memory — never the asking request's own
    /// cache, never a conversation still going.
    #[test]
    fn the_room_a_question_is_weighed_with_is_what_the_release_would_free() {
        let gap = FINISHED_CONVERSATION_AFTER;
        let long_ago = gap + std::time::Duration::from_secs(1);
        let store = KvCacheStore::new(std::time::Duration::from_secs(600));
        conversation(&store, "ended", 4, long_ago);
        let one = store.occupancy().allocated_bytes;
        assert!(one > 0, "the test needs a cache that holds bytes");
        conversation(&store, "asking", 4, long_ago);
        conversation(&store, "going", 4, gap / 2);

        let would = store.finished_conversations_bytes_in(KvHome::Host, gap, "asking");
        assert_eq!(would, one, "the ended conversation only");
        assert_eq!(store.occupancy().entries, 3, "asking freed nothing");
        assert_eq!(
            store.finished_conversations_bytes_in(KvHome::Card, gap, "asking"),
            0,
            "a cache in host memory frees no card memory"
        );
        assert_eq!(store.release_finished_conversations(gap, "asking"), would);
        assert!(store.request_holds_state("m", "asking"));
        assert!(store.request_holds_state("m", "going"));
    }

    /// **An up-front admission outlives the start of each segment's prompt
    /// pass, and goes with the request** (#263). Every local segment's first
    /// forward starts its cache afresh; were that to forget the request's
    /// admission as `clear_request` does, the second segment would admit itself
    /// again — weighed against its own request's promise — and the late refusal
    /// the up-front check exists to prevent would be back.
    #[test]
    fn an_up_front_admission_survives_each_segments_first_forward() {
        let store = KvCacheStore::new(std::time::Duration::from_secs(60));
        assert!(!store.admitted_up_front("r1", (0, 1)), "nothing admitted");
        store.set_reserved_positions("r1", 6656);
        store.record_ranges_admitted("r1", vec![(0, 1), (8, 20), (32, 48)]);
        assert!(store.admitted_up_front("r1", (8, 20)));
        assert!(
            store.admitted_up_front("r1", (34, 48)),
            "a range inside an admitted one counts"
        );
        assert!(
            !store.admitted_up_front("r1", (1, 8)),
            "a peer's range was never ours to admit"
        );
        assert!(!store.admitted_up_front("r2", (8, 20)), "per request");

        store.clear_request_caches("m", "r1");
        assert!(
            store.admitted_up_front("r1", (32, 48)),
            "a segment's first forward keeps the request's admission"
        );
        assert_eq!(store.reserved_positions("r1"), 6656, "and its reservation");

        store.clear_request("m", "r1");
        assert!(
            !store.admitted_up_front("r1", (32, 48)),
            "the request's end"
        );
        assert_eq!(store.reserved_positions("r1"), 0);
        store.record_ranges_admitted("r3", vec![(0, 4)]);
        store.cleanup_request_id("r3");
        assert!(
            !store.admitted_up_front("r3", (0, 4)),
            "and the other cleanup"
        );
    }

    /// The reservation lives exactly as long as the request: recorded by the
    /// worker, read by the executor, gone with `clear_request`.
    #[test]
    fn a_reservation_is_recorded_per_request_and_released_with_it() {
        let store = KvCacheStore::new(std::time::Duration::from_secs(60));
        assert_eq!(
            store.reserved_positions("r1"),
            0,
            "nothing recorded reads as zero"
        );
        store.set_reserved_positions("r1", 2560);
        assert_eq!(store.reserved_positions("r1"), 2560);
        assert_eq!(
            store.reserved_positions("r2"),
            0,
            "another request's figure is its own"
        );
        store.set_reserved_positions("r1", 3072);
        assert_eq!(
            store.reserved_positions("r1"),
            3072,
            "a retry replaces, never stacks"
        );
        store.clear_request("m", "r1");
        assert_eq!(
            store.reserved_positions("r1"),
            0,
            "released with the request"
        );
        store.set_reserved_positions("r1", 512);
        store.set_reserved_positions("r1", 0);
        assert_eq!(store.reserved_positions("r1"), 0, "zero records nothing");
    }

    /// What the guard compares against is the ALLOCATED buffer, not the
    /// positions written: before the first append nothing is allocated, after
    /// it the whole reservation is, however few positions it holds.
    #[test]
    fn allocated_positions_is_the_buffer_not_the_tokens() {
        let store = KvCacheStore::new(std::time::Duration::from_secs(60));
        let key = KvCacheStore::cache_key("m", "r1");
        assert_eq!(store.allocated_positions(&key), 0, "no entry");
        {
            let mut entry = store.get_or_create_keyed(&key, 2);
            entry.layers[1] = Some(new_kv_cache(4096, KvStorage::F32, 2048, None));
        }
        assert_eq!(
            store.allocated_positions(&key),
            0,
            "built but nothing appended yet"
        );
        let k = Tensor::zeros((1usize, 2, 5, 4), DType::F32, &Device::Cpu).unwrap();
        store.get_or_create_keyed(&key, 2).layers[1]
            .as_mut()
            .unwrap()
            .append(&k, &k)
            .unwrap();
        assert_eq!(
            store.allocated_positions(&key),
            2048,
            "five tokens, one reservation"
        );
    }

    /// The question a CUDA graph capture asks before a decode step: would it
    /// allocate anything? Only when every layer has a cache and every buffer
    /// already holds the new position does it not.
    #[test]
    fn a_step_allocates_nothing_only_inside_every_layers_buffer() {
        let store = KvCacheStore::new(std::time::Duration::from_secs(600));
        let key = KvCacheStore::cache_key("m", "r");
        assert!(
            !store.every_cache_holds(&key, 1, |_| false),
            "no entry yet: the first append allocates"
        );
        let fill = |layer: usize| {
            let mut entry = store.get_or_create("m", "r", 2);
            let k = Tensor::zeros((1usize, 2, 5, 4), DType::F32, &Device::Cpu).unwrap();
            let mut kv = new_kv_cache(4096, KvStorage::F32, 0, None);
            kv.append(&k, &k.clone()).unwrap();
            entry.layers[layer] = Some(kv);
        };
        fill(0);
        assert!(
            !store.every_cache_holds(&key, 6, |_| false),
            "a layer with no cache yet allocates on its first append"
        );
        fill(1);
        let capacity = store.allocated_positions(&key);
        assert!(
            capacity > 6,
            "one growth quantum holds more than the prompt"
        );
        assert!(store.every_cache_holds(&key, 6, |_| false));
        assert!(store.every_cache_holds(&key, capacity, |_| false));
        assert!(
            !store.every_cache_holds(&key, capacity + 1, |_| false),
            "one position past the buffer grows it"
        );
    }

    /// Fill one layer of NAMED request's cache with `n` positions.
    fn append_for(store: &KvCacheStore, request_id: &str, n: usize, max_seq_len: usize) {
        let mut entry = store.get_or_create("m", request_id, 1);
        let k = Tensor::zeros((1usize, 2, n, 4), DType::F32, &Device::Cpu).unwrap();
        let mut kv = new_kv_cache(max_seq_len, KvStorage::F32Mirrored, 0, None);
        kv.append(&k, &k.clone()).unwrap();
        entry.layers[0] = Some(kv);
    }

    /// The sequence the batched path actually produces: a prompt is admitted
    /// and its slot marked `Prefilling`, control returns to the message loop,
    /// and the next prompt is weighed — before the first has written a byte.
    ///
    /// Without the claim, that second read sees an empty store and both
    /// prompts pass against the same free memory.
    #[test]
    fn a_prompt_admitted_but_not_yet_prefilled_is_still_charged() {
        let store = KvCacheStore::new(std::time::Duration::from_secs(600));
        assert_eq!(
            store.outstanding_admission_bytes(KvHome::Host),
            0,
            "nothing admitted yet"
        );

        store.record_prompt_admission("first", KvHome::Host, 4_000_000);
        assert_eq!(
            store.outstanding_admission_bytes(KvHome::Host),
            4_000_000,
            "a prompt that has been promised memory must be visible to the next one"
        );
    }

    /// And it must not be charged twice: as the prefill writes, the claim is
    /// drawn down by what that request has actually allocated.
    #[test]
    fn a_claim_shrinks_as_its_own_prefill_allocates() {
        let store = KvCacheStore::new(std::time::Duration::from_secs(600));
        append_for(&store, "r", 8, 64);
        let allocated = store.occupancy().allocated_bytes;
        assert!(allocated > 0, "the fixture must actually allocate");

        // Promised twice what it has written so far.
        store.record_prompt_admission("r", KvHome::Host, allocated * 2);
        assert_eq!(
            store.outstanding_admission_bytes(KvHome::Host),
            allocated,
            "only the part not yet written is still owed"
        );

        // Once it has written at least what it promised, it owes nothing —
        // which is also what bounds a claim nothing ever removes.
        store.record_prompt_admission("r", KvHome::Host, allocated);
        assert_eq!(store.outstanding_admission_bytes(KvHome::Host), 0);
        store.record_prompt_admission("r", KvHome::Host, allocated / 2);
        assert_eq!(
            store.outstanding_admission_bytes(KvHome::Host),
            0,
            "saturating: a request that overshot its promise is not a credit"
        );
    }

    /// Finishing a request releases what it was promised. Every worker path
    /// that ends or abandons a request already calls `clear_request`.
    #[test]
    fn finishing_a_request_releases_its_promise() {
        let store = KvCacheStore::new(std::time::Duration::from_secs(600));
        store.record_prompt_admission("r", KvHome::Host, 4_000_000);
        assert_eq!(store.outstanding_admission_bytes(KvHome::Host), 4_000_000);
        store.clear_request("m", "r");
        assert_eq!(store.outstanding_admission_bytes(KvHome::Host), 0);
    }

    /// A prompt admitted and then never prefilled at all is the one case the
    /// draw-down cannot bound, so the TTL sweep does.
    #[test]
    fn a_promise_that_was_never_taken_up_expires() {
        let store = KvCacheStore::new(std::time::Duration::from_millis(0));
        store.record_prompt_admission("abandoned", KvHome::Host, 4_000_000);
        assert_eq!(store.outstanding_admission_bytes(KvHome::Host), 4_000_000);
        std::thread::sleep(std::time::Duration::from_millis(5));
        store.cleanup_expired();
        assert_eq!(
            store.outstanding_admission_bytes(KvHome::Host),
            0,
            "a claim must not outlive its request for ever"
        );
    }

    /// Fill one layer of a request's cache with `n` positions, the way a
    /// forward pass does.
    fn append(store: &KvCacheStore, n: usize, max_seq_len: usize) {
        let mut entry = store.get_or_create("m", "r", 1);
        let k = Tensor::zeros((1usize, 2, n, 4), DType::F32, &Device::Cpu).unwrap();
        let mut kv = new_kv_cache(max_seq_len, KvStorage::F32Mirrored, 0, None);
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

    /// The prefix cache's snapshots are device memory the head-room guard
    /// must charge (gotcha #440); they arrive through `set_external_reserved`
    /// and are reported beside the live figure, never folded into it.
    #[test]
    fn externally_reserved_bytes_are_reported_beside_the_live_ones() {
        let store = KvCacheStore::new(std::time::Duration::from_secs(60));
        append(&store, 4, 4096);
        let live = store.occupancy().allocated_bytes;
        store.set_external_reserved(12_345);
        let occ = store.occupancy();
        assert_eq!(
            occ.allocated_bytes, live,
            "live bytes must not absorb the external figure"
        );
        assert_eq!(occ.external_bytes, 12_345);
        store.set_external_reserved(0);
        assert_eq!(store.occupancy().external_bytes, 0);
    }

    /// The per-chunk guard shrinks cached prompts before it refuses, and
    /// refuses only when even that leaves the claim over the budget.
    #[test]
    fn the_growth_guard_evicts_cached_prompts_before_refusing() {
        use std::sync::atomic::{AtomicU64, Ordering};
        use std::sync::Arc;
        let store = Arc::new(KvCacheStore::new(std::time::Duration::from_secs(60)));
        append(&store, 4, 4096);
        let live = store.occupancy().allocated_bytes;
        // 1000 bytes of cached prompts, of which an evictor can give back at
        // most 600.
        store.set_external_reserved(1000);
        let held = Arc::new(AtomicU64::new(1000));
        let evictor_calls = Arc::new(AtomicU64::new(0));
        {
            let ks = Arc::downgrade(&store);
            let held = held.clone();
            let calls = evictor_calls.clone();
            store.set_external_evictor(Box::new(move |needed| {
                calls.fetch_add(1, Ordering::Relaxed);
                let can = held.load(Ordering::Relaxed).saturating_sub(400);
                let freed = needed.min(can);
                let now = held.load(Ordering::Relaxed) - freed;
                held.store(now, Ordering::Relaxed);
                if let Some(ks) = ks.upgrade() {
                    ks.set_external_reserved(now);
                }
                freed
            }));
        }
        // Budget covers live + cached + a claim of 10 at 1 byte: fits, no eviction.
        assert!(store
            .claim_room(KvHome::Host, live + 1000 + 10, 10, "r")
            .is_ok());
        assert_eq!(evictor_calls.load(Ordering::Relaxed), 0);
        // 300 over: the evictor gives back 300 and the claim is admitted.
        assert!(store
            .claim_room(KvHome::Host, live + 1000 + 10 - 300, 10, "r")
            .is_ok());
        assert_eq!(store.occupancy().external_bytes, 700);
        // Now 800 over what remains: only 300 more can go; refused, naming
        // what the device still holds after the eviction.
        let refused = store
            .claim_room(KvHome::Host, live + 700 + 10 - 800, 10, "r")
            .expect_err("cannot fit even with every evictable byte gone");
        assert_eq!(store.occupancy().external_bytes, 400);
        assert_eq!(refused.in_use_bytes, live + 400);
        // With no evictor installed a store refuses as it always did.
        let bare = KvCacheStore::new(std::time::Duration::from_secs(60));
        bare.set_external_reserved(1000);
        assert!(bare.claim_room(KvHome::Host, 500, 10, "r").is_err());
    }

    /// A cache for `request`, last touched `idle` ago, holding `n` positions.
    fn conversation(store: &KvCacheStore, request: &str, n: usize, idle: std::time::Duration) {
        let mut entry = store.get_or_create("m", request, 1);
        let k = Tensor::zeros((1usize, 2, n, 4), DType::F32, &Device::Cpu).unwrap();
        let mut kv = new_kv_cache(4096, KvStorage::F32Mirrored, 0, None);
        kv.append(&k, &k.clone()).unwrap();
        entry.layers[0] = Some(kv);
        entry.last_accessed = std::time::Instant::now()
            .checked_sub(idle)
            .expect("a clock old enough to express the idle time");
    }

    /// **A budget is charged only the caches in its own memory** (#262). A
    /// card/processor split's card budget was weighed against every layer's
    /// cache, the processor-held layers' host memory included: a 30B split
    /// 26/22 on an 8 GB card was refused a 2,585-token prompt "for other
    /// conversations' 220 MB" with one live entry — its own host-side cache.
    /// Here the store holds only host-memory caches: the card has room for
    /// all of a claim, the machine for none of it.
    #[test]
    fn a_claim_is_weighed_against_the_caches_in_its_own_memory() {
        let store = KvCacheStore::new(std::time::Duration::from_secs(60));
        conversation(&store, "split", 256, std::time::Duration::ZERO);
        let occ = store.occupancy();
        let host = occ.allocated_bytes;
        assert!(host > 0);
        assert_eq!(occ.allocated_in(KvHome::Host), host);
        assert_eq!(occ.allocated_in(KvHome::Card), 0);
        // A budget the live cache already fills.
        let budget = host;
        assert!(
            store.claim_room(KvHome::Card, budget, 10, "split").is_ok(),
            "the card holds none of these caches"
        );
        let refused = store
            .claim_room(KvHome::Host, budget, 10, "split")
            .expect_err("the machine holds all of them");
        assert_eq!(refused.live_bytes, host);
    }

    /// **A prompt's promise is kept per memory, and drawn down only by what
    /// it has written THERE** (#262): a split's card promise must not be paid
    /// off by its host-side caches, nor its host promise by the card's.
    #[test]
    fn an_admission_promise_is_kept_and_drawn_down_per_memory() {
        let store = KvCacheStore::new(std::time::Duration::from_secs(60));
        store.record_prompt_admission("r", KvHome::Card, 4_000_000);
        store.record_prompt_admission("r", KvHome::Host, 2_000_000);
        assert_eq!(store.outstanding_admission_bytes(KvHome::Card), 4_000_000);
        assert_eq!(store.outstanding_admission_bytes(KvHome::Host), 2_000_000);
        // The request writes host-memory caches: only the host promise shrinks.
        conversation(&store, "r", 256, std::time::Duration::ZERO);
        let written = store.occupancy().allocated_in(KvHome::Host);
        assert!(written > 0 && written < 2_000_000);
        assert_eq!(store.outstanding_admission_bytes(KvHome::Card), 4_000_000);
        assert_eq!(
            store.outstanding_admission_bytes(KvHome::Host),
            2_000_000 - written
        );
        // Withdrawing one memory's promise keeps the other's.
        store.record_prompt_admission("r", KvHome::Card, 0);
        assert_eq!(store.outstanding_admission_bytes(KvHome::Card), 0);
        assert!(store.outstanding_admission_bytes(KvHome::Host) > 0);
        store.cleanup_request_id("r");
        assert_eq!(store.outstanding_admission_bytes(KvHome::Host), 0);
    }

    /// **A conversation that is over gives its room to one that is not**
    /// (report #005, 2026-10-07). A node serving a segment of another
    /// computer's request is never told the reply ended, and a worker never
    /// swept its store, so a live reply 71 s in was refused for 330 MB of
    /// "other conversations" that had finished. Two controls: a conversation
    /// touched recently keeps its cache and the claim is refused as before,
    /// and the claimant's own cache is never released, however old.
    #[test]
    fn a_finished_conversation_gives_its_room_to_a_live_one() {
        let gap = FINISHED_CONVERSATION_AFTER;
        let long_ago = gap + std::time::Duration::from_secs(1);
        let store = KvCacheStore::new(std::time::Duration::from_secs(600));
        conversation(&store, "ended", 4, long_ago);
        let one = store.occupancy().allocated_bytes;
        conversation(&store, "live", 4, std::time::Duration::ZERO);
        store.set_reserved_positions("ended", 1024);
        // Room for both caches only: a claim of 10 more bytes does not fit
        // until the ended conversation goes.
        assert!(store.claim_room(KvHome::Host, 2 * one, 10, "live").is_ok());
        let occ = store.occupancy();
        assert_eq!(
            occ.entries, 1,
            "the ended conversation's cache was released"
        );
        assert!(store.request_holds_state("m", "live"));
        assert!(!store.request_holds_state("m", "ended"));
        assert_eq!(
            store.reserved_positions("ended"),
            0,
            "its bookkeeping went with it"
        );

        // THE CONTROL: the same claim beside a conversation still going.
        let busy = KvCacheStore::new(std::time::Duration::from_secs(600));
        conversation(&busy, "other", 4, gap / 2);
        conversation(&busy, "live", 4, std::time::Duration::ZERO);
        assert!(busy.claim_room(KvHome::Host, 2 * one, 10, "live").is_err());
        assert_eq!(busy.occupancy().entries, 2);

        // The claimant's own cache stays, however long it was silent.
        let own = KvCacheStore::new(std::time::Duration::from_secs(600));
        conversation(&own, "live", 4, long_ago);
        assert_eq!(own.release_finished_conversations(gap, "live"), 0);
        assert!(own.request_holds_state("m", "live"));
    }

    /// The TTL sweep the worker now runs removes what nobody released.
    #[test]
    fn the_ttl_sweep_removes_a_conversation_nobody_released() {
        let store = KvCacheStore::new(std::time::Duration::from_secs(600));
        conversation(
            &store,
            "served-for-a-peer",
            4,
            std::time::Duration::from_secs(601),
        );
        conversation(&store, "recent", 4, std::time::Duration::ZERO);
        assert_eq!(store.cleanup_expired(), 1);
        assert!(store.request_holds_state("m", "recent"));
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
        let mut kv = LayerKv::with_capacity(2, 64, 64);
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
        let mut kv = LayerKv::with_capacity(2, 64, 64);
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

    /// Decode steps leave the mirror behind — nothing reads it until the next
    /// prompt chunk — and that chunk catches it up in one conversion, to
    /// exactly what an eager mirror would have held. Before, every decoded
    /// token paid four launches per layer to keep it in step.
    #[test]
    fn decode_steps_leave_the_mirror_behind_and_the_next_chunk_catches_it_up() {
        let dev = candle_core::Device::Cpu;
        let mut kv = LayerKv::with_capacity(2, 64, 64);
        kv.force_shadow_for_test();
        kv.append(&t(&dev, 1, 2, 3, 4), &t(&dev, 1, 2, 3, 4))
            .unwrap();
        for i in 0..4 {
            let step = (t(&dev, 1, 2, 1, 4) + (i as f64 + 100.0)).unwrap();
            kv.append(&step, &step).unwrap();
        }
        assert_eq!(kv.current_seq_len(), 7);
        assert_eq!(
            kv.shadow_len_for_test(),
            Some(3),
            "decode steps skip the mirror"
        );
        assert!(
            kv.flash_operands().is_none(),
            "a lagging mirror is never handed to the kernel"
        );

        let chunk = (t(&dev, 1, 2, 2, 4) + 500.0).unwrap();
        kv.append(&chunk, &chunk).unwrap();
        assert_eq!(kv.shadow_len_for_test(), Some(9));
        let expect = kv
            .k()
            .unwrap()
            .unwrap()
            .transpose(1, 2)
            .unwrap()
            .contiguous()
            .unwrap()
            .to_dtype(DType::F16)
            .unwrap();
        let (mk, _) = kv.flash_operands().expect("caught up");
        let diff = (mk.to_dtype(DType::F32).unwrap() - expect.to_dtype(DType::F32).unwrap())
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert_eq!(diff, 0.0, "the caught-up mirror must equal the f32 cache");
    }

    /// A step about to be RECORDED catches the mirror up first, outside the
    /// capture: afterwards the mirror holds exactly the f32 cache, so the
    /// recorded step's own append has nothing to catch up — nothing below
    /// `index_pos` is written inside a capture a refusal could throw away.
    #[test]
    fn a_lagging_mirror_is_caught_up_before_a_step_is_recorded() {
        let dev = candle_core::Device::Cpu;
        let mut kv = LayerKv::with_capacity(2, 64, 64);
        kv.force_shadow_for_test();
        kv.append(&t(&dev, 1, 2, 3, 4), &t(&dev, 1, 2, 3, 4))
            .unwrap();
        for i in 0..3 {
            let step = (t(&dev, 1, 2, 1, 4) + (i as f64 + 7.0)).unwrap();
            kv.append(&step, &step).unwrap();
        }
        assert_eq!(
            kv.shadow_len_for_test(),
            Some(3),
            "left behind by decode steps"
        );

        kv.catch_up_mirror();
        assert_eq!(kv.shadow_len_for_test(), Some(6));
        let expect = kv
            .k()
            .unwrap()
            .unwrap()
            .transpose(1, 2)
            .unwrap()
            .contiguous()
            .unwrap()
            .to_dtype(DType::F16)
            .unwrap();
        let (mk, _) = kv.flash_operands().expect("level with the cache");
        let diff = (mk.to_dtype(DType::F32).unwrap() - expect.to_dtype(DType::F32).unwrap())
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert_eq!(diff, 0.0);

        // The recorded step, and its rollback: the mirror and the cache both
        // go back to where the step began, together.
        let chunk = (t(&dev, 1, 2, 2, 4) + 50.0).unwrap();
        kv.append(&chunk, &chunk).unwrap();
        kv.truncate(6);
        assert_eq!(kv.current_seq_len(), 6);
        assert_eq!(kv.shadow_len_for_test(), Some(6));
        // Nothing to do on a mirror already level, or on a cache without one.
        kv.catch_up_mirror();
        assert_eq!(kv.shadow_len_for_test(), Some(6));
        let mut plain = LayerKv::with_capacity(2, 64, 64);
        plain.catch_up_mirror();
        assert_eq!(plain.shadow_len_for_test(), None);
    }

    #[test]
    fn reset_clears_both_representations() {
        let dev = candle_core::Device::Cpu;
        let mut kv = LayerKv::with_capacity(2, 64, 64);
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
            let mut slot = LayerKv::with_capacity(2, 64, 64);
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

    /// A worker refuses a forward that continues a conversation when this
    /// answers false (`model_worker::forward_lacks_its_conversation`), so it
    /// must be false for every way of holding nothing — and a worker restarted
    /// mid-reply is exactly the "no entry" case (`docs/FUTURE_WORK.md` #93).
    #[test]
    fn a_request_holds_state_only_once_something_is_written() {
        let dev = candle_core::Device::Cpu;
        let store = KvCacheStore::new(std::time::Duration::from_secs(60));
        assert!(!store.request_holds_state("m", "r"), "no entry at all");
        {
            let _created = store.get_or_create("m", "r", 1);
        }
        assert!(
            !store.request_holds_state("m", "r"),
            "an entry the forward created but never wrote holds nothing"
        );
        {
            let mut e = store.get_or_create("m", "r", 1);
            let mut slot = LayerKv::with_capacity(2, 64, 64);
            let k = t(&dev, 1, 2, 3, 4);
            slot.append(&k, &k).unwrap();
            e.layers[0] = Some(slot);
        }
        assert!(store.request_holds_state("m", "r"));
        assert!(
            !store.request_holds_state("m", "someone-else"),
            "another request's cache is not this one's"
        );
        assert!(
            !store.request_holds_state("other-segment", "r"),
            "keyed by model segment as well as request"
        );
        store.clear_request("m", "r");
        assert!(!store.request_holds_state("m", "r"));
    }

    /// Distinct values per position, so a truncation that kept the wrong
    /// positions — or read back stale ones — is visible in the numbers.
    fn filled(dev: &candle_core::Device, positions: usize) -> LayerKv {
        let mut kv = LayerKv::with_capacity(2, 64, 64);
        kv.force_shadow_for_test();
        let k = t(dev, 1, 2, positions, 4);
        let v = (t(dev, 1, 2, positions, 4) + 1000.0).unwrap();
        kv.append(&k, &v).unwrap();
        kv
    }

    fn flat(x: &Tensor) -> Vec<f32> {
        x.flatten_all().unwrap().to_vec1::<f32>().unwrap()
    }

    /// The in-place truncation must answer exactly what the copying one did:
    /// the same kept prefix, and the next append landing right after it.
    /// Compared against the old implementation rather than against a
    /// hand-computed expectation, so the equivalence is between the two
    /// things that actually changed places.
    /// A recurrent state cannot be truncated, so an entry holding one refuses
    /// to shorten its attention caches rather than let the two disagree —
    /// except by the one step a card's state keeps the state before
    /// (`SsmState::rewind_to`), which a refused CUDA graph capture takes back.
    #[test]
    fn an_entry_with_recurrent_state_refuses_to_be_truncated() {
        let dev = candle_core::Device::Cpu;
        let slot = || crate::inference::layers::SsmSlot {
            conv_state: Tensor::zeros((1, 4, 3), candle_core::DType::F32, &dev).unwrap(),
            recurrent_state: Tensor::zeros((1, 2, 4, 4), candle_core::DType::F32, &dev).unwrap(),
        };
        let mut entry = KvCacheEntry {
            layers: vec![Some(filled(&dev, 8)), None],
            ssm_states: vec![None, Some(SsmState::new(slot(), 8))],
            last_accessed: std::time::Instant::now(),
        };
        assert!(entry.truncate_to(5).is_err());
        assert_eq!(
            entry.layers[0].as_ref().unwrap().current_seq_len(),
            8,
            "refused = untouched"
        );

        // A card's state after a one-position step from 7: back to 7 only.
        let mut card = SsmState::new(slot(), 7);
        card.advance(slot(), 8);
        entry.ssm_states[1] = Some(card);
        assert!(entry.truncate_to(6).is_err());
        entry.truncate_to(7).unwrap();
        assert_eq!(entry.ssm_states[1].as_ref().unwrap().positions(), 7);
        assert_eq!(entry.layers[0].as_ref().unwrap().current_seq_len(), 7);

        entry.ssm_states[1] = None;
        entry.truncate_to(5).unwrap();
    }

    /// A captured step allocates nothing, so a recurrent layer holds only once
    /// its next step's buffers exist.
    #[test]
    fn a_recurrent_layer_holds_a_captured_step_only_with_its_next_buffers() {
        let dev = candle_core::Device::Cpu;
        let slot = || crate::inference::layers::SsmSlot {
            conv_state: Tensor::zeros((1, 4, 3), candle_core::DType::F32, &dev).unwrap(),
            recurrent_state: Tensor::zeros((1, 2, 4, 4), candle_core::DType::F32, &dev).unwrap(),
        };
        let store = KvCacheStore::new(std::time::Duration::from_secs(600));
        let key = KvCacheStore::cache_key("m", "r");
        {
            let mut entry = store.get_or_create_keyed(&key, 2);
            entry.layers = vec![Some(filled(&dev, 8)), None];
            entry.ssm_states = vec![None, Some(SsmState::new(slot(), 8))];
        }
        assert!(!store.every_cache_holds(&key, 8, |_| false));
        {
            let mut entry = store.get_or_create_keyed(&key, 2);
            entry.ssm_states[1].as_mut().unwrap().advance(slot(), 9);
        }
        assert!(store.every_cache_holds(&key, 8, |_| false));
    }

    /// A Gemma 4 E2B/E4B layer that attends an earlier layer's cache keeps
    /// none of its own: its empty slot holds a captured step, where an empty
    /// slot of any other layer is a cache still to be built (#295).
    #[test]
    fn a_layer_that_keeps_no_cache_holds_a_captured_step() {
        let dev = candle_core::Device::Cpu;
        let store = KvCacheStore::new(std::time::Duration::from_secs(600));
        let key = KvCacheStore::cache_key("m", "r");
        {
            let mut entry = store.get_or_create_keyed(&key, 3);
            entry.layers = vec![Some(filled(&dev, 8)), None, Some(filled(&dev, 8))];
        }
        assert!(
            !store.every_cache_holds(&key, 8, |_| false),
            "an empty slot of a layer that keeps a cache allocates on its first append"
        );
        assert!(store.every_cache_holds(&key, 8, |slot| slot == 1));
        assert!(
            !store.every_cache_holds(&key, 8, |slot| slot == 0),
            "a slot is excused only by its own layer"
        );
    }

    #[test]
    fn an_in_place_truncation_keeps_the_prefix_bitwise_and_the_next_append_continues_from_it() {
        let dev = candle_core::Device::Cpu;
        let mut by_copy = KvCacheEntry {
            layers: vec![Some(filled(&dev, 8))],
            ssm_states: vec![None],
            last_accessed: std::time::Instant::now(),
        };
        let mut in_place = KvCacheEntry {
            layers: vec![Some(filled(&dev, 8))],
            ssm_states: vec![None],
            last_accessed: std::time::Instant::now(),
        };
        let original_k = flat(&in_place.layers[0].as_ref().unwrap().k().unwrap().unwrap());

        by_copy.truncate_to_with(TruncateMode::Copy, 5).unwrap();
        in_place.truncate_to_with(TruncateMode::InPlace, 5).unwrap();

        for (label, e) in [("copy", &by_copy), ("in place", &in_place)] {
            let kv = e.layers[0].as_ref().unwrap();
            assert_eq!(kv.current_seq_len(), 5, "{label}");
            assert_eq!(kv.shadow_len_for_test(), Some(5), "{label}: mirror");
            // BHSD with h=2, d=4: the kept prefix is the first 5 positions of
            // each head, i.e. NOT a contiguous prefix of the flattened buffer.
            let k = kv.k().unwrap().unwrap();
            assert_eq!(k.dims(), &[1, 2, 5, 4], "{label}");
            let want: Vec<f32> = (0..2)
                .flat_map(|h| (0..5 * 4).map(move |i| (h * 8 * 4 + i) as f32))
                .collect();
            assert_eq!(flat(&k), want, "{label}: kept the wrong positions");
        }
        assert_ne!(
            flat(&in_place.layers[0].as_ref().unwrap().k().unwrap().unwrap()),
            original_k,
            "the control: truncation must have changed what k() returns"
        );

        // The next append continues from position 5 on both, and the two
        // caches read back identically afterwards — including the mirror the
        // flash kernel would be handed.
        let more_k = (t(&dev, 1, 2, 2, 4) + 5000.0).unwrap();
        let more_v = (t(&dev, 1, 2, 2, 4) + 6000.0).unwrap();
        for e in [&mut by_copy, &mut in_place] {
            e.layers[0]
                .as_mut()
                .unwrap()
                .append(&more_k, &more_v)
                .unwrap();
        }
        let a = by_copy.layers[0].as_ref().unwrap();
        let b = in_place.layers[0].as_ref().unwrap();
        assert_eq!(a.current_seq_len(), 7);
        assert_eq!(b.current_seq_len(), 7);
        assert_eq!(
            flat(&a.k().unwrap().unwrap()),
            flat(&b.k().unwrap().unwrap())
        );
        assert_eq!(
            flat(&a.v().unwrap().unwrap()),
            flat(&b.v().unwrap().unwrap())
        );
        let (ak, av) = a.flash_operands().unwrap();
        let (bk, bv) = b.flash_operands().unwrap();
        assert_eq!(
            flat(&ak.to_dtype(DType::F32).unwrap()),
            flat(&bk.to_dtype(DType::F32).unwrap())
        );
        assert_eq!(
            flat(&av.to_dtype(DType::F32).unwrap()),
            flat(&bv.to_dtype(DType::F32).unwrap())
        );
        // And the appended positions are where they should be.
        let tail = b.k().unwrap().unwrap().narrow(2, 5, 2).unwrap();
        assert_eq!(flat(&tail), flat(&more_k));
    }

    /// The whole point: a rejected draft must not copy or reallocate the
    /// cache. Pinned on the buffer's identity, with the copying mode as the
    /// control proving the check can see a reallocation when one happens.
    #[test]
    fn an_in_place_truncation_allocates_nothing() {
        let dev = candle_core::Device::Cpu;
        let buffer_id = |e: &KvCacheEntry| {
            let kv = e.layers[0].as_ref().unwrap();
            let k = kv.k_cache().all_data().as_ref().unwrap().id();
            let v = kv.v_cache().all_data().as_ref().unwrap().id();
            (k, v)
        };
        let mut e = KvCacheEntry {
            layers: vec![Some(filled(&dev, 8))],
            ssm_states: vec![None],
            last_accessed: std::time::Instant::now(),
        };
        let before = buffer_id(&e);
        e.truncate_to_with(TruncateMode::InPlace, 3).unwrap();
        assert_eq!(
            buffer_id(&e),
            before,
            "in-place truncation replaced the buffer"
        );

        let mut control = KvCacheEntry {
            layers: vec![Some(filled(&dev, 8))],
            ssm_states: vec![None],
            last_accessed: std::time::Instant::now(),
        };
        let before = buffer_id(&control);
        control.truncate_to_with(TruncateMode::Copy, 3).unwrap();
        assert_ne!(
            buffer_id(&control),
            before,
            "control: the copying mode reallocates, and the check must see it"
        );
    }

    /// A truncation past the end is a no-op, and to zero is a length-only
    /// reset that a later append recovers from.
    #[test]
    fn truncation_bounds_are_no_ops_or_empties_not_errors() {
        let dev = candle_core::Device::Cpu;
        let mut kv = filled(&dev, 4);
        kv.truncate(10);
        assert_eq!(kv.current_seq_len(), 4);
        kv.truncate(0);
        assert_eq!(kv.current_seq_len(), 0);
        assert_eq!(kv.shadow_len_for_test(), Some(0));
        let k = t(&dev, 1, 2, 2, 4);
        kv.append(&k, &k).unwrap();
        assert_eq!(kv.current_seq_len(), 2);
        assert_eq!(flat(&kv.k().unwrap().unwrap()), flat(&k));
    }

    #[test]
    fn the_production_truncation_is_the_in_place_one() {
        // The env switch exists for measurement; unset, the fast path is the
        // one that runs.
        if std::env::var("SWARMLLM_KV_TRUNCATE").is_err() {
            assert_eq!(truncate_mode(), TruncateMode::InPlace);
        }
    }

    #[test]
    fn a_drifted_mirror_is_refused_rather_than_served() {
        // The last line of defence: if the two ever disagree, the flash path
        // must fall back to converting the f32 cache, not hand the kernel a
        // history of the wrong length.
        let dev = candle_core::Device::Cpu;
        let mut kv = LayerKv::with_capacity(2, 64, 64);
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
        let mut kv = LayerKv::with_capacity(1, 64, 64);
        kv.force_shadow_for_test();
        assert!(kv.flash_operands().is_none());
    }

    #[test]
    fn a_cache_that_should_not_mirror_sheds_one_it_inherited() {
        // Prefix-cache hydration builds a cache from a snapshot, which carries
        // `n_kv_head` in its tensor shapes but not `n_head` — so it cannot tell
        // GQA from MHA and may mirror a model whose decode never reads one. The
        // forward pass knows both counts and corrects it; this is that
        // correction. Without it an MHA model resuming a cached conversation
        // pays ~3-8% per token and 50% more KV memory for a copy nothing reads,
        // while the same model starting fresh does not — the same model
        // behaving differently depending on how the conversation began.
        let dev = candle_core::Device::Cpu;
        let mut kv = LayerKv::with_capacity(2, 64, 64);
        kv.force_shadow_for_test();
        let k = t(&dev, 1, 2, 4, 4);
        kv.append(&k, &k).unwrap();
        assert_eq!(kv.shadow_len_for_test(), Some(4), "precondition: mirroring");

        kv.conform_to(KvStorage::F32).unwrap();
        assert_eq!(kv.shadow_len_for_test(), None, "the mirror was not dropped");

        // And it must stay off — a later append must not resurrect it.
        kv.append(&k, &k).unwrap();
        assert_eq!(kv.shadow_len_for_test(), None, "mirroring came back on");
        assert!(kv.flash_operands().is_none());
    }

    #[test]
    fn turning_mirroring_on_late_does_not_create_a_useless_mirror() {
        // The `true` direction is deliberately inert: a mirror started against a
        // cache that already holds positions can never catch up, because it only
        // appends new ones. It would fail `flash_operands`' length check forever
        // while still costing memory and a conversion per token.
        let dev = candle_core::Device::Cpu;
        let mut kv = LayerKv::with_capacity(2, 64, 64);
        let k = t(&dev, 1, 2, 4, 4);
        kv.append(&k, &k).unwrap();
        kv.conform_to(KvStorage::F32Mirrored).unwrap();
        assert_eq!(
            kv.shadow_len_for_test(),
            None,
            "a mirror enabled mid-conversation could never match the cache"
        );
    }

    // ── The half cache (FUTURE_WORK #194) ──
    //
    // The processor never keeps one in production (`layers::kv_storage`), but
    // the cache's own bookkeeping is device-independent, so it is pinned here
    // where every build runs it. The card's two kernels have their own tests
    // under `--features cuda` (`decode_attn`, `kv_write`).

    /// Values f16 rounding actually changes: `t`'s integers up to 2048 are
    /// exact in f16, which would hide a missing conversion.
    fn noisy(b: usize, h: usize, s: usize, d: usize) -> Tensor {
        Tensor::randn(0f32, 1.0, (b, h, s, d), &candle_core::Device::Cpu).unwrap()
    }

    fn flat32(t: &Tensor) -> Vec<f32> {
        t.to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
    }

    /// What a half cache must hold for `t`: each value rounded to f16 once.
    fn rounded(t: &Tensor) -> Vec<f32> {
        flat32(&t.to_dtype(DType::F16).unwrap())
    }

    #[test]
    fn the_half_cache_holds_f16_and_answers_with_it() {
        let mut kv = LayerKv::with_storage(2, 8, 8, KvStorage::F16);
        let (k1, v1) = (noisy(1, 2, 5, 4), noisy(1, 2, 5, 4));
        let (k, v) = kv.append(&k1, &v1).unwrap();
        assert!(kv.holds_half());
        assert_eq!((k.dtype(), v.dtype()), (DType::F16, DType::F16));
        assert_eq!(flat32(&k), rounded(&k1));
        assert_eq!(flat32(&v), rounded(&v1));
        assert_ne!(
            rounded(&k1),
            flat32(&k1),
            "precondition: the test values must not be exact in f16"
        );

        // Past its first allocation (8 positions): it grows, in f16.
        let (k2, v2) = (noisy(1, 2, 6, 4), noisy(1, 2, 6, 4));
        let (k, _) = kv.append(&k2, &v2).unwrap();
        assert_eq!(k.dims4().unwrap(), (1, 2, 11, 4));
        assert_eq!(k.dtype(), DType::F16);
        assert_eq!(flat32(&k.narrow(2, 0, 5).unwrap()), rounded(&k1));
        assert_eq!(flat32(&k.narrow(2, 5, 6).unwrap()), rounded(&k2));

        // No mirror, ever: the half cache is what flash reads.
        assert!(kv.flash_operands().is_none());
        assert_eq!(kv.all_caches().len(), 2);

        // A rejected draft is two lengths moving, and the next step writes
        // over what was dropped.
        kv.truncate(3);
        let k3 = noisy(1, 2, 1, 4);
        let (k, _) = kv.append(&k3, &k3).unwrap();
        assert_eq!(k.dim(2).unwrap(), 4);
        assert_eq!(flat32(&k.narrow(2, 3, 1).unwrap()), rounded(&k3));
    }

    /// The point of it: the same positions cost a third of the f32 cache plus
    /// its mirror, and half of the f32 cache alone — as the store counts them.
    #[test]
    fn the_half_cache_is_counted_at_its_own_width() {
        let bytes = |kv: &LayerKv| -> u64 {
            kv.all_caches()
                .iter()
                .filter_map(|c| c.all_data().as_ref())
                .map(|t| t.elem_count() as u64 * t.dtype().size_in_bytes() as u64)
                .sum()
        };
        let k = noisy(1, 2, 16, 4);
        let mut half = LayerKv::with_storage(2, 16, 16, KvStorage::F16);
        let mut full = LayerKv::with_storage(2, 16, 16, KvStorage::F32);
        half.append(&k, &k).unwrap();
        full.append(&k, &k).unwrap();
        assert_eq!(bytes(&half) * 2, bytes(&full));
        assert_eq!(
            bytes(&half),
            16 * 2 * 4 * 2 * KvStorage::F16.bytes_per_element(),
            "two tensors of 16 positions x 2 heads x 4, at 2 bytes"
        );
    }

    /// A cache built by something that could not know how its layer keeps it —
    /// prefix-cache hydration — is converted, history and all, at the next
    /// forward, and goes on in the new dtype.
    #[test]
    fn a_cache_conforms_to_the_storage_its_layer_keeps() {
        let mut kv = LayerKv::with_capacity(2, 16, 16);
        let (k1, v1) = (noisy(1, 2, 5, 4), noisy(1, 2, 5, 4));
        kv.append(&k1, &v1).unwrap();
        assert!(
            !kv.holds_half(),
            "a cache that follows its source holds f32"
        );

        kv.conform_to(KvStorage::F16).unwrap();
        assert!(kv.holds_half());
        assert_eq!(kv.current_seq_len(), 5);
        assert_eq!(kv.k_cache().max_seq_len(), 16, "the reservation is kept");
        assert_eq!(flat32(&kv.k().unwrap().unwrap()), rounded(&k1));
        assert_eq!(flat32(&kv.v().unwrap().unwrap()), rounded(&v1));

        let k2 = noisy(1, 2, 1, 4);
        let (k, _) = kv.append(&k2, &k2).unwrap();
        assert_eq!(k.dtype(), DType::F16);
        assert_eq!(flat32(&k.narrow(2, 5, 1).unwrap()), rounded(&k2));

        // Already right: nothing moves.
        let before = kv.k_cache().all_data().as_ref().unwrap().id();
        kv.conform_to(KvStorage::F16).unwrap();
        assert_eq!(kv.k_cache().all_data().as_ref().unwrap().id(), before);

        // And back, should a layer ever keep f32: what the half cache held,
        // widened — the rounding cannot be undone, and nothing pretends to.
        kv.conform_to(KvStorage::F32).unwrap();
        assert!(!kv.holds_half());
        let k = kv.k().unwrap().unwrap();
        assert_eq!(k.dtype(), DType::F32);
        assert_eq!(flat32(&k.narrow(2, 0, 5).unwrap()), rounded(&k1));
    }

    #[test]
    fn an_empty_cache_conforms_without_allocating() {
        let mut kv = LayerKv::with_capacity(2, 16, 16);
        kv.conform_to(KvStorage::F16).unwrap();
        assert!(kv.holds_half(), "the next allocation is the half cache");
        assert!(kv.k_cache().all_data().is_none());
        let k = noisy(1, 2, 3, 4);
        let (k_out, _) = kv.append(&k, &k).unwrap();
        assert_eq!(k_out.dtype(), DType::F16);
    }

    /// The forward's one way in: a fresh slot is built the way the layer keeps
    /// it, a hydrated one is conformed, and position 0 starts over.
    #[test]
    fn a_forward_appends_through_one_function_that_conforms_the_cache() {
        use crate::inference::layers::append_to_cache;
        let (k1, v1) = (noisy(1, 2, 5, 4), noisy(1, 2, 5, 4));

        let mut fresh: Option<LayerKv> = None;
        let (k, _) =
            append_to_cache(&mut fresh, &k1, &v1, 0, KvStorage::F16, 4096, 0, None).unwrap();
        assert_eq!(k.dtype(), DType::F16);

        // What hydration leaves: an f32 history in a cache that follows its
        // source. The next step conforms it and lands in the half cache.
        let mut hydrated = LayerKv::with_capacity(2, 16, 16);
        hydrated.append(&k1, &v1).unwrap();
        let mut slot = Some(hydrated);
        let k2 = noisy(1, 2, 1, 4);
        let (k, _) =
            append_to_cache(&mut slot, &k2, &k2, 5, KvStorage::F16, 4096, 0, None).unwrap();
        assert_eq!(k.dtype(), DType::F16);
        assert_eq!(k.dim(2).unwrap(), 6);
        assert_eq!(flat32(&k.narrow(2, 0, 5).unwrap()), rounded(&k1));

        // Position 0 is a new conversation in the same slot.
        let (k, _) =
            append_to_cache(&mut slot, &k2, &k2, 0, KvStorage::F16, 4096, 0, None).unwrap();
        assert_eq!(k.dim(2).unwrap(), 1);
    }

    /// The matmul path reads f32, and a half cache that reaches it — forced
    /// standard attention, a decode step the card kernel declines — is widened
    /// first: the same answer as the f32 cache holding the rounded values.
    #[test]
    fn attention_over_a_half_cache_widens_it_where_it_needs_f32() {
        use crate::inference::layers::standard_attention;
        let (n_head, n_kv_head, d, s_len) = (8usize, 2usize, 16usize, 37usize);
        let q = noisy(1, n_head, 1, d);
        let (k, v) = (noisy(1, n_kv_head, s_len, d), noisy(1, n_kv_head, s_len, d));
        let (k16, v16) = (
            k.to_dtype(DType::F16).unwrap(),
            v.to_dtype(DType::F16).unwrap(),
        );
        let (k_wide, v_wide) = (
            k16.to_dtype(DType::F32).unwrap(),
            v16.to_dtype(DType::F32).unwrap(),
        );
        let got = standard_attention(&q, &k16, &v16, None, d, n_head, n_kv_head, None, 0).unwrap();
        let want =
            standard_attention(&q, &k_wide, &v_wide, None, d, n_head, n_kv_head, None, 0).unwrap();
        assert_eq!(got.dtype(), DType::F32);
        assert_eq!(flat32(&got), flat32(&want));
    }

    // ---- A sliding-window layer keeps its window (FUTURE_WORK #281) ----

    use crate::inference::layers::append_to_cache;
    use candle_core::IndexOp;

    /// Positions `from..from + len` as K rows ([1, 2, len, 4]) whose every
    /// value is the position, so a read-back says which positions it holds.
    fn rows_at(from: usize, len: usize) -> Tensor {
        Tensor::arange(from as f32, (from + len) as f32, &Device::Cpu)
            .unwrap()
            .reshape((1, 1, len, 1))
            .unwrap()
            .broadcast_as((1, 2, len, 4))
            .unwrap()
            .contiguous()
            .unwrap()
    }

    /// The positions a cache reads back, from its first row of head 0.
    fn positions_held(kv: &LayerKv) -> Vec<usize> {
        let k = kv.k().unwrap().unwrap();
        let held = k.dim(2).unwrap();
        k.narrow(1, 0, 1)
            .unwrap()
            .narrow(3, 0, 1)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .into_iter()
            .take(held)
            .map(|p| p as usize)
            .collect()
    }

    /// A windowed cache of a 128-position context, fed `positions` in chunks
    /// of `chunk` — the slide on unless `no_slide`.
    fn windowed_cache(window: usize, positions: usize, chunk: usize, no_slide: bool) -> LayerKv {
        NO_SLIDE_IN_TEST.with(|c| c.set(no_slide));
        let mut slot = None;
        let mut pos = 0;
        while pos < positions {
            let len = chunk.min(positions - pos);
            let k = rows_at(pos, len);
            append_to_cache(&mut slot, &k, &k, pos, KvStorage::F32, 128, 0, Some(window)).unwrap();
            pos += len;
        }
        NO_SLIDE_IN_TEST.with(|c| c.set(false));
        slot.unwrap()
    }

    /// A windowed layer's cache never grows past its window and one quantum:
    /// when full it moves its newest positions to the front, keeping at least
    /// the window and a rewind's worth, and reads back exactly the newest
    /// positions in order. Pieces of 64 are the case that grew a quantum too
    /// far before the growth was capped.
    #[test]
    fn a_windowed_cache_keeps_its_newest_positions_in_one_buffer() {
        let (window, n) = (4, 400);
        let keep = positions_kept_for_window(window);
        for chunk in [1, 7, 64] {
            let kv = windowed_cache(window, n, chunk, false);
            assert_eq!(kv.current_seq_len(), n, "{chunk}: the length is absolute");
            assert!(kv.held_from() > 0, "{chunk}: 400 positions in a 128 buffer");
            assert!(
                kv.k_cache().max_seq_len() <= keep + 128,
                "{chunk}: the buffer grew to {}, past the window and one quantum",
                kv.k_cache().max_seq_len()
            );
            let held = positions_held(&kv);
            assert!(held.len() >= keep, "{chunk}: kept {} < {keep}", held.len());
            let want: Vec<usize> = (kv.held_from()..n).collect();
            assert_eq!(held, want, "{chunk}: the newest positions, in order");
            assert_eq!(kv.serves_from(), kv.held_from() + window - 1);
        }

        // The control: with the slide off, every position is kept and the
        // buffer grows to hold them.
        let kv = windowed_cache(window, n, 7, true);
        assert_eq!(kv.held_from(), 0);
        assert_eq!(positions_held(&kv), (0..n).collect::<Vec<_>>());
        assert!(kv.k_cache().max_seq_len() >= n);
    }

    /// The head-room guard's claim is what the forward's append allocates —
    /// read off every layer's buffer before it, a windowed layer's slide
    /// included (FUTURE_WORK #281) — and a windowed buffer never passes
    /// [`most_positions_a_window_holds`] for the widest forward it was handed,
    /// nor the buffer of a layer that keeps every position. Prompt chunks of
    /// several widths, decode steps, forwards wider than a quantum, and
    /// reservations that start the buffers below and above the window's size.
    /// The third layer keeps an f16 mirror (forced: this runs on the
    /// processor), which decode steps leave behind and a chunk catches up —
    /// claimed as that, never as its storage's 6 bytes a position.
    #[test]
    fn a_forwards_claim_is_what_its_append_allocates() {
        let (window, context, quantum) = (100, 8192, 512);
        let forwards = [300, 1, 1, 64, 700, 1, 1, 1, 128, 1300, 1, 2, 8, 512, 1];
        for reserve in [0, 1024, 4096] {
            let store = KvCacheStore::new(std::time::Duration::from_secs(600));
            let key = KvCacheStore::cache_key("m", "r");
            let (mut pos, mut widest) = (0, 0);
            for (step, &len) in forwards.iter().cycle().take(40).enumerate() {
                if pos + len > context {
                    break;
                }
                let predicted = store.bytes_a_forward_adds(&key, pos, len, KvHome::Host);
                let before = store.allocated_bytes_of(&key, KvHome::Host);
                {
                    let mut entry = store.get_or_create("m", "r", 3);
                    let k = rows_at(pos, len);
                    let layouts = [
                        (KvStorage::F32, Some(window)),
                        (KvStorage::F32, None),
                        (KvStorage::F32Mirrored, None),
                    ];
                    for (slot, (storage, w)) in entry.layers.iter_mut().zip(layouts) {
                        append_to_cache(slot, &k, &k, pos, storage, context, reserve, w).unwrap();
                    }
                    if step == 0 {
                        entry.layers[2].as_mut().unwrap().force_shadow_for_test();
                    }
                }
                let after = store.allocated_bytes_of(&key, KvHome::Host);
                match predicted {
                    Some(claim) => assert_eq!(
                        after - before,
                        claim,
                        "reserve {reserve}, step {step}: {len} at {pos}"
                    ),
                    None => assert_eq!(step, 0, "only the first forward has no buffers"),
                }
                pos += len;
                widest = widest.max(len);
                let entry = store.get_entry(&key).unwrap();
                let windowed = entry.layers[0].as_ref().unwrap().k_cache().max_seq_len();
                let global = entry.layers[1].as_ref().unwrap().k_cache().max_seq_len();
                let bound = most_positions_a_window_holds(window, quantum, widest);
                assert!(
                    windowed <= bound,
                    "reserve {reserve}, step {step}: {windowed} past {bound}"
                );
                assert!(
                    windowed <= global,
                    "reserve {reserve}, step {step}: {windowed} past {global}"
                );
            }
            let entry = store.get_entry(&key).unwrap();
            assert!(entry.layers[0].as_ref().unwrap().held_from() > 0, "it slid");
            assert!(pos > 4000, "a long conversation: {pos}");
            let mirrored = entry.layers[2].as_ref().unwrap();
            assert!(
                mirrored.shadow_len_for_test().unwrap() > 0,
                "the mirror took chunks"
            );
        }
    }

    /// A layer with no window keeps every position, slide or no slide.
    #[test]
    fn a_layer_without_a_window_keeps_every_position() {
        let mut slot = None;
        for pos in (0..300).step_by(10) {
            let k = rows_at(pos, 10);
            append_to_cache(&mut slot, &k, &k, pos, KvStorage::F32, 128, 0, None).unwrap();
        }
        let kv = slot.unwrap();
        assert_eq!(kv.held_from(), 0);
        assert_eq!(positions_held(&kv), (0..300).collect::<Vec<_>>());
    }

    /// A rewind the window still answers is taken — and the next write lands
    /// right after it; one past it is refused rather than served from keys
    /// the cache no longer holds. Both truncation modes agree.
    #[test]
    fn a_slid_cache_takes_a_rewind_its_window_still_answers_and_refuses_one_past_it() {
        let window = 4;
        for mode in [TruncateMode::InPlace, TruncateMode::Copy] {
            let kv = windowed_cache(window, 400, 7, false);
            let (from, serves) = (kv.held_from(), kv.serves_from());
            let mut entry = KvCacheEntry {
                layers: vec![Some(kv)],
                ssm_states: vec![None],
                last_accessed: std::time::Instant::now(),
            };
            let err = entry.truncate_to_with(mode, serves - 1).unwrap_err();
            assert!(
                err.to_string().contains("sliding-window"),
                "{mode:?}: {err}"
            );
            assert_eq!(entry.layers[0].as_ref().unwrap().current_seq_len(), 400);

            entry.truncate_to_with(mode, serves).unwrap();
            let kv = entry.layers[0].as_mut().unwrap();
            assert_eq!(kv.current_seq_len(), serves, "{mode:?}");
            assert_eq!(kv.held_from(), from, "{mode:?}: the start is kept");
            assert_eq!(positions_held(kv), (from..serves).collect::<Vec<_>>());
            let k = rows_at(serves, 3);
            kv.append(&k, &k).unwrap();
            assert_eq!(
                positions_held(kv),
                (from..serves + 3).collect::<Vec<_>>(),
                "{mode:?}: the next write continues from the rewind"
            );
        }
    }

    /// A slid cache reaches past its buffer's length by where it starts: a
    /// step inside that allocates nothing (a CUDA graph may capture it) and the
    /// head-room guard charges nothing for it.
    #[test]
    fn a_slid_cache_reaches_from_where_it_starts() {
        let store = KvCacheStore::new(std::time::Duration::from_secs(60));
        let key = KvCacheStore::cache_key("m", "r");
        let kv = windowed_cache(4, 400, 1, false);
        let reach = kv.held_from() + kv.k_cache().max_seq_len();
        assert!(reach > 400);
        store.get_or_create_keyed(&key, 1).layers[0] = Some(kv);
        assert_eq!(store.allocated_positions(&key), reach);
        assert!(store.every_cache_holds(&key, 401, |_| false));
        assert!(!store.every_cache_holds(&key, reach + 1, |_| false));
    }

    /// The slide copies the newest positions, so it never runs on a cache
    /// that keeps an f16 mirror in step by absolute position.
    #[test]
    fn a_mirrored_cache_never_slides() {
        let mut slot = None;
        for pos in (0..300).step_by(10) {
            let k = rows_at(pos, 10);
            append_to_cache(
                &mut slot,
                &k,
                &k,
                pos,
                KvStorage::F32Mirrored,
                128,
                0,
                Some(4),
            )
            .unwrap();
        }
        let kv = slot.unwrap();
        assert_eq!(kv.held_from(), 0);
        assert_eq!(positions_held(&kv), (0..300).collect::<Vec<_>>());
    }

    /// Converted to the half cache mid-conversation (a hydrated cache meeting a
    /// card layer), a slid cache keeps where it starts and goes on sliding.
    #[test]
    fn a_slid_cache_converted_to_half_stays_slid() {
        let mut slot = Some(windowed_cache(4, 400, 7, false));
        let from = slot.as_ref().unwrap().held_from();
        let k = rows_at(400, 1);
        append_to_cache(&mut slot, &k, &k, 400, KvStorage::F16, 128, 0, Some(4)).unwrap();
        let kv = slot.as_ref().unwrap();
        assert!(kv.holds_half());
        assert!(kv.held_from() >= from);
        let k = kv.k().unwrap().unwrap().to_dtype(DType::F32).unwrap();
        let held = k.dim(2).unwrap();
        assert_eq!(kv.held_from() + held, 401);
        let last: f32 = k.i((0, 0, held - 1, 0)).unwrap().to_scalar().unwrap();
        assert_eq!(last, 400.0);
    }
}
