//! VRAM-aware sizing of the KV cache and RoPE tables.
//!
//! `max_seq_len` bounds how large a KV cache can GROW, and the conversation
//! guard in `forward_inner_impl` is what holds it there. So sizing it from the
//! GGUF's declared `context_length` lets a long conversation grow until it
//! exhausts the card.
//!
//! **This module's premise changed on 2026-08-07 and its conclusion did not.**
//! Until then a cache allocated its whole `[B, H, max_seq_len, D]` buffer on
//! the first append, so `max_seq_len` was not a limit but an immediate
//! *allocation* — every user charged the worst case from token one.
//! `layers::new_kv_cache` now passes a growth quantum instead, so the
//! allocation tracks the conversation. That makes the numbers below a *ceiling*
//! rather than a reservation, and makes this module MORE load-bearing, not
//! less: it is now the only thing standing between on-demand growth and an OOM
//! part-way through a long conversation.
//!
//! That is ruinous on modern long-context models. Measured on an RTX 3070
//! (8 GB) against `llama-3.2-1b-instruct-q8-0` — 1.3 GB of weights, but
//! `context_length = 131072`:
//!
//! | | KV sized to declared ctx | KV sized to 4096 |
//! |---|---|---|
//! | VRAM | 7958 MiB (97% of card) | 3110 MiB |
//! | Throughput | 3.7 tok/s | 46–59 tok/s |
//!
//! Same model, same GPU, ~14x. Two models of near-identical size can differ by
//! 93x in KV cache purely because one declares 2048 and the other 131072
//! (tinyllama-1.1b: 88 MiB; llama-3.2-1b: 8 GiB).
//!
//! Where it does not simply OOM, it silently degrades: the driver's
//! system-memory fallback (WSL2 and Windows) absorbs the overflow, so the
//! model still answers, `device=Cuda` still appears in the log, and only
//! external GPU profiling shows the card sitting idle. Reported 2026-07-29
//! as "GPU loads the model but never computes on it".
//!
//! **This module no longer shortens anyone's context, and must not start
//! again.** It once did: a load-time clamp sized the context to what the card
//! could hold, which shrank every user's conversation so that a single
//! full-length one would fit, and bounded concurrency not at all. That clamp
//! (`fit_context_to_budget`, `MIN_AUTO_CONTEXT`) was deleted on 2026-08-08 and
//! this doc described it for ten days afterwards.
//!
//! What replaced it is [`claim_exceeds_headroom`]: the loader records
//! [`kv_headroom_bytes`] on the model, and every forward checks the positions it
//! is about to claim against that budget before claiming them, refusing with a
//! `ServiceUnavailable` (503) that re-routes to a peer rather than growing into
//! an OOM. So the ceiling is enforced per REQUEST, at the moment memory is
//! actually taken, instead of being pre-paid by everyone at load.

/// Fraction of free VRAM the weights plus KV cache may claim, in percent.
///
/// The remainder absorbs activations, cuBLAS workspaces, allocator
/// fragmentation, and — the case that motivated the margin — a *second* model
/// loading alongside this one. The reported failure had a resident TinyLlama
/// turn a 3B load into a hard OOM, after which the model was pinned to CPU for
/// the rest of the daemon's run.
pub(crate) const VRAM_HEADROOM_PCT: u64 = 85;

/// KV-cache bytes one sequence position costs across a whole segment.
///
/// K and V are cached *before* the GQA repeat (see `LayerWeights::forward`,
/// which reshapes to `n_kv_head` and appends that), so the per-token cost is
/// driven by `n_kv_head`, not `n_head`.
///
/// `storage` is how the cache is kept (`layers::kv_storage`, the one answer):
/// f32 (4 bytes an element), f32 plus the f16 BSHD flash mirror (6 — **the
/// mirror has to be counted**: this figure is what the runtime head-room check
/// charges a request for, so omitting it let a GQA model on a card claim 50%
/// more memory than the budget believed and then run out for real, trading a
/// clean 503 that reroutes to a peer for an OOM), or the half cache a card
/// keeps (2; FUTURE_WORK #194).
pub(crate) fn kv_bytes_per_token(
    layers: usize,
    k_elems: usize,
    v_elems: usize,
    storage: crate::inference::split::kv_cache::KvStorage,
) -> u64 {
    (layers as u64)
        .saturating_mul((k_elems as u64).saturating_add(v_elems as u64))
        .saturating_mul(storage.bytes_per_element())
}

/// What a conversation's KV cache costs as it grows, over the layers one
/// memory holds (FUTURE_WORK #281): the bytes a position costs in the layers
/// that keep every position, and in the sliding-window layers whose caches
/// keep only their window (`kv_cache::window_kept`) — charged no further than
/// the positions such a cache is ever allocated
/// (`kv_cache::most_positions_a_window_holds`: its window, its rewind room and
/// the widest forward it is handed). vLLM prices a windowed layer the same way
/// (`SlidingWindowSpec::max_memory_usage_bytes`: `min(window - 1 + batch,
/// context)` positions); llama.cpp sizes its windowed cache `n_swa + n_ubatch`
/// (`llama_kv_cache_iswa`).
///
/// A model with no window that slides charges `every_position` alone: every
/// position costs [`Self::per_position`], as every estimate here charged
/// before #281.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct KvGrowth {
    /// Bytes a position costs in the layers that keep every position.
    every_position: u64,
    /// Bytes a position costs in the layers that keep only their window.
    windowed: u64,
    /// Their window (`split::SlidingWindow::size`).
    window: usize,
    /// Their caches' growth quantum (`layers::kv_growth_quantum`).
    quantum: usize,
    /// The widest forward their caches are handed.
    widest_forward: usize,
}

impl KvGrowth {
    /// `per_position` bytes for every position of the conversation.
    #[cfg(test)]
    pub(crate) fn flat(per_position: u64) -> Self {
        Self {
            every_position: per_position,
            ..Self::default()
        }
    }

    /// `every_position` bytes a position in the layers that keep each one, and
    /// `windowed` in those whose caches keep only `window` positions, grown by
    /// `quantum`.
    pub(crate) fn with_window(
        every_position: u64,
        windowed: u64,
        window: usize,
        quantum: usize,
    ) -> Self {
        Self {
            every_position,
            windowed,
            window,
            quantum,
            widest_forward: 0,
        }
    }

    /// Does any layer keep only its window?
    pub(crate) fn has_window(&self) -> bool {
        self.windowed > 0
    }

    /// The same caches, handed forwards up to `positions` wide: a windowed
    /// cache holds its window AND the forward.
    pub(crate) fn widened_to(self, positions: usize) -> Self {
        Self {
            widest_forward: self.widest_forward.max(positions),
            ..self
        }
    }

    /// Bytes a position costs while every cache still grows with the
    /// conversation — before any window is full.
    pub(crate) fn per_position(&self) -> u64 {
        self.every_position.saturating_add(self.windowed)
    }

    /// The most positions a windowed layer's cache is allocated.
    fn window_cap(&self) -> usize {
        crate::inference::split::kv_cache::most_positions_a_window_holds(
            self.window,
            self.quantum,
            self.widest_forward,
        )
    }

    /// Bytes the caches of a conversation `positions` long hold — at most.
    pub(crate) fn bytes_for(&self, positions: usize) -> u64 {
        let windowed_positions = if self.has_window() {
            positions.min(self.window_cap())
        } else {
            0
        };
        self.every_position
            .saturating_mul(positions as u64)
            .saturating_add(self.windowed.saturating_mul(windowed_positions as u64))
    }

    /// Bytes the caches take on growing from `from` positions to `to`.
    pub(crate) fn bytes_between(&self, from: usize, to: usize) -> u64 {
        self.bytes_for(to).saturating_sub(self.bytes_for(from))
    }

    /// The longest conversation `bytes` holds: the inverse of
    /// [`Self::bytes_for`]. `usize::MAX` where nothing grows past the windows.
    pub(crate) fn positions_within(&self, bytes: u64) -> usize {
        let per_position = self.per_position();
        if per_position == 0 {
            return usize::MAX;
        }
        let cap = if self.has_window() {
            self.window_cap() as u64
        } else {
            u64::MAX
        };
        let while_every_cache_grows = bytes / per_position;
        // Short of the cap every cache still grows; AT it the windows are
        // full, and what is left past them goes to the other layers alone.
        if while_every_cache_grows < cap {
            return usize::try_from(while_every_cache_grows).unwrap_or(usize::MAX);
        }
        // Past the windows only the layers that keep every position grow.
        if self.every_position == 0 {
            return usize::MAX;
        }
        let rest = bytes.saturating_sub(self.windowed.saturating_mul(cap));
        usize::try_from(rest / self.every_position).unwrap_or(usize::MAX)
    }

    /// Two sets of the same model's layers together — a split's card-held and
    /// processor-held caches.
    pub(crate) fn plus(self, other: Self) -> Self {
        Self {
            every_position: self.every_position.saturating_add(other.every_position),
            windowed: self.windowed.saturating_add(other.windowed),
            window: self.window.max(other.window),
            quantum: self.quantum.max(other.quantum),
            widest_forward: self.widest_forward.max(other.widest_forward),
        }
    }

    /// `part` of every `whole` of these layers — the share of a range's layers
    /// a prompt runs through.
    pub(crate) fn share(self, part: u64, whole: u64) -> Self {
        let whole = whole.max(1);
        Self {
            every_position: self.every_position.saturating_mul(part) / whole,
            windowed: self.windowed.saturating_mul(part) / whole,
            ..self
        }
    }
}

/// One layer's cache as [`KvLayout`] reads it.
#[derive(Debug, Clone, Copy)]
struct LayerKvShape {
    k_elems: usize,
    v_elems: usize,
    n_kv_head: usize,
    head_dim: usize,
    window: Option<usize>,
}

/// How each layer of a model keeps a conversation's cache, read once off its
/// header: which layers keep one (`split::kv_layers` — never a recurrent
/// layer's fixed state), each one's own shape (`attention_shapes`; MLA's
/// elements), how either device stores it (`layers::kv_storage`) and whether
/// that cache slides (`kv_cache::window_kept`). The loader's KV budget and
/// every estimate made before a load ([`KvPrice`]) build their [`KvGrowth`]
/// here, so the two cannot price one model differently (FUTURE_WORK #281).
#[derive(Debug, Clone)]
pub(crate) struct KvLayout {
    /// Per layer of the whole model; `None` where it keeps no cache.
    layers: Vec<Option<LayerKvShape>>,
    attention: crate::inference::layers::KvAttention,
    head_count: usize,
    /// The model's window (`SlidingWindow::size`), 0 for none.
    window: usize,
    /// The caches' growth quantum at this context (`layers::kv_growth_quantum`).
    quantum: usize,
}

impl KvLayout {
    /// The layout `ct` describes, for caches sized to `context_length`.
    pub(crate) fn read(
        ct: &candle_core::quantized::gguf_file::Content,
        meta: &super::GgufTensorMeta,
        context_length: usize,
    ) -> Result<Self, crate::error::SwarmError> {
        use crate::inference::model_arch::ModelArch;
        let arch = super::gguf_arch_str(ct);
        let model_arch = ModelArch::from_gguf_arch(&arch);
        let shapes = super::gguf_meta::attention_shapes(ct)?;
        // DeepSeek-2's MLA caches the decompressed K and V at full head count
        // (`mla_kv_elems`), whatever the layer's grouped shape says.
        let mla = matches!(model_arch, ModelArch::DeepSeek2).then(|| {
            let length = |suffix: &str| {
                ct.metadata
                    .get(&format!("{arch}.attention.{suffix}"))
                    .and_then(|v| v.to_u32().ok())
                    .map_or(meta.head_dim, |v| v as usize)
            };
            mla_kv_elems(
                meta.head_count,
                length("key_length"),
                length("value_length"),
            )
        });
        let window = super::SlidingWindow::read(ct);
        let mut layers = vec![None; shapes.len()];
        for layer in super::kv_layers(ct, 0, shapes.len()) {
            let shape = shapes[layer];
            let (k_elems, v_elems) =
                mla.unwrap_or_else(|| standard_kv_elems(shape.n_kv_head, shape.head_dim));
            layers[layer] = Some(LayerKvShape {
                k_elems,
                v_elems,
                n_kv_head: shape.n_kv_head,
                head_dim: shape.head_dim,
                window: window.as_ref().and_then(|w| w.for_layer(layer)),
            });
        }
        Ok(Self {
            layers,
            attention: crate::inference::layers::KvAttention::of(&model_arch),
            head_count: meta.head_count,
            window: window.map_or(0, |w| w.size),
            quantum: crate::inference::layers::kv_growth_quantum(context_length),
        })
    }

    /// What a conversation's cache costs as it grows over layers `from..to`
    /// kept on a card or on the processor: each layer priced by its own
    /// shape, stored as the ONE rule the cache itself is built by says (the
    /// half cache on a card, f32 plus the flash mirror for a grouped-query
    /// model where it does not apply, f32 for MLA), and a sliding-window
    /// layer whose cache keeps only its window charged as one.
    pub(crate) fn growth(&self, from: usize, to: usize, on_card: bool) -> KvGrowth {
        let (mut every_position, mut windowed) = (0u64, 0u64);
        let to = to.min(self.layers.len());
        for shape in self.layers[from.min(to)..to].iter().flatten() {
            let storage = crate::inference::layers::kv_storage(
                on_card,
                self.attention,
                self.head_count,
                shape.n_kv_head,
                shape.head_dim,
            );
            let bytes = kv_bytes_per_token(1, shape.k_elems, shape.v_elems, storage);
            if super::kv_cache::window_kept(shape.window, storage).is_some() {
                windowed = windowed.saturating_add(bytes);
            } else {
                every_position = every_position.saturating_add(bytes);
            }
        }
        KvGrowth::with_window(every_position, windowed, self.window, self.quantum)
    }

    /// [`Self::growth`] on both devices, for a pool reading prompts in
    /// `prompt_chunk`-position forwards.
    pub(crate) fn price(&self, from: usize, to: usize, prompt_chunk: usize) -> KvPrice {
        KvPrice {
            card: self.growth(from, to, true),
            processor: self.growth(from, to, false),
            prompt_chunk,
        }
    }
}

/// What a conversation's caches cost a segment on either device its layers
/// may be kept on, and the prompt chunk its worker reads with — the ONE
/// figure every estimate made before a load prices KV by: admission's
/// footprint (`vram::VramFootprintInputs`), the cost curve, the card/processor
/// split. Built from the header as the loader builds its own budget
/// ([`KvLayout`]), so a placement is weighed by what the worker will hold: a
/// sliding-window layer charged its window, each layer its own width
/// (FUTURE_WORK #281).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct KvPrice {
    pub(crate) card: KvGrowth,
    pub(crate) processor: KvGrowth,
    /// `inference.prefill_chunk_tokens`; 0 reads a prompt in one forward.
    pub(crate) prompt_chunk: usize,
}

impl KvPrice {
    /// `kv_layers` layers keeping caches of one shape and no window — what
    /// every estimate charged each layer before #281. In the order
    /// `VramFootprintInputs` carried these fields.
    #[cfg(test)]
    pub(crate) fn uniform(
        kv_layers: u64,
        head_count_kv: u64,
        head_count: u64,
        head_dim: u64,
        attention: crate::inference::layers::KvAttention,
    ) -> Self {
        let elems = head_count_kv.saturating_mul(head_dim) as usize;
        let on = |card: bool| {
            let storage = crate::inference::layers::kv_storage(
                card,
                attention,
                head_count as usize,
                head_count_kv as usize,
                head_dim as usize,
            );
            KvGrowth::flat(kv_bytes_per_token(1, elems, elems, storage).saturating_mul(kv_layers))
        };
        Self {
            card: on(true),
            processor: on(false),
            prompt_chunk: 128,
        }
    }

    /// The cache a worker reading prompts as `reader` keeps, handed forwards
    /// as wide as it reads them (`prefill_pacer::widest_prompt_forward`, the
    /// worker's own rule): a windowed cache holds its window AND that forward.
    pub(crate) fn reading(
        self,
        reader: crate::inference::prefill_pacer::PromptReader,
        context: usize,
    ) -> KvGrowth {
        use crate::inference::prefill_pacer::{widest_prompt_forward, PromptReader};
        let growth = match reader {
            PromptReader::Processor => self.processor,
            PromptReader::Card | PromptReader::CardCopyingWeights { .. } => self.card,
        };
        growth.widened_to(widest_prompt_forward(self.prompt_chunk, reader, context))
    }

    /// `part` of every `whole` of the segment's layers.
    pub(crate) fn share(self, part: u64, whole: u64) -> Self {
        Self {
            card: self.card.share(part, whole),
            processor: self.processor.share(part, whole),
            ..self
        }
    }
}

/// Per-token K and V element counts for the standard (MHA/GQA) attention path.
pub(crate) fn standard_kv_elems(head_count_kv: usize, head_dim: usize) -> (usize, usize) {
    let per = head_count_kv.saturating_mul(head_dim);
    (per, per)
}

/// Per-token K and V element counts for DeepSeek-style MLA, which caches the
/// *decompressed* K and V at full head count with asymmetric widths
/// (`key_length` != `value_length`) — see `MlaWeights::forward_mla`.
pub(crate) fn mla_kv_elems(
    head_count: usize,
    key_length: usize,
    value_length: usize,
) -> (usize, usize) {
    (
        head_count.saturating_mul(key_length),
        head_count.saturating_mul(value_length),
    )
}

/// Bytes available for KV cache once the weights are resident.
///
/// The headroom arithmetic that once sized a context up front at load time
/// (`fit_context_to_budget`, removed in f7c3010b, 2026-08-08). It is now the
/// RUNTIME budget: the loader records it on the model and every forward checks
/// against it before claiming another growth quantum.
pub(crate) fn kv_headroom_bytes(weight_bytes: u64, free_vram_bytes: u64) -> u64 {
    (free_vram_bytes / 100 * VRAM_HEADROOM_PCT).saturating_sub(weight_bytes)
}

/// What the card keeps free beyond the KV cache when the budget is reconciled
/// with it: the forward in flight's activations and library workspaces, and
/// allocator fragmentation. A fraction of the CARD with a floor, because those
/// costs do not shrink as the card fills — the same reason [`VRAM_HEADROOM_PCT`]
/// is a share of free memory at load rather than of whatever the model left.
pub(crate) const DEVICE_FREE_MARGIN_PCT: u64 = 5;
/// See [`DEVICE_FREE_MARGIN_PCT`]; the floor on small cards.
pub(crate) const DEVICE_FREE_MARGIN_MIN_BYTES: u64 = 256 << 20;

/// Bytes the card must keep free beside the KV cache, for a card of `total_bytes`.
pub(crate) fn device_free_margin_bytes(total_bytes: u64) -> u64 {
    (total_bytes / 100 * DEVICE_FREE_MARGIN_PCT).max(DEVICE_FREE_MARGIN_MIN_BYTES)
}

/// The KV budget as the card can honour it NOW.
///
/// **The load-time budget is a prediction; the card is the fact.**
/// [`kv_headroom_bytes`] is taken once, from free memory at load, and it
/// cannot see a tenant that arrives afterwards — a second worker loading
/// another model on the same card, the llama.cpp context of the full CUDA
/// build, a prefix-cache snapshot the size of a prompt. Measured on the
/// released v0.3.149 (gotcha #440, third half): a budget of 4491 MB taken
/// when 7541 MB were free, on a card idling at 7827 of 8192 MB once the model,
/// two CUDA contexts and one cached prompt were resident. The real room for
/// live KV was ~2 GB, the budget said 4.5, and the admitted prompt's cache
/// landed in host-backed memory at 1.95 tok/s — WSL2 hands out shared memory
/// rather than failing, so the accounting is the only guard there is.
///
/// The reconciled figure is the smaller of the budget and `live + cached +
/// free_now − margin`: everything the KV cache already holds on the device,
/// plus what the device has left, less the margin the forward needs. Stated
/// that way it is **invariant under evicting a cached prompt** — releasing
/// `x` bytes moves `x` from `cached` to `free_now` and the sum is unchanged —
/// which is what lets [`admit_prompt`]'s evict-then-fit arithmetic hold
/// against it exactly as it does against the load-time budget.
///
/// ⚠ **On a card that holds only once the release has LANDED**, which is why
/// [`device_free_and_total_bytes`] synchronizes before it reads. cudarc frees
/// through the device's memory pool, and the pool hands memory back to the
/// device only at a synchronize (`docs/FUTURE_WORK.md` #121): without one, the
/// bytes left `cached` (or a finished request's live cache) and never reached
/// `free_now`, and the next prompt was judged against a budget short by exactly
/// what the last one released.
///
/// Never larger than the budget: the device having room does not license the
/// cache to take more than the loader set aside for it.
///
/// **The processor needs this as much as the card does, and did not have it**
/// (gotcha #462). A CPU worker's budget comes from the daemon at spawn —
/// "whatever of this node's RAM budget nothing else has claimed" — and nothing
/// revisited it afterwards, so it could not see a second worker starting, the
/// rest of the machine growing, or the worker's OWN weights growing when a
/// failover handed it more layers. Reported from a 16 GB processor-only Mac
/// mini: one request's repeated local-standby failovers took a worker from 12
/// to 29 of a 48-layer model while its cache went on filling against the
/// ceiling it had been given for 12, and the process was killed — losing a
/// generation that had already streamed 238 tokens over ~10 minutes. With the
/// reconciliation the same worker refuses (503, re-routed) as the machine
/// fills, which is the outcome the budget was written to produce.
pub(crate) fn budget_reconciled_with_device(
    budget_bytes: u64,
    live_bytes: u64,
    cached_bytes: u64,
    free_now_bytes: u64,
    total_bytes: u64,
) -> u64 {
    let ours_plus_free = live_bytes
        .saturating_add(cached_bytes)
        .saturating_add(free_now_bytes);
    budget_bytes.min(ours_plus_free.saturating_sub(device_free_margin_bytes(total_bytes)))
}

/// `(free, total)` bytes of the memory `device` allocates out of, or `None`
/// when it cannot be read.
///
/// CUDA: `cudaMemGetInfo` under cudarc's name, a driver call taking
/// microseconds. Processor: the machine's own memory, cached briefly (see
/// [`system_free_and_total_bytes`]).
///
/// **The processor arm is not an afterthought.** Without it a CPU worker's
/// budget was the load-time prediction for ever, which is precisely the
/// failure `budget_reconciled_with_device` exists to prevent — see its own
/// doc, and gotcha #462.
///
/// **The CUDA arm synchronizes the device's stream before it reads**
/// (`docs/FUTURE_WORK.md` #121). cudarc frees every buffer with
/// `cuMemFreeAsync` when the card supports memory pools, and a pool returns
/// what was freed to the device — the only memory `cuMemGetInfo` counts as
/// free — at the next stream, event or context synchronize: the pool's release
/// threshold defaults to zero and nothing here raises it (CUDA Runtime API,
/// "Stream Ordered Memory Allocator", `cudaMemPoolAttrReleaseThreshold`).
/// Unsynchronized, the reading still counted as used the cache the previous
/// request had just dropped and the prefix snapshots admission had just
/// evicted. Measured on the released v0.3.207, llama-3.1-8b on an 8 GB card
/// with nothing else in flight: a 1,835-token prompt served, and the next of
/// the same size refused against 683 of the real 1,336 MB, four runs of four,
/// the prefix cache on or off. Every caller is already off the per-token path
/// (`ensure_room_for_prompt`, `positions_beside_live`, the executor's
/// guard on a growth boundary), so the wait is for work the next step would
/// queue behind anyway. `SWARMLLM_KV_DEVICE_SYNC=0` reads without it, for the
/// A/B inside one binary.
///
/// **Free includes what this process's pool keeps** (`inference::cuda_pool`,
/// #146). The pool no longer hands freed memory back at the synchronize — that
/// hand-back, and fetching the memory fresh at the next admission, was the
/// step that slowed ~1000× with host uptime — so `cuMemGetInfo` alone would
/// read a finished request's cache as used, which is #121 again by another
/// road. Reserved-but-unused pool memory is exactly what the next allocation
/// here draws from first, and the synchronize is still what makes a buffer
/// freed on the stream count as unused.
pub(crate) fn device_free_and_total_bytes(device: &candle_core::Device) -> Option<(u64, u64)> {
    #[cfg(feature = "candle-cuda")]
    if let candle_core::Device::Cuda(dev) = device {
        let stream = dev.cuda_stream();
        if device_read_synchronizes() {
            // A failed synchronize leaves the reading as it was before this
            // existed — conservative, never larger than the truth.
            let _ = stream.synchronize();
        }
        let pool = crate::inference::cuda_pool::usage(device);
        let reading = stream
            .context()
            .mem_get_info()
            .ok()
            .map(|(free, total)| (free as u64, total as u64));
        if let (Some(p), Some((free, _))) = (pool, reading) {
            tracing::debug!(
                device_free_mb = free >> 20,
                pool_reserved_mb = p.reserved >> 20,
                pool_used_mb = p.used >> 20,
                pool_reusable_mb = p.reusable() >> 20,
                "DIAG: card memory reading"
            );
        }
        let kept = pool.map_or(0, crate::inference::cuda_pool::PoolUsage::reusable);
        return reading.map(|(free, total)| (free.saturating_add(kept).min(total), total));
    }
    if matches!(device, candle_core::Device::Cpu) {
        if !processor_reconciliation_enabled() {
            return None;
        }
        return system_free_and_total_bytes();
    }
    let _ = device;
    None
}

/// `SWARMLLM_KV_RECONCILE=0` → the processor keeps its load-time budget, the
/// behaviour before gotcha #462. An A/B switch inside ONE binary, the same
/// discipline as `SWARMLLM_DECODE_ATTN` and `SWARMLLM_KV_TRUNCATE`.
///
/// It exists because this is the one change in its release that can refuse work
/// the node previously completed: a processor node under real memory pressure
/// now returns a 503 that re-routes, and where no peer holds the model there is
/// nowhere for it to go. That is the intended trade — the alternative is
/// swapping, or the OS killing the worker, which is what #462 was reported for
/// — but it is a trade, and a node that turns out to be on the wrong side of it
/// should not need a rollback to say so.
///
/// The CUDA arm is deliberately NOT switchable: it has been in the field since
/// v0.3.152 and nothing about it is in question.
fn processor_reconciliation_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        reconciliation_enabled_for(std::env::var("SWARMLLM_KV_RECONCILE").ok().as_deref())
    })
}

/// The switch's reading of the variable, split out because the wrapper above
/// caches in a `OnceLock` and so cannot be exercised twice in one process.
fn reconciliation_enabled_for(v: Option<&str>) -> bool {
    !matches!(v, Some("0") | Some("off"))
}

/// `SWARMLLM_KV_DEVICE_SYNC=0` → the card is read without synchronizing
/// first, the behaviour before `docs/FUTURE_WORK.md` #121. The A/B switch for
/// that fix, read once; on by default.
#[cfg(feature = "candle-cuda")]
fn device_read_synchronizes() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        device_sync_enabled_for(std::env::var("SWARMLLM_KV_DEVICE_SYNC").ok().as_deref())
    })
}

/// The switch's reading of the variable, apart from its `OnceLock` so a test
/// can ask it twice. Only the CUDA arm consults it.
#[cfg(any(test, feature = "candle-cuda"))]
fn device_sync_enabled_for(v: Option<&str>) -> bool {
    !matches!(v, Some("0") | Some("off") | Some("false"))
}

/// How long a system-memory reading is reused before it is taken again.
///
/// `mem_get_info` is a driver call costing microseconds; a system-memory
/// refresh reads `/proc/meminfo` (or the mach equivalent), which is cheap but
/// not free. Every caller of the reconciliation is already off the per-token
/// path — `ensure_room_for_prompt`, `positions_beside_live`, and the
/// per-chunk guard, which itself only runs on a growth-quantum boundary — so
/// a short window costs nothing and bounds the syscall rate under a prefill
/// that crosses many quanta.
const SYSTEM_MEMORY_TTL: std::time::Duration = std::time::Duration::from_millis(250);

/// `(available, total)` bytes of system memory, or `None` if unreadable.
///
/// `available`, not `free`: reclaimable page cache is memory this process can
/// have, and reading `free` would make a healthy machine look exhausted. Read
/// through `daemon::machine_memory`, so a worker in a container admits against
/// the container's limit — the one figure the planner and the advertisement
/// read too.
fn system_free_and_total_bytes() -> Option<(u64, u64)> {
    use std::sync::Mutex;
    static CACHE: Mutex<Option<(std::time::Instant, u64, u64)>> = Mutex::new(None);
    // A poisoned lock only means some other reader panicked mid-refresh; the
    // reading is a cache, so recovering it is strictly better than propagating.
    let mut guard = CACHE.lock().unwrap_or_else(|e| e.into_inner());
    if let Some((at, free, total)) = *guard {
        if at.elapsed() < SYSTEM_MEMORY_TTL {
            return Some((free, total));
        }
    }
    let m = crate::daemon::machine_memory::machine_memory()?;
    let (free, total) = (m.available_bytes, m.total_bytes);
    *guard = Some((std::time::Instant::now(), free, total));
    Some((free, total))
}

/// Sequence positions this forward will NEWLY allocate, or 0 if it fits
/// inside what the cache already holds.
///
/// `allocated` is the buffer the request's caches already have (0 before the
/// first append), `reserved` what the worker asked the first allocation to
/// cover, `total_seq` the positions the cache must hold after this forward.
/// A forward that fits inside the buffer claims nothing — which is almost
/// every decode step, and what keeps the headroom check off the per-token
/// path. The first allocation is the larger of the reservation and the
/// forward's own rounded need; growth past a buffer is charged in quanta.
///
/// **It returns positions, not "one quantum".** A prefill jumps many quanta at
/// once: 0 -> 5000 tokens at a 512 quantum reserves ten of them in a single
/// forward. Charging one would under-count the largest claim any request
/// makes by an order of magnitude, which is precisely the case the budget
/// exists to catch.
///
/// **It reads the buffer, not the quantum arithmetic.** The previous form
/// derived the buffer from `index_pos` rounded up — true only while caches
/// grew from one quantum. A reserved cache is larger than that from its first
/// append, so the old arithmetic charged a reply for every quantum boundary
/// it crossed inside memory the request already held, and could refuse it
/// for room it did not need.
pub(crate) fn positions_to_allocate(
    allocated: usize,
    reserved: usize,
    total_seq: usize,
    quantum: usize,
) -> usize {
    let q = quantum.max(1);
    let round_up = |n: usize| n.div_ceil(q).saturating_mul(q);
    if allocated == 0 {
        round_up(total_seq).max(reserved)
    } else if total_seq > allocated {
        round_up(total_seq - allocated)
    } else {
        0
    }
}

/// Would allocating `claim` more bytes push total KV occupancy past the
/// budget? The claim is what the forward's caches newly allocate
/// (`KvCacheStore::bytes_a_forward_adds`), not a per-position rate times the
/// positions: a sliding-window layer's cache slides instead of growing, which
/// no rate can see (FUTURE_WORK #281).
///
/// `in_use_bytes` is what the whole store already holds — every request on
/// this worker, not just the one asking. That is the point: the load-time
/// clamp this replaces sized for ONE conversation at full length and so did
/// not bound concurrency at all.
///
/// This is Head-Room Admission (vLLM). The difference here is what happens on
/// refusal: vLLM must preempt and recompute because it has nowhere else to
/// send the work, whereas a refused request in a swarm can be served by a
/// peer — so refusing early is cheap and correct.
pub(crate) fn claim_exceeds_headroom(budget_bytes: u64, in_use_bytes: u64, claim: u64) -> bool {
    in_use_bytes.saturating_add(claim) > budget_bytes
}

/// What to do about a prompt BEFORE its prefill starts (gotcha #440).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PromptAdmission {
    /// Live cache plus cached prompts plus this prompt fit the budget.
    Fits,
    /// It fits once this many bytes of cached prompts are evicted.
    EvictBytes(u64),
    /// It does not fit even with every cached prompt gone; refuse now,
    /// short by this many bytes.
    Refuse { short_by: u64 },
}

/// Decide once, for the whole prompt, whether its KV can live on the device.
///
/// **Why a whole-prompt decision exists beside the per-chunk guard.** The
/// per-chunk guard (`claim_exceeds_headroom`) sees only what the KV store
/// holds. The prefix cache's snapshots are the same device memory — up to
/// the cache's byte cap, and one entry may exceed it by design — and were
/// charged nowhere, while the budget was sized from free memory at LOAD,
/// before any snapshot existed. Measured on an 8 GB card (gotcha #439/#440):
/// the second 6.4k-token request found ~300 MB free, its 2.6 GB cache was
/// allocated anyway — WSL2 hands out host-backed memory rather than failing —
/// and every decode step then read the cache over PCIe: 3-5 tok/s where an
/// empty card did 19-33. Nothing refused, nothing logged.
///
/// The cached prompts are a cache: reconstructible, and worth less than the
/// request in hand. So the order is fit → evict cached prompts → refuse,
/// and the refusal comes at token 0 rather than at chunk N with a partial
/// cache already built (the ratchet of gotcha #387, one level up).
///
/// `claim` is the bytes the prompt's caches will hold — [`KvGrowth::bytes_for`]
/// its positions.
pub(crate) fn admit_prompt(
    budget_bytes: u64,
    live_bytes: u64,
    cached_prompt_bytes: u64,
    claim: u64,
) -> PromptAdmission {
    let total = live_bytes
        .saturating_add(cached_prompt_bytes)
        .saturating_add(claim);
    if total <= budget_bytes {
        return PromptAdmission::Fits;
    }
    let excess = total - budget_bytes;
    if excess <= cached_prompt_bytes {
        PromptAdmission::EvictBytes(excess)
    } else {
        PromptAdmission::Refuse {
            short_by: excess - cached_prompt_bytes,
        }
    }
}

/// How much of a finished prompt to keep as a prefix-cache snapshot, so that
/// the snapshot fits BESIDE the request's own live cache (gotcha #440).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SnapshotPlan {
    /// Bytes of older cached prompts to evict first.
    pub(crate) evict_bytes: u64,
    /// Positions to snapshot after that; 0 means do not snapshot.
    pub(crate) positions: usize,
}

/// A snapshot is a COPY of the request's cache, so on a device where the live
/// cache alone takes a third of the budget the whole-prompt snapshot cannot
/// fit beside it — measured: admission evicted the previous snapshot, the
/// request started warm, and the snapshot taken after its prefill put the
/// card over the top again, so the next growth quantum was refused
/// mid-reply and the tokens before it decoded at 3-6 tok/s. Older cached
/// prompts go first; then the snapshot is cut to what fits, because a
/// partial prefix still saves its length of prefill on the next turn
/// (`lookup` narrows a longer entry to the shared length). A snapshot copies
/// what each layer's cache holds, so a sliding-window layer's costs no more
/// than its window ([`KvGrowth::bytes_for`]).
pub(crate) fn plan_snapshot(
    budget_bytes: u64,
    live_bytes: u64,
    cached_prompt_bytes: u64,
    growth: KvGrowth,
    prompt_positions: usize,
) -> SnapshotPlan {
    if growth.per_position() == 0 {
        return SnapshotPlan {
            evict_bytes: 0,
            positions: prompt_positions,
        };
    }
    let needed = growth.bytes_for(prompt_positions);
    let room = budget_bytes.saturating_sub(live_bytes.saturating_add(cached_prompt_bytes));
    if room >= needed {
        return SnapshotPlan {
            evict_bytes: 0,
            positions: prompt_positions,
        };
    }
    let evict_bytes = (needed - room).min(cached_prompt_bytes);
    let room_after = room.saturating_add(evict_bytes);
    SnapshotPlan {
        evict_bytes,
        positions: growth.positions_within(room_after).min(prompt_positions),
    }
}

/// The share of the machine a card's prefix-cache snapshots never take: one
/// part in this many. See [`host_snapshot_budget`].
pub(crate) const HOST_SNAPSHOT_RESERVE_DIVISOR: u64 = 4;

/// What a card's prefix-cache snapshots may hold in host RAM (#257): the
/// machine's available memory plus what they already hold, less a quarter of
/// the machine. The [`plan_snapshot`] budget for a snapshot whose home is
/// `prefix_cache::SnapshotHome::HostRam`; invariant under evicting one, as
/// [`budget_reconciled_with_device`] is.
///
/// **A quarter, not the card's 5% margin**, because these bytes are a tenant
/// no other budget can reclaim: a processor worker reconciles its live cache
/// against the machine's available memory (gotcha #462) and sees another
/// worker's snapshots only as memory gone. A cache of reconstructible prompts
/// ranks below any request in hand (gotcha #440), so it takes only memory the
/// machine is far from needing. The per-model byte cap
/// (`inference.prefix_cache_max_mb`) still bounds it first; llama.cpp's
/// server bounds the same cache by a fixed 8 GiB (`--cache-ram`).
pub(crate) fn host_snapshot_budget(available_bytes: u64, total_bytes: u64, held_bytes: u64) -> u64 {
    available_bytes
        .saturating_add(held_bytes)
        .saturating_sub(total_bytes / HOST_SNAPSHOT_RESERVE_DIVISOR)
}

/// `(available, total)` bytes of the machine's memory, as every processor
/// budget here reads it, or `None` when unreadable.
pub(crate) fn host_free_and_total_bytes() -> Option<(u64, u64)> {
    system_free_and_total_bytes()
}

#[cfg(test)]
mod promised_tests {
    use super::*;

    /// Two prompts of the same size against a budget with room for one.
    ///
    /// The second is only refused if the first's admission is counted, which
    /// is the whole of the fix: on the batched path the first prompt's cache
    /// has not been written when the second is weighed, so an occupancy-only
    /// read passes both.
    #[test]
    fn counting_an_admitted_prompt_is_what_refuses_the_second_one() {
        let budget = 10_000_000;
        let per_token = 1_000;
        let positions = 6_000; // 6 MB each — two do not fit in 10 MB.

        // First prompt: nothing live, nothing promised.
        assert!(matches!(
            admit_prompt(budget, 0, 0, per_token * positions as u64),
            PromptAdmission::Fits
        ));

        // Second, weighed the OLD way — the first has not allocated yet, so
        // live is still zero and it is admitted too. This is the defect.
        assert!(
            matches!(
                admit_prompt(budget, 0, 0, per_token * positions as u64),
                PromptAdmission::Fits
            ),
            "control: without counting the promise, both prompts pass"
        );

        // Weighed with the first's promise included, it is correctly refused.
        let promised = per_token * positions as u64;
        assert!(
            matches!(
                admit_prompt(budget, promised, 0, per_token * positions as u64),
                PromptAdmission::Refuse { .. }
            ),
            "with the promise counted the second prompt is refused at token 0"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::split::kv_cache::KvStorage;

    /// A card's snapshots in host RAM never take the machine's last quarter,
    /// and the budget does not move when one of them is evicted — the same
    /// property that lets `plan_snapshot`'s evict-then-fit hold (#257).
    #[test]
    fn host_snapshots_leave_the_machine_a_quarter() {
        const GB: u64 = 1 << 30;
        // 16 GB machine, 10 GB available, nothing held: 10 − 4 = 6 GB.
        assert_eq!(host_snapshot_budget(10 * GB, 16 * GB, 0), 6 * GB);
        // Evicting 1 GB of held snapshots moves it to available: unchanged.
        assert_eq!(
            host_snapshot_budget(9 * GB, 16 * GB, GB),
            host_snapshot_budget(10 * GB, 16 * GB, 0)
        );
        // A machine already inside its last quarter offers nothing.
        assert_eq!(host_snapshot_budget(3 * GB, 16 * GB, 0), 0);
        // 216 MB a snapshot (3,774 tokens of a 7B at 57,344 B each): an
        // 8 GB machine with 2.5 GB available keeps it whole.
        let per_token = 57_344;
        let plan = plan_snapshot(
            host_snapshot_budget(2 * GB + GB / 2, 8 * GB, 0),
            0,
            0,
            KvGrowth::flat(per_token),
            3_774,
        );
        assert_eq!(plan.positions, 3_774);
        // ... and at 2.1 GB available keeps the opening that fits in 0.1 GB.
        let plan = plan_snapshot(
            host_snapshot_budget(2 * GB + GB / 10, 8 * GB, 0),
            0,
            0,
            KvGrowth::flat(per_token),
            3_774,
        );
        assert_eq!(plan.positions as u64, (GB / 10) / per_token);
    }

    /// The snapshot is cut to the room beside the live cache, after older
    /// cached prompts have been offered up — and is skipped outright when the
    /// live cache leaves nothing.
    #[test]
    fn a_snapshot_is_cut_to_what_fits_beside_the_live_cache() {
        // 1000-byte budget, 10 bytes a token, a 40-token prompt (400 bytes).
        assert_eq!(
            plan_snapshot(1000, 400, 0, KvGrowth::flat(10), 40),
            SnapshotPlan {
                evict_bytes: 0,
                positions: 40
            }
        );
        // Live 400 + cached 500: room 100 → evict all 500 cached, room 600 → whole prompt.
        assert_eq!(
            plan_snapshot(1000, 400, 500, KvGrowth::flat(10), 40),
            SnapshotPlan {
                evict_bytes: 300,
                positions: 40
            }
        );
        // Live 800, cached 100: room 100, evict 100 → room 200 → 20 positions.
        assert_eq!(
            plan_snapshot(1000, 800, 100, KvGrowth::flat(10), 40),
            SnapshotPlan {
                evict_bytes: 100,
                positions: 20
            }
        );
        // Live cache already at the budget: nothing to snapshot.
        assert_eq!(
            plan_snapshot(1000, 1000, 0, KvGrowth::flat(10), 40),
            SnapshotPlan {
                evict_bytes: 0,
                positions: 0
            }
        );
        // Unknown per-token cost: no judgement, snapshot everything.
        assert_eq!(
            plan_snapshot(1000, 1000, 0, KvGrowth::flat(0), 40),
            SnapshotPlan {
                evict_bytes: 0,
                positions: 40
            }
        );
    }

    /// The measured case: a 6.4k-token prompt whose live cache is 2.6 GB of a
    /// 4.5 GB budget cannot keep a full copy of itself; it keeps what fits.
    #[test]
    fn the_long_prompts_own_snapshot_is_cut_rather_than_overfilling_the_card() {
        let mb = 1024 * 1024u64;
        let per_token = 344 * 1024u64;
        let live = per_token * 7680;
        let plan = plan_snapshot(4523 * mb, live, 0, KvGrowth::flat(per_token), 7649);
        assert_eq!(plan.evict_bytes, 0);
        assert!(plan.positions > 0 && plan.positions < 7649, "{plan:?}");
        assert!(live + per_token * plan.positions as u64 <= 4523 * mb);
    }

    /// The three outcomes, and the boundary between them: cached prompts are
    /// the ONLY thing eviction can recover, so a shortfall past them is a
    /// refusal however large the cache is.
    #[test]
    fn a_prompt_is_admitted_evicts_cached_prompts_or_is_refused_at_token_zero() {
        // 1000-byte budget, 10 bytes a token.
        assert_eq!(admit_prompt(1000, 0, 0, 1000), PromptAdmission::Fits);
        assert_eq!(admit_prompt(1000, 500, 0, 500), PromptAdmission::Fits);
        // Live 500 + cached 300 + claim 300 = 1100: 100 over, cache can cover it.
        assert_eq!(
            admit_prompt(1000, 500, 300, 300),
            PromptAdmission::EvictBytes(100)
        );
        // Live 500 + cached 300 + claim 600 = 1400: 400 over, cache covers 300.
        assert_eq!(
            admit_prompt(1000, 500, 300, 600),
            PromptAdmission::Refuse { short_by: 100 }
        );
        // Exactly at the budget fits; one byte past it evicts one byte.
        assert_eq!(admit_prompt(1000, 700, 100, 200), PromptAdmission::Fits);
        assert_eq!(
            admit_prompt(999, 700, 100, 200),
            PromptAdmission::EvictBytes(1)
        );
    }

    /// The measured case: a 3.1 GB model on an 8 GB card, one 6.4k-token
    /// prompt's snapshot already cached, the same prompt again. Before this
    /// existed the answer was "fits" and the cache spilled to host memory.
    #[test]
    fn the_second_long_prompt_evicts_the_first_ones_snapshot_rather_than_spilling() {
        let mb = 1024 * 1024u64;
        let budget = 4523 * mb;
        let per_token = 344 * 1024u64;
        // The first prompt's snapshot: 7680 positions (6.4k tokens rounded
        // up to the growth quantum) at 344 KB each, ~2.6 GB.
        let cached = per_token * 7680;
        assert!(cached < budget && 2 * cached > budget);
        assert_eq!(
            admit_prompt(budget, 0, cached, per_token * 7680),
            PromptAdmission::EvictBytes(2 * cached - budget)
        );
    }

    /// MLA caches decompressed K/V at full head count with asymmetric widths,
    /// so it is far more expensive per token than GQA at the same head_dim.
    /// Getting this wrong would under-estimate DeepSeek's cache.
    #[test]
    fn mla_costs_more_per_token_than_gqa() {
        let (gk, gv) = standard_kv_elems(8, 128);
        let (mk, mv) = mla_kv_elems(128, 192, 128);
        assert!(
            kv_bytes_per_token(1, mk, mv, KvStorage::F32)
                > kv_bytes_per_token(1, gk, gv, KvStorage::F32),
            "MLA per-token cost must exceed GQA's"
        );
    }

    /// The real geometry behind the 2026-07-29 report: an 8 GB card and a
    /// 1.3 GB model declaring a 128K context. The budget must come out well
    /// short of that context — which is now a warning at load and a refusal
    /// only if a conversation actually gets that long, where it used to be a
    /// permanent cut to everyone's context.
    #[test]
    fn a_long_context_model_on_a_small_card_gets_a_short_budget() {
        let (k, v) = standard_kv_elems(8, 64);
        let per_token = kv_bytes_per_token(16, k, v, KvStorage::F32);
        let free = 7_000u64 * 1024 * 1024;
        let weights = 1_300u64 * 1024 * 1024;
        let budget = kv_headroom_bytes(weights, free);
        let affordable = budget / per_token;
        assert!(affordable > 0, "the card must afford SOME conversation");
        assert!(
            affordable < 131_072,
            "a 128K conversation must not fit: affordable {affordable}"
        );
        // And the budget must respect the headroom margin.
        assert!(budget + weights <= free / 100 * VRAM_HEADROOM_PCT + weights);
    }

    /// Weights that already exceed the headroom leave a zero budget rather
    /// than wrapping around to an enormous one.
    #[test]
    fn weights_larger_than_the_card_give_a_zero_budget() {
        assert_eq!(kv_headroom_bytes(8 << 30, 4 << 30), 0);
    }

    /// Growth is claimed exactly when the buffer is outgrown and nowhere else —
    /// that is what keeps the check off the per-token path.
    #[test]
    fn growth_is_claimed_only_when_the_buffer_is_outgrown() {
        let q = 512;
        assert_eq!(
            positions_to_allocate(0, 0, 1, q),
            512,
            "a fresh request claims one"
        );
        assert_eq!(positions_to_allocate(0, 0, 512, q), 512);
        // Decoding inside the current buffer claims nothing.
        assert_eq!(positions_to_allocate(512, 0, 2, q), 0);
        assert_eq!(positions_to_allocate(512, 0, 511, q), 0);
        assert_eq!(positions_to_allocate(512, 0, 512, q), 0);
        // Outgrowing it claims one.
        assert_eq!(positions_to_allocate(512, 0, 513, q), 512);
        assert_eq!(positions_to_allocate(1024, 0, 514, q), 0);
    }

    /// **A prefill claims MANY quanta in one forward.** Charging it one — which
    /// the first version of this did — under-counts the single largest claim a
    /// request ever makes, letting exactly the allocation the budget exists to
    /// refuse go straight through.
    #[test]
    fn a_large_prefill_claims_every_quantum_it_spans() {
        let q = 512;
        assert_eq!(
            positions_to_allocate(0, 0, 5000, q),
            5120,
            "10 quanta, not 1"
        );
        assert_eq!(positions_to_allocate(0, 0, 2000, q), 2048);
        // Continuing a conversation charges only the new span.
        assert_eq!(positions_to_allocate(1024, 0, 5000, q), 5120 - 1024);
    }

    /// A reserved prompt is charged ONCE, for the reservation, and its reply
    /// then runs inside that memory without being charged again at every
    /// quantum boundary it crosses — the arithmetic the old form got wrong.
    #[test]
    fn a_reservation_is_charged_once_and_not_again_inside_it() {
        let q = 512;
        // First forward: nothing allocated, the worker reserved prompt + reply.
        assert_eq!(
            positions_to_allocate(0, 2560, 2000, q),
            2560,
            "the reservation is the claim"
        );
        // The reservation never undercuts what the forward itself needs.
        assert_eq!(positions_to_allocate(0, 512, 2000, q), 2048);
        // Decode steps inside the reserved buffer cross 2048 and claim nothing.
        assert_eq!(positions_to_allocate(2560, 2560, 2049, q), 0);
        assert_eq!(positions_to_allocate(2560, 2560, 2560, q), 0);
        // Outgrowing the reservation is charged like any other growth.
        assert_eq!(positions_to_allocate(2560, 2560, 2561, q), 512);
    }

    /// And the budget must see that full claim, not a single quantum's worth.
    #[test]
    fn a_large_prefill_is_refused_when_it_does_not_fit() {
        let per_token = 1_000u64;
        let q = 512usize;
        // Room for two quanta only.
        let budget = per_token * (2 * q) as u64;
        // One quantum fits...
        assert!(!claim_exceeds_headroom(
            budget,
            0,
            per_token * positions_to_allocate(0, 0, 1, q) as u64
        ));
        // ...but a ten-quantum prefill must not.
        assert!(
            claim_exceeds_headroom(
                budget,
                0,
                per_token * positions_to_allocate(0, 0, 5000, q) as u64
            ),
            "a prefill spanning 10 quanta must be refused against a 2-quantum budget"
        );
    }

    /// A zero quantum must not divide by zero.
    #[test]
    fn a_zero_quantum_is_treated_as_one() {
        assert!(positions_to_allocate(0, 0, 1, 0) > 0);
    }

    /// The budget is against the WHOLE store, not one request — the case the
    /// load-time clamp it replaces could not see at all.
    #[test]
    fn headroom_counts_every_request_not_just_this_one() {
        let per_token = 1_000u64;
        let q = 512usize;
        let claim = per_token * q as u64; // 512 KB per quantum
        let budget = claim * 4; // room for four quanta in total

        // Nothing in use: fine.
        assert!(!claim_exceeds_headroom(budget, 0, per_token * q as u64));
        // Three already held: the fourth still fits.
        assert!(!claim_exceeds_headroom(
            budget,
            claim * 3,
            per_token * q as u64
        ));
        // Four already held — by any mix of requests — and the fifth does not.
        assert!(claim_exceeds_headroom(
            budget,
            claim * 4,
            per_token * q as u64
        ));
    }

    /// Arithmetic near the limits must saturate rather than wrap: wrapping
    /// would turn "no memory" into "unlimited memory".
    #[test]
    fn headroom_arithmetic_saturates() {
        assert!(claim_exceeds_headroom(0, u64::MAX, 1));
        assert!(claim_exceeds_headroom(10, u64::MAX, u64::MAX));
        // A zero claim never refuses.
        assert!(!claim_exceeds_headroom(0, 0, 0));
    }

    /// The f16 flash mirror is real VRAM and the head-room check must charge
    /// for it.
    ///
    /// `LayerKv` keeps an f16 BSHD copy of K and V for GQA models on CUDA, so
    /// those caches cost 1.5x what the f32 figure alone says. If this were left
    /// uncounted the admission check would let a model claim half again as much
    /// VRAM as the budget believes, and the clean 503 that reroutes the request
    /// to a peer would be replaced by an actual out-of-memory.
    #[test]
    fn a_mirrored_cache_is_charged_for_the_mirror() {
        let (k, v) = standard_kv_elems(8, 128);
        let plain = kv_bytes_per_token(28, k, v, KvStorage::F32);
        let mirrored = kv_bytes_per_token(28, k, v, KvStorage::F32Mirrored);
        assert_eq!(
            mirrored,
            plain + plain / 2,
            "the mirror is the same elements at half the width, so 1.5x"
        );
        // And the half cache is the mirror's width alone: a third of what it
        // replaced (FUTURE_WORK #194).
        assert_eq!(kv_bytes_per_token(28, k, v, KvStorage::F16), mirrored / 3);
        // Stated as a claim about admission, not just arithmetic: the same
        // headroom must admit fewer positions once a mirror is in play.
        let headroom = plain * 1000;
        let fits_plain = headroom / plain;
        let fits_mirrored = headroom / mirrored;
        assert!(
            fits_mirrored < fits_plain,
            "a mirrored cache must exhaust the budget sooner ({fits_mirrored} vs {fits_plain})"
        );
    }
    /// The card, not the load-time figure, has the last word when the card has
    /// less room — and the budget still caps the cache when the card has more.
    #[test]
    fn the_card_caps_the_budget_when_other_tenants_have_taken_the_room() {
        let mb = 1024 * 1024u64;
        let card = 8192 * mb;
        let budget = 4491 * mb;
        // Plenty free: the loader's budget binds.
        assert_eq!(
            budget_reconciled_with_device(budget, 0, 0, 7000 * mb, card),
            budget
        );
        // Another tenant took most of the card: the room left, less the
        // margin, is what the cache may take — well under the budget.
        let reconciled = budget_reconciled_with_device(budget, 0, 0, 1500 * mb, card);
        assert_eq!(reconciled, 1500 * mb - device_free_margin_bytes(card));
        assert!(reconciled < budget);
        // Less free than the margin: nothing, never a wrap-around.
        assert_eq!(
            budget_reconciled_with_device(budget, 0, 0, 100 * mb, card),
            0
        );
    }

    /// Releasing a cached prompt moves bytes from `cached` to `free` and
    /// leaves the reconciled budget where it was — the property that lets
    /// `admit_prompt`'s evict-then-fit arithmetic hold against it.
    #[test]
    fn evicting_cached_prompts_does_not_move_the_reconciled_budget() {
        let mb = 1024 * 1024u64;
        let card = 8192 * mb;
        let before = budget_reconciled_with_device(4491 * mb, 1000 * mb, 500 * mb, 300 * mb, card);
        let after = budget_reconciled_with_device(4491 * mb, 1000 * mb, 0, 800 * mb, card);
        assert_eq!(before, after);
    }

    /// The v0.3.149 case (gotcha #440, third half): a 4491 MB budget taken
    /// when 7541 MB were free, a 2.6 GB cached prompt, and 365 MB left on the
    /// card. The load-time budget admitted the next 6.4k-token prompt after
    /// evicting 689 MB of cache — into ~1 GB of real room for a 2.6 GB cache,
    /// which WSL2 served from host memory at 1.95 tok/s. Reconciled with the
    /// card the same prompt is refused at token 0 (the card is 25 MB short of
    /// its margin even with every cached prompt gone), and a prompt that does
    /// fit after eviction is still admitted.
    #[test]
    fn the_second_long_prompt_is_refused_where_the_load_time_budget_spilled_it() {
        let mb = 1024 * 1024u64;
        let card = 8192 * mb;
        let per_token = 344 * 1024u64;
        let budget = 4491 * mb;
        let cached = 2600 * mb;
        let positions = 7680usize;
        assert!(per_token * positions as u64 > 2500 * mb);
        // Load-time budget: evicts and admits.
        assert!(matches!(
            admit_prompt(budget, 0, cached, per_token * positions as u64),
            PromptAdmission::EvictBytes(_)
        ));
        // The card as it stood: refuse before prefill.
        let now = budget_reconciled_with_device(budget, 0, cached, 365 * mb, card);
        assert!(now < budget);
        assert!(matches!(
            admit_prompt(now, 0, cached, per_token * positions as u64),
            PromptAdmission::Refuse { .. }
        ));
        // A prompt that fits once the cache is gone is still admitted.
        assert!(matches!(
            admit_prompt(now, 0, cached, per_token * 6000),
            PromptAdmission::EvictBytes(_)
        ));
    }

    /// The escape hatch for the one change in its release that can refuse work
    /// the node previously completed. Unset means ON — a switch nobody sets
    /// must not change behaviour.
    #[test]
    fn the_processor_reconciliation_can_be_turned_off_in_the_field() {
        assert!(reconciliation_enabled_for(None), "unset means on");
        assert!(reconciliation_enabled_for(Some("1")));
        assert!(!reconciliation_enabled_for(Some("0")));
        assert!(!reconciliation_enabled_for(Some("off")));
    }

    /// #121's switch: the card is synchronized before its free memory is read
    /// unless the A/B arm turns that off. Unset means ON — the fix is the
    /// default, and a variable nobody sets must not undo it.
    #[test]
    fn the_card_is_synchronized_before_it_is_read_unless_switched_off() {
        assert!(device_sync_enabled_for(None), "unset means on");
        assert!(device_sync_enabled_for(Some("1")));
        assert!(!device_sync_enabled_for(Some("0")));
        assert!(!device_sync_enabled_for(Some("off")));
        assert!(!device_sync_enabled_for(Some("false")));
    }

    /// Gotcha #462: the processor had no live reading, so a CPU worker's KV
    /// budget was the load-time prediction for the whole life of the worker.
    /// `budget_reconciled_with_device` was already correct — it was simply
    /// never given a figure to reconcile against on this device.
    #[test]
    fn the_processor_can_now_say_how_much_memory_is_left() {
        let (free, total) = device_free_and_total_bytes(&candle_core::Device::Cpu)
            .expect("the machine can report its own memory");
        assert!(total > 0, "a total of zero means unreadable, not empty");
        assert!(
            free <= total,
            "available ({free}) cannot exceed total ({total})"
        );
    }

    /// The reading is cached, so a prefill crossing many growth quanta does
    /// not read `/proc/meminfo` once per quantum.
    #[test]
    fn the_system_memory_reading_is_reused_within_its_window() {
        let a = system_free_and_total_bytes().expect("readable");
        let b = system_free_and_total_bytes().expect("readable");
        assert_eq!(a, b, "two reads inside the TTL must be the same reading");
    }

    /// The point of the processor arm: a worker whose machine has filled up
    /// gets a smaller budget than it was handed at load, and so refuses (503,
    /// re-routed) instead of growing its cache into memory that is not there.
    #[test]
    fn a_full_machine_shrinks_the_budget_it_was_given_at_load() {
        let mb = 1024 * 1024u64;
        let total = 16384 * mb; // a 16 GB Mac mini
                                // The daemon granted 7000 MB of cache room at spawn, when the worker
                                // held 12 of 48 layers. Two failovers later it holds 29, the machine
                                // is nearly full, and the grant is no longer honourable.
        let granted = 7000 * mb;
        let live = 1200 * mb;
        let roomy = budget_reconciled_with_device(granted, live, 0, 9000 * mb, total);
        assert_eq!(roomy, granted, "control: a roomy machine changes nothing");

        let tight = budget_reconciled_with_device(granted, live, 0, 400 * mb, total);
        assert!(
            tight < granted,
            "a machine with 400 MB free must not honour a 7000 MB grant"
        );
        assert!(
            tight <= live + 400 * mb,
            "the cache may not be promised more than it holds plus what is free"
        );
    }

    /// The margin is a share of the card with a floor: small cards keep at
    /// least the floor, large cards scale.
    #[test]
    fn the_device_margin_has_a_floor_and_scales_with_the_card() {
        let gib = 1024 * 1024 * 1024u64;
        assert_eq!(
            device_free_margin_bytes(2 * gib),
            DEVICE_FREE_MARGIN_MIN_BYTES
        );
        assert_eq!(
            device_free_margin_bytes(24 * gib),
            24 * gib / 100 * DEVICE_FREE_MARGIN_PCT
        );
        assert!(device_free_margin_bytes(24 * gib) > DEVICE_FREE_MARGIN_MIN_BYTES);
    }

    /// Gemma 3's shape: five layers in six keep a 1,024-position window. Past
    /// the window and one forward they cost nothing more, so a long
    /// conversation costs the global layers' share, not every layer's
    /// (FUTURE_WORK #281). The flat rate is what every estimate charged before.
    #[test]
    fn a_windowed_layer_is_charged_no_further_than_its_window() {
        let growth = KvGrowth::with_window(100, 500, 1024, 512).widened_to(512);
        let cap = crate::inference::split::kv_cache::most_positions_a_window_holds(1024, 512, 512);
        assert_eq!(cap, 1023 + 64 + 512);
        // Inside the window every layer grows.
        assert_eq!(growth.bytes_for(1000), 600 * 1000);
        assert_eq!(growth.bytes_for(1000), KvGrowth::flat(600).bytes_for(1000));
        // Past it only the global layer does.
        assert_eq!(growth.bytes_for(8192), 100 * 8192 + 500 * cap as u64);
        assert!(growth.bytes_for(8192) < KvGrowth::flat(600).bytes_for(8192) / 2);
        assert_eq!(growth.bytes_between(8192, 8704), 100 * 512);
        // A wider forward is held beside the window.
        let wide = growth.widened_to(4096);
        assert_eq!(wide.bytes_for(32_768), 100 * 32_768 + 500 * (1087 + 4096));
        // No window: the old figure exactly.
        assert_eq!(KvGrowth::flat(600).bytes_for(8192), 600 * 8192);
        assert_eq!(KvGrowth::flat(600).per_position(), growth.per_position());
    }

    /// `positions_within` is the inverse of `bytes_for`: the longest
    /// conversation a budget holds, and not one position longer.
    #[test]
    fn the_positions_a_budget_holds_invert_what_they_cost() {
        // The cap is 1,599 positions (window 1,024: keep 1,087 + 512); every
        // budget across it, a step finer than a position's cost, and the one
        // a position past the cap still fits (600 · 1,599 + 100).
        let windowed = KvGrowth::with_window(100, 500, 1024, 512).widened_to(512);
        let across_the_cap = (600 * 1590..600 * 1610).step_by(37);
        let budgets = [0, 599, 600, 959_500, 5_000_000, 123_456_789];
        for growth in [windowed, KvGrowth::flat(600)] {
            for bytes in budgets.into_iter().chain(across_the_cap.clone()) {
                let n = growth.positions_within(bytes);
                assert!(growth.bytes_for(n) <= bytes, "{growth:?} {bytes}: {n}");
                assert!(growth.bytes_for(n + 1) > bytes, "{growth:?} {bytes}: {n}");
            }
        }
        assert_eq!(windowed.positions_within(959_500), 1600);
        assert_eq!(KvGrowth::flat(600).positions_within(6_000), 10);
        // Nothing grows past the windows of a segment that holds only
        // windowed layers — from the budget that fills them on.
        let only_windows = KvGrowth::with_window(0, 500, 1024, 512);
        assert_eq!(only_windows.positions_within(500 * 1599), usize::MAX);
        assert_eq!(only_windows.positions_within(500 * 1599 - 1), 1598);
        assert_eq!(only_windows.positions_within(u64::MAX / 2), usize::MAX);
        assert_eq!(KvGrowth::default().positions_within(1), usize::MAX);
    }

    /// A split's two memories add up, and a range's share scales both kinds.
    #[test]
    fn growths_add_and_share() {
        let card = KvGrowth::with_window(100, 500, 1024, 512);
        let host = KvGrowth::with_window(200, 0, 0, 0);
        let both = card.plus(host);
        assert_eq!(both.per_position(), 800);
        assert_eq!(
            both.bytes_for(8192),
            card.bytes_for(8192) + host.bytes_for(8192)
        );
        let half = card.share(1, 2);
        assert_eq!(half.bytes_for(8192), card.bytes_for(8192) / 2);
    }

    /// A long prompt's snapshot on a windowed model is kept whole where the
    /// flat rate cut it to an opening — it copies the windows, not every
    /// layer's whole history.
    #[test]
    fn a_windowed_snapshot_is_planned_at_what_it_copies() {
        let growth = KvGrowth::with_window(100, 500, 1024, 512).widened_to(512);
        let budget = growth.bytes_for(8192);
        assert_eq!(plan_snapshot(budget, 0, 0, growth, 8192).positions, 8192);
        let flat = plan_snapshot(budget, 0, 0, KvGrowth::flat(600), 8192);
        assert!(flat.positions < 8192 / 2, "control: {flat:?}");
    }

    /// **Every estimate before a load prices a model's caches as the loader
    /// does** (#281 (II)): each layer by its own shape, a sliding-window layer
    /// by its window. Gemma 4 12B on the processor at admission's 4,096
    /// positions: its 40 windowed layers (8 × 256) hold their window and a
    /// chunk, its 8 global ones (1 × 512) every position — 1.1 GiB, where
    /// charging every layer the widest shape at every position cost 3 GiB.
    #[test]
    fn a_layout_prices_each_layer_by_its_own_shape_and_window() {
        use crate::inference::prefill_pacer::PromptReader;
        let mut ct = super::super::gguf_meta::sliding_window_tests::gemma4_12b();
        ct.metadata.insert(
            "gemma4.attention.layer_norm_rms_epsilon".into(),
            candle_core::quantized::gguf_file::Value::F32(1e-6),
        );
        let meta = super::super::GgufTensorMeta::from_content(&ct).unwrap();
        let context = 32_768;
        let layout = KvLayout::read(&ct, &meta, context).unwrap();
        // f32 K and V, a position, per layer.
        let (windowed, global) = (2 * 8 * 256 * 4u64, 2 * 512 * 4u64);
        let growth = layout.growth(0, 48, false);
        assert_eq!(growth.per_position(), 40 * windowed + 8 * global);
        // A range is its own layers: 6..12 holds five windowed and one global.
        assert_eq!(
            layout.growth(6, 12, false).per_position(),
            5 * windowed + global
        );

        let read_in = |chunk| {
            layout
                .price(0, 48, chunk)
                .reading(PromptReader::Processor, context)
        };
        let held = crate::inference::split::kv_cache::most_positions_a_window_holds(
            1024,
            crate::inference::layers::kv_growth_quantum(context),
            128,
        );
        assert!(held < 4096, "{held}");
        assert_eq!(
            read_in(128).bytes_for(4096),
            8 * global * 4096 + 40 * windowed * held as u64
        );
        // A prompt read in ONE forward hands a windowed cache the whole
        // context: nothing slides out from under it.
        assert_eq!(read_in(0).bytes_for(4096), growth.per_position() * 4096);
        // What every estimate charged before: the widest layer's shape on
        // every layer, at every position.
        let flat = KvPrice::uniform(
            48,
            meta.head_count_kv as u64,
            meta.head_count as u64,
            meta.head_dim as u64,
            crate::inference::layers::KvAttention::Standard,
        );
        let before = flat.processor.bytes_for(4096);
        assert_eq!(before, 48 * windowed * 4096);
        assert_eq!(read_in(128).bytes_for(4096), 1_182_138_368);
        assert_eq!(before, 3 << 30);
    }
}
