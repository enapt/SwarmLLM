// ── Split model: loads only a range of layers from a GGUF ──

use crate::inference::residual_norm::RmsNorm;
use candle_core::{Device, Tensor};

use crate::error::SwarmError;

use super::kv_budget::KvGrowth;
use super::kv_cache::{CacheLayout, KvCacheStore, KvHome};
use super::token_embedding::TokenEmbedding;
use super::{LayerVariant, ModelArch, QMatMul, SplitTokenizer};

/// A partial transformer model that loads and runs only a specific range of layers.
/// Used for split inference where each node holds different layers.
/// Supports multiple architectures: Llama, Qwen2, Gemma 2, Phi-3, Mistral, Qwen 3.5.
pub struct SplitModel {
    /// Token embedding table (only loaded by the first segment).
    ///
    /// Quantized with rows read on demand where that is possible, dense
    /// otherwise — see [`TokenEmbedding`]. Both answer `forward` identically,
    /// so nothing below this field needs to know which it got.
    pub(super) tok_embeddings: Option<TokenEmbedding>,
    /// Transformer layers for this segment's range.
    pub(super) layers: Vec<LayerVariant>,
    /// Final RMSNorm (only loaded by the last segment).
    pub(super) norm: Option<RmsNorm>,
    /// LM head / output projection (only loaded by the last segment).
    pub(super) output: Option<QMatMul>,
    /// Additive causal mask for the no-prefix case, cached at its EXACT size.
    /// Tuple is `(query_len, mask)`; `None` means nothing cached yet. See
    /// `SplitModel::mask` for why this is not a narrowed view of a bigger one.
    pub(super) masks: Option<(usize, Tensor)>,
    /// Layer range this model covers: [start, end) out of total_layers.
    pub layer_start: usize,
    pub layer_end: usize,
    pub total_layers: usize,
    /// Hidden dimension (embedding_length).
    pub hidden_dim: usize,
    /// Detected model architecture.
    pub arch: ModelArch,
    /// Device (CPU or CUDA). The segment's PRIMARY device: where the
    /// embedding table, final norm and output head live, and where a forward
    /// pass starts and ends.
    pub(super) device: Device,
    /// Per-layer devices, when this segment is split across two of them.
    ///
    /// **Empty for the overwhelmingly common case** of a segment that fits on
    /// one device, so the forward loop can skip all placement work with a
    /// single `is_empty` check rather than comparing devices per layer on the
    /// hot path. See `hybrid::LayerPlacement`.
    pub(super) layer_devices: Vec<Device>,
    /// Vocabulary from GGUF (token ID → string), for decoding generated tokens.
    pub(super) vocabulary: Option<Vec<String>>,
    /// Tokenizer (BPE or sentencepiece/unigram) built from GGUF metadata.
    pub(super) tokenizer: Option<SplitTokenizer>,
    /// EOS token IDs loaded from GGUF metadata.
    pub(super) eos_tokens: Vec<u32>,
    /// Chat template from GGUF `tokenizer.chat_template` (Jinja2 format).
    pub(super) chat_template: Option<String>,
    /// BOS token string from GGUF metadata.
    pub(super) bos_token: String,
    /// EOS token string from GGUF metadata.
    pub(super) eos_token: String,
    /// Maximum sequence length for KV cache pre-allocation.
    pub(super) max_seq_len: usize,
    /// Bytes of GPU memory this segment's KV cache may occupy in total,
    /// across every concurrent request. `None` on CPU, and on CUDA when free
    /// VRAM could not be read — an unknown budget must not become a zero one.
    ///
    /// Checked before a forward claims another growth quantum. This replaced a
    /// load-time clamp that shrank the model's usable context so ONE
    /// conversation at full length would fit: that made every user's context
    /// shorter to guard against a case most never reach, and did not bound
    /// concurrency at all.
    pub(super) kv_budget_bytes: Option<u64>,
    /// What a conversation's cache costs as it grows over the layers `device`
    /// holds — a card/processor split's CARD-held layers only, a sliding
    /// window's layers charged no further than their window (FUTURE_WORK
    /// #281). Paired with `kv_budget_bytes`; nothing when unknown. Read
    /// through [`Self::kv_budget`], which hands it the widest forward.
    pub(super) kv_growth: KvGrowth,
    /// A card/processor split's second budget (#262): what its processor-held
    /// layers' caches may take of the machine's memory — the daemon's
    /// `--kv-budget-bytes` when it handed one, else what the machine had
    /// available at load less the margin — reconciled with the machine at
    /// every claim, as the card's is with the card. `None` for a model on one
    /// device, and when the machine's memory cannot be read.
    pub(super) host_kv_budget_bytes: Option<u64>,
    /// Paired with `host_kv_budget_bytes`: what a conversation's cache costs
    /// the processor-held layers of a split. Nothing for a model on one
    /// device.
    pub(super) host_kv_growth: KvGrowth,
    /// Pre-computed KV cache store key: "{layer_start}-{layer_end}-{total_layers}".
    /// Avoids a `format!` allocation on every forward pass.
    pub(super) kv_model_key: String,
    /// Gemma 2 final logit soft-capping value (e.g. 30.0).
    pub(super) final_logit_softcap: Option<f32>,
    /// What the head adds to every row of logits so the tokens the model file
    /// lists in `tokenizer.ggml.suppress_tokens` are never chosen (#282) — a
    /// vocabulary-wide row on the head's device, [`SUPPRESSED_LOGIT`] at each
    /// listed id, 0 elsewhere. `None` without the key (every family but Gemma 4
    /// today), and off the last segment.
    ///
    /// [`SUPPRESSED_LOGIT`]: super::executor::SUPPRESSED_LOGIT
    pub(super) suppressed_logits: Option<Tensor>,
    /// How often `forward_batch` actually batched, versus fell back to running
    /// its items one at a time.
    ///
    /// The fallback is silent and its conditions are easy to meet by accident —
    /// requests only share an `index_pos` while they stay in lockstep, which
    /// concurrent requests stop doing as soon as one starts a token earlier
    /// than another. Without a count there is no way to tell a node that is
    /// batching from one that has been running sequentially all along, and
    /// measured throughput does not distinguish them either: batching four
    /// streams was worth about 20% here, which is well inside the noise of a
    /// single benchmark (`docs/FUTURE_WORK.md`).
    pub(super) batch_calls: u64,
    pub(super) batch_fellback: u64,
    /// When the counters above were last reported. `None` until the first
    /// multi-request call, so an idle model never reports.
    pub(super) batch_stats_reported_at: Option<std::time::Instant>,
    /// Whether this model's decode steps go to the card as one CUDA graph,
    /// and the graph if so — see [`crate::inference::cuda_graph`]. Decided at
    /// the first decode step; `SWARMLLM_CUDA_GRAPH=1` only.
    pub(super) decode_graph: crate::inference::cuda_graph::DecodeGraphSlot,
    /// What sizes a prompt chunk when this segment's processor-held layers
    /// copy their weights to the card (#138 (1e)) — [`Self::prompt_reader`].
    pub(super) prompt_room: PromptRoom,
}

/// What sizes a prompt chunk on a split copying its processor-held layers'
/// weights to the card (#138 (1e)): a per-token slope from the layers' shapes
/// and the last prompt forward the card's memory pool measured.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct PromptRoom {
    /// `layers::prompt_bytes_per_token`: the largest copying layer's
    /// activations a token. 0 where no processor-held layer copies its
    /// weights — nothing is then measured or sized.
    pub(crate) bytes_per_token: u64,
    /// `layers::prompt_stack_bytes`: what the forward's expert stacks hold —
    /// what a first forward is predicted to take before its tokens.
    pub(crate) stack_bytes: u64,
    /// The last prompt forward measured on the card.
    pub(crate) last: Option<MeasuredPrompt>,
}

/// One prompt forward as the card's memory pool saw it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MeasuredPrompt {
    /// Its tokens and the position it started at.
    pub(crate) tokens: usize,
    pub(crate) index_pos: usize,
    /// The most the pool held during it beyond what was in use as it began
    /// (`USED_MEM_HIGH` less `USED_MEM_CURRENT`), less the cache it added:
    /// the weights it copied over and its activations. The cache is left out
    /// because what a forward adds depends on what the request already holds,
    /// not on the forward — a reserved request's first chunk allocates every
    /// position it was admitted for and its later chunks none (#292), so a
    /// take counting it charged each later chunk the whole reservation again.
    pub(crate) took_bytes: u64,
}

/// What [`SplitModel::begin_prompt_measure`] hands
/// [`SplitModel::end_prompt_measure`]: the pool's use as the forward began and
/// the request's cache on the card then.
#[derive(Debug, Clone, Copy)]
pub(super) struct PromptMeasure {
    in_use: u64,
    cache_bytes: u64,
}

/// What each further prompt token costs a forward on the card.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PromptTokenBytes {
    /// The largest copying layer's activations.
    pub(crate) activations: u64,
    /// What the card-held layers' caches keep of it.
    pub(crate) kept: u64,
    /// What one card-run processor layer pulls up of its cache per position
    /// already held — the prompt's history, charged per position, not per
    /// token of the chunk.
    pub(crate) history: u64,
}

impl MeasuredPrompt {
    /// The card memory a prompt forward of `tokens` starting at `index_pos`
    /// will take beyond what is in use as it begins, predicted from this one:
    /// its measured take, plus or minus `per`'s activations for each token and
    /// its history for each position of history it differs by, the
    /// difference in causal masks — f32 `[tokens, index_pos + tokens]`,
    /// counted twice for the copy a card-run processor layer takes of it
    /// (`SplitModel::causal_mask`), the one term that grows faster than the
    /// chunk — and what the caches keep of every token it reads.
    pub(crate) fn predicted_bytes(
        &self,
        per: PromptTokenBytes,
        tokens: usize,
        index_pos: usize,
    ) -> u64 {
        let mask = |t: usize, at: usize| 8 * t as i128 * (at + t) as i128;
        let took = i128::from(self.took_bytes)
            + (tokens as i128 - self.tokens as i128) * i128::from(per.activations)
            + tokens as i128 * i128::from(per.kept)
            + ((index_pos + tokens) as i128 - (self.index_pos + self.tokens) as i128)
                * i128::from(per.history)
            + mask(tokens, index_pos)
            - mask(self.tokens, self.index_pos);
        took.clamp(0, i128::from(u64::MAX)) as u64
    }
}

/// `SWARMLLM_PROMPT_CHUNK_ROOM=0` reads a split's prompt 512 tokens a forward,
/// as before #138 (1e) — the A/B of [`SplitModel::prompt_reader`]. Read once.
fn prompt_chunks_sized_by_room() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("SWARMLLM_PROMPT_CHUNK_ROOM").as_deref() != Ok("0"))
}

/// Logged once per process: the first prompt chunk a split sized from the
/// card's room, and what it was sized from (#138 (1e)).
fn note_prompt_chunk_sized(
    tokens: usize,
    room: u64,
    budget: u64,
    last_took: u64,
    per: PromptTokenBytes,
) {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        tracing::info!(
            tokens,
            room_mb = room >> 20,
            budget_mb = budget >> 20,
            last_forward_took_mb = last_took >> 20,
            activation_kb_per_token = per.activations >> 10,
            kept_kb_per_token = per.kept >> 10,
            history_kb_per_position = per.history >> 10,
            "DIAG: a split's prompt chunk sized from the card's room"
        )
    });
}

impl SplitModel {
    /// How many of this segment's layers work on the processor: held there,
    /// or held on the card with their routed experts kept in the machine's
    /// memory (#138 (3)), which the processor runs every decode step.
    ///
    /// This is the number the decode-thread calibration is keyed on
    /// (`cpu_pools::in_phase_pool`): zero for a segment entirely on the card,
    /// the whole segment on a processor-only node, and the processor's share
    /// of a hybrid split. It has to describe THIS forward, not the worker —
    /// one worker serves both a full hybrid model and one-layer card-only
    /// segments of it, and timing the second as though it were the first is
    /// gotcha #432.
    pub(super) fn cpu_layer_count(&self) -> usize {
        if self.device.is_cpu() && self.layer_devices.is_empty() {
            return self.layers.len();
        }
        self.layers
            .iter()
            .enumerate()
            .filter(|(i, layer)| self.works_on_the_processor(*i, layer))
            .count()
    }

    /// Does layer `i` of this segment work on the processor — held there, or
    /// its experts kept in RAM? Never asked of a processor-only segment.
    fn works_on_the_processor(&self, i: usize, layer: &super::LayerVariant) -> bool {
        self.layer_devices.get(i).is_some_and(|d| d.is_cpu()) || layer.experts_on_the_processor()
    }

    /// Does every layer of this segment run on the graphics card?
    ///
    /// What `inference::card_pace` asks before it counts a step's time: a step
    /// with any processor layer can take seconds for reasons that are not the
    /// card's (gotcha #191), so only an all-card segment's time says anything
    /// about the card.
    pub(crate) fn runs_entirely_on_card(&self) -> bool {
        !self.device.is_cpu() && self.cpu_layer_count() == 0
    }

    /// Does this segment read a prompt's heavy work on the graphics card? Every
    /// layer there, or a segment on a card whose every layer working on the
    /// processor — a split's processor layers, a card layer's experts kept in
    /// RAM (#138 (3)) — sends a prompt's heavy work to the card
    /// (`LayerVariant::reads_its_prompt_on_a_card`: its projections, #138 (1b),
    /// or its experts, #138 (1)). What [`Self::prompt_reader`] asks — never
    /// [`Self::runs_entirely_on_card`] alone, which the drafter and admission
    /// timing ask: such a segment reads at least 512 tokens a forward, so each
    /// RAM-held weight crosses to the card once per chunk, not per 128 tokens.
    pub(crate) fn reads_prompts_on_card(&self) -> bool {
        // A split (it has a card by construction), or a segment on one.
        let beside_a_card = !self.layer_devices.is_empty() || self.card().is_some();
        self.runs_entirely_on_card()
            || (beside_a_card
                && self
                    .layers
                    .iter()
                    .enumerate()
                    .filter(|(i, layer)| self.works_on_the_processor(*i, layer))
                    .all(|(_, layer)| layer.reads_its_prompt_on_a_card()))
    }

    /// What reads a prompt chunk starting at `index_pos` on this segment — the
    /// ONE answer every prompt pass sizes its chunk by
    /// (`prefill_pacer::prompt_chunk_ceiling`): the forward's own chunk loop
    /// and the batched table's solo prompt. A split copying its processor
    /// layers' weights to the card is told how many tokens the card has room
    /// for NOW (#138 (1e)): it re-copies every one of them per forward, so a
    /// 2,664-token prompt on the 14B took 10.05 s at 512 a forward, 6.93 s at
    /// 1,024 and 5.81 s whole (`docs/invariants/inference.md`).
    pub(crate) fn prompt_reader(
        &self,
        index_pos: usize,
    ) -> crate::inference::prefill_pacer::PromptReader {
        use crate::inference::prefill_pacer::PromptReader;
        if self.runs_entirely_on_card() {
            PromptReader::Card
        } else if self.reads_prompts_on_card() {
            PromptReader::CardCopyingWeights {
                room_tokens: self.prompt_room_tokens(index_pos),
            }
        } else {
            PromptReader::Processor
        }
    }

    /// How many prompt tokens one forward starting at `index_pos` can read
    /// before the card runs short, or 0 when that is not known — no forward
    /// measured yet, a card that will not say, a prompt layer that has already
    /// fallen back to the processor (out of the card's memory, most likely:
    /// `layers::prompt_card_fell_back`), or standard attention forced (its
    /// scores grow with the square of the chunk). 0 reads at
    /// `CARD_CHUNK_TOKENS`, as before this was measured.
    ///
    /// The card's room is what it has free beside what is in use now, after a
    /// synchronize (`kv_budget::device_free_and_total_bytes`), less the margin
    /// every budget on the card keeps; a forward of `t` tokens is predicted
    /// from the last one measured ([`MeasuredPrompt::predicted_bytes`]). Every
    /// forward of a chunk measures again, so a prediction that was wrong is
    /// corrected at the next chunk. `SWARMLLM_PROMPT_CHUNK_ROOM=0` = 512 a
    /// forward, the A/B.
    fn prompt_room_tokens(&self, index_pos: usize) -> usize {
        use crate::inference::prefill_pacer::{CARD_CHUNK_TOKENS, COPIED_CHUNK_MAX_TOKENS};
        let Some(last) = self.prompt_room.last else {
            return 0;
        };
        let Some(card) = self.card() else {
            return 0;
        };
        if !prompt_chunks_sized_by_room()
            || crate::inference::layers::prompt_card_fell_back()
            || crate::inference::attn_kernel::is_force_standard_attn()
        {
            return 0;
        }
        let Some((room, total)) = super::kv_budget::device_free_and_total_bytes(card) else {
            return 0;
        };
        let budget = room.saturating_sub(super::kv_budget::device_free_margin_bytes(total));
        let per_token = self.prompt_bytes_per_token_on_card();
        let fits = |t: usize| last.predicted_bytes(per_token, t, index_pos) <= budget;
        let (mut fit, mut over) = (CARD_CHUNK_TOKENS, COPIED_CHUNK_MAX_TOKENS + 1);
        if !fits(fit) {
            return 0;
        }
        while over - fit > 1 {
            let mid = fit + (over - fit) / 2;
            if fits(mid) {
                fit = mid;
            } else {
                over = mid;
            }
        }
        note_prompt_chunk_sized(fit, room, budget, last.took_bytes, per_token);
        fit
    }

    /// The card bytes a prompt forward's every further token costs this
    /// segment: the largest copying layer's activations
    /// (`PromptRoom::bytes_per_token`) and what the card-held layers' caches
    /// keep of it.
    fn prompt_bytes_per_token_on_card(&self) -> PromptTokenBytes {
        // `kv_growth` is the card-held layers' figure (the loader's budget
        // counts only what the card holds), so a layer's share is it over the
        // CARD's layers, never over the segment's. A position as it costs
        // while every cache still grows: a forward wider than a window's
        // quantum grows the windowed caches too.
        let kept = self.kv_growth.per_position();
        let on_card = self.layer_devices.iter().filter(|d| d.is_cuda()).count() as u64;
        let per_layer_kv = kept / on_card.max(1);
        PromptTokenBytes {
            activations: self.prompt_room.bytes_per_token,
            kept,
            // A card-run processor layer pulls its cache history up for the
            // chunk (#138 (1c)): as f32 and again as the attention kernel's
            // f16, against a figure kept at f16 (2 bytes) or f32 (4) — three
            // times it covers either. A segment with no processor-held layer
            // (its experts alone in RAM, #138 (3)) keeps every cache on the
            // card and pulls none up.
            history: if self.layer_devices.iter().any(|d| d.is_cpu()) {
                per_layer_kv * 3
            } else {
                0
            },
        }
    }

    /// Measure a prompt forward of `seq_len` tokens at `index_pos`, when this
    /// segment copies its weights to the card for one: synchronize, so what
    /// is in use is what stays, prime the pool for it
    /// ([`Self::prime_pool_for_prompt`]), then start the pool's high-water
    /// mark again — after the prime, which is no part of the forward. `None`
    /// when it is not measured. `cache_key` and `kv_reserve` are the
    /// request's, as the forward's own head-room claim reads them.
    pub(super) fn begin_prompt_measure(
        &self,
        store: &KvCacheStore,
        cache_key: &str,
        kv_reserve: usize,
        seq_len: usize,
        index_pos: usize,
    ) -> Option<PromptMeasure> {
        if self.prompt_room.bytes_per_token == 0
            || seq_len < crate::inference::layers::PROMPT_ON_CARD_MIN_TOKENS
        {
            return None;
        }
        let card = self.card()?;
        card.synchronize().ok()?;
        let claiming =
            self.kv_positions_a_forward_adds(store, cache_key, kv_reserve, index_pos + seq_len);
        let cache_adds = self.kv_bytes_a_forward_adds(
            store,
            cache_key,
            index_pos,
            seq_len,
            claiming,
            KvHome::Card,
        );
        self.prime_pool_for_prompt(card, seq_len, index_pos, cache_adds);
        crate::inference::cuda_pool::take_peak_use(card)?;
        Some(PromptMeasure {
            in_use: crate::inference::cuda_pool::usage(card)?.used,
            cache_bytes: store.allocated_bytes_of(cache_key, KvHome::Card),
        })
    }

    /// Grow the card's pool once, before the forward copies anything, by what
    /// it will take (`cuda_pool::prime`, #292): its copies and activations
    /// predicted from the last forward measured, else from the expert stacks
    /// and each token's activations alone, plus the `cache_adds` bytes its
    /// caches newly allocate on the card — all a reserved request's positions
    /// at its first chunk, which a per-token figure for the chunk missed by
    /// 8.6 s of growth at its first allocation on Gemma 4's 26B — a quarter
    /// over for the pool's fragments, and never past the card's room less the
    /// margin every budget on it keeps. A prediction short of the truth leaves
    /// the rest to grow as before.
    ///
    /// A pool whose free memory already covers it is left alone: asked for as
    /// ONE block, that memory is in pieces around what the last forward kept,
    /// and the driver answers a request no piece fits by remapping the pieces
    /// together or growing (the CUDA Programming Guide, § "Stream-Ordered
    /// Memory Allocator") — a 595 MB prime with ~790 MB free stalled 11.2 s
    /// (#292). Short of it, the pool grows by the whole figure where the card
    /// has that much outside the pool, so the forward finds it in one piece,
    /// and by the shortfall where it has not.
    fn prime_pool_for_prompt(
        &self,
        card: &Device,
        seq_len: usize,
        index_pos: usize,
        cache_adds: u64,
    ) {
        let baseline = self.prompt_room.last.unwrap_or(MeasuredPrompt {
            tokens: 0,
            index_pos,
            took_bytes: self.prompt_room.stack_bytes,
        });
        let per = PromptTokenBytes {
            kept: 0,
            ..self.prompt_bytes_per_token_on_card()
        };
        let want = baseline
            .predicted_bytes(per, seq_len, index_pos)
            .saturating_add(cache_adds);
        let Some((room, total)) = super::kv_budget::device_free_and_total_bytes(card) else {
            return;
        };
        let budget = room.saturating_sub(super::kv_budget::device_free_margin_bytes(total));
        let bytes = (want + want / 4).min(budget);
        let pool_free = crate::inference::cuda_pool::reusable_bytes(card);
        if pool_free >= bytes {
            tracing::debug!(
                mb = bytes >> 20,
                pool_free_mb = pool_free >> 20,
                seq_len,
                index_pos,
                "card pool already holds a prompt forward's memory"
            );
            return;
        }
        let grow = if bytes <= budget.saturating_sub(pool_free) {
            bytes
        } else {
            bytes - pool_free
        };
        if let Some(took) = crate::inference::cuda_pool::prime(card, grow) {
            tracing::debug!(
                mb = grow >> 20,
                cache_mb = cache_adds >> 20,
                pool_free_mb = pool_free >> 20,
                took_ms = took.as_millis() as u64,
                seq_len,
                index_pos,
                "card pool primed for a prompt forward"
            );
        }
    }

    /// Record what the forward begun by [`Self::begin_prompt_measure`] took,
    /// less the cache it added to the card ([`MeasuredPrompt::took_bytes`]).
    pub(super) fn end_prompt_measure(
        &mut self,
        began: Option<PromptMeasure>,
        store: &KvCacheStore,
        cache_key: &str,
        seq_len: usize,
        index_pos: usize,
    ) {
        let Some(began) = began else {
            return;
        };
        let Some(peak) = self.card().and_then(crate::inference::cuda_pool::peak_use) else {
            return;
        };
        let cache_added = store
            .allocated_bytes_of(cache_key, KvHome::Card)
            .saturating_sub(began.cache_bytes);
        self.prompt_room.last = Some(MeasuredPrompt {
            tokens: seq_len,
            index_pos,
            took_bytes: peak
                .saturating_sub(began.in_use)
                .saturating_sub(cache_added),
        });
    }

    /// The cache positions a forward reaching `total_seq` newly allocates for
    /// the request under `cache_key`: its admitted reservation (`kv_reserve`)
    /// at the first append, a quantum at a time past what it holds, none
    /// inside it (`kv_budget::positions_to_allocate`). Read off the buffer the
    /// request holds, not derived from `index_pos`: a reserved cache is larger
    /// than the rounded position from its first append.
    pub(super) fn kv_positions_a_forward_adds(
        &self,
        store: &KvCacheStore,
        cache_key: &str,
        kv_reserve: usize,
        total_seq: usize,
    ) -> usize {
        super::kv_budget::positions_to_allocate(
            store.allocated_positions(cache_key),
            kv_reserve,
            total_seq,
            crate::inference::layers::kv_growth_quantum(self.max_seq_len),
        )
    }

    /// The bytes those `claiming` positions take in `home` for a forward of
    /// `seq_len` tokens at `index_pos`, read off the request's buffers — a
    /// sliding window's slide included, which no per-position rate sees
    /// (FUTURE_WORK #281). Before the first append there are no buffers, and
    /// the model's estimate for a forward this wide prices the first
    /// allocation. The one figure the forward's head-room claim and the
    /// pool's prime (#292) are given.
    pub(super) fn kv_bytes_a_forward_adds(
        &self,
        store: &KvCacheStore,
        cache_key: &str,
        index_pos: usize,
        seq_len: usize,
        claiming: usize,
        home: KvHome,
    ) -> u64 {
        if claiming == 0 {
            return 0;
        }
        store
            .bytes_a_forward_adds(cache_key, index_pos, seq_len, home)
            .unwrap_or_else(|| {
                self.kv_budget(home)
                    .1
                    .widened_to(seq_len)
                    .bytes_for(claiming)
            })
    }

    /// The model's context window in tokens.
    ///
    /// Exposed so a caller that holds the WHOLE prompt can check it before
    /// starting a chunked prefill. The executor's own guard sees one chunk at a
    /// time and can only report a position just past the limit, which is the
    /// same number for every over-long prompt.
    pub fn context_window(&self) -> usize {
        self.max_seq_len
    }

    /// Return a reference to the loaded vocabulary, if available.
    pub fn vocab(&self) -> Option<&Vec<String>> {
        self.vocabulary.as_ref()
    }

    /// Return a reference to the tokenizer, if available.
    pub fn tokenizer(&self) -> Option<&SplitTokenizer> {
        self.tokenizer.as_ref()
    }

    /// Return the device this model is loaded on.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// The card this model uses, if any: its primary device, or the card half
    /// of a card/processor split.
    pub(crate) fn card(&self) -> Option<&Device> {
        std::iter::once(&self.device)
            .chain(&self.layer_devices)
            .find(|d| d.is_cuda())
    }

    /// The order this model keeps query and key rows in, for an adapter's
    /// `B` matrices to be put into — see `lora::QkRowOrder`. Read off the
    /// first dense layer; a model with none has no layer an adapter reaches.
    pub fn lora_qk_row_order(&self) -> crate::model::lora::QkRowOrder {
        self.layers
            .iter()
            .find_map(|l| match l {
                LayerVariant::Dense(w) => Some(crate::model::lora::QkRowOrder::for_arch(
                    &self.arch,
                    w.n_head,
                    w.n_kv_head,
                )),
                _ => None,
            })
            .unwrap_or(crate::model::lora::QkRowOrder::AsTrained)
    }

    /// Record one `forward_batch` call and periodically report how much of the
    /// batching is real.
    ///
    /// Reported at INFO, not debug: nodes run at info, and a metric nobody sees
    /// is the same as no metric — the whole reason this fallback went unnoticed
    /// is that it is silent.
    ///
    /// **Paced by time, not only by call count.** It used to report every 256
    /// calls, and a realistic session never gets there: four people asking one
    /// question each is around 96, so the counter reset with the worker and the
    /// line never appeared. Verifying the batching fix on a real node found
    /// exactly that — the answers were right, and the one diagnostic that could
    /// have said whether the batched path ran at all printed nothing.
    ///
    /// A diagnostic that cannot fire during normal use is not a diagnostic. It
    /// now reports on whichever comes first, so a short burst is visible and a
    /// busy node still gets at most one line per `BATCH_STATS_MIN_GAP`.
    pub(super) fn note_batch_attempt(&mut self, fell_back: bool) {
        /// Roughly one line per few seconds of steady decoding.
        const BATCH_STATS_EVERY: u64 = 256;
        /// Floor on spacing, so a busy node cannot be flooded.
        const BATCH_STATS_MIN_GAP: std::time::Duration = std::time::Duration::from_secs(20);

        self.batch_calls += 1;
        if fell_back {
            self.batch_fellback += 1;
        }
        let now = std::time::Instant::now();
        let due_by_time = match self.batch_stats_reported_at {
            // First multi-request call since load: start the clock rather than
            // reporting a single sample, which says nothing.
            None => {
                self.batch_stats_reported_at = Some(now);
                false
            }
            Some(last) => now.duration_since(last) >= BATCH_STATS_MIN_GAP,
        };
        if !due_by_time && !self.batch_calls.is_multiple_of(BATCH_STATS_EVERY) {
            return;
        }
        self.batch_stats_reported_at = Some(now);
        let batched = self.batch_calls - self.batch_fellback;
        tracing::info!(
            model_key = %self.kv_model_key,
            calls = self.batch_calls,
            batched,
            fell_back = self.batch_fellback,
            batched_pct = (batched as f64 * 100.0 / self.batch_calls as f64).round() as u32,
            "DIAG: forward_batch — share of multi-request calls that actually batched \
             (the rest ran their items one at a time because the requests were not at \
             the same position)"
        );
    }

    /// `(calls, fell_back)` for `forward_batch` since this model was loaded.
    pub fn batch_stats(&self) -> (u64, u64) {
        (self.batch_calls, self.batch_fellback)
    }

    /// The memory this model's primary KV budget is drawn from: its device's.
    pub(crate) fn kv_home(&self) -> KvHome {
        KvHome::of(&self.device)
    }

    /// Every memory this model's caches are charged to: its device's, and for
    /// a card/processor split the machine's as well (#262). A decision about
    /// KV room asks EACH: one that weighed only the card let the processor-held
    /// caches grow against nothing, and one that weighed the card with both
    /// devices' bytes refused a split at a quarter of what its card held.
    pub(crate) fn kv_homes(&self) -> impl Iterator<Item = KvHome> {
        let primary = self.kv_home();
        let host = (primary == KvHome::Card && self.host_kv_growth.per_position() > 0)
            .then_some(KvHome::Host);
        std::iter::once(primary).chain(host)
    }

    /// The runtime KV budget in `home` and what a conversation's cache costs
    /// against it: `(budget_bytes, growth)`. `None` budget means unknown, and
    /// an unknown budget admits everything (never a zero one); `(None,
    /// nothing)` for a memory this model keeps no cache in. A sliding window's
    /// caches are charged their window and this segment's widest prompt
    /// forward ([`Self::widest_prompt_forward`]) — here, so no caller can
    /// charge a window narrower than the caches are allocated.
    pub(crate) fn kv_budget(&self, home: KvHome) -> (Option<u64>, KvGrowth) {
        let (budget, growth) = if home == self.kv_home() {
            (self.kv_budget_bytes, self.kv_growth)
        } else if home == KvHome::Host {
            (self.host_kv_budget_bytes, self.host_kv_growth)
        } else {
            (None, KvGrowth::default())
        };
        if growth.has_window() {
            (budget, growth.widened_to(self.widest_prompt_forward()))
        } else {
            (budget, growth)
        }
    }

    /// The most positions one prompt forward of this segment reads: the chunk
    /// every prompt pass is cut to (`prefill_pacer::widest_prompt_forward`,
    /// from this worker's `inference.prefill_chunk_tokens`), at the most a
    /// split copying its weights to the card reads — or the whole context
    /// where prompts are read in one forward. A sliding window's cache holds
    /// its window AND this (`kv_cache::most_positions_a_window_holds`).
    pub(crate) fn widest_prompt_forward(&self) -> usize {
        use crate::inference::prefill_pacer::{
            widest_prompt_forward, PromptReader, COPIED_CHUNK_MAX_TOKENS,
        };
        let reader = if self.runs_entirely_on_card() {
            PromptReader::Card
        } else if self.reads_prompts_on_card() {
            PromptReader::CardCopyingWeights {
                room_tokens: COPIED_CHUNK_MAX_TOKENS,
            }
        } else {
            PromptReader::Processor
        };
        widest_prompt_forward(
            super::PROMPT_CHUNK_TOKENS.load(std::sync::atomic::Ordering::Relaxed),
            reader,
            self.max_seq_len,
        )
    }

    /// How each of this segment's layers keeps its cache — what a cache
    /// restored from a snapshot is built as
    /// (`PrefixCache::hydrate_request_from_snapshot`), which the snapshot cannot
    /// carry. `None` for a layer whose cache a forward conforms on its own
    /// (MLA, Qwen 3.5's attention layers: no window).
    pub(crate) fn cache_layouts(&self) -> Vec<Option<CacheLayout>> {
        self.layers
            .iter()
            .enumerate()
            .map(|(i, layer)| match layer {
                LayerVariant::Dense(lw) => {
                    Some(lw.cache_layout_on(self.layer_devices.get(i).unwrap_or(&self.device)))
                }
                _ => None,
            })
            .collect()
    }

    /// The KV budget as the device can honour it NOW, for a decision about to
    /// take memory: the load-time figure capped by what the KV cache already
    /// holds (`live_bytes` in the store, `cached_bytes` of prefix-cache
    /// snapshots) plus what the device has left free, less a margin. See
    /// `kv_budget::budget_reconciled_with_device` for why the load-time figure
    /// alone over-promised by ~2 GB on an 8 GB card — and, on the processor,
    /// let a worker keep filling a cache sized for a third of the layers it
    /// had ended up holding (gotcha #462).
    ///
    /// Same contract as [`Self::kv_budget`]: `None` means unknown and admits
    /// everything. A device that cannot report its memory (a driver that will
    /// not say, a machine whose totals read zero) leaves the load-time budget
    /// as it was.
    ///
    /// `live_bytes` are the store's bytes IN `home` (`KvOccupancy::allocated_in`).
    pub(crate) fn kv_budget_now(
        &self,
        home: KvHome,
        live_bytes: u64,
        cached_bytes: u64,
    ) -> (Option<u64>, KvGrowth) {
        let (Some(budget), growth) = self.kv_budget(home) else {
            return (None, self.kv_budget(home).1);
        };
        let processor = Device::Cpu;
        let device = if home == self.kv_home() {
            &self.device
        } else {
            &processor
        };
        let reconciled = match super::kv_budget::device_free_and_total_bytes(device) {
            Some((free, total)) => super::kv_budget::budget_reconciled_with_device(
                budget,
                live_bytes,
                cached_bytes,
                free,
                total,
            ),
            None => budget,
        };
        if reconciled < budget {
            tracing::debug!(
                load_time_budget_mb = budget / (1024 * 1024),
                budget_now_mb = reconciled / (1024 * 1024),
                live_mb = live_bytes / (1024 * 1024),
                cached_mb = cached_bytes / (1024 * 1024),
                device = match home {
                    KvHome::Host => "processor",
                    KvHome::Card => "card",
                },
                "DIAG: KV budget reconciled with the device — less room than the load-time figure"
            );
        }
        (Some(reconciled), growth)
    }

    /// Return the KV cache model key (used for cache cleanup).
    pub fn kv_model_key(&self) -> &str {
        &self.kv_model_key
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

    /// Tokenize a prompt to token IDs as `Vec<u32>`. Used by the prefix
    /// cache for key lookup and by the batched-decode path. Cheap — it's
    /// the same work `tokenize()` does, minus the tensor construction.
    pub fn encode_ids(&self, prompt: &str) -> Vec<u32> {
        if let Some(ref tokenizer) = self.tokenizer {
            tokenizer
                .encode(prompt)
                .into_iter()
                .map(|t| t as u32)
                .collect()
        } else {
            prompt.bytes().map(|b| b as u32).collect()
        }
    }

    /// Does this model carry RECURRENT state between forwards — state a later
    /// token cannot be taken back out of? Qwen 3.5's DeltaNet layers do. The
    /// KV cache of an attention layer can be truncated after a rejected
    /// speculative draft; a recurrent state cannot, so no speculative path may
    /// run on such a model (FUTURE_WORK #117).
    pub fn carries_recurrent_state(&self) -> bool {
        self.layers
            .iter()
            .any(|l| matches!(l, super::LayerVariant::Qwen35Ssm { .. }))
    }

    /// Build an input tensor from an already-tokenized token id slice.
    /// Mirrors `tokenize()`'s output shape (1, seq_len), i64 dtype. Returns
    /// an error if `ids` is empty.
    pub fn tensor_from_ids(&self, ids: &[u32]) -> Result<Tensor, SwarmError> {
        if ids.is_empty() {
            return Err(SwarmError::Internal(
                "tensor_from_ids: empty token slice".into(),
            ));
        }
        let as_i64: Vec<i64> = ids.iter().map(|&t| t as i64).collect();
        Tensor::new(as_i64.as_slice(), &self.device)
            .map_err(|e| SwarmError::Internal(format!("Token tensor: {e}")))?
            .unsqueeze(0)
            .map_err(|e| SwarmError::Internal(format!("Unsqueeze: {e}")))
    }

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
        // DIAG: dump token IDs for debugging tokenizer issues. Debug-level
        // (not info) — fires on every prompt and would flood default-level
        // logs while leaking prompt content. Other DIAG instrumentation in
        // this module already uses debug! / trace!.
        tracing::debug!(
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
