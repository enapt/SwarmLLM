//! Deciding how many of a segment's layers go on the graphics card.
//!
//! **Why this exists.** Placement was all-or-nothing: `force_cpu_for` is
//! `gpu_layers == 0`, so a model needing slightly more graphics memory than is
//! free lost the card *entirely*. Measured on this swarm that is a ~24x cliff,
//! the largest multiplier in the system, and it has now been reported by three
//! different machines — most recently a 14B needing 9347 MB against a 4990 MB
//! budget, which ran on the processor with **5151 MB of the card free and
//! unused**, pinning several cores and pushing the machine past its thermal
//! threshold (gotcha #431, `docs/FUTURE_WORK.md`).
//!
//! **Prior art, and where we deliberately differ** (diagnosis rule 0).
//! llama.cpp's `--n-gpu-layers` is a count, not a switch, and its auto mode
//! (PR #17485) reserves a flat **800 MB** for "KV cache and compute buffers"
//! then divides the remaining VRAM by an estimated per-layer size. Reviewers on
//! that PR pushed toward #16653's approach instead — real trial allocations
//! rather than an a-priori formula — because a formula that disagrees with
//! reality is exactly the trap.
//!
//! We keep the formula but **do not need the flat reserve**, because the thing
//! it stands in for is a number we already compute. KV cache is charged per
//! layer (`kv_budget::KvPrice`), so KV belongs *inside* the
//! per-layer cost rather than in a lump beside it — a layer placed on the card
//! brings its own KV with it, and one placed on the processor does not. That
//! turns a guess into arithmetic, and it is why this takes a context length
//! rather than a magic constant.
//!
//! What survives as a reserve is only what genuinely is not per-layer: the
//! transient activation and attention-score buffers a forward pass allocates,
//! which scale with the prefill chunk and the head count, not with depth.
//!
//! **The split is contiguous and card-first**, matching llama.cpp: layers
//! `[0, n)` of the segment on the card, `[n, end)` on the processor. Contiguous
//! is what keeps the cost at **one** device transition per forward — the same
//! reasoning as #425's shard contiguity, one level down. Which end holds the
//! card does not affect compute (dense layers cost the same), so matching the
//! convention everyone else uses is worth more than a private optimum.

use candle_core::{Device, Tensor};

/// Held back for the transient buffers a forward pass allocates.
///
/// Activations and attention scores, which scale with the prefill chunk and the
/// head count rather than with depth — so unlike KV they cannot be folded into
/// a per-layer cost. At a 128-token chunk and a 4k context a score matrix runs
/// to tens of MB and several are live at once.
///
/// Deliberately smaller than llama.cpp's 800 MB, because that figure covers KV
/// as well and ours does not. Erring high here costs a layer or two on the
/// card; erring low costs an out-of-memory failure mid-request, so it is not a
/// symmetric trade and this is set generously.
pub(crate) const FORWARD_BUFFER_RESERVE_BYTES: u64 = 384 * 1024 * 1024;

/// The card a card/processor split is sized against: the pool's ROOM for it
/// (its card budget less every charge) and, where the worker that will load
/// the split has read the card first (`DaemonMsg::ReadCardFree`, #365), the
/// free memory that load reads — the figure the loader's line is drawn on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SplitCard {
    /// The pool's room for the split, in bytes.
    pub(crate) room_bytes: u64,
    /// The card's free memory as the worker's load will read it, in bytes;
    /// `None` where no worker was asked (a plan, a growth).
    pub(crate) load_reads_free_bytes: Option<u64>,
}

impl SplitCard {
    /// A split with no worker to ask: the room stands in for the load's
    /// reading ([`split_room_bytes`] says why that is safe and what it costs).
    pub(crate) fn room(room_bytes: u64) -> Self {
        Self {
            room_bytes,
            load_reads_free_bytes: None,
        }
    }

    /// A spawn whose worker read the card before loading anything.
    pub(crate) fn read_by_the_load(room_bytes: u64, free_bytes: u64) -> Self {
        Self {
            room_bytes,
            load_reads_free_bytes: Some(free_bytes),
        }
    }
}

/// What a card/processor split may fill with weights and conversation caches
/// of `card`: the smaller of the room less [`FORWARD_BUFFER_RESERVE_BYTES`]
/// and the loader's own share (`kv_budget::weights_and_caches_room`) of the
/// free memory the load reads. Read by both of the pool's placements
/// (`process_pool::split_for_card` → [`plan_gpu_layers`], and the
/// experts-in-RAM rung), so neither offers the card more than its load takes.
///
/// **The loader's share is the binding term on any card with more than
/// ~2.5 GB of room**, and the one that refuses: past it the load gives the
/// conversation cache nothing and moves the whole model to the processor. A
/// 70B split on a 12 GB card was given 17 layers — 9,912 MB of weights,
/// inside its 10,890 MB room less the reserve and the 544 MB cache — where the
/// load let weights and caches take 9,676 MB (field report, v0.3.258). It
/// showed once v0.3.257 gave a split the card memory idle models held (#355):
/// a split beside them was smaller, and so was the shortfall.
///
/// **Where the load's reading is not known the share is taken of the room**,
/// which stands in for it and is never more than it: the pool's room is the
/// card's free memory less a reserve of at least 512 MB
/// (`ResourceConfig::inference_vram_budget_mb`), which covers the new worker's
/// own CUDA context. Safe, and short: on this node's 8 GB card the load read
/// 670 MB more than the room (the desktop's 751 MB counted against the budget,
/// the worker's context ~150 MB), so a 14B split put 23-24 of 48 layers on the
/// card where the load took 27 — decode 0.92x, a 30B-A3B's experts-in-RAM
/// 0.89x (#365, 2026-10-28). The reading cannot be predicted to that
/// precision — a 12 GB card read only 494 MB more than its room — so a spawn
/// ASKS its worker before the load (`process_pool` `refit_split_to_the_load`),
/// as Ollama's server asks its runner to fit a layout before it allocates
/// (`llm.LoadOperationFit`, v0.11.5) and llama.cpp's own fitting reads the
/// device it is about to fill (`common/fit.cpp`, `--fit-target`).
pub(crate) fn split_room_bytes(card: SplitCard) -> u64 {
    let load_reads = card.load_reads_free_bytes.unwrap_or(card.room_bytes);
    card.room_bytes
        .saturating_sub(FORWARD_BUFFER_RESERVE_BYTES)
        .min(super::kv_budget::weights_and_caches_room(load_reads))
}

/// How many of a segment's FIRST layers fit in `room_bytes` of card:
/// `0..=layer_weights.len()`. `room_bytes` is what the layers' own weights and
/// caches may take — [`split_room_bytes`] of the card, less the tables the
/// segment's ends put there first (`process_pool::split_for_card`).
///
/// Zero means the processor takes the whole segment, which is the old
/// behaviour and still correct when nothing fits.
///
/// Each layer is charged its OWN weights (`layer_weights[i]`, the header's
/// tensors of that layer) and its share of the segment's caches at the context
/// charged (`kv_budget::KvPrice` — each layer by its own shape, a
/// sliding-window layer by its window; FUTURE_WORK #281). The cache figure has
/// to be the one admission uses, or a model is placed by one number and then
/// runs out of memory against another — the disagreement that made
/// `estimate_gpu_footprint_mb` and `estimate_vram_from_shard_dir` reach
/// different conclusions about the same model (gotcha #388).
///
/// **The weights were averaged over the segment until 2026-10-25** (#343),
/// while the card takes the first layers. Q4_K_M gives the first and last
/// eighth of a model, and every third layer between, more bits (llama.cpp's
/// `use_more_bits`), so a 70B's first ten layers weigh 519 MiB each against
/// 502 averaged over its first 17: ~170 MiB the card was not charged.
/// llama.cpp's own fitting (`common/fit.cpp`, `llama_params_fit_impl`) weighs
/// each candidate count by the tensors it would place, never by an average.
pub(crate) fn plan_gpu_layers(
    room_bytes: u64,
    layer_weights: &[u64],
    segment_kv_bytes: u64,
) -> usize {
    // An unpriced segment is placed nowhere: no weights is "not known", never
    // "free".
    if layer_weights.iter().all(|&bytes| bytes == 0) {
        return 0;
    }
    // A layer on the card brings its KV with it; one on the processor does not.
    // Folding KV in here is what removes the need for a flat reserve.
    let kv_per_layer = segment_kv_bytes.div_ceil(layer_weights.len() as u64);
    let mut placed = 0u64;
    layer_weights
        .iter()
        .take_while(|&&weights| {
            // Checked, not saturated: a sum past `u64::MAX` saturated to it
            // and then "fitted" a room of `u64::MAX` (gotcha #855's class).
            match placed
                .checked_add(weights)
                .and_then(|sum| sum.checked_add(kv_per_layer))
            {
                Some(sum) if sum <= room_bytes => {
                    placed = sum;
                    true
                }
                _ => false,
            }
        })
        .count()
}

/// How many of a segment's `segment_layers` go on the loader's `device` (the card): the pool's
/// `gpu_layers` when a hybrid placement applies, otherwise all of them.
///
/// **The one reading of the placement**, for the two places that must agree:
/// [`LayerPlacement`], which puts the layers there, and the loader's KV budget,
/// which subtracts what the card holds from what it has free. They disagreed
/// until 2026-09-25: the budget charged the WHOLE segment's weights and its
/// whole KV to the card while only `gpu_layers` of them went there, so a model
/// placed 36-of-40 on an 8 GB card was told it had 15 MB for conversations and
/// refused a 20-token prompt, again on the re-plan (#104, GLM-4-9B, card empty).
///
/// Mirrors the placement's own rule exactly: a count below the segment's size,
/// on a card, for an architecture whose layers may be split. Anything else —
/// no count, a count covering the segment, the processor, an unverified
/// architecture — loads every layer on `device`.
pub(crate) fn layers_on_device(
    gpu_layers: Option<usize>,
    arch_splits: bool,
    device_is_cuda: bool,
    segment_layers: usize,
) -> usize {
    match gpu_layers {
        Some(n) if arch_splits && device_is_cuda && n < segment_layers => n,
        _ => segment_layers,
    }
}

/// May this architecture's layers be split across two devices?
///
/// **Deny by default, and a new architecture is denied until someone checks
/// it.** The loader applies placement by shadowing `device`, `cos` and `sin`
/// at the head of each per-layer loop, which is safe exactly when the loop's
/// tensor loads go through those names. Qwen 3.5's first loop did not: it
/// built its own `q35_cos`/`q35_sin` outside the loop, so a split model would
/// have put a card-resident layer's RoPE tables on the processor.
///
/// That failure is invisible until runtime — candle only objects when two
/// tensors from different devices meet inside an op — which is precisely the
/// hazard `docs/FUTURE_WORK.md` flagged as where the real risk lives. A
/// compiler warning about an unused shadow is what surfaced it here, and
/// nothing would have surfaced it on a machine without that architecture.
///
/// So the list is an allowlist of architectures whose loops have been read,
/// not a denylist of known-bad ones. Adding one means checking that every
/// tensor its loop loads comes from the shadowed names — and ideally running
/// it split on a real card.
pub(crate) fn arch_supports_hybrid(arch: &crate::inference::model_arch::ModelArch) -> bool {
    use crate::inference::model_arch::ModelArch as A;
    match arch {
        // Dense-family: the per-layer loop loads everything through the
        // shadowed `device`, and takes its RoPE from `rope_for` — which is
        // also where Gemma 3's and Gemma 4's windowed layers get their own
        // tables.
        A::Llama
        | A::Qwen2
        | A::Gemma
        | A::Gemma2
        | A::Gemma3
        | A::Gemma4
        | A::Phi3
        | A::Mistral
        | A::Starcoder2
        | A::Glm4
        | A::Llama4 => true,
        // Qwen 3.5 (dense), read 2026-10-07 (FUTURE_WORK #228): the rewritten
        // loop loads every tensor through the shadowed `device` and takes its
        // attention layers' RoPE from the shadowed `cos`/`sin` (the separate
        // `q35_cos`/`q35_sin` this entry once excluded it for are gone), and a
        // DeltaNet layer creates its recurrent and convolution state on its
        // INPUT's device (`layers::qwen35`), which the executor moves to the
        // layer's device before any layer kind runs. Checked split on a real
        // card against llama.cpp — the numbers are in FUTURE_WORK #228.
        // The MoE variant runs the same loop; its feed-forward loads through
        // `load_moe_ffn` with that loop's shadowed `device`, and was checked
        // split on the card against llama.cpp on 2026-10-11 (#228 (1)).
        A::Qwen35 | A::Qwen35Moe => true,
        // MLA attention with its own projection set; unverified.
        A::DeepSeek2 => false,
        // An architecture we do not recognise is one whose loop nobody has
        // read. Deny, for the same reason the list is an allowlist.
        A::Unknown(_) => false,
    }
}

/// Which device each layer of a segment loads onto, and the RoPE tables that
/// go with it.
///
/// **Applied by shadowing, not by threading.** The loader has ~128 `&device`
/// references across four per-architecture loops, and editing each one is how
/// a path gets missed — which for device placement fails at *runtime*, not at
/// compile time, because candle only complains when two tensors from different
/// devices meet in an op. Instead each loop shadows `device`, `cos` and `sin`
/// with this type's answers on its first line, so every load inside the body
/// picks up the right device with no further edit and a new architecture
/// inherits the behaviour by construction.
pub(crate) struct LayerPlacement {
    /// Absolute index of this segment's first layer.
    layer_start: usize,
    /// How many of the segment's layers, from the start, go on the card.
    gpu_layers: usize,
    /// The card, when any layer is going there.
    gpu: Option<Device>,
    /// Where everything else goes.
    cpu: Device,
    /// RoPE tables per device. Built by moving, not recomputing: `to_device`
    /// on a plain tensor is a copy, and these are small next to the weights.
    rope_gpu: Option<(Tensor, Tensor)>,
    rope_cpu: (Tensor, Tensor),
    /// The tables of the sliding-window layers, where those rotate
    /// differently from the rest (Gemma 3, [`super::rope::LayerRope`]);
    /// `None` for every other model, whose layers all take the ones above.
    windowed_rope: Option<WindowedTables>,
    /// How many of the segment's card-held layers, from the start, keep their
    /// routed experts in the machine's memory while everything else of theirs
    /// — attention, its cache, the router, a shared expert — stays on the card
    /// (#138 (3), llama.cpp's `--n-cpu-moe`). 0 for none, and always 0 without
    /// a card. A layer already on the processor holds its experts there anyway.
    experts_on_the_processor: usize,
}

/// [`LayerPlacement::with_windowed_rope`]'s tables and the layers they are for.
struct WindowedTables {
    window: super::SlidingWindow,
    gpu: Option<(Tensor, Tensor)>,
    cpu: (Tensor, Tensor),
}

impl LayerPlacement {
    /// Everything on one device — the behaviour before hybrid placement, and
    /// still what a model that fits (or one with no card) gets.
    pub(crate) fn uniform(
        layer_start: usize,
        device: Device,
        cos: Tensor,
        sin: Tensor,
    ) -> candle_core::Result<Self> {
        let on_gpu = device.is_cuda();
        Ok(Self {
            layer_start,
            gpu_layers: if on_gpu { usize::MAX } else { 0 },
            gpu: if on_gpu { Some(device.clone()) } else { None },
            cpu: if on_gpu { Device::Cpu } else { device },
            rope_gpu: if on_gpu {
                Some((cos.clone(), sin.clone()))
            } else {
                None
            },
            rope_cpu: (cos, sin),
            windowed_rope: None,
            experts_on_the_processor: 0,
        })
    }

    /// Split: the first `gpu_layers` on the card, the rest on the processor.
    pub(crate) fn split(
        layer_start: usize,
        gpu: Device,
        gpu_layers: usize,
        cos: Tensor,
        sin: Tensor,
    ) -> candle_core::Result<Self> {
        let rope_cpu = (cos.to_device(&Device::Cpu)?, sin.to_device(&Device::Cpu)?);
        Ok(Self {
            layer_start,
            gpu_layers,
            gpu: Some(gpu),
            cpu: Device::Cpu,
            rope_gpu: Some((cos, sin)),
            rope_cpu,
            windowed_rope: None,
            experts_on_the_processor: 0,
        })
    }

    /// The RoPE tables of the layers `window` names, on each device a layer
    /// can go to. [`Self::rope_for`] hands them to those layers — the one
    /// place a layer's tables come from, so no per-architecture loop can
    /// rotate a windowed layer with the global ones.
    pub(crate) fn with_windowed_rope(
        mut self,
        window: super::SlidingWindow,
        cos: Tensor,
        sin: Tensor,
    ) -> candle_core::Result<Self> {
        let gpu = match &self.gpu {
            Some(gpu) => Some((cos.to_device(gpu)?, sin.to_device(gpu)?)),
            None => None,
        };
        let cpu = (cos.to_device(&self.cpu)?, sin.to_device(&self.cpu)?);
        self.windowed_rope = Some(WindowedTables { window, gpu, cpu });
        Ok(self)
    }

    /// The routed experts of the first `n` card-held layers kept in the
    /// machine's memory ([`Self::experts_device_for`]). The first, as
    /// llama.cpp's `--n-cpu-moe` picks them ("the first N layers",
    /// `llm_add_n_cpu_ffn_overrides` over `LLM_FFN_EXPS_REGEX`, `common/arg.cpp`
    /// at `4b1a27f` — routed experts only, the shared expert stays): which
    /// layers does not change the work, since every MoE layer of a model costs
    /// the same. A no-op without a card.
    pub(crate) fn with_experts_on_the_processor(mut self, n: usize) -> Self {
        if self.gpu.is_some() {
            self.experts_on_the_processor = n;
        }
        self
    }

    fn on_gpu(&self, abs_layer: usize) -> bool {
        // `checked_sub`, not `saturating_sub`: an index below this segment's
        // start is not in the segment at all, and saturating it to 0 reads as
        // "the first layer", i.e. on the card. Caught by
        // `the_split_puts_the_first_layers_on_the_card_and_the_rest_on_the_processor`,
        // which is the difference between a guard and a comment.
        let Some(offset) = abs_layer.checked_sub(self.layer_start) else {
            return false;
        };
        self.gpu.is_some() && offset < self.gpu_layers
    }

    /// The device layer `abs_layer` loads onto.
    pub(crate) fn device_for(&self, abs_layer: usize) -> Device {
        match (&self.gpu, self.on_gpu(abs_layer)) {
            (Some(gpu), true) => gpu.clone(),
            _ => self.cpu.clone(),
        }
    }

    /// The device layer `abs_layer`'s routed experts load onto: the
    /// processor for one of the first card-held layers
    /// [`Self::with_experts_on_the_processor`] names, the layer's own device
    /// otherwise. `split::loader::load_moe_ffn` asks it — the one loader of
    /// experts — so no per-layer loop can place them differently.
    pub(crate) fn experts_device_for(&self, abs_layer: usize) -> Device {
        if self.experts_in_ram(abs_layer) {
            self.cpu.clone()
        } else {
            self.device_for(abs_layer)
        }
    }

    /// Does card-held layer `abs_layer` keep its experts in the machine's memory?
    fn experts_in_ram(&self, abs_layer: usize) -> bool {
        self.on_gpu(abs_layer)
            && abs_layer
                .checked_sub(self.layer_start)
                .is_some_and(|offset| offset < self.experts_on_the_processor)
    }

    /// The RoPE tables for that layer, already on its device.
    pub(crate) fn rope_for(&self, abs_layer: usize) -> (Tensor, Tensor) {
        let (gpu, cpu) = match &self.windowed_rope {
            Some(w) if w.window.for_layer(abs_layer).is_some() => (&w.gpu, &w.cpu),
            _ => (&self.rope_gpu, &self.rope_cpu),
        };
        match (gpu, self.on_gpu(abs_layer)) {
            (Some((c, s)), true) => (c.clone(), s.clone()),
            _ => (cpu.0.clone(), cpu.1.clone()),
        }
    }

    /// Per-layer devices, in segment order, for the model to keep.
    pub(crate) fn devices(&self, layer_count: usize) -> Vec<Device> {
        (0..layer_count)
            .map(|i| self.device_for(self.layer_start + i))
            .collect()
    }

    /// Is this segment actually split across two devices?
    pub(crate) fn is_hybrid(&self, layer_count: usize) -> bool {
        self.gpu.is_some() && self.gpu_layers < layer_count
    }

    /// The card, when some of this segment's weights sit on the processor —
    /// whole layers, or a card-held layer's routed experts.
    pub(crate) fn card_beside_the_processor(&self, layer_count: usize) -> Option<&Device> {
        self.gpu
            .as_ref()
            .filter(|_| self.is_hybrid(layer_count) || self.experts_on_the_processor > 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::split::kv_budget::kv_bytes_per_token;
    use crate::inference::split::kv_cache::KvStorage;

    const MB: u64 = 1024 * 1024;
    const GB: u64 = 1024 * MB;

    /// `weights` spread evenly over `layers` layers, as every layer of a
    /// segment was charged before #343.
    fn even(weights: u64, layers: usize) -> Vec<u64> {
        vec![weights / layers as u64; layers]
    }

    /// `layers` layers' caches of `elems` K and V elements a token each, kept
    /// as `storage`, at `context` positions.
    fn segment_kv(layers: usize, elems: usize, storage: KvStorage, context: u64) -> u64 {
        kv_bytes_per_token(1, elems, elems, storage) * context * layers as u64
    }

    /// The reported case: a 14B segment wanting 9347 MB against a 4990 MB
    /// budget, on a card with 5151 MB free. It used to get **nothing**.
    #[test]
    fn the_reported_case_now_puts_something_on_the_card() {
        let total = 48;
        let weights = 9347 * MB;
        let n = plan_gpu_layers(
            split_room_bytes(SplitCard::room(4990 * MB)),
            &even(weights, total),
            segment_kv(total, 1024, KvStorage::F32, 4096),
        );
        assert!(n > 0, "still placing nothing on a card with room");
        assert!(n < total, "claimed to fit a segment that does not fit");
        // And what it placed must actually fit inside the budget.
        let per_layer = weights / total as u64;
        let kv = kv_bytes_per_token(1, 1024, 1024, KvStorage::F32) * 4096;
        assert!(
            n as u64 * (per_layer + kv) + FORWARD_BUFFER_RESERVE_BYTES <= 4990 * MB,
            "the plan overcommits the card"
        );
    }

    /// What the card holds is the pool's count when a hybrid split applies, and
    /// the whole segment otherwise — one answer for the placement AND the
    /// loader's KV budget (#104: the budget charged all 40 layers of a 36-of-40
    /// placement to the card and left 15 MB for conversations).
    #[test]
    fn the_card_holds_the_split_count_and_only_when_a_split_applies() {
        assert_eq!(layers_on_device(Some(36), true, true, 40), 36, "hybrid");
        assert_eq!(layers_on_device(None, true, true, 40), 40, "fits whole");
        assert_eq!(
            layers_on_device(Some(40), true, true, 40),
            40,
            "a count covering the segment is not a split"
        );
        assert_eq!(
            layers_on_device(Some(36), false, true, 40),
            40,
            "an architecture that may not be split loads whole on the card"
        );
        assert_eq!(
            layers_on_device(Some(36), true, false, 40),
            40,
            "on the processor there is nothing to split off"
        );
    }

    #[test]
    fn a_segment_that_fits_entirely_goes_entirely_to_the_card() {
        // 2 GB of weights, 8 GB budget: everything, and never more than exists.
        let n = plan_gpu_layers(
            split_room_bytes(SplitCard::room(8 * GB)),
            &even(2 * GB, 28),
            segment_kv(28, 512, KvStorage::F32, 4096),
        );
        assert_eq!(n, 28);
    }

    #[test]
    fn a_card_with_no_room_still_gets_nothing() {
        // The whole budget is swallowed by the forward-buffer reserve.
        assert_eq!(
            plan_gpu_layers(
                split_room_bytes(SplitCard::room(256 * MB)),
                &even(8 * GB, 32),
                segment_kv(32, 1024, KvStorage::F32, 4096)
            ),
            0
        );
        // And a zero budget cannot go negative or wrap.
        assert_eq!(
            plan_gpu_layers(
                0,
                &even(8 * GB, 32),
                segment_kv(32, 1024, KvStorage::F32, 4096)
            ),
            0
        );
    }

    /// **A split fills the share its load reads** (#365): on this node's 8 GB
    /// card a 14B's spawn had 6,621 MB of room and its load read 7,291 MB
    /// free (2026-10-28). Where no worker was asked the room stands in for the
    /// reading; asked, the share is the load's line on what it read — never
    /// past the room less the forward reserve, and smaller where the card was
    /// taken after the room was read.
    #[test]
    fn a_split_fills_the_share_its_load_reads() {
        let room = 6_621 * MB;
        let unasked = split_room_bytes(SplitCard::room(room));
        let read = split_room_bytes(SplitCard::read_by_the_load(room, 7_291 * MB));
        assert_eq!(unasked, room / 100 * 85);
        assert_eq!(read, 7_291 * MB / 100 * 85);
        assert!(
            read > unasked + 500 * MB,
            "the load's reading gave the split {} MB more",
            (read - unasked) / MB
        );
        assert_eq!(
            split_room_bytes(SplitCard::read_by_the_load(room, 20 * GB)),
            room - FORWARD_BUFFER_RESERVE_BYTES,
            "a reading past the room leaves the pool's own line standing"
        );
        assert!(
            split_room_bytes(SplitCard::read_by_the_load(room, 5_000 * MB)) < unasked,
            "a card taken since the room was read shrinks the share"
        );
    }

    /// KV is charged per layer placed, not as a lump — so asking for a longer
    /// conversation must move layers off the card, never leave the plan
    /// unchanged. This is the property that replaces llama.cpp's flat reserve.
    #[test]
    fn a_longer_context_costs_layers_on_the_card() {
        let args = |ctx| {
            plan_gpu_layers(
                split_room_bytes(SplitCard::room(6 * GB)),
                &even(4 * GB, 32),
                segment_kv(32, 1024, KvStorage::F32, ctx),
            )
        };
        let short = args(2048);
        let long = args(32768);
        assert!(
            long < short,
            "context length did not affect placement ({short} at 2k, {long} at 32k) — \
             KV is not being charged per layer"
        );
    }

    #[test]
    fn the_mirror_is_charged_for() {
        // The f16 flash mirror is real graphics memory (see `LayerKv`), so a
        // model maintaining one must not be planned as though it were free.
        let plain = plan_gpu_layers(
            split_room_bytes(SplitCard::room(6 * GB)),
            &even(4 * GB, 32),
            segment_kv(32, 1024, KvStorage::F32, 8192),
        );
        let mirrored = plan_gpu_layers(
            split_room_bytes(SplitCard::room(6 * GB)),
            &even(4 * GB, 32),
            segment_kv(32, 1024, KvStorage::F32Mirrored, 8192),
        );
        assert!(mirrored <= plain, "mirroring must not increase the plan");
        // And the half cache, at a third of the mirrored layout's bytes, leaves
        // room for more of the model on the card (FUTURE_WORK #194).
        let half = plan_gpu_layers(
            split_room_bytes(SplitCard::room(6 * GB)),
            &even(4 * GB, 32),
            segment_kv(32, 1024, KvStorage::F16, 8192),
        );
        assert!(
            half > mirrored,
            "the half cache must place more layers ({half} vs {mirrored})"
        );
    }

    /// The boundary is the whole contract: layers before it on the card,
    /// layers after it on the processor, and the RoPE tables following their
    /// layer so no op ever meets tensors from two devices.
    #[test]
    fn the_split_puts_the_first_layers_on_the_card_and_the_rest_on_the_processor() {
        let cos = Tensor::zeros((4, 2), candle_core::DType::F32, &Device::Cpu).unwrap();
        let sin = cos.clone();
        // `Device::Cpu` stands in for the card here: the placement logic is
        // about indices, and a machine running these tests has no GPU.
        let p = LayerPlacement::split(10, Device::Cpu, 3, cos, sin).unwrap();
        // Absolute indices 10,11,12 are the segment's first three.
        assert!(p.on_gpu(10) && p.on_gpu(11) && p.on_gpu(12));
        assert!(!p.on_gpu(13), "boundary is off by one");
        // And a layer before this segment's start must never read as on-card.
        assert!(!p.on_gpu(9), "index below layer_start wrapped");
        assert_eq!(p.devices(5).len(), 5);
    }

    /// #138 (3): the experts of the first n CARD-held layers go to the
    /// processor and nothing else of theirs does; a layer already on the
    /// processor, a layer past n, and a placement with no card are untouched.
    #[test]
    fn the_first_card_layers_keep_their_experts_in_the_machines_memory() {
        let cos = Tensor::zeros((4, 2), candle_core::DType::F32, &Device::Cpu).unwrap();
        // `Device::Cpu` stands in for the card, as above: `split` records it
        // as the card, and the rule is about indices.
        let p = LayerPlacement::split(10, Device::Cpu, 3, cos.clone(), cos.clone())
            .unwrap()
            .with_experts_on_the_processor(2);
        // Layers 10, 11: on the card, experts in RAM. 12: on the card, experts
        // there too. 13: on the processor, experts with it.
        for (layer, on_card, experts_in_ram) in [
            (10, true, true),
            (11, true, true),
            (12, true, false),
            (13, false, false),
        ] {
            assert_eq!(p.on_gpu(layer), on_card, "layer {layer}");
            assert_eq!(
                p.experts_in_ram(layer),
                experts_in_ram,
                "layer {layer}'s experts"
            );
        }
        assert!(!p.experts_in_ram(9), "a layer below the segment wrapped");
        assert!(p.card_beside_the_processor(5).is_some());

        let whole = LayerPlacement::split(0, Device::Cpu, usize::MAX, cos.clone(), cos.clone())
            .unwrap()
            .with_experts_on_the_processor(1);
        assert!(
            !whole.is_hybrid(4),
            "experts in RAM do not make a layer split"
        );
        assert!(
            whole.card_beside_the_processor(4).is_some(),
            "the processor-held experts are given the card for a prompt"
        );
        assert!(
            LayerPlacement::split(0, Device::Cpu, usize::MAX, cos.clone(), cos.clone())
                .unwrap()
                .card_beside_the_processor(4)
                .is_none(),
            "a card holding everything has nothing beside it"
        );

        // No card: nothing to keep off it.
        let none = LayerPlacement::uniform(0, Device::Cpu, cos.clone(), cos)
            .unwrap()
            .with_experts_on_the_processor(2);
        assert!(!none.experts_in_ram(0));
        assert!(none.card_beside_the_processor(4).is_none());
    }

    /// Gemma 3: the windowed layers take their own tables, on whichever
    /// device each layer went to, and every sixth layer the global ones.
    #[test]
    fn windowed_layers_take_their_own_rope_tables() {
        use crate::inference::split::gguf_meta::sliding_window_tests::header;
        let global = Tensor::zeros((4, 2), candle_core::DType::F32, &Device::Cpu).unwrap();
        let windowed = Tensor::ones((4, 2), candle_core::DType::F32, &Device::Cpu).unwrap();
        let window = super::super::SlidingWindow::from_file(&header(
            "gemma3",
            vec![(
                "gemma3.attention.sliding_window",
                candle_core::quantized::gguf_file::Value::U32(512),
            )],
        ))
        .unwrap();
        // `Device::Cpu` stands in for the card, as above: layers 0-2 "on the card".
        let p = LayerPlacement::split(0, Device::Cpu, 3, global.clone(), global)
            .unwrap()
            .with_windowed_rope(window, windowed.clone(), windowed)
            .unwrap();
        let first = |layer: usize| {
            p.rope_for(layer)
                .0
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()[0]
        };
        for layer in 0..12 {
            let want = if layer % 6 == 5 { 0.0 } else { 1.0 };
            assert_eq!(first(layer), want, "layer {layer}");
        }
    }

    #[test]
    fn a_uniform_placement_is_never_hybrid() {
        let cos = Tensor::zeros((4, 2), candle_core::DType::F32, &Device::Cpu).unwrap();
        let p = LayerPlacement::uniform(0, Device::Cpu, cos.clone(), cos).unwrap();
        assert!(
            !p.is_hybrid(32),
            "a single-device model must not take the transition path"
        );
        assert!(p.devices(32).iter().all(|d| !d.is_cuda()));
    }

    #[test]
    fn degenerate_inputs_do_not_panic_or_wrap() {
        assert_eq!(plan_gpu_layers(u64::MAX, &[0; 8], 0), 0, "unpriced");
        assert_eq!(plan_gpu_layers(0, &[], 1), 0, "no layers");
        // Never more layers than the segment has, however large the budget.
        assert_eq!(plan_gpu_layers(u64::MAX, &[1; 8], 0), 8);
        assert_eq!(plan_gpu_layers(u64::MAX, &[u64::MAX; 3], u64::MAX), 0);
    }

    /// **Each layer is charged its own weights** (#343): the card takes the
    /// FIRST layers, and a quantization mix gives some more bits than others.
    /// Averaged, a heavy opening placed one layer too many and a light one a
    /// layer too few.
    #[test]
    fn the_card_is_charged_its_own_layers_not_their_average() {
        let budget = 450 * MB;
        // 300 + 100 fits 450, a third layer does not; averaged at 150, three did.
        let heavy_first = [300 * MB, 100 * MB, 100 * MB, 100 * MB];
        assert_eq!(plan_gpu_layers(budget, &heavy_first, 0), 2);
        assert_eq!(plan_gpu_layers(budget, &even(600 * MB, 4), 0), 3);
        // Light first: three fit in 350; averaged at 150, only two did.
        let budget = 350 * MB;
        let heavy_last = [100 * MB, 100 * MB, 100 * MB, 300 * MB];
        assert_eq!(plan_gpu_layers(budget, &heavy_last, 0), 3);
        assert_eq!(plan_gpu_layers(budget, &even(600 * MB, 4), 0), 2);
        // A layer's cache comes with it: 40 MB of cache a layer moves the
        // heavy opening's line to one.
        assert_eq!(
            plan_gpu_layers(budget + 100 * MB, &heavy_first, 4 * 40 * MB),
            1
        );
    }
}
