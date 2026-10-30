//! The drafter of speculation across computers (`pipeline::dsd`) as an
//! ordinary small model, run in THIS engine from its own shards
//! (`docs/plans/split_speculation.md` Phase 1 item 3).
//!
//! DSD drafted only with llama.cpp from a whole GGUF named in
//! `inference.draft_model_path`: a build without the `llama` feature could not
//! speculate across computers at all, and a node must never assemble a model
//! file from shards (CLAUDE.md), so the drafter had to be a file the operator
//! put there by hand. Here the drafter is a model the node already holds — a
//! same-family sibling, e.g. Qwen2.5-0.5B for any Qwen2.5 — served by its own
//! worker like any other model, and asked for guesses over the worker IPC
//! (`DaemonMsg::Draft`).
//!
//! **The worker keeps the drafting cache between rounds**, keyed by the target
//! request's id, and this side tracks how much of it is still true
//! ([`EngineDrafter::valid`]): after a call the cache holds the sequence so far
//! plus all but the last guess, and the round's check says how many of those
//! guesses were right. The next call cuts the cache back to that and reads only
//! what is new — the token the check sampled, and the last guess when every one
//! was kept. So a round costs the drafter one short prefill and γ steps, never
//! the conversation again. A round whose guesses came from elsewhere (an n-gram
//! match) needs no call at all: the next one reads everything since.

use std::sync::OnceLock;

use dashmap::DashMap;

use crate::daemon::state::SharedState;
use crate::error::SwarmError;
use crate::inference::process_pool::Requester;
use crate::inference::worker_ipc::IpcDraft;
use crate::types::{ModelId, SamplingParams};

/// The local model that drafts for a target, and the layer range its worker
/// holds it under (the whole model).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DrafterSpec {
    pub model_id: ModelId,
    pub layer_range: (u32, u32),
    /// Token ids below this are the drafter's own; see `compatible_vocabularies`.
    pub reads: u32,
}

/// `(target, drafter) → how many token ids the drafter reads`, or `None` when
/// their vocabularies disagree — decided once per pair: two models' token lists
/// do not change while they are held.
fn vocabulary_matches() -> &'static DashMap<(ModelId, ModelId), Option<u32>> {
    static M: OnceLock<DashMap<(ModelId, ModelId), Option<u32>>> = OnceLock::new();
    M.get_or_init(DashMap::new)
}

/// `tokenizer.ggml.token_type` for an entry the tokenizer never produces
/// (llama.cpp's `LLAMA_TOKEN_TYPE_UNUSED`).
const TOKEN_TYPE_UNUSED: i32 = 5;

/// Do these two models read and write the SAME token ids? The drafter's
/// guesses are ids, checked by the target as ids — a guess in another
/// vocabulary is not a wrong guess but a different word. Answers the number of
/// ids the drafter can read, or `None`.
///
/// Compared as token LISTS, not embedding sizes — and one list may run past the
/// other only in entries the tokenizer never produces. Qwen2.5-Coder-7B lists
/// 152,064 tokens where Qwen2.5-0.5B lists 151,936: the extra 128 are
/// `[PAD151936]`… of type unused, padding the list to the embedding's size,
/// and the 151,936 before them are identical (checked 2026-09-28 on the headers
/// this node holds). A reply containing one of them is possible only by
/// sampling a padding row, and ends the drafting with an error rather than
/// feeding the drafter an id it has no row for.
fn compatible_vocabularies(
    target: &crate::inference::split::GgufTokenizerMeta,
    drafter: &crate::inference::split::GgufTokenizerMeta,
) -> Option<u32> {
    let common = target.vocab.len().min(drafter.vocab.len());
    if common == 0 || target.vocab[..common] != drafter.vocab[..common] {
        return None;
    }
    let longer = if target.vocab.len() > common {
        target
    } else {
        drafter
    };
    let extra_unused =
        (common..longer.vocab.len()).all(|i| longer.token_types.get(i) == Some(&TOKEN_TYPE_UNUSED));
    extra_unused.then_some(drafter.vocab.len() as u32)
}

fn drafter_vocabulary(state: &SharedState, target: &ModelId, drafter: &ModelId) -> Option<u32> {
    let key = (target.clone(), drafter.clone());
    if let Some(known) = vocabulary_matches().get(&key) {
        return *known;
    }
    let meta = |m: &ModelId| {
        let header = state
            .model_dir(&m.0)
            .join(crate::model::shard::HEADER_FILENAME);
        crate::inference::split::GgufTokenizerMeta::from_gguf_file(&header).ok()
    };
    // Unknown is not remembered: a header may still arrive.
    let (t, d) = (meta(target)?, meta(drafter)?);
    let answer = compatible_vocabularies(&t, &d);
    if answer.is_none() {
        tracing::info!(
            target_model = %target,
            drafter = %drafter,
            "the configured drafter does not share the target's vocabulary — not drafting with it"
        );
    }
    vocabulary_matches().insert(key, answer);
    answer
}

/// What choosing a drafter weighs of a model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct DrafterSize {
    /// How much it knows: [`size_counting_experts`].
    size: u64,
    /// What one guess costs: the weight bytes a token reads (`u64::MAX` when the
    /// header cannot say), so of two copies of one model — an fp16 and a Q8_0
    /// Qwen2.5-0.5B — the cheaper guesses. Name order chose the fp16 here
    /// (2026-10-14), the copy that reads the most for the same guesses.
    reads: u64,
}

/// A model's size for choosing a drafter: layers × width², the part of its
/// parameter count that differs between siblings (Qwen2.5-0.5B 19 M against
/// Qwen2.5-Coder-7B's 360 M; Llama-3.2-1B 67 M against Llama-3.1-8B's 537 M) —
/// times, for a mixture-of-experts model, how many times more it holds than a
/// token reads ([`size_counting_experts`]). Bytes on disk alone would compare an
/// fp16 small model with a 4-bit large one.
fn size_proxy(state: &SharedState, model: &ModelId) -> Option<DrafterSize> {
    let meta = state.gguf_meta_for(model)?;
    let geometry = meta.block_count as u64 * (meta.embedding_length as u64).pow(2);
    let active = meta.active_bytes_per_layer();
    Some(DrafterSize {
        size: size_counting_experts(geometry, meta.stored_bytes_per_layer(), active),
        reads: active.map_or(u64::MAX, |a| (a * meta.block_count as f64) as u64),
    })
}

/// `geometry` (layers × width²) scaled by what the model holds over what a
/// token reads — 1 for a dense model, so a dense model's size is unchanged.
///
/// Layers × width² leaves the feed-forward out, which is harmless between dense
/// siblings and wrong for a mixture of experts: it sized Qwen3-30B-A3B (48 ×
/// 2048²) at 201 M, below Llama-3.2-3B, so no Qwen3 sibling was ever within a
/// quarter of it and every split of it guessed with n-grams alone (10-13: 9.8%
/// of rounds hit; 468 ms a token over a 222 ms link). Scaled by its experts it
/// is 1.7 G (Q2_K, 8.5× what a token reads) to 1.9 G (Q4_K_M, 9.6×), and
/// Qwen3-1.7B (117 M) drafts for it. What a check of the target
/// COSTS is not this figure, and on one machine drafting a model that reads only
/// ~3B a token can gain nothing (a 0.8B drafter on Qwen3.6-35B-A3B decoded 3-12%
/// slower on one RTX 3090, hackmd.io/@thc1006/SJly6IE6Wx) — but this drafter
/// only ever runs for a SPLIT, whose check pays the trip across machines, and
/// the γ controller measures that and can choose zero guesses
/// (`dsd_controller::best_gamma_for_check`).
fn size_counting_experts(geometry: u64, stored: Option<f64>, active: Option<f64>) -> u64 {
    match (stored, active) {
        (Some(s), Some(a)) if a > 0.0 && s > a => (geometry as f64 * (s / a)) as u64,
        _ => geometry,
    }
}

/// The largest a SMALL drafter may be, as a fraction of its target: half
/// (2026-10-24). Leviathan et al. 2023 used drafters ~1/10-1/20 of the target
/// because on one machine a guess costs a fair part of a check; this drafter
/// only ever guesses for a SPLIT, whose check pays the round trip between
/// machines, so a guess that is right more often is worth more than one that
/// is cheap. On the real TH↔AU link (eight arms alternating, one binary)
/// Qwen2.5-Coder-3B Q4_K_M — 151 M against Qwen2.5-Coder-7B's 360 M, past the
/// quarter this rule was before — flowed a chat reply at 18.1 tok/s against the
/// 1.5B Q8_0's 16.0, missing 3 times a reply against 11; on a code edit, where
/// copied runs leave neither anything to miss, it was 4% slower (63.0 against
/// 65.4). The γ controller and the tree's row pricing still size the guessing
/// from the measured cost of each guess.
const DRAFTER_MAX_FRACTION: u64 = 2;

/// A drafter reading more than this a guess is held to [`LARGE_DRAFTER_MAX_FRACTION`]
/// of its target: the half was measured for a drafter about this size (the 3B
/// above reads under 2 GB), and a drafter is fetched only up to it
/// ([`MOST_DRAFTER_BYTES_TO_FETCH`]). A larger one — only ever one the owner
/// holds, a 7B for a 14B — costs several times as much a guess, which a flow
/// already spends most of its time on (the 3B's calls filled 65% of the reply).
const SMALL_DRAFTER_READS: u64 = MOST_DRAFTER_BYTES_TO_FETCH;

/// The largest a drafter past [`SMALL_DRAFTER_READS`] may be: a quarter, the
/// rule for every drafter before 2026-10-24.
const LARGE_DRAFTER_MAX_FRACTION: u64 = 4;

/// [`DRAFTER_MAX_FRACTION`], or `SWARMLLM_DRAFTER_MAX_FRACTION=<n>` (≥ 1): the
/// A/B of the rule — a small drafter up to 1/n of its target (4 is the rule
/// before).
fn drafter_max_fraction() -> u64 {
    static FRACTION: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *FRACTION.get_or_init(|| {
        std::env::var("SWARMLLM_DRAFTER_MAX_FRACTION")
            .ok()
            .and_then(|v| v.trim().parse().ok())
            .filter(|n: &u64| *n >= 1)
            .unwrap_or(DRAFTER_MAX_FRACTION)
    })
}

/// The fraction of its target a small drafter may be on THIS node:
/// [`drafter_max_fraction`] where a graphics card is in play here
/// (`ModelProcessPool::card_in_play`) — the half was measured with the drafter
/// on the card — else [`LARGE_DRAFTER_MAX_FRACTION`]. On the processor (rig,
/// both halves and the drafter there, 5 ms apart, 2026-10-24) the coder-3B
/// guessed no better than the 1.5B (kept 0.73 against 0.74 of its guesses) at
/// 73 ms a guess against 65, and a chain on a long link spends its round
/// guessing there, so the quarter stays.
fn small_drafter_fraction(state: &SharedState) -> u64 {
    if state.model_process_pool.card_in_play() {
        drafter_max_fraction()
    } else {
        LARGE_DRAFTER_MAX_FRACTION
    }
}

/// May a drafter of `drafter`'s size guess for a target of `target_size`: up
/// to `1 / small_fraction` of it when it reads at most [`SMALL_DRAFTER_READS`]
/// a guess, else up to [`LARGE_DRAFTER_MAX_FRACTION`] (or the small fraction,
/// were that the tighter). A drafter whose header gives no reads (`u64::MAX`)
/// is a large one.
fn within_reach(drafter: &DrafterSize, target_size: u64, small_fraction: u64) -> bool {
    let fraction = if drafter.reads <= SMALL_DRAFTER_READS {
        small_fraction
    } else {
        LARGE_DRAFTER_MAX_FRACTION.max(small_fraction)
    };
    drafter.size.saturating_mul(fraction) <= target_size
}

/// `model` as a drafter for `target`, if it can be one: held WHOLE here, not
/// the target, no recurrent state (a guess that is refused must be taken back,
/// and a DeltaNet layer's state cannot be), and the target's vocabulary.
fn spec_for(state: &SharedState, target: &ModelId, drafter: ModelId) -> Option<DrafterSpec> {
    if !state.has_complete_split_model(&drafter) {
        return None;
    }
    could_draft(state, target, drafter)
}

/// [`spec_for`] without asking that the model be held: what its header says —
/// not the target, no recurrent state, the target's vocabulary. The download
/// pass asks it of a model it may fetch ([`drafter_to_acquire`]).
fn could_draft(state: &SharedState, target: &ModelId, drafter: ModelId) -> Option<DrafterSpec> {
    if drafter == *target {
        return None;
    }
    let (layers, recurrent) = {
        let meta = state.gguf_meta_for(&drafter)?;
        (
            meta.block_count as u32,
            crate::inference::model_arch::ModelArch::from_gguf_arch(&meta.architecture)
                .is_hybrid_ssm(),
        )
    };
    if recurrent {
        return None;
    }
    let reads = drafter_vocabulary(state, target, &drafter)?;
    Some(DrafterSpec {
        model_id: drafter,
        layer_range: (0, layers),
        reads,
    })
}

/// The model this node drafts with for `target`: `inference.draft_model` when
/// set, else the LARGEST model held whole here that can draft for it
/// ([`spec_for`]) and is [`within_reach`] of its size ([`size_proxy`]) — the
/// best guesser that is still cheap, since across a network the round trip,
/// not the guessing, is what a round mostly costs. Either way only one this
/// node's memory would load ([`admitted`]) for a reply of `conversation`
/// positions (`model_worker::drafter_cache_positions`, the charge its load
/// will take) beside the `here` the plan gives this node. `None` when there is
/// none: speculation across computers then drafts with llama.cpp where that is
/// configured, or not at all.
pub(crate) fn engine_drafter_for(
    state: &SharedState,
    target: &ModelId,
    conversation: u64,
    here: &TargetHere<'_>,
) -> Option<DrafterSpec> {
    chosen(state, target, |s| {
        admitted(state, target, s, conversation, here)
    })
}

/// The target's layers a plan gives this node — `ranges` — and whose work they
/// are (`pipeline::worker_requester`): what is loaded here before the drafter
/// is, and so what it must fit beside ([`admitted`]).
pub(crate) struct TargetHere<'a> {
    pub ranges: &'a [(u32, u32)],
    pub requester: Requester,
}

/// The drafter [`engine_drafter_for`] would choose for `target` were there room
/// for it — what the planner leaves room for beside this node's layers of
/// `target` (`scheduler`, #275), so the request then finds it admitted.
pub(crate) fn drafter_to_leave_room_for(
    state: &SharedState,
    target: &ModelId,
) -> Option<DrafterSpec> {
    chosen(state, target, |_| true)
}

/// How long a split wants a drafter after it last ran
/// ([`note_split_wanting_a_drafter`]): the download pass fetches it meanwhile and
/// prune keeps it. A week — a model split now and then still finds it held.
pub(crate) const DRAFTER_WANTED_FOR: std::time::Duration =
    std::time::Duration::from_secs(7 * 24 * 60 * 60);

/// A split of `target` driven from here is about to guess, with the drafter
/// `drafter_found` says it found or with nothing. Either way it is recorded for
/// the download pass (`ModelMgmt::drafters_wanted`), which acts on the answer
/// [`drafter_to_acquire`] gives: with nothing held — no model here can draft
/// for it — the one to fetch, and the pass is woken now (a split with no
/// drafter checks one token a round trip where one with a drafter keeps 3-6,
/// `docs/plans/large_model_splits.md`); with one held — found, or held but with
/// no room this time (memory, not a missing model) — prune keeps it while
/// splits keep using it, and a larger one is fetched beside it where the swarm
/// has one (#331) on the pass's own rounds. Recorded only on a miss before
/// 2026-10-24, so a node holding a small model for any other reason — the
/// owner's, or one it serves — guessed with it for good. Nothing where the
/// owner named the drafter (`inference.draft_model`).
pub(crate) fn note_split_wanting_a_drafter(
    state: &SharedState,
    target: &ModelId,
    drafter_found: bool,
) {
    if state.cfg().inference.draft_model.is_some() {
        return;
    }
    let now = std::time::Instant::now();
    let before = state.models.drafters_wanted.insert(target.clone(), now);
    if drafter_found || drafter_to_leave_room_for(state, target).is_some() {
        return;
    }
    if before.is_none_or(|at| now.duration_since(at) > NOTE_AGAIN_AFTER) {
        tracing::info!(
            target_model = %target,
            "DIAG: a split of this model ran with no model here to guess ahead for it — \
             the download pass looks for one"
        );
        state.models.auto_manage_notify.notify_one();
    }
}

/// How often a split without a drafter wakes the download pass again.
const NOTE_AGAIN_AFTER: std::time::Duration = std::time::Duration::from_secs(60 * 60);

/// Candidates whose header the download pass learns for one target in a pass —
/// a few MB each.
pub(crate) const MOST_DRAFTER_HEADERS_A_PASS: usize = 2;

/// The most a drafter fetched on its own may take on disk: a small thing (a
/// 3B at Q4, a 1.5B at Q8). A design for split speed may need a user to hold
/// a 0.5B model, never a large one (`memory/feedback_no_full_model_copies.md`,
/// user 2026-09-28) — and half a 70B target would be a 35B. A drafter
/// the owner holds already is chosen at any size.
pub(crate) const MOST_DRAFTER_BYTES_TO_FETCH: u64 = 2 * 1024 * 1024 * 1024;

/// What the download pass does so a split of `target` can guess
/// ([`drafter_to_acquire`]): keep the drafter held here, if there is one, and
/// take the next step.
///
/// A replacement is fetched BESIDE the drafter it replaces (#331), as a
/// rolling update starts the new copy before it stops the old one
/// (`maxUnavailable: 0`): the held one is kept on EVERY answer — while the
/// better one's headers are learned and while its parts arrive — and is let go
/// only once the better one is held whole and [`chosen`], when it becomes
/// `held` itself. Before 2026-10-24 the answer was one of `Held`, `Fetch`,
/// `LearnHeaders`, so there was no answer that kept one model and fetched
/// another, and a node that had fetched the 1.5B under 0.3.248's quarter rule
/// kept guessing with it after 0.3.249 allowed the 3B (18.1 against 16.0 tok/s
/// on the real link).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DrafterToAcquire {
    /// The drafter [`chosen`] picks among the models held whole here — prune
    /// keeps it whatever `next` is.
    pub held: Option<ModelId>,
    pub next: DrafterStep,
}

/// The download pass's next step towards a drafter for a target.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum DrafterStep {
    /// Fetch this model's parts: the drafter [`chosen`] would pick were it
    /// held — larger than [`DrafterToAcquire::held`], where one is.
    Fetch(ModelId),
    /// Learn these candidates' headers first — the vocabulary and the size are
    /// read from them.
    LearnHeaders(Vec<ModelId>),
    /// Nothing to fetch: no model the swarm knows could draft for it, or none
    /// better than the one held.
    Nothing,
}

impl DrafterToAcquire {
    /// The model to fetch now, if any.
    pub(crate) fn fetching(&self) -> Option<&ModelId> {
        match &self.next {
            DrafterStep::Fetch(model) => Some(model),
            _ => None,
        }
    }

    /// The headers to learn before the choice can be made.
    pub(crate) fn headers_to_learn(&self) -> &[ModelId] {
        match &self.next {
            DrafterStep::LearnHeaders(models) => models,
            _ => &[],
        }
    }

    /// Does this answer keep `model` here: the drafter held, or the one being
    /// fetched. The one answer prune and the download pass ask.
    pub(crate) fn keeps(&self, model: &ModelId) -> bool {
        self.held.as_ref() == Some(model) || self.fetching() == Some(model)
    }
}

/// The drafter `target` has here, and the model to fetch so it has the best
/// one: of the models the swarm knows that this node does not hold whole, the
/// one [`chosen`] would pick were it held — the largest that can draft for it,
/// [`within_reach`] of its size ([`largest_that_can_draft`]), and LARGER (by
/// [`size_proxy`]) than the one held, where one is — among those `may_fetch`
/// accepts (the download pass's own gates: trust, refusals, the canonical
/// upload, an uncensored model nobody chose). Larger, not merely cheaper: of
/// two copies of one model the cheaper reads fewer bytes a guess, which a flow
/// turned into waiting (Q8_0 against fp16, 2026-10-23), so a copy is never
/// worth a fetch. Only once every likely candidate's header is here, so a later
/// header cannot turn up a better one after the parts were fetched: a likely
/// candidate is a model of the target's family (the manifest's architecture —
/// the vocabulary check on the header is the final word) no larger on disk
/// than the target or [`MOST_DRAFTER_BYTES_TO_FETCH`].
/// Their headers are named for learning first, [`MOST_DRAFTER_HEADERS_A_PASS`]
/// at a time, largest first. Nothing to fetch where the owner named the
/// drafter: [`chosen`] drafts with that one or with none.
pub(crate) fn drafter_to_acquire(
    state: &SharedState,
    target: &ModelId,
    may_fetch: impl Fn(&crate::types::ModelManifest) -> bool,
) -> DrafterToAcquire {
    let held = drafter_to_leave_room_for(state, target).map(|spec| spec.model_id);
    let next = if state.cfg().inference.draft_model.is_some() {
        DrafterStep::Nothing
    } else {
        next_drafter_step(state, target, held.as_ref(), may_fetch)
    };
    DrafterToAcquire { held, next }
}

/// [`drafter_to_acquire`]'s step: the candidates, then [`acquisition_among`]
/// with the held drafter's size to beat.
fn next_drafter_step(
    state: &SharedState,
    target: &ModelId,
    held: Option<&ModelId>,
    may_fetch: impl Fn(&crate::types::ModelManifest) -> bool,
) -> DrafterStep {
    let target_line = release_line(target);
    let to_beat = match held {
        None => None,
        Some(held) => match size_proxy(state, held) {
            Some(size) => Some(HeldDrafter {
                size: size.size,
                of_the_targets_line: release_line(held) == target_line,
            }),
            // Chosen, so sized a moment ago: a header gone since cannot be
            // compared with — keep it, fetch nothing.
            None => return DrafterStep::Nothing,
        },
    };
    let registry = &state.model_registry;
    let Some(target_manifest) = registry.get_manifest(target) else {
        return DrafterStep::Nothing;
    };
    let family = std::mem::discriminant(&target_manifest.architecture);
    let mut likely: Vec<crate::types::ModelManifest> = registry
        .models()
        .into_iter()
        .filter(|m| {
            m.id != *target
                && std::mem::discriminant(&m.architecture) == family
                && m.total_size_bytes <= target_manifest.total_size_bytes
                && m.total_size_bytes <= MOST_DRAFTER_BYTES_TO_FETCH
                && !state.has_complete_split_model(&m.id)
                && may_fetch(m)
        })
        .collect();
    likely.sort_by(|a, b| {
        b.total_size_bytes
            .cmp(&a.total_size_bytes)
            .then_with(|| a.id.0.cmp(&b.id.0))
    });
    acquisition_among(
        likely.into_iter().map(|m| m.id).collect(),
        |m| state.gguf_meta_for(m).is_some(),
        || size_proxy(state, target).map(|s| s.size),
        &target_line,
        small_drafter_fraction(state),
        to_beat,
        |m| size_proxy(state, m),
        |m| could_draft(state, target, m),
    )
}

/// The drafter held here, as a fetch must beat it.
#[derive(Clone, Copy, Debug)]
struct HeldDrafter {
    /// Its [`size_proxy`].
    size: u64,
    /// Whether it is of the target's own [`release_line`].
    of_the_targets_line: bool,
}

impl HeldDrafter {
    /// Is a drafter of `size`, of the target's line or not, worth fetching
    /// beside this one: LARGER, or the same size where this one is of another
    /// line and that one of the target's — a different model, guessing as the
    /// target does more often ([`release_line`]), never another copy of this one.
    fn beaten_by(&self, size: u64, of_the_targets_line: bool) -> bool {
        size > self.size || (size == self.size && of_the_targets_line && !self.of_the_targets_line)
    }
}

/// [`drafter_to_acquire`]'s decision over the `likely` candidates, largest on
/// disk first: their missing headers first ([`MOST_DRAFTER_HEADERS_A_PASS`] of
/// them), then the largest that can draft ([`largest_that_can_draft`], a
/// small one up to `1 / small_fraction`) against the target's size, which
/// needs only the target's own header — of those that beat `to_beat`, the
/// drafter held here ([`HeldDrafter::beaten_by`]).
#[allow(clippy::too_many_arguments)]
fn acquisition_among(
    likely: Vec<ModelId>,
    has_header: impl Fn(&ModelId) -> bool,
    target_size: impl FnOnce() -> Option<u64>,
    target_line: &str,
    small_fraction: u64,
    to_beat: Option<HeldDrafter>,
    size: impl Fn(&ModelId) -> Option<DrafterSize>,
    can_draft: impl FnMut(ModelId) -> Option<DrafterSpec>,
) -> DrafterStep {
    let headerless: Vec<ModelId> = likely
        .iter()
        .filter(|m| !has_header(m))
        .take(MOST_DRAFTER_HEADERS_A_PASS)
        .cloned()
        .collect();
    if !headerless.is_empty() {
        return DrafterStep::LearnHeaders(headerless);
    }
    let Some(target_size) = target_size() else {
        return DrafterStep::Nothing;
    };
    let sized: Vec<(DrafterSize, ModelId)> = likely
        .into_iter()
        .filter_map(|m| size(&m).map(|s| (s, m)))
        .filter(|(s, m)| {
            to_beat.is_none_or(|held| held.beaten_by(s.size, release_line(m) == target_line))
        })
        .collect();
    match largest_that_can_draft(target_line, target_size, small_fraction, sized, can_draft) {
        Some(spec) => DrafterStep::Fetch(spec.model_id),
        None => DrafterStep::Nothing,
    }
}

/// The one choice behind both: the named drafter, else the largest that can
/// draft, of those `keep` accepts.
fn chosen(
    state: &SharedState,
    target: &ModelId,
    keep: impl Fn(&DrafterSpec) -> bool,
) -> Option<DrafterSpec> {
    if let Some(named) = state.cfg().inference.draft_model.clone() {
        return spec_for(state, target, ModelId(named)).filter(&keep);
    }
    let target_size = size_proxy(state, target)?.size;
    // Collected first: `spec_for` reads the same map, and a read nested inside
    // an iteration of a DashMap can wait on a writer queued between the two.
    let mut held: Vec<ModelId> = state
        .split_models
        .iter()
        .filter(|e| e.value().is_complete)
        .map(|e| e.key().0.clone())
        .collect();
    held.sort_by(|a, b| a.0.cmp(&b.0));
    held.dedup();
    let sized: Vec<(DrafterSize, ModelId)> = held
        .into_iter()
        .filter_map(|m| size_proxy(state, &m).map(|size| (size, m)))
        .collect();
    largest_that_can_draft(
        &release_line(target),
        target_size,
        small_drafter_fraction(state),
        sized,
        |m| spec_for(state, target, m).filter(&keep),
    )
}

/// Would the drafter's worker load once the target's layers `here` are loaded
/// (`ModelProcessPool::guest_fit_beside`)? Asked before the
/// choice, so a drafter the node has no memory for gives way to a smaller one,
/// or to none (#275). Measured on a node without a graphics card holding 28
/// layers of Qwen3-30B-A3B (6,967 of a 7,995 MB budget): Qwen3-1.7B was chosen
/// on every request, refused at its spawn (2,899 MB), and each reply went
/// unguessed after a refusal in the log. Weighed beside the layers the PLAN
/// loads, not those loaded so far: on a cold start those are fewer, or none,
/// and the same refusal came after the choice. A/B
/// `SWARMLLM_DRAFTER_BESIDE=loaded` weighs it beside what is loaded now.
///
/// Where it would load decides too: a drafter in system memory guesses at the
/// processor's pace, and is held to [`may_guess_there`].
fn admitted(
    state: &SharedState,
    target: &ModelId,
    spec: &DrafterSpec,
    conversation: u64,
    here: &TargetHere<'_>,
) -> bool {
    if !drafter_admission_asked() {
        return true;
    }
    let kv = crate::model::auto_manage::vram::KvCharge::Conversation(conversation);
    let pool = &state.model_process_pool;
    let fit = if beside_the_plan() {
        pool.guest_fit_beside(
            &spec.model_id,
            spec.layer_range,
            kv,
            target,
            here.ranges,
            here.requester,
        )
    } else {
        pool.guest_fit(&spec.model_id, spec.layer_range, kv, here.requester)
    };
    if !fit.admitted() {
        tracing::debug!(
            target_model = %target,
            drafter = %spec.model_id,
            beside = if beside_the_plan() { "the plan's layers" } else { "the layers loaded now" },
            "this node's memory has no room for the drafter — not choosing it"
        );
        return false;
    }
    let reads = size_proxy(state, &spec.model_id).map_or(u64::MAX, |s| s.reads);
    if !may_guess_there(fit, reads) {
        tracing::debug!(
            target_model = %target,
            drafter = %spec.model_id,
            reads_a_guess = reads,
            "the drafter would guess on this node's processor, where a guess reading that much is too slow — not choosing it"
        );
        return false;
    }
    true
}

/// The most a drafter may read a guess where it guesses on the PROCESSOR —
/// in system memory, whatever the size of its target (#373): the small
/// drafter's own figure ([`SMALL_DRAFTER_READS`], ~60 ms a guess on this
/// project's test machine).
///
/// The size rule ([`within_reach`]) is a fraction of the target — a quarter of
/// a 70B is a 10 GB drafter — and was measured with the drafter on the CARD,
/// in the flow, where its guesses are not what the reply waits on. On the
/// processor they are: the continuous stream (`dsd_stream`) builds each chunk
/// by drafting it, and a chunk waits for its guesses. Llama-3.3-70B split this
/// machine [0,16) + the Australian RTX 3060 [16,80), the drafter on the
/// processor (2026-10-30, six arms alternating, one binary, route pinned): the
/// 8B Llama-3.1 that size-first chose guessed at 136 ms against the 3B's 60, was
/// kept no more often (0.699-0.792 against 0.730-0.814, the same prompts), and
/// three replies of 96 tokens decoded in 160.2-161.1 s against 133.8-141.6
/// (~1.17x). Offline, over 747 of the 70B's own tokens, the 8B took
/// its first choice 0.855 of the time against the 3B's 0.846. "Decoding
/// Speculative Decoding" (Yan, Agarwal, Venkataraman 2024, arXiv 2402.01528)
/// found the same with a 65B target: the drafter's latency, not its
/// language-modelling accuracy, decides the speed-up.
const PROCESSOR_DRAFTER_MOST_READS: u64 = SMALL_DRAFTER_READS;

/// May a drafter reading `reads` a guess guess where `fit` puts it:
/// anywhere on a card, in system memory only up to
/// [`PROCESSOR_DRAFTER_MOST_READS`]. `SWARMLLM_PROCESSOR_DRAFTER_READS=any`
/// lifts the bound — the choice before #373, for an A/B inside one binary.
fn may_guess_there(fit: crate::inference::process_pool::GuestFit, reads: u64) -> bool {
    static ANY: OnceLock<bool> = OnceLock::new();
    let any = *ANY.get_or_init(|| {
        std::env::var("SWARMLLM_PROCESSOR_DRAFTER_READS").is_ok_and(|v| v.trim() == "any")
    });
    any || fit != crate::inference::process_pool::GuestFit::InSystemMemory
        || reads <= PROCESSOR_DRAFTER_MOST_READS
}

/// `SWARMLLM_DRAFTER_ADMISSION=0` chooses a drafter without asking whether it
/// would load ([`admitted`]) — the behaviour before #275, for an A/B inside one
/// binary. Read once.
fn drafter_admission_asked() -> bool {
    static ASK: OnceLock<bool> = OnceLock::new();
    *ASK.get_or_init(|| {
        std::env::var("SWARMLLM_DRAFTER_ADMISSION").map_or(true, |v| v.trim() != "0")
    })
}

/// `SWARMLLM_DRAFTER_BESIDE=loaded` weighs the drafter beside the target's
/// layers loaded NOW, not those the plan loads here ([`admitted`]) — the
/// behaviour before, for an A/B inside one binary. Read once.
fn beside_the_plan() -> bool {
    static PLAN: OnceLock<bool> = OnceLock::new();
    *PLAN.get_or_init(|| {
        std::env::var("SWARMLLM_DRAFTER_BESIDE").map_or(true, |v| v.trim() != "loaded")
    })
}

/// Of `candidates`, the largest [`within_reach`] of `target_size` — a small
/// one up to `1 / small_fraction` of it ([`small_drafter_fraction`]) — that
/// `can_draft` accepts. Of two the same size, the one of the target's own
/// release line (`target_line`, [`release_line`]); then — copies of one model
/// at different precisions — the one reading fewer bytes a guess; then the
/// model id, so the choice does not depend on map order.
fn largest_that_can_draft(
    target_line: &str,
    target_size: u64,
    small_fraction: u64,
    mut candidates: Vec<(DrafterSize, ModelId)>,
    mut can_draft: impl FnMut(ModelId) -> Option<DrafterSpec>,
) -> Option<DrafterSpec> {
    candidates.retain(|(d, _)| within_reach(d, target_size, small_fraction));
    candidates.sort_by_cached_key(|(d, m)| {
        (
            std::cmp::Reverse(d.size),
            release_line(m) != target_line,
            d.reads,
            m.0.clone(),
        )
    });
    candidates.into_iter().find_map(|(_, m)| can_draft(m))
}

/// The release line a model id names: its base name with the quantisation
/// gone ([`crate::model::auto_manage::quant::inferred_base_name`], the one
/// reading of a quant tag) and its size label too — `qwen2.5-14b-instruct-q4-k-m`
/// and `qwen2.5-3b-instruct-q4-k-m` are both `qwen2.5-instruct`,
/// `qwen2.5-coder-3b-instruct-q4-k-m` is `qwen2.5-coder-instruct`.
///
/// A drafter trained as its target was agrees with it more often at the same
/// size (2026-10-29, `~/swarmllm-318/ranks14/`, the 14B's own greedy replies as
/// the truth, 3,844 positions): Qwen2.5-3B-Instruct took the 14B Instruct's
/// first choice 0.784 of the time in chat and 0.879 in code against
/// Qwen2.5-Coder-3B's 0.745 and 0.868 — the general 1.5B equalled the coder
/// 3B. So the line breaks a tie in size, never more: the coder 3B still beat
/// the general 1.5B replayed on the long link. The headers cannot say it: the
/// general uploads carry no `general.basename` and name themselves
/// `qwen2.5-3b-instruct` where the coder's say `Qwen2.5 Coder 3B Instruct AWQ`.
/// Read from the id, a line it misreads only loses the tie — the order before.
fn release_line(model: &ModelId) -> String {
    let is_size_label = |token: &str| {
        token
            .strip_suffix(['b', 'm'])
            .is_some_and(|n| !n.is_empty() && n.parse::<f64>().is_ok())
    };
    crate::model::auto_manage::quant::inferred_base_name(&model.0)
        .split('-')
        .filter(|token| !is_size_label(token))
        .collect::<Vec<_>>()
        .join("-")
}

/// Have the drafter's worker read the prompt NOW — spawned beside the target's
/// own prompt pass, which runs over the network, so the drafter's read (and, on
/// a request's first use, its worker's spawn and load: 4.4 s measured for
/// qwen2.5-0.5b on an RTX 3070) happens while the reply could not have started
/// anyway, instead of after it. A guess-free call (`gamma` 0) that leaves the
/// prompt in the worker's cache; on success the caller marks it read with
/// [`EngineDrafter::prompt_read`], on failure the first round reads it itself.
/// `key` is the attempt's own ([`draft_key`]), shared with its [`EngineDrafter`],
/// and `requester` whose request it guesses for (`pipeline::worker_requester`).
pub(super) fn read_ahead(
    state: std::sync::Arc<SharedState>,
    spec: &DrafterSpec,
    key: uuid::Uuid,
    prompt_ids: Vec<u32>,
    sampling: &SamplingParams,
    cancel: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
    requester: Requester,
) -> ReadAhead {
    let d = IpcDraft {
        request_id: key,
        model_id: spec.model_id.clone(),
        layer_range: spec.layer_range,
        keep: 0,
        append: prompt_ids,
        gamma: 0,
        sampling: sampling.clone(),
        history: Vec::new(),
        coupling_seed: None,
        for_the_owner: requester == Requester::Owner,
        runner_ups: false,
        stop_below: None,
        tree: None,
        flow: None,
    };
    let release = release_of(state.clone(), key);
    ReadAhead::new(
        tokio::spawn(async move {
            state
                .model_process_pool
                .draft(d, cancel)
                .await
                .map(|drafted| drafted.tokens)
        }),
        release,
    )
}

/// The key one attempt at speculating a request keeps its drafter's cache
/// under — fresh per attempt, never the request's id.
///
/// The router retries a failed request under the SAME id, so a cache keyed by
/// it was shared by the attempt that failed and the one retrying it: the dead
/// attempt's read-ahead, still running, superseded the retry's call on the
/// drafter's worker (which evicted the worker), and its release of the request
/// id then emptied the retry's caches — the target's own segment included, so
/// the retried reply ended after one token (2026-09-28, the failover rig with
/// speculation on). Keyed per attempt, nothing an attempt does reaches another,
/// and its own release ([`release_of`], run when its [`ReadAhead`] or
/// [`EngineDrafter`] is dropped) touches only the drafter's cache.
pub(super) fn draft_key() -> uuid::Uuid {
    uuid::Uuid::new_v4()
}

/// Release the drafter's cache for `key` on every worker holding it — only the
/// drafter's worker ever does, since the key is the attempt's own.
pub(super) fn release_of(state: std::sync::Arc<SharedState>, key: uuid::Uuid) -> Release {
    Box::new(move || {
        Box::pin(async move { state.model_process_pool.release_request_kv(key).await })
    })
}

/// What a dropped [`ReadAhead`] runs once its call has finished.
type Release = Box<dyn FnOnce() -> futures::future::BoxFuture<'static, ()> + Send>;

/// A [`read_ahead`] in flight, owned by the attempt that started it.
///
/// Awaited by [`Self::finish`] on the path that uses it, which hands the cache
/// on to the attempt's [`EngineDrafter`]. On every other path — the prompt pass
/// failing, a fallback to another loop — it is dropped, and a dropped
/// `JoinHandle` detaches its task (gotcha #146): the call runs on and leaves a
/// cache on the drafter's worker that nothing else will release. So dropping
/// this lets the call finish — not an abort, which part-way through the
/// drafter's load would abandon a spawning worker (gotcha #459) — and then
/// releases that cache, under the attempt's own key ([`draft_key`]). A type
/// whose `Drop` runs, not a statement after an `.await` (gotcha #593).
pub(super) struct ReadAhead {
    handle: Option<tokio::task::JoinHandle<Result<Vec<u32>, SwarmError>>>,
    release: Option<Release>,
}

impl ReadAhead {
    fn new(
        handle: tokio::task::JoinHandle<Result<Vec<u32>, SwarmError>>,
        release: Release,
    ) -> Self {
        Self {
            handle: Some(handle),
            release: Some(release),
        }
    }

    /// Wait for the read-ahead: `None` only if it has already been taken.
    pub(super) async fn finish(
        mut self,
    ) -> Option<Result<Result<Vec<u32>, SwarmError>, tokio::task::JoinError>> {
        match self.handle.take() {
            Some(h) => Some(h.await),
            None => None,
        }
    }
}

impl Drop for ReadAhead {
    fn drop(&mut self) {
        let (Some(h), Some(release)) = (self.handle.take(), self.release.take()) else {
            return;
        };
        if let Ok(rt) = tokio::runtime::Handle::try_current() {
            rt.spawn(async move {
                let _ = h.await;
                release().await;
            });
        }
    }
}

/// One attempt's drafting state on this side of the worker IPC. Dropping it
/// releases the drafter's cache (`release`), whichever way the attempt ended.
pub(super) struct EngineDrafter {
    spec: DrafterSpec,
    /// The attempt's own key ([`draft_key`]) — what the drafter's worker keeps
    /// its cache under.
    key: uuid::Uuid,
    release: Option<Release>,
    /// The request's tokens so far — prompt, then every token of the reply —
    /// ending with the one the next guess follows.
    seq: Vec<u32>,
    /// How many leading positions of the drafter's cache are known to hold
    /// `seq`'s tokens.
    valid: usize,
    /// The last call, while its round's check is pending: the sequence length it
    /// read up to, and how many guesses it made.
    open: Option<(usize, usize)>,
    /// While a tree flows: the tree's start as the worker last heard it. A
    /// flow call carries every op queued (a `FlowOp::Rebase` among them), so
    /// once one answers, the worker's cache holds the sequence below the start
    /// the caller had then ([`Self::flow_ended`]). `None`: no call of this
    /// flow has answered.
    flow_heard: Option<usize>,
    calls: u32,
    /// Whose request the guesses are for (`pipeline::worker_requester`): every
    /// call says so (`IpcDraft::for_the_owner`), and its spawn is weighed
    /// against that requester's memory cap (`process_pool::drafter_requester`,
    /// #351). Every call said the owner's before 2026-10-26, and every spawn
    /// was weighed as the swarm's.
    requester: Requester,
}

/// The probability below which a drafter stops guessing for the round
/// (`SplitModel::draft_after`): 0.4, Hugging Face's assisted-generation choice
/// for the same rule. `SWARMLLM_DRAFT_CONFIDENCE=<p>` sets it for an A/B inside
/// one binary; 0 guesses every token the round asks for. Read once.
pub(super) fn draft_confidence_floor() -> Option<f32> {
    static F: OnceLock<Option<f32>> = OnceLock::new();
    *F.get_or_init(|| confidence_floor_from(std::env::var("SWARMLLM_DRAFT_CONFIDENCE").ok(), 0.4))
}

/// The continuous stream's floor (`dsd_stream`): none — a chunk is guessed
/// whole. A guess the drafter is unsure of ended a round where it costs the
/// far computer work for little, but in the stream a chunk cut short was also
/// HELD, nothing built past it until its answer came back, and that wait cost
/// more than the guesses it saved. On the real Thailand ↔ Australia link
/// (`docs/plans/faster_than_local.md` § 8.6) a floor raised to 0.67 made
/// replies 27% slower. `SWARMLLM_STREAM_DRAFT_CONFIDENCE=<p>` restores one for
/// an A/B (0.4 is the rounds'). Read once.
pub(super) fn stream_confidence_floor() -> Option<f32> {
    static F: OnceLock<Option<f32>> = OnceLock::new();
    *F.get_or_init(|| {
        confidence_floor_from(std::env::var("SWARMLLM_STREAM_DRAFT_CONFIDENCE").ok(), 0.0)
    })
}

/// A floor as a knob gives it: a probability above 0, or none.
fn confidence_floor_from(knob: Option<String>, default: f32) -> Option<f32> {
    let p = knob
        .and_then(|v| v.trim().parse::<f32>().ok())
        .unwrap_or(default);
    (p > 0.0).then_some(p)
}

/// `SWARMLLM_FAULT_DRAFT_FAIL=<n>`: this node's in-engine drafter fails its
/// `n`th call of every reply (counting from 1), as a drafter worker that cannot
/// load or dies would. The test that a reply then finishes without guessing
/// ahead instead of ending where the drafter broke (`pipeline::dsd`,
/// `drafting_off`) — the only way to break a drafter on demand. Unset in
/// production; read once.
fn fault_draft_fail_at() -> Option<u32> {
    static N: OnceLock<Option<u32>> = OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("SWARMLLM_FAULT_DRAFT_FAIL")
            .ok()
            .and_then(|v| v.trim().parse::<u32>().ok())
    })
}

impl EngineDrafter {
    /// `prompt_ids` must be the target's own tokenization of the prompt — the
    /// ids its first segment read — with the first reply token appended by
    /// [`Self::push`] once the prompt pass has sampled it.
    /// `key` and `release` are the attempt's ([`draft_key`], [`release_of`]),
    /// shared with its [`read_ahead`], as is `requester`.
    pub(super) fn new(
        spec: DrafterSpec,
        key: uuid::Uuid,
        prompt_ids: Vec<u32>,
        release: Release,
        requester: Requester,
    ) -> Self {
        Self {
            spec,
            key,
            release: Some(release),
            seq: prompt_ids,
            valid: 0,
            open: None,
            flow_heard: None,
            calls: 0,
            requester,
        }
    }

    pub(super) fn model_id(&self) -> &ModelId {
        &self.spec.model_id
    }

    /// The request's tokens so far: the prompt, then the reply — what a copied
    /// run is looked up in (`dsd_flow`).
    pub(super) fn tokens(&self) -> &[u32] {
        &self.seq
    }

    /// Token ids below this are the drafter's own: a row it is handed must
    /// carry one.
    pub(super) fn reads(&self) -> u32 {
        self.spec.reads
    }

    /// Tokens the reply now contains, in order — what a round emitted.
    pub(super) fn push(&mut self, tokens: &[u32]) {
        self.seq.extend_from_slice(tokens);
    }

    /// The drafter's worker has read the whole prompt ([`read_ahead`]): its
    /// cache holds every token of `seq` so far, and the first round reads only
    /// what follows.
    pub(super) fn prompt_read(&mut self) {
        self.valid = self.seq.len();
    }

    /// Ask the drafter's worker for `gamma` guesses following the sequence so
    /// far. `history` is the reply so far (what the penalties read) and
    /// `coupling_seed` the request's shared noise, `None` for argmax guesses.
    /// `stop_below` ends the guesses after one the drafter gives less than
    /// that, keeping it: a round passes [`draft_confidence_floor`], the
    /// stream [`stream_confidence_floor`] — each caller says which, as they
    /// pay for a short answer differently.
    #[allow(clippy::too_many_arguments)]
    pub(super) async fn draft(
        &mut self,
        state: &SharedState,
        gamma: u32,
        sampling: &SamplingParams,
        history: &[u32],
        coupling_seed: Option<u64>,
        stop_below: Option<f32>,
        cancel: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
    ) -> Result<Vec<u32>, SwarmError> {
        self.draft_chain(
            state,
            gamma,
            sampling,
            history,
            coupling_seed,
            stop_below,
            cancel,
            false,
        )
        .await
        .map(|d| d.tokens)
    }

    /// [`Self::draft`] with each guess's runner-up (`worker_ipc::RunnerUp`,
    /// #381) — for greedy guesses only: with a `coupling_seed` none come
    /// back, and neither do any from a worker that did not answer them, so
    /// the caller reads `runner_ups` as possibly empty.
    #[allow(clippy::too_many_arguments)]
    pub(super) async fn draft_with_runner_ups(
        &mut self,
        state: &SharedState,
        gamma: u32,
        sampling: &SamplingParams,
        history: &[u32],
        coupling_seed: Option<u64>,
        stop_below: Option<f32>,
        cancel: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
    ) -> Result<crate::inference::worker_ipc::Drafted, SwarmError> {
        self.draft_chain(
            state,
            gamma,
            sampling,
            history,
            coupling_seed,
            stop_below,
            cancel,
            true,
        )
        .await
    }

    #[allow(clippy::too_many_arguments)]
    async fn draft_chain(
        &mut self,
        state: &SharedState,
        gamma: u32,
        sampling: &SamplingParams,
        history: &[u32],
        coupling_seed: Option<u64>,
        stop_below: Option<f32>,
        cancel: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
        runner_ups: bool,
    ) -> Result<crate::inference::worker_ipc::Drafted, SwarmError> {
        self.before_a_call()?;
        let d = IpcDraft {
            request_id: self.key,
            model_id: self.spec.model_id.clone(),
            layer_range: self.spec.layer_range,
            keep: self.valid as u32,
            append: self.seq[self.valid..].to_vec(),
            gamma,
            sampling: sampling.clone(),
            history: history.to_vec(),
            coupling_seed,
            for_the_owner: self.requester == Requester::Owner,
            runner_ups,
            stop_below,
            tree: None,
            flow: None,
        };
        let drafted = state.model_process_pool.draft(d, cancel).await?;
        self.open = Some((self.seq.len(), drafted.tokens.len()));
        Ok(drafted)
    }

    /// The checks every call makes before asking the worker: the fault switch,
    /// something new to read, and every token inside the drafter's vocabulary.
    fn before_a_call(&mut self) -> Result<(), SwarmError> {
        self.calls += 1;
        if fault_draft_fail_at() == Some(self.calls) {
            return Err(SwarmError::ServiceUnavailable(
                "SWARMLLM_FAULT_DRAFT_FAIL: this drafter call fails on purpose".into(),
            ));
        }
        if self.seq.len() <= self.valid {
            return Err(SwarmError::Internal(
                "drafting with nothing new to read — the sequence was not advanced".into(),
            ));
        }
        if self.seq[self.valid..].iter().any(|&t| t >= self.spec.reads) {
            return Err(SwarmError::Inference(
                "the reply holds a token the drafter's vocabulary has no row for".into(),
            ));
        }
        Ok(())
    }

    /// Ask the drafter's worker for a TREE of guesses following the sequence
    /// so far (#134, `SplitModel::draft_tree_after`): `seed` (a tree already
    /// grown after the sequence's last token; empty for none) and `rows` more,
    /// `per_step` read a forward, fewer once `grow_for_ms` has passed. Answers
    /// each guess's token, its parent among them (`None`: it follows the
    /// sequence's last token) and its path's rating, likeliest path first.
    /// The drafter's cache holds the sequence afterwards and no guess, so the
    /// next call reads exactly what the check kept and the token it chose.
    ///
    /// A tree's guesses are rated from the last token's logits, so a call
    /// with nothing new to read — the second call on one sequence, growing
    /// beside a check the first call's tree went out in — reads that token
    /// again.
    #[allow(clippy::too_many_arguments)]
    pub(super) async fn draft_tree(
        &mut self,
        state: &SharedState,
        seed: &crate::inference::draft_tree::GrownTree,
        rows: u32,
        per_step: u32,
        grow_for_ms: Option<u32>,
        sampling: &SamplingParams,
        history: &[u32],
        coupling_seed: Option<u64>,
        cancel: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
    ) -> Result<crate::inference::draft_tree::GrownTree, SwarmError> {
        if self.seq.len() <= self.valid {
            self.read_last_again();
        }
        self.before_a_call()?;
        let d = IpcDraft {
            request_id: self.key,
            model_id: self.spec.model_id.clone(),
            layer_range: self.spec.layer_range,
            keep: self.valid as u32,
            append: self.seq[self.valid..].to_vec(),
            gamma: 0,
            sampling: sampling.clone(),
            history: history.to_vec(),
            coupling_seed,
            for_the_owner: self.requester == Requester::Owner,
            runner_ups: false,
            stop_below: None,
            tree: Some(crate::inference::worker_ipc::TreeAsk {
                rows,
                per_step,
                grow_for_ms,
                seed: crate::inference::worker_ipc::TreeSeed::of(seed),
            }),
            flow: None,
        };
        let drafted = state.model_process_pool.draft(d, cancel).await?;
        if drafted.parents.len() != drafted.tokens.len()
            || drafted.scores.len() != drafted.tokens.len()
        {
            return Err(SwarmError::Internal(format!(
                "the drafter grew {} guesses with {} parents and {} ratings",
                drafted.tokens.len(),
                drafted.parents.len(),
                drafted.scores.len()
            )));
        }
        self.valid = self.seq.len();
        self.open = None;
        Ok(crate::inference::draft_tree::GrownTree {
            parents: drafted.parents.iter().map(|&w| w.checked_sub(1)).collect(),
            tokens: drafted.tokens,
            path_logp: drafted.scores,
        })
    }

    /// One call of a tree that FLOWS (#134 (b), `SplitModel::grow_flow`):
    /// `start` begins it from the sequence's last token (the cache cut to what
    /// it holds of the sequence before that token, the rest read), else it
    /// continues this attempt's tree after `ops` — the requester's changes
    /// since the last call, which leave the tree starting at `base`
    /// (`FlowTree::base`). Answers up to `rows` proposals a level, `levels`
    /// deep, as (parent row, token) — each already a row of the drafter's
    /// tree, so the caller pushes every one, in order, and sends no op for
    /// it. While the tree flows the cache holds the sequence only below
    /// the tree's start: [`Self::flow_ended`] says where that is once it ends.
    #[allow(clippy::too_many_arguments)]
    pub(super) async fn flow(
        &mut self,
        state: &SharedState,
        start: bool,
        ops: Vec<crate::inference::draft_tree::FlowOp>,
        base: usize,
        rows: u32,
        levels: u32,
        live_most: u32,
        sampling: &SamplingParams,
        coupling_seed: Option<u64>,
        cancel: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
    ) -> Result<Vec<(u16, u32)>, SwarmError> {
        use crate::inference::draft_tree::FlowOp;
        self.calls += 1;
        if fault_draft_fail_at() == Some(self.calls) {
            return Err(SwarmError::ServiceUnavailable(
                "SWARMLLM_FAULT_DRAFT_FAIL: this drafter call fails on purpose".into(),
            ));
        }
        let pushed_past_vocab = ops
            .iter()
            .any(|op| matches!(op, FlowOp::Push { token, .. } | FlowOp::Copy { token, .. } if *token >= self.spec.reads));
        let (keep, append, root) = if start {
            let Some((&root, before)) = self.seq.split_last() else {
                return Err(SwarmError::Internal(
                    "a tree that flows started with no token to follow".into(),
                ));
            };
            let keep = self.valid.min(before.len());
            (keep, before[keep..].to_vec(), Some(root))
        } else {
            (0, Vec::new(), None)
        };
        if pushed_past_vocab
            || root.is_some_and(|r| r >= self.spec.reads)
            || append.iter().any(|&t| t >= self.spec.reads)
        {
            return Err(SwarmError::Inference(
                "the reply holds a token the drafter's vocabulary has no row for".into(),
            ));
        }
        let read_to = keep + append.len();
        let d = IpcDraft {
            request_id: self.key,
            model_id: self.spec.model_id.clone(),
            layer_range: self.spec.layer_range,
            keep: keep as u32,
            append,
            gamma: 0,
            sampling: sampling.clone(),
            history: Vec::new(),
            coupling_seed,
            for_the_owner: self.requester == Requester::Owner,
            runner_ups: false,
            stop_below: None,
            tree: None,
            flow: Some(crate::inference::worker_ipc::FlowAsk {
                start: root,
                ops,
                rows,
                levels,
                live_most,
            }),
        };
        let drafted = state.model_process_pool.draft(d, cancel).await?;
        if drafted.parents.len() != drafted.tokens.len() || drafted.parents.contains(&0) {
            return Err(SwarmError::Internal(format!(
                "the drafter proposed {} rows with {} parents",
                drafted.tokens.len(),
                drafted.parents.len(),
            )));
        }
        self.flow_answered(start, read_to, base);
        Ok(drafted
            .parents
            .iter()
            .zip(drafted.tokens)
            .map(|(&p, token)| (p - 1, token))
            .collect())
    }

    /// A flow call answered: on the flow's first, the cache holds the sequence
    /// up to the tree's root (`read_to`); after any, the worker has heard every
    /// op up to the tree starting at `base`.
    fn flow_answered(&mut self, start: bool, read_to: usize, base: usize) {
        if start {
            self.valid = read_to;
        }
        self.flow_heard = Some(base);
        self.open = None;
    }

    /// A tree that flowed has ended, the sequence holding every token its walk
    /// committed, the tree starting at `base` (`FlowTree::base` — a rebase read
    /// the path below it as context). The cache is trusted below `base` only as
    /// far as the worker HEARD: a rebase no call carried left the old tree's
    /// rows where the path now is, and a flow none of whose calls answered left
    /// the cache as it was. The next call cuts it there and reads on.
    pub(super) fn flow_ended(&mut self, base: usize) {
        self.valid = match self.flow_heard.take() {
            Some(heard) => base.min(heard),
            None => self.valid.min(base),
        }
        .min(self.seq.len());
        self.open = None;
    }

    /// The last token of the sequence is read again by the next call — after a
    /// tree whose check was refused, so a chain can follow it: the tree's call
    /// left the cache holding the whole sequence, and a call must have
    /// something new to read.
    pub(super) fn read_last_again(&mut self) {
        self.valid = self.valid.min(self.seq.len().saturating_sub(1));
        self.open = None;
    }

    /// The pending call's check kept `kept` of its guesses. The drafter's cache
    /// read every guess but the last, so that many of the kept ones are already
    /// in it and stay; the rest of the call's positions are cut next time.
    /// A round drafted elsewhere has no open call and changes nothing here.
    pub(super) fn settle(&mut self, kept: usize) {
        if let Some((read_to, guessed)) = self.open.take() {
            self.valid = read_to + kept.min(guessed.saturating_sub(1));
        }
    }

    /// Take the sequence back to its first `len` tokens — everything after a
    /// guess the check refused, which a stream (`dsd_stream`) may already have
    /// drafted several chunks beyond. The cache is trusted no further than
    /// that; the next call cuts it back and reads from there.
    pub(super) fn rewind(&mut self, len: usize) {
        self.seq.truncate(len);
        self.valid = self.valid.min(len);
        self.open = None;
    }

    /// Take the sequence to the prompt's first `prompt_len` tokens followed by
    /// `reply` — the rounds taking a reply over from a stream that ended
    /// (`dsd`), whose sequence may run chunks past it — keeping the cache as
    /// far as it agrees. The reply's last token is always left to be read:
    /// the next call must have something new to read, and that token's row
    /// is the one it guesses from.
    pub(super) fn follow(&mut self, prompt_len: usize, reply: &[u32]) {
        let agree = self
            .seq
            .get(prompt_len..)
            .unwrap_or(&[])
            .iter()
            .zip(reply)
            .take_while(|(held, said)| held == said)
            .count()
            .min(reply.len().saturating_sub(1));
        self.rewind(prompt_len + agree);
        self.push(&reply[agree..]);
    }

    #[cfg(test)]
    pub(super) fn valid(&self) -> usize {
        self.valid
    }
}

impl Drop for EngineDrafter {
    fn drop(&mut self) {
        // No call is in flight: every one is awaited inside the attempt, and a
        // cancelled one has already told the worker to stop (`ResponseGuard`),
        // which it does before reading the release queued behind it.
        if let (Some(release), Ok(rt)) =
            (self.release.take(), tokio::runtime::Handle::try_current())
        {
            rt.spawn(release());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn drafter(prompt: &[u32]) -> EngineDrafter {
        EngineDrafter::new(
            DrafterSpec {
                model_id: ModelId("d".into()),
                layer_range: (0, 24),
                reads: 1000,
            },
            uuid::Uuid::nil(),
            prompt.to_vec(),
            Box::new(|| Box::pin(async {})),
            Requester::Owner,
        )
    }

    /// The cache bookkeeping over three rounds: what the drafter holds after
    /// each check, and therefore what the next call must re-read.
    #[test]
    fn the_drafters_cache_is_trusted_exactly_as_far_as_the_checks_confirmed() {
        let mut d = drafter(&[1, 2, 3]);
        d.push(&[10]); // the prompt pass sampled 10
                       // Round 1 reads [1,2,3,10] and guesses 4: cache = 4 + 3 positions.
        d.open = Some((4, 4));
        d.settle(2); // two guesses kept: their positions are good
        assert_eq!(d.valid(), 6);
        d.push(&[11, 12, 13]); // kept 11, 12; the check sampled 13
        assert_eq!(
            &d.seq[d.valid..],
            &[13],
            "only the check's own token is new"
        );
        // Round 2: every guess kept — the last was never read into the cache.
        d.open = Some((d.seq.len(), 4));
        d.settle(4);
        assert_eq!(d.valid(), 7 + 3);
        d.push(&[20, 21, 22, 23, 24]);
        assert_eq!(
            &d.seq[d.valid..],
            &[23, 24],
            "the unread last guess, then the sample"
        );
        // Round 3 drafted elsewhere (an n-gram match): nothing changes here, and
        // the next call reads everything since.
        d.settle(3);
        assert_eq!(d.valid(), 10);
        d.push(&[30, 31]);
        assert_eq!(&d.seq[d.valid..], &[23, 24, 30, 31]);
    }

    /// A stream drafts on as if every guess were kept, then rewinds to the
    /// guess the check refused: the next call reads from the confirmed prefix,
    /// never from a chunk drafted on top of the refused guess.
    /// After a tree that flowed, the drafter's cache is trusted only to the
    /// tree start its worker HEARD — a rebase that went out with no call (the
    /// flow ended first) left the old tree's rows where the reply now is.
    #[test]
    fn a_flow_trusts_the_cache_only_to_the_start_its_worker_heard() {
        let seq: Vec<u32> = (0..20).collect();
        // The first call read to the root at 12; a later one carried a
        // rebase to 16; the walk rebased to 18 and the flow ended before any
        // call carried it.
        let mut d = drafter(&seq);
        d.flow_answered(true, 12, 12);
        d.flow_answered(false, 0, 16);
        d.flow_ended(18);
        assert_eq!(d.valid(), 16);
        // Every rebase carried: trusted to the tree's start.
        let mut d = drafter(&seq);
        d.flow_answered(true, 12, 12);
        d.flow_answered(false, 0, 18);
        d.flow_ended(18);
        assert_eq!(d.valid(), 18);
        // A second flow none of whose calls answered leaves the cache as the
        // first left it.
        d.flow_ended(19);
        assert_eq!(d.valid(), 18);
    }

    #[test]
    fn a_stream_rewinds_the_drafter_to_the_refused_guess() {
        let mut d = drafter(&[1, 2, 3]);
        d.push(&[10]); // position 3
                       // Chunk 1 guessed 11,12,13 (13 the look-ahead), taken as kept.
        d.open = Some((4, 3));
        d.settle(3);
        d.push(&[11, 12, 13]);
        // Chunk 2 read 13 and guessed 14,15 on top of it.
        d.open = Some((7, 2));
        d.settle(2);
        d.push(&[14, 15]);
        assert_eq!(d.valid(), 8);
        // Chunk 1's check kept 11 and sampled 20 where 12 was guessed.
        d.rewind(5);
        d.push(&[20]);
        assert_eq!(d.seq, vec![1, 2, 3, 10, 11, 20]);
        assert_eq!(
            d.valid(),
            5,
            "the cache is trusted only up to the refused guess"
        );
        assert_eq!(&d.seq[d.valid..], &[20]);
    }

    /// Rounds taking over from a stream that failed: the sequence becomes the
    /// prompt and the reply so far, whatever chunks the stream drafted past
    /// it, and the cache is kept only where it agrees.
    #[test]
    fn rounds_taking_over_a_stream_follow_the_reply_not_its_chunks() {
        let mut d = drafter(&[1, 2, 3]);
        d.push(&[10]);
        d.open = Some((4, 3));
        d.settle(3);
        d.push(&[11, 12, 13]);
        d.open = Some((7, 2));
        d.settle(2);
        d.push(&[14, 15]);
        // The reply kept 11 and 12, and its check sampled 30.
        d.follow(3, &[10, 11, 12, 30]);
        assert_eq!(d.seq, vec![1, 2, 3, 10, 11, 12, 30]);
        assert_eq!(d.valid(), 6, "kept as far as it agrees");
        assert_eq!(&d.seq[d.valid..], &[30]);
        // A reply the sequence holds whole still leaves its last token to read.
        let mut d = drafter(&[1, 2, 3]);
        d.push(&[10, 11, 12]);
        d.valid = 6;
        d.follow(3, &[10, 11, 12]);
        assert_eq!(d.seq, vec![1, 2, 3, 10, 11, 12]);
        assert_eq!(&d.seq[d.valid..], &[12]);
    }

    fn spec(name: &str) -> DrafterSpec {
        DrafterSpec {
            model_id: ModelId(name.into()),
            layer_range: (0, 1),
            reads: 1,
        }
    }

    /// For Qwen2.5-Coder-7B (360 M by layers × width²) this node holds the 0.5B
    /// (19 M), 1.5B (66 M) and 3B (151 M), all small: the 3B is within half, so
    /// it drafts (2026-10-24 — a quarter, the rule before, chose the 1.5B, and
    /// the 3B flowed 18.1 tok/s against its 16.0 on the real link); then the
    /// 1.5B where the 3B cannot draft, the 0.5B where neither can.
    #[test]
    fn the_largest_small_drafter_within_half_its_target_is_chosen() {
        let sized = |size| DrafterSize {
            size,
            reads: 1_700_000_000,
        };
        let held = || {
            vec![
                (sized(19), ModelId("q-0.5b".into())),
                (sized(151), ModelId("q-3b".into())),
                (sized(66), ModelId("q-1.5b".into())),
            ]
        };
        let any = |m: ModelId| Some(spec(&m.0));
        let chosen = largest_that_can_draft("", 360, DRAFTER_MAX_FRACTION, held(), any);
        assert_eq!(chosen.unwrap().model_id.0, "q-3b");
        let chosen = largest_that_can_draft("", 360, DRAFTER_MAX_FRACTION, held(), |m| {
            (m.0 != "q-3b").then(|| spec(&m.0))
        });
        assert_eq!(chosen.unwrap().model_id.0, "q-1.5b");
        let chosen = largest_that_can_draft("", 360, DRAFTER_MAX_FRACTION, held(), |m| {
            (m.0 == "q-0.5b").then(|| spec(&m.0))
        });
        assert_eq!(chosen.unwrap().model_id.0, "q-0.5b");
        assert!(largest_that_can_draft("", 360, DRAFTER_MAX_FRACTION, held(), |_| None).is_none());
        assert_eq!(
            largest_that_can_draft("", 360, 4, held(), any)
                .unwrap()
                .model_id
                .0,
            "q-1.5b",
            "SWARMLLM_DRAFTER_MAX_FRACTION=4 is the rule before"
        );
        assert!(
            largest_that_can_draft("", 30, DRAFTER_MAX_FRACTION, held(), any).is_none(),
            "nothing within half a tiny target's size"
        );
    }

    /// **A drafter that would guess on the processor reads no more a guess than
    /// a small one** (#373). The field case: Llama-3.3-70B (80 × 8192²) with
    /// Llama-3.1-8B (32 × 4096² — exactly a tenth, ~4.4 GB a guess) and
    /// Llama-3.2-3B (28 × 3072², ~1.65 GB) held. Both are within a quarter, and
    /// size first chose the 8B; landing in system memory, it guessed at 136 ms
    /// against the 3B's 60 and its replies took ~1.17x as long. On the card
    /// it is still the 8B — the rule there is unchanged.
    #[test]
    fn a_drafter_on_the_processor_reads_no_more_a_guess_than_a_small_one() {
        use crate::inference::process_pool::GuestFit;
        let target = 80 * 8192u64.pow(2);
        let held = || {
            vec![
                (
                    DrafterSize {
                        size: 32 * 4096u64.pow(2),
                        reads: 4_400_000_000,
                    },
                    ModelId("meta-llama-3.1-8b-instruct-q4-k-m".into()),
                ),
                (
                    DrafterSize {
                        size: 28 * 3072u64.pow(2),
                        reads: 1_650_000_000,
                    },
                    ModelId("llama-3.2-3b-instruct-q4-k-m".into()),
                ),
            ]
        };
        let reads_of = |m: &ModelId| {
            held()
                .into_iter()
                .find(|(_, id)| id == m)
                .map_or(u64::MAX, |(d, _)| d.reads)
        };
        let landing = |fit: GuestFit| {
            move |m: ModelId| may_guess_there(fit, reads_of(&m)).then(|| spec(&m.0))
        };
        let chosen = |fit| {
            largest_that_can_draft("", target, DRAFTER_MAX_FRACTION, held(), landing(fit))
                .map(|s| s.model_id.0)
        };
        assert_eq!(
            chosen(GuestFit::InSystemMemory).as_deref(),
            Some("llama-3.2-3b-instruct-q4-k-m"),
            "in system memory the 8B's guesses read too much: the 3B drafts"
        );
        assert_eq!(
            chosen(GuestFit::OnCard).as_deref(),
            Some("meta-llama-3.1-8b-instruct-q4-k-m"),
            "on the card the largest within reach still drafts"
        );
        assert!(may_guess_there(
            GuestFit::InSystemMemory,
            PROCESSOR_DRAFTER_MOST_READS
        ));
        assert!(!may_guess_there(
            GuestFit::InSystemMemory,
            PROCESSOR_DRAFTER_MOST_READS + 1
        ));
        assert!(
            !may_guess_there(GuestFit::InSystemMemory, u64::MAX),
            "reads unknown is a large drafter"
        );
        assert!(may_guess_there(GuestFit::OnCard, u64::MAX));
    }

    /// A drafter reading more a guess than one fetched may is held to a
    /// quarter: Qwen2.5-7B (360 M, ~4.4 GB a guess at Q4_K_M) is within half of
    /// Qwen2.5-14B (1,258 M) and not chosen for it — the 3B is; within a quarter
    /// of a target it drafts. Reads unknown (no header) is large.
    #[test]
    fn a_large_drafter_is_held_to_a_quarter_of_its_target() {
        let held = || {
            vec![
                (
                    DrafterSize {
                        size: 360,
                        reads: 4_400_000_000,
                    },
                    ModelId("q-7b".into()),
                ),
                (
                    DrafterSize {
                        size: 151,
                        reads: 1_700_000_000,
                    },
                    ModelId("q-3b".into()),
                ),
            ]
        };
        let any = |m: ModelId| Some(spec(&m.0));
        let chosen = largest_that_can_draft("", 1258, DRAFTER_MAX_FRACTION, held(), any);
        assert_eq!(chosen.unwrap().model_id.0, "q-3b");
        let chosen = largest_that_can_draft("", 1500, DRAFTER_MAX_FRACTION, held(), any);
        assert_eq!(chosen.unwrap().model_id.0, "q-7b", "360 × 4 ≤ 1,500");
        let size = |reads| DrafterSize { size: 150, reads };
        assert!(within_reach(
            &size(SMALL_DRAFTER_READS),
            300,
            DRAFTER_MAX_FRACTION
        ));
        assert!(!within_reach(
            &size(SMALL_DRAFTER_READS + 1),
            300,
            DRAFTER_MAX_FRACTION
        ));
        assert!(!within_reach(&size(u64::MAX), 300, DRAFTER_MAX_FRACTION));
        assert!(
            !within_reach(&size(1), 300, 8),
            "a small drafter is held to a tighter A/B fraction"
        );
        assert!(
            !within_reach(&size(u64::MAX), 1199, 8),
            "a tighter A/B fraction holds a large drafter too"
        );
        assert!(within_reach(&size(u64::MAX), 1200, 8));
    }

    /// The half is for a node whose graphics card is in play, where it was
    /// measured; on a node without one every drafter keeps a quarter (on the
    /// processor the 3B guessed no better than the 1.5B, at 73 ms a guess
    /// against 65).
    #[test]
    fn a_node_without_a_card_holds_every_drafter_to_a_quarter() {
        use crate::identity::Identity;
        use crate::inference::executor::ModelExecutor;
        use crate::storage::db::Database;
        let temp = tempfile::tempdir().unwrap();
        let db = Database::open(temp.path()).unwrap();
        let executor = std::sync::Arc::new(tokio::sync::Mutex::new(ModelExecutor::new()));
        let (state, _, _) = SharedState::new(
            crate::config::Config::default(),
            Identity::generate(),
            db,
            executor,
            None,
        );
        assert_eq!(
            small_drafter_fraction(&state),
            LARGE_DRAFTER_MAX_FRACTION,
            "no card in play"
        );
        state
            .model_process_pool
            .test_card_in_play
            .store(true, std::sync::atomic::Ordering::Relaxed);
        assert_eq!(
            small_drafter_fraction(&state),
            drafter_max_fraction(),
            "a card in play"
        );
    }

    /// The live node holds Qwen2.5-0.5B as fp16, Q8_0 and Q4_K_M — one model, one
    /// size — and name order gave the fp16 (2026-10-14): the same guesses for the
    /// most bytes read. The copy that reads the fewest drafts.
    #[test]
    fn of_two_copies_of_one_model_the_one_reading_fewer_bytes_drafts() {
        let held = vec![
            (
                DrafterSize {
                    size: 19,
                    reads: 988_000_000,
                },
                ModelId("qwen2.5-0.5b-instruct-fp16".into()),
            ),
            (
                DrafterSize {
                    size: 19,
                    reads: 531_000_000,
                },
                ModelId("qwen2.5-0.5b-instruct-q8-0".into()),
            ),
            (
                DrafterSize {
                    size: 19,
                    reads: u64::MAX,
                },
                ModelId("qwen2.5-0.5b-a-no-header".into()),
            ),
        ];
        let chosen =
            largest_that_can_draft("", 360, DRAFTER_MAX_FRACTION, held, |m| Some(spec(&m.0)));
        assert_eq!(chosen.unwrap().model_id.0, "qwen2.5-0.5b-instruct-q8-0");
    }

    /// A model's release line is its id less the quantisation and the size.
    #[test]
    fn a_release_line_is_the_id_less_its_size_and_quantisation() {
        for (id, line) in [
            ("qwen2.5-14b-instruct-q4-k-m", "qwen2.5-instruct"),
            ("qwen2.5-3b-instruct-q4-k-m", "qwen2.5-instruct"),
            ("qwen2.5-0.5b-instruct-fp16", "qwen2.5-instruct"),
            ("qwen2.5-0.5b-instruct-q8-0", "qwen2.5-instruct"),
            ("qwen2.5-coder-1.5b-instruct-q8-0", "qwen2.5-coder-instruct"),
            ("qwen2.5-coder-3b-instruct-q4-k-m", "qwen2.5-coder-instruct"),
            ("qwen2.5-coder-7b-instruct-q4-k-m", "qwen2.5-coder-instruct"),
            (
                "meta-llama-3.1-8b-instruct-q4-k-m",
                "meta-llama-3.1-instruct",
            ),
            ("qwen3-30b-a3b-q4-k-m", "qwen3-a3b"),
        ] {
            assert_eq!(release_line(&ModelId(id.into())), line, "{id}");
        }
    }

    /// Of two drafters the same size, the one of the target's own line drafts
    /// — Qwen2.5-3B-Instruct for Qwen2.5-14B-Instruct took the target's first
    /// choice 0.784 of the time in chat against Qwen2.5-Coder-3B's 0.745
    /// (2026-10-29) — and the coder's for a coder target. The line breaks a
    /// tie in size, never more: a LARGER drafter of another line still drafts.
    /// It comes before the bytes read: those choose between copies of ONE model.
    #[test]
    fn of_two_drafters_the_same_size_the_one_of_the_targets_line_drafts() {
        let sized = |size, reads| DrafterSize { size, reads };
        let general_3b = "qwen2.5-3b-instruct-q4-k-m";
        let coder_3b = "qwen2.5-coder-3b-instruct-q4-k-m";
        let general_1_5b = "qwen2.5-1.5b-instruct-q8-0";
        let held = || {
            vec![
                (sized(151, 1_900), ModelId(coder_3b.into())),
                (sized(151, 2_000), ModelId(general_3b.into())),
                (sized(66, 1_600), ModelId(general_1_5b.into())),
            ]
        };
        let any = |m: ModelId| Some(spec(&m.0));
        let for_target = |target: &str, held| {
            largest_that_can_draft(
                &release_line(&ModelId(target.into())),
                1258,
                DRAFTER_MAX_FRACTION,
                held,
                any,
            )
            .unwrap()
            .model_id
            .0
        };
        assert_eq!(
            for_target("qwen2.5-14b-instruct-q4-k-m", held()),
            general_3b,
            "the target's line, though it reads more a guess"
        );
        assert_eq!(
            for_target("qwen2.5-coder-14b-instruct-q4-k-m", held()),
            coder_3b
        );
        let without_the_general_3b = held()
            .into_iter()
            .filter(|(_, m)| m.0 != general_3b)
            .collect();
        assert_eq!(
            for_target("qwen2.5-14b-instruct-q4-k-m", without_the_general_3b),
            coder_3b,
            "a larger drafter of another line still drafts"
        );
        // The 0.5B pair, both Q8_0 — ids that end in a two-token tag, `q8-0`.
        let general_0_5b = "qwen2.5-0.5b-instruct-q8-0";
        let small = vec![
            (
                sized(30, 500),
                ModelId("qwen2.5-coder-0.5b-instruct-q8-0".into()),
            ),
            (sized(30, 520), ModelId(general_0_5b.into())),
        ];
        assert_eq!(
            for_target("qwen2.5-14b-instruct-q4-k-m", small),
            general_0_5b
        );
    }

    /// A node holding a drafter of another line fetches one the same size of
    /// the target's own; one holding the target's line fetches nothing the
    /// same size — and neither fetches another copy of what it holds.
    #[test]
    fn a_drafter_of_the_targets_line_is_fetched_beside_one_the_same_size_of_another() {
        let id = |s: &str| ModelId(s.into());
        let general_3b = "qwen2.5-3b-instruct-q4-k-m";
        let coder_3b_q8 = "qwen2.5-coder-3b-instruct-q8-0";
        let likely = || vec![id(general_3b), id(coder_3b_q8)];
        let size = |m: &ModelId| {
            Some(DrafterSize {
                size: 151,
                reads: if m.0 == general_3b { 1_900 } else { 3_300 },
            })
        };
        let can_draft = |m: ModelId| Some(spec(&m.0));
        let every = |_: &ModelId| true;
        let line = release_line(&id("qwen2.5-14b-instruct-q4-k-m"));
        let held = |of_the_targets_line| {
            Some(HeldDrafter {
                size: 151,
                of_the_targets_line,
            })
        };
        assert_eq!(
            acquisition_among(
                likely(),
                every,
                || Some(1258),
                &line,
                2,
                held(false),
                size,
                can_draft
            ),
            DrafterStep::Fetch(id(general_3b)),
            "the coder 3B held: the general one, the target's line"
        );
        assert_eq!(
            acquisition_among(
                likely(),
                every,
                || Some(1258),
                &line,
                2,
                held(true),
                size,
                can_draft
            ),
            DrafterStep::Nothing,
            "the general 3B held: nothing the same size, the coder's copy least of all"
        );
        let coder_line = release_line(&id("qwen2.5-coder-14b-instruct-q4-k-m"));
        assert_eq!(
            acquisition_among(
                vec![id(coder_3b_q8)],
                every,
                || Some(1258),
                &coder_line,
                2,
                held(true),
                size,
                can_draft
            ),
            DrafterStep::Nothing,
            "a coder target holding the coder 3B: its Q8_0 copy is never worth a fetch"
        );
    }

    /// Layers × width² sized Qwen3-30B-A3B at 201 M — under a quarter's reach of
    /// Qwen3-1.7B's 117 M — so no sibling ever drafted for it (2026-10-13). Its
    /// experts are counted now; a dense model's size does not move.
    #[test]
    fn a_mixture_of_experts_is_sized_with_all_its_experts() {
        // 48 × 2048², holding 8.5× what a token reads at Q2_K (the live node's
        // header: 232.3 MB a layer stored, 27.3 MB read).
        let moe = size_counting_experts(201_326_592, Some(232_300_000.0), Some(27_300_000.0));
        assert!(
            117_440_512 * LARGE_DRAFTER_MAX_FRACTION <= moe,
            "Qwen3-1.7B can draft: {moe}"
        );
        assert_eq!(
            size_counting_experts(360, Some(5.0), Some(5.0)),
            360,
            "dense: unchanged"
        );
        assert_eq!(
            size_counting_experts(360, None, None),
            360,
            "unknown: unchanged"
        );
    }

    fn meta(tokens: &[&str], types: &[i32]) -> crate::inference::split::GgufTokenizerMeta {
        crate::inference::split::GgufTokenizerMeta {
            vocab: tokens.iter().map(|t| t.to_string()).collect(),
            token_types: types.to_vec(),
            ..Default::default()
        }
    }

    /// Qwen2.5-Coder-7B and Qwen2.5-0.5B: the same list, the 7B's padded with
    /// unused `[PAD…]` entries.
    #[test]
    fn a_list_padded_with_unused_entries_is_the_same_vocabulary() {
        let target = meta(&["a", "b", "c", "[PAD3]", "[PAD4]"], &[1, 1, 3, 5, 5]);
        let drafter = meta(&["a", "b", "c"], &[1, 1, 3]);
        assert_eq!(compatible_vocabularies(&target, &drafter), Some(3));
        // And the other way round: the drafter padded, the target not.
        assert_eq!(compatible_vocabularies(&drafter, &target), Some(5));
    }

    #[test]
    fn a_different_or_meaningfully_longer_list_is_not() {
        let target = meta(&["a", "b", "c"], &[1, 1, 1]);
        assert_eq!(
            compatible_vocabularies(&target, &meta(&["a", "x", "c"], &[1, 1, 1])),
            None,
            "a token differs"
        );
        assert_eq!(
            compatible_vocabularies(&meta(&["a", "b", "c", "d"], &[1, 1, 1, 1]), &target),
            None,
            "the extra entry is a real token the drafter cannot guess or read"
        );
        assert_eq!(
            compatible_vocabularies(&meta(&["a", "b", "c", "d"], &[]), &target),
            None,
            "no token types: the extra entry cannot be shown unused"
        );
    }

    /// The rounds keep 0.4 and the stream guesses a chunk whole unless a knob
    /// says otherwise; 0 or a word is "no floor" / the default, as before.
    #[test]
    fn the_stream_guesses_a_chunk_whole_and_the_rounds_keep_their_floor() {
        assert_eq!(confidence_floor_from(None, 0.4), Some(0.4));
        assert_eq!(confidence_floor_from(None, 0.0), None);
        assert_eq!(confidence_floor_from(Some(" 0.4 ".into()), 0.0), Some(0.4));
        assert_eq!(confidence_floor_from(Some("0".into()), 0.4), None);
        assert_eq!(confidence_floor_from(Some("high".into()), 0.4), Some(0.4));
        if std::env::var_os("SWARMLLM_STREAM_DRAFT_CONFIDENCE").is_none() {
            assert_eq!(stream_confidence_floor(), None);
        }
        if std::env::var_os("SWARMLLM_DRAFT_CONFIDENCE").is_none() {
            assert_eq!(draft_confidence_floor(), Some(0.4));
        }
    }

    #[test]
    fn a_round_that_kept_nothing_trusts_only_what_it_read() {
        let mut d = drafter(&[1, 2]);
        d.push(&[5]);
        d.open = Some((3, 4));
        d.settle(0);
        assert_eq!(d.valid(), 3);
    }

    /// A prompt read ahead leaves only the first reply token for round one —
    /// and marking it read BEFORE that token is pushed is what makes it so.
    #[test]
    fn a_prompt_read_ahead_leaves_round_one_only_the_first_reply_token() {
        let mut d = drafter(&[1, 2, 3, 4]);
        d.prompt_read();
        d.push(&[9]);
        assert_eq!(d.valid(), 4);
        assert_eq!(&d.seq[d.valid..], &[9]);
    }
    fn flagged_release(flag: std::sync::Arc<std::sync::atomic::AtomicBool>) -> Release {
        Box::new(move || {
            Box::pin(async move { flag.store(true, std::sync::atomic::Ordering::SeqCst) })
        })
    }

    /// A read-ahead nobody waits for — the prompt pass failed, or the request
    /// fell back to another loop — is left to FINISH (aborting it part-way
    /// through a load abandons a spawning worker) and then releases the
    /// request's cache on the drafter's worker, which the request's own
    /// release may have missed.
    #[tokio::test]
    async fn a_read_ahead_nobody_waits_for_finishes_then_releases_the_cache() {
        use std::sync::atomic::{AtomicBool, Ordering::SeqCst};
        use std::sync::Arc;
        let released = Arc::new(AtomicBool::new(false));
        let finished = Arc::new(AtomicBool::new(false));
        let (go, wait) = tokio::sync::oneshot::channel::<()>();
        let done = finished.clone();
        let ra = ReadAhead::new(
            tokio::spawn(async move {
                let _ = wait.await;
                done.store(true, SeqCst);
                Ok(Vec::new())
            }),
            flagged_release(released.clone()),
        );
        drop(ra);
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        assert!(
            !released.load(SeqCst),
            "not while its call is still running"
        );
        go.send(()).unwrap();
        for _ in 0..100 {
            if released.load(SeqCst) {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
        assert!(
            finished.load(SeqCst),
            "the call ran to its end, not aborted"
        );
        assert!(released.load(SeqCst), "and the cache it left was released");
    }

    /// The path that uses the read-ahead waits for it and releases nothing —
    /// the reply's own cache is the drafter's from then on.
    #[tokio::test]
    async fn a_read_ahead_that_was_waited_for_releases_nothing() {
        use std::sync::atomic::{AtomicBool, Ordering::SeqCst};
        let released = std::sync::Arc::new(AtomicBool::new(false));
        let ra = ReadAhead::new(
            tokio::spawn(async { Ok(vec![7]) }),
            flagged_release(released.clone()),
        );
        assert!(matches!(ra.finish().await, Some(Ok(Ok(ref t))) if t == &vec![7]));
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        assert!(!released.load(SeqCst));
    }
    /// An attempt's drafter releases its cache when the attempt ends, however
    /// it ends — the request's own release no longer reaches it, since the
    /// cache is keyed by the attempt ([`draft_key`]).
    #[tokio::test]
    async fn an_attempts_drafter_releases_its_cache_when_dropped() {
        use std::sync::atomic::{AtomicBool, Ordering::SeqCst};
        let released = std::sync::Arc::new(AtomicBool::new(false));
        let d = EngineDrafter::new(
            DrafterSpec {
                model_id: ModelId("d".into()),
                layer_range: (0, 24),
                reads: 1000,
            },
            draft_key(),
            vec![1, 2, 3],
            flagged_release(released.clone()),
            Requester::Owner,
        );
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        assert!(!released.load(SeqCst), "not while the attempt is running");
        drop(d);
        for _ in 0..100 {
            if released.load(SeqCst) {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        }
        assert!(released.load(SeqCst));
    }

    /// Two attempts at one request never share a drafter cache.
    #[test]
    fn every_attempt_gets_its_own_drafter_key() {
        assert_ne!(draft_key(), draft_key());
    }

    /// The fetch is decided only once every likely candidate's header is
    /// here — two learned a pass, largest first — and then it is the largest
    /// that can draft, within half the target: a 1.5B before a 0.5B, a
    /// model the vocabulary check refuses passed over, one too big skipped.
    #[test]
    fn a_drafter_is_fetched_only_once_every_likely_header_is_here() {
        let id = |s: &str| ModelId(s.into());
        let likely = || vec![id("3b"), id("1.5b"), id("0.5b"), id("other-vocab")];
        let size = |m: &ModelId| {
            let size = match m.0.as_str() {
                "3b" => 264,
                "1.5b" => 50,
                "0.5b" => 19,
                "other-vocab" => 60,
                _ => return None,
            };
            Some(DrafterSize { size, reads: size })
        };
        let can_draft = |m: ModelId| {
            (m.0 != "other-vocab").then_some(DrafterSpec {
                model_id: m,
                layer_range: (0, 24),
                reads: 151_936,
            })
        };
        let only = |have: &'static [&'static str]| move |m: &ModelId| have.contains(&m.0.as_str());
        assert_eq!(
            acquisition_among(
                likely(),
                only(&[]),
                || Some(360),
                "",
                DRAFTER_MAX_FRACTION,
                None,
                size,
                can_draft
            ),
            DrafterStep::LearnHeaders(vec![id("3b"), id("1.5b")]),
            "the largest two first"
        );
        assert_eq!(
            acquisition_among(
                likely(),
                only(&["3b", "1.5b"]),
                || Some(360),
                "",
                DRAFTER_MAX_FRACTION,
                None,
                size,
                can_draft
            ),
            DrafterStep::LearnHeaders(vec![id("0.5b"), id("other-vocab")]),
            "a later header could still name a better one: not yet"
        );
        let every = only(&["3b", "1.5b", "0.5b", "other-vocab"]);
        assert_eq!(
            acquisition_among(
                likely(),
                every,
                || Some(360),
                "",
                DRAFTER_MAX_FRACTION,
                None,
                size,
                can_draft
            ),
            DrafterStep::Fetch(id("1.5b")),
            "the 3B is more than half; the other vocabulary cannot draft"
        );
        assert_eq!(
            acquisition_among(
                likely(),
                every,
                || None,
                "",
                DRAFTER_MAX_FRACTION,
                None,
                size,
                can_draft
            ),
            DrafterStep::Nothing,
            "no header for the target: no size to judge against"
        );
        assert_eq!(
            acquisition_among(
                vec![id("3b")],
                only(&["3b"]),
                || Some(360),
                "",
                DRAFTER_MAX_FRACTION,
                None,
                size,
                can_draft
            ),
            DrafterStep::Nothing
        );
    }

    /// A node holding a drafter fetches a LARGER one that can draft (#331): a
    /// 0.5B held, the 1.5B is fetched; a 1.5B held, nothing — neither another
    /// copy of the same size (a Q4 of the 1.5B reads fewer bytes a guess, which
    /// a flow turns into waiting) nor the 3B, past the size rule. Every likely
    /// header is still learned first.
    #[test]
    fn a_node_holding_a_drafter_fetches_only_a_larger_one() {
        let id = |s: &str| ModelId(s.into());
        let likely = || vec![id("3b"), id("1.5b-q8"), id("1.5b-q4")];
        let size = |m: &ModelId| {
            let (size, reads) = match m.0.as_str() {
                "3b" => (264, 2_000),
                "1.5b-q8" => (66, 1_600),
                "1.5b-q4" => (66, 900),
                _ => return None,
            };
            Some(DrafterSize { size, reads })
        };
        let can_draft = |m: ModelId| {
            Some(DrafterSpec {
                model_id: m,
                layer_range: (0, 28),
                reads: 151_936,
            })
        };
        let headers =
            |have: &'static [&'static str]| move |m: &ModelId| have.contains(&m.0.as_str());
        let every = headers(&["3b", "1.5b-q8", "1.5b-q4"]);
        let held_0_5b = Some(HeldDrafter {
            size: 19,
            of_the_targets_line: true,
        });
        assert_eq!(
            acquisition_among(
                likely(),
                headers(&["3b"]),
                || Some(360),
                "",
                4,
                held_0_5b,
                size,
                can_draft
            ),
            DrafterStep::LearnHeaders(vec![id("1.5b-q8"), id("1.5b-q4")]),
            "a held drafter does not skip the headers"
        );
        assert_eq!(
            acquisition_among(
                likely(),
                every,
                || Some(360),
                "",
                4,
                held_0_5b,
                size,
                can_draft
            ),
            DrafterStep::Fetch(id("1.5b-q4")),
            "larger than the 0.5B held; of its two copies the one reading less"
        );
        assert_eq!(
            acquisition_among(
                likely(),
                every,
                || Some(360),
                "",
                4,
                Some(HeldDrafter {
                    size: 66,
                    of_the_targets_line: true
                }),
                size,
                can_draft
            ),
            DrafterStep::Nothing,
            "a 1.5B held: a copy of the same size is never worth a fetch"
        );
        assert_eq!(
            acquisition_among(likely(), every, || Some(360), "", 4, None, size, can_draft),
            DrafterStep::Fetch(id("1.5b-q4")),
            "control: nothing held, the same choice"
        );
    }

    /// The held drafter is kept on every answer, and so is the one being
    /// fetched — the replacement arrives beside the drafter it replaces.
    #[test]
    fn an_answer_keeps_the_drafter_held_while_a_better_one_comes() {
        let id = |s: &str| ModelId(s.into());
        let answer = |next| DrafterToAcquire {
            held: Some(id("0.5b")),
            next,
        };
        for next in [
            DrafterStep::LearnHeaders(vec![id("1.5b")]),
            DrafterStep::Fetch(id("1.5b")),
            DrafterStep::Nothing,
        ] {
            assert!(answer(next.clone()).keeps(&id("0.5b")), "{next:?}");
        }
        let fetching = answer(DrafterStep::Fetch(id("1.5b")));
        assert!(fetching.keeps(&id("1.5b")));
        assert_eq!(fetching.fetching(), Some(&id("1.5b")));
        assert!(!fetching.keeps(&id("3b")));
        let learning = answer(DrafterStep::LearnHeaders(vec![id("3b")]));
        assert!(
            !learning.keeps(&id("3b")),
            "a header being learned keeps no parts"
        );
        assert_eq!(learning.headers_to_learn(), &[id("3b")]);
        let replaced = DrafterToAcquire {
            held: Some(id("1.5b")),
            next: DrafterStep::Nothing,
        };
        assert!(
            !replaced.keeps(&id("0.5b")),
            "once the 1.5B is held and chosen, the 0.5B is no longer the drafter's"
        );
    }
}
