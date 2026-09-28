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

/// A model's size for choosing a drafter: layers × width², the part of its
/// parameter count that differs between siblings (Qwen2.5-0.5B 19 M against
/// Qwen2.5-Coder-7B's 360 M; Llama-3.2-1B 67 M against Llama-3.1-8B's 537 M).
/// Bytes on disk would compare an fp16 small model with a 4-bit large one.
fn size_proxy(state: &SharedState, model: &ModelId) -> Option<u64> {
    let meta = state.gguf_meta_for(model)?;
    Some(meta.block_count as u64 * (meta.embedding_length as u64).pow(2))
}

/// The largest a drafter may be, as a fraction of its target: past a quarter
/// its guesses cost too much of what they save (Leviathan et al. 2023 used
/// drafters ~1/10-1/20 of the target; the γ controller then sizes the run from
/// the measured cost of each guess).
const DRAFTER_MAX_FRACTION: u64 = 4;

/// `model` as a drafter for `target`, if it can be one: held WHOLE here, not
/// the target, no recurrent state (a guess that is refused must be taken back,
/// and a DeltaNet layer's state cannot be), and the target's vocabulary.
fn spec_for(state: &SharedState, target: &ModelId, drafter: ModelId) -> Option<DrafterSpec> {
    if drafter == *target || !state.has_complete_split_model(&drafter) {
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
/// ([`spec_for`]) and is at most a quarter of its size ([`size_proxy`]) — the
/// best guesser that is still cheap, since across a network the round trip,
/// not the guessing, is what a round mostly costs. `None` when there is none:
/// speculation across computers then drafts with llama.cpp where that is
/// configured, or not at all.
pub(crate) fn engine_drafter_for(state: &SharedState, target: &ModelId) -> Option<DrafterSpec> {
    if let Some(named) = state.cfg().inference.draft_model.clone() {
        return spec_for(state, target, ModelId(named));
    }
    let target_size = size_proxy(state, target)?;
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
    let sized: Vec<(u64, ModelId)> = held
        .into_iter()
        .filter_map(|m| size_proxy(state, &m).map(|size| (size, m)))
        .collect();
    largest_that_can_draft(target_size, sized, |m| spec_for(state, target, m))
}

/// Of `candidates` (size proxy, model), the largest no bigger than a
/// [`DRAFTER_MAX_FRACTION`] of `target_size` that `can_draft` accepts. Ties go
/// to the model id, so the choice does not depend on map order.
fn largest_that_can_draft(
    target_size: u64,
    mut candidates: Vec<(u64, ModelId)>,
    mut can_draft: impl FnMut(ModelId) -> Option<DrafterSpec>,
) -> Option<DrafterSpec> {
    candidates.retain(|(size, _)| size.saturating_mul(DRAFTER_MAX_FRACTION) <= target_size);
    candidates.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1 .0.cmp(&b.1 .0)));
    candidates.into_iter().find_map(|(_, m)| can_draft(m))
}

/// Have the drafter's worker read the prompt NOW — spawned beside the target's
/// own prompt pass, which runs over the network, so the drafter's read (and, on
/// a request's first use, its worker's spawn and load: 4.4 s measured for
/// qwen2.5-0.5b on an RTX 3070) happens while the reply could not have started
/// anyway, instead of after it. A guess-free call (`gamma` 0) that leaves the
/// prompt in the worker's cache; on success the caller marks it read with
/// [`EngineDrafter::prompt_read`], on failure the first round reads it itself.
/// `key` is the attempt's own ([`draft_key`]), shared with its [`EngineDrafter`].
pub(super) fn read_ahead(
    state: std::sync::Arc<SharedState>,
    spec: &DrafterSpec,
    key: uuid::Uuid,
    prompt_ids: Vec<u32>,
    sampling: &SamplingParams,
    cancel: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
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
        for_the_owner: true,
        stop_below: None,
    };
    let release = release_of(state.clone(), key);
    ReadAhead::new(
        tokio::spawn(async move { state.model_process_pool.draft(d, cancel).await }),
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
    calls: u32,
}

/// The probability below which a drafter stops guessing for the round
/// (`SplitModel::draft_after`): 0.4, Hugging Face's assisted-generation choice
/// for the same rule. `SWARMLLM_DRAFT_CONFIDENCE=<p>` sets it for an A/B inside
/// one binary; 0 guesses every token the round asks for. Read once.
fn draft_confidence_floor() -> Option<f32> {
    static F: OnceLock<Option<f32>> = OnceLock::new();
    *F.get_or_init(|| {
        let p = std::env::var("SWARMLLM_DRAFT_CONFIDENCE")
            .ok()
            .and_then(|v| v.trim().parse::<f32>().ok())
            .unwrap_or(0.4);
        (p > 0.0).then_some(p)
    })
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
    /// shared with its [`read_ahead`].
    pub(super) fn new(
        spec: DrafterSpec,
        key: uuid::Uuid,
        prompt_ids: Vec<u32>,
        release: Release,
    ) -> Self {
        Self {
            spec,
            key,
            release: Some(release),
            seq: prompt_ids,
            valid: 0,
            open: None,
            calls: 0,
        }
    }

    pub(super) fn model_id(&self) -> &ModelId {
        &self.spec.model_id
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
    pub(super) async fn draft(
        &mut self,
        state: &SharedState,
        gamma: u32,
        sampling: &SamplingParams,
        history: &[u32],
        coupling_seed: Option<u64>,
        cancel: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
    ) -> Result<Vec<u32>, SwarmError> {
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
            for_the_owner: true,
            stop_below: draft_confidence_floor(),
        };
        let guesses = state.model_process_pool.draft(d, cancel).await?;
        self.open = Some((self.seq.len(), guesses.len()));
        Ok(guesses)
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

    fn spec(name: &str) -> DrafterSpec {
        DrafterSpec {
            model_id: ModelId(name.into()),
            layer_range: (0, 1),
            reads: 1,
        }
    }

    /// For Qwen2.5-Coder-7B (360 M by layers × width²) this node holds the 0.5B
    /// (19 M), 1.5B (66 M) and 3B (151 M): the 3B is past a quarter, so the 1.5B
    /// — and the 0.5B when the 1.5B cannot draft.
    #[test]
    fn the_largest_drafter_within_a_quarter_that_can_draft_is_chosen() {
        let held = || {
            vec![
                (19, ModelId("q-0.5b".into())),
                (151, ModelId("q-3b".into())),
                (66, ModelId("q-1.5b".into())),
            ]
        };
        let chosen = largest_that_can_draft(360, held(), |m| Some(spec(&m.0)));
        assert_eq!(chosen.unwrap().model_id.0, "q-1.5b");
        let chosen = largest_that_can_draft(360, held(), |m| (m.0 != "q-1.5b").then(|| spec(&m.0)));
        assert_eq!(chosen.unwrap().model_id.0, "q-0.5b");
        assert!(largest_that_can_draft(360, held(), |_| None).is_none());
        assert!(
            largest_that_can_draft(40, held(), |m| Some(spec(&m.0))).is_none(),
            "nothing a quarter of a tiny target's size"
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
}
