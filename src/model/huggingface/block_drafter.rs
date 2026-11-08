//! A block drafter (#428) fetched for a model this node can run from its first
//! layer to its output head — the only place one is attached
//! (`SplitModel::attach_block_drafter`: a whole model, or a split's first
//! segment beside the head's part). Without it every split of Qwen 3.5/3.6
//! decoded one token a round trip (#427); with it the 9B split across 130 ms
//! wrote code 1.6-1.8x and a story 1.10-1.15x faster (#428 stage 3).
//!
//! Prior art: no runtime pairs a drafter with its model on its own —
//! llama.cpp takes `-md`, LM Studio has the user pick one of the same
//! vocabulary, SGLang `--speculative-draft-model-path` (searched 2026-11-08).
//! So the pairing is ours, and it is a PIN, never a search: each entry names
//! its target by the header's `general.name` and width, and the file by its
//! repository, size and SHA-256 as HuggingFace lists them — a byte that
//! differs is refused (Ollama's `verifyBlob`, #422's rule). A wrong pairing
//! costs nothing past the download: the loader refuses a drafter of another
//! width, layer count or token table, and a drafter that guesses badly is set
//! aside by what it writes (`arch-scheduling.md` § "Speculation that does not
//! pay").

use crate::daemon::state::SharedState;
use crate::types::ModelId;

/// A drafter this build fetches for one target.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PinnedDrafter {
    /// The target's `general.name`, compared [`folded`] — every upload of the
    /// model writes it, a fine-tune writes its own.
    pub target_name: &'static str,
    /// The target's `general.architecture`.
    pub target_architecture: &'static str,
    /// The target's `embedding_length` — the drafter's own, by construction.
    pub target_width: usize,
    pub repo: &'static str,
    pub file: &'static str,
    pub size_bytes: u64,
    /// HuggingFace's listed SHA-256, hex.
    pub sha256: &'static str,
}

/// The most a drafter may be: a user holds small things (the split drafter's
/// `MOST_DRAFTER_BYTES_TO_FETCH`, 2 GiB).
pub(crate) const MOST_DRAFTER_BYTES: u64 = 2 << 30;

/// DFlash drafters for the models people run (#427's ranking), Q8_0, one
/// publisher's llama.cpp conversions of z-lab's weights (Anbeeld, read
/// 2026-11-08: each header checked — width, target layers inside the model,
/// the mask token, no tensor this loader lacks). Qwen3.8-27B's drafter is
/// DFlash 2 (dynamic convolutions, a selector), which this loader refuses.
pub(crate) const PINNED: &[PinnedDrafter] = &[
    PinnedDrafter {
        target_name: "Qwen3.6 35B A3B",
        target_architecture: "qwen35moe",
        target_width: 2048,
        repo: "Anbeeld/Qwen3.6-35B-A3B-DFlash-GGUF",
        file: "qwen36-35b-a3b-dflash-Q8_0.gguf",
        size_bytes: 421_060_320,
        sha256: "b9749d1793b8c90706168ec6b7cfedee74d3c76e67c4c61cf477e08ea478761c",
    },
    PinnedDrafter {
        target_name: "Qwen3.6 27B",
        target_architecture: "qwen35",
        target_width: 5120,
        repo: "Anbeeld/Qwen3.6-27B-DFlash-GGUF",
        file: "Qwen3.6-27B-DFlash-Q8_0.gguf",
        size_bytes: 1_849_481_312,
        sha256: "1a13e32db05bac10ce6d060370f5c5006178f19c60955ab619c669bfb67c760d",
    },
    PinnedDrafter {
        target_name: "Qwen3.5 9B",
        target_architecture: "qwen35",
        target_width: 4096,
        repo: "Anbeeld/Qwen3.5-9B-DFlash-GGUF",
        file: "qwen35-9b-dflash-Q8_0.gguf",
        size_bytes: 1_383_767_744,
        sha256: "a745a9cbad9a9869954643aa24fb800068003c452c5b85c91958b8008aa44957",
    },
    PinnedDrafter {
        target_name: "Qwen3.5 4B",
        target_architecture: "qwen35",
        target_width: 2560,
        repo: "Anbeeld/Qwen3.5-4B-DFlash-GGUF",
        file: "qwen35-4b-dflash-Q8_0.gguf",
        size_bytes: 685_133_504,
        sha256: "9b6ffdd36bd9cb88a694461fe0aef68f6983ea1c3a79d63e0da552b326013a02",
    },
    PinnedDrafter {
        target_name: "Qwen3.5 35B A3B",
        target_architecture: "qwen35moe",
        target_width: 2048,
        repo: "Anbeeld/Qwen3.5-35B-A3B-DFlash-GGUF",
        file: "Qwen3.5-35B-A3B-DFlash-Q8_0.gguf",
        size_bytes: 421_060_928,
        sha256: "137520fa4fcd6964d0683189638dc1389bd9f26ddeaac66e642ea439c4bb2029",
    },
    PinnedDrafter {
        target_name: "Qwen3.5 122B A10B",
        target_architecture: "qwen35moe",
        target_width: 3072,
        repo: "Anbeeld/Qwen3.5-122B-A10B-DFlash-GGUF",
        file: "Qwen3.5-122B-A10B-DFlash-Q8_0.gguf",
        size_bytes: 833_339_712,
        sha256: "c5a380222670a4f3a7903eba4e967fb4e97e50415c306fa40bdd621fca3d9d17",
    },
    PinnedDrafter {
        target_name: "Qwen3.5 397B A17B",
        target_architecture: "qwen35moe",
        target_width: 4096,
        repo: "Anbeeld/Qwen3.5-397B-A17B-DFlash-GGUF",
        file: "Qwen3.5-397B-A17B-DFlash-Q8_0.gguf",
        size_bytes: 1_383_768_384,
        sha256: "8e62fe5f759646fa0d3d693b9c99793090791f2b66b4328af047ed013e3cd270",
    },
];

/// A name with case, spaces and punctuation taken out: "Qwen3.5-9B" and
/// "Qwen3.5 9B" are one model's two uploads.
fn folded(name: &str) -> String {
    name.chars()
        .filter(char::is_ascii_alphanumeric)
        .map(|c| c.to_ascii_lowercase())
        .collect()
}

/// The drafter pinned for a model with this header, if any.
pub(crate) fn pinned_for(
    name: Option<&str>,
    architecture: &str,
    width: usize,
) -> Option<&'static PinnedDrafter> {
    let name = folded(name?);
    PINNED.iter().find(|p| {
        folded(p.target_name) == name
            && p.target_architecture == architecture
            && p.target_width == width
    })
}

/// Models whose drafter fetch failed lately, so the download pass, which runs
/// every few minutes, tries again only after [`TRY_AGAIN_AFTER`].
static FAILED_LATELY: std::sync::LazyLock<
    std::sync::Mutex<std::collections::HashMap<ModelId, std::time::Instant>>,
> = std::sync::LazyLock::new(Default::default);

const TRY_AGAIN_AFTER: std::time::Duration = std::time::Duration::from_secs(6 * 3600);

/// One fetch at a time for the whole node.
static FETCHING: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

struct FetchSlot;

impl FetchSlot {
    fn take() -> Option<Self> {
        FETCHING
            .compare_exchange(
                false,
                true,
                std::sync::atomic::Ordering::AcqRel,
                std::sync::atomic::Ordering::Acquire,
            )
            .is_ok()
            .then_some(FetchSlot)
    }
}

impl Drop for FetchSlot {
    fn drop(&mut self) {
        FETCHING.store(false, std::sync::atomic::Ordering::Release);
    }
}

/// What this node should do about one model's drafter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Step {
    Fetch(&'static PinnedDrafter),
    /// A drafter on disk beside a model whose ends left: nothing can attach it.
    GiveBack,
    Nothing,
}

fn step_for(
    pinned: Option<&'static PinnedDrafter>,
    ends: Ends,
    on_disk: bool,
    failed_lately: bool,
) -> Step {
    match (pinned, ends, on_disk) {
        (_, Ends::Gone, true) => Step::GiveBack,
        (Some(p), Ends::Held, false) if !failed_lately => Step::Fetch(p),
        _ => Step::Nothing,
    }
}

/// Where this node stands on the parts a drafter attaches beside: part 0 (the
/// token table) and the head's part (the output head it reads its guesses
/// with, `loader::shards::drafter_head_info`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Ends {
    /// Both whole and checked here, neither being fetched again.
    Held,
    /// Neither has landed here: nothing could attach a drafter.
    Gone,
    /// One of them, one being fetched again, or no manifest to say.
    Unsure,
}

fn ends_here(state: &SharedState, model_id: &ModelId) -> Ends {
    let Some(manifest) = state.model_registry.get_manifest(model_id) else {
        return Ends::Unsure;
    };
    let head = crate::inference::shard_layout::head_part(manifest.shard_count);
    let servable = state.parts_servable_here(&manifest);
    if servable.contains(&0) && servable.contains(&head) {
        return Ends::Held;
    }
    let landed = state.parts_landed_here(&manifest);
    if !landed.contains(&0) && !landed.contains(&head) {
        Ends::Gone
    } else {
        Ends::Unsure
    }
}

/// One download-pass step: fetch the pinned drafter of every model this node
/// holds both ends of, one at a time, and give back a drafter whose model's
/// ends have gone. A download nobody asked for: it waits on a metered
/// connection, and it leaves the disk's reserve free.
pub(crate) async fn fetch_for_models_held_end_to_end(state: &std::sync::Arc<SharedState>) {
    if !crate::inference::split::block_drafter_on() {
        return;
    }
    let Some(_slot) = FetchSlot::take() else {
        return;
    };
    for model_id in crate::model::auto_manage::canonical::models_kept_on_disk(state) {
        let dir = state.model_dir(&model_id.0);
        let path = dir.join(crate::inference::split::BLOCK_DRAFTER_FILE);
        let pinned = state
            .gguf_meta_for(&model_id)
            .and_then(|m| pinned_for(m.model_name.as_deref(), &m.architecture, m.embedding_length));
        let failed_lately = FAILED_LATELY
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(&model_id)
            .is_some_and(|at| at.elapsed() < TRY_AGAIN_AFTER);
        match step_for(
            pinned,
            ends_here(state, &model_id),
            path.is_file(),
            failed_lately,
        ) {
            Step::Nothing => {}
            Step::GiveBack => {
                if state.model_is_in_use(&model_id) {
                    continue;
                }
                match std::fs::remove_file(&path) {
                    Ok(()) => tracing::info!(model = %model_id,
                        "Gave back the model's guesser: this computer no longer holds both its ends"),
                    Err(e) => tracing::debug!(model = %model_id, error = %e,
                        "Could not remove the model's guesser"),
                }
            }
            Step::Fetch(p) => {
                if state.metered_hold().await {
                    return;
                }
                let Some(disk) = crate::model::auto_manage::disk_space_for(&dir) else {
                    continue;
                };
                if disk.free_bytes < p.size_bytes.saturating_add(disk.reserve_bytes()) {
                    tracing::debug!(model = %model_id, "No room on disk for the model's guesser");
                    continue;
                }
                let Ok(sha256) = hex::decode(p.sha256)
                    .ok()
                    .and_then(|b| <[u8; 32]>::try_from(b).ok())
                    .ok_or(())
                else {
                    continue;
                };
                tracing::info!(model = %model_id, repo = p.repo, file = p.file,
                    mb = p.size_bytes >> 20, "Fetching a guesser for the model (#428)");
                match super::download::download_listed_file(
                    p.repo,
                    p.file,
                    p.size_bytes,
                    &sha256,
                    MOST_DRAFTER_BYTES,
                    &dir,
                    crate::inference::split::BLOCK_DRAFTER_FILE,
                )
                .await
                {
                    Ok(_) => tracing::info!(model = %model_id,
                        "Fetched the model's guesser — it is used from the model's next load"),
                    Err(e) => {
                        tracing::warn!(model = %model_id, repo = p.repo, error = %e,
                            "Could not fetch the model's guesser");
                        FAILED_LATELY
                            .lock()
                            .unwrap_or_else(|e| e.into_inner())
                            .insert(model_id.clone(), std::time::Instant::now());
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every pin names a file of a checkable size and a whole SHA-256, under
    /// the size a user is asked to hold, and no two pins claim one target.
    #[test]
    fn every_pin_is_checkable_and_names_one_target() {
        for p in PINNED {
            assert!(
                p.size_bytes > 0 && p.size_bytes <= MOST_DRAFTER_BYTES,
                "{p:?}"
            );
            let sha = hex::decode(p.sha256).expect("hex");
            assert_eq!(sha.len(), 32, "{p:?}");
            assert!(p.file.ends_with(".gguf"), "{p:?}");
            let same = PINNED
                .iter()
                .filter(|q| {
                    folded(q.target_name) == folded(p.target_name)
                        && q.target_architecture == p.target_architecture
                })
                .count();
            assert_eq!(same, 1, "{p:?}");
        }
    }

    /// Two uploads' spellings of one model find its drafter; a fine-tune, a
    /// model of another width or architecture finds none.
    #[test]
    fn a_drafter_is_found_by_the_targets_name_and_shape() {
        let p = pinned_for(Some("Qwen3.5-9B"), "qwen35", 4096).expect("unsloth's 9B");
        assert_eq!(p.repo, "Anbeeld/Qwen3.5-9B-DFlash-GGUF");
        assert_eq!(pinned_for(Some("Qwen3.5 9B"), "qwen35", 4096), Some(p));
        let moe = pinned_for(Some("Qwen3.6 35B A3B"), "qwen35moe", 2048).expect("bartowski's");
        assert_eq!(moe.size_bytes, 421_060_320);
        assert_eq!(pinned_for(Some("Ornith-1.5-35B"), "qwen35moe", 2048), None);
        assert_eq!(pinned_for(Some("Qwen3.6 35B A3B"), "qwen35", 2048), None);
        assert_eq!(pinned_for(Some("Qwen3.5 9B"), "qwen35", 2560), None);
        assert_eq!(pinned_for(None, "qwen35", 4096), None);
        assert_eq!(pinned_for(Some("Qwen3.8 27B"), "qwen35", 5120), None);
    }

    #[test]
    fn a_drafter_is_fetched_only_beside_both_ends_and_given_back_when_they_go() {
        let p = &PINNED[0];
        assert_eq!(step_for(Some(p), Ends::Held, false, false), Step::Fetch(p));
        assert_eq!(step_for(Some(p), Ends::Held, false, true), Step::Nothing);
        assert_eq!(step_for(Some(p), Ends::Held, true, false), Step::Nothing);
        assert_eq!(step_for(Some(p), Ends::Unsure, false, false), Step::Nothing);
        assert_eq!(step_for(Some(p), Ends::Gone, false, false), Step::Nothing);
        // A part being fetched again, or no manifest yet, gives nothing back.
        assert_eq!(step_for(Some(p), Ends::Unsure, true, false), Step::Nothing);
        assert_eq!(step_for(Some(p), Ends::Gone, true, false), Step::GiveBack);
        assert_eq!(step_for(None, Ends::Gone, true, false), Step::GiveBack);
        assert_eq!(step_for(None, Ends::Held, false, false), Step::Nothing);
    }
}
