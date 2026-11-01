//! What people run, read from HuggingFace's model list (#389 (b)).
//!
//! The swarm's models follow what people outside it run, read on a schedule —
//! never a list that changes only at a release. Two things make the raw list
//! wrong (2026-11-01): HuggingFace counts every GET of a weight file
//! (`docs/hub/models-download-stats`), so automated pulls of small embedding
//! and test models top a plain `library=gguf` list (the watcher's trending
//! feed); and one base model's popularity is spread over many uploads of it.
//! So the list is narrowed to the families this build loads, summed PER BASE
//! MODEL (`base_model:quantized:<repo>`), cut to a few per size of computer,
//! and the swarm holds ONE upload of each — the one `model::canonical` would
//! choose (the trusted publishers' order), never simply the most downloaded.
//! A fine-tune is a model of its own with its own count. Nothing here judges
//! content (#389 (c)).
//!
//! The admission rule this feeds is TinyLFU's (Einziger et al., ACM TOS 2017):
//! a newcomer displaces a holding only with a higher estimated frequency — the
//! frequency read from outside the swarm while the swarm is too small for its
//! own counts to mean much.

use serde::{Deserialize, Serialize};

/// GGUF repos, most downloaded over 30 days first, with each repo's GGUF facts
/// (`gguf.architecture`, `gguf.total` = parameters). 4.7 MB for 500 on
/// 2026-11-01: the facts carry each repo's chat template.
pub(crate) const HF_POPULAR_URL: &str = "https://huggingface.co/api/models?filter=gguf\
     &sort=downloads&direction=-1&limit=500&expand[]=gguf&expand[]=downloads\
     &expand[]=likes&expand[]=pipeline_tag&expand[]=tags&expand[]=createdAt";

/// The most of the list read (it measured 4.7 MB).
pub(crate) const MOST_LIST_BYTES: usize = 16 * 1024 * 1024;

/// Bases kept per size of computer. 2026-11-01: Qwen 3.5/3.6/3.8 and Gemma 4
/// filled the top three of every tier, which is what r/LocalLLaMA's 2026
/// threads name as what people run.
pub(crate) const BASES_PER_TIER: usize = 3;

/// The size tiers, by the bytes a Q4_K_M copy takes: what a 4, 8, 16 and
/// 24-32 GB card (or that much memory) runs whole, and beyond — split.
pub(crate) const TIER_CEILINGS_BYTES: [u64; 4] = [
    3 * 1024 * 1024 * 1024,
    6 * 1024 * 1024 * 1024,
    12 * 1024 * 1024 * 1024,
    24 * 1024 * 1024 * 1024,
];

/// A Q4_K_M copy's average bits per weight (llama.cpp's mix: 4.85).
const Q4_K_M_BITS_PER_WEIGHT: f64 = 4.85;

/// At or under this many parameters a model is held at Q8_0 — a drafter's
/// size, where Q4 costs the most quality for the least room.
pub(crate) const Q8_AT_MOST_PARAMS: u64 = 2_000_000_000;

/// Uploads of one base tried for a file before the base is passed over.
pub(crate) const UPLOADS_TRIED_PER_BASE: usize = 3;

/// A snapshot older than this is not acted on (`PopularSnapshot::fresh`): a
/// node that cannot reach HuggingFace gives nothing back for being off a list
/// it cannot read. Refreshed daily, so three missed days.
pub(crate) const MOST_SNAPSHOT_AGE_SECS: i64 = 3 * 24 * 3600;

/// How often the list is read.
pub(crate) const REFRESH_EVERY_SECS: i64 = 24 * 3600;

/// One repo of the list — only the fields used.
#[derive(Clone, Debug, Default, Deserialize)]
pub(crate) struct ListedRepo {
    #[serde(rename = "id")]
    pub repo_id: String,
    #[serde(default)]
    pub downloads: u64,
    #[serde(rename = "pipeline_tag", default)]
    pub pipeline_tag: Option<String>,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(rename = "createdAt", default)]
    pub created_at: Option<String>,
    #[serde(default)]
    pub gguf: Option<ListedGguf>,
}

#[derive(Clone, Debug, Default, Deserialize)]
pub(crate) struct ListedGguf {
    #[serde(default)]
    pub architecture: Option<String>,
    /// Parameters.
    #[serde(default)]
    pub total: Option<u64>,
}

impl ListedRepo {
    fn created_at_secs(&self) -> i64 {
        self.created_at
            .as_deref()
            .and_then(|s| chrono::DateTime::parse_from_rfc3339(s).ok())
            .map(|dt| dt.timestamp())
            .unwrap_or(0)
    }
}

/// One base model people run, and the upload of it the swarm holds.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct PopularBase {
    /// The base model's repo, lower-case (`base_model:quantized:<repo>`; the
    /// upload itself where it names none).
    pub base: String,
    /// 30-day downloads summed over every upload of it in the list.
    pub downloads: u64,
    pub architecture: String,
    pub parameters: u64,
    /// Index into `TIER_CEILINGS_BYTES`, 4 = beyond the last.
    pub tier: u8,
    /// Uploads of it, best first by `upload_rank` — the first one with a
    /// file the swarm can hold is the one used (`PopularSnapshot`).
    pub uploads: Vec<String>,
    /// The upload and file the swarm holds, and the model id that file gives
    /// (`gguf_filename_to_model_id`). Empty until resolved: the list does not
    /// carry file names.
    #[serde(default)]
    pub repo_id: String,
    #[serde(default)]
    pub filename: String,
    #[serde(default)]
    pub model_id: String,
}

/// What people run, as last read.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct PopularSnapshot {
    pub bases: Vec<PopularBase>,
    /// Unix seconds of the read (0 = never).
    pub fetched_at: i64,
}

impl PopularSnapshot {
    /// Read recently enough to act on (`MOST_SNAPSHOT_AGE_SECS`).
    pub fn fresh(&self, now: i64) -> bool {
        self.fetched_at > 0 && now.saturating_sub(self.fetched_at) <= MOST_SNAPSHOT_AGE_SECS
    }

    /// Is `model_id` the copy the swarm holds of a base people run?
    pub fn names_model(&self, model_id: &str) -> bool {
        self.bases.iter().any(|b| b.model_id == model_id)
    }

    /// Due for another read.
    pub fn due(&self, now: i64) -> bool {
        now.saturating_sub(self.fetched_at) >= REFRESH_EVERY_SECS
    }
}

/// Pipeline tags of models that are not chat models whatever their family:
/// a Qwen 3.5 or Gemma 4 upload is often `image-text-to-text` or
/// `any-to-any` and is kept; a reranker, a speech or a video model built on
/// a supported family is not.
const NOT_CHAT_PIPELINES: &[&str] = &[
    "text-classification",
    "token-classification",
    "text-ranking",
    "feature-extraction",
    "sentence-similarity",
    "text-to-speech",
    "automatic-speech-recognition",
    "text-to-audio",
    "text-to-image",
    "text-to-video",
    "image-to-image",
    "image-to-video",
    "fill-mask",
    "zero-shot-classification",
];

/// The base model an upload was made from: `base_model:quantized:<repo>`, else
/// a plain `base_model:<repo>` (a re-upload), else the upload itself.
/// `base_model:finetune:` / `:merge:` / `:adapter:` name what a model was BUILT
/// from — a different model, never its base here.
pub(crate) fn base_of(repo: &ListedRepo) -> String {
    let quantized = repo
        .tags
        .iter()
        .find_map(|t| t.strip_prefix("base_model:quantized:"));
    let plain = || {
        repo.tags
            .iter()
            .filter_map(|t| t.strip_prefix("base_model:"))
            .find(|rest| !rest.contains(':'))
    };
    quantized
        .or_else(plain)
        .unwrap_or(&repo.repo_id)
        .to_ascii_lowercase()
}

/// The size tier of a model of `parameters` held at Q4_K_M.
pub(crate) fn tier_of(parameters: u64) -> u8 {
    let bytes = (parameters as f64 * Q4_K_M_BITS_PER_WEIGHT / 8.0) as u64;
    TIER_CEILINGS_BYTES
        .iter()
        .position(|&ceiling| bytes <= ceiling)
        .unwrap_or(TIER_CEILINGS_BYTES.len()) as u8
}

/// The order uploads of one base are tried in — `model::canonical`'s: a held
/// upload first (the swarm keeps the copy it has), then the trusted
/// publishers' order (`TRUSTED_HF_PUBLISHERS`, a swarm-wide contract), then
/// the repo name, so every node picks the same one.
fn upload_rank(repo_id: &str, held: &dyn Fn(&str) -> bool) -> (bool, usize, String) {
    (
        !held(repo_id),
        super::watcher::trusted_publisher_position(repo_id).unwrap_or(usize::MAX),
        repo_id.to_ascii_lowercase(),
    )
}

/// The bases people run, `BASES_PER_TIER` per size tier, from the list.
///
/// A repo counts toward its base's downloads when its family loads here
/// (`supported`), it is a chat model (`NOT_CHAT_PIPELINES`, never an
/// embedding repo) and its id is well-formed. It can be HELD only when it
/// clears the adoption floor on its OWN downloads (`min_downloads_for_repo`)
/// and is a day old — the watcher's anti-pump rule — so a pumped upload can
/// add to a base's count but never become the swarm's copy of it.
pub(crate) fn choose(
    listed: &[ListedRepo],
    supported: &dyn Fn(&str) -> bool,
    held: &dyn Fn(&str) -> bool,
    now_secs: i64,
) -> Vec<PopularBase> {
    let mut by_base: std::collections::BTreeMap<String, PopularBase> = Default::default();
    for repo in listed {
        let Some(gguf) = repo.gguf.as_ref() else {
            continue;
        };
        let (Some(arch), Some(parameters)) = (gguf.architecture.as_deref(), gguf.total) else {
            continue;
        };
        if !supported(arch)
            || parameters == 0
            || repo
                .pipeline_tag
                .as_deref()
                .is_some_and(|p| NOT_CHAT_PIPELINES.contains(&p))
            || super::is_embedding_repo(repo.pipeline_tag.as_deref(), &repo.tags)
            || super::validate_hf_repo_id(&repo.repo_id).is_err()
        {
            continue;
        }
        let base = base_of(repo);
        let entry = by_base.entry(base.clone()).or_insert_with(|| PopularBase {
            base,
            architecture: arch.to_string(),
            parameters,
            tier: tier_of(parameters),
            ..Default::default()
        });
        entry.downloads = entry.downloads.saturating_add(repo.downloads);
        let adoptable = repo.downloads >= super::watcher::min_downloads_for_repo(&repo.repo_id)
            && repo.created_at_secs() > 0
            && now_secs - repo.created_at_secs() >= 24 * 3600;
        if adoptable {
            entry.uploads.push(repo.repo_id.clone());
        }
    }
    let mut bases: Vec<PopularBase> = by_base
        .into_values()
        .filter(|b| !b.uploads.is_empty())
        .collect();
    for base in &mut bases {
        base.uploads.sort_by_cached_key(|r| upload_rank(r, held));
        base.uploads.truncate(UPLOADS_TRIED_PER_BASE);
    }
    bases.sort_by(|a, b| {
        b.downloads
            .cmp(&a.downloads)
            .then_with(|| a.base.cmp(&b.base))
    });
    let mut per_tier = [0usize; TIER_CEILINGS_BYTES.len() + 1];
    bases
        .into_iter()
        .filter(|b| {
            let n = &mut per_tier[b.tier as usize];
            *n += 1;
            *n <= BASES_PER_TIER
        })
        .collect()
}

/// The file of an upload the swarm holds: Q4_K_M, or Q8_0 for a model of
/// `Q8_AT_MOST_PARAMS` or fewer — one quantisation per model, so the 0.5B is
/// not held four times (#389 (e)). Never a projector (`mmproj`), a multi-token
/// prediction head (`mtp-`), one part of a split upload, or a file in a folder
/// — the download path takes a bare file name only (`is_valid_hf_filename`:
/// letters, digits, `-`, `_`, `.`); of several matches, the shortest name.
pub(crate) fn pick_file<'a>(
    filenames: impl IntoIterator<Item = &'a str>,
    parameters: u64,
) -> Option<String> {
    let wanted = if parameters <= Q8_AT_MOST_PARAMS {
        "Q8_0"
    } else {
        "Q4_K_M"
    };
    filenames
        .into_iter()
        .filter(|f| {
            let lower = f.to_ascii_lowercase();
            f.len() <= 256
                && f.chars()
                    .all(|c| c.is_alphanumeric() || matches!(c, '-' | '_' | '.'))
                && !f.contains("..")
                && lower.ends_with(".gguf")
                && !lower.contains("mmproj")
                && !lower.starts_with("mtp-")
                && !lower.contains("-of-0")
                && super::search::extract_quant_tag(f).as_deref() == Some(wanted)
        })
        .min_by_key(|f| (f.len(), f.to_string()))
        .map(str::to_string)
}

/// Where the day's reading is saved, so a restart has it at once.
pub(crate) const POPULAR_TREE: &str = "hf_popular";
pub(crate) const POPULAR_KEY: &str = "snapshot";

/// Most models one node registers from the reading in a day. Each costs a
/// header — the 16 MB probe, the header itself, any tied output weights — and a
/// node that has heard of none of them yet would otherwise fetch them all at
/// once.
pub(crate) const MOST_REGISTERED_A_DAY: u32 = 3;

/// Nothing is registered before the node has been up this long: the swarm's
/// manifests arrive by gossip in the first minutes, and a model a peer has
/// already registered costs this node nothing.
pub(crate) const REGISTER_AFTER_UPTIME: std::time::Duration = std::time::Duration::from_secs(600);

/// Register `filename` of `repo_id` WITHOUT fetching a part (#389 (b)): the
/// header and any tied output weights into the model's directory, the manifest
/// built from them (`model::manifest::generate_manifest_from_header`, the
/// dashboard download's own step), the origin claimed, and both announced —
/// `HfSourceGossip` and the manifest — as for a model a person added. The
/// download pass then fetches parts as for any trusted model. A model this
/// build will not run is refused before the header moves; where the swarm uses
/// another upload of the model, that one is registered (`model::canonical`).
pub(crate) async fn register_without_parts(
    state: &std::sync::Arc<crate::daemon::SharedState>,
    network_tx: &tokio::sync::mpsc::Sender<crate::types::NetworkCommand>,
    repo_id: &str,
    filename: &str,
) -> Result<crate::types::ModelId, String> {
    super::validate_hf_repo_id(repo_id)?;
    let mid = crate::types::ModelId(crate::model::canonical::model_id_for_gguf_filename(
        filename,
    ));
    let asked = crate::daemon::HfSource {
        repo_id: repo_id.to_string(),
        filename: filename.to_string(),
        mmproj_filename: None,
    };
    state.note_origin_claim(&mid, asked.clone());
    let source = match state.canonical_build(&mid) {
        Some(build)
            if !crate::model::canonical::same_origin(&build.source, &asked)
                && crate::model::canonical::outranks(&mid, &build.source, &asked) =>
        {
            build.source.clone()
        }
        _ => asked,
    };
    let info = super::probe_gguf_file(
        &source.repo_id,
        &source.filename,
        state.config.model.shard_size_bytes(),
    )
    .await?;
    if let Some(refusal) = info.tensor_meta.refusal() {
        return Err(format!("this build will not run it: {refusal:?}"));
    }
    let dest_dir = state.model_dir(&mid.0);
    super::download_gguf_header(
        &source.repo_id,
        &source.filename,
        &dest_dir,
        info.header_size,
    )
    .await?;
    if let Err(e) = super::download_sidecar_tensors(
        &source.repo_id,
        &source.filename,
        &dest_dir,
        &info.tensor_meta,
    )
    .await
    {
        tracing::warn!(model = %mid, error = %e, "A model people run: its tied output weights were not fetched (non-fatal)");
    }
    crate::model::manifest::generate_manifest_from_header(
        &crate::model::manifest::ManifestGenParams {
            header_path: &dest_dir.join(crate::model::shard::HEADER_FILENAME),
            model_id_str: &mid.0,
            filename: &source.filename,
            total_size: info.total_size,
            shard_count: info.shard_count(),
            shard_indices: &[],
            shared: state,
            precomputed_layouts: Some(&info.layouts),
        },
    )?;
    let _ = network_tx
        .send(crate::types::NetworkCommand::Broadcast(
            crate::types::SwarmMessage::HfSourceGossip(crate::types::HfSourceGossip {
                model_id: mid.clone(),
                repo_id: source.repo_id.clone(),
                filename: source.filename.clone(),
                publisher: state.identity.node_id().clone(),
                mmproj_filename: None,
            }),
        ))
        .await;
    if let Some(manifest) = state.model_registry.get_manifest(&mid) {
        let _ = network_tx
            .send(crate::types::NetworkCommand::Broadcast(
                crate::types::SwarmMessage::ModelManifest(manifest),
            ))
            .await;
    }
    Ok(mid)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn repo(id: &str, downloads: u64, arch: &str, params: u64, tags: &[&str]) -> ListedRepo {
        ListedRepo {
            repo_id: id.into(),
            downloads,
            pipeline_tag: None,
            tags: tags.iter().map(|t| t.to_string()).collect(),
            created_at: Some("2026-01-01T00:00:00.000Z".into()),
            gguf: Some(ListedGguf {
                architecture: Some(arch.into()),
                total: Some(params),
            }),
        }
    }

    const NOW: i64 = 1_800_000_000;

    fn loads(arch: &str) -> bool {
        arch != "wan"
    }

    #[test]
    fn a_base_is_summed_over_its_uploads_and_held_as_one_trusted_upload() {
        let base = "base_model:quantized:Qwen/Qwen3.8-27B";
        let listed = [
            repo(
                "someone/Qwen3.8-27B-GGUF",
                900_000,
                "qwen35",
                27_000_000_000,
                &[base],
            ),
            repo(
                "unsloth/Qwen3.8-27B-GGUF",
                500_000,
                "qwen35",
                27_000_000_000,
                &[base],
            ),
            repo(
                "bartowski/Qwen_Qwen3.8-27B-GGUF",
                200_000,
                "qwen35",
                27_000_000_000,
                &[base],
            ),
            repo("x/Wan-video-GGUF", 9_000_000, "wan", 14_000_000_000, &[]),
        ];
        let bases = choose(&listed, &loads, &|_| false, NOW);
        assert_eq!(bases.len(), 1, "{bases:?}");
        assert_eq!(bases[0].base, "qwen/qwen3.8-27b");
        assert_eq!(bases[0].downloads, 1_600_000, "summed over every upload");
        assert_eq!(
            bases[0].uploads[0], "bartowski/Qwen_Qwen3.8-27B-GGUF",
            "the trusted publishers' order, not the most downloaded"
        );
        assert_eq!(bases[0].tier, 3, "27B at Q4_K_M is ~16 GB");
    }

    #[test]
    fn an_upload_the_swarm_holds_is_kept_over_a_better_ranked_one() {
        let base = "base_model:quantized:Qwen/Qwen3.5-9B";
        let listed = [
            repo(
                "unsloth/Qwen3.5-9B-GGUF",
                500_000,
                "qwen35",
                9_000_000_000,
                &[base],
            ),
            repo(
                "bartowski/Qwen_Qwen3.5-9B-GGUF",
                200_000,
                "qwen35",
                9_000_000_000,
                &[base],
            ),
        ];
        let held = |r: &str| r == "unsloth/Qwen3.5-9B-GGUF";
        let bases = choose(&listed, &loads, &held, NOW);
        assert_eq!(bases[0].uploads[0], "unsloth/Qwen3.5-9B-GGUF");
    }

    #[test]
    fn a_pumped_or_new_upload_counts_for_its_base_but_is_never_held() {
        let base = "base_model:quantized:google/gemma-4-12b-it";
        let mut fresh = repo(
            "pump/gemma-4-12b-GGUF",
            5_000_000,
            "gemma4",
            12_000_000_000,
            &[base],
        );
        fresh.created_at = Some("2027-01-15T00:00:00.000Z".into()); // after NOW
        let listed = [
            fresh,
            repo(
                "nobody/gemma-4-12b-GGUF",
                50_000,
                "gemma4",
                12_000_000_000,
                &[base],
            ),
        ];
        let bases = choose(&listed, &loads, &|_| false, NOW);
        assert!(
            bases.is_empty(),
            "a day-old pump and an unknown publisher under 100,000 cannot be held: {bases:?}"
        );
    }

    #[test]
    fn each_size_keeps_its_own_top_bases() {
        let mut listed = Vec::new();
        for i in 0..5u64 {
            listed.push(repo(
                &format!("bartowski/small-{i}-GGUF"),
                1_000_000 - i,
                "qwen3",
                1_500_000_000,
                &[],
            ));
            listed.push(repo(
                &format!("bartowski/large-{i}-GGUF"),
                10_000_000 - i,
                "llama",
                70_000_000_000,
                &[],
            ));
        }
        let bases = choose(&listed, &loads, &|_| false, NOW);
        let per = |t: u8| bases.iter().filter(|b| b.tier == t).count();
        assert_eq!(
            per(0),
            BASES_PER_TIER,
            "the small ones are not crowded out by the large"
        );
        assert_eq!(per(4), BASES_PER_TIER);
        assert_eq!(bases.len(), 2 * BASES_PER_TIER);
    }

    #[test]
    fn a_reranker_or_speech_model_of_a_supported_family_is_not_a_chat_model() {
        let mut reranker = repo(
            "ggml-org/Qwen3-Reranker-0.6B-GGUF",
            9_000_000,
            "qwen3",
            600_000_000,
            &[],
        );
        reranker.pipeline_tag = Some("text-ranking".into());
        let mut vision = repo(
            "bartowski/gemma-4-E4B-it-GGUF",
            900_000,
            "gemma4",
            8_000_000_000,
            &[],
        );
        vision.pipeline_tag = Some("image-text-to-text".into());
        let bases = choose(&[reranker, vision], &loads, &|_| false, NOW);
        assert_eq!(bases.len(), 1);
        assert_eq!(
            bases[0].base, "bartowski/gemma-4-e4b-it-gguf",
            "a chat model that also reads images is kept"
        );
    }

    #[test]
    fn a_fine_tune_is_its_own_base() {
        let r = repo(
            "x/Tuned-GGUF",
            1,
            "qwen35",
            1,
            &[
                "base_model:finetune:Qwen/Qwen3.5-9B",
                "base_model:quantized:x/Tuned",
            ],
        );
        assert_eq!(base_of(&r), "x/tuned");
        let plain = repo(
            "x/Reup-GGUF",
            1,
            "qwen35",
            1,
            &["base_model:Qwen/Qwen3.5-9B"],
        );
        assert_eq!(base_of(&plain), "qwen/qwen3.5-9b");
        let none = repo(
            "x/Own-GGUF",
            1,
            "qwen35",
            1,
            &["base_model:finetune:Qwen/Qwen3.5-9B"],
        );
        assert_eq!(
            base_of(&none),
            "x/own-gguf",
            "built from another model is not a base here"
        );
    }

    #[test]
    fn one_file_per_model_q4_k_m_or_q8_0_for_a_small_one() {
        let files = [
            "Qwen_Qwen3.5-4B-Q8_0.gguf",
            "Qwen_Qwen3.5-4B-Q4_K_M.gguf",
            "Qwen_Qwen3.5-4B-Q4_K_S.gguf",
            "mmproj-Qwen_Qwen3.5-4B-Q4_K_M.gguf",
            "mtp-Qwen_Qwen3.5-4B-Q4_K_M.gguf",
        ];
        assert_eq!(
            pick_file(files, 4_000_000_000).as_deref(),
            Some("Qwen_Qwen3.5-4B-Q4_K_M.gguf")
        );
        assert_eq!(
            pick_file(files, 1_000_000_000).as_deref(),
            Some("Qwen_Qwen3.5-4B-Q8_0.gguf")
        );
        let split = [
            "Big-Q4_K_M-00001-of-00002.gguf",
            "Big-Q4_K_M-00002-of-00002.gguf",
        ];
        assert_eq!(
            pick_file(split, 70_000_000_000),
            None,
            "one part of a split upload is not a model"
        );
        let folder = ["Q4_K_M/Big-Q4_K_M.gguf"];
        assert_eq!(
            pick_file(folder, 70_000_000_000),
            None,
            "the download path takes a bare name"
        );
    }

    #[test]
    fn a_snapshot_is_acted_on_only_while_fresh() {
        let s = PopularSnapshot {
            bases: vec![],
            fetched_at: NOW,
        };
        assert!(s.fresh(NOW + 3600));
        assert!(!s.fresh(NOW + MOST_SNAPSHOT_AGE_SECS + 1));
        assert!(
            !PopularSnapshot::default().fresh(NOW),
            "never read is never fresh"
        );
        assert!(!s.due(NOW + 3600));
        assert!(s.due(NOW + REFRESH_EVERY_SECS));
    }
}
