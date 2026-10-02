//! One upload per model id, on every node.
//!
//! A model id is derived from a GGUF's FILE NAME
//! ([`model_id_for_gguf_filename`]), so every independent upload of one model
//! and quantisation answers to the same id: `bartowski/…/Qwen2.5-Coder-7B-
//! Instruct-Q4_K_M.gguf`, `Qwen/…/qwen2.5-coder-7b-instruct-q4_k_m.gguf` and a
//! third-party requant were all `qwen2.5-coder-7b-instruct-q4-k-m` on the live
//! swarm, within 800 bytes of each other and sharing not one shard hash
//! (gotcha #406). Each was verified against its own upload, so nothing was
//! corrupt — but two computers holding different uploads cannot split the
//! model between them, and on 2026-10-01 nine of twenty models had holders of
//! another upload (`docs/FUTURE_WORK.md` #151). Nothing chose: a node kept the
//! first source it heard of, a dashboard download used whatever was clicked,
//! and a node that did not know a source searched HuggingFace and took the
//! first hit.
//!
//! This module is the rule every node applies to pick the SAME upload:
//!
//! - **A total order over uploads of one model** ([`origin_rank`]): a pinned
//!   reference model first (`model::reference`), then the publisher's place in
//!   `TRUSTED_HF_PUBLISHERS` (official authors, then curators), then anyone
//!   else; ties broken by name. Static, so two nodes that know the same uploads
//!   always choose the same one.
//! - **Choose the best upload anyone has claimed.** Knowledge only grows by
//!   gossip and the choice is the maximum under a fixed order, so every node
//!   converges on one answer — a state-based max-register CRDT (Shapiro et al.,
//!   2011): merge is "keep the better", which is commutative, associative and
//!   idempotent, so delivery order and duplication cannot split the swarm.
//!
//! Prior art drew the line the same way: Petals keys a model's swarm identity
//! by its HuggingFace repo, and Ollama resolves `name:tag` to exactly one
//! manifest digest. A name is not a content identity (gotcha #406's rule); this
//! keeps the friendly name and makes it resolve to exactly one file.
//!
//! The side effects — validating an upload against HuggingFace, moving a node
//! holding another upload onto the chosen one — live in
//! `model::auto_manage::canonical`. Everything here is pure.

use serde::{Deserialize, Serialize};

use crate::daemon::HfSource;
use crate::types::{ModelId, ModelManifest};

/// The model id a GGUF file name gives — lowercase, anything that is not
/// alphanumeric, `-` or `.` replaced by `-`, runs collapsed, extension dropped.
///
/// The ONE derivation: the dashboard download, the HuggingFace search badges
/// and the canonical-upload check all ask it, so an upload claimed for a model
/// can be checked against the id it would really produce.
pub fn model_id_for_gguf_filename(filename: &str) -> String {
    filename
        .trim_end_matches(".gguf")
        .to_lowercase()
        .replace(|c: char| !c.is_alphanumeric() && c != '-' && c != '.', "-")
        .split('-')
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join("-")
}

/// Would downloading `source` produce THIS model id?
///
/// A claim that names another model's file is refused outright. Without the
/// check, one gossip message could point a model at any file on HuggingFace and
/// — once that file outranked the real ones — move every holder onto it.
pub fn origin_names_model(source: &HfSource, model_id: &ModelId) -> bool {
    model_id_for_gguf_filename(&source.filename) == model_id.0
}

/// Two names for the same upload? HuggingFace resolves repo names without
/// regard to case; file names are case-sensitive.
pub fn same_origin(a: &HfSource, b: &HfSource) -> bool {
    a.repo_id.eq_ignore_ascii_case(&b.repo_id) && a.filename == b.filename
}

/// Where an upload stands among the uploads of one model. Lower is better, and
/// the order is total, so any two nodes rank any two uploads identically.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct OriginRank {
    /// 0: a pinned reference model. 1 + position in the publisher allowlist.
    /// 1 + its length: everyone else.
    tier: usize,
    repo_folded: String,
    file_folded: String,
    // Exact strings last, so two spellings of one upload still order the same
    // way everywhere rather than by whichever arrived first.
    repo: String,
    file: String,
}

pub fn origin_rank(model_id: &ModelId, source: &HfSource) -> OriginRank {
    let pinned = crate::model::reference::REFERENCE_MODELS.iter().any(|m| {
        m.model_id == model_id.0
            && m.repo_id.eq_ignore_ascii_case(&source.repo_id)
            && m.filename == source.filename
    });
    let tier = if pinned {
        0
    } else {
        1 + crate::model::huggingface::trusted_publisher_position(&source.repo_id)
            .unwrap_or_else(crate::model::huggingface::trusted_publisher_count)
    };
    OriginRank {
        tier,
        repo_folded: source.repo_id.to_lowercase(),
        file_folded: source.filename.to_lowercase(),
        repo: source.repo_id.clone(),
        file: source.filename.clone(),
    }
}

/// Does `a` outrank `b` as the upload for `model_id`?
pub fn outranks(model_id: &ModelId, a: &HfSource, b: &HfSource) -> bool {
    origin_rank(model_id, a) < origin_rank(model_id, b)
}

/// How many uploads of one model a node remembers. The best ones are kept, so
/// the cap can never drop the answer — only uploads that could not win anyway.
pub const MAX_ORIGIN_CLAIMS_PER_MODEL: usize = 8;

/// Remember that `source` is an upload of `model_id`. Keeps `claims` best-first,
/// without duplicates, and at most [`MAX_ORIGIN_CLAIMS_PER_MODEL`] long.
/// Returns whether anything changed. A claim naming another model is ignored.
pub fn note_claim(claims: &mut Vec<HfSource>, model_id: &ModelId, source: HfSource) -> bool {
    if !origin_names_model(&source, model_id) {
        return false;
    }
    if claims.iter().any(|c| same_origin(c, &source)) {
        return false;
    }
    let rank = origin_rank(model_id, &source);
    let at = claims
        .iter()
        .position(|c| rank < origin_rank(model_id, c))
        .unwrap_or(claims.len());
    if at >= MAX_ORIGIN_CLAIMS_PER_MODEL {
        return false;
    }
    claims.insert(at, source);
    claims.truncate(MAX_ORIGIN_CLAIMS_PER_MODEL);
    true
}

/// The best claimed upload that `excluded` does not rule out (one HuggingFace
/// refused to serve, say). `claims` need not be sorted.
pub fn best_claim<'a>(
    model_id: &ModelId,
    claims: &'a [HfSource],
    excluded: impl Fn(&HfSource) -> bool,
) -> Option<&'a HfSource> {
    claims
        .iter()
        .filter(|c| origin_names_model(c, model_id) && !excluded(c))
        .min_by_key(|c| origin_rank(model_id, c))
}

/// Is the one-upload-per-model machinery on? `SWARMLLM_CANONICAL_UPLOADS=0`
/// switches it off: no upload is verified or adopted, nothing switches, and
/// every acquisition gate answers as before. For rigs and gates that link a
/// node's files into throwaway nodes and must not have them switched mid-step,
/// and as the in-binary A/B control. Read once.
pub fn canonical_uploads_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("SWARMLLM_CANONICAL_UPLOADS").as_deref() != Ok("0"))
}

/// DB tree holding each model's [`CanonicalBuild`], keyed by model id.
pub const CANONICAL_BUILDS_TREE: &str = "canonical_builds";

/// What this node's own copy of a model is, against the canonical upload.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case", tag = "state")]
pub enum Holding {
    /// Holds no part of it.
    Nothing,
    /// Holds parts of the canonical upload — checked against HuggingFace.
    Canonical,
    /// Holds parts of another upload and is fetching the canonical one's to
    /// replace them. The old parts keep serving until the new ones are in.
    Switching { fetched: u32, needed: u32 },
    /// Holds parts of another upload and cannot switch right now. `reason` is
    /// an i18n key suffix (`models.build.stuck_<reason>`).
    Stuck { reason: &'static str },
}

/// Every canonical layout is cut at the DEFAULT shard size, never at this
/// node's `model.shard_size_mb`: a node configured differently would otherwise
/// cut the same file into other parts, and parts are what the swarm trades.
pub fn canonical_shard_size_bytes() -> u64 {
    crate::config::ModelConfig::default().shard_size_bytes()
}

/// The upload the swarm uses for one model, as this node verified it against
/// HuggingFace: where it is, and its shape — what any manifest describing it
/// must say.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CanonicalBuild {
    pub source: HfSource,
    /// Size of the whole GGUF file on HuggingFace.
    pub total_size: u64,
    /// Bytes of GGUF header before the first tensor.
    pub header_size: u64,
    /// Size of each part, by index, at [`canonical_shard_size_bytes`].
    pub shard_sizes: Vec<u64>,
    /// Layer range of each part, `[start, end)`.
    pub shard_layers: Vec<(u32, u32)>,
    /// Each part's first tensor: its offset in the upload and its size. A part
    /// file begins with exactly those bytes, so comparing a slice of them is
    /// how a node learns whether the part it holds came from this upload
    /// without downloading it again.
    pub shard_first_tensor: Vec<(u64, u64)>,
    /// BLAKE3 of the upload's header, as downloaded from HuggingFace.
    pub header_hash: [u8; 32],
    /// When this node verified it (ms since epoch).
    pub resolved_at_ms: u64,
}

impl CanonicalBuild {
    /// The build a probe of `source` describes, with `header` the upload's
    /// header bytes.
    pub fn from_probe(
        source: HfSource,
        info: &crate::model::huggingface::GgufFileInfo,
        header: &[u8],
        now_ms: u64,
    ) -> Self {
        CanonicalBuild {
            source: HfSource {
                mmproj_filename: None,
                ..source
            },
            total_size: info.total_size,
            header_size: info.header_size,
            shard_sizes: info.layouts.iter().map(|l| l.size_bytes).collect(),
            shard_layers: info
                .layouts
                .iter()
                .map(|l| (l.layer_start, l.layer_end))
                .collect(),
            shard_first_tensor: info
                .layouts
                .iter()
                .map(|l| l.tensors.first().map_or((0, 0), |t| (t.1, t.2)))
                .collect(),
            header_hash: blake3::hash(header).into(),
            resolved_at_ms: now_ms,
        }
    }

    /// The layouts this upload is cut into, re-derived from its header — the
    /// same computation `probe_gguf_file` runs, at the canonical shard size.
    /// `None` if the header does not reproduce the recorded part sizes, which
    /// would mean it is not this upload's header.
    pub fn layouts_from_header(
        &self,
        header_path: &std::path::Path,
    ) -> Option<Vec<crate::inference::split::LayerShardLayout>> {
        let meta = crate::inference::split::GgufTensorMeta::from_gguf_file(header_path).ok()?;
        let count = self
            .total_size
            .div_ceil(canonical_shard_size_bytes())
            .max(1) as u32;
        let layouts = crate::inference::split::compute_layer_shard_layouts(&meta, count);
        let sizes: Vec<u64> = layouts.iter().map(|l| l.size_bytes).collect();
        (sizes == self.shard_sizes).then_some(layouts)
    }

    pub fn shard_count(&self) -> u32 {
        self.shard_sizes.len() as u32
    }

    /// Does `manifest` describe THIS upload? Compared on shape — every part's
    /// size and the file's — because a manifest carries real sizes for every
    /// part, held or not, while it carries hashes only for parts its author
    /// holds (same reasoning as `ModelRegistry::describes_a_different_build`).
    /// Uploads of one model measured on HuggingFace differ in total size by
    /// hundreds of bytes; a same-shape different upload is caught by the byte
    /// check in `auto_manage::canonical` before a node treats its parts as
    /// these.
    pub fn describes(&self, manifest: &ModelManifest) -> bool {
        manifest.total_size_bytes == self.total_size
            && manifest.shard_count == self.shard_count()
            && manifest.shards.iter().all(|s| {
                self.shard_sizes
                    .get(s.index as usize)
                    .is_some_and(|&size| size == s.size_bytes)
                    // Where the manifest carries its tensor table, each part's
                    // first tensor must sit where this upload puts it: that
                    // offset follows from the header, and a table built from
                    // another upload's header sends every read to the wrong
                    // place (the .220 gate, step 12k).
                    && s.tensors.first().is_none_or(|t| {
                        self.shard_first_tensor
                            .get(s.index as usize)
                            .is_some_and(|&(offset, _)| offset == t.gguf_offset)
                    })
            })
    }

    /// This upload's parts that hold any layer in `ranges` (each `[start,
    /// end)`), ascending. What a node holding those layers of another upload
    /// fetches to hold the same layers of this one.
    pub fn indices_covering(&self, ranges: &[(u32, u32)]) -> Vec<u32> {
        self.shard_layers
            .iter()
            .enumerate()
            .filter(|(_, &(start, end))| ranges.iter().any(|&(s, e)| start < e && s < end))
            .map(|(i, _)| i as u32)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn src(repo: &str, file: &str) -> HfSource {
        HfSource {
            repo_id: repo.into(),
            filename: file.into(),
            mmproj_filename: None,
        }
    }

    fn coder() -> ModelId {
        ModelId("qwen2.5-coder-7b-instruct-q4-k-m".into())
    }

    /// The three uploads gotcha #406 measured on the live swarm, under the one
    /// id they all produce. The official author's wins.
    #[test]
    fn the_official_upload_outranks_a_curator_and_a_stranger() {
        let id = coder();
        let official = src(
            "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",
            "qwen2.5-coder-7b-instruct-q4_k_m.gguf",
        );
        let curator = src(
            "bartowski/Qwen2.5-Coder-7B-Instruct-GGUF",
            "Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf",
        );
        let stranger = src(
            "stefancosma/Qwen2.5-Coder-7B-Instruct-Q4_K_M-GGUF",
            "qwen2.5-coder-7b-instruct-q4_k_m.gguf",
        );
        for s in [&official, &curator, &stranger] {
            assert!(origin_names_model(s, &id), "{} names the model", s.repo_id);
        }
        assert!(outranks(&id, &official, &curator));
        assert!(outranks(&id, &curator, &stranger));
        assert!(outranks(&id, &official, &stranger));
    }

    /// A pinned reference model is what every benchmark in the docs used, so it
    /// stays the canonical upload even where a higher-listed publisher has one.
    #[test]
    fn a_pinned_reference_model_outranks_everything() {
        let id = ModelId("llama-3.2-3b-instruct-q4-k-m".into());
        let pinned = src(
            "bartowski/Llama-3.2-3B-Instruct-GGUF",
            "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        );
        let official = src(
            "meta-llama/Llama-3.2-3B-Instruct-GGUF",
            "llama-3.2-3b-instruct-q4_k_m.gguf",
        );
        assert!(outranks(&id, &pinned, &official));
    }

    /// Curators rank in allowlist order; anyone unlisted ranks after all of
    /// them, by name, without regard to case.
    #[test]
    fn curators_rank_in_list_order_and_strangers_by_name() {
        let id = ModelId("llama-3.2-1b-instruct-q8-0".into());
        let f = "Llama-3.2-1B-Instruct-Q8_0.gguf";
        let bartowski = src("bartowski/Llama-3.2-1B-Instruct-GGUF", f);
        let unsloth = src("unsloth/Llama-3.2-1B-Instruct-GGUF", f);
        let hq = src("hugging-quants/Llama-3.2-1B-Instruct-Q8_0-GGUF", f);
        let zed = src("Zed/Llama-3.2-1B-Instruct-GGUF", f);
        assert!(outranks(&id, &bartowski, &unsloth));
        assert!(outranks(&id, &unsloth, &hq));
        assert!(outranks(&id, &hq, &zed));
    }

    /// The property the whole design rests on: whatever order claims arrive in,
    /// every node picks the same one.
    #[test]
    fn every_arrival_order_settles_on_the_same_upload() {
        let id = coder();
        let all = [
            src(
                "stefancosma/Qwen2.5-Coder-7B-Instruct-Q4_K_M-GGUF",
                "qwen2.5-coder-7b-instruct-q4_k_m.gguf",
            ),
            src(
                "bartowski/Qwen2.5-Coder-7B-Instruct-GGUF",
                "Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf",
            ),
            src(
                "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",
                "qwen2.5-coder-7b-instruct-q4_k_m.gguf",
            ),
            src(
                "aaa/Qwen2.5-Coder-7B-Instruct-GGUF",
                "qwen2.5-coder-7b-instruct-q4_k_m.gguf",
            ),
        ];
        let orders: [[usize; 4]; 6] = [
            [0, 1, 2, 3],
            [3, 2, 1, 0],
            [1, 3, 0, 2],
            [2, 0, 3, 1],
            [3, 0, 2, 1],
            [1, 2, 3, 0],
        ];
        for order in orders {
            let mut claims = Vec::new();
            for i in order {
                note_claim(&mut claims, &id, all[i].clone());
            }
            let best = best_claim(&id, &claims, |_| false).unwrap();
            assert_eq!(
                best.repo_id, "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",
                "{order:?}"
            );
            assert_eq!(claims[0].repo_id, best.repo_id, "kept best-first");
        }
    }

    /// A claim naming another model's file is not a claim about this one.
    #[test]
    fn a_claim_for_another_models_file_is_refused() {
        let id = coder();
        let wrong = src(
            "Qwen/Qwen2.5-14B-Instruct-GGUF",
            "qwen2.5-14b-instruct-q4_k_m.gguf",
        );
        let mut claims = Vec::new();
        assert!(!note_claim(&mut claims, &id, wrong.clone()));
        assert!(claims.is_empty());
        // Even if one were present, it can never be chosen.
        assert!(best_claim(&id, &[wrong], |_| false).is_none());
    }

    #[test]
    fn claims_are_deduplicated_without_regard_to_repo_case_and_bounded() {
        let id = ModelId("m-q4-k-m".into());
        let mut claims = Vec::new();
        assert!(note_claim(&mut claims, &id, src("Org/M", "M-Q4_K_M.gguf")));
        assert!(!note_claim(&mut claims, &id, src("org/m", "M-Q4_K_M.gguf")));
        for i in 0..20 {
            note_claim(
                &mut claims,
                &id,
                src(&format!("u{i:02}/M"), "M-Q4_K_M.gguf"),
            );
        }
        assert_eq!(claims.len(), MAX_ORIGIN_CLAIMS_PER_MODEL);
        // A better claim still gets in once full; a worse one does not.
        assert!(note_claim(
            &mut claims,
            &id,
            src("bartowski/M", "M-Q4_K_M.gguf")
        ));
        assert_eq!(claims[0].repo_id, "bartowski/M");
        assert!(!note_claim(&mut claims, &id, src("zzz/M", "M-Q4_K_M.gguf")));
    }

    #[test]
    fn an_excluded_upload_hands_the_choice_to_the_next_best() {
        let id = coder();
        let official = src(
            "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",
            "qwen2.5-coder-7b-instruct-q4_k_m.gguf",
        );
        let curator = src(
            "bartowski/Qwen2.5-Coder-7B-Instruct-GGUF",
            "Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf",
        );
        let claims = vec![curator.clone(), official.clone()];
        let best = best_claim(&id, &claims, |c| same_origin(c, &official)).unwrap();
        assert!(same_origin(best, &curator));
    }

    fn build() -> CanonicalBuild {
        CanonicalBuild {
            source: src("Qwen/X", "x-q4_k_m.gguf"),
            total_size: 1_000,
            header_size: 100,
            shard_sizes: vec![300, 300, 300],
            shard_layers: vec![(0, 4), (4, 8), (8, 12)],
            shard_first_tensor: vec![(100, 300), (400, 300), (700, 300)],
            header_hash: [1; 32],
            resolved_at_ms: 0,
        }
    }

    fn manifest_shaped(total: u64, sizes: &[u64]) -> ModelManifest {
        let shards = sizes
            .iter()
            .enumerate()
            .map(|(i, &size_bytes)| crate::types::ShardInfo {
                index: i as u32,
                layer_range: (0, 0),
                size_bytes,
                hash: [0; 32],
                tensors: Vec::new(),
            })
            .collect();
        crate::model::manifest::build_manifest_from_gguf(crate::model::manifest::ManifestFromGguf {
            id: ModelId("x-q4-k-m".into()),
            name: "x".into(),
            architecture: crate::types::ModelArchitecture::Llama,
            num_layers: 12,
            total_size_bytes: total,
            shard_count: sizes.len() as u32,
            shards,
            publisher: crate::types::NodeId([0; 32]),
        })
    }

    #[test]
    fn a_manifest_of_another_shape_is_another_upload() {
        let b = build();
        assert!(b.describes(&manifest_shaped(1_000, &[300, 300, 300])));
        // 800 bytes apart was the real case; one is enough.
        assert!(!b.describes(&manifest_shaped(1_001, &[300, 300, 300])));
        assert!(!b.describes(&manifest_shaped(1_000, &[300, 301, 299])));
        assert!(!b.describes(&manifest_shaped(1_000, &[450, 450])));
    }

    /// A manifest the same shape as the upload but with a tensor table built
    /// from another upload's header is NOT this upload: every offset in it is
    /// wrong. Without the offset check it passed, and the worker read every
    /// tensor from the wrong place (the .220 gate, step 12k).
    #[test]
    fn a_tensor_table_from_another_header_is_another_upload() {
        let b = build();
        let with_first_tensor_at = |offset: u64| {
            let mut m = manifest_shaped(1_000, &[300, 300, 300]);
            for (i, s) in m.shards.iter_mut().enumerate() {
                s.tensors.push(crate::types::ShardTensorEntry {
                    name: format!("blk.{i}.attn_q.weight"),
                    gguf_offset: if i == 1 {
                        offset
                    } else {
                        b.shard_first_tensor[i].0
                    },
                    shard_offset: 0,
                    size: 300,
                });
            }
            m
        };
        assert!(b.describes(&with_first_tensor_at(b.shard_first_tensor[1].0)));
        assert!(!b.describes(&with_first_tensor_at(b.shard_first_tensor[1].0 + 3_808)));
    }

    #[test]
    fn the_parts_covering_held_layers_are_the_parts_to_fetch() {
        let b = build();
        assert_eq!(b.indices_covering(&[(0, 12)]), vec![0, 1, 2]);
        assert_eq!(b.indices_covering(&[(5, 7)]), vec![1]);
        // Straddling a boundary needs both parts.
        assert_eq!(b.indices_covering(&[(3, 5)]), vec![0, 1]);
        assert_eq!(b.indices_covering(&[(12, 14)]), Vec::<u32>::new());
    }

    /// The derivation the dashboard download has always used — moved here, not
    /// changed. Every id on the live swarm still comes out the same.
    #[test]
    fn the_filename_derivation_is_unchanged() {
        assert_eq!(
            model_id_for_gguf_filename("THUDM_GLM-4-9B-0414-Q4_K_M.gguf"),
            "thudm-glm-4-9b-0414-q4-k-m"
        );
        assert_eq!(
            model_id_for_gguf_filename("Phi-3.5-mini-instruct.Q4_K_M.gguf"),
            "phi-3.5-mini-instruct.q4-k-m"
        );
        for m in crate::model::reference::REFERENCE_MODELS {
            assert_eq!(model_id_for_gguf_filename(m.filename), m.model_id);
        }
    }
}
