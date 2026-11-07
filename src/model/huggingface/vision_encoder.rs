//! A vision model's image reader (`mmproj.gguf`, the part at
//! `MMPROJ_SHARD_INDEX`), fetched from HuggingFace only by a node downloading
//! the model from HuggingFace itself (#422) — never because a peer named a
//! file. Every other node takes it from a holder, checked against the pin this
//! fetch puts in the manifest (`manifest::expected_part`).
//!
//! Prior art: BitTorrent BEP 9 takes metadata a peer supplies only once it
//! hashes to the info-hash the user chose; Ollama ships the projector as a
//! manifest layer named by its sha256 digest, and `verifyBlob` refuses a pulled
//! blob whose digest differs; llama.cpp's `-hf` download chooses among several
//! with `common/download.cpp::find_best_sibling`, ported as
//! [`best_encoder_for`]. Here the SHA-256 comes from HuggingFace's own listing,
//! so neither the choice nor the check rests on a peer's word.

use super::private_types::{HfFileInfo, HfModelDetail};

/// One vision encoder a repository lists, with the size and SHA-256
/// HuggingFace stores for it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ListedEncoder {
    pub filename: String,
    pub size_bytes: u64,
    pub sha256: [u8; 32],
}

/// The vision encoders a repository lists (`*mmproj*.gguf`). A file listed
/// without a size or a SHA-256 cannot be checked, so it is left out.
pub(crate) async fn listed_encoders(repo_id: &str) -> Result<Vec<ListedEncoder>, String> {
    super::validate_hf_repo_id(repo_id)?;
    let url = format!("https://huggingface.co/api/models/{repo_id}?blobs=true");
    let resp = super::hf_headers(super::HF_META_CLIENT.get(&url))
        .send()
        .await
        .map_err(|e| format!("File list request failed: {e}"))?;
    if !resp.status().is_success() {
        return Err(format!("File list returned {}", resp.status()));
    }
    let detail: HfModelDetail = resp
        .json()
        .await
        .map_err(|e| format!("Failed to parse file list: {e}"))?;
    Ok(encoders_in(detail.siblings.unwrap_or_default()))
}

fn encoders_in(siblings: Vec<HfFileInfo>) -> Vec<ListedEncoder> {
    siblings
        .into_iter()
        .filter_map(|f| {
            if !f.rfilename.ends_with(".gguf") || !f.rfilename.contains("mmproj") {
                return None;
            }
            let size_bytes = f.size.filter(|s| *s > 0)?;
            let sha256: [u8; 32] = hex::decode(f.lfs?.sha256).ok()?.try_into().ok()?;
            Some(ListedEncoder {
                filename: f.rfilename,
                size_bytes,
                sha256,
            })
        })
        .collect()
}

/// A file's quantization width as llama.cpp reads it off the name
/// (`extract_quant_bits`): the tag after the last `-` or `.` (a split's
/// `-00001-of-00003` dropped first), its first run of digits — `Q4_K_M` → 4,
/// `F16` and `BF16` → 16, none → 0.
fn quant_bits(path: &str) -> i64 {
    let Some(mut prefix) = path.strip_suffix(".gguf") else {
        return 0;
    };
    let b = prefix.as_bytes();
    if b.len() > 15 {
        let s = &b[b.len() - 15..];
        if s[0] == b'-'
            && s[1..6].iter().all(u8::is_ascii_digit)
            && s[6..10].eq_ignore_ascii_case(b"-of-")
            && s[10..15].iter().all(u8::is_ascii_digit)
        {
            prefix = &prefix[..prefix.len() - 15];
        }
    }
    let Some(sep) = prefix.rfind(['-', '.']) else {
        return 0;
    };
    let tag = &prefix[sep + 1..];
    if tag.is_empty() || !tag.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
        return 0;
    }
    let digits: String = tag
        .chars()
        .skip_while(|c| !c.is_ascii_digit())
        .take_while(char::is_ascii_digit)
        .collect();
    digits.parse().unwrap_or(0)
}

/// The vision encoder llama.cpp would load beside `model_file`
/// (`find_best_sibling` with keyword `mmproj` and no tag): one in the model
/// file's own folder or a folder above it, the deepest first, then the closest
/// quantization width to the model's; the first listed wins a tie.
pub(crate) fn best_encoder_for<'a>(
    listed: &'a [ListedEncoder],
    model_file: &str,
) -> Option<&'a ListedEncoder> {
    let model_bits = quant_bits(model_file);
    let model_parts: Vec<&str> = model_file.split('/').collect();
    let model_dir = &model_parts[..model_parts.len() - 1];
    let mut best: Option<(&ListedEncoder, usize, i64)> = None;
    for f in listed {
        let sib_parts: Vec<&str> = f.filename.split('/').collect();
        let sib_dir = &sib_parts[..sib_parts.len() - 1];
        if sib_dir.len() > model_dir.len() || model_dir[..sib_dir.len()] != *sib_dir {
            continue;
        }
        let depth = sib_dir.len();
        let diff = (quant_bits(&f.filename) - model_bits).abs();
        let better = match best {
            None => true,
            Some((_, best_depth, best_diff)) => {
                depth > best_depth || (depth == best_depth && diff < best_diff)
            }
        };
        if better {
            best = Some((f, depth, diff));
        }
    }
    best.map(|(f, _, _)| f)
}

/// The listed file a pin names (#422): its SHA-256 is HuggingFace's own, so a
/// later origin finds the same file without any name travelling, and never
/// chooses again.
pub(crate) fn listed_with_sha256<'a>(
    listed: &'a [ListedEncoder],
    sha256: &[u8; 32],
) -> Option<&'a ListedEncoder> {
    listed.iter().find(|f| &f.sha256 == sha256)
}

/// Models whose repository was asked recently and gave no encoder to fetch —
/// none listed, a type this build cannot read, the pinned file gone — so the
/// download pass, which reaches here once per part it fetches, lists a
/// repository once per [`ASK_AGAIN_AFTER`], not once per part.
static NOTHING_TO_FETCH: std::sync::LazyLock<
    std::sync::Mutex<std::collections::HashMap<crate::types::ModelId, std::time::Instant>>,
> = std::sync::LazyLock::new(Default::default);

const ASK_AGAIN_AFTER: std::time::Duration = std::time::Duration::from_secs(6 * 3600);
const MOST_REMEMBERED: usize = 1024;

fn asked_recently(model_id: &crate::types::ModelId) -> bool {
    let map = NOTHING_TO_FETCH.lock().unwrap_or_else(|e| e.into_inner());
    map.get(model_id)
        .is_some_and(|at| at.elapsed() < ASK_AGAIN_AFTER)
}

fn note_nothing_to_fetch(model_id: &crate::types::ModelId) {
    let mut map = NOTHING_TO_FETCH.lock().unwrap_or_else(|e| e.into_inner());
    map.retain(|_, at| at.elapsed() < ASK_AGAIN_AFTER);
    if map.len() >= MOST_REMEMBERED {
        return;
    }
    map.insert(model_id.clone(), std::time::Instant::now());
}

/// Fetch a vision model's image reader as the model's own origin (#422), when
/// no computer holds one: the file the pin names if the manifest has one,
/// else the one llama.cpp would choose ([`best_encoder_for`]); only a projector
/// type this build reads (`vision::reads_projector_type`, off the header by
/// range); held to HuggingFace's listed size and SHA-256. Then its BLAKE3
/// becomes origin-derived, the pin goes into the manifest, this node announces
/// itself as the holder and gossips the manifest. Answers the pin, or `None`
/// where there was nothing to fetch or the fetch failed (logged).
pub(crate) async fn fetch_from_origin(
    state: &std::sync::Arc<crate::daemon::state::SharedState>,
    net_tx: &tokio::sync::mpsc::Sender<crate::types::NetworkCommand>,
    model_id: &crate::types::ModelId,
    repo_id: &str,
    model_file: &str,
) -> Option<crate::types::MmprojInfo> {
    let sentinel = crate::types::ShardId::mmproj_for(model_id.clone());
    let model_dir = crate::model::shard::model_dir(&state.config.node.data_dir, &model_id.0);
    let held_by = state.model_registry.mmproj_holders(model_id).len();
    let skip = if held_by > 0 {
        "a computer holds it"
    } else if model_dir
        .join(crate::model::shard::MMPROJ_FILENAME)
        .exists()
    {
        "it is on disk here"
    } else if asked_recently(model_id) {
        "its repository gave nothing to fetch lately"
    } else {
        ""
    };
    if !skip.is_empty() {
        tracing::debug!(model = %model_id, held_by, "Not fetching the image reader: {skip}");
        return None;
    }
    // One writer per file: a second origin download of this model waits out.
    let Some(_claim) = state.models.claim_shard_download(&sentinel) else {
        tracing::debug!(model = %model_id, "Not fetching the image reader: another download has it");
        return None;
    };
    let listed = match listed_encoders(repo_id).await {
        Ok(listed) => listed,
        Err(e) => {
            tracing::warn!(model = %model_id, repo = %repo_id, error = %e,
                "Could not list the repository for a vision encoder");
            return None;
        }
    };
    let pinned = state
        .model_registry
        .get_manifest(model_id)
        .and_then(|m| m.mmproj);
    let chosen = match &pinned {
        Some(pin) => listed_with_sha256(&listed, &pin.sha256),
        None => best_encoder_for(&listed, model_file),
    };
    let Some(chosen) = chosen.cloned() else {
        if pinned.is_some() {
            tracing::info!(model = %model_id, repo = %repo_id,
                "The pinned vision encoder is no longer listed — leaving it to the computers that hold it");
        } else {
            tracing::debug!(model = %model_id, repo = %repo_id, "No vision encoder listed");
        }
        note_nothing_to_fetch(model_id);
        return None;
    };
    if chosen.size_bytes > crate::model::manifest::MAX_VISION_ENCODER_BYTES {
        tracing::warn!(model = %model_id, file = %chosen.filename, size = chosen.size_bytes,
            "Vision encoder larger than any this build expects — not fetched");
        note_nothing_to_fetch(model_id);
        return None;
    }
    match super::probe::read_projector_type(repo_id, &chosen.filename).await {
        Ok(kind) if crate::inference::vision::reads_projector_type(&kind) => {}
        Ok(kind) => {
            tracing::info!(model = %model_id, file = %chosen.filename, projector_type = %kind,
                "This model's image reader is a type this build cannot load — not fetched");
            note_nothing_to_fetch(model_id);
            return None;
        }
        Err(e) => {
            tracing::warn!(model = %model_id, file = %chosen.filename, error = %e,
                "Could not read the vision encoder's header");
            return None;
        }
    }
    tracing::info!(model = %model_id, file = %chosen.filename, size = chosen.size_bytes,
        "Fetching the model's image reader from its origin");
    let hash = match super::download::download_vision_encoder(repo_id, &chosen, &model_dir).await {
        Ok(hash) => hash,
        Err(e) => {
            tracing::warn!(model = %model_id, file = %chosen.filename, error = %e,
                "Could not fetch the vision encoder");
            return None;
        }
    };
    let pin = crate::types::MmprojInfo {
        size_bytes: chosen.size_bytes,
        hash,
        sha256: chosen.sha256,
    };
    if pinned.as_ref().is_some_and(|p| p.hash != hash) {
        // The listing's SHA-256 matched the pin's, so the bytes are the pinned
        // file's: a BLAKE3 that differs is the pin's error, and this node's own
        // fetch outranks it as it outranks any gossiped hash.
        tracing::warn!(model = %model_id,
            "The pinned vision encoder's BLAKE3 differs from the file HuggingFace serves under its SHA-256 — keeping the origin's");
    }
    state.accept_origin_vision_encoder(model_id, pin.clone());
    state.announce_shard_acquired(net_tx, &sentinel);
    if let Some(manifest) = state.model_registry.get_manifest(model_id) {
        let _ = net_tx
            .send(crate::types::NetworkCommand::Broadcast(
                crate::types::SwarmMessage::ModelManifest(manifest),
            ))
            .await;
    }
    tracing::info!(model = %model_id, file = %chosen.filename,
        "Image reader fetched from the origin and pinned");
    Some(pin)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn listed(name: &str, byte: u8) -> ListedEncoder {
        ListedEncoder {
            filename: name.to_string(),
            size_bytes: 1,
            sha256: [byte; 32],
        }
    }

    #[test]
    fn a_files_quantization_width_is_read_as_llama_cpp_reads_it() {
        assert_eq!(quant_bits("gemma-3-4b-it-Q4_K_M.gguf"), 4);
        assert_eq!(quant_bits("mmproj-model-f16.gguf"), 16);
        assert_eq!(quant_bits("mmproj-BF16.gguf"), 16);
        assert_eq!(quant_bits("mmproj-F32.gguf"), 32);
        assert_eq!(quant_bits("Q8_0/model-Q8_0-00001-of-00003.gguf"), 8);
        assert_eq!(quant_bits("model.gguf"), 0);
        assert_eq!(quant_bits("dir-x/model"), 0);
    }

    /// llama.cpp's `find_best_sibling` (keyword `mmproj`, no tag): the deepest
    /// folder the model shares first, then the closest width; the first listed
    /// wins a tie.
    #[test]
    fn the_encoder_chosen_is_the_one_llama_cpp_loads_beside_the_model() {
        let files = [
            listed("mmproj-BF16.gguf", 1),
            listed("mmproj-F16.gguf", 2),
            listed("mmproj-F32.gguf", 3),
        ];
        // Q4 → BF16 and F16 tie at 12 away; BF16 is listed first.
        assert_eq!(
            best_encoder_for(&files, "gemma-3-27b-it-Q4_K_M.gguf").map(|f| f.sha256[0]),
            Some(1)
        );
        // A full-width model takes the full-width encoder.
        assert_eq!(
            best_encoder_for(&files, "gemma-3-27b-it-F32.gguf").map(|f| f.sha256[0]),
            Some(3)
        );

        // The model's own folder beats the top level; another folder never counts.
        let foldered = [
            listed("mmproj-f16.gguf", 1),
            listed("Q8_0/mmproj-f16.gguf", 2),
            listed("BF16/mmproj-bf16.gguf", 3),
        ];
        assert_eq!(
            best_encoder_for(&foldered, "Q8_0/model-Q8_0.gguf").map(|f| f.sha256[0]),
            Some(2)
        );
        assert_eq!(
            best_encoder_for(&foldered, "model-Q4_K_M.gguf").map(|f| f.sha256[0]),
            Some(1)
        );
        assert!(best_encoder_for(&foldered[2..], "model-Q4_K_M.gguf").is_none());
    }

    #[test]
    fn a_pin_finds_its_file_by_huggingfaces_sha256() {
        let files = [listed("mmproj-a.gguf", 1), listed("mmproj-b.gguf", 2)];
        assert_eq!(
            listed_with_sha256(&files, &[2; 32]).map(|f| f.filename.as_str()),
            Some("mmproj-b.gguf")
        );
        assert!(listed_with_sha256(&files, &[9; 32]).is_none());
    }

    /// Only a `*mmproj*.gguf` HuggingFace gives a size and a SHA-256 for is a
    /// file a fetch can be held to.
    #[test]
    fn only_a_listed_encoder_with_a_size_and_sha256_can_be_fetched() {
        let sha = "8c0fb064b019a6972856aaae2c7e4792858af3ca4561be2dbf649123ba6c40cb";
        let file = |name: &str, size: Option<u64>, sha: Option<&str>| HfFileInfo {
            rfilename: name.to_string(),
            size,
            lfs: sha.map(|s| super::super::private_types::HfLfs {
                sha256: s.to_string(),
            }),
        };
        let found = encoders_in(vec![
            file("mmproj-model-f16.gguf", Some(851_251_104), Some(sha)),
            file("gemma-3-4b-it-Q4_K_M.gguf", Some(2), Some(sha)),
            file("mmproj-no-sha.gguf", Some(3), None),
            file("mmproj-bad-sha.gguf", Some(3), Some("zz")),
            file("mmproj-no-size.gguf", None, Some(sha)),
            file("mmproj-notes.txt", Some(3), Some(sha)),
        ]);
        assert_eq!(found.len(), 1, "{found:?}");
        assert_eq!(found[0].filename, "mmproj-model-f16.gguf");
        assert_eq!(found[0].size_bytes, 851_251_104);
        assert_eq!(found[0].sha256[..4], [0x8c, 0x0f, 0xb0, 0x64]);
    }
}
