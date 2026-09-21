use crate::error::SwarmError;
use crate::model::shard::ShardStore;

/// Handle an incoming LayerForward from a remote peer: run the local split model
/// segment and send back a LayerResult with either logits (last segment) or
/// hidden-state activations (intermediate segment).
/// Parameters for shard-based model loading.
pub struct ShardLoadParams<'a> {
    pub model_dir: &'a std::path::Path,
    pub shard_store: &'a ShardStore,
    pub model_id: &'a crate::types::ModelId,
    pub layer_start: usize,
    pub layer_end: usize,
    pub is_first: bool,
    pub is_last: bool,
    /// Manifest for this model — provides tensor entries and total size.
    pub manifest: &'a crate::types::ModelManifest,
    /// Load onto the CPU regardless of GPU availability. Set from the worker's
    /// `--gpu-layers 0`, which in turn comes from `inference.gpu_layers`.
    /// Before R146 nothing plumbed this: the loader called
    /// `Device::cuda_if_available(0)` unconditionally and `gpu_layers` was read
    /// only by the legacy llama.cpp executor, so a CUDA build ignored the
    /// setting entirely — configuring `gpu_layers = 0` (the documented
    /// "CPU only" value, and the shipped default) still ran on the GPU.
    pub force_cpu: bool,
}

/// Map an `inference.gpu_layers` value to the loader's `force_cpu` flag.
///
/// The split engine places a worker's entire layer window on one device, so
/// there are only two reachable outcomes. A positive value smaller than the
/// window is *not* silently honoured as partial offload — it means GPU, and
/// the caller logs the discrepancy rather than pretending. True per-layer
/// hybrid placement is tracked in `docs/FUTURE_WORK.md`.
pub fn force_cpu_for(gpu_layers: i32) -> bool {
    gpu_layers == 0
}

/// Try to load a SplitModel from shard files + gguf_header.bin.
/// This is the shard-only loading path — no full GGUF needed.
pub fn try_load_from_shards(
    params: &ShardLoadParams<'_>,
) -> Result<crate::inference::split::SplitModel, SwarmError> {
    let model_dir = params.model_dir;
    let shard_store = params.shard_store;
    let model_id = params.model_id;
    let layer_start = params.layer_start;
    let layer_end = params.layer_end;
    let is_first = params.is_first;
    let is_last = params.is_last;

    // Ensure GGUF header exists (extract from shard_000 if needed)
    if let Err(e) = crate::inference::split::ensure_gguf_header(model_dir) {
        return Err(SwarmError::ModelNotAvailable(crate::types::ModelId(
            format!("Cannot load from shards: {e}"),
        )));
    }

    // Collect available shard files for this model
    let shard_files = shard_store.scan_local_shards(model_id, params.manifest.shard_count);

    if shard_files.is_empty() {
        return Err(SwarmError::ModelNotAvailable(crate::types::ModelId(
            format!(
                "No shard files found for model {} in {}",
                model_id,
                model_dir.display()
            ),
        )));
    }

    // Build tensor entries for each shard file from manifest data.
    // The order must match shard_files (which is sorted by shard index).
    let mut tensor_entries: Vec<Vec<crate::types::ShardTensorEntry>> = shard_files
        .iter()
        .map(|(idx, _)| {
            params
                .manifest
                .shards
                .iter()
                .find(|s| s.index == *idx)
                .map(|s| s.tensors.clone())
                .unwrap_or_default()
        })
        .collect();

    // A manifest that carries no tensor table is not one we cannot use.
    //
    // That table says which tensors live in which shard file and at what
    // offset, and it is **derived, not observed**: `compute_layer_shard_layouts`
    // computes it from the GGUF header and the shard count, deterministically.
    // This node is guaranteed to have that header — `ensure_gguf_header` above
    // returns an error if it cannot produce one, and it can extract one from
    // `shard_000.bin` when no file and no HuggingFace source is available. So
    // the input to the derivation is already on disk by the time we get here.
    //
    // Why it is worth deriving: the table is ~92% of a manifest's bytes and
    // manifests are ~86% of an idle node's gossip, so it is roughly **80% of
    // all gossip traffic** — every byte of it describing something the
    // receiver could work out for itself. This is the half of that fix which
    // lets a receiver cope; the sender stops sending it separately.
    //
    // Deliberately a FALLBACK, not a replacement. A manifest that carries the
    // table is used exactly as before, so no model that works today changes
    // behaviour, and a node is never silently overruled about a model it
    // already holds by a table recomputed from its own header.
    if !shard_files.is_empty() && tensor_entries.iter().all(|t| t.is_empty()) {
        match derive_tensor_entries(model_dir, params.manifest, &shard_files) {
            Some(derived) => {
                tracing::info!(
                    model = %model_id,
                    shards = shard_files.len(),
                    "Rebuilt this model's tensor table from its own header — the \
                     manifest did not carry one"
                );
                tensor_entries = derived;
            }
            None => {
                // Not an error here. A single-shard model legitimately has no
                // table — it is a whole GGUF, and the loader below has an
                // explicit branch for exactly that. Any other failure surfaces
                // in the loader naming the shard it could not read, which is
                // more useful than a message invented at this distance.
                tracing::debug!(
                    model = %model_id,
                    "No tensor table in the manifest and none derivable from the header"
                );
            }
        }
    }

    tracing::info!(
        model = %model_id,
        shards = shard_files.len(),
        layers = format!("[{layer_start}..{layer_end})"),
        force_cpu = params.force_cpu,
        "Loading split model from shard files (no full GGUF)"
    );

    if params.force_cpu {
        return crate::inference::split::SplitModel::load_from_shards_cpu(
            model_dir,
            shard_files,
            &tensor_entries,
            params.manifest.total_size_bytes,
            layer_start,
            layer_end,
            is_first,
            is_last,
        );
    }

    let result = crate::inference::split::SplitModel::load_from_shards(
        model_dir,
        shard_files.clone(),
        &tensor_entries,
        params.manifest.total_size_bytes,
        layer_start,
        layer_end,
        is_first,
        is_last,
    );

    // GPU OOM fallback: retry on CPU so the model is still usable (slower but functional)
    match &result {
        Err(e) if e.to_string().contains("OUT_OF_MEMORY") => {
            tracing::warn!(
                model = %model_id,
                "GPU OOM — retrying model load on CPU"
            );
            crate::inference::split::SplitModel::load_from_shards_cpu(
                model_dir,
                shard_files,
                &tensor_entries,
                params.manifest.total_size_bytes,
                layer_start,
                layer_end,
                is_first,
                is_last,
            )
        }
        _ => result,
    }
}

/// Rebuild the per-shard tensor table from this model's own GGUF header.
///
/// The table is a deterministic function of the header and ONE integer — the
/// shard count the publisher asked `compute_layer_shard_layouts` for. Given
/// that integer, this reproduces the publisher's split exactly, which is what
/// makes the table safe to omit from the wire: it is ~92% of a manifest's
/// bytes and describes nothing the holder of the header cannot work out.
///
/// ⚠ **The requested count is NOT `manifest.shard_count`.** That field records
/// the layout's OUTPUT (`layouts.len()`); the publisher's INPUT was
/// `div_ceil(total_size, shard_size)`, and the algorithm can return fewer
/// shards than asked for. On this node's 15 models the two differ for 5 of
/// them, always by one — feeding the output back in reproduces 10 of 15 and
/// silently mis-places the rest. So the count is SEARCHED for, not assumed.
///
/// **The published per-shard byte sizes are the oracle.** They ride in the
/// compact part of the manifest, so a candidate layout can be checked against
/// what the publisher actually wrote before any of it is used. A table that
/// merely looks plausible is the dangerous outcome: wrong offsets do not fail
/// to load, they read the wrong weights and answer confidently.
///
/// Returns `None` when no candidate reproduces the published sizes — never a
/// partial or unverified table.
fn derive_tensor_entries(
    model_dir: &std::path::Path,
    manifest: &crate::types::ModelManifest,
    shard_files: &[(u32, std::path::PathBuf)],
) -> Option<Vec<Vec<crate::types::ShardTensorEntry>>> {
    let header_path = model_dir.join(crate::model::shard::HEADER_FILENAME);
    let meta = crate::inference::split::GgufTensorMeta::from_gguf_file(&header_path).ok()?;

    // Where the publisher's input most likely sat, tried first so the common
    // case costs one layout computation. `SHARD_SIZE_SEARCH_MAX` bounds the
    // rest: a layout is cheap, but an unbounded search on a malformed manifest
    // is a CPU sink reachable from gossip.
    let likely = manifest
        .total_size_bytes
        .div_ceil(DEFAULT_SHARD_SIZE_BYTES)
        .max(1) as u32;
    let candidates = std::iter::once(likely)
        .chain(1..=SHARD_SIZE_SEARCH_MAX)
        .take(SHARD_SIZE_SEARCH_MAX as usize + 1);

    for requested in candidates {
        let layouts = crate::inference::split::compute_layer_shard_layouts(&meta, requested);
        if layouts.len() != manifest.shards.len() {
            continue;
        }
        // Every shard's size must match what the publisher recorded. This is
        // what turns "a layout" into "the publisher's layout".
        let sizes_agree = manifest
            .shards
            .iter()
            .zip(layouts.iter())
            .all(|(shard, layout)| shard.size_bytes == layout.size_bytes);
        if !sizes_agree {
            continue;
        }
        // Offsets within a shard file run sequentially in layout order — the
        // same rule `build_shard_infos_from_layouts` applies when it writes a
        // table into a manifest. The two must agree or a derived table places
        // tensors where a published one does not.
        let mut by_index: std::collections::HashMap<u32, Vec<crate::types::ShardTensorEntry>> =
            std::collections::HashMap::new();
        for layout in &layouts {
            let mut shard_offset = 0u64;
            let entries = layout
                .tensors
                .iter()
                .map(|(name, gguf_offset, size)| {
                    let entry = crate::types::ShardTensorEntry {
                        name: name.clone(),
                        gguf_offset: *gguf_offset,
                        shard_offset,
                        size: *size,
                    };
                    shard_offset += size;
                    entry
                })
                .collect();
            by_index.insert(layout.index, entries);
        }
        // Every shard present must be covered. A partial answer leaves some
        // shards with a table and others empty, which reads to the loader as a
        // legitimately table-less model and mmaps it wrong.
        let entries: Option<Vec<Vec<crate::types::ShardTensorEntry>>> = shard_files
            .iter()
            .map(|(idx, _)| by_index.get(idx).cloned())
            .collect();
        if let Some(entries) = entries {
            if entries.iter().any(|t| !t.is_empty()) {
                return Some(entries);
            }
        }
    }
    None
}

/// The shard size every node ships with, and therefore the count the publisher
/// most likely asked for. Only a starting guess — the search below does not
/// depend on the publisher having used it.
const DEFAULT_SHARD_SIZE_BYTES: u64 = 512 * 1024 * 1024;

/// Upper bound on the searched shard count. `shard_size_mb` is clamped to a
/// documented range, so a real model cannot need more; the bound exists so a
/// malformed manifest arriving over gossip cannot turn this into a CPU sink.
const SHARD_SIZE_SEARCH_MAX: u32 = 64;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_layers_zero_forces_cpu() {
        // The documented meaning of `gpu_layers = 0`, and — until R146 — a
        // silent no-op for the shard path: the loader called
        // `Device::cuda_if_available(0)` regardless, so a CUDA build ran on
        // the GPU no matter what the config said.
        assert!(force_cpu_for(0));
    }

    #[test]
    fn gpu_layers_auto_and_positive_do_not_force_cpu() {
        assert!(!force_cpu_for(-1), "-1 is auto: use the GPU when present");
        assert!(!force_cpu_for(8));
        assert!(!force_cpu_for(999));
    }

    #[test]
    fn default_gpu_layers_is_auto_not_cpu_only() {
        // Defaulting to 0 would read as "CPU only" and silently drop every
        // existing CUDA node to CPU inference the moment the setting started
        // being honoured.
        let cfg = crate::config::Config::default();
        assert_eq!(cfg.inference.gpu_layers, -1);
        assert!(!force_cpu_for(cfg.inference.gpu_layers));
    }

    /// A DERIVED tensor table must equal the PUBLISHED one, tensor for tensor.
    ///
    /// This is the property the whole idea rests on, and it is the dangerous
    /// kind: a table with a wrong offset does not fail to load, it reads the
    /// wrong weights and answers confidently. Equality is therefore asserted
    /// against real manifests written by the real publisher path, not against
    /// a fixture written by hand from the same assumptions as the code.
    ///
    /// Ignored by default because it needs a populated model directory.
    /// `SWARMLLM_TEST_MODEL_DIR` points at one — the same convention the
    /// real-model integration run uses:
    ///
    /// ```text
    /// SWARMLLM_TEST_MODEL_DIR=~/.local/share/swarmllm/models \
    ///   cargo test --lib derived_tensor_table -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore = "needs a populated model directory (SWARMLLM_TEST_MODEL_DIR)"]
    fn a_derived_tensor_table_matches_the_published_one() {
        use crate::model::manifest::ModelManifestExt;
        let Ok(root) = std::env::var("SWARMLLM_TEST_MODEL_DIR") else {
            panic!("set SWARMLLM_TEST_MODEL_DIR to a models directory");
        };
        let root = std::path::PathBuf::from(shellexpand_home(&root));
        let mut checked = 0usize;
        let mut skipped = 0usize;
        for entry in std::fs::read_dir(&root)
            .expect("models dir is readable")
            .flatten()
        {
            let dir = entry.path();
            if !dir.is_dir() {
                continue;
            }
            let Ok(manifest) = crate::types::ModelManifest::load_from_dir(&dir) else {
                continue;
            };
            // Only models whose manifest actually carries a table can answer
            // the question; the rest have nothing to compare against.
            let published: Vec<(u32, Vec<crate::types::ShardTensorEntry>)> = manifest
                .shards
                .iter()
                .filter(|s| !s.tensors.is_empty())
                .map(|s| (s.index, s.tensors.clone()))
                .collect();
            if published.is_empty() || !dir.join(crate::model::shard::HEADER_FILENAME).exists() {
                skipped += 1;
                continue;
            }
            let files: Vec<(u32, std::path::PathBuf)> = published
                .iter()
                .map(|(i, _)| (*i, dir.join(format!("shard_{i:03}.bin"))))
                .collect();
            let derived = derive_tensor_entries(&dir, &manifest, &files)
                .unwrap_or_else(|| panic!("{}: no table derivable from its header", dir.display()));

            for ((index, want), got) in published.iter().zip(derived.iter()) {
                assert_eq!(
                    want.len(),
                    got.len(),
                    "{}: shard {index} has {} published tensors and {} derived",
                    dir.display(),
                    want.len(),
                    got.len()
                );
                for (w, g) in want.iter().zip(got.iter()) {
                    assert_eq!(
                        w.name,
                        g.name,
                        "{}: shard {index} tensor order",
                        dir.display()
                    );
                    assert_eq!(
                        (w.gguf_offset, w.shard_offset, w.size),
                        (g.gguf_offset, g.shard_offset, g.size),
                        "{}: shard {index} tensor {} placed differently — a wrong \
                         offset reads the wrong weights and does NOT fail",
                        dir.display(),
                        w.name
                    );
                }
            }
            // The size oracle is what makes the SEARCH safe rather than lucky,
            // and the real models cannot exercise it: the first count whose
            // output length matches is also the right split for every one of
            // them, so the oracle never has to reject anything. Verified by
            // sabotage 2026-09-21 — removing it left this test green. So it is
            // exercised deliberately here, by describing a split the publisher
            // did not write and requiring a refusal.
            let mut lying = manifest.clone();
            lying.shards[0].size_bytes = lying.shards[0].size_bytes.wrapping_add(4096);
            assert!(
                derive_tensor_entries(&dir, &lying, &files).is_none(),
                "{}: a manifest whose recorded sizes disagree with every derivable \
                 layout must be refused, not approximated — a wrong offset reads \
                 the wrong weights and does not fail",
                dir.display()
            );

            checked += 1;
            println!(
                "ok  {} ({} shards with a table)",
                dir.display(),
                published.len()
            );
        }
        println!("checked {checked} models, skipped {skipped}");
        assert!(
            checked > 0,
            "no model in {} had both a published table and a header — this test \
             proved nothing, which is a result, not a pass",
            root.display()
        );
    }

    /// `~` in an env var is not expanded by the shell when the value is quoted,
    /// and a silently-wrong path makes the test above skip everything and then
    /// fail its own "proved nothing" check — which is the intended outcome, but
    /// only if the path was genuinely empty.
    #[cfg(test)]
    fn shellexpand_home(p: &str) -> String {
        match p.strip_prefix("~/") {
            Some(rest) => match std::env::var("HOME") {
                Ok(home) => format!("{home}/{rest}"),
                Err(_) => p.to_string(),
            },
            None => p.to_string(),
        }
    }
}
