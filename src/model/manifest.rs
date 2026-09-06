use std::path::Path;

use crate::error::SwarmError;
use crate::types::{Blake3Hash, ModelManifest};

/// Backup-copy artifacts a file manager (or a stray `cp`) leaves on disk.
/// A model id whose final dotted segment is one of these — or that ends in a
/// `~` / " copy" marker — is a local copy of a model, not a real upstream
/// model, and must never be adopted from disk *or* gossip.
///
/// A model's identity has to come from the model, not from whatever a directory
/// happened to be called: a copied folder (`<model>.FULLBACKUP`, `<model>.old`)
/// used to be announced to the swarm under a name that resolves to nothing, so
/// peers recorded shard holders for it and replica counts doubled while no one
/// could ever actually download it. The v0.3.10 fix caught this on the local
/// disk scan, but nothing filtered the name back out once a peer on an older
/// build re-gossiped it — see the raw-pc `.FULLBACKUP` report, 2026-07-23.
const BACKUP_ARTIFACT_SEGMENTS: &[&str] = &[
    "fullbackup",
    "backup",
    "bak",
    "old",
    "orig",
    "copy",
    "save",
    "tmp",
    "temp",
];

/// True if `id` looks like a local backup/copy of a model rather than a real
/// model identity. Used at every point a model id enters the registry
/// (`register_manifest`, gossip ingress) so a copied-folder name can neither be
/// stored nor propagated. Deliberately conservative: it only matches the LAST
/// dotted segment against a fixed keyword list, so a legitimate id carrying
/// dots from its source filename — `tinyllama-1.1b-chat-v1.0.q4-k-m` — is never
/// caught.
pub fn is_backup_artifact_id(id: &str) -> bool {
    let lower = id.trim().to_ascii_lowercase();
    if lower.is_empty() {
        return false;
    }
    // `model~`, "Model copy", "model - copy", "model (copy)" — desktop copies.
    if lower.ends_with('~')
        || lower.ends_with(" copy")
        || lower.ends_with("-copy")
        || lower.ends_with("(copy)")
    {
        return true;
    }
    // Final dotted segment is a known backup keyword: `model.FULLBACKUP`,
    // `model.old`. Guarded to the last segment only.
    match lower.rsplit('.').next() {
        Some(last) => BACKUP_ARTIFACT_SEGMENTS.contains(&last),
        None => false,
    }
}

/// Extension methods for ModelManifest (defined in swarmllm-types crate).
pub trait ModelManifestExt {
    fn load_from_dir(dir: &Path) -> Result<ModelManifest, SwarmError>;
    fn verify_hash(&self) -> Result<(), SwarmError>;
    fn verify_hash_strict(&self) -> Result<(), SwarmError>;
    fn compute_hash(&self) -> Blake3Hash;
    fn save_to_dir(&self, dir: &Path) -> Result<(), SwarmError>;
}

/// Map a GGUF `general.architecture` string to our `ModelArchitecture` enum.
/// Unknown architectures default to Llama (which shares the standard transformer
/// manifest layout). Logs a warning for unrecognized values.
///
/// Delegates string parsing to `ModelArch::from_gguf_arch` so the two enums stay
/// in sync. Handles the bare `"phi"` alias that split inference doesn't recognize.
pub fn gguf_arch_to_model_architecture(arch: &str) -> crate::types::ModelArchitecture {
    // Bare `"phi"` only appears in the manifest path (split inference requires phi3).
    if arch == "phi" {
        return crate::types::ModelArchitecture::Phi;
    }
    let detected = crate::inference::model_arch::ModelArch::from_gguf_arch(arch);
    if matches!(
        detected,
        crate::inference::model_arch::ModelArch::Unknown(_)
    ) {
        // Deliberately does NOT say "defaulting to Llama". That is true of this
        // manifest field only — it exists so a model can still be catalogued,
        // sized and gossiped — and reads as though inference will proceed with
        // Llama handling. It will not: `split::loader` refuses an unrecognised
        // architecture outright (`ModelArch::is_supported`). A tester reading
        // this line reasonably concluded a Qwen3-MoE checkpoint would be run as
        // a dense model and produce garbage; in fact it is rejected at load.
        tracing::warn!(
            arch,
            "This model's architecture is not one this build can run — it can be \
             catalogued and shared, but loading it for inference will be refused. \
             The manifest records it as Llama-like for sizing purposes only."
        );
    }
    detected.to_manifest_architecture()
}

/// Parameters for building a ModelManifest from GGUF metadata.
pub struct ManifestFromGguf {
    pub id: crate::types::ModelId,
    pub name: String,
    pub architecture: crate::types::ModelArchitecture,
    pub num_layers: u32,
    pub total_size_bytes: u64,
    pub shard_count: u32,
    pub shards: Vec<crate::types::ShardInfo>,
    pub publisher: crate::types::NodeId,
}

/// Build a `ModelManifest` with standard zero-defaults for fields that are
/// filled in later (params, quantization, tokenizer hash, license).
/// Computes and sets `manifest_hash` automatically.
pub fn build_manifest_from_gguf(p: ManifestFromGguf) -> ModelManifest {
    let mut manifest = ModelManifest {
        id: p.id,
        name: p.name,
        architecture: p.architecture,
        num_layers: p.num_layers,
        num_params_billions: 0.0,
        quantization: crate::types::Quantization::Q4KM,
        total_size_bytes: p.total_size_bytes,
        shard_count: p.shard_count,
        shards: p.shards,
        tokenizer_hash: [0u8; 32],
        manifest_hash: [0u8; 32],
        publisher: p.publisher,
        publish_date: chrono::Utc::now(),
        license: "Unknown".to_string(),
        mmproj: None,
    };
    manifest.manifest_hash = manifest.compute_hash();
    manifest
}

impl ModelManifestExt for ModelManifest {
    fn load_from_dir(dir: &Path) -> Result<ModelManifest, SwarmError> {
        let manifest_path = dir.join(crate::model::shard::MANIFEST_FILENAME);
        if !manifest_path.exists() {
            return Err(SwarmError::ModelNotAvailable(crate::types::ModelId(
                format!("Manifest not found: {}", manifest_path.display()),
            )));
        }

        let contents = std::fs::read_to_string(&manifest_path).map_err(SwarmError::Io)?;
        let manifest: ModelManifest =
            serde_json::from_str(&contents).map_err(SwarmError::Serialization)?;

        tracing::debug!(
            model = %manifest.id,
            shard_count = manifest.shard_count,
            dir_path = %dir.display(),
            "DIAG: load_from_dir manifest loaded"
        );

        Ok(manifest)
    }

    /// Verify the manifest hash by recomputing it from the manifest content
    /// (excluding the manifest_hash field itself).
    ///
    /// Allows zero-hash manifests (not yet computed, e.g. from local HF downloads).
    /// For network-received manifests, use `verify_hash_strict()` instead.
    fn verify_hash(&self) -> Result<(), SwarmError> {
        // Allow manifests with a zero hash (not yet computed, e.g. from partial
        // HF downloads before hash is set). Local-only — see verify_hash_strict.
        if self.manifest_hash == [0u8; 32] {
            return Ok(());
        }
        let computed = self.compute_hash();
        if computed != self.manifest_hash {
            tracing::debug!(
                model = %self.id,
                expected = %hex::encode(self.manifest_hash),
                actual = %hex::encode(computed),
                "DIAG: verify_hash FAILED"
            );
            return Err(SwarmError::ShardIntegrity {
                expected: hex::encode(self.manifest_hash),
                actual: hex::encode(computed),
            });
        }
        Ok(())
    }

    /// Strict hash verification for network-received manifests.
    /// Rejects zero-hash manifests to prevent gossip-based poisoning.
    fn verify_hash_strict(&self) -> Result<(), SwarmError> {
        if self.manifest_hash == [0u8; 32] {
            return Err(SwarmError::ShardIntegrity {
                expected: "non-zero manifest hash".into(),
                actual: "zero hash (unsigned manifest)".into(),
            });
        }
        let computed = self.compute_hash();
        if computed != self.manifest_hash {
            return Err(SwarmError::ShardIntegrity {
                expected: hex::encode(self.manifest_hash),
                actual: hex::encode(computed),
            });
        }
        Ok(())
    }

    /// Compute the BLAKE3 hash of the manifest content.
    ///
    /// Hashes a canonical representation that excludes the manifest_hash field.
    fn compute_hash(&self) -> Blake3Hash {
        let mut hasher = blake3::Hasher::new();
        hasher.update(self.id.0.as_bytes());
        hasher.update(self.name.as_bytes());
        // Include publisher, license, architecture, quantization in hash
        hasher.update(&self.publisher.0);
        hasher.update(self.license.as_bytes());
        hasher.update(format!("{:?}", self.architecture).as_bytes());
        hasher.update(format!("{:?}", self.quantization).as_bytes());
        hasher.update(&self.num_layers.to_le_bytes());
        hasher.update(&self.num_params_billions.to_le_bytes());
        hasher.update(&self.total_size_bytes.to_le_bytes());
        hasher.update(self.publish_date.to_rfc3339().as_bytes());
        hasher.update(&self.shard_count.to_le_bytes());
        for shard in &self.shards {
            hasher.update(&shard.index.to_le_bytes());
            hasher.update(&shard.layer_range.0.to_le_bytes());
            hasher.update(&shard.layer_range.1.to_le_bytes());
            hasher.update(&shard.size_bytes.to_le_bytes());
            hasher.update(&shard.hash);
            for entry in &shard.tensors {
                hasher.update(entry.name.as_bytes());
                hasher.update(&entry.gguf_offset.to_le_bytes());
                hasher.update(&entry.shard_offset.to_le_bytes());
                hasher.update(&entry.size.to_le_bytes());
            }
        }
        hasher.update(&self.tokenizer_hash);
        *hasher.finalize().as_bytes()
    }

    /// Save the manifest to a model directory as `manifest.json`.
    fn save_to_dir(&self, dir: &Path) -> Result<(), SwarmError> {
        std::fs::create_dir_all(dir).map_err(SwarmError::Io)?;
        let json = serde_json::to_string_pretty(self).map_err(SwarmError::Serialization)?;
        // Atomic write: write to temp file then rename to prevent corruption on kill/crash
        let tmp_path = dir.join(format!("{}.tmp", crate::model::shard::MANIFEST_FILENAME));
        std::fs::write(&tmp_path, json).map_err(SwarmError::Io)?;
        std::fs::rename(&tmp_path, dir.join(crate::model::shard::MANIFEST_FILENAME))
            .map_err(SwarmError::Io)?;
        Ok(())
    }
}

/// What to do with a P2P shard transfer that has just finished arriving.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum P2pShardAcceptance {
    /// The manifest carries a hash — check the bytes against it.
    Verify,
    /// No hash to check against, but we know where the model came from.
    /// Discard the peer's copy and fetch this shard from the origin, which
    /// both supplies trustworthy bytes and teaches us the real hash.
    FetchFromOrigin,
    /// No hash and no origin. Accept, but the shard is UNCHECKED — it must be
    /// reported as such and verified once a hash reaches us.
    AcceptUnchecked,
}

/// The single policy for "may these bytes be accepted, and on what basis?".
///
/// **"No hash available" is not a verdict.** Treating it as one turned the
/// accept path into a corruption PROPAGATION channel: a shard taken on trust is
/// recorded as held and re-served to other peers, so one bad copy becomes
/// several, and every extra holder makes it look more authoritative. Measured
/// 2026-08-24 — two independent peers serving byte-identical corrupt bytes for
/// one shard, proven corrupt against the origin repo (gotcha #382).
///
/// The way out is that a node is not limited to what its peers tell it: it
/// knows the model's origin. Fetching the shard from there supplies bytes worth
/// trusting AND the real hash, which then spreads by gossip (see
/// `merge_known_shard_hashes`) so later transfers verify normally. So the
/// fallback is self-limiting — one origin download for a model whose hashes
/// nobody knew yet, not a permanent retreat from P2P.
///
/// `AcceptUnchecked` remains for a model with no origin to consult (published
/// locally by a peer). Refusing outright was shipped once and soak-caught: it
/// makes such a model impossible to acquire at all.
///
/// **`origin_fetch_available` means the fetch will actually HAPPEN, not merely
/// that an origin exists.** Passing a bare "we know the repo id" would discard a
/// perfectly usable copy and replace it with nothing. **Never throw away data
/// you cannot actually replace** — so every condition that can stop the fetch
/// has to be folded into that argument by the caller. Offline mode is one
/// (`trigger_download` skips the HuggingFace branch by design). Auto-manage
/// being switched off is deliberately NOT one: the fetch runs outside that gate,
/// because it means "do not decide what to fetch for me", not "abandon a shard I
/// already asked for".
pub fn classify_p2p_shard_acceptance(
    manifest_has_hash: bool,
    origin_fetch_available: bool,
) -> P2pShardAcceptance {
    match (manifest_has_hash, origin_fetch_available) {
        (true, _) => P2pShardAcceptance::Verify,
        (false, true) => P2pShardAcceptance::FetchFromOrigin,
        (false, false) => P2pShardAcceptance::AcceptUnchecked,
    }
}

/// Adopt shard hashes we already know for shards `incoming` leaves as
/// placeholders, returning how many were recovered.
///
/// A shard's BLAKE3 hash is a property of the MODEL, not of this node — but a
/// manifest is generated from what the generating node happens to hold on disk
/// (`build_shard_infos_from_layouts` hashes a shard file only when it exists,
/// and writes all-zero otherwise). So every partial holder publishes a manifest
/// that is authoritative for its own shards and blank for the rest.
///
/// The registry's `insert` is blind, so a blank used to overwrite a hash we
/// already held — and that is how a node ends up with nothing to verify a
/// download against: the P2P accept path checks a completed transfer ONLY when
/// the manifest carries a non-zero hash, so a placeholder means the bytes are
/// taken on trust, recorded as held, and re-served to other peers unchecked.
/// Measured on the live node 2026-08-24: five shards fetched from peers against
/// a manifest carrying placeholders for exactly those five, one of them corrupt
/// and surfacing only hours later, after a restart.
///
/// Merging makes hash knowledge MONOTONIC — a hash may go from unknown to
/// known, or be replaced by a differing known one (a genuine re-publish), but
/// never back to unknown. Note the converse is deliberately NOT protected: a
/// real incoming hash still wins over a real stored one, exactly as before.
pub fn merge_known_shard_hashes(incoming: &mut ModelManifest, known: &ModelManifest) -> usize {
    let mut recovered = 0usize;
    for shard in incoming.shards.iter_mut() {
        if shard.hash != [0u8; 32] {
            continue;
        }
        if let Some(prev) = known
            .shards
            .iter()
            .find(|s| s.index == shard.index && s.hash != [0u8; 32])
        {
            shard.hash = prev.hash;
            recovered += 1;
        }
    }
    recovered
}

/// Build ShardInfo entries from `LayerShardLayout` computed by `compute_layer_shard_layouts`.
///
/// For each layout, hashes the on-disk shard file (if present) and builds the
/// `ShardTensorEntry` list from the layout's tensor data.
pub fn build_shard_infos_from_layouts(
    model_dir: &Path,
    layouts: &[crate::inference::split::LayerShardLayout],
) -> Vec<crate::types::ShardInfo> {
    layouts
        .iter()
        .map(|layout| {
            let shard_path = model_dir.join(format!("shard_{:03}.bin", layout.index));
            let hash = if shard_path.exists() {
                crate::model::shard::hash_file_blake3(&shard_path).unwrap_or([0u8; 32])
            } else {
                [0u8; 32]
            };
            let file_size = std::fs::metadata(&shard_path)
                .map(|m| m.len())
                .unwrap_or(layout.size_bytes);

            // Build tensor entries with sequential shard-local offsets
            let mut shard_offset = 0u64;
            let tensors: Vec<crate::types::ShardTensorEntry> = layout
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

            crate::types::ShardInfo {
                index: layout.index,
                layer_range: (layout.layer_start, layout.layer_end),
                size_bytes: file_size,
                hash,
                tensors,
            }
        })
        .collect()
}

// ── Repeat-rejection suppression ────────────────────────────────────────────
//
// Lives here rather than beside any one caller: a manifest we have decided
// against is a manifest-identity fact, and there are now two places that reach
// that verdict — the gossip ingress (hash verification) and the registry (a
// different BUILD of the same model). Two suppression maps would each report
// the other's rejections as new.

/// A manifest we have already verified and rejected, so the next identical
/// copy costs neither a re-hash nor another log line.
///
/// Keyed by `(model, the manifest hash we rejected)`. Keying on the HASH is
/// what keeps this self-correcting: if the publisher fixes its copy the hash
/// changes, the key misses, and the new manifest is verified normally. It can
/// therefore never latch a model into permanent rejection.
pub(crate) struct RejectedManifest {
    last_logged: std::time::Instant,
    /// Rejections swallowed since then, reported with the next emitted line so
    /// the rate is visible even though the repetition is not.
    suppressed: u64,
}

/// How long to hold an identical manifest rejection before logging it again.
///
/// Measured on the live node 2026-08-26: two peers re-gossiped one contradicted
/// `qwen2.5-coder` manifest every 30 s, producing **4709 WARN lines** — 14% of
/// every warning in a month-long log — for a condition correctly handled the
/// first time. An hour against a 30-second cadence turns 120 lines into 1.
pub(crate) const REJECTED_MANIFEST_LOG_WINDOW: std::time::Duration =
    std::time::Duration::from_secs(3600);

/// Bound on the rejection map.
///
/// The bound is only safe because entries that have outlived the window are
/// evicted to make room (see `note_rejection_in`). Without that this map is
/// insert-only, and the cap becomes a CLIFF rather than a limit: once full,
/// every key not already present logs on EVERY rejection for the life of the
/// process, which is precisely the flood the window exists to stop — and it
/// reports `suppressed_since_last=0` while doing it, so the log claims nothing
/// was swallowed.
///
/// It is not a spam-only threshold. A key is `(model, shard, claimed hash)`,
/// and gotcha #406 established that several independent builds of one popular
/// model are live on the swarm at once, each with its own per-shard hashes —
/// so the count grows with models x shards x builds. A peer sending fabricated
/// hashes reaches it deliberately in one round.
pub(crate) const MAX_REJECTED_MANIFESTS: usize = 512;

/// What is being repeatedly rejected.
///
/// One rate limiter covers both because they are the same event seen at two
/// granularities: a peer re-gossiping a manifest we disagree with. Keying them
/// separately would let the finer-grained one flood while the coarser one was
/// quiet, which is exactly what happened — the manifest arm was rate-limited on
/// 2026-08-26 and the per-shard arm was not, so the same publisher on the same
/// 30 s cadence produced **16-28 WARN lines a minute, indefinitely** (one per
/// shard), measured live 2026-08-29.
#[derive(Clone, PartialEq, Eq, Hash)]
pub(crate) enum RejectionKey {
    /// A whole manifest describing a different build.
    Manifest {
        model: crate::types::ModelId,
        manifest_hash: [u8; 32],
    },
    /// One shard hash contradicting what we took from the model's origin.
    ShardHash {
        model: crate::types::ModelId,
        shard: u32,
        claimed: [u8; 32],
    },
}

pub(crate) static REJECTED_MANIFESTS: std::sync::LazyLock<
    dashmap::DashMap<RejectionKey, RejectedManifest>,
> = std::sync::LazyLock::new(dashmap::DashMap::new);

/// Should this rejection be logged, and how many were swallowed since the last
/// one? `None` means "already known and still inside the quiet window".
///
/// Returns `Some(suppressed_count)` the first time a given (model, hash) is
/// rejected and once per window thereafter.
pub(crate) fn note_manifest_rejection(key: RejectionKey) -> Option<u64> {
    note_rejection_in(
        &REJECTED_MANIFESTS,
        key,
        REJECTED_MANIFEST_LOG_WINDOW,
        MAX_REJECTED_MANIFESTS,
    )
}

/// The policy, over an explicit map, window and bound.
///
/// Split out from the globals so the full-map behaviour can be tested against a
/// four-entry map and a zero-length window, rather than against a 512-entry
/// process-wide map and an hour of wall clock — and so a test cannot perturb the
/// real map that other tests share.
fn note_rejection_in(
    map: &dashmap::DashMap<RejectionKey, RejectedManifest>,
    key: RejectionKey,
    window: std::time::Duration,
    max: usize,
) -> Option<u64> {
    if let Some(mut prev) = map.get_mut(&key) {
        if prev.last_logged.elapsed() >= window {
            let n = prev.suppressed;
            prev.last_logged = std::time::Instant::now();
            prev.suppressed = 0;
            return Some(n);
        }
        prev.suppressed = prev.suppressed.saturating_add(1);
        return None;
    }
    if map.len() >= max {
        // Full. Drop the entries that have outlived the window before giving
        // up. Such an entry suppresses NOTHING — the next rejection matching it
        // would be logged anyway — so it is pure occupancy, and evicting it
        // costs at most the suppressed-count on a line that was going to be
        // emitted regardless.
        //
        // This is what keeps the bound a limit rather than a cliff. No guard is
        // held here (the `get_mut` above returns inside its own block), so
        // taking the map's shard locks is safe.
        map.retain(|_, v| v.last_logged.elapsed() < window);
        if map.len() >= max {
            // Still full, so every entry is actively suppressing something.
            // Log rather than silently dropping the report.
            return Some(0);
        }
    }
    map.insert(
        key,
        RejectedManifest {
            last_logged: std::time::Instant::now(),
            suppressed: 0,
        },
    );
    Some(0)
}

/// Build a `ShardAnnounce` — the ONE place that struct is constructed.
///
/// **Every announcement carries a build tag per shard**, taken from this
/// node's own manifest, so a receiver can tell a holder of THIS build from a
/// holder of a different GGUF build sharing the same model id (gotcha #406).
/// Getting that from a call site would mean eight copies of the derivation and
/// eight chances to send an announcement that silently claims nothing; a
/// receiver cannot distinguish "older peer" from "newer peer that forgot", so
/// a missed site is invisible.
///
/// Adding a field to `ShardAnnounce` extends this helper, not the call sites —
/// the same contract `build_spec_verify_forward` keeps for `LayerForward`.
/// `shard_announce_is_built_in_one_place` in `tests/repo_consistency.rs` fails
/// the build on a bare struct literal anywhere else.
pub fn shard_announce(
    registry: &crate::model::registry::ModelRegistry,
    node_id: crate::types::NodeId,
    shards: Vec<crate::types::ShardId>,
    complete_for_models: Vec<crate::types::ModelId>,
) -> crate::types::ShardAnnounce {
    let shard_builds = shards
        .iter()
        .map(|s| registry.expected_build_tag(s))
        .collect();
    crate::types::ShardAnnounce {
        node_id,
        shards,
        timestamp: chrono::Utc::now(),
        complete_for_models,
        shard_builds,
    }
}

#[cfg(test)]
mod tests {
    use super::{classify_p2p_shard_acceptance as classify, P2pShardAcceptance as A};
    use super::{note_rejection_in, RejectionKey};
    use std::time::Duration;

    fn shard_key(shard: u32) -> RejectionKey {
        RejectionKey::ShardHash {
            model: crate::types::ModelId("m".to_string()),
            shard,
            claimed: [0u8; 32],
        }
    }

    /// A full map must still make room for a new key by dropping entries that
    /// have outlived the window, or the bound is a cliff: every key not already
    /// present logs on every rejection, for ever.
    ///
    /// Zero-length window, so every entry is inert the moment it is written —
    /// the same state the real map reaches an hour after an entry's last line,
    /// without an hour of wall clock.
    #[test]
    fn a_full_rejection_map_makes_room_by_dropping_entries_that_suppress_nothing() {
        let map = dashmap::DashMap::new();
        for shard in 0..4 {
            note_rejection_in(&map, shard_key(shard), Duration::ZERO, 4);
        }
        assert_eq!(map.len(), 4, "map should be full");

        let fresh = shard_key(99);
        note_rejection_in(&map, fresh.clone(), Duration::ZERO, 4);

        // Without the eviction this returns early and the key is never stored,
        // so the next rejection of it logs again, and the one after that.
        assert!(
            map.contains_key(&fresh),
            "a new key must be admitted to a full map of inert entries"
        );
    }

    /// The converse, and the reason eviction is filtered rather than wholesale:
    /// an entry still inside its window is actively swallowing repeats, so it
    /// must survive. Dropping it would let the same rejection log twice.
    #[test]
    fn entries_still_inside_their_window_are_never_evicted() {
        let map = dashmap::DashMap::new();
        let window = Duration::from_secs(3600);
        for shard in 0..4 {
            note_rejection_in(&map, shard_key(shard), window, 4);
        }

        let fresh = shard_key(99);
        assert_eq!(
            note_rejection_in(&map, fresh.clone(), window, 4),
            Some(0),
            "a full map of live entries still reports rather than dropping silently"
        );
        assert_eq!(map.len(), 4, "live entries must not be evicted");
        assert!(!map.contains_key(&fresh));
    }

    /// The suppression contract the window exists for, checked on the same
    /// helper the production path uses: the first rejection logs, identical
    /// repeats inside the window do not, and the count comes back with the next
    /// emitted line.
    #[test]
    fn repeats_inside_the_window_are_swallowed_and_counted() {
        let map = dashmap::DashMap::new();
        let window = Duration::from_secs(3600);
        let k = shard_key(0);

        assert_eq!(note_rejection_in(&map, k.clone(), window, 8), Some(0));
        assert_eq!(note_rejection_in(&map, k.clone(), window, 8), None);
        assert_eq!(note_rejection_in(&map, k.clone(), window, 8), None);

        // Window elapsed: the two swallowed repeats are reported.
        assert_eq!(note_rejection_in(&map, k, Duration::ZERO, 8), Some(2));
    }

    /// A distinct key is its own budget — eight shards disagreeing is eight
    /// genuine reports, not one. Guards the granularity the shared limiter was
    /// given so neither arm can silence the other.
    #[test]
    fn each_shard_keeps_its_own_suppression_budget() {
        let map = dashmap::DashMap::new();
        let window = Duration::from_secs(3600);
        for shard in 0..8 {
            assert_eq!(
                note_rejection_in(&map, shard_key(shard), window, 64),
                Some(0),
                "shard {shard} is a different disagreement"
            );
        }
        assert_eq!(map.len(), 8);
    }

    /// Bytes we cannot check are never simply accepted when the origin is
    /// reachable — that is what let a corrupt shard spread between peers.
    #[test]
    fn an_uncheckable_shard_is_fetched_from_the_origin_instead() {
        assert_eq!(classify(false, true), A::FetchFromOrigin);
        // The second argument means the fetch will actually happen. When it
        // cannot (auto-manage off, so nothing performs the fallback), the copy
        // in hand is kept rather than discarded for nothing.
        assert_eq!(classify(false, false), A::AcceptUnchecked);
        // With a hash present the origin is irrelevant — check locally.
        assert_eq!(classify(true, true), A::Verify);
        assert_eq!(classify(true, false), A::Verify);
        // No hash AND no origin: accepting is the only way such a model can be
        // acquired, but it must be reported as unchecked, never as verified.
        assert_eq!(classify(false, false), A::AcceptUnchecked);
    }

    use super::is_backup_artifact_id;
    use crate::types::*;

    #[test]
    fn backup_artifact_ids_are_rejected() {
        // The exact name from the raw-pc report, plus common copy suffixes.
        assert!(is_backup_artifact_id(
            "meta-llama-3.1-8b-instruct-q4-k-m.FULLBACKUP"
        ));
        assert!(is_backup_artifact_id("some-model.old"));
        assert!(is_backup_artifact_id("some-model.bak"));
        assert!(is_backup_artifact_id("some-model.backup"));
        assert!(is_backup_artifact_id("some-model.orig"));
        assert!(is_backup_artifact_id("some-model.COPY")); // case-insensitive
        assert!(is_backup_artifact_id("some-model.tmp"));
        assert!(is_backup_artifact_id("some-model~"));
        assert!(is_backup_artifact_id("some model copy"));
        assert!(is_backup_artifact_id("some-model-copy"));
        assert!(is_backup_artifact_id("some-model(copy)"));
    }

    #[test]
    fn real_model_ids_are_kept() {
        // Real ids carry dots from their source filename — the last segment is a
        // quant tag, not a backup keyword — and must never be caught.
        assert!(!is_backup_artifact_id("tinyllama-1.1b-chat-v1.0.q4-k-m"));
        assert!(!is_backup_artifact_id("qwen2.5-0.5b-instruct-fp16"));
        assert!(!is_backup_artifact_id("meta-llama-3.1-8b-instruct-q4-k-m"));
        assert!(!is_backup_artifact_id("llama-3.2-3b-instruct-q4-k-m"));
        assert!(!is_backup_artifact_id("gemma-2-2b-it-q4-k-m"));
        assert!(!is_backup_artifact_id("")); // empty is not an artifact
    }

    pub(super) fn test_manifest() -> ModelManifest {
        ModelManifest {
            id: ModelId("test-model".into()),
            name: "Test Model".into(),
            architecture: ModelArchitecture::Llama,
            num_layers: 2,
            num_params_billions: 0.001,
            quantization: Quantization::Q4KM,
            total_size_bytes: 1024,
            shard_count: 1,
            shards: vec![ShardInfo {
                index: 0,
                layer_range: (0, 2),
                size_bytes: 1024,
                hash: [0u8; 32],
                tensors: vec![],
            }],
            tokenizer_hash: [0u8; 32],
            manifest_hash: [0u8; 32],
            publisher: NodeId([0u8; 32]),
            publish_date: chrono::Utc::now(),
            license: "MIT".into(),
            mmproj: None,
        }
    }

    #[test]
    fn compute_hash_is_deterministic() {
        let manifest = test_manifest();
        let hash1 = manifest.compute_hash();
        let hash2 = manifest.compute_hash();
        assert_eq!(hash1, hash2);
    }

    #[test]
    fn save_and_load_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let mut manifest = test_manifest();
        manifest.manifest_hash = manifest.compute_hash();

        manifest.save_to_dir(dir.path()).unwrap();
        let loaded = ModelManifest::load_from_dir(dir.path()).unwrap();

        assert_eq!(loaded.id, manifest.id);
        assert_eq!(loaded.name, manifest.name);
        assert_eq!(loaded.shard_count, manifest.shard_count);
    }

    #[test]
    fn verify_hash_with_correct_hash() {
        let mut manifest = test_manifest();
        manifest.manifest_hash = manifest.compute_hash();
        assert!(manifest.verify_hash().is_ok());
    }

    #[test]
    fn verify_hash_with_zero_hash_allowed() {
        let manifest = test_manifest(); // manifest_hash is all zeros
                                        // Zero hash should be allowed (manifest not yet signed)
        assert!(manifest.verify_hash().is_ok());
    }

    #[test]
    fn verify_hash_with_wrong_nonzero_hash() {
        let mut manifest = test_manifest();
        manifest.manifest_hash = [1u8; 32]; // non-zero but wrong
        let result = manifest.verify_hash();
        assert!(result.is_err());
    }

    #[test]
    fn verify_hash_strict_rejects_zero_hash() {
        // A network-received manifest with an unsigned zero hash must
        // be rejected by the strict path; the lenient `verify_hash`
        // accepts zero (for local pre-signing flows).
        let manifest = test_manifest();
        assert_eq!(manifest.manifest_hash, [0u8; 32]);
        assert!(manifest.verify_hash().is_ok());
        assert!(
            manifest.verify_hash_strict().is_err(),
            "strict path must reject unsigned (zero) hash"
        );
    }

    #[test]
    fn verify_hash_strict_rejects_wrong_nonzero_hash() {
        let mut manifest = test_manifest();
        manifest.manifest_hash = [1u8; 32];
        let err = manifest
            .verify_hash_strict()
            .expect_err("wrong hash must be rejected");
        // Verify the error type is ShardIntegrity (mapped to 404 for the API).
        assert!(matches!(
            err,
            crate::error::SwarmError::ShardIntegrity { .. }
        ));
    }

    #[test]
    fn verify_hash_strict_accepts_correct_hash() {
        let mut manifest = test_manifest();
        manifest.manifest_hash = manifest.compute_hash();
        assert!(manifest.verify_hash_strict().is_ok());
    }

    #[test]
    fn shard_path_format() {
        let store = crate::model::shard::ShardStore::new(std::path::Path::new("/data"));
        let path = store.shard_path(&crate::types::ModelId("llama3-70b".into()), 5);
        assert_eq!(
            path.to_string_lossy(),
            "/data/models/llama3-70b/shard_005.bin"
        );
    }
}

#[cfg(test)]
mod wire_roundtrip_tests {
    use super::*;

    /// A manifest that has crossed the wire must still verify.
    ///
    /// `compute_hash` folds in `publish_date.to_rfc3339()`, and the gossip path
    /// is `serde_json`. If that round trip does not preserve the timestamp
    /// exactly, every manifest a peer sends fails `verify_hash_strict` on
    /// arrival and the model becomes undiscoverable through gossip — while
    /// verifying perfectly on the node that built it.
    #[test]
    fn a_manifest_still_verifies_after_a_json_round_trip() {
        let mut m = super::tests::test_manifest();
        // A timestamp with sub-second precision, which is what `Utc::now()`
        // produces and therefore what a real publisher stamps.
        m.publish_date = chrono::DateTime::from_timestamp_nanos(1_788_659_061_123_456_789);
        m.manifest_hash = m.compute_hash();
        m.verify_hash_strict().expect("must verify before the wire");

        let wire = serde_json::to_vec(&m).expect("serialize");
        let back: crate::types::ModelManifest = serde_json::from_slice(&wire).expect("deserialize");

        assert_eq!(
            back.publish_date, m.publish_date,
            "the timestamp did not survive the wire"
        );
        assert_eq!(
            hex::encode(back.compute_hash()),
            hex::encode(m.compute_hash()),
            "the manifest hashes to something different after a round trip, so a \
             peer would reject a manifest its publisher considers valid"
        );
        back.verify_hash_strict()
            .expect("a manifest that crossed the wire must still verify");
    }
}
