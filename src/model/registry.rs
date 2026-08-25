use std::collections::{HashMap, HashSet};
use std::time::Instant;

use dashmap::DashMap;

use crate::error::SwarmError;
use crate::model::manifest::ModelManifestExt;
use crate::storage::db::Database;
use crate::types::{ModelId, ModelManifest, NodeId, ShardId, MMPROJ_SHARD_INDEX};

/// Maximum number of holders tracked per shard in the in-memory cache.
/// At 50K nodes, this bounds memory to O(shards × 50) instead of O(shards × nodes).
/// DHT provider queries fill in the rest on demand.
const MAX_HOLDERS_PER_SHARD: usize = 50;

/// How long a holder's own retraction outranks a DHT provider record naming it.
///
/// Sized to exceed libp2p-kad's provider-record lifetime (24 h by default, with
/// republication at 12 h): the record is what keeps resurrecting the claim, so
/// honouring the retraction for anything less leaves a window where the stale
/// record wins again. A node that genuinely re-acquires the shard does not wait
/// this out — its own announcement clears the entry immediately.
const RETRACTION_HONOURED_SECS: u64 = 26 * 60 * 60;

/// Cap on remembered retractions; entries past `RETRACTION_HONOURED_SECS` are
/// swept when it is reached, so the map stays bounded without its own timer.
const MAX_RETRACTED_CLAIMS: usize = 10_000;

/// Thread-safe registry of known models and shard locations.
///
/// Uses DashMap for concurrent access from multiple daemon tasks.
/// Shard holder tracking is bounded: at most `MAX_HOLDERS_PER_SHARD` holders
/// are cached per shard. The local node is never evicted. For accurate holder
/// counts at scale, use DHT provider queries via `QueryShardProviders`.
/// DB tree holding shard hashes derived from the model's ORIGIN.
pub const ORIGIN_VERIFIED_TREE: &str = "origin_verified_hashes";

/// Reacts to a manifest update: `(manifest, persist, recheck_shards)`.
///
/// `persist` — this registry enriched the manifest's shard hashes, so the result
/// must outlive the process. `recheck_shards` — shard indices THIS NODE HOLDS
/// whose expected hash just changed, and whose bytes must therefore be checked
/// against the new reference.
type ManifestUpdateHook = Box<dyn Fn(&ModelManifest, bool, &[u32]) + Send + Sync>;

pub struct ModelRegistry {
    /// Known model manifests, keyed by model ID.
    manifests: DashMap<ModelId, ModelManifest>,
    /// Shard location tracking: which nodes hold which shards.
    /// HashMap value maps NodeId → last_seen timestamp for LRU eviction.
    /// Bounded at MAX_HOLDERS_PER_SHARD entries per shard.
    shard_holders: DashMap<ShardId, HashMap<NodeId, Instant>>,
    /// Reverse index: which shards each node holds.
    /// Maintained in sync by record_shard_holder() / remove_shard_holder().
    /// Enables O(shards_held) peer departure instead of O(all_shards).
    node_shards: DashMap<NodeId, HashSet<ShardId>>,
    /// Uncapped global holder count from the most recent DHT GetProviders
    /// response, keyed by shard. The bounded `shard_holders` cache above
    /// caps at MAX_HOLDERS_PER_SHARD for routing economy, but the prune
    /// score's redundancy_ratio needs the true swarm-wide count to detect
    /// severely over-replicated shards (gotcha: at 1000-node scale, a 50-cap
    /// cache pegs at 50 regardless of actual replication). Written only by
    /// `record_global_holder_count` from the DHT query result; readers fall
    /// back to the cached count if no DHT data is available.
    global_holder_count: DashMap<ShardId, u32>,
    /// Claims a holder has explicitly RETRACTED, and when.
    ///
    /// A DHT provider record outlives the fact it asserts: a node that deletes
    /// or loses a shard stays a provider until the record expires (hours), and
    /// other peers republish it meanwhile. `merge_dht_providers` is additive —
    /// it can add a holder, never remove one — so without this map a stale
    /// record silently resurrects a claim the holder itself has withdrawn.
    ///
    /// Measured 2026-08-22 on a three-way split: the holder retracted shard 2
    /// correctly and said so every 5 minutes, the coordinator re-merged the
    /// stale DHT record every few seconds, and the scheduler kept planning a
    /// segment onto a node that no longer had the weights — a hard 503 with no
    /// standby rather than a re-route. Retraction alone is futile when
    /// something else re-adds the claim faster than it is withdrawn.
    ///
    /// Cleared by `record_shard_holder`, so a node that genuinely re-acquires
    /// the shard is believed again as soon as it announces it itself. The DHT
    /// path deliberately does NOT clear it — that is the whole point.
    retracted_claims: DashMap<(ShardId, NodeId), Instant>,
    /// Persists a manifest whose shard hashes this registry has just enriched.
    ///
    /// Installed once at startup (see `daemon::state`), in the same shape as
    /// `ModelProcessPool::set_ram_budget_provider` — the registry cannot own a
    /// `Database` handle, and a hook keeps the merge in one place rather than
    /// obliging every caller of `register_manifest` to remember to persist.
    ///
    /// Fires ONLY when a merge actually changed what we hold, which in the
    /// steady state is never: a peer re-gossiping the same placeholder manifest
    /// merges to bytes identical to the stored ones, so this would otherwise
    /// write on every gossip round, for every model, forever.
    /// Shard hashes derived from bytes fetched from the model's ORIGIN, which
    /// therefore outrank anything a peer gossips.
    ///
    /// **Only the origin settles a hash.** Without this, a peer that
    /// self-certified a corrupt shard (hashing its own bad bytes) gossips that
    /// hash, and the last-writer-wins `insert` below adopts it over one we
    /// verified against the origin — after which the re-check quarantines our
    /// GOOD copy and refetches, and every replacement is judged against the
    /// wrong reference, so it can never converge. Observed live 2026-08-25:
    /// `expected ab5bc674… got 597dcfe8…`, where the "got" was the copy checked
    /// byte-for-byte against HuggingFace (gotcha #384).
    ///
    /// Deliberately local and NOT carried in the gossiped manifest: provenance
    /// that travels over the network is just another assertion, and forgeable.
    /// A node trusts only what IT fetched from the origin.
    origin_verified: DashMap<ShardId, crate::types::Blake3Hash>,

    persist_hook: std::sync::OnceLock<ManifestUpdateHook>,
    /// Local node ID — never evicted from holder sets.
    local_node_id: Option<NodeId>,
}

impl ModelRegistry {
    pub fn new() -> Self {
        Self {
            manifests: DashMap::new(),
            shard_holders: DashMap::new(),
            node_shards: DashMap::new(),
            global_holder_count: DashMap::new(),
            retracted_claims: DashMap::new(),
            origin_verified: DashMap::new(),
            persist_hook: std::sync::OnceLock::new(),
            local_node_id: None,
        }
    }

    /// Create a registry with a known local node ID.
    /// The local node is never evicted from bounded holder sets.
    #[cfg(test)]
    pub fn with_local_node(local_node_id: NodeId) -> Self {
        Self {
            manifests: DashMap::new(),
            shard_holders: DashMap::new(),
            node_shards: DashMap::new(),
            global_holder_count: DashMap::new(),
            retracted_claims: DashMap::new(),
            origin_verified: DashMap::new(),
            persist_hook: std::sync::OnceLock::new(),
            local_node_id: Some(local_node_id),
        }
    }

    /// Set the local node ID after construction (e.g., after loading from DB).
    pub fn set_local_node_id(&mut self, node_id: NodeId) {
        self.local_node_id = Some(node_id);
    }

    /// Register a model manifest.
    /// Install the hook that persists a manifest enriched by
    /// `merge_known_shard_hashes`. Idempotent; the first call wins.
    pub fn set_persist_hook(&self, hook: ManifestUpdateHook) {
        let _ = self.persist_hook.set(hook);
    }

    /// Record a shard hash derived from bytes fetched from the model's ORIGIN.
    /// From now on no gossiped manifest can change this shard's hash here.
    pub fn record_origin_verified_hash(&self, shard_id: ShardId, hash: crate::types::Blake3Hash) {
        if hash != [0u8; 32] {
            self.origin_verified.insert(shard_id, hash);
        }
    }

    /// The origin-derived hash for a shard, if this node has ever fetched it
    /// from the origin.
    pub fn origin_verified_hash(&self, shard_id: &ShardId) -> Option<crate::types::Blake3Hash> {
        self.origin_verified.get(shard_id).map(|h| *h)
    }

    pub fn register_manifest(&self, mut manifest: ModelManifest) {
        // Universal net against copied-folder model names (`<model>.FULLBACKUP`,
        // `<model>.old`). This is the single point every adoption path funnels
        // through — gossip ingress, DB reload on startup, local disk scan,
        // acquisition — so rejecting here keeps a backup-copy identity out of
        // the registry regardless of how it arrived, and stops it being
        // persisted and re-gossiped. See `manifest::is_backup_artifact_id`.
        if crate::model::manifest::is_backup_artifact_id(&manifest.id.0) {
            tracing::warn!(
                model = %manifest.id,
                "Refusing manifest with a backup-copy name — a model's identity \
                 must come from the model, not a copied directory. Rename it to \
                 the real model id if this is genuine."
            );
            return;
        }
        // Announce at INFO only when something actually changed.
        //
        // Every peer re-gossips its whole manifest set on a timer, so a settled
        // swarm re-registers the same unchanged manifests forever. Logged
        // unconditionally, that one line was **21% of a real node's log** —
        // 96,850 of 453,591 lines over nine days, for nine models that never
        // changed — and its dispatch-side twin roughly doubled it. A log where
        // getting on for half the volume reports that nothing happened is a log
        // nobody can read when something does.
        //
        // `manifest_hash` is the manifest's own content hash, so this compares
        // what a peer sent against what we hold without a field-by-field walk.
        // A genuine change (new shard, re-publish) still announces itself.
        // Keep any shard hash this manifest leaves blank that we already know.
        //
        // A manifest is generated from what its author happens to hold on disk,
        // so a partial holder publishes real hashes for its own shards and
        // all-zero placeholders for the rest. The `insert` below is blind, so a
        // placeholder used to overwrite a hash we already had — and the P2P
        // accept path verifies a completed transfer ONLY when the manifest
        // carries a non-zero hash. Losing one therefore means the next download
        // of that shard is accepted on trust and announced to the swarm
        // unchecked. See `manifest::merge_known_shard_hashes`.
        let recovered = match self.manifests.get(&manifest.id) {
            Some(prev) => {
                crate::model::manifest::merge_known_shard_hashes(&mut manifest, prev.value())
            }
            None => 0,
        };
        if recovered > 0 {
            // Recompute so the stored manifest stays self-consistent: it is now
            // a local composite rather than verbatim what the publisher sent,
            // and `load_from_dir` re-derives this hash to validate a saved copy.
            //
            // Doing it here also keeps the `changed` check below quiet in the
            // steady state: the merge is deterministic, so an unchanged
            // re-gossip merges to the same bytes and recomputes to the same
            // hash. Without the recompute, every re-gossip of a placeholder
            // manifest would compare unequal and log forever.
            manifest.manifest_hash = manifest.compute_hash();
            tracing::debug!(
                model = %manifest.id,
                recovered,
                "Kept shard hashes this manifest left blank"
            );
        }
        let changed = match self.manifests.get(&manifest.id) {
            Some(prev) => prev.manifest_hash != manifest.manifest_hash,
            None => true,
        };
        // Persist an enrichment so a restart does not throw it away.
        //
        // `load_from_db` is what repopulates this registry at boot, BEFORE the
        // local disk scan, and the disk copy is exactly the thing that carries
        // placeholders. Without this, hashes learned by gossip lasted only as
        // long as the process: on the next boot the node was back to having
        // nothing to check a download against, and (since an uncheckable shard
        // is now fetched from its origin) would re-download it from the origin
        // for no reason. Persisting to the DB alone is enough — the merge above
        // already stops the disk copy's placeholders overwriting anything.
        //
        // Gated on `changed` as well as `recovered` because a peer that keeps
        // gossiping a placeholder manifest merges to the same bytes every time;
        // without the gate this would write on every gossip round forever.
        // A hash we derived from the ORIGIN's own bytes outranks anything a peer
        // says. Applied BEFORE the change-detection below, so adopting a peer's
        // contradicting hash never even registers as a change — otherwise the
        // re-check fires against a reference we have already disproved.
        for shard in manifest.shards.iter_mut() {
            if let Some(known) = self.origin_verified_hash(&ShardId {
                model_id: manifest.id.clone(),
                index: shard.index,
            }) {
                if shard.hash != known {
                    tracing::warn!(
                        model = %manifest.id,
                        shard = shard.index,
                        claimed = %hex::encode(&shard.hash[..8]),
                        origin = %hex::encode(&known[..8]),
                        "Ignoring a shard hash that contradicts the one we took \
                         from the model's origin"
                    );
                    shard.hash = known;
                }
            }
        }

        // Shards WE HOLD whose expected hash just changed. Those bytes were
        // checked against a reference that no longer applies — or, for a hash we
        // never had, were never checked at all — so they must be re-checked.
        //
        // Without this, a node holding a corrupt shard could not discover the
        // fact from gossip: the only re-check of an already-held shard is the
        // startup sweep, which runs seconds after boot against whatever the DB
        // held, i.e. BEFORE the corrected hash arrives. It would take a further
        // restart to notice. Measured on the live swarm — a shard corrupt on at
        // least two peers, whose correct hash those peers could have been told
        // (gotcha #382).
        let recheck: Vec<u32> = match self.manifests.get(&manifest.id) {
            Some(prev) => manifest
                .shards
                .iter()
                .filter(|s| s.hash != [0u8; 32])
                .filter(|s| {
                    prev.shards
                        .iter()
                        .find(|p| p.index == s.index)
                        .is_none_or(|p| p.hash != s.hash)
                })
                .filter(|s| {
                    self.local_node_id.as_ref().is_some_and(|me| {
                        self.shard_holders
                            .get(&ShardId {
                                model_id: manifest.id.clone(),
                                index: s.index,
                            })
                            .is_some_and(|h| h.contains_key(me))
                    })
                })
                .map(|s| s.index)
                .collect(),
            None => Vec::new(),
        };
        let persist = recovered > 0 && changed;
        if persist || !recheck.is_empty() {
            if let Some(hook) = self.persist_hook.get() {
                hook(&manifest, persist, &recheck);
            }
        }
        if changed {
            tracing::info!(
                model = %manifest.id,
                name = %manifest.name,
                shard_count = manifest.shard_count,
                publisher = %manifest.publisher,
                "DIAG: register_manifest"
            );
        } else {
            tracing::debug!(model = %manifest.id, "DIAG: register_manifest (unchanged)");
        }
        self.manifests.insert(manifest.id.clone(), manifest);
    }

    /// Record that a node holds a specific shard.
    ///
    /// Bounded: if the holder set is at capacity, the oldest non-local holder
    /// is evicted to make room. Maintains reverse index.
    pub fn record_shard_holder(&self, shard_id: ShardId, node_id: NodeId) {
        // A first-hand claim supersedes any earlier retraction: the node is
        // telling us it has the shard now. Only this path clears it — the DHT
        // merge must not, or a stale provider record would undo a retraction.
        self.retracted_claims
            .remove(&(shard_id.clone(), node_id.clone()));
        let mut entry = self.shard_holders.entry(shard_id.clone()).or_default();
        let holders = entry.value_mut();

        // Update timestamp if already present (always succeeds, no eviction needed)
        if holders.contains_key(&node_id) {
            holders.insert(node_id.clone(), Instant::now());
            // Reverse index already has this entry
            return;
        }

        // At capacity — evict oldest non-local holder
        if holders.len() >= MAX_HOLDERS_PER_SHARD {
            let evict_id = holders
                .iter()
                .filter(|(nid, _)| Some(*nid) != self.local_node_id.as_ref())
                .min_by_key(|(_, ts)| *ts)
                .map(|(nid, _)| nid.clone());

            if let Some(evict) = evict_id {
                holders.remove(&evict);
                // Mirror in reverse index
                if let Some(mut shards) = self.node_shards.get_mut(&evict) {
                    shards.remove(&shard_id);
                }
            } else {
                // All holders are local (shouldn't happen) — skip insert
                return;
            }
        }

        holders.insert(node_id.clone(), Instant::now());
        // Update reverse index BEFORE dropping shard_holders guard so concurrent
        // remove_peer_from_all_shards sees a consistent state (no half-update window).
        self.node_shards
            .entry(node_id)
            .or_default()
            .insert(shard_id);
        drop(entry);
    }

    /// Remove a node from shard holders (e.g., node went offline).
    /// Maintains reverse index.
    pub fn remove_shard_holder(&self, shard_id: &ShardId, node_id: &NodeId) {
        if let Some(mut holders) = self.shard_holders.get_mut(shard_id) {
            holders.remove(node_id);
        }
        // Remove empty tombstones atomically — `remove_if` holds the
        // shard lock for both the empty check and the removal, so a
        // concurrent `record_shard_holder` inserting a fresh holder
        // can't be silently dropped between a separate check + remove
        // (the prior two-step pattern lost holders under peer churn).
        self.shard_holders
            .remove_if(shard_id, |_, holders| holders.is_empty());
        if let Some(mut shards) = self.node_shards.get_mut(node_id) {
            shards.remove(shard_id);
        }
    }

    /// Make `node_id`'s holder records for `model_id` exactly `keep`, dropping
    /// any shard of that model it is no longer listed as holding.
    ///
    /// This is the only way a node's claim is ever retracted by someone other
    /// than the node itself. `remove_shard_holder` is always called with the
    /// LOCAL node id — a peer dropping a shard had no way to tell us, so its
    /// stale claim survived for as long as it stayed connected and the
    /// scheduler kept routing layers to a node that could not serve them.
    ///
    /// Scoped to one model on purpose: a `ShardAnnounce` is only ever complete
    /// for the models it declares, so retracting beyond them would delete
    /// records the announcement says nothing about.
    ///
    /// Returns the number of records removed.
    pub fn retain_node_shards_for_model(
        &self,
        model_id: &ModelId,
        node_id: &NodeId,
        keep: &std::collections::HashSet<u32>,
    ) -> usize {
        // Collect first: mutating shard_holders while iterating it can deadlock
        // on the same DashMap shard.
        let stale: Vec<ShardId> = self
            .node_shards
            .get(node_id)
            .map(|shards| {
                shards
                    .iter()
                    .filter(|sid| sid.model_id == *model_id && !keep.contains(&sid.index))
                    .cloned()
                    .collect()
            })
            .unwrap_or_default();

        for shard_id in &stale {
            self.remove_shard_holder(shard_id, node_id);
            self.note_retracted_claim(shard_id.clone(), node_id.clone());
        }
        stale.len()
    }

    /// Drop a holder's claim AND remember that it was withdrawn, so a stale DHT
    /// provider record cannot reinstate it (see `retracted_claims`).
    ///
    /// This is the one to call from a retraction path. A bare
    /// `remove_shard_holder` is for eviction and bookkeeping, and on its own it
    /// is undone by the next `GetProviders` response.
    pub fn retract_shard_holder(&self, shard_id: &ShardId, node_id: &NodeId) {
        self.remove_shard_holder(shard_id, node_id);
        self.note_retracted_claim(shard_id.clone(), node_id.clone());
    }

    /// Remember that `node_id` has withdrawn its claim on `shard_id`, so a
    /// stale DHT provider record cannot put it back (see `retracted_claims`).
    fn note_retracted_claim(&self, shard_id: ShardId, node_id: NodeId) {
        if self.retracted_claims.len() >= MAX_RETRACTED_CLAIMS {
            let now = Instant::now();
            self.retracted_claims
                .retain(|_, at| now.duration_since(*at).as_secs() < RETRACTION_HONOURED_SECS);
        }
        self.retracted_claims
            .insert((shard_id, node_id), Instant::now());
    }

    /// Has this node withdrawn its claim on this shard recently enough that a
    /// DHT provider record should not be believed over it?
    fn claim_was_retracted(&self, shard_id: &ShardId, node_id: &NodeId) -> bool {
        self.retracted_claims
            .get(&(shard_id.clone(), node_id.clone()))
            .is_some_and(|at| at.elapsed().as_secs() < RETRACTION_HONOURED_SECS)
    }

    /// Get nodes that hold the mmproj (vision encoder) for a model.
    pub fn mmproj_holders(&self, model_id: &ModelId) -> Vec<NodeId> {
        self.shard_holders(&ShardId {
            model_id: model_id.clone(),
            index: MMPROJ_SHARD_INDEX,
        })
    }

    /// Get all nodes that hold a specific shard (from the bounded cache).
    pub fn shard_holders(&self, shard_id: &ShardId) -> Vec<NodeId> {
        self.shard_holders
            .get(shard_id)
            .map(|v| v.keys().cloned().collect())
            .unwrap_or_default()
    }

    /// Get the cached holder count for a shard without allocating.
    /// This may undercount at scale — use DHT queries for accurate counts.
    #[cfg(test)]
    pub fn shard_holder_count(&self, shard_id: &ShardId) -> usize {
        self.shard_holders
            .get(shard_id)
            .map(|v| v.len())
            .unwrap_or(0)
    }

    /// Get the shard indices held locally by a node for a given model.
    /// Convenience wrapper around `local_shard_indices_in` for call
    /// sites that don't already have the manifest in scope.
    pub fn local_shard_indices(
        &self,
        model_id: &crate::types::ModelId,
        node_id: &NodeId,
    ) -> Vec<u32> {
        self.get_manifest(model_id)
            .map(|m| self.local_shard_indices_in(&m, node_id))
            .unwrap_or_default()
    }

    /// Filter the shards of an already-resolved manifest down to indices
    /// held by `node_id`. Lets hot-path callers (per-token LayerForward)
    /// reuse a manifest they already fetched rather than paying for a
    /// second `get_manifest` clone.
    pub fn local_shard_indices_in(
        &self,
        manifest: &crate::types::ModelManifest,
        node_id: &NodeId,
    ) -> Vec<u32> {
        manifest
            .shards
            .iter()
            .filter(|s| {
                let sid = ShardId {
                    model_id: manifest.id.clone(),
                    index: s.index,
                };
                self.shard_holders(&sid).contains(node_id)
            })
            .map(|s| s.index)
            .collect()
    }

    /// Merge holders discovered via DHT provider queries into the cache.
    /// Updates timestamps for existing entries and adds new ones (with eviction).
    pub fn merge_dht_providers(&self, shard_id: &ShardId, providers: &[NodeId]) {
        for node_id in providers {
            // The holder's own word beats the DHT's memory of it. A provider
            // record is republished for hours after the shard is gone, and this
            // merge is the only writer that cannot remove a holder, so without
            // the check a withdrawn claim comes straight back.
            if self.claim_was_retracted(shard_id, node_id) {
                continue;
            }
            self.record_shard_holder(shard_id.clone(), node_id.clone());
        }
    }

    /// Record the swarm-wide provider count from a DHT GetProviders response.
    /// The count comes from the raw PeerId set (before NodeId resolution drops
    /// any) so it reflects what the DHT reports, independent of our local cache
    /// cap. Overwrites any previous reading.
    pub fn record_global_holder_count(&self, shard_id: ShardId, count: u32) {
        self.global_holder_count.insert(shard_id, count);
    }

    /// Best-effort uncapped holder count for a shard. Returns the most recent
    /// DHT-reported count if any query has resolved, else `None` — caller
    /// should fall back to the cached `shard_holders().len()` for the local
    /// view.
    pub fn global_holder_count(&self, shard_id: &ShardId) -> Option<u32> {
        self.global_holder_count.get(shard_id).map(|v| *v)
    }

    /// Get a model manifest by ID.
    pub fn get_manifest(&self, model_id: &ModelId) -> Option<ModelManifest> {
        self.manifests.get(model_id).map(|v| v.clone())
    }

    /// Shard ids whose layer ranges overlap `layer_range`.
    ///
    /// **This is the answer to "which shards does a pipeline segment actually
    /// touch?", and it is not `segment.shard_id`.** That field holds a single
    /// shard — the first one of the candidate that served the segment — while a
    /// segment routinely spans several. Reading it as though it named the whole
    /// span has already caused one bug: a segment failing on `blk.10` (shard 2)
    /// retracted shard 0, which the holder genuinely had. Sub-range routing
    /// makes multi-shard segments the normal case rather than an edge case, so
    /// every consumer that needs the span must come through here.
    ///
    /// Returns an empty Vec when the manifest is unknown; callers guarding a
    /// destructive operation must treat that as "cannot prove it is unused"
    /// rather than as "unused".
    pub fn shards_overlapping_layers(
        &self,
        model_id: &ModelId,
        layer_range: (u32, u32),
    ) -> Vec<crate::types::ShardId> {
        let Some(manifest) = self.get_manifest(model_id) else {
            return Vec::new();
        };
        manifest
            .shards
            .iter()
            // Half-open ranges: [a,b) overlaps [c,d) iff a < d && c < b.
            .filter(|s| s.layer_range.0 < layer_range.1 && layer_range.0 < s.layer_range.1)
            .map(|s| crate::types::ShardId {
                model_id: model_id.clone(),
                index: s.index,
            })
            .collect()
    }

    /// Shards a pipeline segment actually reads. Falls back to the segment's own
    /// `shard_id` when the manifest is unknown, so a caller never sees an empty
    /// span for a segment that is demonstrably executing.
    pub fn shards_spanned_by_segment(
        &self,
        segment: &swarmllm_types::PipelineSegment,
    ) -> Vec<crate::types::ShardId> {
        let spanned =
            self.shards_overlapping_layers(&segment.shard_id.model_id, segment.layer_range);
        if spanned.is_empty() {
            vec![segment.shard_id.clone()]
        } else {
            spanned
        }
    }

    /// Resolve a manifest for a loaded model by trying slug, display name, and manifest name field.
    pub fn resolve_manifest_by_name(&self, display_name: &str) -> Option<ModelManifest> {
        let slug = crate::types::slugify_model_name(display_name);
        self.get_manifest(&ModelId(slug))
            .or_else(|| self.get_manifest(&ModelId(display_name.to_string())))
            .or_else(|| self.models().into_iter().find(|m| m.name == display_name))
    }

    /// Human-friendly display name for a model (falls back to raw model ID).
    pub fn display_name(&self, model_id: &ModelId) -> String {
        self.manifests
            .get(model_id)
            .map(|m| m.name.clone())
            .unwrap_or_else(|| model_id.0.clone())
    }

    /// Get all known model manifests.
    pub fn models(&self) -> Vec<ModelManifest> {
        self.manifests.iter().map(|v| v.value().clone()).collect()
    }

    /// The manifests this node should re-gossip: ones it published, **and ones
    /// it holds at least one shard of**.
    ///
    /// The second half is the load-bearing one, and leaving it out of the
    /// periodic broadcast broke model discovery swarm-wide. `publisher` is not a
    /// stable "who owns this" — every holder used to rewrite it to itself at
    /// startup purely to earn broadcast rights, and `register_manifest`
    /// overwrites unconditionally, so each holder's claim erased the previous
    /// one. Measured on a 5-node swarm: `phi-3.5-mini` had been registered 81
    /// times under **50 distinct publishers**. With a publisher-only filter that
    /// race converges on *nobody* broadcasting, and since there is no on-demand
    /// manifest fetch, a node that joins later can never learn the model exists
    /// in full — `all_shards_available` stays false and every request for it
    /// answers "No model loaded" while the dashboard lists it as available.
    ///
    /// Holding a shard is the honest signal: a node serving part of a model can
    /// vouch for its manifest, which is why the gossip handler deliberately does
    /// NOT require `sender == publisher`. Any new manifest-broadcast path must
    /// come through here rather than re-deriving the predicate — the one-shot
    /// startup announcement had it right and the 30s timer did not, so discovery
    /// worked for whoever was already connected at boot and for nobody after.
    pub fn manifests_to_gossip(&self, node_id: &NodeId) -> Vec<ModelManifest> {
        let hosted: std::collections::HashSet<String> = self
            .all_shard_entries()
            .into_iter()
            .filter_map(|(shard_id, holders)| {
                holders
                    .contains(node_id)
                    .then(|| shard_id.model_id.0.clone())
            })
            .collect();
        self.models()
            .into_iter()
            .filter(|m| m.publisher == *node_id || hosted.contains(&m.id.0))
            .collect()
    }

    /// Build a "model not found" error carrying the list of known models.
    ///
    /// Callers get `SwarmError::ModelNotAvailable` (mapped to HTTP 404 by the
    /// API layer). The message varies based on whether the registry is empty
    /// to give users a more actionable hint.
    pub fn model_not_found_error(&self, model_id: &ModelId) -> SwarmError {
        let available: Vec<String> = self.models().iter().map(|m| m.id.0.clone()).collect();
        let msg = if available.is_empty() {
            format!(
                "Model '{}' not found. No models are available — download shards first.",
                model_id.0
            )
        } else {
            format!(
                "Model '{}' not found. Available models: {}",
                model_id.0,
                available.join(", ")
            )
        };
        SwarmError::ModelNotAvailable(ModelId(msg))
    }

    /// `Some(error)` when the caller named a model this node does not have
    /// while it DOES have others — i.e. the name is wrong, not merely early.
    ///
    /// **Every API surface that resolves a user-supplied model id should reject
    /// through this**, so the answer does not depend on which endpoint was
    /// used. The distinction it encodes:
    ///
    /// - Some models registered, this one absent → the user mistyped or asked
    ///   for something unavailable. `ModelNotAvailable` → **404**, listing what
    ///   IS available. Returning it immediately also avoids burning the
    ///   cold-start wait on a name that will never appear.
    /// - No models registered at all → the node may still be starting up or
    ///   have nothing downloaded. `None` here, so the caller falls through to
    ///   its cold-start wait or `NoModelLoaded` (503), whose hint tells the
    ///   user to go and get a model.
    ///
    /// The OpenAI handler had this rule inline and `/v1/messages` did not, so
    /// the same typo answered 404 with the model list on one endpoint and 503
    /// "No model is loaded yet. Go to the dashboard and select a model" on the
    /// other — advice that is simply wrong when eight models are loaded.
    pub fn reject_if_unknown_model(&self, model_id: &ModelId) -> Option<SwarmError> {
        if self.get_manifest(model_id).is_none() && !self.manifests.is_empty() {
            Some(self.model_not_found_error(model_id))
        } else {
            None
        }
    }

    /// Get the number of registered models.
    #[cfg(test)]
    pub fn model_count(&self) -> usize {
        self.manifests.len()
    }

    /// Remove a model manifest from the registry.
    pub fn remove_manifest(&self, model_id: &ModelId) {
        self.manifests.remove(model_id);
    }

    /// Remove all shard holder entries for a given model.
    /// Maintains reverse index: removes affected ShardIds from each holder's node_shards.
    pub fn remove_all_model_shards(&self, model_id: &ModelId) {
        // Collect affected (shard, holders) before removing from shard_holders
        let affected: Vec<(ShardId, Vec<NodeId>)> = self
            .shard_holders
            .iter()
            .filter(|e| &e.key().model_id == model_id)
            .map(|e| (e.key().clone(), e.value().keys().cloned().collect()))
            .collect();

        self.shard_holders
            .retain(|shard_id, _| &shard_id.model_id != model_id);
        // Drop the uncapped DHT-derived global count too — stale entries here
        // would falsely inflate a future shard's redundancy_ratio if the same
        // ShardId got reused.
        self.global_holder_count
            .retain(|shard_id, _| &shard_id.model_id != model_id);

        // Mirror removal in reverse index
        for (shard_id, holders) in affected {
            for node_id in holders {
                if let Some(mut shards) = self.node_shards.get_mut(&node_id) {
                    shards.remove(&shard_id);
                }
            }
        }
    }

    /// Remove a peer from all shard holder entries (e.g., peer went offline).
    /// Uses reverse index for O(shards_held) instead of O(all_shards).
    pub fn remove_peer_from_all_shards(&self, node_id: &NodeId) {
        if let Some((_, shards)) = self.node_shards.remove(node_id) {
            for shard_id in shards {
                if let Some(mut holders) = self.shard_holders.get_mut(&shard_id) {
                    holders.remove(node_id);
                }
            }
        }
    }

    /// Get all shards held by a specific node (via reverse index).
    pub fn shards_for_node(&self, node_id: &NodeId) -> Vec<ShardId> {
        self.node_shards
            .get(node_id)
            .map(|s| s.iter().cloned().collect())
            .unwrap_or_default()
    }

    /// Iterate over all tracked shard entries (shard_id, holders).
    pub fn all_shard_entries(&self) -> Vec<(ShardId, Vec<NodeId>)> {
        self.shard_holders
            .iter()
            .map(|entry| (entry.key().clone(), entry.value().keys().cloned().collect()))
            .collect()
    }

    /// Load model and shard metadata from the database.
    pub fn load_from_db(db: &Database) -> Result<Self, SwarmError> {
        let registry = Self::new();

        // Hashes we took from a model's origin, which outrank gossip. Loaded
        // FIRST so the manifests below cannot be registered against a peer's
        // contradicting claim during startup (#384).
        if let Ok(entries) = db.iter_raw(ORIGIN_VERIFIED_TREE) {
            for (key, value) in entries {
                let Ok(key_str) = std::str::from_utf8(&key) else {
                    continue;
                };
                let Ok(shard_id) = serde_json::from_str::<ShardId>(key_str) else {
                    continue;
                };
                if value.len() == 32 {
                    let mut h = [0u8; 32];
                    h.copy_from_slice(&value);
                    registry.record_origin_verified_hash(shard_id, h);
                }
            }
        }

        // Load model manifests from the "model_meta" tree
        let entries = db.iter_raw("model_meta")?;
        let mut purge_backup_ids: Vec<String> = Vec::new();
        for (_key, value) in entries {
            match serde_json::from_slice::<ModelManifest>(&value) {
                Ok(manifest) => {
                    // A backup-copy id (`<model>.FULLBACKUP`, `<model>.old`)
                    // persisted before the register_manifest guard existed must
                    // NOT be re-adopted on reload — otherwise the phantom model
                    // resurrects on every restart. This is the one manifest-entry
                    // path that bypasses register_manifest, so it needs its own
                    // guard. Skip it and purge the stale row so it stops coming
                    // back. See `manifest::is_backup_artifact_id`.
                    if crate::model::manifest::is_backup_artifact_id(&manifest.id.0) {
                        tracing::warn!(
                            model = %manifest.id,
                            "Dropping backup-copy manifest from DB on load (and purging the row)"
                        );
                        purge_backup_ids.push(manifest.id.0.clone());
                        continue;
                    }
                    registry.manifests.insert(manifest.id.clone(), manifest);
                }
                Err(e) => {
                    tracing::warn!(error = %e, "Failed to deserialize model manifest from DB");
                }
            }
        }
        for id in purge_backup_ids {
            if let Err(e) = db.remove("model_meta", &id) {
                tracing::debug!(model = %id, error = %e, "Failed to purge backup-copy manifest row");
            }
        }

        tracing::info!(
            manifests_loaded_count = registry.manifests.len(),
            "DIAG: load_from_db complete"
        );

        Ok(registry)
    }

    /// Persist a model manifest to the database.
    pub fn persist_manifest(
        &self,
        db: &Database,
        manifest: &ModelManifest,
    ) -> Result<(), SwarmError> {
        db.put_json("model_meta", &manifest.id.0, manifest)?;
        Ok(())
    }
}

impl Default for ModelRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::*;

    /// The two branches an API surface must keep apart, because they need
    /// opposite things from the user.
    ///
    /// `/v1/messages` used to answer a misspelled model with 503 "No model is
    /// loaded yet. Go to the dashboard and select a model" while
    /// `/v1/chat/completions` answered 404 with the list of models that were in
    /// fact loaded. Same node, same request, different advice — and the 503
    /// advice was wrong.
    #[test]
    fn an_unknown_model_is_rejected_only_when_others_exist() {
        let registry = ModelRegistry::new();
        let asked = ModelId("typo-model-name".into());

        // Nothing registered: the node may still be starting or have nothing
        // downloaded, so the caller must fall through to its cold-start wait /
        // NoModelLoaded rather than claim the name is wrong.
        assert!(
            registry.reject_if_unknown_model(&asked).is_none(),
            "an empty registry must not accuse the user of a bad model name"
        );

        registry.register_manifest(test_manifest("real-model", "Real"));

        // Now the name really is wrong, and the answer must name what IS here.
        let err = registry
            .reject_if_unknown_model(&asked)
            .expect("a wrong name must be rejected once models are known");
        match err {
            SwarmError::ModelNotAvailable(msg) => {
                assert!(
                    msg.0.contains("typo-model-name"),
                    "names the request: {msg:?}"
                );
                assert!(
                    msg.0.contains("real-model"),
                    "lists what is available: {msg:?}"
                );
            }
            other => panic!("must be ModelNotAvailable (404), got {other:?}"),
        }

        // A model that IS registered is never rejected here.
        assert!(registry
            .reject_if_unknown_model(&ModelId("real-model".into()))
            .is_none());
    }

    /// Holding a shard of a model MUST earn the right to re-gossip its
    /// manifest. Filtering on `publisher` alone stopped model discovery for a
    /// whole swarm: every holder rewrote `publisher` to itself at startup to
    /// earn broadcast rights, `register_manifest` overwrites unconditionally,
    /// so holders erased each other's claim until none of them broadcast. With
    /// no on-demand manifest fetch, a node that joined later could never learn
    /// the model and answered "No model loaded" for a model the dashboard
    /// listed as available.
    #[test]
    fn a_shard_holder_gossips_the_manifest_even_when_someone_else_published_it() {
        let registry = ModelRegistry::new();
        let us = NodeId([7u8; 32]);
        let someone_else = NodeId([9u8; 32]);

        // Published by a peer — exactly what every acquired model looks like,
        // and what our own copy degrades to the moment that peer re-announces.
        let mut foreign = test_manifest("held-model", "Held");
        foreign.publisher = someone_else.clone();
        registry.register_manifest(foreign);

        // A model we neither published nor hold: not ours to vouch for.
        let mut untouched = test_manifest("other-model", "Other");
        untouched.publisher = someone_else.clone();
        registry.register_manifest(untouched);

        assert!(
            registry.manifests_to_gossip(&us).is_empty(),
            "holding nothing must gossip nothing"
        );

        registry.record_shard_holder(
            ShardId {
                model_id: ModelId("held-model".into()),
                index: 0,
            },
            us.clone(),
        );

        let ids: Vec<String> = registry
            .manifests_to_gossip(&us)
            .into_iter()
            .map(|m| m.id.0)
            .collect();
        assert_eq!(
            ids,
            vec!["held-model".to_string()],
            "a shard we hold must earn the manifest a broadcast, and nothing else should"
        );
    }

    fn test_manifest(id: &str, name: &str) -> ModelManifest {
        ModelManifest {
            id: ModelId(id.into()),
            name: name.into(),
            architecture: ModelArchitecture::Llama,
            num_layers: 2,
            num_params_billions: 0.001,
            quantization: Quantization::Q4KM,
            total_size_bytes: 1024,
            shard_count: 1,
            shards: vec![],
            tokenizer_hash: [0u8; 32],
            manifest_hash: [0u8; 32],
            publisher: NodeId([0u8; 32]),
            publish_date: chrono::Utc::now(),
            license: "MIT".into(),
            mmproj: None,
        }
    }

    /// A shard hash may go from unknown to known, never back to unknown.
    ///
    /// A manifest is generated from what its author holds on disk, so a partial
    /// holder publishes real hashes for its own shards and all-zero
    /// placeholders for the rest. Registration used to `insert` blindly, so the
    /// placeholder won and the hash we already had was destroyed — after which
    /// the P2P accept path, which verifies a completed transfer only when a
    /// non-zero hash exists, took that shard's bytes on trust and announced us
    /// as a holder of them.
    ///
    /// Observed on the live node 2026-08-24: five shards fetched from peers
    /// against a manifest carrying placeholders for exactly those five, one of
    /// them corrupt, surfacing only after a restart.
    #[test]
    fn a_placeholder_hash_never_overwrites_one_we_already_know() {
        let registry = ModelRegistry::new();
        let real = [7u8; 32];

        let mut complete = test_manifest("m", "M");
        complete.shard_count = 2;
        complete.shards = vec![test_shard(0, real), test_shard(1, [9u8; 32])];
        registry.register_manifest(complete);

        // The same model as seen by a node that holds neither shard.
        let mut blank = test_manifest("m", "M");
        blank.shard_count = 2;
        blank.shards = vec![test_shard(0, [0u8; 32]), test_shard(1, [0u8; 32])];
        registry.register_manifest(blank.clone());

        let stored = registry.get_manifest(&ModelId("m".into())).unwrap();
        assert_eq!(
            stored.shards[0].hash, real,
            "a blank manifest must not erase a hash we already had — without it \
             the next download of this shard is accepted unverified"
        );
        assert_eq!(stored.shards[1].hash, [9u8; 32]);

        // The stored manifest must stay self-consistent, since `load_from_dir`
        // re-derives this hash to validate a saved copy.
        let stored_hash = stored.manifest_hash;
        assert_eq!(
            stored_hash,
            stored.compute_hash(),
            "a merged manifest must carry a hash matching its own content"
        );

        // Re-gossiping the SAME blank manifest must settle, not oscillate: the
        // peer sends identical bytes each round, so the merge must recompute to
        // an identical hash or the changed-detection logs on every repeat.
        registry.register_manifest(blank);
        let settled = registry.get_manifest(&ModelId("m".into())).unwrap();
        assert_eq!(settled.shards[0].hash, real);
        assert_eq!(
            settled.manifest_hash, stored_hash,
            "an unchanged re-gossip must merge to the same bytes, or the \
             changed-detection logs on every repeat"
        );
    }

    /// A recovered hash is persisted, but only when it actually changed what we
    /// hold — otherwise a peer that keeps gossiping a placeholder manifest would
    /// make this write on every gossip round, for every model, forever.
    #[test]
    fn a_recovered_hash_is_persisted_once_not_on_every_gossip_round() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;

        let registry = ModelRegistry::new();
        let writes = Arc::new(AtomicUsize::new(0));
        let w = writes.clone();
        registry.set_persist_hook(Box::new(move |_m, persist, _recheck| {
            if persist {
                w.fetch_add(1, Ordering::SeqCst);
            }
        }));

        let mut complete = test_manifest("m", "M");
        complete.shards = vec![test_shard(0, [7u8; 32])];
        registry.register_manifest(complete);
        assert_eq!(
            writes.load(Ordering::SeqCst),
            0,
            "nothing was recovered, so nothing needs persisting"
        );

        let mut blank = test_manifest("m", "M");
        blank.shards = vec![test_shard(0, [0u8; 32])];
        registry.register_manifest(blank.clone());
        assert_eq!(
            writes.load(Ordering::SeqCst),
            1,
            "the recovered hash must survive a restart, so it is written once"
        );

        // The same peer gossiping the same placeholder manifest again merges to
        // identical bytes — no new knowledge, so no write.
        registry.register_manifest(blank.clone());
        registry.register_manifest(blank);
        assert_eq!(
            writes.load(Ordering::SeqCst),
            1,
            "an unchanged re-gossip must not write; this fires every 30s per peer"
        );
    }

    /// A hash we took from the model's ORIGIN outranks anything a peer gossips.
    ///
    /// Without this, a peer that self-certified a corrupt shard displaces the
    /// right hash with its wrong one, the re-check then quarantines our GOOD
    /// copy, and every replacement is judged against the same wrong reference,
    /// so it can never converge. Observed live: `expected ab5bc674… got
    /// 597dcfe8…`, where the "got" was verified byte-for-byte against the origin.
    #[test]
    fn an_origin_hash_outranks_a_peers_contradicting_claim() {
        let me = NodeId([1u8; 32]);
        let registry = ModelRegistry::with_local_node(me.clone());
        let sid = ShardId {
            model_id: ModelId("m".into()),
            index: 7,
        };
        let good = [0x59u8; 32];
        let bad = [0xabu8; 32];

        registry.record_origin_verified_hash(sid.clone(), good);

        let mut from_peer = test_manifest("m", "M");
        from_peer.shards = vec![test_shard(7, bad)];
        registry.register_manifest(from_peer);

        let stored = registry.get_manifest(&ModelId("m".into())).unwrap();
        assert_eq!(
            stored.shards[0].hash, good,
            "a peer's self-certified hash must not displace one taken from the \
             origin — that is how a good shard gets quarantined and replaced \
             with a bad one"
        );
    }

    /// The re-check must not fire for a claim the origin has already disproved:
    /// otherwise the node re-hashes a shard it has every reason to trust,
    /// against a reference it has already rejected.
    #[test]
    fn a_disproved_claim_does_not_trigger_a_recheck() {
        use std::sync::{Arc, Mutex};

        let me = NodeId([1u8; 32]);
        let registry = ModelRegistry::with_local_node(me.clone());
        let sid = ShardId {
            model_id: ModelId("m".into()),
            index: 7,
        };
        let good = [0x59u8; 32];

        let mut first = test_manifest("m", "M");
        first.shards = vec![test_shard(7, good)];
        registry.register_manifest(first);
        registry.record_shard_holder(sid.clone(), me.clone());
        registry.record_origin_verified_hash(sid.clone(), good);

        let seen: Arc<Mutex<Vec<u32>>> = Arc::new(Mutex::new(Vec::new()));
        let s2 = seen.clone();
        registry.set_persist_hook(Box::new(move |_m, _p, recheck| {
            s2.lock().unwrap().extend_from_slice(recheck);
        }));

        let mut from_peer = test_manifest("m", "M");
        from_peer.shards = vec![test_shard(7, [0xabu8; 32])];
        registry.register_manifest(from_peer);

        assert!(
            seen.lock().unwrap().is_empty(),
            "a hash the origin has already disproved must not provoke a re-check"
        );
    }

    /// A shard WE HOLD whose expected hash changes must be re-checked against
    /// the new reference — this is how a node learns from the swarm that what it
    /// is serving is wrong. The startup sweep cannot do it: it runs seconds
    /// after boot, before any corrected hash arrives by gossip.
    #[test]
    fn a_held_shard_is_rechecked_when_its_expected_hash_changes() {
        use std::sync::{Arc, Mutex};

        let me = NodeId([1u8; 32]);
        let registry = ModelRegistry::with_local_node(me.clone());
        let seen: Arc<Mutex<Vec<u32>>> = Arc::new(Mutex::new(Vec::new()));
        let s2 = seen.clone();
        registry.set_persist_hook(Box::new(move |_m, _persist, recheck| {
            s2.lock().unwrap().extend_from_slice(recheck);
        }));

        let mut first = test_manifest("m", "M");
        first.shard_count = 2;
        first.shards = vec![test_shard(0, [1u8; 32]), test_shard(1, [1u8; 32])];
        registry.register_manifest(first);

        // We hold shard 0 only.
        registry.record_shard_holder(
            ShardId {
                model_id: ModelId("m".into()),
                index: 0,
            },
            me.clone(),
        );

        // The swarm now reports a different hash for BOTH shards.
        let mut corrected = test_manifest("m", "M");
        corrected.shard_count = 2;
        corrected.shards = vec![test_shard(0, [2u8; 32]), test_shard(1, [2u8; 32])];
        registry.register_manifest(corrected);

        let got = seen.lock().unwrap().clone();
        assert_eq!(
            got,
            vec![0],
            "only the shard we actually hold needs re-checking — re-hashing one \
             we do not have costs hundreds of MB of I/O for no answer"
        );
    }

    /// The converse, which is deliberately NOT protected: a genuine re-publish
    /// changes the bytes, and the newer real hash must win. Only unknown is
    /// treated as "no information".
    #[test]
    fn a_real_hash_still_replaces_an_earlier_real_hash() {
        let registry = ModelRegistry::new();

        let mut first = test_manifest("m", "M");
        first.shards = vec![test_shard(0, [1u8; 32])];
        registry.register_manifest(first);

        let mut republished = test_manifest("m", "M");
        republished.shards = vec![test_shard(0, [2u8; 32])];
        registry.register_manifest(republished);

        let stored = registry.get_manifest(&ModelId("m".into())).unwrap();
        assert_eq!(stored.shards[0].hash, [2u8; 32]);
    }

    fn test_shard(index: u32, hash: Blake3Hash) -> ShardInfo {
        ShardInfo {
            index,
            layer_range: (index, index + 1),
            size_bytes: 512,
            hash,
            tensors: vec![],
        }
    }

    #[test]
    fn register_and_retrieve_manifest() {
        let registry = ModelRegistry::new();
        let manifest = test_manifest("test", "Test");

        registry.register_manifest(manifest.clone());

        let retrieved = registry.get_manifest(&ModelId("test".into()));
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().name, "Test");
    }

    #[test]
    fn shard_holder_tracking() {
        let registry = ModelRegistry::new();
        let shard_id = ShardId {
            model_id: ModelId("test".into()),
            index: 0,
        };
        let node_a = NodeId([1u8; 32]);
        let node_b = NodeId([2u8; 32]);

        registry.record_shard_holder(shard_id.clone(), node_a.clone());
        registry.record_shard_holder(shard_id.clone(), node_b.clone());

        let holders = registry.shard_holders(&shard_id);
        assert_eq!(holders.len(), 2);
    }

    /// The reported failure: a peer deleted shards 0-4 and restarted, but the
    /// scheduler kept assigning it layers 0..32 because nothing could retract
    /// a still-connected peer's claim.
    #[test]
    fn retain_node_shards_drops_what_a_peer_no_longer_holds() {
        let registry = ModelRegistry::new();
        let model = ModelId("llama-3.1-8b".into());
        let peer = NodeId([7u8; 32]);

        for idx in 0..9u32 {
            registry.record_shard_holder(
                ShardId {
                    model_id: model.clone(),
                    index: idx,
                },
                peer.clone(),
            );
        }

        // Peer re-announces holding only 5-8.
        let keep: std::collections::HashSet<u32> = (5..9).collect();
        let dropped = registry.retain_node_shards_for_model(&model, &peer, &keep);
        assert_eq!(dropped, 5, "shards 0-4 should be retracted");

        for idx in 0..5u32 {
            assert!(
                registry
                    .shard_holders(&ShardId {
                        model_id: model.clone(),
                        index: idx
                    })
                    .is_empty(),
                "shard {idx} should have no holders left"
            );
        }
        for idx in 5..9u32 {
            assert!(
                registry
                    .shard_holders(&ShardId {
                        model_id: model.clone(),
                        index: idx
                    })
                    .contains(&peer),
                "shard {idx} should still be held"
            );
        }
    }

    /// Retraction is scoped to one model — an announce complete for model A
    /// says nothing about model B, and must not delete it.
    #[test]
    fn retain_node_shards_leaves_other_models_alone() {
        let registry = ModelRegistry::new();
        let (a, b) = (ModelId("model-a".into()), ModelId("model-b".into()));
        let peer = NodeId([7u8; 32]);

        registry.record_shard_holder(
            ShardId {
                model_id: a.clone(),
                index: 0,
            },
            peer.clone(),
        );
        registry.record_shard_holder(
            ShardId {
                model_id: b.clone(),
                index: 0,
            },
            peer.clone(),
        );

        // Complete for A with nothing held → drops A only.
        let dropped = registry.retain_node_shards_for_model(&a, &peer, &Default::default());
        assert_eq!(dropped, 1);
        assert!(registry
            .shard_holders(&ShardId {
                model_id: a,
                index: 0
            })
            .is_empty());
        assert!(registry
            .shard_holders(&ShardId {
                model_id: b,
                index: 0
            })
            .contains(&peer));
    }

    /// One peer retracting must not evict another peer's holding.
    #[test]
    fn retain_node_shards_only_touches_the_announcing_node() {
        let registry = ModelRegistry::new();
        let model = ModelId("m".into());
        let (peer_a, peer_b) = (NodeId([1u8; 32]), NodeId([2u8; 32]));
        let sid = ShardId {
            model_id: model.clone(),
            index: 0,
        };

        registry.record_shard_holder(sid.clone(), peer_a.clone());
        registry.record_shard_holder(sid.clone(), peer_b.clone());

        registry.retain_node_shards_for_model(&model, &peer_a, &Default::default());
        let holders = registry.shard_holders(&sid);
        assert!(!holders.contains(&peer_a));
        assert!(holders.contains(&peer_b), "peer B must be untouched");
    }

    #[test]
    fn remove_shard_holder() {
        let registry = ModelRegistry::new();
        let shard_id = ShardId {
            model_id: ModelId("test".into()),
            index: 0,
        };
        let node_a = NodeId([1u8; 32]);
        let node_b = NodeId([2u8; 32]);

        registry.record_shard_holder(shard_id.clone(), node_a.clone());
        registry.record_shard_holder(shard_id.clone(), node_b.clone());

        registry.remove_shard_holder(&shard_id, &node_a);
        let holders = registry.shard_holders(&shard_id);
        assert_eq!(holders.len(), 1);
        assert_eq!(holders[0], node_b);
    }

    #[test]
    fn remove_last_shard_holder_cleans_tombstone() {
        // R142.9: empty-tombstone cleanup must be atomic with the empty
        // check (remove_if). Verify a single-holder removal clears the
        // entry entirely (not just empties the inner HashSet) so a
        // subsequent record_shard_holder starts from a clean slot.
        let registry = ModelRegistry::new();
        let shard_id = ShardId {
            model_id: ModelId("test".into()),
            index: 0,
        };
        let node_a = NodeId([1u8; 32]);
        registry.record_shard_holder(shard_id.clone(), node_a.clone());
        assert_eq!(registry.shard_holder_count(&shard_id), 1);

        registry.remove_shard_holder(&shard_id, &node_a);
        assert_eq!(registry.shard_holder_count(&shard_id), 0);
        assert!(registry.shard_holders(&shard_id).is_empty());

        // Re-inserting a fresh holder after the last was removed must work.
        let node_b = NodeId([2u8; 32]);
        registry.record_shard_holder(shard_id.clone(), node_b.clone());
        assert_eq!(registry.shard_holders(&shard_id), vec![node_b]);
    }

    #[test]
    fn bounded_eviction() {
        let local = NodeId([0u8; 32]);
        let registry = ModelRegistry::with_local_node(local.clone());
        let shard_id = ShardId {
            model_id: ModelId("test".into()),
            index: 0,
        };

        // Register local node first
        registry.record_shard_holder(shard_id.clone(), local.clone());

        // Fill to capacity
        for i in 1..MAX_HOLDERS_PER_SHARD {
            let mut bytes = [0u8; 32];
            bytes[0] = (i & 0xFF) as u8;
            bytes[1] = ((i >> 8) & 0xFF) as u8;
            registry.record_shard_holder(shard_id.clone(), NodeId(bytes));
        }

        assert_eq!(
            registry.shard_holder_count(&shard_id),
            MAX_HOLDERS_PER_SHARD
        );

        // Insert one more — should evict oldest non-local
        let overflow = NodeId([255u8; 32]);
        registry.record_shard_holder(shard_id.clone(), overflow.clone());

        // Still at cap
        assert_eq!(
            registry.shard_holder_count(&shard_id),
            MAX_HOLDERS_PER_SHARD
        );

        // Local node still present
        assert!(registry.shard_holders(&shard_id).contains(&local));
        // New node present
        assert!(registry.shard_holders(&shard_id).contains(&overflow));
    }

    #[test]
    fn global_holder_count_overrides_local_cap() {
        let registry = ModelRegistry::new();
        let shard_id = ShardId {
            model_id: ModelId("test".into()),
            index: 0,
        };

        // No DHT data yet — global_holder_count returns None.
        assert!(registry.global_holder_count(&shard_id).is_none());

        // Cache has 3 holders, DHT reports 247 — the uncapped figure wins
        // for redundancy_ratio purposes.
        for i in 1u8..=3 {
            let mut bytes = [0u8; 32];
            bytes[0] = i;
            registry.record_shard_holder(shard_id.clone(), NodeId(bytes));
        }
        registry.record_global_holder_count(shard_id.clone(), 247);

        assert_eq!(registry.shard_holders(&shard_id).len(), 3);
        assert_eq!(registry.global_holder_count(&shard_id), Some(247));

        // remove_all_model_shards must drop the global entry too — a future
        // shard reusing the ShardId must not inherit a stale 247.
        registry.remove_all_model_shards(&ModelId("test".into()));
        assert!(registry.global_holder_count(&shard_id).is_none());
    }

    #[test]
    fn merge_dht_providers() {
        let registry = ModelRegistry::new();
        let shard_id = ShardId {
            model_id: ModelId("test".into()),
            index: 0,
        };
        let nodes: Vec<NodeId> = (1..=5)
            .map(|i| {
                let mut bytes = [0u8; 32];
                bytes[0] = i;
                NodeId(bytes)
            })
            .collect();

        registry.merge_dht_providers(&shard_id, &nodes);
        assert_eq!(registry.shard_holder_count(&shard_id), 5);
    }

    /// A stale DHT provider record must not resurrect a claim its holder has
    /// withdrawn. Measured live on 2026-08-22: the holder retracted the shard
    /// and re-announced that every 5 minutes, the coordinator re-merged the
    /// stale record every few seconds, and every request was then scheduled
    /// onto a node without the weights — a 503 with no standby, indefinitely.
    #[test]
    fn a_dht_record_cannot_undo_a_holders_own_retraction() {
        let registry = ModelRegistry::new();
        let model_id = ModelId("test".into());
        let shard_id = ShardId {
            model_id: model_id.clone(),
            index: 2,
        };
        let node = NodeId([7u8; 32]);

        // The node held it, and the DHT learned so.
        registry.record_shard_holder(shard_id.clone(), node.clone());
        assert_eq!(registry.shard_holder_count(&shard_id), 1);

        // Then it announced a holding that no longer includes shard 2.
        let keep: std::collections::HashSet<u32> = std::collections::HashSet::new();
        assert_eq!(
            registry.retain_node_shards_for_model(&model_id, &node, &keep),
            1
        );
        assert_eq!(registry.shard_holder_count(&shard_id), 0);

        // The DHT still names it. Without honouring the retraction this puts
        // the claim straight back, which is the whole defect.
        registry.merge_dht_providers(&shard_id, std::slice::from_ref(&node));
        assert_eq!(
            registry.shard_holder_count(&shard_id),
            0,
            "a stale DHT provider record resurrected a retracted claim"
        );

        // But the node's own word is still believed: if it re-acquires the
        // shard and says so, it is a holder again, and the DHT agrees freely.
        registry.record_shard_holder(shard_id.clone(), node.clone());
        assert_eq!(registry.shard_holder_count(&shard_id), 1);
        registry.remove_shard_holder(&shard_id, &node);
        registry.merge_dht_providers(&shard_id, std::slice::from_ref(&node));
        assert_eq!(
            registry.shard_holder_count(&shard_id),
            1,
            "a re-announced holding must clear the retraction"
        );
    }

    /// The other way a holder tells us it lost a shard: it fails a live request
    /// with "shard not found". That is first-hand too, and must stick for every
    /// later request — before this, a per-request blacklist covered the retry
    /// and the DHT re-taught the claim in time for the next one, so a coordinator
    /// restarted while a record was stale failed request after request.
    #[test]
    fn a_retraction_from_a_failed_segment_also_survives_the_dht() {
        let registry = ModelRegistry::new();
        let shard_id = ShardId {
            model_id: ModelId("test".into()),
            index: 2,
        };
        let node = NodeId([9u8; 32]);

        registry.record_shard_holder(shard_id.clone(), node.clone());
        registry.retract_shard_holder(&shard_id, &node);
        assert_eq!(registry.shard_holder_count(&shard_id), 0);

        registry.merge_dht_providers(&shard_id, std::slice::from_ref(&node));
        assert_eq!(
            registry.shard_holder_count(&shard_id),
            0,
            "the DHT reinstated a holder that had just failed for want of the shard"
        );
    }

    #[test]
    fn models_returns_all() {
        let registry = ModelRegistry::new();
        assert_eq!(registry.model_count(), 0);

        registry.register_manifest(test_manifest("a", "A"));

        assert_eq!(registry.model_count(), 1);
        assert_eq!(registry.models().len(), 1);
    }

    #[test]
    fn sentinel_shard_is_mmproj() {
        let shard = ShardId {
            model_id: ModelId("test".into()),
            index: MMPROJ_SHARD_INDEX,
        };
        assert!(shard.is_mmproj());

        let regular = ShardId {
            model_id: ModelId("test".into()),
            index: 0,
        };
        assert!(!regular.is_mmproj());
    }

    #[test]
    fn mmproj_for_creates_sentinel() {
        let shard = ShardId::mmproj_for(ModelId("test".into()));
        assert_eq!(shard.index, MMPROJ_SHARD_INDEX);
        assert!(shard.is_mmproj());
        assert_eq!(shard.model_id, ModelId("test".into()));
    }

    #[test]
    fn mmproj_holders_tracking() {
        let registry = ModelRegistry::new();
        let model_id = ModelId("vlm".into());
        let node_a = NodeId([1u8; 32]);
        let node_b = NodeId([2u8; 32]);

        // Initially no holders
        assert!(registry.mmproj_holders(&model_id).is_empty());

        // Register mmproj sentinel shard
        let sentinel = ShardId::mmproj_for(model_id.clone());
        registry.record_shard_holder(sentinel.clone(), node_a.clone());
        assert_eq!(registry.mmproj_holders(&model_id).len(), 1);

        registry.record_shard_holder(sentinel.clone(), node_b.clone());
        assert_eq!(registry.mmproj_holders(&model_id).len(), 2);

        // Remove one holder
        registry.remove_shard_holder(&sentinel, &node_a);
        let holders = registry.mmproj_holders(&model_id);
        assert_eq!(holders.len(), 1);
        assert_eq!(holders[0], node_b);
    }

    #[test]
    fn shard_announce_includes_mmproj_sentinel() {
        let registry = ModelRegistry::new();
        let model_id = ModelId("vlm".into());
        let node = NodeId([1u8; 32]);

        // Register regular shard and mmproj sentinel
        registry.record_shard_holder(
            ShardId {
                model_id: model_id.clone(),
                index: 0,
            },
            node.clone(),
        );
        registry.record_shard_holder(ShardId::mmproj_for(model_id.clone()), node.clone());

        let entries = registry.all_shard_entries();
        assert_eq!(entries.len(), 2);

        let has_mmproj = entries.iter().any(|(sid, _)| sid.is_mmproj());
        assert!(has_mmproj);
    }
}
