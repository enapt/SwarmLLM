//! A bound on the models this node learns of from peers (FUTURE_WORK #333).
//!
//! A peer's key costs nothing to make, and a manifest under a new model id was
//! adopted whatever it named: each one kept a registry entry and its parts'
//! heard hashes, and every model listing walks them all (`compute_wishlist`,
//! `/v1/models`, the dashboard). Two systems with more scars bound the same
//! thing. libp2p's Kademlia record store refuses a new record once it holds
//! `max_records` (1,024: `MemoryStore::put` → `Error::MaxRecords`). Bitcoin's
//! address manager lets a newcomer take an occupied slot only from an entry
//! that is "terrible" (`AddrInfo::IsTerrible`: stale, or never seen to work) —
//! an entry in good standing is never displaced by an unknown one.
//!
//! So at most [`MOST_MODELS`] models adopted as new from peers, weighing at
//! most [`MOST_BYTES`] (a manifest's stored size, [`weight_of`]), none more
//! than [`MOST_BYTES_A_MODEL`]. Past a bound, a model nobody holds and this
//! node has not acted on ([`SharedState::acted_on_model`]) makes room, the
//! longest known first; one that is kept is looked at again later (CLOCK's
//! second chance), and one arrival looks at no more than [`LOOKED_AT_A_TIME`].
//! With none to spare the newcomer is refused, never an entry in good
//! standing. "Holds" counts this node and the nodes CONNECTED to it only
//! ([`SharedState::holder_standing`]): a part announcement needs no more than a
//! key, and a relayed origin in `peer_registry` costs no more (security
//! reviews, #333 and #339), so counting either would let fresh keys keep their
//! own fakes. The wishlist is not a reason to keep one either:
//! it is built from this registry, so a peer could steer its own entries onto
//! it.
//!
//! A model this node knew before a peer told it (its own, or its database's)
//! is not weighed here — and a peer's manifest is stored only for a model with
//! a part here (`register_manifest`), so the database cannot carry one past a
//! restart. Holder records have their own bound, per key
//! (`registry::MOST_PARTS_A_HOLDER`).

use std::collections::{HashMap, VecDeque};

use crate::types::{ModelId, ModelManifest, NodeId, ShardId};

use super::SharedState;

/// Models adopted as new from peers — libp2p-kad's `max_records`. The swarm
/// knows a few dozen.
pub const MOST_MODELS: usize = 1024;

/// What those models weigh in all, in [`weight_of`]. A real manifest is
/// 20-111 KB (the live node's 28 models, 2026-10-24), so 600-3,000 real ones
/// fit; what they cost in memory is about what they weigh stored.
pub const MOST_BYTES: usize = 64 << 20;

/// The most one peer's manifest may weigh: ~9x the largest real one. The
/// message cap alone (4 MB) let one manifest of two parts carry 4 MB of names
/// and licence text, which the part and tensor counts never see.
pub const MOST_BYTES_A_MODEL: usize = 1 << 20;

/// Learned models one arrival looks at for room, across all the room it needs,
/// so a flood of new ids costs a bounded walk — and at most this many stored
/// rows read — each.
const LOOKED_AT_A_TIME: usize = 64;

/// What a manifest weighs: its size stored, as JSON — every string counted,
/// which a count of parts and tensors is not.
pub fn weight_of(manifest: &ModelManifest) -> usize {
    struct Bytes(usize);
    impl std::io::Write for Bytes {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0 = self.0.saturating_add(buf.len());
            Ok(buf.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    let mut bytes = Bytes(0);
    match serde_json::to_writer(&mut bytes, manifest) {
        Ok(()) => bytes.0,
        // Our own well-typed struct does not fail to serialize; if it did,
        // what cannot be weighed is not admitted.
        Err(_) => usize::MAX,
    }
}

/// `state.models.peer_models`: the models adopted as new from peers, in the
/// order they were first adopted, and what each weighs. Written only by
/// [`SharedState::admit_a_peers_manifest`].
#[derive(Default)]
pub struct PeerModels {
    learned: parking_lot::Mutex<Learned>,
}

#[derive(Default)]
struct Learned {
    /// Each learned model once, longest known first (a kept one moves back).
    order: VecDeque<ModelId>,
    bytes: HashMap<ModelId, usize>,
    total_bytes: usize,
}

impl Learned {
    fn forget(&mut self, model: &ModelId) {
        if let Some(bytes) = self.bytes.remove(model) {
            self.total_bytes -= bytes;
        }
    }
}

/// How a learned model's room was freed.
enum Room {
    /// Forgotten here to make room.
    Forgotten(ModelId),
    /// Already gone from the registry by another path (an owner's delete, the
    /// copy repair's replacement) — which owns its stored row, and may be
    /// writing it again right now.
    AlreadyGone,
}

impl PeerModels {
    /// How many models peers have told this node of (tests, diagnostics).
    pub fn len(&self) -> usize {
        self.learned.lock().order.len()
    }
}

impl SharedState {
    /// May this peer's manifest be registered? `false` refuses it; `true` may
    /// first have forgotten models nobody holds to make room. Asked only by
    /// [`SharedState::judge_peer_manifest`], the one door a peer's manifest
    /// comes in by, after every other verdict.
    pub(super) fn admit_a_peers_manifest(&self, manifest: &ModelManifest) -> bool {
        let id = &manifest.id;
        let known = self.model_registry.part_indices(id).is_some();
        let mut forgotten: Vec<ModelId> = Vec::new();
        let admitted = {
            let mut learned = self.models.peer_models.learned.lock();
            let was = learned.bytes.get(id).copied();
            if known && was.is_none() {
                // Known before any peer told us: not a peer's to weigh.
                return true;
            }
            let bytes = weight_of(manifest);
            if bytes > MOST_BYTES_A_MODEL {
                return false;
            }
            let mut looks = LOOKED_AT_A_TIME;
            let admitted = loop {
                let over_count = was.is_none() && learned.order.len() >= MOST_MODELS;
                let over_bytes = learned.total_bytes - was.unwrap_or(0) + bytes > MOST_BYTES;
                if !over_count && !over_bytes {
                    break true;
                }
                match self.room_from_a_model_nobody_holds(&mut learned, id, &mut looks) {
                    Some(Room::Forgotten(gone)) => forgotten.push(gone),
                    Some(Room::AlreadyGone) => {}
                    None => break false,
                }
            };
            if admitted {
                if was.is_none() {
                    learned.order.push_back(id.clone());
                }
                learned.total_bytes = learned.total_bytes - was.unwrap_or(0) + bytes;
                learned.bytes.insert(id.clone(), bytes);
            }
            admitted
        };
        // A forgotten model's manifest may have been stored before this
        // build stored only models with a part here: it would come back at the
        // next start, outside this bound. Read first — a removal waits for the
        // disk.
        for model in &forgotten {
            if matches!(
                self.db
                    .get_json::<serde::de::IgnoredAny>("model_meta", &model.0),
                Ok(Some(_))
            ) {
                if let Err(e) = self.db.remove("model_meta", &model.0) {
                    tracing::warn!(model = %model, error = %e, "Could not remove a forgotten model's stored manifest");
                }
            }
        }
        if !forgotten.is_empty() {
            tracing::debug!(
                model = %id,
                forgotten = forgotten.len(),
                learned = self.models.peer_models.len(),
                "Forgot models nobody holds to make room for one a peer named"
            );
        }
        admitted
    }

    /// Free one learned model's room: a model this node no longer has (the
    /// owner deleted it), or one nobody holds and this node has not acted on —
    /// the longest known first, each look spending one of `looks`. `None`
    /// when they run out first.
    fn room_from_a_model_nobody_holds(
        &self,
        learned: &mut Learned,
        newcomer: &ModelId,
        looks: &mut usize,
    ) -> Option<Room> {
        while *looks > 0 {
            *looks -= 1;
            let model = learned.order.pop_front()?;
            if &model == newcomer {
                learned.order.push_back(model);
                continue;
            }
            let Some(parts) = self.model_registry.part_indices(&model) else {
                learned.forget(&model);
                return Some(Room::AlreadyGone);
            };
            let held = parts.iter().any(|&index| {
                self.model_registry.part_has_a_holder_where(
                    &ShardId {
                        model_id: model.clone(),
                        index,
                    },
                    |holder| self.holder_counts(holder),
                )
            });
            if held || self.acted_on_model(&model) {
                learned.order.push_back(model);
                continue;
            }
            self.model_registry.forget_a_model_nobody_holds(&model);
            learned.forget(&model);
            return Some(Room::Forgotten(model));
        }
        None
    }

    /// Does `holder`'s record keep a model: this node, or a node CONNECTED to
    /// it ([`Self::holder_standing`])? Not a key that only ever announced a
    /// part, and not a registered relayed origin: up to 64 sit in
    /// `peer_registry` at once and anyone can mint one (security review, #339).
    fn holder_counts(&self, holder: &NodeId) -> bool {
        self.holder_standing(holder).is_some()
    }

    /// Where a holder of a part stands with this node — the one answer for the
    /// registry's full lists, its age-out and [`Self::holder_counts`] (#339):
    /// `None` when it is not connected; `Some(network)` when it is this node or
    /// connected, `network` being a key for the network it connects from
    /// (`connected_networks`; a node with none on record has its own).
    pub(crate) fn holder_standing(&self, holder: &NodeId) -> Option<u64> {
        use std::hash::{Hash, Hasher};
        if holder != self.identity.node_id() && !self.connected_node_ids.contains(holder) {
            return None;
        }
        let Some(network) = self.connected_networks.get(holder).map(|n| *n) else {
            return Some(crate::model::registry::own_network(holder));
        };
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        network.hash(&mut hasher);
        Some(hasher.finish())
    }

    /// Has this node done something with this model a peer's word cannot
    /// undo: fetching it (or tried to), fetched part of it from its origin,
    /// verified its upload, pinned or kept here by its owner, or in use now?
    /// Such a model is never forgotten to make room (#333).
    pub(super) fn acted_on_model(&self, model: &ModelId) -> bool {
        self.models.model_download_under_way(model)
            || self.models.acquisition_progress.contains_key(model)
            || self.model_registry.has_origin_knowledge(model)
            || self.models.canonical_builds.contains_key(model)
            || self
                .models
                .model_trust
                .get(model)
                .is_some_and(|t| t.pinned_by_user)
            || self.models.kept_on_this_computer.contains(model)
            || self.model_is_in_use(model)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::daemon::state::PeerManifest;
    use crate::types::{ModelArchitecture, ShardInfo, ShardTensorEntry};

    fn test_state() -> std::sync::Arc<SharedState> {
        let dir = tempfile::tempdir().unwrap().keep();
        let mut config = crate::config::Config::default();
        config.node.data_dir = dir.clone();
        let db = crate::storage::db::Database::open(&dir).unwrap();
        let executor = std::sync::Arc::new(tokio::sync::Mutex::new(
            crate::inference::executor::ModelExecutor::new(),
        ));
        let (state, _, _) = SharedState::new(
            config,
            crate::identity::Identity::generate(),
            db,
            executor,
            None,
        );
        state
    }

    /// A one-part manifest of `tensors` tensor entries, its part's hash set
    /// (so it is heard).
    fn manifest(id: &str, tensors: usize) -> ModelManifest {
        crate::model::manifest::build_manifest_from_gguf(crate::model::manifest::ManifestFromGguf {
            id: ModelId(id.into()),
            name: id.into(),
            architecture: ModelArchitecture::Llama,
            num_layers: 4,
            total_size_bytes: 1_000,
            shard_count: 1,
            shards: vec![ShardInfo {
                index: 0,
                layer_range: (0, 4),
                size_bytes: 990,
                hash: [7; 32],
                tensors: (0..tensors)
                    .map(|i| ShardTensorEntry {
                        name: format!("t{i}"),
                        gguf_offset: i as u64,
                        shard_offset: i as u64,
                        size: 1,
                    })
                    .collect(),
            }],
            publisher: NodeId([9; 32]),
            context_length: None,
        })
    }

    /// A manifest weighing about `bytes`, nearly all of it licence text — a
    /// string, which no count of parts or tensors sees.
    fn weighing(id: &str, bytes: usize) -> ModelManifest {
        let mut m = manifest(id, 0);
        m.license.clear();
        let base = weight_of(&m);
        m.license = "x".repeat(bytes.saturating_sub(base));
        assert_eq!(weight_of(&m), bytes);
        m
    }

    /// The dispatcher's door: judge, then register what it adopts.
    fn arrive(state: &SharedState, m: &ModelManifest) -> PeerManifest {
        let verdict = state.judge_peer_manifest(m);
        if verdict == PeerManifest::Adopt {
            state.model_registry.register_manifest(m.clone());
        }
        verdict
    }

    /// A peer this node is connected to.
    fn connect(state: &SharedState, node: &NodeId) {
        state.connected_node_ids.insert(node.clone());
        register(state, node);
    }

    /// A node in `peer_registry` that is not connected — what a proven
    /// relayed origin is.
    fn register(state: &SharedState, node: &NodeId) {
        state.peer_registry.insert(
            node.clone(),
            crate::types::PeerInfo {
                node_id: node.clone(),
                addresses: vec![],
                capability: None,
                last_seen: chrono::Utc::now(),
                latency_ms: Some(50),
                trust_score: 0.5,
                peer_id_bytes: None,
                ack_srtt_ms: None,
                active_request_count: 0,
                first_seen: 0,
                verified_transaction_count: 0,
                is_lan_peer: false,
                goodput_bytes_per_sec: None,
                goodput_samples: 0,
            },
        );
    }

    fn part(id: &str) -> ShardId {
        part_n(id, 0)
    }

    /// #339's review: up to 64 relayed origins sit in `peer_registry` at once
    /// and anyone can mint one, so a registered node that is not connected
    /// keeps nothing either.
    #[test]
    fn a_registered_node_that_is_not_connected_keeps_nothing() {
        let state = test_state();
        let relayed = NodeId([3; 32]);
        register(&state, &relayed);
        for i in 0..MOST_MODELS {
            arrive(&state, &manifest(&format!("m{i}"), 0));
        }
        state
            .model_registry
            .record_shard_holder(part("m0"), relayed.clone());
        assert_eq!(arrive(&state, &manifest("new-a", 0)), PeerManifest::Adopt);
        assert!(
            !knows(&state, "m0"),
            "the oldest went, its holder not connected"
        );

        state.connected_node_ids.insert(relayed.clone());
        state
            .model_registry
            .record_shard_holder(part("m1"), relayed);
        assert_eq!(arrive(&state, &manifest("new-b", 0)), PeerManifest::Adopt);
        assert!(knows(&state, "m1"), "a connected holder keeps it");
    }

    #[test]
    fn a_holders_standing_is_its_connection_and_the_network_it_comes_from() {
        use crate::network::helpers::{NetGroup, Network};
        let state = test_state();
        let me = state.identity.node_id().clone();
        let (a, b, c, d) = (
            NodeId([1; 32]),
            NodeId([2; 32]),
            NodeId([3; 32]),
            NodeId([4; 32]),
        );
        for n in [&a, &b, &c] {
            connect(&state, n);
        }
        register(&state, &d);
        let one_isp = NetGroup::Direct(Network::Ipv4Slash16([203, 0]));
        state.connected_networks.insert(a.clone(), one_isp);
        state.connected_networks.insert(b.clone(), one_isp);

        assert!(state.holder_standing(&me).is_some(), "this node");
        assert_eq!(state.holder_standing(&a), state.holder_standing(&b));
        assert!(state.holder_standing(&a).is_some());
        let c_net = state
            .holder_standing(&c)
            .expect("connected, no network on record");
        assert_ne!(
            Some(c_net),
            state.holder_standing(&a),
            "a network of its own"
        );
        assert_eq!(state.holder_standing(&d), None, "registered, not connected");
        assert_eq!(state.holder_standing(&NodeId([9; 32])), None);
    }

    fn part_n(id: &str, index: u32) -> ShardId {
        ShardId {
            model_id: ModelId(id.into()),
            index,
        }
    }

    fn knows(state: &SharedState, id: &str) -> bool {
        state
            .model_registry
            .part_indices(&ModelId(id.into()))
            .is_some()
    }

    fn heard(registry: &crate::model::registry::ModelRegistry, shard: &ShardId) -> bool {
        registry
            .heard_hash_with_tag(shard, swarmllm_types::build_tag_from_hash(&[7; 32]), 990)
            .is_some()
    }

    /// The bound: past `MOST_MODELS` a newcomer takes the room of the model
    /// nobody holds that was known longest — its manifest and heard hashes go
    /// — and the count never grows past the bound.
    #[test]
    fn past_the_bound_a_model_nobody_holds_makes_room_the_oldest_first() {
        let state = test_state();
        for i in 0..MOST_MODELS {
            assert_eq!(
                arrive(&state, &manifest(&format!("made-up-{i}"), 0)),
                PeerManifest::Adopt
            );
        }
        assert!(heard(&state.model_registry, &part("made-up-0")));
        assert_eq!(
            arrive(&state, &manifest("newcomer", 0)),
            PeerManifest::Adopt
        );
        assert!(knows(&state, "newcomer"));
        assert!(!knows(&state, "made-up-0"), "the longest known goes");
        assert!(
            !heard(&state.model_registry, &part("made-up-0")),
            "its heard hashes go with it"
        );
        assert!(knows(&state, "made-up-1"));
        assert_eq!(state.models.peer_models.len(), MOST_MODELS);
    }

    /// An entry in good standing is never displaced: a model a connected peer
    /// holds, or one this node acted on, is kept, and a newcomer finding only
    /// those is refused (libp2p-kad's `MaxRecords`). A holder record from a
    /// key this node does not know keeps nothing — announcing a part needs no
    /// more than a fresh key.
    #[test]
    fn a_model_held_or_acted_on_is_kept_and_with_no_room_the_newcomer_is_refused() {
        let state = test_state();
        let peer = NodeId([1; 32]);
        connect(&state, &peer);
        let stranger = NodeId([2; 32]);
        for i in 0..MOST_MODELS {
            arrive(&state, &manifest(&format!("m{i}"), 0));
        }
        // m0 a peer holds; m1 its owner pinned; m2 this node tried to fetch;
        // m3 its owner keeps here; m4 only a stranger "holds".
        state
            .model_registry
            .record_shard_holder(part("m0"), peer.clone());
        state.models.model_trust.insert(
            ModelId("m1".into()),
            crate::types::ModelTrustInfo {
                pinned_by_user: true,
                ..crate::types::ModelTrustInfo::new_discovered()
            },
        );
        let mut failed = crate::model::acquisition::AcquisitionStatus::new_downloading(
            ModelId("m2".into()),
            1,
            990,
            "swarm",
            "test",
            "",
        );
        // Not under way any more — a failed attempt still counts.
        failed.state = crate::model::acquisition::AcquisitionState::Failed {
            reason: "test".into(),
        };
        state
            .models
            .acquisition_progress
            .insert(ModelId("m2".into()), failed);
        state
            .models
            .kept_on_this_computer
            .insert(ModelId("m3".into()));
        state
            .model_registry
            .record_shard_holder(part("m4"), stranger);
        assert_eq!(arrive(&state, &manifest("new-a", 0)), PeerManifest::Adopt);
        for kept in ["m0", "m1", "m2", "m3", "m5"] {
            assert!(knows(&state, kept), "{kept} is kept");
        }
        assert!(
            !knows(&state, "m4"),
            "a stranger's holder record keeps nothing"
        );

        // Every model now held: nothing may make room.
        for i in 0..MOST_MODELS {
            state
                .model_registry
                .record_shard_holder(part(&format!("m{i}")), peer.clone());
        }
        state
            .model_registry
            .record_shard_holder(part("new-a"), peer.clone());
        assert_eq!(arrive(&state, &manifest("new-b", 0)), PeerManifest::NoRoom);
        assert!(!knows(&state, "new-b"));
        assert_eq!(state.models.peer_models.len(), MOST_MODELS);
    }

    /// What the models weigh is bounded too, by every byte stored — strings
    /// included: a few heavy manifests fill `MOST_BYTES` long before the count;
    /// one arrival looks at no more than `LOOKED_AT_A_TIME` models for room,
    /// however much it needs; and one past `MOST_BYTES_A_MODEL` is refused
    /// outright.
    #[test]
    fn what_the_learned_models_weigh_is_bounded_and_one_too_heavy_is_refused() {
        let state = test_state();
        let lights = LOOKED_AT_A_TIME + 36;
        for i in 0..lights {
            assert_eq!(
                arrive(&state, &manifest(&format!("l{i}"), 0)),
                PeerManifest::Adopt
            );
        }
        let light = weight_of(&manifest("l00", 0));
        // 64 heavy ones that leave less than one light one's room over.
        let used = state.models.peer_models.learned.lock().total_bytes;
        let heavy = weighing("h00", (MOST_BYTES - used) / 64);
        assert!(weight_of(&heavy) <= MOST_BYTES_A_MODEL);
        let fit = (MOST_BYTES - used) / weight_of(&heavy);
        for i in 0..fit {
            let mut h = heavy.clone();
            h.id = ModelId(format!("h{i:02}"));
            assert_eq!(arrive(&state, &h), PeerManifest::Adopt);
        }
        let total = state.models.peer_models.learned.lock().total_bytes;
        let need = total + weight_of(&heavy) - MOST_BYTES;
        assert!(
            need > LOOKED_AT_A_TIME * light,
            "the next heavy one needs more room than one walk of light ones frees"
        );

        // One walk: the 64 oldest light ones go, and that is not room enough.
        let mut next = heavy.clone();
        next.id = ModelId("h-next".into());
        assert_eq!(arrive(&state, &next), PeerManifest::NoRoom);
        let gone = (0..lights)
            .filter(|i| !knows(&state, &format!("l{i}")))
            .count();
        assert_eq!(gone, LOOKED_AT_A_TIME, "one arrival, one bounded walk");

        // The next walk passes the 36 light ones left and reaches a heavy one.
        assert_eq!(arrive(&state, &next), PeerManifest::Adopt);
        assert!(
            !knows(&state, "h00"),
            "room is made by weight as well as count"
        );
        assert!(state.models.peer_models.learned.lock().total_bytes <= MOST_BYTES);

        let too_heavy = weighing("too-heavy", MOST_BYTES_A_MODEL + 1);
        assert_eq!(arrive(&state, &too_heavy), PeerManifest::NoRoom);
        assert!(!knows(&state, "too-heavy"));
    }

    /// A model this node knew before any peer told it is not a peer's to
    /// weigh, and one the owner deleted frees its room at once.
    #[test]
    fn a_model_known_here_first_is_not_weighed_and_a_deleted_one_frees_its_room() {
        let state = test_state();
        let peer = NodeId([1; 32]);
        connect(&state, &peer);
        let ours = manifest("ours", 0);
        state.model_registry.register_manifest(ours.clone());
        assert_eq!(arrive(&state, &ours), PeerManifest::Adopt);
        assert_eq!(state.models.peer_models.len(), 0);

        for i in 0..MOST_MODELS {
            arrive(&state, &manifest(&format!("m{i}"), 0));
            state
                .model_registry
                .record_shard_holder(part(&format!("m{i}")), peer.clone());
        }
        state.model_registry.remove_manifest(&ModelId("m0".into()));
        assert_eq!(
            arrive(&state, &manifest("newcomer", 0)),
            PeerManifest::Adopt
        );
        assert!(knows(&state, "m1"), "a held one is not what made the room");
        assert_eq!(state.models.peer_models.len(), MOST_MODELS);
    }

    /// A forgotten model's stored manifest is removed too, or it would come
    /// back at the next start outside the bound.
    #[test]
    fn a_forgotten_models_stored_manifest_is_removed() {
        let state = test_state();
        for i in 0..MOST_MODELS {
            arrive(&state, &manifest(&format!("m{i}"), 0));
        }
        state
            .db
            .put_json("model_meta", "m0", &manifest("m0", 0))
            .unwrap();
        arrive(&state, &manifest("newcomer", 0));
        assert!(!knows(&state, "m0"));
        assert!(state
            .db
            .get_json::<ModelManifest>("model_meta", "m0")
            .unwrap()
            .is_none());
    }

    /// Forgetting a model clears the hashes heard for every part number it
    /// could have had, not only the ones its last manifest named: an earlier
    /// manifest under the same id may have named others.
    #[test]
    fn forgetting_a_model_clears_the_hashes_of_parts_its_last_manifest_did_not_name() {
        let registry = crate::model::registry::ModelRegistry::new();
        let mut earlier = manifest("s", 0);
        earlier.shards[0].index = 5;
        registry.register_manifest(earlier);
        registry.register_manifest(manifest("s", 0));
        assert!(heard(&registry, &part_n("s", 5)));
        assert_eq!(registry.part_indices(&ModelId("s".into())), Some(vec![0]));
        registry.forget_a_model_nobody_holds(&ModelId("s".into()));
        assert!(!heard(&registry, &part_n("s", 5)));
        assert!(!heard(&registry, &part_n("s", 0)));
    }

    /// A peer may name any shard id: past `MOST_PARTS_A_HOLDER` no new claim
    /// of it is recorded, its claims already held are renewed, and this node's
    /// own are never refused.
    #[test]
    fn one_holder_cannot_fill_the_holder_records() {
        use crate::model::registry::MOST_PARTS_A_HOLDER;
        let me = NodeId([3; 32]);
        let registry = crate::model::registry::ModelRegistry::with_local_node(me.clone());
        let peer = NodeId([1; 32]);
        let shard = |i: usize| ShardId {
            model_id: ModelId(format!("fake-{i}")),
            index: 0,
        };
        let held = |s: &ShardId| registry.part_has_a_holder_where(s, |_| true);
        for i in 0..MOST_PARTS_A_HOLDER {
            assert!(registry.record_shard_holder(shard(i), peer.clone()));
        }
        assert!(!registry.record_shard_holder(shard(MOST_PARTS_A_HOLDER), peer.clone()));
        assert!(!held(&shard(MOST_PARTS_A_HOLDER)));
        registry.record_shard_holder(shard(0), peer.clone());
        assert!(held(&shard(0)), "a held claim is renewed");
        for i in 0..=MOST_PARTS_A_HOLDER {
            assert!(registry.record_shard_holder(shard(i + 10_000), me.clone()));
        }
    }
}
