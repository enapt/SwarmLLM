//! A split that ran with nothing to guess for it fetches a model that can
//! (FUTURE_WORK #134, "acquiring a drafter a node does not already hold").
//!
//! Across a network a split's speed is the tokens each round trip keeps: with
//! no drafter a round keeps one, with a same-family small model 3-6
//! (`docs/plans/large_model_splits.md`). The drafter was chosen only among the
//! models this node holds whole (`pipeline::engine_drafter`), so a node that
//! had never fetched one guessed with prompt lookup alone on every split. Here
//! a split that found none (`engine_drafter::note_split_wanting_a_drafter`) has
//! the download pass fetch the one `engine_drafter::drafter_to_acquire` names —
//! a model the swarm already holds, no larger than 2 GiB, its parts through the
//! shard system like any other part, never a whole file and never the target —
//! and prune keeps it while such splits keep coming
//! (`engine_drafter::DRAFTER_WANTED_FOR`).
//!
//! A bigger drafter keeps more of each round: replayed on Qwen2.5-Coder-7B's
//! own replies, a 1.5B kept 6.24 tokens in a best tree of 64 against the 0.5B's
//! 5.18, and 3.21 against 2.81 in a chain of four (2026-10-22,
//! `examples/spec_coverage.py`, `spec_best_tree.py`) — so the largest that can
//! draft is fetched, as `chosen` picks the largest held.
//!
//! And a node already holding a drafter fetches a larger one where the swarm
//! has it (#331): the 3B flowed 18.1 tok/s against the 1.5B's 16.0 on the real
//! link, and a node that fetched the 1.5B under 0.3.248's quarter rule guessed
//! with it for good. The held one stays kept while the larger one comes, and
//! goes only once that one is held and chosen (`DrafterToAcquire::keeps`).

use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};

use dashmap::DashMap;

use crate::inference::pipeline::{drafter_to_acquire, DrafterToAcquire, DRAFTER_WANTED_FOR};
use crate::types::{ModelId, ModelManifest, ModelTrustLevel, ShardId, MMPROJ_SHARD_INDEX};

use super::super::AutoShardManager;

/// How long one target's answer is kept: prune asks about every part it holds,
/// and the inputs move on gossip's minutes.
const ANSWERS_FOR: Duration = Duration::from_secs(60);

/// What the drafter fetch remembers between asks: each target's answer.
#[derive(Default)]
pub(in crate::model::auto_manage) struct DrafterBook {
    answers: DashMap<ModelId, (Instant, DrafterToAcquire)>,
}

impl AutoShardManager {
    /// The targets a split here ran without a drafter within
    /// [`DRAFTER_WANTED_FOR`].
    fn targets_wanting_a_drafter(&self) -> Vec<ModelId> {
        let now = Instant::now();
        self.shared_state
            .models
            .drafters_wanted
            .iter()
            .filter(|e| now.duration_since(*e.value()) < DRAFTER_WANTED_FOR)
            .map(|e| e.key().clone())
            .collect()
    }

    /// `target`'s answer, as last computed within [`ANSWERS_FOR`].
    fn drafter_answer(&self, target: &ModelId) -> DrafterToAcquire {
        let now = Instant::now();
        if let Some(answer) = self
            .drafter_book
            .answers
            .get(target)
            .filter(|a| now < a.0)
            .map(|a| a.1.clone())
        {
            return answer;
        }
        let answer = drafter_to_acquire(&self.shared_state, target, |m| {
            self.may_fetch_as_a_drafter(m)
        });
        self.drafter_book
            .answers
            .insert(target.clone(), (now + ANSWERS_FOR, answer.clone()));
        answer
    }

    /// The download pass's own gates on a model it was not asked for, asked of
    /// a drafter candidate: not a backup copy, a family this build runs, its
    /// upload settled, not switched off by policy, no part deleted by the
    /// owner, trusted the way a new model
    /// must be to be adopted, not one with its safety training removed that the
    /// owner did not choose — and a header that can be learned: here already,
    /// or a HuggingFace source not tried in vain lately
    /// (`SharedState::ensure_model_geometry`'s cooldown).
    fn may_fetch_as_a_drafter(&self, m: &ModelManifest) -> bool {
        let state = &self.shared_state;
        if crate::model::manifest::is_backup_artifact_id(&m.id.0)
            || state.refused_model(&m.id).is_some()
            || !state.canonical_allows_acquisition(&m.id)
            || state.unchosen_uncensored_model(&m.id)
        {
            return false;
        }
        if state
            .models
            .model_auto_manage_policies
            .get(&m.id)
            .is_some_and(|p| !p.enabled)
        {
            return false;
        }
        // A part the owner deleted is never fetched back on our own
        // (`scoring`), so a model with one would be named and never come.
        if m.shards.iter().any(|s| {
            state.shard_removed_by_user(&ShardId {
                model_id: m.id.clone(),
                index: s.index,
            })
        }) {
            return false;
        }
        let trusted =
            state.models.model_trust.get(&m.id).is_some_and(|t| {
                t.pinned_by_user || t.trust_level >= ModelTrustLevel::DemandVerified
            });
        if !trusted {
            return false;
        }
        if state.gguf_meta_for(&m.id).is_some() {
            return true;
        }
        state.models.hf_sources.contains_key(&m.id)
            && !state
                .models
                .geometry_probe_retry_after
                .get(&m.id)
                .is_some_and(|until| Instant::now() < *until)
    }

    /// The drafters to fetch now, one per target wanting one, each with the
    /// drafter held here it would replace (#331), if any.
    fn drafters_to_fetch(&self) -> HashMap<ModelId, Option<ModelId>> {
        self.targets_wanting_a_drafter()
            .iter()
            .filter_map(|target| {
                let answer = self.drafter_answer(target);
                let fetch = answer.fetching()?.clone();
                Some((fetch, answer.held))
            })
            .collect()
    }

    /// The parts of `manifest` this node fetches because it is the drafter a
    /// split here lacks, or a larger one than it holds: every part it does not
    /// hold — and the drafter it replaces, if any. Like a near chain's they
    /// skip the replica target and the hash ring — the target counts the
    /// swarm's copies, and a drafter must be HERE — and nothing else; prune
    /// keeps them ([`Self::copy_drafts_for_a_split`]).
    pub(in crate::model::auto_manage) fn parts_of_a_drafter_to_fetch(
        &self,
        manifest: &ModelManifest,
    ) -> (HashSet<u32>, Option<ModelId>) {
        if self.shared_state.models.drafters_wanted.is_empty() {
            return (HashSet::new(), None);
        }
        let Some(replacing) = self.drafters_to_fetch().remove(&manifest.id) else {
            return (HashSet::new(), None);
        };
        let local = self.shared_state.identity.node_id();
        let parts = manifest
            .shards
            .iter()
            .filter(|s| s.index != MMPROJ_SHARD_INDEX)
            .map(|s| s.index)
            .filter(|&index| {
                !self
                    .shared_state
                    .model_registry
                    .shard_holders(&ShardId {
                        model_id: manifest.id.clone(),
                        index,
                    })
                    .contains(local)
            })
            .collect();
        (parts, replacing)
    }

    /// Is `shard_id` a part of the drafter a split here guesses with, or is
    /// about to — held and chosen, or being fetched for it, beside the held
    /// one it replaces (one cached answer, [`DrafterToAcquire::keeps`])? Prune keeps such a part below
    /// urgent pressure while splits keep wanting it
    /// ([`DRAFTER_WANTED_FOR`]); the download pass asks the same before it
    /// fetches, so a fetched part is not shed again.
    pub(in crate::model::auto_manage) fn copy_drafts_for_a_split(
        &self,
        shard_id: &ShardId,
    ) -> bool {
        if shard_id.index == MMPROJ_SHARD_INDEX
            || self.shared_state.models.drafters_wanted.is_empty()
        {
            return false;
        }
        let model = &shard_id.model_id;
        // The cached answer only: prune asks of every part it holds.
        self.targets_wanting_a_drafter()
            .iter()
            .any(|target| self.drafter_answer(target).keeps(model))
    }

    /// Learn the headers [`drafter_to_acquire`] names before it chooses — a
    /// few MB each. Not offline, not on a metered connection.
    pub(in crate::model::auto_manage) async fn learn_headers_drafters_need(&self) {
        let targets = self.targets_wanting_a_drafter();
        if targets.is_empty()
            || self
                .shared_state
                .credits
                .offline_mode
                .load(std::sync::atomic::Ordering::Relaxed)
        {
            return;
        }
        let mut wanted: Vec<ModelId> = Vec::new();
        for target in &targets {
            wanted.extend_from_slice(self.drafter_answer(target).headers_to_learn());
        }
        wanted.sort_by(|a, b| a.0.cmp(&b.0));
        wanted.dedup();
        if wanted.is_empty() || self.shared_state.metered_hold().await {
            return;
        }
        for model in wanted {
            self.shared_state.ensure_model_geometry(&model).await;
        }
        for target in &targets {
            self.drafter_book.answers.remove(target);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::Ordering::Relaxed;
    use std::time::{Duration, Instant};

    use super::super::super::test_support::{
        fetched, holds, make_test_manager_with_config, offered, peer_with_disk,
        register_manifest_with_sized_shards,
    };
    use crate::config::Config;
    use crate::inference::pipeline::{
        drafter_to_acquire, note_split_wanting_a_drafter, DrafterStep, DrafterToAcquire,
    };
    use crate::types::{ModelId, ModelTrustInfo, ModelTrustLevel, ShardId};

    const MIB: u64 = 1024 * 1024;

    fn trusted(state: &crate::daemon::SharedState, mid: &ModelId) {
        let mut trust = ModelTrustInfo::new_discovered();
        trust.trust_level = ModelTrustLevel::DemandVerified;
        state.models.model_trust.insert(mid.clone(), trust);
    }

    /// A drafter a split here lacked is fetched past the replica target its
    /// one holder already meets, and prune keeps it; with no split wanting one
    /// it is neither fetched nor kept (the null control, both ways). Which model
    /// is the drafter comes from `engine_drafter::drafter_to_acquire` (tested
    /// there); here its answer is planted.
    #[test]
    fn the_drafter_a_split_lacked_is_fetched_and_kept() {
        let mut config = Config::default();
        config.auto_manage.min_replicas = 1;
        let (state, manager) = make_test_manager_with_config(config);
        state.models.auto_manage_default_model_cap.store(0, Relaxed);
        let target = register_manifest_with_sized_shards(
            &state,
            "big-7b",
            28,
            &[(0, 14), (14, 28)],
            2048 * MIB,
        );
        let drafter = register_manifest_with_sized_shards(
            &state,
            "small-0.5b",
            24,
            &[(0, 8), (8, 16), (16, 24)],
            500 * MIB,
        );
        let peer = peer_with_disk(&state, 2, None, 100_000);
        let other = peer_with_disk(&state, 3, None, 100_000);
        holds(&state, &target, &peer, &[0, 1]);
        holds(&state, &drafter, &peer, &[0, 1, 2]);
        holds(&state, &drafter, &other, &[0, 1, 2]);
        trusted(&state, &target);
        trusted(&state, &drafter);
        let planted = || {
            manager.drafter_book.answers.insert(
                target.clone(),
                (
                    Instant::now() + Duration::from_secs(3600),
                    DrafterToAcquire {
                        held: None,
                        next: DrafterStep::Fetch(drafter.clone()),
                    },
                ),
            );
        };
        planted();
        assert!(
            offered(&manager, &state).is_empty(),
            "control: no split wanted a drafter, and the replica target is met"
        );

        state
            .models
            .drafters_wanted
            .insert(target.clone(), Instant::now());
        assert_eq!(offered(&manager, &state), vec![0, 1, 2]);
        let started = fetched(&manager, &state);
        assert!(
            !started.is_empty() && started.iter().all(|i| [0, 1, 2].contains(i)),
            "started {started:?}"
        );

        let local = state.identity.node_id().clone();
        // A MIDDLE part, still alone here: prompt privacy keeps a whole
        // model's ends, and a model held whole is kept anyway.
        holds(&state, &drafter, &local, &[1]);
        let part = ShardId {
            model_id: drafter.clone(),
            index: 1,
        };
        // Three copies, two of them nearby, so no region rule keeps it.
        let three = [local.clone(), peer.clone(), other.clone()];
        assert!(
            !manager.would_shed_copy(&part, &three, 0.9, &[]),
            "a copy too many by the count — kept: a split here guesses with it"
        );
        assert!(
            manager.would_shed_copy(&part, &three, 0.99, &[]),
            "an urgent disk takes it"
        );
        state.models.drafters_wanted.clear();
        assert!(
            manager.would_shed_copy(&part, &three, 0.9, &[]),
            "control: nothing wants it any more"
        );
    }

    /// Only a small model is ever fetched to draft: a 4 GiB one of the
    /// target's family — a quarter of a 70B, so the size rule alone would take
    /// it — is not even a candidate whose header is learned; a 1 GiB one is.
    #[test]
    fn only_a_small_model_is_ever_fetched_to_draft() {
        let (state, _manager) = make_test_manager_with_config(Config::default());
        let target = register_manifest_with_sized_shards(
            &state,
            "big-70b",
            80,
            &[(0, 40), (40, 80)],
            20 * 1024 * MIB,
        );
        register_manifest_with_sized_shards(
            &state,
            "mid-14b",
            48,
            &[(0, 24), (24, 48)],
            2 * 1024 * MIB,
        );
        let small =
            register_manifest_with_sized_shards(&state, "small-1.5b", 28, &[(0, 28)], 1024 * MIB);
        assert_eq!(
            drafter_to_acquire(&state, &target, |_| true),
            DrafterToAcquire {
                held: None,
                next: DrafterStep::LearnHeaders(vec![small]),
            }
        );
    }

    /// Every split that guesses here records its target — one that found its
    /// drafter too, so the download pass can fetch a larger one beside it
    /// (#331; before, a node that never missed guessed with what it had for
    /// good), and each later split keeps it recorded: prune keeps the drafter
    /// while splits keep using it, not a week after the first. Not where the
    /// owner named the drafter (the null control).
    #[test]
    fn every_split_that_guesses_keeps_its_target_wanted() {
        let mut named = Config::default();
        named.inference.draft_model = Some("small-0.5b".into());
        let (state, _manager) = make_test_manager_with_config(named);
        let target = ModelId("big-7b".into());
        note_split_wanting_a_drafter(&state, &target, true);
        note_split_wanting_a_drafter(&state, &target, false);
        assert!(
            state.models.drafters_wanted.is_empty(),
            "control: the owner named the drafter"
        );

        let (state, _manager) = make_test_manager_with_config(Config::default());
        note_split_wanting_a_drafter(&state, &target, true);
        let first = *state
            .models
            .drafters_wanted
            .get(&target)
            .expect("a drafter was found: recorded, so a larger one may come");
        std::thread::sleep(Duration::from_millis(5));
        note_split_wanting_a_drafter(&state, &target, false);
        let second = *state
            .models
            .drafters_wanted
            .get(&target)
            .expect("none here: wanted");
        assert!(second > first);
        std::thread::sleep(Duration::from_millis(5));
        note_split_wanting_a_drafter(&state, &target, true);
        assert!(
            state
                .models
                .drafters_wanted
                .get(&target)
                .is_some_and(|at| *at > second),
            "a split using its drafter keeps it wanted"
        );
    }

    /// A drafter being replaced is kept while its replacement arrives, and
    /// the replacement is fetched (#331): with the planted answer "the 0.5B is
    /// held, fetch the 1.5B", the download pass offers the parts of the 1.5B
    /// not held here and names the 0.5B it replaces, and prune keeps a part of
    /// each; with the answer "the 1.5B is held", it keeps the 1.5B's part and
    /// sheds the 0.5B's.
    #[test]
    fn a_drafter_being_replaced_is_kept_until_its_replacement_is_held() {
        let mut config = Config::default();
        config.auto_manage.min_replicas = 1;
        let (state, manager) = make_test_manager_with_config(config);
        state.models.auto_manage_default_model_cap.store(0, Relaxed);
        let target = register_manifest_with_sized_shards(
            &state,
            "big-7b",
            28,
            &[(0, 14), (14, 28)],
            2048 * MIB,
        );
        let held = register_manifest_with_sized_shards(
            &state,
            "small-0.5b",
            24,
            &[(0, 8), (8, 16), (16, 24)],
            500 * MIB,
        );
        let better = register_manifest_with_sized_shards(
            &state,
            "small-1.5b",
            28,
            &[(0, 10), (10, 20), (20, 28)],
            1600 * MIB,
        );
        let peer = peer_with_disk(&state, 2, None, 100_000);
        let other = peer_with_disk(&state, 3, None, 100_000);
        let local = state.identity.node_id().clone();
        holds(&state, &target, &peer, &[0, 1]);
        for m in [&held, &better] {
            holds(&state, m, &peer, &[0, 1, 2]);
            holds(&state, m, &other, &[0, 1, 2]);
            trusted(&state, m);
        }
        trusted(&state, &target);
        // A MIDDLE part of each here (prompt privacy keeps a whole model's
        // ends), three copies, two of them nearby: no other rule keeps it.
        holds(&state, &held, &local, &[1]);
        holds(&state, &better, &local, &[1]);
        let plant = |answer: DrafterToAcquire| {
            manager.drafter_book.answers.insert(
                target.clone(),
                (Instant::now() + Duration::from_secs(3600), answer),
            );
        };
        state
            .models
            .drafters_wanted
            .insert(target.clone(), Instant::now());
        let part = |m: &ModelId| ShardId {
            model_id: m.clone(),
            index: 1,
        };
        let three = [local.clone(), peer.clone(), other.clone()];

        plant(DrafterToAcquire {
            held: Some(held.clone()),
            next: DrafterStep::Fetch(better.clone()),
        });
        let manifest = state
            .model_registry
            .get_manifest(&better)
            .expect("registered");
        let (parts, replacing) = manager.parts_of_a_drafter_to_fetch(&manifest);
        assert_eq!(
            parts,
            [0, 2].into_iter().collect(),
            "the parts not held here"
        );
        assert_eq!(replacing.as_ref(), Some(&held));
        assert!(
            !manager.would_shed_copy(&part(&held), &three, 0.9, &[]),
            "the 0.5B is kept while the 1.5B comes"
        );
        assert!(
            !manager.would_shed_copy(&part(&better), &three, 0.9, &[]),
            "the 1.5B being fetched is kept"
        );

        plant(DrafterToAcquire {
            held: Some(better.clone()),
            next: DrafterStep::Nothing,
        });
        assert!(
            manager.parts_of_a_drafter_to_fetch(&manifest).0.is_empty(),
            "held: nothing more to fetch"
        );
        assert!(!manager.would_shed_copy(&part(&better), &three, 0.9, &[]));
        assert!(
            manager.would_shed_copy(&part(&held), &three, 0.9, &[]),
            "the 1.5B held and chosen: the 0.5B is no longer kept for drafting"
        );
    }

    /// A model a part of which the owner deleted is not a drafter to fetch:
    /// the download pass never fetches that part back, so naming it would
    /// leave the split with none.
    #[test]
    fn a_model_the_owner_deleted_a_part_of_is_not_fetched_to_draft() {
        let (state, manager) = make_test_manager_with_config(Config::default());
        let small =
            register_manifest_with_sized_shards(&state, "small-1.5b", 28, &[(0, 28)], 1024 * MIB);
        trusted(&state, &small);
        let manifest = state
            .model_registry
            .get_manifest(&small)
            .expect("registered");
        state.models.hf_sources.insert(
            small.clone(),
            crate::daemon::state::HfSource {
                repo_id: "example/small".into(),
                filename: "small.gguf".into(),
                mmproj_filename: None,
            },
        );
        assert!(manager.may_fetch_as_a_drafter(&manifest), "control");
        state.mark_shard_removed_by_user(&ShardId {
            model_id: small.clone(),
            index: 0,
        });
        assert!(!manager.may_fetch_as_a_drafter(&manifest));
    }
}
