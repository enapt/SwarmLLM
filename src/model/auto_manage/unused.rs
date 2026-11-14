//! A model nobody runs is given back (#389 (d), #452).
//!
//! Prune acted on disk pressure and the replica target, and trust decays after
//! seven idle days; neither ever gave a model back for not being used, so a
//! swarm held whatever its testers had once tried (34 models on 2026-11-01,
//! nearly all of 2024, the 0.5B in four copies). The rule here is TinyLFU's
//! admission (Einziger et al., ACM TOS 2017): a model is held only on evidence
//! that it is used. Three pools of evidence, each deciding one thing (#452):
//!
//! - **The swarm's use decides what the swarm KEEPS.** Each holder stamps the
//!   last time it served a model for another computer
//!   (`SharedState::note_request_served_for_a_peer`) and advertises it by the
//!   day ([`models_served`] → `NodeCapability::models_served`). A copy watched
//!   here for the window and run here by nobody in it is given back once no
//!   connected holder served the model in it either ([`SwarmUse`]) — whatever
//!   HuggingFace's list says. Read from that list instead, 0 of 4,222 prune
//!   passes on the Proxmox box gave a model back (2026-11-07 → 11-14): the list
//!   named nearly everything held.
//! - **This computer's use — the owner's included — keeps THIS computer's
//!   copy** (`ModelUse::last_used`), and is never advertised (#389 (a)).
//! - **What people run outside the swarm decides only what is TRIED**
//!   (`huggingface::popular`): a model held nowhere is fetched on it where the
//!   swarm's online cards could run it ([`the_cards_could_run`]), a model
//!   nobody here or outside runs is not — and one the swarm tried and nobody
//!   ran (a holder kept it the window unserved, or this node — or
//!   [`GAVE_BACK_WITNESSES`] others — gave it back for that within
//!   [`GAVE_BACK_FOR_SECS`]) is not fetched again on popularity alone, or the
//!   list would bring it back every window. A peer's stamp is only its word:
//!   "held it unserved" counts only while it holds a part, and none dated in
//!   the future counts (security review 2026-11-14).
//!
//! The owner's choice, a request on the swarm now (`asked_for`), a reference
//! model and a split's drafter keep a model whatever the pools say. The
//! download pass asks the same question (`would_shed_once_fetched`), so it
//! never fetches what prune sheds. Nothing here judges content (#389 (c)).
//!
//! Only a copy WATCHED for the whole window is judged: a model is watched from
//! when this node first held a part of it under a build that records use, so a
//! new copy has the window as its trial. Only a swarm whose every connected
//! holder advertises its use is judged: one older build holding the model, or
//! no computer advertising at all (an offline node), and nothing is given back.
//!
//! Prior art: Scarlett (Ananthanarayanan et al., EuroSys '11) budgets every
//! replica above its base by observed demand over a long window; the kubelet's
//! image GC and IPFS keep only what is in use or pinned — none keeps a copy of
//! content nobody asks for (FUTURE_WORK #452).

use super::manager::AutoShardManager;
use crate::types::{ModelId, ModelServed, NodeId};

/// How long a held model goes without a request before it may be given back —
/// and so the trial a new copy has. Twice trust's seven idle days: a model
/// someone runs at weekends survives a missed one.
pub(crate) const USE_WINDOW_SECS: i64 = 14 * 86_400;

/// How long a model given back because nobody ran it stays TRIED (#452): it is
/// not fetched again on popularity alone, here or by a computer that hears of
/// it, for two windows after its last part here went.
pub(crate) const GAVE_BACK_FOR_SECS: i64 = 2 * USE_WINDOW_SECS;

/// How many connected computers must say they gave a model back because nobody
/// ran it before this node, holding none, takes it as TRIED on their word
/// ([`swarm_use_of`]). A peer's give-back is only its word — it holds nothing
/// that would show it — so one computer could otherwise list every new release
/// as given back and keep the swarm from trying any (security review
/// 2026-11-14; a claim nothing corroborates needs a second witness, gotcha
/// #788). A model nobody runs is usually given back by every holder within days
/// of the others, so two is met where more than one held it; where one did,
/// another computer tries it once more.
pub(crate) const GAVE_BACK_WITNESSES: usize = 2;

/// Where the records are saved.
pub(crate) const USE_TREE: &str = "model_use";

/// A record moves in memory and on disk only when a use moves this far:
/// against a 14-day window, an hour late is nothing, and a request does not
/// cost a database write.
const MOVES_BY_SECS: i64 = 3_600;

/// What this node has seen of one model's use (#389 (d), #452).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ModelUse {
    /// When this node first held a part of the model under a build that
    /// records use (Unix seconds).
    pub watched_since: i64,
    /// The last request for the model that ran here or that this node routed
    /// (Unix seconds; 0 = none seen) — the owner's and peers' alike.
    pub last_used: i64,
    /// The last request for it this node served FOR ANOTHER COMPUTER (Unix
    /// seconds; 0 = none seen): the part of `last_used` the swarm may learn.
    #[serde(default)]
    pub served_for_others: i64,
    /// When this node gave back its last part because nobody ran it (Unix
    /// seconds; 0 = it has not). Such a record outlives the model here by
    /// [`GAVE_BACK_FOR_SECS`].
    #[serde(default)]
    pub gave_back_unused: i64,
    /// When this node began recording the requests for it served for other
    /// computers (Unix seconds): `watched_since` for a record this build began;
    /// for one a build before #452's wrote (read as 0), the first pass after
    /// the update — what was served before it is not known, and the swarm is
    /// told so ([`models_served`], `ModelServed::served_since`).
    #[serde(default)]
    pub served_recorded_since: i64,
}

impl ModelUse {
    fn gave_back(&self) -> bool {
        self.gave_back_unused > 0
    }

    /// `ModelServed::served_since` for this record at `now`: the first whole
    /// day its record of serving for others covers, where that began after the
    /// record did; 0 where it covers the whole record. A record not yet passed
    /// over since the update counts from `now`.
    fn served_since(&self, now: i64) -> i64 {
        let recorded = if self.served_recorded_since > 0 {
            self.served_recorded_since
        } else {
            now
        };
        if recorded > self.watched_since {
            ModelServed::day_of(recorded).saturating_add(86_400)
        } else {
            0
        }
    }

    /// A give-back still remembered at `now`.
    fn gave_back_recently(&self, now: i64) -> bool {
        self.gave_back() && now.saturating_sub(self.gave_back_unused) < GAVE_BACK_FOR_SECS
    }
}

/// Has a copy with this record gone the whole window without a request here?
/// A use from before the record began does not end its trial.
pub(crate) fn quiet_for_the_window(record: &ModelUse, now: i64) -> bool {
    now.saturating_sub(record.watched_since.max(record.last_used)) >= USE_WINDOW_SECS
}

/// What one pass saw of a model ([`next_record`]).
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Seen {
    /// This node holds a part of it now.
    pub held: bool,
    /// The last request that ran here or that this node routed.
    pub used_at: Option<i64>,
    /// The last request served here for another computer.
    pub served_at: Option<i64>,
}

/// What a pass does with a record.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Next {
    Unchanged,
    Write(ModelUse),
    Forget,
}

/// The record after a pass: a new one — a new trial — for a model held now and
/// not before, or held again after it was given back; a use moved where a later
/// one was seen by `MOVES_BY_SECS` or more; a model held no more forgotten,
/// unless it was given back for nobody running it, which is remembered for
/// [`GAVE_BACK_FOR_SECS`].
pub(crate) fn next_record(before: Option<ModelUse>, seen: Seen, now: i64) -> Next {
    let at = |t: Option<i64>| t.filter(|&t| t > 0).map(|t| t.min(now));
    let served_at = at(seen.served_at);
    // A request served here for another computer ran here too.
    let used_at = at(seen.used_at).max(served_at);
    if !seen.held {
        return match before {
            Some(r) if r.gave_back_recently(now) => Next::Unchanged,
            Some(_) => Next::Forget,
            None => Next::Unchanged,
        };
    }
    match before {
        Some(r) if !r.gave_back() => {
            let moved =
                |t: Option<i64>, was: i64| t.filter(|&t| t >= was.saturating_add(MOVES_BY_SECS));
            let (last_used, served) = (
                moved(used_at, r.last_used),
                moved(served_at, r.served_for_others),
            );
            // A record from before serving for others was recorded: it is
            // recorded from this pass on.
            let recorded_from_now = r.served_recorded_since == 0;
            if last_used.is_none() && served.is_none() && !recorded_from_now {
                return Next::Unchanged;
            }
            Next::Write(ModelUse {
                last_used: last_used.unwrap_or(r.last_used),
                served_for_others: served.unwrap_or(r.served_for_others),
                served_recorded_since: if recorded_from_now {
                    now
                } else {
                    r.served_recorded_since
                },
                ..r
            })
        }
        before => Next::Write(ModelUse {
            watched_since: now,
            last_used: used_at.unwrap_or(0).max(before.map_or(0, |r| r.last_used)),
            served_for_others: served_at
                .unwrap_or(0)
                .max(before.map_or(0, |r| r.served_for_others)),
            gave_back_unused: 0,
            served_recorded_since: now,
        }),
    }
}

/// What the swarm's other computers say of a model's use (#452), from the
/// capabilities of the connected ones ([`swarm_use_of`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SwarmUse {
    /// A computer served it for another within the window.
    Served,
    /// Nobody can say: no connected computer advertises its use (this node is
    /// offline, or every peer is an older build), or one holding the model
    /// does not (an older build, or one not heard from yet).
    Unknown,
    /// No connected holder served it within the window. `tried`: one that
    /// holds a part held it that long unserved, or [`GAVE_BACK_WITNESSES`]
    /// gave it back for that within [`GAVE_BACK_FOR_SECS`].
    Quiet { tried: bool },
}

/// One connected computer's word on one model ([`swarm_use_of`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Word {
    /// It advertises no use — an older build, a capability not heard yet, or
    /// a list cut at `ModelServed::MOST_ENTRIES` before this model; `holds`:
    /// it holds a part of the model all the same.
    Silent { holds: bool },
    /// It advertises its use: its entry for the model, if it has one; `holds`:
    /// it holds a part of the model (its capability's parts, or the
    /// registry's). Only then is its word that it HELD the model unserved
    /// weighed — a stamp is only its word, a part is what it announced.
    Says {
        entry: Option<ModelServed>,
        holds: bool,
    },
}

/// The swarm's verdict on a model from each connected computer's word. A
/// stamp later than tomorrow (a clock a little ahead may write tomorrow) is not
/// weighed: no computer served or gave back anything in the future.
pub(crate) fn swarm_use_of(words: impl IntoIterator<Item = Word>, now: i64) -> SwarmUse {
    let window_from = ModelServed::day_of(now.saturating_sub(USE_WINDOW_SECS));
    let gave_back_from = ModelServed::day_of(now.saturating_sub(GAVE_BACK_FOR_SECS));
    let latest = ModelServed::day_of(now).saturating_add(86_400);
    let within = |stamp: i64, from: i64| stamp > 0 && stamp >= from && stamp <= latest;
    let (mut anyone_says, mut a_holder_silent, mut held_unserved) = (false, false, false);
    let mut gave_back_by = 0usize;
    for word in words {
        match word {
            Word::Silent { holds } => a_holder_silent |= holds,
            Word::Says { entry, holds } => {
                anyone_says = true;
                let Some(m) = entry else { continue };
                if within(m.served, window_from) {
                    return SwarmUse::Served;
                }
                // Its record of serving for others began after it began
                // holding, too late to cover the window (it updated from a
                // build that did not keep one): what it served before is not
                // known, so it cannot say nobody ran it — missing data is never
                // a reason to give back (Kubernetes HPA: a pod missing metrics
                // counts as fully used when scaling down).
                let covers_window =
                    m.served_since == 0 || m.served_since <= m.held_since.max(window_from);
                if holds && !covers_window {
                    a_holder_silent = true;
                    continue;
                }
                held_unserved |= holds && m.held_since > 0 && m.held_since <= window_from;
                gave_back_by += usize::from(within(m.gave_back_unused, gave_back_from));
            }
        }
    }
    if a_holder_silent || !anyone_says {
        SwarmUse::Unknown
    } else {
        SwarmUse::Quiet {
            tried: held_unserved || gave_back_by >= GAVE_BACK_WITNESSES,
        }
    }
}

/// Could the swarm's online cards run a model of `total_size_bytes` between
/// them (#452)? Its estimated graphics memory against every card's
/// (`AutoShardManager::swarm_card_mb`); `0` — not yet read, or no card heard
/// of — says nothing, and any model may be tried. A model past it runs only
/// with processors in its chain, at a few tokens a second: the 122B and 397B
/// parts that took the 70B's room on the Proxmox box (2026-11-08) were tried
/// on what people run outside the swarm, on cards that could not hold them.
pub(crate) fn the_cards_could_run(total_size_bytes: u64, swarm_card_mb: u64) -> bool {
    swarm_card_mb == 0 || super::vram::estimate_model_vram_mb(total_size_bytes) <= swarm_card_mb
}

/// The pools' verdict on a model (#452), past the keeps every model has (the
/// owner's choice, a request now, a reference model, a split's drafter):
/// `record` is this node's, `popular` whether it is worth trying on what people
/// run outside the swarm ([`AutoShardManager::worth_trying`]; `None`: no fresh
/// reading), `swarm` what the connected computers say — each asked only where
/// the rest leaves it the deciding word.
pub(crate) fn unused_everywhere(
    record: Option<ModelUse>,
    popular: impl FnOnce() -> Option<bool>,
    swarm: impl FnOnce() -> SwarmUse,
    now: i64,
) -> bool {
    match record {
        // A copy here: on trial for the window, kept by any use of it here —
        // the owner's included, which never leaves this computer — and then
        // by the swarm's.
        Some(r) if !r.gave_back() => {
            quiet_for_the_window(&r, now) && matches!(swarm(), SwarmUse::Quiet { .. })
        }
        // Given back here because nobody ran it: not fetched again unless a
        // computer has served it since.
        Some(r) if r.gave_back_recently(now) => swarm() != SwarmUse::Served,
        // Held none: tried on what people run outside the swarm — never where
        // the swarm tried it and nobody ran it.
        _ => match swarm() {
            SwarmUse::Served => false,
            SwarmUse::Quiet { tried: true } => true,
            SwarmUse::Quiet { tried: false } | SwarmUse::Unknown => popular() == Some(false),
        },
    }
}

/// What this node advertises of its models' use (#452,
/// `NodeCapability::models_served`), from its records: each model it holds —
/// the day it began holding it and the day it last served it for another
/// computer — and each it gave back because nobody ran it, while that is
/// remembered. By the day; sorted by id, so a reader looks one up by halves; at
/// most `ModelServed::MOST_ENTRIES`, the models held first. Never the owner's
/// own use (#389 (a)): `served_for_others` is written only for a peer's request.
/// Never an id longer than a receiver reads (`MAX_MODEL_ID_BYTES`): one such
/// entry makes every receiver drop this computer's whole capability
/// (`dispatch::capability_fits`) — its parts, memory and speed with it — and a
/// model imported from a file keeps its name's length (pre-release review
/// 2026-11-14).
pub(crate) fn models_served(state: &crate::daemon::SharedState) -> Vec<ModelServed> {
    let now = chrono::Utc::now().timestamp();
    let (mut held, mut gave_back): (Vec<ModelServed>, Vec<ModelServed>) = (Vec::new(), Vec::new());
    for entry in state.models.model_use.iter() {
        let r = *entry.value();
        if r.gave_back() && !r.gave_back_recently(now) {
            continue;
        }
        if entry.key().0.len() > crate::daemon::state::manifest_sync::MAX_MODEL_ID_BYTES {
            continue;
        }
        let served = ModelServed {
            model_id: entry.key().0.clone(),
            held_since: if r.gave_back() {
                0
            } else {
                ModelServed::day_of(r.watched_since)
            },
            served: ModelServed::day_of(r.served_for_others),
            gave_back_unused: ModelServed::day_of(r.gave_back_unused),
            served_since: r.served_since(now),
        };
        if r.gave_back() {
            gave_back.push(served);
        } else {
            held.push(served);
        }
    }
    held.sort_by(|a, b| a.model_id.cmp(&b.model_id));
    gave_back.sort_by_key(|m| std::cmp::Reverse(m.gave_back_unused));
    let mut listed: Vec<ModelServed> = held
        .into_iter()
        .chain(gave_back)
        .take(ModelServed::MOST_ENTRIES)
        .collect();
    listed.sort_by(|a, b| a.model_id.cmp(&b.model_id));
    listed
}

/// What the dashboard says of a model no computer in reach holds a part of, that
/// computers gave back because nobody ran it (#452): such a model stays listed —
/// folded away — while the give-back is remembered ([`GAVE_BACK_FOR_SECS`]),
/// rather than vanish while the code that runs it is still here.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct HeldBefore {
    /// The latest request for it any of them saw (Unix seconds; the peers' by
    /// the day): served for another computer, or — this node's own record —
    /// run here. `None`: nobody ran it while it was held.
    pub last_ran: Option<i64>,
    /// The latest give-back still remembered (Unix seconds; a peer's by the day).
    pub given_back: i64,
}

/// [`HeldBefore`] for `model_id`, from this node's record and every connected
/// computer's advertised one (`NodeCapability::models_served`); `None` where
/// none remembers giving it back. The caller asks only of a model nobody in
/// reach holds a part of. This node's own use is shown only to its owner.
pub(crate) fn held_before(
    state: &crate::daemon::SharedState,
    model_id: &ModelId,
    now: i64,
) -> Option<HeldBefore> {
    let gave_back_from = ModelServed::day_of(now.saturating_sub(GAVE_BACK_FOR_SECS));
    let (mut last_ran, mut given_back) = (0i64, 0i64);
    if let Some(r) = state.models.model_use.get(model_id).map(|r| *r) {
        if r.gave_back_recently(now) {
            given_back = r.gave_back_unused;
            last_ran = r.last_used.max(r.served_for_others);
        }
    }
    // Collected first: never one map's guard held while taking another's.
    let connected: Vec<NodeId> = state.connected_node_ids.iter().map(|n| n.clone()).collect();
    for node in connected {
        let entry = state.peer_registry.get(&node).and_then(|p| {
            let served = p.capability.as_ref()?.models_served.as_ref()?;
            served
                .binary_search_by(|m| m.model_id.as_str().cmp(model_id.0.as_str()))
                .ok()
                .map(|i| served[i].clone())
        });
        let Some(m) = entry else { continue };
        if m.gave_back_unused > 0 && m.gave_back_unused >= gave_back_from {
            given_back = given_back.max(m.gave_back_unused);
        }
        last_ran = last_ran.max(m.served);
    }
    (given_back > 0).then(|| HeldBefore {
        last_ran: (last_ran > 0).then_some(last_ran.min(now)),
        given_back: given_back.min(now),
    })
}

impl AutoShardManager {
    /// Is `model_id` worth TRYING for what people run outside the swarm (#452)?
    /// The day's reading names it ([`crate::daemon::SharedState::popular_outside_the_swarm`],
    /// `None` without a fresh one) AND the swarm's cards could run it
    /// ([`the_cards_could_run`]; `total_size_bytes`, asked only for a model the
    /// reading names, `None` where its manifest is unknown). The one answer the
    /// download pass's room (`scoring::is_scarce`'s `run_outside`) and
    /// [`Self::nobody_runs`] read — never the reading alone.
    pub(super) fn worth_trying(
        &self,
        model_id: &ModelId,
        total_size_bytes: impl FnOnce() -> Option<u64>,
    ) -> Option<bool> {
        self.shared_state
            .popular_outside_the_swarm(model_id)
            .map(|named| {
                named
                    && total_size_bytes().is_none_or(|bytes| {
                        the_cards_could_run(
                            bytes,
                            self.swarm_card_mb
                                .load(std::sync::atomic::Ordering::Relaxed),
                        )
                    })
            })
    }

    /// Fold what this node saw of each model's use into its records, saved
    /// (`USE_TREE`): a request a worker ran (`ModelProcessPool::model_used_secs_ago`
    /// — every execution path, the owner's and peers' alike), one the router
    /// took (`ModelTrustInfo::last_request_at`) and one served for another
    /// computer (`served_for_peers_at`, #452). A record begins when this node
    /// first holds a part of the model and ends when it holds none — unless it
    /// gave the model back because nobody ran it ([`next_record`]). Called at
    /// the start of every prune pass, so no copy is judged without one.
    pub(super) fn watch_model_use(&self) {
        let state = &self.shared_state;
        let now = chrono::Utc::now().timestamp();
        let me = state.identity.node_id().clone();
        let mut used_at: std::collections::HashMap<ModelId, i64> = state
            .model_process_pool
            .model_used_secs_ago()
            .into_iter()
            .map(|(m, ago)| {
                (
                    m,
                    now.saturating_sub(i64::try_from(ago).unwrap_or(i64::MAX)),
                )
            })
            .collect();
        for entry in state.models.model_trust.iter() {
            if let Some(at) = entry.value().last_request_at {
                let slot = used_at.entry(entry.key().clone()).or_insert(0);
                *slot = (*slot).max(at.timestamp());
            }
        }
        let held: std::collections::HashSet<ModelId> = state
            .model_registry
            .models()
            .into_iter()
            .filter(|manifest| {
                !state
                    .model_registry
                    .local_shard_indices_in(manifest, &me)
                    .is_empty()
            })
            .map(|manifest| manifest.id)
            .collect();
        let mut models: std::collections::HashSet<ModelId> = held.clone();
        models.extend(state.models.model_use.iter().map(|e| e.key().clone()));
        for model in models {
            let before = state.models.model_use.get(&model).map(|r| *r);
            let seen = Seen {
                held: held.contains(&model),
                used_at: used_at.get(&model).copied(),
                served_at: state.models.served_for_peers_at.get(&model).map(|t| *t),
            };
            match next_record(before, seen, now) {
                Next::Unchanged => {}
                Next::Write(record) => {
                    state.models.model_use.insert(model.clone(), record);
                    if let Err(e) = state.db.put_json(USE_TREE, &model.0, &record) {
                        tracing::warn!(model = %model, error = %e, "A model's use was not saved — watched again from a restart");
                    }
                }
                Next::Forget => {
                    state.models.model_use.remove(&model);
                    let _ = state.db.remove(USE_TREE, &model.0);
                }
            }
        }
    }

    /// Every part of `model_id` this node held went back because nobody ran it
    /// (#452): remembered for [`GAVE_BACK_FOR_SECS`], so this node's download
    /// pass does not fetch it back on popularity alone ([`unused_everywhere`])
    /// and the swarm hears so ([`models_served`]). Called by prune once the
    /// model's last part here is gone.
    pub(super) fn note_gave_back_unused(&self, model_id: &ModelId) {
        let state = &self.shared_state;
        let record = ModelUse {
            gave_back_unused: chrono::Utc::now().timestamp(),
            ..state
                .models
                .model_use
                .get(model_id)
                .map(|r| *r)
                .unwrap_or_default()
        };
        state.models.model_use.insert(model_id.clone(), record);
        if let Err(e) = state.db.put_json(USE_TREE, &model_id.0, &record) {
            tracing::warn!(model = %model_id, error = %e, "A model given back for want of use was not saved — it may be fetched again after a restart");
        }
    }

    /// What the connected computers say of `model_id`'s use (#452): each one's
    /// advertised entry (`NodeCapability::models_served`); one that advertises
    /// none — or whose list is full and stops short of the model — is weighed
    /// only if it holds a part: by its capability's parts, or the registry's
    /// where no capability has been heard. A computer saying it held the model
    /// unserved is weighed only while it holds a part ([`Word::Says`]).
    fn swarm_use(&self, model_id: &ModelId, now: i64) -> SwarmUse {
        /// What one connected computer's capability says, read under its
        /// registry guard: its entry, or that its list is full without one, or
        /// that it advertises none — each with whether its parts name the model.
        enum Heard {
            Lists(Option<ModelServed>, bool),
            ListsNoMore(bool),
            Silent(bool),
            Nothing,
        }
        let state = &self.shared_state;
        let me = state.identity.node_id().clone();
        // Collected first: no guard of the connected set is held while the
        // registry is read.
        let connected: Vec<NodeId> = state
            .connected_node_ids
            .iter()
            .map(|n| n.key().clone())
            .filter(|n| *n != me)
            .collect();
        let mut registry_holders: Option<std::collections::HashSet<NodeId>> = None;
        let mut words = Vec::with_capacity(connected.len());
        for node in connected {
            let heard = state
                .peer_registry
                .get(&node)
                .and_then(|peer| {
                    let cap = peer.capability.as_ref()?;
                    let in_its_parts = cap.hosted_shards.iter().any(|s| s.model_id == *model_id);
                    Some(match &cap.models_served {
                        Some(list) => match list
                            .binary_search_by(|m| m.model_id.as_str().cmp(model_id.0.as_str()))
                        {
                            Ok(i) => Heard::Lists(Some(list[i].clone()), in_its_parts),
                            Err(_) if list.len() >= ModelServed::MOST_ENTRIES => {
                                Heard::ListsNoMore(in_its_parts)
                            }
                            Err(_) => Heard::Lists(None, in_its_parts),
                        },
                        None => Heard::Silent(in_its_parts),
                    })
                })
                .unwrap_or(Heard::Nothing);
            // The registry is read with no peer guard held.
            let mut registry_names = || {
                registry_holders
                    .get_or_insert_with(|| self.registry_holders_of(model_id))
                    .contains(&node)
            };
            words.push(match heard {
                Heard::Lists(entry, in_its_parts) => Word::Says {
                    entry,
                    holds: in_its_parts || registry_names(),
                },
                // A list cut at its cap (`models_served` lists held models
                // first, by id) cannot say this one went unserved (security
                // review 2026-11-14).
                Heard::ListsNoMore(in_its_parts) => Word::Silent {
                    holds: in_its_parts || registry_names(),
                },
                Heard::Silent(in_its_parts) => Word::Silent {
                    holds: in_its_parts,
                },
                Heard::Nothing => Word::Silent {
                    holds: registry_names(),
                },
            });
        }
        swarm_use_of(words, now)
    }

    /// Every computer the registry names as holding a part of `model_id`.
    fn registry_holders_of(&self, model_id: &ModelId) -> std::collections::HashSet<NodeId> {
        let registry = &self.shared_state.model_registry;
        registry
            .get_manifest(model_id)
            .map(|manifest| {
                manifest
                    .shards
                    .iter()
                    .flat_map(|s| {
                        registry.shard_holders(&crate::types::ShardId {
                            model_id: model_id.clone(),
                            index: s.index,
                        })
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    /// May this node give back every part of `model_id` — or leave it unfetched
    /// (#389 (d), #452)? Never a model chosen by this computer's owner
    /// ([`crate::daemon::SharedState::owner_chose`]), a reference model testers
    /// compare results on (`model::reference`), one asked for on the swarm now
    /// (`asked_for`), or the drafter of a split this computer runs
    /// (`drafts_for_a_split` — its use counts toward the model it drafts for).
    /// Past those, the three pools decide ([`unused_everywhere`]).
    pub(super) fn nobody_runs(&self, model_id: &ModelId) -> bool {
        let state = &self.shared_state;
        if state.owner_chose(model_id)
            || crate::model::reference::is_reference_model(&model_id.0)
            || self.asked_for(model_id)
            || self.drafts_for_a_split(model_id)
        {
            return false;
        }
        let now = chrono::Utc::now().timestamp();
        unused_everywhere(
            state.models.model_use.get(model_id).map(|r| *r),
            || {
                self.worth_trying(model_id, || {
                    state
                        .model_registry
                        .get_manifest(model_id)
                        .map(|m| m.total_size_bytes)
                })
            },
            || self.swarm_use(model_id, now),
            now,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DAY: i64 = 86_400;

    /// A record this build began: serving for others recorded from its start.
    fn watched(since: i64, last: i64) -> ModelUse {
        ModelUse {
            watched_since: since,
            last_used: last,
            served_recorded_since: since,
            ..Default::default()
        }
    }

    fn entry(held_since: i64, served: i64, gave_back_unused: i64) -> ModelServed {
        ModelServed {
            model_id: "m".into(),
            held_since,
            served,
            gave_back_unused,
            served_since: 0,
        }
    }

    #[test]
    fn a_copy_is_quiet_only_once_watched_for_the_whole_window() {
        let now = 100 * DAY;
        assert!(
            !quiet_for_the_window(&watched(now - 13 * DAY, 0), now),
            "watched 13 days, never used: still on trial"
        );
        assert!(quiet_for_the_window(&watched(now - 14 * DAY, 0), now));
        assert!(
            !quiet_for_the_window(&watched(now - 60 * DAY, now - 2 * DAY), now),
            "used two days ago"
        );
        assert!(quiet_for_the_window(
            &watched(now - 60 * DAY, now - 15 * DAY),
            now
        ));
        assert!(
            !quiet_for_the_window(&watched(now - 2 * DAY, now - 40 * DAY), now),
            "a use from before the record began does not end the trial"
        );
    }

    #[test]
    fn a_record_begins_when_held_and_moves_only_by_the_hour() {
        let now = 50 * DAY;
        let held = |used_at, served_at| Seen {
            held: true,
            used_at,
            served_at,
        };
        assert_eq!(
            next_record(None, held(None, None), now),
            Next::Write(watched(now, 0))
        );
        assert_eq!(
            next_record(None, held(Some(now - 10), None), now),
            Next::Write(watched(now, now - 10))
        );
        let r = watched(now - DAY, now - 7_200);
        assert_eq!(
            next_record(Some(r), held(None, None), now),
            Next::Unchanged,
            "nothing seen"
        );
        assert_eq!(
            next_record(Some(r), held(Some(now - 7_100), None), now),
            Next::Unchanged,
            "under an hour later: not worth a write"
        );
        assert_eq!(
            next_record(Some(r), held(Some(now), None), now),
            Next::Write(ModelUse {
                last_used: now,
                ..r
            })
        );
        assert_eq!(
            next_record(Some(r), held(Some(now + 9 * DAY), None), now),
            Next::Write(ModelUse {
                last_used: now,
                ..r
            }),
            "a clock ahead of now is held to now"
        );
        assert_eq!(
            next_record(Some(r), held(Some(0), None), now),
            Next::Unchanged,
            "0 is no use"
        );
        assert_eq!(
            next_record(Some(r), held(None, Some(now)), now),
            Next::Write(ModelUse {
                last_used: now,
                served_for_others: now,
                ..r
            }),
            "a request served for another computer ran here too"
        );
    }

    /// A model held no more is forgotten — unless it went back because nobody
    /// ran it, which is remembered for two windows, and held again starts a
    /// new trial.
    #[test]
    fn a_give_back_for_want_of_use_is_remembered_and_a_new_copy_starts_a_new_trial() {
        let now = 200 * DAY;
        let gone = Seen::default();
        let r = watched(now - 30 * DAY, now - 20 * DAY);
        assert_eq!(next_record(Some(r), gone, now), Next::Forget);
        assert_eq!(next_record(None, gone, now), Next::Unchanged);
        let given_back = ModelUse {
            gave_back_unused: now - 27 * DAY,
            served_for_others: now - 40 * DAY,
            ..r
        };
        assert_eq!(
            next_record(Some(given_back), gone, now),
            Next::Unchanged,
            "within two windows: remembered"
        );
        let long_ago = ModelUse {
            gave_back_unused: now - 28 * DAY,
            ..given_back
        };
        assert_eq!(next_record(Some(long_ago), gone, now), Next::Forget);
        let again = Seen {
            held: true,
            ..Seen::default()
        };
        assert_eq!(
            next_record(Some(given_back), again, now),
            Next::Write(ModelUse {
                watched_since: now,
                last_used: r.last_used,
                served_for_others: now - 40 * DAY,
                gave_back_unused: 0,
                served_recorded_since: now,
            }),
            "held again: a new trial, its history kept"
        );
    }

    /// A record from the build before #452's kept no serving for others, so the
    /// first pass after the update records it from then on and the swarm is
    /// told the first whole day it covers. A record this build began covers
    /// itself (`served_since` 0).
    #[test]
    fn a_record_from_before_the_update_records_serving_for_others_from_the_update() {
        let now = 200 * DAY + 3_600;
        let from_the_old_build = ModelUse {
            watched_since: now - 30 * DAY,
            last_used: now - 20 * DAY,
            ..Default::default()
        };
        assert_eq!(
            from_the_old_build.served_since(now),
            200 * DAY + DAY,
            "not passed over yet: from now, the next whole day"
        );
        let held = Seen {
            held: true,
            ..Seen::default()
        };
        let Next::Write(updated) = next_record(Some(from_the_old_build), held, now) else {
            panic!("the first pass after the update writes the record");
        };
        assert_eq!(
            updated,
            ModelUse {
                served_recorded_since: now,
                ..from_the_old_build
            }
        );
        assert_eq!(updated.served_since(now + 10 * DAY), 201 * DAY);
        assert_eq!(
            next_record(Some(updated), held, now + 60),
            Next::Unchanged,
            "recorded once"
        );
        let begun_here = watched(now - 3 * DAY, 0);
        assert_eq!(begun_here.served_since(now), 0, "covers itself");
        let Next::Write(new) = next_record(None, held, now) else {
            panic!("a model held now starts a record");
        };
        assert_eq!(new.served_since(now + DAY), 0);
    }

    /// Right after an update every computer's record of serving for others is
    /// empty, so "served nothing" would be every model's verdict: a holder
    /// whose record does not cover the window cannot say (Kubernetes HPA counts
    /// a pod missing metrics as fully used when scaling down). One that began
    /// holding after it began recording covers what it held.
    #[test]
    fn a_holder_whose_record_is_younger_than_the_window_cannot_say() {
        let now = 100 * DAY;
        let holder = |held_since: i64, served_since: i64| Word::Says {
            entry: Some(ModelServed {
                served_since,
                ..entry(held_since, 0, 0)
            }),
            holds: true,
        };
        assert_eq!(
            swarm_use_of([holder(now - 40 * DAY, now - 3 * DAY)], now),
            SwarmUse::Unknown,
            "held 40 days, serving recorded for 3: it cannot say"
        );
        assert_eq!(
            swarm_use_of([holder(now - 40 * DAY, now - 14 * DAY)], now),
            SwarmUse::Quiet { tried: true },
            "recorded the whole window"
        );
        assert_eq!(
            swarm_use_of([holder(now - 2 * DAY, now - 3 * DAY)], now),
            SwarmUse::Quiet { tried: false },
            "began holding after it began recording: it covers what it held"
        );
        assert_eq!(
            swarm_use_of([holder(now - 40 * DAY, 0)], now),
            SwarmUse::Quiet { tried: true },
            "0: recorded since it began holding"
        );
        assert_eq!(
            swarm_use_of(
                [
                    Word::Says {
                        entry: Some(ModelServed {
                            served_since: now,
                            ..entry(0, 0, now - DAY)
                        }),
                        holds: false,
                    },
                    Word::Says {
                        entry: None,
                        holds: true
                    },
                ],
                now
            ),
            SwarmUse::Quiet { tried: false },
            "a computer holding none of it is no holder that cannot say"
        );
    }

    #[test]
    fn the_swarm_is_judged_only_when_every_holder_can_say() {
        let now = 100 * DAY;
        let says = |entry: Option<ModelServed>| Word::Says { entry, holds: true };
        assert_eq!(swarm_use_of([], now), SwarmUse::Unknown, "offline");
        assert_eq!(
            swarm_use_of([Word::Silent { holds: false }], now),
            SwarmUse::Unknown,
            "nobody advertises"
        );
        assert_eq!(
            swarm_use_of([says(None)], now),
            SwarmUse::Quiet { tried: false }
        );
        assert_eq!(
            swarm_use_of([says(None), Word::Silent { holds: false }], now),
            SwarmUse::Quiet { tried: false },
            "an older build that holds none says nothing of it"
        );
        assert_eq!(
            swarm_use_of([says(None), Word::Silent { holds: true }], now),
            SwarmUse::Unknown,
            "an older build holding a part may be running it"
        );
        let served_yesterday = says(Some(entry(now - 40 * DAY, now - DAY, 0)));
        assert_eq!(
            swarm_use_of([Word::Silent { holds: true }, served_yesterday], now),
            SwarmUse::Served,
            "served: whatever the rest say"
        );
        assert_eq!(
            swarm_use_of([says(Some(entry(now - 40 * DAY, now - 15 * DAY, 0)))], now),
            SwarmUse::Quiet { tried: true },
            "held for the window, last served before it"
        );
        assert_eq!(
            swarm_use_of([says(Some(entry(now - 3 * DAY, 0, 0)))], now),
            SwarmUse::Quiet { tried: false },
            "a holder still on its trial"
        );
        let gave_back = |days: i64| says(Some(entry(0, 0, now - days * DAY)));
        assert_eq!(
            swarm_use_of([gave_back(20), gave_back(3)], now),
            SwarmUse::Quiet { tried: true },
            "two gave it back for want of use within two windows"
        );
        assert_eq!(
            swarm_use_of([gave_back(30), gave_back(3)], now),
            SwarmUse::Quiet { tried: false },
            "one of them past two windows"
        );
    }

    /// A stamp is only a computer's word (security review 2026-11-14): "held
    /// it unserved" counts only from one holding a part, a give-back only with
    /// a second witness, and no stamp from the future — or one connected
    /// computer could list every new release as tried, and none would be.
    #[test]
    fn one_computers_word_alone_never_makes_a_model_tried() {
        let now = 100 * DAY;
        let held_unserved = Some(entry(now - 40 * DAY, 0, 0));
        assert_eq!(
            swarm_use_of(
                [Word::Says {
                    entry: held_unserved.clone(),
                    holds: false
                }],
                now
            ),
            SwarmUse::Quiet { tried: false },
            "it holds no part of it: its stamp is not weighed"
        );
        assert_eq!(
            swarm_use_of(
                [Word::Says {
                    entry: held_unserved,
                    holds: true
                }],
                now
            ),
            SwarmUse::Quiet { tried: true },
            "control: a holder"
        );
        let says = |entry: ModelServed| Word::Says {
            entry: Some(entry),
            holds: false,
        };
        assert_eq!(
            swarm_use_of([says(entry(0, 0, now - DAY))], now),
            SwarmUse::Quiet { tried: false },
            "one give-back alone"
        );
        assert_eq!(
            swarm_use_of(
                [says(entry(0, 0, now - DAY)), says(entry(0, 0, i64::MAX))],
                now
            ),
            SwarmUse::Quiet { tried: false },
            "a give-back dated in the future is no second witness"
        );
        assert_eq!(
            swarm_use_of(
                [says(entry(0, 0, now - DAY)), says(entry(0, 0, now + DAY))],
                now
            ),
            SwarmUse::Quiet { tried: true },
            "a clock a day ahead still counts"
        );
        assert_eq!(
            swarm_use_of([says(entry(1, i64::MAX, 0))], now),
            SwarmUse::Quiet { tried: false },
            "served in the future: not served"
        );
    }

    /// The user (2026-11-14): "nobody_runs should be true for the swarm rather
    /// than huggingface, should have separate statistic pools". A copy held
    /// here is judged by its use here and the swarm's, never by what people run
    /// outside the swarm; that reading only decides what is fetched to try.
    #[test]
    fn a_held_copy_is_judged_by_the_swarm_and_popularity_only_decides_a_trial() {
        let now = 100 * DAY;
        let quiet = || SwarmUse::Quiet { tried: false };
        let on_trial = watched(now - 5 * DAY, 0);
        let done = watched(now - 20 * DAY, now - 16 * DAY);
        for reading in [Some(true), Some(false), None] {
            let popular = || reading;
            assert!(
                unused_everywhere(Some(done), popular, quiet, now),
                "popular {reading:?}: held the window, nobody ran it"
            );
            assert!(!unused_everywhere(Some(on_trial), popular, quiet, now));
            assert!(!unused_everywhere(
                Some(done),
                popular,
                || SwarmUse::Served,
                now
            ));
            assert!(
                !unused_everywhere(Some(done), popular, || SwarmUse::Unknown, now),
                "an older build or no peer: nothing is given back"
            );
            assert!(
                unused_everywhere(None, popular, || SwarmUse::Quiet { tried: true }, now),
                "the swarm tried it and nobody ran it: not fetched"
            );
        }
        assert!(
            !unused_everywhere(
                Some(watched(now - 20 * DAY, now - DAY)),
                || Some(false),
                quiet,
                now
            ),
            "run here yesterday — the owner's use keeps this computer's copy"
        );
        assert!(
            !unused_everywhere(None, || Some(true), quiet, now),
            "held nowhere, run outside: tried"
        );
        assert!(
            !unused_everywhere(None, || None, quiet, now),
            "no reading: fetched as before"
        );
        assert!(
            unused_everywhere(None, || Some(false), quiet, now),
            "nobody here or outside runs it"
        );
        let given_back = ModelUse {
            gave_back_unused: now - DAY,
            ..done
        };
        assert!(
            unused_everywhere(Some(given_back), || Some(true), quiet, now),
            "given back here: not fetched again on popularity"
        );
        assert!(unused_everywhere(
            Some(given_back),
            || Some(true),
            || SwarmUse::Unknown,
            now
        ));
        assert!(
            !unused_everywhere(Some(given_back), || Some(true), || SwarmUse::Served, now),
            "served since: fetched"
        );
        let stale = ModelUse {
            gave_back_unused: now - 29 * DAY,
            ..done
        };
        assert!(
            !unused_everywhere(Some(stale), || Some(true), quiet, now),
            "two windows on, it may be tried again"
        );
    }

    /// The user (2026-11-14): "we need to be leaning much harder towards
    /// usable/popular/recent models". A model is tried on what people run
    /// outside the swarm only where the swarm's online cards could hold it
    /// between them — four cards of 6-12 GB try a 27B, not a 122B.
    #[test]
    fn a_model_is_tried_only_where_the_swarms_cards_could_run_it() {
        const GIB: u64 = 1024 * 1024 * 1024;
        let four_cards_mb = 8_192 + 12_288 + 12_288 + 6_144;
        assert!(
            the_cards_could_run(70 * GIB, 0),
            "no card heard of: anything may be tried"
        );
        assert!(the_cards_could_run(16 * GIB, four_cards_mb), "a 27B");
        assert!(!the_cards_could_run(42 * GIB, four_cards_mb), "a 70B");
        assert!(!the_cards_could_run(70 * GIB, four_cards_mb), "a 122B");
    }
}
