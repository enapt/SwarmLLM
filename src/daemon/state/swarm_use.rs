//! Who may have the swarm's computers run a request: a computer that SHARES —
//! holds at least [`PARTS_TO_USE_THE_SWARM`] model parts — and runs a release at
//! most one behind the newest this node knows of.
//!
//! The swarm runs on what its computers share. Until v0.3.271 a node served
//! any peer's request, so a computer that shared nothing — automatic sharing
//! switched off, or its storage cut to below one part — could use everyone
//! else's: one in Russia (`54c49961`, 2026-11-02 → 11-04) held no parts, ran
//! two releases behind with its updates off, and was twice chosen to carry a
//! model it never fetched (FUTURE_WORK #400). Nothing was checked: the
//! dispatcher's caps limit how MUCH a peer runs here, never WHO.
//!
//! **What is judged**, for a peer: what it says of itself — the parts it holds
//! (its capability's list, or the parts its announcements credited it here,
//! whichever is MORE) and the version its capability names — which every
//! release since the capability existed sends, so a node on an older release
//! is judged like a new one. For this node: the parts its last announcement
//! named (what peers credit it) and its own version. Both against the newest
//! release this node's update check found SIGNED. Never against a version a
//! PEER claims: one peer advertising 9.9.9 would otherwise make every node
//! refuse itself (the same rule as `update::PeerVersionWatch`).
//!
//! **What a third party cannot do** (security review, 2026-11-04): replay a
//! peer's old signed capability to get it refused. A capability naming an
//! older version than the one installed is not installed
//! ([`capability_may_replace`] — a node's version changes only across a
//! restart, which drops its registry entry); fewer parts in a replayed one do
//! not count while its announcements say more (freshness-checked gossip). Nor
//! can a tag on the release page alone cut the swarm: an unsigned tag is not
//! "newest", and a new minor version counts as one release, not as infinitely
//! far ahead.
//!
//! **What it does NOT stop**: a modified build can claim parts and a version
//! it does not have, and a node key costs nothing to make — the limit of any
//! rule keyed on what a peer says about itself (`docs/CREDITS_DESIGN.md` § 6,
//! `docs/book/src/architecture/security.md` § "Misuse of the Network"). It
//! stops the released program, used as released, from taking without giving.
//!
//! **A computer too far behind loses contact, not only use** (the user's call,
//! 2026-11-04: "if they dont update they lose communication"). From v0.3.271 a
//! node disconnects a computer [`RELEASES_BEHIND_TO_LOSE_USE`] or more
//! releases behind the newest it knows, and refuses it for [`CUT_OFF_FOR`] —
//! gossip, parts and the DHT go with the connection
//! ([`SharedState::cut_off_for`], acted on only by the network loop). One
//! release behind keeps everything: that is where every node is for the
//! minutes an update takes to reach it, so the swarm never splits at a
//! release. A computer that falls behind still updates — updates come from
//! GitHub, never from peers — and is let back in on its first connection
//! after. Judged only on what the computer said ITSELF (its Identify, its own
//! capability sent direct, [`SharedState::note_peers_own_version`]): a
//! capability another node gossips or the DHT stores can be a recording.
//! Bitcoin Core drops a peer below its minimum protocol version at the
//! handshake (`MIN_PEER_PROTO_VERSION`, "using obsolete version; disconnecting")
//! and Ethereum's fork id (EIP-2124) rejects a stale peer there while letting
//! one that is merely behind sync; both decide before anything else is sent,
//! which is why a node's version rides in Identify from v0.3.271.
//!
//! **What the parts gate does NOT withhold**: gossip, parts and updates. A
//! computer that shares too few parts still fetches parts (which is how it
//! starts to share) and hears the swarm. Its owner's models held whole still
//! run here, and its own pool's devices still serve it — and are never cut off
//! (`pool::scope`). No LAN exemption: `is_lan_peer` cannot tell the same Wi-Fi
//! from a VPN — the reason the metered hold kept none either
//! (`docs/invariants/network.md` § "A metered connection holds back what this
//! node uploads for others").
//!
//! Prior art: IPFS BitSwap stops sending to a peer whose debt ratio grows, and
//! Petals planned priority for computers that serve; both judge what the
//! serving side can see. Here the serving side sees only the capability, so the
//! rule is a threshold on it, not a ledger.

use crate::daemon::state::SharedState;
use crate::error::SwarmError;
use crate::types::NodeId;

/// The fewest parts a computer holds to have other computers run its requests.
/// Two, not one (the user's call, 2026-11-04): one tiny part kept by hand is a
/// token, two is a start that auto-manage grows.
pub const PARTS_TO_USE_THE_SWARM: usize = 2;

/// How many releases behind the newest a computer may run and still have other
/// computers run its requests — and still be connected to them: one behind is
/// the normal state for the minutes an update takes to reach it; two is a node
/// that is not updating. Counted in PUBLISHED releases ([`ReleasesKnown`]).
pub const RELEASES_BEHIND_TO_LOSE_USE: u64 = 2;

/// How long a computer cut off for running too old a release is refused
/// before it may connect again — and be judged again, on what it says then.
/// Short, because a computer that updated waits out the rest of it: an old one
/// costs a connection and an Identify every ten minutes, a node that just
/// updated at most ten minutes away from each computer that cut it.
pub const CUT_OFF_FOR: std::time::Duration = std::time::Duration::from_secs(10 * 60);

/// The prefix of this build's libp2p Identify agent version: `swarmllm/` and
/// the version (`agent_version`). Older releases send `rust-libp2p/…`, which
/// names no SwarmLLM version.
const AGENT_PREFIX: &str = "swarmllm/";

/// What this node tells every computer it meets in Identify, so a computer
/// can be judged before anything else is sent to it.
pub fn agent_version() -> String {
    format!("{AGENT_PREFIX}{}", env!("CARGO_PKG_VERSION"))
}

/// The SwarmLLM version an Identify agent version names, if it names one.
pub fn version_in_agent(agent: &str) -> Option<&str> {
    let version = agent.strip_prefix(AGENT_PREFIX)?;
    (version.len() <= 64 && version_triple(version).is_some()).then_some(version)
}

/// May a computer have other computers run its requests?
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SwarmUse {
    Allowed,
    /// It holds fewer than [`PARTS_TO_USE_THE_SWARM`] parts.
    TooFewParts {
        held: usize,
    },
    /// It runs [`RELEASES_BEHIND_TO_LOSE_USE`] or more releases behind `newest`.
    TooFarBehind {
        running: String,
        newest: String,
    },
}

impl SwarmUse {
    pub fn is_allowed(&self) -> bool {
        matches!(self, Self::Allowed)
    }

    /// One word for the API and the dashboard.
    pub fn key(&self) -> &'static str {
        match self {
            Self::Allowed => "allowed",
            Self::TooFewParts { .. } => "too_few_parts",
            Self::TooFarBehind { .. } => "too_far_behind",
        }
    }

    /// What this node's owner is told when a request for `model_id` would
    /// need another computer, or `None` when it may use them.
    pub fn refusal(&self, model_id: String) -> Option<SwarmError> {
        match self {
            Self::Allowed => None,
            Self::TooFewParts { held } => Some(SwarmError::TooFewPartsToUseTheSwarm {
                model_id,
                parts_held: *held,
            }),
            Self::TooFarBehind { running, newest } => Some(SwarmError::TooFarBehindToUseTheSwarm {
                model_id,
                running: running.clone(),
                newest: newest.clone(),
            }),
        }
    }

    /// What a PEER is told when this node will not start its request.
    ///
    /// `ServiceUnavailable`'s shape, like the dispatcher's other refusals: the
    /// coordinator bars this node from the request and re-plans elsewhere,
    /// without a penalty and without retracting what this node holds — the
    /// right reading for a release that judges the peer differently from this
    /// node (it may know a newer release), and the only one an older
    /// coordinator has. The words reach its owner where nothing else serves.
    pub fn peer_refusal(&self) -> String {
        let why = match self {
            Self::Allowed => return String::new(),
            Self::TooFewParts { held } => format!(
                "this machine runs requests for computers that share at least \
                 {PARTS_TO_USE_THE_SWARM} model parts with the swarm, and yours shares {held} — \
                 turn on Auto-Download Models in Settings so it fetches some"
            ),
            Self::TooFarBehind { running, newest } => format!(
                "this machine runs requests for computers at most one release behind, and \
                 yours runs {running} while {newest} is out — update it"
            ),
        };
        SwarmError::ServiceUnavailable(why).to_string()
    }
}

/// The version triple of `0.3.270-alpha` (a leading `v` allowed), or `None`
/// for anything else.
fn version_triple(v: &str) -> Option<(u64, u64, u64)> {
    let core = v.trim().trim_start_matches('v').split(['-', '+']).next()?;
    let mut parts = core.split('.');
    let triple = (
        parts.next()?.parse().ok()?,
        parts.next()?.parse().ok()?,
        parts.next()?.parse().ok()?,
    );
    parts.next().is_none().then_some(triple)
}

/// How many releases `running` is behind `newest`: the patch numbers between
/// them within one minor version, 0 when level or ahead, `None` when either is
/// unreadable (then version decides nothing).
///
/// Counts patch NUMBERS, so a number never published (.263, .267) counts as a
/// release — at worst a node one published release behind across such a gap
/// is asked to update a few minutes early.
///
/// **Across a new minor (or major) version, the new series' first release is
/// ONE release** and each of its patches one more — the releases the older
/// series had after `running` are not known here, so this errs toward letting
/// a computer in. It was "infinitely far": every node on 0.3.x would have been
/// refused the moment 0.4.0 was tagged, for the minutes auto-update takes
/// (security review, 2026-11-04).
pub fn releases_behind(running: &str, newest: &str) -> Option<u64> {
    let (r, n) = (version_triple(running)?, version_triple(newest)?);
    Some(if r >= n {
        0
    } else if (r.0, r.1) == (n.0, n.1) {
        n.2 - r.2
    } else {
        n.2.saturating_add(1)
    })
}

/// What this node knows of the releases: the newest its update check found
/// SIGNED (or its own version), and the published releases the check's list
/// named.
#[derive(Clone, Debug, Default)]
pub struct ReleasesKnown {
    newest: String,
    listed: Vec<(u64, u64, u64)>,
}

impl ReleasesKnown {
    pub fn new(newest: &str, listed: &[String]) -> Self {
        Self {
            newest: newest.to_string(),
            listed: listed.iter().filter_map(|v| version_triple(v)).collect(),
        }
    }

    pub fn newest(&self) -> &str {
        &self.newest
    }

    /// How many releases `running` is behind the newest: the PUBLISHED
    /// releases after it, up to the newest — never more than
    /// [`releases_behind`]'s count of version numbers, which it can only
    /// lower (the list is unsigned: a release page that lists less lets more
    /// computers in, never fewer). `None` when either version is unreadable,
    /// and when this node has no list to count (no update check yet, checks
    /// off, GitHub unreachable, the state being written): the numbers alone
    /// count releases never published (.263, .267), so a computer one release
    /// behind across such a gap would have been cut (security review,
    /// 2026-11-04). A node that cannot count judges nobody behind.
    pub fn behind(&self, running: &str) -> Option<u64> {
        let by_number = releases_behind(running, &self.newest)?;
        if by_number == 0 {
            return Some(0);
        }
        if self.listed.is_empty() {
            return None;
        }
        let (r, n) = (version_triple(running)?, version_triple(&self.newest)?);
        let published = self.listed.iter().filter(|v| r < **v && **v <= n).count() as u64;
        Some(by_number.min(published))
    }
}

/// May `incoming` replace the capability `held` for the same peer? Not when it
/// names an OLDER version: a node's version changes only across a restart, and
/// a restart drops its registry entry (`network::manager::connections`), so
/// within one entry's life an older version is a recorded frame played back —
/// which, read by the swarm-use gate, would have every node refuse an honest
/// peer (security review, 2026-11-04). Asked by every place a peer's
/// capability is installed (gossip, the DHT record).
pub(crate) fn capability_may_replace(
    held: Option<&crate::types::NodeCapability>,
    incoming: &crate::types::NodeCapability,
) -> bool {
    version_may_replace(held.map(|held| held.version.as_str()), &incoming.version)
}

/// [`capability_may_replace`] on the two versions.
fn version_may_replace(held: Option<&str>, incoming: &str) -> bool {
    held.is_none_or(|held| !crate::update::is_newer_version(incoming, held))
}

/// The one rule, on plain facts.
pub fn verdict(parts_held: usize, running: &str, releases: &ReleasesKnown) -> SwarmUse {
    if let Some(behind) = too_far_behind(running, releases) {
        return behind;
    }
    if parts_held < PARTS_TO_USE_THE_SWARM {
        return SwarmUse::TooFewParts { held: parts_held };
    }
    SwarmUse::Allowed
}

/// The version half of [`verdict`] — the half that also costs a computer its
/// connections ([`SharedState::cut_off_for`]).
fn too_far_behind(running: &str, releases: &ReleasesKnown) -> Option<SwarmUse> {
    releases
        .behind(running)
        .is_some_and(|b| b >= RELEASES_BEHIND_TO_LOSE_USE)
        .then(|| SwarmUse::TooFarBehind {
            running: running.to_string(),
            newest: releases.newest().to_string(),
        })
}

impl SharedState {
    /// The newest release this node knows of: its own, or a newer one its
    /// update check found SIGNED (`UpdateInfo::signature_verified`). Never a
    /// peer's claim, never a bare tag (module docs).
    pub fn newest_release_known(&self) -> String {
        self.releases_known().newest
    }

    /// [`Self::newest_release_known`] with the published releases the update
    /// check listed, which [`ReleasesKnown::behind`] counts. Never waits: the
    /// network loop asks it; a check writing the state right now reads as
    /// this node's own version and no list, which judges nobody behind.
    pub fn releases_known(&self) -> ReleasesKnown {
        let own = env!("CARGO_PKG_VERSION");
        let Ok(state) = self.events.update_state.try_read() else {
            return ReleasesKnown::new(own, &[]);
        };
        let newest = state
            .update_available
            .as_ref()
            .filter(|info| info.signature_verified)
            .map(|info| info.latest_version.as_str())
            .filter(|latest| crate::update::is_newer_version(own, latest))
            .unwrap_or(own);
        ReleasesKnown::new(newest, &state.releases_listed)
    }

    /// Note the version `node` stated ITSELF: its Identify agent version, or a
    /// capability it sent this node direct (dispatch's `NodeCapabilityUpdate`
    /// arm, only while connected). The newest it stated wins — its version
    /// changes only across a restart, which ends the connection, and the
    /// network manager forgets the note with its last connection
    /// ([`Self::forget_peers_own_version`]). Never a capability gossiped or
    /// stored in the DHT: another node can play an old one back.
    pub fn note_peers_own_version(&self, node: &NodeId, version: &str) {
        if version.len() > 64 || version_triple(version).is_none() {
            return;
        }
        self.peers_own_versions
            .entry(node.clone())
            .and_modify(|held| {
                if crate::update::is_newer_version(held, version) {
                    *held = version.to_string();
                }
            })
            .or_insert_with(|| version.to_string());
    }

    /// A capability `node` sent (dispatch, after its sender check), naming `version`:
    /// its version is the node's own word only when it came DIRECT and the
    /// node is connected — a gossiped one may be another node playing an old
    /// one back, and its node stays judged on what it said itself.
    pub fn note_capability_heard(
        &self,
        node: &NodeId,
        version: &str,
        transport: crate::types::MessageTransport,
    ) {
        if transport == crate::types::MessageTransport::Direct
            && self.connected_node_ids.contains(node)
        {
            self.note_peers_own_version(node, version);
        }
    }

    /// The network manager, on `node`'s last connection closing.
    pub fn forget_peers_own_version(&self, node: &NodeId) {
        self.peers_own_versions.remove(node);
    }

    /// Must this node stop talking to `node`, which says it runs `running`?
    /// `Some(TooFarBehind)` when it is [`RELEASES_BEHIND_TO_LOSE_USE`] or more
    /// published releases behind the newest this node knows. Never a device of
    /// this node's pool — the owner admitted it — and not while the pool state
    /// cannot be read (judged again on the next tick: cutting is the harsh
    /// direction). Acted on only by the network loop: at Identify, and every
    /// liveness tick over [`Self::peers_own_versions`].
    pub fn cut_off_for(&self, node: &NodeId, running: &str) -> Option<SwarmUse> {
        let releases = self.releases_known();
        let behind = too_far_behind(running, &releases)?;
        match crate::pool::scope::pool_membership(self, node) {
            Some(false) => Some(behind),
            Some(true) | None => None,
        }
    }

    /// [`Self::cut_off_for`] on the version `node` last stated itself; `None`
    /// when it has stated none (an older release says it in the capability it
    /// sends as it connects).
    pub fn cut_off_verdict(&self, node: &NodeId) -> Option<SwarmUse> {
        let running = self.peers_own_versions.get(node)?.value().clone();
        self.cut_off_for(node, &running)
    }

    /// May THIS node have other computers run its requests? Read by
    /// `pool::scope::request_scope` (which computers a request may reach) and
    /// the scheduler's up-front refusal, and shown on the dashboard.
    pub fn swarm_use_here(&self) -> SwarmUse {
        verdict(
            self.parts_shared_here(),
            env!("CARGO_PKG_VERSION"),
            &self.releases_known(),
        )
    }

    /// The parts this node shares as its PEERS count them: its last parts
    /// announcement (`own_shard_announce`), which is what they credit it —
    /// withheld copies and parts that fail their checks are not in it. The
    /// registry's own credit counted those too, so the dashboard could say
    /// "allowed" while every peer refused (security review, 2026-11-04). The
    /// registry only before the health round's first announcement.
    fn parts_shared_here(&self) -> usize {
        match self.models.own_shard_announce.load().as_ref() {
            Some(announce) => announce.shards.len(),
            None => self
                .model_registry
                .parts_credited_to(self.identity.node_id()),
        }
    }

    /// This node's verdict as the dashboard shows it, in BOTH stats payloads
    /// (`GET /api/admin/stats` and the WebSocket tick): why chats cannot use
    /// other computers, and what lifts it.
    pub fn swarm_use_json(&self) -> serde_json::Value {
        let held = self.parts_shared_here();
        let running = env!("CARGO_PKG_VERSION");
        let releases = self.releases_known();
        let verdict = verdict(held, running, &releases);
        serde_json::json!({
            "verdict": verdict.key(),
            "parts_held": held,
            "parts_needed": PARTS_TO_USE_THE_SWARM,
            "running": running,
            "newest": releases.newest(),
            "releases_behind": releases.behind(running),
            "auto_manage": self.models.auto_manage_enabled.load(std::sync::atomic::Ordering::Relaxed),
        })
    }

    /// May `peer` have THIS node run a request it starts? Asked by the
    /// dispatcher for every request that starts here (`admits_a_new_request`,
    /// a tensor-parallel layer's prompt pass) and shown in the peer list.
    ///
    /// A device of this node's pool always may — the owner admitted it. A
    /// peer whose capability has not arrived yet may too: silence is no
    /// reason to think otherwise (`NodeCapability::can_serve_inference`'s
    /// rule), and a released node sends its capability when it connects.
    pub fn swarm_use_of_peer(&self, peer: &NodeId) -> SwarmUse {
        match self.peer_registry.get(peer) {
            Some(entry) => self.swarm_use_of_entry(&entry),
            None => SwarmUse::Allowed,
        }
    }

    /// Test fixture: this node announced [`PARTS_TO_USE_THE_SWARM`] parts, so
    /// it may use the swarm. The routing tests build a node holding nothing to
    /// plan its work onto peers; the gate has tests of its own.
    #[cfg(test)]
    pub(crate) fn shares_enough_to_use_the_swarm(&self) {
        let model = crate::types::ModelId("shared-for-the-swarm-use-gate".into());
        let announce = crate::model::manifest::shard_announce(
            &self.model_registry,
            self.identity.node_id().clone(),
            (0..PARTS_TO_USE_THE_SWARM as u32)
                .map(|index| crate::types::ShardId {
                    model_id: model.clone(),
                    index,
                })
                .collect(),
            Vec::new(),
        );
        self.models
            .own_shard_announce
            .store(Some(std::sync::Arc::new(announce)));
    }

    /// [`Self::swarm_use_of_peer`] for a registry entry already in hand — the
    /// peer list builds its rows while iterating the registry, where a second
    /// lookup could wait on a writer behind the iterator's own lock.
    pub fn swarm_use_of_entry(&self, peer: &crate::types::PeerInfo) -> SwarmUse {
        if crate::pool::scope::is_pool_member(self, &peer.node_id) {
            return SwarmUse::Allowed;
        }
        match peer.capability.as_ref() {
            // The more of the two figures it has given: a replayed capability
            // listing fewer parts cannot outvote its announcements.
            Some(c) => verdict(
                c.hosted_shards
                    .len()
                    .max(self.model_registry.parts_credited_to(&peer.node_id)),
                &c.version,
                &self.releases_known(),
            ),
            None => SwarmUse::Allowed,
        }
    }
}

/// Every release number of `newest`'s series up to it: the list an update
/// check keeps where no number was skipped.
#[cfg(test)]
pub(crate) fn every_release_of_the_series_up_to(newest: &str) -> Vec<String> {
    let (major, minor, patch) = version_triple(newest).expect("a version");
    (0..=patch)
        .map(|p| format!("{major}.{minor}.{p}-alpha"))
        .collect()
}

/// Has `state` know every release of this build's series up to this one, as
/// its update check would list them — without a list nobody is judged.
#[cfg(test)]
pub(crate) fn list_every_release_up_to_this_one(state: &SharedState) {
    state
        .events
        .update_state
        .try_write()
        .expect("nothing else holds a test's update state")
        .releases_listed = every_release_of_the_series_up_to(env!("CARGO_PKG_VERSION"));
}

#[cfg(test)]
mod tests {
    use super::*;

    /// What a node knows whose update check listed every release number of
    /// the newest's series, up to it.
    fn known(newest: &str) -> ReleasesKnown {
        ReleasesKnown::new(newest, &every_release_of_the_series_up_to(newest))
    }

    /// This build's version with its patch number moved by `delta`.
    fn own_patch(delta: i64) -> String {
        let (major, minor, patch) = version_triple(env!("CARGO_PKG_VERSION")).unwrap();
        format!("{major}.{minor}.{}-alpha", patch as i64 + delta)
    }

    /// A node whose update check has listed nothing yet.
    fn unlisted_state() -> std::sync::Arc<SharedState> {
        let dir = tempfile::tempdir().unwrap().keep();
        let (state, _, _) = SharedState::new(
            crate::config::Config::default(),
            crate::identity::Identity::generate(),
            crate::storage::db::Database::open(&dir).unwrap(),
            std::sync::Arc::new(tokio::sync::Mutex::new(
                crate::inference::executor::ModelExecutor::new(),
            )),
            None,
        );
        state
    }

    /// A node whose update check listed every release up to this build's.
    fn state() -> std::sync::Arc<SharedState> {
        let state = unlisted_state();
        list_every_release_up_to_this_one(&state);
        state
    }

    async fn make_pool_member(state: &SharedState, member: &NodeId) {
        use crate::pool::types::{PoolMembership, PoolState};
        *state.credits.pool_state.write().await = Some(PoolState {
            pool_id: NodeId([0xC0; 32]),
            name: "home".into(),
            created_at: chrono::Utc::now(),
            owner_signature: vec![],
            members: vec![PoolMembership {
                node_id: member.clone(),
                credits_contributed: 0,
                joined_at: chrono::Utc::now(),
                acceptance_signature: vec![],
                invitation_id: uuid::Uuid::new_v4(),
                invitation_expires_at: chrono::Utc::now(),
                device_name: None,
                last_seen: None,
                online: false,
                device_stats: None,
                contribution_level: 100,
            }],
            shard_pins: Default::default(),
            total_lifetime_credits: 0,
            member_credit_split_pct: 0,
            generation: 0,
        });
    }

    /// **Behind is counted in PUBLISHED releases** where the update check
    /// listed them: .263 and .267 were never published, so a computer on .266
    /// when .268 came out was ONE release behind, not two — and the list can
    /// only lower the count of numbers, never raise it.
    #[test]
    fn a_release_number_never_published_does_not_count_as_a_release() {
        let listed: Vec<String> = [
            "0.3.268-alpha",
            "0.3.266-alpha",
            "0.3.265-alpha",
            "0.3.264-alpha",
        ]
        .map(String::from)
        .to_vec();
        let known = ReleasesKnown::new("0.3.268-alpha", &listed);
        assert_eq!(
            known.behind("0.3.266-alpha"),
            Some(1),
            "only .268 came after"
        );
        assert_eq!(known.behind("0.3.265-alpha"), Some(2));
        assert_eq!(known.behind("0.3.268-alpha"), Some(0));
        // No list, nothing to count: the numbers alone would have called .266
        // two behind. Level is level either way.
        let unlisted = ReleasesKnown::new("0.3.268-alpha", &[]);
        assert_eq!(unlisted.behind("0.3.266-alpha"), None);
        assert_eq!(unlisted.behind("0.3.260-alpha"), None);
        assert_eq!(unlisted.behind("0.3.268-alpha"), Some(0));
        assert!(verdict(5, "0.3.260-alpha", &unlisted).is_allowed());
        // A list naming releases past the newest, or more than the numbers
        // allow, never raises the count.
        let padded: Vec<String> = ["0.3.270-alpha", "0.3.269-alpha", "0.3.268-alpha"]
            .map(String::from)
            .to_vec();
        assert_eq!(
            ReleasesKnown::new("0.3.269-alpha", &padded).behind("0.3.268-alpha"),
            Some(1)
        );
        assert!(verdict(5, "0.3.266-alpha", &known).is_allowed());
    }

    #[test]
    fn this_build_names_its_release_in_identify_and_older_ones_do_not() {
        assert_eq!(
            version_in_agent(&agent_version()),
            Some(env!("CARGO_PKG_VERSION"))
        );
        assert_eq!(
            version_in_agent("rust-libp2p/0.47.0"),
            None,
            "v0.3.270 and older"
        );
        assert_eq!(version_in_agent("swarmllm/not-a-version"), None);
        assert_eq!(
            version_in_agent(&format!("swarmllm/1.2.3-{}", "x".repeat(80))),
            None
        );
    }

    /// **Two releases behind costs a computer its connections; one behind
    /// keeps them** — and a device of this node's pool is never cut off.
    #[tokio::test]
    async fn a_computer_two_releases_behind_is_cut_off_and_a_pool_device_never_is() {
        let peer = NodeId([0x42; 32]);
        assert!(
            unlisted_state()
                .cut_off_for(&peer, &own_patch(-5))
                .is_none(),
            "a node that has not listed the releases judges nobody"
        );
        let state = state();
        assert!(
            state.cut_off_for(&peer, &own_patch(-1)).is_none(),
            "one behind"
        );
        assert!(state.cut_off_for(&peer, &own_patch(0)).is_none());
        assert_eq!(
            state.cut_off_for(&peer, &own_patch(-2)).map(|v| v.key()),
            Some("too_far_behind")
        );
        assert!(state.cut_off_for(&peer, "unreadable").is_none());
        make_pool_member(&state, &peer).await;
        assert!(
            state.cut_off_for(&peer, &own_patch(-5)).is_none(),
            "a pool device"
        );
        // While the pool state is being written, nobody is cut: that is the
        // harsh direction, judged again on the next tick.
        let stranger = NodeId([0x43; 32]);
        let writing = state.credits.pool_state.write().await;
        assert!(state.cut_off_for(&stranger, &own_patch(-5)).is_none());
        drop(writing);
        assert!(state.cut_off_for(&stranger, &own_patch(-5)).is_some());
    }

    /// **Only a computer's own word on its release decides** — a capability
    /// it sent DIRECT while connected, never one gossiped (a recording another
    /// node can play back); the newest it stated wins, and the note goes with
    /// its last connection.
    #[test]
    fn a_computer_is_judged_on_the_release_it_stated_itself() {
        use crate::types::MessageTransport::{Direct, Gossip};
        let state = state();
        let peer = NodeId([0x44; 32]);
        state.note_capability_heard(&peer, &own_patch(-3), Direct);
        assert!(
            state.cut_off_verdict(&peer).is_none(),
            "not connected: nothing noted"
        );
        state.connected_node_ids.insert(peer.clone());
        state.note_capability_heard(&peer, &own_patch(-3), Gossip);
        assert!(
            state.cut_off_verdict(&peer).is_none(),
            "gossiped: nothing noted"
        );
        state.note_capability_heard(&peer, &own_patch(-3), Direct);
        assert!(
            state.cut_off_verdict(&peer).is_some(),
            "its own word, three behind"
        );
        state.note_peers_own_version(&peer, &own_patch(0));
        assert!(
            state.cut_off_verdict(&peer).is_none(),
            "the newest it stated wins"
        );
        state.note_peers_own_version(&peer, &own_patch(-3));
        assert!(
            state.cut_off_verdict(&peer).is_none(),
            "an older one never replaces it"
        );
        state.forget_peers_own_version(&peer);
        assert!(state.peers_own_versions.is_empty());
    }

    #[test]
    fn releases_behind_counts_patch_numbers_within_a_minor_version() {
        assert_eq!(releases_behind("0.3.268-alpha", "0.3.270-alpha"), Some(2));
        assert_eq!(releases_behind("0.3.269-alpha", "0.3.270-alpha"), Some(1));
        assert_eq!(releases_behind("0.3.270-alpha", "0.3.270-alpha"), Some(0));
        assert_eq!(releases_behind("0.3.271-alpha", "0.3.270-alpha"), Some(0));
        assert_eq!(releases_behind("v0.3.268-alpha", "v0.3.270-alpha"), Some(2));
        // A new series: its first release is one, each patch one more.
        assert_eq!(releases_behind("0.3.270-alpha", "0.4.0-alpha"), Some(1));
        assert_eq!(releases_behind("0.3.270-alpha", "0.4.1-alpha"), Some(2));
        assert_eq!(releases_behind("0.3.270-alpha", "1.0.0"), Some(1));
        assert_eq!(releases_behind("", "0.3.270-alpha"), None);
        assert_eq!(releases_behind("custom", "0.3.270-alpha"), None);
    }

    #[test]
    fn a_computer_two_releases_behind_may_not_use_the_swarm_and_one_behind_may() {
        // The Russian node of 2026-11-04: v0.3.268 with v0.3.270 out.
        assert_eq!(
            verdict(40, "0.3.268-alpha", &known("0.3.270-alpha")),
            SwarmUse::TooFarBehind {
                running: "0.3.268-alpha".into(),
                newest: "0.3.270-alpha".into()
            }
        );
        assert!(verdict(40, "0.3.269-alpha", &known("0.3.270-alpha")).is_allowed());
    }

    #[test]
    fn a_computer_must_share_two_parts_to_use_the_swarm() {
        assert_eq!(
            verdict(0, "0.3.270-alpha", &known("0.3.270-alpha")),
            SwarmUse::TooFewParts { held: 0 }
        );
        assert_eq!(
            verdict(1, "0.3.270-alpha", &known("0.3.270-alpha")),
            SwarmUse::TooFewParts { held: 1 }
        );
        assert!(verdict(2, "0.3.270-alpha", &known("0.3.270-alpha")).is_allowed());
    }

    #[test]
    fn a_new_minor_version_does_not_cut_off_every_computer_on_the_last_one() {
        // A mistyped or premature 0.4.0 tag, or the honest first release of a
        // series: one release behind, not refused.
        assert!(verdict(5, "0.3.270-alpha", &known("0.4.0-alpha")).is_allowed());
        assert!(!verdict(5, "0.3.270-alpha", &known("0.4.1-alpha")).is_allowed());
    }

    #[test]
    fn a_capability_naming_an_older_version_never_replaces_a_newer_one() {
        let held = Some("0.3.271-alpha");
        // A frame recorded before the peer updated, played back.
        assert!(!version_may_replace(held, "0.3.270-alpha"));
        assert!(version_may_replace(held, "0.3.271-alpha"));
        assert!(version_may_replace(held, "0.3.272-alpha"));
        assert!(version_may_replace(None, "0.3.270-alpha"));
    }

    #[test]
    fn an_unreadable_version_is_judged_on_its_parts_alone() {
        assert!(verdict(5, "", &known("0.3.270-alpha")).is_allowed());
        assert!(!verdict(0, "", &known("0.3.270-alpha")).is_allowed());
    }

    #[test]
    fn a_peer_is_told_in_the_words_its_coordinator_reads_as_cannot_serve() {
        // `router::message_means_peer_cannot_serve` keys on this prefix: the
        // coordinator re-plans elsewhere, no penalty, no retraction.
        let prefix = SwarmError::ServiceUnavailable(String::new()).to_string();
        for refused in [
            SwarmUse::TooFewParts { held: 0 },
            SwarmUse::TooFarBehind {
                running: "0.3.268-alpha".into(),
                newest: "0.3.270-alpha".into(),
            },
        ] {
            assert!(refused.peer_refusal().starts_with(prefix.trim_end()));
        }
    }
}
