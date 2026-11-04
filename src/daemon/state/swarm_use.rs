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
//! **What is NOT withheld**: gossip, parts and updates. A computer that may not
//! use the swarm still fetches parts (which is how it starts to share), hears
//! the swarm and updates itself — cutting older releases off entirely would
//! split the swarm at every release for the minutes auto-update takes, and
//! drop their parts out of everyone's splits (`.claude/rules/architecture.md`
//! § "Additive Protocol Evolution"). Its owner's models held whole still run
//! here, and its own pool's devices still serve it (`pool::scope`). No LAN
//! exemption: `is_lan_peer` cannot tell the same Wi-Fi from a VPN — the reason
//! the metered hold kept none either (`docs/invariants/network.md` § "A metered
//! connection holds back what this node uploads for others").
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
/// computers run its requests: one behind is the normal state for the minutes
/// an update takes to reach it; two is a node that is not updating.
pub const RELEASES_BEHIND_TO_LOSE_USE: u64 = 2;

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
pub fn verdict(parts_held: usize, running: &str, newest: &str) -> SwarmUse {
    if releases_behind(running, newest).is_some_and(|b| b >= RELEASES_BEHIND_TO_LOSE_USE) {
        return SwarmUse::TooFarBehind {
            running: running.to_string(),
            newest: newest.to_string(),
        };
    }
    if parts_held < PARTS_TO_USE_THE_SWARM {
        return SwarmUse::TooFewParts { held: parts_held };
    }
    SwarmUse::Allowed
}

impl SharedState {
    /// The newest release this node knows of: its own, or a newer one its
    /// update check found SIGNED (`UpdateInfo::signature_verified`). Never a
    /// peer's claim, never a bare tag (module docs).
    pub fn newest_release_known(&self) -> String {
        let own = env!("CARGO_PKG_VERSION");
        self.events
            .update_state
            .try_read()
            .ok()
            .and_then(|s| {
                s.update_available
                    .as_ref()
                    .filter(|info| info.signature_verified)
                    .map(|info| info.latest_version.clone())
            })
            .filter(|latest| crate::update::is_newer_version(own, latest))
            .unwrap_or_else(|| own.to_string())
    }

    /// May THIS node have other computers run its requests? Read by
    /// `pool::scope::request_scope` (which computers a request may reach) and
    /// the scheduler's up-front refusal, and shown on the dashboard.
    pub fn swarm_use_here(&self) -> SwarmUse {
        verdict(
            self.parts_shared_here(),
            env!("CARGO_PKG_VERSION"),
            &self.newest_release_known(),
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
        let newest = self.newest_release_known();
        let verdict = verdict(held, running, &newest);
        serde_json::json!({
            "verdict": verdict.key(),
            "parts_held": held,
            "parts_needed": PARTS_TO_USE_THE_SWARM,
            "running": running,
            "newest": newest,
            "releases_behind": releases_behind(running, &newest),
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
                &self.newest_release_known(),
            ),
            None => SwarmUse::Allowed,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
            verdict(40, "0.3.268-alpha", "0.3.270-alpha"),
            SwarmUse::TooFarBehind {
                running: "0.3.268-alpha".into(),
                newest: "0.3.270-alpha".into()
            }
        );
        assert!(verdict(40, "0.3.269-alpha", "0.3.270-alpha").is_allowed());
    }

    #[test]
    fn a_computer_must_share_two_parts_to_use_the_swarm() {
        assert_eq!(
            verdict(0, "0.3.270-alpha", "0.3.270-alpha"),
            SwarmUse::TooFewParts { held: 0 }
        );
        assert_eq!(
            verdict(1, "0.3.270-alpha", "0.3.270-alpha"),
            SwarmUse::TooFewParts { held: 1 }
        );
        assert!(verdict(2, "0.3.270-alpha", "0.3.270-alpha").is_allowed());
    }

    #[test]
    fn a_new_minor_version_does_not_cut_off_every_computer_on_the_last_one() {
        // A mistyped or premature 0.4.0 tag, or the honest first release of a
        // series: one release behind, not refused.
        assert!(verdict(5, "0.3.270-alpha", "0.4.0-alpha").is_allowed());
        assert!(!verdict(5, "0.3.270-alpha", "0.4.1-alpha").is_allowed());
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
        assert!(verdict(5, "", "0.3.270-alpha").is_allowed());
        assert!(!verdict(0, "", "0.3.270-alpha").is_allowed());
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
