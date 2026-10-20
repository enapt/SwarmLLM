//! Announce-then-pull for model manifests (FUTURE_WORK #91 (2)).
//!
//! A manifest is ~35 KB and changes rarely. Measured on the release node with
//! the fleet on v0.3.240 (2026-10-16, idle, 6 peers), `swarm/models` was 84% of
//! the gossip it sent — every manifest a repeat, each delivered ~3 times — and
//! a node that (re)connected was sent every peer's whole set point to point: a
//! probe node joining on 2026-10-16 received 84 manifests (~2.9 MB) from six
//! peers in its first 20 seconds, though they described 26 versions.
//!
//! A peer advertising `features::MANIFEST_DIGESTS` is instead told WHICH
//! version of each manifest a node holds ([`crate::types::ManifestDigest`],
//! ~200 bytes a model) and pulls ([`crate::types::ManifestPull`]) only a
//! version it has not seen. Bitcoin's `inv`/`getdata`, Plumtree's lazy push and
//! GossipSub's own IHAVE/IWANT make the same trade. The pulled manifest arrives
//! as an ordinary point-to-point `ModelManifest`, so every check ingestion
//! makes (`verify_hash_strict`, the size caps, `judge_peer_manifest`) applies
//! to it unchanged.
//!
//! [`ManifestSync`] is the receiving half's memory: what has been seen, what is
//! being pulled, and what each peer has been answered. The sending half is
//! `health::monitor::broadcast_manifests`.

use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};

use dashmap::DashMap;

use crate::types::{ManifestDigest, ManifestVersion, ModelId, ModelManifest, NodeId};

/// How long a pull may go unanswered before it is asked of every peer that
/// takes digests ([`ManifestSync::retry_overdue_pulls`]). A newcomer hears a
/// digest from every peer it connects to within seconds of each other, so
/// without this it would pull each version once per peer — the very copies
/// this exists to remove.
pub const PULL_PATIENCE: Duration = Duration::from_secs(20);

/// How long a verified version is remembered. A version is content-addressed,
/// so having seen it once means it has been taken in — merged, adopted, or
/// refused by ingestion — and pulling it again changes nothing. Bounded so a
/// refused version is looked at again once a day, and the map holds at most
/// the versions the swarm carried in that time.
pub const SEEN_FOR: Duration = Duration::from_secs(24 * 60 * 60);

/// The most versions one digest is read for, and the most models one pull asks
/// for. A real node gossips a few dozen; anything far beyond is hostile. A
/// sender splits its digest at `MAX_PULL_MODELS`, so every digest it sends can
/// be pulled whole.
pub const MAX_DIGEST_VERSIONS: usize = 1024;
pub const MAX_PULL_MODELS: usize = 64;

/// The longest model id either message is read for — the bound the
/// dispatcher's other arms apply to a peer's model ids.
pub const MAX_MODEL_ID_BYTES: usize = 256;

/// The most pulls out at once, and the most versions remembered as seen. A
/// connected peer can name made-up models as fast as it can send digests, and
/// every one would otherwise be a pull held for [`PULL_PATIENCE`] — or, sent
/// back as a manifest, a version held for [`SEEN_FOR`]. Past either bound
/// nothing new is added, and a version not remembered is merely pulled again
/// at a later digest. One peer may hold no more pulls than it would answer in
/// a window ([`ANSWERS_PER_WINDOW`]), so a single peer cannot fill the bound
/// and starve every other.
pub const MAX_PULLS_IN_FLIGHT: usize = 1024;
pub const MAX_SEEN_VERSIONS: usize = 16_384;

/// The most manifests answered to one peer in [`ANSWER_WINDOW`]. A pull is a
/// few bytes and its answer up to ~35 KB a model, so an unbounded answer is a
/// way to make this node upload for a stranger; a newcomer pulls each version
/// once, well inside this.
pub const ANSWERS_PER_WINDOW: usize = 128;
pub const ANSWER_WINDOW: Duration = Duration::from_secs(60);

/// A model answered to one peer is not answered to it again for this long:
/// one answer per model per peer, which is what the full catch-up it replaces
/// delivered per reconnect. Without it a peer could pull the same few models
/// at the full budget every window, ~75 KB/s for ~2 KB a minute (security
/// review, 2026-10-17). A peer whose answer was lost asks every peer at its
/// retry, so another holder answers it.
pub const ANSWER_AGAIN_AFTER: Duration = Duration::from_secs(10 * 60);

/// How long after a peer's connection its manifests may still be on their
/// way: the peer's own round (30-120 s apart) catches a new connection up, and
/// what its digest names is pulled and answered in seconds. Inside it, a model
/// the peer's capability lists is "not heard yet", never "unknown"
/// (`catching_up`); past it, a model it lists and never sent is not one this
/// node can use.
pub const CATCH_UP_WITHIN: Duration = Duration::from_secs(3 * 60);

/// How long a pull nobody answered waits for the round that asks every peer
/// for it, before it is forgotten — a round runs every 30-120 s.
const RETRY_WAITS_AT_MOST: Duration = Duration::from_secs(5 * 60);

/// One manifest version: the model and its `manifest_hash`.
pub type Version = (ModelId, [u8; 32]);

/// What a digest of `manifests` says: one version per manifest.
pub fn digest_of(manifests: &[ModelManifest]) -> ManifestDigest {
    ManifestDigest {
        versions: manifests
            .iter()
            .map(|m| ManifestVersion {
                model_id: m.id.clone(),
                manifest_hash: m.manifest_hash,
                context_length: m.context_length,
            })
            .collect(),
    }
}

/// One version being pulled.
struct Pull {
    since: Instant,
    /// The peer whose digest it was first asked of — what that peer's share of
    /// the pulls is counted by.
    asked_of: NodeId,
    /// False once the pull could not be sent: its retry is then owed at once.
    sent: bool,
    /// Already asked of every peer that takes digests: once more unanswered,
    /// it is over.
    retried: bool,
}

impl Pull {
    fn is_fresh(&self, now: Instant) -> bool {
        self.sent && now.saturating_duration_since(self.since) < PULL_PATIENCE
    }
}

/// What one asking peer has been answered.
struct Answered {
    window_start: Instant,
    in_window: usize,
    /// Each model answered, and when — not answered again within
    /// [`ANSWER_AGAIN_AFTER`].
    models: HashMap<ModelId, Instant>,
    /// When a new connection last cleared `models` — at most once per
    /// [`ANSWER_AGAIN_AFTER`] (`note_new_connection_at`).
    forgotten_at: Option<Instant>,
}

/// A connected peer's newest connection.
#[derive(Clone, Copy)]
struct Connection {
    number: u64,
    since: Instant,
}

/// The receiving half of announce-then-pull: which versions this node has
/// verified, which it is pulling, and what it has answered whom.
///
/// Written only by the dispatcher (`note_seen` for every VERIFIED manifest
/// whatever its transport, `versions_to_pull` for a digest, `answers_allowed`
/// for a pull) and by the health monitor's round (`retry_overdue_pulls`, then
/// `forget_stale`).
#[derive(Default)]
pub struct ManifestSync {
    /// `(model, version) → (the declared context last seen with it, when)`.
    seen: DashMap<Version, (Option<u32>, Instant)>,
    /// Versions a pull has asked for that have not arrived. Keyed by VERSION,
    /// not model: two peers holding two versions of one model are both
    /// information (`model::canonical` decides between them), and an answer
    /// carrying another version than the one named does not end the pull.
    in_flight: DashMap<Version, Pull>,
    answered: DashMap<NodeId, Answered>,
    /// The newest connection each connected peer opened — its number, from
    /// one counter that only rises, so a peer back on a new connection is
    /// never mistaken for one still on the connection it was told on
    /// (`note_new_connection`, `connection_of`); and when, so a model its
    /// capability lists is awaited only while its catch-up can still be on
    /// its way (`catching_up`). Removed with the peer's LAST connection.
    connections: DashMap<NodeId, Connection>,
    next_connection: std::sync::atomic::AtomicU64,
}

impl ManifestSync {
    /// A verified manifest of `model` at version `manifest_hash` arrived, by
    /// any transport, declaring `context_length`. Ends the pull of exactly
    /// that version — whoever sent it.
    pub fn note_seen(&self, model: &ModelId, manifest_hash: [u8; 32], context_length: Option<u32>) {
        if model.0.len() > MAX_MODEL_ID_BYTES {
            return;
        }
        let now = Instant::now();
        let key = (model.clone(), manifest_hash);
        self.in_flight.remove(&key);
        if let Some(mut entry) = self.seen.get_mut(&key) {
            let (context, at) = entry.value_mut();
            if context_length.is_some() {
                *context = context_length;
            }
            *at = now;
            return;
        }
        if self.seen.len() >= MAX_SEEN_VERSIONS {
            self.seen.retain(|_, (_, at)| at.elapsed() < SEEN_FOR);
            if self.seen.len() >= MAX_SEEN_VERSIONS {
                return;
            }
        }
        self.seen.insert(key, (context_length, now));
    }

    /// Which versions of `digest` (sent by `from`) to pull, given the version
    /// this node holds of each model (`held`: its hash and declared context,
    /// or `None` for a model it does not hold). Marks each one returned as
    /// being pulled from `from`.
    ///
    /// A version is pulled when it has not been seen — or when the sender's
    /// copy declares a context that this node's copy, and every copy of the
    /// version it has seen, do not (the context lies outside the hash, #189).
    /// A version already being pulled is never asked again by a digest, from
    /// anyone: a peer that named it first and does not answer is outlasted by
    /// the retry (`retry_overdue_pulls`), and cannot hold it by naming it again
    /// (security review, 2026-10-17). Nothing new is pulled past
    /// [`MAX_PULLS_IN_FLIGHT`], nor past [`ANSWERS_PER_WINDOW`] from one peer.
    pub fn versions_to_pull(
        &self,
        digest: &ManifestDigest,
        from: &NodeId,
        held: impl Fn(&ModelId) -> Option<([u8; 32], Option<u32>)>,
    ) -> Vec<Version> {
        let now = Instant::now();
        if self.in_flight.len() >= MAX_PULLS_IN_FLIGHT {
            self.forget_unanswerable(now);
        }
        let mut room = MAX_PULLS_IN_FLIGHT.saturating_sub(self.in_flight.len());
        let mut from_room = ANSWERS_PER_WINDOW.saturating_sub(
            self.in_flight
                .iter()
                .filter(|entry| entry.asked_of == *from && !entry.retried && entry.is_fresh(now))
                .count(),
        );
        let mut asked = HashSet::new();
        let mut pull = Vec::new();
        for version in digest.versions.iter().take(MAX_DIGEST_VERSIONS) {
            if pull.len() >= MAX_PULL_MODELS || from_room == 0 {
                break;
            }
            let model = &version.model_id;
            if model.0.len() > MAX_MODEL_ID_BYTES || !asked.insert(model) {
                continue;
            }
            let declares_more = |context: Option<u32>| {
                version.context_length.is_some() && context != version.context_length
            };
            let wanted = match held(model) {
                Some((hash, context)) if hash == version.manifest_hash => declares_more(context),
                _ => true,
            } && !self
                .seen
                .get(&(model.clone(), version.manifest_hash))
                .is_some_and(|entry| {
                    let (context, at) = *entry;
                    at.elapsed() < SEEN_FOR && !declares_more(context)
                });
            if !wanted {
                continue;
            }
            let fresh_pull = Pull {
                since: now,
                asked_of: from.clone(),
                sent: true,
                retried: false,
            };
            match self.in_flight.entry((model.clone(), version.manifest_hash)) {
                dashmap::Entry::Occupied(mut entry) => {
                    let pulling = entry.get();
                    if !(pulling.retried && !pulling.is_fresh(now)) {
                        continue;
                    }
                    // Asked of everyone and still unanswered: over, so a new
                    // digest naming it may start it again.
                    entry.insert(fresh_pull);
                }
                dashmap::Entry::Vacant(entry) => {
                    if room == 0 {
                        continue;
                    }
                    room -= 1;
                    entry.insert(fresh_pull);
                }
            }
            from_room -= 1;
            pull.push((model.clone(), version.manifest_hash));
        }
        pull
    }

    /// A pull that could not be sent: its retry is owed at the next round.
    pub fn pull_not_sent(&self, versions: &[Version]) {
        for version in versions {
            if let Some(mut pulling) = self.in_flight.get_mut(version) {
                pulling.sent = false;
            }
        }
    }

    /// The versions whose pull nobody answered within [`PULL_PATIENCE`], each
    /// marked as now asked of every peer that takes digests — which the caller
    /// does. Without this a peer that named a version first and never
    /// answered — slow, gone or hostile — would leave it unpulled until the
    /// next digest naming it, up to `MANIFEST_QUIET_WINDOW` later, where the
    /// old full catch-up always delivered. Asking every peer costs a pull each
    /// (~100 bytes); a peer holding no part of the model answers nothing.
    pub fn retry_overdue_pulls(&self, now: Instant) -> Vec<Version> {
        let mut overdue = Vec::new();
        for mut entry in self.in_flight.iter_mut() {
            let (version, pulling) = entry.pair_mut();
            if pulling.retried || pulling.is_fresh(now) {
                continue;
            }
            pulling.retried = true;
            pulling.since = now;
            pulling.sent = true;
            overdue.push(version.clone());
        }
        overdue
    }

    /// Of the `models` `peer` asks for, which it may be sent now: none it was
    /// sent within [`ANSWER_AGAIN_AFTER`], and no more than what is left of
    /// its [`ANSWERS_PER_WINDOW`] — which the answer uses up.
    pub fn answers_allowed(&self, peer: &NodeId, models: Vec<ModelId>) -> Vec<ModelId> {
        self.answers_allowed_at(peer, models, Instant::now())
    }

    fn answers_allowed_at(
        &self,
        peer: &NodeId,
        mut models: Vec<ModelId>,
        now: Instant,
    ) -> Vec<ModelId> {
        let mut entry = self
            .answered
            .entry(peer.clone())
            .or_insert_with(|| Answered {
                window_start: now,
                in_window: 0,
                models: HashMap::new(),
                forgotten_at: None,
            });
        let answered = entry.value_mut();
        if now.saturating_duration_since(answered.window_start) >= ANSWER_WINDOW {
            answered.window_start = now;
            answered.in_window = 0;
            answered
                .models
                .retain(|_, at| now.saturating_duration_since(*at) < ANSWER_AGAIN_AFTER);
        }
        models.retain(|model| {
            !answered
                .models
                .get(model)
                .is_some_and(|at| now.saturating_duration_since(*at) < ANSWER_AGAIN_AFTER)
        });
        models.truncate(ANSWERS_PER_WINDOW.saturating_sub(answered.in_window));
        answered.in_window += models.len();
        for model in &models {
            answered.models.insert(model.clone(), now);
        }
        models
    }

    /// `peer` opened a new connection — BitTorrent's rule (BEP 3: the bitfield
    /// goes to the peer that connected, over that connection), which the
    /// catch-up was written to follow and two memories outlived:
    ///
    /// - **It is caught up again.** The health monitor's round sends a digest
    ///   to a digest-taking peer whose connection number differs from the one
    ///   it was told on (`connection_of`). The round kept only a set of peers
    ///   told, cleared of a peer only when a round found it gone — and rounds
    ///   are 30-120 s apart, so a peer that restarted in seconds stayed "told"
    ///   and heard nothing until its repeat, up to `MANIFEST_QUIET_WINDOW`
    ///   (2 h) later.
    /// - **Its pulls are answered again**, keeping the window's count, so a
    ///   peer that reconnects is still answered at most
    ///   [`ANSWERS_PER_WINDOW`] a minute. [`ANSWER_AGAIN_AFTER`] promised "one
    ///   answer per model per peer, which is what the full catch-up it replaces
    ///   delivered per reconnect", but outlived the reconnect. The memory is
    ///   cleared at most once per [`ANSWER_AGAIN_AFTER`]: cleared at every
    ///   connection, a peer opening one more a minute — beside its first, even
    ///   — was answered the same models at the full budget, the ~75 KB/s that
    ///   rule exists to stop (security review, 2026-10-20). A restarted node's
    ///   pulls land within seconds of its connection, so it loses nothing.
    ///
    /// Together they left a node restarted on its own knowing none of the
    /// models it held no part of — a rig's requester, restarted under the same
    /// identity, heard nothing of the 7B in two starts of 2 min each, while a
    /// fresh identity heard it in 15-22 s (2026-10-20). Called by the network
    /// manager on EVERY connection established: a crashed peer's new connection
    /// can open before its old one times out, so "connected again" never fires
    /// for it.
    pub fn note_new_connection(&self, peer: &NodeId) {
        self.note_new_connection_at(peer, Instant::now());
    }

    pub(crate) fn note_new_connection_at(&self, peer: &NodeId, now: Instant) {
        let number = self
            .next_connection
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            + 1;
        self.connections
            .insert(peer.clone(), Connection { number, since: now });
        if let Some(mut answered) = self.answered.get_mut(peer) {
            let may_forget = answered
                .forgotten_at
                .is_none_or(|at| now.saturating_duration_since(at) >= ANSWER_AGAIN_AFTER);
            if may_forget {
                answered.models.clear();
                answered.forgotten_at = Some(now);
            }
        }
    }

    /// The number of `peer`'s newest connection, `None` once it has none.
    pub fn connection_of(&self, peer: &NodeId) -> Option<u64> {
        self.connections.get(peer).map(|c| c.number)
    }

    /// Could `peer`'s manifests still be on their way at `now` — is its newest
    /// connection younger than [`CATCH_UP_WITHIN`]? False for a peer with no
    /// connection on record.
    pub fn catching_up(&self, peer: &NodeId, now: Instant) -> bool {
        self.connections
            .get(peer)
            .is_some_and(|c| now.saturating_duration_since(c.since) < CATCH_UP_WITHIN)
    }

    /// `peer`'s last connection closed. Its next one gets a new number anyway;
    /// this keeps the map to the peers connected now.
    pub fn forget_connection(&self, peer: &NodeId) {
        self.connections.remove(peer);
    }

    /// Drop what can no longer change a decision. Called on the health
    /// monitor's round, after `retry_overdue_pulls`, so each map holds at most
    /// a window's worth.
    pub fn forget_stale(&self) {
        self.forget_stale_at(Instant::now());
    }

    fn forget_stale_at(&self, now: Instant) {
        self.seen
            .retain(|_, (_, at)| now.saturating_duration_since(*at) < SEEN_FOR);
        self.forget_unanswerable(now);
        // A record emptied by a reset is kept while the reset is recent: drop
        // it, and the next connection found no reset to wait out — a peer that
        // reconnected without pulling was answered three times in ten minutes
        // instead of twice (security review, 2026-10-20).
        self.answered.retain(|_, answered| {
            answered
                .models
                .retain(|_, at| now.saturating_duration_since(*at) < ANSWER_AGAIN_AFTER);
            !answered.models.is_empty()
                || now.saturating_duration_since(answered.window_start) < ANSWER_WINDOW
                || answered
                    .forgotten_at
                    .is_some_and(|at| now.saturating_duration_since(at) < ANSWER_AGAIN_AFTER)
        });
    }

    /// A pull asked of everyone and still unanswered is over; one still
    /// waiting for that round is kept at most [`RETRY_WAITS_AT_MOST`].
    fn forget_unanswerable(&self, now: Instant) {
        self.in_flight.retain(|_, pulling| {
            pulling.is_fresh(now)
                || (!pulling.retried
                    && now.saturating_duration_since(pulling.since) < RETRY_WAITS_AT_MOST)
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const PEER: NodeId = NodeId([1; 32]);

    fn version(model: &str, hash: u8, context_length: Option<u32>) -> ManifestVersion {
        ManifestVersion {
            model_id: ModelId(model.into()),
            manifest_hash: [hash; 32],
            context_length,
        }
    }

    fn digest(versions: Vec<ManifestVersion>) -> ManifestDigest {
        ManifestDigest { versions }
    }

    fn models(pulled: Vec<Version>) -> Vec<ModelId> {
        let mut models: Vec<ModelId> = pulled.into_iter().map(|(model, _)| model).collect();
        models.sort_by(|a, b| a.0.cmp(&b.0));
        models
    }

    /// **A digest is answered with a pull only for what this node lacks.** A
    /// model it does not hold, and a version it has not seen, are pulled; the
    /// version it holds is not, nor one it has already verified (a version is
    /// content-addressed: taken in once, it has nothing more to say). A copy
    /// whose declared context differs from the one held is pulled — the
    /// context lies outside the hash, so one peer's wrong figure would
    /// otherwise stand until a gossip repeat.
    #[test]
    fn a_digest_pulls_only_what_this_node_lacks() {
        let sync = ManifestSync::default();
        let held = |m: &ModelId| match m.0.as_str() {
            "held" => Some(([1; 32], Some(4096))),
            "held-no-context" => Some(([2; 32], None)),
            "held-wrong-context" => Some(([6; 32], Some(1))),
            "held-other" => Some(([3; 32], Some(4096))),
            _ => None,
        };
        let d = digest(vec![
            version("held", 1, Some(4096)),
            version("held-no-context", 2, Some(4096)),
            version("held-wrong-context", 6, Some(4096)),
            version("held-other", 4, Some(4096)),
            version("absent", 5, None),
        ]);
        assert_eq!(
            models(sync.versions_to_pull(&d, &PEER, held)),
            [
                "absent",
                "held-no-context",
                "held-other",
                "held-wrong-context"
            ]
            .map(|m| ModelId(m.into()))
        );
        let without_context = digest(vec![version("held-wrong-context", 6, None)]);
        assert!(
            ManifestSync::default()
                .versions_to_pull(&without_context, &PEER, held)
                .is_empty(),
            "a copy declaring nothing tells this node nothing new"
        );

        // Each answered: nothing more to pull.
        let sync = ManifestSync::default();
        sync.note_seen(&ModelId("held-no-context".into()), [2; 32], Some(4096));
        sync.note_seen(&ModelId("held-wrong-context".into()), [6; 32], Some(4096));
        sync.note_seen(&ModelId("held-other".into()), [4; 32], Some(4096));
        sync.note_seen(&ModelId("absent".into()), [5; 32], None);
        assert!(sync.versions_to_pull(&d, &PEER, held).is_empty());

        // Seen WITHOUT the context: still pulled from a copy that has it.
        let sync = ManifestSync::default();
        sync.note_seen(&ModelId("held-no-context".into()), [2; 32], None);
        assert_eq!(
            sync.versions_to_pull(
                &digest(vec![version("held-no-context", 2, Some(4096))]),
                &PEER,
                held
            ),
            [(ModelId("held-no-context".into()), [2; 32])]
        );
    }

    /// **The second peer to name a version is not asked for it while the
    /// first pull is out** — the copies a newcomer would otherwise pull, one
    /// from each peer it connects to. Only that VERSION ends the pull: an
    /// answer carrying another is not the one asked for.
    #[test]
    fn a_version_being_pulled_is_not_asked_of_the_next_peer() {
        let sync = ManifestSync::default();
        let m = ModelId("m".into());
        let d = digest(vec![version("m", 7, None)]);
        assert_eq!(
            sync.versions_to_pull(&d, &PEER, |_| None),
            [(m.clone(), [7; 32])]
        );
        assert!(sync.versions_to_pull(&d, &PEER, |_| None).is_empty());
        assert!(sync
            .versions_to_pull(&d, &NodeId([2; 32]), |_| None)
            .is_empty());
        sync.note_seen(&m, [8; 32], None);
        assert!(
            sync.versions_to_pull(&d, &PEER, |_| None).is_empty(),
            "another version arriving does not end the pull of this one"
        );
        sync.note_seen(&m, [7; 32], None);
        assert!(sync.in_flight.is_empty());
    }

    /// **A pull nobody answers is asked of every peer once its patience is
    /// over** — and a peer that named the version first and stays silent
    /// cannot hold the pull by naming it again, however often (the security
    /// review's eclipse: a hostile peer naming an honest holder's version
    /// ahead of it). Asked of everyone and still unanswered, the pull is over
    /// and the next digest naming the version may start it again.
    #[test]
    fn an_unanswered_pull_is_asked_of_every_peer_and_cannot_be_held() {
        let sync = ManifestSync::default();
        let silent = NodeId([9; 32]);
        let v = (ModelId("m".into()), [7; 32]);
        let d = digest(vec![version("m", 7, None)]);
        assert_eq!(
            sync.versions_to_pull(&d, &silent, |_| None),
            std::slice::from_ref(&v)
        );
        let now = Instant::now();
        assert!(
            sync.retry_overdue_pulls(now).is_empty(),
            "the first peer still has its patience"
        );
        // Past its patience the silent peer names it again: not re-asked.
        sync.in_flight.get_mut(&v).unwrap().sent = false;
        assert!(sync.versions_to_pull(&d, &silent, |_| None).is_empty());
        assert!(sync
            .versions_to_pull(&d, &NodeId([2; 32]), |_| None)
            .is_empty());
        assert_eq!(
            sync.retry_overdue_pulls(now),
            std::slice::from_ref(&v),
            "asked of everyone"
        );
        assert!(sync.retry_overdue_pulls(now).is_empty(), "once");
        sync.forget_stale();
        assert_eq!(sync.in_flight.len(), 1, "everyone has its patience now");
        sync.in_flight.get_mut(&v).unwrap().sent = false;
        assert_eq!(
            sync.versions_to_pull(&d, &NodeId([2; 32]), |_| None),
            std::slice::from_ref(&v),
            "over: a new digest may start it again"
        );
        sync.in_flight.get_mut(&v).unwrap().sent = false;
        sync.in_flight.get_mut(&v).unwrap().retried = true;
        sync.forget_stale();
        assert!(
            sync.in_flight.is_empty(),
            "asked of everyone, unanswered: over"
        );
    }

    /// A pull that could not be sent is owed its retry at once, not after its
    /// patience.
    #[test]
    fn a_pull_not_sent_is_retried_at_the_next_round() {
        let sync = ManifestSync::default();
        let pulled = sync.versions_to_pull(&digest(vec![version("m", 7, None)]), &PEER, |_| None);
        sync.pull_not_sent(&pulled);
        assert_eq!(sync.retry_overdue_pulls(Instant::now()), pulled);
    }

    /// A hostile digest names one model many times and thousands of models:
    /// each model is pulled once, one pull asks for at most
    /// `MAX_PULL_MODELS`, and an id longer than any real one is not read.
    #[test]
    fn a_digest_is_bounded() {
        let sync = ManifestSync::default();
        let repeated = digest((0..10).map(|i| version("same", i, None)).collect());
        assert_eq!(sync.versions_to_pull(&repeated, &PEER, |_| None).len(), 1);
        let huge = digest(
            (0..5_000)
                .map(|i| version(&format!("m{i}"), 1, None))
                .collect(),
        );
        assert_eq!(
            sync.versions_to_pull(&huge, &PEER, |_| None).len(),
            MAX_PULL_MODELS
        );
        let long = "x".repeat(MAX_MODEL_ID_BYTES + 1);
        assert!(sync
            .versions_to_pull(
                &digest(vec![version(&long, 1, None)]),
                &NodeId([2; 32]),
                |_| None
            )
            .is_empty());
        sync.note_seen(&ModelId(long), [1; 32], None);
        assert!(sync.seen.is_empty());
    }

    /// **A peer naming made-up models cannot fill the pulls**: it holds at most
    /// what it would answer in a window, every peer together at most
    /// `MAX_PULLS_IN_FLIGHT`, and a peer that has used its share still has an
    /// honest neighbour's digest pulled.
    #[test]
    fn one_peer_cannot_fill_the_pulls() {
        let sync = ManifestSync::default();
        let made_up = |round: usize| {
            digest(
                (0..MAX_PULL_MODELS)
                    .map(|i| version(&format!("fake-{round}-{i}"), 1, None))
                    .collect(),
            )
        };
        let mut pulled = 0;
        for round in 0..100 {
            pulled += sync
                .versions_to_pull(&made_up(round), &PEER, |_| None)
                .len();
        }
        assert_eq!(pulled, ANSWERS_PER_WINDOW);
        let honest = NodeId([2; 32]);
        assert_eq!(
            sync.versions_to_pull(&digest(vec![version("real", 3, None)]), &honest, |_| None),
            [(ModelId("real".into()), [3; 32])]
        );

        let sync = ManifestSync::default();
        let mut pulled = 0;
        for peer in 0..=255u8 {
            for round in 0..2 {
                let d = digest(
                    (0..MAX_PULL_MODELS)
                        .map(|i| version(&format!("fake-{peer}-{round}-{i}"), 1, None))
                        .collect(),
                );
                pulled += sync
                    .versions_to_pull(&d, &NodeId([peer; 32]), |_| None)
                    .len();
            }
        }
        assert_eq!(pulled, MAX_PULLS_IN_FLIGHT);
    }

    /// The seen versions stop growing at `MAX_SEEN_VERSIONS`; one already
    /// remembered is still updated, and a manifest arriving still ends its pull.
    #[test]
    fn the_seen_versions_are_bounded() {
        let sync = ManifestSync::default();
        for i in 0..MAX_SEEN_VERSIONS + 10 {
            sync.note_seen(&ModelId(format!("m{i}")), [1; 32], None);
        }
        assert_eq!(sync.seen.len(), MAX_SEEN_VERSIONS);
        sync.note_seen(&ModelId("m0".into()), [1; 32], Some(4096));
        assert_eq!(
            sync.seen.get(&(ModelId("m0".into()), [1; 32])).unwrap().0,
            Some(4096)
        );
        let d = digest(vec![version("late", 4, None)]);
        assert_eq!(sync.versions_to_pull(&d, &PEER, |_| None).len(), 1);
        sync.note_seen(&ModelId("late".into()), [4; 32], None);
        assert!(
            !sync
                .in_flight
                .contains_key(&(ModelId("late".into()), [4; 32])),
            "the pull is over even when the version cannot be remembered"
        );
    }

    /// **A peer is answered at most `ANSWERS_PER_WINDOW` manifests a window,
    /// and each model at most once per `ANSWER_AGAIN_AFTER`**: a pull costs a
    /// few bytes and its answer up to ~35 KB a model, so this is what stops a
    /// stranger making this node upload without end — including by pulling
    /// the same few models again every window. Per peer, so one asker does not
    /// use up another's.
    #[test]
    fn answers_to_one_peer_are_bounded() {
        let sync = ManifestSync::default();
        let peer = NodeId([1; 32]);
        let asked = |range: std::ops::Range<usize>| -> Vec<ModelId> {
            range.map(|i| ModelId(format!("m{i}"))).collect()
        };
        let at = Instant::now();
        let allowed =
            |peer: &NodeId, range, now| sync.answers_allowed_at(peer, asked(range), now).len();
        assert_eq!(allowed(&peer, 0..100, at), 100);
        assert_eq!(allowed(&peer, 100..200, at), ANSWERS_PER_WINDOW - 100);
        assert_eq!(allowed(&peer, 200..201, at), 0);
        assert_eq!(allowed(&NodeId([2; 32]), 0..5, at), 5);

        let next_window = at + ANSWER_WINDOW;
        assert_eq!(
            allowed(&peer, 0..10, next_window),
            0,
            "answered minutes ago"
        );
        assert_eq!(
            allowed(&peer, 300..310, next_window),
            10,
            "the budget is back"
        );
        assert_eq!(allowed(&peer, 0..10, at + ANSWER_AGAIN_AFTER), 10);
    }

    /// A peer that restarted within `ANSWER_AGAIN_AFTER` comes back on a new
    /// connection and pulls the same models: it is answered again, and the
    /// window's count still holds it. Without the reset a restarted node
    /// learned none of the swarm's models for 10 minutes (2026-10-20). The
    /// reset comes at most once per `ANSWER_AGAIN_AFTER`, or one more
    /// connection a minute bought the same models again at the full budget
    /// (security review, 2026-10-20).
    #[test]
    fn a_peer_that_reconnects_is_answered_again_within_the_window_budget() {
        let sync = ManifestSync::default();
        let peer = NodeId([1; 32]);
        let other = NodeId([2; 32]);
        let asked = |range: std::ops::Range<usize>| -> Vec<ModelId> {
            range.map(|i| ModelId(format!("m{i}"))).collect()
        };
        let at = Instant::now();
        assert_eq!(sync.answers_allowed_at(&peer, asked(0..10), at).len(), 10);
        assert_eq!(sync.answers_allowed_at(&other, asked(0..10), at).len(), 10);
        assert_eq!(
            sync.answers_allowed_at(&peer, asked(0..10), at).len(),
            0,
            "the same connection is not answered twice"
        );

        assert_eq!(sync.connection_of(&peer), None);
        sync.note_new_connection_at(&peer, at);
        let first = sync.connection_of(&peer).expect("numbered");
        assert_eq!(
            sync.answers_allowed_at(&peer, asked(0..10), at).len(),
            10,
            "a new connection is a new catch-up"
        );
        assert_eq!(
            sync.answers_allowed_at(&other, asked(0..10), at).len(),
            0,
            "another peer's memory is untouched"
        );

        let soon = at + Duration::from_secs(1);
        sync.forget_connection(&peer);
        assert_eq!(sync.connection_of(&peer), None);
        sync.note_new_connection_at(&peer, soon);
        assert_ne!(
            sync.connection_of(&peer),
            Some(first),
            "back after a disconnect, it is on a connection it was never told on"
        );
        assert_eq!(
            sync.answers_allowed_at(&peer, asked(0..10), soon).len(),
            0,
            "one more connection soon after is not one more fresh memory"
        );
        assert_eq!(
            sync.answers_allowed_at(&peer, asked(10..ANSWERS_PER_WINDOW + 10), soon)
                .len(),
            ANSWERS_PER_WINDOW - 20,
            "reconnecting does not refill the window"
        );

        // Answered again a minute before the memory may be cleared again: the
        // next connection after that clears it.
        let late = at + ANSWER_AGAIN_AFTER.saturating_sub(Duration::from_secs(60));
        assert_eq!(
            sync.answers_allowed_at(&peer, asked(500..510), late).len(),
            10
        );
        sync.note_new_connection_at(&peer, at + ANSWER_AGAIN_AFTER);
        assert_eq!(
            sync.answers_allowed_at(&peer, asked(500..510), at + ANSWER_AGAIN_AFTER)
                .len(),
            10,
            "a connection after the pause is a new catch-up again"
        );
    }

    /// A record a reset emptied outlives the sweep while the reset is recent:
    /// dropped, the next connection found no reset to wait out, and a peer
    /// reconnecting without pulling bought a third answer of each model in
    /// ten minutes (the review's sequence, 2026-10-20).
    #[test]
    fn a_swept_answer_record_still_remembers_a_recent_reset() {
        let sync = ManifestSync::default();
        let peer = NodeId([1; 32]);
        let asked = || -> Vec<ModelId> { (0..10).map(|i| ModelId(format!("m{i}"))).collect() };
        let at = Instant::now();
        let secs = |s| at + Duration::from_secs(s);
        assert_eq!(sync.answers_allowed_at(&peer, asked(), at).len(), 10);
        sync.note_new_connection_at(&peer, secs(60));
        sync.forget_stale_at(secs(120));
        assert_eq!(
            sync.answers_allowed_at(&peer, asked(), secs(121)).len(),
            10,
            "the reset was a new catch-up"
        );
        sync.note_new_connection_at(&peer, secs(181));
        assert_eq!(
            sync.answers_allowed_at(&peer, asked(), secs(181)).len(),
            0,
            "a second reset within ANSWER_AGAIN_AFTER is refused, swept or not"
        );
    }

    /// A model a peer's capability lists is awaited only while that peer's
    /// catch-up could still be on its way — a connection younger than
    /// `CATCH_UP_WITHIN`; past it, a listed model it never sent is no reason
    /// to hold back a "not found" (security review, 2026-10-20).
    #[test]
    fn a_peers_manifests_are_awaited_only_while_its_catch_up_can_be_on_its_way() {
        let sync = ManifestSync::default();
        let peer = NodeId([1; 32]);
        let at = Instant::now();
        assert!(!sync.catching_up(&peer, at), "no connection on record");
        sync.note_new_connection_at(&peer, at);
        assert!(sync.catching_up(&peer, at + Duration::from_secs(30)));
        assert!(!sync.catching_up(&peer, at + CATCH_UP_WITHIN));
        sync.note_new_connection_at(&peer, at + CATCH_UP_WITHIN);
        assert!(
            sync.catching_up(&peer, at + CATCH_UP_WITHIN),
            "a new connection is a new catch-up"
        );
        sync.forget_connection(&peer);
        assert!(!sync.catching_up(&peer, at + CATCH_UP_WITHIN));
    }

    #[test]
    fn a_digest_names_each_manifests_version() {
        let manifest = |id: &str, hash: u8, context_length| ModelManifest {
            id: ModelId(id.into()),
            name: id.into(),
            architecture: crate::types::ModelArchitecture::Llama,
            num_layers: 2,
            num_params_billions: 0.0,
            quantization: crate::types::Quantization::Q4KM,
            total_size_bytes: 0,
            shard_count: 0,
            shards: Vec::new(),
            tokenizer_hash: [0; 32],
            manifest_hash: [hash; 32],
            publisher: NodeId([0; 32]),
            publish_date: chrono::Utc::now(),
            license: String::new(),
            mmproj: None,
            context_length,
        };
        let d = digest_of(&[manifest("a", 9, Some(4096)), manifest("b", 3, None)]);
        assert_eq!(
            d.versions,
            vec![version("a", 9, Some(4096)), version("b", 3, None)]
        );
    }
}
