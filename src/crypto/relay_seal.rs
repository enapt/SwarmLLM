//! NETWORKING_PLAN Phase 1 — end-to-end sealing for the application-level
//! inference relay.
//!
//! A relay envelope carries a cleartext routing header (`relay_to`, `origin`,
//! `request_id`) that a relay forwards on, plus a `sealed` inner `SwarmMessage`
//! that is ephemeral-sealed for `relay_to`'s X25519 key — derived from its
//! NodeId (Ed25519 pubkey) via the birational map, so NO prior session setup
//! with the target is required. Only the target can open it; the relay is a
//! dumb pipe. The AAD binds `origin || relay_to || request_id`, so a relay that
//! re-addresses or cross-wires an envelope fails Poly1305 at the recipient
//! before the inner message is ever parsed.
//!
//! **The seal alone proves nothing about who made it** (FUTURE_WORK #298). An
//! ephemeral seal is an anonymous sealed box — libsodium's `crypto_box_seal`
//! says of its own: the recipient "can verify the integrity of the message"
//! but "cannot verify the identity of the sender" — so anyone who knows the
//! target's NodeId can seal a body naming any `origin`. The origin therefore
//! SIGNS the body, inside the seal ([`RELAY_SIGNED`]): the signature covers
//! the kind, both NodeIds, the request id, a timestamp and the inner bytes.
//! Covering the recipient is Davis's repair for sign-then-encrypt ("Defective
//! Sign & Encrypt", USENIX 2001): a signed body cannot be re-sealed to a third
//! node as if written for it. Covering the kind binds the tensor frame's
//! cleartext `is_result`. It lives INSIDE the seal because a relay re-encodes
//! the envelope: a field outside it would be dropped by an older relay.
//!
//! A verbatim replay still opens, so the target also remembers what it took
//! ([`RelayReplayGuard`]): a body dated outside the window, or one already
//! taken within it, is refused.
//!
//! [`RELAY_SIGNED`]: crate::types::features::RELAY_SIGNED

use std::collections::HashMap;

use x25519_dalek::StaticSecret;

use crate::crypto::session::{ed25519_pubkey_to_x25519, ephemeral_open, ephemeral_seal};
use crate::error::SwarmError;
use crate::identity::Identity;
use crate::network::helpers::NetGroup;
use crate::types::{NodeId, RelayedEnvelope, SwarmMessage};

/// Upper bound on a sealed inner message (pre-seal plaintext). Inference relay
/// payloads are prompts and single tokens — small. This caps what a relay will
/// forward and what a recipient will open, so the channel can't be abused to
/// push large blobs through the anchor.
pub const MAX_RELAY_INNER_BYTES: usize = 256 * 1024;

/// Starts a body that carries its origin's signature. Neither older body can
/// start with it: a relayed message was JSON (`{`), a relayed tensor a frame
/// tag (`TENSOR_TAG_FORWARD` 0x01, `TENSOR_TAG_RESULT` 0x02).
const SIGNED_MAGIC: [u8; 4] = *b"SRS1";
/// `SIGNED_MAGIC` + `sent_at_ms` (u64 LE) + the Ed25519 signature.
const SIGNED_HEADER_LEN: usize = 4 + 8 + 64;
/// The digest's context string (BLAKE3 `derive_key`): a signature made for
/// anything else this node signs can never verify here.
const SIGNED_CONTEXT: &str = "swarmllm 2026-10-17 relayed body origin proof v1";

/// How old a relayed body may be, and how far ahead of this node's clock it
/// may be dated. One-sided, per gotcha #44: a symmetric window would let a
/// body dated ahead ride the whole past window as well. Wider than gossip's
/// 30 s ahead (`GOSSIP_SKEW_MS`) because a refusal here fails an inference,
/// not a gossip update; a relayed body is delivered within seconds, so the
/// past side covers an origin's clock running behind, and the memory the
/// window needs is proportional to it.
pub const RELAYED_MAX_AGE_MS: u64 = 5 * 60 * 1000;
pub const RELAYED_SKEW_MS: u64 = 60 * 1000;
/// The most relayed bodies remembered as taken, one origin's share of them,
/// and one NETWORK's — the network the delivering peer reaches this node from
/// (`network::helpers::NetGroup`: a public IPv4 /16, a public IPv6 /32, a
/// private host, a host name our bootstrap list names, or the circuits through
/// one relay, apart from that relay's own deliveries). A body leaves the memory when it is too
/// old to pass the window, so the memory is exactly the window; past any bound
/// a new body is refused (never an old one forgotten, which would open it to
/// replay). Keys cost nothing to mint, so the share is counted by something
/// that does cost: filling the memory takes delivering peers in 8 networks
/// (#299; keyed per peer id, 8 minted key pairs did it — security review of
/// #298). An origin streaming ~25 tokens or forwards a second holds ~7,500 in
/// the window; relays in one network carrying four such streams here, ~30,000.
pub const MAX_RELAYED_REMEMBERED: usize = 262_144;
pub const MAX_RELAYED_REMEMBERED_PER_ORIGIN: usize = 16_384;
pub const MAX_RELAYED_REMEMBERED_PER_NETWORK: usize = 32_768;
/// The memory is swept at most this often: a sweep walks all of it.
const SWEEP_EVERY_MS: u64 = 10_000;

/// What a relayed body is, bound into its signature.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RelayedKind {
    /// A `RelayedEnvelope`'s inner `SwarmMessage`.
    Message = 0,
    /// A `RelayedTensor` carrying a `LayerForward` (`is_result = false`).
    Forward = 1,
    /// A `RelayedTensor` carrying a `LayerResult` (`is_result = true`).
    Result = 2,
}

impl RelayedKind {
    pub fn of_tensor(is_result: bool) -> Self {
        if is_result {
            Self::Result
        } else {
            Self::Forward
        }
    }
}

/// An opened relayed body.
#[derive(Debug)]
pub enum Opened<T> {
    /// Signed by its origin: `inner` is what the origin sealed, at `sent_at_ms`
    /// by its clock. `digest` names the body for [`RelayReplayGuard`].
    Proven {
        inner: T,
        sent_at_ms: u64,
        digest: [u8; 32],
    },
    /// Sealed without a signature, as a node before [`RELAY_SIGNED`] seals —
    /// or as anyone at all can: its origin is a claim. Never dispatched, and
    /// never sent: this build relays only to a target that reads signatures.
    ///
    /// [`RELAY_SIGNED`]: crate::types::features::RELAY_SIGNED
    Unsigned,
}

/// AAD binding `origin || relay_to || request_id`. Any change to these fields
/// on the wire invalidates the seal.
fn relay_aad(origin: &NodeId, relay_to: &NodeId, request_id: &uuid::Uuid) -> Vec<u8> {
    let mut aad = Vec::with_capacity(32 + 32 + 16);
    aad.extend_from_slice(&origin.0);
    aad.extend_from_slice(&relay_to.0);
    aad.extend_from_slice(request_id.as_bytes());
    aad
}

/// What the origin signs: everything the target acts on.
fn origin_digest(
    kind: RelayedKind,
    origin: &NodeId,
    relay_to: &NodeId,
    request_id: &uuid::Uuid,
    sent_at_ms: u64,
    inner: &[u8],
) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new_derive_key(SIGNED_CONTEXT);
    hasher.update(&[kind as u8]);
    hasher.update(&origin.0);
    hasher.update(&relay_to.0);
    hasher.update(request_id.as_bytes());
    hasher.update(&sent_at_ms.to_le_bytes());
    hasher.update(inner);
    *hasher.finalize().as_bytes()
}

fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// The body to seal: `inner` behind its origin's signature.
fn signed_body(
    origin: &Identity,
    kind: RelayedKind,
    relay_to: &NodeId,
    request_id: &uuid::Uuid,
    inner: &[u8],
) -> Vec<u8> {
    let sent_at_ms = now_ms();
    let digest = origin_digest(
        kind,
        origin.node_id(),
        relay_to,
        request_id,
        sent_at_ms,
        inner,
    );
    let mut body = Vec::with_capacity(SIGNED_HEADER_LEN + inner.len());
    body.extend_from_slice(&SIGNED_MAGIC);
    body.extend_from_slice(&sent_at_ms.to_le_bytes());
    body.extend_from_slice(&origin.sign(&digest));
    body.extend_from_slice(inner);
    body
}

/// Split an opened body into its inner bytes and the proof of its origin.
fn prove_origin(
    kind: RelayedKind,
    origin: &NodeId,
    relay_to: &NodeId,
    request_id: &uuid::Uuid,
    body: Vec<u8>,
) -> Result<Opened<Vec<u8>>, SwarmError> {
    if body.len() < SIGNED_HEADER_LEN || body[..4] != SIGNED_MAGIC {
        return Ok(Opened::Unsigned);
    }
    let sent_at_ms = u64::from_le_bytes(body[4..12].try_into().expect("8 bytes"));
    let inner = &body[SIGNED_HEADER_LEN..];
    let digest = origin_digest(kind, origin, relay_to, request_id, sent_at_ms, inner);
    let origin_key = ed25519_dalek::VerifyingKey::from_bytes(&origin.0)
        .map_err(|_| SwarmError::InvalidSignature)?;
    super::verify_ed25519_sig(&body[12..SIGNED_HEADER_LEN], &digest, &origin_key)?;
    Ok(Opened::Proven {
        inner: inner.to_vec(),
        sent_at_ms,
        digest,
    })
}

/// Seal an inner `SwarmMessage` end-to-end for `relay_to`, signed by
/// `origin` (this node), producing an envelope a relay can route but not
/// read. Only for a `relay_to` advertising `RELAY_SIGNED`: an older node
/// cannot parse a signed body, and its reply would arrive unsigned.
pub fn seal_relayed_message(
    origin: &Identity,
    relay_to: NodeId,
    request_id: uuid::Uuid,
    inner: &SwarmMessage,
) -> Result<RelayedEnvelope, SwarmError> {
    let target_x = ed25519_pubkey_to_x25519(&relay_to.0)
        .ok_or_else(|| SwarmError::Encryption("relay target key not convertible".into()))?;
    let plaintext = serde_json::to_vec(inner)
        .map_err(|e| SwarmError::Encryption(format!("serialize relayed inner: {e}")))?;
    if plaintext.len() > MAX_RELAY_INNER_BYTES {
        return Err(SwarmError::Encryption(format!(
            "relayed inner too large: {} bytes (max {MAX_RELAY_INNER_BYTES})",
            plaintext.len()
        )));
    }
    let origin_id = origin.node_id().clone();
    let body = signed_body(
        origin,
        RelayedKind::Message,
        &relay_to,
        &request_id,
        &plaintext,
    );
    let aad = relay_aad(&origin_id, &relay_to, &request_id);
    let (ephemeral_pub, sealed) = ephemeral_seal(&target_x, &body, &aad)?;
    Ok(RelayedEnvelope {
        relay_to,
        origin: origin_id,
        request_id,
        ephemeral_pub,
        sealed,
    })
}

/// Open a relayed envelope addressed to this node. `local_secret` is this
/// node's X25519 static secret (`ed25519_to_x25519_secret` of its signing key).
/// The caller MUST have already checked `env.relay_to == local_node_id`, and
/// must dispatch only a [`Opened::Proven`] body, after [`RelayReplayGuard`].
pub fn open_relayed_message(
    local_secret: &StaticSecret,
    env: &RelayedEnvelope,
) -> Result<Opened<SwarmMessage>, SwarmError> {
    if env.sealed.len() > MAX_RELAY_INNER_BYTES + SIGNED_HEADER_LEN + 64 {
        return Err(SwarmError::Encryption("relayed envelope too large".into()));
    }
    let aad = relay_aad(&env.origin, &env.relay_to, &env.request_id);
    let body = ephemeral_open(local_secret, &env.ephemeral_pub, &env.sealed, &aad)?;
    match prove_origin(
        RelayedKind::Message,
        &env.origin,
        &env.relay_to,
        &env.request_id,
        body,
    )? {
        Opened::Proven {
            inner,
            sent_at_ms,
            digest,
        } => Ok(Opened::Proven {
            inner: serde_json::from_slice(&inner)
                .map_err(|e| SwarmError::Encryption(format!("deserialize relayed inner: {e}")))?,
            sent_at_ms,
            digest,
        }),
        Opened::Unsigned => Ok(Opened::Unsigned),
    }
}

/// Upper bound on a relayed tensor payload (an ephemeral-sealed, already-encoded
/// `LayerForward`/`LayerResult`). Tensors are large — a single-token decode
/// forward is tens of KB, a large-prompt prefill can be a few MB — so this is
/// generous, but still bounds what a relay will forward and a recipient open.
pub const MAX_RELAY_TENSOR_BYTES: usize = 32 * 1024 * 1024;

/// Seal an already-encoded tensor frame end-to-end for `relay_to` (binary
/// sibling of [`seal_relayed_message`], for the distributed-pipeline tensor
/// relay). Returns the ephemeral pubkey + sealed bytes. AAD binds
/// `origin‖relay_to‖request_id`, so a relay cannot re-address or cross-wire
/// it; the signature binds `kind` as well.
pub fn seal_relayed_tensor(
    origin: &Identity,
    relay_to: &NodeId,
    request_id: &uuid::Uuid,
    kind: RelayedKind,
    plaintext: &[u8],
) -> Result<([u8; 32], Vec<u8>), SwarmError> {
    if plaintext.len() > MAX_RELAY_TENSOR_BYTES {
        return Err(SwarmError::Encryption(format!(
            "relayed tensor too large: {} bytes (max {MAX_RELAY_TENSOR_BYTES})",
            plaintext.len()
        )));
    }
    let target_x = ed25519_pubkey_to_x25519(&relay_to.0)
        .ok_or_else(|| SwarmError::Encryption("relay target key not convertible".into()))?;
    let body = signed_body(origin, kind, relay_to, request_id, plaintext);
    let aad = relay_aad(origin.node_id(), relay_to, request_id);
    ephemeral_seal(&target_x, &body, &aad)
}

/// Open a relayed tensor addressed to this node (binary sibling of
/// [`open_relayed_message`]). `local_secret` is this node's X25519 static secret.
pub fn open_relayed_tensor(
    local_secret: &StaticSecret,
    origin: &NodeId,
    relay_to: &NodeId,
    request_id: &uuid::Uuid,
    kind: RelayedKind,
    ephemeral_pub: &[u8; 32],
    sealed: &[u8],
) -> Result<Opened<Vec<u8>>, SwarmError> {
    if sealed.len() > MAX_RELAY_TENSOR_BYTES + SIGNED_HEADER_LEN + 64 {
        return Err(SwarmError::Encryption("relayed tensor too large".into()));
    }
    let aad = relay_aad(origin, relay_to, request_id);
    let body = ephemeral_open(local_secret, ephemeral_pub, sealed, &aad)?;
    prove_origin(kind, origin, relay_to, request_id, body)
}

/// An envelope sealed as a node before `RELAY_SIGNED` seals it — or as
/// anyone can, naming any `origin`: what a target must never dispatch.
#[cfg(test)]
pub(crate) fn seal_unsigned_message(
    origin: &NodeId,
    relay_to: &NodeId,
    request_id: uuid::Uuid,
    inner: &SwarmMessage,
) -> RelayedEnvelope {
    let target_x = ed25519_pubkey_to_x25519(&relay_to.0).expect("a valid NodeId");
    let aad = relay_aad(origin, relay_to, &request_id);
    let (ephemeral_pub, sealed) =
        ephemeral_seal(&target_x, &serde_json::to_vec(inner).unwrap(), &aad).unwrap();
    RelayedEnvelope {
        relay_to: relay_to.clone(),
        origin: origin.clone(),
        request_id,
        ephemeral_pub,
        sealed,
    }
}

/// The tensor sibling of [`seal_unsigned_message`].
#[cfg(test)]
pub(crate) fn seal_unsigned_tensor(
    origin: &NodeId,
    relay_to: &NodeId,
    request_id: &uuid::Uuid,
    frame: &[u8],
) -> ([u8; 32], Vec<u8>) {
    let target_x = ed25519_pubkey_to_x25519(&relay_to.0).expect("a valid NodeId");
    ephemeral_seal(&target_x, frame, &relay_aad(origin, relay_to, request_id)).unwrap()
}

/// Why [`RelayReplayGuard::admit`] refused a body. The clock cases carry how
/// far off the origin's timestamp is, so a skewed clock reads as one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Refused {
    DatedAhead { by_ms: u64 },
    TooOld { by_ms: u64 },
    Replay,
    MemoryFull,
    OriginShareFull,
    NetworkShareFull,
}

impl std::fmt::Display for Refused {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DatedAhead { by_ms } => write!(
                f,
                "dated {} s ahead of this node's clock (at most {} s) — one of the two clocks is wrong",
                by_ms / 1000,
                RELAYED_SKEW_MS / 1000
            ),
            Self::TooOld { by_ms } => write!(
                f,
                "dated {} s ago (at most {} s) — a replay, or one of the two clocks is wrong",
                by_ms / 1000,
                RELAYED_MAX_AGE_MS / 1000
            ),
            Self::Replay => write!(f, "already taken: a replay"),
            Self::MemoryFull => write!(f, "the replay memory is full"),
            Self::OriginShareFull => write!(f, "this origin's share of the replay memory is full"),
            Self::NetworkShareFull => write!(
                f,
                "the share of the replay memory for the network this peer is in is full"
            ),
        }
    }
}

/// A short, unforgeable-enough name for a NodeId or a network, for the share
/// counts (finding a second value with the same 64 bits is 2^64 work).
fn short_key(bytes: &[u8]) -> u64 {
    u64::from_le_bytes(
        blake3::hash(bytes).as_bytes()[..8]
            .try_into()
            .expect("8 bytes"),
    )
}

/// One body taken.
struct Taken {
    sent_at_ms: u64,
    origin: u64,
    network: u64,
}

/// The bounds a [`RelayReplayGuard`] keeps: the constants above, smaller in a
/// test that fills one.
#[derive(Clone, Copy)]
struct Limits {
    total: usize,
    per_origin: usize,
    per_network: usize,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            total: MAX_RELAYED_REMEMBERED,
            per_origin: MAX_RELAYED_REMEMBERED_PER_ORIGIN,
            per_network: MAX_RELAYED_REMEMBERED_PER_NETWORK,
        }
    }
}

/// Which signed relayed bodies this node has taken, so a relay cannot deliver
/// one twice (the seal and the signature both survive a verbatim replay).
/// Owned by the network loop, which alone opens relayed bodies.
#[derive(Default)]
pub struct RelayReplayGuard {
    /// Every body taken within the window, by its digest.
    taken: HashMap<[u8; 32], Taken>,
    per_origin: HashMap<u64, usize>,
    per_network: HashMap<u64, usize>,
    swept_at_ms: u64,
    limits: Limits,
}

impl RelayReplayGuard {
    /// A guard that fills after `per_network` bodies from one network.
    #[cfg(test)]
    pub(crate) fn with_network_share(per_network: usize) -> Self {
        Self {
            limits: Limits {
                per_network,
                ..Limits::default()
            },
            ..Self::default()
        }
    }

    /// Take a proven body from `origin`, delivered by a peer in the network
    /// `delivered_from`, or say why not. Only a body this returns `Ok` for may
    /// be dispatched.
    pub fn admit(
        &mut self,
        origin: &NodeId,
        delivered_from: &NetGroup,
        digest: [u8; 32],
        sent_at_ms: u64,
    ) -> Result<(), Refused> {
        self.admit_at(
            origin,
            &delivered_from.key_bytes(),
            digest,
            sent_at_ms,
            now_ms(),
        )
    }

    fn admit_at(
        &mut self,
        origin: &NodeId,
        network: &[u8],
        digest: [u8; 32],
        sent_at_ms: u64,
        now_ms: u64,
    ) -> Result<(), Refused> {
        if sent_at_ms > now_ms.saturating_add(RELAYED_SKEW_MS) {
            return Err(Refused::DatedAhead {
                by_ms: sent_at_ms - now_ms,
            });
        }
        if now_ms.saturating_sub(sent_at_ms) > RELAYED_MAX_AGE_MS {
            return Err(Refused::TooOld {
                by_ms: now_ms - sent_at_ms,
            });
        }
        if self.taken.contains_key(&digest) {
            return Err(Refused::Replay);
        }
        if now_ms.saturating_sub(self.swept_at_ms) >= SWEEP_EVERY_MS {
            self.sweep(now_ms);
        }
        if self.taken.len() >= self.limits.total {
            return Err(Refused::MemoryFull);
        }
        let (origin, network) = (short_key(&origin.0), short_key(network));
        if self.per_origin.get(&origin).copied().unwrap_or(0) >= self.limits.per_origin {
            return Err(Refused::OriginShareFull);
        }
        if self.per_network.get(&network).copied().unwrap_or(0) >= self.limits.per_network {
            return Err(Refused::NetworkShareFull);
        }
        *self.per_origin.entry(origin).or_insert(0) += 1;
        *self.per_network.entry(network).or_insert(0) += 1;
        self.taken.insert(
            digest,
            Taken {
                sent_at_ms,
                origin,
                network,
            },
        );
        Ok(())
    }

    /// Forget every body the window would refuse anyway — and only those.
    fn sweep(&mut self, now_ms: u64) {
        self.swept_at_ms = now_ms;
        let (per_origin, per_network) = (&mut self.per_origin, &mut self.per_network);
        self.taken.retain(|_, taken| {
            let keep = now_ms.saturating_sub(taken.sent_at_ms) <= RELAYED_MAX_AGE_MS;
            if !keep {
                for (counts, key) in [
                    (&mut *per_origin, taken.origin),
                    (&mut *per_network, taken.network),
                ] {
                    if let Some(count) = counts.get_mut(&key) {
                        *count = count.saturating_sub(1);
                    }
                }
            }
            keep
        });
        per_origin.retain(|_, count| *count > 0);
        per_network.retain(|_, count| *count > 0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crypto::session::ed25519_to_x25519_secret;

    fn inner_msg() -> SwarmMessage {
        SwarmMessage::CancelInference(crate::types::CancelInference {
            request_id: uuid::Uuid::new_v4(),
        })
    }

    fn secret_of(identity: &Identity) -> StaticSecret {
        ed25519_to_x25519_secret(&identity.signing_key_bytes())
    }

    #[test]
    fn seal_open_roundtrip() {
        let origin = Identity::generate();
        let target = Identity::generate();

        let rid = uuid::Uuid::new_v4();
        let env =
            seal_relayed_message(&origin, target.node_id().clone(), rid, &inner_msg()).unwrap();

        assert_eq!(&env.relay_to, target.node_id());
        assert_eq!(&env.origin, origin.node_id());
        assert!(!env.sealed.is_empty());

        match open_relayed_message(&secret_of(&target), &env).unwrap() {
            Opened::Proven { inner, .. } => {
                assert!(matches!(inner, SwarmMessage::CancelInference(_)))
            }
            Opened::Unsigned => panic!("a signed envelope opens as proven"),
        }
    }

    #[test]
    fn wrong_target_cannot_open() {
        let origin = Identity::generate();
        let target = Identity::generate();
        let eavesdropper = Identity::generate();

        let env = seal_relayed_message(
            &origin,
            target.node_id().clone(),
            uuid::Uuid::new_v4(),
            &inner_msg(),
        )
        .unwrap();

        // A relay (or anyone who is not `relay_to`) cannot open the payload.
        assert!(open_relayed_message(&secret_of(&eavesdropper), &env).is_err());
    }

    #[test]
    fn tampered_routing_header_fails_seal() {
        let origin = Identity::generate();
        let target = Identity::generate();

        let mut env = seal_relayed_message(
            &origin,
            target.node_id().clone(),
            uuid::Uuid::new_v4(),
            &inner_msg(),
        )
        .unwrap();

        // A relay swapping the origin in the cleartext header breaks the AAD.
        env.origin = Identity::generate().node_id().clone();
        assert!(open_relayed_message(&secret_of(&target), &env).is_err());
    }

    /// FUTURE_WORK #298: the seal is anonymous, so anyone can seal a body
    /// naming any origin — what proves the origin is its signature. A forger
    /// sealing a fresh envelope (not tampering with one) either signs with its
    /// own key, which does not verify as the origin's, or does not sign, which
    /// opens as `Unsigned` and is never dispatched.
    #[test]
    fn a_forged_origin_is_never_proven() {
        let victim = Identity::generate();
        let forger = Identity::generate();
        let target = Identity::generate();
        let rid = uuid::Uuid::new_v4();

        let target_x = ed25519_pubkey_to_x25519(&target.node_id().0).unwrap();
        let aad = relay_aad(victim.node_id(), target.node_id(), &rid);
        let envelope_of = |body: &[u8]| {
            let (ephemeral_pub, sealed) = ephemeral_seal(&target_x, body, &aad).unwrap();
            RelayedEnvelope {
                relay_to: target.node_id().clone(),
                origin: victim.node_id().clone(),
                request_id: rid,
                ephemeral_pub,
                sealed,
            }
        };
        let body = serde_json::to_vec(&inner_msg()).unwrap();

        // Unsigned, as before RELAY_SIGNED: it opens — the seal cannot tell.
        assert!(matches!(
            open_relayed_message(&secret_of(&target), &envelope_of(&body)).unwrap(),
            Opened::Unsigned
        ));

        // Signed by the forger, naming the victim: the signature fails.
        let signed_by_forger =
            signed_body(&forger, RelayedKind::Message, target.node_id(), &rid, &body);
        assert!(
            open_relayed_message(&secret_of(&target), &envelope_of(&signed_by_forger)).is_err()
        );
    }

    /// A body signed for one node cannot be re-sealed to another as if written
    /// for it (Davis's surreptitious forwarding): the recipient is signed.
    #[test]
    fn a_signed_body_cannot_be_forwarded_to_another_node() {
        let origin = Identity::generate();
        let first = Identity::generate();
        let second = Identity::generate();
        let rid = uuid::Uuid::new_v4();

        let body = signed_body(
            &origin,
            RelayedKind::Message,
            first.node_id(),
            &rid,
            &serde_json::to_vec(&inner_msg()).unwrap(),
        );
        let second_x = ed25519_pubkey_to_x25519(&second.node_id().0).unwrap();
        let aad = relay_aad(origin.node_id(), second.node_id(), &rid);
        let (ephemeral_pub, sealed) = ephemeral_seal(&second_x, &body, &aad).unwrap();
        let env = RelayedEnvelope {
            relay_to: second.node_id().clone(),
            origin: origin.node_id().clone(),
            request_id: rid,
            ephemeral_pub,
            sealed,
        };
        assert!(open_relayed_message(&secret_of(&second), &env).is_err());
    }

    #[test]
    fn oversized_inner_rejected() {
        // A serialized inner larger than the cap is refused at seal time.
        let origin = Identity::generate();
        let target = Identity::generate();
        // ShardDownloadProgress is tiny; instead force the cap via a giant
        // nickname-like message. Use PeerExchangeResponse with huge strings.
        let huge = "x".repeat(MAX_RELAY_INNER_BYTES);
        let msg = SwarmMessage::PeerExchangeResponse(crate::types::PeerExchangeResponse {
            peers: vec![huge],
        });
        let err = seal_relayed_message(
            &origin,
            target.node_id().clone(),
            uuid::Uuid::new_v4(),
            &msg,
        );
        assert!(err.is_err());
    }

    #[test]
    fn tensor_seal_open_roundtrip() {
        let origin = Identity::generate();
        let target = Identity::generate();
        let rid = uuid::Uuid::new_v4();
        let tensor = vec![0xABu8; 100_000]; // ~100KB "encoded tensor"

        let (eph, sealed) = seal_relayed_tensor(
            &origin,
            target.node_id(),
            &rid,
            RelayedKind::Forward,
            &tensor,
        )
        .unwrap();
        assert!(!sealed.is_empty());

        let open = |secret: &StaticSecret, kind| {
            open_relayed_tensor(
                secret,
                origin.node_id(),
                target.node_id(),
                &rid,
                kind,
                &eph,
                &sealed,
            )
        };
        match open(&secret_of(&target), RelayedKind::Forward).unwrap() {
            Opened::Proven { inner, .. } => assert_eq!(inner, tensor),
            Opened::Unsigned => panic!("a signed tensor opens as proven"),
        }

        // A relay (wrong key) cannot open it.
        assert!(open(&secret_of(&Identity::generate()), RelayedKind::Forward).is_err());
        // A relay flipping the frame's cleartext `is_result` breaks the signature.
        assert!(open(&secret_of(&target), RelayedKind::Result).is_err());
    }

    /// The delivering connection's peer id bytes, for the guard tests.
    const RELAY: &[u8] = b"a relay";

    #[test]
    fn a_body_is_taken_once_and_only_inside_its_window() {
        let origin = NodeId([1; 32]);
        let mut guard = RelayReplayGuard::default();
        let now = 1_000_000_000;

        assert!(guard.admit_at(&origin, RELAY, [1; 32], now, now).is_ok());
        assert!(
            guard
                .admit_at(&origin, RELAY, [1; 32], now, now + 1)
                .is_err(),
            "a verbatim replay is refused"
        );
        assert!(guard
            .admit_at(&origin, RELAY, [2; 32], now - RELAYED_MAX_AGE_MS - 1, now)
            .is_err());
        assert!(guard
            .admit_at(&origin, RELAY, [3; 32], now + RELAYED_SKEW_MS + 1, now)
            .is_err());
        assert!(guard
            .admit_at(&origin, RELAY, [4; 32], now + RELAYED_SKEW_MS, now)
            .is_ok());

        // A body is forgotten only once the window would refuse it anyway.
        let later = now + RELAYED_MAX_AGE_MS + 1;
        assert!(guard
            .admit_at(&origin, RELAY, [5; 32], later, later)
            .is_ok());
        assert!(
            !guard.taken.contains_key(&[1; 32]),
            "swept: too old to pass"
        );
        assert_eq!(
            guard.admit_at(
                &origin,
                RELAY,
                [6; 32],
                later + RELAYED_SKEW_MS + 2_000,
                later
            ),
            Err(Refused::DatedAhead {
                by_ms: RELAYED_SKEW_MS + 2_000
            }),
            "a refusal says how far off the clock is"
        );
        assert!(
            guard.taken.contains_key(&[4; 32]),
            "kept: dated ahead, still inside the window"
        );
        assert!(guard.admit_at(&origin, RELAY, [1; 32], now, later).is_err());
    }

    #[test]
    fn one_origin_cannot_fill_the_replay_memory() {
        let flooder = NodeId([1; 32]);
        let other = NodeId([2; 32]);
        let mut guard = RelayReplayGuard::default();
        let now = 1_000_000_000;
        for i in 0..MAX_RELAYED_REMEMBERED_PER_ORIGIN as u64 {
            let digest = *blake3::hash(&i.to_le_bytes()).as_bytes();
            assert!(guard.admit_at(&flooder, RELAY, digest, now, now).is_ok());
        }
        assert!(guard
            .admit_at(&flooder, RELAY, [0xFF; 32], now, now)
            .is_err());
        assert!(guard.admit_at(&other, RELAY, [0xFE; 32], now, now).is_ok());
        assert_eq!(guard.taken.len(), MAX_RELAYED_REMEMBERED_PER_ORIGIN + 1);
    }

    /// Keys cost nothing, so one network presenting fresh origins is held to
    /// its own share, and another network is still served (security review of
    /// #298; per network, not per peer id, since #299).
    #[test]
    fn one_network_cannot_fill_the_replay_memory() {
        let mut guard = RelayReplayGuard::default();
        let now = 1_000_000_000;
        for i in 0..MAX_RELAYED_REMEMBERED_PER_NETWORK as u64 {
            let digest = *blake3::hash(&i.to_le_bytes()).as_bytes();
            let origin = NodeId(digest);
            assert!(guard.admit_at(&origin, b"sybil", digest, now, now).is_ok());
        }
        assert_eq!(
            guard.admit_at(&NodeId([0xEE; 32]), b"sybil", [0xFF; 32], now, now),
            Err(Refused::NetworkShareFull)
        );
        assert!(guard
            .admit_at(&NodeId([0xEE; 32]), b"honest relay", [0xFE; 32], now, now)
            .is_ok());
    }
}
