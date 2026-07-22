use ed25519_dalek::VerifyingKey;

use crate::error::SwarmError;
use crate::identity::Identity;
#[cfg(test)]
use crate::pool::types::{BlindSignature, BlindedToken, BlindingFactor, UnblindedToken};
use crate::pool::types::{
    PoolAcceptance, PoolCreditForward, PoolId, PoolInvitation, PoolMembership, PoolRemoval,
    PoolState, ShardPin,
};
use crate::types::NodeId;

// Domain-separated BLAKE3 prefixes per the plan.
const PREFIX_POOL_CREATE: &[u8] = b"pool_create_v1";
const PREFIX_MEMBER_LEFT: &[u8] = b"pool_member_left_v1";
const PREFIX_INVITATION: &[u8] = b"pool_invitation_v1";
const PREFIX_ACCEPTANCE: &[u8] = b"pool_acceptance_v1";
const PREFIX_REMOVAL: &[u8] = b"pool_removal_v1";
const PREFIX_CREDIT_FORWARD: &[u8] = b"pool_credit_forward_v1";
const PREFIX_POOL_STATE_DIFF: &[u8] = b"pool_state_diff_v1";
const PREFIX_POOL_MODEL_AVAIL: &[u8] = b"pool_model_avail_v1";
#[cfg(test)]
const PREFIX_BLIND_INVITE: &[u8] = b"pool_blind_invite_v1";

/// BLAKE3 payload for pool creation (sign + verify).
pub(crate) fn pool_create_payload(
    owner_id: &NodeId,
    name: &str,
    created_at: &chrono::DateTime<chrono::Utc>,
) -> Vec<u8> {
    let mut h = blake3::Hasher::new();
    h.update(PREFIX_POOL_CREATE);
    h.update(&owner_id.0);
    h.update(name.as_bytes());
    h.update(created_at.to_rfc3339().as_bytes());
    h.finalize().as_bytes().to_vec()
}

/// R134: BLAKE3 payload for inter-pool model availability gossip.
/// Domain-separated and bound to the pool id + sorted model id list +
/// wire timestamp so it can't be replayed against a different pool or
/// with a tampered model list.
pub(crate) fn pool_model_availability_payload(
    pool_id: &NodeId,
    model_ids: &[crate::types::ModelId],
    timestamp_ms: u64,
) -> Vec<u8> {
    let mut sorted: Vec<&crate::types::ModelId> = model_ids.iter().collect();
    sorted.sort_by(|a, b| a.0.cmp(&b.0));
    let mut h = blake3::Hasher::new();
    h.update(PREFIX_POOL_MODEL_AVAIL);
    h.update(&pool_id.0);
    h.update(&(sorted.len() as u32).to_le_bytes());
    for id in sorted {
        h.update(id.0.as_bytes());
    }
    h.update(&timestamp_ms.to_le_bytes());
    h.finalize().as_bytes().to_vec()
}

/// R134: BLAKE3 payload for pool-state diff gossip (sign + verify). Binds
/// the pool id, generation transition, the post-apply state checksum, and
/// the wire timestamp — preventing replay across pools, across generations,
/// or with a swapped checksum.
pub(crate) fn pool_state_diff_payload(
    pool_id: &PoolId,
    parent_generation: u64,
    new_generation: u64,
    state_checksum: &[u8; 32],
    timestamp_ms: u64,
) -> Vec<u8> {
    let mut h = blake3::Hasher::new();
    h.update(PREFIX_POOL_STATE_DIFF);
    h.update(&pool_id.0);
    h.update(&parent_generation.to_le_bytes());
    h.update(&new_generation.to_le_bytes());
    h.update(state_checksum);
    h.update(&timestamp_ms.to_le_bytes());
    h.finalize().as_bytes().to_vec()
}

/// R134: checksum of `state` AS IF its generation were `gen_override`.
/// Used when comparing two states across a generation bump that hasn't
/// been written back yet — the membership/pin bytes carry the only
/// real "did anything change" signal, so we must hash with a consistent
/// generation field on both sides.
pub(crate) fn pool_state_checksum_at(state: &PoolState, gen_override: u64) -> [u8; 32] {
    let mut alt = state.clone();
    alt.generation = gen_override;
    pool_state_checksum(&alt)
}

/// R134: canonical checksum of a `PoolState`'s member set + key fields.
/// Receivers recompute this locally after applying a diff and reject any
/// diff that would produce a different state than the owner intended.
/// Order-independent — members are sorted by `node_id` before hashing.
pub(crate) fn pool_state_checksum(state: &PoolState) -> [u8; 32] {
    let mut members: Vec<&PoolMembership> = state.members.iter().collect();
    members.sort_by_key(|m| m.node_id.0);
    let mut h = blake3::Hasher::new();
    h.update(b"pool_state_checksum_v1");
    h.update(&state.pool_id.0);
    h.update(&state.generation.to_le_bytes());
    h.update(&state.total_lifetime_credits.to_le_bytes());
    h.update(&[state.member_credit_split_pct]);
    h.update(&(members.len() as u32).to_le_bytes());
    for m in members {
        h.update(&m.node_id.0);
        h.update(m.invitation_id.as_bytes());
    }
    h.update(&(state.shard_pins.len() as u32).to_le_bytes());
    let mut pins: Vec<&ShardPin> = state.shard_pins.iter().collect();
    pins.sort_by(|a, b| {
        a.model_id
            .cmp(&b.model_id)
            .then_with(|| a.target_node_id.0.cmp(&b.target_node_id.0))
            .then_with(|| a.shard_indices.cmp(&b.shard_indices))
    });
    for p in pins {
        h.update(p.model_id.as_bytes());
        h.update(&p.target_node_id.0);
        h.update(&(p.shard_indices.len() as u32).to_le_bytes());
        for idx in &p.shard_indices {
            h.update(&idx.to_le_bytes());
        }
    }
    *h.finalize().as_bytes()
}

/// BLAKE3 payload for member-left notice (sign + verify).
/// Includes a timestamp + nonce so replays can be rejected.
pub(crate) fn member_left_payload(
    pool_id: &PoolId,
    node_id: &NodeId,
    left_at: i64,
    nonce: &uuid::Uuid,
) -> Vec<u8> {
    let mut h = blake3::Hasher::new();
    h.update(PREFIX_MEMBER_LEFT);
    h.update(&pool_id.0);
    h.update(&node_id.0);
    h.update(&left_at.to_le_bytes());
    h.update(nonce.as_bytes());
    h.finalize().as_bytes().to_vec()
}

/// Create a pool invitation signed by the pool owner.
pub fn create_invitation(
    identity: &Identity,
    pool_id: &PoolId,
    invitee: &NodeId,
    ttl_hours: u32,
) -> PoolInvitation {
    let id = uuid::Uuid::new_v4();
    let now = chrono::Utc::now();
    let expires_at = now + chrono::Duration::hours(ttl_hours as i64);

    let payload = invitation_payload(&id, pool_id, invitee, &expires_at);
    let signature = identity.sign(&payload);

    tracing::debug!(
        invitation_id = %id,
        pool = %pool_id,
        invitee = %invitee,
        "DIAG: pool invitation created"
    );
    PoolInvitation {
        id,
        pool_id: pool_id.clone(),
        invitee_node_id: invitee.clone(),
        expires_at,
        owner_signature: signature,
        created_at: now,
    }
}

/// Verify an invitation's owner signature.
pub fn verify_invitation(
    invitation: &PoolInvitation,
    owner_key: &VerifyingKey,
) -> Result<(), SwarmError> {
    let payload = invitation_payload(
        &invitation.id,
        &invitation.pool_id,
        &invitation.invitee_node_id,
        &invitation.expires_at,
    );
    verify_sig(&invitation.owner_signature, &payload, owner_key)
}

/// Create an acceptance signed by the invitee.
pub fn create_acceptance(identity: &Identity, invitation: &PoolInvitation) -> PoolAcceptance {
    let now = chrono::Utc::now();
    let payload = acceptance_payload(
        &invitation.id,
        &invitation.pool_id,
        &invitation.invitee_node_id,
        &invitation.expires_at,
    );
    let signature = identity.sign(&payload);

    PoolAcceptance {
        invitation_id: invitation.id,
        pool_id: invitation.pool_id.clone(),
        invitee_node_id: invitation.invitee_node_id.clone(),
        invitee_signature: signature,
        accepted_at: now,
    }
}

/// Verify an acceptance's invitee signature against the invitation the
/// verifier itself holds.
///
/// `invitation_expires_at` MUST come from the verifier's own stored
/// invitation, never from the acceptance — the point is that the invitee
/// signed over a value the attacker doesn't get to pick.
pub fn verify_acceptance(
    acceptance: &PoolAcceptance,
    invitee_key: &VerifyingKey,
    invitation_expires_at: &chrono::DateTime<chrono::Utc>,
) -> Result<(), SwarmError> {
    let payload = acceptance_payload(
        &acceptance.invitation_id,
        &acceptance.pool_id,
        &acceptance.invitee_node_id,
        invitation_expires_at,
    );
    verify_sig(&acceptance.invitee_signature, &payload, invitee_key)
}

/// Create a removal notice signed by the pool owner.
pub fn create_removal(identity: &Identity, pool_id: &PoolId, removed_node: &NodeId) -> PoolRemoval {
    let now = chrono::Utc::now();
    let removal_id = uuid::Uuid::new_v4();
    let payload = removal_payload(pool_id, removed_node, &now, &removal_id);
    let signature = identity.sign(&payload);

    PoolRemoval {
        pool_id: pool_id.clone(),
        removed_node_id: removed_node.clone(),
        owner_signature: signature,
        removed_at: now,
        removal_id,
    }
}

/// Verify a removal notice's owner signature.
pub fn verify_removal(removal: &PoolRemoval, owner_key: &VerifyingKey) -> Result<(), SwarmError> {
    let payload = removal_payload(
        &removal.pool_id,
        &removal.removed_node_id,
        &removal.removed_at,
        &removal.removal_id,
    );
    verify_sig(&removal.owner_signature, &payload, owner_key)
}

/// Create a credit forward transaction signed by the member (first signature).
pub fn create_credit_forward(
    identity: &Identity,
    pool_id: &PoolId,
    from: &NodeId,
    to: &NodeId,
    amount: i64,
) -> PoolCreditForward {
    let id = uuid::Uuid::new_v4();
    let now = chrono::Utc::now();
    let payload = credit_forward_payload(&id, pool_id, from, to, amount, &now);
    let signature = identity.sign(&payload);

    PoolCreditForward {
        id,
        pool_id: pool_id.clone(),
        from_node_id: from.clone(),
        to_node_id: to.clone(),
        amount,
        member_signature: signature,
        owner_signature: Vec::new(),
        timestamp: now,
    }
}

/// Co-sign a credit forward as the pool owner (second signature).
pub fn cosign_credit_forward(
    identity: &Identity,
    forward: &mut PoolCreditForward,
    member_key: &VerifyingKey,
) -> Result<(), SwarmError> {
    // Verify member's signature first
    let payload = credit_forward_payload(
        &forward.id,
        &forward.pool_id,
        &forward.from_node_id,
        &forward.to_node_id,
        forward.amount,
        &forward.timestamp,
    );
    verify_sig(&forward.member_signature, &payload, member_key)?;

    // Owner co-signs — commits to the member's signature for cryptographic separation
    let mut cosign_payload = Vec::with_capacity(payload.len() + forward.member_signature.len());
    cosign_payload.extend_from_slice(&payload);
    cosign_payload.extend_from_slice(&forward.member_signature);
    forward.owner_signature = identity.sign(&cosign_payload);
    Ok(())
}

// ---- Privacy-preserving blind invitations ----

/// Step 1: Invitee generates a blinding factor and computes a blinded token.
/// The blinded token is sent to the pool creator without revealing the invitee's identity.
#[cfg(test)]
fn blind_invite(pool_id: &PoolId, ttl_hours: u32) -> (uuid::Uuid, BlindingFactor, BlindedToken) {
    let invitation_id = uuid::Uuid::new_v4();
    // SEC: use OsRng directly. `rand::random` resolves to `thread_rng()` —
    // a ChaCha12Rng seeded once from OsRng. Every other crypto-material
    // generation in this codebase uses OsRng explicitly; keep this site
    // consistent so a future promotion to non-test never accidentally
    // weakens entropy.
    let mut factor_bytes = [0u8; 32];
    rand::RngCore::fill_bytes(&mut rand::rngs::OsRng, &mut factor_bytes);
    let blinding_factor = BlindingFactor(factor_bytes);
    let commitment = compute_blind_commitment(&invitation_id, &blinding_factor);
    let expires_at = chrono::Utc::now() + chrono::Duration::hours(ttl_hours as i64);

    let token = BlindedToken {
        commitment,
        pool_id: pool_id.clone(),
        expires_at,
    };

    (invitation_id, blinding_factor, token)
}

/// Step 2: Pool creator signs the blinded token without seeing the real invitation identity.
/// The signature now covers (commitment, pool_id, expires_at) — binding expiry cryptographically
/// to prevent indefinite replay of blind invitation tokens.
#[cfg(test)]
fn sign_blinded(identity: &Identity, blinded_token: &BlindedToken) -> BlindSignature {
    let payload = blind_token_payload(
        &blinded_token.commitment,
        &blinded_token.pool_id,
        blinded_token.expires_at,
    );
    let signature = identity.sign(&payload);

    BlindSignature {
        signature,
        commitment: blinded_token.commitment,
        pool_id: blinded_token.pool_id.clone(),
    }
}

/// Step 3: Invitee removes the blinding to produce a valid signed membership token.
#[cfg(test)]
fn unblind_token(
    invitation_id: uuid::Uuid,
    blinding_factor: BlindingFactor,
    blind_signature: BlindSignature,
    expires_at: chrono::DateTime<chrono::Utc>,
) -> UnblindedToken {
    UnblindedToken {
        invitation_id,
        blinding_factor,
        signature: blind_signature.signature,
        pool_id: blind_signature.pool_id,
        expires_at,
    }
}

/// Step 4: Anyone can verify that the unblinded token was signed by the pool creator.
/// Verifier recomputes the commitment from (invitation_id, blinding_factor), then checks
/// that the signature over (commitment, pool_id, expires_at) is valid.
///
/// SEC: All tokens require expiry to prevent permanent blind tokens.
#[cfg(test)]
fn verify_membership(token: &UnblindedToken, owner_key: &VerifyingKey) -> Result<(), SwarmError> {
    let commitment = compute_blind_commitment(&token.invitation_id, &token.blinding_factor);
    let payload = blind_token_payload(&commitment, &token.pool_id, token.expires_at);
    verify_sig(&token.signature, &payload, owner_key)
}

/// Compute the blind commitment: H(PREFIX || invitation_id || blinding_factor)
#[cfg(test)]
fn compute_blind_commitment(
    invitation_id: &uuid::Uuid,
    blinding_factor: &BlindingFactor,
) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(PREFIX_BLIND_INVITE);
    hasher.update(invitation_id.as_bytes());
    hasher.update(&blinding_factor.0);
    *hasher.finalize().as_bytes()
}

#[cfg(test)]
fn blind_token_payload(
    commitment: &[u8; 32],
    pool_id: &PoolId,
    expires_at: chrono::DateTime<chrono::Utc>,
) -> Vec<u8> {
    let mut hasher = blake3::Hasher::new();
    hasher.update(PREFIX_BLIND_INVITE);
    hasher.update(commitment);
    hasher.update(&pool_id.0);
    hasher.update(&expires_at.timestamp().to_le_bytes());
    hasher.finalize().as_bytes().to_vec()
}

// ---- Payload builders ----

fn invitation_payload(
    id: &uuid::Uuid,
    pool_id: &PoolId,
    invitee: &NodeId,
    expires_at: &chrono::DateTime<chrono::Utc>,
) -> Vec<u8> {
    let mut hasher = blake3::Hasher::new();
    hasher.update(PREFIX_INVITATION);
    hasher.update(id.as_bytes());
    hasher.update(&pool_id.0);
    hasher.update(&invitee.0);
    hasher.update(expires_at.to_rfc3339().as_bytes());
    hasher.finalize().as_bytes().to_vec()
}

/// Signing payload for a pool acceptance.
///
/// `expires_at` is the *invitation's* expiry, carried into the acceptance
/// signature so the signature is implicitly time-bounded and cannot be
/// transplanted onto a different invitation record. Before R147 the payload
/// covered only `(invitation_id, pool_id, invitee)` — a captured acceptance
/// signature made no statement about when it was valid, and the only thing
/// preventing reuse was the owner consuming the invitation from
/// `pending_invitations`.
///
/// The verifier gets `expires_at` from its own stored copy of the invitation
/// rather than from the acceptance, so nothing new travels the wire and an
/// attacker cannot choose the value the signature is checked against.
pub(crate) fn acceptance_payload(
    invitation_id: &uuid::Uuid,
    pool_id: &PoolId,
    invitee: &NodeId,
    expires_at: &chrono::DateTime<chrono::Utc>,
) -> Vec<u8> {
    let mut hasher = blake3::Hasher::new();
    hasher.update(PREFIX_ACCEPTANCE);
    hasher.update(invitation_id.as_bytes());
    hasher.update(&pool_id.0);
    hasher.update(&invitee.0);
    hasher.update(expires_at.to_rfc3339().as_bytes());
    hasher.finalize().as_bytes().to_vec()
}

fn removal_payload(
    pool_id: &PoolId,
    removed_node: &NodeId,
    removed_at: &chrono::DateTime<chrono::Utc>,
    removal_id: &uuid::Uuid,
) -> Vec<u8> {
    let mut hasher = blake3::Hasher::new();
    hasher.update(PREFIX_REMOVAL);
    hasher.update(&pool_id.0);
    hasher.update(&removed_node.0);
    hasher.update(removed_at.to_rfc3339().as_bytes());
    hasher.update(removal_id.as_bytes());
    hasher.finalize().as_bytes().to_vec()
}

fn credit_forward_payload(
    id: &uuid::Uuid,
    pool_id: &PoolId,
    from: &NodeId,
    to: &NodeId,
    amount: i64,
    timestamp: &chrono::DateTime<chrono::Utc>,
) -> Vec<u8> {
    let mut hasher = blake3::Hasher::new();
    hasher.update(PREFIX_CREDIT_FORWARD);
    hasher.update(id.as_bytes());
    hasher.update(&pool_id.0);
    hasher.update(&from.0);
    hasher.update(&to.0);
    hasher.update(&amount.to_le_bytes());
    hasher.update(timestamp.to_rfc3339().as_bytes());
    hasher.finalize().as_bytes().to_vec()
}

fn verify_sig(sig_bytes: &[u8], payload: &[u8], key: &VerifyingKey) -> Result<(), SwarmError> {
    crate::crypto::verify_ed25519_sig(sig_bytes, payload, key)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::identity::Identity;

    #[test]
    fn invitation_create_and_verify() {
        let owner = Identity::generate();
        let invitee = Identity::generate();
        let pool_id = owner.node_id().clone();

        let invitation = create_invitation(&owner, &pool_id, invitee.node_id(), 24);

        assert!(verify_invitation(&invitation, &owner.verifying_key()).is_ok());
        // Wrong key should fail
        assert!(verify_invitation(&invitation, &invitee.verifying_key()).is_err());
    }

    #[test]
    fn acceptance_create_and_verify() {
        let owner = Identity::generate();
        let invitee = Identity::generate();
        let pool_id = owner.node_id().clone();

        let invitation = create_invitation(&owner, &pool_id, invitee.node_id(), 24);
        let acceptance = create_acceptance(&invitee, &invitation);

        assert!(verify_acceptance(
            &acceptance,
            &invitee.verifying_key(),
            &invitation.expires_at
        )
        .is_ok());
        // Wrong key should fail
        assert!(
            verify_acceptance(&acceptance, &owner.verifying_key(), &invitation.expires_at).is_err()
        );
    }

    #[test]
    fn removal_create_and_verify() {
        let owner = Identity::generate();
        let member = Identity::generate();
        let pool_id = owner.node_id().clone();

        let removal = create_removal(&owner, &pool_id, member.node_id());

        assert!(verify_removal(&removal, &owner.verifying_key()).is_ok());
        assert!(verify_removal(&removal, &member.verifying_key()).is_err());
    }

    #[test]
    fn credit_forward_create_and_cosign() {
        let owner = Identity::generate();
        let member = Identity::generate();
        let pool_id = owner.node_id().clone();

        let mut forward =
            create_credit_forward(&member, &pool_id, member.node_id(), owner.node_id(), 100);

        assert!(forward.owner_signature.is_empty());

        // Owner co-signs
        cosign_credit_forward(&owner, &mut forward, &member.verifying_key()).unwrap();
        assert!(!forward.owner_signature.is_empty());
    }

    #[test]
    fn credit_forward_rejects_wrong_member_key() {
        let owner = Identity::generate();
        let member = Identity::generate();
        let imposter = Identity::generate();
        let pool_id = owner.node_id().clone();

        let mut forward =
            create_credit_forward(&member, &pool_id, member.node_id(), owner.node_id(), 100);

        // Owner tries to co-sign but uses wrong member key
        assert!(cosign_credit_forward(&owner, &mut forward, &imposter.verifying_key()).is_err());
    }

    #[test]
    fn blind_invite_full_flow() {
        let owner = Identity::generate();
        let pool_id = owner.node_id().clone();

        // Step 1: Invitee generates blinded token (no identity revealed)
        let (invitation_id, blinding_factor, blinded_token) = blind_invite(&pool_id, 24);

        // Step 2: Pool creator signs without seeing who the invitee is
        let blind_sig = sign_blinded(&owner, &blinded_token);

        // Step 3: Invitee unblinds to get a valid membership token
        let membership_token = unblind_token(
            invitation_id,
            blinding_factor,
            blind_sig,
            blinded_token.expires_at,
        );

        // Step 4: Anyone can verify the token was signed by the pool creator
        assert!(verify_membership(&membership_token, &owner.verifying_key()).is_ok());
    }

    #[test]
    fn blind_invite_rejects_wrong_owner() {
        let owner = Identity::generate();
        let imposter = Identity::generate();
        let pool_id = owner.node_id().clone();

        let (invitation_id, blinding_factor, blinded_token) = blind_invite(&pool_id, 24);
        let blind_sig = sign_blinded(&owner, &blinded_token);
        let membership_token = unblind_token(
            invitation_id,
            blinding_factor,
            blind_sig,
            blinded_token.expires_at,
        );

        // Verification with wrong key should fail
        assert!(verify_membership(&membership_token, &imposter.verifying_key()).is_err());
    }

    #[test]
    fn blind_invite_different_factors_produce_different_tokens() {
        let owner = Identity::generate();
        let pool_id = owner.node_id().clone();

        let (_, _, token1) = blind_invite(&pool_id, 24);
        let (_, _, token2) = blind_invite(&pool_id, 24);

        // Two blind invites should have different commitments
        assert_ne!(token1.commitment, token2.commitment);
    }

    #[test]
    fn blind_invite_tampered_factor_fails() {
        let owner = Identity::generate();
        let pool_id = owner.node_id().clone();

        let (invitation_id, _blinding_factor, blinded_token) = blind_invite(&pool_id, 24);
        let blind_sig = sign_blinded(&owner, &blinded_token);

        // Unblind with a different blinding factor — verification should fail
        // because the recomputed commitment won't match
        let wrong_factor = super::BlindingFactor(rand::random::<[u8; 32]>());
        let bad_token = unblind_token(
            invitation_id,
            wrong_factor,
            blind_sig,
            blinded_token.expires_at,
        );

        assert!(verify_membership(&bad_token, &owner.verifying_key()).is_err());
    }

    /// R147: the acceptance signature binds the invitation's expiry, so a
    /// signature produced for one expiry must not verify against another.
    /// This is what stops a captured acceptance being transplanted onto a
    /// different invitation record.
    #[test]
    fn acceptance_signature_does_not_verify_against_a_different_expiry() {
        let owner = Identity::generate();
        let invitee = Identity::generate();
        let pool_id = owner.node_id().clone();
        let invitation = create_invitation(&owner, &pool_id, invitee.node_id(), 24);
        let acceptance = create_acceptance(&invitee, &invitation);

        // Correct expiry verifies.
        assert!(verify_acceptance(
            &acceptance,
            &invitee.verifying_key(),
            &invitation.expires_at
        )
        .is_ok());

        // An attacker-extended expiry does not.
        let extended = invitation.expires_at + chrono::Duration::hours(24);
        assert!(
            verify_acceptance(&acceptance, &invitee.verifying_key(), &extended).is_err(),
            "signature must not verify against an expiry the invitee never signed"
        );

        // Nor does an earlier one.
        let shortened = invitation.expires_at - chrono::Duration::hours(1);
        assert!(verify_acceptance(&acceptance, &invitee.verifying_key(), &shortened).is_err());
    }

    /// The payload must actually incorporate `expires_at` — a helper that
    /// silently ignored the argument would pass the round-trip test above only
    /// if the caller happened to pass the same value both times.
    #[test]
    fn acceptance_payload_changes_with_expiry() {
        let id = uuid::Uuid::new_v4();
        let pool_id = NodeId([1u8; 32]);
        let invitee = NodeId([2u8; 32]);
        let t1 = chrono::Utc::now();
        let t2 = t1 + chrono::Duration::seconds(1);

        let p1 = acceptance_payload(&id, &pool_id, &invitee, &t1);
        let p2 = acceptance_payload(&id, &pool_id, &invitee, &t2);
        assert_ne!(p1, p2);
        assert_eq!(p1, acceptance_payload(&id, &pool_id, &invitee, &t1));
    }

    #[test]
    fn invitation_payload_is_deterministic() {
        let id = uuid::Uuid::new_v4();
        let pool_id = NodeId([1u8; 32]);
        let invitee = NodeId([2u8; 32]);
        let expires = chrono::Utc::now();

        let p1 = invitation_payload(&id, &pool_id, &invitee, &expires);
        let p2 = invitation_payload(&id, &pool_id, &invitee, &expires);
        assert_eq!(p1, p2);
    }

    /// R134: pool model availability payload is deterministic AND
    /// order-independent across the model list — the helper sorts before
    /// hashing so two publishers that build the same list in different
    /// orders produce the same payload.
    #[test]
    fn pool_model_availability_order_independent_and_signed() {
        let pool_id = NodeId([7u8; 32]);
        let m1 = crate::types::ModelId("aaa".to_string());
        let m2 = crate::types::ModelId("bbb".to_string());
        let ts = 12345u64;
        let p1 = pool_model_availability_payload(&pool_id, &[m1.clone(), m2.clone()], ts);
        let p2 = pool_model_availability_payload(&pool_id, &[m2.clone(), m1.clone()], ts);
        assert_eq!(p1, p2);

        // Different timestamp → different payload.
        let p3 = pool_model_availability_payload(&pool_id, &[m1.clone(), m2.clone()], ts + 1);
        assert_ne!(p1, p3);

        // Sign and verify roundtrip.
        let owner = Identity::generate();
        let pool_id = owner.node_id().clone();
        let payload = pool_model_availability_payload(&pool_id, std::slice::from_ref(&m1), ts);
        let sig = owner.sign(&payload);
        let key = owner.verifying_key();
        use ed25519_dalek::Verifier;
        let sig_bytes: [u8; 64] = sig.as_slice().try_into().unwrap();
        let signature = ed25519_dalek::Signature::from_bytes(&sig_bytes);
        assert!(key.verify(&payload, &signature).is_ok());

        // Tampered model list fails.
        let tampered = pool_model_availability_payload(
            &pool_id,
            &[m1, crate::types::ModelId("evil".to_string())],
            ts,
        );
        assert!(key.verify(&tampered, &signature).is_err());
    }
}
