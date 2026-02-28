use ed25519_dalek::{Signature, Verifier, VerifyingKey};

use crate::error::SwarmError;
use crate::identity::Identity;
use crate::pool::types::{
    BlindSignature, BlindedToken, BlindingFactor, PoolAcceptance, PoolCreditForward, PoolId,
    PoolInvitation, PoolRemoval, UnblindedToken,
};
use crate::types::NodeId;

// Domain-separated BLAKE3 prefixes per the plan.
const PREFIX_INVITATION: &[u8] = b"pool_invitation_v1";
const PREFIX_ACCEPTANCE: &[u8] = b"pool_acceptance_v1";
const PREFIX_REMOVAL: &[u8] = b"pool_removal_v1";
const PREFIX_CREDIT_FORWARD: &[u8] = b"pool_credit_forward_v1";
const PREFIX_BLIND_INVITE: &[u8] = b"pool_blind_invite_v1";

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

/// Verify an acceptance's invitee signature.
pub fn verify_acceptance(
    acceptance: &PoolAcceptance,
    invitee_key: &VerifyingKey,
) -> Result<(), SwarmError> {
    let payload = acceptance_payload(
        &acceptance.invitation_id,
        &acceptance.pool_id,
        &acceptance.invitee_node_id,
    );
    verify_sig(&acceptance.invitee_signature, &payload, invitee_key)
}

/// Create a removal notice signed by the pool owner.
pub fn create_removal(identity: &Identity, pool_id: &PoolId, removed_node: &NodeId) -> PoolRemoval {
    let now = chrono::Utc::now();
    let payload = removal_payload(pool_id, removed_node, &now);
    let signature = identity.sign(&payload);

    PoolRemoval {
        pool_id: pool_id.clone(),
        removed_node_id: removed_node.clone(),
        owner_signature: signature,
        removed_at: now,
    }
}

/// Verify a removal notice's owner signature.
pub fn verify_removal(removal: &PoolRemoval, owner_key: &VerifyingKey) -> Result<(), SwarmError> {
    let payload = removal_payload(
        &removal.pool_id,
        &removal.removed_node_id,
        &removal.removed_at,
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

    // Owner co-signs
    forward.owner_signature = identity.sign(&payload);
    Ok(())
}

// ---- Privacy-preserving blind invitations ----

/// Step 1: Invitee generates a blinding factor and computes a blinded token.
/// The blinded token is sent to the pool creator without revealing the invitee's identity.
pub fn blind_invite(
    pool_id: &PoolId,
    ttl_hours: u32,
) -> (uuid::Uuid, BlindingFactor, BlindedToken) {
    let invitation_id = uuid::Uuid::new_v4();
    let blinding_factor = BlindingFactor(rand::random::<[u8; 32]>());
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
/// The signature covers (commitment, pool_id) — expiry is enforced as policy, not signed.
pub fn sign_blinded(
    identity: &Identity,
    blinded_token: &BlindedToken,
) -> BlindSignature {
    let payload = blind_token_payload_no_expiry(&blinded_token.commitment, &blinded_token.pool_id);
    let signature = identity.sign(&payload);

    BlindSignature {
        signature,
        commitment: blinded_token.commitment,
        pool_id: blinded_token.pool_id.clone(),
    }
}

/// Step 3: Invitee removes the blinding to produce a valid signed membership token.
pub fn unblind_token(
    invitation_id: uuid::Uuid,
    blinding_factor: BlindingFactor,
    blind_signature: BlindSignature,
) -> UnblindedToken {
    UnblindedToken {
        invitation_id,
        blinding_factor,
        signature: blind_signature.signature,
        pool_id: blind_signature.pool_id,
    }
}

/// Step 4: Anyone can verify that the unblinded token was signed by the pool creator.
/// Verifier recomputes the commitment from (invitation_id, blinding_factor), then checks
/// that the signature over (commitment, pool_id) is valid.
pub fn verify_membership(
    token: &UnblindedToken,
    owner_key: &VerifyingKey,
) -> Result<(), SwarmError> {
    let commitment = compute_blind_commitment(&token.invitation_id, &token.blinding_factor);
    let payload = blind_token_payload_no_expiry(&commitment, &token.pool_id);
    verify_sig(&token.signature, &payload, owner_key)
}

/// Compute the blind commitment: H(PREFIX || invitation_id || blinding_factor)
fn compute_blind_commitment(invitation_id: &uuid::Uuid, blinding_factor: &BlindingFactor) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(PREFIX_BLIND_INVITE);
    hasher.update(invitation_id.as_bytes());
    hasher.update(&blinding_factor.0);
    *hasher.finalize().as_bytes()
}

fn blind_token_payload_no_expiry(
    commitment: &[u8; 32],
    pool_id: &PoolId,
) -> Vec<u8> {
    let mut hasher = blake3::Hasher::new();
    hasher.update(PREFIX_BLIND_INVITE);
    hasher.update(commitment);
    hasher.update(&pool_id.0);
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

fn acceptance_payload(invitation_id: &uuid::Uuid, pool_id: &PoolId, invitee: &NodeId) -> Vec<u8> {
    let mut hasher = blake3::Hasher::new();
    hasher.update(PREFIX_ACCEPTANCE);
    hasher.update(invitation_id.as_bytes());
    hasher.update(&pool_id.0);
    hasher.update(&invitee.0);
    hasher.finalize().as_bytes().to_vec()
}

fn removal_payload(
    pool_id: &PoolId,
    removed_node: &NodeId,
    removed_at: &chrono::DateTime<chrono::Utc>,
) -> Vec<u8> {
    let mut hasher = blake3::Hasher::new();
    hasher.update(PREFIX_REMOVAL);
    hasher.update(&pool_id.0);
    hasher.update(&removed_node.0);
    hasher.update(removed_at.to_rfc3339().as_bytes());
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
    if sig_bytes.len() != 64 {
        return Err(SwarmError::InvalidSignature);
    }
    let sig = Signature::from_bytes(
        sig_bytes
            .try_into()
            .map_err(|_| SwarmError::Internal("Invalid signature length".into()))?,
    );
    key.verify(payload, &sig)
        .map_err(|_| SwarmError::InvalidSignature)
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

        assert!(verify_acceptance(&acceptance, &invitee.verifying_key()).is_ok());
        // Wrong key should fail
        assert!(verify_acceptance(&acceptance, &owner.verifying_key()).is_err());
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
        let membership_token = unblind_token(invitation_id, blinding_factor, blind_sig);

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
        let membership_token = unblind_token(invitation_id, blinding_factor, blind_sig);

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
        let bad_token = unblind_token(invitation_id, wrong_factor, blind_sig);

        assert!(verify_membership(&bad_token, &owner.verifying_key()).is_err());
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
}
