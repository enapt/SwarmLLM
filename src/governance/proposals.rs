use crate::error::SwarmError;
use crate::storage::db::Database;
use crate::types::{
    Blake3Hash, GovernanceParams, GovernanceRole, NodeId, Proposal, ProposalAmendment,
    ProposalCategory, ProposalPatch, ProposalStatus, ProposalStatusChange,
};

/// Database tree name for proposals.
const TREE_PROPOSALS: &str = "proposals";

/// Maximum title length in characters.
const MAX_TITLE_LEN: usize = 200;

/// Maximum body length in characters.
const MAX_BODY_LEN: usize = 50_000;

/// Maximum inline diff size in bytes (64KB).
const MAX_INLINE_DIFF_BYTES: u64 = 64 * 1024;

/// Parameters for creating a new proposal.
pub struct CreateProposalParams {
    pub author: NodeId,
    pub title: String,
    pub body: String,
    pub category: ProposalCategory,
    pub linked_issues: Vec<Blake3Hash>,
    pub patch: Option<ProposalPatch>,
    pub role: GovernanceRole,
}

/// Create a new proposal. Validates all fields and persists to the database.
///
/// Returns the proposal hash used as its identifier.
pub fn create_proposal(
    db: &Database,
    p: CreateProposalParams,
    params: &GovernanceParams,
) -> Result<Proposal, SwarmError> {
    let CreateProposalParams {
        author,
        title,
        body,
        category,
        linked_issues,
        patch,
        role,
    } = p;
    // Permission check
    if !role.can_create_proposals() {
        return Err(SwarmError::InsufficientPermissions {
            action: "create proposal".into(),
            required_role: "Contributor".into(),
        });
    }

    // Validate fields
    if title.is_empty() || title.len() > MAX_TITLE_LEN {
        return Err(SwarmError::Governance(format!(
            "Title must be 1-{MAX_TITLE_LEN} characters"
        )));
    }
    if body.is_empty() || body.len() > MAX_BODY_LEN {
        return Err(SwarmError::Governance(format!(
            "Body must be 1-{MAX_BODY_LEN} characters"
        )));
    }
    if linked_issues.len() > 20 {
        return Err(SwarmError::Governance(
            "Too many linked issues (max 20)".into(),
        ));
    }

    // Validate patch if provided
    if let Some(ref p) = patch {
        if let Some(ref inline) = p.inline_diff {
            if inline.len() as u64 > MAX_INLINE_DIFF_BYTES {
                return Err(SwarmError::Governance(
                    "Inline diff exceeds 64KB limit; use shard distribution".into(),
                ));
            }
        }
    }

    let now = chrono::Utc::now();
    let voting_days = voting_period_days(&category, params);
    let voting_deadline = now + chrono::Duration::days(voting_days as i64);

    // Compute content hash
    let hash = compute_proposal_hash(&author, &title, &body, &now);

    let proposal = Proposal {
        hash,
        author,
        title,
        body,
        category,
        status: ProposalStatus::Draft,
        linked_issues,
        created_at: now,
        voting_deadline,
        signature: vec![], // Caller signs after creation
        patch,
    };

    // Persist
    let key = hex::encode(proposal.hash);
    db.put_json(TREE_PROPOSALS, &key, &proposal)?;

    Ok(proposal)
}

/// Open a draft proposal for voting.
pub fn open_proposal(
    db: &Database,
    proposal_hash: &Blake3Hash,
    author: &NodeId,
    params: &GovernanceParams,
) -> Result<Proposal, SwarmError> {
    let key = hex::encode(proposal_hash);
    let mut proposal: Proposal = db
        .get_json(TREE_PROPOSALS, &key)?
        .ok_or_else(|| SwarmError::ProposalNotFound(key.clone()))?;

    if &proposal.author != author {
        return Err(SwarmError::Governance(
            "Only the author can open a proposal".into(),
        ));
    }
    if proposal.status != ProposalStatus::Draft {
        return Err(SwarmError::Governance(format!(
            "Cannot open proposal in {:?} status",
            proposal.status
        )));
    }

    let now = chrono::Utc::now();
    let voting_days = voting_period_days(&proposal.category, params);
    proposal.status = ProposalStatus::Open;
    proposal.voting_deadline = now + chrono::Duration::days(voting_days as i64);

    db.put_json(TREE_PROPOSALS, &key, &proposal)?;
    Ok(proposal)
}

/// Amend a proposal (resets voting deadline). Max amendments enforced.
pub fn amend_proposal(
    db: &Database,
    amendment: &ProposalAmendment,
    amendment_count: u32,
    params: &GovernanceParams,
) -> Result<Proposal, SwarmError> {
    let key = hex::encode(amendment.proposal_hash);
    let mut proposal: Proposal = db
        .get_json(TREE_PROPOSALS, &key)?
        .ok_or_else(|| SwarmError::ProposalNotFound(key.clone()))?;

    if proposal.author != amendment.author {
        return Err(SwarmError::Governance(
            "Only the author can amend a proposal".into(),
        ));
    }

    if !matches!(
        proposal.status,
        ProposalStatus::Open | ProposalStatus::Amended
    ) {
        return Err(SwarmError::Governance(format!(
            "Cannot amend proposal in {:?} status",
            proposal.status
        )));
    }

    if amendment_count >= params.max_proposal_amendments {
        return Err(SwarmError::Governance(format!(
            "Maximum {} amendments reached; withdraw and resubmit",
            params.max_proposal_amendments
        )));
    }

    if !amendment.body.is_empty() {
        proposal.body = amendment.body.clone();
    }
    if let Some(ref new_patch) = amendment.new_patch {
        proposal.patch = Some(new_patch.clone());
    }

    let voting_days = voting_period_days(&proposal.category, params);
    let now = chrono::Utc::now();
    proposal.status = ProposalStatus::Amended;
    proposal.voting_deadline = now + chrono::Duration::days(voting_days as i64);

    db.put_json(TREE_PROPOSALS, &key, &proposal)?;
    Ok(proposal)
}

/// Withdraw a proposal (only by author).
pub fn withdraw_proposal(
    db: &Database,
    proposal_hash: &Blake3Hash,
    author: &NodeId,
) -> Result<Proposal, SwarmError> {
    let key = hex::encode(proposal_hash);
    let mut proposal: Proposal = db
        .get_json(TREE_PROPOSALS, &key)?
        .ok_or_else(|| SwarmError::ProposalNotFound(key.clone()))?;

    if &proposal.author != author {
        return Err(SwarmError::Governance(
            "Only the author can withdraw a proposal".into(),
        ));
    }

    if matches!(
        proposal.status,
        ProposalStatus::Released | ProposalStatus::Implemented
    ) {
        return Err(SwarmError::Governance(
            "Cannot withdraw an implemented/released proposal".into(),
        ));
    }

    proposal.status = ProposalStatus::Withdrawn;
    db.put_json(TREE_PROPOSALS, &key, &proposal)?;
    Ok(proposal)
}

/// Apply a status change to a proposal (used by voting and release system).
pub fn apply_status_change(
    db: &Database,
    change: &ProposalStatusChange,
) -> Result<Proposal, SwarmError> {
    let key = hex::encode(change.proposal_hash);
    let mut proposal: Proposal = db
        .get_json(TREE_PROPOSALS, &key)?
        .ok_or_else(|| SwarmError::ProposalNotFound(key.clone()))?;

    // Validate the transition
    validate_status_transition(&proposal.status, &change.new_status)?;

    proposal.status = change.new_status.clone();
    db.put_json(TREE_PROPOSALS, &key, &proposal)?;
    Ok(proposal)
}

/// Get a proposal by hash.
pub fn get_proposal(db: &Database, hash: &Blake3Hash) -> Result<Option<Proposal>, SwarmError> {
    let key = hex::encode(hash);
    db.get_json(TREE_PROPOSALS, &key)
}

/// List all proposals.
pub fn list_proposals(db: &Database) -> Result<Vec<Proposal>, SwarmError> {
    let tree = db.tree(TREE_PROPOSALS)?;
    let mut proposals = Vec::new();
    for entry in tree.iter() {
        let (_, value) = entry.map_err(SwarmError::Database)?;
        if let Ok(proposal) = serde_json::from_slice::<Proposal>(&value) {
            proposals.push(proposal);
        }
    }
    Ok(proposals)
}

/// List proposals filtered by status.
pub fn list_proposals_by_status(
    db: &Database,
    status: &ProposalStatus,
) -> Result<Vec<Proposal>, SwarmError> {
    let all = list_proposals(db)?;
    Ok(all.into_iter().filter(|p| &p.status == status).collect())
}

// ---- Helpers ----

fn voting_period_days(category: &ProposalCategory, params: &GovernanceParams) -> u32 {
    match category {
        ProposalCategory::CodeChange
        | ProposalCategory::ModelAddition
        | ProposalCategory::ParameterTuning => params.code_change_voting_days,
        ProposalCategory::ProtocolChange | ProposalCategory::ModelDeprecation => {
            params.protocol_change_voting_days
        }
        ProposalCategory::GovernanceChange => params.governance_change_voting_days,
    }
}

fn validate_status_transition(
    current: &ProposalStatus,
    new: &ProposalStatus,
) -> Result<(), SwarmError> {
    let valid = matches!(
        (current, new),
        (ProposalStatus::Draft, ProposalStatus::Open)
            | (ProposalStatus::Draft, ProposalStatus::Withdrawn)
            | (ProposalStatus::Open, ProposalStatus::Accepted)
            | (ProposalStatus::Open, ProposalStatus::Rejected)
            | (ProposalStatus::Open, ProposalStatus::Amended)
            | (ProposalStatus::Open, ProposalStatus::Withdrawn)
            | (ProposalStatus::Amended, ProposalStatus::Accepted)
            | (ProposalStatus::Amended, ProposalStatus::Rejected)
            | (ProposalStatus::Amended, ProposalStatus::Withdrawn)
            | (ProposalStatus::Accepted, ProposalStatus::Implemented)
            | (ProposalStatus::Accepted, ProposalStatus::Rejected) // emergency veto
            | (ProposalStatus::Implemented, ProposalStatus::Released)
    );

    if !valid {
        return Err(SwarmError::Governance(format!(
            "Invalid status transition: {current:?} -> {new:?}"
        )));
    }
    Ok(())
}

fn compute_proposal_hash(
    author: &NodeId,
    title: &str,
    body: &str,
    created_at: &chrono::DateTime<chrono::Utc>,
) -> Blake3Hash {
    let mut hasher = blake3::Hasher::new();
    hasher.update(&author.0);
    hasher.update(title.as_bytes());
    hasher.update(body.as_bytes());
    hasher.update(created_at.to_rfc3339().as_bytes());
    *hasher.finalize().as_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_db() -> Database {
        let dir = tempfile::tempdir().unwrap();
        Database::open(dir.path()).unwrap()
    }

    fn test_params() -> GovernanceParams {
        GovernanceParams::default()
    }

    fn test_node_id() -> NodeId {
        NodeId([1u8; 32])
    }

    fn make_params(title: &str, role: GovernanceRole) -> CreateProposalParams {
        CreateProposalParams {
            author: test_node_id(),
            title: title.into(),
            body: "Description body".into(),
            category: ProposalCategory::CodeChange,
            linked_issues: vec![],
            patch: None,
            role,
        }
    }

    #[test]
    fn create_and_get_proposal() {
        let db = test_db();
        let proposal = create_proposal(
            &db,
            make_params("Fix bug in pipeline", GovernanceRole::Contributor),
            &test_params(),
        )
        .unwrap();

        assert_eq!(proposal.status, ProposalStatus::Draft);
        assert_eq!(proposal.title, "Fix bug in pipeline");

        let fetched = get_proposal(&db, &proposal.hash).unwrap().unwrap();
        assert_eq!(fetched.title, proposal.title);
    }

    #[test]
    fn member_cannot_create_proposal() {
        let db = test_db();
        let result = create_proposal(
            &db,
            make_params("My proposal", GovernanceRole::Member),
            &test_params(),
        );
        assert!(result.is_err());
    }

    #[test]
    fn proposal_lifecycle() {
        let db = test_db();
        let params = test_params();
        let author = test_node_id();

        let proposal = create_proposal(
            &db,
            make_params("Add feature", GovernanceRole::Contributor),
            &params,
        )
        .unwrap();

        // Open it
        let opened = open_proposal(&db, &proposal.hash, &author, &params).unwrap();
        assert_eq!(opened.status, ProposalStatus::Open);

        // Accept it via status change
        let change = ProposalStatusChange {
            proposal_hash: proposal.hash,
            new_status: ProposalStatus::Accepted,
            changed_by: NodeId([2u8; 32]),
            timestamp: chrono::Utc::now(),
            signature: vec![],
        };
        let accepted = apply_status_change(&db, &change).unwrap();
        assert_eq!(accepted.status, ProposalStatus::Accepted);
    }

    #[test]
    fn invalid_status_transition_rejected() {
        let db = test_db();
        let params = test_params();

        let proposal = create_proposal(
            &db,
            make_params("Test", GovernanceRole::Contributor),
            &params,
        )
        .unwrap();

        // Cannot go from Draft directly to Accepted
        let change = ProposalStatusChange {
            proposal_hash: proposal.hash,
            new_status: ProposalStatus::Accepted,
            changed_by: NodeId([2u8; 32]),
            timestamp: chrono::Utc::now(),
            signature: vec![],
        };
        assert!(apply_status_change(&db, &change).is_err());
    }

    #[test]
    fn empty_title_rejected() {
        let db = test_db();
        let result = create_proposal(
            &db,
            make_params("", GovernanceRole::Contributor),
            &test_params(),
        );
        assert!(result.is_err());
    }
}
