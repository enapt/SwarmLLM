use crate::error::SwarmError;
use crate::storage::db::Database;
use crate::types::{
    Blake3Hash, GovernanceParams, NetworkStats, NodeId, Proposal, ProposalCategory, ProposalStatus,
    ProposalVote, VoteChoice,
};

/// Database tree for proposal votes.
const TREE_VOTES: &str = "proposal_votes";

/// Result of tallying votes on a proposal.
#[derive(Clone, Debug)]
pub struct VoteTallyResult {
    pub total_weight: u64,
    pub approve_weight: u64,
    pub reject_weight: u64,
    pub abstain_weight: u64,
    pub unique_voters: u32,
    pub quorum_met: bool,
    pub approved: bool,
}

/// Cast a vote on a proposal. Validates the vote and persists it.
pub fn cast_vote(
    db: &Database,
    vote: &ProposalVote,
    proposal: &Proposal,
) -> Result<(), SwarmError> {
    // Can only vote on Open or Amended proposals
    if !matches!(
        proposal.status,
        ProposalStatus::Open | ProposalStatus::Amended
    ) {
        return Err(SwarmError::Governance(format!(
            "Cannot vote on proposal in {:?} status",
            proposal.status
        )));
    }

    // Check if voting deadline has passed
    let now = chrono::Utc::now();
    if now > proposal.voting_deadline {
        return Err(SwarmError::Governance("Voting deadline has passed".into()));
    }

    // Weight must be positive
    if vote.weight == 0 {
        return Err(SwarmError::Governance(
            "Vote weight must be positive".into(),
        ));
    }

    // Store vote (key: proposal_hash/voter_hex to deduplicate)
    let key = format!(
        "{}/{}",
        hex::encode(vote.proposal_hash),
        hex::encode(&vote.voter.0[..8])
    );
    db.put_json(TREE_VOTES, &key, vote)?;

    Ok(())
}

/// Get all votes for a proposal.
pub fn get_votes(
    db: &Database,
    proposal_hash: &Blake3Hash,
) -> Result<Vec<ProposalVote>, SwarmError> {
    let tree = db.tree(TREE_VOTES)?;
    let prefix = hex::encode(proposal_hash);
    let mut votes = Vec::new();

    for entry in tree.scan_prefix(prefix.as_bytes()) {
        let (_, value) = entry.map_err(SwarmError::Database)?;
        if let Ok(vote) = serde_json::from_slice::<ProposalVote>(&value) {
            votes.push(vote);
        }
    }

    Ok(votes)
}

/// Tally votes for a proposal and determine outcome.
pub fn tally_votes(
    votes: &[ProposalVote],
    proposal: &Proposal,
    network_stats: &NetworkStats,
    params: &GovernanceParams,
) -> VoteTallyResult {
    let mut approve_weight = 0u64;
    let mut reject_weight = 0u64;
    let mut abstain_weight = 0u64;

    for vote in votes {
        match vote.vote {
            VoteChoice::Approve => approve_weight += vote.weight,
            VoteChoice::Reject => reject_weight += vote.weight,
            VoteChoice::Abstain => abstain_weight += vote.weight,
        }
    }

    let total_weight = approve_weight + reject_weight + abstain_weight;
    let unique_voters = votes.len() as u32;

    // Quorum check (percentage of total active network weight)
    let quorum_pct = quorum_requirement(&proposal.category, params);
    let quorum_threshold =
        (network_stats.total_active_vote_weight as f64 * quorum_pct as f64) as u64;
    let quorum_met = total_weight >= quorum_threshold;

    // Approval check (percentage of non-abstain votes cast)
    let approval_pct = approval_requirement(&proposal.category, params);
    let cast_weight = approve_weight + reject_weight;
    let approved = if cast_weight > 0 {
        let approval_ratio = approve_weight as f64 / cast_weight as f64;
        quorum_met && approval_ratio >= approval_pct as f64
    } else {
        false
    };

    VoteTallyResult {
        total_weight,
        approve_weight,
        reject_weight,
        abstain_weight,
        unique_voters,
        quorum_met,
        approved,
    }
}

/// Check whether a proposal's voting period has ended and determine its final status.
///
/// Returns `Some(ProposalStatus::Accepted)` or `Some(ProposalStatus::Rejected)` if
/// the deadline has passed, or `None` if voting is still active.
pub fn check_deadline(
    proposal: &Proposal,
    votes: &[ProposalVote],
    network_stats: &NetworkStats,
    params: &GovernanceParams,
) -> Option<ProposalStatus> {
    let now = chrono::Utc::now();
    if now <= proposal.voting_deadline {
        return None; // Still active
    }

    let tally = tally_votes(votes, proposal, network_stats, params);
    if tally.approved {
        Some(ProposalStatus::Accepted)
    } else {
        Some(ProposalStatus::Rejected)
    }
}

/// Emergency veto by council. Requires 5/7 council votes within 48h of acceptance.
pub fn emergency_veto(
    db: &Database,
    proposal_hash: &Blake3Hash,
    council_vetoes: &[(NodeId, Vec<u8>)],
    params: &GovernanceParams,
) -> Result<bool, SwarmError> {
    // Need 5 out of council_seats (default 7)
    let required = (params.council_seats as f32 * 5.0 / 7.0).ceil() as usize;
    if council_vetoes.len() >= required {
        // Veto the proposal
        let key = hex::encode(proposal_hash);
        let mut proposal: Proposal = db
            .get_json("proposals", &key)?
            .ok_or_else(|| SwarmError::ProposalNotFound(key.clone()))?;

        if proposal.status != ProposalStatus::Accepted {
            return Err(SwarmError::Governance(
                "Can only veto accepted proposals".into(),
            ));
        }

        proposal.status = ProposalStatus::Rejected;
        db.put_json("proposals", &key, &proposal)?;
        Ok(true)
    } else {
        Ok(false)
    }
}

// ---- Helpers ----

fn quorum_requirement(category: &ProposalCategory, params: &GovernanceParams) -> f32 {
    match category {
        ProposalCategory::CodeChange
        | ProposalCategory::ModelAddition
        | ProposalCategory::ParameterTuning => params.code_change_quorum_pct,
        ProposalCategory::ProtocolChange | ProposalCategory::ModelDeprecation => {
            params.protocol_change_quorum_pct
        }
        ProposalCategory::GovernanceChange => params.governance_change_quorum_pct,
    }
}

fn approval_requirement(category: &ProposalCategory, params: &GovernanceParams) -> f32 {
    match category {
        ProposalCategory::CodeChange
        | ProposalCategory::ParameterTuning
        | ProposalCategory::ModelDeprecation => params.code_change_approval_pct,
        ProposalCategory::ModelAddition => 0.30, // Lower threshold for model additions
        ProposalCategory::ProtocolChange => params.protocol_change_approval_pct,
        ProposalCategory::GovernanceChange => params.governance_change_approval_pct,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{GovernanceParams, GovernanceRole};

    fn make_vote(voter_byte: u8, choice: VoteChoice, weight: u64) -> ProposalVote {
        ProposalVote {
            proposal_hash: [0u8; 32],
            voter: crate::types::NodeId([voter_byte; 32]),
            vote: choice,
            weight,
            role: GovernanceRole::Member,
            timestamp: chrono::Utc::now(),
            signature: vec![],
        }
    }

    fn make_proposal() -> Proposal {
        Proposal {
            hash: [0u8; 32],
            author: crate::types::NodeId([1u8; 32]),
            title: "Test".into(),
            body: "Body".into(),
            category: ProposalCategory::CodeChange,
            status: ProposalStatus::Open,
            linked_issues: vec![],
            created_at: chrono::Utc::now(),
            voting_deadline: chrono::Utc::now() + chrono::Duration::days(7),
            signature: vec![],
            patch: None,
        }
    }

    #[test]
    fn tally_simple_majority() {
        let params = GovernanceParams::default();
        let proposal = make_proposal();
        let network_stats = NetworkStats {
            total_active_vote_weight: 1000,
            nodes_with_30d_uptime: 100,
            total_active_nodes: 200,
        };

        let votes = vec![
            make_vote(1, VoteChoice::Approve, 60),
            make_vote(2, VoteChoice::Approve, 40),
            make_vote(3, VoteChoice::Reject, 30),
        ];

        let result = tally_votes(&votes, &proposal, &network_stats, &params);
        assert_eq!(result.approve_weight, 100);
        assert_eq!(result.reject_weight, 30);
        assert_eq!(result.unique_voters, 3);
        assert!(result.quorum_met); // 130 >= 100 (10% of 1000)
        assert!(result.approved); // 100/(100+30) > 50%
    }

    #[test]
    fn tally_quorum_not_met() {
        let params = GovernanceParams::default();
        let proposal = make_proposal();
        let network_stats = NetworkStats {
            total_active_vote_weight: 10_000,
            nodes_with_30d_uptime: 100,
            total_active_nodes: 200,
        };

        let votes = vec![make_vote(1, VoteChoice::Approve, 50)];

        let result = tally_votes(&votes, &proposal, &network_stats, &params);
        assert!(!result.quorum_met); // 50 < 1000 (10% of 10000)
        assert!(!result.approved);
    }

    #[test]
    fn tally_governance_change_requires_supermajority() {
        let params = GovernanceParams::default();
        let mut proposal = make_proposal();
        proposal.category = ProposalCategory::GovernanceChange;

        let network_stats = NetworkStats {
            total_active_vote_weight: 1000,
            nodes_with_30d_uptime: 100,
            total_active_nodes: 200,
        };

        // 70% approval — not enough for 75% threshold
        let votes = vec![
            make_vote(1, VoteChoice::Approve, 250),
            make_vote(2, VoteChoice::Reject, 100),
        ];

        let result = tally_votes(&votes, &proposal, &network_stats, &params);
        assert!(result.quorum_met); // 350 >= 250 (25% of 1000)
                                    // 250/350 = 71.4% < 75%
        assert!(!result.approved);
    }

    #[test]
    fn abstain_counts_toward_quorum_not_approval() {
        let params = GovernanceParams::default();
        let proposal = make_proposal();
        let network_stats = NetworkStats {
            total_active_vote_weight: 1000,
            nodes_with_30d_uptime: 100,
            total_active_nodes: 200,
        };

        let votes = vec![
            make_vote(1, VoteChoice::Approve, 30),
            make_vote(2, VoteChoice::Reject, 20),
            make_vote(3, VoteChoice::Abstain, 80),
        ];

        let result = tally_votes(&votes, &proposal, &network_stats, &params);
        assert_eq!(result.total_weight, 130);
        assert!(result.quorum_met); // 130 >= 100
                                    // Approval only considers non-abstain: 30/(30+20) = 60% > 50%
        assert!(result.approved);
    }
}
