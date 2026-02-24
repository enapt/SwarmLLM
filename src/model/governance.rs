use std::collections::HashSet;

use dashmap::DashMap;

use crate::error::SwarmError;
use crate::types::{Blake3Hash, ModelVote, NodeId};

/// Required weighted vote ratio for a model to be accepted (2/3 majority).
pub const ACCEPTANCE_THRESHOLD: f32 = 0.67;

/// Minimum credit weight required to cast a valid vote.
pub const MIN_VOTE_WEIGHT: u64 = 100;

/// Duration of the voting window (7 days).
pub const VOTE_WINDOW_SECS: u64 = 86400 * 7;

/// Final verdict for a model vote tally.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VoteVerdict {
    Accepted,
    Rejected,
    Pending,
}

/// Accumulated vote state for a single model manifest.
#[derive(Debug, Clone)]
pub struct VoteTally {
    pub model_manifest_hash: Blake3Hash,
    pub votes_for: u64,
    pub votes_against: u64,
    pub unique_voters: HashSet<NodeId>,
    pub opened_at: chrono::DateTime<chrono::Utc>,
    pub closed: bool,
}

impl VoteTally {
    pub fn new(model_manifest_hash: Blake3Hash) -> Self {
        Self {
            model_manifest_hash,
            votes_for: 0,
            votes_against: 0,
            unique_voters: HashSet::new(),
            opened_at: chrono::Utc::now(),
            closed: false,
        }
    }

    /// Record a new vote.
    ///
    /// Returns an error if the voter has already voted or the vote weight
    /// is below the minimum threshold.
    pub fn record_vote(&mut self, vote: &ModelVote) -> Result<(), SwarmError> {
        if self.closed {
            return Err(SwarmError::Internal("Voting period has closed".into()));
        }

        if vote.weight < MIN_VOTE_WEIGHT {
            return Err(SwarmError::InsufficientCredits {
                balance: vote.weight as i64,
                required: MIN_VOTE_WEIGHT as i64,
            });
        }

        if self.unique_voters.contains(&vote.voter) {
            return Err(SwarmError::Internal("Voter has already voted".into()));
        }

        self.unique_voters.insert(vote.voter.clone());

        if vote.vote {
            self.votes_for += vote.weight;
        } else {
            self.votes_against += vote.weight;
        }

        Ok(())
    }

    /// Current acceptance ratio (0.0 to 1.0).
    pub fn acceptance_ratio(&self) -> f32 {
        let total = self.votes_for + self.votes_against;
        if total == 0 {
            return 0.0;
        }
        self.votes_for as f32 / total as f32
    }

    /// Whether the voting window has expired.
    pub fn is_expired(&self) -> bool {
        let age = chrono::Utc::now()
            .signed_duration_since(self.opened_at)
            .num_seconds() as u64;
        age > VOTE_WINDOW_SECS
    }

    /// Determine the final verdict based on current votes and expiration.
    pub fn verdict(&self) -> VoteVerdict {
        if self.closed || self.is_expired() {
            if self.acceptance_ratio() >= ACCEPTANCE_THRESHOLD {
                VoteVerdict::Accepted
            } else {
                VoteVerdict::Rejected
            }
        } else {
            VoteVerdict::Pending
        }
    }

    /// Close voting and seal the tally.
    pub fn close(&mut self) {
        self.closed = true;
    }
}

/// Process an incoming ModelVote gossip message.
///
/// Looks up or creates the tally for the voted manifest, records the vote,
/// and returns the current verdict if the vote changed the outcome.
pub fn process_vote(
    tallies: &DashMap<Blake3Hash, VoteTally>,
    vote: ModelVote,
) -> Result<Option<VoteVerdict>, SwarmError> {
    let mut tally = tallies
        .entry(vote.model_manifest_hash)
        .or_insert_with(|| VoteTally::new(vote.model_manifest_hash));

    tally.record_vote(&vote)?;

    let verdict = tally.verdict();
    if verdict != VoteVerdict::Pending {
        Ok(Some(verdict))
    } else {
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_vote(voter_byte: u8, vote: bool, weight: u64) -> ModelVote {
        ModelVote {
            voter: NodeId([voter_byte; 32]),
            model_manifest_hash: [0xAB; 32],
            vote,
            weight,
            signature: vec![],
        }
    }

    #[test]
    fn empty_tally() {
        let tally = VoteTally::new([0xAB; 32]);
        assert_eq!(tally.acceptance_ratio(), 0.0);
        assert_eq!(tally.verdict(), VoteVerdict::Pending);
    }

    #[test]
    fn record_vote_success() {
        let mut tally = VoteTally::new([0xAB; 32]);
        let vote = make_vote(1, true, 1000);
        assert!(tally.record_vote(&vote).is_ok());
        assert_eq!(tally.votes_for, 1000);
        assert_eq!(tally.unique_voters.len(), 1);
    }

    #[test]
    fn duplicate_voter_rejected() {
        let mut tally = VoteTally::new([0xAB; 32]);
        let vote = make_vote(1, true, 1000);
        assert!(tally.record_vote(&vote).is_ok());
        let duplicate = make_vote(1, false, 500);
        assert!(tally.record_vote(&duplicate).is_err());
    }

    #[test]
    fn insufficient_weight_rejected() {
        let mut tally = VoteTally::new([0xAB; 32]);
        let vote = make_vote(1, true, 50); // below MIN_VOTE_WEIGHT
        assert!(tally.record_vote(&vote).is_err());
    }

    #[test]
    fn acceptance_ratio_calculation() {
        let mut tally = VoteTally::new([0xAB; 32]);
        tally.record_vote(&make_vote(1, true, 700)).unwrap();
        tally.record_vote(&make_vote(2, false, 300)).unwrap();
        assert!((tally.acceptance_ratio() - 0.7).abs() < f32::EPSILON);
    }

    #[test]
    fn verdict_accepted_when_closed() {
        let mut tally = VoteTally::new([0xAB; 32]);
        tally.record_vote(&make_vote(1, true, 700)).unwrap();
        tally.record_vote(&make_vote(2, false, 300)).unwrap();
        tally.close();
        assert_eq!(tally.verdict(), VoteVerdict::Accepted);
    }

    #[test]
    fn verdict_rejected_when_closed_below_threshold() {
        let mut tally = VoteTally::new([0xAB; 32]);
        tally.record_vote(&make_vote(1, true, 300)).unwrap();
        tally.record_vote(&make_vote(2, false, 700)).unwrap();
        tally.close();
        assert_eq!(tally.verdict(), VoteVerdict::Rejected);
    }

    #[test]
    fn process_vote_creates_tally() {
        let tallies = DashMap::new();
        let vote = make_vote(1, true, 1000);
        let result = process_vote(&tallies, vote);
        assert!(result.is_ok());
        assert_eq!(tallies.len(), 1);
    }

    #[test]
    fn closed_tally_rejects_new_votes() {
        let mut tally = VoteTally::new([0xAB; 32]);
        tally.close();
        let vote = make_vote(1, true, 1000);
        assert!(tally.record_vote(&vote).is_err());
    }
}
