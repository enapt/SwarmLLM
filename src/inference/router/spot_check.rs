//! Well-formedness checking of distributed inference output, and the trust
//! that a peer earns — or does not earn — from having served it.
//!
//! # Why the check is not sampled
//!
//! It used to run on 5% of results, after every participating peer had
//! already been credited `InferenceSuccess`. That made the expected trust
//! movement of a peer returning degenerate output **positive**:
//! `+0.01` on every request against `-0.1` on one in twenty is `+0.005` a
//! request, so such a peer climbed the candidate ranking indefinitely
//! (reported on issue #21). Sampling exists to amortise expensive
//! verification; these three checks are assertions over a string the
//! coordinator is already holding, so there was never a cost to amortise —
//! and a check that gates a reward has to run whenever the reward would be
//! paid.
//!
//! # What it can and cannot see
//!
//! These are well-formedness assertions, not verification. A peer returning
//! fluent, confident, wrong tokens passes every one of them, and still earns
//! trust. Detecting *that* requires the same work computed twice and
//! compared — replication with consensus, as in BOINC's validator, or
//! spot-checking against work whose answer is already known, as in Sarmenta's
//! sabotage-tolerance analysis. Both rest on having a ground truth to compare
//! against; we have none, so what follows is the weaker claim it can actually
//! support. The gap is written up in `docs/FUTURE_WORK.md`.
//!
//! - D. P. Anderson, *BOINC: A Platform for Volunteer Computing*, J. Grid
//!   Computing 18(1), 2020 — adaptive replication and the validator.
//! - L. F. G. Sarmenta, *Sabotage-tolerance mechanisms for volunteer
//!   computing systems*, Future Generation Computer Systems 18(4), 2001 —
//!   spot-checking, blacklisting, and credibility-based fault tolerance.

use dashmap::DashMap;

use crate::credit::trust::TrustManager;
use crate::types::{NodeId, PeerInfo, PipelineAssignment};

use super::types::InferenceOutput;

/// What the well-formedness check concluded about a distributed result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum ResultCheck {
    /// Every segment ran on this node. There is no peer to credit or doubt.
    NoRemoteSegments,
    /// The output is well-formed. The peers that served it earn trust for it.
    WellFormed,
    /// The output is malformed, for the named reason. No peer earns trust
    /// from a request that came back like this.
    Malformed(&'static str),
}

/// Inspect a completed distributed result for the shapes that mean the
/// pipeline produced nothing usable, whatever the peers reported.
///
/// Pure and synchronous on purpose: it takes no locks and touches no shared
/// state, because it now runs on the completion path of every distributed
/// request rather than on one in twenty.
pub(super) fn check_distributed_result(
    assignment: &PipelineAssignment,
    local_node_id: &NodeId,
    output: &InferenceOutput,
) -> ResultCheck {
    if !assignment
        .segments
        .iter()
        .any(|s| s.node_id != *local_node_id)
    {
        return ResultCheck::NoRemoteSegments;
    }

    let text = &output.content;

    // Tokens were reported but no text came back with them.
    if text.is_empty() && output.completion_tokens > 0 {
        return ResultCheck::Malformed("empty text with non-zero completion_tokens");
    }

    // Text came back but nothing claims to have generated it.
    if output.completion_tokens == 0 && !text.is_empty() {
        return ResultCheck::Malformed("non-empty text with zero completion_tokens");
    }

    // A long reply that is one character repeated is what a broken segment
    // emits; it is not something a working model produces.
    if output.completion_tokens > 10 {
        let mut chars = text.chars();
        if let Some(first) = chars.next() {
            if chars.all(|c| c == first) {
                return ResultCheck::Malformed("output is a single repeated character");
            }
        }
    }

    ResultCheck::WellFormed
}

/// Apply a verdict to the peers that served the request.
///
/// A peer is credited **once per request**, not once per segment: this is a
/// record of how a peer behaved on one piece of work, and a peer that was
/// handed three segments of a pipeline still only behaved once.
pub(super) fn settle_participant_trust(
    trust_manager: &TrustManager,
    peer_registry: &DashMap<NodeId, PeerInfo>,
    request_id: uuid::Uuid,
    assignment: &PipelineAssignment,
    local_node_id: &NodeId,
    verdict: &ResultCheck,
) {
    let mut peers: Vec<NodeId> = Vec::new();
    for seg in &assignment.segments {
        if seg.node_id != *local_node_id && !peers.contains(&seg.node_id) {
            peers.push(seg.node_id.clone());
        }
    }
    if peers.is_empty() {
        return;
    }

    match verdict {
        ResultCheck::NoRemoteSegments => {}
        ResultCheck::WellFormed => {
            for peer in &peers {
                trust_manager.update_trust(
                    peer_registry,
                    peer,
                    crate::credit::trust::TrustEvent::InferenceSuccess,
                );
            }
        }
        ResultCheck::Malformed(reason) => {
            // Nobody earns trust from this request. That much needs no
            // attribution, and it is on its own enough to stop a peer whose
            // output is always malformed from accumulating trust.
            //
            // The penalty does need attribution, and we only have it when a
            // single peer served the request. Docking every participant of a
            // multi-peer pipeline would let one bad node drag down the trust
            // of every honest node it manages to share a pipeline with —
            // and since candidates are ordered by trust, that is a way to
            // demote competitors rather than collateral damage.
            if peers.len() == 1 {
                trust_manager.update_trust(
                    peer_registry,
                    &peers[0],
                    crate::credit::trust::TrustEvent::SpotCheckFail,
                );
                tracing::warn!(
                    %request_id,
                    peer = %peers[0],
                    reason,
                    "Distributed result was malformed — the one peer that served it loses trust"
                );
            } else {
                tracing::warn!(
                    %request_id,
                    peers = peers.len(),
                    reason,
                    "Distributed result was malformed — no peer earns trust for it, and \
                     which of them produced it is not attributable"
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::db::Database;
    use crate::types::PipelineSegment;
    use swarmllm_types::ShardId;

    fn node(b: u8) -> NodeId {
        NodeId([b; 32])
    }

    fn segment(owner: NodeId) -> PipelineSegment {
        PipelineSegment {
            node_id: owner,
            shard_id: ShardId {
                model_id: crate::types::ModelId("m".into()),
                index: 0,
            },
            layer_range: (0, 1),
        }
    }

    fn assignment(owners: &[NodeId]) -> PipelineAssignment {
        PipelineAssignment {
            request_id: uuid::Uuid::new_v4(),
            segments: owners.iter().cloned().map(segment).collect(),
            standbys: Vec::new(),
            tp_groups: Vec::new(),
            supports_speculative: false,
        }
    }

    fn output(content: &str, completion_tokens: u32) -> InferenceOutput {
        InferenceOutput {
            request_id: uuid::Uuid::new_v4(),
            content: content.to_string(),
            prompt_tokens: 4,
            completion_tokens,
            finish_reason: "stop".into(),
            session_id: None,
            token_logprobs: Vec::new(),
            matched_stop_sequence: None,
            trace: None,
        }
    }

    #[test]
    fn an_all_local_pipeline_has_nobody_to_judge() {
        let me = node(1);
        let a = assignment(std::slice::from_ref(&me));
        assert_eq!(
            check_distributed_result(&a, &me, &output("hello", 2)),
            ResultCheck::NoRemoteSegments
        );
    }

    #[test]
    fn an_ordinary_reply_is_well_formed() {
        let me = node(1);
        let a = assignment(&[me.clone(), node(2)]);
        assert_eq!(
            check_distributed_result(&a, &me, &output("the capital is Paris", 5)),
            ResultCheck::WellFormed
        );
    }

    #[test]
    fn the_three_malformed_shapes_are_each_named() {
        let me = node(1);
        let a = assignment(&[me.clone(), node(2)]);

        assert!(matches!(
            check_distributed_result(&a, &me, &output("", 7)),
            ResultCheck::Malformed(_)
        ));
        assert!(matches!(
            check_distributed_result(&a, &me, &output("text", 0)),
            ResultCheck::Malformed(_)
        ));
        assert!(matches!(
            check_distributed_result(&a, &me, &output(&"x".repeat(40), 20)),
            ResultCheck::Malformed(_)
        ));
    }

    #[test]
    fn a_short_repetitive_reply_is_not_condemned() {
        // Under the repeat threshold this is a legitimate reply — "aaa" is a
        // thing a model can be asked for.
        let me = node(1);
        let a = assignment(&[me.clone(), node(2)]);
        assert_eq!(
            check_distributed_result(&a, &me, &output("aaa", 3)),
            ResultCheck::WellFormed
        );
    }

    /// The classifier half: degenerate output is never called well-formed,
    /// however many times it is seen. (The credit gate is pinned separately
    /// by `a_peer_whose_output_is_always_malformed_gains_no_trust` — this
    /// test stays green if the gate is removed, so it is not the regression
    /// pin and must not be mistaken for one.)
    #[test]
    fn a_degenerate_reply_is_never_judged_well_formed() {
        let me = node(1);
        let a = assignment(&[me.clone(), node(2)]);

        // Every request this peer serves comes back degenerate.
        for _ in 0..1000 {
            let verdict = check_distributed_result(&a, &me, &output(&"z".repeat(64), 32));
            assert!(
                !matches!(verdict, ResultCheck::WellFormed),
                "a degenerate reply must never be judged well-formed"
            );
        }
    }

    /// Drive `settle_participant_trust` the way the completion path does, and
    /// report where the peer's trust ended up.
    fn trust_after(
        segments: &[NodeId],
        me: &NodeId,
        subject: &NodeId,
        verdict: ResultCheck,
        rounds: usize,
    ) -> f32 {
        let tm = TrustManager::new(Database::open_temp().unwrap());
        let registry: DashMap<NodeId, PeerInfo> = DashMap::new();
        let a = assignment(segments);
        for _ in 0..rounds {
            settle_participant_trust(&tm, &registry, uuid::Uuid::new_v4(), &a, me, &verdict);
        }
        tm.get_trust(subject)
    }

    /// The regression this module exists for, stated as the reporter stated
    /// it: a peer whose output is always malformed must not accumulate trust.
    ///
    /// Fails if the credit is ever moved back ahead of the check, or made
    /// unconditional again: 1000 rounds of `+0.01` saturates at 1.0.
    #[test]
    fn a_peer_whose_output_is_always_malformed_gains_no_trust() {
        let me = node(1);
        let bad = node(2);
        let with_a_witness = [me.clone(), bad.clone(), node(3)];

        let score = trust_after(
            &with_a_witness,
            &me,
            &bad,
            ResultCheck::Malformed("output is a single repeated character"),
            1000,
        );

        assert!(
            score <= crate::credit::trust::DEFAULT_TRUST,
            "a peer that never returns a usable answer must not rise above where \
             it started; got {score}"
        );
    }

    /// The other half: a peer that is the ONLY one that served the request is
    /// attributable, so it is actually docked rather than merely unpaid.
    #[test]
    fn the_sole_server_of_a_malformed_result_is_docked() {
        let me = node(1);
        let bad = node(2);
        let score = trust_after(
            &[me.clone(), bad.clone()],
            &me,
            &bad,
            ResultCheck::Malformed("empty text with non-zero completion_tokens"),
            3,
        );
        assert!(
            score < crate::credit::trust::DEFAULT_TRUST - 0.2,
            "three attributable malformed results should cost ~0.3; got {score}"
        );
    }

    /// A peer sharing the blame with others is NOT docked — one bad node must
    /// not be able to demote the honest peers it shares a pipeline with.
    #[test]
    fn a_peer_in_a_crowd_is_not_docked_for_an_unattributable_failure() {
        let me = node(1);
        let honest = node(3);
        let score = trust_after(
            &[me.clone(), node(2), honest.clone()],
            &me,
            &honest,
            ResultCheck::Malformed("empty text with non-zero completion_tokens"),
            50,
        );
        assert!(
            (score - crate::credit::trust::DEFAULT_TRUST).abs() < f32::EPSILON,
            "an unattributable failure must leave a participant exactly where it \
             was, neither paid nor punished; got {score}"
        );
    }

    #[test]
    fn a_well_formed_result_does_pay_the_peers_that_served_it() {
        let me = node(1);
        let good = node(2);
        let score = trust_after(
            &[me.clone(), good.clone()],
            &me,
            &good,
            ResultCheck::WellFormed,
            5,
        );
        assert!(
            score > crate::credit::trust::DEFAULT_TRUST,
            "well-formed work still earns trust; got {score}"
        );
    }

    /// A peer handed three segments of one pipeline behaved once, not three
    /// times — otherwise grabbing more of a pipeline ratchets trust faster.
    #[test]
    fn a_peer_holding_several_segments_is_still_one_observation() {
        let me = node(1);
        let peer = node(2);

        let many = trust_after(
            &[me.clone(), peer.clone(), peer.clone(), peer.clone()],
            &me,
            &peer,
            ResultCheck::WellFormed,
            1,
        );
        let one = trust_after(
            &[me.clone(), peer.clone()],
            &me,
            &peer,
            ResultCheck::WellFormed,
            1,
        );
        assert!(
            (many - one).abs() < f32::EPSILON,
            "three segments on one peer paid {many}, one segment paid {one}"
        );
    }
}
