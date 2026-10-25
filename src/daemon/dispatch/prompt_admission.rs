//! Answering a peer's `PromptAdmissionAsk` (FUTURE_WORK #252 (b)): would this
//! node's conversation memory hold a split's prompt over the ranges the plan
//! gives it? Asked by the split's coordinator before the prompt pass, so a "no"
//! costs one round trip instead of every earlier machine's read of the prompt.
//!
//! The question is WEIGHED, never granted (vLLM's `can_allocate` beside its
//! `allocate`): `ModelProcessPool::weigh_peer_prompt` runs the same arithmetic
//! this node's own pipeline is admitted by (#263) and records nothing — no
//! promise, no reserved positions, no release of an ended conversation, no
//! entry under the asker's request id. The request's forwards are admitted as
//! they always were when they arrive. A security review (2026-10-18) is why:
//! an ask that held room let any peer park this node's whole conversation
//! memory, refreshable forever, and an ask under a live request's id took that
//! request's answer slot.

use std::sync::Arc;

use tokio::sync::mpsc;

use crate::daemon::state::SharedState;
use crate::inference::process_pool::PeerPromptWeight;
use crate::types::{
    NetworkCommand, NodeId, PromptAdmissionAnswer, PromptAdmissionAsk, PromptAdmissionVerdict,
    SwarmMessage,
};

/// The most ranges one ask may name. A plan gives one machine a handful of
/// segments at most; an ask naming more is not one an honest coordinator sends.
const MAX_ASKED_RANGES: usize = 16;

/// Is this an ask an honest coordinator could have sent? Every range non-empty,
/// a few of them, and a prompt to weigh.
fn ask_is_well_formed(ask: &PromptAdmissionAsk) -> bool {
    !ask.layer_ranges.is_empty()
        && ask.layer_ranges.len() <= MAX_ASKED_RANGES
        && ask.layer_ranges.iter().all(|&(start, end)| start < end)
        && ask.prompt_tokens > 0
}

/// Take `asker`'s ask: answer at once what needs no worker, and weigh the rest
/// in a task of its own (the worker may take up to its bound to answer).
///
/// The cheap verdicts come first so that only a question that will really be
/// weighed costs a task. Past this node's [`AskBudget`] the asker is told "go
/// ahead" straight away — the answer every split had before the question
/// existed — rather than made to wait out its own bound.
///
/// [`AskBudget`]: crate::daemon::state::prompt_admission::AskBudget
pub(super) fn on_ask(
    shared_state: Arc<SharedState>,
    network_tx: mpsc::Sender<NetworkCommand>,
    asker: NodeId,
    ask: PromptAdmissionAsk,
) {
    if !ask_is_well_formed(&ask) {
        tracing::debug!(sender = %asker, request_id = %ask.request_id, "Dropping a malformed PromptAdmissionAsk");
        return;
    }
    // Answered over the connection the ask came in on, never a dial to a
    // stale address the registry still carries for it.
    let Some(asker_peer_bytes) = shared_state.resolve_connected_peer_id_bytes(&asker) else {
        tracing::debug!(sender = %asker, "PromptAdmissionAsk from a peer no longer connected");
        return;
    };
    // A node on a metered connection takes no new work for other machines; its
    // forwards would be refused, so the re-plan should plan none here.
    if shared_state.metered_hold_now() {
        let verdict = PromptAdmissionVerdict::Refused {
            message: super::metered_refusal(),
            layers_that_fit: 0,
        };
        answer_unweighed(&network_tx, asker, asker_peer_bytes, &ask, verdict);
        return;
    }
    if !shared_state
        .prompt_ask_budget
        .try_spend(&asker, std::time::Instant::now())
    {
        let verdict = PromptAdmissionVerdict::Proceed;
        answer_unweighed(&network_tx, asker, asker_peer_bytes, &ask, verdict);
        return;
    }
    tokio::spawn(async move {
        let weight = shared_state
            .model_process_pool
            .weigh_peer_prompt(&ask.model_id, ask.layer_ranges.clone(), ask.prompt_tokens)
            .await;
        let verdict = match weight {
            PeerPromptWeight::Fits => PromptAdmissionVerdict::Proceed,
            PeerPromptWeight::DoesNotFit {
                message,
                layers_that_fit,
            } => PromptAdmissionVerdict::Refused {
                // The words a forward's own refusal would have carried.
                message: super::layer_forward::sanitize_peer_facing_error(&message),
                layers_that_fit,
            },
        };
        let load_ahead = matches!(verdict, PromptAdmissionVerdict::Proceed)
            && loads_ahead()
            && shared_state.inference_outage().is_none()
            && may_load_ahead(
                &shared_state.model_registry,
                shared_state.identity.node_id(),
                &ask.model_id,
                &ask.layer_ranges,
            );
        let (model_id, ranges) = (ask.model_id.clone(), ask.layer_ranges.clone());
        answer(network_tx, asker, asker_peer_bytes, ask, verdict).await;
        // After the answer, so the asker's wait never includes the load.
        if load_ahead {
            shared_state
                .model_process_pool
                .load_ahead_for_a_peer(&model_id, &ranges)
                .await;
        }
    });
}

/// The most ranges one ask may have loaded ahead. A plan gives one machine a
/// segment or two; an ask naming more is answered, but its forwards load the
/// rest as they always did.
const MAX_RANGES_LOADED_AHEAD: usize = 2;

/// May a peer's ask have these ranges loaded now, ahead of their forwards?
/// Only what a forward naming them could load: each range inside the model
/// (`layer_forward::layer_range_is_valid`, the forward's own check — a range
/// past the end would make the worker drop a resident range it subsumes, then
/// fail its load and leave the charge behind; security review, 2026-10-25),
/// every part covering it held here (a load of layers this node lacks spawns
/// a worker to fail), at most [`MAX_RANGES_LOADED_AHEAD`] of them and none
/// overlapping another (nested ranges would each be held as a copy of their
/// own). A refusal here loads nothing; the answer is unchanged.
fn may_load_ahead(
    registry: &crate::model::registry::ModelRegistry,
    here: &NodeId,
    model_id: &crate::types::ModelId,
    ranges: &[(u32, u32)],
) -> bool {
    let Some(manifest) = registry.get_manifest(model_id) else {
        return false;
    };
    let total = manifest.num_layers as usize;
    let overlap = |a: (u32, u32), b: (u32, u32)| a.0 < b.1 && b.0 < a.1;
    ranges.len() <= MAX_RANGES_LOADED_AHEAD
        && ranges
            .iter()
            .enumerate()
            .all(|(i, &a)| ranges[i + 1..].iter().all(|&b| !overlap(a, b)))
        && ranges.iter().all(|&range| {
            let parts = registry.shards_overlapping_layers(model_id, range);
            super::layer_forward::layer_range_is_valid(range.0 as usize, range.1 as usize, total)
                && !parts.is_empty()
                && parts
                    .iter()
                    .all(|part| registry.shard_holders(part).contains(here))
        })
}

/// `SWARMLLM_LOAD_AHEAD=0` loads a peer's layers only when its first forward
/// arrives, as before 2026-10-25 — the control arm for the measurement.
fn loads_ahead() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("SWARMLLM_LOAD_AHEAD").map_or(true, |v| v != "0"))
}

/// A verdict that needed no worker, sent from the dispatcher itself: no task,
/// and no wait on a full network queue — a question sent at wire speed costs
/// one `try_send` and a debug line, never a task and an info line each
/// (security review, 2026-10-18). Dropped if the queue is full: the asker's
/// wait ends and its segments admit themselves, as for any unanswered ask.
fn answer_unweighed(
    network_tx: &mpsc::Sender<NetworkCommand>,
    asker: NodeId,
    asker_peer_bytes: Vec<u8>,
    ask: &PromptAdmissionAsk,
    verdict: PromptAdmissionVerdict,
) {
    tracing::debug!(
        request_id = %ask.request_id,
        sender = %asker,
        verdict = ?verdict,
        "answered a peer's up-front prompt admission without weighing it"
    );
    if network_tx
        .try_send(answer_to(asker_peer_bytes, ask, verdict))
        .is_err()
    {
        tracing::debug!(sender = %asker, "network queue full — PromptAdmissionAnswer dropped");
    }
}

fn answer_to(
    asker_peer_bytes: Vec<u8>,
    ask: &PromptAdmissionAsk,
    verdict: PromptAdmissionVerdict,
) -> NetworkCommand {
    NetworkCommand::SendDirectMessage {
        target_peer_bytes: asker_peer_bytes,
        message: SwarmMessage::PromptAdmissionAnswer(PromptAdmissionAnswer {
            request_id: ask.request_id,
            ask_id: ask.ask_id,
            verdict,
        }),
        delivery_request_id: None,
    }
}

/// A verdict the worker weighed — at most [`AskBudget`]'s figures a minute,
/// so each is worth an info line.
///
/// [`AskBudget`]: crate::daemon::state::prompt_admission::AskBudget
async fn answer(
    network_tx: mpsc::Sender<NetworkCommand>,
    asker: NodeId,
    asker_peer_bytes: Vec<u8>,
    ask: PromptAdmissionAsk,
    verdict: PromptAdmissionVerdict,
) {
    tracing::info!(
        request_id = %ask.request_id,
        sender = %asker,
        model_id = %ask.model_id,
        ranges = ?ask.layer_ranges,
        prompt_tokens = ask.prompt_tokens,
        verdict = ?verdict,
        "DIAG: answered a peer's up-front prompt admission"
    );
    if let Err(e) = network_tx
        .send(answer_to(asker_peer_bytes, &ask, verdict))
        .await
    {
        tracing::warn!(error = %e, "Failed to send PromptAdmissionAnswer");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ask(ranges: Vec<(u32, u32)>, prompt_tokens: u32) -> PromptAdmissionAsk {
        PromptAdmissionAsk {
            request_id: uuid::Uuid::new_v4(),
            ask_id: uuid::Uuid::new_v4(),
            model_id: crate::types::ModelId("m".into()),
            layer_ranges: ranges,
            prompt_tokens,
        }
    }

    /// A peer's layers are loaded ahead only where a forward naming them could
    /// load them: every part covering them here, inside the model, a range or
    /// two that do not overlap. A range reaching into a part this node lacks,
    /// past the model's end, or a model whose parts are unknown, would spawn a
    /// worker only to fail its load.
    #[test]
    fn a_peers_layers_are_loaded_ahead_only_where_a_forward_could_load_them() {
        use crate::model::registry::ModelRegistry;
        use crate::types::{ModelId, ShardId, ShardInfo};
        let registry = ModelRegistry::new();
        let model = ModelId("m".into());
        let (here, other) = (NodeId([1u8; 32]), NodeId([2u8; 32]));
        let part = |index: u32, layer_range: (u32, u32)| ShardInfo {
            index,
            layer_range,
            size_bytes: 1,
            hash: [0u8; 32],
            tensors: vec![],
        };
        registry.register_manifest(crate::types::ModelManifest {
            id: model.clone(),
            name: "M".into(),
            architecture: crate::types::ModelArchitecture::Llama,
            num_layers: 30,
            num_params_billions: 0.001,
            quantization: crate::types::Quantization::Q4KM,
            total_size_bytes: 3,
            shard_count: 3,
            shards: vec![part(0, (0, 10)), part(1, (10, 20)), part(2, (20, 30))],
            tokenizer_hash: [0u8; 32],
            manifest_hash: [0u8; 32],
            publisher: other.clone(),
            publish_date: chrono::Utc::now(),
            license: "MIT".into(),
            mmproj: None,
            context_length: None,
        });
        let shard = |index| ShardId {
            model_id: model.clone(),
            index,
        };
        registry.record_shard_holder(shard(1), here.clone());
        registry.record_shard_holder(shard(2), here.clone());
        registry.record_shard_holder(shard(0), other.clone());

        let may = |ranges: &[(u32, u32)]| may_load_ahead(&registry, &here, &model, ranges);
        assert!(may(&[(10, 30)]));
        assert!(may(&[(12, 18), (20, 25)]));
        assert!(
            !may(&[(5, 15)]),
            "layers 5-9 are in a part only another computer holds"
        );
        assert!(
            !may(&[(10, 20), (0, 3)]),
            "one range out of reach is enough to load nothing"
        );
        assert!(
            !may(&[(10, 31)]),
            "a range past the model's last layer is refused, as its forward would be"
        );
        assert!(
            !may(&[(10, 20), (10, 25)]),
            "nested ranges would each be held as a copy of their own"
        );
        assert!(
            !may(&[(10, 12), (12, 14), (14, 16)]),
            "more ranges than a plan gives one machine"
        );
        assert!(
            !may_load_ahead(&registry, &here, &ModelId("unknown".into()), &[(0, 3)]),
            "a model whose parts are unknown here"
        );
    }

    #[test]
    fn only_an_ask_an_honest_coordinator_could_send_is_answered() {
        assert!(ask_is_well_formed(&ask(vec![(0, 12), (20, 22)], 500)));
        assert!(!ask_is_well_formed(&ask(vec![], 500)), "no range");
        assert!(
            !ask_is_well_formed(&ask(vec![(12, 12)], 500)),
            "an empty range"
        );
        assert!(
            !ask_is_well_formed(&ask(vec![(12, 4)], 500)),
            "a reversed range"
        );
        assert!(!ask_is_well_formed(&ask(vec![(0, 1)], 0)), "no prompt");
        assert!(!ask_is_well_formed(&ask(
            (0..17).map(|i| (i, i + 1)).collect(),
            500
        )));
    }
}
