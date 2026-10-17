//! The dispatcher's half of announce-then-pull for model manifests
//! (FUTURE_WORK #91 (2)). The memory and its rules are
//! `daemon::state::manifest_sync`; the sending half is
//! `health::monitor::broadcast_manifests`.

use std::collections::HashSet;
use std::sync::Arc;

use tokio::sync::mpsc;

use crate::daemon::state::manifest_sync::{
    MAX_DIGEST_VERSIONS, MAX_MODEL_ID_BYTES, MAX_PULL_MODELS,
};
use crate::daemon::SharedState;
use crate::types::{
    ManifestDigest, ManifestPull, MessageTransport, ModelId, NetworkCommand, NodeId, SwarmMessage,
};

/// A peer has said which version of each manifest it holds: pull from it the
/// ones this node lacks.
///
/// Read only point to point, from a known sender — this node never puts a
/// digest on gossip, so one arriving there is not a peer's catch-up and is not
/// answered. A copied-folder name is never pulled; ingestion would refuse it.
pub(super) fn on_digest(
    shared_state: &SharedState,
    network_tx: &mpsc::Sender<NetworkCommand>,
    sender: Option<&NodeId>,
    transport: MessageTransport,
    digest: &ManifestDigest,
) {
    let Some(sender) = sender else {
        tracing::debug!("Dropping unauthenticated ManifestDigest");
        return;
    };
    if transport != MessageTransport::Direct {
        return;
    }
    let digest = ManifestDigest {
        versions: digest
            .versions
            .iter()
            .take(MAX_DIGEST_VERSIONS)
            .filter(|v| !crate::model::manifest::is_backup_artifact_id(&v.model_id.0))
            .cloned()
            .collect(),
    };
    let sync = &shared_state.models.manifest_sync;
    let wanted = sync.versions_to_pull(&digest, sender, |model| {
        shared_state.model_registry.manifest_version(model)
    });
    if wanted.is_empty() {
        return;
    }
    let Some(target_peer_bytes) = shared_state.resolve_connected_peer_id_bytes(sender) else {
        sync.pull_not_sent(&wanted);
        return;
    };
    tracing::debug!(
        peer = %sender,
        named = digest.versions.len(),
        pulling = wanted.len(),
        "DIAG: pulling the manifest versions a peer's digest named that this node lacks"
    );
    let pull = SwarmMessage::ManifestPull(ManifestPull {
        models: wanted.iter().map(|(model, _)| model.clone()).collect(),
    });
    if network_tx
        .try_send(NetworkCommand::SendDirectMessage {
            target_peer_bytes,
            message: pull,
            delivery_request_id: None,
        })
        .is_err()
    {
        sync.pull_not_sent(&wanted);
    }
}

/// A peer asked for this node's manifest of some models. Answered, within the
/// peer's budget (`ManifestSync::answers_allowed`: each model once per
/// `ANSWER_AGAIN_AFTER`, `ANSWERS_PER_WINDOW` a minute), with the manifests
/// this node gossips — never one it withholds or holds no part of
/// (`ModelRegistry::manifest_to_gossip`) — each followed by the model's upload
/// claim, exactly as the newcomer catch-up sends them.
///
/// Everything cheap is refused before any manifest is copied: a peer that is
/// not connected (an answer could not reach it), an empty or over-long ask,
/// a spent budget. Sent from a task of its own: an answer is up to ~35 KB a
/// model, and the dispatcher must not wait on a busy network loop.
pub(super) fn on_pull(
    shared_state: &Arc<SharedState>,
    network_tx: &mpsc::Sender<NetworkCommand>,
    sender: Option<&NodeId>,
    transport: MessageTransport,
    pull: &ManifestPull,
) {
    let Some(sender) = sender else {
        tracing::debug!("Dropping unauthenticated ManifestPull");
        return;
    };
    if transport != MessageTransport::Direct {
        return;
    }
    let Some(target_peer_bytes) = shared_state.resolve_connected_peer_id_bytes(sender) else {
        return;
    };
    let our_id = shared_state.identity.node_id().clone();
    let registry = &shared_state.model_registry;
    let mut seen = HashSet::new();
    let gossiped: Vec<ModelId> = pull
        .models
        .iter()
        .take(MAX_PULL_MODELS)
        .filter(|model| model.0.len() <= MAX_MODEL_ID_BYTES && seen.insert(*model))
        .filter(|model| registry.gossips_model(&our_id, model))
        .cloned()
        .collect();
    if gossiped.is_empty() {
        return;
    }
    let asked = gossiped.len();
    let allowed = shared_state
        .models
        .manifest_sync
        .answers_allowed(sender, gossiped);
    if allowed.len() < asked {
        tracing::debug!(
            peer = %sender,
            asked,
            allowed = allowed.len(),
            "DIAG: answering only part of a manifest pull — answered lately, or the peer has used its budget"
        );
    }
    let answer: Vec<_> = allowed
        .iter()
        .filter_map(|model| registry.manifest_to_gossip(&our_id, model))
        .collect();
    if answer.is_empty() {
        return;
    }
    let mut messages = Vec::with_capacity(answer.len() * 2);
    for manifest in answer {
        let model = manifest.id.clone();
        messages.push(SwarmMessage::ModelManifest(manifest));
        messages.extend(crate::health::monitor::upload_claim(
            shared_state,
            &model,
            &our_id,
        ));
    }
    let network_tx = network_tx.clone();
    tokio::spawn(async move {
        for message in messages {
            let send = NetworkCommand::SendDirectMessage {
                target_peer_bytes: target_peer_bytes.clone(),
                message,
                delivery_request_id: None,
            };
            if network_tx.send(send).await.is_err() {
                break;
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ManifestVersion, ModelManifest};

    fn state() -> Arc<SharedState> {
        let temp = tempfile::tempdir().unwrap();
        let (state, _, _) = SharedState::new(
            crate::config::Config::default(),
            crate::identity::Identity::generate(),
            crate::storage::db::Database::open(temp.path()).unwrap(),
            Arc::new(tokio::sync::Mutex::new(
                crate::inference::executor::ModelExecutor::new(),
            )),
            None,
        );
        state
    }

    fn manifest(id: &str, hash: u8, publisher: NodeId) -> ModelManifest {
        ModelManifest {
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
            publisher,
            publish_date: chrono::Utc::now(),
            license: String::new(),
            mmproj: None,
            context_length: Some(4096),
        }
    }

    fn connected_peer(state: &SharedState) -> NodeId {
        let peer = NodeId([1; 32]);
        state.peer_id_map.insert(peer.clone(), vec![1; 4]);
        state.connected_node_ids.insert(peer.clone());
        peer
    }

    fn version(id: &str, hash: u8) -> ManifestVersion {
        ManifestVersion {
            model_id: ModelId(id.into()),
            manifest_hash: [hash; 32],
            context_length: Some(4096),
        }
    }

    /// A digest pulls what this node lacks from the peer that sent it, and
    /// only point to point — one arriving on gossip is nobody's catch-up.
    #[tokio::test]
    async fn a_digest_pulls_from_its_sender_what_this_node_lacks() {
        let state = state();
        let peer = connected_peer(&state);
        state
            .model_registry
            .register_manifest(manifest("held", 5, peer.clone()));
        let (tx, mut rx) = mpsc::channel(16);
        let digest = ManifestDigest {
            versions: vec![
                version("held", 5),
                version("unknown", 6),
                version("unknown.FULLBACKUP", 7),
            ],
        };

        on_digest(&state, &tx, Some(&peer), MessageTransport::Gossip, &digest);
        assert!(rx.try_recv().is_err(), "a gossiped digest is not answered");

        on_digest(&state, &tx, Some(&peer), MessageTransport::Direct, &digest);
        match rx.try_recv() {
            Ok(NetworkCommand::SendDirectMessage {
                target_peer_bytes,
                message: SwarmMessage::ManifestPull(pull),
                ..
            }) => {
                assert_eq!(target_peer_bytes, vec![1; 4]);
                assert_eq!(
                    pull.models,
                    vec![ModelId("unknown".into())],
                    "the held version and a copied-folder name are not pulled"
                );
            }
            _ => panic!("the digest's sender is asked for what this node lacks"),
        }
    }

    /// A pull is answered with what this node gossips, and nothing else: a
    /// model it holds no part of, or one it was never asked for, stays home.
    #[tokio::test]
    async fn a_pull_is_answered_with_only_the_asked_manifests_this_node_gossips() {
        let state = state();
        let peer = connected_peer(&state);
        let our_id = state.identity.node_id().clone();
        state
            .model_registry
            .register_manifest(manifest("ours", 5, our_id.clone()));
        state
            .model_registry
            .register_manifest(manifest("also-ours", 6, our_id));
        state
            .model_registry
            .register_manifest(manifest("theirs", 7, peer.clone()));
        let (tx, mut rx) = mpsc::channel(16);
        let pull = ManifestPull {
            models: vec![ModelId("ours".into()), ModelId("theirs".into())],
        };

        on_pull(&state, &tx, Some(&peer), MessageTransport::Gossip, &pull);
        on_pull(&state, &tx, Some(&peer), MessageTransport::Direct, &pull);
        let mut answered = Vec::new();
        while let Ok(Some(command)) =
            tokio::time::timeout(std::time::Duration::from_millis(200), rx.recv()).await
        {
            if let NetworkCommand::SendDirectMessage {
                message: SwarmMessage::ModelManifest(m),
                ..
            } = command
            {
                answered.push(m.id.0);
            }
        }
        assert_eq!(answered, vec!["ours".to_string()]);
    }

    /// How many manifests were sent within 200 ms.
    async fn answered(rx: &mut mpsc::Receiver<NetworkCommand>) -> usize {
        let mut count = 0;
        while let Ok(Some(command)) =
            tokio::time::timeout(std::time::Duration::from_millis(200), rx.recv()).await
        {
            if let NetworkCommand::SendDirectMessage {
                message: SwarmMessage::ModelManifest(_),
                ..
            } = command
            {
                count += 1;
            }
        }
        count
    }

    /// **A model is answered to one peer once, not once per pull** — the
    /// same pull repeated is answered nothing, so a ~40-byte request cannot be
    /// turned into ~35 KB a model every window (security review, 2026-10-17).
    /// A peer that is not connected is never answered: nothing could reach it.
    #[tokio::test]
    async fn a_model_is_answered_once_and_never_to_a_peer_not_connected() {
        let state = state();
        let peer = connected_peer(&state);
        let our_id = state.identity.node_id().clone();
        state
            .model_registry
            .register_manifest(manifest("ours", 5, our_id));
        let (tx, mut rx) = mpsc::channel(16);
        let pull = ManifestPull {
            models: vec![ModelId("ours".into())],
        };
        on_pull(&state, &tx, Some(&peer), MessageTransport::Direct, &pull);
        assert_eq!(answered(&mut rx).await, 1);
        on_pull(&state, &tx, Some(&peer), MessageTransport::Direct, &pull);
        assert_eq!(answered(&mut rx).await, 0, "answered minutes ago");

        let stranger = NodeId([2; 32]);
        state.peer_id_map.insert(stranger.clone(), vec![2; 4]);
        on_pull(
            &state,
            &tx,
            Some(&stranger),
            MessageTransport::Direct,
            &pull,
        );
        assert_eq!(answered(&mut rx).await, 0, "not connected");
    }
}
