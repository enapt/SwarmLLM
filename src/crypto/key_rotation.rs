use std::sync::Arc;
use std::time::Duration;

use tokio::sync::{mpsc, watch};

use super::session::SessionManager;
use crate::types::{EphemeralKeyExchange, NetworkCommand, NodeId, SwarmMessage};

/// Session eviction interval (10 minutes).
const SESSION_EVICTION_INTERVAL: Duration = Duration::from_secs(600);

/// Maximum session age before eviction (10 minutes).
const MAX_SESSION_AGE: Duration = Duration::from_secs(600);

/// Key rotation interval — re-key active sessions every 10 minutes for
/// forward secrecy. Static DH sessions are replaced with ephemeral ones.
const KEY_ROTATION_INTERVAL: Duration = Duration::from_secs(600);

/// Run the background key rotation and session cleanup task.
///
/// - Evicts stale encryption sessions every 10 minutes.
/// - Initiates ephemeral ECDH re-keying with active peers every 10 minutes
///   for forward secrecy. Old static-key sessions are replaced with
///   ephemeral ones derived from fresh keypairs.
/// - Runs until shutdown signal.
pub async fn run_key_rotation(
    session_manager: Arc<SessionManager>,
    network_tx: mpsc::Sender<NetworkCommand>,
    local_node_id: NodeId,
    shared_state: Arc<crate::daemon::SharedState>,
    mut shutdown_rx: watch::Receiver<bool>,
) {
    let mut eviction_interval = tokio::time::interval(SESSION_EVICTION_INTERVAL);
    eviction_interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

    let mut rotation_interval = tokio::time::interval(KEY_ROTATION_INTERVAL);
    rotation_interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    // Skip the first tick (don't rotate immediately on startup)
    rotation_interval.tick().await;

    tracing::info!("Key rotation task started (eviction + ephemeral re-keying)");

    loop {
        tokio::select! {
            _ = shutdown_rx.changed() => {
                if *shutdown_rx.borrow() {
                    tracing::info!("Key rotation task shutting down");
                    break;
                }
            }
            _ = eviction_interval.tick() => {
                let before = session_manager.session_count();
                session_manager.evict_stale(MAX_SESSION_AGE);
                let after = session_manager.session_count();
                tracing::debug!(
                    active_sessions = before,
                    stale_evicted = before - after,
                    remaining = after,
                    "DIAG: key rotation tick (eviction)"
                );
            }
            _ = rotation_interval.tick() => {
                // Initiate ephemeral re-keying with all active peers
                let peers = session_manager.active_peers();
                if peers.is_empty() {
                    continue;
                }
                tracing::info!(
                    active_sessions = peers.len(),
                    rekey_initiated = peers.len(),
                    "DIAG: key rotation tick (re-keying)"
                );
                for peer in &peers {
                    let ephemeral_pub = session_manager.initiate_ephemeral_exchange(peer);
                    let msg = SwarmMessage::EphemeralKeyExchange(EphemeralKeyExchange {
                        session_id: uuid::Uuid::new_v4(),
                        node_id: local_node_id.clone(),
                        ephemeral_pubkey: ephemeral_pub,
                        is_initiator: true,
                    });
                    // Send directly to the target peer via request_response (not gossip).
                    // Gossip broadcast silently dropped EphemeralKeyExchange (no topic match).
                    // Direct send also ensures the recipient can authenticate the sender
                    // via the request_response protocol's peer identity.
                    let target_bytes = shared_state
                        .peer_id_map
                        .get(peer)
                        .map(|r| r.value().clone());
                    if let Some(target) = target_bytes {
                        if let Err(e) = network_tx.try_send(NetworkCommand::SendDirectMessage {
                            target_peer_bytes: target,
                            message: msg,
                        }) {
                            tracing::debug!(
                                peer = %peer,
                                error = %e,
                                "Failed to send ephemeral key exchange (channel full)"
                            );
                        }
                    }
                }
            }
        }
    }
}
