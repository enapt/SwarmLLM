use std::sync::Arc;
use std::time::Duration;

use tokio::sync::watch;

use super::session::SessionManager;

/// Session eviction interval (10 minutes).
const SESSION_EVICTION_INTERVAL: Duration = Duration::from_secs(600);

/// Maximum session age before eviction (10 minutes).
const MAX_SESSION_AGE: Duration = Duration::from_secs(600);

/// Run the background session cleanup task.
///
/// SEC-M19: This module is named `key_rotation` but currently only performs session eviction.
/// TODO: Add actual key rotation — periodically re-establish sessions with active peers
/// using fresh ephemeral keys, or rename this module to `session_cleanup`.
///
/// - Evicts stale encryption sessions every 10 minutes.
/// - Runs until shutdown signal.
pub async fn run_key_rotation(
    session_manager: Arc<SessionManager>,
    mut shutdown_rx: watch::Receiver<bool>,
) {
    let mut eviction_interval = tokio::time::interval(SESSION_EVICTION_INTERVAL);
    eviction_interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

    tracing::info!("Key rotation task started");

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
                if before != after {
                    tracing::info!(
                        evicted = before - after,
                        remaining = after,
                        "Evicted stale encryption sessions"
                    );
                }
            }
        }
    }
}
