use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Instant;

use dashmap::DashMap;
use tokio::sync::{broadcast, RwLock};

use super::activity::{ActivityEvent, DashboardSignal};

/// Event bus: activity events + dashboard signals + update state.
pub struct EventBus {
    pub activity_tx: broadcast::Sender<ActivityEvent>,
    /// Rolling buffer of recent events — guarded by a fast parking_lot mutex so
    /// emit_activity() does not contend on a poisoning-aware std mutex.
    pub activity_history: parking_lot::Mutex<VecDeque<ActivityEvent>>,
    pub dashboard_tx: broadcast::Sender<DashboardSignal>,
    pub update_state: Arc<RwLock<crate::update::UpdateState>>,
    /// Short-lived single-use tickets for WebSocket upgrade authentication.
    /// Issued by `POST /api/admin/ws-ticket` (Bearer-authed), consumed
    /// atomically by the `/api/admin/ws` upgrade handler via
    /// `DashMap::remove()`. Value is the issuance `Instant`; handler
    /// rejects tickets older than `WS_TICKET_TTL`. Browsers cannot set
    /// an `Authorization` header on WebSocket upgrades, so this ticket
    /// round-trip is how we keep Bearer-only auth on the WS endpoint
    /// instead of exempting it on loopback.
    pub ws_tickets: DashMap<String, Instant>,
    /// Peers advertising a newer version than ours (`update::PeerVersionWatch`).
    /// Written by the capability-gossip handler through [`Self::note_peer_version`].
    pub peer_versions: parking_lot::Mutex<crate::update::PeerVersionWatch>,
    /// Wakes the update checker to bring its next GitHub check forward. A
    /// `Notify`, not a broadcast channel: there is one listener and the
    /// signal carries nothing — the checker decides what, if anything, is new.
    pub update_nudge: tokio::sync::Notify,
}

impl EventBus {
    /// A peer advertised `version` in its capability gossip. When enough distinct
    /// peers agree on a believably newer version, wake the update checker so the
    /// node learns about a release from the swarm in minutes rather than waiting
    /// for its hourly GitHub poll — observed 2026-07-29 with the anchor on 0.3.47
    /// and two connected nodes still on 0.3.46 with no way to notice.
    ///
    /// The gossiped version is self-attested and never load-bearing: it only
    /// brings the check forward; GitHub's signed release remains the sole source
    /// of what is installed. See `update::PeerVersionWatch` for the guards.
    pub fn note_peer_version(&self, peer: &crate::types::NodeId, version: &str) {
        let current = env!("CARGO_PKG_VERSION");
        let corroborated = self
            .peer_versions
            .lock()
            .observe(peer.clone(), version, current);
        if let Some(v) = corroborated {
            tracing::info!(
                peers_version = %v,
                ours = current,
                "Several peers are running a newer version — the update check is being \
                 brought forward (GitHub still decides what, if anything, is installed)"
            );
            self.update_nudge.notify_one();
        }
    }

    /// Remove stale `model_loaded` history entries for a model (e.g., after load/unload/delete).
    pub fn clear_model_load_history(&self, model_id: &str) {
        let mut history = self.activity_history.lock();
        history.retain(|e| !(e.kind == "model_loaded" && e.model_id.as_deref() == Some(model_id)));
    }
}
