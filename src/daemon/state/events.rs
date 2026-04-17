use std::collections::VecDeque;
use std::sync::Arc;

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
}

impl EventBus {
    /// Remove stale `model_loaded` history entries for a model (e.g., after load/unload/delete).
    pub fn clear_model_load_history(&self, model_id: &str) {
        let mut history = self.activity_history.lock();
        history.retain(|e| !(e.kind == "model_loaded" && e.model_id.as_deref() == Some(model_id)));
    }
}
