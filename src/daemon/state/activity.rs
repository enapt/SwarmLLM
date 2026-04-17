/// Unified activity event for the dashboard — the single event bus.
/// Pushed over WebSocket as `activity_event` messages. Replaces the former
/// separate `prune_event`, `lan_peer_discovered`, and `system_notification`
/// WS message types (all now flow through this struct).
#[derive(Clone, Debug, serde::Serialize)]
pub struct ActivityEvent {
    /// Event category for frontend grouping/filtering.
    pub category: &'static str,
    /// Machine-readable event kind.
    pub kind: &'static str,
    /// Human-readable description (English; frontend may i18n-override).
    pub message: String,
    /// Optional model ID for per-model ticker routing.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_id: Option<String>,
    /// Optional model display name.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_name: Option<String>,
    /// Optional peer/node ID (short hex).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub node_id: Option<String>,
    /// Optional numeric detail (e.g. shard index, credit amount, latency).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail_num: Option<i64>,
    /// Optional string detail (e.g. reason, source, error message).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail_str: Option<String>,
    /// If set, the frontend shows a toast at this level ("success", "info", "warning", "error").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub toast_level: Option<&'static str>,
    /// Toast auto-dismiss duration in ms (default 5000 if toast_level is set but this is None).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub toast_duration_ms: Option<u32>,
    /// Shard index (for prune/shard events that need structured data beyond detail_num).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub shard_index: Option<u32>,
    /// Bytes freed (for prune events).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub freed_bytes: Option<u64>,
    /// Holder count before an operation (for prune events).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub holder_count_before: Option<usize>,
    /// Holder count after an operation (for prune events).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub holder_count_after: Option<usize>,
    /// Remaining local shards after an operation (for prune events).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub remaining_local_shards: Option<u32>,
    /// ISO 8601 timestamp for events that need a backend-authoritative time (e.g. prune).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timestamp: Option<String>,
}

impl ActivityEvent {
    /// Create an event with only the core fields; all extended fields default to None.
    pub fn new(category: &'static str, kind: &'static str, message: String) -> Self {
        Self {
            category,
            kind,
            message,
            model_id: None,
            model_name: None,
            node_id: None,
            detail_num: None,
            detail_str: None,
            toast_level: None,
            toast_duration_ms: None,
            shard_index: None,
            freed_bytes: None,
            holder_count_before: None,
            holder_count_after: None,
            remaining_local_shards: None,
            timestamp: None,
        }
    }

    /// Builder: set model_id.
    pub fn with_model(mut self, model_id: impl Into<String>) -> Self {
        self.model_id = Some(model_id.into());
        self
    }

    /// Builder: set model_name.
    pub fn with_model_name(mut self, name: impl Into<String>) -> Self {
        self.model_name = Some(name.into());
        self
    }

    /// Builder: set node_id.
    pub fn with_node(mut self, node_id: impl Into<String>) -> Self {
        self.node_id = Some(node_id.into());
        self
    }

    /// Builder: set detail_num.
    pub fn with_detail_num(mut self, n: i64) -> Self {
        self.detail_num = Some(n);
        self
    }

    /// Builder: set detail_str.
    pub fn with_detail_str(mut self, s: impl Into<String>) -> Self {
        self.detail_str = Some(s.into());
        self
    }

    /// Builder: request a frontend toast.
    pub fn with_toast(mut self, level: &'static str, duration_ms: u32) -> Self {
        self.toast_level = Some(level);
        self.toast_duration_ms = Some(duration_ms);
        self
    }

    pub fn with_shard_index(mut self, idx: u32) -> Self {
        self.shard_index = Some(idx);
        self
    }

    pub fn with_freed_bytes(mut self, bytes: u64) -> Self {
        self.freed_bytes = Some(bytes);
        self
    }

    pub fn with_holders(mut self, before: usize, after: usize) -> Self {
        self.holder_count_before = Some(before);
        self.holder_count_after = Some(after);
        self
    }

    pub fn with_remaining_local(mut self, n: u32) -> Self {
        self.remaining_local_shards = Some(n);
        self
    }

    pub fn with_timestamp(mut self, ts: impl Into<String>) -> Self {
        self.timestamp = Some(ts.into());
        self
    }
}

/// Signal enum for dashboard-targeted WS pushes.
/// Consolidates peer_list_changed, models_changed, and update_available
/// into a single broadcast channel to reduce channel proliferation.
#[derive(Clone, Debug)]
pub enum DashboardSignal {
    /// Peer registry changed — push full peer list to dashboard.
    PeersChanged,
    /// Model state changed (shard download, load, prune) — frontend should re-fetch models.
    ModelsChanged,
    /// Software update available — push banner to dashboard.
    UpdateAvailable(crate::update::UpdateInfo),
}

/// Cached info about a locally loaded model (lock-free reads).
#[derive(Clone, Debug)]
pub struct LoadedModelInfo {
    pub name: String,
    pub size_bytes: u64,
    /// EOS token IDs loaded from GGUF metadata.
    pub eos_tokens: Vec<u32>,
    /// Chat template from GGUF `tokenizer.chat_template` metadata (Jinja2 format).
    pub chat_template: Option<String>,
    /// BOS token string from GGUF metadata.
    pub bos_token: String,
    /// EOS token string from GGUF metadata.
    pub eos_token: String,
}
