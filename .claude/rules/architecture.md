# Architecture Rules

## SharedState Sub-Structs

SharedState is organized into 4 sub-structs. Always use the correct accessor:

- `state.events.activity_tx` — NOT `state.activity_tx`
- `state.events.dashboard_tx` — NOT `state.dashboard_tx`
- `state.credits.credit_balance` — NOT `state.credit_balance`
- `state.credits.pool_state` — NOT `state.pool_state`
- `state.models.acquisition_progress` — NOT `state.acquisition_progress`
- `state.models.hf_sources` — NOT `state.hf_sources`
- `state.metrics.node_stats` — NOT `state.node_stats`
- `state.metrics.providers_config` — NOT `state.providers_config`

When adding new fields to SharedState, put them in the appropriate sub-struct unless they're accessed by 10+ files across 3+ subsystem boundaries.

## Event System

All events flow through `state.events.activity_tx` (ActivityEvent). Use the builder:
```rust
state.emit_activity(
    ActivityEvent::new("category", "kind", format!("message"))
        .with_model(model_id)
        .with_toast("info", 4000)
);
```

For dashboard refresh signals, use `state.events.dashboard_tx`:
- `DashboardSignal::ModelsChanged` — after shard download/load/prune/delete
- `DashboardSignal::PeersChanged` — after peer connect/disconnect
- `DashboardSignal::UpdateAvailable(info)` — after update check

There are ONLY 2 broadcast channels. Do NOT add new ones.

## Frontend Event Handling

All WS events are handled by `_handleActivityEvent()` in notifications.js. Do NOT:
- Add new WS message types (use activity_event with a new `kind`)
- Add direct `showToast()` calls for backend events (set `toast_level` on the ActivityEvent instead)
- Add direct `logActivity()` calls from WS handlers (everything goes through `_handleActivityEvent`)

## Frontend Storage

All storage keys are registered as constants on `App` in state.js (e.g., `App.MODEL_SORT_KEY`). Do NOT use raw string literals for localStorage/sessionStorage keys.

## Frontend Data Fetching

Use `App.data.loadModels()` and `App.data.loadStats()` for model/stats data. Do NOT make independent `authFetch('/api/admin/models')` calls from components — this bypasses the dedup cache.
