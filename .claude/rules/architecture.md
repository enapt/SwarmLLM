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

## Centralised Wire-Format Helpers

These helpers exist as the single source of truth for invariants that
silently break at the wire if duplicated:

- **`network::protocol::build_layer_forward_aad`** — encryption AAD
  bytes for `LayerForward` envelopes. Both encrypt
  (`network/manager/tensors.rs`) and decrypt
  (`decode_layer_forward_encrypted`) MUST go through it. Adding a
  new authenticated field to `LayerForward` means extending this
  helper, not appending bytes on the encrypt side.
- **`daemon::dispatch::gossip_timestamp_fresh`** — one-sided
  staleness check for `u64`-millisecond regional gossip
  (`RegionShardSummary`, `ModelDemandGossip`). New gossip types use
  this; do NOT re-implement `if ts > now + skew { drop } else if
  now - ts > max_age { drop }` per gotcha #44.
- **`credit::ledger::check_signed_freshness`** — one-sided
  staleness check for `chrono::DateTime<Utc>`-typed signed messages
  (balance reports, credit transactions). Constants
  `CLOCK_SKEW_TOLERANCE_SECS` / `BALANCE_REPORT_MAX_AGE_SECS` are
  `pub(crate)` so all credit-typed callers share the same window
  (gotcha #32).

## Cross-feature compile checks

`cargo check` with default features does NOT compile the `llama` cfg
path. Visibility-tightening or cross-file refactors that touch
`pipeline/dsd.rs` or any spec/llama-gated code in
`pipeline/speculative.rs` MUST verify with
`cargo check --features llama` before push. Pre-push hook only runs
default-features `cargo check`. R91 caught a regression introduced
in R90 that default-features had silently let through.
