# Architecture Rules

Invariants this codebase has paid to learn. Each heading is the rule; the text
under it is what to do.

**This file holds only what applies everywhere.** Everything else is
**path-scoped**: the rules for a subsystem load automatically the moment you
read a file in it, and cost nothing when you don't. The evidence — what a rule
replaced, what it was measured at, what a change must keep — lives in
`docs/invariants/`, read on demand.

| Working in | Rules that load | Evidence |
|---|---|---|
| `src/inference/{scheduler,router,pipeline}/` | `arch-scheduling.md` | `docs/invariants/scheduling.md` |
| `src/api/`, `src/error.rs`, `chat_template/` | `arch-api-surfaces.md` | `docs/invariants/api-surfaces.md` |
| `src/network/`, `src/daemon/dispatch/`, `src/pool/`, `src/crypto/` | `arch-network.md` | `docs/invariants/network.md` |
| `src/inference/{split,layers}/`, kernels, `vendor/candle*/` | `arch-inference.md` | `docs/invariants/inference.md` |
| `process_pool.rs`, `model_worker.rs`, `kv_*`, `auto_manage/` | `arch-worker-memory.md` | `docs/invariants/memory.md` |
| `src/daemon/state/`, `src/config/`, `src/credit/`, `src/storage/` | `arch-state-and-config.md` | `docs/invariants/state-and-config.md` |
| `frontend/` | `arch-frontend.md` | `docs/invariants/frontend.md` |
| `tests/`, `examples/` | `arch-guards-and-tests.md` | — |
| `frontend/`, `error.rs`, `auto_manage/`, `reference.rs` | `i18n.md` | — |

**These load on the Read tool ONLY.** Opening a file with `cat`, `sed -n`,
`head` or `grep` through Bash does NOT trigger them — measured 2026-09-16
against `.claude/logs/instructions-loaded.jsonl`. A session that reads through
Bash therefore has NO subsystem rules in context and gets no read-before-edit
check either, because that one is attached to the Edit tool. `research-gate.sh`
blocks a mutation whose governing rules never loaded; if you are reasoning about
a subsystem without opening its files, Read its rules file directly.

**A new user-facing string — from the frontend OR minted in Rust as an i18n key
— must be translated into all 21 locales. There is no English fallback.**

## SharedState Sub-Structs

SharedState is organized into 4 sub-structs. Always use the correct accessor —
never the bare root field:

- `state.events.*` — `activity_tx`, `dashboard_tx`, `activity_history`, `update_state`, `ws_tickets`
- `state.credits.*` — `credit_balance`, `pool_state`, `pool_registry`, `trust_manager`, `escrow_manager`, `private_mode`, …
- `state.models.*` — `acquisition_progress`, `hf_sources`, `wishlist`, `locked_shards`, `removed_by_user`, `shard_download_claims`, `disputed_shards`, …
- `state.metrics.*` — `node_stats`, `providers_config`, `swarm_capacity`, `hedge_tracker`, `peer_speed`, `bandwidth`, …

Two settings accessors that are read everywhere and must not be re-derived:

- **`SharedState::cfg()`** is the live config and the single answer to "what is
  this setting **now**". `state.config` is the BOOT SNAPSHOT — correct only for
  startup-only decisions. Reading it for a user-changeable setting is this
  repo's most-recurring defect (#281, four recurrences).
- **`api::dashboard_trust::classify`** is the single answer to "may this request
  be handed the API key automatically?". Never re-derive it with
  `addr.ip().is_loopback()` — that predicate is both too broad (a same-host
  reverse proxy satisfies it for a remote client) and too narrow (a container
  publish or Tailscale subnet router never does). Gotcha #195.

When adding a field, put it in the appropriate sub-struct unless it is accessed
by 10+ files across 3+ subsystem boundaries. **Per-field rules — what each map
means, who may write it, and what reads it through which accessor — are in
`arch-state-and-config.md`.**

→ `docs/invariants/state-and-config.md`

## One invariant, N paths — the recurring bug of this codebase

The single most repeated defect here is a **shared invariant implemented per
path**, where fixing the path in the bug report leaves the others broken. It
recurred *seven times* on 2026-07-25/26 alone. In every case a correct helper
already existed and one consumer didn't call it.

**Before fixing anything in the request/response path, enumerate the paths.**
There are more than you expect — three inference text sources, two OpenAI
response paths, four Anthropic ones, two Responses API ones. The full map is in
`docs/invariants/api-surfaces.md` § "One invariant, N paths"; `arch-api-surfaces.md`
carries the per-surface rules.

**A shared helper is not enough — put it where the caller cannot skip it.** A
helper nobody is *obliged* to call will eventually not be called. Three
escalating ways to make it obligatory, best first:

1. **Do it at the choke point, not in the callers.** Find the single place the
   value crosses the boundary and transform it there.
2. **Make the wrong call unrepresentable.** Make the context a required
   parameter, not an `Option` with a convenience wrapper that passes `None` —
   that wrapper disabled the template fallback on 6 of 7 paths (gotcha #171).
3. **Assert the property on the shared helper**, not once per path, so a new
   path inherits the coverage.

Only when none of those fit, fall back to a doc comment saying forgetting it is
the bug.

**Verify by running the request, not by reading the diff.** All seven passed
review. Where a report names a specific model, that model is part of the
reproduction (gotcha #168).

**Bad reply content is evidence about the PROMPT first, the output second.**
The `<|im_end|>` leak was chased across four releases as an output-scrubbing
problem and was a prompt problem (gotcha #169). Check
`grep "chat template failed" node.log` before touching any scrubber.

→ `docs/invariants/api-surfaces.md`

## Timeouts: bound what actually varies

A fixed deadline is only correct when the work behind it has a fixed size.
Where it does not, the constant silently becomes a **minimum-capability
requirement for the user** that nobody chose. Five instances were found in one
night (2026-07-27, gotcha #190) — one required a sustained 3.1 MB/s to ever
complete an update.

1. **Prefer an inactivity timeout to a total one.** `reqwest`'s `read_timeout`
   catches a stalled transfer just as fast and needs no guess about size or
   bandwidth. Use it for every download and every streamed proxy response.
2. **Where inactivity does not apply, scale by the input and cap it** —
   `pipeline::remote_generate::first_token_timeout(prompt_tokens)` is the
   shared helper; do not invent another rule.
3. **Generation gets no blanket deadline.** `generation_routes` in
   `api/server.rs` merge OUTSIDE the `TimeoutLayer`, and that merge MUST stay
   before the auth layer — pinned by `generation_routes_still_require_a_key`.
4. **When you change a limit, grep the whole path for other limits.** A budget
   is only as generous as the tightest ceiling above it, usually in another
   file, behind a comment that went stale before the code did.
5. **Read the comment against the code.** In four of the five, the comment
   reasoned about one quantity and the constant bounded another.
6. **The frontend is part of "the whole path".** A hardcoded 45 s
   `AbortController` in `compare.js` was the tightest ceiling on requests the
   daemon deliberately serves outside its own `TimeoutLayer`, and the number was
   baked into 21 translated strings so it was not even greppable.

→ `docs/invariants/api-surfaces.md` § Timeouts

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

## Additive Protocol Evolution (NETWORKING_PLAN cross-cutting)

Version-breaking network changes were a top adoption blocker: a node on vN
couldn't talk to vN±1 because a new/repurposed `SwarmMessage` variant failed to
deserialize on the other side. The rule that fixes this:

- **Never repurpose or remove a `SwarmMessage` variant** across a release, and
  never change an existing variant's wire shape incompatibly. Add a NEW variant
  instead and keep handling the old one.
- **Gate every new/optional message type on a negotiated feature.** Each node
  advertises the features it implements in `NodeCapability::features` (a `u64`
  bitfield, `swarmllm_types::features`). A sender MUST check the recipient
  advertises the matching bit before sending the new variant — an older node
  advertises `0` and is correctly skipped, so it is never handed something it
  can't decode. `features::supports(advertised, needed)` is the check;
  `features::ALL` is what this build advertises (set in `health/monitor.rs`).
  The Phase-1 relay (`features::RELAY`) is the reference example: see
  `network/manager/relay.rs::target_supports_relay`.
- **`PROTOCOL_VERSION`** (`swarmllm_types`) is the wire epoch — bump ONLY on a
  genuinely breaking change (which the first rule forbids without a fallback),
  NOT for additive feature bits. Adding a `features` bit does not bump it.
- New `NodeCapability` fields MUST be `#[serde(default)]` so older nodes'
  announcements still deserialize.