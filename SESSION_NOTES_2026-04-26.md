# Overnight session — 2026-04-25 → 2026-04-26

> Self-paced loop run while user slept. ~7 cycles, ~3 hours of compute.
> 95 files changed, +1271/-423. Every commit cargo fmt + clippy clean,
> 821 lib tests passing throughout (was 816 at session start).

## Headline

Two things landed that move the needle:

1. **zstd compression on `WIRE_TAG_PREFIX_KV`** (top item in
   `next_steps.md`) shipped behind `NetworkConfig::prefix_kv_compression`
   (default off). Reuses existing tensor-compression helpers; receivers
   always decompress regardless of the flag, so single-peer flip is
   safe. 5 new round-trip tests. Awaiting WAN bench to decide
   default-on. **Commit `c10956e`.**

2. **macOS CI extended from build-only to test+clippy**
   (closes `next_steps.md` § 4 release-hygiene gap). `cargo test --lib
   --bins` and clippy now run on `macos-15` in addition to Linux;
   integration tests stay Linux-only with explicit guards until the
   first macOS failure decides whether to fix or skip-list per test.
   **Commit `ccfbf14`.**

Plus one critical-bug fix: `App.notifications.toast()` was called 4
times in `responses.js` but the function doesn't exist — silently
no-op'd cancel and delete buttons. Replaced with `showToast` and
folded the dup into a `_action` helper. **Commit `36af419`.**

And one critical security fix: `previous_response_id` validation was
only running on the local-inference path, not the cloud-proxy or
Anthropic-bridge paths, so a 1 MB junk id could reach upstream
provider request bodies. Hoisted into a new `validate_responses_ingress`
that runs first and also caps `instructions` / `user` / `model` /
`truncation` / `service_tier` / `metadata`. **Commit `c9acbfc`.**

And one critical logic bug: `escrow.rs cleanup_expired` was
incrementing `count` BEFORE the persist+refund flow could rollback,
so the "Cleaned up N expired escrows" log overcounted by every
rolled-back attempt. **Commit `d1d8185`.**

## Per-cycle summary

| # | Commit | Scope | Headline |
|---|---|---|---|
| C1 | `36af419` | Sweep R54 (12 fixes) | Critical: broken `responses.js .toast()` + dead code + dup |
| C2 | `c9acbfc` | Sweep R55 (16 fixes) | Critical: validation hoisting + max_tokens cap + 5-site `resolve_peer_id_bytes` helper |
| C3 | `c10956e` | Code: zstd prefix-KV | Top deferred item shipped flag-gated, 5 new tests |
| C4 | `ccfbf14` | Sweep R56 (10 fixes) + macOS CI | Hot-path syscall savings, `hkdf_sha256_derive_32` helper, missing i18n key, **macOS test+clippy in CI** |
| C5 | `d4b00c9` + `6dcecf3` | CHANGELOG + Sweep R57 + Book deep review (16 fixes) | New `api/responses.md` page filling a major doc gap; libp2p deferred → landed marker; many stale refs in book/ |
| C6 | `b729a92` + `d1d8185` | Items 14/17/18 research + Sweep R58 (4 fixes) | New `items_14_17_18_research.md` with concrete Mooncake > HELIOS > Mirror ranking; **critical escrow `count++` bug** |
| C7 | (in flight) | Sweep R59 inference primitives | Pending |

## What's now done that wasn't before

- **`docs/plans/next_steps.md` § 1 (zstd-prefix-KV)** — ✅ shipped
  flag-gated. Documented in `next_steps.md`, `distributed_inference_speedup.md`,
  `book/configuration/reference.md`, `book/architecture/networking.md`.
- **`docs/plans/next_steps.md` § 4 (macOS CI matrix)** — ✅ shipped.
- **`docs/plans/next_steps.md` § 3 (Items 14/17/18 research)** —
  ✅ assessment written. Headline: **17 (Mooncake disaggregated
  prefill/decode) > 18 (HELIOS per-token early-exit) > 14 (Mirror
  Spec)** by signal-to-effort. All gated on WAN bench. Item 14
  inherits SWIFT (Item 6) blocker; defer until SWIFT lands.
- **Responses API documentation** — new `book/src/api/responses.md`
  page. Covers all 5 endpoints, routing decision tree, capabilities,
  validation caps, dashboard, deferred items.
- **CHANGELOG `[Unreleased]` post-v0.1.0** — section added. Marked
  v0.1.0's stale "libp2p 0.55→0.56 deferred" as ✅ landed.

## What's still gated on WAN bench (unchanged)

- WAN bench itself (needs hardware — 2 daemons, 2 regions).
- zstd-prefix-KV default-on decision.
- Pick of which Item 14/17/18 to actually build.
- DSD multi-segment (Item 12) end-to-end measurement.
- Item 13 activation-compression default-on.

## Refactor helpers extracted (single-source contracts)

| Helper | Location | Replaces |
|---|---|---|
| `SharedState::resolve_peer_id_bytes` | `src/daemon/state/mod.rs` | 5 sites in inference/pipeline/{distributed,remote_generate,speculative,dsd,tensor_parallel}.rs |
| `crypto::hkdf_sha256_derive_32` | `src/crypto/mod.rs` | 3 sites in `provider_keys`, `gossip_seal`, `session` |
| `network::helpers::is_non_public_ipv4_bytes` | `src/network/helpers.rs` | Hand-rolled inline RFC check that was MISSING link-local + CGN/Tailscale ranges |
| `auto_manage::manager::read_shard_pins` | `src/model/auto_manage/manager.rs` | 2 sites in `scoring.rs`, `prune.rs` |
| `cli::discover_model` | `src/cli/mod.rs` | 14-line block dup'd in `bench.rs` + `chat.rs` |
| `cli::bench::tokens_per_sec` | `src/cli/bench.rs` | 2 sites |
| `pub const SWARMLLM_GITHUB_REPO` | `src/update.rs` | Bare `"enapt/SwarmLLM"` literal in 2 sites |
| `BASELINE_LAYER_COUNT`, `UNKNOWN_COMPUTE_MS` | `src/inference/scheduler/parallax.rs` | Magic 32.0 + misnamed `DEFAULT_TOKENS_PER_SEC` across 2 files |
| `validate_responses_ingress` + 6 caps | `src/api/openai/responses/mod.rs` | Closes 3 unbounded ingress fields + hoists prior-id validation before all 3 routing branches |
| `MAX_TOKENS_HARD_CAP = 32768` | `src/api/anthropic/mod.rs` | Bare u32 max_tokens accepted (silently clamped, raw-forwarded to upstream) |
| `build_response_skeleton` (R54) | `src/api/openai/responses/mod.rs` | 4 sites in `stream`, `background`, `anthropic_bridge`, `mod` |
| `post_openai_compat` (R54) | `src/api/providers.rs` | 40-line dup in 2 sites |

## Remaining sweep deferrals (logged in `.claude/sweep-log.jsonl`)

These are real but each needed a focused refactor I didn't want to
rush at 2 AM:

- `src/credit/escrow.rs apply_credit_direct` integration (R56) —
  EscrowManager bypasses the ledger's centralized credit-write
  helper; needs Arc<RwLock<CreditBalance>> threading or a new
  variant.
- `src/inference/split/prefix_cache.rs` write-lock-on-every-cache-hit
  (R56) — only bumps a `last_hit` timestamp; should be AtomicU64.
- `src/inference/router/local_exec.rs` double `loaded_model_info`
  RwLock acquire per batch request (R56).
- `src/inference/kv_cache.rs check_multi_turn_reuse` O(peers)
  Vec::contains (R56) — caller in `router/mod.rs:581` builds a fresh
  Vec from the DashMap on every request.
- `src/api/admin.rs GET /api/admin/responses` O(N) full-table scan
  (R55) — pre-v1 fine, needs a redb-iterator early-exit before scale.
- `src/inference/pipeline/{speculative,dsd,remote_generate}.rs`
  larger dedup (eligible-guards, accept-reject loop, send_verify
  scaffolding) (R55) — touches inference hot path with subtly
  divergent control flow per site.
- `src/pool/manager/mod.rs handle_leave_pool` rate-limit-inside-read-lock
  (R58) — TOCTOU concern; needs a focused look at all 4 pool handlers
  for ordering consistency.

## Stats

- **Tests**: 816 → 821 (+5 from new prefix-KV roundtrip tests).
- **Sweep log**: 527 → 597+ entries (50+ new findings — fixed,
  deferred, or wontfix with rationale).
- **Frontend**: i18n keys 1012 → 1014 (added
  `dashboard.contribution_tier_maximum` across 21 locales).
- **Repo size**: +1271 / -423 across 95 files.

## What to look at first when you wake up

1. **`docs/plans/items_14_17_18_research.md`** — the decision-basis
   write-up for the deferred research candidates. Pick one (or
   neither) to bench after WAN.
2. **`CHANGELOG.md` `[Unreleased]`** — the working changelog for
   what's accumulated since v0.1.0.
3. **`.claude/sweep-log.jsonl`** — every finding with file/line/kind/
   summary/status. Search for `"status":"deferred","date":"2026-04-26"`
   to see what was knowingly left for later.
4. **`docs/book/src/api/responses.md`** — new docs page. Read it
   end-to-end to make sure the routing decision tree matches your
   mental model.
