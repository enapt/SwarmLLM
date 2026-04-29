# Changelog

All notable changes to SwarmLLM are documented here.

## [Unreleased] — post-v0.1.0

Working changelog for commits after the v0.1.0 tag. Will roll into the
next tagged release.

### Deferred-item follow-ups (R72/R75 leftovers, 2026-04-29)

Three commits cleaning up the structurally-deferred items the sweep
arc surfaced:

- **Worker crash-loop backoff** (`bab361c`) — `ModelProcessPool` now
  tracks per-model spawn failures; arriving requests during the
  cooldown window get `ServiceUnavailable` instead of waiting for
  `WORKER_CONNECT_TIMEOUT_SECS=30s` per attempt. Backoff steps
  1→2→4→8→16→32→60 s, reset on first successful spawn.
- **Cancellation token in inference loop** (`bab361c`) —
  `InferenceRequest.cancel: Option<Arc<AtomicBool>>` (`#[serde(skip)]`),
  `SharedState.cancel_signals` map keyed by an opaque token via
  the `x-swarmllm-cancel-token` HTTP header. Pipeline checks
  `request.is_cancelled()` per-token. Wired end-to-end through
  `responses/background.rs` so `/v1/responses/{id}/cancel`
  actually interrupts in-flight inference within one forward.
- **Stop-sequence KV truncate** (`9f9f22e`) —
  `pipeline/distributed.rs` sends a finalising
  `LayerForward(truncate_kv_to=ptc, activations=[])` to every remote
  segment after a stop string fires on a session-keyed request.
  Without this the next session turn would see the stop tokens
  still in the remote KV and produce contaminated output.
- **Escrow refund on inference failure** (`9f9f22e`) —
  `finalize_request` now calls `refund_escrow` on the `Err` arm.
  Previously `refund_escrow` had no production caller; failed
  requests left credits locked until the cleanup tick. Credit
  enforcement isn't gating users yet, but the bookkeeping is solid
  for when it is.
- **Pre-release surface trim** (`1d7eb39`) — every internal module
  in `src/lib.rs` now `#[doc(hidden)]`; `api`, `config`, `error`,
  `types`, `update` are the documented stable API. New
  `tests/integration/end_to_end.rs` exercises the full HTTP +
  shutdown lifecycle (`#[ignore]`'d, ~10 s).

C1 (binary signing) stays open with a fully-researched options write-
up at `memory/signing_options.md` — recommendation is **minisign**;
landing it requires a key-custody decision from the maintainer.

### Sweep arc R69 → R75 (2026-04-29 / 2026-04-30)

Long-running self-managed sweep covering: post-audit follow-ups (R69),
hot-path performance (R70), concurrency + lifecycle (R71), error
recovery + resilience (R72), observability + operability (R73), API +
wire-format correctness (R74), and pre-release readiness (R75). Each
round spawned 4–5 parallel review agents, applied auto-fixable
findings in batched commits, and surfaced architectural items for
follow-up. ~20 commits, broad coverage:

- **Performance** — `PendingLayerResultGuard` shared between dsd /
  speculative; `req_id_str` and `SamplingParams` no longer cloned per
  decode tick in `step_decode_pool`; per-token `default_eos` HashSet
  hoisted out of the decode loop; `verify_tokens.to_vec()` dropped
  from spec-round LayerForward; admin/stats and websocket
  build_stats_message no longer hold RwLock guards across blocking
  work; `health/monitor.rs` switched from `block_in_place` to
  `spawn_blocking` for the 30s sysinfo refresh; frontend
  notifications + dashboard render skips on equal state +
  sessionStorage debounced.
- **Concurrency** — `batch_scheduler_loop` and
  `pipeline_stream::spawn_accept_loop` + `handle_inbound_stream` now
  observe the watch-channel shutdown signal; PendingLayerResultGuard
  applied to speculative.rs's two leak sites; `apply_pipeline_guard`
  RAII catches panic-induced leaks of `active_pipelines` /
  `active_count`; `dashboard_rx` Lagged path now sends a re-sync
  message instead of silently dropping.
- **Resilience** — `apply_update` collapsed into a single signature
  that always re-checks version; HF probe wrapped with the
  NETWORK_RETRY_DELAYS exponential backoff (was a bare `.await?`);
  pool slaves now forward the `penalty_serve_failure` to the master
  via the same path as success-case spends (was hitting the slave's
  local balance and gating its own future requests); update.rs
  `info.downloaded = true` only set when staging path is the
  preferred (same-filesystem) location, not the temp_dir EPERM
  fallback.
- **Observability** — credit spend logs promoted from debug to info
  with `DIAG:` prefix (the earn side already had it; the spend side
  was dark); `escrow_held` + `escrow_pending_count` exposed on
  `/api/admin/credits`; `subnet_counts` retain on register so a NAT
  flip doesn't pile a peer into multiple buckets.
- **API correctness** — `SwarmError::InsufficientCredits` returns a
  distinct `insufficient_credits` error_type instead of the
  `rate_limit_error` mismatch (SDK retry logic was treating credit
  stops as rate-limits); `SwarmError::Network` maps to 502 +
  `network_error`; streaming finish_reason='error' replaced with
  'stop' (not a valid OpenAI value); `/v1/models` `created` is the
  manifest publish_date timestamp not Utc::now() (was unstable per
  call); `sse.rs::data_frame` emits a structured error on
  serialization failure instead of silently emitting an empty
  `data:` line; spec_logits doc-comment in swarmllm-types matches
  gotcha #29's γ+1 contract.
- **Hygiene** — `WorkerOptions` struct bundles run_worker's 7 runtime
  knobs (was 13 args under `#[allow(too_many_arguments)]`);
  `SplitLoadOptions` for split/loader; `speculative::softmax`
  wrapper deleted; `// SYNC:` comment dropped (gotcha #18 is the
  canonical 3-path warning); inline "deferred ChaCha encryption"
  paragraphs collapsed to one ARCHITECTURE.md pointer; mmap SAFETY
  comment expanded with candle's qtensor_from_ggml copy-semantics
  proof.
- **Release prep** — i18n files now sorted alphabetically by key for
  reliable parity audits (1015 keys + 2 metadata = 1017 entries per
  locale, all 21 languages confirmed in parity); `default.toml`
  documents the speculative-fast-path ChaCha bypass next to
  `enable_encryption = true`; `yamux_substream` CI tests now run on
  macOS (the Linux-only guard was a copy-paste from the
  multi-process integration tests).

890 lib tests pass throughout; clippy clean both feature sets;
pre-push hooks passing on every commit.

### Performance

- **`PrefixCache` lookup hit no longer needs a write lock** —
  `Entry::last_hit` is now an `AtomicU64` driven by a per-cache logical
  clock; the read-lock walk records the hit via a `Relaxed` store
  instead of upgrading to write. Removes the entire write-acquire from
  every cache hit and its sleep-based test fixture. Touches
  `src/inference/split/prefix_cache.rs` only.
- **`local_exec` holds `loaded_model_info` once per batch item** — was
  acquired twice per request (chat-template prompt + stop-string list).
  Now extracted from a single guard.
- **`KvCacheManager::check_multi_turn_reuse` takes `&HashSet<NodeId>`
  instead of `&[NodeId]`** — the per-holder `contains()` was O(peers).
  Caller in `inference/router/mod.rs` collects the DashMap directly
  into a HashSet so the conversion is paid once.
- **`GET /api/admin/responses` streams + bounded heap** — new
  `Database::for_each_json` helper streams JSON-encoded records without
  materialising a full Vec; the listing endpoint maintains a min-heap
  of size `≤ limit` ordered by `created_at`. Memory now O(limit)
  instead of O(total_records); only survivors get the full preview JSON
  built.

- **zstd compression on `WIRE_TAG_PREFIX_KV`** — flag-gated via
  `NetworkConfig::prefix_kv_compression` (default off). Send-side reuses
  the existing tensor-compression helpers and falls back to raw when the
  compressed form isn't smaller. Receivers always decompress regardless
  of the flag, so flipping it on a single peer doesn't require a
  coordinated upgrade. Expect 30–50% wire reduction on KV snapshots
  (zero-padded regions compress well); WAN bench will decide default-on.
- **Hot-path syscall savings** — gated per-layer `Instant::now()` in
  `split/executor.rs` behind `tracing::enabled!(TRACE)` (28 syscalls/token
  on a 28-layer model wasted at default log level), and same for
  `pipeline/distributed.rs` per-token `fwd_start` behind `enabled!(DEBUG)`
  with the matching DIAG emit dropped to debug-level for consistency.
- **Worker IPC capacity** — bumped reader→main channel from 16 to 64 in
  `model_worker.rs`. The admit-coalescing drain loop pulls 16/tick and a
  single decode tick can be 100–500 ms on CPU 7B; bursts would block the
  reader on `.send()` and delay the cross-node `PrefixFetchResult`
  fast-path short-circuit.

### Security & validation

- **Black-hat sweep 2026-04-29** — six parallel adversarial reviewers
  (network, crypto/auth, pool/credit, HTTP API, inference path, supply
  chain). 7 commits landed (`f4ff02b..fff42b4`); see
  `memory/audit_2026-04-29.md` for the full rollup. Highlights:
  - **Network**: gossip timestamp staleness in `events.rs` was
    symmetric `saturating_sub` — doubled the replay window to ~10 min;
    now one-sided per gotcha #44. PEX response no longer calls
    `kademlia.add_address` on unauthenticated peer/multiaddr pairs
    (Kademlia eclipse vector). `spec_logits` decoder hard-caps
    `num_positions ≤ 32` and `vocab_len ≤ 512_000`.
  - **Pool & credit**: `TREE_POOL_REMOVAL_REPLAYS` is no longer cleared
    on restart — the 5-min freshness window IS the replay window, and
    a saved `PoolRemoval` could re-evict after a planned restart. Now
    timestamps each entry. `handle_inbound_acceptance` verifies the
    acceptance's `invitee_node_id` matches the pending invitation
    (anyone learning the `invitation_id` could otherwise steal the
    slot). `track_forward_participation` caps peer-controlled
    token count at 8192 — `LayerForward.token_count = u32::MAX`
    would otherwise mint ~43B credits per serving node per flush.
  - **Inference**: intermediate-segment activations are now
    shape-validated against the input we forwarded; a malicious peer
    returning a wrong-shaped tensor would otherwise crash the next
    worker (gotcha #20). NaN/Inf rejected from peer-supplied f32
    tensors at three sites (ring AllReduce, TP attn/ffn AllReduce
    result, `spec_logits` rows before argmax in
    `greedy_accept_reject` — IEEE 754 NaN argmax is non-deterministic
    and lets a malicious peer steer accepted tokens).
  - **Shard & adapter integrity**: shard startup size check tightened
    to exact match (was ±10%); LoRA safetensors now BLAKE3-pinned via
    a `<filename>.blake3` sidecar so a swapped adapter can't silently
    produce wrong inference output.
  - **HTTP API**: `POST /api/admin/update/{check,apply}` now
    loopback-only (auto-downloads + binary swap should not be remote).
    `provider-model-status` rejects path-injection in `model_id` via
    allowlist (`[A-Za-z0-9._:@-]`, max 256, no `..`). `join-network`
    rejects private/loopback/link-local multiaddrs (P2P-layer SSRF).
    `GET /api/admin/responses` only returns `input_preview` /
    `output_text_preview` to loopback callers (single shared API key
    leaked prompt prefixes across users). Removed unreachable dead
    code: the non-loopback `x-swarm-internal-token` block in
    middleware.rs (per gotcha #30, `internal_auth_token` is
    per-process random and never crosses node boundaries).
  - **Update + supply chain**: `apply_update` re-verifies version is
    strictly newer than `env!("CARGO_PKG_VERSION")` at apply time
    (downgrade-by-replay protection). All GitHub Actions in `ci.yml`
    and `release.yml` pinned to commit SHAs with the tag in a trailing
    comment; `.github/dependabot.yml` opens weekly bumps. `libc::umask(0o177)`
    around the AF_UNIX socket bind in `process_pool.rs` so the IPC
    socket is created at 0o600 atomically (closes a TOCTOU between
    bind and post-bind `set_permissions` where a local attacker
    racing inotify on `/tmp` could connect first and impersonate
    the worker). `db.redb` chmod'd to 0o600 on Unix after create.
    `.env` loaded BEFORE the Tokio runtime spawns worker threads
    (`std::env::set_var` is unsound in a multi-threaded process).
  - **Deferred (still open)**: C1 — auto-update binary signing.
    SHA256 sidecar comes from the same release as the binary, so a
    compromised account/CI token publishes both together. Real fix
    needs an offline keypair embedded as `env!()` pubkey + cosign
    or minisign signature as a third release asset. Tracked in
    `docs/ARCHITECTURE.md` § Deferred Items.

- **PRIVACY: `chat_completions` no longer leaks prompt content into
  the activity event bus** — the event message included up to 60
  characters of the last user message via `prompt_preview`, and
  `emit_activity` broadcasts to every authenticated dashboard
  subscriber AND replays via `activity_history` to new connections.
  On a multi-tenant or shared-host node this leaked one user's
  prompt to others without their knowledge. The Anthropic handler
  did NOT reproduce this pattern. Removed the preview entirely;
  the event now carries only model, message count, and max tokens.
- **`lock_shard` / `unlock` now propagate DB write errors** — the
  handler used `let _ = db.insert_raw / .remove(...)` for the
  `locked_shards` persist, silently discarding write failures while
  still mutating the in-memory DashMap. On a DB error the handler
  returned `status: "ok"` despite a state divergence from disk;
  after restart the auto-manage pruner could remove a shard the
  operator believed was pinned. Same pattern was already corrected
  for pool pins, auto-manage policy, encrypted pipeline, and HF
  trust pin in earlier sweeps. Now persists first and surfaces any
  error as 500.

- **CRITICAL: `ResponsesRequest.extras` was an unbounded ingress vector**
  — the `#[serde(flatten)]` catch-all for unknown top-level JSON keys had
  no cap on count or per-value size. A request with thousands of unknown
  keys (or one very large value) was materialised into the in-process
  `HashMap<String, Value>` before any validation could reject it AND
  forwarded verbatim to upstream cloud providers on the proxy path.
  Added `MAX_RESPONSES_EXTRAS_COUNT = 32` and
  `MAX_RESPONSES_EXTRA_VALUE_BYTES = 4096`, checked first in
  `validate_responses_ingress`.
- **`/v1/responses/{id}/input_items` query parameters length-capped** —
  `after`, `before`, `order`, `include` were unbounded. Synthetic
  `item_N` cursors are short by construction so any large value is
  hostile. Capped at 64 bytes each at handler entry.
- **CSP meta tag in `frontend/index.html` synced to server header** —
  the meta lacked `blob:` in `img-src`, `frame-ancestors 'none'`,
  `base-uri 'self'`, and `form-action 'self'`. The server header in
  `middleware.rs::security_headers` already had these and takes
  precedence at runtime, but the meta covers the
  static-server / file-open / cached paths where the daemon's header
  isn't applied. Added an inline comment marking middleware.rs as
  authoritative.
- **CRITICAL: `responses.js` was calling a non-existent function** —
  `App.data.authFetch` doesn't exist; the symbol is `App.authFetch`.
  Every Responses dashboard interaction (load, retrieve, cancel,
  delete) was silently failing with `TypeError`. Same pattern that
  was caught in `auto-manage-status.js` previously — the responses.js
  refactor that introduced the `_action` helper reintroduced it.
  Fixed via 3-site replace.
- **`/v1/responses/{id}` path-param validation** — added
  `validate_response_id` helper mirroring the
  `previous_response_id` cap (≤64 ASCII alphanumeric `+_-`). Called
  at the top of `get_response`, `cancel_response`, `delete_response`,
  `list_input_items`, and `get_response_maybe_stream`. Closes a path
  where a megabyte-long `{id}` could inflate logs / DashMap key
  materialization for `BACKGROUND_CANCEL` / `BACKGROUND_STATE`.
- **`GET /api/admin/responses ?status=` length cap** — raw status
  string was split with no length cap, allowing an unbounded
  `Vec<String>` allocation. Added a 256-byte cap before splitting.
- **`Q8_0` tensor non-finite guard** — F32 path already rejected
  NaN/Inf in deserialized activations; Q8_0 path dequantized blindly.
  Added a matching `is_finite()` check after `dequantize_q8_0` so a
  malicious or broken peer can't poison subsequent attention via
  NaN/Inf-dequantizing Q8_0 blocks. Also flipped the F32+Q8_0
  truncation error type from `Internal` (HTTP 500) to `Inference`
  (a truncated wire payload from a peer is a network fault, not a
  local code bug).

- **`/v1/responses` ingress validation** — `validate_responses_ingress`
  runs BEFORE the cloud-proxy / Anthropic-bridge / local-inference
  branches. Caps `previous_response_id` ≤64, `instructions` ≤2 MB,
  `user` ≤256, `model` 1..=256, `truncation`/`service_tier` ≤64,
  `metadata` aggregate ≤64 KB. Closes a path where attacker-sized
  strings could reach upstream provider request bodies and log lines.
- **`/v1/messages` `max_tokens` cap** — Anthropic handler now rejects
  `max_tokens > 32768` (matches the local sampling-params clamp ceiling)
  at ingress instead of silently clamping for local + forwarding raw to
  upstream proxies.
- **HF-proxy rate limiting** — `/api/admin/hf/probe` and
  `/api/admin/hf/search` no longer get the loopback admin-GET exemption.
  A runaway local script or a malicious browser extension on
  `localhost:8800` can no longer loop-call them and burn HuggingFace
  API quota.
- **Pool invite-code validator** — now checks ASCII alphanumeric AND
  length (was: length only after trim+uppercase, accepting some malformed
  inputs silently).
- **Identity leaderboard `?limit=`** — clamped to `[1, MAX]` (was: silent
  empty array on `limit=0`).

### CI

- **macOS test + clippy** — `.github/workflows/ci.yml` matrix now runs
  `cargo test --lib --bins` and clippy on `macos-15` in addition to
  Linux. Integration tests stay Linux-only with explicit guards until
  the first macOS failure decides whether to fix or skip-list per test.
  Build job already had macOS — this closes the test-coverage gap from
  `docs/plans/next_steps.md` § 4.

### Correctness

- **`pool::handle_leave_pool` lock ordering** — was the only pool
  handler holding the `pool_state` read lock across
  `rate_limiter.check_and_record()`. Now matches
  `handle_create_invitation` / `handle_accept_invitation`: extract
  `pool_id` under the guard, drop, rate-limit, then later acquire the
  write lock for mutation. All 4 pool handlers audited for ordering
  consistency.

### Reliability

- **CRITICAL: M9 background spawn now `catch_unwind`s a panic** —
  the non-stream `/v1/responses` background path was a bare
  `tokio::spawn` with no panic guard, while the V8 streaming path
  has had `AssertUnwindSafe(...).catch_unwind()` around it since
  V8 landed. On a panic anywhere in `run_background_inference`
  (translate / chat_completions / buffer / parse / chat→responses
  translate) the redb record was stranded at `status:in_progress`
  forever — no terminal state ever written — and a polling client
  would see `in_progress` indefinitely. Wrapped the spawn body in
  `catch_unwind`; on panic we stamp a terminal `failed` record into
  redb and call `unregister_background_cancel`.
- **`POST /v1/responses` (M9 background) now returns 202 Accepted**
  — was returning HTTP 200 even though the response was queued for
  async completion. V8 streaming was already correct. Clients that
  branch on status code to detect sync vs deferred completion would
  have misclassified M9 background as a synchronous result.

- **`BACKGROUND_CANCEL` TTL sweep** — the response-id → cancel-flag
  registry could leak entries when a Tokio task was cancelled
  externally (e.g. process shutdown mid-flight) before its cleanup
  path ran. Added a parallel `BACKGROUND_CANCEL_AGES` map keyed by
  response id with insert-time `Instant`; new
  `register_background_cancel` / `unregister_background_cancel`
  helpers wrap every insert / remove site so the two maps stay
  consistent. New `prune_stale_background_state` drops entries
  older than `BACKGROUND_CANCEL_MAX_AGE_SECS = 7200` (2 h —
  generously above any real background-inference run) plus the
  matching `BACKGROUND_STATE` entry. Wired into the existing
  hourly responses sweep.
- **`/api/admin/network_code` peer-iter caps** — the dashboard's
  invite-code refresh handler walked every peer and every
  advertised address with no bound. Added
  `NETWORK_CODE_PEER_SCAN_CAP = 64` and
  `NETWORK_CODE_ADDR_PER_PEER_CAP = 16`. A public-facing IP is
  almost always advertised by the first few peers, so capping the
  inner loops preserves the happy path and bounds the worst case.

### Polish

- **`dispatch_inference` / `router_inference_stream` no longer
  thread an unused `&AppState`** — the parameter was passed
  through both functions and read by neither. Dropped from both
  signatures + 3 call sites in `openai/mod.rs`.
- **`is_prefill` DIAG field uses `PREFILL_ACTIVATION_THRESHOLD_BYTES`
  instead of a hardcoded `100_000`** — the constant is `pub(crate)`
  and documented as a tuning knob; the bare literal would silently
  diverge the diagnostic from the real classifier.
- **`responses.js` `_statusCell` fallback fixed** — `I18n.t(key,
  string)` passes the second arg as an interpolation **object**,
  not a fallback string. Switched to the
  `translated !== key ? translated : raw_status` pattern used
  elsewhere so unknown future status values render as the raw word
  rather than the literal i18n key.
- **`notifications.js` removed a leftover `console.warn`** — the
  ws-ticket fetch catch block logged on every transient reconnect
  failure (visible in DevTools console). The user-visible
  connection-lost banner via the `onclose` path is the right
  surface; logging adds nothing.

### Observability

- **`forward_batch` fallback log gained `model_id`** — the
  `kv_offset mismatch — falling back to sequential` debug log was
  missing the model key, making it impossible to correlate which
  model triggered the fallback in a multi-model deployment.

### Refactor / dedup

- **`buffer_and_translate_chat_response` helper** collapses the
  three-step buffer/parse/translate boilerplate that
  `create_response` and `run_background_inference` each hand-rolled
  with its own error-handling shape. Returns
  `Result<ResponsesResponse, String>`; each caller wraps the
  human-readable error string into its native error type. Site 1
  went from 13 lines + 2 match arms to 8 lines + 1 `.map_err`;
  site 2 went from 32 lines + 3 match arms to 14 lines + 1 match.
- **`speculative::greedy_accept_reject`** extracts the bit-identical
  14-line accept-reject arithmetic shared between
  `try_speculative_distributed` (Item 2) and `try_dsd_distributed`
  (Item 12). Token emission and gamma-controller bookkeeping that
  surrounds the call stay per-path because the divergence there is
  real.
- **`build_chat_completion_response` helper** in
  `api/openai/streaming.rs` collapses the response-shape
  construction shared between `router_inference` (with logprobs +
  session id) and `split_non_stream_response` (without). The
  empty-vec → `None` logprob conversion lives in one place.

- **`ResponseError::new(code, message)` constructor** replaces 11
  identical `ResponseError { code, message, extras: HashMap::new() }`
  struct literals across `responses/{mod, stream}.rs`.
- **`new_response_id()` / `new_message_id()` helpers** in
  `responses::mod` replace the
  `format!("resp_{}", uuid::Uuid::new_v4().simple())` /
  `format!("msg_{}", ...)` idiom that appeared at 7+ sites across
  the responses module. The `rs_` (reasoning item) and `resp_test_`
  (test fixture) prefixes intentionally stay inline — different
  conventions.
- **`apply_shard_window_change` helper** in
  `admin_models::helpers` extracts the post-window-compute /
  pre-event-emit block shared between `unload_shard` and
  `load_shard`: empty-window→`evict_and_unload` else
  `restart_with_window` + `evict_split_models`, then
  `clear_model_load_history` and `signal_dashboard(ModelsChanged)`.
  The activity-event + tracing tails stay per-handler (different
  messaging is the right answer there).
- **`DEFAULT_TEMPERATURE` / `DEFAULT_TOP_P` promoted to `pub(super)`**
  in `responses::mod`, mirroring the existing `DEFAULT_MAX_OUTPUT_TOKENS`
  shape. `translate.rs` now imports via `super::` instead of holding
  its own private literal — closes a drift hazard.
- **`_isActiveDlState(state)` helper** in `downloads.js` replaces 4
  duplicated `(state === 'downloading' || state === 'awaiting_manifest')`
  predicates and normalises a missing `typeof === 'string'` type guard
  in one of the four sites.
- **`detect_tp_groups` dropped its unused `_manifest: &ModelManifest`
  parameter** — the function operates on `candidates` and `segments`
  only.

- **SSE parser dedup across `responses/{stream,anthropic_bridge}.rs`** —
  promoted `drain_sse_data_payloads` and `find_subslice` from private
  to `pub(super)`, extracted a new `parse_sse_block_data_lines` helper
  for the per-block parsing both call sites need, and dropped
  `anthropic_bridge::find_event_boundary` (was the same logic under a
  different name).
- **`speculative_common_eligible` helper** — `speculative.rs` and
  `dsd.rs` shared a 6-line `eligible()` prefix (decoding flag, draft
  model loaded, greedy temperature, no encryption, etc.). Extracted
  to `pipeline/mod.rs`; each path now calls it after its own
  path-specific flag check.
- **`network-map.js` map-stats helper** — `render()` and
  `updateFromWs()` built the same I18n-formatted node/region count
  text inline; extracted `_updateMapStats(totalNodes, totalRegions, maxCount)`.
- **`responses.status_unknown` i18n key** — added across all 21 locale
  files; was missing, so any non-English user seeing a response with
  an unrecognized status value got the raw English `unknown`.
- **`auto-manage-status.js` plural fallback** — the active-download
  fallback string used `shard(s)` parenthetical pluralization;
  replaced with a proper singular/plural branch.
- **`detect_tp_groups` cleanup** — dropped an unused `_manifest:
  &ModelManifest` parameter that did nothing at the call sites; the
  function operates on `candidates` and `segments` only. Also renamed
  `_is_last` → `is_last` in `tensor_parallel.rs::execute_tp_segment`
  (the underscore convention falsely implied unused; the param is
  read at lines 316 and 357).

- **`CreditDelta` enum replaces `is_earning: bool`** on
  `apply_credit_direct`. Variants `Earning` / `Spending` / `Refund`;
  the new `Refund` leaves both monotonic counters
  (`lifetime_earned`/`lifetime_spent`) untouched, which is the
  semantics escrow refunds need but couldn't get from the old bool.
  `EscrowManager::create_escrow` and `refund_escrow` now route through
  the helper instead of hand-rolling the balance write + persist.
  `cleanup_expired` keeps its manual block (its retry-on-failure
  semantics need an in-memory revert that `apply_credit_direct`
  deliberately doesn't do); the reason is documented inline.
- **`pipeline::fastpath_request_disqualified`** helper extracts the
  shared TP-empty + LoRA + vision-images guard from
  `remote_generate.rs`, `speculative.rs`, and `dsd.rs`. Per-path
  shape / encryption / flag preconditions stay separate because they
  ARE subtly divergent (1-segment vs 2+-segment vs all-remote, and
  remote_generate has a per-model encryption gate the others don't).
- `SharedState::resolve_peer_id_bytes` helper replaces a 5-site
  `peer_id_map.or_else(peer_registry)` lookup duplication across
  `inference/pipeline/{distributed,remote_generate,speculative,dsd,
  tensor_parallel}.rs`.
- `crypto::hkdf_sha256_derive_32` helper replaces 3 sites that
  duplicated the same `Hkdf::new + expand + .expect("32 bytes is a
  valid HKDF-SHA256 output length")` pattern across `provider_keys`,
  `gossip_seal`, `session`.
- `cli::discover_model` helper replaces a 14-line `GET /v1/models`
  block duplicated in `bench.rs` + `chat.rs`.
- `auto_manage::manager::read_shard_pins` helper replaces a `pool_state.try_read()`
  + `shard_pins` extraction duplicated in `scoring.rs` + `prune.rs`.
- `cli::bench::tokens_per_sec` and a `pub const SWARMLLM_GITHUB_REPO`
  in `update.rs` (previously bare `"enapt/SwarmLLM"` string in two
  places).
- Various small stale-doc + dead-code cleanups; see commits `36af419`,
  `c9acbfc`, `c10956e`, `ccfbf14`, `712a4da`, `d8a840c` for the
  per-sweep summaries.

### Tests

- 821 lib tests passing (was 816 at v0.1.0). Clippy clean both feature
  sets. Pre-push hooks enforce `cargo fmt && cargo clippy --all-targets
  -- -D warnings` on every commit.

## v0.1.0 — 2026-04-25

First non-alpha tag. Cuts off the v0.1.0-alpha.2 line and rolls every
post-alpha commit (`v0.1.0-alpha.2..main`, ~56 commits) into a single
release.

### What's new since alpha.2

- **OpenAI Responses API (`/v1/responses`)** — full v1 (M1–M9) + v2
  (V1–V8) surface. Cloud-proxy passthrough, local inference via Chat
  translation, Claude/Anthropic-Messages translation bridge,
  multimodal input, redb persistence with 30-day TTL, chained
  `previous_response_id`, background mode, resumable SSE +
  background streaming (202 + Location handshake), `input_items`
  pagination, dashboard panel under the new Responses tab. See
  `docs/plans/responses_api.md` and `docs/plans/responses_api_v2.md`
  for design + full bench numbers in `docs/bench_results/`.
- **Audit fixes (`audit_2026-04-24.md`)** — 2 CVE backports for
  vendored libp2p-gossipsub (CVE-2026-33040 + CVE-2026-34219), 5
  cargo-update CVE patches (quinn-proto + rustls-webpki), spec_logits
  IPC wire-format fix (no more 5–7× JSON bloat for f32 row-major
  payloads), Anthropic / OpenAI proxy unknown-field preservation +
  beta-header forwarding, cache-token usage roundtrip, escrow refund
  loss + false-toast + total_bytes + spawn_region + no_peers_interval
  fixes, hot-path mem::take in forward_through_segments. See
  `audit_2026-04-24.md` in memory for full chronology.
- **Audit fixes (this session)** — DNS TOCTOU closed via custom
  reqwest resolver that filters private IPs at request time; shard
  size check tightened to exact-match for zero-hash placeholder
  manifests; prune cycle re-checks resource pressure between scan and
  execution so over-pruning can't happen; Grafana docker-compose
  refuses to start without explicit env vars (no more silent
  admin/admin).
- **Dep upgrades** — axum 0.7 → 0.8 (route patterns rewritten,
  WebSocket Message types adapted, async_trait removed, tower-http
  TimeoutLayer signature change), redb 3 → 4 (1.5× write throughput,
  on-disk format compatible via existing UpgradeRequired migration
  path), tower-http 0.5 → 0.6, candle 0.9 → 0.10 (vendor was already
  at 0.10.1 — refreshed dep declaration to match).
- **Distributed inference** — Items 1–7 + Item 8 (full architecture
  + cross-over demo: TinyLlama 1.1B GPU loopback ~100 ms slower vs
  control on small prompts; Qwen2.5-7B CPU loopback **12.9× iter-1
  TTFT** on 640-token prompt), Item 12 DSD multi-segment spec
  decoding (flag), Item 13 Q8_0 activation compression (~3.76× wire
  saving, flag), Item 16 Parallax scheduler phases A+B+B.2+C+C.2 on
  by default. Headlines: prefix-cache hit gives 29.4× wall-clock on
  repeat prompts; remote-generate fastpath gives 1.93× decode; Phase
  4 batched chunked prefill gives 1.57× tok/s @ c=4. See
  `docs/plans/distributed_inference_speedup.md`.
- **Cross-platform IPC** — Daemon ↔ model-worker IPC ported from
  Unix-socket-only to the `interprocess` crate (AF_UNIX with 0o600
  perms on Linux/macOS, named pipes with default-DACL on Windows).
  Runtime-validated on Windows (CPU + GPU, single-node + multi-node
  + split shards) on 2026-04-23. See note below.

### Test + lint baseline at v0.1.0

- 816 lib tests passing (782 → 816 since alpha.2), clippy clean both
  with default features and `--no-default-features --features
  dev,claude-subscription`.
- Integration tests green.
- 38/38 M1–M9 + 27/27 V1–V8 curl matrix pass against a live daemon.

### Known deferred items (post-v0.1.0)

- **libp2p 0.55 → 0.56** — ✅ landed post-v0.1.0 (commit `be9c32c`).
  Vendored libp2p-request-response re-ported to upstream 0.29.0 with the
  Tokio watchdog patch re-applied; the `confirmed`-flag patch (gotcha
  #16) is OBSOLETE in 0.29 and was dropped. Vendored libp2p-gossipsub
  removed entirely — upstream 0.49.4 ships the CVE-2026-33040 +
  CVE-2026-34219 fixes our backport carried.
- **`POST /v1/responses/compact`** — V9 of the v2 plan, deferred
  indefinitely until a concrete caller asks for it.
- See `docs/ARCHITECTURE.md` § "Deferred Items" for the full list.

### Notes for upgraders from v0.1.0-alpha.2

- Provider proxy now uses a custom DNS resolver. Hostnames that
  resolve to private/internal IPs (RFC 1918, link-local, loopback)
  are rejected at request time. The pre-flight `validate_provider_url`
  helper still runs for friendlier error messages but is no longer
  the only defense.
- `monitoring/docker-compose.yml` no longer supplies `admin/admin`
  defaults for Grafana. Run `cp .env.example .env && $EDITOR .env`
  before `docker compose up -d`.
- Anyone with a v3 redb file: the existing `Database::open` path
  already backs up the file (`db.redb.bak`) and recreates fresh, so
  in-place upgrade Just Works but loses prior state. New deployments
  start clean on v4.

---

### Detailed change log: alpha.2 → v0.1.0

> Prior tag: `v0.1.0-alpha.1` (2026-03-18, 674 tests) → `v0.1.0-alpha.2` →
> `v0.1.0` (2026-04-25, 816 tests). The summary above is the user-facing
> rollup; this section preserves the per-feature detail captured during
> the alpha → 1.0 grind so reviewers don't have to read 56 commits.

#### Cross-Platform IPC

Daemon ↔ model-worker IPC was Unix-socket-only between 2026-04-18 and
now, silently breaking Windows builds. Ported to the `interprocess`
crate (`local_socket` + tokio): AF_UNIX with 0o600 perms on
Linux/macOS, named pipes with default-DACL (current-logon-session)
on Windows. Security parity, no protocol changes (the framed codec
was already transport-agnostic).

**Runtime-validated on Windows (2026-04-23):** Both the CPU
(`swarmllm-windows-x86_64-cpu.zip`) and GPU
(`swarmllm-windows-x86_64-gpu.zip`) Windows binaries from this release
were smoke-tested on a real Windows host (Ryzen 7 5800H + RTX 3070
Laptop, 8 GB VRAM).

- **CPU single-node:** startup → BLAKE3 shard verify → named-pipe IPC
  handshake → worker subprocess model load → inference (HTTP 200 in
  5 s on TinyLlama 1.1B Q4_K_M) → API-triggered graceful shutdown
  (drain in 3 s, no orphaned worker).
- **CPU multi-node, single-segment:** node B bootstrapped from node A
  over loopback TCP. Noise + Yamux handshake green, GossipSub
  propagated model availability, node B's `/v1/models` listed
  TinyLlama as `owned_by: network`. Cross-node inference via Item 4
  remote-generate fast path: 8 tokens, trust score updated.
- **CPU multi-node, split shards (forced 2-segment pipeline):** A
  hosting shard 0 (layers 0-12), B hosting shard 1 (layers 12-22),
  auto-manage disabled to prevent cross-fill. Pipeline scheduler
  assembled `["A:0-12", "B:12-22"]`, encrypted activation tensor
  (172 KB) shipped over libp2p, B's worker subprocess computed the
  tail layers, response channel returned tokens. Generated coherent
  output ("The capital of France is Paris…") in 6 s.
- **GPU single-node:** dynamic-loading cudarc resolved bundled CUDA
  redist DLLs (cublas64_12.dll, cublasLt64_12.dll, cudart64_12.dll,
  curand64_10.dll, nvrtc64_120_0.dll, nvrtc-builtins64_124.dll) at
  process load — no CUDA Toolkit on the test host. RTX 3070 detected,
  worker subprocess loaded model on `device=Cuda(CudaDevice(1))`,
  inference returned coherent content. Graceful shutdown clean.

**Side-effect validation:** auto-manage's peer-to-peer shard transfer
also exercised on Windows during the multi-node run — A's missing
shard 1 was downloaded from B and vice versa via libp2p
request-response before auto-manage was disabled for the split test.

macOS aarch64 binary remains compile-validated only.

#### Distributed Inference Speedup Arc

A multi-session effort to speed up distributed inference, tracked in
`docs/plans/archive/distributed_inference_speedup.md`. Items 1–16 numbered in
plan order; default-on items landed as they shipped, flag-gated items
are off until benchmarked on real workloads.

**Default-on stack (user-facing in [Performance chapter](docs/book/src/operations/performance.md)):**

- **Item 3 — Continuous batching** (2026-04-19): fused `forward_batch`
  over concurrent Generate requests. 1.34–1.55× GPU throughput at batch
  2–8. CPU falls through to sequential with no regression.
- **Item 4 — Remote-generate fast path**: single-segment distributed
  inference runs the full decode loop on the remote worker instead of
  per-token coordinator round-trips. **1.93× decode speedup**.
- **Item 5 — Cross-request prefix cache**: worker keeps an LRU of prefill
  KV snapshots keyed by prompt prefix. **29.4× wall-clock** on
  re-submission of the same 513-token prompt.
- **Item 7 Phase 1+2 — BatchGenerate + Sarathi chunked prefill**
  (2026-04-19): SlotTable admits concurrent requests, each Prefilling
  slot advances by `prefill_chunk_tokens` (default 128) per decode tick.
  **17–23× TTFT fairness** at concurrency 2/4/8 on RTX 3070 +
  TinyLlama Q4. See `docs/plans/benchmarks/round4.md`.
- **Item 7 Phase 4 — Batched prefill forward** (2026-04-19): fuses
  concurrent same-shape prefill chunks into one `forward_batch`.
  **1.57× aggregate tok/s @ c=4** with uniform 180/180/180 ms TTFT.
  See `docs/plans/benchmarks/round5.md`.
- **Item 8 — Cross-node prefix KV sharing** (2026-04-19/20): when node B
  receives a prompt whose prefix peer A already prefilled, B fetches
  A's KV snapshot over the wire instead of re-prefilling locally.
  Full pipeline: PrefixCacheAnnounce gossip → cross-node index →
  PrefixFetchProbe → trust-gated SendPrefixKvFetch → BLAKE3 verify →
  NaN/Inf scan → hydrate → suffix-prefill.
  **Measured 12.9× iter-1 TTFT speedup on Qwen-7B CPU-CPU localhost**
  (151.7 s → 11.8 s on a 672-token prompt, Round 6 bench 2026-04-20).
  TinyLlama on GPU is the fast-prefill corner case where the fetch path
  is ~100 ms slower than re-prefilling (28 MB snapshot vs 460 ms
  prefill).
- **Item 16 — Parallax scheduler** (2026-04-18/19): shortest-path DP
  over observed per-layer latencies (EMA over recent forwards), replacing
  the greedy latency-only sort. Phase B.2 cross-gossips top-32 observed
  latencies via `NodeCapability.observed_latencies`. Phase C.2 adds a
  soft acquire/prune bias in `AutoShardManager` driven by a per-shard
  stability counter (≥3 consistent ticks before it acts); hard
  constraints (pinning, trust, VRAM) always win.

**Flag-gated:**

- **Item 2 — Distributed speculative decoding** (`speculative_distributed`):
  draft-target speculation across nodes. 40–52% accept rate in a
  llama-cpp-draft / candle-target pairing.
- **Item 6 — SWIFT self-speculative** (`swift_self_speculative`):
  target model acts as its own draft by skipping a layer range. Shelved
  on CPU until flash-attn-with-mask lands.
- **Item 12 — DSD (decentralized speculative decoding)**
  (`decentralized_spec_decoding`, 2026-04-18): multi-segment pipeline
  with γ-token speculation + KV truncation primitives + ~410 LOC
  coordinator loop in `pipeline/dsd.rs`. End-to-end WAN benchmark
  pending.
- **Item 13 — Activation compression Q8_0** (`activation_compression`):
  intermediate pipeline hidden states quantized to Q8_0 on the wire.
  ~3.76× compression, RMS error <0.005. End-to-end multi-segment
  benchmark pending.
- **Item 1 — Persistent pipeline stream** (`persistent_pipeline_stream`):
  one long-lived libp2p bidirectional stream per pipeline session.
  Wire-verified; no measured latency win because the bottleneck was
  elsewhere (Items 4 + 7 solved it).

#### Round 6 Bench Findings (2026-04-20)

The Item 8 two-daemon loopback bench caught three wire bugs before the
measured numbers above landed:

1. `SwarmMessage::PrefixCacheAnnounce` missing from the `TOPIC_MODELS`
   arm in `NetworkManager::handle_broadcast` — Phase 1 announces
   silently dropped at the gossip layer. Loopback self-index path
   masked it in single-node tests.
2. `WorkerMsg::PrefixSnapshotResponse` / `DaemonMsg::PrefixFetchResult`
   carried `payload: Option<Vec<u8>>` inside the JSON-framed IPC header.
   `serde_json` encodes `Vec<u8>` as a JSON array of integers (~5× size
   bloat), so a 28 MB snapshot became a ~102 MB header and blew past
   the 64 MiB `MAX_HEADER` cap.
3. Three chained cross-node-fetch timeouts (`PREFIX_FETCH_TIMEOUT_MS=500`
   in the worker, 400 ms daemon network timeout, 500 ms serving-worker
   IPC timeout) were sized for TinyLlama's 28 MB snapshot. A Qwen-7B
   snapshot is 73 MB and takes ~500–1000 ms to serialize+wire — every
   timeout fired and silently converted real hits into misses. Bumped
   to 3000 / 2500 / 2000 ms respectively, keeping the worker timeout as
   the outer bound.

#### Code Sweep (105 issues found, 58 fixed)
- **Round 1**: 10 parallel review agents across all 109 .rs files — 68 issues (9 CRITICAL, 32 HIGH, 22 MEDIUM), 41 fixed
- **Round 2**: Second pass — 37 new issues (5 CRITICAL, 22 HIGH, 10 MEDIUM), 17 fixed
- Key fixes: max_seq_len 2048 cap, ShardReader cross-tensor bleed, TensorPayload auth, escrow double-charge, IPC framing overflow, API key leak, Gemma embedding scale in forward_batch, hardcoded sampling in distributed forward

#### Credit System Overhaul
- Balanced rates: `rate × tokens` on both earn and spend (no layer multiplier)
- Minimum balance enforcement: `MIN_BALANCE_FOR_INFERENCE = -1000`
- Atomic credit accumulation via `pending_credit_earn` AtomicI64
- Anti-Sybil peer balance deduplication by NodeId
- Priority tiers require positive balance for Gold/Platinum

#### Device Pool Invite Codes
- 8-char one-time codes (e.g., `A3F7K2M9`), 24h expiry, Ed25519 signed
- CLI: `swarmllm pool create/invite-code/join/status/leave`
- API: `/api/pool/generate-code`, `/api/pool/join`, `/api/pool/device-name`, `/api/pool/credit-split`

#### Pool UX Overhaul
- Device nicknames, online/offline status, per-device stats, combined VRAM display
- QR code for invite codes, credit split configuration (0-50%)
- "My Devices" tab with full management UI

#### Terminology Clarification
- "My Devices" vs "Swarm Peers" — clear separation in setup wizard, share popover, dashboard

## [0.1.0-alpha.2] - 2026-03-18

### Release & Scale Readiness (Phase 19)
- **Docker release packaging**: Production `docker-compose.yml` (CPU default, GPU via `--profile gpu`), `.env.example` with all configurable env vars, GitHub Actions CI/CD pushing CPU + CUDA images to GHCR on git tag
- **Docker dev cluster**: 3-node `docker-compose.dev.yml` with static subnet, TCP bootstrap, container-optimized config (`config/docker-cluster.toml`)
- **Setup wizard redesign**: 4 steps → 3 steps (About You → Connect → Ready), invite code paste field, auto-download ON by default, hardware-aware model recommendations based on VRAM, dynamic summary
- **mDNS simultaneous-dial race fix**: When two nodes discover each other via mDNS simultaneously, both connections could fail. Added `pending_redial` queue with hash-based jitter (2-5s) for automatic recovery
- **Upload bandwidth enforcement**: `max_bandwidth_mbps` config now enforced on shard serving with proportional delay (was stored but never applied)
- **Manifest publisher claim**: Copied shard directories now properly gossiped — publisher set to local node_id with manifest hash recomputed on startup shard scan
- **Invite code error messages**: Invalid invite codes now return descriptive 400 errors instead of generic 500
- **Scalability (S1)**: Shard announce delta compression — only broadcasts when shard set changes or periodic re-announce every 10 cycles
- **Scalability (S2)**: P2P shard transfer fallback in auto-manage — when no HuggingFace source known, downloads from peer holders instead of doing nothing
- **Scalability (S3)**: peer_registry capped at 200 entries — evicts highest-latency non-LAN non-pipeline peers when over limit
- **Scalability (S4)**: Gossip broadcast frequency scales with `log(peer_count)` — 30s at ≤10 peers, 120s at 1K, 240s at 10K. Health pings stay at 30s
- **Docker image**: 181MB CPU image (debian:bookworm-slim), multi-stage build, non-root user, health checks
- **Tested**: 5-node Phi-3.5 distribution on Proxmox server (trust promotion, auto-manage, target replicas)

### Local Embedding Privacy
- **Config**: `local_embedding_privacy: true` in `[inference]` — requesting node embeds tokens locally, sends hidden-state activations (not raw token IDs) to remote first-segment nodes
- **LocalEmbedder**: Loaded from shard_000.bin at startup, uses candle for token→embedding conversion
- **Pipeline integration**: Pre-embedded activations skip remote embedding, reducing token exposure to relay nodes

### Deep Code Sweep — 56 fixes across 4 passes (16 parallel review agents)
- **Pass 1 (15 fixes)**: gossip_seal future-epoch removal, manager.rs unwrap safety + shard download cap, AllReduce zstd zip-bomb cap, shard atomic truncate, KV-cache eviction map fix, pipeline pending_vision cleanup + VLM hidden_dim expansion, escrow persist-before-balance, ledger bucket_balance div_euclid, error body truncation, manifest streaming BLAKE3, huggingface u64 range + progress retry, acquisition duplicate guard, protocol pre_embedded defaults
- **Pass 2 (10 fixes)**: sampling order consistency (temperature→top-k→softmax), TP block_in_place wrapper, stale logprobs clear, pending_tensor_channels leak fix (Instant timestamps + periodic sweep), num_layers saturating_sub, protocol unwrap→expect, anthropic proxy error truncation, escrow cleanup balance persist, model_id u16 length guard, max_tokens=0 early return
- **Pass 3 (17 fixes)**: duplicate streaming finish event guard, pending_layer_results failover leak, KV-cache orphan eviction, multi-turn session overwrite prevention, TP tp_size minimum guard (≥2), peer_http_url LAN/Tailscale fix, DB backup-on-upgrade, inverted --shards range validation, tied_output_weight streaming read, pool cosign cryptographic separation, pool_registry cleanup, escrow cleanup count, gossip epoch bound (reject >2 epochs old), invite code decrypt-fail no fallthrough, auto_update download gate, multi-image VLM remote guard, dead code removal
- **Pass 4 (14 fixes)**: completion_tokens EOS corruption (used clean_tokens.len → generated_tokens.len), keystore 0o600 permissions on private key files, WebSocket connection counter RAII guard, credit earn crash-window (single persist after earn+forwarding), MCP error body API key scrubbing, all_shards_available cache cap (1000 entries), TP attention GQA modulo wrap for tp_size > n_kv_head, AllReduce duplicate rank warning, pool state db.remove() on leave (fixes null deserialization type mismatch), health monitor future-timestamp evasion (clamp to zero), supervisor dead code cleanup (MAX_RESTART_ATTEMPTS), GGUF total_size saturating_add (overflow on malicious headers), gossip epoch fallback tightened (3→2 epoch window)
- **Model/storage (6 fixes)**: huggingface total_size==0 guard, retry HTTP status check, atomic tmp+rename for mmproj/header, LoRA rank==0 guard, auto_manage path traversal sanitization

### Feature Wiring — 8 previously unwired features now fully integrated
- **Priority tier enforcement**: `calculate_tier()` with real network percentile from peer credit gossip; `max_concurrent_for_tier()` enforces per-tier concurrent request limits in `drain_queue()`
- **Apply penalty on failure**: Credit penalty (configurable `penalty_serve_failure`, default -50) applied on distributed inference failure; penalty uses `apply_credit_direct` for immediate balance update
- **AllReduce registry cleanup**: `cleanup_stale()` removes entries where the receiver was dropped (timed out), wired into HealthMonitor's periodic 30s tick
- **Pipeline error broadcast**: `broadcast_pipeline_error()` notifies all pipeline participants on distributed inference failure, enabling peers to update shard availability
- **Pipeline affinity (KV cache reuse)**: Multi-turn sessions reuse previous pipeline assignment when all nodes are still connected, avoiding cold KV-cache on every turn
- **Relay service credits**: Tracks relay circuit open/close times in SharedState (`active_relay_circuits` DashMap), accumulates seconds in `relay_seconds_served` atomic counter, drains periodically in CreditLedger to `earn_relay_service()`
- **DHT record verification**: `verify_dht_value()` Ed25519 signature check on all Kademlia `GetRecordOk` results in NetworkManager — unsigned/invalid records are logged and ignored
- **Logprobs in API response**: `sample_token_with_params_and_logprobs()` in tensor_util collects per-token log probabilities via `SamplingContext`, stored in `PipelineExecutor.collected_logprobs` (Mutex), mapped to OpenAI-compatible `ChoiceLogProbs` in the `/v1/chat/completions` response. Works for split model (candle) inference paths

### Security Audit (Phase 16) — ~90 fixes across 5 rounds
- **Round 1-3**: Mandatory gossip signing, transport-authenticated dispatch, RFC 6479 anti-replay, signed DHT records, ephemeral key auth, path traversal fix, HF input validation, constant-time auth, CSP hardening, rate limiter cleanup, queue caps, input limits, WebSocket Origin validation, credit signature verification, XSS fixes
- **Round 4**: StreamingToken auth guard, peer IP bypass scoped to inference paths only, `.env` loader blocks dangerous env vars (LD_PRELOAD/PATH/DYLD_*), TOCTOU guard via `loading_models` DashMap with RAII `LoadGuard`, metadata hostname blocklist (Azure/AWS/DO/Oracle/Alibaba), IPv6 multiaddr extraction
- **Round 5**: All dispatch handlers require `authenticated_sender` (LayerResult, InferenceRequest, PipelineAssignment, InferenceError, TpAllReduceResponse), plaintext fallback removed (seal failure → drop), PEX SSRF filter (private/loopback/link-local IPs), shard serve requires peer_registry membership, pending_tensor_channels capped at 256, pending_tp_partials capped at 512, image_data 20MB cap, PoolMessage identity binding (CreditForward.from_node_id/MemberLeft.node_id must match sender), tool params size limits, lora_adapter validation, peer error body truncation/scrubbing, invite code capped at 4K, MCP research restricted to local/network models

### Frontend Polish
- CSS: removed unused variables, fixed hardcoded colors → CSS vars, removed duplicate rules, `@media (prefers-reduced-motion)`, light theme semantic color overrides
- Accessibility: `role="alert"` + `aria-live` on WebSocket banner, `aria-expanded` on hamburger, `aria-live="polite"` on chat messages, `scope="col"` on table headers
- JS: replaced inline styles with CSS classes, wired aria-expanded toggle

### Bug Fixes (Post-Audit)
- **Critical**: Credit balance overflow (`i64 +=` → `saturating_add`) and missing persistence in `track_forward_participation` — credits now survive daemon restart
- **High**: Divide-by-zero panic from malformed GGUF with `head_count == 0` (4 sites, remotely triggerable via HF probe)
- **High**: Silent null body sent to cloud provider on serialization failure (`unwrap_or_default` → proper error propagation)
- **Medium**: `model_request_counts` DashMap unbounded growth — now gated on registered models only
- **Medium**: `peer_shard_downloads` orphaned entries on peer disconnect — cleanup in ConnectionClosed handler
- **Medium**: Rate limiter cleanup task ignoring shutdown signal — now uses `tokio::select` with `shutdown_rx`

### Infrastructure
- **Workspace migration**: 3-crate Cargo workspace (`swarmllm`, `swarmllm-types`, `swarmllm-frontend`)
- **Ring AllReduce**: Bandwidth-optimal for ≥4 TP ranks, auto-selected by `choose_allreduce_strategy()`
- **Package distribution**: Homebrew formula, AUR PKGBUILD, deb/rpm packages, systemd service file
- **macOS CI**: Re-enabled on macos-15 runner
- **Docker**: Fixed Dockerfiles for workspace build
- 674 tests passing (606 unit + 22 integration + 31 module + 14 yamux + 1 VLM E2E)

### UX & Internationalization
- **i18n** — 20 languages (Arabic, Chinese, Czech, Dutch, English, French, German, Hindi, Indonesian, Italian, Japanese, Korean, Polish, Portuguese, Russian, Spanish, Swedish, Thai, Turkish, Ukrainian, Vietnamese)
- **Theme toggle** — Light / Dark / System theme with persistent preference
- **Basic/Advanced mode** — Toggle for simplified vs power-user UI
- **Plain-English UX pass** — Removed jargon, clearer labels and error messages for beginners
- **Compare UX** — Prompt textarea moved out of collapsed section, All/Local/Cloud filter buttons, chat source indicators, tok/s display fix for slow models (shows 0.5 instead of 0)
- **Provider UX** — `.env` file support for API keys, key source selector (auto/env/dashboard), error badges with click-to-settings
- **GPU OOM → CPU fallback** — Models that exceed GPU VRAM automatically retry on CPU (split fast-path preserved, not slow pipeline path)
- **Anthropic API model routing fix** — Requests now route to the correct model instead of always using the first loaded model

### Codebase Quality
- **Refactored**: `daemon.rs` (4015 lines → module directory), `admin.rs` (4225 lines → 4 modules), `split.rs` (10K lines → 6 modules)
- **Extracted**: `swarmllm-frontend` crate with dev mode for instant UI changes without full rebuild
- 674 tests passing (606 unit + 22 integration + 31 module + 14 yamux + 1 VLM E2E)

### Model Trust & On-Demand Loading (Phase 14)
- **Model Trust System** — demand-driven trust prevents trash models from auto-propagating
  - `ModelTrustLevel` enum: Discovered → Pinned → DemandVerified → NetworkPopular
  - Auto-manage only downloads shards for `DemandVerified`+ or user-`Pinned` models
  - Models promoted to `DemandVerified` after 3 real inference requests
  - `NetworkPopular` promotion when 3+ unique holder nodes serve a model
  - 7-day inactivity decay (Pinned models immune), persisted to redb
  - Trust level exposed in admin API (`trust_level` field on all model objects)
- **On-Demand Shard Loading** — inference requests trigger auto-loading from disk
  - Router detects shards on disk but not loaded in VRAM, triggers `check_and_load_model()`
  - LRU eviction makes room automatically (protected: active pipeline models)
  - Loading coordination via `DashMap<ModelId, Notify>` prevents concurrent loads
  - No more need to pre-load all models at startup
- **Kimi 2.5 support** — `k2*` prefix routing to Moonshot provider
  - Static fallback models: kimi-k2-0527, moonshot-v1-8k/32k/128k
  - Existing kimi* and moonshot-* routing preserved
- **UI improvements**
  - Trust level badges on model cards: Popular (green), Verified (accent), Pinned (yellow), Unverified (gray)
  - HF browser: prominent "On Swarm — N nodes" badge vs "New to network"
  - Download button renamed to "Add to node" (clarifies seed shard semantics)
  - "Local only" indicator when no peers host the model
- **Storage**: `get_all_json()` method on Database for key-value iteration with subkeys
### Claude Code Integration (Phase 13)
- **Full Anthropic Messages API** (`POST /v1/messages`) — complete Claude Code compatibility
  - `tools`, `tool_choice`, `metadata`, `thinking` (extended thinking) request fields
  - `tool_use`, `tool_result`, `thinking`, `redacted_thinking` content blocks
  - `cache_control` on system blocks (Anthropic prompt caching)
  - Full pass-through to Anthropic cloud (all fields preserved including tools and thinking)
  - Anthropic→OpenAI translation proxy for non-Claude cloud models (GPT-4o, DeepSeek, etc.)
  - Tool calls and thinking blocks converted to text for local GGUF inference
  - `ResponseContentBlock` refactored from struct to enum (Text, ToolUse, Thinking variants)
- **MCP `compare` tool** — send same prompt to multiple models concurrently (up to 10)
  - Returns side-by-side results with `content`, `latency_ms`, `input_tokens`, `output_tokens`, `status`
  - Supports local, network, and cloud models in same comparison
  - Routes through `/v1/messages` for consistent routing logic
- **Claude Code as client**: `ANTHROPIC_BASE_URL=http://localhost:8800 claude --model qwen2.5-coder-7b`
- **Model Compare dashboard page** — side-by-side multi-model comparison UI with streaming
- 6 new unit tests (tool_use, tool_result, thinking, tools request, response serialization, internal conversion)
- 665 tests passing (597 unit + 22 integration + 31 module + 14 yamux + 1 VLM E2E)

### Published Benchmark Data
- **GPU (RTX 3070 8GB):** TinyLlama 1.1B 27.2 tok/s, Gemma-2 2B 20.6 tok/s, Phi-3.5 3.8B 46.4 tok/s, Qwen2.5 7B 29.0 tok/s
- **CPU (Ryzen 7 5800H):** TinyLlama 4.2 tok/s, Gemma-2 3.5 tok/s, Phi-3.5 1.8 tok/s, Qwen2.5 2.4 tok/s
- GPU speedups: 6.5x to 25.8x depending on architecture
- Methodology: 100 output tokens, 3-run average, single model loaded, Q4_K_M quantization

## [0.1.0-alpha.1] — 2026-03-07

First public release. Single Rust binary (~31MB) for decentralized P2P LLM inference.

### Inference Engine
- **11 model architectures**: Llama, Llama 4, Qwen2, Qwen 3.5 (hybrid SSM+attention), Gemma/2, Phi-3, Mistral, Starcoder2, DeepSeek-V2/V3 (MoE+MLA), GLM-4
- **4 architectures verified** with real models: Llama (TinyLlama-1.1B), Qwen2 (Qwen2.5-Coder-7B), Phi-3 (Phi-3.5-mini), Gemma2 (Gemma-2-2B-IT)
- **Distributed inference** verified on 2-node real LAN (WSL2 laptop + Proxmox server) with 5 models, crash recovery, auto-reconnect
- **Tensor parallelism** via AllReduce (star topology) with RTT-based LAN peer detection
- **VLM support**: LLaVA-v1.5-7B verified end-to-end (CLIP vision encoder + correct fine-tuned text model from second-state/Llava-v1.5-7B-GGUF), distributed mmproj, chat UI image upload (camera button, paste, drag-drop)
- **LoRA adapters**: per-request loading, verified with Qwen2.5-Coder-7B + rank-16 adapter
- **Speculative decoding** with draft model + rejection sampling
- **Cross-request batching** (GPU batch tensors, configurable `max_batch_size`)
- **Multi-turn KV-cache** with session reuse, cross-request prefix caching, chunked prefill
- **Flash attention** (CPU + GPU) and **paged attention** (CUDA block pool)
- **Structured output**: ResponseFormat API with JSON grammar state machine + schema validation
- **Sampling**: temperature, top-k, top-p, frequency/presence penalty, stop sequences

### API & Compatibility
- **OpenAI-compatible API**: `POST /v1/chat/completions` with streaming (SSE), `tool_calls`, `tool_choice`, `logprobs`, `top_logprobs`, Tool role
- **Anthropic Messages API**: `POST /v1/messages` — full Claude Code compatibility (tools, tool_choice, thinking, cache_control, metadata)
- **MCP server** at `/mcp` — `chat`, `models`, and `compare` (multi-model comparison) tools for Claude Code, Cursor, and MCP-compatible agents
- **12 cloud provider fallback**: OpenAI, Anthropic, DeepSeek, Mistral, Groq, NVIDIA NIM, Cerebras, SambaNova, Fireworks, Together, DeepInfra, Moonshot/Kimi
- **Hidden states API**: `/v1/internal/hidden-states` for research (activation inspection, adapter insertion)
- **Embeddings**: `POST /v1/embeddings`
- **~62 admin REST routes** for dashboard, config, model management, downloads, providers
- **WebSocket** live updates (2s stats + prune event notifications)
- **Prometheus metrics** at `/metrics` (6 gauges + histogram)

### SDKs & Integrations
- **Python SDK**: `pip install swarmllm-client` — sync + async clients, streaming
- **JavaScript/TypeScript SDK**: zero-dependency, streaming support
- **LangChain integration**: `ChatSwarmLLM` provider
- **LlamaIndex integration**: `SwarmLLM` provider
- **Benchmark CLI**: `swarmllm bench` — sequential latency + concurrent throughput, JSON output

### Networking
- **P2P**: libp2p 0.55 with TCP+Yamux (primary) and QUIC transport
- **5-layer discovery**: mDNS (LAN), persistent peer cache (redb), encrypted invite codes, peer exchange (PEX), Kademlia DHT
- **NAT traversal**: libp2p relay circuits + DCUtR hole punching
- **GossipSub**: 6 topics for shard announcements, credit gossip, health, governance
- **Unified protocol**: `/swarmllm/1.0.0` — JSON control messages + binary tensor payloads (type-tag byte)
- **Wire compression**: zstd for tensor payloads

### Security
- **E2E encryption**: X25519 key exchange + ChaCha20-Poly1305 symmetric encryption
- **Forward secrecy**: ephemeral re-keying with key rotation
- **Sealed gossip**: all gossip messages authenticated (no plaintext fallback)
- **Replay protection**: nonce tracking + rejection
- **Shard integrity**: BLAKE3 content hash verified on every load
- **API auth**: Bearer token with auto-generation, loopback-only key retrieval
- **Provider key security**: at-rest encryption (AES-GCM), zeroize on drop, log scrubbing
- **Content-Security-Policy** header, IP-based rate limiting, CORS lockdown, SSRF protection
- **KV-cache privacy mode**: configurable per-session data isolation

### Model Management
- **Shard-only operation**: nodes download individual shards (~512MB each), never need a full model
- **HuggingFace integration**: search, browse, byte-range shard downloads with resume/retry
- **VRAM-aware auto shard management**: rarity-scored acquisition, popularity-based scoring
- **Smart shard pruning**: auto-remove over-replicated shards based on demand, resource pressure, and region diversity
- **Per-shard lock/pin** and per-model prune toggle
- **BLAKE3 integrity verification** on every shard load

### Credit System
- **Credit ledger**: earn credits by serving inference, hosting shards, seeding data
- **4 priority tiers**: Platinum (top 10%), Gold (top 30%), Silver (positive), Bronze (zero/negative)
- **Dual-signed transactions**: Ed25519 signatures from both parties
- **Credit escrow** for large requests
- **Anti-gaming**: rate limits, spot-check verification, subnet clustering detection
- **Sybil resistance**: trust scoring with decay, reputation tracking

### Identity & Pools
- **Ed25519 cryptographic identity** per node
- **Nicknames** with leaderboard
- **Device pools**: multi-device credit pooling with dual-signature invitation protocol

### Frontend
- **Embedded web dashboard** (vanilla HTML/CSS/JS, no build step, < 200KB)
- **4-step setup wizard** for first-run experience
- **Chat interface**: multi-turn + streaming, switchable Linear/Messenger layout, image upload (camera button, paste, drag-drop) for VLM models
- **Model browser**: HuggingFace search, shard grid visualization, download progress
- **Network map**: peer visualization with region grouping
- **Mobile-responsive** layout with dark theme
- **Reasoning model support**: DeepSeek R1 think token rendering

### Operations
- **Single binary**: ~31MB, zero runtime dependencies
- **CLI**: `run`, `status`, `chat`, `bench`, `peers`, `test-split`, `version`
- **Config priority**: CLI flags > env vars (`SWARMLLM_` prefix) > config.toml > defaults
- **Config hot-reload** via SIGHUP or API
- **Graceful shutdown**: SIGTERM handler with subsystem drain
- **Auto-updater**: checks GitHub Releases, downloads + self-replaces with restart prompt
- **JoinSet task supervisor**: automatic restart-on-crash for all 10 subsystems
- **Database**: redb v3 (embedded, ACID, ~15% faster than v2)

### Platform Support
- Linux x86_64 (CPU + CUDA + ROCm)
- macOS aarch64 Apple Silicon (Metal)
- macOS x86_64 Intel (CPU)
- Windows x86_64 (CPU + CUDA)

### Model Loading
- **Auto-extract gguf_header.bin** from shard_000.bin when header is missing (daemon pre-pass)
- **Single-GGUF manifest**: Full GGUF files stored as shard_000.bin generate 1-shard manifests (not split into logical shards)
- **Single-shard mmap fallback**: Models with 1 shard and no tensor entries load via mmap instead of ShardReader
- **Probed flag fix**: Models with gguf_header.bin correctly show as probed in admin API

### Test Suite
- 659 tests: 591 unit + 22 integration + 31 module + 14 yamux + 1 VLM E2E
- All passing, clippy clean, rustfmt clean
- CI: GitHub Actions (fmt → clippy → test → build)

[0.1.0-alpha.1]: https://github.com/enapt/SwarmLLM/releases/tag/v0.1.0-alpha.1
