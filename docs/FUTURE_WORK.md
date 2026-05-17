# Future Work — Out of R110-R115 Scope

Captures items deliberately deferred from the model-management redesign and from prior sweeps. Each entry has enough context that a future implementer (or a future me) can pick it up without re-deriving the rationale.

## Model management — deliberately out of R110-R115

### Inter-pool model sharing policy
**Context.** Pools today are private membership scopes. A model's shards stay within a pool; cross-pool peers don't see each other's hosted shards. The wishlist + capacity computation in R110-R115 also stays scope-local.

**What's needed.** A protocol-level "model interest" channel that pools can opt into. Joining the channel announces "we host model X" without exposing pool composition. Outsiders can see which pools serve a model and route inference accordingly. Likely needs a per-model k-anonymity floor (refuse to advertise if pool has < N hosts) so it can't be used to enumerate small pools.

**Why deferred.** Designing the privacy/trust boundary is a real architectural decision. Doing it inside R111's wishlist refactor would mix concerns.

_**Partially closed R134** (2026-05-16) — discovery layer shipped; the
routing layer remains deferred pending a separate user discussion about
the private-mode contract. What landed:_

_- New `SwarmMessage::PoolModelAvailability { pool_id, model_ids,
  timestamp_ms, owner_signature }` on the existing `swarm/regions`
  GossipSub topic. Domain-separated BLAKE3 sign payload
  `pool_model_avail_v1` binds pool id + sorted model id list +
  timestamp, so a replayed announcement across pools or with a
  tampered model list fails verification._
_- Opt-in via `pool.share_model_catalog` (default `false`). Even when
  on, k-anonymity floor `share_model_catalog_min_members` (default 3)
  blocks pools smaller than the floor from publishing — prevents the
  channel from being used to enumerate small private pools._
_- Receivers cache in `state.credits.foreign_pool_catalog: DashMap<
  (PoolId, ModelId), received_at_ms>` (cap 5000 with oldest-first
  eviction, 2h freshness window — matches the wishlist signal cadence).
  Always-on ingest so privacy-conscious nodes benefit from the
  discovery signal without publishing their own catalog._
_- Publisher: `HealthMonitor::broadcast_pool_model_availability` fires
  on the same gossip cadence as wishlist + region summary. Pool owner
  only — non-owners never publish even if the flag is on._
_- Admin surface: `GET /api/admin/foreign-pool-catalog` returns the
  cached signal grouped by pool with stale trim._
_- 1 new unit test covers determinism + sign/verify roundtrip + tamper
  detection._

_**Frontend tile closed R134.5** (2026-05-16). Models → Running-now
subview gets a "Models other pools serve" tile fed from a size-bounded
snapshot in the WS stats payload. Hidden when the catalog is empty so
users not in a federated setup never see it. +3 i18n keys × 21
locales. Subline "Discovery only — your inference stays in your pool"
reinforces the unchanged contract._

_**Routing layer closed R134.7** (2026-05-16). Cross-pool routing is
now wired but stays strictly opt-in. New `pool.allow_cross_pool_inference`
flag (default false) gates the contract change; on top of that
`private_mode` must also be on (otherwise normal global routing
already applies and no fallback is needed). New
`pool::scope::cross_pool_extras(state, &model_id)` returns the set of
NodeIds in foreign pools that have advertised this specific model via
`foreign_pool_catalog`. The scheduler unions this with the existing
`allowed_node_set` so cross-pool requests only fall through when the
local pool genuinely can't serve the model (`any_local_pool_holder ==
false`). Three guard rails:_
_1. Both sides must opt in — the FOREIGN pool must have advertised
   the model via `share_model_catalog`, AND the LOCAL pool must have
   set `allow_cross_pool_inference`. No drive-by routing._
_2. Per-model scope — extras are only computed for the model that's
   currently being requested AND that the local pool can't serve. The
   "stays in pool" contract is preserved for any model the local
   pool does host._
_3. The `PrivateModeUnavailable` error builder also includes extras
   so error messages stay consistent with what was actually
   considered eligible._
_2 unit tests cover the flag-off and private-mode-off early-return
paths. Billing model + UI consent toast deferred — operators flip the
config flag directly today; the dashboard surface can grow a consent
banner in a future dashboard refresh._

---

### Model quality benchmarking
**Context.** Wishlist scores rely on HF download count, gossip-derived demand, and rarity. None of these correlate well with answer quality.

**What's needed.** Optional local eval harness — a small set of held-out prompts (MMLU subset, GSM8K-tiny, HumanEval-tiny) the node runs once after a model becomes serveable. Results gossip via `swarm/quality` topic. Aggregate becomes a quality factor in wishlist scoring.

**Why deferred.** Eval framework is a substantial subsystem. Also raises governance questions (which prompts? versioned how? cheating?) that are out of scope for the current redesign.

---

### Quantisation choice automation
**Context.** Today the user picks `Q4_K_M` / `Q5_K_M` / `Q8_0` etc. when adding a model. A 70B model at Q4 is ~40 GB; same model at Q8 is ~70 GB. Auto-manage doesn't reason about quantisation level — it treats each quant as a distinct model.

**What's needed.** A capability-aware quant selector: given the swarm's aggregate VRAM and the model's parameter count, pick the highest-quality quant that fits with reasonable replication. UI surface: "We're hosting Q4_K_M because the swarm only has X TB; with 5 more nodes we'd switch to Q5_K_M."

**Why deferred.** Touches the GGUF probe path, the ModelManifest schema (would need a `quant_family` link), and the wishlist evaluation function. Big change, separate phase.

_**Closed R133** (2026-05-15) — recommendation surface landed; the
auto-action layer (actually switching which quant the daemon downloads)
remains a follow-on for the user to opt into. What shipped:_

_- `Quantization` enum expanded from 5 → 29 variants (K-quants Q2_K..Q6_K,
  legacy Q4_0..Q5_1+Q8_0, I-quants IQ1_S..IQ4_NL, floats F16/BF16/F32,
  Unknown fallback). Each carries `parse(str)`, `bits_per_weight()`,
  `quality_score()` (calibrated against llama.cpp perplexity-loss
  tables) and `label()`._
_- New `model/auto_manage/quant.rs` module: groups registry models by
  inferred base name (model name with quant tag stripped via
  `inferred_base_name`), picks the highest-quality variant whose
  VRAM footprint fits the swarm budget (local GPU VRAM OR the
  aggregate pool VRAM divided by a 3-replica target). Rationale
  surfaced as an i18n-formatted tag (`quant.rec.{best_fit,
  would_upgrade, too_big, cpu_only, no_variants}`) with parameter
  interpolation (`|next=Q5_K_M&need_mb=15000`) matching the existing
  `wishlist.why.*` pattern._
_- `state.models.quant_recommendations: ArcSwap<QuantRecommendations>`
  refreshed alongside the wishlist on every auto-manage tick AND on
  every WS stats build._
_- New `GET /api/admin/quant-recommendations` endpoint exposes the
  cached snapshot to the dashboard / external integrators._
_- 5 new i18n keys translated across all 21 locales._
_- 7 unit tests including a registry-backed end-to-end._
_- Manifest schema NOT changed — recommender derives the family
  grouping on-the-fly to keep the wire format stable._

_**Frontend tile closed R134** (2026-05-16). Models → Running-now subview
gets a "Quality tips for your hosted models" tile (`#quant-tips` +
`_renderQuantTips` in swarm-tab.js) that only renders when there's an
actionable hint (`would_upgrade` or `too_big` — `best_fit` rows hide
so a swarm already running the optimal quant sees nothing). +3 i18n
keys × 21 locales (`quant.tips_title`, `quant.tips_sub`,
`quant.tip_current`). WS payload picks up `quant_recommendations` so
the tile updates in real-time without a dedicated REST round-trip._

_**Auto-action layer closed R134.6** (2026-05-16). New
`apply_quant_auto_action` walks the recommendation snapshot once per
auto-manage tick (after `refresh_quant_recommendations`) and promotes
the recommended variant's `ModelTrustInfo` to `DemandVerified` for any
family where the user currently hosts a *different* variant. The normal
scoring/download path then opportunistically acquires the better quant.
Opt-in via `auto_manage.auto_switch_quants` (default `false`).
Non-destructive: the OLD variant is NOT proactively pruned, so there's
no in-flight inference disruption window — standard prune cycle
handles dedup once VRAM pressure hits. Net effect: when the flag is on,
running a Q4_K_M and the swarm grows enough to fit Q5_K_M, the daemon
quietly starts downloading Q5_K_M; both stay hosted until the natural
prune cycle decides one is over-replicated. Activity events emitted
post-iteration (so the broadcast send never interacts with the trust
DashMap entry guard). 1 unit test covers the flag-off no-op._

---

### GGUF conversion / fine-tune support
**Context.** Users can only add pre-quantised GGUFs from HF. Fine-tunes / LoRA adapters / merged models that exist only as PyTorch checkpoints aren't reachable.

**What's needed.** Either (a) integrate a conversion pipeline (llama.cpp's `convert_hf_to_gguf.py` driven from the daemon), or (b) defer to external tooling and let users upload a converted GGUF.

**Why deferred.** Conversion is heavy (CPU-bound, multi-minute per model) and adds Python toolchain dependency. Out of scope for a Rust-only daemon.

---

### Cross-pool wishlist coordination
**Context.** Each node maintains its own wishlist (R111). Two pools that should converge on the same models for shared regional demand currently don't share their wishlists.

**What's needed.** Optional `WishlistAnnouncement` gossip variant; nodes opt-in to publishing their top-K wishlist entries; receivers boost matching entries in their own wishlist.

**Why deferred.** R111 already gets convergence indirectly through `region_demand` and consistent-hash placement. Direct gossip is an optimisation, not a correctness fix.

_**Closed R130** (2026-05-15). `SwarmMessage::WishlistAnnouncement` carries
`(publisher, top-K (model_id, 0..100 score), timestamp_ms)` on the existing
`swarm/regions` topic, reusing the same 30s broadcast cadence as
`RegionShardSummary`. Inbound entries land in
`state.models.foreign_wishlist: DashMap<(NodeId, ModelId), (score, ts_ms)>`
(capped at `MAX_FOREIGN_WISHLIST_ENTRIES = 10_000`, stale entries pruned
after `FOREIGN_WISHLIST_MAX_AGE_MS = 2h` on read). `compute_wishlist`
adds a 0..10 boost = `10 * log10(publisher_count+1).min(1.0) * max_score/100`,
blending breadth (how many nodes care) with depth (how strongly the
loudest voter cares). New `wishlist.why.other_nodes_want_this` i18n tag
translated across all 21 locales. Publishing is opt-in via
`auto_manage.wishlist_gossip_publish` (default `false` — privacy
default); the receive side is always on so privacy-conscious nodes
still benefit from the signal without leaking their own interests.
Wire schema only exposes model granularity — does not leak pool
composition, region, or per-shard interest. 3 unit tests for
`apply_wishlist_announcement`._

---

### Smarter eviction policy under multi-tenancy
**Context.** R110's eviction guard prevents pruning shards in active sessions, but `prune.rs` still uses simple "highest score" selection within the prunable set. A future-considered model might be evicted just before demand spikes.

**What's needed.** Predictive eviction: project demand forward 1-2 hours, weigh against immediate VRAM pressure. Possibly time-window the eviction so a model that might be needed in 30 minutes survives.

**Why deferred.** Requires demand forecasting infrastructure that doesn't exist yet.

_**Closed R134.7** (2026-05-16). Time-windowed protection landed without
the heavy forecasting subsystem — `prune.rs` reads
`state.models.model_trust.get(&model_id).last_request_at` (already
updated per request) and subtracts `RECENT_REQUEST_PENALTY = 1.5` from
the prune score when the last request is within
`RECENT_REQUEST_PROTECT_SECS = 3600`. Effect: a model that served a
swarm request in the last hour is protected from eviction regardless of
replication ratio. The penalty is calibrated to dominate the strongest
existing `region_demand` signal (max 1.0), so local recent usage out-
weighs cross-region averaging at the local-node prune decision. The
"forecast next 30 min" intuition is captured indirectly via "I used it
in the last 60 min" without standing up a separate prediction
pipeline. 2 unit tests verify the constants stay consistent +
fresh-vs-stale scoring discrimination._

---

### Decentralised reputation for HF model legitimacy
**Context.** R112's HfWatcher trusts HF's `downloads` count as a quality proxy. A coordinated download-pump on HF could trick auto-manage into wasting resources on a junk model.

**What's needed.** Cross-reference HF downloads with swarm-side metrics (request volume, completion rate, post-eval quality if R-future-eval ships). A model that trends on HF but generates zero swarm requests over 7 days gets de-promoted.

**Why deferred.** Anti-gaming is a substantial design space. The 24h-aging + min-100k-downloads gate in R112 covers the obvious case.

_**Closed R134** (2026-05-16). `ModelTrustInfo` gains two anti-gaming
fields (both `#[serde(default)]` for backward-compat with prior DB
contents):_

_- `last_auto_promoted_at: Option<DateTime<Utc>>` — set by the watcher
  on each `Discovered → DemandVerified` lift._
_- `failed_promotions: u32` — strike counter, bumped by `maybe_decay`
  when an auto-promoted model decays back to `Discovered` with
  `total_requests == 0`, and reset by `record_request` on the first
  real swarm request (so a model that briefly fell off the radar but
  then earned real usage is fully forgiven)._

_`HfWatcher::should_auto_promote` is the gate: virgin entries pass
unconditionally; subsequent attempts enforce a linear cooldown of
`7 * failed_promotions` days (capped at 60d). After
`MAX_AUTO_PROMOTION_FAILURES = 4` strikes the model is locked out from
auto-promotion entirely — only an explicit user pin via the admin API
can lift it. This defeats the "pump HF downloads then watch SwarmLLM
auto-host a junk model" attack without permanently blacklisting models
that simply haven't been discovered yet. Completion-rate signal not
wired in this pass — current eviction reasons + activity events already
surface failed inference, and adding a separate ratio risks
double-counting until a proper telemetry pipeline ships. 8 unit tests
cover virgin / pinned / cap / cooldown / growing-cooldown /
decay-bumps-strike / no-strike-when-real-usage / record-clears-strikes._

---

## Audit-deferred items (from sweep rounds R100-R109)

### MoE routing per-arch config
**Context.** R109 documented that `topk_cpu` is mathematically equivalent to Mixtral / Qwen3-with-norm. DeepSeek-V2 strict (no renorm) and V3 (sigmoid) need different paths.

**What's needed.** `ModelArchitecture` extension: `routing_mode: RoutingMode { Softmax, SoftmaxNoNorm, Sigmoid }`, plumbed from GGUF metadata (`{arch}.expert_norm`, `{arch}.expert_gating_func`). Conditional path in `MoeFfn::forward`.

**Why deferred.** No live model in this codebase exercises the divergent paths today. Would be wasted work without a test corpus.

**Sweep log:** `src/inference/layers/mod.rs:397` (R108 finding).

_**Closed R132** (2026-05-15). New `MoeGatingFunc { Softmax, Sigmoid }` +
`MoeRoutingConfig { gating_func, renormalize_weights }` in
`src/inference/layers/mod.rs`. Carried directly on `MoeFfn` (not on the
manifest schema — kept off `ModelArchitecture` to avoid a wire-format
version bump for a GGUF-loader-only concern; the manifest still
identifies models, the runtime resolves routing from GGUF metadata).
`topk_cpu` takes the config and branches on the four combinations,
keeping the Softmax+renorm fast path (existing algebraic identity).
GGUF loader reads `{arch}.expert_gating_func` (uint: 1 = softmax,
2 = sigmoid — matches llama.cpp's `LLM_EXPERT_GATING_FUNC_*` enum)
and `{arch}.expert_weights_norm` (bool) once and threads through all
3 `MoeFfn` construction sites. Default (Softmax + renormalize) matches
Mixtral / Qwen3-MoE (with `norm_topk_prob=true`) / Llama 4 / DeepSeek-V3
default — historical behaviour preserved when metadata is missing.
Now correctly handles DeepSeek-V2 strict (softmax + no renorm) and
DeepSeek-V3 sigmoid gating from the same code path.

Tests: 3 new in `inference/split/tests/moe_mla.rs` covering each
non-default combination. Numerical correctness referenced against
llama.cpp `build_moe_ffn` semantics and HuggingFace
`modeling_deepseek_v3.py::topk_weights`. Live-model integration
remains the original deferral reason — no DeepSeek-V2/V3 GGUF in the
test fixtures — but the routing path is now exercised by unit tests
and ready for the first live-corpus addition._

---

_(R126 closures: matched_stop_sequence wire plumbing, cross-node logprobs, cancel-over-wire for remote-generate, TrustManager bulk hydrate, manifest tensor-cap ActivityEvent — moved from this section.)_

---

### Pool-state gossip coalescing under high churn
**Context.** Every join/leave triggers a full `PoolState` re-gossip. A pool with 50 members rotating membership produces pool_state floods.

**What's needed.** Diff-based gossip: send only changed members + a checksum; receiver validates checksum and falls back to full state on mismatch. Or epoch-based — gossip the full state every K epochs but interspersed with diffs.

**Why deferred.** Existing 5-min coalescing window in `gossip_pool_state` keeps the load tractable for typical pools.

_**Partially closed R131** (2026-05-15) via the "Or epoch-based" path
explicitly invited above. `PoolManager` gains `last_pool_gossip_at` +
`pool_gossip_dirty` debounce state with a 15s minimum interval between
broadcasts. Post-acceptance handler now routes through new
`maybe_gossip_pool_state` (the periodic full-broadcast tick bypasses the
gate — anti-poison + late-joiner recovery). A 3s coalesce timer in the
select! loop drains the dirty flag once the cooldown elapses. 50
acceptances in <15s now coalesce to ≤2 broadcasts (one immediate, one
trailing). 3 unit tests verify the state machine._

_**Closed R134** (2026-05-16). `PoolState` gains a `generation: u64`
counter (default 0 — backward-compatible). New `PoolMessage::StateDiff`
variant carries `(pool_id, parent_generation, new_generation,
added_members, removed_node_ids, shard_pins?, total_lifetime_credits?,
member_credit_split_pct?, state_checksum, timestamp_ms,
owner_signature)`. Domain-separated BLAKE3 sign payload
`pool_state_diff_v1` binds pool id + generation transition + post-apply
checksum + wire timestamp so a replayed diff across pools, across
generations, or with a swapped checksum fails signature verification.
Receivers also recompute `pool_state_checksum` post-apply and reject
any diff that lands on a different state than the owner intended; each
added member's `acceptance_signature` is verified the same way the
StateGossip handler does. Owner forces a fresh full broadcast every
`MAX_DIFFS_BEFORE_FULL = 4` diffs so late joiners recover within bounded
time. Opt-in via `pool.state_diff_gossip` (default `false`) so the
legacy wire path stays unchanged on existing deployments until a WAN
bench shows the trailing-full-state broadcast is actually
bandwidth-constrained for the operator's pool size. 5 new unit tests
cover diff-off / diff-on / cap-forces-full / receiver-applies-and-
advances / receiver-drops-stale-parent. 948 lib tests pass._

---

_(R121 Setup-wizard contribution toggle closed in R126.)_

---

### Ease-of-use audit follow-ups — bigger UX changes (R125 follow-up)

**Context.** The 2026-05-13 ease-of-use audit (R125) applied ~40 copy fixes
to make the UI usable by non-technical users. Three structural / behavioural
items came up that warrant direction beyond a copy refresh:

1. **README architecture section is intimidating.** _(Closed R126: wrapped behind `<details>` titled "Implementation details (for contributors)".)_

2. **Header is overloaded on first load.** _(Closed R127:
   `frontend/js/components/welcome.js` + `#welcome-modal` in `index.html`
   ship a one-time tour overlay highlighting four key elements (model
   picker / + Find model / auto-manage / settings). Fires from
   `App.setup.finish()` and `App.setup.complete()`, and on first load if
   either flag is already set but `WELCOME_SEEN_KEY` isn't. `Got it`
   button + close icon + backdrop click all dismiss & persist the seen
   flag. `Show welcome tour` button in Settings re-opens without
   clearing the flag. 13 new i18n keys (12 welcome.* + close_aria) +
   2 settings.* keys translated across all 21 locales — verified live
   in English, Japanese, and Arabic (RTL flips correctly).)_

3. **`activity.worker_*` events duplicate `activity.model_*` events.** _(Closed R126: both emit sites removed from `process_pool.rs`; i18n keys deleted from all 21 locales; `tracing::info!` retained for operator debugging.)_

4. **`models.hf_score_breakdown` exposes a 4-component score
   decomposition.** _(Closed R127: i18n key was orphaned — the swarm-tab
   HF browser already renders a single human verdict via `_fitPill()`
   from the backend's `fits_boomerang`/`fits_shard`/`network_replicas`
   signals. Removed `models.hf_score_breakdown` + 3 sibling orphans
   (`hf_score_pts`, `hf_on_swarm`, `likes_count`) from all 21 locales
   per the R120 verification-gating rule. Backend `score_breakdown`
   JSON field retained as a dev-only surface.)_

5. **GGUF metadata panel surfaces raw hyperparameters.** _(Closed R126: renamed `models.metadata_header` to "Technical Details" across 21 locales; refactored `renderMetadataPanel()` to split basic params (context, layers, embedding, heads, vocab, tokenizer model) from collapsible `<details>` "Advanced" sub-section that hides KV heads / RoPE / RMS epsilon / BOS-EOS-padding ids / tensor offsets.)_

6. **`models.encrypted_pipeline` and `enc.unprotected_detail`** _(Closed R126: refreshed 19 `enc.*` + `models.encrypted_pipeline` keys across 21 locales — "first piece of the model" / "last piece of the model" replace shard jargon; copy honestly distinguishes "end-to-end encrypted" (when user holds both endpoints) from "encrypted in transit" (when entry/exit nodes are remote).)_

7. **`dashboard.api_log_link` text stays English in 20 non-English
   locales.** _(Closed R127: translated "View API request log →" across
   all 21 locales. Arabic flips the arrow to `←` for RTL.)_

8. **46 country names hardcoded English in `network-map.js:430`.**
   _(Closed R127: replaced the hand-maintained `countryNames` map with
   `Intl.DisplayNames([currentLocale], { type: 'region' })` keyed off
   `I18n.getLang()`, with a `_displayNamesCache` to amortise the cost
   per locale. Browser support: Chrome 81+/FF 86+/Safari 14.1+. Falls
   back to the raw ISO code on the very-old-browser path. Covers every
   ISO 3166-1 alpha-2 code in the native language of each locale,
   without 966 hand-translated entries.)_

**Status.** All R125 follow-up items now closed. R126 closed
#1/#3/#5/#6; R127 closed #2/#4/#7/#8.

---

_(R121 follow-up — True global holder count: closed R127. `ModelRegistry`
gained `global_holder_count: DashMap<ShardId, u32>` plus `record_/get_`
accessors. `network/manager/dht.rs::handle_dht_providers_found` records
`providers.len()` (the raw PeerId count, not the resolved one) on every
DHT GetProviders response. `model/auto_manage/prune.rs` uses
`max(cached_holder_count, global_holder_count)` as the redundancy_ratio
numerator and for the severe-saturation bonus, while gates / region /
busy / rarest checks keep using the filtered-live cache. Stale entries
are cleared in `remove_all_model_shards`. Test: `registry::tests::
global_holder_count_overrides_local_cap`.)_

---

_(R104 follow-up — auto-manage interval hot-reload spurious first tick:
closed R128. `src/model/auto_manage/manager.rs::run` now calls
`interval.tick().await` immediately after rebuilding the interval in
the `config_watch_rx.changed()` arm, mirroring the existing pattern at
line 260 for `request_reset_interval`. Without this, `tokio::time::
interval`'s first `.tick()` fires at t=0 — every interval hot-reload
fired a spurious auto-manage evaluation cycle the moment the operator
changed the value, instead of waiting the new interval. Tracking
entry was the only thing in the original deferred decision; no
architectural change.)_

---

## Audit deferral — R128 sweep-log triage

The 2026-05-14 audit pass reviewed all 88 sweep-log `deferred` entries
and confirmed the remaining items genuinely require user discussion or
have explicit "won't fix" semantics. Concretely:

- **R103 `x-swarm-forwarded` dual-gate dead code** (`src/api/middleware.rs:432`)
  — already cleaned up; the inline comment at lines 461-466 explains
  what was dead and why falling through to Bearer auth is correct.
  Move to `fixed` on next sweep close.

- **R103 `batch_scheduler_loop` no JoinHandle / catch_unwind**
  (`src/inference/process_pool.rs:792`) — kept deferred. A panic in
  the scheduler degrades to direct-execution fallback (forward() only
  uses the channel when batching is on and the channel is alive), so
  this is performance-degradation not correctness. Adding catch_unwind
  across an async boundary requires structural changes (UnwindSafe
  bounds, poison handling) that aren't justified by the failure mode.
  Won't fix unless a concrete panic site appears.

- **R105 libp2p relay `..Default::default()` "possibly open"**
  (`src/network/relay.rs:51`) — verified safe against libp2p-relay
  0.21.1: the only fields covered by the default fallback are
  `reservation_rate_limiters` and `circuit_src_rate_limiters`, both of
  which default to **conservative** per-peer/per-IP rate limits
  (30/peer/2min, 60/ip/min). No "open" surface. Move to `wontfix` on
  next sweep close.

- **R105 latency sample ring time-coverage**
  (`src/api/metrics.rs:160`) — kept deferred. Requires changing the
  sample storage from `VecDeque<f64>` to `VecDeque<(Instant, f64)>`
  plus drop-by-age on insert and at every `compute_latency_stats`
  call. Affects multiple call sites and adds memory overhead for a
  cosmetic improvement on lightly-loaded nodes. Not correctness;
  defer until p99 inaccuracy becomes a real operator complaint.

- All remaining deferred entries (Qwen 3.5 DeltaNet shape verification,
  GossipSub PeerScore tuning, DHT provider capability challenge, etc.)
  are explicitly architectural / design discussions matching the
  user's "defer items needing discussion" directive.

---

## R135 cross-pool review — hot-reloadable flag deferral

`pool.allow_cross_pool_inference` and `pool.share_model_catalog` are
read from `state.config.pool` which is startup-frozen. Unlike
`private_mode` (an `AtomicBool` on `state.credits`) and
`contribution_auto` (an `AtomicBool` on `state.models`), these flags
have no runtime-toggle path. A user who disables
`allow_cross_pool_inference` via a future
`PUT /api/admin/config { allow_cross_pool_inference: false }` would
not actually stop cross-pool routing until daemon restart.

**Status.** Not exploitable today — the API endpoint doesn't currently
expose these fields, so flipping them via the dashboard isn't
possible. The risk is purely "future API endpoint will silently
no-op until restart". Closed properly in a follow-up that:
1. Adds `allow_cross_pool_inference: AtomicBool` + `share_model_catalog:
   AtomicBool` to `state.credits` (sub-struct accessor rule —
   architecture.md).
2. Mirrors them from `state.config.pool` at startup in
   `SharedState::new()`.
3. Adds `allow_cross_pool_inference` + `share_model_catalog` Option
   fields to `ConfigUpdate` in `api/admin.rs`; on PUT, writes the
   atomic AND persists the config TOML.
4. Switches `pool::scope::cross_pool_extras` and
   `health/monitor.rs::broadcast_pool_model_availability` to read
   from the atomic.

Pattern is identical to R121's `contribution_auto`. Mechanical work
but lives in the user-facing config surface, so deferred until a
config-API pass makes sense.

---

## R135 sweep — security findings deferred for discussion

The R135 review surfaced two signature/security items that touch wire
formats and require user direction before changing:

### Acceptance signature timestamp omission
**Context.** `acceptance_payload` in `src/pool/crypto.rs:419` signs
`(PREFIX_ACCEPTANCE | invitation_id | pool_id | invitee_node_id)`. No
timestamp, no nonce. The invitation itself has an `expires_at` but the
acceptance does not bind to it.

**Risk.** An adversary who captures an acceptance signature in transit
(or from a compromised node's storage) can in principle replay it.
Practical exploitability is low — `invitation_id` is UUIDv4 (~2^122
collision space) and accepted invitations are tracked in member state,
so double-acceptance is already prevented by the pool registry. The
finding is real but the attack scenario requires the same
`(invitation_id, pool_id, invitee_node_id)` triple to be re-used, which
the inviter controls.

**What's needed.** Either (a) extend `acceptance_payload` to bind
`invitation.expires_at` so the sig is implicitly time-bounded, or
(b) add a per-acceptance nonce to `PoolMembership`. Both are
wire-format changes; (a) is simpler and matches the existing
invitation expiry contract.

**Why deferred.** Wire-format change with backward-compat implications
across all pool members. Want explicit user sign-off on the
compat strategy (versioned variant vs. flag-day).

### `apply_pool_model_availability` cap eviction relies on single-task dispatch
**Context.** R135 reordered the eviction in `daemon/state/credits.rs`
to be POST-insert, which makes the post-condition `catalog.len() <=
max_entries` structural rather than depending on the pre-insert size
estimate being accurate. But the eviction loop itself still scans
the DashMap to find the oldest entry, which is O(n) per eviction —
not O(log n). For `max_entries = 5000` this is 25M ops per drain in
the worst case (full re-fill from a single 5000-entry announcement,
which is bounded above by `MAX_POOL_MODEL_ANNOUNCE_ENTRIES = 128`,
so the real worst case is bounded). Fine today, would want
attention if either cap grows materially.

**Why deferred.** Not a correctness bug. Performance discussion for
when the network scales past current cap assumptions.

---

## Inference performance — research backlog (R135 brief)

Compiled 2026-05-16 from a survey of state-of-the-art LLM inference
optimization (FlashAttention-3/4, FlowSpec, P-EAGLE, EAGLE-3, DSD,
PowerInfer, DejaVu, SGLang RadixAttention, vLLM PagedAttention,
Parallax, NVIDIA Dynamo/NIXL, NVFP4 KV, KIVI, etc.) cross-referenced
against SwarmLLM's existing inference stack.

### Already shipped (do not re-research)

The following are NOT gaps — they are live:

- **SWIFT self-speculative decoding** (`inference/swift.rs`,
  arxiv 2410.06916) — layer-skip draft from the same model. v2
  calibration with multiple candidate patterns.
- **DSD (Distributed Speculative Decoding)** (`pipeline/dsd.rs`,
  `inference/dsd_controller.rs`).
- **Sarathi-style chunked prefill** via the slot table state
  machine (`inference/slot_table.rs`) — `Prefilling` slots advance
  by `prefill_chunk_tokens` per decode tick.
- **Continuous batching** via the same slot table.
- **Tensor parallelism with AllReduce** (`pipeline/tensor_parallel.rs`,
  `inference/allreduce.rs`).
- **Activation compression** (`tensor_util.rs::tensor_to_bytes_q8_0`,
  Q8_0 group-32, ~3.76× over f32) — config-gated by
  `inference.activation_compression` (default false). Wired through
  the worker IPC path.
- **Prefix cache** (`split/prefix_cache.rs`, 53K bytes).
- **Pipeline failover** with hot-standby nodes per segment.

### Tier 1 — high-leverage, low-risk wins (do these first)

#### A. Default-on activation compression with quality gate
**Context.** `inference.activation_compression` defaults to `false`.
For P2P/WAN inference the wire is the bottleneck — every layer
boundary ships ~`hidden_dim × 4` bytes/token in f32, ~`hidden_dim × 1`
in Q8_0. A 4096-dim model on a 100 Mbps link spends 1.3ms vs 0.33ms
per layer hop on f32 vs Q8_0; multiplied across 4-8 pipeline hops per
token this dominates latency.

**State of the implementation today** (verified R135).
`src/inference/quant.rs` ships a group-32 Q8_0 with per-block f16
scale — same algorithm as llama.cpp's weight Q8_0. Per-block scale
handles activation outliers (GLU-spike concern from the literature
applies to per-tensor scale only; group-32 adapts to local dynamic
range). Wire-format already supports it via `TensorFormat::INT8`
(tag 2). End-to-end plumbing is in
`model_worker.rs:817-823`: `if activation_compression {
tensor_to_bytes_q8_0(output_t) } else { tensor_to_bytes(output_t) }`.
Process pool fans the flag through `AtomicBool::set_activation_compression`.
The dispatch path's `decode_layer_forward_encrypted` reads the dtype
tag from the byte stream so receiver doesn't need a separate flag —
sender decides per-forward.

**What the literature says about quality.**
- llama.cpp empirical: Q8_0 perplexity ≈ FP16 perplexity ± 0.01-0.05
  on standard benchmarks (Wikitext-2 ≈ 7.49 for both).
- ATQ (2024): INT8 W8A8 keeps perplexity Δ < 1.0 on OPT + LLaMA.
- HF docs: 8-bit activation Δ < 2% perplexity in typical case.
- GLU variants (Gemma, Llama-3, Phi-3): activation-spike risk in
  GATE × UP intermediate is real but our Q8_0 applies to hidden
  state OUT of layer, not the FFN-internal intermediate, so the
  spike is mostly absorbed by the residual + layernorm before
  quantization.

**What's needed to ship default-on.**
1. **Quality gate**: a one-shot calibration on first
   `attach_request_to_model` per model. Ship both formats over the
   IPC pair for ONE prefill, compute relative L∞ error on the
   output hidden state, abort Q8_0 default if `err > 1e-2`. Cache
   the verdict in `state.models.activation_compression_per_model:
   DashMap<ModelId, bool>` so subsequent requests skip recalibration.
2. **Per-request override**: extend `SamplingParams` with
   `wire_precision: Option<TensorFormat>` so coding agents with
   logprobs can opt back to f32.
3. **Diagnostic**: emit `ActivityEvent { kind: "activation_compression_decision",
   model_id, decision: "q8_0" | "f32", calibration_error_l_inf }` so
   the dashboard can show "Hosting Llama-3-8B at Q8_0 — measured
   wire-quality 0.003".
4. **Default flip** in `config/inference.rs`: `activation_compression:
   true` for non-private-mode, `false` when private-mode-local-only
   (the saving is wasted on loopback).

**Effort.** 1-2 days net.
- Quality gate: ~150 LOC + 1 test fixture (use TinyLlama-1.1B
  prefill output as ground truth).
- Override field: ~30 LOC.
- Activity event + i18n: ~50 LOC + 1 i18n key × 21 locales.
- Default flip: 1 line, gated on a `--features default-q8` build
  feature for risk staging.

**Why deferred.** Default change touches every distributed
inference request. Wants explicit user authorization. The Q8_0
implementation itself is mature and shipped; this deferral is
purely about flipping the default safely.

#### B. Tail-latency hedging for slow pipeline hops
**Context.** P2P pipeline hops have long-tail latency (geo, NAT
traversal, transient CPU pressure). `network/manager` already has
`pending_rr_observability` with a 10s ACK timeout; the
`is_transient_remote_failure` retry kicks in only AFTER timeout.

**What's needed.** Pre-emptive hedging: if a `LayerForward` rr send
has not produced a `LayerResult` within `p99 × 1.5` for that
segment's holder, fire a duplicate to the next-best holder.
Whichever Response arrives first is used; the other is logged as a
cancelled hedge. The scheduler already knows the
`busy_until_ms_per_holder` cache so picking a hedge target is cheap.
Add a per-segment `latency_ewma_ms` on `ShardHolderStats` to drive
the threshold.

**Why deferred.** Need to size the duplicate-work budget — at 5%
hedge rate the network cost is negligible, at 30% it doubles credit
spend. Default 5% threshold is the natural starting point but the
operator-facing dial belongs in a config discussion.

**Effort.** 2-3 days; touches `dispatch/layer_forward.rs`, the
scheduler, and adds a config knob with sane default.

#### C. EAGLE-3 self-speculative draft head
**Context.** SWIFT is layer-skip — it reuses the same weights but a
shorter compute. EAGLE-3 trains a 1-2 layer draft head that reads
the target's intermediate features and predicts. EAGLE-3 reports
3.0×-6.5× over vanilla; SWIFT plateaus around 1.8-2.2×.

**What's needed.** A draft-head loader path. The draft head is a
~30M-100M parameter file alongside the target GGUF. Distribute
draft heads through the existing manifest/shard system (one shard,
no splitting). At decode time the worker maintains both target and
draft-head residency on the LOCAL node only — the draft-head
forward never goes over the wire. Verification fuses with the
existing target-pass.

**Why deferred.** EAGLE-3 weights need to be trained (or sourced)
per target model. There's a community repo (SafeAILab/EAGLE) with
heads for common models but the ones that match SwarmLLM's test set
(TinyLlama / Qwen2.5 / Phi-3.5 / Gemma-2) are not all available.
Production rollout depends on a draft-head distribution policy.

**Effort.** 1-2 weeks. New `inference/eagle.rs` module + manifest
extension + auto-manage acquisition path.

### Tier 2 — moderate-leverage, moderate complexity

#### D. Tree-based draft expansion (FlowSpec-style)
**Context.** Current `pipeline/speculative.rs` ships a *linear chain*
of γ draft tokens for verification. FlowSpec ships a *tree* of K
candidates per position (e.g., top-3 at position +1 × top-2 at +2 ×
top-2 at +3 = 12-leaf tree). Verification accepts the longest
matching root-to-leaf path; rejection branches are pruned.

**What's needed.** Extend `LayerForward.draft_tokens: Vec<u32>` to
`draft_tree: DraftTreeNode { token, children: Vec<DraftTreeNode> }`,
gated on the same flag. Wire-format extension; `pack_verify_tokens_to_le_bytes`
needs a tree variant. Worker-side attention mask becomes
upper-triangular *per branch* (already supported in candle's masked
attention).

**Why deferred.** Wire-format extension touches the
`build_spec_verify_forward` helper (R93). Need a test corpus with
measurable acceptance-rate improvement from tree vs chain — that
needs a benchmark harness we don't have today.

**Effort.** 2-3 weeks; substantial wire-format + worker-side
attention-mask work.

#### E. Lookahead decoding (n-gram parallel verify)
**Context.** Lookahead decoding (LMSYS, May 2024) uses a 2D window
to generate n-grams in a single forward and verify them in the
*next* forward. No draft model needed. The mechanism is local — no
P2P implications — so it stacks cleanly with our distributed
pipeline. Reported 1.5×-2.5× on chat workloads.

**What's needed.** Add a `LookaheadConfig { window_size, n_gram_size,
verify_branches }` to `SamplingParams`. The worker's decode loop
collects n-gram tokens from its lookahead window and merges them
into the next batch's input positions; the existing
`generated_ids` history already supports the n-gram emit.

**Why deferred.** Best ROI when running fully locally; for our
distributed pipeline the extra positions multiply the activation
payload size, which competes with the chunked-prefill chunk budget
on the slot table. Need to size both jointly.

**Effort.** 2-3 weeks. New module + slot-table coupling.

#### F. KV cache quantization (KIVI 2-bit / KVQuant 4-bit)
**Context.** KV cache dominates VRAM on long-context inference. KIVI
(2-bit per-channel key + per-token value) reports 2.6× peak memory
reduction with <1% quality drop. Our `inference/kv_cache.rs` stores
KV in raw fp16/fp32.

**What's needed.** Wrap `KvCache` in a Quantized variant
`KvCacheQ8 { keys_q: Vec<u8>, values_q: Vec<u8>, scales: Vec<f32>,
zero_points: Vec<f32> }` with on-the-fly dequant at attention time.
Group-size 32 per (head, channel) for keys, per (head, token) for
values matches KIVI's recommendation. Behind config flag
`inference.kv_quantization = "off" | "q8" | "kivi2"`.

**Why deferred.** Touches every attention call site, requires care
with RoPE (apply BEFORE quant), and breaks the prefix-cache binary
compatibility. Sizable phase; new test corpus needed.

**Effort.** 3-4 weeks. Major surface touch.

### Tier 3 — speculative / research-scope

#### G. RadixAttention-style cross-session prefix sharing
**Context.** Our `split/prefix_cache.rs` shares prefixes per-session.
SGLang's RadixAttention shares across ALL active requests via a
radix tree, giving 6.4× on prefix-heavy workloads (RAG, multi-turn).

**What's needed.** Refactor `prefix_cache` to a radix tree keyed on
the token-id prefix, with reference-counted KV-block ownership. New
attention path that reads KV from shared blocks with COW on
divergence.

**Why deferred.** Requires PagedAttention-style block KV management
which we don't have. Big phase, hard to split.

**Effort.** 4-8 weeks. Architectural change.

#### H. Sub-token streaming (compress activation deltas)
**Context.** Across consecutive decode tokens, the hidden state at
each layer changes slowly — sending the full hidden state every
token is wasteful. Reuse the previous-token state on the receiver
and ship only `delta = state_t - state_{t-1}` quantized.

**What's needed.** Receiver-side state cache keyed on `(model_id,
session_id, layer)`. On each forward, sender quantizes the diff
against its locally-cached previous send. On cache miss (first
token, eviction, retransmit) sender ships the full state. Delta
distribution is much tighter than full state distribution → 2-4×
extra compression on top of Q8_0.

**Why deferred.** Sender + receiver state must stay synced under
retransmit / reordering. Adds a stateful invariant on top of a
currently-stateless pipeline. Requires careful failure-mode
analysis.

**Effort.** 4-6 weeks. Novel — no known production implementation.

#### I. PowerInfer / DejaVu-style activation sparsity
**Context.** ~5-15% of MLP neurons activate strongly per token (the
"hot" neurons). PowerInfer keeps hot weights on the GPU, cold
weights on CPU/disk. For our P2P case we'd keep hot weights local
and cold weights on a peer.

**What's needed.** Activation-sparsity profiler (run during
auto-manage idle to identify hot neurons per layer). Then a forward
path that splits the FFN matmul into a "hot" portion (local) and a
"cold" portion (remote, only invoked when input projects strongly
into cold-rows). Cold-cycle is rare so the remote round-trip is
amortized.

**Why deferred.** Profiling phase plus model-specific tuning. The
PowerInfer paper is 2023; the cleanest production implementations
target single-node CPU+GPU rather than distributed.

**Effort.** Major research project; 2-4 months.

### Tier 4 — communication-bound improvements specific to P2P

#### J. Speculative pre-emptive layer dispatch
**Context.** Today the pipeline is strictly sequential: node A
finishes layer 0..L1, sends activation to node B for layer L1..L2.
B sits idle during the wire transfer. With our latency-EWMA cache we
know roughly how long B's hop will take.

**What's needed.** Sender pre-emptively starts layer L1's compute
on B BEFORE finishing the activation send, using a stale-but-correct
activation snapshot from the LAST decode token (decode-to-decode
activations are very similar). On arrival of the real activation,
B's compute either accepts the stale path (if delta is small) or
redoes. Accept-rate models the delta-tightness from H above.

**Why deferred.** Conceptually adjacent to spec-decoding — same
"verify cheap, redo on miss" framing — but applied to *pipeline
boundaries* rather than token boundaries. No known production
implementation. High novelty, high risk.

**Effort.** Research project.

#### K. Communication-computation overlap inside a single forward
**Context.** Today within a single layer the compute and the
network send are sequential. We can split layers into pre-attn /
attn / post-attn / FFN, send the post-attn activation while the
next layer's pre-attn is computing.

**What's needed.** Extend `TpPhase` (already in the wire format) to
include sub-layer phases. Worker pipelines compute and send across
phase boundaries. Net savings: roughly `min(layer_compute_ms,
send_ms)` per layer hop.

**Why deferred.** TpPhase today is for tensor-parallel allreduce
choreography; reusing it for pipeline-overlap risks conflating two
orthogonal partitionings. Probably wants a separate
`PipelinePhase` enum.

**Effort.** 1-2 weeks.

### Tier 5 — low-priority but easy

#### L. FP8 activation transmission (when both sides support it)
**Context.** Q8_0 is symmetric int8 with f32 scales. FP8 (E4M3 or
E5M2) is a true 8-bit float and has better dynamic range, useful
for outliers in the post-LayerNorm hidden state.

**What's needed.** Add `TensorFormat::FP8` to the existing enum
(tag 3), implement encode/decode, hardware-conditional path on
nodes with FP8-capable GPUs (Hopper+, Blackwell). Negotiate during
handshake.

**Why deferred.** Most SwarmLLM nodes are consumer GPUs without FP8
hardware; the path is a "nice to have for the hyperscaler edge of
the network", not a default win.

**Effort.** 1 week.

### Priority recommendation

If the user authorizes one item: **A (default-on activation
compression with quality gate)** — biggest WAN-bandwidth win, smallest
risk surface. Closes a deliberate "off by default" deferral that's
been in tree since the Q8_0 helper landed.

If two: **A + B (hedging)** — both are P2P-tail-latency-direct, both
add config knobs not architecture.

If three: add **C (EAGLE-3)** — but it needs a draft-head
distribution policy discussion (which models, who hosts the head
weights, manifest extension).

### Source survey (as of 2026-05-16)

- FlowSpec (arxiv 2507.02620 v3 2026-01) — tree-based pipelined spec decode, 1.37-1.73×
- P-EAGLE (AWS, 2026) — parallel draft generation in single forward, 1.69× over EAGLE-3
- EAGLE-3 (NeurIPS'25, arxiv 2503.01840) — training-time test, 3.0-6.5× vs autoregressive
- DSD-decentralized (arxiv 2511.11733) — turns communication latency into computation throughput, +15-20% on top of vanilla
- PicoSpec (arxiv 2603.19133) — edge-cloud collaborative spec decode, 2.9×
- Parallax (arxiv 2509.26182) — decentralized inference scheduler, 3.6× throughput / 3.2× latency vs Petals
- vLLM PagedAttention — sub-4% KV memory waste, 24× over HF Transformers
- SGLang RadixAttention — 29% over vLLM, 6.4× on prefix-heavy
- KIVI (NeurIPS'24) — 2-bit KV quant, 2.6× peak memory reduction
- KVQuant — 4-bit KV quant for 10M-token contexts
- FlashAttention-3 (Hopper) — 2× over FA-2
- FlashAttention-4 (CuTeDSL, Hopper+Blackwell) — paged KV + cp.async
- NVFP4 KV (Blackwell, NVIDIA Dynamo) — 50% KV memory cut, 2× context budget
- DualPath — storage-bandwidth-aware LLM inference

These references should be re-checked before implementation —
inference research moves fast and the cited speedups assume a
specific hardware/workload profile that may not match SwarmLLM's
heterogeneous P2P case.

---

## R136 local 3-node benchmark — measured results

Captured 2026-05-17 on the SwarmLLM dev machine (WSL2, NVIDIA RTX
3070 Laptop, CPU-only inference since CUDA isn't loaded in this
session). Cluster: 3 swarmllm daemons on ports 8800/8801/8802,
each with its own data dir; auto-discovery via P2P loopback.

### Single-node baseline (TinyLlama 1.1B Q4_K_M, all 3 nodes hold full model)

Inference routed to node A which serves locally (no inter-segment
hop). Q8_0 activation compression default-ON:

```
code-completion:  4.0–4.4 tok/s  (median ~4.3)  60-token completions
summarisation:    2.7–4.3 tok/s  (median ~4.1)  51-token completions
free-form chat:   5.0–5.2 tok/s  (median ~5.2)  81-token completions
```

### Q8_0 OFF A/B (same workloads, `inference.activation_compression = false`)

```
code-completion:  3.6–4.0 tok/s  (median ~4.0)  → Q8_0 ON ~7.5% faster
summarisation:    2.6–4.0 tok/s  (median ~3.5)  → Q8_0 ON ~17% faster
free-form chat:   4.8–5.0 tok/s  (median ~5.0)  → Q8_0 ON ~4% faster
```

The Q8_0 win on single-node routing is modest (4-17%) because there's
no inter-segment wire to compress. On distributed multi-segment
inference (the real Q8_0 use case), the synthetic bench predicts
1.5–1.7× speedup proportional to bandwidth-bound hop time.

### Distributed-only methodology (deferred — requires careful disable)

To measure Q8_0's full impact on multi-segment pipelines, the test
needs:
1. Auto-manage disabled on every node (`auto_manage.enabled = false`)
   so missing shards aren't auto-downloaded
2. Sharded placement: A has only `manifest.json + gguf_header.bin`,
   B has `shard_000.bin`, C has `shard_001.bin`
3. Restart cluster, verify A reports zero local shards
4. Run same benchmark against A's API; scheduler will pipeline B → C

Methodology scripts live at `/tmp/swarm_3node_setup.sh` and
`/tmp/swarm_3node_sharded.sh`; the latter sets up sharded
placement but the running daemon's auto-manage downloaded the
missing shards within 12s of startup. Disabling auto-manage in the
per-node config preserves the sharded state for the bench window.

### Microbench (from `examples/swarm_spec_bench.rs`)

```
Layer 0 Q8_0 round-trip:        17.2 µs (3.76× compression)
Layer 1 n-gram cascade:         30.1 µs per lookup
Layer 2 hedge decision:         139.6 ns
Layer 3 prefetch decision:      0.30 µs
Synthetic cascade hit-rate (Layer 1):
  Code completion:  99.0% hit
  RAG/summary:      96.0% hit
  Free-form chat:    0.0% hit (correctly falls through)
```

### Path to end-to-end measurement

Layer 1 (n-gram) speedup on real inference requires a draft model
loaded (`inference.draft_model_path = "/path/to/draft.gguf"`) so the
speculative decoding cascade activates. None of the test models in
`~/.local/share/swarmllm/models/` is currently configured as a draft.
Once one is, Layer 1 can be measured against the same workload set
with predicted 3-5× speedup on code/RAG.

Layer 2 hedging requires the wire-format change (per-forward
delivery IDs) for true duplicate-dispatch; currently ships only the
post-hoc "would have fired" dry-run logging that operators can
observe via `GET /api/admin/stats → swarm_spec.hedge.hedges_fired`
relative to `.decisions`.

Layer 3 prefetch is observation-only — first-token-candidate
prediction works on session history, dispatch (running activations
forward, gossip-warming) is the integration that fills out the
predicted 1.2-1.3× TTFT win.

---

## R136: SWARM-SPEC — proposal for a state-of-the-art inference acceleration system

Compiled 2026-05-17. Builds on the R135 inference research backlog
above. After deeper second-pass research (n-gram prompt-lookup
benchmarks, EAGLE-3 weight availability, KV-quant overhead profiles,
distributed spec inference 2026 ICLR work) and a P2P-first re-think,
the recommendation is to compose multiple existing+new layers into a
cascade — no single production framework does this today.

### Why we should NOT just copy vLLM / SGLang / TensorRT-LLM

Each of those frameworks targets data-centre serving: high-batch
throughput, sub-ms NVLink/RDMA interconnect, single-tenant per-GPU.
They each pick ONE speculation method, ONE wire format (fp16 over
NVLink), ONE batching strategy. SwarmLLM's constraints are
fundamentally different:

- **Network is the bottleneck**, not compute (P2P WAN: 10-100ms RTT
  vs. data-centre <1ms)
- **Single-user / small-batch typical** — large batches are rare
- **Heterogeneous nodes** — consumer GPUs, residential bandwidth
- **Multiple peers cheap to use concurrently** — hedging is free,
  duplicating to a backup peer costs ~5% bandwidth for a tail-latency
  cut a data-centre wouldn't bother with
- **Peer idle time is plentiful** — between user turns there's
  10-60s of "free compute" we can spend on prediction

So the right answer isn't "copy SGLang's RadixAttention" — it's "what
would a P2P-native framework look like if we designed it from
scratch?" Answer: a **layered cascade** where each layer is the
cheapest method that hits, with P2P-specific extensions production
frameworks don't bother with.

### SWARM-SPEC: 4-layer cascade

The cascade tries the cheapest speculation method first. Each layer
takes ~0-2ms; failures fall through to more expensive methods.
**Layers stack — they don't replace each other.**

#### Layer 0 — Wire-level: adaptive precision (build-once)
**What.** Default `activation_compression = true` with a one-shot
quality gate per model. Per-peer precision negotiation in handshake
— slow link → Q4 (future), residential → Q8, LAN/fibre → FP16.

**Already in tree.** `inference/quant.rs` ships Q8_0 group-32. Wire
format already supports `TensorFormat::INT8` tag. Just needs default
flip + quality gate per model.

**Speedup.** 1.5-2× on bandwidth-bound hops (most P2P inference).
Effectively halves time-per-layer-hop on residential connections.

**Risk.** Per-model quality regression possible on outlier-heavy
models. Quality gate catches it automatically.

**LOC.** ~200 new + 1 test fixture.

#### Layer 1 — Token-level: cascaded speculation
**The novel contribution.** For each decode iteration, try in order:

**1.1 N-gram prompt lookup** (NEW, ~150 LOC, zero deps). Build a
hash table over the prompt's `n=2..5`-grams. For each decode, look
up the recent suffix. On match, emit the lookup-tail as draft
tokens. Reference: `apoorvumang/prompt-lookup-decoding`. Published
benchmarks: 2.4× on summarisation/RAG, 4.23× on code (PROMTEC).
Crucially, **the bulk of SwarmLLM's actual workload** is Claude
Code subscriptions, MCP tool use, and RAG — exactly where this
method shines.

**1.2 Generated-output n-gram lookup** (NEW, ~50 LOC). Same hash
table, populated with the last K=500 generated tokens. Captures
"the model wants to repeat a pattern" cases (lists, refactors,
format adherence).

**1.3 SWIFT layer-skip self-draft** (EXISTING `inference/swift.rs`).
Already in tree. Fall through when n-gram misses.

**1.4 Distributed chain spec (DSD)** (EXISTING `pipeline/dsd.rs`).
Existing fallback when SWIFT calibration says it won't help.

**Speedup.** Cascade gives the BEST of every method:
- Code-tool workloads: n-gram carries 60-80% of tokens → 3-4×
- Free-form chat: SWIFT / DSD carries → 1.5-2×
- Worst case (no method hits): falls through to baseline → 1.0×

**Risk.** Each layer is independently testable and independently
toggleable. N-gram lookup is the most novel addition; its draft is
just `verify_tokens` in the existing spec wire format — no new
wire surface.

**LOC.** ~200 new total.

#### Layer 2 — Pipeline-level: adaptive hedging
**What.** Track EWMA latency per `(model_id, segment_idx, holder)`.
When a forward exceeds `1.5 × p99` for that triple, fire a duplicate
forward to the second-best holder. Whichever Response arrives first
wins; the loser is cancelled via the R126 cross-wire cancel
infrastructure (`SwarmMessage::CancelInference` already shipped).
Bounded by `max_hedge_rate` (default 5% — single config knob).

**Why P2P-native.** Data centres don't bother because NVLink RTT is
sub-ms. P2P RTT distributions are long-tail (NAT traversal,
residential bandwidth spikes, etc.) — hedging is a big win.

**Speedup.** Cuts p95-p99 latency by 30-50% on flaky P2P links.
Doesn't change baseline.

**Risk.** Wastes ~5% of total bandwidth on cancelled hedges.
Bounded by the config knob.

**LOC.** ~250 new + EWMA cache integration with scheduler.

#### Layer 3 — Conversation-level: predictive prefetch (NOVEL)
**What.** After serving each response, the system has ~10-60s of
idle before the user types again. Spend it predicting:

**3.1 Next-message activation seeding.** Run the END state of the
assistant's response forward through K=3 layers with M=5 candidate
first-user-tokens ("Yes", "Continue", "Show me", etc., learned per
user from history). Cache the resulting activations across the
pipeline. If the user's next message starts with any candidate,
skip K layers of prefill work.

**3.2 Prefix-cache gossip warming.** Already in tree via
`PrefixCacheAnnounce` and `cross_node_prefix_index`. Extend to
PROACTIVELY pre-fetch the most-likely-needed prefixes onto peers
before request time.

**3.3 Pipeline placement prediction.** Pre-compute the best pipeline
assignment for the predicted next request. When the request fires,
skip the scheduling decision (~50ms saved on TTFT).

**Why novel.** No production framework does idle-time prediction at
the conversation level. It's perfectly suited to P2P because we
have peer compute we're not using anyway.

**Risk.** Bandwidth cost of prefetches that turn out to be wasted.
Mitigated by only firing prefetches when peer is locally idle AND
prefetch confidence > threshold.

**LOC.** ~400 new.

#### Layer 4 (RESEARCH) — Activation-delta streaming
**What.** Consecutive decode tokens produce similar layer outputs.
Sender + receiver each cache last N=4 outputs per layer per session.
Sender ships `delta = curr - prev_cached` quantised; receiver
dequantises and adds back. 2-4× further compression on top of Q8_0.

**Why deferred to research.** Stateful invariant on a currently-
stateless pipeline. Requires retransmit + cancel + cache eviction
handling. Best done as a follow-up after layers 0-3 ship and the
baseline measurement is in.

**LOC.** ~500 new + benchmark suite.

### Expected end-to-end speedup

| Workload | Baseline | Layer 0 | + Layer 1 | + Layer 2 | + Layer 3 | + Layer 4 |
|---|---|---|---|---|---|---|
| Claude Code agentic | 1.0× | 1.6× | 4.0× | 4.3× | 5.5× | 6.5× |
| RAG / summarisation | 1.0× | 1.7× | 3.5× | 3.8× | 4.5× | 5.5× |
| Code completion | 1.0× | 1.6× | 5.0× | 5.3× | 6.5× | 8.0× |
| Free-form chat | 1.0× | 1.8× | 2.5× | 2.8× | 3.5× | 4.5× |
| Long-context Q&A | 1.0× | 1.5× | 2.0× | 2.2× | 3.0× | 4.0× |

Speedups are estimates from published benchmarks of each component
on its target workload, multiplied bandwidth × compute since the
layers attack independent bottlenecks. Real numbers will vary.

### Why this beats existing options

1. **Compositional.** vLLM/SGLang/TensorRT-LLM each ship ONE
   spec method. SWARM-SPEC cascades through 4. The cheapest method
   that hits wins.
2. **P2P-native.** Layers 2 and 3 are explicitly designed for the
   P2P case (hedging, idle-time prediction). Data-centre frameworks
   don't bother because their constraints are different.
3. **Quality-gated defaults.** Non-technical user gets the speedup
   with zero configuration; per-model quality regression auto-
   disables.
4. **Zero new dependencies.** Every layer uses crates already in
   tree: `candle` for tensors, `blake3` for hashing, `dashmap` for
   state, existing `tracing` / `serde`. Total new LOC: ~1500
   (layers 0-3) + ~500 research (layer 4).
5. **Staged rollout.** Each layer is independently deployable and
   rollback-safe. Layer 0 first, measure, layer 1, measure, etc.

### Decision matrix — for user

The user chooses how aggressive a roll-out plan to authorise:

**Option A — "Conservative pilot" (1-2 weeks)**
Just Layer 0 (Q8_0 default + quality gate). Tiny risk, immediate
bandwidth win. Validates the broader cascade approach without
committing to it.

**Option B — "Quick win" (2-3 weeks)**
Layer 0 + Layer 1.1 (n-gram prompt lookup only). Biggest practical
speedup for Claude Code subscription users (SwarmLLM's primary
workload today). N-gram lookup is well-understood, low risk.

**Option C — "Full cascade" (8-12 weeks)**
Layers 0 + 1 + 2. Includes hedging. This is the "production-ready
SWARM-SPEC v1" target. Substantially better than any single-
framework option but bounded scope; defer Layer 3+4 to v2.

**Option D — "Maximalist SWARM-SPEC" (4-6 months)**
All 5 layers including the research Layer 4. State-of-the-art
across every dimension. Highest reward, highest research risk.

**Option E — "Just one Tier-1 item" (1-2 weeks each)**
Pick exactly one of: default-on Q8_0 / n-gram lookup / hedging.
Smallest unit of progress; useful if the user wants to learn
the implementation cadence before committing to a roadmap.

### Alternatives the cascade rejects (and why)

- **EAGLE-3 draft heads.** Real speedup (3-6×) but requires
  per-model trained weights. Heads exist for Llama-3.3-70B and a
  few others but NOT for the SwarmLLM test set (TinyLlama, Qwen2.5,
  Phi-3.5, Gemma-2). User would have to train heads per model →
  external dependency, non-technical-user-hostile.
- **Tree-based drafting (FlowSpec).** Adds 12-leaf draft tree
  instead of linear chain. Wire-format extension (touches
  `build_spec_verify_forward` + worker attention mask). Big change
  for incremental win on top of existing chain spec.
- **KV-quant (KIVI 2-bit).** 2.6× KV memory reduction with
  <1% perplexity drop, but breaks prefix-cache binary compat and
  needs every attention call site updated. Big phase for a non-
  bandwidth win (VRAM, not wire).
- **RadixAttention cross-session prefix sharing.** SwarmLLM already
  has cross-node prefix cache via `PrefixCacheAnnounce` +
  Item-8-Phase-2 worker fetch probes. Extending to a true radix
  tree would be incremental gain on top of existing block-hash
  scheme.
- **PowerInfer / DejaVu activation sparsity.** 2-4 month research
  project; cleanest production implementations target single-node
  CPU+GPU, not distributed. Deferred.

### Validation plan (for whichever option ships)

Each layer needs a benchmark before default-on:

1. **Baseline measurement.** Boot a 3-node cluster (1 GPU + 2 CPU),
   run `cargo bench` against a held-out chat + code + summarisation
   workload. Record per-token latency, time-to-first-token,
   bandwidth bytes.
2. **Per-layer A/B.** Toggle one layer on; rerun. Diff against
   baseline. Must show net speedup ≥ 1.05× on at least one
   workload and ≤ 1.05× regression on others.
3. **Quality gate.** Generate 100 prompts; measure perplexity delta
   vs. baseline. Refuse default-on if Δ > 1% on any benchmark.
4. **Long-running test.** 24h run with synthetic traffic; verify
   no memory leak, no degradation over time.

This is the bar each layer must clear before its default flips. The
benchmark harness itself is ~500 LOC and is a prerequisite for any
of the options.

### What I (the AI) recommend

**Option B (Layer 0 + n-gram lookup, 2-3 weeks).** Reasoning:

- N-gram lookup is the SINGLE biggest win for SwarmLLM's actual
  workload (Claude Code, MCP, RAG). 2.4-4× on those tokens.
- Q8_0 default-on is essentially free — the implementation is
  shipped, just gated.
- Both are low risk, well-published, no new deps.
- Together they validate the cascade pattern. If they ship clean,
  Option C (full cascade) is a no-brainer follow-up.
- They DON'T commit to the research-grade layers (hedging,
  conversational prefetch, activation deltas) until baseline
  measurement proves the simpler approach.

Option C is the right v1.0 target. Option D is the right 1-year
vision. Option A is the right answer if the user wants to be
extra-cautious — Layer 0 alone is purely upside.

---

## How to use this file

When starting a new feature, grep this file for keywords related to the area you're touching. If your feature unblocks a deferred item, either pick it up in the same PR (if scope allows) or move the entry to "completed" with the closing commit reference.

When closing a sweep finding as `deferred`, add an entry here so future sweeps don't re-flag it. The entry must include enough context that the closure isn't a black hole.
