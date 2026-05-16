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

_What remains deferred:_
_- Cross-pool inference routing. Today inference stays within the
  caller's pool/private-mode scope; the discovery signal is currently
  read-only telemetry. Wiring the scheduler to actually route requests
  to a foreign pool changes the privacy contract — needs explicit user
  consent UI, billing model decisions, and trust escalation policy._
_- Frontend tile. The REST surface is ready but no dashboard component
  renders it yet; pairs naturally with the next dashboard refresh._

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

_What remains deferred (auto-action + UI surfacing):_
_- Auto-action layer (download the recommended quant when the user
  toggles "auto-switch quants" — needs a UI surface for explicit
  consent + a migration path for replacing a hosted quant with a
  better variant)._
_- Frontend integration (the recommendations are exposed via REST
  + WS-cached state but no dashboard tile renders them yet; will
  pair naturally with the next dashboard refresh)._

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

## How to use this file

When starting a new feature, grep this file for keywords related to the area you're touching. If your feature unblocks a deferred item, either pick it up in the same PR (if scope allows) or move the entry to "completed" with the closing commit reference.

When closing a sweep finding as `deferred`, add an entry here so future sweeps don't re-flag it. The entry must include enough context that the closure isn't a black hole.
