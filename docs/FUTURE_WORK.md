# Future Work — Out of R110-R115 Scope

Captures items deliberately deferred from the model-management redesign and from prior sweeps. Each entry has enough context that a future implementer (or a future me) can pick it up without re-deriving the rationale.

## Model management — deliberately out of R110-R115

### Inter-pool model sharing policy
**Context.** Pools today are private membership scopes. A model's shards stay within a pool; cross-pool peers don't see each other's hosted shards. The wishlist + capacity computation in R110-R115 also stays scope-local.

**What's needed.** A protocol-level "model interest" channel that pools can opt into. Joining the channel announces "we host model X" without exposing pool composition. Outsiders can see which pools serve a model and route inference accordingly. Likely needs a per-model k-anonymity floor (refuse to advertise if pool has < N hosts) so it can't be used to enumerate small pools.

**Why deferred.** Designing the privacy/trust boundary is a real architectural decision. Doing it inside R111's wishlist refactor would mix concerns.

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

---

## Audit-deferred items (from sweep rounds R100-R109)

### MoE routing per-arch config
**Context.** R109 documented that `topk_cpu` is mathematically equivalent to Mixtral / Qwen3-with-norm. DeepSeek-V2 strict (no renorm) and V3 (sigmoid) need different paths.

**What's needed.** `ModelArchitecture` extension: `routing_mode: RoutingMode { Softmax, SoftmaxNoNorm, Sigmoid }`, plumbed from GGUF metadata (`{arch}.expert_norm`, `{arch}.expert_gating_func`). Conditional path in `MoeFfn::forward`.

**Why deferred.** No live model in this codebase exercises the divergent paths today. Would be wasted work without a test corpus.

**Sweep log:** `src/inference/layers/mod.rs:397` (R108 finding).

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
trailing). 3 unit tests verify the state machine. **Wire-format diffs
remain deferred** — debounce captures the bursty-churn win without a
protocol change; true `PoolStateDiff` variants need a generation
counter + checksum verification + Full/Diff variant on `PoolMessage`,
worth doing only if a WAN bench shows the trailing-full-state broadcast
is actually bandwidth-constrained._

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
