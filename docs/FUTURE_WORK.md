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

### Distributed-pipeline matched_stop_sequence plumbing
**Context.** R109 plumbed `matched_stop_sequence` through the local-worker path. Distributed pipeline (`pipeline/distributed.rs:516`) leaves it as `None` because `LayerResult.finish_reason: NetworkFinishReason::Stop` doesn't carry a string.

**What's needed.** Extend `NetworkFinishReason::Stop` to `Stop { matched_sequence: Option<String> }` (with `#[serde(default)]` for wire-compat). Propagate in remote-worker decode loop's stop-detection sites. Update `decode_layer_result` to pass through.

**Why deferred.** Multi-segment distributed inference with custom stop sequences is rare in practice. Local-worker path covers the common Anthropic case.

**Sweep log:** `src/inference/pipeline/distributed.rs:528` (R109 closure).

---

### Cross-node logprobs in distributed path
**Context.** R106 plumbed logprobs through the local worker path. Distributed pipeline's `collected_logprobs` field still resolves to empty.

**What's needed.** Worker emits per-token logprob in `LayerResult` → coordinator's `collect_streaming_token` accumulates → final `InferenceOutput.token_logprobs` populated.

**Why deferred.** Same rationale: distributed-path logprobs are a niche feature for billing telemetry; local path covers most users.

**Sweep log:** `src/inference/pipeline/distributed.rs:533` (R106 commentary).

---

### Cancel-token plumbing across the wire
**Context.** `InferenceRequest.cancel: Option<Arc<AtomicBool>>` is `#[serde(skip)]` — local-only (gotcha #66). Cross-node cancellation does not propagate; a flipped cancel only stops the originating node's decode loop. Remote segments keep computing until their next-forward times out.

**What's needed.** Carry cancellation as a `NetworkCommand::CancelInference { request_id }` rather than via the `InferenceRequest` struct. Network manager broadcasts to the peer set involved in the pipeline.

**Why deferred.** Real-cancel-on-wire is observable as a UX win (faster client-disconnect propagation), but the network-level timeout already bounds wasted compute. Lower priority than user-facing model management work.

**Sweep log:** `src/types.rs` `InferenceRequest.cancel` (gotcha #66).

---

### TrustManager bulk hydrate optimisation
**Context.** R109 confirmed `TrustManager::hydrate_from_db` is a test-only helper. Production trust restore is per-peer at connect time via `get_trust()` in identify handler.

**What's needed (only if measured).** If we ever observe slow first-connect latency due to repeated DB reads, switch identify handler to use a single bulk hydrate at startup that warms a `DashMap` cache.

**Why deferred.** Not a performance issue today; existing per-peer reads are sub-millisecond on redb.

---

### Manifest tensor-cap UX
**Context.** R104 added a max-tensors cap in `manifest.rs` to defend against malicious manifests. The cap silently rejects oversized manifests; no operator-visible alert when one is dropped.

**What's needed.** Surface rejection as an `ActivityEvent` (`category: "models", kind: "manifest_rejected"`) with peer + size context. Helps operators diagnose "why is this model not appearing?"

**Why deferred.** Edge case; rejection would only fire on adversarial input, and the security log catches it.

---

### Pool-state gossip coalescing under high churn
**Context.** Every join/leave triggers a full `PoolState` re-gossip. A pool with 50 members rotating membership produces pool_state floods.

**What's needed.** Diff-based gossip: send only changed members + a checksum; receiver validates checksum and falls back to full state on mismatch. Or epoch-based — gossip the full state every K epochs but interspersed with diffs.

**Why deferred.** Existing 5-min coalescing window in `gossip_pool_state` keeps the load tractable for typical pools.

---

### Contribution-mode toggle in Setup wizard (R121 follow-up)
**Context.** R121 added an Auto/Manual contribution toggle to the Settings panel. The Setup wizard (`frontend/js/components/setup.js`, `index.html:56-113`) has the same contribution segmented control + an auto-manage checkbox, but doesn't yet expose the new toggle.

**What's needed.** Mirror the Settings-panel toggle in the wizard step 1, default to Auto, and include `contribution_auto` in the `setup.js::submit()` payload. The wizard's save path already uses `PUT /api/admin/config`, so the wire change is one field.

**Why deferred.** The toggle defaults to Auto on the backend, so new nodes get the recommended behaviour without wizard exposure. It's a UX nicety, not load-bearing.

---

### Ease-of-use audit follow-ups — bigger UX changes (R125 follow-up)

**Context.** The 2026-05-13 ease-of-use audit (R125) applied ~40 copy fixes
to make the UI usable by non-technical users. Three structural / behavioural
items came up that warrant direction beyond a copy refresh:

1. **README architecture section is intimidating.** The "12 async Tokio
   tasks wired via mpsc channels, sharing `Arc<SharedState>` + DashMap"
   sentence + the subsystem-name diagram + the node-tier table all appear
   mid-document and have no value for the first-time-user persona the
   README leads with. Proposed: wrap behind `<details>` titled
   "Implementation details (for contributors)".

2. **Header is overloaded on first load.** 12+ icons (hamburger, logo, 7
   tabs, model dropdown, "+ Find model", share, auto-manage, private-mode
   lock, settings gear, setup chip, language picker, theme toggle, node ID
   + tier badge + credits, shutdown) appear on the first visit without any
   explanation. Proposed: a first-run guided tour (single overlay walking
   through the 4 most important elements) or `?` tooltips on each.

3. **`activity.worker_*` events duplicate `activity.model_*` events.** Both
   fire on every model load/unload. The "worker" copy uses internal
   process-management language. Proposed: stop emitting `worker_spawned` /
   `worker_unloaded` to the user activity feed; keep them as internal
   trace events only, or merge into the `model_loaded` / `model_unloaded`
   path. Requires a backend change (the `with_toast` flag on those events
   in `daemon/state/activity.rs`).

4. **`models.hf_score_breakdown` exposes a 4-component score
   decomposition** (quality / fit / demand / size) on every HF browser
   result row. Useful for tuning auto-manage but meaningless to lay users.
   Proposed: replace with a single human verdict ("Good match for your
   computer", "Works but uses lots of space", "Needs more memory than
   available"); keep the raw decomposition as a developer tooltip.

5. **GGUF metadata panel surfaces raw hyperparameters.** `models.meta_rope_dim`,
   `models.meta_kv_heads`, `models.meta_rms_epsilon`, etc. are gated
   behind the `ⓘ` button but the section label "GGUF Metadata" is itself
   jargon. Proposed: rename to "Technical Details"; group human-readable
   items (parameters, context length) above a collapsible "Advanced"
   sub-section that hides RoPE / KV heads / tensor offsets.

6. **`models.encrypted_pipeline` and `enc.unprotected_detail`** reference
   "first AND last shard" — opaque to a lay user. Proposed: label as
   "Private mode (end-to-end encryption)"; status as "Available — your
   computer holds the key parts needed".

7. **`dashboard.api_log_link` text stays English in 20 non-English
   locales** ("View API request log →"). The arrow may render poorly in
   RTL locales (ar). Proposed: decide whether to translate (per-locale
   recordings exist for similar nav strings) or formally adopt a
   carve-out and document it.

8. **46 country names hardcoded English in `network-map.js:430`** —
   `countryNames` map. Translating × 21 locales = ~966 entries. Common
   carve-out for world-map UIs, but inconsistent with the otherwise
   fully-translated UI. Same decision as #7.

**Why deferred.** Each of these is bigger than a copy fix and needs a UX
decision (and #3/#4/#5 need backend or schema changes). The R125 copy
pass got the bulk of the value; these structural items can land
piecewise once direction is set.

---

### True global holder count for prune redundancy_ratio (R121 follow-up)
**Context.** `ModelRegistry::shard_holders` caps stored holders per shard at `MAX_HOLDERS_PER_SHARD = 50`. The R121 scale-back prune sees up to 50 holders even when the swarm has 1000+. Prune still fires correctly — the `holder_count > effective_target` gate triggers as long as 50 > target, which is always true for realistic targets — but the displayed `redundancy_ratio = holder_count / effective_target` underestimates by the cap ratio, so the prune score is artificially low for severely over-replicated shards.

**What's needed.** A separate uncapped `DashMap<ShardId, u32>` populated from DHT `get_providers` query results (today the results merge back into the same 50-capped cache, so the data is lost). Prune reads from this uncapped map for `redundancy_ratio` computation while keeping the bounded `shard_holders` map for routing decisions.

**Why deferred.** The +1.0 severe-saturation score bonus already kicks in at `holder_count >= 2 × target`, which the 50-cap still detects for any target ≤ 25 (i.e., all realistic swarm sizes). At 50+ target replicas the score saturates at 50/target but that's still high enough to prune. Refactoring the holder-count storage is a bigger change with no current behavioural delta.

---

## How to use this file

When starting a new feature, grep this file for keywords related to the area you're touching. If your feature unblocks a deferred item, either pick it up in the same PR (if scope allows) or move the entry to "completed" with the closing commit reference.

When closing a sweep finding as `deferred`, add an entry here so future sweeps don't re-flag it. The entry must include enough context that the closure isn't a black hole.
