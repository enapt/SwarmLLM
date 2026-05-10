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

## How to use this file

When starting a new feature, grep this file for keywords related to the area you're touching. If your feature unblocks a deferred item, either pick it up in the same PR (if scope allows) or move the entry to "completed" with the closing commit reference.

When closing a sweep finding as `deferred`, add an entry here so future sweeps don't re-flag it. The entry must include enough context that the closure isn't a black hole.
