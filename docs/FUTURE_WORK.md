# Future Work — Out of R110-R115 Scope

Captures items deliberately deferred from the model-management redesign and from prior sweeps. Each entry has enough context that a future implementer (or a future me) can pick it up without re-deriving the rationale.

## NETWORKING_PLAN — fully implemented (2026-07-24) ✅

The networking plan's Phases 1–3 + the version handshake shipped (see
`docs/NETWORKING_PLAN.md` §"Implementation status"). The two Phase-3 sub-items
originally parked here are now **also done**:

- **Explicit DHT relay-provider record** — DONE. `network/discovery.rs::
  {relay_service_key, start_providing_relay_service, query_relay_providers}`; a
  relay-forwarding node registers under `/swarm/relay-service/v1`, and a node
  short on relay connections (`MIN_RELAY_CONNECTIONS`) queries it each discovery
  tick and dials the discovered relays (`handle_relay_providers_found`). Lets a
  node stranded on a dying anchor find fresh relays.
- **Mixed-version integration test** — DONE, as a deterministic wire+crypto test
  (`tests/integration/test_relay_mixed_version.rs`): the full A → relay → B
  sealed round-trip (relay stays blind, target recovers the exact inner message,
  tampered header rejected) plus the vN↔vN-1 feature gate. A heavier full-daemon
  two-node libp2p test would add little over this and risks flakiness — not
  pursued. Unit coverage also in
  `crates/swarmllm-types/src/node.rs::version_compat_tests`.

Nothing from the original networking plan remains deferred.

### Tensor-relay large-forward chunking (post-plan follow-on, deferred)

The v0.3.18/v0.3.19 **tensor relay** (`SwarmRequest::RelayedTensor`) that routes
distributed-pipeline activations between two un-connectable NAT'd nodes seals
each forward/result as a single sealed blob, bounded by
`crypto::relay_seal::MAX_RELAY_TENSOR_BYTES = 32 MB`. A forward whose encoded
(and, when `activation_compression` is on, Q8_0/zstd-compressed) size exceeds
that cap is refused at seal time — `try_relay_tensor` returns false and the
forward is dropped rather than sent oversized.

In practice this is rarely hit: activation compression keeps even a long-prompt
prefill for a 7B-class model well under 32 MB. It only bites an **uncompressed**
forward of a very long prompt over the relay path. The fix is to reuse the
existing STREAM-chunk machinery (`network/pipeline_stream::chunk_layer_forward`
+ `SharedState.pending_activation_chunks` receiver assembly, already used for the
direct Tier-4K path) on the relayed path — split a large forward into
≤32 MB chunks, each ephemeral-sealed, reassembled at the target. Deferred because
no measured workload has hit the cap yet; revisit if a large-model + long-prompt
distributed run over pure app-relay reports a dropped forward.

### Connection churn on multi-interface hosts — deterministic dialer partial (2026-07-25)

**Shipped (v0.3.20, `network/manager/events.rs`):** the mDNS `Discovered`
handler now groups a peer's addresses into ONE dial and only the smaller-PeerId
node dials, eliminating the *bidirectional* simultaneous-dial race so exactly one
connection forms per peer. This is the LAN/mDNS fix for the multi-connection
churn that (on hosts advertising several interfaces — WSL2's `10.255.255.254`
NAT-gateway + `169.254` link-local + Docker bridge + LAN) let libp2p route a
tensor forward to a stale/half-open connection and silently drop it (upstream
"keeps all connections, uses an arbitrary one": go-libp2p #634 / rust-libp2p
#912).


**Observed 2026-07-27 — two LAN peers mutually forgot each other and did not
retry.** After a deliberate load test (12 concurrent requests + several
multi-minute prefills) produced repeated `connection closed reason="io_err…"`
churn, the WSL2 node and the Proxmox LXC node ended up in a symmetric state:
each listed the anchor and a remote tester as peers, neither listed the other,
and this persisted for 17+ minutes. Both daemons were healthy the whole time
(`NRestarts=0`, both serving requests), and both remained connected to the same
anchor — so DHT/PEX had a path to rediscover the pair and did not take it. The
local node made **zero** dial attempts to `192.168.1.60` in that window, and
mDNS logged one event at startup and none afterwards.

User-visible effect: requests failed with `Pipeline assembly failed: No node
available for layer 10` — the local node held layers 0-10 and the only holder of
10-16 had become invisible.

**Root-caused and partially fixed (2026-07-27).** `handle_connection_closed` had
exactly two re-dial triggers — "active pipeline needs this peer" and "peer was
never registered (died before Identify)". A peer that was registered AND idle at
the moment it dropped matched neither, so it was removed from the registry with
no re-dial scheduled, and re-discovery depends on the peer ANNOUNCING itself,
which only happens when it restarts. Measured: killing the peer daemon and
restarting it reconnected in 13s (it re-announced), while a peer that stayed up
after a connection drop was never re-dialled at all. A jittered single re-dial is
now scheduled on that path too.

**Still open**: that is ONE attempt. If the peer is unreachable for longer than
the jitter delay the dial fails and nothing re-enqueues it, since a failed dial
raises `OutgoingConnectionError` rather than `ConnectionClosed`. A bounded
backoff schedule for a peer we have previously identified would close the
remaining gap; it was left out deliberately to avoid re-dial storms against peers
that have genuinely left.

Original evidence:

**Restarting one side reconnected the pair in 6 seconds.** That is the decisive
datapoint: it rules out unreachable addresses, a poisoned address cache, mDNS
being blocked by the LXC bridge, and the peer actually being gone — a fresh
process dialled it immediately using the same discovery paths. The fault is
therefore **in-process reconnect state on a node that has been running through
churn**, not in what the node knows about the peer.

That narrows the search considerably. Likely candidates, in order: a peer dropped
under repeated `io_err` closes landing in a backoff / negative-dial entry that
nothing ever expires; `pending_redial` dedup (`try_enqueue_redial`) retaining an
entry that was never drained, so subsequent enqueues are suppressed as duplicates;
or the reconnect path being reachable only from `handle_connection_closed`'s
`in_active_pipeline` branch, which would not fire once the pipeline had already
failed. Note `peer_registry` is deliberately preserved across disconnects for
exactly this reconnect case, so the peer was almost certainly still *known* while
not being *dialled*.

Reproduction is load-dependent. Capture `/api/admin/peers` from BOTH sides plus
dial attempts before restarting, because a restart clears it — and the fact that
it clears is itself the main clue.

**What it does NOT cover, and what was ruled out:**
- **Only the mDNS (LAN) dial path is gated.** Internet peers discovered via
  bootstrap/DHT/PEX go through a different dial path that this rule does not
  touch. If the same multi-connection churn is ever confirmed for internet
  peers, extend the deterministic-dialer discipline (and grouped dialing) to
  those paths too — that is the remaining connection-management work.
- **Same-host 4-interface worst case is not 100% eliminated.** On a single
  loopback host with 4 mutually-reachable interfaces + zero holder redundancy,
  a residual stale connection can still form (reconnect/PEX paths aren't gated).
  Real deployments (distinct hosts + `min_replicas ≥ 2`) tolerate this via
  failover; the pathological same-host repro does not.
- **NOT our bug — the cross-NAT failure was tester-side.** A native-Windows-node
  test (WSL→Windows interop) proved cross-network inference works end-to-end: a
  clean native node served a model from our node over the real internet. The
  intermittent silent-drops seen against one external tester node were **that
  node's serving side** (it never acknowledged inference requests from *any*
  requester, while ours served fine), not a routing/connection bug in shipped
  code. Ping (liveness) and the persistent-stream path were both tried and are
  the wrong tool — the connection is alive, it's connection *selection*. Full
  investigation: `memory/round_log_distributed_conn_bug.md`.

**Trigger to build the internet-peer extension:** a reproduction on distinct
hosts (not same-host loopback, not a tester-side serving failure) showing the
multi-connection stale-route drop. Until then the LAN fix + failover cover the
observed cases.

## Demand-driven resource management — VRAM done, disk contraction deferred (2026-07-24)

External report (`Rapport_VRAM_Idle`): a contributor node holds models in VRAM
indefinitely (pressure-only eviction, no demand signal). **Shipped:** demand-driven
**VRAM unload** — `auto_manage/prune.rs::try_idle_vram_unload` frees a loaded
model's GPU memory after `idle_unload_secs` (default 1800s) with no local requests
AND low region demand (`region_demand` EMA < `IDLE_DEMAND_EMA_THRESHOLD`). Shards
stay on disk, holder status is unchanged, so it reloads (cold start) on the next
request — **zero availability impact**, VRAM follows real demand. Controls surfaced
in `config/default.toml` + `docs/book/.../troubleshooting.md`.

**Demand-driven DISK replica contraction — SUBSTANTIALLY ALREADY REALIZED at the
default config (re-assessed 2026-07-25).** On closer reading, the existing
redundancy-prune already does this down to the floor the design specified. The
saturation-bypass in `prune.rs::effective_prune_target` — for a
`contribution_auto` node, a shard held by ≥`SATURATION_FACTOR_AUTO`×target — uses
the *raw* target instead of the pressure-relaxed nudge, so **a node with zero
local pressure already sheds an idle, over-replicated shard**. For an idle model
`geo_target_replicas` collapses `target` to `min_replicas` (demand_factor=1.0),
and `effective_prune_target` floors at `min_replicas`. At the **default
`min_replicas = 2`, that floor IS the `IDLE_REPLICA_FLOOR (≥2, never a single
point)`** the deferred design called for. All the guards the design wanted are
already applied per-shard in `evaluate_and_prune`: active-pipeline, region-last-
holder (`would_eliminate_region`), holder-load (`remaining_holders_busy`),
reacquire (`can_reacquire`), recent-request protection, cooldown, and
pinned/locked/reference/encrypted exemptions. Regression-guarded by
`prune.rs::idle_over_replicated_contracts_to_floor_never_below`.

**Only unbuilt piece — contracting BELOW a *higher operator-set* `min_replicas`
(intentionally left alone).** The design's "contract below `min_replicas`" only
differs from today's behaviour when an operator has raised `min_replicas` above
2. Letting idle contraction dip under that would override the operator's explicit
redundancy floor — a deliberate configuration choice — so it is *not* done. If
ever wanted, add an opt-in `auto_manage.idle_replica_floor` (default =
`min_replicas`, i.e. off) with a hard ≥2 clamp, gated on
`global_holder_count` comfortably exceeding it. Rationale unchanged: (a) it frees
disk, not VRAM (the VRAM unload above already covers the reported concern); (b)
it trades swarm redundancy for a model that may be wanted again — acceptable only
once adoption makes each shard richly held, which the `min_replicas=2` floor
already respects by never going to a single point.

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
**R141**: default flipped from `false` to `true`. A recommendation
surface gated by a manual button-click isn't a recommendation; trust +
prune cooldown already cap the bandwidth cost. Operators on metered
links can flip back off via `[auto_manage] auto_switch_quants = false`.
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

_(R141 — Auto-manage cold-start UX: shipped 2026-05-21.
Closed the long-standing "fresh node has nothing to chat with" gap via
six coordinated changes. **Trusted-publisher allowlist**
(`TRUSTED_HF_PUBLISHERS` in `huggingface/watcher.rs` — official authors
+ curator community) drops the HF-trending promotion threshold from
100k to 10k downloads for vetted accounts; unknown publishers retain
the 100k floor and 24h age gate. **Wishlist `Candidate` status**
(`compute_wishlist` merges HF trending entries the swarm hasn't
adopted, cap 24, with `hf_repo_id` + `task_tags` fields). **Chat empty
state swarm catalog** (`createEmptyState` in `utils.js` builds 3 rows
— Serveable / Aspirational / Candidate — when no model selected; chip
click selects model + opens fresh chat OR routes to HF browse for
Candidate). **`auto_switch_quants` default → true** (R134.6 flip).
**`P2P_PERMIT_STALL_SECS` 600 → 180** (silent libp2p drops fail over
to HF fallback in 3 min, not 10). **`hf_sources` cap activity event**
(`activity.hf_sources_cap_reached`, throttled 1st + every 50th, fires
warning toast pointing the user at Settings cleanup). 15 new i18n keys
× 21 locales (1156 → 1171 entries). 1030 → 1053 lib tests.)_

---

## Audit deferral — R128 sweep-log triage

The 2026-05-14 audit pass reviewed all 88 sweep-log `deferred` entries
and confirmed the remaining items genuinely require user discussion or
have explicit "won't fix" semantics. Concretely:

- **R103 `x-swarm-forwarded` dual-gate dead code** (`src/api/middleware.rs:432`)
  — **closed R138** (2026-05-18). The dead branch was already removed
  in an intervening round; the inline comment at lines 461-466 documents
  why falling through to Bearer auth is correct. R138 moved the
  sweep-log entry from `deferred` to `fixed`.

- **R103 `batch_scheduler_loop` no JoinHandle / catch_unwind**
  (`src/inference/process_pool.rs:792`) — kept deferred. A panic in
  the scheduler degrades to direct-execution fallback (forward() only
  uses the channel when batching is on and the channel is alive), so
  this is performance-degradation not correctness. Adding catch_unwind
  across an async boundary requires structural changes (UnwindSafe
  bounds, poison handling) that aren't justified by the failure mode.
  Won't fix unless a concrete panic site appears.

- **R105 libp2p relay `..Default::default()` "possibly open"**
  (`src/network/relay.rs:51`) — **closed R138** (`wontfix`). Verified
  against libp2p-relay 0.20.x: the only fields covered by the default
  fallback are `reservation_rate_limiters` and `circuit_src_rate_limiters`,
  both of which default to **conservative** per-peer/per-IP rate limits
  (30/peer/2min, 60/ip/min). No "open" surface.

- **R105 latency sample ring time-coverage**
  (`src/api/metrics.rs:160`) — **closed R137** (2026-05-17). Sample
  storage migrated from `VecDeque<f64>` to
  `VecDeque<(Instant, f64)>` with `LATENCY_SAMPLE_MAX_AGE = 600s`
  (matches Prometheus `rate(...[10m])` typical). Drop-by-age applies
  both on insert (in `router/mod.rs` completion path) and on every
  read (`compute_latency_stats` + `write_latency_histogram`) — the
  per-call drop is needed because at low rates the writer's own
  drop pass can be hours apart. Memory overhead: +16 bytes/entry
  (a u128 Instant), capped at 1000 entries = +16KB worst case.
  Regression test `latency_age_filter_drops_old_entries` verifies
  the filter logic without needing a full SharedState build.

- All remaining deferred entries (Qwen 3.5 DeltaNet shape verification,
  GossipSub PeerScore tuning, DHT provider capability challenge, etc.)
  are explicitly architectural / design discussions matching the
  user's "defer items needing discussion" directive.

## Audit deferral — R138 sweep-log triage

The 2026-05-18 autonomous defer-batch round closed ~20 sweep-log
deferrals via a mix of real fixes and verification-only entries.
The headline closures:

- **R103 `active_count.fetch_add` before RAII guard armed**
  (`src/inference/router/mod.rs:689`) — `fetch_add` moved INSIDE the
  spawned task so a `tokio::spawn` OOM panic can no longer leak the
  tier-cap counter.

- **R103 umask race during concurrent worker spawn**
  (`src/inference/process_pool.rs:1048`) — verified already serialised
  by the `spawn_lock` (tokio Mutex held across the entire
  `spawn_worker` call); no second caller. R138 marked `fixed`.

- **R103 cuda-keyring .deb no SHA verify** + **R103 Docker base images
  not pinned by digest** — both supply-chain integrity items: cuda-keyring
  pinned to SHA256 d93190d5... in both ci.yml + release.yml; all 4 FROM
  directives in Dockerfile/Dockerfile.cuda pinned with @sha256: digest
  alongside their tag.

- **R104 scan ignores `auto_manage_paused` for re-announce**
  (`src/model/auto_manage/scan.rs`) — `model/auto_manage/manager.rs`
  passes `Option<&network_tx>` based on the `auto_manage_enabled`
  atomic. Rescan still runs locally (correctness — picking up manually-
  placed shards) but the network re-announce is gated on the pause
  toggle. Manual `POST /api/admin/rescan-shards` always announces.

- **R104 config hot-reload first-tick** — verified already fixed by
  `interval.tick().await` at `model/auto_manage/manager.rs:377`.

- **R105 `CreditBalance` schema-upgrade safety** — `#[serde(default)]`
  on numeric + timestamp fields; type-level doc encodes the rule for
  future field additions. Drive-by: `[dev-dependencies] serde_json`
  added to `swarmllm-types` Cargo.toml so 15 previously-dead lib tests
  now run.

- **R105 `private_mode`/`offline_mode` mixed-type in `pool_state`
  tree** — moved to a new `node_modes` redb tree via the
  `restore_node_mode()` migration helper. Each tree now single-typed;
  no namespace collision risk for `iter_json::<PoolState>`.

- **R105 `check_integrity` validates JSON only not types** —
  `validate_strict` routes each `CRITICAL_TREES` entry through the
  actual `swarmllm_types` type. Type mismatches that previously
  passed JSON-Value validation are now reported as corrupt. Dropped
  the unused "identity" tree from `CRITICAL_TREES`.

- **R105 HF .tmp resume vulnerable to layout change** — added a
  BLAKE3 layout-hash sidecar `<shard>.tmp.layout` written BEFORE
  any data lands in .tmp. Resume path verifies hash match; mismatch
  → discard both files and restart. Closes the coincidental-
  size-match-across-layout-revisions vector.

- **R97 `credit_percentile_cache` lock held across `DashMap` iter** —
  three-phase pattern (peek under lock → iter outside → re-lock to
  write) so the router task no longer blocks on long iters.

- **R101/R102 `/metrics` no auth + credit-balance disclosure** —
  new `api.metrics_auth_required: bool` config flag tightens the
  loopback exemption for public-internet nodes. Default false
  preserves Prometheus convention.

- **R102 credit forward per-window TOTAL value cap** —
  `CREDIT_FORWARD_MAX_VALUE_PER_WINDOW = 200_000` credits/min/member
  on top of the existing count cap (60 forwards). Either limit
  alone is sufficient to reject.

Plus ~15 verification-only closures across R66/R67/R68/R89/R97/R102/
R103/R104/R105/R123 for items intervening rounds had already
addressed (anti_gaming TTL bounds; SWIFT `emit_token` cap;
`peer_cache` replace_tree atomicity; `BACKGROUND_CANCEL_AGES` TTL
sweep; MCP `sampling/createMessage` explicit arm; Anthropic→OpenAI
tool block translation; `dashboard.api_log_link` translated; etc.).

Test count: 1005 → 1015 lib tests + 15 newly-runnable
`swarmllm-types` tests. Clippy clean default + features
dev,claude-subscription + features llama. Detail: commits
d122e9e8..e6dad63f in `.claude/sweep-log.jsonl` R138 entries.

---

## R142 deferred items (autonomous 8-hour sweep, 2026-05-22→05-23)

The R142 sweep loop closed 60+ findings across 14 rounds. The
following items were flagged with confidence but explicitly NOT
fixed because they need either real-model integration testing, a
maintainer architectural decision, or a feature addition beyond
sweep scope. Full per-item context is in `.claude/sweep-log.jsonl`
under `"status":"deferred"`.

### VLM weight loading: `ffn_up`/`ffn_down` inverted — **CLOSED R148** (2026-07-22)

_Resolved by reading the shapes out of the mmproj GGUF rather than by running
the model, which is why it stayed open: the deferral asked for a llama.cpp
cosine-similarity comparison, but the tensor metadata answers it outright._

_**The existing loader was correct.** In `llava-v1.5-7b-mmproj-f16.gguf`
(CLIP ViT-L/14-336, hidden 1024, n_ff 4096), `v.blk.0.ffn_down.weight` has
GGUF `ne = [1024, 4096]` with `bias = [4096]`, and `ffn_up` has
`ne = [4096, 1024]` with `bias = [1024]`. Since a bias length must equal the
output width, `ffn_down` is unambiguously the 1024→4096 **expansion** and
`ffn_up` the contraction — inverted relative to the text-model convention,
exactly as the loader assumed. The R142 finding was a false alarm for this
file._

_**But the hardcoded assumption was still a bug for every other mmproj.**_
_llama.cpp gates its swap on a shape test (`tools/mtmd/clip.cpp:1913`:
`ff_down_w->ne[0] == hparams.n_embd`, plus a legacy projector-type allowlist),
precisely because newer exports name these correctly. SwarmLLM assumed the
legacy layout unconditionally, so a correctly-named mmproj (Pixtral, InternVL,
newer conversions) hard-failed on a dimension mismatch in the first MLP matmul._

_Now resolved per-file by `vision.rs::clip_ffn_is_swapped`, mirroring
llama.cpp's check. Note candle reverses GGUF's `ne` on read
(`vendor/candle/candle-core/src/quantized/gguf_file.rs:438`), so `dims()` is
`[out, in]` and the test reads as `dims[1] == hidden_size`. A square FFN is
undecidable from shape and keeps the legacy reading; no CLIP-family tower is
square in practice. 4 unit tests._

### LLaVA chat template eval-failure fallback path — **CLOSED R148** (2026-07-22)

_Closed, and the investigation found a larger bug underneath it._

_The reported gap was real: when a template existed but failed, the fallback
chain checked only for `start_of_turn` (gemma) before dropping to ChatML — the
model-name heuristic that picks vicuna for LLaVA lived exclusively in the
`template.is_none()` branch. Extracted to `fallback_by_model_name` and now
consulted by both branches, with template-body evidence still taking priority
over the name since it describes the model that shipped it._

_The larger bug: **that failure path was almost never reached**, because
`apply_chat_template` rarely returns `None`. Probing the evaluator showed only
structural token errors fail — an unclosed `{% for %}` or a bare `{{`. An
unknown filter, an unknown variable, a stray `{% endfor %}` and an unclosed
`{% if %}` all evaluate quietly to `Some("")`. `build_prompt_with_model`
treated that as success and returned an **empty prompt**: no system message,
no user turn, and for a VLM no `<image>` placeholder — the exact
vision-embeddings-prepended-instead-of-inserted symptom the deferral described,
arrived at by a different route and with the whole conversation dropped too._

_`apply_chat_template` now reports an empty render from a non-empty message
list as `None`. Fixed at that level rather than in `build_prompt_with_model`
because `cli/split_test.rs` is a second caller with the same `unwrap_or_else`
ChatML fallback and the same exposure. An empty message list may still
legitimately render empty. 6 unit tests._

### Python SDK missing R140 pool endpoints — **CLOSED R147** (2026-07-22)

_`PoolClient.generate_code()` and `PoolClient.join(code)` added,
wrapping `POST /api/pool/generate-code` and `POST /api/pool/join`.
`generate_code` returns the `swarmpool://` blob directly (unwrapping the
`code` field) since that is the only thing a caller does with the
response. `python/README.md` gained the two-machine bootstrap example._

### Test infra: spawn_test_server duplicated across binaries — **CLOSED R147** (2026-07-22)

_Extracted to `tests/integration/test_server_common.rs`, pulled into both
binaries with `#[path = "test_server_common.rs"] mod test_server_common;`.
Each binary compiles its own copy (that's inherent to the `[[test]]` split)
but there is now one source of truth. `#![allow(dead_code)]` at module
level because each binary uses a different subset — `api_test` needs both
helpers, `test_metrics_health` only `spawn_test_server`. Both binaries pass
(30 + 30 tests); macOS CI will confirm the path import there._

### Configuration-reference doc additions — **CLOSED R147** (2026-07-22)

_The gap was larger than the 4 streaming knobs originally flagged: a
field-by-field diff of `InferenceConfig` against the reference table found
**45 undocumented `[inference]` options**. All are now documented, grouped
into new subsections (batching, prefix cache, speculative decoding,
SWARM-SPEC hedging/prefetch, activation transfer) rather than one
unreadable 60-row table. Defaults were read from the `default_*()`
functions rather than transcribed from prose, so they match the code.
Also added the missing `auto_manage.interval_seconds` /
`model_policies` and `inference.shard_range` rows. 91 → 133 documented
options._

_`swarmpool://` v2 documented in `docs/book/src/architecture/networking.md`
under a new "Invite code formats" subsection: full payload schema, the
listeners ∪ external-addresses union and why a NAT'd node needs it, the
legacy 8-char fallback, and the `ServiceUnavailable` / `invite_lan_only`
generation outcomes._

### `update.rs::apply_update_with_version` dead Option branch — **CLOSED R147** (2026-07-22)

_Signature tightened to `latest_version: &str`. The downgrade-prevention
check is now unconditional — there is no longer a way to call this
function in a mode that skips it._

### Worker compute waste on request cancel — **CLOSED R147** (2026-07-22)

_`DaemonMsg::CancelRequest { request_id }` shipped. `ResponseGuard` gained
a `worker: Option<Arc<WorkerHandle>>` field and a `disarm()` method: every
terminal return in `forward_direct` / `generate` disarms, so only a genuine
drop-before-completion (client disconnect, `tokio::select!` timeout, hedge
loser) fires the cancel. `Drop` is sync, so the IPC write is handed to
`Handle::try_current().spawn(...)` — best-effort, skipped outside a runtime._

_Worker side: the reader task short-circuits `CancelRequest` into a shared
`CancelledSet` (`DashMap<Uuid, Instant>`) rather than queueing it on `ipc_rx`,
for the same reason `PrefixFetchResult` is short-circuited — `handle_generate`
owns the main loop while it decodes, so a queued cancel would only be seen
after the work it was meant to stop had finished. Three consumption points:
`handle_daemon_msg` skips an already-cancelled `Forward`/`Generate` before
starting it; the sequential generate loop checks per token (bounding waste to
one forward instead of the full `max_tokens`); the main loop drops cancelled
decode slots via the new `SlotTable::take_matching` and frees their KV.
Unconsumed entries are swept after `CANCEL_RETENTION_SECS = 60` — a cancel
can legitimately arrive for an already-finished request._

_`BatchForward` is deliberately excluded (both in `cancelled_request_id` and
via `worker: None` on its guards): it is one fused matmul over N requests, so
one cancelled member cannot be skipped without dropping work the others still
want. Research note: this matches the explicit-abort design vLLM converged on
([PR #11190](https://github.com/vllm-project/vllm/pull/11190)) rather than the
`is_disconnected()` polling approach, which is unreliable behind middleware
([issue #10087](https://github.com/vllm-project/vllm/issues/10087))._

_Remote cancellation closed in the same round. Two paths:_

_**Remote-generate** was closed as a side effect: the inbound handler already
aborted its `gen_fut` on `SwarmMessage::CancelInference`, and aborting that
task now drops `ModelProcessPool::generate`'s future → drops the armed
`ResponseGuard` → messages the worker. Before the guard change, `abort()`
stopped the daemon-side task while the worker generated the full token budget
anyway._

_**Hedge losers** needed the explicit send. `hedge_dispatch` now dispatches
`CancelInference` to the losing holder after the race resolves — hedging
deliberately creates a duplicate remote forward on every fire, so leaving the
loser to compute a result we discard on arrival was the single largest source
of hedging's bandwidth/compute cost. The inbound handler gained a
`ModelProcessPool::cancel_request(request_id)` fan-out alongside the existing
abort-handle lookup, because an in-flight `LayerForward` has no abort handle —
the coordinator simply stopped waiting. Fan-out across workers (rather than
keyed by model) because the wire message carries only the request id; workers
are few and an unknown id is a no-op._

### Batch JoinHandle discard (wontfix)

`src/inference/router/mod.rs:595` `tokio::spawn(execute_batch)`
discards the JoinHandle. Mirrors the single-request dispatch
pattern at line 754; in-flight batches complete after the router
exits accepting work. R142.9 audit confirmed this matches the
documented design. Marked wontfix.

---

## R135 cross-pool review — hot-reloadable flag deferral

_**Closed R137** (2026-05-17). All 4 steps from the original plan
shipped:_

_1. `state.credits.allow_cross_pool_inference: AtomicBool` and
   `state.credits.share_model_catalog: AtomicBool` added to the
   `CreditPool` sub-struct (`src/daemon/state/credits.rs`)._
_2. Both mirrored from `config.pool.*` at startup in
   `SharedState::new()` (`src/daemon/state/mod.rs:351-356`)._
_3. `ConfigUpdate` in `src/api/admin.rs` extended with
   `allow_cross_pool_inference: Option<bool>` and
   `share_model_catalog: Option<bool>` Option fields. On PUT, both the
   runtime atomic AND the persisted TOML config are written. The
   `GET /api/admin/config` getter surfaces the runtime atomic values
   (not the startup-frozen config) so the dashboard reflects post-PUT
   state immediately._
_4. `pool::scope::cross_pool_extras` and
   `health/monitor.rs::broadcast_pool_model_availability` switched to
   read from the atomic. The k-anonymity floor
   (`share_model_catalog_min_members`) remains startup-frozen — that
   value is a privacy-policy parameter, not an on/off gate, and a
   restart pulse is acceptable when the operator wants to widen it._

_Pattern follows R121's `contribution_auto` mirror exactly. Regression
test `cross_pool_extras_honors_runtime_flag_toggle` in
`src/pool/scope.rs` verifies the runtime mirror takes precedence over
the startup-frozen config. 999 lib tests pass._

**Original deferral note preserved below for context.**

`pool.allow_cross_pool_inference` and `pool.share_model_catalog` were
read from `state.config.pool` which is startup-frozen. Unlike
`private_mode` (an `AtomicBool` on `state.credits`) and
`contribution_auto` (an `AtomicBool` on `state.models`), these flags
had no runtime-toggle path. A user who disables
`allow_cross_pool_inference` via a future
`PUT /api/admin/config { allow_cross_pool_inference: false }` would
not actually stop cross-pool routing until daemon restart.

---

## R135 sweep — security findings deferred for discussion

The R135 review surfaced two signature/security items that touch wire
formats and require user direction before changing:

### Acceptance signature timestamp omission — **CLOSED R147** (2026-07-22)

_Option (a) taken, per maintainer decision: `acceptance_payload` now binds the
invitation's `expires_at`, so the signature is implicitly time-bounded and
cannot be transplanted onto a different invitation record._

_The non-obvious part was the verification topology. There are **two**
verifiers, not one:_

1. _The **pool owner** validating an inbound `PoolAcceptance`. It takes
   `expires_at` from its own `pending_invitations` entry, never from the
   acceptance — the whole point is that the signer doesn't choose the value the
   signature is checked against. `handle_inbound_acceptance` was reordered to
   look the invitation up *before* verifying (it previously verified first,
   then looked up), and gained an explicit "invitation has expired" rejection
   as the enforcement half of the binding._
2. _**Any gossip receiver** re-verifying `PoolMembership.acceptance_signature`
   from a `PoolState` broadcast. It never saw the invitation, so the expiry has
   to travel with the membership: new `PoolMembership.invitation_expires_at`,
   populated by the owner from the value it verified against. Here the value
   arriving alongside the signature is fine — a third party is checking "did
   this node really accept this invitation", and an attacker replaying a real
   pair is asserting something true. The owner's path is where the adversarial
   choice mattered, and that one uses local state._

_**Flag day, as accepted.** `invitation_expires_at` is `#[serde(default)]` so
pre-R147 pool state still deserializes (missing → epoch) rather than failing to
parse, but it will fail signature verification. Mixed-version pools will reject
each other's member lists until both sides upgrade. Pools are personal-device
scale (`max_pool_size` 10) and the project is pre-release, so a versioned
variant wasn't worth the permanent complexity._

_Tests: signature must not verify against a lengthened or shortened expiry;
payload must actually change with the expiry (guards against a helper that
accepts the argument and ignores it)._

### `apply_pool_model_availability` cap eviction relies on single-task dispatch

_**Closed R137** (2026-05-18). Single batched partial-sort via
`select_nth_unstable_by_key` replaces the K × full-scan loop. For
`max_entries=5000` + `to_evict=128` overflow, ops drop from ~640K
(128 × 5000) to ~5K + 128 removes — roughly 10× faster on the
eviction path, with the same oldest-first contract. The
`apply_batched_eviction_drops_correct_oldest_set` stress test
exercises a 1000-entry catalog with 200 overflow to verify the
partial-sort branch retains the correct subset._

**Original deferral note preserved below.**

R135 reordered the eviction in `daemon/state/credits.rs`
to be POST-insert, which makes the post-condition `catalog.len() <=
max_entries` structural rather than depending on the pre-insert size
estimate being accurate. But the eviction loop itself still scans
the DashMap to find the oldest entry, which is O(n) per eviction —
not O(log n). For `max_entries = 5000` this is 25M ops per drain in
the worst case (full re-fill from a single 5000-entry announcement,
which is bounded above by `MAX_POOL_MODEL_ANNOUNCE_ENTRIES = 128`,
so the real worst case is bounded). Fine today, would want
attention if either cap grows materially.

---

## Inference performance — research backlog (R135 brief)

> **STATUS as of 2026-05-20 (post-R139):**
>
> **Shipped (in tree, not gaps):**
> - Tier 1A — activation compression default-on (R136)
> - Tier 1B — tail-latency hedging single-segment (R136); multi-segment open
> - Tier 2E — compute-win shipped via R136 Layer 1 (different algorithm —
>   prompt-lookup-decoding rather than LMSYS 2D-window — but same gain
>   in the input-grounded workload). LMSYS 2D variant itself open.
> - Tier 3G — cross-REQUEST prefix sharing (block-boundary BLAKE3
>   chain in `prefix_cache.rs` gives system-prompt dedup across users).
>   True radix-tree refinement open.
> - Tier 4K — daemon-side STREAM-chunked send + receive (R139). RR
>   chunked + WAN bench script + worker row-tiled streaming open.
>
> **Genuinely open:**
> - Tier 1C (EAGLE-3) — needs externally-trained draft heads per model.
> - Tier 2D (tree-based draft expansion / FlowSpec) — `speculative.rs`
>   is linear γ-chain only.
> - Tier 2F (KV cache quantization / KIVI) — no `kv_quantization`
>   config; kv_cache stores raw.
> - Tier 3H (sub-token activation deltas) — no delta path.
> - Tier 3I (PowerInfer-style activation sparsity) — no sparsity
>   profiler.
> - Tier 4J (pre-emptive layer dispatch) — speculative novelty.
> - Tier 5L (FP8 activation) — needs Hopper+ hardware.
>
> See `## A node holding every shard monopolises the model (measured 2026-07-27)

**Status: root-caused, not fixed.** Deliberately left for a deliberate change —
this is core routing, and it should not be altered without an A/B across a real
swarm.

### Symptom

A GPU node holding shards 0 and 1 of `llama-3.2-1b-instruct-q8-0` **in VRAM**
sent every request, in full, to a CPU-only node that happened to hold all three
shards — including the 10 of 16 layers it could have run locally on the GPU.
Given the measured prefill cost of that CPU node (§ "CPU prefill throughput"),
this is the difference between minutes and seconds on a long prompt.

### Root cause

`scheduler/parallax.rs::route_shortest_path` builds **one DP vertex per
(candidate, available_range)**, and treats each range as indivisible. Edges
require an exact boundary match:

```rust
if vertices[w_idx].range.0 != v_end { continue; }
```

For the live case the candidate set was:

```
Pipeline candidate node=9684263580c6660f ranges=[(0, 16)]  can_be_first=true can_be_last=true
Pipeline candidate node=0718d8b987a4975a ranges=[(0, 10)]  can_be_first=true can_be_last=false  <- local, GPU
DIAG: parallax routing selected chain segments=1
```

The local vertex `(0,10)` needs a successor starting at layer 10. The remote
node's only vertex is `(0,16)`, which starts at 0. **No vertex starts at layer
10**, so the two-segment chain is not representable at all — the DP is not
choosing the single-segment route over a split, it is the only path that exists.

The cost model is therefore never consulted on the question. Note that it would
have chosen correctly if asked: with the observed `ms_per_layer=107` for that
peer, splitting costs ~643ms against ~1715ms for all-remote.

### Consequences

- Any node holding a complete model monopolises every request for it, however
  slow that node is, and no other node's shards can contribute.
- The effect is strongest exactly where it hurts most: a small model that fits
  entirely on one modest node is also the model most likely to be fully held.
- Load cannot be spread across holders of a fully-held model.
- It is self-concealing: the pipeline reports `segments=1` and succeeds, so it
  reads as a healthy fast path rather than a missed opportunity.

### Fix sketch

Allow a candidate's range to be sub-divided at the boundaries that actually
matter, rather than splitting everywhere (which would make the vertex set
O(L²) per candidate for no benefit). The useful split points are the union of
every candidate's range starts and ends: emit a vertex for each sub-range of
`available_ranges` delimited by those points. In the live case that adds the
vertex `(10,16)` for the remote node, making the split representable, and the
existing cost model then picks it on merit.

### Update 2026-07-27: capability built and gated OFF; the cost model is the real blocker

Partial ranges are now implemented (`config.inference.parallax_partial_ranges`,
**default off**) and the split routes correctly — verified live as
`LOCAL(0-10, shards [0,1]) + remote(10-16, shard [2])`, with the shard span
reported accurately for the first time. The `shard_id` hazard below is resolved:
segments are re-pointed at the first shard their range covers, and every consumer
needing the full span goes through `ModelRegistry::shards_spanned_by_segment`.

**Consequence found 2026-07-27 (external report):** `inference.encrypted_pipeline`
is **non-functional while this is off**. Encryption forces first and last segments
local, so the middle must come from a peer — and a peer holding the whole model
has one indivisible range that can be neither a middle segment nor a remote
encrypted source/sink. A tester found `encrypted_pipeline = true` unable to
assemble in either topology tried, including the nominal boomerang (local holds
head + tail, peer holds the middle). Proven to be this and not a separate defect
by `encrypted_boomerang_is_unroutable_without_partial_ranges`, which fails to
route with partial ranges off and produces the correct
local(0,3) → peer(3,21) → local(21,28) chain with them on. So this option is not
only a throughput question — it gates a shipped privacy feature, which raises the
priority of closing the cost-model gap below.

**It is off because measurement contradicted the prediction in this section.**
The claim above — that the cost model "would have chosen correctly if asked"
(~643ms split vs ~1715ms whole) — was wrong, because it compared the cost of ONE
forward pass. A request is many forward passes, and the two options do not scale
the same way:

| | route | measured |
|---|---|---|
| `llama-3.2-1b-instruct-q8-0`, 16 tokens | whole on CPU node | **11.2s** |
| same | split GPU 0-10 / CPU 10-16 | **17.8s** |
| `tinyllama-1.1b`, 16 tokens | whole on CPU node | **3.2s** |
| same | split GPU 0-12 / CPU 12-22 | **5.9s** |

A single remote segment covering every layer is delegated in ONE message and
decodes remotely with no per-token network. Any multi-segment chain exchanges
activations **once per token**. `vertex_cost` charges a remote hop's
`2 * latency_ms` once per *segment*, so it cannot see that difference and will
keep choosing the split. On this LAN pair the per-token cost of the extra
boundary exceeded everything the GPU saved.

**Second, larger problem found while measuring: without observations the DP is
blind.** `compute_ms` falls back to `UNKNOWN_COMPUTE_MS = 0` when a candidate has
neither an observed per-layer latency nor a gossiped throughput estimate, and the
local node is deliberately given `observed_latency_ms_per_layer = None`. On a
freshly restarted node every candidate therefore costs only its network term, and
whole-vs-split **ties at ~10ms** — the outcome then depends on vertex iteration
order, not on merit. This is why the same configuration split consistently in one
session and not at all after a restart. It also means the routing quality of a
node silently depends on how long it has been up.

### Cost-model work done 2026-07-27, and the ONE piece still missing

Three of the four gaps are fixed and shipped (all default-on, they improve
routing generally, not only partial ranges):

1. **`UNKNOWN_COMPUTE_MS` 0 → 25.** An unmeasured candidate was free, so cold
   nodes tied on network alone and iteration order decided; an unknown node also
   outranked a measured-good one. Now scales with layers taken on.
2. **The local node is measured.** `record_peer_segment_latency` was called only
   for remote segments, and `gather_candidates` hardcoded the local node's
   observation to `None` — so local compute was free at any width and the router
   would pile every layer onto a slow local CPU rather than use a faster peer.
   Local segments are now timed and used.
3. **Per-token network for mid-chain segments.** A vertex whose range does not
   start at layer 0 is entered from the previous segment, so the coordinator
   round-trips into it per token; it is charged
   `2 * latency * ASSUMED_FORWARD_PASSES`. A segment starting at 0 (local, or the
   delegated whole-model case) pays network once. This is expressible per-vertex
   after all — `range.0 != 0` is exactly the predicate — so no chain-shaped
   comparison was needed.

**Still missing, and it is why partial ranges remain off:** the per-token term
only bites in the *unobserved* branch. Once a peer has an observed
`ms_per_layer`, `vertex_cost` sets `network_ms = 0` and folds everything into
compute, on the (correct) grounds that the observation already includes the
round trip. The problem is that the SAME per-layer figure is then used for the
whole-model delegated alternative, which does **not** pay a round trip per pass.
An observation taken while serving a 6-layer mid-chain segment carries that
segment's RTT amortised over 6 layers; reusing it for a 16-layer delegated
segment charges the RTT ~2.7 times over. The delegated option is systematically
overcharged, so the router keeps preferring the split.

Verified after the three fixes above, with the flag forced on (LAN pair, GPU +
6-core CPU, 16-token replies, warm, `segments=` confirmed in the log both ways):

| route | runs (ms) | median |
|---|---|---|
| single segment (default) | 8057, 8093, 10242, 10464, 11786 | **~10.2s** |
| split GPU 0-10 / CPU 10-16 | 11469, 11817, 12155, 12422 | **~12.0s** |

**The fix is to record two distinct figures rather than one.** A peer needs
`observed_midchain_ms_per_layer` (includes per-pass network — what
`record_peer_segment_latency` already produces from `forward_through_segments`)
and `observed_delegated_ms_per_layer` (pure remote compute). The second has no
source today because **the `remote_generate` fast path never records anything** —
`record_peer_segment_latency` has exactly one production caller, in the
multi-segment path. That is also a self-reinforcing blindness worth fixing on its
own: a node whose requests all take the fast path never learns a thing about its
peers, and only ever gets numbers second-hand via `merge_peer_segment_latency`
gossip. Deriving a per-pass figure from the fast path needs care about units —
the segment wall-clock there covers prefill plus every decode step, so the usable
quantity is `(total_ms - ttft_ms) / max(1, completion_tokens - 1) / layers`, both
of which the trace already carries.

A caution for whoever picks this up: two A/B runs during this work were invalid
and nearly produced the wrong conclusion. One measured the prefix cache rather
than the router (repeat prompts collapse to ~1.5s), and one measured a stale
process because a restart silently failed to rebind port 8800 and the health
check passed against the *old* daemon. Confirm the PID changed and confirm
`segments=` in the log before trusting any number here.

Worth noting the shape of request that should favour splitting and was not
isolated here: a **prefill-dominated** one, where the prompt is a single large
forward pass and per-token round trips apply only to the few decode steps. A
585-token prompt measured 129.8s whole; the comparable split run could not be
attributed cleanly because the DP had by then reverted to a single segment for the
tie reason above. That experiment is worth redoing once (1) and (2) exist.

**Implementation hazard (RESOLVED 2026-07-27 — kept for context).** `PipelineSegment.shard_id` is set
to the candidate's *first* shard regardless of the segment's layer range, so a
sub-range segment covering layers 10-16 would be labelled with shard 0. That
mismatch already exists today — it is why `retract_shard_holder_claims_for_range`
had to be added, after a `blk.10` (shard 2) failure retracted shard 0 — but
making sub-ranges routable turns it from an edge case into the normal path.
Every consumer of `segment.shard_id` needs auditing against `layer_range`
before this lands, and the field should probably carry the shards the range
actually spans rather than a single id.

Then verify against the cost model's blind spot: the local node is
deliberately given `observed_latency_ms_per_layer = None`
(`scheduler/mod.rs`), so it costs ~0 and will look attractive for any range it
can serve. That is fine while local really is the fastest option, and wrong on
a slow local machine paired with a fast peer — worth measuring once splits are
representable, since today the question never arises.

### Greedy decoding is far more load-sensitive than sampling (observed 2026-07-27)

Not a bug, but worth knowing before reading a report about it. `temperature = 0`
makes a request eligible for the SWARM-SPEC L1 n-gram-only speculative path,
which drives the pipeline with **per-token `LayerForward` round trips**. Normal
sampling takes the `remote_generate` fast path instead, which delegates the whole
generation in one message.

Consequence: greedy decoding is far more exposed to a busy remote. Observed
while deliberately saturating the CPU node — single-token forwards that normally
cost ~1.7s (16 layers at the measured 107ms/layer) sat in the queue for 113s and
196s, blowing `compute_segment_timeout`'s 32s decode budget. With only one holder
of the model there is no standby, so the request failed outright with
`Segment 0 failed with no standby available`, while sampling requests to the same
node in the same window succeeded. On an idle node the same greedy request
completes normally.

The timeout itself is fine — it already scales with layer count and
prefill-vs-decode, and 32s is generous against 1.7s of real compute. Two things
would genuinely help, both listed above: making the split representable so a
saturated single holder is no longer a single point of failure, and feeding the
observed per-layer latency (which already includes peer-side queuing) into the
timeout instead of the fixed 2s/layer guess. Note this compounds with the
monopolisation bug — a fully-held model has exactly one candidate, so it can
never have a standby.

Relevant because greedy decoding is what tool calling, benchmarks and
reproducible runs use, so it is over-represented in exactly the traffic testers
generate.

## Two reported crashes, one cause — a partial holder accepting whole-model work (FIXED 2026-07-27)

Reported as two separate crashes while chasing cross-node prefix-KV. They are the
same bug seen from opposite ends, and **neither is a prefix-KV fault**.

`daemon/dispatch/remote_generate.rs::has_model_locally` answered "is this model
known here" by checking that `manifest.json` exists. Every holder of a *single*
shard keeps the manifest, so a node holding any part of a model accepted a peer's
request to run **all** of it. The worker then ran a full decode over a shard
window that does not cover the whole model, and failed in one of two places
depending on which end was missing:

| missing end | what happens | reported error |
|---|---|---|
| the **head** | the embedding table is only loaded for a first segment, so raw token ids reach the first attention block | `attn_norm: shape mismatch in rms-norm [1, 128] [3072]` — 128 is `prefill_chunk_tokens`, 3072 the hidden size |
| the **tail** | `executor.rs` returns `Ok(layer_in)` — hidden states — for a non-last segment, and that reaches the sampler | `unexpected rank, expected: 1, got: 2 ([20, 3072])` |

**The prefix-KV fetch in the second report was incidental.** It shortened the
final prefill chunk to 20 tokens, which is the only reason 20 appears in the
message; without a cache hit the same crash would report the full prompt length.
Cross-node prefix-KV transfer was confirmed working in that same run
(`kind="prefix_kv_data"`), and nothing here implicates it.

**Fix**: the guard now asks whether this node can serve the *requested layer
range*, and the range must sit inside ONE contiguous run of locally-held layers —
holding both ends of a model is not the same as holding the middle, and a decode
cannot skip a gap. Refusal was already handled by the caller, which reports the
error and picks another holder. Decision extracted as `range_is_covered` and unit
tested, including the prompt-privacy layout (both ends, no middle) which must
NOT count as coverage.

**Why it did not reproduce from a clean start** (attempted: tail-only node,
`gpu_layers = 0`, single request, no OOM or reschedule in the history — 3-segment
pipeline ran clean 3/3, then 6/6). It needs the *sender* to ask for a range wider
than the receiver holds, which a correct scheduler does not do. Stale holder
information is enough to produce it, which fits both reports arriving after
repeated restarts and role changes. The receiving-side guard is the right place
to fix it regardless of how the sender got it wrong.

**Cross-node prefix-KV: measured and CONFIRMED (2026-07-27).** Retried once the
crash above was fixed. Two real machines (GPU node + 6-core CPU node), tinyllama
Q4_K_M, 709-token prompt:

| | |
|---|---|
| CPU node, prompt never seen anywhere | **182s** |
| CPU node, same prompt already cached on the peer | **11s** |
| | **16.5x** |

`DIAG: cross-node prefix HIT — hydrated KV matched_tokens=1536` on the consumer.
The README's 12.9x (2026-04-20, loopback) therefore holds and is if anything
conservative — this is 16.5x across a real network.

**But the producer side is intermittent, and that is the open item.** The first
attempt at this measurement showed no speedup at all (180s vs 182s cold) and zero
fetches, because the producing node never snapshotted its prefix: a `route=local`
request with `prompt_tokens=709` — comfortably inside `min_tokens=32` and
`max_prompt_tokens=8192` — logged no insert. A later identical-shaped request on
the same node inserted 24 blocks and the fetch then worked.

`insert_from_kv` had five silent early returns, so there was nothing in the log to
say which condition stopped it. All five now log at debug with the values that
decided them. The next occurrence should name itself; until then, treat the
speedup as real but not reliably available.

### Idle VRAM unload: the region-demand reprieve had no ceiling (fixed 2026-07-27)

Reported externally: `qwen2.5-coder-7b` and `llama-3.2-3b` model-workers resident
on an 8 GB card **2h16 past the last request** with `idle_unload_secs = 300`
configured, on a node that subsequently hit GPU-OOM.

`try_idle_vram_unload` has two gates. The first — idle for `idle_unload_secs` —
was satisfied. The second refuses to unload while regional demand is at or above
`IDLE_DEMAND_EMA_THRESHOLD = 0.1`. The two reported models sat *just* over it:

```
llama-3.2-3b        BE 0.167   TH 0.126
qwen2.5-coder-7b    BE 0.107
```

so the reprieve applied indefinitely and `idle_unload_secs` never meant anything
for them.

**The deeper issue is that the demand gate is a weak proxy.** It exists because
`last_request_at` is set by `ModelTrustInfo::record_request`, which is called
only from the OUTBOUND router path (`distributed_exec.rs`) — serving a peer never
updates it. So without the gate a node would evict models it was actively
serving. But regional demand says nothing about whether requests are reaching
THIS node: a model nobody ever asks us for stays pinned as long as some region
wants it in the abstract.

**Shipped fix**: the reprieve now expires. Past `IDLE_HARD_UNLOAD_MULTIPLIER`
(12x) the configured window — one hour at the 5-minute default — VRAM is
reclaimed regardless of regional demand. Short enough that an unused model cannot
hold a card all day, long enough that a genuinely useful one is never evicted
mid-use.

**Better fix, not done**: track when we last *served* a model, and use
`max(last_request_at, last_served_at)` for the idle test. Then the gate answers
"has anyone asked me for this", which is the question that actually matters, and
the regional-demand proxy can go entirely. It needs a new per-model timestamp
(additive on `ModelTrustInfo`, or in `state.metrics` alongside
`record_segment_served`, which is currently aggregate and carries no model id).

A test caught an edge case worth keeping in mind: `idle_unload_secs as i64` wraps
a very large configured window to a negative number, which inverts the comparison
and unloads immediately rather than never. Converted with `try_from` instead.

### Observed once, not reproduced: SentencePiece markers in reply text (2026-07-27)

Recording because it is output corruption and a literal capture exists, not
because it is understood. On the first request after starting a v0.3.37 node,
`tinyllama-1.1b-chat-v1.0.q4-k-m` returned:

```
Sure.▁What▁are▁the▁key▁ingredes▁for▁this
```

`▁` (U+2581) is SentencePiece's word-boundary marker; it should have become a
space during detokenization. Note "ingredes" is also not a word, so the whole
span looks like raw pieces rather than one stray character.

**Not reproduced in ~8 further attempts**: five repeats of the identical prompt,
two other models (including a BPE-tokenizer one for contrast), and a deliberate
cold start with the model freshly loaded — the exact condition it first appeared
under. All returned `markers=0`.

Ruled out: it was NOT served by a peer on an older build — the trace shows
`route=local segments=1`, so our own path produced it.

If a tester reports garbled spacing, this is the first thing to check, and the
detokenization path for SPM models is where to look. A reproduction would make it
actionable; without one there is nothing to fix against.

**A second, similar one-off the same day**, recorded because two rare
text-assembly corruptions may share a cause. A 3-segment distributed reply began
`" waterThe cycle, also known as the hydrologic"` — the first two tokens
transposed, where every other run of the identical prompt gave
`"The water cycle, ..."`. Six immediate repeats were clean, as were six repeats
in the marker case. Both are reply text arriving wrong rather than the model
choosing badly, both appeared once in heavy use, and neither reproduces on
demand. If either is ever caught reliably, check whether the other goes with it —
a shared ordering or buffering fault in how token text is accumulated would
explain both, and would be a single fix rather than two.

## Continuous batching engages but yields almost nothing (diagnosed 2026-07-27)

**Status: root-caused, not fixed — the fix is feature-scale.** Reported externally
as "zero aggregate throughput gain on CPU" with `continuous_batching = true` and
`max_slots=8`. Reproduced and traced.

The report's framing — "requests are being processed one at a time regardless of
the batching config" — is close but not quite right, and the difference matters.
Batching **is** engaging. With worker debug logging finally visible (see below),
all four concurrent requests were admitted to the slot table:

```
slot admission accepted — request will decode in a shared batch occupied=0
slot admission accepted — request will decode in a shared batch occupied=1
slot admission accepted — request will decode in a shared batch occupied=2
slot admission accepted — request will decode in a shared batch occupied=3
```

The loss is one level down, in `split/executor.rs::forward_batch`, which requires
**homogeneity**: every item must share the same `(seq_len, index_pos)`, and a
mixed batch "falls back to sequential forwards so a slow slot doesn't block the
fast ones". Independent requests almost never align — different prompt lengths
give different `index_pos` immediately — so the admitted batch decodes
sequentially anyway.

Measured on tinyllama Q4_K_M, CPU, 4×16 tokens:

| workload | wall | aggregate |
|---|---|---|
| sequential, one at a time | — | ~2.5 tok/s |
| 4 concurrent, varied prompt lengths | 27.6s | **2.30 tok/s** |
| 4 concurrent, same prompt length | 21.2s | **3.02 tok/s** |

The same-length row indicated the gate is real, though note the later GPU run
found that repeating identical prompts across trials inflates such numbers via
prefix-cache hits — treat it as directional rather than exact. It is also well short of the near-linear scaling batching implies,
which suggests alignment additionally breaks during decode, not only at
admission.

**Fixing it properly is ragged batching**, i.e. one batched forward over
sequences at *different* positions, which needs per-sequence KV offsets and
attention masks rather than a single shared `index_pos` — the problem paged
attention exists to solve. That is a feature, not a patch, and it should be
measured against the CPU prefill work above since both target the same
bottleneck.

**Meanwhile the config over-promises.** `continuous_batching = true` and
`max_concurrent_decode_batch = 8` read as though concurrency will scale
throughput, and on this path it does not. Worth either documenting the
homogeneity condition on those settings or gating the claim until ragged
batching lands.

### Ragged batching — spec, and the measurement that says don't build it yet

**Research run first, because it decides whether the feature is worth building.**
Aligned prompts (same length, so they pass the homogeneity check and genuinely
batch), tinyllama Q4_K_M on CPU, 16 tokens each:

| batch | wall | aggregate | vs batch=1 |
|---|---|---|---|
| 1 | 6.9s | 2.31 tok/s | — |
| 2 | 11.7s | 2.73 tok/s | +18% |
| 4 | 22.2s | 2.88 tok/s | **+25%** |

**Four times the work for 25% more throughput.** That is nearly flat, and it is
the answer: batching amortises *weight loading*, which only pays when decode is
memory-bandwidth-bound. On this path it is **compute-bound** — the same finding
as the prefill section above, where candle's CPU quantized matmul runs at roughly
5-9% of the chip's FP32 peak. When you are compute-limited, batching N sequences
costs N times the arithmetic and returns nothing.

So ragged batching is a **GPU-path feature**, not a CPU one. Building it to fix
the reported CPU result would be building the wrong thing. Re-run the table above
on a GPU node before starting; if it scales there, the spec below applies.

#### GPU research run (RTX 3070 Laptop, 2026-07-27) — the answer is NO

The spec above said to re-run the scaling table on a GPU before building
anything, because batching only pays when decode is memory-bandwidth-bound.
Done, on `llama-3.2-3b-instruct-q4-k-m`, all 28 layers on CUDA (verified:
`Split model using CUDA GPU layer_start=0 layer_end=28`, 7.8 GB of 8 GB VRAM):

| batch | median | vs batch=1 |
|---|---|---|
| 1 | 24.6 tok/s | — |
| 2 | 27.0 tok/s | 1.10x |
| 4 | 30.2 tok/s | **1.23x** |

**Four times the work for 23% more throughput — the same near-flat result as
CPU.** On this evidence ragged batching is not worth building. The staged design
above stands if someone wants it, but nothing in these numbers justifies the
work.

**Getting a trustworthy number took three attempts, and the first two were
wrong in opposite directions.** Recording them because each is an easy trap:

1. **Repeated identical prompts** across trials produced 1.77x at batch=2 on
   tinyllama — prefix-cache hits, not batching. Use fresh content per trial.
2. **Unique prompts of varying length** produced 1.02x on the 3B — but varying
   length fails `forward_batch`'s homogeneity check, so that measured the
   *sequential fallback*, not batching. The control needs unique content at
   **identical token length**: same prompt shape with a different one-token word.
3. Background load (a CUDA build, the soak, and browser tabs at ~190% CPU)
   produced 3x variance within a single configuration. Check `uptime` first.

**Caveats on the conclusion.** One consumer laptop GPU, and the 3B nearly fills
its 8 GB, so there is little headroom for the larger activations a batch needs —
a datacentre card with room to spare could behave differently. If anyone wants to
revisit this, the measurement to take first is not batching at all: establish
where the time actually goes at batch=1 (HTTP → router → worker IPC → compute),
because a fixed per-request overhead would cap the achievable gain no matter how
good the batching is.

#### What actually blocks it today

`split/executor.rs::forward_batch` requires every item to share
`(seq_len, index_pos)` and otherwise falls back to sequential forwards. Two
distinct halves to that:

1. **`seq_len` is already uniform during decode** — every sequence contributes
   exactly one token, so this half is satisfied for free. It only bites during
   prefill, where chunked prefill already exists to even lengths out.
2. **`index_pos` diverges immediately**, because sequences sit at different
   lengths. This is the whole of the decode blocker.

Encouragingly, `mask_with_offset(query_len, kv_len)` already exists and builds a
causal mask at an arbitrary offset — it was added for the prefix-cache path. The
masking primitive is therefore not missing.

The real structural obstacle is the KV cache: `split/kv_cache.rs` keys per
`(model_key, request_id)`, so each sequence owns **separate** K/V tensors. A
single batched attention wants them gathered. That is what paged attention exists
to solve, and a full paged rewrite is the expensive reading of this work.

#### Staged design that avoids the paged rewrite

The insight that makes a first version cheap: at decode, per-sequence attention
is over a few hundred KV entries and is *tiny*, while the projections and MLP are
where the weight-bound cost lives. So split the layer rather than the cache:

- **Batch the position-independent parts.** Embedding lookup, layer norms, QKV
  projections, MLP, and the output head all take `[N, 1, hidden]` and need no
  knowledge of sequence position. This is where the amortisation win is, and it
  needs no KV changes at all.
- **Loop attention per sequence**, using each one's own KV tensors and its own
  `index_pos` via the existing `mask_with_offset`. N small attention calls, no
  gathering, no padding, no block tables.
- **Keep the homogeneity fallback** for prefill, where `seq_len` genuinely
  differs and chunked prefill is the right tool.

That captures most of the theoretical gain for a fraction of the work, and it is
measurable against the table above before committing to anything larger. Only if
attention itself becomes the bottleneck does paged KV become worth it.

#### Also fix the promise, regardless

`continuous_batching = true` and `max_concurrent_decode_batch = 8` read as though
concurrency scales throughput. On the CPU path it does not, and the settings do
not say so. Either document the homogeneity condition and the compute-bound
caveat on those options, or stop defaulting them on where they cannot deliver.

### Prerequisite fixed along the way: the worker was undebuggable

`model-worker` subprocesses were spawned with no verbosity flag, so they fell
back to the config file's `logging.level` and emitted INFO only. Running the
daemon with `-v` produced nothing extra from the process where inference actually
happens, and a `debug!` added there while chasing this silently never appeared —
which nearly produced the wrong conclusion (that batching never engaged, when the
admission logs simply could not be seen). The daemon's `-v` count is now passed
through to the worker.

## CPU prefill throughput is the dominant cost for modest nodes (measured 2026-07-27)

**Status: measured, not addressed.** Recorded because it sets the ceiling on
what a CPU-only volunteer node can usefully serve, and because the numbers make
the trade-offs concrete.

Measured on a Dell OptiPlex 3090 Micro (Intel i5-10500T, 6 cores @ 2.3GHz, no
GPU) serving `llama-3.2-1b-instruct-q8-0`, via the request tracing added in
v0.3.30:

| Prompt | Prefill (time to first token) | Notes |
|---|---|---|
| 13 tokens | ~4.0s | of a 4.6s total request |
| 613 tokens | ~285s | fresh prompt, prefix cache cold |
| 1322 tokens | ~319s | `ttft_ms=319470`, `total_ms=321539` |

Two things follow.

**Prefill dominates completely.** On the 1322-token request, time-to-first-token
was 99.4% of the total. Decode measured ~230ms/token; prompt processing measured
~267ms/token. Any optimisation aimed at decode is close to irrelevant for these
nodes — the lever is prompt processing.

**It is not an algorithmic bug, and that was checked.** Doubling the prompt
doubled the time (2.02×), so it is linear, not quadratic. `model.forward()` does
receive the whole prompt tensor in one pass, so prefill is batched as intended.
The cost is candle's CPU quantized matmul: roughly 9 GFLOPS against a chip whose
FP32 peak is on the order of 100-200 GFLOPS, i.e. ~5-9% of peak. llama.cpp's
hand-written quantized CPU kernels are the reference point for what this could
be; candle is not competitive with them on CPU.

The per-token ratio is misleading on its own and shouldn't be used as evidence
of a batching bug — a batched prefill *should* be much cheaper per token than
decode, and here it isn't, but only because the CPU is compute-starved rather
than bandwidth-starved. See gotcha #181 for the full reasoning, including the
prefix-cache trap that made an early measurement read 10s instead of 285s.

Options, roughly in order of effort:

1. **Route long prompts away from slow nodes.** The scheduler already has
   `ms_per_layer` per peer (`/api/admin/performance`). A node that cannot
   plausibly prefill the prompt inside the budget could be skipped when a
   faster holder exists. Cheapest real win; does nothing when the slow node is
   the only holder. **Blocked on the section above** — today a node holding
   every shard is the only representable route, so there is no alternative for
   the cost model to prefer.
2. **Report prefill progress from the serving node.** Today a long prefill is
   indistinguishable from a dead peer — the remote sends nothing for minutes,
   which is why the timeout had to be lengthened rather than made adaptive. A
   feature-gated progress signal would let the coordinator hold the connection
   open on evidence rather than on a guess, and fail fast on genuine silence.
   Must be additive and feature-gated per `.claude/rules/architecture.md`.
3. **Faster CPU quantized matmul.** Either an optional llama.cpp-backed CPU
   path or improved kernels. Largest effort, largest payoff, and the only
   option that changes the ceiling rather than routing around it.

Until one of these lands, the practical guidance for operators is that a
CPU-only node is well suited to short prompts and poorly suited to long-context
work, and that this scales with the machine rather than being a fixed limit.

## R136 local 3-node benchmark — measured results` below for
> the actual numbers, and the per-Tier sections below for full
> context on what's open.

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
  `inference.activation_compression`. Default **TRUE** since R136
  (was deferred when first written). Wired through the worker IPC
  path.
- **Prefix cache** (`split/prefix_cache.rs`, 53K bytes).
- **Pipeline failover** with hot-standby nodes per segment.
- **Q8_0 wire compression (L0)**, **n-gram cascade (L1, draft-free
  + draft+ngram, single + multi-segment)**, **tail-latency hedging
  (L2, single-segment race-then-discard)**, **predictive prefetch
  (L3, observability-complete decision dispatch)** — all R136.
  See `## R136 local 3-node benchmark — measured results` below.
- **L1 hit/miss telemetry** (R137) surfaced via
  `GET /api/admin/stats → swarm_spec.ngram = { hits, misses, total,
  hit_rate }` — operator-facing signal of whether L1 is firing.

### Tier 1 — high-leverage, low-risk wins (do these first)

#### A. Default-on activation compression with quality gate — **SHIPPED R136**

**Status:** flipped `default_activation_compression() = true` in
`src/config/inference.rs`; quality gate ships as a unit test in
`src/inference/quant.rs::quality_gate_typical_hidden_state_distribution`
that asserts L∞ < 0.05 and MAE < 0.01 on a representative
post-LayerNorm distribution with 3-5σ outliers every 100 lanes.
Real-inference A/B (3-node loopback): single-segment routing
+4-17% across workloads; distributed-pipeline LOOPBACK shows no
win because the encode CPU (~17µs/forward) rivals the saved
wire-time on sub-ms loopback hops. Per-model override available
via config.toml when a quality regression is observed. See R136
benchmarks below.

Original analysis preserved below.

#### A. Default-on activation compression with quality gate (original analysis)
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

#### B. Tail-latency hedging for slow pipeline hops — **SHIPPED R136** (single-segment dispatch; multi-segment deferred)

**Status:** decision logic + EWMA tracker + true dispatch all ship.
`inference/hedging.rs::HedgeTracker` tracks per-(model, segment, holder)
EWMA latency + rate budget; `pipeline/hedge_dispatch.rs::forward_verify_with_hedge`
races primary vs duplicate-to-alt-holder via tokio::select! with a
fresh UUID for the hedge so pending_layer_results doesn't collide.
Default off via `inference.hedge_enabled`; loopback bench doesn't
trigger it because RTT variance is too consistent to exceed
1.5×p99 — wire is in place for WAN deployments where it will
fire and matter.

**Still deferred (multi-segment hedging):** v0 only handles
single-segment pipelines (where the alt holder picks from
`shard_holders` for the same shard_id). Multi-segment would need
a full alternative-pipeline assembly (duplicate the B→C chain to
B'→C'). Substantial design + bandwidth cost.

Original analysis preserved below.

#### B. Tail-latency hedging for slow pipeline hops (original analysis)
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

#### E. Lookahead decoding (n-gram parallel verify) — **compute-win SHIPPED R136** (different algorithm, same gain); LMSYS 2D-window variant remains open

**Status.** The headline win of LMSYS Lookahead Decoding — draft-free
spec from n-gram lookup — ships via R136 Layer 1
(`inference/ngram_lookup.rs` + `pipeline/ngram_only_spec.rs::try_ngram_only_distributed`).
Real-inference single-segment bench: **+45% summary throughput at 77%
n-gram hit-rate** (`docs/FUTURE_WORK.md § R136 local 3-node benchmark`).
Different algorithm — we do prompt-lookup-decoding (apoorvumang /
PROMTEC ACL 2025) rather than LMSYS's 2D-window — but same compute
gain in the input-grounded workload where it matters.

**What remains open.** The specific LMSYS 2D-window mechanism (n-grams
emitted in the SAME forward, verified in the NEXT) is a separate
algorithm and not implemented. It could stack on top of L1 for the
non-prompt-grounded fraction of the workload (free-form chat where L1
hit-rate dropped to 0% in the synthetic bench).

**Why this gap is low-priority.** L1 already covers code completion +
RAG (the workloads that matter for Claude Code / MCP). Free-form chat
is the residual case, and there the gain over greedy decode is the
~5-25ms per-token saved by parallel verify on n-grams that may or may
not match — a smaller absolute win than L1's 45%. Defer until a
workload mix surfaces where the residual matters.

**Effort.** 2-3 weeks if pursued. New module + slot-table coupling.

Original analysis preserved below.

#### E. Lookahead decoding — original analysis
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

**Research update (R147, 2026-07-22) — do NOT implement KIVI as specced
above.** Two findings change the recommendation:

1. **KIVI's numbers depend on kernels we don't have.** The speedup comes
   from fusing dequantisation into the attention matmul plus a Triton
   kernel for group-wise quantisation; the authors state plainly that
   "dequantization can be computationally expensive" without them.
   SwarmLLM runs on candle with no custom CUDA/Triton. Implementing the
   storage format without the kernels buys the memory saving and pays
   full dequant cost on every attention call — on a decode step that is
   very likely a net loss. There is also a `residual length` buffer of
   unquantised recent tokens that partially offsets the compression,
   sized by a parameter the paper's summary doesn't pin down.
2. **Production converged on FP8, not 2-bit.** vLLM and TensorRT-LLM
   ship FP8 KV caches with quantised-domain execution via
   FlashAttention-3. That is the design with real deployment evidence
   behind it, and it also depends on kernel support we lack.

**Revised recommendation.** If KV memory becomes the binding constraint,
the tractable version for this codebase is **Q8_0 KV reusing
`inference/quant.rs`** — the group-32 + f16-scale path already shipped
for R136 Layer 0, already has a quality gate (`L∞ < 0.05`, `MAE < 0.01`
on representative hidden states), and is already exercised on the wire.
Roughly 2× KV reduction rather than KIVI's 2.6×, with no new dependency
and no new numerical risk. **Measure first**: instrument KV bytes vs
total VRAM on a real long-context run to confirm KV actually dominates
for our workloads before touching the attention path at all. On the
short-to-medium contexts this project's users actually run, model
weights dominate and shard windows / `gpu_layers = 0` are the cheaper
lever (see R146).

### Tier 3 — speculative / research-scope

#### G. RadixAttention-style cross-session prefix sharing — **cross-REQUEST sharing SHIPPED**; true radix-tree refinement open

**Status.** `inference/split/prefix_cache.rs` ships cross-REQUEST (not
cross-session) prefix-KV sharing. Block-boundary BLAKE3 hash chain
(`compute_block_hashes`) means two prompts sharing the first N blocks
get an instant prefix hit on the second request — including across
DIFFERENT users with the SAME system prompt. The big win from
RadixAttention (system-prompt dedup, multi-turn reuse) is captured.

**What remains open.** True radix-tree storage with reference-counted
KV-block ownership and COW on divergence. The current implementation
is a flat set of entries per model with LRU eviction; storage is
`O(entries × prompt_len × hidden × layers)` worst-case, so a
deployment with 1000s of distinct prompts pays for redundant prefix
storage that a radix tree would dedup at the block level.

**Why this gap is low-priority for SwarmLLM specifically.** Radix-tree
storage matters when one worker has many concurrent same-prefix
requests — vLLM / SGLang on a single GPU serving a public API. Our
P2P case rarely concentrates that many requests on a single worker
(load is sharded across the pipeline + spread across peers), so the
block-boundary flat cache already captures most of the realistic gain.

**Effort if pursued.** 4-8 weeks. Requires PagedAttention-style block
KV management which we don't have today; this is an architectural
change to `kv_cache.rs` and every attention call site.

Original analysis preserved below.

#### G. RadixAttention — original analysis
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

> _Note: the "shares prefixes per-session" framing in the original
> analysis was inaccurate even at the time it was written —
> `prefix_cache.rs` was already cross-request via block-boundary
> hashing. Status note above corrects this._

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

#### K. Communication-computation overlap inside a single forward — **SHIPPED R139** (Phase C + A-rev), one piece remains

**Status.** Closed in R139 across 4 commits via a research-driven
pivot from the original "worker streams row-tiled output during
matmul" framing to "daemon-side STREAM-chunked encrypt+send on a
single libp2p stream":

- **Phase C** (commit 11333f67) — ChaCha20-Poly1305 encrypt+decrypt
  offloaded from the NetworkManager event loop to tokio::spawn tasks
  via the new `NetworkCommand::SendEncodedTensor` continuation. Saves
  ~50–200µs of event-loop block time per forward (default config
  `enable_encryption=true`, `persistent_pipeline_stream=false`).
- **A-rev.1** (commit 4b5fc10c) — wire-format trailer 0x05 carrying
  `ChunkMeta { chunk_idx, total_chunks }` on `LayerForward`. Bound
  into the AAD via `build_layer_forward_aad` so reorder /
  wrong-total / cross-transfer-substitution attempts fail Poly1305
  before reaching the dispatch path. Backward compat: frames
  without the trailer decode to `chunk_meta=None` and run today's
  single-frame path.
- **A-rev.2/3** (commit 1d0a5d55) — receiver-side assembly state on
  `SharedState.pending_activation_chunks: DashMap<Uuid,
  ChunkAssemblyState>` (root-level — cross-cuts RR + stream paths,
  mirrors `pending_layer_results` precedent).
  `try_assemble_chunked_forward` accumulates chunks under entry-lock
  with `total_chunks` consistency check + sender-peer binding +
  duplicate-chunk_idx rejection. `chunk_layer_forward` helper splits
  a LayerForward at byte-offset boundaries. Config knobs:
  `streaming_chunked_send` (default false), `streaming_chunk_size_bytes`
  (default 262144 / 256 KiB — age STREAM + TokenWeave K=2-4 sweet
  spot), `streaming_min_activation_bytes` (default 65536 / 64 KiB
  floor), `streaming_chunk_assembly_ttl_secs` (default 30s).
- **A-rev.4** (commit e32c0a5d) — wired the chunked send into
  `pipeline/distributed.rs::forward_through_segments` persistent-
  stream path. All K chunks ride the same libp2p stream → QUIC
  preserves byte order → no receiver assembly race.

**Pivot rationale (research, 2026-05-19).** Original proposal
("worker emits row-tiled chunks during matmul") matches the SGLang
PD-disaggregation anti-pattern: SGLang explicitly rolled back per-
tile streaming because per-chunk fixed costs outran overlap gains.
No production inference system today (Triton decoupled, vLLM v1,
NVIDIA Dynamo/NIXL) streams forward-output tensors. Per-token text
streaming yes; tensor streaming no. The age STREAM AEAD
construction + single-libp2p-stream pattern is the cited best
practice (TokenWeave MLSys 2026, FlashOverlap 2025, age spec, Tink
Streaming AEAD, RFC 9771). See commit message bodies for full
citations.

**Phase B (multi-request token interleaving)** turned out to be
already shipped via the existing architecture: `router/distributed_exec.rs`
spawns one tokio task per concurrent request; `pipeline_stream`
keys streams by `(peer, request_id)` so concurrent requests fan
out across separate streams; `process_pool::batch_scheduler_loop`
auto-coalesces concurrent worker `forward()` calls into batched
IPC. R139 documented this and skipped Phase B.

**Remaining for full Tier 4K close-out** (small follow-ons):

1. **TTL sweep wired to HealthMonitor periodic tick** —
   **closed 2026-05-19** (commit ff2f7b4d). Wired
   `SharedState.sweep_stale_chunk_assemblies(ttl_secs)` into the
   existing 30s cleanup block in `src/health/monitor.rs` alongside
   the AllReduce/RingChunk cleanups. TTL sourced from
   `config.inference.streaming_chunk_assembly_ttl_secs` (default
   30s). Debug-level log when evictions occur.

2. **Chunked-send on RR fallback path** — **deferred 2026-05-19**
   pending WAN bench data. End-to-end call-graph trace confirmed
   the mechanism would work (sender: K `NetworkCommand::SendTensor`
   commands per forward; receiver: inspect `chunk_meta` BEFORE the
   decrypt spawn and return `None` from `handle_tensor_payload` for
   non-final chunks so `requests.rs:340-381` sends the Ack instead
   of storing the ResponseChannel; FINAL chunk's RR exchange holds
   the channel for the eventual LayerResult). True scope is ~50 LOC
   spread across `distributed.rs::forward_through_segments` (sender
   split + K-send loop), `tensors.rs::handle_tensor_payload`
   (encrypted + unencrypted arms, the latter currently has no
   assembly wiring), plus 3-4 receiver tests covering the
   out-of-order arrival cases (final chunk first, duplicate
   chunk_idx, total_chunks mismatch on second chunk). Reason for
   deferral: the microbench (item 3, closed) shows chunked is 3.3×
   slower in pure CPU terms on multi-chunk paths; the win comes
   from encrypt/decrypt + wire overlap which the persistent stream
   path already captures. On the RR path each chunk pays its own
   per-request overhead (separate libp2p substreams, separate
   encrypt context), so the overlap window is narrower than on
   stream. Without real WAN bench data confirming the RR win, we'd
   be adding ~50 LOC + branch surface for a code path that ships
   default-off. Re-open when WAN bench script (Tier 4K item 5
   below) demonstrates a measurable RR win.

3. **Microbench** in `examples/swarm_spec_bench.rs` —
   **closed 2026-05-19** (commit dd0a6b74). Added `bench_chunked_send`
   covering 32/64/256 KiB + 1/1.6 MiB activation sizes at the
   default 256 KiB chunk size. Reports mono encode+decode, full
   chunked split→encode×K→decode×K→assemble, split-only, and
   assembly-only timings. Mirrors production dispatch (skips
   assembly when K=1 so passthrough rows are honest). Measured
   3.3× CPU overhead at K=4/7 on this host (WSL2); WAN overlap
   win not captured here — needs a separate harness.

5. **WAN bench script** — needs two daemons on different networks
   (or VPN'd cloud regions). Compare `streaming_chunked_send=true`
   vs `false` on a workload that fits the prefill-class activation
   regime (1+ MiB activations). Win is "chunked completes earlier
   than monolithic because encrypt+send and recv+decrypt overlap";
   loss is "fixed-cost-per-chunk dominates on low-latency links".
   Crossover point is RTT-dependent, so test 5/25/100/200 ms RTT.
   Wire up via existing `examples/3node_inference_bench.sh` recipe
   with a `--chunked-send` flag toggle on each daemon's config.toml.
   Deferred — needs a real two-network test setup.

4. **Worker-side row-tiled output streaming** — the literal "true"
   form of Tier 4K. Requires worker IPC streaming protocol changes
   (multi-frame `WorkerMsg::ActivationChunk` matching today's
   per-token `WorkerMsg::Token` precedent), plus candle Tensor
   row-slicing in `SplitExecutor::forward_inner_impl`'s final-layer
   matmul. SGLang's evidence is that this loses on RDMA fabrics;
   re-evaluate when slow-WAN bench data justifies. Estimated 3–4
   weeks. Tracked in this entry but not actively scheduled.

**Win this delivered (default config, no flag flip).** Phase C ships
unconditionally: ~50–200µs/forward event-loop block savings on the
default RR encrypted path. Multiplied across concurrent decode
traffic this is the difference between smooth event-loop responsiveness
and observable jitter on libp2p ping / gossip / connection events.

**Win this delivered (flag-on, when both ends opt in).** Sender chunks
+ pipelined encrypt + receiver reassembly across a single libp2p
stream. On encrypt-dominated paths (1 MB activation,
ChaCha20-Poly1305 ~50–200µs each), per-forward saving is
~100–150µs. On wire-dominated paths (slow WAN <30 Mbps) the
saving comes from receiver-side decrypt+forward overlap with
remaining sender-side send. LAN/loopback: no measurable win
(activation send is already sub-millisecond).

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

### All-layers-active bench (L0+L1+L2+L3 simultaneously)

Captured 2026-05-17 with all four R136 layers explicitly enabled:
`activation_compression=true`, `ngram_lookup_enabled=true`,
`hedge_enabled=true`, `prefetch_enabled=true`. Single-segment
routing (B+C each hold full model).

```
Workload    tok/s   L1 hit-rate   L2 fires   L3 fires
code        4.0     25.5%         0          0
summary     3.2     23.1%         0          0
chat        5.2     9.6-10%       0          0
```

L2 hedge: 0 actual fires on loopback (30 dry-run decisions
recorded — would-have-fired counter for tuning). Expected: loopback
RTT is too consistent to exceed `1.5 × p99` threshold. On flaky WAN
the dispatch would activate.

L3 prefetch: 0 dispatches. Expected: bench requests don't carry
session_id, so the orchestrator never sees `observe_user_turn`
data. On multi-turn chat workloads with stable session IDs, the
candidate-emission would fire.

**Throughput**: within noise of the L1-only baseline (4.0/3.2/5.2 vs
4.3/4.0/4.7 — the variance dominates within 3 trials per workload).
Notably no regression from having L2/L3 enabled even when they
don't fire — the fast-path optimisations in each dispatch wrapper
correctly degenerate to a straight call.

**Correctness validated**: All requests completed with expected
token counts. KV-truncate fix from earlier review prevents
silent output corruption on partial-accept rounds.

### Layer 1 multi-segment (sharded production case) measured

Captured 2026-05-17 after extending `try_ngram_only_distributed` to
multi-segment pipelines via the extracted `forward_verify_through_segments`
helper. Cluster: A=manifest, B=shard_000, C=shard_001 (true sharded
2-segment pipeline). L1 fires on every request — verified in logs
via `SWARM-SPEC L1 ngram-only: complete`.

```
Workload    tok/s   L1 hit-rate
code        3.7     35.7%
summary     3.3     8-77% (high variance across trials — cache effect)
chat        4.7     17.2%
```

vs the single-segment-routing case (B/C each holding full model
where pipeline = 1 hop): L1 active gave 4.3/4.0/4.5 tok/s.
Multi-segment is slightly slower per token because each verify
round is 2 hops (B→C) vs 1, but **now works for the real
production sharded scenario**.

**Bug found and fixed during integration**: the shape-match check
in DSD's `forward_verify_through_segments` fired incorrectly on
segment 0 transitions (input = token IDs 8 bytes × N, output =
hidden state hidden_dim × 2-4 × N — naturally different total
bytes). Was latent in DSD (multi-segment + draft model is an
uncommon combination, never exercised in prod). Fixed by gating
the check on `idx >= 1` (intermediate-to-intermediate, where
shape preservation IS expected).

### Layer 1 (n-gram-only spec, draft-free) measured on real inference

Captured 2026-05-17 after shipping `try_ngram_only_distributed`
(the no-draft-model n-gram cascade). Cluster: A=manifest only,
B+C=full model. Single-segment routing A→B with the n-gram-only
spec path active. Same workload set as the Q8_0 A/B above.

```
Workload       Baseline (remote_generate)   L1 active   Gain   L1 hit-rate
code (60 tok)  4.0 tok/s                    4.3         +7%    23%
summary (50)   2.75 tok/s                   4.0        +45%    77%
chat (80)      4.5 tok/s                    ~4.5       mixed  18-23%
```

The summarisation 45% win on a 77% n-gram hit-rate is the strongest
real-inference validation of the cascade design to date. The
synthetic bench predicted 96% hit rate on RAG-shaped workloads; the
real TinyLlama summary workload achieved 77%, consistent with the
prediction's order of magnitude.

The code-completion result is more modest because TinyLlama 1.1B
has limited overlap on the short fibonacci prompt — synthetic
benches used 200-token contrived high-repeat prompts; real prompts
rarely have that density. The hit-rate scales with prompt-overlap;
a longer multi-file code completion would hit higher.

Free-form chat is correctly low-hit (~20%) — the cascade's
design-intent fallthrough path. Falls back to single-token
forwards, throughput within noise of the remote_generate baseline.

Per-request hit-rate metrics now visible in tracing logs:
`SWARM-SPEC L1 ngram-only: complete generated_tokens=N
ngram_rounds=X fallback_rounds=Y ngram_hit_rate="Z%"`

**Dispatch order change.** L1 now precedes `try_remote_generate_fastpath`
in `pipeline/distributed.rs::execute_distributed`. The reasoning:
when n-gram hits, accepting multiple tokens per round beats
remote_generate's one-token-per-RTT throughput. When n-gram misses,
single-token fallback within L1 has comparable per-token cost to
remote_generate. So L1 is dominant on hit, neutral on miss.

### Distributed-pipeline measurement (run completed)

Forced 2-segment pipeline (A=manifest only, B=shard_000,
C=shard_001, all with `auto_manage.enabled = false` to preserve
sharded state past startup). Every request from A's API forces
B → C activation forwards. Same workload set as the single-node
baseline above.

```
                  Q8_0 ON (loopback)   Q8_0 OFF (loopback)   Δ
code-completion   4.0 tok/s            3.5 tok/s             ON +14%
summarisation     2.75 tok/s           2.9 tok/s             OFF +5%
free-form chat    4.5 tok/s            4.8 tok/s             OFF +7%
```

(Medians of full-length trials. Several trials in both runs hit
early EOS at 6 tokens — TinyLlama 1.1B Q4_K_M chat-template
artefact, not a SWARM-SPEC issue; those trials excluded from
the median.)

**Honest finding: Q8_0 doesn't win on loopback distributed pipelines.**
The Q8_0 encode/decode CPU cost (~17 µs per 4096-dim hidden state)
rivals the wire-time saved on sub-ms loopback hops. The synthetic
bench prediction (1.5–1.7× on bandwidth-bound hops) only applies
when the wire IS bandwidth-bound — i.e., 10–100ms WAN RTT, not
loopback.

**Practical implication for the default-on flip.** Q8_0 default-ON
is still the right call for WAN deployments (the real SwarmLLM
target — geo-distributed P2P nodes). But operators running
loopback / single-host clusters for testing or LAN-isolated pools
should see a small regression. The activation_compression flag is
already exposable per-node in config.toml so operators can override.

**Future bench work needed.** Real-WAN measurement requires a
multi-host setup or artificial latency injection (e.g., Linux
`tc qdisc add netem delay 50ms`) to simulate the WAN case where
bandwidth IS the bottleneck. Without that, the loopback A/B is a
lower bound on the WAN gain — not the WAN gain itself.

### Original sharded test attempt (auto-manage on)

The first attempt placed sharded files but left auto-manage on by
default; the daemon auto-downloaded missing shards within ~12s,
defeating the sharded test. The fix (per-node config with
`auto_manage.enabled = false`) is now reflected in the example
scripts.

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

### Path to end-to-end measurement — **UPDATE post-R136**

The original entry below claimed Layer 1 needed a draft model,
Layer 2 needed a wire-format change, and Layer 3 was observation-
only. **All three are now obsolete**:

- **Layer 1**: SHIPPED draft-free via
  `pipeline/ngram_only_spec.rs::try_ngram_only_distributed` —
  uses standalone tokenizer cache (lazy-loaded from
  `gguf_header.bin`) so no draft model is required.
  Real-inference measured: summary **+45% on 77% hit-rate**
  single-segment, multi-segment sharded validated. See
  `## R136 local 3-node benchmark — measured results` above.

- **Layer 2**: SHIPPED true duplicate-dispatch WITHOUT a wire-format
  change — uses a fresh `Uuid` for the hedge so the
  `pending_layer_results` map doesn't collide with the primary.
  See `pipeline/hedge_dispatch.rs::forward_verify_with_hedge`.
  Loopback bench shows 0 actual fires (RTT too consistent to
  exceed 1.5×p99) — wire correct, waiting for WAN deployment to
  measure win. Multi-segment hedging remains deferred (would
  need full alternative-pipeline assembly).

- **Layer 3**: SHIPPED observability-complete dispatch —
  `should_prefetch` decision + `record_dispatch` + ActivityEvent
  emit at response-completion site in `router/mod.rs`. The
  observation side is complete; the K-layer activation prefetch
  COMPUTE itself remains deferred because the win is workload-
  dependent (small models on fast hardware have negligible
  prefill — the savings are in the noise; large models on slow
  hardware would see meaningful TTFT cut).

Original (now-obsolete) text preserved below.

### Path to end-to-end measurement (original, pre-R136 implementation)

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

## R136: SWARM-SPEC — proposal (HISTORICAL, shipped)

> **STATUS:** This was the original design proposal that opened
> R136. **All 4 layers are now SHIPPED with true dispatch.** See
> `## R136 local 3-node benchmark — measured results` above for the
> actual numbers and `## Inference performance — research backlog
> (R135 brief)` for what specifically remains deferred. The
> "decision matrix — for user" with Options A-E below is no longer
> active — the user authorized Option D (maximalist) over multiple
> autonomous-loop iterations and we shipped that.
>
> Original proposal preserved unchanged below for historical
> reference (how we decided what to build).

## R136: SWARM-SPEC — proposal for a state-of-the-art inference acceleration system (original proposal)

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

## CI / build infra

### Further cut the Windows CUDA release build (~30 min) — PRIMARY FIX APPLIED 2026-07-25

**Primary fix shipped (commit `1e65a9c7`):** `cache-warm.yml` now warms the
**Windows-GPU** rust-cache shared-key too, not just Linux CUDA + Linux CPU. That
was the actual gap: the Windows-GPU release rebuilt llama.cpp-Vulkan +
candle-kernels cold *because its cache was never warmed on `main`* — a tag-scoped
release cache can't be restored by a later tag (GitHub cache ref-scoping), so
without a `main`-warmed key every release started cold. The new cell mirrors
release.yml's Windows setup byte-for-byte (CUDA 12.4, Vulkan SDK, MSVC, Ninja,
nvcc /MD, build env) so the key matches.

This corrects the earlier assumption in this entry that "`rust-cache` doesn't
cache CMake object files." It does — `Swatinem/rust-cache` caches `target/`
including dependency **build-script output** (`target/release/build/
llama-cpp-sys-2-*/out/`, the compiled llama.cpp objects). The proof is the Linux
CUDA job, which *also* builds llama.cpp via CMake and dropped 59m→~10m purely
from the warm cache. The Windows problem was never "CMake isn't cacheable" — it
was "Windows GPU was never in `cache-warm.yml`."

**Measure the next Windows-GPU release** against the Linux ~10-min figure. If a
large CMake-rebuild cost remains after the warm cache lands (e.g. rust-cache's
`target/` cleanup evicts the llama.cpp `out/` dir on Windows for some reason), the
remaining levers, in order:

- **sccache/ccache around the CMake build (heavier).** Wire
  `-DCMAKE_C_COMPILER_LAUNCHER=sccache -DCMAKE_CXX_COMPILER_LAUNCHER=sccache`
  (and `RUSTC_WRAPPER=sccache` / `NVCC` support) via the GHA sccache backend.
  sccache caches nvcc + MSVC compilation even on a cold rust-cache miss. Notably
  finicky on Windows+MSVC+CUDA — only worth it if the warm-cache route leaves a
  real gap.
- **Cache the CUDA toolkit install + redist download**, keyed on the CUDA version.
- **Larger GitHub-hosted runner** (more vCPUs) — weigh cost vs. saving.

**Keep any new Windows cache keys byte-identical to `cache-warm.yml`** and
produced on `main` — a release runs on a tag push and per GitHub cache scoping
"cannot restore caches created for different tag names", so a tag-scoped key
writes a cache no later release can read (the exact bug R148 fixed on Linux).

### CUDA release build: pin `CMAKE_CUDA_ARCHITECTURES`? (R147, 2026-07-22)

**Context.** The Linux CUDA release job is the release long pole (~1 h). Two
causes were investigated:

1. *Cache never restored across tags* — **fixed** in R147. GitHub scopes
   Actions caches by ref: a run "cannot restore caches created for different
   tag names", though tags CAN read the default branch's caches. `release.yml`
   only runs on tag pushes, so it wrote a cache scoped to its own tag that no
   later release could read — a measured 1268 MB CUDA dependency cache sat
   orphaned on `refs/tags/v0.3.4-alpha`. Fixed by switching the release cache
   step from `key:` to `shared-key:` (which excludes the job id) and adding
   `.github/workflows/cache-warm.yml`, which runs the same builds on `main` so
   the cache lands where tag runs can restore it.

1b. *Cache-warm ran on the wrong runner image* — **fixed R148**
   (2026-07-22). The `cache-warm.yml` added above hardcoded
   `runs-on: ubuntu-latest` for every matrix entry, while `release.yml` pins
   its CUDA job to `ubuntu-22.04`. Two consequences: the job failed outright
   (the `ubuntu2204` CUDA repo's `nsight-systems` needs `libtinfo5`, which is
   not installable on 24.04), and even had it succeeded the runner image is
   part of the `rust-cache` key, so a 24.04 cache could never be restored by a
   22.04 release build. The matrix now carries a per-entry `runner:` mirroring
   `release.yml`. Invariant to remember: cache warming must match the release
   job on runner image, profile, features and env, or it silently warms a
   cache nothing reads.

2. *llama.cpp compiles a wide CUDA arch spread* — **DONE (R150, 2026-07-23).**
   `llama-cpp-sys-2`'s `build.rs` forwards any `CMAKE_*` env to CMake
   (build.rs:531-534, verified), so the `cuda` (Linux) build pins
   `CMAKE_CUDA_ARCHITECTURES: "61-real;75-real;80-real;86-real;89-real;90-real;120-real;120-virtual"`
   — native SASS Pascal→**Blackwell** + compute_120 PTX for a future
   post-Blackwell card. Because that build.rs does NOT `rerun-if-env-changed`
   on it, the rust-cache `shared-key` carries a `-gpuarchN` discriminator; bump
   it whenever the arch list changes or a warm cache will silently ship the old
   set (currently `-gpuarch3`).
   - **Phase 1** (commit 87222c75, main, CI-green): arch pin without `120`, on
     CUDA 12.4.
   - **Phase 2** (PR #17 → f9356e7c, merged): Linux toolkit **12.4 → 12.8**
     (first nvcc that knows sm_120; `cudarc 0.19` compiled clean against 12.8 in
     the PR's feature-check — candle #3249 is a CUDA *13.0* rejection, not 12.8)
     + native `120-real`. **Windows deliberately stayed on 12.4** — its only CUDA
     consumer is candle (PTX/driver-JIT, so Blackwell already works) and its
     llama.cpp is Vulkan, so 12.8 buys nothing there and it dodges the
     `Jimver@v0.2.19`-may-not-know-`12.8.0` risk.
   - **Remaining validation** (not blockers, but not yet green): the full native
     `sm_120` llama.cpp build is only exercised by `main`'s **cache-warm**
     (gpuarch3/12.8) post-merge — if the bundled `llama-cpp-2 0.1` rejects arch
     `120`, revert just the `120` addition (keep 12.8 + the rest). And real
     runtime correctness on RTX 50 hardware needs a Discord tester before any
     *stable* (non-alpha) tag — the maintainer's box is sm_86.

**candle side — Phase 1 DONE.** `CUDA_COMPUTE_CAP` lowered `80 → 75`.
candle-kernels emits PTX, so this is a *floor*: the driver JIT-compiles it to
any GPU ≥ sm_75, so one binary now covers **RTX 20-series / GTX 16 → Blackwell**
(the 80 floor silently excluded Turing/Pascal). candle-kernels disables its
bf16-WMMA kernels below sm_80; harmless for SwarmLLM's quantized GGUF path
(quantized kernels, not bf16 WMMA).

### candle bf16-WMMA recovery via a compute_80 variant (R150, 2026-07-23)
**Context.** The single `CUDA_COMPUTE_CAP=75` binary above disables candle's
bf16-WMMA (tensor-core) kernels for *everyone*, including Ampere+ cards that
have the hardware. For SwarmLLM's quantized workload that's expected to be
negligible, but a bf16-heavy path would lose throughput on RTX 30/40/50.

**What it would take.** A second CUDA build variant at `CUDA_COMPUTE_CAP=80`
(bf16 WMMA on) alongside the `75` build, plus **runtime compute-cap selection**:
today `update.rs::build_variant_suffix` keys purely on `cfg!(feature=…)`
(`-cuda`/`-gpu`) and `bin/launcher.rs` detects only GPU *presence*
(`nvcuda.dll` / `nvidia-smi`), neither reads the compute capability. So a
Turing-vs-Ampere split needs new `nvidia-smi --query-gpu=compute_cap` detection
in the launcher + a compute-cap-aware asset picker in the updater, plus a
doubled CUDA build matrix and its own warm cache (watch the 10 GB repo-cache
ceiling — R148).

**Why deferred.** Needs a real GPU benchmark to prove the bf16 loss is material
for our quantized path before doubling the build matrix + shipping untestable
(no local GPU) runtime detection. Measure first.

**Validation gap (both phases).** The maintainer's box is sm_86; arch changes
ship untested on other gens. Mitigations: PTX/JIT covers newer-than-tested, CI
compile-checks catch build breaks (not runtime), and Discord testers with
RTX 20/40/50 should confirm before a *stable* (non-alpha) tag.

### Windows-GPU auto-update carries stale CUDA redist DLLs (v0.3.20 asset audit, 2026-07-25)

**Context.** The Windows GPU **archive** bundles NVIDIA redist DLLs
(`cudart64_*`, `cublas64_*`, `cublasLt64_*`, `curand64_*`, `nvrtc64_*`,
`nvrtc-builtins64_*`) next to `swarmllm.exe`, because cudarc resolves them with
`LoadLibraryW` at runtime and end users are not expected to have the CUDA
Toolkit installed. That is why `swarmllm-windows-x86_64-gpu.zip` is ~490MB
while the bare `swarmllm-windows-x86_64-gpu.exe` is ~119MB.

**The gap.** `update.rs` downloads the *bare* variant asset and does an atomic
single-file swap. The DLLs sitting beside the binary are whatever the user's
original zip install shipped, and are never refreshed by an auto-update.

- **Safe within a CUDA major.** `cudart64_12.dll` serves all 12.x and Windows
  is pinned to 12.4 (`Jimver/cuda-toolkit` `cuda: '12.4.0'`), so today every
  auto-updated Windows GPU node keeps working.
- **Silently fatal across a major.** A 12→13 bump produces an exe that
  `LoadLibraryW`s `cudart64_13.dll` into a directory holding only
  `cudart64_12.dll`. Every auto-updated Windows GPU node then fails at CUDA
  init, with no CPU fallback and no obvious cause for the user.

**Linux CUDA is not affected** — its staging dir is binary + config + licenses
only, and cudarc dlopens `libcuda.so.1` from the user's *driver*, so the 933MB
bare binary is self-contained. This is Windows-specific precisely because we
ship the toolkit runtime rather than relying on one being installed.

**Options when the Windows toolkit pin is next touched** (do this *before*
bumping it, not after):
1. Teach `update.rs` to fetch the `.zip` for the `-gpu` variant and unpack the
   DLLs alongside the exe. Most correct; makes the GPU update path a
   multi-file apply, so the atomic-swap + rollback logic needs extending.
2. Ship the CUDA major in the asset name (e.g. `-gpu-cu12`) and refuse to
   auto-update across a major, directing those users to a fresh installer.
   Cheapest, and reuses the existing "no exact variant match → skip" behaviour.
3. Statically link / bundle the runtime into the exe, as the Linux build
   effectively does. Largest binary, simplest update story.

Cross-referenced from gotcha #162. Note the Windows/Linux CUDA versions are
already deliberately split (12.4 vs 12.8) with a "bump both together" comment in
`release.yml` — that comment should point here too when either moves.

### Per-request holder blacklist on retry (networking audit, 2026-07-25)

**Context.** `router::is_transient_remote_failure` triggers exactly one retry
with a fresh pipeline assembly. The retry avoids a *dead* peer only as a side
effect: a peer that dropped its connection leaves `connected_node_ids` and so
stops being a candidate.

**The gap.** Nothing records "this holder just failed *this* request", so a peer
that fails **without** disconnecting can be re-picked immediately. Two cases:

- Pre-existing: a connected peer that accepts the request and then stalls (GPU
  wedged, worker crash-looping) stays in `connected_node_ids` throughout.
- Widened by the §4 Phase 1 relay tier: a relay-reachable holder was never in
  `connected_node_ids` at all, so a stale-but-unexpired relay route can see the
  same holder selected on the retry.

Neither is a correctness bug — the request still terminates, and re-picking the
sole holder of a shard is the right call. It costs a wasted timeout when a
better alternative existed.

**Shape of the fix.** Thread a small `HashSet<NodeId>` of holders that already
failed this request through `execute_request` into `gather_candidates`, and skip
them. Bounded by the single retry, so it needs no eviction policy. Worth doing
alongside any move to more than one retry, where the current behaviour would
degrade from "one wasted timeout" to "N wasted timeouts".

### Does contribution-weighted credit recreate the hierarchy it exists to avoid? (raised 2026-07-25)

**Not a bug — an open design question**, raised by an external contributor and
worth a deliberate answer rather than a default.

**The question.** SwarmLLM's stated point is keeping inference out of a handful
of corporate hands, and the fixes that matter most are the ones lowering the
barrier to participate — modest hardware, home NAT, mobile. But credit is earned
by contribution (VRAM, uptime, bandwidth, shards served). If contribution sets
your access rate, does that structurally favour whoever already owns the best
hardware — the same imbalance, denominated in tokens instead of dollars?

**What the design already does, which is half an answer.** Access is *tiered*,
never gated:

- `credit::priority::calculate_tier` — Bronze is zero **or negative** balance.
  There is no "blocked" tier.
- `max_concurrent_for_tier` — Bronze gets `(base_max / 4).max(1)`. The `.max(1)`
  is the floor: a node with a permanently negative balance still gets at least
  one concurrent request, forever.
- Project rule (`.claude/rules/completeness.md`): *"Credit errors: degrade
  priority tier, never block."*
- Negative balances decay hourly back toward zero (R149), so a deficit is not
  permanent.

So a zero-contribution node is slower, not excluded. That is a meaningful
difference from pay-to-play, and it should be stated explicitly somewhere
user-facing — right now it is an emergent property of three separate mechanisms
rather than a documented guarantee.

**What is genuinely unresolved.** Under contention the ratio still scales with
hardware: Platinum gets `base_max * 2`, Bronze `base_max / 4` — an 8x spread. On
a busy network that is the difference between usable and painful, and the people
on the wrong end are exactly the modest-hardware users the project is for.

**Directions worth weighing** (none chosen):

1. **Document the floor as a guarantee.** Cheapest, and possibly sufficient:
   state that participation is never required for access, only for speed.
2. **Contribution measured as effort, not capacity.** Credit uptime and
   availability rather than throughput, so a Raspberry Pi seeding shards
   reliably earns comparably to a 4090 serving occasionally. Changes what the
   system rewards without changing that it rewards.
3. **Raise the Bronze floor / compress the spread.** An 8x range is a choice, not
   a constant.
4. **Demand-weighted floor.** Guarantee a share of *idle* capacity to low-tier
   nodes, so the floor rises when the network is quiet and only tightens under
   real contention.

**Why it matters beyond fairness:** a network that feels unusable to newcomers
loses them, and this is a network whose value grows with participation. The
incentive design and the adoption goal are the same problem.

**Evidence from the closest comparable system (researched 2026-07-26).**
[Petals](https://github.com/bigscience-workshop/petals) is the nearest analogue
— BitTorrent-style distributed LLM inference across volunteer consumer GPUs. It
has **no incentive layer at all**, and that turns out to be the more instructive
data point:

- The [Petals paper](https://arxiv.org/pdf/2209.01188) names the absence as a
  design problem in its own right: without incentives there is "an imbalance
  between supply (peers who dedicate GPUs to serve model layers) and demand
  (peers using the servers)".
- Its proposed remedy is almost exactly what SwarmLLM already implements —
  *"peers running servers would earn special points, which can be spent on
  high-priority inference"*. Note **priority**, not access. That is the same
  distinction our tiering makes, and it is the thing that separates this from
  pay-to-play.
- Petals peaked around 800 contributor nodes and did not sustain it. Pure
  volunteerism is not the safe default it appears to be; it fails on the supply
  side, which hurts exactly the users who own no capable hardware.

So the honest framing for the discussion is not "contribution-weighting versus
fairness". It is: **an unincentivised network stops having capacity to share,
and the users who lose most are the ones with the least hardware.** The design
question is where the floor sits and how the spread is shaped, not whether to
reward contribution at all.

That reframes the four options above: (1) documenting the floor becomes more
valuable, not less, because the floor *is* the fairness guarantee; and (4) the
demand-weighted floor is the most interesting, since it gives low-tier nodes
more when the network is idle — precisely when generosity costs nothing.

### Node.js-20 GitHub Actions deprecation (R145 sweep, 2026-07-21)
**Context.** GitHub is deprecating the Node.js 20 runtime on Actions runners; three pinned actions in `.github/workflows/release.yml` still target Node 20 and are currently *force-upgraded* to Node 24 (a warning annotation, not a failure — the v0.3.3-alpha release built and published fine):

- `ilammy/msvc-dev-cmd@v1` — **no fix available**: latest is v1.13.0 (2024-01-01); the maintainer never shipped a Node-24 release. Bumping the pin won't help. Either wait for upstream or replace the MSVC-setup step.
- `jakoch/install-vulkan-sdk-action@v1` — a newer v1.6.0 (2026-06-26) exists, but **gotcha #132** records this action as broken on Linux; a blind bump is risky and needs CI validation on all platforms.
- `Jimver/cuda-toolkit@v0.2.19` — newer v0.2.35 (2026-03-29) exists (keep the `cuda: '12.4.0'` pin — that's a candle/RTX-50 constraint, not the action version). Only validatable through the ~56-min CUDA CI job.

**Why deferred.** All three work today; GitHub has not yet removed the Node-20 runtime. Two of the three are either unfixable-by-bump or historically fragile, and validation is slow/expensive (CUDA build is the release long-pole). Revisit when GitHub sets a hard Node-20 removal date, and validate any bump on a throwaway tag before a real release.

**Also noted this sweep (no action needed):** `hickory-proto 0.25.2` carries RUSTSEC-2026-0118 + -0119 (transitive via libp2p 0.56 DNS/mDNS). CI ignores both — no upgrade path until libp2p bumps its hickory deps. `anyhow`/`memmap2` unsound advisories were cleared by patch bumps in commit `4b4d5307`.

---

## Inference engine

### True per-layer GPU/CPU hybrid offload (R146, 2026-07-22)
**Context.** `inference.gpu_layers` now reaches the shard/worker path and is honoured for the two outcomes the split engine can actually express: `0` = CPU only, `-1`/`>0` = GPU. What it still cannot do is llama.cpp-style *partial* offload — "put 8 of these 22 layers on the GPU and the rest on the CPU". A positive value below the worker's layer count logs a warning rather than silently pretending.

**What it would take.** `SplitModel` holds one `Device` for the whole model (`split/loader/mod.rs`, `loader/shards.rs`). Real hybrid placement needs:
- per-layer `Device` on the layer structs, chosen at load time from a boundary index;
- a `to_device` transition in the forward loop at the boundary — one PCIe copy per boundary crossing per token, cheap if the split stays contiguous;
- KV-cache blocks allocated on the same device as their layer (`split/kv_cache.rs` currently takes the model device);
- the same treatment across every arch path (`model_arch.rs` dispatch, `layers/qwen35.rs`, MoE/MLA, gemma2) — this is where the real risk lives, since a missed path silently mixes devices and fails at runtime, not at compile time;
- prefix-cache snapshot serialization is device-tagged, so cross-node prefix reuse would need a device-agnostic representation.

**Why deferred.** Not what the reporting bug needed — the user's ask was "reducing gpu_layers should reduce VRAM", and `gpu_layers = 0` now genuinely does that, as does the automatic CPU pin after a GPU OOM. Hybrid offload is a performance feature on top. It touches the latency-critical forward path across every architecture, and validating it requires a CUDA build (~56 min) plus per-arch GPU testing. Worth doing when there's a concrete workload that needs a model slightly too big for available VRAM and where CPU-only is too slow to be acceptable.

**Interim workarounds:** shard windows (`ModelProcessPool::restart_with_window`) bound VRAM by loading fewer shards; `gpu_layers = 0` forces CPU; a GPU OOM auto-pins the model to CPU for the rest of the run.

---

## Adaptive shard sizing from node capability (2026-07-22)

**Context.** `model.shard_size_mb` is a single global constant (default 512,
min 64) applied when a node first probes a model:
`shard_count = ceil(total_size / shard_size_mb)` in
`model/huggingface/probe.rs`. Every model on every node in the swarm is cut the
same way regardless of who will end up hosting it.

That is a poor fit for a swarm of unlike machines. A 24 GB workstation and a
4 GB laptop get identical granularity, so the small node either cannot take a
shard at all or takes one sized for hardware it does not have. Finer shards
would let weak nodes contribute a slice they can actually hold; coarser shards
would cut bookkeeping for strong ones.

**What it would take.**
- A capability signal already exists — `swarm_capacity` (R110) aggregates VRAM
  and node counts, and `auto_manage/vram.rs` computes per-node budgets. Sizing
  would consume those rather than adding new telemetry.
- The hard constraint is that **shard layout is global, not local**. The layout
  travels with the manifest and is fixed by whichever node probes the model
  first; two nodes that disagree produce incompatible manifests for identical
  weights. So adaptive sizing cannot be a per-node decision — it needs either a
  negotiated layout at model-adoption time, or a manifest that expresses
  multiple granularities (e.g. sub-shard ranges a weak node can hold a subset
  of) so nodes can choose within one agreed layout.
- Splits are layer-aligned, so the achievable sizes are quantised by layer
  size. A 32-layer model cannot be cut into more than 32 pieces, and uneven
  layer sizes mean the target is approximate.

**Why deferred.** The second point makes this a protocol design problem rather
than a tuning knob, and getting it wrong splits the swarm into groups that
cannot share shards of the same model. It also interacts with a gap worth
fixing first: nothing in `auto_manage/scoring.rs` prefers *contiguous* shards,
so a node can hold 0, 1, 4, 5, which
`shard_layout.rs::available_layer_ranges_from_manifest` turns into two segments
and an extra hop. Finer shards multiply that effect. Contiguity-aware
acquisition is the cheaper win and probably a prerequisite — adaptive sizing on
top of scattering placement would make pipelines deeper, not better.

**Interim.** `shard_size_mb` is settable per node for deliberate splits; see
`docs/REFERENCE_MODELS.md` for when lowering it helps and when it backfires.

---

## Separate LAN and public peer caches (2026-07-22)

**Context.** One `peer_cache` tree holds every remembered address. R148 split
the questions "is this worth *storing*" (`filter_storable`) from "is this worth
*dialling from here*" (`filter_dialable`), the latter deciding from whether this
node has a private address of its own. That fixed the public anchor retrying a
home user's LAN, docker-bridge and libvirt addresses forever, without costing
LAN reconnection for home nodes or pools.

**What separate trees would add.** The split above keeps everything and filters
on read, so nothing is lost — a laptop moving between a home network and a
hotspot retains both sets. Distinct trees would additionally allow: per-network
eviction (a LAN cache from a network you have left is dead weight the 200-entry
cap still counts), keying LAN entries by which network they belong to so a
laptop with two homes does not mix them, and different retention policies (LAN
addresses are stable for years, public ones churn with DHCP).

**Why deferred.** The read-time filter already fixes the reported problem, and
splitting storage does not remove the need for the same "where am I now?"
judgement — it relocates it. The extra structure only pays off for a device that
moves between several networks, which is not the deployment causing trouble
today. Revisit if laptops roaming between networks turn out to reconnect poorly.

---

## Distribution & networking (external report, 2026-07-23)

Two proposals from the raw-pc / raw-proxamd5 external user. Neither is a bug;
both are researched below with a recommendation. No code was written for either
this round — the user asked only for evaluation.

### Pear (Holepunch) as an opt-in P2P over-the-air distribution channel

**Context.** The user's day-to-day friction wasn't SwarmLLM itself — it was
*rebuilding from source* every time a fix landed (CUDA toolkit mismatches,
missing `libclang-dev`/`cmake`, stale cargo caches). Note the asymmetry: raw-pc
ran the **prebuilt CUDA binary** with no trouble; only raw-proxamd5 compiled,
because Debian 12's older glibc rejected the prebuilt. So the reported pain is a
**binary-portability** problem, not a distribution-mechanism gap.

**The proposal.** Wrap the *unmodified* `swarmllm` binary in a thin
[`pear-runtime`](https://github.com/holepunchto/pear-runtime) app (npm v1.1.1,
Mar 2026 — the embeddable Bare/JS runtime with P2P OTA + `bare-subprocess`). The
wrapper would (1) check whether the local daemon's HTTP API is reachable, (2)
spawn it via `bare-subprocess` if not, (3) otherwise get out of the way. It only
ever talks to `localhost:8800` — it never touches libp2p/QUIC/DHT. The payoff is
one-click cross-platform install + decentralized OTA updates with **no
infrastructure on our end** (`pear stage` a build, connected installs pick it up
via Pear's own updater). The user validated a working prototype (spawn +
health-check against a real daemon, clean single PID). Caveat they hit: the CLI's
`pear run` dev loop was removed in v3.0.0; the embeddable `pear-runtime` module is
the intended path now.

**Relation to what already exists.** SwarmLLM already ships prebuilt binaries via
GitHub Releases with a SHA256-verified, atomic-apply auto-updater
(`src/update.rs`, `UpdateChecker`; opt-in `[update] auto_update`). Pear would be a
*second, opt-in* surface, never a replacement.

**Recommendation.** Viable and low-risk *as an opt-in channel* — the wrapper's
blast radius is tiny (localhost only). But it adds a JS/Bare app + a Pear staging
key to maintain, and it does **not** address the actual reported friction, which
is that the prebuilt binary won't run on an older glibc. **Do the
portable-binary fix first.** _(Done, 2026-07-23: the Linux CPU release binary is
now built on `ubuntu-22.04` / glibc 2.35 instead of `ubuntu-latest` / glibc 2.39,
so it runs on Debian 12 (2.36) and other older-baseline distros without
compiling — `release.yml` + `cache-warm.yml`, mirrored per the runner-image cache
invariant. The CUDA binary already pinned 22.04.)_ Reconsider the Pear wrapper
only if there's demand for a genuinely zero-toolchain, self-updating install
beyond what GitHub Releases + `UpdateChecker` already give.

### peeroxide / Hyperswarm as an additional libp2p transport

**Context.** For NAT situations our AutoNAT-v2 / relay / DCUtR / UPnP stack (R143)
can't reach, the user looked at wrapping
[`peeroxide`](https://github.com/Rightbracket/peeroxide) — a pure-Rust
implementation of the Hyperswarm stack (HyperDHT + UDX with BBR congestion
control), wire-compatible with the Node.js Hyperswarm network,
`#![forbid(unsafe_code)]` on the network/crypto core, with cross-language interop
tests. Their estimate: 1–2 weeks for a basic wrapper, 3–4 for something to trust
with real traffic. They explicitly framed it as "your call entirely… not
something we're asking for."

**Key technical caveat.** peeroxide is **not** a `libp2p::Transport` — it's an
*independent* P2P stack with its own DHT and its own peer identity. "Wrap it as an
additional transport" understates the work: a libp2p `Transport` yields
`AsyncRead + AsyncWrite` streams keyed by libp2p `PeerId`, whereas peeroxide
speaks the Hyperswarm/UDX handshake and keys on Hyperswarm keypairs. Bridging
means either a bespoke shim that reconciles two identity/DHT namespaces, or
running peeroxide side-by-side as a second discovery+dial path and mapping its
peers back onto our `NodeId` — both materially more than "implement one trait."

**Maturity/trust caveat.** ~3 months old, essentially one maintainer, no external
audit. That's a heavy dependency to place on the critical path of a
security-sensitive P2P *inference* network, where a transport bug is an
activation-exfiltration or DoS vector.

**Recommendation.** **Defer.** Our existing NAT-traversal stack already covers the
common cases with relay as the CGNAT fallback (R143). The marginal NAT coverage
peeroxide might add does not, today, justify taking on a young single-maintainer
crypto/transport dependency plus a non-trivial identity-bridging effort. Revisit
only with (a) concrete data on NAT scenarios our relay path genuinely cannot
reach, and (b) peeroxide reaching more maintainers / a stable release / an
external audit.

## Observability: routing, performance and per-node attribution (2026-07-26) — SHIPPED

> **Implemented 2026-07-26** across `inference/trace.rs` (the single
> `RequestTrace`), response headers + `Server-Timing`, the `DIAG: request
> complete` line, `GET /api/admin/{diagnostics,performance}`, OTel-named
> Prometheus histograms, serving-side counters, the dashboard chat route line
> and Models → Performance panel, and hourly redb rollups. Kept here as the
> design record — the reasoning about cardinality, about why "tok/s per node"
> needs care, and about headers flushing before the body is what a future
> change needs to not undo. **Still open**: streaming responses carry route
> identity in headers but token-level timings only in the final SSE usage
> event for the dashboard's own consumption; exposing TTFT/TPOT to third-party
> streaming API clients is not done. K-layer prefetch and multi-segment
> hedging remain separately deferred.

**Asked for**: "when using a model over inference, on the chat for example, I
know it times the result but does it also give performance status, routing info
(how many peers it routed through etc)", "these sorts of things should be
included in diagnostics also so logfiles etc are analysable", and then, after
two days where every bug cost hours of cross-machine log-grepping: "we need to
add the diagnostics, statistics, routing info … so we can move forward faster.
It should also help with performance, scaling, regional routing, peer latency,
token per second per node per shard".

### 1. The finding: we measure nearly all of this already and expose almost none of it

An audit of the request path (2026-07-26) against the 26-point lifecycle in
`docs/DIAGNOSTICS.md`:

| Signal | Computed today | Reachable by a user or a script? |
|---|---|---|
| `schedule_ms`, `execute_ms`, `total_ms` | `router/distributed_exec.rs` | log line only |
| per-segment `segment_ms`, `activation_bytes` | `pipeline/distributed.rs` | **DEBUG** log only |
| route (segment count, node ids, layer ranges) | scheduler assignment | `/pipeline-plan` — the *plan*, not what actually ran |
| peer RTT | `PeerInfo.latency_ms` | `/api/admin/peers` |
| per-peer ms/layer EMA | `state.metrics.peer_segment_latency_ms_per_layer` | scheduler-internal |
| per-(model × segment × holder) latency EWMA **+ variance + sample count** | `state.metrics.hedge_tracker` | scheduler-internal |
| peer region (ISO-3166) | `NodeCapability.region` | `/pipeline-plan` only |
| estimated tok/s | `NodeCapability.est_tokens_per_sec_7b` | scheduler-internal |
| failures (ring buffer) | `recent_failures` | `/api/admin/diagnostics` |

So the gap is **exposure and retention, not measurement**. `HedgeStats` alone
already carries per-(model, segment, holder) EWMA latency *with variance* — most
of "peer latency per node per shard" — and nothing can read it.

Genuinely absent:

- **Time to first token, server-side.** The single most important LLM serving
  number. It exists only in `cli/bench.rs`, measured client-side. OpenTelemetry
  makes `gen_ai.server.time_to_first_token` a first-class server metric.
- **Time per output token** (`gen_ai.server.time_per_output_token`) — the decode
  phase. Wall-clock total cannot distinguish a slow queue from a slow decode.
- **Queue wait** as distinct from scheduling.
- **A success record.** `recent_failures` has no sibling ring, so "why was that
  one slow" is unanswerable after the fact — only "why did that one break".
- **Serving-side telemetry.** Every counter is requester-side. A node that
  serves segments for others records nothing about it, so an operator cannot
  answer "is my node actually contributing, and how well?"
- **Any persistence.** All of it is in-memory and lost on restart. No trend, no
  before/after for a release.

### 2. Name it the way the industry already names it

Do not invent a vocabulary. OpenTelemetry's GenAI semantic conventions define
exactly this shape, and matching them means Grafana dashboards and OTel
collectors work with no translation layer:

- `gen_ai.server.request.duration` (histogram, seconds)
- `gen_ai.server.time_to_first_token` (histogram, seconds)
- `gen_ai.server.time_per_output_token` (histogram, seconds)
- `gen_ai.client.token.usage` (histogram, `{token}`, attribute `gen_ai.token.type` = input|output)
- attributes: `gen_ai.operation.name`, `gen_ai.request.model`, `gen_ai.response.model`, `error.type`

Our existing `swarmllm_inference_latency_seconds` is `gen_ai.server.request.duration`
under a local name. Keep the old series for one release, emit both, then retire.

The swarm-specific dimensions (route shape, segment count, peer, shard, region)
have no OTel equivalent and stay under `swarmllm_*`.

### 3. One record, built once — `RequestTrace`

The recurring defect in this codebase is a shared rule implemented per path
(`.claude/rules/architecture.md`). Observability is the worst possible place to
repeat it: four response paths each assembling their own timing struct would
drift immediately, and the drift is invisible because nothing fails.

So: **one `RequestTrace`, threaded through the request, written once at
completion, and the sole input to every surface.** Headers, the log line, the
diagnostics ring, Prometheus and the dashboard all render *from it*. Adding a
field means adding it once.

```rust
pub struct RequestTrace {
    request_id: Uuid,
    model: ModelId,
    operation: &'static str,        // chat | completion | embedding | responses
    route: Route,                   // Local | Split | Distributed | Relayed | Cloud
    // timeline — monotonic Instants, resolved to ms at emit
    t_admitted: Instant,
    t_dequeued: Option<Instant>,    // → queue_ms
    t_assembled: Option<Instant>,   // → schedule_ms
    t_first_token: Option<Instant>, // → ttft_ms      ← new
    t_finished: Option<Instant>,    // → decode_ms, tpot_ms
    segments: Vec<SegmentTrace>,
    prompt_tokens: u32,
    completion_tokens: u32,
    outcome: Outcome,               // Ok | Error(kind) | Cancelled
}

pub struct SegmentTrace {
    index: u16,
    node_id: NodeId,                // local node for a local segment
    is_local: bool,
    region: Option<String>,
    shard_indices: Vec<u32>,
    layer_range: (u32, u32),
    elapsed_ms: u32,                // already logged, currently discarded
    activation_bytes: u32,          // already logged, currently discarded
    transport: Transport,           // Direct | Relayed | Loopback
    hedged: bool,
    failed_over_from: Option<NodeId>,
}
```

`Route` must be derived from the assignment, never guessed from segment count —
a 1-segment remote pipeline and a 1-segment local one are different routes and
that distinction is exactly what a confused user needs.

### 4. Where and when to capture

Phase boundaries only. Never per token, with the single exception of a
`t_first_token.get_or_insert(Instant::now())` on the emit path — one predictable
branch, no allocation. `.claude/rules` forbids hot-path overhead in
`pipeline.rs`, `split/executor.rs::forward` and `forward_through_segments`; a
`Vec<SegmentTrace>` with capacity 4 allocated once per *request* is well inside
budget, per-token work is not.

| # | When | Where | Field |
|---|---|---|---|
| 1 | request admitted | `api/openai.rs`, `anthropic/*`, `mcp`, `responses` | `t_admitted`, `model`, `operation` |
| 2 | enqueued | `router/mod.rs` "Queued inference request" | trace created, tier recorded |
| 3 | dequeued | `router/mod.rs::dispatch_single` | `t_dequeued` → **queue_ms** |
| 4 | pipeline assembled | `router/distributed_exec.rs` | `t_assembled` → **schedule_ms**, `segments[]`, `route` |
| 5 | per-segment done | `pipeline/distributed.rs` (both local + remote arms) | `elapsed_ms`, `activation_bytes`, `transport` |
| 6 | **first token emitted** | the three text sources — `executor.rs`, `process_pool.rs`, `pipeline/distributed.rs` | `t_first_token` → **TTFT** |
| 7 | completion | `router/mod.rs` "inference completed" | `t_finished`, tokens, outcome → **TPOT** |
| 8 | failure / cancel | existing `recent_failures` site | `outcome` |

Point 6 is the one that needs care: TTFT must be stamped at all three text
sources or it is silently wrong on one path — the exact failure mode of gotchas
#167/#173. Stamp it inside `finalize_reply_text`'s streaming sibling (the single
emit choke point) rather than at three call sites.

Point 5 already logs everything needed; the values are formatted into a string
and dropped. Capturing them is nearly free.

### 5. "Tokens per second per node per shard" — the honest version

This needs restating before it is built, because the obvious reading is not
measurable and would produce a confidently wrong number.

In a pipeline the segments are **serialised**: every token traverses node A's
layers, then node B's. There is no independent "tok/s of node A" — A and B
produce the *same* token stream. Reporting `tokens / A_time` would show two
nodes each "doing" the full token rate, and the numbers would not compose.

What is real and useful:

- **Per-segment share of inter-token latency** — `segment_ms` per token, per
  (node, layer_range). These *do* sum to the total, so they identify the
  bottleneck hop, which is the actual question.
- **Normalised throughput: ms per layer per token.** Comparable across nodes
  serving different-sized segments, and already approximated by
  `peer_segment_latency_ms_per_layer`.
- **Derived node capacity**: `1000 / (ms_per_layer × layers_served)` = the tok/s
  that node *would* sustain if it served the whole model. Useful for scheduling
  and leaderboards; must be labelled as derived, not measured.

For a **non**-pipelined route (one node serves all layers) tok/s per node is
exactly the request's tok/s, and should be reported plainly.

### 6. The missing half: serving-side telemetry

Everything today is requester-side. Add, on the node that *serves* a segment
(`daemon/dispatch/layer_forward.rs`, which already computes `elapsed_ms`):

- segments served, by `(model, layer_range, requester)` — counter
- forward compute time — histogram
- activation bytes in/out — counter (this is the bandwidth cost of contributing)
- rejections by reason (queue full, shard missing, refused)

Without this an operator cannot tell a well-behaved node from one whose
segments everyone times out on, and neither can trust scoring.

### 7. Surfaces — and what belongs on each

**Response headers** (route identity, known at assembly time, so streaming-safe):

```
x-swarm-route: distributed
x-swarm-segments: 2
x-swarm-nodes: 0718d8b9,96842635
x-swarm-regions: TH,TH
```

**`Server-Timing`** for the durations, rather than bespoke `x-swarm-*-ms`
headers — it is a W3C standard that browser devtools renders natively and
`PerformanceServerTiming` exposes to JS:

```
Server-Timing: queue;dur=3, sched;dur=1, ttft;dur=180, decode;dur=1420,
               seg0;dur=520;desc="0718d8b9 L0-10", seg1;dur=900;desc="96842635 L10-16"
```

Cross-origin callers need `Timing-Allow-Origin`; the dashboard is same-origin so
it needs nothing. **Header caveat that the earlier draft got wrong**: headers
flush *before* the body, so on a streaming response only pre-body facts (route,
nodes, queue, schedule) can go there. TTFT/decode/tok-s must ride in the final
SSE event — the `usage` chunk `include_usage` already emits is the natural
carrier. HTTP trailers are the "correct" answer and have poor client support;
not worth it.

**One greppable log line** at completion — this is what makes a logfile
analysable without reconstructing a route from a dozen interleaved DIAG lines:

```
DIAG: request complete request_id=… route=distributed segments=2
      nodes=0718d8b9,96842635 regions=TH,TH queue_ms=3 sched_ms=1 ttft_ms=180
      decode_ms=1420 tokens=48 tok_per_sec=33.8 seg0_ms=520 seg1_ms=900
      bytes=39188 outcome=ok
```

**`GET /api/admin/diagnostics`** — add a `recent requests` ring (the successful
sibling of `recent_failures`, same size), so a tester pastes one block instead of
a log excerpt. Add a per-peer table: RTT, ms/layer, EWMA + variance + samples
from `hedge_tracker`, region, segments served, last-seen.

**Prometheus** — low cardinality ONLY. `route`, `model`, `outcome`, `error.type`
are bounded; `peer × model × shard` is not (50 peers × 10 models × 10 shards =
5 000 series from one node, and it grows with the swarm). Per-peer detail belongs
in the JSON endpoint, which is pulled on demand and never retained. This
distinction is the single most important thing to get right — an unbounded label
set will take down the scrape long before anyone notices the dashboard is useful.

**Dashboard** — chat shows "answered by 2 peers · 33.8 tok/s" with the route on
hover (needs i18n across 21 locales). A swarm-tab panel renders the per-peer
table. Both read the same trace.

### 8. Retention

Do not build a time-series database. `monitoring/` already ships Prometheus +
Grafana; that is the trend store. In-process, keep:

- last N `RequestTrace` (N ≈ 50, same as the failures ring) — in memory, for the
  "what just happened" view
- hourly rollups (count, p50/p95 TTFT, p50/p95 tok/s, bytes, per route) persisted
  to redb — small, bounded, survives restart, enough for "is this release slower
  than the last one" without a scrape target

### 9. Sequencing

1. `RequestTrace` + capture points 1-8 + the one log line. Behind nothing — it is
   strictly additive and immediately makes logs analysable.
2. TTFT/TPOT at the emit choke point; OTel-named Prometheus histograms.
3. Response headers + `Server-Timing`; final-SSE-event metrics for streaming.
4. Diagnostics ring + per-peer table.
5. Serving-side counters.
6. Dashboard + i18n.
7. redb hourly rollups.

Steps 1-2 pay for themselves the first time a tester reports something slow.

## `download-shards` takes ~25s to start writing (reported 2026-07-26)

A tester exercising `POST /api/admin/hf/download-shards` for the first time
reported it works, but sits on a zero-byte `.tmp` for roughly 25 seconds before
data starts flowing. Not a failure, and not urgent — but it is the *first* thing
a new user does after picking a model, and 25 seconds of an apparently stalled
download is exactly when someone concludes it is broken and kills it.

Worth finding out where the time goes before deciding anything: the HEAD probe
and range-probe both retry with a 5/30/120s backoff (`retry_hf`), so a single
slow or rate-limited HF response would explain it, as would GGUF-header parsing
or layer-shard layout computation on a large file. If it is the probe, the fix is
progress reporting rather than speed — say "checking the file on HuggingFace"
instead of showing a 0-byte file.

## "Newest direct connection" can select a dead one (observed 2026-07-26) — FIXED v0.3.34

> **Fixed by option 1 below, adapted.** Rather than plumbing the application's
> ACK timeout back into the behaviour, selection now uses a signal the crate
> already tracks: `pending_outbound_responses` is inserted on send and removed
> when the response arrives, so a connection that is answering drains it while a
> half-open one only accumulates. Selection prefers the direct connection with
> the FEWEST un-answered requests, breaking ties toward the newest — which
> preserves the DCUtR behaviour the newest-wins rule existed for. No new state,
> no API change, no extra round trip. The notes below are kept as the record of
> what was observed.

The vendored `libp2p-request-response` patch picks the **newest direct**
connection to a peer, on the reasoning that "a half-open connection is almost
always an older one that died quietly, and DCUtR's upgraded direct connection is
by definition the newest".

Observed live contradicting that. Local held three connections to a LAN peer:

```
1  15:07:46  /ip4/192.168.1.60/udp/8800/quic-v1/p2p/…   direct, outbound
2  15:27:10  /ip4/192.168.1.60/udp/8800/quic-v1          direct, inbound
3  15:27:12  /p2p/…                                      relayed, inbound
```

Connection 1 had just served three consecutive successful requests. Connections 2
and 3 appeared during a pool join. With the v0.3.33 fix, 3 is correctly excluded
as relayed — and selection then lands on **2**, which silently swallows every
send (no response, no `OutboundFailure`, 10s ACK timeout), while the known-good
connection 1 sits unused. Restarting both ends clears it.

**Why this is hard**: a half-open QUIC connection is indistinguishable from a
live idle one without probing. Age is a heuristic and this is the case where it
points the wrong way.

**Candidate fixes, cheapest first:**

1. **Feed the ACK timeout back into selection.** `RR_ACK_TIMEOUT_SECS` already
   detects the silent drop; today it only fails the request. Recording the
   connection id as suspect and skipping it on the retry would make the existing
   retry actually change the outcome instead of re-picking the same dead path.
   Small, local, and needs no protocol change — this is the one to do.
2. Prefer the connection that most recently carried a **successful** response,
   falling back to newest. Turns age into a tiebreaker rather than the rule.
3. Probe liveness before selecting. Correct but adds a round trip to every send.

Note the interaction with `max_established_per_peer = 3` (raised so DCUtR can
hold a relayed and a direct connection at once, gotcha #163): the higher cap is
what makes several connections routine, so it is also what makes mis-selection
routine.

## Shard-holder retraction depends on gossip reaching the peer (observed 2026-07-26) — MITIGATED v0.3.31, completed post-.33

> **Requester-side mitigation shipped in v0.3.31.** A holder that reports missing
> shard data now loses its claim over the layer span it was asked to serve
> (`pipeline::remote_error_means_missing_shard` →
> `retract_shard_holder_claims_for_range`), and the request retries against a
> fresh assembly, so the stale claim costs one internal retry rather than every
> request until the announcement lands. The underlying gossip dependence is
> unchanged and the note below still describes it.
>
> **The per-request holder blacklist is now implemented too, and it turned out to
> be required, not optional.** Live testing showed retraction alone is futile: the
> DHT still advertises the holder, so the retry's assembly re-learns the claim and
> picks the same dead peer, failing identically. Before: 6 retractions per request
> (two rounds). After: 3, and the retry excludes the holder outright. The
> blacklist is keyed by request id and cleaned up alongside `active_traces`, so one
> bad data point cannot ban a peer globally. It also covers the pre-existing case
> of a connected peer that fails without disconnecting.

**Observed live**, by an external tester, during the v0.3.30 split testing.

`DELETE /api/admin/models/{id}/shards/{index}` correctly re-announces
immediately, with `complete_for_models: vec![model_id]` so peers know that
whatever is absent from the announcement was deliberately removed rather than
merely unmentioned (R146/R147). The design is right.

But the announcement is a **GossipSub broadcast**, and a NAT'd internet peer may
not receive it promptly. Until it does, that peer's registry still lists us as a
holder of the deleted shard, and it will route requests we cannot serve. The
tester saw exactly this: after two shards were deleted from a node mid-session,
their node routed a *whole-model* remote-generate to it and got

```
Inference error: Internal error: blk.0.attn_q: ShardReader: position 345977248
is in a missing region
```

which is the honest failure — the node was asked for layer 0 and genuinely does
not hold shard 0.

**Cost today**: one failed request per stale router, self-healing on the next
periodic `ShardAnnounce` from `health/monitor.rs`. Not silent — the error names
the missing tensor — and `failure_is_penalty_worthy` correctly declines to
penalise anyone for it.

**Worth doing if this bites in practice**: on receiving a `ShardNotFound` /
missing-region error from a holder, drop that holder's claim for the shard
locally rather than waiting for the next announcement. That converts a repeated
failure into a single one and needs no protocol change. A per-request holder
blacklist (already wanted for the `is_transient_remote_failure` retry path) would
subsume it.

**Note the negative result that came with it**: the same test confirmed the
weight-tied output-head fix is *narrow*. A node missing shard 0 still fails
correctly when asked for layer 0 — the sidecar only stands in for the tied LM
head, and does not paper over genuinely absent shards.

## Collapse the parallel response paths behind one core loop (2026-07-26)

**The single most expensive recurring defect in this codebase is a shared rule
implemented per path.** It appeared seven times on 2026-07-25/26 (stop strings,
tool-call buffering ×2, `include_usage` ×2, control-token scrubbing,
`strip_provider_prefix`) and four more times on 2026-07-26 alone (the Anthropic
prefix, `build_prompt`'s model name, the reply-text ordering divergence, the
direct `chatml_fallback` calls). Every instance had a correct helper that one
consumer didn't call.

Two structural fixes have shipped, and they work — see `.claude/rules/
architecture.md` § "One invariant, N paths" for the escalation ladder:

- **Choke point over convention.** `inference::finalize_reply_text` now owns the
  whole ordered reply-text sequence, and `providers::strip_prefix_in_body` runs
  inside the three functions that actually send. A new caller is correct with no
  author action.
- **Required over optional.** `build_prompt` takes the model name as a required
  argument rather than an `Option` behind a wrapper that passed `None`.

**What remains** is the duplication those fixes route around rather than remove:
`api/openai/streaming.rs` (1155 lines), `api/anthropic/handlers.rs` (1457) and
`api/openai/responses/stream.rs` (1067) each implement their own
streaming/non-streaming pair over the same router output. That is why there are
so many places for a rule to be forgotten in the first place.

**Proposed**: one core generator loop producing a neutral event stream
(`TextDelta`, `ToolCall`, `Usage`, `Finish`), with three thin per-surface
adapters that only serialise those events into OpenAI SSE, Anthropic SSE, or
Responses events. Every cross-cutting rule then has exactly one home: applied to
the event stream, not to each transcription of it. Estimated to remove
substantially more than half of those 3,679 lines.

**Why not now**: it rewrites every response path at once, so it needs its own
release and a live A/B on all four surfaces (OpenAI stream/non-stream, Anthropic
stream/non-stream, Responses foreground/background, MCP). The choke-point fixes
above are the correct interim: they make the current duplication safe without
pretending it isn't there.

## Replace the hand-rolled chat-template evaluator with minijinja (2026-07-26)

`src/inference/chat_template/` is a hand-written mini-Jinja subset (~600 lines
across `parser.rs` + `eval.rs`). It is the single highest-consequence component
per line in the codebase: when it cannot render a template, the caller silently
falls back to a *different model family's* prompt format, and the model answers
in that format. That is the root cause of the `<|im_end|>` leak four releases
chased through the output scrubber (gotcha #169).

**Measured state as of 2026-07-26** (survey test over real templates pulled from
GGUF headers on disk + HuggingFace `tokenizer_config.json`):

| Template (real, not simplified) | Renders? |
|---|---|
| Llama-3.2-1B / 3B, Llama-3.1-8B | ✅ (after the alias fix) |
| Qwen2.5 (0.5B, Coder-7B) | ✅ |
| Phi-3.5-mini | ✅ |
| TinyLlama / Zephyr | ✅ |
| DeepSeek-R1-Distill-Qwen | ✅ |
| **Mistral-7B-Instruct-v0.3** | **❌ with a system message**, ✅ user-only |

Mistral's official template needs `namespace()`, `selectattr(...) | list`,
`messages[1:]` slice-binding, `is defined` / `is not none`, and dict iteration
(`for key, val in tool.items()`). Implementing those is not a bug fix, it is
writing a Jinja interpreter — and templates keep getting more complex, because
tool-calling and reasoning blocks live in them now.

**The ecosystem has already converged on not doing this by hand.** llama.cpp,
Jan, GPT4All and Docker Model Runner all use [google/minja](https://github.com/google/minja),
a dedicated C++ Jinja subset whose stated goal is "each and every major LLM
found on HuggingFace". The Rust equivalent is
[`minijinja`](https://docs.rs/minijinja/) (mitsuhiko), which is what
[mistral.rs](https://github.com/EricLBuehler/mistral.rs) uses for exactly this
job, alongside HuggingFace and BAML. Core deps are just `memo-map` + `serde`,
with `builtins` covering `namespace()`, `selectattr`, slicing, `is defined` and
`loop.index0`; the rest of the surface is feature-gated, so the footprint is
controllable.

**Proposed**: swap `apply_chat_template`'s internals for `minijinja`, keeping
the current signature and the fallback chain as the safety net for genuinely
broken templates. Keep the real-template survey test as the acceptance gate —
extend it to pull the top ~20 families' `tokenizer_config.json` and assert every
one renders, so the next Mistral-shaped gap fails CI rather than a user's chat.

**Why it is not in this release**: it changes prompt construction for every
model at once, which wants its own release and a careful A/B against the current
renderer on the models we hold locally. The interim mitigation (shipped
2026-07-26) is that `build_prompt` now *requires* a model name and every family
we can name has a real fallback, so a template failure degrades to the right
format instead of ChatML.

---

## How to use this file

When starting a new feature, grep this file for keywords related to the area you're touching. If your feature unblocks a deferred item, either pick it up in the same PR (if scope allows) or move the entry to "completed" with the closing commit reference.

When closing a sweep finding as `deferred`, add an entry here so future sweeps don't re-flag it. The entry must include enough context that the closure isn't a black hole.
