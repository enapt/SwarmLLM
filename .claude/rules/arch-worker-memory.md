---
paths:
  - "src/inference/process_pool.rs"
  - "src/inference/model_worker.rs"
  - "src/inference/worker_ipc.rs"
  - "src/inference/slot_table.rs"
  - "src/inference/split/kv_cache.rs"
  - "src/inference/split/kv_budget.rs"
  - "src/model/auto_manage/**"
  - "src/daemon/shard_loader.rs"
  - "src/daemon/gpu_support.rs"
---

# Worker memory: graphics, RAM and the KV cache

Invariants this codebase has paid to learn. Each heading is the rule; the text
under it is what to do. The evidence — what the rule replaced, what it was
measured at, and what a change must keep — lives in `docs/invariants/`.

**This file loads only when you touch the code it governs.** Read the linked
`docs/invariants/` topic file before changing code a rule names.

## What a task took, a task gives back by being dropped — an abort is not a return

Three resources were released by plain statements placed after an `.await`,
which run only when that await RETURNS. Two ordinary things stop it doing so:
`cancel::unless_cancelled` cancels by DROPPING the future, and
`SwarmMessage::CancelInference` calls `abort_handle().abort()` — and **no
in-band checkpoint can defend against an external abort**, so `bail_if_cancelled`
bracketing is not a defence either.

- **`PendingSpawnCharge`** (`inference::process_pool`) — memory charged by
  `admit_to_gpu`/`admit_to_cpu` before `spawn_worker(..).await`. No
  `WorkerHandle` exists yet, so nothing downstream reconciles it; the budget is
  keyed by model and ACCUMULATES until a restart.
- **`InboundForwardSlot`** (`daemon::dispatch`) — the per-peer forward count and
  the abort-registry entry. The sweep removes entries that reach ZERO and cannot
  repair one stuck above it, and `max_forwards_per_peer` floors at 4, so four
  aborts permanently refuse everything that peer sends afterwards.
- **`ShardDownloadClaim`** — see the rule above.

**`local_generate.rs` WRAPPING `pool.generate(..)` in `unless_cancelled` is how
this came back**: `forward_direct` brackets the same call instead, and its
comment says why — "dropping a load half-done abandons a spawning subprocess"
(gotcha #459). A rule that lives only in a comment gets re-broken by the next
file.

→ `docs/invariants/memory.md`

## "Used recently" has four answers, and one of them is never written locally

`model::auto_manage::prune::effective_idle_secs` combines all of them — a
request this node routed, one it served for a peer, the worker's own
`last_used`, and residency as a hard upper bound. **`model_trust.last_request_at`
alone is not an answer**: `record_request` has one caller
(`router::distributed_exec`), so the local fast path and peer-served work both
leave it untouched, and a persisted value from an older build can be stale by
days.

Both consumers read it through that helper now — the idle-VRAM unload since
2026-09-02 (gotcha #437), and R134.7's prune protection since 2026-09-14, which
had been left on the broken signal 400 lines below the fix.

→ `docs/invariants/memory.md`

## A budget is charged by whatever owns the resource

**`SharedState::committed_memory_mb` is the single answer to "how much of this
memory is committed right now"**, and it asks `ModelProcessPool`
(`vram_committed_mb` / `ram_committed_mb`) because the pool admits, charges and
reclaims. Never charge a budget by summing `split_models` — those entries are
GGUF headers read at scan time with no worker behind them, so the figure can
only go up. It read `loaded_mb=5124` eighteen seconds after boot with zero
workers, and nodes delegated models smaller than their free graphics memory.
Guard: `a_memory_budget_is_charged_by_the_pool_never_by_the_metadata_map`.

→ `docs/invariants/memory.md`

## A subprocess you are waiting for can die instead, and the graphics stack can go mid-run

`spawn_worker` races `child.wait()` against `listener.accept()`. Watching the
socket alone made every startup failure cost the full 30 s
`WORKER_CONNECT_TIMEOUT_SECS` — per model, per arriving request — and arrive as
one contentless `worker connect timeout`, while the real cause sat in the log
as raw untimestamped `ld.so` output matching nothing anyone would grep for
(#646/#647). **Whenever code waits for a subprocess to do something, handle it
dying instead.**

**`daemon::gpu_support::gpu_runtime_has_failed` is the single answer to "has the
graphics stack stopped working since we started?"** — the half of the GPU
question that is NOT a property of the card, and the half `local_gpu_is_supported`
originally cached away on the reasoning that the answer "cannot change while the
process runs". A driver update changes it: the daemon keeps running on libraries
it already mapped while every worker it `exec`s dies, so only a failed worker
start reveals it. Set by `diagnose_failed_start` (which re-runs `--version`
rather than guessing from loader text), cleared by any worker that starts, read
by `cpu_reason` (`CpuReason::GpuUnavailable`) and by the health monitor, which
**withdraws the advertised GPU so peers stop routing work this node would fail**.

⚠ **Do not tell the owner the processor takes over.** This binary links
`libcuda.so.1`, so a bad library stops every new process in the loader and a
CPU-only worker exits 127 exactly as a GPU one does.

**Which is why it also withdraws INFERENCE, not just the card.**
`SharedState::inference_outage` is the single answer to "can this node run a
request right now", and it answers for this AND for a stalled message
dispatcher — two unrelated faults with one shape: a total inference outage that
reports itself as healthy. `NodeCapability::can_serve_inference` carries it, and
the scheduler's `gather_candidates` acts on it. **Shard serving stays up** — a
byte-range read needs no worker — and the field defaults to `true` on the wire,
because a peer that says nothing has not said no. Guard:
`both_inference_outages_withdraw_through_one_predicate`.
→ `docs/FUTURE_WORK.md` #89, #90.

→ `docs/invariants/memory.md`

## A fan-out to every worker is bounded, and the two waits mean different things

**`ModelProcessPool::notify_every_worker` is the one place a fire-and-forget
message goes to every live worker** — `cancel_request` and `release_request_kv`
both go through it. It matters because `cancel_request` is awaited INLINE from
the dispatch loop's `CancelInference` arm, and that loop is the network event
loop's only consumer: an unbounded wait there is not a slow cancel, it is a node
that stops receiving anything (gotcha #74).

Both waits are bounded **separately**, because the wrong response to either is
worse than the wait:

- **The writer LOCK timing out** says another task is mid-message. Nothing has
  been written, so stand down at `debug!` — that task owns the worker's fate.
- **The SEND timing out** says the socket will not take a few dozen bytes.
  Dropping that future can leave a PARTIAL FRAME, desynchronising every later
  message, so the worker is marked `dead` and reaped rather than left in a state
  no reader could parse.

Sends run concurrently, so the fan-out costs one timeout, not one per worker.

→ `docs/invariants/memory.md`

## Single-source-of-truth helpers — Worker memory: graphics, RAM and the KV cache

Each names the ONE place a decision is made. A second implementation of any of
them is this codebase's most-repeated defect — see `.claude/rules/architecture.md`
§ "One invariant, N paths". **Read the topic file before changing one.**

Full evidence: `docs/invariants/memory.md`

- **`DaemonMsg::ReleaseRequestKv` — a finished request releases its conversation cache, wherever it is held.** Sent by `ModelProcessPool::release_request_kv` at the one place a request finishes. The `Generate` handlers clear their own; the FORWARD path had nothing, and the daemon's own `cleanup_request_id` is a DIFFERENT PROCESS's store. Unconditional: the worker keys by REQUEST id, so no later turn can find the entry. A KV-admission refusal is `LocalMemoryUnavailable`, carried over IPC as `WorkerMsg::Error::local_memory_refusal` because the wording is deliberately identical to a peer's refusal.
- **`api::process_memory::resident_bytes` — a process's memory is the LARGEST accounting the platform offers.** macOS keeps two that differ by orders of magnitude; under-reporting is the failure that matters.

- **`inference::worker_ipc::worker_error_is_fatal`** — the single source of truth for "did this worker error destroy the worker's device state, or just this request?".
- **`daemon::shard_loader::force_cpu_for`** — the single mapping from `inference.gpu_layers` (`-1` auto / `0` CPU only / `>0` GPU) to the loader's `force_cpu` flag.
- **`daemon::gpu_support::MIN_COMPUTE_CAP` + `local_gpu_is_supported`** — the single answer to "can this card run OUR kernels?".
- **`model::auto_manage::vram::ADMISSION_KV_CONTEXT`** — the context length admission charges KV cache for, on either device, whatever the user configured.
- **`ModelProcessPool::free_vram_for_admission` + `plan_vram_reclaim`** — reclaim graphics memory from models nothing is using rather than demoting the requested one to the processor.
- **`should_return_to_gpu` + `ModelProcessPool::worker_should_return_to_gpu`** — the single answer to "is this resident worker still in the right place?", asked on the request path in `get_or_spawn` rather than on a timer.
- **Graphics memory has ONE owner: `ModelProcessPool`** — it admits (`admit_to_gpu`), charges (`vram_reserved_mb`) and reclaims (`free_vram_for_admission`, `try_idle_vram_unload`). Nothing else may take memory away from a loaded model.
- **A worker's growth is weighed by the budget its spawn charged** — `WorkerHandle::holds_gpu_memory` reads `charged_against_ram` (what `charges_ram` decided at spawn), NEVER `placed_on_cpu_because`, which records why a model was *demoted* and so reads `None` on a machine with no card exactly as it does for a worker holding one. Re-deriving it sent every later layer-range growth on every GPU-less node to `admit_to_gpu`, which has no ceiling when `vram_budget_mb` is 0 — the anti-swap gate ran once per model and never again, and a 16 GB Mac swapped. Growth is the COMMON case on a swarm node. `a_live_workers_growth_is_weighed_by_the_budget_its_spawn_charged` in `tests/repo_consistency.rs` fails the build on a re-derivation.
- **`model::auto_manage::storage_budget` is the ONE answer to "how much shard storage may this node hold?", and `held_disk_bytes` the one answer to "how much does it hold?"** — `storage_budget_now(&state)` gives both, live; the download pass, prune's disk pressure, the settings storage bar, the pool page and the diagnostics report all read it instead of re-deriving one. **The byte figure measures the DIRECTORY, not the manifest** (`held_shard_bytes` prices the manifest and so sees only `shard_NNN.bin`, missing the header, the tied output weight, mmproj, quarantined files and dead `.tmp` — 5% on the live node, 95% of it one file kind). **It is only safe to charge for them because `shard::cleanup_orphaned_model_files` makes them reclaimable**, called once per prune cycle: prune deletes shards and nothing else, so a figure counting bytes it cannot free makes a node near its limit shed shards forever chasing an unreachable floor. Never re-couple the two — changing the byte source without the reclaim pass is `docs/FUTURE_WORK.md` item 49's shape. And a test fixture that exercises this must own its `data_dir`, or it measures the developer's real node.
- **`model::auto_manage::prune::effective_idle_secs` — residency is a hard UPPER BOUND on "idle since", and the worker's own `last_used` is the signal that moves** — NOT `model_trust.last_request_at`, which nothing in the current code writes — a stale persisted value once unloaded a model five seconds after it answered.
- **An admitted prompt is RECORDED, not just decided** — `KvCacheStore::record_prompt_admission` / `outstanding_admission_bytes`; `ensure_room_for_prompt` adds the outstanding total to the live figure before `admit_prompt`, and records its own claim once admitted. **A request's claim and its reservation are released TOGETHER, by `forget_request_bookkeeping`, which `clear_request` and `cleanup_request_id` both call** — the draw-down that makes a stale claim harmless reads the request's CACHE ENTRY, so a cleanup that removes the entry and keeps the claim charges the full admission for memory it just freed (gotcha #637: 1848 MB owed against a 3042 MB budget, identical prompts refused for ten minutes). Never release one of the two maps alone.
- **`inference::split::kv_budget::admit_prompt` + `PrefixCache::release`** — ONE decision for a whole prompt, before prefill, charging live caches PLUS the prefix cache's snapshots (the same device memory, previously charged nowhere): fit → evict cached prompts, oldest hit first → refuse with a 503 at token 0. **A budget must see every tenant of the memory it bounds.**
- **`inference::split::kv_budget`** — the KV memory budget and the admission check against it.
- **A prompt of known length is RESERVED, not grown into** — `KvCacheStore::set_reserved_positions` (written by the worker at the top of `ensure_room_for_prompt`, before any budget question) sizes every layer's FIRST allocation via `new_kv_cache(.., reserve_positions)`; growth by `Tensor::cat` is O(n²/quantum) in copies and one device allocation per step, none of it charged (97 GB copied and 2279 allocations for a 20837-token prompt). The guard charges what will be allocated, read off the buffer (`kv_budget::positions_to_allocate`), never derived from `index_pos`. Absent means "grow as before"; `SWARMLLM_KV_RESERVE=0` is the in-binary A/B. → `docs/invariants/memory.md` § "A prompt of known length is reserved"
- **`inference::process_pool::worker_socket_path`** — the worker IPC socket path, and the ONLY place it is built.
