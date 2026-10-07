---
paths:
  - "src/inference/process_pool.rs"
  - "src/inference/model_worker.rs"
  - "src/inference/worker_ipc.rs"
  - "src/inference/cuda_pool.rs"
  - "src/inference/card_pace.rs"
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

A resource taken before an `.await` is returned by a guard that releases on DROP: `cancel::unless_cancelled` and `CancelInference`'s `abort_handle().abort()` skip every statement after the await, and no in-band checkpoint (`bail_if_cancelled`) can defend. Guards: **`PendingSpawnCharge`** (`inference::process_pool`), **`InboundForwardSlot`** (`daemon::dispatch`), `ShardDownloadClaim`. Never wrap `pool.generate(..)` in `unless_cancelled` in `local_generate.rs` (gotcha #459).

→ `docs/invariants/memory.md` § "What a task took, a task gives back by being dropped"

## "Used recently" has four answers, and one of them is never written locally

**`model::auto_manage::prune::effective_idle_secs`** is the single answer to "used recently" (a request this node routed, one it served for a peer, the worker's `last_used`, residency as an upper bound). `model_trust.last_request_at` alone is not an answer — `record_request` has one caller. The idle-VRAM unload and prune protection both read it.

→ `docs/invariants/memory.md` § ""Used recently" has four answers, and one of them is never written locally"

## A budget is charged by whatever owns the resource

**`SharedState::committed_memory_mb`** is the single answer to "how much memory is committed", asked of `ModelProcessPool` (`vram_committed_mb` / `ram_committed_mb`). Never sum `split_models` (headers, no worker). Guard: `a_memory_budget_is_charged_by_the_pool_never_by_the_metadata_map`.

→ `docs/invariants/memory.md` § "A budget is charged by whatever owns the resource"

## The contribution level caps the SWARM's work, not the owner's prompt (2026-09-26)

**`process_pool::Requester`** (`Owner` / `Swarm`) is a REQUIRED argument of `ModelProcessPool::generate` and `forward_for_request` (`IpcForward::for_the_owner`); the worker records it once (`KvCacheStore::mark_owner_request`) and `cpu_pools::in_phase_pool` reads it. The owner's PROMPT reading runs on `ResourceConfig::owner_prefill_threads`; decode and the swarm's work keep the contribution width.

→ `docs/invariants/memory.md` § "The contribution level caps the swarm's work, not the owner's prompt (2026-09-26)"

## A subprocess you are waiting for can die instead, and the graphics stack can go mid-run

Whenever code waits for a subprocess, handle it dying instead (`spawn_worker` races `child.wait()` against `listener.accept()`). **`daemon::gpu_support::gpu_runtime_has_failed`** is the single answer to "has the graphics stack stopped working since we started?" (health monitor withdraws the GPU). ⚠ Never tell the owner the processor takes over — a CPU-only worker dies the same way. **`SharedState::inference_outage`** is "can this node run a request" (`can_serve_inference`); shard serving stays up. Guard: `both_inference_outages_withdraw_through_one_predicate`.

→ `docs/invariants/memory.md` § "A worker that dies is not a worker that is slow, and the graphics stack can go mid-run"

## A reply between two forwards is not idle

**`WorkerHandle::in_use` is the single answer to "is this worker in use?"** — a response in flight OR a conversation held between forwards (`kv_holders`, `CONVERSATION_GAP_SECS`); promotion, both reclaims and `models_with_inflight_requests` ask it. Guard: `whether_a_worker_is_in_use_is_decided_in_one_place`. Promotion waits for `reason_still_holds`; a forward past the prompt pass with no conversation is refused (`model_worker::forward_lacks_its_conversation`).

→ `docs/invariants/memory.md` § "A reply between two forwards is in use, and a lost conversation is refused"

## A worker that cannot grow gets the spawn's ladder, not a refusal (2026-09-27)

**`ModelProcessPool::grow_worker` is the one place a live worker takes on a new range**: reclaim idle models' graphics memory (`free_vram_for_admission`), then — only if `WorkerHandle::in_use` is false — retire the worker so `get_or_spawn` places the range afresh. Never retire one mid-conversation (#93).

→ `docs/invariants/memory.md` § "A worker that cannot grow gets the spawn's ladder (2026-09-27)"

## A model loaded for ANOTHER model's request is a guest (2026-09-28)

**`process_pool::Tenancy` is a REQUIRED argument of `get_or_spawn`.** A `Guest` (the drafter, `ModelProcessPool::draft`) takes only memory that stands free: no reclaim, split, promotion, growth or notice. `dsd.rs` reads ahead only once `holds_segment`.

→ `docs/invariants/memory.md` § "A model loaded for another model's request is a guest (2026-09-28)"

## A fan-out to every worker is bounded, and the two waits mean different things

**`ModelProcessPool::notify_every_worker` is the one fire-and-forget fan-out** (`cancel_request`, `release_request_kv`), awaited inline from the dispatch loop (gotcha #74). The writer LOCK timing out stands down at `debug!`; the SEND timing out marks the worker `dead` (a partial frame desynchronises it). Sends run concurrently.

→ `docs/invariants/memory.md` § "`ModelProcessPool::notify_every_worker` — a fan-out to every worker is bounded"

## A card's free memory is read after a synchronize (2026-09-26)

**`kv_budget::device_free_and_total_bytes`** synchronizes the stream before `mem_get_info` and adds `cuda_pool::reusable_bytes`; every budget decision reaches the card through `SplitModel::kv_budget_now` → it. Guard: `the_cards_free_memory_is_read_after_a_synchronize`; A/B `SWARMLLM_KV_DEVICE_SYNC=0`. `reply_reserve_positions(max_tokens)` is a REQUIRED argument of `ensure_room_for_prompt`; a refusal caused by OTHER conversations says wait (`other_conversations_hold_the_room`).

→ `docs/invariants/memory.md` § "A card's free memory is read after a synchronize"

## The daemon holds no CUDA context — only workers do (2026-10-05)

**`gpu_support::describe_local_gpu`** (driver API: `cuInit`, name, total — no context) is how the daemon learns its card. Never llama.cpp's device list or `Device::cuda_if_available` in the daemon: each leaves a context that held 137 MiB of an 8 GB card for the node's life. Free card memory is read live (`vram::query_gpu_vram_free_mb`), never kept from startup — `GpuInfo` has no free field.

→ `docs/invariants/memory.md` § "The daemon holds no CUDA context (2026-10-05)"

## nvidia-smi is asked through one bounded helper (2026-10-05)

**`vram::nvidia_smi`** (a `BoundedCommand`: 10 s, never two copies at once — while a stuck one lives, the next reading is `None` at once) is the only way the daemon and workers run `nvidia-smi`; a bare `Command::new("nvidia-smi")` waits as long as the driver does — 14 min during one card reset. Guard: `nvidia_smi_is_asked_only_through_the_bounded_helper`. Once a copy has been stuck for 60 s (`DRIVER_NOT_ANSWERING_AFTER` — one slow answer is not a reset, #802), `vram::graphics_driver_not_answering` is true and new workers go to the processor (`CpuReason::DriverNotAnswering`, transient — promotion brings them back); a shorter stall is waited out on the card.

→ `docs/invariants/memory.md` § "nvidia-smi is asked through one bounded helper (2026-10-05)"

## A card that stalls is given less work, not more (2026-09-28)

**`inference::card_pace::CardPace` is the one answer to "may this worker start another generation on its card now"**: a device-bound step ≥ 2 s halves the ceiling. The gate sits in the `Generate` arm BEFORE the batched/sequential choice; a new device-bound step is timed into it, a new generation path gated by it. Refusal = `LocalMemoryUnavailable`. A/B `SWARMLLM_CARD_PACE=0`.

→ `docs/invariants/memory.md` § "A card that stalls is given less work (2026-09-28)"

## A card's memory pool keeps what the worker frees, until the worker is idle (2026-09-28)

**`split::loader::load_device` is the one place a worker picks the card** and raises the pool release threshold (`cuda_pool::keep_freed_memory`); `model_worker::run_worker` trims after `cuda_pool::IDLE_TRIM`. Any "free for this process" reading adds `cuda_pool::reusable_bytes`. Guard: `every_split_model_reaches_the_card_through_load_device`; A/B `SWARMLLM_CUDA_POOL_KEEP=0`.

→ `docs/invariants/memory.md` § "A card's memory pool keeps what the worker frees (2026-09-28)"

## The owner is told when fresh card memory has become slow (2026-10-01)

**`cuda_pool::probe_once`** times 16 fresh 4 MB allocations once per worker at `load_device`, never per request; a probe ≥ 0.5 s reaches `ModelProcessPool::slow_card_notice` (`WorkerMsg::CardAllocationProbe`), WSL2 only, at most every 12 h. It changes nothing the node does.

→ `docs/invariants/memory.md` § "The owner is told when the card has become slow to hand out memory (2026-10-01)"

## A lone decode stream is never held for a batch (2026-09-29)

**`process_pool::collection_target`** waits for a batch only when ANOTHER request is decoding on the same model (`ActiveStreams`). Never reintroduce a wait a lone stream pays.

→ `docs/invariants/memory.md` § "A lone decode stream is never held for a batch (2026-09-29)"

## The download pass fetches only what prune would keep (2026-10-06)

**`AutoShardManager::would_shed_copy` is the one answer to "would prune shed this copy?"** — prune asks it of what it holds; EVERY automatic fetch path asks it before fetching, at the disk pressure AFTER the fetch: `select_within_budget` (cumulative per cycle) and `complete_pending_shard_fetches` (which DROPS the entry). A finished origin download leaves `shard_p2p_failed` in `announce_shard_acquired` (gotcha #797); an entry the origin cannot serve (`pending_fetch_can_proceed` → `can_fetch_shard_from_origin`) is dropped back to the peers, never left to block them. Every file decision reads DISK pressure, never graphics memory; the filesystem's last 10% is never ours (`FREE_DISK_RESERVE_PCT`). Guards: `fetch_what_prune_keeps::*`.

→ `docs/invariants/memory.md` § "`AutoShardManager::would_shed_copy` is the ONE answer"

## A model is admitted against the card as it stands NOW (2026-09-29)

**`ModelProcessPool::vram_budget_now`** is what `admit_to_gpu` weighs a model against, re-read at admission. Never cache a live condition at startup.

→ `docs/invariants/memory.md` § "A model is admitted against the card as it stands NOW (2026-09-29)"

## Single-source-of-truth helpers — Worker memory: graphics, RAM and the KV cache

A second implementation of any of these is this codebase's most-repeated defect (`architecture.md` § "One invariant, N paths").

- **`DaemonMsg::ReleaseRequestKv`** — a finished request releases its cache, wherever held.
- **`api::process_memory::resident_bytes`** — the LARGEST accounting the platform offers.
- **`ModelProcessPool::reply_channel_closed`** — a closed reply channel is judged in ONE place.
- **`inference::worker_ipc::worker_error_is_fatal`** — did the error destroy device state?
- **`daemon::shard_loader::force_cpu_for`** — `inference.gpu_layers` → `force_cpu`.
- **`daemon::gpu_support::MIN_COMPUTE_CAP` + `local_gpu_is_supported`** — can this card run OUR kernels?
- **`model::auto_manage::vram::ADMISSION_KV_CONTEXT`** — the context admission charges KV for.
- **`ModelProcessPool::footprint_inputs` reads a model's WEIGHTS from its header** (`split::tensor_byte_size` over the tensor table), never from the shards on this node's disk — which is the model only on a full holder (a node holding none priced a 9B at 1346 MB; one holding half charged half; gotcha #803). → `docs/invariants/memory.md` § "A model's weights are its header's".
- **`split::layers_keeping_kv`** (+ `GgufTensorMeta::layers_keeping_kv`, same marker) — the layers charged a KV cache, by admission (`VramFootprintInputs::kv_layers`), the split planner, the loader's KV budget, the scheduler's peer bound (`kv_bytes_per_position_per_layer`) and the cost curve (`process_pool::cost_curve_of` scales `kv_layers` with the segment, or the whole cache lands in its fixed term) alike; a recurrent layer (Qwen 3.5's DeltaNet, `split::layer_is_recurrent`) keeps none (#228).
- **`ModelProcessPool::free_vram_for_admission` + `plan_vram_reclaim`** — reclaim idle models first; **`fits_in_budget`** (every fit verdict, so `serves_on_cpu`) and the planner's ceiling count that reclaim (`reclaimable_vram_mb` / `idle_vram_reclaimable_mb`), or a model held here goes abroad (#125, #129).
- **`should_return_to_gpu` + `worker_should_return_to_gpu`** — asked in `get_or_spawn`, not on a timer.
- **Graphics memory has ONE owner: `ModelProcessPool`.**
- **A worker's growth is weighed by the budget its spawn charged** — `WorkerHandle::holds_gpu_memory`, never `placed_on_cpu_because` (`a_live_workers_growth_is_weighed_by_the_budget_its_spawn_charged`).
- **`model::auto_manage::storage_budget` / `held_disk_bytes`** — the ONE answer to shard storage; counted kinds must be reclaimable.
- **`prune::effective_idle_secs`** — residency is a hard UPPER BOUND on "idle since".
- **An admitted prompt is RECORDED** — claim and reservation go TOGETHER via `forget_request_bookkeeping`.
- **`kv_budget::admit_prompt` + `PrefixCache::release`** — ONE decision per prompt.
- **A prompt of known length is RESERVED** — `set_reserved_positions`; `SWARMLLM_KV_RESERVE=0`.
- **`process_pool::worker_socket_path`** — the ONLY place the IPC socket path is built.
- **`process_pool::end_with_this_daemon`** — on Windows every worker joins the daemon's kill-on-close job; a new spawn path calls it (#153).

→ `docs/invariants/memory.md` § "Single-source-of-truth helpers — Worker memory: graphics, RAM and the KV cache"
