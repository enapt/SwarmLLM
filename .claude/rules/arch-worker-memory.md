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

A resource taken before an `.await` is returned by a guard that releases on DROP; `cancel::unless_cancelled` and `CancelInference`'s `abort_handle().abort()` skip everything after the await, so no `bail_if_cancelled` checkpoint can defend. Guards: **`PendingSpawnCharge`** (`inference::process_pool`), **`InboundForwardSlot`** (`daemon::dispatch`), `ShardDownloadClaim`. Never wrap `pool.generate(..)` in `unless_cancelled` in `local_generate.rs` (gotcha #459).

→ `docs/invariants/memory.md` § "What a task took, a task gives back by being dropped"

## "Used recently" has four answers, and one of them is never written locally

**`model::auto_manage::prune::effective_idle_secs`** is the single answer to "used recently" (routed, served for a peer, the worker's `last_used`, residency as an upper bound). `model_trust.last_request_at` alone is not an answer. The idle-VRAM unload and prune protection both read it.

→ `docs/invariants/memory.md` § ""Used recently" has four answers, and one of them is never written locally"

## A budget is charged by whatever owns the resource

**`SharedState::committed_memory_mb`** is the single answer to "how much memory is committed", asked of `ModelProcessPool` (`vram_committed_mb` / `ram_committed_mb`). Never sum `split_models`. Guard: `a_memory_budget_is_charged_by_the_pool_never_by_the_metadata_map`.

- **Every byte a spawn places is charged to the device it lands on**: a split's card share to the card (`card_share_of_split_mb`), its processor share to the machine (`process_pool::ram_charged_share`; #264); experts in RAM (`CardSplit::ExpertsInRam`) charge the machine only those (`experts_in_ram_mb`). `SpawnCharge` puts both on the worker (`WorkerHandle::host_share_mb`), a departure releases both, a grown range is charged the same (`RangeCharge`, #265).
- **`VramFootprintInputs::host_resident_bytes`** holds what the loader keeps in host memory whatever the device (`split::host_resident`); **`ModelProcessPool::machine_charge`** is the ONE figure a spawn charges the machine (#296).
- **A segment is charged the tables its ends load** (#342): `split::segment_tensor` is the ONE classification; head and token table (`VramFootprintInputs::head_bytes`) go on the main device, and `split_for_card` places layers by EACH ONE'S OWN weights (`LayerBytes::weights`, #343; `head_of_processor_run_mb`).
- **A split fills the share its LOAD takes**: **`hybrid::split_room_bytes`** reads `kv_budget::weights_and_caches_room` (85% of free), never the budget less the forward reserve (#364).
- **A spawn's split is sized by the memory its worker READS** (#365: `DaemonMsg::ReadCardFree`): **`ModelProcessPool::refit_split_to_the_load`**, only toward the card, then `PlaceRange` before anything loads; a plan or growth uses `SplitCard::room`. A/B `SWARMLLM_SPLIT_READS_CARD=0`.
- **A WHOLE card placement is read too (#322)**: **`ModelProcessPool::fit_whole_placement_to_the_load`** splits it where its weights leave less than the spawn's cache (`whole_placement_short_by_bytes`); promotion waits for the shortfall (`SplitByTheLoad`). Never a guest. A/B `SWARMLLM_WHOLE_READS_CARD=0`.

→ `docs/invariants/memory.md` § "A budget is charged by whatever owns the resource"

## The contribution level caps the SWARM's work, not the owner's prompt or memory (2026-09-26, #266)

**`process_pool::Requester`** (`Owner` / `Swarm`) is a REQUIRED argument of `ModelProcessPool::generate` and `forward_for_request`; the worker records it (`KvCacheStore::mark_owner_request`), `cpu_pools::in_phase_pool` reads it.

- The owner's prompt and short checks (`cpu_pools::check_pool`; A/B `SWARMLLM_CHECK_POOL=narrow`) run on `ResourceConfig::owner_prefill_threads`; the swarm's prompts AND its 2-32 position checks at the contribution width on physical cores (`cpu_pools::swarm_check_threads`; A/B `SWARMLLM_CHECK_POOL=owner-only`).
- A one-token decode keeps its calibrated width, widened for the OWNER's step only where processor work is routed experts in RAM (`SplitModel::processor_works_only_on_experts`, #399; A/B `SWARMLLM_OWNER_DECODE=contribution`). A wider pick must beat the default (`cpu_pools::WIDEN_MARGIN_PCT`), is re-checked (#396; A/B `SWARMLLM_DECODE_CONFIRM=0`) and proved in blocks (`cpu_pools::PROVE_BLOCK_STEPS`, #453; A/B `SWARMLLM_DECODE_PROVE=0`).
- **Memory:** `ModelProcessPool::ram_budget_now(requester)` is REQUIRED everywhere a budget is read and picks the cap via `RamBudget::judging` — swarm `inference_ram_budget_mb`, owner `ResourceConfig::owners_ram_budget_mb`; anti-swap headroom binds both. The scheduler names it with `pipeline::requester_for_override`. A drafter is weighed as the request it guesses for (`process_pool::drafter_requester`; #351; A/B `SWARMLLM_DRAFTER_CAP=swarm`). What the OWNER may run is **`vram::owners_model_budget_mb`** (`fits_here`); `node_model_budget_mb` stays the swarm's (#438).
- **The card too** (#483 (a)): a reclaim reads idle time as the ASKER sees it — **`WorkerHandle::idle_secs_for(asking)`**. The swarm's request sees every use; the owner's sees only the owner's own (`owner_last_used`). So the owner's request takes the card from a model only the swarm used lately.
  - `asking` is REQUIRED from `vram_reclaim_candidates` through both reclaims and every planner reader (`fits_in_budget`, `split_budget_for_planning`, `serves_on_cpu`, …).
  - `register_response` REQUIRES the requester.
  - Never mid-reply (`in_use`).

→ `docs/invariants/memory.md` § "The contribution level caps the swarm's work, not the owner's prompt (2026-09-26)"

## A worker's stream has ONE writer at a time (2026-11-12, #430)

The worker's IPC writer is the main loop's (`&mut IpcWriter`). **`IpcWriter::second_handle`** exists for ONE use: `model_worker::PromptProgressOut::open`, whose window (`ReadingPrompt`) BORROWS the writer so the prompt pass can send `WorkerMsg::Progress`. Two writers interleave frames. A forward reports through `KvCacheStore::note_prompt_progress`; a sink that is not open sends nothing.

→ `docs/invariants/network.md` § "A hand-off's prompt read is reported while it runs"

## A subprocess you are waiting for can die instead, and the graphics stack can go mid-run

Code waiting for a subprocess handles it dying instead (`spawn_worker` races `child.wait()` against `listener.accept()`). **`daemon::gpu_support::gpu_runtime_has_failed`** is the single answer to "has the graphics stack stopped working?". Never tell the owner the processor takes over. **`SharedState::inference_outage`** is "can this node run a request" (`can_serve_inference`); shard serving stays up. Guard: `both_inference_outages_withdraw_through_one_predicate`.

→ `docs/invariants/memory.md` § "A worker that dies is not a worker that is slow, and the graphics stack can go mid-run"

## A reply between two forwards is not idle

- Idle time starts at a reply's END (`ResponseGuard::used_until_now`, set only by `register_response`, never a peer's question).
- **`WorkerHandle::in_use` is the single answer to "is this worker in use?"** — a response in flight OR a conversation held between forwards (`kv_holders`, `CONVERSATION_GAP_SECS`); promotion, both reclaims and `models_with_inflight_requests` ask it. Guard: `whether_a_worker_is_in_use_is_decided_in_one_place`. A forward past the prompt pass with no conversation is refused (`model_worker::forward_lacks_its_conversation`).
- **A prompt pass awaiting its first step keeps its model from the IDLE unload only** (#448 (b): `ModelProcessPool::models_awaiting_a_first_step`, read by `try_idle_vram_unload`, up to `kv_cache::PROMPT_PASS_HOLD`) — never `in_use`.
- **A pass whose sender or coordinator left the swarm is released** (#448 (d)): `PassParties` SEEN since the pass began and absent from `SharedState::still_in_the_swarm` for `PASS_PARTY_GONE_AFTER` (300 s) → `release_abandoned_prompt_passes`, from the idle tick before the unload.

→ `docs/invariants/memory.md` § "A reply between two forwards is in use, and a lost conversation is refused"

## A worker that cannot grow gets the spawn's ladder, not a refusal (2026-09-27)

**`ModelProcessPool::grow_worker` is the one place a live worker takes on a new range**: reclaim idle models' graphics memory (`free_vram_for_admission`; `free_vram_for_split`, #355), then the range across card and processor (`charge_segment_as_split`, #265), then — only if `WorkerHandle::in_use` is false — retire the worker so `get_or_spawn` places it afresh. Never retire one mid-conversation (#93).

- A range taking in one charged in the worker's BASE (`takes_in_a_range_charged_in_its_base`) **skips the split rung while idle** (#461). A/B `SWARMLLM_GROWTH_OVER_BASE_RANGE=split`.
- A worker in system memory skips the two card rungs, never the retirement; it retires only where the range would load in its place (`a_fresh_spawn_fits_in_place_of_the_worker`, #348/#349).
- **A peer asking up front is told no where the replacement runs fewer of the held layers on the card** (#461): **`replacement_keeps_the_card`** — of the held layers inside the run, no fewer than now, or than a spawn of just what it holds would place (like with like) — asked ONLY by the up-front weighing (`forward_would_load` → `peer_load_refusal`). Since a far range's processor takes its FIRST layers (#461 (4), `arch-inference.md`), a range widened at its START keeps its held layers on the card wherever the card has room for them; what it still refuses is a range widened at its END, or a card too short for the held layers beside a split's own costs. Never at the forward (`grow_worker`, `forward_replaces_the_worker`): a refused forward bars this node from the re-plan, and an unasked forward is served as before. A/B `SWARMLLM_REPLACEMENT_KEEPS_CARD=0`.
- **A spawn leaves room for the plan's other ranges here** (#481 (5)): `admit_prompt` notes them (`PlannedHere`, its one writer), and a spawn of one — its WHOLE admission (`fits_whole_leaving`), its split sizing AND its refit (#365) — keeps `room_for_the_plans_other_ranges` (their layers at growth's card rate) free, so a split that starts and finishes here can grow its last range. A/B `SWARMLLM_SPAWN_LEAVES_PLANNED=0`.
- **A re-plan is never in its own way** (#481 (5)): a question or this node's admission weighs a worker as busy only BESIDE the request it names (`busy_beside`), and a pass from position 0 (`starts_a_pass`) releases that request's earlier hold (`release_an_earlier_pass`) before `get_or_spawn` — ONLY over layers its earlier pass was sent for here (`WorkerHandle::pass_ranges`): a chain's other range here is the same pass, and retiring its worker is #93 — a re-plan keeps its id (#448).
- **A range is loaded where it was charged**: `DaemonMsg::PlaceRange` (under `spawn_lock`, before any forward) and the loader asks `split::placement_of` — never `--gpu-layers` / `--cpu-moe-layers` alone.

→ `docs/invariants/memory.md` § "A worker that cannot grow gets the spawn's ladder (2026-09-27)"

## A model process holds ONE copy of the output head (2026-11-16, #458)

**`split::OutputHeadLent::of`** (in `model_worker::ensure_model_loaded`, BEFORE a covered range is dropped, single-rank ranges only) lends the head to the next load, taken only through **`split::lent_output_head`** — for a range ending the model and a first segment's borrowed head (`drafter_head`). ⛔ **A borrowing load runs on the LENDER's device** (`split::device_of_the_lent_head`, asked first by `loader::load_device`): a head shared across two `CudaDevice`s is freed on unordered streams (gotcha #1118); never relax to "same card". A/B `SWARMLLM_SHARE_OUTPUT_HEAD=0`.

→ `docs/invariants/memory.md` § "A model process holds one copy of the output head"

## A long prompt asks for the room back (2026-11-15, #454)

A whole card placement fitted by `fit_whole_placement_to_the_load` (`WorkerHandle::room_sized_for`) that refuses a prompt as too long (`Shortfall::TooLong` only → `WorkerMsg::PromptRoomWanted`, gated at the sender) is reloaded fitted to it: **`ModelProcessPool::make_room_for_a_longer_prompt`**, only where that keeps MORE and no other request uses it (`in_use_beside`), under the spawn lock. Kept 30 min in `LongPrompts`, read ONLY by `whole_placement_kv`; admission is never asked with the longer charge; the reload keeps the forward's buffers beneath the line (`SplitCard::for_one_long_prompt`). A/B `SWARMLLM_LONG_PROMPT_RELOAD=0`.

→ `docs/invariants/memory.md` § "A long prompt asks for the room back"

## A worker serves any run inside a range it loaded (2026-11-13, #449)

**`split::view::view_of`** builds a `SplitModel` over a run inside a loaded range (weights shared; own KV key, graphs, masks; no guess head, block drafter or stand-in prior), filed by `ensure_model_loaded` — never a view of a view; **`drop_views_of`** removes views with their copy. Refused for tensor parallelism and layers reading another layer's cache (Gemma 4 E2B/E4B).

- **`WorkerHandle::serves_segment(segment, RunShape)`** is the ONE answer at every charge, load and residency site; `RunShape` is REQUIRED from the forward through `get_or_spawn`, `grow_worker` and both charges; `segment_is_charged` is test-only.
- **What the daemon did not charge is never loaded**: a forward for a contained run carries the charged range (`IpcForward::served_from`); the worker loads THAT range or refuses (`ServiceUnavailable`).
- **At most `view::MOST_VIEWS_A_COPY` (8) views**: the least recently used one not in use gives way — in use = a conversation holds it AND a forward ran within `view::VIEW_BUSY_FOR` (30 s); all in use → refused.
- The planner credits it through **`HeldRange::serves_slices`**: this node by the switch, a peer only where it advertises **`features::SERVES_SLICES`** (bit 42, not in `ALL`). A/B `SWARMLLM_ELASTIC_SPANS=0` (worker, daemon, advertisement together).

→ `docs/invariants/memory.md` § "A worker serves any run inside a range it loaded"

## A split's stored prompt lives in the worker's prefix cache (2026-10-09, #10)

Chain-keyed entries in the worker's `PrefixCache` (`lookup_chain` / `insert_chain_from_kv`), never a second store (gotcha #440). In `handle_forward`: look up BEFORE admission, admit restored + computed positions, hydrate AFTER; a mismatch is `PromptCacheMiss`. **Planned at the final piece, copied once the answer is SENT** (#433: `plan_chain_insert`, `finish_chain_insert`) — never before the first token. A/B `SWARMLLM_SPLIT_PROMPT_SAVE=before-answer`.

→ `docs/invariants/memory.md` § "A split's stored prompt lives in the worker's prefix cache"

## A prompt pass read in pieces is admitted once, for its whole span (2026-10-10, #171)

In `handle_forward` the FIRST piece of a `prompt_span` clears, restores and is admitted for `span.end - index_pos` positions (gotcha #447); a later piece is admitted nothing and continues exactly where the cache ends. #10's store runs at the FINAL piece only.

→ `docs/invariants/memory.md` § "A prompt pass read in pieces is admitted once"

## A split's local segments are admitted together, before the first forward (2026-10-13, #263)

**`ModelProcessPool::admit_prompt`** (`DaemonMsg::AdmitPrompt`), asked once from `forward_through_segments` when a peer takes part, admits the prompt over every range the plan gives this node (`model_worker::admit_prompt_over`). An admitted segment clears with `clear_request_caches` (never `clear_request`) and skips its own admission; a refusal carries `layers_that_fit`. **Up front, nothing is evicted** (`Admitting::UpFront`, #321): a prompt fitting only once saved prompts go is `Admitted::LeftToTheForwards`. Bounded by `UP_FRONT_ADMISSION_WAIT`. A worker the forward will replace is not admitted on (`forward_replaces_the_worker`, #350).

**Its PEERS are asked at the same moment (#252 (b))**: `PromptAdmissionAsk` (`pipeline::prompt_admission`, `features::PROMPT_ADMISSION`).

- The asked worker only WEIGHS — `Admitting::OnlyWeigh`, a REQUIRED argument of `admit_prompt_over`: no reserved positions, promise, release or cleanup, under a fresh correlation id (`weigh_peer_prompt`), in a slot that does not stamp `last_used` (`register_question`). Never let a peer's question write worker state.
- ONE exception: a "go ahead" loads the asked ranges (`dispatch::prompt_admission::may_load_ahead`: valid, every part held, ≤ 2 ranges, none overlapping) via `load_ahead_for_a_peer` → `DaemonMsg::LoadRange`, through `get_or_spawn` as the swarm's tenant.
- With NO worker, or none holding the first range, it weighs the LOAD first, charging nothing (`ModelProcessPool::peer_load_refusal`, #345/#349).
- A "no" holds the peer to `layers_that_fit` on the re-plan (`note_peer_prompt_refusal`), never bars it.

→ `docs/invariants/memory.md` § "A split's local segments are admitted together", § "A split's peers are asked about room before the pass too"

## A model loaded for ANOTHER model's request is a guest (2026-09-28)

**`process_pool::Tenancy` is a REQUIRED argument of `get_or_spawn`.** A `Guest` (the drafter, `ModelProcessPool::draft`) takes only memory that stands free: no reclaim, split, promotion, growth or notice; charged the cache of the reply it guesses for (`Tenancy::kv_charge`), not `ADMISSION_KV_CONTEXT` (#275). `dsd.rs` reads ahead only once `holds_segment`.

- Whether it WOULD load is asked without loading — **`ModelProcessPool::guest_fit`**, the spawn's own arithmetic (`card_verdict` / `ram_verdict`), beside the plan's layers (`guest_fit_beside`; #275, `arch-scheduling.md`); in system memory AFTER the plan's share (`RamBudget::after_loading`, #351). It says where it lands (`GuestFit`: `Refused` / `OnCard` / `InSystemMemory`) so placement is decided once (#373).
- Weighed on this node's FIRST range as its spawn charges it (`local_layers_leaving`, #376); A/B `SWARMLLM_DRAFTER_ROOM_FIRST=average`.
- **On the card it leaves its target room for its conversations**: **`guest_fits_the_card`** — its estimate plus **`room_a_guest_leaves_mb(target)`** — is the ONE check, asked by `guest_fit` and the spawn (`Tenancy::Guest { leaves_mb }`); `draft`'s `beside` is REQUIRED. The planner leaves the same (`room_a_guest_leaves_per_card_layer_mb` per added layer, #3 (j10)).

→ `docs/invariants/memory.md` § "A model loaded for another model's request is a guest (2026-09-28)"

## A fan-out to every worker is bounded, and the two waits mean different things

**`ModelProcessPool::notify_every_worker` is the one fire-and-forget fan-out** (`cancel_request`, `release_request_kv`), awaited inline from the dispatch loop (gotcha #74). The writer LOCK timing out stands down at `debug!`; the SEND timing out marks the worker `dead`. Sends run concurrently.

→ `docs/invariants/memory.md` § "`ModelProcessPool::notify_every_worker` — a fan-out to every worker is bounded"

## A conversation that is over gives its room to one that is not (2026-10-08)

**`KvCacheStore::release_finished_conversations`** runs before a prompt is refused (`ensure_room_for_prompt`) and before a growth claim (`claim_room`, `claimant` REQUIRED): caches silent for `FINISHED_CONVERSATION_AFTER` (= `process_pool::CONVERSATION_GAP_SECS`) go first; a segment served for another computer is never released by its coordinator. The worker runs `cleanup_expired` on its 30 s tick (guard `a_workers_kv_store_is_swept_by_the_worker_that_builds_it`). At a REQUEST's end (never an attempt's, #749) the coordinator tells every peer (`router::release_request_on_peers`, `CancelInference`); a worker drops a cancelled request's cache (`release_caches_of_cancelled`, #238).

- **A cache whose prompt pass ran here is held past the TTL until its first step** (#448: `note_forward_ran`), up to `kv_cache::PROMPT_PASS_HOLD` = `remote_generate::MEASURED_FIRST_TOKEN_MAX` (one figure).
- **A peer's question is never weighed against that request's own caches**: it names the request (`IpcAdmitPrompt::weighs_for`, via `weigh_peer_prompt`'s REQUIRED `for_request`) and `admit_prompt_in` takes off `bytes_a_prompt_pass_replaces` (`a_prompt_pass_clears`) for `OnlyWeigh` only. A re-plan keeps the request id.

→ `docs/invariants/memory.md` § "A conversation that is over gives its room to one that is not"

## A card's free memory is read after a synchronize (2026-09-26)

**`kv_budget::device_free_and_total_bytes`** synchronizes the stream before `mem_get_info` and adds `cuda_pool::reusable_bytes`; every budget decision reaches the card through `SplitModel::kv_budget_now`. Guard: `the_cards_free_memory_is_read_after_a_synchronize`; A/B `SWARMLLM_KV_DEVICE_SYNC=0`. `reply_reserve_positions(max_tokens)` is a REQUIRED argument of `ensure_room_for_prompt`; a refusal caused by OTHER conversations says wait (`other_conversations_hold_the_room`).

→ `docs/invariants/memory.md` § "A card's free memory is read after a synchronize"

## A cache is charged to the memory it lives in (2026-10-13, #262)

**`KvHome`** names the memory, **`KvOccupancy::allocated_in(home)`** is the only figure a budget is weighed against (`allocated_bytes` counts every device — report it, never compare it), **`SplitModel::kv_homes`** lists every memory a model charges; the growth claim (`claim_room(home, …)`) and admission (`admit_prompt_in`) ask each. A split's processor side has its own budget (`host_kv_budget_bytes`: `--kv-budget-bytes`, else available memory less the margin). **A saved prompt is charged the same way** (gotcha #1105): **`PrefixCache::bytes_on(home)`** is the only snapshot figure a budget reads, `release(needed, home)` counts only what THAT memory gets back, one external figure per memory (`KvOccupancy::external_in`, set only by `model_worker::reserve_beside_live`) — never `bytes_in(SnapshotHome::BesideLive)`.

→ `docs/invariants/memory.md` § "A cache is charged to the memory it lives in"

## A card's KV budget is kept whatever the context setting (2026-10-09)

The loader records the card's KV budget with or without `inference.max_seq_len_override` (#440). A budget refuses one request that does not fit NOW; it never shortens a context.

→ `docs/invariants/memory.md` § "A card's KV budget is kept whatever the context setting"

## A retired card worker's memory is waited for before the next load reads the card (2026-11-14, #3 (j13))

`unload_model` (a LIVE worker only) waits after reaping until three quarters of its card charge reads free (`await_card_memory_back`: 250 ms polls, 5 s bound — Ollama's `waitForVRAMRecovery`) or the rise has settled, THEN releases the budget. **Only a worker asked to load is owed its charge** (`WorkerHandle::asked_to_load`, set BEFORE a request or load ahead is sent; `DepartedWorker::card_mb_it_frees` is 0 without it); a new path that sends loading work sets it first. A/B `SWARMLLM_CARD_MEMORY_BACK=0`.

→ `docs/invariants/memory.md` § "Graphics memory has ONE owner" (**And a retired card worker's memory is waited for**)

## The daemon holds no CUDA context — only workers do (2026-10-05)

**`gpu_support::describe_local_gpu`** (driver API: `cuInit`, name, total) is how the daemon learns its card. Never llama.cpp's device list or `Device::cuda_if_available` in the daemon (each leaves a 137 MiB context). Free memory is read live (`vram::query_gpu_vram_free_mb`); `GpuInfo` has no free field.

→ `docs/invariants/memory.md` § "The daemon holds no CUDA context (2026-10-05)"

## nvidia-smi is asked through one bounded helper (2026-10-05)

**`vram::nvidia_smi`** (a `BoundedCommand`: 10 s, never two copies at once) is the only way to run `nvidia-smi`. Guard: `nvidia_smi_is_asked_only_through_the_bounded_helper`. A copy stuck 60 s (`DRIVER_NOT_ANSWERING_AFTER`, #802) makes `vram::graphics_driver_not_answering` true: new workers go to the processor (`CpuReason::DriverNotAnswering`, transient). **Its wait never holds a Tokio runtime worker** (`executor::without_starving_the_runtime`, #244; test `a_slow_bounded_command_does_not_hold_the_runtime_worker`).

→ `docs/invariants/memory.md` § "nvidia-smi is asked through one bounded helper (2026-10-05)"

## A card full of our own model is not pressure (2026-10-12, field report #035)

Prune's soft-unload narrows a model only when our CHARGED memory (`vram_committed_mb`) exceeds `compute_vram_budget`, and only the largest model NOT in use (`models_with_inflight_requests`), priced by its own charge (`vram_committed_by`) — never a fraction of the card's total nor the whole file. Choice: `prune::card_room_to_give_back`. **Nor a model used within `NARROW_ONLY_AFTER_IDLE_SECS`** (2 × `cuda_pool::IDLE_TRIM`, by `effective_idle_secs` — `prune::held_back_from_narrowing`; #3 (j8)). Idle unload asks `prune::stays_loaded_while_idle`; regional demand buys no reprieve.

→ `docs/invariants/memory.md` § "A card full of our own model is not pressure"

## A memory pin ends, and it ends in `cpu_reason` (2026-10-31)

**`process_pool::CpuPins`**: a card that FAILED a model (`classify_worker_error`) holds it on the processor `PIN_HOLD_FIRST` (120 s), doubled up to 30 min while the card keeps failing it SOON after (a worker that ran `PIN_CLEAN_RUN`, 3 min, starts over); then `cpu_reason` stops reporting it. Never end a pin in one path, nor wait for another worker to leave: the worker that ran out is evicted BEFORE its pin is taken. `ArchitectureTooOld` never ends.

→ `docs/invariants/memory.md` § "Graphics memory has ONE owner" (**A memory pin ENDS**)

## A card that stalls is given less work, not more (2026-09-28)

**`inference::card_pace::CardPace` is the one answer to "may this worker start another generation on its card now"**: a device-bound step ≥ 2 s halves the ceiling. The gate sits in the `Generate` arm BEFORE the batched/sequential choice; a new device-bound step is timed into it, a new generation path gated by it. Refusal = `LocalMemoryUnavailable`. A/B `SWARMLLM_CARD_PACE=0`.

→ `docs/invariants/memory.md` § "A card that stalls is given less work (2026-09-28)"

## A hot computer's own work is slowed, never refused (2026-11-11, #440 step 4)

**`inference::pace::Step`** is the one place a worker's work is paced, opened at the top of each forward entry point (`SplitModel::forward_inner_impl`, `forward_batch`, `forward_tp_phase`, `BlockDrafter::draft`, `MtpGuesses::guesses`); only the OUTERMOST on a thread pauses. While the daemon holds back for heat (`host_heat_hold_now` → `ModelProcessPool::pace_steps` → `DaemonMsg::Pace`; a spawn starts at `pace::WORKER_PACE_ENV`) it synchronizes the device and owes `HOT_PAUSE_PERMILLE` of the step (≤ 5 s, 50 ms slices a cancel cuts), **slept in pieces of `PAUSE_PIECE` (250 ms), never after every step** (A/B `SWARMLLM_PACE_PAUSE_FROM_MS=2`); rest between steps pays what is owed (`after_rest`). A new entry point opens a `Step`; work no reply waits for does not run while paced (`pace::work_ahead_may_run`). **A timer judging a step's length reads `pace::Stopwatch::worked()`** (`CardPace`'s decode tick does). Never fewer threads.

→ `docs/invariants/memory.md` § "A hot computer's own work is slowed, never refused"

## Machine memory is read in one place, under the cgroup's limit (2026-10-09)

**`daemon::machine_memory::machine_memory()`** is the only reading of how much memory this process may use: host figures under its cgroup's limit, available = limit less what the kernel could not reclaim (`memory.current` − `active_file` − `inactive_file` − `slab_reclaimable`, LXCFS's `MemAvailable`) — never the kubelet's working set nor `sysinfo`'s cgroup "free". Guard: `machine_memory_is_read_in_one_place`.

→ `docs/invariants/memory.md` § "Machine memory is read in one place, and a container's limit counts"

## A card's memory pool keeps what the worker frees, until the worker is idle (2026-09-28)

**`split::loader::load_device` is the one place a worker picks the card** and raises the pool release threshold (`cuda_pool::keep_freed_memory`); `model_worker::run_worker` trims after `cuda_pool::IDLE_TRIM`. Any "free for this process" reading adds `cuda_pool::reusable_bytes`. Guard: `every_split_model_reaches_the_card_through_load_device`; A/B `SWARMLLM_CUDA_POOL_KEEP=0`.

- **A load hands back what it freed as it ends** (`model_worker::give_back_what_the_load_freed`, before it measures its cost): a refused guess head's 994 MB held in the pool read as another program's to the daemon and the 27B's tail growth fell 261 MB short (2026-11-19). A/B `SWARMLLM_TRIM_AFTER_LOAD=0`.
- **The driver keeps its margin too** (#385): the card's one reading asks **`cuda_pool::hold_driver_margin`**. A/B `SWARMLLM_CUDA_POOL_MARGIN=0`.
- **A refusal is asked again ONCE** (#367 (d) (3a), PyTorch's `mallocAsync`): vendored **`CudaDevice::ask_again_after_a_refusal`** — every allocation of the device's own buffers, `clone_htod`'s included, never cudarc's allocating calls directly — synchronizes, trims the pool and asks again on `CUDA_ERROR_OUT_OF_MEMORY`, never inside a capture. A/B `SWARMLLM_CUDA_ASK_AGAIN=0`; fault `SWARMLLM_FAULT_CARD_OOM=<n>` (`SWARMLLM_FAULT_CARD_OOM_MIN_BYTES`).
- **A forward still refused runs ONCE more after the saved prompts go** (#367 (d) (3b)): `forward_inner_impl` → `forward_once`; where **`executor::worth_running_again`** holds (a card, `permanent_gpu_failure` → `OutOfMemory`, caches in place: `KvCacheStore::positions_in_place`) **`KvCacheStore::let_saved_prompts_go`** (0 → no rerun) and `forward_once` again. Never a loop, never a conversation's cache.
- **A batched step too** (#367 (d) (3c)): a refused step puts every item's caches back (**`executor::put_back_where_the_step_began`**, the ONE error arm of both paths), then `forward_batch` — where `worth_running_again` holds for EVERY item — lets saved prompts go and reruns, else two halves by the same rule; a half still refused fails the tick, no request is shed.
- A step that cannot be put back leaves its entry EMPTY; a forward on one is refused (`KvCacheEntry::holds_every_layer`, both paths), never indexed (gotcha #1047). A/B `SWARMLLM_FORWARD_RERUN=0` (both).

→ `docs/invariants/memory.md` § "A card's memory pool keeps what the worker frees (2026-09-28)"

## Under WSL2 a card's saved prompts live in host RAM (2026-10-13, #257)

**`prefix_cache::SnapshotHome::for_device`** is the one answer to where a snapshot is kept: host RAM for a card reached through WSL2 (`cuda_pool::card_is_reached_through_wsl`), beside the live cache elsewhere. `model_worker::snapshot_room` is REQUIRED by both inserts; host RAM is bounded by `kv_budget::host_snapshot_budget`; a card's copy that does not fit whole goes to host RAM (`room_off_the_card_if_more`, #321); the card's budget and `release` see only `BesideLive`. Judge a change by `pool_reserved_mb` rises, not tok/s on a fresh boot (gotcha #834). A/B `SWARMLLM_PREFIX_CACHE_HOME=card|host`.

→ `docs/invariants/memory.md` § "Under WSL2 a card's saved prompts live in host RAM"

## The owner is told when fresh card memory has become slow (2026-10-01)

**`cuda_pool::probe_once`** times 16 fresh 4 MB allocations once per worker at `load_device`, never per request; a probe ≥ 0.5 s reaches `ModelProcessPool::slow_card_notice` (`WorkerMsg::CardAllocationProbe`), WSL2 only, at most every 12 h. It changes nothing the node does.

→ `docs/invariants/memory.md` § "The owner is told when the card has become slow to hand out memory (2026-10-01)"

## A lone decode stream is never held for a batch (2026-09-29)

**`process_pool::collection_target`** waits for a batch only when ANOTHER request is decoding on the same model (`ActiveStreams`). Never reintroduce a wait a lone stream pays.

→ `docs/invariants/memory.md` § "A lone decode stream is never held for a batch (2026-09-29)"

## The download pass fetches only what prune would keep (2026-10-06)

**`AutoShardManager::would_shed_copy` is the one answer to "would prune shed this copy?"** — prune asks it of what it holds; EVERY automatic fetch path asks it before fetching, at the disk pressure AFTER the fetch (`select_within_budget`; `complete_pending_shard_fetches`, which DROPS the entry). An entry the origin cannot serve (`pending_fetch_can_proceed`) is dropped back to the peers; a finished origin download leaves `shard_p2p_failed` (gotcha #797). File decisions read DISK pressure, never graphics memory; the filesystem's last 10% is never ours (`FREE_DISK_RESERVE_PCT`). Guards: `fetch_what_prune_keeps::*`.

- **The target moves while a part does not (#286)**: prune sheds against `shed_target_replicas` (`target_hold`), the pass reads the target as it is; never the reverse. A part of a model held whole here (or all but one in eight parts) is kept below urgent (`coverage::completes_model_here_counting`) and its missing parts fetched past the target (`parts_completing_model_here`, #320).
- **Carrying (#231)**: `would_shed_copy` keeps a copy the model would fall short without (`coverage::copy_carries_model`); `coverage::carrier_for` chooses the one carrier, fetched past the target and the ring. A new rule about which copies matter goes in `coverage.rs`, asked by both passes. Guards: `coverage::tests::*`.
- **No part for a computer that does not fetch (#400)**: `coverage::fetches_parts`; **`room_to_fetch_bytes`** is the ONE disk reading a carrier or chain segment is planned by (never `disk_available_mb`), 0 for it and for one at its storage limit — it reads **`NodeCapability::storage`** (`StorageRoom`, `prune::spare_for_any_room`, `RoomAsk::advertised`, never the owner's use, #389 (a)), built ONLY by **`auto_manage::storage_room`**; the ring takes a computer only with `RING_ROOM_MB` free (**`coverage::takes_ring_places`**).
- **Near chains**: for an asked-for model no NEAR machine (≤ 30 ms) can run whole, every near machine plans one chain alike (`coverage::near_chain`; `Planner::reach`, #353; `Planner::{compute_us, hop_into_us}`, #356; `worth_moving`, #318; `MOST_CARD_SEGMENTS`) and fetches its segment (`parts_this_node_plans`); prune keeps it (`copy_in_near_chain`, below urgent; #876). Where they cannot carry it, or one runs it whole only beside its processor (`Planned::OneRunsItWhole`, `ACardHoldsItWhole`, #318), the chain is planned over every machine in scope (`Reach::Far`, #346), its parts kept at ANY pressure (`ChainPart`); **carrying stands down while any chain is planned** (`near_chain::chain_planned_for`; `scheduler::layers_carried`). Layers loaded outside this node's segment of a chain every machine HOLDS (`chain_held_by_its_machines`) go back once idle past the conversation gap (`loaded_outside_near_segment`). A/B `SWARMLLM_NEAR_CHAIN=0`; rig `examples/near_chain_rig.sh`.
- **A model nobody runs is given back in two steps, never fetched (#389 (d), #452)**: **`AutoShardManager::unused_verdict`** (`auto_manage/unused.rs`, `unused::judge`) — never the owner's (`SharedState::owner_keeps`; a dashboard FETCH, `owner_fetched`, lapses after **`unused::OWNER_FETCH_LAPSES_SECS`** (30 days) with storage pressed, `unused::owner_fetch_lapsed`; the rest reads `owner_chose`), asked for, a reference model or a split's drafter. THREE POOLS:
  - **The swarm's use decides what is KEPT**: run here by nobody AND served by no connected holder (**`unused::SwarmUse`**, from advertised `NodeCapability::models_served`, built ONLY by **`unused::models_served`**, never the owner's use, #389 (a)); `watch_model_use` runs every tick OUTSIDE the enabled gate. A holder that cannot say (`ModelServed::served_since`), or none advertising, gives back nothing.
  - **Quiet for the quiet window it COOLS** (**`unused::quiet_window_secs`**, `unused::cools`): every copy of each part goes but TWO (`coverage::COOLED_COPIES`; **`coverage::keeps_a_cooled_copy`** ranks live holders for the MODEL), none fetched, the FIRST kept for the week (`coverage::keeps_the_first_cooled_copy`), the second until a room another model's part asks and an origin could return it (`cooled_copy_kept_because`, `KEPT_NO_ORIGIN`; `prune::room_order`). **Quiet for `USE_WINDOW_SECS` (7 days, any pressure) every part goes**, but the two a cooled part keeps where no origin could return them (`unused_copy_kept_because`). `unused::Verdict` says which pool decided.
  - **This computer's use keeps this computer's copy** (never advertised). **What people run outside the swarm decides only what is TRIED** (`popular_outside_the_swarm`): fetched only where this node holds none — never one the swarm tried and nobody ran (given back here or by `GAVE_BACK_WITNESSES` (2) others within `GAVE_BACK_FOR_SECS`: `note_gave_back_unused`).
  - **A model no computer in reach holds stays usable and listed**: the request is answered `ModelIncompleteInSwarm`, counted as asked for, wakes the pass (**`SharedState::no_route_error`**, never `NoModelLoaded` for a known model; `unused::held_before`). Asked in `copy_kept_because` and the prune pass (`PruneCandidate::nobody_runs` / `cooled`); the advertised room asks `cools_whoever_asked`. Content is never judged (#389 (c)): `unchosen_uncensored_model` only keeps a model from being pre-selected.
- **A drafter a split lacked (#134), or a larger one (#331)**: a split that guesses here records its target (`engine_drafter::note_split_wanting_a_drafter`) and the pass fetches **`engine_drafter::drafter_to_acquire`** — what `chosen` would pick were it held, of the target's family, ≤ `MOST_DRAFTER_BYTES_TO_FETCH` (2 GiB), trusted as adoption requires (`coverage::drafter::may_fetch_as_a_drafter`), once every likely candidate's header is here, never another copy of what is held (`HeldDrafter::beaten_by`, #370) — past the target and the ring (`parts_of_a_drafter_to_fetch`); prune keeps below urgent (`copy_drafts_for_a_split`) what **`DrafterToAcquire::keeps`**: the held one on EVERY step and the one being fetched — while splits want one (`DRAFTER_WANTED_FOR`, 7 days).

**Holes first (2026-10-25)**: a part no computer holds or is fetching, of a model somebody asked for that the swarm holds part of (**`scoring::fills_a_hole`**), is chosen before any further copy whatever its score — the sort and the first tier of `select_within_budget`, never a multiplier.

- **A full node makes room for one — and for its planned part, and a part too few hold (#371)**: `ShardCandidate::asks_room` (**`manager::RoomFor`** `Hole` > `Placed` > `Scarce` — `scoring::is_scarce`, `AutoShardManager::worth_trying`, #452, `coverage::only_a_split_runs`); strongest tier first, and only a HOLE that did not fit, with no hole chosen, stops the tiers below.
- The pass notes the room the FIRST that asks lacks (`scoring::room_lacked_after`, `first_part_wanting_room`; `AutoShardManager::room_wanted`, written by it alone every cycle, `budget == 0` included). While it stands `copy_kept_because` sheds a copy kept ONLY by the replica count or region rule if **`prune::spare_for_room`**: above the room's floor of live holders (`floor_for_room`; `Scarce` also `a ≥ b + 2`, #401), this node not among the first that many by rendezvous (`coverage::kept_by_rank`), not of the wanted part's model or a model with holes, not loaded, not completing a model here unless the wanted part's model is asked for MORE (**`coverage::demand_for`**, `RoomWanted::demand`), not `SharedState::owner_chose` (#400; `KEPT_WHOLE_HERE`) — up to `PruneCandidate::to_make_room`, and only where the pass could make ALL of it (**`prune::room_this_pass_could_make`**, #452). The owner is told at most every 6 h (`tell_owner_storage_is_full`). Never past a rule that keeps a copy for what it does (carrying, near chain, drafter, privacy, pins).

**A download nobody asked for waits while the connection is metered (2026-10-15)**: **`SharedState::metered_hold`** (`daemon::metered`) is the one answer, asked just before the start by every path that starts one on its own; setting `auto_manage.on_metered_connection`. Repairs, the owner's downloads and running transfers are not paused. Guard: `every_automatic_download_asks_whether_the_connection_is_metered`. It holds back uploads to peers too (`arch-network.md`, #277).

→ `docs/invariants/memory.md` § "`AutoShardManager::would_shed_copy` is the ONE answer"

## A model is admitted against the card as it stands NOW (2026-09-29)

**`ModelProcessPool::vram_budget_now`** is what `admit_to_gpu` weighs a model against, re-read at admission, and what promotion weighs (`promotion_weighed`; #260). Promotion weighs every worker placed short of the card (**`WorkerHandle::short_of_the_card`**: the processor OR a split) with the split's own share given back (`own_card_mb`), never above `ceiling_mb`. **Never for a request it serves as a slice of what it holds** (`promotion_keeps_what_it_holds`; `segment` REQUIRED in `worker_should_return_to_gpu`, #3 (j12)). **A plan reads it through `card_budget_for_planning`** — live where a spawn could follow, one reading per `PLANNING_CARD_READING_FOR` (#250) — and weighs a fresh split's processor share against the requester's RAM cap (`fresh_run_layers_for_planning`, `idle_ram_reclaimable_mb`, one predicate with `free_ram_for_admission`: `ram_reclaim_eligible`). Never cache a live condition. **Peers are told the same room** (#447): **`ModelProcessPool::gives_way`** puts on each `ResidentModelLayers` what its worker would give — only where admission would reclaim it, idle ≥ `GIVES_WAY_AFTER_IDLE_SECS` (30), not a drafter — the card's charge and its share of the machine room's gain (`headroom_after_reclaim`).

→ `docs/invariants/memory.md` § "A model is admitted against the card as it stands NOW (2026-09-29)"

## Single-source-of-truth helpers — Worker memory: graphics, RAM and the KV cache

A second implementation of any of these is this codebase's most-repeated defect (`architecture.md` § "One invariant, N paths").

- **`DaemonMsg::ReleaseRequestKv`** — a finished request releases its cache, wherever held.
- **`api::process_memory::resident_bytes`** — the LARGEST accounting the platform offers.
- **`ModelProcessPool::reply_channel_closed`** — a closed reply channel is judged in ONE place.
- **`inference::worker_ipc::worker_error_is_fatal`** — did the error destroy device state?
- **`daemon::shard_loader::force_cpu_for`** — `inference.gpu_layers` → `force_cpu`.
- **`daemon::gpu_support::MIN_COMPUTE_CAP` + `local_gpu_is_supported`** — can this card run OUR kernels?
- **`model::auto_manage::vram::ADMISSION_KV_CONTEXT`** — the context admission charges KV for; `vram::KvCharge` says which (a guest's is its reply's).
- **`ModelProcessPool::footprint_inputs`** reads WEIGHTS from the model's header (`split::tensor_byte_size`), never this node's shards (gotcha #803), through `split::HeaderCache` (`segment_geometry`; #360). → `docs/invariants/memory.md` § "A model's weights are its header's".
- **`split::kv_layers`** (+ `GgufTensorMeta::layers_keeping_kv`) — the layers charged a KV cache, read by `kv_budget::KvLayout` and the scheduler's peer bound (`KvLayout::growth_on_peer`, #281 (4)); a recurrent layer (`split::layer_is_recurrent`) keeps none (#228).
- **`kv_budget::KvLayout` → `KvGrowth` / `KvPrice`** — the ONE price of a conversation's caches, per layer off the header: the loader's budget, admission, the snapshot plan and the reported context read `KvGrowth`; before a load, admission's footprint, `process_pool::cost_curve_of` and the split read `VramFootprintInputs::kv` (`KvPrice::reading(reader, context)`, #281 (3)). Never restate `layers × heads × head_dim × positions`. The head-room guard reads the BUFFERS (`KvCacheStore::bytes_a_forward_adds`). → `docs/invariants/inference.md` § "A windowed layer is charged as its window".
- **`ModelProcessPool::free_vram_for_admission` + `plan_vram_reclaim`** — reclaim idle models first; **`fits_in_budget`** (every fit verdict, so `serves_on_cpu`) and the planner's ceiling count it (`reclaimable_vram_mb` / `idle_vram_reclaimable_mb`), or a held model goes abroad (#125, #129). A model that will not fit whole gets idle models' card memory for its SPLIT (`free_vram_for_split` + `plan_vram_reclaim_for_split`), never from a drafter (`split_reclaim_eligible`) — but for a WHOLE placement (`SplitFor::Whole`) of the one model it guessed for (`WorkerHandle::guesses_only_for`, #442); every split planner reads `split_budget_for_planning` with the same `SplitFor` (#355). On the processor the bound counts `idle_ram_reclaimable_mb` (`RamBudget::headroom_after_reclaim`, #276), never beside a guest; **an unload is counted as the spawn will READ it** (`RamBudget::after_unloading`, #372), and every reclaim reader goes through `headroom_after_reclaim`. A/B `SWARMLLM_RECLAIM_HEADROOM=whole`.
- **`should_return_to_gpu` + `worker_should_return_to_gpu`** — asked in `get_or_spawn`, not on a timer.
- **Graphics memory has ONE owner: `ModelProcessPool`** — every admission that says yes charges it, a split included (`card_share_of_split_mb`, #240).
- **A worker's growth is weighed by the budget its spawn charged** — `WorkerHandle::holds_gpu_memory`, never `placed_on_cpu_because` (`a_live_workers_growth_is_weighed_by_the_budget_its_spawn_charged`).
- **`model::auto_manage::storage_budget` / `held_disk_bytes`** — the ONE answer to shard storage; counted kinds must be reclaimable.
- **`model::auto_manage::disk_space_for`** — the ONE reading of the disk a path is on (#334 (4); guard `a_disk_is_read_only_through_disk_space_for`): `statvfs` of the nearest existing ancestor on Linux, never sysinfo's disk list; the path is followed through any link first (`canonicalize`). **`auto_manage::models_disk`** is the disk the MODELS are on, asked of `models/` (`models_root`), never the data directory (#368; guard `the_models_disk_is_asked_of_the_models_directory`).
- **`prune::effective_idle_secs`** — residency is a hard UPPER BOUND on "idle since".
- **An admitted prompt is RECORDED** — claim and reservation go TOGETHER via `forget_request_bookkeeping`.
- **`kv_budget::admit_prompt` + `PrefixCache::release`** — ONE decision per prompt.
- **A prompt of known length is RESERVED** — `set_reserved_positions`; `SWARMLLM_KV_RESERVE=0`.
- **`process_pool::worker_socket_path`** — the ONLY place the IPC socket path is built.
- **`process_pool::end_with_this_daemon`** — on Windows every worker joins the daemon's kill-on-close job; a new spawn path calls it (#153).
- **`logging::forward_worker_output`** — while the log has a file (`logging::pipes_worker_output`), a worker's output is piped through the daemon: ONE writer, so the file can rotate; a new spawn path pipes through it too. → `docs/invariants/memory.md` § "A model process's output reaches the log file through the daemon"

→ `docs/invariants/memory.md` § "Single-source-of-truth helpers — Worker memory: graphics, RAM and the KV cache"
