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

**Every byte a spawn places is charged to the device it lands on** — a card/processor split's card share to the card (`card_share_of_split_mb`) AND its processor share to the machine (`process_pool::ram_charged_share`, through the same `admit_to_cpu_as` / `record_cpu_kv_budget` as a processor placement; #264) — for experts kept in RAM (`CardSplit::ExpertsInRam`), the card less those experts and the machine those experts alone (`experts_in_ram_mb`). `SpawnCharge` carries both onto the worker (`WorkerHandle::host_share_mb`) and a departure releases both. A later range grown as a split is charged the same way, each share recorded on the range (`RangeCharge`, #265). **What the loader keeps in host memory whatever the device** (`split::host_resident`: Gemma 4 E2B/E4B's 1.9 GB per-layer table) is `VramFootprintInputs::host_resident_bytes` — a card's estimate holds `card_weight_bytes`, the processor's everything — and **`ModelProcessPool::machine_charge`** is the ONE figure a spawn charges the machine (the `ram_charged_share` priced, plus those tensors beside any card placement), asked by the spawn, `fresh_run_layers_for_planning` and the card bound (#296). **A segment is charged the tables its ends load** (#342): `split::segment_tensor` is the ONE classification (the loader's `segment_weight_bytes` and `process_pool::segment_weights`); the token table and output head (`VramFootprintInputs::head_bytes`) go on the main device, so `split_for_card` pays them first and places the card's layers by EACH ONE'S OWN weights (`LayerBytes::weights`, #343 — never the segment's average: the card takes the first layers, and Q4_K_M gives a model's ends more bits), a processor run ending the model leaves its head to the card's share (`head_of_processor_run_mb`), and a fresh run is weighed at BOTH ends.

→ `docs/invariants/memory.md` § "A budget is charged by whatever owns the resource"

## The contribution level caps the SWARM's work, not the owner's prompt or memory (2026-09-26, #266)

**`process_pool::Requester`** (`Owner` / `Swarm`) is a REQUIRED argument of `ModelProcessPool::generate` and `forward_for_request` (`IpcForward::for_the_owner`); the worker records it once (`KvCacheStore::mark_owner_request`) and `cpu_pools::in_phase_pool` reads it. The owner's PROMPT reading runs on `ResourceConfig::owner_prefill_threads`; decode and the swarm's work keep the contribution width.

**Memory too (#266):** `ModelProcessPool::ram_budget_now(requester)` — REQUIRED, through `get_or_spawn`, growth, reclaim, the processor-side cache grant and the planner's local bound — picks the cap via `RamBudget::judging`: the swarm's is `inference_ram_budget_mb` (contribution-scaled), the owner's `ResourceConfig::owners_ram_budget_mb` (80%, or an explicit `max_ram_mb` for both); the live anti-swap headroom binds both. The scheduler names the requester with `pipeline::requester_for_override` (a delegated split is the swarm's). A drafter is weighed as the request it guesses for (`process_pool::drafter_requester`, from `IpcDraft::for_the_owner`, which `EngineDrafter` sets from `worker_requester`; #351, A/B `SWARMLLM_DRAFTER_CAP=swarm`); a decode batch as the swarm's.

→ `docs/invariants/memory.md` § "The contribution level caps the swarm's work, not the owner's prompt (2026-09-26)"

## A subprocess you are waiting for can die instead, and the graphics stack can go mid-run

Whenever code waits for a subprocess, handle it dying instead (`spawn_worker` races `child.wait()` against `listener.accept()`). **`daemon::gpu_support::gpu_runtime_has_failed`** is the single answer to "has the graphics stack stopped working since we started?" (health monitor withdraws the GPU). ⚠ Never tell the owner the processor takes over — a CPU-only worker dies the same way. **`SharedState::inference_outage`** is "can this node run a request" (`can_serve_inference`); shard serving stays up. Guard: `both_inference_outages_withdraw_through_one_predicate`.

→ `docs/invariants/memory.md` § "A worker that dies is not a worker that is slow, and the graphics stack can go mid-run"

## A reply between two forwards is not idle

**`WorkerHandle::in_use` is the single answer to "is this worker in use?"** — a response in flight OR a conversation held between forwards (`kv_holders`, `CONVERSATION_GAP_SECS`); promotion, both reclaims and `models_with_inflight_requests` ask it. Guard: `whether_a_worker_is_in_use_is_decided_in_one_place`. Promotion waits for `reason_still_holds`; a forward past the prompt pass with no conversation is refused (`model_worker::forward_lacks_its_conversation`).

→ `docs/invariants/memory.md` § "A reply between two forwards is in use, and a lost conversation is refused"

## A worker that cannot grow gets the spawn's ladder, not a refusal (2026-09-27)

**`ModelProcessPool::grow_worker` is the one place a live worker takes on a new range**: reclaim idle models' graphics memory (`free_vram_for_admission`), then the range across the card and the processor (`charge_segment_as_split`, #265: the spawn's `card_split`, each share charged to its own budget and recorded on the range), then — only if `WorkerHandle::in_use` is false — retire the worker so `get_or_spawn` places the range afresh. Never retire one mid-conversation (#93). A worker in system memory skips the two card rungs, never the retirement; on either device it retires only where the range would load in its place (`a_fresh_spawn_fits_in_place_of_the_worker`, #348/#349). **A range is loaded where it was charged**: the worker is told (`DaemonMsg::PlaceRange`, under `spawn_lock`, before any forward for it) and the loader asks `split::placement_of` — never the process-wide `--gpu-layers` / `--cpu-moe-layers` alone, which a split spawn would apply to every later range.

→ `docs/invariants/memory.md` § "A worker that cannot grow gets the spawn's ladder (2026-09-27)"

## A split's stored prompt lives in the worker's prefix cache (2026-10-09, #10)

A segment keeps a split's prompt between turns as chain-keyed entries in the worker's `PrefixCache` (`lookup_chain` / `insert_chain_from_kv`), never a second store: one budget, one eviction (gotcha #440). In `handle_forward`: look up BEFORE admission, admit restored + computed positions, hydrate AFTER; any mismatch is `PromptCacheMiss`, never a forward from the wrong opening.

→ `docs/invariants/memory.md` § "A split's stored prompt lives in the worker's prefix cache"

## A prompt pass read in pieces is admitted once, for its whole span (2026-10-10, #171)

In `handle_forward` the FIRST piece of a `prompt_span` clears, restores and is admitted for `span.end - index_pos` positions (gotcha #447); a later piece is admitted nothing and must continue exactly where the cache ends. #10's store runs at the FINAL piece only.

→ `docs/invariants/memory.md` § "A prompt pass read in pieces is admitted once"

## A split's local segments are admitted together, before the first forward (2026-10-13, #263)

**`ModelProcessPool::admit_prompt`** (`DaemonMsg::AdmitPrompt`), asked once from `forward_through_segments` when a peer takes part, admits the prompt over every range the plan gives this node — the same decision as one segment's (`model_worker::admit_prompt_over`) — before any peer reads it. A segment admitted up front clears with `clear_request_caches` (never `clear_request`, which forgets the promise) and skips its own admission. A refusal carries `layers_that_fit` for the re-plan. **Up front, nothing is evicted** (`Admitting::UpFront`, #321): a prompt that fits only once saved prompts go is `Admitted::LeftToTheForwards` — each segment admits itself AFTER its lookup, or the eviction takes the opening the pass resumes from. The wait is bounded (`UP_FRONT_ADMISSION_WAIT`): a worker busy with another forward answers late, and the pass goes ahead per segment.

**Its PEERS are asked at the same moment (#252 (b))**: `PromptAdmissionAsk` to every peer whose segment runs after other work (`pipeline::prompt_admission`, `features::PROMPT_ADMISSION`). The asked worker WEIGHS with the same arithmetic and records nothing — `Admitting::OnlyWeigh`, a REQUIRED argument of `admit_prompt_over`: no reserved positions, no promise, no release (it counts `finished_conversations_bytes_in`), no cleanup — under a fresh correlation id (`ModelProcessPool::weigh_peer_prompt`), in a slot that does not stamp `last_used` (`WorkerHandle::register_question` — a question is not use, or one every 5 s pins the model). Never let a peer's question write worker state — with ONE exception: a "go ahead" loads the asked ranges a forward could load (`dispatch::prompt_admission::may_load_ahead`: inside the model by `layer_range_is_valid`, every part held here, ≤ 2 ranges, none overlapping — security review 2026-10-25) via `ModelProcessPool::load_ahead_for_a_peer` → `DaemonMsg::LoadRange`, after the answer, through `get_or_spawn` as the swarm's tenant exactly as their forward would, skipping a range already charged — so a cold split's machines load side by side, not one after another. With NO worker for the model, or one that does not hold the first range, the asked node weighs the LOAD first, charging nothing (`ModelProcessPool::peer_load_refusal`, #345/#349: `forward_would_load` — a fresh spawn by the spawn's rungs, `fresh_spawn_machine_mb`, on a card through the planner's own `machine_mb_beside_card`; a live worker by its growth, `growth_would_be_admitted`, else, idle, a fresh spawn in its place); "go ahead" there let the first forward's refusal bar the peer. A "no" holds the peer to `layers_that_fit` on the re-plan (`note_peer_prompt_refusal`), never bars it.

→ `docs/invariants/memory.md` § "A split's local segments are admitted together", § "A split's peers are asked about room before the pass too"

## A model loaded for ANOTHER model's request is a guest (2026-09-28)

**`process_pool::Tenancy` is a REQUIRED argument of `get_or_spawn`.** A `Guest` (the drafter, `ModelProcessPool::draft`) takes only memory that stands free: no reclaim, split, promotion, growth or notice — and is charged the cache of the reply it guesses for (`Guest { conversation }` → `Tenancy::kv_charge`), not `ADMISSION_KV_CONTEXT`; its worker's grant is unchanged (#275). `dsd.rs` reads ahead only once `holds_segment`. Whether a guest WOULD load is asked without loading it — **`ModelProcessPool::guest_would_be_admitted`**, the spawn's own arithmetic (`card_verdict` / `ram_verdict`, shared with `admit_to_gpu` / `admit_to_cpu`) — before the drafter is chosen, beside the layers the plan loads here (`guest_would_be_admitted_beside`; #275, `arch-scheduling.md`) — in system memory AFTER the plan's own share of it has loaded (`machine_mb_a_plan_adds` → `RamBudget::after_loading`, #351).

→ `docs/invariants/memory.md` § "A model loaded for another model's request is a guest (2026-09-28)"

## A fan-out to every worker is bounded, and the two waits mean different things

**`ModelProcessPool::notify_every_worker` is the one fire-and-forget fan-out** (`cancel_request`, `release_request_kv`), awaited inline from the dispatch loop (gotcha #74). The writer LOCK timing out stands down at `debug!`; the SEND timing out marks the worker `dead` (a partial frame desynchronises it). Sends run concurrently.

→ `docs/invariants/memory.md` § "`ModelProcessPool::notify_every_worker` — a fan-out to every worker is bounded"

## A conversation that is over gives its room to one that is not (2026-10-08)

**`KvCacheStore::release_finished_conversations`** runs before a prompt is refused (`ensure_room_for_prompt`) and before a growth claim is (`claim_room`, whose `claimant` is REQUIRED): caches silent for `FINISHED_CONVERSATION_AFTER` (= `process_pool::CONVERSATION_GAP_SECS`, the one figure) go first — a segment served for another computer is never released by its coordinator. The worker runs its store's TTL sweep (`cleanup_expired`) on its 30 s tick; it never had (#235). Guard: `a_workers_kv_store_is_swept_by_the_worker_that_builds_it`. At a REQUEST's end (never an attempt's — #749) the coordinator tells every peer it ran segments on (`router::release_request_on_peers`, `CancelInference`), and a worker drops a cancelled request's cache (`release_caches_of_cancelled`, #238).

→ `docs/invariants/memory.md` § "A conversation that is over gives its room to one that is not"

## A card's free memory is read after a synchronize (2026-09-26)

**`kv_budget::device_free_and_total_bytes`** synchronizes the stream before `mem_get_info` and adds `cuda_pool::reusable_bytes`; every budget decision reaches the card through `SplitModel::kv_budget_now` → it. Guard: `the_cards_free_memory_is_read_after_a_synchronize`; A/B `SWARMLLM_KV_DEVICE_SYNC=0`. `reply_reserve_positions(max_tokens)` is a REQUIRED argument of `ensure_room_for_prompt`; a refusal caused by OTHER conversations says wait (`other_conversations_hold_the_room`).

→ `docs/invariants/memory.md` § "A card's free memory is read after a synchronize"

## A cache is charged to the memory it lives in (2026-10-13, #262)

A card/processor split keeps its card-held layers' caches on the card and the rest in host memory: **`KvHome`** names the memory, **`KvOccupancy::allocated_in(home)`** is the only figure a budget is weighed against (`allocated_bytes` counts every device — report it, never compare it), and **`SplitModel::kv_homes`** lists every memory a model charges — the growth claim (`claim_room(home, …)`) and admission (`admit_prompt_over` → `admit_prompt_in`, promises per home) ask each. A split's processor side has its own budget (`host_kv_budget_bytes`: `--kv-budget-bytes`, else the machine's available memory less the margin), reconciled with the machine at every claim.

→ `docs/invariants/memory.md` § "A cache is charged to the memory it lives in"

## A card's KV budget is kept whatever the context setting (2026-10-09)

The loader records the card's KV budget with or without `inference.max_seq_len_override`. Skipped (as it was until 2026-10-09), a node set up for an agent admitted every prompt and spilled the cache into host memory — #440's crawl. A budget refuses one request that does not fit NOW; it never shortens a context.

→ `docs/invariants/memory.md` § "A card's KV budget is kept whatever the context setting"

## The daemon holds no CUDA context — only workers do (2026-10-05)

**`gpu_support::describe_local_gpu`** (driver API: `cuInit`, name, total — no context) is how the daemon learns its card. Never llama.cpp's device list or `Device::cuda_if_available` in the daemon: each leaves a context that held 137 MiB of an 8 GB card for the node's life. Free card memory is read live (`vram::query_gpu_vram_free_mb`), never kept from startup — `GpuInfo` has no free field.

→ `docs/invariants/memory.md` § "The daemon holds no CUDA context (2026-10-05)"

## nvidia-smi is asked through one bounded helper (2026-10-05)

**`vram::nvidia_smi`** (a `BoundedCommand`: 10 s, never two copies at once — while a stuck one lives, the next reading is `None` at once) is the only way the daemon and workers run `nvidia-smi`; a bare `Command::new("nvidia-smi")` waits as long as the driver does — 14 min during one card reset. Guard: `nvidia_smi_is_asked_only_through_the_bounded_helper`. Once a copy has been stuck for 60 s (`DRIVER_NOT_ANSWERING_AFTER` — one slow answer is not a reset, #802), `vram::graphics_driver_not_answering` is true and new workers go to the processor (`CpuReason::DriverNotAnswering`, transient — promotion brings them back); a shorter stall is waited out on the card. **Its wait never holds a Tokio runtime worker** (`executor::without_starving_the_runtime`, #244): 2-6 s of `nvidia-smi` parked the network manager in that worker's unstealable LIFO slot. Test: `a_slow_bounded_command_does_not_hold_the_runtime_worker`.

→ `docs/invariants/memory.md` § "nvidia-smi is asked through one bounded helper (2026-10-05)"

## A card full of our own model is not pressure (2026-10-12, field report #035)

Prune's soft-unload narrows a loaded model only when our CHARGED graphics memory (`vram_committed_mb`) exceeds `compute_vram_budget` — other programs took room — and only the largest model NOT in use (`models_with_inflight_requests`), priced by its own charge (`vram_committed_by`). Never a fraction of the card's total (a loaded model fills it) nor the whole file (a split model is charged its card share). Choice: `prune::card_room_to_give_back`. Idle unload asks `prune::stays_loaded_while_idle`; regional demand buys no reprieve.

→ `docs/invariants/memory.md` § "A card full of our own model is not pressure"

## A card that stalls is given less work, not more (2026-09-28)

**`inference::card_pace::CardPace` is the one answer to "may this worker start another generation on its card now"**: a device-bound step ≥ 2 s halves the ceiling. The gate sits in the `Generate` arm BEFORE the batched/sequential choice; a new device-bound step is timed into it, a new generation path gated by it. Refusal = `LocalMemoryUnavailable`. A/B `SWARMLLM_CARD_PACE=0`.

→ `docs/invariants/memory.md` § "A card that stalls is given less work (2026-09-28)"

## Machine memory is read in one place, under the cgroup's limit (2026-10-09)

**`daemon::machine_memory::machine_memory()`** is the only reading of how much memory this process may use — host figures under its cgroup's limit (a container's `--memory`, a systemd `MemoryMax`), available = limit less what the kernel could not reclaim (`memory.current` − `active_file` − `inactive_file` − `slab_reclaimable`, LXCFS's `MemAvailable`) — never the kubelet's working set (counts ACTIVE cache as used: the .233 gate's nodes refused with nothing loaded) nor `sysinfo`'s cgroup "free" (counts all cache). Guard: `machine_memory_is_read_in_one_place`.

→ `docs/invariants/memory.md` § "Machine memory is read in one place, and a container's limit counts"

## A card's memory pool keeps what the worker frees, until the worker is idle (2026-09-28)

**`split::loader::load_device` is the one place a worker picks the card** and raises the pool release threshold (`cuda_pool::keep_freed_memory`); `model_worker::run_worker` trims after `cuda_pool::IDLE_TRIM`. Any "free for this process" reading adds `cuda_pool::reusable_bytes`. Guard: `every_split_model_reaches_the_card_through_load_device`; A/B `SWARMLLM_CUDA_POOL_KEEP=0`.

→ `docs/invariants/memory.md` § "A card's memory pool keeps what the worker frees (2026-09-28)"

## Under WSL2 a card's saved prompts live in host RAM (2026-10-13, #257)

**`prefix_cache::SnapshotHome::for_device`** is the one answer to where a prefix-cache snapshot is kept: host RAM for a card reached through WSL2 (`cuda_pool::card_is_reached_through_wsl`), beside the live cache elsewhere. A snapshot held on the card made the NEXT prompt grow the pool, which under WSL2 slows with uptime (1.1-6.9 s at 56 h). `model_worker::snapshot_room` plans home + positions, REQUIRED by both inserts; host RAM is bounded by `kv_budget::host_snapshot_budget`; elsewhere a card's copy that does not fit whole beside the live cache goes to host RAM (`room_off_the_card_if_more`, #321); the card's budget and `release` see only `BesideLive`. A narrowed host layer stays a view. Judge a change by `pool_reserved_mb` rises, not tok/s on a fresh boot (gotcha #834). A/B `SWARMLLM_PREFIX_CACHE_HOME=card|host`.

→ `docs/invariants/memory.md` § "Under WSL2 a card's saved prompts live in host RAM"

## The owner is told when fresh card memory has become slow (2026-10-01)

**`cuda_pool::probe_once`** times 16 fresh 4 MB allocations once per worker at `load_device`, never per request; a probe ≥ 0.5 s reaches `ModelProcessPool::slow_card_notice` (`WorkerMsg::CardAllocationProbe`), WSL2 only, at most every 12 h. It changes nothing the node does.

→ `docs/invariants/memory.md` § "The owner is told when the card has become slow to hand out memory (2026-10-01)"

## A lone decode stream is never held for a batch (2026-09-29)

**`process_pool::collection_target`** waits for a batch only when ANOTHER request is decoding on the same model (`ActiveStreams`). Never reintroduce a wait a lone stream pays.

→ `docs/invariants/memory.md` § "A lone decode stream is never held for a batch (2026-09-29)"

## The download pass fetches only what prune would keep (2026-10-06)

**`AutoShardManager::would_shed_copy` is the one answer to "would prune shed this copy?"** — prune asks it of what it holds; EVERY automatic fetch path asks it before fetching, at the disk pressure AFTER the fetch: `select_within_budget` (cumulative per cycle) and `complete_pending_shard_fetches` (which DROPS the entry). A finished origin download leaves `shard_p2p_failed` in `announce_shard_acquired` (gotcha #797); an entry the origin cannot serve (`pending_fetch_can_proceed` → `can_fetch_shard_from_origin`) is dropped back to the peers, never left to block them. Every file decision reads DISK pressure, never graphics memory; the filesystem's last 10% is never ours (`FREE_DISK_RESERVE_PCT`). Guards: `fetch_what_prune_keeps::*`. **The target moves while a part does not (#286)**: prune sheds against `shed_target_replicas` — the highest `geo_target_replicas` read in 24 h (`target_hold`, HPA's scale-down rule, kept across restarts) — while the download pass reads it as it is; never the reverse. A part of a model held whole here — or all but one in eight of its parts — counting the part, is kept below urgent pressure (`coverage::completes_model_here_counting`), and the parts such a model lacks are fetched past the target (`parts_completing_model_here`, #320); one under 30 min old is not pruned below eager pressure. **Carrying (#231)**: replicas count copies, not whether a holder can RUN the model — `would_shed_copy` keeps a copy the model would fall short without (`coverage::copy_carries_model`, asked before the replica count), and the one carrier `coverage::carrier_for` chooses (in scope; a lease kept only by progress) fetches past the target and the ring; a new rule about which copies matter goes in `coverage.rs`, asked by both passes. Guards: `coverage::tests::*`. **Near chains (Stage 3)**: for an asked-for model no NEAR machine (measured ≤ 30 ms) can run whole, every near machine plans one chain alike (`coverage::near_chain`) and fetches its segment (`parts_this_node_plans`); prune keeps it (`copy_in_near_chain`, below urgent) — one chain, in-flight counted as held, a near machine's place held 30 min and a named part 24 h (#876). **Where the near machines cannot carry it, the same chain is planned over every machine in scope** (`Reach::Far`, #346 — a 70B whose holders are 300-410 ms apart) — its parts kept at ANY pressure while it is planned, as carrying's (it is how the model runs at all; a near chain's, and a part named only within the day, below urgent: `ChainPart`) — and **carrying stands down while any chain is planned** (`near_chain::chain_planned_for`, asked by `needs_carrier` and `copy_carries_model`); carrying's count is a maximum flow (`scheduler::layers_carried` — a sum counted a layer once per holder: 80 of 80 "carried" where 67 could be placed). Layers this node has LOADED outside its segment of a chain every machine HOLDS (`chain_held_by_its_machines` — a peer's fetch is only its word) are given back once idle past the conversation gap, not the idle window (`loaded_outside_near_segment`, then the ordinary rule for 30 min): a router holds a peer to its room BESIDE what it has loaded, so the chain was never planned. A/B `SWARMLLM_NEAR_CHAIN=0`; rig `examples/near_chain_rig.sh`. **A model with its safety training removed that the owner did not choose** (`SharedState::unchosen_uncensored_model`) is shed and never fetched — asked in `copy_kept_because` before prompt privacy's ends, never as a gate per trust writer — and planned no near chain (`may_plan_a_near_chain`). **A drafter a split lacked (#134), or a larger one (#331)**: every split that guesses here records its target (`engine_drafter::note_split_wanting_a_drafter`, from `dsd.rs` and `requester_guesses::start`; a miss wakes the pass) and the download pass fetches the one **`engine_drafter::drafter_to_acquire`** names — what `chosen` would pick were it held, of the target's family, ≤ `MOST_DRAFTER_BYTES_TO_FETCH` (2 GiB: a user holds small things, never a large model), trusted as adoption requires (`coverage::drafter::may_fetch_as_a_drafter`), only once every likely candidate's header is here, and where one is held only one LARGER by `size_proxy` — past the target and the ring (`parts_of_a_drafter_to_fetch`); prune keeps below urgent (`copy_drafts_for_a_split`) what **`DrafterToAcquire::keeps`**: the held one on EVERY step and the one being fetched, so a replacement arrives beside what it replaces and the old one goes only once the new one is held and chosen — while splits want one (`DRAFTER_WANTED_FOR`, 7 days). Never let a consumer match the step alone: it would drop the held one while headers are learned.

**Holes first (2026-10-25)**: a part no computer holds or is fetching, of a model somebody asked for that the swarm holds part of (**`scoring::fills_a_hole`**), is chosen before any further copy whatever its score — the sort and the first tier of `select_within_budget`'s round-robin, never a multiplier (the score's popularity and `vram_fitness` both rank a large model being assembled below copies of small ones). **A full node makes room for one**: the download pass notes the room the first hole it cannot fit lacks (`AutoShardManager::room_for_holes`, written by it alone every cycle, `budget == 0` included; `scoring::room_a_hole_lacks`), and while it stands `copy_kept_because` sheds a copy kept ONLY by the replica count or the region rule if **`prune::spare_for_a_hole`** — more than `floor_for_a_hole` live holders (two others remain) and this node not among the first that many by rendezvous (`coverage::kept_by_rank` — holders deciding at once never all give back), not of a model with holes of its own, not loaded, not completing a model here — given back up to the room noted (`PruneCandidate::to_make_room`, re-checked against the floor). Never past a rule that keeps a copy for what it does (carrying, near chain, drafter, privacy, pins).

**A download nobody asked for waits while the connection is metered (2026-10-15)**: **`SharedState::metered_hold`** (`daemon::metered`) is the one answer, asked just before the start by every path that starts one on its own (the download pass; the rebalancer only wakes it, never fetches); setting `auto_manage.on_metered_connection`. Repairs, the owner's downloads and transfers already running are not paused. Guard: `every_automatic_download_asks_whether_the_connection_is_metered`. The same answer holds back uploads to peers (`arch-network.md`, #277).

→ `docs/invariants/memory.md` § "`AutoShardManager::would_shed_copy` is the ONE answer"

## A model is admitted against the card as it stands NOW (2026-09-29)

**`ModelProcessPool::vram_budget_now`** is what `admit_to_gpu` weighs a model against, re-read at admission — and what the promotion back to the card weighs (`promotion_weighed`, read only once its cheap guards pass; #260: the figure a refusal stored kept Mistral on the processor after Ollama let go). Promotion weighs every worker placed short of the card — **`WorkerHandle::short_of_the_card`**: the processor OR a card/processor split, which admission makes first — with a split's own share given back (`own_card_mb`), and never a model above `ceiling_mb` (`auto_manage::vram_budget_ceiling`, no card reading). **A plan reads it through `card_budget_for_planning`** — live where a spawn could follow, one reading per `PLANNING_CARD_READING_FOR` (#250, report #036: a stored 7.5 GB planned a split the spawn found 556 MB for) — and a fresh split's processor share is weighed against the requester's RAM cap as the spawn will (`fresh_run_layers_for_planning`), counting the idle models the spawn unloads first (`idle_ram_reclaimable_mb`, one predicate with `free_ram_for_admission`: `ram_reclaim_eligible`). Never cache a live condition, at startup or at the last admission.

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
- **`ModelProcessPool::footprint_inputs` reads a model's WEIGHTS from its header** (`split::tensor_byte_size` over the tensor table), never from the shards on this node's disk — which is the model only on a full holder (a node holding none priced a 9B at 1346 MB; one holding half charged half; gotcha #803). → `docs/invariants/memory.md` § "A model's weights are its header's".
- **`split::kv_layers`** (+ `GgufTensorMeta::layers_keeping_kv`, same marker) — the layers charged a KV cache, read by `kv_budget::KvLayout` (so by every estimate below) and the scheduler's peer bound (`prompt_kv_bytes_per_layer` → `KvLayout::growth_on_peer`, #281 (4); `kv_bytes_per_position_per_layer` only without a layout); a recurrent layer (Qwen 3.5's DeltaNet, `split::layer_is_recurrent`) keeps none (#228).
- **`kv_budget::KvLayout` → `KvGrowth` / `KvPrice`** — the ONE price of a conversation's caches, built per layer off the header (each layer's shape and storage, a windowed one by its window): the loader's budget, the worker's admission, the snapshot plan and the context reported read `KvGrowth`; before a load, admission's footprint, `process_pool::cost_curve_of` (`KvPrice::share` with the segment, or the whole cache lands in its fixed term) and the card/processor split read `VramFootprintInputs::kv` (`KvPrice::reading(reader, context)`, #281 (3)). Never restate `layers × heads × head_dim × positions`. The head-room guard reads the BUFFERS instead (`KvCacheStore::bytes_a_forward_adds`). → `docs/invariants/inference.md` § "A windowed layer is charged as its window".
- **`ModelProcessPool::free_vram_for_admission` + `plan_vram_reclaim`** — reclaim idle models first; **`fits_in_budget`** (every fit verdict, so `serves_on_cpu`) and the planner's ceiling count that reclaim (`reclaimable_vram_mb` / `idle_vram_reclaimable_mb`), or a model held here goes abroad (#125, #129). On the processor the bound counts `idle_ram_reclaimable_mb` (`RamBudget::headroom_after_reclaim`, #276) — but never beside a guest, which takes only memory standing free.
- **`should_return_to_gpu` + `worker_should_return_to_gpu`** — asked in `get_or_spawn`, not on a timer.
- **Graphics memory has ONE owner: `ModelProcessPool`** — and every admission that says yes charges it, a card/processor split included (`card_share_of_split_mb`, #240).
- **A worker's growth is weighed by the budget its spawn charged** — `WorkerHandle::holds_gpu_memory`, never `placed_on_cpu_because` (`a_live_workers_growth_is_weighed_by_the_budget_its_spawn_charged`).
- **`model::auto_manage::storage_budget` / `held_disk_bytes`** — the ONE answer to shard storage; counted kinds must be reclaimable.
- **`prune::effective_idle_secs`** — residency is a hard UPPER BOUND on "idle since".
- **An admitted prompt is RECORDED** — claim and reservation go TOGETHER via `forget_request_bookkeeping`.
- **`kv_budget::admit_prompt` + `PrefixCache::release`** — ONE decision per prompt.
- **A prompt of known length is RESERVED** — `set_reserved_positions`; `SWARMLLM_KV_RESERVE=0`.
- **`process_pool::worker_socket_path`** — the ONLY place the IPC socket path is built.
- **`process_pool::end_with_this_daemon`** — on Windows every worker joins the daemon's kill-on-close job; a new spawn path calls it (#153).
- **`logging::forward_worker_output`** — while the log has a file (`logging::pipes_worker_output`), a worker's stdout and stderr are piped through the daemon into its window and the file: ONE writer, so the file can rotate (Windows refuses to rename a file another process holds). A new spawn path pipes through it too. → `docs/invariants/memory.md` § "A model process's output reaches the log file through the daemon"

→ `docs/invariants/memory.md` § "Single-source-of-truth helpers — Worker memory: graphics, RAM and the KV cache"
