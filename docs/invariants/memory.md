# Worker memory: graphics, RAM and the KV cache

The evidence behind the rules in `.claude/rules/architecture.md`: what each
rule replaced, what it was measured at, and what a change must keep.

Every entry here was paid for. **Read the entry before changing the code it
names** — the rule statement in `architecture.md` is the summary, this is the
reasoning, and several of these describe a fix that looked obviously correct
and was not.

## A process's memory is the largest accounting the platform offers

**`api::process_memory::resident_bytes(pid, sysinfo_bytes)`** is the single
answer to "how much memory is one of our processes holding", and it takes the
maximum of every reading the platform has rather than choosing one.

macOS keeps two and they are not interchangeable: `sysinfo` reports
`pti_resident_size` from `proc_pidinfo`, Activity Monitor reports the memory
footprint (`ri_phys_footprint` from `proc_pid_rusage`), and the two can differ by
orders of magnitude depending on how the pages were obtained. Reported (report
#017, 2026-09-11): a worker holding a 14B showed **13 MB** in the dashboard's
tooltip while Activity Monitor showed ~13 GB and the machine was at 13-15 of
16 GB. The system-wide figure the same endpoint reads was correct throughout —
only the per-process half collapsed.

**The direction is not symmetric and that is why the maximum is right.**
Under-reporting is the observed failure and the one that matters: a bar near zero
on a nearly-full machine tells a user their node is idle when it is the thing
filling their memory. Over-reporting is bounded by the machine's own total, which
the bar is drawn against.

⚠ **The report's stated mechanism is probably backwards** — footprint is the
accounting that EXCLUDES clean file-backed pages, and footprint is what showed
13 GB, so "mmap'd pages invisible to the process figure" predicts the opposite of
what was seen. The fix deliberately does not depend on which is right. Not
testable from Linux; CI's macos-15 job compiles and runs it, and
`this_process_reports_a_footprint_on_macos` fails there if the syscall stops
answering — without it a permanently-failing call would leave the figure silently
equal to the one it exists to correct.

## A finished request releases its conversation cache, wherever it is held

**`DaemonMsg::ReleaseRequestKv`**, sent by
`ModelProcessPool::release_request_kv` at the one place a request finishes, is
how a segment's KV entry is freed. The `Generate` handlers clear their own on
the way out; the FORWARD path had nothing, so what it held stayed charged
against the shared budget until the ten-minute idle sweep.

The daemon's own `kv_cache_store.cleanup_request_id` looks like it covers this
and does not: **the worker's store is a different process's.** That is the whole
defect in one line.

**Unconditional, session or not.** The worker keys entries by REQUEST id —
`session_id` rides along on `IpcGenerate` and nothing in the worker reads it —
so the next turn of a conversation arrives under a new id and could never find
the old entry. What carries a conversation forward is the prefix cache, and its
snapshot is taken before the entry is cleared. The daemon-side skip for
session-keyed requests is about a different store and is left alone.

Measured (report #019, 2026-09-11) on a 16 GB processor-only machine: three
refusals seconds apart, across two different conversations one of which the user
had finished, all reporting the identical `live_mb=2472` — a figure that cannot
move because nothing ever gave anything back.

**A KV-admission refusal is `LocalMemoryUnavailable`, and the class travels as a
flag.** `WorkerMsg::Error::local_memory_refusal` carries it across the IPC
boundary because the wire WORDING is deliberately identical to a peer's memory
refusal (see `SwarmError::LocalMemoryUnavailable`), so
`classify_worker_error` cannot and must not re-derive it from the text —
`reclassify_flattened_error` deliberately never produces this variant. It is the
one local failure the router re-plans, and in the reported case a five-segment
route across peers had been priced one line earlier in the same scheduling pass
and was discarded when the refusal read as final.

## `inference::worker_ipc::worker_error_is_fatal`

(R146) — the single
source of truth for "did this worker error destroy the worker's device
state, or just this request?". Used by the worker to stamp
`WorkerMsg::Error.fatal` AND by the daemon's
`ModelProcessPool::classify_worker_error` to re-derive the verdict from
the message text (the field is `#[serde(default)]`, so a worker binary
older than the field always reports `false`). A `true` verdict evicts
the worker from the pool, which drops the last `Arc<WorkerHandle>` and
lets `Drop` kill the child — the only thing that actually returns VRAM
to the OS. New fatal-error classes go in the pattern list, not into a
caller-side special case; divergence between the two sides means a
stranded worker holding its whole allocation for the daemon's lifetime.
Lean inclusive: a needless respawn costs one model reload.

## `daemon::shard_loader::force_cpu_for`

(R146) — the single mapping
from `inference.gpu_layers` (`-1` auto / `0` CPU only / `>0` GPU) to the
loader's `force_cpu` flag. Every device-placement decision goes through
it: `ModelProcessPool::effective_gpu_layers` → `--gpu-layers` spawn arg
→ `model_worker::set_worker_force_cpu` → `ShardLoadParams.force_cpu` /
`SplitModel::load_from_gguf(force_cpu)`. Do NOT re-derive placement by
calling `Device::cuda_if_available` directly in a new load path — that
is exactly how `gpu_layers` came to be silently ignored for every
sharded model. Partial offload is not expressible (see
`docs/FUTURE_WORK.md`); a fractional value logs a warning rather than
being quietly rounded.

## `daemon::gpu_support::MIN_COMPUTE_CAP` + `local_gpu_is_supported`

(2026-08-07) — the single answer to "can this card run OUR kernels?".
`MIN_COMPUTE_CAP` is a property of the BUILD and MUST equal
`CUDA_COMPUTE_CAP` in `release.yml` / `cache-warm.yml` / `ci.yml`;
`compute_cap_matches_release_workflow` fails the build if they drift, and
`flash_attn_and_the_compute_cap_floor_agree` ties the floor to the feature
in BOTH directions (8.0 is only worth paying for because of flash-attn).
Do NOT ask `Device::cuda_if_available` whether the GPU is usable — it
SUCCEEDS on a pre-Ampere card and only module load fails, per request, so
the node starts cleanly, logs "GPU detected", advertises itself to the
swarm as a GPU node, and then fails everything with
`CUDA_ERROR_NO_BINARY_FOR_GPU`. An unreadable capability is **unknown,
never unsupported**: sending a working card to the CPU because nvidia-smi
misbehaved is a worse bug than the one this prevents. Enforcement is at
`ModelProcessPool::effective_gpu_layers` (the same choke point as
`gpu_layers` and OOM CPU-pinning), with
`worker_ipc::permanent_gpu_failure` as the backstop for when the probe
returned unknown.

## `model::auto_manage::vram::ADMISSION_KV_CONTEXT`

(2026-08-18; CPU since
2026-08-21) — the context length admission charges KV cache for, on either device,
whatever the user configured.
**Admission may charge less than the worst case exactly where a runtime check
catches the difference, and nowhere else.** Both workers now have one:
`kv_budget::claim_exceeds_headroom` in `forward_inner_impl`, a 503 that re-routes.
The GPU worker derives its budget from free VRAM at load; the CPU worker is HANDED
its budget by the daemon at spawn (`--kv-budget-bytes` →
`inference::split::CPU_KV_BUDGET_BYTES`, computed by
`ModelProcessPool::record_cpu_kv_budget` as the typical-context charge plus the RAM
budget still uncommitted at admission), because only the daemon knows what else is
resident. **`ModelProcessPool::charges_ram` decides whether a spawn is charged against
RAM at all** — going to the CPU, OR no GPU detected, OR a build without CUDA. The
first alone missed every CPU-only node: with no GPU there is no VRAM budget,
`admit_to_gpu` admits everything, and the model landed in RAM uncharged and
un-budgeted. It changes charging, never placement — the worker still falls back
to the CPU on its own, so a working card whose probe failed is not sent to the CPU
by it (unreadable is unknown, not absent). Until 2026-08-21 the CPU had no guard, so `estimate_worker_ram_mb` priced
the WHOLE ceiling — correct for the mechanism that existed, and what turned a 2.3 GB
phi-3.5 into a "needs 27125 MB" refusal at a 32k override (external report; MHA,
0.75 MB/token). **Do not remove one side without the other**: admission at a typical
context with no runtime guard means swapping, which degrades every request on the
machine; a guard with ceiling-priced admission means refusing models that fit.
`a_cpu_refusal_itemises_weights_kv_and_context` pins that the same model is now
admitted and that the old ceiling figure is still what `resident_footprint` reports
when asked for it.
**The RAM budget itself is a LIVE snapshot, never a startup figure**:
`vram::ram_budget_now` → `RamBudget { cap_mb (from `cfg()`), live_headroom_mb
(max(70% of available NOW, total/4)) }`, installed into the pool as a provider
closure (`set_ram_budget_provider`, `Weak<SharedState>`) and asked at EVERY
admission, by `free_ram_for_admission`'s retry loop, and by `record_cpu_kv_budget`.
The clamp used to be folded into the cap once at startup, so a daemon restarted
while memory was busy carried the smaller figure for life — the same
`max_ram_mb = 18000` answered "budget allows 13370 MB" one day and "10500 MB" the
next with 14773 MB actually free (external report, gotcha #362). The refusal names
whichever limit applied (`RamBudget::limiting_figure`). `set_ram_budget_mb` is the
no-provider fallback (tests) and the startup log figure; do not read it for a
decision.
Why it exists: `inference.max_seq_len_override` is the only way to hold an agentic
client's system prompt (~5000 tokens of tool schema before the user speaks), and
raising it used to raise this charge in step, so the model stopped fitting the card
and was loaded on the CPU — measured at 396 s of prompt processing and a thermal
warning (external report 2026-08-17). Pre-paying at load bought nothing the runtime
check was not already enforcing.
**Deliberately NOT `DEFAULT_MAX_SEQ_LEN`.** That is a product default and moves with
the audience — it went 4096 → 8192 the same day — while this is a statement about a
typical working conversation. Tying them would mean raising the default silently
re-broke the case above. `raising_the_context_no_longer_costs_a_model_its_place_on_the_gpu`
and `the_cap_is_inert_at_the_context_it_was_derived_from` pin both halves.

## `ModelProcessPool::free_vram_for_admission` + `plan_vram_reclaim`

(2026-08-25)
— reclaim graphics memory from models nothing is using rather than demoting the
requested one to the processor. Called from the admission-refusal branch in
`get_or_spawn`, BEFORE the CPU fallback is taken. **The exact sibling of
`free_ram_for_admission`**, which has done reclaim-then-retry for the RAM budget
since v0.3.111; the GPU side simply never had it, so a model that happened to
load first kept the card for as long as it stayed resident and everything asked
for afterwards ran on the CPU (gotcha #388).
**Why the pre-existing LRU eviction did not cover it**: `evict_split_models_lru`
frees against `estimate_vram_from_shard_dir` (shard bytes × layer fraction —
weights ONLY) while `admit_to_gpu` weighs `estimate_gpu_footprint_mb` (weights +
KV at `ADMISSION_KV_CONTEXT`). Two estimates of one quantity, and the SMALLER one
decides how much to free — so eviction reaches its own stop condition while
admission is still short, every time. Measured live: eviction freed 1202 MB, then
admission refused against a 2449 MB shortfall and the model loaded on the CPU at
~7 tok/s with the card at 12%.
**Neither timer can do this job.** `try_idle_vram_unload` runs on the auto-manage
tick, so it cannot act on the request arriving *now*, and its regional-demand
clause deliberately keeps a model the SWARM wants resident for up to
`idle_hard_unload_secs` (1 hour) — precisely a popular 8B.
Three properties a change here must keep. **Plan before destroying**: the whole
plan is costed first and abandoned whole if it cannot succeed, because unloading
models and still not fitting costs a cold start and buys nothing. The RAM sibling
may unload opportunistically — its alternative is failing the request outright;
here the alternative is a slower answer, so a wasted eviction is a real
regression. **An idle floor** (`VRAM_MAKE_ROOM_MIN_IDLE_SECS`): two models
alternating faster than they load would otherwise evict each other on every
request; below the floor the previous behaviour is kept, so this can only improve
placement. **LRU by real idle time, not residency**: `spawned_at` cannot tell a
worker answering steadily for an hour from one loaded an hour ago and never used
since, so `WorkerHandle::last_used` is stamped in `register_response` — the ONE
place every execution path (local, distributed, peer-served) passes through.
**Do not "correct" the estimate.** 6033 MB charged against a measured
`vram_after_load_mb=4853` is not a 24% over-charge; the difference is the KV
headroom doing its job, and trusting the measured figure would admit a model and
then OOM it.

## `should_return_to_gpu` + `ModelProcessPool::worker_should_return_to_gpu`

(2026-08-27) — the single answer to "is this resident worker still in the right
place?", asked on the request path in `get_or_spawn` rather than on a timer.
**A pin that clears buys nothing while the worker it produced is still running.**
A momentarily-full card demotes a model and pins it; `unload_model` lifts the pin
when a GPU tenant goes away and logs `clearing CPU pins`; and `get_or_spawn`'s
fast path then returned the processor worker regardless of device. `clear_cpu_pin`
is documented as letting "the next worker spawn" use the card, and there is no
next spawn — the worker survives until `idle_unload_secs` (15 min) of *no requests
at all*, which someone actively using the model never reaches. Reported by an
external tester: gemma-2-2b-it re-requested against a card at 653 of 6141 MB
reused its processor worker (gotcha #401).
**The decision reads the WORKER, not the model.**
`WorkerHandle::placed_on_cpu_because` records why the process actually went to
the processor, at the moment it was spawned; `cpu_reason` (and so
`effective_gpu_layers`) answers for a spawn happening *now*, off the pin that is
exactly the thing being cleared. **A running worker is a fact; `cpu_reason` is a
prediction**, and they differ precisely in the window this change creates. Three
callers were asking the prediction about a resident worker and are now corrected
to the fact: `unload_model` (whether killing this worker freed graphics memory —
a model on its way back to the card would have lifted every other model's pin on
the strength of memory it never held), `cpu_placement_reason` (the dashboard's
"why is this not on my GPU?" — which would have dropped its explanation from a
model still on the processor, and explained a model happily on the card as "you
configured CPU-only"), and `would_fit_on_gpu`'s already-charged short-circuit,
which would otherwise have re-created the 2026-08-18 contradiction its own
comment describes. **A new caller asking about a model that has a worker should
ask the worker — a LIVE one.** `ModelProcessPool::live_worker` is that
accessor. The reader task marks a worker dead the instant its socket closes
and leaves it in the map for whoever notices to retire, so a bare
`workers.get` can hand back a corpse, and a corpse's placement describes a
process that no longer exists while the prediction for the next spawn is
available and correct. On the request path the same gap cost one request per
worker death: `get_or_spawn` returned the dead handle, the caller's own
liveness check failed with `worker is dead`, and only the NEXT request — which
found the map cleaned — succeeded (gotcha #475). It now retires the corpse and
spawns, in both places a resident worker is handed back, since one can die
between them.
`WorkerHandle::gpu_estimate_mb` caches what admission priced the model at,
because `estimate_gpu_footprint_mb` re-reads `gguf_header.bin` and scans the
model directory: fine once per spawn, not fine once per request.
Four guards, three carried over from `plan_vram_reclaim` because this destroys
something that works. Only a worker THIS NODE demoted (on a machine with no card
`cpu_placed` is false, and there is nowhere to promote to). The demotion must have
stopped applying — of `cpu_reason`'s three causes only the OOM pin ever clears, so
this fires on the event that lifted it rather than polling. Never a busy worker,
and not one used inside `VRAM_MAKE_ROOM_MIN_IDLE_SECS`, because unloading kills
the subprocess and the idle floor makes the race window empty rather than merely
unlikely (the residual — a model under continuous load waits for a gap — is in
`docs/FUTURE_WORK.md`). And **cost the move before making it**: an unreadable
footprint or an unset budget is not evidence, and leaves the model where it is.
`admit_to_gpu` treats the same gap as "do not judge" and lets a spawn through;
the question here is whether to destroy something working, and the answer on no
information is no.
**The outcome is announced after admission, not before it.** The retirement logs
what it is doing; the user-facing `model_gpu_restored` event is emitted only once
the worker is on the card, because admission prices the model again and has the
last word — the same correction `admit_to_gpu`'s refusal log already carries.

## Graphics memory has ONE owner: `ModelProcessPool`

(2026-08-27). It admits
(`admit_to_gpu`), charges (`vram_reserved_mb`), and reclaims — on demand
(`free_vram_for_admission`) and on the idle timer (`try_idle_vram_unload`,
which runs outside the auto-manage gate). **Nothing else may take memory away
from a loaded model.**
**What this replaced.** `SharedState.split_models` — a cache of per-segment
metadata read out of `gguf_header.bin` — was governed by its own VRAM budget
that evicted entries *and unloaded their workers*. Two accountants for one
card, and the second was the weaker: a different estimate (weights only,
against the pool's weights + KV — gotcha #388's shape), a different in-flight
oracle (`active_pipelines`, which per gotcha #194 cannot see peer-served work
or the split fast path), and **no idle floor at all** — measured, it evicted a
model that had answered a request three seconds earlier, which
`free_vram_for_admission` refuses by design. Worse, it was placement-blind, so
registering a metadata entry for a segment bound for the PROCESSOR killed a
model running happily on the card (gotcha #402).
**Researched, not guessed.** Ollama's scheduler is the direct analogue and
keeps one owner: a single centralised free-space tracker, `runnerRef.vramSize`
reported by the runner, victims chosen by refCount (in-flight) → keep-alive →
`lastUsedAt`. Our pool already had the equivalent of all three.
**The split-model map is now a metadata CACHE**, bounded by
`MAX_SPLIT_MODEL_ENTRIES` via `inference::split::trim_split_model_cache` —
count-capped, LRU, active-pipeline-protected, and structurally unable to
unload anything (it takes no pool and returns only keys). **Do not restore the
unload**: it existed (2026-07-21) because eviction was *supposed* to free
graphics memory and did not, so the budget was enforced against a phantom.
That premise is gone — this no longer claims to free anything. Trimming an
entry still wanted now costs a header re-read, not a killed worker.
**A registration budget survives, and may refuse but never take.**
`SharedState::split_model_budget_with` + `split_models_committed_mb` + the
`MemoryScope` enum answer "should this node advertise another segment as
locally servable?" — `compute_vram_budget` (the card) or, on a node with no
card, `inference.max_split_model_memory_mb`. `MemoryScope` exists because
those two were reached through one `.or()` and describe different memory:
filtering by graphics residency under the second would disable it on exactly
the machines it is for. **`ModelProcessPool::model_uses_gpu_memory`** is the
single answer to "does this model occupy graphics memory", resident from the
worker's own `placed_on_cpu_because` and otherwise predicted from
`cpu_reason`, read through `charges_ram` so "sent to the processor", "no card
detected" and "a build without CUDA" give one answer rather than three.
**The rule to carry**: a budget over a collection whose members live in
different places must be told which place each one is in — and a component
that does not own a resource must not be able to reclaim it.

**`evict_worker_where` / `evict_this_worker` is how a worker leaves
`workers`** (2026-09-05, gotcha #467), and `unload_model` is the one
exception — it must DRAIN before killing, so it removes the entry itself and
ends in the same `after_worker_gone`.
`a_worker_only_leaves_the_pool_where_its_memory_is_released` in
`tests/repo_consistency.rs` fails the build on a tenth site, with a self-test
that plants the violation.
**Why**: #461 fixed the three `handle.dead` fast-fail sites; there were NINE.
The other six are the paths a worker actually dies on — a failed IPC send and
a closed reader channel on each of `forward` / `forward_batch` / `generate`,
plus `classify_worker_error`'s fatal arm, which is the CUDA-OOM path. And the
health-tick reap could not cover them: it scans `workers` for `dead` entries,
and these had already removed the entry, so the charge leaked exactly as
before on the six paths that matter most.
**`evict_this_worker` compares identity, not just the key** (`Arc::ptr_eq`).
A bare `remove(key)` lets a caller holding a handle that has already been
replaced evict the LIVE worker that replaced it — a latent hazard on all six
sites, closed on the way past.
**When a report names three call sites, it is describing symptoms, not scope**
— grep the operation and count them before calling the fix complete, and ask
what the safety net you just added can actually see.

**`after_worker_gone` is everything that follows a worker's process no longer
existing**, however it stopped: release BOTH budgets, and — if it held
graphics memory — lift the CPU pins its occupancy caused. `unload_model` and
`retire_dead_worker` both end in it.
**Why it is one function.** The two halves are one event and were written
separately: `unload_model` had done both since gotcha #401,
`retire_dead_worker` was added for #461 and did only the release. So a GPU
worker that CRASHED or was OOM-killed — the exact case #461 was written for —
freed the card and left every other model pinned to the processor at ~10x the
cost, indefinitely (gotcha #466). The pin's clearing condition is "GPU memory
freed"; it does not care how the process ended.
**After adding a second path to an existing invariant, read what the old path
does AFTER the part you copied.** Knowing the "one invariant, N paths" rule
did not prevent this; applying it deliberately, as a checklist over the new
path's siblings, is what caught it within hours.

**A range that subsumes loaded ranges replaces them, before it loads**
(2026-09-05, report #010). `model_worker::subsumed_segment_keys`; the worker's
`models` map is keyed by the exact `(start, end, tp_rank, tp_size)`, so a plan
restating coverage the worker already has — [16..48) plus [0..16) becoming
[0..48) — misses, and the whole model is read from disk again beside the copy
already resident. Measured: 63 seconds, and sustained swap on a 16 GB
processor-only node.
**Dropped BEFORE the load**, so the process holds `max(old, new)` and not
their sum; the peak is the thing that kills a small machine. **Strict
subsumption only** — a partial overlap describes layers each range still needs
— and **within one tensor-parallel shape**, since another rank's range says
nothing about this one's.
**Do not "fix" this by skipping the CHARGE for a covered range**, which is what
the report proposed: the worker really did load a second copy, so the charge
was accurate, and skipping it would have under-counted real memory on a machine
that was already swapping. The accounting was the part working correctly. Ask
which side of a double-count is the lie before removing either.

**What WAS missing is the release, the mirror image of that** (2026-09-08).
The worker drops the covered ranges and frees their memory; the daemon went on
charging for them, because `charged_segments` recorded which ranges existed and
nothing ever removed one. A worker that had consolidated its coverage kept
paying for what it had dropped — and since the charge is what admission weighs,
the node then refused later models that would have fitted.
`WorkerHandle::release_subsumed_segments` mirrors `subsumed_segment_keys` on the
daemon side, which is why `charged_segments` now records what each range COST
rather than just which ranges exist.
Three things a change must keep. **The release happens AFTER admission**, never
before: admission is deliberately weighed against everything still charged, and
a refusal means the forward is never sent and the worker never drops anything,
so releasing first would free a charge for memory still held. **Strict
containment only**, the worker's own rule — a partial overlap is two ranges that
each still need their layers. And **the subtraction saturates**, because an
under-run on a `u64` budget is 18 exabytes of free memory and admits everything
for ever.
**Known gap, pre-existing rather than introduced**: the worker keys its map by
`(start, end, tp_rank, tp_size)` and drops within one tensor-parallel shape,
while the daemon's charges carry no rank — `record_charged_segment` is
idempotent on the range, so a range serving several ranks is charged once for
all of them. Under tensor parallelism the release can free a charge the worker
only partly dropped. Smaller than the error being fixed and in the same
direction as the daemon's existing simplification; a rank-aware daemon model is
the real fix (`docs/FUTURE_WORK.md`).

**A worker's charge is released by SUBTRACTING what THAT worker owed**
(2026-09-05). `WorkerHandle::charged_mb` records the spawn's admission charge
plus every range `charge_additional_segment` adds; `charged_segments` records
the ranges. `release_reserved` subtracts and removes the key only at zero.
**Why**: `ram_reserved_mb` / `vram_reserved_mb` are keyed by `ModelId` and
`add_reserved` accumulates — right, because one worker can hold several
segments — so a release that dropped the key could not say "only this
worker's share". Several workers' lifetimes overlap under one id: a
replacement is admitted and charged while the corpse is still in `workers`,
and dropping the key discarded the replacement's charge, leaving it running
un-accounted. That is UNDER-charging, the opposite direction from #461/#467
and milder, but the same class.
Taking `spawn_lock` in the eviction paths would also close it and is the
wrong trade — six of those callers are on the request hot path and that lock
is held for a whole model load, minutes on a processor.
Three things a change must keep. **The subtraction saturates**: an under-run
wraps a `u64` budget to 18 exabytes and admits everything for ever. **It
comes off the budget the worker was charged against** (`charged_against_ram`),
or the two drift apart on churn. And **`unload_model` reads the figures before
dropping the handle** — it waits for the process to exit afterwards, which
outlives the handle; `DepartedWorker` carries them across.
A test that charges the pool but leaves the handle's figure at zero now
asserts nothing; `admit_and_insert_cpu_worker` does what a spawn does, in the
order it does it.

**A worker that DIED gives its budget back too** (2026-09-04, gotcha #461).
`ModelProcessPool::retire_dead_worker` is the single answer to "this worker's
process is gone": under `spawn_lock`, `remove_if(dead)` then
`release_vram_charge` + `release_ram_charge`. Every site that discovers
`handle.dead` goes through it, and so does `reap_dead_workers` on the health
tick.
**Why both.** Only the graceful `unload_model` released the charge; the three
`dead` fast-fail sites did `workers.remove(&model_id)` and nothing else. So a
worker that exited any other way — an internal crash, an OS OOM-kill, or a
user closing the process in a system monitor to free memory — left its whole
reservation charged until the daemon restarted. Measured on a 16 GB Mac mini:
six consecutive requests refused with a byte-for-byte identical "11487 MB is
already in use", over a minute, immediately after real free memory had gone
UP.
And the call-site half alone is not enough: the charge is ONE shared budget
(`ram_committed_mb` sums every model), so a dead 14B refuses every OTHER
model, while the call-site check only fires if someone asks for *that* model
again — which nobody need ever do. `spawn_lock` is required because a spawn
charges and inserts under it, and releasing between the two would free the
NEW worker's charge; `remove_if` is required so a late caller holding the
corpse cannot retire the live worker that replaced it.
**Ask of any "clean up on next use" fix: what if there is no next use, and
who else is paying meanwhile?**

**Retirement DRAINS, it does not kill** (2026-08-29).
`unload_model` waits for the worker's `responses` map to empty
(`await_responses_drained`, bounded by `WORKER_DRAIN_WAIT`) before sending
`DaemonMsg::Shutdown`. Every retirement funnels through `unload_model` —
`free_vram_for_admission` against its victims, `worker_should_return_to_gpu`
against the processor copy it replaces, the idle timer — so all of them
inherit it, and a new displacement path gets it for free.
**Two things worth knowing before touching it.** The "stop admitting" half is
already done by `workers.remove`, because every forward path re-acquires the
handle from that map; a request arriving after the remove spawns a fresh
worker. And **dropping the handle is not what kills the in-flight request** —
the map holds an `Arc` and the caller holds another, so the child outlives the
drop. The explicit `Shutdown` is the entire race, which is why one wait in one
place closes it rather than the cross-cutting change `docs/FUTURE_WORK.md`
anticipated.
The bound is not negotiable: a request that never completes must not hold that
model's memory for the daemon's lifetime, since that would refuse every later
load on the device — the same trade `WORKER_EXIT_WAIT` already makes, and a
worse failure than the race. It reports what it stranded. Retirement only ever
targets idle models, so the common path returns without sleeping at all, and a
test pins that so no unload pays for a race that is not happening.

## `model::auto_manage::storage_budget` is the ONE answer to "how much shard storage may this node hold?", and `held_shard_bytes` the one answer to "how much does it hold?"

(2026-09-03, gotcha #448). `storage_budget_now(&state)`
gives both for this node, live. Consumers: the download pass
(`scoring::remaining_budget`, whose refusal logs every figure and the rule
that produced it), prune's disk pressure (`prune::compute_resource_pressure`),
the settings storage bar (`api::admin::storage_breakdown`), the pool page
(`api::pool`) and the diagnostics report's `storage:` line.
`the_storage_budget_has_one_accountant` in `tests/repo_consistency.rs`
fails the build on a new spelling of the rule.
**Why**: there were three. The download pass quartered the figure for
Minimal contribution (the DEFAULT level) — half of `max_disk_mb`, then a
quarter of that, 6.25 GB on a stock install — while prune pressure and the
pool page used the unscaled figure, and the settings bar drew the cap as
headroom AFTER "used". A tester holding 18 GB against an explicit 50 GB
read "no remaining storage budget" every cycle with nothing on any surface
saying what the budget was, and built a careful theory about phantom
manifest reservations — there is none; held bytes come from the registry's
reverse index, so a manifest with no local shard contributes nothing. The
node was over budget for downloading and at 36% for pruning, so it refused
every download and pruned nothing, for ever. **Two accountants for one
resource wedge exactly where they disagree** — the same shape as the
graphics-memory rule above, on the disk.
The rule: an explicit `max_storage_mb` is honoured as written (the VRAM
precedent — a number the user typed is not silently scaled by a level they
may not connect to it); otherwise 25 / 50 / 75% of `max_disk_mb` by
contribution level, the shares the setup wizard has always promised
(`contribution_disk_share_pct`); never above `max_disk_mb`; never above
held + 80% of free disk — the held term makes the clamp invariant under our
own holdings, where the old form subtracted held from a figure that already
excluded it. **A refusal must name its arithmetic**: `held_mb`,
`budget_mb`, `budget_from`, and what to do about it. A competent reader
handed a bare "no remaining budget" will build a theory from the numbers
they CAN see.
Two siblings fixed in the same pass: `evaluate_and_download` read
`max_storage_mb`/`max_shards` from the boot snapshot through a local
binding the live-config guard cannot see (#281's shape); and the quarantine
sweep named only `.quarantine`, so `.mismatched` files (2026-07-27) were
never reclaimed — `QUARANTINE_EXTENSIONS` now lists both.

## `model::auto_manage::prune::effective_idle_secs` — residency is a hard UPPER BOUND on "idle since", and the worker's own `last_used` is the signal that moves

(2026-09-02, gotcha #437). The idle unload used to trust
`model_trust.last_request_at`, which NOTHING in the current code writes; a
two-day-old persisted value outranked a worker loaded 215 s earlier and the
model was unloaded five seconds after answering, so every request after that
paid a cold reload and lost the worker's prefix cache. `ModelProcessPool::
model_idle_secs` (seconds since `register_response`, the one place every
execution path passes) is now a fourth input, and residency clamps the answer.
A new "how long has X been idle" judgement must be bounded by "how long has
X existed", and must read a signal the COMMON path actually writes — grep for
the writer before trusting a doc comment that names one.

## An admitted prompt is RECORDED, not just decided

(2026-09-05).
`KvCacheStore::record_prompt_admission` / `outstanding_admission_bytes`;
`ensure_room_for_prompt` adds the outstanding total to the live figure before
calling `admit_prompt`, and records its own claim once admitted.
**Why**: admission reads occupancy, decides, evicts and returns — and the
prefill allocates afterwards. On the batched path they are not even adjacent:
`admit_slot` marks the slot `Prefilling` and returns to the worker's message
loop, with the chunks run on later scheduler ticks. So the next prompt is
weighed against memory the previous one has already been promised, and the
loop being strictly sequential does not help — the window spans a return to
it. The coordinator-side reservation (`peer_vram_commitments`, #457) makes two
large prompts reaching one worker rare, not impossible, and the peer's own
admission is what these rules call the backstop.
Three things a change must keep. **The claim is drawn down by what that
request has actually allocated**, so nothing is charged twice — and a claim
nothing removed contributes zero once its prefill finished, which is what
bounds a leak. **`clear_request` releases it**, so all eight worker paths that
end or abandon a request inherited the release unedited and a new one cannot
forget. And **the TTL sweep covers the case draw-down cannot** — a prompt
admitted and then never prefilled at all. `promised_mb` appears beside
`live_mb` in both DIAG lines, because a refusal caused by an invisible
reservation is the kind of thing a reader invents a mechanism to explain.

## `inference::split::kv_budget::admit_prompt` + `PrefixCache::release`

(2026-09-02, gotcha #440) — ONE decision for a whole prompt, before prefill,
charging live caches PLUS the prefix cache's snapshots (the same device
memory, previously charged nowhere): fit → evict cached prompts, oldest hit
first → refuse with a 503 at token 0. Wired at both worker entry points
(`ensure_room_for_prompt`, after the lookup and before hydration — hydration
is itself an allocation of the matched prefix). The per-chunk guard below now
charges `KvOccupancy::external_bytes`, set by the worker after every
snapshot insert or release. **A budget must see every tenant of the memory
it bounds**: on an 8 GB card the second 6.4k-token prompt found ~300 MB free,
was admitted against a KV-store-only figure, and its cache landed in WSL2's
host-backed memory — 3-5 tok/s where the empty card did 19-33, with nothing
refused and nothing logged. A cache of reconstructible data ranks below the
request in hand. **The snapshot taken AFTER a prefill is sized the same
way** (`plan_snapshot` → `insert_from_kv(.., max_positions)`): it is a full
copy of the request's own cache, and taken whole it put the card straight
back over the top — every reply cut at the next growth quantum, measured
after the admission half alone had shipped to the card. Older prompts go
first, then the snapshot is cut to what fits (a partial prefix still saves
its length next turn — `lookup` narrows), then skipped.
`SWARMLLM_KV_PREFIX_CHARGE=0` disables all of it for A/B.
**The per-chunk guard goes through `KvCacheStore::claim_room`**, which
evicts through `set_external_evictor` (installed by the worker over its
prefix cache, `Weak` on both sides) BEFORE refusing. Shipping the charge
without the eviction (v0.3.149) refused admitted prompts mid-prefill
where the release before had served them slowly. **A guard that can see a
reclaimable tenant must be able to reclaim it, or it is stricter than the
guard it replaced.**
**The reconciliation covers the PROCESSOR too** (2026-09-04, gotcha #462).
`device_free_and_total_bytes` answers for `Device::Cpu` from sysinfo's
`available_memory` (cached 250 ms — every caller is already off the
per-token path), so a CPU worker's budget is reconciled exactly as a card's
is. It had a CUDA arm and no processor arm, so `kv_budget_now` returned the
load-time figure unchanged and a CPU worker's ceiling stayed the grant the
daemon made at spawn, for life: it could not see a second worker start, the
machine fill, or **its own weights grow** when repeated local-standby
failovers took it from 12 to 29 of a 48-layer model — the process was killed
and a reply that had already streamed 238 tokens over ~10 minutes was lost.
`available`, never `free`: reclaimable page cache is memory this process can
have. **A reconciliation written for one device is not device-independent
because its arithmetic is — grep the "cannot say" arm and ask which
population lands there.** Here it was every processor-only node.

**The budget is reconciled with the CARD at every decision that takes
device memory** (2026-09-03): `SplitModel::kv_budget_now(live, cached)` =
`min(load-time budget, live + cached + free_now − margin)`, `free_now` from
cudarc's `mem_get_info` (microseconds), margin 5% of the card with a
256 MB floor (`kv_budget::budget_reconciled_with_device`). Asked by
`ensure_room_for_prompt`, `snapshot_positions_that_fit` and the per-chunk
guard — never on the per-token path. **The load-time budget is a
prediction; the card is the fact**: `kv_headroom_bytes` was taken from free
memory at load and could not see a tenant that arrived later (a second
worker, the full build's llama.cpp context, a snapshot), so on the released
v0.3.149 it said 4491 MB of room where the card had ~2 GB, and the admitted
prompt's cache spilled to host memory at 1.95 tok/s. The reconciled form is
invariant under evicting a cached prompt (bytes move from `cached` to
`free_now`), which is what keeps `admit_prompt`'s evict-then-fit arithmetic
valid against it. A device that cannot say (the processor) leaves the
load-time figure alone; `None` still means unknown, never zero.

## `inference::split::kv_budget`

(2026-08-08) — the KV memory budget and the
admission check against it. The loader records `kv_headroom_bytes` on the
model; `forward_inner_impl` checks `quantum_exceeds_headroom` before a forward
claims another growth quantum, and refuses with `ServiceUnavailable` (503,
so a coordinator re-routes to a peer). **Do NOT re-introduce a load-time
context clamp** — one existed, it shrank every user's context so a single
full-length conversation would fit, and it did not bound concurrency at all.
Three invariants a new caller must preserve: the check runs ONLY when
`positions_claimed` is non-zero (otherwise it walks the whole store per
generated token for an answer that is almost always "no"); it charges the
POSITIONS claimed, not one quantum, because a prefill jumps many quanta in a
single forward and charging one under-counted the largest claim a request
ever makes by 10x; and `kv_budget_bytes: None` means UNKNOWN, never zero — every CPU node
and any GPU node whose free VRAM could not be read records `None`, and reading
that as a zero budget refuses everything.

## `inference::process_pool::worker_socket_path`

(2026-09-04, gotcha #449) —
the worker IPC socket path, and the ONLY place it is built. `sun_path` in
`sockaddr_un` is a fixed array of **104 bytes on macOS/BSD and 108 on
Linux**, so a path one byte over does not truncate — `bind` refuses, the
worker never starts, and since prompt privacy keeps the first and last
layers local, the node answers nothing at all whatever the swarm holds.
That was every request on every Mac: macOS hands each user a private
per-boot temp dir (49 characters, measured) and the name was
`swarmllm-worker-<36-char uuid>.sock` (57).
Three things a change must keep. The name stays SHORT
(`worker_socket_filename`, 12 hex chars — every character here is one the
directory cannot use). `$TMPDIR` is tried first (per-user and private on
macOS, short on Linux) and `/tmp/swarmllm-<uid>` only as a fallback,
created 0700 and **verified after creation** — `/tmp` is world-writable, so
a pre-created hostile directory is the attack, and the check is
`symlink_metadata` + uid + `mode & 0o077 == 0`, refusing rather than
repairing. And the arithmetic lives in `first_dir_that_fits`, a pure
function tested against the literal 104: **a platform-dependent length
limit is invisible to a single-platform suite**, so a test that asks the
host cannot see the bug that only exists on the other host.
