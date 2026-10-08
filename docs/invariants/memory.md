# Worker memory: graphics, RAM and the KV cache

The evidence behind the rules in `.claude/rules/arch-worker-memory.md`: what each
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

## A conversation that is over gives its room to one that is not

(2026-10-08, FUTURE_WORK #235, report #005.) A reply 71 s into a whole-model
hand-off was refused by the peer serving it — "other conversations on this node
are using 330 MB of the 469 MB budget. It fits once they finish" — and, already
streaming, could not be re-planned. Two things made those conversations hold the
room, and neither is the peer's fault:

- **A node serving a SEGMENT is never told the reply finished.** `ReleaseRequestKv`
  (above) is sent by `release_request_kv` to the COORDINATOR's own workers; a peer
  running layers for that request keeps the cache. The section above assumed a
  ten-minute idle sweep would bound it.
- **A worker never ran that sweep.** Its `KvCacheStore` was built with the TTL
  (`--kv-cache-ttl`, 600 s) since the store's first version, and `cleanup_expired`
  was called only on the DAEMON's store (`router/mod.rs`'s cache tick). So every
  segment a node served for another computer stayed in its worker's store for as
  long as that worker lived, and counted as "other conversations" against every
  live one.

Fixed on both sides of the trade. The worker sweeps on its 30 s occupancy tick
(guard `a_workers_kv_store_is_swept_by_the_worker_that_builds_it`, with a planted
self-test). And when a live conversation needs the room — a prompt's admission, a
reply's growth claim — caches silent for `FINISHED_CONVERSATION_AFTER` go first,
before the prefix cache's snapshots (which may yet be reused) and before any
refusal. That figure IS `process_pool::CONVERSATION_GAP_SECS`, the daemon's
reading of "this conversation is over", so the two cannot disagree.

**The trade, stated.** A conversation that was only WAITING that long — a long
prompt pass on a later machine of its chain — loses its cache here and is refused
at its next forward (`forward_lacks_its_conversation`, visibly, never decoded from
nothing). It has shown its reader nothing yet, so the router re-plans it. A reply
refused mid-stream cannot be re-planned at all. vLLM's preemption chooses the same
way: the request that can be recomputed gives way to the one being served.

Tests: `a_finished_conversation_gives_its_room_to_a_live_one` (red with the release
off; controls: a conversation silent for half the gap keeps its cache and the claim
is refused as before, and the claimant's own cache is never released),
`the_ttl_sweep_removes_a_conversation_nobody_released`.

**And the peer is now told (#238).** Each attempt's executor records the peers it
ran segments on — as planned and as it ended, so a stand-in that took a segment
over is included (`SharedState::note_request_peers`). When the REQUEST is over,
`finalize_request` sends each of them the existing `CancelInference` once
(`router::release_request_on_peers`), and the worker drops a cancelled request's
cache in its main loop, between messages (`release_caches_of_cancelled`). It is the
request's end and never an attempt's: a worker skips the next forward of an id it
saw cancelled (60 s), and a router retry reuses the request id — gotcha #749's
shape, which is what placed it. An older peer forgets the conversation and keeps
the cache to its own timers. Tests `a_finished_request_tells_each_of_its_peers_once`,
`a_cancelled_requests_cache_is_released_once` (each red with its half off). Rig
(`~/swarmllm-rig-234/release238.sh`, TinyLlama split over two processor nodes, debug
logs): for each of two requests the coordinator logged `told=1` as it finished and
the peer logged that request's `CancelInference` ~10 ms later, its worker
`request cancelled by daemon`; both replies 200. What
remains: seeing the peer's room before choosing it (#237); a reply whose peer
refuses mid-stream still ends (#236).

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

## A worker's growth is weighed by the budget its spawn charged

(2026-09-13, report #030, gotcha #586.)
**`WorkerHandle::holds_gpu_memory` is the single answer to "does this worker's
memory come out of the graphics budget or the system-RAM one?"**, and it reads
`charged_against_ram` — what `charges_ram` decided at spawn — never
`placed_on_cpu_because`.

**Why the two are not interchangeable.** `placed_on_cpu_because` records why a
model was *demoted*, for the user and the dashboard. On a machine with no
graphics card nothing was demoted — there is nowhere to demote to — so it reads
`None`, exactly as it does for a worker holding a card. `charges_ram` is the
other question, "who pays", and it knows the two cases `cpu_reason` cannot see:
*no card detected* and *this build has no CUDA*.

**What it replaced.** `charge_additional_segment` re-derived `on_gpu =
handle.placed_on_cpu_because.is_none()`, so on every GPU-less node each later
layer-range growth of a live worker was weighed by `admit_to_gpu` — which
returns `true` unconditionally when `vram_budget_mb` is 0, as it is on a
machine with no card. The delta was also priced with the VRAM estimator and
subsumed ranges released from the VRAM map. Only a worker's FIRST admission
ever met the real anti-swap check.

**Measured.** A 16 GB CPU-only Mac mini on v0.3.177: the 14B's spawn was weighed
honestly (`estimated_mb=470 cap_mb=13107 available_mb=12092
live_headroom_mb=8400`), then four growths — `delta_mb=6090`, `2730`, `2520`,
`210`, ≈ 11.5 GB — were each logged `on_gpu=true` and admitted without a check.
`grep -c "admitting model to system RAM"` over the whole run returned 3, one per
model spawned. No refusal line anywhere; the machine swapped.

**What a change must keep.**

- **Growth is the common case, not the rare one.** A swarm node's coverage is
  reassigned by scheduling, failover and re-plans. A gate that only runs on the
  create path is a gate that mostly does not run — the same lesson the KV grant
  learned as gotcha #440, recorded in this very function's own comment ("the
  bound lives with the worker … reconciles at every decision that takes
  memory") while the weights beside it still decided once.
- **One accountant per worker, chosen once.** `charged_against_ram` is set at
  spawn and every later charge, release and price must follow it. A worker
  charged against RAM must hold no VRAM reservation: the spawn's own
  `admit_to_gpu` charge is released the moment `charges_ram` says RAM, which
  the RAM-refusal arm below it had always done and the success arm had not.
- **`placed_on_cpu_because` answers only "why was this demoted".** Three other
  readers were asking it the accountant's question. `would_fit_on_gpu` and
  `gpu_estimate_and_fit` answered `Some(true)` — "it fits on your GPU" — for
  every resident model on every Mac; the honest answer with no card and no
  budget is `None`, which is what the API documents that field to mean.
  `DepartedWorker::freed_gpu_memory` logged `device="gpu"` for workers that had
  never touched one.
- **The guard.** `a_live_workers_growth_is_weighed_by_the_budget_its_spawn_charged`
  in `tests/repo_consistency.rs` fails the build if `charge_additional_segment`
  mentions `placed_on_cpu_because`, scanning statements so a chain rustfmt has
  wrapped is still seen, and with its own planted-violation self-test.

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
**And it charges what it places, a split included (2026-10-08, #240).** A worker
the pool placed part on the card and part on the processor (`partial_gpu_layers`)
was charged NOTHING — the spawn's `charged_vram_mb` stayed 0 on that branch. The
card share then read as ANOTHER program's memory (`compute_vram_budget`'s
`other_process_mb` is the card's used total less what WE charged), which hides it
from two decisions: a configured `max_gpu_vram_mb` caps only what is charged, so a
14B split on the card took 5,782 MB against a 5,000 MB cap once a further range
grew onto the card (found by the #234 rig); and the reclaim's candidates are
workers WITH a charge, so an idle split worker could never be unloaded for another
model. The share is now the first `n` layers of the segment by the admission
estimator, never past the room the split was sized against
(`card_share_of_split_mb`). Measured on the card with TinyLlama admitted beside a
14B split (16 of 48 layers on the card, 3,851 MB measured): v0.3.230's admission
read `committed_mb=0 budget_mb=2666`, the fix `committed_mb=3691 budget_mb=5000`.
Test `a_split_charges_the_card_for_the_layers_it_puts_there`.
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
`SharedState::split_model_budget_with` + `committed_memory_mb` + the
`MemoryScope` enum answer "should this node advertise another segment as
locally servable?" — `compute_vram_budget` (the card) or, on a node with no
card, `inference.max_split_model_memory_mb`. `MemoryScope` exists because
those two were reached through one `.or()` and describe different memory.

**And the budget is charged by the component that OWNS the memory** —
`ModelProcessPool::vram_committed_mb` / `ram_committed_mb` — not by summing
`estimated_vram_mb` over `split_models`, which is what it did until v0.3.190
(`docs/FUTURE_WORK.md` #55). Those entries are GGUF headers read while scanning
the models directory: no worker, no allocation, just a prediction about a model
that may never load. So the cap filled at scan time with memory nothing held and
could never fall — measured on the live node 2026-09-17 as `loaded_mb=5124`
**eighteen seconds after boot with zero workers spawned**, against 2027 MiB
actually on the card and the pool's own `committed_mb=1044` in the same second.
The consequence was not a log line: the cap consumed itself permanently in scan
order, so a node locally served only the first card's-worth of models it
happened to scan, and the scheduler reported even a 0.5B as "does not fit our
GPU" and delegated it to a peer.

**The tell was already in the code.** The deleted helper's own doc said "this is
a registration figure, not a residency figure" and named `vram_committed_mb` as
the residency one; the budget read the registration figure regardless. A doc
comment describing the trap did not stop the trap — the guard
`a_memory_budget_is_charged_by_the_pool_never_by_the_metadata_map` does, and
`model_uses_gpu_memory` (the placement predicate that existed only to filter
that sum) went with it. The placement rule itself is unchanged: `charges_ram`
decides at spawn, is recorded as `charged_against_ram`, and is read back by
`holds_gpu_memory`.

**The rule to carry**: a budget must be charged by whatever admits and releases
the resource, so the figure falls again when the resource is freed — and a
component that does not own a resource must not be able to reclaim it.

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

**And the ADMISSION weighs the same peak the drop produces** (2026-09-24,
FUTURE_WORK #95). The worker drops the subsumed ranges BEFORE loading, so its
peak is the new range LESS what they held — yet admission weighed the full
width against a total still counting them, and refused a consolidation that
fits: [0..2) + [14..40) → [0..40) of a 40-layer model was charged 40 new layers
when it needed 12. `charge_additional_segment` now admits
`delta − WorkerHandle::subsumed_charge_mb` (read-only, same predicate as the
release, `strictly_subsumes`), then records the segment at its full cost and
drops the subsumed records. **It must NOT release them from the pool as well**:
`admit_to_*` both checks AND charges, so the pool already took the net figure,
and releasing the subsumed charge again frees it twice (the test pins the pool
at 4250 where a double release reads 1650). "Admission before release" is kept
— nothing is released until admission has answered, so a refusal leaves every
charge standing. The planner prices a local range with the same discount
(`process_pool::layers_added_by`), which is why the SPAWN's segment discounts
nothing: it is recorded at zero MB, so its subsumption releases nothing. →
`docs/invariants/scheduling.md` § "Room for more layers is not room for the
layers already held".

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
held + what is free beyond 10% of the filesystem (`FREE_DISK_RESERVE_PCT`,
2026-10-06 — it was held + 80% of free, which had no floor; see the next
section) — the held term makes the clamp invariant under our own holdings,
where the old form subtracted held from a figure that already excluded it.
**A refusal must name its arithmetic**: `held_mb`,
`budget_mb`, `budget_from`, and what to do about it. A competent reader
handed a bare "no remaining budget" will build a theory from the numbers
they CAN see.
Two siblings fixed in the same pass: `evaluate_and_download` read
`max_storage_mb`/`max_shards` from the boot snapshot through a local
binding the live-config guard cannot see (#281's shape); and the quarantine
sweep named only `.quarantine`, so `.mismatched` files (2026-07-27) were
never reclaimed — `QUARANTINE_EXTENSIONS` now lists both.

## `AutoShardManager::would_shed_copy` is the ONE answer to "would prune shed this copy?" — and the download pass asks it before every fetch

(2026-10-06, gotcha #795.) Prune asks it of each part it holds; the download
pass asks it in `select_within_budget` of each part it is about to fetch,
counting this node among the live holders and judging at the disk pressure
that part — and the others chosen the same cycle — would leave behind
(`StorageReading::with_added`). A part prune would shed is not fetched.
Guards: `fetch_what_prune_keeps::*` in `auto_manage/scoring.rs` (each fails
with its half of the fix switched off, checked 2026-10-06).

**The download pass was not the only way in (found the same day, on v0.3.228,
gotcha #797).** `complete_pending_shard_fetches` finishes fetches a peer
transfer could not prove intact, from `shard_p2p_failed`, and asked nobody.
That set was drained only by a successful PEER transfer, so a part fetched
from the origin stayed in it for the node's life, and "the file is on disk"
was the only thing stopping a re-fetch. On a tester's RTX 4050 node
(`bf7b3263`, disk pressure 0.56, nothing to do with the storage limit), prune
deleted Qwen3-30B part 17 as surplus (`holders=3 target=2`) and the pass
fetched it back from HuggingFace 15 s later (`Fetching from the model's
origin — no peer copy could be verified`; the auto-manage cycle itself logged
`0 download(s) started`) — every ~5-6 min from 11:36 UTC, ~4 GB an hour. Two
changes: `SharedState::announce_shard_acquired`, where every origin download
ends, removes the part from the set (its own doc already said "cleared when
a download for the shard successfully completes"); and the pending pass asks
`would_shed_once_fetched` and DROPS an entry prune would delete again —
dropped, because a stale entry also keeps the evaluation cooldown bypassed
(the tester's log shows prune evaluating every ~15 s). Guards:
`pending_fetches_follow_prune::*` in `auto_manage/manager.rs`, each red with
its half switched off. **A change must keep**: any new automatic fetch path
asks `would_shed_copy` too — this rule is about FETCHING, not about one pass.

**And an entry nothing can fetch is dropped (2026-10-07, the residual).** The
pass skipped (`continue`) an entry whose model had no recorded origin, and its
own check was `hf_sources.contains_key` — not `can_fetch_shard_from_origin`,
the single answer, so offline mode slipped through too. Such an entry is not
pending, it is stuck, and worse than the CPU cost recorded first: the
download pass reads the set as "peers are exhausted for this part" and goes
straight to the origin, so with no origin the part was never fetched again
for the node's life. It arises from the stalled-permit sweep, which adds an
entry with no regard for an origin — a model with no HuggingFace source, or
an offline node, whose peer download stalls once. Now the pass asks
`pending_fetch_can_proceed` (a repair, or `can_fetch_shard_from_origin`) and
drops what it cannot serve; the part is the download pass's again, and with no
origin the accept path keeps a peer's copy. Guards
`a_pending_fetch_with_no_origin_goes_back_to_its_peers`,
`an_offline_node_hands_a_pending_fetch_back_to_its_peers` (both red with the
predicate forced true); the prune test now gives its model an origin, so the
prune check — not the missing origin — is what drops its part.

**Why**: the two passes answered "how many copies are enough?" separately.
Prune shed above `pressure_adjusted_target` — one copy fewer above 0.8 disk
pressure, two above 0.95 — while `gather_candidates` fetched below the RAW
`geo_target_replicas`. A part between the two was surplus to one pass and
missing to the other. Field evidence, all from the live node's log of peer
announcements: a tester's 30 GB container (`e561df35`) went 62 → 59 → 62
parts every ~25-30 min from 02:00 UTC 2026-10-06, one Qwen3-30B-A3B part
dropped per `prune_cooldown_secs` (300 s) and the SAME indices re-fetched once
`SHARD_RECENTLY_ACQUIRED_SECS` (30 min) lapsed — shard 4 re-fetched 04:03,
dropped 04:37, back 04:38; ~1.6 GB a cycle. It began when the same tester's
second Australian node took copies of those parts, which stopped
`would_eliminate_region` blocking the prune. Two other peers showed the same
signature for days (`9594e1ff` ~100 parts a day across 7 models,
`bf7b3263` 170 on 10-01). It is #448's shape again: two accountants for one
decision, and the node oscillates exactly where they disagree.

What the shared method covers, and what it deliberately does not:
- **In**: what DECIDES a prune — prune disabled, the per-model policy, a locked
  part, a pool pin to this node, prompt privacy's ends, a user-pinned model,
  the configured `--shards` range, the replica count at disk pressure
  (`sheds_at`, over `effective_prune_target`), and the last copy in the
  region. A download refused on any of these would under-fetch a part prune
  keeps — `a_part_prune_never_deletes_is_fetched_at_any_pressure` pins the
  configured-range case.
- **Out**: what only POSTPONES one — cooldown, a download in flight, a part on
  an active pipeline, busy holders, a recent request, a way to re-fetch. A
  part they shield today is shed tomorrow; the download pass must not count on
  them.
- **Pressure is DISK pressure for every file decision.** Prune used
  `max(disk, vram)`: deleting a part frees no graphics memory (prune even
  prefers parts that are not loaded), and a card fills whenever a model is
  resident, so a GPU node shed parts the download pass fetched back as soon as
  its card emptied. `max(disk, vram)` now gates only phase 0's soft-unload,
  unchanged.
- **The reserve is a FLOOR on free space, not a share of it.** "Held + 80% of
  free" kept offering room while a part fit in 80% of what was left: 550 MB
  parts filled a 30 GB container (3 GB of system) to 148 MB free, 99.5% —
  into the tester's own fill safeguard, and to disk pressure ~1.0 where prune
  is most eager. Kubernetes' kubelet treats a node filesystem under 10%
  available as a hard eviction threshold (`evictionHard: nodefs.available<10%`,
  KubeletConfiguration default), and its image GC deletes above 85% and stops
  under 80% — a delete trigger above the point acquisition stops at, which is
  the property `select_within_budget` now gives this pair. The same container
  now stops at 88.8% (`downloading_until_the_budget_says_stop_leaves_a_tenth_of_the_disk_free`).

What a change must keep: one method answers for both passes; the download
side judges at the pressure AFTER the fetch, cumulatively within a cycle
(`parts_chosen_together_are_judged_together`); file decisions never read
graphics memory.

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
request has actually allocated**, so nothing is charged twice. **`clear_request`
releases it**, so all eight worker paths that end or abandon a request inherited
the release unedited and a new one cannot forget. And **the TTL sweep covers the case draw-down cannot** — a prompt
admitted and then never prefilled at all.

⚠ **The draw-down is NOT a leak bound, and this file said it was for two
releases** (corrected 2026-09-17, gotcha #637). "A claim nothing removed
contributes zero once its prefill finished" holds only while the request's cache
entry is in `caches`, because that is where the draw-down reads what was
allocated. Delete the entry and the subtrahend goes with it: the claim springs
back from zero to its FULL size. `cleanup_request_id` did exactly that — it
released `reserved_positions` and the caches and left `admitted_claims` — so
every request on the SEGMENT path (`DaemonMsg::ReleaseRequestKv`, which is every
request on a node serving a model it holds, since a single local segment is
still assembled as a pipeline) left its whole admission owed until the TTL
sweep. Measured on the live node: a 5101-token prompt left 1848 MB owed against
a 3042 MB conversation budget, and identical prompts were refused for ten
minutes with advice ("close other programs") that could not work. The two maps
are released together by `forget_request_bookkeeping`, which both cleanup paths
call. **Introduced by the fix for report #019** — before it the forward path
released nothing, so the entry stayed and the draw-down was correct; freeing the
memory is what started the phantom charge. Pinned by
`a_finished_requests_admission_claim_does_not_outlive_its_cache`.

**The general form**: a quantity defined as "X minus what Y has already done"
becomes plain X the moment Y is deleted. Any offset that reads a second
structure must be asked what it reads when that structure is gone. `promised_mb` appears beside
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

## A card's free memory is read after a synchronize

(2026-09-26, `docs/FUTURE_WORK.md` #121.) `kv_budget::device_free_and_total_bytes`
synchronizes the device's stream before `mem_get_info`, and it is the only
reading of device memory behind a budget decision (`SplitModel::kv_budget_now`
→ admission, snapshot sizing, the executor's growth guard).

**What it replaced.** The reconciliation above says an eviction "moves x from
`cached` to `free_now`". On a card that was never true at the moment it was
read. cudarc 0.19.9 frees every `CudaSlice` with `cuMemFreeAsync` when the
device supports memory pools (`Drop for CudaSlice`, gated on `has_async_alloc`),
and a pool returns freed memory to the device — the only memory `cuMemGetInfo`
counts as free — at the next stream, event or context synchronize: its release
threshold defaults to zero and nothing in this repo or the vendored candle
raises it (CUDA Runtime API § Stream Ordered Memory Allocator,
`cudaMemPoolAttrReleaseThreshold`). So the reading after an eviction, or after
the previous request's cache was dropped, still counted those bytes as used.

**Measured** on the released v0.3.207, isolated node, RTX 3070 Laptop 8 GB, one
model, nothing else in flight. llama-3.1-8b: a 1,835-token prompt served, the
next of the same size refused — "0 MB in use, 683 MB available" against a
1,336 MB budget — four runs of four, with the prefix cache ON and OFF alike, so
the cache was not the cause: the first request's own released cache was enough.
The same shape on llama-3.2-3b, qwen2.5-coder-7b (right after the loader logged
that the budget "comfortably covers" 9,405 tokens), phi-3.5-mini and
qwen3-1.7b. The budget dropped by exactly what had just been released: phi
2,197 → 780 MB (the 1,417 MB evicted 674 µs earlier); qwen3 2,899 → 1,410 MB a
full 1.4 s later, so this is not a race to be waited out.

**Verified**, A/B/A/B inside the new `--features cuda` build with
`SWARMLLM_KV_DEVICE_SYNC=0` as the control, confirmed in the worker's own
environment: with the synchronize, three 1,835-token prompts on the 8B all
served, zero refusals, twice; without it the second was refused against
661 MB, twice. qwen2.5-coder-7b's three 4,029-token prompts and phi-3.5's
three 2,011-token prompts, each refused on v0.3.207, all served.

**What a change here must keep.** The synchronize stays inside the one reading,
not at its callers — every caller would otherwise have to remember it, and a
new one would not (`.claude/rules/architecture.md` § "One invariant, N paths").
It costs a wait for queued work, so the reading stays off the per-token path:
admission, snapshot sizing and the guard on a growth boundary only. The
loader's figure at load (`query_gpu_vram_free_mb`, before the model's weights
exist) has nothing of ours to miss and is left alone. Guard:
`the_cards_free_memory_is_read_after_a_synchronize`, with the unsynchronized
source planted in `the_free_memory_sync_guard_catches_an_unsynchronized_read`
— the arm is CUDA-gated, so no default build or test compiles it.

### A reply is reserved by what it can reach (#122)

`ensure_room_for_prompt` takes the reply's reserve as a REQUIRED argument:
`reply_reserve_positions(max_tokens)` — the granted budget plus a 32-position
speculative-draft margin, never more than one quantum — from both chat entry
points; the full `REPLY_RESERVE_POSITIONS` from a segment's prompt pass, which
never sees the budget. With the quantum as the reserve the figure is exactly the
old one (`round_up(p + 512) == round_up(p) + 512`), so only a reply budgeted
under a quantum changes. Before, every short chat held two quanta: on the 8B,
f32 cache plus its f16 mirror, 384 MB a chat, and a fourth 45-token chat was
refused against 1,152 of 1,336 MB. After, four chats at once served on the 8B
and on phi-3.5 (768 MB a chat before, full multi-head attention). The item
above this one — "a twenty-token prompt reserves two quanta … ~4 MB per
request" on a 3B — was wrong by ~30x (a quantum there is ~117 MB f32); the
reserve is now one quantum for such a chat.

**Advice follows the cause.** A refusal where the prompt would fit an empty cache
but other live conversations hold the room (`other_conversations_hold_the_room`,
and the executor's matching check) says it fits once they finish; only a prompt
too long for the card is told a shorter one will work. Still open: WAITING
behind live requests instead of refusing. Batched admission runs inside the
worker's one message loop, so a wait there would stall every other chat's
decode; it needs a deferred-admission queue, and in a swarm it trades against
the re-plan to a peer that the refusal already buys.

**From the rules file (moved 2026-10-02):**

**`kv_budget::device_free_and_total_bytes` synchronizes the device's stream
before `mem_get_info`, and adds what this process's pool keeps unused.** cudarc
frees through the card's memory pool (`cuMemFreeAsync`); a buffer freed on the
stream counts as unused only after a synchronize, so an unsynchronized reading
counts what the previous request just released as still in use — the next long
prompt was refused against half its budget, every time, on the released
v0.3.207 (#121). Since #146 the pool KEEPS freed memory (below), which
`mem_get_info` reports as used; `cuda_pool::reusable_bytes` is the other half of
"free for this process". Every budget decision reaches the card through
`SplitModel::kv_budget_now` → this one function; a new reading of device memory
goes through it too. Guard: `the_cards_free_memory_is_read_after_a_synchronize`;
A/B: `SWARMLLM_KV_DEVICE_SYNC=0`.

**A reply is reserved by what it can reach** — `reply_reserve_positions(max_tokens)`,
a REQUIRED argument of `ensure_room_for_prompt`; a segment's prompt pass, which
never sees the budget, passes the full `REPLY_RESERVE_POSITIONS`. And a refusal
caused by OTHER live conversations says to wait, never to shorten the prompt
(`other_conversations_hold_the_room`, both refusal sites) (#122).

## `inference::split::kv_budget`

(2026-08-08) — the KV memory budget and the
admission check against it. The loader records the model's `kv_budget_bytes`
(from `kv_headroom_bytes`); before a forward ALLOCATES more positions,
`forward_inner_body` asks `KvCacheStore::claim_room` (→
`kv_budget::claim_exceeds_headroom`) against `kv_budget_now` — the load-time
figure capped by what the card has left — and claim_room evicts cached prompts
before it refuses. A refusal is `LocalMemoryUnavailable` (503; the router
re-plans, in a swarm onto a peer). Updated 2026-10-02: until then this section
named `quantum_exceeds_headroom` and `positions_claimed`, both since renamed. **Do NOT re-introduce a load-time
context clamp** — one existed, it shrank every user's context so a single
full-length conversation would fit, and it did not bound concurrency at all.
Three invariants a new caller must preserve: the check runs ONLY when
`kv_budget::positions_to_allocate` is non-zero — zero for almost every decode
step, read off the buffer the request already holds, reservation included
(otherwise it walks the whole store per generated token for an answer that is
almost always "no"); it charges the
POSITIONS claimed, not one quantum, because a prefill jumps many quanta in a
single forward and charging one under-counted the largest claim a request
ever makes by 10x; and `kv_budget_bytes: None` means UNKNOWN, never zero — every CPU node
and any GPU node whose free VRAM could not be read records `None`, and reading
that as a zero budget refuses everything.

## A prompt of known length is reserved, not grown into

(2026-09-12, FUTURE_WORK #32.) `KvCacheStore::set_reserved_positions` is
written by the worker at the top of `ensure_room_for_prompt` — the first
point the prompt's length is known, before any budget question, since a node
with no budget still pays for growth — with `kv_cache_reservation(prompt) +
REPLY_RESERVE_POSITIONS`, the same figure admission charges. The executor
reads it (`reserved_positions`, clamped to the context window) and hands it to
`new_kv_cache` as the size of each layer's FIRST allocation; prefix-cache
hydration builds its caches to the larger of the snapshot and the reservation,
so the suffix appends into the same buffer. **`SeqCache::with_capacity(dim,
initial, grow_by)`** separates the first allocation from the growth quantum;
`new` remains the case where they agree, and the f16 mirror is born the same
size as the f32 cache it shadows.

**Why.** The cache grew by `Tensor::cat` in 512-position quanta, and a
concatenation copies everything so far: a 20837-token prompt on a 3B was 41
grows per layer, 2279 `cat` calls, **97 GB copied**, each step holding the old
buffer, the new block and the result live at once — ~170 MB of transient per
layer at that length — and every one of those was a fresh device allocation on
a card admission had just judged full. None of it was charged: admission
charges the FINAL size. That is the shape of the field OOM in FUTURE_WORK #32
(v0.3.168, `xlam-2-3b`, prompt privacy on, 20837 tokens: `CUDA_ERROR_OUT_OF_MEMORY`
during the forward, and a 16126-token retry that ran five minutes without a
first token). vLLM allocates a request's KV blocks for the whole prompt before
prefill runs (Kwon et al., *Efficient Memory Management for LLM Serving with
PagedAttention*, 2023, §4.1) — a 32K request holds 32K of capacity from the
start, never grown into — and the general rule in every production cache is
the same: allocate once, write in place, never `cat`.

**What a change here must keep.**

- **The guard charges what will be ALLOCATED, read off the buffer.**
  `kv_budget::positions_to_allocate(allocated, reserved, total_seq, quantum)`
  replaced `positions_claimed(index_pos, total_seq, quantum)`. The old form
  derived the buffer from `index_pos` rounded up, which is true only while a
  cache grows from one quantum; a reserved cache is larger than that from its
  first append, so the old arithmetic charged a reply for every quantum
  boundary it crossed inside memory it already held, and could refuse it for
  room it did not need. `allocated` comes from `KvCacheStore::allocated_positions`
  — the buffer, not the tokens; 0 before the first append. The first
  allocation's claim is `max(reserved, round_up(total_seq))`, which is what
  `new_kv_cache` then allocates, so the claim and the allocation cannot drift.
  Rounding is by `layers::kv_growth_quantum(max_seq_len)` — the quantum the
  cache is actually built with (`KV_CACHE_GROWTH_TOKENS`, or the whole window
  when that is smaller) — not the bare constant, which over-charged any model
  whose window is under a quantum by up to 4x and was one more way for the
  claim and the allocation to disagree.
- **The test that guards the ratchet drives a real growth.**
  `a_refused_request_gives_back_the_cache_it_had_taken` used to re-run
  `index_pos = 0` on the same request, which never grows anything; it passed
  only because the old guard derived its claim from the position. It now
  sends a first chunk that fits and a second that outgrows the buffer, on a
  model whose window is wider than a quantum (`make_test_split_model_with_window`)
  — the default 128-position test model can never grow, which is why no test
  had exercised a growth boundary against the guard until now.
- **Absent means "grow as before".** A forward whose worker never recorded a
  reservation — speculative verify rounds, tensor-parallel phases, the bench
  harnesses, a request on a build that predates this — reads 0, `new_kv_cache`
  clamps that to one quantum, and every number the budget sees is the number
  it saw before. Nothing may treat 0 as "hold nothing".
- **`SWARMLLM_KV_RESERVE=0`** records no reservation, so the two behaviours
  compare inside ONE binary: the executor's `DIAG: KV cache growth during a prompt chunk` line reports `growth_steps` (from the process-wide
  `KV_GROWTH_STEPS` counter, one per `cat`) — with the switch on it must read
  0 for the prompt pass, with it off the old count. That line, not a faster
  wall clock, is the mechanism check. Measured 2026-09-12 on an isolated
  node, qwen2.5-0.5b (24 layers) on the processor, a 1901-token prompt:
  `reserved_positions=2560 growth_steps=0` with the reservation;
  `reserved_positions=0 growth_steps=144` without — three concatenations
  (512→1024→1536→2048) × K and V × 24 layers, exactly the arithmetic.
- **The reservation is per REQUEST and dies with it** — recorded beside
  `admitted_claims`, released by `clear_request`, `cleanup_request_id` and the
  TTL sweep. A retry replaces it, never stacks.
- **A small chat now holds its reply reserve.** Admission has charged
  `REPLY_RESERVE_POSITIONS` since gotcha #440; the memory is now actually held
  from the first token rather than claimed at the first quantum boundary, so a
  twenty-token prompt reserves two quanta where it held one. On a 3B that is
  ~117 MB per request (one 512-position f32 quantum; this line first said
  ~4 MB, wrong by ~30x), and on an 8 GB card it capped an 8B at three chats —
  so since 2026-09-26 a reply budgeted under a quantum reserves only what it
  can reach (§ "A reply is reserved by what it can reach", #122).
- **Not reproduced here.** The card is 8 GB with ~3.6 GB in use, so admission
  refuses the prompt lengths that reach the failure; the mechanism is
  established by reading, arithmetic and the growth counter, not by observing
  the OOM stop. The reporter's machine is the place that can confirm it.

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

## What a task took, a task gives back by being dropped

*Rule: `.claude/rules/arch-worker-memory.md` § "What a task took, a task gives back by
being dropped". Found by audit 2026-09-14, each verified against the code.*

### The shape

A resource acquired, then an `.await`, then a plain statement giving it back.
That statement runs only if the await RETURNS. Two ordinary things stop it:

- **`cancel::unless_cancelled` cancels by dropping the future.** It is the
  standard way a client disconnect or Stop reaches running work.
- **`abort_handle().abort()`** from the `CancelInference` handler. Tokio drops
  the task's future wherever it is suspended. **No in-band checkpoint can help**
  — `bail_if_cancelled` bracketing defends against cooperative cancellation and
  is powerless here.

### The three instances

**`PendingSpawnCharge`** — `get_or_spawn` charges `vram_reserved_mb` /
`ram_reserved_mb`, then awaits `spawn_worker` (a subprocess spawn plus an IPC
connect; `WORKER_CONNECT_TIMEOUT_SECS` is 30, so the window is seconds). Release
lived in the `Err` arm and in the `WorkerHandle` the `Ok` arm creates. Dropped
mid-await, neither exists — and the `Err` arm's own comment already said what
that costs: *"Leaving the charge would shrink the budget permanently."* The
budget is keyed by model and accumulates, so repeated cancellations during cold
starts shrink the node's usable memory monotonically until a restart, while the
logs report headroom that is not real.

Two live triggers. `pipeline::local_generate::try_local_generate_fastpath`
**wraps** `pool.generate(..)` in `unless_cancelled` — the exact pattern
`forward_direct` rules out in a comment two thousand lines away, so this is
gotcha #459 reintroduced by a file added later. And `CancelInference` aborts the
task serving an inbound segment, which no bracketing can survive.

The guard is armed only ACROSS the await and disarmed the moment it returns
(`handed_back_to_the_caller`), so the `Ok`/`Err` arms keep owning the charge
exactly as before and cannot double-release.

**`InboundForwardSlot`** — the per-peer concurrency count and the
abort-registry entry for one inbound `LayerForward`. Both were statements after
`handle_layer_forward(..).await`, and that function processes untrusted network
input, so a panic skips them too. The sweep is
`peer_forward_counts.retain(|_, v| v.load(..) > 0)`: it removes entries that
have reached ZERO and **cannot repair one stuck above it**.
`max_forwards_per_peer` is `(max_concurrent_forwards / 2).max(4)`, so **four**
leaked forwards from one peer refuse every later forward from it with "per-peer
limit reached" while nothing is in flight. A `NodeId` is cryptographically
stable, so reconnecting does not clear it; only a restart does.

### What a change here must keep

- Release by `Drop`, never by a statement after an await. If a resource must
  survive the await's return, disarm the guard at that point rather than moving
  the release back out.
- A guard that is armed across an await must not double-release on the normal
  path — disarm, and test both directions.
- Do not answer "is this cancellation-safe?" by finding a `bail_if_cancelled`.
  That defends against cooperative cancellation only; an external `abort()`
  ignores it entirely.

**From the rules file (moved 2026-10-02):**

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

## A storage budget must count what is on disk, and may only count what can be freed

**Rule:** `.claude/rules/arch-worker-memory.md` § the `storage_budget` /
`held_disk_bytes` bullet.

### What happened (measured 2026-09-14, fixed 2026-09-17)

`held_shard_bytes` prices the MANIFEST, so it can only see `shard_NNN.bin`. The
live node held **33062 MB on disk against 31423 MB counted** — a 1637 MB gap,
~5%, invisible to the setting the user typed. The breakdown matters, because it
is not evenly spread:

| file | bytes here | note |
|---|---|---|
| `tied_output_weight.bin` | **1566 MB** (4 files) | 95% of the gap; 279 MB for one 1.3 GB model |
| `gguf_header.bin` | 71 MB (13 files) | one per model, ~6 MB |
| `mmproj.gguf` | 0 here | 595 MB for LLaVA-7B when present |
| quarantined / dead `.tmp` | varies | nothing sweeps them |

**The sharp edge is a cancelled download.** The parts are cleaned up correctly,
but the header and tied weight stay — ~287 MB attached to a model that then
counts as zero shards and therefore zero bytes. Repeated cancels accumulate disk
the budget cannot see at all.

### Why the accounting fix alone would have been a REGRESSION

Prune deletes `shard_NNN.bin` and nothing else. `compute_resource_pressure`
divides `held_bytes` by the budget, so counting bytes prune cannot free gives a
node near its limit a floor it can never get under: it sheds two shards per
model per cycle, forever, and the redundancy check is the only brake. That is
exactly the shape of the still-open sole-replica prune report
(`docs/FUTURE_WORK.md` item 49), which describes a node in a continuous
urgent-pressure prune loop that ended up missing shards it was the sole replica
of.

So the fix is a PAIR, and the halves must not be separated:

1. `held_disk_bytes` measures the models directory.
2. `prune::reclaim_orphaned_model_files` → `shard::cleanup_orphaned_model_files`
   gives back the derived files of any model this node holds no part of, once
   per prune cycle.

Convergence is preserved because shedding a model's last shard makes its
leftovers reclaimable on the next cycle.

### What a change must keep

- **Three guards on the deletion, all required**, each verified by removing it
  and watching its test go red: the registry records no held part (which
  includes the `MMPROJ_SHARD_INDEX` sentinel, so a node holding only a vision
  encoder is skipped); no `shard_NNN.bin` remains on disk (a file present but
  unregistered means mid-adoption, not gone); no shard of the model holds a
  download claim. `manifest.json` and `hf_source.json` are kept — they are what
  the model IS.
- **Losing the derived files is cheap where it is not free.**
  `tied_output_weight.bin` (and every other sidecar in
  `GgufTensorMeta::sidecar_tensors`) is extracted from `shard_000.bin` by
  `daemon::manifest::extract_sidecar_tensors` and re-extracted automatically
  at startup.
- **The byte figure and the shard COUNT are different questions.**
  `storage_budget_now` takes bytes from the directory and the count from the
  registry; pricing the count off the directory would count a header as a shard.
- **Not cached.** The consumers are a timer pass and user-triggered handlers; a
  cached storage figure that lags a prune is its own defect.

### The trap found underneath it

`auto_manage::test_support::make_test_manager` never set
`config.node.data_dir`, so `Config::default()` pointed every auto-manage unit
test at the developer's real `~/.local/share/swarmllm`. Harmless only while
nothing under test touched the filesystem — the moment the budget measured a
directory, a test asserting a node "holding 14 GB" measured this machine's
actual 33 GB. **When a pure-logic function starts touching the filesystem, audit
the fixtures before the code** (gotcha #629). `write_sparse_shards` gives a test
real multi-gigabyte holdings via `set_len` at no disk cost.

## `ModelProcessPool::notify_every_worker` — a fan-out to every worker is bounded

**The rule.** One fire-and-forget message to every live worker, with both waits
bounded, running concurrently. `cancel_request` and `release_request_kv` are its
only callers.

**What it replaced.** Both open-coded the same loop: `worker.writer.lock().await`
then `send_daemon(..).await`, neither bounded.

**Why it matters.** `cancel_request` is awaited INLINE from the dispatch loop's
`CancelInference` arm (`daemon/dispatch/mod.rs`). That loop is the single
consumer of `network_out`, which carries gossip AND every inbound `LayerForward`
/ `LayerResult` / `StreamingToken` / `RemoteGenerateRequest`. An unbounded wait
there makes the node a black hole for the swarm while `/health/ready` still
answers `true` — which is what `docs/FUTURE_WORK.md` #90 records for 45 minutes
on 2026-09-18. **It is NOT established that this call site caused that
incident** — the worker had been idle-unloaded two hours earlier, so the fan-out
would have short-circuited on its `is_empty` guard. It is fixed because it is
the shape gotcha #74 already forbids, found while eliminating candidates.

It was also the ONLY inline un-spawned `.await` in the 2932-line dispatch file
besides two `RwLock` reads, and every other site that takes `writer`
(`process_pool.rs:1857,4808,4991,5176`) scopes the guard and bounds the wait
*specifically so a stalled worker cannot block the manager*. This fan-out was
the exception.

**What a change must keep.**

- **Two separate bounds, two different responses.** A lock timeout proves
  another task is mid-message and nothing has been written — stand down quietly.
  A send timeout proves the socket will not take a few dozen bytes, and dropping
  that future can leave a **partial frame** that desynchronises every later
  message — so mark the worker `dead` (reaped by `reap_dead_workers`, which also
  returns its memory charge). Collapsing the two into one bound means either
  condemning a healthy busy worker or leaving a corrupted stream in service.
- **Concurrent fan-out.** Sequential, N stuck workers cost N timeouts and the
  dispatch loop pays every one.
- **A worker that is reading never reaches either bound** — the message is a few
  dozen bytes into a socket buffer.

**Tests.** `a_cancel_cannot_be_held_up_for_ever_by_a_busy_worker` and
`the_fan_out_is_bounded_once_not_once_per_worker`. Both verified by removing the
bound: they do not merely fail, they hang until the harness timeout.

**From the rules file (moved 2026-10-02):**

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

## A worker that dies is not a worker that is slow, and the graphics stack can go mid-run

**The rule**: `.claude/rules/arch-worker-memory.md` § "A subprocess you are
waiting for can die instead, and the graphics stack can go mid-run".

### What it replaced

`spawn_worker` waited on `listener.accept()` with a 30 s
`WORKER_CONNECT_TIMEOUT_SECS` and nothing else. A worker that died before
connecting was indistinguishable from one that was merely slow, so:

- every startup failure cost the **full timeout**, per model and per arriving
  request, on a node that could not serve at all;
- every startup failure arrived as the same `worker connect timeout`, which
  names the symptom and nothing an owner can act on;
- the actual cause was **discarded by being unreadable rather than by being
  absent**. The worker inherits the daemon's stderr, so on 2026-09-18 the
  reason was in `node.log` 14 lines above the timeout:

  ```
  Inconsistency detected by ld.so: dl-setup_hash.c: 36: _dl_setup_hash: Assertion `(bitmask_nwords & (bitmask_nwords - 1)) == 0' failed!
  ```

  Raw loader output: no timestamp, no level, no module, no request id — so it
  appears in no level filter and no structured view, and matches none of
  `libcuda`, `CUDA`, `shared libraries` or `ERROR`. The whole of gotcha #646 was
  diagnosed believing nothing had been logged.

### Why the probe re-runs the binary instead of reading the error

Classifying that loader text would be guesswork: it names no library and
carries no error code, and the next libc release may word it differently.
Re-running the binary is not guesswork. `executable_still_starts` runs
`--version` — the cheapest thing this binary does, and on a healthy node an
answer in milliseconds that never touches the GPU. An executable that has
stopped starting, on a build linked against `libcuda.so.1`, **is** the graphics
stack failing. The probe runs only on a path that has already failed.

### Why it is a latch, and why it must clear

The NVIDIA driver is a kernel module plus userspace libraries that must agree.
An update replaces the libraries but cannot replace a module that is loaded and
in use, so processes that already mapped the old libraries keep working while
every new process fails. (Under WSL the same shape arrives from the Windows
host driver behind `/usr/lib/wsl/lib`.) That asymmetry is why the daemon cannot
detect this by asking about itself — it is the process that still works — and
why the state is latched rather than re-probed: `exec` resolves the libraries,
so the node cannot recover without a restart.

It clears on the one observation that disproves it, a worker that starts.
**Marking a device unhealthy is the easy half; coming back is the half that
gets forgotten** — NVIDIA's own Kubernetes device plugin is repeatedly
bug-reported for exactly that (k8s-device-plugin #1014, gpu-operator #1065).

### What a change must keep

- **The happy path must not pay for it.** `accept` wins the race and the
  `wait` future is dropped; tokio fuses the child, and `wait`'s only side
  effect is closing a stdin this worker never opened.
- **The advertised capability must follow.** `health/monitor.rs` withdraws
  `gpu_info` from the broadcast capability while the latch is set — a node that
  keeps claiming a GPU keeps being sent segments it will fail, which harms
  peers, not just itself. It is rebuilt every broadcast, so withdrawal and
  return both take effect on the next cycle.
- **The copy must not promise the processor takes over.** It is false here:
  this binary links `libcuda.so.1`, so a CPU-only worker exits 127 exactly as a
  GPU one does (verified 2026-09-18 by running `model-worker --help` under a
  deliberately corrupt `libcuda.so.1`). Guarded by
  `the_unavailable_message_does_not_promise_the_processor_takes_over`.

**From the rules file (moved 2026-10-02):**

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
→ `docs/FUTURE_WORK.md` #90 (#89 is closed).

## On Windows a model worker ends with its daemon (2026-10-04, FUTURE_WORK #153)

`process_pool::end_with_this_daemon` puts every worker `spawn_worker` starts in one Windows job
object with `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`; the job handle is created NOT inheritable and
never closed, so when the daemon exits — cleanly, killed or crashed — Windows terminates the
workers. **A new place that starts a worker process calls it too.**

What it replaced: std's `Command` passes a Windows child every inheritable handle, and the
node's QUIC socket is one. A worker that outlived its daemon held UDP on the node's port, and
the next start failed. Reproduced on Windows before the fix (`C:\temp\swarm153\worker153load.ps1`,
GNU cross-build): daemon killed as its worker began loading Llama-3.2-3B → at +300 ms the
worker alive and UDP 8950 still listed under the DEAD daemon's pid → the restart died on
"Port 8950 is already in use". After: worker gone at +300 ms, the restart ready in 2 s.

What a change must keep: (1) the job handle stays non-inheritable — a worker holding it would
keep the job, and itself, alive; (2) a failed `AssignProcessToJobObject` (a host job that
forbids it) warns and the worker runs as before — never fail the spawn on it; (3) the daemon
itself is never put in the job, or the update hand-off's replacement would die with it.
Killing a worker mid-GENERATION never reproduced the bug (it reads end-of-file and exits in
~50 ms); the window is a phase that writes nothing to the pipe, such as a model load — test
there.

## A reply between two forwards is in use, and a lost conversation is refused

**The rule**: `.claude/rules/arch-worker-memory.md` § "A reply between two
forwards is not idle".

### What it replaced

Five places in `ModelProcessPool` decided "is this worker in use?" as
`!worker.responses.is_empty()`: promotion to the card
(`worker_should_return_to_gpu`), the graphics-memory reclaim and its dry run,
the RAM reclaim, and `models_with_inflight_requests`, which auto-manage reads.
A split reply reaches its worker as **one forward per token**, so for most of
its life nothing is in flight — and `last_used` is stamped at the START of a
forward, so a slow processor forward already reads as idle past the 5 s floor
the moment it returns.

Promotion made it fatal. Its doc listed "the reason must actually be gone" as a
guard, but the body only asked whether `gpu_estimate_mb` fit — and the
estimate was for the slice the worker had been spawned with. Two layers of
GLM-4-9B (631 MB) always fit, so a model still pinned after a graphics
out-of-memory was retired; the respawn read the pin first and went straight
back to the processor; the next forward retired it again.

Measured 2026-09-23 at the v0.3.201 gate, on the released v0.3.200: a node
short of graphics memory planned the default start-and-finish split — its own
CPU for layers 0-2 and 14-40, a peer for 2-14. **14 retirements and 28 model
loads in one 96-token reply**, the log saying "Graphics memory has freed up —
retiring this model's processor worker" before each. Every reload began with an
empty cache and the worker computed the next token anyway: `使用命令命令命令…`.
Qwen2.5-Coder-7B and Mistral-7B produced the same kind of nonsense.

### How the cause was established

Five hypotheses died on a two-node rig with room for every worker: the n-gram
path (off → same garbage), activation compression, CPU↔GPU mixing, the
three-segment shape, and model size — a Mistral-7B boomerang answered
correctly. What the failing runs had and the rig did not was **memory
pressure on the coordinator**. Then removal: `SWARMLLM_VRAM_SWAP_MIN_IDLE_SECS=100000`
on the same node, same split, same peer → **0 retirements, 3 loads, "The
capital of France is Paris."**; the default restored → the garbage back.

### The fix, and why it has three parts

- **`WorkerHandle::in_use`** — a response in flight OR a conversation held
  between forwards (`kv_holders`, stamped by every forward, removed by
  `release_request_kv` / `cancel_request`). All five sites ask it; guarded by
  `whether_a_worker_is_in_use_is_decided_in_one_place`.
- **`PromotionInputs::reason_still_holds`**, fed by `cpu_reason` — the
  predicate the respawn itself reads first, so promotion and respawn cannot
  disagree about where the model will land.
- **`model_worker::forward_lacks_its_conversation`** — a forward past the
  prompt pass whose request holds nothing on this worker is refused as
  `ServiceUnavailable`. The first two stop THIS cause; this one turns every
  other way of losing a worker mid-reply (a crash, an explicit unload, a peer
  that restarted) into a visible failure instead of a wrong answer. vLLM's rule
  for a preempted sequence is the same: swap its cache back or recompute it,
  never decode without it.

### Why a conversation stops counting after two minutes

`CONVERSATION_GAP_SECS` (120 s since the last forward, or the cache TTL if
shorter). A node serving a slice for a REMOTE coordinator is never told that
reply finished — only the coordinator calls `release_request_kv` — so without
a bound, a model that answered a peer once would stay "in use" for the whole
ten-minute cache TTL, and a small machine would refuse its owner's own request
for memory. An active reply touches its worker every few seconds (3-10 s per
token on the slowest chains measured here), and its coordinator abandons a
decode step long before two minutes. A reply that is silent longer and then
resumes on a retired worker meets the refusal above — visibly.

### What a change must keep

- **Never read `responses` alone to mean "in use".** A sixth site would have
  the same hole; the guard fails the build.
- **Stamp on every forward path.** `forward_direct` and `forward_batch` both
  call `note_conversation`; `generate` does not need to (its worker clears the
  cache when the call returns).
- **The refusal must stay `ServiceUnavailable`** across the IPC and network
  hops (`the_refusal_keeps_its_type_across_the_process_boundary`) — that is what
  lets the router re-plan a request that has not streamed yet.
- **A fused decode batch declines when any member lacks its conversation**, so
  the per-request path refuses that one and runs the rest.

### Residual, deliberately left

Promotion prices the slice a worker was SPAWNED with, while admission prices
the slice being requested now, so between requests a promotion can still cost
one pointless reload (never a wrong answer). And a serving node could be told
when a remote reply ends — a completion notice would release its cache and its
protection at once instead of after the gap. `docs/FUTURE_WORK.md` #93.

**From the rules file (moved 2026-10-02):**

**`WorkerHandle::in_use` is the single answer to "is this worker in use?"** —
a response in flight OR a conversation it holds between forwards
(`kv_holders`, stamped by every forward, released by `release_request_kv` /
`cancel_request`, aged out `CONVERSATION_GAP_SECS` after the last forward).
Promotion, both reclaims and `models_with_inflight_requests` ask it. Reading
`responses` alone saw a split reply as idle between tokens, and promotion
retired its worker 14 times in one reply — each respawn decoding from an empty
cache (#93, gotcha #690). Guard:
`whether_a_worker_is_in_use_is_decided_in_one_place`.

**Promotion waits for the reason to be GONE** — `reason_still_holds` reads
`cpu_reason`, the predicate the respawn reads first. **And a forward past the
prompt pass whose conversation is gone is refused**
(`model_worker::forward_lacks_its_conversation`, `ServiceUnavailable`), never
decoded from nothing — whatever lost the worker.

## The contribution level caps the swarm's work, not the owner's prompt (2026-09-26)

The Settings page defines the contribution level as "how much of your computer
SwarmLLM may use to answer requests from the swarm". Since #237 (2026-08-05) it
sized the worker's whole rayon pool — `RAYON_NUM_THREADS = round(physical ×
{0.5, 0.75, 1.0})` — so the owner's OWN prompts were read on half the physical
cores at the default level: 4 threads on an 8-core machine where llama-server
uses 8. That was most of a field report's "3x slower than llama.cpp"
(FUTURE_WORK #119, measured: at matched threads the gap is 1.17x).

**What changed.** `process_pool::Requester` (`Owner` / `Swarm`) is a required
argument of `ModelProcessPool::generate`; the four callers are the API fast
path (twice, streaming and not), the router's local plan (the router only ever
receives this node's API requests — `api/mod.rs` and `api/openai/streaming.rs`
are its only submitters) and the peer-serving `dispatch/remote_generate.rs`.
`IpcGenerate::for_the_owner` carries it; the worker marks the request ONCE at
its `DaemonMsg::Generate` entry, before either admission path, on the KV store
(`mark_owner_request`) — released by `forget_request_bookkeeping` with the
reservation and the admission claim, and swept by TTL like them. The two pool
choke points (`forward_inner_impl`, `forward_batch`) ask `serves_the_owner`;
one owner item makes a batch the owner's. The width comes from the daemon as
`SWARMLLM_OWNER_PREFILL_THREADS` (an operator's own export wins, as for
`RAYON_NUM_THREADS`).

**What did not change, and why.** Decode keeps its calibrated width: it is
bandwidth-bound and #237's sweep put its optimum at or below the cap (16 threads
37% slower than 6). The swarm's prompts keep the cap — the case #237 was about
(529-534% of 600% on a 6-core Minimal node). An explicit `max_cpu_threads` binds
the owner too: a number the operator chose is not overridden. Physical, not
logical cores: the plateau argument, and llama.cpp's default.

**Measured** (Ryzen 7 5800H, llama-3.2-3b Q4_K_M, node at Minimal, one binary,
control `SWARMLLM_OWNER_PREFILL_THREADS=0`): 2,427 tokens 35.5 → 49.5 tok/s
(+39%); 6,911 tokens 28.8 → 32.4 (+12%). The long prompt barely scales where
llama.cpp gains 49% for the same 4→8 — attention there is the next lever.
Tests: `only_the_owners_prompt_is_read_on_the_owners_width` (asserts the pool
the work RAN on; fails with the branch disabled),
`an_owner_request_is_marked_per_request_and_released_with_it`,
`the_owners_prompt_is_read_on_every_core_unless_the_operator_said_otherwise`.

**The forward path too (pre-release review, 2026-09-26).** The change above
marked only `Generate`, so when the router SPLIT the owner's request, this
node's own segment — `PipelineExecutor::process_local_segment`, a `Forward` —
read its prompt at the contribution width: the "one invariant, N paths" shape,
found by a reviewer before it shipped. `ModelProcessPool::forward_for_request`
now takes the `Requester` as a REQUIRED argument, as `generate` does;
`IpcForward::for_the_owner` carries it and `handle_forward` marks the request —
**after** the prompt pass's `clear_request`, which releases the owner mark with
the rest of the request's bookkeeping. The first version of this fix marked at
the top of the handler, so the clear wiped it on every prompt pass; the unit
tests and the review passed it, and a split request run on the rig showed the
owner's pool was never built. Verified on `split_rig.sh split` (llama-3.2-3b,
both nodes on the processor, A holding parts 0-1): the coordinator's worker
logs `This computer's own prompts are read on every core … owner_prefill_threads=8
swarm_threads=4` for its own segment, and the node serving the peer's segment
logs nothing — before the move, neither did.
The plain `forward` (a peer's segment, a tensor-parallel phase) is the swarm's,
the batch-scheduled decode steps carry `false` (decode keeps the contribution
width whoever asked, and the prompt pass's mark persists), and the flag is set
only by the daemon — `LayerForward`, the network form, has no such field, so a
peer cannot claim the owner's width. Guard:
`the_owners_own_segment_is_forwarded_as_the_owners` (planted forms in
`the_owner_segment_guard_catches_a_forward_that_says_nothing`); round trip:
`a_forward_says_whether_it_is_the_owners`.

**From the rules file (moved 2026-10-02):**

**`process_pool::Requester`** is a REQUIRED argument of `ModelProcessPool::generate`
AND of `forward_for_request` — the owner's own segment of a split request is
the owner's too (`IpcForward::for_the_owner`) (`Owner` = this node's own API,
`Swarm` = a peer's). The worker records it once, at
its `Generate` entry (`KvCacheStore::mark_owner_request`), and `cpu_pools::in_phase_pool`
reads it where every forward picks its pool: the owner's PROMPT reading runs on
`ResourceConfig::owner_prefill_threads` (physical cores, or an explicit
`max_cpu_threads`); decode and the swarm's work keep the contribution width. A
default node read its owner's prompts on half its cores (+39% at 2.4K tokens).

## A worker that cannot grow gets the spawn's ladder (2026-09-27)

**`ModelProcessPool::grow_worker` is the one place a live worker is given a
range it has not held**, called from `get_or_spawn`'s fast path. A spawn has
always run a ladder — the whole model on the card, else reclaim the card from
idle models (`free_vram_for_admission`), else part of the model on the card and
the rest on the processor (`partial_gpu_layers`), else the processor. Growth
(`charge_additional_segment`) had only the first rung and refused on the rest.

### What it replaced, measured on the live node

RTX 3070 Laptop, graphics budget 6019-6162 MB (Windows and a browser hold
~1.3 GB of the card), v0.3.209, found by the GPU spread benchmark:

- **An idle model blocked one the node holds whole.** An idle llama-3.2-3b held
  3138 MB. A plain request for meta-llama-3.1-8b (6033 MB estimated) was
  planned around it — a boomerang through a peer in Belgium — which left an 8B
  worker holding one layer; every attempt to widen it was refused, and the
  request 503'd after 11.8 s (`fb92fd13`). The reclaim that would have admitted
  it existed, in the spawn path only; FUTURE_WORK had deferred it from growth
  on 2026-09-04 "until a real workload shows this refusing ranges a reclaim
  would have admitted". Root-caused CAUSED by a `root-cause` agent: the only
  difference between this and three earlier clean 8B loads on the same node was
  `committed_mb` 0 against 3138.
- **A partial worker wedged the model.** After a split left the 8B worker
  holding [0..1) and [26..32) (1570 MB), EVERY request for the whole model was
  refused in ~50 ms — the streaming chat path included, which goes straight to
  the local worker (`local_fast_path_for`) and never reaches the router. An 8B
  14 MB over the budget can only run PART on the card; a fresh spawn would have
  placed 31 of its 32 layers there, and growth could not.
- **The same on the peer.** A partial 14B worker left on the peer by an earlier
  failed plan refused the whole model three times in a row ("need about 10080 MB
  more … its worker is already holding 2440 MB"), and each refusal touched the
  worker's idle clock, so it never aged out while it was being asked.

### The fix, and what it must keep

1. **Reclaim first**, outside `spawn_lock` (the charge takes and releases it;
   a reclaim waits for workers to exit), with the spawn's planner and guards:
   the idle floor, never a busy worker, all-or-nothing.
2. **Then replace, only an idle worker.** If the range still does not fit and
   `WorkerHandle::in_use` is false, retire the worker and fall through to the
   slow path, which places the range afresh. `in_use` is the single answer
   because it sees a conversation between two forwards — retiring one of those
   is #93's garbage-reply defect.
3. **A worker in use keeps its refusal** (`LocalMemoryUnavailable`, which the
   router re-plans). A worker charged against system memory also keeps it: both
   extra rungs are about the card.
4. **Retire the handle this call was given, never whichever the map holds** —
   a concurrent spawn may already have replaced it (`Arc::ptr_eq`).

Tests: `a_worker_that_cannot_grow_takes_the_card_back_from_an_idle_model`,
`an_idle_worker_that_cannot_grow_on_the_card_is_replaced_not_refused`, and the
control `a_worker_in_use_that_cannot_grow_keeps_its_refusal` — the first two go
red with the reclaim, respectively the retirement, planted out. The admission
guard (`an_admission_refusal_is_the_variant_the_router_re_plans`) was
re-pointed: the growth refusal now carries its variant inside `GrowthRefused`,
and the guard's needle `return Err(SwarmError::` counted one refusal where there
are two. Its self-test now plants the wrapped spelling too.

### The planner's half (added the same day)

`max_local_hostable_layers` — this node's room for the search — counted an idle
model's charge as spent, so the planner routed around memory the spawn would
reclaim in one call: with an idle 3B on the card it offered 14 of an 8B's 32
layers and sent 22 to Belgium. It now adds `idle_vram_reclaimable_mb`: every
charge `vram_reclaim_eligible` admits — the ONE predicate the reclaim itself
filters with, so the planner can never be promised room admission would refuse
to make. A ceiling, not a plan: admission still frees least-recently-used first
and only what it needs. Test: `room_an_idle_model_holds_is_room_the_planner_may_use`
(controls: a model in use, and one used moments ago, give nothing back); red with
the credit removed. ⚠ `max_local_hostable_layers` also feeds what this node
ADVERTISES; peers now see the same room, and their segment triggers the same
reclaim here — consistent with "graphics memory has ONE owner".

### The fit verdict's half (2026-10-05, FUTURE_WORK #129)

The ceiling counted the reclaim; the FIT VERDICT did not. `fits_in_budget` —
behind `would_fit_on_gpu`, `gpu_estimate_and_fit`, and through
`is_cpu_bound_for_lack_of_vram` behind `serves_on_cpu` (the local fast path's
gate, the router's local price, the hand-off trigger) — compared
`committed + estimate` with the budget, an idle model's charge included. With
qwen2.5-0.5b and qwen3-1.7b idle on the card, a cold Coder-7B held here "did not
fit": the fast path stood aside, the router priced this node at processor speed
(`local_processor_cost_ms=27025`) and sent the request to a cold peer in Italy
(46.5 s for 120 tokens, `predicted_ms=4495`). Kept here, the loader freed both
(`freed_mb=3754`), admitted the 7B whole (`estimated_mb=5232 budget_mb=6354`) and
answered in 13.2 s, load included — the same release binary, the same hour.
`fits_in_budget` now answers "fits" when admission would MAKE the room:
`reclaimable_vram_mb` is the reclaim's own dry run (`vram_reclaim_candidates`,
`plan_vram_reclaim`: idle floor, never busy, all or nothing). Ollama places a
model the same way — it unloads idle runners before it settles for less
(`server/sched.go`, `findRunnerToUnload`). Test:
`a_model_fits_the_card_if_admission_would_free_an_idle_one_for_it` (controls: in
use, and used moments ago); red with the reclaim term removed. ⚠ The dashboard's
"fits on your GPU" now reads yes for a model that would displace an idle one —
which is what loading it does.

### Hardened in pre-release review (same day)

Two races a `code-reviewer` pass found in the fix, both closed and tested:

- **A retirement could kill a worker another request had just grown.** That
  request's smaller range fit where ours did not, and until it registers its use
  `in_use` reads false. The retirement is now decided under `spawn_lock` by
  `retirement_decision`, against the charge total the refusal saw
  (`GrowthRefused::worker_charged_mb`, read under the same lock): a changed total
  is someone's admitted range → `Keep`, refuse ours. Test
  `a_worker_grown_by_another_request_since_it_refused_is_kept` (red without the
  total check).
- **The spawn path could hand back a worker that did not hold the caller's
  range.** Its "another task spawned it while we waited" return predates this
  fix, but a retirement now sends traffic there: returned uncharged, the worker
  loads whatever a forward names and memory the pool never admitted is in use.
  `get_or_spawn` is now bounded rounds of `get_or_spawn_once`, and that return
  hands the worker back only if it holds the range — otherwise another round,
  whose fast path grows it. Three rounds that each find a replacement fail as a
  lifecycle error (`ServiceUnavailable`), which is contention, not progress.

**From the rules file (moved 2026-10-02):**

**`ModelProcessPool::grow_worker` is the one place a live worker takes on a new
range.** Where the card will not take it: reclaim idle models' graphics memory
(the spawn's `free_vram_for_admission`, outside `spawn_lock`), then — only if
`WorkerHandle::in_use` is false — retire the worker so `get_or_spawn`'s slow
path places the range afresh, part on the processor if need be. Growth used to
refuse on the first rung, and a partial worker then wedged its model — every
chat request refused in ~50 ms — until it aged out. A worker in use keeps the
refusal; never retire one mid-conversation (#93).

## A model loaded for another model's request is a guest (2026-09-28)

### What it replaced

The in-engine drafter (`pipeline::engine_drafter`, v0.3.212) is an ordinary
model this node holds, run by its own worker through `ModelProcessPool::draft`,
and it went through `get_or_spawn` like any request FOR that model. Admission
therefore did what it does for a tenant: refused as it stands, it reclaimed
graphics memory from every model not in use and idle past
`VRAM_MAKE_ROOM_MIN_IDLE_SECS` (5 s). Between two chat turns the TARGET's own
segment worker is exactly that — its conversation released, idle for longer than
the floor — so the drafter's load could unload the model it was about to guess
for. The target's prompt pass then reloaded its segment beside the drafter,
which now held the card, and admission placed it part-card or on the processor.
On the next turn promotion could evict the idle drafter and the drafter's load
evict the idle target again: the "two models alternating faster than they load"
case the idle floor exists for, with the floor too short to cover a chat turn.

The read-ahead (`engine_drafter::read_ahead`, spawned before the target's prompt
pass so the drafter's load overlaps it) made the cold start a race too: with
neither loaded, whichever took `spawn_lock` first took the free memory, and the
drafter usually got there first.

Found by reading the path before a default flip, not by a failure on the WAN
bench: that run's target segment was warm and the card had room for both.

### Research

vLLM hit the static form: the draft model's weights were missing from its
memory profiler, so it went out of memory once the KV cache had taken the
remainder (vllm-project/vllm#14067); its fix plans draft and target memory
together, up front, and llama.cpp's server likewise loads both at startup. A
swarm node loads models on demand and the planner prices only the target, so
there is no up-front plan to join; the drafter is made subordinate instead.

### The rule

`Tenancy::Guest` takes memory only as it stands free: `admit_to_gpu_as` and
`admit_to_cpu_as` skip the reclaim, the partial card/processor placement is not
offered (a guess is paced by its slowest layer), promotion is skipped, a worker
that would have to GROW is refused rather than grown (growth reclaims or
replaces), and neither the processor-fallback nor the RAM-refusal notice reaches
the dashboard — nobody asked for the drafter by name, and a refused drafter only
means a reply goes unguessed (`drafting_off`). The TARGET may still reclaim an
idle drafter: guest status is a property of the load, not of the model, so the
same model asked for by name is an ordinary tenant.

The ordering half is in `dsd.rs`: the read-ahead starts only when
`holds_segment` says every segment of the plan assigned to this node is already
loaded. Otherwise the first round reads the prompt, as it did before the
read-ahead existed. Two contiguous local segments that the loader merged into
one run read as "not loaded" and lose only the overlap.

### What a change must keep

- Every load names its tenancy; the parameter is required so a new caller
  cannot inherit the tenant's reclaim by omission.
- Tests: `a_guest_never_takes_the_card_from_an_idle_model`,
  `a_guest_never_takes_system_memory_from_an_idle_model` (each with the tenant
  as control), `a_guest_does_not_grow_a_worker_it_finds` — all three red with
  the guest checks removed.
- **A closed reply channel is judged in one place** (`reply_channel_closed`, 2026-09-28, gotcha #749):
  superseded by a later call under the same id → a plain error and the worker stays; otherwise evict.
  The draft wait had copied `forward_direct`'s loop, which never got #180's check, and evicted the
  healthy drafter a router retry had just started on. Test
  `a_superseded_draft_leaves_the_worker_for_the_call_that_superseded_it`, red without the check.
- Residual: a guest placed on the processor stays there until it is unloaded
  idle, even if the card frees up; promoting it would mean taking memory.

**From the rules file (moved 2026-10-02):**

**`process_pool::Tenancy` is a REQUIRED argument of `get_or_spawn`.** A `Guest`
— the split drafter, `ModelProcessPool::draft` — takes memory only as it stands
free: no reclaim, no card/processor split, no promotion, no growth, no dashboard
notice. As an ordinary load it evicted the very target it guessed for, which is
idle between turns after the 5 s floor, and the target's next prompt pass then
reloaded beside it, on the processor if the card was full. **`dsd.rs` reads
ahead only once this node's target segments are loaded** (`holds_segment`): on a
cold start the free memory is the memory those segments are about to take.


## A card that stalls is given less work (2026-09-28)

**What it replaced.** Nothing watched what the card did. Admission on the card was the slot
count (`batch_generate_max_slots`, default 8) and the KV budget; #122's fix (v0.3.208) let an
8 GB card with an 8B take six short chats instead of three, verified by counting refusals.
`split::kv_budget` already recorded that on WSL2 and Windows the driver answers an
over-commitment with host-backed memory rather than an error, so the model keeps answering
while decode crawls — "the accounting is the only guard there is".

**What it was measured at.** The live node, 2026-09-28 (RTX 3070 Laptop 8 GB that also drives
a 320 Hz desktop, hardware GPU scheduling on, WSL2, driver 616.92, Llama-3.1-8B Q4_K_M), from
`node.log` alone: the device calls between the prefix-cache lookup and slot registration took
**< 0.1 s every time from 2026-09-17 to 09-28 morning** (a handful at 0.1-2 s), then **2-60 s,
34 times in ~90 minutes** of 2-8 simultaneous chats. Every running chat froze for as long —
the worker is one loop, admission runs inline before the tick — and per-chat speed fell to
2-7 tok/s against 50 alone. The hour ended with a hard hang of the whole PC (gotcha #754). Why
the card stalled is NOT known (host paging, contention with the compositor — nvlddmkm FECS
exceptions recur on heavy-GPU days —, the laptop, or v0.3.212 itself); the guard does not
need it.

**The rule.** `inference::card_pace::CardPace`, one per worker: a device-bound step (the
admission section of `try_register_generate_slot`, after the model load and before the
network-bound remote prefix probe; each `step_decode_pool` tick) of ≥ `STALL` = 2 s on a model
that runs entirely on the card (`SplitModel::runs_entirely_on_card`) sets the ceiling to
`max(1, running / 2)` if that is lower; one more is allowed after each `HOLD` = 60 s without a
stall, back to capacity. The first `WARM_STEPS` = 16 card steps of a worker never count.
Past a LOWERED ceiling, `Generate` is refused with `LocalMemoryUnavailable` BEFORE the
batched/sequential choice; at capacity the pace refuses nothing — a full table's extra request
falls through as it always did, and a refusal saying the card stalled would be untrue. The constants are Windows' own: `TdrDelay` 2 s ("the number of
seconds that the GPU can delay the preempt request from the GPU scheduler") and
`TdrLimitTime` 60 s (learn.microsoft.com, "TDR registry keys"). The shape is congestion
control on concurrency — Netflix `concurrency-limits`, Envoy adaptive concurrency — reduced to
AIMD on one unambiguous signal.

**What a change must keep.**
- **Processor steps never count** — a CPU tick legitimately takes seconds (gotcha #191), and a
  card/processor split is excluded for the same reason.
- **The gate precedes the path choice.** A request the full table turns away falls through to
  `handle_generate` and runs on the card; a gate on the table alone caps nothing.
- **Never below one**, so the owner is never locked out of their own card.
- **The refusal must not read as fatal** (`worker_ipc::worker_error_is_fatal`): the pool would
  kill a healthy worker. `the_refusal_is_never_mistaken_for_a_broken_worker`.
- **Refuse, never queue inside the loop**: a wait there stalls every running chat (the #122
  "still open" note says the same about queueing).

**Not covered** (`docs/FUTURE_WORK.md` #146): segment forwards of split pipelines are neither
timed nor refused — refusing one mid-request kills that request, so only a prompt pass could be
refused; `handle_generate`'s per-token forwards are not timed; the daemon does not advertise a
tripped card to peers. **Measured on a card: not yet** — that is a stress test, and follows
`memory/feedback_research_before_stress_tests.md`.

**From the rules file (moved 2026-10-02):**

**Memory arithmetic is not a limit on what the CARD can do.** Every admission
gate on the card was arithmetic (slot count, KV budget), and on WSL2/Windows the
driver hands out host memory instead of failing — so a worker kept admitting
chats while its card stalled 2-60 s per step, for an hour, until the PC hung
(gotcha #754, #146). **`inference::card_pace::CardPace` is the one answer to "may
this worker start another generation on its card now"**: a device-bound step of
≥ 2 s (Windows' `TdrDelay`) on an all-card model halves the ceiling, one comes
back per quiet minute, never below one. The gate sits in the `Generate` arm
BEFORE the batched/sequential choice — a full table falls through to
`handle_generate`, which runs on the card anyway, so lowering the slot count caps
nothing. A new device-bound step on the worker loop is timed into it; a new
generation path is gated by it. Refusal = `LocalMemoryUnavailable` (the busy 503
the router re-plans). A/B: `SWARMLLM_CARD_PACE=0`.

## A card's memory pool keeps what the worker frees (2026-09-28)

**What it replaced.** cudarc allocates every candle buffer with `cuMemAllocAsync` from the
device's current memory pool and frees into it, and nothing in cudarc, candle or this repo set
the pool's release threshold — so it was the driver default, ZERO: "all unused memory in the
pool is released back to the OS during every synchronization operation" (NVIDIA, "Using the
CUDA Stream-Ordered Memory Allocator", part 1). The worker synchronizes at every admission
(#121's budget reading) and at every token's logits copy, so every admission's buffers, every
prefix snapshot and every token's ~650 temporaries went back to the driver and were fetched
fresh. On WSL2 a fresh fetch crosses the host's GPU channel, and on the dev machine that cost
grew ~1000× over two days of Windows uptime (the same first snapshot copy: 0.007 s at ~8 h,
8.8 s at ~49 h — `docs/FUTURE_WORK.md` #146 deep dive, gotcha #755) until a worker admitting
several chats stalled 2-60 s per admission and the PC hung (#754).

**Research.** PyTorch's `cudaMallocAsync` backend sets the threshold to `UINT64_MAX`; RAPIDS RMM
holds its pool for the process's life; ggml plans its buffers once. NVIDIA, part 2: "Exclusive
to a single process: Use the maximum release threshold"; "Shared among cooperating processes:
… set each process pool to an appropriate value to avoid any one process monopolizing all
device memory". And part 1: memory in a pool "can also be released implicitly by the CUDA
driver to enable an unrelated memory allocation request in the SAME process to succeed" — never
for another process. A node runs one worker process per model beside the desktop, so a pool
that keeps everything forever would hold room another worker, or a model the daemon wants to
load, cannot get; and the daemon reads the card through `nvidia-smi`, which charges a worker's
kept memory to "other programs" (`compute_vram_budget`).

**The rule.** `inference::cuda_pool`:
- `split::loader::load_device` — the one place a worker picks the card — raises the threshold
  to max once per process (`keep_freed_memory`). Guard:
  `every_split_model_reaches_the_card_through_load_device` (planted violation verified red).
- `kv_budget::device_free_and_total_bytes` adds the pool's reserved-but-unused bytes
  (`RESERVED_MEM_CURRENT − USED_MEM_CURRENT`) to `cuMemGetInfo`'s free figure, after the
  synchronize, so #121 stays fixed without the hand-back; the loader's load-time budget does the
  same. The reading is logged at debug: `DIAG: card memory reading`.
- `model_worker::run_worker` hands the unused part back (`cuMemPoolTrimTo(pool, 0)` after a
  synchronize) once the worker has had nothing to do for `IDLE_TRIM` = 60 s, logged as `DIAG:
  card memory pool handed its unused memory back after idle` — inside the daemon's own idle
  unload (`idle_unload_secs`, 300 s by default).
- A/B in one binary: `SWARMLLM_CUDA_POOL_KEEP=0` (driver default, no trim).

**Measured 2026-09-28 23:04-23:12 +07** on the CUDA release build of `4abb05be`, RTX 3070 Laptop
8 GB, Llama-3.1-8B Q4_K_M alone on the card, isolated node, live node stopped, Windows uptime
5.5 h, SINGLE requests only (`~/swarmllm-pool-0928/pool_ab.sh`, arms interleaved keep /
control / keep / control):
- **Mechanism fired**: with the switch unset the pool held 5,856 MB reserved between requests
  with ~850 MB of it unused and counted free (device 974 MB + pool 853 MB); the idle hand-back
  took 11 ms and returned it to 5,248 / 5,120 MB (in use 5,013). The control arm logged the
  driver-default line.
- **#121 stays fixed**: three ~1,835-token prompts in sequence served in both arms, zero
  refusals of any kind.
- **Speed at 5.5 h uptime**: prompt 2.36-2.47 s keep vs 2.42-2.48 s control (no difference —
  fresh allocations are still cheap this soon after a boot, which is the deep dive's
  prediction); decode best-of-3 50.1 / 50.4 tok/s keep vs 48.6 / 48.7 control, the same
  direction in both pairs (~3%, inside this box's spread, so no claim beyond "not slower").
  The win this exists for — admissions at high host uptime — needs a run after ~2 days of
  uptime, the same script.
- **Replies byte-identical** between the arms, greedy, after a long prompt had used and freed
  memory, on Llama-3.1-8B and Qwen2.5-Coder-7B (`replies_ab.sh`) — a reused buffer read before
  it is written would show there.
- ⚠ The prompt sent after 75 s idle took ~0.2-0.3 s longer in BOTH arms, so it is not the
  hand-back (the card clocks down when idle); without the control arm it would have read as
  the trim's cost.

**What a change must keep.**
- **Count kept memory as free wherever this process asks how much room it has** — otherwise the
  budget reads a finished request's cache as used and refuses the next long prompt (#121 by
  another road).
- **Hand it back when idle**; the driver will not lend it to another process, and the daemon
  cannot tell it from another program's memory.
- **The synchronize before the reading stays**: a buffer freed on the stream counts as unused
  only after it.

**From the rules file (moved 2026-10-02):**

**The pool's release threshold defaults to 0** — every synchronize handed all
freed card memory back to the driver, and the next admission (and, with the
per-token logits copy, the next token) fetched it fresh; on WSL2 that fresh
fetch slowed ~1000× over two days of host uptime (#146, #755). **`split::loader::load_device`
is the one place a worker picks the card**, and it raises the threshold to max
(`inference::cuda_pool::keep_freed_memory`); **`model_worker::run_worker` hands
the unused part back after `cuda_pool::IDLE_TRIM` (60 s) idle**, because the
driver lends a pool's spare memory only within its own process and the daemon's
`nvidia-smi` reading cannot tell a worker's kept memory from another program's.
Any reading of "free for this process" adds `cuda_pool::reusable_bytes`. Guard:
`every_split_model_reaches_the_card_through_load_device`; A/B:
`SWARMLLM_CUDA_POOL_KEEP=0`.

## The daemon holds no CUDA context (2026-10-05)

**What it cost.** The daemon never runs a model — its workers do — yet `nvidia-smi` listed it as a
compute process. Startup detection asked llama.cpp's device list
(`llama_cpp_2::list_llama_ggml_backend_devices`, linked into every release CUDA build through the
`cuda` feature), which reads free memory and so creates the CUDA runtime's primary context; the
runtime keeps that until the process exits. Measured 2026-10-05, RTX 3070 Laptop 8 GB, WSL2: the
card read 1021 MiB used with the idle daemon up and 884 MiB with it stopped — **137 MiB**, exactly
what one idle `cudaFree(0)` context costs on its own (1018 → 1155 MiB). That is card memory a
model's layers or its conversation could use, held for the node's whole life, on every CUDA node —
and on an 8 GB card the difference between a model fitting and going abroad (#129). It was also
one more context on a card whose context-switch engine has logged faults under load (FUTURE_WORK
#146, #220).

**The rule.** `gpu_support::describe_local_gpu` asks the driver API — `cuInit`, `cuDeviceGetName`,
`cuDeviceTotalMem` — none of which makes a context, guarded by `is_culib_present` because
cudarc's loader panics when the driver library is missing. llama.cpp's list answers only on a
build without candle's CUDA. `Device::cuda_if_available` is not used either: it, too, makes a
context. Free memory is read live where it is needed (`vram::query_gpu_vram_free_mb`, nvidia-smi)
and `GpuInfo` no longer carries a free figure at all — the startup one had already made every node
advertise zero room once (`health::monitor`), and a stale field nobody can read cannot do that
again. The diagnostics report reads it live; the dashboard's fallback no longer subtracts it.

**Verified on a `--features cuda` build** (2026-10-05, an isolated idle daemon, nothing else
starting): v0.3.225 card 1929 → 2066 MiB (+137) and its pid listed by `nvidia-smi
--query-compute-apps`; the new build 1929 → 1929 MiB, not listed — twice. A program that only calls
`cuInit`, `cuDeviceGetName` and `cuDeviceTotalMem` costs +0 MiB. Still `GPU detected gpu=NVIDIA
GeForce RTX 3070 Laptop GPU vram_mb=8191 backend=CUDA`.

**One deliberate change of answer.** The Windows GPU build carries candle's CUDA AND llama.cpp's
Vulkan (`windows-gpu`). On a Windows machine with an AMD or Intel card and no NVIDIA driver, the
old detection reported the Vulkan device as "GPU detected" — but the workers that serve shards are
candle and can only use CUDA, so the node planned and advertised graphics memory its models never
ran in. It now reports no card there (`is_culib_present` is false), which is what its workers do.
A build without candle's CUDA (macOS, Metal) still asks llama.cpp.

## A model its holders cannot run is fetched by a machine that can (2026-10-07, #231)

**What happened.** A tester's report on v0.3.229: the only computer holding a whole Qwen 3.5 9B
was a 6 GB processor-only peer capped at 5200 MB (the model needs 6688 MB at its admission),
while an RTX 3060, an RTX 4060 and the reporter's RTX 4050 held none of it. Every part had a
holder, so the replica target (`geo_target_replicas`) was met and nothing asked a machine that
could RUN the model to fetch it. #230 made the request refuse at once ("not enough memory in the
swarm") instead of being sent to be refused — honest, and still no answer.

**The rule** (`auto_manage::coverage`). Replicas count copies; carrying counts what the copies can
DO: the layers a model's live holders could ever carry between them, each at the smaller of the
layers it holds and its ceiling (`NodeCapability::model_memory_ceiling_mb` weighed with the
model's admission curve — the planner's arithmetic, `scheduler::layers_carried`). When that falls
short of the model, for a model somebody asked for (regional demand, or this node's requests) and
only when the connected swarm's ceilings could carry it at all:

- **One carrier at a time, fetching only the shortfall.** `parts_to_carry` is the ONE plan: the
  parts a machine lacks, in model order, until they close the shortfall, never past its room (its
  ceiling less what it holds). The carrier is the highest rendezvous weight
  (`blake3(model ‖ node)`) among machines whose plan is not empty and fits the disk they advertise,
  judged from the figures every node gossips — its own included (`local_capability`) — so all of
  them pick the same one, and the one picked is one that will fetch. Its parts skip the replica
  target and the hash ring, still through the trust gate, the budget and `would_shed_copy`. **Only
  the shortfall** — a machine with room for the whole model is not asked to hold it: "no machine
  holds the whole model" (the user, 2026-09-28, `docs/plans/wan_parallel.md`); the first cut
  fetched up to the carrier's ceiling and was changed before release.
- **Prune keeps what carries.** `would_shed_copy` asks `copy_carries_model` before the replica
  count: the chosen carrier's parts while the model needs one (the download pass asks BEFORE the
  part is held), and any copy whose holder adds to what carries the model while it would fall short
  without it. The download pass asks the same function before each fetch, so the two agree and a
  carried part is never shed and refetched (gotcha #795's loop).
- **A carrier keeps its lease only by progress.** A machine whose own storage budget or disk
  reserve will not take its plan never fetches, and nothing it gossips says so — it stayed elected
  for ever in the first cut (the review). After `CARRIER_PATIENCE` (20 min) with no part gained and
  none being fetched (`peer_shard_downloads`, this node's own claims), every node passes it over for
  `PASSED_OVER_FOR` (2 h) and the next in the same ranking carries: a stalled leader replaced the way
  a lease does it (Kubernetes' leader lease, CRUSH re-placing data off an "out" OSD).
- **In scope only.** In private mode the carrier candidates and "could the swarm carry it" are the
  pool's (`pool::scope::allowed_node_set`, which `holdings` already applied): a machine outside the
  pool was electable in the first cut.
- **A partial mesh can elect two.** Nodes that see different machines can rank different winners;
  the cost is duplicate parts, bounded by the shortfall, never a loop.
- **Unknown changes nothing**: no header here, or any machine with no ceiling advertised (older than
  v0.3.230) — no carrying, no protection, everything as before.

**Evidence.** `coverage::tests::*` — the carrier is the machine that can, not the holder that
cannot nor one without room; 8 layers short, a carrier with room for 16 is offered part 0 alone,
and 24 short it is offered the two parts its room holds; nothing with no demand; prune keeps a
carrying copy and sheds it once another holder can; an unknown ceiling changes nothing; a
carrier with no progress past the patience is passed over (one that gained a part is not); a
machine outside the pool is never the carrier (red with the scope removed; the lease test red
with the lease never lapsing). The carrier
test is red with carrying switched off and again with the shortfall ignored; the prune test red with
the protection switched off (its first version passed both ways — two holders, where prune would
not shed anyway; three make it real). Rig `examples/carry_test.sh`, three nodes in a private
network namespace (S whole but capped at half its footprint, C holding part 0, client K in another
region): first request refused in 0.0 s ("room for about 19 of its 22 layers"); C named carrying
30 s after K's demand reached it (K decays its counts every 600 s, then gossips) and fetched part 1
at score 1500 against routine 30; the next request was served in 1.2 s. With routine replication
satisfied C had fetched nothing for ten minutes. **The rig's first run left `min_replicas` at 2,
and C fetched the part by ROUTINE replication 30 s in, before any request — the right outcome for
the wrong reason, caught only by the mechanism check** (the carry line in C's log); the rig now
asserts it.

**A change must keep**: a new rule about which copies matter goes in `coverage.rs` and is asked by
BOTH passes; carrying never bypasses the trust gate, the storage budget or `would_shed_copy`, and
never asks a machine for more than the shortfall.

## A model's weights are its header's, on every node (2026-10-07)

**What happened.** `ModelProcessPool::footprint_inputs` — the one source of a model's memory
footprint, for admission, the local planner (`segment_cost_curve`), the model list
(`estimated_vram_mb`, `fits_on_gpu`) and `would_fit_on_gpu` — took a model's weight bytes from the
shard FILES on this node's disk, and `segment_shape` charges a segment those bytes over ALL the
model's layers. That is the model only on a node holding all of it:

- a node holding NONE of a model still has its header — routing fetches it to price peers
  (`ensure_model_geometry`) — and was charged no weights at all: our node's list showed Qwen 3.5 9B
  needing 1346 MB and Qwen3-30B 1092 MB, both `fits_on_gpu: true` (2026-10-07, v0.3.229);
- a node holding PART was charged about that part's share of its part: `xlam-2-3b` held 2 of 4
  parts, 951 of 1840 MiB on disk, so a segment of the half it held was charged about half its
  weights — an under-charge at admission on exactly the partial holders a split uses;
- and the same disk read made the first cut of #230's peer pricing compute nothing on a coordinator
  holding none of the model (gotcha #803).

**The rule.** The weights are summed from the header's tensor table
(`split::tensor_byte_size`, the size the tables are cut by): a property of the MODEL, identical on
every node, and what a load's shards carry. Measured against the files on a node holding whole
models: TinyLlama 636 vs 638 MiB, Mistral-7B 4170 vs 4170 MiB. Test
`a_models_weights_are_its_headers_whatever_this_node_holds` (none, part, and a segment's share of
the WHOLE model; red with a disk sum restored).

## nvidia-smi is asked through one bounded helper (2026-10-05)

**What happened.** At 18:52:06 UTC on 2026-10-04 a worker died of an illegal memory access and
the graphics driver began resetting the card ("UCodeReset TDR"); the reset finished at 19:06:16.
The node's re-plan logged "Starting pipeline execution" at 18:52:06.7 and its next line for that
request — "admitting model to GPU" — at 19:06:16.5, the second the driver came back: GPU admission
reads the card's free memory through `nvidia-smi` (`compute_vram_budget` →
`query_gpu_vram_free_mb`), and `nvidia-smi` does not answer while the driver resets. Every
`nvidia-smi` in the crate was a bare `Command::output()` — no bound — and several run on async
threads: the capability broadcast every cycle, admission, the dashboard's stats. During a reset
each one parks its thread for the reset's length, and each new call starts one more process
waiting on a driver that is trying to recover (FUTURE_WORK #220).

**The rule.** `vram::nvidia_smi` runs it through a `BoundedCommand`: at most 10 s (it answers in
~90 ms; the bound is generous on purpose), then `None` — the "unknown" every caller already handles
(admission charges no other program; the broadcast advertises 0 room; the dashboard shows nothing).
The timed-out process is killed and KEPT: a process blocked in the driver dies only when the driver
answers, so while it is still alive the next reading is `None` at once instead of a second process
queued on the same driver, and when it has gone readings resume ("answers again after a stall",
with the stall's length — which dates the reset in the node's own log). The lock is held for the
whole run, so two never overlap. Test: `a_reading_that_does_not_answer_is_unknown_and_never_asked_twice_at_once`;
guard: `nvidia_smi_is_asked_only_through_the_bounded_helper` (green on the tree, red on a planted
bare spawn). The launcher (`src/bin/launcher.rs`) is exempt: it asks once, before the daemon
exists.

**Placement while the driver is not answering (v0.3.230, FUTURE_WORK #220 (2)).** A worker started
on the card during a reset blocks in its context creation for as long as the reset takes, so
`vram::graphics_driver_not_answering` puts new workers on the processor
(`CpuReason::DriverNotAnswering`; promotion moves them back). It asks
`BoundedCommand::still_not_answering` with `try_lock` — a placement question never queues behind
the driver it is asking about — and only once the stuck copy has been stuck for
`DRIVER_NOT_ANSWERING_AFTER` (60 s). **One slow answer is not a reset** (gotcha #802): the first
draft of v0.3.230 acted on the 10 s bound alone, and in its gate (Windows up 151 h) the node's
`nvidia-smi` outlived it while a 9B worker handed back 5.3 GB; 34 ms later Mistral-7B — which fits
only on the card — went to the processor and was refused for lack of system memory, where v0.3.229
had served it. The driver answered within ~15 s (the safety kit's own readings). Windows detects
a hung card after `TdrDelay` (2 s) and gives threads `TdrDdiDelay` (5 s) to leave the driver
(Microsoft, "TDR registry keys"); a real reset that morning held `nvidia-smi` 4-8 s, while the one
the rule exists for (2026-10-04) held it 14 minutes — so the threshold separates them, like a
health probe's failure threshold. Tests: `a_driver_that_has_not_answered_is_reported_without_waiting_for_it`,
`a_driver_that_answers_late_is_not_reported_as_not_answering` (red with the threshold ignored).

## The owner is told when the card has become slow to hand out memory (2026-10-01)

**Why.** The pool above and `card_pace` defend the node against a host whose fresh card
allocations have slowed; neither tells the OWNER, and the owner holds the only fix. On WSL2 the
slowdown grows with Windows uptime (#146's series: a few hundred MB of fresh allocations took
0.007 s at ~8 h, 2.3 s at ~24 h, 4.0 s at ~35 h, 8.8 s at ~49 h), and only a Windows restart
clears it. The dev PC then crashed twice under GPU load at 52-54 h of uptime — a hard hang during
twelve simultaneous chats (#754) and bugcheck 0x116 VIDEO_TDR_FAILURE right after an 8B worker
exited and the next node started (#762).

**Research (2026-10-01).** microsoft/WSL#41701 is the same shape — CUDA through GPU-PV slowing
over days of host uptime, `dxgvmb_send_sync_msg` failures accumulating — and is OPEN with no
root cause or fix from Microsoft or NVIDIA; `wsl --shutdown` does not clear it; one report says
`pnputil /restart-device` on the card does, which is not advice for a card that drives the
display. NVIDIA 617.14's notes list no WSL, CUDA or TDR fix. Microsoft documents the TDR registry
keys for driver development only. The 0x116 arguments (`c000009a`, `4`) are identical to
Microsoft's own documentation example — generic, not a diagnosis. No source links the pool or
CUDA graphs to GPU-PV failures (the CUDA-on-WSL known limitations list neither).

**The rule.** `inference::cuda_pool::probe_once`, called by `split::loader::load_device` beside
`keep_freed_memory` — once per worker, before the model's weights load — times 16 fresh 4 MB
allocations straight from the driver (`cuMemAlloc`, never the pool) with their frees, after one
untimed warm-up allocation. Every worker logs `DIAG: card allocation probe` with the figure, so
the uptime curve builds up in `node.log`. The worker sends it once
(`WorkerMsg::CardAllocationProbe`, side-band like `Progress`); `ModelProcessPool::slow_card_notice`
keeps the slowest one at or over `SLOW_FRESH_ALLOCATIONS` (0.5 s); the health monitor's tick
(`maybe_warn_slow_card`) logs a warning everywhere and, under WSL2 only — where its advice
applies — shows `activity.card_slow_restart_windows` as a toast, at most once per 12 h.
A/B / off: `SWARMLLM_CARD_PROBE=0`; `SWARMLLM_CARD_PROBE_SLOW_MS=N` moves the threshold (0 = every probe slow — how the chain was checked end to end on a healthy card).

**What it deliberately does NOT do.** It changes nothing the node does: no pacing, no refusal.
The threshold is a CALIBRATION GUESS from #146's series (16 calls, 64 MB — some of the ~64 calls
and a fraction of the bytes that slowed, since whether the cost grows per call or per byte is not
known); the probe's own figure at high uptime is the measurement still owed. Read the DIAG lines
across a few days of uptime before tuning the threshold or letting the figure drive anything.

**What a change must keep.**
- **The probe goes straight to the driver.** Timed through the pool it measures nothing once the
  pool holds memory.
- **Once per worker, at the device choice** — a probe per request adds the very allocations it
  measures to every admission.
- **The advice only where it applies**: the log line on every platform, "restart Windows" only
  under WSL2.

**From the rules file (moved 2026-10-02):**

**`cuda_pool::probe_once` times 16 fresh 4 MB allocations straight from the driver, once per
worker, at `load_device`** — never through the pool, never per request. Every worker logs
`DIAG: card allocation probe`; a probe ≥ 0.5 s reaches the health monitor
(`WorkerMsg::CardAllocationProbe` → `ModelProcessPool::slow_card_notice`), which tells the owner
to restart Windows, under WSL2 only, at most every 12 h (#762, WSL#41701). It changes nothing the
node does; the threshold is a guess until the DIAG lines cover a few days of uptime.

## A lone decode stream is never held for a batch (2026-09-29)

**What it replaced.** `process_pool::batch_scheduler_loop` — which every decode forward a node
serves passes through while `inference.continuous_batching` is on (the default), on the
coordinator's own segment and on every peer's — waited `batch_collection_ms` (5 ms) after the
FIRST forward for others to join, whether or not anything else was decoding. The config said
"single-request workloads are unaffected" and that WSL2's ~15 ms timer resolution made the
window moot. Neither held.

**Measured** (RTX 3070 Laptop, both nodes of a two-node split of qwen2.5-coder-7b on the one
card, ~0 ms of network, `~/swarmllm-split-0929/split_speed.sh` + the worker's `DIAG: worker
forward received` / `answered` lines): per decoded token, 6.6 ms from the daemon's send to the
worker's receipt on EACH node — 13 of a 36 ms token; the split ran at 25.3-26.7 tok/s against
46-49 for the same model on one node (54%). The per-token breakdown after the fix: 0.3 ms per
hand-off, 5 ms of layers per node, ~3.6 / ~5.3 ms after each forward (the card finishing, the
hidden state or logits to the host, sampling at the tail), under 1 ms per network hop.

**The rule.** `collection_target(others_decoding, collection_ms, max_batch)`: no wait unless
ANOTHER request is decoding on the same model, and then only until each of those has arrived,
capped by the window. `ActiveStreams` says who is decoding by each stream's own pace — a request
counts while its next forward is due, within twice the gap between its last two (a first
forward: 250 ms; a cap of 3 s). The first version used a fixed 2 s window and the rig showed
the next of two back-to-back requests paying the full wait for its first 2 s — half of 390
tokens — which is an agent's next call exactly. Iteration-level schedulers (Orca, OSDI '22;
vLLM) never delay one request for a batch that is not there; Triton's dynamic batcher waits
only up to a queue delay it is configured to accept.

**After** (same rig, one binary, arms interleaved): one chat through the split **38.4-42.0
tok/s** against 50.3-50.9 local (~80%); two chats at once 15.1 + 18.2 tok/s against 13.7 +
15.5 on the released v0.3.212. Pooled (Nehanth/pooled) reports two devices at 0 ms emulated
latency at 86% of one.

⚠ **Two chats through a split were NOT batched before or after** — 0 `batched forward
complete` lines on either node, on either binary: their forwards reach each node a segment's
compute apart (5-9 ms), past the window. The config's "1.34-1.55× at batch 2-8" was not
reproduced in a split; FUTURE_WORK #148.
⚠ The rig puts both nodes' workers on ONE card, so what is left includes two CUDA contexts
sharing it — a real split does not pay that; the numbers above are an upper bound on its cost.

**From the rules file (moved 2026-10-02):**

**`process_pool::collection_target` decides whether the batch scheduler waits**: only when
ANOTHER request is decoding on the same model (`ActiveStreams`, judged by each stream's own
pace — twice its last forward gap), and only until those have arrived. The old rule waited
`batch_collection_ms` after every first forward: 6.6 ms per node per token for a lone split
request, the split at 54% of local; now ~80%. Never reintroduce a wait a lone stream pays.

## A model is admitted against the card as it stands NOW (2026-09-29)

**What it replaced.** `ModelProcessPool::admit_to_gpu` — the one decision that puts a model
(or a range of one) on the card — weighed it against `vram_budget_mb`, computed ONCE in
`SharedState::new`. That budget subtracts what OTHER programs hold on the card, so memory any
program took after the node started was invisible to admission, while every other reader of
the same budget (routing, downloads, scans, pruning) called `compute_vram_budget` live.

**Found** on the split rig (two nodes on one card, DSD on, a drafter on A): A admitted the
3,081 MB drafter beside its 1,621 MB segment against 6,314 MB — the figure from before B's
worker took 1,705 MB of the same card. Free memory then sat under the KV margin, and BOTH of
A's workers refused every prompt with a 0 MB conversation budget (18 refusals, three requests
failed after the router's retries), on the released v0.3.212 as on main — and with the #146
pool keep switched off, so not that change. On a desktop the other program is a game or a
browser started after the node.

**The rule.** `vram_budget_now()` re-reads the budget at every admission through a source
`SharedState::new` sets (`compute_vram_budget`: `nvidia-smi`, ~0.1 s, beside a spawn that takes
seconds), falling back to the last reading when it fails. Test
`admission_reads_the_card_budget_at_the_moment_of_admission` replays the case (verified red
with the fix reverted). **After, same rig:** the drafter was refused the card ("does not fit
the graphics memory that is free"), ran on the processor, 0 refusals, every request answered.

**From the rules file (moved 2026-10-02):**

**`ModelProcessPool::vram_budget_now` is what `admit_to_gpu` weighs a model against** —
`compute_vram_budget` re-read at the admission (other programs' use included), the last
reading only as a fallback. It was the startup figure, so a program that took part of the card
later let the node overcommit it: a drafter admitted that way left both workers with a 0 MB
conversation budget and every prompt refused. Never cache a live condition at startup.

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

## A budget is charged by whatever owns the resource

**`SharedState::committed_memory_mb` is the single answer to "how much of this
memory is committed right now"**, and it asks `ModelProcessPool`
(`vram_committed_mb` / `ram_committed_mb`) because the pool admits, charges and
reclaims. Never charge a budget by summing `split_models` — those entries are
GGUF headers read at scan time with no worker behind them, so the figure can
only go up. It read `loaded_mb=5124` eighteen seconds after boot with zero
workers, and nodes delegated models smaller than their free graphics memory.
Guard: `a_memory_budget_is_charged_by_the_pool_never_by_the_metadata_map`.

## Single-source-of-truth helpers — Worker memory: graphics, RAM and the KV cache

Each names the ONE place a decision is made. A second implementation of any of
them is this codebase's most-repeated defect — see `.claude/rules/architecture.md`
§ "One invariant, N paths". **Read the topic file before changing one.**

Full evidence: `docs/invariants/memory.md`

- **`DaemonMsg::ReleaseRequestKv` — a finished request releases its conversation cache, wherever it is held.** Sent by `ModelProcessPool::release_request_kv` at the one place a request finishes. The `Generate` handlers clear their own; the FORWARD path had nothing, and the daemon's own `cleanup_request_id` is a DIFFERENT PROCESS's store. Unconditional: the worker keys by REQUEST id, so no later turn can find the entry. A KV-admission refusal is `LocalMemoryUnavailable`, carried over IPC as `WorkerMsg::Error::local_memory_refusal` because the wording is deliberately identical to a peer's refusal.
- **`api::process_memory::resident_bytes` — a process's memory is the LARGEST accounting the platform offers.** macOS keeps two that differ by orders of magnitude; under-reporting is the failure that matters.

- **`ModelProcessPool::reply_channel_closed` — a closed reply channel is judged in ONE place.** A later call under the same `request_id` displaces the entry (the worker is healthy and busy for it) or the worker died; only the second evicts. `generate` had the check since gotcha #180 and forward, batch and draft did not — the draft wait evicted the drafter a router retry was about to use (#749). Every wait on a worker ends there. → `docs/invariants/memory.md` § "A model loaded for another model's request is a guest"
- **`inference::worker_ipc::worker_error_is_fatal`** — the single source of truth for "did this worker error destroy the worker's device state, or just this request?".
- **`daemon::shard_loader::force_cpu_for`** — the single mapping from `inference.gpu_layers` (`-1` auto / `0` CPU only / `>0` GPU) to the loader's `force_cpu` flag.
- **`daemon::gpu_support::MIN_COMPUTE_CAP` + `local_gpu_is_supported`** — the single answer to "can this card run OUR kernels?".
- **`model::auto_manage::vram::ADMISSION_KV_CONTEXT`** — the context length admission charges KV cache for, on either device, whatever the user configured.
- **`ModelProcessPool::free_vram_for_admission` + `plan_vram_reclaim`** — reclaim graphics memory from models nothing is using rather than demoting the requested one to the processor.
- **`should_return_to_gpu` + `ModelProcessPool::worker_should_return_to_gpu`** — the single answer to "is this resident worker still in the right place?", asked on the request path in `get_or_spawn` rather than on a timer.
- **Graphics memory has ONE owner: `ModelProcessPool`** — it admits (`admit_to_gpu`), charges (`vram_reserved_mb`) and reclaims (`free_vram_for_admission`, `try_idle_vram_unload`). Nothing else may take memory away from a loaded model.
- **A worker's growth is weighed by the budget its spawn charged** — `WorkerHandle::holds_gpu_memory` reads `charged_against_ram` (what `charges_ram` decided at spawn), NEVER `placed_on_cpu_because`, which records why a model was *demoted* and so reads `None` on a machine with no card exactly as it does for a worker holding one. Re-deriving it sent every later layer-range growth on every GPU-less node to `admit_to_gpu`, which has no ceiling when `vram_budget_mb` is 0 — the anti-swap gate ran once per model and never again, and a 16 GB Mac swapped. Growth is the COMMON case on a swarm node. `a_live_workers_growth_is_weighed_by_the_budget_its_spawn_charged` in `tests/repo_consistency.rs` fails the build on a re-derivation.
- **`model::auto_manage::storage_budget` is the ONE answer to "how much shard storage may this node hold?", and `held_disk_bytes` the one answer to "how much does it hold?"** — `storage_budget_now(&state)` gives both, live; the download pass, prune's disk pressure, the settings storage bar, the pool page and the diagnostics report all read it instead of re-deriving one. **The byte figure measures the DIRECTORY, not the manifest** (`held_shard_bytes` prices the manifest and so sees only `shard_NNN.bin`, missing the header, the tied output weight, mmproj, quarantined files and dead `.tmp` — 5% on the live node, 95% of it one file kind). **It is only safe to charge for them because each kind is reclaimable**: derived files by `shard::cleanup_orphaned_model_files` (once per prune cycle), quarantined shards and `.tmp` partials by `daemon::background::spawn_failed_download_reclaim` (every 10 min; over budget with NO retention, partials only when no download claims them). Prune deletes shards and nothing else, so a figure counting bytes nothing frees makes a node near its limit shed shards forever chasing an unreachable floor — or, where nothing may be shed, wedge: quarantines were swept only at STARTUP until 2026-09-28, and 9.37 GiB of them filled a field node's disk and blocked the update carrying the fix (gotcha #751). Never re-couple the two — changing the byte source without the reclaim pass is `docs/FUTURE_WORK.md` item 49's shape. And a test fixture that exercises this must own its `data_dir`, or it measures the developer's real node.
- **`model::auto_manage::prune::effective_idle_secs` — residency is a hard UPPER BOUND on "idle since", and the worker's own `last_used` is the signal that moves** — NOT `model_trust.last_request_at`, which has ONE writer (`router::distributed_exec`, so the local fast path and peer-served work never touch it) — a stale persisted value once unloaded a model five seconds after it answered.
- **An admitted prompt is RECORDED, not just decided** — `KvCacheStore::record_prompt_admission` / `outstanding_admission_bytes`; `ensure_room_for_prompt` adds the outstanding total to the live figure before `admit_prompt`, and records its own claim once admitted. **A request's claim and its reservation are released TOGETHER, by `forget_request_bookkeeping`, which `clear_request` and `cleanup_request_id` both call** — the draw-down that makes a stale claim harmless reads the request's CACHE ENTRY, so a cleanup that removes the entry and keeps the claim charges the full admission for memory it just freed (gotcha #637: 1848 MB owed against a 3042 MB budget, identical prompts refused for ten minutes). Never release one of the two maps alone.
- **`inference::split::kv_budget::admit_prompt` + `PrefixCache::release`** — ONE decision for a whole prompt, before prefill, charging live caches PLUS the prefix cache's snapshots (the same device memory, previously charged nowhere): fit → evict cached prompts, oldest hit first → refuse with a 503 at token 0. **A budget must see every tenant of the memory it bounds.**
- **`inference::split::kv_budget`** — the KV memory budget and the admission check against it.
- **A prompt of known length is RESERVED, not grown into** — `KvCacheStore::set_reserved_positions` (written by the worker at the top of `ensure_room_for_prompt`, before any budget question) sizes every layer's FIRST allocation via `new_kv_cache(.., reserve_positions)`; growth by `Tensor::cat` is O(n²/quantum) in copies and one device allocation per step, none of it charged (97 GB copied and 2279 allocations for a 20837-token prompt). The guard charges what will be allocated, read off the buffer (`kv_budget::positions_to_allocate`), never derived from `index_pos`. Absent means "grow as before"; `SWARMLLM_KV_RESERVE=0` is the in-binary A/B. → `docs/invariants/memory.md` § "A prompt of known length is reserved"
- **`inference::process_pool::worker_socket_path`** — the worker IPC socket path, and the ONLY place it is built.
