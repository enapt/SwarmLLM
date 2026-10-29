# Scheduling, routing and failover

The evidence behind the rules in `.claude/rules/arch-scheduling.md`: what each
rule replaced, what it was measured at, and what a change must keep.

Every entry here was paid for. **Read the entry before changing the code it
names** — the rule statement in `architecture.md` is the summary, this is the
reasoning, and several of these describe a fix that looked obviously correct
and was not.

## A plan that names this node for the whole model is a local generation

**`inference::pipeline::local_generate::try_local_generate_fastpath`** runs a
single-segment plan assigned to this node the way the same request runs when
there is nobody to ask — through `ModelProcessPool::generate` — instead of
sending our own worker a `LayerForward` per token.

**Why.** `SharedState::local_fast_path_for` stands the API fast path aside
whenever this node would run the model on its PROCESSOR and has peers, so the
scheduler gets to consider delegating. That is right, and its doc says the cost
when nobody better is found is "only a scheduling pass". It was not. The plan
came back naming this node and the pipeline carried it out on the forward path —
**the one execution path never wired to the prefix cache**. Continuous batching,
slot admission and n-gram speculation live on the `Generate` path and were lost
with it.

Measured (report #018, 2026-09-11): `prefix-cache HIT` appears **zero times in a
week of logs**, across several models, on a processor-only node holding a
complete 14B with six peers. Every turn of every conversation re-read the whole
prompt, on exactly the machines least able to afford it.

`remote_generate::eligible` has asserted this arrangement since it was written —
it excludes the local node with "local inference is handled by `execute_local`,
which has its own faster path". True only of requests that never reached the
router.

Four things a change here must keep.

- **A node that holds the model but has no `split_models` ENTRY is still
  eligible.** That map has one writer, `auto_manage::scan`, which refuses to
  register past a ceiling on what the node offers — so absence means "the
  ceiling was full when we scanned it", not "not ours". Requiring an entry left
  this path firing ZERO times in a full day's log while the scheduler kept
  assigning this node single local segments, i.e. #018's loss restored in full
  on a report that had been closed (gotcha #638, 2026-09-17). The fallback reads
  the MANIFEST — the same source `scan.rs` derives `is_first`/`is_last` from, so
  it cannot disagree with the registration it stands in for — and requires this
  node to hold shard 0 and the LAST shard, because a whole-model range without
  them loads a model with no `tok_embeddings` and pushes raw token ids into the
  first block (gotcha #187). A budget on what to OFFER must never decide how
  work already assigned here is EXECUTED.

- **The span is checked, not assumed from the segment count.**
  `local_whole_model_segment` requires the segment to equal the complete local
  split model's range. A segment short of either end produces hidden states, not
  tokens, and handing it to `generate` would answer with the output of a
  fraction of the model.
- **Prompt privacy is NOT a disqualifier here.** What it guarantees is that the
  prompt and the sampled tokens stay on this machine; a plan running every layer
  here satisfies it completely, with no boomerang to build. The remote sibling
  declines on it because that path puts the raw prompt on the wire.
- **It changes how the answer is carried out, never what was decided.** The
  scheduler still runs, still prices the swarm, and still delegates whenever a
  peer is worth it.

**From the rules file (moved 2026-10-02):**

**`pipeline::local_generate::try_local_generate_fastpath`** runs a single-segment
plan assigned to this node through `ModelProcessPool::generate`, not as a
`LayerForward` per token to our own worker — which is the one path the prefix
cache, continuous batching and n-gram speculation are absent from. The span is
checked against the complete model, never assumed from the segment count.

**A missing `split_models` entry does not disqualify it.** That map's one writer
refuses to register past a ceiling on what the node OFFERS, so absence means the
ceiling was full when it scanned — not "not ours". Reading it as a disqualifier
made this path fire ZERO times in a full day's log (gotcha #638). The fallback
reads the MANIFEST, the same source `scan.rs` derives `is_first`/`is_last` from,
and requires shard 0 and the last shard so the worker is never handed a
whole-model range with no embedding table (#187). **A budget on what to OFFER
must not decide how work already assigned here is EXECUTED.**

→ `docs/invariants/scheduling.md`

## The component that will refuse must be asked while the plan can still change

Two halves of one rule, both learned from a 16 GB processor-only Mac mini that
was assigned 36 of a 48-layer 14B, refused them at load, retried, and produced
the identical plan (gotcha #452).

**`ModelProcessPool::max_local_hostable_layers` is the scheduler's bound on the
LOCAL node**, built from the same estimator and the same budgets the loader will
use. Every peer already carried a `max_hostable_layers` from its advertised free
memory; the local node carried `None`, on the reasoning that *"our own loader's
admission check is the authority on what we can fit"*. That is right about WHO
decides and wrong about WHEN — admission runs at load time, after the plan is
committed and too late to reshape it. The one candidate with the best
information about its own memory was the only one priced as unbounded. `None`
still means unknowable, never "no room".

**The same rule from the other side (2026-10-04, #129): the bound must also say
YES wherever the loader would.** On a card it counted the card alone, while the
loader does not refuse a model that does not fit the card whole — it splits it
(`partial_gpu_layers`: the first layers on the card, the rest on the processor,
uncharged against the RAM budget). So a cold Mistral-7B on an 8 GB card bounded
this node at 17 of 32 layers, `local_can_hold_every_layer` answered no, and the
search's boomerang through Italy (29.8 s priced, 44.7 s measured) was taken over
this node's own route priced at 16.8 s — which, forced, ran as a 17/32 hybrid in
7.2 s warm. The bound now asks `partial_gpu_layers` itself, so a plan and the
load cannot disagree in either direction. Guard:
`a_model_the_loader_would_split_is_one_this_node_can_hold`, with an
architecture the loader will not split as its control.

**And it must be asked on the device the loader uses, not the device the
request is priced at (2026-10-05).** v0.3.225 shipped the above and the field
check failed: the planner passed `serves_on_cpu` — the SPEED answer, which is
"processor" for a model too big for the card (#444) — as the device to weigh
room on. With TinyLlama already on the card, Mistral-7B "did not fit", RAM
alone held 29 of 32 layers, `local_route_available=false`, and the request went
through Italy at 47 s. The fix lived in the card branch; its test called that
branch directly. The planner now asks `max_hostable_layers_for_planning` /
`held_layer_ranges_for_planning`, which choose the device themselves
(`planning_on_card`: the card whenever one is in play — whole or split — the
processor when there is none or the model neither fits nor splits). Guard
`the_planner_weighs_a_local_model_the_loader_would_split_on_the_card` goes
through `gather_candidates` with the speed answer "processor" and RAM for
nothing: red (`Some(0)`) on the old call, `Some(32)` now.

**And the split is a SPAWN's, for ONE range (2026-10-08, #234).** The #129 bound
answered with the split's width for every plan, but only a spawn splits — and only
the range it is spawned for. A further range GROWS that worker, and
`charge_additional_segment` weighs growth against the card alone: there is no
processor rung for growth. So a plan giving this node a second run had it grow a
worker whose card the split had just filled. A tester's RTX 3060 (v0.3.229),
holding 28 of a Qwen3-30B-A3B's 34 parts beside a worker already holding 16
layers, was planned layers 19..27 and 43..48 as well; both were refused ("need
about 2608 MB more … another holder will have to take that part"), and every
request failed.

Now the two answers are two fields. `max_hostable_layers` (`max_local_hostable_layers`)
is the card's room — what a running worker can ADD, in any number of ranges.
`fresh_run_layers` (`fresh_run_layers_for_planning`) is the split's WIDTH, offered
only while a spawn can be had (no live worker of the model, or an idle one on the
card, which `grow_worker` retires and replaces; never one in use, never one
charged to system memory). The search keeps the local share of a chain to ONE run
when it leans on the split (`parallax::local_fits`, carried along the DP as the
run count and the closed run's width, and re-checked after reconstruction);
`local_can_hold_every_layer` accepts the whole model as that one run. Tests:
`the_split_a_fresh_worker_would_make_covers_one_local_run` (its control — the old
figure — hands the node both stretches; red with the one-run limit off),
`a_whole_model_the_card_would_split_stays_one_local_run`,
`a_worker_in_use_withdraws_the_fresh_worker_split`, and the pool's
`a_model_the_loader_would_split_is_one_this_node_can_hold`.

**The run's width is a split point too (2026-10-27, #354 (b)).** The vertex filter
admitted a range up to `fresh_run_layers` for this node, but `route_over` proposed
split points only at `max_hostable_layers` and the rung's `vertex_cap` — so a run
that wide was a vertex only where another holder's boundary happened to fall. On
2026-10-25 (15:53 UTC, Llama 3.3 70B, four candidates, this node with no worker
and a fresh run of 22) this node was planned layers 0..19; nothing proposed 22.
`route_over` now pushes `a + fresh_run` and `b − fresh_run` for this node's ranges
where `capacity.binds_local()` (one closure, `fresh_run_of`, read by the points and
the vertex filter alike), as points of the pass, so they survive the budget's
first fallback. The budget, on that plan's candidates: the declared points cost
88 vertices, with the two new points (6 and 22) 131, against
`MAX_SUBRANGE_VERTICES` = 4096. In 117 logged local candidates with a fresh run
(2026-10-25/26), 32 had a range wider than the run. Where the range is no wider
than the run, nothing is added (02:08 UTC 10-27: range 0..28, run 32). Test
`a_fresh_workers_run_is_a_split_point` (this node 0..40, room 10, run 25, a peer
holding 15..40: the route is one local run of 25), red with the points removed.

**On the card** (`~/swarmllm-rig-234/run.sh`, 2026-10-08, safety kit, 0 driver
events): a node holding Qwen2.5-14B layers 0..32 and 38..48 with
`max_gpu_vram_mb = 5000`, beside a processor node holding all 48. v0.3.230
planned BOTH stretches here — a [0..32) split (16 layers on the card) and
[38..48) grown onto the card — and answered, but only by overrunning the cap:
5,782 MB on the card, because the split's share was charged nothing (#240, fixed
beside this). The refusal the tester saw needs a CHARGED worker beside another
model, which this card did not reproduce. The fix planned [0..12) and [38..48) —
22 layers, the card's room in two runs — and 26 on the peer: 200 both, 5,114 MB,
16.2 s for 64 tokens against the overrun's 13.2 s.

**`process_pool::segment_shape` prices what the worker will actually map.**
`VramFootprintInputs` has always documented `segment_layers` as "Layers in THIS
segment, not the whole model" and `quantized_weight_bytes` as "the shard bytes
this worker will map" — and its only caller passed `block_count` and every shard
on disk, with `is_first: true`, whatever the spawn was about to load. On a node
holding every shard, which is the node most likely to be handed a fraction of a
big model, that priced a 36-of-48 segment as all 48 and a privacy boomerang's
two end layers as the entire model: such a node could never serve ANY part of a
model it could not hold whole, which is the case pipeline parallelism exists
for.

Three things a change here must keep.

- **One worker can hold several segments.** Its `models` map is keyed by
  `(layer_start, layer_end, tp_rank, tp_size)`, so `charged_segments` records
  what has been paid for and `charge_additional_segment` weighs a range the
  first time it is asked for. `add_reserved` ACCUMULATES; the overwrite it
  replaced was correct only while a spawn charged the whole model regardless.
  **The spawn path records its own segment** — the fast path is the common path,
  and without that record every later forward for the same range re-charges it.
- **Nothing restates the estimator's arithmetic.** `segment_cost_curve` gets
  `(fixed_mb, per_layer_mb)` by pricing a one-layer and a two-layer segment
  through the estimator itself and differencing them, so the planner's bound,
  the incremental charge and the loader cannot drift. The estimate is affine in
  the layer count, which is why two points determine it.
- **A refused additional range is a 503**, the answer that fails over. It
  deliberately does not reclaim memory first: a first segment is worth evicting
  an idle model for, a second range on a worker already serving is not
  obviously, and reclaiming would re-enter `unload_model` under the spawn lock.

`Some(0)` from the bound still moves one layer (`cap.max(1)` in
`route_shortest_path`), so a privacy end can always be served and the loop
terminates.

**The general rule**: asking the gate after the decision only converts a bad
plan into an error. And when a struct's own field documentation describes a
generality — "THIS segment, not the whole model" — check what its callers
actually pass.

**The bound must read what the gate reads, when the gate reads it (2026-10-15,
#250, report #036).** The fresh-worker split (`fresh_run_layers_for_planning`)
was asked of a card reading the last admission had stored and never of the
RAM admission its processor share goes through, so with another program
holding most of the card the plan put a 30B here whole and the spawn refused
it — the very conversion of a bad plan into an error this section exists to
prevent. The plan now reads the card live where a spawn could follow and
weighs the processor share against the same requester's cap. Evidence and
tests: `docs/invariants/memory.md` § "A model is admitted against the card as
it stands NOW".

## A split leaves room for the model that guesses for it (2026-10-14, #275)

### What it replaced

The drafter of speculation across computers (`pipeline::engine_drafter`, a
model this node holds whole) was chosen by size alone and asked for memory
only at its spawn, as a guest that takes what stands free (`Tenancy::Guest`).
The planner had already given this node's memory to the target's layers. On a
node with no card holding Qwen3-30B-A3B shards 0-9 beside a peer holding
10-19, the plan put 28 layers here (6,967 of a 7,995 MB budget), and Qwen3-1.7B
(2,899 MB: 1,743 of weights, 896 of cache at 4,096 positions) was refused on
every request — "DSD: the drafter failed", `proposed=0` — so every reply went
unguessed, with a refusal in the log each time. A card build put the layers on
the card, the drafter fitted beside them, and the same split ran twice as fast
(#273), which is why it showed only without a card.

### Measured (`share.sh`, 2026-10-14, processor-only build of `0c0d0723`, live node stopped, safety kit)

Qwen3-30B-A3B Q2_K split between this node and a peer with an RTX 3060 in
Australia (~330 ms), three prompts per arm, each arm on a freshly started node,
in the order H Q X X Q H, replies identical in every arm:

| Arm | Here | Drafter | tok/s (wall, incl. prompt) |
|---|---|---|---|
| H | shards 0-9 (28 layers) | refused for memory | 2.79 2.68 2.75 · 2.78 2.77 2.99 |
| Q | shards 0-4 | beside them: α 0.85-0.88, γ 6-7, 76-85 ms a guess | 6.32 4.51 4.88 · 5.51 3.98 5.82 |
| X | shards 0-4 | forced off (`SWARMLLM_FAULT_DRAFT_FAIL=1`) | 2.47 2.66 2.78 · 2.45 2.67 2.71 |

X ≈ H: the smaller share here is worth nothing by itself; the room it leaves is
worth ~1.9×. Without the X arm the gain would have been credited to moving
layers onto the peer's card.

### The rule

Two halves.

1. **The drafter is chosen only if it would load now**
   (`ModelProcessPool::guest_would_be_admitted`, asked by
   `engine_drafter::engine_drafter_for`): the guest's own way through
   `get_or_spawn_once`, charging nothing, with the same arithmetic
   (`card_verdict` / `ram_verdict`, which `admit_to_gpu` / `admit_to_cpu` now
   call) and placement (`guest_charges_ram`). One that would not gives way to a
   smaller sibling, or to none. A/B `SWARMLLM_DRAFTER_ADMISSION=0`.
2. **Where this node cannot hold every layer, each planner first plans with
   room left for the drafter** (`PipelineScheduler::candidates_leaving_room_for_the_drafter`
   → `ModelProcessPool::max_hostable_layers_beside_guest`): this node's bound
   cut to the most layers after whose load the drafter's admission still says
   yes — against the cap its spawn uses: that of the request it guesses for
   (`process_pool::drafter_requester`; the swarm's for every request before
   #351, 2026-10-26). It is
   tried ONLY where every peer is held to what it offers: parallax's first
   rung (`CapacityBound::Everyone`) and greedy's strict pass
   (`greedy_assign_beside`). No route that way and each plans exactly as
   before. A/B `SWARMLLM_DRAFTER_ROOM=0`.

**Why only there, measured the hard way** (`room.sh`, 2026-10-14, the live
swarm, live node stopped). The first version tried the cut where it is now and
on a run of six arms it never fired — the Australian peer then offered too
little for any strict plan, every split came from greedy's relaxed pass, and
all arms ran 2.5-3.3 tok/s alike. Taking that for the cut being too timid, a
second version tried it on every rung and pass that bounds THIS node, the cut
holding its bound through greedy's relaxed pass. Its first plan came from
`PeersAtCeiling`: 14 layers here, 34 on the peer, which offered fewer than 20.
The peer refused (`gpu-kv-budget-exhausted`, no room for the conversation's
cache), the re-plan — that peer barred — found no route, and **the reply was
lost** where the uncut plan would have answered at 2.7 tok/s. A relaxed rung
holds a peer to what it could EVER hold, not what it offers now; the gamble is
taken because there is otherwise no plan (§ "The relaxation is scoped to the
figures that are actually unreliable": a figure "must never make a routable
request fail"). The cut always has another plan, so it may spend only what is
offered. Later in that run the peer offered enough (the cut fired at
`Everyone`, which holds peers to their offers): 6.94 / 4.10 / 5.09 tok/s with the drafter beside 14 layers (α
0.85-0.89, 85 of 102 guesses kept on the first), against 2.45-2.69 for the
same arm earlier, uncut.

**Two plan shapes for one model is what the cut adds — and that exposed a gap.**
After a cut plan the peer holds [14..48); an uncut plan (the cut switched off, or
not possible that request) asks it for [28..48), which its exact-key worker
would load as 20 new layers. Every such request was refused (`room.sh` runs 2
and 3: the `SWARMLLM_DRAFTER_ROOM=0` arms never answered). Fixed where the plan
is made, not by avoiding the second shape: a rung with no route is retried with
the parts that meet held ranges, so the held [14..48) is planned — § "Room for
more layers is not room for the layers already held", the peer half. Live (run
4, the arm that never answered, first): `parallax routed by meeting the ranges a
machine already holds rung=Everyone segments=[(0, 14), (14, 48)]`, 6.50 / 4.05 /
5.69 tok/s, the drafter admitted beside the 14 layers; the cut arm after it
6.55 / 4.01 / 5.75.

The drafter stays a guest — it still never reclaims anything. vLLM's answer to
the same conflict is to plan draft and target memory together, up front
(vllm-project/vllm#14067); the planner's local bound is the only up-front plan a
swarm node has, so that is where the drafter joins it.

### The room is the reply's cache, not a typical conversation's (2026-10-15)

**What it replaced.** The room left, the choice and the spawn all priced the
drafter's cache at `ADMISSION_KV_CONTEXT` (4,096 positions) whatever the reply:
896 of Qwen3-1.7B's 2,899 MB, for a chat whose prompt and first reply quantum
fill 1,024. A tenant's load is right to charge a typical conversation — it will
serve ones not seen yet — but a guest is loaded for ONE reply whose prompt is in
hand. The surplus came out of the target's layers: the cut held this node to 20
layers of Qwen3-30B-A3B where 23 leave the drafter its reply's room — three
layers more the peers had to offer before the cut had any route.

**Research.** SmartSpec (arXiv 2406.14066, the source of the γ = 0 rule) gives
the drafter as many cache slots as the target; vLLM allocates both models'
blocks per token from one pool. Here the target is admitted per request at its
prompt plus one reply quantum (`model_worker::admit_prompt_over`) — and the
drafter's own worker already admitted exactly that at its first call
(`handle_draft` → `ensure_room_for_prompt`). Only the load charge was 4,096.

**The rule.** `vram::KvCharge` says what a load's cache is charged for:
`Typical` (a tenant, and a plan with no prompt in hand) or `Conversation(n)`.
`n` is `model_worker::drafter_cache_positions` — the prompt and the reply's
reserve in whole quanta, its margin above every γ a check can ask
(`DRAFT_MARGIN_POSITIONS` > `MAX_GUESSES_PER_CHECK`, asserted at compile time),
the planner's (no reply budget known) never below the spawn's. It reaches the
spawn on `Tenancy::Guest { conversation }` (`IpcDraft::conversation_positions`),
the choice through `engine_drafter_for(.., conversation)` (`dsd.rs` tokenizes
first) and the planner through `candidates_leaving_room_for_the_drafter(..,
prompt_tokens)`. **The worker's grant does not shrink**:
`record_cpu_kv_budget` is the charged cache plus `RamBudget::headroom_after`,
whose two terms both fall by exactly the charge, so a later, longer reply may
grow the cache as far as before. A/B `SWARMLLM_DRAFTER_KV_CHARGE=typical`.

**Measured.** Live (`charge.sh`, 2026-10-15, processor-only release build, live
node stopped, safety kit, Windows up 48.8 h): on the real request the planner's
debug line read `room_for=23 … drafter_cache=Conversation(1024)` where the same
model planned without a prompt read `room_for=20 … Typical` — the predicted
three layers (672 MB at ~227 MB a layer). The speed A/B did not run: the
Australian peer refused even the uncut 20-layer share that hour ("0 MB available
for conversations"), so every arm failed alike. `~/swarmllm-275/charge.sh` (R vs
T, the peer narrowed to shards 7-19 and 8-19) is the rerun when it offers room.
Rerun the same morning against peer `aa3d1e66` (`beside.sh`'s R7/T7 arms, peer
narrowed to shards 7-19 = layers 20-48): the cut was computed as predicted
(`room_for=23` R, `room_for=20` T), but neither cut had a route at
`CapacityBound::Everyone` — the peer held 28-48 loaded (`new_layer_room=16`),
and its charge for 20-48 (8 new layers, the held layers' cache per #99, our
margin; not decomposed further) did not fit — so every plan ran uncut (local 0-28),
the drafter was rightly not chosen beside 28 layers, and R and T ran alike
(2.75-2.81 tok/s, first request 47.4 vs 48.0 / 47.7 s). The arm pair cannot
answer until a peer offers the cut's layers past the margin; the remaining arms
were stopped as unanswerable.

### The drafter is judged beside the layers its plan loads (2026-10-15)

**What it replaced.** The choice asked "would the drafter load NOW" — but it
loads after the target's layers here (`dsd.rs` reads ahead only once they are
held), so on a cold start it was weighed beside the layers loaded so far: none,
or a smaller range a worker held. Seen live: Qwen3-1.7B chosen beside 14 held
layers of Qwen3-30B-A3B, the plan grew them to 28, the drafter's spawn was
refused, and that reply ran with no guessing at all ("the drafter failed —
finishing this reply without guessing ahead") — the slowest regime, slower than
the n-gram loop a refused choice falls back to. The planner's cut prevents it
only where it fires; an uncut plan (no strict route, or `SWARMLLM_DRAFTER_ROOM=0`)
was the gap.

**Research.** vLLM loads the drafter beside the target before it measures the
memory left for caches (v1 `GPUModelRunner.load_model` → `drafter.load_model`,
then `Worker.determine_available_memory`) — it never judges one model's room
before the other's load. Here the loads are per request and in order, so the
equivalent is to judge the drafter against the memory the target's load will
leave.

**The rule.** `engine_drafter_for` takes the plan's ranges for this node and
their requester (`TargetHere`, built in `dsd.rs` from the assignment and
`pipeline::worker_requester`) — REQUIRED, so no caller can ask the old question.
`ModelProcessPool::guest_would_be_admitted_beside` asks the spawn's question
(`guest_would_be_admitted`) and then the planner's own: do the layers the plan
ADDS here (`layers_added_by` against `held_layer_ranges_for_planning`) fit within
`max_hostable_layers_beside_guest`? A plan the planner cut passes by
construction — its room is priced at the whole reply reserve, never below the
request's (`a_drafters_charge_covers_its_first_calls_admission`). Where it
cannot say (card plans, a drafter already loaded) it answers as before. A
refusal gives way to a smaller drafter, or to the n-gram loop. A/B
`SWARMLLM_DRAFTER_BESIDE=loaded`.

**Measured** (`~/swarmllm-275/beside.sh`, 2026-10-15, processor-only build of
`43600800`, live node stopped, safety kit; this node pretends shards 0-9 =
layers 0-28, peer `aa3d1e66` 9-19 = 26-48, so the cut has no route and the plan
loads 28 layers here against a room of 23; each arm a cold start, N N O O
interleaved as N O O N). **Mechanism:** both N arms logged `layers_to_load=28
room_beside_drafter=23` → "not choosing it" on the first request, which then
ran the n-gram loop (`ngram-only: complete … payoff_x100=111-112`); both O arms
chose Qwen3-1.7B, its spawn was refused ("the drafter failed — finishing this
reply without guessing ahead", `proposed=0`). **First request, 96 tokens:** N
51.0 / 50.4 s, O 54.9 / 52.9 s (~3 s, or 6%, to the n-gram loop over no
guessing at all). Later requests are the control — both arms then weigh the
loaded layers and refuse alike: 2.65-2.93 tok/s in every arm.

### What a change must keep

- The drafter the planner leaves room for is the one the request then
  chooses: both come from `engine_drafter::chosen`, which differ only in
  the admission filter.
- The cut is a first attempt, never a bound the ladder keeps — a drafter is
  worth layers, not a request — and only within what every peer OFFERS. Never
  on a rung or pass that relaxes a peer's bound.
- Not applied where this node runs the whole model (`local_runs_whole_model`),
  so a cut can never turn a local generation into a split.
- **On a card the room is left on the card** (#275 (4), 2026-10-22). Until
  then a card plan was not cut: the target's layers filled the card, the guest
  went to the processor, and a split's trees — which pay only with the drafter
  on a card (+41-44% on the real link, § "A split checks a tree of guesses") —
  stood aside. `max_hostable_layers_beside_guest` now leaves the guest's CARD
  footprint free on a card plan (`local_layers_leaving(.., true, .., Some(mb))`:
  what stands free after the layers, no reclaim counted, as the guest's own
  `card_verdict` will weigh it), where the guest may use the card at all
  (`spawn_may_take_the_card`, the room-free half of `guest_charges_ram`). A
  fresh worker's card/processor split (`fresh_run_layers`) fills the card
  whatever the room, so its offer alone calls for the cut and the cut withdraws
  it (`leaving_room_here`, which now decides and builds the cut, never raising
  this node's room). A card plan past the room still has a drafter where system
  memory takes it (`guest_would_be_admitted_beside`), as every card plan did
  before. SpecExec (arXiv 2406.02532) makes the same trade on consumer cards:
  the drafter keeps the card, the target's layers go elsewhere. Tests:
  `room_is_left_on_the_card_beside_the_layers_for_their_drafter`,
  `a_fresh_card_split_gives_way_to_the_drafters_room`.
- Measured on the real link (2026-10-22, `~/swarmllm-318/cardfill275q8.sh`:
  Thailand↔Australia ~411 ms, Qwen2.5-Coder-7B with this machine holding
  shards 0-5 and its card capped at 4,000 MB, the peer's RTX 3060 held to
  shards 2-7, the Q8_0 0.5B drafter at 966 MB; arms alternating by node start):
  with `SWARMLLM_DRAFTER_ROOM=0` the plan put 22 layers here (20 on the card),
  the drafter on the processor (17.6 ms a guess) and no trees ran — 5.22/5.47
  tok/s at 160 tokens, 5.97/5.87 at 400; the default was cut ("Leaving room
  beside this node's layers for the drafter", `room_for=Some(16)`,
  `fresh_run_withdrawn=Some(28)`), the drafter loaded on the card (5.4 ms a
  guess) and trees ran — 7.85/11.05 and 9.13/9.94. **Only that first pair
  compares the cut**: the peer then still held layers 6-28, and both settings
  planned 0-6 here with the drafter on the card (~10 tok/s each) — the held-layers
  residual below, seen from the far side. With the fp16 copy (2,991 MB, its F16
  weights widened to f32 at load — FUTURE_WORK #308) the room was 4 layers,
  short of the 6 this node had to hold, so no plan took the cut and every arm
  ran the card-filling plan (3.3-3.4 tok/s).
- Residuals: a target worker that already HOLDS the layers keeps them (held
  layers cost the plan nothing), so the room appears from the next cold load;
  of one model's copies the drafter is the one that reads the fewest bytes
  (`largest_that_can_draft`: Q4_K_M before Q8_0 before fp16), so the room a
  card leaves is the smallest copy's — whether a Q4_K_M 0.5B guesses as well as
  the Q8_0 is unmeasured (#134 item 3, a device-aware choice); a swarm whose peers offer too little for any strict
  plan gets no room, by design (the drafter is then not chosen where the plan
  leaves it none).
- Tests: `whether_a_drafter_would_load_is_asked_without_loading_it` (the probe
  and the spawn's guest admission agree, beside 6,967 MB and 3,000 MB),
  `a_loaded_drafter_is_asked_about_its_own_range`,
  `room_is_left_beside_the_layers_for_their_drafter` (the bound is exactly the
  most layers the drafter still fits beside; one more is refused) — the first
  and third red with the mechanisms off ("30 beside the drafter, 30 alone");
  `the_drafters_room_is_only_taken_from_what_the_peers_offer` (offered 40, the
  room is left; offered 5, the relaxed pass plans exactly as uncut — red on the
  every-rung version, 10 layers here against 48).
- The room, the choice and the spawn price ONE figure for the drafter's cache
  (`drafter_cache_positions`), the spawn's never above the room's.
  `a_drafter_is_charged_the_cache_of_its_reply` (1,024 positions charge 672 MB
  less; the room grows; the spawn's own charge agrees at its edge — red with the
  spawn charged typical: "agrees beside 21 layers", and with the estimator
  ignoring the conversation: 0 vs 672),
  `a_drafters_charge_covers_its_first_calls_admission`,
  `a_smaller_drafter_charge_leaves_its_worker_the_same_room` (red with the grant
  priced typical: 2,938 vs 3,610 MB).
- The drafter is judged beside the layers the PLAN loads here, never only
  those loaded now. `a_drafter_is_judged_beside_the_layers_its_plan_loads`
  (cold: yes up to the planner's room, no one layer past it, and the spawn
  after that load agrees; warm: held layers add nothing) — red with the plan
  ignored: "a plan loading 22 layers here", true where the spawn says false.

## Room for more layers is not room for the layers already held (2026-09-24, #95)

**`NodeCandidate::held_ranges` + `layers_it_would_add` — the local node is
charged, per range, only the layers a segment would ADD to what its live worker
already holds.** `max_local_hostable_layers` is room for layers NOT yet loaded,
weighed against everything committed — this model's own resident worker
included — and that is correct for what it answers. It was being read as "how
many layers can this node run".

**What it broke.** Found at the .201 gate, present in every release before it:
GLM-4-9B on an 8 GB card whose card had refused the model. Request 1 priced the
local candidate at `max_hostable_layers=Some(30)` and split three ways — [0..2)
here, [2..14) on a peer's card, [14..40) here — and answered. It left the worker
holding those two ranges (28 layers, 4595 MB). Request 2, seconds later, priced
the same node at **`Some(0)`**: no room for more. Charged at their width, both
held ranges exceeded zero, no local vertex could be built, every capacity rung
failed `no valid source vertex`, the `LocalUnbounded` rung planned the whole
model here, the loader refused it, the re-plan did the same, 503. The split that
had just worked could not be planned again because the planner could not see
that its layers were already in memory.

**Why per range and not a bigger count.** The worker keys its map by the EXACT
range (`model_worker::ensure_model_loaded`), so what a segment costs depends on
its shape, not on how many layers are resident:

- a range it holds EXACTLY is a cache hit — nothing is loaded;
- a range that strictly contains held ones drops them BEFORE it loads, so it
  costs its width less what the daemon releases for them;
- a range CONTAINED in a held superset is read from disk a second time (report
  #021), as is a partial overlap — full width.

Crediting the resident COUNT would have let the search pick, say, [20..40) under
a held [14..40) — the peer being faster — and the worker would then load 20
layers a second time into memory that is not there. `process_pool::layers_added_by`
is the worker's three cases, once, and both the planner and the tests read it.

**Three things a change here must keep.**

- **`releasable_layers` is the recorded CHARGE, not the width.** The spawn
  records its first segment at zero MB (its admission already paid), so when a
  later range subsumes it the daemon releases nothing for it. Pricing the spawn
  range at its width would plan consolidations the loader then refuses — the
  planner and the loader must be answering the same question.
- **The DP prices a local RUN, not each local vertex.** `merge_contiguous` runs
  after routing, so [0..2) + [2..14) reach the loader as [0..14) — which may
  subsume a held range or not, and so costs differently from its parts. The DP
  carries `(local_base, local_run)` along the best path; the post-reconstruction
  check merges the same way before summing.
- **Held ranges are split points.** They were cut at boundaries some earlier plan
  offered, often a CAPACITY point that has since moved (a peer that is now warm
  reaches further), so without them the one shape a full node can still run is
  not expressible.

**The loader was fixed to match** (`charge_additional_segment`): it admitted a
consolidating range at its full width against a total still counting the ranges
about to be dropped. It now admits the net peak — see `docs/invariants/memory.md`
§ "A range that subsumes loaded ranges replaces them, before it loads".

**And the refusal fact now reaches the last rung.** `LocalUnbounded` exists so
the loader decides and its refusal names the shortfall; on a request whose
loader has already refused, planning past our bound again re-attempts that load.
Both places that could — the rung itself and the parallax-failure fallback's
`local_only_assignment` — now consult `local_memory_refused_for_request`. The
router reports the ORIGINAL itemised refusal when the re-plan fails, so the user
sees no difference except one fewer doomed load.

Also: a live worker on the same device has already paid the fixed terms, so
`max_local_hostable_layers` no longer charges them to a further range —
`charge_additional_segment` never did.

Tests: `a_split_this_node_is_already_holding_is_planned_again`,
`adjacent_local_segments_are_priced_as_the_range_the_loader_is_handed`,
`holding_part_of_a_model_counts_toward_holding_all_of_it`,
`a_replan_after_a_local_refusal_does_not_plan_past_the_local_bound_again`,
`a_consolidating_range_is_admitted_for_what_it_adds_not_its_width`,
`a_consolidation_that_does_not_fit_releases_nothing`,
`held_ranges_carry_what_the_loader_would_release_for_them`,
`a_live_worker_is_not_charged_its_fixed_terms_twice`,
`layers_added_by_mirrors_what_the_worker_loads` — nine fixes toggled off one at a
time, each turning its test red. ⚠ The refusal-arm test's first cut set
`set_ram_budget_mb` on a pool inside a real `SharedState`, which installs a live
budget provider that figure never reaches; both arms planned with the machine's
whole RAM and the control passed for the wrong reason. The fixture now asserts
its own premise (`max_local_hostable_layers == Some(0)`).

### The peer half (2026-09-24, #99)

**A peer is priced by the RANGES it holds when it says which they are.**
`ResidentModelLayers::ranges` (`#[serde(default)]`, additive: gossip is JSON and
nothing denies unknown fields) carries each range a peer's worker holds and what
its loader would release for it, published by `resident_model_layers` from the
same `ranges_held_by` the local planner reads — so a coordinator prices a peer's
loader by the rules the peer prices itself by. Petals routes on exactly this:
servers announce the contiguous block spans they serve.

**What it broke.** Measured while verifying #95: the live node held [2..14) of
GLM-4 and gossiped "12 resident". `PeerResidency::Layers(12)` credits those 12
to ANY range, so the plan handed it [10..22) — the same credit, a different
range — which its worker keys exactly and so had to load in full: 12 new layers,
refused, failed over to a public peer.

**The trap the fix had to avoid.** For peers `max_hostable_layers` means TOTAL
layers, resident included, and `delegation_target`, `standby_has_room`, the
greedy fallback and `cheapest_whole_model_peer` all read it that way — so it
keeps that meaning. Only the pipeline search gets the finer figure,
`PublishedRoom`: room for NEW layers (the Cold-shaped bound over the same free
memory, from the same call) and the prompt's KV share. `capacity_charge` charges
a range `added + ceil((width − added) × kv/(weights+kv))` — the bytes inequality
`added × weights + width × kv ≤ usable`, divided through by one new layer's
price. **The KV term is not optional**: a layer a peer holds has not paid for
THIS prompt, and dropping it re-opens gotcha #447 (a warm 6 GB card handed 24
layers of an 8,111-token prompt, dead in attention 22 s in). Held-range
endpoints are split points for peers too, so the held shape is expressible.

**A peer that publishes no ranges is priced exactly as before** — every build
before this one, and `WarmAmountUnknown` keeps its generosity.

**Evidence.** `a_peer_is_planned_the_range_it_holds_not_one_of_the_same_length`
carries its own control (the count-only pricing hands the peer a range needing
more than its 8 new layers) and goes red with the DP change toggled off —
measured: `(10, 30)` instead of `(2, 14)`. Also
`a_peer_that_publishes_its_ranges_is_priced_by_them` (gather: ranges read, total
unchanged, room = total − held), `a_layer_a_peer_already_holds_still_costs_this_prompts_kv`,
the publisher assertion in `held_ranges_carry_what_the_loader_would_release_for_them`,
and `a_residency_entry_with_or_without_ranges_reads_as_it_was_sent` (types crate).
**Live, two nodes of the new build** (`candle-cuda`, the TinyLlama split rig):
request 2's per-candidate line read the peer as `held_ranges=[(12, 22)]
new_layer_room=Some(22)` — published by its worker, gossiped, read by the
planner; request 1, before anything was loaded, read `[]` / `None`.
**Exercised on a split whose plan MOVES between requests (2026-10-14, #275's
rig) — and it failed with partial ranges off, the shipped default.** This node
held layers 0-28 with room for 39; the Australian peer held [14..48) loaded
(planned by a request that left room for a drafter) with room for 5 new layers.
A plan that kept 28 layers here asked the peer for [28..48): 20 NEW layers by its
exact-key worker, refused — for eleven minutes, every request, until the peer let
the range go. The held endpoints WERE split points, but with partial ranges off a
candidate is cut at split points only when it cannot take its whole range, so
this node (room to spare) was only ever its whole (0, 28) and the peer's held
range could not be reached. The #99 test above passes `partial_ranges = true`,
which is why it never showed. **Fix: `parallax::route_shortest_path` retries a
rung that found NO route with `route_over(.., meet_held = true)`** — each
candidate also offered the parts of its range ending or beginning at a held
endpoint, and its own held ranges — so 0..14 here + the peer's own [14..48) is
planned. Only on no route, so no route that exists today is re-cut (partial
ranges stay off for the hop-pricing reason in `parallax_partial_ranges`'s doc).
Tests `a_held_range_is_reachable_from_a_neighbour_with_room` (control: no route
without the retry) and `a_route_that_exists_is_not_recut_at_a_held_range`
(control: offered unconditionally, the vertices DO re-cut a slow node's whole
route). Live: see `room.sh`'s confirm runs in § "A split leaves room for the
model that guesses for it".

**A machine's OWN held range is a vertex on every pass; only the cuts in OTHER
ranges wait for a pass with no route (2026-10-18, FUTURE_WORK #3 (e)).** The
retry above bundled two kinds of vertex: a candidate's range cut at someone
else's held endpoint (a re-cut of a route that may exist) and the candidate's
own held range (the one shape it runs with nothing to load, cutting nobody).
Behind the retry, the second was invisible whenever ANY route existed. Measured
on the live swarm, Qwen3-30B-A3B (48 layers): an Italian peer held shards
[0..19) with a worker on [0..18); a Belgian peer held every shard with a worker
on [18..48), ceiling 30. Italy's whole [0..19) fits, so it was never cut, and
the plan was [0..19) + [19..48) — 29 layers Belgium's exact-key worker reads
again (`layers_added_by`: contained in a held superset). Delegated to Italy,
which re-planned onto the two held ranges, warm repeats were priced 84.9 / 90.3 s
(78-83 s of it `cold_load_ms`) and took 19.4 / 29.4 s; driven from here, the
same plan would have made Belgium load the 29 layers. Petals routes over the
blocks each server has LOADED (`sequence_manager.py`), never a span it would
load — the same shape.

A second fault put that plan on the riskiest rung. Belgium read
`new_layer_room=Some(0)`: its room for new layers (weights + this prompt's KV)
rounded DOWN, against a held range's charge — the reused layers' KV — rounded
UP to a whole layer. 30 held layers of a 25-token prompt carry ~1.5 MB of KV
against ~360 MB a layer, yet "under one layer free" refused the peer its own
range on `Everyone` and `PeersAtFaceValue`, and the plan went to
`PeersAtCeiling` ("a peer may refuse"). `PublishedRoom` now keeps the room
unrounded (`room_for_new_layers`, `usable_bytes_for_layers` shared with
`max_hostable_layers`) and `NodeCandidate::capacity_charge` returns the
fractional charge; whole-layer arithmetic for THIS node's running total reads
`layers_charged`.

**Evidence.** `two_peers_whose_workers_already_chain_are_planned_that_chain`
(the live figures): with the own-held vertices back behind `meet_held` it plans
`[(0, 19), (19, 48)]` on `PeersAtCeiling`, the live plan; with the room and
charge rounded it finds no route on `Everyone`. Both red, both restored green.
`a_peer_with_less_than_a_layer_free_still_runs_the_range_it_holds` (charge 1
of a 0.6 room when rounded). `a_route_that_exists_is_not_recut_at_a_held_range`
stays green: the re-cut it guards is a cut in ANOTHER candidate's range. Live
(`~/swarmllm-3e/check.sh`, the dev build as this node, 14:42-14:46Z): 5/5
answered, Belgium's room read 0.39 where the release read `Some(0)`, and a
held-range route was found on `Everyone`; the morning's exact shape had moved
on by then, so the test above carries its reproduction.

**From the rules file (moved 2026-10-02):**

**`NodeCandidate::held_ranges` + `layers_it_would_add`** charge a local segment
only the layers it would ADD to what this node's live worker holds, by the
worker's own exact-key rules (`process_pool::layers_added_by`) — never its
width, and never a resident COUNT. `max_local_hostable_layers` is room for NEW
layers; reading it as "layers this node can run" 503'd the second request for
every split this node had just served (#95). The DP prices a local RUN (what
`merge_contiguous` will hand the loader), held ranges are split points, and a
re-plan after the loader's refusal never plans past the local bound again.

**A peer that publishes its ranges (`ResidentModelLayers::ranges`) is priced the
same way** — `NodeCandidate::capacity_charge` against `PublishedRoom`'s room for
NEW layers, KV of reused layers included (#99). ⚠ **`max_hostable_layers` keeps
meaning TOTAL for peers** — four consumers read it so; only the search gets the
finer figure. A peer publishing no ranges is priced as before.

→ `docs/invariants/scheduling.md`

## The hand-off gate proposes; the priced search decides

`assemble_pipeline_for` no longer RETURNS the whole-model hand-off. When the
priced search is going to run — this node on its processor, `parallax_routing`
on, more than one candidate — the gate's plan is held in `hand_off` and the
search chooses; the plan is taken only where the search declined to price
anything (`ProcessorRouteVerdict::NoComparison`, or the search failing to route
at all), and never where it made a real comparison and this node won.

**Why.** A gate that returns before the search means the one case where the gate
is confidently wrong is the one case nothing checks it. Every routing defect of
the last three releases was an instance, each fixed by teaching the gate one
more thing the search already knew: **#447** chose a card 500 ms away over the
LAN cards it structurally could not see (they hold halves, so they are not
delegation candidates at all); **#478** priced `(0, num_layers)` while assigning
a boomerang's middle; **#479** vetoed a chain on a term the search does not use.
Three fixes in two releases converging on one structural fact — there were two
decision-makers for one decision, which is the shape `graphics memory has ONE
owner` and `the storage budget has ONE accountant` already name elsewhere in
this file.

**The search only recently became able to subsume it, and both preconditions
postdate the reasoning it replaced.** `gather_candidates` has priced the local
candidate at PROCESSOR speed since #444 (2026-09-03); `route_shortest_path` has
added split points at 1 and n−1 when this node holds every layer since v0.3.163.
The note on `boomerang_assignment` — "constructed rather than searched ...
nothing in its cost model knows the local node is about to fall back to its
CPU", verified 2026-08-18 — was true of a cost model that no longer exists. **A
comment recording a verification records the date it was true.**

Four things a change here must keep.

- **This must not become "never delegate".** The same peer offered the model is
  still taken when it really is the cheapest route — pinned by
  `the_whole_model_peer_is_still_chosen_when_it_is_genuinely_cheapest`, the
  control beside `a_whole_model_peer_no_longer_ends_the_search_before_it_runs`.
  The feature exists because a processor node beside an idle card is the failure
  being fixed (#442/#444).
- **`StayHere` and `NoComparison` are different answers.** They were one `Err`
  while the only thing the caller did with either was keep the request local.
  With the hand-off as a fallback they part company: a search that priced this
  node and preferred it has overruled the gate; a search that could not price it
  has said nothing, and discarding the peer would strand a node whose own speed
  is merely not yet measured. Do not collapse them back.
- **The gate is still the whole decision where the search cannot run** — parallax
  routing off, or nothing else to compare against. `search_will_decide` is
  computed BEFORE the gate, because it decides what the gate's answer is FOR.
- **`delegation_target` keeps the one judgement the search does not make**:
  trust. Its latency and speed terms are performance heuristics the search
  prices properly; its capacity term is `max_hostable_layers`, which the search
  also honours.

**Still open**: `DELEGATE_MAX_LATENCY_MS` and `DELEGATE_MIN_CPU_SPEEDUP` are now
belt-and-braces on a plan that has to survive pricing anyway, and the gate could
in principle be reduced to the trust filter alone. Left standing because they
are the only thing deciding the fallback on a node where the search cannot run.

**Field-verified 2026-09-08, and the measurement moved the open question.** A proper
A/B — both arms the installed CUDA release binaries, same data dir and swarm,
`SWARMLLM_INFERENCE_GPU_LAYERS=0` on both, matched uptime, holder map asserted
identical — changed the plan on **5/5 models**. Two distinct behaviours, both as
designed: on privacy-on models .163's gate CONSTRUCTS a boomerang unconditionally while
.164 prices it and keeps the request local; on the privacy-off model `delegation_target`
sorts by LATENCY and takes the first survivor, so .163 took the NEAREST peer (15.2 tok/s
@ 551 ms) where the priced search took the genuinely cheaper one (23.9 tok/s @ 606 ms).
That second case is the change doing precisely what it was written for.

**But the throughput did not follow, and that is now the live risk in this design.**
meta-llama-3.1-8b, 150-token reply, warm: .163's boomerang 4.53-4.56 tok/s against
.164's all-local 3.87-4.52 — a tie, where `vertex_cost` prices the boomerang's middle at
`2 * latency * ASSUMED_FORWARD_PASSES` = ~70 s of network against ~13 s local and so
predicts local by ~5x. **Handing the decision to the search made that constant
load-bearing**: it no longer merely colours a plan the gate had already chosen, it
decides every delegation. An overestimate here biases the whole swarm toward keeping
work local, which is the opposite of what pipeline parallelism is for. Do NOT tune the
constant before establishing which half is wrong — the forward-pass count, or the
per-token `2 * latency` — and note `ack_srtt_ms` is already measured on real forwards
where `latency_ms` is a ping. See `docs/FUTURE_WORK.md` #3 ("The routing cost model's
network term overestimates a boomerang").

**From the rules file (moved 2026-10-02):**

`assemble_pipeline_for` no longer RETURNS the whole-model hand-off. When the
priced search is going to run — this node on its processor, `parallax_routing`
on, more than one candidate — the gate's plan is held in `hand_off` and the
search chooses; the plan is taken only where the search declined to price
anything (`ProcessorRouteVerdict::NoComparison`, or the search failing to route
at all), and never where it made a real comparison and this node won.

→ `docs/invariants/scheduling.md`

## A peer that will read the plaintext prompt clears a trust bar, on every path that can assign it

`scheduler::trusted_with_the_plaintext_prompt` is the one bar. **Three paths can
put a node on layer 0, and it is read by all three**: `delegation_target` (the
hand-off), `route_shortest_path`'s source filter, and `greedy_assign_inner`'s
first-segment narrowing. `standby_may_take` applies it to the fourth place the
assignment can happen — a standby, which is handed the segment's input on
failover.

**Why it is shared.** `trust_score` was consulted in exactly one place in the
whole scheduler: the hand-off gate, whose own comment called it "not trusted
enough to be shown the prompt". The search applied nothing, so any chain it
built could put a docked peer on layer 0. That was invisible while the gate
returned first — and making the search the decision-maker would have retired the
only trust check there was.

**The bar shipped on two of the three paths, and the gap composed badly.**
`greedy_assign_inner` is reached whenever `parallax_routing` is off OR
`route_shortest_path` returns `Err` — and **the bar itself can cause that
`Err`**, by removing the only layer-0 source. So tightening the search increased
how often the unguarded path ran, and on that shape the docked peer took layer 0
anyway by a longer road. A confidentiality check has to be asked at every site
that can make the assignment; asking it at some of them can be worse than asking
it at none.

**The stand-down is a statement about the ROUTE, not about a vertex.**
`route_shortest_path` runs its pass seeded from trusted sources, and only if that
reaches no sink does it re-run seeded from all of them. The old form asked
whether a trusted source VERTEX existed, which gets this shape wrong: a trusted
peer holding only `(0, 4)` makes the bar "enforceable", the docked peer holding
the model whole is dropped, and with nothing covering the rest the search fails a
request it previously served — the exact opposite of what the comment beside it
promised. The second pass costs nothing in the common case, because it runs only
when the first found nothing, which is when the request was about to fail anyway.

**Prompt privacy is structural, and a standby is part of the structure.**
`find_standbys` took no `encrypted_pipeline` parameter at all, so a remote node
could stand by for the first segment (which reads the plaintext prompt) or the
last (which samples the tokens) — and one failover would have sent it exactly
what the boomerang exists to keep local. The guarantee held until the first
failure. Refusing means such a segment may have NO standby; that is the trade the
user asked for, and `segments_without_standby` reports it honestly.

Four things a change here must keep.

- **Every narrowing stands down rather than failing a routable request**, in the
  shape `CapacityBound` uses for the memory figures. A bar that refuses to serve
  is worse than the exposure it prevents. Pinned in both directions:
  `greedy_still_answers_when_every_layer_zero_holder_is_docked` and
  `the_trust_bar_stands_down_on_the_route_not_on_a_vertex`, against the controls
  `a_trusted_route_is_preferred_over_a_cheaper_docked_one` and
  `the_greedy_fallback_applies_the_prompt_trust_bar` — which use a docked peer
  priced CHEAPER, since that is when the bar actually has to bite.
- **One predicate, parameterised.** The source test was written out three times
  with subtle differences, so a clause added to one would silently desynchronise
  the others and the bar would stand down believing in an alternative the filter
  rejects. It is now `source_ok(v, apply_trust)`.
- **It does not apply to a middle segment.** Under `encrypted_pipeline` the
  source is this node by construction, and a peer running middle layers sees
  hidden states, never the token ids (structural, not cryptographic: the boomerang
  keeps the prompt's text on this machine, it does not hide the activations from
  the peer computing them) — so narrowing who may take the middle
  would cost the boomerang its whole point.
- **It is about CONFIDENTIALITY, not speed or reach.** `DELEGATE_MAX_LATENCY_MS`
  and the reach tier stay in the gate; the search prices those itself.

**And the warning that reports a stand-down is rate-limited and names its
subject.** The condition is persistent — a docked peer stays the only layer-0
holder until someone's trust or holdings change — and `assemble_pipeline_for`
runs the search up to three times per assembly as it relaxes `CapacityBound`,
with the dashboard's route preview calling it too. Unrate-limited that is three
WARN lines per request for ever, with nothing that could ever silence it
(`PROMPT_TRUST_WARN_EVERY`). It now carries the peers it let through and their
trust scores, because "no sufficiently trusted node holds layer 0" with no
structured fields gives an operator nothing to look up and no machine to act on.

**From the rules file (moved 2026-10-02):**

`scheduler::trusted_with_the_plaintext_prompt` is the one bar. **Three paths can
put a node on layer 0, and it is read by all three**: `delegation_target` (the
hand-off), `route_shortest_path`'s source filter, and `greedy_assign_inner`'s
first-segment narrowing. `standby_may_take` applies it to the fourth place the
assignment can happen — a standby, which is handed the segment's input on
failover.

→ `docs/invariants/scheduling.md`

## An unmeasured candidate is priced pessimistically, never excluded

`priced_from_a_measurement` is one predicate with one meaning — "does the cost
model have anything real about this candidate" — and both consumers must read
it the same way. They did not.

`delegation_target` treats unmeasured as PERMISSIVE: the price gate stands
aside and the peer may still be handed the whole model, because "a peer never
measured is still tried; the measurement that first request produces is what
stops the second". `pipeline_may_replace_processor_route` treated the same fact
as DISQUALIFYING, vetoing any chain containing such a peer — while its own doc
claimed to hold "the same standard". The multi-hop path was strictly stricter
than the single-hop one and nothing said why (gotcha #479).

**The prior is the conservatism.** `UNKNOWN_COMPUTE_MS` was raised from 0
precisely so an unknown competes on a pessimistic footing — "deliberately
nearer the pessimistic end so an unmeasured node does not outrank a measured
good one". A veto on top means it can never do the job it was raised to do.

**And the arithmetic says what the veto blocked.** An unpriced peer over `L`
layers costs `UNKNOWN_COMPUTE_MS * L * ASSUMED_FORWARD_PASSES` (1600·L ms) plus
`2 * latency * ASSUMED_FORWARD_PASSES`; a local processor at `e` tok/s costs
`2000·L/e`. The peer wins only below **e = 1.25 tok/s** ignoring network, and
below **~0.5 tok/s** for a peer 500 ms away. The search therefore reaches for an
unmeasured peer only when running here would take minutes — the one case where
trying the unknown is warranted.

What replaced the veto: **the BASELINE must be priced.** Giving up "running it
here" is the question, so "here" needs a price; with none, stay home — home has
no network term and no peer to be wrong about. Exploration is bounded by
machinery that already exists (ACK fast-fail, `find_standbys` sorting local
FIRST, `is_transient_remote_failure`), which is the abandonability half of
hedged requests without the cost of running both.

Two things a change here must keep. **No invented desperation threshold** — the
crossover is derived from the cost model's constants, and a product judgment
about "how long is too long" is the #451 mistake. And **a routing test must pin
the local speed** (`with_local_processor_speed`): `mem_bandwidth` under-reads in
a debug build (~0.85 tok/s against ~5 in release, gotcha #427), which straddles
the 1.25 crossover, so an unpinned test asserts a property of the machine it
runs on.

## A gate named for a comparison must make it, and against a route that exists

Two reports from one machine, one knot (2026-09-07, reports #017/#018).

**`pipeline_may_replace_processor_route` now takes `RoutePrices`** — the local
figure, the chain figure, and whether running the whole model here is a route
this node's memory can actually offer. It refuses a chain priced at or above
the local processor, and the caller logs the reason the gate returned rather
than a fixed sentence.

**What it replaced.** The function checked two things, neither a price: is the
chain remote, and is our own speed measured. `local_ms` and `chain_ms` were
computed at the call site *only to be logged* — the comment said so, meaning
checked by a person afterwards. Live: a two-segment chain priced **22117 ms**
took a request from a processor priced **6313 ms**, under a log line reading
"a pipeline across peers' cards is priced faster" directly above its own
contradicting numbers.

**Why the DP does not already answer this, which is the part worth keeping.**
`route_shortest_path` minimises, so the omission looked safe and its own doc
asserted the search "has come back cheaper than running here". But the
capacity-respecting pass DROPS any vertex a candidate cannot hold, including
the local node's whole-model vertex once `max_local_hostable_layers` bounds it
(gotcha #452). What the search returns is therefore the cheapest **feasible**
chain, which is a different claim from "cheaper than staying home" — and the
two diverge exactly on the machine that provoked both reports.

**So the comparison is only sound where the local route is real.**
`local_ms` prices the whole model as one local segment; on a node whose loader
will refuse that, it prices something that cannot run — gotcha #478's error
again, one function along. `local_route_is_available` is the discriminator: when
it is false there is no baseline to give up, the chain wins at any price, and
the log says *that* instead of claiming it is faster. Adding the comparison
without this flag would have sent report #018's request home to a 503.

**And the fast path asks the same question.** A node was chosen for the whole
model on `available_ranges` alone — a fact about STORAGE — and
`local_can_hold_every_layer` is now asked beside it. Holding every shard is not
holding every layer in memory: a 14B priced at 10374 MB against 9240 MB of live
headroom was committed to the node before anything checked, and `admit_to_cpu`
has **no re-plan behind it** — it fails the spawn, the caller returns 503, and
no retry follows because `should_retry_after` sees `used_remote_segment == 0`
for a local-only assignment. Two peers had offered to serve that model. A node
that cannot hold everything now falls through to the priced search, which has
known how to give it only what it can hold since #452.

Four things a change here must keep. **Unknown never excludes** — a `None`
capacity is an unreadable footprint or an unset budget, not evidence, and plans
the request exactly as before. **Coverage and capacity stay separate
variables**: the decision below still needs to know this node holds the model in
order to explain itself, and a candidate silently withdrawn cannot say why it
went. **The all-local arm is tested first**, since a chain with no remote
segment is the fast path by another name whatever the prices say. And **every
refusal keeps its control**: a genuinely cheaper chain must still displace the
processor, or this becomes "never delegate" — the failure #444 exists to
prevent.

**The general rule.** When a function's name, its log line, or its doc asserts a
comparison, check that something performs it. Two of this project's rules
already say a comment describing a mechanism elsewhere is a claim rather than a
fact; this is the same trap turned inward — the claim was about the function's
own caller, and it had been true of nothing since before gotcha #479 edited the
function without touching it.

**From the rules file (moved 2026-10-02):**

Two reports from one machine, one knot (2026-09-07, reports #017/#018).

**`pipeline_may_replace_processor_route` now takes `RoutePrices`** — the local
figure, the chain figure, and whether running the whole model here is a route
this node's memory can actually offer. It refuses a chain priced at or above
the local processor, and the caller logs the reason the gate returned rather
than a fixed sentence.

→ `docs/invariants/scheduling.md`

## A re-plan is warranted by a changed fact, never by a failed attempt

**`SwarmError::LocalMemoryUnavailable`** is what this node's own loader returns
when its memory budget refuses a model, and it is the one local failure
`should_retry_after` re-plans with no remote segment involved. Before the retry,
the router records `SharedState::note_local_memory_refusal(request_id)`;
`local_can_hold_every_layer` lets that outrank both of its estimates, so the
second plan **cannot** hand this node the whole model.

**Retrying is the dangerous half, and the recorded fact is what makes it safe.**
A retry against an exhausted resource is the amplification pattern behind most
metastable failures — retry storms account for over half of them in the
published surveys — and admission here refuses *before* allocating anything, so
every live figure the second plan reads is the figure the first plan read. A
blanket retry would therefore re-derive the identical route and re-attempt the
load that just failed: strictly more load on the memory that ran out, and two
failures where there had been one.

The shape that makes it a failover instead is Kubernetes' scheduler: an
unschedulable pod is not retried on a timer, it is moved to `UnschedulablePods`
and requeued when a **queueing hint** says an event has occurred that could
change the answer. The loader's verdict is our event, and it is the only new
information in the system — which is why it is recorded rather than re-derived.
The retry then puts *no* further load on the exhausted budget, because the plan
it produces cannot include the load that failed.

Four things a change here must keep.

- **The wire wording is deliberately identical to `ServiceUnavailable`'s.** A
  peer's refusal crosses the network as text, `message_means_peer_cannot_serve`
  matches that prefix, and `reclassify_flattened_error` deliberately does NOT
  produce this variant — so a remote refusal is blacklisted and retried exactly
  as it always was, in both directions of a mixed-version swarm. The variant is
  a LOCAL routing distinction, not a new thing to tell anyone.
- **Its `ServiceUnavailable` sibling must NOT gain the same retry.** A dead
  worker or a failed spawn re-plans to the identical route; that is why its
  retry is gated on a remote segment having been involved, and the control test
  asserts it still is.
- **The original error survives a failed re-plan.** Where nothing else can serve
  the model, the user gets the itemised shortfall — the footprint, the budget,
  the setting to raise — not the re-plan's "no route", which is a true statement
  about a search they never asked for and can do nothing with.
- **It never docks a peer.** The failure names this machine, so
  `failure_is_penalty_worthy` exempts it beside its sibling.

Verified live: a node whose budget refused a 3074 MB model against 2200 MB
answered the request after the re-plan — `assemblies=2`, `segments=1` becoming
`segments=3`, the middle segment on a peer. And on a node with no peers at all,
the constrained-node harness confirms the refusal message is unchanged.

**The fact must bar what was refused, not only the whole model (2026-10-08, #234).**
"Cannot run the whole model" was the only record, and it bars nothing from a node
that holds PART of one: the tester's node above was re-planned the very ranges it
had refused, and the request failed with the advice "another holder will have to
take that part" never acted on, though the other holder sat 0.31 ms away. The pool
now records what a LOAD refused — the layers it would have ADDED to what the
worker holds, on the planner's device (`ModelProcessPool::note_load_refusal`, at
`forward_for_request`, `forward_batch` and `generate_attempt`); the router's
`note_local_memory_refusal` takes it into `local_memory_refusals` (the fewest across
refusals), and the re-plan gives this node at most one layer fewer to add and no
fresh-worker split. A refusal that named no layers (a conversation's memory) keeps
the old effect alone. Test: `a_replan_after_a_refused_range_gives_this_node_fewer_new_layers`
(red with the cap ignored; its control plans the estimate, and another request is
untouched).

**A refusal of the PROMPT bars all this node's layers, held ones included
(2026-10-13, #263).** A tester's node held 30 of Qwen2.5-14B's 48 layers in three
local segments between peers in Italy, Belgium and Thailand; a 6,002-token prompt's
cache did not fit beside the weights, and the LAST local segment refused it after
the peers had read the prompt (277 s, 125 s and 30 s). The refusal named no layers,
so the re-plan — which charges the local node only the layers it would ADD
(`layers_it_would_add`, right for weights: held ones cost nothing new) — built the
same 30-layer plan and failed the same way: two requests, four attempts, ~15 min of
peers' time, no reply. A held layer's WEIGHTS are free; its share of this prompt's
cache is not, which is exactly what `PublishedRoom` already charged a peer (gotcha
#447) and nothing charged us. Now the refusal comes from the up-front admission
(`ModelProcessPool::admit_prompt`, `docs/invariants/memory.md` § "A split's local
segments are admitted together") and carries the layers the cache fits at this
length (`model_worker::layers_that_fit`, rounded DOWN so the re-plan cannot fall
one layer short again); `note_local_memory_refusal` keeps it beside the load's
figure (`LocalMemoryRefusal`, each the fewest across refusals);
`gather_candidates` turns it into a TOTAL — `max_hostable_layers = min(held + room
for new, fit)` — and sets `NodeCandidate::prompt_memory_refused`, under which
`capacity_charge` is a range's WIDTH. `parallax`'s `local_added` asks
`capacity_charge` (it asked `layers_it_would_add` directly; identical for the local
node before this), so the vertex filter and the DP's running total agree. Greedy
already reads `max_hostable_layers` as a total. vLLM's `AllocStatus.NEVER` is the
same idea from the other side: a prompt that cannot fit is not re-queued for the
same answer. Tests: `a_replan_after_a_refused_prompt_holds_this_node_to_the_layers_that_fit`
(gather side) and `a_prompt_this_nodes_memory_refused_is_replanned_with_its_held_layers_counted`
(the search: 30 held layers planned as the control, ≤ 10 after) — both red with
the clamp and the width charge removed.

**…and on every pass of every planner.** The real-path rig (`split_rig.sh latekv`,
A's cache 74 MB short of a 5,097-token prompt) refused up front in 0.8 s against
the release's 261 s, but still re-planned TWICE: the search's last rung declined
to plan past the refused node ("not planning past that bound a second time"), and
the greedy fallback beneath it relaxed EVERY bound on its second pass — the local
node's included — and rebuilt the refused plan. `NodeCandidate::memory_refused_this_request`
(set for this node from `local_memory_refused_for_request`, load and prompt
refusals alike) now keeps `max_hostable_layers` in force in `greedy_assign_inner`'s
`cap_of` on the relaxed pass too, so the answer is `SwarmShortOfMemory` at once
when nobody else holds the layers. A field on the candidate, set where candidates
are built, rather than an argument each greedy caller could pass wrong. Test:
`a_relaxed_greedy_pass_keeps_this_nodes_bound_once_its_memory_refused_the_request`
(red with the field ignored).

**From the rules file (moved 2026-10-02):**

**`SwarmError::LocalMemoryUnavailable`** is what this node's own loader returns
when its memory budget refuses a model, and it is the one local failure
`should_retry_after` re-plans with no remote segment involved. Before the retry,
the router records `SharedState::note_local_memory_refusal(request_id)`;
`local_can_hold_every_layer` lets that outrank both of its estimates, so the
second plan **cannot** hand this node the whole model.

→ `docs/invariants/scheduling.md`

## What prompt privacy costs is said for the route actually taken (#165)

**`privacy_cost_of_plan`** prices the plan the planner RETURNS — a boomerang:
starts and ends here, someone else in the middle — against the search's own
cheapest route with privacy off (bounded rungs only: a route past the peers'
offers is no fair price), both through `parallax::chain_cost_ms`. It runs at the
planner's exit, beside `plan_exceeds_offered_memory`, so every rung is covered.
**`announce_privacy_cost`** is the one place the notice is decided and said, for
this pricing and the gate's one-peer pricing (`report_privacy_cost` →
`privacy_cost_ms`): the bars, the rate limit, the wording.

**What it replaced.** The only call sat on the gate's path, gated on
`!search_will_decide`, so with the priced search on (default) and more than one
candidate it never ran. GLM-4-9B, held whole here but not fitting the card, ran
a five-segment boomerang at 95 s for 48 tokens on 2026-10-04 with nothing said.

Three things a change must keep:

- **Reported, never acted on.** A downgrade triggered by slowness is one a slow
  peer can trigger; RFC 7507 is the reason (see `privacy_cost_ms`).
- **A preview never speaks**: it is nobody's request, and the rate limit it would
  spend belongs to the user's next real one.
- **Only a plan privacy shaped is priced.** A plan that never leaves this node
  costs nothing by it.

Verified on a live request (dev build): Llama-3.1-8B through `4a3ac72e`,
`privacy_extra_ms=0`, `without_privacy_ms=78060` — the far peer, not privacy, was
the cost, and nothing was said. `the_search_s_boomerang_says_what_privacy_costs`
(red without the call), `a_route_preview_never_announces_a_privacy_cost`,
`only_a_boomerang_is_priced_for_privacy`.

## Prompt privacy embeds the prompt with the first segment's own loader, or refuses (2026-10-20, #316)

`inference.local_embedding_privacy` promises that a first segment on another
computer is handed hidden states, never the prompt's token ids. ⚠ It does not
hide the prompt from that segment: layer 0's input is a row of the model's own
embedding table per token, which the segment holds and matches back EXACTLY —
describe it as keeping token ids off the wire, never as hiding the prompt
(the docs said "harder to invert" until this change). **It had never held on
a real node.** Found checking a release reviewer's finding (a Gemma 3
picture's spans dropped on this path) on `split_rig.sh repeat`: the "on" arm's
node logged `Failed to load local embedder from shard_000 ... unknown magic
0x3f800000` and its replies were the "off" arm's — the n-gram path had served
all three.

**Three defects, each enough on its own:**

1. `LocalEmbedder::load` read `shard_000.bin` as a GGUF file. Shards stopped
   being GGUF files before the feature was written (2026-02-25/28 vs 03-11):
   a part's first bytes are tensor data (`0000 803f` = 1.0f), the header is
   `gguf_header.bin`. Every load failed, `local_embedders` stayed empty, and the
   pipeline's `if local_embedder.is_some()` guard — the only check — never ran.
2. Had it loaded, the loop set `index_pos = token_count` BEFORE the prompt
   pass, so the pass went out at the prompt's length. The worker writes its
   cache at its own length but rotates by `index_pos`: every prompt row was
   rotated N..2N-1 while decode continued from N — the reply decoded from
   "before" its prompt. The same value hid a Gemma 3 picture's spans
   (`image_spans_from(index_pos)` answers nothing once `index_pos` is past
   every picture) and doubled `advertised_context_refusal`'s count.
3. The DSD, speculative and n-gram paths never asked the setting; each sends
   the first segment token ids. The hand-off and the led split did ask it.

**What replaced it.** The embedder is the first segment's loader asked for no
layers — `daemon::shard_loader::load_model_range` (extracted from the worker,
so both read a model's weights one way: a reconstructed `model.gguf`, a
`source_path`, else the parts through `first_header_disagreement` and the
tensor table), on the processor (the daemon holds no graphics context). Its
rows go through `SplitModel::embed_ids` → `embed_tokens`, the one lookup —
equal to a first segment loaded with one layer, bit for bit, on Llama 3.2 3B,
TinyLlama and Gemma 2 2B (`local_embedder_load_from_real_model`). The table is
the rows-on-demand quantized one (308 MB for Llama 3.2 3B), where the old
design dequantized the whole table to f32 (1.5 GB). It loads when a request
first asks (`local_embedder_for`, on a blocking thread), not at startup, where a
model fetched later never got one.

**Refuse rather than send.** Where it cannot be built and
`segment_zero_may_leave_this_node` (the first segment elsewhere, or a stand-in
elsewhere for layer 0), the request is `EmbeddingPrivacyUnavailable` — nothing
is sent. Its own variant: `PromptPrivacyUnavailable`'s message and hint name
"Start and finish on this computer", a dashboard switch that may be off, and
the way out here is a config-file setting. The fast paths stand aside for such a plan inside
`fastpath_request_disqualified`, which all of them ask. Failover hands a
stand-in for segment 0 an embedded prompt only if it advertises
`FORWARD_PRE_EMBEDDED` (the planned first segment already had to), and never
token ids under the setting. `SharedState::local_embedding_privacy` is the one,
live reading (the hand-off read the boot snapshot).

**Only where the first layers may leave (the review, 2026-10-20).** The first
version embedded whenever the setting was on and the embedder loaded — a
first segment run HERE too, where the token ids never reach a wire. That input
skipped everything only the token path does: the worker's pre-embedded branch
passed no LoRA adapter (a LoRA request, always a whole local plan, answered
from the base model without a word — it now applies the adapter, which changes
only the layers' projections, `model::lora`); both tensor-parallel paths read
their input as the prompt's text (the member's `EmbedOnly` step, the
non-member's plain forward to its own worker); the split prompt cache and the
prompt read in pieces both stand aside for `pre_embedded`. Now
**`PipelineExecutor::prompt_embedder`** embeds only under
`segment_zero_may_leave_this_node`, and **`tensor_parallel_group_for`** gives
no group an embedded first segment (it goes the ordinary way, which says the
input is embedded). The boomerang's read of `local_embedders` is gone: no
release ever filled the map, and filled now by this setting, an entry left
after it is turned off would have embedded a boomerang's local first segment.

**A whole hand-off asks too.** The HTTP forward to a pool member
(`api::openai::resolver::find_peer_with_model`, for a model this node does not
know) hands the peer the request whole — it renders the prompt from the text —
and asked neither privacy setting, while the libp2p hand-off and the led split
asked both. **`SharedState::privacy_refuses_a_whole_hand_off`** is now the one
question all three ask; the HTTP path answers with the setting's refusal
(`EmbeddingPrivacyUnavailable` / `PromptPrivacyUnavailable`) when a member
would have been chosen, never a "no model loaded" after the cold-start wait,
and only then — a request no member could take goes on to a cloud provider as
before.

**What a change must keep.** A new path that sends segment 0's input asks
`segment_zero_may_leave_this_node` under the setting (through
`prompt_embedder`), or goes through the plain loop. A new path that hands a
request whole asks `privacy_refuses_a_whole_hand_off`. The prompt pass is read
from position 0 on every path. The boomerang (`encrypted_pipeline_for`) never
embeds: its first segment is local by construction. Tests:
`the_prompt_is_embedded_only_where_the_first_layers_may_leave`,
`an_embedded_first_segment_is_never_run_by_a_tensor_parallel_group`,
`the_http_forward_to_a_pool_member_asks_both_privacy_settings`,
`an_embedding_privacy_refusal_names_the_setting_that_refused` — each red with
its fix off.

## A holder's refusal is reported as what it was

**`router::report_after_a_replan`** decides what the caller hears when a holder
turned a request down and the re-plan found nothing; **`router::
a_refusal_the_caller_should_hear`** decides which failure is kept for it. A
peer's refusal (`a_peer_declined`: a remote segment took part, `Service
unavailable` in its words, not our own loader) is kept beside our own memory
shortfall and a peer's context limit, in the one slot.

**What it replaced.** The refusing holder is barred for the request
(`blacklist_holder_for_request`) so the re-plan cannot return to it — right for
the retry, and the reason the re-plan's own answer was wrong for the caller. With
the only holder of a range barred, the coverage walk said "No reachable node holds
layers X-Y … the peer that held that piece has gone", or the candidate gather said
"Insufficient network capacity", about computers that were connected and had
answered in under a second. Seen 2026-10-04 on every route shape: a whole-model
hand-off refused for memory (Qwen3-30B-A3B on `4a3ac72e`), a segment refused
mid-pipeline (TinyLlama layers 12-21, arriving wrapped in
`SegmentFailoverExhausted`), and a hand-off to the one remaining holder (TinyLlama
on `9594e1ff`).

**Two variants, because the advice is opposite** (`arch-errors.md`). The user
asked whether those refusals meant the model could NEVER load there — and for two
of the three it did: the M4's budget is 80 % of its 16 GB at Maximum contribution
against an 18.7 GB model, and `9594e1ff`'s budget left 0 MB for a conversation
after TinyLlama's weights. The swarm already knew in the first case: the search
logged "no route fits even what the peers themselves advertised" and handed the
M4 all 48 layers against an offer of 0. So the scheduler records, at its exit and
for real requests only, that a plan hands some peer more layers than it offered
(`plan_exceeds_offered_memory` — peers only: this node's figure is room to ADD,
not a total — and `layers_offered`, the figure the caller is shown).

- Refused plan past the offers → **`SwarmShortOfMemory`**, "the computers holding
  it have room for about N of its M layers between them": retrying cannot help.
- Refused plan inside the offers → **`HoldersDeclined`**, in the last holder's own
  words (the innermost `Service unavailable: …`): usually busy, retry later.
- Offers that add up to the whole model → `HoldersDeclined`: the room exists and no
  route lined it up, which a later attempt may get past; never "room for about 50
  of its 48 layers".

Four things a change here must keep:

- **The figure is read at the refusal**, not at the end: the re-plan runs with the
  refusing holder barred, so a later total describes fewer computers.
- **A re-plan that failed another way is that failure** — a peer that then went
  silent, our own loader's shortfall (actionable on this machine), any 4xx.
- **A peer that DIED did not say no.** `SegmentFailoverExhausted` with "connection
  closed" is not kept, and "has gone" is then the right answer.
- **The re-plan's own `SwarmShortOfMemory` is no route, like "has gone"** (2026-10-22,
  gotcha #931). A peer that refused a prompt up front (#252 (b)) is held on the re-plan to
  its `layers_that_fit` — 0 on a busy moment — so the planner's "no plan fits the holders'
  ceilings" (`scheduler::short_of_memory`) described that one prompt's moment, and was
  passed through as the swarm's verdict: "trying again won't help", on the card rig's first
  request, with the next one 10 s later running. It now joins `no_route`, so the first
  plan's offers decide. Test `a_re_plans_shortfall_after_a_peer_said_no_is_the_holders_declining`.
- **Neither variant docks a peer** (`failure_is_penalty_worthy`): a holder's
  admission check refusing is that check doing its job.

Verified live with the fixed build (2026-10-04): Qwen3-30B-A3B → "Not enough
memory in the swarm … room for about 43 of its 48 layers between them" (offers
19 + 11 + 8 + 5; the re-plan's own error was "No reachable node holds layers
0-4"); TinyLlama through `9594e1ff` alone → "… online but turned it down — the
last one said: Not enough free memory on this node for a 43-token prompt (…)".
Tests: `a_holder_that_said_no_is_not_reported_as_gone`,
`a_refused_plan_past_the_holders_offers_is_a_swarm_short_of_memory`,
`a_re_plan_that_failed_another_way_is_reported_as_itself` — red with the peer
refusal not kept; `a_plan_past_a_peers_offer_is_recognised_and_one_inside_it_is_not`
on the Qwen3-30B holders' real figures. Its sibling #219 — a peer whose offer it
could never honour, so its refusal read as "busy" — is closed under § "A peer
advertises the memory it will HONOUR".

## The relaxation is scoped to the figures that are actually unreliable

**`parallax::CapacityBound`** says whose `max_hostable_layers` a routing pass
honours, in four rungs: `Everyone`, `PeersAtFaceValue`, `PeersAtCeiling`,
`LocalUnbounded`. `assemble_pipeline_for` walks them in that order, and the
local layer budget is enforced INSIDE the DP — carried along the best path,
exactly as the capped-peer bitmask is — as well as by the exact summed check
after reconstruction.

**Why the bound is scoped at all.** The relaxation exists because a PEER's
figure is a self-report: stale by up to a health tick, zero on any node older
than v0.3.103, absent for a peer that has gossiped no capability. Such a figure
may make a route better and must never make a routable request fail. But
`respect_capacity` was one boolean over every vertex, and the local node is a
vertex — so the pass also discarded a figure that is none of those things. Ours
comes from our own loader, inside the very scheduling call that consumes it,
from live memory, from the estimator `admit_to_cpu` will use minutes later. So
dropping it never rescued a request; it moved the refusal from the planner,
where the plan can still change, to the loader, where it cannot.

Measured on a 16 GB machine (report #025, gotcha #489): `max_hostable_layers=
Some(40)` logged one line above, the constrained pass refusing 48 layers by
name, the relaxed pass then returning the identical all-local chain, and
`admit_to_cpu` refusing it 50 ms later. **Every request to that model failed,
for as long as the memory picture held.** The v0.3.162 fix (report #018)
changed which log line explained the failure, not whether it happened, because
it closed the fast path and this is the search's own second pass.

**A capacity ceiling is a split point, not only a cap** (report #029,
2026-09-09). `route_shortest_path` builds its candidate boundaries from
`available_ranges` — facts about DISK, where a candidate's shards begin and
end — plus the model's ends and, under encryption, the boomerang's. Capacity was
consulted only to CAP what a candidate is handed, and a cap can reject a range
the search proposes but cannot propose the range that would fit. So where no
declared boundary happened to fall at the point memory runs out, a
capacity-respecting route was not merely passed over: it was **not expressible**,
and every rung of `CapacityBound` refused in turn down to the one binding
nobody.

Live shape: a 48-layer model whose four candidates could hold 17, 9, 14 and 19
layers — 59 between them — with every one holding the whole model on disk, so
the only boundaries on offer were 0, 48 and the boomerang's 1 and 47. The search
cut at 1 and 29 and handed 28 layers to a peer that could take 9. It refused;
the retry produced the same two boundaries with a different peer; the request
failed. The node with 17 layers spare was fragmented into slivers of 1, 3 and
13. Eight requests in ninety seconds, same shape.

`max_hostable_layers` now contributes two points per (candidate, range): the
furthest it reaches from the range's start, and the earliest it can start and
still reach the end — **and only if they fit the same sub-range budget the
splitting is already held to** (`capacity_points_fit_budget`). The emission is
O(k²) per range in the points inside it, so two more per candidate is free for a
handful of holders and quadratic at swarm scale: with 42 candidates, keeping
them took the routing call from microseconds to **4.6 seconds**, on the
request's own scheduling path. They are an enhancement, so when they do not fit
the declared boundaries simply stand.

Note what the budget does NOT govern. A candidate that cannot hold its whole
range is split regardless — `!split_enabled && !over_capacity` continues only
for candidates that FIT — because that is a correctness bound, not the
throughput optimisation. So dropping the capacity points costs expressiveness
for candidates that fit, never correctness for those that do not. The first
version of this note assumed otherwise, and the test written on that assumption
could not fail — a candidate may take a prefix or a suffix of what it
holds, and which is useful depends on its neighbours. Pinned by
`a_route_everyone_can_afford_is_expressible_even_with_no_declared_boundary`,
whose null control removes those points and fails with `parallax: no candidate
ranges` — one of the three `constrained_err` values seen 36 times in one node's
own log on the day this was found.

**This is why the relaxation fired so often.** Both halves are the same defect
seen from opposite ends: #028 made the relaxation prefer routes that respect the
peers' figures, and #029 is why such a route was frequently not among the ones
on offer.

**Why a relaxation spends the margin first** (report #028, 2026-09-09). The
three reasons the relaxation names for unbinding a peer are *stale*, *zero on a
pre-v0.3.103 node*, and *no capability gossiped at all*. Two of those never
reach this enum: `max_hostable_layers` returns `None` for an absent capability
(`let cap = capability?`), for `free_mb == 0` ("not 'no room': no information"),
and for an uncomputable per-layer size — and `None` is unbounded on every rung.
So the only thing an unbinding rung can act on is a figure that is PRESENT and
NON-ZERO, which is to say a peer that told us a real number. Staleness is what
is left, and staleness is exactly what `DELEGATE_VRAM_MARGIN` was already
discounting for.

Hence `max_hostable_layers_at_face_value`: the same figure with the margin
spent, computed from the same call so the two cannot drift, and a rung that
uses it before any rung goes past a peer's own word. Live shape, 2026-09-09: a
coordinator planning `qwen2.5-14b` put layers 0-29 — about 5526 MB — on a peer
advertising a 4096 MB budget, twice in eleven minutes six minutes apart, and it
refused both times quoting the number it had been advertising all along.

This is where every cluster scheduler landed. Kubernetes filters on fit and
never relaxes the memory predicate to place a pod; an infeasible pod stays
Pending with the shortfall itemised ("0/2 nodes are available: 2 Insufficient
memory"). Where overcommit is allowed it is a bounded declared ratio between
request and limit (OpenShift), never the constraint being dropped. Omega's
answer to a stale view is to resolve the conflict at commit time and re-plan —
which `note_local_memory_refusal` already does here for the local node — not to
place beyond what the machine reported.

**But it is a preference, not a wall**, and report #025 is why. There the only
route ran across a peer whose figure refused it, and relaxing that figure kept
the request alive. A self-report we cannot re-ask must not fail a request
outright. So the third rung (`PeersUnbounded` until v0.3.230, `PeersAtCeiling`
since) still rescues #025; it is simply no longer reached while a route that
respects the peers' own numbers exists. Pinned by
`a_peer_is_not_handed_more_than_it_says_it_can_hold_while_a_route_exists`, whose
null control — making `PeersAtFaceValue` return `None` — hands a peer
advertising 9 layers all 48.

**Never past what a peer could EVER hold** (v0.3.230, a tester's report of
2026-10-07 on v0.3.229). #025's rescue never needed "held to nothing": a peer's
CURRENT figure can sit below what its admission will grant because it is stale,
or because an idle model holds memory admission will reclaim — but never by more
than the peer's whole ceiling. The rung relaxed to nothing, and a plan past the
ceiling is a refusal booked in advance:

- the one holder of a whole Qwen 3.5 9B was a 6 GB processor-only peer in
  Australia capped at `max_ram_mb = 5200`; the model needs 6688 MB at its
  admission; handed over, refused 8.8 s later, no other route;
- a 30B was planned 1 / 46 / 1 layers, the 46 to a 16 GB machine ~16.5 GB short
  of them; refused after 87 s;
- our own node (also v0.3.229) planned the same 9B to the same peer.

The current-figure arithmetic could not see it: it charges the model's FILE bytes
per layer plus the prompt's KV, which is lighter than admission's footprint (KV
for the admission context, output head, overhead) — so 22 of TinyLlama's 22
layers "fit" a cap of 784 MB against a 980 MB footprint in the rig below.

So each node advertises `NodeCapability::model_memory_ceiling_mb` (additive,
`#[serde(default)]`; `vram::node_model_memory_ceiling_mb`): the RAM cap on a
processor node — exactly what admission refuses past — and the whole card plus
the swarm's RAM cap on a card node. (It was the card plus the whole machine's
memory, written when the loader's split did not charge the processor's share to
the RAM budget; since #264 that share is admitted against the requester's cap,
and every reader of a ceiling is a peer, judged at the swarm's `cap_mb` (#266).
Left as it was, a 12 GB card whose owner gives the swarm 24 GB advertised room
for all 80 layers of a 70B where a peer could be given ~66 — 2026-10-25, #344.)
An upper bound by construction, because a ceiling set too low would refuse
routes the peer would accept. The planner weighs it with the PEER's own admission
arithmetic, which is now ours too: `segment_cost_curve(model, processor)`, read
once per plan — a model's weights come from its HEADER on every node
(`footprint_inputs`, `docs/invariants/memory.md` § "A model's weights are its
header's"), so a coordinator holding none of the model prices a peer exactly as
the peer prices itself (the first version read this node's DISK and computed
nothing for exactly the report's case; the rig caught it, gotcha #803). The processor curve, because it
has no f16 KV mirror, so a card peer is never weighed more strictly than its own
admission weighs it; the slope is rounded UP, so the line is exact for the whole
model and never above the estimator for a shorter segment.
It is applied ONCE, where candidates are built (`clamp_to_ceiling` in
`gather_candidates`, over `max_hostable_layers`, the face value and a published room), so
every reader inherits it — the rungs and greedy, and delegation, the whole-model disqualifier
and the standbys, which read `max_hostable_layers` directly and were missed by the first cut
(the review of these commits; a peer with an unknown current figure and a known ceiling was
still handed a whole model there). A peer whose current figure is unknown is held to its
ceiling everywhere; greedy's second pass runs whenever a ceiling is known (it was skipped,
and the caller got a generic planner error); and a plan
no ceiling fits is answered by the planner itself as `SwarmShortOfMemory`
(`scheduler::short_of_memory`), "retrying won't help", before any peer is asked.
Where the holders' ceilings add up, the model is SPLIT across them instead of
handed whole to one.

That verdict needs every holder to be known, so it is given only where every layer HAS one
(`first_layers_held_by_nobody`); a layer nobody holds is `ModelIncompleteInSwarm`, which the
router waits on and plans again (`assembly_failed_for_lack_of_holders`). Greedy walks the
layers in order, so a full holder met BEFORE the gap answered first: on 2026-10-26, 3 min
after a restart (the 70B A/B's arm 2), Llama 3.3 70B was refused "not enough memory" with two
candidates — this node holding 0-28 and a peer holding 0-54 in pieces, capped at 23 — and
nobody yet known for layers 54-80; the peer holding all 80 was a candidate 20 s later and the
next ask was answered. Petals treats a block no server is known to hold the same way
(`MissingBlocksError`, retried by the request path; its client's first table update "was
likely to produce incorrect information and cause MissingBlocksErrors until the next
update"). Pinned by `a_full_holder_before_a_layer_nobody_holds_is_the_gap_not_a_shortage`.

What it needs, and what happens without it: the model's header on the
coordinator. `ensure_model_geometry` fetches it from HuggingFace before planning
(8 s budget) and keeps it, so a connected node has it from its first request; a
coordinator without it, and a peer that advertises no ceiling (older than
v0.3.230), are bounded exactly as before — unknown is never "no room".

Pinned by `a_peer_is_never_handed_more_than_it_could_ever_hold` (parallax; null
control: an older peer stays unbounded), `a_model_past_every_holders_ceiling_is_refused_before_anyone_is_asked`
(greedy) and `a_model_no_holder_could_ever_hold_is_refused_by_the_planner` (end to
end through `gather_candidates`: split at 24 + 24 of 32, refused at 11 + 11,
control unbounded; every routine figure out of `gather_candidates` is within the ceiling —
red without the clamp, `Some(32)` against 24) — all three red with `within_ceiling` ignoring
the ceiling, the last also red with a peer priced from zero bytes. Rig:
`examples/ceiling_test.sh` (two nodes, private network namespace, the server's
cap at 80% of and just above its own admission footprint).

The same work found the cost curve itself bent by #228 (2b): the estimator
charges KV by `kv_layers`, and `segment_cost_curve` scaled the layer count but
not `kv_layers`, so every point charged the WHOLE model's cache and a partial
segment was charged all of it (the local planner under-counted what fits).
`process_pool::cost_curve_of` scales it, from points at the whole model and half
of it — Pinned by `the_segment_cost_curve_recovers_the_fixed_and_per_layer_terms`
(red without the scaling).

A note on what was NOT built. Learning from the refusal itself (Omega's other
half) would need the coordinator to tell a memory refusal from any other 503,
and it cannot: `LocalMemoryUnavailable` displays as `Service unavailable: {0}`,
so it flattens to `ServiceUnavailable` across the wire and is indistinguishable
from a spawn failure or a broken pipe. Recovering it from the message prose is
the #295 trap. Carrying it structurally is an additive protocol change, and the
benefit it buys — roughly one second on a request that fails either way — did
not justify it against the risk of refusing routes that would have worked.
(The 2026-10-07 report measured that second at 8.8 s and 87 s; the ceiling above
answers it without learning from the refusal at all.)

**Why the DP, and not only the check after it.** The local node is exempt from
"a capped candidate appears at most once" — prompt privacy needs it at both ends
(gotcha #481) — so the per-vertex cap cannot bound what it takes in TOTAL:
several local sub-ranges, each inside the cap, sum to the whole model, and
`merge_contiguous` hands it exactly that. The summed check ran after path
reconstruction, where failing abandons the WHOLE search rather than the one
chain that broke the rule. So a perfectly good boomerang through the peer that
held every layer was discarded along with it. Checked inside the DP, the bad
chain is simply never built and the search returns the cheapest one that fits.

Four things a change here must keep.

- **`LocalUnbounded` stays, as the LAST resort.** With no route even inside our own
  memory there is nothing to protect, and the loader's itemised refusal —
  which names the footprint, the budget and what to raise — is a better answer
  to a single-node install than "no route". This is why the fix is not simply
  "respect the local bound always".
- **The DP bound is a sound bound, not a complete search.** It is carried along
  the single best path, so a cheaper predecessor that exhausts the budget can
  hide a costlier one that would have fitted — the same approximation
  `used_capped` already makes. Both backstops behind it are unchanged: the
  exact summed check, and the next relaxation.
- **Every pass says which one it is.** The `PeersAtCeiling` line promises a
  re-plan and can now keep it: the only refusal it invites is a peer's, and
  `should_retry_after` retries that. The `PeersAtFaceValue` line says the
  margin has been spent and no more. The `LocalUnbounded` line promises
  nothing and says the loader will decide.
- **`Some(0)` still moves one layer.** Both the DP bound and the summed check
  apply `cap.max(1)`, so a privacy end can always be served and the search
  terminates.

**The general rule.** When a flag's name, doc or log line describes one
population and its parameter reaches all of them, that gap is the bug — and a
constraint checked after a search kills the search instead of the candidate.

**From the rules file (moved 2026-10-02):**

**`parallax::CapacityBound`** says whose `max_hostable_layers` a routing pass
honours, in four rungs: `Everyone`, `PeersAtFaceValue`, `PeersAtCeiling`,
`LocalUnbounded`. `assemble_pipeline_for` walks them in that order, and the
local layer budget is enforced INSIDE the DP — carried along the best path,
exactly as the capped-peer bitmask is — as well as by the exact summed check
after reconstruction.

**A capacity ceiling is a split POINT, not only a cap.** `route_shortest_path`
builds boundaries from `available_ranges` (disk) plus the model's and the
boomerang's ends; `max_hostable_layers` now also contributes the furthest a
candidate reaches from each range's start and the earliest it can start and
still reach the end. A cap can reject a proposed range but cannot propose the
one that fits, so without these a capacity-respecting route is not passed over —
it is not expressible, and every rung refuses down to the one binding nobody.

**A relaxation spends the safety margin before it spends the peer's own
number.** `max_hostable_layers_at_face_value` is the peer taken at its word with
`DELEGATE_VRAM_MARGIN` spent, and `PeersAtFaceValue` sits above
`PeersAtCeiling` so a route that respects what peers actually claimed is always
preferred to one that does not. Unknown is unbounded on every rung and always
was — `max_hostable_layers` answers `None` for an absent capability, a gossiped
zero, or an uncomputable per-layer size — so a relaxation can only ever act on a
figure that is present and real.

→ `docs/invariants/scheduling.md`

## A result the peer sent and a result we made up are not the same delivery

`LayerResult::locally_constructed` is the discriminator, and `#[serde(skip)]`
plus an explicit `false` in the binary decoder is the whole mechanism: the field
cannot survive either codec, so **anything that arrived over the network reads
false by construction**. `pipeline::local::wait_for_result` reads it to choose
between `SegmentOutcome::Returned` and `SegmentOutcome::AbandonedLocally`.

**Why it is needed.** Three paths end a forward by handing the waiter a
manufactured `LayerResult::error` rather than letting the wait expire — the ACK
fast-fail sweep (`fail_tensor_forward`), a peer whose connection closed and whose
re-dial failed (`fail_layer_results_awaiting`), and a closed pipeline stream. All
three complete the oneshot, so they land in the same `Ok(Ok(result))` arm as a
peer's own refusal, and the arm scored every one of them as an intact delivery.
Since the ACK deadline is 10-90 s inside a segment budget that ran to 300 s (the ceiling is now `SEGMENT_TIMEOUT_MAX_SECS` = 600 s),
that was the NORMAL way a dead link was observed: the peer-reliability term
shipped in v0.3.164 credited a peer whose link had died with a perfect delivery,
and the only thing that could ever score against a peer was the local compute
deadline expiring — a slow processor, not a lossy link.

**Not a string match.** The reason is a `String` and matching on it is the #295
trap; the wire format answers the question directly and cannot be reworded.

Three things a change here must keep.

- **A peer's refusal is still an intact delivery.** Out of memory or a missing
  shard is a perfect delivery of a "no", and the distinction is compute against
  transport — the one `failure_is_penalty_worthy` draws. Pricing a refusal as a
  lossy link steers traffic away from a peer whose network is fine. Pinned by
  `a_returned_segment_is_recorded_as_an_intact_delivery`, the control beside
  `a_forward_this_node_abandoned_is_not_credited_to_the_peer`.
- **A serving-side constructor may set it freely.** `LayerResult::error` sets it
  unconditionally because the wire strips it: a refusal built on the serving node
  reaches the coordinator as false. That is what makes the rule hold with no
  per-call-site decision to forget — the failure mode this codebase keeps
  hitting.
- **A test harness standing in for a peer must clear it.** Otherwise it simulates
  our own ACK sweep rather than the peer replying.

**The intact sample is taken on the prompt pass only; a failure is always
taken.** `note_segment_delivery` owns that rule. Two reasons. The path runs once
per segment PER TOKEN while the whole-model fast path (`remote_generate`) records
once per reply, and both feed one EMA at `ALPHA = 0.3` — `1 - 0.7^n` passes 0.99
by fifteen samples, so a 150-token reply over three segments (450 samples) buried
the single failure that ended the request and left the multiplier inert on the
path it was added for. And it is real cost on the per-token forward path: a
DashMap exclusive shard lock plus a `NodeId` clone per segment per token.

**`peer_delivery_samples` is logged beside `expected_attempts`**, because the
multiplier reads 1.0 both for a reliable peer and for one nothing is recording
for — the ambiguity that hid all of this, and the reason an accessor added to
resolve it is worthless while only tests can reach it.

**What this still cannot see, and must not be stretched to cover.** Loss on a
healthy TCP path shows up as retransmission LATENCY, not delivery failure — the
forward completes, slowly. So this term catches links that break, not links that
are merely bad, and it is not the fix for the netem case in issue #21. That needs
per-peer goodput (built since: `GoodputEstimator`, in the next section). Do not weight samples by payload size
as a substitute: the ACK estimator already declines transfer-dominated samples
(`ACK_OBSERVE_MAX_BYTES`) precisely because they measure the payload rather than
the peer.

**From the rules file (moved 2026-10-02):**

`LayerResult::locally_constructed` is the discriminator, and `#[serde(skip)]`
plus an explicit `false` in the binary decoder is the whole mechanism: the field
cannot survive either codec, so **anything that arrived over the network reads
false by construction**. `pipeline::local::wait_for_result` reads it to choose
between `SegmentOutcome::Returned` and `SegmentOutcome::AbandonedLocally`.

→ `docs/invariants/scheduling.md`

## Latency wants an average; capacity wants a maximum

`AckRttEstimator` (RFC 6298 smoothing) and `GoodputEstimator` (a windowed max)
are the two halves of "how good is our path to this peer", and they are
deliberately opposite in every respect. **`ACK_OBSERVE_MAX_BYTES` and
`GOODPUT_SAMPLE_MIN_BYTES` are the same number**: the round-trip figure is taken
only from SMALL forwards, where the time is the peer's, and throughput only from
LARGE ones, where the time is the payload's. A sample is dominated by one or the
other and cannot measure both.

**Why goodput exists at all.** Loss on a healthy TCP path is absorbed by
retransmission, so it appears as a transfer taking longer and NEVER as a forward
failing. That is why the delivery-ratio term (#495) structurally cannot see it,
and why a peer at 60 ms with 3% loss out-sorted one at 81 ms with none while
being 2.9x slower on a 513 KB payload — measured from outside, in a contributor's
netem lab (issue #21). It also captures a rate limit, which no small-message
probe can detect.

**Shaped after BBR's bottleneck-bandwidth estimator**, which solves the same
problem — deriving a path's capacity from whatever transfers an application
happens to make. Three rules taken from it, each of which is easy to get wrong
and two of which were:

- **A windowed MAX, not an average.** Samples come in low for reasons that say
  nothing about capacity. This is the same argument
  `mem_bandwidth::remeasure_keeping_the_best` already makes locally, and the
  exact opposite of what latency wants.
- **An app-limited sample may RAISE the estimate but never establish or lower
  one.** BBR uses such a sample only when it exceeds the current estimate; with
  no current estimate there is nothing to exceed, so it is discarded. Both
  halves matter and the second was missing at first: the max filter already
  stops a small sample lowering anything WITHIN a window, so the rule looks
  redundant until the window rotates — and a long conversation is one prefill
  then thousands of decode steps, so two rotations later a 2 KB forward would
  have established the figure at the speed of the decode loop. **A wrongly-low
  estimate is far worse than none**, because unknown charges no transfer while a
  low one charges an enormous one and routes around a healthy peer.
- **The round trip is subtracted before dividing.** An acknowledgement is sent
  once the whole message has ARRIVED (gotcha #446), so the observed time is
  propagation plus transfer, and `vertex_cost` charges latency separately.
  Leaving it in would double-charge it and would understate throughput worst on
  exactly the distant peers this exists to rank.

**How it is consumed.** `VertexCost::transfer_ms`, a term of its own — NOT folded
into `network_ms`, which is multiplied by `ASSUMED_FORWARD_PASSES`. The large
payload crosses once, on the prompt pass; every decode step after it carries one
position. It applies only to a segment that is entered per token AND does not
start at layer 0, because a first segment receives the PROMPT
(`ActivationUnits::PromptBytes`) rather than hidden states — three orders of
magnitude smaller — and charging a transfer that does not happen would penalise
exactly the split the search should be free to choose. Only the inbound
direction is charged: the return payload varies by shape, `2 * latency_ms`
already carries the round trip, and a conservative stated term beats a
speculative one on a cost model whose own calibration is an open question
(`ASSUMED_FORWARD_PASSES`, `docs/FUTURE_WORK.md`).

**Unknown charges nothing**, so a peer never sent a large forward is priced
exactly as before this existed — the standing contract of every routing input
here. Local and NEVER gossiped, like `ack_srtt_ms`: it describes OUR path to that
peer, which is not a fact about the peer. And `goodput_samples` is published and
logged beside the estimate for the reason `peer_delivery_samples` is: an
unmeasured path and a fast one are indistinguishable from the figure alone, and
that ambiguity is what hid #495 being inert.

**A test for a max filter must cross a window boundary.** `rotate_for_test`
exists because each half-window is five minutes (`GOODPUT_WINDOW_HALF`) and the app-limited rule governs only
what happens across rotations — the first version of its test passed with the
rule disabled.

**From the rules file (moved 2026-10-02):**

`AckRttEstimator` (RFC 6298 smoothing) and `GoodputEstimator` (a windowed max)
are the two halves of "how good is our path to this peer", and they are
deliberately opposite in every respect. **`ACK_OBSERVE_MAX_BYTES` and
`GOODPUT_SAMPLE_MIN_BYTES` are the same number**: the round-trip figure is taken
only from SMALL forwards, where the time is the peer's, and throughput only from
LARGE ones, where the time is the payload's. A sample is dominated by one or the
other and cannot measure both.

→ `docs/invariants/scheduling.md`

## What a peer costs per VISIT is not what it costs per layer

**Rule**: `.claude/rules/arch-scheduling.md` § "What a peer costs per VISIT".

### The measurement

A forced 4-segment spread on 2026-09-20 (`llama-xlam-2-8b`, gotcha #659)
produced three readings that together say the cost model was missing a term:

1. **The slow member dominates regardless of how few layers it gets.** One
   sample put **2 of 32 layers** on the slow peer and time-per-token went
   **152 ms → 3768 ms** — the other 30 layers cost less than those 2. Identical
   shape to the 2026-09-05 privacy report, where 2 of 36 layers were 55.3 s of
   73.8 s.
2. **Peer speed does not track RTT.** Solo on one model: 105 ms → 5.39 tok/s,
   1043 ms → 2.66, 643 ms → **0.48**. The peer a second away is five times
   faster than the one two-thirds of a second away.
3. **A wider spread is not costly for its hops.** The chain runs peer-to-peer;
   the coordinator makes one round trip. The cost is compute on the worst
   member.

### Why the model could not express it

`PeerSpeed` normalised every decode sample to `segment_ms / layers` and priced a
candidate at `coefficient × layers`. That is a line through the origin, so it
asserts a peer given half as many layers costs half as much — and reading (1)
says that is false by more than an order of magnitude.

The arithmetic of the mistake: a peer whose real cost is `1000 + 5 × layers`,
measured over 32 layers, records `1160/32 = 36.3` ms/layer. Priced for 2 layers
that is **73 ms**, against a true **1010 ms**. The router picked the cheapest
option available and the option was fictional.

### The fix

`LinearFit` — decayed least squares on `(layers, segment_ms)`, weighted like the
EMAs beside it so one sample moves it by `ALPHA`. Five running sums, no history
kept. `decode_terms()` solves it, and `NodeCandidate::observed_fixed_ms_per_visit`
carries the fixed half into `vertex_cost`'s existing per-visit slot — the one
already multiplied by `ASSUMED_FORWARD_PASSES` for a segment entered per token.

### What a change must keep

- **The fallback, which is most peers.** Two terms are separable only when the
  samples span differing widths; a peer holding one shard of one model is
  always given the same width, and `TWO_TERM_MIN_LAYER_VARIANCE` declines there.
  `None` prices exactly as before this existed — that is what makes this safe to
  ship without a live routing experiment.
- **The two fields are read together.** The fitted SLOPE beside a fixed cost of
  zero prices a peer below what either model says alone.
- **Never derive the fixed term from `latency_ms`.** Reading (2) is the whole
  reason it is measured; a ping says nothing about what a peer does to a chain.
- **A negative slope declines; a negative intercept clamps.** Cost falling with
  width is not a peer behaviour, it is one stall on a narrow segment. An
  intercept below zero is noise around a quantity that cannot be negative, and
  declining there would make this fire only for expensive peers — a bias, not a
  safeguard.
- **The fit expires with the ranking figure** (`RANKING_STALE_AFTER`), or a peer
  that fell out of rotation keeps its price for ever. That is the routing
  ratchet, and it falls hardest on modest hardware.

### Still open

**The precondition IS met on the live fleet** — checked rather than assumed,
2026-09-21, from the release node's own log: **12 distinct models** routed
(22 to 48 layers) and chains of 1, 2, 3, 4 and 7 segments, so a peer in regular
use is given several different widths. A fit that could never identify would
make this inert, which is the failure mode to rule out first.

**Not yet confirmed on a live multi-node run.** It is guarded by unit tests
built from the measured numbers, and a null control — reverting the one line in
`vertex_cost` turns
`a_narrow_segment_on_an_expensive_peer_is_priced_near_its_full_cost` red. What
is missing is a field reading: `observed_fixed_ms_per_visit` is on the candidate
DIAG line and on `/api/admin/peers` as `decode_fixed_ms_per_visit` precisely so
the first live occurrence can be read off rather than inferred. **Check it is
ever non-null before claiming this changed anything** — a fit that never
identifies is inert, and this repo has shipped an inert instrument twice.

**From the rules file (moved 2026-10-02):**

`PeerSpeed::decode_terms` fits `segment_ms ≈ fixed + slope × layers` per peer,
and `NodeCandidate::observed_fixed_ms_per_visit` carries the fixed half into
`vertex_cost` as the per-visit term — the one already multiplied by
`ASSUMED_FORWARD_PASSES` when a segment is entered per token.

**Sizing a segment small does not make it cheap.** A peer given 2 of 32 layers
took a chain from 152 ms/token to 3768: the other 30 layers cost less than those
2 (gotcha #659). A purely proportional coefficient prices those 2 layers at a
sixteenth of the peer, which is precisely why the router put them there.

⚠ **The fixed cost is NOT the round trip and must never be derived from one.**
On this fleet a peer 1043 ms away is five times faster than one at 643. Ping
measures the path; this measures what the peer spends per visit.

**The fallback is the safety argument, not a detail.** The two terms are only
separable when a peer's samples span differing segment widths — a peer holding
one shard of one model never does — and `None` prices it exactly as before.
`observed_latency_ms_per_layer` and `observed_fixed_ms_per_visit` are read
TOGETHER: the slope beside a zero fixed cost prices a peer lower than either
model alone.

→ `docs/invariants/scheduling.md`

## A reply is priced for no more passes than it may take

**Rule (2026-10-18, FUTURE_WORK #3, #3 (b)).** `NodeCandidate::forward_passes` is the
number of passes every per-token cost is charged for: the mean length of this
model's recent replies (`daemon::state::reply_lengths::ReplyLengths::expected`,
the last 32 that finished, once there are 4), else `ASSUMED_FORWARD_PASSES`
(64); then capped by the request's `max_tokens` where that is smaller, and at
least 1. `publish_request_trace` records each completed reply's length (the one
writer); `gather_candidates` reads the expectation (the one reader). The
router sets the budget once, where its two prompt plans meet
(`distributed_exec`: `PromptPlan::with_reply_of_at_most`); `gather_candidates`
copies it onto every candidate; `vertex_cost` (compute, a per-token hop),
greedy (`estimated_cost_per_layer`, which spreads the load and the prompt over
the passes) and `note_route_prediction` (the logged `assumed_forward_passes`)
read it. The candidate log line prints `forward_passes`.

**What it replaced.** A flat 64 for every request. The cost model weighs terms
paid ONCE (reading the prompt, loading the model, a hand-off's one trip, the
prompt pass's transfer) against terms paid per token; at 64 passes a reply
allowed 16 tokens had its per-token side priced four times over, so a load or a
long prompt counted for a quarter of what it does.

**Measured** (live node, 2026-10-18 06:32-06:38Z, `~/swarmllm-3/collect.sh`:
qwen3.5-9b, xLAM-2-8B and Qwen3-30B-A3B at `max_tokens` 16/64/160, priced
splits through AU, IT and BE peers, `examples/split_cost_terms.py --rows`):

- **The six replies capped at 16 tokens** ran at a median 0.49 of their price
  (0.10-1.08). Re-priced at 16 passes (compute and per-token hops × 16/64; the
  prompt, the load and a hand-off's one trip unchanged) the same rows read 0.97
  (0.25-1.41).
- **A reply the request does not bound is still mis-priced by its length.** Two
  whole-model hand-offs measured per token (AU, BE) ran 2.49x and 2.62x their
  price at 160 tokens, 1.18x at 64 — the error is tokens ÷ 64 where the
  per-token figure is right. **Fixed the same day (#3 (b))**: the expectation is
  the model's own recent mean, so a model answering at 160 tokens is priced for
  160 passes. A request a caller bounds below that is still capped by its bound.
  Why the MEAN: a pass costs the same wherever it falls, so the expected cost of
  a reply is the cost of its expected length; a median would price a model with
  a few very long replies as if they never happened. It is the baseline the
  length predictors in the literature are measured against (S³, Jin et al.
  2023; "Response Length Perception", Zheng et al. 2023). What it leaves: the
  spread of ONE model's replies around its mean — a request has no length the
  router can read before it is written.
- The field (2026-10-06): `max_tokens=24` replies priced 16-19x their time.

**What a change must keep.**

- **Never raise the assumption from `max_tokens`.** It is a loose ceiling — 2048
  when the caller names none (`DEFAULT_REPLY_BUDGET`) — and charging a split
  2048 crossings would price every boundary out (the reason the constant was
  fixed in the first place). Only a MEASURED length raises it.
- **Only a finished reply is a sample.** `record` takes `Outcome::Ok` replies
  with at least one token; a failure stops early and would drag the mean down
  for the routes that failed most.
- **One reading.** A path that multiplies by the bare constant prices the same
  request two ways, and the gates that compare two prices (`RoutePrices`,
  `costs_more_than_staying_here`) then compare different units.
- **The once-per-request terms are not scaled.** A load or a prompt is paid
  once whatever the reply's length; the bound changes only how many times the
  per-token side is counted beside them.

Tests: `per_token_costs_are_charged_for_no_more_passes_than_the_reply_may_take`,
`a_short_reply_is_not_sent_to_wait_for_a_faster_peer_to_load` (a 16-token reply
stays on a slow local copy rather than wait for a 4x faster peer's load), and
`a_reply_allowed_sixteen_tokens_is_priced_for_sixteen_passes` (the budget
reaches the recorded prediction, and once the model has 4 replies of 200
tokens a request allowed 2,048 is priced for 200, one allowed 16 for 16) — all
three red with the bound off, the last red with the learned length off.
`reply_lengths`: `the_expectation_is_the_mean_of_recent_replies_once_there_are_enough`,
`only_the_most_recent_replies_count`.

## A split is priced for the rounds it takes, not the tokens it writes

**Rule (2026-10-19, FUTURE_WORK #3 (d)).** `NodeCandidate::rounds_entered(range,
num_layers)` is the single answer to "how many rounds are a split segment's
per-token costs charged for": `forward_passes()` divided — for a SEGMENT (a range
short of the whole model) in a chain this node DRIVES — by the tokens a round
of this model's recent splits yielded (`daemon::state::split_rounds::SplitRounds::
tokens_per_round`: the last 32 finished splits' tokens over their rounds, once
there are 4, never below 1), at least one round. `vertex_cost` (role
`DrivenFromHere`), greedy's spread of the once-per-request terms and the logged
`assumed_tokens_per_round` (`parallax::chain_tokens_per_round`) read it. A whole
model, and a chain its head leads, keep `forward_passes()`.

- **Counted where a round LEAVES, whichever path ran**: `RequestTrace::
  note_decode_round` at `forward_through_segments` past the prompt pass (the
  plain loop) and `forward_verify_through_segments` with any guesses (the
  drafter's rounds, the n-gram loop's hits and misses). A new path that sends
  through those is counted without knowing it.
- **A path whose forwards OVERLAP counts the round trips it WAITED** (2026-10-29):
  the flowing tree (`dsd_flow`) and the continuous stream (`dsd_stream`) count,
  when they end, their decode time over the least time any forward took to be
  answered, at most one a forward (`split_rounds::round_trips_waited` →
  `SharedState::note_decode_rounds`). Counted one a forward — as both were — the
  14B's flow over TH ↔ AU (~420 ms, 0.5B drafter, `fr/real_link_{finishhere,leaf369}`)
  read 0.44-0.63 tokens a round on prose and 1.1-1.7 on code: floored to 1, a
  split that ran at 7-9 and 17-22 tokens a second was priced at ~2, and since
  the figure is per model one such reply among four tree-round replies took
  their ~4 tokens a round to 1.7. Counted as waited, the same replies read
  3.2-4.3 and 7.7-10.3 (37-50 and 15-21 rounds at a least answer of 405-474 ms),
  which at the plan's round of hop + one pass (~450 ms) is the speed they ran
  at; with the far card saturated (this node 12 of 48 layers, answers queued up
  to 2.2 s, `fr/real_link_flow`) 1.6-1.8 and 4.3-4.4, so a queue is charged as
  rounds, not hidden. The LEAST answer, not a mean: a queued answer's time is
  the queue's, which the count must show. What a change must keep: the count is
  in the PRICE's unit (a round = one crossing + one pass), so read the plan's
  round cost against the least answer before changing either; and at most one
  round a forward, so no reply is priced slower than the old count priced it.
  The figure stays per model: a lopsided split's low yield prices an even split
  of the same model too (`split_rounds` module doc).
- **Recorded once per finished attempt** by `PipelineExecutor::execute`
  (`note_split_rounds`: tokens over rounds + 1, the prompt pass yielding the
  first token, so a plain loop reads exactly 1) — only for two or more segments,
  a round that ran HERE, and a request that may guess ahead.
- **One request-side question on both sides**: `pipeline::request_may_guess_ahead`
  (a guess-check path switched on — the drafter's or the n-gram loop's — no
  adapter or image, a model that can take a guess back). The router asks it
  into `PromptPlan::with_guessing_ahead`; the executor asks it before
  recording. A request it answers no for runs one round a token, and recording
  it would lower the figure for the requests it prices.

**What it replaced.** Every split was charged its compute and its hops once per
TOKEN. The guess-check paths cross once per ROUND — several tokens where the
guesses hold. On this node's #273/#275 rigs (2026-10-14/15, n=25) the splits with
a drafter ran in **0.39** of their price.

**Research.** vLLM reports exactly this figure as its "mean acceptance length"
(tokens per verification step, the bonus token included; `docs.vllm.ai` §
Per-Request Acceptance Metrics); SmartSpec (Liu et al. 2024, arXiv 2406.14066)
and SGLang's adaptive speculation steer by a running average of it. The
expectation is Leviathan et al.'s (1−α^(γ+1))/(1−α), here MEASURED rather than
derived from α and γ, so the n-gram loop, the stream and a γ that walks are all
covered by one number. SmartSpec's known trap is the one this leaves: a check of
several positions costs more than one step where a segment computes on a
processor (its batch time is linear in positions). The γ controller holds such
a check short and stops guessing where it does not pay
(`dsd_controller::best_gamma_for_check`), and the stream's overlap — counted as
whole rounds — errs the other way.

**Measured.** `examples/split_rig.sh repeat`, 2026-10-19 (`RIG_NETNS=1`, both nodes on the
processor: Qwen2.5-Coder-7B split A [shard 0] + B [the rest], the 0.5B drafter on A,
seven greedy asks of 120 tokens; `scratchpad/rounds3d.sh`). Every ask took the
drafter's stream and logged `decode_rounds=72 tokens=120` — 1.64 tokens a round —
and from the fifth ask (four to trust) the planner read
`split_tokens_per_round=Some(1.64)` and the line `assumed_tokens_per_round=1.64`.
Asks 5-7 (65.8 s each) were priced 85.4 s (0.77); priced a round a token at the
same 120-token reply length they would have been 133.4 s (0.49). What is left
is the price of a round (1.03 s priced, ~0.8 s taken), not their number.
**Why the count and not α:** the drafter's own figures (α 0.73, γ 2) give
Leviathan's 2.26 tokens a round — ~53 rounds — but the stream sent 72 chunks,
39 of them on guesses already refused, and on this rig the time followed the
chunks. A figure derived from α would have priced it ~30% low.

**What a change must keep.**

- **Measured, never derived.** A figure from α and γ prices the drafter's rounds
  and nothing else; the count is what every path actually did.
- **Never below one token a round.** A stream's chunk sent on guesses that were
  then refused overlapped another; it must not price a split above the one round
  a token it was priced at before.
- **Not for a whole model, not for a led chain.** A hand-off or a local run
  guesses inside one worker, timed as it does; a led chain's head guesses with
  what IT holds, which this node has never measured.
- **The prompt is not divided.** It is read once whatever the rounds.

Tests: `a_split_whose_rounds_carry_three_tokens_is_priced_for_a_third_of_the_rounds`
(compute and hops ÷ 3, the prompt unchanged, a request that may not guess ahead
priced as before, the prediction records 3.00),
`a_round_of_a_split_is_counted_where_it_leaves_and_the_prompt_pass_is_not`,
`a_finished_split_is_recorded_at_its_tokens_a_round_only_where_it_is_priced_so`
— each red with its part switched off (the divisor, the plain-step count, the
check count, the request gate, the split gate). `split_rounds`:
`a_models_splits_yield_their_tokens_over_their_rounds`,
`a_round_is_never_priced_at_less_than_a_token`, `only_the_most_recent_splits_count`.

## A chain its head will lead is priced as its head leading it

**Rule (2026-10-18, FUTURE_WORK #3 (a)).** `parallax::chain_is_led_by_its_head`
is the single answer to "is this chain priced as led by its head". It is
`remote_generate::delegation_eligible`'s shape — two or more segments, none of
them this node's, the head holding layer 0 and advertising `DELEGATED_SPLIT` —
for a request the hand-off takes (`remote_generate::request_may_be_led_by_its_head`,
the request half of `delegation_eligible`, which the router asks once into
`PromptPlan::with_head_leading`), plus two conditions of the cost model's own:
every peer forwards to the next (`PIPELINE_CHAIN`) and has a usable network
coordinate (`NodeCandidate::ring`). Such a chain costs:

- the head: handed the request ONCE, as a whole-model hand-off is
  (`ChainRole::LeadsTheChain` — its network once, the delegated figure for its
  compute);
- each later peer: its compute, plus its own per-visit work where measured (the
  fixed term less OUR round trip to it), entered from the peer before it
  (`ChainRole::InLedChain`);
- every hop, the tail's back to the head included: `parallax::led_hop_ms`, half
  the round trip the two coordinates predict, once per pass.

The search runs one led pass per head that may lead (`route_over`'s `run_pass(..,
Some(h))`: seeded from that head, through peers only, closed by the hop back),
beside the driven pass, and keeps the cheapest; `chain_cost_terms` prices a
chain the same way, so the gates and the logged `predicted_terms` read the price
the chain was chosen at.

**What it replaced.** Every peer segment priced as a round trip from HERE per
pass (`2 × latency_ms` unmeasured). The delegated split (#143) had moved the
loop to the head on 2026-10-02 and nothing told the cost model: a chain of
three peers 20-300 ms apart, 220-420 ms from here, was priced at the
requester's distance three times over.

**Research.** Petals prices the same graph the same way: an edge between two
servers is `rtt / 2` from the pings each server publishes to its next servers
(`next_pings`), plus a closing edge back to the client
(`client/routing/sequence_manager.py`, `_build_inference_graph` /
`_rtt_to_delay`, read 2026-10-18). Here the "client" of a led chain is its head.

**Measured** (live node, 2026-10-18; the working-tree build run AS the local
node for five requests, `~/swarmllm-3a/check.sh`; Qwen3-30B-A3B held only by
peers, so every plan was handed to the Italian head):

- The hop price, against the morning's warm three-peer chain IT → BE → AU (64
  tokens, 25,907 ms, ~400 ms a token): one-way hops plus compute ~357 ms a
  token; a round trip per hop would be ~670.
- The network term, same peers and passes: 57,856 ms (IT, BE, 64 passes) in
  the morning, 3,620 with this; IT, BE, AU read 114,304 at 64 passes then and
  6,506 at 16 now (driven, 16 passes would be ~28,600). The coordinates against measured round trips here, the same day:
  224 → 253, 274 → 282, 417 → 454, 557 → 453 ms; a 1 ms LAN peer reads 60.
- Totals, NOT yet right, for two other reasons this check exposed (FUTURE_WORK
  #3 (e), (f)): the first, cold request priced 89,807 ms and took 99,854
  (0.90); the warm repeats priced 84,863 / 90,273 and took 19,437 / 29,448,
  because each still carried 78-83 s of loading that never happened — this
  node's plan cut the model at layer 19, the head re-planned onto the ranges its
  peers' workers hold ([0,18) and [18,48)), and a peer's worker keys ranges
  exactly. Without that term the same two read 2,261 and 12,071: the compute
  of a led chain's peers is a stated-speed prior that is never corrected, since
  a delegated reply records no speed sample.

**What a change must keep.**

- **One predicate for the price.** A chain priced led by the search and driven
  by `chain_cost_terms` (or the reverse) is chosen at one price and gated at
  another — the gates (`RoutePrices`, the hand-off comparison) compare
  `chain_cost_ms` figures.
- **The request half is the hand-off's own.** `delegation_eligible` calls
  `request_may_be_led_by_its_head`; a condition added to the hand-off goes
  THERE, or a chain the hand-off refuses is priced as handed over.
- **The driven pass still starts from a head that may lead.** A chain through a
  peer the led pass cannot price (no coordinate, no chaining) must still be a
  route; excluding such heads left it none
  (`a_chain_the_led_pass_cannot_price_is_still_routed_driven_from_here`). A chain
  the led pass can price is never cheaper driven: by the triangle inequality on
  the coordinates, one-way legs round a ring cost no more than a round trip
  from here to each machine, so the cheaper price kept is the led one.
- **Unknown prices as before**: no coordinate, no feature, a request the
  hand-off would not take — the chain is priced driven from here, exactly as it
  was.
- ⚠ **Coordinates predict a near pair poorly** (the 60 ms LAN peer above): a led
  chain between close machines is priced high, never low. Do not ORDER near
  candidates by `led_hop_ms`; it prices, it does not rank the near half.

Tests: `a_chain_its_head_leads_is_priced_by_the_hops_between_its_peers` (and
every condition that prices it driven), `a_far_requester_takes_the_chain_its_peers_lead_over_a_slower_whole_copy`
(red with led pricing off), `a_led_peer_is_charged_its_own_visit_not_our_round_trip_to_it`
(red with the fixed term kept whole), `a_chain_the_led_pass_cannot_price_is_still_routed_driven_from_here`
(red with leading heads excluded from the driven pass).

## A reply whose machine failed mid-stream is continued

(2026-10-08, FUTURE_WORK #236; design `docs/plans/mid_reply_continuation.md`.) Once a
streamed reply has started, `should_retry_after` refuses a re-plan — a retry from the
prompt would show the reader the answer beginning again — so a peer that refused,
restarted or went silent mid-reply ended the reply with an error after part of it was
shown (report #005: 71 s in). Petals does not end it: `InferenceSession` replays a
failed server's input history to its replacement; vLLM recomputes a preempted sequence
from its tokens. Both resume from what was generated.

So the router resumes from what the READER RECEIVED, the one record that is exact:
`StreamingTokenTx` — the sender every emit site already goes through for TTFT —
appends each event's text to the trace once its send SUCCEEDED (a token that never
reached the reader is never resumed past; `try_send` on a full channel is the case),
kept out of every snapshot and dropped when the request finishes. `continuation_after`
decides: the re-plan's failure classes, plus a reply that lost tokens in transit, not a
peer's context limit (the continuation is longer), never once the caller left, the
budget is spent or `MAX_CONTINUATIONS` (2) is reached. The continued request carries the
text as `InferenceRequest::continuation` (`#[serde(skip)]`, local) and `max_tokens` less
what was sent; `PipelineExecutor::build_prompt_with_header` — every path's prompt —
appends it inside the opened assistant turn. A delegated split (the delegate renders
`messages`) and the in-process executor stand aside. The reply reported at the end is
everything the reader received, finalised once.

**This is not the takeover below.** That one moves a segment's KV state to a stand-in
mid-pipeline and refuses a partial replay; a continuation re-reads the whole
conversation from text on a fresh route, so it needs no retained state at all.

Rig `split_rig.sh continue` (TinyLlama, A holds the header, B and C hold the model, all
on the processor; the serving peer's worker is killed mid-reply): v0.3.230 — 32 chunks,
then an error, no continuation; main — killed after 27 chunks, continued on the other
peer 25 ms later, 300 chunks in all (`finish=length`: the budget held across both parts),
no error, the opening once, joined as "…evaporator to cool" + " a space. The compressor
is responsible". Tests: `a_reply_under_way_is_continued_only_when_another_route_could_answer`,
`the_token_channel_records_only_what_reached_the_reader`,
`a_continuation_follows_the_rendered_prompt_and_nothing_else_renders_it`, the delegated
split's case in `a_split_none_of_which_is_ours_is_led_by_its_head_and_only_that` (each
red with its half off). Not covered: the batched dispatch path, which has no re-plan of
any kind.

## A reply under way is never moved to a machine that cannot continue it

`distributed::failover_can_restore_state(sequence_num)` — true only on the
PROMPT PASS — is asked by `failover_segment` BEFORE it looks for a stand-in. A
reply already under way ends with `SegmentFailoverExhausted` carrying
`cannot_resume_message`, and the machines that just failed are barred for that
request id.

**Why.** `failover_segment` re-sends the current step and nothing else, and the
KV cache is keyed by `(layer range, request id)` — so a machine that has not
served this segment for this request holds nothing, and no path rebuilds it.
`split::executor` then derives `kv_offset` from the CACHE rather than from
`index_pos`, and the worker's decode arm has no check that the two agree. The
replacement therefore answers from the current token alone while the reply
carries on looking normal.

**Measured, so do not re-derive it** (`examples/failover_kv_probe.rs`,
llama-3.2-3b, P = probability of the token the healthy machine would have
chosen; every run carries a control holding the history, which reproduced the
healthy machine EXACTLY at cosine 1.000000):

| segment replaced | decode steps first | control | stand-in |
|---|---|---|---|
| 14 of 28 layers | 24 | 0.9966 | **0.0054** |
| 4 of 28 layers | 24 | 0.9966 | **0.1186** |
| 4 of 28 | 1 | 0.0522 | **0.0000** |

**The second half: giving the stand-in the state** (2026-09-09). The
coordinator drives every hop, so it already SEES each segment's input. Keeping
those inputs lets a replacement be replayed them; `state.retained_activations`
is where they are kept and `assemble_replay` builds the forward.

Measured with the same probe, extended with a replayed arm — the retained
inputs concatenated with the takeover step, sent as ONE forward at position 0,
which is what a fresh cache makes it:

| segment replaced | stand-in today | replayed | intact |
|---|---|---|---|
| 5 of 28 layers | 0.1186 | **0.9965** | 0.9966 |
| 14 of 28 | 0.0054 | **0.9963** | 0.9966 |
| 21 of 28 | 0.0163 | **0.9968** | 0.9966 |

The replayed cosine is 0.9997-0.9999 rather than the control's exact 1.000000.
That is accumulation ORDER — one wide prefill sums differently from a run of
single-position decodes — not the missing cache, which shows as cosine -0.08 to
0.61 in the arm beside it.

**Prior art, and it is the same design.** Petals handles the identical failure
identically: servers keep past K/V for their layers, the client keeps "past
inputs sent to a given pipeline stage", and on a disconnect it "can find another
server with that pipeline stage and use client-side cache to restore the server
state" — O(t) bytes in one round, recomputing only the failed stages rather than
re-running the pipeline (arXiv 2312.08361, Algorithm 3). Retaining the boundary
input is also the cheaper of the two things one could keep: ONE hidden vector
per position against the segment's `2 x layers x kv_dim` of KV — 12 KB vs 32 KB
per position for a 4-layer segment of llama-3.2-3b, widening linearly with
segment size, and costing no traffic until something actually fails.

**Four more things the replay half must keep.**

- **The history is keyed by LAYER RANGE, not by segment index.** A segment is
  its range and ranges do not move; an index does. Anything that inserts or
  removes a segment — which is what item 17's multi-node cover needs — shifts
  every index after it, and an index-keyed history would then be handed to the
  wrong segment: a replay assembled from another segment's inputs, rebuilding a
  cache that is plausible and wrong. Pinned by
  `a_history_belongs_to_a_layer_range_not_to_a_position`, whose two segments
  write DIFFERENT payloads on purpose — an earlier version had both write the
  same bytes and passed with the histories swapped.
- **A partial history is never replayed.** This is the whole design. A replay
  built from a history with a hole rebuilds a cache that is plausible and wrong,
  and nothing downstream can tell — the same invisibility as the defect above.
  Every way of losing a step marks the segment unrestorable rather than
  shortening the replay: the byte budget, a chained run whose middle the
  coordinator never saw, a tensor-parallel segment driven elsewhere, an
  unreadable header, or a step that does not continue the last.
  `restorable_history` proves contiguity from 0 against recorded spans.
- **The payload and the position move together.** A replay covers `0..=current`
  and is correct only at position 0; sent at the current position it rotates
  every position wrongly and stays fluent. Pinned by
  `a_replayed_failover_is_sent_from_the_position_it_was_assembled_for` — whose
  FIRST version could not fail, because `activations` is a suffix of
  `send_activations` and `contains` cannot tell them apart. Only the null
  control said so.
- **Segment 0 is excluded unless pre-embedded.** Its input is token ids, and
  `[1, seq]` ids are indistinguishable from a flat `[seq, hidden]` state by
  shape alone, so the span cannot be read. Excluded rather than guessed.
- **Retention is armed only where a standby covers the range.** A segment
  nothing can take over gains nothing from being restorable, and retaining it
  would spend the budget protecting the segments that can. It is also the bound
  that keeps this affordable, and the one place Petals is deliberately not
  followed — its client cache is unbounded and persists throughout inference;
  this node is a server for other people's traffic too.

Four things a change here must keep.

- **The prompt pass still fails over, and must.** There the stand-in is handed
  the whole prompt and builds its own cache; it is the case standbys exist for
  and the only one the existing failover tests exercise.
- **There is no safe early window.** Failing over one decode step in is WORSE,
  not gentler, because the missing state is the PROMPT rather than the decoded
  history. Do not add a "recent enough" exemption; the test is the WORK KIND.
- **Ending is not losing the reply.** `should_retry_after` retries this variant
  when a remote segment was involved, so a non-streamed request re-runs from the
  prompt on a fresh route — a correct whole answer beats a long one that is
  quietly wrong. Streamed, the retry is suppressed and the reader keeps what
  they were sent; otherwise `note_salvaged_reply` returns what was generated.
  Removing the blacklist would break this: the retry would re-learn the same
  holder and reproduce the failure.
- **The guard is on the CALL SITE, not the predicate.** The unit tests beside
  `failover_can_restore_state` and `cannot_resume_message` all still pass with
  the call removed from `failover_segment`, which is the one edit that
  reintroduces the defect —
  `a_reply_under_way_is_never_moved_to_a_machine_that_cannot_continue_it` in
  `tests/repo_consistency.rs` checks the call and its ORDERING, with a
  planted-violation self-test for both "deleted" and "present but too late".

**Confidently wrong is a real state, and margin will not catch it.** At one
decode step the stand-in's top-1 margin EXCEEDS the healthy machine's. Judge a
change here by P(the reference's token), never by confidence or entropy — and
never by raw-logit cosine, which shares a large frequency-prior component and
scored −0.076 on a case where the argmax agreed.

**Still open** (`docs/FUTURE_WORK.md` #17 and #18; since 2026-09-09 the replay half
above ships and only a composite cover, a chained middle and a tensor-parallel
segment remain unrestorable): mid-reply failover now
does not happen at all. Making it WORK needs the boundary activations retained
for segments that have a standby — and that, not more standbys, is what a
multi-node standby would need first.

**From the rules file (moved 2026-10-02):**

`distributed::failover_can_restore_state(sequence_num)` — true only on the
PROMPT PASS — is asked by `failover_segment` BEFORE it looks for a stand-in. A
reply already under way ends with `SegmentFailoverExhausted` carrying
`cannot_resume_message`, and the machines that just failed are barred for that
request id.

**Unless the stand-in can be given the state.** `state.retained_activations`
keeps what this node sent to each segment that a standby covers, and
`assemble_replay` concatenates that history with the takeover step into ONE
forward at position 0 — a prompt pass to a machine holding no cache, so no new
message type and nothing an older peer refuses. It is keyed by LAYER RANGE, never by segment index — a segment is its range and
ranges do not shift, so nothing that inserts a segment can hand one segment's
history to another. `restorable_history` answers
`None` unless it holds positions `0..index_pos` CONTIGUOUSLY, and every way of
losing a step marks the segment unrestorable rather than shortening the replay:
a partial replay rebuilds a cache that is plausible and wrong, which is the same
invisible failure the refusal exists to prevent. The replayed payload and
position 0 move together — sending the replay at the current position rotates
every position wrongly and is equally silent.

→ `docs/invariants/scheduling.md`

## A failed request hands back the work it had already done

`SharedState::salvaged_replies` holds what a request had generated when it
died, and `router::salvaged_reply_if_lost` is the single place it is handed to
the caller — called at the one point per dispatch path where the attempt is
definitively over, which is after the retry in `dispatch_single` and after the
sole attempt on the batched path.

**Why.** A 4m43s reply on a 14B, already decoding, was discarded outright when
its tail peer's connection dropped (report #028). On a streamed request the
client at least keeps the text it was sent; on a non-streaming one the caller
gets a 503 and every token is thrown away. Nothing was wrong with the error —
the peer really had gone — but "the request failed" and "there is nothing to
show for it" are two different claims, and only the first was true.

**The failure stays a failure.** `PipelineExecutor::execute` still returns the
`Err`; the salvage is recorded on the way past. So the log line, the peer
penalty, the trust update and the error broadcast in `execute_request` all fire
exactly as before, and nothing here can make a lost peer look healthy. The only
thing that changes is what the caller is handed at the very end.

**It is the last resort, never the first.** A retry that produces a COMPLETE
answer beats a truncated one, so `may_salvage` only records; the taking happens
after `should_retry_after` has had its turn. Reversing that order would trade a
whole answer for half of one on every retryable failure.

Four things a change here must keep.

- **An empty salvage is not a salvage.** `note_salvaged_reply` refuses one, and
  a failure with nothing recorded keeps its error — which carries the class, the
  hint and the peer attribution. Replacing that with a `200` carrying nothing is
  gotcha #433's lie pointing the other way, and it is the control test beside
  the positive one.
- **Streamed replies are excluded, and not only for taste.** The text has
  already reached the client, so the honest terminal event is the error it
  already gets; and `api::openai::streaming` treats "no finish event arrived" as
  "this path never streamed" and re-emits the whole content as one delta, so
  turning a streamed failure into an `Ok` would hand the reader the reply twice
  (gotcha #414).
- **The caller must be able to tell.** `inference::FINISH_REASON_INTERRUPTED`
  is `"error"`, which is vLLM's own value for this (`FinishReason::ERROR`,
  beside `ABORT`). The OpenAI schema defines no member meaning "the machinery
  gave up part-way", and reusing one that exists is the same lie in a new place:
  `"stop"` claims the model chose to end, `"length"` claims a limit was reached.
  **The Anthropic surface cannot pass it through** — that vocabulary has no
  member for an interrupted turn, and an undefined `stop_reason` was removed
  from it once already (gotcha #300) — so `map_finish_reason` gets an explicit
  arm to `max_tokens`, the only defined value meaning "incomplete, cut off".
  Without that arm the catch-all reports it as `end_turn`, which Anthropic
  defines as the turn completing naturally. `pause_turn` was considered and
  rejected: it instructs the caller to resend and continue, so a client obeying
  it would retry into the failure with nothing said.
- **Both attempts may salvage, and the longer one wins.** They describe the same
  prompt, so the reply that got further is strictly the more useful one — and
  the tie-break must be length, not arrival order, or a retry that dies early
  overwrites a first attempt that nearly finished.

**Still open, and deliberately not conflated with this** (`docs/FUTURE_WORK.md`
item 17): a standby still cannot be assembled from several nodes that cover a
segment's range between them, which is why that request had no redundancy to
fail over to in the first place. Salvage makes the loss partial; it does not
make the request survivable.

### The salvage was only on the path the fix was written in (2026-09-18, #88)

`execute_distributed` tries FIVE alternative generation paths before reaching
the decode loop the salvage lived in — `try_dsd_distributed`,
`try_speculative_distributed`, `try_ngram_only_distributed`,
`try_local_generate_fastpath`, `try_remote_generate_fastpath` — and each was
called with `?`. Each runs a complete decode of its own, so a failure part-way
through a reply left the function before the loop, the `Err` arms,
`may_salvage` and `note_salvaged_reply` had ever been reached.
`grep -c may_salvage` was **0** in `ngram_only_spec.rs`, `dsd.rs` and
`speculative.rs`. Observed on a probe: `try_ngram_only_distributed ELIGIBLE`,
~220 tokens over 40 s, failed, **no** "keeping the partial reply" line and no
"Pipeline failed and failover was unsuccessful" line either.

**And the n-gram path is the DEFAULT**, so the population was a node holding
nothing — every new user. The severity is still narrower than it looks and that
is worth keeping honest: `forward_through_segments` attempts segment failover
internally and the router retries a transient failure, so a lost reply needs
both of those to fail as well. What made it worth a round is who is on the path,
not how often it fires.

**Why a per-request accumulator rather than three more calls.** "Call the helper
in three more places" is the fix this entry is a counter-example to — a helper
nobody is *obliged* to call will eventually not be called, and the salvage had
just proved it about itself. The choke point could not assemble the reply
either: `execute_distributed` has no access to the tokens a spec path holds in
its own local. So the reply is accumulated as it is produced and the salvage is
a read:

- **`pipeline::PartialReply`** lives on the executor (a `std::sync::Mutex`, the
  same pattern and the same reason as `collected_logprobs`) and is filled by
  `emit_streaming_batch` / `emit_first_streaming_token`. Those are the one place
  all three coordinators turn accepted tokens into reply text, and
  `streamed_reply_text_goes_through_the_shared_emit_helpers` in
  `tests/repo_consistency.rs` already **obliges** a new coordinator in
  `src/inference/pipeline/` to use them. The obligation was already enforced;
  this borrows it.
- **`keeping_the_partial`** is the single reader, wrapping all five calls, so a
  sixth path added to that list inherits the salvage without knowing it exists.
- **The recording happens BEFORE `emit_streaming_batch`'s
  `token_tx`-is-`None` early return.** That return is what made the old code
  correct-looking and wrong: a non-streamed request is the only one a salvage
  is for, and it was exactly the one leaving that function having done nothing.

This is vLLM's shape, arrived at for the same reason: the accumulated output
lives in the engine's per-request state and an abort returns it, rather than
each decode strategy remembering to report what it had.

**What the accumulator stores, and why two things.** The TEXT is accumulated per
emitted batch, because the reader of a salvage has no decoder in hand — the same
granularity a streaming client receives, with the same edge (a multi-byte
character split across a batch boundary decodes to a replacement character).
The token IDS are kept beside it for two jobs a count cannot share: they are the
honest `completion_tokens` — the trace logged `tokens=0` for these failures
while ~220 had been generated, so it could not even say how much was lost — and
they are what `may_salvage` is asked about, unchanged.

**A salvaged reply is finalised.** The accumulated text has been through no
scrub at all, so `keeping_the_partial` calls `finalize_reply_text` with
`reply_stops` like every other reply source (gotcha #643, found in the same
files while researching this).

**Verified by removing the mechanism.** Restoring the pre-fix early return in
both emitters turns
`a_path_that_fails_part_way_still_hands_back_what_it_generated` red on "the
tokens it generated must be kept", while the three null controls — a streamed
request, a failure with nothing generated, and a success — stay green, which is
what says they are controls and not passengers.

**And verified on a real request, against the released binary.** Two throwaway
nodes with TinyLlama's two shards split one each, a non-streamed 200-token
request on the n-gram coordinator, and the tail peer **killed 14 s in** — report
#028's own scenario, with no standby to fail over to:

| | v0.3.187-alpha | fixed |
|---|---|---|
| HTTP | **500** | **200** |
| body | `{"error": {"message": "Inference error: spec verify segment 1: Peer departed…"}}` | 180 characters of the reply |
| `finish_reason` | — | `error` (`FINISH_REASON_INTERRUPTED`) |
| "keeping the partial reply" in the log | **0** | 1, then `salvaged_reply_if_lost` after the retry also failed |

The ordering shows in the log and is the part worth keeping: the choke point
records at 07:13:28, and the router hands it over nine minutes later only after
`should_retry_after` has had its turn and the second attempt has also died.

⚠ **Reproduction trap**: the n-gram path self-disables after one request —
`payoff_justifies_the_wire` lets an unknown figure through, this workload scores
106 against a bar of 130, and the figure is a per-process static. **A probe on
this path needs a freshly started coordinator**, not just a fresh request. The
first attempt at this probe silently measured `pipeline/distributed.rs`'s own
loop, which salvages correctly on .187 — so it read as "already fixed", and only
the `try_ngram_only_distributed ELIGIBLE` line in the log said otherwise. Assert
WHICH PATH a probe exercised before believing either result.

**From the rules file (moved 2026-10-02):**

`SharedState::salvaged_replies` holds what a request had generated when it
died, and `router::salvaged_reply_if_lost` is the single place it is handed to
the caller — called at the one point per dispatch path where the attempt is
definitively over, which is after the retry in `dispatch_single` and after the
sole attempt on the batched path.

**Recording it must not be something a path remembers.** The salvage shipped
inside `execute_distributed`'s own decode loop, and the FIVE alternative paths
it tries first — DSD, draft-model speculative, n-gram-only, local-generate,
remote-generate — were each called with `?`, so a failure part-way through a
reply propagated straight past it: ~220 tokens generated, request failed,
nothing kept (FUTURE_WORK #88). `grep -c may_salvage` was 0 in all three
speculative files. This entry is "a helper nobody is obliged to call will
eventually not be called" firing on the salvage itself.

So it is a READ, not a call: **`pipeline::PartialReply` on the executor is
filled by the shared emit helpers** — the one place those coordinators turn
accepted tokens into reply text, and a place
`streamed_reply_text_goes_through_the_shared_emit_helpers` already obliges a new
coordinator to use — and **`keeping_the_partial` is the one choke point that
reads it**, wrapping all five calls. Recording is what emitting IS, and a sixth
path inherits both. `emit_streaming_batch` records BEFORE its
`token_tx`-is-`None` early return: the non-streamed request is the only one a
salvage is for, and it was the one leaving there having done nothing.

A salvaged reply is finalised like any other (§ "A reply a PEER generated…") —
the accumulated text has been through no scrub at all.

→ `docs/invariants/scheduling.md`

## A peer advertises the memory it will HONOUR, not the memory it has

**`NodeCapability::memory_for_model_layers_mb` is the single answer to "how much
memory can this peer give a model's layers"**, and `ram_model_budget_mb` is the
figure a node without a graphics card puts behind it.

**Why.** `ram_available_mb` is `sysinfo`'s raw reading with no margin. The same
node's own admission sizes a swap-safe budget from total RAM and contribution
level — `total × 0.8 × (0.5/0.8)`, so **4096 MB on an 8 GB machine at the
default level**, about half what it advertises. The scheduler routed on the
first number and the peer enforced the second, so segments were offered and
refused on arrival: measured, one peer refused 46 layers and then 28 layers
1.7 s later, each costing a round trip before the refusal was known, inside a
request whose first token took 154 s (report #022). Nothing bars a peer that
refused for capacity from being re-offered work — the blacklist fires only for
missing-shard errors — so the re-plan met the same stale figure and made the
same mistake.

**The stated budget is tested BEFORE the card, and that ordering is the
point.** The figure is computed only on the branch where models load into
system memory, so its presence carries the placement decision rather than
merely a number. A node that HAS a card and has been told not to use it
(`inference.gpu_layers = 0`) still gossips that card, because the card is
really there — so asking about `gpu` first judged it by memory its models would
never occupy while it loaded every one of them into RAM. That ordering was
right only while the card was the only thing that answered.

**The warm exemption is bounded by the model, because it is broader than its
own justification.** Free memory excludes what is resident, so charging a warm
peer for weights again double-counts — but `peer_model_is_warm` answers about
the MODEL, not which of its layers are resident, so the exemption is applied to
every layer under consideration. Charged KV only, a short prompt makes the
divisor a few hundred KB: the field reported the same peer at **97 layers cold
and 14008 warm** for a 32-layer model (2026-09-11), which is the capacity bound
quietly ceasing to bound a warm peer at all. `max_hostable_layers` now takes the
model's layer count and clamps by it. That makes the figure honest and stops a
warm peer being effectively unbounded; it does NOT make the exemption correct,
and a peer warm for part of a model can still be credited with weights it has
not paid for. `a_warm_peers_bound_cannot_exceed_the_layers_the_model_has`
carries its own control — the same inputs without the ceiling still reproduce
the field's four-digit figure.

**Then fixed properly, the same day.** `NodeCapability::resident_layers` carries
what each node has LOADED — distinct from `hosted_shards`, which is disk —
summed from each live worker's charged segments, and **`PeerResidency` replaces
the boolean with three states**, because collapsing any two of them costs
something:

- `Layers(n)` exempts n layers and charges full weight beyond them. Those
  resident layers still pay THIS prompt's KV, which the peer has not paid for,
  so a prompt whose cache will not fit beside them reduces how many this request
  can use.
- `Cold` charges everything, as before.
- **`WarmAmountUnknown` keeps the old generous pricing**, and that third state is
  the whole reason this is safe to ship into a mixed-version swarm. Reading a
  silent peer as cold would charge full weights to every node on an older build
  and route around the machines best placed to answer — the additive-protocol
  rule's exact failure, and gotcha #329 from the other side.

Absence of a model from a REPORTING peer's list is also not proof it is cold:
gossip is up to 30 s old, so a model just loaded is missing from it, and the call
site falls back to the recency signal rather than charging in full.

Petals publishes the same fact for the same reason — its servers announce the
blocks they are actively serving, not the ones they could load
(<https://arxiv.org/pdf/2209.01188>).

`only_the_layers_a_peer_actually_holds_are_exempt_from_their_weight` is the
regression guard, verified by regressing the arm: with the old pricing a peer
holding 8 of 32 layers is credited with all 32 where the honest answer is 18.

Three further things a change here must keep. **The accessor owns the device
choice**: two callers were writing the `match &c.gpu` themselves, and a third
would have had to get it right again. **The graphics branch is otherwise
untouched** — free VRAM already excludes what is resident, which is the property
`already_warm` pricing depends on; the RAM budget has the same property because
it subtracts `ram_committed_mb`. And **unknown never excludes**: a node that has
stated neither falls back to `ram_available_mb` and behaves exactly as before,
which is what keeps a mixed-version swarm routable. The field is additive and
`#[serde(default)]` per the protocol rule; the meaning of the existing field is
deliberately NOT changed, because the peer list displays it as free memory and
that reading is legitimate and different.

**The offer is bounded by the worker's own reserve (2026-10-04, #219).** The
figure behind `ram_model_budget_mb` is `vram::live_headroom_mb` — 70 % of
available memory — while a processor worker admits a prompt against available
memory less `kv_budget::device_free_margin_bytes` (5 % of the machine, at least
256 MB). Two rules for one quantity: below about 3.3 margins of free memory the
offer exceeded what the worker honours, and `9594e1ff` offered every TinyLlama
layer, then refused a 43-token prompt with "0 MB available for conversations".
`live_headroom_mb` now takes the smaller of the two, the margin rounded UP to
whole MB (truncated, it over-offered by 1 MB on an 8 GB machine — the property
test found it). Petals states the rule plainly: a server announces
`cache_tokens_left` from the cache that admits (`server.py`,
`memory_cache.bytes_left // bytes_per_token`). Guard:
`a_node_never_offers_room_its_own_worker_would_refuse`, a sweep over 4-64 GB
machines against `budget_reconciled_with_device` itself — so a change to EITHER
side that breaks the pairing fails. It also refuses a RAM load that would leave
the worker under its margin (`a_nearly_full_machine_refuses_a_model_that_would_swap`
now refuses 1300 MB with 2 GB free on 16 GB): such a load could not have served
a single conversation.

## A hand-off is priced as the shape it will be given, not as the whole model

**`inference::scheduler::delegated_shape_cost_ms`** is the one answer to "what
does this request cost if that peer is given `layers_to_assign` of it". The
price gate (`costs_more_than_staying_here`), the line that logs the gate's
verdict, and `privacy_cost_ms` all go through it, so none of them can price a
peer differently from the others.

**Why the shape is the whole question.** `parallax::vertex_cost` exempts
exactly one shape from per-token network — a remote candidate covering the
WHOLE model, entered once for the entire request and decoding remotely. Every
other remote range is entered once per token and charged
`2 * latency * ASSUMED_FORWARD_PASSES`; the function's own comment says so.
So the two shapes a hand-off can take differ by a factor of the token count in
their network term, and "the whole model" is the cheap one.

Prompt privacy is auto-on whenever this node holds both ends, which makes
`boomerang_assignment` — peer gets `(1, n-1)` — the COMMON shape, not an edge
case. The gate priced `(0, num_layers)` regardless. Measured on the release
pair 2026-09-06 (gotcha #478): a processor-only node holding llama-3.2-1b
whole handed the middle to a card **496 ms away** and took **9.1 s to return
one token**, against a local processor decoding that model at 4.28 tok/s.

Three things a change here must keep.

- **The site that decides the shape passes the shape.** `DelegationInput`
  already carried `layers_to_assign`, documented as "how many layers the peer
  would ACTUALLY be given ... the middle for a boomerang" — the capacity term
  read it (gotcha #454) and the price term did not. A shape parameter that only
  some terms consult is worse than none, because the ones that ignore it look
  correct in a suite where every test passes the same shape.
- **A model too short to cut a middle from is priced whole**, because
  `delegated_layer_span` hands it over whole. The two must agree by
  construction, which is why both read `BOOMERANG_MIN_LAYERS`.
- **This must not become "never delegate".** The feature exists because a
  processor node beside an idle card is the failure being fixed. The same peer
  offered the WHOLE model is still taken — pinned by
  `the_same_peer_still_gets_the_whole_model_when_that_is_the_shape`, the
  control beside
  `a_boomerangs_middle_is_priced_as_the_middle_it_will_be_given`.

**The general rule**: a cost model parameterised by shape must be handed the
shape that will actually execute. Same class as gotcha #434 — there a decode
step was budgeted as a prefill; here a boomerang was budgeted as a delegation.
Ask of any price whether the thing being priced is the thing about to be done.

## Delegation asks the same capacity bound routing does, and the retry it promises must exist

Three defects reported from one live node on v0.3.153, all in the path that
hands a whole model — or a boomerang's middle — to a single peer.

**`inference::scheduler::delegation_target` gates on `max_hostable_layers`**,
the same bound `route_shortest_path` uses, checked against
`delegated_layer_span(num_layers, encrypted)` — every layer for a whole-model
hand-off, `num_layers - 2` for a boomerang, read by both the gate and
`boomerang_assignment` so the span checked is the span handed over.

**What it replaced.** The function had two accept branches and neither could
see this request. The processor-speed branch had no memory test whatsoever, so
anything clearing `2x` our processor won; and `boomerang_assignment` checked
only `covers()` — whether the peer HOLDS those layers, never whether it can run
them. Measured: a peer whose own `max_hostable_layers` read 2-15 throughout was
handed 34 of a 36-layer model, timed out at 156 s, and answered the immediate
retry with `CUDA_ERROR_OUT_OF_MEMORY` (gotcha #454). The bound was on the very
candidate being accepted.

The whole-model branch did have a check, priced at `ADMISSION_KV_CONTEXT` — a
fixed 4,096 tokens however long the prompt is. The same peer took a 29-token
request in 0.98 s, an 8,841-token one in 238 s, and returned **nothing at all**
for an ~18,000-token one across its full 600 s deadline; proportional scaling
predicts ~486 s, so it was not merely slow (gotcha #455). The code comment
defended the constant by naming the peer's runtime head-room check as the thing
that "refuses gracefully" past it. No refusal ever arrived.

`needed` survives as the discriminator between the two accept reasons — "has a
card worth preferring to our processor" against "is a measurably faster
processor" — and can no longer admit anything the bound refuses.

**And the gate is priced on the prompt** (2026-09-05).
`costs_more_than_staying_here` runs `parallax::vertex_cost` — the routing
search's own function — over the peer and over the local candidate, and refuses
a peer that prices above running the model here. Before this the gate and the
search could disagree about the same machine, and did: a reporter's Apple M4
(`est_tokens_per_sec` 14.82 against a local 6.46, the best figure on their
network) was fully delegated two ~11-12k-token prompts and took **5-6 minutes
to the first token, twice**, while the search had priced it at ~234 minutes of
prefill and avoided it.
**Why the old terms could not see it**: `est_tokens_per_sec` is
`bandwidth / 4.4 * efficiency`, a memory-bandwidth estimate of how fast a
machine WRITES tokens. Prefill is compute-bound and `vertex_cost`'s own comment
puts the hardware spread at ~55x on prefill against ~6x on decode — so an M4,
with excellent unified-memory bandwidth and ten cores doing the matmuls, looks
fast on the decode axis and is dreadful on the one that dominates a long
prompt. **A gate comparing machines needs the axis the WORK is on.**
Unknown still never excludes — no prompt length, no local candidate, or a peer
at the shared prior all leave it open, since refusing on missing information
strands a node beside a machine that may be faster. The first request produces
the measurement that stops the second; the report showed the same peer chosen
twice because nothing learned. Unknown
capacity still never excludes, per `max_hostable_layers`'s own contract.

**Both accept branches log the same fields.** The whole-model line carried
`peer_free_vram_mb` and the boomerang line carried none, so the log written to
explain "why this peer" omitted the one number that showed the mismatch.

**`router::should_retry_after` is the whole retry decision, as four terms.**
Single-peer delegation sets `standbys: vec![]` deliberately, with a comment
naming its safety net: *"the retry in `dispatch_single` re-routes, and this node
still holds every layer, so the request can always come home."* That retry fired
on two error classes and the error a departed peer actually produces —
`SegmentFailoverExhausted` — was in neither, so both of two concurrent requests
died with the local node in the same candidate list holding every layer, and no
retry line in the log (gotcha #456).

Three things a change here must keep:

- **The bar and the retry move together.** `failover_segment`'s exhaustion arm
  blacklists the failed node and every standby it tried, for this request id
  only. `is_transient_remote_failure`'s doc already states the principle —
  *"the blacklist is what makes the retry actually work"* — and without it the
  retry re-learns the same holders and reproduces the plan that just ran out of
  memory. Pinned by `an_exhausted_segment_bars_the_machines_that_just_failed_it`,
  which is itself pinned by planting the violation (#413).
- **Nothing is retried once text has reached the client.** A retry restarts
  generation from the prompt, so on a streamed reply the reader watches the
  answer begin a second time. `TraceSnapshot::ttft_ms` is stamped on the first
  event carrying real text — empty terminal events do not set it, a
  non-streaming request never does — so it is exactly "output has left this
  node". This binds the two pre-existing classes as well.
- **The condition stays a function.** It had four terms inline, and the missing
  one could not have been tested for while it lived in the `if`.

**The general rule.** A comment describing a mechanism in another module is a
claim, not a fact: grep that mechanism for the case being relied on. Two of
these three were exactly that shape, as were #437 (a doc naming a writer nothing
writes) and #451 (a guard whose input nothing fills).

**A decision that rejects a cheaper option names it.**
`scheduler::cheapest_whole_model_peer` is reported as `cheapest_peer` /
`cheapest_peer_cost_ms` on every arm that keeps a request on this node, and the
fast-path line also carries `candidates`, `local_runs_on_processor` and
`parallax_routing` so a reader can tell which of the three ways it got there.

The candidate list has always been logged with a cost per holder, and every arm
has always logged a reason — but no arm named the peer the reason was ABOUT. One
tester filed three reports in a day off a log showing a peer priced 55x cheaper
beside a local decision that never mentioned it, twice concluding there was a
penalty mechanism overriding cost. There is none: `delegation_target` is a yes/no
gate that runs before any pricing (#447 (iii), open), and
`pipeline_may_replace_processor_route` declines a chain whose remote peers are
priced from a prior rather than a measurement (#444). Both correct, both logged,
neither legible (gotcha #460). **A reason without its subject makes a competent
reader invent a mechanism.**

**Every long wait in a request's life has a cancel checkpoint — including the
load.** `inference::cancel`'s module doc named two waits and read as complete;
the third, `ModelProcessPool::get_or_spawn`, is the longest and had none, so a
client that gave up while its segment was loading cancelled nothing and the
request went on to claim a KV cache and prefill for nobody (gotcha #459). That
one is BRACKETED by `cancel::bail_if_cancelled` rather than wrapped in
`unless_cancelled`: the wrap stops a wait by dropping the future, and dropping a
load half-done abandons a spawning subprocess and a partly-registered worker —
and the model may be exactly what the next request wants. A new wait that can run
for minutes needs one of the two forms, and the choice between them is whether
the work can safely be abandoned mid-flight.

**A failover reproduces the forward the segment was given, and the local node is
run in-process.** `FailoverInput` carries everything `failover_segment` needs —
activations, `pre_embedded`, `generated_ids`, the vision embeddings, `is_last`,
and the originating failure — as a struct, so a field added to the wire forward
has to be decided about rather than defaulted to nothing by omission.

`find_standbys` sorts the LOCAL node first, deliberately: a node holding every
shard is the most reliable fallback there is, and its own comment says so.
`failover_segment` only knew how to dial, and the local node has no
`peer_id_bytes` — so the most-preferred standby was a guaranteed second failure,
and a request whose holder died with `CUDA_ERROR_OUT_OF_MEMORY` ended one line
later with `Network error: No peer_id_bytes for backup node`, with the machine
that could have answered sitting right there (gotcha #458). The main loop has run
local segments in-process since the beginning; failover never learned to.

Three things a change must keep. **A standby that cannot be addressed is that
standby's failure, not the request's** — the `None` arm continues to the next
one, where returning `Err` ended the whole failover on the first unaddressable
entry. **The local attempt is subject to the same rule**: if running here fails,
the next standby is still tried. And **nothing is defaulted by omission** — the
three fields that had been hardcoded to nothing (`pre_embedded`,
`generated_ids`, `vision_embeddings`) each silently changed the answer rather
than failing, which is why none of them was ever reported.

**A gossiped figure is a snapshot, and a decision made between two of them must
remember itself.** `SharedState::peer_vram_commitments` holds
`request_id -> [(peer, MB)]` for work this node has scheduled onto peers and not
yet seen reflected in their capability broadcast, which arrives every 30 s on a
small swarm. `gather_candidates` subtracts it ONCE, so both consumers — the
advertised `gpu_vram_available_mb` and `max_hostable_layers` — inherit it.

Measured live: two requests 3 ms apart, both accepted whole onto one peer
against the identical `peer_free_vram_mb=Some(4598)`; sixteen seconds later one
died with `CUDA_ERROR_OUT_OF_MEMORY` inside `mlp` while the other kept decoding
on that peer, and the loser resent alone afterwards completed in 62 s
(gotcha #457).

Three things a change must keep. **The charge is what the bound weighs** — cold
peers pay weights plus this prompt's KV, warm peers pay the KV alone — so the
reservation and the capacity bound cannot come to describe different quantities.
**It is recorded on the EXECUTING path** (`assemble_awaiting_dht`), never in
`assemble_pipeline_for`, which the dashboard also calls to preview a route: a
preview that booked memory would never release it, since nothing calls
`release_request_state` for a request that does not exist. And **the request
being scheduled is excluded from its own total**, or a re-assembly charges
itself twice and routes around memory it reserved for nobody but itself.

This is not a second accountant for the peer's memory — the peer owns that, and
its own admission remains the backstop. It makes this node's estimate honest
about the commitments this node has itself made.

## A cap sized in units of the WORK is a ceiling on the product

**`inference::tensor_util::bytes_to_tensor` bounds its allocation by the
PAYLOAD, and deliberately has no ceiling on the element count.** The declared
shape is compared against the bytes actually present — `num_elements * 4` for
f32, `quant::q8_0_byte_len_checked` for Q8_0 — before `Vec::with_capacity` is
reached. That caps the allocation at roughly one message the transport already
accepted (`MAX_ACTIVATION_SIZE` 128 MB on the wire, `MAX_PAYLOAD` 512 MB over
worker IPC) and it is exact.

**What it replaced, and why the replacement is not a weakening.** A March 2026
hardening pass added `MAX_TENSOR_ELEMENTS = 32 * 1024 * 1024`, and it was right
about the hazard: the count comes off the wire, so a twelve-byte message
declaring a billion elements reserves 4 GB before the first bounds check. But
an element count is not a memory bound — it is a bound on the WORK. A hidden
state is `positions × hidden_dim` elements, so 32 M is **exactly 8192 positions
at hidden 4096** and only 4096 at the 8192-wide hidden of a 70 B. Every prompt
past that was unroutable across nodes, on every model, for six months, with a
message (`Tensor too large: 43876352 elements`) that named no model, no shape
and no prompt. Reported from the field on v0.3.153 — a 32-layer 8 B split over
two nodes, an 11.2 k-token agent prompt, four identical failures (gotcha #451).
The element count varying with prompt length was the whole tell.

**The rule to carry.** When adding a limit to something a peer declares, ask
what the EXACT bound is before reaching for a round number — it is usually the
input you are already holding. A constant that happens to be large enough today
is a product limit nobody chose, and it will be discovered by a user rather than
by a test. And when a refusal is a hard wall, its message must name the shape it
refused, or the person who hits it cannot tell a policy from a bug.

**From the rules file (moved 2026-10-02):**

**`inference::tensor_util::bytes_to_tensor` bounds its allocation by the
PAYLOAD, and deliberately has no ceiling on the element count.** The declared
shape is compared against the bytes actually present — `num_elements * 4` for
f32, `quant::q8_0_byte_len_checked` for Q8_0 — before `Vec::with_capacity` is
reached. That caps the allocation at roughly one message the transport already
accepted (`MAX_ACTIVATION_SIZE` 128 MB on the wire, `MAX_PAYLOAD` 512 MB over
worker IPC) and it is exact.

→ `docs/invariants/scheduling.md`

## A model's geometry is learned in one place, and unknown must not be silent

**`SharedState::gguf_meta_for` is the only read of `gguf_meta`**, and it learns
the geometry from the local `gguf_header.bin` on a miss.
`the_model_geometry_is_read_through_one_accessor` in
`tests/repo_consistency.rs` fails the build on a bare `gguf_meta.get(`.

**Why.** The map was filled at startup, by the admin HuggingFace shard download
and by local manifest generation — and by nothing at all on the path a model
takes when its shards arrive from the swarm while the daemon runs. Its one
reader is `gather_candidates`, which uses it to charge a peer for what THIS
prompt's KV cache costs per layer (`max_hostable_layers`, the #447 fix), and
that bound treats an absent geometry as *charge nothing*. So the bound was inert
on precisely the case it was written for — a model being distributed for the
first time — and a warm 6 GB card was handed 28 layers of an 11.2 k-token prompt
on a release that already contained the fix (gotcha #451).

Three things a change must keep. The accessor reads only an EXISTING header
(one `exists()` on a miss); materialising a header out of `shard_000` copies
megabytes and belongs on the startup and shard-landing paths, which call
`ensure_gguf_header` first. `check_and_load_model` — the choke point every
shard landing funnels through — warms it, so the routing path does not normally
pay the parse. And **`None` still means unknown, never zero**: a coordinator
holding none of a model's shards genuinely cannot know its shape, and the
capacity bound must keep declining to judge rather than charging nothing while
pretending to.

**The general shape**: a guard whose input is "unknown → do not apply" is
worthless until you check that something fills that input on the path the guard
exists for. Grep for the writer, and check it runs where the reader runs.

**From the rules file (moved 2026-10-02):**

**`SharedState::gguf_meta_for` is the only read of `gguf_meta`**, and it learns
the geometry from the local `gguf_header.bin` on a miss.
`the_model_geometry_is_read_through_one_accessor` in
`tests/repo_consistency.rs` fails the build on a bare `gguf_meta.get(`.

⚠ **"Local" is the catch: a coordinator routing a model it holds no part of has
no geometry, so `max_hostable_layers` charged peers NOTHING for the prompt's KV
cache** — weights-only, and on the `WarmAmountUnknown` branch no bound at all.
That is gotcha #447's mechanism (a warm 6 GB card handed 24 layers of an
8,111-token prompt, dead in attention 22 s in with no standby), and the
second half of the 2026-09-21 field report.

**`SharedState::ensure_model_geometry` closes it, called from
`assemble_awaiting_dht` BEFORE the plan** — `assemble_pipeline_for` is
synchronous and this is a fetch. Guard:
`a_route_learns_the_model_geometry_before_it_prices_peer_memory`, because both
orderings compile and the wrong one is silent.

- **It adds no fetch that was not already happening.** Every distributed
  coordinator's `extract_model_cache` pulls the same header into the same
  directory, so geometry already self-healed after one request — **only the
  FIRST request per model was ever exposed.** This moves that fetch ahead of
  the plan that needed it.
- **Best effort, bounded by a TOTAL timeout** — the exception to § Timeouts,
  justified because `probe_gguf_file` retries on `NETWORK_RETRY_DELAYS`
  (~155 s), no router may wait that long, and the work has a known small size
  (6-9 MB, tokenizer-dominated). **Timing out costs nothing beyond the old
  behaviour.** A failure sets a cooldown: "HuggingFace has no such file" does
  not change between two requests a second apart.
- ⚠ **The admin `pipeline_plan` preview must NOT warm.** It passes
  `prompt_tokens: None`, so its KV term is `(0, 0)` whatever the geometry, and
  it fires once per visible model card on every shard-total change — warming
  there would put a HuggingFace fetch on a dashboard refresh.

→ `docs/invariants/scheduling.md`

## A stand-in may be SEVERAL nodes, and the segment count is then read live

**`scheduler::standby_cover_for(standbys, range, tried)` is the single answer to
"what could take this layer range over"** (2026-09-19, `docs/FUTURE_WORK.md`
#17). It returns one standby holding all of the range — always preferred, and
returned as a ONE-ELEMENT cover so the pre-existing path is unchanged down to
the node chosen — or a greedy tiling from the range's start, taking at each step
the part that reaches FURTHEST past the frontier. Fewest hops, because each hop
is a network round trip on the token path.

**Why it exists.** A stand-in had to hold the whole of a failed segment by
itself. On a swarm of many small holders that is often nobody, so a segment the
swarm could collectively have replaced had none at all and the reply ended —
report #028's residual, left open because item 18's shape 2 had to land first.

**The refusals are the substance.** A cover with a HOLE, or one that stops short
of the end, answers `None`. A partial tiling would run the reply through layers
nobody executed, and nothing downstream could detect it: the coordinator
receives a well-formed activation of the right shape either way. That is the
same class of silent wrongness `failover_can_restore_state` refuses a replay
for, and the reason the greedy loop returns `None` rather than its best effort.

**Restricted to the prompt pass, and not casually liftable.** Mid-reply a
stand-in must be replayed the segment's retained input history, and
`retained_activations` holds only what the COORDINATOR sent — i.e. the input to
the whole range. The second part of a composite is fed by the first part's
OUTPUT, which never passed through the coordinator and was never retained. There
is nothing to replay onto it, so lifting this needs a new retention scheme, not
a relaxed condition.

**The splice, and the hot-path rule it forces.** A composite must persist into
every LATER decode step, not just the forward that failed over — so the
assignment genuinely changes shape and `install_takeover` splices N segments in
at the failed index. (A side map was considered and does not work for exactly
this reason.) It splices only AFTER the first part has answered, so a cover that
cannot be reached leaves the assignment untouched and the caller tries the next
one; splice-then-unwind would leave a half-installed chain the first time an
unwind was missed.

`forward_through_segments_inner` therefore reads `self.assignment.segments.len()`
LIVE, in both the loop bound and `is_last`. It used to cache it once before the
loop, which was correct only because nothing spliced. With a splice a cached
count leaves the spliced tail unrun, and — the silent half — makes `is_last`
name a middle segment. **`is_last` decides which segment SAMPLES**, so an
off-by-one there ends the pipeline early and the reply is quietly not the
model's. Only the forward that failed over could ever see the stale value, since
the function re-enters per token; that is precisely the reading that survives
casual testing. Guards:
`the_pipelines_segment_count_is_never_cached_across_the_forward_loop` (verified
by re-introducing the cached binding in the real source) and
`the_segment_count_guard_catches_a_cached_length`.

`segments_without_standby` asks through the cover too, so a plan cannot report a
composite-backed segment as bare — gotcha #451's shape, a count contradicting
what failover then does with the same plan.

**Run live 2026-09-25 (`examples/split_rig.sh failover`), and the first run
found what the unit tests and the guard could not.** Four CPU nodes on
llama-3.2-3b: A holds part 0 and coordinates, B holds the rest, C all of B's
range but the last part, D the last part — so B's segment has no single
standby. B's worker is killed the moment its prompt pass starts.

1. **The takeover of a LAST segment returned its first part's output as the
   answer.** The guard pins the loop bound and `is_last`; all three callers of
   `failover_segment` returned BEFORE the loop re-read anything, on an
   `is_last`/`run_is_last` computed before the splice. C's hidden states for
   layers 3-21 came back as the pipeline's result, D never saw the prompt, and
   the next token failed on D with "no longer holds the conversation" — the
   #93 check turning a would-be wrong reply into a refusal, then the router's
   retry through B. **A guard on the loop does not cover an early return out of
   it.** `failover_segment` now returns `Takeover::{Finished, Continue}`, so
   the function that splices is the one that says whether the pipeline is
   done. Test:
   `a_failed_last_segment_taken_over_by_several_nodes_runs_every_part_before_answering`,
   red with the old meaning restored.
2. **That refusal cost 290 s, because it went to the wrong node** — see
   `docs/invariants/network.md` and gotcha #707: four of `layer_forward.rs`'s
   seven reply paths answered the previous hop of a chain.

After both fixes: the takeover is logged, the router does not retry, the prompt
pass runs through both parts in 4.4 s including D's cold load, decode continues
at ~130 ms a token, and the reply — scored against llama.cpp by
`examples/score_against_reference.py` — ranks exactly like a same-topology run
with no failover (109 of 120 tokens at the reference's rank 1 in each). A
broken cache state would not: `failover_kv_probe.rs` measured the right token's
probability at 0.005 for a stand-in missing half the context. **Byte-equality
with the control is the wrong test** — two greedy runs of one topology split at
the same near-tie (`docs/FUTURE_WORK.md` #106 carries that lead).

⚠ Still not run live: a composite in the MIDDLE of a pipeline (unit-tested by
`a_takeover_installs_one_segment_or_the_whole_cover_in_its_place`; the rig's
shape fails the last segment), and a mid-reply composite, which is not offered
by design.

**From the rules file (moved 2026-10-02):**

**`scheduler::standby_cover_for` is the single answer to "what could take this
layer range over"** — one standby holding all of it (always preferred, returned
as a one-element cover) or several that tile it between them. A cover with a
HOLE, or one short of the end, answers `None`: a partial tiling runs the reply
through layers nobody executed, which nothing downstream can detect. Ask through
it, not `standby_covers`, or a plan reports a composite-backed segment as bare.

**Offered only on the prompt pass.** Mid-reply a stand-in must be replayed the
segment's retained input history, and part 2's input is part 1's OUTPUT, which
never passed through the coordinator and was never retained. Lifting that needs
a retention scheme that does not exist.

**`forward_through_segments_inner` therefore reads `segments.len()` LIVE**, in
the loop bound and in `is_last`. A takeover splices one segment into several, so
a count cached before the loop leaves the spliced tail unrun and — worse
silently — makes `is_last` name a middle segment, and `is_last` decides WHICH
SEGMENT SAMPLES. `install_takeover` splices only AFTER the first part answers,
so an unreachable cover leaves the assignment untouched. Guard:
`the_pipelines_segment_count_is_never_cached_across_the_forward_loop`.

**Whether a failover FINISHED the pipeline is `failover_segment`'s answer**
(`Takeover::{Finished, Continue}`), never the caller's `is_last` — that was
computed before the splice. The three callers returned part 1's hidden states
as the answer whenever the LAST segment was taken over by several nodes: the
loop guard cannot see an early return out of the loop. Found by the first live
run (`split_rig.sh failover`, gotcha #706); test
`a_failed_last_segment_taken_over_by_several_nodes_runs_every_part_before_answering`.

→ `docs/invariants/scheduling.md`

## A standby is a capacity commitment, not just a coverage claim

`scheduler::standby_has_room(max_hostable_layers, already_committed,
segment_layers)` is asked of every standby candidate, beside
`standby_covers`. The two are the same pair of questions #452 and #454 each had
to separate: `standby_covers` asks whether a node HOLDS the range,
`standby_has_room` whether it could RUN it.

**Why the running total.** `find_standbys` picks one standby per segment and
sorts the LOCAL node first — deliberately, since a node holding everything is
the most reliable fallback there is. Each pick was weighed against nothing, so a
16 GB processor-only Mac mini holding 12 of a 48-layer 14B was named standby for
all four remote segments; three failed over to it in turn and its worker was
killed mid-reply (gotcha #464). The plan logged `standbys=4` and it had the
capacity to be one — what HA practice calls "HA capable on paper".

`primary_layer_commitments` seeds the tally from the plan's primaries and each
chosen standby adds to it, so the fourth segment's search sees what the first
three committed. This is **decide-time accounting**, the same thing Kubernetes'
scheduler does with assumed pods: capacity is decremented when an assignment is
DECIDED, not when it is bound. Ours needs no cache because a plan is built
synchronously in one pass.

Four things a change must keep.

- **Primary duty counts.** A failover is precisely when a node runs its own
  segment and the one it stood in for, at the same time.
- **Standbys are charged at FULL weight**, never discounted by the chance of
  being used: `charge_additional_segment` never gives a range back, so a
  worker charged for a failed-over segment holds it for its life.
- **Unknown never excludes** — `max_hostable_layers`'s own contract, and what
  keeps a mixed-version swarm routable during a rollout.
- **This cannot lose a usable standby.** One candidate is picked per segment and
  one that does not fit is refused at `charge_additional_segment` with a 503, so
  before this the request died. Where nothing fits there was never a standby and
  `segments_without_standby` now reports that honestly instead of counting a
  fiction.

Note what the tally does NOT duplicate: `max_hostable_layers` already nets off
`peer_vram_commitments`, but that is CROSS-request and deliberately excludes the
request being scheduled — so within one plan it is always zero. This is the
missing within-plan piece, and it is not double-counting for a warm peer either,
since a warm peer's advertised free memory excludes its resident weights but not
the KV the new segment will claim.

**Each node is charged in the unit of its own bound (2026-10-27, #354 (a)).**
`NodeCandidate::charge_in_bound` is what a range takes from `max_hostable_layers`,
read by both standby checks (the whole range and each part of a cover), by the
tally's commit, and by `primary_layer_commitments`, which seeds it. A peer's bound
is a TOTAL, so a range costs what it loads (`width_in_bound`, its ends' tables
included, #353). THIS node's bound is room to ADD (#95), so a range costs
`layers_charged` — nothing for a range its worker holds loaded — until its
conversation memory refused the prompt, when its bound is a total too (#263) and
it is charged the width. Before, every check weighed this node's room against the
range's width, and the tally was seeded with bare primary widths. A node with a
full card holding 0..13 loaded was never a standby for a peer's 0..13, though
running it adds nothing — the "prefer local" comment in `find_standbys` assumes it
can — and its own loaded primaries used up room they never took. The same seed
counted a peer's tail short of its output head: `a_peers_primary_is_committed_as_it_loads`
(a 70B tail, 73..80, committed 9 against a total of 10, so a 2-layer standby does
not fit), red with bare widths. `this_node_stands_in_for_a_range_it_holds_loaded`
(this node holding 0..13 loaded, room 7, primary on 13..20; a peer primary on
0..13): a standby for 0..13, red with width charging; its null control, the held
range cleared, is not. Both predate v0.3.255; neither was seen in a field log —
found by the v0.3.256 pre-release sweep. `layers_offered_at` reads the same figure
the same way (d5302e77).

## A wait on a peer's first answer allows for its load, on every path

**Rule.** `pipeline::LoadAllowance` is the single answer to "may this peer
first have to load the model before it can answer?" — cold unless the peer
answered a forward of the model within `PEER_MODEL_WARM_TTL_SECS` (15 min),
worth `COLD_MODEL_LOAD_ALLOWANCE_SECS` (240 s) per cold machine. Every wait on a
peer's FIRST answer asks it: `SegmentBudget::for_forward` (a driven split's
forward), and `remote_generate::first_token_timeout`, which takes it as a
REQUIRED argument — the whole-model hand-off (one segment), a delegated split
(every remote segment, summed: a prompt pass reaches each in turn) and the HTTP
forward to a pool peer (`resolver::PeerTarget::load`, asked where the peer is
chosen). Outside tests a `LoadAllowance` can only be built by asking about
peers (`none()` is `#[cfg(test)]`), so no caller can pass a bare duration.
The allowance is added ON TOP of the prompt's cap, as on the segment path.

**What it replaced.** The 2026-08-01 fix (archive § "The three follow-ups",
1) gave the allowance to the segment path only, for "a CPU peer took ~120 s to
load an 8B model and was cut off by a flat 120 s deadline". The hand-off — the
path most single-model requests take — kept a flat `FIRST_TOKEN_TIMEOUT`
(120 s + 0.5 s per prompt token) whose comment still claimed it covered "model
load". On 2026-10-07 00:23 UTC (v0.3.229, live node): Qwen 3.5 9B handed whole
to an Apple M4 peer that had restarted six minutes earlier (its measured latency
10-30 s while it settled) gave no first token in 132 s, was penalised, and the
re-plan had no other holder of layers 13-24 → 503; asked again 47 s later it
answered in 2.7 s. One invariant, three paths, one of them fixed.

**The wait ends when the peer leaves.** A serving node aborts a hand-off the
moment its last connection to the requester closes (no route back for tokens —
`connections::handle_connection_closed`), but the requester waited out its whole
budget, which with the allowance is minutes. The first-token wait is taken in
`PEER_PRESENCE_CHECK` (5 s) slices against an absolute deadline
(`first_token_wait`): a slice that ends with the peer out of
`connected_node_ids` fails at once as `PeerUnresponsive` ("disconnected before
its first token"), barred from the retry like any silent peer. Only the
first-token wait does this: closing the token channel from the network manager
instead would have turned a mid-reply disconnect into a truncated reply
returned as a normal `stop` (the loop treats a closed channel after tokens as
the end of the stream).

**Being generous is cheap.** An unreachable peer is still failed by the 10 s
ACK fast-fail; a vanished one by the presence check. The allowance only extends
the wait on a peer that is connected and silent — a busy machine loading, which
is the case it exists for.

**Verify** with `examples/cold_load_test.sh <server> <client>` (two nodes in a
private network namespace; the server's load delayed by
`SWARMLLM_FAULT_LOAD_DELAY_SECS`). A client predating the fix must fail the
`load` arm at ~132 s and sit out the `leave` arm for ~132 s, or the run cannot
see the fix.

## A candidate that must first load the model is charged for it

**Rule.** `parallax::vertex_cost` adds `cold_load_ms` =
`NodeCandidate::cold_load_ms_per_layer` × `layers_it_would_add(range)`: the
layers of the range the candidate does not hold, at its own load speed. Charged
ONCE and never multiplied by `expected_attempts` (a retry finds the model
loaded). Every price goes through `vertex_cost` — the search's vertices,
`chain_cost_ms`, the local processor price `local_ms` and the hand-off gate's
`delegated_shape_cost_ms` — so all of them see it.

**The greedy fallback was the price that did not (2026-10-10, FUTURE_WORK #10).**
Its `estimated_cost_per_layer` priced latency, compute and load, and nothing for
reading layers in — and greedy is not a corner case: with
`parallax_partial_ranges` off the search uses whole ranges only, so every split
whose holders OVERLAP has no chain and falls back (16 of 16 plans in 8 rig
runs of A 0-21 / peers 3-28 on Llama-3.2-3B logged "parallax: no reachable
source→sink path").
With a spare holding the warm peer's parts (`split_rig.sh splitcache SPARE=1`),
turn 2 went to the spare in 3 of 3 runs: it advertised 4.92 tok/s (6.35 ms a
layer) against the warm peer's MEASURED 7.08, and nothing charged it the load.
The stored prompt was left behind and turn 2 took 58-61 s against ~7. The
formula is now `NodeCandidate::cold_load_ms(range)` for both routers; greedy's
figure is per token, so it divides by covered layers and by
`ASSUMED_FORWARD_PASSES`, the factor the search multiplies compute by — the same
proportion between the two terms. Test
`greedy_keeps_a_segment_on_the_warm_holder_over_a_cold_spare` (the rig's figures;
it picks the spare with the term zeroed). The same rig on the fix: turn 2 stayed
on the warm peer and resumed from token 2,240 in 3 of 3 runs, 5.6-8.0 s.

**One residency reading.** `gather_candidates` sets the per-layer figure from the
same `PeerResidency` it bounds memory with: published ranges decide which
layers are held (`layers_it_would_add` applies the loader's own rule — a range
that strictly contains a held one re-reads all of it); `WarmAmountUnknown`
(seen serving it, nothing published) is 0, as it is "every layer exempt" for
memory; a count with no ranges covering the whole model is 0. A pricing that
read residency differently from the bound would let them disagree about the
same layers.

**The rate is measured, per node, by that node.** A worker times each load
inside `ensure_model_loaded` (not the process start or a card's first context —
FUTURE_WORK #212's reverted attempt recorded those) and reports
`WorkerMsg::ModelLoadTimed` (gated on `DAEMON_READS_LOAD_TIMING`, the IPC twin of a
feature bit); `process_pool::LoadRate` averages ms per GiB over loads of at
least 256 MiB; the health monitor advertises it as
`NodeCapability::model_load_ms_per_gib` (`#[serde(default)]`). Bytes are the
model's size × the share of layers loaded — the basis the planner prices on, so
the units cancel. ServerlessLLM (OSDI '24) schedules on each server's measured
loading speed for the same reason: it varies too much to assume. Measured from
this node's own log on 2026-10-07 (335 loads): ~1.3 s/GiB onto the card, ~4 s/GiB
onto the processor; the field's slow end is ~26 s/GiB (an 8B on a processor
peer, 2026-08-01) and over 25 s/GiB (a busy Mac just restarted, 2026-10-07).

**This node is charged too.** With only peers charged, a cold local route would
look free beside a cold peer — the direction #129 already erred in (a peer
predicted at 4.5 s took 46 s, loading included).

**Unknown is the prior, not zero.** `UNMEASURED_LOAD_MS_PER_GIB` (10 s/GiB) stands
in for a node that has reported no rate (an older build, or no load timed yet —
this node too, until its first). Zero would break the symmetry above as soon as
this node measures itself; the slowest figure ever seen would rule out every
cold peer on an older build. Like `UNKNOWN_COMPUTE_MS` it sits on the pessimistic
side of the measured spread, so an unknown never outranks a measured fast
loader (gotcha #734: know what stands in when the figure is missing).

**A measured rate outlives a restart (2026-10-15).** `LoadRate` lived in memory, so
every restart — and so every node at every release's auto-update — went back to
the prior and advertised nothing until its next load: "no load timed yet" was the
state of every node after every update, not a corner. Fitting #3's logged cost
terms on this node (2026-10-14, Qwen3-30B-A3B split with an Australian peer, our
node restarted before each rig arm) found the cold term the largest single error:
34 completed first requests at 0.52× their price, ~58 s charged for a local load
that took ~9-20 s. Now `daemon::state::load_rate` stores the rate (tree
`measured_here`, written on the health monitor's tick when it changed and once at
shutdown, beside #226's lifetime record) and `SharedState::new` seeds the average
from it (`LoadRate::seed` — never over a load already timed in this run); the
next timed load moves it as before. As TCP keeps a measured round trip for the
next connection (Linux `tcp_metrics`) instead of starting each from RFC 6298's
conservative 1 s. Tests: `a_restart_keeps_the_measured_load_rate` (red with the
restore off), `an_unchanged_rate_is_not_rewritten_and_a_newer_one_wins`,
`a_failed_write_is_retried`. Startup line: "This node loads a model at the speed
an earlier run measured…".
Every figure entering the average — a timed load or the stored seed — is bounded at
`LoadRate::MAX_MS_PER_GIB` (60 s/GiB, over twice the slowest load the fleet measured):
kept, one load the machine was suspended through priced and advertised this node's
cold loads for many loads after, now across restarts too (v0.3.239's pre-release
review). A stored figure past it is taken in bounded and written back so. Test
`an_absurd_rate_is_bounded_whether_stored_or_timed`.

**What a test must do.** A routing test that plans through `gather_candidates`
now sees real load charges: pin rates where the subject is something else
(`the_search_s_boomerang_says_what_privacy_costs` gives every node 1 ms/GiB —
at the prior, ~37 s of loading entered the plan without privacy too and the
privacy notice's ratio fell under its threshold, which is true of a first
request and not that test's subject).

## The units decide whether a forward is a prefill, not the byte count

**`inference::pipeline::local::PipelineExecutor::forward_is_prefill(activation_bytes, units)`**
is the single answer to "is this forward doing a prefill?", for the deadline
(`compute_segment_timeout`) and for the DIAG that reports it. `SegmentBudget`
carries the resolved verdict (`is_prefill()`) so the log cannot contradict the
budget it is describing.

**The work KIND decides first (2026-09-02, gotcha #434).** `work_kind_for
(sequence_num)` is authoritative — 0 is the prompt pass, everything later is a
single-token decode step — and a decode step is a decode step whatever it
carries. Segment 0 of a DECODE step is handed the sampled token as
`PromptBytes` (that is the unit the first segment takes), and while the units
alone decided, every decode step of a remote first segment was budgeted as a
PREFILL: 240 s for 16 layers, measured live against a silent peer with a
standby idle throughout, where the decode budget is 32 s. The decode
coefficient is per-layer and never reads the byte count, so a decode-kind
forward also uses the MEASURED basis regardless of units. The spec-verify
sites pass `WorkKind::Decode` with `PromptBytes` and rely on exactly this.

**Within the prompt pass, `ActivationUnits::PromptBytes` means prefill,
unconditionally.** Segment 0 of a non-pre-embedded pipeline is handed the
prompt itself; every later hop carries hidden states. A forward carrying the
prompt performs the whole prefill by construction, however short the prompt —
so the size test does not apply to it. Only `HiddenStates` may be classified
by size, because `PREFILL_ACTIVATION_THRESHOLD_BYTES` (100_000) is a
**hidden-state** scale: one token of hidden state is thousands of bytes, one
token of prompt is a few.

**Why it is a rule.** The units were already explicit, and already honoured on
the *measured* path — `for_forward` refuses to predict from the peer-speed
coefficient for `PromptBytes`, with a comment saying that feeding those into the
same average "would silently corrupt the estimate". It then fell through to a
fallback that made exactly that unit error: prompt bytes compared against the
hidden-state threshold answered "decode" for anything under ~100 KB of text
(~25k tokens), i.e. essentially every real prompt. The forward that does the
entire prefill got `DECODE_SECS_PER_LAYER = 2` rather than
`PREFILL_SECS_PER_LAYER = 15` — 32 s instead of 240 s at 16 layers. Measured on
the live swarm 2026-08-29 (gotcha #407): a 4728-token prompt, `activation_bytes
= 24045`, holder abandoned after 32 s of a job needing minutes; the standby
answered correctly and the request succeeded, so **nothing failed and nothing
alerted** — the whole cost was a wasted deadline inside a 176 s request.

The guard existed on the sophisticated path and was missing from the crude one
beneath it. When a value's units depend on which path produced it, every
consumer needs the units — the fallback included. A threshold is a comparison
against a scale, so it is a unit conversion wearing a constant's clothes.

**Do not re-derive the verdict at a call site**, and do not widen
`PREFILL_ACTIVATION_THRESHOLD_BYTES` to "cover" prompts: that would break the
`HiddenStates` classification it was chosen for. `SegmentBudget` remains
constructible only through `for_forward` for the same reason it always was.

## A peer's stated reason is the answer; do not substitute one of your own

Two helpers in `inference::pipeline` own what happens when a serving node
refuses a forward. Both exist because the correct handling was implemented on
the **verify** hops and missing on the **prefill** hops, in the same files, with
a comment on one of them explaining exactly why it mattered.

- **`peer_error_from_result`** — recovers the peer's failure from a completed
  `LayerResult`. A serving node reports why it refused in
  `finish_reason: NetworkFinishReason::Error(msg)`, and `LayerResult::error`
  leaves `token_ids` empty when it does, so a caller testing only
  `token_ids.is_empty()` throws the reason away. It is applied at the single
  choke point: `forward_through_segments` is now a thin wrapper over
  `forward_through_segments_inner`, because the inner function has SIX `Ok`
  return sites and checking each one is the mistake being fixed. The one path
  that does not go through it — `speculative.rs`'s prefill, which awaits
  `wait_for_result` directly — carries its own call and says so.
- **`failover_segment` LOOPS over standbys, and a standby's error is a failure
  of that standby, not the segment's output** (2026-09-02, gotcha #435). It
  used to return whatever the first standby sent; a refusal (out of memory, a
  missing shard) is an error `LayerResult` with EMPTY activations, and those
  were forwarded to the next segment, which failed them as `Tensor bytes too
  short` — an internal error blamed on a segment that was fine, seen live and
  independently reported by a tester the same day. Every node already tried is
  excluded from the standby search; exhaustion carries the last standby's
  stated reason (`exhausted_message`). Every consumer of a `LayerResult` must
  check `finish_reason` before using the payload.
- **`every_holder_would_refuse`** — asked BEFORE failing over. Deliberately
  narrow: only `Validation`, because that describes the REQUEST and every holder
  reproduces it identically. A missing shard or a dead worker says nothing about
  the next holder and must still fail over; narrowing this predicate too far
  would disable failover itself, which is why the negative control test lists
  five such errors by name.

**What was measured (2026-08-30, released v0.3.135).** One over-long prompt gave
three different answers depending on topology — `500 server_error` naming an
internal mechanism, `503 Segment failover exhausted` advising
`swarmllm get-model` for a prompt that is simply too long, and `400` after a
wasted round trip per peer — while the same prompt on a LOCAL model was always a
clean `400` naming the exact number of tokens to cut. Whose fault a mistake was
depended on which machine held the model: gotcha #304's shape, on a path #304's
fix did not reach. And because the class was flattened to `Inference`,
`failure_is_penalty_worthy` (which exempts `Validation`) docked the serving peer
for the caller's mistake. Gotcha #415.

**The rule to carry.** Before failing over, ask whether the next holder could
possibly answer differently — and never replace a reason you were given with one
you invented.

## `ModelProcessPool::serves_on_cpu` is the whole-model delegation precondition

(2026-09-02, gotcha #442) — "would this request run on our
processor": no usable card, told to use the processor, a build without
CUDA, or a card the model does not fit. It replaced
`is_cpu_bound_for_lack_of_vram` alone, whose doc read a node with NO card
as "working normally" — so a processor-only node holding every shard ran
the model itself with GPU peers idle on the same pool. Peer-side gates in
`delegation_target` are unchanged. The case it could not reach — a model no
single peer's card holds — is the priced comparison below.

## A node holding every layer that would run the model on its processor lets the priced search compete with its fast path

(2026-09-03, gotcha #444).
`assemble_pipeline_for` answers `serves_on_cpu` ONCE (a lazy `OnceCell`,
since it prices the model against the graphics budget and reads the header
for a model with no worker) and threads it into `gather_candidates`, which
prices the LOCAL candidate by the device the request would USE: processor
speed from measured bandwidth, the processor prefill prior, and — on a node
that has a card — no `observed_latency_ms_per_layer`, because that figure
is per node and was measured on whatever the card served for someone else.
When no whole-model peer qualifies (`delegation_target` → `None`),
`pipeline_may_beat_local` skips the fast path and `route_shortest_path`
runs; `pipeline_may_replace_processor_route` then keeps the request home
for an all-local chain or one whose remote segments include a peer priced
at `UNKNOWN_COMPUTE_MS` — **a route this node can price is never given up
for one it cannot**. Greedy never makes this call (nothing to compare), and
a parallax error falls back to the fast path, not greedy. The decision line
logs `local_processor_cost_ms` and `pipeline_cost_ms` (`parallax::
chain_cost_ms`) so the choice can be checked from a log.
**A split is priced as a split (2026-10-08, FUTURE_WORK #129's last residual).** "The device the
request would use" was read as one device: a model too big for the card was priced at the
processor's speed although the loader runs part of it on the card. A token's time is the sum
of its layers' times, each on its own device (llama.cpp's partial offload, `-ngl`, behaves the
same), so `split_tokens_per_sec` blends the two speeds harmonically by the card's share of the
layers — `ModelProcessPool::card_share_for_planning`: the resident worker's own split, else the
one a fresh worker would make. `PipelineScheduler::local_speed_off_the_card` is the one answer,
read by the local candidate here AND by the hand-off gate's baseline (`delegation_target`'s
`local_cpu_tokens_per_sec`), so a peer is not judged against a slower "here" than the search
uses. The prompt pass keeps the processor's prior (`has_gpu` false) — conservative for the
card's share of the prompt. Tests `a_split_is_priced_by_the_share_of_its_layers_on_each_device`,
`a_local_model_the_loader_splits_is_priced_with_the_cards_share` (red with the processor figure).
**Checked on a `--features cuda` build run as the live node** (`~/swarmllm-129b/verify129{c,d}.sh`,
safety kit, 0 driver events, Windows up 20 h, 7-8 peers on .230 including an RTX 3060 and an RTX 4050):
Qwen2.5-14B, whole here, which the loader runs as 22 of its 48 layers on the 8 GB card. The release
priced this node at the processor's 4.55 tok/s (7B reference; `local_processor_cost_ms=106500`), the
new build at 6.95 / 6.34 (98650 / 100075). Both stayed local: no peer had room for the 48 layers
(the 3060 offered 7, then 42). Forced local, warm: 80 tokens in 19.3-19.8 s, ~4.0 tok/s — against
~3.0 predicted from the processor figure and ~4.6 from the blend. Once the split worker was
RESIDENT, the release priced it at the card's whole 56 tok/s (`serves_on_cpu` reads "card" for a
worker holding card memory); `gather_candidates` now treats any split (`card_share_for_planning`)
as off the card, and the new build priced the warm requests at 6.34 too (test: the `&|| false` arm of
`a_local_model_the_loader_splits_is_priced_with_the_cards_share`, red with that half off;
`a_resident_split_worker_reports_its_card_share_to_the_planner`).
**A range that runs partly on a card is priced per device, and a card's machine memory is offered
too (2026-10-25).** The blend above is right for a WHOLE model; for a segment it is wrong twice.
Llama-3.3-70B Q4_K_M (80 layers, ~40 GB) with this PC holding layers 0-16 on a 6.6 GB card budget:
the local candidate read `has_gpu=false`, est 7.1-7.4 (the blend, ~8 of 80 layers on the card),
`max_hostable_layers` 8-12 (the card alone) and `fresh_run_layers=None` (the only split offered was
the whole model's, whose ~32 GB processor share never fits 16 GB), so the Mac mini in Italy (est
15.4, 241 ms) was planned layers 0-13. And no card machine could offer its RAM at all:
`memory_for_model_layers_mb` is the card's free memory, so the RTX 3060 in Australia (12 GB card,
`max_ram_mb = 24000`) advertised 2 layers, the `PeersAtCeiling` rung gave it 60, its spawn went to
the processor (card full) and refused 30,432 MB against 24,000. llama.cpp's RPC backend spreads a
model's weights and cache over EVERY device, card and processor, local and remote, in proportion to
free memory (`--tensor-split` over `ggml-rpc` devices) — a card machine's RAM is part of what it can
give a split. Three changes:
(1) a card node states **`NodeCapability::machine_room_beside_card_mb`**
(`vram::node_machine_room_beside_card_mb`: the RAM budget less what its pool committed, under the
live headroom — the figure its spawn's RAM admission weighs; `SWARMLLM_HYBRID_OFFLOAD=0` states
none) and `processor_tokens_per_sec_7b`, read only through `NodeCapability::processor_beside_card`;
`gather_candidates` adds `layers_beside_card` (room ÷ margin ÷ (processor-side weights + prompt KV
per layer)) to the card's bound, its face value and its published room, still under
`clamp_to_ceiling`, and records the card's part as `NodeCandidate::card_layers`. An older card node
states nothing and is planned exactly as before.
(2) **`NodeCandidate::layers_by_device`** puts a range's first `card_layers` on the card — the
loader's order (`split::hybrid`: `offset < gpu_layers`) — and the rest at
`processor_tokens_per_sec`: decode by `prior_ms_per_layer_over` (`parallax::DecodePrice::Prior`,
greedy and `vertex_cost` alike), the prompt by `prefill_ms_per_layer` at each device's speed-up. A
MEASURED figure is still taken as it is. The local candidate is `has_gpu` at the card's speed with
`card_layers = on_card` wherever the loader splits; `local_speed_off_the_card` keeps the blend for
the hand-off gate, which compares whole models.
(3) **`fresh_run_layers_for_planning` offers a SEGMENT card+processor run** where the whole model's
processor share does not fit: the widest run whose machine charge (`machine_charge` over that
run's `partial_gpu_layers` split) the RAM admission takes (`machine_would_take`), weighed at BOTH
ends of the model — the width is placed wherever this node holds layers, and the ends cost most:
the first carries the token table and what the loader keeps in the machine whatever the device
(#296), the last the output head, on the card, which pushes layers onto the processor (#342) —
found by bisection; offered only where it is wider than the card alone. A run wholly on the card
is still weighed for what the loader keeps in the machine.
Tests: `the_processor_beside_a_card_is_offered_only_by_a_card_node_that_states_it` (types),
`a_card_node_offers_its_processors_room_beside_the_card`,
`a_range_split_across_card_and_processor_is_priced_per_device` (the control, priced at the blend,
routes to the farther peer), `a_model_too_big_for_the_card_and_the_machine_is_offered_as_a_segment`
(its control, with RAM for the whole processor share, offers all 80).
**A warm peer's card layers are the ones its card runs (2026-10-27, #361).** (1) above recorded
the card's part as the card's bound, and for a peer with layers loaded that bound counts every
resident layer (`max_hostable_layers`' `PeerResidency::Layers` arm: resident + what the free
memory adds) — so a peer running a model across card and processor read as running all of it on
the card. The 02:08Z 10-27 candidate line: the RTX 3060 in Australia (12 GB card) holding
`[(23,80)]`, 57 layers of the 70B at ~0.53 GB each, read `card_layers=Some(59)`; others read 66,
70 and 76 over the day. Its card holds about 20. Where the prior priced it (no measurement by this
node, no published timing of its own), those 37 processor layers were priced at the card's speed.
The same blind spot was on this node's side: `card_share_for_planning` read the SPAWN's split, so
a range grown later — on the card alone, or split itself (#265) — never counted. Now ONE count:
**`WorkerHandle::layers_on_card`**, range by range as each was placed (`RangeCharge::split`: the
spawn's split for its own range and for one placed "as the spawn said", the growth's for a split
growth, none for a card-only growth), read by `card_share_for_planning` and published as
`ResidentModelLayers::card_layers` (`#[serde(default)]`, JSON gossip, no feature bit — the type's
rule). The planner bounds a warm peer's card by `PeerResidency::Layers(card_layers)` — the KV of
its processor layers is in system memory too — and every peer's card by its whole memory over the
layer's weights, which is all an older peer's announcement allows (24 for that card). What a peer
may be GIVEN is unchanged; only the speed its layers are priced at. Still a count, not a
placement: a range starting mid-way through what it holds is priced as if its first
`card_layers` were on the card. Petals prices a span from each server's measured throughput,
not a device split (research changed nothing here; our analogue is the peer's own timing floor,
`decode_us_per_layer`, which already outranks this prior). Tests:
`a_warm_peers_card_layers_are_the_ones_its_card_runs` (scheduler),
`a_workers_card_layers_count_every_range_as_it_was_placed` (pool).
**Why this is not `cbbed678` again**: that pass priced local layers at a
constant 10,000 — a penalty, not a price — so the search could not see the
LAN split that was best and sent a request abroad. Here the local figure is
the same measured one every peer advertises about itself, and every remote
hop is still charged per token, so a short prompt with only distant cards
stays home (pinned by `a_short_prompt_stays_on_the_processor_when_the_
cards_are_far_away`, whose near-peer arm proves the route exists).
**Under prompt privacy the router adds split points 1 and N−1 when the
local node holds the whole model.** Privacy is auto-on for a node holding
both ends, so the tester's shape is a boomerang across SEVERAL peers —
local(0,1), card, card, local(N−1,N) — and `route_shortest_path` only cut
ranges at shard boundaries, of which a node holding everything has none in
the interior. Added only for that topology, so every other encrypted route
is exactly what it was. Test it from `assemble_pipeline_for`, not from the
router: the first cut passed the router's tests and stayed local end to
end for exactly this reason.

## A peer's capacity for a prompt is weights PLUS that prompt's KV cache

(2026-09-03 evening, gotcha #447 follow-up). `scheduler::max_hostable_layers`
takes `prompt_kv_bytes_per_layer` — `kv_bytes_per_position_per_layer(gguf_meta,
on_gpu) × prompt_tokens`, the same arithmetic the worker charges at admission,
mirror included for a GQA model on a card — on top of the weights for a cold
peer and ALONE for a warm one (since #281 (4) through the header's per-layer
layout, a windowed layer at its window where the peer advertises
`features::KV_WINDOW_SLIDES` — `docs/invariants/inference.md` § "A windowed layer
is charged as its window"). A warm peer used to be uncapped ("it has already
paid for the weights"), which is true and was the whole story until the prompt
was 8,000 tokens: the #447 card was warm. Unknown prompt or geometry → 0, and
unknown never excludes. **The peer's own admission is the backstop**:
`model_worker::handle_forward` now runs `ensure_room_for_prompt` for the prompt
pass (`sequence_num == 0`, the work kind of #434) of a SEGMENT too, so an
over-committed peer refuses at token 0 with the 503 the coordinator fails over
from, instead of dying in attention 22 s in. A new admission check on the
`Generate` path has to be asked whether the segment path got it too.

## A peer's range is weighed as what it loads, its ends' tables included (2026-10-26, field report #039, #353)

**What it replaced.** Every bound a coordinator holds a peer to is a count of
the model's layers at the file's AVERAGE: `max_hostable_layers` and the
published room divide by `weight_bytes_per_layer` (the file over its layers),
the ceiling by `segment_cost_curve`'s slope, which keeps the tables "in its
weights, spread evenly" (#342's residual). That is exact for a whole model —
`delegation_target` and the whole-model disqualifier were always right — and
wrong for a part: a range ending the model loads the whole output head, one
starting it the whole token table, a middle one neither. Llama 3.3 70B Q4_K_M:
506.8 MB a layer on average, an 822 MB head (`output.weight` + norm), a 564 MB
token table. A Proxmox peer whose ceiling (its RAM cap at the reduced-hours
level, 4096 MB) held 7 average layers was planned 73-80 by a tester's node and
refused it up front: "needs about 4387 MB but this node's budget allows 4096
MB". The up-front question (#345/#349) caught it and the re-plan found 75-80
four seconds later, so the reply ran — the cost was a refusal and a re-plan on
every cold request, and a refusal that read as a failure in the tester's log.

**How others do it.** llama.cpp prices its output layer as a unit of its own,
always whole on the last device (`common/fit.cpp`: "the last device has the
output layer, which cannot be a partial layer"; `src/llama-model.cpp` counts
it as one layer more than the model's, `max_offloadable_layers = n_layer + 1`),
and measures each device's memory rather than averaging. Petals keeps the
embeddings and the head on the client, so its servers' blocks are uniform.

**The rule.** `process_pool::RangeWeights` (one per PEER candidate, built from
the header only) gives `excess_layers(range, held)`: what loading the range
really adds — `bytes_added_by`, the byte twin of `layers_added_by` over
`ModelWeights` (the loader's own `segment_tensor` classification, one walk of
the table; `segment_weights` now reads it too) — less `layers × average`, over
one layer's price (average + this prompt's KV). About +1.4 layers for a 70B
tail, +0.5 for its first range, −0.03 a layer in its middle, 0 for the whole
model. It enters the ONE charge the search compares with a bound
(`NodeCandidate::capacity_charge`), and `width_in_bound` (rounded up, beside the
ranges the peer holds — see the second correction below) every reader of a TOTAL: parallax's capacity split points (a tail's point
moves in by its head, or the range that fits is not even a vertex), greedy's
run end (`furthest_end_in_bound`), both standby checks,
`plan_exceeds_offered_memory`; `record_peer_commitments` books the real bytes.
Auto-manage's chain planner weighs each run the same way
(`near_chain::Planner::reach`, the ceiling's slope as the price). This node is
untouched: its loader prices its own ranges (#342). A peer counted warm
without naming its ranges keeps the average — its bound divides by the KV
alone, and nothing says which layers it paid for.

**Measured** (unit, the 70B's shape — 80 × 500 MB, 564 + 822 MB tables; a
peer of 76 average layers beside a faster one holding 70-80 at 7): planned
73-80 at the average, 75-80 weighed; with 73 a declared boundary and the
charge removed, 73-80 again (`a_range_ending_the_model_is_weighed_with_its_output_head`
goes red). Chain: a machine at 7 is planned 73-80 unweighed, every run within
its machine's ceiling weighed (`a_chains_runs_are_weighed_with_the_tables_at_the_models_ends`).

**What a change must keep.** A new reader of a peer's bound weighs a range by
`capacity_charge` (a bound on NEW layers) or `width_in_bound` (a total), never
by its width; a new bound stays in average layers or changes `RangeWeights`'
average with it. A/B `SWARMLLM_PEER_RANGE_WEIGHTS=average` (`ModelWeights::for_planning`).
**Still at the average**: carrying's max-flow count and the ceilings' sum a
chain is judged carriable by (a layer and a half of slack in a count that only
decides whether to plan).

⚠ **Corrected 2026-10-26.** This section first said a refusal's
`layers_that_fit` stayed at the average "on the cautious side". It was not
cautious: that count is of REAL layers, and once the charge was weighed a peer
that said 45 light layers would load was given 47 heavier ones under it — the
v0.3.255 field check failed on exactly that. § "A peer's count of what fits is
weighed in its own unit" below.

⚠ **Corrected 2026-10-26, second** (pre-release sweep of v0.3.256, code reading).
`width_in_bound` was first written cold-shaped — `excess_layers(range, &[])`, as if
the peer held nothing. A cold peer's total is `usable / full_price`, so that was
right for it; a WARM peer's is `resident + extra` (`PeerResidency::Layers`), each
loaded layer counted as ONE with its tables already paid. A peer holding 73-80 of
the 70B with no room to spare was charged 9 of its 7 for the tail it had loaded:
no standby from it (`find_standbys`), its plan read as past its offer
(`plan_exceeds_offered_memory` — a later refusal then reported `SwarmShortOfMemory`,
"retrying cannot help", for `HoldersDeclined`), greedy's run cut at 79, and
parallax's first-pass points cut inside a held range. `RangeWeights::width_in_bound`
now takes the held ranges as a REQUIRED argument: the candidate passes its own
(`NodeCandidate::held_ranges`, from the same `ResidentModelLayers` report as
`resident`), the chain planner `&[]` (a ceiling is cold). Test
`a_range_a_peer_has_loaded_costs_its_width_in_its_total` (red with `&[]`: 9, false,
true, 79 for 7, true, false, 80).

## Each rung cuts its split points at the bound it applies (2026-10-26, #350)

**What it replaced.** `route_over` adds a capacity split point where each
candidate's prefix or suffix stops fitting (report #029). It cut them at
`max_hostable_layers` — the FIRST rung's figure — on every rung, so a rung that
relaxes a peer to its ceiling held the peer to a bound no split point marked:
it could accept the range that filled the ceiling and never propose it.
Request `c2cde5c3` (2026-10-25, live node, Llama 3.3 70B), the re-plan after
two up-front refusals: this node's worker held 0-13 with the card full (room
for 0 new layers); AU held 13-73, at most 44 layers (its refusal), current
figure 40; Proxmox 28-80, 6; the Mac 0-24, 27-59, 61-62, 63-80, ceiling 23.
A route existed at `PeersAtCeiling` — 0-13 here, AU 13-57, Proxmox 57-63,
Mac 63-80 — but AU's point was 53 (13 + 40), the 13-57 vertex did not exist,
both that rung and `LocalUnbounded`'s retry after this node's loader refused
0-28 found "no reachable source→sink path", greedy found none either, and the
request failed. (FUTURE_WORK's entry said the worker held 0-19; the
candidate log reads `held_ranges=[(0, 13)]`.)

**The rule.** `route_over` asks the vertex filter's own cap (`vertex_cap`:
this node's loader answer under `binds_local`, a peer's `CapacityBound::peer_cap`)
against its own charge (`capacity_charge`) and adds the furthest prefix and
suffix that fit as points, BESIDE the first rung's (still valid, and a narrower
neighbour may need them). For this node that also cuts by what a range ADDS
beside what its worker holds, so a range can reach past the held layers by
exactly its room. Where the two sets together are past `MAX_SUBRANGE_VERTICES`,
this rung's alone are tried before the declared boundaries. Test:
`a_pass_holding_a_peer_to_its_ceiling_can_cut_its_range_there`, the request's
four candidates, red before the change with the live log's error.

## A peer's count of what fits is weighed in its own unit (2026-10-26)

**What it replaced.** A peer refusing a request up front (#252 (b), #345)
says how many layers fit, and the re-plan held its ceiling to that count. Two
counts reach that one figure: the widest run from the first asked layer whose
weights would LOAD (`peer_load_refusal`, by bisection over real layers), and
the share of the asked layers whose CACHES fit (`model_worker::layers_that_fit`,
every layer costing the prompt the same). Both were read as a bare number of
average layers — right while every bound charged a range its width, wrong once
#353 weighed a range as what it loads, the same day. The v0.3.255 field check
(live node, 07:37-07:43Z, three novel prompts to Llama 3.3 70B over TH/TH/AU/IT):
the AU node refused [16..66) — 25,312 MB — saying 45 would load, which were
[16..61), 21,585 MB of the model's lighter layers (460-484 MB). The re-plan
gave it [28..75): 47 layers of 484 MB, 22,740 MB, weighed 44.86 average layers
at the file's 506.9 — under 45. AU refused again ("needs about 24012 MB …
allows 24000 MB") and all three requests failed `HoldersDeclined`. Read off
the model's header (`gguf_weights.py` over `gguf_header.bin`: layers
460.1-518.9 MB, head 822.0, table 563.6).

**How others do it.** llama.cpp's `common/fit.cpp` never converts: it measures
each device's memory for the exact layers it would place, per layer. Here the
asked node did measure the exact run; the coordinator lost it by keeping only
its length. A count is a measurement of a SPECIFIC run, and stays one.

**The rule.** The answer says which count it is —
`PromptAdmissionVerdict::Refused::counted` (`swarmllm_types::LayersCounted`:
`Loading`, `Caches`, `Unsaid` from a node that predates it; a `#[serde(default)]`
JSON field an older coordinator ignores, pinned both ways by
`a_refusals_count_kind_crosses_between_old_and_new_nodes`) — and the
coordinator keeps it whole with the start it was counted from
(`daemon::state::PeerPromptFit`, every verdict per start and kind, the fewest
layers each). `gather_candidates` weighs each in its own unit: `Loading` →
`RangeWeights::ceiling_of_a_fit` (the charge of THAT run beside what the peer
holds, rounded down, so anything admitted weighs no more than what loaded);
`Caches` → the bare count, with the candidate never charging a range below its
width (`NodeCandidate::prompt_fit_counts_caches`, in `capacity_charge` and
`width_in_bound`: a lighter range's weights buy its caches no room); `Unsaid` →
both. The tightest verdict holds. Without a header the bare count is used, as
before #353.

**Measured** (unit, the 70B's real per-layer sizes): the fit [16..61) is held to
42; [28..75) (44.9) is out where 45 admitted it; every range the bound admits
weighs at most the 21,492 MB that loaded (an exhaustive sweep of all 3,240
ranges); a 5-layer tail with its head is 6
(`a_fit_counted_in_real_layers_is_held_to_the_charge_of_that_run`). Through
`gather_candidates` on an uneven 32-layer model: a `Loading` 18 from layer 8 →
15, with the null control that 18 admits the heavier [12..30); `Caches` 18
refuses 19 light layers the weights alone would admit; `Unsaid` both
(`a_peers_count_of_what_fits_is_weighed_in_its_own_unit`).

**End to end** (2026-10-26, `examples/split_rig.sh peerload`, dev build of
`25d94f53`, Llama-3.2-3B on the processor; A holds layers 0-21, B 12-28 and
TinyLlama whole; B starts a 600-token TinyLlama reply after A has held its room
for 40 s, then A is asked). At `RAM_B=1700`: A priced B from its idle room
(`max_hostable_layers=Some(14)`) and asked it for [21..28); B refused before the
pass ("needs about 1205 MB … budget allows 1700 MB and 980 MB is already in
use", `layers_that_fit=5`); A logged `counted_from=21 counted=Loading`, re-priced
B at 4 (`ceiling_of_a_fit`: five middle layers, under the 7 + head the range
needs), asked it nothing more, and answered `HoldersDeclined` in 2.3 s — B is
busy, so "try again" is the right advice. `RAM_B=1450`: fit 2, re-priced 11 → 1,
the same shape. The .255 field check failed on exactly this path, AU refusing
twice. With A holding everything but B's last part, no narrower route exists,
so this rig checks the refusal and the hold, not a narrower reply; at 2,200 MB
and above, B's busy room still took [21..28) and answered.

**What a change must keep.** A number a peer measured is kept with what it was
measured OVER. When a bound's unit changes, every figure compared with it is
re-read in the new unit — the refusal count was the one #353's sweep of readers
missed, because it entered as a ceiling, not as a reader (gotcha #970). A new
`LayersCounted` variant would make an older coordinator fail to decode the whole
answer (read as "go ahead"): gate it on a feature bit.

**The figure it was reported with.** The refusal read "room for about 67 of
its 80 layers" — `SwarmShortOfMemory`, "retrying cannot help" — because
`layers_offered_at` counted this node at its figure, which is room to ADD
(`layers_it_would_add`), not a total; 67 is exactly layers 13-80. It now adds
the layers this node's worker holds (`this_nodes_offer_counts_the_layers_its_worker_holds`).

## An idle worker's own memory is the fresh run's, and its own request does not pin it (2026-10-26, #350)

**What it replaced.** In request `c2cde5c3` (§ "Each rung cuts its split points
at the bound it applies") this node offered 0 layers beyond the [0..13) its 70B
worker held. The entry blamed an idle Gemma-2-2B on the card that the bound did
not count; **that was wrong**: node.log shows Gemma reclaimed for the 70B at
15:56:19 ("Freeing graphics memory from an idle model … reclaimed_mb=2367"). At
16:29 the card held only the 70B's worker — 10 of its 13 layers, filling the
6,009 MB budget — so the card bound, what a RUNNING worker can add (gotcha #806),
was rightly 0. Two other things failed, and a third check came back clean:

1. **The fresh run counted the worker's own memory as taken.** The run a fresh
   worker would take in its place — what `grow_worker` makes of a range the
   worker cannot grow into, once `a_fresh_spawn_fits_in_place_of_the_worker`
   says it loads there — was weighed against the card less every charge, the
   worker's own included. A card/processor split always fills the card, so an
   idle split worker never had a run on offer: `fresh_run_layers=None` on the
   request's FIRST plan (16:29:41), with the worker idle since ~16:15. The run
   now gets the worker's own `vram_reserved_mb` and `ram_reserved_mb` back, as
   the retirement gate does. Ollama re-places a runner the same way: one that
   needs a reload is unloaded before free GPU memory is read for its new
   placement (`server/sched.go`: `needsReload`, then `updateFreeSpace`).
2. **The request's own up-front admission pinned the worker.** Attempt 1 planned
   the held [0..13) here; `admit_prompt` admitted it and marked the worker in
   use for the request (`note_conversation`). The re-plan put [0..28) here
   (`LocalUnbounded`), the growth was refused, and the retirement answered
   `Keep` — the worker was "in use", by this same request: "28 layers 0..28 …
   need about 14644 MB more than this node has left (its worker is already
   holding 6236 MB)". The admission's verdict was about caches the forward was
   going to throw away. `admit_prompt` now skips a worker its forward will
   replace (`forward_replaces_the_worker`: not held, not grown beside what it
   holds, nothing using it, and loading in its place), as it skips a model with
   no worker. A worker in use by an earlier attempt of the same request is still
   kept: the planner then offers no fresh run (it asks `in_use` too), so no plan
   needs the retirement. Ollama never lets a request pin a runner it needs
   reloaded: a request takes its reference (`refCount++`) only when it is handed
   a LOADED runner — `useLoadedRunner`, or `load`'s goroutine after
   `WaitUntilRunning` — and the `needsReload` branch skips both, so the runner's
   count holds only OTHER requests when it is expired (`server/sched.go`,
   `processPending`, read 2026-10-26).
3. **`planning_on_card` was not a cause** (checked, 2026-10-26): it was
   suspected of weighing the 70B in system memory, but `would_fit_on_gpu`
   answers "fits" for a model whose worker holds card memory, so it read "the
   card". A test written on that theory passed with its fix switched off and
   was dropped.

The field confirms the size: the user unloaded the 70B at 16:31 and the next
spawn took 28 layers, 10 on the card, reclaiming an idle drafter's 2,290 MB of
system memory.

**Tests** (`process_pool`, the field's shape: an 80-layer 40 GB model, a 6,000 MB
card, the worker's 10 layers filling it, nothing using it):
`an_idle_workers_own_memory_is_the_fresh_runs` (the card bound stays 0; the run
offered is the one with no worker, wider than the worker holds, admitted by the
retirement's gate; none once the worker is in use, #93) and
`an_up_front_admission_leaves_alone_a_worker_its_forward_will_replace` (nothing
sent, the worker not marked used; controls: the held range, and the worker in
use). Each was red with its own fix switched off.

**Left.** The local candidate's `cold_load_ms` charges only the layers a run
ADDS, so a run that replaces an idle worker is priced without reloading the
layers it held — it under-prices the respawn, which matters only against a peer
holding them warm.

## `inference::scheduler::delegation_target`

(2026-08-18) — the single decision
to hand a WHOLE model to a peer rather than run it on this node's CPU. Fires only
when `ModelProcessPool::is_cpu_bound_for_lack_of_vram` says we have a working GPU
this model does not fit, and only for a peer that holds every layer, is directly
reachable within `DELEGATE_MAX_LATENCY_MS`, is trusted at least as much as an
ordinary peer, and advertises GPU room with margin.
**It returns a peer or nothing, and never falls through to the routing search.**
That is the whole difference from the version reverted in `cbbed678`: that one
priced a full local node at 10,000/layer and let the DP decide, which made local
layers unusable, priced out the good split (some layers here, rest on a peer 5 ms
away) and picked a node in another country. Both outcomes here are a single
segment. Do NOT reintroduce a penalty term — it distorts every other route.
**Prompt privacy changes the SHAPE, it does not disqualify the peer.** With privacy
off the peer gets the whole model. With privacy on, `boomerang_assignment` keeps
layer 0 and the final layer local — the embedding and the sampling, which is what
the guarantee actually is — and gives the peer everything between, as encrypted
activations. Since `encrypted_pipeline_auto` is on by default for any model whose
ends this node holds, that is the COMMON path, not an edge case: treating privacy
as a veto stranded the default configuration on its CPU for no privacy gain.
**The boomerang is constructed, not searched, for the same reason.** Asked to route
it, the general search answers "all of it locally" — that satisfies the encrypted
constraint at zero network cost and nothing in its cost model knows this node is
about to fall back to its CPU. Verified 2026-08-18: merely standing the fast path
aside produced `segments=1 node=<local> layer_start=0 layer_end=28`. Teaching the
search that local compute is expensive here is what `cbbed678` did, and it
distorted every other route.
Three things made it inert until it was run on real machines, all now fixed and
all worth knowing before touching this: gotcha #329 (`would_fit_on_gpu` said yes
for a model resident on the CPU), #330 (every node gossips zero free VRAM), #331
(the latency bound was calibrated against network intuition, not against what
`peer_registry.latency_ms` actually measures).

## `inference::router::distributed_exec::failure_is_penalty_worthy`

(R146) — gates `penalty_serve_failure` on (a) the assignment actually
having had a remote segment and (b) the error not being locally
attributable. Any new automatic credit or reputation penalty MUST route
through an equivalent attribution check. `ServiceUnavailable` means
"THIS server can't serve" and `Internal` means our own bug — neither can
ever justify charging a peer.

## Trust is paid for work that was checked, and the check runs whenever the payment would

**Rule:** `.claude/rules/arch-scheduling.md` § "Trust is paid for work that was
checked, and the check runs whenever the payment would".

### What it replaced

`router/distributed_exec.rs` credited `TrustEvent::InferenceSuccess` to every
remote segment holder the moment a pipeline returned `Ok`, and only then called
a spot check that sampled 5% of results. The constants are `+0.01` for success
and `-0.1` for a spot-check failure, so the expected movement for a peer whose
output is always degenerate was:

```
+0.01  +  0.05 x (-0.1)  =  +0.005 per request
```

Positive. Not "escapes detection sometimes" — **ahead on average, forever.** The
break-even sampling rate is 10% and the configured rate was 5%. Reported by a
contributor on issue #21 (2026-09-10), who derived the arithmetic from the
source; reproduced here as a test that reads **1.0**, the clamp ceiling, with
the fix reverted.

Two things made it worse than the arithmetic suggests:

- **The credit was per SEGMENT.** A peer handed three segments of one pipeline
  was credited three times for one piece of behaviour, so taking more of a
  pipeline ratcheted faster.
- **Trust is not decorative.** `DELEGATE_MIN_TRUST` gates
  `trusted_with_the_plaintext_prompt` — who may be handed a user's prompt in
  cleartext — and `trust_score` is a tie-breaker in the candidate ordering. A
  peer drifting upward gains both.

### What the check can and cannot establish

The three checks are *well-formedness* assertions over the assembled reply:
empty text with a non-zero token count, text with a zero token count, and a long
reply that is one character repeated. **A peer returning fluent, confident,
wrong tokens passes all three**, and still earns trust. That limit is
structural, not an oversight to tighten:

- BOINC detects wrong results by **replication with consensus** — the same job
  on two hosts, accepted only if they agree — and its *spot*-checking sends work
  whose answer is already known. Its reputation is per (host, app version), and
  adaptive replication lowers the replication probability as consecutive
  validated results accumulate (Anderson, *BOINC: A Platform for Volunteer
  Computing*, J. Grid Computing 18(1), 2020).
- Sarmenta's analysis is the same shape: spot-checking works because the checked
  work has a ground truth, and combining it with voting is what shrinks the
  error rate exponentially (*Sabotage-tolerance mechanisms for volunteer
  computing systems*, FGCS 18(4), 2001).

We have no ground truth for an inference result and no second computation to
compare against, so the honest claim is the weak one the checks support. Do not
let the module's name imply more.

### What a change must keep

- **The check runs on every distributed result.** It is three assertions over a
  string already in hand — there is no cost to amortise, and that was never the
  reason it was sampled. `effective_spot_check_rate` still exists and still
  governs *credit-transaction* verification in `anti_gaming.rs`; that is a
  different question with a real cost behind it.
- **The verdict comes before the credit.** Re-ordering these re-introduces the
  defect exactly.
- **One request, one observation per peer.** Dedupe segment holders.
- **A malformed result pays nobody, and convicts nobody it cannot identify.**
  Withholding the reward needs no attribution and is by itself enough to stop
  the ratchet. The `-0.1` needs attribution, and the only case that supplies it
  is a single remote peer. Docking all participants of a multi-peer pipeline
  would let one bad node lower the trust of every honest node it shares a
  pipeline with — and since candidates are ordered by trust, that is an attack
  on competitors, not collateral damage.
- **Take no locks on this path.** It runs on every completion now. The previous
  version acquired the `anti_gaming` mutex twice per check; the helper it called
  (`report_spot_check_failure`) ignored its node argument and returned a
  constant that was only logged, alongside the real `-0.1` it had already
  applied, so the log read as though two penalties had landed. Both it and its
  single-variant `PenaltyAction` enum are gone.

### What is still open

The bar this trust feeds is `DELEGATE_MIN_TRUST`, which equals `DEFAULT_TRUST` —
the score an unknown peer starts on. See `docs/FUTURE_WORK.md` #30 ("The
plaintext-prompt trust bar sits exactly at the score an unknown peer starts on").

**From the rules file (moved 2026-10-02):**

`router::spot_check::check_distributed_result` returns a verdict and
`settle_participant_trust` applies it — in that order, on **every** distributed
result. A peer earns `InferenceSuccess` only from a result judged well-formed,
once per REQUEST rather than once per segment, and a malformed result pays
nobody. The penalty is applied only where a single peer served the request,
because with several the fault is not attributable to any of them and docking
all of them is a way to demote honest competitors.

Sampling a check that gates a reward inverts the reward: crediting every
participant and then checking one result in twenty gave a peer returning
degenerate output **+0.005 per request**, and it climbed to the 1.0 ceiling.

→ `docs/invariants/scheduling.md`

## A forward the peer could not open goes to it again, once the link is re-keyed

(2026-09-24, FUTURE_WORK #92.) A node that cannot open a forward's seal
answers `LayerResult::error("Could not decrypt forward")` at once and, inside
`SessionManager::open`, arms a repair handshake that completes in about a
round trip and fixes the link for every later request. The request in front of
it never got that far: the coordinator treated the refusal like any other
error, `failover_segment` looked for a standby, single-peer delegation has none
by design, and the caller got `Segment 0 failed with no standby available (last
failure: Could not decrypt forward)` — both of the reporter's traces on
v0.3.193. The router's retry could not help either, because the failover bars
the node it just tried, so with one holder there was nowhere to go.

**The decision.** A decrypt failure is not what `blacklist_holder_for_request`
exists for. Envoy's `previous_hosts` rule — a host that just failed is likely
to fail again — is what justifies barring a peer, and it does not hold for a
failure whose fix is already under way. And sending the same step AGAIN to the
SAME peer is safe where failing over is not: the worker never saw the forward,
so the peer's KV cache for the conversation is exactly as it was, at any point
in a reply.

**Research (rule 0).** gRPC's gRFC A6 names it: a *transparent retry* is one
for an RPC that "has never been seen by the server application logic" — "the
client library will immediately retry it once", and transparent retries "do
not count toward the limit of configured RPC attempts" nor as failures for
throttling. WireGuard holds a packet it cannot yet send in a staged queue while
the handshake runs and sends it once the keypair exists. Both fix the same two
details this needed: retry ONCE, and only after the thing that made it fail
has changed.

**What changed.**

- `ForwardRefusal::Undecryptable` on `LayerResult::refusal`, as the `0x06`
  result trailer — LAST in the frame, because an older decoder reads 0x03-0x05
  in order and stops at the first marker it does not know. A result's seal
  does not depend on its trailers (a forward's does), so an older coordinator
  skips it harmlessly; the serving node still sends it only to a coordinator
  advertising `features::FORWARD_REFUSAL_REASON`. The message stays generic —
  it must not tell an attacker whether a forgery had the right key — and
  nothing matches on it (#295).
- `SessionManager::rekeyed_since(peer, t)`: was the key this node SEALS with
  installed after `t`? The node that failed to decrypt INITIATES the repair,
  so the coordinator is its responder, and a responder's new key waits in
  `unconfirmed` — still sealing with the old one — until the peer's
  confirmation opens. Resending on "the exchange was answered" would go out
  under the key the peer just refused.
- `pipeline::local::wait_for_result` takes a required `ResendOnRefusal`. On
  the typed refusal it polls `rekeyed_since(sent_at)` for
  `2 s + 4 × ack SRTT` (capped 10 s; 5 s with no SRTT — the repair costs round
  trips and this fleet spans ~1 ms to ~350 ms), re-registers the waiter pinned
  to the same node, rebuilds the forward through the caller's closure and
  sends it. At most once per forward; a second refusal, a wait that expires,
  or a cancelled request hands the refusal to the caller's ordinary handling —
  where it went before.
- Every coordinator wait passes it: the main loop, the failover standby loop,
  `forward_verify_through_segments` (ngram and DSD), the speculative prefill
  and `send_verify_batch`. A CHAINED forward passes `Never` — the refusal may
  come from any hop, the hops before it have already appended this step, and
  the chain branch already re-runs the segment unchained with a KV rewind.
- The forward is REBUILT by a closure rather than kept, so nothing is copied
  on the path where no refusal comes; the main loop already holds the input
  for failover.
- `SWARMLLM_FAULT_REFUSE_DECRYPT=<n>` makes a serving node refuse its `n`th
  encrypted forward, once, and arm the repair — the only way to exercise this
  on demand.

**Verified live** (2026-09-24, `split_rig.sh split`, TinyLlama A=[0] B=[1],
CPU release build, B wrapped with the fault switch): every fault run logged
`link re-keyed — sent the refused forward again to the same node` with
`repaired_after_ms` 21-23, zero failovers, 200. The refused request's reply —
refusal landing mid-reply, on a verify round — matched the unfaulted run in all
17 fault runs (the whole of the first question's reply; the first 90 characters
where the refusal landed in the second). Two early runs showed the NEXT
request's reply flipped at a near-tie; 26 later runs (13 with the fault) did
not, so the refusal is not a sufficient cause — FUTURE_WORK #106 holds that
evidence and the instrumentation that localises it.

**What a change here must keep.**

- `a_forward_the_peer_could_not_open_goes_to_it_again_once_the_link_is_rekeyed`
  — same step, same node, not before the re-key; goes red with the resend off
  and with the re-key wait skipped.
- `a_link_that_is_never_rekeyed_gets_no_resend_and_the_refusal_stands` — goes
  red if the resend does not wait for the re-key.
- `only_a_typed_refusal_on_a_path_that_allows_it_is_resent_and_only_once` —
  the untyped words, a `Never` path, and a second refusal are not resent, with
  the link repaired during each wait so a wrong resend would actually send.
- `a_link_reads_as_rekeyed_once_a_resend_would_open_and_not_before` —
  answering the exchange is not adopting its key.
- `the_refusal_trailer_is_the_last_thing_in_the_frame` — what makes the
  trailer safe for older decoders.
- **Not covered: the opt-in persistent pipeline stream**, whose receiver drops
  a forward it cannot open without answering at all → FUTURE_WORK #105 (since closed, v0.3.205).

**From the rules file (moved 2026-10-02):**

The exception to "a host that just failed is barred": a peer that could not
decrypt a forward never ran it, and has already armed the repair.
**`pipeline::local::wait_for_result` takes a REQUIRED `ResendOnRefusal`**, and
on `ForwardRefusal::Undecryptable` (a TYPE on `LayerResult`, never the
message) it waits for `SessionManager::rekeyed_since`, then rebuilds the same
forward and sends it to the same node, once — correct even mid-reply, since
the peer's KV is untouched (gRFC A6 "transparent retries"). A chained forward
passes `Never`: the refusal may come from any hop. A new wait on a remote
segment gets this by construction; do not add a second resend beside it.

→ `docs/invariants/scheduling.md` § "A forward the peer could not open"

## A peer that went silent is barred from the retry, and the retry happens

(2026-09-12.) Three producers of `SwarmError::PeerUnresponsive`: the ACK
sweep ("never acknowledged"), the first-token wait ("remote-generate timed
out"), and the per-segment result deadline (`pipeline/local.rs::segment_timeout_error`,
"Timed out waiting for segment result"). The router's retry classifier
`is_transient_remote_failure` matched on the first two by their WORDING —
the #295 trap — so the third reached the router as a 503 and was never
retried. On the ordinary distributed path that did not matter, because
`forward_through_segments` hands a deadline expiry to `failover_segment`,
which bars the failed node at every exit and returns
`SegmentFailoverExhausted` (retryable). On the speculative verify rounds
(`forward_verify_through_segments`, used by `ngram_only_spec` and the
speculative pipelines) there is no failover: the `?` propagates the raw
variant straight to the router, and the request ended there.

And neither fast-path arm barred the silent peer before returning, so the
retry those two already had could re-plan onto the peer that had just gone
quiet and wait the same silence out again — "the retry re-assembles a fresh
pipeline that can pick the same holder; with a single holder it simply waits
the same deadline twice" (FUTURE_WORK, "Speculative distributed decode has no
failover", § Related). The pairing that makes a retry work here is stated at
`remote_peer_could_not_serve`: the blacklist is what stops the re-plan
re-learning the same holder from the DHT.

**What changed.** `blacklist_holder_for_request(request_id, node)` is called
at all three producers before the error is returned — in the deadline arm
only when `is_remote`, since the local candidate must never be barred from
its own request — and `router::peer_went_silent` (the TYPE) joins
`should_retry_after` under `used_remote_segment`, beside
`remote_peer_could_not_serve` and `segment_ran_out_of_machines`. The prose
list is untouched, so nothing that retried before stops retrying.

**Research (rule 0).** Envoy's `previous_hosts` retry-host predicate: "choose
a different host than the host where the previous request has failed, because
typically failures on that same host are likely to continue for some time and
immediate retry would have less chance of success" — the same rule, and the
same shape (per-attempt exclusion, not a global mark against the host).

**What a change here must keep.**

- `a_deadline_that_expires_bars_the_peer_from_this_request` — the bar is
  scoped to the request id; another request may still use the peer.
- `a_dropped_response_channel_does_not_bar_the_peer` — the control: our own
  dropped sender says nothing about the peer and bars nobody, the same
  attribution rule `segment_delivery_verdict` follows.
- `a_segment_deadline_is_retried_only_with_a_remote_segment_involved` — the
  variant is retried by type, never without a remote segment, and never once
  text has streamed.
- A NEW producer of `PeerUnresponsive` must bar the peer before returning.
  The retry assumes it; without it the retry waits the deadline twice.
- What is NOT changed: the speculative decode path still has no
  mid-request failover of its own (the KV-state question in that entry
  stands). What it gains is that a non-streaming request whose peer goes
  quiet is re-planned once on a different holder instead of ending.

## A prompt longer than one peer serves is that peer's limit

Field report 2026-09-25: an 8560-token agent prompt was planned through a peer
serving 8192 (the shipped default). The peer checked length one 128-token chunk
at a time (`forward_prompt_in_chunks`), so it computed 8192 positions through its
layers and refused 177 s after dispatch as "8320 tokens" — the chunk boundary, not
the prompt. The coordinator's `every_holder_would_refuse` read every `Validation`
as the request's own mistake and returned it, although the segment had a standby
and the coordinator itself served 65536. The message told the user to raise
`max_seq_len_override` — their own, which cannot change another machine's.

The 2026-08-30 rule that `Validation` never fails over was right for what it was
written against (a malformed argument is reproduced by every holder) and wrong
for this one case, because a node's served context is a property of the NODE.

**Four parts, each checkable alone:**

1. **The peer refuses the whole prompt before any layer runs** —
   `model_worker::refuse_a_prompt_past_the_served_context`, beside the
   whole-prompt KV admission. Rig: 3 ms after the model loaded, where v0.3.205
   computed 512 positions first.
2. **The refusal is read back with its numbers** — `error::longer_than_served`
   writes it, `error::served_context_refusal` reads it. The opening sentence is
   the wording every release since v0.3.101 sends, pinned by a frozen v0.3.205
   literal, so older peers are understood and older coordinators relay it as
   before. At the model's declared limit (`SharedState::model_declared_context`)
   it is still every holder's answer.
3. **Peers advertise their ceiling** (`NodeCapability::context_ceiling_tokens`,
   additive, `None` = unknown = still a candidate). The coordinator does not send
   a prompt pass to a peer that advertised too short
   (`distributed::advertised_context_refusal`, exact positions read off the
   hidden-state header) and `failover_segment` skips such standbys.
4. **Running out of machines is the caller's 400 naming whose limit it is**
   (`longer_than_the_swarm_serves`), never `SegmentFailoverExhausted` — whose
   "too few machines hold this model" advice is #295's trap.

**Deliberately not at plan time.** Routing sees only `estimate_prompt_tokens`
(bytes/4), which over-counts English prose by ~10% and would exclude peers that
could serve; the send-time check uses exact positions.

**Verified** on `examples/split_rig.sh context` (four nodes, llama-3.2-3b, a
563-token prompt, the middle holder at 512): all-v0.3.205 reproduces the report
(400, "Raise it in Settings", standbys unused); new coordinator + old peer fails
over from the refusal to the C+D composite, 200; new + new never sends to the
peer, 200; with no cover, the new 400.

### The whole-model half (2026-09-25, night)

The path where ONE peer runs the whole model (`remote_generate`, including a
delegation) returned the peer's refusal as the request's 400 and never tried
another machine. Its refusal comes from `model_worker::resolve_max_new_tokens`,
worded "…too long for {model}: N tokens of prompt against a limit of W…" (or
"…plus the reply you asked to reserve is M, and the model's limit is W"), and
**`W` is the window that worker LOADED** (`SplitModel::context_window` =
`max_seq_len`, capped at the node's ceiling by the loader) — so "the model's
limit" was the node's limit whenever the model declares more. A test had pinned
the opposite premise ("already says the model's own limit"); it was wrong and was
replaced by a frozen literal of the two released wordings (v0.3.183+).

- **Read back** by `error::served_context_refusal`, which now recognises both
  wordings; the numbers are the conversation (prompt, or prompt + the explicit
  reply budget — what another machine needs room for) and the window.
- **Decided** by the same `every_holder_would_refuse` as segments: below the
  model's declared context it is the peer's limit.
- **Acted on** by `remote_generate::longer_than_this_peer_serves`: bar the peer
  (`blacklist_holder_for_request`) and return `SwarmError::LongerThanPeerServes`
  — worded and classified like `Validation` (400), its own variant for ONE
  decision: the router re-plans it (`peer_serves_shorter_context`, gated on a
  remote segment, and a further re-plan is earned like a memory refusal's).
- **Reported** as the original refusal when the re-plan fails too, unless the
  re-plan's own error is a 4xx — this node's worker refusing at ITS limit, whose
  advice to raise the setting here is then correct
  (`router::report_after_a_replan` — ONE "first informative refusal" slot shared with the local memory shortfall, so the two cannot stack).

**Still not done, deliberately:** `delegation_target` does not read the
advertised ceiling. A delegate refuses in milliseconds (tokenize, compare) and
the re-plan follows at once, while the plan-time estimate over-counts ~10% and
would hand work that a delegate could serve to this node's processor instead.

Rig: `examples/split_rig.sh whole` (A holds only the header, B all of it at 512,
C all of it at the default).

## A warm prompt is priced warm (2026-09-26)

**Field report, v0.3.206** (CPU-only Ryzen 7 5700U, xlam-2-3b 4/4 shards, a
GPU peer with ~2.4 GB free, agent harness nanobot, ~9.2K-token prompt a turn):
turn 1 ran here in 677.8 s and cached its prompt; turn 2 — the first ~9,200
tokens identical — went `7c10ea04,bf7b3263,7c10ea04` and took 847.1 s. The
router had priced running here on all 9,280 tokens: nothing in the cost model
knew what this node's worker held. On a node without a card
`local_fast_path_allowed` stands aside whenever any peer is connected, so the
router decides every request such a node makes, every turn.

**The mechanism** (`scheduler::cached_prefix`): render the prompt exactly as the
executor will (`pipeline::render_prompt_from_header` — one function, both
callers), tokenize with the standalone tokenizer built from the same header as
the worker's, chain-hash at the worker's block size
(`ModelProcessPool::prefix_cache_block_tokens`), and count the LEADING blocks
present in `models.peer_prefix_blocks[our id][model]` — the index the loopback
forwarder rewrites from every worker insert (the full bucket's manifest, not
just the new entry). That count is `PromptPlan::cached_locally`, carried on the
LOCAL candidate as `NodeCandidate::cached_prefix_tokens`, and
`parallax::vertex_cost` takes it off the prompt of a WHOLE-MODEL vertex only.
Every price — the search, the hand-off gate (`costs_more_than_staying_here`),
the logs and `note_route_prediction` — goes through `vertex_cost` with the
candidate, so none can disagree. Tokenizing only happens when the worker has
reported a cache for the model: the common request pays one map lookup.

**Why only this node's cache.** A local whole-model plan runs through
`try_local_generate_fastpath` → `Generate`, deterministically: DSD needs two
segments, draft speculation a draft model, and the n-gram path declines a plan
that is entirely local. A PEER's whole-model plan does not: the n-gram path runs
BEFORE `remote_generate` for a remote single segment, and it sends forwards,
which never look the cache up. A peer credit would be priced and not delivered
whenever that path won — so peers get none until that is resolved (FUTURE_WORK #10).
*Resolved 2026-09-27* — the n-gram path stands aside for every plan the hand-off
takes (`ngram_only_spec::eligible` asks `remote_generate::eligible`) — and a peer
is credited since 2026-10-18 (the subsection "A peer handed a prompt…" below).

**What it keeps from SGLang** (v0.4 cache-aware routing, and
sgl-project/sglang#26263): the key is the full rendered prefill input, tools
included — keying on `messages[0]` made unrelated conversations collide — and a
hit is a PRICE that load and a faster route can still beat, never a pin.

**Errs toward cold.** Whole blocks only, and never the last token, so the credit
never exceeds what `PrefixCache::lookup` hydrates. A stale self-entry (evicted,
or lost with the worker) costs one uncached run — what the request cost before
— and that run re-announces.

**Measured** (`examples/split_rig.sh cache`, qwen2.5-0.5b, both nodes on the
processor): turn 2 of 2,744 tokens — planner credit 2,688 (42 blocks), local
plan, worker `prefix-cache HIT matched_tokens=2713`. Tests:
`a_processor_holder_keeps_a_prompt_its_worker_already_holds` (the field shape;
fails without the credit — the plan is the four-segment boomerang),
`a_small_cached_prefix_does_not_keep_a_long_prompt_off_faster_cards` (the price,
not a pin), `a_cached_prefix_is_credited_only_to_a_whole_model_run`.

**Also fixed with it:** `estimate_prompt_tokens` now counts tool definitions — an
agent sends dozens and they were priced at zero, in prefill and in the KV a peer
is asked to hold.

### A split segment is credited what this node believes it holds (2026-10-11, #10)

**The shape.** v0.3.234 made a split's prompt pass resume where every segment is
believed to hold it (`pipeline::split_prompt_cache`), but the planner never read
those beliefs. With two WARM peers able to run the same layers, only one holding
the conversation, the next turn went to whichever priced faster. Rig
(`split_rig.sh splitcache SPARE=1 WARM_SPARE=1`, Llama-3.2-3B on the processor,
A shard 0, B and C shards 1-3, debug builds): turn 1 was kept on C; B was then
asked a conversation of its own (its plan named A + C, so it handed the lead to
A, which put it on B); turn 2 went to B — measured a moment before on a short
request — and read all 2,306 tokens again, 196 / 195 s against turn 1's 184 /
180, in 2 of 2 runs.

**The mechanism.** `cached_prefix::plan_prompt` also tokenizes when the belief
table names the model (`pipeline::believes_any_of`), and asks
`pipeline::held_openings` for every (node, layer range) believed to hold this
prompt's opening, judged by `believed_tokens` — the rule a prompt pass resumes
by; `resume_at` is now the least of it over a plan's segments, so the two cannot
disagree. `gather_candidates` hands each candidate its own entries
(`NodeCandidate::split_openings`, this node's included), and
**`NodeCandidate::prompt_tokens_held(range, num_layers)` is the one accessor**:
`cached_prefix_tokens` for a whole model, the exact-range entry for a segment.
`vertex_cost` takes it off the prompt (the prefill and the transfer — held
positions are never sent), so the search, `chain_cost_ms`, the hand-off gate and
`predicted_ms` all see it.

**Greedy prices the UNREAD prompt, not a credit.** The fallback plans every split
whose holders overlap (#180) and priced no prompt at all. A credit alone, against
a model that charges nobody for the prompt, hands a slow holder a long turn a card
reads whole in a fraction of the time: holder at 7.08 ms a layer, card at 0.8,
2,306 tokens — the credit-only figure for the holder goes NEGATIVE. So greedy now
charges `parallax::prefill_ms_per_layer` (the rate `vertex_cost` uses, factored
out) for the tokens a candidate does not hold, once per request, spread over
`ASSUMED_FORWARD_PASSES` as the cold load is. Tests
`greedy_sends_the_next_turn_to_the_warm_peer_holding_its_opening`,
`greedy_does_not_keep_a_turn_on_a_holder_a_card_outpaces`.

**Exact range only, per vertex.** A worker keys stored prompts by its layer range
(`SplitModel::kv_model_key`), so a range one layer wider holds none. A split
resumes only as far as EVERY segment holds, while the credit is per vertex: a
plan pairing a credited segment with an empty one is priced as if it resumed —
wrong only where the holder is not the faster machine, by at most its credit, and
that pass stores the prompt on every segment. A delegated split (no segment of
ours) is led by its head with the head's beliefs, so a credit from ours resumes
nothing there: a bounded miss, the cost of the request before this change.

**Measured on the change** (same rig, same tree): turn 2 went to C and resumed
from token 2,240 (C's `HIT on a split's stored prompt`), 55.8 / 53.6 s against
turn 1's 184.8 / 183.8, in 2 of 2 runs; the chosen chain's `predicted_ms` for
turn 2 was 113,587 against the control's 421,272 — the credit, in the number the
router acted on. The overlapping shape (`SHARDS_A=0,1,2 SHARDS_B=1,2,3`, so greedy
plans: "falling back to greedy" on both turns of both arms): control turn 2 read
from 0, 142.8 s after turn 1's 132.0 (`predicted_ms` 825,870); the change resumed
from 2,240 on C, 52.3 s after 134.0 (`predicted_ms` 124,308). Tests: `a_split_segment_is_credited_only_for_the_range_it_holds`,
`the_next_turn_of_a_split_goes_to_the_peer_holding_its_opening` (through
`assemble_pipeline_for`; picks the faster C without the credit),
`the_planner_is_told_each_segment_holding_the_opening`.

**From the rules file (moved 2026-10-02):**

**`scheduler::cached_prefix::plan_prompt`** tokenizes the prompt the way the
executor renders it (`pipeline::render_prompt_from_header` — never a second
rendering) and counts the leading blocks our own worker holds
(`models.peer_prefix_blocks[our id]`). `vertex_cost` takes that off a
WHOLE-MODEL vertex only, via `NodeCandidate::cached_prefix_tokens` — LOCAL
only: a peer's plan can be taken by the n-gram path, which never reads its
cache. Priced cold, an agent's turn 2 left a 9,200-token hit for an 847 s chain.

**`NodeCapability::models_run_on_card`** is "does this peer's work run on its
card" — a card beside a stated `ram_model_budget_mb` (`gpu_layers = 0`) does
not. Never `gpu.is_some()`.

→ `docs/invariants/scheduling.md` § "A warm prompt is priced warm"

### A peer handed a prompt is credited its opening on the next turn (2026-10-18, FUTURE_WORK #3 (c))

**What was wrong.** The live node handed xLAM-2-8B's 505-token prompt to a peer
(BE) a second time: priced 20.2 s of prompt, the whole request took 7.1 s
(0.25x, twice). The peer's worker had read the prompt through `Generate` on the
first send and kept it in its `PrefixCache`; nothing this node prices with knew.
The paragraph above barred a peer credit because the n-gram path once ran before
the hand-off — true until 2026-09-27, stale since.

**Why not the peer's announcement.** `peer_prefix_blocks` holds a peer's own
blocks only if its owner opted in (`share_prefix_cache_with_peers`, default off
— a block hash commits to the prompt), and says nothing of whether the hand-off
will run.

**The mechanism** (`pipeline::handoff_prompt_cache`): after a whole-model
hand-off answers, this node records the prompt's block chain against that peer
and model (`note_handed_off`, the keyed `split_prompt_cache::chain_keys` at 64
tokens, the default `prefix_cache_block_tokens`; the ids from the tokenizing
the hand-off already did to size its first-token wait); a failed hand-off
forgets it. `plan_prompt` then fills `PromptPlan::handed_off` — whole blocks of
the new prompt matching a remembered chain, never the last token — only when
`remote_generate::request_may_be_handed_off` holds: the hand-off's own request
conditions, split out of `eligible` so the price and the path cannot disagree.
`gather_candidates` gives it to the peer as `cached_prefix_tokens` only while
the peer reports a resident range that IS the whole model (unloaded, or holding
part of it, the worker that read the prompt is gone), and `vertex_cost` takes it
off the whole-model vertex as it does this node's.

**Precedent and its trap.** SGLang's cache-aware router keeps an approximate
radix tree per worker from the requests it routed, never asking the worker;
llm-d's approximate prefix scorer records the pod after scheduling. Both drift:
the router never sees an eviction or a restart (llm-d's "precise" scorer exists
to close that gap with the engine's KV events — our opt-in announcement). So a
belief lasts `BELIEF_TTL` (300 s, the default idle unload — it cannot describe a
worker unloaded and loaded again), at most 8 per (peer, model), the longer of a
growing conversation replacing the shorter as the worker's covered-prefix
pruning does; and it is a price, never a pin. Unseen: an eviction under memory
pressure, a card reclaim and reload inside the window, a peer with a
non-default block size — each costs one read priced as a hit.

**Measured** (`examples/split_rig.sh handoffcache`, qwen2.5-0.5b fp16, A holding
the header only and B every part, both on the processor, dev build, 2026-10-18):
turn 2 of 2,744 tokens — A's planner credited B 2,688 (42 blocks), B's worker
`prefix-cache HIT matched_tokens=2713` on the same request id; the turn was
priced 7,847 ms and took 15,085 ms. The released v0.3.243 as A, same B: no
credit, priced 148,827 ms for 14,811 ms (10x). Turn 1 is over-priced ~2.3x on
both (151 s / 162 s for 67 s) — the rest of the cost model on a debug build,
not this credit.

Tests: `a_peer_handed_a_prompt_is_credited_its_opening_while_it_holds_the_model`
(scheduler: the slower of two warm holders wins a 4,000-token prompt it holds;
no credit once it holds part of the model), `a_peer_is_credited_the_opening_it_was_handed`,
`a_belief_ages_out_and_a_failure_forgets_it`, `a_growing_conversation_is_one_belief`.

## Where a peer's models RUN is not whether it owns a card (2026-09-26)

**`NodeCapability::models_run_on_card`** — a card AND no stated system-memory
budget. A node told to use its processor (`gpu_layers = 0`) still gossips its
card (the card is real) and states `ram_model_budget_mb`, which a node running
on its card leaves `None`. `memory_for_model_layers_mb` already read it that way
(the RAM budget first); the three routing readers that ask "does this peer's
work run on its card" did not: the prefill prior (`has_gpu` → 40x vs 4.5x, so
such a peer read prompts ~9x cheaper than it can), the KV booked by
`record_peer_commitments` (card-sized, with the f16 mirror), and
`gpu_vram_available_mb` (card room the hand-off gate would weigh). All three now
ask the predicate. A node predating the budget field is read as before.
Test: `a_peer_running_on_its_processor_is_not_priced_as_its_card`.

Not established as the field report's cause — its peer's configuration is not
known — but it is one way a chain is priced far below what it delivers, and
847 s against a price that preferred the chain says something did.

## A whole model on one peer is handed over, never driven token by token (2026-09-27)

**A plan `remote_generate::eligible` accepts is the hand-off's.** The n-gram loop
(`ngram_only_spec`), which `execute_distributed` tries first, asks that same
predicate and stands aside for every plan it accepts.

**What it replaced.** The n-gram loop ran first on the premise, stated at the
call site, that the hand-off gives "one token per round trip". It never has:
the peer decodes on its own card — speculating there with its OWN n-gram lookup
(`model_worker::ngram_spec_eligible`), where a miss costs nothing on the wire —
and streams the reply. The loop pays one round trip per round. Its payoff gate
switches it off after one poor request, so the cost fell on the first such
request after every restart (and nodes auto-update). Measured on the live node,
one peer (RTX 4050, Belgium) 443 ms away, 2026-09-27: the loop 0.78 tok/s
(1283 ms/token, llama-3.1-8b), the hand-off 10.64-11.15 tok/s (qwen2.5-coder-7b);
the project's own 2026-09-20 figure is 6.57 tok/s by hand-off at 1043 ms.

**And the hand-off asks privacy the way the plan was made.**
`remote_generate::eligible` now reads `encrypted_pipeline_for_request`, the form
the scheduler plans with. With the model-only form, a `swarm_route` override
that released this node's shards (so the automatic default stepped aside and the
scheduler planned a single remote segment) had the hand-off refuse that plan,
and it ran as a per-token pipeline instead — only override requests could reach
it, but they are how spread is benchmarked. An explicit per-model or global
privacy setting still refuses. `distributed.rs`'s local-embedding check still
asks the model-only form deliberately: embedding locally before a remote
segment 0 is harmless defence in depth.

**Unblocks FUTURE_WORK #10's peer credit**: the stated reason peers' prefix
caches are not priced is that the n-gram loop could take a peer's whole-model
plan and never read the cache. It no longer can (a draft-model path still can,
when one is configured).

Test: `a_single_remote_segment_is_the_hand_offs_and_never_the_n_gram_loops`
(`pipeline/local_generate.rs`) — with a positive control proving the loop's own
preconditions pass in the fixture, and a privacy control; each assertion goes
red with its half of the fix planted out.

**From the rules file (moved 2026-10-02):**

**`remote_generate::eligible` is the single answer to "does the hand-off take
this plan"**, and the n-gram loop — tried first — stands aside for every plan it
accepts. The peer speculates on its own card, where a miss costs nothing; the
loop pays a round trip per round (0.78 tok/s against the hand-off's ~11 from one
peer 443 ms away). It asks privacy with `encrypted_pipeline_for_request`, the
form the plan was made with.

→ `docs/invariants/scheduling.md` § "A whole model on one peer is handed over"

## A node is ranked on warm work only — including this node (2026-09-27)

**`PeerSpeed::ranking_ms_per_layer` falls back to the WARM prefill coefficient
(`warm_prefill_ms_per_layer_byte`), never the one that sizes timeouts.** The
timeout coefficient keeps cold samples on purpose — pessimism about how long to
wait is safe — and ranking read it as its fallback when no decode step had been
timed. The map is keyed by node id, THIS node included.

**Measured, and root-caused to seven figures** (a `root-cause` agent, CAUSED):
right after a restart, one 3-segment plan ran a 1-layer local segment that paid
for spawning a worker and loading the model — 2448 ms carrying 204,564 bytes.
`2448 / (1 × 204564) × 16384 = 196.0659`, and the next candidate line priced
this node `observed_ms_per_layer=Some(196.06593)`: an RTX 3070 that decodes in
~1 ms a layer, priced 223.6× its advertised figure (`cost_compute_ms` 401,543
against 1,796). The agent found no unconfounded route it changed in the window
read — the one candidate was under a benchmark override — but the mechanism is
a 224× input to `vertex_cost`, with no local exemption.

**Why the fallback was the only source for this node**:
`forward_verify_through_segments`'s LOCAL branch — the default n-gram verify
path for a split this node coordinates — recorded nothing, while its remote
branch has recorded single-token verifies since gotcha #418. It now records
ours the same way (single-token rule included), so this node gets real decode
samples, which outrank the fallback.

Test: `a_cold_prefill_sample_times_a_load_and_does_not_rank` (the live
numbers; red with the fallback pointed back at the cold-inclusive coefficient),
beside `a_cold_segment_still_sizes_the_timeout`, which keeps the timeout half.

**From the rules file (moved 2026-10-02):**

`PeerSpeed::ranking_ms_per_layer` falls back to the WARM prefill coefficient; the
cold-inclusive one sizes timeouts only. The map is keyed by node id, ours too:
one cold 1-layer segment (a worker spawn) priced this RTX 3070 at 196 ms/layer,
224× its advertised speed. Every path that times a segment — local ones too —
records it (`forward_verify_through_segments` did not, for ours).

→ `docs/invariants/scheduling.md` § "A node is ranked on warm work only"

### A prompt is priced from warm forwards that carried enough to measure a rate (2026-10-16, FUTURE_WORK #287)

**`ModelSpeed::prompt_ms_per_layer_byte` is the one figure a prompt's price
multiplies** (`SharedState::observed_prompt_ms_per_layer_byte` →
`NodeCandidate::observed_prefill_ms_per_layer_byte` → `parallax::prefill_ms_per_layer`).
Three figures come from the same prefill samples and want different things:

| figure | reads | keeps |
|---|---|---|
| `prefill_ms_per_layer_byte` | the segment TIMEOUT (`predict_ms`) | every sample — cold ones are pessimism, which is safe for a wait |
| `warm_prefill_ms_per_layer_byte` | RANKING, converted to ONE token (`ranking_ms_per_layer`) | warm samples — a visit's fixed cost belongs in a one-token figure |
| `prompt_ms_per_layer_byte` | a prompt's PRICE, scaled to its tokens | warm samples of at least `PROMPT_RATE_MIN_BYTES`; a smaller one may only lower it |

**What it replaced.** The price read the timeout's figure. Report #037 (a
tester, v0.3.240): a 5-segment route through two peers was predicted
`predicted_ms=63844516` — `prompt:63277600`, 17.6 hours — and took 145 s.
Two errors multiplied: a cold sample is a model load (the reason ranking was
moved off it on 2026-09-27, above), and a forward of a few tokens to a peer
abroad spends its ~700 ms on the round trip and the peer's per-visit costs,
so ms ÷ (layers × 40 KB) is a "rate" a 3,700-token prompt multiplies by ninety.

**Why the threshold, and why it may still lower.** It is BBR's app-limited rule
turned over: a throughput sample that did not fill the pipe may RAISE the
estimate, never lower it; a per-byte TIME is the inverse, so a small forward may
LOWER the rate, never set or raise it. Same number as
`GOODPUT_SAMPLE_MIN_BYTES` (256 KB), for the same split — a sample is either
the round trip or the payload (§ "Latency wants an average; capacity wants a
maximum"). With no rate, the device-class prior prices the prompt, as for a node
never measured — pessimistic, never excluding.

**This node too.** A split's local segments have been timed under our own id
since `forward_through_segments_inner` began measuring itself, but the planner
gave the local candidate `None` behind a comment saying they never were. It
reads the same figure now; a model only ever run whole here (the `Generate`
path, untimed this way) keeps the prior.

**What it does not fix.** The figure is still a ratio through the origin: a
warm 256 KB forward to a far peer carries some of the visit's fixed cost, and a
prompt many times its size scales that too. A two-term prefill fit (fixed +
slope × bytes, as `LinearFit` does for decode) is the next step if a field
trace shows the price still well above the time on long prompts — collect
first (FUTURE_WORK #3).

Test: `a_prompt_is_priced_from_warm_forwards_large_enough_to_measure_a_rate`.

## A split between two machines that could each run the model is measured by restricting BOTH (2026-09-27)

**`swarm_route.pretend_peer_holds` — `{"<prefix>": "4-7"}` — plans as though
that peer held only those shards**, the peer-side sibling of
`pretend_local_holds`, same grammar and 400s (`route_override::parse_holdings`,
`parse_node_prefix`), applied in `gather_candidates` beside `excludes_peer`. It
only SHRINKS the candidate set, so the override's contract holds: the plan is
still the search's, over a smaller input.

**Why it exists.** The spread benchmark had two GPUs (RTX 3070, RTX 4050) that
could each hold a 7B/8B. Every unrestricted plan was one machine — the search is
right to prefer that — so there was no split to measure, and `exclude_nodes`
cannot help (excluding the peer removes it from the split too). Restricting us
to shards 0-3 and the peer to 4-7 leaves the split as the only route. Test:
`restricting_a_peer_to_the_upper_shards_forces_a_two_machine_split`, with the
control that restricting only ourselves does NOT produce that split; red with
the filter removed. ⚠ It is a measurement tool: a plan forced this way is
evidence about what a split COSTS, never about what the router would choose.

## State that belongs to an attempt is keyed by the attempt, never by the request id (2026-09-28)

### What happened

Found by the v0.3.212 default-decision rig (`split_rig.sh failover` with speculation across
computers on, qwen2.5-coder-7b + a qwen2.5-0.5b drafter on the coordinator), before release. B's
worker was killed at the start of its prompt pass; the plan had no standby, the request failed and
the router retried it — under the same request id, as it always does. Then, in order:

1. The failed attempt's drafter read-ahead (`engine_drafter::read_ahead`, a spawned task) was still
   waiting on the drafter's worker. The retry's read-ahead registered a reply channel for the same
   id on the same worker, which displaces the older one (`claim_response_slot`).
2. `ModelProcessPool::draft`'s wait saw its channel close and took it for a dead worker: it EVICTED
   the healthy drafter. `generate` had had the check that tells the two apart since gotcha #180; the
   forward, batch and draft waits never got it.
3. The stale read-ahead then finished, and its guard (`ReadAhead`, added the night before to stop a
   leak) ran `release_request_kv(request_id)` — which is a fan-out to EVERY worker. It emptied the
   retry's cache on the TARGET's own segment worker, so the retry's first guessed round was refused
   ("this computer no longer holds the conversation", position 557) and the salvage returned one token
   as a 200.

### The rule

Anything scoped to one attempt is keyed by that attempt. `engine_drafter::draft_key()` is a fresh
UUID per speculating attempt, shared by its `ReadAhead` and `EngineDrafter`; each releases only that
key when dropped, so no attempt's release reaches another's state or the target's. And
`ModelProcessPool::reply_channel_closed` is now the ONE place a closed reply channel is judged —
superseded → plain error, closed → evict — used by generate, forward, batch and draft
(`docs/invariants/memory.md`).

### Verified

The same rig on the rebuilt binary: 0 drafter evictions, 0 lost conversations, takeover reply full
length (119/120 rank-1 vs llama.cpp, = control). Test
`a_superseded_draft_leaves_the_worker_for_the_call_that_superseded_it` is red with the supersede
check removed. What remains is separate: with the drafter resident the plan still had no standby
(FUTURE_WORK #140), which is why speculation ships off by default in v0.3.212.

### What a change must keep

Before keying anything by `request.id`, ask whether a retry of the same request could hold it at the
same moment — and whether a release keyed that way reaches workers you did not mean.

**From the rules file (moved 2026-10-02):**

The router retries a failed request under the SAME id, and the old attempt's
spawned work can still be running. Anything scoped to one attempt — the split
drafter's cache is the worked example (`engine_drafter::draft_key`, released by
the attempt's own `ReadAhead` / `EngineDrafter` on drop) — gets its own key.
Keyed by the request id, the dead attempt's read-ahead displaced the retry's call
(which evicted the drafter) and its release, which fans out to EVERY worker,
emptied the retry's target segment: a one-token reply returned as 200 (#749).
**A release keyed by request id reaches every worker, not just the one you meant.**

→ `docs/invariants/scheduling.md` § "State that belongs to an attempt"

## A split none of which is ours is LED by its head, never driven from here (2026-10-02)

**The defect it closes** (FUTURE_WORK #143, measured 2026-09-28): a requester
holding none of a split model sat inside every token's round trip. The tail
answered it, and it started the next token at the head, so a Thailand request
split among European peers 20 ms apart ran at the Thailand↔Europe loop. On the
default n-gram path it was worse: a guess-check round is star-shaped
(`forward_verify_through_segments` visits each segment through the
coordinator), so a plan of N remote segments paid N of the requester's round
trips per round.

**The rule.** `remote_generate::delegation_eligible` is the single answer to
"is this request handed to its head to lead?" — several segments, none of them
ours, the head advertising `features::DELEGATED_SPLIT`, prompt privacy off, our
private mode off, no `swarm_route` override, no LoRA/vision/TP. It runs FIRST in
`execute_distributed`, ahead of DSD and the n-gram loop, because each of those
would put the loop here. The head runs it as an ordinary request of its own router
(`dispatch::remote_generate::handle_delegated_split`), so chaining, n-gram and
drafter speculation, the continuous stream and segment failover all apply there
unchanged — the drafter, the n-gram loop and failover are `PipelineExecutor`
methods and move with the loop.

**What the delegate must keep** — `RoutePlanOverride::lead_here`, read at
candidate gathering, the one place a candidate is admitted:
- **No peer leads.** `can_be_first` is cleared on every peer, so the DP's source
  filter, the greedy first-segment narrowing and `delegation_target`'s
  whole-model hand-off all refuse a peer for layer 0. The requester applied its
  plaintext-prompt trust bar when it picked the head; a different head would read
  the prompt unvetted, and a whole-model hand-off would delegate twice.
- **No peer stands by for segment 0** (`standby_may_take`), for the same reason.
- **Its local segments are the swarm's work** (`pipeline::worker_requester` →
  `Requester::Swarm`), within the contribution level. Four sites hardcoded the
  owner because the router used to see only this node's API requests; the
  verify helper now takes the requester as a REQUIRED parameter.

**One id per attempt, and the gap a recovery needs.** The hand-off travels under
its own id (`try_delegated_split` mints it; gotchas #180/#229): after a fallback
this node sends ITS forwards to the same delegate under the request id, and a
router retry may delegate again, so a cancel of the abandoned hand-off — sent on
every failure — and its late tokens must not be able to name either. Between two
tokens a delegated stream waits `INTER_TOKEN_TIMEOUT` plus one whole-model decode
deadline: the delegate may be waiting out its own segment's and failing over.
Under `lead_here` the greedy fallback REFUSES where it would otherwise fall
through (narrow-if-non-empty is right only where a refusal fails the request; here
the requester runs the plan itself), and `detect_tp_groups` admits to a layer-0
group only a peer that may lead.

**Reading a failure.** A delegate's error may name a peer IT chose, so the
requester retracts no claim and bars nobody on it, and the reply's pace is the
plan's, not the delegate's — no speed sample. Before any token arrives, any
failure falls back to running the same plan here (`try_delegated_split` returns
`Ok(None)`); after, it is the request's failure, as for a whole-model hand-off.

**What it costs, knowingly.** Petals keeps the client in the loop precisely so it
can replay a failed server's input history to a replacement (arXiv 2209.01188 §
3.2; its servers also push activations straight to the next server, our
chaining). Delegation moves that replay to the delegate: a failed C or D is
still taken over, but losing the DELEGATE mid-reply loses the request, exactly
as losing the one peer of a whole-model hand-off does. Parallax (arXiv
2509.26182) runs the loop as a ring with the client outside for the same latency
reason.

**Tests**: `a_request_delegated_to_us_is_led_here_and_never_handed_on`
(scheduler; fails with the `lead_here` clause removed — the peer then runs the
whole model), `a_split_none_of_which_is_ours_is_led_by_its_head_and_only_that`,
`a_delegated_requests_local_work_is_the_swarms`, and the wire pair in
`swarmllm-types` (`a_whole_model_hand_off_carries_no_delegation_on_the_wire`).
Rig: `examples/split_rig.sh remote` (`REMOTE_NODES=2|3`, `DELAY_A=ms`), A/B in
one binary with `SWARMLLM_DELEGATE_SPLIT=0`.

**From the rules file (moved 2026-10-02):**

**`remote_generate::delegation_eligible` is the single answer** — several
segments, none ours, head advertising `DELEGATED_SPLIT`, no privacy / private
mode / `swarm_route` / LoRA / vision / TP — and `try_delegated_split` runs FIRST
in `execute_distributed`. The head runs it through its own router under
`RoutePlanOverride::lead_here`: **no peer may lead or stand by for segment 0, and
nothing is handed on** (cleared `can_be_first`, read by every path that assigns
layer 0). Its local work is the swarm's: **`pipeline::worker_requester`**, never a
hardcoded `Requester::Owner`. A delegate's error blames no one; before any token,
the same plan runs here instead.

→ `docs/invariants/scheduling.md` § "A split none of which is ours is LED by its head"

## A requester guesses for the split its head leads (2026-10-20, #318)

**The gap.** A led split guesses with what the HEAD holds — usually nothing better than prompt
lookup — while the user's machine, which handed it over, often holds a small model of the family
whole. Driving the split from the user instead uses that drafter but puts the user↔head trip into
every pass. Measured on the remote rig (`~/swarmllm-318/drive318.sh`: Qwen2.5-Coder-7B over two
processor nodes, the requester holding none of it and Qwen2.5-0.5B whole, every link 206 ms):
led without guesses 1.27-1.31 tok/s, driven with the drafter (α 0.65-0.67, γ 2) 1.10. The trip
cost more than the guesses earned.

**The design.** The head leads, and waits for a guess only where the wait pays (below, 2026-10-21).
The requester follows the reply as it
streams in — it receives it anyway — and drafts ahead of it with its own drafter
(`pipeline::requester_guesses`: `EngineDrafter::follow` + `draft`, a chunk of `guesses_per_check`
at a time, kept up to four checks' worth ahead), sending each window to the head
(`SwarmMessage::RequesterGuesses { request_id: the hand-off's wire id, after, tokens }`). Each
pass the head's loop (`ngram_only_spec`) takes the newest window's guesses past its reply where
the window's overlap equals what it produced since `after` (`RequesterGuesses::guesses_after`),
else prompt lookup, else a one-token check. A late window costs rows, never a token: a guess is
kept only where the target's own sample equals it (SpecExec), at any temperature. A window stays
good across passes while the reply agrees with it, so the requester drafts past the reply rather
than once per pass. No published system does this (prior-art search 10-20); its parts are AMUSD's
rollback, PEARL's post-verify drafting and PipeInfer's handling of a stale run. "Speculation at a
Distance" (arXiv 2606.25091) prefers a drafter beside the target over a client's — the
alternative here is a head with none.

**What a change must keep:**
- **Wait for a window only where it pays** (2026-10-21, #318 (a); it was "never wait"). On the
  real TH↔AU run 65 of 109 passes checked one token: no fresh window was there, because the
  requester's drafter took 200-450 ms after a miss. Where neither the requester's window nor
  prompt lookup has anything to check, the loop asks the slot when the next window is due
  (`GuessSlot::window_due_in`: the requester's turnaround this reply — from the reply reaching a
  length to the window drafted from it arriving, the median of the last 8 — less the time since
  the reply reached its current length; `None` before any window arrived, or while the newest
  already answers the reply) and waits for it only while `W < E·(P_1 + α·P_g)/(1 + α·E) − P_g`
  (`ngram_only_spec::worth_waiting_ms`; `P_1`, `P_g` the fitted check cost at 1 and γ+1
  positions, `E = expected_tokens_per_round(α, γ)`), up to that bound, waking on the window
  (`GuessSlot::wait_for_window`, a `Notify` the offer wakes). The derivation: going now runs a
  one-token pass, during which the window lands, and it still counts at probability α (its
  first guess = the token that pass chose) — so the two choices compare as `E / (W + P_g)`
  against `(1 + α·E) / (P_1 + α·P_g)`. **The plan's first rule, `W < P·(E − 1)`, was too eager**:
  it treated the one-token pass as throwing the window away. TH↔AU's figures (`P_1` 550, `P_g`
  600, α 0.6, γ 2): ~220 ms allowed, not ~530. The bound is always under `P_1` (tested), which
  is what "the window lands during it" rests on. Unknown cost, γ = 0, or a requester whose waits
  ran out three more times than they paid (`GuessWaits::given_up`): no wait. No published
  system waits for a drafter a round trip away (AMUSD, PEARL, SwiftSpec keep it beside the
  checker; searched 2026-10-21). The completion line reports `guess_waits`, `guess_waits_paid`,
  `guess_waited_ms`, and why it did not wait: `guess_wait_asked` (passes with nothing to check),
  `guess_wait_untimed` (no window to time), and the recent medians `guess_due_ms` /
  `guess_bound_ms`. **It pays only with the requester NEAR the head and the tail far**: on the
  card-tail rig (10-21, every packet 206 ms, the requester as far as the tail) it never waited in
  ~60 one-token passes a reply — the bound was ~160-175 ms (`P_1` ≈ 348, a 4-guess check ≈ 440,
  α 0.62-0.68) and a window cannot come back sooner than a round trip. A/B
  `SWARMLLM_HEAD_GUESS_WAIT=0`; tests
  `the_head_waits_only_for_a_window_due_soon_enough_to_pay`,
  `the_bound_is_where_waiting_stops_beating_a_one_token_pass` (both policies simulated: waiting
  wins at 0.8× the bound and loses at 1.2×), `the_wait_ends_on_a_window_or_its_deadline`.
- **The reply's ids.** The head's stream carries text; a drafter reads ids. The loop records each
  token in the request's slot BEFORE it emits it (`GuessSlot::record`, the first token too), and
  the task streaming the reply attaches the new ones to the next `StreamingToken`
  (`reply_ids: Option<ReplyIds{from, ids}>`, `take_unsent`) — retained with it, so a resend
  carries them. The requester puts them together in order and drops ids past a gap
  (`ReplyIdFeed`): it follows the reply whole or not at all.
- **Shared noise.** The requester picks the seed and sends it with the hand-off
  (`DelegatedSplit::guesses = GuessesAsked { coupling_seed, prompt_tokens }`); every sample the
  head's loop draws for the reply — prompt lookup's rounds and one-token checks included — uses
  it, keyed by absolute position, so a close drafter reproduces the SAMPLE. Only where the head
  read the prompt as the same number of tokens; otherwise the guesses are checked uncoupled.
  Fixed guesses lose nothing to coupling: each is kept with its probability either way.
- **Trust.** A window is taken ONLY from the peer that handed the reply over (its authenticated
  peer id, recorded when the slot is opened) and only while the reply runs (the slot's guard drops
  with `handle_delegated_split`); at most `MAX_REQUESTER_GUESSES` (64) ids; never one that claims
  to follow more of the reply than was sent; a window that crossed a newer one does not replace it;
  each id is checked against the vocabulary before it is embedded (`requester_drafts`). A forged or
  wrong window can waste rows, never change a token — and only the requester's own reply's. The
  requester in turn takes no more ids than the reply's `max_tokens` (`ReplyIdFeed`), and its
  drafter refuses an id outside its own vocabulary (`EngineDrafter::draft`), which ends the
  guessing — a lying head only slows itself. Behind the head's vocabulary cut, every first
  segment refuses an id its embedding table has no row for (`model_worker::wire_token_ids`, the
  one reading of a forward's ids, single and batched): a card's row gather checks nothing, and
  its kernel's assert ends the worker for everyone on it. That gap predates the guesses — any
  coordinator chooses the ids it sends a first segment (security review of #318, 10-20).
- **The stream is HANDED its slot.** The requester chooses the wire id, so a second hand-off can
  name one already open here: `RequesterGuessSlots::open` refuses an open id (that reply goes on
  without guesses) and `stream_reply_to_requester` takes the slot as a required argument, never
  looking one up by the wire id — a lookup would hand the second requester the first reply's ids
  (found in the in-session review, 10-20). The loop finds its slot by the head's OWN fresh id
  (`for_led`).
- **Wire.** Bit 28 `REQUESTER_GUESSES`: the requester offers guesses and sends windows only to a
  head advertising it; the head attaches ids only to a reply that asked. The bit is NOT in
  `features::ALL`: a head advertises it only where it would open a slot —
  `daemon::state::requester_guesses::takes_guesses`, the one answer both ask
  (`health::monitor::advertised_features`). Advertised unconditionally, a head with speculation
  off made every requester load a drafter and read the prompt ahead for nothing (security review
  of #318, 10-20; test `a_head_with_speculation_off_does_not_advertise_taking_guesses`). New fields are
  `skip_serializing_if`, so an older peer sees the bytes it always did. The message is relayable
  like the hand-off it belongs to.
- **The head checks only as many guesses as pay for their positions.** A guess is free to
  the requester and NOT to the head: each checked position is a row through every segment. On
  processors a row costs about half a one-token pass (the 10-20 rig: one-token checks ~0.84 s a
  pass, checks of 3-4 guesses ~2 s, 59% kept — the guesses broke even with none). So the loop
  fits the check's cost to the positions it carried (`dsd_controller::CheckCost`, the line DSD's
  own controller fits) and the guesses' acceptance (`AcceptanceEstimate`), and asks
  `best_gamma_for_check` how many to check next — zero is an answer, and then the reply runs as
  a led split with none. Where the tail is a graphics card a row is nearly free and the count
  climbs to `guesses_per_check`. The fit is the head's own measurement, so the requester cannot
  push it. A/B `SWARMLLM_HEAD_GUESS_GAMMA=fixed` (always `guesses_per_check`); test
  `the_head_checks_as_many_guesses_as_pay_for_their_positions`.
- **Where it stands aside.** A continued reply (`InferenceRequest::continuation`) takes no
  guesses — its positions are not the ones the requester follows. A head whose owner turned
  speculation off opens no slot, and a head that takes none (that, or one running DSD with its
  own drafter) records no ids: the reply's first token reaches the requester without any, and
  the requester stops drafting there (`GuessingAhead::hear`) — a head that takes guesses records
  every token before it emits it, so its first always carries one. A head with its own drafter runs DSD first, as before. The
  prompt-lookup payoff gate is bypassed for a reply with guesses and the reply's result is not
  recorded into it: the payoff measures prompt lookup, not a model.

**Tests**: `a_window_is_taken_only_from_the_peer_the_reply_is_for`,
`an_oversized_or_impossible_window_is_refused`, `a_window_that_crossed_a_newer_one_does_not_replace_it`,
`the_reply_ids_go_out_once_in_order`, `a_slot_closes_when_its_reply_ends`,
`a_second_reply_under_an_open_id_gets_no_slot` (`daemon::state::requester_guesses`); `a_guess_outside_the_vocabulary_is_never_checked`
(`ngram_only_spec`); `the_reply_ids_are_put_together_in_order`, `the_reply_ids_stop_at_the_replys_length`,
`guesses_the_reply_agrees_with_are_kept_and_the_rest_dropped`, `a_window_is_cut_to_what_a_head_takes`,
`a_head_that_does_not_take_guesses_is_never_offered_them`, `a_head_whose_first_token_carries_no_ids_is_not_guessed_for`
(`pipeline::requester_guesses`); `a_forward_naming_a_token_the_table_has_no_row_for_is_refused`
(`model_worker`); the wire
pair in `swarmllm-types`. A/B in one binary: `SWARMLLM_REQUESTER_GUESSES=0` on the requester.

### A route instruction goes to the head with the request (2026-10-21, #318 (b))

**The gap.** Any `swarm_route` instruction switched the hand-off off
(`request_may_be_led_by_its_head`: "a peer delegated to us carries `lead_here`"), so a led split
could not be measured through chosen peers — the TH↔AU run above was led only because the
requester did not yet trust the far peer with the prompt, and its "driven" arm, sent with an
override, measured the driven path again.

**The design.** Only `lead_here` keeps a request here now. Any other instruction travels with the
hand-off (`DelegatedSplit::route` = `route_override::carried`, unchanged) to a head advertising
`features::CARRIED_ROUTE` (bit 30, in `ALL`), which plans under it read from its own side
(`route_override::led_override`): a `peer_holds` entry naming the head (the first, as everywhere) is
what the head pretends to hold; `requester_holds` — what the requester pretended about itself —
goes FIRST in the head's `peer_holds`, for the authenticated sender's id, so it decides for the
requester as it did there (a requester that said nothing is `Everything`); exclusions go on as
they are. **Every part only shrinks**: the entry naming the head stays in `peer_holds` too, since
a short prefix may name other machines, and dropping it would lift their restriction (test
`a_carried_route_never_lifts_a_restriction`). A route excluding the head, leaving it nothing,
naming a node by anything but a 4-64 character lowercase hex prefix, naming more than 64 per
list, or a range of shards that ends before it starts (the request API refuses one; the wire does
not, and one naming the head read as "holds nothing" — security review, 2026-10-21) is refused before the reply is registered (`DELEGATED_SPLIT_ROUTE_REFUSED`), and the
requester runs the plan it has, as for any refusal before a token. **One answer for "may this head
lead it"**: `remote_generate::features_a_head_needs` (`DELEGATED_SPLIT`, plus `CARRIED_ROUTE` for a
request with an instruction), asked by `delegation_eligible` AND by the planner pricing a chain as
led (`PromptPlan::head_features` → `RingFacts::leads_splits`), so a chain is never priced as led by
a head the hand-off then passes over. The head's `delegation: submitted` line says
`route_carried`. What it cannot do: make a head lead a model whose first layers it does not hold,
or reach any request but the one it came with. Tests `a_head_plans_a_carried_route_from_its_own_side`,
`a_route_the_requester_could_not_have_planned_is_refused`, the wire pair in `swarmllm-types`.

## A split checks a tree of guesses (2026-10-20, #134)

**The gap.** Over a long link a check costs a round trip whatever it carries, and a chain keeps
only the guesses before its first miss. Measured TH↔AU (2026-10-20, a 7B split between this
machine's RTX 3070 and a peer's RTX 3060, ~385 ms apart): no guessing 1.96-2.29 tok/s, rounds of
chains 3.6-8.9, the continuous stream 5.0-12.6 — every pass paid ~400-465 ms fixed and 4-18 ms
for each row it carried. A tree keeps the longest run along whichever branch the model takes, for
the same round trip (SpecInfer, SpecExec, Sequoia, EAGLE-2).

**The design.**
- **What a check carries.** Rows are `[kept tokens read again, the token the last check chose,
  the guesses]`; `LayerForward::draft_parents` gives each row's parent among the forward's rows
  (0 = the cache's last position, k = row k − 1). Trailer `0x10`, last in the frame, bound in
  the AAD, at most `draft_tree::MAX_TREE_ROWS` (512); a malformed one is REFUSED, never read as a
  chain — a tree read as a chain attends every row to its siblings and answers plausibly and
  wrongly. Sent only where `pipeline::tree_check_possible`: every peer segment advertises
  `features::DRAFT_TREES` (bit 29, in `ALL`) and the tail walks. A chained hop hands the shape on.
- **Every segment** reads it through `SplitModel::with_draft_tree`: row i rotated at
  `index_pos + depth(i)` (`layers::RowPositions`), attending the cache plus its own ancestors
  (`DraftTree::rows_for`), on standard attention (the card's flash kernel takes causality as a
  flag), never as a CUDA graph. A layer keeping keys by position or one state per request — a
  sliding window, another layer's cache, MLA, a recurrent state, a picture's spans — refuses it
  up front (`SplitModel::draft_tree_refusal` → `Validation` from `draft_tree_to_apply`), and the
  coordinator checks chains for the rest of the reply, every segment cut back to where it was.
  The families are asked first (`ModelArch::reads_draft_trees`, an allowlist: Llama, Qwen2,
  Phi3, Mistral, Glm4 — the target's and the drafter's, which grows its tree the same way).
- **The tail walks it** (`sampling::walk_verified_tree`): sample at the row before the guesses,
  descend to the child carrying the sample, repeat. A node's children carry distinct tokens, so
  "keep the child equal to the target's own draw" is SpecExec's rule along one path — exact at
  any temperature. Shared noise is keyed `index_pos + 1 + depth(row)`, as the drafter grew it
  (`base + depth(parent) + 1`).
- **After a check: cut and read again, never gather.** The drafter lays its likeliest path out
  first (`draft_tree::trunk_first`), so the kept path usually IS the first rows: every cache keeps
  the lead rows and the kept rows sitting in place after them (`TreeOutcome::rows_in_place`), is
  cut there by the next forward's `truncate_kv_to`, and the rest of the kept tokens head the next
  check (`TreeRounds::reread`). A gather would need a second trailer and a KV copy in every
  segment (FlowSpec broadcasts a retain set; more wire). While anything waits to be read again,
  the next round MUST be a tree round — a segment that stops reading trees then fails the reply.
- **The drafter grows it in one call** (`SplitModel::draft_tree_after`, the worker's
  `TreeAsk { rows, per_step, grow_for_ms, seed }` → `split::TreeGrowth`): best-first over path
  log-probability in `dsd_tree::STEPS` (8) forwards of the drafter, each expanding the best
  `per_step` unexpanded nodes and queueing their `TREE_CHILDREN` (8) likeliest followers; then its
  cache is cut back to the context. It answers tokens, parents and each node's path
  log-probability (`WorkerMsg::Drafted::scores`). **On a card it ranks the followers where the
  logits are** (`draft::likeliest_rows_where_they_are`, 2026-10-21): `k` picks of each row's
  maximum, each written over with −∞, and only those pairs come back — a Qwen step had read 8 ×
  152K logits (4.9 MB) to the host and scanned each row three times. Measured on the 0.5B drafter
  (`prefill_bench` `SWARM_BENCH_GROW=64,8`, one CUDA process): 64 rows from nothing 208 → 155 ms,
  on a seed 100 → 35 ms. Not where the noise is drawn per row (`coupled`, a sampled reply), nor on
  the processor, where one pass over a row is cheaper than `k`. A/B `SWARMLLM_TREE_RANK=host`.
- **The next tree grows while the check is on its way** (2026-10-20). Grown only before its check,
  a tree put the drafter on every round's path: on the rig, a processor drafter's ~1 s a round
  made trees 0.86× the stream, though they kept twice the tokens a round. So `dsd`'s tree round
  joins the check with a second drafter call (`tokio::join!`) that GROWS ON the tree the check
  carries (`TreeAsk::seed`: the seed's rows read in one forward, their followers queued without
  the children it holds — the queue one call would have had), best-first by path probability from
  the same root: a guess's chance of lying on the reply's path, in this check if it went out, else
  in the next. When the check answers, the part under what the round emitted
  (`GrownTree::after(kept, next, SEED_MOST)`, re-rated from its new root) is the next check's
  tree — with no drafting in its way. It is grown further first only where
  `TreeRounds::grow_before_the_check` says the tokens added outweigh the time added
  (`(1 + grown)/(check + grow) > (1 + seed)/check`). The call beside is given
  `BESIDE_SHARE` (0.85) of the check's predicted time less its recent overrun
  (`beside_budget_ms`); past it a step's guesses are added unread. The check's own time feeds
  `CheckCost`, never the round's. This is the tree version of Speculative Speculative Decoding
  (Saguaro, arXiv 2603.03251), which caches outcomes of a CHAIN and falls back to drafting on a
  miss; a tree holds the outcomes as its branches, and a miss (`seed_misses`) just grows.
- **What is sent is priced, row by row** (`draft_tree::worth_checking`): expected kept =
  Σ calibration · exp(path log-probability) over the top-n guesses by rating (closed under
  parents, so still a tree); cost = the check's fitted `fixed + (lead + n) · per_row`
  (`dsd_controller::CheckCost`, the line chain checks fit); the n maximising (E + 1) / cost is
  sent — none is an answer. The calibration is kept ÷ rated, an EWMA clamped to [0.05, 2], so a
  drafter that over-promises is trusted less. With no slope measured yet, every other round is
  capped at a quarter of the rows so the fit sees two sizes. The tree grown is
  `2 × recent sent`, in [16, 64]. A guess that would run past the reply's room is never sent.

**Measured — and why trees are on only where every segment runs on a card.** First with the
head's part on a processor (2026-10-20, `~/swarmllm-318/drafter318.sh`: a
7B split, B's parts 1-7 on this machine's RTX 3070, A's part 0 and the 0.5B drafter on its
processor, QUIC, every packet delayed for a 385 ms round trip; one CUDA release build per row,
three requests of 160 tokens, the second and third warm):

| tok/s, warm | 4-bit drafter | 8-bit drafter |
|---|---|---|
| tree grown only before its check | 4.20, 4.50 | 4.00, 4.10 |
| tree grown beside its check | 4.86, 5.06 | 4.88, 4.75 |
| continuous stream of chains | 5.03, 5.09 | 5.51, 5.52 |

Before #323's kernels (the drafter's steps 2.5-3.4x dearer) trees beside were 2.81 / 2.74 against
the stream's 4.12 / 4.09. The cheaper drafter gained trees ~80% and the stream ~10-24%, and the
stream still wins: a tree grown beside its check kept 2.9-3.3 guesses a round (warm; 3.5-4.0
grown before it) for one round trip, and still waited 180-245 ms a round for the drafter (385-494
before it), where the stream keeps several chains in flight and never waits for one. A row
there cost the check 10-19 ms on the head's processor. Replies scored against llama.cpp
(`score_trees.out` beside the rig): the stream arms 154-158 of 160 rank 1, worst rank 2.

Then on the REAL link with the head's part on a card (2026-10-21, `~/swarmllm-318/cardtree.sh`,
results in `~/swarmllm-318/cardtree/`): a 7B split, H on this machine (Thailand) holding parts
0-3 on its RTX 3070 and the 0.5B Q4 drafter kept on its processor
(`SWARMLLM_PROCESSOR_MODELS`, this boot's one-CUDA-process rule), the far half on a swarm peer's
RTX 3060 in Australia ~405-409 ms away, direct QUIC; R, a second node here holding only the
header, hands the split to H with the route pinned to those two machines (`R: nodes
970194f1,aa3d1e66` on every arm, the instruction carried, #318 (b)). One binary, the arms
interleaved over two rounds, two replies each:

| tok/s | continuous stream | trees (`SWARMLLM_DRAFT_TREES=1`) |
|---|---|---|
| 160-token reply | 5.05, 5.01, 5.02, 5.05 | 5.66, 5.74, 5.33, 5.59 |
| 28-token reply (ends at its stop) | 4.54, 3.78, 3.82, 3.81 | 3.99, 3.45, 3.52, 3.42 |

A row cost the check 2.3-3.2 ms over a 477-520 ms fixed cost, so a tree carried 28-37 rows
and kept 4.1-4.5 guesses a round (3.68 in one reply), against the stream's α 0.69-0.70.
Weighted by tokens (one long and one short reply) trees are ~6% ahead; over long replies ~11%.
**What still costs a tree round** is the drafter: a seed held the check's outcome in only 3-10
of 29-34 rounds (`tree_seed_misses` 17-20), and a miss grows the next tree from nothing before
the check goes out — 327-416 ms a round waited on the processor drafter, of a ~0.95 s round. A
short reply pays it from its first round, with no seed and an uncalibrated drafter
(`tree_calibration` 0.64-0.87 there against 1.20-1.48 over long replies): that is its loss. Replies
scored against llama.cpp (`score_open.out`): the stream 160 of 160 rank 1; trees 157-158, every
other token llama.cpp's second choice at a near-tie (largest gap 0.076 logits, once 0.345) — the
different shape of the same arithmetic (gotcha #370), not a different reply; the short replies
28 of 28 rank 1 in every arm (`score_notes.out`; prompt tokens 42 and 555, as the node counted).

Then LONGER replies on the same shape the same day (`~/swarmllm-318/longtree.sh`, one 400-token
reply of the open prompt, three per arm, the arms one after the other; results in
`~/swarmllm-318/longtree/`). A 1500-token ask was refused by the Australian peer before any of it
ran — "0 MB available for conversations… short by 14 MB": a reply's reserve is its prompt plus up
to 512 positions (`model_worker::REPLY_RESERVE_POSITIONS`), 1024 for that ask, and the peer's card
had room for one quantum only (warm-ups of 8 tokens passed; `longtree_1500/`). At 400:

| tok/s, 400-token reply | 1 | 2 | 3 |
|---|---|---|---|
| continuous stream (α 0.745) | 6.18 | 6.04 | 5.97 |
| trees | 5.05 | 6.40 | 4.82 |

The trees' replies kept 3.20 / 4.62 / 3.24 guesses a round over 95 / 71 / 94 rounds, 19-30 of
them seeded and 40-52 seed misses, waiting 289-331 ms a round for the processor drafter. The
stream grew with the reply (5.01-5.05 at 160 tokens → ~6.06 at 400; its first chunks are its
cost) and trees did not (~5.5 at both). Scored against llama.cpp (`replies400.jsonl` →
`score_open.out`): the stream 398 of 400 rank 1, trees 395-396, every other token a rank-2 near-tie
(largest gaps 0.191 and 0.121 logits).

**So trees are the default only where every segment AND the drafter run whole on a card**
(`pipeline::every_segment_runs_on_a_card`: this node by `ModelProcessPool::runs_whole_on_the_card`
— a live worker not short of the card — a peer by `NodeCapability::models_run_on_card`; the
drafter by `runs_whole_on_the_card` on its own worker, live once its prompt is read; logged as
`drafter_on_card` on `DSD: request complete`), and the stream elsewhere;
`SWARMLLM_DRAFT_TREES=1` / `=0` forces either (`dsd_tree::trees_wanted`). With the drafter on a
processor trees won at 160 tokens and lost at 28 and 400 — not a default. On a card the eight
forwards a missed seed waits for take ~40 ms instead of ~300; that shape is the default by
PROJECTION, unmeasured: this boot barred two processes on one card. **Owed before the release
that carries it: a card head with its drafter on the card, against the stream, at 160 and 400
tokens; if trees do not win there, the default goes off.** ⚠ Correction (the same day,
6057274f said otherwise): a card head's drafter is on the card only where the target's layers
leave the card room — a CARD plan is never cut to make it some
(`max_hostable_layers_beside_guest` returns `None` under `planning_on_card`, #275 (4)) — so with a
large model filling the head's card it runs on the processor, and that reply takes the stream.

**A missed seed is regrown only as deep as pays, and trees are the default only over a long
link (2026-10-21, #134 (a)).** The schedule model (`examples/tree_schedule_sim.py`, Sequoia's
positional acceptance, within ~10% of TH↔AU) priced each shape at 400 tokens: over 425 ms every
schedule lands within ~15% of the others (card drafter: tree rounds 7.18, chain stream 7.32,
stream of trees 8.00), but over 20 ms eight drafter forwards before a ~30 ms check put tree rounds
at 0.45× the chain stream (21.7 vs 47.6), and a regrow sized by what it buys recovered most of it
(41.0). `TreeRounds::grow_size` sizes it (1, 2, 4 or 8 forwards: the most tokens per ms of the
round, from the check's fitted price, the drafter's measured forward and `level_acceptance`, a
censored geometric estimate in which a walk that stopped at a leaf is no miss). Rig
`~/swarmllm-318/regrow318.sh` (QUIC netns, B: 7B parts 1-7 on this card, A: part 0 + the Q4
0.5B drafter on the processor, trees forced, 160 tokens; tok/s of the 2nd and 3rd replies, the
1st carries the load):

| link | sized regrow | eight forwards (`SWARMLLM_TREE_REGROW=fixed`) | chain stream |
|---|---|---|---|
| 20 ms | 11.90 / 11.12 (3.2 / 3.0 forwards a regrow) | 7.33 / 7.48 (8.0) | 15.84 / 16.24 |
| 300 ms | 5.54 / 5.40 (5.5 / 5.6) | 5.12 / 5.34 (8.0) | 6.00 / 6.03 |

+57% over the short link, +4% over the long one, and with the drafter on the processor the
stream still wins both — as the default already had it. Each new test fails with its mechanism
removed (no deepest-first choice, a leaf counted as a miss, no distance gate). **Trees are
therefore the default only where the farthest peer segment is `TREES_LEAST_ROUND_TRIP_MS` (300)
or more away** — near chains (#192) are exactly the short links where they lose. **The distance
is the least round trip measured** (`metrics.network_coord_samples`'s window minimum, as the
coordinate and the near chains read it), never `PeerInfo::round_trip_ms`: on the rig's first
request that smoothed COST read 2,500-5,000 ms on a 20 ms link (forwards up to 256 KB feed it,
transfer and a loading peer included, gotcha #446), so the first request of a short link would
have taken trees (`a_plan_is_as_far_as_the_least_round_trip_measured_not_its_smoothed_cost`, red
on the cost reading). Re-run on the rig (`~/swarmllm-318/distance318.sh`, default settings): the
first request read 22 ms over the 20 ms link (2,500 before), and nothing over the 300 ms link —
no sample yet, so not far — then 21 and 301. **The card-drafter rig (2026-10-22, the first boot
that allowed two card processes; `~/swarmllm-318/cardrig247.sh` on the v0.3.247 release binary)**:
a 7B split over the real TH↔AU link (shards 0-3 and the 0.5B fp16 drafter on this machine's card,
layers 14-27 on an RTX 3060 ~405 ms away), arms alternating by node start — stream
(`SWARMLLM_DRAFT_TREES=0`), trees (the default), stream, trees — four replies a length each,
`drafter_on_card=true` on every line. Trees: 8.80 / 9.53 / 6.68 / 8.63 tok/s at 160 tokens (mean
8.41; the 6.68 a first request) and 9.95 / 9.05 / 9.69 / 8.76 at 400 (9.36); the stream: 5.88 /
5.92 / 6.05 (5.95; its first request refused — gotcha #931) and 6.42 / 6.59 / 6.58 / 6.32
(6.48). **+41% and +44%: the default stands.** Tree rounds 26-70 a reply, 4.7-5.3 tokens kept a
round, 33-49 rows. The replies differ (trees "a device that maintains…", the stream "a household
appliance designed…") at a near-tie: scored against llama.cpp
(`examples/score_against_reference.py`), every reply is rank-1 at 394-397 of 400 tokens, worst
rank 4, largest gap 0.21 (trees) / 0.05 (stream). What could widen trees further: fewer seed
misses (10-35 a reply), a tree STREAM (several trees in flight; weighed in
`docs/plans/large_model_splits.md` § "Draft trees"). #275 (4) — a card plan leaving its drafter
room on the card — is built (§ "A split leaves room for the model that guesses for it").

**A tree step pays for its tree once, and grows wide beside its check (2026-10-22, Track A).**
Where a tree round's time went on TH↔AU (`~/swarmllm-318/cardfill275q8.sh`'s default arm, 192
checks timed at their receipts): a round of 608-654 ms was ~485 ms in flight and 130-168 ms (mean)
HERE, waiting for the drafter — a seed that missed the outcome on 32-51% of rounds, was sent as it
was on 12-23%, and was grown ~8 forwards before the check on the rest. `prefill_bench` on this card
(idle, live node stopped): a tree forward paid 6 ms (0.5B drafter, 8 rows: 14.9 against a chain's
8.5) to 10 ms (7B head, 52 rows: 34.0 against 24.4) over a chain, and made 48 host-to-card copies
a step — each layer uploaded the rows' positions and gathered its rope rows again
(`row_positions_on`, gotcha #758's kind of copy); width was nearly free (a drafter forward 17.4 ms
at 32 rows against 14.9 at 8). Two changes:
- **The rows' rope tables are uploaded and gathered once per forward and table**
  (`layers::TreeRows` in the thread-local `RowPositions`, `tree_rope_rows`; llama.cpp builds one
  `inp_pos` per graph the same way). A pool job carries the positions, never the cache (#919: a
  job must set what it reads). A/B `SWARMLLM_TREE_ROPE=per-call`, `treeprof2.sh`, min of 7,
  arms adjacent: a 0.5B tree forward makes 3 host-to-card copies (positions, mask, ids) and 535
  launches against 50 and 629; it takes 12.3 / 15.0 ms at 8 / 32 rows against 16.6 / 20.5 (chain
  7.9 / 10.9); the 7B head 15.8 / 22.0 / 33.2 at 8 / 32 / 52 against 16.5 / 23.6 / 35.8; a tree grown from nothing 118.5 ms against 145.7
  (64 rows, 8 a step) and 222.1 against 272.5 (416, 32). What remains over a chain at 52 rows
  (~9 ms on the 7B) is the masked standard attention.
- **A call beside a check grows 32 rows a forward, up to 416** (`dsd_tree::beside_growth`, was 8
  and 128; A/B `SWARMLLM_TREE_BESIDE=8,128`): the same ~380 ms grows ~3.5× the tree, so the
  outcome's subtree is there more often and larger. 416 + a full seed of 96 is the 512 one tree
  may hold; a tree also grown before its check stops there (`draft_tree_after`). The schedule model
  put it at +6% over 8/128 at 425 ms (`examples/tree_schedule_sim.py`, 7.46 against 7.06). **The
  real-link A/B is owed**: at 04:44Z 10-22 (`~/swarmllm-318/beside318.sh`, the plan pinned 0-3 here
  and 4-7 on aa3d1e66) the far peer refused every prompt with no conversation room on its card
  (`holders_declined`, "0 MB available for conversations"), the warm-up excepted.
- **Next: a tree that flows** — rows streamed as grown, the far end appending them to the tree it
  holds and answering each row's pick, the requester walking: +40-47% over tree rounds in the
  schedule model at 425 ms (`docs/plans/large_model_splits.md` § "A tree that flows"; the old
  +11% stream of trees restarted every check).

**What a change must keep:**
- **The shape is bound in the AAD and refused malformed** — a relay reshaping a tree changes
  which rows a guess reads.
- **One wrapper reads it** (`with_draft_tree`): the worker's check and every step of the
  drafter's growth go through it, so both see the same positions and mask.
- **Never a gather.** The cache after a check is a prefix of the forward's rows; anything else
  is read again.
- **The refusal is a type, and the fallback only where it can go on**: `Validation` before any
  layer runs, and chains only while `reread` is empty.
- **Priced from this request's measurements**, never a fixed size: on the real link a row is the
  hidden state crossing it (~3.8 KB at Q8 for a 7B), and on a processor stage a row is compute.
- **A tree's row positions are set on the thread that runs the layers, and SET — `None`
  included — for every job a pool runs** (`cpu_pools::with_the_callers_scope`, beside the
  attention override): a pool thread waiting inside one forward's parallel work runs other jobs,
  a second forward too, which read the first one's tree ("a forward of 1 rows inside a tree of 4
  positions" between tests sharing the decode pool; at the tree's own length it would have been
  rotated wrong, silently). **Its rope rows are made once per forward and table, and never
  carried into a job** — a job reading a tree of its own makes its own, and the caller's survive
  it (`a_trees_rope_rows_are_made_once_per_forward`).

**Tests**: `a_row_reads_the_context_and_its_ancestors_only`, `a_chain_is_the_causal_mask`,
`a_tree_grown_over_several_forwards_reads_its_cached_rows`,
`a_parent_after_its_child_or_a_tree_too_large_is_refused`, `the_kept_tokens_are_read_back_as_rows`,
`the_likeliest_path_is_laid_out_first`, `a_guess_is_checked_only_while_it_pays_for_its_row`,
`the_guesses_worth_checking_are_a_tree` (`draft_tree`);
`every_row_of_a_tree_reads_what_its_own_path_reads` (with a chain over the same rows as the
control, which must differ), `a_kept_path_read_again_continues_as_if_read_alone`,
`a_tree_grown_a_step_at_a_time_reads_its_earlier_rows`,
`a_model_whose_layers_cannot_read_a_tree_refuses_it_before_running`,
`a_tree_that_does_not_fit_the_cache_is_refused_and_the_cache_kept` (`split/tests/draft_trees.rs`);
`a_grown_tree_guesses_after_each_path_what_a_fresh_read_guesses`,
`a_tree_grown_on_an_earlier_one_is_the_tree_one_call_grows`,
`a_call_out_of_time_adds_one_step_it_does_not_read` (`split/tests/drafting.rs`);
`the_tree_grown_on_gives_the_next_check_the_part_under_its_outcome` (`draft_tree`);
`a_job_reads_its_callers_tree_positions_not_the_threads_it_runs_on` (`cpu_pools`);
`a_chain_shaped_tree_walks_as_the_chain_does`, `the_walk_follows_the_branch_that_carries_each_pick`,
`a_tree_no_honest_coordinator_sends_is_refused` (`sampling`); the `dsd_tree` tests (`a_call_beside_a_check_grows_wide` among them);
`a_trees_rope_rows_are_made_once_per_forward` (`layers`);
`a_tree_of_guesses_is_applied_only_to_a_check_it_fits` (`model_worker`);
`a_tree_shape_survives_the_wire_and_a_malformed_one_is_refused`,
`a_tree_shape_is_sealed_and_survives_the_encrypted_frame` (`network::protocol`). A/B in one
binary: `SWARMLLM_DRAFT_TREES=1` / `=0` (trees forced on / off; by default on where every segment
runs on a card, above), `SWARMLLM_TREE_GROW_BESIDE=0` (each tree grown only before its check),
`SWARMLLM_PROCESSOR_MODELS=<id,…>` (those models' workers kept on the processor — the drafter
here). **Under failure** (2026-10-21, `~/swarmllm-318/treekill.sh`: `split_rig.sh kill`, a 7B split
on the processor, trees forced, the tail's worker killed after 42 results): the attempt ended at a
check (`DSD: pipeline verify failed … worker closed connection`), the router re-planned once, and
the retry checked trees for the whole reply (177 rounds) — 200, 400 tokens, the opening once.
Tests for the default: `trees_are_wanted_on_cards_unless_the_setting_says_otherwise`,
`a_plan_is_on_cards_only_where_every_segment_is`,
`only_a_worker_wholly_on_the_card_runs_whole_on_the_card`. Rigs: `~/swarmllm-318/cardtree.sh` (the
real link, above), `~/swarmllm-318/trees318.sh` (both halves on the
card — two CUDA processes, the FECS shape: one came at its 5th arm, 10:36Z 10-20),
`~/swarmllm-318/overlap318.sh` (the far half on the card, the requester's part and drafter on its
processor, QUIC, every packet delayed).

## A tree that flows (2026-10-22, #134 (b))

**What it replaced.** A tree round sends one tree, waits the whole round trip for its walk, and
grows the next tree while it waits. On TH↔AU (~411 ms) a round was 608-654 ms, and the reply
advanced only as deep as one check's tree reached, once per round trip. In a flow nothing waits
for an answer but the walk: the drafter grows ONE tree, three levels of 8 rows a call
(`SWARMLLM_FLOW_LEVELS`, `SWARMLLM_FLOW_ROWS`), each call's rows go through this node's layers and
on to the far segment at once, and the
far segment answers every row with the token the model draws there. This node walks those draws
(`draft_tree::walk_draws`) and emits each token the moment the walk reaches it. The reply stalls
only where no row holds the model's draw — a miss — and a miss costs one round trip, where a
round costs one per check.

**The rule** (`pipeline::dsd_flow`):

- **One tree, kept alike in three places.** Every segment's cache keeps each row sent since the
  tree's start (`LayerForward::draft_tree_from`, trailer `0x11`), in the order sent; every forward
  carries the whole tree's shape (`0x10`), so a row reads the context and its own ancestors and a
  lost branch stays in the caches, read by nothing. This node's `FlowTree` and the drafter's
  worker (`SplitModel::grow_flow`, a `FlowGrowth` per attempt, kept in `model_worker`'s `FlowSessions`) change the rows only by `FlowOp`s
  (Push — a miss's row —, Copy — a copied run's row —, Advance, Rebase), applied alike, and by the
  drafter's proposals: a row of its tree the moment it is made (its next level reads it), pushed
  here in its order, no op.
- **The tail DRAWS, it does not walk.** One draw per row (`sampling::draw_each_row`), its noise
  keyed by `tree_start + 1 + depth` — the draw at a row does not depend on which forward carried
  it, so the walk is exact at any temperature. A sampler reading history (penalties) cannot be
  drawn so and keeps the rounds (`flow_possible`, `refuse_draws_with_penalties`).
- **A miss is a row of its own.** The draw no child carries is pushed as a row under the walk's
  row, the walk moves onto it, and the drafter grows under it with its next call.
- **A copied run is depth the drafter does not pay for** (`FlowRun::copy_ahead`). Where the
  reply's last tokens occurred before — in the prompt, then the reply's last 500 tokens, the
  n-gram lookup's own cascade (`ngram_lookup::copied_run`) — what followed them is pushed under
  the walk's row: through the rows the tree already holds (`copy_under`), new rows for the rest,
  once a walk position, sent before the drafter's next call and within the guard's live budget.
  Its length is how far back the match goes, at least the lookup's `num_pred_tokens` and at most
  `COPY_LEAD_MOST` (32) — SuffixDecoding's (2411.04975) α·p with α = 1: a copy the walk keeps
  following runs further ahead each time. **And it ends where the earlier occurrences stop
  agreeing** (#338, `ngram_lookup::agreed_run`): the matched n-gram's last 32 occurrences are
  walked together, each token the one most of those still following continue with (a tie to the
  most recent), and the run ends once that token holds under 67% of them — SuffixDecoding's
  `min_token_prob`. Offline on 400 SWE-agent sessions, capped as above, the reply tokens a copy
  covered stayed at 64.6% while copied rows fell 3.29 → 2.36 a reply token (−28%); the Rust
  function replayed over 100 of them gives the script's figures (`docs/plans/faster_than_local.md`
  § 7.6). The same lookup feeds the rounds' n-gram guesses and one computer's own
  (`model_worker::LocalSpec`). A/B `SWARMLLM_COPY_AGREEMENT=0`. Its rows reach the drafter's worker as `FlowOp::Copy`,
  which rates each by the drafter's OWN log-probability of that token there (the least it listed
  when it did not list it), so a wrong copy's followers rank where the drafter would have put
  them and never crowd out its own guesses. Between this node and its own worker only — no wire
  change. Off with `inference.ngram_lookup_enabled` (read live); A/B `SWARMLLM_FLOW_COPY=0`.
- **Where this conversation has no match, the owner's EARLIER replies are looked in — only
  for the owner, and only those every peer of the split has seen** (#337,
  `inference::owner_replies`). An agent's opening calls repeat its earlier sessions; offline on
  400 SWE-agent sessions a store of earlier replies took a copy's share of a session's first
  three replies from 28% to 60%, and of whole sessions from 64.6% to 73.3% at 256K tokens a
  model (11.2 of an unbounded store's 11.7 points; `faster_than_local.md` § 7.6). Three rules
  make it safe to keep:
  1. **Only the owner's request reads it or is kept in it.** `OwnerReplies::for_the_owner` is the
     one way to a store's contents, and asks `InferenceRequest::requested_by` (local only,
     `#[serde(skip)]`, a REQUIRED argument of `InferenceRequest::local`; `Owner` only where the
     API saw this node's key and no `x-swarm-forwarded`, `api::requested_by`) AND
     `worker_requester` (a split led here for another node is the swarm's). A store's matches show
     in a reply's speed — REST's datastore leaked through it at over 25 tokens a second
     (2411.01076) — so a pool member's forward and a peer's hand-off are never the owner. One key
     per node, and its holder already reads every stored Responses reply: the key is the boundary.
  2. **A peer is never shown a copy of text it has not already seen.** A flow's copied rows are
     SENT: the first layers run them, the tail draws them, so a copy from another conversation
     shows that conversation's fragments to today's peers (and ~60% of copies are wrong, so it is
     shown text the reply never contains). Each kept reply carries the peers that served it, and
     a run is copied from it only where the split's peers are ALL among them. Its last token is
     not kept: the tail drew it, the peers before the tail may never have been sent it. **And in
     the form they saw it** (`Exposure::covers`): a run goes to a peer as TOKEN IDS only from a
     reply that peer knew as ids. Prompt privacy treats ids leaving as worse than states, and a
     reply kept under it would otherwise go out as ids once the setting was off (security
     review, 2026-10-24). Each path says who sees what twice — for a guess it would SEND (naming
     more peers or ids than it reaches is safe) and for the reply it KEEPS (naming fewer is
     safe): the flow (`dsd_flow::flow_exposure`) sends ids only to its first peer where this node
     runs none of the first layers (`shape.near` empty; the tail gets `drafts: &[]`), so both
     are that. The rounds (`PipelineExecutor::exposure_of_a_check`, read LIVE each round —
     failover rewrites the segments — and a reply whose peers changed is not kept) send a
     guess's ids to a remote first segment, to the tail that walks them (`draft_tokens`) and,
     where a check may chain, to every hop of the run ending at the tail (`verify_onward` carries
     them) — every remote segment after the last one run here, since a chain never crosses this
     node (`[me, a, me]` sends `a` none); a reply is known as ids only by a remote first segment
     (every token but the last is its input — not under prompt privacy) and a tail that WALKS
     (`peer_walks_at_tail`: sent every guess, it draws every other token, the first on the
     prompt pass), never a middle peer, nor an older tail that returns distributions (security
     review, 2026-10-24). So a chained check through a middle peer copies nothing from the
     store; a tensor-parallel group's machines count as reached, never as having seen. All three daemon loops read it at their one choke point, `speculative::ngram_lookup_drafts`
     (a REQUIRED `earlier` argument): the n-gram loop (`ngram_only_spec`), DSD's rounds (the part
     of the reply they write; a flow keeps its own, the stream's is not kept) and the draft-model
     loop.
  3. **Memory only, bounded** (256K tokens and 4,096 replies a model, 4 models, ~5.5 MB a
     model; replies of one split share one record of its peers): never saved, shared or
     announced. In a pool every device shares the key, so a pool member asking THIS node directly
     is its owner here; it could already read the node's stored Responses replies.

  The lookup is llama.cpp's for its cross-run cache (`common/ngram-cache.cpp`) at its LAX
  thresholds — the last 32 occurrences vote, the leader must hold half, a bigram must have been
  seen twice, and a longer n-gram that fails ends the lookup (no fallback): the vote cut the
  store's rows 24% for 0.2 points. #338's agreement cut, which pays on the conversation's own
  text (and runs there since 2026-10-27), saved no rows on the store and cost 0.5 points
  (`combo.py`), so the store's run is the newest leading hit's continuation. A run never crosses a reply's end, nor a match its start.
  Logged `earlier_runs` / `earlier_rows` / `earlier_walked` on the flow's line, `earlier_rounds`
  (and `earlier_checked` / `earlier_kept` in the n-gram loop) on the rounds'. A/B
  `SWARMLLM_FLOW_COPY_STORE=0`. Tests in `owner_replies` (subset rule, ids against states, owner
  only, vote, bounds), `pipeline::tests::a_checks_guesses_and_its_reply_are_counted_for_every_peer_they_reach`,
  `speculative::earlier_replies_tests`. **Rig (2026-10-24, one binary, 385 ms, the chat prompt
  three times, each a new conversation)**: with the store on, the third reply copied 29 runs —
  209 rows, 19 on the walk's path — from the second; with it off, 0. Each arm's first reply ran
  the chain stream (the drafter not yet on the card), which keeps nothing, so the second had
  nothing to copy. The replies' text varied between repeats in EVERY arm, the store's off ones
  too (greedy near-ties under the flow's batch shapes), so it is not the store's.
- **Several levels a drafter call** (`levels_a_call` 3 × `rows_a_level` 8): each level reads the
  last one's proposals in one forward and proposes after them. At ONE level a call a branch grew
  one row deeper a call and the walk with it — 0.63-0.74 tokens a call whatever the rows a call
  (8, 16, 32) or the live budget (32, 64, 128): the depth a call adds, not its rows, was the bound
  (gotcha #933). EAGLE-2 grows a draft the same way (one forward a depth, top-8, 5 deep; its
  `cnets.py`), and FlowSpec (2507.02620) grows depth 5 a step.
- **A rebase, not a cut per verdict.** At the row limit (`MAX_TREE_ROWS`, 512) or once the walk
  is `REBASE_PATH_ROWS` (128) deep, the walk's row becomes the new start: ONE forward cuts every
  cache back to the old start and reads the walk's path there, then the live rows
  (`FlowTree::rebase_forward`). It is a stream restart (`forward_streams`): turns before it not
  yet run are skipped and refused, never timed out. Answers from before it still count — mapped
  onto the kept rows through the last rebase's remap; anything older, and an old epoch's
  refusal, is dropped (`stale_answers`). **A rebase waits for the one before it to be ANSWERED**
  (`FlowRun::settled_base`): a restart is safe only where every turn it builds on ran
  (`forward_streams`' premise), and a rebase sent early could skip the one before it at the tail,
  leaving that tree's rows where its path belongs. A miss while the tree is full and a rebase is
  out holds its row (`bonus_waiting`, its token already emitted) until the answer.
- **The window**: 12 forwards out for one peer, 8 through a chain (a hop remembers 16 of a
  stream's forwards on the rule that a coordinator keeps half that out). A serving node holds 16
  of one stream (`MAX_STREAM_CHUNKS_HERE`).
- **Live rows ≤ 128** (`SWARMLLM_FLOW_LIVE`): the walk's row and those under it.
- **Gated** on `features::DRAFT_TREE_FLOW` (bit 31) for every peer of the run, beside
  `DRAFT_TREES`; off with `SWARMLLM_TREE_FLOW=0` (the tree rounds), on by default wherever trees
  are checked from the start (`dsd_tree::trees_wanted`).
- **A failed flow finishes in tree rounds** where the peers answered (`rounds_may_finish`:
  `Validation`, `Inference`, `Internal` — a stale reading of a peer's features, a refusal, a
  fault of the flow's own): every forward still out is answered first, every segment is cut back
  to the last start they ANSWERED at and the reply past it is read again in front of the next
  check (`rounds_take_over` → `TreeRounds::reread`, a kept path's own rule), and the drafter
  follows the reply (`EngineDrafter::follow`). ⚠ Past the tree's start a cache holds rows in the
  order SENT: the first take-over, cut to the reply's length as the chain stream's is, read those
  rows as the reply and its next token ranked 8th against llama.cpp (logit gap 4.1, `fault` arm,
  2026-10-22, gotcha #932). A peer
  gone silent is the router's to re-plan, never waited on twice. Fault switch:
  `SWARMLLM_FAULT_FLOW_FAIL=<n>`.
- **A run that finishes HERE flows too, its rows drawn by this node's last layers** (#369,
  2026-10-29). "Start and finish on this computer" is automatic wherever this node holds both
  ends, and that shape ran tree rounds — 1.6-2.2x slower than the flow at the same split on TH↔AU.
  Now the peers are sent hidden states and the tree's shape only (no sampler, nothing to draw) and
  answer hidden states; this node runs its layers after theirs on each answer
  (`FlowRun::finish_here`), the last of them drawing every row with the walk the far tail would
  have had. What a change must keep: (1) **those layers run each answer in the order its forward
  was SENT** (`TurnOrder`), the order every peer ran them (`forward_streams`) — the worker refuses
  rows that do not continue its cache (`model_worker::draft_tree_to_apply`), and several answers
  ready at once come out of the forwards out in any order (55 of 285 a reply were held); (2) an
  answer the peers RAN is run here whatever its epoch, so the two caches hold the same rows in the
  same order and a rebase cuts both alike — a refused one (a turn a rebase skipped) is run nowhere;
  (3) a rebase settles only once this node's layers ran it, so the rounds' take-over cut
  (`settled_base`) is one every cache holds; (4) the shared noise goes to this node's own last
  layers whatever the peers advertise. A check's recorded cost includes those layers. Measured in
  one binary on the real link (14B, 0.5B Q8_0 drafter, this machine 25 of 48 layers): tree rounds
  5.74-6.99 / 15.03-15.52 tok/s prose / code, the flow finishing here 7.44-7.80 / 17.31-18.88, the
  default shape at the same split 8.53-10.18 / 21.90-22.33 (`faster_than_local.md` § 9); a second
  run 7.52-8.28 / 18.49-20.06 against 7.74-8.05 / 19.41-27.88. The default shape stays ~1.1x ahead
  on prose and ~1.2x on code, inside its own code spread; misses are alike in both shapes and
  mostly wrong guesses, not the walk outrunning the tree — FUTURE_WORK #369. A/B
  `SWARMLLM_FLOW_FINISH_HERE=0`.

**Measured.** *Correctness, on the processor* (2026-10-22, `flow318.sh CPU=1 LEN=64`, a 557-token
prompt, 7B split 0-14 | 14-28 over QUIC at 385 ms, greedy, two replies an arm): the flow ran 64-65
forwards of 10.4-10.9 rows, 1 miss and 1 rebase a reply, no stale answer and no refusal; its two
replies agreed with each other and the tree rounds' two with plain rounds' two, the two sets one
token apart (" cycle" against " process", token 46). Scored against llama.cpp
(`score_against_reference.py`): every token of every reply rank 1 but for two near-ties — token 5
(" by", gap 0.051, in all of them) and token 46, where llama.cpp itself picks the flow's " cycle"
(gap 0.011). A tree row is read on standard attention (`ForceStandardAttnGuard`), whose sums move
with the forward that carries it (#313), so arms that group rows differently may part at a
near-tie.

*Speed, on the card* (2026-10-22, `flow318.sh LEN=400 ARM_REPEAT=3`; the drafter, this node's
half and the far half all on one RTX 3070 Laptop; the 0.5B Q8_0 drafter; each arm's first request
runs before the drafter is on the card and is left out — warm replies):

| arm | tok/s | forwards a reply | rows a forward | tokens a drafter call | drafting / near s |
|---|---|---|---|---|---|
| tree rounds | 8.03-8.27 (4) | — | — | — | — |
| flow, 1 level × 16 (the first build) | 8.25-8.63 (4) | 624-657 | 16.8-16.9 | 0.61-0.64 | 24.3-24.6 / 21-22 |
| 2 levels × 8 | 8.60-9.21 | 459-468 | 17.4 | 0.86 | 29-32 / 12-13 |
| **3 levels × 8 (default)** | **8.99-9.44 (4)** | 342-380 | 24.5-25.6 | 1.05-1.17 | 30.5-32.7 / 10.4-11.5 |
| 4 levels × 8 | 8.92-9.04 | 296-302 | 33.0 | 1.32-1.35 | 32.3-32.5 / 9.3-10.7 |

A level adds ~0.24 tokens a call for ~27 ms of drafter (8 rows of a 0.5B on standard attention,
never a CUDA graph, the card shared with both halves) — about the rate the flow already runs at,
so it levels off past three. The requester is compute-bound: waiting under 1.2 s of 42-47, the
drafter ~72% of the rest; misses 16-24 a reply, rebases 15-23. The three-level replies against
llama.cpp: every token rank 1 or a rank-2 near-tie (6-8 a reply, the largest gap 0.13). **The
drafter's call beside this node's layers was a loss** (built, measured, removed: 7.53-8.24 tok/s;
drafting 31 → 46-49 s, near 10-11 → 19-21 s): two processes on one card time-slice it (gotcha
#759), and in the field the drafter and this node's half share a card too — removed. The `fault`
arm still finished in rounds (11 tokens read again, every reply whole). Short of the 1.3× exit;
what is left is the drafter's level — a graph saves ~9 ms of a 0.5B's decode step (gotcha #233),
and an 8-row tree step costs ~12 ms uncaptured on an idle card (§ "A tree step pays for its tree
once") — and the real link, where the far half is not on this card. **The level is one kernel and
one CUDA graph since 2026-10-22** (`docs/invariants/inference.md` § "A tree level on a card"): on
this rig the drafter's time fell ~9% and this node's half rose by about as much, the three
processes sharing one card.

**On a thin uplink: the rows that DIE are the cost, and a guard holds them down** (2026-10-22,
`FlowGuard`). With the netns rate limit on node↔node traffic only (`RIG_RATE`, the worker IPC is
untouched), the fixed flow lost to the tree rounds — what it sends is ~26 rows a token at 8 rows a
level, against the rounds' ~6. Every token the walk advances kills the rows not under it and the
drafter grows the tree back: a WIDE call grows rows the next advance kills before they are read,
a SMALL live budget leaves the walk too little tree ahead. Fixed settings, warm replies, 385 ms:

| link | tree rounds | 8×128 (default) | 4×128 | 2×128 | 8×64 | 8×32 | 8×16 |
|---|---|---|---|---|---|---|---|
| 5 Mbit | 6.77-7.05 | 4.49-4.79 (26-28 rows/token, waiting 53 of 82 s) | 4.85-6.17 | 6.31-6.71 (~10, near-bound) | 5.70-5.77 | 6.25-6.52 (waiting 45 s) | 4.47-4.48 |
| 10 Mbit | 7.70-7.85 | 7.65-7.69 | 8.56-8.64 | 7.07-7.17 | — | — | — |

So no fixed point wins on every link — the guard is a congestion window in rows, LEDBAT++'s
(draft-irtf-iccrg-ledbat-plus-plus-06: a 60 ms queueing-delay target over the least delay seen,
the least of the 4 latest samples, `W -= min(W/2, W·(queue/target − 1))`): a queue narrows the
rows a level first (8 → 2) and only then the live budget (128 → 32); room grows the budget back
first, then the levels; one cut per window of forwards (an answer counts only if its forward left
after the last cut, TCP's one reduction a window); rises spaced a least delay apart; the state is
kept 10 min per (model, machines) for the next reply (RFC 6817's base-delay window). ⚠ **An
answer is timed on ARRIVAL, on a task of its own** (`Arrival`): the first build read each delay
when this node got round to it — between its drafter calls and its own layers' forwards, ~100 ms
of its own work that drifts against the answers — and cut an UNTHROTTLED link 8×128 → 2×32
(24 cuts, queue up to 141 ms, 7.8-8.0 tok/s against 8.8-8.9 unguarded). RFC 6817 § 3.4.1
timestamps on arrival for the same reason; the min-of-4 filter does not cover a delay every
sample carries. The rounds' check cost (`reply.check`) reads the arrival too. Measured with
arrival timing (`guardrig2.sh`, one binary, A/B `SWARMLLM_FLOW_GUARD=0`, two warm replies an arm):

| link | guard on | guard off | where it settled |
|---|---|---|---|
| 5 Mbit | **7.44-7.50** | 4.73-4.80 (waiting 54-55 s) | 2×32-38, 16-22 cuts |
| 10 Mbit | **8.52-8.82** | 8.23 | 2×113 → 4×128 |
| none | 9.02-9.12 | 9.59-10.19 | 8×128; one reply cut 6 times in bursts of the far half's own queue (answers at 630-820 ms against a least of 462: the far half shares this rig's card), the other never |

With no limit, the reply that never cut ran 9.12 against 9.59-10.19 guard-off — the guard's code
did nothing there, so that gap is the arms (the guard-on arm always ran first). The alternating
check (`copyrig.sh`'s chat pass, below; copied runs on in both arms) found no gap: guard off
9.19-9.82 then 8.79-9.41, guard on 9.09-9.58 then 8.83-8.98 tok/s, the guard-off arm first in
one pair and second in the other.

**Copied runs** (2026-10-22, `~/swarmllm-318/copyrig.sh`, one binary, 385 ms, no rate limit, A/B
`SWARMLLM_FLOW_COPY=0`, arms alternating, two warm 400-token replies an arm, request-level tok/s):

| prompt | copy on | copy off | copied rows → walked a reply | the drafter's time a reply |
|---|---|---|---|---|
| a code edit whose reply copies a 135-line file | **43.5-46.7, 42.1-42.4** (flowed 47.5-53.2) | 22.6-24.5, 23.3-24.1 | 259-291 → 232-263 of 400 | 3.4-4.4 s vs 8.9-10.2 |
| the rig's chat prompt | 9.09-9.58, 8.83-8.98 | 9.20-9.66, 8.72-9.00 | 292-369 → 1-6 | 24.6-27.6 s vs 26.0-28.5 |

Every edit reply was the same text with copies on and off — a copy changes what is checked,
never what the tail draws — and llama.cpp ranks it first at all 400 tokens
(`score_against_reference.py`, the 1,513-token prompt the node counted). On chat almost every copy was wrong and cost nothing measurable: its
rows die like any unread branch, and the drafter ranks their followers by its own guesses. With
the far half on this rig's card, the edit's denser forwards queued the far half's answers
(200-360 ms over the least) and the guard narrowed to 3-6×128 with copies on and off alike.

**On the REAL link** (2026-10-23 01:51-02:16Z, `~/swarmllm-318/flowreal248.sh`). This machine ran
layers 0-13 of Qwen2.5-Coder-7B Q4_K_M and the 0.5B fp16 drafter, both on its RTX 3070 Laptop.
aa3d1e66 ran layers 14-27 on an RTX 3060, on .248, at a 428-437 ms least round trip. Greedy, the
rig's chat prompt (42 tokens); each node start asked 160, 400, 160 and 400 tokens; eight starts,
arms alternating; Windows up 25.4-25.8 h. Decode tok/s, four replies an arm and length:

| arm | 160 tokens | 400 tokens |
|---|---|---|
| flow, the .248 release binary | 12.21-16.14 (mean 13.45) | 14.03-15.03 (mean 14.56) |
| flow, main `e8e649d0` (flash context + heads-first) | 13.01-16.53 (13.95) | 14.35-15.23 (14.63) |
| tree rounds (`SWARMLLM_TREE_FLOW=0`) | 10.46-11.90 (11.37) | 11.04-12.52 (11.57) |
| chain stream (`SWARMLLM_DRAFT_TREES=0`), one start | 6.18-6.24 | 6.78 |

The flow beats the tree rounds by 18% at 160 tokens and 26% at 400, and the chain stream 2.1×. The
same link with .247 on 10-22 gave trees 8.41 / 9.36 and the stream 5.95 / 6.48 (§ drafter room), so
the default is 1.6× faster at 400 tokens. ⚠ The other stream start (16.3-23.1 tok/s after a 42.8 s
first token) was not a split. A peer holding the whole model joined after the rig built its exclude
list and was handed the reply whole (gotcha #940), so that start is left out.

*Where a flowed reply's time goes* (the `DSD: a tree that flowed is complete` line, 400-token
replies). On .248, of 26.8-30.9 s: drafting 15.1-17.4 s (55-56%), this node's half 8.2-9.2 s
(29-30%), waiting 3.5-4.3 s (13-14%). That was 476-541 forwards of 23.5-24.2 rows, with 12-17
misses and 22-24 rebases a reply. On main, drafting took 30.4 ms a forward against 32.0 (−5%) and the
near half 15.2 against 17.0 (−10%). Tok/s stayed the same because main ran MORE forwards (533-549
against 476-541) for the same 400 tokens: tokens a forward fell from 0.78-0.84 to 0.73-0.75. **So
the time split misleads: a cheaper requester grows more rows, not more reply.** What binds is how far
ahead along the model's path the tree reaches each round trip: ~6 tokens a 430 ms round trip. Yet a
forward adds three levels, and 9-10 forwards fit in a round trip.

Scored against llama.cpp, every reply has worst rank 2 (gap ≤ 0.238). The tree arms part from the
stream at token 4 (" device" rank 2, gap 0.076), a near-tie.

*The knobs, on the same link* (02:20-02:43Z, `~/swarmllm-318/flowsweep.sh`, main's build, arms by
env only, two starts an arm alternating, four replies an arm and length). Each request rebuilt
its exclude list and was kept only if the node logged a two-segment split through aa3d1e66
(gotcha #940); none was dropped.

| arm | 160 tokens | 400 tokens | 400-token replies: forwards, rebases, misses, waiting |
|---|---|---|---|
| **8 rows × 3 levels, 128 live (default)** | **14.00-16.78 (15.28)** | **13.84-15.37 (14.28)** | 510-569, 23-26, 14-19, 5.0 s |
| 256 live (`SWARMLLM_FLOW_LIVE`) | 12.22-16.07 (13.79) | 12.96-16.38 (14.53) | 475-578, 23-29, 10-19, 0 |
| 448 live | 12.48-14.38 (13.45) | 13.12-14.46 (13.75) | 598-637, 31-33, 18-23, 0 |
| 4 rows × 6 levels (`SWARMLLM_FLOW_ROWS`, `_LEVELS`) | 11.65-12.66 (12.22) | 12.24-13.19 (12.68) | 425-446, 19-20, 18-21, 1.0-1.4 s |

**No knob beats the default; the defaults stand.** A larger live budget ends the waiting but buys
no reply. The extra rows die, and with 448 kept the 512-row tree (`MAX_TREE_ROWS`) refills in 2-3
forwards. A rebase may only be sent once the one before it is answered, so rebases rise to 31-33 a
reply. Deeper, narrower calls cost the drafter 52-59 ms a forward against 29-31 and miss as often.

*Rows a level, again* (2026-10-23 12:55-13:14Z, `~/swarmllm-318/harness/realwide.sh`, the 656848b5
card build, greedy, 128 live, arms ABCCBA × 2): **16 a level** (37 rows a call, 47 ms busy — 21%
cheaper a row) 15.05-16.23 tok/s, waiting 7.6-9.4 s; **8 (default)** 15.57-18.21 unqueued, waiting
4.6-4.9 s (its first arm 11.9-12.3 under a far-end queue, guard cuts 14-22); **4**: one arm under the
queue (11.6-13.5), the other ran no flow (round trip unmeasured, gotcha #942). The default stands;
wider rows grow more rows a token and wait longer.

**The law this reads as: a flow advances about E (the tokens the drafter's live rows get right)
per round trip.** 14-15 tok/s × 0.45 s ≈ 6.5 tokens. The tree rounds on the same link get 6.3-6.8 a round
(59-63 rounds for 400 tokens, ~0.58 s each, with up to 416 rows grown beside the check). The flow's win is
entirely in the period: it pays the round trip and nothing else, as the formula at the top of
`docs/plans/large_model_splits.md` (tok/s = E_N / P) says. E grows only about logarithmically with the
rows. So the next factor has to come from E itself: a drafter that guesses better on the card,
copied text where the reply repeats (§ copied runs: 2× on an edit), or a shorter round trip (near
chains). Not from rows, levels or the window.

*The drafter, on the same link* (02:44-03:11Z, `~/swarmllm-318/flowdraft.sh` then `flowfmt.sh`,
main's build, arms by the one drafter H holds, alternating; every DSD line named that drafter with
`drafter_on_card=true`). 400-token replies:

| drafter (on the card) | tok/s | misses | drafting a forward | waiting a reply | tokens a forward |
|---|---|---|---|---|---|
| **Qwen2.5-Coder-1.5B Q8_0** (1.8 GB) | **16.30-16.77 (16.46)** | **10-11** | 25.0-25.7 ms | 4.5-5.9 s | 0.80-0.85 |
| Qwen2.5-0.5B Q8_0 (645 MB) | 14.74-16.38 (15.57) | 17-19 | 14.5-15.2 ms | 9.5-11.3 s | 0.73-0.80 |
| Qwen2.5-0.5B fp16 (1.2 GB, f32 on the card, #308) | 13.60-14.25 (13.98) | 17-22 | 28.0-28.6 ms | 4.5-5.6 s | 0.70-0.74 |
| Qwen2.5-Coder-0.5B Q8_0 (645 MB) | 11.96-12.20 (12.06) | 39-41 | 13.9-14.4 ms | 15.9-16.7 s | 0.66-0.68 |

At 160 tokens the order is the same: 16.33, 14.55, 13.87-14.19 and 10.97. The two copies of one
0.5B guess alike (17-19 misses against 17-22). The Q8_0 copy pays half the drafting, and that half
comes back as waiting, because the flow is already at its round-trip wall: +10-14% (15.57 against
13.61-14.16). The 1.5B misses half as often and wins although each of its forwards costs more,
exactly as the law above says. A miss costs about a round trip: the coder 0.5B's 40 misses at
~0.43 s are its 16 s of waiting. ⚠ **Every figure before this table used the fp16 0.5B**, a copy no
node picks over a Q8_0 one (`engine_drafter::largest_that_can_draft` sorts the largest model first,
then the fewest bytes a guess reads). With both held, a node takes the 1.5B, so the field already
runs the winner where the 1.5B is there. On this card that ranking is right; on a processor it is
unmeasured (FUTURE_WORK #134, device-aware drafter choice). The coder 0.5B lost on this chat
prompt; a code prompt is unmeasured.

**A small drafter may be HALF its target, where a card is in play** (2026-10-24,
`engine_drafter::within_reach` + `small_drafter_fraction`). The table's winner was the largest the
size rule then allowed — a quarter of the target by layers × width², set from Leviathan et al.'s
one-machine drafters, never measured here. Qwen2.5-Coder-3B Q4_K_M (151 M against the 7B's 360 M,
0.42; 1.96 GiB, under the fetch cap) was shut out. Same link and far peer (aa3d1e66, RTX 3060,
0.3.248), one binary (`~/swarmllm-318/f7/swarmllm` = `99bff582` + `SWARMLLM_DRAFTER_MAX_FRACTION`),
QUIC forced per arm and checked (`H: to the far peer over: 1 udp` every arm), greedy, both
drafters on this card, the DSD line naming each arm's drafter (`~/swarmllm-318/harness/realdraft3b.sh`
17:14-17:31Z 10-23, ABBAABBA × 2 × 400 tokens; the code edit `drafterval.sh` part 1, ABBA × 2 × 400,
`copy_edit_prompt.txt`; Windows up 10.5-10.9 h):

| drafter (on the card) | chat tok/s | chat misses | code edit tok/s | code misses |
|---|---|---|---|---|
| **Qwen2.5-Coder-3B Q4_K_M** | **16.93-19.79 (18.09)** | **0-5 (3.2)** | 62.12-64.34 (63.0) | 0 |
| Qwen2.5-Coder-1.5B Q8_0 | 12.84-17.72 (15.95) | 8-21 (10.8) | 64.29-66.30 (65.4) | 0 |

A flow call of the 3B costs ~33 ms against the 1.5B's ~25 (drafting 13.4-14.2 s of a ~21 s chat
reply, 404-427 forwards against 487-594). Where the 1.5B misses, each miss is a round trip and the
3B wins 13%; where neither misses — a code edit, 40-50 copied runs a reply — the extra drafting is
a 4% loss. Chat is the common case, so the half.

**Where it does NOT hold** — the rig (`~/swarmllm-318/harness/drafter318.sh`, QUIC netns at 5 ms,
Qwen2.5-Coder-7B split 0-3 | 4-7, arms c15, c3, c3, c15):

- **Both halves on this one card**: no room left for either drafter there, so both ran on the
  processor; the first 200-token reply took 38.2/38.7 s with the 3B against 32.6/31.0 s with the
  1.5B (its load and slower guesses), then γ = 0 for both ("guessing did not pay on these machines
  last time") and the same 42-44 tok/s.
- **Everything on the processor** (`CPU=1 FORCE_TREES=0`): the 3B kept 0.729 of its guesses against
  the 1.5B's 0.744, at 73.1-73.4 ms a guess against 64.8; first 120-token reply 1.86-1.89 tok/s
  against 2.12-2.20, then γ = 0 and 6.39-6.40 against 6.46-6.64. On a long link a chain from a processor
  spends its round guessing (a missed seed waits ~300 ms, above), so a dearer guesser that guesses
  no better can only lose there.

**So the rule**: a drafter reading at most `SMALL_DRAFTER_READS` (= `MOST_DRAFTER_BYTES_TO_FETCH`,
2 GiB) a guess may be `1 / drafter_max_fraction()` (½) of its target on a node whose card is in
play (`ModelProcessPool::card_in_play`); any larger drafter, and every drafter on a node without a
card, keeps `LARGE_DRAFTER_MAX_FRACTION` (¼). A larger one is only ever one the owner holds — a
7B for a 14B, ~3× the 3B's cost a guess, where the flow's drafter already filled 65% of the reply.
Planner, request and download pass choose through the one `largest_that_can_draft`, so the room
left (#275) is the drafter's that will be chosen. A/B `SWARMLLM_DRAFTER_MAX_FRACTION=4` (the rule
before, on a card). Tests: `the_largest_small_drafter_within_half_its_target_is_chosen` (red at 4),
`a_large_drafter_is_held_to_a_quarter_of_its_target`, `a_node_without_a_card_holds_every_drafter_to_a_quarter`.
**A node already holding a smaller drafter fetches the larger one beside it** (#331, 2026-10-24):
`drafter_to_acquire` answers the held drafter AND the next step, and prune keeps both until the
larger one is held and chosen — `docs/invariants/memory.md` § "A split with no drafter fetches one".

**With the 3B the flow keeps three levels of 8 rows and 128 live rows** (2026-10-23 night, the
real link, round trip 393-400 ms, the same binary, QUIC, greedy chat; a throwaway first arm, then
ABBAABBA or ABCCBA, 8 replies of 400 tokens an arm; `~/swarmllm-318/harness/realflow{lv8,rows,wide}.sh`).
The sweep that chose them ran with the 0.5B while the flow still waited; with the 3B it drafts, so
it was run again. Means a reply; a forward is one drafter call and this node's layers after it:

| arm | tok/s | tokens a forward | drafter ms a forward | near ms a forward | waiting s |
|---|---|---|---|---|---|
| **3 levels × 8 rows, live 128** | 15.39-18.27 (17.21-17.77 by run) | 0.90 | 33.1-33.3 | 13.9 | 1.8-2.0 |
| 2 levels × 8 | 16.69-18.34 (17.42) | 0.75 | 26.7 | 13.2 | 1.6 |
| 3 × 4 rows | 15.12-17.85 (16.60) | 0.76 | 33.1 | 12.3 | 0 |
| 3 × 16 rows, live 128 | 15.03-15.90 (15.48) | 1.14 | 43.2 | 16.7 | 4.8 |
| 3 × 16 rows, live 256 | 13.79-16.87 (15.53) | 1.06 | 48.1 | 18.3 | 0.55 |

A first ABBA read had two levels ahead by 4% (18.33 against 17.68); the longer run did not hold it.
Half the rows cost the drafter nothing less and walked a sixth less. A level of 16 rows is not
captured (`cuda_graph::MAX_POSITIONS`, 8) and costs ~10 ms more a call. At 128 live rows it filled
the budget and waited. At 256 it stopped waiting but paid more a forward than it walked.

*The drafter's call is the model, not the flow's path* (`~/swarmllm-318/harness/drafter3b_levels.sh`,
`prefill_bench SWARM_BENCH_GROW=<rows>,8`, the 3B Q4_K_M 600 deep, idle card, min of 7, two passes):
8 rows from nothing (one read of the root) 8.1-8.2 ms, 16 rows 19.2-19.5, 24 rows 30.3-30.4, 32
rows 41.3-41.4. Each level of 8 rows costs ~11.0 ms, and the flow's three-level call (33.1) is that
plus ~3. So the flow sits at two bounds at once: its reach, ~7.4 tokens a 434 ms answer, and this
node's compute, 47 ms a forward for 0.9 tokens. Every knob above trades one for the other.
Nor does the ranking: a row-level replay (`examples/flow_rows_sim.py`, which reproduces the
measured order) found nothing better than best-first by path probability. Holding back rows past
the walk's round-trip reach lost 1-9%, and a bonus for depth or 192 live rows gained under 1%.
What moved it was the drafter's first-choice rate, ~15% a reply for each 4 points.
**What is left**: a cheaper level (8 rows cost 11.0 ms against one row's 8.1), or a drafter that
guesses better for what it costs. A smaller model drafting for the 3B (staged drafting) does not
pay: `docs/plans/split_speculation.md` § "What the literature says".

**The flow's speed is the drafter's ranks** (2026-10-24, `examples/drafter_ranks.py`). A greedy
reply's far-end draws are the target's first choice at each position of that reply, so ONE
teacher-forced pass of the 7B's own reply through the 7B and through a drafter
(`logits_reference_probe` with `LOGITS_PROBE_MODEL`, on the processor, a minute each) gives every
draw's rank in the drafter's list, with no link. `flow_rows_sim.run(trace=…)` replays the flow on
those ranks at the call cost the real flow logged, and on the refrigerator prompt (400 tokens) it
reproduced the real link (table above and § "A small drafter may be HALF its target"):

| drafter (call) | first choice | outside its 8 | replay tok/s | real tok/s |
|---|---|---|---|---|
| Coder-3B Q4_K_M (33 ms) | 0.820 | 1.2% | 17.2 | 17.2-18.1 |
| Coder-1.5B Q8_0 (25 ms) | 0.820 | 2.0% | 15.8 | 15.95-16.46 |
| 0.5B fp16 (28 ms) | 0.756 | 3.7% | 14.0 | 13.98 |
| Coder-0.5B Q8_0 (14 ms) | 0.701 | 8.2% | 11.1 | 12.06 |

It counts 1.3-2× the real misses (8 against 3.2 for the 3B) but matches tokens a forward (0.93
against 0.90). Priced on the 3B, refrigerator / a code-writing prompt (23.6 tok/s replayed):
- a call at 2/3 or 1/2 its cost: +6-8% (18.2 / 25.0-25.5) — the cheaper level is worth that much;
- 16-row levels as cheap as 8 (a 16-row capture): +5-8% at best (2 × 16, 256 live, 25 ms: 18.6 /
  24.8); at the measured 16-row cost it reproduces the real loss; 256 live rows −2%, 4 levels ±0;
- a temperature on the ratings: the probabilities are near calibrated already (the 3B's log-loss
  of the target's choice is least at T 0.85, 0.560 against 0.573 at 1), and T 0.85 gave +5% / −1%;
- every draw the drafter's first choice: 26.5 / 37.5, **+54-59%**. Of that, words outside the top 8
  cost 4-6%; the draws at ranks 1-7 cost ~30% (24.5 / 34.7 made first choices). They ARE in the
  tree, but best-first put most rows under the first choice, so the walk enters a shallow branch
  and waits a round trip for rows under it.

So on a long link the flow is held by how often the target takes the drafter's second or third
choice, and no knob, ranking or cheaper call gets that back — only a drafter that agrees with the
target more often (or one fed the target's own state). The 3B over the 1.5B holds across ten prompts
(five chat, five code-writing, 3,720 positions): first choice 0.779 against 0.755 in chat and 0.916
against 0.902 in code, outside its 8 1.6% against 3.0% and 0.1% against 0.2%, and it replays faster on
nine of the ten (chat +2-14%, code +1-9%; only the one CSV prompt went to the 1.5B, 25.8 against
23.6). T 0.85 across the ten: chat +0-4%, code −0.2 to −2.7%, +0.6% on average — not a switch.
A code edit is copied: every drafter 0.998-1.000. A drafter's flow speed varies more with the
prompt than with any knob: the 3B replays 10.1 tok/s on a short story and 43.3 on a binary tree.

**For a 14B target the 3B is still the drafter to hold** (2026-10-24, `~/swarmllm-318/ranks14/`:
`gen14.py` — the 14B's own greedy replies from the llama.cpp reference GGUF, ids kept as generated —
then `run14.sh` and `sim14.py`; eleven prompts, 4,245 positions, Qwen2.5-14B-Instruct Q4_K_M as the
truth). First choice: coder-7B 0.859, coder-3B 0.838, coder-1.5B 0.817, 0.5B 0.796 (outside the 8:
0.7%, 1.2%, 2.4%, 3.2%). Replayed on the long link with the halves' compute doubled for a 14B (far
40 + 1.0 ms a row, near 19 + 0.36): the 3B at its measured 33 ms 19.9 tok/s (code 27.4, chat 10.9);
the 7B 19.4 at an assumed 55 ms a call and 17.8 at 70 (its call on the card is unmeasured; scaled
from its 18.5 ms a decoded token, ~60), the 1.5B 18.0, the 0.5B 16.3. So the 7B's +2 points pay only
if its call is ≤ 55 ms, and only in chat (12.2 against 10.9) — code goes to the 3B. Nothing to
change: the 7B (4.7 GB) is past `MOST_DRAFTER_BYTES_TO_FETCH` (2 GB), so `largest_that_can_draft`
already picks the 3B. Every draw the 3B's first choice replays 26.4 (+33%) — as for the 7B target,
the headroom is a drafter that agrees with the target more, not a larger one.

**A drafter that agrees with the target more is one trained AS the target was** (2026-10-29, #370).
Every drafter ranked above was a coder model; the 14B is the general Instruct. The general Qwen2.5
1.5B (Q8_0) and 3B (Q4_K_M) and the coder 0.5B, ranked the same way (`run_general.sh`, `run_c05.sh`;
chat = the four chat prompts, 1,438 positions; code = the six code-writing ones, 2,406):

| drafter | chat first choice | code first choice | replayed chat / code tok/s |
|---|---|---|---|
| coder 0.5B Q8_0 | 0.592 | 0.820 | — |
| general 0.5B Q8_0 | 0.691 | 0.827 | 8.30 / 22.88 |
| coder 1.5B Q8_0 | 0.698 | 0.858 | 9.23 / 25.27 |
| general 1.5B Q8_0 | 0.745 | 0.868 | 10.39 / 25.33 |
| coder 3B Q4_K_M | 0.745 | 0.868 | 10.92 / 27.40 |
| general 3B Q4_K_M | 0.784 | 0.879 | 11.78 / 26.31 |

The general one wins at every size, by 4-10 points in chat and ~1 in code; the general 1.5B equals the
coder 3B. Replayed (`sim_general.py`, the halves doubled as above) the coder 3B keeps a 4% lead on
code — its code guesses are more confident, so the tree goes deeper under them — and the general one
leads chat by 8%; by time a token over both, the general one is ~4% ahead. On the REAL TH↔AU link
(binary `cb68c5ca`, the default shape at 24 of 48 layers here, arms alternating in one run, Windows up
24.6-25.0 h, 0 driver events): the 0.5B pair in the FLOW (`TAG=drafterline05`) — prose 7.67 / 8.73
tok/s against 6.32 / 6.53, misses 17-20 a reply against 30-35, round trips waited 45 against 54-58;
code 20.4 / 28.8 against 20.7 / 22.2, misses 1-2 and 16-18 round trips both (the spread is the link's).
The 3B pair (`TAG=drafterline`) did not fit on this 8 GB card beside 24 layers (2.7 GB of it held by
the desktop) and ran the continuous stream with the drafter on the processor (62 ms a guess): prose
4.18-4.38 against 3.43-3.61 (123 of 207 guesses kept against 107 of 222; 77 round trips against
100), code 7.28-7.91 against 7.26-7.90 (148 of 174 kept both). So the line breaks a TIE in size and
nothing more: size still decides first (the coder 3B replays ahead of the general 1.5B), and a node
holding a drafter of another line fetches one the same size of the target's
(`HeldDrafter::beaten_by`) — a different model, never another copy. The line is read from the model
id (`release_line`): the general uploads' headers have no `general.basename` and name themselves
`qwen2.5-3b-instruct` where the coder's say `Qwen2.5 Coder 3B Instruct AWQ`; a misread line only
loses the tie, the order before. An id spells a two-token tag with a hyphen (`q8-0`, `q6-k`,
`iq4-xs`), and until the pre-release review the quantisation was stripped only as one token or
three, so every Q8_0 drafter — the 0.5B and 1.5B ones here — kept its tag and matched no target
(gotcha #991; `quant::inferred_base_name` now tries the last two). Literature: DistillSpec (Zhou et al., ICLR 2024, arXiv 2310.08461)
names alignment of the drafter with the target the main lever and gains 10-45% over standard
speculation by distilling the drafter toward it; a sibling fine-tuned as the target was is part of
that alignment at no cost.

**The flow has its own distance line: 150 ms, not the tree rounds' 300** (2026-10-23,
`FLOW_LEAST_ROUND_TRIP_MS`). The 300 ms line (`TREES_LEAST_ROUND_TRIP_MS`) was set for tree ROUNDS
before the flow existed, and the flow inherited it, so every link under 300 ms — most links between
continents: this machine reaches the Europe peers at 206-236 ms — took the chain stream. Rig:
`~/swarmllm-318/flow318.sh` (QUIC netns, every packet delayed for the round trip, Qwen2.5-Coder-7B
Q4_K_M split 0-3 | 4-7, both halves AND the 0.5B Q8_0 drafter on this one RTX 3070 Laptop), the
main build `e8e649d0`'s code (`~/swarmllm-318/f2/swarmllm`), arms by env only —
`SWARMLLM_DRAFT_TREES=1` (the flow, forced past the line) against `=0` (the chain stream) —
alternating flow, stream, flow, stream; 400-token greedy replies, the first of each node start left
out (it carries the load), so four warm replies an arm (outputs `~/swarmllm-318/flowgate{100,120,150,220,300}/`,
logs `flowgate.out`, `flowgate2.out`; Windows up 0.7-1.2 h):

| round trip | flow tok/s | chain stream tok/s | flow ÷ stream | flow waiting a reply |
|---|---|---|---|---|
| 100 ms | 19.44-21.00 (20.5) | 21.76-21.86 (21.8) | 0.94 | 0-0.1 s |
| 120 ms | 18.97-21.81 (20.0) | 18.99-19.31 (19.2) | 1.04 | 0-0.08 s |
| 150 ms | 18.85-20.04 (19.4) | 16.24-16.29 (16.3) | **1.19** | 0-0.15 s |
| 220 ms | 14.88-18.92 (16.5) | 11.60-11.69 (11.7) | **1.42** | 0.36-0.68 s |
| 300 ms | 12.56-13.28 (12.9) | 8.65-8.67 (8.7) | **1.48** | 0.7-2.2 s |

The chain stream fits `2.78 tokens ÷ (round trip + 21 ms)` at every point. Below ~220 ms the flow
stops waiting and is bound by this node's own work (drafting + its half, ~50-56 ms a forward on the
shared card), near 20 tok/s: so the two cross at ~110-120 ms here. **The rig leans against the
flow**: the far half shares this card, and the flow sends ~22 rows a forward against the stream's
few, so in the field — the far half on its own machine, the requester's forward ~30 ms (the real
link's drafter table) — the flow's ceiling is higher and the crossing nearer. The line is put where
the flow measured a clear win on this rig, 150 ms, not at the tie. **A reply whose tree cannot flow
keeps the rounds' line** (`dsd_tree::least_round_trip_ms`): a peer without bit 31 gets tree rounds,
which lose to the stream under 300 ms. So the decode loop asks `flow_possible` on the stream shape
BEFORE the line (`dsd::…`, `flow_shape_if_trees`); a flow that fails mid-reply still finishes in
tree rounds, at any distance. Test: `a_tree_that_flows_is_the_default_over_a_shorter_link_than_tree_rounds`
(red with the flow's line removed). **Mechanism**, the CUDA build of this change with no switch set
(`flowmech.out`, 200-token replies): at 220 ms the warm replies flowed (`a tree that flowed is
complete`, `farthest_round_trip_ms=Some(221)`, 14.95-15.61 tok/s) and at 120 ms they streamed
(18.27-18.50); each start's first request streamed at both, its drafter not yet on the card
(`drafter_on_card=false`).

**What a change must keep.** The draw at a row must not depend on the forward that carried it
(noise by position, the row's own ancestors in its mask); the three trees change only by the
ordered ops; a rebase's forward goes out at once, before any proposal, so path + kept never
exceeds the old tree; an answer is never applied to a row it was not drawn at (the remap, the
epoch); and every forward carries the WHOLE tree's shape — a segment checks it against its real
cache (`DraftTree::rows_for`), which is what makes a wrong one a refusal, not a wrong reply.

**Tests**: `a_tree_read_in_three_forwards_reads_as_one` (`split/tests/draft_trees.rs`, a chain over
the same rows the control); `a_tree_that_flows_proposes_what_a_fresh_read_of_each_path_ranks_likeliest`
(`split/tests/drafting.rs`, red with the rows read as a chain, and its three-level call red at one
level a call: 8 proposals for 24); `a_tree_that_flows_keeps_its_live_rows_through_a_rebase`
(`draft_tree`); `walking_each_rows_draw_is_the_tails_own_walk` (`sampling`);
`a_tree_that_flows_is_applied_from_where_it_starts` (`model_worker`);
`a_flowing_trees_start_survives_the_wire_and_a_malformed_one_is_refused`,
`a_flowing_trees_start_is_sealed_and_survives_the_encrypted_frame` (`network::protocol`). Rig: `~/swarmllm-318/flow318.sh` (arms `flow`, `trees`, `rounds`,
`fault`, and `flow@NAME=v+NAME=v` for a knob; `CPU=1` the correctness pass). Security review 2026-10-22: no High or Medium; a stale
capability reading now finishes in rounds (L1); a hop flipping draw/walk semantics only fails
the step, as it could already (L2).

## A peer that fails a model on every request is left out of its plans for a while (2026-10-02)

**What it replaced.** Field report (2026-10-01, v0.3.218): peer `bf7b3263`, a
6 GB laptop card holding layers 7..18 of qwen3-30b-a3b, answered every prompt
pass and went silent at the first decode step — 5 of 6 pipelines including it
died (3 × "segment TIMED OUT … timeout_secs=30", 1 × lost conversation), 3 of 3
without it were fine. Inside a request the rules held: the silent peer was
barred from that request's retry. Nothing carried the lesson to the NEXT
request: the delivery-reliability multiplier priced the peer at
`expected_attempts=2.04`, which a fast peer survives, so the scheduler kept
picking it and every pick cost a 30 s deadline.

**The rule** (`daemon::state::peer_outliers`, Envoy's outlier detection —
`consecutive_gateway_failure`, `base_ejection_time` × times ejected, and the
panic threshold): `FAILURES_TO_EJECT` (2) failures of one model by one peer
with no request completed in between eject it from that model's plans for
`BASE_EJECTION` (2 min), doubling per repeat up to `MAX_EJECTION` (30 min). A
request the peer sees through forgives everything. Two, not Envoy's five: each
of ours costs a person tens of seconds, each of Envoy's costs a load balancer
milliseconds.

**Where it is fed and read:** failures through `SharedState::record_peer_delivery`
— the one place a segment timeout, an abandoned forward or a silent whole-model
reply is recorded, which now takes the model as a REQUIRED argument so no
recorder can feed the reliability figure and skip the count; resets through
`note_peer_completed_request` (every remote segment of a successful distributed
request, and a whole-model reply that arrived intact). Read by
`gather_candidates`, per PART: an ejected peer is admitted for a part only when
no healthy holder of it is admitted (this node counts as healthy) — an ejection
can slow a plan, never make a model unroutable.

**What a change must keep:**
- Refusals (out of memory, prompt too long) are NOT failures here — they arrive
  fast with a reason and are re-planned inside the request.
- The prompt-pass-only sampling of INTACT deliveries
  (`pipeline::local::note_segment_delivery`) is why the reset is a completed
  REQUEST and not an intact delivery: the field failure delivered every prompt
  pass and failed every decode, so a reset on delivery would never let the count
  reach two.
- Pinned by `a_peer_failing_a_model_every_time_is_left_out_unless_it_is_the_only_way`
  (red with the ejection disabled, 2026-10-02) and the `peer_outliers` unit tests.

**From the rules file (moved 2026-10-02):**

**`daemon::state::peer_outliers`** — Envoy's outlier ejection per (peer, model):
two failures in a row (timed out, abandoned, silent — fed ONLY by
`record_peer_delivery`, model required) eject it for 2 min, doubling to 30;
a completed request (`note_peer_completed_request`) forgives. `gather_candidates`
skips an ejected peer for a part unless no healthy holder of that part is
admitted (the panic threshold). Refusals are not failures. A peer silent at every
first decode step kept being picked at a 30 s deadline each (field report).

→ `docs/invariants/scheduling.md` § "A peer that fails a model on every request"

## Speculation that does not pay steps aside — γ = 0 is an answer (2026-09-29)

**`dsd_controller::best_gamma_for_check` may choose zero guesses** (a plain round through
the same path) where guessing costs more than the round trips it saves — SmartSpec's rule. It
could not before, and DSD ran a near split at 6.8 tok/s against 38-42 plain. What a request
learned is handed to the next on the same model and machines (`recall` / `remember`), and the
drafter's cold first call never counts as a guess's cost. A new speculating path gets a way to
stop guessing, or it is only safe over long links. **The continuous stream (on by default since
v0.3.216) walks no γ**: it remembers `best_gamma_overall` over the costs it measured, so a zero
still steps the next request aside — it remembered its starting 4 until the flip (2026-10-01).

**Because a zero is remembered, one stalled check must not be able to decide it** (#244,
2026-10-12). The check-cost line (`CheckCost`, weighted least squares, decay 0.8) was taken over
by ONE check that waited ~3-4 s among ~40 ms ones: at the v0.3.235 release check it became
0 ms + 209-468 ms a position — γ 0, and the next request stepped aside for `REPROBE_AFTER`
(3 of 4 streamed runs, 28 → 21 tok/s on the request after); on the same rig later the same stall
landed at the mean position count and read as 316-770 ms FIXED (γ 6-8). `CheckCost::record` now
holds back a check over `OUTLIER_FACTOR` (3×) its expected cost — the line, or the mean scaled by
positions for a length the line has not seen — until the next round: a normal one drops it, a
second slow one fits both, so a lasting change is learned one round late. Measured on the 12h rig,
ONE binary, `SWARMLLM_DSD_CHECK_OUTLIERS=keep` the old fit, ABBA ×2 stream + ×2 rounds: the old fit
was poisoned in 3 of 4 streamed runs, the held-back fit in none, `checks_held_back=1` on every
stalled request (`DSD: request complete`). The drafter's cost already used a median for the same
reason (`RecentMedian`); robust RLS bounds one sample's influence the same way (Ruckdeschel,
arXiv 1204.3358). The stalled request itself stays slow — that is the stall, not the controller.

**The hold cannot act in a request's first rounds, so the line has a floor** (2026-10-23). The hold
needs `HOLD_AFTER` (3) fitted rounds. On the real TH↔AU link (`~/swarmllm-318/harness/realdelay.sh`
arm 2, the first request of a fresh node, the drafter not yet on the card) the stream recorded three
checks: 3 rows at ~771 ms, 3 at ~1,529 (the stall, second) and 1 at ~406. The fit was 12.4 ms +
393.1 a position (`DSD: request complete`), which made `best_gamma_overall` 0. The next two
400-token replies logged "guessing did not pay on these machines last time" and ran the n-gram loop
at 2.58 and 2.17 tok/s, against ~13 for the flow. The other five first requests of that run fitted
398-409 ms fixed + 3-6 a position.

A check that crosses to a peer costs at least the round trip there, so the fit's fixed part is
floored by it. `CheckCost::floor_ms` is the least round trip measured to the plan's farthest peer
segment (`farthest_segment_round_trip_ms`, the window's minimum). It is capped by the cheapest
check recorded, because that reading was 757-809 ms on a 380 ms link for the first request of a
fresh node in 4 of 6 arms, and a check cannot cost less than one was measured at. Where the free
intercept is under the floor, the fit is the weighted least-squares line through `(0, floor)`,
slope ≥ 0. This is bounded-variable least squares (Stark & Parker 1995): with one bound, the
optimum lies on it.

Those checks fit 406 + 243 a position, γ = 2; at the link's own 380, 380 + 253, γ = 1. A line
already above the floor is unchanged (`a_line_above_the_round_trip_is_left_alone`). The constructor
is the guard: production has no `CheckCost::default()`, only `for_a_plan` and a recalled fit's
`on_a_plan`, so neither speculating path (`dsd`, `ngram_only_spec`) can fit without the floor. Test:
`a_check_is_never_fitted_below_the_round_trip_it_crossed`, red with the floor clause removed.

**On one computer the step is priced on its own** (2026-10-25, `model_worker::LocalSpec`). The
worker's n-gram checked every guess it found, up to `num_pred_tokens`, at any width. Its own
measurement (the arms of `a_guessed_reply_is_the_reply_read_one_token_at_a_time`, release build,
RTX 3070 and this laptop's processor, 96 greedy tokens, each arm run twice and the second timed,
two runs) found that a loss in as many places as a gain: Llama-3.2-3B code on the card 88-112 tok/s
against 115 one at a time, its processor 15.9-16.2 against 17.4; qwen3-1.7B prose on the processor
17.4 against 18.9. A card's step is ONE captured graph (7.7-8.8 ms); a check of up to 8 positions
is a graph of its own shape (10-13 ms warm), wider ones run op by op (17-28 ms), and a
conversation's first check of each width runs uncaptured whatever its width (`cuda_graph`: a
capture follows an ordinary step of the same shape; 25-49 ms). One line through steps and checks
prices a short check as nearly a step — the worst buy looks cheapest — so
`best_gamma_beside_a_step` takes the step's median (`RecentMedian`) and a line fitted from checks
only, and is asked per round over the guesses FOUND (two guesses can be worth less than their
check where seven are not). The acceptance is `AcceptanceEstimate::recent`
(each round weighs the ones before it by 0.9; a declined round fades it towards a prior of 0.7 over
2 trials), because whether text is being copied changes inside one reply.

**A refusal is tried again.** The first fit of the first version saw one check — the cold one —
chose zero and checked nothing more: Qwen3.5-4B's repetitive story, 102-104 tok/s with every guess
checked, stayed at 71 (72 one at a time); qwen3-1.7B code fitted 24.9 ms from its first 8-position
check and declined the rest. So after 8 rounds of declined guesses the next are checked anyway; a
check that paid (more tokens than steps in its time) starts the count over, one that did not doubles it to at most 64 —
`SpecBackoff`'s doubling, the in-request counterpart of `REPROBE_AFTER`. Measured after it, the
guesses that pay against one at a time: processor qwen3-1.7B code 17.9-19.0 → 24.6-27.6, prose
18.9-19.0 → 18.4-18.8; Llama-3.2-3B code 17.4-17.5 → 18.0-18.1, prose 17.5 → 17.1-17.9; card
qwen3-1.7B code 126.7-127.8 → 115.4-165.5 (every guess 127.7-135.3), prose 128.7-129.4 →
126.5-128.3; Llama-3.2-3B code 114.8-115.7 → 113.8-114.6, prose 112.8-113.8 → 113.1-116.5. The
card's spread is the choice turning on a few early timings.

**A one-off is not a price** (2026-10-25). Much of that spread was forwards that cost more than
their shape ever will again: the first of a shape (run the ordinary way so a capture knows its
output), one that instantiates a graph, one whose capture is refused — a Qwen3.5-4B check that made
its graph took 240-300 ms against 16-30 replayed, and one such check fitted into the line priced
every later check of the reply. `LocalSpec::next` reads `SplitModel::one_off_forwards` before and
after a round and records neither the step nor the check from a round that paid one; a probe that
paid one starts the count over, as one that paid does, instead of doubling its wait (it said
nothing of what a check costs). Qwen 3.5 guesses on a card since then (`docs/invariants/inference.md` § "A
recurrent state is taken back into its last check by replaying the rows it kept"). A/B
`SWARMLLM_LOCAL_GUESS_GAMMA=fixed` (every guess, the old rule); the test runs both arms. Tests:
`a_check_dearer_than_a_step_is_bought_only_when_its_guesses_are_kept`,
`a_recent_acceptance_follows_the_reply_and_fades_to_its_prior`.

**A width not yet timed is priced through the step** (2026-10-25,
`CheckCost::predict_beside_a_step`). A line fitted from checks of ONE width has no slope (`fit`
holds it flat until the widths spread), and a request's first checks are all as wide as it may
guess, so the count priced a check of two as the check of eleven and chose zero. Qwen3.5-4B's story
on the processor, once a rejected check stopped being read again at the head of the next (which had
varied the widths by luck): an 11-position check at 305 ms beside a 95 ms step, then 30 rounds
declined — 13.1 tok/s against 15.7 checking every guess; Qwen3.5-0.8B's code on the card declined
10-14 rounds the same way, its first check made dearer by the buffers it keeps its rows in (28.8 ms
against 12.3; counted as a one-off since, `DecodeGraph::note_not_ready` — a conversation's first
decode step and a step whose caches must grow are run uncaptured and cost more than their shape
will again). Until a second width is timed, a check is priced by the line through the step (a
forward of one position) and the mean check: 116 ms for two positions there, against 93 fitted
later — dearer than the truth, still bought, and the check of another width it buys fits the slope.
Optimism where a cost is unknown, as UCB prices an untried arm (BanditSpec, Hou et al., ICML 2025,
chooses γ that way). A check that crosses to a peer is priced by its fit alone, round trip floor
included. After it, no round declined in 16 runs of the test; the 4B story 15.9-17.0 tok/s on the
processor (10.4-10.5 one at a time). Test: `a_check_dearer_than_a_step_is_bought_only_when_its_guesses_are_kept`
(a check of one width, red with the step's line removed).

→ `docs/plans/split_speculation.md` § "Guessing steps aside where it does not pay"

## What a peer HOLDS on disk and what it has LOADED are different facts

`NodeCapability::hosted_shards` is disk; `NodeCapability::resident_layers` is
memory. **`inference::scheduler::PeerResidency` is the single reading of the
second**, in three states: `Layers(n)` exempts those layers' weights and charges
full price beyond them, `Cold` charges everything, and `WarmAmountUnknown` — a
peer that has published nothing — keeps the older, more generous pricing.

The third state is not politeness. Reading silence as "holding nothing" charges
full weights to every node on an older build and routes around the machines best
placed to answer, which is the additive-protocol rule's exact failure. A bare
"is it warm" boolean has the opposite fault: it exempts every layer under
consideration, so a peer warm for part of a model is credited with the whole of
it.

→ `docs/invariants/scheduling.md`

## Three consumers have now read `standbys.len()` as an answer it cannot give

`scheduler::standby_covers` (does this standby hold that range),
`standby_has_room` (could it run it), and `peer_segment_has_standby` (can
anything take over what THIS peer is serving) are the three questions a plan's
standby list is actually asked. The bare count answers none of them, and each
consumer that used it was wrong in its own way:

- **#451** — a plan logged `standbys=1` beside "NO standby available for failed
  segment"; both true, and a tester lost an hour to the contradiction. Fixed by
  reporting `segments_without_standby`.
- **#464** — a count is not a statement about CAPACITY. One node was named
  standby for four segments it could run one of.
- **#465** — a count is not a statement about the SEGMENT in front of you.
  `request_has_standby` gated the ACK fast-fail on the whole request, so the
  four segments of a five-segment plan that had no backup were abandoned early
  anyway. Abandoning without a replica can only turn a slow success into a 503
  (Dean & Barroso, *The Tail at Scale*; measured in #386, where the result
  arrived 1.6 s after we gave up).

The three are related in the direction that matters: #464 made #465 WORSE, since
segments that used to carry a fictional standby now honestly carry none.

**Before using a collection's length as an answer, check the granularity the
question is asked at.** `standbys.len()` is a fact about a plan; every consumer
so far has wanted a fact about one segment.

## A count and an outcome that disagree are two different questions

`scheduler::standby_covers` is the one predicate for "could this standby take
that segment over", used by `segments_without_standby` (reported with the plan)
and by `pipeline::distributed::failover_segment`'s search.

Standbys are chosen per segment, so a plan can log `standbys=1` and still have
none for the range that fails — the failure then logs `total_standbys=1` beside
"NO standby available for failed segment", and both lines are true. A tester
read the pair as a contradiction and lost an hour to it. The plan now names
`segments_without_standby`, and the failure reports
`standbys_covering_this_segment` next to the total. When a summary count cannot
answer the question a reader will ask of it, print the answer, not the count.

## A layer is priced as the model it belongs to (2026-10-12, FUTURE_WORK #250)

**What it replaced.** Every node states one speed: tokens a second for a 7B Q4_K_M, from its
memory bandwidth (`vram::estimate_tokens_per_sec_7b`: bandwidth ÷ `MODEL_SIZE_7B_Q4_GB` ×
efficiency). Both routers turned it into a layer's price as `1000 / speed / 32` for EVERY model,
so a 0.5B's layer, a 30B-A3B's and a 70B's all cost one 7B layer. Between two plans that only
trade compute the error cancels. Against the network term (round trips, real milliseconds) and
the cold-load term it does not: a 30B-A3B Q4_K_M layer reads ~35 MB a token (attention plus 8 of
128 experts) against the reference's ~137 MB, so its compute was ~4× too dear, and a remote
card's advantage over this processor looked ~4× bigger than it is. That is report #035's
shape: a mixture-of-experts model this node held whole was planned through two peers. A dense
32B layer (~310 MB) was ~2.3× too cheap, the other way.

**The rule.** `scheduler::layer_work_for` gives every candidate of a model one `layer_work`:
the bytes one token reads per layer (`GgufTensorMeta::active_bytes_per_layer` — every `blk.*`
tensor, expert stacks `_exps` at `expert_used_count / expert_count`, the output projection, the
embedding only where it IS the output) over the reference's bytes per layer.
`NodeCandidate::prior_ms_per_layer` is the one prior from a stated speed, read by
`parallax::vertex_cost` and greedy's `estimated_cost_per_layer`. `est_tokens_per_sec` itself
stays in 7B units, so the ratio and sort comparisons (the hand-off gate's
`DELEGATE_MIN_CPU_SPEEDUP`, greedy's ordering) are untouched. Unknown (no header, experts without
a used count) prices as before: 1.0. Petals does the same thing by measurement: a server times
one real block of the model it serves and caches the figure per model
(`petals/server/throughput.py`, key `model_{name}_device_…_dtype_…`), so a mixture-of-experts
block counts only the experts it routes to.

**Evidence.** `a_light_layer_is_not_sent_a_round_trip_away_for_a_faster_card` (parallax): 48
layers, this node's processor at 5 tok/s against a card at 40 tok/s 50 ms away holding the
second half. At `layer_work` 1 the half goes abroad (1,200 + 6,400 ms against 9,600); at 0.28
it stays (336 + 6,400 against 2,688). Red with the factor removed from the prior, while its
control arm still chose the split. `a_layer_weighs_what_it_reads`,
`a_token_reads_the_experts_it_is_routed_to_and_the_layers_whole`. A/B inside one binary:
`SWARMLLM_LAYER_PRICE=reference`. The plan logs it (`DIAG: pipeline candidate … layer_work=`).

**The measured half: a peer's figures are kept per model (same day).** `PeerSpeed` was keyed by
NODE, so a peer timed on a 2B priced a 30B's layers at the 2B's figure — in the ranking, and in
the deadline `SegmentBudget` gives a segment (`predict_segment_ms`), where a figure from a
smaller model is too short. It now holds a `ModelSpeed` per model, and every read for pricing or
waiting goes through `SharedState::peer_model_speed(node, model)`: a model the peer has not run
for us is UNMEASURED and priced from its stated speed — the answer an unmeasured or stale peer
already got, which bounds the risk. Delivery stays per node (it measures the path, not the
model). The design first said "store per reference layer and scale"; it was not done, because
scaling a measured figure by bytes is wrong twice: a card's decode step is bound by how many
kernels it submits (a small model's layer costs more than its bytes say), and a decode figure
has the round trip folded in, which does not scale with the layer. Petals keys by model for the
same reason. Gossip is the one place a figure crosses models: `LatencyObservation` keeps
`ms_per_layer` raw (how an older node reads it) and adds `layer_work` (`#[serde(default)]`,
ignored by older nodes); a receiver divides by it — clamped to `LAYER_WORK_RANGE`, it is a
stranger's number — and puts the result on each model's scale. Display (`peer_performance_rows`,
the peer listing) shows the model measured most recently (`PeerSpeed::latest`). Tests:
`a_peer_measured_on_one_model_is_unmeasured_for_another` and
`a_peers_figure_on_one_model_never_prices_another`, both red when `PeerSpeed::model` answers with
the latest model; `gossip_is_put_on_the_scale_of_each_model_it_prices`, red without the scale.

**What a change must keep, and what is open.** Never answer a pricing or deadline read from
another model's figure (`PeerSpeed::latest` is for display only). Dense models above 7B now cost
more compute against the network, so a split onto a faster card is priced more attractive.
Before release, a WAN split rig must show that this adds no route slower than the one it
replaced (`docs/plans/regional_pipelines.md`).

## A peer's own timing is a floor under its stated speed (2026-10-18, FUTURE_WORK #3 (f))

**What it replaced.** A peer this node has not timed is priced from its stated speed — memory
bandwidth for a 7B, scaled by `layer_work`. A chain its head leads (#3 (a)) is never timed from
here: the head drives it, and the coordinator records no sample for the peers behind it. So those
peers stayed on the stated-speed prior for good, and it was the optimistic side: on the live swarm
(2026-10-18, five requests on Qwen3-30B-A3B) a warm two-peer 16-token reply priced 2.3 s of compute
took 19.4 s.

**The change, after Petals.** A Petals server times its own block (`server/throughput.py`:
`inference_rps` from single-token steps on the real block, cached per model, device and dtype) and
publishes it; its client prices a span's compute from it (`sequence_manager.py`, 300 rps when
unknown). Here each worker times the decode steps it runs for PEERS — `ModelProcessPool::forward_for_request`,
the one call every segment served for a peer's pipeline passes through, and a split a peer handed
this node to lead (`Requester::Swarm`), from send to result, answered steps only, a step being
ONE position of the base model (`process_pool::forward_is_one_decode_step`: the count the serve
credit uses, `tensor_util::positions_carried`; no share of a layer, image or adapter) whatever else
it asks — into `OwnDecodeSpeed` on the
`WorkerHandle`, so a worker respawned on the other device starts with no figure (per device, as
Petals keeps it). Published per resident model
as `ResidentModelLayers::decode_us_per_layer` (`#[serde(default, skip_serializing_if)]`; an older
node reads past it, as it reads past `ranges`).

**Four choices, each against a measured or named failure:**

- **The FASTEST of the last 32 steps — not the EMA `peer_speed` keeps, and not a median.** Waiting
  only ever ADDS to a step (behind another request's prompt pass, in a batch's collection window),
  which is the round trip's shape, and the answer is the same: BBR's windowed minimum RTT, and
  `LatencyFilter` here, which takes the minimum over Serf's median because the slow mode is the
  MAJORITY (`docs/invariants/network.md` § network coordinates). The first build took a median;
  the security review (2026-10-18) found any peer that sent this node work could queue half its
  steps behind other forwards and make it publish a slow figure for an hour, steering coordinators
  away. Against the minimum it must slow EVERY step in the window — the node really being that
  busy. An EMA at 0.3 swings several-fold on one such step and back, and each swing past the
  deadband is capability news to every peer (#91); the minimum moves only when a faster step
  arrives or the fastest ages out. Published after 4 steps; withdrawn an hour after the last, and
  a step after that starts a NEW window — a figure that prices a node slow keeps away the steps
  that would correct it, `RANKING_STALE_AFTER`'s ratchet.
- **Peers' work only.** The owner's own split steps were counted at first; the review noted the
  figure's appearing and going would then tell every peer — a private pool's outsiders included,
  capability gossip has no scope — when the owner last ran a split of a model. Serving work is
  what peers already sent.
- **A floor, not a replacement**: `prior_ms_per_layer` = `max(peer's timing, stated-speed prior)`.
  The error #3 (f) measured runs that way (the prior was ~8x fast), and the max means the field
  opens no new way to look FASTER than the stated speed already allows — a peer that wants work can
  only state a speed it could always state. `stated_decode_ms_per_layer` drops 0 and holds a figure
  past 60 s a layer at the bound (dropping it would price the peer faster). Anything this node
  measured (`observed_*`) still outranks it; `priced_from_a_measurement` is unchanged.
- **A 1.5x deadband** in `health::monitor::capability_news` (`CAPABILITY_SPEED_DEADBAND`), with
  appearing and going as news.

**Tests, each red with its part switched off:** `a_led_chain_is_priced_by_its_peers_own_timing_where_nothing_was_measured_here`
(two led peers stating a card's speed and timing 8x slower lose to a whole copy measured at 3 ms a
layer; red without the timing and with a faster figure taken as a discount),
`a_peers_published_timing_reaches_its_candidate_as_a_floor_on_its_price` (red when
`gather_candidates` does not read it), `a_workers_own_decode_steps_are_published_beside_its_layers`
(red without the spawn guard: a step sent before the worker existed was run by another),
`a_workers_own_speed_is_the_fastest_of_its_latest_steps` (red as the latest step, and without the
expiry), `a_workers_own_speed_is_news_only_past_its_deadband`,
`a_one_position_check_is_timed_as_a_decode_step` (red with the batch path's test in front).

**On the wire (#333's lesson — a gossiped field nobody checks on a real node is unverified):**
`split_rig.sh owntiming`, TinyLlama on two processor nodes in a private network namespace
(`RIG_NETNS=1`, beside the live node), 2026-10-18, four checks. A's first plan read no figure for B;
after three split replies A's plan read B's `stated_ms_per_layer=Some(3.192)` beside its own timing
of B, `observed_ms_per_layer=Some(4.373)` (the round trip on top). B's plan read NO figure for A
while A had run only its own requests (the first build, which timed the owner's steps, handed B
`Some(2.592)` there — the control). After B drove a 48-token reply through A — speculating with
n-grams, so 37 of A's steps were checks of one position and 10 of several — B's plan read A's
`Some(2.284)`, its own timing `Some(3.193)`. PASS 4/4. The run before that FAILED the last check on
a 7-token reply: A served 5 such one-position checks, the batch path's test refused them all, and A
published nothing — no test reached that path, the rig did.

**What a change must keep, and what is open.** Never let the figure lower a price (the max), and
never count it as this node's measurement. A model run here only WHOLE (`Generate`) times nothing
this way, so a node that never serves a segment of it publishes no figure; its peers keep the
stated speed for it, as before.

## Inference Router Queue

`drain_queue` only fires on `RouterCommand::Submit` / `StreamSubmit` or
`queue_notify.notified()`. **Every code path that calls
`active_count.fetch_sub(1)` on completion MUST also call
`queue_notify.notify_one()`** — otherwise queued requests beyond the
per-tier cap (Bronze=¼ of `max_concurrent_requests`; while credits are dormant every request is `DORMANT_TIER` = Silver, i.e. ½) sit indefinitely
until a new Submit arrives. Four enforced sites:

- `ActivePipelineGuard::drop` (panic path) in `router/mod.rs`
- normal-completion arm in `dispatch_single`
- `execute_distributed_batch` spawn body + join-loop panic arm in
  `router/distributed_exec.rs`
- `BatchCleanup::complete_one` + `Drop` in `router/local_exec.rs`

Adding a new dispatch path that fetches active_count down without
notifying is a silent stall under load (gotcha #85).

## Scheduler Liveness Oracle

`inference/scheduler/mod.rs::gather_candidates` filters shard holders
against `connected_node_ids.contains(&node_id)` (skipping the local
node). `peer_registry` is intentionally preserved across mid-pipeline
disconnects (for reconnect attempts in `handle_connection_closed`'s
`in_active_pipeline` branch) so it's NOT the right oracle. Tests that
populate peers via `state.peer_registry.insert(...)` MUST also call
`state.connected_node_ids.insert(node_id.clone())` or the test peer
will be filtered out (gotcha #86).

## A reply a PEER generated is finalised here, not taken as it arrives

`inference::finalize_reply_text` is the single place reply text is finalised —
control-token scrub, leading `<think>` reasoning block, stop truncation, the
newlines that step strands. It has now been missed TWICE, and each time on the
commonest distributed shape of the day:

- **`pipeline::remote_generate`**, the path taken whenever ONE peer holds the
  whole model. A reasoning model asked over the swarm answered with its raw
  scratchpad while the same request answered locally came back clean (#634).
- **All three speculative coordinators**, which share one finaliser —
  `finish_speculative` — and gave it no stops at all. Filtering EOS *ids* is
  not finalising, so a control marker the tokenizer never declared as EOS
  reached the user as text, a `<think>` block came back as the answer, and a
  caller's `stop` was ignored outright. The n-gram one is the DEFAULT
  distributed path: a node holding nothing takes it for every request
  (2026-09-18, gotcha #643).

**`PipelineExecutor::reply_stops` is the single answer to "what stops end this
reply"** — the caller's own plus the template's, warmed by
`build_prompt_with_header` so the stops always describe the template the prompt
was built from. It is a value on the executor rather than a parameter because a
parameter is something seven call sites can get wrong, and the standard
distributed loop derived only the template half and never read
`sampling_params.stop` at all.

Finalise on the COORDINATOR, never by trusting the serving node: only the
coordinator covers peers on builds that never learned to strip anything, and the
helper is documented idempotent. **`remote_generate` is the one caller that
passes an EMPTY stop set** — there the peer ran the decode, applied the caller's
stops and its own template's, and reports what matched in `matched_stop_seq`, so
re-deciding it here against stops derived for a model this node may not hold can
truncate a reply the peer correctly kept. A coordinator that sampled the tokens
ITSELF has no such peer and must pass `reply_stops`.

An UNCLOSED `<think>` is still shown, and must stay that way: nothing knows
where an unfinished thought ends.

→ `docs/invariants/api-surfaces.md` § "A reply is finalised on the coordinator"

## A prompt longer than ONE peer serves is that peer's limit, not the request's (2026-09-25)

What a node serves is the model's context capped by ITS ceiling
(`max_seq_len_override`, else 8192), so peers disagree. `every_holder_would_refuse`
treats a length refusal as the request's own only at the MODEL's declared limit;
below it the coordinator bars the peer and fails over. **`error::longer_than_served`
writes the refusal and `error::served_context_refusal` reads it** — the wording is
wire format (every release since v0.3.101). Peers advertise
`NodeCapability::context_ceiling_tokens`; `SharedState::peer_served_context` is the
one reading, and a coordinator does not SEND a prompt pass to a peer that advertised
too short, nor pick such a standby. Running out ends on a 400 naming the other
machines' limit (`longer_than_the_swarm_serves`), never the 503 "too few machines".

**The whole-model path too** (`remote_generate`): its worker's "too long for
{model} … limit of W" names the window THAT node loaded, not the model's. Below
the declared context the coordinator bars the peer and returns
`SwarmError::LongerThanPeerServes` — 400-shaped, its own variant only so the
router re-plans it (`peer_serves_shorter_context`); a failed re-plan reports it
unless the re-plan's own answer is a 4xx (`report_after_a_replan`, one slot shared with the memory shortfall).

→ `docs/invariants/scheduling.md` § "A prompt longer than one peer serves"

## Single-source-of-truth helpers — Scheduling, routing and failover

Each names the ONE place a decision is made. A second implementation of any of
them is this codebase's most-repeated defect — see `.claude/rules/architecture.md`
§ "One invariant, N paths". **Read the topic file before changing one.**

Full evidence: `docs/invariants/scheduling.md`

- **`ModelProcessPool::serves_on_cpu` is the whole-model delegation precondition** — "would this request run on our processor": no usable card, told to use the processor, a build without CUDA, or a card the model does not fit.
- **A node holding every layer that would run the model on its processor lets the priced search compete with its fast path** — `assemble_pipeline_for` answers `serves_on_cpu` ONCE (a lazy `OnceCell`) and threads it into `gather_candidates`, which PRICES the local candidate rather than excluding it.
- **A peer's capacity for a prompt is weights PLUS that prompt's KV cache** — `scheduler::max_hostable_layers` takes `prompt_kv_bytes_per_layer` — the same arithmetic the worker charges at admission, f16 mirror included.
- **`inference::scheduler::delegation_target`** — the single decision to hand a WHOLE model to a peer rather than run it on this node's CPU.
- **`inference::router::distributed_exec::failure_is_penalty_worthy`** — gates `penalty_serve_failure` on (a) the assignment actually having had a remote segment and (b) the error not being locally attributable.
- **A peer that went silent is barred from THIS request's retry, and the retry happens** — every producer of `SwarmError::PeerUnresponsive` (the per-segment deadline in `pipeline/local.rs`, both fast-path arms in `remote_generate.rs`) calls `blacklist_holder_for_request` before returning, and `router::peer_went_silent` retries the variant by TYPE under `used_remote_segment`. Envoy's `previous_hosts` rule: a host that just failed is likely to keep failing, so the retry goes elsewhere or nowhere. A producer that bars nobody makes the retry wait the same deadline twice; a retry keyed on prose misses the producer whose words differ. → `docs/invariants/scheduling.md` § "A peer that went silent is barred from the retry"

## A request its reader stopped is cancelled, not failed (2026-10-15)

A tester pressed Stop after 30 s of an embedding model repeating one word. The
node logged `outcome=error tokens=0` at warn, and its diagnostics listed the
request among "recent inference failures" as "served peer <its own id>"
(field report 2026-10-15). A day later the same list told another tester to
"suspect that peer" for two of their node's OWN memory refusals — each split
named the peer it ran through (report #037). A listed failure now carries
`RequestFailure::peer_may_be_at_fault`, the penalty's answer
(`failure_is_penalty_worthy`, never a second reading of the error); one that is
not is shown "(not that peer's doing)" and left out of the repeated-peer note
(FUTURE_WORK #288). Three things were wrong the first time, each in its own place:

- **The outcome.** Three router completion paths (`dispatch_single`, the
  batched local and distributed paths) each mapped a result to an outcome with
  `Err(_) => Error(kind), 0, 0`. `Outcome::Cancelled` existed and nothing wrote
  it. `RequestTrace::finish_from` is now the one mapping: an error that is
  `cancel::is_request_abandoned` is `Cancelled`, and any failure counts the
  tokens its reader was shown (`streamed_so_far`, read before `mark_finished`
  clears it) instead of zero.
- **The level.** `REQUEST_ABANDONED` is a `ServiceUnavailable` by design (it is
  penalty-exempt and the router does not retry it), so `failure_log_level`
  gave it a warning; it now answers `Info` for it.
- **The list.** `router::failure_worth_listing` decides what the diagnostics
  keep: never a request whose reader left; and the server it names is the first
  segment a PEER ran — the plan's first segment, which it used to take, is this
  node on every local plan.

Tests: `a_request_its_reader_left_is_cancelled_with_what_it_was_shown`,
`a_request_its_reader_left_is_logged_as_information` — each red with its half
of the fix removed.
