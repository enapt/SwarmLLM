# Scheduling, routing and failover

The evidence behind the rules in `.claude/rules/architecture.md`: what each
rule replaced, what it was measured at, and what a change must keep.

Every entry here was paid for. **Read the entry before changing the code it
names** — the rule statement in `architecture.md` is the summary, this is the
reasoning, and several of these describe a fix that looked obviously correct
and was not.

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
where `latency_ms` is a ping. See `docs/FUTURE_WORK.md` § "The routing cost model's
network term overestimates a boomerang".

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
  encrypted activations, never the prompt — so narrowing who may take the middle
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

## The relaxation is scoped to the figures that are actually unreliable

**`parallax::CapacityBound`** says whose `max_hostable_layers` a routing pass
honours, in four rungs: `Everyone`, `PeersAtFaceValue`, `PeersUnbounded`,
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
outright. So `PeersUnbounded` is still the third rung and still does exactly
what it used to; it is simply no longer reached while a route that respects the
peers' own numbers exists. Pinned by
`a_peer_is_not_handed_more_than_it_says_it_can_hold_while_a_route_exists`, whose
null control — making `PeersAtFaceValue` return `None` — hands a peer
advertising 9 layers all 48.

A note on what was NOT built. Learning from the refusal itself (Omega's other
half) would need the coordinator to tell a memory refusal from any other 503,
and it cannot: `LocalMemoryUnavailable` displays as `Service unavailable: {0}`,
so it flattens to `ServiceUnavailable` across the wire and is indistinguishable
from a spawn failure or a broken pipe. Recovering it from the message prose is
the #295 trap. Carrying it structurally is an additive protocol change, and the
benefit it buys — roughly one second on a request that fails either way — did
not justify it against the risk of refusing routes that would have worked.

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
- **Every pass says which one it is.** The `PeersUnbounded` line promises a
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
Since the ACK deadline is 10-90 s inside a segment budget that runs to 300 s,
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
per-peer goodput (`docs/FUTURE_WORK.md`). Do not weight samples by payload size
as a substitute: the ACK estimator already declines transfer-dominated samples
(`ACK_OBSERVE_MAX_BYTES`) precisely because they measure the payload rather than
the peer.

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
exists because the window is five minutes and the app-limited rule governs only
what happens across rotations — the first version of its test passed with the
rule disabled.

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

**Still open** (`docs/FUTURE_WORK.md` items 17 and 18): mid-reply failover now
does not happen at all. Making it WORK needs the boundary activations retained
for segments that have a standby — and that, not more standbys, is what a
multi-node standby would need first.

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
peer and ALONE for a warm one. A warm peer used to be uncapped ("it has already
paid for the weights"), which is true and was the whole story until the prompt
was 8,000 tokens: the #447 card was warm. Unknown prompt or geometry → 0, and
unknown never excludes. **The peer's own admission is the backstop**:
`model_worker::handle_forward` now runs `ensure_room_for_prompt` for the prompt
pass (`sequence_num == 0`, the work kind of #434) of a SEGMENT too, so an
over-committed peer refuses at token 0 with the 503 the coordinator fails over
from, instead of dying in attention 22 s in. A new admission check on the
`Generate` path has to be asked whether the segment path got it too.

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
