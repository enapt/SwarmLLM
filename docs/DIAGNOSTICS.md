# SwarmLLM Diagnostic Instrumentation Guide

> **For contributors and developers.** This guide covers the internal diagnostic logging system used for debugging distributed inference, networking, and pipeline issues.

> **Accuracy is enforced.** Every `DIAG:` marker listed here is checked against
> the source by `every_documented_diag_line_exists_in_the_source` in
> `tests/repo_consistency.rs` — a guide whose greps come back empty is worse than
> no guide, because a failed grep looks exactly like the thing not happening.
> 28 markers had been renamed or deleted out of the code before that check
> existed (2026-08-09); rename or remove the entry when you change a log line.
>
> **Per-message network events are `debug`, not `info`** — `Received request`,
> `DIAG: received response`, `DIAG: ResponseSent event` and `DIAG: rr_ping sent`
> fire once per request_response message, which means once per streamed token
> under load. Run with `-v` to see them. Failures stay at info/warn.

All diagnostic log lines are prefixed with `DIAG:` for easy filtering.

## Start here: one line per request

Before tracing anything through the 26 lifecycle points below, read the
completion summary. Every finished inference emits **one** line carrying the
whole route and where the time went:

```bash
grep "DIAG: request complete" node.log
```

```
DIAG: request complete request_id=1ddd2912-… route=distributed segments=2
  model=llama-3.2-1b-instruct-q8-0 nodes=0718d8b9,96842635 regions=TH,TH
  queue_ms=3 sched_ms=1 ttft_ms=180 decode_ms=1420 total_ms=1604
  prompt_tokens=22 tokens=48 tok_per_sec=33.8 tpot_ms=30.2
  predicted_ms=1890 assumed_forward_passes=64
  predicted_terms=net:640/compute:1180/prompt:70/queue:0/transfer:0/cold:0
  seg0_ms=520 seg1_ms=900 activation_bytes=39188 outcome=ok
```

`predicted_ms` is what the scheduler's cost model expected this route to cost,
and it appears only when a route was actually priced (a purely local
single-segment request has no chain to price). **It is not a target and nothing
acts on it** — it is there so the model can be checked against outcomes, which
had never been done: the field A/B for the v0.3.164 routing change measured a
topology the model said should be ~5x apart running at a dead heat, with no way
to tell whether the error was the assumed forward-pass count or the per-token
network term.

`predicted_terms` is the same prediction term by term, summed over the route's
segments (`parallax::chain_cost_terms`, whose total IS `predicted_ms`): the
per-token network charge, compute, reading the prompt, the candidates' queue,
moving the prompt's activations, and reading the weights in. Fit those against
`total_ms` across many lines to find WHICH term is off — the total alone could
not: over 333 lines (FUTURE_WORK #3, 2026-10-10) its error grew with the plan's
segments (1.08×, 2.00×, 3.35×) and not at all with the reply's length.

To check the count specifically, compare `tokens` against
`assumed_forward_passes` on the same line — the assumption is recorded rather
than looked up, so an old log still says what the constant was when it ran.
**Do not tune anything from a single request**; see `docs/FUTURE_WORK.md`.

This answers most questions on its own:

| Symptom in the line | Where to look next |
|---|---|
| `queue_ms` large | node is saturated — the router runs half of `max_concurrent_requests` at once (every request has the same tier while credits are dormant) |
| `sched_ms` large | scheduler struggling to find holders — check `-- peer serving performance --` |
| `predicted_ms` far from `total_ms` | the routing cost model is wrong about this shape — see below |
| `assemblies=2` present | the request FAILED once and retried. Whatever else the line says, start here: the first attempt's cause is in the log just above |
| `ttft_ms` large, `decode_ms` small | prefill or a cold model load, not the network |
| a LATER turn of a conversation slow, on a node without a card | was the repeated prompt priced warm? `grep "already holds the start of this prompt"` (the planner's credit, `cached_locally=`) then `prefix-cache HIT` (what the worker matched). No credit line: the worker holds nothing for it — look for `prefix-cache inserted snapshot` / `keeping its opening` on the previous turn. A credit but a chain anyway: a faster route out-priced it, which is allowed |
| `decode_ms` large, `tpot_ms` high | per-token cost — find the slow hop via `segN_ms` |
| one `segN_ms` dominates | that peer is the bottleneck; cross-check its row in the peer table |
| `route=relayed` | no direct path to a holder; ~1 extra RTT each way, see NAT section |
| `outcome=error error_type=…` | the variant name points at the subsystem |

`sched_ms` is time spent *assembling*, summed across attempts — not
time-since-dequeue, which would charge a failed attempt's whole execution to
"scheduling". `assemblies` appears only when it is >1.

Absent fields mean "not measured", never zero. `ttft_ms` and `decode_ms` are
omitted on a path that never emitted an incremental token, because there is no
honest way to split decode out of the total there.

## "The reply came back empty or one token"

`finish_reason: "stop"` cannot tell you why, and the OpenAI schema has no field
for it — a reply cut off by the caller's own stop sequence and a model that
ended its turn immediately look identical to a client. Since v0.3.116 the node
says which happened, so this is one grep rather than a testing session
(gotcha #372, from an external report that took several rounds to narrow):

```bash
# A stop sequence in the REQUEST matched almost at once — names the culprit.
grep "stop sequence in the request matched" node.log

# The model emitted end-of-turn straight away, nothing cut it off.
# Points at the prompt: check the chat template for this model.
grep "ended its turn immediately" node.log

# Finalisation removed everything the model generated (leaked markers, a stop
# matching at position 0). Older, and covers the fully-empty case only.
grep "empty after finalisation" node.log
```

Neither of the first two fires when the caller legitimately asked for a short
reply — `max_tokens: 1` yielding one token is not a fault and is deliberately
silent, so the warning stays meaningful.

**"ended its turn immediately" points at the prompt, and the prompt is not
always the text.** That warning's own advice — check the chat template, check
that the rendered prompt ends where the model expects to answer — was followed
for seven releases against an external report and was a dead end every time:
the template was fine and the prompt was well-formed (gotcha #400). What was
wrong was WHERE the model was told to continue from.

Two numbers must be equal, and they are printed in adjacent lines at `-v`:

```bash
grep -E "starting forward_through_segments|SplitModel forward pass complete" node.log
```

```
seq_num=0  index_pos=0     seq_len=5529   kv_offset=0        <- prefill wrote 5529 positions
seq_num=1  index_pos=6053  seq_len=1      kv_offset=5529     <- decode asked for 6053
```

`index_pos` is the rotary position the next token is computed at; `kv_offset` is
where the model's cache actually ends. **A gap means the model is being asked to
continue from somewhere its own memory of the prompt does not reach**, and the
logits are noise — which surfaces either as end-of-turn at once, or as one token
repeated to `max_tokens`. Both shapes, one cause.

The check that needs no log at all, and works against a node you cannot see:

```bash
# Same body twice: once with the model unloaded, once warm.
curl -X POST .../api/admin/models/$MODEL/unload -H "Authorization: Bearer $K"
COLD=$(curl -s ... -d @body.json); WARM=$(curl -s ... -d @body.json)
# usage.prompt_tokens MUST be identical. It is a property of the prompt.
```

A disagreement is the fault itself rather than a symptom of it, and it is what
`examples/release_shapes.sh` now asserts. Reply length cannot substitute: the
repetition shape passes any "more than N tokens" check comfortably.

**Three things that make this class of bug look intermittent, all of which cost
time on that report.** It fires only on the PIPELINE path, taken while the model
is not loaded — so the same request fails cold and succeeds warm, and every
retry is warm. The error scales with prompt length, so a minimal reproduction is
below the threshold at which anything goes wrong. And `chars / 4` lands in the
right ballpark, so an estimate reads as a plausible token count to anyone
eyeballing it. Compare it against something, never against your expectations.


## "Why is my model on the processor?"

The whole placement decision is greppable, in the order it is taken. On one
request against a node whose card is occupied:

```
DIAG: admitting model to GPU  model=X estimated_mb= committed_mb= budget_mb= headroom_mb=
DIAG: GPU admission refused — not enough budget at this moment          (DEBUG)
Freeing graphics memory from an idle model ... reclaimed_mb= for_model=  (reclaim fired)
No idle model could be reclaimed to fit this one                        (DEBUG; nothing eligible)
Model will run on the CPU  model=X reason=  configured_gpu_layers= estimated_vram_mb=
model-worker: Model loaded ... device=Cuda(...) | device=Cpu  vram_after_load_mb=
```

`reason=` is one of `not_enough_vram`, `configured_cpu_only`,
`gpu_too_old_for_this_build`, `gpu_stopped_responding` and (v0.3.230+)
`driver_not_answering` — the graphics driver is resetting the card (an
`nvidia-smi` blocked for 60 s or more; one slow answer past its 10 s bound is
waited out on the card, gotcha #802); it clears when the driver answers and
the model moves back by itself. Different situations that all
produce `--gpu-layers 0` and were indistinguishable before v0.3.x.
**`device=` in the worker's own line is the answer**, not the daemon's intent.

Coming back the other way (v0.3.130+):

```
Graphics memory has freed up — retiring this model's processor worker ...
Model worker stopped and its memory budget released  device="cpu"
Freeing graphics memory from an idle model ... for_model=X
DIAG: admitting model to GPU  model=X
model-worker: Model loaded ... device=Cuda(...)
```

**Things that are NOT the explanation, each having cost a session:**

- **An eviction is not necessarily the reclaim.** `try_idle_vram_unload` (timer,
  keeps a model the swarm wants for up to an hour) and
  `free_vram_for_admission` (on demand, 5 s idle floor, plans first) both log
  about freeing memory. An external tester read the first as the second and
  concluded the floor was broken; it was a third mechanism entirely (#402).
- **A model on the processor does not occupy the card.** Anything summing
  `split_models[*].estimated_vram_mb` without filtering by device is answering
  a different question — that is what `MemoryScope` exists for.
- **`cpu_reason` is a prediction about the next spawn, not a fact about what is
  running.** For a resident worker, ask `placed_on_cpu_because` /
  `cpu_placement_reason` (#401).

## "Why did my machine run this itself instead of using the swarm?"

A node that holds every layer of a model decides, on every pipeline assembly,
whether to run it alone. Since v0.3.152 the whole decision is at `info`:

> **These lines describe a request you made.** The dashboard also asks the
> scheduler what *would* happen, once per visible model card whenever any peer's
> shard total changes, and those previews log the same lines at `debug` —
> otherwise an idle node with the dashboard open spends most of its log
> describing routes for models nobody asked for (61% of it, measured). So the
> lines below appear for your request and not for the dashboard's polling; if
> you want the preview's reasoning too, `-v`. See `scheduler::Purpose`.

```
Local node has full layer coverage — single local segment
        candidates= cheapest_peer= cheapest_peer_cost_ms=
        local_runs_on_processor= parallax_routing=
        (the card runs it, or there is no peer to ask — the fast path.
         `local_runs_on_processor=false` means the card is expected to
         handle it, so no peer was sought; `parallax_routing=false` means
         the priced search was switched off and the fast path stands)
Not handing this model to peer: <reason>  peer= latency_ms= free_vram_mb= peer_tokens_per_sec= local_cpu_tokens_per_sec=
        (one line per peer that did not qualify for a whole-model hand-off)
This model does not fit our GPU, so a nearby peer runs the whole of it ...
        (whole-model delegation fired)
DIAG: pipeline candidate  node= est_tokens_per_sec= has_gpu= cost_prefill_ms= cost_compute_ms= ...
        (every holder priced over the whole model — the LOCAL line's
         est_tokens_per_sec and has_gpu describe the device the request
         would USE: on a node whose card is too small they are the
         processor's figures, gotcha #444)
This node holds the whole model but would run it on its processor; a pipeline across
  peers' cards is priced faster, so the request goes there
        local_processor_cost_ms= pipeline_cost_ms= segments= prompt_tokens=
This node holds the whole model and runs it on its processor: <reason>
        local_processor_cost_ms= pipeline_cost_ms=
        cheapest_peer= cheapest_peer_cost_ms=
        (reasons: "no pipeline across peers is priced faster", or the
         faster-looking chain includes a peer whose speed is unknown)
DIAG: parallax routing unavailable — this node holds the whole model, so it runs here
        err= cheapest_peer= cheapest_peer_cost_ms=
```

**`cheapest_peer_cost_ms` is the line to read first** when a local answer took
minutes and a peer was priced at seconds. The candidate list already prices
every holder, so the two numbers have always been in the log — but the DECISION
never named either of them, and three separate reports in one day reduced to "a
cheaper option was right there and nothing says why it was not used", two of
them reasonably inferring a penalty mechanism that does not exist. The reason
was always logged; what it was a reason ABOUT was not. A peer priced far cheaper
and still passed over means one of the stated reasons applied to it — most often
that its speed is a prior rather than a measurement, which the candidate line's
`est_tokens_per_sec` will show as zero.

The two `_cost_ms` figures are in the router's own milliseconds — decode
priced over `ASSUMED_FORWARD_PASSES` tokens plus the prompt — and are
comparable to each other, not to a wall clock. Under prompt privacy the
chosen pipeline is a boomerang, `local(0,1)` … `local(N−1,N)`, with the
peers' cards in between.

**Not the explanation**: a `DIAG: parallax routing selected chain` line with
`segments=1` beside a local-only result is the search AGREEING with the fast
path, not failing to run.

## "No reachable node holds layers X-Y … the peer that held that piece has gone"

⚠ **On v0.3.224 and older, often not true (#218).** The same 503 comes back when every holder
of that range is connected but REFUSED it for memory: each refusal bars that holder for the
request, and the re-plan, finding none left, reports the range as missing. Follow the request
id: `grep -a "<request_id>" node.log | grep -E "need about .* MB more|Not enough free
memory|Pipeline segment"` — a refusal there means the holders are full, not gone.
`peers_hosting` in `/api/admin/models` says whether anyone holds it at all.

**From v0.3.225** a refusal is reported as itself: `Not enough memory in the
swarm for <model>: the computers holding it have room for about N of its M layers` (the refused
plan went past what the holders offer — retrying will not help) or `The computers holding the
part of <model> … are online but turned it down — the last one said: …` (busy — retry later).
The router logs `DIAG: re-plan after a refusal found no other route` with the re-plan's own
error beside what it reported. "Has gone" is then true: a holder that DIED (connection closed)
did not say no.

## "Why is this route so slow?" — what prompt privacy adds (#165)

With "Start and finish on this computer" on (auto-on where this node holds both ends of a
model), a route through peers is a boomerang: the first and last layers run here, and every
token comes back here twice. From v0.3.225 each assembly that takes such a
route logs `DIAG: what keeping the first and last layers here adds to the route taken` with
`privacy_extra_ms` (the route taken minus the search's cheapest route with privacy OFF) and
`without_privacy_ms`, in the router's own milliseconds. `privacy_extra_ms=0` says privacy is
not the slow part — a far peer is; a large one, past 5 s AND past the route without it, also
puts a "Starting and finishing replies on this computer is adding about Ns…" notice in the
activity feed, once per model per ten minutes. The figure is never acted on: turning privacy
off is the owner's choice (the model's card on the Dashboard).

## An ERROR that is not a fault: reads outside the shards this node holds

`ShardReader` is a virtual view of a whole GGUF built from the shards this node
actually has. Asking it for a byte range covered by a shard the node does NOT
hold is the ordinary case — "no full model download required" is a design
decision — so it returns `UnexpectedEof` and logs at **debug**:

```
ShardReader: position is in a region this node does not hold pos=... tensor_map_sample=[...]
```

Whether that matters is the CALLER's to say. A loader that needed the tensor
fails with its own message; the load-time integrity probe ignores it. Until
2026-09-05 this was logged at ERROR, and the load-time probe read
`blk.0.attn_norm.weight` unconditionally — layer 0, which lives in the model's
first shard. A node serving a middle segment has no reason to hold that shard,
so it emitted an ERROR and then loaded and served the segment perfectly:

```
ERROR ShardReader: position is in a missing shard region pos=443912320 ...
INFO  Loaded split model segment layers="[29..32)" total=48
INFO  DIAG: LayerForward processed via worker subprocess elapsed_ms=6558
```

The probe now reads the segment's OWN first layer (`blk.<layer_start>.
attn_norm.weight`), which is both a real check of the data the worker is about
to use and one the node can satisfy. If you are reading an older log, those
ERROR lines are noise — check whether a `Loaded split model segment` line
follows.

## "A peer keeps retracting the same part" — who reinstates a withdrawn claim

A peer announces its holdings every few minutes. When that announcement omits a
shard we had recorded for it, `retain_node_shards_for_model` drops the claim and
remembers the retraction, so a stale DHT provider record cannot put it back
(gotcha #364). That should happen **once**.

Measured on the live node 2026-09-16, it happens repeatedly — 3039 events in one
log, one peer having the same GLM-4 shard retracted **344 times** over five days
at a median gap of 330 s (the announce cadence), across only 21 restarts. Since
`retain_node_shards_for_model` returns what it actually REMOVED, every one of
those is a genuine reinstatement in between:

```
grep "Peer retracted shards it no longer hosts" node.log \
  | sed -E 's/.*node_id=([0-9a-f]+) model=([^ ]+).*/\1 \2/' | sort | uniq -c | sort -rn
```

More than one line per (peer, model) is the symptom. To find what puts the claim
back, grep the line that fires at the moment a retraction is undone — it names
the call site via `#[track_caller]`:

```
grep "DIAG: a holder claim this peer had withdrawn was reinstated" node.log
```

⚠ **If every line names `src/model/registry.rs` itself, the probe is broken, not
the answer.** `Location::caller()` reports the caller of the nearest frame that
opted into `#[track_caller]`, so `record_shard_holder` and
`record_shard_holder_with_build` must BOTH carry the attribute — drop it from the
outer one and every call through it reports the wrapper's own line. Gotcha #170's
shape: correct in itself, inert in production.

⚠ **A fresh node does not reproduce it** (zero in 25 minutes), so whatever does
this needs state a new node lacks — reach for a node that has been running, or
one restored from an existing `db.redb`.

**ANSWERED 2026-09-17, by the probe, on its first firing.** The site was
`daemon/dispatch/mod.rs`'s `ShardDownloadProgress` handler, which treated
`state == Complete || progress_pct >= 100` as completion.
`acquisition::maybe_broadcast_shard_progress` broadcasts at `pct == 100`
regardless of its threshold and sends `DownloadState::Downloading` when it does
— the bytes have landed, the BLAKE3 check has not run. A download that then
FAILS verification emits that message and never the `Complete` one, so every
receiver recorded a holder for a shard the peer had just discarded, the peer's
own next announce retracted it, and the next retry reinstated it. Fixed on both
sides: `progress_claims_the_peer_holds_it` requires `Complete`, and the sender
caps an in-flight broadcast at 99 so nodes running ≤ v0.3.184 stop being misled
too.

⚠ **This is why the earlier read-through ruled the path out and was wrong.** It
searched for `Complete` messages and found none — correctly, because the
messages doing the damage say `Downloading`. "Shard-download progress (no such
messages at all)" was a true observation of the wrong predicate. When a handler
fires on `A || B`, grepping for A tells you nothing about B.

The remaining producers really are ruled out: `merge_dht_providers` (correctly
gated, and the only DHT path), the incremental single-shard announces (3318 in
that log, all passing an empty `complete_for_models`), the three full-announce
producers, and manifest registration (records the LOCAL node).

**The probe is deliberately KEPT for one release.** With the fix in, only a
genuine `Complete` can reach that call site, so if the line still fires from
`dispatch/mod.rs` the cause is a peer whose downloads keep being pruned — a
different bug — and if it goes quiet, this was it. Delete it once a deployed
release has been read.

Why it matters beyond tidiness: a claim that is back in the registry is a
routing candidate, so the scheduler can hand a segment to a peer that does not
hold those weights — which is one way a request comes to spend its first-token
deadline waiting on a peer that was never going to answer.

## "Why hasn't this node fixed its copy?" — the copy repair's own account (#217)

From v0.3.225, `swarmllm diagnostics` has a `-- copy repair --` section:
when the repair task last started and finished a pass, and for each model why that pass left
it as it was. Read it before anything else when a node keeps parts the swarm disagrees with
(`peers_other_build_nodes` names it):

- `a pass has been RUNNING since N min ago` — the task is stuck; nothing below it is current.
- `not judged this pass: its download … is still marked under way` — a download of the model
  holds judgement off (for up to 6 h by design: `model_download_under_way`).
- `checked holders disagree with parts [..], which wait for a re-check of their bytes` — a
  re-check is pending (`shards_pending_verification`), drained by the auto-manage loop.
- `… already fetched again from the upload this run` — the run will not judge those again.
- `no upload verified with HuggingFace this pass` / `could not compare its parts with
  HuggingFace` — judged by the checked holders instead (`settle_by_checked_holders`).
- `replacing parts: …` — the repair is acting; disputed parts wait while the model is in use.

## "Which copy of this model does this node hold?" — one upload per model (2026-10-02, #151)

Every node uses the same HuggingFace upload of a model (`model::canonical`); a
node holding parts that are not that upload's bytes DELETES them and fetches the
upload's own through the repair queue — from a peer when it knows the part's hash,
from HuggingFace otherwise (`model::auto_manage::canonical::replace_parts`, since
2026-10-03; before, a staged switch kept the old parts until a swap).
`/api/admin/models` → `shared_copy`: the upload the swarm uses, and
`this_computer.state` = `nothing` / `canonical` / `replacing` (`parts` waiting to be
deleted: the model is in use, or HuggingFace did not answer) / `own_file` (a `-m`
model, never touched); absent = not judged yet (right after a replacement, until the
re-fetched parts are in). `peers_other_build` should fall to 0 swarm-wide as nodes
update.

| Level | Line | Means |
|---|---|---|
| INFO | `DIAG: canonical upload — every node uses this file for this model` | an upload was verified (anonymous probe) and adopted; `replaces` names the one before |
| INFO | `An upload of this model could not be checked on HuggingFace` | skipped for 24 h (`permanent`) or 30 min; the next-best is tried |
| INFO | `DIAG: registered the canonical upload's manifest` | a node holding none of the model now fetches against the canonical upload |
| INFO | `DIAG: this node's parts are the canonical upload's` | the 64 KB-per-part byte check against HuggingFace passed |
| WARN | `DIAG: parts on this node are not the canonical upload's bytes — deleting them and fetching the upload's` | `not_the_upload` = parts whose first 64 KB differ from the upload on HuggingFace; `in_dispute` = parts whose bytes disagree with the swarm's hash (#61) — settled by the upload's own bytes. Bytes that fail the check go at once even mid-request (that request was computing garbage, and fails and re-routes instead); a copy of another layout and disputed parts wait until the model is idle (`replacing`, withheld) |
| WARN | `DIAG: deleted this node's parts that are not the canonical upload's — fetching the upload's in their place` | `deleted`, `fetching` = the upload's parts covering the same layers, queued for repair (`Fetching from the model's origin` / `P2P shard download complete` follow). A copy of another LAYOUT is deleted whole, the upload's header and manifest installed (`registered the canonical upload's manifest`) |
| INFO | `This node holds another upload of this model and cannot reach the swarm's yet — keeping it, withheld, until it can` | HuggingFace did not answer, so nothing is deleted (nothing could be fetched back); state `replacing` |
| WARN | `Replaced a header from another upload of this model` | the parts were right, `gguf_header.bin` was another upload's — replaced, model reloaded |
| WARN | `Not fetching — this source is another upload than the manifest describes` | the old splice-two-uploads path, refused |
| INFO | `Ignoring a manifest of another upload of this model` | a peer still on another upload (it switches too) |
| INFO | `Ignoring a manifest of another build of this model while this node downloads it` | before the canonical upload is known, a peer's other build may not replace the manifest a running download fetches against (#158) |
| WARN | `Refusing to load: this model's header and its tensor table describe different uploads` | a MIXED copy (#156): loading would read every tensor from the wrong place. Locally a 503 (`mixed_model_copy`); on a peer, the coordinator sees `Required shards not available`, retracts it and re-routes |
| INFO | `DIAG: this node's copy is not the swarm's upload — no longer offering it to peers` / `DIAG: offering this node's copy of the model to peers again` | state `replacing`: its parts leave every announcement within one broadcast tick (30 s; peers retract them), its manifest is not gossiped, the DHT stops naming us and a peer asking for a part gets nothing |
| WARN | `A peer's part matched our hash but is not the swarm's upload of this model` | the accept path's byte check: the hash our manifest held for that part was another upload's — the part is discarded and fetched from the upload itself; the peer is not penalised |
| INFO | `This node is keeping bytes the swarm disagrees with … It announces the part as the bytes it is` (`bytes_build=`) | a dispute (#61): the part is announced under its OWN bytes' build tag (`unhashed` = a size mismatch, announced under a tag no peer expects) — before 2026-10-03 it was announced under the build it was told of |
| WARN | `DIAG: parts on this node differ from what the holders that checked theirs against the upload agree on — this node cannot ask HuggingFace itself, so it is fetching them from those holders` | a node with NO origin to ask (offline mode, or HuggingFace not answering it): `parts` = parts whose bytes differ from what ≥ 2 connected holders that CHECKED their copies agree on, none of them holding ours (`settle_by_checked_holders`, #160). Followed by `deleted this node's parts … from=the other computers that checked theirs` and `P2P shard download complete` |
| DEBUG | `Could not compare parts with HuggingFace — judging them by the holders that checked theirs` | HuggingFace failed this pass; the rest of the pass does not ask it again (each call retries ~155 s) |
| DEBUG | `The holders that checked this part agree on other bytes than ours, and no manifest of the same layout has named their hash yet — waiting for one` | the verdict is in, the hash to check the replacement against is not: a manifest gossip round brings it |
| WARN | `DIAG: parts on this node are not the canonical upload's bytes — deleting them and fetching the upload's` with `checked_holders_disagree=[…]` | a node that CAN ask HuggingFace: parts that passed the 64 KB check but that every checked holder holds other bytes of — the check reads only a part's first tensor. Settled by the upload's own bytes (as `in_dispute`), once per part per run |

**Who checked their copy.** Since v0.3.224 an announcement lists the models whose
parts the sender's heal checked against HuggingFace this run
(`ShardAnnounce::origin_checked_models` — only `canonical` copies; a copy settled by
agreeing with peers is NOT listed, so one checked holder never counts twice). A
holder record keeps it (`checked`), set or cleared only by the holder's own
announcement. `/api/admin/models` → `peers_other_build_nodes` names the peers holding
another build of each model (#215).

**Who is right about a part — settle it from the origin.** When peers disagree about a
part's hash, hash the part's byte range straight from HuggingFace and compare: the range is
the part's first tensor's `gguf_offset − shard_offset` and its `size_bytes` (manifest), fetched
with `Range: bytes=start-end` from `resolve/main/<file>` and BLAKE3'd. First check the repo's
commit history (`/api/models/<repo>/commits/main`): a file replaced in place would make both
sides "right" for different dates. 2026-10-03: GLM-4 part 4 on this node = HuggingFace
(`35f07d7f…`), repo unchanged since 2025-04-30 — the disagreeing peers held wrong bytes.
2026-10-04: Llama-3.1-8B parts 0-1 and GLM-4 part 5 here = HuggingFace; the one peer
disagreeing held wrong bytes (#217). A part is ONE range in these models, but not always —
coalesce its tensors into runs and assert the byte count equals `size_bytes` (§ "Is a shard
actually corrupt?").

**Matching a log's build tag to a hash.** A build tag is the hash's first 8 bytes read
LITTLE-endian (`swarmllm_types::build_tag_from_hash`), so `claimed_build="1374a189dd776009"`
is hash `096077dd89a17413…` byte-reversed — `int.from_bytes(bytes.fromhex(h16),'little')`.
That is how the tags a peer announced were tied to the hashes in a contested-hash warning.
**`publisher=` in those warnings names who FIRST published the manifest, not who sent it**
(it survives every hash merge): on 2026-10-04 it named THIS node's own id while the manifest
arrived with a peer's catch-up — tie a manifest to its sender by timestamp against the
`Heard of an upload` / `Peer connected` lines.

**A peer still counted in `peers_other_build` long after an update** is a peer
that cannot replace its parts — `peers_other_build_nodes` says which. Before v0.3.224
a node with no route to HuggingFace, or in offline mode, never judged its copy at all
(`9594e1ff`, #160); since then it is judged by the checked holders, so a peer still
counted needs fewer than two connected checked holders of the part, holds another
LAYOUT (its own header, table and bytes — consistent, and no way to fetch the swarm's
header without HuggingFace), or — where it can ask HuggingFace — keeps a model with
disputed parts in use (those wait for idle). Its operator's
`shared_copy.this_computer` says which. **Or none of these:** on 2026-10-04 `e561df35`
(HuggingFace reachable, two checked holders disagreeing, copy never withheld) kept three
wrong parts ~3 h on v0.3.224 and replaced them within minutes of a restart — something
that lives for one run of the heal (#217). Note the peer's `uptime_seconds`
(`/api/admin/peers`) with every reading, so a restart is visible as the experiment it is. Before v0.3.222, one model that could
not switch also kept every model after it (in name order) from switching (gotcha
#780); since the prune-and-fetch heal there is no queue to block.

⚠ **`A peer holds a different build` is logged when a holder's build CHANGES (or a
record is re-added), not per announcement** — a peer that stops appearing in it has
not necessarily switched. Read the CURRENT state from `peers_other_build`; to see who,
restart nothing and wait for a re-add, or compare its claimed builds after this node's
next restart against the day before (the 2026-10-03 misreading, gotcha #780).

`SWARMLLM_CANONICAL_UPLOADS=0` switches all of it off (rigs and gates that link a
node's files into throwaway nodes set it; it is also the A/B control). The
header-source check (`fetch_model_header`) and the load-time header check stay
on either way.

## "Why is this node talking to a stranger?"

```
Ignoring a peer that does not speak SwarmLLM ... protocol_version= agent=   (INFO, once/peer)
Not dialling a peer that does not speak SwarmLLM  peer_id= site=           (DEBUG)
```

The second names the dial site (`pex`, `mdns`, `relay_providers`,
`connection_race`, `invite_code`). If foreign peers keep reconnecting and that
line never appears, **the dials are not coming from our code** — that null
result is what identified #404. Count connections per peer rather than trusting
the peer list, which has been clean since v0.3.125 while the reconnections
continued:

```bash
for p in $(grep -a "does not speak SwarmLLM" node.log | grep -ao "peer_id=12D3KooW[A-Za-z0-9]*" | cut -d= -f2 | sort -u); do
  echo "$p: $(grep -ac "connection established peer_id=$p" node.log)"
done   # 1 each is correct — a node must be spoken to before it can be identified
```

**Was it served locally or by a peer?** That changes which code path to suspect
entirely, and it is a response header rather than a log line:

```bash
curl -sD- -o /dev/null http://localhost:8800/v1/chat/completions ... | grep x-swarm
# x-swarm-route: local | x-swarm-segments: 1 | x-swarm-peers: 0 | x-swarm-nodes: …
```

Ask for that FIRST on any report from a multi-node setup — model, request body
and version were all obtained for the report above and none of them
discriminated, while this header would have (gotcha #374).

## Is speculative decoding actually helping?

A local model drafts ahead of itself when the reply repeats something already in
the context. One line per request, at debug:

```bash
grep "local n-gram speculation complete" node.log
# rounds=6 drafted=53 accepted=47 paused_rounds=0 tokens_per_round=8.83
```

`tokens_per_round` is the number that matters: ~8.8 means it is working, ~1.0
means this workload has nothing to copy. `paused_rounds` counts rounds where the
backoff suppressed drafting — high is CORRECT on prose, not a fault. A request
that is not alone on the worker joins the batch instead and logs nothing.

### Across computers (DSD) — on by default since v0.3.213 (off in v0.3.212)

With `speculative_decoding` and `decentralized_spec_decoding` on, a split request
whose coordinator holds a small same-family model guesses with it
(`pipeline::engine_drafter`). One line per request, at info:

```bash
grep "DSD: request complete" node.log
# drafter=qwen2.5-0.5b-instruct-fp16 proposed=71 accepted=50 final_gamma=3 alpha=0.70
#   check_fixed_ms=258.7 check_ms_per_position=0.0 draft_ms_each=119.5
```

`accepted / proposed` is how often the guesses were kept; `final_gamma` is how
far it guessed by the end. **γ settling at 1 on a loopback rig is correct** —
there a check costs less than a guess; across a real link it settled at 3-7.
`check_ms_per_position` above 0 means the far computer checks on its processor.
No line at all: grep `DSD: no drafter available` at debug — no held model shares
the target's vocabulary within a quarter of its size.

```bash
# The guesser was refused the card and runs on the processor — by design, it
# never takes memory from the model it guesses for (Tenancy::Guest, #747).
grep "The guessing model does not fit the graphics memory that is free" node.log
# The guesser failed; the reply finished at plain speed (drafting_off).
grep "DSD: the drafter failed" node.log
# A call displaced by a router retry of the same request — the worker was
# kept, not evicted (#749). Harmless on its own.
grep "superseded by a retry of the same request" node.log
```

A split reply that came back one token after a router retry, with speculation
on, is #749's shape: check for `no longer holds the conversation` beside a
`superseding the earlier attempt` line.

## "Something is slow" — split user time from kernel time FIRST

Before theorising about which code is slow, spend one command finding out
whether the CPU is running your code at all. Fields 14 and 15 of
`/proc/<pid>/stat` are the process's user and system ticks (100 per second):

```bash
read u1 s1 <<< "$(awk '{print $14, $15}' /proc/<pid>/stat)"
curl -s -m 60 -o /dev/null -w "wall=%{time_total}s\n" -H "Authorization: Bearer $KEY" <url>
read u2 s2 <<< "$(awk '{print $14, $15}' /proc/<pid>/stat)"
echo "user=$((u2-u1)) system=$((s2-s1))"
```

Read it like this:

- **System ticks dominate** → syscalls. Something is doing an enormous number of
  small operations against the kernel. This is what found gotcha #410:
  `/api/admin/models` spent 962 of 1192 ticks in the kernel because GGUF headers
  were being parsed off an unbuffered `File`, one syscall per tiny read.
- **User ticks dominate** → your code. Parsing, allocation, arithmetic.
  Optimisation and algorithms are on the table.
- **Neither, but the wall clock is long** → waiting. A lock, a peer, a timeout.

**Two readings that come free with it.** A *stable* wall time across runs
(11.30 / 11.11 / 11.14 s) is a fixed amount of work, not contention — contention
is noisy, so go and find the count. And a request whose CPU time is close to its
wall time is running on ONE thread the whole way, which for an async handler
means a worker thread is blocked for that long.

The trap it saves you from is real: "parses a lot of metadata" and "makes 820k
read calls" both fit the symptom, look identical in the source, and no amount of
reading distinguishes them. It is also why the optimised release binary was no
faster than a debug build on that path — optimisation cannot remove a syscall.

## "Did the graphics card fault?" — read the driver's events by their data (WSL2 / Windows)

`DIAG:` A worker that dies with `CUDA_ERROR_ILLEGAL_ADDRESS` (or any card error) reports it
at the NEXT call that waits for the card — "Encode Q8_0" is just the copy of a step's output
to the host, so the fault was in that step's forward, not in the encoder. Whether the
DRIVER noticed too is in Windows' System log, and an nvlddmkm event's `Message` is EMPTY on
these machines — its text is in the event's data:

```bash
powershell.exe -NoProfile -Command 'Get-WinEvent -FilterHashtable @{LogName="System"; ProviderName="nvlddmkm"; StartTime=(Get-Date).AddDays(-2)} | ForEach-Object { "{0:u} id={1} {2}" -f $_.TimeCreated.ToUniversalTime(), $_.Id, $_.Properties[1].Value }'
```

Id 13 is an engine exception ("Graphics FECS Exception" = the context-switch engine); 14
carries a raw dump; **153 is a TDR — the driver resetting the card**, with its stages
("UCodeReset", "Resetting", "Reset", "Restarting"). On 2026-10-04 a reset took 14 minutes, and
every CUDA call on the machine — a node's next worker spawn included — waited it out
(FUTURE_WORK #220). Times print in UTC, to match the node's log; match them to the second.
`Kernel-Power` id 105 ("power source change") is worth listing beside them on a laptop.

**Which kernel faulted** needs `compute-sanitizer --tool memcheck --target-processes all`
around the daemon (it follows the workers). On WSL2 it refuses ("Failed to initialize WDDM
debugger interface") until two registry values are set to DWORD 1 as administrator:
`HKLM\SYSTEM\CurrentControlSet\Services\nvlddmkm` `EnableDebugInterface` and
`HKLM\SOFTWARE\NVIDIA Corporation\GPUDebugger` `EnableInterface`.

## No log file? Use the endpoint

`GET /api/admin/diagnostics` renders plain text for a shell, and includes the
last 50 completed requests, the failure ring, per-peer serving performance
(round-trip time, ms/layer, EWMA latency, sample count, region) and what this
node has served for others. One command instead of a log excerpt:

**`-- this machine --`** — CPU, GPU, measured memory bandwidth, and the
`advertised speed` derived from it. That last figure is what every other node's
scheduler ranks this one on, so it is the first thing to read when someone asks
why work is never routed to their machine, or why their fast box loses to a
slower one. A GPU node takes its bandwidth from the card's spec table; a
processor-only node reports what `inference::mem_bandwidth` actually measured.
"Could not be measured" is a distinct answer from a low number and is printed
as one.

**`in_flight: N traces, M pipelines`** — both should be `0` on an idle node.
Non-zero with no traffic means bookkeeping has been left behind, and the trace
count is the one that bites: it is the oracle behind `model_is_in_use`, so a
stale entry makes deleting that model fail with "in use" **permanently**, on a
node serving nobody. There is no sweep behind the RAII cleanup, so this number
is the only way to see it.

```bash
swarmllm diagnostics          # safe to paste in public
swarmllm diagnostics --full   # keeps network addresses, for your own machine
```

Or straight from the endpoint, which the command is a wrapper over:

```bash
curl -s -H "Authorization: Bearer $(cat ~/.local/share/swarmllm/api_key)" \
  'localhost:8800/api/admin/diagnostics?full=1'
```

**Without `?full=1` every network address is replaced** by a placeholder naming
its kind — `<public-ip-a3f1>`, `<private-ip-…>`, `<host-…>` — while transport,
port, peer id and `/p2p-circuit` structure survive, so the report still answers
"is this node public?" and "is that hop relayed?". Two occurrences of one host
share a tag, so "ten cache entries, all the same machine" is still visible; the
tag is salted per report and means nothing across two of them. The project's own
bootstrap anchor is exempt, since it ships in every binary. The default is
redacted because the dashboard's **Copy diagnostics** button is this endpoint's
main consumer and its output gets pasted into public channels.

Per-request routing is also on **every response**, so a failing client can be
diagnosed without server access at all:

```bash
curl -i -X POST localhost:8800/v1/chat/completions -H '…' -d '…' | grep -i '^x-swarm-\|^server-timing'
```

```
x-swarm-route: distributed
x-swarm-segments: 2
x-swarm-nodes: 0718d8b9,96842635
server-timing: queue;dur=3, sched;dur=1, ttft;dur=180, decode;dur=1420
```

On a streaming response the `Server-Timing` header carries only what is known
before the body flushes (queue, schedule); the token-level figures arrive in the
final SSE usage event.

## Quick Start — Filtering Diagnostic Logs

```bash
# Run with debug logging, filter to DIAG lines only (an installed binary: `swarmllm run -vv`)
cargo dev-run -- run -vv 2>&1 | grep "DIAG:"

# Full trace (very verbose) — includes encryption nonce details
cargo dev-run -- run -vvv 2>&1 | grep "DIAG:"

# Filter to specific subsystem
cargo dev-run -- run -vv 2>&1 | grep "DIAG:.*encrypt"    # Encryption issues
cargo dev-run -- run -vv 2>&1 | grep "DIAG:.*segment"     # Pipeline segment timing
cargo dev-run -- run -vv 2>&1 | grep "DIAG:.*connection"   # Connection lifecycle
cargo dev-run -- run -vv 2>&1 | grep "DIAG:.*LayerForward" # Tensor forward path
cargo dev-run -- run -vv 2>&1 | grep "DIAG:.*SSE"          # SSE streaming path
cargo dev-run -- run -vv 2>&1 | grep "DIAG:.*KV-cache"     # KV-cache hit/miss
cargo dev-run -- run -vv 2>&1 | grep "DIAG:.*split stream"  # Split model decode loop
cargo dev-run -- run -vv 2>&1 | grep "DIAG:.*execute_request" # End-to-end request timing
cargo dev-run -- run -vv 2>&1 | grep "DIAG:.*codec"           # Wire-protocol codec frames
```

## End-to-End Request Trace

Every inference request gets a `request_id` (UUID) that appears in logs across all subsystems. To trace a single request:

```bash
cargo dev-run -- run -vv 2>&1 | grep "request_id=<UUID>"
```

### Request Lifecycle (log points)

1. **API entry** → `Queued inference request` (router/mod.rs)
2. **Dispatch** → `DIAG: dispatch_single starting inference` (router/mod.rs)
3. **Pipeline assembly** → `DIAG: pipeline assembled` with `segments`, `standbys`, `schedule_ms` (router/distributed_exec.rs)
4. **Forward start** → `DIAG: starting forward_through_segments` with `seq_num`, `index_pos`, `activation_bytes` (pipeline/distributed.rs)
5. **Tensor forward send** → `DIAG: sent tensor forward via send_request` with `is_connected`, `total_connections`, `pending_tensor_count`, `outbound_id` (manager/tensors.rs)
6. **Codec write** → `DIAG: codec write_request start/done` with `frame_len` (protocol.rs)
7. **Encryption (if enabled, R139)** → encrypt offloaded from event loop via `tokio::spawn`. Failure: `DIAG: tensor encrypt+encode failed — dropping forward` (manager/tensors.rs). On success the spawn task posts `NetworkCommand::SendEncodedTensor` back through `internal_cmd_tx`; the critical task then performs only the `send_request` step. Decode/decrypt offloaded symmetrically in the inbound path; failures log `DIAG: decrypt FAILED — possible AAD mismatch, key mismatch, or corruption`
9. **Inbound dispatch** → `DIAG: inbound TensorPayload request` → `DIAG: acknowledged tensor forward on receipt` (manager/requests.rs) — the request is ACKed immediately; the result travels back as its own request (`features::FORWARD_ACK`, 2026-08-21). A coordinator that sees no ACK within the RTT-scaled deadline logs `DIAG: tensor forward not acknowledged within the ACK deadline` (manager/mod.rs, the stale sweep) and the pipeline fails over; an un-ACKed forward no longer waits out the segment deadline.
10. **Dispatcher** → `DIAG: dispatcher received LayerForward, spawning handler` with `seq`, `layer_range`, `activation_bytes` (daemon/dispatch/mod.rs)
11. **Local execution** → `DIAG: processing LayerForward locally` with `elapsed_ms` (daemon/dispatch/layer_forward.rs)
12. **Split model forward** → `DIAG: SplitModel forward pass complete` with `forward_ms`, `seq_len`, `num_layers` (split/executor.rs)
13. **Result send** → `DIAG: LayerForward processed via worker subprocess` with `tokens`, `activations_bytes`, `elapsed_ms`, `layer_start`, `layer_end` (daemon/dispatch/layer_forward.rs)
14. **Response write** → `DIAG: codec write_response start/done` with `frame_len` (protocol.rs)
15. **ResponseSent event** → `DIAG: ResponseSent event — response written to wire` (manager/events.rs)
16. **Response read** → `DIAG: codec read_response done` with `tag`, `len` (protocol.rs)
17. **Response received** → `DIAG: received response` with `kind`, `was_tensor_forward`, `pending_tensor_out` (manager/events.rs)
18. **Response dispatch** → `DIAG: received TensorPayload response` (manager/requests.rs)
19. **Result delivery** → `DIAG: dispatcher received LayerResult` → `DIAG: LayerResult delivered to pipeline` (daemon/dispatch/mod.rs)
20. **Forward complete** → `DIAG: forward_through_segments returned OK` with `fwd_ms`, `tokens`, `activations_bytes` (pipeline/distributed.rs)
21. **Local segment** → `DIAG: local segment complete` with `segment_ms`, `activation_bytes` (pipeline/distributed.rs)
22. **Remote segment** → `DIAG: remote segment complete` with `segment_ms`, `activation_bytes` (pipeline/distributed.rs)
23. **Segment result** → `DIAG: segment result received` with `elapsed_ms` (pipeline/local.rs)
24. **Pipeline complete** → `DIAG: forward_through_segments completed` with `pipeline_ms` (pipeline/distributed.rs)
25. **Execute complete** → `DIAG: execute_request completed successfully` with `schedule_ms`, `execute_ms`, `total_ms` (router/distributed_exec.rs)
26. **Completion** → `DIAG: request complete` — the single summary line described at the top of this guide, carrying route, nodes, regions, per-phase timings, per-segment timings, tok/s, the cost model's own prediction and outcome (`daemon/state/relay.rs::publish_request_trace`, called from router/mod.rs)

All 26 points are built from one `RequestTrace` (`inference/trace.rs`), which is
also what feeds the response headers, the diagnostics ring and the Prometheus
histograms. Adding a field means adding it there once, not at each surface.

### Network Event Diagnostics

| Level | What | Where |
|-------|------|-------|
| DEBUG | `DIAG: processing swarm event` — event type name for every swarm event | manager/events.rs |
| DEBUG | `DIAG: handling outbound command` — command type for every outbound command | manager/commands.rs |
| WARN  | `DIAG: OutboundFailure` — `is_connected`, `pending_tensor_out`, `pending_channels` | manager/events.rs |
| WARN  | `DIAG: InboundFailure` — `pending_channels` | manager/events.rs |
| DEBUG | `DIAG: remote-generate stream complete` — `streamed_count`, the number the done token carries so the coordinator can tell a finished stream from one whose end overtook its middle | daemon/dispatch/remote_generate.rs |
| DEBUG | `DIAG: ResponseSent event` — confirms response written to wire. Per-message, so `-v`: at info these were three quarters of an idle node's log, and one line per streamed token under load | manager/events.rs |
| DEBUG | `DIAG: manifest received` — every VERIFIED manifest: `model`, `manifest_hash`, each part's hash prefix (`0000` = a part the sender's copy does not know), `publisher`, `sender`, `transport`. Count distinct hashes per model to see how many VERSIONS of a model the swarm is carrying — the reading that found #91's remaining traffic (#61's disagreements re-announced every round). Run it on a throwaway probe with `SWARMLLM_LOGGING_LEVEL=debug`, never by raising the live node's level | daemon/dispatch/mod.rs |

### Failure Paths

- **Timeout** → `DIAG: segment TIMED OUT — no result received` (pipeline/local.rs)
- **Outbound failure** → `DIAG: OutboundFailure` → `Tensor forward OutboundFailure — notifying pipeline` (manager/events.rs)
- **Inbound failure** → `DIAG: InboundFailure — response send may have failed` (manager/events.rs)
- **Decryption fail** → `DIAG: decrypt FAILED — possible AAD mismatch` (manager/tensors.rs)
- **No standby** → `DIAG: NO standby available for failed segment` (pipeline/distributed.rs) — carries `tried` (every node the segment was attempted on) and `last_failure` since 2026-09-02
- **Departed peer** → `DIAG: peer departed with forwards outstanding — failed them so the pipeline can fail over` (network/manager/connections.rs) — fires when a peer's connection closed AND its re-dial failed while forwards were still pinned to it; the pipeline then fails over immediately instead of waiting the segment deadline
- **Client disconnect** → `DIAG: result_tx receiver dropped` (router/mod.rs)
- **Channel drop** → `DIAG: LayerResult delivered but pipeline receiver DROPPED` (daemon/dispatch/mod.rs)
- **No pending channel** → `DIAG: No pending channel for LayerResult — timed out, duplicate, or hedge loser` (daemon/dispatch/mod.rs)
- **Streaming done event** → `DIAG: streaming done_event send failed` (router/mod.rs or router/distributed_exec.rs)

## Comparing a change against the released binary (null control)

The cheapest way to prove a change caused a difference — rather than something
ambient (peer set, model state, load) — is to run the **same data dir and the
same request** under both binaries.

```bash
T=/tmp/nullctl; rm -rf "$T"; mkdir -p "$T/models"
cp -r ~/.local/share/swarmllm/models/<model> "$T/models/"
printf '[auto_manage]\nenabled = false\nprune_enabled = false\n' > "$T/config.toml"

# candidate
SWARMLLM_NODE_DATA_DIR="$T" ./target/release/swarmllm run -p 8872 &
# ...run the probe, record the output...

# stop ONLY this node: match the data dir, never a bare pkill (gotcha #283)
for p in $(pgrep -x swarmllm); do
  tr '\0' '\n' < /proc/$p/environ 2>/dev/null | grep -q "SWARMLLM_NODE_DATA_DIR=$T" && kill $p
done

# control — the RELEASED binary, same dir, same probe
SWARMLLM_NODE_DATA_DIR="$T" ~/.local/bin/swarmllm run -p 8872 &
```

The match also catches that node's `model-worker` child, so kill the daemon, not
just the first hit. Peer-dependent probes need ~45 s after a restart for the peer
set to settle — a different failure right after start is usually the swarm, not
the change.

## "The client left — did the work stop?"

Since v0.3.152 every long wait in a request watches `InferenceRequest::cancel`
(`inference::cancel`). The lines, in order, for a client that disconnects
during the prompt pass:

```
DIAG: SSE client disconnected (connection closed) — cancelling pipeline     (streaming; flag set here)
DIAG: client disconnected before completion — cancelling request           (non-streaming, any surface; flag set here, carries request_id)
DIAG: request cancelled while a remote segment was computing — telling the peer to stop, not failing over
model-worker: skipping already-cancelled request                            (DEBUG; a queued forward dropped)
...cancelled between layers → GenerateDone finish_reason="cancelled"        (a running multi-layer forward)
```

The request then ends with `Request abandoned by the client before the reply
was ready` (a 503 nobody receives) and is NOT retried. A running prompt pass
stops at its next chunk boundary (`prefill_chunk_tokens`, 128 positions) or
layer boundary, whichever comes first; on a processor that is seconds. If a
worker stays busy longer than that after the lines above, the cancel did not
reach it: check for `CancelRequest` on the IPC path.

A non-streaming request whose client left and that never logs the second line
was not asked to stop: the line is written by `api::submit_to_router`, so a
surface that awaits the router some other way is skipping the guard (before
2026-10-10 the OpenAI path for a split model did, and its replies ran on).

## "Was this prompt read in pieces?" (FUTURE_WORK #171)

On the coordinator, at `info`:

```
DIAG: a prompt pass in pieces pieces=5 prompt_tokens=2274 from=0 segments=2
DIAG: a prompt pass in pieces completed pieces=5 pipeline_ms=57049
a prompt pass in pieces did not finish — reading it whole instead error=…   (WARN; the whole pass follows)
```

No first line on a split you expected to be cut: the plan has a boundary between two peers and
the pass does not keep its prompt (#10), a machine twice (the boomerang), a peer without `features::PROMPT_CHUNKS`, fewer than two pieces'
worth of positions (1,024 by default), an image, or `SWARMLLM_PROMPT_CHUNKS=0`. On a serving node
each piece is its own `DIAG: processing LayerForward locally` line, with `seq=0`. A piece refused with
"holds N positions … not the M the next piece of its prompt continues from" is a worker that lost
the conversation mid-pass (or a piece left from an abandoned pass); the coordinator reads the
pass whole after it.

## Measuring cancellation (what NOT to use)

`active_requests` from `/api/admin/stats` reads **0 even mid-stream** on the
local split fast path, because that path bypasses the router. It cannot measure
whether a client walking away stopped the work. Use the worker's CPU instead:

```bash
# worker pid for a given data dir
pgrep -x swarmllm | while read p; do
  tr '\0' ' ' < /proc/$p/cmdline | grep -q "model-worker.*$DATA_DIR" && echo $p
done
# utime+stime from /proc/<pid>/stat fields 14,15, sampled over N seconds
```

Healthy cancellation on this box: ~330% of one core during generation → ~30% 4 s
after the client closes (the in-flight forward finishing) → 0% by 8 s.

## SSE Streaming Diagnostics

All three streaming paths are instrumented with timing and error reporting:

### Distributed Pipeline Streaming (split_non_stream_response)

| Level | What | Where |
|-------|------|-------|
| WARN  | `DIAG: SSE role delta send failed` — client disconnected before stream started | api/openai/streaming.rs |
| WARN  | `DIAG: SSE final text delta send failed` — client disconnected on last token | api/openai/streaming.rs |
| DEBUG | `DIAG: SSE finish delta send failed` — client disconnected at finish | api/openai/streaming.rs |
| DEBUG | `DIAG: SSE stream no finish event from pipeline` — falling back to result_rx | api/openai/streaming.rs |
| WARN  | `DIAG: SSE result_rx channel dropped` — pipeline task died | api/openai/streaming.rs |
| INFO  | `DIAG: SSE distributed stream completed` — `elapsed_ms`, `token_count` | api/openai/streaming.rs |

### Split Model Streaming (split_stream_response)

| Level | What | Where |
|-------|------|-------|
| DEBUG | `DIAG: split stream model not found` — model evicted during request | api/openai/streaming.rs |
| INFO  | `DIAG: split stream decode loop complete (subprocess)` — `decode_ms`, `tok_per_sec` | api/openai/streaming.rs |
| WARN  | `DIAG: split stream client disconnected (connection closed) — cancelling decode` — `token_count`, `elapsed_ms` | api/openai/streaming.rs |
| INFO  | `DIAG: split stream completed` — `elapsed_ms`, `token_count` | api/openai/streaming.rs |

### Local Executor Streaming (stream_response)

| Level | What | Where |
|-------|------|-------|
| DEBUG | `DIAG: local stream role delta send failed` — client disconnected early | api/openai/streaming.rs |
| WARN  | `DIAG: local stream token send failed` — channel full or client disconnected | api/openai/streaming.rs |
| ERROR | `DIAG: local stream generate_stream error` — executor error | api/openai/streaming.rs |
| INFO  | `DIAG: local stream completed` — `elapsed_ms`, `token_count` | api/openai/streaming.rs |

## Encryption Diagnostics

The encrypted tensor path logs at multiple levels:

| Level | What | Where |
|-------|------|-------|
| DEBUG | `DIAG: decrypting tensor` — AAD length, sealed length, session existence | manager/tensors.rs |
| TRACE | `DIAG: seal() success` — nonce counter, ciphertext length | session.rs |
| TRACE | `DIAG: open() decryption success` — nonce, plaintext length | session.rs |
| ERROR | `DIAG: seal() encryption failed` — full context on encryption failure | manager/tensors.rs |
| ERROR | `DIAG: decrypt FAILED` — AAD mismatch, key mismatch, or corruption | manager/tensors.rs, session.rs |
| ERROR | `DIAG: open() decryption FAILED` — nonce state, AAD/sealed lengths | session.rs |

### Common Encryption Failures

**AAD Mismatch**: The sender and receiver construct AAD from the cleartext header fields (uuid + seq + idx_pos + fmt + layer_range + model_id). If these don't match byte-for-byte, decryption fails. Look for `aad_len` differences between send and receive logs.

**No Session**: The sender has an encryption session but the receiver doesn't (or vice versa). Check `has_session` in logs. Sessions are established via ECDH key exchange during peer discovery.

**Nonce Replay**: If `Rejecting replayed nonce` appears, a duplicate or out-of-order message was received. This can happen with connection flapping.

## Transport Layer

SwarmLLM uses dual transport: **TCP** (primary, Noise+Yamux) and **QUIC** (fallback).

### Port Layout

| Service | Port | Protocol |
|---------|------|----------|
| HTTP API (Axum) | `port` (default 8800) | TCP |
| P2P TCP (Noise+Yamux) | `port + 10` (default 8810) | TCP |
| P2P QUIC | `port` (default 8800) | UDP |

TCP P2P uses `port+10` to avoid conflicting with the Axum HTTP server on the same TCP port.

### Why TCP Primary

QUIC substream negotiation on WSL2 (and potentially other virtualized networks) can take **14-25 seconds per substream**. Since `request_response` serializes outbound requests through a single substream at a time, this creates a fatal bottleneck — tensor forwards queue behind health pings and never reach the codec before the 30-second pipeline timeout.

TCP+Yamux substream opening is sub-millisecond, enabling per-token round trips of ~20-26ms for distributed inference.

### Bootstrap with TCP

When connecting nodes, use TCP addresses for bootstrap:

```bash
# Node 1 on port 8800 (TCP P2P on 8810)
swarmllm run -p 8800

# Node 2 bootstraps to Node 1's TCP P2P address
swarmllm run -p 8801 --bootstrap /ip4/<node1-ip>/tcp/8810
```

## Connection Diagnostics

### Local Multi-Node Testing

When running multiple nodes on the same machine (localhost), connection management is more complex:

- mDNS discovers the local node on multiple interfaces (loopback, LAN, WSL)
- Both sides dial simultaneously, creating multiple connection attempts
- `network.max_connections_per_peer` (default 3; below 2 disables hole punching). A connection the swarm denies is forgotten by request-response, so it is never routed to (#774)
- Identify handler adds only the **connected** address to Kademlia (not all listen_addrs)
- `connection_addrs: HashMap<ConnectionId, Multiaddr>` tracks which address each connection uses

Look for `is_loopback=true` in `DIAG: connection established` logs to confirm same-machine connections.

### A request that vanished — no response, no failure (2026-10-02, gotcha #774)

A deterministic "loss" (the SAME token ids missing on every run, say) is a routing
bug until shown otherwise. Trace it end to end at `-vv` (request-response logs only appear there;
`SWARMLLM_LOGGING_LEVEL=debug` is `swarmllm=debug` alone):
- serving side, one line per token with its `token_id`:
  `DIAG: hand-off reply token queued for the requester`;
- requester, the same `token_id` (with `routed=false` when the sink belongs to
  another attempt or peer): `DIAG: streaming token arrived`;
- in between, from the vendored request-response, the connection each request was
  put on with every candidate's (id, pending, has-answered):
  `rr: request assigned to a connection`. A candidate the swarm never reported in
  `DIAG: connection established` is a ghost, and
  `request_response: forgetting a connection the swarm denied` is the fix
  removing one.
On the rig, `examples/split_rig.sh` takes a wrapper binary that appends `-vv` to
`run` (`printf '#!/bin/bash\ncase "$1" in run) exec BIN "$@" -vv ;; *) exec BIN "$@" ;; esac'`).

### Connection Lifecycle

```
DIAG: connection established  — peer_id, connection_id, count, remote_addr, is_loopback, is_dialer, total_established, total_peers, pending_tensor_forwards
DIAG: connection closed        — peer_id, cause, remaining, pending_tensor_forwards, affected_request_ids, total_peers
```

If `pending_tensor_forwards > 0` when a connection closes, those requests will get `OutboundFailure` and the pipeline will attempt failover.

## Tensor Compression Diagnostics

| Level | What | Where |
|-------|------|-------|
| ERROR | `DIAG: {label} tensor decompression failed` — zstd decompress error | protocol.rs |
| DEBUG | `DIAG: {label} tensor decompressed` — `compressed_len`, `decompressed_len`, `ratio` | protocol.rs |

## KV-Cache Diagnostics

### Multi-turn Session Cache (kv_cache.rs)

| Level | What | Where |
|-------|------|-------|
| DEBUG | `DIAG: KV-cache MISS — no multi-turn session found` — `total_sessions`, `total_multi_turn` | kv_cache.rs |
| INFO  | `DIAG: KV-cache MISS — internal session evicted` — session removed from store | kv_cache.rs |
| INFO  | `DIAG: KV-cache MISS — session expired` — `elapsed_secs`, `ttl_secs` | kv_cache.rs |
| INFO  | `DIAG: KV-cache MISS — pipeline degraded` — `missing` nodes, `total_holders` | kv_cache.rs |
| INFO  | `DIAG: KV-cache MISS — prompt prefix mismatch` — `cached_prompt_len`, `new_prompt_len` | kv_cache.rs |
| INFO  | `DIAG: KV-cache HIT — skipping prefill` — `start_pos`, `cached_tokens`, `cache_holders` | kv_cache.rs |

### Per-Request KV-Cache Store (split/kv_cache.rs)

| Level | What | Where |
|-------|------|-------|
| INFO  | `DIAG: KV-cache store cleanup — expired entries removed` — `removed`, `remaining` | split/kv_cache.rs |
| INFO  | `DIAG: KV admission — evicted cached prompts so this prompt's cache fits on the device` — `budget_mb` (as the card can honour it NOW), `load_time_budget_mb`, `live_mb`, `cached_mb`, `freed_mb` | model_worker.rs |
| WARN  | `DIAG: KV admission — refusing this prompt before prefill: it would not fit on the device` — the 503 that re-routes; `short_by_mb` is against the reconciled budget | model_worker.rs |
| DEBUG | `DIAG: KV budget reconciled with the device — less room than the load-time figure` — `load_time_budget_mb`, `budget_now_mb`, `live_mb`, `cached_mb`, `device`; fires whenever the device has less room than the loader predicted (another tenant, a snapshot, the llama.cpp context — or, on a processor, the rest of the machine filling up, or this worker's own weights growing when a failover hands it more layers). `device` is `card` or `processor`; the processor arm arrived with gotcha #462. `budget_mb` in the two lines above is this figure. | split/model.rs |
| WARN  | `DIAG: refusing to grow the KV cache past this worker's budget` — the per-chunk guard, at a growth-quantum boundary, after evicting cached prompts | split/executor.rs |
| WARN  | `DIAG: the reply's machine failed mid-stream — continuing it on a fresh route from what the reader has received` — `error`, `continuation` (1 or 2), `tokens_sent`; the reply goes on from the text already streamed, on a fresh plan (#236). The next `remote-generate fast path: request sent` (or pipeline segment lines) names where | router/mod.rs |
| DEBUG | `Told the peers this request ran on that it is over, so they free its memory` — `peers`, `told`; at the end of a request this node led, one `CancelInference` per peer any attempt ran a segment on (#238). On the peer: `CancelInference: nothing in flight for request` (debug) and the worker drops the cache | router/distributed_exec.rs |
| INFO  | `DIAG: KV cache — released conversations that had ended, for one that needs the room` — `conversations`, `freed_mb`, `idle_secs`; caches silent for `CONVERSATION_GAP_SECS` (most often segments served for ANOTHER computer, whose coordinator never says the reply ended) given up for a prompt's admission or a reply's growth (#235). Since 2026-10-08 the worker also runs the `KV-cache store cleanup` line above on its own store; before that only the daemon's store was ever swept | split/kv_cache.rs |
| DEBUG | `DIAG: KV cache growth during a prompt chunk` — `growth_steps` (concatenations this forward took, from the process-wide `KV_GROWTH_STEPS`), `chunk_positions`, `index_pos`, `reserved_positions`. **The mechanism check for a reserved prompt (FUTURE_WORK #32)**: with the reservation on it reads `growth_steps=0` on every chunk; with `SWARMLLM_KV_RESERVE=0` the old growth comes back (one `cat` per K and per V per layer per quantum crossed). Emitted on both forward paths; a fused prefill batch reports `reserved_positions` per slot. `scratchpad`-style harness: an isolated node, a ~1500-token prompt, grep this line — see `docs/invariants/memory.md` § "A prompt of known length is reserved" | split/executor.rs |

## Split Model Forward Pass Diagnostics

| Level | What | Where |
|-------|------|-------|
| TRACE | `DIAG: layer forward complete` — `layer`, `layer_ms` (per-layer timing) | split/executor.rs |
| DEBUG | `DIAG: SplitModel forward pass complete` — `forward_ms`, `seq_len`, `num_layers`, `is_first`, `is_last`, `kv_offset` | split/executor.rs |

For per-token decode analysis, combine the forward pass timing with the decode loop timing from `DIAG: split stream decode loop complete` which reports `tok_per_sec`. Use `-vvv` (trace) to see per-layer timing.

**Emulating a link on one machine**: `SWARMLLM_TEST_TENSOR_DELAY_MS=N` holds every outbound
tensor forward and result N ms in the network manager, so two nodes on one box see a round trip
2N longer (`tc netem` needs root). The split's speed at a same-city or same-country distance is
measured this way (`~/swarmllm-link-0930/sweep.sh`); it delays each message once — an encrypted
forward re-enters the handler as `SendEncodedTensor`, which is not delayed again.

**Split speculation's continuous stream** (`pipeline::dsd_stream`, ON by default on the coordinator since
v0.3.216 — `SWARMLLM_SPEC_STREAM=0` keeps the rounds; it streams only to a node on v0.3.216 or later, `features::STREAM_AS_ONE_WORK`): `SWARMLLM_SPEC_STREAM_GUESSES` guesses per
chunk, the look-ahead included (default 3), `SWARMLLM_SPEC_STREAM_WINDOW` chunks out at once
(default 3). One INFO line per reply on the coordinator, `DIAG`-free:
`DSD: streamed checks complete chunks=… kept_whole=… restarts=…` — `kept_whole / chunks` is how
often a chunk's guesses AND its look-ahead were all kept (~0.25-0.35 on a 0.5B drafter for a 7B);
each restart is logged at DEBUG (`a streamed check refused a guess`, with the turn and how many of
its guesses were kept). On the serving node a superseded chunk is answered `streamed check N was
skipped` and never computed. ⚠ **A rig that runs both halves on ONE card measures the stream's
cost, not its gain** — the far half's checks and this node's next chunk share the device, so the
stream reads ~15% SLOWER there than rounds (25-27 vs 28-33 tok/s at an emulated 24 ms, 2026-09-30),
while with the far half on other hardware it is ~15-20% faster. Put B on the processor
(`NODE_B_TOML='gpu_layers = 0\n[resources]\nmax_cpu_threads = 8'` in `split_speed.sh`) or on
another machine to see the stream work.
`DIAG:` **a stream that stalls ~52-60 s and then re-plans** (or, on a two-node split, returns a few tokens marked
`error`): on the SERVING node grep `LayerForward rejected` and `waited 60s for the check before it`. Both together
are the v0.3.213-215 shape (#767 — the per-peer cap refused a chunk, the rest waited for its turn); from v0.3.216 a
stream holds one slot, so a refusal of a streamed chunk means the node was genuinely full or the stream exceeded
`MAX_STREAM_CHUNKS_HERE` (`its stream already has as many checks here as a stream may`), and the refused turn is
stepped over, never waited for. Reproduce with a window over an old node's cap of 4:
`SWARMLLM_SPEC_STREAM_WINDOW=6 MODEL=qwen2.5-coder-7b-instruct-q4-k-m DRAFTER=qwen2.5-0.5b-instruct-fp16 GPU_A=-1
SHARDS_A=0,1,2,3 SHARDS_B=4,5,6,7 EXTRA_TOML=<DSD on> REPEAT=3 examples/split_rig.sh repeat <binary>` — v0.3.215
refused 19 chunks and cut 3 of 3 replies short; v0.3.216 refused none.

**CUDA-graph decode** (on by default since 2026-09-30; `SWARMLLM_CUDA_GRAPH=0` or `SWARMLLM_CUDA_OWN_STREAM=0` turns it off; recorded in groups of `SWARMLLM_CUDA_GRAPH_GROUP` layers, default 2, `0` = one graph per step):

| Level | What | Where |
|-------|------|-------|
| INFO | `DIAG: decode steps on this model go to the card as one CUDA graph` / `DIAG: decode steps on this model stay uncaptured` + `reason` — decided once, at a model's first decode step | cuda_graph.rs |
| INFO | `DIAG: decode graph` — `launched`, `recording_ms_per_launch` (host time from capture start to launch, which the card waits through), `updated_in_place`, `instantiated`, `uncaptured`, `refused`; at most once a minute while decoding. `instantiated` should stay at 1 per model; `uncaptured` is one per request (its first decode step) plus any step that grew a KV buffer | cuda_graph.rs |
| INFO/DEBUG | `DIAG: decode graph capture refused` — `kind`, `detail`; the first of each kind at info. `a host-to-device copy inside the capture` means something on the decode path now uploads from the host — find it with `SWARMLLM_COUNT_KERNELS=1`'s `htod` rows. ⚠ During SPECULATION (several positions per step) it was #761's signature — the f16 mirror's catch-up recorded inside the capture; since `4fac99bf` it is caught up before recording and a speculative check should log none. If one appears, compare replies at an identical prefix with graphs on and off (`SWARMLLM_CUDA_GRAPH=0`) before calling a drift a near-tie | cuda_graph.rs |
| INFO/DEBUG | `DIAG: decode graph could not be updated in place — rebuilt (first of this kind)` (info, ONCE per kind — a count of these lines is not a count of rebuilds) and `DIAG: decode graph: a group's update was refused — its nodes before and now` (debug, every time) — `group`, `before` / `now` = the graph's node census by type (`KERNEL 49, MEM_ALLOC 44, MEM_FREE 44`). A TOPOLOGY refusal names no node; the census difference is what changed (#233: 49 ↔ 50 kernels in group 0 = an id cast inside the capture) | cuda_graph.rs |
| INFO/DEBUG | `DIAG: decode graph: forwards of this many positions keep being rebuilt — they run the ordinary way for a while` / `still rebuilt every launch — resting again` — `positions`, `rest_steps`, `figures` (the weighing: graph ms a step at its rebuild share against the ordinary way, or which figure is not timed yet) | cuda_graph.rs |
| INFO | `DIAG: decode graph: rebuilt often, and still faster as a graph — no rest` / `faster as a graph even with its rebuilds — capturing again` — `rebuilt_share`, `graph_ms`, `ordinary_ms`: the card's own timeline says a rest would cost more than the rebuilds (#233). The 60-second `decode graph` line counts them as `rests_declined`. `SWARMLLM_CUDA_GRAPH_REST=count` restores the count-only rule for an A/B | cuda_graph.rs |
| WARN | `decode graph: this model's forwards of this many positions kept being refused` — three refusals of the capture's own making for one `positions` count; forwards of that many positions run the ordinary way from then on, others keep capturing | cuda_graph.rs |

**Verify the mechanism fired** by `launched` > 0 on a `--features cuda` build, and compare replies against a
run without the switches — the same kernels in the same order, so byte-identical.

## Performance Diagnostics

### Identifying Slow Requests

The `elapsed_ms` field appears at multiple points:

1. `DIAG: SplitModel forward pass complete` — time for a single forward pass (compute only)
2. `DIAG: local segment complete` — time for a local pipeline segment
3. `DIAG: remote segment complete` — time for a remote pipeline segment (network + compute)
4. `DIAG: segment result received` — time for a single segment (network + compute)
5. `DIAG: forward_through_segments completed` — total pipeline forwarding time
6. `DIAG: execute_request completed successfully` — `schedule_ms` (pipeline assembly) + `execute_ms` (pipeline execution)
7. `DIAG: split stream decode loop complete (subprocess)` — decode time with `tok_per_sec`
8. `DIAG: request complete` — total end-to-end time

If `schedule_ms` is high, the bottleneck is pipeline assembly. If `execute_ms` is high but individual `segment_ms` values are low, the bottleneck is inter-segment overhead. If a single segment is slow, check that node's compute or network latency.

### 27-Second Response Times

Common causes:
- **Timeout-then-failover**: A segment times out at 30s (the floor of `SEGMENT_TIMEOUT_MIN_SECS`; the deadline grows with layers and prompt, to 600 s), then failover succeeds quickly → ~30s total. Check for `DIAG: segment TIMED OUT` followed by `DIAG: failing over to standby`.
- **Connection not established**: Tensor sent to a peer that's not connected. Check `is_connected=false` in `Sent tensor forward` logs.
- **Encryption failure + fallback**: Encrypted send fails, falls back to plaintext, which also fails. Check for `DIAG: seal() encryption failed` logs.
- **Channel backpressure**: Result arrives but the dispatcher channel is full. Check for `Outbound channel full, dropping tensor result`. **Read `nothing_accepted_for_secs` on that line before anything else** — a few hundred milliseconds is a burst under load, while tens of seconds means the dispatcher has stopped consuming and every inbound swarm message is being dropped, not just this one (`docs/FUTURE_WORK.md` #90, gotcha #648). The line is rate-limited to one per 30 s per channel, so `dropped_since_last` is the real volume.
- **SSE fallback path**: If `DIAG: SSE stream no finish event from pipeline` appears, the streaming token channel broke and the system fell back to waiting for the full result — check pipeline errors above.

## Health Monitor Diagnostics

```
DIAG: removing stale peers      — stale_count, total_peers, active_pipelines
DIAG: cleaning up stale pending_layer_results — count, total_pending, request_ids
DIAG: cleaning up stale streaming_token_txs   — count, total_streaming
```

If stale channel cleanup is happening frequently, requests are timing out or being abandoned before results arrive.

## Network Subsystem Diagnostics

### Gossip that cannot be decrypted

```
Gossip from a peer we cannot decrypt — most likely a node on a different private network (network.gossip_network_id); ignoring
```

A `WARN` from `network/manager/events.rs`, rate-limited per sender (`suppressed_since_last` counts the repeats). Gossip is never read as plaintext. The usual cause is a node configured with a different `network.gossip_network_id`; if every peer's gossip fails, compare that setting first.

### Bootstrap Failures

```
DIAG: bootstrap dial failed            — addr, peer_id, error
DIAG: Kademlia bootstrap failed        — connected_peers
```

Promoted from DEBUG to WARN so they're visible in production. A bootstrap failure with 0 connected peers means the node is isolated.

### Shard Download Failures

```
DIAG: shard download OutboundFailure   — model, shard_index, error, bytes_downloaded
```

Shows exactly which shard download failed, how far it got, and why.

## Credit Ledger Diagnostics

```
DIAG: failed to read credit balance from database — starting at zero
```

Only logged on startup if the database is corrupted. The node will function but starts with 0 credits.

## WSL2 Mitigations

WSL2's Hyper-V Networking Stack (HNS) causes multi-address connection races when autonat/mDNS discover the WSL2 NAT adapter (10.255.255.254). With a per-peer limit of 1 (`network.max_connections_per_peer = 1`; the default is 3), both nodes simultaneously establish connections via multiple interfaces, sending mutual yamux GoAway frames that kill ALL connections. Two mitigations are available via config:

### Disable autonat/dcutr

AutoNAT and DCUtR trigger mDNS multi-address discovery on WSL2 (loopback + LAN + NAT adapter), causing connection races. Disable for WSL2 testing. Both default to `true`.

```toml
# config/default.toml or ~/.local/share/swarmllm/config.toml
[network]
enable_autonat = false
enable_dcutr = false
```

Both protocols use `Toggle<T>` wrappers — when disabled, no events are emitted and no network traffic is generated. NAT detection and hole-punching are not needed for loopback/LAN testing.

### Yamux configuration

Yamux uses 0.13 defaults with auto-tuned windows (1 GiB max connection window). Do NOT call the deprecated `set_receive_window_size` or `set_max_buffer_size` methods — they silently downgrade to yamux 0.12 which has severe substream opening delays (~30s between successful outbound requests).

### WSL2 networking mode

For best results, use mirrored networking in `~/.wslconfig`:

```ini
[wsl2]
networkingMode=mirrored
```

This avoids the virtual NAT layer that causes additional latency and routing issues.

### The graphics card slows with Windows uptime (#146, #762, WSL#41701)

`DIAG: card allocation probe — fresh memory from the driver … took_ms=N` — one line per worker
that picks the card: 16 fresh 4 MB allocations straight from the driver, timed. A few ms on a
healthy card; it grows with Windows uptime under WSL2 and only a Windows restart clears it
(`wsl --shutdown` does not). `slow=true` (≥ 500 ms) also makes the health monitor warn
(`The graphics card has become slow to hand out memory`) and, under WSL2, show the owner a
"restart Windows" notice at most every 12 h. To read the curve against uptime:

```bash
grep -a "DIAG: card allocation probe" ~/.local/share/swarmllm/node.log | grep -o 'took_ms=[0-9]*\|^[^ ]*'
# Windows boots (uptime at each line = its time minus the boot before it):
/mnt/c/WINDOWS/System32/WindowsPowerShell/v1.0/powershell.exe -NoProfile -Command \
  "Get-WinEvent -FilterHashtable @{LogName='System'; ProviderName='Microsoft-Windows-Kernel-General'; Id=12} -MaxEvents 5 | Select TimeCreated"
```

`SWARMLLM_CARD_PROBE=0` turns the probe off; `SWARMLLM_CARD_PROBE_SLOW_MS=0` makes every probe count as slow, to watch the notice fire end to end on a healthy card.

### Recommendation

For production testing, use native Linux (dual boot or bare metal). WSL2 is suitable for single-node development and basic multi-node testing with the above mitigations, but production distributed inference should run on native networking.

## Inference Subsystem Diagnostics

### Scheduler (inference/scheduler/)

| Level | What | Fields |
|-------|------|--------|
| INFO  | `DIAG: assemble_pipeline_for` | `candidates_count`, `segments`, `standbys`, `elapsed_ms` |
| DEBUG | `DIAG: gather_candidates` | `candidates_count` |
| DEBUG | `DIAG: find_standbys` | `segment_count`, `standby_count` |

### Executor (executor.rs)

| Level | What | Fields |
|-------|------|--------|
| INFO  | `DIAG: load_model` | `path`, `backend_type`, `elapsed_ms` |
| DEBUG | `DIAG: generate_stream starting` | `prompt_len`, `temperature`, `max_tokens` |

### Sampling (sampling.rs)

| Level | What | Fields |
|-------|------|--------|
| TRACE | `DIAG: sample_token complete` | `token`, `vocab_size`, `mode` (greedy or stochastic); a stochastic draw adds `temperature`, `top_k`, `top_p` |
| WARN  | `DIAG: sampling fallback` | `vocab_size`, `sum` (cumulative probability rounding) |

### Speculative Decoding (speculative.rs)

| Level | What | Fields |
|-------|------|--------|
| DEBUG | `DIAG: speculative batch` | `drafted_count`, `accepted_count`, `acceptance_rate` |

### Vision (vision.rs + pipeline/mod.rs + daemon/dispatch/mod.rs)

| Level | What | Fields |
|-------|------|--------|
| DEBUG | `DIAG: encode_images` | `image_count`, `patch_count`, `elapsed_ms` |
| DEBUG | `DIAG: merge_vision_text_embeddings` | `text_seq`, `num_vision`, `hidden`, `positions` |
| INFO  | `DIAG: precompute_vision_embeddings local` | `image_count`, `compressed_bytes` |
| INFO  | `DIAG: precompute_vision_embeddings remote` | `remote_node` |

### Chat Template (inference/chat_template/)

| Level | What | Fields |
|-------|------|--------|
| DEBUG | `DIAG: chat template applied` | `template_matched` |
| DEBUG | `DIAG: chat template applied with the system turn folded into the first user turn` | `template_matched` |
| WARN  | `DIAG: chat template failed, using gemma fallback` / `DIAG: chat template failed, using model-name fallback` / `DIAG: chat template failed, using fallback` | `fallback`; the model-name line adds `model_name` |
| DEBUG | `DIAG: no chat template, using model-name fallback` / `DIAG: no chat template, using fallback` | `fallback`; the model-name line adds `model_name`, the other `template_matched` (false) |
| WARN  | `DIAG: chat template rendered a prompt with the user's question missing` — the render is discarded for the fallback chain | — |
| DEBUG | `DIAG: build_prompt from header` (pipeline/prompt.rs) | `model`, `prompt_len` |

## Model Subsystem Diagnostics

### Shard Store (shard.rs)

| Level | What | Fields |
|-------|------|--------|
| INFO  | `DIAG: verify_shard FAILED` | `model`, `shard` |
| INFO  | `DIAG: load_all_local complete` | `model_count`, `total_shards`, `rejected_count` |

### Model Registry (registry.rs)

| Level | What | Fields |
|-------|------|--------|
| INFO  | `DIAG: register_manifest` — **only when new or changed** (`manifest_hash` differs) | `model`, `name`, `shard_count`, `publisher` |
| DEBUG | `DIAG: register_manifest (unchanged)` — a re-gossip of a manifest we already hold | `model` |
| INFO  | `DIAG: load_from_db complete` | `manifests_loaded_count` |

### HuggingFace (model/huggingface/)

| Level | What | Fields |
|-------|------|--------|
| DEBUG | `DIAG: search_gguf_models` | `query`, `repos_count`, `gguf_files_found` |

### Manifest (manifest.rs)

| Level | What | Fields |
|-------|------|--------|
| DEBUG | `DIAG: load_from_dir` | `model`, `shard_count`, `dir_path` |

### LoRA (lora.rs)

| Level | What | Fields |
|-------|------|--------|
| DEBUG | `DIAG: lora adapter loaded` | `adapter_id`, `name`, `base_model`, `rank`, `num_layers`, `size_bytes` |

### Acquisition (acquisition.rs)

| Level | What | Fields |
|-------|------|--------|
| INFO  | `DIAG: handle_acquire` | `model`, `needed_shards` |

### Auto-Manage (auto_manage/)

| Level | What | Fields |
|-------|------|--------|
| INFO  | `DIAG: evaluate_and_prune` | `disk_pressure` (decides every file), `resource_pressure` (max with VRAM — gates soft-unload only), `pressure_urgent` (disk) |
| DEBUG | `Not fetching a part prune would delete again once it landed` | `model`, `shard`, `holders`, `pressure_after` (gotcha #795) |
| DEBUG | `DIAG: register_local_shard` | `model`, `shard` |
| INFO  | `DIAG: check_and_load_model` | `model`, `available_shards`, `missing_shards`, `total_shards`, `ranges`, `ready`, `local_shard_indices` |
| DEBUG | `Skipping model — insufficient trust for auto-manage` | `model`, `trust` |
| INFO  | `Model promoted to NetworkPopular` | `model`, `holders` |
| INFO  | `HfWatcher: promoted to DemandVerified` | `model`, `repo`, `downloads` (R141 — fires at 10k for trusted publishers, 100k for unknown) |
| DEBUG | `HfWatcher: re-promotion blocked by failed-promotion cooldown` | `model`, `repo` |
| WARN  | `HfSourceGossip dropped — hf_sources at capacity` | `model`, `cap` (R141 — fires alongside `activity.hf_sources_cap_reached`) |
| WARN  | `Auto-manage: released stalled P2P download permit; the next pass fetches it from the model's origin` | `model`, `shard`, `stall_secs` (R141 — `P2P_PERMIT_STALL_SECS = 180`). With no origin to ask (none recorded, or offline mode) the next pass logs `No origin to fetch this part from` and the part goes back to its peers (gotcha #797's residual, 2026-10-07) |
| INFO  | `On-demand loading: model has shards on disk but not loaded` | `request_id`, `model` |

`DIAG:` **Is a node deleting and re-fetching the same parts?** It cannot be seen
from that node's activity feed, but every peer logs it. Count, per peer, the two
halves of the cycle in YOUR node.log:

```bash
grep -a "Peer retracted shards" ~/.local/share/swarmllm/node.log | grep -o "node_id=[0-9a-f]\{8\}" | sort | uniq -c
grep -a "was reinstated" ~/.local/share/swarmllm/node.log | grep -o "node=[0-9a-f]\{8\}" | sort | uniq -c
```

A peer near-equal in both, steadily (tens a day), with `dropped=1` every ~5 min
(`prune_cooldown_secs`) is churning; its announced totals (`Received shard
announce … shards=N`) saw-tooth. A burst on one day is more often a canonical
heal (wrong parts deleted and re-fetched, once). Before v0.3.228 the cause was
gotcha #795: the download pass fetched below the raw replica target while prune
shed above the pressure-adjusted one. On the node itself,
`DIAG: evaluate_and_prune starting` shows the pressure it shed at.

Which pass fetched it back, on the node itself: `AutoShardManager: downloading
shards` before the fetch is the download pass; `Fetching from the model's origin
— no peer copy could be verified` is the pending-fetch pass (a part once refused
from a peer). On v0.3.228 the second kept looping after #795's fix (gotcha #797,
fixed after .228; a restart empties the in-memory set). `Not fetching a part
prune would delete again once it landed — dropped from the pending fetches` is
that fix firing.

## API Subsystem Diagnostics

### Server (server.rs)

| Level | What | Fields |
|-------|------|--------|
| DEBUG | `DIAG: server startup` | `addr` |

### Admin HF (admin_hf/shards.rs)

| Level | What | Fields |
|-------|------|--------|
| INFO  | `DIAG: hf_download_shards` | `repo_id`, `filename`, `shard_count`, `peer_fair_share`, `all_shards` |

### Providers (providers.rs)

| Level | What | Fields |
|-------|------|--------|
| DEBUG | `DIAG: provider resolution` | `model_id`, `resolved_provider` |

### WebSocket (websocket.rs)

| Level | What | Fields |
|-------|------|--------|
| DEBUG | `DIAG: client connected` | `subsystem = "websocket"` |
| DEBUG | `DIAG: client disconnected` | `subsystem = "websocket"` |
| DEBUG | `DIAG: push_task exited first` / `DIAG: receiver loop exited first` | `subsystem = "websocket"` |

### Middleware (middleware.rs)

| Level | What | Fields |
|-------|------|--------|
| DEBUG | `DIAG: auth failure` | `request_path`, `auth_present` |

### Anthropic (anthropic/mod.rs)

| Level | What | Fields |
|-------|------|--------|
| INFO  | `DIAG: anthropic messages request` | `request_id`, `model`, `messages`, `stream`, `max_tokens` |
| DEBUG | `DIAG: anthropic connectivity probe` | `request_id` |
| DEBUG | `DIAG: anthropic inference path resolution` | `request_id`, `has_local_split_model`, `network_available` |
| DEBUG | `DIAG: anthropic proxying to cloud API` | `model` |

### Identity (identity.rs)

| Level | What | Fields |
|-------|------|--------|
| DEBUG | `DIAG: set_nickname persisted` | `nickname` |
| DEBUG | `DIAG: leaderboard query` | `peer_count`, `limit` |

### Metrics (metrics.rs)

| Level | What | Fields |
|-------|------|--------|
| DEBUG | `DIAG: metrics scrape` | — |
| DEBUG | `DIAG: health_ready probe` | `ready` |

### Pool (pool.rs)

| Level | What | Fields |
|-------|------|--------|
| DEBUG | `DIAG: pool_create request` | `name` |
| DEBUG | `DIAG: pool_invite request` | — |
| DEBUG | `DIAG: pool_rates_set request` | `pool_id` |

## Config Diagnostics

### Config (config/mod.rs)

| Level | What | Fields |
|-------|------|--------|
| DEBUG | `DIAG: config load_or_create starting` | `config_path`, `cli_port`, `cli_data_dir` |
| DEBUG | `DIAG: config load_or_create complete` | `port`, `data_dir` |

## Update Diagnostics

### Update Checker (update.rs)

| Level | What | Fields |
|-------|------|--------|
| DEBUG | `DIAG: check_for_update starting` | — |
| DEBUG | `DIAG: check_for_update version compare` | `current`, `latest` |
| DEBUG | `DIAG: apply_update starting` | `path` |

### A Windows node that never updates, or stops after updating (2026-10-01, #768/#769)

- **Never moves off a version** — grep its log for `No matching binary asset for this platform
  and build variant`; `expected=` names what it asked for. Every Windows GPU build up to v0.3.216
  asked for `swarmllm-windows-x86_64-cuda.exe` (#768). `release.yml` publishes the GPU exe under
  that name too (`legacy_alias`) — taken DOWN from v0.3.217 (whose restart still failed), back from
  v0.3.218, which carries the #769 fix below (a real .204 → .218 update of the GPU build PASSED through it).
  A peer's version and OS are on `/api/identity/leaderboard` (`capability.os`, `version`).
- **Updates, then is gone** — `Started by an update` → `The previous version has exited` → `Port
  … is already in use` → `Daemon shutdown complete`. **The replacement itself holds the old QUIC
  socket** (#769): `exec_into` up to v0.3.217 spawned it with std's `Command`, which passes every
  inheritable handle. ⚠ `Get-NetUDPEndpoint` names the process that CREATED a socket, so the port
  reads as "still the old pid's" after that pid is gone — the tell is that it frees ~0.1 s after
  the REPLACEMENT exits (kill it mid-wait to check). A v0.3.217 replacement waits 30 s
  (`A port this node needs is still taken`) and still stops. Fixed builds relaunch once without
  inherited handles (`relaunching without inherited handles` → `The previous version's ports are
  free — starting`), and `exec_into` no longer hands the socket down.
- ⚠ **Reproduce it with a parent that hands down an INHERITABLE socket**:
  `C:\temp\swarm-updcpu\handoff_inherit.py` (Windows Python: UDP bound on the port +
  `set_inheritable(True)`, spawn with `close_fds=False` and the handoff variable, `os._exit`).
  A v0.3.217 replacement fails under it exactly as in the real update; a fixed one survives.
  `Stop-Process` emulations and a Python parent with default (non-inheritable) sockets do NOT
  reproduce it — both passed on a broken build. The real check is still an old release updating
  itself to the published one: unpack it into `C:\temp\…`, run it hidden on a spare port with its
  own `-d` data dir and `auto_manage`/`prune` off (`realupdate.sh` there).
- ⚠ **Launch it with a Windows working folder** (`Start-Process -WorkingDirectory C:\temp\…`).
  Started from a WSL shell its folder is `\\wsl.localhost\…`, where builds up to v0.3.216 read
  WSL's `/proc/version`, decide they are in WSL2, and turn QUIC off (#770).

## Daemon Startup Diagnostics

### Main (main.rs)

| Level | What | Fields |
|-------|------|--------|
| DEBUG | `DIAG: daemon starting` | `version` |

## Credit Subsystem Diagnostics

### Ledger (ledger.rs)

| Level | What | Fields |
|-------|------|--------|
| ERROR | `DIAG: failed to read credit balance from database — starting at zero` | `error` |

### Escrow (escrow.rs)

| Level | What | Fields |
|-------|------|--------|
| INFO  | `DIAG: escrow created` | `escrow_id`, `request_id`, `amount`, `from` |
| INFO  | `DIAG: escrow release` | `escrow_id`, `reserved`, `actual`, `reconciled`, `to_node`, `state` |

### Trust (trust.rs)

| Level | What | Fields |
|-------|------|--------|
| DEBUG | `DIAG: trust score update` | `node`, `score_delta`, `new_score` |

## Crypto Subsystem Diagnostics

### Key Rotation (key_rotation.rs)

| Level | What | Fields |
|-------|------|--------|
| DEBUG | `DIAG: key rotation tick (eviction)` | `active_sessions`, `stale_evicted` |
| INFO  | `DIAG: key rotation tick (re-keying)` | `active_sessions`, `rekey_initiated` |

### Key Exchange (manager/identify.rs)

| Level | What | Fields |
|-------|------|--------|
| INFO  | `DIAG: encryption session established` | `peer_id`, `node_id`, `session_type`, `session_count` |
| TRACE | `DIAG: session already present — Identify left it intact` | `peer_id`, `node_id` |

## Infrastructure Diagnostics

### Database (db.rs)

| Level | What | Fields |
|-------|------|--------|
| DEBUG | `DIAG: db_open` | `path` |

### Identity (keypair.rs)

| Level | What | Fields |
|-------|------|--------|
| DEBUG | `DIAG: identity key loaded from disk` | `node_id` |

### Peer Cache (peer_cache.rs)

| Level | What | Fields |
|-------|------|--------|
| DEBUG | `DIAG: peer cache saved` | `count` |

### Relay (relay.rs)

| Level | What | Fields |
|-------|------|--------|
| INFO  | `DIAG: relay reservation` | `peer` |

## Files Modified

| File | Diagnostics Added |
|------|-------------------|
| `src/crypto/session.rs` | seal/open success+failure logging with nonce, AAD, key state |
| `src/crypto/key_rotation.rs` | Eviction tick, re-keying tick with session counts |
| `src/network/manager/` | Encrypted tensor send/receive, connection lifecycle, gossip audit, outbound tracking, key exchange |
| `src/network/behaviour.rs` | Connection limits, autonat/dcutr toggle state |
| `src/network/discovery.rs` | Bootstrap failures promoted to WARN with peer counts |
| `src/network/protocol/mod.rs` | Tensor decompression success/failure with sizes and compression ratios |
| `src/network/relay.rs` | Relay reservation logging |
| `src/network/peer_cache.rs` | Peer cache save count |
| `src/inference/pipeline/` | Segment timing (local + remote), pipeline total timing, failover details, wait_for_result context |
| `src/inference/router/` | Pipeline schedule vs execute timing breakdown (distributed_exec.rs), result channel delivery, streaming done event (mod.rs) |
| `src/inference/split/` | Per-forward-pass timing (executor.rs), KV-cache cleanup (kv_cache.rs) |
| `src/inference/kv_cache.rs` | KV-cache hit/miss with detailed miss reasons (expired, degraded, prefix mismatch, evicted) |
| `src/inference/scheduler/mod.rs` | Pipeline assembly timing, candidate counts, standby counts |
| `src/inference/executor.rs` | Model load timing with backend type, generate_stream params |
| `src/inference/speculative.rs` | Batch acceptance rate tracking |
| `src/inference/vision.rs` | Image encoding timing |
| `src/inference/chat_template/mod.rs` | Template matching and fallback detection |
| `src/model/shard.rs` | Shard verification failures, load_all_local summary |
| `src/model/registry.rs` | Manifest registration, DB load counts |
| `src/model/huggingface/search.rs` | Search result counts, HF_TOKEN auth |
| `src/model/manifest.rs` | Manifest load with shard count |
| `src/model/lora.rs` | Adapter load with rank, alpha, target modules |
| `src/model/acquisition.rs` | Acquisition requests, peer selection |
| `src/model/auto_manage/` | Prune evaluation (prune.rs), shard registration (download.rs), model readiness (scan.rs) |
| `src/api/server.rs` | Server startup with bind address |
| `src/api/openai/streaming.rs` | All 3 streaming paths with per-token timing, client disconnect detection, fallback path logging |
| `src/api/admin_hf/shards.rs` | HF shard download initiation |
| `src/api/providers.rs` | Provider resolution |
| `src/api/websocket.rs` | WebSocket connection lifecycle |
| `src/api/middleware.rs` | Auth failure with path context |
| `src/credit/ledger.rs` | Transaction recording with balance changes, DB restore failure |
| `src/credit/escrow.rs` | Escrow create/release with state |
| `src/credit/trust.rs` | Trust score updates |
| `src/storage/db.rs` | Database open with path |
| `src/identity/keypair.rs` | Identity key load |
| `src/daemon/dispatch/mod.rs` | LayerForward timing, LayerResult delivery, pending channel state |
| `src/health/monitor.rs` | Broadcast failures, stale peer counts, channel cleanup details |
| `src/api/anthropic/` | Messages API request entry, connectivity probe fast-path, inference path resolution, cloud proxy |
| `src/api/identity.rs` | Nickname set/gossip, leaderboard query with peer filtering |
| `src/api/metrics.rs` | Metrics scrape, health readiness probe |
| `src/api/pool.rs` | Pool create, invite, rate set operations |
| `src/config/mod.rs` | Config load source, data_dir resolution, validation complete |
| `src/update.rs` | Update check start, version compare, apply start |
| `src/main.rs` | Daemon startup |

## Coverage

Measured 2026-10-02: **393 `DIAG:` lines in 99 of the 300 source files under
`src/`** (`grep -r '"DIAG:' src | wc -l`). The tables above are the reference
for what each line means; `every_documented_diag_line_exists_in_the_source` in
`tests/repo_consistency.rs` fails the build when a documented line no longer
exists in the code. (A per-file coverage list from 2026-03-08 used to sit here;
it named files that have since been removed, so it was dropped rather than kept
stale — it is in git history.)


## Stage profiler — where a forward pass actually spends its time

`SWARMLLM_PROFILE=1` makes every forward pass print a per-stage breakdown to
stderr and reset. Stages are wall-clock and non-overlapping, so they sum to
roughly the block time; the report also prints what they do NOT account for,
which is as informative as the stages.

```
SWARMLLM_PROFILE=1 swarmllm run -p 8899
...
PROF seq_len=128 index_pos=384 layers=28 — total 10045 ms
   4571.7 ms   45.5%  attention scores + softmax + AV
   2558.2 ms   25.5%  ffn up + gate        (quantized matmul)
   1330.7 ms   13.2%  ffn down             (quantized matmul)
    848.0 ms    8.4%  qkv projections      (quantized matmul)
    ...
     30.4 ms    0.3%  unattributed (allocation, copies, dispatch)
```

Accumulation is unconditional — `Instant::now` is ~25 ns against stages that run
for milliseconds — so only the dump is gated. Implementation in
`src/inference/prof.rs`; add a stage by extending the `stages!` macro and wrapping
the call site in `timed!`.

**This is what found the CPU attention kernel** (2026-08-06): attention was 2.3%
of the arithmetic but 45% of prompt-processing time, i.e. running 37x slower per
MAC than the quantized matmul beside it. Reach for it before optimising anything
in the forward path — the previous round tuned a matmul that turned out to be a
quarter of the cost.

## Attention-kernel A/B — `SWARMLLM_FORCE_STANDARD_ATTN=1`

Forces every attention call in the process onto `standard_attention`, so the
whole daemon runs without the fused kernel and nothing else changes. Pair a run
with it against a run without it to price the fused path end to end:

```
SWARMLLM_FORCE_STANDARD_ATTN=1 swarmllm run -p 8899   # A: standard everywhere
swarmllm run -p 8899                                  # B: normal dispatch
```

It sets the *initial* value of the per-thread override that
`ForceStandardAttnGuard` manipulates, so the speculative-decoding paths that
deliberately nest a `false` guard still behave correctly — a debug switch that
changed the guard's semantics would be its own bug.

**Why this exists rather than two builds.** Two separately-built binaries differ
in link order, inlining and codegen, so a difference between them is not
attributable to the kernel (diagnosis rule 4 — prove the mechanism fired, not
just that the number moved). One binary, one branch, identical weights is the
only comparison that isolates it. This is how the CPU prefill/decode crossovers
in `run_attention` were measured, and how flash-attention-2 was priced on CUDA
when it was re-enabled.

For the kernel in isolation, without a daemon or a model, there is a microbench
at the bottom of `src/inference/layers/mod.rs`:

```
CUDA_COMPUTE_CAP=86 cargo test --release \
  --no-default-features --features dev,claude-subscription,flash-attn \
  flash_vs_standard -- --ignored --nocapture
```

It sweeps prefill and decode shapes for an MHA and a GQA model, and asserts the
two kernels agree numerically before reporting any speed figure — flash runs in
F16 where standard runs in F32, and a fast wrong answer is not an optimisation.

## Network event loop stalls (2026-08-21)

Every arm of `NetworkManager::run`'s `select!` is timed; an iteration over
100 ms logs `DIAG: network event loop stalled` with `arm=` (the interval or
queue that was being serviced, or `swarm_event:<kind>`) and `took_ms`. Every
latency this node measures — the PEX ping that becomes `latency_ms`, per-hop
pipeline timings, ACK deadlines — is measured across this loop, so a stall
here is added to every number routing uses. Zero lines on a live node is the
normal reading; it is how the relay-carried-inbound-connection bug (#356) was
separated from a loop problem in four minutes. `grep "loop stalled" node.log |
grep -oE "arm=[^ ]+ took_ms=[0-9]+" | sort | uniq -c | sort -rn` names the
culprit when there is one.

## A gossip topic reports more bytes sent than the node sent in total (2026-09-21)

Reported from the field on v0.3.196: a node whose `network_traffic.out_bytes`
read **178.9 MB** reported `swarm/models sent_bytes = 389.48 MB` — a part 2.2x
its whole — with the marginal ratio steady near **5x** across three readings.

**This is not a broken counter, and the gap is the finding.** GossipSub's
`sent`/`sent_bytes` count **attempts, once per recipient**: `msg_sent` runs at
the very top of `send_message`, before the connected-peer lookup and before
`peer.sender.send_message(rpc)`, which returns `Err` when that peer's handler
queue is full. A forward dropped for a slow peer is counted and never reaches
the interface. So `sum(topic.sent_bytes) <= out_bytes` does **not** hold by
construction — do not assert it.

What to read instead, all in the same traffic payload:

| field | means |
|---|---|
| `gossip_dropped_msgs` | counted as sent, then **expired** in a peer's queue |
| `gossip_send_failures_forward` | relays **refused outright** — queue was full |
| `gossip_send_failures_publish` | this node's own messages refused the same way |
| `<topic>.dropped_forward_msgs` | the same expiries, per topic |

A healthy node reads `gossip_sent_bytes` slightly **below** `out_bytes` — the
difference is Noise/yamux/QUIC framing, about 5%. Measured here 2026-09-21:
2.674 GB gossip against 2.830 GB total, ratio 0.94, drops zero. A node whose
ratio is **above 1** is failing to relay what the mesh is handing it, which is a
capacity problem on that node, not an accounting one.

⚠ **The log cannot answer this.** The default filter is `swarmllm=info`, scoped
to our own crate, so libp2p's own `Send Queue full. Could not send` warning is
suppressed on every default node — an 82 MB log here had zero `libp2p_*` lines
of any kind. Run with `-vv` to see them, or read the counters above, which is
why they exist. `DIAG: gossip send queue full` is our own line for the same
event and is also at `debug`, because gossipsub re-raises it per peer per
heartbeat while congestion lasts.

⚠ **`sent_msgs` is not a publish rate.** It includes forwards, multiplied by
recipients. **`published_msgs` is the field that says whether this node's own
timer is still firing** — reading `sent_msgs` as a publish rate is what made
`swarm/regions` look like it was still on a timer after v0.3.196 change-gated
it (measured after: 0.12 published/s, against 236 sent/s of pure relay).

## "no receipt acknowledgement within Ns" — late, or missing? (2026-08-25)

A peer that fails every distributed request with this looks dead. It may simply
be busy: the ACK is emitted by the network event loop, so it arrives late exactly
when that loop is loaded, and a ping RTT cannot see that.

**The sender logs ACK receipt at `debug`**, so an info-level log shows nothing
either way — absence there is not evidence. Run the node with `-v` and look:

```bash
grep 'DIAG: received response' node.log | grep 'kind="ack"'
```

ACKs present, including from the failing peer, means late-not-missing, and the
deadline is the thing to look at rather than the peer. Measured 2026-08-25: a
peer failed for about an hour, then served the same request in 6.1 s, with its
ACKs arriving the whole time. Since v0.3.124 the deadline is per-peer
(`AckRttEstimator`, RFC 6298) and backs off on a miss, so this should
self-correct — if it does not, that estimator is the place to look.

**A cheap way to reproduce without touching the live node**: start a throwaway
node (`SWARMLLM_NODE_DATA_DIR=$(mktemp -d)`, its own port, `[auto_manage]
enabled=false`) with `-v` and issue the same request to it. It joins the swarm
from gossip within about a minute.

## Is a shard actually corrupt? Ask the origin, not the swarm (2026-08-25)

**Peer agreement is not evidence in a network that copies from itself.** Two
independent peers served byte-identical bytes that failed verification here, which
reads as "our expected hash must be wrong" — it was not; the corruption had
spread. Only the model's ORIGIN settles it.

A shard is a byte range of the upstream GGUF, so fetch exactly that range and hash
it. The ranges come from `manifest.json`:

```python
# coalesce the shard's tensors into contiguous GGUF runs, in shard_offset order
for t in sorted(shard["tensors"], key=lambda t: t["shard_offset"]):
    if runs and t["gguf_offset"] == runs[-1][1]: runs[-1][1] = t["gguf_offset"] + t["size"]
    else: runs.append([t["gguf_offset"], t["gguf_offset"] + t["size"]])
# then Range-GET each run in order into one blake3 hasher
```

**⚠ A shard is NOT always one contiguous range.** One llama-3.1-8b shard has two
runs separated by a 122 MB gap; reconstructing it as a single span from
`min(gguf_offset)` to `max(gguf_offset+size)` produced 646021120 bytes against a
declared 523304960 — and a confident FALSE mismatch on a healthy file. **Assert
the reconstructed byte count equals the manifest's `size_bytes` before believing
any verdict**; that one check catches it.

Recovery, once the origin has spoken: `POST /api/admin/hf/download-shards` with
`{"model_id": …, "shards": [n]}` refetches from the origin and (since v0.3.123)
records the hash as origin-verified, so no peer's claim can displace it.

## Benchmarks

### `examples/constrained_node_test.sh` — a SMALL node, reproduced locally

Every memory defect reported from the field since #452 (#454-#457, #461-#468)
came from one 16 GB processor-only machine, and none of them reproduced on a
developer box — which is why all of them shipped. Admission keys off
`resources.max_ram_mb` rather than real RAM, so a small configured budget
reproduces the class anywhere.

Runs an isolated node (private gossip id, no bootstrap, no mDNS, auto-manage
off, a COPY of one model) and checks: an over-budget model is refused with its
arithmetic shown; the same model loads with room; `SIGKILL` on the worker —
which is what an OS OOM-kill looks like — is survived by the NEXT request; and
exactly one worker is charged afterwards.

**Isolation is not cosmetic.** With peers reachable, prompt privacy turns a
whole-model request into a two-layer boomerang, the budget never binds, and the
script silently proves nothing. It found the "worker is dead" defect on its
first run, which a by-hand attempt minutes earlier had missed by asking a second
time.



Every harness below runs against an ISOLATED node or no daemon at all. None of
them touch a running node; several used to, and that is where most of the traps
in this section came from. **The two Python harnesses are the deliberate
exception**: they measure a LIVE node over its own API, because that is what a
user gets, and they change nothing on it — they are how #432 and #433 were found
on the released binary when a tester's own node ruled out test nodes.

**Before quoting a GPU number, check WHICH DEVICE the model is actually on.**
`GET /api/admin/models` reports `cpu_placement_reason` per model, read from the
worker's recorded placement rather than re-predicted, so it stays truthful even
after the memory frees. A model demoted at admission — because another model
took the budget first — runs perhaps 5x slower and nothing about the request
says so. Measured 2026-08-31: llama-3.2-3b read a stable 9.0-9.5 tok/s against
41.0 on the same box and binary, purely because phi-3.5 had taken 5676 MB of a
6616 MB budget.

**And a back-to-back benchmark loop can prevent the recovery it is measuring.**
`worker_should_return_to_gpu` refuses to promote a worker used within
`VRAM_MAKE_ROOM_MIN_IDLE_SECS` (5 s), so a loop that issues requests with no gap
holds the model on the processor indefinitely. Inserting a 12 s gap between reps
produced 9.7 -> 15.6 -> 33.1 tok/s as it walked back onto the card. Note the
shape: **a large change with a TIGHT spread is a different configuration, not
noise** — the opposite of the contention signature, where the mean moves and the
spread widens with it. See gotcha #422.

| harness | what it measures | notes |
|---|---|---|
| `examples/prefill_bench.rs` | prompt processing + decode, driving `SplitModel::forward` directly | no daemon, no scheduler, no API in the way. `SWARM_BENCH_MODEL` (a model dir holding every shard), `SWARM_BENCH_PROMPT` (896), `SWARM_BENCH_DECODE` (32), `SWARM_BENCH_REPS` (3), `SWARM_BENCH_DEVICE=cuda`. Pair with `SWARMLLM_PROFILE=1` for the per-stage breakdown |
| `examples/qmatmul_bench.rs` | the quantized matmul against batch size | ALSO asserts the tiled path is bit-identical to the upstream ordering — run it after touching either kernel |
| `examples/tokenizer_scaling.rs` | `SplitTokenizer::encode` against prompt length | tells an O(n) tokenizer from an O(n²) one — point `SWARM_TOK_HEADER` at a model's `gguf_header.bin`. It prints `tokenizer_model` / `merges` / `scores`, which is what decides WHICH encode path a GGUF takes (#420); a doubling that quadruples the time is the signature |
| `examples/attn_bench.rs` | attention ops in isolation | ⚠ an isolated call is not a forward pass (#255/#266) |
| `examples/sysinfo_probe.rs` | what it costs to describe this machine — `System::new_all()`+`refresh_all()` against a targeted refresh, and that both report the SAME facts | the admin `stats` endpoint spent 182 ms of its 273 ms here (#417). Prints a value comparison first: a cheaper call that answers `Unknown` for the CPU name is a regression, not a win |
| `examples/spread_bench.py MODEL --arm local:only=<ourid> --arm 'peer:holds=none,only=<peer>' --arm 'split:holds=0-9,only=<peer>' [--kind decode\|prefill] [--order blocks]` | **one model on this node, on a named peer, and split across both**, interleaved (or in blocks), through the live API — decode tok/s, TTFT, prompt tok/s, this card's utilisation, and **which PATH each request took** (n-gram loop / hand-off / local generate / streaming local) with the plan's layer ranges, read from `node.log` by request id | Arms are `swarm_route` overrides, so they only SHRINK the candidate set — to measure a split between machines that could each run the model, restrict BOTH: `holds=0-3,only=P,peer=P:4-7` (`pretend_peer_holds`; gotcha #738). `--order blocks` unloads the benchmarked model (and `--unload-also` ones) before each block — never everything, the node serves peers (#737). A streamed reply carries no route headers; without the log the paths are indistinguishable (#732). First use found #125-#129 |
| `examples/stream_bench.py MODEL [--reps N --max-tokens N --port P]` | what a user gets from a running node: streaming TTFT, decode tok/s (`(n-1)/(t_last-t_first)`, a client-side window — #312), whole-request tok/s, the card's memory before/after | reads the API key from the data dir. Compare arms WITHIN one session only (decode spreads ~9-19% on this box); verify the mechanism per arm — placement log lines, `vram_after_load_mb`, `cpu_placement_reason` — not just the number. Found #432 |
| `examples/remote_checks.py [MODEL...]` | remote inference through the real swarm: route headers (`x-swarm-nodes`, `Server-Timing` per segment), one finish per stream (#414), multi-byte replies whole (#416) | non-streaming for the headers, streaming for the finish/duplication checks. ⚠ Run at steady state — ~60 s after a restart everything peer-held 503s "insufficient capacity" (rule 3). **Check the FAILURE paths too** (a model nobody holds, streaming): that is where #433 was |
| `examples/frontend_load_check.js` | **loads every frontend module in `index.html`'s order and reports what throws.** `node -c`, which the pre-push check runs, is a syntax check and nothing more (#568): a reference to a deleted symbol at module scope passes it, and so does a component whose IIFE throws on load. Also checks `index.html` does not point at a missing script, and that a short list of exports other components depend on still exists | ⚠ **A pass is not "the frontend works".** The DOM stub is shallow on purpose. A missing `var U = App.utils` — the R111 regression — passes cleanly, because `U` is only referenced inside functions no load-time code runs; planted and confirmed. Each thing it DOES catch was verified by planting it, and the file lists which |
| `examples/peer_path_matrix.sh [base] [model]` | **which computers actually serve one model, across every routing shape** — baseline, every layer elsewhere, first part local (the boomerang), and each peer excluded in turn. Reads `x-swarm-route` / `x-swarm-nodes` / `x-swarm-regions` off the response, so it needs no log access | Uses the `swarm_route` request field, so nothing moves on disk and nothing restarts — the whole matrix is a few short prompts. ⚠ **It can only make the candidate set SMALLER**: a row that fails may be saying the swarm has no route, not that routing is broken, so read NODES rather than just the result. A peer holding the whole model wins every route until excluded, which is what the per-peer rows are for. Treats a 200 with no text as a failure, not a pass, and prints each reply's first 60 characters (since 2026-10-04) — READ them: a split through wrong bytes answered `给给给…` with a 200 (#156) |
| `examples/smoke_test.sh [binary] [port]` | 9 end-to-end checks on an isolated node | run it on the DOWNLOADED release artifact, not a local build (#268) |
| `examples/release_shapes.sh [binary] [port]` | 7 pre-release shape checks — cold start, long cold prompt, `prompt_tokens` agreeing cold and warm (#400), greedy determinism WITH a live control, tool-heavy | also on the DOWNLOADED artifact, BEFORE tagging. Local verification used to be a strict subset of CI's |
| `examples/check_ci_gate.sh [owner/repo] [branch]` | **does branch protection still require the checks CI actually produces?** Reports drift BOTH ways — required-but-never-produced (blocks every PR for ever) and produced-but-not-required (the job runs and gates nothing) | A required check is matched to a job by NAME, so a rename leaves the rule naming a job that never reports. Two of this repo's were in that state on 2026-09-10 and every PR was unmergeable (gotcha #530). Reading protection needs admin rights the workflow `GITHUB_TOKEN` does not have, so this is a script you run, not a job — **part of the release gate**. Exit 1 on drift, 2 if it could not check |
| `examples/family_conformance.sh [binary] [port] [model...]` | **whether each model FAMILY produces a sane reply, or only bytes.** Per family: it answers a question with a checkable answer, it stops by itself (`finish_reason`), no control marker reached `content`, a tool call was parsed into `tool_calls` rather than left as text, the model's own template rendered (no silent fallback), nothing logged as ERROR | The four releases v0.3.169-v0.3.172 each fixed a field-reported, family-specific prompt/stop/tool defect, and **all four pass `release_shapes.sh`** — which runs one family and asserts `>3` tokens came back. Its tools check sends twelve schemas with a prompt that should produce no call and verifies a REPLY exists, so a node where tool calling is entirely broken passes it. Unit tests cannot cover this either: they render templates and compare strings (llama.cpp's `test-chat-template.cpp` shape), while every one of those bugs was GENERATION-level — the reply was produced, and wrong. Absent or metadata-only models are reported as COULD NOT RUN, never as passes, and a request that returns no `choices` retires the whole family to COULD NOT RUN rather than emitting checks that measure nothing |
| `examples/swap_patience.sh` | what the GPU swap floor costs, in CONVERSATION | two models that each fit the card but not together, alternating multi-turn so a warm prefix is worth something. Arms switched by `SWARMLLM_VRAM_SWAP_MIN_IDLE_SECS`, never by rebuilding. Measured 2026-08-28: floor 60 s → 299 s, floor 0 → 82 s, floor 5 s → 89 s (#403) |
| `examples/soak_test.sh [binary]` | sustained inference, sampling worker RSS / KV / threads / fds / ok-fail | `HOURS=` must be a WHOLE number (shell arithmetic); data dir is `/tmp/swarm_soak-$PORT`, per-port so two soaks cannot kill each other; analyse with `soak_report.sh` |
| `examples/departed_peer_test.sh` | #436 — a segment's peer killed mid-request must fail over in seconds | Two ISOLATED nodes with partial COPIES of one model (both partial, or the whole-model fast path swallows the test); asserts the `peer departed with forwards outstanding` DIAG and a prompt failure. Passed 2026-09-02: clean 503 in 10.4 s |
| `examples/release_shapes.sh [binary] [port] [model]` — note (2026-09-03) | the request shapes smoke cannot see | Since v0.3.150 the long-prompt check is REFUSED (503, no usage → "COULD NOT RUN") when the shapes node shares the card with a live node: the two models leave it a few hundred MB of KV budget and admission now says so instead of squeezing the cache onto host memory. Run it with the live node stopped for 7/7 (0.3.150 artifact: 7/7 with the card free, 6/7 + 1 could-not-run beside the live node) |
| `examples/split_rig.sh split\|kill\|failover\|repeat\|fetch <binary> [binary-for-B]` | a model SPLIT across nodes on one machine — the shape conformance never makes. `split`: two greedy questions through the split. `kill`: break one side mid-reply (A's worker, B's worker, or an unload on B). `failover`: FOUR nodes, B's segment covered only by C+D together, B's worker killed at the start of its prompt pass (#17's composite stand-in). `repeat`: the long greedy prompt REPEAT times — request 1 takes the n-gram path, the rest the standard loop — for scoring against llama.cpp (#106). `fetch`: B downloads a part from A over P2P and the rig prints every event-loop stall B logged (#108). `EXTRA_TOML` is appended to every node's config, one variable per arm | Hard-links shards, so no disk. **Refuses to run beside ANY other SwarmLLM process** — every node's loopback probe dials ±10 of its own port and the 8800/8900/… bases whatever its config says, and peer exchange then brings in the public swarm (gotcha #708). `failover`'s PASS is mechanism only: the takeover DIAG logged, ZERO router retries (a failed takeover is rescued by a retry that produces a correct reply — gotcha #706), every request answered. Byte-equality with the control is printed as information and is NOT the test |
| `examples/outvoted_rig.sh <binary for A and C> [binary for B]` (`HFLESS=proxy\|offline`, `WAIT_SECS`) | **does a node that cannot reach HuggingFace replace a wrong part?** (#160) B holds TinyLlama with its last part zeroed under a manifest vouching for it and no route to HuggingFace (dead proxy) or offline mode; A and C reach HuggingFace and check theirs. PASS = B logs `differ from what the holders that checked theirs`, deletes the part, fetches it over P2P, ends BLAKE3-identical to the live node's | Live node STOPPED (#708). A and C re-fetch their own part from HuggingFace once first (they adopt B's vouched hash → a dispute) — expected. Heal ON with hard-linked parts is safe: the heal unlinks or renames, never writes into a part; everything it may rewrite is a copy. .223 as B = the null control (FAIL, keeps the part 600 s) |
| `examples/score_against_reference.py [--lora adapter.gguf] <model.gguf> <replies.jsonl> <prompt-file> [labels]` | **is a greedy reply the model's, or only plausible text?** Teacher-forces each reply through llama.cpp and prints, per reply, how many tokens the reference ranks first, the worst rank and the largest logit gap | The test for a split or failed-over reply, where byte-equality is not: two runs of one topology can split at a near-tie. Compare SCORES — a takeover should score like a same-topology control (109/120 rank-1 each on 2026-09-25), and a broken cache is nowhere near. Needs llama-cpp-python and a whole GGUF (`~/swarmllm-ref/`). Its prompt count matching the coordinator's `prompt_tokens_local` proves the template rendered the same. **`--lora`: is a LoRA reply the adapter's?** `examples/peft_lora_to_gguf.py` turns a PEFT adapter into llama.cpp's form (q/k permuted for Llama/Mistral as its converter does; `--no-permute` writes the WRONG layout, to prove a comparison can tell them apart). ⚠ Use an adapter with non-zero B — a fresh PEFT init has B = 0 and applies nothing, so it passes every check (#110: `tinyllama_lora.safetensors` is one). ⚠ Compare on a BPE model (Llama-3.2-3B, Qwen): TinyLlama's prompt tokenizes differently in llama.cpp (SPM whitespace) and the base replies already differ |
| `examples/dropped_token_test.sh [binary]` | #438 — one content token of a peer-served reply LOST on the serving side must still yield a whole reply | Two ISOLATED nodes, the server holding the whole model (so the client's only route is the fast path). The server drops content token 5 of every reply once (`SWARMLLM_FAULT_DROP_STREAM_TOKEN=5`); the fix arm asserts `asking the peer to resend` + `resending tokens the coordinator never received` and no `gave up`; the control arm (`SWARMLLM_RESEND_TOKENS=0` on the client) must reproduce the old truncation, or the test cannot see the fix |
| `examples/cold_load_test.sh <server> <client> [load\|leave\|price\|both]` | #129 — a peer handed a whole model must be WAITED FOR while it loads, given up on at once if it leaves, and its load PRICED at its own rate | Two nodes in a private network namespace (it re-runs itself under `unshare -rn`, so it can run beside a live node). The server's model load is delayed (`SWARMLLM_FAULT_LOAD_DELAY_SECS`, default 200 s — well past the old ~148 s budget). `load`: the reply must arrive (client log: `fast path: request sent … cold_loads=1`). `leave`: the server is killed 20 s into the load; the client must fail within ~5 s with `disconnected before its first token`. Run it again with a client that predates the fix (v0.3.229): it fails `load` at ~132 s (`timed out waiting for token (first=true)`) and waits out `leave` for ~132 s — or the run cannot see the fix. `price` (`PRICE_DELAY`, default 20 s): client A, which has heard no rate, logs `pipeline candidate node=<server> … cold_load_ms=` at the prior (10 s/GiB × the model); the server logs `DIAG: model load timed` (field `ms_per_gib`); after an unload and a 40 s broadcast a fresh client B logs that rate × the model. ✅ 2026-10-07: 6,229 ms prior, 33,818 ms/GiB timed, 21,064 ms priced |
| `examples/ceiling_test.sh <server> <client>` | A tester's report, 2026-10-07 — a peer must never be planned more than it could EVER hold (`NodeCapability::model_memory_ceiling_mb`, weighed with its own admission arithmetic) | Two nodes in a private network namespace (re-runs itself under `unshare -rn`). The server holds TinyLlama on its processor; its own admission names the footprint (asked locally under a 64 MB cap: "needs about N MB"). `below`: server cap 80% of N — the client must refuse ITSELF at once (`Not enough memory in the swarm … room for about K of its L layers`) and the server log must show no request (`server asked: 0`). `above`: cap N + 64 MB — served (the ceiling excludes nothing admission accepts). The client is given the model's HEADER only: a connected coordinator fetches it from HuggingFace before planning, and the namespace has no internet — without it the ceiling is unknown and the run measures nothing (the client's `pipeline candidate … max_hostable_layers_at_ceiling=None`). Control: a v0.3.229 client asks the server 3 times in `below`. ✅ 2026-10-07: N = 980 MB; below (784 MB) → ceiling 16 layers, 503 in 0.1 s, server asked 0; above (1044 MB) → 22 layers, 200; v0.3.229 client: server asked 3 |
| `examples/capability_gate_rig.sh <binary> [out] [window_s]` | #91 — an idle node's capability goes out on a change or every 5 min (`CAPABILITY_HEARTBEAT`), not every round | Two empty nodes in a private network namespace (re-runs itself under `unshare -rn`); counts the `NodeCapabilityUpdate` first copies B receives over the window (default 420 s). Expect 1 (the heartbeat); ~14 on v0.3.233 or older. `VERBOSE=1` prints node A's `DIAG: capability published` lines (debug, field `why`): `first`, `heartbeat`, or the names of the fields that differed — a field name on an idle node is a figure holding the gate open. `CHURN=1` allocates and frees 1.5 GB every 15 s beside the nodes (a busy host moves free RAM). ✅ 2026-10-10: v0.3.232 14; free RAM deadbanded 2 (`ram_available_mb` ×2); free RAM carried over + CHURN 1 (`first`, `heartbeat`) |
| `examples/carry_test.sh <S-and-K binary> <C binary>` | #231 — a model its holders cannot run is fetched by a machine that can (`auto_manage::coverage`), then served | Three nodes in a private network namespace: S holds the model whole, capped at half its own admission footprint; C holds part 0, plenty of memory, auto-manage on (prune off — the prune half is unit-tested), `min_replicas = 1`; client K in another region with a 64 MB cap. Expect K's first request refused at once (`Not enough memory in the swarm … room for about K of its L layers`), C to log `DIAG: the computers holding this model cannot run it — fetching parts of it to carry it (#231)` once K's demand arrives (K decays its counts every 600 s, then gossips: ~11 min) and fetch part 1 (the `requesting shard download … score=` line carries the ×50 carry bonus), then K's next request served. A carrier that gains no part for 20 min is passed over for 2 h: `DIAG: the machine chosen to carry this model has made no progress — passing it over for the next (#231)`. ⚠ **Read the carry line, not the outcome**: at `min_replicas = 2` C fetched the part by ROUTINE replication 30 s in. Control: a v0.3.229 C never fetches. ✅ 2026-10-07: first 503 in 0.0 s (19 of 22 layers), carry named 10:06:14 (score 1500 vs routine 30), part 1 fetched, second request 200 in 1.2 s |
| `examples/two_node_test.sh`, `3node_setup.sh`, `3node_sharded_setup.sh` | cross-node paths | EXPECTED to fail on a single multi-interface host — that is the documented connection-churn case, not a regression. Validate on two real machines |
| `examples/decode_bound_by.py [model] [tokens] [reps]` | **what bounds decode: the GPU, or one CPU thread.** Worker CPU-time per token from `/proc/<pid>/stat` against wall time per token, with GPU utilization sampled alongside | No profiler, no restart — the cheap FIRST reading. `cpu/token ≈ wall/token` means a CPU thread is the bottleneck and the card is waiting; `<<` means the cost is GPU-side. ⚠ Many cores busy = the CPU backend, not the GPU — check placement first. Meaningless if a PEER served the request, so it prints the route header |
| `examples/decode_submissions.sh [model] [tokens] [binary]` | **GPU submissions per decoded token** — `cuLaunchKernel` / `cuMemsetD8Async` / alloc / event counts, from nsys, counted over the steady-state decode window only | Judge a submission-count change by the COUNT; it is deterministic, while tok/s here spreads 10-18%. ⚠ Per-call TIMES are nsys-inflated — never quote them as the real cost. RESTARTS the node it profiles |

### Where a decode token actually goes (2026-09-22)

The first reading to take when a decode number will not move, and the order to
take them in. Established that GPU decode on this box spends most of a token
submitting work rather than doing it — 1,085 submissions and 17.6 of 23.0
ms/token in the driver API, card at 52%. Full evidence and the numbers per
model: `docs/invariants/inference.md` § "A decode token is bound by GPU
submission COUNT, not bandwidth".

1. **Is it even bandwidth?** Compare `ms/layer` across models of different
   size. `SWARMLLM_PROFILE=1` makes the worker print a per-stage breakdown per
   forward pass, and its `total` brackets `SplitModel::forward` alone — no IPC,
   sampling or HTTP. **If `ms/layer` is flat while bytes/token moves, the cost
   is per-layer dispatch and model size is not the variable.**
   ⚠ Because sampling is OUTSIDE that bracket, no profile ever showed it — and
   until 2026-09-24 it cost ~1.8 ms a token at a 152k vocabulary (default
   top-k 40 / top-p 0.9), about a tenth of a GPU token on the small models. Price it
   separately: `docs/invariants/inference.md` § "Top-k shrinks the candidate
   set before anything else runs".
2. **GPU or CPU?** `examples/decode_bound_by.py`. Costs nothing and rules out
   half the hypotheses.
3. **How many submissions?** `examples/decode_submissions.sh`.
3b. **Which kernels, and did my change remove the one I think?**
   `examples/kernel_count_ab.sh VAR ON OFF` runs both arms of one binary with
   `SWARMLLM_COUNT_KERNELS=1`, prints the per-kernel-name launch counts side by
   side, and **diffs the generated text**. It needs no profiler, so it is cheap
   enough to run on every fusion — and it is the right instrument, because a
   single fusion is worth ~1 launch per layer, well under the ~10% this box's
   clock can resolve.
   ⚠ **Take the reply diff as seriously as the counts.** Fused kernels here are
   written bit-identical to the candle ops they replace, so a reply that moves
   is a correctness bug, not rounding.
   ⚠ It reads the LAST `seq_len=1` block: prefill's mix is different and much
   larger, and the warm-up request's forwards are in the same log.
3c. **Whose host→device copies?** The same switch prints `htod` rows under the
   kernel table: every `cuMemcpyHtoDAsync` since the previous forward, by the
   SOURCE LINE that asked for it (`#[track_caller]` through the vendored
   candle's tensor-creation chain, so a `Tensor::new` reports OUR line and a
   copy candle makes for its own reasons reports the op's line —
   `cuda_backend/mod.rs` `copy_strided_src` is a `.contiguous()` of a strided
   view). `kernel_count_ab.sh` diffs them per arm. It is the only attribution
   this box has: nsys cannot take CPU backtraces on WSL2, and the release
   binary is stripped. First use (2026-09-24): 22 of TinyLlama's 24 copies per
   token were one per layer, attention copying the whole V cache
   (`layers::value_for_matmul`) — where the plan had read them as fixed per token.
4. **Where inside a layer?** `SWARMLLM_PROFILE=1` **plus**
   `SWARMLLM_PROFILE_SYNC=1` for correct per-stage attribution on CUDA.
   ⚠ **Read the two runs for different questions.** Sync inserts a
   `cuStreamSynchronize` at all 11 stage boundaries of every layer — 242 device
   round trips per token on a 22-layer model — so it answers "where does the
   time go" while inflating "how long does it take" (12.0 → ~29 ms/token
   measured). The CHEAP stages are the ones it distorts most: `residual adds`
   read 2.5 ms and `rms norms` 2.7 ms for work on a few KB, which is the sync
   cost, not the add. Trust it for the big stages only.
   Since 2026-09-23 those two are ONE stage, `residual add + rms norm`: on CUDA
   the add is fused into the norm's kernel (`inference::residual_norm`), so a
   separate add no longer exists to time. Older readings quote them apart.

⚠ **`SWARMLLM_PROFILE=1` prints ~11 lines per forward pass**, i.e. per token.
The dump is excluded from the `total` it reports, so per-forward figures stay
clean — but end-to-end tok/s measured with it on is NOT comparable to a normal
run. Take user-visible throughput with profiling off.

⚠ **A processor decode A/B is taken at the DECODE width a node runs, not at the
core count** (2026-09-26, #119). `examples/prefill_bench` with
`RAYON_NUM_THREADS=8` leaves decode to calibration, which picks 4 on the Ryzen
5800H; forcing `SWARMLLM_DECODE_THREADS=8` doubled every decode figure there
(llama-3.2-3b at ~544 cached: 62 → 135-140 ms/token), swung ±10 ms between
identical runs, and read a Qwen2.5-7B kernel A/B as level where the same pair at
4 threads read 141.6 → 134.1. Pin the width explicitly for an A/B
(`SWARMLLM_DECODE_THREADS=4` here) and say which one a figure came from.

### Where a split's token goes (2026-09-29)

For "a split is slower than it should be" with the network ruled out. Run both nodes at
`-v` and read, per decoded token, the coordinator's `starting forward_through_segments` →
each worker's `DIAG: worker forward received` → the executor's `forward pass complete`
(`forward_ms` = launch time of the layers) → `DIAG: worker forward answered`
(`received_to_computed_ms`, `computed_to_sent_ms`) → `sent tensor forward` / `processing
LayerForward locally` (the network) → `segment result received`. `~/swarmllm-split-0929/`
holds the harness (`split_speed.sh`: `local` / `split` / `split2` arms, streamed timing in
`stream_time.py`) and the merge-by-timestamp breakdown used on 2026-09-29, which found 6.6 ms
per node per token in the batch scheduler (#148).
⚠ **Split the tokens by their total before taking medians** — the first fix left a bimodal
distribution (21-23 ms and 33-36 ms) whose median described neither; the slow half was the
previous request still counting as active.
⚠ **Streamed deltas count tokens only when each delta is one token** — speculation emits
several per delta; pair it with `completion_tokens`.

### Current baseline — 2026-08-29, v0.3.132-alpha

(Dated 2026-08-29 and not re-taken since for the CPU rows. The GPU row predates CUDA-graph decode, on by
default since 2026-09-30, and the fused kernels: `docs/plans/local_decode_submissions.md` and `memory/perf_spread_0927_gpu.md`
carry newer GPU figures. Re-take before quoting any of it as today's.)

**Re-take with the same command before claiming a delta.** These were taken on
an idle box (AMD Ryzen 7 5800H / RTX 3070 Laptop, WSL2) with the live node
running but idle.

```bash
SWARM_BENCH_MODEL=~/.local/share/swarmllm/models/llama-3.2-3b-instruct-q4-k-m \
RAYON_NUM_THREADS=4 SWARM_BENCH_REPS=3 \
./target/release/examples/prefill_bench     # --no-default-features --features dev
```

| metric | 2026-08-15 (v0.3.97) | **2026-08-29 (v0.3.132)** | |
|---|---|---|---|
| CPU prompt processing | 20.97 tok/s | **32.58 tok/s** (896 tok in 27.51 s) | 1.55x |
| CPU decode | 4.71 tok/s | **10.44 tok/s** (95.7 ms/tok @ ~912 KV) | 2.21x |
| model load, 28 layers | 14.0 s | **5.3 s** warm | |
| KV cache | 235 MB alloc / 213 used / 91% | unchanged | |
| GPU, end-to-end via the API | — | **45.4 tok/s** warm (17.7 cold) | |

`RAYON_NUM_THREADS=4` is half the physical cores and is kept only for
comparability with the 0815 series — a run at 8 threads is a different
configuration, not an improvement.

The GPU row is **end-to-end through `/v1/chat/completions`** (200 tokens,
`temperature=0`), so it includes prefill, templating and HTTP. It is a
user-visible number and is **not** comparable with the CPU rows, which drive
`SplitModel::forward` with no daemon in the way.

Two things that will otherwise be misread:

- **Decode spread is now 8.8%** (104.1 / 95.7 / 99.1 / 96.3 ms across four runs),
  against the ~3.5% recorded in the 0815 baseline. Prefill is still tight
  (2.4%). So this box currently cannot resolve a decode change below ~10% —
  quoting the old 3.5% would license a false positive. Re-check the spread
  before trusting a small delta, rather than assuming the recorded one still
  holds.
- **A first run after a build reports model load at ~68 s, not ~5 s.** That is
  cold page cache — a release build evicts it and the shards are ~2 GB on the
  WSL vhdx. An immediate re-run loaded in 5.3 s with prefill and decode
  reproducing to within 2.4% and 0.6%. **Do not report a load-time regression
  without a warm second run.**

### Traps that have cost real time

- **The box must be idle.** The same unchanged code path measured 0.42 ms and
  0.97 ms here. A run taken while a build or another bench is going is worthless,
  and it will not look wrong — it will look like a result.
- **min-of-N is for BENCHMARKS, not live measurement** (#367). Controlled
  environment, every error adds time → the minimum is the least contaminated.
  Samples taken from live traffic are different tokens at different cache
  lengths on a busy machine → the minimum is the LUCKIEST one.
- **A/B inside ONE binary**, via an env switch — `SWARMLLM_DECODE_CALIBRATE=0`,
  `SWARMLLM_DECODE_ATTN=standard`, `SWARMLLM_FORCE_STANDARD_ATTN`,
  `SWARMLLM_DECODE_THREADS=0`, `SWARMLLM_VRAM_SWAP_MIN_IDLE_SECS`,
  `SWARMLLM_KV_RESERVE=0` (grow the KV cache into a prompt a quantum at a time,
  as before 2026-09-12, instead of reserving its admitted length),
  `SWARMLLM_KV_DEVICE_SYNC=0` (read the card's free memory without synchronizing
  first, as before 2026-09-26 — the second long prompt is then refused, #121),
  `SWARMLLM_KV_F16=0` (keep the KV cache on a card as f32 plus the f16 flash
  mirror, as before 2026-10-09, instead of the half cache — FUTURE_WORK #194; it
  also stops the node advertising `features::KV_HALF_ON_CARD`),
  `SWARMLLM_KV_WRITE=compose` (write into the half cache as a candle cast + copy,
  two launches, instead of the one-launch `kernels/kv_append.cu`).
  Comparing two builds compares two builds.
- **A switch reaches the WORKER only if the worker inherits it** — prove it from
  `/proc/<worker pid>/environ`, not from the command you typed (gotcha #616).
  And `SWARMLLM_<SECTION>_<KEY>` is not a generic config override: only seven
  settings read the environment (`Config::load_or_create`; the book's
  Configuration page lists them); switch anything else in the node's
  `config.toml` and read the worker's command line back (gotcha #722).
- **A one-shot benchmark cannot see a cost that only appears across turns.** The
  GPU swap floor was defended on the grounds that eviction discards a model's
  warm prefix cache — true, and it still lost 3.65x once measured in
  conversation, because the processor is slower at *every* turn than the reload
  it spares (#403). If the mechanism you are arguing about only bites on the
  second request, the benchmark has to make a second request.
- **Do not start a node and run a long request in ONE bash call.** The 2-minute
  harness timeout SIGTERMs the process group, which includes a node launched
  with `nohup … &` in the same invocation — it kills the daemon mid-load and the
  resulting "early eof / worker closed connection before reply" reads exactly
  like a crash. Use `setsid`, and keep requests in separate calls.
- **Verify the mechanism fired.** An outcome can improve for unrelated reasons;
  assert on the log line or counter the change emits.
- **A short run magnifies a one-time cost** into what looks like a standing
  loss. Vary the length it should amortise against before believing it.
- **A prompt-pass bench that GROWS its KV cache measures card allocations.**
  A worker reserves an admitted prompt's whole cache before the prompt pass;
  a bench that does not grows it a quantum per chunk, each step a fresh card
  allocation, and on this laptop at ~40 h of uptime that read 244-300 tok/s on a
  7B's 6000-token prompt in EVERY arm of an A/B, against ~1200 reserved — the
  "3x slower" first reading of the half KV cache (#194) was this, plus single
  slow repetitions of either arm at random. `prefill_bench` reserves by default
  now (`SWARM_BENCH_RESERVE=0` grows); read a disputed prompt pass with
  `SWARMLLM_PROFILE=1 SWARMLLM_PROFILE_SYNC=1` before believing a total.
- **A reply score that moves between arms is checked on LOGITS over one token
  sequence.** Two greedy replies diverge at a near-tie and are then scored on
  different text; a SentencePiece model's re-tokenization by llama.cpp can put a
  3.4-logit "gap" on a token neither arm's logits disagree about. Teacher-force
  both arms on the same ids (`logits_reference_probe`, `LOGITS_PROBE_DEVICE=cuda`
  for the card) and compare position by position — and give llama.cpp the
  node's context, or a LongRoPE model runs its short factors.
- **The benches have no tracing subscriber**, so `tracing::info!` from the code
  under test goes nowhere. If a decision needs to be observed, give it an
  explicit `eprintln!` behind an env var.
