# Future Work

What is still wrong or unbuilt, and what was decided against. **Every entry here was
verified against the code on 2026-10-02** (v0.3.221-alpha), when this file was rebuilt from
the 1.2 MB, 16k-line document it had become — about 345 entries, of which roughly 150 had
long since shipped without their entries being updated.

## How to use this file

- **Numbers are stable.** An item keeps its `#NNN` for life; a new item takes the next free
  number (**next free: #327**). Numbers below #165 come from the old triage index; #165 and
  up were given on 2026-10-02 to open items that had no number. Several old entries were
  merged into one — the entry says which numbers it absorbed, and § "Closed" lists every
  number that is no longer open, with where it went.
- **Read the entry's BODY before planning from it.** Its scope is a hypothesis (gotcha #654)
  and any line number in it drifts (#645). Trace a producer to its CONSUMER first.
- **History lives in [`FUTURE_WORK_ARCHIVE.md`](FUTURE_WORK_ARCHIVE.md)** — the old file word
  for word: measurements, rejected experiments and design reasoning. Each entry names where
  its history is ("archive row #N" — grep `^| N |`; or "archive § …"). The archive's
  statuses are frozen and often stale; **this file is authoritative**.
- **Closing an item**: move it to § "Closed" as one line (how and in which version), delete
  its entry, and append whatever history is worth keeping to the archive's § "Closed after
  2026-10-02". Re-rank § "▶ PRIORITIES" if it was there.
- **Deferring a sweep finding**: add an entry (or a line under § "Decided") with enough
  context that the closure is not a black hole, so later sweeps do not re-report it.
- Priority is **user-visible impact × how many users × whether it fails SILENTLY**, with new
  users first (their first hour decides whether they stay).

Each entry: a heading with its number and the problem as it stands today, a line
`priority · area · status — since · history`, then what remains and the next concrete step.
Status is **OPEN** (nothing material done) or **PARTIAL** (part shipped; the entry is the
residual).

## ▶ PRIORITIES — read this first

Re-ranked 2026-10-17 after v0.3.241 (released 2026-10-17), which closed #265, #286-#290, #292,
#296, #298 and #115's Gemma 4 26B-A4B, E2B and E4B, and shipped #91 (2)'s manifest digests; the
items it filed (#291, #293-#295, #297, #299) are all P3, so no rank moved. What each release carried
is in § "Closed" and `CHANGELOG.md`; the running log of re-rankings from 2026-10-02 to this one
moved word for word to the archive, § "PRIORITIES re-rank log". Only OPEN work is listed here.
`docs/plans/` holds the multi-step designs; the entries point at them. **2026-10-20:** #318 goes
first among P2 by the user's direction: running models too large for any one member at a usable
speed is what the swarm is for, so split speed for large models is the top track
(`docs/plans/large_model_splits.md`).

**P0 — wrong answers, silently**
1. *(none open since #156 closed on 2026-10-03: a mixed copy is refused at load, a part that is
   not the upload's bytes is deleted and re-fetched, and every part entering from a peer is
   byte-checked — `docs/invariants/network.md` § "One upload per model id".)*
2. **#217** — WATCH: a node kept parts the checked holders disagreed with until it restarted
   (v0.3.224); the mechanism that kept wrong HuggingFace replacements closed in v0.3.225 and it
   has not recurred since 2026-10-04. Nothing to build until it does.

**P1 — silent, or broken for a whole class of users**
3. **#1** — every Mac runs on the processor; no GPU backend is compiled for Apple Silicon (needs a
   Mac to verify on).
4. **#220** — WATCH: a worker's card faulted once (illegal memory access, 2026-10-04) and the driver
   took 14 minutes to reset; the kernel is still unknown. Every planned run is done (memcheck clean
   over ~27 min on two boots, FECS shown with no SwarmLLM code at 48 h, the drafter-in-one-process
   change weighed and declined); nothing to run until the illegal access recurs.
5. **#250** — a model held whole here, which the card holds only part of, can be routed through
   far peers instead of answering locally. Its causes closed in v0.3.236 and v0.3.239; left: the
   prompt priced on the processor prior for a card/processor split, and a far hop priced at
   `net:0` (with #3). Passive: the next evidence is a field `predicted_terms` line.

**P2 — speed and completeness**, ranked by how many users meet it in their first hour
6. **#318** — a large model split across machines is too slow to use: 0.35 tok/s on a 4-segment
   intercontinental chain, 5-13 tok/s on a 7B Thailand↔Belgium with the stream. Projected
   lossless 20-31 tok/s across continents and 50-80 inside a region on cards. The stream
   through a chain of machines, guessing for a user who holds none of the model and trees in
   every segment are built (2026-10-20); the head's bounded wait and a route carried to the head
   (10-21); trees the default where every segment runs on a card (5.33-5.74 vs the stream's
   5.01-5.05 on a real ~405 ms link, #134); a card multiplies a check's rows with llama.cpp's
   tensor-core MMQ (#325, 10-21: 2.4-3.2× faster at 9-64 rows). The regrow after a missed seed
   is sized by what it buys, and trees are on only past 300 ms (#134 (a), 10-21: +57% at 20 ms,
   +4% at 300 on the rig). Next: near peers over far ones (#192); a drafter fed by the model's
   own state (`large_model_splits.md` build-order step 4); a tree that flows (#134 (b), built
   10-22: +11-14% over tree rounds on the card rig; its drafter's tree levels are one kernel and
   one CUDA graph since 10-22; on a thin uplink a guard holds its rows down — 5 Mbit 7.44-7.50 vs
   4.73-4.80 tok/s unguarded; copied runs from the conversation give it free depth — a code edit
   23-24 → 42-47 tok/s; next the real link).
7. **#115** — the rest of Gemma 4: its image reader and its drafters, refused by name today,
   never loaded wrong. (Gemma 3 reads pictures as llama.cpp does since 2026-10-19; Gemma 4's
   E2B/E4B run since 2026-10-17 and fit a 4 GB / 6 GB card whole since #296.)
8. **#91** — an idle node's upload (the complaint that uninstalls a P2P app): 5.5 KB/s on the .241
   fleet (6.7 on .240, 8.8 on .239), measured 2026-10-17. Manifests now arrive about once an hour
   (0.83 a minute before (2)), and a restart no longer re-gossips them ((2c), v0.3.242). The part
   that is not gossip is 2.5 KB/s on 10-18 (4.1 out), and our own direct messages are only 0.19 of it
   ((2d), `direct_by_kind`): the rest is libp2p's protocols. `by_protocol` ((2e)) named the largest:
   Kademlia answering other nodes' lookups, ~2.1 KB/s; (2f) republishes provider records every 4 h
   instead of 20 min (v0.3.243). Field re-read 10-18: upload 2.7 KB/s, but the Kademlia answers
   fell to 0.90 KB/s, 2.3x, not the 12x the republication alone predicted. `kad_inbound` ((2g),
   v0.3.244) read on the field 10-19: upload 2.4 KB/s, Kademlia answers 0.51 KB/s, every one a
   `find_node` — peers' routing-table refreshes. (2h) refreshes every 10 min instead of 5 (released in
   v0.3.245); next, re-read once the fleet runs it, then the answers' size.
9. **#3**, **#180** — the routing cost model mis-prices a split by regime, which keeps partial
    ranges off. Fit on three peers 2026-10-18: a short reply's `max_tokens` now caps the passes
    (16-token replies 0.49 → 0.97 of their price), a long one is priced at its model's mean
    reply length (160-token replies were 2.5x their price), and a chain its head leads is priced
    by its own hops (network term 57.9 s → 3.6 s, same peers and passes), all in v0.3.243; two
    peers whose workers already chain are planned that chain (#3 (e)) and a led chain's peers are
    priced by their own published decode time (#3 (f)), and a peer handed a prompt is priced for
    its cached opening on the next turn (#3 (c)), all fixed 10-18, and a split is priced for the
    rounds it takes, a guessed token not crossing alone (#3 (d), 0.49 → 0.77 of its price on the
    drafter rig), 10-19, all in v0.3.244; next, a chain driven from here (#3 (a2)). Never tune the
    constant.
10. **#10**, **#171** — conversation prefixes across computers (no digests gossiped, caches end at
    the 10-minute expiry), and a split's prompt pieces chained between two peers.
11. **#138**, **#129** — a model that does not fit the card (MoE and a split's prompt stages
    shipped in v0.3.235-v0.3.237; the entries hold what is left), and a near-fit model sent far.
12. **#228** — Qwen 3.5 without speculation (its state cannot be wound back yet).
13. **#30** — the plaintext-prompt trust bar sits at the score an unknown peer starts on: a policy
    decision, with research, when picked up.
14. **#326** — on a card newer than a release's machine code (Hopper, Blackwell: RTX 50-series),
    the first reply waits ~2 min while the driver compiles flash attention's PTX (MMQ's ~8 s a
    weight type), then it is cached. Correct since the PTX went in (before: wrong output).

Everything else is ranked in its own entry. **P3** is narrow or cosmetic, **P4** is process,
maintenance or an idea with no user waiting on it.

## Open items

### Wrong answers and the one-upload-per-model rule

#### #217 — A node can keep parts the checked holders disagree with until it restarts
`P0` · heal · **OPEN — WATCH: nothing to build until it recurs** — 2026-10-04 (the v0.3.224 swarm reading; re-read 2026-10-15) · history: archive § "Closed after 2026-10-02" → "P0-2"

`e561df35` (a tester's Proxmox node, LXC, 30 GB disk — #142) ran v0.3.224 from ~02:39 to
05:39 UTC holding Llama-3.1-8B parts 0-1 and GLM-4 part 5 whose bytes are not the upload's.
Hashed straight from HuggingFace on this node: its own copies are the upload (`0243a766…`,
`a89e0f92…`, `16f2f130…`, shared by `4a3ac72e`); `e561df35`'s tags are exactly an old
manifest's hashes (`096077dd…` → tag `1374a189…`, `7344e585…`, `ba7d90ae…`) — bytes that pass
the 64 KB check. Two connected checked holders disagreed throughout; it announced the parts
unchanged every 5 min and never withheld them, so its heal never judged them wrong. No node
routed those parts to it; its own runs of the two models used them. It restarted at ~05:39:
within 3 min it announced part 0 as a THIRD build (`78d07c74…`, tag `7ab9f261…`), at 05:44
withdrew parts 0-1, at 05:45 announced both as the upload; GLM-4 part 5 was dropped.

A restart clearing it points at state that lives for one run of `auto_manage::canonical`
(undetermined which — that node's log was not visible here):
- `CheckedParts::from_origin` — a dispute settled "from the upload" is skipped by `outvoted`
  for the rest of the run, whatever bytes actually came back;
- `CheckedParts::by_model` — a part recorded as checked and undisputed is not asked again
  (though `outvoted` is evaluated regardless);
- a stale `active_traces` / `serving_models` entry (`model_is_in_use`) or a recent
  `Downloading` entry (`model_download_under_way`) — the first would have withheld the copy
  (`Holding::Replacing`), which was not seen.
Not the vote: `checked_holder_tags` reads the raw holder map, which keeps other builds. The
interim third build after the restart says a replacement can also ARRIVE wrong.

**Rigged 2026-10-04 — the shape alone heals.** `outvoted_rig.sh` with `HFLESS=none DAMAGE=tail`
(B reaches HuggingFace; its last part keeps the first 1 MB and is zeroed after, so it passes
the 64 KB check; its manifest vouches for those bytes — `e561df35`'s shape, since
`CanonicalBuild::describes` compares layout, not part hashes): v0.3.224 PASSES in 167 s — first
pass "canonical", next pass `in_dispute=[1]`, deleted, fetched from HuggingFace, identical to
the upload. So what held `e561df35` was state of that run, not the shape. Its flat part count
(60 from 03:06 to 05:40) is NOT evidence of a frozen task — it has a 30 GB disk (#142), and a
full disk with no heal looks exactly like that.

**Made observable (v0.3.225):** the repair records why each pass left each model
as it was (`SharedState::models.heal_verdicts`, written only by `canonical::note_verdict`) and
when its last pass started and finished (`heal_pass_times`); `swarmllm diagnostics` prints them
under `-- copy repair --` — including which parts the checked holders dispute but that wait
for a re-check (`shards_pending_verification`) or were already fetched from the upload this run
(`CheckedParts::from_origin`), and "a pass has been RUNNING since …" for a stuck task.
Verified on the rig: "no pass yet" → "the canonical upload" → "replacing parts: … in dispute
[1]".

**Recurred 2026-10-04 09:01 UTC — and a replacement ARRIVED wrong, twice.** `e561df35`
restarts daily at ~09:00 UTC (its announces show it since 09-18). Six seconds after this
restart it withdrew Gemma-2-2B part 0, Qwen2.5-14B part 1 and Qwen2.5-Coder-7B part 3 — parts
it had announced as the upload for hours, so the startup BLAKE3 check found their files no
longer matched their origin-backed hashes. Within ~40 s each came back as a wrong build
(`31ac3faf…`, `f6c55cc8…`, `953ba1d6…`). The heal's second pass (~09:04) found them in dispute,
deleted them and fetched them from HuggingFace again; they came back as three OTHER wrong builds
(`aa9622c0…`, `67f41f50…`, `473b1184…`) and stayed for 4.5 h. That is the
`CheckedParts::from_origin` door named above: a part re-fetched from the upload was exempt for
the run whatever bytes had come back. And the HuggingFace download path believed whatever it
wrote — hashed after the fact, recorded as the origin's own bytes — so two different wrong
results each became "the upload's". Two downloads of one byte range giving two hashes means the
bytes changed in transit or on that disk; which one is undetermined (huggingface/huggingface_hub#3643
reports large downloads arriving the right size with a different hash on every attempt; a disk
losing writes — a 30 GB LXC, #142 — looks the same, and would also explain the startup failures).

**Fixed in v0.3.225 (the mechanism, not that node's cause):**
`huggingface::download_shard` hashes the bytes as they arrive, syncs the file and checks it reads
back as them; the P2P accept syncs before hashing (a failed write-back is reported only there —
PostgreSQL's fsyncgate). `SharedState::accept_origin_part`, the one step both HuggingFace paths
pass, keeps a download only when something corroborates it — a connected holder that checked its
copy holds those bytes, or the previous download of the part brought the same ones — and
otherwise deletes it and fetches again, telling the owner when two downloads disagree. The heal's
exemption now covers only a part the upload's bytes actually settled.

Its wrong parts also cost other nodes downloads: a node announcing a copy as checked counts
as a checked holder, and one checked holder disagreeing is enough for a node that can ask the
upload to re-fetch its own part. Seen here 13:55 UTC after a restart (`e561df35` the only
checked holder connected yet): our correct Qwen-Coder part 3 was deleted and fetched again
(10 s, back as `a48c1ca2…`). By design ("a wrong vote costs one download"); with the fix a
node no longer vouches for bytes nothing corroborated.

Known cost, accepted for now (pre-release review, 2026-10-04): the record that two downloads
agreed (`models.uncorroborated_origin_parts`) lives for one run, while the part's origin hash
persists. After a restart, a part kept that way loses its exemption, so while checked holders
still disagree the heal deletes it and downloads it twice more (one discarded, one kept) per
restart — bounded, never a loop within a run. Persisting the record beside `origin_verified`
would end it.

Next: `e561df35`'s first 09:00 UTC restart on v0.3.225 (from 2026-10-05) is the field check — its parts should
come back as the upload's, or not at all, with "came out different on two downloads" in its log.
The 02:39-05:39 occurrence (parts announced as an OLD manifest's exact hashes for 3 h, never
re-fetched) is not explained by this and stays open; its journal (asked of its owner once) or a
recurrence's `-- copy repair --` decides it.

**Re-read 2026-10-15 on the v0.3.239 fleet: no recurrence since 10-04.** `e561df35` is not among the
seven connected peers (its id appears only as a manifest's first publisher), and every model reads
`peers_other_build` 0 from this project's node at 2026-10-14 08:57 UTC and 2026-10-15 11:15 UTC,
since #270 (v0.3.238) brings every node onto the one upload. Nothing is built for the 02:39-05:39 case
until a recurrence's `-- copy repair --` names its state. The one buildable piece is the known cost
above: persisting `uncorroborated_origin_parts` beside `origin_verified`.

#### #246 — A node near its storage limit fetched back a part prune had just deleted, once in six runs
`P3` · heal · **OPEN** — 2026-10-09 · history: the v0.3.232 gate, step 12t

`gate232_resume.sh` 12t (`prune_loop_inner.sh`: A at 87% of its storage budget, B and C holding
every part, a request a minute): the .232 artifact deleted the middle part, fetched it back
(14:55:15 and 14:56:12 UTC, 57 s apart) and deleted it again — 3 deletions, 2 fetches in 600 s —
while .231 in the same run did 1/0. Four alternated re-runs (`r12t_ab.sh`: .231, .232, .231, .232)
were all 1/0, and every gate since .228 was 1/0 on both binaries: not the release, an input that
flips. Between the refusals (12, each `pressure_after=0.87`, holders 2) the download pass let the
part through twice, so one rule of `would_shed_copy` read differently for that pass; prune shed
it 18-28 s later on the same pressure. Not pressure (both readings logged alike), not a user lock
(`locked_shards` is user-set) and not the carry check (a processor node advertises its configured
RAM cap, not live headroom). Candidate: `would_eliminate_region`'s fallback when no region is
known — "fewer than two other holders answered a ping under 200 ms" — reads `PeerInfo::latency_ms`,
a single health-ping COST sample measured through the dispatcher, not a distance
(`arch-network.md`: a reader meaning distance goes through `physical_rtt_ms`). One slow pong keeps
the copy for one pass. **The log now names the rule** (`copy_kept_because`, "DIAG: a part prune
would keep once it landed — fetching it", and the fallback's pings): read it on the next
occurrence before changing the rule.

#### #313 — A greedy reply on a CARD is not the same with the prompt cache on as off
`P3` · correctness · **OPEN** — 2026-10-19 · history: #179 (the processor half, closed)

#179 found why a `temperature: 0` reply changed after its first run: a prompt-cache hit runs the
last prompt token as a decode step, and the decode kernel summed a row differently from the
prompt kernel. On the processor both now share one arithmetic (`inference::attn_rows`). On a
card they still do not: a decode step is `decode_attn`'s CUDA kernel (`decode_attn::chunk_for`,
a chunk that follows the cache length), a prompt pass is `layers::matmul_attention` (cuBLAS
products and a separate softmax), so a hit can still change a card's reply. The fix has
SGLang's shape — the same split-KV of absolute position in BOTH card paths, which means a card
prompt kernel that reads a row's keys in the decode kernel's chunks. #309's chunk (16/32/64) can
stay if it remains a function of the ROW's visible length, the same function in both paths; a
row's cache length at decode is its visible length at prompt time. Measure as #179 was, on a CUDA build:
one isolated node, the same request four times with the cache on, restarted between two rounds,
the replies' md5s compared; and `examples/logits_reference_probe.rs` with
`LOGITS_PROBE_DECODE` 2 vs 3. Until then: on a card, generated text cannot compare two code
paths across a cache hit.

### Reliability and failover

#### #167 — A speculative split decode has no mid-reply failover
`P2` · reliability · **PARTIAL** — 2026-08-25 · history: archive § "Speculative distributed decode has no failover", archive row #149 item (d)

Guessing ahead is on by default for splits, and every speculative loop fails as a unit:
`pipeline/ngram_only_spec.rs` calls `forward_verify_through_segments(...).await?` bare, and a
failure mid-reply in the continuous stream (`dsd_stream`) ends the attempt. What shipped:
`router::peer_went_silent` re-plans a `PeerUnresponsive` once onto another holder, so the
user usually gets a 200 — after the whole reply restarts (measured 2026-10-01: B unloading
mid-reply → 200 via one retry). A takeover is offered only on the prompt pass. Two shapes for
the fix: fail the segment over through `failover_segment` (the verify path must be
re-entrant; carry a `truncate_kv_to` for positions in doubt and test a peer that fails AFTER
applying them as well as before), or stop speculating for the rest of the request and
continue on the ordinary decode loop (the chained-run rewind shape — cheaper). Harness:
`split_rig.sh failover` / `kill` with `SHARDS_A=0,1,2,3 SHARDS_B=4,5,6,7` (the gate's default
plan never streams).

**What the caller sees now, by kind of request (2026-10-10).** A STREAMED reply that loses its
machine is continued from what its reader received (#236). A NON-streamed one is not: nothing
marks its first token (`ttft_ms` is stamped only by the streaming sender), so the router
re-plans it and generates it again from the first token, and only if that fails too does the
caller get the partial, as `finish_reason: "interrupted"` (`salvaged_reply_if_lost`). Measured
with `split_rig.sh continue NONSTREAM=1` (Llama-3.2-3B handed whole to one of two holders, its
worker killed 8 s in): 200 in 47.9 s, 300 tokens, the reply generated twice over. Continuing it
instead needs the RAW text of every attempt — the salvaged partial and a successful attempt's
content are both finalised (leading whitespace trimmed, scratchpad stripped), and joining two
finalised halves glues the words at the seam — so the trace would have to keep generated text
for non-streamed requests as it keeps streamed text now. In the live node's log (09-17 →
10-07, mostly test traffic) this path ran ~10 times in 898 requests.

#### #17 — A long generation whose failed segment no single standby covers cannot fail over mid-reply
`P3` · reliability · **PARTIAL** — 2026-09-08 · history: archive row #17 (its body § "A long generation with no segment redundancy is lost entirely" is frozen at 09-08 — the row overrides it)

Takeover works (`Takeover` in `pipeline/distributed.rs`, `scheduler::standby_cover_for`),
including a composite cover assembled from several nodes on the prompt pass (live runs
2026-09-25, `split_rig.sh failover` / `failover_mid` PASS). Mid-reply a composite is refused:
a stand-in needs the retained input history, which exists only for the range the coordinator
sent to, and the second part of a composite is fed by the first part's output, which nobody
retains. Lifting it needs a new retention scheme. A live mid-reply takeover has also never
been run.

#### #18 — A failover after the prompt pass cannot restore a chained or tensor-parallel segment
`P3` · reliability · **PARTIAL** — 2026-09-09 · history: archive § "A failover after the prompt pass silently loses the failed segment's KV context", row #18

The silent drift is FIXED (shapes 1 and 2, 2026-09-09: P 0.9965 vs 0.9966 after restore), and
`failover_can_restore_state` is pinned by a guard. Residual: a chained run's middle inputs
never pass through the coordinator, and a tensor-parallel segment has no single history, so
`retained_activations::mark_unrestorable` marks both and a failure there ends the reply
visibly instead of moving it. Closing it needs per-hop retention on the peers, or a
chained-run rewind that re-sends from the coordinator. (A middle worker's death with
chaining on is already recovered by re-running that segment unchained on the same node.)

#### #161 — Outlier ejection does not count a peer that loses the conversation mid-reply
`P3` · reliability · **OPEN** — 2026-10-02 · history: archive row #161

The bf7b3263 field report's 4th failure ("this computer no longer holds the conversation",
`model_worker.rs`) arrives as a peer REFUSAL, which `segment_delivery_verdict` correctly
scores as an intact delivery; `daemon::state::peer_outliers` counts timeouts, abandoned
forwards and silent whole-model replies only. Needs a typed refusal
(`ForwardRefusal::ConversationLost` beside `Undecryptable`, gated at the sender on a feature
bit) so the coordinator counts it without matching prose (#295's trap).

#### #146 — A stalling graphics card is guarded; the cause, the high-uptime reading and segment forwards are not
`P3` · reliability · **PARTIAL** — 2026-09-28 · history: archive row #146 and § "A graphics card that stalls is handed more work (#146, 2026-09-28)"

Shipped in v0.3.213 (the archive row still says "not released"): `inference::card_pace` halves
how many generations a worker runs once its card stalls ≥ 2 s and refuses the rest as the
busy 503 the router re-plans; `inference::cuda_pool` keeps freed card memory while serving
(fresh allocations become ~1000× slower after two days of Windows uptime — microsoft/WSL#41701's
shape, gotcha #754/#755). Open: (1) the pool's effect at ~2 days of uptime
(`~/swarmllm-pool-0928/pool_ab.sh`; only tied at 5.5 h so far — a card measurement is a
stress test: safety kit, live node stopped); (2) split-pipeline segment forwards
(`Forward`/`BatchForward`) are neither observed nor refused, and sequential `handle_generate`
per-token forwards are unobserved; (3) the daemon does not advertise a tripped card to peers.
The stalls' root cause stays unknown and the guard does not need it. A card FAULT (not a stall)
and the 14-minute driver reset after it are #220.

#### #220 — A worker's card faulted once (illegal memory access), and the driver then took 14 minutes to reset
`P1` · reliability · **OPEN — WATCH: nothing to run until it recurs** — 2026-10-05 (WATCH from 2026-10-17) · history: gotcha #790

On 2026-10-04 at 18:52:06 UTC, in the .225 gate's baselines (`~/swarmllm-gate-0225/repro_b5_old2_w6`,
the v0.3.224 binary: Qwen2.5-Coder-7B layers 0-14 on the card, an fp16 Qwen2.5-0.5B drafter in a
second process on the same card, split speculation to a processor node), node A's 7B worker failed
with `CUDA_ERROR_ILLEGAL_ADDRESS`, surfaced at the copy of the step's output to the host ("Encode
Q8_0" — CUDA reports a kernel's fault at the next synchronising call, so the fault was in that
step's forward, not in the encoder). One second later Windows logged nvlddmkm 14 and "UCodeReset
TDR"; the reset finished only at 19:06:16 ("Reset" / "Restarting TDR"), and A's re-plan sat in its
next card call for those 14 minutes. Three "Graphics FECS Exception" events (18:34:19, 18:42:21,
18:47:05) each fell in the same second as a speculation request ending and the next starting on
that worker pair, in three other arms — .224 and .225 alike, so not a regression. The gate record
mentioned none of it: the safety kit watched temperatures only (it now stops at the first driver
event, `~/swarmllm-gate-common/safety.sh`).

Known: one illegal access in every log on this machine; FECS events also arrive with no work of
ours (13:09:31 the same day, 16 s after a laptop power-source change; 13:25:11 idle). Ruled out:
graph updates ignoring a grown allocation (`docs/invariants/inference.md` § "A graph pays only when
it is updated"). Unknown: which kernel. *(The steps below were DONE 2026-10-10 → 10-13 — see
"Step (1)" and "(c) weighed" further down; the current state is at the end of this entry.)* Next, in order: (1) run that rig shape under
`compute-sanitizer --tool memcheck --target-processes all` — on WSL2 it first needs the WDDM
debugger interface: `HKLM\SYSTEM\CurrentControlSet\Services\nvlddmkm` `EnableDebugInterface` = 1,
set as administrator (the forum's second value, `GPUDebugger` `EnableInterface`, proved
unnecessary here — run 2026-10-10, below);
(2) the re-plan sat in GPU admission because `nvidia-smi` waits as long as the driver does — FIXED
2026-10-05: `vram::nvidia_smi` bounds it at 10 s and never runs two at once (while a stuck one
lives the next reading is unknown at once; `docs/invariants/memory.md` § "nvidia-smi is asked
through one bounded helper"). And FIXED 2026-10-07, released in v0.3.230: while that
stuck `nvidia-smi` lives and has for a minute (`vram::graphics_driver_not_answering`, asked
without waiting on the helper's lock; `DRIVER_NOT_ANSWERING_AFTER` — the release gate caught
the first draft acting on one 10-15 s stall and sending a card-only model to a processor it did
not fit, gotcha #802), a new worker is placed on the processor with `CpuReason::DriverNotAnswering`
(`driver_not_answering`; amber on the dashboard, `placement.driver_not_answering` in all 21
locales, the CLI's words) instead of blocking in its context creation on a resetting driver;
the reason clears when the driver answers, and promotion (`reason_still_holds`) moves the model
back — `GpuUnavailable`'s advice to restart is not given. Checked on an isolated CUDA node:
with `SWARMLLM_FAULT_DRIVER_NOT_ANSWERING=1` (a stuck `nvidia-smi` cannot be faked — the real one
survives SIGKILL, a stand-in does not) the worker started on the processor with that reason on
`/v1/status`, `/api/admin/models` and `swarmllm status`, answering in 2.0 s; without it, the card; (3) whether two CUDA
processes switching on one card (target + drafter) is what the FECS events have in common — the
08:25:56 one on 10-04 also came 3 s after a third worker started.

**After #221 (2026-10-05, `~/swarmllm-gpu-1005/dsd221.sh`, the same shape on `703b511d`, safety kit
with the driver stop):** 4 arms, 12/12 replies, 0 card faults, 0 nvlddmkm events (confirmed by a
direct `Get-WinEvent` query; the telemetry ran throughout, peak 73 °C). The 7B's checks now update
their graphs in place — 179 of 182 launches, 3.8-4.0 ms recording each, where 10-04's arms rebuilt
71-197 launches at 43-108 ms. Requests 46.9-51.4 s (first) and 31.4-32.7 s, against 50-124 s and
32.6-64.5 s on 10-04. Evidence, not proof: with faults in half of 10-04's arms, four clean arms
would come by chance ~6% of the time, and 10-04 ran with ~3.5 GB free RAM against ~12.8 GB here.
The constant graph rebuilding — driver work on every launch — is now the leading suspect for the
FECS events; the illegal access stays unexplained.

**Step (1), 2026-10-10 (released .232, Windows up 61.6 h):** compute-sanitizer 2025.3.1 ATTACHES on
this WSL2 box with only `nvlddmkm\EnableDebugInterface` = 1 — the `GPUDebugger` key does not exist
here and is not needed. Checked against a known answer: a planted one-past-the-end write was
reported at its line and thread (`~/swarmllm-220/attach.sh`). The 10-04 shape with node A and
every worker it starts under `memcheck --target-processes all` (`~/swarmllm-220/rig.sh`, B plain):
~15 min of one request's check steps and drafter work, graphs updated in place, reported NO invalid
access. Under the tool a graph records in 3.6-8 s a launch (~4 ms plain) and `nvidia-smi` stalls
29-30 s, so `driver_not_answering` would send a NEW worker to the processor — start a sanitizer
run's workers before the stalls begin. Then eight FECS (nvlddmkm 13), no TDR: five at
02:44:37-39Z, 2-4 s after the next request's prompt pass began on the card while the first was
still running there — its client had hung up at 900 s and the reply had not stopped (a defect of
its own, fixed the same day: `api::submit_to_router` now cancels it) — and three at 02:45:00-01Z
as the kit killed the workers. So a request boundary on that worker pair again, as on 10-04, and
memcheck saw no bad access across it. One arm: evidence, not proof. Next, on a fresh boot and
under the kit: (a) two plain CUDA processes launching long kernels at the same time vs one after
the other, with nothing of ours — whether FECS follow two processes' work meeting on this driver;
(b) the shape under memcheck again with several request boundaries (`REPEAT` ≥ 3, `max_tokens`
small enough to finish inside the client's 900 s), to give the illegal access more chances to show.

**Step (a), 2026-10-10 06:12-06:17Z (fresh boot, Windows up 0.1 h):** `~/swarmllm-220/twoproc.sh` —
two plain CUDA processes (`twoproc.cu`: busy kernels, a small copy back, a 256 MB allocation freed
every 20 rounds; 200 rounds of 50 ms each), three arms one after the other and three at once,
alternating, under the kit. **0 driver events in all six arms** (kit: 9 polls, 0 events, 0 failed
queries; the System log for the window holds no nvlddmkm event). That does NOT clear two processes:
every FECS so far came at 49-80 h of uptime (10-03 52 h, 10-04 ~80 h, 10-09 49 h, 10-10 61.7 h), and
card behaviour drifts with uptime (#755). The discriminating run is the same rig, with arms ~10x
longer, at ~48-60 h — FECS in "at once" and none in "one after the other" there would put the cause in
the driver with two contexts, with nothing of ours. Not past ~100 h (#804: TDRs came at 150 h+).

**Step (a) at 48 h, 2026-10-12 06:09-06:21Z (Windows up 48 h, booted 06:08:59Z 10-10):** the same rig,
arms 10x longer (2000 rounds, ~100 s), live node stopped, under the kit. serial 06:09:54 — 0;
overlap 06:12:53 — 0; serial 06:14:35 — 0; **overlap 06:17:47 — 3 FECS** (nvlddmkm 13 "Graphics FECS
Exception: Logging error 0x2", all at 06:18:52Z, ~65 s in, both processes running); no TDR (no
Display 4101, no nvlddmkm 14). The kit stopped the rig, but one more serial arm ran (06:20:26, 0): it
checked its abort marker only BEFORE settling, and the 30 s Windows poll wrote the event during the
settle — `safety_settle` now checks again after; the per-arm count, which printed 0 for the arm with
the events, now waits one poll. So: ~6 min of one-after-the-other card work, 0 events; ~3 min of
two-at-once, one burst of 3 — with no SwarmLLM code on the card. FECS is the unit that runs the
graphics engine's context switches (nouveau's `gf100` "fecs methods for pausing ctxsw"; NVIDIA
publishes nothing on its multi-process switching), so two contexts meeting on this driver at ~48 h
is enough to log one. One burst is evidence, not proof, and it does not explain 10-04's illegal
access or clear our code — it says a FECS by itself is not evidence of a fault of ours. Next: (b)
above; and whether keeping a card's work in ONE process on Windows/WSL (the drafter in the target's
worker, never a second CUDA process on the card) is worth its cost — #754/#755's drafter shape is
the one product path known to put two processes on one card.

**Step (b), 2026-10-13 02:44-02:58Z (fresh boot, Windows up 2.5 h, released .236):** the 10-04 shape under memcheck
again (`~/swarmllm-220/rig.sh 1 6 32`, rebuilt after the 10-12 disk cleanup deleted the 10-10 copy): six requests
of 32 tokens (`split_rig.sh`'s new `REPEAT_MAX_TOKENS`, so six request boundaries fit where 120 tokens gave one), all
answered, 78-136 s each; **no invalid access reported, 0 card faults, 0 driver events in 29 kit polls.** The tool
was proven IN both card workers this time, not inferred from their slowness: their `/proc/<pid>/maps` held
`libsanitizer-collection.so` and four more of its libraries, and node B's processor worker (the control) none. Two
rig facts learned on the way: `--log-file …/%p.log` names the sanitizer FRONT-END's pid, so every process of one
session writes one file (the second file is `split_rig.sh`'s `--version` call); and no log ends in an `ERROR
SUMMARY`, even with the node stopped first and 120 s given (`san-a.sh` now forwards the stop to A's daemon alone) —
a worker is killed with its daemon, so only an error line, written as it is found, can appear. With 10-10's ~15 min
at 61 h, ~27 min of this shape under memcheck have reported nothing; the 10-04 illegal access stays unreproduced and
its kernel unknown.

**(c) weighed 2026-10-13 — not doing it on this evidence.** Moving the drafter into the target's worker was the
proposed way to keep a card's work in one process. Two findings stop it. First, the drafter is not the only such
path: every model gets its own worker (`ModelProcessPool.workers` is keyed by model id), so ANY two models on the
card are two CUDA processes — the shape Ollama runs everywhere with a runner per model. Second, a worker is built
for one model: its `models` map is keyed by layer range alone (`ensure_model_loaded` and `subsumed_segment_keys`
never compare the model id, so a drafter `(0,24)` beside a target `(0,24)` would be served the wrong weights, and
a target `(0,28)` would evict it), `kv_model_key` and the prefix cache are range-scoped, every charge, idle unload
and reclaim is per worker, and the drafter's prompt read-ahead would serialise behind the target's prompt pass in
one worker loop (`dsd.rs` `read_ahead`). That is a redesign of the worker for one case of many, with nothing yet
tying the shutdowns to two contexts (step (a) logged FECS, never a TDR). Revisit if a fault or reset is caught with
two of our processes on the card and not with one.

Field reading, report #036 (2026-10-14, an RTX 3060 node on Linux, 0.3.229 → 0.3.238, its whole log since
10-08): 0 `illegal memory access` / CUDA error lines and 0 kernel `NVRM: Xid` events in six days.

**WATCH from 2026-10-17.** Every step above is done and none named a next one, so the entry waits on a
recurrence of the illegal access (or a fault caught with two of our processes on the card and not with one —
(c)'s condition). One more FECS, 2026-10-17 05:39:31Z (Windows up ~101.5 h, past the ~100 h line, #804), in the
.241 gate's `moe/prev` arm moments after the 30B MoE loaded (card 7.0 GB): no TDR, and every reply of the arm
finished, but `nvidia-smi` stayed blocked ~3 min and the whole PC froze for ~78 s (both
sides' telemetry stop; the user saw it), then recovered; the 06:04Z reboot was a clean Windows Update restart.
The arm's node log was deleted with its temp dir, so whether a drafter shared the card is unknown — the gate
keeps each arm's log since. The safety kit's own loop was blocked by that `nvidia-smi` and stopped the nodes
2 min 48 s after Windows logged the event; fixed (gotcha #887).

#### #193 — A whole-model reply travels as independent messages; only the serving node dying now loses it
`P3` · reliability · **PARTIAL** — 2026-09-02 · absorbs archive § "The reply stream has no reliability layer…" and § "Intermittent token loss on the remote-generate fast path"; history: archive § "Replies truncated on the remote-generate fast path"

Shipped: the resend ladder (`ResendTokens`, `features::RESEND_TOKENS`,
`daemon/state/retained_replies.rs`), a truncated reply is a 503 "Reply truncated in transit"
rather than `finish_reason: stop`, and a per-peer delivery ratio feeds routing
(`examples/dropped_token_test.sh`: 40/40 with a dropped token, 5/40 before). Not built: one
ordered stream per reply (the R139 `pipeline_stream` machinery under a new protocol id and
feature bit, request-response kept for old peers) — a latency and robustness gain, not a
correctness fix now. Two observations never isolated: tail losses on 2026-09-17 after a
serving node swapped its resident model (needs a pinned-server two-node lab), and no
network-level test drops a `StreamingToken` (only the shell harness does).

#### #181 — A serving node reading a long prompt sends nothing until it finishes
`P3` · reliability · **OPEN** — 2026-07-27 · history: archive § "CPU prefill throughput is the dominant cost for modest nodes"

A processor-only holder can read a long prompt for minutes in silence, so the coordinator
cannot tell slow reading from a dead peer and its deadlines must stay long. (The kernel half
of that entry is closed — CPU prompt reading is level with llama.cpp, #119.) Build an
additive prefill-progress message from the serving node (feature-bit gated, per the protocol
rule) so the coordinator holds on evidence and fails fast on silence. Skipping slow holders
for long prompts when a faster one exists waits on #180.

#### #142 — Three residuals of the 2026-09-28 storage field report (node `e561df35`, LXC, 30 GB disk)
`P3` · storage · **OPEN** — 2026-09-28 · history: archive row #142

None reproduced here; the disk-full deadlock itself is fixed. (a) An update download that hits
`ENOSPC` is not retried after freeing space — reserve the update binary (~57 MB) in the
budget's free-disk clamp, or reclaim and retry once on os error 28. (b) "No candidate shards
to download" with 5,800 MB free — possibly explained now by the one-upload rule (#151); ask
for `swarmllm diagnostics` if it recurs. (c) The sign-in page after an update should tell
"your session ended" from "wrong key".

#### #132 — A peer reporting its own task cancelled reaches the caller as a 500 and costs the peer trust
`P4` · reliability · **OPEN** — seen once 2026-09-27 · history: archive row #132

"task join: … was cancelled" from a peer's own restart or cleanup is not classified as
retryable. If a second sighting shows it transient, add a rule in `reclassify_flattened_error`
(through `classify_error`, never at a call site) mapping it to a class the router re-plans,
without the penalty. Needs the peer's log first.

#### #133 — On the persistent pipeline stream, a healthy peer never finished reading a prompt pass
`P4` · reliability · **OPEN** — 2026-09-27 · history: archive row #133

Only with `inference.persistent_pipeline_stream` on (default off): the coordinator waited out
600 s. Reproduce with debug logging on both ends around `read_frame` / the writer on a 1.8 MB
frame, then add a receipt ACK or a per-frame read deadline mirroring request-response's
`FORWARD_ACK`. A prerequisite to ever defaulting the stream on (`docs/plans/faster_than_local.md`
§ 4 item 7); with V1Lazy (#136) request-response is as fast, so there is no speed reason to.

### Routing and placement

#### #250 — A model held whole here, which the card holds only part of, can be routed through peers at a fraction of local speed
`P1` · routing · **PARTIAL** — 2026-10-12 · history: report #035 (a tester, RTX 3060 12 GB, v0.3.232-.234)

Qwen3-30B-A3B Q4_K_M, every part held here, 18 of 48 layers on the card: a 2,728-token prompt
was planned through two peers in Italy and Belgium (`predicted_ms=278167`, `cold:100339`,
`net:94208`) and ran at **0.1 tok/s**, where Ollama on the same machine, card plus processor,
answers in ~5 s. Several contributors (diagnosis rule 6). Each is a hypothesis until a rig runs the shape:
- **The soft-unload** (#248, closed): a card more than 70% full of this model narrowed its
  window to the first and last parts plus a few middles, so this node could no longer run it alone
  and the middle had to go abroad. The route `aa3d1e66 → 99aafc41 → 7c10ea04 → aa3d1e66` is that
  shape.
- **No layer is priced as the model it belongs to.** (Corrected 2026-10-12: this bullet first said
  the hybrid price came from the whole file's bytes; it never sees the model.) Every candidate's
  per-layer compute is `1000 / est_tokens_per_sec_7b / BASELINE_LAYER_COUNT` (`parallax.rs`
  `vertex_cost`), where `node_tokens_per_sec_7b` = memory bandwidth ÷ `MODEL_SIZE_7B_Q4_GB` (4.4 GB) ×
  efficiency. That holds for this node, for a gossiped peer, and for a measured peer, whose
  `PeerSpeed` ms per layer is keyed by node only. Its samples from a 2B therefore price that peer's
  30B layers too. Nothing in the scheduler, `parallax` or the planning half of the process pool
  reads `expert_count` or `expert_used_count`. Between two compute plans the error cancels; against
  the network and cold-load terms, which are real milliseconds, it does not. A 30B-A3B Q4_K_M
  layer reads ~35 MB per token (~19M attention parameters plus 8 of 128 experts at 4.7M each)
  against the reference layer's ~137 MB, so its compute is priced ~4× too dear and a remote card's
  advantage over the local processor is overstated by as much. A dense 32B Q4_K_M layer (~310 MB)
  is priced ~2.3× too cheap, the other way.
  **Design** (Petals measures throughput per MODEL, running one real block of it — `throughput.py`,
  cache key `model_{name}_device_…_dtype_…`; a mixture-of-experts block then counts only the experts
  it routes to): scale each layer's prior by the model's ACTIVE bytes per layer over the reference
  layer's (`MODEL_SIZE_7B_Q4_GB / BASELINE_LAYER_COUNT`). Active bytes are the layer's non-expert
  tensors plus its `_exps` tensors × `expert_used_count / expert_count`, from the manifest's tensor
  table (sizes and names) and the header (`GgufTensorMeta` needs `expert_used_count`, serde
  default). Store a measured peer's coefficient per REFERENCE layer (divide by the sampled model's
  factor when observed, multiply by the priced model's), so a sample on one model prices another.
  **Done, the prior half (2026-10-12, released in v0.3.236):** `NodeCandidate::layer_work` from
  `GgufTensorMeta::active_bytes_per_layer` (the header's own tensor sizes; the manifest's table is
  not needed), read through `NodeCandidate::prior_ms_per_layer` by both routers;
  `a_light_layer_is_not_sent_a_round_trip_away_for_a_faster_card` flips the route on the factor
  alone and is red without it. A/B `SWARMLLM_LAYER_PRICE=reference`.
  **Done, the measured half (2026-10-12, released in v0.3.236), changed from the design:** a peer's figures
  are kept per (peer, model) (`peer_speed::ModelSpeed`, read only through
  `SharedState::peer_model_speed`), and a model the peer has not run for us is priced from its
  stated speed. NOT per reference layer: a card's decode step is bound by kernel submissions (a
  small model's layer costs more than its bytes say) and a decode figure has the round trip folded
  in, so a scaled figure would be wrong twice. Gossip keeps `ms_per_layer` raw for older nodes and
  adds `LatencyObservation::layer_work` (serde default), which a receiver divides by. Red without
  it: `a_peer_measured_on_one_model_is_unmeasured_for_another`.
- **The prompt is priced on the processor prior** for a hybrid local route even with layers on the
  card, while #138 made a mixture-of-experts prompt 3.5-4.7× faster on both. (#129 points here:
  this is the one place it is worked.)
- **An idle model held the card** (#249, closed): a 14B "idle 20m" still had 9.2 GB, which is what
  left the 30B 18 layers.
- **The planner weighed a card reading the last admission had stored, and never weighed a fresh
  split's processor share** (closed 2026-10-15, report #036 — the detour REPRODUCED on v0.3.238 by the
  same tester). Qwen3-30B-A3B Q4_K_M, 3,729-token prompt: card free, `route=local` (cold 51.8 s, warm
  6.7 s); with Ollama holding ~10 GB of the card the planner still planned all 48 layers here, the
  spawn read a 556 MB budget, put the model on the processor and was refused 18,719 MB of system
  memory in 247 ms, and the re-plan (`local_memory_refused_the_load`) sent 8 layers to a peer abroad:
  cold 137.5 s at 0.2 tok/s, warm 10.6 s. Two gaps: every planning reader of the card budget
  (`would_fit_on_gpu`, `gpu_estimate_and_fit`, `partial_gpu_layers` for a plan,
  `local_layers_leaving`) read the atomic the card-free spawn had set to ~7.5 GB, so the plan saw a
  fresh card/processor split (`fresh_run_layers_for_planning` = 48) where the spawn found none; and
  that allowance never weighed the split's processor share against the RAM cap the spawn would.
  **Done:** `ModelProcessPool::card_budget_for_planning` reads the card live where a spawn could
  follow (card in play, no live worker of the model), one reading per 2 s
  (`PLANNING_CARD_READING_FOR`) — Ollama re-reads GPU and system memory before every placement
  (`server/sched.go`), and the promotion already did (#260); `fresh_run_layers_for_planning` takes
  the requester and weighs the split's processor share as the spawn will (`ram_charged_share` →
  `ram_verdict`). Red without each: `a_plan_weighs_the_card_as_it_stands_where_a_spawn_could_follow`,
  `a_fresh_split_is_planned_only_if_its_processor_share_would_be_admitted`. The router's line for
  this re-plan no longer says "transient". **Answered by the same tester (2026-10-15): the CAP
  refused it**, and the cap was their own: "budget allows 16000 MB (resources.max_ram_mb is 16000
  MB) and 0 MB is already in use", with ~31 GB of the machine available. An explicit `max_ram_mb`
  binds the owner too (#266), so 40 of 48 layers fit here (`max_hostable_layers=Some(40)` on the
  re-plan) and the refusal was the setting doing its job.
`split_rig.sh softunload` (2026-10-12; Qwen2.5-14B whole here, a peer holding its middle) did NOT
reproduce the detour: v0.3.235 narrowed the window, then the next request reloaded the model
locally and answered here (23.2 s vs 9.6 s on main). By the 7B prior the local hybrid route prices
at ~84 s against the reported pipeline's 278 s, so in the report the local route was most likely
unavailable (narrowed) rather than out-priced. Priced by the design above, the report's plan
changes little: compute scales down on both routes, and the local route then wins by more.
**Pre-tag rigs (2026-10-12, main `76d765d6`, `split_rig.sh layerprice`, ~/swarmllm-250/pretag*.out):**
Qwen3-30B-A3B Q2_K whole here (card + processor), a peer holding the middle on the processor —
both prices plan it here alone and answer 200; `layer_work=0.199` with the fix (1.0 without), so
the mechanism fires, but the OLD price did not detour it either: this shape does not reproduce the
report. Qwen2.5-14B in the same shape (`layer_work=1.29`, the dense model priced dearer): here
alone under both prices, the same reply — no route moved. A far split of the 14B on the processor
could not be built on this machine (the peer holds 30 of its 48 layers: `SwarmShortOfMemory` in
both arms). The v0.3.236 gate repeated both shapes on the released build (steps 12ax and 12bb),
with the same plans and the same `layer_work`. Left: the report's own shape (a 12 GB card, peers
abroad, a 2,728-token prompt) has not been reproduced, and the prompt-pricing bullet above is
untouched.
**The detour's ROUTE is "start and finish on this computer" (2026-10-15, from report #036's
`DIAG: pipeline candidate` lines, sent by the tester).** The re-plan offered `7c10ea04` holding all
48 layers (est 7.27 tok/s, 349 ms) and `96842635` holding `[(19,27),(33,39),(42,43)]` (4.88, 303 ms),
and took local L0-19 → `96842635` L19-27 → local L27-48: cold 137.5 s at 0.2 tok/s (`968ad99f`), warm
10.6 s (`b9af8f3e`). Priced by `parallax::vertex_cost`, handing `7c10ea04` the whole model costs
~0.7 s of network once, and the middle segment ~38.8 s (2 × 303 ms × `ASSUMED_FORWARD_PASSES`), so the
search did not CHOOSE the split on price: the whole-model vertex was not eligible. The node held
all 34 parts (its diagnostics), so `encrypted_pipeline_auto` had switched prompt privacy on, which
keeps the first and last segments here (`parallax.rs` `source_ok`: `encrypted_pipeline &&
c.node_id != local_node_id` → not a source) and leaves a holder of a middle range as the only way
to place the 8 layers the owner's cap left over. #035's route `aa3d1e66 → 99aafc41 → 7c10ea04 →
aa3d1e66` has the same shape on the same node: the soft-unload narrowed the window, and privacy is
why the rest went to a middle instead of to one peer whole. Not confirmed from the node itself (no
`decided_by` then; the same node's report 1 shows a model with both ends here reading
`encrypted_pipeline: true` with no override); the other explanation, `7c10ea04` below the
plaintext-prompt trust bar (`trusted_with_the_plaintext_prompt`), is not excluded by these lines.
Dropping privacy because a route is slow was researched and rejected (`privacy_cost_ms`: a
downgrade an adversary triggers by being slow, RFC 7507); the owner is told the cost instead (#165).
So this detour is the setting working as designed. What stays open here: the prompt-pricing bullet,
and whether #165's notice reached this owner (nothing in the report mentions it).

#### #237 — A peer's room for a conversation is not advertised, and the hand-off counts the prompt only
`P3` · routing · **OPEN** — 2026-10-08 · history: report #005 (a tester, v0.3.230 code); related #194

**Re-ranked P2 → P3 on 2026-10-09, from what could be measured.** This node led ONE request in
the first ~10 h of v0.3.231, so its log cannot count the swarm's refusals; the user-visible failure
of #005 — a reply cut off mid-stream — is CONTINUED on another computer since v0.3.231 (#236,
`router::continuation_after` covers a peer's mid-reply refusal); #235 removed the room's most common
thief; and since #194 a card holds three times the positions in the same memory, and a coordinator
prices a peer's prompt cache as the peer keeps it (`features::KV_HALF_ON_CARD`). What remains is a
hand-off that wastes its first attempt. The shape below still stands if refusals are seen again.

The coordinator handed a 30B whole to a peer advertising `free_vram_mb=7598` and priced it at
48 layers (`max_hostable_layers`: the PROMPT's KV per layer against the advertised free memory)
while that peer's worker admitted conversations against a 469 MB KV budget. Nothing in
`NodeCapability` says how much conversation memory a peer's worker for a model would give, and
nothing reserves the REPLY on the coordinator's side (the worker itself reserves
`reply_reserve_positions` — at most one 512-position quantum — at admission, and claims the rest
as it grows). Petals' server publishes `ServerInfo.cache_tokens_left` and its client routes with
`cache_tokens_needed = max_length` (prompt + new tokens): a server without that room is
PENALISED (`alloc_delay`, 10 s on the edge), not excluded (`client/routing/sequence_manager.py`
`_has_cache_for`). Shape: a `#[serde(default)]` figure per resident model
(`ResidentModelLayers`, which is already gossiped) — the positions its worker would admit now —
reported by the worker over IPC; the hand-off and the search charge prompt + reply reserve
against it, and price (not bar) prompt + `max_tokens` beyond it, the stale-figure argument
Petals makes. #235 removed the way that room most often vanished (finished conversations kept
for the worker's life), so measure how often a refusal still happens before building this.

#### #3 — The routing cost model mis-prices a split by regime: a chain its head leads priced as if driven from here, a long reply under-priced by its length, speculation unpriced
`P2` · routing · **OPEN** — 2026-09-08 · history: archive row #3 and § "The routing cost model's network term overestimates a boomerang" (three dated measurements)

**Where it stands (2026-10-18, the fit below):** the reply's own `max_tokens` now caps the
passes the per-token terms are charged for (`NodeCandidate::forward_passes`; 16-token replies
0.49 → 0.97 of their price). Left, largest first: (a) ~~a chain holding no part of this node is
LED by its head (`remote_generate::delegation_eligible`) but priced as if this node drove every
token through every segment~~ (0.10-0.24 of its price warm) — **fixed 2026-10-18, released in v0.3.243**:
`parallax::chain_is_led_by_its_head` prices it by its hops (half the coordinates' round trip
between consecutive peers and back to the head, as Petals prices its edges), in one search pass
per head; the network term of an IT → BE chain at 64 passes went 57,856 → 3,620 ms
(`docs/invariants/scheduling.md` § "A chain its head will lead is priced as its head leading
it"). A chain this node drives through chaining peers is still priced a round trip from here per
segment, and so is the head's own re-plan of a led chain — (a2), smaller, untouched. The only field evidence is three driven chains through two consecutive peers on 10-14 (`225e6fe7 → BE → AU`, 0.72 and 0.92 of their price at 64 passes for 96-token replies, ~0.5-0.6 re-priced at their length), and they ran during the drafter rigs, so (d) and (a2) cannot be told apart in them; a fix needs a driven chain over WAN peers to measure; (b) ~~a reply longer than 64 tokens is
under-priced by its length ÷ 64~~ (2.49x / 2.62x at 160 tokens on two peers' hand-offs) —
**fixed 2026-10-18, released in v0.3.243**: the passes are this model's mean recent reply length
(`daemon::state::reply_lengths`, the last 32 finished replies once there are 4), still capped by
`max_tokens`; left is one reply's spread around its model's mean, which nothing before the reply
can read; (c) ~~a peer's prompt cache is unpriced~~ (a repeated 505-token prompt priced
20 s, read in ~0: 0.25x) — **fixed 2026-10-18, released in v0.3.244** (§ Closed): a peer is
credited the opening of a prompt this node handed it in the last 300 s; (d) ~~the drafter's gain is
unpriced~~ (0.39x, 10-15) — **fixed 2026-10-19, released in v0.3.244** (§ Closed): a segment of a split
driven from here is priced for the rounds its model's splits took (tokens per round, measured where
each round leaves); found by (a)'s check (2026-10-18, five requests on
the live swarm): (e) ~~a led chain is charged a load that never happens~~ — **fixed 2026-10-18,
released in v0.3.244** (§ Closed); (f) ~~a led chain's peers are never measured from here~~ (a
warm two-peer 16-token reply priced 2.3 s of compute, took 19.4 s) — **fixed 2026-10-18, released
in v0.3.244** (§ Closed): each worker publishes its own decode time per layer, as Petals servers
publish `inference_rps`, and a coordinator takes it as a floor under the stated speed.

`scheduler/parallax.rs` charges a remote hop a flat `ASSUMED_FORWARD_PASSES = 64.0`. The error
grows with reply length — 1.0 at 8 tokens, 2.4-3.3 at 120-239 — and biases every delegation
decision since v0.3.164, silently. The structural claim is settled; **do not tune the
constant**. ~~Next: an estimator of the expected reply length~~ — superseded 2026-10-10 by the
collected data below: the error follows the HOPS, not the reply length, so do not build it
(but see 2026-10-18: within ONE shape the error IS the length). Also: peer speed does not track RTT, and a chain runs at its worst
segment's pace, so a term linear in layers cannot express it. This gates turning
`parallax_partial_ranges` on (#180) — on 2026-09-20 the flag on made a 4-segment, 3-peer
route that did not finish in 580 s. Method: `memory/archive/perf_baseline_0920_post192.md`,
gotchas #658-#660.

**Field evidence, 2026-10-06 (a tester, two nodes, 0.3.173 and 0.3.228):** the flat 64 passes
multiply the COMPUTE term too, not only the network one — `cost_compute_ms = observed_ms_per_layer
× 32 layers × 64` reproduces his logged figures to the digit (44.52 × 32 × 64 = 91 180.8). His
requests carried `max_tokens=24` and produced 2 tokens: a warm peer (with an observed per-layer
figure) was priced 16.6-19.5x too slow, a cold one (falling back to `est_tokens_per_sec`) 2.8-3.1x
too fast. On 0.3.228: `predicted_ms=2817` against `total_ms=7107`. A request's own `max_tokens`
is an upper bound on its decode passes — capping the 64 by it is a bound the request states, not
the tuning this entry rules out; it does not replace the reply-length estimator. He also saw
`max_hostable_layers` read 97-103 cold, 14 008 warm, and `Some(145)` for a 32-layer request —
unverified here; check whether it is a capacity bound never capped at the model's layer count.

**The collected `predicted_ms` says the error is the SHAPE, not the reply length (2026-10-10).** 333
distributed requests on the live node's log (2026-09-17 → 10-07, `request complete … predicted_ms=
… tokens=`, mostly this project's own gate and rig traffic): the error's correlation with
64 ÷ tokens is 0.00 (log-log, slope 0.00). Replies of 33-64 tokens are priced 1.10× their real
time, of 65-129 tokens 2.29× — LONGER than the assumed 64 and still over-priced, where a
reply-length error would under-price them — and rescaling each prediction by tokens ÷ 64 makes the
fit worse (0.17-3.21×). By shape: one remote segment 1.08×, two 2.00×, three 3.35×. So the
reply-length estimator would not fix what was collected; the over-charge grows with the HOPS.
Candidates, unmeasured: every split path here guesses ahead (the n-gram path, the guess-check
stream), so a reply crosses each hop fewer times than it has tokens and the per-hop term charges
for crossings that never happen; the per-visit fixed cost may also be counted on top of an
observation that already carries it. ~~Next: log each chosen plan's cost terms beside its
outcome~~ — **done 2026-10-10** (`b8f0b82b`, released in v0.3.234: `net:/compute:/prompt:` on each
priced route). Next: once enough split traffic has collected, fit per term — collect, do not tune. Script: re-run the analysis with
`grep "request complete" node.log | grep predicted_ms` over the fields above.
Field point, 2026-10-13 (a tester, v0.3.236): a six-segment boomerang through three countries
(37-token prompt, 120 tokens) was predicted 8,772 ms and took 59,457 (6.8×), its terms
`net:0/compute:5441/prompt:2902` — `net:0` is by design where every peer is measured and no
per-visit split is known (`vertex_cost`: the round trip is inside the per-layer observation), so
here the hops' cost sits inside a compute term that priced the chain at 5.4 s of the 59.5 s it
took.
Field data, report #036 (2026-10-14, v0.3.234-.238): 26 priced requests from one node (10-11 →
10-14), actual over predicted 0.02× to 6.8× — but 20 failed (14 of them local refusals in 2-4 s
after a 0.8-1.1 M ms price), so only the 6 that completed are comparable. New since: a warm 3-segment
route through Thailand (3,729-token prompt, 29 tokens) predicted 26,792 ms, took 10,603 (0.40×,
`net:0`); the same route cold 119,766 → 137,448 (1.15×).
**Fit per term, 2026-10-15** (this node's log, 135 completed splits 10-14 09:15 → 10-15 02:51 —
all this project's #273/#275 rigs: Qwen3-30B-A3B with one Australian peer, so ONE peer's terms;
`examples/split_cost_terms.py` over `request complete … predicted_terms=`), actual ÷ predicted for 2+ segments:
warm with the per-visit term measured (`net` ≈ 64 × 650 ms) **0.78** (n=31, 0.14-0.97) — the
10-10 "two segments priced 2.00×" does not reappear; warm with `net:0` (the hop inside the
per-layer figure) **1.53** (n=35, 1.19-1.86 — under-priced); cold **0.52** (n=34: the cold term
~58 s for a local load of ~9-20 s — fixed the same day, the rate is kept across restarts, § Closed);
the drafter arms **0.39** (n=25: speculation's gain is not priced). Left: the `net:0` regime's
under-charge (a peer whose samples never separated the per-visit cost), and pricing a drafter.
**Hypothesis, unverified**: every reply in that regime ran 96 tokens against the 64 passes assumed,
and 96 ÷ 64 = 1.50 against the measured 1.53 — there, the error IS the reply length; the 10-10 "0.00
correlation" pooled regimes whose errors run in opposite directions. In the `net>0` regime the
per-visit term (~650 ms) is ~1.7× a real whole token (~400 ms), which a 96 ÷ 64 under-count
partly cancels. Needs a second peer's traffic, and replies of several lengths, before any term is
changed.
**Fit, 2026-10-18** (live node, 06:32-06:38Z, `~/swarmllm-3/collect.sh`: 15 priced requests, one at
a time — qwen3.5-9b and xLAM-2-8B at `max_tokens` 16/64/160 × a 29- and a 490-token prompt,
Qwen3-30B-A3B at 16/64; peers AU, IT, BE; `split_cost_terms.py --rows`):
- **The hypothesis holds, on two more peers.** Whole-model hand-offs priced from a per-token
  figure measured on that shape ran 2.49x (AU) and 2.62x (BE) at 160 tokens, 1.18x at 64 — the
  error is tokens ÷ 64. **Fixed for the short side**: a reply's `max_tokens` caps the passes
  (`NodeCandidate::forward_passes`, `docs/invariants/scheduling.md` § "A reply is priced for no
  more passes than it may take"); the six 16-token replies 0.49 (0.10-1.08) → 0.97 (0.25-1.41)
  re-priced. The long side needs a length no request states — an estimate, not a bound.
- **A chain this node holds no part of is priced as if driven from here.** All three such plans
  (AU+IT for qwen3.5, IT+BE and IT+BE+AU for the 30B) were handed to their head to lead
  (`delegated split: this node holds none of the plan`), yet `vertex_cost` charged every segment
  this node's own per-visit cost × the passes: 0.10, 0.14, 0.21, 0.24 warm (net 96,640 ms
  predicted for 16 tokens that took 13.5 s in all), 0.73 cold. The head's segment is entered
  once, and the other hops are the head's, which this node has never measured.
- **A peer's prompt cache is unpriced**: xLAM's 505-token prompt sent to BE a second time was
  priced 20.2 s of prompt and the whole request took 7.1 s (0.25x, twice).
- The first request for qwen3.5-9b split it across AU (room for 9 layers) and IT (26) although
  both hold all 32 — capacity, not price: neither could take the whole model at that moment.

**(g) A cold prompt priced 3.6x its time** (field report 2026-10-19, a tester, v0.3.244,
`e736f3cc`): 3,718 tokens over their node (L0-14 and L42-48, on a 12 GB card) and BE (L14-42),
predicted 549 s with `prompt:498030`, took 137.7 s — BE's segment 116.5 s of it. The warm repeat's
gap (29.3 s predicted, 120.7 s taken) is #321's re-sent prompt, fixed; this one is not. Unread:
which segment's prompt figure was high — a prior, or a rate taken from one small forward. Next: a
cold request's per-segment prompt terms at `info`, then compare each with its segment's time.

#### #180 — A node holding the whole model takes every request for it; partial ranges stay off by default
`P2` · routing · **PARTIAL** — 2026-07-28 · history: archive § "A node holding every shard monopolises the model"

Delegation to a faster GPU peer shipped (2026-08-18), with the delegated-versus-mid-chain cost
split (2026-08-20). But `inference.parallax_partial_ranges` defaults false (forced on only for
`encrypted_pipeline`): measured splits were slower (12.0 s vs 10.2 s median), so a busy or
slow sole holder keeps all its requests — no load spreading, no standby for a sole holder.
Re-run the A/B (whole vs split, including a prefill-dominated 585-token prompt) now that
`observed_delegated_ms_per_layer` and per-token mid-chain charging exist, on a real
multi-node swarm (confirm the PID changed and `segments=` in the log), and flip only if the
split wins. Also feed observed per-layer latency into `compute_segment_timeout` instead of
the fixed 2 s/layer guess. Depends on #3. **With it off, the GREEDY fallback plans every split
whose holders overlap** (whole ranges A 0-21 / peer 3-28 have no chain: 16 of 16 plans in
2026-10-10's #10 rigs), so greedy's pricing is the pricing those splits get — it gained the
cold-load charge that day.

#### #129 — A model a few MB too large for the card is sent on a boomerang across continents
`P2` · routing · **PARTIAL** — 2026-09-27, narrowed 2026-10-04 and 2026-10-05 (fit verdict fixed in v0.3.227), 2026-10-07 (cold loads waited for and priced, released in v0.3.230) · history: archive row #129

**v0.3.225 shipped half the fix, and the field check after deploy failed** (2026-10-05 01:30
UTC): a cold Mistral-7B on this 8 GB node still went through `4a3ac72e` (Italy), 47 s. The
pool's bound (`max_local_hostable_layers`) learned to count the loader's card/processor split
(`partial_gpu_layers`) — but only in its card branch, and the planner asked with the SPEED
answer `serves_on_cpu`, which is "processor" for any model that does not fit the card whole
(#444). With TinyLlama already on the card, Mistral "did not fit", RAM alone held 29 of 32
layers, `local_route_available=false`. (On 10-04 the same request asked the card and got 17
— the card alone — which is the case the first fix covered.)

**Fixed in v0.3.226 (released 2026-10-05):** the planner asks
`ModelProcessPool::max_hostable_layers_for_planning` and `held_layer_ranges_for_planning`,
which choose the device the LOADER would use (`planning_on_card`). Test
`the_planner_weighs_a_local_model_the_loader_would_split_on_the_card` goes through
`gather_candidates`: `Some(0)` on the old call, `Some(32)` now. **Verified 2026-10-05 on a
`--features cuda` build of main run as the live node** (`~/swarmllm-gpu-1005/verify129.sh`): with
TinyLlama on the card, a cold Mistral-7B answered `route=local segments=1` in 6.7 s, loading
included (v0.3.225: Italy, 47 s); admission put it on the card whole beside TinyLlama
(`committed_mb=1044 estimated_mb=5517 budget_mb=6561 headroom_mb=0` — it fitted with nothing
to spare; that build's daemon no longer holds the 137 MiB context, and this one run cannot say
whether that is what made the difference). The .226 gate's 12q ran it on the artifact in an
isolated node: `route=local`, admitted whole (headroom 24 MB) — but .225 also runs locally with no
peers to send it to, so only a node WITH peers can tell the two apart; re-read a cold 7-8B's route
on the live node after .226 spreads.

**Residual:** the local candidate is still priced at its processor speed rather than as a
hybrid — conservative. A remote hop's network cost is charged per segment rather than per
token / per crossing (the regional-pipelines plan, § "What is actually missing", 2). Do not
tune `ASSUMED_FORWARD_PASSES` (#3).

**The field reading with peers, on .226 (2026-10-05 08:57 UTC, live node, every peer on
.225/.226):** Qwen2.5-Coder-7B — every part here, two smaller models' workers on the card
(~5.6 GB free) — went whole to `4a3ac72e` (Italy, 349 ms): `local_route_available=true`,
`local_processor_cost_ms=18747` against `pipeline_cost_ms=5057`. The first answer took 58 s (the
peer loaded the model), the next ones ~2 s. That is this residual deciding, not a regression of
the fix above: the planner now SEES the local route and then prices it at processor speed.

**Reproduced and narrowed 2026-10-05 11:44 UTC (release binary, live node, peers on .226) —
the planner and the loader disagree about what FITS, before any question of hybrid pricing:**
with qwen2.5-0.5b and qwen3-1.7b idle on the card (3.8 GB used), Coder-7B (cold) was judged "does
not fit our GPU", priced `local_processor_cost_ms=27025` against `pipeline_cost_ms=4495`, and sent
to `4a3ac72e` — which was cold too: **46.5 s** for 120 tokens (`predicted_ms=4495`). The same
request with every peer excluded: the LOADER freed both idle workers ("Freeing graphics memory from
an idle model so the requested one can use the GPU", `freed_mb=3754`), admitted the 7B whole on the
card (`estimated_mb=5232 budget_mb=6354`) and answered in **13.2 s, load included**; warm, both
routes then stayed local at ~60 tok/s (150 tokens in 2.5 s). So the fix is the planner asking the
pool's admission WITH the idle-model reclaim (`planning_on_card` / `serves_on_cpu` ignore it), not
re-pricing a split; a cold peer's load time is also unpriced (46 s against 4.5 s predicted) — a
second contributor. Measure that shape: warm two small models, then the 7B cold, normal routing vs
`swarm_route.exclude_nodes` = every peer (on the live node with peers; the isolated gate step 12s reads `fits_on_gpu`).

**Fixed in v0.3.227 (released 2026-10-05 23:45 UTC) — the fit verdict counts what admission would
reclaim:** `ModelProcessPool::fits_in_budget` (behind `would_fit_on_gpu`, `gpu_estimate_and_fit`
and `serves_on_cpu`) answers "fits" when the reclaim's own dry run (`reclaimable_vram_mb`) would
make the room, as the planner's ceiling already did (#125). Test
`a_model_fits_the_card_if_admission_would_free_an_idle_one_for_it` (red without the reclaim term).
**Verified 13:36 UTC on a `--features cuda` build run as the live node, same shape, safety kit, 0
driver events** (`~/swarmllm-129b/verify129b.sh`): the cold Coder-7B stayed **local in 11.0 s**
(`sched_ms=0` — the local fast path took it), the loader freed both idle models (791 + 2963 MB) and
admitted the 7B whole; warm 2.2 s for 120 tokens. Release binary, same shape: Italy, 46.5 s.

**Residual (one part left):** A model that does NOT fit even after the reclaim is still priced
at processor speed rather than as the loader's card/processor split (above). **Fixed 2026-10-08,
released in v0.3.231:** `PipelineScheduler::local_speed_off_the_card` blends the card's share
of the split in (`split_tokens_per_sec`), for the search and the hand-off gate alike —
`docs/invariants/scheduling.md` § "A node holding every layer that would run the model on its
processor…", "A split is priced as a split". What #129 still carries: the prompt pass keeps the
processor's prior for a split (tracked in #250 — one defect, worked there), and a remote hop's
network is charged per segment rather than per crossing (the regional-pipelines plan).

**(1) — a cold candidate's load was not priced — fixed 2026-10-07, released in v0.3.230.**
`PeerResidency` bounded a peer's MEMORY, never its time, so a peer that must first load the
model was predicted like a warm one (4.5 s predicted, 46 s taken). Now `parallax::vertex_cost`
charges `cold_load_ms` = the candidate's own ms-per-layer × the layers it would ADD, once per
request, from the same residency reading as the memory bound; this node is charged its own
too. Each node times its loads in the worker (`WorkerMsg::ModelLoadTimed`), averages them
(`process_pool::LoadRate`) and advertises `NodeCapability::model_load_ms_per_gib`; a node that
reports none is charged `UNMEASURED_LOAD_MS_PER_GIB` (10 s/GiB, the pessimistic side of a
measured 1.3-26 s/GiB spread). ServerlessLLM (OSDI '24) schedules on each server's measured
loading speed for the same reason. Verified end to end (`cold_load_test.sh … price`, netns,
2026-10-07): a client that had heard no rate priced the server's cold TinyLlama load at 6,229 ms
(the prior); the server timed its own slowed load at 33,818 ms/GiB; a fresh client then priced
the same load at 21,064 ms — that rate × 0.62 GiB. The field reading needs peers on the release:
a cold 7-8B peer should then lose to a warm one or to a local route that is truly cheaper.

**(1)'s other half — the DEADLINE — fixed 2026-10-07, released in v0.3.230:** a cold load
did not only cost time, it could fail the request. Live on v0.3.229 (00:23 UTC): Qwen 3.5 9B
handed whole to `4a3ac72e` (Apple M4, restarted ~6 min earlier, its measured latency 10-30 s
while it settled) gave no first token inside 132 s (`FIRST_TOKEN_TIMEOUT` 120 s + 0.5 s × 24
prompt tokens), was penalised, and the re-plan had no other holder of layers 13-24 → 503
`model_incomplete_in_swarm`; asked again 47 s later, 2.7 s. The segment path had been given a
240 s cold-load allowance on 2026-08-01 for the same failure ("a CPU peer took ~120 s to load an
8B model"); the hand-off — the path most single-model requests take — and the HTTP forward to a
pool peer never were. Now `pipeline::LoadAllowance` is the one reading (cold unless the peer
answered a forward of the model within 15 min), a REQUIRED argument of `first_token_timeout`
and asked by `SegmentBudget::for_forward`; a delegated split is charged per cold segment. The
first-token wait is taken in 5 s slices and ends at once when the peer's last connection closes
(the serving node aborts the generation then — `handle_connection_closed`), so the longer
budget never makes a vanished peer cost more. Rig: `examples/cold_load_test.sh` (the server's
load delayed by `SWARMLLM_FAULT_LOAD_DELAY_SECS`), run 2026-10-07 at 150 s, safety kit, 0 driver
events: the fix answered (200, 146.5 s wall; client logged `first_token_budget_s=388
cold_loads=1`) where the v0.3.229 client failed (503 at 143 s, `timed out waiting for token
(first=true)` after 148 s); server killed 20 s into its load — the fix gave up at 23.1 s
(`disconnected before its first token`), v0.3.229 at 144 s. The harness default is now 200 s
for margin over the old ~148 s budget (this VM's monotonic clock ran ~10% fast against the
log's wall clock: a 150 s sleep spanned 139 s of timestamps).

**Field evidence for (1), 2026-10-06 (a tester, CPU node, five runs to one peer):** cold 19.7 s and
19.7 s against warm 5.9, 6.3 and 5.2 s — the first segment alone 16.9-17.3 s cold vs 2.4-2.8 s warm.
The peer unloaded between 6 and 16 min idle (he predicted warm after a 3.5 min gap and got 5.2 s).
His point for anyone benchmarking: a first-touch figure reads ~3x worse than steady state.

#### #192 — Nothing routes on the distance between two peers
`P2` · routing · **PARTIAL** — 2026-09-02 · absorbs archive § "Speeding up inference BETWEEN nodes" ideas 3 (ring decode) and 4 (peer-to-peer RTT); plan: `docs/plans/regional_pipelines.md`

A split is only fast when its machines are close (0.35 tok/s Thailand↔Italy against 6.76 at
18 ms). `ack_srtt_ms` measured on real forwards already feeds the coordinator's view of its
own links, and network coordinates exist (`swarmllm_types::netcoord`), but no plan prices
the A↔B hop between two peers in a chain — a 2× swing. The design and its order live in
`regional_pipelines.md` (coordinates → a `predicted_rtt_ms` consumer). ⚠ Until a consumer is
validated, coordinates may EXCLUDE the far half but must not ORDER the near half. A true
ring (the tail hands each token straight to the head) would save one more leg per token
beyond the delegated split (#143, shipped); revisit only with measured peer-to-peer RTT.

**Placement by distance — phase 1 built 2026-10-21** (`auto_manage/coverage/near_chain.rs`,
plan § Stage 3): nearby machines (measured ≤ 30 ms) plan one chain alike for an asked-for model
none of them can run whole, and each fetches and keeps its segment; chosen by simulation
(`examples/placement_sim.py`). **Verified end to end 2026-10-21** (`examples/near_chain_rig.sh`):
a near machine fetched its part and, once it gave back the layers it held loaded outside its
segment (`near_chain::loaded_outside_near_segment` — routers price a peer's room beside what it
has loaded), the next 160-token reply ran over the two near machines in 13.9 s against 36.8-41.4 s
split with a far one; the control (`SWARMLLM_NEAR_CHAIN=0`) fetched nothing. Residual: the field
read on this machine + the LAN box (the `DIAG: fetching this computer's part of a chain of nearby
machines` line, then a split over them), and phase 2 — published near lists — only if the field
shows neighbourhoods that overlap without coinciding. **Security review 2026-10-21 (v0.3.247)**:
two of four findings fixed in v0.3.247 — loaded layers are given back early only once every
machine HOLDS its segment (a peer's fetch is its own word, `chain_held_by_its_machines`), and an
unchosen model with its safety training removed is planned no chain and no header
(`may_plan_a_near_chain`). Open, both Low: (a) a peer within 30 ms can steer which of an asked-for
model's parts this node fetches, inside the owner's disk budget — the plan trusts its
neighbours' holdings as every router does; (b) ten near machines (`MOST_MACHINES`) make the
exact plan up to ~10^7 memo steps, once a `CHAIN_ANSWERS_FOR` minute per model, on the
download pass — measure it before bounding it.

#### #166 — A verify round that carries several guessed tokens never teaches routing what the peer costs
`P3` · routing · **PARTIAL** — 2026-08-30 · history: archive § "Routing never learns a peer is slow on the speculative path"

The single-token half is FIXED and live-verified (16.87 s → 1.36 s). `pipeline/mod.rs` records a
`WorkKind::Decode` sample only when `verify_tokens.len() == 1`; a K-token batch recorded as
decode would inflate ms/layer ~K× and also corrupt segment-timeout sizing. So a workload whose
guesses always hit never refreshes the peer's speed. Fix at the choke point: record in
`pipeline::local::wait_for_result` with the work kind threaded through (covers `dsd.rs` and
`speculative.rs` too), or add a `WorkKind::Verify` with its own coefficient. Whether a workload
is affected: `ngram_rounds` vs `fallback_rounds` in the `SWARM-SPEC L1 ngram-only: complete`
line.

#### #178 — A peer priced slow is never measured again (the routing ratchet)
`P3` · routing · **PARTIAL** — 2026-07-28 · history: archive § "Slow nodes go dark and never come back — the routing ratchet"

Fix (1) shipped in v0.3.70: a measurement older than `RANKING_STALE_AFTER` (600 s) falls back
to the advertised prior. Not built: real exploration (occasionally route a segment or a
prefetch to a stale holder to refresh its estimate, bounded by standby failover) and
preferring slow nodes for background work. The entry's own correction stands: decay alone
does not close the ratchet. Cheapest safe first step: background/prefetch work to slow
nodes. Needs two nodes with a real speed gap and a before/after selection count. Matters
for contributor retention (modest machines that lose one comparison stay unused), not for
correctness.

#### #197 — Whether to hand a whole model to a peer is decided by two constants, not a comparison
`P3` · routing · **OPEN** — 2026-09-03 · history: archive § "Delegation distance should be a comparison, not a constant"; also § "The whole-model hand-off is a yes/no gate…" residual (3)

`DELEGATE_MAX_LATENCY_MS = 1000` and `DELEGATE_MIN_CPU_SPEEDUP = 2.0` still gate
`delegation_target` where the priced search cannot run. Compare `predict_segment_ms` for the
peer plus one RTT against the local processor estimate (`is_cpu_bound_for_lack_of_vram`) and
drop both; keep the candidate ordering (pool first, reachability, latency — test
`a_nearer_peer_still_wins_over_a_distant_one`) and the exclusion of relayed or unmeasured
peers. The delegated split (#143) is a second hand-off shape that should use the same
comparison.

#### #170 — Acquisition has no speed term, so a fast node leaves most of a model on a slow peer and splits it
`P3` · storage · **OPEN** — 2026-07-24 · history: archive § "Splitting a model across the internet is a CAPACITY mechanism being used as a SPEED mechanism"

`auto_manage/scoring.rs` scores what to download without asking whether this node is
materially faster than the best current holder, so a node that could hold the whole model
runs split at the slow peer's pace (35× slower in the measured case). Cheap proof first:
download the missing parts by hand on the fast node and confirm one local segment and the
speed-up. Then add a bounded multiplicative term to scoring — and to retention in
`prune.rs`, or it loops — for parts this node lacks, can fit, and where its MEASURED speed
(`PeerSpeed::ranking_ms_per_layer`, not the advertised figure) clearly beats the holders'.
Must not make fast nodes hoover up the swarm (cap it; keep `spread_bonus` multiplicative).

#### #190 — A GPU node with every layer but no room for the request runs it locally instead of using an idle peer
`P3` · routing · **PARTIAL** — 2026-08-03 · history: archive § "Full local coverage bypasses the network, even with no headroom" (its heading says FIXED; its body says the fix was REVERTED)

`local_fast_path_for` stands aside only for `shed_load_when_busy` (a load count) or a
processor-served model with peers; there is no headroom condition. `would_fit_on_gpu` was kept
as a primitive. `admit_prompt` refuses cleanly at token 0, but whether that re-routes is
unverified. Before re-attempting: log what `assemble_pipeline_for` returns when a gate fires,
whether the parallax search is reached or silently falls back, and whether `route=local` is
merely the label of a single-segment plan. Do not recalibrate the estimator down (measured
+5% high, correct).

#### #4 — Replica-count targets ignore how often a part's holders fail to serve it
`P3` · storage · **OPEN** — 2026-09-07 · history: archive row #4 and § "Replica counts do not react to holders being unusable"

`geo_target_replicas` (`auto_manage/scoring.rs`) takes pool size and demand only, so parts
can sit under-replicated while the system believes they are safe. Deliberately not built yet:
the failures that prompted it were a pricing bug (#022). Precondition: #022 confirmed fixed in
the field, plus a failure class that separates "unreachable / no worker"
(`remote_error_means_missing_shard`) from "refused on capacity"
(`message_means_peer_cannot_serve`). Smallest defensible version: a decayed per-(part,
holder-set) counter of the first class only, with hysteresis — never a continuous multiplier.

#### #198 — A node that appears twice in one chain is capped per segment, not in total
`P4` · routing · **OPEN** — 2026-09-04 · history: archive § "A node that appears twice in one chain is bounded per segment, not in total"

With prompt privacy the local node can hold two ranges of one chain; `max_hostable_layers` is
applied per segment, so the search can give it more than it can load. Admission still refuses
the second range and the coordinator fails over — a 503 and a re-plan, not a wrong answer.
Carry layers-already-assigned-to-this-node in `route_shortest_path`'s state so the cap checks
the running total; measure on a real two-segment plan first (synthetic tests get the marginal
case wrong).

#### #319 — A node that just joined cannot plan with a peer whose parts its capability already lists
`P3` · routing · **OPEN** — 2026-10-20 · found by the real-link rig for #318 (`~/swarmllm-318/rl318.sh`)

A newcomer's planner reads holders only from holder records (`ModelRegistry::shard_holders`).
Those come from `ShardAnnounce` gossip (dispatch's `ShardAnnounce` arm) and DHT providers
(`merge_dht_providers`). The `hosted_shards` in a peer's capability, which arrives at identify
time, is only bounds-checked (`dispatch/mod.rs`, `MAX_HOSTED_SHARDS_IN_CAP`) and recorded
nowhere. So a fresh requester connected to the machine holding a model's first layers answered
"No reachable node holds layers 0-13" for 10 s to over 4 min, until an announcement or a DHT
answer arrived. The rig then kept its requesters running between rounds. The `/v1/chat/completions` "not found" of the same window is fixed
(`SharedState::reject_if_unknown_model` asks the capabilities of peers connected in the last
three minutes, `ManifestSync::catching_up`); planning is not.
Direction: let a connected peer's capability seed holder records it does not contradict, with the
same build check `shard_holders` applies (`expected_build_tag`), withdrawn when the peer
disconnects — or have the newcomer's catch-up carry the holdings point to point, as manifests
are. Either way a capability is a peer's claim, so it may add a holder for planning and never
remove one. Measure the newcomer's first-plan delay before and after.

### Network and transport

#### #155 — AutoNAT redials a peer this node is already connected to, and the per-peer cap denies it, every ~5 s
`P3` · network · **PARTIAL** — 2026-10-02, narrowed 2026-10-07 (measured on the swarm: not seen there) · history: archive row #155, gotcha #774

**Narrowed 2026-10-07.** Settled in the pinned source (libp2p-autonat 0.15.0, `v2`): the server
dials back with `PeerCondition::Always` (`server/behaviour.rs`), our per-peer cap then denies the
connection, the swarm reports `DialFailure`, the server answers `E_DIAL_ERROR`, and the client
records `AddressNotReachable { NoConnection }` — status `Failed`, not retried
(`client/handler/dial_request.rs`, `client/behaviour.rs`). So a denied dial-back IS a false
"unreachable" for a reachable node whose server already holds three connections to it, and our
handler then turns the relay listener on. **But it was not seen on the swarm:** a probe node
joined at `-vv` for 5 min (04:45-04:51 UTC, 9 peers): 13 denials — 11 from the live node on the
same machine over loopback (`127.0.0.1:8810`), 1 a hole-punch upgrade's extra connection
(`count=3` just before), 1 unattributed; 0 AutoNAT dial-backs denied; every AutoNAT verdict
true (the probe is behind NAT, no UPnP). The every-~5 s pattern is co-located nodes: the AutoNAT
client asks servers to test non-internet candidates (loopback, LAN — our handler already
ignores those verdicts, "probed a non-internet address"), and a server on the same host can
dial them. Left: raise the cap to 4 only with evidence from a reachable node (a VPS peer's
log); the loopback churn only matters for two nodes on one machine.

**As filed:**

Seen on the rig at `-vv`: `request_response: forgetting a connection the swarm denied` every
~5 s per peer, each preceded on the dialling node by `AutoNAT server: served a dial-back
probe` to an address of a peer it already holds three connections to (`max_per_peer = 3`).
Each is a TCP + Noise + Yamux handshake thrown away on both ends for the life of the node,
and a dial-back the cap denies is a FAILED probe to the client — AutoNAT may read a reachable
node as unreachable, which feeds relay and hole-punch decisions. Settle first, in the pinned
libp2p-autonat's `dial_back` result handling, whether a denied dial-back counts as a failure.
Then give AutoNAT room (raise the per-peer cap to 4, or a dedicated limit —
libp2p-connection-limits has no per-protocol exemption). A dial by bare address bypasses
`PeerCondition::Disconnected`; check no other dialler (loopback probe, redial queue, PEX)
does the same.

#### #91 — An idle node still uploads ~5.5 KB/s, most of it no longer gossip, and a total traffic ceiling does not exist
`P2` · network · **PARTIAL** — 2026-10-19 · history: archive row #91, `docs/invariants/network.md`

Trickle suppression (`MANIFEST_QUIET_WINDOW`), per-(model, hash) `manifest_heard`,
per-category counters and the region-demand split shipped in .200-.206 and halved it (42 →
~20 KB/s each way; it once extrapolated to ~159 GB/month — the complaint that makes people
uninstall a P2P app). The capability broadcast is change-gated since v0.3.234 (KEP-589).
Re-measured 2026-10-11 on the release node with the fleet on .234 (4.1 h up, 7 peers, idle):
8.7 KB/s out, 5.8 of it gossip; the capability's first copies 976 → 167 B/s; manifests
59% of the gossip sent, every one a repeat, each arriving ~4.2 times. So the manifest
quiet window went 30 min → 2 h (released in v0.3.235, 2026-10-12; RFC 6550's Trickle ceiling is ~2.3 h),
`docs/invariants/network.md` § "The manifests again, once the capability was gated". Re-measured 2026-10-15
(.239, 3.3 → 4.3 h up, idle): 8.8 KB/s out, 6.5 gossip; manifests' first copies 0.0031 a second
(the 2 h window held), but `RegionShardSummary` was a third of the gossip sent, 372 an hour from
the anti-entropy round alone, and read by nothing. It is no longer sent (released in v0.3.240), and scoring
counts its region from holder records (`docs/invariants/network.md` § "Region summaries were a
third of the gossip"). Every topic delivers ~6.3 copies of each message: the mesh was sized from the peer cache.
Open: (1) re-measure once the fleet stops sending region summaries (`swarm/regions` should fall
to demand gossip alone); (1b) the copy factor: the mesh is sized once at startup from the peer CACHE
(`known_peers` 203 on the release node, so the 100+ tier, `mesh_n` 6, on a swarm of ~8: a full
mesh, plus flood publishing). The cache counts ADDRESSES, not peers (`load_peer_cache`, capped at
`MAX_CACHED_PEERS` = 200: "Loaded cached peers from last session count=200" + 3 bootstrap), so
every node that has run a while lands on that tier, and a FRESH node (empty cache, bootstrap
only) runs the smallest (`mesh_n` 2, low 1, high 4) for its whole first session. **Counting
fixed 2026-10-15 (released in v0.3.240)**: `known_peers` is the PEERS the cache and bootstrap list name
(`discovery::peers_named`, the grouping the startup dial already uses). A probe started on a copy
of the release node's database (no network): `count=200 known_peers=26` → `mesh_n` 3, high 6, the
tier written for a swarm that size. **(1) and (1b) re-measured 2026-10-16 with the fleet on .240**
(release node, 6 peers all on .240, idle — no inference, no parts — 2.1 → 3.1 h up, a 59-min window,
`/api/admin/stats` twice): **6.7 KB/s out (8.8 at .239), 3.3 of it gossip (6.5)**; every topic
delivers **2.7-3.2 copies** of each message (`recv_msgs_unfiltered` ÷ `recv_msgs`, from ~6.3);
`swarm/regions` carries demand gossip alone (`ModelDemandGossip` 3.0 a minute, 70 B/s forwarded —
no region summaries); no drops. **`swarm/models` is now 84% of the gossip sent** (2.8 KB/s):
`ModelManifest` arrives 0.83 a minute at ~35 KB each, and each is forwarded on to the mesh. The
3.4 KB/s that is not gossip (Kademlia, identify, pings, direct messages) is the next-largest share.
**(2) shipped 2026-10-17, released in v0.3.241**: a peer advertising `features::MANIFEST_DIGESTS` is caught up
with the versions (`ManifestDigest`, ~200 B a model) and pulls (`ManifestPull`) only what it lacks; each
such peer is sent the versions again every 2 h, and the gossiped repeat waits 24 h while every connected
peer takes digests. A change still goes on gossip in full; older peers keep the old path both ways
(`docs/invariants/network.md` § "Announce, then pull"). It replaces the catch-up that sent a probe node
84 manifests (~2.9 MB) from six peers in 20 s, describing 26 versions. Rig
`examples/manifest_digest_rig.sh`: a newcomer pulls once, a reconnect pulls nothing, mixed versions
learn the model in full.
**Corrected 2026-10-20:** the rig's restarted A tells B and C (its round starts empty); it never
checked what B and C tell A — nothing: their rounds (30-120 s apart) kept a set of peers told and
dropped one only when a round found it gone, so a node restarted on its own heard nothing for up to
2 h. A new connection is now a new catch-up (`ManifestSync::note_new_connection`, gotcha #918). **(2b) re-measured 2026-10-17 with the fleet on .241** (release node, 8
peers all on .241, idle, 1.0 → 2.0 h up, a 60-min window, `/api/admin/stats` twice): **5.5 KB/s out
(6.7 at .240), 2.1 of it gossip (3.3)**. `ModelManifest` arrived ONCE in the hour (0.83 a minute at
.240), so the digests hold. `swarm/models` is still 68% of the gossip sent, but at 1.4 KB/s (from
2.8), and it is now shard announcements (1.4 a minute, ~7.5 KB) and capability updates (1.7 a
minute, ~5.5 KB), each arriving ~2.85 times. What is not gossip (Kademlia, identify, pings, direct
messages) is 3.5 KB/s, now 63% of the upload and the largest share left. The reconnect catch-up
could not be separated in these counters; the rig showed a reconnect pulls nothing. **(2d) a
per-kind meter for direct traffic, 2026-10-18, released in v0.3.243**: `direct_by_kind` in the traffic
payload counts every request-response frame by message kind, request or answer, both ways, in the
codec (`network::bandwidth::DirectKindMeter`). libp2p cannot name its own protocols' bytes from rust
(`libp2p-metrics` labels by transport only; go-libp2p's `GetBandwidthByProtocol` has no rust
counterpart), so what `other_out_bytes` keeps beyond this list's non-inference, non-part rows is
Kademlia, identify, ping and relay. **Measured 2026-10-18** (live node on the meter build, 7 peers,
idle, 1.0 → 1.9 h up, a 3453 s window, `/api/admin/stats` twice): 4.1 KB/s out, 1.6 of it gossip,
2.5 the remainder — and our own messages are **0.19 KB/s** of that (peer exchange answers 115 B/s,
health pongs 55, key exchanges 15, receipts 5). The remainder is libp2p's: ~2.3 KB/s. Gossip's
envelope (sender, sequence number, topic, a 64-byte signature — outside `gossip_sent_bytes`, which
is message data) is ~135 B on each of 1.09 copies a second, ~150 B/s; IHAVE at a 10 s heartbeat
adds perhaps as much. That leaves ~2 KB/s for Kademlia (110 shard records republished every 20 min,
and the lists of up to 20 peers with their addresses every answer to a lookup carries), identify,
AutoNAT and relay — estimates, so: **(2e) a per-protocol meter, 2026-10-18, released in v0.3.243**:
`by_protocol` in the traffic payload counts every substream's bytes by the protocol it negotiated
and by which side opened it — our own lookups vs our answers to others' (`network::protocol_traffic`, read off the multistream-select line this node wrote), counted on the
same substreams as the total and adding up to it (tested over real TCP and QUIC swarms; on the live
node too: Σ rows = `out_bytes` and `in_bytes` to the byte). **Read 2026-10-18** on the live node,
26 min from a restart (startup included, so a ceiling): gossip 2.6 KB/s, **Kademlia answering other
nodes' lookups 2.1 KB/s** (896 answers, ~3.6 KB each — the whole peer list with addresses), our own
Kademlia 0.4, our direct messages 0.3, relay setup 0.15 but in 2 415 streams (~1.5 a second, to
look at), AutoNAT 84 KB once at startup (v2's dial-data deposit). ⚠ A stream this node opens
whose FIRST proposal the peer refuses is booked under the refused name: the `na` skip covers only
our own refusals (found by the v0.3.243 pre-release review). A label error only, the rows still
add up, and rare between nodes of one build; label from the proposal the peer echoed if a reading
ever shows a refused family. **(2f) 2026-10-18, released in v0.3.243**: provider records republish every 4 h, not 20 min, and live 12 h, not 1 h
(`network::behaviour::KAD_*`; still under the 26 h retraction window, checked at compile time) — in a
swarm smaller than k every republication is a lookup that asks every node, so the answering cost
falls ~12x once the fleet updates. Next: re-read `by_protocol` on the field after the release
(expect the `/swarmllm/kad/1.0.0` `peer` row to fall), then the relay-stream rate. **Re-read 10-18
15:20-16:16Z** (`~/swarmllm-91/out_243/r3,r4`, 56 min, 5 of 6 peers on .243, one on .242): upload 2716 B/s
(5.5 KB/s on .241) = gossip 1493 (message data 1304), kad `peer` 901 (2.1 → 0.59 in the first hour → 0.90),
relay hop 91, `/swarmllm/1.0.0` 190, our own direct messages 169 of it. The kad row fell 2.3x, not 12x: the
one .242 peer explains ~0.3 KB/s, so ~0.6 KB/s is lookups other than republication — the periodic
routing-table bootstrap (libp2p's default, every 5 min, untouched here) and `get_providers` are the
candidates, and nothing counted them apart. **(2g) 2026-10-19, released in v0.3.244**: `kad_inbound` in the
traffic payload counts every Kademlia request peers send, by kind (`find_node` = a routing-table
refresh or a publication's lookup, `get_provider` = a holder lookup, `add_provider` = a publication),
with the peers each answer listed (`KadInboundMeter`, the five kinds libp2p-metrics' `inbound_requests`
counts; DIAGNOSTICS § "What is the upload that is not gossip?"). Three idle nodes on loopback
(`unshare -rn`, 400 s): the node the other two joined answered 34 requests on exactly 34 Kademlia
streams, and each joiner's 5-minute routing-table refresh asked it 5 `find_node`. A swarm of N nodes
therefore sends each ~5(N-1) refresh lookups every 5 min, each answered with up to 20 peers and their
addresses — the first suspect; go-libp2p-kad-dht refreshes every 10 min and skips a bucket refreshed
within that interval (`RoutingTable.RefreshInterval`, `rtrefresh/rt_refresh_manager.go`). **Field
re-read 10-19 06:10-07:12Z** (`~/swarmllm-91/out_244/r1,r2`, 3714 s, the live node up 32 min at r1):
upload 2410 B/s = gossip 1340 (message data), kad `peer` 509 B/s on 8.3 streams a minute, relay hop
195, our direct messages 157. `kad_inbound`: 516 `find_node`, 0 `get_provider`, 0 `add_provider` —
the whole row is peers' refreshes, each answer ~3.7 KB listing 17 peers. **(2h) 2026-10-19, released in
v0.3.245**: libp2p's refresh runs every 10 min (`behaviour::KAD_REFRESH_SECS`, go-libp2p-kad-dht's
interval), and the discovery tick no longer starts one of its own. That call ran every ~295 s, just
inside libp2p's 300 s, so it only pushed libp2p's timer back: the old cadence was ONE refresh per
5 min, not two (each joiner's log, `kad_refresh_rig.sh` on v0.3.244). Rig (`examples/kad_refresh_rig.sh`,
three arms at once, 1503 s): v0.3.244 5 refreshes ~296 s apart, 27 `find_node` at the hub; this
build at `SWARMLLM_KAD_REFRESH_SECS=300` 5, 15; this build 2, 592 s apart, 6. Expected on the field
once peers update: about half the `find_node`, ~0.25 KB/s. Next: re-read `kad_inbound` after the
release; then the answers' SIZE — ~215 B per listed peer is several addresses each (unmeasured:
`dial_bootstrap_peers` hands Kademlia every cached address of a peer, and a routing entry's address
list has no cap in libp2p-kad 0.48), and the walk itself (go skips a bucket refreshed within the
interval; libp2p-kad refreshes every bucket past the nearest neighbour's each time). **(2c) fixed 2026-10-17, released in v0.3.242**: a restart gossiped every
manifest it holds in full, twice, a few seconds apart (§ Closed). Left: the
mesh is still sized once at startup, from peers ever met rather than reachable now (a rig first
before going further: a thinner mesh partitions sooner); (3) a both-direction traffic ceiling at the transport
(risky — throttling gossip can partition a node); `max_bandwidth_mbps` shapes part serving only.

#### #169 — Connection churn on multi-interface hosts: only the LAN dial is deterministic
`P3` · network · **PARTIAL** — 2026-07-25 · history: archive § "Connection churn on multi-interface hosts — deterministic dialer partial", gotcha #353

The re-dial fixes closed on 2026-08-02; the vendored `connection_rank` now picks "fewest
pending, then answered-before, then oldest", and an unacknowledged forward fails fast when a
standby exists. Open, trigger-gated: extend the deterministic-dialer discipline to
bootstrap / DHT / PEX dials only if a reproduction on distinct hosts shows a stale-route drop,
and optionally retry once on a different connection before failing over. The root cause of
the one-way-dead connection (suspected TCP port-reuse 4-tuple collision) is unproven. The
same-host churn the example scripts detect and name is environmental.

#### #195 — A node cannot test whether its LAN neighbours can reach it
`P3` · network · **PARTIAL** — 2026-08-18 · history: archive § "A node cannot test its own inbound reachability"

AutoNAT v2 answers the internet half, and `observed_inbound_connection` evidence shipped
(gotcha #335). `autonat_verdict` returns `Uninformative` for any private tested address, so a
node whose same-LAN neighbours cannot dial it (a WSL firewall, say) is only told "none has
connected" while a relay hides it at a latency cost. Extend the verdict with the server's
vantage (a probe server on our subnet makes a private address a real verdict); ask a LAN peer
only when the warning is about to fire; never feed the result to `try_activate_relay`. Worth
doing with pool onboarding (`pool::invite::any_internet_reachable` inspects addresses rather
than testing them).

#### #177 — Gossip has no peer scoring, and DHT provider records are never challenged
`P3` · security · **OPEN** — 2026-07-29 · absorbs archive § "Audit deferral — R128 sweep-log triage" (its architectural deferrals); history: archive § "Gossip has no peer scoring"

Strict signing, a 4 MiB cap, a freshness filter and bounded `foreign_*` maps exist, but a peer
flooding valid signed messages is never penalised or pruned (no `PeerScoreParams` anywhere).
Enable gossipsub peer scoring only after deriving parameters from measured delivery times on
a real multi-node swarm, with permissive thresholds and a per-peer score metric so a wrongly
penalised slow or CGNAT peer is visible — the risk is the mesh partitioning itself. Second
half: DHT provider records are add-only and can outlive the fact by up to their 12 h TTL (see #49); a
capability challenge before trusting one would address it. No incident is known.

#### #315 — Two key exchanges in flight to one peer: the first answer meets the second secret
`P4` · crypto · **OPEN** — 2026-10-20 · history: #304's security review (LOW, pre-existing)

`SessionManager::initiate_ephemeral_exchange` overwrites `pending_ephemeral` for the peer, and
`complete_ephemeral_session` does not match the answer's `session_id`, so two exchanges begun
within one round trip (a rotation or #304's re-key beside a `request_rekey` repair) combine the
FIRST answer with the SECOND secret: a key the peer does not hold is installed and the second
answer is dropped ("No pending ephemeral exchange"). It heals itself — the next unopenable message
asks for a repair — at the cost of one refused forward. Next: keep the session id with the pending
secret and drop an answer that names another (`EphemeralKeyExchange` carries a `session_id`;
check the answer echoes the initiator's), or skip a new exchange while one is pending for the peer.

#### #314 — The network event loop stalls 0.2-2.6 s, most often while this machine is busy
`P3` · network · **NARROWED** — 2026-10-20 · history: the live node's log, 10-19; `docs/invariants/network.md` § "The network loop never waits for the disk"

The live node (v0.3.244, serving nothing) logged 65 `DIAG: network event loop stalled` lines on
10-19 (190 ms-2.6 s: 43 in `swarm_event:RequestResponse`, 15 `Identify`, 7 `peer_cache_interval`),
bunched while this machine's disk was busy. Sampled across a full `cargo dev-test`, 6 of 8 stalls
had a runtime worker in `fdatasync` — a redb commit, which is the only caller of it. Fixed
(2026-10-20, released in v0.3.246): every commit hands its worker's queue off first
(`Database::with_write_table` → `without_starving_the_runtime`), so another task's commit no longer
holds the loop in that worker's LIFO slot, and the loop's own peer-cache save runs on the blocking
pool (`peer_cache::CacheWriter`). What remains: the 2 sampled stalls with no worker in a sync, and
whether the daily count falls. Next: once deployed, count the stall lines on the live node over a
day of ordinary use (10-19: 65) and sample the next one that comes while the disk is idle.
*Interim, v0.3.246, 03:27-06:50Z 10-21 (four restarts for rigs, no build on the disk):* 4 stalls,
every one the first `swarm_event:Identify` within 2 s of a start, 165-225 ms with 151-190 of it on
the processor — none in `RequestResponse`, none while running. The day's count is still owed.

A tester's node (field report 2026-10-20, v0.3.245, which lacks the fix) logged 42 stalls (median
289 ms, max 1.7 s; arms `RequestResponse`, `liveness_interval`, `Identify`) from the moment an
80 MB/s rsync to the same NVMe began, with no forward in flight, and none in the 26 h before.
Their question — is the loop doing synchronous disk I/O, log or redb? — has two answers: redb's
commit (fixed above), and the log file, which whichever thread logs writes itself
(`logging::write_to_file`: a `Mutex` and a `write` on the caller). On ext4 with ordered data a
small append can wait on the journal for 100-700 ms while another writer floods the disk (kernel
bugzilla 201461, a 6 MB/s logger; Ts'o: the ordered-mode writeback colliding with the flusher).
Measured here 2026-10-20 (ext4, `data=ordered`, an rsync of 13 GB of build files to the same disk,
`~/swarmllm-318/append_under_rsync.sh`): a 200-byte append every 5 ms took at most 0.2 ms at
80 MB/s and 77 ms unlimited (260 MB/s), never ≥ 190 ms; the same append followed by `fdatasync`
every 200 ms — a redb commit's wait — went ≥ 190 ms 6 times in 245 at 80 MB/s (max 808 ms) and 29
in 202 unlimited (max 1.35 s): the tester's distribution. So the log is not it, and the commit,
fixed above, is; no writer thread for the log. One thing neither explains: some of the tester's
stalls were in `liveness_interval`, an arm with no `.await` that only writes a map in memory, and
a commit elsewhere delays an arm only BEFORE it starts. So each stall line now says what the arm's
thread was doing (`network::manager::loop_stall`: time on and off the processor, page faults that
read the disk, waits, preemptions, and a `likely` reading). Next, besides the count above: the
first stall line from a node on this build under a large copy names its cause.

#### #191 — A part download that loses its peer restarts from byte 0 on the next one
`P3` · storage · **OPEN** — 2026-09-03 · history: archive § "A multi-megabyte forward over ONE QUIC stream can kill the connection" (its banner: the cause was quinn-proto, #112)

The connection-killing cause is fixed (#112, v0.3.206); the section's mitigations (TCP ranked
above QUIC by size, chunked forwards) are moot. What remains true: `handle_shard_transfer_retry`
(`network/manager/shard_transfer.rs`) retries with `chunk_offset: 0`, so a peer dying late in a
part re-fetches everything already received in 8 MiB chunks. Pass the offset recorded in
`shard_download_progress` instead (`handle_send_shard_request` already seeds the partial `.tmp`
from a non-zero offset). Check first that the replacement peer serves the SAME upload (near
certain since #151; the final BLAKE3 check is the backstop) and gotcha #424 (an unflushed-length
truncation race in exactly this code).

#### #168 — A relayed tensor forward over 32 MB is refused instead of chunked
`P4` · network · **OPEN** — 2026-07-24 · history: archive § "Tensor-relay large-forward chunking (post-plan follow-on, deferred)"

`crypto/relay_seal.rs` refuses above `MAX_RELAY_TENSOR_BYTES` (32 MiB). Only a pure
application-relay path with an uncompressed, very long prompt reaches it; no case has been
measured. Reuse `pipeline_stream::chunk_layer_forward` and the `pending_activation_chunks`
receiver on the relayed path (each chunk sealed separately) when a dropped relayed forward is
reported.

### Speed on one computer

Local GPU decode runs at ~97% of llama.cpp (`docs/plans/local_decode_submissions.md`); what
is left is prompt reading, the processor half of a hybrid, and MoE placement —
`docs/plans/faster_than_local.md` § 3 and § 4 carry the order.

#### #147 — Prompt reading on the card: the smaller items left after f16 accumulation
`P3` · perf-local · **PARTIAL** — 2026-10-09 · history: archive row #147 and § "Prompt reading on the card (#147, 2026-09-29)"

Shipped in v0.3.213: ≥ 64 rows go to f16 dequant + cuBLAS (llama.cpp's own rule for a dp4a MMQ)
and a prompt alone on the card reads 512 per forward — Llama-3.1-8B ~740 → ~1,150 tok/s. On main
since 2026-10-09: that path ACCUMULATES in f16 (llama.cpp's default; `SWARMLLM_QMATMUL_CUBLAS_ACC=32`
= all f32), after a card-side check at a 4,096-token context on eight families — median logit
cosine against llama.cpp moved ≤ 2e-5, every argmax the arms disagreed on a near-tie — with
llama.cpp's `GGML_PREC_F32` marks copied (`ModelArch::accumulates_outputs_in_f32`: GLM-4's attention
output and FFN down). `docs/invariants/inference.md` § "A prompt pass on the card multiplies
quantized weights on the tensor cores". Left, smaller (archive body): share the f16 activation
cast across q/k/v and gate/up (~1% of a chunk); remove the 8 `ucopy_f32` per layer in the prompt
pass (~5% by the kernel table); one untraced single-row `mul_mat_vec_q4_K` per layer; an int8
tensor-core MMQ port (large). Not checked at long context: Qwen 3.5 (reference is llama.cpp master)
and a MoE model whole on the card (none fits 8 GB). Separately, GLM-4-9B disagrees with llama.cpp
at a few positions on every path (tokenizer re-tokenisation is the lead) — pre-existing.

#### #258 — One long prompt in about ten pauses ~0.7 s in a single middle chunk on the card
`P3` · perf-local · **OPEN** — 2026-10-13 · history: none (seen while measuring #257)

Measured 2026-10-13 on the RTX 3070 Laptop, Windows up 1.4-1.8 h (`~/swarmllm-257/run{1,2}/`,
Qwen2.5-Coder-7B whole on the card, isolated node at `-v`, three unique ~4,800-token prompts
a round, `max_tokens` 1). Of 30 long prompts in ten rounds, three read at 1,200-1,264 tok/s
instead of ~1,450-1,580. In each, ONE 512-token chunk took 0.96-1.04 s against ~0.32 s for its
neighbours (positions 2,560 and 3,584; `DIAG: BatchGenerate prefill chunk ran` timestamps). The
card memory pool did not grow at that chunk, and there was one in each kind of round: snapshots
on the card, in host RAM, and the prefix cache OFF. So it is not #257's pool growth. It did not
line up with the safety kit's 10 s `nvidia-smi` readings. Next: catch one under nsys and see what the stream waits on (a
Windows-side poller, a clock change, a WDDM residency move).

#### #137 — Our processor decode trails llama.cpp by 3% at 4 threads and 5-8% at 8, both at the box's memory bandwidth
`P3` · perf-local · **PARTIAL — the 8-thread loss fixed 2026-10-14 (idle rayon workers spin); thread scaling level with llama.cpp's 2026-10-15** — 2026-09-27 · history: archive row #137; plan: `docs/plans/faster_than_local.md` § 3.3

**Measured 2026-10-15, ONE session** (HEAD `adf77d06` + the spin, Qwen2.5-Coder-7B Q4_K_M, processor
only, 256 tokens a run, ours and llama.cpp interleaved o4 l4 o8 l8 l8 o8 l4 o4, live node stopped,
safety kit, 0 driver events; `~/swarmllm-137/ll.sh`): ms a token — ours the mean of 251 one-position
forwards, llama.cpp `llama-bench -ngl 0 -p 0 -n 256 -r 2` (`build-cuda`, not repacked), per rep.

| ms a token | 4 threads | 8 threads |
|---|---|---|
| ours | 118.4 / 117.6 | 116.2 / 115.7 |
| llama.cpp | 114.3, 114.0 / 114.3, 114.5 | 110.7, 106.3 / 110.2, 106.4 |

**Both gain a little from 4 threads to 8** (ours 2%, llama.cpp 3-7%). The 2026-10-14 residual
("8 threads no faster than 4 where llama.cpp gains ~8%") set ours against llama.cpp figures from
another session — the box's two regimes (gotcha #845) — and dissolved once both ran in one. Every
product reads its weights at 38-41 GB/s at either width (FFN up + gate 50-54 ms for 2.14 GB, the
output head ~11 ms for 447 MB inside "unattributed"), which is llama.cpp's rate over its whole
token: this laptop's dual-channel DDR4 is the limit, and more threads cannot add bandwidth.

**Refuted 2026-10-15 — the one-row split (lead 2)**: one binary, `SWARMLLM_MATVEC` choosing how the
vendored m == 1 product shares its rows (scratchpad `mv.sh`, 12 runs, t4/t8 interleaved): rayon's
adaptive 128-512 rows (today) 117/116 at 4, 112/116 at 8; llama.cpp's 64-row chunks from a shared
counter 113/116, 115/113; upstream candle's one range a thread (huggingface/candle#4000) 112/116,
116/116. Stage times level in every arm, identical tokens in all 12. The switch was not kept.
Upstream candle replaced rayon with a persistent barrier pool for its quantized products
(huggingface/candle#3627: Qwen3-0.6B decode 97 → 110 tok/s, on a model small enough to sit in cache).

**Residual (the 3-8%)**: the products match llama.cpp's bandwidth, so the gap is what runs
between them — ~10 rayon regions a layer (280 a token) with a fork/join each, against llama.cpp's
one parallel region synced by `ggml_barrier`, plus the small ops (~3 ms). Next lead, only if a
processor-only user's speed rises in rank: a one-region decode for a layer.

**History — measured 2026-10-14, before the spin** on the Ryzen 7 5800H (8 cores), Qwen2.5-Coder-7B Q4_K_M, processor
only — ours `swarmllm test-split --gpu-layers 0 --max-tokens 48` at
`RAYON_NUM_THREADS = SWARMLLM_DECODE_THREADS = t`, the median `SWARMLLM_PROFILE=1` forward
total; llama.cpp `llama-bench -ngl 0 -p 0 -n 48` (`4b1a27f`). Live node stopped, safety kit,
60 s cool-down before each arm, arms interleaved (`cpu_threads.sh`, `thp.sh`, removed 2026-10-15 — superseded by `ll.sh`):

| ms a token | 4 threads | 8 threads |
|---|---|---|
| ours, v0.3.237 (3 runs) | 124 / 127 / 151 | 172 / 185 / 182 |
| ours, v0.3.236 (2 runs) | 154 / 159 | 201 / 194 |
| llama.cpp (median of 5 reps, `-o jsonl`) | 118.8 | 110.4 |
| llama.cpp (2 runs, CSV mean) | 120 / 119 | 123 / 126 |

- **At 4 threads, the width calibration picks on this box, we trail by ~5%** (124-127 against
  119). The 151-159 runs are this box's slow regime, not a slower release: one .237 run held
  151 ms for 40 tokens and then 127 to the end (gotcha #845, `docs/DIAGNOSTICS.md` § Benchmarks).
- **Going 4 → 8 threads costs us ~40%; llama.cpp moves −7% to +5%.** Calibration keeps 4 here,
  so this box loses nothing today, but a machine with the memory for more threads would stop
  scaling with us where it keeps going with llama.cpp — a processor-only user's speed.

**Refuted 2026-10-14 — do not chase again:**
- **llama.cpp's Q4_K repack** (the 10-11 lead). The 10-11 llama.cpp figures were never
  repacked: a CUDA build ranks the card's pinned host buffer above the repack buffers even at
  `-ngl 0` (gotcha #843). With it let through (`--no-host 1`, `CPU_REPACK model buffer size =
  2976.75 MiB`) llama.cpp ran 117-122 ms at 4 threads (no gain) and 137-144 at 8 (a loss).
- **Huge pages.** Our weights already sit on transparent huge pages (`AnonHugePages` 4.2-4.6 GB
  in the measured process; THP is `madvise`). `MIMALLOC_ALLOW_LARGE_OS_PAGES=1` changed nothing
  at 4 threads (150 / 127 / 129) and was no better at 8 (187 / 193 / 206) (gotcha #844).

**Leads for the 8-thread loss (10-14)** — 1 fixed that day, 2 refuted 2026-10-15 (above), 3 measured:
1. **Every parallel op is a rayon region whose workers may have gone to sleep.** rayon-core
   1.13 parks a worker after 32 idle rounds (`ROUNDS_UNTIL_SLEEPY`); a token runs several
   regions a layer × 28 layers, each waking the workers it needs, and 8 workers pay more
   wake-ups. llama.cpp runs the whole graph in ONE parallel region and syncs its ops with a spin
   barrier (`ggml_barrier`).
2. **The one-row matvec splits a small projection coarsely**: the vendored `k_quants.rs` matmul
   (m == 1) uses `with_min_len(128).with_max_len(512)`, so a 512-row K or V projection gets at
   most 4 tasks at any width.
3. ~~**First find which stage grows**~~ — **measured 2026-10-14** (`SWARMLLM_PROFILE=1`, 256
   tokens a run, t4 / t8 / t8 / t4 interleaved, live node stopped, safety kit, 0 driver events;
   scratchpad `t48.sh`; median ms a token over each run's 251 one-position forwards, the two runs
   of each width averaged): **the SMALL ops grow, the big products barely do.**

   | stage | 4 threads | 8 threads | × |
   |---|---|---|---|
   | FFN up + gate | 57.8 | 63.2 | 1.09 |
   | FFN down | 38.9 | 46.4 | 1.19 |
   | QKV projections | 10.0 | 22.3 | 2.2 |
   | output projection | 5.7 | 9.8 | 1.7 |
   | rope / transpose / q-k norm | 0.7 | 9.5 | 13 |
   | attention scores + softmax + AV | 1.7 | 9.4 | 5.7 |
   | unattributed | 12.2 | 12.9 | 1.06 |
   | **total** | **128** (131, 125) | **174** (175, 173) | 1.36 |

   33 of the 46 ms are QKV, rope, attention and the output projection — a per-op cost that grows
   with the thread count, which a bandwidth limit would not do (the FFN products read 2/3 of the
   bytes and gained 13 ms). That points at lead 1 (wake-ups / fork-join per op) or at idle
   workers of the OTHER pool contending for the 8 cores' SMT siblings (`RAYON_NUM_THREADS` and
   `SWARMLLM_DECODE_THREADS` both 8 → 16 threads on 8 cores, 16 logical), and away from lead 2 alone
   (QKV doubles, but rope ×13 has no matvec in it).

   **Refuted the same day** (scratchpad `t48b.sh`, `t48c.sh`; each arm in its own scope, its
   cgroup's `cpu.stat` read): **the rig's CPU cap** — the safety kit's `CPUQuota=1000%` throttled
   16-26 of ~490 periods (mostly the load), but an uncapped 8-thread run (0 throttled) takes the
   same 172-177 ms; **SMT placement** — 8 threads pinned one per physical core
   (`taskset -c 0,2,…,14`) 171-172 against 173-177 unpinned, rope and attention still ~9 ms each.
   ⚠ **A decode pool is never wider than the global pool** (`cpu_pools`: "no wider than prompt
   processing's"), so `RAYON_NUM_THREADS=4 SWARMLLM_DECODE_THREADS=8` decodes at 4 (125-127 ms) —
   read `decode_threads=` in the log before trusting such an arm.

   **Lead 1 was it, and is fixed (2026-10-14, released in v0.3.239)**: upstream rayon puts an idle worker
   to sleep within microseconds, so each small region of a token paid a wake per worker — 0.75 µs
   of RoPE work cost 230-270 µs at 8 workers after a 20-100 µs gap. The vendored `rayon-core` spins
   200 µs first: **4 threads 126/127 → 121/122 ms, 8 threads 172/171 → 123/127**, replies identical,
   less processor time in all (`docs/invariants/inference.md` § "Idle rayon workers spin 200 µs
   before they sleep"; A/B `SWARMLLM_RAYON_SPIN_US=0`).

*(The 10-14 residual — "8 threads still do not beat 4, where llama.cpp's do (110 against 119)" —
compared sessions; superseded by the one-session table at the top.)*

History: on 2026-10-11 decode ran OUTSIDE its thread pool on every default node — fixed
(§ Closed 2026-10-11): a 14B split 5.3 → 6.8-6.9 tok/s, level with llama.cpp; a 3B
processor-only 1.14-1.39x. That day's 14B processor-only gap (~312 against 278 ms at 4 threads)
was measured against the never-repacked reference above. Not ours, and llama.cpp's too: the card
clocks down between steps of a split (255-600 MHz between 1755 MHz bursts), and back-to-back
runs on this laptop heat it enough to slow a split step 133 → 185 ms within 40 tokens (gotcha #825).

#### #295 — Gemma 4's E2B/E4B split between card and processor copies a shared layer's history every step
`P3` · perf-local · **PARTIAL** — 2026-10-17 · history: #115 (3c); (a) captured, (b) launches fixed, 2026-10-19

(a) ~~not captured as a CUDA graph~~ and (b) ~~each group launch waiting on graph memory~~ — **both
closed 2026-10-19** (§ Closed): whole on the card the E2B decodes 44.6-49.1 → 92.9-93.3 tokens a
second (llama.cpp 114.6, 0.81 — the same per-step gap as the other small models, #309) and the
E4B 33.9-39.9 → 59.8-60.4. What is left: a card/processor split of one (a card under ~3 GB since
#296) copies each shared layer's source history to the processor every step where the cut parts
them (`layers::history_beside_queries`; unmeasured, ~50 MB a step at 4K positions by the shapes):
cutting at the first source (the E2B's 13) would trade it for up to 21 more processor layers, so
measure both before choosing.

#### #308 — An unquantized model is held on a card at twice its size, and decodes at half llama.cpp's rate
`P3` · perf-local · **OPEN** — 2026-10-19 · history: gotcha #288

`candle_core::quantized::QMatMul::from_arc` turns every F16, BF16 and F32 weight into a dense f32
tensor at load, on a card as on the processor, so an unquantized upload holds twice its file on the
card (the estimator knows: `vram::unquantized_bytes_per_element`) and every decode step reads twice
its bytes. Qwen2.5-0.5B-Instruct fp16 whole on the RTX 3070 Laptop (2026-10-19 sweep, `test-split`
after the first 8 steps): 107-116 tokens a second against llama.cpp's 227.6 (median sample), 0.49 —
the worst of 16 families; its tied 151,936-row output head alone is 545 MB at f32. llama.cpp keeps
the weights at f16 and multiplies one to eight rows in `mul_mat_vec_f` (`ggml-cuda/mmvf.cu`, master
4b1a27f: f16 pairs widened to f32 and summed in f32 — or in `half2` on a card with fast f16 at the
default precision), more rows through cuBLAS at f16. Not measured: how much of the gap is the bytes
and how much cuBLAS's one-row sgemm. Next: keep an F16/BF16 weight quantized-shaped on a card (the
vendored `QCudaStorage`, like a Q-type), a one-to-eight-row kernel summing in f32 for decode and the
existing `mul_mat_via_f16_cublas` (no dequantize step) above that; the estimator's per-device bytes;
`forward_shared`, `FusedSlice`, LoRA and the prompt-card copy each checked against an f16 weight; a
replies-against-llama.cpp check (`score_against_reference.py`), since a prompt pass would then round
activations to f16 as llama.cpp's does. Who it helps: anyone who downloads a repo's "F16" file of a
small model — the drafter choice already prefers a quantized copy (gotcha #849). Beside a split the
cost is card room too: the 0.5B fp16 drafter is charged 2,991 MB, so on a 4,000 MB card the room it
left was 4 of the 7B's layers, short of the 6 this node had to hold, and no plan took the cut
(2026-10-22, `cardfill275.sh`; the Q8_0 copy, 966 MB, left 16).

#### #309 — Small dense models decoded at 0.80-0.83 of llama.cpp on a card; 0.94-1.00 since the attention chunk follows the cache, Gemma 4 E2B 0.87
`P3` · perf-local · **PARTIAL** — 2026-10-19 · history: #307's sweep

**Residual (2026-10-19):** Gemma 4 E2B decodes at 0.87 of llama.cpp (108-109 against 124.8 tok/s,
same session), TinyLlama and Llama 3.2 1B at 0.95-0.97. Next: an nsys kernel sum of an E2B step
against llama.cpp's (`docs/DIAGNOSTICS.md` step 3e) to name where its remaining 1.2 ms a step goes;
then, if decode attention is still above llama.cpp's on the small models, an `mma` body for GQA ≥ 2
on sm_80+ (below). The history follows.

Whole on the RTX 3070 Laptop (2026-10-19, `test-split` after the first 8 steps, live node stopped;
llama.cpp `llama-bench -ngl 99 -n 128`, median sample): TinyLlama 1.1B Q4_K_M 236-241 against 299.2
(0.80), Gemma 3 1B Q4_K_M 162-166 against 201.3 (0.82), Gemma 4 E2B 92.9-93.3 against 114.6
(0.81, since #295 (b)). Every 3B-and-larger family reads 0.84-0.97
(`docs/DIAGNOSTICS.md` § "One model's decode rate on this machine" has the table), so what is left
is a per-step cost that a small model's short step cannot hide: ~4.2 ms against ~3.3 ms on
TinyLlama, with every step captured as CUDA graphs. **It is on the card, not the host**
(2026-10-19, `SWARMLLM_PROFILE` with graphs on, whose `total` is then the host's recording time;
median of a 60-token reply's last steps): TinyLlama host 2.0 of a 4.29 ms step, Gemma 3 1B 3.0 of
6.08, Qwen 3.5 0.8B 5.0 of 7.07, Llama 3.2 1B 2.0 of 5.71, Gemma 4 E2B 6.0 of 10.45; graph launches
0.1-0.5 ms a step on every family (so none has #295 (b)'s defect). Kernel counts are near
llama.cpp's by construction (Gemma 3 1B 26 a layer: 7 quantized matmuls, 4 q8_1 quantizes, 6
norms, rope and cache writes twice, decode attention and its merge, GELU × up), so the difference
is per-kernel time. **It is decode attention** (nsys does record kernels here once configured,
`docs/DIAGNOSTICS.md` step 3e): TinyLlama's kernels sum to 3,501 µs a step against llama.cpp's
2,912, and decode attention alone is 778 against 173 (35 µs a layer against 6-9 at ~40 positions)
— the whole gap. The matrix-vector products are already faster than llama.cpp's (nsys, and
`examples/qmatvec_graph_bench.rs`, each product timed inside one CUDA graph).

**Tried and not shipped (2026-10-19): keeping more loads in flight inside one block.** The guess was
that a short cache is a few blocks (TinyLlama: 4 at 40 positions) each waiting on memory at every
position. A body that loaded the K rows of 4 positions before using any, ran a position's
reductions side by side and kept 16 V rows in flight was bit-identical to the current one (a
deliberately broken copy failed the test, 1,405 of 2,048 values differing), used 94 registers
against 170 with no spills — and was SLOWER where it was aimed. 160-token replies, both bodies
alternated twice in one binary, Windows up 2.2 h: TinyLlama 229-232 tok/s against 256-260, Gemma 3
1B 146 against 180-184, Gemma 2 2B 95 against 118, Llama 3.2 3B 105 against 118, Qwen 3.5 0.8B
level; faster only behind a 1,700-token prompt (214-215 against 201-202). Replies identical. So one
block's memory latency is not what costs the time at short lengths.

**What llama.cpp runs instead (2026-10-19)**: not its vector kernel. The same nsys trace names
`flash_attn_ext_f16` (6.32 µs a layer) plus `flash_attn_stream_k_fixup_uniform` (1.54 µs): on
Ampere with an f16 cache and a GQA ratio ≥ 2, `ggml_cuda_get_best_fattn_kernel` (`fattn.cu`) picks
the tensor-core MMA kernel — a KV head's query heads become the rows of one `mma` product — and
stream-K spreads the cache over the SMs; the vector kernel is for Ada and later, or no GQA. Ours is
scalar f32 math in 4 blocks at 40 positions (one per KV head, 64-position chunks) on a 40-SM card.
The vendored flash kernel is no shortcut: it has no split-KV kernel, so one decode position cannot
fill the card (up to 25x slower per call, `docs/invariants/inference.md`). Next, cheapest first:
more blocks at short lengths (a smaller chunk, or a KV head's query heads split across blocks),
measured by nsys kernel sums AND tokens a second; then an `mma` body for GQA ≥ 2 on sm_80+.

**Done (2026-10-19): the chunk follows the cache length** (`decode_attn::chunk_for`: the smallest
of 16, 32 and 64 positions keeping the grid within 256 blocks and 64 chunks; the chunk is a kernel
argument, so a captured step only updates as it grows). Smaller chunks win at short caches
(TinyLlama at ~200 positions: 39.8 → 12.0 µs a layer) and lose at long ones, where each chunk
costs the merge ~0.45 µs (Llama 3.2 3B at ~4,000 positions: 253 µs in 64s, 294 in 32s). Against
the fixed 64, alternated in one binary, Windows up 3.8-4.1 h, llama.cpp re-measured the same
session: TinyLlama 297-304 tok/s against 252-258 (llama.cpp 312.1), Gemma 3 1B 196-209 against
175-187 over four prompts (209.7), Gemma 2 2B 126-127 against 118-120 (128.0), Llama 3.2 3B 121-122
against 116-117 (126.1), Llama 3.2 1B 197-202 against 187-189 (208.8), Gemma 4 E2B 108-109 against
101-104 (124.8); level behind a 4,320-token prompt, where the rule picks 64. Replies score against
llama.cpp as the fixed chunk's did, no graph was rebuilt in 47 runs (a positive control logged
one) — `docs/invariants/inference.md` § "Its chunk follows the cache length".

#### #293 — A split's first prompt chunk after load spends 1-6 s in its first attention, once its experts go to the card
`P3` · speed · **OPEN** — 2026-10-16 · history: found measuring #292

Gemma 4 26B-A4B Q2_K, 30 layers on the card, a 1,527-token prompt in 512-token chunks (Windows up
87 h, `~/swarmllm-g4/s292b/p_*.log`, `SWARMLLM_PROFILE=1`): the first chunk's "attention scores +
softmax + AV" took 1.2-6.0 s in 11 of 12 runs whose RAM-held experts were read on the card (once
38 ms), 41-196 ms in all 4 runs whose processor read them, and ~3 ms in every later chunk. A nsys
trace of one card run showed `cudaLaunchKernel` taking 2,972 ms over its first launches. Not
established: the cause. The guess is a first-use allocation outside the pool (cuBLAS workspace,
a lazily loaded module — `CUDA_MODULE_LOADING=LAZY` is the default) meeting #292's WSL2 stall after
pageable copies, which would make it a once-a-worker cost on any card model under WSL2. Next:
nsys a card run and name the call; try `CUDA_MODULE_LOADING=EAGER` (and its memory) or one warm
prompt-shaped forward at load; check a dense model on the card.

#### #138 — A model that does not fit the card: MoE decode level with llama.cpp since 2026-10-11, a split's prompt read on the card since 2026-10-13, a 512-token forward 1.1-1.3x llama.cpp's
`P2` · perf-local · **PARTIAL** — 2026-09-27 · history: archive row #138; plan: `docs/plans/faster_than_local.md` § 3.2

**Measured 2026-10-11** on the RTX 3070 Laptop 8 GB + Ryzen 7 5800H, Qwen3-30B-A3B-Instruct-2507
Q2_K (11.3 GB: 48 layers, 128 experts, top-8 — the shape of Qwen3.5-35B-A3B), llama.cpp
`4b1a27f` `llama-bench -t 4 -lm none -p 512 -n 64`, live node stopped, safety kit, cooled
between arms (`~/swarmllm-138/{llama_moe2,ours_moe,moe_prof}.sh`):

- **What placement buys llama.cpp**: ~6.5 GB on the card either way — 29 whole layers 23.4-23.7
  tok/s; every layer's attention on the card and the experts of 20 layers in RAM
  (`-ngl 99 -ncmoe 20`) 32.3 (1.37x); every expert in RAM 13.9. Prompt speed about equal (537 vs 571).
  The 32-44 tok/s figures quoted from others were 8 GB DESKTOP cards.
- **Where we stand, same 26 whole layers** (`-ngl 27`; 29 does not fit ours — `test-split`
  reserves the 8,192-token cache): llama.cpp tg64 25.4 ± 9.2 and 21.5 ± 2.1, pp512 496-499; ours
  a median decode step of 77-100 ms (10-13 tok/s), the reply identical in every arm.
- **Our step, by the new profile stages** (`SWARMLLM_PROFILE=1`, `moe ffn on the card` /
  `on the processor`, also with `SWARMLLM_PROFILE_SYNC=1`): the card's 26 MoE layers take
  40-42 ms (~1.6 ms a layer, for ~12 MB of experts a ~400 GB/s card reads in ~30 µs — host
  overhead: a device→host copy of the router scores, ~80 launches and ~16 host→device copies
  per layer); the processor's 22 took 22-28 ms; attention, norms and RoPE ~15 ms.

**Done 2026-10-11 (released in v0.3.235):** the processor's experts for ≤ 8 tokens run as (token,
expert) pairs in ONE parallel region per projection (`MoeFfn::experts_in_one_region` →
vendored `quantized::matmul_many`, llama.cpp's `mul_mat_id` shape), bit-identical to the
per-expert path: processor stage 22.1 / 28.3 / 24.9 → 19.2 / 16.5 / 17.8 ms (−29%, ABBA + a
synced pair, one binary, `SWARMLLM_MOE_CPU=per-expert` = old).

**Done 2026-10-11 (released in v0.3.235), the card:** a card keeps a layer's experts STACKED as the
file holds them, routes on the card and runs each projection for every chosen pair in ONE
launch (`MoeFfn::experts_stacked`, candle-nn `moe_gemm_gguf`: one-row kernel ≤ 8 tokens,
tensor-core kernel for a prompt). Decode step 74-89 → 40-55 ms, ~23 tok/s — level with
llama.cpp's 21.5-25.4 at the same 26 layers; `moe ffn on the card` 43-52 → 5.5-7.9 ms
(synced 47.5 → 13.2); logits as close to llama.cpp as the per-expert path. It found two
races: candle's tensor-core prompt kernel lacked a barrier (replies changed run to run —
vendored candle-kernels with the one line, gotcha #827), and a card-to-card `to_device` was
unordered (gotcha #828). A/B `SWARMLLM_MOE_CARD=per-expert`, `SWARMLLM_MOE_CARD_PROMPT=rows`.

**Done 2026-10-11 (released in v0.3.235), a prompt on the processor:** the one-region path takes every
token count — vendored `quantized::matmul_many` with several activation rows per job, each
64-row task applying one expert's weight rows to ALL of that expert's tokens (`vec_dot_rows`,
`k_quants::matmul`'s tiling), `MOE_PAIRS_PER_REGION` pairs a region; still bit-identical.
A 681-token prompt read 128 tokens a forward, as a node reads it (`test-split --prompt-chunk
128`): 46.2-46.6 → 12.9-13.3 s split 26/22, 93.5-94.9 → 19.9-20.0 s with every layer on the
processor (its experts 84-85 → 11 s); in one forward 19.5-19.9 → 9.9-10.3 s. ABBA, one
binary, replies identical (`docs/invariants/inference.md`).

**Done 2026-10-13, (1) — a split's processor-held experts read a prompt ON THE CARD, as
llama.cpp runs them:** llama.cpp hands any `MUL_MAT_ID` of ≥ 32 tokens whose weights sit in RAM
to the card (`ggml_backend_cuda_device_offload_op`, `GGML_OP_OFFLOAD_MIN_BATCH` = 32,
`ggml-cuda.cu:5615-5634`) and copies only the experts the ids use (`ggml-backend.cpp:1693-1770`,
read at `4b1a27f`). Ours: such a forward is routed on the processor, the chosen experts are
copied into a zeroed stack on the card (`QTensor::stack_some`) and run by the card's own
kernel (`MoeFfn::processor_experts_on_the_card` → `stacked_products`); a refusal runs the
processor path. A split whose every processor layer does this reads 512 tokens a forward
(`SplitModel::reads_prompts_on_card` → `prompt_chunk_ceiling`; at 128 it is four times the
copies). Qwen3-30B-A3B Q2_K split 26/22, an 867-token prompt, ABBA, one binary
(`SWARMLLM_MOE_PROMPT_CARD=0` = old), Windows up 3.1 h, `~/swarmllm-138/ab.sh`:

| | 128 a forward | 512 a forward |
|---|---|---|
| processor's experts (old) | 11.00 / 11.07 s (experts 4.5 s) | 9.23 / 9.39 s (experts 4.1 s) |
| on the card (new) | 12.69 / 12.90 s (experts 6.1-6.2 s) | **7.80 / 7.92 s** (experts 2.7 s) |

What a node did before is the top-left (128 is the default ceiling off the card); what it does
now is the bottom-right: **11.0 → 7.9 s, 1.4x** (~110 tok/s; llama.cpp 537-571). As close to
llama.cpp as the processor path (`logits_reference_probe`, the same prompt read 512 a chunk, its
last 64 positions + 4 decode steps, `~/swarmllm-138/probe.sh`): median cosine 0.99952 vs 0.99955,
worst 0.9653 vs 0.9683 at the SAME position, top-1 65/68 vs 64/68; the replies part at a
three-way near-tie (0.105 logits, llama.cpp's own pick being the old path's).

**Done 2026-10-13, (1b) — every processor-held projection reads a prompt ON THE CARD:**
llama.cpp's offload is not MoE-specific — pass "1.off" (`ggml-backend.cpp:970-975`) sends ANY
op whose weight is in host memory to the card at ≥ 32 rows (`get_op_batch_size`: `ne[1]` for
`MUL_MAT`); `llama-bench -nopo 1` on Qwen2.5-14B Q4_K_M split 20/28 drops pp512 from 399 to
19.1 tok/s, and we read at 22. Ours: `QMatMul::forward` / `forward_shared` (the one choke
point) copy a marked weight to the card for a forward of ≥ `PROMPT_ON_CARD_MIN_TOKENS`
(`QMatMul::on_the_card`; a fused weight once, its f32 mark with it), every layer kind's
projections marked once (`let_processor_layers_read_prompts_on`), host-held weights only.
One binary (`SWARMLLM_PROMPT_CARD_PROJECTIONS=0` = old), Windows up 4.3 h,
`~/swarmllm-138b/ab.sh`: the 14B's 888-token prompt 35.3-36.8 s → **6.5-7.0 s** at 512 a
forward (11.3 s at 128); the 30B above 8.43 / 8.45 → 6.13 / 7.17 s. Judged against
llama.cpp (teacher-forced): the 14B's new 16-token reply is llama.cpp's first pick at every
token, both old paths part at a near-tie; the 30B's is as close as (1)'s
(`docs/invariants/inference.md` § "A split's processor-held projections read a prompt on the
card").

**Done 2026-10-13, (1c) — a processor-held layer reads a prompt chunk WHOLE on the card:**
llama.cpp's scheduler puts every op between two offloaded ones on the card too (pass 2's
"expand gpu down/up", `ggml-backend.cpp` 1129-1170) and copies the cache views up (pass 5),
only the K/V writes staying on the host. Ours: `split::executor::dense_layer_on_the_card` runs a
processor-held standard attention layer of a ≥ 32-token chunk on the card (`LayerOnTheCard`:
norms, biases, router and RoPE rows copied over, the projections copying their own weights),
its cache kept on the processor — new rows down, history up (`LayerWeights::attend`'s
`cache_home`); any error cuts the cache back and reruns the layer on the processor. One binary
(`SWARMLLM_PROMPT_CARD_LAYERS=0` = (1b)), ABBA, `~/swarmllm-138c/ab.sh`: the 14B's 888-token
prompt 8.18 / 8.24 → **5.21 / 5.17 s** (1.6x; the box read (1b) slower than its morning's
6.5-7.0), the 30B's 6.74 / 7.12 → 6.21 / 5.22 s (`docs/invariants/inference.md` § "A split's
processor-held layers read a prompt whole on the card"). The 14B's 512-token forward is now
2.0-2.8 s (the box varies) against llama.cpp's ~1.28.

**Done 2026-10-13, (1d) — a processor-held weight crosses to the card WHILE the card computes:**
`CudaDevice::clone_htod_beside` (vendored candle) copies on a second stream into a 256 MB
ring; the device's stream waits for that copy, moves it card-to-card into a buffer of its own
pool and marks the region read before the ring reuses it. Reached through `layers::sent_to_card`
and `QMatMul::weight_on` only. Two simpler designs grew the card's pool without bound — this
driver reuses another stream's free only after a SYNCHRONIZE (gotcha #836: 768 MB for 48 × 16
MB). One binary (`SWARMLLM_PROMPT_CARD_COPY_STREAM=0` = before), ABBA, Windows up 10.8 h,
`~/swarmllm-138c/ab1e.sh`: the 14B's 888-token prompt 3.27 / 3.56 → **3.15 / 3.07 s** at 512 a
forward and 8.71 / 8.71 → **7.91 / 7.90 s** at 128; the 30B level (5.45 / 5.73 → 5.69 / 5.53 s,
its experts still copy the waiting way); +256 MB of card, replies identical in every pair
(`docs/invariants/inference.md` § "A split's prompt copies run beside the card's work").

**Done 2026-10-13, (1e) — a split reads a prompt in chunks as large as the card has room
for:** every forward re-copies every processor-held weight (4.6 GB for the 14B), so fewer
forwards win. `SplitModel::prompt_reader(index_pos)` answers the ceiling for every prompt pass:
512-4096 tokens from the card's free memory (after a synchronize, less the budget margin) over
the forward PREDICTED from the last one MEASURED (`begin_prompt_measure` /
`end_prompt_measure` on the pool's high-water mark; `MeasuredPrompt::predicted_bytes` adds the
largest copying layer's activations per token, `layers::prompt_bytes_per_token`, the card's
caches, the history a card-run processor layer pulls up and the causal mask). 512 until
something is measured, after any card fallback, or under forced standard attention. One binary
(`SWARMLLM_PROMPT_CHUNK_ROOM=0` = before), ABBA, Windows up 12.3 h, `~/swarmllm-138c/ab1h.sh`,
`test-split --prompt-chunk 128` through the node's own chunk loop: the 14B 20/28's 2,606-token
prompt 10.13 / 8.77 → **5.92 / 5.15 s** (two forwards, 512 + 2,094), 4,324 tokens 14.55 →
**9.48 s**; the card at most 6.9 GB of 8, replies llama.cpp's first choice at every pick (2,606)
or a near-tie in both arms (4,324). The 30B 26/22 had no room and read 512 in both arms
(`docs/invariants/inference.md` § "A split's prompt chunk is as large as the card has room
for"). Its every arm, both binaries, was refused part-way by the card's conversation budget:
#262, closed 2026-10-13 (the budget was charged the processor-held layers' caches too).

**Done 2026-10-13, (3) stage A — a card-held layer can keep its experts in RAM:** llama.cpp's
`--n-cpu-moe N` keeps the routed experts of the first N layers in host memory and every other
tensor on the card (`llm_add_n_cpu_ffn_overrides`, `common/arg.cpp` at `4b1a27f`).
`hybrid::LayerPlacement::with_experts_on_the_processor` says which layers; `load_moe_ffn`
takes the placement, not a device, so every family's loop loads experts where it says; the
loader's card budget takes those experts' bytes off. The forward was already shaped for it:
routed on the card, the activation and scores come down, the processor runs the experts in one
region, the answer goes back; a prompt of ≥ 32 copies the chosen experts to the card (1). A
segment with such a layer works on the processor (`SplitModel::cpu_layer_count`, the decode
pool's key) and reads prompts through `CardCopyingWeights`. Reached by `test-split
--cpu-moe-layers N` here; the daemon chooses it since stage B (below). Qwen3-30B-A3B
Q2_K at the SAME ~5.9 GB of layers on the card, 867-token prompt read 128 a forward, 64 reply
tokens, ABBA, one binary, Windows up 16.3 h (`~/swarmllm-138d/ab3.sh`): 26 whole layers a
median decode step of 49 / 68 ms, every layer on the card with 23 layers' experts in RAM **40 /
40 ms** (~25 tok/s, 1.2-1.7x); prompt 4.96 / 4.96 s against 5.07 / 4.90. llama.cpp the same day:
`-ngl 27` 25.0 ± 3.9 tok/s, `-ngl 99 -ncmoe 23` 34.5 ± 7.9 (1.38x), `-ncmoe 20` 41.4 ± 15.6 —
we run this placement at ~72% of its speed. Replies (`~/swarmllm-138d/score.py`, teacher-forced
through llama-cpp-python 0.3.16): both arms leave llama.cpp's first choice at the same two
tokens (7: 0.76, 11: 0.105, our numerics either way); where they part (token 25) the new
placement takes llama.cpp's first choice and 26 whole layers one 0.193 behind it.

**Done 2026-10-14, (3) stage B — the daemon keeps experts in RAM before it splits by layer:**
llama.cpp's `--fit` first asks whether a model fits with only its dense weights on the card,
then overflows the fewest layers' experts. `process_pool::card_split` does the same on
`split_for_card`'s terms (every layer's cache at the full context, #104): the fewest first
layers' experts in RAM (`experts_in_ram_for_card`, bytes from `split::routed_expert_bytes` in
the footprint's own header read), else the whole-layer split. The worker takes `--gpu-layers -1
--cpu-moe-layers K`; the card is charged the rest, the machine those experts (`RamShare::Experts`);
the planner, `/v1/status`, the models list and the dashboard say so (`expert_layers_in_ram`).
A/B `SWARMLLM_EXPERTS_IN_RAM=0`. Through the daemon, ABBA, Windows up ~18-19 h, profiled
(`~/swarmllm-138e/verify3.sh`): 48 layers on the card with 25 layers' experts in RAM against 26
whole layers — card 6262 MiB against 6678-6966, RAM 5213 MB against 5530, median decode step
**50 / 54 ms** against 55 / 55, replies 2-3 **20.2-22.4 / 18.7-19.5 tok/s** against 17.2-18.8
(`docs/invariants/inference.md` § "The pool keeps experts in RAM before it splits by layer").
Its first rig found #266: at the default contribution level a 16 GB node's RAM budget refused the
model under either placement — closed the same day (#266).

**Left, in order of what they cost us:** (1f) the warm forward is not yet bound by its copies: profiled, copies
833 ms + unattributed 528 (before: 1,215 + 248). This card has ONE async copy engine, so the
ring's card-to-card move queues behind the next upload — measured not to matter alone
(`~/swarmllm-138c/reuse.cu` variant 7, the move as a kernel: 296-298 ms against 282-307).
Then the experts: the MoE stage costs ~57-62 ms a layer whatever the tokens (1.25 s for 22 layers at 355
tokens, 1.49 s at 512), ~34 ms of it ~226 MB at this card's LINK speed, and `QTensor::stack_some`
still copies on the device's stream, waiting for its queued work. Measured 2026-10-13
(`~/swarmllm-138b/bench.sh`, a CUDA micro-bench, live node stopped, min of 10): the link tops
out at 6.7 GB/s even from pinned memory, and one copy per expert from pageable memory already
moves 6.5 (0.58 MB pieces; 4.7 at 0.15 MB); gathering into a pinned staging buffer first is
SLOWER, 3.4. So llama.cpp's pinned RAM weights (`llama-model.cpp:1090-1104`) buy nothing
here — what could: the experts' copies through the ring too (≥ 32 tokens at top-8 of 128
touch 87-100% of the experts anyway) and the tensor-core kernel's own speed. (1g) one
layer's copies (~200 MB here) and the 256 MB ring are a transient admission does not charge —
a full card falls back to the processor (WARN once); under WSL2 a pool grown after an idle trim
may be slow at high uptime (#257), measured at 3.1 h only. (2) ~~**MoE layers in decode-step
capture**~~ — **captured since 2026-10-14** where the experts are stacked on the card
(`ffn_capture_ineligibility`): Qwen3-30B-A3B Q2_K's first 24 layers cut to fit the 8 GB card
whole (`~/swarmllm-138f/moegraph2.sh`, 6,000 tokens, two interleaved rounds) gave the same ids
with capture off, on and one layer a group, 0 refused, and 14.8-15.5 → 12.2-12.7 ms a token
(64-68 → 79-82 tok/s; `docs/invariants/inference.md` § "A mixture-of-experts layer stacked on
the card is captured too"). A 26/22 split stays uncaptured regardless. (3) **the gap to llama.cpp at
the experts-in-RAM placement** (40 vs 29 ms in stage A's rig): the processor's 23-25 layers of
experts take 21.6-31.8 ms of a step at the default 4 decode threads. **Attributed 2026-10-14**
(`~/swarmllm-138f/prof.sh`, `test-split --cpu-moe-layers 25`, 64 tokens, median of 60 steps; the
new profile sub-stage `of which: the processor's expert products` splits the products from the
routing, the copies and the wait around them): unsynced 39-40 ms a step at 4 threads, 37 at 8;
the products **19.2-19.4 ms at 4 threads, 14.5 at 8** — level with llama.cpp's ~0.8 ms a layer —
and the rest of the processor stage 5.2 ms (the copies, the top-k, the combine, and the wait for
the card's queued work at the copy). The card stages cost ~15 ms of HOST time unsynced: kernels
enqueued one by one, because a step with experts in RAM is never captured
(`decode_graph_ineligibility`: MoE). Synced (`SWARMLLM_PROFILE_SYNC=1`) 58 / 54 ms, products
16.1 / 15.5. llama.cpp 4b1a27f at the same placement, `-t 4`: tg64 32.57 ± 2.58 (30.7 ms); its
CUDA graphs off vs on (`GGML_CUDA_DISABLE_GRAPHS=1`, 3 interleaved rounds, `graphs.sh`) 26.5 / 30.0
/ 30.4 against 37.0 / 36.0 / 29.7 tok/s, ± up to 16 on a busy host — leaning, not decisive.
**Tried 2026-10-14, not shipped: the card's part of such a step captured in groups cut at
each processor section** (the router's scores, the input and the residual parked outside the
capture, the group launched, the experts run on the processor, recording resumed from the
parked copies — `~/swarmllm-138f/pause_capture.patch`). The ids were the same and every step
updated in place, but no faster: 43.9 / 44.8 against 43.0 / 46.6 ms a token uncaptured
(`pausegraph.sh`, 834 tokens, 4 threads, two rounds). Every step is RE-RECORDED, and at each
pause the recording, the update and the launch run with the card idle — the whole-card capture
(#138 (2)) wins because recording overlaps the previous group's run. llama.cpp 4b1a27f captures
once and from then on only relaunches each split's graph while its nodes are unchanged
(`ggml_cuda_graph_update_required`; positions and KV rows reach it as tensors at fixed
addresses). **What would close this gap is a decode step replayed without re-recording** —
every buffer it reads or writes at a fixed address — a change to how a step allocates, not to
the capture. Smaller, not tried: capture only from the last processor section on (the stacked
tail; ~0.1 ms a layer by #138 (2)'s rig, ~2.5 ms of a 44 ms step here). Seen, not chased: the first 1-2 decode
steps after a 681-token single-forward prompt took 2-4x a settled step on both the stacked
and per-expert paths (not after a one-row prompt) — prompt temporaries pressing card memory
is the likely reading; a node chunks card prompts at 512 tokens.
**Measured 2026-10-14, before building it: KTransformers' design (the processor's part as a
HOST NODE inside the graph — fixed pinned buffers, D2H → `cudaLaunchHostFunc` → H2D, so a step
needs no pause) costs too much per host node under WSL2.** `examples/cuda_graph_host_node_probe.cu`,
RTX 3070 Laptop, live node stopped, three rounds: one host node's round trip is 0.29-0.44 ms
with no host work around it and 0.81-0.85 ms after a step that spun 0.77 ms per RAM layer. For
the 30B's 25 RAM layers that is 7-21 ms a token, against the ~15 ms of submission capture would
save. Its synthetic step did run 2.1x faster captured (52.5 vs 110-115 ms; overhead alone 26-29
vs 82-84), but the synthetic submits ~4,000 ops a token, far more than the real step, so it
overstates the gain. KTransformers runs on native Linux, where a host callback costs tens of
µs. Not built; the fixed-address replay above remains the route.

#### #174 — CPU decode attention sits ~2× above the memory-bandwidth floor
`P3` · perf-local · **OPEN** — 2026-08-22 · history: archive § "CPU decode attention: the kernel sits 2x above the DRAM floor"

Modest: about half of ~15 ms/token at long context on processor-only nodes. Three options in
the archive (fold `from_vec`/`to_vec1` into a CustomOp, f16 KV on the processor, single-thread
below ~128 KV). Smallest step: single-thread the rayon split below ~128 KV and measure with
`SWARMLLM_PROFILE=1 prefill_bench`'s attention bucket (min-of-N, idle box).

#### #173 — A request that joins a busy worker never speculates
`P3` · perf-local · **OPEN** — 2026-08-23 · history: archive § "Speculation inside the batched decode scheduler"

`slot_admission_eligible` sends speculating requests down the sequential path; a batched slot
is never speculated (`SlotTable` advances one token per tick). Needs a per-sequence draft and
accepted count in `step_decode_pool`, one batched `forward_verify_all_positions` per tick, a
`SlotTable` able to advance a variable number of tokens, and `truncate_request_to` on reject.
Only worth it if concurrent local serving becomes common.

### Speed across computers (splits)

Designs: `docs/plans/large_model_splits.md` (the target, the bound and the build order, #318),
`docs/plans/split_speculation.md` (guess-and-check across a split),
`docs/plans/wan_parallel.md` (parallelise the reply, not the token), `docs/plans/regional_pipelines.md`
(placement by distance, #192), `docs/plans/faster_than_local.md` (the ranked list).

#### #318 — A large model split across machines is too slow to use
`P2` · perf-split · **OPEN** — 2026-10-20 · plan: `docs/plans/large_model_splits.md` · draws on #134, #145, #148, #149, #192

The swarm exists to run models no member can hold, and a split of one runs at 0.35 tok/s on a
4-segment intercontinental chain, 2.85 tok/s plain and 5-13 tok/s with the stream on a 7B
Thailand↔Belgium. The bound is tok/s = E_N / P: tokens guessed right per ring pass over the
pass time, E_N growing about logarithmically with the rows N a pass carries (measured replays:
5.2-8.1 at 64-128 rows). Projected without any loss in quality: 20-31 tok/s across continents
once the ~95 ms per pass that is neither network nor compute is gone, 50-80 inside a region on
cards, 4-5 on processors for a dense 70B; an opt-in fast mode that checks less strictly ~60-70
across continents. Build order, each step ending on a measurement: (1) the per-pass breakdown
and the largest term removed — rig half measured 10-20: with every packet delayed (`split_rig.sh`
`RIG_RTT_MS`), a 7B split costs exactly one round trip per token more at 206 ms than at 0, and
13-25 ms a token more than the whole model on one node, so the 95 ms is the real link or the far
machine (a relayed peer answered in 330 ms against 231-236 direct). **Found and fixed 10-20:** QUIC
connections (~90% of the live node's) ran Cubic, whose window a split's bursts never fill, so each
was spread over ~0.4 of a round trip (43 KB at 100 ms: 142 ms, 107 under BBR; 280 KB over 20 Mbit/s
with 1% loss: 493, 219) — they now run quinn's BBR (`docs/invariants/network.md` § "A burst is
paced by the link"); TCP's slow start after idle has the same shape and is a system setting, left.
Left: the real link; (2) tree verification in every segment (#134) — **built 2026-10-20, the
default since 10-21 where every segment runs on a card**: 5.33-5.74 tok/s against the stream's
5.01-5.05 on the real TH↔AU link with both halves on cards, 4.75-5.06 against 5.03-5.52 with the
head's part on a processor (385 ms rig); (3) the continuous stream
through a chain of machines — **built 2026-10-20** (`features::CHAINED_STREAM`, bit 27): 1.24 tok/s
against 1.07 for chained rounds and 1.03 without guessing, on a 3-machine processor rig at 206 ms
(the three share one processor, which caps what overlapping their work can win). Building it found
that a chained CHECK failed the whole reply after one token wherever the middle machine had not
met the last: a failed chained check is now re-run unchained, a failed chained stream finished in
rounds, and the middle machine refuses before working and dials its next hop
(`docs/invariants/network.md` § "A stream of checks travels a chain"). Left in this step: the dial
after a DHT lookup is unverified on a real link (the loopback rig's DHT holds no address for a node
that only connected inbound), and a next hop reachable only through a relay is never dialled, so its
run stays unchained. A user who holds NONE of the model now guesses for the head that leads it —
**built 2026-10-20** (`features::REQUESTER_GUESSES`, bit 28; `docs/invariants/scheduling.md` §
"A requester guesses for the split its head leads"). *Real link, one run* (this machine in
Thailand holding layers 0-13 of a 7B on its processor, a peer's RTX 3060 in Australia the rest,
~420 ms apart, the requester on the Thailand machine): led with guesses 2.5 tok/s over 160 tokens
(109 passes: 39 checked guesses, 49 of 86 kept, 65 one-token passes) against 2.27-2.38 driven
from the requester (α 0.66, γ 2); the whole model on the Australian card alone writes 56.6. The 65
one-token passes are passes no fresh window was ready for — the requester's drafter shared the
head's busy processor and took 200-450 ms after a miss, and the head never waited. **(a) Built
2026-10-21**: the head waits for a window it expects, only where the wait pays
(`ngram_only_spec::worth_waiting_ms`, `docs/invariants/scheduling.md` § "A requester guesses for
the split its head leads"). The rule first written here, w < P·(E − 1) (~700 ms on this link), was
too eager — a one-token pass does not throw the window away, it lands during it and still counts at
probability α — and the derived bound is ~220 ms on this link's figures, so the gain is smaller
than the 1.3x first projected. *Real link, 2026-10-21* (the same two machines, the route pinned,
160-token replies): waiting 3.54 / 3.13 / 3.26 / 3.20 tok/s against 3.21 / 3.18 / 3.17 / 3.25
not waiting (`SWARMLLM_HEAD_GUESS_WAIT=0`) — +2.5%, inside the noise: the requester's windows were
due 315-346 ms out against a bound of 261-292 with the head's part on its processor, so the rule
seldom waited, as it should. **(b) Built
2026-10-21**: a `swarm_route` instruction goes with the hand-off to a head advertising
`features::CARRIED_ROUTE` (bit 30), which plans under it from its own side, only ever shrinking
(§ "A route instruction goes to the head with the request"); before, any instruction kept the split
driven from the requester, so the run above was led only because the requester did not yet trust
the Australian peer with the prompt. Verified on the real-link runs: every led reply logged
`route_carried=true` and ran on exactly the two pinned machines. **(c) Measured 2026-10-21** with
the head's part on this machine's card: requester guesses 3.33-3.72 tok/s (the head's first token
2.6-2.7 s, against 15.4-17.1 s on its processor), and the head's own drafter 5.01-5.05 (stream) /
5.33-5.74 (trees, now the default there, #134). The whole model on the Australian card alone
writes 56.6-59.1: a ~405 ms round trip a pass caps a split there, whatever guesses it. Left: (d) a requester
still loads its drafter and reads the prompt for a head that then guesses with its own (DSD runs
first there) — bounded: it stops at the first token, which carries no ids. Then trees in
it, the verdict sent to every segment so dead rows are dropped before they are computed (#149 (b)); (4) a drafter fed by the model's own state: MTP at the last stage (#145),
EAGLE-3 taps where a head is published; (5) an MoE tree priced by the experts it touches; (6)
regional chains (#192); (7) the opt-in fast mode. The plan records one retracted projection
(~40 tok/s across continents, from treating the miss rate as free).

#### #10 — Conversation prefixes across computers: a cache this node did not fill is invisible
`P2` · perf-split · **PARTIAL** — 2026-09-07 · absorbs #139 and archive § "Speeding up inference BETWEEN nodes" idea 1 (prefix-keyed remote KV); history: archive row #10 and § "A conversation's later turns do not seek out the peer holding its prefix"

The LOCAL half shipped in v0.3.208 (`scheduler::cached_prefix`; `split_rig.sh cache` HIT
2,713 of 2,744 tokens). **The split half is built (2026-10-09, released in v0.3.234, `pipeline::split_prompt_cache`,
`docs/plans/split_prompt_cache.md`)**: a split's prompt pass carries a keyed block chain and
resumes where every segment is believed to hold it; each segment keeps its part in its worker's
`PrefixCache`; a miss re-sends from 0 once. It keys on the prompt's CONTENT, so the old note that
the Anthropic surface could not reach a session-id design no longer applies — every surface's
prompt pass goes through `forward_through_segments`. What is left:
- **Route to the peer that holds it — built 2026-10-11 for what THIS node knows, released in
  v0.3.235.** The planner reads the belief table: each (node, exact layer range) believed to
  hold the prompt's opening is credited what it would not read again
  (`NodeCandidate::prompt_tokens_held`, both routers; greedy now prices the unread prompt —
  `docs/invariants/scheduling.md` § "A split segment is credited what this node believes it
  holds"). *Measured* (`split_rig.sh splitcache SPARE=1 WARM_SPARE=1`, both peers warm, only one
  holding the conversation): before, turn 2 went to the other peer and read 2,306 tokens again in
  2 of 2 runs (196 / 195 s); after, it went to the holder and resumed from 2,240 in 2 of 2
  (55.8 / 53.6 s, debug builds); the overlapping shape greedy plans, 142.8 s → 52.3 s. The cold spare of 2026-10-10 (`SPARE=1`, fixed by
  `NodeCandidate::cold_load_ms`) is the other half of the same choice. What is left: a peer's
  cache THIS node did not fill is invisible — gossip prefix digests (the credit must then require
  `cross_node_prefix_trust_min` and `share_prefix_cache_with_peers`, default off); keep caches
  longer than the 10-minute KV idle expiry with a RAM/disk tier; price a CPU requester's long
  prompt against a GPU peer's rate (`faster_than_local.md` § 3.1). (The delivery trap noted here
  until 10-19 — the n-gram path taking a whole-model hand-off first — was already closed:
  `ngram_only_spec::eligible` declines every plan `remote_generate::eligible` takes, since
  2026-09-27, and `split_rig.sh handoffcache` read the peer's HIT for #3 (c).)
- **Limits the split half accepts**: a resumed request gives up the mid-reply replay onto a
  stand-in (the router continues the reply instead, #236); a model with a recurrent state (Qwen
  3.5) stores nothing; images, pre-embedded input and tensor-parallel groups run whole; the
  beliefs live in one process and start empty after a restart.

#### #171 — The prompt pass through a split runs one stage at a time
`P2` · perf-split · **PARTIAL** — 2026-08-24, released in v0.3.234 for the splits this node leads · absorbs archive § "Speeding up inference BETWEEN nodes" idea 2 (prefill microbatching); history: archive § "The pipeline is idle (N-1)/N of the time during the phase that dominates a long request"

**Built 2026-10-10, released in v0.3.234 (`pipeline::prompt_chunks`, `docs/plans/pipelined_prompt_pass.md`):** a prompt
pass of two or more 512-position pieces is read in pieces, every segment at once, where no
machine appears twice and either the pass keeps its prompt for #10 (it is never chained, so the
relay through this node is already its cost — three- and four-peer splits included) or this node
is on every boundary between segments; `LayerForward::prompt_span`
(`0x0E`, `features::PROMPT_CHUNKS`, bit 21) tells the worker to start the pass at the first piece
— admitted for the WHOLE span — and continue it at the rest. Measured on Llama-3.2-3B, A layers
0-12 / B 12-28, each node pinned to half of this 8-core machine (`split_rig.sh splitcache`,
`CPUS_A/B`), a 2,274-token pass: 71.3 / 69.6 s whole, 58.2 / 57.0 s in pieces (1.22×); split
three ways, A 0-12 / B 12-21 / C 21-28 (`SHARDS_C`, a boundary between two peers): 76.8 / 70.1 s
whole, 46.5 / 47.0 s in pieces (1.49-1.65×); replies byte-identical in all eight arms. One box understates it: the two nodes share one memory bus, and B's pieces took
13.3 → 4.5 s once A had finished its part; on two machines the pass approaches B's own time.
What is left: (a) pieces CHAINED across a boundary between two peers — each hop handing its
piece on (the "compute each chunk on arrival" form below) — for the pass that does not keep its
prompt (today it is chained whole there, since relaying every piece through a far coordinator
could cost more than it saves; a kept pass relays either way and reads in pieces); (b) a speed
sample per piece (none is recorded — a piece's wait includes the one queued before it); (c) a
measurement on two real machines.

During the prompt pass — 94% of a long request's cost — only one machine of N works at a
time, so the ceiling is ~2-2.8× on long prompts. *(The original design — compute each chunk
on arrival instead of reassembling the whole forward, behind a feature bit — is what shipped in
v0.3.234 (`features::PROMPT_CHUNKS`, above); only (a)-(c) remain.)* This is also step 3 of `split_speculation.md` (pipelined
prompt pass) and an item of `wan_parallel.md`.

#### #149 — Split speculation's continuous stream ships on by default; four refinements remain
`P3` · perf-split · **PARTIAL** — 2026-09-30 · history: archive row #149; design: `split_speculation.md` § 4b

On by default since v0.3.216 (`SWARMLLM_SPEC_STREAM=0` keeps the rounds): +20-57% over rounds
on a real link with a processor peer. Open: (a) a window sized from acceptance × round trip
(W stays 3; W=6 read −7% / +5% / +18%) — and, from #152's boomerang rig, whether to stream at
all: with a slow far node (~250 ms a check) at a 24 ms round trip the stream was only level with
rounds (+2% prose, −5% code), against +12% / +32% at 270 ms, so the round trip's share of a
check is the input; (b) cancelling a stale chunk the far node has already
STARTED — only unstarted ones are skipped, and two forwards of one request at a worker cross
their replies (gotcha #180), so it needs the worker's cancel path; (c) the drafter's near
layers as the verify input where the head segment is the coordinator's own; (d) mid-reply
failover is #167. The TH↔BE two-CARD case was never measured — #151 now makes a same-build
card split possible. A same-finetune drafter is NOT better here (coder 0.5B lost to the
general 0.5B even on code); the tie-break to the general model stays.

#### #148 — A split still pays ~3.6-5.3 ms per token after each segment's forward, and two chats through a split are never batched
`P3` · perf-split · **PARTIAL** — 2026-09-29 · history: archive row #148 and § "A split on a fast link (#148, 2026-09-29)"

The main cause (the batch scheduler holding every lone decode forward 5 ms) shipped in
v0.3.213 — a 0 ms split went from 54% to ~80% of local. Both remaining items need TWO real
machines (the rig's shared card hides interleaving): (1) what each segment spends after its
forward (the card finishing, the hidden state to the host, Q8 encode, tail sampling over a
152k vocabulary); (2) whether two concurrent chats through a split interleave usefully —
measure unbatched aggregate against forced batching before changing anything. A local
segment still ends the chain when the requester holds the tail (the delegated split #143
covers only requesters holding nothing).

#### #134 — Split decode: draft trees wait for their drafter, and the drafter is chosen by size, not by its device (the pipelined prompt pass is #171)
`P3` · perf-split · **PARTIAL** — 2026-10-22 · history: archive row #134, § Closed 2026-10-22 ((c)); plan: `split_speculation.md`, `large_model_splits.md` § "Draft trees"

Shipped: guessing ahead on by default (v0.3.213), the continuous stream (v0.3.216, #149), walk
at the tail, γ control, the engine drafter; **draft trees** (v0.3.246: trailer `0x10`, bit 29,
tree masks in every segment, the next tree grown beside its check). **Since 2026-10-21 trees are
the default only where every segment AND the drafter run whole on a card**
(`dsd_tree::trees_wanted`, `drafter_on_card` on `DSD: request complete`), the stream elsewhere:
on the real TH↔AU link (~405 ms, both halves on cards, the drafter on the head's processor) trees
ran 5.33-5.74 tok/s against the stream's 5.01-5.05 over 160 tokens but 4.82/5.05/6.40 against
5.97/6.04/6.18 over 400 and ~10% slower over 28; with the head's part on a processor (385 ms
rig) 4.75-5.06 against 5.03-5.52 (`docs/invariants/scheduling.md` § "A split checks a tree of
guesses", Measured). Left for trees: (a) **the drafter's wait after a missed seed** — the tree
grown beside a check held its outcome in only 3-30 of 59-95 rounds, and each miss grows the next
tree from nothing before the check: 289-416 ms of a ~0.85-0.95 s round with the drafter on the
processor — which is why the default needs the drafter on the card, and why a card plan now
leaves it room there (#275 (4), closed 2026-10-22). The drafter's processor width is the
contribution's (4 of 8 threads on that run: 16 ms a single-row step). **On the card
(2026-10-21, TH↔AU, receipt DIAG):** a round of ~760-820 ms was the network's ~425 ms (minimum
RTT 419), the regrow after a missed seed 172-211 ms (10-15 of 24-29 rounds), the head's tree
forward ~100 ms received→computed beside the drafter's call, the far side 59 ms. The drafter now
ranks its followers on the card (`draft::likeliest_rows_where_they_are`): a 64-row growth from
nothing 208 → 155 ms, a seed read 100 → 35 ms (`SWARM_BENCH_GROW`, min of N). What is left of a
regrow is the tree steps' standard attention (~40 ms; a chain step on flash is 11.9 ms at 8 rows,
on standard attention 16.5, as a tree 18.2) and the rows' quantized matmul on both machines (#325,
closed 2026-10-21: llama.cpp's tensor-core MMQ, 9-64 rows 2.4-3.2× faster). **Built 2026-10-21:
the regrow is sized by what it buys** (`TreeRounds::grow_size`, 1/2/4/8 forwards, the most tokens
per ms of the round; A/B `SWARMLLM_TREE_REGROW=fixed`) — on the rig (7B, the drafter on the
processor, trees forced) 11.12-11.90 tok/s against 7.33-7.48 at 20 ms and 5.40-5.54 against
5.12-5.34 at 300 ms; the chain stream still led both (15.84-16.24, 6.00-6.03), and **trees are
the default only where the farthest peer segment is a measured 300 ms or more away**, read as the
least round trip measured (the smoothed cost read 2.5-5 s on a 20 ms link through a first
request) — `docs/invariants/scheduling.md` § "A missed seed is regrown only as deep as pays".
Left of (a): the drafter's own speed on the processor. **Built 2026-10-22 (Track A): a tree
step pays for its tree once** — the rows' rope tables gathered once per forward (3 host-to-card
copies a tree forward against 50; a 0.5B tree forward 12.3 against 16.6 ms at 8 rows; a tree
grown from nothing 118.5 against 145.7 ms) — **and a call beside a check grows 32 rows a forward
up to 416** (was 8 and 128; the real-link A/B is owed: the far peer had no conversation room at
04:44Z) — `docs/invariants/scheduling.md` § "A tree step pays for its tree once". The wait it
targets: per reply the seed missed on 32-51% of rounds and was grown ~8 forwards before the check
on most others, 145-163 ms a round. (b) **a tree that FLOWS — built 2026-10-22, unreleased**:
rows streamed as grown, the far end appending them to the tree it holds and answering each row
with its draw, the requester walking the draws (`pipeline::dsd_flow`, trailer `0x11`, bit 31),
three levels of 8 rows a drafter call; on the card rig (385 ms, all three processes on one card)
8.99-9.44 tok/s against tree rounds' 8.03-8.27 over 400 tokens, replies scored against llama.cpp
— +11-14%, short of the 1.3× exit (`docs/invariants/scheduling.md` § "A tree that flows",
Measured). Left of (b): the real link (the far peer must advertise bit 31 — after a release);
the drafter's level, ~72% of the requester's time — **built 2026-10-22, unreleased: a level of
≤ 8 rows is one tree kernel and one captured graph** (`decode_attn::gqa_tree_attention_cuda`, a
bitmask per row; `cuda_graph::StepShape`; `docs/invariants/inference.md` § "A tree level on a
card") — on an idle card a flow call's three levels take 26.5 ms against 36.0 (−26%); on the
one-card rig the drafter's time fell ~9% and this node's half rose
about as much (three processes on one card, gotcha #759), so the reply's speed did not move
there. Left of it: a whole drafter call as ONE graph with its rows chosen on the card (SGLang's
EAGLE captures its draft levels and the top-k between them; ours chooses on the host between
levels, a sync each); a kernel sharing K/V loads across rows (FlashInfer's query tile) to lift
the 8-row cap off the head's ~25-row forwards; how many levels a call pays once a level is
cheap (re-measure the 2/3/4-level table off the one-card rig); misses (16-24 a reply, each about a round trip); the flow sends ~25 rows a
token against rounds' ~6 (~0.8 MB/s up at 9 tok/s, and four times the far peer's work) — on a thin
uplink that lost to rounds (5 Mbit: 4.5-4.8 vs 6.8-7.1 tok/s), so **since 2026-10-22 a guard holds
it down** (`FlowGuard`, LEDBAT++-shaped, answers timed on arrival: 5 Mbit 7.44-7.50 vs 4.73-4.80
unguarded, 10 Mbit 8.52-8.82 vs 8.23; `docs/invariants/scheduling.md` § "A tree that flows") —
its alternating check with no rate limit found no cost (guard on 8.83-9.58, off 8.79-9.82) — left of
it: the guard on the real link and on a busy peer. **Built 2026-10-22, unreleased: copied runs**
(`FlowRun::copy_ahead`, SuffixDecoding-shaped: the n-gram lookup's continuation pushed under the
walk's row as `FlowOp::Copy`, rated by the drafter's own guess) — a code edit that copies a file
23-24 → 42-47 tok/s, chat unchanged (§ "A tree that flows", Copied runs); left of it: a copy
source beyond this conversation (SuffixDecoding's global tree of earlier replies), and the
copies through the tree ROUNDS, which still guess with the drafter alone; the drafter's call
beside this node's layers is a loss while they are two processes on one card (gotcha #759) and
needs them in one; (c) closed
2026-10-22 — with the drafter on the card trees beat the stream 8.41 vs 5.95 tok/s at 160
tokens and 9.36 vs 6.48 at 400 on the real link, so the default stands (§ Closed). **Built 2026-10-22: a split with no drafter fetches one** — the largest of the
target's family the swarm holds that can draft, ≤ 2 GiB, trusted, through the shard system, kept
while splits want it (`docs/invariants/memory.md` § "A split with no drafter fetches one"). Left
of it: a fetched drafter refused for memory at request time does not lead to a smaller one; a
family the swarm holds no small model of is not looked for on HuggingFace; end to end on the rig it
fetched the 0.5B from its one holder and the next split guessed with it (`split_rig.sh repeat` + `FETCH_DRAFTER`,
`docs/invariants/memory.md`); the real-link run on a CUDA build is owed (gotcha #849). **The largest
is chosen (and fetched) whatever device it drafts on** — right on a card, marginally wrong on a
processor. Leviathan et al.'s speedup (1−α^(γ+1)) / ((1−α)(γc+1)) at a 400 ms round, c = the
drafter's step over the round: on a processor (0.5B q8 ~16 ms a step, α ~0.75; 1.7B ~50 ms, α
~0.85; γ 6) 2.80 vs 2.59 — the 0.5B by 7%; on a card (c ~0.01-0.02) 3.27 vs 4.04 — the 1.7B by
24%. Choose by expected tokens per ms on the drafter's own device once the per-pair α is
measured (`dsd_controller` keeps it per model and machines); estimate, 2026-10-21. The pipelined
prompt pass (Phase 3) is #171. The
same-model-at-fewer-bits drafter is decided against as a default (#144, § "Decided").

#### #275 — A target already loaded keeps its layers, so the drafter's room appears only at its next cold load
`P3` · perf-split · **PARTIAL** — 2026-10-22 · history: § Closed 2026-10-14 (#275's first half), 2026-10-15 (the reply's charge; judged beside the plan), 2026-10-22 ((4), card plans)

Shipped: the drafter is asked before it is chosen, beside the layers the plan loads here; a
split leaves it room where the peers offer enough; that room is the reply's own cache, not
4,096 positions (`docs/invariants/scheduling.md` § "A split leaves room for the model that
guesses for it"); on a card, on the card (#275 (4), closed 2026-10-22). Left: (3) a target
worker already holding more layers keeps them; the room appears from its next cold load. The
room is sized for the drafter `engine_drafter::chosen` picks, the largest by bytes — on a card
the 0.5B fp16 (~1 GB) over a Q8 copy (~0.65 GB), unmeasured as a guesser (#134's device-aware
choice).

#### #145 — One reply uses one row of a ring pass that could carry dozens
`P3` · perf-split · **OPEN** — 2026-09-28 · history: archive row #145; plan: `docs/plans/wan_parallel.md`

Every machine in a split idles ~99% of each pass. The plan — "parallelise the reply, not the
token", within the rule that nobody holds a whole model — has its first step done (the
continuous stream, v0.3.216) and #143 done (v0.3.221). Next in its order: multi-token-prediction
heads at the last stage (Qwen 3.5 runs since v0.3.229 but without speculation, #228 (5); GLM-4.5+ is
unsupported), a stage batching forwards of several streams into one
visit, multi-block attention for workers sharing a cache, outline-then-expand with a
classifier router. Research-grade; the largest multiplier on the list
(`wan_parallel.md` § 6 has the projection). Prior art read 2026-10-20 (github.com/JustVugg/colibri,
a single-machine MoE engine with MTP drafting; its docs, not reproduced here): an MTP head cut to
int4 collapsed to 0-4% acceptance and is shipped at int8 (its `docs/tuning.md`) — a head we load
keeps at least Q8 — and drafting ran at half the speed of plain decoding on GLM-5.2 because the
checked rows pulled in extra experts (its `docs/experiments/glm52-4xa6000-2026-08-02.md` § 9), the
cost #318's step (5) prices.

#### #182 — SWARM-SPEC Layer 3 (conversation prefetch) is observation-only code in the request path
`P3` · perf-split · **OPEN** — since R136 · history: archive § "R136: SWARM-SPEC — proposal…" → "Layer 3 — Conversation-level: predictive prefetch (NOVEL)"

`inference/prefetch.rs` (predictor and learner) and a router decision block exist, but the
dispatch only logs "prefetch would fire — observability-only (K-layer compute deferred)" and
emits an activity event; `prefetch_enabled` defaults false. No K-layer compute, no gossip
warming, no placement precompute — a stub, against the "no stubs" rule. **Decision: delete**
`prefetch.rs`, the router block, its admin/websocket metrics and config keys, unless a
benchmark harness first shows a TTFT win for the idle-time K-layer seeding. (Its learner's
per-session response histories are a candidate input for #3 — keep that in mind when
deleting.)

### Model families and platforms

#### #326 — A card newer than the release's machine code waits minutes for its first reply
`P2` · platform · **OPEN** — 2026-10-21 · history: gotcha #928, #325 (closed)

The vendored kernel libraries (MoE, MMQ, flash attention) carry machine code for the release's
`CUDA_COMPUTE_CAP` (80, which runs on 8.x cards only) and, since 2026-10-21, its PTX. A Hopper or
Blackwell card (9.0, 12.0: the RTX 50-series) runs the PTX, which the driver compiles on first
use. Measured with `CUDA_FORCE_PTX_JIT=1` on this RTX 3070 and an empty cache
(`~/swarmllm-318/jittime.sh`; `docs/invariants/inference.md` § "A card multiplies several rows"):
flash attention's first non-causal call 66.7 s and first causal call 45.2 s at one head width,
MMQ 7.3 s (Q4_K) and 8.6 s (Q6_K) per weight type; with the cache filled, tens of ms. So a new
user's first reply on such a card waits ~2 min before a token, with nothing on screen saying why;
later replies, and later runs, read the driver's cache (34 MB for flash at one head width; the
driver's cache size limit is not checked). Before the PTX went in, the same calls ran nothing
and answered garbage. Untested: what the node's own deadlines do with a 1-2 min first forward (a
peer's segment is timed by the coordinator), and candle's own PTX modules' JIT time. Options:
machine code for 9.0 and 12.0 beside 8.0 in the release build (nvcc 12.8 builds sm_120; the
flash kernels are cached across releases, so the cost is one cold build and the binary's size),
or a warm-up at worker start that says what it is doing. No such card is here to check either
on.

#### #228 — Qwen 3.5: what the first version leaves out (speculation, and the 4B's decode speed)
`P2` · model-support · **PARTIAL** — 2026-10-06 · history: #117 (closed); plan: `docs/plans/qwen35_support.md`

Dense Qwen 3.5 runs since #117. Still out: (1) ~~`qwen35moe` refused~~ — **runs since 2026-10-11,
released in v0.3.235**: the loader already read its layout (the dense loop with `load_moe_ffn`; the
popular unsloth Qwen3.5-35B-A3B Q4_K_M header, read by range request, names exactly those tensors —
separate gate/up/down expert stacks, a sigmoid-gated shared expert, no fused `ffn_gate_up_exps`), so
the change is `is_supported`, `supported_list` and `arch_supports_hybrid`. Checked against llama.cpp
master on a real file (`Flexan/kshitijthakkar-qwen3.5-moe-0.87B-d0.8B-GGUF`, 24 layers, 8 experts,
top-2; `~/swarmllm-228/card{,2}.sh`, live node stopped, safety kit, 0 driver events): the f16 file
agrees at every position on the processor (worst cos 0.999999, top-1 24/24), split at layer 13 and
at 4, and on the card whole and split (0.999999 — llama.cpp's own card reads 0.999897 against its
processor). `test-split`, three prompts × 40 tokens, `score_ids_dump.py`: f16 120/120 tokens llama.cpp's first
choice both whole on the card and split 12/24 across card and processor; Q4_K_M 119/120 split and
118/120 whole (near-ties of 0.337 at the same token in both, and 0.278 — Llama-3.2-3B misses that
0.05 margin by 0.306). The released
.234 refuses the file (the control). Q4_K_M logits sit 0.997-0.989 from llama.cpp's processor
(processor / card) and so does llama.cpp's OWN card (0.983) — and dense Qwen3.5-4B Q4_K_M, a shipped
family, reads 0.998 / 0.994 the same way: 4-bit rounding, amplified in a MoE by a top-k choice it
flips. Not run here: the 35B itself (22 GB, more than this machine's memory). (2) ~~the card/processor
split~~ — **released in v0.3.230 (2026-10-07)**: dense Qwen 3.5 is on `arch_supports_hybrid`,
checked split on the card against llama.cpp (`docs/invariants/inference.md` § "A card/processor
split is placed in every per-layer loop"). Its sizing followed the same day:
admission, the split planner and the loader's KV budget charged every layer an attention
layer's KV while three in four are DeltaNet with a fixed state — ~805 MB too much for the 9B at
the admission context, the margin by which it missed an 8 GB card. All three, and the
scheduler's bound on a peer (`kv_bytes_per_position_per_layer`, from the manifest's tensor
table), now ask `split::layers_keeping_kv` (the real 9B header: attention at layers 3, 7, …, 31, llama.cpp's
rule). Still uncharged: each DeltaNet layer's per-request state (~2 MB a layer for the 9B,
covered by the forward-buffer reserve); (3) ~~the delta rule runs token by token~~ — **one op per
layer since 2026-10-11, released in v0.3.235.** llama.cpp master no longer chunks it either: its fused
`GGML_OP_GATED_DELTA_NET` runs every prompt and decode step (`fused_gdn_ar` / `fused_gdn_ch` on by
default, read at `4b1a27f`), so neither do we — `layers::qwen35::delta_rule_cpu` (a task per sequence
and value head, each state row decayed, corrected and read in one pass) and on a card
`kernels/delta_rule.cu` (llama.cpp's `gated_delta_net_cuda`, a warp per state row). A 685-token prompt on the 4B: 55.8 → 19.8-21.1 s on the processor (llama.cpp 14.9 s), 2.6 → 0.49-0.53 s whole on the card; the card kernel agrees with llama.cpp at top-1 200/200.
`SWARMLLM_DELTA_RULE=steps` is the old composition (`docs/invariants/inference.md` § "Qwen 3.5's
delta rule is ONE op per layer"); (4) ~~decode is never captured as a CUDA graph~~ —
**captured since 2026-10-11, released in v0.3.235**: on a card each DeltaNet layer keeps its state in
two buffer pairs and a step writes the pair it did not read (`SsmState`), as llama.cpp keeps fixed
cache tensors; the 0.8B writes 121-126 tokens a second instead of 50-54 whole on the card, the 4B is unchanged
(~37, card-bound), ids identical with capture on and off (`docs/invariants/inference.md` § "Qwen 3.5's decode steps are captured
too"). MoE layers are captured where their experts are stacked on the card (#138 (2), 2026-10-14); (5) no speculation of any kind — a
snapshot of the recurrent state per draft would allow it; (6) ~~the 4B's gap to llama.cpp~~ —
checked 2026-10-07, not a defect: six prompts scored on the processor against llama.cpp master
(`score_ids_dump.py`, margin 0.05) — Qwen 3.5 4B Q4_K_M misses on 3 (gaps 0.11-0.21), and
Llama-3.2-3B Q4_K_M, a long-verified family, misses on 1 by MORE (0.306, "why the sky is blue").
Unquantized, the 0.8B matches llama.cpp exactly (worst cos 0.999999), at Q8_0 within 0.012: the
model's arithmetic is right and 4-bit near-ties fall the other way through our kernels' rounding. Re-check with
`~/llama.cpp-ref/{dump_logits,ref_generate}` (llama-cpp-python 0.3.16 cannot load `qwen35`), and
render a new template with jinja2 first (gotcha #798); (7) ~~the 4B decodes at ~37 tokens a second whole on an RTX 3070 Laptop with or without CUDA-graph capture~~ —
**no longer true, re-measured 2026-10-19**: 70.8-75.3 tokens a second through a node (an isolated node serving
the file's parts, a streamed 200-token reply, after its first 8) and 72-76 in `test-split`, which now prints its
decode rate, against llama.cpp's 80.3 (`llama-bench -ngl 99 -n 128`, same file, same sitting, live node
stopped): 89-94%. Without capture it reads 40, the old figure; on 10-11 capture did not lift the 4B and now it
does, and which change between did it is not established. Left: the 6-11% to llama.cpp — 1,540 launches a
step against Phi-4-mini's 804: each DeltaNet layer quantizes one input four times (`QMatMul::forward_shared`
would do it once), runs its convolution as ~12 launches, softplus and the L2 norm as ~5 each; (8) ~~every
turn re-read the whole conversation~~ — **saved since 2026-10-18, released in v0.3.243**: the prefix cache
skipped any request holding recurrent state, because a DeltaNet layer's state cannot be cut back to a
shorter prefix the way attention KV can. Now the prompt is saved 16 tokens short of its end (llama.cpp's
context checkpoints, PR #20288, read at `4b1a27f`: the next turn's template re-renders the last few
tokens differently, so a save AT the end never matches), the prefill stops at that point to copy the
state, and a hit restores it whole or not at all (`docs/invariants/inference.md` § "Qwen 3.5's prompt is
saved short of its end"). Qwen3.5-0.8B Q8_0 on the processor (dev build, live node stopped,
`~/swarmllm-228c/rig.sh`, both slot and sequential paths): a second turn of a 2,416-token conversation
took 16.7 / 15.0 s instead of 127.6 / 135.8 s cold, replies identical to the cold run. Such a save is
never announced to peers or exported — a peer's copy could not be narrowed either.

A family in `supported_list` is a claim — check it against a REAL file's header (#715), and
never flip `ModelArch::is_supported` without a real-file comparison against llama.cpp
(`examples/logits_reference_probe.rs` + `compare_logits_reference.py`; replies with
`score_against_reference.py`).

#### #1 — Every Mac runs inference on the processor: no GPU backend is compiled for Apple Silicon
`P1` · platform · **OPEN** — 2026-09-07 · history: archive row #1 and § "GPU on Apple Silicon: no backend is compiled, on either path"

A large population, always, visible only as slowness. The macOS release jobs build with no
GPU feature, and the split loader falls back with `Device::cuda_if_available(0)
.unwrap_or(Device::Cpu)`. Real support means wiring `candle-core/metal` (vendored candle
already carries the feature) through an analogue of the `candle-cuda` feature and taking the
split executor's kernels with it — `split/token_embedding.rs` notes Metal lacks the row-gather
path. Adding only `llama-cpp-2/metal` would fix the DISPLAY while inference stays on the
processor, which is worse. GitHub's macOS runners compile but cannot run Metal; it needs a Mac
to test on (the swarm's Italian M4 peer belongs to a tester).

#### #115 — Gemma 4's vision projector and its drafters are unsupported
`P2` · model-support · **PARTIAL** — 2026-09-25 · history: archive row #115

Requested for gemma-4-26B-A4B. Order: ~~(1) Gemma 3 text~~, ~~(3) Gemma 4 dense text~~ (the
12B and 31B), ~~(3b) the 26B-A4B~~ and ~~(3c) the E2B/E4B~~ (closed 2026-10-16/17, § Closed;
what (3c) left is #294-#296); ~~(2a) the `gemma3` projector and pictures at their place in the
prompt~~ and ~~(2b) a picture's rows read both ways~~ (both closed 2026-10-19, § Closed);
(4) `gemma4v`; (5) the `-assistant` drafters
(`gemma4-assistant` in llama.cpp, an unknown family here). Experts carrying a gate/up scale are
refused by name (`GgufTensorMeta::unbuilt_gemma4_part`; llama.cpp reads them, no file met so far
has one). The 31B
has the 12B's layout (60 layers, 16 / 4 key/value heads) and is unchecked. Gemma 4's reference is llama.cpp MASTER
(`~/llama.cpp-ref/dump_logits`, `tok_cases`); llama-cpp-python 0.3.16 predates it.

#### #294 — Gemma 4's E2B and E4B run only whole on one computer
`P3` · model-support · **OPEN** — 2026-10-17 · history: #115 (3c)

Every layer reads a per-layer input built from the prompt's TOKENS (`split::per_layer`), which
only the first segment sees, and the last layers (the E2B's 20 of 35) attend caches that two
earlier layers keep (`split::kv_sources`: the E2B's 13 and 14). So `GgufTensorMeta::runs_only_whole`
keeps them to one computer holding every layer: the scheduler offers only a whole holder's plan
(`assemble_pipeline_for` → `InsufficientCapacity`), no partial range or tensor-parallel group;
the loader refuses a partial range; "Start and finish on this computer" runs locally or refuses
(`PromptPrivacyUnavailable`); hidden states without their tokens are refused. At Q4_K_M the E2B
is 3.4 GB and the E4B 5.4 GB, so a computer that cannot hold one is rare; what it costs is that a
node holding PART of one cannot serve it. To split: send each forward's per-layer inputs beside
the hidden states (vLLM's Gemma 3n carries `per_layer_inputs` in its `IntermediateTensors`
between pipeline stages), behind a feature bit, and cut only where every shared layer's source
is in its segment — at or before the first source layer.

#### #116 — The DeepSeek-2 family is recognised and refused: no real file can load
`P2` · model-support · **OPEN** — 2026-09-25 · history: archive row #116

DeepSeek-V2/V2-Lite/V3/R1, Kimi-K2, GLM-4.7-Flash. The loader's MLA branch, `MlaWeights` and
`LayerVariant::DeepSeek` stay as the starting point. A real file needs: (1) the
`attn_kv_a_mqa` tensor name; (2) per-layer MLA detection when `attn_q_a` is absent
(V2-Lite); (3) the split `attn_k_b`/`attn_v_b` layout and `key_length_mla` keys; (4) YaRN on
the MLA branch, which builds its own RoPE table, with the `yarn_log_multiplier` mscale (YaRN
itself is read since #280: `split::rope::RopeScaling`); (5) routing — `expert_weights_norm`
default false, `expert_weights_scale`, the `exp_probs_b` bias, expert groups, the gating
function. Verify on a tiny GGUF of each layout, then flip `is_supported` and review the MLA
projections for `arch_supports_hybrid`.

#### #114 — Mixture-of-experts: Llama-4's long-context attention is not implemented (real Qwen3/Qwen3.5 MoE files verified)
`P2` · model-support · **PARTIAL** — 2026-09-25 · history: archive row #114

Experts stay quantized, one matrix per expert (`loader::load_moe_ffn`, `split_expert_stack`);
Qwen3-MoE and Qwen2-MoE are verified on tiny random models, Llama-4 routing on tiny GGUFs
(v0.3.207). **Done:** (a) a real Qwen3-30B-A3B Q2_K file scored against llama.cpp on the
processor and the card (#138, 2026-10-11/13: median cosine 0.99952 vs llama.cpp's own 0.99955);
(c) the Qwen3.5-MoE sigmoid-gated shared expert, checked on a real `qwen35moe` file (#228 (1),
worst cos 0.999999). Open: (b) Llama-4 NoPE temperature tuning and 8,192-token chunked
attention are not implemented, so replies past 8,191 tokens differ; (d) device-side top-k only
if a profile asks. Placement on
a small card is #138.

#### #118 — StarCoder2 is recognised and refused: a LayerNorm model with biases the loader cannot read
`P3` · model-support · **OPEN** — 2026-09-25 · history: archive row #118

Needs a LayerNorm variant (mean-subtracted, with bias), bias sites on `attn_q/k/v/output` and
`ffn_up/down`, the `layer_norm_epsilon` key in `GgufTensorMeta::from_content` (it requires
`layer_norm_rms_epsilon` today), no `ffn_gate` (GELU MLP), tied output. Verify on a tiny GGUF
against llama.cpp 0.3.16, then flip `is_supported` and re-read `arch_supports_hybrid` (it lists
StarCoder2, inert while refused).

### Memory and admission

#### #322 — A cold load's first request was refused for a 9-token prompt, its card cache budget 0
`P3` · memory · **OPEN** — 2026-10-20 · history: field report 2026-10-19 (a tester, RTX 3060 12 GB, v0.3.244), request `3acb5624`

The first request after a cold load of Qwen3-30B-A3B (this node L0-14 and L42-48, a peer
L14-42) failed after 46 s. The second local range (L42-48, 2,262 MB) was admitted to the card
with `headroom_mb=114` (`committed_mb=5595 budget_mb=7971`), and its worker's cache budget then
read 0 (`budget_mb=0 load_time_budget_mb=187 short_by_mb=40`); the re-plan found
`layers_that_fit=0`, and the owner was told to close other programs though nothing else held
the card. The same plan answered 49 s later, in 1.3 s; in the cold run after it the range was
admitted with `headroom_mb=385` and answered. Lead, UNDETERMINED: the pool admits a range
(`card_verdict`) without the worker's own cache margin (`device_free_margin_bytes`: the larger of
5% of the card and 256 MB, ~600 MB on 12 GB), so a range admitted with less headroom than that
loads with no room for any cache; why the headroom was 114 MB once and 385 MB a minute later is
not known. Next: log the card reading (`device_free_and_total_bytes`, the pool's reusable bytes)
at the refusal at `info`, then decide whether `card_verdict` charges the margin (the spawn's
ladder would split the range instead of loading it with no cache). The refusal's advice also
blames other programs when this node's own weights fill the card.

#### #297 — Auto-manage prices a card by the file's size, so Gemma 4's E2B/E4B read heavier than they run
`P3` · memory · **OPEN** — 2026-10-17 · history: #296's review

`auto_manage::vram::estimate_model_vram_mb` (the file × 1.15) charges a card the whole file,
including the per-layer table the loader keeps in the machine's memory (#296). It reads the
E2B Q4_K_M at ~3,780 MB, where the loader's card estimate is 1,859, and the E4B at ~5,910 against
3,483. Five places decide with it, none of them on the request path:

- `quant.rs` picks which quant of a family to recommend for this node's card (or a third of the
  pool's), so a 4 GB card can be pointed at a lower quant of the E2B than it runs.
- `scoring.rs` sets the pool-fitness factor in the download score.
- `wishlist.rs` shows the "Memory" figure and the "fits your memory" tag.
- `capacity_plan.rs` decides "too large for the swarm today" and what a new node would unlock.
- `scan.rs` (`estimate_segment_vram_mb`) decides whether a segment is registered in
  `split_models`. Nothing that serves a request has asked for that registration since #259.

The model listing asks the loader (`gpu_estimate_and_fit`); it uses the file's size only for a
model that peers alone hold. The file-size figure is rough for every model: its own doc puts it
56% low on phi-3.5-mini, and it knows nothing of card/processor splits or experts kept in RAM.
So the fix is to price from the header where one is present (the pool's estimator, or at least
`GgufTensorMeta::host_resident_bytes`), not a special case for Gemma. Not yet measured: what each
of the five reads for the E2B on a 4 GB card.

#### #291 — A running split worker's further room is not offered to the planner, nor weighed for promotion
`P3` · memory · **OPEN** — 2026-10-16 · history: what #265's fix left

Since #265, a live card worker that cannot take a range whole takes it across the card and the
processor (`charge_segment_as_split`), so report #037's plans now run. Three narrower things are
left. (a) The planner still bounds a running worker by the card's room (`parallax::local_fits`),
so such a plan is found only on the relaxed `LocalUnbounded` rung; a separate allowance for one
further split run would find it first, but gotcha #806 is the trap (an allowance that holds for
one loader call handed out for every call). (b) A worker spawned whole on the card that grew a
split range keeps `card_split: None` on its handle, so `short_of_the_card` and promotion never
weigh it — the range stays split after the card frees until the worker is retired or unloaded.
(c) A later range's processor-held caches are bounded inside the worker by the spawn's grant
(`--kv-budget-bytes`), else by the machine; the pool charges them, the worker's own ceiling is
not raised. None of these fails a request. Next, if a field report shows (b) or (c): keep each
range's placement on the handle beside its charge (`RangeCharge`), and have `short_of_the_card`
read it.

#### #256 — On WSL, `nvidia-smi` waits for a busy card, so a reading taken mid-reply is "unknown"
`P3` · memory · **OPEN** — 2026-10-12 · history: #244's fix (archive § "Closed after 2026-10-02" → #244)

On WSL2 with the card busy, `nvidia-smi` waits in the graphics driver. Thread samples show it in
`dxgvmb_send_sync_msg` and `dxgprocess_close_adapter`. In the 12h rig shape (two nodes and their
model processes on one card) a call made mid-reply lived 20.9 s, and the 10 s bound returned
"unknown". Before #244's fix the same call lived 2.4 s, because the wait stalled the node's network
task, the split's work stopped, and the card went idle. Over a week of ordinary use the live node
hit the bound once (2026-10-05, 13 s). What it costs: a capability broadcast advertises no room for a cycle, and admission
charges no other program. If one call is held 60 s, `graphics_driver_not_answering` sends NEW
workers to the processor until it answers. Not seen outside the rig. Next: read memory through
NVML in-process with one handle held open (what `nvidia-smi` does inside, without a process start
and adapter close per reading — the closing is where it waited longest). Then the 12h shape with
the thread sampler (`examples/thread_sample.py`).

#### #252 — `/v1/models` advertises a context the card cannot hold
`P3` · memory · **PARTIAL — (a) parked; (b) (2) fixed 2026-10-18, released in v0.3.244; (b) (1) parked** — 2026-10-12 · history: report #035 (a tester, RTX 3060 12 GB, v0.3.234)

Qwen3-14B on a 12 GB card: `/v1/models` says `max_model_len: 8192` (the declared limit under
`DEFAULT_MAX_SEQ_LEN`), but with the weights loaded the KV budget is 860 MB, and a 6,228-token
prompt was refused (`short_by_mb=259`; usable ≈ 5k). A client (Open WebUI, Home Assistant) learns this only from failed
requests. **Done (2), 2026-10-12, released in v0.3.236:** the refusal now names the longest prompt that fits
("up to about 4608 tokens with this model here", `model_worker::longest_prompt_that_fits`, rounded
down to the cache's growth step and less the reply reserve). **Done (1), 2026-10-14, released in v0.3.237:** while
the model is loaded here whole and no connected holder inside the request's scope could take
part, `/v1/models` reports what this node's memory holds in place of the declared figure — the
longest prompt + reply admission takes however the two divide it
(`api::openai::context_served_here`; the worker reports it as `WorkerMsg::ContextFits`, gated
on `DAEMON_READS_CONTEXT_FITS`, kept on the worker's handle, `ModelProcessPool::context_fits`).
Live, 2026-10-14 (keep on this computer, the peers out of scope): Qwen2.5-14B 30/48 layers on an
8 GB card listed 8,192 before a load and 5,088 while loaded; a 5,055-token prompt with a
32-token reply answered, a 5,775-token one was refused before prefill. Qwen3-30B-A3B (experts in
RAM): 8,192 → 7,136, the same shape (`~/swarmllm-252/ctx252.sh`).
Left: (a) with no worker loaded the declared figure stands — a client that reads the list
before the first load still learns the limit from a refusal; (b) the same figure, gossiped,
would let a planner size this node's and a PEER's share of a split before asking: since #263 a
split's LOCAL segments are admitted together before any peer reads the prompt, and a re-plan
holds this node to what fits, but a peer's segment is still admitted when its forward arrives.
Never shorten the context itself (`arch-worker-memory.md` § "A card's KV budget is kept
whatever the context setting").
Field-confirmed, report #036 (2026-10-14, v0.3.238, the reporter's 12 GB card): Qwen3-14B kept on
that computer lists 8,192 before a load and 5,600 while loaded.
Considered for (a) and NOT done (2026-10-15): keeping the last worker's figure after it unloads
(idle unload, #249, flips the list back to the declared figure). It is a cached live condition
(`arch-worker-memory.md`): once the card frees up it UNDER-reports, and a client that trims
history to the listed context would then silently drop context the node could hold. (a) needs a
pre-load figure from the loader's own arithmetic, not a memory of the last one.
**(a) parked, 2026-10-15**: the clients the report names do not read the listed context. Open
WebUI and LibreChat take it from their own settings; Cursor assumes 1M for a custom endpoint and
detects nothing; Cline's OpenAI-compatible provider is set by hand. Each learns the limit from the
refusal, which already names it (`longest_prompt_that_fits`). A pre-load figure would be a THIRD
estimate of the card's room, beside admission's contribution-scaled budget and the loader's 85% of
free memory (#187's two bases). Its errors fall partly below the truth: the worker's CUDA context
(~137 MiB), whether admission reclaims an idle model (only when the estimate does not fit
without it), and the placement (card, card/processor split, processor). A figure below the truth
makes the few clients that do read it (SDK-style frameworks) drop context silently. Reopen if a
report names a client that sizes prompts from `/v1/models`.
**(b) parked, 2026-10-18**, after a rig and a code reading. What a gossiped figure would fix is
narrower than filed. (1) A processor peer's worker holds conversation room its advertised free
memory does not show — the `typical_kv` inside its reservation (`record_cpu_kv_budget`: budget =
typical + uncommitted headroom; 512 of 660 MB on `split_rig.sh peerkv`, RAM_B tight) — so the
planner under-counts its room for the layers it already holds. Publishing it would have to be
the RESERVED part only (the worker's room less the free memory it overlaps, or the shared
headroom is counted twice when a plan both reuses and adds layers), charged first to the reused
layers' prompt cache. But with one route the under-count changes only which rung finds the plan:
`PeersAtCeiling` sends it anyway, by design, and the peer admits it. (2) The case that costs —
the peer's real room SMALLER than the planner's view, refused after earlier machines have read
the prompt (peerkv's tight case: 503 after up to 712 s) — a gossiped figure cannot fix, being as
stale as the free memory beside it. That needs the peer asked BEFORE the pass, as #263 asks this
node's own segments (`ModelProcessPool::admit_prompt`): an up-front admission message gated on a
feature bit, answered with `layers_that_fit` so the re-plan holds the peer to a fact (vLLM's
`can_allocate` covers the whole request before prefill; Petals prices `cache_tokens_left` as a
routing penalty, never an exclusion), at one round trip per split. Field frequency is low: the
live node logged 11 prompt refusals in a month, all on test days, none since 10-02. Reopen with
a field refusal of a peer's segment, and build the up-front ask (2) before any gossip (1). The
planner also charges a peer the prompt's positions while the peer admits
`kv_cache_reservation(prompt + reply reserve)`; fold that in with (2).

**Field refusal, 2026-10-18 12:17Z (v0.3.243, the reopen condition above).** The live node asked
`qwen3-30b-a3b-instruct-2507-q4-k-m` (19-token prompt, 64 max) with no part held here. No route fit
what the peers advertised, so the planner took the `PeersAtCeiling` rung ("a peer may refuse") and
handed the chain to `aa3d1e66` (AU, RTX 3060, .243), which refused as delegate (layers 35..48 "need
about 4901 MB more than this node has left") and then, driven from here, refused segment 0..3 for
the PROMPT ("0 MB available for conversations … short by 6 MB"; no standby, logged at ERROR). The
re-plan served it on `99aafc41` + `7c10ea04` (IT, BE): 200, ttft 134.7 s (most of it their cold
load), predicted 93 s. The same request warm, 4 min later: AU + BE, ttft 24.8 s, 200. Shape: the
relaxed rung's gamble lost twice on one peer, not a stale room figure under the strict bound — the
up-front ask (2) would have turned both refusals into one round trip each.

**(b) (2) fixed 2026-10-18, released in v0.3.244** (§ Closed): the coordinator asks each peer whose
segment runs after other work before the pass, and a "no" holds that peer to the layers it said
fit on the re-plan. Left: (b) (1), the gossiped figure — parked as above; the planner still
charges a peer the prompt's positions where the peer admits `kv_cache_reservation(prompt + reply
reserve)` (not folded in); and no rig has yet shown "the peer refuses, the re-plan gives the excess
to another, 200" — the clamp it rests on is a scheduler test
(`docs/invariants/memory.md` § "A split's peers are asked about room before the pass too").

#### #187 — Head-room admission prices load and runtime on two different bases
`P3` · memory · **PARTIAL** — 2026-08-08 · history: archive § "Head-room admission: two things the live test found"

The claim arithmetic is fixed (`kv_budget::positions_to_allocate`). `kv_budget.rs` bases runtime
head-room on free card memory (`mem_get_info`, `kv_headroom_bytes`) while the load-time estimator uses the
contribution-derived budget; nobody tried to reconcile them, and no end-to-end refusal under
real pressure was ever constructed. Put both on one number, then occupy card memory before a
load so the estimator passes but the runtime budget binds, and watch the refusal. Narrow
(multi-model or another program on the card); the guard is a backstop.

Measured 2026-10-07 (the v0.3.230 gate's GLM-4-9B step, freshly booted card): admission charged
6514 MB including 4096 tokens of cache (`ADMISSION_KV_CONTEXT`) against a 6688 MB budget and
placed the model WHOLE on the card, while the loader's own head-room then covered only 3262 of
its 8192 tokens (5872 MiB of weights, ~1.1 GB reserve). On a busier card the same build places
35 of 40 layers there and keeps ~8000 tokens. So the two bases disagree in exactly the marginal
case, and a whole-card fit can cost ~60% of the context a slightly smaller split keeps. Same
figures on v0.3.229 and v0.3.230.

#### #122 — Four short chats at once on one card: the fourth is refused, not queued
`P3` · memory · **PARTIAL** — 2026-09-26 · history: archive row #122

Reserve sizing from `max_tokens` and the concurrent-case wording shipped in v0.3.208. In a swarm
the 503 re-plans to a peer; alone it is a retryable refusal. Waiting briefly behind live
requests needs a deferred-admission queue: batched admission runs inside the worker's single
message loop, so a wait there stalls every other chat's decode. Low priority.

#### #123 — Work for peers is admitted per forward, not per request
`P3` · memory · **PARTIAL** — 2026-09-26 · history: archive row #123 and § "Work for peers is admitted per FORWARD, not per request (#123, open, 2026-09-26)"

Mitigated in v0.3.209 by a running-request reserve (`dispatch::admits_a_new_request`); 0
refusals in 9 days of logs. Real per-request admission needs a completion signal: (1) the
coordinator sends it at the ONE point a request is definitively over (after router retries,
which reuse the request id); (2) the serving node then releases that request's cache
(`release_request_kv` reaches only the coordinator's own worker today — this would also free
peer memory 10 minutes early); (3) admission keyed by (peer, request id), with a short timeout
for peers lacking the feature bit. Never key on the sender-chosen `sequence_num`. Also: refused
tensor-parallel forwards are answered on a channel the AllReduce collector never reads (10 s
stall per layer; TP off by default), and `VisionEncodeResponse` has no error field, so an
image-encode refusal stays silent.

### API surface

#### #312 — A Responses tool output that is a list of parts is refused
`P3` · api · **OPEN** — 2026-10-19 · history: found closing #310

OpenAI's Responses API lets a `function_call_output`'s `output` be a list of `input_text` /
`input_image` / `input_file` parts as well as a string; `FunctionCallOutputItem::output` is a
`String`, so such a request is refused with a 400 naming the type (never silently dropped), and a
tool that returns a screenshot through `/v1/responses` cannot hand it over. Fix: accept both
shapes, translate the list through `collect_content_parts` into the tool message's parts (the chat
surface carries a tool message's pictures, the Anthropic one since #310), and the Responses →
Anthropic bridge (`anthropic_bridge.rs`) the same. Separately, the cloud-provider translation of an
Anthropic request (`anthropic::handlers::anthropic_messages_to_openai`) writes `[image]` for a tool
result's picture, because OpenAI's chat API refuses images in a tool turn; a user turn carrying them
right after the results would hand them over.

#### #185 — Three API surfaces each implement streaming and non-streaming replies separately
`P3` · api · **OPEN** — 2026-07-26 · history: archive § "Collapse the parallel response paths behind one core loop"

OpenAI (`api/openai/streaming.rs`), Anthropic (`api/anthropic/handlers.rs`) and Responses
(`api/openai/responses/stream.rs`) are 4,610 lines between them (3,679 when filed) — the shape
behind this repo's most recurring defect, a rule fixed on one surface and forgotten on another
(11 recurrences in July). The proposal: one generator loop emitting neutral events
(TextDelta / ToolCall / Usage / Finish) with three thin serialisers. Needs its own release and a
live A/B on every surface (OpenAI stream / non-stream, Anthropic stream / non-stream, Responses
foreground / background, MCP). Until then: choke-point helpers (`finalize_reply_text`,
`strip_prefix_in_body`) and `.claude/rules/architecture.md` § "One invariant, N paths".

#### #141 (b) — The Anthropic surface never shows a reasoning model's scratchpad, even when the request enables thinking
`P4` · api · **OPEN** — 2026-10-19 (the OpenAI half closed) · history: archive row #141

OpenAI's chat surface now carries a finished scratchpad both ways (`message.reasoning_content`,
`delta.reasoning_content`); `/v1/messages` withholds it on both, consistently
(`ReasoningOut::Withheld`, `InferenceOutput::reasoning` unread). Anthropic's shape is a
`thinking` block, emitted only when the request sets `thinking`, and it carries a `signature`
this server cannot produce; llama.cpp's server emits one before the text under
`--reasoning-format deepseek`. Next: confirm what `api/anthropic` does with a `thinking` block a
client sends back (Claude Code does), then emit one for a request that enabled thinking, from the
same `InferenceOutput::reasoning`, streamed as `thinking_delta`.

#### #188 — MCP `node_info` repeats the whole cloud-model catalogue the `models` tool also returns
`P3` · api · **OPEN** — 2026-08-12 · history: archive § "MCP `node_info` duplicates the whole cloud-model catalogue"

~2,300 tokens per orientation (~8.5k in a session) for MCP clients. Replace the list with the
count, a ~10-item `cloud_models_sample` and a pointer to the `models` tool, keeping an empty
list distinguishable from an absent one (a provider key that never loaded). `docs/CREDITS_DESIGN.md`
also wants `node_info`'s credit balance decided — do both in one pass.

#### #203 — A streaming client cannot read time-to-first-token or time-per-token
`P4` · observability · **PARTIAL** — 2026-07-26 · history: archive § "Observability: routing, performance and per-node attribution"

`RequestTrace`, the `Server-Timing` header, diagnostics and rollups shipped. A streamed reply's
headers flush before the body, so they carry the route but not the timings; only the
dashboard's final usage event has them. If a client asks: an extra field in the final usage
chunk, without breaking the OpenAI/Anthropic wire shapes.

### Dashboard and first-hour experience

#### #62 — The Network map is blank on a node that has not served across regions
`P3` · ux · **PARTIAL** — 2026-09-13 · history: archive row #62 and § "The Network map is blank on a node that has not served across regions"

The honest half shipped: the map has a key for its arcs and says, on a node that has run
nothing, why it is empty. Arcs come only from this node's own `recent_routes`, so a new user
sees pins and no traffic. Seeding from the swarm's traffic needs a new gossiped route summary
(an additive message and a feature bit) AND a privacy decision, since it would publish who
served whom. Never fake it with sample data.

#### #176 — Cross-pool inference has an admin-API switch but no dashboard control
`P3` · ux · **PARTIAL** — 2026-07 (R134) · history: archive § "Inter-pool model sharing policy"

Discovery and opt-in routing between pools shipped (R134-R134.7:
`PoolModelAvailability`, `foreign_pool_catalog`, `pool.allow_cross_pool_inference`), and R137
made the setting live through `PUT /api/admin/config` (`allow_cross_pool_inference`); no
dashboard control or consent banner exists. Add the toggle with its consent wording (i18n ×21).
Billing between pools waits on credits, which are dormant (`docs/CREDITS_DESIGN.md`).

#### #261 — A model's manifest carries the file's own name and placeholder metadata
`P3` · ux · **OPEN** — 2026-10-13 · history: none (a tester's report, 2026-10-13)

`model::manifest::build_manifest_from_gguf` takes `name` from the GGUF's `general.name` and
fills `quantization` (`Q4KM`), `num_params_billions` (0.0) and a coarse `architecture` as
placeholders. Seen on a tester's node: `qwen3-14b-q4-k-m` is named "Qwen3 14B Awq Compatible
Instruct" (what its uploader wrote into the file), and that name is what refusals print;
`qwen3-30b-a3b-instruct-2507-q2-k` reads `Q4KM`; both read `Qwen2` and 0.0 B. The one reader of
`quantization` (`auto_manage::quant::parse_quant_from_manifest`) already prefers the id's tag
over the placeholder, so nothing routes on it. ⚠ Every field is inside `compute_hash`: filling
them changes every model's manifest hash, so check what a new manifest version does across
the swarm (`manifest_heard`, `model::canonical`) before changing the builder. The cheaper
half is the display name: derive it from the id or the HF repo where the file's own name
disagrees with the id's size and quant tags.

### Packaging, release and operations

#### #241 — llama-cpp-2 is pinned at 0.1.157; 0.1.158 moved tokenisation onto the vocabulary
`P4` · build · **OPEN** — 2026-10-08 · history: Dependabot PR #34 (its llama compile-check failed)

`llama-cpp-2` 0.1.158 — a PATCH release — removed `LlamaModel::str_to_token`, `token_eos`,
`token_to_piece` and `AddBos`; they live on `LlamaModel::vocab()` now (`LlamaVocab::tokenize(bytes,
add_special, parse_special)`, `eos()`, `token_to_piece(..)`). Twelve call sites in
`inference/executor.rs` and `inference/pipeline/speculative.rs` (llama-gated). Pinned `=0.1.157` in
`Cargo.toml` and ignored in `.github/dependabot.yml` so the weekly grouped bump stops failing.
Moving: map each call to the vocabulary API keeping the old semantics (read 0.1.157's
`str_to_token` for what `AddBos` and special-token parsing meant), then build `--features cuda`
(a new llama-cpp-sys rebuilds llama.cpp's CUDA kernels — cold, about an hour here) and run the
gate's llama-path steps on the artifact.

#### #311 — RHEL 9 and its rebuilds cannot run SwarmLLM natively
`P3` · packaging · **OPEN** — 2026-10-19 · history: found closing #163

The Linux binaries are built on ubuntu-22.04, so they need glibc 2.35 (`libm.so.6(GLIBC_2.35)`
is in the .rpm's automatic requirements); RHEL 9.8 has 2.34. Measured in `ubi9/ubi`: `rpm -i
--test` refuses the .rpm with that dependency (a clean refusal, not a broken service), and the bare
binary would not start either. RHEL 10.2 (2.39) and Fedora 44 install it. Rocky / Alma 9 are the
same as RHEL 9, a large share of servers. The installation guide now says so and points them at
Docker. A fix builds the Linux assets against an older glibc — `cargo zigbuild --target
x86_64-unknown-linux-gnu.2.28` (what `memory/env_proxmox_crossbuild.md` uses) or a manylinux_2_28
container — for the CPU AND CUDA builds, each run on RHEL 9 before it ships.

#### #183 — A Windows GPU node's auto-update never refreshes the bundled CUDA libraries
`P3` · packaging · **OPEN** — 2026-08 (v0.3.20 asset audit) · history: archive § "Windows-GPU auto-update carries stale CUDA redist DLLs"

`update.rs` swaps only the executable; the CUDA redistributable DLLs ship in the `-gpu` zip
only. Safe while the Windows toolkit stays pinned to 12.x (`gpu-build-env`), and a silent
failure for every auto-updated Windows GPU node the day the CUDA major is bumped. Before
touching that pin: teach the updater to fetch and unpack the zip's DLLs (multi-file apply with
rollback), or put the CUDA major in the asset name and refuse a cross-major auto-update. Add a
pointer to this entry beside the `cuda-windows-version` default in
`.github/actions/gpu-build-env/action.yml` (the single definition of both toolkit pins).

#### #109 — The bundled Prometheus/Grafana stack scrapes nothing
`P3` · observability · **OPEN** — 2026-09-24 · history: archive row #109

`monitoring/prometheus.yml` targets `localhost:8800`, which inside the container is the
container; `docker-compose.yml` maps `host.docker.internal` but nothing uses it; and `/metrics`
is exempt from the API key only for loopback, so a bridge-network scrape gets 401 (the file's
comment now says so, and the book documents `credentials_file`, but the compose stack never
mounts one). Pick one: a compose-only scrape config targeting
`host.docker.internal:8800` with a bearer `credentials_file` mounted from the node's api_key
(its path differs per OS), or `network_mode: host` for Prometheus (Linux-first). Needs Docker
to verify. Keep `prometheus.yml` correct for bare-metal Prometheus.

#### #199 — Two major-version dependency migrations Dependabot will not do
`P4` · maintenance · **OPEN** — 2026-09-02 · history: archive § "Major-version dependency migrations Dependabot may not do"

`rand` 0.8 → 0.9 (the tree carries three copies; `thread_rng` → `rng`, `gen` → `random`,
`distributions` → `distr`) — do it when touching the RNG call sites anyway. `reqwest` 0.12 →
0.13 (HF downloads, the provider proxy, the update checker) — read its changelog (TLS backend,
timeouts) first, keep the inactivity-timeout rule, and re-run each path against real
endpoints. `candle-*` stays on 0.10 deliberately (§ "Decided").

### Security, privacy and credits

Credits are DORMANT — they gate nothing, and `credits_stay_dormant` fails the build if one
starts. Read `docs/CREDITS_DESIGN.md` before touching them; the credit items below are
preconditions for ever switching enforcement on, not work for now.

#### #317 — Another program can hold `[::1]` on the node's port while the node runs
`P3` · api · **OPEN** — 2026-10-20 · history: found with #255's key proof

The HTTP server binds IPv4 only (`api::server::api_bind_addr`: `127.0.0.1` or `0.0.0.0`), so
`[::1]:8800` stays free for any local program. A browser opening `http://localhost:8800` usually
tries `::1` first and would load that program's page under the dashboard's own origin — its
saved chats (`swarmllm_sessions`) and any key typed in by hand (`swarmllm_manual_api_key`) are
that origin's storage. The client commands no longer use `localhost` (#255, `cli::daemon_base`).
Next: also listen on `[::1]:port` (loopback only, IPv6-only socket, best effort where a host has
no IPv6), never `[::]`, which would newly expose the API on every IPv6 address; check
`RequestOrigin::is_this_machine` and `dashboard_trust::classify` treat `::1` as this machine.

#### #301 — The router's per-conversation table (`KvCacheManager`) decides nothing
`P4` · cleanup · **OPEN** — 2026-10-17 · history: found by #251 (2b)'s split rig, whose control turn never reused a route

The router keeps a table of conversations that send `session_id` (`inference::kv_cache`,
`register_multi_turn` / `check_multi_turn_reuse`), saved to the `kv_sessions` tree at shutdown.
Every entry is recorded with an EMPTY route and nothing fills it in, so its `cache_holders` are
always empty and the "a holder went away" check can never fire; the hit/miss answer
(`cache_start_pos` in `router::mod`) only decides whether to re-register and is logged — it is
never handed to the request. The reuse that works is the worker's own prefix cache, keyed by the
prompt's tokens. The one branch that read a recorded route (`distributed_exec`'s "KV cache
affinity") could not fire and was deleted with #251 (2b). Next: confirm nothing else reads the
table (`UpdateCacheTokens`, the persisted `kv_sessions`, the diagnostics), then delete it or make
it record the route it claims to — deleting is likelier right, since the worker cache already
answers "where is this conversation's KV" by content.

#### #300 — A sender can fill the replay-memory share of the relay it sends through, refusing that relay's honest bodies here
`P3` · security · **OPEN** — 2026-10-17 · history: the second `security-reviewer` pass on #299

The relayed-body replay memory's per-network share (#299) keeps a flood from one network off
OTHER deliverers, never off the honest relay carrying it. A peer connected to the anchor sends
32,768 signed bodies addressed to this node from two minted origins (16,384 each, the per-origin
share); the anchor forwards them at its per-origin limit (`RELAY_FORWARD_MAX_PER_WINDOW`, 2,000
per 10 s, `daemon/state/relay.rs`), so in ~82 s the anchor's share here is full, and every body
the anchor relays to this node is refused for 5 min, again whenever the sender repeats it. Since
v0.3.241 (#298); #299 does not change it. Only relaying stops; direct paths are untouched. A
smaller per-origin share does not help (origins are free), and a per-target cap at the relay
only moves the refusal there. Next: have the relay count what it forwards per (sending peer's
`NetGroup`, target), so one network cannot push more than a fraction of a target's share through
one relay; size it against an honest split decode (~25 forwards and results a second per
stream). Wire → `security-reviewer` on the diff.

#### #30 — The plaintext-prompt trust bar sits exactly at the score an unknown peer starts on
`P2` · security · **OPEN** — 2026-09-10 · history: archive row #30 and § "The plaintext-prompt trust bar sits exactly at the score an unknown peer starts on"

`DELEGATE_MIN_TRUST == DEFAULT_TRUST == 0.5`, so a peer never observed — any fresh identity —
clears the bar that decides who may be handed a user's prompt in cleartext; the bar excludes
only peers with recorded misbehaviour. The equality is pinned by a test on purpose (the
coincidence must be re-affirmed, not inherited). Options: (2) separate "starting score" from
"required score" constants first — the smallest change; (1) raise the bar to
`DEFAULT_TRUST + 0.05` with a fallback when nobody is eligible (a small swarm could make a model
unservable); (3) require n observed well-formed results (BOINC-style adaptive replication). The
trust ratchet is already fixed (`a_peer_whose_output_is_always_malformed_gains_no_trust`).

#### #175 — Pipeline result sealing is never active for remote segments
`P4` · security · **OPEN** — 2026-08-21 · history: archive § "Pipeline seal — never active for remote segments; re-enable deliberately"

`dispatch/layer_forward.rs` calls `seal_layer_result(&mut result, None)`; `docs/ARCHITECTURE.md`
lists "Tier 2: Pipeline Sealing — NOT ACTIVE". Result token ids travel inside link encryption
(`network.enable_encryption`, default on) but not sealed end to end. To enable: unseal on every
coordinator result path (the decode steps in `distributed.rs`, not only `prompt.rs`), decide
whether one-hop forwards carry the `0x07` reply-to trailer (feature-bit gated), and measure the
per-token cost. First decide whether it buys anything over link encryption — the boomerang is
structural, not cryptographic, and peers see hidden states regardless.

#### #201 — Credits for hosting parts are self-attested
`P4` · credits · **OPEN** — 2026-07-28 · history: archive § "Shard-hosting credits are self-attested — no proof of storage"

`credit/ledger.rs` earns hosting credit from the node's own registry; nothing challenges it, so
a patched client can claim storage it does not hold. `docs/CREDITS_DESIGN.md` does not cover
proof of storage. Options recorded: random-range challenge-response (`credit/anti_gaming.rs`),
proof of retrievability / data possession, proof of replication, or paying only for observed
service. Two adjacent gaps: a Sybil can reset its trust with a new key, and escrow token counts
come from peers (#202). Belongs in `CREDITS_DESIGN.md` § 6's exit criteria.

#### #202 — A serving peer's own usage figure prices the request it served
`P4` · credits · **PARTIAL** — 2026-07-29 · history: archive § "A serving peer sets the price of the request it served"

`remote_generate.rs` adopts the peer's `usage.completion_tokens` (replaced by the coordinator's
delivered count only when the stream was truncated); `router` builds the cost from it and
escrow release clamps only at 0. Cheap bound: clamp to the request's `max_tokens`, prefer the
coordinator's own prompt-token count, and always take the completion count from tokens the
coordinator received (`stream.emitted()`). Belongs with credits enforcement.

#### #268 — Two DNS advisories wait on the libp2p 0.57 upgrade
`P3` · security · **OPEN** — 2026-10-14 (the v0.3.237 gate's `cargo audit`) · history: `SECURITY.md` rows RUSTSEC-2026-0118/0119

`hickory-proto 0.25.2` (an NSEC3 proof loop, an O(n²) name-compression cost) reaches the daemon
only through `libp2p-dns` (bootstrap multiaddrs at startup) and `libp2p-mdns` (link-local), which
is why `SECURITY.md` accepts them. libp2p 0.57.0 (2026-09-11) ships `libp2p-dns 0.45` on
`hickory-resolver ^0.26`, the fixed line. The upgrade is not a version bump: the vendored
`libp2p-request-response` patch must be re-applied to 0.57's crate, the direct `libp2p-gossipsub`
pin must move with the facade (two copies otherwise, see CLAUDE.md), and a 0.56 node must still
talk to a 0.57 one. Next: read 0.57's CHANGELOG for wire changes (gossipsub, identify, yamux),
then upgrade on a branch and run `split_rig.sh` with one node on each version.

### Parked — waiting on field data, a reporter, hardware or credentials

Open, but nothing to build until the named evidence arrives. Write field reports with their
leading zero (report #033 ≠ FUTURE_WORK #33).

#### #232 — A 14B split across a processor peer and an RTX 3060 took 48 s and gave no tokens
`P3` · routing · **PARKED** (field data) — 2026-10-07 · history: the report behind #230 and #231

A tester's `qwen3-14b` request on v0.3.229 split `4a3ac72e` (processor) L0-11 + `aa3d1e66` (RTX 3060)
L11-40: 48 s, 0.0 tok/s, where the processor peer alone gave 5.9-6.2 tok/s (95 s to a first token
when cold). No logs from either node. Two v0.3.230 changes bear on it — a computer that must first
load the model is waited for (+240 s) and charged for the load in routing (#129) — and a WAN split
is slow by design where the machines are far apart (`docs/plans/faster_than_local.md`). Re-check the
same shape once both peers run v0.3.230 (the coordinator's `pipeline candidate … cold_load_ms=` lines
and `DIAG: request complete`) before treating it as a defect.

#### #227 — An advertised cloud model answered 404 in ~1 ms, and the cloud catalogue read 0 until a restart
`P3` · api · **PARKED: unconfirmed since 0.3.224-0.3.228** — 2026-10-06 · history: a tester's report, 2026-10-06

On a tester's CPU node (since retired, with the provider key): `cloud_models_available` read 0 for
days and 80 after a restart (stale cached state that reads exactly like a dead API key). With 80
advertised, `mistralai/mistral-large` and `deepseek-ai/deepseek-v4-flash-0731` answered HTTP 404
in ~1 ms with a body listing only local models — never reaching the provider; `openai/gpt-oss-20b`
routed but returned `status: ok` with empty content after spending its whole token budget (a
reasoning model's output may have gone to its scratchpad). Needs a node with a provider key to
reproduce: compare `/v1/models` with what the router resolves for each listed cloud id.

#### #90 — The message dispatcher once stopped consuming for 45 minutes while the node reported itself healthy
`P2` · reliability · **PARKED: waiting for a recurrence** — 2026-09-18 · history: archive row #90, `memory/next_up.md` § "#90 — what is ELIMINATED"

Every inbound swarm message shares one consumer, so a stall is a total outage. Now detected
(`dispatch_stalled_for` / `DISPATCH_STALL_AFTER`, the liveness marker names the last message
kind — `last_dispatch_kind`, at `error!`) and the node withdraws its capacity, but the stall
itself is unexplained. The DashMap-guard-across-await fix (`clippy.toml` lint) fits best and is
unconfirmed; no genuine recurrence since 2026-09-24 (the 09-23 and 09-25 firings were a
whole-PC freeze and a host network loss). Eliminated, so not re-derived: `cancel_request`
(bounded in .189, stall recurred on .189), the manifest-arm lead, the key-rotation coincidence,
blocking DB calls (fixed in .191, fired 0 times). ⚠ Two arms log below the node's level, so the
last visible line before a wedge can be unrelated. `release_request_kv` has the same unbounded
shape but runs in a router task, and the supervisor still cannot see a hung (non-exited) task.
Field reading, report #036 (2026-10-14): 0 "The message dispatcher has taken nothing off its channel" lines in a
tester's six days of logs (0.3.229 → 0.3.238).

#### #49 — Auto-prune reportedly evicted parts the swarm held no other copy of
`P2` · storage · **PARKED: one line from the reporter's log** — 2026-09-11 · history: archive row #49

Not reproduced; do not fix on reasoning alone. Prune treats a holder claim as proof of
redundancy (`can_reacquire`) — an over-count is unsafe, an under-count only conservative. Two
changes on 2026-09-17 shifted the priors (a holder is claimed only after VERIFIED; the budget
measures the directory, with an orphan reclaim pass). Needed: whether `Peer returned empty shard
data` precedes the stalls in the reporter's log (a stale DHT provider record — see #177) or
nothing from the peer does (a vanishing request). If it recurs, check the new directory-based
disk pressure against the orphan reclaim pass first.

**The reporter's fuller account, 2026-10-06** (his CPU node, ~0.3.173, since retired):
`auto_manage.max_storage_mb = 8000` with 7839 held, `resource_pressure=0.98 pressure_urgent=true`;
afterwards gemma-2-2b-it lacked shard 0 and qwen2.5-0.5b its shard 2 — two models he believes it
was the sole holder of. Re-fetching gemma's shard 0 looped `stalled shard download — cancelling +
retrying … stall_secs=30` against two different peers, both stalling rather than refusing. He
recovered it through `/api/admin/hf/download-shards`. The question above is asked of him again
(he offered logs). Since then, on the code side: a sole live holder cannot be shed (`sheds_at`
needs more live holders than a target that never drops below `min_replicas`), and the stall
loop is the shape of a stale provider record.

#### A disputed shard is kept but the disagreement is never settled (#61)
`P3` · storage · **PARKED: one field reading, none disputed** — 2026-09-13 · history: archive row #61 and § "A disputed shard is kept but the disagreement is never settled"; report #036

Fallout of #60 and strictly better than what it replaced (deleting good data): a part that
disagrees with an unbacked hash is kept, served and recorded (`ModelRegistry::bytes_disputed`
since 2026-10-03, with the hash the bytes DO have, which is what the node now announces for
the part; guard `every_path_that_keeps_disagreeing_bytes_records_the_dispute`; a "Disagrees"
badge and a diagnostics line), but nothing settles it. For a model whose canonical upload is
known the heal now re-checks a part that falls into dispute against the upload and replaces
the copy if its bytes are not the upload's; a dispute whose bytes pass that check stays open. Do not build before a reading: look for `-- shards
kept despite disagreeing --` in a diagnostics report from a peer-fed node. The one-upload rule
(#151, v0.3.221) should make disputes rarer. Designs: (1) self-attestation — free, but makes a
node refuse to converge on the swarm's dominant build; (2) settle from the origin without
deleting first, under a NEW marker (not `shards_needing_repair`, whose drain clears marks for
on-disk files), rate-limited, requiring an `hf_source`. A second narrow gap: the startup sweep
keeps and advertises such a part while the auto-manage rescan skips registration, so the file
sits unserved with its storage spent.
**The reading, 2026-10-14 (report #036):** a peer-fed node on v0.3.238 holding 169 parts of 29
models — `shards kept despite disagreeing (0) — none`. One node, zero disputes: still nothing to
build; read the line again in the next peer-fed report.

#### #32 — A long prompt on the boomerang path died with a card out-of-memory mid-compute
`P3` · memory · **PARKED: the reporter's card** — 2026-09-09 · history: archive row #32 and § "A long prompt on the boomerang path dies with a CUDA OOM mid-compute"

The mechanism is fixed (the cache grew by `Tensor::cat` in 512-position steps; now reserved up
front, `kv_cache::set_reserved_positions`, A/B via `SWARMLLM_KV_RESERVE`). The field OOM was
never reproduced — an 8 GB card here cannot reach 20k tokens. Closes when the reporter (or a
larger-card rig) re-runs a 20k-token prompt on v0.3.208+ with `growth_steps` 0 in the debug line
and no mid-forward `CUDA_ERROR_OUT_OF_MEMORY`. Still unmeasured: activation and workspace
memory during the prompt pass is not part of what admission charges.

#### #36 — Qwen3-8B with tools reportedly emits Harmony tokens (`<|start|>`/`<|end|>`)
`P3` · model-support · **PARKED: the reporter re-testing** — 2026-09-10 · history: archive row #36

Not reproduced; nothing in `src/` emits those tokens. #35 (a model's own tool framing was never
rendered) fixed the likely cause in v0.3.171. Needs the reporter on a current release — and, if
it survives, their model file and a request log. Close if no reply.

#### #82 — The iOS on-screen keyboard has never been exercised against the locked page shell
`P3` · ux · **PARKED: an iOS device or the #034 reporter** — 2026-09-14 · history: archive row #82

Raised by the #81 fix (the shell is locked to `dvh`, the chat input at the bottom). There is no
WebKit engine here to test. Ask the report-#034 tester whether the input stays visible with the
keyboard up; only if not, try `interactive-widget=resizes-content` in the viewport meta.

#### #200 — The OpenClaw provider plugin is built but not published
`P4` · packaging · **PARKED: the user's ClawHub/npm credentials** — 2026-09-02 · history: archive § "OpenClaw provider plugin"

`integrations/openclaw/` (tests and its own CI workflow) shipped in v0.3.148; the config-only
path works today. Publish with `npm exec clawhub -- package publish .` (trusted/OIDC publishing
would need the plugin in its own repo via `git subtree split`), then ask upstream about
bundling. Independent leftovers: `/v1/embeddings` answers 501 by design until this node runs an
embedding model (`LocalEmbedder` holds only the token table and is NOT one — the doc comment on
`openai::embeddings` says why), and the 8192 default context wants a decision (#189 helps).

### Ideas — researched, not scheduled

Each has a reason it is not being built now. Pick one up only when its trigger appears.

- **#204 — A trained draft head (EAGLE-3)** (archive § "Tier 1" → C). 3-6× decode on supported
  models, but needs externally trained heads per model, a manifest extension to distribute
  them, and a hosting policy. Local GPU decode is already ~97% of llama.cpp. Trigger: decode
  speed against llama.cpp becomes the priority again.
- **#205 — Stream activation deltas instead of activations** (archive § "Tier 3" → H, and
  SWARM-SPEC Layer 4). 2-4× fewer bytes on the wire, but WAN splits are bound by round trips,
  not bytes (`faster_than_local.md`). Trigger: evidence that bytes bound a real WAN split.
- **#206 — Overlap send and compute inside one forward (Tier 4K)** (archive § "Tier 4" → K).
  The chunked send shipped in R139 behind `inference.streaming_chunked_send`, default OFF and
  never field-tested; request-response-path chunking (~50 LOC), the WAN bench and worker
  row-tiling remain. Trigger: `examples/3node_inference_bench.sh` with the flag on and off at
  5/25/100/200 ms between two real networks — if it never wins, delete the flag and its code.
- **#207 — FP8 activations on the wire** (archive § "Tier 5" → L). Needs Hopper/Blackwell
  hardware; consumer cards have none. Trigger: FP8-capable nodes in the swarm.
- **#208 — Size parts to each node's capability** (archive § "Adaptive shard sizing from node
  capability"). Part layout is global, so per-node sizing needs a negotiated layout or sub-part
  ranges inside one manifest — and must keep every node on the SAME upload (#151). The
  prerequisite contiguity bonus exists; measure whether it keeps pipelines shallow first.
- **#209 — Stream layers through the card for a model too big for it** (archive § "Rolling
  shard load — stream layers through the card"). Weeks of work: a layer-major streamed prompt
  pass reusing per-layer placement (`hybrid.rs`), measured at 14k tokens with
  `examples/prefill_bench.rs`; decode streaming pays only with wide speculation (#194's f16
  card cache, its prerequisite, shipped in v0.3.232).
- **#210 — A quantized matrix-matrix kernel (tinyBLAS-style)** (archive § "Why the quantized
  matmul plateaus at ~1.5x…"). Batching plateaus at ~1.5× because there is no quantized GEMM;
  dequantize-once measured 11.8× worse. A hand-written SIMD kernel per format and instruction
  set, aimed at the prompt pass, not decode batching.
- **#211 — Runtime AVX2 dispatch in vendored candle** (archive § "CPU nodes ship with the fast
  quantized kernels compiled out"). Since 2026-08-06 every x86 asset is `x86-64-v3` with a
  `-baseline` sibling the updater picks (`update::host_asset_name`); one binary would mean
  carrying a SIMD patch (~21 functions in `quantized/avx.rs`, ~8 sites in `k_quants.rs`) across
  every candle upgrade (upstream huggingface/candle#1818 is open). No baseline .deb/.rpm
  exists. Trigger: reports from pre-2013 CPUs.
- **#212 — Cost a model swap on the card instead of an idle-time floor** (archive § "Cost the
  GPU swap properly, instead of using idle time as a proxy"). `VRAM_MAKE_ROOM_MIN_IDLE_SECS`
  (5 s) already banked 12×; replace it with load time versus the expected processor cost over
  the next turns only with both figures measured on a machine where they differ from the dev
  laptop (a first attempt recorded processor cold starts and was reverted).

## Decided — not doing

Each was researched or measured and decided against, or deliberately left until a trigger
appears. Listed so sweeps and future sessions do not re-file them; the reasoning is in the
archive under the named heading. Reopen one only with the evidence its line names.

**Engine and speed**
- **Q4_K block-interleaved repack** — the 1.8× did not survive our own kernel (ceiling
  1.24-1.27×). Archive § "Q4_K block-interleaved repack: the 1.8x does not survive our own kernel".
- **Ragged batching** — measured +25% (processor) / +23% (card) at batch 4; decode batching was
  unblocked differently. Archive § "Ragged batching — spec, and the measurement that says don't
  build it yet".
- **Lookahead decoding (the LMSYS 2-D window)** — its compute win shipped as prompt-lookup n-gram
  speculation; the residual waits for a workload where it matters. Archive § "Tier 2" → E.
- **A true radix tree for prefix sharing** — cross-request sharing shipped (block-hash chain,
  `split/prefix_cache.rs`); copy-on-write radix storage is weeks of paged-KV work for a P2P
  load pattern that does not need it. Archive § "Tier 3" → G.
- **Activation sparsity (PowerInfer / DejaVu)** and **pre-emptive layer dispatch** — research
  projects of months with model-specific tuning, no production precedent for the second.
  Archive § "Tier 3" → I, § "Tier 4" → J.
- **Hedged dispatch of slow hops** (survey Tier 1B, SWARM-SPEC Layer 2) — built and REMOVED
  2026-09-24: a hedge runs on a machine that never saw the conversation (#94).
- **Decode quantized matmuls on the processor, 7-19% behind llama.cpp** — what remained of #119
  after prompt reading reached parity; not pursued while card and split work rank higher.
- **The same model at fewer bits as the drafter** (#144) — 94% agreement, but no design may need a
  user to hold a whole model, not even a low-bit copy (user, 2026-09-28). At most a possible
  opt-in. `docs/plans/split_speculation.md` § "Draft on your own card, check on everyone else's".
- **Disk speed does not bias routing** — verified that nothing the scheduler reads comes from a
  disk benchmark; only the cold first load is unpriced (`COLD_MODEL_LOAD_ALLOWANCE_SECS`
  widens a timeout). Archive § "Not bugs, and deliberately not ranked".

**Network and distribution**
- **Separate LAN and public peer caches** — the read-time `filter_dialable` (R148) fixed the
  report; revisit only if roaming laptops reconnect poorly. Archive § "Separate LAN and public
  peer caches".
- **Pear (Holepunch) over-the-air channel** and **peeroxide / Hyperswarm as a transport** — the
  real friction (old glibc) was fixed by building on ubuntu-22.04; the signed GitHub-Releases
  updater stays the channel, and the relay covers CGNAT. Archive § "Pear (Holepunch)…", §
  "peeroxide / Hyperswarm…".
- **A per-peer backoff for failing request-response sends** — its only reproduction was a
  Windows-firewall inbound block, not an unhealthy peer, and that evidence argued against
  building it as specified. Reopen with a peer that fails independently of a directional
  transport quirk (model it on `shard_download_backoff`, and make the cooldown expire). Archive §
  "A repeatedly-failing peer is retried indefinitely with no backoff".
- **A quorum on the part hashes a coordinator holding none of a model adopts** (#159's residual,
  decided 2026-10-11) — such a node fills its zero hashes from the first canonical-SHAPED
  manifest a peer gossips (`register_manifest` → `merge_known_shard_hashes`). Shape is total
  size, every part's size and each part's first tensor at its ABSOLUTE file offset
  (`tensor_data_offset + offset`, so a header one byte longer fails it; HuggingFace uploads of
  one model differ by hundreds of bytes). A node on v0.3.223 or later never gossips a copy of
  another upload, and a part fetched against a wrong hash fails the byte check at accept. What
  is left needs, at once, a holder on v0.3.222 or older (6/6 peers on v0.3.234 at 12:20 UTC
  2026-10-11; auto-update moves a node in minutes) AND an upload of the same layout to the
  byte with different contents. A quorum would cost a provenance per adopted hash for a case no
  node can now produce. Reopen with a peer on v0.3.222 or older holding a model, or two uploads
  of one model whose shapes `CanonicalBuild::describes` cannot tell apart. Archive row #159.
- **Disk contraction below an operator's `min_replicas`** — idle card unload shipped; shrinking
  below the operator's floor is intentionally not built. Archive § "Demand-driven resource
  management — VRAM done, disk contraction deferred".

**Product scope**
- **GGUF conversion / fine-tuning from PyTorch checkpoints** — out of scope for a Rust daemon;
  users bring a converted GGUF. Archive § "GGUF conversion / fine-tune support".
- **A local model-quality eval harness gossiped into wishlist scoring** — a subsystem with
  governance questions nobody is asking. Archive § "Model quality benchmarking".
- **Generic `SWARMLLM_<SECTION>_<KEY>` environment overrides** — only seven named variables exist
  on purpose (#722); add one when a headless deployment needs it. Archive § "Generic
  `SWARMLLM_<SECTION>_<KEY>` environment overrides".
- **A request's `seed` on local models**, **per-token logprobs from local inference** and the
  other "won't fix unless a caller appears" API items — `docs/ARCHITECTURE.md` § "Deferred
  Items". (A greedy reply that is not reproducible is a defect, #179.)
- **Enforcing the `MCP-Protocol-Version` header** — would 400 newer clients; supported versions
  stop at 2025-11-25. Archive § "MCP reports every tool failure…" (its last subsection).
- **Credits never move between nodes** — by design while credits are dormant;
  `docs/CREDITS_DESIGN.md` carries the finding and the exit criteria. Archive § "Credits never
  move between nodes".
- **An emptied reply as an error** (#186, decided 2026-10-19) — a reply finalisation empties
  stays HTTP 200 with `content: ""`, as OpenAI, vLLM and llama.cpp's server answer an empty
  generation; an error would fail a legitimate reply too (a reasoning model whose answer is all
  scratchpad, #141). The WARN in `finalize_reply_text` and `swarmllm_empty_replies_total` stay the
  diagnosis: 0 on the live node, and no "reply is empty after finalisation" line in its log since
  2026-09-17. Reopen with a field node's counter above 0 and the template it names.

**Code and process**
- **`manifest.json` lost update across read-modify-write sites** (#66) — the corruption half is
  fixed (unique staging name + lock); the lost update is covered by the registry merge at boot.
  Reopen only with evidence.
- **The self-update staging file shared with a manual `swarmllm update`** (#67) — deliberate (the
  apply hand-off); `apply_update` re-hashes before the rename. A cross-process lock is not worth
  it on this evidence.
- **A release CUDA build that runs cold after a new stable rustc** (#154) — remedy procedural
  (check Cache warm ran today's stable before tagging, `memory/release_gate.md`); pinning the
  toolchain would hide new lints until someone remembers to bump it.
- **Release build time beyond ~16 min** — sccache, larger Windows runners or dropping the
  baseline Windows variant only if it becomes a problem. Archive § "Release build time: what is
  left after the 2026-08-17 fix".
- **`batch_scheduler_loop` and the router's batch tasks keep no JoinHandle / catch_unwind** — a
  panic degrades to direct execution; won't fix until a concrete panic site appears. Archive §
  "Audit deferral — R128 sweep-log triage", § "Batch JoinHandle discard (wontfix)".
- **"Shard file missing on disk — skipping registration" once per boot per missing part** —
  startup cannot tell never-held from vanished; a per-model summary is the optional quieting.
  Archive § "Log noise observed on a live node".
- **`candle-*` held at 0.10** — vendored and patched; bump only when upstream carries something
  we need. Archive § "Major-version dependency migrations Dependabot may not do".

## Closed

Every number that is no longer open, with how it closed. Numbers 6-9, 13-16 and 19-28 were
retired before the 2026-09-09 index existed. The history of each is in the archive (rows:
grep `^| N |`).

**Closed 2026-10-22, unreleased** (#275 (4); #134 (c))
- #275 (4) — **A large model filling the head's card leaves its drafter room there**, so the
  split's trees of guesses run. A card plan was not cut: the target's layers filled the card, the
  drafter went to the processor, and trees — which pay only with the drafter on a card — stood
  aside for the chain stream. `ModelProcessPool::max_hostable_layers_beside_guest` now leaves the
  drafter's card footprint free on a card plan, as its own `card_verdict` will weigh it; a fresh
  worker's card/processor split (which fills the card whatever the room) is withdrawn by the cut,
  and its offer alone calls for one (`scheduler::leaving_room_here`). Past the room the drafter
  still loads in system memory where that has room, as before. SpecExec (arXiv 2406.02532) makes
  the same trade. Tests `room_is_left_on_the_card_beside_the_layers_for_their_drafter`,
  `a_fresh_card_split_gives_way_to_the_drafters_room`. On the real link (Thailand↔Australia
  ~411 ms, a 7B split, this card capped at 4,000 MB, the Q8_0 0.5B drafter at 966 MB): with no room
  the plan put 22 layers here and the drafter on the processor — 5.22-5.97 tok/s; the default plan
  was cut to 16 ("Leaving room beside this node's layers for the drafter"), the drafter loaded on
  the card and trees ran — 7.85-11.05. Only that first pair compares the cut: afterwards the peer
  still held layers 6-28 and both settings planned 0-6 here (~10 tok/s each).
- #134 (c) — **Trees with the drafter on the card beat the chain stream on the real link**:
  Thailand↔Australia (~405 ms), a 7B split with the head's half and the 0.5B drafter on this
  machine's card, arms alternating by node start on the v0.3.247 release binary — trees 8.41 tok/s
  mean at 160 tokens against the stream's 5.95, 9.36 against 6.48 at 400 (four replies each; every
  reply rank-1 against llama.cpp at 394-397 of 400 tokens, near-ties only). The .247 default
  (trees there) stands. `docs/invariants/scheduling.md` § "A split checks a tree of guesses".

**Closed 2026-10-21, released in v0.3.247** (#324, #325)
- #325 — **A card multiplies a split's check rows and a prompt with llama.cpp's current
  tensor-core kernel.** A tree check carries 16-64 rows through every machine of a split; the
  vendored candle sent 9-63 rows to llama.cpp's OLD dp4a MMQ (fixed 64-column tile, no tensor
  cores) and 64+ to dequantize + cuBLAS — 3.3-3.6× llama.cpp's time at 16-32 rows (RTX 3070,
  Qwen2.5-Coder-7B Q4_K_M, 300 deep: 16 rows 83.3 ms against 22.9). Now llama.cpp's int8 `mma`
  MMQ (commit 4b1a27fa0eb8, its headers verbatim in `candle-kernels/src/ggml_mmq`, in a namespace
  of its own — a `cuda` build links llama.cpp's ggml too, gotcha #927) takes every product of ≥ 9
  rows wherever the card has int8 tensor cores, and 5-8 rows outside a CUDA graph capture (a tree
  check, the drafter's tree steps: 8 rows 12.1-12.3 ms against 23.5-23.6 on the 7B's first 15
  layers, 25.0 against 52.8 whole). Measured in one binary against the old kernels:
  9/16/32/64 rows 27.4/28.2/34.7/46.7 ms against 86.6/87.3/89.7/113.4 (1.23-1.34× llama.cpp's
  time); the 7B's first 15 layers as a tree of 16/32/64 rows 15.7/19.5/30.8 against
  43.7/46.3/65.4; a 512-token prompt 2,108-2,127 tok/s against 1,677-1,698; decode unchanged.
  Replies: 13 of 18 identical, the 5 forks near-ties against llama.cpp
  (`docs/invariants/inference.md` § "A card multiplies several rows with llama.cpp's int8
  tensor-core MMQ"). Found on the way: the vendored kernel LIBRARIES are machine code for the
  release's floor (8.x) only, so a Hopper or Blackwell card had no image of the MoE kernels, whose
  launches check nothing — their output was never written; flash attention's library likewise
  (its checks are no-ops), and its `run_mha` now returns its launch status. The MoE, MMQ and
  flash-attention libraries now carry
  the floor's PTX too, which such a card compiles on first use; MMQ also probes for an image before
  its first launch. A/B `SWARMLLM_QMATMUL_MMQ=dp4a`.
- #324 — **A request that finds every cold-start wait place taken is answered as the wait would
  answer it.** Found by v0.3.246's pre-release review: the early return said `NoModelLoaded` ("no
  model is loaded yet, download one") on every node, holding models or not. Now
  `api::openai::resolver::cold_start_slot` answers through `SharedState::no_route_error`, the
  end-of-wait answer — a 404 naming the models the node knows, `NoModelLoaded` only for a node
  that knows none. Not a "busy, retry" class: only a model with no manifest here reaches the wait
  (a known one goes straight to the router), and a retry after its manifest lands skips the wait
  altogether, so a full wait is not load. Test `a_full_cold_start_wait_answers_as_the_wait_would`.

**Closed 2026-10-20, released in v0.3.246** (#255, #320, #321, #323)
- #323 — **Q5_0, Q4_1 and Q5_1 run on AVX2, and the five 32-value formats share a pass over
  several rows.** Found profiling the drafter's tree step (#134): 8 rows of Qwen2.5-0.5B cost
  7.1× one row. Upstream candle ran Q5_0 / Q4_1 / Q5_1 through the scalar loop on x86 (llama.cpp
  writes them wherever a row is too narrow for a K-quant: 44% of the 0.5B's Q4_K_M, 12.5% of
  GLM-4-9B's, every Gemma 4 26B-A4B's down experts); none of Q8_0 / Q4_0 / Q5_0 / Q4_1 / Q5_1 had
  a several-row kernel; and a scalar `f16::to_f32` compiled to a `vpinsrw` that merged into the
  accumulator, serialising every block (gotcha #920). Now llama.cpp's AVX2 kernels,
  `avx.rs::rows_of_a_32_value_format` (bit-identical to the per-row loop) and
  `scale_product` / `block_scale_products`. 0.5B Q4_K_M, 4 threads: a guess 26.3 → 14.7 ms, 8
  rows 182.7 → 53.0, a prompt 51 → 182 tok/s (Q8_0: 20.7 → 16.0, 141.6 → 55.8, 76 → 185);
  GLM-4-9B on the processor: decode 220 → 170 ms a token, a prompt 7.2 → 13.9 tok/s. Q8_0 and
  Q4_0 replies are bit-identical; the other three sum as llama.cpp does (GLM-4 against llama.cpp:
  median cosine 0.999610 vs 0.999633 before, the misses near-ties). A/B `SWARMLLM_QDOT_32=scalar`;
  `docs/invariants/inference.md` § "The 32-value formats run on AVX2". Left: the K-quant kernels
  still convert scales with `to_f32` (unmeasured), and our Q8_0 activation rounding is not
  llama.cpp's (half away from zero with `1/d` against to-even with `127/max`).
- #320 — **A node holding all but a few parts of a model fetches the rest, past the replica
  target, and keeps them.** Field report 2026-10-19 (a tester, RTX 3060 12 GB, v0.3.244):
  Qwen3-30B-A3B sat at 33/34 parts since a prune before #286 took part 20 (`holders=4 target=3`).
  The download pass offers a part only below its replica target, and #286's keep protected only a
  model already whole, so the part never came back and every request went through a peer in
  Belgium: 120.7 s warm, against 2.3 s with 34/34 on 10-16. Now
  `coverage::parts_completing_model_here` names the parts such a model lacks — one held all but
  ≤ 1 in 8 of (`COMPLETE_WHEN_SHORT_BY_ONE_PART_IN`), judged BEFORE any lands, so never a model
  in fewer than 8 parts and never one held by half — the replica skip in `scoring.rs` lets them
  through, and prune keeps every part of such a model below urgent pressure
  (`coverage::completes_model_here_counting`, which replaced `holds_whole_model_counting`), so each
  fetch lands in a state prune keeps (gotcha #795). Disk pressure, scope and the metered hold
  still apply: the fetch asks `would_shed_copy` as every fetch does. Test
  `the_part_a_model_held_but_for_it_lacks_is_fetched_past_the_target` (8 parts, part 3 held by
  three close peers: fetched, then kept; control, 2 of 8 missing: nothing fetched), red with the
  skip's exemption off. "Keep on this computer" still fetches nothing itself (its words say the
  rest comes from the model card); a model it keeps is completed by this rule when short by few.
- #321 — **A split's saved prompt opening survives a full card and the up-front admission.**
  Field report 2026-10-19: the same 3,718-token prompt, warm, over the same three segments (this
  node L0-14 and L42-48 on a 12 GB card, Belgium L14-42) went out whole again
  (`activation_bytes=16,180,776` = 2 × 3,718 × 2,176 + 40: every segment from 0, since the
  coordinator resumes from the least any segment holds) and Belgium read it for 103 s; with the
  local segments on the processor (Ollama holding the card) it sent 26,152 bytes and Belgium took
  1.0 s. Two causes on this node. (a) A card outside WSL2 keeps a snapshot beside its live cache,
  and with ~190 MB left for caches beside two segments of a 30B the copy was cut to what fit
  (`snapshot_room`). A card's copy that does not fit whole now goes to host RAM when that holds
  more (`model_worker::room_off_the_card_if_more`, `SnapshotHome::may_leave_the_card`; never
  under `SWARMLLM_PREFIX_CACHE_HOME=card`) — #257's copy, 59-169 ms per 264 MB, against a whole
  prompt read again on every segment. (b) The up-front admission (#263) runs before any segment's
  `lookup_chain`, so a prompt that fit only once saved prompts went evicted the least recently
  used ones — the opening the pass was about to resume from. Up front (`Admitting::UpFront`) it
  now evicts nothing: such a prompt is `Admitted::LeftToTheForwards`, no ranges are recorded,
  and each segment admits itself after its lookup, as before #263; a refusal and a plain fit are
  unchanged. Tests `an_up_front_admission_leaves_saved_prompts_to_the_forwards` (tight room: up
  front keeps the snapshot, a forward's own admission evicts it; room for both; no room: refused,
  nothing evicted) and `a_snapshot_the_card_cannot_hold_whole_goes_to_host_memory`, each red with
  its half off. Not run on a card: this machine's card is reached through WSL2, where every copy
  already goes to host RAM.
- #255 — **A command run without `-d` finds the running daemon's data directory, and sends its
  key only to the node that proves it holds it.** `/health` names the directory (`data_dir`,
  absolute) to a request from this machine (`RequestOrigin::is_this_machine`) and to no one else;
  every client command without `-d` or `SWARMLLM_NODE_DATA_DIR` asks the daemon on its port
  (`cli::client_access`), falling back to the default when nothing answers, an older daemon names
  none, or the path is not on this machine (a container's). The pre-release review (2026-10-20)
  found the first version handed the key to a squatter: any local program on the port while the
  node is down could name `/var/lib/swarmllm`, and `sudo swarmllm status` read the service's key
  and sent it. So the command sends a fresh 32-byte `nonce` and the node answers `key_proof` =
  `api::server::health_key_proof` (BLAKE3 keyed hash of the port it serves on and the nonce,
  under a key derived from the API key and used for nothing else); the command checks it with the
  port it dialled (`choose_access`) and otherwise refuses, sending nothing. A second review of
  the proof found three holes, each closed: (1) a command re-read the key file after the check,
  so a FIFO swapped for a symlink between the reads made root send any one-line file — the key is
  now read ONCE, by `client_access`, and carried in `DaemonAccess` (guard
  `no_command_reads_the_key_file_itself`, with a planted self-test); (2) a program on one port
  could pass the challenge on to a node on another — the port is in the proof; (3) every
  key-bearing request took `HTTP_PROXY` — all go through `cli::daemon_client` (`no_proxy`). Every
  command reaches the node at `cli::daemon_base` (`127.0.0.1`, where it listens): six used
  `localhost`, which resolves to `::1` first, where the node does not listen and any program may.
  While the node runs, a program on this machine can already be handed its key as the dashboard
  is, so the proof guards the time it is down. Research: Jupyter's runtime files serve the same
  user only, the packaged service runs as `swarmllm`; Docker and Tailscale ask the daemon; a
  challenge over a keyed hash is HTTP Digest's and the SSH agent's shape. Tests:
  `a_daemon_that_cannot_prove_it_holds_the_key_is_refused` (no proof, garbage, a guessed key, a
  replayed proof, another port's proof), `health_names_the_data_dir_only_to_this_machine`,
  `no_command_reads_the_key_file_itself`, each red with its fix off;
  `a_command_without_a_data_dir_reads_the_running_daemons`,
  `the_daemons_data_dir_is_used_only_where_it_exists_here`, `no_daemon_leaves_the_default`.
  Residual: a node that names no directory (0.3.244 and older) is still sent the default
  directory's key unproven, as every release did; and the dashboard's own `localhost` address can
  reach a program on `[::1]` → #317.

**Closed 2026-10-19, released in v0.3.245** (#316, #304, #269, #305, #184, #179, #115 (2a), #115 (2b), #91 (2h), #285, #141, #196, #302, #163, #310; #186 decided)
- #316 — **`inference.local_embedding_privacy` keeps the prompt's tokens on this computer, as it
  says.** It had never worked on a real node, three ways at once (found 2026-10-20 checking a
  release reviewer's finding on a rig): (1) the embedder opened `shard_000.bin` as a whole GGUF —
  a part is tensor data from its first byte, so every load failed with `unknown magic
  0x3f800000`, the setting did nothing, and a first segment on another computer was sent the token
  ids it exists to keep here; (2) had it loaded, the prompt pass went out at `index_pos` = the
  prompt's length (every row rotated that far, the reply decoded from before them, a picture's
  spans dropped); (3) the DSD, speculative and n-gram paths never asked the setting — the n-gram
  path served every such request a rig offered it. Now the embedder IS the first segment's loader
  asked for no layers (`daemon::shard_loader::load_model_range`, one function with the worker), on
  the processor, loaded when a request first asks (`local_embedder::local_embedder_for`; it was
  loaded only at startup); its rows are the first segment's own (`SplitModel::embed_ids` →
  `embed_tokens`, bit for bit on Llama 3.2 3B, TinyLlama and Gemma 2 2B). The pass is read from 0.
  Where the embedder cannot be built and the first segment — or a stand-in for layer 0 — is
  another computer's, the request is refused (`EmbeddingPrivacyUnavailable` — its message and hint
  name this setting; `PromptPrivacyUnavailable` names "Start and finish on this computer"), never
  sent; the fast paths stand aside for such a plan (`fastpath_request_disqualified` →
  `segment_zero_may_leave_this_node`); a stand-in for segment 0 is handed an embedded prompt only
  if it advertises `FORWARD_PRE_EMBEDDED`. The prompt is embedded ONLY there
  (`PipelineExecutor::prompt_embedder`): the review found a first segment run HERE embedded too,
  which dropped a LoRA adapter (the worker's pre-embedded branch passed none — it now does), handed
  both tensor-parallel paths floats as text (no group takes an embedded first segment now,
  `tensor_parallel_group_for`) and skipped the split prompt cache. The boomerang's read of
  `local_embedders` is gone (no release filled it; a stale entry would have embedded a local first
  segment). The HTTP forward to a pool member asked neither privacy setting; it now asks
  `SharedState::privacy_refuses_a_whole_hand_off`, as the hand-off and the led split do. The setting has one live reading
  (`SharedState::local_embedding_privacy`). Tests (each red with its fix off):
  `a_prompt_embedded_here_is_sent_embedded_and_read_from_position_zero`,
  `a_prompt_that_cannot_be_embedded_here_is_never_sent_to_another_computer`,
  `prompt_privacy_keeps_the_fast_paths_off_a_plan_whose_first_layers_may_leave`,
  `the_prompt_is_embedded_only_where_the_first_layers_may_leave`,
  `an_embedded_first_segment_is_never_run_by_a_tensor_parallel_group`,
  `the_http_forward_to_a_pool_member_asks_both_privacy_settings`,
  `an_embedding_privacy_refusal_names_the_setting_that_refused`; real files:
  `local_embedder_load_from_real_model` (ignored). Evidence: `docs/invariants/scheduling.md` §
  "Prompt privacy embeds the prompt with the first segment's own loader".
- #304 — **A link whose key rotation did not land keeps working.** Every 600 s
  `SessionManager::evict_stale` destroyed each session older than 600 s — which a session reaches
  only when no rotation renewed it — while the peer went on sealing with it: its next message opened
  under nothing (`had_session=false had_previous=false had_retired=false`, the 2026-10-18 `peerkv`
  rig's tenth prompt piece) and this node could not seal to it until something re-keyed. Now the
  aged session is retired like a disconnected one (openable, never sealable, its own replay window,
  `PREVIOUS_KEY_GRACE`) and the rotation task starts a fresh exchange with each such peer at once
  (WireGuard re-keys before it rejects). Whether this was the rig's cause is not proven (its logs
  are gone); it is the one path that fits without a disconnect. Test
  `a_session_that_ages_out_still_opens_what_the_peer_sealed_with_it` (red without the
  retirement). The security review (nothing at MEDIUM or above) found a LOW it makes easier to
  reach, fixed with it: a key promoted from `unconfirmed` into an EMPTY session slot
  (`install_with_window`'s vacant arm) started a fresh replay window, so the frame that confirmed
  it could open once more (`the_confirming_message_cannot_be_replayed_after_the_session_aged_out`,
  red without). Its other LOW is #315. Evidence: `docs/invariants/network.md` § "An aged-out
  session is retired and re-keyed".
- #269 — **A node that stops no longer logs its model workers' end as a warning.** A stop signals
  the worker and the daemon together (a terminal's Ctrl-C, systemd's control group, a `kill` of
  every `swarmllm` process), so every stop logged `Worker reader exiting — evicting` at WARN — 65
  times in the live node's log, every one at a stop, which a field report's excerpt read as a
  crash. The reader now evicts first and then judges (`process_pool::worker_ended_with_the_node`,
  on its own task — dropping the worker's handle aborts the reader): the daemon's stop asked
  already, or within 2 s (`WORKER_ENDED_WITH_THE_NODE_WITHIN`; in 4 of the 65 the worker's end
  reached the daemon first, by under 0.2 ms) → `debug!` "Model worker stopped with the node";
  otherwise the WARN, as before. The pool reads the daemon's stop watch
  (`ModelProcessPool::watch_daemon_stop`). Checked on one TinyLlama node: v0.3.244 stopped by a
  group SIGTERM logged the WARN, the fix logged only the debug line, and a worker killed alone
  while the node ran still logged the WARN.
- #305 — **A machine in the middle of a split passes a refusal from the machine after it on to
  the computer that is waiting.** A machine that cannot open a forward answers the connection it
  came in on — in a chain, the machine before it, which was not waiting and dropped it, so the
  computer driving the request waited out the step's deadline. Every result a node takes in now
  goes through one intake (`NetworkManager::dispatch_layer_result`); a hop that handed a forward
  on remembers the coordinator and the step past the next machine's receipt
  (`network::manager::hop_forwards`, capped, 600 s) and passes on an error from that machine
  about that step, once, in its own name. A failed forward's error now names its step on every
  path (`fail_tensor_forward`'s step is required), so a streamed wait hears of it too.
  `split_rig.sh refusechain` (TinyLlama, B → C, C refusing one chained step): v0.3.244 dropped
  it at B and took 45.8 s with A timing the step out; now 1.5 s, 200, nothing timed out.
  security-reviewer: no finding at MEDIUM or above. Evidence: `docs/invariants/network.md` §
  "A refusal that cannot be addressed past its hop is passed on by that hop".
- #184 — **A chain whose last machine stops mid-reply ends in seconds, not at the step's
  deadline — measured, no further change.** `split_rig.sh killchain` (TinyLlama, A driving a
  decode step over B → C, C's node killed 40 results in), v0.3.244: 6.3 s after the kill, A's own
  departure check (`fail_forwards_awaiting_departed_peer`, which accepts any chain member) failed
  the step and the reply came back with what it had (200, `finish_reason: "error"`); with #305: 1.6
  s, the kill having caught a forward before C's receipt, so B's `OutboundFailure` went through
  the hop's report. Nothing waited out a deadline either way. Not done, deliberately: a hop does
  not report its onward machine's departure AFTER that machine acknowledged — the tail answers the
  coordinator directly, so B losing C says nothing about whether C can still reach A, and
  reporting it would turn a working reply into a failure (#386's lesson). That leaves a
  coordinator with no connection of its own to a hop that dies after acknowledging, which the
  step's deadline still bounds.
- #179 — **A `temperature: 0` reply on the processor is the same every time, prompt cache or
  not.** The first run of a request and every repeat gave two different texts (Llama-3.2-3B:
  `2d85feea` then `f60b97f4`, in both rounds of a restart); with the prompt cache off all eight
  replies agreed. A cache hit runs the last prompt token as a decode step, and the decode and
  prompt kernels summed a row differently — the prompt kernel even split a row by the
  FORWARD's length. Both now run one row arithmetic (`inference::attn_rows`): a row reads only
  its visible keys, in fixed 256-key chunks of absolute position, each chunk scored, softmaxed
  and summed by one defined order of fused multiply-adds, the chunks folded in order by the
  online merge (Thinking Machines' batch invariance; SGLang's fixed split-KV). Speed kept:
  Llama-3.2-3B reads a 6,144-token prompt at 58.2 tok/s (was 56.6) and a 2,048-token one at
  64.2 (63.0), decode 10.6 / 14.2 (10.5 / 14.1), best of 2 in one binary. Verified on the request: the same greedy request eight times across a restart, prompt cache on,
  gives one text; a row read alone and inside the prompt are bit-identical, and the reply scores
  against llama.cpp as before (197/200 rank 1, worst rank 2). Left: the card (#313). Evidence: `docs/invariants/inference.md` §
  "A row's attention on the processor does not depend on the forward it is in".
- #310 — **LLaVA reads a picture as LLaVA does.** Five differences, measured: the class row reached
  the model (577 rows, now 576), the feed-forward ran the erf GELU (CLIP's is quick GELU — read off
  `clip.use_gelu` / `clip.use_silu` as llama.cpp does, `EncoderActivation::from_keys`), a file
  without `v.post_ln` got a weight-1 norm that still normalised (now none), the picture was
  stretched and resized bilinear (now letterboxed on the mean colour, bicubic, as LLaVA's
  `expand2square` and llama.cpp), and the depth is now reasoned (`encoder_depth`: every block the
  file holds — the 23 LLaVA reads — or its one `feature_layer`). Rows against a float64 numpy reading
  of the file (`examples/llava_reader_reference.py`): every row at cosine 1.000000 (was median
  0.917). llama.cpp is not the reference for LLaVA: its graph puts the class token LAST and runs one
  block fewer (median 0.525 against exact arithmetic); with both patched it agrees with this node at
  0.99999 but for ~2% of rows it alone makes high-norm (cause undetermined). Gemma 3 unchanged
  (0.999999). Also: a picture in an Anthropic `tool_result` reaches its tool turn
  (`tool_result_content`; it became the caption alone). Left: #312. Evidence:
  `docs/invariants/inference.md` § "LLaVA's image reader is checked against exact arithmetic".
- #163 — **The .rpm creates the user its service runs as, and does everything else the .deb's
  scripts do.** It shipped no scriptlets: after `rpm -i` the service restarted in a loop for want of
  its user, the node's config was never seeded, an upgrade left the old binary running and removal
  left the service enabled. `packaging/rpm/` now holds `pre` (the user), `post` (data folder,
  config seeded once), `posttrans` (`try-restart` — in the NEW package, so the first upgrade from a
  scriptless .rpm restarts too; rpm runs the OLD package's `%postun`, gotcha #905), `preun`
  (stop + disable on removal only, gotcha #313), `postun`. rpm's own sysusers support needs
  rpmbuild, which cargo-generate-rpm is not. Also dropped the `openssl >= 3.0` requirement: it
  asked for the command-line tool, which a minimal Fedora lacks (`rpm -i` failed) and the node never
  runs; the libraries are required automatically. Guard `every_rpm_scriptlet_names_a_file` (a
  mistyped path is packaged as the script text, silently). Verified in Fedora 44 containers booted
  with systemd: fresh install → enable → upgrade (restarted, still enabled, admin's config edit
  kept) → remove (stopped, disabled, user and data kept) → reinstall; upgrading from the released
  v0.3.244 .rpm both with and without the guide's hand-made user (the stuck node came up by
  itself). RHEL 10.2 installs it; RHEL 9 refuses it for glibc (#311).
- #302 — **A request re-planned after a failure names every computer it reached, not only the
  route it finished on.** `mark_assembled` replaced the trace's segments per plan, so a request
  that failed on a peer and was re-planned here said `local`, no peers, while that peer had been
  handed its hidden states (or wrote a continued reply's first part). The trace keeps the union
  of every route it was given — each plan, a takeover through `RequestTrace::retake` (installed
  only by `PipelineExecutor::install_takeover`), each standby tried before it is sent anything —
  and `RouteSummary` adds `attempts`, `all_nodes`, `all_peers` (headers `x-swarm-attempts`,
  `-all-peers`, `-all-nodes`); the dashboard's chat says "tried N other computers first". A
  continuation keeps the same trace (checked). Limits: a chain its head leads that fails over on
  the head's side reaches computers this node never hears of; a planned node is counted even if
  it failed before receiving anything (the safe direction). Also fixed: the dashboard's streamed
  chat never showed a route label (it read headers a stream lacks; now the finish chunk). Tests:
  3, the takeover one red with `retake` removed; `split_rig.sh continue` live: both peers named,
  `attempts: 2` — v0.3.244 fails the same check. `docs/invariants/api-surfaces.md` § "A reply
  says which computers served it".
- #196 — **An MCP tool that ran and failed answers with a result the model reads (`isError:
  true`), not a protocol error.** MCP 2025-11-25 § "Error Handling" puts input validation, API
  failures and business logic there (SEP-1303: a protocol error may never reach the model); the
  official TypeScript and Python SDKs turn every failure inside a tool into it. One constructor,
  `mcp::types::tool_failed`, for every such failure of the seven tools — 22 argument checks, a
  failed or timed-out inference, "no model to research with / delegate to" — with the client-safe
  message and its hint as the text and `error_body` under `_meta["com.swarmllm/error"]`, so the
  class `tool_error_code` coded as `-32602` / `-32000` / `-32603` is still there for a program
  (`tool_error_code` and `RESOURCE_UNAVAILABLE` are gone; the archive's objection 2 answered).
  A 500's detail no longer reaches the caller, as on every other surface. An unknown tool, an
  unreadable keep instruction and a missing router stay JSON-RPC errors. Checked against Claude
  Code itself (`examples/mcp_client_rig.sh`, Haiku): it showed its model the old protocol error
  too (`is_error=True`, message only), so there the change brings the advice; for a client that
  drops protocol errors it brings the failure at all.
- #141 — **A non-streamed reply carries a reasoning model's finished scratchpad, as the stream
  does.** Streamed, a `<think>` block (or Gemma 4's thought channel) arrived as
  `delta.reasoning_content`; non-streamed, `finalize_reply_text` removed it and nothing kept it.
  The finaliser now returns it (`FinishedReply`, `#[must_use]`), every reply source puts it in the
  REQUIRED `InferenceOutput::reasoning` — the worker pool, the in-process executor, the three
  distributed paths, speculation, the hand-off, a continued reply — and the OpenAI reply writes it
  as `message.reasoning_content` (llama.cpp's, SGLang's and DeepSeek's name), tool calls included.
  The in-process speculative path returned its text unfinalised and now goes through the same
  finaliser. A reply that is only a scratchpad now shows it both ways (it was listed as
  deliberately left). Rig `split_rig.sh reasoning` (qwen3-1.7b): split 558/558 and local 537/537
  characters, same content; v0.3.244: 0 against 558. Anthropic stays withheld both ways: #141 (b).
- #186 — an emptied reply stays a 200: moved to § "Decided — not doing" (Product scope).
- #285 — **A tensor forward is never sent unsealed while encryption is on.** Both senders
  (`handle_send_tensor`, the pipeline stream's `encode_forward_for_wire`) looked the peer's NodeId
  up in a map that a reconnect before Identify or a pruned entry could leave empty, and then sent
  the forward plain. The transport still kept it from third parties; what went was the session
  seal and its AAD binding. `pipeline_stream::seal_for` is the one answer now, and DERIVES the
  node from the peer's id (both come from one Ed25519 key); with no session yet the seal fails and
  so does the forward — and the encrypt task's failure goes back to the event loop
  (`FailEncodedTensor` → `fail_tensor_forward`), so a chained hop tells its coordinator instead of
  leaving it to its whole deadline (security review). Receivers still take a plain forward: the
  connection is authenticated end to end (Noise or TLS, relay circuits too), so only the peer
  itself can send one, and a node with encryption off must keep working. Split rig at debug: 63
  forwards sealed, 0 plain; kill-B fail-over: 382 sealed, 0 plain, answered.
- #91 (2h) — **An idle node answers about half as many routing-table lookups.** Every node
  refreshed its Kademlia routing table every 5 min, and in a swarm smaller than 20 each refresh asks
  every peer ~5 lookups, each answered with ~3.7 KB of peers and addresses: on 10-19 that was all of
  the live node's 509 B/s of Kademlia upload. libp2p now refreshes every 10 min
  (`behaviour::KAD_REFRESH_SECS`, go-libp2p-kad-dht's interval) and the discovery tick no longer
  starts a refresh of its own. Rig `examples/kad_refresh_rig.sh`: 5 refreshes in 25 min → 2, the
  hub's answered `find_node` 15 → 6 on one binary (A/B `SWARMLLM_KAD_REFRESH_SECS=300`). A peer still
  enters the table the moment it connects. Left in #91: the answers' size, re-read on the field.
- #115 (2b) — **Gemma 3 reads a picture as a whole, as llama.cpp does.** Its picture rows were
  read in order like text; Gemma 3 was trained with every row of a picture seeing every other
  (llama.cpp `mtmd_decode_use_non_causal`), and read in order it called a red circle, a blue
  square and a green stripe on black "a tricolour flag". The coordinator now works out where each
  picture's rows sit (`PipelineExecutor::image_spans_for`) and every forward reaching into one
  carries them to every computer holding part of the model (`LayerForward::image_spans`, trailer
  `0x0F`, sealed; each must advertise `features::IMAGES_AT_MARKERS`), where the attention mask lets
  a picture's rows see each other (`causal_mask_with_spans`, the windowed layers keeping their
  window). Scored by teacher forcing through llama.cpp's own image path
  (`examples/score_image_reply_against_reference.cpp`): 95.5% / 97.3% of the reply's tokens
  llama.cpp's first choice, every miss a near-tie, against 87.3% / 90.7% with misses 6-12 logits
  deep before. A/B `SWARMLLM_IMAGE_BOTH_WAYS=0`. `docs/invariants/inference.md` § "A picture's rows
  attend both ways".
- #115 (2a) — **Pictures reach the model where you put them, and Gemma 3 can read images.** Every
  picture used to go in FRONT of the whole prompt, ahead of the chat template's opening: no surface
  marked where it belonged (the OpenAI surface dropped the image part's place, the Anthropic
  surface dropped image blocks, the fallback wrote the text `<image>`), and the splice prepended the
  rows when it found no mark. Every surface now writes llama.cpp's marker where the picture was
  (`vision::MEDIA_MARKER`), the rows go there (`vision::splice_image_rows`, the one splice, which
  refuses a prompt with no mark), the prompt's positions count each picture's rows
  (`vision::positions_with_images`), and a picture that cannot be read is the request's error,
  never a reply without it. Gemma 3's image reader (`clip.projector_type` `gemma3`: letterboxed to
  896, SigLIP, 4×4 pooled to 256 rows, its own norm and projection) is new; a projector type not
  built here is refused by name. A first segment must advertise `features::IMAGES_AT_MARKERS`
  (bit 26). On the way, three older faults that each lost the picture silently (gotcha #901):
  every node dropped its claim to an image reader minutes after downloading it
  (`ShardStore::shard_path` named the wrong file), a picture sent to a model held whole here took a
  text-only shortcut (`pipeline::FastPathNeeds`), and on a processor the splice handed the first
  layer 16-bit values it refuses (gotcha #241's fault, also in tensor parallelism's embed step;
  guard `every_token_lookup_goes_through_embed_tokens`). Verified on Gemma 3 4B: `shapes.png` and
  a 1969 newspaper front page described as llama.cpp (causal) describes them — `shapes.png` word
  for word but one verb — on both surfaces. Evidence: `docs/invariants/inference.md` § "An image
  sits at its marker". Left: (2b) Gemma 3's picture attention both ways; #310.

**Closed 2026-10-19, released in v0.3.244** (#3 (d); #307; #295 (a), (b))
- #295 (b) — **Gemma 4 E2B writes replies about half as fast again on a graphics card, E4B a fifth.**
  Once captured, an E2B step was bound by the host: launching its 17 graph groups took 5.8 ms
  (Gemma 3 4B: 0.4). Its per-layer inputs were built inside the first group, so every later group
  read memory a graph had allocated, and the driver remaps or waits for such memory at each launch
  (CUDA Programming Guide § 4.2.5). They are now built before the capture (`executor::Prepared`):
  launches 0.7 ms, E2B 60.4-60.9 → 92.9-93.3 tokens a second, E4B 50.3-51.3 → 59.8-60.4, replies
  identical on all 16 runs; A/B `SWARMLLM_PER_LAYER_INPUTS=in-graph`. Found with four new
  `SWARMLLM_PROFILE` stages (token embedding, Gemma 4's per-layer embedding, output head, and the
  graph group boundaries split into end / update / launch). Evidence: `docs/invariants/inference.md`
  § "Gemma 4's E2B and E4B", "The rest was graph memory crossing groups".
- #295 (a) — **Gemma 4 E2B and E4B write replies about a third faster on a graphics card.** Their
  decode steps were never recorded as CUDA graphs: each step's per-layer rows are gathered on the
  processor and copied to the card, a copy a recorded step would replay from a stale address, and
  a layer that reads another layer's cache keeps an empty slot that read as a cache still to be
  built. The rows are now copied before the recording (`executor::RecordedStep`), and an empty
  slot its layer keeps by design holds (`SplitModel::slot_keeps_no_cache`). Whole on the RTX 3070
  Laptop: E2B 44.6-49.1 → 60.7-62.8 tokens a second, E4B 33.9-39.9 → 49.9-50.3, replies identical
  with capture on and off (`docs/invariants/inference.md` § "Gemma 4's E2B and E4B"). Left: #295.
- #307 — **Replies on a graphics card no longer stall once a conversation reaches 64 tokens, and
  Phi-4-mini, GLM-4 and Qwen 3.5 decode with fewer launches.** Found by a card-decode sweep of
  every reference family against llama.cpp (2026-10-19, `docs/DIAGNOSTICS.md` § "One model's
  decode rate on this machine"). (a) The card's decode attention added its merge kernel only past
  one 64-position chunk, so the step at position 64 changed every layer's kernels and every CUDA
  graph group was rebuilt at once — 790 ms in one Gemma 3 4B step; the merge now runs at every
  length (`decode_attn::combine_every_length`, bit-identical for one chunk). Gemma 3 4B 51-79 →
  78.6 tokens a second steady, Phi-4-mini 51-67 → 92. (b) A rotary width narrower than the head
  was five launches for each of Q and K; it is one (`inference::partial_rope`, llama.cpp's
  `rope_neox` / `rope_norm` shape, bit-identical to the composition): Phi-4-mini 87.0 → 92.0.
  Replies identical on every arm; A/B `SWARMLLM_DECODE_ATTN_COMBINE=when-needed`,
  `SWARMLLM_PARTIAL_ROPE=compose`. Evidence: `docs/invariants/inference.md` § "That remainder was
  priced on 2026-10-19", § "On a card it is one launch".
- #3 (d) — **A split that guesses ahead is priced for the rounds it takes, not the tokens it
  writes.** A split driven from here crosses its hops and runs its segments once a ROUND, and the
  guess-check paths (the drafter's rounds and stream, the n-gram loop) carry several tokens a
  round; the price charged every token a crossing, and the drafter rigs ran in 0.39 of it
  (2026-10-15, n=25). Now each round is counted where it leaves (`RequestTrace::note_decode_round`
  in `forward_through_segments` past the prompt pass, `forward_verify_through_segments`,
  `dsd_stream`'s chunk), each finished split recorded per model by `PipelineExecutor::execute`
  (`daemon::state::split_rounds`: tokens over rounds, the last 32 once there are 4, never below 1 —
  vLLM's "mean acceptance length", the figure SmartSpec and SGLang steer by), and a segment of a
  split driven from here priced for the passes over it (`NodeCandidate::rounds_entered`); a whole
  model and a led chain keep the passes. Only for a request that may guess ahead
  (`pipeline::request_may_guess_ahead`, asked by the router and before recording). Rig
  `split_rig.sh repeat` with the 0.5B drafter (7B on the processor): 72 rounds for 120 tokens,
  1.64 a round, and the asks priced with it read 0.77 of their price where a round a token reads
  0.49. Counting, not deriving from α and γ: those give 2.26 tokens a round, and the stream sent
  72 chunks whose time it took. Left: the price of one round (#3's per-term fit).

**Closed 2026-10-18, released in v0.3.244** (#306; #3 (c); #3 (e); #3 (f); #252 (b) (2))
- #3 (c) — **A conversation's next turn on the same peer is priced for what that peer already
  read.** A peer that runs a whole model for this node keeps the prompt in its prefix cache and
  reads only what is new next turn, but the planner priced every turn as a full read: on the live
  swarm a repeated 505-token prompt was priced 20.2 s and the whole request took 7.1 s. Now, as
  SGLang's cache-aware router and llm-d's approximate prefix scorer do, this node remembers what
  it handed each peer (`pipeline::handoff_prompt_cache`: the prompt's 64-token block chain, keyed
  with this process's secret and never sent anywhere, for 300 s — the idle unload — and dropped
  on a failed hand-off) and credits the matching opening on a whole-model run there, only for a
  request the hand-off will take (`remote_generate::request_may_be_handed_off`, split out of the
  hand-off's own check) and only while the peer still holds the whole model. It is a price, never
  a pin. Rig `split_rig.sh handoffcache`: turn 2 of 2,744 tokens credited 2,688, the peer's
  worker hit 2,713; priced 7.8 s for 15.1 s, against 148.8 s for 14.8 s on v0.3.243. The old
  reason for crediting no peer — the n-gram path running before the hand-off — had been stale
  since 2026-09-27.
- #252 (b) (2) — **A split whose peer has no room for the prompt finds out before anyone reads
  it.** A peer's segment was admitted only when its forward arrived — after every machine before
  it had read the prompt — and its refusal barred it from the re-plan. On `split_rig.sh peerkv`'s
  tight case v0.3.242 answered 503 after 132.8 s with the peer refusing two forwards; now 1.4 s,
  the peer refusing one question in 3 ms and no forward. The coordinator sends
  `PromptAdmissionAsk` (feature bit 25) beside #263's own admission, waits at most ~1-5 s, and a
  "no" carries `layers_that_fit`, to which the re-plan holds that peer (never barring it, as
  Petals prices `cache_tokens_left` rather than excluding). The asked peer only WEIGHS — vLLM's
  `can_allocate`, not `allocate`: `Admitting::OnlyWeigh` is a required argument of the worker's
  admission, under a fresh correlation id, holding, releasing and stamping nothing, rationed to
  12 questions a minute per peer and 60 in all. Two security-review passes shaped it: the first
  design ADMITTED on a question under the asker's request id, which let a peer displace a live
  forward's answer and park the owner's conversation memory (two HIGH); the second pass caught a
  question keeping a model from ever unloading, a late answer settling a re-plan's question, an
  overflow in the refusal arithmetic, and per-question task/log cost — all fixed, each with a test
  that fails without its fix. The coordinator gives a refusal one class whatever the peer wrote,
  and repeats its words bounded and on one line. Gotcha #896.
- #3 (f) — **A peer inside a chain another peer leads is priced by its own measured speed.** The
  head drives such a chain, so this node never timed the peers behind it and priced their compute
  from their stated speed — the optimistic side: a warm two-peer 16-token reply on Qwen3-30B-A3B
  priced 2.3 s of compute took 19.4 s. Now, after Petals (`server/throughput.py` times
  `inference_rps` on the real block; `sequence_manager.py` prices a span from it), each worker
  times the one-position decode steps it runs FOR PEERS (`ModelProcessPool::forward_for_request`,
  `OwnDecodeSpeed`) and publishes the fastest of its last 32 as
  `ResidentModelLayers::decode_us_per_layer` (after 4 steps; gone an hour after the last), which a
  coordinator takes as a FLOOR under the stated speed (`prior_ms_per_layer` = the max), never a
  discount; anything it measured itself still outranks it. A 1.5x deadband keeps it from being
  capability news on every wobble (#91). A security review (2026-10-18) changed two choices: a
  median let any peer queue a node's steps into a slow figure (now the minimum, which only a node
  really that busy moves), and timing the owner's own steps would have told the swarm when the
  owner ran a split (now peers' work only). Tests (each red with its part off):
  `a_led_chain_is_priced_by_its_peers_own_timing_where_nothing_was_measured_here`,
  `a_peers_published_timing_reaches_its_candidate_as_a_floor_on_its_price`,
  `a_workers_own_decode_steps_are_published_beside_its_layers`,
  `a_workers_own_speed_is_the_fastest_of_its_latest_steps`,
  `a_workers_own_speed_is_news_only_past_its_deadband`, `a_one_position_check_is_timed_as_a_decode_step`.
  Rig `split_rig.sh owntiming` (`RIG_NETNS=1` runs it beside a live node): PASS 4/4 — each node's
  plan carries the other's own timing once it has served it, and none while it ran only its own.
  Its first run FAILED the last check, and found a gap no test had: a coordinator speculating with
  n-grams sends each step that found no guess as a one-position check, which the batch path's test
  of "a decode step" refused, so a chain driven that way published nothing. A step is now one
  position counted the way the serve credit counts it (`tensor_util::positions_carried`, moved there
  so the two cannot disagree). `docs/invariants/scheduling.md` § "A peer's own timing is a floor
  under its stated speed".
- #3 (e) — **Two peers whose loaded ranges already form a chain are planned that chain.** A
  machine's own loaded range was offered to the route search only on a pass that found no route
  (#275's retry, which also cuts OTHER machines' ranges and is rightly held back). Live, Qwen3-30B-A3B:
  an Italian peer held shards [0..19) with a worker on [0..18), a Belgian one a worker on
  [18..48); Italy's whole [0..19) fit, so the plan was [0..19) + [19..48) — 29 layers Belgium's
  worker reads again — priced 84.9 / 90.3 s for replies that took 19.4 / 29.4 s (the head
  re-planned onto the loaded ranges; a chain driven from here would have loaded them). Now a
  machine's own loaded range is a vertex on every pass. And a peer's room for new layers is
  compared unrounded with what its loaded range costs this prompt: rounded down against a KV
  charge rounded up to a whole layer, Belgium (0.6 of a layer free, ~1.5 MB of cache needed) was
  refused its own range on the two strict rungs and the plan went to the one where "a peer may
  refuse". Tests `two_peers_whose_workers_already_chain_are_planned_that_chain` (the live
  figures; red both ways: `[(0, 19), (19, 48)]` with the vertices behind the retry, no route on
  `Everyone` with the rounding) and `a_peer_with_less_than_a_layer_free_still_runs_the_range_it_holds`.
  Live (`~/swarmllm-3e/check.sh`, the dev build as this node, 2026-10-18 14:42-14:46Z): 5/5
  answered; Belgium's room read 0.39 (was `Some(0)`) and a held-range route was found on
  `Everyone`; warm 16-token replies priced 17.3 s, took 15.5 s (the release, same peers minutes
  earlier: 7.6 s priced, 23.8 s taken). The peers' loaded ranges had moved by then, so the
  morning's exact shape was not on offer live. One request was refused by an Australian peer
  whose advertised room (~16 new layers) was stale (its own three workers held 23.7 GB): #252
  (b)'s case, recovered by the re-plan.
- #306 — **A streamed reply this computer served alone now names this computer.** Found on the
  live node 20 minutes after v0.3.243 shipped #251 (3): a streamed OpenAI, Anthropic or Responses
  reply from a model held here said `"route": "local", "segments": 0, "nodes": []`, while the same
  request unstreamed named this node. Both streamed surfaces answer such a request on the local fast path
  (`spawn_split_stream`), which bypasses the router and never recorded its route on the trace the
  finish chunk reads; only the unstreamed fast path did. #251 (3)'s tests drive a fake router and
  its rig a two-node split, so neither reached this path. The route is now recorded inside the one
  function both surfaces call, before generating (`mark_served_here`, shared with the unstreamed
  path). Test: `a_reply_streamed_by_a_model_held_here_names_this_computer` (red with the call
  removed: `segments: 0`). Rig: `split_rig.sh route` with `SHARDS_A` naming every shard now keeps
  the model on A (a processor node with a peer up is planned by the router, so a plain whole-A run
  never reached the fast path and PASSED on .243) and expects A alone: v0.3.243 FAILS the three
  streamed forms (`segments: 0`), the fix PASSES all 8; the split topology still PASSES 8/8.

**Closed 2026-10-18, released in v0.3.243** (#303; #251 (3) — #251 closed; #228 (8); #3 (b); #3 (a))
- #3 (a) — **A chain run entirely on other people's computers is priced by the distance between
  them, not by its distance from you.** The chain is handed to its first machine, which drives
  every token from there, but routing priced each machine as a round trip from here per token —
  4-10x too slow for chains of nearby peers far from the requester. It is now priced by its hops:
  half the round trip the network coordinates predict between consecutive machines and from the
  last back to the first. Measured on the live swarm: the network part of a two-peer chain's
  estimate went from 57.9 s to 3.6 s (same peers, same reply length). Tests: `a_chain_its_head_leads_is_priced_by_the_hops_between_its_peers`,
  `a_far_requester_takes_the_chain_its_peers_lead_over_a_slower_whole_copy`,
  `a_led_peer_is_charged_its_own_visit_not_our_round_trip_to_it`,
  `a_chain_the_led_pass_cannot_price_is_still_routed_driven_from_here`. Left: #3 (e), (f).
- #3 (b) — **A long reply is no longer priced as if it were 64 tokens.** Routing charged every
  reply's per-token costs for 64 passes unless the request allowed fewer, so a model that answers
  at length had its split and hand-off routes under-priced by length ÷ 64 (2.49x and 2.62x at 160
  tokens, two peers, 2026-10-18). The passes are now the mean of this model's last 32 finished
  replies (once there are 4), still capped by the request's `max_tokens`. Test:
  `a_reply_allowed_sixteen_tokens_is_priced_for_sixteen_passes` (extended);
  `docs/invariants/scheduling.md` § "A reply is priced for no more passes than it may take".
- #228 (8) — **Qwen 3.5 conversations no longer re-read their whole prompt every turn.** The
  prompt cache left out every model with recurrent layers; it now saves Qwen 3.5's state 16 tokens
  before the prompt ends and restores it whole. A second turn of a 2,416-token conversation on the
  0.8B (processor, dev build): 16.7 s instead of 127.6 s, the reply identical. Tests:
  `a_recurrent_snapshot_is_restored_whole_or_not_at_all`,
  `a_recurrent_state_is_saved_only_at_the_attention_layers_position`,
  `a_saved_recurrent_state_is_a_copy_on_both_sides`, `a_prefill_chunk_ends_at_the_save_point`;
  `docs/invariants/inference.md` § "Qwen 3.5's prompt is saved short of its end".
- #303 — **A split no longer waits out a deadline when a peer cannot open one piece of the
  prompt.** A peer that cannot decrypt a forward refuses it, and the coordinator sends it again
  once the link re-keys (#92) — but the refusal named no step, and a prompt piece's (or a streamed
  verify's) wait is keyed by its number, so the refusal reached no one: the piece waited its whole
  segment deadline and the request re-planned or failed. `split_rig.sh refusepiece`
  (`SWARMLLM_FAULT_REFUSE_DECRYPT=3`): 503 after 403.8 s before; 200 after, the refusal taken 189 ms
  after the send, the piece resent 28 ms later. The refusal now names its forward
  (`LayerForward::step`, the one construction). Found by #252 (b)'s rig, where B also held no key for
  A mid-prompt (#304, open). Test: `a_refused_prompt_piece_reaches_the_wait_for_that_piece`;
  `docs/invariants/network.md` § "A disconnect retires a session key".
- #251 (3) — **Every reply now says which computers served it, streamed or not, on every
  surface.** Closes #251 (`owned_by: "local"` read as "my data stays here"; it means "every part is
  held here"): (1) the book's OpenAI page says what `owned_by` means and what a peer sees, (2) "Keep
  on this computer" per model (2026-10-12) and (2b) per request (2026-10-17), (4) private mode
  never sends a swarm model to a cloud provider (2026-10-12) — and now (3), the route. The gap was
  wider than filed: the route was only ever in the `x-swarm-*` headers of a reply that was NOT
  streamed. `attach_route_headers` claimed four surfaces and took a `streaming` flag no caller
  passed — measured on the live node (v0.3.242), `stream: true` answered no route header at all,
  and chat streaming did not "already wait for planning" as this entry said. Holding headers until
  planning would silence the dashboard's queued/planning status comments and could name a route a
  re-plan later replaced, so the route travels in the BODY as `swarmllm: {"route": {...}}` (the
  reply-side twin of the request's `swarmllm` field), from ONE reading of the trace
  (`TraceSnapshot::route_summary`, which the headers are now built from too): a chat completion's
  body or its streamed FINISH chunk (`StreamEvent::Finish`, route REQUIRED); Anthropic's body or
  `message_delta` (`send_sse_epilogue_with_stop`'s REQUIRED `route`); the Responses object in every
  form, copied from the chat reply under it (`translate::reply_extras_from_chat`); MCP `chat` /
  `delegate` in `_meta["com.swarmllm/route"]` (`tool_text_result_with_route`), and each fan-out
  result its own `route` (`spawn_model_call_task`). A streamed reply reads the route at FINISH from
  the trace its sender was given (`StreamingTokenTx::channel` → `StreamingTokenRx::route`), never
  `active_traces` (racy). A reply with no swarm route (cloud, refused before dispatch) carries no
  `swarmllm`. Research: OpenRouter's extra top-level `provider` field; never a `choices: []`
  chunk (azure-sdk-for-go #21086); Anthropic's Python SDK drops an unknown event NAME; MCP
  2025-11-25 § `_meta` reverse-DNS keys. A re-planned request named only its last route (#302, closed 2026-10-19).
  Tests: 11, 13 mechanisms toggled off and shown red; `split_rig.sh route` live (all eight forms
  named both computers); `docs/invariants/api-surfaces.md` § "A
  reply says which computers served it, in every form it can take".

**Closed 2026-10-17, released in v0.3.242 (2026-10-18 00:27 UTC)** (#299, #91 (2c), #251 (2b))
- #251 (2b) — **One request can now be kept on this computer, whatever the model's setting.** A
  caller sends `X-SwarmLLM-Keep-On-This-Computer: true`, or `"swarmllm": {"keep_on_this_computer":
  true}` in the body of `/v1/chat/completions`, `/v1/messages` or `/v1/responses`, or MCP's
  `params._meta["com.swarmllm/keep_on_this_computer"]` (`api::keep_here`). The strictest
  instruction wins and `false` never loosens anything; one the node cannot read (another header
  value, a misspelt key — `deny_unknown_fields`) is a 400, never a silently unkept request. The
  answer is a value carried with the request (`InferenceRequest.keep`, `KeepAsked`, a REQUIRED
  argument of `InferenceRequest::local`, never on the wire) into every scope question
  (`pool::scope::request_scope(.., asked)`, `local_fast_path_for`, `cloud_refused_for`,
  `find_peer_with_model`, the image encoder, the prefix-KV probe); a request this computer cannot
  run whole is `RequestKeptOnThisComputer` (503, `kept_on_this_computer_error`, its own hint in 21
  languages naming the instruction). The `swarmllm` field never reaches a cloud provider. MCP's
  fan-out tools put the header on their loopback calls. A kept request for a name no manifest
  knows is a 404 (`ModelNotAvailable`), not advice to download a model there is none of.
  Enumerating the paths found two more: the worker's prompt cache now marks an entry stored for
  a kept request (`Entry::kept_here`, from `IpcGenerate.keep_here`; inherited by an entry covering
  it), whose manifest is then empty, so it is never announced or exported; and `distributed_exec`
  re-ran a conversation's previous route without asking the scope — in a branch that could never
  fire (the router records every conversation with an empty route), so it was deleted, not gated.
  A `KeepAsked::Unasked` written for a request must be in `KEEP_UNASKED_IS_THE_ANSWER` (guard
  `a_requests_keep_instruction_is_never_replaced_by_none`, with its self-test). The
  `security-reviewer` pass found no leak; its two Lows were a non-object MCP `_meta` read as "no
  instruction" (now a 400) and an older worker dropping the cache flag (cannot happen: a worker
  only talks to the daemon that spawned it, over a per-spawn close-on-exec socket). Tests: 11 new
  or extended, 7 mechanisms toggled off and shown red, `split_rig.sh keep` live (every form
  refused or served as it should, the second computer handed nothing while asked);
  `docs/invariants/network.md` § "The same choice per request".
- #91 (2c) — **A restart no longer puts every model manifest on gossip twice.** Found 2026-10-17
  measuring (2b): at boot `daemon::background` broadcast every manifest the node gossips, and the
  health monitor's first round, a few seconds later, broadcast them again, because neither recorded
  anything and what the swarm had heard (`manifest_heard`) lived only in memory. On the release
  node (27 manifests) the first 39 min after a restart sent ~7.4 MB more on `swarm/models` than its
  steady rate, and every release restarts the fleet within minutes. Now the boot loop is gone, a
  manifest round with no peer connected gossips nothing (nobody hears it, and its window cannot
  know whether peers take digests), and `manifest_heard` is stored in the `manifests_heard` tree
  (`SharedState::persist_manifests_heard`, by the round, only when it changed) and read back at
  boot on the wall clock (`load_manifests_heard`; a record older than a day, dated in the future
  or unreadable is left out, so a lost one costs a repeat and never holds one back). The digest
  catch-up tells every reconnecting peer — since 2026-10-20 (gotcha #918); before, one back within
  a round was not told. Bitcoin Core's unbroadcast set (PR #18038) is the
  same split: an item is re-announced until a peer has asked for it, then only a daily repeat. Rig
  `examples/manifest_digest_rig.sh`, B and C's manifests by gossip: first boot 2 → 1, after A's
  restart 3 → 0 (A's first round after it: `suppressed=1`, window 86,400 s). Tests: 3, and 6
  mechanisms toggled off and shown red; `docs/invariants/network.md` § "Announce, then pull".
- #299 — **One machine minting key pairs can no longer stop relaying to this node, and relayed
  senders never crowd out a connected peer.** Found 2026-10-17 by the security review of #298.
  #298's replay memory (262,144 relayed bodies, refused when full, never forgetting one early)
  was shared out per delivering peer id, 32,768 each, and a peer id costs nothing: eight key
  pairs on one machine filled it, and every relayed body addressed here was refused while they
  kept sending and up to 6 min after. The share is now per NETWORK the delivering peer reaches
  this node from (`network::helpers::NetGroup`, Bitcoin Core's `GetGroup` rule: a public IPv4
  address counts as its /16, a public IPv6 one as its /32; a private address as its host). A
  host name counts only when this node's own bootstrap list names it (`ConfiguredNames`; libp2p
  records a dial by its name, unresolved, and the default anchor is `/dns4/…`); any other name,
  and anything naming no IP, is ONE group. A relay circuit is counted as `Circuit` of the
  network this node's own connection TO the relay was counted in — never the relay's own share,
  and never an IP or name written in the circuit's address, which libp2p-relay never dials when
  it already holds a connection to the relay (an inbound circuit's relay is read from the
  address it arrived on, since libp2p-relay 0.21 reports that remote as a bare `/p2p/<source>`). The group is
  recorded with the connection (`PeerRemote`, in `peer_remote_addrs`, which at its cap now
  drops only peers no longer connected) and applied by one admission both relayed kinds call
  (`admit_relayed`). Filling the memory now takes machines in eight networks. Second, each
  newly proven origin went into `peer_registry` and `peer_id_map` with no cap, and as such an
  entry has no measured latency, the identify path's cap (200, evicting the highest-latency
  peer and disconnecting it) dropped connected peers before it. One rule now serves both
  writers (`evict_one_registered_peer`, `unconnected_registry_entries`): an entry for a node
  this node is not connected to goes first, the one unseen longest, and only then a connected
  peer. Proven origins hold at most 64 such entries beside the identified peers, never
  displace a connected one, and every proven body marks its origin seen, so one still sending
  is never the one dropped. `peer_id_map` has one capped writer (`remember_peer_id`: 10,000; a
  mapping for a node no longer registered makes room). The `security-reviewer` pass on the
  first version found that it grouped the anchor by "no IP": the anchor's own deliveries, every
  circuit through it and every self-chosen name shared one share, so ONE peer with a circuit
  through the anchor stopped relaying to every NAT'd node — worse than before (hence
  `Circuit` and `ConfiguredNames`); and that an origin never marked seen was the one minted
  origins pushed out (hence the mark). Its second pass found a circuit's network read from the
  circuit's address, which nobody dials: one attacker relay and a PEX reply of fake prefixes
  minted a group per prefix (hence the relay's own connection). Its third found that relay's
  network read from its LATEST connection, so a relay that also came by circuit put every
  circuit through it in one shared group (hence `PeerRemote::direct`, the network of its latest
  DIRECT connection, kept while any lasts). Left by design: addresses in
  eight /16s (cloud regions) still fill the memory for 5 min; an attacker reaching this node
  through honest relays' circuits gains one group per such relay's network (it no longer
  touches the relays' own deliveries); hosts inside a private network or tailnet are free groups
  to its members; every unconfigured DNS-dialed peer shares one group; an origin whose bodies
  are minutes apart can still be pushed out of the registry in the milliseconds between its
  body and dispatch. Filed: #300, a sender filling the share of the very relay it sends
  through. "Let the total grow with the relays actually connected" was weighed and not taken:
  eight minted peers ARE connected. Tests: 3 for `NetGroup`, 7 against a real `NetworkManager`,
  12 mechanisms toggled off and shown red; `docs/invariants/network.md` § "A relayed body's
  origin is signed".

**Closed 2026-10-16/17, released in v0.3.241 (2026-10-17)** (#265, #286, #287, #288, #289, #290, #115 (3b), #292, #115 (3c), #296, #298)
- #298 — **A relayed message or tensor is taken as its sender's only when the sender signed it.**
  Found 2026-10-17 by the security review of #91 (2), filed and fixed in one commit. The
  application relay (NAT'd nodes' remote-generate traffic and split tensors) sealed each body
  with an ephemeral key to the target — an anonymous sealed box, as libsodium says of its own
  ("cannot verify the identity of the sender") — and named the origin only in the AAD, which
  stops a relay re-addressing a body, not anyone sealing a new one. The target took a body from
  any connected peer and handed its inner message to dispatch as the named origin's, after
  learning a route, a registry entry and proven features for it, and never checked that the
  inner message was one a node relays; a verbatim replay also opened. Now the origin signs a
  digest of the kind, both NodeIds, the request id, a timestamp and the inner bytes, inside the
  seal (an older relay re-encodes the envelope and would drop an outer field); covering the
  recipient is Davis's sign-then-encrypt repair (USENIX 2001). A node relays only to a target
  advertising `features::RELAY_SIGNED` (bit 24), always signed; the target refuses an unsigned
  body, a replay (`RelayReplayGuard`: 5 min old / 60 s ahead, each body once, its memory shared
  out per origin and per delivering peer and refusing when full) and a message no node relays,
  and learns nothing before; a refusal is logged once a window per delivering peer. Mixed
  versions: a node before this build is not relayed to, and what it relays here as its own is
  dropped, so relaying between it and this one pauses until its auto-update (minutes) and that
  traffic goes direct meanwhile; an older node still relays between two new ones, and nothing
  that does not relay changes. The `security-reviewer` pass on the first version found it still
  relayed unsigned to an older target, whose reply then came back unsigned and was dropped after
  the work was done (hence the send gate), and a replay memory one peer could fill (hence the
  shares); what it left is #299. Tests: 5 new in `crypto::relay_seal`, 3 against a real
  `NetworkManager`, 10 mechanisms each toggled off and shown red; `docs/invariants/network.md`
  § "A relayed body's origin is signed".
- #296 — **Gemma 4's E2B and E4B are priced as a card holds them.** Their per-layer table (1,837 MiB
  of the E2B Q4_K_M, 2,205 of the E4B's) stays in the machine's memory on every device, as
  llama.cpp keeps its input layer on the processor, and was charged to the card as weights while
  the machine was charged nothing. `VramFootprintInputs::host_resident_bytes` names it apart; a
  card's estimate and split sizing hold `card_weight_bytes`; `ModelProcessPool::machine_charge`
  charges it to the machine beside any card placement — the spawn, `fresh_run_layers_for_planning`
  and the card bound ask it — and a coordinator prices a card peer the same way
  (`scheduler::weight_bytes_per_layer`). The real headers through the pool, at the 4,096-position
  admission context: the E2B's card estimate 1,859 MB (was 3,696) with 1,838 MB charged to the
  machine, the E4B's 3,483 MB (was 5,688) with 2,205 — the E2B now whole on a 4 GB card, the E4B on
  a 6 GB one; the processor's unchanged (3,632 / 5,624). Decode's per-layer read no longer counts
  the table (a row per token): the E2B's layers had been priced at over twice their work. The
  review found one more: weighed against memory that still counted the model's own processor
  worker, which the promotion retires first, an E2B on the processor with the machine near its
  cap was offered no card room. Five tests, each red with its part off. Auto-manage's file-size
  estimate is #297. Not taken: cutting a card/processor split at the first shared
  source — it would cost more processor layers than the copy it saves; measured question in #295.
  `docs/invariants/memory.md` § "Tensors the loader keeps in the machine's memory".
- #115 (3c) — **Gemma 4's E2B and E4B run (text), whole on one computer.** Their last layers (the
  E2B's 20 of 35) keep no cache and attend the last earlier layer of their kind's
  (`split::kv_sources`; `LayerWeights::kv_source` / `kv_slot` on every executor path), and every
  layer adds an embedding of its own gated by an input built from the tokens (`split::per_layer`,
  `executor::scaled_output`), its 1.9 GB table kept in the machine's memory. Against llama.cpp
  master: tiny f32 files with either mechanism or both agree at worst 0.99983-0.99998, whole and
  in 5- and 1-token pieces, and either mechanism off drops the worst position by 0.15-0.20; the
  real E2B Q4_K_M on the processor reads 28 tokens at worst 0.99624 / median 0.99983 (llama.cpp's
  own repacked-vs-plain: 0.99415 / 0.99987) and 1,536 tokens, past its 512-position window, at
  median 0.99974 (its own 0.99971), the worst positions the ones its own two builds part at.
  On the card, whole and split with the processor (20 or 14 layers on the card), every worst and
  median sits inside llama.cpp's own card-vs-processor spread. The E4B is a softer file — llama.cpp's
  two builds part at a median of 0.9989 on 28 tokens — and we sit closer to each of them than they
  sit to each other. On 1,536 tokens its worst position (0.930 on the processor) is one where llama.cpp's own card parts from its processor nearly as far (0.943). A review before the commit found that a split
  leaving a shared layer on the processor and its source on the card failed every decode step (a
  device mismatch); fixed where the history is read (`layers::history_beside_queries`), checked
  failing on the build before and agreeing after.
  Splitting them across computers is #294, their card decode graph #295, the table's memory
  charge #296. `docs/invariants/inference.md` § "Gemma 4's E2B and E4B"; gotcha #884.
- #115 (3b) — **Gemma 4's 26B-A4B runs (text).** Every layer runs a dense GELU feed-forward
  BESIDE 128 GELU experts (8 per token), each branch with its own norms; the router reads its own
  norm of the layer's input with its own scale, and each expert's answer has its own scale; the
  experts' gate and up come as one stack, cut in two at load (`layers::DenseBeside`,
  `ExpertScales`, `split_expert_stack_rows`; every executor path runs a layer's feed-forward
  through `FfnVariant::forward`). Against llama.cpp master: a tiny f32 file with every expert used
  agrees at 0.99998 on the processor and against llama.cpp's card, whole, in pieces and split, and
  each mechanism set to identity in a copy moves llama.cpp off our unchanged logits (0.63-0.98)
  while our run on the copy follows. The tiny top-2 files' gaps are llama.cpp's own: its card and
  processor builds part there by as much as either parts from us. The real Q2_K file (bartowski,
  10.9 GB) on the processor: median 0.99911 / 0.99917 on 28 / 1,536 tokens against llama.cpp's
  own repacked-vs-plain 0.99895 / 0.99907. On the card with every layer's attention there and the
  experts in RAM, some on the card, or 10 layers on the processor: median 0.99856-0.99914, worst
  0.71-0.75 at `<bos>` where llama.cpp's own card reads 0.70 against its processor. Its experts
  run on a card since #292 (below).
  `docs/invariants/inference.md` § "Gemma 4's 26B-A4B".
- #292 — **Gemma 4's 26B-A4B reads a prompt with its experts in RAM twice as fast (each chunk
  after the first 3x), and its card-held experts run as one launch.** Its down experts are stored in a 32-block type (Q4_0 in a
  Q2_K or Q3_K upload, Q5_0 in Q4_K_M, Q5_1 in Q5_K) that the card's expert kernels did not read,
  so a card ran them one expert at a time and RAM-held ones read a prompt on the processor
  (~10 s a 512-token chunk). The vendored kernels read Q4_0, Q4_1, Q5_0 and Q5_1 now
  (`layers::moe_card::kernel_type`, the one list; guard
  `every_type_the_card_moe_kernels_are_handed_is_launched`); against llama.cpp the card reads every
  worst and median inside llama.cpp's own card-vs-processor spread. Reading a prompt on the card
  first came out SLOWER (23.6-24.8 s for the first chunk): under WSL2 the card's memory pool
  stalls seconds whenever it grows after pageable copies. A prompt forward that copies weights to
  the card now grows the pool once, beforehand, by what it will take — the cache it allocates
  included (`SplitModel::prime_pool_for_prompt`, `SWARMLLM_PROMPT_POOL_PRIME=0`) — and the take it
  is measured at no longer counts that cache, which charged a reserved request's whole cache to
  every later chunk. With every expert in RAM a chunk now reads in 3.1-3.2 s after the first
  (llama.cpp: 1.5 s), the first in 4.0-10.3 s — its spread is #293. IQ uploads (IQ4_NL down experts) do not load in this build
  at all: the header probe refuses one before a byte moves. `docs/invariants/inference.md` § "The
  card's expert kernels read the 32-block types" and § "A prompt forward that copies weights to
  the card grows the card's pool once"; gotcha #883.
- #265 — **A node holding part of a model across its graphics card and its processor no longer
  fails a request that gives it a second run of layers.** Field report #037 (RTX 3060 12 GB,
  Qwen3-30B-A3B, v0.3.239 and v0.3.240): the worker took the first run as a split (26 layers'
  experts in RAM) and the next was refused on the card alone — `8 layers 40..48 … need about
  3016 MB more`, `26 layers 13..39 … 9802 MB more` — with the machine's memory cap far from
  reached; the worker was in use, so it could not be retired, and the request failed where a whole
  copy ran locally in 6.7 s. Worse, the worker's loader applied the spawn's process-wide
  `--gpu-layers` / `--cpu-moe-layers` to every range, so the refusal priced a placement it would
  never have used. Growth now has the spawn's split rung (`charge_segment_as_split`, after the
  reclaim and before retirement): `card_split` against the card's room, each share charged to its
  own budget and recorded on the range (`RangeCharge`), the machine's through the requester's cap.
  The worker is told each range's placement before any forward for it (`DaemonMsg::PlaceRange`,
  under `spawn_lock`) and the loader asks `split::placement_of` at all three load sites. Tests
  `a_running_split_worker_grows_a_range_across_the_card_and_the_processor` (red with the rung
  skipped — report #037's own refusal — and red without the whole-card placement message),
  `a_range_loads_where_the_daemon_placed_it_and_any_other_as_the_spawn_said` (red with the table
  ignored). Checked on this RTX card the same day in report #037's shape (one part held by a second
  node, so the worker takes a second run): v0.3.240 refused it ("need about 8584 MB more"), the fix
  placed it across the card and the processor (4,205 / 4,379 MB) and answered. The planner's bound
  is unchanged (gotcha #806). What is left is #291.
- #290 — **A connection that has just died no longer reads as a peer that stopped answering.** The
  live node logged "Peer has not answered 20 requests in a row — closing the connection" 16 times
  on 2026-10-16 between 06:51 and 07:20 (1-2 a day before), mostly against field report #037's
  node. In 14 of them the warning and an `io_error` close share the same microsecond: the transport
  had lost the connection, every request in flight on it failed at once (QUIC's `Io(NotConnected)`
  "connection lost" — 11 of the 14 logged failure lines; 3 were timeouts), and those failures
  reach the guard BEFORE the close event that resets its count. Once (07:19:40) the peer had a
  second connection, and the guard's `disconnect_peer_id` closed that live one too
  (`reason=clean_close`). The guard is for a connection that stays up while nothing comes back, so
  it now counts only `Timeout` (and `UnsupportedProtocols`, as before) — decided inside
  `note_rr_failure` (`failure_counts_as_silence`), the one entry point. Lighthouse's peer manager
  draws the same line (`handle_rpc_error`: no penalty for a disconnection). The thresholds (20, 8
  after 90 s) were measured on runs of timeouts and are unchanged. Test
  `a_connection_the_transport_lost_is_not_counted_as_a_silent_peer`. Not shown: whether the warning
  ever closed a connection that would otherwise have recovered — the fleet's warning count after
  the release is the check.
- #287 — **A prompt is no longer priced from a forward that timed a load or a round trip.** Field
  report #037: a 5-segment route was predicted `predicted_ms=63844516` (17.7 h) with
  `prompt:63277600`, and took 145 s. The prompt term multiplies a measured ms per (layer × byte) by
  the prompt's tokens, and that figure was the one the TIMEOUT reads — cold samples kept on purpose
  (a forward that loaded the model), and forwards of a few tokens whose ~700 ms is the visit, not
  the bytes, then multiplied by a prompt ninety times longer. A prompt is now priced from its own
  figure (`ModelSpeed::prompt_ms_per_layer_byte`): warm forwards only, and one carrying less than
  `PROMPT_RATE_MIN_BYTES` (256 KB, `GOODPUT_SAMPLE_MIN_BYTES`' number) may lower it but never set
  or raise it — BBR's app-limited rule, inverted for a time per byte. With none, the device-class
  prior, as for a node never measured. This node's own split segments, timed under its id all
  along, now price it too (they were ignored behind a comment saying they were never timed). The
  timeout and ranking figures are unchanged. Test
  `a_prompt_is_priced_from_warm_forwards_large_enough_to_measure_a_rate`. The same report's
  privacy notice ("adding about 533 s", the request took 196 s) is priced by the same terms —
  re-read it on the next report before changing anything there. What remains of the cost model's
  error is #3's.
- #288 — **The diagnostics no longer say "suspect that peer" for this node's own refusals.** Report
  #037's two failures were both this node's memory refusing its own segment of a split; the list
  named the peer the split ran through, and the repeated-peer note told the tester to suspect it.
  Each listed failure now carries whether that peer could have caused it — the penalty's answer
  (`failure_is_penalty_worthy`), not a second reading of the error — shown as "(not that peer's
  doing)" and left out of the note. Test `a_request_its_reader_left_is_cancelled_with_what_it_was_shown`
  (router tests), extended.
- #289 — **A long prompt is planned at its real length.** Report #037: the first plan read
  `prompt_tokens=Some(2495)` for a prompt of 3,729 — the router's four-bytes-a-token estimate, which
  under-counts code and dense text, priced the prefill and every holder's conversation memory for
  two-thirds of it. A prompt the estimate puts at 1,024 tokens or more is now rendered and counted
  (`cached_prefix::worth_counting`, ~46 ms for 2,744 tokens) wherever this node holds the model's
  header; a short chat keeps the free estimate. Test
  `a_long_prompt_is_counted_exactly_and_a_short_one_is_estimated`.
- #286 — **A part fetched for one request is no longer deleted again when that request's demand
  fades, and a model held whole is no longer holed by the replica count.** Field report #037
  (2026-10-16, an RTX 3060 node holding all 34 parts of Qwen3-30B-A3B): prune deleted part 25
  (`holders=3 target=2`), the download pass fetched it back, prune deleted it 28 s later — three
  times in 13 h — and a request 0.3.238 answered locally failed on the hole. Our own log showed the
  same churn across the swarm for a week: `e561df35` dropped 53 parts of a 14B in a day and a half,
  `d7054949` was at it on v0.3.240. Cause: both passes ask the one predicate (`would_shed_copy`,
  gotcha #795), but its target moves while the part does not — one request lifts the demand
  factor to 1.5 (2 → 3 copies) for ~30 min, a ninth peer lifts the floor 2 → 3, and a restart (every
  update) forgets the demand. Three changes: prune sheds against the highest target of the last
  24 h (`auto_manage::target_hold`, Kubernetes HPA's scale-down stabilization, kept in redb
  `measured_here` across restarts) while the download pass reads it as it is; a part of a model
  this node holds whole (counting the part, so both passes agree) is kept below urgent disk
  pressure (`holds_whole_model_counting`, now `completes_model_here_counting`, #320); a part that arrived within 30 min is not pruned below
  eager pressure (it only lowered the score). Disk pressure still lowers the held target, so a
  filling disk frees space as before. Tests `a_part_fetched_on_demand_is_kept_once_the_demand_decays`,
  `a_part_of_a_model_held_whole_here_is_not_shed_by_the_count`,
  `prune_waits_for_a_part_that_has_just_arrived` (each red with its change off) and
  `target_hold::tests::*`. `docs/invariants/memory.md` § "A target that moves is held for a day".

**Closed 2026-10-16 by the doc sweep — shipped earlier, found still listed open** (#172)
- #172 — **The greedy fallback assigner prices a long prompt.** `scheduler::estimated_cost_per_layer`
  takes `prompt_tokens` and charges the prefill term at the search's rate
  (`parallax::prefill_ms_per_layer`), less what the candidate already holds
  (`NodeCandidate::prompt_tokens_held`), and `distributed_exec.rs`'s reuse branch says why later
  turns keep the first turn's route (KV affinity). Shipped in `af5159cb` (2026-10-11), released in
  v0.3.235.

**Closed 2026-10-15/16, released in v0.3.240 (2026-10-16)** (#276, #277, #278, #279, #115 (1), #280, #115 (3), #282 calls + channel, #283, #281 (1), #281 (2), #281 (3), #281 (4), #281 (5) — #281 closed, #284, #282 closed)
- #282 (residual) — **A token the model file says is never to be sampled is never sampled.**
  Gemma 4's files list `tokenizer.ggml.suppress_tokens` (the 12B: `<audio|>`, `<image|>`), which
  llama.cpp merges into every request as a −∞ logit bias (`common/sampling.cpp`); nothing here
  read the key. Now the output HEAD adds a row built once at load (`split::loader::
  suppressed_logits_row`) inside `SplitModel::logits_from` — the one place a forward makes logits,
  single, batched and captured alike, so the 67 argmax and sampler sites never need to know (the
  batched loop's own copy of Gemma 2's soft cap went with it). The value is
  `SUPPRESSED_LOGIT` (−1e30), not −∞: a speculative check's rows from a peer must be finite
  (`verify_rows_are_finite`); probability 0 after softmax either way.
  `compare_logits_reference.py` compares those columns at llama.cpp's raw values, after
  checking ours are out of reach. Test `a_suppressed_token_is_out_of_reach_on_every_forward_path`
  (single, every position of a check, a batch of two; red with the row off). Not covered: a whole
  file named in `inference.model_path` runs through llama.cpp's own sampler chain (the `llama`
  feature's legacy executor), which is built without them.
- #284 — **A plain chat reaches Qwen3, Gemma 4, Phi-3.5, Phi-4-mini, GLM-4 and TinyLlama the way
  llama.cpp shows it.** A request with no system message was given "You are a helpful assistant."
  whenever the model's template wrote no system turn of its own (#100's rule, kept for a TinyLlama
  failure); Hugging Face's `apply_chat_template`, llama.cpp, vLLM and Ollama add none, so those six
  families were asked every plain question with a turn they are never shown elsewhere. Now
  nothing is added (`chat_template::without_blank_system_turns` only drops a BLANK caller system
  message, so a template's own default — Qwen2.5's, Llama-3.x's — still applies). TinyLlama's old
  failure no longer reproduces. Measured on an isolated node (`examples/reply_ab.sh`, processor,
  greedy) against llama.cpp on the real files: "What is the capital of France?" identical on
  TinyLlama, Phi-3.5, Phi-4-mini and GLM-4; Phi-4-mini's code reply now agrees with llama.cpp's
  for 232 characters, 94 of 95 tokens its top pick, worst a 0.075-logit rank 2 (with the turn:
  differed from the first word); TinyLlama's code reply parts from llama.cpp's at "returns" /
  "calculates", which llama.cpp itself scores 14.160 / 14.148. Byte-exact renders of Gemma 4's,
  GLM-4's and Qwen3's shipped templates against jinja2 pin it
  (`a_template_with_no_system_turn_of_its_own_is_given_none`, red with the turn restored).
- #281 (5) — **A long Gemma conversation is kept for the next turn again.** The prefix cache sized
  a snapshot by its byte budget over what a position weighs on EVERY layer
  (`PrefixCache::positions_ceiling`), but a snapshot copies a slid windowed layer only from where
  it starts (`snapshot_at`) — and can never stop short of where that layer still answers
  (`serves_from`, near the prompt's end). Gemma 4 12B on a processor weighs 786 KB a position:
  the default 2 GiB held a snapshot to 2,730 positions, so no Gemma 4 prompt past about 3,000
  was kept at all, and every turn of an agent conversation (4-14K tokens) re-read it whole. Now
  the ceiling is `(budget + slid_away) / per_position` (`prefix_cache::SnapshotCost`): each slid
  layer is charged the positions it holds — exact, since that is what the copy takes.
  Measured (Gemma 2 2B on a processor, an isolated node, `prefix_cache_max_mb = 1200`,
  `~/swarmllm-g3/pc281.sh`): a 6,943-token prompt kept whole (1,153 MB, where the old ceiling of
  5,907 fell short of its window and kept nothing); the next turn read 6,943 of its 6,974 tokens
  from it — 36.8 s against 357.9 s cold, the same 16 greedy tokens.
- #281 (4) — **A computer is offered a long Gemma prompt by the memory it will really use.** A
  coordinator weighing whether a peer has room for a prompt charged every layer the widest
  layer's cache at every position (`scheduler::kv_bytes_per_position_per_layer`): Gemma 4 12B at
  32,768 positions on a processor at 24 GiB, where the peer keeps ~4.2 — enough to leave a peer
  out of a plan it would have admitted. Now the geometry a coordinator learns from the header
  carries the loader's layout (`GgufTensorMeta::kv_layout`), and the bound
  (`scheduler::prompt_kv_bytes_per_layer` → `KvLayout::growth_on_peer`) prices each layer by its
  own shape and storage, a windowed one at its window only where the peer says its caches slide:
  `features::KV_WINDOW_SLIDES`, bit 22, a statement like bit 19, advertised where the node
  attends windows, its caches slide and it reads a prompt in chunks no wider than
  `COPIED_CHUNK_MAX_TOKENS` (`kv_budget::windows_advertised`) — the width it is charged at. An
  older peer says nothing and is charged every position, as before; a card keeping the f16
  mirror never slides. And a peer's own admission no longer charges a forward it reads in its
  own chunks at the forward's whole width (`model_worker::width_handed_to_the_caches`). Residual
  — the prefix cache's budget — stays open as #281.
- #281 (1) — **A windowed layer's cache keeps its window, not the whole conversation.** Gemma
  2, 3 and 4 attend over a window on most layers, and those layers' caches kept every position
  (a Gemma 4 12B conversation at 32K tokens: 10.5 GiB of cache, now ~1.0 by the layout's
  arithmetic; llama.cpp ~0.8). A full buffer moves its newest W − 1 + 64 positions to the
  front instead of growing (llama.cpp's `n_swa + n_ubatch`); a rewind past what the window
  still answers is refused, never served from lost keys; a prefix-cache snapshot records where
  each layer starts, is found only for a prefix its window answers (vLLM's sliding-window rule)
  and never leaves the node. Gemma 3 1B at 1,500 tokens and 4B at 2,000, past their windows:
  logits bit-identical with the slide on and off, and as close to llama.cpp as before
  (worst 0.999451 / 0.998622, top-1 12/12 both); on the card the half cache slides as well —
  mid-decode, past the CUDA graphs — bit-identical on and off, the 4B whole and split 17/34
  equal to the card run before it. Residual — the estimates — stays open as #281.
- #281 (2) — **A Gemma model is charged for the window its sliding layers keep, not the whole
  conversation.** The worker priced a conversation at every position on every layer at the
  widest layer's rate, so a node holding ~1.0 GiB of cache for Gemma 4 12B at 32K tokens
  refused prompts as if it needed 12 GiB, and reported a context to match. Now one price
  (`kv_budget::KvGrowth`, built per layer by the loader) charges a global layer every position
  at its own width and a windowed one no more than its window, a rewind and the widest forward
  it is handed (vLLM's `SlidingWindowSpec.max_memory_usage_bytes`, llama.cpp's
  `n_swa + n_ubatch`); the loader's budget, admission, the snapshot plan and the context
  reported all read it. The head-room guard reads the buffers instead of an estimate: a
  forward's claim is what its append allocates, slide and f16 mirror as `append` treats
  them. A snapshot is restored to each layer as that layer keeps its cache. Residual — the
  daemon's estimates and a peer's bound — stays open as #281.
- #281 (3) — **A Gemma model is placed by the memory it will use.** Before loading a model this
  node weighs it — may it go on the card, how much of it fits beside the processor, how many
  layers to offer a plan — and every one of those charged each layer the widest layer's cache at
  every position: Gemma 4 12B's 4,096-position admission at 3 GiB (processor) where the worker
  holds 1.1, its global layers' narrower shape ignored. Now the loader's own layout
  (`kv_budget::KvLayout`, read once off the header: the layers that keep a cache, each by its
  own shape and storage, a windowed one by its window and the widest prompt forward its reader
  takes, `prefill_pacer::widest_prompt_forward`) prices admission's footprint, the cost curve and
  the card/processor split alike (`VramFootprintInputs::kv`), so the estimate and the worker's
  budget cannot price one model differently. MLA's decompressed cache, which the estimates
  priced at its grouped shape, is now priced as the loader keeps it too. Residual — a peer's
  bound — stays open as #281.
- #282 (calls, channel) — **Gemma 4's tool calls are tool calls.** It writes
  `<|tool_call>call:name{key:<|"|>text<|"|>,n:2}<tool_call|>` — its own syntax, not JSON — and
  every call reached the client as reply text. `tool_parse::try_gemma4` reads it as llama.cpp
  master's grammar does (`common/parsers/gemma4.cpp`) and rebuilds the JSON object, keys in the
  order written; `<|tool_call>` holds a streamed reply back like every other call marker. Its
  thought channel (`<|channel>thought … <channel|>`) is a reasoning block beside `<think>`
  (`inference::REASONING_BLOCKS`, the one list both finalisers read) — after a tool's answer the
  12B opens an empty one itself, which led the reply. Fixtures are the 12B's own greedy replies
  under llama.cpp; six tests, all red with the fix off. The file's `suppress_tokens` closed
  the next day (#282 (residual), above).
- #283 — **A conversation's earlier tool calls and their answers reach the model.** OpenAI's
  `to_chat_message` dropped `tool_calls` / `tool_call_id` (Responses goes through it too) and
  Anthropic flattened a call to `[Tool call: name(input)]` text, so the template saw an empty
  assistant turn where the call was — and Gemma 4's, which renders an answer only beside its
  call, dropped the ANSWER: the model never saw what its tool returned. `ChatMessage::tool`
  (`ToolTurn`, flattened, absent from the JSON when empty — additive on the wire) carries the
  structure from every surface; `chat_template::tool_turns` decides at the one choke point how
  this template is shown it: as is where a probe render (once per template, minja's
  `chat_template_caps`) finds calls and answers in the prompt, otherwise as minja's polyfills —
  the call as the `{"tool_calls": …}` JSON the prose description asks for, the answer as a user
  turn merged to keep alternation. Gemma 4 and Qwen3 now render a tool conversation exactly as
  jinja2 does.
- #115 (3) — **Gemma 4 runs (text, the dense 12B and 31B).** Each layer's attention has its own
  shape — on the 12B, five layers in six have 8 key/value heads 256 wide over the last 1,024
  positions, every sixth one head 512 wide over everything, with no V projection (V is K) —
  read per layer as llama.cpp reads it for every family (`split::attention_shapes`). Q and K are
  normed with weights and V without, the attention is unscaled, each layer's output has its own
  scale, and the windowed layers turn their own 256 at base 10,000 without the file's
  proportional factors. Its tokenizer has its own rules (spaces as `▁` with none in front,
  newline runs whole, BOS always; only DECLARED markers matched whole — 75 ordinary entries are
  `<…>`-shaped): 18/18 against llama.cpp master's own tokenizer. The chat template renders as
  jinja2 does. Measured against llama.cpp MASTER on the processor (no Python binding runs Gemma
  4): a tiny f32 file with every mechanism agrees at 0.99996, whole, in pieces and split (each
  mechanism off: 0.23-0.29, the window off −0.05). The real 12B Q4_K_M sits as close to llama.cpp as llama.cpp's own two processor kernels sit to
  each other: worst 0.99402 against their 0.99380 on 28 tokens, median 0.99815 against 0.99833
  past the window at 1,536 (window off there: 0.784; windowed layers on the global rotation:
  0.771; queries scaled as other families: −0.729). On a card (split 24/48) against llama.cpp's
  own card: median 0.9991, worst 0.988 at the one position every implementation finds hardest
  (llama.cpp's card vs its processor: 0.992 there); past the window, median 0.9981 against
  llama.cpp's own card-vs-processor 0.9979. Getting there fixed the vendored flash kernel's
  512-wide block, wrong at every shape on RTX 30/40 cards (gotcha #868). The 31B has the 12B's layout and is unchecked.
  The 26B-A4B (experts) and E2B/E4B (per-layer embeddings) are refused by name until built;
  LoRA refuses the family. `docs/invariants/inference.md` § "Gemma 4: each layer's attention has
  a shape of its own".
- #280 — **a model file that declares YaRN context extension is served as llama.cpp serves it.**
  Before, such a file (Qwen2/Qwen3 uploads re-converted for 128K context) loaded with plain RoPE:
  the slow dimensions turned 4x too fast and the attention ran unlifted, at EVERY length. Now
  `split::rope::RopeScaling::Yarn` applies llama.cpp's ramp (`ggml_rope_yarn_corr_dims`, beta 32 /
  1) and lifts cos/sin by 1 + 0.1 ln(factor), into the same table builder. Qwen3-1.7B Q8_0 with
  Qwen's documented keys (factor 4, original context 32,768) against llama.cpp 0.3.16: worst
  cosine 0.99943 at 300 tokens (the plain file: 0.99971), median 0.99988 at 1,500; with
  `SWARMLLM_ROPE_SCALING=0` (the old behaviour) 0.971 / 0.966. The 1,500 run's one low position
  (0.99666) moves to 0.99888 with llama.cpp's KV cache in f32, and a tiny f32 file with the ramp
  across 5 of 16 pairs agrees at 1.000000 (0.997 / 0.987 without). DeepSeek-2's log multiplier
  stays with #116. `docs/invariants/inference.md` § "A file's RoPE scaling is read for every
  family".
- #115 (1) — **Gemma 3 runs (text).** The `gemma3` family loads, splits and is offered for
  download: Gemma 2's layout plus per-head Q/K norms, no soft-capping, a window on five layers in
  six (512 positions on the 1B, 1,024 from the 4B up), and a rotation per layer — the windowed
  layers turn at base 10,000 unscaled, the global ones at 1,000,000 with the file's linear factor
  (8 from the 4B up). `split::rope::LayerRope` reads both, for any family: no family's
  `rope.scaling.factor` had been read before (YaRN is #280). The 27B takes #279's query scale.
  Against llama.cpp 0.3.16 on the processor: gemma-3-1b-it Q4_K_M worst cosine 0.99977 at 300
  tokens and 0.99945 at 1,500, gemma-3-4b-it Q4_K_M 0.99872 / 0.99862 at 300 / 2,000 (median
  0.9994 / 0.9997, top-1 11/12 and 12/12); a tiny f32 file with every 4B mechanism agrees at
  1.000000 whole, in pieces and split. Each mechanism off in the same binary moves it far:
  windowed layers on the global rotation 0.769 (4B), no scaling 0.992, no window 0.742 at 2,000.
  The tokenizer matched llama.cpp on 8/8 cases. On the card: 1B 0.99916, 4B 0.99401 whole and
  0.99439 split, its low position one where llama.cpp's own card and processor differ by 0.9969.
  `docs/invariants/inference.md` § "Gemma 3: each layer turns at its own rate".
- #279 — **Gemma 2 27B's attention uses its own scale.** llama.cpp takes a 46-layer Gemma 2 for
  the 27B and scales its queries by 1/sqrt(n_embd / n_head) = 1/sqrt(144), as Google's reference
  does; we used 1/sqrt(head_dim) = 1/sqrt(128), so every 27B attention score was 6% too large.
  The queries are now multiplied where llama.cpp scales them (`split::query_scale`,
  `LayerWeights::query_scale`, after RoPE) — only on that shape, one elementwise op per layer. On a
  46-layer tiny file (`examples/make_tiny_gemma_gguf.py`, `QK_STD=0.25`) worst cosine 0.727 →
  0.99987 against llama.cpp; same binary, `SWARMLLM_QUERY_SCALE=0` is the old arm. With the file's
  decisive default weights, 0.657 → 0.966: the rest is rounding (llama.cpp's f16 KV cache alone
  moves it to 0.992) amplified over 46 sharp layers. Gemma 3's 27B (62 layers) takes the same rule
  when #115 lands. `docs/invariants/inference.md` § "A family's attention scale is read".
- #278 — **Gemma 2's replies past 4,096 tokens now match llama.cpp's.** Gemma 2 alternates
  layers that see only the last 4,096 positions with layers that see everything; the loader
  never read the window, so every layer saw everything. Shorter conversations were unaffected,
  which is why no check caught it (gotcha #863). At 4,600 tokens the last positions sat at
  cosine 0.986 from llama.cpp, top-1 10/12; now 0.99939, 11/12. The window is read as llama.cpp
  reads it (`split::SlidingWindow`) and cut from the attention's keys where it binds
  (`LayerWeights::keys_in_window`, `ForwardMasks`); a conversation shorter than the window runs
  exactly the old path. It is also the first half of Gemma 3 (#115). Left as it is: a windowed
  layer still keeps every position in memory (llama.cpp keeps the window only).
  `docs/invariants/inference.md` § "A sliding window is cut where it binds".
- #276 — **a computer with no graphics card that holds an idle model is offered the whole of the
  next one.** The planner weighed its memory with the idle model's share counted as spent, but the
  load that follows unloads that model first, so it offered fewer layers than it would load and
  sent the rest to other computers (#125's shape, fixed on the card). It now counts what the load
  would free (`RamBudget::headroom_after_reclaim`). Room left for a drafter is still weighed as it
  stands, because a drafter never frees memory and the target frees only what it needs. Left as it
  is: the memory a node advertises to peers counts an idle model as spent on both devices.
  `docs/invariants/memory.md` § "A model is admitted against the card as it stands NOW (2026-09-29)"
  ("The processor's planning bound counts it too").
- #277 — **a computer on a metered connection no longer uploads for other computers.** Since
  v0.3.239 the downloads nobody asked for waited; parts served to peers, new work for peers and
  relaying still went over the metered link. Now `SharedState::metered_hold` holds back all three:
  the capability withdraws inference, a request that comes anyway is refused out loud
  (`admits_a_new_request`, no penalty), parts and prompt-cache snapshots are answered as a miss,
  and the automatic relay stops. Running work, the owner's requests and gossip go on; no LAN
  exemption (a VPN peer looks local). Found beside it: the relay switch was read from the boot
  snapshot, its DHT record was never withdrawn and the circuit server was not gated at all.
  `docs/invariants/network.md` § "A metered connection holds back what this node uploads for
  others".

**Closed 2026-10-15, released in v0.3.239 (2026-10-15 10:46 UTC)** (#275's reply-sized charge; #275 (2), judged beside the plan; #250's stale card reading; #3's cold-load share)
- #3's cold-load share — **a restarted node no longer prices its first cold load as a slow
  loader.** Each node's measured load speed (#129 (1)) lived in memory, so every restart — every
  node at every release's auto-update — went back to the 10 s/GiB prior and advertised nothing
  until it next loaded a model. Fitting #3's logged cost terms (this node, 2026-10-14) found it
  the largest single error: 34 first requests after a restart at 0.52× their price, ~58 s charged
  for a load that took ~9-20 s. The rate is now stored and restored (`daemon::state::load_rate`,
  written on the monitor tick when it changed and at shutdown); the next timed load moves it.
  `docs/invariants/scheduling.md` § "A candidate that must first load the model is charged for it".
- #250 (report #036) — **a model the card holds only part of is no longer planned here whole
  after another program has taken the card.** The planner weighed the card reading the last
  admission had stored (~7.5 GB, taken while the card stood free) after Ollama had taken ~10 GB,
  planned a fresh card/processor split of Qwen3-30B-A3B here, and the spawn — reading 556 MB —
  was refused 18.7 GB of system memory; the re-plan sent 8 layers abroad (0.2 tok/s cold). Now
  the plan reads the card live where a spawn could follow (`card_budget_for_planning`, one
  reading per 2 s) and a fresh split's processor share is weighed against the requester's cap as
  the spawn weighs it. A manual unload reports what the model's workers held
  (`memory_committed_by`), not the whole file's estimate ("~20351MB VRAM freed" on a 12 GB card).
  Found beside it: a card other programs had filled past our reserve gave a 0 MB budget, which
  admission reads as "none configured" and admits anything against — now 1 MB, admitting nothing.
  The prompt-pricing and `net:0` halves stay open as #250 and #3.
- #275 (1) — **the room a split leaves for its drafter is the reply's cache, not a typical
  conversation's.** The room, the choice and the drafter's spawn priced its cache at 4,096
  positions whatever the reply — 896 of Qwen3-1.7B's 2,899 MB for a chat that fills 1,024 — so
  the cut took three more of Qwen3-30B-A3B's layers off this node than the drafter needed. Now all
  three price the prompt and one reply quantum (`model_worker::drafter_cache_positions`, the
  figure the drafter's worker already admitted at its first call; `vram::KvCharge`, carried on
  `Tenancy::Guest`), and the worker's grant to grow is unchanged. Live: the planner's room on a
  real request 20 → 23 layers (`drafter_cache=Conversation(1024)`); the speed A/B waits for the
  Australian peer to offer room (it refused every arm that hour). A/B
  `SWARMLLM_DRAFTER_KV_CHARGE=typical`. (2)-(4) stay open as #275.
- #275 (2) — **a drafter is judged beside the layers the plan loads here, not those loaded so
  far.** It loads after them, so on a cold start "would it load now" said yes beside nothing:
  seen choosing Qwen3-1.7B beside 14 held layers of Qwen3-30B-A3B, refused at its spawn once the
  plan grew them to 28, and that reply ran with no guessing at all. The choice now asks the
  planner's question of the plan it made (`ModelProcessPool::guest_would_be_admitted_beside`:
  the layers the plan ADDS within `max_hostable_layers_beside_guest`), so such a drafter gives
  way to a smaller one or to the n-gram loop. vLLM likewise loads the drafter beside the target
  before it sizes caches. A/B `SWARMLLM_DRAFTER_BESIDE=loaded`. (3)-(4) stay open as #275.

**Closed 2026-10-14, released in v0.3.239 (2026-10-15 10:46 UTC)** (#273, #274; #137's 8-thread half; #275's first half)
- #275 (first half) — **a split of a big model guesses ahead on a node without a graphics card,
  where the peers offer room.** The drafter was chosen by size and asked for memory only at its
  spawn, after the planner had given this node's memory to the target's layers: 28 layers of
  Qwen3-30B-A3B here (6,967 of 7,995 MB) left Qwen3-1.7B (2,899 MB) refused on every request.
  Now (a) a drafter that would not load is not chosen (`guest_would_be_admitted`), so the n-gram
  loop runs instead — 2.75-3.07 tok/s against 2.51-2.83 for the refused drafter; (b) each
  planner first tries a plan with this node's bound cut to leave the drafter room
  (`candidates_leaving_room_for_the_drafter`), ONLY where every peer is held to what it offers.
  Live split with an Australian card peer: 2.45-2.69 tok/s uncut → 4.10-6.94 with the drafter
  beside 14 layers (α 0.85-0.89); `share.sh`'s control arm (fewer layers, guessing off) shows the
  smaller share alone is worth nothing. Tried on relaxed rungs too, the cut handed the peer 34
  layers it had not offered, the peer refused and a reply was lost — so it never runs there
  (gotcha #852). The residual stays open as #275. Found verifying it: **a range a peer already
  holds could not be reached by a neighbour with room** (partial ranges off, the default), so a
  plan that moved between requests — the peer holding [14..48), this node asking [28..48) — was
  refused for eleven minutes until the peer let the range go; a rung with no route is now retried
  with the parts that meet held ranges (`parallax::route_over(.., meet_held)`).
- #137 (8-thread half) — **processor decode no longer slows down from 4 threads to 8.** Each
  small parallel region of a token (RoPE, attention, the small projections) paid a wake per rayon
  worker, because upstream rayon puts an idle worker to sleep within microseconds; the vendored
  `rayon-core` spins 200 µs first. Qwen2.5-Coder-7B Q4_K_M: 8 threads 172 → 123-127 ms a token, 4
  threads 126 → 121-122, replies identical. The entry stays open for 8 threads still not beating 4.
- #273 — **a split of a mixture-of-experts model guesses ahead with a held small model again,
  and of two copies of one drafter the lighter one drafts.** A real Thailand↔Belgium split of
  Qwen3-30B-A3B (2026-10-13) guessed with n-grams only (9.8% of rounds kept, 468 ms a token over
  a 222 ms link) although the node held Qwen3-1.7B: `engine_drafter::size_proxy` was layers ×
  width², which counts no experts, so the 30B-A3B measured below Llama-3.2-3B and no MoE model
  could ever have a drafter. The size is now scaled by stored over active bytes per layer
  (`GgufTensorMeta::stored_bytes_per_layer`, 8.5× at Q2_K and 9.6× at Q4_K_M from the real
  headers); dense models are unchanged. Equal sizes were tied by model id, which picked the fp16
  Qwen2.5-0.5B over its Q8_0/Q4_K_M copies; ties now go to the copy reading the fewest bytes.
  Checked on a real split (this node layers 0-27, an Australian peer 28-47, 2026-10-14): v0.3.238
  weighed only the Qwen2.5-0.5B copies (wrong vocabulary) and ran the n-gram loop; this build
  chose `drafter=qwen3-1.7b-q8-0` and ran DSD. Same route, prompt and 96 tokens at temperature
  0, a card build of each: 41.1-41.3 s a reply (2.33 tok/s) → 20.9-21.7 s (4.4-4.6 tok/s), 82 of
  109 guesses kept (α 0.85), γ settling at 6-7. Gotcha #849.
- #274 — **a speculative check on a processor costs ~20 ms an extra position instead of ~60.**
  The several-row AVX2 kernels took rows four at a time and ran a 1-3 row tail one row each —
  the WHOLE call of a check of 1-2 guesses, so a 3-position forward cost more than a 4-position
  one (7B Q4_K_M, 4 threads: 257 against 223 ms) and the guess-length controller chose zero
  guesses against processor peers. A 2-3 row tail now shares one pass (`avx.rs::tail_in_one_pass`,
  bit-identical): 2 / 3 positions 190 / 248 → 151-156 / 173-177 ms, llama.cpp 191-206 / 260-274.
  A/B `SWARMLLM_QMATMUL_TAIL=per-row`; `docs/invariants/inference.md` § "A 2-3 row tail shares
  one pass too".

**Closed 2026-10-14, released in v0.3.238 (2026-10-14 07:31 UTC)** (#270, #271, #272)
- #270 — **a computer that once held another upload of a model no longer refuses the swarm's
  copy of it.** Found checking the swarm after v0.3.237: the Proxmox node refused all four
  holders of qwen2.5-coder-7b-instruct-q4-k-m as "a different build" — it held NO part of the
  model, only another upload's manifest on disk since at least 2026-09-04, and kept it because
  `register_manifest` refuses a different build while the node has any origin hash of the model;
  prune and delete take the parts and leave those hashes, and the copy repair visited only models
  a node holds a part of or may fetch. The holders were right (part 3 rebuilt from HuggingFace by
  its tensor table: identical). Now each copy-repair pass forgets the origin hashes of a model
  with no part here and none on its way, and judges every directory under `models/` — a
  metadata-only one's manifest on disk included, replaced by the upload's when it is another's
  (`docs/invariants/network.md` § "One upload per model id", gotcha #846).
- #271 — **a node told to stop stops, even while its update finishes downloading.** At
  03:11:02 on 2026-10-14 the live node took a SIGTERM with v0.3.237 still downloading; the
  download finished inside the shutdown grace, and the node installed it and `exec`ed into it
  at 03:11:05 — so the stop restarted it, and the deploy that had sent the signal found the
  database held. `update_restart::restart_into`, now the only way to restart into an update
  (the automatic install and the dashboard's Apply & Restart), refuses once a stop was asked, and
  `install_and_restart` also asks before installing; the staged download is reused at the next start
  (`docs/invariants/state-and-config.md` § "A node asked to stop does not restart into its
  update").
- #272 — **the dashboard asks again about cloud models the node told it to wait on.** A 429 from
  `/api/admin/provider-model-status` (20 a minute, each one calls a provider) left every model in
  that batch marked "being asked" for the life of the tab, so their badges never filled in. A
  refusal now clears them and re-queues the batch for after the reply's `Retry-After`.

**Closed 2026-10-14, released in v0.3.237 (2026-10-14 03:01 UTC)** (#266, #267)
- #267 — **the node no longer opens a new dashboard tab every time it restarts.** With
  `ui.open_browser_on_start` (the default) every start opened one: each automatic update, each
  restart by a service manager, each restart by hand — beside the tab that was already open and
  reconnecting by itself. One machine restarted by test scripts collected 25 tabs holding 8.5 GB.
  As Syncthing does (`STRESTART`), a start an update made opens nothing
  (`update_restart::RESTARTED_BY_UPDATE_VAR`, set by `exec_into` on both platforms; the Windows
  handoff variable counts too, so the first update off an older Windows build is recognised); and a
  start after a run during which a dashboard was connected (the `dashboard_open` file, sampled every
  5 s and left alone at shutdown, so a crash is covered) waits up to 45 s for that tab to reconnect
  and opens one only if it does not. The page retries every ≤ 30 s (`WS_RETRY_MAX_MS`); guard
  `the_dashboard_reconnect_window_outlasts_the_pages_retry_backoff`. A launch with no dashboard open
  still opens one at once. Verified on the dev binary with a logging `xdg-open`: launch → 1 open;
  restart with a dashboard retrying → 0 and "reconnected — not opening another"; restart after it
  closed while the node was down → 1 open at 43 s; a start marked as an update → 0. Tests
  `the_process_an_update_starts_knows_an_update_started_it` (red without the marker),
  `only_a_launch_with_no_dashboard_open_opens_one_at_once`,
  `the_wait_for_a_reconnecting_dashboard_ends_either_way`. A Unix build from before this execs with
  no marker, so the update INTO this one still opens one tab.
- #266 — **your own requests are no longer held to the memory share you give the swarm.** Since
  #264 (never released) a model split between the graphics card and the processor charged its
  processor share to the memory limit, and on a computer with a card that limit is the SWARM's: half
  its RAM scaled by the contribution level, 31% — 4997 MB on 16 GB — at the default. The owner's own
  Qwen3-30B-A3B (5213 MB of experts kept in RAM, or 5530 MB as 26 whole layers) was refused with
  13 GB free; v0.3.236 ran it. As the owner's prompt reading already was (#119), the owner's own
  loads are now weighed against the owner's cap — `ResourceConfig::owners_ram_budget_mb`, 80% of
  RAM, or an explicit `resources.max_ram_mb` for everyone — and the swarm's keep theirs:
  `ModelProcessPool::ram_budget_now(requester)` is REQUIRED (`RamBudget::judging`) at the spawn,
  growth, reclaim, the processor-side cache grant, the refusal's note and the planner's bound on this
  node (the scheduler names the requester with `pipeline::requester_for_override`; a split a peer
  delegated here is the swarm's). The live anti-swap headroom still binds both. On the live node the
  same request that had been weighed at 4997 MB was weighed at 12792 ("for this computer's own
  requests"); that evening the machine had 5.7-7.4 GB free, and the anti-swap headroom refused it.
  Tests `the_owners_own_load_is_weighed_against_the_owners_memory_cap`,
  `a_budget_judges_the_owner_against_the_owners_cap_and_the_machine_for_both` (both red with the
  requester ignored), `the_owners_memory_budget_is_the_machines_whole_job_share_at_any_level`
  (`docs/invariants/memory.md` § "The owner's own load is weighed against the owner's memory cap").

**Closed 2026-10-13, released in v0.3.237 (2026-10-14 03:01 UTC)** (#257, #259, #260, #262, #263, #264)
- #264 — **a model split between the graphics card and the processor now counts its processor
  side against this computer's memory limit.** Only the card share was charged; the processor-held
  layers' weights (~5 GB of a 14B split), their conversation memory and the worker process sat
  outside `resources.max_ram_mb`, so a processor-only model loaded beside the split was admitted
  into memory the split already held, and the split's processor-side conversation memory had no
  grant of its own. The spawn now charges that share through the same memory admission, refusal
  and conversation grant as a processor placement (`process_pool::ram_charged_share`,
  `SpawnCharge`), and a departing split gives both charges back. On this node (Windows up 15.1 h):
  Qwen2.5-14B split 30/18, the processor share charged, its processor-side conversation memory
  granted 1,525 MB (`--kv-budget-bytes`, read by the worker in place of the machine's 12 GB), one
  reply in 20 s with the load. Tests `a_split_charges_the_machine_for_the_layers_it_keeps_on_the_processor`,
  `a_departing_split_gives_back_both_of_its_charges` (red without the fix). Left open: a later range
  loaded into a split worker — #265, closed 2026-10-16.
- #262 — **a model split between the graphics card and the processor no longer refuses a long
  conversation the card has room for.** The card's conversation budget was charged the caches of
  the layers on the PROCESSOR too, and the processor side had no budget of its own. On an 8 GB card
  the 30B-A3B split 26/22 was refused a 2,585-token prompt at position 2,176 — "other conversations
  are using 176 MB … a shorter conversation will not help", with no other conversation: the 176 MB
  was its own processor-side cache. Each cache is now charged to the memory it lives in
  (`KvHome`, `KvOccupancy::allocated_in`), a split is weighed in both (`SplitModel::kv_homes`: the
  growth claim, prompt admission and its per-memory promises), and its processor side gets its own
  budget from the machine's memory, reconciled at every claim. Real card (`~/swarmllm-262/kv.sh`,
  `test-split`, Windows up 14.5 h): v0.3.236 refused the 30B prompt, this fix read it and replied;
  the 14B 20/28 on 6,901 tokens replied the same on both. Tests
  `a_claim_is_weighed_against_the_caches_in_its_own_memory`,
  `an_admission_promise_is_kept_and_drawn_down_per_memory` (red without the fix). The split's
  processor-held weights, charged to no memory limit, were #264, closed the same day.
- #263 — **a split whose part on this computer cannot hold a long prompt's conversation memory
  now says so before any other computer reads the prompt, and the retry does not plan the same
  split again.** A tester's node (v0.3.236, RTX 3060 12 GB, Qwen2.5-14B, ~6,000 tokens, "start
  and finish on this computer") held 30 of 48 layers in three local segments between peers in
  Italy, Belgium and Thailand; each segment was admitted on its own as its forward arrived, so
  the LAST refused after the peers had read the prompt for 30-277 s, and the re-plan — which
  charged this node's held layers nothing for the prompt — built the same plan: two requests,
  four attempts, ~15 min of peers' time, no reply. Now (1) every local range is admitted at
  once before the first forward (`ModelProcessPool::admit_prompt` → `DaemonMsg::AdmitPrompt`,
  vLLM's `can_allocate` before prefill; a worker busy with another forward is waited 1 s, then
  each segment admits itself as before); (2) a refusal carries the layers the cache fits
  (`model_worker::layers_that_fit`), and the re-plan holds ALL this node's layers, held ones
  included, to that (`NodeCandidate::prompt_memory_refused`); (3) no planner's relaxed pass
  lifts a refused node's bound (`memory_refused_this_request`, greedy's `cap_of` — it had
  rebuilt the refused plan twice). Real path (`split_rig.sh latekv`, both nodes on the
  processor, A's cache 74 MB short of a 5,097-token prompt, Windows up 13.7-14.1 h): v0.3.236
  answered 503 after **261 s** with B reading the prompt twice; this fix answers 503 in
  **0.3 s** with one up-front refusal, one re-plan that stops at once, nothing logged on B, and
  the refusal naming the length that fits ("up to about 3584 tokens"). Tests
  `an_up_front_admission_survives_each_segments_first_forward`,
  `a_refused_prompt_says_how_many_of_its_layers_fit`,
  `a_prompt_is_admitted_up_front_only_where_a_worker_runs_the_model`,
  `an_up_front_admission_does_not_wait_on_a_busy_worker`,
  `a_replan_after_a_refused_prompt_holds_this_node_to_the_layers_that_fit`,
  `a_prompt_this_nodes_memory_refused_is_replanned_with_its_held_layers_counted`,
  `a_relaxed_greedy_pass_keeps_this_nodes_bound_once_its_memory_refused_the_request` (the
  planner tests red with their fix off). Left open: a PEER's segment is still admitted when its
  forward arrives, and a planner that knew each node's cache room could size the split before
  asking — both #252 (1). The tester's cold run that went to a peer was this node's own
  restart (a rig's), whose card budget was then 1,439 MB: chosen correctly.
- #259 — **"Keep on this computer" no longer refuses a model this computer holds whole and
  runs.** The request gate and the model's `whole_model_here` asked for a registration in
  `split_models`, and `auto_manage::scan` registers nothing past the graphics budget, so a model
  bigger than the card, held whole and run here as a card/processor split, read "not here":
  a tester's 14B (16/16 parts) and 30B (20/20) were refused in ~30 ms with the setting on,
  while the 14B answered `route=local segments=1` with it off. Both now ask
  `SharedState::parts_missing_here` (manifest, the registry's holdings, the files), and the
  refusal names the parts this computer lacks. Test
  `a_kept_model_held_whole_runs_here_without_a_registration` (red with the old gate). The same
  tester's follow-up that day found the same gate on a model that FITS the card: Mistral 7B (9/9
  parts) read `whole_model_here: false` and was refused in 70 ms while Ollama held ~9.6 GB of a
  12 GB card, and ran once it let go — the registration follows the card's free memory when the
  scan looks. Reproduced on the real path (`~/swarmllm-260/keep.sh`, `HOLD_FIRST=1`: 5,000 MB held
  before the node starts, Windows up 11.5 h): v0.3.236 answered `false` and refused in 204 ms with
  the report's sentence; this fix answered `true` and replied here as a card/processor split (6 of
  32 layers on the card, 21 s with the load), then whole on the card once the hold was gone
  (0.75 s for 32 tokens), the reply the same.
- #260 — **a model squeezed off the graphics card by another program moves back to the card,
  whole, once that program lets go.** Two halves. (1) The promotion check weighed the card
  budget the last ADMISSION had stored, so a model refused the card while (in the report)
  Ollama held 6.2 GB kept that squeezed figure: Mistral 7B stayed at 12 tok/s with ~8 GB free,
  and reached 55 tok/s only after an unload. `ModelProcessPool::promotion_weighed` now reads the
  card as it stands (`vram_budget_now`, the reading admission takes), only for a demoted worker
  idle past the floor, so a busy worker costs no `nvidia-smi`. Test
  `promotion_reads_the_card_as_it_stands_now` (red reading the stored figure). (2) Only a worker
  wholly on the processor was ever weighed, but admission SPLITS a model between the card and
  the processor before it gives the card up — the report's 1.4 GB "too small to be Mistral" was
  a split's share, and the real-path rig reproduced it (gemma-2-2b split 11 of 26 beside a
  5,000 MB hold, still split after the hold was released). `WorkerHandle::short_of_the_card`
  takes the split too, its own card share counted as given back, and a model larger than the
  card could ever give (`vram_budget_ceiling`) is never weighed, so a 14B split on an 8 GB card
  reads nothing. Tests `a_split_worker_moves_whole_once_the_card_has_room`,
  `a_split_worker_counts_its_own_share_of_the_card`,
  `a_model_larger_than_the_card_could_ever_give_is_not_weighed`. On the real path
  (`~/swarmllm-260/keep.sh`, Mistral 7B kept here, split 6 of 32 beside a 5,000 MB hold): once
  the hold was released this fix moved it whole to the card (`admitting model to GPU`, 0.74 s for
  32 tokens); v0.3.236 left it split (7.7 s, twice).
- #257 — **under WSL2, a long prompt after a node's first no longer waits seconds on the graphics
  card before it is read.** With the prefix cache on, each long prompt ended by keeping a copy of
  its conversation memory on the card (264 MB for a ~4,800-token prompt on a 7B). Held between
  requests, that copy took room the card's memory pool needed for the NEXT prompt, so the pool
  had to take fresh memory from the driver at that prompt's peak, and under WSL2 a fresh
  allocation slows with Windows uptime (gotcha #755, microsoft/WSL#41701). At 56 h nsys timed it
  at 1.1-6.9 s inside the next prompt's first forward (2026-10-12): later prompts read at 474-640
  tok/s against 1,355-1,580 with the cache off, on v0.3.235 and v0.3.236 alike. A card's
  saved prompts are now kept in host RAM under WSL2, as llama.cpp's server keeps its prompt cache
  (`--cache-ram`). The `prefix_cache::SnapshotHome` choice is one decision, planned before the
  copy and bounded by the machine's memory less a quarter (`kv_budget::host_snapshot_budget`).
  Native Linux keeps them on the card: its pool grows fast at any uptime, so the copy would be
  pure cost. Native Windows is not measured. A/B `SWARMLLM_PREFIX_CACHE_HOME=card|host`.
  Measured on a FRESH boot (1.4-1.8 h, two binaries, rounds card / host / off / host / card,
  `~/swarmllm-257/ab.sh`), where neither arm stalls (first forwards 56-234 ms): the card pool's
  reserved size (`pool_reserved_mb`, `-v`) grew three times after the first prompt with the
  copies on the card (5,184 → 5,472 → 5,824 → 5,856 MB, all four card rounds) and NEVER with them in host RAM
  (5,184 flat through a cache hit, all four host rounds). Those growths are the step that
  stalls at high uptime. The price, at any uptime: 59-169 ms to copy a snapshot off the card
  inside its request (662 and 924 ms in one round, right after a 74-min build had filled the
  page cache), 48-58 ms
  to restore one, so a hit takes ~0.5 s against ~0.3 s on the card. A hit no longer compacts
  the stored copy on the host first (that cost 1.22 s on the first build). Every reply to the
  repeated prompt was identical across all rounds, the cache-off recompute included. Rig
  recipe: an isolated node (`split_rig.sh`'s config), three unique ~4,800-token prompts then
  prompt 1 again with a question appended, rounds alternated under the safety kit; read
  `pool_reserved_mb`, `home=` and `copy_ms=` on `DIAG: prefix-cache inserted snapshot`. Still
  owed: the end-to-end figure at ≥ 48 h of uptime (`ab.sh` again). Leaves #258.

**Closed 2026-10-12, released in v0.3.236 (2026-10-12)** (#244, #248, #249, #253, #254; #251 (1), #251 (2) per model, #251 (4) and #252 (2) of open entries)
- #244 — **a split's reply no longer pauses for seconds while the node reads the graphics
  card's memory.** `nvidia-smi` (2-6 s on WSL with the card busy) was waited for on a thread
  that also runs the node's network task, so a peer's answers sat unread. The 12h release-check
  shape read one request in three at 11.5-16 tok/s instead of 24-27. Same binary, old wait
  restored by `SWARMLLM_NVIDIA_SMI_ON_RUNTIME=1`: the old wait stalled 3 rounds of 3 (13.6, 11.5,
  13.0 tok/s), the fix 0 of 3 (23.4-27.6). The guess-check controller also no longer lets one slow
  check decide how many guesses to make (shipped first, 2026-10-12). Leaves #256.
  ⚠ The stall was first put down to the HuggingFace watcher and the canonical-upload check, which
  run at the same moment. That was a coincidence: sampling every thread found the runtime thread
  waiting on `nvidia-smi`.
- #248 — **a model no longer loses part of its place on the graphics card just because it fills
  the card.** The memory check that narrows a loaded model counted the model's own memory as
  pressure, priced a card/processor split model at its whole file, and could retire a worker
  mid-reply ("stranded_requests=1"). A model that had been running locally then sent part of
  every request to peers. It now acts only when other programs take the card's room, and never on
  a model that is in use (report #035). `split_rig.sh softunload` (Qwen2.5-14B whole here, a peer
  holding its middle, the memory check every 30 s): v0.3.235 narrowed the loaded model to 8 of 16
  parts at "pressure 0.80" 4 s after it loaded, and the next request took 23.2 s; main narrowed
  nothing and answered it in 9.6 s.
- #251 (2) — **a model can now be kept on this computer.** "Keep on this computer" on a model's
  ⚙ panel runs its requests only here: no other computer, in the swarm or a cloud provider, is
  handed the prompt, the work in progress, a picture or the reply. Where this computer cannot run a
  request alone, it is refused with a reason naming the setting and how to fix it, never sent out
  (report #035). Making every path ask one question (`pool::scope::request_scope`) closed three gaps
  private mode already had: a picture went to the nearest computer that could read images
  (`c84c12dd`), a computer outside the pool could be asked for stored work matching a prompt's
  opening (by its hash), and every failed request was announced to the whole swarm with its error
  text. #251 (4): in private mode a model of the swarm is no longer sent to a cloud provider whose
  name it happens to match (`deepseek…`, `mistral…`, `gemma…`) when the pool lacks a part of it;
  the refusal names the missing parts. Guards:
  `every_path_a_request_leaves_by_asks_its_scope`, `every_cloud_path_asks_whether_the_model_is_kept_here`.
- #253 — **a Linux NVIDIA download missing NVIDIA's CUDA libraries now says so at startup and
  runs on the processor**, instead of failing every request with "worker closed connection
  mid-generate". cuBLAS, cuBLASLt, cuRAND and NVRTC are looked up once, the same way the loader
  would (`gpu_support::missing_cuda_libraries`); the log names the missing ones and the
  `apt install` line that brings the card back, and the dashboard and `swarmllm status` give it
  its own reason (`cuda_libraries_missing`), not "card too old". Checked with the libraries hidden from one run
  (an empty file mounted over each, in a private mount namespace): the warning named cuBLAS,
  cuBLASLt and cuRAND, Gemma-2-2B was listed `cuda_libraries_missing` and answered on the
  processor (3.4 s); the control run answered on the card (2.4 s). The install page and
  `INSTALL.txt` list the packages. The `.deb` gets no `Recommends:` — it is the processor build,
  and updates never switch a build's kind (report #035).
- #249 — **an idle model gives its graphics memory back after the idle time (5 minutes by
  default)**, not up to an hour later. Regional demand, which includes this computer's own
  recent requests, no longer keeps an idle model loaded; a model the owner pins or locks still
  stays up to an hour (report #035: a 14B "idle 20m" still holding 9.2 GB).
- #254 — **`--gpu-layers -1` (auto) is accepted on the command line**; it was read as an
  unknown flag `-1` (report #035).

**Closed 2026-10-11, released in v0.3.235 (2026-10-12)** (#224, #137's main half, #228 (1), (3) and (4), #138's processor decode, card and prompt steps and its Q2_K/Q3_K kernels)
- #228 (4) — **Qwen 3.5 writes replies faster on a graphics card**: its decode steps are
  captured as CUDA graphs like every dense family's — refused until now because each DeltaNet
  layer replaced its state with new memory every step. On an RTX 3070 Laptop the 0.8B writes 121-126 tokens a second
  instead of 50-54; the 4B, which the card itself limits, is unchanged (~37). The same reply,
  token for token, with capture on or off.
- #228 (3) — **Qwen 3.5 reads a prompt about 2.7 times faster on the processor and 5 times on the card**: three layers in four are DeltaNet layers, whose
  recurrence ran as ~10 tensor operations per token per layer (a 681-token prompt on the 4B made
  ~160,000 of them); it is one operation per layer now, on the processor and on the card, as
  llama.cpp master runs it. A 685-token prompt on the 4B: 55.8 → 19.8-21.1 s on the processor, 2.6-2.7 → 0.49-0.53 s on the card (`~/swarmllm-138/rows_chunk.sh` § 3). Replies agree with llama.cpp as closely as before (not
  bit-identical: the sums run in another order).
- #138's Q2_K/Q3_K kernels — **a Q2_K mixture-of-experts file's experts multiply several prompt
  rows per pass on the processor**, as Q4_K and Q6_K already did (each weight block unpacked once
  for four rows): on Qwen3-30B-A3B Q2_K the experts' share of a 681-token prompt is 8.0-8.2 s instead of 10.0-10.2 on the processor alone and 4.2-4.4 instead of 4.8-4.9 split 26/22 (`~/swarmllm-138/rows_chunk.sh` § 1). The reply is identical, bit for bit.
- #138's processor prompt step — **a mixture-of-experts model reads a prompt 3.5-4.7x faster
  when its experts are on the processor**: every expert's products for every token in one
  parallel region per projection, each expert's weights read once per pass for all of its
  tokens. Qwen3-30B-A3B Q2_K, a 681-token prompt read as a node reads it: 46 → 13 s split
  between an 8 GB laptop card and the processor, 94 → 20 s on the processor alone; the reply
  identical, bit for bit.
- #138's card step — **a mixture-of-experts model runs its experts on the graphics card as one
  launch per projection**, picked on the card, instead of three products per expert behind a
  copy of the router scores to the processor. Qwen3-30B-A3B Q2_K split 26/22 between an 8 GB
  laptop card and the processor: 74-89 → 40-55 ms per token, level with llama.cpp at the same
  placement. candle's prompt kernel for these experts could give a different reply each run
  (a missing barrier, fixed in a vendored copy of candle-kernels).
- #138's processor decode step — **a mixture-of-experts model decodes its processor layers'
  experts in one parallel region per projection**, as llama.cpp's `mul_mat_id` does, instead
  of each chosen expert's three products as separate parallel calls with tensor plumbing around
  them. Qwen3-30B-A3B Q2_K split 26/22 between the card and the processor: the processor's
  expert time per token 25.1 → 17.8 ms on average (ABBA in one binary), the reply identical.
  The card's experts and a prompt's followed the same day (above).
- #228 (1) — **Qwen 3.5's mixture-of-experts models run** (`qwen35moe`: Qwen3.5-35B-A3B and its
  larger siblings were refused outright). Checked against llama.cpp master on a real file, whole,
  split and on the card; the numbers are in #228's entry.
- #137's main half — **every node at the default contribution decoded outside its own thread
  pool.** `cpu_pools::pool_for` returned "use the global pool" for a decode width equal to the
  global pool's, and at the default that is every machine (each level is at or below the
  physical cores), so the worker ran each decode forward as `f()` on its tokio thread. rayon-core
  answers a parallel call from outside every pool by injecting a job and blocking the caller
  (`Registry::in_worker_cold`): a cross-thread handoff per op, hundreds a token. Now every decode
  width is a pool and the forward is installed on it once. Same binary, 4 threads, the global
  pool 4 (a node) vs 10 (installed), mean ms per token, two ABBA passes after 60 s cool-downs
  (`~/swarmllm-137/pool.sh`): llama-3.2-3b processor-only 110.9/112.2 → 83.1/80.5 (**1.36x**);
  Qwen2.5-14B with 18 of 48 layers on the processor 180.6/189.5 → 159.8/160.0 (**1.16x**); the
  3B whole on the card (null — no processor layers) 3.7/4.2 vs 4.2/4.0. No benchmark could see
  it: they run under the default global pool (every logical core), so decode was always
  installed there. Found by a cooled node measuring 5.3-5.4 tok/s on the 14B where `test-split`
  measured 6.3-6.6. **Through a real node** (one binary, `SWARMLLM_DECODE_INSTALL=0` the other
  arm, ABBA, `~/swarmllm-137/node_ab.sh`; the worker's `swarm-decode4` pool present in every fix
  arm, absent in every other): the 14B split 5.64/5.93 → 6.79/6.93 tok/s — llama.cpp's own
  6.8-7.0 at that placement; the 3B on a processor-only node 10.97/13.04 → 14.53/15.29 (the
  13.04 run's calibration had fled to 3 threads, an installed pool, which is how the defect
  half-hid on some machines). Guard: `decode_at_the_global_pools_width_still_runs_inside_a_pool`
  (red with the fix switched off).
- #224 — a node that cannot reach HuggingFace sometimes never took the checked holders' verdict
  (gate step 12m's proxy arm failed at the .228, .231, .232 and .234 gates, on whichever binary was
  B, and passed every re-run). Two independent causes; the old hypothesis (a race on
  `shards_pending_verification`) was wrong. (1) **A check that passed against a hash the manifest no
  longer named ended the dispute the newer hash had recorded.** B's startup sweep reads every
  manifest once, then hashes; gossip replaced shard 1's hash 2.7 s after start, the re-check recorded
  the dispute (21:33:04.426), and the sweep — still hashing against the hash its own zeroed bytes
  match — passed and cleared it 310 ms later. B then announced the SWARM's build for those bytes and
  `outvoted_by_checked_holders` found its own tag among the checked holders'. The one failure of five
  debug re-runs was the only one with the dispute before the sweep's end; the .228 and .231 failures
  carry the same signature (14 ms, 339 ms). Fix: a pass reaches the dispute through
  `SharedState::note_shard_verified(shard, hash_checked_against)`, a failure through
  `note_shard_disputed`, and both through `ModelRegistry::note_bytes_checked`, which compares the
  bytes' hash with the hash the manifest names NOW under the manifest's map guard (HDFS's
  generation-stamp rule); `verify_pending_shards` keeps a re-check queued while its part downloads
  instead of dropping it. (2) **A holder's return to a manifest version it had broadcast before was
  suppressed as already heard** for the 30-minute quiet window, so a newcomer caught up while the
  holder's manifest was off that version never learned the agreed hash ("no manifest of the same
  layout has named their hash yet", every pass). `manifest_needs_broadcast` now counts a heard record
  on the "changed" arm only when it postdates the version we are leaving (RFC 6206 §4.2: our own
  change starts a new interval). Measured with `examples/outvoted_rig.sh` (`~/swarmllm-224/`, B a dev
  build, live node stopped): `SWEEP_HOLD=150` (`SWARMLLM_FAULT_SWEEP_HOLD_SECS`, holding the sweep's
  window open so the agreed hash lands inside it every time) — B with a pass that clears outright
  FAIL in 600 s with "shards kept despite disagreeing (0)" on its zeroed part, B with the fix PASS in
  254 s, both in the failing order; `ORDER=late` (A and C healed before B joins) — A and C on .234
  stalled past 800 s with B never sent the agreed hash, A and C on the fix PASS in 371 s (C's return
  broadcast at 02:01:30, B had `b299,fcef` 2 s later); the gate's plain arm on the fix PASS in 245 s.
  Tests (each red with its change undone): `a_pass_against_a_hash_the_manifest_has_since_replaced_keeps_the_dispute`,
  `returning_to_a_version_heard_before_we_left_it_is_announced`; guard
  `a_passing_check_of_held_bytes_is_judged_against_the_hash_named_now`. `docs/invariants/network.md`
  § "A node vouches only for bytes…" door 6 and § "The interval starts at our own change".

**Closed 2026-10-09 — released in v0.3.234 (2026-10-11; v0.3.233 was built and never released)** (#69, #247, #162)
- #162 — the release gate: (a) a cloned gate lost its helpers — every gate since .227 refuses to start
  when a `$G/` helper it names is missing (exit 4), and each clone's retarget script asserts every
  substitution. (b) step 12e could not test a takeover deterministically here. Two causes, measured
  (`~/swarmllm-162/run{,2,3,4}.out`): a 7B across four nodes on this 16 GB box gives C and D only the
  machine's free memory as standby room (28 layers on one request, 20-23 on the next once A and B had
  loaded) — an honest figure no setting should hide (`split_rig.sh` can now put a standby in a memory
  scope of its own, `MEM_C/D/E`, which the node reads since #247 — it cannot create memory); and, since
  #10 (`5e53baa3`), `failover`'s takeover arm sent the healthy arm's prompt again, RESUMED it from what B
  had stored, and a failed resumed pass is read again from 0 through the same plan by design — B's
  respawned worker answered and no stand-in was asked. (That second cause was first misread as the
  kill's timing in `d412ddbe`.) `split_rig.sh failover` now keeps no prompt on the coordinator, and the
  gate's 12e failover arms run Llama-3.2-3B with the 1B drafter: constant room (28), `failover` and
  `failover_mid` each took over 2/2 with replies byte-identical to the A→C→D control.
- #247 — a node in a memory-limited container planned against the HOST's memory (opened and closed
  2026-10-09, found while making #162 b's rig deterministic). Five places read machine memory through
  `sysinfo`, which is `/proc/meminfo` — the host's figures inside a container too — so a node under
  `docker run --memory=4g` or a Kubernetes limit planned, advertised and admitted against the whole
  host and left the fit to the OOM killer. `daemon::machine_memory` is now the one reader: the host's
  figures under the tightest limit on this process's cgroup or an ancestor, available = limit less the
  WORKING SET (cAdvisor's figure; `sysinfo`'s cgroup "free" counts page cache). Guard
  `machine_memory_is_read_in_one_place`. Measured: under the cargo shim's 10 GiB slice a node reads
  10,240 MB total, the host 15,991. `docs/invariants/memory.md` § "Machine memory is read in one place".
- #69 — panels read a failed load as "there is nothing here". The `load*` helpers in `data.js`
  now return the cache, so a failed load answers with the last good value (TanStack Query's
  refetch-error result) and `loadReachedDaemon` says it failed; the cache's one writer is
  `remember`, and a WebSocket value goes through `acceptPushed`. Fixed with it: Compare's "No
  models available yet" to a node it could not ask, Settings' provider badges left at "Not set",
  the header panel losing its cloud providers (and reading "Offline") or saying "Looking for other
  computers" without having reached the node (new state "Not answering", one key in 21 locales),
  a model picker emptied of cloud models when only that list failed, `models_changed` emptying the
  cloud list before every reload, the peer list worded by a stale REST load over a WebSocket list
  and blanked into "share your address" when the WebSocket closed, and the pool owner shown as a
  member (`/api/pool/state` carries `node_id` and `is_owner`). The guard now scans every frontend
  file by statement, with a planted-violation self-test. `docs/invariants/frontend.md` § "A failed
  load answers with the last good value".

**Closed 2026-10-09, released in v0.3.232** (#194, #152, #245)
- #245 — GitHub moves `ubuntu-latest` to Ubuntu 26 from 2026-10-19 (notice on CI run 37914655701),
  under the CUDA repository (`ubuntu2204`'s `cuda-nvcc-12-8`) and the system packages CI installs.
  CI's Linux jobs now run on a pinned `ubuntu-24.04`, the image every required check passes on
  today. The matrix jobs name themselves by `matrix.os` ("Test (ubuntu-latest)") and branch
  protection requires those 14 names, so `matrix.os` keeps naming them and `runs-on` maps the
  runner (`matrix.os == 'ubuntu-latest' && 'ubuntu-24.04' || matrix.os`); renaming a job would
  leave a required check that never reports. The release builds were already on `ubuntu-22.04`
  (glibc). Checked with the CI's own pinned actionlint before pushing.
- #152 — the continuous guess-check stream (on by default since v0.3.216) ran only where the
  peer held the model's LAST layers: 4 of 78 node-model holdings on 2026-10-01. A node holding
  both ends of its model runs the boomerang by default ("Start and finish on this computer"),
  and those requests — 58 of the 78 — fell back to rounds. Now `dsd_stream::stream_shape` takes
  any plan with ONE peer segment: last, middle or first. Where this node holds the last layers
  the peer is sent hidden states with no guess, history or sampler, and this node's last
  segment walks each chunk as its answer is TAKEN, in order (gotcha #180); the peer's side
  needed nothing new. A turn a restart skips is no longer logged on the serving node as a
  failure (38 warnings in one rig arm). Rig, Qwen2.5-Coder-7B + 0.5B drafter, this node's ends on
  the card and the middle on the processor, one binary: +12% prose / +32% code at a 270 ms
  round trip, level at 24 ms (the stream overlaps the round trip; a slow middle leaves little
  at a short one); replies scored against llama.cpp like the rounds'. Not measured over a real
  link: the Italy peer of 10-01 was online for minutes on 10-08 only. `split_speculation.md`
  § 4b, `docs/invariants/network.md` § "A stream of verifies runs in its order".
- #194 — an agent-sized prompt filled an 8 GB card with conversation memory: a card kept the KV
  cache as f32 plus an f16 flash mirror, 6 bytes an element against llama.cpp's 2, and a node with
  `inference.max_seq_len_override` set (the README's advice to agent users) kept NO card KV budget
  at all — a condition left from the 07-29 load-time shrink — so a long prompt overflowed the card.
  Now a card keeps the half cache (f16 BHSD, read in place by flash and the decode kernel) wherever
  both kernels cover the model, through ONE rule asked by the cache and every estimate of it
  (`layers::kv_storage`; peers by `features::KV_HALF_ON_CARD`), and the budget is kept whatever the
  context setting. Qwen2.5-Coder-7B on an 8 GB card at a 32768 context: room for 38,164 tokens
  (was 12,721); a 15,648-token prompt answered in 51.5 s where v0.3.231 took 307 s with the card
  full; decode and the prompt pass level; replies scored against llama.cpp equal (two byte-identical).
  The research that removed the f32 copy's justification: arXiv 2604.15409 compares cache-on with
  cache-off decoding, both in f16, and llama.cpp stores f16 by default. Not built (no user waiting
  on them): Q8_0 KV (another ~2x, needs the dequant fused into the decode kernel) and a hybrid
  placement that keeps the cache on the processor. `docs/invariants/inference.md` § "A card keeps
  its KV cache as f16"; `docs/invariants/memory.md` § "A card's KV budget is kept whatever the
  context setting".

**Closed 2026-10-08, released in v0.3.231 (2026-10-08)** (#233, #234, #235, #236, #238, #239, #240, #242, #243)
- #243 — on a narrow screen the chat's model picker kept naming the last model picked after a switch
  to an older chat, while the chat's own model answered (report #007, a tester at 590 px).
  `selectDropdown` set the desktop picker and not `#mobile-model-select`; only the list refresh did.
  It sets both now (guard `every_model_selection_sets_both_pickers`). Same report: the picker named
  models from the file's own `name` (four quantisations shared one, a Q8_0 file called itself AWQ)
  while the badge used the id — now the id everywhere, with the quantisation written `Q4_K_M`;
  the word count cleared on send; the header's message count kept current; the touch-screen
  "Settings"/"Off" text came from CSS literals (now the translated `aria-label`; the redundant
  "Off" removed); toasts no longer cover Send below 768 px. Checked in Chrome at 590 px against the
  v0.3.230 frontend as control. `docs/invariants/frontend.md` § "One choice is shown by every
  control that shows it".
- #242 — choosing "Unlimited" for model sharing stored a 1 Mbps cap (report #006, a tester). The
  settings API floored `max_bandwidth_mbps` at 1 since April (`4dc09cab`), but 0 means AUTOMATIC
  (10 / 50 Mbps / none by contribution level, since `8853d345`) — and before v0.3.180 every
  Settings save sent the slider, so saving ANY setting stored it. The slider steps by 10, so 1 was
  never the panel's choice, and it went on showing "Unlimited" (it snaps 1 to 0). The same floor
  hit `auto_manage_max_storage_mb` (0 = a share of the disk) and `batch_timeout_ms` (0 = at once).
  Fixed: `api::admin::apply_numeric_limits` keeps a meaningful 0 and enforces only ceilings; a 1 on
  disk reads back as automatic through `config::parse_config_file`, now the ONE reader of
  config.toml — a dashboard save, a reload and the settings read each parsed it raw, so only the
  loader had ever applied `migrate_superseded_defaults` (a save put a stranded value back live);
  guard `the_config_file_is_read_through_one_parser`. The slider says Automatic and its hint gives
  the three rates (21 locales). Repro on a throwaway node: .230 PUT 0 → GET 1; fix → 0, a stored 1
  → 0 at start, the next save drops it from the file, one log line.
- #233 — guess-and-check across a split on one card had read slower since v0.3.226 (bisected: .224
  fast, .226 slow, one boot). The drafter's decode graph changed shape at every switch between
  host-fed (i64) and card-fed (u32) token ids — the embedding's cast was inside the capture — and
  .226's graph REST, tuned on a 7B whose rebuild costs 65-108 ms, then ran the 0.5B drafter without
  its graph (~6 ms rebuild vs ~9 ms saved a step): drafts 6 → 15 ms, the controller stopped
  guessing. Ids are made u32 before the capture, and a rest is weighed on the card's own timeline
  (two CUDA events a step, only for a shape that rebuilds). One-binary rig: old 19.5-21.9 tok/s
  stream / 24.7-29.2 rounds → 25.3-28.4 / 23.9-34.1, as .224. `docs/invariants/inference.md` § "A
  rest is weighed on the card's timeline, and a step's ids are one type".
- #236 — a streamed reply whose machine refused, restarted or went silent mid-reply ended with an
  error after part of it was shown (report #005). It is now CONTINUED on a fresh route from exactly
  what the reader received (Petals' history replay, vLLM's recompute): `StreamingTokenTx` records
  delivered text, `router::continuation_after` decides, `InferenceRequest::continuation` is appended
  in the one prompt builder, at most twice, with the budget less what was sent. Rig `split_rig.sh
  continue`: v0.3.230 ended the stream with an error after the serving worker was killed; main
  continued it on the other peer 25 ms later, 300 chunks, no repeat, the join seamless.
  `docs/invariants/scheduling.md` § "A reply whose machine failed mid-stream is continued".
- #238 — a node serving a segment learned that the reply had ended only from its own timers (#235
  bounded that). Each attempt's executor now records the peers it ran segments on, stand-ins
  included (`SharedState::note_request_peers`), and the request's end tells each of them once with
  the existing `CancelInference` (`router::release_request_on_peers`, from `finalize_request`); the
  worker drops a cancelled request's cache between messages (`release_caches_of_cancelled`). Only
  at the REQUEST's end, never an attempt's: a worker skips the next forward of a cancelled id, and a
  retry reuses the id (gotcha #749 — the research that placed it). Older peers forget the
  conversation and keep the cache to their timers. Tests red with each half off.
- #240 — a worker split across the card and the processor was charged nothing against the card
  budget, so its card share read as another program's memory: a configured `max_gpu_vram_mb` was
  overrun by that share (a 14B split took 5,782 MB against a 5,000 MB cap, found by the #234 card
  rig) and an idle split worker was never a reclaim candidate. The spawn now charges the first `n`
  layers by the admission estimator, within the room the split was sized for. Card check: the next
  admission read `committed_mb=0` on v0.3.230 and `committed_mb=3691` (of a 3,851 MB measured
  share) on the fix. `docs/invariants/memory.md` § "Graphics memory has ONE owner".
- #239 — a node behind a router that refused its UPnP mapping said nothing, asked again without
  pause for its whole life, and a node that stopped kept its ports on the router for up to an hour
  (a tester's two nodes behind one router, 2026-10-08). libp2p-upnp is now the direct 0.6
  dependency (backoff, five tries; libp2p PR 6128; guard `upnp_is_the_release_that_backs_off`),
  `upnp_watch` explains a silent router once after three minutes, and a clean shutdown hands the
  mappings back. Book + `docs/NETWORKING.md`: run a second node on another port.
  `docs/invariants/network.md` § "UPnP that the router refuses is said out loud".
- #234 — a node holding PART of a model was planned ranges its worker could not grow into, and the
  re-plan repeated them (a tester's report, v0.3.229: an RTX 3060 holding 28 of a 30B's 34 parts,
  refused layers 19..27 and 43..48 — "another holder will have to take that part" — five requests
  in a row, the other holder 0.31 ms away). Two defects. (1) The planner's room for this node on a
  card was the card/processor split's width (#129) for every plan, but only a SPAWN splits, and
  only its one range; a further range grows that worker on the card alone. Now
  `max_hostable_layers` is the card's room and `fresh_run_layers` the split's, for ONE local run
  (`parallax::local_fits`, `local_can_hold_every_layer`), offered only while a spawn can be had.
  (2) The re-plan's record said only "cannot run the whole model"; the pool now records the
  layers a load refused to ADD (`note_load_refusal`) and the re-plan gives this node fewer, with
  no split. Tests red with each half off. `docs/invariants/scheduling.md` § "The component that
  will refuse…" and § "A re-plan is warranted by a changed fact".
- #235 — a live reply was refused for conversations that had ended (report #005: 330 MB of a
  469 MB budget, 71 s into a reply, on a peer serving a 30B). A worker never ran its KV store's
  TTL sweep, and a node serving a segment for another computer is never told the reply ended, so
  those caches stayed for the worker's life. The worker sweeps on its 30 s tick (guard
  `a_workers_kv_store_is_swept_by_the_worker_that_builds_it`), and caches silent for
  `CONVERSATION_GAP_SECS` give way to a live prompt or a reply's growth before anything is
  refused (`KvCacheStore::release_finished_conversations`). What it leaves: #236, #237, #238.
  `docs/invariants/memory.md` § "A conversation that is over gives its room to one that is not".

**Closed 2026-10-07, released in v0.3.230 (2026-10-07)** (#229, #230, #231)
- #231 — a model its holders cannot run was never fetched by a machine that could (the report
  behind #230: the 9B's one holder capped below it, three graphics cards with room holding none of
  it). Replicas counted copies, not whether a holder can run the model. `auto_manage::coverage`:
  the layers a model's live holders could ever carry between them (`scheduler::layers_carried`,
  each holder at the smaller of what it holds and its ceiling); when short, for a model somebody
  asked for and the connected swarm could carry, ONE carrier at a time — highest rendezvous weight
  among machines with room for a part and the disk, judged from the gossiped figures, its own
  included — fetches the parts it lacks in model order until the SHORTFALL is closed, within its
  ceiling (never the whole model because it has room: "no machine holds the whole model"), past the
  replica target and the hash ring, still through the trust gate, the budget and `would_shed_copy`;
  and `would_shed_copy` keeps every copy
  the model would fall short without (`copy_carries_model`), so download and prune agree. Only
  machines in scope are candidates (the pool's, in private mode), and a carrier keeps the role only
  by progress: 20 min with no part gained or being fetched and every node passes it over for 2 h
  (`DIAG: the machine chosen to carry this model has made no progress`). Unit tests
  `coverage::tests::*` (the carrier test red with carrying off and again with the shortfall
  ignored, the prune test red with the protection off, the lease and scope tests red without
  theirs). Rig `examples/carry_test.sh` (three nodes, private network namespace): the first
  request refused in 0.0 s ("room for about 19 of its 22 layers"), the carrier named carrying 30 s
  after the client's demand reached it (score 1500 against routine 30) and fetched the part, the
  next request served in 1.2 s; with routine replication already satisfied it had fetched nothing
  for ten minutes. (The rig's first run left `min_replicas` at 2 and the part was fetched by
  ROUTINE replication before any request — caught by the mechanism check, not the outcome.)
- #230 — a peer was handed a model it could never hold (a tester's report, v0.3.229): a whole
  Qwen 3.5 9B (6688 MB at its admission) to a 6 GB peer capped at 5200 MB, refused 8.8 s later from
  another continent; 46 of a 30B's 48 layers to a 16 GB machine, refused after 87 s. The last
  capacity rung held peers to NOTHING (report #025's rescue), and the current-figure arithmetic
  (file bytes per layer) let both through. Each node now advertises
  `NodeCapability::model_memory_ceiling_mb` (its RAM cap; card + memory on a card node); the
  planner weighs it with the peer's own admission arithmetic (`segment_cost_curve`; a model's
  weights now come from its header on every node), never plans past it on any rung or greedy pass
  (clamped where candidates are built, so delegation and the standbys inherit it), splits across holders where their ceilings add up, and
  refuses as `SwarmShortOfMemory` before asking anyone where they do not. Rig
  `examples/ceiling_test.sh`: 784 MB cap against a 980 MB footprint → 503 in 0.1 s, server asked
  0 times (v0.3.229 client: 3); 1044 MB → served. Also fixed: the cost curve put the whole KV cache
  in its fixed term since #228 (2b). What it leaves: #231. `docs/invariants/scheduling.md` §
  "Never past what a peer could EVER hold".
- #229 — a request forwarded to a pool member over HTTP was cut off mid-reply at its first-token
  budget: reqwest 0.12's REQUEST timeout runs until the body has finished. A streamed forward is
  now bounded by inactivity (`peer_forward::PEER_STREAM_IDLE_SECS`, four of the peer's 15 s
  keep-alives, on a second client — `read_timeout` is per client and also covers the wait for
  headers, which a streaming peer sends before its first token); a non-streamed one, silent until
  complete, by the first-token budget plus its `max_tokens` at 0.5 tok/s, capped at an hour
  (`forward_deadline`). The client-wide 600 s total is gone. Tests `forward_deadline_tests`.

**Closed 2026-10-06, released in v0.3.229 (2026-10-07)** (#117, #225, #226)
- #117 — Qwen 3.5 (dense) was refused; it runs now. The local branch's rewrite against llama.cpp
  master rebased onto main and was checked again: logits (`~/llama.cpp-ref/dump_logits`) — 0.8B F32
  worst cos 0.999999 and top-1 24/24, Q8_0 median 0.99965 24/24 whole and split at layer 10
  (identical), 4B Q4_K_M median 0.99943 20/24 with the four misses near-ties (llama.cpp's own gap
  0.008-0.061); `test-split` against llama.cpp's tokenizer and greedy reply
  (`~/llama.cpp-ref/ref_generate`) on the processor and the card — prompt tokens identical, replies
  identical up to near-ties (gaps 0.009-0.10); an isolated daemon fetched the 0.8B from HuggingFace
  and answered, a second turn and streaming included. That check found the template not parsing
  (gotcha #798, fixed beside it). Speculation of every kind is refused for recurrent state
  (`speculation_can_roll_back`, `SplitModel::carries_recurrent_state`), truncation refuses while it is
  held, the prefix cache never snapshots it. What is left: #228.
- #225 — a graphics card advertised about half the speed it decodes at. The 0.35 in
  `vram::estimate_tokens_per_sec_7b` was calibrated on 2026-09-01 (35.32 tok/s, RTX 3070 Laptop,
  `prefill_bench`, Coder-7B Q4_K_M, 896 prompt, ~912 KV); the CUDA-graph decode work took the same
  card at the same shape to **59.53 tok/s** (re-measured 2026-10-06 on a `--features cuda` release
  build, live node stopped, safety kit, two runs of three with identical best — 0.585 of the
  roofline; Windows up 5 d 14 h). Now **0.55**, under the measurement as the processor's 0.75 is.
  A tester's RTX 3060 advertised 28.64 against ~60 measured over HTTP (not calibration data).
- #226 — every contribution counter restarted with the daemon, so an operator could not tell
  whether their node had ever served anyone (a tester's node: `99.3 ms per layer served` →
  `(no segments served yet)` after an update). `daemon::state::lifetime`: the serving totals as
  this run found them in redb (`lifetime/served`) plus this run's counters, written as one
  absolute figure on the health monitor's 30 s tick when changed and at shutdown — a crash loses
  at most a tick, nothing counts twice. Shown in `swarmllm diagnostics` (`since <date>: …`), in
  `/api/admin/performance` (`served_lifetime`) and on the dashboard's "Served for the swarm" panel
  (two keys, 21 locales). A description of work, not a balance — credits stay dormant. Tests
  `lifetime::tests::*`. Traffic totals (bytes sent/received) are still per run.

**Closed 2026-10-06, released in v0.3.228 (2026-10-06)** (#222, #223) (`docs/invariants/memory.md` § "`AutoShardManager::would_shed_copy` is the ONE answer")
- #222 — a node near its storage limit deleted parts and fetched the same parts back, for ever.
  **A second way in, found on v0.3.228 the same day and fixed in v0.3.229:** a part
  once refused from a peer stayed in `shard_p2p_failed` after it was fetched from HuggingFace, and
  the pending-fetch pass re-fetched it whenever prune deleted it as surplus — a tester's node
  (`bf7b3263`), ~4 GB an hour at disk pressure 0.56 (gotcha #797; the set lives in memory, so a
  restart stops it until the release).
  **A third way in, fixed in v0.3.230:** a pending fetch the ORIGIN cannot serve (a model with no
  HuggingFace source, or a node in offline mode) blocked that part for the node's life — the
  download pass read the entry as "peers exhausted" and never asked them again. It now goes back
  to the peers (`pending_fetch_can_proceed` → `can_fetch_shard_from_origin`, which the pass had
  re-derived as `hf_sources.contains_key` and so missed offline mode).
  Found in the post-v0.3.227 swarm check from peers' announcements in our own log: a tester's 30 GB
  node (`e561df35`) sawed 62 → 59 → 62 parts every ~25-30 min from 02:00 UTC (one Qwen3-30B-A3B
  part dropped per 5-min prune cooldown, the same indices back once their 30-min protection
  lapsed, ~1.6 GB a cycle); `9594e1ff` had done ~100 parts a day across 7 models for days. Three
  causes, all fixed: the download pass fetched below the RAW replica target while prune shed above
  the pressure-adjusted one (now both ask `AutoShardManager::would_shed_copy`, the download side
  at the disk pressure after the fetch); prune judged files by graphics-memory pressure too (now
  disk only); and the free-disk clamp (held + 80% of free) filled a disk-limited node to 99.5%,
  into the tester's own fill safeguard (now the filesystem's last 10% is never taken — kubelet's
  `nodefs.available<10%`). Guards `fetch_what_prune_keeps::*` and
  `downloading_until_the_budget_says_stop_leaves_a_tenth_of_the_disk_free`, each red with its fix
  switched off. Field check after the release: `e561df35`'s `Peer retracted shards` count in our
  log (gotcha #795, `docs/DIAGNOSTICS.md`).
- #223 — a node with no region forgot its own demand every ten minutes. The decay filed it in
  `region_demand` under `"??"`; the replica target, prune's demand penalty and the idle-VRAM unload
  read it under `""`. So such a node's target fell back to the raw counter the decay had just
  zeroed, and its idle unload never saw a model was wanted. One key now
  (`AutoShardManager::demand_region`); guard
  `a_node_with_no_region_reads_back_the_demand_it_filed`. Found building the #222 rig, whose
  nodes have no region.

**Closed 2026-10-05, released in v0.3.226 (2026-10-05)** (#221) (`docs/invariants/inference.md` § "A graph pays only when it is updated")
- #221 — a speculative check's CUDA graph could never be updated in place: candle-flash-attn
  zero-filled its `softmax_lse` scratch with `alloc_zeros` inside the captured step, and the
  driver refuses to update ANY graph whose memset targets memory the graph allocates — 39/39
  refused even at a constant size, 39/39 updated when the memset's target was allocated outside
  (probe `graph_memset.cu`, 2026-10-05). So every several-position check on a card (split
  speculation, n-gram over a split) was rebuilt at 10-100 ms instead of updated — the .225 gate's
  split rig rebuilt 105 of 135 launches. With `num_splits = 1` the forward kernel only writes
  `softmax_lse` and nothing reads it, so it is now allocated uninitialized (both paths in
  `vendor/candle-flash-attn/src/lib.rs`; upstream flash-attn uses `torch::empty`). Verified on a
  `--features cuda` build (`~/swarmllm-gpu-1005/verify221b.sh`, n-gram split rig, one CUDA
  process): before, `positions=5 … CU_GRAPH_EXEC_UPDATE_ERROR at a node of type
  CU_GRAPH_NODE_TYPE_MEMSET` then rested; after, 225 of 227 launches updated in place, 0 rebuilt;
  replies byte-identical to v0.3.225 and to the build before, 3/3.

**Closed 2026-10-04, released in v0.3.225 (2026-10-05)** (#150, #153, #214, #218, #219, #165, #189) (`docs/invariants/scheduling.md` §
"A holder's refusal is reported as what it was")
- #153 — on Windows a model worker that outlived its daemon held the node's QUIC port, so the
  next start failed. `Command` passes every inheritable handle, and the QUIC socket is one (why,
  despite `WSA_FLAG_NO_HANDLE_INHERIT`, is still unexplained — moot for workers now; other
  children are short-lived). Reproduced on Windows first (GNU cross-build of `fc75e036`,
  `C:\temp\swarm153\worker153load.ps1`): daemon killed the moment its worker started loading
  Llama-3.2-3B → 300 ms later the worker alive and UDP 8950 still held under the dead daemon's
  pid → the restart died on "Port 8950 is already in use". (Killed mid-GENERATION it does not
  reproduce: a worker writing tokens reads end-of-file and exits in ~50 ms — only a blind phase
  such as a load leaves the window.) Fixed with a Windows job object instead of the planned
  `WorkerChild`: `process_pool::end_with_this_daemon` puts every worker in one job with
  `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`, its handle created non-inheritable, so Windows ends the
  workers when the daemon exits, however it exits. Same harness on the fix: worker gone at
  300 ms, port free, the restart ready in 2 s; an ordinary request is served normally and a
  forced stop leaves no process behind.
- #150 — the daemon's worker reader decoded a message's header before reading its payload, so
  one message from a NEWER worker (an update the daemon has not restarted into spawns workers
  from the new binary, gotcha #188) misaligned the stream and evicted the worker on every load
  (gotcha #765). `worker_ipc::recv_worker` now reads the whole frame first; a header whose tag no
  `WorkerMsgTag` names (decided by deserialising the tag, never by serde's error text — #295;
  the mirror is kept complete by an exhaustive match in `every_worker_message_has_a_tag`) comes
  back as `UnknownWorkerMsg`, which `reader_actor` skips with one warning per kind; a KNOWN tag
  that fails to decode stays fatal. Test red without it: an unknown frame with a payload, then a
  known one, read correctly. ⚠ Daemons up to v0.3.224 still evict, so a new worker→daemon
  message is still gated at the sender (`DAEMON_READS_CARD_PROBE`'s pattern) while any runs.
- #214 — Qwen2.5-Coder-7B (Qwen's official upload) answered a tool request as the text
  `<{{"name": "get_time", "arguments": {"zone": "UTC"}}}}`, no `tool_calls`. The model's habit,
  not our prompt: llama.cpp (llama-cpp-python 0.3.16) gives the identical reply on the
  identical prompt rendered from the GGUF's own template, which shows its example call with
  the braces doubled. `tool_parse::try_template_doubled_braces` reads that one shape when it
  is the WHOLE reply (an optional `<` / `<tool_call>`, the doubled brace, one call object, only
  closing braces after), and `retract_over_angle_opener` keeps the `<` out of the content.
  Live, dev build: OpenAI plain + streamed and Anthropic plain + streamed all return the
  call; v0.3.224 returned the text on all four. (The Anthropic stream's empty text block
  before a `tool_use` is the preamble's, by design, and predates this — xLAM shows it too.)
- #189 — `/v1/models` reported `max_model_len: null` for every model this node holds none of
  (no header to read), so OpenClaw assumed 128k and its agent turns were refused. The model's
  declared context now travels in its manifest: `ModelManifest::context_length`,
  `#[serde(default)]`, outside `manifest_hash` (like `mmproj`) so old and new nodes agree on
  every hash; set by the builders from the header (`ManifestFromGguf::context_length` is
  required, `model::manifest::declared_context`); kept unknown → known by the registry merge;
  filled into manifests written before the field by `SharedState::fill_declared_contexts` before
  every manifest gossip; read by `max_model_len_for` when there is no header
  (`context_from_manifest`). Rig (two dev nodes, A holding TinyLlama with an OLD-format manifest,
  B holding nothing): B reported 2048 within 10 s; null control, v0.3.224 on both: B `null` for
  90 s. Peers on older builds send none, so a network-only model reads `null` until a holder
  of it runs this release.
- #165 — the notice saying what prompt privacy costs never fired where the priced search
  decides (parallax on, more than one candidate — the default): `report_privacy_cost` sat on
  the gate's path only, so GLM-4-9B ran a five-segment boomerang at 95 s for 48 tokens with
  nothing said. The plan actually taken is now priced at the planner's exit
  (`privacy_cost_of_plan`: a plan that starts and ends here with someone else in the middle,
  against the search's own cheapest route with privacy off, bounded rungs only, both through
  `parallax::chain_cost_ms`), and said through the one `announce_privacy_cost` — same bars
  (≥ 5 s and ≥ the route without it), same once-per-model-per-10-min limit, same translated
  wording; previews never speak. Still reported, never acted on (RFC 7507's reason). A `DIAG:
  what keeping the first and last layers here adds to the route taken` line gives both
  figures per assembly. Live, dev build: Llama-3.1-8B through `4a3ac72e` priced
  `privacy_extra_ms=0`, `without_privacy_ms=78060` — privacy was not the slow part there (the
  far peer was), and nothing was said. Test `the_search_s_boomerang_says_what_privacy_costs`
  is red without the call.
- #219 — a processor-only peer offered room it then refused: `9594e1ff` offered every
  TinyLlama layer and refused a 43-token prompt, "0 MB available for conversations". The offer
  (`live_headroom_mb`, 70 % of available memory) and the worker's admission (available less
  `kv_budget::device_free_margin_bytes`, 5 % of the machine, at least 256 MB) were two rules;
  below about 3.3 margins of free memory the first exceeded the second. The offer is now bounded
  by the worker's reserve (rounded UP — truncating it over-offered 1 MB on an 8 GB machine, which
  the new property test caught). Petals' rule: announce from the cache that admits
  (`cache_tokens_left = memory_cache.bytes_left // bytes_per_token`). Test
  `a_node_never_offers_room_its_own_worker_would_refuse` sweeps 4-64 GB machines against the
  worker's own reconciliation; red without the bound. **Residual — field check after the next
  release:** it changes what a PEER advertises, so it acts once `9594e1ff` runs it; read
  `pipeline candidate node=9594e1ff … max_hostable_layers` for TinyLlama when it is short of
  memory (0, not 22). This node has a card and advertises no RAM figure, so it could not be
  observed here.
- #218 — a request every holder turned down was reported as "No reachable node holds layers X-Y
  of … the peer that held that piece has gone", or as "Insufficient network capacity": the
  re-plan, with the refusing holders barred, described its own search, not the refusal. The
  router now keeps a peer's refusal as it keeps its own memory shortfall
  (`router::a_refusal_the_caller_should_hear`), and when the re-plan finds nothing reports what
  happened: `SwarmShortOfMemory` when the refused plan had gone past the memory the holders
  OFFER — the scheduler records that per request (`plan_exceeds_offered_memory`,
  `layers_offered`, `SharedState::planned_past_offered_memory`) and retrying cannot help — or
  `HoldersDeclined`, in the last holder's own words, when the plan fitted their offers (usually
  busy: retry later). Two hint keys in 21 locales. Verified live with the fixed build: Qwen3-30B-A3B
  (18.7 GB) → "Not enough memory in the swarm … the computers holding it have room for about 43
  of its 48 layers between them" (offers 19 + 11 + 8 + 5); TinyLlama through `9594e1ff` alone →
  "… online but turned it down — the last one said: Not enough free memory on this node for a
  43-token prompt (…)". The distinction came from the user's question: a refusal at a holder's
  BUDGET can never succeed, a busy holder's can — opposite advice, so two variants.

**Closed 2026-10-03, after v0.3.223** (`docs/invariants/network.md` § "One upload per model id,
on every node", paragraph "A node with no origin to ask is healed by the holders that checked theirs")
- #160 — a node that could not reach HuggingFace never judged or replaced its copy: `9594e1ff`
  replaced no part on .221, .222 or .223 while every other peer's heal acted within the hour
  (its owner could not be asked; every branch that fits is closed). A node whose heal checked
  its copy against the upload now says so (`ShardAnnounce::origin_checked_models`, additive),
  and a node with no origin to ask — offline mode, which used to skip the heal outright, or no
  route to HuggingFace, including one that verified an upload once and lost its route — deletes
  a part whose bytes differ from what two connected checked holders agree on (none holding
  ours) and fetches it from them against the hash their tags name
  (`auto_manage::canonical::settle_by_checked_holders`). Where HuggingFace answers, a part every
  checked holder disagrees with is a dispute the upload settles — the 64 KB check reads only a
  part's first tensor. One failed HuggingFace call ends asking for the pass (each retries
  ~155 s). A `source_path` exempts parts only while its file exists. `replace_parts` keeps a
  part it could not delete (Windows, a file still open) recorded as held. Rig
  `examples/outvoted_rig.sh`, 2026-10-03: B with no route to HuggingFace PASS (verdict 3 min
  after start, part back over P2P 19 s later, byte-identical), B in offline mode PASS (285 s);
  null control (v0.3.223 as B, no route to HuggingFace) FAILS as it must: the part stayed
  zeroed for 600 s beside two checked holders — `9594e1ff`'s shape, reproduced. **Residual, by design:** a node with no origin to
  ask cannot replace a copy of another LAYOUT (it is one upload's header, table and bytes,
  computes correctly, and the swarm's header comes only from HuggingFace — it stays, announced
  as the build it is), nor a part with fewer than two connected checked holders.
- #216 — a part the heal deleted right after a restart waited for the auto-manage loop's
  per-node phase offset (up to 300 s; 4.5 min measured on the Windows rig) before it was fetched
  again. The offset still spreads `evaluate()`; a repair wake-up during it runs the repair calls
  (`AutoShardManager::run`). Verified on the `outvoted` rig (node A, 2026-10-03): part deleted
  14:25:06, download started the same millisecond and done at 14:25:17, while A's loop only
  started at 14:28:03 — the fetch ran inside the offset it used to wait out.
- #215 — `/api/admin/models` named no peer behind `peers_other_build`: it now lists them as
  `peers_other_build_nodes` (node ids, sorted), and the count is that list's length
  (`ModelPeerCounts`), so the two cannot disagree. Finding the one peer that never healed after
  v0.3.223 took log archaeology the same day (the conflict line is written only on a change).

**Closed 2026-10-03, after v0.3.222** (`docs/invariants/network.md` § "One upload per model id,
on every node", paragraph "A node vouches only for bytes that are the swarm's upload"; gotcha #782)
- #164 — `.deb` installs made 2026-07-28 → 2026-10-02 with batching off: the template was fixed
  for new installs (`aa1f75c8`) and the release note shipped with v0.3.222; the project's Proxmox
  node had no `max_batch_size = 1` line at the .222 and .223 deploys. Not migrated automatically
  (a value someone may have chosen on purpose).
- #157 — a node holding another upload no longer stages the canonical copy from HuggingFace
  beside the old one: parts that are not the upload's bytes are DELETED and the upload's parts
  covering the same layers re-fetched through the repair queue — from peers when the part's
  hash is known, from HuggingFace otherwise (`auto_manage::canonical::replace_parts`). The
  staged switch, its queue (#213's `SwitchQueue`) and the `switching`/`stuck` states are gone;
  a copy that cannot be replaced yet (model in use, HuggingFace not answering) is `replacing`
  and withheld from the swarm. Rigs: `split_rig.sh spliced`, `canon_rig.sh`.

**Closed 2026-10-03** (next release after v0.3.221; history in the archive's § "Closed after 2026-10-02")
- #156 — a node computing with a header that describes another upload than its parts: the
  shard loader now compares every tensor-table entry with the header (name, offset, size) and
  refuses the load as `MixedModelCopy` (`split::loader::shards::first_header_disagreement`); a
  peer's refusal reaches the coordinator as missing shards on both serving paths (a layer
  forward and a whole-model hand-off), which retracts it and re-routes. `split_rig.sh mixed`
  reproduces it on v0.3.221 (a 200 answer of one repeated character, as in the field) and
  passes on the fix; 18 of 18 real copies on the release node pass the check. Of the entry's other parts, (b) a canonical check
  before the first load is not needed (a mixed copy is refused whenever it loads) and (c) a
  feature bit was decided against: it would exclude every not-yet-updated peer from splits for
  each rollout, while the refusal already tells the coordinator. Not established from the two
  peers' side which file was wrong there; the check covers every case of the shape.
- #158 — a peer's manifest of another build replacing the one a running download fetches
  against: refused while `model_download_under_way` (`SharedState::judge_peer_manifest`, the
  dispatcher's one decision for a peer's manifest).
- #213 — one model whose switch to the canonical upload could not go ahead blocked every model
  after it, on every pass, for ever (the turn was taken before the attempt) and was retried
  against HuggingFace every 2 minutes. Found live 13 h after v0.3.221: 12 holdings on at least
  4 peers never converged, one of them (`9594e1ff`, Docker) having switched none of its ~8
  models (gotcha #780). `SwitchQueue`: only a switch that fetched holds the turn; a
  HuggingFace failure backs off 10 min → 6 h. The newcomer catch-up also carries the upload
  claim with each manifest now.

**Closed during the 2026-10-02 rebuild** (no longer open, though the archive still lists them so)
- #11 — the KV store's "wandering" `allocated_bytes` was a store-wide figure read as one
  request's; the refusal now prints `live_bytes` / `external_bytes` / `entries`. Reopen on a
  refusal with `live_entries=1` and a total above that request's cache.
- #12 — `r134_receiver_applies_diff_and_advances_generation` load-sensitivity: not reproduced in
  48 runs at load 15.95; the `try_recv` hypothesis disproved.
- #113 — a result lost in transit is resent (`RESULT_STEP`, released v0.3.207; the .219 gate's
  three arms pass). One rig observation under `SWARMLLM_FAULT_RESULT=duplicate` (92 copies sent,
  20 logged, at INFO, which is lossy) stays unexplained, with no user impact.
- #139 — merged into #10. · #140 — merged into #162: the standby loss was the shared-RAM rig, and
  guessing ahead has been on by default since v0.3.213. · #146, #147, #148 — their shipped halves
  are in v0.3.213; the entries above are the residuals.
- "A computed segment result never reaches the coordinator" (2026-08-04) — not reproduced in two
  months; the one-way-dead-connection family it most likely belongs to (gotcha #353) received
  the `connection_rank` tie-break and the ACK fast-fail. Reopen with a log naming the pair.

**Shipped** (date of the fix; the version is in CHANGELOG)
- #2 peer ranking sees loss and bandwidth (goodput; v0.3.164, verified in a shaped lab) ·
  #5 the chat-template renderer is minijinja + pycompat (2026-09-10) · #29 split points honour
  what a peer can LOAD (2026-09-09) · #31 Qwen3 requests reached the model without the question
  (2026-09-10) · #33 a context that does not fit is shrunk, not refused (2026-09-10) · #34
  `release_escrow` after a failed persist (2026-09-10) · #35 a model's own tool framing is
  rendered (2026-09-10) · #37 a stray `<think>` in a tool reply (2026-09-10) · #38 a model's
  declared EOS was not its turn end (2026-09-11; #39 the same cause, recorded twice) · #40 Phi-4's
  native tool format (2026-09-11) · #41 a template reading tools per message (2026-09-11) · #42 a
  fenced tool call followed by prose (2026-09-11) · #43 partial-RoPE models (2026-09-11) · #44 a
  template refusing a system role (2026-09-11) · #45 streams stopping at 65 tokens (2026-09-11) ·
  #46 tool schemas alphabetised (2026-09-12) · #47 `tojson` HTML-escaping (2026-09-11) · #48 the
  cloud routing catalog rebuilt only by the admin page (2026-09-11).
- #50 a warm peer's capacity bound (2026-09-11) · #51 a second download of a part already on
  its way (2026-09-11) · #52 a peer reconnecting mid-request became undecryptable (2026-09-12) ·
  #53 a finished conversation's memory never released (2026-09-12) · #54 a request routed back
  here skipped the prefix cache (2026-09-12) · #55 (two entries) the card budget charged by
  metadata (2026-09-19) and Stop belonging to the page (2026-09-12) · #56 (two entries) Docker
  on Windows bound to loopback (2026-09-17) and the macOS memory bar (2026-09-12) · #57
  `max_bandwidth_mbps` and the traffic figure (2026-09-12) · #58 the traffic figure in every
  status payload (2026-09-12) · #59 the traffic-metric rename warning (2026-09-13) · #60 a peer's
  gossip could delete a correct part (2026-09-13) · #63 the activity ticker in the model card
  (2026-09-13) · #64 a scratchpad streamed after leading whitespace (2026-09-13) · #65 one word
  each for "part" and "computer" (2026-09-14) · #68 a settings save discarded hand edits
  (2026-09-14).
- The 2026-09-14 audit round: #70 a size read as zero on a failed stat · #71 undatable parts
  lost prune protection · #72 the tensor-parallel trailer unauthenticated · #73 "used recently"
  read a signal local requests never write · #74 the rebalancer counted unusable holders · #75
  Settings overwrote the config on a failed load · #76 unsaved settings reverted after 30 s ·
  #77 five writes announcing success on refusal · #78 the "Key source" select deleted by
  translation · #79 "also on N other computers" dropped from a finished download · #80 the
  scratchpad streamed when no tools were requested · #81 iOS scrolling the whole page.
- #83 a model card counted holders that could never send the file · #84 the activity list
  flooded by announcements (both 2026-09-14) · #85 a pipeline's context-window 400 discarded the
  reply (2026-09-19) · #86 "no route" answered 500 (2026-09-19; its premises were wrong — read
  the archive row) · #87 routing previews logged at INFO (2026-09-18) · #88 partial-reply salvage
  on distributed paths (2026-09-18) · #89 a dead graphics stack still advertised (2026-09-19).
- #92 a decrypt failure ended the request (2026-09-24) · #93 a worker retired on every token
  under card pressure (v0.3.201) · #94 the hedged verify step, removed (v0.3.204) · #95 a second
  split request failed for memory · #96 GLM-4/Llama-4/DeepSeek-2 RoPE layout · #97 prompts
  mis-tokenized for four families · #98 an uncompressed peer reply treated as broken · #99 a
  layer COUNT credited where ranges are keyed · #100 our system prompt replaced the template's ·
  #101 the Private Mode LAN switch needed a restart · #102 replies without a route · #103 a
  leading space from SentencePiece models (all v0.3.204) · #104 a first request planned onto a
  refusing card (v0.3.207) · #105 an undecryptable stream forward dropped silently (v0.3.205) ·
  #106 a remote sampling segment ignored the caller's sampling (v0.3.206) · #107 a CLI/env
  override dropped by a settings save (v0.3.205) · #108 shard hashing and manifest writes on
  the event loop (v0.3.206).
- #110 a LoRA request answered by the base model · #111 a peer serving a shorter context sank
  the request · #112 quinn-proto's "too many gaps" regression (all v0.3.206) · #119 CPU prompt
  reading level with llama.cpp (v0.3.208-.209; decode residual under § "Decided") · #120 the
  first `auto` request after a restart · #121 a second long prompt refused on a card with room ·
  #124 Llama 3.x `rope_freqs.weight` never read (v0.3.208-.209).
- #125 a worker that could not grow · #126 a whole-model peer run as an n-gram loop · #127 our
  own card priced from a cold load · #128 the hand-off ignored the request's privacy override ·
  #131 the stream map keyed by request alone (all 2026-09-27) · #130 the stream's speed gap —
  obsolete since V1Lazy (#136) · #135 card/processor split under concurrency (v0.3.211) · #136
  the protocol negotiation round trip (v0.3.211) · #143 the delegated split, a requester holding
  nothing no longer pays its distance per token (v0.3.219-.221) · #151 one upload per model id
  (v0.3.221; follow-ups #156-#160).
