# Future Work

What is still wrong or unbuilt, and what was decided against. **Every entry here was
verified against the code on 2026-10-02** (v0.3.221-alpha), when this file was rebuilt from
the 1.2 MB, 16k-line document it had become — about 345 entries, of which roughly 150 had
long since shipped without their entries being updated.

## How to use this file

- **Numbers are stable.** An item keeps its `#NNN` for life; a new item takes the next free
  number (**next free: #268**). Numbers below #165 come from the old triage index; #165 and
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

Re-ranked 2026-10-02 after the verification, and 2026-10-03 when #156, #158 and #213 closed,
then again after v0.3.223 (#157 and #164 closed, #160 raised to P1 on the swarm reading), and
after v0.3.224 (#160, #215 and #216 closed; #217 and #218 opened from the swarm reading and the inference test after it). #150, #153,
#165, #189, #214, #218 and #219 closed in v0.3.225 (released 2026-10-05); #221 closed and #129's second half shipped in
v0.3.226 (released 2026-10-05), #220 opened from the .225 gate's unrecorded driver resets; #129's fit verdict (idle-model
reclaim) shipped in v0.3.227 (released 2026-10-05 23:45 UTC); #222 (a node near its storage limit deleting and
re-fetching parts) and #223 closed in v0.3.228 (released 2026-10-06 09:42 UTC), #224 opened from its gate;
#225, #226 and #227 opened from a tester's two reports the same day; #117 (Qwen 3.5, dense), #225 and #226 closed in v0.3.229 (released 2026-10-07 00:00 UTC), #228 opened for what Qwen 3.5 leaves out; #129's cold loads (waited for and priced), #228 (2) and (2b), #220 (2), #229, #230 and #231 shipped in v0.3.230 (released 2026-10-07 16:38 UTC), #232 parked from their report; #234, #235, #239 and #240 closed on 2026-10-08 from two tester reports and the card check after them, #236-#238 opened from the second (#236 and #238 closed the same day); #242 and #243 opened and closed the same day from reports #006 (a 1 Mbps cap stored for "Unlimited") and #007 (the narrow-screen model picker); #194 closed on 2026-10-09 (a card keeps its conversation memory as f16, and its memory guard under a context override), #237 re-ranked to P3 from what could be measured; #152 closed the same day (the guess-check stream on the boomerang), and #147 narrowed to P3 (f16 accumulation in a card's prompt pass shipped). v0.3.233 was built and never released (its gate found #247 counting active file cache as used); v0.3.234 (released 2026-10-11 00:37 UTC) carries #69, #247 and its fix, #162, #10's split half, #171's pieces, #91's capability change-gate and #3's terms DIAG; #224's trigger was caught at its gate. v0.3.235 (released 2026-10-12 01:58 UTC) carries #224, #137's main half, #138's decode, card and prompt steps and #228 (1), (3) and (4); v0.3.236 (released 2026-10-12 14:31 UTC) carries #244, #248, #249, #250's two halves, #251 (1), (2) and (4), #252 (2), #253 and #254, and #257 opened from its gate; #257 closed on 2026-10-13 (under WSL2 a card's saved prompts are kept in host RAM) and #258 opened from its measurement; #138 (1), (1b), (1c) and (1d) shipped on 2026-10-13 (a card/processor split runs a prompt's processor-held experts, then every processor-held projection, then each processor-held layer whole, on the card, copying the next weights while the card computes); #259 and #260 opened and closed the same day from a tester's report (a kept model bigger than the card was refused; a model sent to the processor while another program held the card never moved back), #261 opened from it; #138 (1e) the same day (a split reads a prompt in chunks as large as the card has room for), and #262 opened from its rig (a split's conversation budget charges the processor's caches to the card); #263 opened and closed the same day from a tester's report (a split's local segments refused a long prompt only after the peers had read it, and the retry planned the same split), then #262 (each cache charged to the memory it lives in), and #264 opened from it and closed the same day (a split's processor-held weights are charged to the memory limit), #265 opened from that; #138 (3)'s first stage the same day (a mixture-of-experts layer on the card can keep its experts in RAM, measured from `test-split`: 1.2-1.7x decode at the same card bytes); its second on 2026-10-14 (the daemon chooses it for a mixture-of-experts model that does not fit the card), #266 opened from its rig and closed the same day (the owner's own loads are weighed against the owner's memory cap, not the swarm's); #267 opened and closed the same day (a restart no longer opens a dashboard tab beside the one already reconnecting).
`docs/plans/` holds the multi-step designs; the entries point at them.

**P0 — wrong answers, silently**
1. *(none open: #156 closed 2026-10-03 — a mixed copy is refused at load. Its last doors
   closed after v0.3.222 the same day: a part in dispute is announced as the bytes it is, a
   part that is not the upload's bytes is DELETED and re-fetched (from peers or HuggingFace),
   a copy waiting for that is withheld from the swarm, and every part entering from a peer is
   byte-checked against the upload — `docs/invariants/network.md` § "One upload per model id". Rigs
   `split_rig.sh disputed` and `spliced`: v0.3.222 answered garbage through such a peer, and
   from such a copy on its own node (`给给给…`); the fix deletes and re-fetches.)*
2. **#217** — a node kept three parts that are not the upload's bytes for ~3 h on v0.3.224,
   with two checked holders disagreeing, and replaced them within minutes of a restart. It
   recurred at that node's next restart (09:01 UTC 10-04) as replacements that ARRIVED wrong
   from HuggingFace, twice, and were kept — that mechanism is closed in v0.3.225 (a download is
   kept only when corroborated; the downloader syncs and reads back). The
   swarm itself converged: `peers_other_build` 0 on every model at 06:19 UTC 2026-10-04 (all
   six nodes on .224; the watch this slot held is closed — its readings are in the archive,
   § "Closed after 2026-10-02" → "P0-2").

**P1 — silent, or broken for a whole class of users**
3. *(#159 decided 2026-10-11 — what is left needs a v0.3.222 holder of a same-layout copy; §
   "Decided — not doing".)*
4. **#1** — every Mac runs on the processor; no GPU backend is compiled for Apple Silicon.
5. **#220** — a worker's card faulted once (illegal memory access) and the driver took 14 minutes
   to reset; the kernel is still unknown — ~27 min of its shape under memcheck (10-10, 10-13, the tool proven in
   the card workers) found nothing; (c) one process per card weighed and declined on that evidence (see the entry).
5a. **#250** — a model held whole here, which the card holds only part of, was routed through two
   peers at 0.1 tok/s where the same machine answers locally in ~5 s (report #035). Its
   window-narrowing contributor closed as #248, the idle model holding the card as #249, and a layer
   is priced as its own model's, a measured peer's figure per model too (all released in v0.3.236).
   The report is NOT reproduced: in every shape this box can build, both prices keep the model
   local; a far split needs a peer that can hold the middle (see the entry).

**P2 — speed and completeness**
6. *(#152 closed 2026-10-09 — the guess-check stream runs on the boomerang and with a peer
    holding the first layers, not only with a peer holding the last.)*
7. **#10** — conversation prefixes across computers: a peer's cache this node did not fill is
    invisible (no digests gossiped) and caches end at the 10-minute idle expiry (the split half
    released in v0.3.234; routing to the holder this node knows of released in v0.3.235).
8. *(#162 closed 2026-10-09 — cloned gates refuse a missing helper; 12e's failover arms moved to a
    shape this box holds, and `split_rig.sh failover` keeps no prompt since #10.)*
9. *(#155 narrowed to P3 on 2026-10-07 — not seen on the swarm, see its entry.)*
10. *(#194 closed 2026-10-09 — a card keeps its conversation memory as f16, a third of the
    memory, and keeps its memory guard under a context override.)*
11. **#138**, **#129** — MoE placement, and near-fit models sent across continents (#147's f16
    accumulation shipped 2026-10-09; its smaller items are P3; #138 (1) on 2026-10-13: a split's
    prompt experts run on the card, 1.4x; (1b) the same day: every processor-held projection,
    a dense 14B split's prompt 5.3-5.7x; (1c) the same day: each processor layer whole on the card,
    1.6x more; (1d) the same day: the weight copies beside the products, 3-9%; (1e) the same
    day: prompt chunks as large as the card has room for, 1.5-1.7x; (3) stage A the same day:
    a MoE layer on the card keeps its experts in RAM, decode 1.2-1.7x at equal card bytes;
    stage B on 2026-10-14: the daemon chooses it, ~10% faster decode and ~0.4-0.7 GB less of
    the card than the whole-layer split). *(#137 narrowed to P3 on
    2026-10-11: decode ran outside its thread pool on every default node — fixed: a 14B split
    1.19x, now level with llama.cpp; a 3B processor-only 1.14-1.39x; ~12% behind it remains.)* (`docs/plans/faster_than_local.md`).
12. **#3**, **#180** — the routing cost model charges a constant where the reply length
    belongs, which keeps partial ranges (load spreading) off.
13. **#171** — the prompt pass through a split runs one stage at a time (released in v0.3.234; left:
    pieces chained between two peers for a pass that does not keep its prompt).
14. **#228** — Qwen 3.5's first version leaves out speculation (its card/processor split shipped
    in v0.3.230; its MoE models run, its delta rule is one op per layer and its decode steps are
    captured as CUDA graphs since v0.3.235).

Everything else is ranked in its own entry. **P3** is narrow or cosmetic, **P4** is process,
maintenance or an idea with no user waiting on it.

## Open items

### Wrong answers and the one-upload-per-model rule

#### #217 — A node can keep parts the checked holders disagree with until it restarts
`P0` · heal · **OPEN** — 2026-10-04 (the v0.3.224 swarm reading) · history: archive § "Closed after 2026-10-02" → "P0-2"

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

#### #179 — A greedy reply (`temperature: 0`) is not reproducible run to run on the processor
`P3` · correctness · **OPEN** — 2026-08-18 · history: archive § "`temperature: 0` with a fixed seed is not reproducible", gotcha #327

Same binary, node and prompt at `temperature: 0` gave different (coherent, correct) wording on
two runs; the null control (old code path) was equally non-reproducible. Never investigated:
a float reduction whose order varies (rayon in the CPU pools) or near-tied logits are the
suspects. Measure by running greedy twice on one node and diffing logits
(`examples/logits_reference_probe.rs`). Consequence until then: generated text cannot compare
two code paths — assert on the tensor a change touches. (Honouring a request's `seed` for
sampled replies is a separate, deliberately deferred item: `docs/ARCHITECTURE.md` § "Deferred
Items".)

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

#### #184 — A chain hop whose onward peer departs waits out the whole segment deadline
`P3` · reliability · **PARTIAL** — 2026-09-02 · history: archive § "A pipeline whose peer departs with no standby waits the whole segment deadline"

Fixed for the coordinator's own forwards (`schedule_redial_retry` →
`fail_forwards_awaiting_departed_peer`, gotcha #436). A CHAIN HOP's onward forward is not
covered: `hop_reply_to` (network manager) does not record the onward peer, so a departed
onward peer in a 3+-hop chain still costs the ACK deadline. Record it and fail the hop
promptly; add the no-standby test (connection closed → fails before the deadline) with the
control that a silent-but-connected peer keeps its full deadline (#386).

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
`P1` · reliability · **OPEN** — 2026-10-05 · history: gotcha #790

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
it is updated"). Unknown: which kernel. Next, in order: (1) run that rig shape under
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
  `vertex_cost`), where `node_tokens_per_sec_7b` = memory bandwidth ÷ `MODEL_SIZE_7B_Q4` (4.4 GB) ×
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
  layer's (`MODEL_SIZE_7B_Q4 / BASELINE_LAYER_COUNT`). Active bytes are the layer's non-expert
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
  card, while #138 made a mixture-of-experts prompt 3.5-4.7× faster on both.
- **An idle model held the card** (#249, closed): a 14B "idle 20m" still had 9.2 GB, which is what
  left the 30B 18 layers.
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
untouched. The next evidence is a `DIAG: pipeline candidate … layer_work=` line from a node that
detours, or a second machine able to hold the middle.

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

#### #3 — The routing cost model's network term overestimates a boomerang: a constant stands where the reply length belongs
`P2` · routing · **OPEN** — 2026-09-08 · history: archive row #3 and § "The routing cost model's network term overestimates a boomerang" (three dated measurements)

`scheduler/parallax.rs` charges a remote hop a flat `ASSUMED_FORWARD_PASSES = 64.0`. The error
grows with reply length — 1.0 at 8 tokens, 2.4-3.3 at 120-239 — and biases every delegation
decision since v0.3.164, silently. The structural claim is settled; **do not tune the
constant**. Next: an estimator of the expected reply length (candidate source: the per-session
response histories in `state.metrics.prefetch_orchestrator`, unused for this) plus a
per-token hop term. Also: peer speed does not track RTT, and a chain runs at its worst
segment's pace, so a term linear in layers cannot express it. This gates turning
`parallax_partial_ranges` on (#180) — on 2026-09-20 the flag on made a 4-segment, 3-peer
route that did not finish in 580 s. Method: `memory/perf_baseline_0920_post192.md`,
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
observation that already carries it. Next: log each chosen plan's cost terms beside its outcome
(today only the total is kept), then fit per term. Script: re-run the analysis with
`grep "request complete" node.log | grep predicted_ms` over the fields above.
Field point, 2026-10-13 (a tester, v0.3.236): a six-segment boomerang through three countries
(37-token prompt, 120 tokens) was predicted 8,772 ms and took 59,457 (6.8×), its terms
`net:0/compute:5441/prompt:2902` — `net:0` is by design where every peer is measured and no
per-visit split is known (`vertex_cost`: the round trip is inside the per-layer observation), so
here the hops' cost sits inside a compute term that priced the chain at 5.4 s of the 59.5 s it
took.

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
processor's prior for a split, and a remote hop's network is charged per segment rather than per
crossing (the regional-pipelines plan).

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

#### #172 — The greedy fallback assigner prices a long prompt like a short one
`P4` · routing · **PARTIAL** — 2026-07-24 · history: archive § "Two cost models are still prompt-blind"

The delegation half is fixed (`DelegationInput` carries `prompt_tokens`).
`scheduler::estimated_cost_per_layer` still takes no prompt length; it runs only when the
priced search finds no path (rare). Thread `prompt_tokens` into it (the prefill term as in
`vertex_cost`), and add a comment at `distributed_exec.rs`'s reuse-previous-assignment branch
saying later turns keep the first turn's route on purpose (KV affinity).

#### #198 — A node that appears twice in one chain is capped per segment, not in total
`P4` · routing · **OPEN** — 2026-09-04 · history: archive § "A node that appears twice in one chain is bounded per segment, not in total"

With prompt privacy the local node can hold two ranges of one chain; `max_hostable_layers` is
applied per segment, so the search can give it more than it can load. Admission still refuses
the second range and the coordinator fails over — a 503 and a re-plan, not a wrong answer.
Carry layers-already-assigned-to-this-node in `route_shortest_path`'s state so the cap checks
the running total; measure on a real two-segment plan first (synthetic tests get the marginal
case wrong).

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

#### #91 — An idle node still uploads ~20 KB/s of gossip, and a total traffic ceiling does not exist
`P2` · network · **PARTIAL** — 2026-10-11 · history: archive row #91, `docs/invariants/network.md`

Trickle suppression (`MANIFEST_QUIET_WINDOW`), per-(model, hash) `manifest_heard`,
per-category counters and the region-demand split shipped in .200-.206 and halved it (42 →
~20 KB/s each way; it once extrapolated to ~159 GB/month — the complaint that makes people
uninstall a P2P app). The capability broadcast is change-gated since v0.3.234 (KEP-589).
Re-measured 2026-10-11 on the release node with the fleet on .234 (4.1 h up, 7 peers, idle):
8.7 KB/s out, 5.8 of it gossip; the capability's first copies 976 → 167 B/s; manifests
59% of the gossip sent, every one a repeat, each arriving ~4.2 times. So the manifest
quiet window went 30 min → 2 h (released in v0.3.235, 2026-10-12; RFC 6550's Trickle ceiling is ~2.3 h),
`docs/invariants/network.md` § "The manifests again, once the capability was gated". Open:
(1) re-measure once the fleet runs the 2 h window (`ModelManifest` first copies ≤ ~0.004 a
second expected, from 0.0123); (2) announce-then-pull for manifests — gossip `(model, hash)`
and fetch the manifest point to point only for a version one lacks (Bitcoin
`inv`/`getdata`, Plumtree lazy push), since the copies, not the size, are now the cost; it
replaces the table-less manifest, which needed `compute_hash`'s verify-before-header order
reversed. A new `SwarmMessage` variant plus a pull request, feature-bit gated at the sender —
a broadcast, so digest-only when every connected peer advertises the bit; (3) a
both-direction traffic ceiling at the transport (risky — throttling gossip can partition a
node); `max_bandwidth_mbps` shapes part serving only.

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
half: DHT provider records are add-only and can outlive the fact by up to 24 h (see #49); a
capability challenge before trusting one would address it. No incident is known.

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

#### #137 — Our processor decode trails llama.cpp's by ~12% at the same width, and loses where llama.cpp gains from 8 threads
`P3` · perf-local · **OPEN, narrowed 2026-10-11** — 2026-09-27 · history: archive row #137; plan: `docs/plans/faster_than_local.md` § 3.3

The old title — a split's processor layers read at ~17 GB/s, a third of the memory — was a
09-27 figure (28 processor layers, 3.35 tok/s) and no longer holds. Measured 2026-10-11 on the
Ryzen 7 5800H, Qwen2.5-14B Q4_K_M, live node stopped, 60 s cool-down before each arm, arms
interleaved (`~/swarmllm-137/`): the default node placed 30 of 48 layers on the card and decoded
at 5.3-5.4 tok/s against llama.cpp's 6.8-7.0 (`llama-bench -ngl 30 -t 4`, CUDA build of
`4b1a27f`). That gap was decode running OUTSIDE its thread pool on every default node — fixed
(§ Closed 2026-10-11): the same node now decodes the split at 6.79-6.93 tok/s, level with
llama.cpp. What is left, forward-only (`SWARMLLM_PROFILE=1` totals, installed):
- **Split step**: ours 151-160 ms vs llama.cpp's 143-147 ms (5-10%).
- **Processor only, 4 threads**: ours ~312 ms vs llama.cpp's 278 ms (3.60 tok/s), ~12%.
- **At 8 threads llama.cpp gains** (3.97 tok/s, 252 ms) **where we lose**: calibration keeps 4,
  and forcing 8 cost the split ~10% (4.9-5.0 tok/s). The per-core draw that saturates this
  box's memory at 4 threads for us does not for llama.cpp — its CPU path repacks Q4_K into
  interleaved rows (`CPU_REPACK`, Q4_K_8x8); compare one Q4_K matvec there before anything else.
Not ours, and llama.cpp's too: the card clocks down between steps of a split (255-600 MHz
between 1755 MHz bursts), and back-to-back runs on this laptop heat it enough to slow a split
step 133 → 185 ms within 40 tokens (gotcha #825). Neither thread count nor placement is the
lever: calibration already picks the width, and `SWARMLLM_DECODE_THREADS=8` was slower.

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
vendored `quantized::matvec_many`, llama.cpp's `mul_mat_id` shape), bit-identical to the
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
may be slow at high uptime (#257), measured at 3.1 h only. (2) **MoE layers in decode-step
capture**: a stacked layer has
no host copy, so a MoE model whole on a card could be captured (`executor.rs` still excludes
MoE as unchecked; a 26/22 split stays uncaptured regardless). (3) **the gap to llama.cpp at
the experts-in-RAM placement** (40 vs 29 ms in stage A's rig): the processor's 23-25 layers of
experts take 21.6-31.8 ms of a step at the default 4 decode threads. Seen, not chased: the first 1-2 decode
steps after a 681-token single-forward prompt took 2-4x a settled step on both the stacked
and per-expert paths (not after a one-row prompt) — prompt temporaries pressing card memory
is the likely reading; a node chunks card prompts at 512 tokens.

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

Designs: `docs/plans/split_speculation.md` (guess-and-check across a split),
`docs/plans/wan_parallel.md` (parallelise the reply, not the token), `docs/plans/regional_pipelines.md`
(placement by distance, #192), `docs/plans/faster_than_local.md` (the ranked list).

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
  prompt against a GPU peer's rate (`faster_than_local.md` § 3.1). Delivery trap for a WHOLE
  model on a peer (no split credit reaches it): `try_ngram_only_distributed` runs before
  `try_remote_generate_fastpath` and takes a remote single segment, so a priced credit would not
  be delivered — make the n-gram path decline a plan priced warm.
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
time, so the ceiling is ~2-2.8× on long prompts. The receiver reassembles the whole chunked
forward (`try_assemble_chunked_forward`) before computing; instead compute each chunk on
arrival and pass it on (chaining is already the default). Needs a feature bit and a KV
rewind for a partially streamed prompt pass. Measure first: per-segment times from
`RequestTrace::record_segment_timing` (largest stage ÷ sum = the achievable ceiling); A/B
inside one binary with an env switch. This is also step 3 of `split_speculation.md` (pipelined
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

#### #134 — Split decode: draft trees, a drafter the node does not hold, and the pipelined prompt pass are unbuilt
`P3` · perf-split · **PARTIAL** — 2026-09-27 · history: archive row #134; plan: `split_speculation.md`

Shipped: guessing ahead on by default (v0.3.213), the continuous stream (v0.3.216, #149), walk
at the tail, γ control, the engine drafter. Remaining from the plan: acquiring a drafter a
node does not already hold (today it chooses only among held models); draft TREES (Phase 2,
projected ~10-15 tok/s at 300 ms; needs a `LayerForward` trailer gated on a feature bit and
tree attention masks in the worker — absorbs the old survey's Tier 2D); the pipelined prompt
pass (Phase 3) is #171. The same-model-at-fewer-bits drafter is decided against as a default
(#144, § "Decided").

#### #145 — One reply uses one row of a ring pass that could carry dozens
`P3` · perf-split · **OPEN** — 2026-09-28 · history: archive row #145; plan: `docs/plans/wan_parallel.md`

Every machine in a split idles ~99% of each pass. The plan — "parallelise the reply, not the
token", within the rule that nobody holds a whole model — has its first step done (the
continuous stream, v0.3.216) and #143 done (v0.3.221; the plan's § 8 still lists it as
pending). Next in its order: multi-token-prediction heads at the last stage (needs Qwen 3.5/3.6
or GLM-4.5+ support first, #117), a stage batching forwards of several streams into one
visit, multi-block attention for workers sharing a cache, outline-then-expand with a
classifier router. Research-grade; the largest multiplier on the list
(`wan_parallel.md` § 6 has the projection).

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

#### #228 — Qwen 3.5: what the first version leaves out
`P2` · model-support · **OPEN** — 2026-10-06 · history: #117 (closed); plan: `docs/plans/qwen35_support.md`

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
too"). MoE layers still stay out (#138 (2)); (5) no speculation of any kind — a
snapshot of the recurrent state per draft would allow it; (6) ~~the 4B's gap to llama.cpp~~ —
checked 2026-10-07, not a defect: six prompts scored on the processor against llama.cpp master
(`score_ids_dump.py`, margin 0.05) — Qwen 3.5 4B Q4_K_M misses on 3 (gaps 0.11-0.21), and
Llama-3.2-3B Q4_K_M, a long-verified family, misses on 1 by MORE (0.306, "why the sky is blue").
Unquantized, the 0.8B matches llama.cpp exactly (worst cos 0.999999), at Q8_0 within 0.012: the
model's arithmetic is right and 4-bit near-ties fall the other way through our kernels' rounding. Re-check with
`~/llama.cpp-ref/{dump_logits,ref_generate}` (llama-cpp-python 0.3.16 cannot load `qwen35`), and
render a new template with jinja2 first (gotcha #798); (7) **the 4B decodes at ~37 tokens a second whole
on an RTX 3070 Laptop with or without CUDA-graph capture** (2026-10-11): ~26 ms a step reads its 2.5 GB of
weights at ~96 GB/s against the card's 448, while the 0.8B went 50-54 → 121-126 with capture — so the 4B's step
is bound by something on the card. Profile it (`SWARMLLM_PROFILE=1` turns capture off, which is fine here)
and compare llama.cpp's `llama-bench -ngl 99 tg128` on the same file before changing a kernel.

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

#### #115 — Gemma 3 and Gemma 4, and their vision projectors, are unsupported
`P2` · model-support · **OPEN** — 2026-09-25 · history: archive row #115

Requested for gemma-4-26B-A4B. Order: (1) Gemma 3 text needs sliding-window attention — a
mask plus a second RoPE base on local layers (5 local : 1 global), while the loader builds one
cos/sin table per segment; (2) the `gemma3` projector (SigLIP + pool + RMSNorm + linear), after
`vision.rs` gets a projector-type dispatch; (3) Gemma 4 text — cross-layer KV sharing,
per-layer embeddings, dense FFN + GELU MoE; (4) `gemma4v`. Gemma 3 can be checked against the
local llama-cpp-python 0.3.16 now; Gemma 4 has no local reference (0.3.16 predates it; sources
in `~/swarmllm-ref/gemma4/`).

#### #116 — The DeepSeek-2 family is recognised and refused: no real file can load
`P2` · model-support · **OPEN** — 2026-09-25 · history: archive row #116

DeepSeek-V2/V2-Lite/V3/R1, Kimi-K2, GLM-4.7-Flash. The loader's MLA branch, `MlaWeights` and
`LayerVariant::DeepSeek` stay as the starting point. A real file needs: (1) the
`attn_kv_a_mqa` tensor name; (2) per-layer MLA detection when `attn_q_a` is absent
(V2-Lite); (3) the split `attn_k_b`/`attn_v_b` layout and `key_length_mla` keys; (4) YaRN RoPE
scaling and attention mscale (the engine has none); (5) routing — `expert_weights_norm`
default false, `expert_weights_scale`, the `exp_probs_b` bias, expert groups, the gating
function. Verify on a tiny GGUF of each layout, then flip `is_supported` and review the MLA
projections for `arch_supports_hybrid`.

#### #114 — No real mixture-of-experts model has ever run here
`P2` · model-support · **PARTIAL** — 2026-09-25 · history: archive row #114

Experts stay quantized, one matrix per expert (`loader::load_moe_ffn`, `split_expert_stack`);
Qwen3-MoE and Qwen2-MoE are verified on tiny random models, Llama-4 routing on tiny GGUFs
(v0.3.207). Open: (a) download one small real MoE (Qwen3-30B-A3B class) and score it with
`score_against_reference.py`; (b) Llama-4 NoPE temperature tuning and 8,192-token chunked
attention are not implemented, so replies past 8,191 tokens differ; (c) the Qwen3.5-MoE
shared-expert gate is unverified; (d) device-side top-k only if a profile asks. Placement on
a small card is #138.

#### #118 — StarCoder2 is recognised and refused: a LayerNorm model with biases the loader cannot read
`P3` · model-support · **OPEN** — 2026-09-25 · history: archive row #118

Needs a LayerNorm variant (mean-subtracted, with bias), bias sites on `attn_q/k/v/output` and
`ffn_up/down`, the `layer_norm_epsilon` key in `GgufTensorMeta::from_content` (it requires
`layer_norm_rms_epsilon` today), no `ffn_gate` (GELU MLP), tied output. Verify on a tiny GGUF
against llama.cpp 0.3.16, then flip `is_supported` and re-read `arch_supports_hybrid` (it lists
StarCoder2, inert while refused).

### Memory and admission

#### #265 — A later range loaded into a card/processor split worker is charged to the card whatever the loader does with it
`P3` · memory · **OPEN** — 2026-10-13 · history: found closing #264

A split worker is spawned with `--gpu-layers N` (`GPU_LAYER_LIMIT`), and the loader puts the first
N layers of EVERY range it loads on the card — so a second range (a failover onto this node, a
re-plan) puts N more there than admission sized the card for. `charge_additional_segment` charges
the whole range to the worker's one accountant (`holds_gpu_memory` → the card) at the card's
per-layer rate, which refuses it when the card is full (a `LocalMemoryUnavailable` the router
re-plans) rather than overcommitting — so the cost seen is a split worker that cannot take a second
range, not a crash. Not seen in a report. Next: place a later range's layers by the room each
device has (the card's share against `vram_budget_now`, the rest on the processor) and charge each
share to its own budget, as the spawn now does (#264).

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
`P2` · memory · **PARTIAL** — 2026-10-12 · history: report #035 (a tester, RTX 3060 12 GB, v0.3.234)

Qwen3-14B on a 12 GB card: `/v1/models` says `max_model_len: 8192` (the declared limit under
`DEFAULT_MAX_SEQ_LEN`), but with the weights loaded the KV budget is 860 MB, and a 6,228-token
prompt was refused (`short_by_mb=259`; usable ≈ 5k). A client (Open WebUI, Home Assistant) learns this only from failed
requests. **Done (2), 2026-10-12, released in v0.3.236:** the refusal now names the longest prompt that fits
("up to about 4608 tokens with this model here", `model_worker::longest_prompt_that_fits`, rounded
down to the cache's growth step and less the reply reserve). Next: (1) while a worker is loaded,
report the positions its KV budget admits as the effective limit (`max_model_len_for`, both
`/v1/models` and the MCP model list) — the daemon does not know the worker's card KV budget, so the
worker has to report it first. The same figure, gossiped, would let a planner size this node's and
a PEER's share of a split before asking: since #263 a split's LOCAL segments are admitted together
before any peer reads the prompt, and a re-plan holds this node to what fits, but a peer's segment
is still admitted when its forward arrives. Never shorten the context itself (`arch-worker-memory.md` § "A
card's KV budget is kept whatever the context setting").

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

#### #141 — A non-streamed reply does not carry a reasoning model's scratchpad; the stream does
`P3` · api · **OPEN** — deferred 2026-09-28 · history: archive row #141

Streamed, a `<think>` block arrives as `delta.reasoning_content`; non-streamed, a FINISHED block
is drained by `finalize_reply_text` and dropped. Make `finalize_reply_text` return a struct
(content + reasoning) that every one of the eight reply sources must destructure (`local_exec`,
`executor`, `process_pool`, three distributed, speculative, `remote_generate`) — not a second
helper nobody is obliged to call. Anthropic: emit a `thinking` block only when the request
enabled thinking, with a signature this server cannot produce; first confirm `api/anthropic`
drops thinking blocks on input (Claude Code sends them back).

#### #186 — A reply emptied by finalisation is still returned as a success
`P3` · api · **PARTIAL** — 2026-07-29 · history: archive § "An empty completion is still reported as success"

A WARN in `finalize_reply_text` and the `swarmllm_empty_replies_total` counter shipped; the
reply is still HTTP 200 with `finish_reason: stop`. Decide from the counter whether an emptied
reply becomes an error so retry-capable clients re-route. Only when the model generated
something and finalisation removed all of it — an empty reply can be legitimate. Implement at
`finalize_reply_text` only (all text sources inherit it).

#### #196 — MCP reports every tool failure as a protocol error, never `isError`
`P3` · api · **OPEN** — 2026-08-18 · history: archive § "MCP reports every tool failure as a protocol error, never `isError`"

Recoverable failures ("no models loaded", a peer timeout) reach MCP clients such as Claude
Code as a broken server, and the model may never read them. Add a `tool_failed_result` beside
`tool_text_result`, move the recoverable failures there, and keep `tool_error_code` for genuine
server faults (`delegate` already classifies through it). Verify against Claude Code itself.

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

#### #176 — Cross-pool inference can only be switched on by editing the config file
`P3` · ux · **PARTIAL** — 2026-07 (R134) · history: archive § "Inter-pool model sharing policy"

Discovery and opt-in routing between pools shipped (R134-R134.7:
`PoolModelAvailability`, `foreign_pool_catalog`, `pool.allow_cross_pool_inference`); no
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

#### #255 — `swarmllm status` reads the default data directory, not the running daemon's
`P3` · cli · **OPEN** — 2026-10-12 · history: report #035 (a tester)

Without `-d`, the CLI reads `~/.local/share/swarmllm/api_key` and fails against a daemon started
with `/opt/swarmllm/data` (the error explains it). Next: the CLI asks the daemon on the port for
its data directory, or the daemon leaves its data directory where the CLI looks first (a runtime
file), before falling back to the default.

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

#### #163 — The .rpm installs a service whose user it never creates
`P3` · packaging · **OPEN** — 2026-10-02 · history: archive row #163

`packaging/swarmllm.service` runs as `User=swarmllm`; the .deb's `postinst` runs `useradd`, the
.rpm (`[package.metadata.generate-rpm]` in Cargo.toml) has no scriptlets, so the service cannot
start after `rpm -i`. The installation guide gives the `useradd` step meanwhile. Fix: a
`pre_install_script` doing `getent passwd swarmllm || useradd --system --no-create-home --shell
/usr/sbin/nologin swarmllm`, verified by installing the built .rpm in a Fedora/Rocky container
and starting the unit; then drop the doc workaround.

#### #183 — A Windows GPU node's auto-update never refreshes the bundled CUDA libraries
`P3` · packaging · **OPEN** — 2026-08 (v0.3.20 asset audit) · history: archive § "Windows-GPU auto-update carries stale CUDA redist DLLs"

`update.rs` swaps only the executable; the CUDA redistributable DLLs ship in the `-gpu` zip
only. Safe while the Windows toolkit stays pinned to 12.x (`gpu-build-env`), and a silent
failure for every auto-updated Windows GPU node the day the CUDA major is bumped. Before
touching that pin: teach the updater to fetch and unpack the zip's DLLs (multi-file apply with
rollback), or put the CUDA major in the asset name and refuse a cross-major auto-update. Add a
pointer beside the "bump both together" comment in `release.yml`.

#### #109 — The bundled Prometheus/Grafana stack scrapes nothing
`P3` · observability · **OPEN** — 2026-09-24 · history: archive row #109

`monitoring/prometheus.yml` targets `localhost:8800`, which inside the container is the
container; `docker-compose.yml` maps `host.docker.internal` but nothing uses it; and `/metrics`
is exempt from the API key only for loopback, so a bridge-network scrape gets 401. The file's
"no auth required" comment is misleading. Pick one: a compose-only scrape config targeting
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

#### #251 — `owned_by: "local"` reads as "my data stays here"; it means "every part is held here"
`P2` · privacy · **PARTIAL** — 2026-10-12 · history: report #035 (a tester, v0.3.232-.234); (1), (2) per model and (4) done 2026-10-12

`api::model_source_for` answers `local` when this node holds every part, which says nothing about
where a request runs. A model the card cannot hold whole may be split with peers (#250). The default boomerang starts and
ends that route here, so peers never see the prompt TEXT. They do see hidden states, ~81% invertible
to text (`docs/ARCHITECTURE.md` § Pipeline Privacy Model). Today a user can keep requests on their own
computer only node-wide (a pool with `private_mode = true` and `private_mode_allow_lan = false`).
`x-swarm-route` reports the route on the OpenAI and Anthropic surfaces. **Done (1), 2026-10-12:**
the book's OpenAI page says what `owned_by` means, what a peer sees, how to keep requests here and
where the route is reported. **Done (2) per model, 2026-10-12:** "Keep on this computer" on the
model's ⚙ panel (`PUT /api/admin/models/{id}/keep-on-this-computer`). `pool::scope::request_scope`
is now the one answer to which nodes a request may reach (kept → this node; else private mode's set),
asked by every path a request's data leaves by (guard `every_path_a_request_leaves_by_asks_its_scope`);
a kept model this computer cannot run whole is refused as `KeptOnThisComputer` (503, never "retry").
Enumerating the paths found three more holes, closed with it: the image encoder went to its nearest
holder in private mode (`c84c12dd`), the prefix-KV probe asked any holder of the cached blocks, and
every failed request was broadcast to the whole swarm with its error text (only ever logged by the
receivers — removed). And a cloud provider is chosen by NAME, so a local model called `mistral…`,
`llama-…`, `gemma…` or `deepseek…` could go to that provider when the swarm lacked a part of it:
`providers::cloud_refused_for` now refuses a kept model on every cloud path (guard
`every_cloud_path_asks_whether_the_model_is_kept_here`). A kept model's stored prompt openings are
neither announced nor served to peers. **Done (4), 2026-10-12:** private mode had the same cloud
hole for a model of the swarm; in private mode a model with a manifest here now never goes to a
cloud provider (`PrivateModeUnavailable`, naming the parts no pool device holds —
`pool::scope::parts_outside_scope`, shared with the scheduler). A cloud-only name still goes, being
the owner's own configuration, and so does an explicit `provider:` prefix (the caller's own choice of
a cloud model; the same prompt could go under the provider's model name). A kept model refuses both. Left: (2b) the same choice per REQUEST (a `swarm_route` value or
header on all four surfaces); (3) the route on the surfaces that lack it. A NON-streamed `/v1/responses` already carries
`x-swarm-route`, `-segments`, `-peers`, `-nodes` and `server-timing`, copied from the chat
handler's response (`create_response`; checked on v0.3.235, 2026-10-12). A STREAMED one carries
none: `stream::run_streaming` returns the SSE before it awaits the chat call, so its headers
flush before a route exists. That is deliberate, so the lifecycle events go out before inference
setup blocks. The choice is between waiting for planning before `response.created` (as
`/v1/chat/completions` streaming already does) and naming the route in an event. MCP answers
JSON-RPC, and one call may fan out to several models, so its route belongs per result (`_meta`),
not in a header. `attach_route_headers`' doc claims all four surfaces; correct it with the fix.

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
`P3` · storage · **PARKED: no field reading yet** — 2026-09-13 · history: archive row #61 and § "A disputed shard is kept but the disagreement is never settled"

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
bundling. Independent leftovers: `/v1/embeddings` still answers 501 (wire
`inference::local_embedder` to it), and the 8192 default context wants a decision (#189 helps).

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
  `examples/prefill_bench.rs`; decode streaming pays only with wide speculation, and #194's f16
  cache comes first.
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
- **A reply that is ONLY a finished `<think>` block** shows the scratchpad when streamed and
  nothing when not — left deliberately (no field report; #141 is the general fix). Archive §
  "Known and deliberately left: a reply that is ONLY a scratchpad".
- **`candle-*` held at 0.10** — vendored and patched; bump only when upstream carries something
  we need. Archive § "Major-version dependency migrations Dependabot may not do".

## Closed

Every number that is no longer open, with how it closed. Numbers 6-9, 13-16 and 19-28 were
retired before the 2026-09-09 index existed. The history of each is in the archive (rows:
grep `^| N |`).

**Closed 2026-10-14, ships in the next release** (#266, #267)
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

**Closed 2026-10-13, ships in the next release** (#257, #259, #260, #262, #263, #264)
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
  loaded into a split worker — #265.
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

**Closed 2026-10-03, after v0.3.223** (`docs/invariants/network.md` § "A node with no
origin to ask is healed by the holders that checked theirs")
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

**Closed 2026-10-03, after v0.3.222** (`docs/invariants/network.md` § "A node vouches only for
bytes that are the swarm's upload"; gotcha #782)
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
