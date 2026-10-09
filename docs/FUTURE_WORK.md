# Future Work

What is still wrong or unbuilt, and what was decided against. **Every entry here was
verified against the code on 2026-10-02** (v0.3.221-alpha), when this file was rebuilt from
the 1.2 MB, 16k-line document it had become — about 345 entries, of which roughly 150 had
long since shipped without their entries being updated.

## How to use this file

- **Numbers are stable.** An item keeps its `#NNN` for life; a new item takes the next free
  number (**next free: #245**). Numbers below #165 come from the old triage index; #165 and
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
#225, #226 and #227 opened from a tester's two reports the same day; #117 (Qwen 3.5, dense), #225 and #226 closed in v0.3.229 (released 2026-10-07 00:00 UTC), #228 opened for what Qwen 3.5 leaves out; #129's cold loads (waited for and priced), #228 (2) and (2b), #220 (2), #229, #230 and #231 shipped in v0.3.230 (released 2026-10-07 16:38 UTC), #232 parked from their report; #234, #235, #239 and #240 closed on 2026-10-08 from two tester reports and the card check after them, #236-#238 opened from the second (#236 and #238 closed the same day); #242 and #243 opened and closed the same day from reports #006 (a 1 Mbps cap stored for "Unlimited") and #007 (the narrow-screen model picker); #194 closed on 2026-10-09 (a card keeps its conversation memory as f16, and its memory guard under a context override), #237 re-ranked to P3 from what could be measured; #152 closed the same day (the guess-check stream on the boomerang).
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
3. **#159** — the remaining way a node can act on another upload's description of a model
   (#156's other door; #158 closed 2026-10-03; narrowed the same day to hashes gossiped by
   holders on v0.3.222 and older, and DHT-only holders).
4. **#1** — every Mac runs on the processor; no GPU backend is compiled for Apple Silicon.
5. **#220** — a worker's card faulted once (illegal memory access) and the driver took 14 minutes
   to reset; the kernel is unknown until a sanitizer run (needs the owner's administrator rights).

**P2 — speed and completeness**
6. *(#152 closed 2026-10-09 — the guess-check stream runs on the boomerang and with a peer
    holding the first layers, not only with a peer holding the last.)*
7. **#10** — conversation prefixes across computers: no routing to the peer holding the
    cache, and a split chain keeps no KV across turns (absorbs #139).
8. **#162** — the release gate: a cloned gate loses its helpers, and step 12e cannot test a
    takeover on this box (absorbs #140).
9. *(#155 narrowed to P3 on 2026-10-07 — not seen on the swarm, see its entry.)*
10. *(#194 closed 2026-10-09 — a card keeps its conversation memory as f16, a third of the
    memory, and keeps its memory guard under a context override.)*
11. **#147**, **#137**, **#138**, **#129** — prompt reading on the card, the processor half of
    a hybrid, MoE placement, and near-fit models sent across continents
    (`docs/plans/faster_than_local.md`).
12. **#3**, **#180** — the routing cost model charges a constant where the reply length
    belongs, which keeps partial ranges (load spreading) off.
13. **#171** — the prompt pass through a split runs one stage at a time.
14. **#228** — Qwen 3.5's first version leaves out its MoE models, the card/processor split and
    speculation.

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

#### #159 — A coordinator holding none of a model routes on placeholder part hashes
`P1` · routing · **PARTIAL** — 2026-10-02 · history: archive row #159

`register_for_fetching` registers the canonical manifest built from the header with zero part
hashes, so `expected_build_tag` is unknown and `shard_holders` cannot exclude holders of
another upload until a canonical holder's gossip fills the hashes
(`merge_known_shard_hashes`) — and the first hashes heard win.

**Narrowed 2026-10-03 (after v0.3.222):** a node on v0.3.223 or later withholds a copy of
another upload (no announcement, no manifest gossip, no serving) and refuses to load parts
whose bytes are not its table's upload, so neither the holder nor its hashes can be taken for
the swarm's; a part fetched against a hash that was another upload's is caught at accept by
the byte check. **Residual:** holders on v0.3.222 and older still gossip such hashes until they
update; a holder known only from a DHT provider record carries no build and is not judged.
Options for the rest: exclude holders whose tag is unknown when the canonical upload is known,
or prefer hashes from holders whose announced tags agree with each other.

#### #224 — A node that cannot reach HuggingFace sometimes never takes the checked holders' verdict
`P3` · heal · **OPEN** — 2026-10-06 · history: the v0.3.228 gate, step 12m

`examples/outvoted_rig.sh` with `HFLESS=proxy` (B behind a dead proxy, holding a zeroed part; A
and C checking theirs on HuggingFace) FAILED twice in the .228 gate, 15:35-16:05 local: B with
.228 and B with .227 (A and C .228 both times) — B logged `An upload of this model could not be
checked on HuggingFace` once and never `parts on this node differ from what the holders that
checked theirs … agree on`, in 900 s. The offline arm passed between them (281 s). Re-run at
16:33 with the live node stopped: all-.227 PASS (231 s), all-.228 PASS (250 s); every earlier
gate (.224-.227) passed it. So not .228's doing (only `auto_manage` storage/prune code changed,
none of it on this path), and not deterministic. Both failing runs had A and C announce their
checked copies 30 s after start, as the passing ones did. One difference seen: B's "Our copy of
this shard disagrees with the hash the swarm reports … asking the model's origin to settle it"
landed 14 ms BEFORE its background shard verification finished, where the passing run had it
after — a candidate race on `shards_pending_verification`, which `settle_by_checked_holders`
skips. Next: run the proxy arm ~10 times with B at `-v` and log why `settle_by_checked_holders`
returned (pending / no checked tags / `heard_hash_with_tag` None — the last is `debug!`).
**Recurred at the v0.3.231 gate (2026-10-08 21:21-21:53 local)**: proxy arm FAIL with B = .231
(900 s) and with B = .230 (A, C = .231; 600 s), the same signature (B's disagreement 339 ms
before its background verification finished; `could not be checked` once, never `differ from what
the holders`). Re-run with the live node stopped: all-.230 PASS (262 s), all-.231 PASS (263 s).
Eight gates have run it (.224 on); it failed at two (.228, .231), on both builds each time.

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
debugger interface, two DWORD 1 registry values set as administrator
(`HKLM\SYSTEM\CurrentControlSet\Services\nvlddmkm` `EnableDebugInterface`,
`HKLM\SOFTWARE\NVIDIA Corporation\GPUDebugger` `EnableInterface`), which is the owner's hands;
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
the fixed 2 s/layer guess. Depends on #3.

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
at processor speed rather than as the loader's card/processor split (above). **Fixed 2026-10-08 on
main (not yet released):** `PipelineScheduler::local_speed_off_the_card` blends the card's share
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
`P2` · network · **PARTIAL** — 2026-09-18 · history: archive row #91, `docs/invariants/network.md`

Trickle suppression (`MANIFEST_QUIET_WINDOW`), per-(model, hash) `manifest_heard`,
per-category counters and the region-demand split shipped in .200-.206 and halved it (42 →
~20 KB/s each way; it once extrapolated to ~159 GB/month — the complaint that makes people
uninstall a P2P app). Open: (1) re-measure on a fleet fully on .206+ (target ~0.02 manifest
messages/s; 0.046 measured with a mixed fleet) before any redesign; (2) a manifest without
its tensor table — blocked on an integrity order: `compute_hash` covers the table and
`acquisition.rs` calls `verify_hash_strict` before fetching a header, so it must become
fetch header → derive → verify, feature-bit gated, omitted only when every connected peer
advertises it; (3) a both-direction traffic ceiling at the transport (risky — throttling
gossip can partition a node); `max_bandwidth_mbps` shapes part serving only. A
`NodeCapabilityUpdate` change-gate was deliberately skipped (3.4% of bytes).

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

#### #147 — Prompt reading on the card leaves ~20% on the table: f16 accumulation is still opt-in
`P2` · perf-local · **PARTIAL** — 2026-09-29 · history: archive row #147 and § "Prompt reading on the card (#147, 2026-09-29)"

Shipped in v0.3.213 (the archive says "not released"): ≥ 64 rows go to f16 dequant + cuBLAS
(llama.cpp's own rule for a dp4a MMQ) and a prompt alone on the card reads 512 per forward —
Llama-3.1-8B ~740 → ~1,150 tok/s. `SWARMLLM_QMATMUL_CUBLAS_ACC=16` (f16 accumulation,
llama.cpp's default) gives ~1,400 and scored the same, but stays opt-in until a card-side
family check at LONG context (4-8K: Llama, Qwen2.5/3, Gemma-2, Phi, Mistral, GLM, and MoE
attention) scored with `examples/score_against_reference.py` — the release gate's family
check runs on the processor and cannot see this path. Read which ops llama.cpp guards with
`GGML_PREC_F32` first. Smaller items in the archive body: share the f16 activation cast
across q/k/v and gate/up; remove 8 `ucopy_f32` per layer in the prompt pass; one untraced
single-row `mul_mat_vec_q4_K` per layer; an int8 tensor-core MMQ port (large). Separately,
GLM-4-9B disagrees with llama.cpp at a few positions on every path (tokenizer
re-tokenisation is the lead) — pre-existing.

#### #137 — A model split between card and processor reads its processor layers at ~17 GB/s, a third of what the memory can do
`P2` · perf-local · **OPEN** — 2026-09-27 · history: archive row #137; plan: `docs/plans/faster_than_local.md` § 3.3

Hits anyone whose model is slightly too large for their card (a 14B at 3.35 tok/s). First
measure the owner's decode on a hybrid worker at the contribution width
(`cpu_pools::in_phase_pool` widens prompt reading only; half the cores at Minimal), then
compare our quantized matvec against llama.cpp's.

#### #138 — A mixture-of-experts model that does not fit the card is placed by whole layers
`P2` · perf-local · **OPEN** — 2026-09-27 · history: archive row #138; plan: `docs/plans/faster_than_local.md` § 3.2

llama.cpp keeps attention and the router on the card and the experts in RAM, and reaches
32-44 tok/s on Qwen3-30B-A3B-class models on consumer cards; we place whole layers, so most
of the model runs on the processor. Per-tensor placement for MoE layers beside
`split::hybrid::LayerPlacement`; experts already load quantized one by one
(`load_moe_ffn`). Depends on a real MoE model having run at all (#114).

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

#### #244 — A third streamed guess-check request in a row runs at the old speed
`P3` · perf-split · **OPEN** — 2026-10-08 · history: the v0.3.231 release check after #233

After #233 the drafter keeps guessing on every request (final_gamma 1, not 0). But in the 12h
shape (7B split A/B on one card, 24 ms emulated link, 0.5B q8 drafter on A, streamed checks), the
third request of each run reads 17.8-21.5 tok/s, while the first two read 24.5-27.6. The old-rule
arm reads 18.4-21.7 for all three. Measured on the downloaded v0.3.231 build, one binary, arms
old / weigh / fixed ×2 (`~/swarmllm-gate-0231/h12_post.sh`, Windows up 27 h, 0 driver events).
The same rig on a main build at 18 h uptime that morning (`~/swarmllm-bisect233/h12_three.sh`) had
all three fast (25.3-28.4). Not the graph rest: in run 12 the rest began 0.8 s into request 3 on
the count rule ("no ordinary step timed yet") and was ended 0.19 s later by the weighing (graph
4.1 ms vs ordinary 13.1 ms). Accepted guesses are identical across the three requests (139/81), so
the time per step changed, not the guessing. Next: per-step card timings for request 3 vs 2
(draft and check `_ms` in the DSD line), and the same rig at low uptime, to separate machine state
from code.

#### #10 — Conversation prefixes across computers: no routing to the peer holding the cache, no cache in a split chain
`P2` · perf-split · **PARTIAL** — 2026-09-07 · absorbs #139 and archive § "Speeding up inference BETWEEN nodes" idea 1 (prefix-keyed remote KV); history: archive row #10 and § "A conversation's later turns do not seek out the peer holding its prefix"

The LOCAL half shipped in v0.3.208 (`scheduler::cached_prefix`; `split_rig.sh cache` HIT
2,713 of 2,744 tokens). Three gaps, the largest single win left for agents (long, repeated
prompts — one report spent 847 s re-reading):
- **Route to the peer that holds it.** Peers' caches are neither gossiped nor credited.
  Gossip prefix digests, keep caches longer than the 10-minute KV idle expiry with a RAM/disk
  tier, and price a CPU requester's long prompt against a GPU peer's rate
  (`faster_than_local.md` § 3.1). Delivery trap: `try_ngram_only_distributed` runs before
  `try_remote_generate_fastpath` and takes a remote single segment, so a priced credit would
  not be delivered — make the n-gram path decline a plan priced warm. The credit must require
  `cross_node_prefix_trust_min` and `share_prefix_cache_with_peers` (default off).
- **A split chain keeps no KV across turns of a stateless client.** Every agent turn
  re-ships and re-reads the whole prompt through every segment (~79 MB per hop and ~50 s of
  wire per turn at 25 Mbps on a 14B). The coordinator already computes the block-hash chain
  (`prefix_cache::compute_block_hashes`); ship it as an opaque prefix id with the prompt pass,
  let each segment store its range under it and answer "held to position k", and send only
  the delta on turn 2+. An additive trailer gated at the sender on a new feature bit, with an
  LRU / byte cap per segment. With a delegated split (#143) the delegate owns the id.
- The Anthropic surface cannot reach the session-id design without a signature change
  through four handlers.

#### #171 — The prompt pass through a split runs one stage at a time
`P2` · perf-split · **OPEN** — 2026-08-24 · absorbs archive § "Speeding up inference BETWEEN nodes" idea 2 (prefill microbatching); history: archive § "The pipeline is idle (N-1)/N of the time during the phase that dominates a long request"

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

Dense Qwen 3.5 runs since #117. Still out: (1) `qwen35moe` stays refused until checked against a
real file (HF: `Flexan/kshitijthakkar-qwen3.5-moe-0.87B-d0.8B-GGUF`); (2) ~~the card/processor
split~~ — **released in v0.3.230 (2026-10-07)**: dense Qwen 3.5 is on `arch_supports_hybrid`,
checked split on the card against llama.cpp (`docs/invariants/inference.md` § "A card/processor
split is placed in every per-layer loop"). Its sizing followed the same day:
admission, the split planner and the loader's KV budget charged every layer an attention
layer's KV while three in four are DeltaNet with a fixed state — ~805 MB too much for the 9B at
the admission context, the margin by which it missed an 8 GB card. All three, and the
scheduler's bound on a peer (`kv_bytes_per_position_per_layer`, from the manifest's tensor
table), now ask `split::layers_keeping_kv` (the real 9B header: attention at layers 3, 7, …, 31, llama.cpp's
rule). Still uncharged: each DeltaNet layer's per-request state (~2 MB a layer for the 9B,
covered by the forward-buffer reserve); (3) the delta rule runs token by token — correct, slow for a long prompt on a card
(llama.cpp's `build_delta_net_chunking` is the reference); (4) decode is never captured as a CUDA
graph ("a layer type the capture has not been checked on"); (5) no speculation of any kind — a
snapshot of the recurrent state per draft would allow it; (6) ~~the 4B's gap to llama.cpp~~ —
checked 2026-10-07, not a defect: six prompts scored on the processor against llama.cpp master
(`score_ids_dump.py`, margin 0.05) — Qwen 3.5 4B Q4_K_M misses on 3 (gaps 0.11-0.21), and
Llama-3.2-3B Q4_K_M, a long-verified family, misses on 1 by MORE (0.306, "why the sky is blue").
Unquantized, the 0.8B matches llama.cpp exactly (worst cos 0.999999), at Q8_0 within 0.012: the
model's arithmetic is right and 4-bit near-ties fall the other way through our kernels' rounding. Re-check with
`~/llama.cpp-ref/{dump_logits,ref_generate}` (llama-cpp-python 0.3.16 cannot load `qwen35`), and
render a new template with jinja2 first (gotcha #798).

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

#### #69 — Some panels still read a failed fetch as "there is nothing here"
`P3` · ux · **PARTIAL** — 2026-09-14 · history: archive row #69

`App.data.loadReachedDaemon(key)` gates cache writes (guard
`a_failed_fetch_never_overwrites_the_frontend_cache`), and the three consumers that ACTED on a
failed fetch are fixed. The other `load*` consumers still show their ordinary empty state when
the daemon is unreachable or answers 401/500. Each needs its own honest wording through
`loadReachedDaemon` — new i18n keys in all 21 locales.

#### #176 — Cross-pool inference can only be switched on by editing the config file
`P3` · ux · **PARTIAL** — 2026-07 (R134) · history: archive § "Inter-pool model sharing policy"

Discovery and opt-in routing between pools shipped (R134-R134.7:
`PoolModelAvailability`, `foreign_pool_catalog`, `pool.allow_cross_pool_inference`); no
dashboard control or consent banner exists. Add the toggle with its consent wording (i18n ×21).
Billing between pools waits on credits, which are dormant (`docs/CREDITS_DESIGN.md`).

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

#### #162 — The release gate: a cloned gate loses its helpers, and step 12e cannot test a takeover on this box
`P2` · process · **OPEN** — 2026-10-02 · absorbs #140; history: archive rows #162 and #140

(a) `gate220.sh` was made from `gate219.sh` with paths rewritten; its helpers (`kv104.sh`,
`lora_gate.sh`, `kv121.sh`, `guest.sh`, `repro767.sh`, `probe.py`) were not copied, so five
steps printed `No such file` and moved on, and the gate exited 0 (they ran later in a
supplement, all PASS). Write `new_gate.sh <from> <to>` that copies every `$G/*.sh|py` the
script references and fails if one is missing; make a step that cannot find its helper FAIL;
`safety_start` must name the new gate. (b) Step 12e (failover + guessing ahead) fails on ANY
build on this box once B's worker loads: the four rig nodes share one 16 GB machine and read
their room from its free RAM, so C and D advertise `max_hostable_layers` 28 for the first
request and 23 for the second, while covering B's range as a pair needs 24 — no standby, a
retry instead of a takeover (re-observed at the v0.3.219 gate; .218 behaves the same, so it is
no regression). Give C/D a fixed `ram_model_budget_mb` in `EXTRA_TOML` so 12e tests the
takeover again. Until then 12e is not a product signal (`memory/open_cautions.md`).

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

**Closed 2026-10-09, on main (not yet released)** (#194, #152)
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
