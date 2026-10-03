# Future Work

What is still wrong or unbuilt, and what was decided against. **Every entry here was
verified against the code on 2026-10-02** (v0.3.221-alpha), when this file was rebuilt from
the 1.2 MB, 16k-line document it had become — about 345 entries, of which roughly 150 had
long since shipped without their entries being updated.

## How to use this file

- **Numbers are stable.** An item keeps its `#NNN` for life; a new item takes the next free
  number (**next free: #215**). Numbers below #165 come from the old triage index; #165 and
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

Re-ranked 2026-10-02 after the verification, and 2026-10-03 when #156, #158 and #213 closed.
`docs/plans/` holds the multi-step designs; the entries point at them.

**P0 — wrong answers, silently**
1. *(none open: #156 closed 2026-10-03 — a mixed copy is refused at load.)*
2. **Watch the live swarm converge (no code)**: `peers_other_build` → 0 per model once peers
   run the release after v0.3.221 — on .221 it did NOT (12 holdings on 3 peers after 13 h,
   #213). A peer still counted a day after updating is one that cannot switch (disk,
   HuggingFace, offline mode); `docs/DIAGNOSTICS.md` § "Which copy of this model does this
   node hold?".

**P1 — silent, or broken for a whole class of users**
3. **#159** — the remaining way a node can act on another upload's description of a model
   (#156's other door; #158 closed 2026-10-03).
4. **#165** — prompt privacy is on by default for every node holding both ends of a model
   (58 of 78 holdings in the 2026-10-01 census) and costs 9-14× on a far middle peer; the
   notice that was meant to tell the user never fires.
5. **#117** — Qwen 3.5: a working dense implementation sits unmerged on branch
   `qwen35-support`; the most-downloaded family the swarm refuses.
6. **#1** — every Mac runs on the processor; no GPU backend is compiled for Apple Silicon.
7. **#153** — Windows: a worker that outlives the daemon holds its port, and the next start
   fails.
8. **#164** — .deb installs from 2026-07-28 to 2026-10-02 run with batching off; the release
   note ships with the next release, and the project's own Proxmox node needs the line removed.

**P2 — speed and completeness**
9. **#189** — `/v1/models` reports no context length for any model this node does not hold
   (agents such as OpenClaw then guess 128k). Small, additive.
10. **#152** — the continuous guess-check stream never runs on the split shapes the swarm
    actually makes (the boomerang above all).
11. **#10** — conversation prefixes across computers: no routing to the peer holding the
    cache, and a split chain keeps no KV across turns (absorbs #139).
12. **#162** — the release gate: a cloned gate loses its helpers, and step 12e cannot test a
    takeover on this box (absorbs #140).
13. **#155** — AutoNAT dial-backs denied by the per-peer cap every ~5 s, for the life of a node.
14. **#157** — a node switching copies fetches from HuggingFace even when peers hold the right one.
15. **#194** — an agent-sized prompt fills an 8 GB card with conversation memory.
16. **#147**, **#137**, **#138**, **#129** — prompt reading on the card, the processor half of
    a hybrid, MoE placement, and near-fit models sent across continents
    (`docs/plans/faster_than_local.md`).
17. **#3**, **#180** — the routing cost model charges a constant where the reply length
    belongs, which keeps partial ranges (load spreading) off.
18. **#171** — the prompt pass through a split runs one stage at a time.

Everything else is ranked in its own entry. **P3** is narrow or cosmetic, **P4** is process,
maintenance or an idea with no user waiting on it.

## Open items

### Wrong answers and the one-upload-per-model rule

#### #159 — A coordinator holding none of a model routes on placeholder part hashes
`P1` · routing · **OPEN** — 2026-10-02 · history: archive row #159

`register_for_fetching` registers the canonical manifest built from the header with zero part
hashes, so `expected_build_tag` is unknown and `shard_holders` cannot exclude holders of
another upload until a canonical holder's gossip fills the hashes
(`merge_known_shard_hashes`). Known limit of #151's first cut (`docs/invariants/network.md` §
"One upload per model id"). Options: take part hashes from the first canonical-SHAPED
manifest a peer gossips (shape is already the adoption test), or exclude holders whose tag is
unknown when the canonical upload is known.

#### #157 — A node switching copies fetches from HuggingFace even when peers hold the canonical upload
`P2` · storage · **OPEN** — 2026-10-02 · history: archive row #157

`switch_to` stages parts with `download_shard` (HF byte ranges) into
`<data_dir>/canonical/<model>`; the P2P transfer writes into the model directory and is
accepted against the registered manifest, which during a switch is the OLD upload's. Needs
the P2P path to take a destination and a manifest. Costs HF bandwidth and fails when HF is
unreachable (measured: Qwen2.5-Coder-7B, 8 parts / 4.4 GB, switched in 56 s from HF).
Prerequisite of #160.

#### #160 — A node in offline mode never verifies or switches its copies
`P3` · storage · **OPEN** — 2026-10-02 · history: archive row #160

`auto_manage::canonical` needs HuggingFace (anonymous probe, byte check, staged fetch). An
offline node could still adopt the canonical choice from peers' claims and fetch canonical
parts over P2P against a canonical-shaped manifest — after #157.

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

#### #153 — Windows: every child the daemon starts inherits its QUIC socket, model workers included
`P1` · reliability · **OPEN** — 2026-10-01 · history: archive row #153

`Command` on Windows calls `CreateProcessW` with `bInheritHandles = TRUE`, and the QUIC
listener socket turned out inheritable although libp2p-quic creates it through
`socket2::Socket::new` (which asks for `WSA_FLAG_NO_HANDLE_INHERIT`) — measured: an updated
replacement held UDP 8950 until it was killed (`docs/DIAGNOSTICS.md` § "A Windows node that
never updates"). A worker that outlives the daemon (crash, kill) holds the node's port, and
the next start fails on "already in use" until it exits. Only the update hand-off uses
`update_restart::spawn_without_inherited_handles` today.
(a) Start workers through it (it restricts inheritance to stdio — check first that the worker
IPC depends on no other inherited handle in `process_pool`'s spawn); (b) find WHY the socket
is inheritable (`GetHandleInformation` on the socket after `listen_on`, on a real Windows
node). Test on Windows with the MinGW build (`memory/env_windows_test_node.md`), never by
reading; a reproduction must fail on the broken build first.

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

#### #150 — The daemon's worker reader cannot skip a message it does not know
`P3` · reliability · **OPEN** — 2026-10-01 · history: archive row #150, gotcha #765

`worker_ipc::recv_framed` decodes the JSON header BEFORE reading the payload length, so an
unknown `WorkerMsg` leaves the stream misaligned and `reader_actor` evicts the worker. A
worker is spawned from the binary on disk now (gotcha #188), so after an update not yet
restarted into, a NEWER worker talks to an OLDER daemon. Owed for daemons from .215 on: read
the whole frame before decoding, and skip — with one warning — a SIDE-BAND variant that fails
to decode as unknown, while a response-carrying one stays fatal and must wake its waiter.
Settle first how to tell unknown from known-but-malformed without matching serde's error text
(#295): pin the tag list with a test that serialises every variant through an exhaustive
match. Until then **gate every new worker→daemon message at the sender** on something only a
new-enough daemon sets (as `CardAllocationProbe` is, `SWARMLLM_DAEMON_READS_CARD_PROBE`).

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
The stalls' root cause stays unknown and the guard does not need it.

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

#### #165 — Auto-enabled prompt privacy can cost 9-14× on a far middle peer, and the notice meant to say so never fires
`P1` · routing · **OPEN** — 2026-09-05, reopened 2026-09-20 · absorbs the privacy bullet of archive § "Not bugs, and deliberately not ranked" and residual (1) of § "The whole-model hand-off is a yes/no gate…"; history: archive § "Auto-enabled prompt privacy can cost 6x on a long prompt"

`encrypted_pipeline_auto` (default ON) keeps the first and last layers of a model on any node
holding both ends — "Start and finish on this computer", which is STRUCTURAL, not
cryptographic (CLAUDE.md). The cost is one round trip per token to the middle peer: measured
2026-09-20 at 9× and 14× against handing the model over, 6× on a long prompt's reading.
The policy is decided and stays: **tell the user, never downgrade automatically** — a
downgrade triggered by slowness is one an adversary triggers by being slow (RFC 7507's
reason for TLS); the figure is reported, never read by the router. What is broken is the
telling: `report_privacy_cost`'s one call site is gated on `!search_will_decide`, and with
`parallax_routing` on (default) and more than one candidate it is never reached — it fires
only for single-candidate models. Fix: price the route the search actually chose against the
cheapest plan without the boomerang (the search already prices both with `vertex_cost`) and
report from there, keeping the existing rate limit (once per model per 10 min) and wording.
This is wiring, not a policy change.

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
`P2` · routing · **OPEN** — 2026-09-27 · history: archive row #129

Instead of running a layer or two on the processor locally (a hybrid placement), the planner
picks an intercontinental boomerang. #127 (pricing our own GPU from a cold load) is fixed, so
first re-measure. Then price the local hybrid placement (`partial_gpu_layers`) as a candidate
in the search, and charge a remote hop's network cost per token / per crossing rather than per
segment (`docs/plans/regional_pipelines.md` § "What is actually missing", 2). Do not tune
`ASSUMED_FORWARD_PASSES` (#3).

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
`P2` · network · **OPEN** — 2026-10-02 · history: archive row #155, gotcha #774

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

#### #152 — The continuous guess-check stream never runs on the split shapes the swarm actually makes
`P2` · perf-split · **OPEN** — measured 2026-10-01 · history: archive row #152

`dsd_stream::stream_tail` needs the plan to be [this node's segments…, ONE remote tail]. A node
holding both ends gets the boomerang (#165) — tail local, far node in the middle; a node
holding nothing has a remote head. Both fall back to rounds. Census 2026-10-01: 58 of 78
holdings hold both ends, 11 middle only, 5 last only, **4 first-only** (the only streaming
shape). Design, not a tweak: in the boomerang the far node returns hidden states per chunk
and the walk happens HERE, so the stream's chunk numbering and restart (`forward_streams`,
`truncate_kv_to`) must carry a middle segment's forwards, and the coordinator runs tail + walk
per chunk and restarts the middle on a miss; `peer_walks_at_tail` no longer applies. Rank the
remote-head shape after it. Write the design into `split_speculation.md` § 4b first; measure
on the TH↔IT link with `~/swarmllm-wan-1001/ab149.sh`'s method (one binary, env switch).

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
(W stays 3; W=6 read −7% / +5% / +18%); (b) cancelling a stale chunk the far node has already
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

A family in `supported_list` is a claim — check it against a REAL file's header (#715), and
never flip `ModelArch::is_supported` without a real-file comparison against llama.cpp
(`examples/logits_reference_probe.rs` + `compare_logits_reference.py`; replies with
`score_against_reference.py`).

#### #117 — Qwen 3.5 is refused on main; a working dense implementation sits unmerged on a branch
`P1` · model-support · **PARTIAL** — 2026-09-25 · history: archive row #117; plan: `docs/plans/qwen35_support.md`

Among the most-downloaded GGUFs of 2026 (Qwen3.5-4B/9B), and refused today
(`model_arch.rs` refuses `qwen35` and `qwen35moe`). Branch `qwen35-support` (local, 2 commits on
merge-base `aa2eac32`, last 2026-09-26): `824489b5` rewrites dense Qwen 3.5 against llama.cpp
("model math verified, serving path not yet safe"; CUDA 0.8B passes) and `b5ced886` refuses
speculation on a model with recurrent state and refuses truncation while such state is held.
To merge: finish the serving path for recurrent state (no speculation, no KV truncation or
prefix reuse, split-boundary handling, session expiry), rebase onto current main, verify the
0.8B and a 4B against llama.cpp master (`~/llama.cpp-ref/dump_logits` +
`compare_f32_logits.py`), then admit dense `qwen35`; `qwen35moe` stays refused. After it lands:
add Qwen 3.5 to `hybrid::arch_supports_hybrid` (its `q35_cos`/`q35_sin` tables must follow
each layer's device and its state buffers be checked), or a card too small for the model
loses the card entirely.

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

#### #194 — An agent-sized prompt fills an 8 GB card with conversation memory
`P2` · memory · **PARTIAL** — 2026-09-02 · absorbs the old survey's Tier 2F (KV quantisation); history: archive § "An agent-sized prompt fills an 8 GB card with KV cache, and decode crawls"

The crawl's cause is fixed (gotcha #440: `kv_budget::admit_prompt` + `SplitModel::kv_budget_now`
reconcile with the live card). But a token of cache costs ~344 KB (f32 plus the f16 mirror,
`split/kv_budget.rs`), so a 14k-token agent prompt takes ~5 GB beside the weights and ends in
refusals or a 503 to a peer instead of a warm turn. Options, in order: (1)
`inference.kv_cache_dtype = "f16"` with f32 the default, measuring divergence with the
split-model tests (arXiv 2604.15409's caution); (4) a hybrid placement that keeps the cache on
the processor; (2) depends on `cuda_decode_prefers_standard`. Q8_0 KV (group 32, ~2×) only
with the dequant fused into `kernels/decode_attn.cu` — outside it is likely a net loss — and it
breaks prefix-cache binary compatibility; never KIVI. A refusal printing `live_entries=1` with
a total well above that request's own cache would reopen #11.

#### #187 — Head-room admission prices load and runtime on two different bases
`P3` · memory · **PARTIAL** — 2026-08-08 · history: archive § "Head-room admission: two things the live test found"

The claim arithmetic is fixed (`kv_budget::positions_to_allocate`). `kv_budget.rs` bases runtime
head-room on free card memory (`mem_get_info`, `kv_headroom_bytes`) while the load-time estimator uses the
contribution-derived budget; nobody tried to reconcile them, and no end-to-end refusal under
real pressure was ever constructed. Put both on one number, then occupy card memory before a
load so the estimator passes but the runtime budget binds, and watch the refusal. Narrow
(multi-model or another program on the card); the guard is a backstop.

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

#### #214 — Qwen2.5-Coder-7B on Qwen's official upload answers a tool request in a wrapper nobody parses
`P2` · api · **OPEN** — 2026-10-03 (the v0.3.222 gate)

Since v0.3.221 moved every node to Qwen's official upload of `qwen2.5-coder-7b-instruct-q4-k-m`
(4,683,073,536 B, chat template 2,509 chars, with a `tools` branch), conformance's tool check
("Call get_time for zone UTC") gets `<{{"name": "get_time", "arguments": {"zone": "UTC"}}}}` back
as content: no `tool_calls`. The previous (third-party) upload's reply parsed. Not a build
regression — v0.3.221 and v0.3.222 answer identically on the current files (`~/swarmllm-gate-0222/
qconf_{221,222}.log`), and on a plain prompt .222's reply equals llama.cpp's on the official file
96/96 tokens. Next: render the same tool prompt through llama.cpp (llama-cpp-python, the GGUF's
own template with `tools`) and compare its greedy reply — if llama.cpp produces the same wrapper,
it is the model's habit and `tool_parse` may accept a `<{…}>`-wrapped JSON call (as it accepts
other near-misses, `docs/invariants/api-surfaces.md`); if not, the prompt differs and that is the
bug. Conformance's baseline (`~/swarmllm-gate-0221/conformance.log`) predates the switch — the
next gate diffs against `~/swarmllm-gate-0222/conformance.log`.

#### #189 — `/v1/models` reports no context length for a model this node does not hold
`P2` · api · **OPEN** — 2026-08-12 · history: archive § "`max_model_len` is unknown for network-only models"

`max_model_len_for` (`api/openai/mod.rs`) reads only the local `gguf_header.bin`, and
`ModelManifest` carries no context length, so every network-only model — the swarm's main
use — reports `max_model_len: null`. OpenClaw then assumes 128k and its agent turns are refused
as too long unless `contextWindow` is set by hand. Add `context_length: Option<u32>` with
`#[serde(default)]` to `ModelManifest` (additive, no version bump), fill it where the header is
read (`daemon::manifest`, `huggingface::probe`), and fall back to it in `max_model_len_for` /
`ModelInfo::new`. Mixed-version swarms stay null, so the API must keep tolerating it.

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

#### #164 — .deb installs made between 2026-07-28 and 2026-10-02 run with batching off
`P1` · packaging · **PARTIAL** — 2026-10-02 · history: archive row #164

Since `b29f183a` (2026-07-28) `packaging/deb/postinst` copies `/etc/swarmllm/default.toml` into a
NEW install's `/var/lib/swarmllm/config.toml`, and that template said `max_batch_size = 1` until
2026-10-02 — so those installs kept batching off when `f82e4199` (2026-08-21) made 8 the
default ("about 40% more work from the same card"). Silent. The template is fixed for new
installs (`aa1f75c8`); the release note is drafted in CHANGELOG `[Unreleased]` and ships with
the next release. (1) The project's own Proxmox node is such an install: at its next deploy read
`/var/lib/swarmllm/config.toml` and delete a `max_batch_size = 1` nobody chose. (2) Not added to
`config::migrate_superseded_defaults`: its rule admits only a value nobody could have chosen on
purpose, and `max_batch_size = 1` is a documented way to turn batching off. A migration keyed
on that section still matching the old template byte for byte (proof it was never edited)
would be the edit-proof addition.

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

#### A disputed shard is kept but the disagreement is never settled (#61)
`P3` · storage · **PARKED: no field reading yet** — 2026-09-13 · history: archive row #61 and § "A disputed shard is kept but the disagreement is never settled"

Fallout of #60 and strictly better than what it replaced (deleting good data): a part that
disagrees with an unbacked hash is kept, served and recorded in `disputed_shards` (guard
`every_path_that_keeps_disagreeing_bytes_records_the_dispute`; a "Disagrees" badge and a
diagnostics line), but nothing settles it. Do not build before a reading: look for `-- shards
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
  against HuggingFace every 2 minutes. Found live 13 h after v0.3.221: 12 holdings on 3 peers
  never converged (gotcha #780). `SwitchQueue`: only a switch that fetched holds the turn; a
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
