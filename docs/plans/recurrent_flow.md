# A tree that flows through recurrent layers (Qwen 3.5 / 3.6 / 3.8) — plan, 2026-11-22

Status: stages 1-4 BUILT (2026-11-22: the layer on the processor and on a card, the worker and the wire, the
recurrent drafter's own growth, the coordinator's gates and walk, with tests) and RELEASED in v0.3.306 — every node on it
reads the walk (feature bit 47). Stage 5 MEASURED on the real link (2026-11-22, § Stage 5 as measured: the 9B split
1.86× prose, 3.4× code): the computer driving a reply now flows by default (`SWARMLLM_RECURRENT_FLOW=0` is the A/B),
and the planner and the download pass leave room for and fetch the flow's drafter. On main, not in a release yet.
Figures are MEASURED (our records), READ (a header or source) or DERIVED.

## Why

The two most-asked models of the swarm, Qwen3.8-27B and Qwen3.6-35B-A3B (`memory/model_popularity_1108.md`), are
the `qwen35` / `qwen35moe` families: three of every four layers are Gated DeltaNet, a recurrent state a request.
Every drafter path is closed to them (`pipeline::distributed::speculation_can_roll_back`), because their state can
be taken back only into the LAST check (`SsmState::rewind_to`, `docs/invariants/inference.md` § "A recurrent state
is taken back into its last check"). So a split of these models checks one block at a time and waits a whole round
trip for each verdict:

- MEASURED 2026-11-22 (`real_link27e.sh`): Qwen3.8-27B, this card + AU's RTX 3060 (128 ms), DFlash 2 keeping 4.04
  tokens a code judgement: 5.4-5.9 tok/s; a round 468 ms median (AU's own time 310 ms, the link 157 ms).
- The flow (`dsd_flow`, #134 (b)) is what took the 14B past the bar on the same link: code 42-48, prose ~19 tok/s
  with a card drafter (`docs/plans/standing_ring.md` § 7), against 7.2 prose for the drafter on the processor.
  It keeps a dozen forwards in flight; the round trip stops setting the pace.

The user's bar is 20-30 tok/s for a split reply (2026-11-20). A 27B split cannot reach it one check at a time over
any WAN round trip; with a flow it has the 14B's shape. This plan is the flow for recurrent models.

## What a flow asks of a recurrent layer

A flow grows ONE tree, a few rows a forward (`levels_a_call` × `rows_a_level` 8 — two levels for a recurrent
target since 2026-11-23, three for another family, `dsd_flow::levels_by_default`), each forward's rows the tree's
LAST (`LayerForward::draft_tree_from`; `draft_parents` names every row from the tree's base), while the coordinator
walks the draws that come back (`FlowTree::walk`). Rows from earlier forwards stay in each segment's cache; a
forward's rows have their PARENTS in earlier forwards. Lost branches stay until a rebase (≤ `MAX_TREE_ROWS` 512).

A DeltaNet row needs its parent's state and its line's last `kernel − 1` (3) convolution inputs. Today a layer reads
a tree only within ONE forward (#437: `qwen35::tree_parents_now`, the parents among the step's own rows), and a
tree's state is marked branched until the verdict rewinds into that same step.

## Prior art (looked up 2026-11-22; sonnet agent, second-hand through fetches where noted)

- **vLLM** (`v1/attention/backends/gdn_attn.py`): `num_spec + 1` state slots a request, a chain only; the accepted
  count picks the slot. One check in flight.
- **SGLang** (PyTorch blog "Hybrid models meet SGLang"): a state slot a draft token; trees by parent index (each
  node's state from its parent's); Qwen3-Next-80B, topk 4 / 8 drafts, acceptance 4.23. One check in flight.
- **llama.cpp** (#22400 in #22673): the last `n_rs_seq` per-token states (~500 MB at 3 drafts), else a checkpoint and
  a re-run (#19493); n_parallel = 1; nothing in flight, no trees.
- **Dao AI Lab's ReplaySSM** (2026): per-draft state snapshots cost 4.85× a decode step at six guesses on
  Qwen3.5-122B; cache each step's INPUTS and replay from one checkpoint. Our one-check rewind is this idea (#228 (5)).
- **STree** (arXiv 2505.14969), **TreeWY** (2608.20961, Gated DeltaNet / Qwen3.5), **Twig** (2608.01651): trees
  for state-space layers without a state per node — per-row activations, the accepted state rebuilt from the
  accepted path. All within one verification step.

**None keeps several checks in flight on a recurrent model.** Every published system verifies a tree and commits
before the next is sent; a flow's forwards build on rows whose verdict has not come back, across a WAN. The
per-node state they keep (SGLang, vLLM) is 3 MiB a layer a row on the 27B (READ: 48 value heads × 128 × 128 × f32)
— a flow's ~150 live rows would be ~450 MiB A LAYER. What transfers: inputs, not states (ReplaySSM, STree); a
state rebuilt only along the path that needs it (TreeWY).

## The design

### One checkpoint, and the inputs of every row that may still matter

Each DeltaNet layer of a flowing request keeps:

- **The anchor**: the state (convolution window + delta-rule state) after the row the WALK stands on — the last row
  whose token is the reply's. Everything before it is final.
- **The base**: the state at the tree's start, for a rebase (which cuts every cache to the base and reads the path
  again, `FlowTree::rebase`).
- **The inputs of every LIVE row** (the walk's row and every row following it — `FlowTree`'s own definition) and of
  the path from the anchor to the walk's row: the convolution's inputs, k, v, g and β (q only for this forward's own
  rows). READ: ~80 KB a row a layer on the 27B (10,240 convolution channels + 2,048 + 6,144 + 2 × 48), ~48 KB on
  the 0.8B.

A forward's new rows each start from their parent's state. That state is rebuilt from the anchor along the parent's
line, **in one depth-first pass over the union of the new rows' lines** — each row of the union updated once, a stack
of states down the current branch (on a card: a warp's row of each state on the stack in shared memory; the 27B's
128-float state row is 512 B, a 32-deep stack 16 KB a warp). The rows on those lines are already-computed inputs:
nothing is projected or attended again, as in today's take-back.

**What the walk's position buys.** Without it, the anchor stays at the base, every row since the base is kept (512
rows × 80 KB × 24 layers ≈ 1 GB on a 32-layer share of the 27B), and each new row's line runs up to 128 rows deep.
With it: ~150 rows (~0.3 GB on that share), lines as deep as the forwards in flight (DERIVED: a dozen forwards ×
≤ 3 levels, minus the walk's advance; ~10-30), so a forward's union is ~50 row updates a layer against a tree
check's 24 today.

**The commit**: when the walk advances, the anchor is replayed forward along the walk's path (the rows between
the old and new walk row — about one forward's kept tokens) and every row not following the new walk row is dropped.
The replay goes through the one delta-rule kernel a step uses, as today's rewind, so the anchor is bit for bit the
state a chain read of the reply would reach.

### Who tells a segment where the walk stands

- **A peer segment**: a new forward trailer, the walk's row when the forward was sent (`u16`, after `0x14`,
  AAD-bound), gated at the SENDER on a new feature bit `RECURRENT_FLOW` that a node advertises only where its
  loaded recurrent layers can read a flowing tree. A forward on a recurrent segment without it is refused
  (`Validation`, before running anything), as a tree is today.
- **The drafter** (an engine drafter of the same family: Qwen3.5-0.8B shares Qwen3.8-27B's 248,320 tokens,
  merges, types and end id — READ from both headers 2026-11-22): it grows the flow's tree itself
  (`FlowGrowth`) and applies the same `FlowOp::Advance`, so its worker is told over IPC. Its levels are several-row
  forwards (`split::FlowStep::levels`), the same mechanism as the target's.

### What opens, and only there

- `speculation_can_roll_back` stays false for every path but the flow; the flow is opened for a recurrent target
  only where every peer segment advertises `RECURRENT_FLOW`, the drafter's worker reads a flowing tree, and every
  segment and the drafter run on a CARD (the flow's existing rule, `dsd_tree::trees_wanted`).
- `engine_drafter` admits a recurrent drafter only for the flow.
- A failed flow finishes in tree rounds today (`rounds_may_finish`); for a recurrent model it finishes one check at a
  time (as built: a token a round, § Stage 4 as built — the n-gram loop cannot take a reply over part-way).

## Stages and their kill tests

1. **The layer, on the processor** (`layers/mod.rs` `SsmState`, `layers/qwen35.rs`): an anchor, a base, a store of
   rows with parents across forwards; a tree step whose parents may be rows of earlier forwards; advance and
   rebase. Test: a flowing tree grown over four forwards, with walk advances and a rebase between them — every row's
   output and the anchor equal to its line read alone as a chain, 0.0 apart (the projections exact, as the #437
   tests); red with the parent read from the anchor, with the union replayed in row order, with a dropped row read.
2. **The layer, on a card**: the delta rule's union pass with a state stack, the convolution along lines that leave
   the forward; within 1e-4 of the processor, every row, the anchor after advances (RTX 3070).
3. **The worker and the wire**: `draft_tree_to_apply` and `rows_are_the_replys` take a recurrent model's flowing
   tree; the trailer, the bit, the drafter's IPC. Wire → `security-reviewer`.
4. **The coordinator**: the gates above; the drafter chosen (`engine_drafter::release_line` — Qwen3.5-0.8B for every
   `qwen35` target of the same vocabulary).
5. **Measure, one binary, A/B by `SWARMLLM_RECURRENT_FLOW=1` / unset**: first Qwen3.5-9B Q4_K_M split here ↔ AU (both
   cards hold their share whole), the 0.8B drafting on this card, against the n-gram loop with the file's guess
   head. **Kill**: under 1.5× the one-check loop's tok/s on that link, or replies parting from the one-check loop's
   anywhere but at near ties. Then Qwen3.8-27B all-card (here + AU + US, `run_allcard_rig.sh`'s shares).

## Stage 1 as built (2026-11-22)

`layers::qwen35::FlowRows` (on `SsmState::flow`) and `DeltaNetWeights::forward_flowing`, entered where the forward's
scope carries the walk's row (`TreeScope::flow_walk`, `RowPositions::for_tree_walked`, `layers::flow_walk_now`);
parents across forwards read off the visibility (`RowVisibility::parents_across_forwards`). What the build settled:

- **The extended forward**: the kept rows on the new rows' lines (`FlowLines::earlier`, ascending — a parent is always
  a smaller row) then the new rows, read as ONE tree from the anchor by the step's own `causal_conv_along` and
  `delta_rule_into`. Earlier rows are never projected again: their q/k/v/g/β are kept; only the convolution re-reads
  their `mixed` (their outputs are not used).
- **The walk's row may be a row of the forward carrying it** (a missed draw's row, pushed and walked onto at once):
  the anchor moves as far as kept rows reach BEFORE the forward (fewer rows read again) and again AFTER it.
- **A row whose line leaves the anchor's subtree** — sent before the walk moved past its branch — is read from the
  anchor (finite, never walked) and kept by nothing.
- **A rebase** is `truncate` to the tree's start (`SsmState::rewind_to` → `FlowRows::state_at`), then one forward
  holding its whole tree (the path as a chain, then the kept rows) whose walk is the path's end; the next forward's
  `from` is past it, and the layer renumbers from there (`FlowRows::rebase_onto`) — valid only where the walk's row
  IS that start's first row and its line the chain before it. The state there is the anchor's `before_anchor` (the
  state after the walk row's parent), so no row's inputs are kept for it.
- **Two starts are kept** (`FlowRows::starts`): a failed flow cuts to `dsd_flow`'s `settled_base` — the current
  start, or with a rebase unanswered the one before it.
- **Test** `a_flowing_trees_rows_are_their_lines_read_alone`: 16 rows over four forwards (both branches below the
  walk read again, the walk onto a row of its own forward, a walk back up refused), a rebase, a forward of the
  rebased tree; every reachable row and each anchor 0.0 from its line read alone; both cuts land on their starts.
  Red (checked): earlier rows replayed as one chain (row 6), parents in earlier forwards ignored (row 5), the anchor's
  path replay cut to its last row (row 9).

## Stage 2 as built (2026-11-22)

`forward_flowing` on a card runs what a tree check runs: the convolution and SiLU over the extended forward in one
launch along its lines (`conv_on_the_card` → `ssm_conv_silu_tree_f32_k4`; it walks only `kernel − 1` ancestors, so a
deep line costs it nothing), the L2 norm in one (`l2_norm_on_the_card`), the delta rule in one. What the build
settled:

- **Not the state stack this plan named — the host names each row's walk.** The tree kernel
  (`delta_rule_tree_f32_d*`) already restarts a row whose parent is not the row before it from the anchor and walks
  its line, but collects the line in `int line[32]` and silently drops what is deeper. A flowing forward's lines run
  as deep as the forwards in flight. `delta_rule_flow_f32_d*` is the same body reading `walks` instead: for each such
  row, its parent's line root first, built on the host once a forward (`qwen35::flow_walks`, `TreeLines::for_flow`;
  `[rows + 1]` offsets, then the rows), checked for length before the launch (`WalksView::of`). Every row the walk
  names is below the row it walks for, as `flow_walks` builds it from the same parents.
- **Cost, DERIVED**: a branching row re-reads its whole line, so a forward's updates are the sum of its branching
  rows' depths — on the test's deep forward ~70 against 38 minimal. A depth-first stack would read each row once;
  worth building only if stage 5 times the delta rule as a share of the segment's forward that matters.
- **Test** `a_flowing_tree_on_the_card_is_the_processors` (the stage 1 script, hoisted into `flowing_script`, now
  ending with a forward whose last row's line runs 35 rows below the anchor): every reachable row, each anchor and
  each start a cut puts back within 1e-4 × the largest value of the processor's (`head_width_weights`: 4 value
  heads of 128, the 27B's head width). Red (checked, RTX 3070): the walks switched off, row 41 — the 35-deep line —
  0.11 off of 1.38.

## Stage 3 (a) as built (2026-11-22): the worker and the wire

- **The wire**: `LayerForward::flow_walk` (`Option<u16>`), trailer `0x15 | walk u16 LE` after 0x14, written only
  beside 0x11 (`draft_tree_from`) and a non-empty shape (`append_flow_walk_trailer`); a reader refuses it cut short
  or with no tree that flows (`read_flow_walk_trailer`), never reads it as absent. Bound into the encrypted frame's
  AAD after 0x14. Feature bit `RECURRENT_FLOW` (47, in `ALL`); the onward hop hands the walk on with the tree's start
  (`flow_walk_onward`) only to a next segment advertising it (`onward_refusal`), else refuses the forward.
- **The worker** checks the walk against the tree before anything runs (`model_worker::flow_walk_to_apply`): a
  recurrent segment's flowing tree with no walk, a walk with no tree that flows, a walk past the tree's rows — each a
  400. A segment with no recurrent layer never reads it. **A flowing tree that happens to be a chain still goes
  through `with_draft_tree` with its walk**: the worker's chain shortcut ran it as a plain forward, which would have
  left the recurrent layers' flow unread and refused the next forward.
- **`SplitModel::with_draft_tree(tree, walk, f)`** — the one way a tree reaches a forward now takes the walk; the
  executor's row scopes are built by `RowPositions::for_tree_walked`. The drafter's own growth passes `None` (3 (b)).
- **A failed flowing forward is put back where it began** (`SsmState::flow_begun`): `executor::put_back_where_the_
  step_began` asks every layer to rewind to the step's start, and a layer could rewind a flow only to a tree's start
  — so a forward refused part-way (and a card's rerun after a refusal, `worth_running_again`) dropped the
  conversation. Each layer now keeps the flow as a flowing forward found it until the next one begins; a layer that
  kept nothing is where it began. Counted once in the cache's occupancy (only tensors the live flow no longer shares).
- **Test, real file** `a_flowing_tree_through_a_recurrent_model_reads_each_rows_path` (Qwen3.5-0.8B Q8_0,
  `SWARMLLM_GUESS_TEST_GGUF`): the stage 1 script through the whole model — 24 rows over six forwards, the rebase,
  a refused forward with no walk (the conversation kept), the walked forward after it — every row BIT FOR BIT the
  same tree read whole in one forward (worst 0.0 on the processor). Not its path read as a chain: the processor's
  tiled attention puts a tree row's keys in other vector lanes than a chain read's where masked rows sit between
  them, and Q8 activations turned the last bit into 1% at layer 11 (gotcha #1160) — a property of every processor
  tree check, not the flow. Layer test: the put-back and a rerun landing where the first run did; red (checked)
  without the snapshot.
- **Security review 2026-11-22** (`security-reviewer`, the wire and the layer): the trailer, its seal and every
  bound on the walk hold — nothing exploitable. Three Low, each fixed with a test that fails without it: (1) a
  flowing tree is not held to the 32-row bound of a tree check, and the processor's delta rule kept a state a row
  on every pool thread for good (512 rows: 32 MB a thread) — now a state only until its last child reads it
  (`qwen35::RowSlots`, `a_tree_keeps_only_the_states_still_to_be_read`) and a buffer past 32 states given back;
  (2) the rows a flow keeps were in no head-room claim, which ran only when the attention cache crossed a buffer
  quantum — every flowing forward now claims what it may newly hold in the recurrent layers
  (`SsmState::bytes_a_flowing_forward_may_add`: its kept rows gathered again with the new ones and the two moved
  states; ~80 KB a row a layer on a 27B-class model, checked against what the flow keeps on both devices);
  (3) `flow_from` rebased before its checks and a refused forward left the layer rebased — now put back as it was.
- **On a card** (RTX 3070, `SWARMLLM_GUESS_TEST_ON_CARD=1`) a forward's kernels differ with its WIDTH, so the whole
  read is no bit-for-bit reference there: the first five rows read as a plain tree of five sit 0.38 (of 20-28) from
  the same rows read in the tree of sixteen, flow or no flow. The test measures that figure and holds every flowing
  row to twice it and to the whole read's decision (each row's likeliest token the same, or within
  `CARD_NEAR_TIE_LOGITS`); the flow's first forward to the plain tree of its width at 1e-3 of the scale; and a
  chain read of a tree row must land past the bound. Measured: all 24 rows within the width's own 0.38, every
  decision alike; the rebase's forward (seven rows, the width of its whole read) 0.0 on every row.

## Stage 3 (b) as built (2026-11-22): the drafter's own flow

`SplitModel::read_flow_rows` — every read of a `FlowGrowth`'s rows — carries the walk (`FlowTree::at`), so a Qwen 3.5
drafter reads rows whose parents an earlier call read from its checkpoint, as the target's segments do. Nothing
else changed: the drafter's rebase already cut its cache to the tree's start (which ends the layer's flow,
`rewind_to` → `state_at`) and read the path as a prompt, and the kept rows start a new flow there. Other families
ignore the walk; a recurrent drafter's flowing reads are claimed as the segments' are. Test, real file
`a_recurrent_drafter_grows_a_tree_that_flows` (Qwen3.5-0.8B, processor): four calls — rows following an earlier
call's, the walk onto a draw no row carried, a rebase — and each of the 30 proposals is among its parent's 8 likeliest
followers in the same tree read whole; red (checked) without the walk: the second call's forward is refused.

## Stage 4 as built (2026-11-22): the coordinator

- **The entry** (`distributed.rs`): the drafter's path (`try_dsd_distributed`) runs for a recurrent target too where
  `SWARMLLM_RECURRENT_FLOW=1` — OFF by default on the computer driving a reply until stage 5 measures it (every node
  reads the walk as a segment regardless, so peers need only the release); the single-segment path stays closed. Before any token it asks
  `dsd_flow::recurrent_flow_may_start` — the flow switch, `tree_check_possible`, a stream shape whose every peer
  advertises `RECURRENT_FLOW` beside `DRAFT_TREE_FLOW | DRAFT_TREES` (`flow_possible`'s new `with_the_walk`, from
  `dsd_flow::flows_with_the_walk`, the one answer), and the tree line (`trees_wanted`, the drafter taken on a card).
  Refused → `Ok(None)`, and the n-gram loop runs the reply as before. A llama.cpp drafter is refused the same way.
- **The drafter** (`engine_drafter`): `DrafterUse` is REQUIRED on `engine_drafter_for`. `OnlyInAFlow` (passed only
  for a recurrent target) admits a recurrent drafter, or one whose family reads trees; `Anywhere` (the requester's
  chains, the planner's room, the download pass) keeps refusing recurrent drafters. The chosen one must guess on a
  card where it would load (`guesses_on_a_card`, `admitted`'s place), else `Ok(None)` before the prompt pass. Nothing
  is fetched or noted for a recurrent target (`note_split_wanting_a_drafter` is skipped): the download pass and the
  planner's room move to a recurrent drafter only once stage 5 measures the flow pays. (Moved since: § Stage 5 as
  measured, `DrafterUse::for_target`.)
- **The walk** (`FlowRun::send`, `read_here`, `take_together`): every flowing forward — this node's segments before
  the run, the far one, this node's after it — carries `flow_walk` = `walk_on_the_forward(rows.lead, flow.at())`: the
  walk's row in that forward's tree, which for a rebase's forward is the path's length (the rebased tree's row 0 after
  the path, test `a_rebases_forward_names_the_walk_after_the_path`). A read joining several answers names the last
  one's walk: the walk only moves down the tree, so the joined rows hold every row it stood on.
- **After the prompt pass** (`dsd.rs`): a recurrent target's tree flows wherever `flow_possible` still holds — its line
  was asked before the pass, and outside a flow it guesses nothing. Never tree rounds, never the chain stream. Where it
  does not flow, or the flow fails, the rest goes a token a round with drafting off: the failure's take-over
  (`rounds_take_over`) cuts every segment to the flow's settled start — a start each recurrent layer keeps
  (`FlowRows::state_at`) — and reads the reply past it again as a chain-shaped tree, which the worker runs as an
  ordinary forward (`handle_forward`). The n-gram loop would guess there, but cannot take a reply over part-way.
- **Tests** (each red-checked by breaking its rule): `a_rebases_forward_names_the_walk_after_the_path`,
  `a_recurrent_targets_tree_flows_only_through_peers_that_read_the_walk` (every peer of the run, the middle hop too),
  `a_recurrent_drafter_is_chosen_only_to_grow_a_flow`. The path end to end — the walk accepted by every segment, the
  take-over on a recurrent target (`SWARMLLM_FAULT_FLOW_FAIL`) — is stage 5's first run, on a real link.
- **First run, on this machine (2026-11-22)**: `split_rig.sh repeat`, Qwen3.5-4B Q4_K_M split over two nodes on this
  card through a private 300 ms link (A: part 0 = layer 0 only, B: the rest and the head), Qwen3.5-0.8B drafting on A,
  128-token greedy replies, one binary, arms `SWARMLLM_RECURRENT_FLOW=1` / `=0` / `=1` + `SWARMLLM_FAULT_FLOW_FAIL=6`.
  It found a fault older than the flow: a segment with no attention layer measured its caches by attention layers
  alone and read 0, so the first flowing forward was refused twice over (the worker's order check, then the tree's
  position check) — fixed in all three readings (`kv_cache::positions_without_attention`, gotcha #1161). After it:
  both flowing replies complete (40-45 forwards, one rebase, one miss each), every fault-arm reply is finished by the
  take-over, and each reply is one the one-check loop itself writes (that loop's own repeats differ at near-ties on
  the card, so byte-equality is no bar; the real-link run scores against llama.cpp). Not a speed figure — a repeated
  prompt, the owner's earlier replies copied — but replies 2-3 took 9.5 / 6.3 s flowing against 25.0 / 13.1 s.

## Stage 5 as measured (2026-11-22)

- **Rig** (`~/swarmllm-gate/deep/fr/real_link9b.sh`, out `real_link_rf9b_1122/`): Qwen3.5-9B Q4_K_M, parts 0-4
  here on this card (8 GB) and parts 5-9 on Australia's RTX 3060 (v0.3.306, ~115-134 ms round trip), Qwen3.5-0.8B
  Q8_0 drafting on this card, 160-token greedy replies (a prose and a code prompt) after a 64-token warm-up, ONE
  binary (v0.3.306), arms `SWARMLLM_RECURRENT_FLOW=1` (flowcon) / unset (flowcoff, the n-gram loop with the file's
  guess head) in the order on, off, off, on, on, off; the live node stopped.
- **Measured** (decode tok/s): off — prose 6.50, 5.70, 5.87; code 5.53, 6.23, 6.36. On, arm 1 — prose **11.18**
  (1.86× the off median), code **20.59** (3.4×); each reply opens as the one-check loop's does. Counters, arm 1:
  prose 194 forwards, 9 rebases, 29 misses; code 109 forwards, 4 rebases, 5 misses — prose loses its rows to misses,
  where a higher E is the only gain (`faster_than_local.md` § 10).
- **Arms 4 and 5 failed**, both at the same token (prose 16, code 22): Australia refused a flowing forward — "190 MB
  of KV cache already in use, budget 301 MB" — its card nearly full of other work. A flowing forward claims room for
  the tree's rows beside the reply's cache (`SsmState::bytes_a_flowing_forward_may_add`: two states and the kept
  rows' inputs, each recurrent layer), which a plain round never claims; the refusal arrived as
  `ServiceUnavailable`, outside `rounds_may_finish`, so the whole reply failed — three times, as the router continued
  it each time into another flow. **Fixed**: a memory refusal is finished in rounds (`rounds_may_finish`; test
  `a_peers_memory_refusal_is_finished_in_rounds_and_silence_is_not`).
- **The bar** (§ Stages, 5): 1.5× the one-check loop — met by arm 1 on both kinds of reply. The flow is ON by
  default on the computer driving a reply (`dsd_flow::recurrent_flow_on`; `SWARMLLM_RECURRENT_FLOW=0` is the A/B),
  and the planner's room, the download pass and the record that a split wanted a drafter now ask
  `engine_drafter::DrafterUse::for_target` — a recurrent target's drafter grows its flow, so a node fetches the
  0.8B and a plan leaves it room (beside a held block drafter, room for whichever needs more).
- **Re-run on main after the fix** (`real_link_rf9b2_1122/`, same rig and prompts, the flow ON by default, off arms
  `SWARMLLM_RECURRENT_FLOW=0`, order on, off, off, on, on, off): on — prose **12.42, 12.46, 12.30**, code **21.15,
  19.81, 19.60**; off — prose 6.43, 6.88, 6.53, code 6.29, 6.58, 6.87 (prose ~1.9×, code ~3.0×). Every arm wrote
  its 160 tokens. Australia had 9.1 GB free this time and refused nothing, so the take-over was NOT exercised on the
  link — its test is the evidence (red without the fix).
- **Where a forward's time goes** (arm 7, `SWARMLLM_COUNT_KERNELS=1`): the drafter's call was 46 ms (3 levels), its
  level **870 launches and 59 host-to-card copies**; a flowing forward of the 9B's first 17 layers here, 612
  launches and 48 copies. **The bookkeeping of kept rows, not the arithmetic, was most of both**: `copy2d_f32` 369
  and the gathers 98 of a drafter level (54%), 297 and 93 of the target's (64%) — every recurrent layer gathered its
  earlier rows, joined them to the new ones, and joined the new ones to the held ones apart — and 55 of the 59 copies
  were three tables every recurrent layer uploaded again (parents, walks, gather rows; each pageable copy waits for
  the stream to drain). Now one join and one gather a kind (`FlowRows::put` since the store, `rows_the_lines_read`), the
  tables uploaded once while their content stands (`qwen35::rows_on`), and the walk's replay in two launches
  (`FlowRows::advance_to` hands `cuda::replay` buffers of its own). A/B `SWARMLLM_FLOW_STORE=per-layer`
  (+ `SWARMLLM_SSM_REPLAY=composed`).
  **Measured** (`real_link_rf9b3_1122/`, same rig, ABBA, old arms `SWARMLLM_FLOW_STORE=per-layer
  SWARMLLM_SSM_REPLAY=composed`): the drafter's call **43.2-45.3 → 38.8-42.2 ms** a flowing forward, the head's 17
  layers unchanged (22.0-25.1 → 22.1-23.8); prose 12.19, 12.84 against 10.94, 11.27, 12.03, code 22.50, 20.80, 19.24
  against 24.99, 20.03, 19.62 — within the runs' spread. The count arm: a drafter level **870 → 705 launches**, host
  copies 59 → 7; the head 612 → 466. Of a level's 705, `copy2d_f32` 202 and gathers 100 remain — the join of the held
  rows to the new ones (two copies a kind) and the lines' gather (one a kind), every recurrent layer. Next: an
  append-only store a kind (one copy of the new rows, never the held ones) and the two kernels reading rows through
  a table (no gather).
- **The store** (2026-11-23, `real_link_rf9b5_1123/`, same rig, ten arms in the order store, joined, 2B, 2B, joined,
  store, store, 2B, joined, store + count): each kind's rows in a buffer made for twice the rows kept (half as many
  again since the memory refusals below), a forward's
  new rows copied once after the rows written (`FlowRows::put`), a row the walk left kept in place until the buffer
  is full (`FlowRows::make_room`); the joined arm is `SWARMLLM_FLOW_STORE=joined`. Drafting a flowing forward with
  Qwen3.5-0.8B Q8_0: store **36.9-38.5 ms** (prose 37.7-38.5, code 36.9-37.7) against joined 38.4-40.5; tok/s within
  the runs' spread — store prose 12.89, 14.00, 14.14, code 22.73, 23.36, 27.26; joined prose 12.77, 13.05, 13.47,
  code 22.46, 25.52, 26.68. The count arm: a level **705 → 644 launches** (`copy2d_f32` 202 → 137), the head's 17
  layers 466 → 440. **Qwen3.5-2B Q8_0 drafting** (store; fetched to this node for the arm): 46.9-48.4 ms a prose
  forward, 42.6-46.4 code, but a 160-token prose reply took 150-162 forwards against the 0.8B's 190-207 — prose
  13.99, 14.28, 14.74, code 22.51, 28.89, 29.47.
- **Captured levels** (2026-11-23, `real_link_rf9b6/7/8/9_1123/`): a flowing level staged before the capture — the
  walk's replay, its row tables, a scratch pair, room in a full store (`layers::stage_flowing_states`) and its memory
  claim (`claim_room_for_forward`; inside, the claim's synchronize broke the capture, gotcha #1163) — and always
  gathering its lines' rows, so every level records one topology (with none read the update was refused at every
  level, 588 in an arm). On the card alone it holds: `flowing_levels_pace_captured_and_op_by_op`, 40 drafter levels of
  Qwen3.5-0.8B, **9.6-9.8 ms a level captured against 13.1-13.2** op by op, 1 graph made and 38 updated in place,
  logits bit-equal (`a_captured_flowing_level_reads_as_the_ordinary_one`). On the TH↔AU link it does NOT yet: with
  the drafter's recording finished (below), most captures were refused inside (`q35_deltanet:
  CUDA_ERROR_INVALID_VALUE`, 25 in an arm), and the captured arms met memory refusals the uncaptured ones did not —
  prose 4.81 and 8.36, code 7.71 and 7.74 tok/s, against the store uncaptured 14.04, 12.56 / 22.04, 25.94 and joined
  13.13, 13.37 / 26.24, 19.65 (`real_link_rf9b9_1123/`, arms captured, joined, store, store, joined, captured). So
  it was turned OFF until the refusal was understood. **The
  refusal, found the same night** (gotcha #1165): staging replaced flow buffers and left the pre-stage copy their
  last holder; the forward's own `flow_from` inside the capture overwrote it, freeing memory made outside the
  capture there — refused by the driver and LEAKED (the drafter's pool 0.8 → 2.5-2.9 GB, hence the memory refusals
  too); the error surfaced only on 1-6-row levels, a capture-status check swallowing it at 7-8. A staged begin now
  keeps its copy through the capture (`SsmState::flow_from`); the probe at 2, 4 and 8 rows, 0 refused: 6.1, 7.2,
  8.3 ms a level captured against 11.0, 11.5-11.8, 14.2 (red at 4 rows without it). On the
  link, captured against the store uncaptured (`real_link_rf9b10_1123/`, arms G S S G G S): prose 13.95, 14.90,
  14.72 and code 24.18, 24.27, 31.65 tok/s against 12.55, 13.69, 14.07 and 22.90, 22.89, 22.67; the drafter's call
  25.7-28.4 ms a forward against 37.3-40.2 (pace 47.4-52.6 against 59.3-62.7); no refusal of any kind in six arms,
  the drafter's pool flat in both (2.7 → 2.9-3.0 GB, a peak 52 MB higher captured). **ON by default since**
  (`SWARMLLM_FLOW_GRAPH=0` turns it off). An
  earlier sitting where most levels were captured drafted at 27.2-28.3 ms a forward against 37.3-39.0
  (`real_link_rf9b7_1123/`). Found on the way, and kept: (1) **the drafter's idle recording was never marked done**
  when a message came just after its last graph — then nothing was left to record, so every tree shape stayed "left
  to the recording" for the worker's life, and whether a sitting captured anything was message timing
  (`record_check_widths` now asks whether a shape is done before whether to stop; gotcha #1164); (2) **a full store's
  claim is its growth**, not a whole new store on top of the old one already counted (`rows_a_forward_may_allocate`;
  ⚠ the growing forward's rollback copy holds the old store until the next forward, so that forward's peak is the
  claim plus the old store — FUTURE_WORK #228), and the store grows by half, not twice: claimed whole, a growth was refused on the drafter at 268 + 65 MB of a
  308 MB budget and on the 9B's head at 308 + 184 of 491, and the reply lost its drafter or its flow.
- **A flowing forward refused for memory keeps its conversation** (the same sitting's first arm: Australia's card had
  no room, `budget 0 MB`, the flow failed over to the rounds — and the rounds' first check was answered "no longer
  holds the conversation", because the refusal had cleared the request as a request-ending refusal does). The tree
  rounds finish the reply from the tree's start, which allocates nothing there; `executor`'s claim now clears only a
  refusal that ends the reply. Test `a_flowing_forward_refused_for_memory_keeps_the_conversation` (red without it:
  "a DeltaNet layer's state describes None positions").
- **The 2B drafter, captured** (2026-11-23 night, `real_link_rf9b23_1123/`, the released v0.3.308, arms 0.8B, 2B ×2
  levels, 2B ×1 level, 0.8B, 2B ×1, 2B ×2, 0.8B; parts 0-4 here, 5-9 and the head on the Australian card). With its
  levels captured, Qwen3.5-2B Q8_0 drafts **prose 16.89, 15.92 and code 29.69, 25.06 tok/s at two levels, prose
  18.25, 16.57 and code 23.03, 25.55 at one**, against Qwen3.5-0.8B's prose 13.35, 14.92, 14.19 and code 21.05, 22.67,
  28.84 — every 2B prose arm above every 0.8B one. Prose misses roughly halve (0.8B 28-31 a reply, 2B 14 at two
  levels, 10 at one); a 2B level costs ~14 ms on the card (forward_ms 5,023 over 362 levels) against the 0.8B's ~9,
  this computer's time a forward 48-49 ms at two levels and 30-32 at one against the 0.8B's 38-42. The choice
  (`engine_drafter::chosen`) already takes the 2B wherever it is held: 1.87 GiB is under the fetch bound, ×2 its size
  proxy (24 × 2048²) is within half the 9B's (32 × 4096²), and the download pass fetches the largest such the swarm
  knows. The rigs had been naming the 0.8B (`DRAFTER=`); the figures above are what a computer holding the 2B gets.
  One level against two moves prose up and code down within the runs' spread, so a call now may stop growing where
  its last level is unlikely to be reached — the depth gate (`split::FlowStep::depth_floor`, `SWARMLLM_FLOW_DEPTH`;
  Dynamic Depth Decoding's summed path probability). **Measured and left off** (`real_link_rf9b24_1123/`, the 2B in
  every arm, two runs each in one sitting, ABCDEEDCBA): up to three levels gated at 0.5 — prose 17.09 / 17.29, code
  26.45 / 28.54, a prose reply stopping after its first level in ~90% of calls — and at 0.3 (16.41 / 16.22, 23.33 /
  27.97), against two fixed levels' 17.14 / 16.48 and 30.76 / 27.04, one level's 17.77 / 16.92 and 22.11 / 21.99,
  and four rows a level at two levels' 16.59 / 16.53 and 24.80 / 24.47 (fewer rows cost misses and save little: a
  2B level is ~14 ms at 8 rows and ~12 at 2). This computer was the slower side in every arm (`here_ms_a_forward`
  46-52 ms at two levels, 28-34 at one, against the far end's 16-30), which is where one level costs code its depth;
  the gate is kept as the A/B for a link where the far end is the slower side (next_up NEXT 2).

## Not in this plan

The chain stream (`dsd_stream`) and a recurrent drafter's own γ-step chains: one-position steps run as captured
graphs, whose writes go to fixed addresses, so keeping their inputs across steps is a different piece of work
(#228 (5)). A flowing level IS captured (§ Stage 5, "Captured levels"); a tree check that does not flow is not
(`SplitModel::tree_steps_may_be_captured`).
