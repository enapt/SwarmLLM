# A tree that flows through recurrent layers (Qwen 3.5 / 3.6 / 3.8) — plan, 2026-11-22

Status: stages 1 and 2 BUILT (2026-11-22, the layer on the processor and on a card, with tests; nothing sets a
flowing scope outside the tests until stage 3). Stages 3-5 are PLAN. Figures are MEASURED (our records), READ (a
header or source) or DERIVED.

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

A flow grows ONE tree, a few rows a forward (`levels_a_call` 3 × `rows_a_level` 8), each forward's rows the tree's
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
- A failed flow finishes in tree rounds today (`rounds_may_finish`); for a recurrent model it finishes in the n-gram
  loop's rounds (one check at a time), which every segment already takes back.

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
5. **Measure, one binary, A/B by `SWARMLLM_RECURRENT_FLOW=0`**: first Qwen3.5-9B Q4_K_M split here ↔ AU (both
   cards hold their share whole), the 0.8B drafting on this card, against the n-gram loop with the file's guess
   head. **Kill**: under 1.5× the one-check loop's tok/s on that link, or replies parting from the one-check loop's
   anywhere but at near ties. Then Qwen3.8-27B all-card (here + AU + US, `run_allcard_rig.sh`'s shares).

## Stage 1 as built (2026-11-22)

`layers::qwen35::FlowRows` (on `SsmState::flow`) and `DeltaNetWeights::forward_flowing`, entered where the forward's
scope carries the walk's row (`TreeScope::flow_walk`, `RowPositions::for_flowing_tree`, `layers::flow_walk_now`);
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

## Not in this plan

The chain stream (`dsd_stream`) and a recurrent drafter's own γ-step chains: one-position steps run as captured
graphs, whose writes go to fixed addresses, so keeping their inputs across steps is a different piece of work
(#228 (5)). The flow's forwards are several rows, never captured on a recurrent model
(`SplitModel::tree_steps_may_be_captured`).
