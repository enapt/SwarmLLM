# SharedState, live config and credits

The evidence behind the rules in `.claude/rules/architecture.md`: what each
rule replaced, what it was measured at, and what a change must keep.

Every entry here was paid for. **Read the entry before changing the code it
names** — the rule statement in `architecture.md` is the summary, this is the
reasoning, and several of these describe a fix that looked obviously correct
and was not.

## `state.models.shards_needing_repair`

`state.models.shards_needing_repair` — 2026-08-25 (gotchas #381/#382). `DashSet<ShardId>`
of shards whose bytes were found WRONG and which need a fresh, verified copy. Written
ONLY by `SharedState::mark_shard_for_repair`; drained by
`AutoShardManager::complete_pending_shard_fetches`; cleared by
`clear_shard_repair` on a landed copy.
**Why it exists**: three places can catch a bad shard — the P2P accept path, the
background verification sweep, the auto-manage rescan — and all three removed the file
and stopped. "Removed" was implemented three times and "and get a good one" nowhere, so
repair happened only as a side effect of auto-manage noticing the gap: a node with
auto-manage OFF kept a permanently incomplete model, and every rescan re-hashed the same
bad file to reach the same conclusion. A new site that detects a bad shard calls the
helper; it must not open-code the removal.
**Deliberately NOT `shard_p2p_failed`**, which forces the HuggingFace path. Having
DETECTED the corruption means we hold the real hash, so a peer copy is checked against
it — and if that one is bad too the accept path quarantines it and docks the sender,
which is the behaviour wanted. Repair therefore runs from peers OR the origin.
**It runs outside the `auto_manage.enabled` gate** for the same reason
`try_idle_vram_unload` does: a shard this node already held is not a new acquisition
decision. And it refuses a shard in `removed_by_user` — a deletion is an instruction,
not a gap.

## `state.models.shards_pending_verification`

`state.models.shards_pending_verification` — 2026-08-25. `DashSet<ShardId>` of shards
THIS NODE HOLDS whose expected hash just changed. Written by the registry's
manifest-update hook (`set_persist_hook`, which now carries
`(manifest, persist, recheck_shards)`); drained by
`AutoShardManager::verify_pending_shards`, which re-hashes and, on mismatch,
drops the holder claim and calls `mark_shard_for_repair`.
**This is how a node learns from the SWARM that what it is serving is wrong.**
The only other re-check of an already-held shard is the one-shot startup sweep,
which runs ~2 s after boot against whatever the DB held — i.e. BEFORE a corrected
hash can arrive by gossip — and the rescan explicitly skips shards already
registered. So a node holding a corrupt shard could not discover the fact from its
peers at all; it took a further restart, after the persist hook had written the
corrected hash to the DB. Measured on the live swarm (gotcha #382).
Three things to keep: only shards we ACTUALLY HOLD are queued (re-hashing one we do
not have is hundreds of MB of I/O for no answer — pinned by
`a_held_shard_is_rechecked_when_its_expected_hash_changes`); a shard with a download
in flight is skipped, since re-hashing a partly-written file is a false alarm; and
the drain runs OUTSIDE the `auto_manage.enabled` gate, because serving neighbours
bad bytes is not a disk-management preference.

## `ModelRegistry::origin_verified`

(2026-08-25) — a shard hash derived from bytes
THIS node fetched from the model's origin, which outranks any gossiped claim. Applied
in `register_manifest` BEFORE change-detection, so a claim the origin has already
disproved does not even provoke a re-check. Persisted (`ORIGIN_VERIFIED_TREE`) and
loaded FIRST in `load_from_db`, or a restart hands the argument back to gossip.
Recorded by `SharedState::record_origin_downloaded_shard` from BOTH origin-download
paths — the auto-manage downloader and the admin "download this part" handler; the
second recorded nothing until it was added, which is the same one-invariant-N-paths
trap as everything else in this file.
**Why**: manifest registration is last-writer-wins, and a real hash replaces a real
hash (deliberately, for re-publishes). A peer that had self-certified a corrupt shard
gossiped its wrong hash, our node adopted it over one verified against the origin, and
v0.3.121's new re-check then faithfully **quarantined the GOOD copy** and refetched
against the same wrong reference — an unbounded ~500 MB loop, observed live within an
hour of shipping (gotcha #384).
**Deliberately local and NEVER gossiped**: provenance that travels the network is just
another assertion, and forgeable. A node trusts only what IT fetched.
**The general rule this encodes**: a repair mechanism is a destruction mechanism
pointed at whatever it believes is wrong. Before adding one, ask what happens when the
REFERENCE is the thing that is wrong.

## `network::manager::tensors::AckRttEstimator`

(2026-08-25) — how long a peer gets to
acknowledge a tensor forward before the pipeline gives up on it. **RFC 6298**, the same
algorithm TCP uses to decide a packet is lost: `deadline = SRTT + 4*RTTVAR`, alpha=1/8,
beta=1/4, clamped to `RR_ACK_TIMEOUT_SECS`..`RR_ACK_TIMEOUT_MAX_SECS`, over the
observed send→ACK latency of THAT peer.
**The variance term is the point.** The rule it replaces scaled a ping RTT, and a ping
cannot see queueing delay on a loaded node — the ACK is emitted by the network event
loop, so it is late exactly when that loop is busy. Measured: a peer at ~500 ms RTT (so
the 10 s floor) failed EVERY distributed request for about an hour, then served the
same request in 6.1 s once it settled. Its ACKs were arriving late, not missing —
proved by running a node at `-v` and seeing `kind="ack"` from six peers including that
one (gotcha #386).
**`observe_timeout` (RFC 6298 §5.5) is not optional.** The estimator only ever sees
acknowledgements that ARRIVE, and an abandoned forward is one whose ACK we stopped
waiting for — so a peer needing longer than the current deadline can never produce the
sample that would widen it. Without backing off on a miss the whole scheme is INERT
exactly where it is needed. Bounded by the same ceiling; a peer that recovers returns
to the floor.
**The fast-fail also requires a standby** (`request_has_standby`). It is justified by
the failover it enables — the premise of hedged requests in Dean & Barroso's *The Tail
at Scale*, which send a second copy to a DIFFERENT replica. With none, abandoning
cannot buy a failover and can only turn a slow success into a 503: measured, a reply
arrived 1.6 s after we gave up and was discarded. Absence of the pipeline entry counts
as "yes", so paths this check cannot see (a chain hop) keep the old behaviour.
**The standby gate is for a peer that is CONNECTED but silent.** A peer whose
connection closed AND whose re-dial failed is a different fact: no result can
arrive over a connection that does not exist, and the serving side abandons
inbound forwards on the close, so there is no slow success left to protect.
That case bypasses the gate deliberately — `schedule_redial_retry` →
`fail_forwards_awaiting_departed_peer` → `SharedState::
fail_layer_results_awaiting`, which fails only waiters PINNED to the departed
node via `resolve_pending_layer_result` (2026-09-02, gotcha #436). A new
fast-fail must decide which of the two facts it has: silence gets the gate,
proven departure does not.
**The deadline includes the payload** (2026-09-03, gotcha #446):
`tensors::ack_deadline_with_payload(base, activation_bytes)` adds
`bytes / ACK_ASSUMED_TRANSFER_BYTES_PER_SEC` (1 MiB/s, rounded down so a
decode step adds nothing), capped at the protocol timeout. An
acknowledgement is sent when the WHOLE message has arrived; the RTT-only
deadline gave a 20 MB prompt pass to a peer 625 ms away the same 10 s floor
as a 2 KB decode step and failed it in flight. **And the estimator only sees
small forwards** (`ACK_OBSERVE_MAX_BYTES`, 256 KiB): a transfer-dominated
sample measures the payload, not the peer, and routing reads the estimator.

## A KV-budget refusal MUST release what the request already took

(2026-08-25) —
`executor.rs` calls `kv_cache_store.clear_request` before returning the
`ServiceUnavailable`. A chunked prefill claims a quantum per boundary, so a request
refused at chunk N has already allocated N-1; leaving them made a refusal a RATCHET —
the request failed, its cache survived the 10-minute session TTL, and the next attempt
started from a higher floor. Measured in exact 1152 MB steps: 1152 → 2304 → 3456
against a 1166 MB budget, card at 97%, decode 29 → 1.0 tok/s (gotcha #387).
Safe because the request is over: an error is being returned, no later forward reuses
the cache, and a retry rebuilds it from the prompt.
**Generalise it**: a resource check that refuses but does not release is worse than no
check under repetition, because the failed attempts are exactly the ones that
accumulate. Ask of any admission check what happens to what the refused request took.

## `SharedState::can_fetch_shard_from_origin`

2026-08-25 — the single answer to
"will an origin fetch actually HAPPEN?", which is NOT "does an origin exist". Every
caller about to discard local bytes in favour of an origin copy asks this first:
**never throw away data you cannot replace.** Two conditions, found one at a time and
each after concluding the other was the only one — a recorded `hf_source`, AND not
offline mode (`trigger_download` skips the HuggingFace branch entirely when set, by
design). Auto-manage being off is deliberately NOT one, since the drain runs outside
that gate. Consumed by the P2P accept path (`classify_p2p_shard_acceptance`) and by the
rescan's no-hash branch. A third condition belongs here, not at a call site.

## `state.observed_inbound_connection`

`state.observed_inbound_connection` — `AtomicBool` on the ROOT SharedState,
**persisted, and written only via `SharedState::record_inbound_connection_observed`**.
Set by `handle_connection_established` for a non-loopback connection where we
are the LISTENER. **The only direct evidence that inbound reaches this node**:
outbound succeeds from behind almost anything, so a node with peers, a real
LAN address and clean logs can still be silently dropping every inbound packet
and look perfectly healthy from the inside. Read by the WSL2 firewall check in
`health::monitor`. Any future "are we reachable" question should use this
rather than inferring from peer count — having peers proves only that WE
dialled successfully.
**It is seeded from the database at startup, and the persistence is the load-
bearing part.** The fact being recorded is a property of the machine's
network, and restarting the daemon does not reconfigure a firewall — so an
in-memory-only observation made the check re-decide that question every start
from whatever happened in the next few minutes. A reachable node routinely
sees nothing in that window: it dials every peer it already knows in the first
seconds of starting (10 connections inside 2 seconds, measured), so it is the
dialer on every link and may never be dialled back at all. The result
was a node telling its owner to run Administrator PowerShell firewall commands
it did not need. Measured 2026-08-18 on this development machine: inbound TCP
open and verified by hand from a peer on the same subnet, 181 inbound
connections across the log's history, **zero in a 9-hour run**, and a run that
warned at 06:47 contradicted by its own inbound connection at 07:41. Three of
the four most recent runs warned; all three were wrong (gotcha #335).
**There is no length of silence that proves unreachability**, so the grace
period is a noise control, not the fix, and the message reports what was
observed rather than naming a cause. A new check that wants to conclude
something about this machine's network must persist its evidence the same way;
an observation whose lifetime is shorter than the fact's cannot support the
claim.

## `SharedState::model_is_in_use` is the answer to "may I delete this model's files?"

— NOT `active_pipelines` on its own. That is the COORDINATOR's map of
DISTRIBUTED assignments (gotcha #194) and holds nothing for peer-served work or
for a reply the local model is producing through the split fast path, which
bypasses the router entirely. Deleting during a local reply therefore returned
`200 files_removed: 8` and killed the worker mid-stream (measured 2026-08-05) —
the exact outcome the guard below exists to prevent, on the most common
single-node path. The helper asks `active_traces` first, because every
in-flight request registers one for progress reporting whichever path serves
it; `serving_models` and `active_pipelines` remain as belt-and-braces. Both
`delete_model` and `delete_shard` go through it.

## Config defaults must stay live

The daemon must write **only values that differ from the compiled default**.
`config::to_minimal_toml` is the one serializer for the config file; do not call
`toml::to_string_pretty(&config)` directly.

**Why.** A `#[serde(default)]` fills a key that is *missing*. Once a key is
written to disk it wins forever, so any later change to that default can never
reach that install — and the file looks like a deliberate user choice, which is
indistinguishable from one. `PUT /api/admin/config` is called by the setup
wizard on "Start SwarmLLM", so in practice every field landed on disk on first
run. This produced three separate user-visible faults before it was fixed:

- `bootstrap_peers = []` stranded every node set up before 2026-07-21 with no
  bootstrap peer, no DHT route and no log line (gotcha #198).
- A default-on dashboard-trust flag shipped *off* to exactly the fresh installs
  it was written for (gotcha #196).
- `check_interval_hours = 6` kept nodes on a six-hour update check after the
  default became hourly — found live on 2026-07-29 while watching a node fail
  to notice a release.

Rules that follow:

1. **Never serialize the whole `Config` to disk.** Use `to_minimal_toml`.
2. **A section's `impl Default` MUST agree with its fields' `#[serde(default)]`.**
   These are different code paths: a *missing* section uses `impl Default`, a
   *present but empty* section uses each field's serde default. They disagreed
   for `updates.mode` (`Some(Notify)` vs `None`), which made the effective
   update mode depend on whether the `[updates]` header happened to exist.
   Pinned by `empty_section_matches_missing_section`, which checks every
   section, so a new one inherits the coverage.
3. **Every field needs a serde default**, or a pruned file will not reload.
   Pinned by `empty_toml_parses_to_full_default`.
4. **Changing a default does not reach existing installs.** If the old value is
   already on disk it stays. When a default changes in a way that matters, add
   an entry to `migrate_superseded_defaults` — and only when the old value was
   the daemon's, never something a user could plausibly have chosen, because
   silently overriding a deliberate setting is worse than a stale default.
5. **Unknown keys warn, they do not fail.** `deny_unknown_fields` would refuse
   to start on a config mentioning a later release's key. `warn_unknown_keys_in`
   names the key and continues.

## `SharedState::release_request_state`

(2026-08-09) — clears the maps a
finished request leaves behind: `active_pipelines`, `active_traces`,
`request_holder_blacklist`, `peer_vram_commitments` and
`local_memory_refusals`. They are keyed by request id and share one
lifetime. Five call sites removed all three by hand and the invariant was held
by three adjacent lines plus a comment asserting it — the shape this codebase
keeps getting caught by. Dropping one is silent and unbounded: `active_traces`
is the oracle behind `model_is_in_use`, so a stranded entry refuses to delete
that model for the rest of the daemon's life, and a stale blacklist entry keeps
barring a peer that was only meant to be skipped once.
It deliberately does NOT touch `active_count` or `queue_notify` — those belong
to the dispatch path that owns the slot, and must move together (§ Inference
Router Queue). `per_request_state_is_released_in_one_place` fails the build on
a new direct removal; `TraceGuard` is allowlisted because it registers a trace
for the split fast path and owns nothing else.

## Credits are DORMANT — nothing may publish or act on a balance

(2026-08-17).
`MIN_BALANCE_FOR_INFERENCE = 0` and `credit::priority::calculate_tier` returns
`DORMANT_TIER` regardless of its arguments, so no balance affects who is
served or how fast; the leaderboard neither ranks by credits nor publishes
them. The figure is self-minted — no credit has ever moved between nodes as
payment for work — so acting on it meant rationing the product by a number
nobody can stand behind. `credits_stay_dormant` in `tests/repo_consistency.rs`
fails the build if that changes, and it scans the WHOLE of `api/identity.rs`
rather than the lines that were fixed: the leaderboard's *self* entry is built
by different code from its peer entries, so the first fix left the node still
publishing its own (gotcha #317). **It also scans `frontend/js` for code that
reads a credit figure** (2026-08-30) — the backend half was checked and the
dashboard half was not, so an unreachable `sortKey === 'credits'` branch
survived the cleanup that removed every element rendering the balance. It
sorted on a field the peer payload has not carried for releases, so it read
`undefined` and nobody noticed; one restored column header would have had the
peer list ranking by a self-minted number with nothing to catch it. Comments
are excluded, because one of them documents precisely why nothing renders the
figure. Design and exit criteria in `docs/CREDITS_DESIGN.md`.

## A settle that cannot be written down has not happened

(2026-09-10). `credit::escrow` has three paths that change an escrow's status
and then move a balance — `release_escrow`, `refund_escrow`, `cleanup_expired`
— and **all three must leave the entry `Pending` and the balance untouched when
the status write fails.** Two of them did. `release_escrow` logged a warning and
reconciled anyway, which mints credits.

The mechanism is the restart. `EscrowManager::new` re-inserts every `Pending`
entry it finds in the `escrow` tree, so a settle whose status never reached the
disk comes back claimable; `cleanup_expired` then refunds the **full**
reservation at TTL with no caller involved at all. Measured on the replayed
counterexample: 500 in, 40 of real compute consumed, **560 out**.

Reported against v0.3.170 by a contributor who modelled the file in TLA+ and ran
TLC against it (issue #21). It is a good illustration of what model checking
finds and review does not: **the hazard was already understood twenty lines
below, in the same file, with a comment explaining it** — "if DB write fails,
revert status to prevent double-refund on restart" — and each path reads as
careful in isolation. Nothing about `release_escrow` looks wrong until you ask
what the *other* two do with the same failure. This is the codebase's
one-invariant-N-paths defect again (`.claude/rules/architecture.md`), in the one
shape a reviewer cannot catch by reading the function in front of them.

The direction of the trade-off is fixed and is the one `create_escrow`'s own
`SEC:` comment already chose: **credits are lost-or-refunded, never
double-paid.** On a failed release the requester keeps the whole reservation and
the serving node is paid nothing — wrong, but wrong in the direction that cannot
inflate the supply. Do not "fix" that asymmetry by settling optimistically.

Pinned by three tests in `credit::escrow`, one per path, each verified by
planting the violation the test exists to catch (gotcha #413). They are only
possible because of `Database::set_write_failure`, below.

**The other 30 `Failed to persist` sites were swept and are not this class.**
The dangerous shape is narrow: a persisted record meaning *"this is still
owed"* left behind while memory proceeds as though it were settled, **and a
startup path that re-reads that record and acts on it**. Most of the rest are
settings that revert on restart, and their own warning says so. The three
credit-adjacent ones are all safe for reasons worth writing down rather than
re-deriving: `apply_credit_direct_noted` (the inference charge) reverts its own
in-memory mutation, so memory and disk agree that nothing happened;
`CreditLedger`'s periodic balance flush is a retry loop and the next tick
carries the same figure; and the pool credit-forward totals are running
statistics, so a failure under-counts a contribution rather than creating a
second claim on the same credits.

## `Database::set_write_failure` — a persist failure you can actually cause

(2026-09-10). Several subsystems reason in comments about what happens when a
redb write fails, and until this existed **none of those reverts had ever been
executed**. They were argued for, reviewed, and never run; one of the three in
`credit::escrow` turned out not to be there at all.

The switch is armed at `Database::with_write_table`, the single write-transaction
site (`begin_write` appears exactly once in `storage/db.rs`), so no mutating
method can quietly keep working while another fails.

**It is scoped to a TREE, and that is the whole point rather than a
convenience.** The first version failed every write, and under it the escrow
double-pay *did not reproduce* — because the balance write failed too, and
`apply_credit_direct_noted` reverts its in-memory mutation when its persist
fails, so the books came out even. A blanket failure is a different experiment
from a partial one: the interesting case is a subsystem writing its own state
and then moving a balance, with only the first write failing. **An instrument
that makes the defect disappear is not a null result** — check what the
injection actually covers before reading "no minting" as "no bug".

`the_injected_write_failure_fires_on_one_tree_through_a_clone` pins the
instrument itself: the armed tree fails, a second tree keeps working, the write
really does not land, and the switch reaches a *clone* of the handle — which is
how every subsystem holds its `Database`.

## `SharedState::cfg()`

(2026-08-09) — the live config, and the single answer
to "what is this setting **now**". `state.config` is the boot-time snapshot:
correct for what is decided once at startup (listen addresses, data dir, how
the swarm was built), wrong for anything the Settings panel can change.
**Reading a user-settable value from `state.config` is the recurring bug this
ends**: the setting saves, answers `{"status":"ok"}`, shows its new value, and
the running daemon carries on with the old one. Measured on the released
v0.3.87 — `max_disk_mb` 50000 → 123456 still reported 50000; contribution →
Maximum left the storage target at 6250 MB.
`PUT /api/admin/config` stores the whole updated config here, so a new setting
is live with no extra wiring. The `OperationalParams` watch channel remains,
but ONLY to wake subsystems that must *react* rather than re-read — resizing
the router's concurrency, retiming the auto-manage interval. A value that is
merely read each tick needs nothing but `cfg()`.
**Do not add another per-setting mirror.** Four already existed
(`contribution_auto` from R121, `dashboard_trust_lan`, the two cross-pool
toggles) because each was bolted on when someone noticed one setting doing
nothing, which left the next one broken; `OperationalParams` meanwhile carried
five fields nothing consumed while documenting itself as hot-reloadable.
`user_settable_config_is_read_live_not_from_the_boot_snapshot` in
`tests/repo_consistency.rs` fails the build on a new frozen read. It checks
whole SECTIONS, not field names, because the frozen value is just as often
reached through a method — `config.resources.shard_upload_mbps(..)` never
mentions `max_bandwidth_mbps`, and that is how that one survived a first pass.
**Some things genuinely cannot follow live and the UI must say so** rather than
implying otherwise: libp2p connection limits are fixed when the swarm is built,
and CPU thread counts are handed to a worker as it spawns (recycling a live
worker would drop whatever it is answering). `settings.contribution_restart_note`
is where that is said.

## `SharedState::record_peer_serve`

(2026-08-09) — the single answer to "this
node did inference work for a peer", counting it AND billing for it. Reached
from exactly two places, the only two inbound paths that serve someone else:
`dispatch/layer_forward.rs` (one segment of a pipeline) and
`dispatch/remote_generate.rs` (the whole decode, the fast path).
**Do not count or bill serving at a call site**, and do not write
`requests_served_atomic`, `forwards_served_atomic` or `pending_credit_earn`
anywhere else — `serving_is_counted_and_paid_in_exactly_one_place` in
`tests/repo_consistency.rs` fails the build if you do.
**Why it is enforced rather than documented**: the previous helper,
`track_forward_participation`, had a doc comment saying exactly this and was
still called by only one of the two paths — the *less* travelled one. The fast
path is how a machine holding a whole model answers a peer, so in practice
most serving recorded nothing and earned nothing while the requester was still
debited (gotcha #279).
**The converse is equally load-bearing**: work the node does for ITSELF must
not come through here. The router's completion hook and the local-segment path
both used to bump these counters, and `pipeline/distributed.rs` used to credit
the node for its own segment, so a user whose only traffic was their own chat
was told they had served the swarm and was paid for it. The product promises
"earn credits by serving inference for others" and "inference across your own
devices is free"; both directions have to hold for that to be true.
Note that `release_escrow` transfers nothing to `to_node` despite recording it,
and `credit::transaction::create_transaction` has no production callers — so
this accumulator is the ONLY way a serving node is ever paid (gotcha #280).

## `config::InferenceConfig::claims_shard`

(2026-08-09) — the single answer to
"does this node claim shard N?", i.e. how `inference.shard_range` is read.
**Never read `shard_range` directly.** Five places asked the question with
their own copy of the comparison and THREE never asked at all: the startup
disk scan, the periodic rescan, and one manifest path. The rescan is the one
that mattered — startup applied the range correctly and then, minutes later,
the rescan found the remaining files still on disk and re-registered them, so
a node configured for shards 0-1 of a four-shard model came up serving
`layers=[0..12)` and was serving `[0..28)` on its own five minutes later.
The feature then fails twice over: the node stops being half of a split model
AND loads the whole thing into memory, which is the saving being asked for.
Silent — no error, no warning, and the config key parses.
**A new shard-registration path MUST call this**; that is the whole reason it
is a method on the config that owns the field rather than a free function
someone can forget. Verified on two machines: the restriction held for 10
minutes against the 4m47s it previously took to lose it, and a genuine
two-segment pipeline then answered correctly across both.

## `SharedState::local_fast_path_for` is the single answer to "may this request take the local split fast path?"

(2026-09-03, gotcha #443). Both
API surfaces used to compose it themselves (`has_complete_split_model &&
!should_offer_work_to_the_swarm`), and the fast path skips the router —
which is where `delegation_target` lives. On a node whose card is too
small the model is not REGISTERED (`scan.rs` refuses over the graphics
budget), so the fast path was skipped by accident and delegation looked
reachable; on a node with no card the model is always registered, the fast
path always won, and the #442 fix shipped in v0.3.150 was unreachable on
the very node it was for. The predicate now stands aside when
`serves_on_cpu` AND a connected peer exists; the scheduler then delegates
or assigns locally. A feature behind a gate is only as reachable as the
gate's callers: test it from the API, not from the function.


## Prompt privacy is read through one accessor, and the map is not it

(2026-09-11.) `SharedState::encrypted_pipeline_for` resolves three cases in
order: an explicit per-model choice, an explicit global `encrypted_pipeline`,
and — since 2026-07-27 — `encrypted_pipeline_auto`, which is **ON by default**
and switches privacy on wherever this node holds both ends of a model. That
third case is how prompt privacy is normally in force at all, because it is the
only one that needs no user action.

`encrypted_pipeline_models` holds the FIRST case alone. Four sites re-derived
the setting from it, each implementing a different prefix of the precedence
rule, and every one of them under-reported privacy:

- **`auto_manage::prune`** — its skip read the map, so it protected models a
  user had toggled by hand and left unprotected exactly the models privacy was
  actually in force for. Pruning an END shard strands the setting: it stays on,
  nothing can satisfy it, and every request for that model then fails at
  pipeline assembly. A live node reached that state on 2026-08-09, and the
  investigation named `delete_shard` as the suspect — which was guarded, and
  which is why the entry stayed open with its own question unanswered. Prune had
  been reading the map since before auto-enable existed, so the guard was correct
  when written and was silently outgrown by the default changing underneath it.
- **`pipeline::distributed`** and **`pipeline::remote_generate`** — both did
  `map.get(..).unwrap_or(config.inference.encrypted_pipeline)`, i.e. cases 1 and
  2 without 3. The first decides whether to embed locally so a peer sees
  activations rather than token ids; the second decides whether a request may
  take the fast path that puts the RAW PROMPT on the wire to a peer. Both are
  defence in depth for the case they were mis-answering. No live leak is
  demonstrated — the scheduler reads the accessor and would not produce those
  shapes with privacy on — but that masking is a property of today's scheduler,
  not a guarantee, and two components disagreeing about one setting is the
  standing defect of this codebase.

What a change here must keep:

- `privacy_required_shards` is the shared rule for "which shards may not be
  removed", and BOTH `delete_shard` (refuses) and prune (skips) ask it.
- Prune keeps a second, broader hold for an EXPLICIT choice: every shard, not
  just the ends. Privacy needs only the ends, so this is not correctness —
  narrowing a protection a user deliberately asked for is a privacy-affecting
  change and does not belong in a fix for the automatic case.
  `privacy_holds_shard` states both holds in one pure function, tested directly.
- `prompt_privacy_is_never_re_derived_from_the_per_model_map` scans `src/` for
  READS of the map (writes are how tests set the explicit choice), allowing only
  `daemon/state` (owner) and `api/admin_models/lifecycle.rs` (the endpoints that
  deliberately expose the explicit value as distinct from the effective one).
  Its self-test plants the violation in the wrapped shape rustfmt produced,
  which is the form that went unnoticed; the guard was also verified by
  restoring the original prune code and watching it fail.
